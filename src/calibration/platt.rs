//! # Platt Scaling Calibration
//!
//! Platt scaling for complex non-linear calibration cases in PHASE 4.
//! Used for challenging calibration problems where isotonic and temperature scaling are insufficient.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

use super::CalibrationSample;

/// Platt scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlattConfig {
    /// Maximum optimization iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub convergence_tolerance: f64,
}

/// Platt scaling calibrator using sigmoid parameters A and B
#[derive(Debug, Clone)]
pub struct PlattScaler {
    config: PlattConfig,
    /// Parameter A for sigmoid: P(y=1|f) = 1 / (1 + exp(A*f + B))
    parameter_a: f64,
    /// Parameter B for sigmoid: P(y=1|f) = 1 / (1 + exp(A*f + B))
    parameter_b: f64,
    /// Current ECE for this slice
    slice_ece: f32,
    /// Training convergence status
    converged: bool,
    /// Number of training samples
    training_samples: usize,
    /// Final training likelihood
    final_likelihood: f64,
}

/// Optimization state for Platt scaling
#[derive(Debug, Clone)]
struct PlattOptimizationState {
    a: f64,
    b: f64,
    likelihood: f64,
    gradient_a: f64,
    gradient_b: f64,
    hessian_aa: f64,
    hessian_ab: f64,
    hessian_bb: f64,
}

impl PlattScaler {
    /// Create new Platt scaler
    pub fn new(config: PlattConfig) -> Self {
        Self {
            config,
            parameter_a: -1.0, // Initial guess (negative for typical case)
            parameter_b: 0.0,
            slice_ece: 0.0,
            converged: false,
            training_samples: 0,
            final_likelihood: f64::NEG_INFINITY,
        }
    }

    /// Train Platt scaling on provided samples
    pub async fn train(&mut self, samples: &[&CalibrationSample]) -> Result<()> {
        if samples.len() < 30 {
            anyhow::bail!("Insufficient samples for Platt scaling: {} < 30", samples.len());
        }

        info!(
            "Training Platt scaler on {} samples (max_iter={})",
            samples.len(),
            self.config.max_iterations
        );

        self.training_samples = samples.len();

        // Extract predictions and targets
        let predictions: Vec<f64> = samples.iter().map(|s| s.prediction as f64).collect();
        let targets: Vec<f64> = samples.iter().map(|s| s.ground_truth as f64).collect();

        // Optimize parameters using Newton's method
        let final_state = self.optimize_platt_parameters(&predictions, &targets).await?;
        
        self.parameter_a = final_state.a;
        self.parameter_b = final_state.b;
        self.final_likelihood = final_state.likelihood;

        // Calculate final ECE
        self.slice_ece = self.calculate_ece_with_platt(samples).await?;
        
        // Check convergence
        self.converged = self.check_convergence()?;

        info!(
            "Platt scaling trained: A={:.4}, B={:.4}, ECE={:.4}, likelihood={:.4}, converged={}",
            self.parameter_a, self.parameter_b, self.slice_ece, self.final_likelihood, self.converged
        );

        if self.slice_ece > 0.015 {
            warn!(
                "Platt scaling ECE {:.4} exceeds PHASE 4 target ≤ 0.015",
                self.slice_ece
            );
        }

        Ok(())
    }

    /// Apply Platt scaling to a prediction
    pub async fn calibrate(&self, prediction: f32, _features: &HashMap<String, f32>) -> Result<f32> {
        let pred_f64 = prediction as f64;
        
        // Apply Platt scaling: P(y=1|f) = 1 / (1 + exp(A*f + B))
        let exponent = self.parameter_a * pred_f64 + self.parameter_b;
        let calibrated = 1.0 / (1.0 + exponent.exp());
        
        debug!(
            "Platt scaling: {:.4} -> {:.4} (A={:.3}, B={:.3})",
            prediction, calibrated, self.parameter_a, self.parameter_b
        );

        Ok((calibrated as f32).clamp(0.001, 0.999))
    }

    /// Get learned parameters (A, B)
    pub fn get_parameters(&self) -> (f32, f32) {
        (self.parameter_a as f32, self.parameter_b as f32)
    }

    /// Get current ECE
    pub fn get_ece(&self) -> f32 {
        self.slice_ece
    }

    /// Check if training converged
    pub fn is_converged(&self) -> bool {
        self.converged
    }

    /// Get number of training samples
    pub fn get_training_samples(&self) -> usize {
        self.training_samples
    }

    /// Get final training likelihood
    pub fn get_final_likelihood(&self) -> f64 {
        self.final_likelihood
    }

    // Private implementation methods

    /// Optimize Platt parameters using Newton's method
    async fn optimize_platt_parameters(&self, predictions: &[f64], targets: &[f64]) -> Result<PlattOptimizationState> {
        let mut state = PlattOptimizationState {
            a: -1.0,
            b: 0.0,
            likelihood: f64::NEG_INFINITY,
            gradient_a: 0.0,
            gradient_b: 0.0,
            hessian_aa: 0.0,
            hessian_ab: 0.0,
            hessian_bb: 0.0,
        };

        // Newton's method optimization
        for iteration in 0..self.config.max_iterations {
            // Calculate likelihood, gradient, and Hessian
            self.calculate_likelihood_and_derivatives(predictions, targets, &mut state)?;

            // Store previous parameters for convergence check
            let prev_a = state.a;
            let prev_b = state.b;

            // Newton's method update: θ_new = θ_old - H^(-1) * g
            let det = state.hessian_aa * state.hessian_bb - state.hessian_ab * state.hessian_ab;
            
            if det.abs() < 1e-12 {
                warn!("Platt optimization: Hessian nearly singular at iteration {}", iteration);
                break;
            }

            // Invert 2x2 Hessian matrix
            let inv_hessian_aa = state.hessian_bb / det;
            let inv_hessian_ab = -state.hessian_ab / det;
            let inv_hessian_bb = state.hessian_aa / det;

            // Update parameters
            let delta_a = -(inv_hessian_aa * state.gradient_a + inv_hessian_ab * state.gradient_b);
            let delta_b = -(inv_hessian_ab * state.gradient_a + inv_hessian_bb * state.gradient_b);

            state.a += delta_a;
            state.b += delta_b;

            // Check convergence
            let param_change = ((state.a - prev_a).powi(2) + (state.b - prev_b).powi(2)).sqrt();
            
            if param_change < self.config.convergence_tolerance {
                info!("Platt optimization converged at iteration {} (change: {:.8})", iteration, param_change);
                break;
            }

            if iteration % 20 == 0 {
                debug!(
                    "Platt optimization iter {}: likelihood={:.6}, A={:.4}, B={:.4}, change={:.8}",
                    iteration, state.likelihood, state.a, state.b, param_change
                );
            }
        }

        Ok(state)
    }

    /// Calculate log-likelihood, gradient, and Hessian for Platt scaling
    fn calculate_likelihood_and_derivatives(
        &self,
        predictions: &[f64],
        targets: &[f64],
        state: &mut PlattOptimizationState,
    ) -> Result<()> {
        let n = predictions.len();
        
        // Reset accumulators
        state.likelihood = 0.0;
        state.gradient_a = 0.0;
        state.gradient_b = 0.0;
        state.hessian_aa = 0.0;
        state.hessian_ab = 0.0;
        state.hessian_bb = 0.0;

        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            // Calculate probability: P(y=1|f) = 1 / (1 + exp(A*f + B))
            let linear_term = state.a * pred + state.b;
            let exp_term = linear_term.exp();
            let prob = 1.0 / (1.0 + exp_term);
            
            // Avoid numerical issues
            let prob_clamped = prob.clamp(1e-15, 1.0 - 1e-15);
            
            // Log-likelihood: L = Σ [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
            let sample_likelihood = target * prob_clamped.ln() + (1.0 - target) * (1.0 - prob_clamped).ln();
            state.likelihood += sample_likelihood;

            // Gradient calculations
            let error = prob - target;
            let weighted_error_pred = error * pred;
            
            state.gradient_a += weighted_error_pred;
            state.gradient_b += error;

            // Hessian calculations (second derivatives)
            let hessian_factor = prob * (1.0 - prob); // p * (1 - p)
            
            state.hessian_aa += hessian_factor * pred * pred;
            state.hessian_ab += hessian_factor * pred;
            state.hessian_bb += hessian_factor;
        }

        // Normalize by sample count
        let n_f64 = n as f64;
        state.likelihood /= n_f64;
        state.gradient_a /= n_f64;
        state.gradient_b /= n_f64;
        state.hessian_aa /= n_f64;
        state.hessian_ab /= n_f64;
        state.hessian_bb /= n_f64;

        Ok(())
    }

    /// Calculate ECE with current Platt parameters
    async fn calculate_ece_with_platt(&self, samples: &[&CalibrationSample]) -> Result<f32> {
        const NUM_BINS: usize = 10;
        let mut bins = vec![Vec::new(); NUM_BINS];

        // Apply Platt scaling to get calibrated predictions
        for sample in samples {
            let pred_f64 = sample.prediction as f64;
            let exponent = self.parameter_a * pred_f64 + self.parameter_b;
            let calibrated_pred = (1.0 / (1.0 + exponent.exp())) as f32;
            
            let bin_idx = ((calibrated_pred * NUM_BINS as f32) as usize).min(NUM_BINS - 1);
            bins[bin_idx].push(sample.ground_truth);
        }

        // Calculate ECE
        let mut ece = 0.0;
        let total_samples = samples.len() as f32;

        for (i, bin) in bins.iter().enumerate() {
            if bin.is_empty() {
                continue;
            }

            let bin_center = (i as f32 + 0.5) / NUM_BINS as f32;
            let bin_accuracy = bin.iter().sum::<f32>() / bin.len() as f32;
            let bin_error = (bin_center - bin_accuracy).abs();
            let bin_weight = bin.len() as f32 / total_samples;

            ece += bin_error * bin_weight;
        }

        Ok(ece)
    }

    /// Check if training converged
    fn check_convergence(&self) -> Result<bool> {
        // Check if parameters are reasonable
        let params_ok = self.parameter_a.is_finite() && 
                       self.parameter_b.is_finite() &&
                       self.parameter_a.abs() < 100.0 && 
                       self.parameter_b.abs() < 100.0;
        
        // Check if ECE is within target
        let ece_ok = self.slice_ece <= 0.015;
        
        // Check if likelihood is reasonable
        let likelihood_ok = self.final_likelihood > f64::NEG_INFINITY && 
                           self.final_likelihood.is_finite();

        Ok(params_ok && ece_ok && likelihood_ok)
    }
}

impl Default for PlattConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_tolerance: 1e-6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_sample(prediction: f32, ground_truth: f32) -> CalibrationSample {
        CalibrationSample {
            prediction,
            ground_truth,
            intent: "test".to_string(),
            language: Some("rust".to_string()),
            features: HashMap::new(),
            weight: 1.0,
        }
    }

    #[tokio::test]
    async fn test_platt_scaler_creation() {
        let config = PlattConfig::default();
        let scaler = PlattScaler::new(config);
        
        let (a, b) = scaler.get_parameters();
        assert_eq!(a, -1.0);
        assert_eq!(b, 0.0);
        assert_eq!(scaler.get_ece(), 0.0);
        assert!(!scaler.is_converged());
    }

    #[tokio::test]
    async fn test_platt_training_insufficient_samples() {
        let config = PlattConfig::default();
        let mut scaler = PlattScaler::new(config);
        
        let samples = vec![
            create_test_sample(0.8, 1.0),
            create_test_sample(0.9, 1.0),
        ];
        let sample_refs: Vec<&CalibrationSample> = samples.iter().collect();
        
        let result = scaler.train(&sample_refs).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Insufficient samples"));
    }

    #[tokio::test]
    async fn test_platt_calibration() {
        let config = PlattConfig {
            max_iterations: 50,
            convergence_tolerance: 1e-5,
        };
        let mut scaler = PlattScaler::new(config);
        
        // Create non-linear calibration problem
        let mut samples = Vec::new();
        for i in 0..50 {
            let pred = (i as f32) / 50.0; // [0, 1]
            // Create non-linear relationship: sigmoid-like ground truth
            let sigmoid_input = (pred - 0.5) * 10.0; // Center around 0.5
            let ground_truth = 1.0 / (1.0 + (-sigmoid_input).exp());
            samples.push(create_test_sample(pred, ground_truth));
        }
        
        let sample_refs: Vec<&CalibrationSample> = samples.iter().collect();
        scaler.train(&sample_refs).await.unwrap();
        
        // Test calibration
        let features = HashMap::new();
        let calibrated = scaler.calibrate(0.7, &features).await.unwrap();
        
        // Should be a valid probability
        assert!(calibrated >= 0.001 && calibrated <= 0.999);
        
        // Check that parameters were learned
        let (a, b) = scaler.get_parameters();
        assert!(a.is_finite());
        assert!(b.is_finite());
        assert!(a.abs() < 100.0);
        assert!(b.abs() < 100.0);
        
        println!("Platt parameters: A={:.4}, B={:.4}, ECE={:.4}", a, b, scaler.get_ece());
    }

    #[tokio::test]
    async fn test_platt_overconfident_predictions() {
        let config = PlattConfig {
            max_iterations: 40,
            convergence_tolerance: 1e-4,
        };
        let mut scaler = PlattScaler::new(config);
        
        // Create overconfident predictions that need strong calibration
        let mut samples = Vec::new();
        for i in 0..40 {
            let pred = 0.8 + (i as f32 / 40.0) * 0.19; // High predictions [0.8, 0.99]
            let ground_truth = if i < 20 { 1.0 } else { 0.0 }; // 50% should be positive
            samples.push(create_test_sample(pred, ground_truth));
        }
        
        let sample_refs: Vec<&CalibrationSample> = samples.iter().collect();
        scaler.train(&sample_refs).await.unwrap();
        
        // Test calibration - should reduce overconfident predictions
        let features = HashMap::new();
        let calibrated_high = scaler.calibrate(0.95, &features).await.unwrap();
        let calibrated_med = scaler.calibrate(0.85, &features).await.unwrap();
        
        // Both should be valid probabilities
        assert!(calibrated_high >= 0.001 && calibrated_high <= 0.999);
        assert!(calibrated_med >= 0.001 && calibrated_med <= 0.999);
        
        // Higher prediction should generally give higher calibrated score
        assert!(calibrated_high >= calibrated_med);
        
        println!("Platt calibration: 0.95 -> {:.4}, 0.85 -> {:.4}", calibrated_high, calibrated_med);
    }

    #[tokio::test] 
    async fn test_platt_likelihood_calculation() {
        let config = PlattConfig::default();
        let scaler = PlattScaler::new(config);
        
        let predictions = vec![0.3, 0.7, 0.9];
        let targets = vec![0.0, 1.0, 1.0];
        
        let mut state = PlattOptimizationState {
            a: -1.0,
            b: 0.0,
            likelihood: 0.0,
            gradient_a: 0.0,
            gradient_b: 0.0,
            hessian_aa: 0.0,
            hessian_ab: 0.0,
            hessian_bb: 0.0,
        };
        
        let result = scaler.calculate_likelihood_and_derivatives(&predictions, &targets, &mut state);
        assert!(result.is_ok());
        
        // Check that all values are finite
        assert!(state.likelihood.is_finite());
        assert!(state.gradient_a.is_finite());
        assert!(state.gradient_b.is_finite());
        assert!(state.hessian_aa.is_finite());
        assert!(state.hessian_ab.is_finite());
        assert!(state.hessian_bb.is_finite());
        
        // Likelihood should be negative (log probabilities)
        assert!(state.likelihood < 0.0);
    }

    #[test]
    fn test_parameter_bounds() {
        let config = PlattConfig::default();
        let mut scaler = PlattScaler::new(config);
        
        // Set extreme parameters
        scaler.parameter_a = 150.0; // Too large
        scaler.parameter_b = -200.0; // Too large
        scaler.final_likelihood = f64::INFINITY;
        
        // Should not converge with extreme parameters
        assert!(!scaler.check_convergence().unwrap());
        
        // Set reasonable parameters
        scaler.parameter_a = -2.0;
        scaler.parameter_b = 1.0;
        scaler.final_likelihood = -0.5;
        scaler.slice_ece = 0.01;
        
        // Should converge with reasonable parameters and ECE
        assert!(scaler.check_convergence().unwrap());
    }
}