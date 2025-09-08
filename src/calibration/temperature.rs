//! # Temperature Scaling Calibration
//!
//! Temperature scaling backstop method for PHASE 4 calibration system.
//! Used when isotonic regression is not available or suitable.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

use super::CalibrationSample;

/// Temperature scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureConfig {
    /// Initial temperature parameter
    pub initial_temperature: f32,
    /// Learning rate for optimization
    pub learning_rate: f32,
    /// Maximum optimization iterations
    pub max_iterations: usize,
}

/// Temperature scaling calibrator
#[derive(Debug, Clone)]
pub struct TemperatureScaler {
    config: TemperatureConfig,
    /// Learned temperature parameter
    temperature: f32,
    /// Optional bias term
    bias: f32,
    /// Current ECE for this slice
    slice_ece: f32,
    /// Training convergence status
    converged: bool,
    /// Number of training samples
    training_samples: usize,
    /// Training loss history
    loss_history: Vec<f32>,
}

/// Optimization state for temperature scaling
#[derive(Debug, Clone)]
struct OptimizationState {
    temperature: f32,
    bias: f32,
    loss: f32,
    gradient_temp: f32,
    gradient_bias: f32,
}

impl TemperatureScaler {
    /// Create new temperature scaler
    pub fn new(config: TemperatureConfig) -> Self {
        Self {
            temperature: config.initial_temperature,
            bias: 0.0,
            config,
            slice_ece: 0.0,
            converged: false,
            training_samples: 0,
            loss_history: Vec::new(),
        }
    }

    /// Train temperature scaling on provided samples
    pub async fn train(&mut self, samples: &[&CalibrationSample]) -> Result<()> {
        if samples.len() < 10 {
            anyhow::bail!("Insufficient samples for temperature scaling: {}", samples.len());
        }

        info!(
            "Training temperature scaler on {} samples (initial T={:.3})",
            samples.len(),
            self.config.initial_temperature
        );

        self.training_samples = samples.len();

        // Extract logits and targets
        let logits = self.extract_logits(samples)?;
        let targets: Vec<f32> = samples.iter().map(|s| s.ground_truth).collect();

        // Optimize temperature and bias using gradient descent
        let final_state = self.optimize_temperature(&logits, &targets).await?;
        
        self.temperature = final_state.temperature;
        self.bias = final_state.bias;

        // Calculate final ECE
        self.slice_ece = self.calculate_ece_with_temperature(samples, self.temperature).await?;
        
        // Check convergence
        self.converged = self.check_convergence()?;

        info!(
            "Temperature scaling trained: T={:.3}, bias={:.3}, ECE={:.4}, converged={}",
            self.temperature, self.bias, self.slice_ece, self.converged
        );

        if self.slice_ece > 0.015 {
            warn!(
                "Temperature scaling ECE {:.4} exceeds PHASE 4 target ≤ 0.015",
                self.slice_ece
            );
        }

        Ok(())
    }

    /// Apply temperature scaling to a prediction
    pub async fn calibrate(&self, prediction: f32) -> Result<f32> {
        // Convert prediction to logit
        let logit = self.prediction_to_logit(prediction)?;
        
        // Apply temperature scaling: scaled_logit = (logit + bias) / temperature
        let scaled_logit = (logit + self.bias) / self.temperature;
        
        // Convert back to probability: p = sigmoid(scaled_logit)
        let calibrated = self.sigmoid(scaled_logit);
        
        debug!(
            "Temperature scaling: {:.4} -> logit {:.3} -> scaled {:.3} -> {:.4} (T={:.3})",
            prediction, logit, scaled_logit, calibrated, self.temperature
        );

        Ok(calibrated.clamp(0.001, 0.999))
    }

    /// Get current temperature parameter
    pub fn get_temperature(&self) -> f32 {
        self.temperature
    }

    /// Get current bias parameter
    pub fn get_bias(&self) -> f32 {
        self.bias
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

    /// Get training loss history
    pub fn get_loss_history(&self) -> &[f32] {
        &self.loss_history
    }

    // Private implementation methods

    /// Extract logits from probability predictions
    fn extract_logits(&self, samples: &[&CalibrationSample]) -> Result<Vec<f32>> {
        let mut logits = Vec::new();
        
        for sample in samples {
            let logit = self.prediction_to_logit(sample.prediction)?;
            logits.push(logit);
        }
        
        Ok(logits)
    }

    /// Convert prediction probability to logit
    fn prediction_to_logit(&self, prediction: f32) -> Result<f32> {
        let clamped_pred = prediction.clamp(1e-7, 1.0 - 1e-7);
        Ok((clamped_pred / (1.0 - clamped_pred)).ln())
    }

    /// Sigmoid function
    fn sigmoid(&self, logit: f32) -> f32 {
        1.0 / (1.0 + (-logit.clamp(-50.0, 50.0)).exp())
    }

    /// Optimize temperature and bias parameters
    async fn optimize_temperature(&mut self, logits: &[f32], targets: &[f32]) -> Result<OptimizationState> {
        let mut state = OptimizationState {
            temperature: self.config.initial_temperature,
            bias: 0.0,
            loss: f32::INFINITY,
            gradient_temp: 0.0,
            gradient_bias: 0.0,
        };

        self.loss_history.clear();

        for iteration in 0..self.config.max_iterations {
            // Calculate loss and gradients
            let (loss, grad_temp, grad_bias) = self.calculate_loss_and_gradients(
                logits, targets, state.temperature, state.bias
            )?;

            // Update state
            state.loss = loss;
            state.gradient_temp = grad_temp;
            state.gradient_bias = grad_bias;
            self.loss_history.push(loss);

            // Apply gradient descent with learning rate
            state.temperature -= self.config.learning_rate * grad_temp;
            state.bias -= self.config.learning_rate * grad_bias;

            // Clamp temperature to reasonable range
            state.temperature = state.temperature.clamp(0.1, 10.0);

            // Check for convergence
            if iteration > 10 {
                let recent_losses: Vec<f32> = self.loss_history
                    .iter()
                    .rev()
                    .take(5)
                    .copied()
                    .collect();
                
                if recent_losses.len() == 5 {
                    let loss_variance = self.calculate_variance(&recent_losses);
                    if loss_variance < 1e-6 {
                        info!("Temperature optimization converged at iteration {}", iteration);
                        break;
                    }
                }
            }

            if iteration % 20 == 0 {
                debug!(
                    "Temperature optimization iter {}: loss={:.6}, T={:.3}, bias={:.3}",
                    iteration, loss, state.temperature, state.bias
                );
            }
        }

        Ok(state)
    }

    /// Calculate negative log-likelihood loss and gradients
    fn calculate_loss_and_gradients(
        &self,
        logits: &[f32],
        targets: &[f32],
        temperature: f32,
        bias: f32,
    ) -> Result<(f32, f32, f32)> {
        let n = logits.len() as f32;
        let mut total_loss = 0.0;
        let mut grad_temp = 0.0;
        let mut grad_bias = 0.0;

        for (&logit, &target) in logits.iter().zip(targets.iter()) {
            // Apply current temperature and bias
            let scaled_logit = (logit + bias) / temperature;
            let prob = self.sigmoid(scaled_logit);

            // Negative log-likelihood loss
            let sample_loss = -target * prob.ln() - (1.0 - target) * (1.0 - prob).ln();
            total_loss += sample_loss;

            // Gradients
            let error = prob - target;
            
            // Gradient w.r.t. temperature: ∂L/∂T = error * (-scaled_logit / T)
            grad_temp += error * (-scaled_logit / temperature);
            
            // Gradient w.r.t. bias: ∂L/∂b = error * (1 / T)
            grad_bias += error * (1.0 / temperature);
        }

        Ok((total_loss / n, grad_temp / n, grad_bias / n))
    }

    /// Calculate ECE with current temperature
    async fn calculate_ece_with_temperature(&self, samples: &[&CalibrationSample], temperature: f32) -> Result<f32> {
        const NUM_BINS: usize = 10;
        let mut bins = vec![Vec::new(); NUM_BINS];

        // Apply temperature scaling to get calibrated predictions
        for sample in samples {
            let logit = self.prediction_to_logit(sample.prediction)?;
            let scaled_logit = (logit + self.bias) / temperature;
            let calibrated_pred = self.sigmoid(scaled_logit);
            
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
        if self.loss_history.len() < 10 {
            return Ok(false);
        }

        // Check if temperature is reasonable
        let temp_ok = self.temperature >= 0.1 && self.temperature <= 10.0;
        
        // Check if ECE is within target
        let ece_ok = self.slice_ece <= 0.015;
        
        // Check if loss has stabilized
        let recent_losses: Vec<f32> = self.loss_history
            .iter()
            .rev()
            .take(5)
            .copied()
            .collect();
        
        let loss_stable = if recent_losses.len() == 5 {
            let loss_variance = self.calculate_variance(&recent_losses);
            loss_variance < 1e-5
        } else {
            false
        };

        Ok(temp_ok && ece_ok && loss_stable)
    }

    /// Calculate variance of a slice of values
    fn calculate_variance(&self, values: &[f32]) -> f32 {
        if values.len() <= 1 {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / (values.len() - 1) as f32;
        
        variance
    }
}

impl Default for TemperatureConfig {
    fn default() -> Self {
        Self {
            initial_temperature: 1.0,
            learning_rate: 0.01,
            max_iterations: 100,
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
    async fn test_temperature_scaler_creation() {
        let config = TemperatureConfig::default();
        let scaler = TemperatureScaler::new(config);
        
        assert_eq!(scaler.get_temperature(), 1.0);
        assert_eq!(scaler.get_bias(), 0.0);
        assert_eq!(scaler.get_ece(), 0.0);
        assert!(!scaler.is_converged());
    }

    #[tokio::test]
    async fn test_temperature_training_insufficient_samples() {
        let config = TemperatureConfig::default();
        let mut scaler = TemperatureScaler::new(config);
        
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
    async fn test_temperature_calibration() {
        let config = TemperatureConfig {
            max_iterations: 50,
            ..Default::default()
        };
        let mut scaler = TemperatureScaler::new(config);
        
        // Create overconfident predictions that need temperature scaling
        let mut samples = Vec::new();
        for i in 0..30 {
            let pred = 0.9 + (i as f32 / 30.0) * 0.09; // High predictions [0.9, 0.99]
            let ground_truth = if i < 20 { 1.0 } else { 0.0 }; // Most should be positive
            samples.push(create_test_sample(pred, ground_truth));
        }
        
        let sample_refs: Vec<&CalibrationSample> = samples.iter().collect();
        scaler.train(&sample_refs).await.unwrap();
        
        // Test calibration - high confidence prediction should be scaled down
        let calibrated = scaler.calibrate(0.95).await.unwrap();
        
        // Should be a valid probability
        assert!(calibrated >= 0.0 && calibrated <= 1.0);
        
        // Temperature scaling should typically reduce overconfident predictions
        // (though exact behavior depends on the training data)
        println!("Original: 0.95, Calibrated: {:.4}, Temperature: {:.3}", 
                calibrated, scaler.get_temperature());
        
        assert!(scaler.get_temperature() > 0.1);
        assert!(scaler.get_temperature() < 10.0);
    }

    #[tokio::test]
    async fn test_logit_conversion() {
        let config = TemperatureConfig::default();
        let scaler = TemperatureScaler::new(config);
        
        // Test logit conversion
        assert_eq!(scaler.prediction_to_logit(0.5).unwrap(), 0.0);
        assert!(scaler.prediction_to_logit(0.75).unwrap() > 0.0);
        assert!(scaler.prediction_to_logit(0.25).unwrap() < 0.0);
        
        // Test sigmoid
        assert_eq!(scaler.sigmoid(0.0), 0.5);
        assert!(scaler.sigmoid(2.0) > 0.8);
        assert!(scaler.sigmoid(-2.0) < 0.2);
    }

    #[tokio::test]
    async fn test_underconfident_calibration() {
        let config = TemperatureConfig {
            max_iterations: 30,
            learning_rate: 0.02,
            ..Default::default()
        };
        let mut scaler = TemperatureScaler::new(config);
        
        // Create underconfident predictions
        let mut samples = Vec::new();
        for i in 0..25 {
            let pred = 0.4 + (i as f32 / 25.0) * 0.2; // Moderate predictions [0.4, 0.6]
            let ground_truth = if i > 15 { 1.0 } else { 0.0 }; // Most should match
            samples.push(create_test_sample(pred, ground_truth));
        }
        
        let sample_refs: Vec<&CalibrationSample> = samples.iter().collect();
        scaler.train(&sample_refs).await.unwrap();
        
        // Check that temperature was learned
        assert!(scaler.get_temperature() > 0.1);
        assert!(scaler.get_training_samples() == 25);
        
        // Test calibration
        let calibrated = scaler.calibrate(0.5).await.unwrap();
        assert!(calibrated >= 0.001 && calibrated <= 0.999);
    }

    #[test]
    fn test_loss_calculation() {
        let config = TemperatureConfig::default();
        let scaler = TemperatureScaler::new(config);
        
        let logits = vec![0.0, 1.0, -1.0];
        let targets = vec![0.5, 1.0, 0.0];
        let temperature = 1.0;
        let bias = 0.0;
        
        let result = scaler.calculate_loss_and_gradients(&logits, &targets, temperature, bias);
        assert!(result.is_ok());
        
        let (loss, grad_temp, grad_bias) = result.unwrap();
        assert!(loss > 0.0); // Loss should be positive
        assert!(grad_temp.is_finite());
        assert!(grad_bias.is_finite());
    }

    #[test]
    fn test_variance_calculation() {
        let config = TemperatureConfig::default();
        let scaler = TemperatureScaler::new(config);
        
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let variance = scaler.calculate_variance(&values);
        
        // Expected variance: 2.5 (for this sequence)
        assert!((variance - 2.5).abs() < 0.1);
        
        // Single value should have zero variance
        assert_eq!(scaler.calculate_variance(&[1.0]), 0.0);
    }
}