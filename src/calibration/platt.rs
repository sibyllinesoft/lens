/// Platt Scaling Calibration for CALIB_V22
/// Provides sigmoid-based calibration for prediction probabilities using Platt scaling

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};
use thiserror::Error;

/// Platt calibrator for transforming uncalibrated scores into calibrated probabilities
/// Uses sigmoid function: P(y=1|s) = 1 / (1 + exp(A*s + B))
#[derive(Debug, Clone)]
pub struct PlattCalibrator {
    /// Sigmoid parameter A
    pub parameter_a: f64,
    
    /// Sigmoid parameter B  
    pub parameter_b: f64,
    
    /// Metadata about the calibration fitting process
    pub fit_metadata: PlattFitMetadata,
    
    /// Whether the calibrator has been fitted
    pub is_fitted: bool,
}

/// Metadata about the Platt calibration fitting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlattFitMetadata {
    /// Total number of training samples used
    pub total_samples: usize,
    
    /// Number of positive samples in training
    pub positive_samples: usize,
    
    /// Number of negative samples in training
    pub negative_samples: usize,
    
    /// Training set ECE (Expected Calibration Error)
    pub training_ece: f64,
    
    /// Fitting convergence status
    pub converged: bool,
    
    /// Number of iterations used in fitting
    pub iterations_used: u32,
    
    /// Final log-likelihood value
    pub final_log_likelihood: f64,
    
    /// Fit timestamp
    pub fit_timestamp: std::time::SystemTime,
}

#[derive(Debug, Error)]
pub enum PlattError {
    #[error("Insufficient data for calibration: need at least {required} samples, got {actual}")]
    InsufficientData { required: usize, actual: usize },
    
    #[error("Invalid probability values: {message}")]
    InvalidProbabilities { message: String },
    
    #[error("Calibration fitting failed: {message}")]
    FittingFailed { message: String },
    
    #[error("Prediction failed: {message}")]
    PredictionFailed { message: String },
    
    #[error("Optimization did not converge after {iterations} iterations")]
    ConvergenceError { iterations: u32 },
}

impl Default for PlattCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl PlattCalibrator {
    /// Create new Platt calibrator
    pub fn new() -> Self {
        Self {
            parameter_a: 0.0,
            parameter_b: 0.0,
            fit_metadata: PlattFitMetadata {
                total_samples: 0,
                positive_samples: 0,
                negative_samples: 0,
                training_ece: 0.0,
                converged: false,
                iterations_used: 0,
                final_log_likelihood: f64::NEG_INFINITY,
                fit_timestamp: std::time::SystemTime::now(),
            },
            is_fitted: false,
        }
    }
    
    /// Fit Platt scaling on training data using maximum likelihood estimation
    /// 
    /// # Arguments
    /// * `scores` - Uncalibrated prediction scores (decision values)
    /// * `true_labels` - Ground truth binary labels (0.0 or 1.0)
    /// 
    /// # Returns
    /// * `Ok(())` if fitting succeeds
    /// * `Err(PlattError)` if fitting fails
    pub fn fit(&mut self, scores: &[f64], true_labels: &[f64]) -> Result<(), PlattError> {
        if scores.len() != true_labels.len() {
            return Err(PlattError::InvalidProbabilities {
                message: format!("Scores and labels length mismatch: {} vs {}", scores.len(), true_labels.len())
            });
        }
        
        if scores.len() < 2 {
            return Err(PlattError::InsufficientData { 
                required: 2, 
                actual: scores.len() 
            });
        }
        
        info!("Fitting Platt scaling on {} samples", scores.len());
        
        // Count positive and negative samples
        let positive_count = true_labels.iter().filter(|&&label| label > 0.5).count();
        let negative_count = scores.len() - positive_count;
        
        if positive_count == 0 || negative_count == 0 {
            return Err(PlattError::InvalidProbabilities {
                message: "Need both positive and negative samples for Platt scaling".to_string()
            });
        }
        
        // Transform labels to target values (with label smoothing to avoid overfitting)
        let target_pos = (positive_count as f64 + 1.0) / (positive_count as f64 + 2.0);
        let target_neg = 1.0 / (negative_count as f64 + 2.0);
        
        let targets: Vec<f64> = true_labels.iter().map(|&label| {
            if label > 0.5 { target_pos } else { target_neg }
        }).collect();
        
        // Fit sigmoid parameters using Newton-Raphson method
        let (param_a, param_b, converged, iterations, final_ll) = 
            self.fit_sigmoid_parameters(scores, &targets)?;
        
        // Store parameters and metadata
        self.parameter_a = param_a;
        self.parameter_b = param_b;
        self.is_fitted = true;
        
        // Calculate training ECE
        let training_ece = self.calculate_ece(scores, true_labels);
        
        self.fit_metadata = PlattFitMetadata {
            total_samples: scores.len(),
            positive_samples: positive_count,
            negative_samples: negative_count,
            training_ece,
            converged,
            iterations_used: iterations,
            final_log_likelihood: final_ll,
            fit_timestamp: std::time::SystemTime::now(),
        };
        
        info!("Platt scaling fitted: A={:.4}, B={:.4}, ECE={:.4}, converged={}", 
              param_a, param_b, training_ece, converged);
        
        Ok(())
    }
    
    /// Fit sigmoid parameters using Newton-Raphson optimization
    fn fit_sigmoid_parameters(&self, scores: &[f64], targets: &[f64]) -> Result<(f64, f64, bool, u32, f64), PlattError> {
        const MAX_ITERATIONS: u32 = 100;
        const TOLERANCE: f64 = 1e-12;
        const MIN_STEP: f64 = 1e-10;
        
        // Initialize parameters
        let mut a = 0.0;
        let mut b = 0.0;
        
        // Precompute some statistics for initialization
        let n = scores.len() as f64;
        let sum_targets: f64 = targets.iter().sum();
        
        // Better initialization based on data
        let mean_score: f64 = scores.iter().sum::<f64>() / n;
        let mean_target = sum_targets / n;
        
        // Initialize B to get reasonable starting probabilities
        b = -(mean_target.ln() - (1.0 - mean_target).ln());
        
        for iteration in 0..MAX_ITERATIONS {
            let mut gradient_a = 0.0;
            let mut gradient_b = 0.0;
            let mut hessian_aa = 0.0;
            let mut hessian_ab = 0.0;
            let mut hessian_bb = 0.0;
            let mut log_likelihood = 0.0;
            
            // Compute gradients and Hessian
            for i in 0..scores.len() {
                let score = scores[i];
                let target = targets[i];
                
                // Compute probability using current parameters
                let fval = a * score + b;
                let p = sigmoid(fval);
                
                // Avoid numerical issues
                let p_clamped = p.clamp(1e-15, 1.0 - 1e-15);
                
                // Log likelihood
                log_likelihood += target * p_clamped.ln() + (1.0 - target) * (1.0 - p_clamped).ln();
                
                // First derivatives
                let d1 = target - p;
                gradient_a += d1 * score;
                gradient_b += d1;
                
                // Second derivatives 
                let d2 = p * (1.0 - p);
                hessian_aa += score * score * d2;
                hessian_ab += score * d2;
                hessian_bb += d2;
            }
            
            // Check convergence
            let grad_norm = (gradient_a * gradient_a + gradient_b * gradient_b).sqrt();
            if grad_norm < TOLERANCE {
                return Ok((a, b, true, iteration, log_likelihood));
            }
            
            // Compute Newton step
            let det = hessian_aa * hessian_bb - hessian_ab * hessian_ab;
            if det.abs() < MIN_STEP {
                // Singular Hessian, use gradient descent
                let step_size = 0.01;
                a += step_size * gradient_a;
                b += step_size * gradient_b;
            } else {
                // Newton-Raphson step
                let step_a = (hessian_bb * gradient_a - hessian_ab * gradient_b) / det;
                let step_b = (hessian_aa * gradient_b - hessian_ab * gradient_a) / det;
                
                a += step_a;
                b += step_b;
            }
        }
        
        // Did not converge
        warn!("Platt scaling did not converge after {} iterations", MAX_ITERATIONS);
        Ok((a, b, false, MAX_ITERATIONS, f64::NEG_INFINITY))
    }
    
    /// Predict calibrated probabilities for new scores
    pub fn predict(&self, scores: &[f64]) -> Result<Vec<f64>, PlattError> {
        if !self.is_fitted {
            return Err(PlattError::PredictionFailed {
                message: "Calibrator has not been fitted yet".to_string()
            });
        }
        
        let mut predictions = Vec::with_capacity(scores.len());
        
        for &score in scores {
            let fval = self.parameter_a * score + self.parameter_b;
            let probability = sigmoid(fval);
            predictions.push(probability);
        }
        
        Ok(predictions)
    }
    
    /// Calculate Expected Calibration Error
    fn calculate_ece(&self, scores: &[f64], true_labels: &[f64]) -> f64 {
        if scores.is_empty() || !self.is_fitted {
            return 0.0;
        }
        
        let predictions = match self.predict(scores) {
            Ok(preds) => preds,
            Err(_) => return f64::NAN,
        };
        
        // Use 10 bins for ECE calculation
        const NUM_BINS: usize = 10;
        let mut bins: Vec<Vec<(f64, f64)>> = vec![Vec::new(); NUM_BINS];
        
        // Assign predictions to bins
        for i in 0..predictions.len() {
            let bin_idx = ((predictions[i] * NUM_BINS as f64).floor() as usize).min(NUM_BINS - 1);
            bins[bin_idx].push((predictions[i], true_labels[i]));
        }
        
        // Calculate ECE
        let mut ece = 0.0;
        let total_samples = predictions.len() as f64;
        
        for bin in &bins {
            if bin.is_empty() {
                continue;
            }
            
            let bin_size = bin.len() as f64;
            let avg_confidence: f64 = bin.iter().map(|(conf, _)| conf).sum::<f64>() / bin_size;
            let avg_accuracy: f64 = bin.iter().map(|(_, acc)| acc).sum::<f64>() / bin_size;
            
            ece += (bin_size / total_samples) * (avg_confidence - avg_accuracy).abs();
        }
        
        ece
    }
    
    /// Get calibration statistics for monitoring
    pub fn get_calibration_stats(&self) -> CalibrationStats {
        CalibrationStats {
            parameter_a: self.parameter_a,
            parameter_b: self.parameter_b,
            is_fitted: self.is_fitted,
            converged: self.fit_metadata.converged,
            training_ece: self.fit_metadata.training_ece,
            total_training_samples: self.fit_metadata.total_samples,
            positive_training_samples: self.fit_metadata.positive_samples,
            negative_training_samples: self.fit_metadata.negative_samples,
        }
    }
}

/// Statistics about the Platt calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationStats {
    pub parameter_a: f64,
    pub parameter_b: f64,
    pub is_fitted: bool,
    pub converged: bool,
    pub training_ece: f64,
    pub total_training_samples: usize,
    pub positive_training_samples: usize,
    pub negative_training_samples: usize,
}

/// Sigmoid function: 1 / (1 + exp(-x))
fn sigmoid(x: f64) -> f64 {
    // Numerical stable sigmoid implementation
    if x >= 0.0 {
        let exp_neg_x = (-x).exp();
        1.0 / (1.0 + exp_neg_x)
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_platt_calibrator_creation() {
        let calibrator = PlattCalibrator::new();
        assert!(!calibrator.is_fitted);
        assert_eq!(calibrator.parameter_a, 0.0);
        assert_eq!(calibrator.parameter_b, 0.0);
    }
    
    #[test]
    fn test_sigmoid_function() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(1000.0) > 0.99);
        assert!(sigmoid(-1000.0) < 0.01);
    }
    
    #[test]
    fn test_platt_calibrator_fit() {
        let mut calibrator = PlattCalibrator::new();
        
        // Create test data with known pattern
        let scores = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        
        let result = calibrator.fit(&scores, &labels);
        assert!(result.is_ok());
        assert!(calibrator.is_fitted);
        
        // Parameters should be reasonable
        assert!(calibrator.parameter_a.is_finite());
        assert!(calibrator.parameter_b.is_finite());
    }
    
    #[test]
    fn test_platt_prediction() {
        let mut calibrator = PlattCalibrator::new();
        
        // Fit calibrator with simple data
        let scores = vec![-1.0, 0.0, 1.0];
        let labels = vec![0.0, 0.5, 1.0];
        calibrator.fit(&scores, &labels).unwrap();
        
        // Test prediction
        let test_scores = vec![-0.5, 0.5];
        let predictions = calibrator.predict(&test_scores);
        assert!(predictions.is_ok());
        
        let preds = predictions.unwrap();
        assert_eq!(preds.len(), 2);
        
        // All predictions should be valid probabilities
        for &pred in &preds {
            assert!(pred >= 0.0 && pred <= 1.0);
        }
        
        // Higher scores should give higher probabilities
        assert!(preds[1] > preds[0]);
    }
    
    #[test]
    fn test_platt_insufficient_data() {
        let mut calibrator = PlattCalibrator::new();
        
        // Test with insufficient data
        let scores = vec![0.0];
        let labels = vec![1.0];
        
        let result = calibrator.fit(&scores, &labels);
        assert!(result.is_err());
        
        match result {
            Err(PlattError::InsufficientData { required: 2, actual: 1 }) => {},
            _ => panic!("Expected InsufficientData error"),
        }
    }
    
    #[test]
    fn test_platt_no_positive_samples() {
        let mut calibrator = PlattCalibrator::new();
        
        // Test with no positive samples
        let scores = vec![0.0, 1.0, 2.0];
        let labels = vec![0.0, 0.0, 0.0];
        
        let result = calibrator.fit(&scores, &labels);
        assert!(result.is_err());
        
        match result {
            Err(PlattError::InvalidProbabilities { .. }) => {},
            _ => panic!("Expected InvalidProbabilities error"),
        }
    }
}