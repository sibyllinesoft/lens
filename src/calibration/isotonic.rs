/// Isotonic Calibration for CALIB_V22
/// Provides isotonic regression for calibrating prediction probabilities

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};
use thiserror::Error;

/// Isotonic calibrator for transforming uncalibrated scores into calibrated probabilities
#[derive(Debug, Clone)]
pub struct IsotonicCalibrator {
    /// Monotonic mapping from uncalibrated scores to calibrated probabilities
    pub calibration_map: Vec<CalibrationPoint>,
    
    /// Metadata about the calibration fitting process
    pub fit_metadata: IsotonicFitMetadata,
}

/// Single point in the isotonic calibration mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationPoint {
    /// Input score (uncalibrated)
    pub score: f64,
    
    /// Output probability (calibrated)
    pub probability: f64,
    
    /// Number of samples that contributed to this point
    pub sample_count: u32,
}

/// Metadata about the isotonic calibration fitting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsotonicFitMetadata {
    /// Number of calibration points in the final mapping
    pub num_points: usize,
    
    /// Total number of training samples used
    pub total_samples: usize,
    
    /// Training set ECE (Expected Calibration Error)
    pub training_ece: f64,
    
    /// Whether the calibration is strictly monotonic
    pub is_monotonic: bool,
    
    /// Fit timestamp
    pub fit_timestamp: std::time::SystemTime,
}

#[derive(Debug, Error)]
pub enum IsotonicError {
    #[error("Insufficient data for calibration: need at least {required} samples, got {actual}")]
    InsufficientData { required: usize, actual: usize },
    
    #[error("Invalid probability values: {message}")]
    InvalidProbabilities { message: String },
    
    #[error("Calibration fitting failed: {message}")]
    FittingFailed { message: String },
    
    #[error("Prediction failed: {message}")]
    PredictionFailed { message: String },
}

impl Default for IsotonicCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl IsotonicCalibrator {
    /// Create new isotonic calibrator
    pub fn new() -> Self {
        Self {
            calibration_map: Vec::new(),
            fit_metadata: IsotonicFitMetadata {
                num_points: 0,
                total_samples: 0,
                training_ece: 0.0,
                is_monotonic: false,
                fit_timestamp: std::time::SystemTime::now(),
            },
        }
    }
    
    /// Fit isotonic calibration on training data
    /// 
    /// # Arguments
    /// * `scores` - Uncalibrated prediction scores
    /// * `true_labels` - Ground truth binary labels (0.0 or 1.0)
    /// 
    /// # Returns
    /// * `Ok(())` if fitting succeeds
    /// * `Err(IsotonicError)` if fitting fails
    pub fn fit(&mut self, scores: &[f64], true_labels: &[f64]) -> Result<(), IsotonicError> {
        if scores.len() != true_labels.len() {
            return Err(IsotonicError::InvalidProbabilities {
                message: format!("Scores and labels length mismatch: {} vs {}", scores.len(), true_labels.len())
            });
        }
        
        if scores.len() < 10 {
            return Err(IsotonicError::InsufficientData { 
                required: 10, 
                actual: scores.len() 
            });
        }
        
        info!("Fitting isotonic calibration on {} samples", scores.len());
        
        // Combine scores and labels, sort by score
        let mut data: Vec<(f64, f64)> = scores.iter().zip(true_labels.iter()).map(|(&s, &l)| (s, l)).collect();
        data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        
        // Apply isotonic regression using Pool Adjacent Violators Algorithm (PAVA)
        let calibrated_points = self.apply_pava(&data)?;
        
        // Store calibration mapping
        self.calibration_map = calibrated_points;
        
        // Calculate metadata
        let training_ece = self.calculate_ece(scores, true_labels);
        self.fit_metadata = IsotonicFitMetadata {
            num_points: self.calibration_map.len(),
            total_samples: scores.len(),
            training_ece,
            is_monotonic: self.verify_monotonicity(),
            fit_timestamp: std::time::SystemTime::now(),
        };
        
        info!("Isotonic calibration fitted: {} points, ECE: {:.4}", 
              self.fit_metadata.num_points, training_ece);
        
        Ok(())
    }
    
    /// Apply Pool Adjacent Violators Algorithm for isotonic regression
    fn apply_pava(&self, data: &[(f64, f64)]) -> Result<Vec<CalibrationPoint>, IsotonicError> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        
        // Group data points by score bins for efficiency
        let mut bins = HashMap::new();
        for &(score, label) in data {
            let entry = bins.entry(score).or_insert((0.0, 0));
            entry.0 += label;
            entry.1 += 1;
        }
        
        // Convert to sorted list of (score, avg_label, count)
        let mut sorted_bins: Vec<(f64, f64, u32)> = bins.into_iter()
            .map(|(score, (sum_labels, count))| {
                (score, sum_labels / count as f64, count)
            })
            .collect();
        sorted_bins.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        
        // Apply PAVA to ensure monotonicity
        let mut result = Vec::new();
        let mut i = 0;
        
        while i < sorted_bins.len() {
            let mut current_score = sorted_bins[i].0;
            let mut current_prob = sorted_bins[i].1;
            let mut current_count = sorted_bins[i].2;
            let mut j = i + 1;
            
            // Look ahead to find any violations and pool them
            while j < sorted_bins.len() && sorted_bins[j].1 < current_prob {
                // Pool adjacent violators
                let total_samples = current_count + sorted_bins[j].2;
                current_prob = (current_prob * current_count as f64 + sorted_bins[j].1 * sorted_bins[j].2 as f64) / total_samples as f64;
                current_count = total_samples;
                current_score = sorted_bins[j].0; // Use the rightmost score
                j += 1;
            }
            
            result.push(CalibrationPoint {
                score: current_score,
                probability: current_prob.clamp(0.0, 1.0),
                sample_count: current_count,
            });
            
            i = j;
        }
        
        Ok(result)
    }
    
    /// Predict calibrated probabilities for new scores
    pub fn predict(&self, scores: &[f64]) -> Result<Vec<f64>, IsotonicError> {
        if self.calibration_map.is_empty() {
            return Err(IsotonicError::PredictionFailed {
                message: "Calibrator has not been fitted yet".to_string()
            });
        }
        
        let mut predictions = Vec::with_capacity(scores.len());
        
        for &score in scores {
            let calibrated_prob = self.interpolate_probability(score);
            predictions.push(calibrated_prob);
        }
        
        Ok(predictions)
    }
    
    /// Interpolate calibrated probability for a single score
    fn interpolate_probability(&self, score: f64) -> f64 {
        if self.calibration_map.is_empty() {
            return 0.5; // Default to neutral probability
        }
        
        // Handle edge cases
        if score <= self.calibration_map[0].score {
            return self.calibration_map[0].probability;
        }
        
        if score >= self.calibration_map.last().unwrap().score {
            return self.calibration_map.last().unwrap().probability;
        }
        
        // Find the interpolation interval
        for i in 0..self.calibration_map.len() - 1 {
            if score >= self.calibration_map[i].score && score <= self.calibration_map[i + 1].score {
                // Linear interpolation
                let x0 = self.calibration_map[i].score;
                let x1 = self.calibration_map[i + 1].score;
                let y0 = self.calibration_map[i].probability;
                let y1 = self.calibration_map[i + 1].probability;
                
                if x1 == x0 {
                    return y0;
                }
                
                let interpolated = y0 + (score - x0) * (y1 - y0) / (x1 - x0);
                return interpolated.clamp(0.0, 1.0);
            }
        }
        
        // Fallback (should not reach here)
        0.5
    }
    
    /// Calculate Expected Calibration Error
    fn calculate_ece(&self, scores: &[f64], true_labels: &[f64]) -> f64 {
        if scores.is_empty() {
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
    
    /// Verify that the calibration mapping is monotonic
    fn verify_monotonicity(&self) -> bool {
        if self.calibration_map.len() <= 1 {
            return true;
        }
        
        for i in 1..self.calibration_map.len() {
            if self.calibration_map[i].probability < self.calibration_map[i - 1].probability {
                return false;
            }
        }
        
        true
    }
    
    /// Get calibration statistics for monitoring
    pub fn get_calibration_stats(&self) -> CalibrationStats {
        CalibrationStats {
            num_calibration_points: self.calibration_map.len(),
            score_range: if self.calibration_map.is_empty() {
                (0.0, 0.0)
            } else {
                (self.calibration_map[0].score, self.calibration_map.last().unwrap().score)
            },
            probability_range: if self.calibration_map.is_empty() {
                (0.0, 0.0)
            } else {
                let probs: Vec<f64> = self.calibration_map.iter().map(|p| p.probability).collect();
                (*probs.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
                 *probs.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())
            },
            is_monotonic: self.fit_metadata.is_monotonic,
            training_ece: self.fit_metadata.training_ece,
            total_training_samples: self.fit_metadata.total_samples,
        }
    }
}

/// Statistics about the isotonic calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationStats {
    pub num_calibration_points: usize,
    pub score_range: (f64, f64),
    pub probability_range: (f64, f64),
    pub is_monotonic: bool,
    pub training_ece: f64,
    pub total_training_samples: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_isotonic_calibrator_creation() {
        let calibrator = IsotonicCalibrator::new();
        assert_eq!(calibrator.calibration_map.len(), 0);
        assert_eq!(calibrator.fit_metadata.total_samples, 0);
    }
    
    #[test]
    fn test_isotonic_calibrator_fit() {
        let mut calibrator = IsotonicCalibrator::new();
        
        // Create test data with perfect calibration
        let scores = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let labels = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        
        let result = calibrator.fit(&scores, &labels);
        assert!(result.is_ok());
        assert!(calibrator.calibration_map.len() > 0);
    }
    
    #[test]
    fn test_isotonic_prediction() {
        let mut calibrator = IsotonicCalibrator::new();
        
        // Fit calibrator
        let scores = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let labels = vec![0.0, 0.0, 0.5, 1.0, 1.0];
        calibrator.fit(&scores, &labels).unwrap();
        
        // Test prediction
        let test_scores = vec![0.2, 0.6, 0.8];
        let predictions = calibrator.predict(&test_scores);
        assert!(predictions.is_ok());
        
        let preds = predictions.unwrap();
        assert_eq!(preds.len(), 3);
        
        // All predictions should be valid probabilities
        for &pred in &preds {
            assert!(pred >= 0.0 && pred <= 1.0);
        }
    }
    
    #[test]
    fn test_monotonicity_verification() {
        let calibrator = IsotonicCalibrator {
            calibration_map: vec![
                CalibrationPoint { score: 0.1, probability: 0.1, sample_count: 10 },
                CalibrationPoint { score: 0.5, probability: 0.5, sample_count: 10 },
                CalibrationPoint { score: 0.9, probability: 0.9, sample_count: 10 },
            ],
            fit_metadata: IsotonicFitMetadata {
                num_points: 3,
                total_samples: 30,
                training_ece: 0.01,
                is_monotonic: true,
                fit_timestamp: std::time::SystemTime::now(),
            },
        };
        
        assert!(calibrator.verify_monotonicity());
    }
}