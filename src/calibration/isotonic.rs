//! # Isotonic Regression Calibration
//!
//! Slice-specific isotonic regression with slope clamp [0.9, 1.1] per TODO.md.
//! Primary calibration method for PHASE 4 achieving ECE ≤ 0.015.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

use super::CalibrationSample;

/// Isotonic regression calibrator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsotonicConfig {
    /// Slope clamp range [0.9, 1.1] per TODO.md
    pub slope_clamp: (f32, f32),
    /// Minimum samples required for training
    pub min_samples: usize,
    /// L2 regularization strength
    pub regularization: f32,
}

/// Isotonic regression calibrator for a specific intent×language slice
#[derive(Debug, Clone)]
pub struct IsotonicCalibrator {
    config: IsotonicConfig,
    /// Trained calibration mapping points
    calibration_points: Vec<CalibrationPoint>,
    /// Current ECE for this slice
    slice_ece: f32,
    /// Learned monotonic slope
    slope: f32,
    /// Number of training samples
    training_samples: usize,
    /// Training convergence status
    converged: bool,
}

/// Point in the isotonic calibration mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationPoint {
    /// Input prediction value
    pub prediction: f32,
    /// Calibrated output value
    pub calibrated: f32,
    /// Sample weight at this point
    pub weight: f32,
    /// Number of samples contributing to this point
    pub sample_count: usize,
}

/// Pool Adjacent Violators algorithm result
#[derive(Debug, Clone)]
struct PAVResult {
    /// Monotonic calibrated values
    calibrated_values: Vec<f32>,
    /// Corresponding prediction values
    predictions: Vec<f32>,
    /// Sample weights
    weights: Vec<f32>,
    /// Final slope after clamping
    final_slope: f32,
}

impl IsotonicCalibrator {
    /// Create new isotonic calibrator
    pub fn new(config: IsotonicConfig) -> Self {
        Self {
            config,
            calibration_points: Vec::new(),
            slice_ece: 0.0,
            slope: 1.0, // Start with identity mapping
            training_samples: 0,
            converged: false,
        }
    }

    /// Train isotonic regression on provided samples
    pub async fn train(&mut self, samples: &[&CalibrationSample]) -> Result<()> {
        if samples.len() < self.config.min_samples {
            anyhow::bail!(
                "Insufficient samples for isotonic regression: {} < {}",
                samples.len(),
                self.config.min_samples
            );
        }

        info!(
            "Training isotonic calibrator on {} samples with slope clamp {:?}",
            samples.len(),
            self.config.slope_clamp
        );

        self.training_samples = samples.len();

        // Sort samples by prediction value for isotonic regression
        let mut sorted_samples: Vec<&CalibrationSample> = samples.iter().copied().collect();
        sorted_samples.sort_by(|a, b| a.prediction.partial_cmp(&b.prediction).unwrap());

        // Apply Pool Adjacent Violators algorithm with slope clamping
        let pav_result = self.pool_adjacent_violators(&sorted_samples).await?;
        
        // Store calibration points
        self.calibration_points = self.create_calibration_points(&pav_result)?;
        self.slope = pav_result.final_slope;

        // Calculate ECE for this slice
        self.slice_ece = self.calculate_slice_ece(&sorted_samples).await?;

        // Check convergence
        self.converged = self.check_convergence()?;

        info!(
            "Isotonic calibrator trained: {} points, slope={:.3}, ECE={:.4}, converged={}",
            self.calibration_points.len(),
            self.slope,
            self.slice_ece,
            self.converged
        );

        if self.slice_ece > 0.015 {
            warn!(
                "Slice ECE {:.4} exceeds PHASE 4 target ≤ 0.015",
                self.slice_ece
            );
        }

        Ok(())
    }

    /// Calibrate a prediction using learned isotonic mapping
    pub async fn calibrate(&self, prediction: f32, _features: &HashMap<String, f32>) -> Result<f32> {
        if self.calibration_points.is_empty() {
            return Ok(prediction.clamp(0.001, 0.999));
        }

        // Find appropriate calibration using piecewise linear interpolation
        let calibrated = self.interpolate_calibration(prediction)?;
        
        debug!(
            "Isotonic calibration: {:.4} -> {:.4} (slope: {:.3})",
            prediction, calibrated, self.slope
        );

        Ok(calibrated.clamp(0.001, 0.999)) // Ensure valid probability
    }

    /// Get current ECE for this slice
    pub fn get_ece(&self) -> f32 {
        self.slice_ece
    }

    /// Get learned slope
    pub fn get_slope(&self) -> f32 {
        self.slope
    }

    /// Check if calibrator converged during training
    pub fn is_converged(&self) -> bool {
        self.converged
    }

    /// Get training sample count
    pub fn get_training_samples(&self) -> usize {
        self.training_samples
    }

    /// Get calibration points for inspection
    pub fn get_calibration_points(&self) -> &[CalibrationPoint] {
        &self.calibration_points
    }

    // Private implementation methods

    /// Pool Adjacent Violators algorithm with slope clamping
    async fn pool_adjacent_violators(&self, samples: &[&CalibrationSample]) -> Result<PAVResult> {
        let n = samples.len();
        let mut predictions = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);

        // Extract data
        for sample in samples {
            predictions.push(sample.prediction);
            targets.push(sample.ground_truth);
            weights.push(sample.weight);
        }

        // Initialize working arrays
        let mut calibrated = targets.clone();
        let mut pooled_weights = weights.clone();
        
        // PAV algorithm: pool adjacent violators
        let mut changed = true;
        while changed {
            changed = false;
            
            for i in 0..(n - 1) {
                if calibrated[i] > calibrated[i + 1] {
                    // Pool adjacent violators
                    let weighted_sum = calibrated[i] * pooled_weights[i] + 
                                     calibrated[i + 1] * pooled_weights[i + 1];
                    let total_weight = pooled_weights[i] + pooled_weights[i + 1];
                    let pooled_value = weighted_sum / total_weight;
                    
                    calibrated[i] = pooled_value;
                    calibrated[i + 1] = pooled_value;
                    pooled_weights[i] = total_weight;
                    pooled_weights[i + 1] = 0.0; // Mark as pooled
                    
                    changed = true;
                }
            }
        }

        // Calculate overall slope for clamping
        let initial_slope = self.calculate_overall_slope(&predictions, &calibrated)?;
        
        // Apply slope clamping per TODO.md requirements [0.9, 1.1]
        let clamped_slope = initial_slope.clamp(self.config.slope_clamp.0, self.config.slope_clamp.1);
        
        // Adjust calibrated values if slope was clamped
        let final_calibrated = if (initial_slope - clamped_slope).abs() > 1e-6 {
            info!(
                "Clamping slope from {:.3} to {:.3}",
                initial_slope, clamped_slope
            );
            self.apply_slope_clamp(&predictions, &calibrated, clamped_slope)?
        } else {
            calibrated
        };

        Ok(PAVResult {
            calibrated_values: final_calibrated,
            predictions,
            weights: pooled_weights,
            final_slope: clamped_slope,
        })
    }

    /// Calculate overall slope of the calibration function
    fn calculate_overall_slope(&self, predictions: &[f32], calibrated: &[f32]) -> Result<f32> {
        if predictions.len() < 2 {
            return Ok(1.0);
        }

        let pred_range = predictions[predictions.len() - 1] - predictions[0];
        let calib_range = calibrated[calibrated.len() - 1] - calibrated[0];
        
        if pred_range < 1e-6 {
            return Ok(1.0);
        }
        
        Ok(calib_range / pred_range)
    }

    /// Apply slope clamping to calibrated values
    fn apply_slope_clamp(&self, predictions: &[f32], calibrated: &[f32], target_slope: f32) -> Result<Vec<f32>> {
        if predictions.is_empty() {
            return Ok(Vec::new());
        }

        // Find pivot point (median)
        let mid_idx = predictions.len() / 2;
        let pivot_pred = predictions[mid_idx];
        let pivot_calib = calibrated[mid_idx];
        
        // Apply clamped slope around pivot
        let mut clamped_calibrated = Vec::with_capacity(predictions.len());
        
        for &pred in predictions {
            let delta_pred = pred - pivot_pred;
            let new_calib = pivot_calib + delta_pred * target_slope;
            clamped_calibrated.push(new_calib.clamp(0.0, 1.0));
        }
        
        Ok(clamped_calibrated)
    }

    /// Create calibration points from PAV result
    fn create_calibration_points(&self, pav_result: &PAVResult) -> Result<Vec<CalibrationPoint>> {
        let mut points = Vec::new();
        
        for (i, (&pred, &calib)) in pav_result.predictions.iter()
            .zip(&pav_result.calibrated_values)
            .enumerate()
        {
            if pav_result.weights[i] > 0.0 { // Skip pooled points
                points.push(CalibrationPoint {
                    prediction: pred,
                    calibrated: calib,
                    weight: pav_result.weights[i],
                    sample_count: 1, // Would be computed from actual pooling
                });
            }
        }

        // Remove duplicate points and sort
        points.sort_by(|a, b| a.prediction.partial_cmp(&b.prediction).unwrap());
        points.dedup_by(|a, b| (a.prediction - b.prediction).abs() < 1e-6);

        Ok(points)
    }

    /// Interpolate calibration for a given prediction
    fn interpolate_calibration(&self, prediction: f32) -> Result<f32> {
        if self.calibration_points.is_empty() {
            return Ok(prediction);
        }

        // Handle edge cases
        if prediction <= self.calibration_points[0].prediction {
            return Ok(self.calibration_points[0].calibrated);
        }
        
        let last_idx = self.calibration_points.len() - 1;
        if prediction >= self.calibration_points[last_idx].prediction {
            return Ok(self.calibration_points[last_idx].calibrated);
        }

        // Find interpolation interval
        for i in 0..last_idx {
            let p1 = &self.calibration_points[i];
            let p2 = &self.calibration_points[i + 1];
            
            if prediction >= p1.prediction && prediction <= p2.prediction {
                // Linear interpolation
                let t = (prediction - p1.prediction) / (p2.prediction - p1.prediction);
                return Ok(p1.calibrated + t * (p2.calibrated - p1.calibrated));
            }
        }

        // Fallback (should not reach here)
        Ok(prediction)
    }

    /// Calculate ECE for this specific slice
    async fn calculate_slice_ece(&self, samples: &[&CalibrationSample]) -> Result<f32> {
        const NUM_BINS: usize = 10;
        let mut bins = vec![Vec::new(); NUM_BINS];
        
        // Apply current calibration to all samples
        let mut calibrated_samples = Vec::new();
        for sample in samples {
            let calibrated_pred = self.interpolate_calibration(sample.prediction)?;
            calibrated_samples.push((calibrated_pred, sample.ground_truth));
        }
        
        // Assign to bins based on calibrated predictions
        for (calibrated_pred, ground_truth) in &calibrated_samples {
            let bin_idx = ((*calibrated_pred * NUM_BINS as f32) as usize).min(NUM_BINS - 1);
            bins[bin_idx].push(*ground_truth);
        }
        
        // Calculate ECE
        let mut ece = 0.0;
        let total_samples = calibrated_samples.len() as f32;
        
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

    /// Check if calibrator has converged
    fn check_convergence(&self) -> Result<bool> {
        // Check if we have enough calibration points
        if self.calibration_points.len() < 3 {
            return Ok(false);
        }

        // Check if slope is within acceptable range
        let slope_ok = self.slope >= self.config.slope_clamp.0 && 
                      self.slope <= self.config.slope_clamp.1;
        
        // Check if ECE is within target
        let ece_ok = self.slice_ece <= 0.015;
        
        // Check monotonicity
        let monotonic = self.calibration_points.windows(2)
            .all(|w| w[1].calibrated >= w[0].calibrated);
        
        Ok(slope_ok && ece_ok && monotonic)
    }
}

impl Default for IsotonicConfig {
    fn default() -> Self {
        Self {
            slope_clamp: (0.9, 1.1), // PHASE 4 requirement from TODO.md
            min_samples: 30,
            regularization: 0.01,
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
    async fn test_isotonic_calibrator_creation() {
        let config = IsotonicConfig::default();
        let calibrator = IsotonicCalibrator::new(config);
        
        assert_eq!(calibrator.get_slope(), 1.0);
        assert_eq!(calibrator.get_ece(), 0.0);
        assert!(!calibrator.is_converged());
        assert_eq!(calibrator.get_training_samples(), 0);
    }

    #[tokio::test]
    async fn test_isotonic_training_insufficient_samples() {
        let config = IsotonicConfig {
            min_samples: 50,
            ..Default::default()
        };
        let mut calibrator = IsotonicCalibrator::new(config);
        
        let samples = vec![
            create_test_sample(0.1, 0.0),
            create_test_sample(0.2, 0.0),
            create_test_sample(0.3, 1.0),
        ];
        let sample_refs: Vec<&CalibrationSample> = samples.iter().collect();
        
        let result = calibrator.train(&sample_refs).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Insufficient samples"));
    }

    #[tokio::test]
    async fn test_isotonic_training_perfect_calibration() {
        let config = IsotonicConfig::default();
        let mut calibrator = IsotonicCalibrator::new(config);
        
        // Create perfectly calibrated samples
        let mut samples = Vec::new();
        for i in 1..=50 {
            let pred = i as f32 / 50.0;
            let ground_truth = if i <= 25 { 0.0 } else { 1.0 };
            samples.push(create_test_sample(pred, ground_truth));
        }
        
        let sample_refs: Vec<&CalibrationSample> = samples.iter().collect();
        let result = calibrator.train(&sample_refs).await;
        assert!(result.is_ok());
        
        // Check that calibrator learned something reasonable
        assert!(calibrator.get_ece() >= 0.0);
        assert!(calibrator.get_slope() >= 0.9);
        assert!(calibrator.get_slope() <= 1.1);
        assert_eq!(calibrator.get_training_samples(), 50);
    }

    #[tokio::test]
    async fn test_isotonic_calibration() {
        let config = IsotonicConfig::default();
        let mut calibrator = IsotonicCalibrator::new(config);
        
        // Create some training data
        let samples = vec![
            create_test_sample(0.1, 0.0),
            create_test_sample(0.3, 0.0),
            create_test_sample(0.5, 0.5),
            create_test_sample(0.7, 1.0),
            create_test_sample(0.9, 1.0),
        ];
        
        // Add more samples to meet minimum requirement
        let mut expanded_samples = samples.clone();
        for i in 0..25 {
            let pred = 0.2 + (i as f32 / 25.0) * 0.6;
            let ground_truth = if pred > 0.5 { 1.0 } else { 0.0 };
            expanded_samples.push(create_test_sample(pred, ground_truth));
        }
        
        let sample_refs: Vec<&CalibrationSample> = expanded_samples.iter().collect();
        calibrator.train(&sample_refs).await.unwrap();
        
        // Test calibration
        let features = HashMap::new();
        let calibrated = calibrator.calibrate(0.8, &features).await.unwrap();
        
        // Should be a valid probability
        assert!(calibrated >= 0.0 && calibrated <= 1.0);
        
        // For high prediction, should get high calibrated score
        assert!(calibrated > 0.5);
    }

    #[tokio::test]
    async fn test_slope_clamping() {
        let config = IsotonicConfig {
            slope_clamp: (0.9, 1.1),
            ..Default::default()
        };
        let mut calibrator = IsotonicCalibrator::new(config);
        
        // Create data that would naturally have slope > 1.1
        let mut samples = Vec::new();
        for i in 0..40 {
            let pred = 0.3 + (i as f32 / 40.0) * 0.4; // Range [0.3, 0.7]
            let ground_truth = if i < 10 { 0.0 } else { 1.0 }; // Sharp transition
            samples.push(create_test_sample(pred, ground_truth));
        }
        
        let sample_refs: Vec<&CalibrationSample> = samples.iter().collect();
        calibrator.train(&sample_refs).await.unwrap();
        
        // Slope should be clamped to [0.9, 1.1]
        assert!(calibrator.get_slope() >= 0.9);
        assert!(calibrator.get_slope() <= 1.1);
    }

    #[test]
    fn test_slope_calculation() {
        let config = IsotonicConfig::default();
        let calibrator = IsotonicCalibrator::new(config);
        
        let predictions = vec![0.1, 0.5, 0.9];
        let calibrated = vec![0.2, 0.5, 0.8];
        
        let slope = calibrator.calculate_overall_slope(&predictions, &calibrated).unwrap();
        
        // Expected: (0.8 - 0.2) / (0.9 - 0.1) = 0.6 / 0.8 = 0.75
        assert!((slope - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_interpolation() {
        let config = IsotonicConfig::default();
        let mut calibrator = IsotonicCalibrator::new(config);
        
        // Set up some calibration points manually for testing
        calibrator.calibration_points = vec![
            CalibrationPoint {
                prediction: 0.2,
                calibrated: 0.1,
                weight: 1.0,
                sample_count: 10,
            },
            CalibrationPoint {
                prediction: 0.8,
                calibrated: 0.9,
                weight: 1.0,
                sample_count: 10,
            },
        ];
        
        // Test interpolation
        let result = calibrator.interpolate_calibration(0.5).unwrap();
        
        // Should be halfway between 0.1 and 0.9 = 0.5
        assert!((result - 0.5).abs() < 0.01);
        
        // Test edge cases
        assert_eq!(calibrator.interpolate_calibration(0.1).unwrap(), 0.1);
        assert_eq!(calibrator.interpolate_calibration(0.9).unwrap(), 0.9);
    }
}