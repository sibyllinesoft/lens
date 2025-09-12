// WASM/TypeScript Parity Enforcement System
// Ensures perfect cross-language consistency with strict tolerance validation

use crate::calibration::isotonic::IsotonicCalibrator;
use crate::calibration::platt::PlattScaler;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

/// Strict parity tolerances for cross-language validation
pub const PREDICTION_TOLERANCE: f64 = 1e-6;
pub const ECE_TOLERANCE: f64 = 1e-4;
pub const BIN_COUNT_CONSISTENCY_MIN: usize = 10;

/// Parity test result with detailed diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityTestResult {
    pub passed: bool,
    pub prediction_max_diff: f64,
    pub ece_diff: f64,
    pub bin_count_consistent: bool,
    pub test_case_count: usize,
    pub failed_cases: Vec<ParityFailure>,
}

/// Details of a parity test failure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityFailure {
    pub test_index: usize,
    pub rust_prediction: f64,
    pub ts_prediction: f64,
    pub prediction_diff: f64,
    pub rust_ece: f64,
    pub ts_ece: f64,
    pub ece_diff: f64,
}

/// WASM binning core interface for TypeScript integration
#[wasm_bindgen]
pub struct WasmBinningCore {
    isotonic: IsotonicCalibrator,
    platt: PlattScaler,
}

#[wasm_bindgen]
impl WasmBinningCore {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        use crate::calibration::isotonic::IsotonicConfig;
        use crate::calibration::platt::PlattConfig;
        
        let isotonic_config = IsotonicConfig {
            slope_clamp: (0.9, 1.1),
            min_samples: 30,
            regularization: 0.01,
            input_hygiene: true,
            equal_mass_bins: true,
            ece_bins: 15,
            convex_mixing: 0.15,
        };
        
        let platt_config = PlattConfig {
            max_iterations: 100,
            convergence_tolerance: 1e-6,
        };
        
        Self {
            isotonic: IsotonicCalibrator::new(isotonic_config),
            platt: PlattScaler::new(platt_config),
        }
    }

    /// Calibrate predictions using isotonic regression
    #[wasm_bindgen]
    pub fn isotonic_calibrate(&mut self, predictions: &[f64], labels: &[f64]) -> Vec<f64> {
        // Convert to CalibrationSample format for training
        let samples: Vec<crate::calibration::CalibrationSample> = predictions.iter()
            .zip(labels.iter())
            .enumerate()
            .map(|(i, (&pred, &label))| crate::calibration::CalibrationSample {
                prediction: pred as f32,
                ground_truth: label as f32,
                intent: format!("wasm_test_{}", i),
                language: Some("typescript".to_string()),
                features: std::collections::HashMap::new(),
                weight: 1.0,
            })
            .collect();
        
        // Train the calibrator (simplified synchronous version for WASM)
        let sample_refs: Vec<&crate::calibration::CalibrationSample> = samples.iter().collect();
        let _ = futures::executor::block_on(self.isotonic.train(&sample_refs));
        
        // Calibrate predictions
        predictions.iter().map(|&pred| {
            let features = std::collections::HashMap::new();
            futures::executor::block_on(self.isotonic.calibrate(pred as f32, &features))
                .unwrap_or(pred as f32) as f64
        }).collect()
    }

    /// Calibrate predictions using Platt scaling
    #[wasm_bindgen]
    pub fn platt_calibrate(&mut self, predictions: &[f64], labels: &[f64]) -> Vec<f64> {
        // Convert to CalibrationSample format for training
        let samples: Vec<crate::calibration::CalibrationSample> = predictions.iter()
            .zip(labels.iter())
            .enumerate()
            .map(|(i, (&pred, &label))| crate::calibration::CalibrationSample {
                prediction: pred as f32,
                ground_truth: label as f32,
                intent: format!("wasm_test_{}", i),
                language: Some("typescript".to_string()),
                features: std::collections::HashMap::new(),
                weight: 1.0,
            })
            .collect();
        
        // Train the calibrator (simplified synchronous version for WASM)
        let sample_refs: Vec<&crate::calibration::CalibrationSample> = samples.iter().collect();
        let _ = futures::executor::block_on(self.platt.train(&sample_refs));
        
        // Calibrate predictions
        predictions.iter().map(|&pred| {
            let features = std::collections::HashMap::new();
            futures::executor::block_on(self.platt.calibrate(pred as f32, &features))
                .unwrap_or(pred as f32) as f64
        }).collect()
    }

    /// Calculate ECE with specified bin count
    #[wasm_bindgen]
    pub fn calculate_ece(&self, predictions: &[f64], labels: &[f64], bins: usize) -> f64 {
        calculate_expected_calibration_error(predictions, labels, bins)
    }

    /// Get calibration bin boundaries
    #[wasm_bindgen]
    pub fn get_bin_boundaries(&self, bins: usize) -> Vec<f64> {
        (0..=bins).map(|i| i as f64 / bins as f64).collect()
    }
}

/// Comprehensive parity test suite runner
pub struct ParityTestSuite {
    test_cases: Vec<ParityTestCase>,
}

#[derive(Debug, Clone)]
pub struct ParityTestCase {
    pub predictions: Vec<f64>,
    pub labels: Vec<f64>,
    pub bins: usize,
    pub calibrator_type: CalibratorType,
}

#[derive(Debug, Clone)]
pub enum CalibratorType {
    Isotonic,
    Platt,
}

impl ParityTestSuite {
    /// Create 1000-tuple parity suite for comprehensive validation
    pub fn create_comprehensive_suite() -> Self {
        let mut test_cases = Vec::new();
        let _rng = rand::thread_rng();
        
        // Generate diverse test cases
        for i in 0..1000 {
            let size = 50 + (i % 200); // Varying dataset sizes
            let predictions = generate_test_predictions(size, i as u64);
            let labels = generate_test_labels(size, i as u64);
            let bins = 10 + (i % 15); // Varying bin counts
            let calibrator_type = if i % 2 == 0 { 
                CalibratorType::Isotonic 
            } else { 
                CalibratorType::Platt 
            };

            test_cases.push(ParityTestCase {
                predictions,
                labels,
                bins,
                calibrator_type,
            });
        }

        Self { test_cases }
    }

    /// Run comprehensive parity validation against TypeScript implementation
    pub fn validate_cross_language_parity(&self, ts_results: &[TypeScriptResult]) -> ParityTestResult {
        let mut max_prediction_diff: f64 = 0.0;
        let mut max_ece_diff: f64 = 0.0;
        let mut failed_cases = Vec::new();
        let mut bin_count_consistent = true;

        for (i, test_case) in self.test_cases.iter().enumerate() {
            let rust_result = self.run_rust_calibration(test_case);
            
            if i >= ts_results.len() {
                failed_cases.push(ParityFailure {
                    test_index: i,
                    rust_prediction: 0.0,
                    ts_prediction: 0.0,
                    prediction_diff: f64::INFINITY,
                    rust_ece: 0.0,
                    ts_ece: 0.0,
                    ece_diff: f64::INFINITY,
                });
                continue;
            }

            let ts_result = &ts_results[i];
            
            // Validate prediction parity
            let prediction_diff = calculate_max_difference(&rust_result.predictions, &ts_result.predictions);
            max_prediction_diff = max_prediction_diff.max(prediction_diff);

            // Validate ECE parity
            let ece_diff = (rust_result.ece - ts_result.ece).abs();
            max_ece_diff = max_ece_diff.max(ece_diff);

            // Validate bin count consistency
            if rust_result.bin_count != ts_result.bin_count {
                bin_count_consistent = false;
            }

            // Record failures
            if prediction_diff > PREDICTION_TOLERANCE || ece_diff > ECE_TOLERANCE {
                failed_cases.push(ParityFailure {
                    test_index: i,
                    rust_prediction: rust_result.predictions.get(0).copied().unwrap_or(0.0),
                    ts_prediction: ts_result.predictions.get(0).copied().unwrap_or(0.0),
                    prediction_diff,
                    rust_ece: rust_result.ece,
                    ts_ece: ts_result.ece,
                    ece_diff,
                });
            }
        }

        let passed = max_prediction_diff <= PREDICTION_TOLERANCE && 
                    max_ece_diff <= ECE_TOLERANCE && 
                    bin_count_consistent &&
                    failed_cases.is_empty();

        ParityTestResult {
            passed,
            prediction_max_diff: max_prediction_diff,
            ece_diff: max_ece_diff,
            bin_count_consistent,
            test_case_count: self.test_cases.len(),
            failed_cases,
        }
    }

    /// Run Rust calibration for a test case
    fn run_rust_calibration(&self, test_case: &ParityTestCase) -> RustCalibrationResult {
        use crate::calibration::isotonic::{IsotonicCalibrator, IsotonicConfig};
        use crate::calibration::platt::{PlattScaler, PlattConfig};
        
        match test_case.calibrator_type {
            CalibratorType::Isotonic => {
                let config = IsotonicConfig {
                    slope_clamp: (0.9, 1.1),
                    min_samples: 30,
                    regularization: 0.01,
                    input_hygiene: true,
                    equal_mass_bins: true,
                    ece_bins: 15,
                    convex_mixing: 0.15,
                };
                
                let mut calibrator = IsotonicCalibrator::new(config);
                
                // Convert test case data to CalibrationSamples
                let samples: Vec<crate::calibration::CalibrationSample> = test_case.predictions.iter()
                    .zip(test_case.labels.iter())
                    .enumerate()
                    .map(|(i, (&pred, &label))| crate::calibration::CalibrationSample {
                        prediction: pred as f32,
                        ground_truth: label as f32,
                        intent: format!("test_{}", i),
                        language: Some("rust".to_string()),
                        features: HashMap::new(),
                        weight: 1.0,
                    })
                    .collect();
                
                let sample_refs: Vec<&crate::calibration::CalibrationSample> = samples.iter().collect();
                let _ = futures::executor::block_on(calibrator.train(&sample_refs));
                
                // Get calibrated predictions
                let predictions: Vec<f64> = test_case.predictions.iter().map(|&pred| {
                    let features = HashMap::new();
                    futures::executor::block_on(calibrator.calibrate(pred as f32, &features))
                        .unwrap_or(pred as f32) as f64
                }).collect();
                
                let ece = calculate_expected_calibration_error(&predictions, &test_case.labels, test_case.bins);
                
                RustCalibrationResult {
                    predictions,
                    ece,
                    bin_count: test_case.bins,
                }
            }
            CalibratorType::Platt => {
                let config = PlattConfig {
                    max_iterations: 100,
                    convergence_tolerance: 1e-6,
                };
                
                let mut calibrator = PlattScaler::new(config);
                
                // Convert test case data to CalibrationSamples
                let samples: Vec<crate::calibration::CalibrationSample> = test_case.predictions.iter()
                    .zip(test_case.labels.iter())
                    .enumerate()
                    .map(|(i, (&pred, &label))| crate::calibration::CalibrationSample {
                        prediction: pred as f32,
                        ground_truth: label as f32,
                        intent: format!("test_{}", i),
                        language: Some("rust".to_string()),
                        features: HashMap::new(),
                        weight: 1.0,
                    })
                    .collect();
                
                let sample_refs: Vec<&crate::calibration::CalibrationSample> = samples.iter().collect();
                let _ = futures::executor::block_on(calibrator.train(&sample_refs));
                
                // Get calibrated predictions
                let predictions: Vec<f64> = test_case.predictions.iter().map(|&pred| {
                    let features = HashMap::new();
                    futures::executor::block_on(calibrator.calibrate(pred as f32, &features))
                        .unwrap_or(pred as f32) as f64
                }).collect();
                
                let ece = calculate_expected_calibration_error(&predictions, &test_case.labels, test_case.bins);
                
                RustCalibrationResult {
                    predictions,
                    ece,
                    bin_count: test_case.bins,
                }
            }
        }
    }
}

/// Rust calibration result
#[derive(Debug, Clone)]
struct RustCalibrationResult {
    predictions: Vec<f64>,
    ece: f64,
    bin_count: usize,
}

/// TypeScript result structure (for comparison)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeScriptResult {
    pub predictions: Vec<f64>,
    pub ece: f64,
    pub bin_count: usize,
}

/// CI enforcement utilities
pub struct CiParityEnforcement;

impl CiParityEnforcement {
    /// Enforce parity requirements in CI pipeline
    pub fn enforce_ci_requirements(result: &ParityTestResult) -> Result<(), String> {
        if !result.passed {
            return Err(format!(
                "PARITY ENFORCEMENT FAILED: Max prediction diff: {:.2e} (limit: {:.2e}), ECE diff: {:.2e} (limit: {:.2e}), Failed cases: {}",
                result.prediction_max_diff,
                PREDICTION_TOLERANCE,
                result.ece_diff,
                ECE_TOLERANCE,
                result.failed_cases.len()
            ));
        }

        if !result.bin_count_consistent {
            return Err("PARITY ENFORCEMENT FAILED: Bin count inconsistency detected".to_string());
        }

        if result.test_case_count < 1000 {
            return Err(format!(
                "PARITY ENFORCEMENT FAILED: Insufficient test coverage: {} cases (minimum: 1000)",
                result.test_case_count
            ));
        }

        Ok(())
    }

    /// Generate CI report for parity validation
    pub fn generate_ci_report(result: &ParityTestResult) -> String {
        let status = if result.passed { "PASSED" } else { "FAILED" };
        
        format!(
            r#"
# WASM/TypeScript Parity Validation Report

## Status: {}

### Metrics
- Test Cases: {}
- Prediction Max Diff: {:.2e} (tolerance: {:.2e})
- ECE Max Diff: {:.2e} (tolerance: {:.2e})
- Bin Count Consistent: {}
- Failed Cases: {}

### Failed Cases Details
{}

### Enforcement Result
{}
"#,
            status,
            result.test_case_count,
            result.prediction_max_diff,
            PREDICTION_TOLERANCE,
            result.ece_diff,
            ECE_TOLERANCE,
            result.bin_count_consistent,
            result.failed_cases.len(),
            format_failed_cases(&result.failed_cases),
            if result.passed { "✅ All parity requirements satisfied" } else { "❌ Parity requirements violated - blocking release" }
        )
    }
}

/// Helper functions

fn generate_test_predictions(size: usize, seed: u64) -> Vec<f64> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..size).map(|_| rng.gen::<f64>()).collect()
}

fn generate_test_labels(size: usize, seed: u64) -> Vec<f64> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed + 1000);
    (0..size).map(|_| if rng.gen::<f64>() > 0.5 { 1.0 } else { 0.0 }).collect()
}

fn calculate_max_difference(rust_preds: &[f64], ts_preds: &[f64]) -> f64 {
    if rust_preds.len() != ts_preds.len() {
        return f64::INFINITY;
    }
    
    rust_preds.iter()
        .zip(ts_preds.iter())
        .map(|(r, t)| (r - t).abs())
        .fold(0.0, f64::max)
}

fn calculate_expected_calibration_error(predictions: &[f64], labels: &[f64], bins: usize) -> f64 {
    let mut bin_boundaries = Vec::new();
    for i in 0..=bins {
        bin_boundaries.push(i as f64 / bins as f64);
    }

    let mut bin_accuracies = vec![0.0; bins];
    let mut bin_confidences = vec![0.0; bins];
    let mut bin_counts = vec![0; bins];

    for (pred, label) in predictions.iter().zip(labels.iter()) {
        let bin_idx = (pred * bins as f64).floor() as usize;
        let bin_idx = bin_idx.min(bins - 1);

        bin_accuracies[bin_idx] += label;
        bin_confidences[bin_idx] += pred;
        bin_counts[bin_idx] += 1;
    }

    let total_samples = predictions.len() as f64;
    let mut ece = 0.0;

    for i in 0..bins {
        if bin_counts[i] > 0 {
            let bin_accuracy = bin_accuracies[i] / bin_counts[i] as f64;
            let bin_confidence = bin_confidences[i] / bin_counts[i] as f64;
            let bin_weight = bin_counts[i] as f64 / total_samples;
            
            ece += bin_weight * (bin_accuracy - bin_confidence).abs();
        }
    }

    ece
}

fn format_failed_cases(failed_cases: &[ParityFailure]) -> String {
    if failed_cases.is_empty() {
        return "None".to_string();
    }

    failed_cases.iter()
        .take(10) // Limit to first 10 for readability
        .enumerate()
        .map(|(i, failure)| {
            format!(
                "{}. Test {}: pred_diff={:.2e}, ece_diff={:.2e}",
                i + 1,
                failure.test_index,
                failure.prediction_diff,
                failure.ece_diff
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parity_suite_creation() {
        let suite = ParityTestSuite::create_comprehensive_suite();
        assert_eq!(suite.test_cases.len(), 1000);
    }

    #[test]
    fn test_wasm_binning_core() {
        let mut core = WasmBinningCore::new();
        let predictions = vec![0.1, 0.4, 0.6, 0.9];
        let labels = vec![0.0, 0.0, 1.0, 1.0];
        
        let calibrated = core.isotonic_calibrate(&predictions, &labels);
        assert_eq!(calibrated.len(), predictions.len());
        
        let ece = core.calculate_ece(&calibrated, &labels, 10);
        assert!(ece >= 0.0);
    }

    #[test]
    fn test_ci_enforcement() {
        let passing_result = ParityTestResult {
            passed: true,
            prediction_max_diff: 1e-7,
            ece_diff: 1e-5,
            bin_count_consistent: true,
            test_case_count: 1000,
            failed_cases: vec![],
        };

        assert!(CiParityEnforcement::enforce_ci_requirements(&passing_result).is_ok());

        let failing_result = ParityTestResult {
            passed: false,
            prediction_max_diff: 1e-5,
            ece_diff: 1e-3,
            bin_count_consistent: false,
            test_case_count: 500,
            failed_cases: vec![],
        };

        assert!(CiParityEnforcement::enforce_ci_requirements(&failing_result).is_err());
    }
}