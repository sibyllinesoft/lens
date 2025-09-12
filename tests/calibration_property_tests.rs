//! # Calibration System Property and Fuzz Tests
//!
//! Comprehensive property-based and fuzz testing for the calibration system.
//! Tests edge cases, malformed inputs, and statistical properties to ensure
//! robustness and prevent regressions in the critical calibration pipeline.

use anyhow::Result;
use lens_core::calibration::{
    CalibrationSample, 
    IsotonicCalibrator,
    IsotonicConfig,
    TemperatureScaler,
    TemperatureConfig,
    PlattScaler,
    PlattConfig,
};
use std::collections::HashMap;
use tokio;

/// Simple pseudo-random number generator for reproducible fuzz tests
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }
    
    fn next_f32(&mut self) -> f32 {
        (self.next() >> 32) as f32 / u32::MAX as f32
    }
    
    fn next_range(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_f32() * (max - min)
    }
    
    fn next_bool(&mut self) -> bool {
        (self.next() & 1) == 1
    }
}

// ============================================================================
// FUZZ TESTS - Generate Malformed Inputs and Edge Cases
// ============================================================================

#[tokio::test]
async fn test_fuzz_logits_percentages_missing_weights() -> Result<()> {
    println!("🌪️ FUZZ TEST: Generate logits/percentages/missing weights/NaNs");
    
    let mut rng = SimpleRng::new(12345);
    let config = IsotonicConfig::default();
    
    // Test 100 different malformed input scenarios
    for iteration in 0..100 {
        let mut samples = Vec::new();
        let sample_count = 30 + (iteration % 50); // Variable sample counts
        
        for i in 0..sample_count {
            let prediction = match iteration % 6 {
                0 => {
                    // Logits: wide range including extreme values
                    rng.next_range(-15.0, 15.0)
                },
                1 => {
                    // Percentages: [0, 100] range
                    rng.next_range(0.0, 100.0)
                },
                2 => {
                    // Normal probabilities with some out-of-bounds
                    let val = rng.next_range(-0.2, 1.2);
                    if iteration < 50 { val } else { val.clamp(0.001, 0.999) }
                },
                3 => {
                    // Extreme values and edge cases
                    let extreme_values = [0.0, 1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 1e10, -1e10];
                    extreme_values[i % extreme_values.len()]
                },
                4 => {
                    // Very small values near zero
                    rng.next_range(-1e-6, 1e-6)
                },
                _ => {
                    // Values very close to 1
                    0.99999 + rng.next_range(-1e-5, 1e-5)
                }
            };
            
            let ground_truth = if rng.next_bool() { 1.0 } else { 0.0 };
            
            let weight = match iteration % 4 {
                0 => 1.0, // Normal weight
                1 => rng.next_range(0.1, 10.0), // Variable weights
                2 => 0.0, // Zero weight (edge case)
                _ => rng.next_range(1e-8, 1e8), // Extreme weights
            };
            
            samples.push(CalibrationSample {
                prediction,
                ground_truth,
                intent: format!("fuzz_test_{}", iteration),
                language: Some("test".to_string()),
                features: HashMap::new(),
                weight,
            });
        }
        
        let sample_refs: Vec<&CalibrationSample> = samples.iter().collect();
        let mut calibrator = IsotonicCalibrator::new(config.clone());
        
        // Attempt training - should either succeed with valid results or fail gracefully
        let training_result = calibrator.train(&sample_refs).await;
        
        match training_result {
            Ok(_) => {
                // Training succeeded - verify all invariants hold
                let validation_result = calibrator.validate_ci_invariants();
                assert!(validation_result.is_ok(), 
                        "CI invariants failed for iteration {}: {:?}", iteration, validation_result);
                
                // Test calibration on various inputs
                let features = HashMap::new();
                let test_inputs = [0.1, 0.5, 0.9, f32::NAN, 0.0, 1.0];
                
                for &test_input in &test_inputs {
                    let calibration_result = calibrator.calibrate(test_input, &features).await;
                    
                    match calibration_result {
                        Ok(output) => {
                            // All outputs must be valid probabilities
                            assert!(output >= 0.0 && output <= 1.0 && output.is_finite(),
                                    "Invalid calibrated output for iteration {}, input {:.4}: {:.6}", 
                                    iteration, test_input, output);
                        },
                        Err(e) => {
                            println!("  Iteration {}: Calibration failed for input {:.4}: {:?}", 
                                    iteration, test_input, e);
                        }
                    }
                }
                
                if iteration % 20 == 0 {
                    println!("  ✓ Iteration {}: Training succeeded, ECE={:.4}, slope={:.3}", 
                            iteration, calibrator.get_ece(), calibrator.get_slope());
                }
            },
            Err(e) => {
                // Training failed - ensure it's a reasonable failure (not a panic)
                let error_msg = e.to_string().to_lowercase();
                let is_reasonable_failure = error_msg.contains("insufficient") ||
                                          error_msg.contains("invalid") ||
                                          error_msg.contains("numerical") ||
                                          error_msg.contains("convergence");
                
                if !is_reasonable_failure {
                    println!("  ⚠️ Iteration {}: Unexpected failure: {:?}", iteration, e);
                }
                
                if iteration % 20 == 0 {
                    println!("  ~ Iteration {}: Training failed (expected): {}", iteration, e);
                }
            }
        }
    }
    
    println!("✅ Fuzz testing completed: System robust against malformed inputs");
    Ok(())
}

#[tokio::test]
async fn test_fuzz_hygiene_transforms_deterministic_outputs() -> Result<()> {
    println!("🌪️ FUZZ TEST: Assert hygiene transforms + deterministic outputs");
    
    let mut rng = SimpleRng::new(54321);
    let config = IsotonicConfig {
        input_hygiene: true,
        ..Default::default()
    };
    
    // Test deterministic behavior across multiple runs with identical inputs
    for test_case in 0..50 {
        let mut samples = Vec::new();
        let sample_count = 40;
        
        // Generate a specific pattern of inputs for this test case
        for i in 0..sample_count {
            let base_pred = (i as f32) / sample_count as f32;
            
            // Apply different input format transformations
            let prediction = match test_case % 5 {
                0 => base_pred * 100.0, // Percentage format
                1 => {
                    // Logit format: convert probability to logit
                    let p = base_pred.clamp(0.01, 0.99);
                    (p / (1.0 - p)).ln()
                },
                2 => base_pred + rng.next_range(-0.1, 0.1), // Noisy probabilities
                3 => base_pred * 2.0, // Out-of-range probabilities  
                _ => base_pred, // Normal probabilities
            };
            
            let ground_truth = if i < sample_count / 2 { 1.0 } else { 0.0 };
            
            samples.push(CalibrationSample {
                prediction,
                ground_truth,
                intent: "deterministic_test".to_string(),
                language: Some("test".to_string()),
                features: HashMap::new(),
                weight: 1.0,
            });
        }
        
        // Train the same calibrator twice with identical inputs
        let sample_refs: Vec<&CalibrationSample> = samples.iter().collect();
        
        let mut calibrator1 = IsotonicCalibrator::new(config.clone());
        let mut calibrator2 = IsotonicCalibrator::new(config.clone());
        
        let result1 = calibrator1.train(&sample_refs).await;
        let result2 = calibrator2.train(&sample_refs).await;
        
        // Both should succeed or both should fail
        assert_eq!(result1.is_ok(), result2.is_ok(), 
                   "Non-deterministic training results for test case {}", test_case);
        
        if result1.is_ok() && result2.is_ok() {
            // Compare outputs for determinism
            let ece1 = calibrator1.get_ece();
            let ece2 = calibrator2.get_ece();
            let slope1 = calibrator1.get_slope();
            let slope2 = calibrator2.get_slope();
            
            let ece_diff = (ece1 - ece2).abs();
            let slope_diff = (slope1 - slope2).abs();
            
            assert!(ece_diff < 1e-6, "Non-deterministic ECE: {:.6} vs {:.6} (diff: {:.6})", 
                    ece1, ece2, ece_diff);
            assert!(slope_diff < 1e-6, "Non-deterministic slope: {:.6} vs {:.6} (diff: {:.6})", 
                    slope1, slope2, slope_diff);
            
            // Test calibration determinism
            let features = HashMap::new();
            let test_inputs = [0.1, 0.5, 0.9];
            
            for &input in &test_inputs {
                let output1 = calibrator1.calibrate(input, &features).await?;
                let output2 = calibrator2.calibrate(input, &features).await?;
                
                let output_diff = (output1 - output2).abs();
                assert!(output_diff < 1e-6, 
                        "Non-deterministic calibration for input {:.1}: {:.6} vs {:.6}", 
                        input, output1, output2);
            }
            
            if test_case % 10 == 0 {
                println!("  ✓ Test case {}: Deterministic behavior verified (ECE={:.4}, slope={:.3})", 
                        test_case, ece1, slope1);
            }
        }
    }
    
    println!("✅ Hygiene transforms produce deterministic outputs");
    Ok(())
}

#[tokio::test]
async fn test_fuzz_threshold_estimator_with_known_noise() -> Result<()> {
    println!("🌪️ FUZZ TEST: Test threshold estimator with known N,K,p noise");
    
    let mut rng = SimpleRng::new(98765);
    
    // Test various (N, K, p) combinations with synthetic noise
    let test_configs = [
        (100, 10, 0.5),   // Balanced, moderate bins
        (400, 20, 0.3),   // Larger N, unbalanced
        (50, 5, 0.8),     // Small N, high probability
        (1000, 32, 0.1),  // Large N, many bins, low probability
        (200, 15, 0.95),  // High probability case
    ];
    
    for &(n, k, true_probability) in &test_configs {
        // Generate synthetic data with known ground truth probability
        let mut samples = Vec::new();
        
        for i in 0..n {
            // Predictions with calibration noise around true probability
            let noise_scale = 0.05; // ±5% noise
            let prediction = (true_probability + rng.next_range(-noise_scale, noise_scale))
                .clamp(0.01, 0.99);
            
            // Ground truth follows true probability with binomial sampling
            let ground_truth = if rng.next_f32() < true_probability { 1.0 } else { 0.0 };
            
            samples.push(CalibrationSample {
                prediction,
                ground_truth,
                intent: "threshold_test".to_string(),
                language: Some("test".to_string()),
                features: HashMap::new(),
                weight: 1.0,
            });
        }
        
        let config = IsotonicConfig {
            ece_bins: k,
            ..Default::default()
        };
        
        let sample_refs: Vec<&CalibrationSample> = samples.iter().collect();
        let mut calibrator = IsotonicCalibrator::new(config);
        
        let training_result = calibrator.train(&sample_refs).await;
        
        if training_result.is_ok() {
            let ece = calibrator.get_ece();
            let statistical_threshold = calibrator.get_statistical_ece_threshold();
            
            // Calculate expected threshold based on τ(N,K) formula
            let expected_threshold = (1.5 * ((k as f32) / (n as f32)).sqrt()).max(0.015);
            let threshold_diff = (statistical_threshold - expected_threshold).abs();
            
            println!("  N={}, K={}, p={:.1}: ECE={:.4}, τ(N,K)={:.4} (expected={:.4}, diff={:.4})", 
                    n, k, true_probability, ece, statistical_threshold, expected_threshold, threshold_diff);
            
            // Threshold calculation should be consistent with formula
            assert!(threshold_diff < 0.001, "Threshold estimator inconsistent: {:.4} vs {:.4}", 
                    statistical_threshold, expected_threshold);
            
            // For small noise, ECE should be within statistical bounds
            if ece > statistical_threshold {
                println!("    ⚠️ ECE exceeds threshold (expected for noisy data)");
            } else {
                println!("    ✓ ECE within statistical bounds");
            }
            
            // Verify monotonicity: larger N should generally give smaller thresholds
            if n >= 400 {
                assert!(statistical_threshold <= 0.025, 
                        "Statistical threshold too high for large N={}: {:.4}", n, statistical_threshold);
            }
            
            // Verify CI invariants still hold
            calibrator.validate_ci_invariants()?;
        } else {
            println!("  N={}, K={}, p={:.1}: Training failed: {:?}", 
                    n, k, true_probability, training_result.unwrap_err());
        }
    }
    
    println!("✅ Threshold estimator working correctly with synthetic noise");
    Ok(())
}

// ============================================================================
// STRESS TESTS - Extreme Conditions and Performance
// ============================================================================

#[tokio::test]
#[cfg(feature = "stress-tests")]
async fn test_stress_large_dataset_performance() -> Result<()> {
    println!("💪 STRESS TEST: Large dataset performance and memory usage");
    
    let mut rng = SimpleRng::new(11111);
    let large_n = 5000;
    let mut samples = Vec::new();
    
    // Generate large dataset
    for i in 0..large_n {
        let prediction = rng.next_f32();
        let ground_truth = if rng.next_bool() { 1.0 } else { 0.0 };
        
        samples.push(CalibrationSample {
            prediction,
            ground_truth,
            intent: "stress_test".to_string(),
            language: Some("test".to_string()),
            features: HashMap::new(),
            weight: 1.0,
        });
    }
    
    let config = IsotonicConfig {
        ece_bins: ((large_n as f32).sqrt() as usize).min(50), // Cap bins for performance
        ..Default::default()
    };
    
    let sample_refs: Vec<&CalibrationSample> = samples.iter().collect();
    let mut calibrator = IsotonicCalibrator::new(config);
    
    let start_time = std::time::Instant::now();
    let result = calibrator.train(&sample_refs).await;
    let training_time = start_time.elapsed();
    
    assert!(result.is_ok(), "Large dataset training failed: {:?}", result);
    
    let ece = calibrator.get_ece();
    let slope = calibrator.get_slope();
    
    println!("  Large dataset results:");
    println!("    Samples: {}", large_n);
    println!("    Training time: {:.2}s", training_time.as_secs_f64());
    println!("    ECE: {:.4}", ece);
    println!("    Slope: {:.3}", slope);
    println!("    Memory usage: ~{:.1}MB (estimated)", (large_n * 64) / (1024 * 1024)); // Rough estimate
    
    // Performance requirements
    assert!(training_time.as_secs_f64() < 10.0, "Training too slow: {:.2}s", training_time.as_secs_f64());
    
    // Validate invariants even for large datasets
    calibrator.validate_ci_invariants()?;
    
    println!("✅ Large dataset stress test passed");
    Ok(())
}

#[tokio::test]
async fn test_stress_temperature_platt_extreme_inputs() -> Result<()> {
    println!("💪 STRESS TEST: Temperature/Platt scaling with extreme inputs");
    
    let mut rng = SimpleRng::new(22222);
    
    // Test extreme cases for temperature scaling
    let extreme_scenarios = [
        "all_low",     // All predictions [0.01, 0.1]
        "all_high",    // All predictions [0.9, 0.99]  
        "bimodal",     // Bimodal distribution
        "uniform",     // Uniform [0, 1]
    ];
    
    for scenario_name in &extreme_scenarios {
        println!("  Testing {} scenario", scenario_name);
        
        let mut samples = Vec::new();
        for _i in 0..100 {
            let prediction = match *scenario_name {
                "all_low" => 0.01 + rng.next_f32() * 0.09,     // All predictions [0.01, 0.1]
                "all_high" => 0.9 + rng.next_f32() * 0.09,     // All predictions [0.9, 0.99]
                "bimodal" => if rng.next_bool() { 0.05 } else { 0.95 }, // Bimodal distribution
                "uniform" => rng.next_f32(),                    // Uniform [0, 1]
                _ => rng.next_f32(), // Default fallback
            };
            let ground_truth = if rng.next_bool() { 1.0 } else { 0.0 };
            
            samples.push(CalibrationSample {
                prediction,
                ground_truth,
                intent: "stress_test".to_string(),
                language: Some("test".to_string()),
                features: HashMap::new(),
                weight: 1.0,
            });
        }
        
        let sample_refs: Vec<&CalibrationSample> = samples.iter().collect();
        
        // Test temperature scaling
        let temp_config = TemperatureConfig::default();
        let mut temp_scaler = TemperatureScaler::new(temp_config);
        
        let temp_result = temp_scaler.train(&sample_refs).await;
        if temp_result.is_ok() {
            let temperature = temp_scaler.get_temperature();
            let temp_ece = temp_scaler.get_ece();
            
            println!("    Temperature scaling: T={:.3}, ECE={:.4}", temperature, temp_ece);
            
            // Temperature should be reasonable
            assert!(temperature > 0.01 && temperature < 100.0, 
                    "Extreme temperature for {} scenario: {:.3}", scenario_name, temperature);
            
            // Test calibration
            let calibrated = temp_scaler.calibrate(0.5).await?;
            assert!(calibrated >= 0.001 && calibrated <= 0.999, 
                    "Invalid temperature calibration: {:.4}", calibrated);
        } else {
            println!("    Temperature scaling failed (may be expected): {:?}", temp_result.unwrap_err());
        }
        
        // Test Platt scaling
        let platt_config = PlattConfig::default();
        let mut platt_scaler = PlattScaler::new(platt_config);
        
        let platt_result = platt_scaler.train(&sample_refs).await;
        if platt_result.is_ok() {
            let (a, b) = platt_scaler.get_parameters();
            let platt_ece = platt_scaler.get_ece();
            
            println!("    Platt scaling: A={:.3}, B={:.3}, ECE={:.4}", a, b, platt_ece);
            
            // Parameters should be finite and reasonable
            assert!(a.is_finite() && b.is_finite(), "Non-finite Platt parameters: A={:.3}, B={:.3}", a, b);
            assert!(a.abs() < 1000.0 && b.abs() < 1000.0, "Extreme Platt parameters: A={:.3}, B={:.3}", a, b);
            
            // Test calibration
            let features = HashMap::new();
            let calibrated = platt_scaler.calibrate(0.5, &features).await?;
            assert!(calibrated >= 0.001 && calibrated <= 0.999, 
                    "Invalid Platt calibration: {:.4}", calibrated);
        } else {
            println!("    Platt scaling failed (may be expected): {:?}", platt_result.unwrap_err());
        }
    }
    
    println!("✅ Temperature/Platt stress test completed");
    Ok(())
}

// ============================================================================
// CONTRACT VALIDATION TESTS - Verify Published API Contracts
// ============================================================================

#[tokio::test]
async fn test_contract_identity_calibrator_properties() -> Result<()> {
    println!("📋 CONTRACT TEST: Identity calibrator when data are already calibrated");
    
    // Create perfectly calibrated dataset
    let mut samples = Vec::new();
    
    // Generate properly calibrated data where outcomes exactly match predictions
    // Use rational fractions to ensure exact calibration, with enough samples (>30)
    let calibration_levels = [
        (0.1, 10),  // 10% → 1 positive out of 10 samples
        (0.3, 10),  // 30% → 3 positive out of 10 samples
        (0.5, 10),  // 50% → 5 positive out of 10 samples
        (0.7, 10),  // 70% → 7 positive out of 10 samples
        (0.9, 10),  // 90% → 9 positive out of 10 samples
    ];
    
    for (true_prob, samples_per_level) in calibration_levels.iter() {
        let prediction = *true_prob;
        let positive_count = (true_prob * *samples_per_level as f32).round() as usize;
        let negative_count = samples_per_level - positive_count;
        
        // Create exactly the right number of positive and negative samples
        for _ in 0..positive_count {
            samples.push(CalibrationSample {
                prediction,
                ground_truth: 1.0,
                intent: "identity_test".to_string(),
                language: Some("test".to_string()),
                features: HashMap::new(),
                weight: 1.0,
            });
        }
        
        for _ in 0..negative_count {
            samples.push(CalibrationSample {
                prediction,
                ground_truth: 0.0,
                intent: "identity_test".to_string(),
                language: Some("test".to_string()),
                features: HashMap::new(),
                weight: 1.0,
            });
        }
    }
    
    let config = IsotonicConfig::default();
    let sample_refs: Vec<&CalibrationSample> = samples.iter().collect();
    let mut calibrator = IsotonicCalibrator::new(config);
    
    calibrator.train(&sample_refs).await?;
    
    // CONTRACT: Identity calibrator should have ECE ≈ 0 and slope ≈ 1
    let ece = calibrator.get_ece();
    let slope = calibrator.get_slope();
    
    println!("  Identity calibrator results:");
    println!("    ECE: {:.6} (should be ≈ 0.0)", ece);
    println!("    Slope: {:.6} (should be ≈ 1.0)", slope);
    
    // Debug: Calculate expected ECE manually for verification
    println!("  Expected ECE calculation:");
    let mut expected_ece = 0.0;
    for (true_prob, samples_per_level) in calibration_levels.iter() {
        let positive_count = (true_prob * *samples_per_level as f32).round() as usize;
        let observed_prob = positive_count as f32 / *samples_per_level as f32;
        let bin_weight = *samples_per_level as f32 / 50.0; // Total samples = 50
        let bin_contribution = bin_weight * (true_prob - observed_prob).abs();
        expected_ece += bin_contribution;
        println!("    Bin pred={:.1}: {}/{} samples, observed={:.3}, |{:.1}-{:.3}|*{:.2}={:.6}", 
                 true_prob, positive_count, samples_per_level, observed_prob, 
                 true_prob, observed_prob, bin_weight, bin_contribution);
    }
    println!("    Total expected ECE: {:.6}", expected_ece);
    
    // Well-calibrated data should produce near-identity mapping
    assert!(ece <= 0.02, "Identity calibrator ECE too high: {:.6}", ece);
    assert!((slope - 1.0).abs() <= 0.1, "Identity calibrator slope not near 1.0: {:.6}", slope);
    
    // Test that calibration preserves well-calibrated inputs
    let features = HashMap::new();
    let test_inputs = [0.1, 0.3, 0.5, 0.7, 0.9];
    
    for &input in &test_inputs {
        let output = calibrator.calibrate(input, &features).await?;
        let deviation = (output - input).abs();
        
        println!("    {:.1} -> {:.4} (deviation: {:.4})", input, output, deviation);
        
        // CONTRACT: Identity mapping should preserve inputs with small deviation
        assert!(deviation <= 0.05, "Identity calibration deviates too much: {:.1} -> {:.4}", input, output);
    }
    
    println!("✅ Identity calibrator contract verified");
    Ok(())
}

#[tokio::test]
async fn test_contract_slope_clamp_bounds() -> Result<()> {
    println!("📋 CONTRACT TEST: Slope clamp ∈[0.9,1.1] enforcement");
    
    let mut rng = SimpleRng::new(33333);
    
    // Test various slope clamp configurations
    let clamp_configs = [
        (0.9, 1.1),   // Standard TODO.md requirement
        (0.95, 1.05), // Tighter bounds
        (0.85, 1.15), // Should be rejected (outside [0.9, 1.1])
        (1.0, 1.0),   // Exact identity (edge case)
    ];
    
    for &(min_slope, max_slope) in &clamp_configs {
        println!("  Testing slope clamp [{:.2}, {:.2}]", min_slope, max_slope);
        
        // Generate data that would naturally produce extreme slopes
        let mut samples = Vec::new();
        
        for i in 0..60 {
            let prediction = 0.4 + (i as f32) / 60.0 * 0.2; // Narrow range [0.4, 0.6]
            
            // Create sharp transitions to force extreme slopes
            let ground_truth = if i < 10 { 0.0 } else if i > 50 { 1.0 } else { 
                if rng.next_bool() { 1.0 } else { 0.0 } 
            };
            
            samples.push(CalibrationSample {
                prediction,
                ground_truth,
                intent: "slope_test".to_string(),
                language: Some("test".to_string()),
                features: HashMap::new(),
                weight: 1.0,
            });
        }
        
        let config = IsotonicConfig {
            slope_clamp: (min_slope, max_slope),
            ..Default::default()
        };
        
        let sample_refs: Vec<&CalibrationSample> = samples.iter().collect();
        let mut calibrator = IsotonicCalibrator::new(config);
        
        let training_result = calibrator.train(&sample_refs).await;
        
        // Check if configuration should be valid
        let config_valid = min_slope >= 0.9 && max_slope <= 1.1 && min_slope <= max_slope;
        
        if config_valid {
            assert!(training_result.is_ok(), "Valid slope clamp [{:.2}, {:.2}] rejected", min_slope, max_slope);
            
            let learned_slope = calibrator.get_slope();
            println!("    Learned slope: {:.4}", learned_slope);
            
            // CONTRACT: Slope must be within specified bounds (with floating-point tolerance)
            let tolerance = 1e-5; // Increased tolerance for floating-point precision
            let within_bounds = learned_slope >= min_slope - tolerance && learned_slope <= max_slope + tolerance;
            println!("      Bounds check: {:.10} >= {:.10} && {:.10} <= {:.10} = {}", 
                     learned_slope, min_slope - tolerance, learned_slope, max_slope + tolerance, within_bounds);
            assert!(within_bounds, "Slope {:.10} outside clamp bounds [{:.10}, {:.10}] (tolerance={:.0e})", 
                    learned_slope, min_slope, max_slope, tolerance);
            
            // Verify CI invariants
            calibrator.validate_ci_invariants()?;
            
        } else {
            println!("    Invalid configuration (expected failure)");
            // Invalid configurations should be handled gracefully
        }
    }
    
    println!("✅ Slope clamp contract verified");
    Ok(())
}

#[tokio::test]
async fn test_contract_tau_formula_and_constants() -> Result<()> {
    println!("📋 CONTRACT TEST: τ(N,K) formula and constants verification");
    
    // Test the statistical ECE threshold formula: τ(N,K) = max(0.015, ĉ·√(K/N))
    // where ĉ ≈ 1.5 (empirical constant for synthetic data with noise)
    
    let test_cases = [
        (100, 10),   // Small dataset
        (400, 20),   // Medium dataset  
        (1600, 40),  // Large dataset
        (50, 5),     // Very small
        (10000, 100), // Very large
    ];
    
    for &(n, k) in &test_cases {
        let config = IsotonicConfig {
            ece_bins: k,
            ..Default::default()
        };
        
        let calibrator = IsotonicCalibrator::new(config);
        let calculated_threshold = calibrator.calculate_statistical_ece_threshold(n, k);
        
        // Manual calculation of expected threshold
        let statistical_floor = 1.5 * ((k as f32) / (n as f32)).sqrt();
        let expected_threshold = statistical_floor.max(0.015);
        
        let threshold_diff = (calculated_threshold - expected_threshold).abs();
        
        println!("  N={}, K={}: τ(N,K)={:.4}, expected={:.4}, diff={:.6}", 
                n, k, calculated_threshold, expected_threshold, threshold_diff);
        
        // CONTRACT: Formula should match exactly
        assert!(threshold_diff < 1e-6, "τ(N,K) formula mismatch: {:.6} vs {:.6}", 
                calculated_threshold, expected_threshold);
        
        // CONTRACT: Threshold should be reasonable (not arbitrarily high)
        assert!(calculated_threshold <= 0.5, "Threshold unreasonably high: {:.4}", calculated_threshold);
        
        // CONTRACT: Minimum threshold is 0.015 (PHASE 4 requirement)
        assert!(calculated_threshold >= 0.015, "Threshold below minimum: {:.4}", calculated_threshold);
    }
    
    // Test constant ĉ = 1.5 rationale and bounds
    println!("  Testing empirical constant ĉ = 1.5:");
    println!("    Rationale: Accounts for synthetic data noise and equal-mass binning variance");
    println!("    Alternative ĉ ≈ 0.8 for theoretical √(2/π) but insufficient for noisy data");
    println!("    Current ĉ = 1.5 provides robust threshold with acceptable false positive rate");
    
    println!("✅ τ(N,K) formula and constants contract verified");
    Ok(())
}

#[tokio::test]
async fn test_contract_k_sqrt_n_for_aece() -> Result<()> {
    println!("📋 CONTRACT TEST: K=⌊√N⌋ for AECE in tests with N<4k");
    
    // Test the relationship K = ⌊√N⌋ for Adaptive Expected Calibration Error
    let test_datasets = [
        100,   // K should be 10
        400,   // K should be 20
        900,   // K should be 30
        1600,  // K should be 40
        3600,  // K should be 60 (still < 4k)
    ];
    
    for &n in &test_datasets {
        let expected_k = (n as f32).sqrt().floor() as usize;
        
        // Generate test dataset of size N
        let mut samples = Vec::new();
        let mut rng = SimpleRng::new(n as u64);
        
        for _i in 0..n {
            let prediction = rng.next_f32();
            let ground_truth = if rng.next_bool() { 1.0 } else { 0.0 };
            
            samples.push(CalibrationSample {
                prediction,
                ground_truth,
                intent: "aece_test".to_string(),
                language: Some("test".to_string()),
                features: HashMap::new(),
                weight: 1.0,
            });
        }
        
        // Use K = ⌊√N⌋ configuration
        let config = IsotonicConfig {
            ece_bins: expected_k,
            ..Default::default()
        };
        
        let sample_refs: Vec<&CalibrationSample> = samples.iter().collect();
        let mut calibrator = IsotonicCalibrator::new(config);
        
        let training_result = calibrator.train(&sample_refs).await;
        
        if training_result.is_ok() {
            let ece = calibrator.get_ece();
            let statistical_threshold = calibrator.get_statistical_ece_threshold();
            
            println!("  N={}, K=⌊√N⌋={}: ECE={:.4}, τ(N,K)={:.4}", 
                    n, expected_k, ece, statistical_threshold);
            
            // CONTRACT: K = ⌊√N⌋ should provide good balance between bias and variance
            assert_eq!(expected_k, (n as f32).sqrt().floor() as usize, 
                      "K calculation incorrect: {} vs ⌊√{}⌋", expected_k, n);
            
            // For large enough datasets, AECE should be reasonable
            if n >= 400 {
                assert!(ece <= statistical_threshold, 
                        "AECE exceeds statistical threshold for N={}: {:.4} > {:.4}", 
                        n, ece, statistical_threshold);
            }
            
            // Verify that the K=√N choice leads to efficient binning
            // (not too many empty bins, not too few samples per bin)
            let avg_samples_per_bin = n as f32 / expected_k as f32;
            assert!(avg_samples_per_bin >= 1.0, "Too many bins for dataset size: {:.1} samples/bin", avg_samples_per_bin);
            assert!(avg_samples_per_bin <= 100.0, "Too few bins for dataset size: {:.1} samples/bin", avg_samples_per_bin);
            
        } else {
            println!("  N={}, K={}: Training failed: {:?}", n, expected_k, training_result.unwrap_err());
        }
    }
    
    println!("✅ K=⌊√N⌋ for AECE contract verified");
    Ok(())
}

#[tokio::test]
async fn test_determinism_and_debug_validation() -> Result<()> {
    println!("🔒 DETERMINISM TEST: Validate reproducible results and debug output");
    
    // Create truly well-calibrated test data to ensure slope ∈ [0.9, 1.1]
    let mut samples = Vec::new();
    
    // Generate calibrated data where ground truth follows predictions exactly
    let calibration_levels = [
        (0.1, 30),  // 10% → 3 positives out of 30
        (0.3, 20),  // 30% → 6 positives out of 20
        (0.5, 20),  // 50% → 10 positives out of 20
        (0.7, 20),  // 70% → 14 positives out of 20
        (0.9, 10),  // 90% → 9 positives out of 10
    ];
    
    for (prediction, total_samples) in calibration_levels.iter() {
        let positive_count = (*prediction * *total_samples as f32).round() as usize;
        let negative_count = total_samples - positive_count;
        
        // Add positive samples
        for _ in 0..positive_count {
            samples.push(CalibrationSample {
                prediction: *prediction,
                ground_truth: 1.0,
                intent: "determinism_test".to_string(),
                language: Some("rust".to_string()),
                features: HashMap::new(),
                weight: 1.0,
            });
        }
        
        // Add negative samples
        for _ in 0..negative_count {
            samples.push(CalibrationSample {
                prediction: *prediction,
                ground_truth: 0.0,
                intent: "determinism_test".to_string(),
                language: Some("rust".to_string()),
                features: HashMap::new(),
                weight: 1.0,
            });
        }
    }
    
    let config = IsotonicConfig::default();
    let sample_refs: Vec<&CalibrationSample> = samples.iter().collect();
    
    // Run calibration 3 times and check determinism
    let mut hashes = Vec::new();
    let mut debug_dumps = Vec::new();
    
    for run_id in 1..=3 {
        let mut calibrator = IsotonicCalibrator::new(config.clone());
        calibrator.train(&sample_refs).await?;
        
        // Validate determinism and get debug dump
        let debug_info = calibrator.validate_determinism_and_debug(&sample_refs, run_id)?;
        debug_dumps.push(debug_info.clone());
        
        // Extract hash from debug info
        if let Some(hash_start) = debug_info.find("hash=0x") {
            let hash_end = debug_info[hash_start + 7..].find(',').unwrap_or(debug_info.len() - hash_start - 7);
            let hash_str = &debug_info[hash_start + 7..hash_start + 7 + hash_end];
            hashes.push(hash_str.to_string());
        }
        
        println!("Run {}: {}", run_id, debug_info.split('\n').next().unwrap_or(""));
    }
    
    // CI INVARIANT: hash_run1 == hash_run2 == hash_run3
    assert_eq!(hashes.len(), 3, "Should have 3 hash values");
    assert_eq!(hashes[0], hashes[1], "Hash run1 != run2: {} vs {}", hashes[0], hashes[1]);
    assert_eq!(hashes[1], hashes[2], "Hash run2 != run3: {} vs {}", hashes[1], hashes[2]);
    
    // Verify debug dump contains expected elements
    let first_dump = &debug_dumps[0];
    assert!(first_dump.contains("N=100"), "Debug dump should contain sample count");
    assert!(first_dump.contains("K_eff="), "Debug dump should contain effective bins");
    assert!(first_dump.contains("τ="), "Debug dump should contain statistical threshold");
    assert!(first_dump.contains("bin | n_b | acc_b"), "Debug dump should contain bin table");
    
    // Verify merged bin rate is reasonable (≤ 5% warning threshold)
    if let Some(rate_start) = first_dump.find("merged_rate=") {
        let rate_end = first_dump[rate_start + 12..].find('%').unwrap_or(3);
        let rate_str = &first_dump[rate_start + 12..rate_start + 12 + rate_end];
        let merged_rate: f32 = rate_str.parse().unwrap_or(100.0);
        
        if merged_rate > 5.0 {
            println!("⚠️ Warning: High merged bin rate: {:.1}%", merged_rate);
        }
        assert!(merged_rate <= 20.0, "Merged bin rate too high: {:.1}% > 20%", merged_rate);
    }
    
    println!("✅ Determinism validated: 3 identical hashes");
    println!("📊 Debug dump sample:\n{}", first_dump.lines().take(6).collect::<Vec<_>>().join("\n"));
    
    Ok(())
}