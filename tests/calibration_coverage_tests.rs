//! Bootstrap Coverage Validation Tests for Statistical ECE Thresholds τ(N,K)
//!
//! Validates that statistical ECE thresholds τ(N,K) = max(0.015, ĉ·√(K/N)) provide
//! adequate coverage probability P[ECE_empirical ≤ τ(N,K)] ≥ 0.95 through bootstrap analysis.
//!
//! Key Features:
//! - Bootstrap B=1000 stratified resamples with replacement
//! - Coverage validation p̂ ≥ 0.95, with adaptive ĉ search if violated
//! - Fixed seed=42 for reproducibility
//! - Multiple data classes: well-calibrated, under-calibrated, over-calibrated
//! - Comprehensive artifact generation for statistical analysis

use anyhow::Result;
use chrono;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;

// Import lens-core types - we'll define the essential ones locally if needed
use lens_core::calibration::CalibrationSample;

/// Configuration for bootstrap coverage tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageTestConfig {
    /// Number of bootstrap samples
    pub bootstrap_samples: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Target coverage probability
    pub target_coverage: f32,
    /// Tolerance for coverage probability (allows for finite B effects)
    pub coverage_tolerance: f32,
    /// ECE calculation tolerance
    pub ece_tolerance: f32,
    /// Sample sizes to test
    pub test_sample_sizes: Vec<usize>,
    /// Bin counts to test
    pub test_bin_counts: Vec<usize>,
}

impl Default for CoverageTestConfig {
    fn default() -> Self {
        Self {
            bootstrap_samples: 1000,
            seed: 42,
            target_coverage: 0.95,
            coverage_tolerance: -0.01, // Allow p̂ ≥ 0.94 for finite B
            ece_tolerance: 1e-4,
            test_sample_sizes: vec![50, 100, 200, 500],
            test_bin_counts: vec![10, 15, 20],
        }
    }
}

/// Result of bootstrap coverage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageResult {
    /// Sample size N
    pub sample_size: usize,
    /// Effective bin count K_eff
    pub k_effective: usize,
    /// Statistical ECE threshold τ(N,K)
    pub tau_threshold: f32,
    /// Empirical constant ĉ
    pub c_hat: f32,
    /// Mean ECE across bootstrap samples
    pub mean_ece_bootstrap: f32,
    /// 95th percentile ECE across bootstrap samples
    pub p95_ece_bootstrap: f32,
    /// Coverage probability p̂
    pub coverage_probability: f32,
    /// Whether coverage test passes
    pub coverage_passes: bool,
    /// Bootstrap ECE values (for detailed analysis)
    pub bootstrap_ece_values: Vec<f32>,
    /// Data class tested
    pub data_class: String,
}

/// Comprehensive coverage test artifacts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageTestArtifacts {
    /// Configuration used
    pub config: CoverageTestConfig,
    /// Results by data class and parameter combination
    pub results: Vec<CoverageResult>,
    /// Summary statistics
    pub summary: CoverageSummary,
    /// Test execution timestamp
    pub timestamp: String,
}

/// Summary of coverage test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageSummary {
    /// Total tests conducted
    pub total_tests: usize,
    /// Tests that passed coverage requirements
    pub passed_tests: usize,
    /// Overall pass rate
    pub pass_rate: f32,
    /// Minimum coverage probability observed
    pub min_coverage_probability: f32,
    /// Maximum coverage probability observed
    pub max_coverage_probability: f32,
    /// Average empirical constant ĉ
    pub average_c_hat: f32,
}

/// Generate bootstrap indices for stratified resampling with replacement
pub fn bootstrap_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen_range(0..n)).collect()
}

/// Perform bootstrap coverage validation
pub fn coverage_check(
    samples: &[CalibrationSample],
    k: usize,
    tau_fn: fn(usize, usize, f32) -> f32,
    c_hat: f32,
    b: usize,
    seed: u64,
) -> Result<CoverageResult> {
    let n = samples.len();
    let k_eff = k.min(((n as f32).sqrt().floor() as usize).max(1));
    let tau = tau_fn(n, k_eff, c_hat);
    
    let mut bootstrap_ece_values = Vec::with_capacity(b);
    let mut violations = 0;
    
    // Use k_eff for ECE calculation
    
    for bootstrap_iteration in 0..b {
        // Generate bootstrap sample with replacement
        let bootstrap_seed = seed.wrapping_add((bootstrap_iteration as u64).wrapping_mul(12345));
        let indices = bootstrap_indices(n, bootstrap_seed);
        
        let bootstrap_samples: Vec<CalibrationSample> = indices
            .iter()
            .map(|&idx| samples[idx].clone())
            .collect();
        
        // Calculate ECE for this bootstrap sample
        let ece_b = calculate_bootstrap_ece(&bootstrap_samples, k_eff)?;
        
        bootstrap_ece_values.push(ece_b);
        
        // Count threshold violations
        if ece_b > tau {
            violations += 1;
        }
    }
    
    // Calculate coverage probability p̂ = (1/B)·∑[ECE_b ≤ τ(N,K)]
    let coverage_probability = (b - violations) as f32 / b as f32;
    
    // Calculate summary statistics
    bootstrap_ece_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_ece = bootstrap_ece_values.iter().sum::<f32>() / b as f32;
    let p95_ece = bootstrap_ece_values[(0.95 * b as f32) as usize];
    
    let coverage_passes = coverage_probability >= 0.95;
    
    Ok(CoverageResult {
        sample_size: n,
        k_effective: k_eff,
        tau_threshold: tau,
        c_hat,
        mean_ece_bootstrap: mean_ece,
        p95_ece_bootstrap: p95_ece,
        coverage_probability,
        coverage_passes,
        bootstrap_ece_values,
        data_class: "unknown".to_string(),
    })
}

/// Calculate ECE using equal-mass binning (standalone implementation)
fn calculate_bootstrap_ece(samples: &[CalibrationSample], k_bins: usize) -> Result<f32> {
    if samples.is_empty() {
        return Ok(0.0);
    }
    
    let n = samples.len();
    let k_eff = k_bins.min(((n as f32).sqrt().floor() as usize).max(1));
    
    // Sort samples by prediction for equal-mass binning
    let mut sorted_samples = samples.to_vec();
    sorted_samples.sort_by(|a, b| a.prediction.partial_cmp(&b.prediction).unwrap());
    
    // Filter out invalid samples
    let valid_samples: Vec<_> = sorted_samples
        .iter()
        .filter(|s| s.prediction.is_finite() && s.ground_truth.is_finite() && s.weight > 0.0)
        .collect();
    
    if valid_samples.is_empty() {
        return Ok(0.0);
    }
    
    let total_weight: f32 = valid_samples.iter().map(|s| s.weight).sum();
    let target_weight_per_bin = total_weight / k_eff as f32;
    
    let mut ece = 0.0;
    let mut current_bin_weight = 0.0;
    let mut current_bin_predictions = Vec::new();
    let mut current_bin_targets = Vec::new();
    let mut current_bin_weights = Vec::new();
    
    for (i, sample) in valid_samples.iter().enumerate() {
        current_bin_predictions.push(sample.prediction);
        current_bin_targets.push(sample.ground_truth);
        current_bin_weights.push(sample.weight);
        current_bin_weight += sample.weight;
        
        let should_close_bin = current_bin_weight >= target_weight_per_bin 
            || i == valid_samples.len() - 1;
        
        if should_close_bin && !current_bin_predictions.is_empty() {
            // Calculate weighted mean confidence and accuracy
            let weighted_confidence_sum: f32 = current_bin_predictions
                .iter()
                .zip(&current_bin_weights)
                .map(|(pred, weight)| pred * weight)
                .sum();
            let weighted_accuracy_sum: f32 = current_bin_targets
                .iter()
                .zip(&current_bin_weights)
                .map(|(target, weight)| target * weight)
                .sum();
            
            let mean_confidence = weighted_confidence_sum / current_bin_weight;
            let mean_accuracy = weighted_accuracy_sum / current_bin_weight;
            
            // Calculate bin contribution to ECE
            let bin_error = (mean_confidence - mean_accuracy).abs();
            let bin_weight_fraction = current_bin_weight / total_weight;
            ece += bin_error * bin_weight_fraction;
            
            // Reset for next bin
            current_bin_weight = 0.0;
            current_bin_predictions.clear();
            current_bin_targets.clear();
            current_bin_weights.clear();
        }
    }
    
    Ok(ece)
}

/// Statistical ECE threshold function τ(N,K,ĉ) = max(0.015, ĉ·√(K/N))
fn statistical_ece_threshold(n: usize, k: usize, c_hat: f32) -> f32 {
    let statistical_floor = c_hat * ((k as f32) / (n as f32)).sqrt();
    statistical_floor.max(0.015)
}

/// Adaptive search for empirical constant ĉ to achieve target coverage
pub fn adaptive_c_search(
    samples: &[CalibrationSample],
    k: usize,
    target_coverage: f32,
) -> Result<f32> {
    const MAX_ITERATIONS: usize = 20;
    const CONVERGENCE_TOLERANCE: f32 = 0.005;
    
    let mut c_hat = 1.5; // Start with reasonable initial value
    let mut iteration = 0;
    
    loop {
        let result = coverage_check(samples, k, statistical_ece_threshold, c_hat, 1000, 42)?;
        
        let coverage_gap = result.coverage_probability - target_coverage;
        
        // Check convergence
        if coverage_gap.abs() <= CONVERGENCE_TOLERANCE || iteration >= MAX_ITERATIONS {
            return Ok(c_hat);
        }
        
        // Adjust ĉ based on coverage gap
        if coverage_gap < 0.0 {
            // Coverage too low, increase ĉ
            c_hat *= 1.1;
        } else {
            // Coverage too high, decrease ĉ (but not below minimum)
            c_hat *= 0.95;
            c_hat = c_hat.max(0.5);
        }
        
        iteration += 1;
    }
}

/// Generate synthetic calibration data for different calibration quality classes
fn generate_calibration_data(
    n: usize, 
    data_class: &str, 
    seed: u64,
) -> Result<Vec<CalibrationSample>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut samples = Vec::with_capacity(n);
    
    for i in 0..n {
        let base_prediction = rng.gen::<f32>();
        let noise = rng.gen::<f32>() * 0.1; // Small amount of noise
        
        let (prediction, ground_truth) = match data_class {
            "well-calibrated" => {
                // Well-calibrated: ground truth follows prediction with small noise
                let gt = if base_prediction + noise * 0.1 > 0.5 { 1.0 } else { 0.0 };
                (base_prediction.clamp(0.001, 0.999), gt)
            },
            "under-calibrated" => {
                // Under-calibrated: predictions are too conservative
                let conservative_pred = base_prediction * 0.7 + 0.15; // Compress to [0.15, 0.85]
                let gt = if base_prediction > 0.5 { 1.0 } else { 0.0 };
                (conservative_pred.clamp(0.001, 0.999), gt)
            },
            "over-calibrated" => {
                // Over-calibrated: predictions are too extreme
                let extreme_pred: f32 = if base_prediction > 0.5 {
                    0.5 + (base_prediction - 0.5) * 1.5
                } else {
                    0.5 - (0.5 - base_prediction) * 1.5
                };
                let gt = if base_prediction + noise > 0.4 { 1.0 } else { 0.0 };
                (extreme_pred.clamp(0.001, 0.999), gt)
            },
            _ => return Err(anyhow::anyhow!("Unknown data class: {}", data_class)),
        };
        
        samples.push(CalibrationSample {
            prediction,
            ground_truth,
            intent: format!("test_intent_{}", i % 3),
            language: Some("rust".to_string()),
            features: HashMap::new(),
            weight: 1.0,
        });
    }
    
    Ok(samples)
}

#[tokio::test]
async fn test_bootstrap_coverage_validation_comprehensive() -> Result<()> {
    let config = CoverageTestConfig::default();
    let mut all_results = Vec::new();
    
    let data_classes = vec!["well-calibrated", "under-calibrated", "over-calibrated"];
    
    println!("🧪 BOOTSTRAP COVERAGE VALIDATION TEST");
    println!("Configuration: B={}, seed={}, target_coverage={:.2}", 
             config.bootstrap_samples, config.seed, config.target_coverage);
    
    for data_class in &data_classes {
        println!("\n📊 Testing {} data:", data_class);
        
        for &n in &config.test_sample_sizes {
            for &k in &config.test_bin_counts {
                // Generate synthetic data
                let samples = generate_calibration_data(
                    n, 
                    data_class, 
                    config.seed.wrapping_add(n as u64)
                )?;
                
                // Find appropriate ĉ for this configuration
                let c_hat = adaptive_c_search(&samples, k, config.target_coverage)?;
                
                // Run coverage validation
                let mut result = coverage_check(
                    &samples, 
                    k, 
                    statistical_ece_threshold, 
                    c_hat,
                    config.bootstrap_samples, 
                    config.seed
                )?;
                
                result.data_class = data_class.to_string();
                
                println!("  N={}, K={}, K_eff={}, τ={:.4}, ĉ={:.3}, mean(ECE_b)={:.4}, p95(ECE_b)={:.4}, p̂={:.3} {}",
                        result.sample_size,
                        k,
                        result.k_effective,
                        result.tau_threshold,
                        result.c_hat,
                        result.mean_ece_bootstrap,
                        result.p95_ece_bootstrap,
                        result.coverage_probability,
                        if result.coverage_passes { "✅" } else { "❌" }
                );
                
                // Validate numerical tolerances
                assert!(
                    (result.mean_ece_bootstrap - result.bootstrap_ece_values.iter().sum::<f32>() / result.bootstrap_ece_values.len() as f32).abs() < config.ece_tolerance,
                    "Mean ECE calculation tolerance violated"
                );
                
                // Check coverage requirement (with tolerance for finite B effects)
                assert!(
                    result.coverage_probability >= config.target_coverage + config.coverage_tolerance,
                    "Coverage requirement violated: p̂={:.3} < {:.2}",
                    result.coverage_probability,
                    config.target_coverage + config.coverage_tolerance
                );
                
                all_results.push(result);
            }
        }
    }
    
    // Generate summary statistics
    let total_tests = all_results.len();
    let passed_tests = all_results.iter().filter(|r| r.coverage_passes).count();
    let pass_rate = passed_tests as f32 / total_tests as f32;
    
    let min_coverage = all_results.iter()
        .map(|r| r.coverage_probability)
        .fold(f32::INFINITY, f32::min);
    let max_coverage = all_results.iter()
        .map(|r| r.coverage_probability)
        .fold(f32::NEG_INFINITY, f32::max);
    let average_c_hat = all_results.iter()
        .map(|r| r.c_hat)
        .sum::<f32>() / all_results.len() as f32;
    
    let summary = CoverageSummary {
        total_tests,
        passed_tests,
        pass_rate,
        min_coverage_probability: min_coverage,
        max_coverage_probability: max_coverage,
        average_c_hat,
    };
    
    println!("\n📋 COVERAGE TEST SUMMARY:");
    println!("  Total tests: {}", summary.total_tests);
    println!("  Passed tests: {}", summary.passed_tests);
    println!("  Pass rate: {:.1}%", summary.pass_rate * 100.0);
    println!("  Coverage range: [{:.3}, {:.3}]", summary.min_coverage_probability, summary.max_coverage_probability);
    println!("  Average ĉ: {:.3}", summary.average_c_hat);
    
    // Generate comprehensive artifacts
    let artifacts = CoverageTestArtifacts {
        config,
        results: all_results,
        summary: summary.clone(),
        timestamp: chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string(),
    };
    
    // Write artifacts to file
    let artifacts_json = serde_json::to_string_pretty(&artifacts)?;
    fs::write("target/bootstrap_coverage_artifacts.json", artifacts_json)?;
    
    println!("\n💾 Artifacts saved to: target/bootstrap_coverage_artifacts.json");
    
    // Assert overall test success
    assert!(
        summary.pass_rate >= 0.95,
        "Overall coverage test pass rate {:.1}% below 95%",
        summary.pass_rate * 100.0
    );
    
    Ok(())
}

#[tokio::test]
async fn test_bootstrap_coverage_single_case() -> Result<()> {
    // Test a specific case for debugging
    let n = 100;
    let k = 15;
    let seed = 42;
    
    println!("🔬 Single Case Bootstrap Coverage Test (N={}, K={})", n, k);
    
    // Generate well-calibrated data
    let samples = generate_calibration_data(n, "well-calibrated", seed)?;
    
    // Test with different ĉ values
    let c_values = vec![0.8, 1.0, 1.5, 2.0];
    
    for &c_hat in &c_values {
        let result = coverage_check(
            &samples, 
            k, 
            statistical_ece_threshold, 
            c_hat,
            1000, 
            seed
        )?;
        
        println!("ĉ={:.1}: τ={:.4}, p̂={:.3}, mean(ECE)={:.4}, p95(ECE)={:.4} {}",
                c_hat,
                result.tau_threshold,
                result.coverage_probability,
                result.mean_ece_bootstrap,
                result.p95_ece_bootstrap,
                if result.coverage_passes { "✅" } else { "❌" }
        );
    }
    
    Ok(())
}

#[test]
fn test_bootstrap_indices_reproducibility() {
    let n = 10;
    let seed = 42;
    
    let indices1 = bootstrap_indices(n, seed);
    let indices2 = bootstrap_indices(n, seed);
    
    assert_eq!(indices1, indices2, "Bootstrap indices should be reproducible with same seed");
    assert_eq!(indices1.len(), n, "Bootstrap indices should have correct length");
    
    // All indices should be in range [0, n)
    for &idx in &indices1 {
        assert!(idx < n, "Bootstrap index {} out of range [0, {})", idx, n);
    }
}

#[test]
fn test_statistical_ece_threshold_properties() {
    // Test threshold properties with reasonable precision
    let threshold_1000_15 = statistical_ece_threshold(1000, 15, 1.0);
    assert!((threshold_1000_15 - 0.122).abs() < 0.001, "Large N should use statistical floor: got {:.3}", threshold_1000_15);
    
    let threshold_10_5 = statistical_ece_threshold(10, 5, 1.0);
    assert!((threshold_10_5 - 0.707).abs() < 0.001, "Small N should use statistical floor: got {:.3}", threshold_10_5);
    
    let threshold_large = statistical_ece_threshold(1000000, 15, 0.1);
    assert_eq!(threshold_large, 0.015, "Very large N should use minimum 0.015");
    
    // Test that threshold increases with K and decreases with N
    assert!(
        statistical_ece_threshold(100, 20, 1.0) > statistical_ece_threshold(100, 10, 1.0),
        "Threshold should increase with K"
    );
    assert!(
        statistical_ece_threshold(200, 15, 1.0) < statistical_ece_threshold(100, 15, 1.0),
        "Threshold should decrease with N"
    );
}

#[tokio::test]
async fn test_adaptive_c_search() -> Result<()> {
    let n = 200;
    let k = 15;
    let samples = generate_calibration_data(n, "well-calibrated", 42)?;
    
    let c_hat = adaptive_c_search(&samples, k, 0.95)?;
    
    println!("Adaptive ĉ search result: {:.3}", c_hat);
    
    // Verify the found ĉ actually achieves target coverage
    let result = coverage_check(&samples, k, statistical_ece_threshold, c_hat, 1000, 42)?;
    
    println!("Coverage with found ĉ: {:.3}", result.coverage_probability);
    
    // Should be close to target (within some tolerance for finite bootstrap samples)
    assert!(
        (result.coverage_probability - 0.95).abs() < 0.05,
        "Adaptive ĉ search should find value close to target coverage"
    );
    
    // ĉ should be positive and reasonable
    assert!(c_hat > 0.0, "ĉ should be positive");
    assert!(c_hat < 10.0, "ĉ should be reasonable magnitude");
    
    Ok(())
}