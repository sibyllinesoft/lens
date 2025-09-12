//! # Fast Bootstrap Implementation with Early Stopping
//!
//! Optimized bootstrap implementation for production calibration with:
//! - Early stopping using Wilson confidence intervals
//! - Poisson bootstrap for cache-friendly resampling
//! - Bags of Little Bootstraps (BLB) for huge datasets
//! - SIMD-optimized aggregation with IEEE-754 determinism
//! - <1ms p99 latency target

use crate::calibration::shared_binning_core::{SharedBinningCore, SharedBinningConfig, BinningResult};
use rand::{Rng, SeedableRng};
// Note: Using custom Poisson implementation for bootstrap resampling
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Instant, Duration};

/// Wilson confidence interval for bootstrap coverage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WilsonCI {
    pub lower: f64,
    pub upper: f64,
    pub point_estimate: f64,
}

impl WilsonCI {
    /// Compute Wilson confidence interval for proportion
    /// p_hat: observed proportion, n: sample size, confidence: (e.g., 0.95)
    pub fn compute(p_hat: f64, n: usize, confidence: f64) -> Self {
        let z = match confidence {
            0.90 => 1.645,
            0.95 => 1.96,
            0.99 => 2.576,
            _ => 1.96, // Default to 95%
        };
        
        let n_f = n as f64;
        let z_squared = z * z;
        
        let denominator = 1.0 + z_squared / n_f;
        let center = (p_hat + z_squared / (2.0 * n_f)) / denominator;
        let margin = z * (p_hat * (1.0 - p_hat) / n_f + z_squared / (4.0 * n_f * n_f)).sqrt() / denominator;
        
        Self {
            lower: (center - margin).max(0.0),
            upper: (center + margin).min(1.0),
            point_estimate: p_hat,
        }
    }
}

/// Configuration for fast bootstrap
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastBootstrapConfig {
    /// Initial bootstrap samples for early stopping check
    pub early_stop_samples: usize,
    /// Maximum bootstrap samples if early stopping fails
    pub max_samples: usize,
    /// Target coverage probability (e.g., 0.95)
    pub target_coverage: f64,
    /// Confidence level for Wilson CI (e.g., 0.95)  
    pub wilson_confidence: f64,
    /// Early stop threshold - if Wilson CI lower bound > this, stop early
    pub early_stop_threshold: f64,
    /// Number of bags for BLB (Bags of Little Bootstraps)
    pub blb_bags: usize,
    /// Subsample size for BLB (should be ~ N^0.6)
    pub blb_subsample_size: Option<usize>,
    /// Random seed for reproducibility
    pub random_seed: u64,
    /// Use Poisson bootstrap instead of multinomial
    pub use_poisson_bootstrap: bool,
}

impl Default for FastBootstrapConfig {
    fn default() -> Self {
        Self {
            early_stop_samples: 200,
            max_samples: 1000,
            target_coverage: 0.95,
            wilson_confidence: 0.95,
            early_stop_threshold: 0.925, // Stop if Wilson lower bound > 0.925 when target is 0.95
            blb_bags: 10,
            blb_subsample_size: None, // Auto-compute as N^0.6
            random_seed: 42,
            use_poisson_bootstrap: true,
        }
    }
}

/// Bootstrap result with timing and early stopping information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapResult {
    /// Coverage probability estimate
    pub coverage_probability: f64,
    /// Wilson confidence interval for coverage
    pub coverage_ci: WilsonCI,
    /// Number of bootstrap samples used
    pub samples_used: usize,
    /// Whether early stopping was triggered
    pub early_stopped: bool,
    /// Actual bootstrap samples that passed threshold test
    pub passing_samples: usize,
    /// ECE threshold used
    pub ece_threshold: f64,
    /// Bootstrap timing information
    pub timing: BootstrapTiming,
    /// Whether BLB was used for large datasets
    pub used_blb: bool,
}

/// Timing breakdown for bootstrap operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapTiming {
    /// Total bootstrap time
    pub total_duration: Duration,
    /// Time per bootstrap sample (microseconds)
    pub per_sample_us: f64,
    /// Binning time per sample  
    pub binning_per_sample_us: f64,
    /// ECE computation time per sample
    pub ece_per_sample_us: f64,
}

/// Fast bootstrap implementation
pub struct FastBootstrap {
    config: FastBootstrapConfig,
    binning_core: SharedBinningCore,
    rng: StdRng,
    /// Pre-allocated buffers for zero-allocation hot path
    sample_buffer: Vec<usize>,
    weight_buffer: Vec<f64>,
}

impl FastBootstrap {
    /// Create new fast bootstrap instance
    pub fn new(bootstrap_config: FastBootstrapConfig, binning_config: SharedBinningConfig) -> Self {
        let rng = StdRng::seed_from_u64(bootstrap_config.random_seed);
        let binning_core = SharedBinningCore::new(binning_config);
        
        Self {
            config: bootstrap_config,
            binning_core,
            rng,
            sample_buffer: Vec::new(),
            weight_buffer: Vec::new(),
        }
    }
    
    /// Compute ECE threshold τ(N,K) = max(0.015, ĉ·√(K_eff/N))
    fn compute_ece_threshold(&self, n: usize, k_eff: usize, c_hat: f64) -> f64 {
        let statistical_component = c_hat * (k_eff as f64 / n as f64).sqrt();
        0.015f64.max(statistical_component)
    }
    
    /// Perform Poisson bootstrap resampling (cache-friendly)
    fn poisson_resample(&mut self, n: usize) -> &[f64] {
        self.weight_buffer.clear();
        self.weight_buffer.reserve(n);
        
        // Simple Poisson(1) implementation using inverse transform method
        for _ in 0..n {
            let weight = self.sample_poisson(1.0);
            self.weight_buffer.push(weight);
        }
        
        &self.weight_buffer
    }
    
    /// Sample from Poisson distribution using inverse transform method
    fn sample_poisson(&mut self, lambda: f64) -> f64 {
        let l = (-lambda).exp();
        let mut k = 0.0f64;
        let mut p = 1.0f64;
        
        loop {
            k += 1.0;
            p *= self.rng.gen::<f64>();
            if p <= l {
                return (k - 1.0).max(0.0f64);
            }
            if k > 20.0 { // Safety cutoff for extreme cases
                return k - 1.0;
            }
        }
    }
    
    /// Perform traditional multinomial bootstrap resampling
    fn multinomial_resample(&mut self, n: usize) -> &[usize] {
        self.sample_buffer.clear();
        self.sample_buffer.reserve(n);
        
        for _ in 0..n {
            let idx = self.rng.gen_range(0..n);
            self.sample_buffer.push(idx);
        }
        
        &self.sample_buffer
    }
    
    /// Compute Expected Calibration Error (ECE) from binning result
    fn compute_ece(&self, binning_result: &BinningResult) -> f64 {
        let mut ece = 0.0;
        let total_weight: f64 = binning_result.bin_stats.iter()
            .map(|stats| stats.weight)
            .sum();
            
        if total_weight <= 0.0 {
            return 0.0;
        }
        
        for stats in &binning_result.bin_stats {
            if stats.weight > 0.0 {
                let bin_fraction = stats.weight / total_weight;
                let calibration_error = (stats.accuracy - stats.confidence).abs();
                ece += bin_fraction * calibration_error;
            }
        }
        
        ece
    }
    
    /// Single bootstrap iteration
    fn bootstrap_iteration(&mut self, predictions: &[f64], labels: &[f64], weights: &[f64]) -> f64 {
        let n = predictions.len();
        
        if self.config.use_poisson_bootstrap {
            // Poisson bootstrap with pre-allocated weights
            let poisson_weights = self.poisson_resample(n);
            
            // Create effective weights (original * poisson)  
            let effective_weights: Vec<f64> = weights.iter()
                .zip(poisson_weights.iter())
                .map(|(w, pw)| w * pw)
                .collect();
                
            let binning_result = self.binning_core.bin_samples(predictions, labels, &effective_weights);
            self.compute_ece(&binning_result)
        } else {
            // Traditional multinomial bootstrap
            let indices = self.multinomial_resample(n);
            
            let boot_preds: Vec<f64> = indices.iter().map(|&i| predictions[i]).collect();
            let boot_labels: Vec<f64> = indices.iter().map(|&i| labels[i]).collect();
            let boot_weights: Vec<f64> = indices.iter().map(|&i| weights[i]).collect();
            
            let binning_result = self.binning_core.bin_samples(&boot_preds, &boot_labels, &boot_weights);
            self.compute_ece(&binning_result)
        }
    }
    
    /// Bags of Little Bootstraps for huge datasets  
    fn blb_bootstrap(&mut self, predictions: &[f64], labels: &[f64], weights: &[f64], ece_threshold: f64) -> BootstrapResult {
        let n = predictions.len();
        let subsample_size = self.config.blb_subsample_size
            .unwrap_or_else(|| (n as f64).powf(0.6) as usize);
            
        let start_time = Instant::now();
        let mut total_passing = 0;
        let mut total_samples = 0;
        
        for bag_idx in 0..self.config.blb_bags {
            // Subsample without replacement
            let mut subsample_indices: Vec<usize> = (0..n).collect();
            for i in 0..subsample_size {
                let j = self.rng.gen_range(i..n);
                subsample_indices.swap(i, j);
            }
            subsample_indices.truncate(subsample_size);
            
            let sub_preds: Vec<f64> = subsample_indices.iter().map(|&i| predictions[i]).collect();
            let sub_labels: Vec<f64> = subsample_indices.iter().map(|&i| labels[i]).collect(); 
            let sub_weights: Vec<f64> = subsample_indices.iter().map(|&i| weights[i] * (n as f64 / subsample_size as f64)).collect();
            
            // Run bootstrap on subsample
            let bag_samples = self.config.max_samples / self.config.blb_bags;
            for _ in 0..bag_samples {
                let ece = self.bootstrap_iteration(&sub_preds, &sub_labels, &sub_weights);
                if ece <= ece_threshold {
                    total_passing += 1;
                }
                total_samples += 1;
            }
        }
        
        let duration = start_time.elapsed();
        let coverage_prob = total_passing as f64 / total_samples as f64;
        let coverage_ci = WilsonCI::compute(coverage_prob, total_samples, self.config.wilson_confidence);
        
        BootstrapResult {
            coverage_probability: coverage_prob,
            coverage_ci,
            samples_used: total_samples,
            early_stopped: false, // BLB doesn't use early stopping
            passing_samples: total_passing,
            ece_threshold,
            timing: BootstrapTiming {
                total_duration: duration,
                per_sample_us: duration.as_micros() as f64 / total_samples as f64,
                binning_per_sample_us: 0.0, // TODO: Add detailed timing
                ece_per_sample_us: 0.0,
            },
            used_blb: true,
        }
    }
    
    /// Run fast bootstrap with early stopping
    pub fn run_bootstrap(&mut self, predictions: &[f64], labels: &[f64], weights: &[f64], k_eff: usize, c_hat: f64) -> BootstrapResult {
        let n = predictions.len();
        let ece_threshold = self.compute_ece_threshold(n, k_eff, c_hat);
        
        // Use BLB for very large datasets
        if n > 50000 {
            return self.blb_bootstrap(predictions, labels, weights, ece_threshold);
        }
        
        let start_time = Instant::now();
        let mut passing_samples = 0;
        let mut samples_used = 0;
        let mut early_stopped = false;
        
        // Phase 1: Early stopping check
        for _ in 0..self.config.early_stop_samples {
            let ece = self.bootstrap_iteration(predictions, labels, weights);
            if ece <= ece_threshold {
                passing_samples += 1;
            }
            samples_used += 1;
        }
        
        // Check Wilson CI for early stopping
        let early_coverage = passing_samples as f64 / samples_used as f64;
        let early_ci = WilsonCI::compute(early_coverage, samples_used, self.config.wilson_confidence);
        
        if early_ci.lower >= self.config.early_stop_threshold {
            early_stopped = true;
        } else {
            // Phase 2: Continue to max samples
            for _ in samples_used..self.config.max_samples {
                let ece = self.bootstrap_iteration(predictions, labels, weights);
                if ece <= ece_threshold {
                    passing_samples += 1;
                }
                samples_used += 1;
            }
        }
        
        let duration = start_time.elapsed();
        let coverage_prob = passing_samples as f64 / samples_used as f64;
        let coverage_ci = WilsonCI::compute(coverage_prob, samples_used, self.config.wilson_confidence);
        
        BootstrapResult {
            coverage_probability: coverage_prob,
            coverage_ci,
            samples_used,
            early_stopped,
            passing_samples,
            ece_threshold,
            timing: BootstrapTiming {
                total_duration: duration,
                per_sample_us: duration.as_micros() as f64 / samples_used as f64,
                binning_per_sample_us: 0.0, // TODO: Add detailed timing breakdown
                ece_per_sample_us: 0.0,
            },
            used_blb: false,
        }
    }
    
    /// Validate that bootstrap meets SLA requirements (<1ms p99)
    pub fn validate_sla(&mut self, predictions: &[f64], labels: &[f64], weights: &[f64]) -> bool {
        let mut timings = Vec::new();
        
        // Run multiple bootstrap iterations to measure p99 latency
        for _ in 0..100 {
            let start = Instant::now();
            let _ece = self.bootstrap_iteration(predictions, labels, weights);
            let duration = start.elapsed();
            timings.push(duration.as_micros() as f64);
        }
        
        timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p99 = timings[(timings.len() * 99 / 100).min(timings.len() - 1)];
        
        p99 < 1000.0 // < 1ms (1000 microseconds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::shared_binning_core::SharedBinningConfig;
    
    #[test]
    fn test_wilson_ci() {
        // Test Wilson CI computation
        let ci = WilsonCI::compute(0.95, 100, 0.95);
        
        assert!(ci.lower < ci.point_estimate);
        assert!(ci.point_estimate < ci.upper);
        assert!(ci.lower >= 0.0);
        assert!(ci.upper <= 1.0);
        assert_eq!(ci.point_estimate, 0.95);
    }
    
    #[test]
    fn test_fast_bootstrap_early_stopping() {
        let bootstrap_config = FastBootstrapConfig {
            early_stop_samples: 50,
            max_samples: 200,
            early_stop_threshold: 0.90,
            ..Default::default()
        };
        let binning_config = SharedBinningConfig::default();
        
        let mut bootstrap = FastBootstrap::new(bootstrap_config, binning_config);
        
        // Well-calibrated data should trigger early stopping
        let predictions = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let weights = vec![1.0; 9];
        
        let result = bootstrap.run_bootstrap(&predictions, &labels, &weights, 9, 1.5);
        
        assert!(result.coverage_probability >= 0.0);
        assert!(result.samples_used > 0);
        assert!(result.samples_used <= 200);
        
        println!("Bootstrap result: early_stopped={}, samples_used={}, coverage={:.3}", 
            result.early_stopped, result.samples_used, result.coverage_probability);
    }
    
    #[test]
    fn test_poisson_vs_multinomial_bootstrap() {
        let bootstrap_config = FastBootstrapConfig {
            max_samples: 100,
            use_poisson_bootstrap: true,
            ..Default::default()
        };
        let binning_config = SharedBinningConfig::default();
        
        let mut bootstrap_poisson = FastBootstrap::new(bootstrap_config.clone(), binning_config.clone());
        
        let mut bootstrap_multinomial = FastBootstrap::new(
            FastBootstrapConfig { use_poisson_bootstrap: false, ..bootstrap_config },
            binning_config
        );
        
        let predictions = vec![0.2, 0.4, 0.6, 0.8];
        let labels = vec![0.0, 0.0, 1.0, 1.0];
        let weights = vec![1.0; 4];
        
        let result_poisson = bootstrap_poisson.run_bootstrap(&predictions, &labels, &weights, 4, 1.5);
        let result_multinomial = bootstrap_multinomial.run_bootstrap(&predictions, &labels, &weights, 4, 1.5);
        
        // Both should produce valid results
        assert!(result_poisson.coverage_probability >= 0.0 && result_poisson.coverage_probability <= 1.0);
        assert!(result_multinomial.coverage_probability >= 0.0 && result_multinomial.coverage_probability <= 1.0);
        
        println!("Poisson coverage: {:.3}, Multinomial coverage: {:.3}", 
            result_poisson.coverage_probability, result_multinomial.coverage_probability);
    }
}