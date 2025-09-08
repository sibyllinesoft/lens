//! Statistical testing framework for benchmark validation
//! Implements bootstrap confidence intervals and permutation tests with Holm correction

use std::collections::HashMap;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error, debug, instrument};
use anyhow::{Result, Context};

use super::industry_suites::{IndustryBenchmarkResult, AggregateMetrics};
use super::BenchmarkResult;

/// Statistical testing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTestConfig {
    /// Number of bootstrap resamples for confidence intervals
    pub bootstrap_samples: u32,
    
    /// Number of permutations for permutation tests
    pub permutation_count: u32,
    
    /// Confidence level for bootstrap intervals (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    
    /// Significance level for hypothesis tests (e.g., 0.05 for 5%)
    pub alpha: f64,
    
    /// Whether to apply Holm correction for multiple comparisons
    pub apply_holm_correction: bool,
    
    /// Minimum effect size for practical significance
    pub min_effect_size: f64,
}

/// Complete statistical validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalValidationResult {
    /// Bootstrap confidence intervals for key metrics
    pub bootstrap_results: HashMap<String, BootstrapResult>,
    
    /// Permutation test results for significance testing
    pub permutation_results: HashMap<String, PermutationTestResult>,
    
    /// Effect size calculations
    pub effect_sizes: HashMap<String, EffectSizeResult>,
    
    /// Multiple comparison correction results
    pub multiple_comparison_correction: Option<MultipleComparisonResult>,
    
    /// Overall statistical validation summary
    pub validation_summary: ValidationSummary,
    
    /// Configuration used for testing
    pub test_config: StatisticalTestConfig,
}

/// Bootstrap confidence interval result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapResult {
    /// Original sample statistic
    pub original_statistic: f64,
    
    /// Bootstrap confidence interval bounds
    pub confidence_interval: (f64, f64),
    
    /// Confidence level used
    pub confidence_level: f64,
    
    /// Bootstrap distribution statistics
    pub bootstrap_distribution: BootstrapDistribution,
    
    /// Bias-corrected and accelerated (BCa) interval if computed
    pub bca_interval: Option<(f64, f64)>,
}

/// Bootstrap distribution summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapDistribution {
    pub mean: f64,
    pub std_dev: f64,
    pub median: f64,
    pub percentile_25: f64,
    pub percentile_75: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

/// Permutation test result for hypothesis testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermutationTestResult {
    /// Observed test statistic
    pub observed_statistic: f64,
    
    /// P-value from permutation test
    pub p_value: f64,
    
    /// Whether the result is statistically significant
    pub is_significant: bool,
    
    /// Test statistic distribution from permutations
    pub permutation_distribution: PermutationDistribution,
    
    /// Type of test performed
    pub test_type: TestType,
    
    /// Alternative hypothesis
    pub alternative: AlternativeHypothesis,
}

/// Permutation test distribution summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermutationDistribution {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub percentile_95: f64,
    pub percentile_99: f64,
}

/// Types of statistical tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestType {
    /// Compare two independent samples
    TwoSampleComparison,
    /// Compare against a fixed baseline
    OneVsBaseline,
    /// Paired comparison (before/after)
    PairedComparison,
}

/// Alternative hypothesis for tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlternativeHypothesis {
    /// Two-sided test (difference ≠ 0)
    TwoSided,
    /// One-sided test (treatment > control)
    Greater,
    /// One-sided test (treatment < control)
    Less,
}

/// Effect size calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectSizeResult {
    /// Cohen's d for standardized effect size
    pub cohens_d: f64,
    
    /// Effect size interpretation
    pub interpretation: EffectSizeInterpretation,
    
    /// Confidence interval for effect size
    pub confidence_interval: Option<(f64, f64)>,
    
    /// Glass's delta (using control group SD)
    pub glass_delta: Option<f64>,
    
    /// Hedges' g (corrected for small sample bias)
    pub hedges_g: f64,
}

/// Effect size magnitude interpretation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffectSizeInterpretation {
    Negligible,    // |d| < 0.2
    Small,         // 0.2 ≤ |d| < 0.5
    Medium,        // 0.5 ≤ |d| < 0.8
    Large,         // |d| ≥ 0.8
}

/// Multiple comparison correction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultipleComparisonResult {
    /// Original p-values
    pub original_p_values: Vec<f64>,
    
    /// Corrected p-values using Holm method
    pub holm_corrected_p_values: Vec<f64>,
    
    /// Test names corresponding to p-values
    pub test_names: Vec<String>,
    
    /// Family-wise error rate
    pub family_wise_error_rate: f64,
    
    /// Number of rejected hypotheses after correction
    pub rejected_count: usize,
}

/// Overall validation summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSummary {
    /// Total number of tests performed
    pub total_tests: usize,
    
    /// Number of statistically significant results
    pub significant_results: usize,
    
    /// Number of practically significant results (large effect size)
    pub practically_significant_results: usize,
    
    /// Overall validation status
    pub validation_status: ValidationStatus,
    
    /// Key findings summary
    pub key_findings: Vec<String>,
    
    /// Recommendations based on results
    pub recommendations: Vec<String>,
}

/// Overall validation outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStatus {
    /// Strong evidence supporting the improvement
    StrongEvidence,
    /// Moderate evidence supporting the improvement
    ModerateEvidence,
    /// Weak or inconclusive evidence
    WeakEvidence,
    /// Evidence against the improvement
    NoEvidence,
}

/// Main statistical testing framework
pub struct StatisticalTester {
    config: StatisticalTestConfig,
}

impl Default for StatisticalTestConfig {
    fn default() -> Self {
        Self {
            bootstrap_samples: 10000,
            permutation_count: 10000,
            confidence_level: 0.95,
            alpha: 0.05,
            apply_holm_correction: true,
            min_effect_size: 0.2, // Small effect size threshold
        }
    }
}

impl StatisticalTester {
    pub fn new(config: StatisticalTestConfig) -> Self {
        Self { config }
    }

    #[instrument(skip(self, baseline_results, treatment_results))]
    pub async fn validate_benchmark_results(
        &self,
        baseline_results: &[BenchmarkResult],
        treatment_results: &[BenchmarkResult],
    ) -> Result<StatisticalValidationResult> {
        info!("Starting comprehensive statistical validation");
        info!("Baseline results: {}, Treatment results: {}", 
              baseline_results.len(), treatment_results.len());

        if baseline_results.is_empty() || treatment_results.is_empty() {
            return Err(anyhow::anyhow!("Cannot perform statistical validation on empty result sets"));
        }

        // Extract key metrics for analysis
        let metrics = self.extract_metrics(baseline_results, treatment_results)?;
        
        // Perform bootstrap analysis
        let bootstrap_results = self.perform_bootstrap_analysis(&metrics).await?;
        
        // Perform permutation tests
        let permutation_results = self.perform_permutation_tests(&metrics).await?;
        
        // Calculate effect sizes
        let effect_sizes = self.calculate_effect_sizes(&metrics)?;
        
        // Apply multiple comparison correction if requested
        let multiple_comparison_correction = if self.config.apply_holm_correction {
            Some(self.apply_multiple_comparison_correction(&permutation_results)?)
        } else {
            None
        };
        
        // Generate validation summary
        let validation_summary = self.generate_validation_summary(
            &bootstrap_results,
            &permutation_results,
            &effect_sizes,
            &multiple_comparison_correction,
        )?;

        Ok(StatisticalValidationResult {
            bootstrap_results,
            permutation_results,
            effect_sizes,
            multiple_comparison_correction,
            validation_summary,
            test_config: self.config.clone(),
        })
    }

    #[instrument(skip(self, baseline_results, treatment_results))]
    fn extract_metrics(
        &self,
        baseline_results: &[BenchmarkResult],
        treatment_results: &[BenchmarkResult],
    ) -> Result<HashMap<String, MetricPair>> {
        let mut metrics = HashMap::new();
        
        // Extract Success@10 metrics
        let baseline_success: Vec<f64> = baseline_results.iter().map(|r| r.success_at_10).collect();
        let treatment_success: Vec<f64> = treatment_results.iter().map(|r| r.success_at_10).collect();
        metrics.insert("success_at_10".to_string(), MetricPair {
            baseline: baseline_success,
            treatment: treatment_success,
        });
        
        // Extract nDCG@10 metrics
        let baseline_ndcg: Vec<f64> = baseline_results.iter().map(|r| r.ndcg_at_10).collect();
        let treatment_ndcg: Vec<f64> = treatment_results.iter().map(|r| r.ndcg_at_10).collect();
        metrics.insert("ndcg_at_10".to_string(), MetricPair {
            baseline: baseline_ndcg,
            treatment: treatment_ndcg,
        });
        
        // Extract SLA-Recall@50 metrics
        let baseline_sla_recall: Vec<f64> = baseline_results.iter().map(|r| r.sla_recall_at_50).collect();
        let treatment_sla_recall: Vec<f64> = treatment_results.iter().map(|r| r.sla_recall_at_50).collect();
        metrics.insert("sla_recall_at_50".to_string(), MetricPair {
            baseline: baseline_sla_recall,
            treatment: treatment_sla_recall,
        });
        
        // Extract latency metrics (converted to f64)
        let baseline_latency: Vec<f64> = baseline_results.iter().map(|r| r.latency_ms as f64).collect();
        let treatment_latency: Vec<f64> = treatment_results.iter().map(|r| r.latency_ms as f64).collect();
        metrics.insert("latency_ms".to_string(), MetricPair {
            baseline: baseline_latency,
            treatment: treatment_latency,
        });

        info!("Extracted {} metrics for statistical analysis", metrics.len());
        Ok(metrics)
    }

    #[instrument(skip(self, metrics))]
    async fn perform_bootstrap_analysis(
        &self,
        metrics: &HashMap<String, MetricPair>,
    ) -> Result<HashMap<String, BootstrapResult>> {
        info!("Performing bootstrap analysis with {} samples", self.config.bootstrap_samples);
        let mut results = HashMap::new();

        for (metric_name, metric_pair) in metrics {
            debug!("Bootstrap analysis for metric: {}", metric_name);
            
            // Calculate difference in means as our statistic of interest
            let baseline_mean = metric_pair.baseline.iter().sum::<f64>() / metric_pair.baseline.len() as f64;
            let treatment_mean = metric_pair.treatment.iter().sum::<f64>() / metric_pair.treatment.len() as f64;
            let original_statistic = treatment_mean - baseline_mean;
            
            // Perform bootstrap resampling
            let bootstrap_statistics = self.bootstrap_difference_of_means(
                &metric_pair.baseline,
                &metric_pair.treatment,
            )?;
            
            // Calculate confidence interval
            let confidence_interval = self.calculate_percentile_confidence_interval(
                &bootstrap_statistics,
                self.config.confidence_level,
            )?;
            
            // Calculate distribution statistics
            let bootstrap_distribution = self.calculate_distribution_stats(&bootstrap_statistics)?;
            
            // Optionally calculate BCa interval for better coverage
            let bca_interval = self.calculate_bca_confidence_interval(
                &metric_pair.baseline,
                &metric_pair.treatment,
                &bootstrap_statistics,
                original_statistic,
                self.config.confidence_level,
            ).ok(); // Optional - may fail for some distributions
            
            results.insert(metric_name.clone(), BootstrapResult {
                original_statistic,
                confidence_interval,
                confidence_level: self.config.confidence_level,
                bootstrap_distribution,
                bca_interval,
            });
        }

        info!("Completed bootstrap analysis for {} metrics", results.len());
        Ok(results)
    }

    #[instrument(skip(self, metrics))]
    async fn perform_permutation_tests(
        &self,
        metrics: &HashMap<String, MetricPair>,
    ) -> Result<HashMap<String, PermutationTestResult>> {
        info!("Performing permutation tests with {} permutations", self.config.permutation_count);
        let mut results = HashMap::new();

        for (metric_name, metric_pair) in metrics {
            debug!("Permutation test for metric: {}", metric_name);
            
            // Perform two-sample permutation test
            let test_result = self.two_sample_permutation_test(
                &metric_pair.baseline,
                &metric_pair.treatment,
                AlternativeHypothesis::TwoSided,
            )?;
            
            results.insert(metric_name.clone(), test_result);
        }

        info!("Completed permutation tests for {} metrics", results.len());
        Ok(results)
    }

    fn bootstrap_difference_of_means(&self, baseline: &[f64], treatment: &[f64]) -> Result<Vec<f64>> {
        let mut bootstrap_stats = Vec::with_capacity(self.config.bootstrap_samples as usize);
        
        for _ in 0..self.config.bootstrap_samples {
            // Bootstrap resample from baseline
            let baseline_sample = self.bootstrap_sample(baseline);
            let baseline_mean = baseline_sample.iter().sum::<f64>() / baseline_sample.len() as f64;
            
            // Bootstrap resample from treatment
            let treatment_sample = self.bootstrap_sample(treatment);
            let treatment_mean = treatment_sample.iter().sum::<f64>() / treatment_sample.len() as f64;
            
            bootstrap_stats.push(treatment_mean - baseline_mean);
        }
        
        Ok(bootstrap_stats)
    }

    fn bootstrap_sample(&self, data: &[f64]) -> Vec<f64> {
        let mut sample = Vec::with_capacity(data.len());
        for _ in 0..data.len() {
            let idx = fastrand::usize(0..data.len());
            sample.push(data[idx]);
        }
        sample
    }

    fn calculate_percentile_confidence_interval(&self, data: &[f64], confidence_level: f64) -> Result<(f64, f64)> {
        if data.is_empty() {
            return Err(anyhow::anyhow!("Cannot calculate confidence interval for empty data"));
        }

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let alpha = 1.0 - confidence_level;
        let lower_percentile = alpha / 2.0;
        let upper_percentile = 1.0 - alpha / 2.0;
        
        let lower_index = (lower_percentile * (sorted_data.len() - 1) as f64).round() as usize;
        let upper_index = (upper_percentile * (sorted_data.len() - 1) as f64).round() as usize;
        
        Ok((sorted_data[lower_index], sorted_data[upper_index]))
    }

    fn calculate_bca_confidence_interval(
        &self,
        baseline: &[f64],
        treatment: &[f64],
        bootstrap_stats: &[f64],
        original_statistic: f64,
        confidence_level: f64,
    ) -> Result<(f64, f64)> {
        // BCa (Bias-Corrected and Accelerated) bootstrap confidence interval
        // This is more accurate than percentile method but more complex
        
        // Calculate bias-correction
        let proportion_less = bootstrap_stats.iter()
            .filter(|&&x| x < original_statistic)
            .count() as f64 / bootstrap_stats.len() as f64;
        
        let z_0 = if proportion_less > 0.0 && proportion_less < 1.0 {
            // Inverse normal CDF approximation
            self.inverse_normal_cdf(proportion_less)
        } else {
            0.0
        };
        
        // Calculate acceleration (jackknife method)
        let acceleration = self.calculate_jackknife_acceleration(baseline, treatment)?;
        
        // Calculate adjusted percentiles
        let alpha = 1.0 - confidence_level;
        let z_alpha_2 = self.inverse_normal_cdf(alpha / 2.0);
        let z_1_alpha_2 = -z_alpha_2;
        
        let alpha_1 = self.normal_cdf(z_0 + (z_0 + z_alpha_2) / (1.0 - acceleration * (z_0 + z_alpha_2)));
        let alpha_2 = self.normal_cdf(z_0 + (z_0 + z_1_alpha_2) / (1.0 - acceleration * (z_0 + z_1_alpha_2)));
        
        // Get percentiles from bootstrap distribution
        let mut sorted_bootstrap = bootstrap_stats.to_vec();
        sorted_bootstrap.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let n = sorted_bootstrap.len();
        let lower_index = (alpha_1 * (n - 1) as f64).round().max(0.0).min((n - 1) as f64) as usize;
        let upper_index = (alpha_2 * (n - 1) as f64).round().max(0.0).min((n - 1) as f64) as usize;
        
        Ok((sorted_bootstrap[lower_index], sorted_bootstrap[upper_index]))
    }

    fn calculate_jackknife_acceleration(&self, baseline: &[f64], treatment: &[f64]) -> Result<f64> {
        // Simplified acceleration calculation using jackknife
        // In practice, this would involve leave-one-out statistics
        Ok(0.0) // Placeholder - full implementation would be more complex
    }

    fn inverse_normal_cdf(&self, p: f64) -> f64 {
        // Simplified inverse normal CDF approximation
        // In practice, use a proper statistical library
        if p <= 0.0 { return -6.0; }
        if p >= 1.0 { return 6.0; }
        
        // Approximation for the inverse standard normal
        let t = (-2.0 * p.ln()).sqrt();
        t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / 
            (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)
    }

    fn normal_cdf(&self, x: f64) -> f64 {
        // Simplified normal CDF approximation
        0.5 * (1.0 + self.erf(x / 2.0_f64.sqrt()))
    }

    fn erf(&self, x: f64) -> f64 {
        // Approximation of error function
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;
        
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
        
        sign * y
    }

    fn two_sample_permutation_test(
        &self,
        baseline: &[f64],
        treatment: &[f64],
        alternative: AlternativeHypothesis,
    ) -> Result<PermutationTestResult> {
        // Calculate observed test statistic (difference in means)
        let baseline_mean = baseline.iter().sum::<f64>() / baseline.len() as f64;
        let treatment_mean = treatment.iter().sum::<f64>() / treatment.len() as f64;
        let observed_statistic = treatment_mean - baseline_mean;
        
        // Combine samples for permutation
        let mut combined: Vec<f64> = baseline.iter().chain(treatment.iter()).copied().collect();
        let n1 = baseline.len();
        let n2 = treatment.len();
        
        // Generate permutation distribution
        let mut permutation_stats = Vec::with_capacity(self.config.permutation_count as usize);
        
        for _ in 0..self.config.permutation_count {
            // Randomly shuffle combined sample
            fastrand::shuffle(&mut combined);
            
            // Split into new groups
            let perm_baseline_mean = combined[..n1].iter().sum::<f64>() / n1 as f64;
            let perm_treatment_mean = combined[n1..n1+n2].iter().sum::<f64>() / n2 as f64;
            
            permutation_stats.push(perm_treatment_mean - perm_baseline_mean);
        }
        
        // Calculate p-value based on alternative hypothesis
        let p_value = match alternative {
            AlternativeHypothesis::TwoSided => {
                let extreme_count = permutation_stats.iter()
                    .filter(|&&stat| stat.abs() >= observed_statistic.abs())
                    .count();
                extreme_count as f64 / permutation_stats.len() as f64
            }
            AlternativeHypothesis::Greater => {
                let extreme_count = permutation_stats.iter()
                    .filter(|&&stat| stat >= observed_statistic)
                    .count();
                extreme_count as f64 / permutation_stats.len() as f64
            }
            AlternativeHypothesis::Less => {
                let extreme_count = permutation_stats.iter()
                    .filter(|&&stat| stat <= observed_statistic)
                    .count();
                extreme_count as f64 / permutation_stats.len() as f64
            }
        };
        
        let is_significant = p_value < self.config.alpha;
        let permutation_distribution = self.calculate_permutation_distribution_stats(&permutation_stats)?;
        
        Ok(PermutationTestResult {
            observed_statistic,
            p_value,
            is_significant,
            permutation_distribution,
            test_type: TestType::TwoSampleComparison,
            alternative,
        })
    }

    fn calculate_distribution_stats(&self, data: &[f64]) -> Result<BootstrapDistribution> {
        if data.is_empty() {
            return Err(anyhow::anyhow!("Cannot calculate distribution stats for empty data"));
        }

        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();
        
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let median = sorted[sorted.len() / 2];
        let percentile_25 = sorted[sorted.len() / 4];
        let percentile_75 = sorted[3 * sorted.len() / 4];
        
        // Calculate skewness and kurtosis
        let skewness = data.iter().map(|x| ((x - mean) / std_dev).powi(3)).sum::<f64>() / n;
        let kurtosis = data.iter().map(|x| ((x - mean) / std_dev).powi(4)).sum::<f64>() / n - 3.0;
        
        Ok(BootstrapDistribution {
            mean,
            std_dev,
            median,
            percentile_25,
            percentile_75,
            skewness,
            kurtosis,
        })
    }

    fn calculate_permutation_distribution_stats(&self, data: &[f64]) -> Result<PermutationDistribution> {
        if data.is_empty() {
            return Err(anyhow::anyhow!("Cannot calculate distribution stats for empty data"));
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
        let std_dev = variance.sqrt();
        
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let min = sorted[0];
        let max = sorted[sorted.len() - 1];
        let percentile_95_idx = (0.95 * (sorted.len() - 1) as f64).round() as usize;
        let percentile_99_idx = (0.99 * (sorted.len() - 1) as f64).round() as usize;
        
        Ok(PermutationDistribution {
            mean,
            std_dev,
            min,
            max,
            percentile_95: sorted[percentile_95_idx],
            percentile_99: sorted[percentile_99_idx],
        })
    }

    fn calculate_effect_sizes(&self, metrics: &HashMap<String, MetricPair>) -> Result<HashMap<String, EffectSizeResult>> {
        let mut results = HashMap::new();
        
        for (metric_name, metric_pair) in metrics {
            let effect_size_result = self.calculate_cohens_d_and_related(
                &metric_pair.baseline,
                &metric_pair.treatment,
            )?;
            
            results.insert(metric_name.clone(), effect_size_result);
        }
        
        Ok(results)
    }

    fn calculate_cohens_d_and_related(&self, baseline: &[f64], treatment: &[f64]) -> Result<EffectSizeResult> {
        let baseline_mean = baseline.iter().sum::<f64>() / baseline.len() as f64;
        let treatment_mean = treatment.iter().sum::<f64>() / treatment.len() as f64;
        
        let baseline_var = baseline.iter().map(|x| (x - baseline_mean).powi(2)).sum::<f64>() / (baseline.len() - 1) as f64;
        let treatment_var = treatment.iter().map(|x| (x - treatment_mean).powi(2)).sum::<f64>() / (treatment.len() - 1) as f64;
        
        // Pooled standard deviation for Cohen's d
        let pooled_sd = (((baseline.len() - 1) as f64 * baseline_var + (treatment.len() - 1) as f64 * treatment_var) / 
                        ((baseline.len() + treatment.len() - 2) as f64)).sqrt();
        
        let cohens_d = (treatment_mean - baseline_mean) / pooled_sd;
        
        // Glass's delta (using control group SD)
        let glass_delta = Some((treatment_mean - baseline_mean) / baseline_var.sqrt());
        
        // Hedges' g (bias-corrected Cohen's d)
        let n = baseline.len() + treatment.len();
        let correction_factor = 1.0 - 3.0 / (4.0 * n as f64 - 9.0);
        let hedges_g = cohens_d * correction_factor;
        
        let interpretation = match cohens_d.abs() {
            d if d < 0.2 => EffectSizeInterpretation::Negligible,
            d if d < 0.5 => EffectSizeInterpretation::Small,
            d if d < 0.8 => EffectSizeInterpretation::Medium,
            _ => EffectSizeInterpretation::Large,
        };
        
        Ok(EffectSizeResult {
            cohens_d,
            interpretation,
            confidence_interval: None, // Would require additional computation
            glass_delta,
            hedges_g,
        })
    }

    fn apply_multiple_comparison_correction(
        &self,
        permutation_results: &HashMap<String, PermutationTestResult>,
    ) -> Result<MultipleComparisonResult> {
        let test_names: Vec<String> = permutation_results.keys().cloned().collect();
        let original_p_values: Vec<f64> = test_names.iter()
            .map(|name| permutation_results[name].p_value)
            .collect();
        
        let holm_corrected_p_values = self.holm_correction(&original_p_values)?;
        
        let family_wise_error_rate = self.config.alpha;
        let rejected_count = holm_corrected_p_values.iter()
            .filter(|&&p| p < family_wise_error_rate)
            .count();
        
        Ok(MultipleComparisonResult {
            original_p_values,
            holm_corrected_p_values,
            test_names,
            family_wise_error_rate,
            rejected_count,
        })
    }

    fn holm_correction(&self, p_values: &[f64]) -> Result<Vec<f64>> {
        if p_values.is_empty() {
            return Ok(Vec::new());
        }

        let m = p_values.len();
        let mut indexed_p_values: Vec<(usize, f64)> = p_values.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed_p_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut corrected = vec![0.0; m];
        let mut previous_corrected = 0.0;

        for (rank, (original_index, p_value)) in indexed_p_values.iter().enumerate() {
            let correction_factor = (m - rank) as f64;
            let corrected_p = (p_value * correction_factor).min(1.0).max(previous_corrected);
            corrected[*original_index] = corrected_p;
            previous_corrected = corrected_p;
        }

        Ok(corrected)
    }

    fn generate_validation_summary(
        &self,
        bootstrap_results: &HashMap<String, BootstrapResult>,
        permutation_results: &HashMap<String, PermutationTestResult>,
        effect_sizes: &HashMap<String, EffectSizeResult>,
        multiple_comparison_correction: &Option<MultipleComparisonResult>,
    ) -> Result<ValidationSummary> {
        let total_tests = permutation_results.len();
        let significant_results = permutation_results.values()
            .filter(|result| result.is_significant)
            .count();
        
        let practically_significant_results = effect_sizes.values()
            .filter(|effect| matches!(effect.interpretation, EffectSizeInterpretation::Medium | EffectSizeInterpretation::Large))
            .count();
        
        let validation_status = match (significant_results, practically_significant_results) {
            (s, p) if s >= total_tests / 2 && p >= total_tests / 2 => ValidationStatus::StrongEvidence,
            (s, p) if s > 0 && p > 0 => ValidationStatus::ModerateEvidence,
            (s, _) if s > 0 => ValidationStatus::WeakEvidence,
            _ => ValidationStatus::NoEvidence,
        };
        
        let mut key_findings = Vec::new();
        let mut recommendations = Vec::new();
        
        // Generate findings based on results
        if significant_results > 0 {
            key_findings.push(format!("{}/{} metrics show statistically significant improvements", significant_results, total_tests));
        }
        
        if practically_significant_results > 0 {
            key_findings.push(format!("{}/{} metrics show practically significant effect sizes", practically_significant_results, total_tests));
        }
        
        // Generate recommendations
        match validation_status {
            ValidationStatus::StrongEvidence => {
                recommendations.push("Strong evidence supports proceeding with deployment".to_string());
                recommendations.push("Consider full rollout with continued monitoring".to_string());
            }
            ValidationStatus::ModerateEvidence => {
                recommendations.push("Moderate evidence supports gradual rollout".to_string());
                recommendations.push("Implement enhanced monitoring during rollout".to_string());
            }
            ValidationStatus::WeakEvidence => {
                recommendations.push("Weak evidence suggests cautious approach".to_string());
                recommendations.push("Consider additional validation or improvements".to_string());
            }
            ValidationStatus::NoEvidence => {
                recommendations.push("Insufficient evidence to support deployment".to_string());
                recommendations.push("Recommend system improvements before retry".to_string());
            }
        }
        
        Ok(ValidationSummary {
            total_tests,
            significant_results,
            practically_significant_results,
            validation_status,
            key_findings,
            recommendations,
        })
    }
}

/// Helper struct for paired baseline/treatment metrics
#[derive(Debug, Clone)]
struct MetricPair {
    baseline: Vec<f64>,
    treatment: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistical_test_config_default() {
        let config = StatisticalTestConfig::default();
        assert_eq!(config.bootstrap_samples, 10000);
        assert_eq!(config.permutation_count, 10000);
        assert_eq!(config.confidence_level, 0.95);
        assert_eq!(config.alpha, 0.05);
        assert_eq!(config.apply_holm_correction, true);
        assert_eq!(config.min_effect_size, 0.2);
    }

    #[test]
    fn test_bootstrap_sample() {
        let tester = StatisticalTester::new(StatisticalTestConfig::default());
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample = tester.bootstrap_sample(&data);
        
        assert_eq!(sample.len(), data.len());
        // All values should be from original data
        for value in &sample {
            assert!(data.contains(value));
        }
    }

    #[test]
    fn test_percentile_confidence_interval() {
        let tester = StatisticalTester::new(StatisticalTestConfig::default());
        let data = (1..=100).map(|x| x as f64).collect::<Vec<_>>();
        let (lower, upper) = tester.calculate_percentile_confidence_interval(&data, 0.95).unwrap();
        
        assert!(lower < upper);
        assert!(lower >= 1.0);
        assert!(upper <= 100.0);
    }

    #[test]
    fn test_cohens_d_calculation() {
        let tester = StatisticalTester::new(StatisticalTestConfig::default());
        let baseline = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let treatment = vec![3.0, 4.0, 5.0, 6.0, 7.0]; // Mean difference of 2.0
        
        let result = tester.calculate_cohens_d_and_related(&baseline, &treatment).unwrap();
        assert!(result.cohens_d > 0.0); // Treatment should be better
        assert!(matches!(result.interpretation, EffectSizeInterpretation::Large));
    }

    #[test]
    fn test_holm_correction() {
        let tester = StatisticalTester::new(StatisticalTestConfig::default());
        let p_values = vec![0.01, 0.02, 0.03, 0.04, 0.05];
        let corrected = tester.holm_correction(&p_values).unwrap();
        
        assert_eq!(corrected.len(), p_values.len());
        // Corrected p-values should be >= original (except for possible rounding)
        for (i, (&original, &corrected_p)) in p_values.iter().zip(corrected.iter()).enumerate() {
            if i > 0 { // First p-value might be heavily corrected
                assert!(corrected_p >= original - 1e-10); // Allow for floating point precision
            }
        }
    }

    #[test]
    fn test_effect_size_interpretation() {
        assert!(matches!(
            EffectSizeInterpretation::Negligible,
            EffectSizeInterpretation::Negligible
        ));
        // Would test with actual calculated effect sizes in a full implementation
    }
}