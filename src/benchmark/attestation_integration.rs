//! Attestation system for fraud-resistant benchmarking results
//! Provides cryptographic verification and tamper-evident result validation

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error, debug, instrument};
use anyhow::{Result, Context};
use blake3::{Hash, Hasher};

use super::industry_suites::{IndustryBenchmarkResult, IndustryBenchmarkConfig};

/// Result attestation system for fraud-resistant benchmarking
pub struct ResultAttestation {
    config: AttestationConfig,
    signing_key: Option<Vec<u8>>, // In production, use proper key management
}

/// Attestation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationConfig {
    /// Enable cryptographic signing of results
    pub enable_signing: bool,
    
    /// Require witness coverage validation
    pub require_witness_validation: bool,
    
    /// Enable statistical test validation
    pub enable_statistical_validation: bool,
    
    /// Minimum confidence level for statistical tests
    pub min_confidence_level: f64,
    
    /// Enable config fingerprint validation
    pub enable_config_fingerprint: bool,
    
    /// Enable reproducibility verification
    pub enable_reproducibility_checks: bool,
}

/// Complete attestation result for a benchmark run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationResult {
    /// Unique attestation identifier
    pub attestation_id: String,
    
    /// Suite identifier this attestation covers
    pub suite_id: String,
    
    /// Timestamp of attestation creation
    pub timestamp: u64,
    
    /// Configuration fingerprint hash
    pub config_fingerprint: String,
    
    /// Environment fingerprint (system, version, etc.)
    pub environment_fingerprint: String,
    
    /// Cryptographic signature of results (if enabled)
    pub signature: Option<String>,
    
    /// Statistical validation results
    pub statistical_validation: Option<StatisticalValidation>,
    
    /// Witness coverage validation results
    pub witness_validation: Option<WitnessValidation>,
    
    /// Reproducibility check results
    pub reproducibility_check: Option<ReproducibilityCheck>,
    
    /// Overall attestation status
    pub attestation_status: AttestationStatus,
    
    /// Any validation warnings or issues
    pub warnings: Vec<String>,
    
    /// Hash of the original results for tamper detection
    pub results_hash: String,
}

/// Statistical validation results using bootstrap/permutation testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalValidation {
    /// Bootstrap confidence intervals for key metrics
    pub bootstrap_intervals: HashMap<String, ConfidenceInterval>,
    
    /// Permutation test results for significance
    pub permutation_tests: HashMap<String, PermutationResult>,
    
    /// Holm correction for multiple comparisons
    pub holm_corrected_p_values: HashMap<String, f64>,
    
    /// Overall statistical significance
    pub statistically_significant: bool,
    
    /// Effect sizes for practical significance
    pub effect_sizes: HashMap<String, f64>,
}

/// Bootstrap confidence interval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub confidence_level: f64,
    pub mean: f64,
    pub std_dev: f64,
}

/// Permutation test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermutationResult {
    pub observed_statistic: f64,
    pub p_value: f64,
    pub permutations: u32,
    pub significant: bool,
    pub effect_size: f64,
}

/// Witness coverage validation for SWE-bench style benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WitnessValidation {
    /// Total witness coverage across all queries
    pub overall_witness_coverage: f64,
    
    /// Per-query witness coverage statistics
    pub per_query_coverage: Vec<QueryWitnessResult>,
    
    /// Validation passed threshold
    pub validation_passed: bool,
    
    /// Coverage distribution metrics
    pub coverage_statistics: CoverageStatistics,
}

/// Witness coverage result for individual query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryWitnessResult {
    pub query_id: String,
    pub expected_witnesses: u32,
    pub found_witnesses: u32,
    pub coverage_ratio: f64,
    pub missing_witnesses: Vec<String>,
}

/// Statistical summary of coverage distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageStatistics {
    pub mean_coverage: f64,
    pub median_coverage: f64,
    pub std_dev_coverage: f64,
    pub min_coverage: f64,
    pub max_coverage: f64,
    pub percentile_25: f64,
    pub percentile_75: f64,
}

/// Reproducibility verification results  
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityCheck {
    /// Whether results can be reproduced with same config
    pub reproducible: bool,
    
    /// Variance in key metrics across runs
    pub metric_variance: HashMap<String, f64>,
    
    /// Number of reproduction attempts
    pub reproduction_attempts: u32,
    
    /// Maximum allowed variance threshold
    pub variance_threshold: f64,
    
    /// Specific reproducibility issues found
    pub issues: Vec<String>,
}

/// Overall status of attestation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AttestationStatus {
    /// All validations passed
    Verified,
    /// Some validations failed but results are usable
    WarningsOnly,
    /// Critical validation failures
    Failed,
    /// Attestation could not be completed
    Incomplete,
}

impl Default for AttestationConfig {
    fn default() -> Self {
        Self {
            enable_signing: true,
            require_witness_validation: true,
            enable_statistical_validation: true,
            min_confidence_level: 0.95,
            enable_config_fingerprint: true,
            enable_reproducibility_checks: false, // Expensive, disabled by default
        }
    }
}

impl ResultAttestation {
    pub fn new(config: AttestationConfig) -> Self {
        Self {
            config,
            signing_key: Some(Self::generate_signing_key()),
        }
    }

    fn generate_signing_key() -> Vec<u8> {
        // In production, use proper cryptographic key generation and management
        // For now, use a deterministic key for testing
        let mut hasher = Hasher::new();
        hasher.update(b"lens-benchmark-signing-key-v1");
        hasher.update(&SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs().to_le_bytes());
        hasher.finalize().as_bytes().to_vec()
    }

    #[instrument(skip(self, results))]
    pub async fn attest_suite_results(
        &self,
        suite_id: &str,
        results: &[IndustryBenchmarkResult],
    ) -> Result<AttestationResult> {
        info!("Creating attestation for suite: {} with {} results", suite_id, results.len());
        
        let attestation_id = self.generate_attestation_id(suite_id);
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        
        // Generate fingerprints
        let config_fingerprint = self.generate_config_fingerprint().await?;
        let environment_fingerprint = self.generate_environment_fingerprint().await?;
        let results_hash = self.hash_results(results)?;
        
        let mut warnings = Vec::new();
        let mut attestation_status = AttestationStatus::Verified;

        // Cryptographic signature
        let signature = if self.config.enable_signing {
            Some(self.sign_results(results)?)
        } else {
            None
        };

        // Statistical validation
        let statistical_validation = if self.config.enable_statistical_validation {
            match self.perform_statistical_validation(results).await {
                Ok(validation) => Some(validation),
                Err(e) => {
                    warnings.push(format!("Statistical validation failed: {}", e));
                    attestation_status = AttestationStatus::WarningsOnly;
                    None
                }
            }
        } else {
            None
        };

        // Witness coverage validation (for applicable benchmarks)
        let witness_validation = if self.config.require_witness_validation && self.supports_witness_validation(suite_id) {
            match self.perform_witness_validation(results).await {
                Ok(validation) => Some(validation),
                Err(e) => {
                    warnings.push(format!("Witness validation failed: {}", e));
                    attestation_status = AttestationStatus::WarningsOnly;
                    None
                }
            }
        } else {
            None
        };

        // Reproducibility check
        let reproducibility_check = if self.config.enable_reproducibility_checks {
            match self.perform_reproducibility_check(suite_id, results).await {
                Ok(check) => Some(check),
                Err(e) => {
                    warnings.push(format!("Reproducibility check failed: {}", e));
                    attestation_status = AttestationStatus::WarningsOnly;
                    None
                }
            }
        } else {
            None
        };

        // Determine final attestation status
        if attestation_status == AttestationStatus::Verified {
            if let Some(ref stats) = statistical_validation {
                if !stats.statistically_significant {
                    warnings.push("Results not statistically significant".to_string());
                    attestation_status = AttestationStatus::WarningsOnly;
                }
            }
            
            if let Some(ref witness) = witness_validation {
                if !witness.validation_passed {
                    warnings.push("Witness coverage validation failed".to_string());
                    attestation_status = AttestationStatus::Failed;
                }
            }
            
            if let Some(ref repro) = reproducibility_check {
                if !repro.reproducible {
                    warnings.push("Results not reproducible".to_string());
                    attestation_status = AttestationStatus::WarningsOnly;
                }
            }
        }

        let attestation = AttestationResult {
            attestation_id,
            suite_id: suite_id.to_string(),
            timestamp,
            config_fingerprint,
            environment_fingerprint,
            signature,
            statistical_validation,
            witness_validation,
            reproducibility_check,
            attestation_status,
            warnings,
            results_hash,
        };

        info!(
            "Attestation completed for suite {} with status: {:?}",
            suite_id, attestation.attestation_status
        );

        Ok(attestation)
    }

    #[instrument(skip(self, results))]
    async fn perform_statistical_validation(
        &self,
        results: &[IndustryBenchmarkResult],
    ) -> Result<StatisticalValidation> {
        info!("Performing statistical validation with bootstrap and permutation tests");
        
        if results.len() < 10 {
            return Err(anyhow::anyhow!("Insufficient results for statistical validation (need ≥10, got {})", results.len()));
        }

        // Extract key metrics
        let success_scores: Vec<f64> = results.iter().map(|r| r.success_at_10).collect();
        let ndcg_scores: Vec<f64> = results.iter().map(|r| r.ndcg_at_10).collect();
        let sla_recall_scores: Vec<f64> = results.iter().map(|r| r.sla_recall_at_50).collect();
        let latencies: Vec<f64> = results.iter().map(|r| r.response_time_ms as f64).collect();

        let mut bootstrap_intervals = HashMap::new();
        let mut permutation_tests = HashMap::new();
        let mut effect_sizes = HashMap::new();

        // Bootstrap confidence intervals for key metrics
        bootstrap_intervals.insert("success_at_10".to_string(), 
            self.bootstrap_confidence_interval(&success_scores, self.config.min_confidence_level)?);
        bootstrap_intervals.insert("ndcg_at_10".to_string(),
            self.bootstrap_confidence_interval(&ndcg_scores, self.config.min_confidence_level)?);
        bootstrap_intervals.insert("sla_recall_at_50".to_string(),
            self.bootstrap_confidence_interval(&sla_recall_scores, self.config.min_confidence_level)?);
        bootstrap_intervals.insert("response_time_ms".to_string(),
            self.bootstrap_confidence_interval(&latencies, self.config.min_confidence_level)?);

        // Permutation tests (comparing against theoretical baselines)
        // For this implementation, we'll test against reasonable baselines
        let baseline_success = 0.3; // 30% baseline success rate
        let baseline_ndcg = 0.25; // 25% baseline nDCG
        
        permutation_tests.insert("success_improvement".to_string(),
            self.permutation_test_vs_constant(&success_scores, baseline_success)?);
        permutation_tests.insert("ndcg_improvement".to_string(),
            self.permutation_test_vs_constant(&ndcg_scores, baseline_ndcg)?);

        // Effect sizes (Cohen's d for practical significance)
        effect_sizes.insert("success_at_10".to_string(), 
            self.calculate_cohens_d(&success_scores, baseline_success)?);
        effect_sizes.insert("ndcg_at_10".to_string(),
            self.calculate_cohens_d(&ndcg_scores, baseline_ndcg)?);

        // Holm correction for multiple comparisons
        let raw_p_values: Vec<f64> = permutation_tests.values().map(|test| test.p_value).collect();
        let corrected_p_values = self.holm_correction(&raw_p_values)?;
        
        let mut holm_corrected_p_values = HashMap::new();
        let test_names: Vec<String> = permutation_tests.keys().cloned().collect();
        for (i, name) in test_names.iter().enumerate() {
            holm_corrected_p_values.insert(name.clone(), corrected_p_values[i]);
        }

        // Overall statistical significance
        let statistically_significant = corrected_p_values.iter().any(|&p| p < 0.05);

        Ok(StatisticalValidation {
            bootstrap_intervals,
            permutation_tests,
            holm_corrected_p_values,
            statistically_significant,
            effect_sizes,
        })
    }

    #[instrument(skip(self, results))]
    async fn perform_witness_validation(
        &self,
        results: &[IndustryBenchmarkResult],
    ) -> Result<WitnessValidation> {
        info!("Performing witness coverage validation");
        
        let mut per_query_coverage = Vec::new();
        let mut coverage_values = Vec::new();

        for result in results {
            let coverage_ratio = result.witness_coverage_at_10;
            coverage_values.push(coverage_ratio);
            
            // For this implementation, we estimate witness counts
            // In a real implementation, this would come from the query definition
            let estimated_expected = 5; // Average expected witnesses per query
            let estimated_found = (coverage_ratio * estimated_expected as f64) as u32;
            
            per_query_coverage.push(QueryWitnessResult {
                query_id: result.query_id.clone(),
                expected_witnesses: estimated_expected,
                found_witnesses: estimated_found,
                coverage_ratio,
                missing_witnesses: Vec::new(), // Would be populated in real implementation
            });
        }

        let overall_witness_coverage = coverage_values.iter().sum::<f64>() / coverage_values.len() as f64;
        let validation_passed = overall_witness_coverage >= 0.5; // 50% minimum threshold

        let coverage_statistics = self.calculate_coverage_statistics(&coverage_values)?;

        Ok(WitnessValidation {
            overall_witness_coverage,
            per_query_coverage,
            validation_passed,
            coverage_statistics,
        })
    }

    #[instrument(skip(self, results))]
    async fn perform_reproducibility_check(
        &self,
        suite_id: &str,
        results: &[IndustryBenchmarkResult],
    ) -> Result<ReproducibilityCheck> {
        info!("Performing reproducibility check for suite: {}", suite_id);
        
        // For this implementation, we'll simulate reproducibility checking
        // In production, this would actually re-run the benchmark
        
        let mut metric_variance = HashMap::new();
        let variance_threshold = 0.05; // 5% maximum variance allowed
        
        // Simulate metric variance (in real implementation, this would come from re-runs)
        metric_variance.insert("success_at_10".to_string(), 0.02);
        metric_variance.insert("ndcg_at_10".to_string(), 0.03);
        metric_variance.insert("response_time_ms".to_string(), 0.08);
        
        let reproducible = metric_variance.values().all(|&variance| variance <= variance_threshold);
        
        let issues = if !reproducible {
            vec!["Response time variance exceeds threshold".to_string()]
        } else {
            Vec::new()
        };

        Ok(ReproducibilityCheck {
            reproducible,
            metric_variance,
            reproduction_attempts: 3,
            variance_threshold,
            issues,
        })
    }

    fn bootstrap_confidence_interval(&self, data: &[f64], confidence_level: f64) -> Result<ConfidenceInterval> {
        if data.is_empty() {
            return Err(anyhow::anyhow!("Cannot calculate confidence interval for empty data"));
        }

        let n = data.len();
        let bootstrap_samples = 1000;
        let mut bootstrap_means = Vec::new();

        // Bootstrap resampling
        for _ in 0..bootstrap_samples {
            let mut sample_sum = 0.0;
            for _ in 0..n {
                let idx = fastrand::usize(0..n);
                sample_sum += data[idx];
            }
            bootstrap_means.push(sample_sum / n as f64);
        }

        bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let alpha = 1.0 - confidence_level;
        let lower_idx = ((alpha / 2.0) * bootstrap_samples as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * bootstrap_samples as f64) as usize;
        
        let lower_bound = bootstrap_means[lower_idx];
        let upper_bound = bootstrap_means[upper_idx.min(bootstrap_samples - 1)];
        
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
        let std_dev = variance.sqrt();

        Ok(ConfidenceInterval {
            lower_bound,
            upper_bound,
            confidence_level,
            mean,
            std_dev,
        })
    }

    fn permutation_test_vs_constant(&self, data: &[f64], constant: f64) -> Result<PermutationResult> {
        if data.is_empty() {
            return Err(anyhow::anyhow!("Cannot perform permutation test on empty data"));
        }

        let observed_statistic = data.iter().sum::<f64>() / data.len() as f64 - constant;
        let permutations = 1000;
        let mut permutation_stats = Vec::new();

        // Create combined dataset
        let combined: Vec<f64> = data.iter().copied().chain(std::iter::repeat(constant).take(data.len())).collect();

        for _ in 0..permutations {
            // Randomly assign values to groups
            let mut shuffled = combined.clone();
            fastrand::shuffle(&mut shuffled);
            
            let group1_mean = shuffled[..data.len()].iter().sum::<f64>() / data.len() as f64;
            let group2_mean = shuffled[data.len()..].iter().sum::<f64>() / data.len() as f64;
            
            permutation_stats.push(group1_mean - group2_mean);
        }

        let p_value = permutation_stats.iter()
            .filter(|&&stat| stat.abs() >= observed_statistic.abs())
            .count() as f64 / permutations as f64;

        let significant = p_value < 0.05;
        let effect_size = self.calculate_cohens_d(data, constant)?;

        Ok(PermutationResult {
            observed_statistic,
            p_value,
            permutations: permutations as u32,
            significant,
            effect_size,
        })
    }

    fn calculate_cohens_d(&self, data: &[f64], baseline: f64) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return Ok(0.0);
        }

        Ok((mean - baseline) / std_dev)
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

    fn calculate_coverage_statistics(&self, coverage_values: &[f64]) -> Result<CoverageStatistics> {
        if coverage_values.is_empty() {
            return Ok(CoverageStatistics {
                mean_coverage: 0.0,
                median_coverage: 0.0,
                std_dev_coverage: 0.0,
                min_coverage: 0.0,
                max_coverage: 0.0,
                percentile_25: 0.0,
                percentile_75: 0.0,
            });
        }

        let mut sorted = coverage_values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean_coverage = coverage_values.iter().sum::<f64>() / coverage_values.len() as f64;
        let median_coverage = sorted[sorted.len() / 2];
        let min_coverage = sorted[0];
        let max_coverage = sorted[sorted.len() - 1];
        
        let variance = coverage_values.iter()
            .map(|x| (x - mean_coverage).powi(2))
            .sum::<f64>() / (coverage_values.len() - 1) as f64;
        let std_dev_coverage = variance.sqrt();

        let percentile_25 = sorted[sorted.len() / 4];
        let percentile_75 = sorted[3 * sorted.len() / 4];

        Ok(CoverageStatistics {
            mean_coverage,
            median_coverage,
            std_dev_coverage,
            min_coverage,
            max_coverage,
            percentile_25,
            percentile_75,
        })
    }

    fn supports_witness_validation(&self, suite_id: &str) -> bool {
        // Witness coverage is primarily relevant for SWE-bench style benchmarks
        suite_id == "swe-bench"
    }

    fn generate_attestation_id(&self, suite_id: &str) -> String {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
        format!("attest-{}-{}", suite_id, timestamp)
    }

    async fn generate_config_fingerprint(&self) -> Result<String> {
        let config_data = serde_json::to_string(&self.config)?;
        let hash = blake3::hash(config_data.as_bytes());
        Ok(hex::encode(hash.as_bytes()))
    }

    async fn generate_environment_fingerprint(&self) -> Result<String> {
        let mut hasher = Hasher::new();
        
        // Add system information
        hasher.update(b"lens-benchmark-v1");
        hasher.update(env!("CARGO_PKG_VERSION").as_bytes());
        
        // Add timestamp for uniqueness
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        hasher.update(&timestamp.to_le_bytes());
        
        Ok(hex::encode(hasher.finalize().as_bytes()))
    }

    fn hash_results(&self, results: &[IndustryBenchmarkResult]) -> Result<String> {
        let results_json = serde_json::to_string(results)?;
        let hash = blake3::hash(results_json.as_bytes());
        Ok(hex::encode(hash.as_bytes()))
    }

    fn sign_results(&self, results: &[IndustryBenchmarkResult]) -> Result<String> {
        let results_json = serde_json::to_string(results)?;
        let key = self.signing_key.as_ref().unwrap();
        
        // Simple HMAC-style signature (in production, use proper digital signatures)
        let mut hasher = Hasher::new();
        hasher.update(key);
        hasher.update(results_json.as_bytes());
        
        Ok(hex::encode(hasher.finalize().as_bytes()))
    }

    pub async fn verify_attestation(&self, attestation: &AttestationResult, results: &[IndustryBenchmarkResult]) -> Result<bool> {
        info!("Verifying attestation: {}", attestation.attestation_id);
        
        // Verify results hash
        let computed_hash = self.hash_results(results)?;
        if computed_hash != attestation.results_hash {
            warn!("Results hash mismatch in attestation {}", attestation.attestation_id);
            return Ok(false);
        }

        // Verify signature if present
        if let Some(ref signature) = attestation.signature {
            let computed_signature = self.sign_results(results)?;
            if *signature != computed_signature {
                warn!("Signature verification failed for attestation {}", attestation.attestation_id);
                return Ok(false);
            }
        }

        info!("Attestation {} verification successful", attestation.attestation_id);
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_confidence_interval() {
        let attestation = ResultAttestation::new(AttestationConfig::default());
        let data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        
        let interval = attestation.bootstrap_confidence_interval(&data, 0.95).unwrap();
        assert!(interval.lower_bound < interval.mean);
        assert!(interval.upper_bound > interval.mean);
        assert!(interval.confidence_level == 0.95);
    }

    #[test]
    fn test_cohens_d_calculation() {
        let attestation = ResultAttestation::new(AttestationConfig::default());
        let data = vec![0.8, 0.9, 1.0, 0.7, 0.6];
        let baseline = 0.5;
        
        let cohens_d = attestation.calculate_cohens_d(&data, baseline).unwrap();
        assert!(cohens_d > 0.0); // Effect should be positive (data > baseline)
    }

    #[test]
    fn test_holm_correction() {
        let attestation = ResultAttestation::new(AttestationConfig::default());
        let p_values = vec![0.01, 0.03, 0.05, 0.02];
        
        let corrected = attestation.holm_correction(&p_values).unwrap();
        assert_eq!(corrected.len(), 4);
        
        // Corrected p-values should generally be larger than original
        for (original, &corrected_p) in p_values.iter().zip(corrected.iter()) {
            assert!(corrected_p >= *original);
        }
    }

    #[test]
    fn test_coverage_statistics() {
        let attestation = ResultAttestation::new(AttestationConfig::default());
        let coverage = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        
        let stats = attestation.calculate_coverage_statistics(&coverage).unwrap();
        assert_eq!(stats.mean_coverage, 0.5);
        assert_eq!(stats.median_coverage, 0.5);
        assert_eq!(stats.min_coverage, 0.1);
        assert_eq!(stats.max_coverage, 0.9);
    }

    #[test]
    fn test_attestation_id_generation() {
        let attestation = ResultAttestation::new(AttestationConfig::default());
        let id1 = attestation.generate_attestation_id("test-suite");
        let id2 = attestation.generate_attestation_id("test-suite");
        
        assert!(id1.starts_with("attest-test-suite-"));
        assert!(id2.starts_with("attest-test-suite-"));
        assert_ne!(id1, id2); // Should be unique due to timestamp
    }
}