//! SLA-Bounded Evaluation System
//!
//! Implements evaluation metrics respecting the 150ms SLA constraint per query.
//! Measures nDCG@10, Expected Calibration Error (ECE), and statistical significance
//! via paired bootstrap testing.
//!
//! **TODO.md Requirements:**
//! - SLA cap: 150ms per query
//! - nDCG@10 calculation with proper ranking cutoffs
//! - ECE ≤ 0.02 validation
//! - Paired bootstrap for statistical significance (10,000 samples, α=0.05)
//! - Baseline comparison against `policy://lexical_struct_only@<fingerprint>`

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use rand::seq::SliceRandom;
use rand::Rng;
use anyhow::{Result, Context};
use tracing::{info, warn, error};

/// SLA constraint configuration
pub const DEFAULT_SLA_MS: u64 = 150;
pub const MAX_ECE_THRESHOLD: f32 = 0.02;
pub const BOOTSTRAP_SAMPLES: usize = 10_000;
pub const SIGNIFICANCE_ALPHA: f32 = 0.05;

/// Query evaluation result within SLA bounds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SLABoundedQueryResult {
    pub query_id: String,
    pub execution_time_ms: u64,
    pub within_sla: bool,
    pub rankings: Vec<RankedResult>,
    pub ground_truth: Vec<GroundTruthItem>,
    pub ndcg_at_10: Option<f32>,
    pub calibration_scores: Vec<CalibrationPoint>,
}

/// Individual ranked result from search system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedResult {
    pub document_id: String,
    pub score: f32,
    pub calibrated_probability: Option<f32>,
    pub rank: usize,
}

/// Ground truth relevance for evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruthItem {
    pub document_id: String,
    pub relevance: f32, // 0.0 to 1.0
    pub intent_category: String,
    pub language: String,
}

/// Calibration point for ECE calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationPoint {
    pub predicted_probability: f32,
    pub actual_relevance: f32,
    pub confidence_bin: usize,
}

/// Complete evaluation results for a dataset slice
#[derive(Debug, Serialize, Deserialize)]
pub struct SLABoundedEvaluationResult {
    pub slice_name: String,
    pub total_queries: usize,
    pub within_sla_queries: usize,
    pub sla_recall: f32, // Fraction of queries processed within SLA
    pub mean_ndcg_at_10: f32,
    pub std_ndcg_at_10: f32,
    pub expected_calibration_error: f32,
    pub calibration_bins: Vec<CalibrationBin>,
    pub bootstrap_confidence_interval: Option<BootstrapCI>,
    pub baseline_comparison: Option<BaselineComparison>,
    pub execution_time_stats: ExecutionTimeStats,
    pub artifact_path: String,
}

/// Calibration bin for ECE calculation
#[derive(Debug, Serialize, Deserialize)]
pub struct CalibrationBin {
    pub bin_id: usize,
    pub confidence_range: (f32, f32),
    pub count: usize,
    pub avg_confidence: f32,
    pub avg_accuracy: f32,
    pub bin_ece: f32,
}

/// Bootstrap confidence interval
#[derive(Debug, Serialize, Deserialize)]
pub struct BootstrapCI {
    pub metric_name: String,
    pub point_estimate: f32,
    pub lower_bound: f32, // 2.5th percentile
    pub upper_bound: f32, // 97.5th percentile
    pub p_value: Option<f32>,
}

/// Baseline policy comparison results
#[derive(Debug, Serialize, Deserialize)]
pub struct BaselineComparison {
    pub baseline_policy: String,
    pub baseline_ndcg: f32,
    pub candidate_ndcg: f32,
    pub delta_ndcg: f32,
    pub statistical_significance: bool,
    pub p_value: f32,
}

/// Execution time statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct ExecutionTimeStats {
    pub mean_ms: f32,
    pub median_ms: f32,
    pub p95_ms: f32,
    pub p99_ms: f32,
    pub max_ms: f32,
    pub timeout_count: usize,
}

/// Configuration for SLA-bounded evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SLAEvaluationConfig {
    pub sla_timeout_ms: u64,
    pub max_ece_threshold: f32,
    pub bootstrap_samples: usize,
    pub significance_alpha: f32,
    pub ndcg_cutoff: usize,
    pub calibration_bins: usize,
    pub baseline_policy_fingerprint: String,
}

impl Default for SLAEvaluationConfig {
    fn default() -> Self {
        Self {
            sla_timeout_ms: DEFAULT_SLA_MS,
            max_ece_threshold: MAX_ECE_THRESHOLD,
            bootstrap_samples: BOOTSTRAP_SAMPLES,
            significance_alpha: SIGNIFICANCE_ALPHA,
            ndcg_cutoff: 10,
            calibration_bins: 10,
            baseline_policy_fingerprint: "lexical_struct_only".to_string(),
        }
    }
}

/// Main SLA-bounded evaluation system
pub struct SLABoundedEvaluator {
    config: SLAEvaluationConfig,
    rng: rand::rngs::StdRng,
}

impl SLABoundedEvaluator {
    pub fn new(config: SLAEvaluationConfig) -> Self {
        use rand::SeedableRng;
        Self {
            config,
            rng: rand::rngs::StdRng::from_entropy(),
        }
    }

    /// Evaluate a query with SLA bounds
    pub async fn evaluate_query_bounded<F, Fut>(
        &self,
        query_id: &str,
        query: &str,
        ground_truth: Vec<GroundTruthItem>,
        search_fn: F,
    ) -> Result<SLABoundedQueryResult>
    where
        F: FnOnce(&str) -> Fut,
        Fut: std::future::Future<Output = Result<Vec<RankedResult>>>,
    {
        let start_time = Instant::now();
        let timeout_duration = Duration::from_millis(self.config.sla_timeout_ms);

        // Execute search with timeout
        let search_result = tokio::time::timeout(timeout_duration, search_fn(query)).await;
        
        let execution_time = start_time.elapsed();
        let execution_time_ms = execution_time.as_millis() as u64;
        let within_sla = execution_time_ms <= self.config.sla_timeout_ms;

        match search_result {
            Ok(Ok(rankings)) => {
                // Calculate nDCG@10
                let ndcg_at_10 = if within_sla {
                    Some(self.calculate_ndcg_at_k(&rankings, &ground_truth, self.config.ndcg_cutoff)?)
                } else {
                    None
                };

                // Extract calibration points
                let calibration_scores = self.extract_calibration_points(&rankings, &ground_truth)?;

                Ok(SLABoundedQueryResult {
                    query_id: query_id.to_string(),
                    execution_time_ms,
                    within_sla,
                    rankings,
                    ground_truth,
                    ndcg_at_10,
                    calibration_scores,
                })
            }
            Ok(Err(e)) => {
                warn!("Search failed for query {}: {}", query_id, e);
                Ok(SLABoundedQueryResult {
                    query_id: query_id.to_string(),
                    execution_time_ms,
                    within_sla,
                    rankings: vec![],
                    ground_truth,
                    ndcg_at_10: None,
                    calibration_scores: vec![],
                })
            }
            Err(_) => {
                warn!("Query {} timed out after {}ms", query_id, self.config.sla_timeout_ms);
                Ok(SLABoundedQueryResult {
                    query_id: query_id.to_string(),
                    execution_time_ms,
                    within_sla: false,
                    rankings: vec![],
                    ground_truth,
                    ndcg_at_10: None,
                    calibration_scores: vec![],
                })
            }
        }
    }

    /// Evaluate a complete dataset slice
    pub async fn evaluate_slice<F, Fut>(
        &mut self,
        slice_name: &str,
        queries: Vec<(String, String, Vec<GroundTruthItem>)>, // (id, query, ground_truth)
        search_fn: F,
        baseline_results: Option<Vec<SLABoundedQueryResult>>,
    ) -> Result<SLABoundedEvaluationResult>
    where
        F: Fn(&str) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<Vec<RankedResult>>> + Send,
    {
        info!("Starting SLA-bounded evaluation for slice: {}", slice_name);
        
        let mut query_results = Vec::new();
        let total_queries = queries.len();

        // Evaluate each query
        for (query_id, query, ground_truth) in queries {
            let result = self.evaluate_query_bounded(
                &query_id, 
                &query, 
                ground_truth, 
                &search_fn
            ).await?;
            
            query_results.push(result);
        }

        // Calculate aggregate metrics
        let within_sla_results: Vec<_> = query_results
            .iter()
            .filter(|r| r.within_sla && r.ndcg_at_10.is_some())
            .collect();

        let within_sla_queries = within_sla_results.len();
        let sla_recall = within_sla_queries as f32 / total_queries as f32;

        // nDCG@10 statistics
        let ndcg_values: Vec<f32> = within_sla_results
            .iter()
            .filter_map(|r| r.ndcg_at_10)
            .collect();

        let (mean_ndcg, std_ndcg) = if !ndcg_values.is_empty() {
            let mean = ndcg_values.iter().sum::<f32>() / ndcg_values.len() as f32;
            let variance = ndcg_values.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>() / ndcg_values.len() as f32;
            (mean, variance.sqrt())
        } else {
            (0.0, 0.0)
        };

        // Calculate Expected Calibration Error
        let all_calibration_points: Vec<_> = query_results
            .iter()
            .flat_map(|r| r.calibration_scores.iter())
            .cloned()
            .collect();

        let (ece, calibration_bins) = self.calculate_expected_calibration_error(&all_calibration_points)?;

        // Bootstrap confidence interval for nDCG
        let bootstrap_ci = if !ndcg_values.is_empty() {
            Some(self.calculate_bootstrap_confidence_interval(&ndcg_values, "nDCG@10")?)
        } else {
            None
        };

        // Baseline comparison
        let baseline_comparison = if let Some(baseline_results) = baseline_results {
            Some(self.compare_with_baseline(&query_results, &baseline_results)?)
        } else {
            None
        };

        // Execution time statistics
        let execution_times: Vec<u64> = query_results.iter().map(|r| r.execution_time_ms).collect();
        let execution_time_stats = self.calculate_execution_time_stats(&execution_times);

        // Generate artifact path
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string();
        let artifact_path = format!("artifact://eval/sla_eval_{}_{}.json", slice_name, timestamp);

        let result = SLABoundedEvaluationResult {
            slice_name: slice_name.to_string(),
            total_queries,
            within_sla_queries,
            sla_recall,
            mean_ndcg_at_10: mean_ndcg,
            std_ndcg_at_10: std_ndcg,
            expected_calibration_error: ece,
            calibration_bins,
            bootstrap_confidence_interval: bootstrap_ci,
            baseline_comparison,
            execution_time_stats,
            artifact_path: artifact_path.clone(),
        };

        // Save artifact
        self.save_evaluation_artifact(&result).await?;

        info!(
            "SLA-bounded evaluation complete. SLA Recall: {:.3}, Mean nDCG@10: {:.4}, ECE: {:.4}",
            sla_recall, mean_ndcg, ece
        );

        Ok(result)
    }

    /// Calculate nDCG@k for a query result
    fn calculate_ndcg_at_k(
        &self,
        rankings: &[RankedResult],
        ground_truth: &[GroundTruthItem],
        k: usize,
    ) -> Result<f32> {
        // Create relevance map
        let relevance_map: HashMap<String, f32> = ground_truth
            .iter()
            .map(|gt| (gt.document_id.clone(), gt.relevance))
            .collect();

        // Calculate DCG@k
        let dcg = rankings
            .iter()
            .take(k)
            .enumerate()
            .map(|(i, result)| {
                let relevance = relevance_map.get(&result.document_id).unwrap_or(&0.0);
                let discount = (i + 2) as f32; // i+2 because log2(1) = 0
                relevance / discount.log2()
            })
            .sum::<f32>();

        // Calculate IDCG@k (ideal DCG)
        let mut ideal_relevances: Vec<f32> = ground_truth.iter().map(|gt| gt.relevance).collect();
        ideal_relevances.sort_by(|a, b| b.partial_cmp(a).unwrap());
        
        let idcg = ideal_relevances
            .iter()
            .take(k)
            .enumerate()
            .map(|(i, relevance)| {
                let discount = (i + 2) as f32;
                relevance / discount.log2()
            })
            .sum::<f32>();

        // Calculate nDCG@k
        let ndcg = if idcg > 0.0 { dcg / idcg } else { 0.0 };
        Ok(ndcg)
    }

    /// Extract calibration points from results
    fn extract_calibration_points(
        &self,
        rankings: &[RankedResult],
        ground_truth: &[GroundTruthItem],
    ) -> Result<Vec<CalibrationPoint>> {
        let relevance_map: HashMap<String, f32> = ground_truth
            .iter()
            .map(|gt| (gt.document_id.clone(), gt.relevance))
            .collect();

        let calibration_points = rankings
            .iter()
            .filter_map(|result| {
                if let Some(predicted_prob) = result.calibrated_probability {
                    let actual_relevance = relevance_map.get(&result.document_id).unwrap_or(&0.0);
                    let confidence_bin = (predicted_prob * self.config.calibration_bins as f32).floor() as usize;
                    let confidence_bin = confidence_bin.min(self.config.calibration_bins - 1);
                    
                    Some(CalibrationPoint {
                        predicted_probability: predicted_prob,
                        actual_relevance: *actual_relevance,
                        confidence_bin,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(calibration_points)
    }

    /// Calculate Expected Calibration Error
    fn calculate_expected_calibration_error(
        &self,
        calibration_points: &[CalibrationPoint],
    ) -> Result<(f32, Vec<CalibrationBin>)> {
        let mut bins: Vec<CalibrationBin> = (0..self.config.calibration_bins)
            .map(|i| {
                let bin_start = i as f32 / self.config.calibration_bins as f32;
                let bin_end = (i + 1) as f32 / self.config.calibration_bins as f32;
                CalibrationBin {
                    bin_id: i,
                    confidence_range: (bin_start, bin_end),
                    count: 0,
                    avg_confidence: 0.0,
                    avg_accuracy: 0.0,
                    bin_ece: 0.0,
                }
            })
            .collect();

        // Populate bins
        for point in calibration_points {
            let bin = &mut bins[point.confidence_bin];
            bin.count += 1;
            bin.avg_confidence += point.predicted_probability;
            bin.avg_accuracy += point.actual_relevance;
        }

        // Calculate bin statistics
        let total_points = calibration_points.len() as f32;
        let mut total_ece = 0.0;

        for bin in &mut bins {
            if bin.count > 0 {
                bin.avg_confidence /= bin.count as f32;
                bin.avg_accuracy /= bin.count as f32;
                bin.bin_ece = (bin.avg_confidence - bin.avg_accuracy).abs();
                
                // Weight by bin size for ECE calculation
                let bin_weight = bin.count as f32 / total_points;
                total_ece += bin_weight * bin.bin_ece;
            }
        }

        Ok((total_ece, bins))
    }

    /// Calculate bootstrap confidence interval
    fn calculate_bootstrap_confidence_interval(
        &mut self,
        values: &[f32],
        metric_name: &str,
    ) -> Result<BootstrapCI> {
        let point_estimate = values.iter().sum::<f32>() / values.len() as f32;
        let mut bootstrap_means = Vec::with_capacity(self.config.bootstrap_samples);

        for _ in 0..self.config.bootstrap_samples {
            let bootstrap_sample: Vec<f32> = (0..values.len())
                .map(|_| {
                    let idx = self.rng.gen_range(0..values.len());
                    values[idx]
                })
                .collect();
            
            let bootstrap_mean = bootstrap_sample.iter().sum::<f32>() / bootstrap_sample.len() as f32;
            bootstrap_means.push(bootstrap_mean);
        }

        bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let lower_idx = (self.config.bootstrap_samples as f32 * 0.025) as usize;
        let upper_idx = (self.config.bootstrap_samples as f32 * 0.975) as usize;

        Ok(BootstrapCI {
            metric_name: metric_name.to_string(),
            point_estimate,
            lower_bound: bootstrap_means[lower_idx],
            upper_bound: bootstrap_means[upper_idx],
            p_value: None,
        })
    }

    /// Compare candidate with baseline policy
    fn compare_with_baseline(
        &mut self,
        candidate_results: &[SLABoundedQueryResult],
        baseline_results: &[SLABoundedQueryResult],
    ) -> Result<BaselineComparison> {
        // Extract nDCG values for both systems
        let candidate_ndcg: Vec<f32> = candidate_results
            .iter()
            .filter_map(|r| r.ndcg_at_10)
            .collect();

        let baseline_ndcg: Vec<f32> = baseline_results
            .iter()
            .filter_map(|r| r.ndcg_at_10)
            .collect();

        let candidate_mean = candidate_ndcg.iter().sum::<f32>() / candidate_ndcg.len() as f32;
        let baseline_mean = baseline_ndcg.iter().sum::<f32>() / baseline_ndcg.len() as f32;
        let delta_ndcg = candidate_mean - baseline_mean;

        // Paired bootstrap test
        let mut delta_samples = Vec::with_capacity(self.config.bootstrap_samples);
        
        for _ in 0..self.config.bootstrap_samples {
            let sample_size = candidate_ndcg.len().min(baseline_ndcg.len());
            let mut candidate_sample = 0.0;
            let mut baseline_sample = 0.0;
            
            for _ in 0..sample_size {
                let idx = self.rng.gen_range(0..sample_size);
                candidate_sample += candidate_ndcg[idx];
                baseline_sample += baseline_ndcg[idx];
            }
            
            let delta = (candidate_sample / sample_size as f32) - (baseline_sample / sample_size as f32);
            delta_samples.push(delta);
        }

        delta_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Calculate p-value (two-tailed test)
        let negative_deltas = delta_samples.iter().filter(|&&d| d <= 0.0).count();
        let p_value = 2.0 * (negative_deltas as f32 / delta_samples.len() as f32).min(0.5);
        let statistical_significance = p_value < self.config.significance_alpha;

        Ok(BaselineComparison {
            baseline_policy: format!("policy://lexical_struct_only@{}", self.config.baseline_policy_fingerprint),
            baseline_ndcg: baseline_mean,
            candidate_ndcg: candidate_mean,
            delta_ndcg,
            statistical_significance,
            p_value,
        })
    }

    /// Calculate execution time statistics
    fn calculate_execution_time_stats(&self, execution_times: &[u64]) -> ExecutionTimeStats {
        let mut sorted_times = execution_times.to_vec();
        sorted_times.sort();

        let mean = sorted_times.iter().sum::<u64>() as f32 / sorted_times.len() as f32;
        let median = sorted_times[sorted_times.len() / 2] as f32;
        let p95_idx = (sorted_times.len() as f32 * 0.95) as usize;
        let p99_idx = (sorted_times.len() as f32 * 0.99) as usize;
        let p95 = sorted_times[p95_idx.min(sorted_times.len() - 1)] as f32;
        let p99 = sorted_times[p99_idx.min(sorted_times.len() - 1)] as f32;
        let max = sorted_times.last().unwrap_or(&0) as f32;
        let timeout_count = execution_times.iter().filter(|&&t| t > self.config.sla_timeout_ms).count();

        ExecutionTimeStats {
            mean_ms: mean,
            median_ms: median,
            p95_ms: p95,
            p99_ms: p99,
            max_ms: max,
            timeout_count,
        }
    }

    /// Save evaluation artifact
    async fn save_evaluation_artifact(&self, result: &SLABoundedEvaluationResult) -> Result<()> {
        let artifact_content = serde_json::to_string_pretty(result)
            .context("Failed to serialize evaluation result")?;

        // Extract filename from artifact path
        let filename = result.artifact_path
            .split("://")
            .nth(1)
            .unwrap_or("sla_eval_result.json")
            .replace('/', "_");

        let artifact_dir = std::path::Path::new("artifact").join("eval");
        tokio::fs::create_dir_all(&artifact_dir).await
            .context("Failed to create artifact directory")?;

        let filepath = artifact_dir.join(&filename);
        tokio::fs::write(&filepath, artifact_content).await
            .context("Failed to write evaluation artifact")?;

        info!("Saved SLA-bounded evaluation artifact: {}", filepath.display());
        Ok(())
    }

    /// Validate gate requirements
    pub fn validate_gates(&self, result: &SLABoundedEvaluationResult) -> Result<GateValidationResult> {
        let mut passed = true;
        let mut violations = Vec::new();

        // Check SLA-Recall ≥ 0 (should be automatic, but validate)
        if result.sla_recall < 0.0 {
            passed = false;
            violations.push(format!("SLA-Recall below 0: {:.3}", result.sla_recall));
        }

        // Check ECE ≤ 0.02
        if result.expected_calibration_error > self.config.max_ece_threshold {
            passed = false;
            violations.push(format!(
                "ECE exceeds threshold: {:.4} > {:.4}",
                result.expected_calibration_error,
                self.config.max_ece_threshold
            ));
        }

        // Check ΔnDCG ≥ +4.0pp (if baseline comparison available)
        if let Some(baseline_comp) = &result.baseline_comparison {
            let delta_pp = baseline_comp.delta_ndcg * 100.0; // Convert to percentage points
            if delta_pp < 4.0 {
                passed = false;
                violations.push(format!(
                    "Semantic lift insufficient: +{:.1}pp < +4.0pp required",
                    delta_pp
                ));
            }
        }

        Ok(GateValidationResult {
            passed,
            violations,
            sla_recall: result.sla_recall,
            ece: result.expected_calibration_error,
            delta_ndcg_pp: result.baseline_comparison.as_ref().map(|b| b.delta_ndcg * 100.0),
        })
    }
}

/// Gate validation result
#[derive(Debug, Serialize, Deserialize)]
pub struct GateValidationResult {
    pub passed: bool,
    pub violations: Vec<String>,
    pub sla_recall: f32,
    pub ece: f32,
    pub delta_ndcg_pp: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sla_bounded_evaluation() {
        let config = SLAEvaluationConfig::default();
        let evaluator = SLABoundedEvaluator::new(config);

        // Mock search function that always returns empty results
        let search_fn = |_query: &str| async { Ok(vec![]) };

        let ground_truth = vec![GroundTruthItem {
            document_id: "doc1".to_string(),
            relevance: 1.0,
            intent_category: "exact_match".to_string(),
            language: "rust".to_string(),
        }];

        let result = evaluator.evaluate_query_bounded(
            "test_query",
            "test query",
            ground_truth,
            search_fn,
        ).await.unwrap();

        assert_eq!(result.query_id, "test_query");
        assert!(result.within_sla);
        assert!(result.execution_time_ms < 150);
    }

    #[test]
    fn test_ndcg_calculation() {
        let evaluator = SLABoundedEvaluator::new(SLAEvaluationConfig::default());
        
        let rankings = vec![
            RankedResult {
                document_id: "doc1".to_string(),
                score: 0.9,
                calibrated_probability: Some(0.8),
                rank: 0,
            },
            RankedResult {
                document_id: "doc2".to_string(),
                score: 0.7,
                calibrated_probability: Some(0.6),
                rank: 1,
            },
        ];

        let ground_truth = vec![
            GroundTruthItem {
                document_id: "doc1".to_string(),
                relevance: 1.0,
                intent_category: "exact_match".to_string(),
                language: "rust".to_string(),
            },
            GroundTruthItem {
                document_id: "doc2".to_string(),
                relevance: 0.5,
                intent_category: "structural".to_string(),
                language: "rust".to_string(),
            },
        ];

        let ndcg = evaluator.calculate_ndcg_at_k(&rankings, &ground_truth, 10).unwrap();
        assert!(ndcg > 0.0 && ndcg <= 1.0);
    }

    #[test]
    fn test_ece_calculation() {
        let evaluator = SLABoundedEvaluator::new(SLAEvaluationConfig::default());
        
        let calibration_points = vec![
            CalibrationPoint {
                predicted_probability: 0.8,
                actual_relevance: 1.0,
                confidence_bin: 7,
            },
            CalibrationPoint {
                predicted_probability: 0.3,
                actual_relevance: 0.0,
                confidence_bin: 2,
            },
        ];

        let (ece, bins) = evaluator.calculate_expected_calibration_error(&calibration_points).unwrap();
        assert!(ece >= 0.0 && ece <= 1.0);
        assert_eq!(bins.len(), 10);
    }
}