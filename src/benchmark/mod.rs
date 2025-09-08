use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::fs;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error, debug, instrument};

use crate::search::{SearchEngine, SearchRequest, SearchResponse};
use crate::metrics::{MetricsCollector, SlaMetrics};

pub mod runner;
pub mod ground_truth;
pub mod evaluator;
pub mod smoke_test;
pub mod industry_suites;
pub mod attestation_integration;
pub mod rollout;
pub mod statistical_testing;
pub mod reporting;
pub mod todo_validation;
pub mod production_evaluation;
pub mod ablation_analysis;
pub mod competitor_harness;
pub mod canary_rollout;

pub use runner::BenchmarkRunner;
pub use ground_truth::GroundTruthLoader;
pub use evaluator::ResultEvaluator;
pub use smoke_test::SmokeTestSuite;
pub use industry_suites::IndustryBenchmarkRunner;
pub use attestation_integration::ResultAttestation;
pub use rollout::RolloutExecutor;
pub use statistical_testing::StatisticalTester;
pub use reporting::BenchmarkReporter;
pub use todo_validation::TodoValidationOrchestrator;
pub use production_evaluation::ProductionEvaluationRunner;
pub use ablation_analysis::AblationStudyRunner;
pub use competitor_harness::CompetitorHarnessRunner;
pub use canary_rollout::CanaryRolloutRunner;

/// Benchmark configuration matching TODO.md requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Performance gates from TODO.md
    pub performance_gates: PerformanceGates,
    
    /// Dataset configurations
    pub datasets: HashMap<String, DatasetConfig>,
    
    /// System configurations to test
    pub systems: Vec<SystemConfig>,
    
    /// Output settings
    pub output_path: String,
    pub generate_reports: bool,
    pub validate_corpus: bool,
}

/// Performance gates as specified in TODO.md
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceGates {
    /// Minimum performance gain in percentage points
    pub min_gain_pp: f64, // ≥10pp per TODO.md
    
    /// Maximum p95 latency increase in ms
    pub max_latency_increase_ms: u64, // ≤+1ms per TODO.md
    
    /// Maximum overall p95 latency in ms
    pub max_total_latency_ms: u64, // ≤150ms per TODO.md
    
    /// LSP routing percentage targets
    pub lsp_routing_min: f64, // 40% minimum
    pub lsp_routing_max: f64, // 60% maximum
    
    /// Gap closure target
    pub gap_closure_target_pp: f64, // 32.8pp per TODO.md
    
    /// Buffer for performance variation
    pub performance_buffer_pp: f64, // 8-10pp buffer
}

/// Dataset configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub name: String,
    pub path: String,
    pub golden_queries_path: String,
    pub corpus_path: String,
    pub query_limit: Option<u32>,
    pub stratified_sampling: bool,
}

/// System under test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    pub name: String,
    pub description: String,
    pub enable_lsp: bool,
    pub enable_semantic: bool,
    pub enable_symbols: bool,
    pub baseline: bool, // True for baseline system
}

/// Individual benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub system_name: String,
    pub query_id: String,
    pub query_text: String,
    pub success_at_10: f64,
    pub ndcg_at_10: f64,
    pub sla_recall_at_50: f64,
    pub latency_ms: u64,
    pub sla_compliant: bool,
    pub lsp_routed: bool,
    pub results_count: u32,
    pub error: Option<String>,
}

/// System-level benchmark summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemSummary {
    pub system_name: String,
    pub total_queries: u32,
    pub successful_queries: u32,
    pub performance_gain_pp: f64,
    pub p95_latency_ms: u64,
    pub meets_sla: bool,
    pub lsp_routing_percentage: f64,
    pub avg_success_at_10: f64,
    pub avg_ndcg_at_10: f64,
    pub avg_sla_recall_at_50: f64,
}

/// Complete benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub results: Vec<BenchmarkResult>,
    pub summary: BenchmarkSummary,
    pub report_path: Option<String>,
    pub config_fingerprint: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Overall benchmark summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub total_queries: u32,
    pub successful_queries: u32,
    pub average_success_at_10: f64,
    pub average_ndcg_at_10: f64,
    pub average_sla_recall_at_50: f64,
    pub average_latency_ms: u64,
    pub p95_latency_ms: u64,
    pub sla_compliance_rate: f64,
    pub system_summaries: Vec<SystemSummary>,
    pub passes_performance_gates: bool,
    pub gate_analysis: Vec<GateResult>,
}

/// Performance gate evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    pub gate_name: String,
    pub target_value: f64,
    pub actual_value: f64,
    pub passed: bool,
    pub margin: f64,
    pub description: String,
}

/// Golden query from ground truth dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenQuery {
    pub id: String,
    pub query: String,
    pub expected_files: Vec<String>,
    pub expected_symbols: Vec<String>,
    pub query_type: QueryType,
    pub language: Option<String>,
    pub difficulty: QueryDifficulty,
    pub slice: QuerySlice,
}

/// Query classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryType {
    ExactMatch,
    Identifier,
    Structural,
    Semantic,
    CrossReference,
    TypeDefinition,
}

/// Query difficulty level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryDifficulty {
    Easy,
    Medium,
    Hard,
}

/// Query slice for stratified sampling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuerySlice {
    SmokeDefault,
    Full,
    LanguageSpecific(String),
    TypeSpecific(QueryType),
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            performance_gates: PerformanceGates {
                min_gain_pp: 10.0,           // ≥10pp gain
                max_latency_increase_ms: 1,   // ≤+1ms p95
                max_total_latency_ms: 150,   // ≤150ms p95
                lsp_routing_min: 40.0,       // 40% minimum
                lsp_routing_max: 60.0,       // 60% maximum
                gap_closure_target_pp: 32.8, // 32.8pp target
                performance_buffer_pp: 9.0,  // 8-10pp buffer
            },
            datasets: {
                let mut datasets = HashMap::new();
                datasets.insert("storyviz".to_string(), DatasetConfig {
                    name: "storyviz".to_string(),
                    path: "./indexed-content".to_string(),
                    golden_queries_path: "./pinned-datasets/golden-pinned-current.json".to_string(),
                    corpus_path: "./indexed-content".to_string(),
                    query_limit: Some(390), // Full pinned dataset
                    stratified_sampling: true,
                });
                datasets
            },
            systems: vec![
                SystemConfig {
                    name: "baseline".to_string(),
                    description: "Text search only (baseline)".to_string(),
                    enable_lsp: false,
                    enable_semantic: false,
                    enable_symbols: false,
                    baseline: true,
                },
                SystemConfig {
                    name: "lsp".to_string(),
                    description: "LSP integration enabled".to_string(),
                    enable_lsp: true,
                    enable_semantic: false,
                    enable_symbols: true,
                    baseline: false,
                },
                SystemConfig {
                    name: "lsp_semantic".to_string(),
                    description: "LSP + semantic search".to_string(),
                    enable_lsp: true,
                    enable_semantic: true,
                    enable_symbols: true,
                    baseline: false,
                },
            ],
            output_path: "./benchmark-results".to_string(),
            generate_reports: true,
            validate_corpus: true,
        }
    }
}

/// Evaluation metrics calculator
pub struct MetricsCalculator;

impl MetricsCalculator {
    /// Calculate Success@10 metric
    pub fn calculate_success_at_10(predicted: &[String], expected: &[String]) -> f64 {
        if expected.is_empty() {
            return 0.0;
        }

        let top_10: Vec<_> = predicted.iter().take(10).collect();
        let found = expected.iter().any(|exp| top_10.contains(&exp));
        
        if found { 1.0 } else { 0.0 }
    }

    /// Calculate nDCG@10 metric
    pub fn calculate_ndcg_at_10(predicted: &[String], expected: &[String]) -> f64 {
        if expected.is_empty() {
            return 0.0;
        }

        let k = 10;
        let mut dcg = 0.0;
        let mut idcg = 0.0;

        // Calculate DCG@10
        for (i, pred) in predicted.iter().take(k).enumerate() {
            let relevance = if expected.contains(pred) { 1.0 } else { 0.0 };
            dcg += relevance / (i as f64 + 2.0).log2(); // Standard NDCG formula: rel / log2(i+2)
        }

        // Calculate ideal DCG@10
        let ideal_relevance = vec![1.0; std::cmp::min(k, expected.len())];
        for (i, &rel) in ideal_relevance.iter().enumerate() {
            idcg += rel / (i as f64 + 2.0).log2(); // Standard NDCG formula: rel / log2(i+2)
        }

        if idcg == 0.0 { 0.0 } else { dcg / idcg }
    }

    /// Calculate SLA-Recall@50 metric
    pub fn calculate_sla_recall_at_50(
        predicted: &[String], 
        expected: &[String], 
        sla_compliant: bool
    ) -> f64 {
        if !sla_compliant {
            return 0.0;
        }

        if expected.is_empty() {
            return 1.0; // No expected results, trivially satisfied
        }

        let top_50: Vec<_> = predicted.iter().take(50).collect();
        let found_count = expected.iter().filter(|exp| top_50.contains(exp)).count();
        
        found_count as f64 / expected.len() as f64
    }

    /// Calculate performance gain in percentage points
    pub fn calculate_performance_gain_pp(baseline_metric: f64, test_metric: f64) -> f64 {
        (test_metric - baseline_metric) * 100.0
    }

    /// Calculate p95 latency from latency samples
    pub fn calculate_p95_latency(latencies: &[u64]) -> u64 {
        if latencies.is_empty() {
            return 0;
        }

        let mut sorted = latencies.to_vec();
        sorted.sort_unstable();
        
        let index = ((latencies.len() as f64 * 0.95).ceil() as usize).min(latencies.len()) - 1;
        sorted[index]
    }

    /// Evaluate performance gates
    pub fn evaluate_performance_gates(
        config: &PerformanceGates,
        baseline_summary: &SystemSummary,
        test_summary: &SystemSummary,
    ) -> Vec<GateResult> {
        let mut results = Vec::new();

        // Gate 1: Minimum performance gain ≥10pp
        let performance_gain = Self::calculate_performance_gain_pp(
            baseline_summary.avg_success_at_10,
            test_summary.avg_success_at_10
        );
        results.push(GateResult {
            gate_name: "Performance Gain".to_string(),
            target_value: config.min_gain_pp,
            actual_value: performance_gain,
            passed: performance_gain >= config.min_gain_pp,
            margin: performance_gain - config.min_gain_pp,
            description: format!("≥{}pp gain requirement", config.min_gain_pp),
        });

        // Gate 2: Maximum latency increase ≤+1ms p95
        let latency_increase = test_summary.p95_latency_ms as i64 - baseline_summary.p95_latency_ms as i64;
        results.push(GateResult {
            gate_name: "Latency Increase".to_string(),
            target_value: config.max_latency_increase_ms as f64,
            actual_value: latency_increase as f64,
            passed: latency_increase <= config.max_latency_increase_ms as i64,
            margin: config.max_latency_increase_ms as f64 - latency_increase as f64,
            description: format!("≤+{}ms p95 latency increase", config.max_latency_increase_ms),
        });

        // Gate 3: Maximum total latency ≤150ms p95
        results.push(GateResult {
            gate_name: "Total Latency".to_string(),
            target_value: config.max_total_latency_ms as f64,
            actual_value: test_summary.p95_latency_ms as f64,
            passed: test_summary.p95_latency_ms <= config.max_total_latency_ms,
            margin: config.max_total_latency_ms as f64 - test_summary.p95_latency_ms as f64,
            description: format!("≤{}ms p95 total latency", config.max_total_latency_ms),
        });

        // Gate 4: LSP routing percentage 40-60%
        let lsp_routing_ok = test_summary.lsp_routing_percentage >= config.lsp_routing_min 
            && test_summary.lsp_routing_percentage <= config.lsp_routing_max;
        results.push(GateResult {
            gate_name: "LSP Routing".to_string(),
            target_value: (config.lsp_routing_min + config.lsp_routing_max) / 2.0,
            actual_value: test_summary.lsp_routing_percentage,
            passed: lsp_routing_ok,
            margin: if test_summary.lsp_routing_percentage < config.lsp_routing_min {
                test_summary.lsp_routing_percentage - config.lsp_routing_min
            } else if test_summary.lsp_routing_percentage > config.lsp_routing_max {
                config.lsp_routing_max - test_summary.lsp_routing_percentage  
            } else {
                0.0
            },
            description: format!("{}%-{}% LSP routing target", config.lsp_routing_min, config.lsp_routing_max),
        });

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_success_at_10() {
        let predicted = vec!["file1.rs".to_string(), "file2.rs".to_string(), "file3.rs".to_string()];
        let expected = vec!["file2.rs".to_string(), "file4.rs".to_string()];
        
        let score = MetricsCalculator::calculate_success_at_10(&predicted, &expected);
        assert_eq!(score, 1.0); // file2.rs found in top 10
    }

    #[test]
    fn test_ndcg_at_10() {
        let predicted = vec!["file1.rs".to_string(), "file2.rs".to_string()];
        let expected = vec!["file2.rs".to_string()];
        
        let score = MetricsCalculator::calculate_ndcg_at_10(&predicted, &expected);
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_p95_calculation() {
        let latencies = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        let p95 = MetricsCalculator::calculate_p95_latency(&latencies);
        assert_eq!(p95, 100); // p95 of 10 elements should be the max
    }

    #[test]
    fn test_performance_gain() {
        let gain = MetricsCalculator::calculate_performance_gain_pp(0.50, 0.60);
        assert!((gain - 10.0).abs() < 1e-10); // 10 percentage points improvement (with float precision tolerance)
    }
}