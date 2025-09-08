use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error, debug, instrument};

use super::{BenchmarkResult, SystemSummary, BenchmarkSummary, GoldenQuery, MetricsCalculator};

/// Evaluates benchmark results and calculates performance metrics
#[derive(Debug)]
pub struct ResultEvaluator {
    results: Arc<RwLock<Vec<BenchmarkResult>>>,
}

impl ResultEvaluator {
    /// Create a new result evaluator
    pub fn new() -> Self {
        Self {
            results: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Add a benchmark result for evaluation
    #[instrument(skip(self))]
    pub async fn add_result(&self, result: BenchmarkResult) {
        let mut results = self.results.write().await;
        results.push(result);
    }

    /// Calculate system-level summary for a specific system
    #[instrument(skip(self))]
    pub async fn calculate_system_summary(&self, system_name: &str) -> Option<SystemSummary> {
        let results = self.results.read().await;
        let system_results: Vec<_> = results
            .iter()
            .filter(|r| r.system_name == system_name)
            .collect();

        if system_results.is_empty() {
            return None;
        }

        let total_queries = system_results.len() as u32;
        let successful_queries = system_results
            .iter()
            .filter(|r| r.error.is_none())
            .count() as u32;

        let avg_success_at_10: f64 = system_results
            .iter()
            .filter(|r| r.error.is_none())
            .map(|r| r.success_at_10)
            .sum::<f64>() / successful_queries as f64;

        let avg_ndcg_at_10: f64 = system_results
            .iter()
            .filter(|r| r.error.is_none())
            .map(|r| r.ndcg_at_10)
            .sum::<f64>() / successful_queries as f64;

        let avg_sla_recall_at_50: f64 = system_results
            .iter()
            .filter(|r| r.error.is_none())
            .map(|r| r.sla_recall_at_50)
            .sum::<f64>() / successful_queries as f64;

        let latencies: Vec<u64> = system_results
            .iter()
            .filter(|r| r.error.is_none())
            .map(|r| r.latency_ms)
            .collect();
        
        let p95_latency_ms = MetricsCalculator::calculate_p95_latency(&latencies);
        
        let lsp_routing_percentage = system_results
            .iter()
            .filter(|r| r.lsp_routed)
            .count() as f64 / total_queries as f64 * 100.0;

        let sla_compliant_count = system_results
            .iter()
            .filter(|r| r.sla_compliant)
            .count();
        
        let meets_sla = (sla_compliant_count as f64 / total_queries as f64) >= 0.8; // 80% SLA compliance

        Some(SystemSummary {
            system_name: system_name.to_string(),
            total_queries,
            successful_queries,
            performance_gain_pp: 0.0, // Will be calculated relative to baseline
            p95_latency_ms,
            meets_sla,
            lsp_routing_percentage,
            avg_success_at_10,
            avg_ndcg_at_10,
            avg_sla_recall_at_50,
        })
    }

    /// Calculate overall benchmark summary
    #[instrument(skip(self))]
    pub async fn calculate_benchmark_summary(&self) -> BenchmarkSummary {
        let results = self.results.read().await;
        
        if results.is_empty() {
            return BenchmarkSummary {
                total_queries: 0,
                successful_queries: 0,
                average_success_at_10: 0.0,
                average_ndcg_at_10: 0.0,
                average_sla_recall_at_50: 0.0,
                average_latency_ms: 0,
                p95_latency_ms: 0,
                sla_compliance_rate: 0.0,
                system_summaries: Vec::new(),
                passes_performance_gates: false,
                gate_analysis: Vec::new(),
            };
        }

        let total_queries = results.len() as u32;
        let successful_queries = results
            .iter()
            .filter(|r| r.error.is_none())
            .count() as u32;

        let average_success_at_10: f64 = results
            .iter()
            .filter(|r| r.error.is_none())
            .map(|r| r.success_at_10)
            .sum::<f64>() / successful_queries as f64;

        let average_ndcg_at_10: f64 = results
            .iter()
            .filter(|r| r.error.is_none())
            .map(|r| r.ndcg_at_10)
            .sum::<f64>() / successful_queries as f64;

        let average_sla_recall_at_50: f64 = results
            .iter()
            .filter(|r| r.error.is_none())
            .map(|r| r.sla_recall_at_50)
            .sum::<f64>() / successful_queries as f64;

        let latencies: Vec<u64> = results
            .iter()
            .filter(|r| r.error.is_none())
            .map(|r| r.latency_ms)
            .collect();
        
        let average_latency_ms = if latencies.is_empty() {
            0
        } else {
            (latencies.iter().sum::<u64>() / latencies.len() as u64)
        };

        let p95_latency_ms = MetricsCalculator::calculate_p95_latency(&latencies);
        
        let sla_compliant_count = results
            .iter()
            .filter(|r| r.sla_compliant)
            .count();
        
        let sla_compliance_rate = sla_compliant_count as f64 / total_queries as f64;

        // Get unique system names
        let mut system_names: Vec<String> = results
            .iter()
            .map(|r| r.system_name.clone())
            .collect();
        system_names.sort();
        system_names.dedup();

        // Calculate system summaries
        let mut system_summaries = Vec::new();
        for system_name in system_names {
            if let Some(summary) = self.calculate_system_summary(&system_name).await {
                system_summaries.push(summary);
            }
        }

        // Calculate performance gains relative to baseline
        let baseline_summary = system_summaries
            .iter()
            .find(|s| s.system_name == "baseline")
            .cloned();

        if let Some(baseline) = baseline_summary {
            for summary in &mut system_summaries {
                if summary.system_name != "baseline" {
                    summary.performance_gain_pp = MetricsCalculator::calculate_performance_gain_pp(
                        baseline.avg_success_at_10,
                        summary.avg_success_at_10
                    );
                }
            }
        }

        BenchmarkSummary {
            total_queries,
            successful_queries,
            average_success_at_10,
            average_ndcg_at_10,
            average_sla_recall_at_50,
            average_latency_ms,
            p95_latency_ms,
            sla_compliance_rate,
            system_summaries,
            passes_performance_gates: false, // Will be evaluated separately
            gate_analysis: Vec::new(),      // Will be calculated separately
        }
    }

    /// Clear all results
    pub async fn clear_results(&self) {
        let mut results = self.results.write().await;
        results.clear();
    }

    /// Get current result count
    pub async fn result_count(&self) -> usize {
        let results = self.results.read().await;
        results.len()
    }
}

impl Default for ResultEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{QueryType, QueryDifficulty, QuerySlice};

    #[tokio::test]
    async fn test_result_evaluator() {
        let evaluator = ResultEvaluator::new();

        let result = BenchmarkResult {
            system_name: "test_system".to_string(),
            query_id: "q1".to_string(),
            query_text: "test query".to_string(),
            success_at_10: 0.8,
            ndcg_at_10: 0.7,
            sla_recall_at_50: 0.9,
            latency_ms: 100,
            sla_compliant: true,
            lsp_routed: true,
            results_count: 5,
            error: None,
        };

        evaluator.add_result(result).await;
        assert_eq!(evaluator.result_count().await, 1);

        let summary = evaluator.calculate_system_summary("test_system").await;
        assert!(summary.is_some());

        let summary = summary.unwrap();
        assert_eq!(summary.total_queries, 1);
        assert_eq!(summary.successful_queries, 1);
        assert_eq!(summary.avg_success_at_10, 0.8);
    }
}