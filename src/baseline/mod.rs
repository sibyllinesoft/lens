//! # Baseline Competitors Module
//!
//! Implements BM25+proximity and hybrid lexical+dense baseline competitors
//! as specified in TODO.md Step 3 - Baseline fortification.
//!
//! Competitors:
//! - BM25+proximity (lexical with positional awareness)
//! - Hybrid lexical+dense (traditional hybrid search)
//! - Tuned variants optimized for code search
//!
//! Gate: retain ≥ +3 pp margin across SWE-bench Verified and CoIR, under SLA

pub mod bm25_proximity;
pub mod hybrid_lexical_dense;
pub mod baseline_orchestrator;
pub mod competitive_benchmarking;

pub use bm25_proximity::{BM25ProximitySearcher, ProximityConfig};
pub use hybrid_lexical_dense::{HybridSearcher, HybridConfig};
pub use baseline_orchestrator::{BaselineOrchestrator, BaselineConfig};
pub use competitive_benchmarking::{CompetitiveBenchmark, BenchmarkResult};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Baseline search system trait
#[async_trait::async_trait]
pub trait BaselineSearcher: Send + Sync {
    /// Get system name for identification
    fn system_name(&self) -> &str;
    
    /// Search with the baseline system
    async fn search(&self, query: &str, intent: &str, language: &str, max_results: usize) -> Result<Vec<SearchResult>>;
    
    /// Get system configuration for reproducibility
    fn get_config(&self) -> BaselineSystemConfig;
    
    /// Warm up the system (optional)
    async fn warmup(&self) -> Result<()> {
        Ok(())
    }
    
    /// Get system statistics
    async fn get_statistics(&self) -> Result<SystemStatistics>;
}

/// Search result from baseline system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub file_path: String,
    pub score: f32,
    pub snippet: String,
    pub rank: usize,
    pub metadata: ResultMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultMetadata {
    pub line_number: Option<usize>,
    pub function_name: Option<String>,
    pub class_name: Option<String>,
    pub language: String,
    pub file_size: usize,
    pub last_modified: Option<chrono::DateTime<chrono::Utc>>,
    pub scoring_breakdown: ScoringBreakdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringBreakdown {
    pub lexical_score: f32,
    pub semantic_score: Option<f32>,
    pub proximity_score: Option<f32>,
    pub recency_score: Option<f32>,
    pub popularity_score: Option<f32>,
    pub final_score: f32,
}

/// Baseline system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineSystemConfig {
    pub system_name: String,
    pub version: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub index_config: IndexConfig,
    pub scoring_config: ScoringConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    pub tokenizer: String,
    pub stemming: bool,
    pub stop_words: bool,
    pub n_grams: Vec<usize>,
    pub case_sensitive: bool,
    pub special_characters: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringConfig {
    pub bm25_k1: f32,
    pub bm25_b: f32,
    pub proximity_weight: Option<f32>,
    pub semantic_weight: Option<f32>,
    pub recency_weight: Option<f32>,
    pub normalization: String,
}

/// System performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatistics {
    pub queries_processed: usize,
    pub average_latency_ms: f32,
    pub p95_latency_ms: f32,
    pub p99_latency_ms: f32,
    pub cache_hit_rate: f32,
    pub index_size_mb: f32,
    pub memory_usage_mb: f32,
}

/// Performance comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceComparison {
    pub baseline_system: String,
    pub lens_system: String,
    pub metrics: HashMap<String, MetricComparison>,
    pub statistical_significance: HashMap<String, f32>,
    pub margin_analysis: MarginAnalysis,
    pub sla_compliance: SlaCompliance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricComparison {
    pub baseline_value: f32,
    pub lens_value: f32,
    pub improvement_pp: f32,
    pub improvement_percentage: f32,
    pub confidence_interval: (f32, f32),
    pub statistical_significance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarginAnalysis {
    pub required_margin_pp: f32,
    pub achieved_margin_pp: f32,
    pub margin_maintained: bool,
    pub risk_factors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaCompliance {
    pub sla_threshold_ms: f32,
    pub baseline_p99_ms: f32,
    pub lens_p99_ms: f32,
    pub both_systems_compliant: bool,
    pub comparative_advantage: f32,
}

/// Create all baseline competitors
pub async fn create_baseline_competitors() -> Result<Vec<Box<dyn BaselineSearcher>>> {
    let mut competitors = Vec::new();
    
    // BM25 + Proximity competitor
    let bm25_proximity = bm25_proximity::BM25ProximitySearcher::new(
        bm25_proximity::ProximityConfig::optimized_for_code()
    ).await?;
    competitors.push(Box::new(bm25_proximity) as Box<dyn BaselineSearcher>);
    
    // Hybrid Lexical + Dense competitor
    let hybrid = hybrid_lexical_dense::HybridSearcher::new(
        hybrid_lexical_dense::HybridConfig::balanced_hybrid()
    ).await?;
    competitors.push(Box::new(hybrid) as Box<dyn BaselineSearcher>);
    
    // Additional tuned variants
    let bm25_tuned = bm25_proximity::BM25ProximitySearcher::new(
        bm25_proximity::ProximityConfig::high_precision()
    ).await?;
    competitors.push(Box::new(bm25_tuned) as Box<dyn BaselineSearcher>);
    
    let hybrid_tuned = hybrid_lexical_dense::HybridSearcher::new(
        hybrid_lexical_dense::HybridConfig::code_optimized()
    ).await?;
    competitors.push(Box::new(hybrid_tuned) as Box<dyn BaselineSearcher>);
    
    Ok(competitors)
}

/// Validate minimum performance margin
pub fn validate_performance_margin(comparison: &PerformanceComparison) -> Result<bool> {
    const MINIMUM_MARGIN_PP: f32 = 3.0; // ≥ +3 pp margin requirement
    
    let key_metrics = vec!["ndcg_at_10", "recall_at_50"];
    
    for metric_name in &key_metrics {
        if let Some(metric) = comparison.metrics.get(*metric_name) {
            if metric.improvement_pp < MINIMUM_MARGIN_PP {
                tracing::warn!(
                    "Insufficient margin for {}: {:.1}pp < {:.1}pp required",
                    metric_name, metric.improvement_pp, MINIMUM_MARGIN_PP
                );
                return Ok(false);
            }
        }
    }
    
    // Check SLA compliance for both systems
    if !comparison.sla_compliance.both_systems_compliant {
        tracing::warn!("SLA compliance failed for comparison");
        return Ok(false);
    }
    
    // Verify statistical significance
    for (metric_name, &p_value) in &comparison.statistical_significance {
        if key_metrics.contains(&metric_name.as_str()) && p_value > 0.05 {
            tracing::warn!(
                "No statistical significance for {}: p={:.3} > 0.05",
                metric_name, p_value
            );
            return Ok(false);
        }
    }
    
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_comparison_validation() {
        let mut metrics = HashMap::new();
        metrics.insert("ndcg_at_10".to_string(), MetricComparison {
            baseline_value: 0.65,
            lens_value: 0.68,
            improvement_pp: 3.5, // Above 3pp requirement
            improvement_percentage: 5.4,
            confidence_interval: (3.1, 3.9),
            statistical_significance: 0.001,
        });
        
        let comparison = PerformanceComparison {
            baseline_system: "BM25+Proximity".to_string(),
            lens_system: "Lens+Semantic".to_string(),
            metrics,
            statistical_significance: [("ndcg_at_10".to_string(), 0.001)].into(),
            margin_analysis: MarginAnalysis {
                required_margin_pp: 3.0,
                achieved_margin_pp: 3.5,
                margin_maintained: true,
                risk_factors: vec![],
            },
            sla_compliance: SlaCompliance {
                sla_threshold_ms: 150.0,
                baseline_p99_ms: 145.0,
                lens_p99_ms: 147.0,
                both_systems_compliant: true,
                comparative_advantage: 2.0,
            },
        };
        
        assert!(validate_performance_margin(&comparison).unwrap());
    }

    #[test]
    fn test_insufficient_margin_detection() {
        let mut metrics = HashMap::new();
        metrics.insert("ndcg_at_10".to_string(), MetricComparison {
            baseline_value: 0.65,
            lens_value: 0.67,
            improvement_pp: 2.0, // Below 3pp requirement
            improvement_percentage: 3.1,
            confidence_interval: (1.5, 2.5),
            statistical_significance: 0.001,
        });
        
        let comparison = PerformanceComparison {
            baseline_system: "BM25+Proximity".to_string(),
            lens_system: "Lens+Semantic".to_string(),
            metrics,
            statistical_significance: [("ndcg_at_10".to_string(), 0.001)].into(),
            margin_analysis: MarginAnalysis {
                required_margin_pp: 3.0,
                achieved_margin_pp: 2.0,
                margin_maintained: false,
                risk_factors: vec!["Narrow margin".to_string()],
            },
            sla_compliance: SlaCompliance {
                sla_threshold_ms: 150.0,
                baseline_p99_ms: 145.0,
                lens_p99_ms: 147.0,
                both_systems_compliant: true,
                comparative_advantage: 2.0,
            },
        };
        
        assert!(!validate_performance_margin(&comparison).unwrap());
    }
}