//! # Semantic Search Module
//!
//! Phase 3: Advanced semantic search with performance constraints
//! - 2048-token encoder for long code context
//! - Hard negatives from SymbolGraph relationships  
//! - Learned reranking with isotonic regression
//! - Optional cross-encoder for precision boost
//! - Calibration preservation with ECE ≤ 0.005 drift

pub mod encoder;
pub mod hard_negatives; 
pub mod rerank;
pub mod cross_encoder;
pub mod calibration;
pub mod isotonic_calibration;
pub mod sla_bounded_evaluation;
pub mod pipeline;
pub mod validation;
pub mod ltr_trainer;

// Re-export main types for easier usage
pub use pipeline::{SemanticPipeline, SemanticSearchRequest, SemanticSearchResponse, initialize_semantic_pipeline};
pub use encoder::{SemanticEncoder, CodeEmbedding};
pub use rerank::{LearnedReranker, SearchResult, RerankedResult};
pub use cross_encoder::{CrossEncoder, QueryAnalysis};
pub use calibration::{CalibrationSystem, CalibrationMetrics};
pub use isotonic_calibration::{IsotonicCalibrationSystem, IsotonicCalibrationConfig, CalibrationTrainingSample};
pub use sla_bounded_evaluation::{SLABoundedEvaluator, SLAEvaluationConfig, SLABoundedEvaluationResult};
pub use hard_negatives::{HardNegativesGenerator, ContrastivePair};
pub use validation::{validate_phase3_implementation, ValidationResults};
pub use ltr_trainer::{LTRTrainer, LTRConfig, LTRObjective, BoundedLTRModel, TrainingReport};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for semantic search components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticConfig {
    /// 2048-token encoder settings
    pub encoder: EncoderConfig,
    /// Learned reranking configuration  
    pub rerank: RerankConfig,
    /// Cross-encoder for precision boost
    pub cross_encoder: CrossEncoderConfig,
    /// Calibration preservation settings
    pub calibration: CalibrationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderConfig {
    /// Model architecture (CodeT5/UniXcoder-class)
    pub model_type: String,
    /// Maximum token context window
    pub max_tokens: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Batch size for inference
    pub batch_size: usize,
    /// Device for inference (cpu/cuda/mps)
    pub device: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]  
pub struct RerankConfig {
    /// Top-K results to rerank
    pub top_k: usize,
    /// Use isotonic regression for score calibration
    pub use_isotonic: bool,
    /// Learning rate for isotonic fitting
    pub learning_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossEncoderConfig {
    /// Enable cross-encoder for precision boost  
    pub enabled: bool,
    /// Maximum inference time budget (≤50ms p95)
    pub max_inference_ms: u64,
    /// Query complexity threshold for activation
    pub complexity_threshold: f32,
    /// Top-K candidates for cross-encoding
    pub top_k: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// Maximum ECE drift allowed (≤0.005)
    pub max_ece_drift: f32,
    /// Cap dense features in log-odds space
    pub log_odds_cap: f32,
    /// Temperature scaling factor
    pub temperature: f32,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            encoder: EncoderConfig {
                model_type: "codet5-base".to_string(),
                max_tokens: 2048,
                embedding_dim: 768,
                batch_size: 16,
                device: "cpu".to_string(),
            },
            rerank: RerankConfig {
                top_k: 100,
                use_isotonic: true,
                learning_rate: 0.01,
            },
            cross_encoder: CrossEncoderConfig {
                enabled: false, // Start disabled for performance
                max_inference_ms: 50,
                complexity_threshold: 0.7,
                top_k: 10,
            },
            calibration: CalibrationConfig {
                max_ece_drift: 0.005,
                log_odds_cap: 5.0,
                temperature: 1.0,
            },
        }
    }
}

/// Performance constraints for semantic search
pub const SEMANTIC_INFERENCE_TARGET_MS: u64 = 50;
pub const SEMANTIC_P95_TARGET_MS: u64 = 50;
pub const COIR_NDCG_TARGET: f32 = 0.52;
pub const NL_IMPROVEMENT_TARGET_PP: f32 = 4.0; // 4-6pp target

/// Semantic search metrics for performance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMetrics {
    /// Inference latency statistics
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    
    /// Quality metrics
    pub ndcg_at_10: f32,
    pub success_at_10: f32,
    pub recall_at_50: f32,
    
    /// Calibration metrics
    pub expected_calibration_error: f32,
    pub ece_drift_from_baseline: f32,
    
    /// Natural language query performance  
    pub nl_slice_improvement_pp: f32,
    
    /// Performance gates
    pub meets_coir_target: bool,
    pub meets_latency_target: bool,
    pub meets_ece_target: bool,
}

impl SemanticMetrics {
    /// Check if all performance gates are met
    pub fn passes_gates(&self) -> bool {
        self.meets_coir_target && self.meets_latency_target && self.meets_ece_target
    }
    
    /// Validate against Phase 3 success criteria
    pub fn validate_phase3_gates(&self) -> Result<()> {
        if self.ndcg_at_10 < COIR_NDCG_TARGET {
            anyhow::bail!(
                "CoIR nDCG@10 {} < target {}", 
                self.ndcg_at_10, 
                COIR_NDCG_TARGET
            );
        }
        
        if self.latency_p95_ms > SEMANTIC_P95_TARGET_MS as f64 {
            anyhow::bail!(
                "Semantic p95 latency {}ms > target {}ms",
                self.latency_p95_ms,
                SEMANTIC_P95_TARGET_MS
            );
        }
        
        if self.ece_drift_from_baseline > 0.005 {
            anyhow::bail!(
                "ECE drift {} > target 0.005",
                self.ece_drift_from_baseline
            );
        }
        
        if self.nl_slice_improvement_pp < NL_IMPROVEMENT_TARGET_PP {
            anyhow::bail!(
                "NL improvement {}pp < target {}pp",
                self.nl_slice_improvement_pp,
                NL_IMPROVEMENT_TARGET_PP
            );
        }
        
        Ok(())
    }
}

/// Initialize semantic search module
pub async fn initialize_semantic(config: &SemanticConfig) -> Result<()> {
    tracing::info!("Initializing semantic search module");
    tracing::info!("Encoder: {} with {} tokens", config.encoder.model_type, config.encoder.max_tokens);
    tracing::info!("Rerank: top-{} with isotonic={}", config.rerank.top_k, config.rerank.use_isotonic);
    tracing::info!("Cross-encoder: enabled={}", config.cross_encoder.enabled);
    tracing::info!("Performance targets: CoIR nDCG@10 ≥ {}, p95 ≤ {}ms", 
                   COIR_NDCG_TARGET, SEMANTIC_P95_TARGET_MS);
    
    // Initialize encoder
    encoder::initialize_encoder(&config.encoder).await?;
    
    // Initialize reranker
    let rerank_config = rerank::RerankConfig {
        top_k: config.rerank.top_k,
        use_isotonic: config.rerank.use_isotonic,
        learning_rate: config.rerank.learning_rate,
        l2_regularization: 0.01, // Default value since not in semantic config
        min_training_samples: 100, // Default value
        combination_strategy: rerank::CombinationStrategy::LearnedWeights,
    };
    rerank::initialize_reranker(&rerank_config).await?;
    
    // Initialize cross-encoder if enabled
    if config.cross_encoder.enabled {
        let cross_encoder_config = cross_encoder::CrossEncoderConfig {
            enabled: config.cross_encoder.enabled,
            max_inference_ms: config.cross_encoder.max_inference_ms,
            complexity_threshold: config.cross_encoder.complexity_threshold,
            top_k: config.cross_encoder.top_k,
            model_type: "bert-base-uncased".to_string(), // Default model type
            max_batch_size: 16, // Default value
            budget_strategy: cross_encoder::BudgetStrategy::FixedPerQuery,
        };
        cross_encoder::initialize_cross_encoder(&cross_encoder_config).await?;
    }
    
    // Initialize calibration system
    let calibration_config = calibration::CalibrationConfig {
        max_ece_drift: config.calibration.max_ece_drift,
        log_odds_cap: config.calibration.log_odds_cap,
        temperature: config.calibration.temperature,
        min_samples_for_calibration: 1000, // Default value
        measurement_window_size: 10000,    // Default value
        auto_temperature_adjustment: true,  // Enable automatic adjustment
    };
    calibration::initialize_calibration(&calibration_config).await?;
    
    tracing::info!("Semantic search module initialized successfully");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_config_default() {
        let config = SemanticConfig::default();
        assert_eq!(config.encoder.max_tokens, 2048);
        assert_eq!(config.encoder.model_type, "codet5-base");
        assert!(config.rerank.use_isotonic);
        assert!(!config.cross_encoder.enabled); // Start disabled
    }

    #[test] 
    fn test_semantic_metrics_validation() {
        let mut metrics = SemanticMetrics {
            latency_p50_ms: 25.0,
            latency_p95_ms: 45.0, // Within 50ms target
            latency_p99_ms: 75.0,
            ndcg_at_10: 0.53, // Above 0.52 target
            success_at_10: 0.65,
            recall_at_50: 0.85,
            expected_calibration_error: 0.012,
            ece_drift_from_baseline: 0.003, // Within 0.005 target
            nl_slice_improvement_pp: 5.2, // Above 4pp target
            meets_coir_target: true,
            meets_latency_target: true, 
            meets_ece_target: true,
        };
        
        // Should pass all gates
        assert!(metrics.validate_phase3_gates().is_ok());
        
        // Test failure cases
        metrics.ndcg_at_10 = 0.49; // Below target
        assert!(metrics.validate_phase3_gates().is_err());
        
        metrics.ndcg_at_10 = 0.53;
        metrics.latency_p95_ms = 65.0; // Above target  
        assert!(metrics.validate_phase3_gates().is_err());
    }
}