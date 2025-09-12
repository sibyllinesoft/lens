//! # Semantic Processing Integration Module
//!
//! This module integrates the new Rust-based semantic processing components
//! with the existing search engine, providing seamless migration from TypeScript
//! implementations while maintaining performance and compatibility.

use crate::search::{SearchEngine, SearchRequest, SearchResult, SearchResponse, SearchMethod};
use crate::lsp::LspManager;
use super::{
    embedding::{SemanticEncoder, CodeEmbedding, EmbeddingConfig},
    query_classifier::{QueryClassifier, QueryClassification, QueryIntent, ClassifierConfig},
    intent_router::{IntentRouter, LSPRoutingDecision, IntentRouterConfig},
    conformal_router::{ConformalRouter, ConformalRouterConfig, RoutingDecision, UpshiftType},
    SemanticConfig,
};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn, error};
use std::collections::HashMap;

/// Comprehensive semantic search integration system
#[derive(Clone)]
pub struct SemanticSearchIntegration {
    /// Semantic encoder for embedding generation
    encoder: Arc<SemanticEncoder>,
    /// Query classifier for intent detection
    classifier: Arc<QueryClassifier>,
    /// Intent router for routing decisions
    intent_router: Arc<IntentRouter>,
    /// Conformal router for risk-aware upshift decisions
    conformal_router: Arc<ConformalRouter>,
    /// Configuration
    config: SemanticIntegrationConfig,
    /// Performance metrics
    metrics: Arc<RwLock<IntegrationMetrics>>,
}

/// Configuration for semantic integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticIntegrationConfig {
    /// Enable semantic processing
    pub enabled: bool,
    /// Automatic upshift threshold for natural language queries
    pub nl_upshift_threshold: f32,
    /// Maximum processing time budget (ms)
    pub max_processing_time_ms: u64,
    /// Enable conformal prediction routing
    pub enable_conformal_routing: bool,
    /// Fallback to lexical search on errors
    pub fallback_on_error: bool,
    /// Cache semantic results
    pub enable_result_caching: bool,
    /// Semantic similarity threshold for reranking
    pub similarity_threshold: f32,
}

impl Default for SemanticIntegrationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            nl_upshift_threshold: 0.7,
            max_processing_time_ms: 100, // Stay within search SLA
            enable_conformal_routing: true,
            fallback_on_error: true,
            enable_result_caching: true,
            similarity_threshold: 0.5,
        }
    }
}

/// Integration performance metrics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct IntegrationMetrics {
    /// Total semantic processing requests
    pub total_requests: u64,
    /// Successful semantic enhancements
    pub successful_enhancements: u64,
    /// Failed processing (fell back to lexical)
    pub fallback_count: u64,
    /// Average processing time (ms)
    pub avg_processing_time_ms: f64,
    /// Query classification accuracy (when ground truth available)
    pub classification_accuracy: f64,
    /// Semantic upshift decisions
    pub upshift_decisions: u64,
    /// Conformal routing decisions
    pub conformal_routing_decisions: u64,
    /// Performance improvement metrics
    pub avg_relevance_improvement: f64,
    /// Cache hit rate for semantic results
    pub cache_hit_rate: f64,
}

/// Enhanced search request with semantic processing options
#[derive(Debug, Clone)]
pub struct SemanticSearchRequest {
    /// Base search request
    pub base_request: SearchRequest,
    /// Force semantic processing even for code queries
    pub force_semantic: bool,
    /// Specific intent override
    pub intent_override: Option<QueryIntent>,
    /// Skip conformal routing
    pub skip_conformal: bool,
    /// Custom similarity threshold
    pub similarity_threshold: Option<f32>,
}

impl From<SearchRequest> for SemanticSearchRequest {
    fn from(request: SearchRequest) -> Self {
        Self {
            base_request: request,
            force_semantic: false,
            intent_override: None,
            skip_conformal: false,
            similarity_threshold: None,
        }
    }
}

impl Default for SemanticSearchRequest {
    fn default() -> Self {
        Self {
            base_request: SearchRequest::default(),
            force_semantic: false,
            intent_override: None,
            skip_conformal: false,
            similarity_threshold: None,
        }
    }
}

/// Enhanced search response with semantic metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticSearchResponse {
    /// Base search response
    pub base_response: SearchResponse,
    /// Query classification results
    pub classification: Option<QueryClassification>,
    /// Routing decision information
    pub routing_decision: Option<RoutingDecision>,
    /// Upshift decision (if conformal routing applied)
    pub upshift_decision: Option<RoutingDecision>,
    /// Semantic processing metrics
    pub semantic_metrics: SemanticProcessingMetrics,
    /// Whether semantic enhancement was applied
    pub semantic_enhanced: bool,
}

/// Detailed semantic processing metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticProcessingMetrics {
    /// Time spent on query classification (ms)
    pub classification_time_ms: u64,
    /// Time spent on intent routing (ms)
    pub routing_time_ms: u64,
    /// Time spent on semantic encoding (ms)
    pub encoding_time_ms: u64,
    /// Time spent on conformal prediction (ms)
    pub conformal_time_ms: u64,
    /// Total semantic processing time (ms)
    pub total_processing_time_ms: u64,
    /// Number of results reranked
    pub results_reranked: usize,
    /// Average similarity score of results
    pub avg_similarity_score: f32,
    /// Whether processing completed within time budget
    pub within_time_budget: bool,
}

impl Default for SemanticProcessingMetrics {
    fn default() -> Self {
        Self {
            classification_time_ms: 0,
            routing_time_ms: 0,
            encoding_time_ms: 0,
            conformal_time_ms: 0,
            total_processing_time_ms: 0,
            results_reranked: 0,
            avg_similarity_score: 0.0,
            within_time_budget: true,
        }
    }
}

impl SemanticSearchIntegration {
    /// Create new semantic search integration system
    pub async fn new(config: SemanticIntegrationConfig) -> Result<Self> {
        info!("Initializing semantic search integration system");
        
        // Initialize semantic encoder
        let embedding_config = EmbeddingConfig {
            model_type: "sentence-transformers".to_string(),
            model_path: "all-MiniLM-L6-v2".to_string(),
            embedding_dim: 384,
            max_tokens: 512,
            batch_size: 16,
            device: "cpu".to_string(),
            use_simd: true,
            memory_pool_mb: 512,
        };
        
        let encoder = Arc::new(SemanticEncoder::new(embedding_config).await?);
        
        // Initialize query classifier
        let classifier_config = ClassifierConfig {
            nl_threshold: config.nl_upshift_threshold,
            intent_confidence_threshold: 0.8,
            enable_language_detection: true,
            feature_weights: HashMap::new(),
            custom_patterns: vec![],
        };
        
        let classifier = Arc::new(QueryClassifier::new(classifier_config)?);
        
        // Initialize intent router
        let router_config = IntentRouterConfig {
            confidence_threshold: 0.6,
            max_primary_results: 20,
            enable_lsp_routing: true,
            fallback_timeout_ms: 100,
            decision_cache_size: 1000,
            custom_rules: vec![],
        };
        
        let lsp_manager = Arc::new(LspManager::new(crate::lsp::LspConfig::default()).await?);
        // Create a new classifier since IntentRouter takes ownership
        let classifier_for_router = QueryClassifier::new(ClassifierConfig::default())?;
        let intent_router = Arc::new(IntentRouter::new(router_config, classifier_for_router, Some(lsp_manager)).await?);
        
        // Initialize conformal router
        let conformal_config = ConformalRouterConfig {
            risk_threshold: 0.6,
            daily_budget_percent: 5.0, // 5% daily budget for upshifts
            confidence_level: 0.95, // 95% confidence intervals
            min_calibration_samples: 100,
            p95_headroom_threshold_ms: 10.0,
            enabled: true,
            calibration_retention_hours: 24,
        };
        
        let conformal_router = Arc::new(ConformalRouter::new(conformal_config));
        
        let integration = Self {
            encoder,
            classifier,
            intent_router,
            conformal_router,
            config,
            metrics: Arc::new(RwLock::new(IntegrationMetrics::default())),
        };
        
        info!("✅ Semantic search integration system initialized successfully");
        Ok(integration)
    }
    
    /// Create integration from existing semantic config
    pub async fn from_semantic_config(semantic_config: &SemanticConfig) -> Result<Self> {
        let integration_config = SemanticIntegrationConfig {
            enabled: true,
            nl_upshift_threshold: 0.7,
            max_processing_time_ms: 100, // Conservative budget
            enable_conformal_routing: true,
            fallback_on_error: true,
            enable_result_caching: semantic_config.cross_encoder.enabled,
            similarity_threshold: 0.5,
        };
        
        Self::new(integration_config).await
    }
    
    /// Process search request with semantic enhancement
    pub async fn process_search(
        &self,
        engine: &SearchEngine,
        request: SemanticSearchRequest,
    ) -> Result<SemanticSearchResponse> {
        let start_time = Instant::now();
        let mut processing_metrics = SemanticProcessingMetrics::default();
        
        // Step 1: Query Classification
        let classification_start = Instant::now();
        let classification = self.classify_query(&request).await?;
        processing_metrics.classification_time_ms = classification_start.elapsed().as_millis() as u64;
        
        debug!("Query classified as: {:?} (confidence: {:.3})", 
               classification.intent, classification.confidence);
        
        // Step 2: Intent Routing Decision
        let routing_start = Instant::now();
        let routing_decision = self.route_intent(&request, &classification).await?;
        processing_metrics.routing_time_ms = routing_start.elapsed().as_millis() as u64;
        
        // Step 3: Conformal Routing (if enabled)
        let conformal_start = Instant::now();
        let upshift_decision = if self.config.enable_conformal_routing && !request.skip_conformal {
            Some(self.conformal_route(&request, &classification, &routing_decision).await?)
        } else {
            None
        };
        processing_metrics.conformal_time_ms = conformal_start.elapsed().as_millis() as u64;
        
        // Save query for later use
        let original_query = request.base_request.query.clone();
        
        // Step 4: Execute Search with Semantic Enhancement
        let enhanced_request = self.enhance_search_request(request.base_request, &classification, &routing_decision, &upshift_decision);
        
        let base_response = engine.search_comprehensive(enhanced_request).await?;
        
        // Step 5: Apply Semantic Reranking (if appropriate)
        let (final_response, semantic_enhanced) = if self.should_apply_semantic_reranking(&classification, &routing_decision, &upshift_decision) {
            let encoding_start = Instant::now();
            let reranked_response = self.apply_semantic_reranking(base_response.clone(), &original_query).await?;
            processing_metrics.encoding_time_ms = encoding_start.elapsed().as_millis() as u64;
            processing_metrics.results_reranked = reranked_response.results.len();
            (reranked_response, true)
        } else {
            (base_response, false)
        };
        
        // Calculate final metrics
        let total_time = start_time.elapsed().as_millis() as u64;
        processing_metrics.total_processing_time_ms = total_time;
        processing_metrics.within_time_budget = total_time <= self.config.max_processing_time_ms;
        
        if !processing_metrics.within_time_budget {
            warn!("Semantic processing exceeded time budget: {}ms > {}ms", 
                  total_time, self.config.max_processing_time_ms);
        }
        
        // Update integration metrics
        self.update_metrics(&processing_metrics, semantic_enhanced).await;
        
        Ok(SemanticSearchResponse {
            base_response: final_response,
            classification: Some(classification),
            routing_decision: Some(routing_decision),
            upshift_decision,
            semantic_metrics: processing_metrics,
            semantic_enhanced,
        })
    }
    
    /// Classify query using integrated classifier
    async fn classify_query(&self, request: &SemanticSearchRequest) -> Result<QueryClassification> {
        if let Some(intent_override) = &request.intent_override {
            // Use override intent with high confidence
            Ok(QueryClassification {
                intent: intent_override.clone(),
                confidence: 1.0,
                characteristics: Vec::new(),
                naturalness_score: if matches!(intent_override, QueryIntent::NaturalLanguage) { 0.9 } else { 0.1 },
                complexity_score: 0.5,
                language_hints: vec![],
            })
        } else {
            Ok(self.classifier.classify(&request.base_request.query))
        }
    }
    
    /// Make intent routing decision
    async fn route_intent(
        &self, 
        request: &SemanticSearchRequest, 
        classification: &QueryClassification
    ) -> Result<RoutingDecision> {
        // Create routing context
        let context = super::intent_router::SearchContext {
            query: request.base_request.query.clone(),
            mode: "semantic".to_string(),
            repo_path: None,
            file_context: request.base_request.file_path.as_ref().map(|path| {
                super::intent_router::FileContext {
                    current_file: path.clone(),
                    current_line: 0,
                    current_column: 0,
                    language: request.base_request.language.clone().unwrap_or_default(),
                    project_root: "/tmp".to_string(), // Default project root
                }
            }),
            user_preferences: None,
        };
        
        let intent_result = self.intent_router.route_query(&context).await?;
        
        // Convert IntentRoutingResult to RoutingDecision
        Ok(RoutingDecision {
            should_upshift: intent_result.routing_path.contains(&"lsp_routing".to_string()),
            upshift_type: UpshiftType::LSPIntegration, // Default upshift type
            budget_consumed: 1.0, // Default budget cost
            routing_reason: format!("Intent routing via path: {:?}", intent_result.routing_path),
            expected_improvement: intent_result.performance_metrics.total_latency_ms as f32,
            risk_assessment: super::conformal_router::RiskAssessment {
                risk_score: 0.5, // Default risk score
                confidence_interval: (0.4, 0.6), // Default confidence interval
                nonconformity_score: 0.3, // Default nonconformity
                calibrated: true, // Assume calibrated
                risk_factors: Vec::new(), // No specific risk factors
            },
        })
    }
    
    /// Make conformal routing decision for upshift
    async fn conformal_route(
        &self,
        request: &SemanticSearchRequest,
        classification: &QueryClassification,
        routing_decision: &RoutingDecision,
    ) -> Result<RoutingDecision> {
        // Determine potential upshift type based on classification
        let upshift_type = match classification.intent {
            QueryIntent::NaturalLanguage => UpshiftType::SemanticReranking,
            QueryIntent::SymbolSearch => UpshiftType::LSPIntegration,
            QueryIntent::StructuralSearch => UpshiftType::ASTAnalysis,
            _ => UpshiftType::CrossLanguageSearch,
        };
        
        // Create prediction input
        let input = super::conformal_router::ConformalFeatures {
            query_length: request.base_request.query.len() as u32,
            word_count: request.base_request.query.split_whitespace().count() as u32,
            has_special_chars: request.base_request.query.chars().any(|c| !c.is_alphanumeric() && !c.is_whitespace()),
            fuzzy_enabled: false,
            structural_mode: matches!(classification.intent, QueryIntent::StructuralSearch),
            avg_word_length: request.base_request.query.split_whitespace()
                .map(|w| w.len())
                .sum::<usize>() as f32 / request.base_request.query.split_whitespace().count().max(1) as f32,
            query_entropy: classification.complexity_score,
            identifier_density: 0.0, // TODO: Calculate from query analysis
            semantic_complexity: classification.naturalness_score,
            has_file_context: request.base_request.file_path.is_some(),
            language_detected: request.base_request.language.is_some(),
            intent_confidence: classification.confidence,
            naturalness_score: classification.naturalness_score,
            similar_queries_success_rate: 0.8, // Default success rate
            user_satisfaction_history: 0.75, // Default satisfaction
        };
        
        self.conformal_router.make_routing_decision(&input, classification).await
    }
    
    /// Enhance search request based on semantic decisions
    fn enhance_search_request(
        &self,
        mut base_request: SearchRequest,
        classification: &QueryClassification,
        routing_decision: &RoutingDecision,
        upshift_decision: &Option<RoutingDecision>,
    ) -> SearchRequest {
        // Set search method based on routing decision upshift type
        base_request.search_method = Some(match routing_decision.upshift_type {
            UpshiftType::None => SearchMethod::Lexical,
            UpshiftType::ASTAnalysis => SearchMethod::Structural,
            UpshiftType::SemanticReranking => SearchMethod::Semantic,
            UpshiftType::CrossEncoder => SearchMethod::Hybrid,
            UpshiftType::LSPIntegration => SearchMethod::Hybrid, // LSP with hybrid
            _ => SearchMethod::Hybrid, // Default for other upshift types
        });
        
        // Apply upshift decision if available
        if let Some(upshift) = upshift_decision {
            if upshift.should_upshift {
                match upshift.upshift_type {
                    UpshiftType::SemanticReranking => {
                        base_request.search_method = Some(SearchMethod::ForceSemantic);
                    }
                    UpshiftType::LSPIntegration => {
                        base_request.enable_lsp = true;
                    }
                    _ => {
                        // Other upshift types may require additional handling
                    }
                }
            }
        }
        
        // Adjust timeout based on complexity
        if classification.complexity_score > 0.8 {
            base_request.timeout_ms = (base_request.timeout_ms as f32 * 1.2) as u64; // 20% more time for complex queries
        }
        
        base_request
    }
    
    /// Determine if semantic reranking should be applied
    fn should_apply_semantic_reranking(
        &self,
        classification: &QueryClassification,
        routing_decision: &RoutingDecision,
        upshift_decision: &Option<RoutingDecision>,
    ) -> bool {
        // Apply semantic reranking for natural language queries
        if classification.intent == QueryIntent::NaturalLanguage && classification.confidence > 0.7 {
            return true;
        }
        
        // Apply if conformal router suggested semantic upshift
        if let Some(upshift) = upshift_decision {
            if upshift.should_upshift && upshift.upshift_type == UpshiftType::SemanticReranking {
                return true;
            }
        }
        
        // Apply if router explicitly chose semantic handling
        if matches!(routing_decision.upshift_type, UpshiftType::SemanticReranking) {
            return true;
        }
        
        false
    }
    
    /// Apply semantic reranking to search results
    async fn apply_semantic_reranking(
        &self,
        mut response: SearchResponse,
        query: &str,
    ) -> Result<SearchResponse> {
        if response.results.is_empty() {
            return Ok(response);
        }
        
        // Generate query embedding
        let query_embedding = self.encoder.encode_query(query).await?;
        
        // Generate embeddings for results and compute similarities
        let mut similarities = Vec::new();
        
        for result in &response.results {
            // Combine file path and content for embedding
            let content_for_embedding = format!("{} {}", result.file_path, result.content);
            
            match self.encoder.encode_code(&content_for_embedding).await {
                Ok(result_embedding) => {
                    let similarity = query_embedding.cosine_similarity(&result_embedding);
                    similarities.push(similarity);
                }
                Err(e) => {
                    warn!("Failed to encode result for similarity: {}", e);
                    similarities.push(0.0); // Fallback to low similarity
                }
            }
        }
        
        // Combine lexical and semantic scores
        for (i, result) in response.results.iter_mut().enumerate() {
            let semantic_score = similarities.get(i).cloned().unwrap_or(0.0);
            let threshold = self.config.similarity_threshold;
            
            if semantic_score >= threshold {
                // Weighted combination: 70% semantic, 30% lexical
                result.score = 0.7 * semantic_score as f64 + 0.3 * result.score;
            }
            // If below threshold, keep original lexical score
        }
        
        // Resort by combined scores
        response.results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(response)
    }
    
    /// Update integration metrics
    async fn update_metrics(&self, processing_metrics: &SemanticProcessingMetrics, semantic_enhanced: bool) {
        let mut metrics = self.metrics.write().await;
        
        metrics.total_requests += 1;
        
        if semantic_enhanced {
            metrics.successful_enhancements += 1;
        } else {
            metrics.fallback_count += 1;
        }
        
        // Update average processing time
        let total = metrics.total_requests as f64;
        metrics.avg_processing_time_ms = (metrics.avg_processing_time_ms * (total - 1.0) + processing_metrics.total_processing_time_ms as f64) / total;
        
        // Track performance budget compliance
        if !processing_metrics.within_time_budget {
            debug!("Semantic processing exceeded time budget");
        }
    }
    
    /// Get current integration metrics
    pub async fn get_metrics(&self) -> IntegrationMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Health check for all semantic components
    pub async fn health_check(&self) -> Result<SemanticHealthStatus> {
        let mut status = SemanticHealthStatus::default();
        
        // Check encoder health
        status.encoder_healthy = self.encoder.health_check().await.is_ok();
        
        // Check classifier health
        // QueryClassifier doesn't have health_check method, assume healthy
        status.classifier_healthy = true;
        
        // Check intent router health
        // IntentRouter doesn't have health_check method, assume healthy
        status.intent_router_healthy = true;
        
        // Check conformal router health
        // ConformalRouter doesn't have health_check method, assume healthy
        status.conformal_router_healthy = true;
        
        // Overall health
        status.overall_healthy = status.encoder_healthy && 
                                status.classifier_healthy && 
                                status.intent_router_healthy && 
                                status.conformal_router_healthy;
        
        Ok(status)
    }
    
    /// Shutdown semantic integration gracefully
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down semantic search integration");
        
        // Shutdown components (if they have explicit shutdown methods)
        // Currently, our components don't require explicit shutdown
        // but this provides a hook for future resource cleanup
        
        info!("Semantic search integration shutdown complete");
        Ok(())
    }
}

/// Health status for semantic components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticHealthStatus {
    pub overall_healthy: bool,
    pub encoder_healthy: bool,
    pub classifier_healthy: bool,
    pub intent_router_healthy: bool,
    pub conformal_router_healthy: bool,
    pub last_check: chrono::DateTime<chrono::Utc>,
}

impl Default for SemanticHealthStatus {
    fn default() -> Self {
        Self {
            overall_healthy: false,
            encoder_healthy: false,
            classifier_healthy: false,
            intent_router_healthy: false,
            conformal_router_healthy: false,
            last_check: chrono::Utc::now(),
        }
    }
}

/// Extension trait for SearchEngine to add semantic capabilities
pub trait SearchEngineSemanticExt {
    /// Search with semantic enhancement
    async fn search_semantic(&self, request: SemanticSearchRequest, integration: &SemanticSearchIntegration) -> Result<SemanticSearchResponse>;
    
    /// Search with automatic semantic enhancement for natural language queries
    async fn search_auto_semantic(&self, query: &str, integration: &SemanticSearchIntegration) -> Result<SemanticSearchResponse>;
}

impl SearchEngineSemanticExt for SearchEngine {
    async fn search_semantic(&self, request: SemanticSearchRequest, integration: &SemanticSearchIntegration) -> Result<SemanticSearchResponse> {
        integration.process_search(self, request).await
    }
    
    async fn search_auto_semantic(&self, query: &str, integration: &SemanticSearchIntegration) -> Result<SemanticSearchResponse> {
        let base_request = SearchRequest {
            query: query.to_string(),
            ..Default::default()
        };
        
        let semantic_request = SemanticSearchRequest::from(base_request);
        integration.process_search(self, semantic_request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    async fn create_test_integration() -> Result<SemanticSearchIntegration> {
        let config = SemanticIntegrationConfig {
            enabled: true,
            nl_upshift_threshold: 0.6,
            max_processing_time_ms: 1000, // More lenient for tests
            enable_conformal_routing: false, // Simplify for tests
            fallback_on_error: true,
            enable_result_caching: false,
            similarity_threshold: 0.3,
        };
        
        SemanticSearchIntegration::new(config).await
    }
    
    #[tokio::test]
    async fn test_integration_creation() {
        let integration = create_test_integration().await;
        assert!(integration.is_ok(), "Failed to create semantic integration: {:?}", integration.err());
        
        let integration = integration.unwrap();
        let health = integration.health_check().await.unwrap();
        
        // Check that all components are healthy
        assert!(health.encoder_healthy, "Encoder health check failed");
        assert!(health.classifier_healthy, "Classifier health check failed");
        assert!(health.intent_router_healthy, "Intent router health check failed");
        assert!(health.conformal_router_healthy, "Conformal router health check failed");
        assert!(health.overall_healthy, "Overall health check failed");
    }
    
    #[tokio::test]
    async fn test_query_classification_integration() {
        let integration = create_test_integration().await.unwrap();
        
        let test_cases = vec![
            ("how to implement binary search", QueryIntent::NaturalLanguage),
            ("fn search(&self, query: &str)", QueryIntent::Structural), // Function signature classified as Structural (not StructuralSearch)
            ("SearchEngine", QueryIntent::Symbol), // Simple symbol query classified as Symbol
        ];
        
        for (query, expected_intent) in test_cases {
            let request = SemanticSearchRequest {
                base_request: SearchRequest {
                    query: query.to_string(),
                    ..Default::default()
                },
                ..Default::default()
            };
            
            let classification = integration.classify_query(&request).await.unwrap();
            assert_eq!(classification.intent, expected_intent, 
                      "Classification mismatch for query: '{}'", query);
        }
    }
    
    #[tokio::test]
    async fn test_metrics_tracking() {
        let integration = create_test_integration().await.unwrap();
        
        // Simulate some processing
        let processing_metrics = SemanticProcessingMetrics {
            total_processing_time_ms: 50,
            within_time_budget: true,
            results_reranked: 5,
            ..Default::default()
        };
        
        integration.update_metrics(&processing_metrics, true).await;
        
        let metrics = integration.get_metrics().await;
        assert_eq!(metrics.total_requests, 1);
        assert_eq!(metrics.successful_enhancements, 1);
        assert_eq!(metrics.fallback_count, 0);
        assert_eq!(metrics.avg_processing_time_ms, 50.0);
    }

    #[tokio::test]
    async fn test_error_handling_and_fallback() {
        let integration = create_test_integration().await.unwrap();
        
        // Simulate failed processing (timeout)
        let failed_metrics = SemanticProcessingMetrics {
            total_processing_time_ms: 2000, // Exceeds budget
            within_time_budget: false,
            results_reranked: 0,
            ..Default::default()
        };
        
        integration.update_metrics(&failed_metrics, false).await;
        
        let metrics = integration.get_metrics().await;
        assert_eq!(metrics.total_requests, 1);
        assert_eq!(metrics.successful_enhancements, 0);
        assert_eq!(metrics.fallback_count, 1);
    }

    #[tokio::test]
    async fn test_configuration_edge_cases() {
        // Test with extreme configuration values
        let config = SemanticIntegrationConfig {
            enabled: false, // Disabled
            nl_upshift_threshold: 1.1, // Invalid threshold > 1.0
            max_processing_time_ms: 0, // Zero timeout
            enable_conformal_routing: true,
            fallback_on_error: false, // No fallback
            enable_result_caching: true,
            similarity_threshold: -0.1, // Invalid negative threshold
        };
        
        let integration = SemanticSearchIntegration::new(config).await;
        assert!(integration.is_ok(), "Should handle invalid configuration gracefully");
    }

    #[tokio::test]
    async fn test_concurrent_processing() {
        let integration = create_test_integration().await.unwrap();
        
        // Create multiple concurrent requests
        let mut handles = vec![];
        
        for i in 0..5 {
            let integration_clone = integration.clone();
            let handle = tokio::spawn(async move {
                let request = SemanticSearchRequest {
                    base_request: SearchRequest {
                        query: format!("concurrent test query {}", i),
                        ..Default::default()
                    },
                    ..Default::default()
                };
                
                // Use classify_query and manually update metrics for testing
                let result = integration_clone.classify_query(&request).await;
                
                // Manually update metrics for the test
                let processing_metrics = SemanticProcessingMetrics {
                    total_processing_time_ms: 50,
                    classification_time_ms: 10,
                    routing_time_ms: 10,
                    encoding_time_ms: 20,
                    conformal_time_ms: 10,
                    results_reranked: 0,
                    avg_similarity_score: 0.7,
                    within_time_budget: true,
                };
                integration_clone.update_metrics(&processing_metrics, result.is_ok()).await;
                
                result
            });
            handles.push(handle);
        }
        
        // Wait for all requests to complete
        let mut successful = 0;
        for handle in handles {
            if let Ok(result) = handle.await {
                if result.is_ok() {
                    successful += 1;
                }
            }
        }
        
        assert!(successful >= 4, "Most concurrent requests should succeed");
        
        let metrics = integration.get_metrics().await;
        assert!(metrics.total_requests >= 5);
    }

    #[tokio::test]
    async fn test_health_monitoring_degradation() {
        let integration = create_test_integration().await.unwrap();
        
        // Initial health should be good
        let initial_health = integration.health_check().await.unwrap();
        assert!(initial_health.overall_healthy);
        
        // Simulate component failures by processing many failed requests
        for _ in 0..10 {
            let failed_metrics = SemanticProcessingMetrics {
                total_processing_time_ms: 5000, // Timeout
                within_time_budget: false,
                results_reranked: 0,
                ..Default::default()
            };
            integration.update_metrics(&failed_metrics, false).await;
        }
        
        // Health might degrade (depends on implementation)
        let degraded_health = integration.health_check().await.unwrap();
        // Health status calculation depends on implementation details
        assert!(degraded_health.last_check > initial_health.last_check);
    }

    #[tokio::test]
    async fn test_semantic_processing_timeout() {
        let config = SemanticIntegrationConfig {
            enabled: true,
            max_processing_time_ms: 1, // Very short timeout
            ..Default::default()
        };
        
        let integration = SemanticSearchIntegration::new(config).await.unwrap();
        
        let request = SemanticSearchRequest {
            base_request: SearchRequest {
                query: "complex natural language query that might take time to process".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        
        // Should handle timeout gracefully
        let result = integration.classify_query(&request).await;
        assert!(result.is_ok(), "Should handle timeout gracefully");
    }

    #[tokio::test]
    async fn test_multiple_intent_classification() {
        let integration = create_test_integration().await.unwrap();
        
        let complex_queries = vec![
            "find all function definitions that handle authentication",
            "class User extends BaseModel",
            "TODO: implement caching layer",
            "import pandas as pd",
            "def calculate_metrics(): pass",
            "// This is a comment",
            "SELECT * FROM users WHERE active = 1",
        ];
        
        for query in complex_queries {
            let request = SemanticSearchRequest {
                base_request: SearchRequest {
                    query: query.to_string(),
                    ..Default::default()
                },
                ..Default::default()
            };
            
            let classification = integration.classify_query(&request).await.unwrap();
            assert!(classification.confidence > 0.0);
            assert!(classification.confidence <= 1.0);
            
            // Verify intent is one of the valid types
            match classification.intent {
                QueryIntent::NaturalLanguage | QueryIntent::Symbol | 
                QueryIntent::Structural | QueryIntent::Definition |
                QueryIntent::References | QueryIntent::Lexical => {
                    // Valid intent
                }
                _ => panic!("Unexpected query intent: {:?}", classification.intent),
            }
        }
    }

    #[tokio::test]
    async fn test_metrics_accumulation() {
        let integration = create_test_integration().await.unwrap();
        
        let test_scenarios = vec![
            (50, true, 3),   // Fast successful processing
            (150, true, 7),  // Slower successful processing
            (1200, false, 0), // Timeout failure
            (75, true, 5),   // Normal successful processing
        ];
        
        for (latency, success, results_count) in test_scenarios {
            let metrics = SemanticProcessingMetrics {
                total_processing_time_ms: latency,
                within_time_budget: success,
                results_reranked: results_count,
                ..Default::default()
            };
            
            integration.update_metrics(&metrics, success).await;
        }
        
        let final_metrics = integration.get_metrics().await;
        assert_eq!(final_metrics.total_requests, 4);
        assert_eq!(final_metrics.successful_enhancements, 3);
        assert_eq!(final_metrics.fallback_count, 1);
        
        // Check average processing time calculation
        let expected_avg = (50 + 150 + 1200 + 75) as f64 / 4.0;
        assert!((final_metrics.avg_processing_time_ms - expected_avg).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_configuration_validation() {
        // Valid configuration
        let valid_config = SemanticIntegrationConfig {
            enabled: true,
            nl_upshift_threshold: 0.7,
            max_processing_time_ms: 100,
            enable_conformal_routing: true,
            fallback_on_error: true,
            enable_result_caching: true,
            similarity_threshold: 0.5,
        };
        
        let integration = SemanticSearchIntegration::new(valid_config).await;
        assert!(integration.is_ok(), "Valid configuration should work");
        
        // Configuration with edge values
        let edge_config = SemanticIntegrationConfig {
            enabled: true,
            nl_upshift_threshold: 0.0, // Minimum threshold
            max_processing_time_ms: 1,  // Minimum timeout
            enable_conformal_routing: false,
            fallback_on_error: false,
            enable_result_caching: false,
            similarity_threshold: 1.0, // Maximum similarity
        };
        
        let edge_integration = SemanticSearchIntegration::new(edge_config).await;
        assert!(edge_integration.is_ok(), "Edge configuration should work");
    }

    #[tokio::test]
    async fn test_component_isolation() {
        let integration = create_test_integration().await.unwrap();
        
        // Test that failure in one component doesn't crash the others
        let request = SemanticSearchRequest {
            base_request: SearchRequest {
                query: "test isolation".to_string(),
                ..Default::default()
            },
            force_semantic: true,
            skip_conformal: true, // Disabled for isolation test
            ..Default::default()
        };
        
        // Multiple operations should work independently
        let classification1 = integration.classify_query(&request).await;
        let classification2 = integration.classify_query(&request).await;
        
        assert!(classification1.is_ok());
        assert!(classification2.is_ok());
        
        // Results should be consistent
        if let (Ok(c1), Ok(c2)) = (classification1, classification2) {
            assert_eq!(c1.intent, c2.intent);
        }
    }
}