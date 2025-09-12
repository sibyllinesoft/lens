//! # High-Performance Intent Router for Query Classification
//!
//! Production-ready intent routing system with:
//! - Zero-allocation pattern matching for hot paths
//! - LSP integration for symbol-aware routing
//! - Fallback strategies with performance monitoring
//! - Confidence-based routing decisions
//! - Extensible routing rules and custom handlers

use anyhow::{Context, Result};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, instrument, span, Level};

use super::query_classifier::{QueryClassification, QueryClassifier, QueryIntent};
use crate::lsp::{LspManager, LspSearchResponse, LspSearchResult, HintType, LspServerType, SymbolHint};
use crate::search::{SearchResult, SearchResultType};

/// LSP capability types for routing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LSPCapability {
    GotoDefinition,
    FindReferences,
    DocumentSymbol,
}

/// Intent routing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentRouterConfig {
    /// Confidence threshold for specialized routing
    pub confidence_threshold: f32,
    /// Maximum results for primary routing
    pub max_primary_results: usize,
    /// Enable LSP integration
    pub enable_lsp_routing: bool,
    /// Fallback timeout in milliseconds
    pub fallback_timeout_ms: u64,
    /// Cache size for routing decisions
    pub decision_cache_size: usize,
    /// Custom routing rules
    pub custom_rules: Vec<CustomRoutingRule>,
}

impl Default for IntentRouterConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.7,
            max_primary_results: 20,
            enable_lsp_routing: true,
            fallback_timeout_ms: 100,
            decision_cache_size: 1000,
            custom_rules: Vec::new(),
        }
    }
}

/// Custom routing rule for domain-specific behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomRoutingRule {
    pub name: String,
    pub pattern: String,
    pub target_intent: QueryIntent,
    pub confidence_boost: f32,
    pub custom_handler: Option<String>,
}

/// Intent routing result with comprehensive metadata
#[derive(Debug, Clone)]
pub struct IntentRoutingResult {
    /// Original query classification
    pub classification: QueryClassification,
    /// Primary candidates from specialized routing
    pub primary_candidates: Vec<SearchResult>,
    /// Whether fallback was triggered
    pub fallback_triggered: bool,
    /// Routing path taken
    pub routing_path: SmallVec<[String; 4]>,
    /// Confidence threshold met
    pub confidence_threshold_met: bool,
    /// LSP routing decision if applicable
    pub lsp_routing_decision: Option<LSPRoutingDecision>,
    /// Performance metrics for this routing
    pub performance_metrics: RoutingMetrics,
}

/// LSP-based routing decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSPRoutingDecision {
    /// Should route through LSP
    pub should_route: bool,
    /// Target LSP capability
    pub capability: LSPCapability,
    /// Routing confidence
    pub confidence: f32,
    /// Reasoning for routing decision
    pub reasoning: String,
    /// Expected LSP results to use
    pub expected_hints: Vec<LspSearchResult>,
}

/// Performance metrics for routing operations
#[derive(Debug, Clone, Default)]
pub struct RoutingMetrics {
    pub total_latency_ms: f64,
    pub lsp_routing_latency_ms: f64,
    pub primary_search_latency_ms: f64,
    pub fallback_search_latency_ms: f64,
    pub classification_latency_ms: f64,
    pub candidates_from_primary: usize,
    pub candidates_from_fallback: usize,
}

/// Search context for routing decisions
#[derive(Debug, Clone)]
pub struct SearchContext {
    pub query: String,
    pub mode: String,
    pub repo_path: Option<String>,
    pub file_context: Option<FileContext>,
    pub user_preferences: Option<UserPreferences>,
}

/// File context for LSP-aware routing
#[derive(Debug, Clone)]
pub struct FileContext {
    pub current_file: String,
    pub current_line: usize,
    pub current_column: usize,
    pub language: String,
    pub project_root: String,
}

/// User preferences for routing behavior
#[derive(Debug, Clone)]
pub struct UserPreferences {
    pub prefer_local_results: bool,
    pub language_preferences: Vec<String>,
    pub max_results: usize,
}

/// Result from a search handler
// SearchResult is already imported from crate::search at the top

/// Handler function type for different intents
pub type IntentHandler = Box<dyn for<'a> Fn(&'a str, &'a SearchContext) -> futures::future::BoxFuture<'a, Result<Vec<SearchResult>>> + Send + Sync>;

/// High-performance intent router with LSP integration
pub struct IntentRouter {
    config: IntentRouterConfig,
    classifier: QueryClassifier,
    lsp_manager: Option<Arc<LspManager>>,
    
    // Routing handlers
    definition_handler: Option<IntentHandler>,
    references_handler: Option<IntentHandler>,
    symbol_handler: Option<IntentHandler>,
    structural_handler: Option<IntentHandler>,
    natural_language_handler: Option<IntentHandler>,
    fallback_handler: Option<IntentHandler>,
    
    // Performance optimization
    decision_cache: Arc<DashMap<String, CachedRoutingDecision>>,
    hot_patterns: Arc<RwLock<Vec<HotPattern>>>,
    
    // Metrics
    metrics: Arc<parking_lot::RwLock<IntentRouterMetrics>>,
}

/// Cached routing decision for performance
#[derive(Debug, Clone)]
struct CachedRoutingDecision {
    classification: QueryClassification,
    lsp_decision: Option<LSPRoutingDecision>,
    cached_at: Instant,
    hit_count: u32,
}

/// Hot pattern for zero-allocation matching
#[derive(Debug, Clone)]
struct HotPattern {
    pattern: String,
    intent: QueryIntent,
    confidence: f32,
    hit_count: u64,
}

impl IntentRouter {
    /// Create new intent router with configuration
    pub async fn new(
        config: IntentRouterConfig,
        classifier: QueryClassifier,
        lsp_manager: Option<Arc<LspManager>>,
    ) -> Result<Self> {
        let decision_cache = Arc::new(DashMap::with_capacity(config.decision_cache_size));
        let hot_patterns = Arc::new(RwLock::new(Vec::new()));
        
        info!(
            "Initializing IntentRouter: confidence_threshold={}, lsp_enabled={}",
            config.confidence_threshold,
            config.enable_lsp_routing
        );
        
        Ok(Self {
            config,
            classifier,
            lsp_manager,
            definition_handler: None,
            references_handler: None,
            symbol_handler: None,
            structural_handler: None,
            natural_language_handler: None,
            fallback_handler: None,
            decision_cache,
            hot_patterns,
            metrics: Arc::new(parking_lot::RwLock::new(IntentRouterMetrics::default())),
        })
    }
    
    /// Register handler for specific intent
    pub fn register_handler(&mut self, intent: QueryIntent, handler: IntentHandler) {
        match intent {
            QueryIntent::Definition => self.definition_handler = Some(handler),
            QueryIntent::References => self.references_handler = Some(handler),
            QueryIntent::Symbol => self.symbol_handler = Some(handler),
            QueryIntent::Structural => self.structural_handler = Some(handler),
            QueryIntent::NaturalLanguage => self.natural_language_handler = Some(handler),
            QueryIntent::Lexical => self.fallback_handler = Some(handler),
            QueryIntent::SymbolSearch => self.symbol_handler = Some(handler),
            QueryIntent::StructuralSearch => self.structural_handler = Some(handler),
        }
        
        debug!("Registered handler for intent: {:?}", intent);
    }
    
    /// Route query with full intent analysis and LSP integration
    #[instrument(skip(self, context), fields(query = %context.query, mode = %context.mode))]
    pub async fn route_query(&self, context: &SearchContext) -> Result<IntentRoutingResult> {
        let start_time = Instant::now();
        let mut metrics = RoutingMetrics::default();
        
        // Check cache first
        let cache_key = self.generate_cache_key(context);
        if let Some(cached) = self.get_cached_decision(&cache_key) {
            return self.execute_cached_routing(context, cached, start_time).await;
        }
        
        // Step 1: Classify query intent
        let classification_start = Instant::now();
        let classification = self.classifier.classify(&context.query);
        metrics.classification_latency_ms = classification_start.elapsed().as_millis() as f64;
        
        // Step 2: LSP routing decision (if enabled and available)
        let mut lsp_decision = None;
        if self.config.enable_lsp_routing && self.lsp_manager.is_some() {
            let lsp_start = Instant::now();
            lsp_decision = self.make_lsp_routing_decision(context, &classification).await?;
            metrics.lsp_routing_latency_ms = lsp_start.elapsed().as_millis() as f64;
        }
        
        // Cache the decision
        self.cache_routing_decision(&cache_key, &classification, &lsp_decision);
        
        // Step 3: Execute routing strategy
        let routing_result = self.execute_routing_strategy(
            context,
            &classification,
            lsp_decision,
            metrics,
            start_time,
        ).await?;
        
        // Step 4: Record metrics and update hot patterns
        self.record_routing_metrics(&routing_result);
        self.update_hot_patterns(&context.query, &classification).await;
        
        Ok(routing_result)
    }
    
    /// Fast path routing for hot queries
    #[instrument(skip(self, context))]
    pub async fn route_query_fast(&self, context: &SearchContext) -> Result<(QueryIntent, Vec<SearchResult>)> {
        // Check hot patterns first
        if let Some((intent, confidence)) = self.match_hot_patterns(&context.query).await {
            if confidence > self.config.confidence_threshold {
                let results = self.execute_intent_handler(intent, context).await.unwrap_or_default();
                return Ok((intent, results));
            }
        }
        
        // Fast classification
        let (intent, confidence) = self.classifier.classify_fast(&context.query);
        
        if confidence > self.config.confidence_threshold {
            let results = self.execute_intent_handler(intent, context).await.unwrap_or_default();
            Ok((intent, results))
        } else {
            // Fallback to default handler
            let results = self.execute_fallback_handler(context).await.unwrap_or_default();
            Ok((QueryIntent::Lexical, results))
        }
    }
    
    /// Make LSP routing decision based on context and classification
    async fn make_lsp_routing_decision(
        &self,
        context: &SearchContext,
        classification: &QueryClassification,
    ) -> Result<Option<LSPRoutingDecision>> {
        let lsp_manager = match &self.lsp_manager {
            Some(manager) => manager,
            None => return Ok(None),
        };
        
        // Determine if LSP routing is beneficial
        let should_route = match classification.intent {
            QueryIntent::Definition | QueryIntent::References => {
                // Check if we have file context for precise LSP queries
                context.file_context.is_some() && classification.confidence > 0.8
            }
            QueryIntent::Symbol => {
                // Symbol queries benefit from LSP when we have project context
                context.repo_path.is_some() && classification.confidence > 0.7
            }
            _ => false,
        };
        
        if !should_route {
            return Ok(None);
        }
        
        // Select appropriate LSP capability
        let capability = match classification.intent {
            QueryIntent::Definition => LSPCapability::GotoDefinition,
            QueryIntent::References => LSPCapability::FindReferences,
            QueryIntent::Symbol => LSPCapability::DocumentSymbol,
            _ => return Ok(None),
        };
        
        // Get LSP search results
        let lsp_response = lsp_manager.search(&context.query, None).await
            .unwrap_or_else(|_| LspSearchResponse::default());
        
        let confidence = if lsp_response.lsp_results.is_empty() {
            classification.confidence * 0.7 // Reduce confidence if no hints available
        } else {
            classification.confidence * 1.1 // Boost confidence with LSP support
        }.min(1.0);
        
        Ok(Some(LSPRoutingDecision {
            should_route: true,
            capability,
            confidence,
            reasoning: format!("LSP routing for {} with {} results", 
                             classification.intent, lsp_response.lsp_results.len()),
            expected_hints: lsp_response.lsp_results,
        }))
    }
    
    /// Execute routing strategy based on classification and LSP decision
    async fn execute_routing_strategy(
        &self,
        context: &SearchContext,
        classification: &QueryClassification,
        lsp_decision: Option<LSPRoutingDecision>,
        mut metrics: RoutingMetrics,
        start_time: Instant,
    ) -> Result<IntentRoutingResult> {
        let mut routing_path = SmallVec::new();
        let mut primary_candidates = Vec::new();
        let mut fallback_triggered = false;
        
        // Check confidence threshold
        let confidence_threshold_met = classification.confidence >= self.config.confidence_threshold;
        
        // Execute LSP routing if decided
        if let Some(ref lsp_dec) = lsp_decision {
            if lsp_dec.should_route {
                routing_path.push("lsp_routing".to_string());
                
                if let Some(lsp_manager) = &self.lsp_manager {
                    let lsp_start = Instant::now();
                    let lsp_response = lsp_manager.search(
                        &context.query,
                        context.file_context.as_ref().map(|fc| fc.current_file.as_str()),
                    ).await.unwrap_or_default();
                    
                    // Convert LSP results to SearchResult format
                    primary_candidates = lsp_response.fallback_results;
                    
                    metrics.primary_search_latency_ms += lsp_start.elapsed().as_millis() as f64;
                }
                
                routing_path.push(format!("lsp_{:?}", lsp_dec.capability));
            }
        }
        
        // Execute intent-specific routing if LSP didn't provide results
        if primary_candidates.is_empty() && confidence_threshold_met {
            let intent_start = Instant::now();
            primary_candidates = self.execute_intent_handler(classification.intent, context).await
                .unwrap_or_default();
            metrics.primary_search_latency_ms += intent_start.elapsed().as_millis() as f64;
            
            routing_path.push(format!("intent_{}", classification.intent));
        }
        
        // Fallback if no results from specialized routing
        if primary_candidates.is_empty() && 
           matches!(classification.intent, QueryIntent::Definition | QueryIntent::References | QueryIntent::Symbol) {
            
            let fallback_start = Instant::now();
            primary_candidates = self.execute_fallback_handler(context).await
                .unwrap_or_default();
            metrics.fallback_search_latency_ms = fallback_start.elapsed().as_millis() as f64;
            
            routing_path.push("fallback_triggered".to_string());
            fallback_triggered = true;
        }
        
        // Truncate results to configured maximum
        primary_candidates.truncate(self.config.max_primary_results);
        
        metrics.total_latency_ms = start_time.elapsed().as_millis() as f64;
        metrics.candidates_from_primary = primary_candidates.len();
        
        Ok(IntentRoutingResult {
            classification: classification.clone(),
            primary_candidates,
            fallback_triggered,
            routing_path,
            confidence_threshold_met,
            lsp_routing_decision: lsp_decision,
            performance_metrics: metrics,
        })
    }
    
    /// Execute handler for specific intent
    async fn execute_intent_handler(
        &self,
        intent: QueryIntent,
        context: &SearchContext,
    ) -> Option<Vec<SearchResult>> {
        let handler = match intent {
            QueryIntent::Definition => &self.definition_handler,
            QueryIntent::References => &self.references_handler,
            QueryIntent::Symbol => &self.symbol_handler,
            QueryIntent::Structural => &self.structural_handler,
            QueryIntent::NaturalLanguage => &self.natural_language_handler,
            QueryIntent::Lexical => &self.fallback_handler,
            QueryIntent::SymbolSearch => &self.symbol_handler,
            QueryIntent::StructuralSearch => &self.structural_handler,
        };
        
        if let Some(handler) = handler {
            match handler(&context.query, context).await {
                Ok(results) => Some(results),
                Err(e) => {
                    debug!("Intent handler failed for {:?}: {}", intent, e);
                    None
                }
            }
        } else {
            debug!("No handler registered for intent: {:?}", intent);
            None
        }
    }
    
    /// Execute fallback handler
    async fn execute_fallback_handler(&self, context: &SearchContext) -> Option<Vec<SearchResult>> {
        if let Some(handler) = &self.fallback_handler {
            match handler(&context.query, context).await {
                Ok(results) => Some(results),
                Err(e) => {
                    debug!("Fallback handler failed: {}", e);
                    None
                }
            }
        } else {
            None
        }
    }
    
    /// Match query against hot patterns for fast routing
    async fn match_hot_patterns(&self, query: &str) -> Option<(QueryIntent, f32)> {
        let patterns = self.hot_patterns.read().await;
        
        for pattern in patterns.iter() {
            if query.starts_with(&pattern.pattern) || query.contains(&pattern.pattern) {
                return Some((pattern.intent, pattern.confidence));
            }
        }
        
        None
    }
    
    /// Update hot patterns based on routing frequency
    async fn update_hot_patterns(&self, query: &str, classification: &QueryClassification) {
        if classification.confidence < 0.8 {
            return; // Only track high-confidence patterns
        }
        
        // Extract pattern from query (simplified)
        let pattern = if query.len() > 20 {
            query[..20].to_string()
        } else {
            query.to_string()
        };
        
        let mut patterns = self.hot_patterns.write().await;
        
        // Update existing pattern or add new one
        if let Some(hot_pattern) = patterns.iter_mut().find(|p| p.pattern == pattern) {
            hot_pattern.hit_count += 1;
            hot_pattern.confidence = (hot_pattern.confidence + classification.confidence) / 2.0;
        } else if patterns.len() < 100 { // Limit hot patterns
            patterns.push(HotPattern {
                pattern,
                intent: classification.intent,
                confidence: classification.confidence,
                hit_count: 1,
            });
        }
        
        // Sort by hit count to prioritize frequent patterns
        patterns.sort_by(|a, b| b.hit_count.cmp(&a.hit_count));
    }
    
    /// Generate cache key for routing decision
    fn generate_cache_key(&self, context: &SearchContext) -> String {
        use std::hash::{Hash, Hasher};
        
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        context.query.hash(&mut hasher);
        context.mode.hash(&mut hasher);
        if let Some(ref file_ctx) = context.file_context {
            file_ctx.current_file.hash(&mut hasher);
            file_ctx.language.hash(&mut hasher);
        }
        
        format!("route_{}", hasher.finish())
    }
    
    /// Cache routing decision
    fn cache_routing_decision(
        &self,
        cache_key: &str,
        classification: &QueryClassification,
        lsp_decision: &Option<LSPRoutingDecision>,
    ) {
        if self.decision_cache.len() >= self.config.decision_cache_size {
            // Simple LRU: remove oldest entry
            if let Some(entry) = self.decision_cache.iter().next() {
                let key = entry.key().clone();
                self.decision_cache.remove(&key);
            }
        }
        
        self.decision_cache.insert(
            cache_key.to_string(),
            CachedRoutingDecision {
                classification: classification.clone(),
                lsp_decision: lsp_decision.clone(),
                cached_at: Instant::now(),
                hit_count: 0,
            },
        );
    }
    
    /// Get cached routing decision if available and valid
    fn get_cached_decision(&self, cache_key: &str) -> Option<CachedRoutingDecision> {
        if let Some(mut cached) = self.decision_cache.get_mut(cache_key) {
            // Check if cache entry is still valid (5 minutes TTL)
            if cached.cached_at.elapsed() < Duration::from_secs(300) {
                cached.hit_count += 1;
                return Some(cached.clone());
            } else {
                // Remove expired entry
                drop(cached);
                self.decision_cache.remove(cache_key);
            }
        }
        None
    }
    
    /// Execute cached routing decision
    async fn execute_cached_routing(
        &self,
        context: &SearchContext,
        cached: CachedRoutingDecision,
        start_time: Instant,
    ) -> Result<IntentRoutingResult> {
        // Execute the same routing strategy as the cached decision
        let mut metrics = RoutingMetrics::default();
        metrics.total_latency_ms = start_time.elapsed().as_millis() as f64;
        
        let primary_candidates = self.execute_intent_handler(
            cached.classification.intent,
            context,
        ).await.unwrap_or_default();
        
        Ok(IntentRoutingResult {
            classification: cached.classification,
            primary_candidates,
            fallback_triggered: false,
            routing_path: smallvec::smallvec!["cached".to_string()],
            confidence_threshold_met: true,
            lsp_routing_decision: cached.lsp_decision,
            performance_metrics: metrics,
        })
    }
    
    /// Record routing metrics
    fn record_routing_metrics(&self, result: &IntentRoutingResult) {
        let mut metrics = self.metrics.write();
        metrics.total_routes += 1;
        metrics.total_latency += Duration::from_millis(result.performance_metrics.total_latency_ms as u64);
        
        // Record intent distribution
        *metrics.intent_counts.entry(result.classification.intent).or_insert(0) += 1;
        
        // Record routing path statistics
        for path_component in &result.routing_path {
            *metrics.path_counts.entry(path_component.clone()).or_insert(0) += 1;
        }
        
        // Record performance metrics
        if result.performance_metrics.lsp_routing_latency_ms > 0.0 {
            metrics.lsp_routes += 1;
            metrics.lsp_latency += Duration::from_millis(result.performance_metrics.lsp_routing_latency_ms as u64);
        }
        
        if result.fallback_triggered {
            metrics.fallback_routes += 1;
        }
        
        metrics.cache_hits += if result.routing_path.contains(&"cached".to_string()) { 1 } else { 0 };
    }
    
    /// Get router performance metrics
    pub fn get_metrics(&self) -> IntentRouterMetrics {
        self.metrics.read().clone()
    }
    
    /// Clear routing caches
    pub fn clear_caches(&self) {
        self.decision_cache.clear();
        info!("Cleared intent router caches");
    }
}

/// Performance metrics for intent router
#[derive(Debug, Clone, Default)]
pub struct IntentRouterMetrics {
    pub total_routes: u64,
    pub total_latency: Duration,
    pub lsp_routes: u64,
    pub lsp_latency: Duration,
    pub fallback_routes: u64,
    pub cache_hits: u64,
    pub intent_counts: std::collections::HashMap<QueryIntent, u64>,
    pub path_counts: std::collections::HashMap<String, u64>,
}

impl IntentRouterMetrics {
    pub fn avg_latency_ms(&self) -> f64 {
        if self.total_routes == 0 {
            0.0
        } else {
            self.total_latency.as_millis() as f64 / self.total_routes as f64
        }
    }
    
    pub fn lsp_route_percentage(&self) -> f64 {
        if self.total_routes == 0 {
            0.0
        } else {
            (self.lsp_routes as f64 / self.total_routes as f64) * 100.0
        }
    }
    
    pub fn fallback_rate(&self) -> f64 {
        if self.total_routes == 0 {
            0.0
        } else {
            (self.fallback_routes as f64 / self.total_routes as f64) * 100.0
        }
    }
    
    pub fn cache_hit_rate(&self) -> f64 {
        if self.total_routes == 0 {
            0.0
        } else {
            (self.cache_hits as f64 / self.total_routes as f64) * 100.0
        }
    }
}

/// Initialize intent router module
pub async fn initialize_router(config: &IntentRouterConfig) -> Result<()> {
    tracing::info!("Initializing intent router module");
    tracing::info!("Confidence threshold: {}", config.confidence_threshold);
    tracing::info!("Max primary results: {}", config.max_primary_results);
    tracing::info!("LSP routing enabled: {}", config.enable_lsp_routing);
    tracing::info!("Fallback timeout: {}ms", config.fallback_timeout_ms);
    tracing::info!("Decision cache size: {}", config.decision_cache_size);
    tracing::info!("Custom rules: {}", config.custom_rules.len());
    
    // Validate configuration
    if config.confidence_threshold < 0.0 || config.confidence_threshold > 1.0 {
        anyhow::bail!("Confidence threshold must be in range [0.0, 1.0]");
    }
    
    if config.max_primary_results == 0 {
        anyhow::bail!("Max primary results must be greater than 0");
    }
    
    if config.decision_cache_size == 0 {
        anyhow::bail!("Decision cache size must be greater than 0");
    }
    
    tracing::info!("Intent router module initialized successfully");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::query_classifier::{ClassifierConfig, QueryClassifier};
    
    #[tokio::test]
    async fn test_intent_router_creation() {
        let config = IntentRouterConfig::default();
        let classifier_config = ClassifierConfig::default();
        let classifier = QueryClassifier::new(classifier_config).unwrap();
        
        let router = IntentRouter::new(config, classifier, None).await;
        assert!(router.is_ok());
    }
    
    #[tokio::test]
    async fn test_routing_decision_caching() {
        let config = IntentRouterConfig::default();
        let classifier_config = ClassifierConfig::default();
        let classifier = QueryClassifier::new(classifier_config).unwrap();
        let router = IntentRouter::new(config, classifier, None).await.unwrap();
        
        let context = SearchContext {
            query: "def calculateSum".to_string(),
            mode: "hybrid".to_string(),
            repo_path: None,
            file_context: None,
            user_preferences: None,
        };
        
        let cache_key = router.generate_cache_key(&context);
        assert!(!cache_key.is_empty());
        
        // Test cache miss
        let cached = router.get_cached_decision(&cache_key);
        assert!(cached.is_none());
    }
    
    #[tokio::test]
    async fn test_hot_pattern_matching() {
        let config = IntentRouterConfig::default();
        let classifier_config = ClassifierConfig::default();
        let classifier = QueryClassifier::new(classifier_config).unwrap();
        let router = IntentRouter::new(config, classifier, None).await.unwrap();
        
        // Add a hot pattern
        {
            let mut patterns = router.hot_patterns.write().await;
            patterns.push(HotPattern {
                pattern: "def ".to_string(),
                intent: QueryIntent::Definition,
                confidence: 0.9,
                hit_count: 10,
            });
        }
        
        let result = router.match_hot_patterns("def calculateSum").await;
        assert!(result.is_some());
        let (intent, confidence) = result.unwrap();
        assert_eq!(intent, QueryIntent::Definition);
        assert_eq!(confidence, 0.9);
    }
    
    #[tokio::test]
    async fn test_lsp_routing_decision() {
        let config = IntentRouterConfig::default();
        let classifier_config = ClassifierConfig::default();
        let classifier = QueryClassifier::new(classifier_config).unwrap();
        let router = IntentRouter::new(config, classifier, None).await.unwrap();
        
        let context = SearchContext {
            query: "def calculateSum".to_string(),
            mode: "hybrid".to_string(),
            repo_path: Some("/path/to/repo".to_string()),
            file_context: Some(FileContext {
                current_file: "main.py".to_string(),
                current_line: 10,
                current_column: 5,
                language: "python".to_string(),
                project_root: "/path/to/repo".to_string(),
            }),
            user_preferences: None,
        };
        
        let classification = router.classifier.classify(&context.query);
        let decision = router.make_lsp_routing_decision(&context, &classification).await;
        
        assert!(decision.is_ok());
        // Without LSP manager, should return None
        assert!(decision.unwrap().is_none());
    }
    
    #[tokio::test]
    async fn test_fast_routing() {
        let config = IntentRouterConfig::default();
        let classifier_config = ClassifierConfig::default();
        let classifier = QueryClassifier::new(classifier_config).unwrap();
        let mut router = IntentRouter::new(config, classifier, None).await.unwrap();
        
        // Register a simple fallback handler
        router.register_handler(
            QueryIntent::Lexical,
            Box::new(|query, _context| {
                Box::pin(async move {
                    Ok(vec![SearchResult {
                        file_path: "test.py".to_string(),
                        line_number: 1,
                        column: 1,
                        content: query.to_string(),
                        score: 0.8,
                        result_type: SearchResultType::TextMatch,
                        language: Some("python".to_string()),
                        context_lines: None,
                        lsp_metadata: None,
                    }])
                })
            })
        );
        
        let context = SearchContext {
            query: "simple query".to_string(),
            mode: "hybrid".to_string(),
            repo_path: None,
            file_context: None,
            user_preferences: None,
        };
        
        let result = router.route_query_fast(&context).await;
        assert!(result.is_ok());
        
        let (intent, results) = result.unwrap();
        assert!(!results.is_empty());
    }
    
    #[test]
    fn test_metrics_calculation() {
        let mut metrics = IntentRouterMetrics::default();
        metrics.total_routes = 100;
        metrics.lsp_routes = 30;
        metrics.fallback_routes = 20;
        metrics.cache_hits = 15;
        metrics.total_latency = Duration::from_millis(5000);
        
        assert_eq!(metrics.lsp_route_percentage(), 30.0);
        assert_eq!(metrics.fallback_rate(), 20.0);
        assert_eq!(metrics.cache_hit_rate(), 15.0);
        assert_eq!(metrics.avg_latency_ms(), 50.0);
    }
    
    #[tokio::test]
    async fn test_configuration_validation() {
        let mut config = IntentRouterConfig::default();
        config.confidence_threshold = 1.5; // Invalid
        
        let result = initialize_router(&config).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Confidence threshold"));
        
        config.confidence_threshold = 0.7; // Valid
        config.max_primary_results = 0; // Invalid
        
        let result = initialize_router(&config).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Max primary results"));
    }
}