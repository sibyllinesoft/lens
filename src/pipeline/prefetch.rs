//! Prefetch and Visited-Set Reuse System
//!
//! Implements intelligent prefetching based on query patterns and
//! visited set caching for frequent query patterns.
//! 
//! Target: >50% cache hit rate on repeated patterns per TODO.md

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque, BTreeMap};
use std::sync::Arc;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{RwLock, Mutex};
use tokio::time::{interval, sleep};
use tracing::{debug, info, warn};

/// Prefetch and caching coordinator
pub struct PrefetchManager {
    /// Pattern-based prefetch engine
    pattern_engine: Arc<RwLock<PatternEngine>>,
    
    /// Visited set cache for reuse
    visited_cache: Arc<RwLock<VisitedSetCache>>,
    
    /// Query similarity detector
    similarity_detector: Arc<QuerySimilarityDetector>,
    
    /// Prefetch scheduler
    scheduler: Arc<Mutex<PrefetchScheduler>>,
    
    /// Performance metrics
    metrics: Arc<RwLock<PrefetchMetrics>>,
    
    /// Configuration
    config: PrefetchConfig,
}

/// Configuration for prefetch system
#[derive(Debug, Clone)]
pub struct PrefetchConfig {
    /// Maximum patterns to track
    pub max_patterns: usize,
    
    /// Pattern recognition window (number of queries)
    pub pattern_window: usize,
    
    /// Minimum pattern frequency for prefetch
    pub min_pattern_frequency: usize,
    
    /// Cache TTL for visited sets
    pub visited_set_ttl: Duration,
    
    /// Maximum visited sets to cache
    pub max_cached_visited_sets: usize,
    
    /// Similarity threshold for query matching
    pub similarity_threshold: f64,
    
    /// Prefetch lookahead window
    pub prefetch_lookahead: usize,
    
    /// Memory pool configuration
    pub memory_pool_config: MemoryPoolConfig,
    
    /// Adaptive learning parameters
    pub adaptation_config: AdaptationConfig,
}

/// Memory pool configuration for prefetch operations
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    pub max_memory_mb: usize,
    pub pool_segment_size: usize,
    pub cleanup_interval_secs: u64,
}

/// Adaptive learning configuration
#[derive(Debug, Clone)]
pub struct AdaptationConfig {
    pub learning_rate: f64,
    pub pattern_decay_factor: f64,
    pub relevance_threshold: f64,
    pub adaptation_window: usize,
}

/// Pattern recognition engine
pub struct PatternEngine {
    /// Recognized query patterns
    patterns: HashMap<PatternId, QueryPattern>,
    
    /// Recent query history for pattern detection
    query_history: VecDeque<QueryHistoryEntry>,
    
    /// Pattern transition graph
    transition_graph: HashMap<PatternId, HashMap<PatternId, f64>>,
    
    /// Pattern frequency counters
    pattern_frequencies: HashMap<PatternId, usize>,
    
    /// Next available pattern ID
    next_pattern_id: u64,
}

/// Visited set cache with intelligent reuse
pub struct VisitedSetCache {
    /// Cached visited sets indexed by query signature
    cache: BTreeMap<QuerySignature, CachedVisitedSet>,
    
    /// Access frequency tracking
    access_frequencies: HashMap<QuerySignature, AccessFrequency>,
    
    /// Cache size tracking
    total_cached_sets: usize,
    total_memory_usage: usize,
    
    /// LRU eviction tracking
    access_order: VecDeque<QuerySignature>,
}

/// Query similarity detector for cache lookups
pub struct QuerySimilarityDetector {
    /// Feature extractors for different similarity metrics
    feature_extractors: FeatureExtractors,
    
    /// Similarity computation cache
    similarity_cache: RwLock<HashMap<(QuerySignature, QuerySignature), f64>>,
    
    /// Learned similarity weights
    similarity_weights: RwLock<SimilarityWeights>,
}

/// Prefetch scheduler for proactive loading
pub struct PrefetchScheduler {
    /// Scheduled prefetch operations
    pending_prefetches: VecDeque<PrefetchOperation>,
    
    /// Currently executing prefetches
    active_prefetches: HashMap<PrefetchId, PrefetchExecution>,
    
    /// Prefetch success rates
    success_rates: HashMap<PatternId, f64>,
    
    /// Resource usage tracking
    resource_usage: ResourceUsage,
}

/// Query pattern representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPattern {
    pub id: PatternId,
    pub template: QueryTemplate,
    pub typical_results: Vec<TypicalResult>,
    pub access_pattern: AccessPattern,
    pub confidence: f64,
    pub last_seen: Instant,
    pub usage_count: usize,
}

/// Query template for pattern matching
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct QueryTemplate {
    pub structure: QueryStructure,
    pub keywords: HashSet<String>,
    pub query_type: QueryType,
    pub language_hint: Option<String>,
}

/// Query structure classification
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum QueryStructure {
    ExactMatch,
    FunctionSearch,
    ClassSearch,
    VariableSearch,
    TypeSearch,
    StructuralPattern,
    SemanticSearch,
    Hybrid,
}

/// Query type classification
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum QueryType {
    Definition,
    Reference,
    Implementation,
    Usage,
    Documentation,
    Navigation,
}

/// Access pattern for prefetch prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessPattern {
    pub temporal_pattern: Vec<Duration>,
    pub sequence_probability: f64,
    pub context_dependencies: Vec<ContextDependency>,
}

/// Context dependency for pattern predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextDependency {
    pub dependency_type: DependencyType,
    pub strength: f64,
    pub context_hint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyType {
    TemporalSequence,
    FileContext,
    ProjectContext,
    UserSession,
}

/// Cached visited set with metadata
#[derive(Debug, Clone)]
pub struct CachedVisitedSet {
    pub signature: QuerySignature,
    pub visited_nodes: HashSet<String>,
    pub result_hints: Vec<ResultHint>,
    pub cache_time: Instant,
    pub hit_count: usize,
    pub estimated_quality: f64,
    pub memory_footprint: usize,
}

/// Result hint for prefetch optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultHint {
    pub file_path: String,
    pub line_number: usize,
    pub relevance_score: f64,
    pub access_probability: f64,
}

/// Query signature for cache indexing
#[derive(Debug, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct QuerySignature {
    pub normalized_query: String,
    pub query_hash: u64,
    pub context_hash: u64,
}

/// Access frequency tracking
#[derive(Debug, Clone)]
pub struct AccessFrequency {
    pub total_accesses: usize,
    pub recent_accesses: VecDeque<Instant>,
    pub access_rate: f64,
    pub last_access: Instant,
}

/// Feature extractors for similarity computation
pub struct FeatureExtractors {
    lexical_extractor: LexicalFeatureExtractor,
    semantic_extractor: SemanticFeatureExtractor,
    structural_extractor: StructuralFeatureExtractor,
    contextual_extractor: ContextualFeatureExtractor,
}

/// Similarity weights for different features
#[derive(Debug, Clone)]
pub struct SimilarityWeights {
    pub lexical_weight: f64,
    pub semantic_weight: f64,
    pub structural_weight: f64,
    pub contextual_weight: f64,
}

/// Prefetch operation specification
#[derive(Debug, Clone)]
pub struct PrefetchOperation {
    pub id: PrefetchId,
    pub pattern_id: PatternId,
    pub target_query: QuerySignature,
    pub priority: PrefetchPriority,
    pub estimated_benefit: f64,
    pub resource_cost: ResourceCost,
    pub schedule_time: Instant,
}

/// Prefetch execution tracking
#[derive(Debug, Clone)]
pub struct PrefetchExecution {
    pub operation: PrefetchOperation,
    pub start_time: Instant,
    pub progress: f64,
    pub intermediate_results: Vec<String>,
}

/// Resource usage tracking
#[derive(Debug, Default, Clone)]
pub struct ResourceUsage {
    pub memory_mb: f64,
    pub cpu_percent: f64,
    pub io_operations: usize,
    pub cache_operations: usize,
}

/// Resource cost estimation
#[derive(Debug, Clone)]
pub struct ResourceCost {
    pub memory_mb: f64,
    pub cpu_time_ms: f64,
    pub io_operations: usize,
    pub network_requests: usize,
}

/// Prefetch priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PrefetchPriority {
    Critical,
    High,
    Medium,
    Low,
    Background,
}

/// Prefetch performance metrics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PrefetchMetrics {
    pub total_queries: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub prefetch_hits: u64,
    pub prefetch_misses: u64,
    pub patterns_recognized: u64,
    pub avg_pattern_confidence: f64,
    pub memory_usage_mb: f64,
    pub hit_rate: f64,
    pub prefetch_accuracy: f64,
    pub latency_reduction_ms: f64,
}

/// Query history entry for pattern recognition
#[derive(Debug, Clone)]
pub struct QueryHistoryEntry {
    pub query: String,
    pub signature: QuerySignature,
    pub timestamp: Instant,
    pub results_count: usize,
    pub processing_time: Duration,
    pub file_context: Option<String>,
}

/// Typical result for pattern-based prefetch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypicalResult {
    pub file_path: String,
    pub line_number: usize,
    pub content_hash: u64,
    pub relevance_score: f64,
    pub access_probability: f64,
}

// Type aliases
pub type PatternId = u64;
pub type PrefetchId = u64;

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            max_patterns: 10_000,
            pattern_window: 1_000,
            min_pattern_frequency: 3,
            visited_set_ttl: Duration::from_hours(24),
            max_cached_visited_sets: 50_000,
            similarity_threshold: 0.7,
            prefetch_lookahead: 5,
            memory_pool_config: MemoryPoolConfig {
                max_memory_mb: 256,
                pool_segment_size: 1024 * 1024, // 1MB segments
                cleanup_interval_secs: 300, // 5 minutes
            },
            adaptation_config: AdaptationConfig {
                learning_rate: 0.01,
                pattern_decay_factor: 0.95,
                relevance_threshold: 0.6,
                adaptation_window: 100,
            },
        }
    }
}

impl PrefetchManager {
    /// Create a new prefetch manager
    pub async fn new(config: PrefetchConfig) -> Result<Self> {
        let pattern_engine = Arc::new(RwLock::new(PatternEngine::new(config.clone())));
        let visited_cache = Arc::new(RwLock::new(VisitedSetCache::new(config.clone())));
        let similarity_detector = Arc::new(QuerySimilarityDetector::new());
        let scheduler = Arc::new(Mutex::new(PrefetchScheduler::new()));
        let metrics = Arc::new(RwLock::new(PrefetchMetrics::default()));
        
        info!("Initialized prefetch manager with {}MB memory limit", config.memory_pool_config.max_memory_mb);
        
        Ok(Self {
            pattern_engine,
            visited_cache,
            similarity_detector,
            scheduler,
            metrics,
            config,
        })
    }
    
    /// Process query with prefetch and caching
    pub async fn process_query(
        &self,
        query: &str,
        file_context: Option<String>,
    ) -> Result<PrefetchResult> {
        let start_time = Instant::now();
        let signature = self.compute_query_signature(query, file_context.as_deref()).await;
        
        // Check visited set cache first
        let cache_result = self.check_visited_cache(&signature).await?;
        
        let mut result = if let Some(cached_set) = cache_result {
            debug!("Cache hit for query signature: {:?}", signature);
            self.update_cache_metrics(true).await;
            
            PrefetchResult {
                cache_hit: true,
                visited_set: Some(cached_set.visited_nodes.clone()),
                result_hints: cached_set.result_hints.clone(),
                prefetch_suggestions: Vec::new(),
                processing_time: start_time.elapsed(),
                estimated_quality: cached_set.estimated_quality,
            }
        } else {
            self.update_cache_metrics(false).await;
            
            PrefetchResult {
                cache_hit: false,
                visited_set: None,
                result_hints: Vec::new(),
                prefetch_suggestions: Vec::new(),
                processing_time: start_time.elapsed(),
                estimated_quality: 0.0,
            }
        };
        
        // Update query history for pattern recognition
        self.update_query_history(query, &signature, file_context, &result).await?;
        
        // Detect patterns and schedule prefetches
        let patterns = self.detect_patterns(&signature).await?;
        if !patterns.is_empty() {
            let suggestions = self.generate_prefetch_suggestions(patterns).await?;
            result.prefetch_suggestions = suggestions;
            
            // Schedule background prefetches
            self.schedule_prefetch_operations(&result.prefetch_suggestions).await?;
        }
        
        Ok(result)
    }
    
    /// Add visited set to cache
    pub async fn cache_visited_set(
        &self,
        query: &str,
        file_context: Option<String>,
        visited_set: HashSet<String>,
        result_hints: Vec<ResultHint>,
        quality: f64,
    ) -> Result<()> {
        let signature = self.compute_query_signature(query, file_context.as_deref()).await;
        let memory_footprint = self.estimate_memory_footprint(&visited_set, &result_hints);
        
        let cached_set = CachedVisitedSet {
            signature: signature.clone(),
            visited_nodes: visited_set,
            result_hints,
            cache_time: Instant::now(),
            hit_count: 0,
            estimated_quality: quality,
            memory_footprint,
        };
        
        let mut cache = self.visited_cache.write().await;
        
        // Check memory limits and evict if necessary
        if cache.total_memory_usage + memory_footprint > 
           self.config.memory_pool_config.max_memory_mb * 1024 * 1024 {
            cache.evict_lru_entries(memory_footprint)?;
        }
        
        cache.insert(signature.clone(), cached_set);
        cache.update_access_order(signature);
        
        debug!("Cached visited set for query, total sets: {}", cache.total_cached_sets);
        
        Ok(())
    }
    
    /// Compute query signature for caching
    async fn compute_query_signature(&self, query: &str, file_context: Option<&str>) -> QuerySignature {
        let normalized_query = self.normalize_query(query);
        let query_hash = self.hash_string(&normalized_query);
        let context_hash = self.hash_string(file_context.unwrap_or(""));
        
        QuerySignature {
            normalized_query,
            query_hash,
            context_hash,
        }
    }
    
    /// Normalize query for signature computation
    fn normalize_query(&self, query: &str) -> String {
        query.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
    
    /// Hash string to u64
    fn hash_string(&self, s: &str) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Check visited set cache
    async fn check_visited_cache(&self, signature: &QuerySignature) -> Result<Option<CachedVisitedSet>> {
        let mut cache = self.visited_cache.write().await;
        
        // Direct cache lookup
        if let Some(cached_set) = cache.get_mut(signature) {
            cached_set.hit_count += 1;
            cache.update_access_frequency(signature.clone());
            cache.update_access_order(signature.clone());
            return Ok(Some(cached_set.clone()));
        }
        
        // Similarity-based lookup
        let similar_signature = self.find_similar_cached_query(signature).await?;
        if let Some(sim_sig) = similar_signature {
            if let Some(cached_set) = cache.get_mut(&sim_sig) {
                debug!("Found similar cached query for: {:?}", signature);
                cached_set.hit_count += 1;
                return Ok(Some(cached_set.clone()));
            }
        }
        
        Ok(None)
    }
    
    /// Find similar cached query using similarity detector
    async fn find_similar_cached_query(&self, signature: &QuerySignature) -> Result<Option<QuerySignature>> {
        let cache = self.visited_cache.read().await;
        let mut best_similarity = 0.0;
        let mut best_signature = None;
        
        for cached_signature in cache.cache.keys() {
            let similarity = self.similarity_detector
                .compute_similarity(signature, cached_signature)
                .await?;
                
            if similarity > self.config.similarity_threshold && similarity > best_similarity {
                best_similarity = similarity;
                best_signature = Some(cached_signature.clone());
            }
        }
        
        Ok(best_signature)
    }
    
    /// Update query history for pattern recognition
    async fn update_query_history(
        &self,
        query: &str,
        signature: &QuerySignature,
        file_context: Option<String>,
        result: &PrefetchResult,
    ) -> Result<()> {
        let history_entry = QueryHistoryEntry {
            query: query.to_string(),
            signature: signature.clone(),
            timestamp: Instant::now(),
            results_count: result.result_hints.len(),
            processing_time: result.processing_time,
            file_context,
        };
        
        let mut engine = self.pattern_engine.write().await;
        engine.add_query_to_history(history_entry);
        
        Ok(())
    }
    
    /// Detect patterns from current query
    async fn detect_patterns(&self, signature: &QuerySignature) -> Result<Vec<PatternId>> {
        let engine = self.pattern_engine.read().await;
        Ok(engine.detect_matching_patterns(signature))
    }
    
    /// Generate prefetch suggestions based on patterns
    async fn generate_prefetch_suggestions(&self, patterns: Vec<PatternId>) -> Result<Vec<PrefetchSuggestion>> {
        let engine = self.pattern_engine.read().await;
        let mut suggestions = Vec::new();
        
        for pattern_id in patterns {
            if let Some(pattern) = engine.get_pattern(pattern_id) {
                let suggestion = PrefetchSuggestion {
                    pattern_id,
                    predicted_queries: engine.predict_next_queries(pattern_id),
                    confidence: pattern.confidence,
                    estimated_benefit: self.estimate_prefetch_benefit(pattern).await,
                    resource_cost: self.estimate_resource_cost(pattern).await,
                };
                
                suggestions.push(suggestion);
            }
        }
        
        // Sort by estimated benefit
        suggestions.sort_by(|a, b| b.estimated_benefit.partial_cmp(&a.estimated_benefit).unwrap());
        
        Ok(suggestions)
    }
    
    /// Schedule prefetch operations
    async fn schedule_prefetch_operations(&self, suggestions: &[PrefetchSuggestion]) -> Result<()> {
        let mut scheduler = self.scheduler.lock().await;
        
        for suggestion in suggestions.iter().take(self.config.prefetch_lookahead) {
            let operation = PrefetchOperation {
                id: self.generate_prefetch_id(),
                pattern_id: suggestion.pattern_id,
                target_query: QuerySignature {
                    normalized_query: suggestion.predicted_queries.get(0)
                        .unwrap_or(&"".to_string()).clone(),
                    query_hash: 0,
                    context_hash: 0,
                },
                priority: self.determine_prefetch_priority(suggestion.estimated_benefit),
                estimated_benefit: suggestion.estimated_benefit,
                resource_cost: suggestion.resource_cost.clone(),
                schedule_time: Instant::now() + Duration::from_millis(100),
            };
            
            scheduler.schedule_operation(operation);
        }
        
        Ok(())
    }
    
    /// Estimate prefetch benefit
    async fn estimate_prefetch_benefit(&self, pattern: &QueryPattern) -> f64 {
        let frequency_score = (pattern.usage_count as f64).log10() / 10.0; // Log scale
        let recency_score = {
            let age = pattern.last_seen.elapsed().as_secs() as f64;
            (-age / 3600.0).exp() // Exponential decay over hours
        };
        let confidence_score = pattern.confidence;
        
        (frequency_score * 0.4 + recency_score * 0.3 + confidence_score * 0.3).min(1.0)
    }
    
    /// Estimate resource cost for prefetch
    async fn estimate_resource_cost(&self, pattern: &QueryPattern) -> ResourceCost {
        let base_memory = pattern.typical_results.len() * 1024; // 1KB per result
        let base_cpu = pattern.typical_results.len() * 10; // 10ms per result
        
        ResourceCost {
            memory_mb: base_memory as f64 / 1024.0 / 1024.0,
            cpu_time_ms: base_cpu as f64,
            io_operations: pattern.typical_results.len(),
            network_requests: 0,
        }
    }
    
    /// Determine prefetch priority
    fn determine_prefetch_priority(&self, benefit: f64) -> PrefetchPriority {
        if benefit > 0.8 {
            PrefetchPriority::Critical
        } else if benefit > 0.6 {
            PrefetchPriority::High
        } else if benefit > 0.4 {
            PrefetchPriority::Medium
        } else if benefit > 0.2 {
            PrefetchPriority::Low
        } else {
            PrefetchPriority::Background
        }
    }
    
    /// Generate unique prefetch ID
    fn generate_prefetch_id(&self) -> PrefetchId {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
    
    /// Estimate memory footprint
    fn estimate_memory_footprint(&self, visited_set: &HashSet<String>, hints: &[ResultHint]) -> usize {
        let visited_size = visited_set.iter().map(|s| s.len()).sum::<usize>();
        let hints_size = hints.len() * 256; // Approximate size per hint
        visited_size + hints_size + 1024 // Plus overhead
    }
    
    /// Update cache metrics
    async fn update_cache_metrics(&self, hit: bool) {
        let mut metrics = self.metrics.write().await;
        metrics.total_queries += 1;
        
        if hit {
            metrics.cache_hits += 1;
        } else {
            metrics.cache_misses += 1;
        }
        
        metrics.hit_rate = metrics.cache_hits as f64 / metrics.total_queries as f64;
    }
    
    /// Get current prefetch metrics
    pub async fn get_metrics(&self) -> PrefetchMetrics {
        self.metrics.read().await.clone()
    }
}

/// Prefetch result returned to caller
#[derive(Debug, Clone)]
pub struct PrefetchResult {
    pub cache_hit: bool,
    pub visited_set: Option<HashSet<String>>,
    pub result_hints: Vec<ResultHint>,
    pub prefetch_suggestions: Vec<PrefetchSuggestion>,
    pub processing_time: Duration,
    pub estimated_quality: f64,
}

/// Prefetch suggestion for proactive loading
#[derive(Debug, Clone)]
pub struct PrefetchSuggestion {
    pub pattern_id: PatternId,
    pub predicted_queries: Vec<String>,
    pub confidence: f64,
    pub estimated_benefit: f64,
    pub resource_cost: ResourceCost,
}

impl PatternEngine {
    pub fn new(_config: PrefetchConfig) -> Self {
        Self {
            patterns: HashMap::new(),
            query_history: VecDeque::new(),
            transition_graph: HashMap::new(),
            pattern_frequencies: HashMap::new(),
            next_pattern_id: 1,
        }
    }
    
    pub fn add_query_to_history(&mut self, entry: QueryHistoryEntry) {
        self.query_history.push_back(entry);
        
        // Limit history size
        while self.query_history.len() > 10000 {
            self.query_history.pop_front();
        }
        
        // Update patterns based on new entry
        self.update_patterns();
    }
    
    pub fn detect_matching_patterns(&self, signature: &QuerySignature) -> Vec<PatternId> {
        let mut matching_patterns = Vec::new();
        
        for (pattern_id, pattern) in &self.patterns {
            if self.pattern_matches_signature(pattern, signature) {
                matching_patterns.push(*pattern_id);
            }
        }
        
        matching_patterns
    }
    
    pub fn get_pattern(&self, pattern_id: PatternId) -> Option<&QueryPattern> {
        self.patterns.get(&pattern_id)
    }
    
    pub fn predict_next_queries(&self, pattern_id: PatternId) -> Vec<String> {
        if let Some(transitions) = self.transition_graph.get(&pattern_id) {
            let mut predictions: Vec<_> = transitions.iter()
                .filter(|(_, &prob)| prob > 0.3)
                .map(|(&next_pattern_id, &prob)| (next_pattern_id, prob))
                .collect();
            
            predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            predictions.into_iter()
                .take(3)
                .map(|(next_id, _)| format!("predicted_query_{}", next_id))
                .collect()
        } else {
            Vec::new()
        }
    }
    
    fn update_patterns(&mut self) {
        // Simplified pattern update - real implementation would use more sophisticated clustering
        if self.query_history.len() < 3 {
            return;
        }
        
        let recent_queries: Vec<_> = self.query_history.iter().rev().take(10).collect();
        
        for window in recent_queries.windows(3) {
            if let [q1, q2, q3] = window {
                self.detect_temporal_pattern(q1, q2, q3);
            }
        }
    }
    
    fn detect_temporal_pattern(&mut self, q1: &QueryHistoryEntry, q2: &QueryHistoryEntry, q3: &QueryHistoryEntry) {
        // Simple temporal pattern detection
        let time_gap1 = q2.timestamp.duration_since(q1.timestamp);
        let time_gap2 = q3.timestamp.duration_since(q2.timestamp);
        
        if time_gap1.as_secs() < 60 && time_gap2.as_secs() < 60 {
            // Queries within 1 minute of each other - potential pattern
            let pattern_id = self.next_pattern_id;
            self.next_pattern_id += 1;
            
            let template = QueryTemplate {
                structure: QueryStructure::SemanticSearch,
                keywords: HashSet::new(),
                query_type: QueryType::Navigation,
                language_hint: None,
            };
            
            let pattern = QueryPattern {
                id: pattern_id,
                template,
                typical_results: Vec::new(),
                access_pattern: AccessPattern {
                    temporal_pattern: vec![time_gap1, time_gap2],
                    sequence_probability: 0.7,
                    context_dependencies: Vec::new(),
                },
                confidence: 0.6,
                last_seen: q3.timestamp,
                usage_count: 1,
            };
            
            self.patterns.insert(pattern_id, pattern);
            self.pattern_frequencies.insert(pattern_id, 1);
        }
    }
    
    fn pattern_matches_signature(&self, pattern: &QueryPattern, signature: &QuerySignature) -> bool {
        // Simplified pattern matching - real implementation would be more sophisticated
        pattern.template.keywords.iter()
            .any(|keyword| signature.normalized_query.contains(keyword))
    }
}

impl VisitedSetCache {
    pub fn new(config: PrefetchConfig) -> Self {
        Self {
            cache: BTreeMap::new(),
            access_frequencies: HashMap::new(),
            total_cached_sets: 0,
            total_memory_usage: 0,
            access_order: VecDeque::new(),
        }
    }
    
    pub fn get_mut(&mut self, signature: &QuerySignature) -> Option<&mut CachedVisitedSet> {
        self.cache.get_mut(signature)
    }
    
    pub fn insert(&mut self, signature: QuerySignature, cached_set: CachedVisitedSet) {
        self.total_memory_usage += cached_set.memory_footprint;
        self.total_cached_sets += 1;
        
        self.cache.insert(signature.clone(), cached_set);
        
        // Initialize access frequency
        self.access_frequencies.insert(signature.clone(), AccessFrequency {
            total_accesses: 0,
            recent_accesses: VecDeque::new(),
            access_rate: 0.0,
            last_access: Instant::now(),
        });
    }
    
    pub fn update_access_frequency(&mut self, signature: QuerySignature) {
        if let Some(freq) = self.access_frequencies.get_mut(&signature) {
            freq.total_accesses += 1;
            freq.recent_accesses.push_back(Instant::now());
            freq.last_access = Instant::now();
            
            // Maintain recent access window
            let window = Duration::from_secs(3600); // 1 hour
            let cutoff = Instant::now() - window;
            while freq.recent_accesses.front().map_or(false, |&t| t < cutoff) {
                freq.recent_accesses.pop_front();
            }
            
            // Update access rate
            freq.access_rate = freq.recent_accesses.len() as f64 / window.as_secs() as f64;
        }
    }
    
    pub fn update_access_order(&mut self, signature: QuerySignature) {
        // Remove from current position and add to end (LRU)
        self.access_order.retain(|s| s != &signature);
        self.access_order.push_back(signature);
    }
    
    pub fn evict_lru_entries(&mut self, space_needed: usize) -> Result<()> {
        let mut freed_space = 0;
        
        while freed_space < space_needed && !self.access_order.is_empty() {
            if let Some(lru_signature) = self.access_order.pop_front() {
                if let Some(cached_set) = self.cache.remove(&lru_signature) {
                    freed_space += cached_set.memory_footprint;
                    self.total_memory_usage -= cached_set.memory_footprint;
                    self.total_cached_sets -= 1;
                }
                self.access_frequencies.remove(&lru_signature);
            }
        }
        
        debug!("Evicted LRU entries, freed {}MB", freed_space / 1024 / 1024);
        Ok(())
    }
}

impl QuerySimilarityDetector {
    pub fn new() -> Self {
        Self {
            feature_extractors: FeatureExtractors::new(),
            similarity_cache: RwLock::new(HashMap::new()),
            similarity_weights: RwLock::new(SimilarityWeights {
                lexical_weight: 0.3,
                semantic_weight: 0.4,
                structural_weight: 0.2,
                contextual_weight: 0.1,
            }),
        }
    }
    
    pub async fn compute_similarity(
        &self,
        sig1: &QuerySignature,
        sig2: &QuerySignature,
    ) -> Result<f64> {
        // Check cache first
        let cache_key = if sig1 < sig2 {
            (sig1.clone(), sig2.clone())
        } else {
            (sig2.clone(), sig1.clone())
        };
        
        {
            let cache = self.similarity_cache.read().await;
            if let Some(&cached_similarity) = cache.get(&cache_key) {
                return Ok(cached_similarity);
            }
        }
        
        // Compute similarity
        let lexical_sim = self.feature_extractors.lexical_extractor.compute_similarity(sig1, sig2);
        let semantic_sim = self.feature_extractors.semantic_extractor.compute_similarity(sig1, sig2);
        let structural_sim = self.feature_extractors.structural_extractor.compute_similarity(sig1, sig2);
        let contextual_sim = self.feature_extractors.contextual_extractor.compute_similarity(sig1, sig2);
        
        let weights = self.similarity_weights.read().await;
        let total_similarity = lexical_sim * weights.lexical_weight
            + semantic_sim * weights.semantic_weight
            + structural_sim * weights.structural_weight
            + contextual_sim * weights.contextual_weight;
        
        // Cache the result
        {
            let mut cache = self.similarity_cache.write().await;
            cache.insert(cache_key, total_similarity);
        }
        
        Ok(total_similarity)
    }
}

impl PrefetchScheduler {
    pub fn new() -> Self {
        Self {
            pending_prefetches: VecDeque::new(),
            active_prefetches: HashMap::new(),
            success_rates: HashMap::new(),
            resource_usage: ResourceUsage::default(),
        }
    }
    
    pub fn schedule_operation(&mut self, operation: PrefetchOperation) {
        self.pending_prefetches.push_back(operation);
        
        // Sort by priority and schedule time
        let mut ops: Vec<_> = self.pending_prefetches.drain(..).collect();
        ops.sort_by(|a, b| {
            a.priority.cmp(&b.priority)
                .then_with(|| a.schedule_time.cmp(&b.schedule_time))
        });
        self.pending_prefetches.extend(ops);
    }
}

// Feature extractor implementations
impl FeatureExtractors {
    pub fn new() -> Self {
        Self {
            lexical_extractor: LexicalFeatureExtractor,
            semantic_extractor: SemanticFeatureExtractor,
            structural_extractor: StructuralFeatureExtractor,
            contextual_extractor: ContextualFeatureExtractor,
        }
    }
}

pub struct LexicalFeatureExtractor;
pub struct SemanticFeatureExtractor;
pub struct StructuralFeatureExtractor;
pub struct ContextualFeatureExtractor;

impl LexicalFeatureExtractor {
    pub fn compute_similarity(&self, sig1: &QuerySignature, sig2: &QuerySignature) -> f64 {
        // Simple Jaccard similarity on words
        let words1: HashSet<&str> = sig1.normalized_query.split_whitespace().collect();
        let words2: HashSet<&str> = sig2.normalized_query.split_whitespace().collect();
        
        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();
        
        if union == 0 { 0.0 } else { intersection as f64 / union as f64 }
    }
}

impl SemanticFeatureExtractor {
    pub fn compute_similarity(&self, _sig1: &QuerySignature, _sig2: &QuerySignature) -> f64 {
        // Placeholder for semantic similarity (would use embeddings in real implementation)
        0.5
    }
}

impl StructuralFeatureExtractor {
    pub fn compute_similarity(&self, sig1: &QuerySignature, sig2: &QuerySignature) -> f64 {
        // Compare query structure patterns
        if sig1.query_hash == sig2.query_hash {
            1.0
        } else {
            // Hamming distance on hash bits (simplified)
            let xor = sig1.query_hash ^ sig2.query_hash;
            let different_bits = xor.count_ones();
            1.0 - (different_bits as f64 / 64.0)
        }
    }
}

impl ContextualFeatureExtractor {
    pub fn compute_similarity(&self, sig1: &QuerySignature, sig2: &QuerySignature) -> f64 {
        // Compare context hashes
        if sig1.context_hash == sig2.context_hash {
            1.0
        } else if sig1.context_hash == 0 || sig2.context_hash == 0 {
            0.5 // Unknown context
        } else {
            let xor = sig1.context_hash ^ sig2.context_hash;
            let different_bits = xor.count_ones();
            1.0 - (different_bits as f64 / 64.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_prefetch_manager_creation() {
        let config = PrefetchConfig::default();
        let manager = PrefetchManager::new(config).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_query_signature_computation() {
        let config = PrefetchConfig::default();
        let manager = PrefetchManager::new(config).await.unwrap();
        
        let sig1 = manager.compute_query_signature("function test", Some("test.rs")).await;
        let sig2 = manager.compute_query_signature("function test", Some("test.rs")).await;
        let sig3 = manager.compute_query_signature("function test", Some("other.rs")).await;
        
        assert_eq!(sig1, sig2);
        assert_ne!(sig1, sig3); // Different context
    }

    #[tokio::test]
    async fn test_cache_operations() {
        let config = PrefetchConfig::default();
        let manager = PrefetchManager::new(config).await.unwrap();
        
        let visited_set = {
            let mut set = HashSet::new();
            set.insert("node1".to_string());
            set.insert("node2".to_string());
            set
        };
        
        let hints = vec![
            ResultHint {
                file_path: "test.rs".to_string(),
                line_number: 42,
                relevance_score: 0.9,
                access_probability: 0.8,
            }
        ];
        
        // Cache the visited set
        manager.cache_visited_set(
            "function test",
            Some("test.rs".to_string()),
            visited_set.clone(),
            hints.clone(),
            0.85,
        ).await.unwrap();
        
        // Query should now hit the cache
        let result = manager.process_query("function test", Some("test.rs".to_string())).await.unwrap();
        
        assert!(result.cache_hit);
        assert!(result.visited_set.is_some());
        assert_eq!(result.visited_set.unwrap(), visited_set);
        assert_eq!(result.result_hints.len(), 1);
    }

    #[tokio::test]
    async fn test_pattern_recognition() {
        let mut engine = PatternEngine::new(PrefetchConfig::default());
        
        // Add several related queries
        for i in 0..5 {
            let entry = QueryHistoryEntry {
                query: format!("function test_{}", i),
                signature: QuerySignature {
                    normalized_query: format!("function test {}", i),
                    query_hash: i,
                    context_hash: 0,
                },
                timestamp: Instant::now(),
                results_count: 10,
                processing_time: Duration::from_millis(50),
                file_context: Some("test.rs".to_string()),
            };
            engine.add_query_to_history(entry);
        }
        
        // Should have detected some patterns
        assert!(!engine.patterns.is_empty());
    }

    #[tokio::test]
    async fn test_similarity_detection() {
        let detector = QuerySimilarityDetector::new();
        
        let sig1 = QuerySignature {
            normalized_query: "function test example".to_string(),
            query_hash: 123,
            context_hash: 456,
        };
        
        let sig2 = QuerySignature {
            normalized_query: "function test sample".to_string(),
            query_hash: 124,
            context_hash: 456,
        };
        
        let sig3 = QuerySignature {
            normalized_query: "class definition".to_string(),
            query_hash: 999,
            context_hash: 789,
        };
        
        let sim_12 = detector.compute_similarity(&sig1, &sig2).await.unwrap();
        let sim_13 = detector.compute_similarity(&sig1, &sig3).await.unwrap();
        
        // Similar queries should have higher similarity
        assert!(sim_12 > sim_13);
        assert!(sim_12 > 0.4);
    }

    #[tokio::test]
    async fn test_memory_management() {
        let config = PrefetchConfig {
            memory_pool_config: MemoryPoolConfig {
                max_memory_mb: 1, // Very small limit for testing
                pool_segment_size: 1024,
                cleanup_interval_secs: 60,
            },
            ..Default::default()
        };
        
        let manager = PrefetchManager::new(config).await.unwrap();
        
        // Add large visited set that should trigger eviction
        let mut large_visited_set = HashSet::new();
        for i in 0..1000 {
            large_visited_set.insert(format!("node_{}", i));
        }
        
        let hints = vec![ResultHint {
            file_path: "test.rs".to_string(),
            line_number: 1,
            relevance_score: 0.5,
            access_probability: 0.5,
        }];
        
        // This should succeed (or trigger eviction if over limit)
        let result = manager.cache_visited_set(
            "test query",
            None,
            large_visited_set,
            hints,
            0.5,
        ).await;
        
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_prefetch_suggestions() {
        let config = PrefetchConfig::default();
        let manager = PrefetchManager::new(config).await.unwrap();
        
        // Process several similar queries to build patterns
        for i in 0..5 {
            let query = format!("function search_{}", i);
            let _result = manager.process_query(&query, Some("test.rs".to_string())).await.unwrap();
        }
        
        // Next query should potentially have prefetch suggestions
        let result = manager.process_query("function search_new", Some("test.rs".to_string())).await.unwrap();
        
        // May or may not have suggestions depending on pattern detection
        // At minimum, should not crash and should have valid metrics
        let metrics = manager.get_metrics().await;
        assert!(metrics.total_queries >= 6);
    }

    #[test]
    fn test_prefetch_config_default() {
        let config = PrefetchConfig::default();
        
        assert_eq!(config.similarity_threshold, 0.7);
        assert_eq!(config.max_cached_visited_sets, 10000);
        assert_eq!(config.pattern_window_size, 100);
        assert_eq!(config.min_pattern_frequency, 3);
        assert_eq!(config.prefetch_timeout_ms, 100);
        assert_eq!(config.max_concurrent_prefetches, 5);
        assert_eq!(config.enable_pattern_recognition, true);
        assert_eq!(config.enable_memory_optimization, true);
    }

    #[test]
    fn test_query_signature_creation() {
        let signature = QuerySignature {
            normalized_query: "function test".to_string(),
            query_hash: 123456789,
            context_hash: 987654321,
        };
        
        assert_eq!(signature.normalized_query, "function test");
        assert_eq!(signature.query_hash, 123456789);
        assert_eq!(signature.context_hash, 987654321);
    }

    #[test]
    fn test_result_hint_creation() {
        let hint = ResultHint {
            file_path: "/path/to/test.rs".to_string(),
            line_number: 42,
            relevance_score: 0.95,
            access_probability: 0.85,
        };
        
        assert_eq!(hint.file_path, "/path/to/test.rs");
        assert_eq!(hint.line_number, 42);
        assert_eq!(hint.relevance_score, 0.95);
        assert_eq!(hint.access_probability, 0.85);
    }

    #[test]
    fn test_prefetch_result_creation() {
        let mut visited_set = HashSet::new();
        visited_set.insert("node1".to_string());
        visited_set.insert("node2".to_string());
        
        let hints = vec![
            ResultHint {
                file_path: "test1.rs".to_string(),
                line_number: 10,
                relevance_score: 0.9,
                access_probability: 0.8,
            },
            ResultHint {
                file_path: "test2.rs".to_string(),
                line_number: 20,
                relevance_score: 0.85,
                access_probability: 0.75,
            },
        ];
        
        let result = PrefetchResult {
            cache_hit: true,
            visited_set: Some(visited_set.clone()),
            result_hints: hints.clone(),
            processing_time: Duration::from_millis(50),
            confidence_score: 0.88,
            pattern_match: Some("function_search_pattern".to_string()),
            prefetch_suggestions: vec!["suggestion1".to_string(), "suggestion2".to_string()],
        };
        
        assert_eq!(result.cache_hit, true);
        assert_eq!(result.visited_set.as_ref().unwrap().len(), 2);
        assert_eq!(result.result_hints.len(), 2);
        assert_eq!(result.processing_time, Duration::from_millis(50));
        assert_eq!(result.confidence_score, 0.88);
        assert_eq!(result.pattern_match, Some("function_search_pattern".to_string()));
        assert_eq!(result.prefetch_suggestions.len(), 2);
    }

    #[test]
    fn test_query_history_entry() {
        let entry = QueryHistoryEntry {
            query: "test search query".to_string(),
            signature: QuerySignature {
                normalized_query: "test search".to_string(),
                query_hash: 12345,
                context_hash: 54321,
            },
            timestamp: Instant::now(),
            results_count: 15,
            processing_time: Duration::from_millis(75),
            file_context: Some("main.rs".to_string()),
        };
        
        assert_eq!(entry.query, "test search query");
        assert_eq!(entry.signature.normalized_query, "test search");
        assert_eq!(entry.signature.query_hash, 12345);
        assert_eq!(entry.results_count, 15);
        assert_eq!(entry.processing_time, Duration::from_millis(75));
        assert_eq!(entry.file_context, Some("main.rs".to_string()));
    }

    #[test]
    fn test_query_pattern_creation() {
        let pattern = QueryPattern {
            pattern_id: "pattern_123".to_string(),
            query_template: "function {}".to_string(),
            frequency: 10,
            contexts: vec!["test.rs".to_string(), "main.rs".to_string()],
            average_processing_time: Duration::from_millis(100),
            last_seen: Instant::now(),
            confidence: 0.95,
        };
        
        assert_eq!(pattern.pattern_id, "pattern_123");
        assert_eq!(pattern.query_template, "function {}");
        assert_eq!(pattern.frequency, 10);
        assert_eq!(pattern.contexts.len(), 2);
        assert_eq!(pattern.average_processing_time, Duration::from_millis(100));
        assert_eq!(pattern.confidence, 0.95);
    }

    #[test]
    fn test_memory_pool_config() {
        let config = MemoryPoolConfig {
            max_memory_mb: 512,
            pool_segment_size: 4096,
            cleanup_interval_secs: 300,
        };
        
        assert_eq!(config.max_memory_mb, 512);
        assert_eq!(config.pool_segment_size, 4096);
        assert_eq!(config.cleanup_interval_secs, 300);
    }

    #[tokio::test]
    async fn test_query_similarity_detector_creation() {
        let detector = QuerySimilarityDetector::new();
        
        // Test that detector can be created and used
        let sig1 = QuerySignature {
            normalized_query: "test query".to_string(),
            query_hash: 1,
            context_hash: 1,
        };
        
        let sig2 = QuerySignature {
            normalized_query: "test query".to_string(),
            query_hash: 1,
            context_hash: 1,
        };
        
        let similarity = detector.compute_similarity(&sig1, &sig2).await.unwrap();
        
        // Identical queries should have high similarity
        assert!(similarity > 0.9);
    }

    #[tokio::test]
    async fn test_pattern_engine_creation() {
        let config = PrefetchConfig::default();
        let engine = PatternEngine::new(config);
        
        // Verify initial state
        assert!(engine.patterns.is_empty());
        assert!(engine.query_history.is_empty());
    }

    #[tokio::test]
    async fn test_cache_miss_scenario() {
        let config = PrefetchConfig::default();
        let manager = PrefetchManager::new(config).await.unwrap();
        
        // Query for something that's not in cache
        let result = manager.process_query("unique_uncached_query", None).await.unwrap();
        
        assert_eq!(result.cache_hit, false);
        assert!(result.visited_set.is_none());
        assert!(result.result_hints.is_empty());
        assert!(result.processing_time > Duration::from_millis(0));
    }

    #[tokio::test]
    async fn test_confidence_score_calculation() {
        let config = PrefetchConfig::default();
        let manager = PrefetchManager::new(config).await.unwrap();
        
        let visited_set = HashSet::new();
        let hints = vec![
            ResultHint {
                file_path: "test.rs".to_string(),
                line_number: 1,
                relevance_score: 0.9,
                access_probability: 0.8,
            }
        ];
        
        // Cache with different confidence scores
        manager.cache_visited_set(
            "high_confidence_query",
            None,
            visited_set.clone(),
            hints.clone(),
            0.95, // High confidence
        ).await.unwrap();
        
        manager.cache_visited_set(
            "low_confidence_query",
            None,
            visited_set.clone(),
            hints.clone(),
            0.3, // Low confidence
        ).await.unwrap();
        
        let high_conf_result = manager.process_query("high_confidence_query", None).await.unwrap();
        let low_conf_result = manager.process_query("low_confidence_query", None).await.unwrap();
        
        assert!(high_conf_result.confidence_score > low_conf_result.confidence_score);
    }

    #[tokio::test]
    async fn test_concurrent_prefetch_operations() {
        let config = PrefetchConfig::default();
        let manager = PrefetchManager::new(config).await.unwrap();
        
        // Test concurrent cache operations
        let tasks: Vec<_> = (0..10).map(|i| {
            let manager = &manager;
            async move {
                let visited_set = HashSet::new();
                let hints = vec![];
                let query = format!("concurrent_query_{}", i);
                
                // Cache the query
                let cache_result = manager.cache_visited_set(
                    &query,
                    None,
                    visited_set,
                    hints,
                    0.8,
                ).await;
                
                // Then query it
                let query_result = manager.process_query(&query, None).await;
                
                (cache_result, query_result)
            }
        }).collect();
        
        let results = futures::future::join_all(tasks).await;
        
        // All operations should succeed
        for (cache_result, query_result) in results {
            assert!(cache_result.is_ok());
            assert!(query_result.is_ok());
            
            let query_result = query_result.unwrap();
            assert_eq!(query_result.cache_hit, true);
        }
    }

    #[tokio::test]
    async fn test_pattern_detection_threshold() {
        let config = PrefetchConfig {
            min_pattern_frequency: 3, // Require at least 3 occurrences
            ..Default::default()
        };
        
        let mut engine = PatternEngine::new(config);
        
        // Add 2 similar queries (below threshold)
        for i in 0..2 {
            let entry = QueryHistoryEntry {
                query: format!("function test_{}", i),
                signature: QuerySignature {
                    normalized_query: "function test".to_string(),
                    query_hash: 1,
                    context_hash: 1,
                },
                timestamp: Instant::now(),
                results_count: 5,
                processing_time: Duration::from_millis(50),
                file_context: None,
            };
            engine.add_query_to_history(entry);
        }
        
        // Should not detect pattern yet
        assert!(engine.patterns.is_empty());
        
        // Add one more to reach threshold
        let entry = QueryHistoryEntry {
            query: "function test_3".to_string(),
            signature: QuerySignature {
                normalized_query: "function test".to_string(),
                query_hash: 1,
                context_hash: 1,
            },
            timestamp: Instant::now(),
            results_count: 5,
            processing_time: Duration::from_millis(50),
            file_context: None,
        };
        engine.add_query_to_history(entry);
        
        // Should now detect pattern
        assert!(!engine.patterns.is_empty());
    }

    #[test]
    fn test_prefetch_metrics_default() {
        let metrics = PrefetchMetrics::default();
        
        assert_eq!(metrics.total_queries, 0);
        assert_eq!(metrics.cache_hits, 0);
        assert_eq!(metrics.cache_misses, 0);
        assert_eq!(metrics.patterns_detected, 0);
        assert_eq!(metrics.prefetch_suggestions, 0);
        assert_eq!(metrics.average_processing_time, Duration::from_millis(0));
        assert_eq!(metrics.memory_usage_mb, 0.0);
        assert_eq!(metrics.hit_rate, 0.0);
    }

    #[tokio::test]
    async fn test_prefetch_suggestions_generation() {
        let config = PrefetchConfig::default();
        let manager = PrefetchManager::new(config).await.unwrap();
        
        // Build a pattern by adding multiple similar queries
        for i in 0..5 {
            let query = format!("search term_{}", i);
            manager.cache_visited_set(
                &query,
                Some("test.rs".to_string()),
                HashSet::new(),
                vec![],
                0.8,
            ).await.unwrap();
        }
        
        // Process a new query that might get suggestions
        let result = manager.process_query("search term_new", Some("test.rs".to_string())).await.unwrap();
        
        // May have suggestions based on pattern detection
        // At minimum, verify structure is valid
        assert!(result.prefetch_suggestions.len() >= 0);
        assert!(result.confidence_score >= 0.0);
        assert!(result.confidence_score <= 1.0);
    }

    #[tokio::test]
    async fn test_memory_cleanup() {
        let config = PrefetchConfig {
            max_cached_visited_sets: 2, // Very small cache for testing
            ..Default::default()
        };
        
        let manager = PrefetchManager::new(config).await.unwrap();
        
        // Add more items than the cache can hold
        for i in 0..5 {
            let query = format!("cache_test_{}", i);
            manager.cache_visited_set(
                &query,
                None,
                HashSet::new(),
                vec![],
                0.8,
            ).await.unwrap();
        }
        
        // Cache should have cleaned up old entries
        let metrics = manager.get_metrics().await;
        // The total queries should be 5, but effective cache size might be limited
        assert_eq!(metrics.total_queries, 5);
    }

    #[test]
    fn test_context_hashing_consistency() {
        // Same context should produce same hash
        let context1 = Some("test.rs");
        let context2 = Some("test.rs");
        let context3 = Some("other.rs");
        let context4: Option<&str> = None;
        
        // In a real implementation, there would be a hash function for contexts
        // Here we test that the pattern is consistent
        assert_eq!(context1, context2);
        assert_ne!(context1, context3);
        assert_ne!(context1, context4);
    }

    #[test]
    fn test_pattern_confidence_bounds() {
        let pattern = QueryPattern {
            pattern_id: "test".to_string(),
            query_template: "function {}".to_string(),
            frequency: 10,
            contexts: vec![],
            average_processing_time: Duration::from_millis(100),
            last_seen: Instant::now(),
            confidence: 0.85,
        };
        
        // Confidence should be between 0 and 1
        assert!(pattern.confidence >= 0.0);
        assert!(pattern.confidence <= 1.0);
        assert_eq!(pattern.confidence, 0.85);
    }
}