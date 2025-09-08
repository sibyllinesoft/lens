//! # Cross-Encoder for Precision Boost
//!
//! Optional cross-encoder for highest-precision queries with strict budget constraints:
//! - Query-specific activation based on complexity  
//! - Tight budget constraints (≤50ms p95 inference)
//! - Target: +1-2pp additional improvement on complex NL queries
//! - Smart resource allocation and query routing

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Cross-encoder for precision boost on high-value queries
pub struct CrossEncoder {
    config: CrossEncoderConfig,
    /// Model instance (mock for development)
    model: Arc<RwLock<Option<CrossEncoderModel>>>,
    /// Query complexity analyzer
    complexity_analyzer: QueryComplexityAnalyzer,
    /// Performance tracker
    performance_tracker: Arc<RwLock<PerformanceTracker>>,
    /// Budget manager for resource allocation
    budget_manager: Arc<RwLock<BudgetManager>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossEncoderConfig {
    /// Enable cross-encoder
    pub enabled: bool,
    /// Maximum inference time budget (≤50ms p95)
    pub max_inference_ms: u64,
    /// Query complexity threshold for activation (0.0-1.0)
    pub complexity_threshold: f32,
    /// Top-K candidates to cross-encode
    pub top_k: usize,
    /// Model architecture
    pub model_type: String,
    /// Maximum batch size for efficiency
    pub max_batch_size: usize,
    /// Budget allocation strategy
    pub budget_strategy: BudgetStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BudgetStrategy {
    /// Fixed time budget per query
    FixedPerQuery,
    /// Dynamic budget based on query importance
    DynamicImportance,
    /// Adaptive budget based on performance history
    AdaptiveHistorical,
}

/// Mock cross-encoder model
#[derive(Debug)]
pub struct CrossEncoderModel {
    model_type: String,
    initialized: bool,
    inference_time_ms: u64,
}

/// Query complexity analysis for activation decisions
#[derive(Debug)]
pub struct QueryComplexityAnalyzer {
    /// NL vs code query classifier
    nl_classifier: NLClassifier,
    /// Complexity metrics cache
    complexity_cache: HashMap<String, f32>,
}

/// Natural language query classifier
#[derive(Debug, Default)]
pub struct NLClassifier {
    /// Keywords indicating natural language queries
    nl_indicators: Vec<String>,
    /// Code-specific patterns
    code_patterns: Vec<String>,
}

/// Performance tracking for cross-encoder
#[derive(Debug, Default)]
pub struct PerformanceTracker {
    /// Inference latency percentiles
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    /// Query activation statistics
    pub queries_activated: u64,
    pub queries_skipped: u64,
    /// Accuracy improvements
    pub precision_improvement: f32,
    pub ndcg_improvement: f32,
    /// Budget utilization
    pub budget_utilization: f32,
    /// Recent latency samples
    latency_samples: Vec<u64>,
}

/// Budget manager for resource allocation
#[derive(Debug)]
pub struct BudgetManager {
    /// Current budget allocation
    pub current_budget_ms: u64,
    /// Budget per time window
    pub budget_per_window_ms: u64,
    /// Time window for budget reset
    pub window_duration: Duration,
    /// Last budget reset time
    pub last_reset: Instant,
    /// Query priority scores
    pub query_priorities: HashMap<String, f32>,
}

/// Query analysis result
#[derive(Debug, Clone)]
pub struct QueryAnalysis {
    pub query: String,
    pub complexity_score: f32,
    pub is_natural_language: bool,
    pub should_activate_cross_encoder: bool,
    pub priority_score: f32,
    pub estimated_benefit: f32,
}

/// Cross-encoder input pair
#[derive(Debug, Clone)]
pub struct CrossEncoderPair {
    pub query: String,
    pub candidate: String,
    pub initial_score: f32,
    pub metadata: HashMap<String, String>,
}

/// Cross-encoder result with confidence
#[derive(Debug, Clone)]
pub struct CrossEncoderResult {
    pub query: String,
    pub candidate: String,
    pub relevance_score: f32,
    pub confidence: f32,
    pub inference_time_ms: u64,
    pub model_version: String,
}

impl CrossEncoder {
    /// Create new cross-encoder
    pub async fn new(config: CrossEncoderConfig) -> Result<Self> {
        info!("Creating cross-encoder with config: {:?}", config);
        
        if !config.enabled {
            info!("Cross-encoder is disabled");
        }
        
        let complexity_analyzer = QueryComplexityAnalyzer::new();
        let performance_tracker = Arc::new(RwLock::new(PerformanceTracker::default()));
        let budget_manager = Arc::new(RwLock::new(BudgetManager::new(config.max_inference_ms)));
        
        Ok(Self {
            config,
            model: Arc::new(RwLock::new(None)),
            complexity_analyzer,
            performance_tracker,
            budget_manager,
        })
    }
    
    /// Initialize cross-encoder model
    pub async fn initialize(&self) -> Result<()> {
        if !self.config.enabled {
            info!("Cross-encoder disabled, skipping initialization");
            return Ok(());
        }
        
        info!("Initializing cross-encoder model: {}", self.config.model_type);
        
        // Create mock model
        let model = CrossEncoderModel {
            model_type: self.config.model_type.clone(),
            initialized: true,
            inference_time_ms: 25, // Mock inference time
        };
        
        *self.model.write().await = Some(model);
        
        info!("Cross-encoder initialized successfully");
        Ok(())
    }
    
    /// Analyze query to determine if cross-encoder should be activated
    pub async fn analyze_query(&self, query: &str) -> Result<QueryAnalysis> {
        // Calculate query complexity
        let complexity_score = self.complexity_analyzer.calculate_complexity(query).await?;
        
        // Classify as natural language vs code
        let is_natural_language = self.complexity_analyzer.is_natural_language(query);
        
        // Calculate priority score
        let priority_score = self.calculate_priority_score(query, complexity_score, is_natural_language).await?;
        
        // Check budget availability
        let budget_available = self.check_budget_availability(priority_score).await?;
        
        // Decision logic for activation
        let should_activate = self.config.enabled &&
                             complexity_score >= self.config.complexity_threshold &&
                             is_natural_language &&
                             budget_available;
        
        // Estimate potential benefit
        let estimated_benefit = if should_activate {
            self.estimate_benefit(complexity_score, is_natural_language)
        } else {
            0.0
        };
        
        let analysis = QueryAnalysis {
            query: query.to_string(),
            complexity_score,
            is_natural_language,
            should_activate_cross_encoder: should_activate,
            priority_score,
            estimated_benefit,
        };
        
        debug!("Query analysis: complexity={:.3}, NL={}, activate={}, benefit={:.3}",
               complexity_score, is_natural_language, should_activate, estimated_benefit);
        
        Ok(analysis)
    }
    
    /// Apply cross-encoder to re-score top candidates
    pub async fn cross_encode(&self, pairs: Vec<CrossEncoderPair>) -> Result<Vec<CrossEncoderResult>> {
        if !self.config.enabled || pairs.is_empty() {
            return Ok(Vec::new());
        }
        
        let start_time = Instant::now();
        
        // Check budget before proceeding
        let query = &pairs[0].query;
        let priority = self.calculate_query_priority(query).await?;
        
        if !self.allocate_budget(priority).await? {
            debug!("Budget exhausted, skipping cross-encoder for query");
            self.update_skip_statistics().await;
            return Ok(Vec::new());
        }
        
        // Take top-K candidates
        let candidates_to_process = pairs.into_iter()
            .take(self.config.top_k)
            .collect::<Vec<_>>();
        
        // Batch process for efficiency
        let results = self.batch_cross_encode(candidates_to_process).await
            .context("Cross-encoder inference failed")?;
        
        let inference_time = start_time.elapsed().as_millis() as u64;
        
        // Update performance tracking
        self.update_performance_metrics(inference_time, &results).await;
        
        // Check if we exceeded budget
        if inference_time > self.config.max_inference_ms {
            warn!("Cross-encoder exceeded budget: {}ms > {}ms", 
                  inference_time, self.config.max_inference_ms);
        }
        
        debug!("Cross-encoded {} pairs in {}ms", results.len(), inference_time);
        
        Ok(results)
    }
    
    /// Get cross-encoder performance metrics
    pub async fn get_metrics(&self) -> CrossEncoderMetrics {
        let tracker = self.performance_tracker.read().await;
        let budget = self.budget_manager.read().await;
        
        CrossEncoderMetrics {
            enabled: self.config.enabled,
            queries_activated: tracker.queries_activated,
            queries_skipped: tracker.queries_skipped,
            activation_rate: if tracker.queries_activated + tracker.queries_skipped > 0 {
                tracker.queries_activated as f32 / (tracker.queries_activated + tracker.queries_skipped) as f32
            } else {
                0.0
            },
            latency_p50_ms: tracker.latency_p50_ms,
            latency_p95_ms: tracker.latency_p95_ms,
            latency_p99_ms: tracker.latency_p99_ms,
            budget_utilization: budget.budget_utilization(),
            precision_improvement: tracker.precision_improvement,
            ndcg_improvement: tracker.ndcg_improvement,
            meets_latency_target: tracker.latency_p95_ms <= self.config.max_inference_ms as f64,
        }
    }
    
    /// Update cross-encoder with performance feedback
    pub async fn update_performance_feedback(&self, query: &str, actual_improvement: f32) -> Result<()> {
        // Update performance estimates based on actual results
        let mut tracker = self.performance_tracker.write().await;
        
        // Simple moving average update
        let alpha = 0.1; // Learning rate
        if actual_improvement > 0.0 {
            tracker.precision_improvement = tracker.precision_improvement * (1.0 - alpha) + actual_improvement * alpha;
        }
        
        // Update query priority based on performance
        let mut budget = self.budget_manager.write().await;
        let current_priority = budget.query_priorities.get(query).cloned().unwrap_or(0.5);
        let new_priority = (current_priority + actual_improvement * 0.5).clamp(0.0, 1.0);
        budget.query_priorities.insert(query.to_string(), new_priority);
        
        debug!("Updated performance feedback for query '{}': improvement={:.3}, new_priority={:.3}",
               query, actual_improvement, new_priority);
        
        Ok(())
    }
    
    // Private implementation methods
    
    async fn calculate_priority_score(&self, query: &str, complexity: f32, is_nl: bool) -> Result<f32> {
        let mut score = complexity;
        
        // Bonus for natural language queries
        if is_nl {
            score += 0.2;
        }
        
        // Historical performance bonus
        let budget = self.budget_manager.read().await;
        if let Some(historical_priority) = budget.query_priorities.get(query) {
            score = (score + historical_priority) / 2.0;
        }
        
        Ok(score.clamp(0.0, 1.0))
    }
    
    async fn check_budget_availability(&self, priority: f32) -> Result<bool> {
        let budget = self.budget_manager.read().await;
        
        // Check if we have budget remaining
        let has_budget = budget.current_budget_ms > 0;
        
        // Check if priority is high enough for remaining budget
        let priority_threshold = match budget.current_budget_ms {
            0..=10 => 0.9,      // Very selective when budget is low
            11..=25 => 0.7,     // Moderate selectivity
            _ => priority,       // Use query priority when budget is available
        };
        
        Ok(has_budget && priority >= priority_threshold)
    }
    
    async fn allocate_budget(&self, priority: f32) -> Result<bool> {
        let mut budget = self.budget_manager.write().await;
        
        // Reset budget if window has elapsed
        if budget.last_reset.elapsed() >= budget.window_duration {
            budget.current_budget_ms = budget.budget_per_window_ms;
            budget.last_reset = Instant::now();
            debug!("Budget reset: {}ms available", budget.current_budget_ms);
        }
        
        // Estimate cost for this query
        let estimated_cost = match self.config.budget_strategy {
            BudgetStrategy::FixedPerQuery => 30, // Fixed 30ms estimate
            BudgetStrategy::DynamicImportance => (20.0 + priority * 30.0) as u64,
            BudgetStrategy::AdaptiveHistorical => {
                let tracker = self.performance_tracker.read().await;
                tracker.latency_p95_ms as u64
            }
        };
        
        if budget.current_budget_ms >= estimated_cost {
            budget.current_budget_ms -= estimated_cost;
            // No need to set budget_utilization here as it's computed by the budget_utilization() method
            
            debug!("Budget allocated: {}ms, remaining: {}ms", estimated_cost, budget.current_budget_ms);
            Ok(true)
        } else {
            debug!("Budget allocation failed: need {}ms, have {}ms", estimated_cost, budget.current_budget_ms);
            Ok(false)
        }
    }
    
    async fn batch_cross_encode(&self, pairs: Vec<CrossEncoderPair>) -> Result<Vec<CrossEncoderResult>> {
        let model_guard = self.model.read().await;
        let model = model_guard.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Cross-encoder model not initialized"))?;
        
        let mut results = Vec::with_capacity(pairs.len());
        
        // Process in batches for efficiency
        for chunk in pairs.chunks(self.config.max_batch_size) {
            let batch_start = Instant::now();
            
            for pair in chunk {
                // Mock cross-encoder inference
                let result = self.mock_cross_encode(pair, model).await?;
                results.push(result);
            }
            
            let batch_time = batch_start.elapsed().as_millis() as u64;
            debug!("Processed batch of {} pairs in {}ms", chunk.len(), batch_time);
            
            // Check if we're approaching budget limits
            if batch_time > self.config.max_inference_ms / 2 {
                warn!("Batch processing time {}ms approaching budget limit", batch_time);
            }
        }
        
        Ok(results)
    }
    
    async fn mock_cross_encode(&self, pair: &CrossEncoderPair, model: &CrossEncoderModel) -> Result<CrossEncoderResult> {
        // Mock cross-encoder inference
        // In real implementation, this would run the actual model
        
        // Simulate processing time
        tokio::time::sleep(Duration::from_millis(model.inference_time_ms / 4)).await;
        
        // Mock relevance scoring based on simple heuristics
        let query_lower = pair.query.to_lowercase();
        let candidate_lower = pair.candidate.to_lowercase();
        
        // Term overlap scoring
        let query_terms: Vec<&str> = query_lower.split_whitespace().collect();
        let matches = query_terms.iter()
            .filter(|term| candidate_lower.contains(*term))
            .count();
        
        let term_overlap = matches as f32 / query_terms.len().max(1) as f32;
        
        // Combine with initial score
        let relevance_score = (pair.initial_score * 0.6 + term_overlap * 0.4).clamp(0.0, 1.0);
        
        // Mock confidence based on score certainty
        let confidence = if relevance_score > 0.8 || relevance_score < 0.2 {
            0.9 // High confidence for extreme scores
        } else {
            0.6 // Lower confidence for middle scores
        };
        
        Ok(CrossEncoderResult {
            query: pair.query.clone(),
            candidate: pair.candidate.clone(),
            relevance_score,
            confidence,
            inference_time_ms: model.inference_time_ms,
            model_version: model.model_type.clone(),
        })
    }
    
    fn estimate_benefit(&self, complexity: f32, is_nl: bool) -> f32 {
        let mut benefit = complexity * 0.02; // Base 2% improvement per complexity unit
        
        if is_nl {
            benefit += 0.015; // Additional 1.5% for NL queries
        }
        
        benefit.clamp(0.0, 0.025) // Cap at 2.5% improvement estimate
    }
    
    async fn calculate_query_priority(&self, query: &str) -> Result<f32> {
        let budget = self.budget_manager.read().await;
        Ok(budget.query_priorities.get(query).cloned().unwrap_or(0.5))
    }
    
    async fn update_skip_statistics(&self) {
        let mut tracker = self.performance_tracker.write().await;
        tracker.queries_skipped += 1;
    }
    
    async fn update_performance_metrics(&self, inference_time: u64, _results: &[CrossEncoderResult]) {
        let mut tracker = self.performance_tracker.write().await;
        
        tracker.queries_activated += 1;
        tracker.latency_samples.push(inference_time);
        
        // Keep only recent samples
        if tracker.latency_samples.len() > 1000 {
            tracker.latency_samples.drain(0..500);
        }
        
        // Update percentiles
        if !tracker.latency_samples.is_empty() {
            let mut sorted_samples = tracker.latency_samples.clone();
            sorted_samples.sort_unstable();
            
            let len = sorted_samples.len();
            tracker.latency_p50_ms = sorted_samples[len / 2] as f64;
            tracker.latency_p95_ms = sorted_samples[(len * 95) / 100] as f64;
            tracker.latency_p99_ms = sorted_samples[(len * 99) / 100] as f64;
        }
    }
}

impl QueryComplexityAnalyzer {
    pub fn new() -> Self {
        let nl_indicators = vec![
            "find".to_string(),
            "show".to_string(),
            "get".to_string(),
            "how".to_string(),
            "what".to_string(),
            "where".to_string(),
            "functions".to_string(),
            "methods".to_string(),
            "classes".to_string(),
        ];
        
        let code_patterns = vec![
            "def ".to_string(),
            "function".to_string(),
            "class ".to_string(),
            "import ".to_string(),
            "const ".to_string(),
            "let ".to_string(),
            "fn ".to_string(),
        ];
        
        Self {
            nl_classifier: NLClassifier {
                nl_indicators,
                code_patterns,
            },
            complexity_cache: HashMap::new(),
        }
    }
    
    pub async fn calculate_complexity(&self, query: &str) -> Result<f32> {
        // Simple complexity heuristics
        let mut complexity = 0.0;
        
        // Length factor
        let length_factor = (query.len() as f32 / 20.0).min(1.0);
        complexity += length_factor * 0.3;
        
        // Word count factor
        let word_count = query.split_whitespace().count();
        let word_factor = (word_count as f32 / 5.0).min(1.0);
        complexity += word_factor * 0.3;
        
        // Natural language indicators
        if self.nl_classifier.is_natural_language_query(query) {
            complexity += 0.4;
        }
        
        Ok(complexity.clamp(0.0, 1.0))
    }
    
    pub fn is_natural_language(&self, query: &str) -> bool {
        self.nl_classifier.is_natural_language_query(query)
    }
}

impl NLClassifier {
    pub fn is_natural_language_query(&self, query: &str) -> bool {
        let query_lower = query.to_lowercase();
        
        // Check for NL indicators
        let has_nl_indicators = self.nl_indicators.iter()
            .any(|indicator| query_lower.contains(indicator));
        
        // Check for code patterns (negative indicator for NL)
        let has_code_patterns = self.code_patterns.iter()
            .any(|pattern| query_lower.contains(pattern));
        
        // Simple heuristic: NL if has indicators and no code patterns
        has_nl_indicators && !has_code_patterns
    }
}

impl BudgetManager {
    pub fn new(max_inference_ms: u64) -> Self {
        Self {
            current_budget_ms: max_inference_ms * 60, // 60x inference budget per minute
            budget_per_window_ms: max_inference_ms * 60,
            window_duration: Duration::from_secs(60), // 1 minute window
            last_reset: Instant::now(),
            query_priorities: HashMap::new(),
        }
    }
    
    pub fn budget_utilization(&self) -> f32 {
        1.0 - (self.current_budget_ms as f32 / self.budget_per_window_ms as f32)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossEncoderMetrics {
    pub enabled: bool,
    pub queries_activated: u64,
    pub queries_skipped: u64,
    pub activation_rate: f32,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub budget_utilization: f32,
    pub precision_improvement: f32,
    pub ndcg_improvement: f32,
    pub meets_latency_target: bool,
}

/// Initialize cross-encoder
pub async fn initialize_cross_encoder(config: &CrossEncoderConfig) -> Result<()> {
    info!("Initializing cross-encoder");
    info!("Enabled: {}, max inference: {}ms, complexity threshold: {}", 
          config.enabled, config.max_inference_ms, config.complexity_threshold);
    
    if config.enabled {
        // Validate performance constraints
        if config.max_inference_ms > 50 {
            warn!("Max inference time {}ms > 50ms target", config.max_inference_ms);
        }
        
        if config.complexity_threshold < 0.5 {
            warn!("Complexity threshold {} may activate too frequently", config.complexity_threshold);
        }
        
        info!("Cross-encoder will target +1-2pp improvement on complex NL queries");
    }
    
    info!("Cross-encoder initialization complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cross_encoder_creation() {
        let config = CrossEncoderConfig {
            enabled: true,
            max_inference_ms: 50,
            complexity_threshold: 0.7,
            top_k: 10,
            model_type: "cross-encoder-ms-marco-MiniLM-L-6-v2".to_string(),
            max_batch_size: 8,
            budget_strategy: BudgetStrategy::FixedPerQuery,
        };
        
        let encoder = CrossEncoder::new(config).await.unwrap();
        let metrics = encoder.get_metrics().await;
        assert_eq!(metrics.queries_activated, 0);
        assert!(metrics.enabled);
    }

    #[tokio::test]
    async fn test_query_analysis() {
        let config = CrossEncoderConfig {
            enabled: true,
            max_inference_ms: 50,
            complexity_threshold: 0.5,
            top_k: 10,
            model_type: "test".to_string(),
            max_batch_size: 8,
            budget_strategy: BudgetStrategy::FixedPerQuery,
        };
        
        let encoder = CrossEncoder::new(config).await.unwrap();
        
        let nl_query = "find all functions that handle user authentication";
        let analysis = encoder.analyze_query(nl_query).await.unwrap();
        
        assert!(analysis.is_natural_language);
        assert!(analysis.complexity_score > 0.0);
    }

    #[test]
    fn test_nl_classifier() {
        let classifier = NLClassifier::default();
        
        // Should classify as natural language
        assert!(classifier.is_natural_language_query("find all functions that process user data"));
        assert!(classifier.is_natural_language_query("show me methods for error handling"));
        
        // Should not classify as natural language
        assert!(!classifier.is_natural_language_query("def process_user_data():"));
        assert!(!classifier.is_natural_language_query("function authenticate(user)"));
    }

    #[tokio::test]
    async fn test_complexity_calculation() {
        let analyzer = QueryComplexityAnalyzer::new();
        
        let simple_query = "test";
        let complex_query = "find all functions that handle user authentication and error processing";
        
        let simple_complexity = analyzer.calculate_complexity(simple_query).await.unwrap();
        let complex_complexity = analyzer.calculate_complexity(complex_query).await.unwrap();
        
        assert!(complex_complexity > simple_complexity);
        assert!(simple_complexity >= 0.0 && simple_complexity <= 1.0);
        assert!(complex_complexity >= 0.0 && complex_complexity <= 1.0);
    }

    #[tokio::test]
    async fn test_budget_management() {
        let mut budget = BudgetManager::new(50);
        
        assert_eq!(budget.current_budget_ms, 3000); // 50 * 60
        assert!(budget.current_budget_ms > 0);
        
        // Test budget utilization calculation
        budget.current_budget_ms = 1500; // Half used
        assert!((budget.budget_utilization() - 0.5).abs() < 0.01);
    }
}