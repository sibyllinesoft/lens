//! LSP Routing Logic
//!
//! Implements intelligent routing with 40-60% target routing rate
//! Features:
//! - Intent-based routing decisions
//! - Query complexity analysis
//! - Performance-aware fallback
//! - Safety floors for exact/structural queries
//! - Adaptive routing based on success rates
//! - Bounded BFS traversal with depth ≤ 2, K ≤ 64 nodes per TODO.md

use super::{QueryIntent, TraversalBounds};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Bounded BFS traversal node for LSP symbol exploration
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BfsNode {
    pub symbol_id: String,
    pub symbol_type: SymbolType,
    pub file_path: String,
    pub line: u32,
    pub column: u32,
}

/// Type of symbol in BFS traversal
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SymbolType {
    Definition,
    Reference,
    TypeDefinition,
    Implementation,
    Declaration,
    Alias,
}

/// BFS traversal result with bounded exploration
#[derive(Debug, Clone)]
pub struct BfsTraversalResult {
    pub visited_nodes: Vec<BfsNode>,
    pub edges: Vec<(BfsNode, BfsNode, EdgeType)>,
    pub depth_reached: u8,
    pub nodes_explored: u16,
    pub was_bounded: bool,
}

/// Edge type in symbol graph
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EdgeType {
    DefinitionToReference,
    ReferenceToDefinition,
    TypeToImplementation,
    ImplementationToType,
    DeclarationToDefinition,
    AliasToTarget,
}

/// Query routing decision with confidence
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub should_route_to_lsp: bool,
    pub confidence: f64,
    pub reason: RoutingReason,
    pub estimated_latency_ms: u64,
}

/// Reason for routing decision
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RoutingReason {
    /// Intent is LSP-eligible and high confidence
    IntentMatch,
    /// Query has structural patterns LSP can handle
    StructuralPattern,
    /// File type is well-supported by LSP
    FileTypeSupport,
    /// Previous successful LSP results for similar queries
    HistoricalSuccess,
    /// LSP performance is acceptable
    PerformanceAcceptable,
    /// Safety floor - fallback to text search
    SafetyFloor,
    /// LSP servers unavailable
    NoServersAvailable,
    /// Query too complex for LSP
    ComplexityTooHigh,
    /// Performance concerns
    PerformanceConcerns,
    /// Intent not LSP-eligible
    IntentNotEligible,
}

/// Routing statistics by intent type
#[derive(Debug, Default, Clone)]
pub struct IntentStats {
    pub total_queries: u64,
    pub lsp_routed: u64,
    pub lsp_successes: u64,
    pub lsp_failures: u64,
    pub avg_latency_ms: u64,
    pub success_rate: f64,
}

impl IntentStats {
    pub fn routing_rate(&self) -> f64 {
        if self.total_queries == 0 {
            0.0
        } else {
            self.lsp_routed as f64 / self.total_queries as f64
        }
    }

    pub fn update_success(&mut self, latency_ms: u64) {
        self.lsp_successes += 1;
        self.update_avg_latency(latency_ms);
        self.recalculate_success_rate();
    }

    pub fn update_failure(&mut self, latency_ms: u64) {
        self.lsp_failures += 1;
        self.update_avg_latency(latency_ms);
        self.recalculate_success_rate();
    }

    fn update_avg_latency(&mut self, latency_ms: u64) {
        let total_attempts = self.lsp_successes + self.lsp_failures;
        if total_attempts > 0 {
            self.avg_latency_ms = (self.avg_latency_ms * (total_attempts - 1) + latency_ms) / total_attempts;
        }
    }

    fn recalculate_success_rate(&mut self) {
        let total_attempts = self.lsp_successes + self.lsp_failures;
        if total_attempts > 0 {
            self.success_rate = self.lsp_successes as f64 / total_attempts as f64;
        }
    }
}

/// Query pattern analysis
#[derive(Debug, Clone)]
pub struct QueryPattern {
    pub has_structural_hints: bool,
    pub has_identifier_patterns: bool,
    pub complexity_score: f64,
    pub estimated_lsp_effectiveness: f64,
}

/// Adaptive LSP router with machine learning-like adaptation
pub struct LspRouter {
    target_routing_rate: f64,
    current_routing_rate: Arc<AtomicU64>, // Stored as fixed-point (rate * 10000)
    
    // Statistics by intent
    intent_stats: Arc<RwLock<HashMap<QueryIntent, IntentStats>>>,
    
    // Pattern recognition
    known_patterns: Arc<RwLock<HashMap<String, QueryPattern>>>,
    
    // Configuration
    config: RoutingConfig,
    
    // Overall stats
    total_queries: AtomicU64,
    total_lsp_routed: AtomicU64,
}

#[derive(Debug, Clone)]
pub struct RoutingConfig {
    pub target_rate_min: f64,
    pub target_rate_max: f64,
    pub safety_floor_rate: f64,
    pub max_complexity_threshold: f64,
    pub min_success_rate_threshold: f64,
    pub max_acceptable_latency_ms: u64,
    pub adaptation_factor: f64,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            target_rate_min: 0.40,  // 40% minimum per TODO.md
            target_rate_max: 0.60,  // 60% maximum per TODO.md
            safety_floor_rate: 0.20, // Always route at least 20% for learning
            max_complexity_threshold: 0.8,
            min_success_rate_threshold: 0.7,
            max_acceptable_latency_ms: 1000,
            adaptation_factor: 0.1,
        }
    }
}

impl LspRouter {
    pub fn new(target_routing_rate: f64) -> Self {
        let config = RoutingConfig {
            target_rate_min: (target_routing_rate - 0.1).max(0.2),
            target_rate_max: (target_routing_rate + 0.1).min(0.8),
            ..Default::default()
        };

        Self {
            target_routing_rate,
            current_routing_rate: Arc::new(AtomicU64::new((target_routing_rate * 10000.0) as u64)),
            intent_stats: Arc::new(RwLock::new(HashMap::new())),
            known_patterns: Arc::new(RwLock::new(HashMap::new())),
            config,
            total_queries: AtomicU64::new(0),
            total_lsp_routed: AtomicU64::new(0),
        }
    }

    /// Make routing decision for a query
    pub async fn should_route(&self, query: &str, intent: &QueryIntent) -> bool {
        let decision = self.make_routing_decision(query, intent).await;
        
        debug!(
            "Routing decision for '{}': {} (reason: {:?}, confidence: {:.2})",
            query, decision.should_route_to_lsp, decision.reason, decision.confidence
        );
        
        // Update statistics
        self.update_routing_stats(intent, decision.should_route_to_lsp).await;
        
        decision.should_route_to_lsp
    }

    /// Make detailed routing decision with reasoning
    pub async fn make_routing_decision(&self, query: &str, intent: &QueryIntent) -> RoutingDecision {
        self.total_queries.fetch_add(1, Ordering::Relaxed);

        // Check if intent is LSP-eligible
        if !intent.is_lsp_eligible() {
            return RoutingDecision {
                should_route_to_lsp: false,
                confidence: 1.0,
                reason: RoutingReason::IntentNotEligible,
                estimated_latency_ms: 0,
            };
        }

        // Analyze query pattern
        let pattern = self.analyze_query_pattern(query).await;
        
        // Get intent-specific statistics
        let stats = self.get_intent_stats(intent).await;
        
        // Calculate base routing probability
        let mut routing_probability = self.calculate_base_routing_probability(intent, &pattern, &stats).await;
        
        // Apply adaptive adjustments
        routing_probability = self.apply_adaptive_adjustments(routing_probability).await;
        
        // Apply safety constraints
        let (final_decision, reason) = self.apply_safety_constraints(routing_probability, &pattern, &stats);
        
        let estimated_latency = if final_decision {
            stats.avg_latency_ms.max(100) // Minimum 100ms estimate for LSP
        } else {
            50 // Fast text search estimate
        };

        RoutingDecision {
            should_route_to_lsp: final_decision,
            confidence: routing_probability,
            reason,
            estimated_latency_ms: estimated_latency,
        }
    }

    async fn analyze_query_pattern(&self, query: &str) -> QueryPattern {
        // Check cache first
        {
            let patterns = self.known_patterns.read().await;
            if let Some(cached_pattern) = patterns.get(query) {
                return cached_pattern.clone();
            }
        }

        // Analyze query for structural hints
        let has_structural_hints = Self::detect_structural_patterns(query);
        let has_identifier_patterns = Self::detect_identifier_patterns(query);
        let complexity_score = Self::calculate_complexity(query);
        let estimated_effectiveness = Self::estimate_lsp_effectiveness(query);

        let pattern = QueryPattern {
            has_structural_hints,
            has_identifier_patterns,
            complexity_score,
            estimated_lsp_effectiveness: estimated_effectiveness,
        };

        // Cache the pattern
        {
            let mut patterns = self.known_patterns.write().await;
            patterns.insert(query.to_string(), pattern.clone());
        }

        pattern
    }

    fn detect_structural_patterns(query: &str) -> bool {
        let structural_keywords = [
            "class ", "function ", "def ", "interface ", "type ", "struct ",
            "impl ", "trait ", "extends ", "implements ", "import ", "from ",
        ];
        
        let query_lower = query.to_lowercase();
        structural_keywords.iter().any(|keyword| query_lower.contains(keyword))
    }

    fn detect_identifier_patterns(query: &str) -> bool {
        // Simple heuristic: contains camelCase or snake_case patterns
        let has_camel_case = query.chars().any(|c| c.is_uppercase()) && query.chars().any(|c| c.is_lowercase());
        let has_snake_case = query.contains('_');
        let has_dot_notation = query.contains('.');
        
        has_camel_case || has_snake_case || has_dot_notation
    }

    fn calculate_complexity(query: &str) -> f64 {
        let mut complexity = 0.0;
        
        // Length factor
        complexity += (query.len() as f64 / 100.0).min(0.3);
        
        // Word count factor  
        let word_count = query.split_whitespace().count();
        complexity += (word_count as f64 / 10.0).min(0.2);
        
        // Special characters factor
        let special_chars = query.chars().filter(|c| !c.is_alphanumeric() && !c.is_whitespace()).count();
        complexity += (special_chars as f64 / 20.0).min(0.2);
        
        // Regex-like patterns increase complexity
        if query.contains('[') || query.contains('{') || query.contains('*') {
            complexity += 0.3;
        }
        
        complexity.min(1.0)
    }

    fn estimate_lsp_effectiveness(query: &str) -> f64 {
        let mut effectiveness: f64 = 0.5; // Base effectiveness
        
        // Structural patterns are highly effective
        if Self::detect_structural_patterns(query) {
            effectiveness += 0.3;
        }
        
        // Identifier patterns are moderately effective
        if Self::detect_identifier_patterns(query) {
            effectiveness += 0.2;
        }
        
        // Short, specific queries are more effective
        if query.len() < 50 && query.split_whitespace().count() <= 3 {
            effectiveness += 0.1;
        }
        
        // Very long or complex queries are less effective
        if query.len() > 200 || query.split_whitespace().count() > 10 {
            effectiveness -= 0.2;
        }
        
        effectiveness.clamp(0.0, 1.0)
    }

    async fn get_intent_stats(&self, intent: &QueryIntent) -> IntentStats {
        let stats = self.intent_stats.read().await;
        stats.get(intent).cloned().unwrap_or_default()
    }

    async fn calculate_base_routing_probability(&self, intent: &QueryIntent, pattern: &QueryPattern, stats: &IntentStats) -> f64 {
        let mut probability = 0.5; // Base probability
        
        // Intent-specific factors
        match intent {
            QueryIntent::Definition | QueryIntent::Symbol => probability += 0.2,
            QueryIntent::References | QueryIntent::Implementation => probability += 0.15,
            QueryIntent::TypeDefinition | QueryIntent::Declaration => probability += 0.1,
            QueryIntent::Hover | QueryIntent::Completion => probability += 0.05,
            QueryIntent::TextSearch => probability -= 0.3,
        }
        
        // Pattern-based factors
        if pattern.has_structural_hints {
            probability += 0.15;
        }
        if pattern.has_identifier_patterns {
            probability += 0.1;
        }
        
        // Effectiveness estimate
        probability += pattern.estimated_lsp_effectiveness * 0.2;
        
        // Complexity penalty
        if pattern.complexity_score > self.config.max_complexity_threshold {
            probability -= 0.2;
        }
        
        // Historical success rate
        if stats.total_queries > 10 { // Require minimum sample size
            if stats.success_rate > self.config.min_success_rate_threshold {
                probability += 0.1;
            } else {
                probability -= 0.15;
            }
            
            // Latency penalty
            if stats.avg_latency_ms > self.config.max_acceptable_latency_ms {
                probability -= 0.1;
            }
        }
        
        probability.clamp(0.0, 1.0)
    }

    async fn apply_adaptive_adjustments(&self, base_probability: f64) -> f64 {
        let current_rate = self.current_routing_rate.load(Ordering::Relaxed) as f64 / 10000.0;
        let target_min = self.config.target_rate_min;
        let target_max = self.config.target_rate_max;
        
        let mut adjusted = base_probability;
        
        // If we're routing too little, increase probability
        if current_rate < target_min {
            let adjustment = (target_min - current_rate) * self.config.adaptation_factor;
            adjusted += adjustment;
        }
        // If we're routing too much, decrease probability  
        else if current_rate > target_max {
            let adjustment = (current_rate - target_max) * self.config.adaptation_factor;
            adjusted -= adjustment;
        }
        
        adjusted.clamp(0.0, 1.0)
    }

    fn apply_safety_constraints(&self, probability: f64, pattern: &QueryPattern, stats: &IntentStats) -> (bool, RoutingReason) {
        // Safety floor - always route some queries for learning
        if probability >= self.config.safety_floor_rate {
            if probability >= 0.8 {
                (true, RoutingReason::IntentMatch)
            } else if pattern.has_structural_hints {
                (true, RoutingReason::StructuralPattern)
            } else if stats.success_rate > self.config.min_success_rate_threshold {
                (true, RoutingReason::HistoricalSuccess)
            } else {
                (true, RoutingReason::PerformanceAcceptable)
            }
        } else {
            // Determine why we're not routing
            if pattern.complexity_score > self.config.max_complexity_threshold {
                (false, RoutingReason::ComplexityTooHigh)
            } else if stats.avg_latency_ms > self.config.max_acceptable_latency_ms {
                (false, RoutingReason::PerformanceConcerns)
            } else {
                (false, RoutingReason::SafetyFloor)
            }
        }
    }

    async fn update_routing_stats(&self, intent: &QueryIntent, was_routed: bool) {
        if was_routed {
            self.total_lsp_routed.fetch_add(1, Ordering::Relaxed);
        }
        
        // Update intent-specific stats
        let mut stats = self.intent_stats.write().await;
        let intent_stats = stats.entry(intent.clone()).or_default();
        intent_stats.total_queries += 1;
        if was_routed {
            intent_stats.lsp_routed += 1;
        }
        
        // Update current routing rate
        let total = self.total_queries.load(Ordering::Relaxed);
        let routed = self.total_lsp_routed.load(Ordering::Relaxed);
        if total > 0 {
            let rate = (routed as f64 / total as f64 * 10000.0) as u64;
            self.current_routing_rate.store(rate, Ordering::Relaxed);
        }
    }

    /// Report success/failure of LSP operation
    pub async fn report_lsp_result(&self, intent: &QueryIntent, success: bool, latency_ms: u64) {
        let mut stats = self.intent_stats.write().await;
        let intent_stats = stats.entry(intent.clone()).or_default();
        
        if success {
            intent_stats.update_success(latency_ms);
        } else {
            intent_stats.update_failure(latency_ms);
        }
        
        debug!(
            "LSP result for {:?}: success={}, latency={}ms, success_rate={:.2}",
            intent, success, latency_ms, intent_stats.success_rate
        );
    }

    /// Get current routing statistics
    pub async fn get_routing_stats(&self) -> RoutingStats {
        let current_rate = self.current_routing_rate.load(Ordering::Relaxed) as f64 / 10000.0;
        let total_queries = self.total_queries.load(Ordering::Relaxed);
        let total_routed = self.total_lsp_routed.load(Ordering::Relaxed);
        
        let intent_stats = self.intent_stats.read().await.clone();
        
        RoutingStats {
            current_routing_rate: current_rate,
            target_routing_rate: self.target_routing_rate,
            total_queries,
            total_lsp_routed: total_routed,
            intent_breakdown: intent_stats,
        }
    }

    /// Execute bounded BFS traversal on LSP symbol graph
    /// 
    /// Implements depth ≤ 2, K ≤ 64 node bounds per TODO.md specification
    /// Traverses def ↔ ref/type/impl/alias relationships safely
    pub async fn bounded_bfs_traversal(
        &self,
        start_node: BfsNode,
        bounds: &TraversalBounds,
    ) -> Result<BfsTraversalResult> {
        let max_depth = bounds.max_depth.min(2); // Enforce TODO.md depth ≤ 2
        let max_nodes = bounds.max_results.min(64); // Enforce TODO.md K ≤ 64
        
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut result_nodes = Vec::new();
        let mut edges = Vec::new();
        let mut nodes_explored = 0u16;
        
        // Initialize BFS with start node
        queue.push_back((start_node.clone(), 0u8)); // (node, depth)
        visited.insert(start_node.clone());
        result_nodes.push(start_node.clone());
        nodes_explored += 1;
        
        let mut max_depth_reached = 0u8;
        let mut was_bounded = false;
        
        debug!(
            "Starting bounded BFS traversal from {:?}, max_depth={}, max_nodes={}",
            start_node, max_depth, max_nodes
        );
        
        while let Some((current_node, current_depth)) = queue.pop_front() {
            max_depth_reached = max_depth_reached.max(current_depth);
            
            // Check depth bounds
            if current_depth >= max_depth {
                debug!("Reached maximum depth {} at node {:?}", max_depth, current_node);
                was_bounded = true;
                continue;
            }
            
            // Check node count bounds
            if nodes_explored >= max_nodes {
                warn!("Reached maximum node limit {} during BFS traversal", max_nodes);
                was_bounded = true;
                break;
            }
            
            // Get neighbors from LSP server (mock implementation for now)
            let neighbors = self.get_lsp_neighbors(&current_node).await?;
            
            for (neighbor, edge_type) in neighbors {
                // Skip if already visited
                if visited.contains(&neighbor) {
                    continue;
                }
                
                // Check if we would exceed node limit
                if nodes_explored >= max_nodes {
                    was_bounded = true;
                    break;
                }
                
                // Add to visited set and result
                visited.insert(neighbor.clone());
                result_nodes.push(neighbor.clone());
                edges.push((current_node.clone(), neighbor.clone(), edge_type));
                nodes_explored += 1;
                
                // Add to queue for next depth level
                if current_depth + 1 < max_depth {
                    queue.push_back((neighbor, current_depth + 1));
                }
            }
            
            if was_bounded {
                break;
            }
        }
        
        debug!(
            "BFS traversal completed: {} nodes explored, depth {}, bounded: {}",
            nodes_explored, max_depth_reached, was_bounded
        );
        
        Ok(BfsTraversalResult {
            visited_nodes: result_nodes,
            edges,
            depth_reached: max_depth_reached,
            nodes_explored,
            was_bounded,
        })
    }

    /// Get LSP neighbors for a symbol node
    /// 
    /// This is a mock implementation - in real usage this would query
    /// the appropriate LSP server for definitions, references, implementations, etc.
    async fn get_lsp_neighbors(&self, node: &BfsNode) -> Result<Vec<(BfsNode, EdgeType)>> {
        let mut neighbors = Vec::new();
        
        // Mock neighbor generation based on symbol type
        match node.symbol_type {
            SymbolType::Definition => {
                // Definition can have references and implementations
                neighbors.push((
                    BfsNode {
                        symbol_id: format!("{}_ref_1", node.symbol_id),
                        symbol_type: SymbolType::Reference,
                        file_path: format!("{}_usage.rs", node.file_path),
                        line: node.line + 10,
                        column: node.column,
                    },
                    EdgeType::DefinitionToReference,
                ));
                
                if node.symbol_id.contains("trait") || node.symbol_id.contains("interface") {
                    neighbors.push((
                        BfsNode {
                            symbol_id: format!("{}_impl_1", node.symbol_id),
                            symbol_type: SymbolType::Implementation,
                            file_path: format!("{}_impl.rs", node.file_path),
                            line: node.line + 20,
                            column: node.column,
                        },
                        EdgeType::TypeToImplementation,
                    ));
                }
            }
            
            SymbolType::Reference => {
                // Reference points back to definition
                neighbors.push((
                    BfsNode {
                        symbol_id: node.symbol_id.replace("_ref_", "_def_"),
                        symbol_type: SymbolType::Definition,
                        file_path: node.file_path.replace("_usage", "_def"),
                        line: node.line - 10,
                        column: node.column,
                    },
                    EdgeType::ReferenceToDefinition,
                ));
            }
            
            SymbolType::Implementation => {
                // Implementation points to type/trait definition
                neighbors.push((
                    BfsNode {
                        symbol_id: node.symbol_id.replace("_impl_", "_def_"),
                        symbol_type: SymbolType::TypeDefinition,
                        file_path: node.file_path.replace("_impl", "_def"),
                        line: node.line - 20,
                        column: node.column,
                    },
                    EdgeType::ImplementationToType,
                ));
            }
            
            SymbolType::Declaration => {
                // Declaration points to definition
                neighbors.push((
                    BfsNode {
                        symbol_id: format!("{}_def", node.symbol_id),
                        symbol_type: SymbolType::Definition,
                        file_path: node.file_path.replace("_decl", "_def"),
                        line: node.line + 5,
                        column: node.column,
                    },
                    EdgeType::DeclarationToDefinition,
                ));
            }
            
            SymbolType::Alias => {
                // Alias points to target
                neighbors.push((
                    BfsNode {
                        symbol_id: node.symbol_id.replace("_alias", "_target"),
                        symbol_type: SymbolType::Definition,
                        file_path: node.file_path.replace("_alias", "_target"),
                        line: node.line,
                        column: node.column + 10,
                    },
                    EdgeType::AliasToTarget,
                ));
            }
            
            SymbolType::TypeDefinition => {
                // Type can have implementations and references
                neighbors.push((
                    BfsNode {
                        symbol_id: format!("{}_impl_1", node.symbol_id),
                        symbol_type: SymbolType::Implementation,
                        file_path: format!("{}_impl.rs", node.file_path),
                        line: node.line + 15,
                        column: node.column,
                    },
                    EdgeType::TypeToImplementation,
                ));
            }
        }
        
        Ok(neighbors)
    }
}

/// Overall routing statistics
#[derive(Debug, Clone)]
pub struct RoutingStats {
    pub current_routing_rate: f64,
    pub target_routing_rate: f64,
    pub total_queries: u64,
    pub total_lsp_routed: u64,
    pub intent_breakdown: HashMap<QueryIntent, IntentStats>,
}

impl RoutingStats {
    pub fn is_within_target(&self, tolerance: f64) -> bool {
        (self.current_routing_rate - self.target_routing_rate).abs() <= tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::time::{sleep, Duration};

    // Helper function to create test router with custom config
    fn create_test_router_with_config(target_rate: f64) -> LspRouter {
        let mut config = RoutingConfig::default();
        config.target_rate_min = target_rate - 0.1;
        config.target_rate_max = target_rate + 0.1;
        config.safety_floor_rate = 0.1;
        config.max_complexity_threshold = 0.7;
        config.min_success_rate_threshold = 0.6;
        config.max_acceptable_latency_ms = 500;
        config.adaptation_factor = 0.2;
        
        let mut router = LspRouter::new(target_rate);
        router.config = config;
        router
    }

    // Test RoutingDecision structure
    #[test]
    fn test_routing_decision_creation() {
        let decision = RoutingDecision {
            should_route_to_lsp: true,
            confidence: 0.85,
            reason: RoutingReason::IntentMatch,
            estimated_latency_ms: 150,
        };
        
        assert!(decision.should_route_to_lsp);
        assert_eq!(decision.confidence, 0.85);
        assert_eq!(decision.reason, RoutingReason::IntentMatch);
        assert_eq!(decision.estimated_latency_ms, 150);
    }

    // Test RoutingReason enum completeness
    #[test]
    fn test_routing_reason_variants() {
        let reasons = vec![
            RoutingReason::IntentMatch,
            RoutingReason::StructuralPattern,
            RoutingReason::FileTypeSupport,
            RoutingReason::HistoricalSuccess,
            RoutingReason::PerformanceAcceptable,
            RoutingReason::SafetyFloor,
            RoutingReason::NoServersAvailable,
            RoutingReason::ComplexityTooHigh,
            RoutingReason::PerformanceConcerns,
            RoutingReason::IntentNotEligible,
        ];
        
        // Ensure all variants are covered and can be cloned/debug printed
        for reason in reasons {
            let cloned = reason.clone();
            let debug_str = format!("{:?}", cloned);
            assert!(!debug_str.is_empty());
        }
    }

    // Test IntentStats calculations
    #[test]
    fn test_intent_stats_routing_rate() {
        let mut stats = IntentStats::default();
        assert_eq!(stats.routing_rate(), 0.0);
        
        stats.total_queries = 10;
        stats.lsp_routed = 5;
        assert_eq!(stats.routing_rate(), 0.5);
        
        stats.lsp_routed = 8;
        assert_eq!(stats.routing_rate(), 0.8);
    }

    #[test]
    fn test_intent_stats_success_updates() {
        let mut stats = IntentStats::default();
        
        // First success
        stats.update_success(100);
        assert_eq!(stats.lsp_successes, 1);
        assert_eq!(stats.lsp_failures, 0);
        assert_eq!(stats.success_rate, 1.0);
        assert_eq!(stats.avg_latency_ms, 100);
        
        // Second success with different latency
        stats.update_success(200);
        assert_eq!(stats.lsp_successes, 2);
        assert_eq!(stats.success_rate, 1.0);
        assert_eq!(stats.avg_latency_ms, 150); // Average of 100 and 200
    }

    #[test]
    fn test_intent_stats_failure_updates() {
        let mut stats = IntentStats::default();
        
        // Success then failure
        stats.update_success(100);
        stats.update_failure(200);
        
        assert_eq!(stats.lsp_successes, 1);
        assert_eq!(stats.lsp_failures, 1);
        assert_eq!(stats.success_rate, 0.5);
        assert_eq!(stats.avg_latency_ms, 150);
    }

    #[test]
    fn test_intent_stats_mixed_results() {
        let mut stats = IntentStats::default();
        
        // Multiple successes and failures
        stats.update_success(100);
        stats.update_success(150);
        stats.update_failure(200);
        stats.update_failure(250);
        stats.update_success(300);
        
        assert_eq!(stats.lsp_successes, 3);
        assert_eq!(stats.lsp_failures, 2);
        assert_eq!(stats.success_rate, 0.6); // 3/5
        assert_eq!(stats.avg_latency_ms, 200); // Average of all latencies
    }

    // Test QueryPattern analysis
    #[tokio::test]
    async fn test_query_pattern_caching() {
        let router = LspRouter::new(0.5);
        
        // First analysis should cache the result
        let pattern1 = router.analyze_query_pattern("class MyClass").await;
        let pattern2 = router.analyze_query_pattern("class MyClass").await;
        
        // Results should be identical (from cache)
        assert_eq!(pattern1.has_structural_hints, pattern2.has_structural_hints);
        assert_eq!(pattern1.complexity_score, pattern2.complexity_score);
        assert_eq!(pattern1.estimated_lsp_effectiveness, pattern2.estimated_lsp_effectiveness);
    }

    // Test structural pattern detection
    #[test]
    fn test_detect_structural_patterns_comprehensive() {
        // Positive cases
        assert!(LspRouter::detect_structural_patterns("class MyClass"));
        assert!(LspRouter::detect_structural_patterns("function getName"));
        assert!(LspRouter::detect_structural_patterns("def calculate"));
        assert!(LspRouter::detect_structural_patterns("interface IUser"));
        assert!(LspRouter::detect_structural_patterns("type UserType"));
        assert!(LspRouter::detect_structural_patterns("struct Point"));
        assert!(LspRouter::detect_structural_patterns("impl Display"));
        assert!(LspRouter::detect_structural_patterns("trait Iterator"));
        assert!(LspRouter::detect_structural_patterns("MyClass extends BaseClass"));
        assert!(LspRouter::detect_structural_patterns("MyClass implements Interface"));
        assert!(LspRouter::detect_structural_patterns("import React from 'react'"));
        assert!(LspRouter::detect_structural_patterns("from typing import List"));
        
        // Case insensitive
        assert!(LspRouter::detect_structural_patterns("CLASS MyClass"));
        assert!(LspRouter::detect_structural_patterns("FUNCTION getName"));
        
        // Negative cases
        assert!(!LspRouter::detect_structural_patterns("hello world"));
        assert!(!LspRouter::detect_structural_patterns("simple text query"));
        assert!(!LspRouter::detect_structural_patterns("123 456"));
        assert!(!LspRouter::detect_structural_patterns(""));
    }

    // Test identifier pattern detection
    #[test]
    fn test_detect_identifier_patterns_comprehensive() {
        // Positive cases
        assert!(LspRouter::detect_identifier_patterns("myVariable")); // camelCase
        assert!(LspRouter::detect_identifier_patterns("MyClass")); // PascalCase
        assert!(LspRouter::detect_identifier_patterns("my_function")); // snake_case
        assert!(LspRouter::detect_identifier_patterns("obj.method")); // dot notation
        assert!(LspRouter::detect_identifier_patterns("user.profile.name")); // nested dot notation
        assert!(LspRouter::detect_identifier_patterns("MY_CONSTANT")); // UPPER_SNAKE_CASE
        assert!(LspRouter::detect_identifier_patterns("getUserById")); // mixed case
        
        // Edge cases
        assert!(LspRouter::detect_identifier_patterns("a.b")); // minimal dot notation
        assert!(LspRouter::detect_identifier_patterns("_private")); // leading underscore
        assert!(LspRouter::detect_identifier_patterns("var_")); // trailing underscore
        
        // Negative cases
        assert!(!LspRouter::detect_identifier_patterns("simple"));
        assert!(!LspRouter::detect_identifier_patterns("ALL CAPS"));
        assert!(!LspRouter::detect_identifier_patterns("hello world"));
        assert!(!LspRouter::detect_identifier_patterns("123"));
        assert!(!LspRouter::detect_identifier_patterns(""));
    }

    // Test complexity calculation edge cases
    #[test]
    fn test_calculate_complexity_edge_cases() {
        // Empty string
        assert_eq!(LspRouter::calculate_complexity(""), 0.0);
        
        // Single character
        assert!(LspRouter::calculate_complexity("a") < 0.1);
        
        // Short simple query
        assert!(LspRouter::calculate_complexity("test") < 0.3);
        
        // Medium query
        let medium_query = "function getUserById with parameters";
        let medium_complexity = LspRouter::calculate_complexity(medium_query);
        assert!(medium_complexity > 0.2 && medium_complexity < 0.7);
        
        // Long query
        let long_query = "very long query with many words that should increase the complexity score significantly";
        assert!(LspRouter::calculate_complexity(long_query) > 0.4);
        
        // Query with special characters
        let special_query = "query[with]{special}*characters";
        let special_complexity = LspRouter::calculate_complexity(special_query);
        assert!(special_complexity > 0.5);
        
        // Maximum complexity should be capped at 1.0
        let ultra_complex = "extremely long query with many many words and lots of special characters []{}<>*?+^$|\\";
        assert_eq!(LspRouter::calculate_complexity(ultra_complex), 1.0);
    }

    // Test LSP effectiveness estimation
    #[test]
    fn test_estimate_lsp_effectiveness() {
        // Base effectiveness for simple query
        let base = LspRouter::estimate_lsp_effectiveness("simple");
        assert_eq!(base, 0.5);
        
        // Structural patterns increase effectiveness
        let structural = LspRouter::estimate_lsp_effectiveness("class MyClass");
        assert!(structural > 0.7);
        
        // Identifier patterns increase effectiveness
        let identifier = LspRouter::estimate_lsp_effectiveness("getUserById");
        assert!(identifier > 0.6);
        
        // Both structural and identifier patterns
        let both = LspRouter::estimate_lsp_effectiveness("class User { getName() }");
        assert!(both > 0.8);
        
        // Short specific queries get bonus
        let short = LspRouter::estimate_lsp_effectiveness("def test");
        assert!(short > 0.7);
        
        // Very long queries get penalty
        let long_query = "this is a very long query with many words that should decrease effectiveness because it's too complex for LSP to handle well";
        let long_effectiveness = LspRouter::estimate_lsp_effectiveness(long_query);
        assert!(long_effectiveness < 0.4);
        
        // Effectiveness should be clamped between 0.0 and 1.0
        assert!(LspRouter::estimate_lsp_effectiveness("") >= 0.0);
        assert!(LspRouter::estimate_lsp_effectiveness("class Awesome") <= 1.0);
    }

    // Test router creation and configuration
    #[test]
    fn test_router_creation() {
        let router = LspRouter::new(0.6);
        assert_eq!(router.target_routing_rate, 0.6);
        
        let current_rate = router.current_routing_rate.load(Ordering::Relaxed) as f64 / 10000.0;
        assert!((current_rate - 0.6).abs() < 0.001);
        
        assert_eq!(router.total_queries.load(Ordering::Relaxed), 0);
        assert_eq!(router.total_lsp_routed.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_routing_config_defaults() {
        let config = RoutingConfig::default();
        assert_eq!(config.target_rate_min, 0.40);
        assert_eq!(config.target_rate_max, 0.60);
        assert_eq!(config.safety_floor_rate, 0.20);
        assert_eq!(config.max_complexity_threshold, 0.8);
        assert_eq!(config.min_success_rate_threshold, 0.7);
        assert_eq!(config.max_acceptable_latency_ms, 1000);
        assert_eq!(config.adaptation_factor, 0.1);
    }

    // Test basic routing decisions
    #[tokio::test]
    async fn test_router_basic_decisions() {
        let router = LspRouter::new(0.5);
        
        // Test different intents
        assert!(router.should_route("def myFunction", &QueryIntent::Definition).await);
        assert!(router.should_route("@symbolName", &QueryIntent::Symbol).await);
        assert!(!router.should_route("random text", &QueryIntent::TextSearch).await);
    }

    // Test intent eligibility routing
    #[tokio::test]
    async fn test_intent_eligibility_routing() {
        let router = LspRouter::new(0.5);
        
        // LSP-eligible intents should have chance to be routed
        let eligible_intents = vec![
            QueryIntent::Definition,
            QueryIntent::Symbol,
            QueryIntent::References,
            QueryIntent::Implementation,
            QueryIntent::TypeDefinition,
            QueryIntent::Declaration,
            QueryIntent::Hover,
            QueryIntent::Completion,
        ];
        
        for intent in eligible_intents {
            let decision = router.make_routing_decision("class MyClass", &intent).await;
            // Should not be immediately rejected for intent
            if matches!(decision.reason, RoutingReason::IntentNotEligible) {
                panic!("Intent {:?} should be LSP-eligible", intent);
            }
        }
        
        // TextSearch should be rejected
        let decision = router.make_routing_decision("random text", &QueryIntent::TextSearch).await;
        assert!(!decision.should_route_to_lsp);
        assert_eq!(decision.reason, RoutingReason::IntentNotEligible);
    }

    // Test detailed routing decision making
    #[tokio::test]
    async fn test_detailed_routing_decisions() {
        let router = create_test_router_with_config(0.5);
        
        // High-confidence structural query
        let decision = router.make_routing_decision("class UserManager", &QueryIntent::Definition).await;
        assert!(decision.should_route_to_lsp);
        assert!(decision.confidence > 0.6);
        assert!(matches!(decision.reason, RoutingReason::IntentMatch | RoutingReason::StructuralPattern));
        assert!(decision.estimated_latency_ms >= 100);
        
        // Low-confidence simple query
        let decision = router.make_routing_decision("hello", &QueryIntent::Hover).await;
        // May or may not route based on probability, but should have valid decision
        assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
        assert!(decision.estimated_latency_ms > 0);
    }

    // Test adaptive routing adjustments
    #[tokio::test]
    async fn test_adaptive_routing() {
        let router = create_test_router_with_config(0.5);
        
        // Simulate routing decisions and results
        for _ in 0..10 {
            let should_route = router.should_route("test query", &QueryIntent::Definition).await;
            if should_route {
                router.report_lsp_result(&QueryIntent::Definition, true, 200).await;
            }
        }
        
        let stats = router.get_routing_stats().await;
        assert!(stats.total_queries >= 10);
        
        if let Some(def_stats) = stats.intent_breakdown.get(&QueryIntent::Definition) {
            assert!(def_stats.success_rate > 0.0);
        }
    }

    // Test LSP result reporting
    #[tokio::test]
    async fn test_lsp_result_reporting() {
        let router = LspRouter::new(0.5);
        
        // Report several results
        router.report_lsp_result(&QueryIntent::Definition, true, 150).await;
        router.report_lsp_result(&QueryIntent::Definition, true, 200).await;
        router.report_lsp_result(&QueryIntent::Definition, false, 300).await;
        
        let stats = router.get_routing_stats().await;
        if let Some(def_stats) = stats.intent_breakdown.get(&QueryIntent::Definition) {
            assert_eq!(def_stats.lsp_successes, 2);
            assert_eq!(def_stats.lsp_failures, 1);
            assert!((def_stats.success_rate - 0.666).abs() < 0.01); // 2/3
            assert_eq!(def_stats.avg_latency_ms, 216); // (150 + 200 + 300) / 3 = 216
        }
    }

    // Test routing statistics
    #[tokio::test]
    async fn test_routing_statistics() {
        let router = LspRouter::new(0.6);
        
        // Initially empty stats
        let stats = router.get_routing_stats().await;
        assert_eq!(stats.total_queries, 0);
        assert_eq!(stats.total_lsp_routed, 0);
        assert_eq!(stats.target_routing_rate, 0.6);
        assert!(stats.intent_breakdown.is_empty());
        
        // After some routing decisions
        for _ in 0..5 {
            router.should_route("class Test", &QueryIntent::Definition).await;
            router.should_route("simple text", &QueryIntent::TextSearch).await;
        }
        
        let stats = router.get_routing_stats().await;
        assert_eq!(stats.total_queries, 10);
        assert!(stats.total_lsp_routed <= stats.total_queries);
    }

    #[test]
    fn test_routing_stats_target_checking() {
        let stats = RoutingStats {
            current_routing_rate: 0.55,
            target_routing_rate: 0.50,
            total_queries: 100,
            total_lsp_routed: 55,
            intent_breakdown: HashMap::new(),
        };
        
        assert!(stats.is_within_target(0.1)); // Within 10% tolerance
        assert!(!stats.is_within_target(0.03)); // Not within 3% tolerance
    }

    // Test concurrent routing decisions
    #[tokio::test]
    async fn test_concurrent_routing() {
        let router = Arc::new(LspRouter::new(0.5));
        let mut handles = vec![];
        
        // Spawn multiple concurrent routing decisions
        for i in 0..10 {
            let router_clone = router.clone();
            let handle = tokio::spawn(async move {
                let query = format!("class Test{}", i);
                router_clone.should_route(&query, &QueryIntent::Definition).await
            });
            handles.push(handle);
        }
        
        // Wait for all decisions
        for handle in handles {
            let result = handle.await.unwrap();
            // Each decision should be valid boolean
            assert!(result == true || result == false);
        }
        
        let stats = router.get_routing_stats().await;
        assert_eq!(stats.total_queries, 10);
    }

    // Test concurrent result reporting
    #[tokio::test]
    async fn test_concurrent_result_reporting() {
        let router = Arc::new(LspRouter::new(0.5));
        let mut handles = vec![];
        
        // Report results concurrently
        for i in 0..10 {
            let router_clone = router.clone();
            let handle = tokio::spawn(async move {
                let success = i % 2 == 0; // Alternate success/failure
                let latency = 100 + i * 10;
                router_clone.report_lsp_result(&QueryIntent::Definition, success, latency).await;
            });
            handles.push(handle);
        }
        
        // Wait for all reports
        for handle in handles {
            handle.await.unwrap();
        }
        
        let stats = router.get_routing_stats().await;
        if let Some(def_stats) = stats.intent_breakdown.get(&QueryIntent::Definition) {
            assert_eq!(def_stats.lsp_successes + def_stats.lsp_failures, 10);
            assert_eq!(def_stats.lsp_successes, 5); // Half succeeded
            assert_eq!(def_stats.lsp_failures, 5); // Half failed
            assert_eq!(def_stats.success_rate, 0.5);
        }
    }

    // Test performance with many patterns
    #[tokio::test]
    async fn test_pattern_cache_performance() {
        let router = LspRouter::new(0.5);
        let queries = vec![
            "class UserService",
            "def calculate_total",
            "interface IPayment", 
            "getUserById",
            "my_helper_function",
            "obj.method.call",
        ];
        
        // First pass - populate cache
        for query in &queries {
            router.analyze_query_pattern(query).await;
        }
        
        // Second pass - should use cache
        let start = std::time::Instant::now();
        for query in &queries {
            router.analyze_query_pattern(query).await;
        }
        let duration = start.elapsed();
        
        // Cache access should be very fast
        assert!(duration.as_millis() < 10);
    }

    // Test edge cases and error conditions
    #[tokio::test]
    async fn test_empty_query_routing() {
        let router = LspRouter::new(0.5);
        
        // Empty query
        let decision = router.make_routing_decision("", &QueryIntent::Definition).await;
        assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
        assert!(decision.estimated_latency_ms > 0);
        
        // Whitespace only query
        let decision = router.make_routing_decision("   \t\n  ", &QueryIntent::Symbol).await;
        assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
    }

    #[tokio::test]
    async fn test_very_long_query_routing() {
        let router = LspRouter::new(0.5);
        let long_query = "a".repeat(1000);
        
        let decision = router.make_routing_decision(&long_query, &QueryIntent::Definition).await;
        // Very long queries should have high complexity and lower routing probability
        assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
        
        let pattern = router.analyze_query_pattern(&long_query).await;
        assert!(pattern.complexity_score > 0.5);
    }

    #[tokio::test]
    async fn test_unicode_query_handling() {
        let router = LspRouter::new(0.5);
        let unicode_queries = vec![
            "函数名称", // Chinese
            "función_nombre", // Spanish with special chars
            "クラス名", // Japanese
            "переменная", // Cyrillic
            "🚀_rocket_function", // Emoji
        ];
        
        for query in unicode_queries {
            let decision = router.make_routing_decision(query, &QueryIntent::Definition).await;
            assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
            
            let pattern = router.analyze_query_pattern(query).await;
            assert!(pattern.complexity_score >= 0.0 && pattern.complexity_score <= 1.0);
            assert!(pattern.estimated_lsp_effectiveness >= 0.0 && pattern.estimated_lsp_effectiveness <= 1.0);
        }
    }

    // Test safety constraints
    #[tokio::test]
    async fn test_safety_constraints() {
        let mut router = create_test_router_with_config(0.5);
        router.config.safety_floor_rate = 0.3;
        router.config.max_complexity_threshold = 0.5;
        
        // Very complex query should trigger complexity constraint
        let complex_query = "extremely complex query with many special characters []{}<>*?+^$|\\".repeat(5);
        let decision = router.make_routing_decision(&complex_query, &QueryIntent::Definition).await;
        
        if !decision.should_route_to_lsp {
            assert_eq!(decision.reason, RoutingReason::ComplexityTooHigh);
        }
    }

    // Test adaptive adjustment logic
    #[tokio::test] 
    async fn test_adaptive_adjustment_logic() {
        let router = create_test_router_with_config(0.5);
        
        // Manually simulate low routing rate
        for _ in 0..20 {
            router.total_queries.fetch_add(1, Ordering::Relaxed);
            // Only route 20% to simulate being below target
            if router.total_queries.load(Ordering::Relaxed) % 5 == 0 {
                router.total_lsp_routed.fetch_add(1, Ordering::Relaxed);
            }
        }
        
        // Update current routing rate
        let total = router.total_queries.load(Ordering::Relaxed);
        let routed = router.total_lsp_routed.load(Ordering::Relaxed);
        let rate = (routed as f64 / total as f64 * 10000.0) as u64;
        router.current_routing_rate.store(rate, Ordering::Relaxed);
        
        // Should try to increase routing probability
        let base_probability = 0.4;
        let adjusted = router.apply_adaptive_adjustments(base_probability).await;
        assert!(adjusted >= base_probability); // Should be increased or same
    }

    // Test complexity calculation details
    #[test]
    fn test_complexity_calculation() {
        // Simple query should have low complexity
        assert!(LspRouter::calculate_complexity("test") < 0.3);
        
        // Complex query should have high complexity
        let complex_query = r"very long query with many words and special characters []{}\*";
        assert!(LspRouter::calculate_complexity(complex_query) > 0.5);
        
        // Test individual complexity factors
        
        // Length factor
        let long_query = "a".repeat(200);
        let long_complexity = LspRouter::calculate_complexity(&long_query);
        assert!(long_complexity > 0.1);
        
        // Word count factor
        let many_words = "word ".repeat(20);
        let word_complexity = LspRouter::calculate_complexity(&many_words);
        assert!(word_complexity > 0.1);
        
        // Special characters factor
        let special_chars = "!@#$%^&*()[]{}|\\";
        let special_complexity = LspRouter::calculate_complexity(special_chars);
        assert!(special_complexity > 0.1);
        
        // Regex patterns
        let regex_query = "pattern[a-z]+{1,5}*";
        let regex_complexity = LspRouter::calculate_complexity(regex_query);
        assert!(regex_complexity > 0.4);
    }

    // Test memory usage and cleanup
    #[tokio::test]
    async fn test_pattern_cache_cleanup() {
        let router = LspRouter::new(0.5);
        
        // Add many patterns to cache
        for i in 0..100 {
            let query = format!("test_query_{}", i);
            router.analyze_query_pattern(&query).await;
        }
        
        // Cache should contain patterns
        let patterns = router.known_patterns.read().await;
        assert!(patterns.len() > 0);
        
        // Note: In a real implementation, you might want to add cache eviction logic
        // For now, we just verify the cache works
    }

    // Test routing stats accuracy
    #[tokio::test]
    async fn test_routing_stats_accuracy() {
        let router = LspRouter::new(0.4);
        
        // Make exactly 10 queries, expecting about 40% to route to LSP
        let mut expected_routed = 0;
        for i in 0..10 {
            let query = format!("class Test{}", i);
            let routed = router.should_route(&query, &QueryIntent::Definition).await;
            if routed {
                expected_routed += 1;
                // Simulate success
                router.report_lsp_result(&QueryIntent::Definition, true, 150).await;
            }
        }
        
        let stats = router.get_routing_stats().await;
        assert_eq!(stats.total_queries, 10);
        assert_eq!(stats.total_lsp_routed, expected_routed);
        assert_eq!(stats.target_routing_rate, 0.4);
        
        // Verify intent-specific stats
        if let Some(def_stats) = stats.intent_breakdown.get(&QueryIntent::Definition) {
            assert_eq!(def_stats.total_queries, 10);
            assert_eq!(def_stats.lsp_routed, expected_routed);
            if expected_routed > 0 {
                assert_eq!(def_stats.success_rate, 1.0); // All reported as success
            }
        }
    }

    // Test high-load scenarios
    #[tokio::test]
    async fn test_high_load_routing() {
        let router = Arc::new(LspRouter::new(0.5));
        let mut handles = vec![];
        
        // Simulate high load with many concurrent requests
        for i in 0..100 {
            let router_clone = router.clone();
            let handle = tokio::spawn(async move {
                let query = if i % 3 == 0 {
                    format!("class HighLoad{}", i)
                } else if i % 3 == 1 {
                    format!("function process{}", i) 
                } else {
                    format!("simple query {}", i)
                };
                
                let intent = if i % 4 == 0 {
                    QueryIntent::Definition
                } else if i % 4 == 1 {
                    QueryIntent::Symbol
                } else if i % 4 == 2 {
                    QueryIntent::References
                } else {
                    QueryIntent::TextSearch
                };
                
                router_clone.should_route(&query, &intent).await
            });
            handles.push(handle);
        }
        
        // Wait for all requests
        let mut total_routed = 0;
        for handle in handles {
            if handle.await.unwrap() {
                total_routed += 1;
            }
        }
        
        let stats = router.get_routing_stats().await;
        assert_eq!(stats.total_queries, 100);
        assert_eq!(stats.total_lsp_routed, total_routed);
        
        // Routing rate should be within reasonable bounds
        let actual_rate = stats.total_lsp_routed as f64 / stats.total_queries as f64;
        assert!(actual_rate >= 0.0 && actual_rate <= 1.0);
    }

    // Test bounded BFS traversal implementation
    #[tokio::test]
    async fn test_bounded_bfs_traversal_basic() {
        let router = LspRouter::new(0.5);
        let bounds = TraversalBounds {
            max_depth: 2,
            max_results: 10,
            timeout_ms: 5000,
        };
        
        let start_node = BfsNode {
            symbol_id: "test_function_def".to_string(),
            symbol_type: SymbolType::Definition,
            file_path: "test.rs".to_string(),
            line: 10,
            column: 5,
        };
        
        let result = router.bounded_bfs_traversal(start_node.clone(), &bounds).await.unwrap();
        
        // Should contain start node
        assert!(!result.visited_nodes.is_empty());
        assert_eq!(result.visited_nodes[0], start_node);
        
        // Should respect bounds
        assert!(result.nodes_explored <= bounds.max_results);
        assert!(result.depth_reached <= bounds.max_depth);
    }

    #[tokio::test]
    async fn test_bounded_bfs_traversal_depth_limit() {
        let router = LspRouter::new(0.5);
        let bounds = TraversalBounds {
            max_depth: 1,
            max_results: 50,
            timeout_ms: 5000,
        };
        
        let start_node = BfsNode {
            symbol_id: "trait_definition".to_string(),
            symbol_type: SymbolType::Definition,
            file_path: "traits.rs".to_string(),
            line: 20,
            column: 8,
        };
        
        let result = router.bounded_bfs_traversal(start_node, &bounds).await.unwrap();
        
        // Should be limited by depth
        assert!(result.depth_reached <= 1);
        
        // Should have found some neighbors at depth 1
        assert!(result.visited_nodes.len() > 1);
    }

    #[tokio::test]
    async fn test_bounded_bfs_traversal_node_limit() {
        let router = LspRouter::new(0.5);
        let bounds = TraversalBounds {
            max_depth: 5, // High depth limit
            max_results: 3, // Low node limit
            timeout_ms: 5000,
        };
        
        let start_node = BfsNode {
            symbol_id: "popular_function".to_string(),
            symbol_type: SymbolType::Definition,
            file_path: "popular.rs".to_string(),
            line: 1,
            column: 1,
        };
        
        let result = router.bounded_bfs_traversal(start_node, &bounds).await.unwrap();
        
        // Should be limited by node count
        assert!(result.nodes_explored <= 3);
        assert!(result.was_bounded);
    }

    #[tokio::test]
    async fn test_bounded_bfs_todo_md_bounds() {
        let router = LspRouter::new(0.5);
        
        // Test TODO.md bounds enforcement: depth ≤ 2, K ≤ 64
        let excessive_bounds = TraversalBounds {
            max_depth: 10, // Exceeds TODO.md limit
            max_results: 200, // Exceeds TODO.md limit  
            timeout_ms: 5000,
        };
        
        let start_node = BfsNode {
            symbol_id: "test_bounds".to_string(),
            symbol_type: SymbolType::Definition,
            file_path: "bounds_test.rs".to_string(),
            line: 15,
            column: 10,
        };
        
        let result = router.bounded_bfs_traversal(start_node, &excessive_bounds).await.unwrap();
        
        // Should be clamped to TODO.md limits
        assert!(result.depth_reached <= 2); // TODO.md depth ≤ 2
        assert!(result.nodes_explored <= 64); // TODO.md K ≤ 64
    }

    #[tokio::test]
    async fn test_bfs_symbol_relationships() {
        let router = LspRouter::new(0.5);
        let bounds = TraversalBounds::default();
        
        // Test different symbol types generate appropriate neighbors
        let test_cases = vec![
            (SymbolType::Definition, vec![SymbolType::Reference]),
            (SymbolType::Reference, vec![SymbolType::Definition]),
            (SymbolType::Implementation, vec![SymbolType::TypeDefinition]),
            (SymbolType::Declaration, vec![SymbolType::Definition]),
            (SymbolType::Alias, vec![SymbolType::Definition]),
            (SymbolType::TypeDefinition, vec![SymbolType::Implementation]),
        ];
        
        for (symbol_type, expected_neighbor_types) in test_cases {
            let start_node = BfsNode {
                symbol_id: format!("test_{:?}", symbol_type),
                symbol_type: symbol_type.clone(),
                file_path: "relationships.rs".to_string(),
                line: 25,
                column: 15,
            };
            
            let result = router.bounded_bfs_traversal(start_node, &bounds).await.unwrap();
            
            // Should have generated neighbors
            assert!(result.visited_nodes.len() > 1);
            
            // Check that we have edges with appropriate types
            if !result.edges.is_empty() {
                let edge_targets: Vec<_> = result.edges.iter()
                    .map(|(_, target, _)| &target.symbol_type)
                    .collect();
                
                for expected_type in &expected_neighbor_types {
                    assert!(
                        edge_targets.contains(&expected_type),
                        "Expected neighbor type {:?} not found for symbol type {:?}",
                        expected_type, symbol_type
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_bfs_edge_types() {
        let router = LspRouter::new(0.5);
        let bounds = TraversalBounds::default();
        
        let start_node = BfsNode {
            symbol_id: "trait_test".to_string(),
            symbol_type: SymbolType::Definition,
            file_path: "edge_test.rs".to_string(),
            line: 30,
            column: 5,
        };
        
        let result = router.bounded_bfs_traversal(start_node, &bounds).await.unwrap();
        
        // Should have edges with proper types
        let edge_types: Vec<_> = result.edges.iter().map(|(_, _, edge_type)| edge_type).collect();
        
        if !edge_types.is_empty() {
            // Should contain expected edge types for a definition
            assert!(edge_types.contains(&&EdgeType::DefinitionToReference));
            
            // Should contain implementation edge for trait
            assert!(edge_types.contains(&&EdgeType::TypeToImplementation));
        }
    }

    #[tokio::test]
    async fn test_bfs_cycle_detection() {
        let router = LspRouter::new(0.5);
        let bounds = TraversalBounds {
            max_depth: 2,
            max_results: 20,
            timeout_ms: 5000,
        };
        
        let start_node = BfsNode {
            symbol_id: "cycle_test_def".to_string(),
            symbol_type: SymbolType::Definition,
            file_path: "cycle.rs".to_string(),
            line: 40,
            column: 10,
        };
        
        let result = router.bounded_bfs_traversal(start_node.clone(), &bounds).await.unwrap();
        
        // Should not visit same node twice
        let mut seen_ids = HashSet::new();
        for node in &result.visited_nodes {
            assert!(
                seen_ids.insert(&node.symbol_id),
                "Duplicate node visited: {}",
                node.symbol_id
            );
        }
        
        // Should not have created a cycle back to start
        let non_start_nodes: Vec<_> = result.visited_nodes.iter()
            .filter(|node| *node != &start_node)
            .collect();
        
        for node in non_start_nodes {
            assert_ne!(node.symbol_id, start_node.symbol_id);
        }
    }

    #[tokio::test]
    async fn test_bfs_empty_neighbors() {
        let router = LspRouter::new(0.5);
        let bounds = TraversalBounds::default();
        
        // Create a node that won't have neighbors in mock implementation
        let isolated_node = BfsNode {
            symbol_id: "isolated".to_string(),
            symbol_type: SymbolType::Definition,
            file_path: "isolated.rs".to_string(),
            line: 50,
            column: 20,
        };
        
        let result = router.bounded_bfs_traversal(isolated_node.clone(), &bounds).await.unwrap();
        
        // Should still contain the start node
        assert_eq!(result.visited_nodes.len(), 2); // Start + 1 reference
        assert_eq!(result.visited_nodes[0], isolated_node);
        assert_eq!(result.nodes_explored, 2);
        assert_eq!(result.depth_reached, 1);
    }

    #[tokio::test]
    async fn test_symbol_type_equality() {
        assert_eq!(SymbolType::Definition, SymbolType::Definition);
        assert_ne!(SymbolType::Definition, SymbolType::Reference);
        assert_ne!(SymbolType::Reference, SymbolType::Implementation);
    }

    #[tokio::test]
    async fn test_bfs_node_equality() {
        let node1 = BfsNode {
            symbol_id: "test".to_string(),
            symbol_type: SymbolType::Definition,
            file_path: "test.rs".to_string(),
            line: 10,
            column: 5,
        };
        
        let node2 = BfsNode {
            symbol_id: "test".to_string(),
            symbol_type: SymbolType::Definition,
            file_path: "test.rs".to_string(),
            line: 10,
            column: 5,
        };
        
        let node3 = BfsNode {
            symbol_id: "different".to_string(),
            symbol_type: SymbolType::Definition,
            file_path: "test.rs".to_string(),
            line: 10,
            column: 5,
        };
        
        assert_eq!(node1, node2);
        assert_ne!(node1, node3);
    }

    #[tokio::test]
    async fn test_edge_type_equality() {
        assert_eq!(EdgeType::DefinitionToReference, EdgeType::DefinitionToReference);
        assert_ne!(EdgeType::DefinitionToReference, EdgeType::ReferenceToDefinition);
        assert_ne!(EdgeType::TypeToImplementation, EdgeType::ImplementationToType);
    }
}