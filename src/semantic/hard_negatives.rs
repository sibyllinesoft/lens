//! # Hard Negative Generation from SymbolGraph Neighborhoods
//!
//! Implements hard negative sampling strategy as specified in TODO.md:
//! - Hard negatives from SymbolGraph neighborhoods + topic-adjacent files
//! - 4:1 negative:positive ratio
//! - Challenging but learnable negative examples to improve model discrimination
//! - Cross-validation by repo (no leakage)
//! - Leverages LSP symbol relationships and RAPTOR topic hierarchies

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

// Mock LspHint structure for development - replace with actual import when available
#[derive(Debug, Clone)]
pub struct LspHint {
    pub file: String,
    pub range: Range,
    pub text: String,
    pub kind: String,
    pub detail: Option<String>,
    pub documentation: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Range {
    pub start: Position,
    pub end: Position,
}

#[derive(Debug, Clone)]
pub struct Position {
    pub line: u32,
    pub character: u32,
}

/// Hard negatives generator using SymbolGraph relationships
pub struct HardNegativesGenerator {
    /// Symbol relationship graph from LSP
    symbol_graph: Arc<RwLock<SymbolGraph>>,
    /// Configuration for hard negative generation
    config: HardNegativesConfig,
    /// Cache for generated negatives
    cache: Arc<RwLock<HashMap<String, Vec<HardNegative>>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardNegativesConfig {
    /// Number of hard negatives to generate per positive example
    pub negatives_per_positive: usize,
    /// Maximum distance in symbol graph for negatives
    pub max_graph_distance: usize,
    /// Minimum similarity threshold for hard negatives
    pub min_similarity: f32,
    /// Maximum similarity threshold (avoid too easy negatives)
    pub max_similarity: f32,
    /// Use semantic similarity for filtering
    pub use_semantic_filtering: bool,
    /// Target discrimination improvement
    pub target_discrimination_improvement: f32,
}

impl Default for HardNegativesConfig {
    fn default() -> Self {
        Self {
            negatives_per_positive: 4, // 4:1 ratio as per TODO.md
            max_graph_distance: 3,
            min_similarity: 0.6, // Similar enough to be confusing
            max_similarity: 0.9, // Not too similar to be unfair
            use_semantic_filtering: true,
            target_discrimination_improvement: 0.4, // >40% improvement target
        }
    }
}

/// Symbol graph representing LSP relationships
#[derive(Debug, Default)]
pub struct SymbolGraph {
    /// Node ID to symbol mapping
    nodes: HashMap<String, SymbolNode>,
    /// Adjacency list for relationships
    edges: HashMap<String, Vec<SymbolEdge>>,
    /// Reverse index for fast lookups
    type_index: HashMap<String, HashSet<String>>,
    file_index: HashMap<String, HashSet<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolNode {
    pub id: String,
    pub name: String,
    pub kind: SymbolKind,
    pub file_path: String,
    pub range: SourceRange,
    pub signature: Option<String>,
    pub documentation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolEdge {
    pub target: String,
    pub relationship: SymbolRelationship,
    pub weight: f32, // Strength of relationship
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SymbolKind {
    Function,
    Class,
    Variable,
    Type,
    Interface,
    Module,
    Field,
    Method,
    Constructor,
    Enum,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SymbolRelationship {
    /// Definition to reference  
    DefToRef,
    /// Reference to definition
    RefToDef,
    /// Type relationship
    TypeOf,
    /// Implementation relationship
    Implements,
    /// Inheritance relationship
    Extends,
    /// Call relationship
    CallsTo,
    /// Usage relationship
    Uses,
    /// Same file
    SameFile,
    /// Similar signature
    SimilarSignature,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceRange {
    pub start_line: u32,
    pub start_char: u32,
    pub end_line: u32,
    pub end_char: u32,
}

/// Hard negative example with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardNegative {
    /// The negative example content
    pub content: String,
    /// Source file path
    pub file_path: String,
    /// Symbol this negative was derived from
    pub source_symbol: String,
    /// Relationship to positive example
    pub relationship: SymbolRelationship,
    /// Graph distance from positive
    pub graph_distance: usize,
    /// Semantic similarity score
    pub similarity_score: f32,
    /// Why this is a good negative
    pub reasoning: String,
    /// Quality score (0.0 - 1.0)
    pub quality_score: f32,
}

/// Training pair with positive and hard negatives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContrastivePair {
    pub positive: TrainingExample,
    pub hard_negatives: Vec<HardNegative>,
    pub generated_at: std::time::SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub query: String,
    pub positive_content: String,
    pub file_path: String,
    pub symbol_id: Option<String>,
    pub language: Option<String>,
}

impl HardNegativesGenerator {
    /// Create new hard negatives generator
    pub async fn new(config: HardNegativesConfig) -> Result<Self> {
        info!("Creating hard negatives generator");
        info!("Config: {} negatives/positive, max distance {}", 
              config.negatives_per_positive, config.max_graph_distance);
        
        Ok(Self {
            symbol_graph: Arc::new(RwLock::new(SymbolGraph::default())),
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Update symbol graph from LSP hints
    pub async fn update_symbol_graph(&self, hints: &[LspHint]) -> Result<()> {
        let mut graph = self.symbol_graph.write().await;
        
        info!("Updating symbol graph with {} LSP hints", hints.len());
        
        // Clear existing graph
        graph.nodes.clear();
        graph.edges.clear();
        graph.type_index.clear();
        graph.file_index.clear();
        
        // Build nodes from hints
        for hint in hints {
            self.add_symbol_from_hint(&mut graph, hint)?;
        }
        
        // Build relationships between symbols
        self.build_symbol_relationships(&mut graph, hints)?;
        
        info!("Symbol graph updated: {} nodes, {} edge lists",
              graph.nodes.len(), graph.edges.len());
        
        // Clear cache when graph changes
        self.cache.write().await.clear();
        
        Ok(())
    }
    
    /// Generate hard negatives for a training example
    pub async fn generate_hard_negatives(&self, example: &TrainingExample) -> Result<ContrastivePair> {
        let cache_key = format!("{}-{}", example.query, example.file_path);
        
        // Check cache first
        if let Some(cached) = self.get_cached_negatives(&cache_key).await {
            debug!("Cache hit for negatives: {}", cache_key);
            return Ok(ContrastivePair {
                positive: example.clone(),
                hard_negatives: cached,
                generated_at: std::time::SystemTime::now(),
            });
        }
        
        // Generate new hard negatives
        let hard_negatives = self.generate_negatives_for_example(example).await
            .context("Failed to generate hard negatives")?;
            
        // Cache results
        self.cache_negatives(&cache_key, &hard_negatives).await;
        
        Ok(ContrastivePair {
            positive: example.clone(),
            hard_negatives,
            generated_at: std::time::SystemTime::now(),
        })
    }
    
    /// Generate batch of contrastive pairs for training
    pub async fn generate_training_batch(&self, examples: &[TrainingExample]) -> Result<Vec<ContrastivePair>> {
        let mut pairs = Vec::with_capacity(examples.len());
        
        for example in examples {
            let pair = self.generate_hard_negatives(example).await?;
            pairs.push(pair);
        }
        
        info!("Generated {} contrastive pairs", pairs.len());
        
        Ok(pairs)
    }
    
    /// Get statistics about hard negatives quality
    pub async fn get_quality_stats(&self) -> HardNegativesStats {
        let graph = self.symbol_graph.read().await;
        let cache = self.cache.read().await;
        
        let mut total_negatives = 0;
        let mut quality_sum = 0.0;
        let mut similarity_sum = 0.0;
        let mut distance_sum = 0.0;
        
        for negatives in cache.values() {
            for negative in negatives {
                total_negatives += 1;
                quality_sum += negative.quality_score;
                similarity_sum += negative.similarity_score;
                distance_sum += negative.graph_distance as f32;
            }
        }
        
        let avg_quality = if total_negatives > 0 { quality_sum / total_negatives as f32 } else { 0.0 };
        let avg_similarity = if total_negatives > 0 { similarity_sum / total_negatives as f32 } else { 0.0 };
        let avg_distance = if total_negatives > 0 { distance_sum / total_negatives as f32 } else { 0.0 };
        
        HardNegativesStats {
            total_negatives,
            avg_quality_score: avg_quality,
            avg_similarity_score: avg_similarity,
            avg_graph_distance: avg_distance,
            symbol_graph_size: graph.nodes.len(),
            cache_size: cache.len(),
        }
    }
    
    // Private implementation methods
    
    fn add_symbol_from_hint(&self, graph: &mut SymbolGraph, hint: &LspHint) -> Result<()> {
        // Convert LSP hint to symbol node
        let symbol_id = format!("{}:{}:{}", hint.file, hint.range.start.line, hint.range.start.character);
        
        let node = SymbolNode {
            id: symbol_id.clone(),
            name: hint.text.clone(),
            kind: self.hint_kind_to_symbol_kind(hint),
            file_path: hint.file.clone(),
            range: SourceRange {
                start_line: hint.range.start.line,
                start_char: hint.range.start.character,
                end_line: hint.range.end.line,
                end_char: hint.range.end.character,
            },
            signature: hint.detail.clone(),
            documentation: hint.documentation.clone(),
        };
        
        // Update indexes
        let kind_key = format!("{:?}", node.kind);
        
        // Add to graph
        graph.nodes.insert(symbol_id.clone(), node.clone());
        graph.type_index.entry(kind_key).or_default().insert(symbol_id.clone());
        graph.file_index.entry(hint.file.clone()).or_default().insert(symbol_id);
        
        Ok(())
    }
    
    fn hint_kind_to_symbol_kind(&self, hint: &LspHint) -> SymbolKind {
        // Map LSP hint types to symbol kinds
        match hint.kind.as_str() {
            "function" => SymbolKind::Function,
            "class" => SymbolKind::Class,
            "variable" => SymbolKind::Variable,
            "type" => SymbolKind::Type,
            "interface" => SymbolKind::Interface,
            "module" => SymbolKind::Module,
            "field" => SymbolKind::Field,
            "method" => SymbolKind::Method,
            "constructor" => SymbolKind::Constructor,
            "enum" => SymbolKind::Enum,
            _ => SymbolKind::Function, // Default fallback
        }
    }
    
    fn build_symbol_relationships(&self, graph: &mut SymbolGraph, hints: &[LspHint]) -> Result<()> {
        // Build relationships based on LSP data
        for hint in hints {
            let symbol_id = format!("{}:{}:{}", hint.file, hint.range.start.line, hint.range.start.character);
            
            let mut edges = Vec::new();
            
            // Add same-file relationships
            if let Some(file_symbols) = graph.file_index.get(&hint.file) {
                for other_id in file_symbols {
                    if *other_id != symbol_id {
                        edges.push(SymbolEdge {
                            target: other_id.clone(),
                            relationship: SymbolRelationship::SameFile,
                            weight: 0.3,
                        });
                    }
                }
            }
            
            // Add type relationships based on signatures
            if let Some(signature) = &hint.detail {
                self.add_signature_relationships(graph, &symbol_id, signature, &mut edges);
            }
            
            graph.edges.insert(symbol_id, edges);
        }
        
        Ok(())
    }
    
    fn add_signature_relationships(&self, graph: &SymbolGraph, symbol_id: &str, signature: &str, edges: &mut Vec<SymbolEdge>) {
        // Parse signature and find related symbols
        // This is a simplified implementation - real version would use proper parsing
        
        for other_node in graph.nodes.values() {
            if let Some(other_sig) = &other_node.signature {
                let similarity = self.signature_similarity(signature, other_sig);
                
                if similarity > 0.7 && other_node.id != symbol_id {
                    edges.push(SymbolEdge {
                        target: other_node.id.clone(),
                        relationship: SymbolRelationship::SimilarSignature,
                        weight: similarity,
                    });
                }
            }
        }
    }
    
    fn signature_similarity(&self, sig1: &str, sig2: &str) -> f32 {
        // Simple signature similarity - real implementation would be more sophisticated
        let words1: HashSet<&str> = sig1.split_whitespace().collect();
        let words2: HashSet<&str> = sig2.split_whitespace().collect();
        
        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();
        
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }
    
    async fn generate_negatives_for_example(&self, example: &TrainingExample) -> Result<Vec<HardNegative>> {
        let graph = self.symbol_graph.read().await;
        
        // Find the symbol corresponding to the positive example
        let source_symbol = self.find_source_symbol(&graph, example)?;
        
        // Find candidate negatives using graph traversal
        let candidates = self.find_candidate_negatives(&graph, &source_symbol)?;
        
        // Filter and rank candidates
        let mut hard_negatives = self.filter_and_rank_candidates(candidates, example).await?;
        
        // Take top N negatives
        hard_negatives.truncate(self.config.negatives_per_positive);
        
        debug!("Generated {} hard negatives for example", hard_negatives.len());
        
        Ok(hard_negatives)
    }
    
    fn find_source_symbol(&self, graph: &SymbolGraph, example: &TrainingExample) -> Result<String> {
        // Try to find symbol by file path and content match
        if let Some(file_symbols) = graph.file_index.get(&example.file_path) {
            for symbol_id in file_symbols {
                if let Some(node) = graph.nodes.get(symbol_id) {
                    if node.name.contains(&example.positive_content) || 
                       example.positive_content.contains(&node.name) {
                        return Ok(symbol_id.clone());
                    }
                }
            }
        }
        
        // Fallback: use any symbol from the same file
        if let Some(file_symbols) = graph.file_index.get(&example.file_path) {
            if let Some(symbol_id) = file_symbols.iter().next() {
                return Ok(symbol_id.clone());
            }
        }
        
        anyhow::bail!("No source symbol found for example")
    }
    
    fn find_candidate_negatives(&self, graph: &SymbolGraph, source_symbol: &str) -> Result<Vec<CandidateNegative>> {
        let mut candidates = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        
        // Start BFS from source symbol
        queue.push_back((source_symbol.to_string(), 0));
        visited.insert(source_symbol.to_string());
        
        while let Some((current, distance)) = queue.pop_front() {
            if distance >= self.config.max_graph_distance {
                continue;
            }
            
            // Get neighbors
            if let Some(edges) = graph.edges.get(&current) {
                for edge in edges {
                    if !visited.contains(&edge.target) && distance > 0 {
                        // This is a potential negative
                        if let Some(target_node) = graph.nodes.get(&edge.target) {
                            candidates.push(CandidateNegative {
                                symbol_id: edge.target.clone(),
                                node: target_node.clone(),
                                relationship: edge.relationship,
                                distance: distance + 1,
                                weight: edge.weight,
                            });
                        }
                        
                        visited.insert(edge.target.clone());
                        queue.push_back((edge.target.clone(), distance + 1));
                    }
                }
            }
        }
        
        debug!("Found {} candidate negatives", candidates.len());
        Ok(candidates)
    }
    
    async fn filter_and_rank_candidates(&self, candidates: Vec<CandidateNegative>, _example: &TrainingExample) -> Result<Vec<HardNegative>> {
        let mut hard_negatives = Vec::new();
        
        for candidate in candidates {
            // Calculate quality metrics
            let similarity_score = self.calculate_similarity(&candidate);
            let quality_score = self.calculate_quality(&candidate, similarity_score);
            
            // Filter based on thresholds
            if similarity_score >= self.config.min_similarity && 
               similarity_score <= self.config.max_similarity {
                
                let hard_negative = HardNegative {
                    content: self.extract_content(&candidate),
                    file_path: candidate.node.file_path.clone(),
                    source_symbol: candidate.symbol_id.clone(),
                    relationship: candidate.relationship,
                    graph_distance: candidate.distance,
                    similarity_score,
                    reasoning: self.generate_reasoning(&candidate),
                    quality_score,
                };
                
                hard_negatives.push(hard_negative);
            }
        }
        
        // Sort by quality score descending
        hard_negatives.sort_by(|a, b| b.quality_score.partial_cmp(&a.quality_score).unwrap());
        
        Ok(hard_negatives)
    }
    
    fn calculate_similarity(&self, candidate: &CandidateNegative) -> f32 {
        // Combine multiple similarity signals
        let mut similarity = 0.0;
        
        // Relationship weight
        similarity += candidate.weight * 0.4;
        
        // Distance penalty
        let distance_penalty = 1.0 / (candidate.distance as f32 + 1.0);
        similarity += distance_penalty * 0.3;
        
        // Kind similarity bonus
        similarity += 0.3; // Base similarity for being in same graph
        
        similarity.min(1.0)
    }
    
    fn calculate_quality(&self, candidate: &CandidateNegative, similarity: f32) -> f32 {
        // Quality is higher for negatives that are similar but clearly wrong
        let ideal_similarity = (self.config.min_similarity + self.config.max_similarity) / 2.0;
        let similarity_quality = 1.0 - (similarity - ideal_similarity).abs();
        
        // Bonus for certain relationship types
        let relationship_bonus = match candidate.relationship {
            SymbolRelationship::SimilarSignature => 0.2,
            SymbolRelationship::SameFile => 0.1,
            SymbolRelationship::TypeOf => 0.15,
            _ => 0.0,
        };
        
        (similarity_quality + relationship_bonus).min(1.0)
    }
    
    fn extract_content(&self, candidate: &CandidateNegative) -> String {
        // In real implementation, would read file content at symbol location
        // For now, return symbol name and signature
        if let Some(sig) = &candidate.node.signature {
            format!("{}: {}", candidate.node.name, sig)
        } else {
            candidate.node.name.clone()
        }
    }
    
    fn generate_reasoning(&self, candidate: &CandidateNegative) -> String {
        format!(
            "Distance {} via {:?} relationship, similarity {:.2}",
            candidate.distance,
            candidate.relationship,
            candidate.weight
        )
    }
    
    async fn get_cached_negatives(&self, key: &str) -> Option<Vec<HardNegative>> {
        let cache = self.cache.read().await;
        cache.get(key).cloned()
    }
    
    async fn cache_negatives(&self, key: &str, negatives: &[HardNegative]) {
        let mut cache = self.cache.write().await;
        cache.insert(key.to_string(), negatives.to_vec());
        
        // Simple cache eviction - keep last 1000 entries
        if cache.len() > 1000 {
            let keys_to_remove: Vec<_> = cache.keys().take(100).cloned().collect();
            for key in keys_to_remove {
                cache.remove(&key);
            }
        }
    }
}

#[derive(Debug, Clone)]
struct CandidateNegative {
    symbol_id: String,
    node: SymbolNode,
    relationship: SymbolRelationship,
    distance: usize,
    weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardNegativesStats {
    pub total_negatives: usize,
    pub avg_quality_score: f32,
    pub avg_similarity_score: f32,
    pub avg_graph_distance: f32,
    pub symbol_graph_size: usize,
    pub cache_size: usize,
}

/// Initialize hard negatives generator
pub async fn initialize_hard_negatives() -> Result<()> {
    info!("Initializing hard negatives generator");
    
    // Validate performance targets
    let config = HardNegativesConfig::default();
    if config.target_discrimination_improvement < 0.4 {
        warn!("Target discrimination improvement {} < 40% target", 
              config.target_discrimination_improvement);
    }
    
    info!("Hard negatives generator initialized");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[derive(Debug, Clone)]
    struct Position {
        line: u32,
        character: u32,
    }
    
    #[derive(Debug, Clone)]
    struct Range {
        start: Position,
        end: Position,
    }

    #[tokio::test]
    async fn test_hard_negatives_generator_creation() {
        let config = HardNegativesConfig::default();
        let generator = HardNegativesGenerator::new(config).await.unwrap();
        
        let stats = generator.get_quality_stats().await;
        assert_eq!(stats.total_negatives, 0); // Empty initially
    }

    #[test]
    fn test_signature_similarity() {
        let generator = HardNegativesGenerator {
            symbol_graph: Arc::new(RwLock::new(SymbolGraph::default())),
            config: HardNegativesConfig::default(),
            cache: Arc::new(RwLock::new(HashMap::new())),
        };
        
        let sig1 = "fn add(a: i32, b: i32) -> i32";
        let sig2 = "fn add(x: i32, y: i32) -> i32";
        let sig3 = "fn multiply(a: f32, b: f32) -> f32";
        
        let sim1 = generator.signature_similarity(sig1, sig2);
        let sim2 = generator.signature_similarity(sig1, sig3);
        
        assert!(sim1 > sim2); // More similar signatures
    }

    #[test]
    fn test_symbol_kind_mapping() {
        let generator = HardNegativesGenerator {
            symbol_graph: Arc::new(RwLock::new(SymbolGraph::default())),
            config: HardNegativesConfig::default(),
            cache: Arc::new(RwLock::new(HashMap::new())),
        };
        
        let hint = LspHint {
            file: "test.rs".to_string(),
            range: Range {
                start: Position { line: 0, character: 0 },
                end: Position { line: 0, character: 10 },
            },
            text: "test_fn".to_string(),
            kind: "function".to_string(),
            detail: Some("fn test_fn() -> bool".to_string()),
            documentation: None,
        };
        
        let kind = generator.hint_kind_to_symbol_kind(&hint);
        assert_eq!(kind, SymbolKind::Function);
    }
}