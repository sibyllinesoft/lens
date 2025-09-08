use crate::search::SearchResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use anyhow::Result;

/// Result fusion engine for combining multiple search results
pub struct ResultFusion {
    strategies: Vec<Box<dyn FusionStrategy + Send + Sync>>,
    weights: HashMap<String, f64>,
}

/// Trait for different fusion strategies
#[async_trait::async_trait]
pub trait FusionStrategy {
    /// Get strategy name
    fn name(&self) -> &str;
    
    /// Fuse results from multiple search systems
    async fn fuse(&self, results: &[SystemResults]) -> Result<Vec<SearchResult>>;
    
    /// Calculate confidence score for fusion
    fn confidence(&self, results: &[SystemResults]) -> f64;
}

/// Results from a single search system
#[derive(Debug, Clone)]
pub struct SystemResults {
    pub system_name: String,
    pub results: Vec<SearchResult>,
    pub latency_ms: f64,
    pub confidence: f64,
}

/// Fused result with provenance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedResult {
    pub result: SearchResult,
    pub fusion_score: f64,
    pub contributing_systems: Vec<String>,
    pub fusion_strategy: String,
    pub confidence: f64,
}

impl ResultFusion {
    /// Create new result fusion engine
    pub fn new() -> Self {
        let mut fusion = Self {
            strategies: Vec::new(),
            weights: HashMap::new(),
        };
        
        // Register default strategies
        fusion.add_strategy(Box::new(CombSumStrategy::new()));
        fusion.add_strategy(Box::new(CombMnzStrategy::new()));
        fusion.add_strategy(Box::new(RankBasedFusion::new()));
        fusion.add_strategy(Box::new(BordaCountFusion::new()));
        
        // Set default weights
        fusion.set_weight("lex".to_string(), 0.3);
        fusion.set_weight("symbols".to_string(), 0.4);
        fusion.set_weight("semantic".to_string(), 0.3);
        
        fusion
    }

    /// Add fusion strategy
    pub fn add_strategy(&mut self, strategy: Box<dyn FusionStrategy + Send + Sync>) {
        self.strategies.push(strategy);
    }

    /// Set weight for search system
    pub fn set_weight(&mut self, system: String, weight: f64) {
        self.weights.insert(system, weight);
    }

    /// Fuse results from multiple systems
    pub async fn fuse_results(&self, system_results: &[SystemResults]) -> Result<Vec<FusedResult>> {
        if system_results.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_fused_results = Vec::new();

        // Apply each fusion strategy
        for strategy in &self.strategies {
            let fused = strategy.fuse(system_results).await?;
            let confidence = strategy.confidence(system_results);
            
            for result in fused {
                let contributing_systems = system_results
                    .iter()
                    .filter(|sys| sys.results.iter().any(|r| r.file_path == result.file_path))
                    .map(|sys| sys.system_name.clone())
                    .collect();
                    
                all_fused_results.push(FusedResult {
                    fusion_score: result.score,
                    contributing_systems,
                    fusion_strategy: strategy.name().to_string(),
                    confidence,
                    result,
                });
            }
        }

        // Select best fusion results (could use ensemble methods here)
        Ok(self.select_best_fusion(all_fused_results))
    }

    /// Select the best fusion results using ensemble approach
    fn select_best_fusion(&self, fused_results: Vec<FusedResult>) -> Vec<FusedResult> {
        // Group by file path and select highest scoring fusion for each
        let mut best_results: HashMap<String, FusedResult> = HashMap::new();
        
        for result in fused_results {
            let key = format!("{}:{}", result.result.file_path, result.result.line_number);
            
            if let Some(existing) = best_results.get(&key) {
                if result.fusion_score > existing.fusion_score {
                    best_results.insert(key, result);
                }
            } else {
                best_results.insert(key, result);
            }
        }

        let mut final_results: Vec<FusedResult> = best_results.into_values().collect();
        final_results.sort_by(|a, b| b.fusion_score.partial_cmp(&a.fusion_score).unwrap());
        
        // Limit to top results
        final_results.truncate(50);
        
        final_results
    }
}

/// CombSum fusion strategy (sum of normalized scores)
pub struct CombSumStrategy {
    name: String,
}

impl CombSumStrategy {
    pub fn new() -> Self {
        Self {
            name: "combsum".to_string(),
        }
    }
}

#[async_trait::async_trait]
impl FusionStrategy for CombSumStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    async fn fuse(&self, results: &[SystemResults]) -> Result<Vec<SearchResult>> {
        let mut score_map: HashMap<String, (SearchResult, f64)> = HashMap::new();
        
        // Normalize scores per system and combine
        for system_result in results {
            let max_score = system_result.results
                .iter()
                .map(|r| r.score)
                .fold(0.0, f64::max);
                
            if max_score > 0.0 {
                for result in &system_result.results {
                    let normalized_score = result.score / max_score;
                    let key = format!("{}:{}", result.file_path, result.line_number);
                    
                    if let Some((_, current_score)) = score_map.get(&key) {
                        score_map.insert(key, (result.clone(), current_score + normalized_score));
                    } else {
                        score_map.insert(key, (result.clone(), normalized_score));
                    }
                }
            }
        }

        let mut fused_results: Vec<SearchResult> = score_map
            .into_values()
            .map(|(mut result, score)| {
                result.score = score;
                result
            })
            .collect();
            
        fused_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        Ok(fused_results)
    }

    fn confidence(&self, results: &[SystemResults]) -> f64 {
        if results.is_empty() {
            return 0.0;
        }
        
        // Higher confidence when multiple systems agree
        let avg_confidence: f64 = results.iter().map(|r| r.confidence).sum::<f64>() / results.len() as f64;
        let agreement_bonus = if results.len() > 1 { 0.1 } else { 0.0 };
        
        (avg_confidence + agreement_bonus).min(1.0)
    }
}

/// CombMNZ fusion strategy (CombSum * number of non-zero systems)
pub struct CombMnzStrategy {
    name: String,
}

impl CombMnzStrategy {
    pub fn new() -> Self {
        Self {
            name: "combmnz".to_string(),
        }
    }
}

#[async_trait::async_trait]
impl FusionStrategy for CombMnzStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    async fn fuse(&self, results: &[SystemResults]) -> Result<Vec<SearchResult>> {
        let mut score_map: HashMap<String, (SearchResult, f64, usize)> = HashMap::new();
        
        // Track sum of scores and count of contributing systems
        for system_result in results {
            let max_score = system_result.results
                .iter()
                .map(|r| r.score)
                .fold(0.0, f64::max);
                
            if max_score > 0.0 {
                for result in &system_result.results {
                    let normalized_score = result.score / max_score;
                    let key = format!("{}:{}", result.file_path, result.line_number);
                    
                    if let Some((_, current_score, count)) = score_map.get(&key) {
                        score_map.insert(key, (result.clone(), current_score + normalized_score, count + 1));
                    } else {
                        score_map.insert(key, (result.clone(), normalized_score, 1));
                    }
                }
            }
        }

        let mut fused_results: Vec<SearchResult> = score_map
            .into_values()
            .map(|(mut result, sum_score, count)| {
                result.score = sum_score * count as f64; // CombMNZ formula
                result
            })
            .collect();
            
        fused_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        Ok(fused_results)
    }

    fn confidence(&self, results: &[SystemResults]) -> f64 {
        // CombMNZ gives higher confidence to results agreed upon by multiple systems
        if results.is_empty() {
            return 0.0;
        }
        
        let base_confidence: f64 = results.iter().map(|r| r.confidence).sum::<f64>() / results.len() as f64;
        let system_bonus = (results.len() as f64 - 1.0) * 0.1; // Bonus for multiple systems
        
        (base_confidence + system_bonus).min(1.0)
    }
}

/// Rank-based fusion using reciprocal rank
pub struct RankBasedFusion {
    name: String,
}

impl RankBasedFusion {
    pub fn new() -> Self {
        Self {
            name: "rank_fusion".to_string(),
        }
    }
}

#[async_trait::async_trait]
impl FusionStrategy for RankBasedFusion {
    fn name(&self) -> &str {
        &self.name
    }

    async fn fuse(&self, results: &[SystemResults]) -> Result<Vec<SearchResult>> {
        let mut score_map: HashMap<String, (SearchResult, f64)> = HashMap::new();
        
        for system_result in results {
            for (rank, result) in system_result.results.iter().enumerate() {
                let reciprocal_rank = 1.0 / (rank + 1) as f64;
                let key = format!("{}:{}", result.file_path, result.line_number);
                
                if let Some((_, current_score)) = score_map.get(&key) {
                    score_map.insert(key, (result.clone(), current_score + reciprocal_rank));
                } else {
                    score_map.insert(key, (result.clone(), reciprocal_rank));
                }
            }
        }

        let mut fused_results: Vec<SearchResult> = score_map
            .into_values()
            .map(|(mut result, score)| {
                result.score = score;
                result
            })
            .collect();
            
        fused_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        Ok(fused_results)
    }

    fn confidence(&self, _results: &[SystemResults]) -> f64 {
        0.8 // Rank-based fusion is generally reliable
    }
}

/// Borda count fusion strategy
pub struct BordaCountFusion {
    name: String,
}

impl BordaCountFusion {
    pub fn new() -> Self {
        Self {
            name: "borda_count".to_string(),
        }
    }
}

#[async_trait::async_trait]
impl FusionStrategy for BordaCountFusion {
    fn name(&self) -> &str {
        &self.name
    }

    async fn fuse(&self, results: &[SystemResults]) -> Result<Vec<SearchResult>> {
        let mut score_map: HashMap<String, (SearchResult, f64)> = HashMap::new();
        
        for system_result in results {
            let num_results = system_result.results.len();
            
            for (rank, result) in system_result.results.iter().enumerate() {
                let borda_score = (num_results - rank) as f64; // Higher score for higher rank
                let key = format!("{}:{}", result.file_path, result.line_number);
                
                if let Some((_, current_score)) = score_map.get(&key) {
                    score_map.insert(key, (result.clone(), current_score + borda_score));
                } else {
                    score_map.insert(key, (result.clone(), borda_score));
                }
            }
        }

        let mut fused_results: Vec<SearchResult> = score_map
            .into_values()
            .map(|(mut result, score)| {
                result.score = score;
                result
            })
            .collect();
            
        fused_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        Ok(fused_results)
    }

    fn confidence(&self, _results: &[SystemResults]) -> f64 {
        0.75 // Borda count is moderately reliable
    }
}

impl Default for ResultFusion {
    fn default() -> Self {
        Self::new()
    }
}