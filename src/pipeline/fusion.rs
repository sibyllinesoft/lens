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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::SearchResult;

    fn create_test_search_result(file_path: &str, line_number: u32, score: f64) -> SearchResult {
        SearchResult {
            file_path: file_path.to_string(),
            line_number,
            column: 1,
            content: format!("test content for {}", file_path),
            score,
            result_type: crate::search::SearchResultType::TextMatch,
            language: Some("rust".to_string()),
            context_lines: Some(vec![]),
            lsp_metadata: None,
        }
    }

    fn create_test_system_results(system_name: &str, results: Vec<SearchResult>) -> SystemResults {
        SystemResults {
            system_name: system_name.to_string(),
            results,
            latency_ms: 50.0,
            confidence: 0.8,
        }
    }

    #[test]
    fn test_result_fusion_creation() {
        let fusion = ResultFusion::new();
        assert_eq!(fusion.strategies.len(), 4);
        assert_eq!(fusion.weights.len(), 3);
        assert_eq!(fusion.weights.get("lex"), Some(&0.3));
        assert_eq!(fusion.weights.get("symbols"), Some(&0.4));
        assert_eq!(fusion.weights.get("semantic"), Some(&0.3));
    }

    #[test]
    fn test_result_fusion_add_strategy() {
        let mut fusion = ResultFusion::new();
        let initial_count = fusion.strategies.len();
        
        fusion.add_strategy(Box::new(CombSumStrategy::new()));
        assert_eq!(fusion.strategies.len(), initial_count + 1);
    }

    #[test]
    fn test_result_fusion_set_weight() {
        let mut fusion = ResultFusion::new();
        fusion.set_weight("new_system".to_string(), 0.5);
        assert_eq!(fusion.weights.get("new_system"), Some(&0.5));
    }

    #[tokio::test]
    async fn test_fuse_empty_results() {
        let fusion = ResultFusion::new();
        let results = fusion.fuse_results(&[]).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_fuse_single_system() {
        let fusion = ResultFusion::new();
        let search_results = vec![
            create_test_search_result("file1.rs", 10, 0.9),
            create_test_search_result("file2.rs", 20, 0.7),
        ];
        let system_results = vec![create_test_system_results("lex", search_results)];
        
        let fused = fusion.fuse_results(&system_results).await.unwrap();
        assert!(!fused.is_empty());
        
        // Check that fusion scores are assigned
        for result in &fused {
            assert!(result.fusion_score > 0.0);
            assert_eq!(result.contributing_systems.len(), 1);
            assert_eq!(result.contributing_systems[0], "lex");
        }
    }

    #[tokio::test]
    async fn test_fuse_multiple_systems() {
        let fusion = ResultFusion::new();
        
        let lex_results = vec![
            create_test_search_result("file1.rs", 10, 0.9),
            create_test_search_result("file2.rs", 20, 0.8),
        ];
        let symbols_results = vec![
            create_test_search_result("file1.rs", 10, 0.8), // Same file/line as lex
            create_test_search_result("file3.rs", 30, 0.7),
        ];
        
        let system_results = vec![
            create_test_system_results("lex", lex_results),
            create_test_system_results("symbols", symbols_results),
        ];
        
        let fused = fusion.fuse_results(&system_results).await.unwrap();
        assert!(!fused.is_empty());
        
        // Check for results from both systems
        let has_overlapping_result = fused.iter().any(|r| 
            r.result.file_path == "file1.rs" && r.result.line_number == 10
        );
        assert!(has_overlapping_result);
    }

    #[test]
    fn test_select_best_fusion() {
        let fusion = ResultFusion::new();
        let search_result = create_test_search_result("file1.rs", 10, 0.9);
        
        let fused_results = vec![
            FusedResult {
                result: search_result.clone(),
                fusion_score: 0.8,
                contributing_systems: vec!["lex".to_string()],
                fusion_strategy: "combsum".to_string(),
                confidence: 0.8,
            },
            FusedResult {
                result: search_result.clone(),
                fusion_score: 0.9, // Higher score
                contributing_systems: vec!["symbols".to_string()],
                fusion_strategy: "combmnz".to_string(),
                confidence: 0.9,
            },
        ];
        
        let best = fusion.select_best_fusion(fused_results);
        assert_eq!(best.len(), 1);
        assert_eq!(best[0].fusion_score, 0.9);
        assert_eq!(best[0].fusion_strategy, "combmnz");
    }

    #[test]
    fn test_combsum_strategy() {
        let strategy = CombSumStrategy::new();
        assert_eq!(strategy.name(), "combsum");
    }

    #[tokio::test]
    async fn test_combsum_fuse() {
        let strategy = CombSumStrategy::new();
        let system_results = vec![
            create_test_system_results("system1", vec![
                create_test_search_result("file1.rs", 10, 1.0),
                create_test_search_result("file2.rs", 20, 0.8),
            ]),
            create_test_system_results("system2", vec![
                create_test_search_result("file1.rs", 10, 0.6), // Same file, should combine
                create_test_search_result("file3.rs", 30, 1.0),
            ]),
        ];
        
        let fused = strategy.fuse(&system_results).await.unwrap();
        assert!(!fused.is_empty());
        
        // Check that scores are combined for overlapping results
        let file1_result = fused.iter().find(|r| r.file_path == "file1.rs" && r.line_number == 10);
        assert!(file1_result.is_some());
        assert!(file1_result.unwrap().score > 1.0); // Should be sum of normalized scores
    }

    #[test]
    fn test_combsum_confidence() {
        let strategy = CombSumStrategy::new();
        let system_results = vec![
            SystemResults {
                system_name: "system1".to_string(),
                results: vec![],
                latency_ms: 50.0,
                confidence: 0.8,
            },
            SystemResults {
                system_name: "system2".to_string(),
                results: vec![],
                latency_ms: 60.0,
                confidence: 0.9,
            },
        ];
        
        let confidence = strategy.confidence(&system_results);
        assert!(confidence > 0.8); // Should include agreement bonus
        assert!(confidence <= 1.0);
        
        // Test empty results
        assert_eq!(strategy.confidence(&[]), 0.0);
    }

    #[tokio::test]
    async fn test_combmnz_fuse() {
        let strategy = CombMnzStrategy::new();
        assert_eq!(strategy.name(), "combmnz");
        
        let system_results = vec![
            create_test_system_results("system1", vec![
                create_test_search_result("file1.rs", 10, 1.0),
            ]),
            create_test_system_results("system2", vec![
                create_test_search_result("file1.rs", 10, 0.8), // Same result in both systems
            ]),
        ];
        
        let fused = strategy.fuse(&system_results).await.unwrap();
        assert!(!fused.is_empty());
        
        // CombMNZ should boost scores for results found in multiple systems
        let result = &fused[0];
        assert!(result.score > 1.0); // Score boosted by multiple systems
    }

    #[test]
    fn test_combmnz_confidence() {
        let strategy = CombMnzStrategy::new();
        let system_results = vec![
            SystemResults {
                system_name: "system1".to_string(),
                results: vec![],
                latency_ms: 50.0,
                confidence: 0.8,
            },
            SystemResults {
                system_name: "system2".to_string(),
                results: vec![],
                latency_ms: 60.0,
                confidence: 0.8,
            },
        ];
        
        let confidence = strategy.confidence(&system_results);
        assert!(confidence > 0.8); // Should include system bonus
        assert_eq!(strategy.confidence(&[]), 0.0);
    }

    #[tokio::test]
    async fn test_rank_based_fusion() {
        let strategy = RankBasedFusion::new();
        assert_eq!(strategy.name(), "rank_fusion");
        assert_eq!(strategy.confidence(&[]), 0.8);
        
        let system_results = vec![
            create_test_system_results("system1", vec![
                create_test_search_result("file1.rs", 10, 1.0), // Rank 1
                create_test_search_result("file2.rs", 20, 0.9), // Rank 2
            ]),
        ];
        
        let fused = strategy.fuse(&system_results).await.unwrap();
        assert_eq!(fused.len(), 2);
        
        // Higher ranked results should have higher reciprocal rank scores
        assert!(fused[0].score >= fused[1].score);
        assert_eq!(fused[0].score, 1.0); // 1/(0+1)
        assert_eq!(fused[1].score, 0.5); // 1/(1+1)
    }

    #[tokio::test]
    async fn test_borda_count_fusion() {
        let strategy = BordaCountFusion::new();
        assert_eq!(strategy.name(), "borda_count");
        assert_eq!(strategy.confidence(&[]), 0.75);
        
        let system_results = vec![
            create_test_system_results("system1", vec![
                create_test_search_result("file1.rs", 10, 1.0), // Rank 1: score = 2
                create_test_search_result("file2.rs", 20, 0.9), // Rank 2: score = 1
            ]),
        ];
        
        let fused = strategy.fuse(&system_results).await.unwrap();
        assert_eq!(fused.len(), 2);
        
        // Borda count: (num_results - rank)
        assert_eq!(fused[0].score, 2.0); // 2 - 0
        assert_eq!(fused[1].score, 1.0); // 2 - 1
    }

    #[test]
    fn test_system_results_creation() {
        let results = vec![create_test_search_result("test.rs", 1, 0.9)];
        let system_results = create_test_system_results("test_system", results);
        
        assert_eq!(system_results.system_name, "test_system");
        assert_eq!(system_results.results.len(), 1);
        assert_eq!(system_results.latency_ms, 50.0);
        assert_eq!(system_results.confidence, 0.8);
    }

    #[test]
    fn test_fused_result_creation() {
        let search_result = create_test_search_result("test.rs", 1, 0.9);
        let fused_result = FusedResult {
            result: search_result,
            fusion_score: 1.5,
            contributing_systems: vec!["system1".to_string(), "system2".to_string()],
            fusion_strategy: "combsum".to_string(),
            confidence: 0.9,
        };
        
        assert_eq!(fused_result.fusion_score, 1.5);
        assert_eq!(fused_result.contributing_systems.len(), 2);
        assert_eq!(fused_result.fusion_strategy, "combsum");
        assert_eq!(fused_result.confidence, 0.9);
    }
}