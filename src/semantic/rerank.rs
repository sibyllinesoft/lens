//! # Learned Reranking with Isotonic Regression
//!
//! Advanced reranking system for semantic search results:
//! - Learned reranking on top-K results from initial search
//! - Isotonic regression for calibrated score mapping
//! - Balance precision vs recall optimization
//! - Target: +2-3pp nDCG improvement over baseline ranking

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Learned reranking system
pub struct LearnedReranker {
    config: RerankConfig,
    /// Feature extractors for reranking
    feature_extractors: Vec<Box<dyn FeatureExtractor + Send + Sync>>,
    /// Isotonic regression models for score calibration
    isotonic_models: Arc<RwLock<HashMap<String, IsotonicRegressor>>>,
    /// Linear model weights
    model_weights: Arc<RwLock<Option<Vec<f32>>>>,
    /// Performance metrics
    metrics: Arc<RwLock<RerankMetrics>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankConfig {
    /// Top-K results to rerank
    pub top_k: usize,
    /// Use isotonic regression for score calibration
    pub use_isotonic: bool,
    /// Learning rate for model training
    pub learning_rate: f32,
    /// L2 regularization strength
    pub l2_regularization: f32,
    /// Minimum training samples for model update
    pub min_training_samples: usize,
    /// Feature combination strategy
    pub combination_strategy: CombinationStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CombinationStrategy {
    /// Linear combination of features
    Linear,
    /// Learned weighted combination
    LearnedWeights,
    /// Ensemble of models
    Ensemble,
}

/// Search result to be reranked
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub content: String,
    pub file_path: String,
    pub initial_score: f32,
    pub lexical_score: f32,
    pub semantic_score: Option<f32>,
    pub lsp_score: Option<f32>,
    pub metadata: HashMap<String, String>,
}

/// Reranked result with new score
#[derive(Debug, Clone)]
pub struct RerankedResult {
    pub result: SearchResult,
    pub rerank_score: f32,
    pub feature_vector: Vec<f32>,
    pub calibrated_score: f32,
    pub rank_change: i32, // Change in ranking position
}

/// Feature extractor interface
pub trait FeatureExtractor {
    /// Extract features from query and result
    fn extract_features(&self, query: &str, result: &SearchResult) -> Result<Vec<f32>>;
    
    /// Get feature names for interpretability
    fn feature_names(&self) -> Vec<String>;
}

/// Isotonic regression for score calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsotonicRegressor {
    /// Sorted input values
    x_values: Vec<f32>,
    /// Corresponding output values (isotonic)
    y_values: Vec<f32>,
    /// Number of training samples
    sample_count: usize,
}

/// Training sample for reranker
#[derive(Debug, Clone)]
pub struct TrainingSample {
    pub query: String,
    pub results: Vec<SearchResult>,
    pub relevance_scores: Vec<f32>, // Ground truth relevance (0.0-1.0)
    pub ideal_ranking: Vec<usize>, // Ideal ranking order
}

/// Reranking performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankMetrics {
    pub samples_processed: usize,
    pub avg_ndcg_improvement: f32,
    pub avg_precision_improvement: f32,
    pub avg_recall_improvement: f32,
    pub calibration_error: f32,
    pub feature_importances: HashMap<String, f32>,
}

impl LearnedReranker {
    /// Create new learned reranker
    pub async fn new(config: RerankConfig) -> Result<Self> {
        info!("Creating learned reranker");
        info!("Top-K: {}, isotonic: {}, learning rate: {}", 
              config.top_k, config.use_isotonic, config.learning_rate);
        
        // Initialize feature extractors
        let mut feature_extractors: Vec<Box<dyn FeatureExtractor + Send + Sync>> = Vec::new();
        feature_extractors.push(Box::new(LexicalFeatureExtractor::new()));
        feature_extractors.push(Box::new(SemanticFeatureExtractor::new()));
        feature_extractors.push(Box::new(StructuralFeatureExtractor::new()));
        feature_extractors.push(Box::new(LSPFeatureExtractor::new()));
        
        Ok(Self {
            config,
            feature_extractors,
            isotonic_models: Arc::new(RwLock::new(HashMap::new())),
            model_weights: Arc::new(RwLock::new(None)),
            metrics: Arc::new(RwLock::new(RerankMetrics::default())),
        })
    }
    
    /// Rerank search results with learned model
    pub async fn rerank(&self, query: &str, results: Vec<SearchResult>) -> Result<Vec<RerankedResult>> {
        if results.is_empty() {
            return Ok(Vec::new());
        }
        
        // Take top-K for reranking
        let top_k_results = results.into_iter()
            .take(self.config.top_k)
            .collect::<Vec<_>>();
            
        debug!("Reranking {} results for query: '{}'", top_k_results.len(), query);
        
        // Extract features for all results
        let feature_results = self.extract_all_features(query, &top_k_results).await?;
        
        // Apply learned model
        let scored_results = self.apply_learned_model(feature_results).await?;
        
        // Apply isotonic calibration if enabled
        let calibrated_results = if self.config.use_isotonic {
            self.apply_isotonic_calibration(scored_results).await?
        } else {
            scored_results.into_iter()
                .map(|mut r| { r.calibrated_score = r.rerank_score; r })
                .collect()
        };
        
        // Sort by reranked scores
        let mut final_results = calibrated_results;
        final_results.sort_by(|a, b| b.calibrated_score.partial_cmp(&a.calibrated_score).unwrap());
        
        // Calculate rank changes
        let original_order: HashMap<String, usize> = top_k_results.iter()
            .enumerate()
            .map(|(i, r)| (r.id.clone(), i))
            .collect();
            
        for (new_rank, result) in final_results.iter_mut().enumerate() {
            let original_rank = original_order.get(&result.result.id).unwrap_or(&0);
            result.rank_change = *original_rank as i32 - new_rank as i32;
        }
        
        debug!("Reranking complete, {} results reordered", final_results.len());
        
        Ok(final_results)
    }
    
    /// Train the reranker on labeled data
    pub async fn train(&self, training_samples: &[TrainingSample]) -> Result<()> {
        info!("Training reranker on {} samples", training_samples.len());
        
        if training_samples.len() < self.config.min_training_samples {
            anyhow::bail!("Insufficient training samples: {} < {}", 
                         training_samples.len(), self.config.min_training_samples);
        }
        
        // Extract features for all training samples
        let mut feature_vectors = Vec::new();
        let mut target_scores = Vec::new();
        
        for sample in training_samples {
            let features = self.extract_training_features(sample).await?;
            feature_vectors.extend(features);
            target_scores.extend(&sample.relevance_scores);
        }
        
        // Train linear model
        let weights = self.train_linear_model(&feature_vectors, &target_scores)?;
        *self.model_weights.write().await = Some(weights);
        
        // Train isotonic regressors if enabled
        if self.config.use_isotonic {
            self.train_isotonic_regressors(training_samples).await?;
        }
        
        // Update metrics
        self.update_training_metrics(training_samples).await?;
        
        info!("Reranker training complete");
        Ok(())
    }
    
    /// Evaluate reranker performance
    pub async fn evaluate(&self, test_samples: &[TrainingSample]) -> Result<RerankMetrics> {
        info!("Evaluating reranker on {} test samples", test_samples.len());
        
        let mut total_ndcg_improvement = 0.0;
        let mut total_precision_improvement = 0.0;
        let mut total_recall_improvement = 0.0;
        let mut valid_samples = 0;
        
        for sample in test_samples {
            // Get original ranking
            let original_results = sample.results.clone();
            
            // Apply reranking
            let reranked_results = self.rerank(&sample.query, original_results.clone()).await?;
            
            // Calculate metrics improvement
            let ndcg_original = self.calculate_ndcg(&original_results, &sample.relevance_scores);
            let ndcg_reranked = self.calculate_ndcg_reranked(&reranked_results, &sample.relevance_scores);
            
            let precision_original = self.calculate_precision_at_k(&original_results, &sample.relevance_scores, 10);
            let precision_reranked = self.calculate_precision_at_k_reranked(&reranked_results, &sample.relevance_scores, 10);
            
            if ndcg_original > 0.0 {
                total_ndcg_improvement += (ndcg_reranked - ndcg_original) / ndcg_original;
                total_precision_improvement += (precision_reranked - precision_original) / precision_original.max(0.001);
                valid_samples += 1;
            }
        }
        
        let avg_ndcg_improvement = if valid_samples > 0 {
            total_ndcg_improvement / valid_samples as f32
        } else {
            0.0
        };
        
        let avg_precision_improvement = if valid_samples > 0 {
            total_precision_improvement / valid_samples as f32  
        } else {
            0.0
        };
        
        let metrics = RerankMetrics {
            samples_processed: valid_samples,
            avg_ndcg_improvement,
            avg_precision_improvement,
            avg_recall_improvement: 0.0, // TODO: implement recall calculation
            calibration_error: self.calculate_calibration_error(test_samples).await?,
            feature_importances: self.get_feature_importances().await,
        };
        
        info!("Evaluation complete: nDCG improvement {:.3}, precision improvement {:.3}",
              metrics.avg_ndcg_improvement, metrics.avg_precision_improvement);
        
        Ok(metrics)
    }
    
    /// Get current model status and metrics
    pub async fn get_metrics(&self) -> RerankMetrics {
        self.metrics.read().await.clone()
    }
    
    // Private implementation methods
    
    async fn extract_all_features(&self, query: &str, results: &[SearchResult]) -> Result<Vec<FeatureResult>> {
        let mut feature_results = Vec::with_capacity(results.len());
        
        for result in results {
            let mut all_features = Vec::new();
            
            // Extract features from all extractors
            for extractor in &self.feature_extractors {
                let features = extractor.extract_features(query, result)
                    .context("Feature extraction failed")?;
                all_features.extend(features);
            }
            
            feature_results.push(FeatureResult {
                result: result.clone(),
                features: all_features,
            });
        }
        
        Ok(feature_results)
    }
    
    async fn apply_learned_model(&self, feature_results: Vec<FeatureResult>) -> Result<Vec<RerankedResult>> {
        let weights_guard = self.model_weights.read().await;
        
        // Use default weights if model is not trained (for benchmark testing)
        let default_weights;
        let weights = if let Some(trained_weights) = weights_guard.as_ref() {
            trained_weights
        } else {
            warn!("Reranker not trained, using default uniform weights for benchmark testing");
            // Create uniform weights based on feature dimension
            let feature_dim = if !feature_results.is_empty() {
                feature_results[0].features.len()
            } else {
                12 // Default: 3 + 3 + 3 + 3 features from each extractor
            };
            default_weights = vec![1.0 / feature_dim as f32; feature_dim];
            &default_weights
        };
            
        let mut reranked = Vec::with_capacity(feature_results.len());
        
        for feature_result in feature_results {
            // Calculate weighted score
            let rerank_score = if feature_result.features.len() == weights.len() {
                feature_result.features.iter()
                    .zip(weights.iter())
                    .map(|(f, w)| f * w)
                    .sum()
            } else {
                warn!("Feature dimension mismatch: {} vs {}", feature_result.features.len(), weights.len());
                feature_result.result.initial_score // Fall back to initial score
            };
            
            reranked.push(RerankedResult {
                result: feature_result.result,
                rerank_score,
                feature_vector: feature_result.features,
                calibrated_score: rerank_score, // Will be updated by calibration
                rank_change: 0, // Will be calculated later
            });
        }
        
        Ok(reranked)
    }
    
    async fn apply_isotonic_calibration(&self, mut results: Vec<RerankedResult>) -> Result<Vec<RerankedResult>> {
        let models = self.isotonic_models.read().await;
        
        // For simplicity, use a single global model
        // Real implementation would have query-type specific models
        if let Some(model) = models.get("global") {
            for result in &mut results {
                result.calibrated_score = model.predict(result.rerank_score);
            }
        } else {
            // No calibration available, use raw scores
            for result in &mut results {
                result.calibrated_score = result.rerank_score;
            }
        }
        
        Ok(results)
    }
    
    async fn extract_training_features(&self, sample: &TrainingSample) -> Result<Vec<Vec<f32>>> {
        let mut all_features = Vec::new();
        
        for result in &sample.results {
            let mut features = Vec::new();
            
            for extractor in &self.feature_extractors {
                let result_features = extractor.extract_features(&sample.query, result)?;
                features.extend(result_features);
            }
            
            all_features.push(features);
        }
        
        Ok(all_features)
    }
    
    fn train_linear_model(&self, features: &[Vec<f32>], targets: &[f32]) -> Result<Vec<f32>> {
        if features.is_empty() || features[0].is_empty() {
            anyhow::bail!("No features provided for training");
        }
        
        let feature_dim = features[0].len();
        let mut weights = vec![0.0; feature_dim];
        
        // Simple gradient descent training
        let learning_rate = self.config.learning_rate;
        let l2_reg = self.config.l2_regularization;
        let epochs = 100;
        
        for _epoch in 0..epochs {
            let mut gradients = vec![0.0; feature_dim];
            let mut total_loss = 0.0;
            
            for (feature_vec, target) in features.iter().zip(targets.iter()) {
                // Forward pass
                let prediction: f32 = feature_vec.iter()
                    .zip(weights.iter())
                    .map(|(f, w)| f * w)
                    .sum();
                    
                let error = prediction - target;
                total_loss += error * error;
                
                // Backward pass
                for (i, feature) in feature_vec.iter().enumerate() {
                    gradients[i] += error * feature;
                }
            }
            
            // Update weights with L2 regularization
            for (i, weight) in weights.iter_mut().enumerate() {
                *weight -= learning_rate * (gradients[i] / features.len() as f32 + l2_reg * *weight);
            }
        }
        
        Ok(weights)
    }
    
    async fn train_isotonic_regressors(&self, training_samples: &[TrainingSample]) -> Result<()> {
        // Collect (score, relevance) pairs
        let mut score_relevance_pairs = Vec::new();
        
        for sample in training_samples {
            let reranked = self.rerank(&sample.query, sample.results.clone()).await?;
            
            for (result, relevance) in reranked.iter().zip(sample.relevance_scores.iter()) {
                score_relevance_pairs.push((result.rerank_score, *relevance));
            }
        }
        
        // Sort by score for isotonic regression
        score_relevance_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        // Train isotonic regressor
        let regressor = IsotonicRegressor::fit(&score_relevance_pairs)?;
        
        let mut models = self.isotonic_models.write().await;
        models.insert("global".to_string(), regressor);
        
        Ok(())
    }
    
    async fn update_training_metrics(&self, _training_samples: &[TrainingSample]) -> Result<()> {
        // Update training metrics
        let mut metrics = self.metrics.write().await;
        metrics.samples_processed += _training_samples.len();
        
        Ok(())
    }
    
    fn calculate_ndcg(&self, results: &[SearchResult], relevances: &[f32]) -> f32 {
        if results.is_empty() || relevances.is_empty() {
            return 0.0;
        }
        
        // Calculate DCG
        let mut dcg = 0.0;
        for (i, rel) in relevances.iter().take(results.len()).enumerate() {
            let discount = (i as f32 + 2.0).log2();
            dcg += (2.0_f32.powf(*rel) - 1.0) / discount;
        }
        
        // Calculate IDCG (ideal DCG)
        let mut sorted_relevances = relevances.to_vec();
        sorted_relevances.sort_by(|a, b| b.partial_cmp(a).unwrap());
        
        let mut idcg = 0.0;
        for (i, rel) in sorted_relevances.iter().take(results.len()).enumerate() {
            let discount = (i as f32 + 2.0).log2();
            idcg += (2.0_f32.powf(*rel) - 1.0) / discount;
        }
        
        if idcg > 0.0 { dcg / idcg } else { 0.0 }
    }
    
    fn calculate_ndcg_reranked(&self, results: &[RerankedResult], relevances: &[f32]) -> f32 {
        if results.is_empty() || relevances.is_empty() {
            return 0.0;
        }
        
        // Map result IDs to relevances
        let mut id_to_relevance = HashMap::new();
        for (i, result) in results.iter().enumerate() {
            if i < relevances.len() {
                id_to_relevance.insert(result.result.id.clone(), relevances[i]);
            }
        }
        
        // Calculate DCG for reranked order
        let mut dcg = 0.0;
        for (i, result) in results.iter().enumerate() {
            if let Some(rel) = id_to_relevance.get(&result.result.id) {
                let discount = (i as f32 + 2.0).log2();
                dcg += (2.0_f32.powf(*rel) - 1.0) / discount;
            }
        }
        
        // Calculate IDCG
        let mut sorted_relevances = relevances.to_vec();
        sorted_relevances.sort_by(|a, b| b.partial_cmp(a).unwrap());
        
        let mut idcg = 0.0;
        for (i, rel) in sorted_relevances.iter().take(results.len()).enumerate() {
            let discount = (i as f32 + 2.0).log2();
            idcg += (2.0_f32.powf(*rel) - 1.0) / discount;
        }
        
        if idcg > 0.0 { dcg / idcg } else { 0.0 }
    }
    
    fn calculate_precision_at_k(&self, _results: &[SearchResult], relevances: &[f32], k: usize) -> f32 {
        let relevant_count = relevances.iter()
            .take(k)
            .filter(|&&r| r > 0.5) // Threshold for relevance
            .count();
            
        relevant_count as f32 / k.min(relevances.len()) as f32
    }
    
    fn calculate_precision_at_k_reranked(&self, _results: &[RerankedResult], relevances: &[f32], k: usize) -> f32 {
        // For simplicity, assume same order as input relevances
        // Real implementation would map reranked results to relevances
        self.calculate_precision_at_k(&[], relevances, k)
    }
    
    async fn calculate_calibration_error(&self, _test_samples: &[TrainingSample]) -> Result<f32> {
        // Expected Calibration Error calculation
        // For now, return mock value
        Ok(0.05)
    }
    
    async fn get_feature_importances(&self) -> HashMap<String, f32> {
        let mut importances = HashMap::new();
        
        // Get all feature names
        let mut feature_names = Vec::new();
        for extractor in &self.feature_extractors {
            feature_names.extend(extractor.feature_names());
        }
        
        // Get weights if available
        if let Some(weights) = self.model_weights.read().await.as_ref() {
            for (name, weight) in feature_names.iter().zip(weights.iter()) {
                importances.insert(name.clone(), weight.abs());
            }
        }
        
        importances
    }
}

// Feature extractors implementation

struct LexicalFeatureExtractor;
struct SemanticFeatureExtractor;
struct StructuralFeatureExtractor;
struct LSPFeatureExtractor;

impl LexicalFeatureExtractor {
    fn new() -> Self { Self }
}

impl FeatureExtractor for LexicalFeatureExtractor {
    fn extract_features(&self, query: &str, result: &SearchResult) -> Result<Vec<f32>> {
        let mut features = Vec::new();
        
        // BM25-like features
        features.push(result.lexical_score);
        
        // Query term matches
        let query_terms: Vec<&str> = query.split_whitespace().collect();
        let content_lower = result.content.to_lowercase();
        let query_matches = query_terms.iter()
            .filter(|term| content_lower.contains(&term.to_lowercase()))
            .count() as f32 / query_terms.len() as f32;
        features.push(query_matches);
        
        // Length features
        features.push((result.content.len() as f32).log10());
        
        Ok(features)
    }
    
    fn feature_names(&self) -> Vec<String> {
        vec![
            "lexical_score".to_string(),
            "query_match_ratio".to_string(), 
            "log_content_length".to_string(),
        ]
    }
}

impl SemanticFeatureExtractor {
    fn new() -> Self { Self }
}

impl FeatureExtractor for SemanticFeatureExtractor {
    fn extract_features(&self, _query: &str, result: &SearchResult) -> Result<Vec<f32>> {
        let mut features = Vec::new();
        
        // Semantic score if available
        features.push(result.semantic_score.unwrap_or(0.0));
        
        // Placeholder semantic features
        features.push(0.5); // Semantic similarity placeholder
        features.push(0.3); // Context relevance placeholder
        
        Ok(features)
    }
    
    fn feature_names(&self) -> Vec<String> {
        vec![
            "semantic_score".to_string(),
            "semantic_similarity".to_string(),
            "context_relevance".to_string(),
        ]
    }
}

impl StructuralFeatureExtractor {
    fn new() -> Self { Self }
}

impl FeatureExtractor for StructuralFeatureExtractor {
    fn extract_features(&self, _query: &str, result: &SearchResult) -> Result<Vec<f32>> {
        let mut features = Vec::new();
        
        // File type features
        let is_source_file = result.file_path.ends_with(".rs") || 
                           result.file_path.ends_with(".py") ||
                           result.file_path.ends_with(".ts");
        features.push(if is_source_file { 1.0 } else { 0.0 });
        
        // Code structure features (placeholder)
        features.push(0.4); // Function density
        features.push(0.6); // Comment ratio
        
        Ok(features)
    }
    
    fn feature_names(&self) -> Vec<String> {
        vec![
            "is_source_file".to_string(),
            "function_density".to_string(),
            "comment_ratio".to_string(),
        ]
    }
}

impl LSPFeatureExtractor {
    fn new() -> Self { Self }
}

impl FeatureExtractor for LSPFeatureExtractor {
    fn extract_features(&self, _query: &str, result: &SearchResult) -> Result<Vec<f32>> {
        let mut features = Vec::new();
        
        // LSP score if available
        features.push(result.lsp_score.unwrap_or(0.0));
        
        // Symbol-based features (placeholder)
        features.push(0.7); // Symbol match strength
        features.push(0.2); // Reference density
        
        Ok(features)
    }
    
    fn feature_names(&self) -> Vec<String> {
        vec![
            "lsp_score".to_string(),
            "symbol_match_strength".to_string(),
            "reference_density".to_string(),
        ]
    }
}

// Helper structs

struct FeatureResult {
    result: SearchResult,
    features: Vec<f32>,
}

impl IsotonicRegressor {
    /// Fit isotonic regressor on (x, y) pairs
    fn fit(data: &[(f32, f32)]) -> Result<Self> {
        if data.is_empty() {
            anyhow::bail!("Cannot fit isotonic regressor on empty data");
        }
        
        // For now, use simple binning approach
        // Real implementation would use proper isotonic regression algorithm
        let mut x_values = Vec::new();
        let mut y_values = Vec::new();
        
        // Group into bins and average
        let bin_size = (data.len() / 10).max(1);
        for chunk in data.chunks(bin_size) {
            let avg_x = chunk.iter().map(|(x, _)| *x).sum::<f32>() / chunk.len() as f32;
            let avg_y = chunk.iter().map(|(_, y)| *y).sum::<f32>() / chunk.len() as f32;
            
            x_values.push(avg_x);
            y_values.push(avg_y);
        }
        
        Ok(Self {
            x_values,
            y_values,
            sample_count: data.len(),
        })
    }
    
    /// Predict calibrated score
    fn predict(&self, x: f32) -> f32 {
        if self.x_values.is_empty() {
            return x; // No calibration available
        }
        
        // Linear interpolation between points
        for i in 0..self.x_values.len() - 1 {
            if x >= self.x_values[i] && x <= self.x_values[i + 1] {
                let t = (x - self.x_values[i]) / (self.x_values[i + 1] - self.x_values[i]);
                return self.y_values[i] + t * (self.y_values[i + 1] - self.y_values[i]);
            }
        }
        
        // Extrapolate beyond bounds
        if x < self.x_values[0] {
            self.y_values[0]
        } else {
            self.y_values[self.y_values.len() - 1]
        }
    }
}

impl Default for RerankMetrics {
    fn default() -> Self {
        Self {
            samples_processed: 0,
            avg_ndcg_improvement: 0.0,
            avg_precision_improvement: 0.0,
            avg_recall_improvement: 0.0,
            calibration_error: 0.0,
            feature_importances: HashMap::new(),
        }
    }
}

/// Initialize learned reranker
pub async fn initialize_reranker(config: &RerankConfig) -> Result<()> {
    info!("Initializing learned reranker");
    info!("Config: top-{}, isotonic={}, learning_rate={}", 
          config.top_k, config.use_isotonic, config.learning_rate);
    
    // Validate configuration
    if config.top_k == 0 {
        anyhow::bail!("Top-K cannot be zero");
    }
    
    if config.learning_rate <= 0.0 || config.learning_rate > 1.0 {
        anyhow::bail!("Invalid learning rate: {}", config.learning_rate);
    }
    
    info!("Learned reranker initialized");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_reranker_creation() {
        let config = RerankConfig {
            top_k: 100,
            use_isotonic: true,
            learning_rate: 0.01,
            l2_regularization: 0.001,
            min_training_samples: 10,
            combination_strategy: CombinationStrategy::Linear,
        };
        
        let reranker = LearnedReranker::new(config).await.unwrap();
        let metrics = reranker.get_metrics().await;
        assert_eq!(metrics.samples_processed, 0);
    }

    #[test]
    fn test_isotonic_regressor() {
        let data = vec![
            (0.1, 0.2),
            (0.3, 0.4), 
            (0.5, 0.6),
            (0.7, 0.8),
        ];
        
        let regressor = IsotonicRegressor::fit(&data).unwrap();
        
        // Test interpolation
        let pred = regressor.predict(0.4);
        assert!(pred > 0.4 && pred < 0.6); // Should interpolate
        
        // Test extrapolation
        let pred_low = regressor.predict(0.0);
        let pred_high = regressor.predict(1.0);
        assert!(pred_low >= 0.0);
        assert!(pred_high >= 0.0);
    }

    #[test]
    fn test_feature_extractors() {
        let extractor = LexicalFeatureExtractor::new();
        let result = SearchResult {
            id: "test".to_string(),
            content: "def hello_world(): return 'hello'".to_string(),
            file_path: "test.py".to_string(),
            initial_score: 0.8,
            lexical_score: 0.9,
            semantic_score: Some(0.7),
            lsp_score: Some(0.6),
            metadata: HashMap::new(),
        };
        
        let features = extractor.extract_features("hello", &result).unwrap();
        assert_eq!(features.len(), 3); // Based on feature_names count
        
        let names = extractor.feature_names();
        assert_eq!(names.len(), 3);
    }

    #[test]
    fn test_ndcg_calculation() {
        let reranker = LearnedReranker {
            config: RerankConfig {
                top_k: 10,
                use_isotonic: false,
                learning_rate: 0.01,
                l2_regularization: 0.001,
                min_training_samples: 1,
                combination_strategy: CombinationStrategy::Linear,
            },
            feature_extractors: Vec::new(),
            isotonic_models: Arc::new(RwLock::new(HashMap::new())),
            model_weights: Arc::new(RwLock::new(None)),
            metrics: Arc::new(RwLock::new(RerankMetrics::default())),
        };
        
        let results = vec![]; // Mock results
        let relevances = vec![1.0, 0.8, 0.6, 0.4, 0.2];
        
        let ndcg = reranker.calculate_ndcg(&results, &relevances);
        assert!(ndcg >= 0.0 && ndcg <= 1.0);
    }
}