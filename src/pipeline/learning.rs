//! Learning-to-Stop Models for WAND/HNSW Early Termination
//!
//! Implements machine learning models for adaptive early stopping in WAND queries
//! and HNSW vector search with confidence-based termination.
//! 
//! Target: Dynamic threshold learning with >90% accuracy per TODO.md

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Learning-to-stop model coordinator
pub struct LearningStopModel {
    /// WAND query early stopping predictor
    wand_predictor: Arc<RwLock<WandStoppingPredictor>>,
    
    /// HNSW vector search early stopping predictor
    hnsw_predictor: Arc<RwLock<HnswStoppingPredictor>>,
    
    /// Confidence-based termination model
    confidence_model: Arc<RwLock<ConfidenceModel>>,
    
    /// Feature extractors for different query types
    feature_extractors: FeatureExtractors,
    
    /// Model training and adaptation
    training_scheduler: Arc<RwLock<TrainingScheduler>>,
    
    /// Performance metrics
    metrics: Arc<RwLock<LearningMetrics>>,
    
    /// Configuration
    config: LearningConfig,
}

/// Configuration for learning models
#[derive(Debug, Clone)]
pub struct LearningConfig {
    /// Training data window size
    pub training_window_size: usize,
    
    /// Model update frequency (queries)
    pub update_frequency: usize,
    
    /// Learning rate for model adaptation
    pub learning_rate: f64,
    
    /// Confidence threshold for early stopping decisions
    pub confidence_threshold: f64,
    
    /// Minimum training samples before making predictions
    pub min_training_samples: usize,
    
    /// Feature normalization parameters
    pub feature_normalization: bool,
    
    /// WAND-specific configuration
    pub wand_config: WandLearningConfig,
    
    /// HNSW-specific configuration
    pub hnsw_config: HnswLearningConfig,
}

/// WAND learning configuration
#[derive(Debug, Clone)]
pub struct WandLearningConfig {
    /// Maximum WAND iterations before forced stop
    pub max_iterations: usize,
    
    /// Quality degradation threshold
    pub quality_threshold: f64,
    
    /// Score improvement tolerance
    pub score_improvement_tolerance: f64,
    
    /// Term contribution threshold
    pub term_contribution_threshold: f64,
}

/// HNSW learning configuration
#[derive(Debug, Clone)]
pub struct HnswLearningConfig {
    /// Maximum HNSW layers to explore
    pub max_layers: usize,
    
    /// Beam search width
    pub beam_width: usize,
    
    /// Distance threshold for early termination
    pub distance_threshold: f64,
    
    /// Neighbor exploration limit
    pub max_neighbors: usize,
}

/// WAND stopping predictor using learned features
pub struct WandStoppingPredictor {
    /// Learned weights for different features
    weights: HashMap<WandFeature, f64>,
    
    /// Training history for adaptation
    training_history: VecDeque<WandTrainingSample>,
    
    /// Current performance metrics
    accuracy: f64,
    precision: f64,
    recall: f64,
    
    /// Model state
    is_trained: bool,
    last_update: std::time::Instant,
}

/// HNSW stopping predictor
pub struct HnswStoppingPredictor {
    /// Distance-based stopping thresholds per layer
    layer_thresholds: HashMap<usize, f64>,
    
    /// Neighbor quality predictors
    neighbor_quality_weights: HashMap<HnswFeature, f64>,
    
    /// Training samples for HNSW navigation
    training_samples: VecDeque<HnswTrainingSample>,
    
    /// Performance tracking
    search_efficiency: f64,
    quality_maintained: f64,
    
    /// Adaptive parameters
    beam_width_adaptation: f64,
    exploration_decay: f64,
}

/// Confidence-based termination model
pub struct ConfidenceModel {
    /// Confidence predictors for different result types
    confidence_predictors: HashMap<ConfidenceFeature, LinearPredictor>,
    
    /// Calibration parameters for confidence scores
    calibration_params: CalibrationParams,
    
    /// Historical accuracy of confidence predictions
    confidence_accuracy: f64,
    
    /// Training data for confidence calibration
    calibration_data: VecDeque<ConfidenceTrainingSample>,
}

/// Feature extractors for different components
pub struct FeatureExtractors {
    wand_extractor: WandFeatureExtractor,
    hnsw_extractor: HnswFeatureExtractor,
    confidence_extractor: ConfidenceFeatureExtractor,
}

/// Training scheduler for model updates
pub struct TrainingScheduler {
    queries_since_update: usize,
    update_frequency: usize,
    next_training_time: std::time::Instant,
    is_training: bool,
}

/// Learning metrics and performance tracking
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct LearningMetrics {
    pub total_predictions: u64,
    pub correct_early_stops: u64,
    pub incorrect_early_stops: u64,
    pub missed_stopping_opportunities: u64,
    pub avg_computation_saved: f64,
    pub avg_quality_maintained: f64,
    pub model_accuracy: f64,
    pub adaptation_events: u64,
    pub feature_importance: HashMap<String, f64>,
}

/// WAND feature types for learning
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum WandFeature {
    IterationCount,
    ScoreImprovement,
    TermContribution,
    DocumentFrequency,
    QualityEstimate,
    TimeElapsed,
    CandidateSetSize,
    ThresholdConvergence,
}

/// HNSW feature types for learning
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum HnswFeature {
    LayerDepth,
    DistanceToQuery,
    NeighborCount,
    SearchRadius,
    BeamPosition,
    ExplorationRatio,
    DistanceImprovement,
    GraphConnectivity,
}

/// Confidence feature types
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum ConfidenceFeature {
    ResultCount,
    ScoreDistribution,
    SystemAgreement,
    QueryComplexity,
    ProcessingTime,
    ResourceUtilization,
}

/// Training sample for WAND predictor
#[derive(Debug, Clone)]
pub struct WandTrainingSample {
    pub features: HashMap<WandFeature, f64>,
    pub should_have_stopped: bool,
    pub actual_quality: f64,
    pub computation_saved: f64,
    pub timestamp: std::time::Instant,
}

/// Training sample for HNSW predictor
#[derive(Debug, Clone)]
pub struct HnswTrainingSample {
    pub features: HashMap<HnswFeature, f64>,
    pub optimal_stopping_point: usize,
    pub final_quality: f64,
    pub search_efficiency: f64,
    pub timestamp: std::time::Instant,
}

/// Training sample for confidence model
#[derive(Debug, Clone)]
pub struct ConfidenceTrainingSample {
    pub features: HashMap<ConfidenceFeature, f64>,
    pub predicted_confidence: f64,
    pub actual_quality: f64,
    pub timestamp: std::time::Instant,
}

/// Linear predictor for confidence calibration
#[derive(Debug, Clone)]
pub struct LinearPredictor {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
}

/// Confidence calibration parameters
#[derive(Debug, Clone)]
pub struct CalibrationParams {
    temperature: f64,
    shift: f64,
    scale: f64,
}

/// Feature extractors implementations
pub struct WandFeatureExtractor;
pub struct HnswFeatureExtractor;
pub struct ConfidenceFeatureExtractor;

/// Query context for feature extraction
#[derive(Debug, Clone)]
pub struct QueryContext {
    pub query_terms: Vec<String>,
    pub query_vector: Option<Vec<f32>>,
    pub start_time: std::time::Instant,
    pub complexity_score: f64,
    pub expected_result_count: usize,
}

/// Search state for WAND queries
#[derive(Debug, Clone)]
pub struct WandSearchState {
    pub iteration: usize,
    pub current_threshold: f64,
    pub candidate_count: usize,
    pub score_improvements: Vec<f64>,
    pub term_contributions: HashMap<String, f64>,
    pub processing_time: std::time::Duration,
}

/// Search state for HNSW queries
#[derive(Debug, Clone)]
pub struct HnswSearchState {
    pub current_layer: usize,
    pub beam_candidates: Vec<HnswCandidate>,
    pub visited_nodes: usize,
    pub best_distance: f32,
    pub exploration_ratio: f64,
}

/// HNSW candidate representation
#[derive(Debug, Clone)]
pub struct HnswCandidate {
    pub node_id: usize,
    pub distance: f32,
    pub layer: usize,
    pub neighbor_count: usize,
}

/// Stopping decision with learned confidence
#[derive(Debug, Clone)]
pub struct LearnedStoppingDecision {
    pub should_stop: bool,
    pub confidence: f64,
    pub predicted_quality: f64,
    pub estimated_computation_saved: f64,
    pub reasoning: StoppingReasoning,
    pub algorithm_used: String,
}

/// Reasoning for stopping decisions
#[derive(Debug, Clone)]
pub struct StoppingReasoning {
    pub primary_factor: String,
    pub feature_contributions: HashMap<String, f64>,
    pub threshold_exceeded: bool,
    pub quality_sufficient: bool,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            training_window_size: 1000,
            update_frequency: 100,
            learning_rate: 0.01,
            confidence_threshold: 0.85,
            min_training_samples: 50,
            feature_normalization: true,
            wand_config: WandLearningConfig {
                max_iterations: 100,
                quality_threshold: 0.8,
                score_improvement_tolerance: 0.01,
                term_contribution_threshold: 0.05,
            },
            hnsw_config: HnswLearningConfig {
                max_layers: 5,
                beam_width: 64,
                distance_threshold: 0.1,
                max_neighbors: 16,
            },
        }
    }
}

impl LearningStopModel {
    /// Create a new learning-to-stop model
    pub async fn new(config: LearningConfig) -> Result<Self> {
        let wand_predictor = Arc::new(RwLock::new(WandStoppingPredictor::new(config.wand_config.clone())));
        let hnsw_predictor = Arc::new(RwLock::new(HnswStoppingPredictor::new(config.hnsw_config.clone())));
        let confidence_model = Arc::new(RwLock::new(ConfidenceModel::new()));
        
        let feature_extractors = FeatureExtractors {
            wand_extractor: WandFeatureExtractor,
            hnsw_extractor: HnswFeatureExtractor,
            confidence_extractor: ConfidenceFeatureExtractor,
        };
        
        let training_scheduler = Arc::new(RwLock::new(TrainingScheduler {
            queries_since_update: 0,
            update_frequency: config.update_frequency,
            next_training_time: std::time::Instant::now(),
            is_training: false,
        }));
        
        let metrics = Arc::new(RwLock::new(LearningMetrics::default()));
        
        info!("Initialized learning-to-stop model with training window: {}", config.training_window_size);
        
        Ok(Self {
            wand_predictor,
            hnsw_predictor,
            confidence_model,
            feature_extractors,
            training_scheduler,
            metrics,
            config,
        })
    }
    
    /// Predict WAND early stopping decision
    pub async fn predict_wand_stopping(
        &self,
        context: &QueryContext,
        state: &WandSearchState,
    ) -> Result<LearnedStoppingDecision> {
        // Extract features for WAND prediction
        let features = self.feature_extractors.wand_extractor.extract_features(context, state);
        
        // Get prediction from WAND predictor
        let wand_predictor = self.wand_predictor.read().await;
        let (should_stop, confidence) = wand_predictor.predict(&features);
        
        // Estimate quality and computation savings
        let predicted_quality = self.estimate_wand_quality(&features);
        let computation_saved = self.estimate_computation_saved(state.iteration, self.config.wand_config.max_iterations);
        
        // Build reasoning
        let reasoning = self.build_wand_reasoning(&features, should_stop, confidence);
        
        // Update metrics
        self.update_prediction_metrics("wand", should_stop, confidence).await;
        
        Ok(LearnedStoppingDecision {
            should_stop,
            confidence,
            predicted_quality,
            estimated_computation_saved: computation_saved,
            reasoning,
            algorithm_used: "WAND-Learned".to_string(),
        })
    }
    
    /// Predict HNSW early stopping decision
    pub async fn predict_hnsw_stopping(
        &self,
        context: &QueryContext,
        state: &HnswSearchState,
    ) -> Result<LearnedStoppingDecision> {
        // Extract features for HNSW prediction
        let features = self.feature_extractors.hnsw_extractor.extract_features(context, state);
        
        // Get prediction from HNSW predictor
        let hnsw_predictor = self.hnsw_predictor.read().await;
        let (should_stop, confidence) = hnsw_predictor.predict(&features);
        
        // Estimate quality and computation savings
        let predicted_quality = self.estimate_hnsw_quality(&features, state);
        let computation_saved = self.estimate_hnsw_computation_saved(state);
        
        // Build reasoning
        let reasoning = self.build_hnsw_reasoning(&features, should_stop, confidence);
        
        // Update metrics
        self.update_prediction_metrics("hnsw", should_stop, confidence).await;
        
        Ok(LearnedStoppingDecision {
            should_stop,
            confidence,
            predicted_quality,
            estimated_computation_saved: computation_saved,
            reasoning,
            algorithm_used: "HNSW-Learned".to_string(),
        })
    }
    
    /// Train models with new feedback
    pub async fn train_with_feedback(
        &self,
        query_type: &str,
        decision: &LearnedStoppingDecision,
        actual_quality: f64,
        actual_computation_saved: f64,
    ) -> Result<()> {
        let mut scheduler = self.training_scheduler.write().await;
        scheduler.queries_since_update += 1;
        
        match query_type {
            "wand" => {
                let mut predictor = self.wand_predictor.write().await;
                predictor.add_training_sample(WandTrainingSample {
                    features: HashMap::new(), // Would be populated with actual features
                    should_have_stopped: decision.should_stop,
                    actual_quality,
                    computation_saved: actual_computation_saved,
                    timestamp: std::time::Instant::now(),
                });
            }
            "hnsw" => {
                let mut predictor = self.hnsw_predictor.write().await;
                predictor.add_training_sample(HnswTrainingSample {
                    features: HashMap::new(), // Would be populated with actual features
                    optimal_stopping_point: 0, // Would be calculated
                    final_quality: actual_quality,
                    search_efficiency: actual_computation_saved,
                    timestamp: std::time::Instant::now(),
                });
            }
            _ => return Err(anyhow!("Unknown query type: {}", query_type)),
        }
        
        // Update models if needed
        if scheduler.queries_since_update >= scheduler.update_frequency {
            self.update_models().await?;
            scheduler.queries_since_update = 0;
        }
        
        Ok(())
    }
    
    /// Update all models with accumulated training data
    async fn update_models(&self) -> Result<()> {
        let mut scheduler = self.training_scheduler.write().await;
        
        if scheduler.is_training {
            return Ok(()); // Already training
        }
        
        scheduler.is_training = true;
        drop(scheduler);
        
        // Update WAND predictor
        {
            let mut wand_predictor = self.wand_predictor.write().await;
            wand_predictor.update_model(self.config.learning_rate)?;
        }
        
        // Update HNSW predictor
        {
            let mut hnsw_predictor = self.hnsw_predictor.write().await;
            hnsw_predictor.update_model(self.config.learning_rate)?;
        }
        
        // Update confidence model
        {
            let mut confidence_model = self.confidence_model.write().await;
            confidence_model.update_calibration()?;
        }
        
        // Reset training flag
        {
            let mut scheduler = self.training_scheduler.write().await;
            scheduler.is_training = false;
            scheduler.next_training_time = std::time::Instant::now() + std::time::Duration::from_secs(300); // 5 min
        }
        
        // Update adaptation metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.adaptation_events += 1;
        }
        
        info!("Updated learning models with new training data");
        
        Ok(())
    }
    
    /// Estimate WAND quality from features
    fn estimate_wand_quality(&self, features: &HashMap<WandFeature, f64>) -> f64 {
        let score_improvement = features.get(&WandFeature::ScoreImprovement).unwrap_or(&0.0);
        let quality_estimate = features.get(&WandFeature::QualityEstimate).unwrap_or(&0.5);
        let threshold_convergence = features.get(&WandFeature::ThresholdConvergence).unwrap_or(&0.0);
        
        // Simple quality estimation model
        (score_improvement * 0.4 + quality_estimate * 0.4 + threshold_convergence * 0.2).min(1.0)
    }
    
    /// Estimate HNSW quality from features and state
    fn estimate_hnsw_quality(&self, features: &HashMap<HnswFeature, f64>, state: &HnswSearchState) -> f64 {
        let distance_improvement = features.get(&HnswFeature::DistanceImprovement).unwrap_or(&0.0);
        let exploration_ratio = features.get(&HnswFeature::ExplorationRatio).unwrap_or(&0.5);
        
        let distance_quality = if state.best_distance > 0.0 {
            (1.0 - state.best_distance).max(0.0)
        } else {
            0.0
        };
        
        (distance_improvement * 0.3 + exploration_ratio * 0.3 + distance_quality * 0.4).min(1.0)
    }
    
    /// Estimate computation saved based on early stopping
    fn estimate_computation_saved(&self, current_iteration: usize, max_iterations: usize) -> f64 {
        if max_iterations == 0 {
            return 0.0;
        }
        
        let remaining_iterations = max_iterations.saturating_sub(current_iteration);
        remaining_iterations as f64 / max_iterations as f64
    }
    
    /// Estimate computation saved for HNSW search
    fn estimate_hnsw_computation_saved(&self, state: &HnswSearchState) -> f64 {
        let max_possible_visits = self.config.hnsw_config.max_neighbors * self.config.hnsw_config.max_layers;
        let remaining_visits = max_possible_visits.saturating_sub(state.visited_nodes);
        
        remaining_visits as f64 / max_possible_visits as f64
    }
    
    /// Build reasoning for WAND stopping decision
    fn build_wand_reasoning(&self, features: &HashMap<WandFeature, f64>, should_stop: bool, confidence: f64) -> StoppingReasoning {
        let mut feature_contributions = HashMap::new();
        
        // Calculate feature contributions (simplified)
        for (feature, value) in features {
            let contribution = value * confidence; // Weighted by confidence
            feature_contributions.insert(format!("{:?}", feature), contribution);
        }
        
        let primary_factor = if should_stop {
            "Score convergence detected"
        } else {
            "Continued exploration needed"
        }.to_string();
        
        StoppingReasoning {
            primary_factor,
            feature_contributions,
            threshold_exceeded: confidence > self.config.confidence_threshold,
            quality_sufficient: features.get(&WandFeature::QualityEstimate).unwrap_or(&0.0) > &self.config.wand_config.quality_threshold,
        }
    }
    
    /// Build reasoning for HNSW stopping decision
    fn build_hnsw_reasoning(&self, features: &HashMap<HnswFeature, f64>, should_stop: bool, confidence: f64) -> StoppingReasoning {
        let mut feature_contributions = HashMap::new();
        
        for (feature, value) in features {
            let contribution = value * confidence;
            feature_contributions.insert(format!("{:?}", feature), contribution);
        }
        
        let primary_factor = if should_stop {
            "Distance threshold reached"
        } else {
            "Further exploration beneficial"
        }.to_string();
        
        StoppingReasoning {
            primary_factor,
            feature_contributions,
            threshold_exceeded: confidence > self.config.confidence_threshold,
            quality_sufficient: features.get(&HnswFeature::DistanceToQuery).unwrap_or(&1.0) < &self.config.hnsw_config.distance_threshold,
        }
    }
    
    /// Update prediction metrics
    async fn update_prediction_metrics(&self, algorithm: &str, prediction: bool, confidence: f64) {
        let mut metrics = self.metrics.write().await;
        metrics.total_predictions += 1;
        
        // Update algorithm-specific metrics (simplified)
        if prediction {
            debug!("Predicted early stop for {} with confidence {:.3}", algorithm, confidence);
        }
    }
    
    /// Get current learning metrics
    pub async fn get_metrics(&self) -> LearningMetrics {
        self.metrics.read().await.clone()
    }
}

impl WandStoppingPredictor {
    pub fn new(_config: WandLearningConfig) -> Self {
        let mut weights = HashMap::new();
        
        // Initialize feature weights
        weights.insert(WandFeature::IterationCount, -0.1);
        weights.insert(WandFeature::ScoreImprovement, 0.8);
        weights.insert(WandFeature::TermContribution, 0.6);
        weights.insert(WandFeature::QualityEstimate, 0.9);
        weights.insert(WandFeature::ThresholdConvergence, 0.7);
        
        Self {
            weights,
            training_history: VecDeque::new(),
            accuracy: 0.5,
            precision: 0.5,
            recall: 0.5,
            is_trained: false,
            last_update: std::time::Instant::now(),
        }
    }
    
    pub fn predict(&self, features: &HashMap<WandFeature, f64>) -> (bool, f64) {
        let mut score = 0.0;
        let mut feature_count = 0;
        
        for (feature, weight) in &self.weights {
            if let Some(feature_value) = features.get(feature) {
                score += feature_value * weight;
                feature_count += 1;
            }
        }
        
        if feature_count > 0 {
            score /= feature_count as f64;
        }
        
        let confidence = (score.tanh() + 1.0) / 2.0; // Normalize to [0,1]
        let should_stop = confidence > 0.5;
        
        (should_stop, confidence)
    }
    
    pub fn add_training_sample(&mut self, sample: WandTrainingSample) {
        self.training_history.push_back(sample);
        
        // Keep training window size limited
        while self.training_history.len() > 1000 {
            self.training_history.pop_front();
        }
    }
    
    pub fn update_model(&mut self, learning_rate: f64) -> Result<()> {
        if self.training_history.len() < 10 {
            return Ok(()); // Not enough data
        }
        
        // Simple gradient descent update (simplified)
        for sample in self.training_history.iter().rev().take(100) {
            let (predicted, _) = self.predict(&sample.features);
            let error = if sample.should_have_stopped { 1.0 } else { 0.0 } - if predicted { 1.0 } else { 0.0 };
            
            // Update weights based on error
            for (feature, feature_value) in &sample.features {
                if let Some(weight) = self.weights.get_mut(feature) {
                    *weight += learning_rate * error * feature_value;
                }
            }
        }
        
        self.last_update = std::time::Instant::now();
        self.is_trained = true;
        
        Ok(())
    }
}

impl HnswStoppingPredictor {
    pub fn new(_config: HnswLearningConfig) -> Self {
        let mut layer_thresholds = HashMap::new();
        let mut neighbor_quality_weights = HashMap::new();
        
        // Initialize per-layer thresholds
        for layer in 0..5 {
            layer_thresholds.insert(layer, 0.1 * (layer + 1) as f64);
        }
        
        // Initialize feature weights
        neighbor_quality_weights.insert(HnswFeature::DistanceToQuery, 0.9);
        neighbor_quality_weights.insert(HnswFeature::DistanceImprovement, 0.8);
        neighbor_quality_weights.insert(HnswFeature::ExplorationRatio, 0.6);
        neighbor_quality_weights.insert(HnswFeature::GraphConnectivity, 0.4);
        
        Self {
            layer_thresholds,
            neighbor_quality_weights,
            training_samples: VecDeque::new(),
            search_efficiency: 0.5,
            quality_maintained: 0.5,
            beam_width_adaptation: 1.0,
            exploration_decay: 0.95,
        }
    }
    
    pub fn predict(&self, features: &HashMap<HnswFeature, f64>) -> (bool, f64) {
        let mut quality_score = 0.0;
        let mut feature_count = 0;
        
        for (feature, weight) in &self.neighbor_quality_weights {
            if let Some(feature_value) = features.get(feature) {
                quality_score += feature_value * weight;
                feature_count += 1;
            }
        }
        
        if feature_count > 0 {
            quality_score /= feature_count as f64;
        }
        
        let confidence = quality_score.min(1.0).max(0.0);
        let should_stop = confidence > 0.7; // Higher threshold for HNSW
        
        (should_stop, confidence)
    }
    
    pub fn add_training_sample(&mut self, sample: HnswTrainingSample) {
        self.training_samples.push_back(sample);
        
        while self.training_samples.len() > 1000 {
            self.training_samples.pop_front();
        }
    }
    
    pub fn update_model(&mut self, learning_rate: f64) -> Result<()> {
        if self.training_samples.len() < 10 {
            return Ok(());
        }
        
        // Update model parameters based on training samples
        // This is a simplified version - real implementation would use more sophisticated ML
        let recent_samples: Vec<_> = self.training_samples.iter().rev().take(50).collect();
        
        let avg_efficiency: f64 = recent_samples.iter().map(|s| s.search_efficiency).sum::<f64>() / recent_samples.len() as f64;
        let avg_quality: f64 = recent_samples.iter().map(|s| s.final_quality).sum::<f64>() / recent_samples.len() as f64;
        
        // Adapt parameters
        self.search_efficiency = self.search_efficiency * (1.0 - learning_rate) + avg_efficiency * learning_rate;
        self.quality_maintained = self.quality_maintained * (1.0 - learning_rate) + avg_quality * learning_rate;
        
        // Adapt exploration parameters
        if avg_efficiency < 0.6 {
            self.beam_width_adaptation *= 1.1; // Increase beam width
        } else if avg_efficiency > 0.8 {
            self.beam_width_adaptation *= 0.95; // Decrease beam width
        }
        
        Ok(())
    }
}

impl ConfidenceModel {
    pub fn new() -> Self {
        Self {
            confidence_predictors: HashMap::new(),
            calibration_params: CalibrationParams {
                temperature: 1.0,
                shift: 0.0,
                scale: 1.0,
            },
            confidence_accuracy: 0.5,
            calibration_data: VecDeque::new(),
        }
    }
    
    pub fn update_calibration(&mut self) -> Result<()> {
        // Update confidence calibration based on historical data
        if self.calibration_data.len() < 20 {
            return Ok(());
        }
        
        // Simple calibration update (real implementation would use isotonic regression)
        let recent_data: Vec<_> = self.calibration_data.iter().rev().take(100).collect();
        
        let avg_predicted: f64 = recent_data.iter().map(|s| s.predicted_confidence).sum::<f64>() / recent_data.len() as f64;
        let avg_actual: f64 = recent_data.iter().map(|s| s.actual_quality).sum::<f64>() / recent_data.len() as f64;
        
        // Adjust calibration parameters
        if (avg_predicted - avg_actual).abs() > 0.1 {
            self.calibration_params.shift += (avg_actual - avg_predicted) * 0.01;
        }
        
        Ok(())
    }
}

// Feature extractor implementations
impl WandFeatureExtractor {
    pub fn extract_features(&self, context: &QueryContext, state: &WandSearchState) -> HashMap<WandFeature, f64> {
        let mut features = HashMap::new();
        
        let elapsed = context.start_time.elapsed().as_millis() as f64;
        
        features.insert(WandFeature::IterationCount, state.iteration as f64);
        features.insert(WandFeature::TimeElapsed, elapsed);
        features.insert(WandFeature::CandidateSetSize, state.candidate_count as f64);
        
        // Calculate score improvement trend
        let score_improvement = if state.score_improvements.len() >= 2 {
            let recent = &state.score_improvements[state.score_improvements.len()-2..];
            recent[1] - recent[0]
        } else {
            0.0
        };
        features.insert(WandFeature::ScoreImprovement, score_improvement);
        
        // Average term contribution
        let avg_term_contribution = if !state.term_contributions.is_empty() {
            state.term_contributions.values().sum::<f64>() / state.term_contributions.len() as f64
        } else {
            0.0
        };
        features.insert(WandFeature::TermContribution, avg_term_contribution);
        
        // Quality estimate based on threshold stability
        let quality_estimate = if state.iteration > 0 { 
            (state.current_threshold / state.iteration as f64).min(1.0) 
        } else { 
            0.5 
        };
        features.insert(WandFeature::QualityEstimate, quality_estimate);
        
        // Threshold convergence
        let threshold_convergence = if state.score_improvements.len() >= 3 {
            let recent_variance: f64 = state.score_improvements.iter().rev().take(3)
                .map(|&x| (x - score_improvement).powi(2))
                .sum::<f64>() / 3.0;
            (1.0 / (1.0 + recent_variance)).min(1.0)
        } else {
            0.0
        };
        features.insert(WandFeature::ThresholdConvergence, threshold_convergence);
        
        features
    }
}

impl HnswFeatureExtractor {
    pub fn extract_features(&self, context: &QueryContext, state: &HnswSearchState) -> HashMap<HnswFeature, f64> {
        let mut features = HashMap::new();
        
        features.insert(HnswFeature::LayerDepth, state.current_layer as f64);
        features.insert(HnswFeature::DistanceToQuery, state.best_distance as f64);
        features.insert(HnswFeature::BeamPosition, state.beam_candidates.len() as f64);
        features.insert(HnswFeature::ExplorationRatio, state.exploration_ratio);
        
        // Average neighbor count
        let avg_neighbors = if !state.beam_candidates.is_empty() {
            state.beam_candidates.iter().map(|c| c.neighbor_count as f64).sum::<f64>() / state.beam_candidates.len() as f64
        } else {
            0.0
        };
        features.insert(HnswFeature::NeighborCount, avg_neighbors);
        
        // Distance improvement (if we have previous best distances)
        let distance_improvement = if state.best_distance < 1.0 {
            1.0 - state.best_distance as f64
        } else {
            0.0
        };
        features.insert(HnswFeature::DistanceImprovement, distance_improvement);
        
        // Graph connectivity estimate
        let connectivity = if !state.beam_candidates.is_empty() {
            let total_connections: usize = state.beam_candidates.iter().map(|c| c.neighbor_count).sum();
            (total_connections as f64 / state.beam_candidates.len() as f64) / 16.0 // Normalize by max degree
        } else {
            0.0
        };
        features.insert(HnswFeature::GraphConnectivity, connectivity);
        
        features
    }
}

impl ConfidenceFeatureExtractor {
    pub fn extract_features(&self, _context: &QueryContext, result_count: usize, processing_time: f64) -> HashMap<ConfidenceFeature, f64> {
        let mut features = HashMap::new();
        
        features.insert(ConfidenceFeature::ResultCount, result_count as f64);
        features.insert(ConfidenceFeature::ProcessingTime, processing_time);
        
        // Simplified features for confidence prediction
        let score_distribution = if result_count > 0 { 1.0 } else { 0.0 };
        features.insert(ConfidenceFeature::ScoreDistribution, score_distribution);
        
        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_learning_model_creation() {
        let config = LearningConfig::default();
        let model = LearningStopModel::new(config).await;
        assert!(model.is_ok());
    }

    #[tokio::test]
    async fn test_wand_feature_extraction() {
        let extractor = WandFeatureExtractor;
        let context = QueryContext {
            query_terms: vec!["test".to_string()],
            query_vector: None,
            start_time: std::time::Instant::now(),
            complexity_score: 0.5,
            expected_result_count: 10,
        };
        
        let state = WandSearchState {
            iteration: 5,
            current_threshold: 0.8,
            candidate_count: 20,
            score_improvements: vec![0.1, 0.15, 0.18],
            term_contributions: HashMap::new(),
            processing_time: std::time::Duration::from_millis(50),
        };
        
        let features = extractor.extract_features(&context, &state);
        
        assert!(features.contains_key(&WandFeature::IterationCount));
        assert!(features.contains_key(&WandFeature::ScoreImprovement));
        assert!(features.contains_key(&WandFeature::CandidateSetSize));
        
        assert_eq!(*features.get(&WandFeature::IterationCount).unwrap(), 5.0);
    }

    #[tokio::test]
    async fn test_wand_predictor() {
        let config = WandLearningConfig::default();
        let predictor = WandStoppingPredictor::new(config);
        
        let mut features = HashMap::new();
        features.insert(WandFeature::QualityEstimate, 0.9);
        features.insert(WandFeature::ScoreImprovement, 0.1);
        features.insert(WandFeature::ThresholdConvergence, 0.8);
        
        let (should_stop, confidence) = predictor.predict(&features);
        
        // With high quality features, should have reasonable confidence
        assert!(confidence > 0.3);
        
        // Test with low quality features
        let mut low_features = HashMap::new();
        low_features.insert(WandFeature::QualityEstimate, 0.1);
        low_features.insert(WandFeature::ScoreImprovement, 0.01);
        
        let (low_stop, low_confidence) = predictor.predict(&low_features);
        
        // Low quality should result in lower confidence
        assert!(low_confidence < confidence);
    }

    #[tokio::test]
    async fn test_hnsw_feature_extraction() {
        let extractor = HnswFeatureExtractor;
        let context = QueryContext {
            query_terms: vec![],
            query_vector: Some(vec![0.1, 0.2, 0.3]),
            start_time: std::time::Instant::now(),
            complexity_score: 0.7,
            expected_result_count: 5,
        };
        
        let candidates = vec![
            HnswCandidate { node_id: 1, distance: 0.1, layer: 0, neighbor_count: 8 },
            HnswCandidate { node_id: 2, distance: 0.15, layer: 0, neighbor_count: 12 },
        ];
        
        let state = HnswSearchState {
            current_layer: 1,
            beam_candidates: candidates,
            visited_nodes: 25,
            best_distance: 0.1,
            exploration_ratio: 0.6,
        };
        
        let features = extractor.extract_features(&context, &state);
        
        assert!(features.contains_key(&HnswFeature::LayerDepth));
        assert!(features.contains_key(&HnswFeature::DistanceToQuery));
        assert!(features.contains_key(&HnswFeature::ExplorationRatio));
        
        assert_eq!(*features.get(&HnswFeature::LayerDepth).unwrap(), 1.0);
        assert_eq!(*features.get(&HnswFeature::ExplorationRatio).unwrap(), 0.6);
    }

    #[tokio::test]
    async fn test_learning_model_prediction() {
        let config = LearningConfig::default();
        let model = LearningStopModel::new(config).await.unwrap();
        
        let context = QueryContext {
            query_terms: vec!["function".to_string(), "test".to_string()],
            query_vector: None,
            start_time: std::time::Instant::now(),
            complexity_score: 0.6,
            expected_result_count: 15,
        };
        
        let wand_state = WandSearchState {
            iteration: 10,
            current_threshold: 0.75,
            candidate_count: 30,
            score_improvements: vec![0.2, 0.25, 0.27],
            term_contributions: HashMap::new(),
            processing_time: std::time::Duration::from_millis(80),
        };
        
        let decision = model.predict_wand_stopping(&context, &wand_state).await.unwrap();
        
        assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
        assert!(decision.predicted_quality >= 0.0 && decision.predicted_quality <= 1.0);
        assert!(decision.estimated_computation_saved >= 0.0);
        assert!(!decision.reasoning.feature_contributions.is_empty());
    }
}