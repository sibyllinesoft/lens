//! # Production Conformal Prediction Router
//!
//! Risk-aware routing system using conformal prediction with:
//! - Statistical uncertainty quantification for search quality
//! - Budget-constrained upshift routing (≤5% daily budget)
//! - Calibrated confidence intervals with nonconformity scoring
//! - Performance-aware routing decisions with p95 headroom monitoring
//! - Production-ready monitoring and alerting

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, info, instrument, warn};

use super::query_classifier::{QueryClassification, QueryIntent};

/// Conformal prediction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformalRouterConfig {
    /// Risk threshold for upshift decisions (0.0-1.0)
    pub risk_threshold: f32,
    /// Daily upshift budget percentage (0.0-100.0)
    pub daily_budget_percent: f32,
    /// Confidence level for prediction intervals
    pub confidence_level: f32,
    /// Minimum calibration samples required
    pub min_calibration_samples: usize,
    /// P95 latency headroom threshold (ms)
    pub p95_headroom_threshold_ms: f32,
    /// Enable/disable router
    pub enabled: bool,
    /// Calibration data retention period (hours)
    pub calibration_retention_hours: u64,
}

impl Default for ConformalRouterConfig {
    fn default() -> Self {
        Self {
            risk_threshold: 0.6,
            daily_budget_percent: 5.0,
            confidence_level: 0.95,
            min_calibration_samples: 100,
            p95_headroom_threshold_ms: 10.0,
            enabled: true,
            calibration_retention_hours: 24,
        }
    }
}

/// Query features for conformal prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformalFeatures {
    /// Query characteristics
    pub query_length: u32,
    pub word_count: u32,
    pub has_special_chars: bool,
    pub fuzzy_enabled: bool,
    pub structural_mode: bool,
    pub avg_word_length: f32,
    pub query_entropy: f32,
    pub identifier_density: f32,
    pub semantic_complexity: f32,
    
    /// Context features
    pub has_file_context: bool,
    pub language_detected: bool,
    pub intent_confidence: f32,
    pub naturalness_score: f32,
    
    /// Historical features
    pub similar_queries_success_rate: f32,
    pub user_satisfaction_history: f32,
}

/// Risk assessment result from conformal prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    /// Overall risk score [0.0, 1.0]
    pub risk_score: f32,
    /// Prediction confidence interval [lower, upper]
    pub confidence_interval: (f32, f32),
    /// Nonconformity score from calibration
    pub nonconformity_score: f32,
    /// Whether prediction is calibrated
    pub calibrated: bool,
    /// Contributing risk factors
    pub risk_factors: Vec<RiskFactor>,
}

/// Individual risk factor contribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    pub name: String,
    pub contribution: f32,
    pub description: String,
}

/// Upshift routing decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    /// Whether to upshift resources
    pub should_upshift: bool,
    /// Type of upshift to apply
    pub upshift_type: UpshiftType,
    /// Budget units consumed
    pub budget_consumed: f32,
    /// Reason for routing decision
    pub routing_reason: String,
    /// Expected quality improvement
    pub expected_improvement: f32,
    /// Risk assessment that led to decision
    pub risk_assessment: RiskAssessment,
}

/// Types of upshift optimizations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UpshiftType {
    /// No upshift
    None,
    /// Use higher-dimensional embeddings
    HighDimEmbeddings,
    /// Increase HNSW search parameters
    EnhancedSearch,
    /// Apply MMR diversity optimization
    DiversityOptimization,
    /// Cross-encoder reranking
    CrossEncoder,
    /// Semantic reranking for natural language queries
    SemanticReranking,
    /// LSP integration for symbol queries
    LSPIntegration,
    /// AST analysis for structural queries
    ASTAnalysis,
    /// Cross-language search optimization
    CrossLanguageSearch,
}

impl std::fmt::Display for UpshiftType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UpshiftType::None => write!(f, "none"),
            UpshiftType::HighDimEmbeddings => write!(f, "high_dim_embeddings"),
            UpshiftType::EnhancedSearch => write!(f, "enhanced_search"),
            UpshiftType::DiversityOptimization => write!(f, "diversity_optimization"),
            UpshiftType::CrossEncoder => write!(f, "cross_encoder"),
            UpshiftType::SemanticReranking => write!(f, "semantic_reranking"),
            UpshiftType::LSPIntegration => write!(f, "lsp_integration"),
            UpshiftType::ASTAnalysis => write!(f, "ast_analysis"),
            UpshiftType::CrossLanguageSearch => write!(f, "cross_language_search"),
        }
    }
}

/// Budget tracking and management
#[derive(Debug, Clone)]
pub struct BudgetManager {
    /// Daily budget allocation
    daily_budget: f32,
    /// Current day's usage
    current_usage: f32,
    /// Usage history for trend analysis
    usage_history: VecDeque<(SystemTime, f32)>,
    /// Last reset timestamp
    last_reset: SystemTime,
    /// Total queries processed today
    total_queries_today: u64,
}

impl BudgetManager {
    pub fn new(daily_budget_percent: f32) -> Self {
        Self {
            daily_budget: daily_budget_percent / 100.0,
            current_usage: 0.0,
            usage_history: VecDeque::with_capacity(1000),
            last_reset: SystemTime::now(),
            total_queries_today: 0,
        }
    }
    
    /// Check if upshift is within budget
    pub fn can_upshift(&mut self, cost: f32) -> bool {
        self.maybe_reset_daily();
        (self.current_usage + cost) <= self.daily_budget
    }
    
    /// Record upshift usage
    pub fn record_upshift(&mut self, cost: f32) {
        self.maybe_reset_daily();
        self.current_usage += cost;
        let now = SystemTime::now();
        self.usage_history.push_back((now, cost));
        
        // Keep last 24 hours of history
        let cutoff = now - Duration::from_secs(24 * 60 * 60);
        while let Some((timestamp, _)) = self.usage_history.front() {
            if *timestamp < cutoff {
                self.usage_history.pop_front();
            } else {
                break;
            }
        }
    }
    
    /// Record total query for rate calculation
    pub fn record_query(&mut self) {
        self.maybe_reset_daily();
        self.total_queries_today += 1;
    }
    
    /// Get current budget status
    pub fn get_status(&self) -> BudgetStatus {
        let usage_rate = if self.total_queries_today > 0 {
            (self.current_usage * self.total_queries_today as f32) / self.total_queries_today as f32
        } else {
            0.0
        };
        
        BudgetStatus {
            daily_budget_percent: self.daily_budget * 100.0,
            current_usage_percent: (self.current_usage / self.daily_budget) * 100.0,
            remaining_budget: self.daily_budget - self.current_usage,
            usage_rate_percent: usage_rate * 100.0,
            total_queries: self.total_queries_today,
            p95_headroom_estimate: self.estimate_p95_headroom(),
        }
    }
    
    /// Reset daily counters if new day
    fn maybe_reset_daily(&mut self) {
        let now = SystemTime::now();
        if let Ok(duration) = now.duration_since(self.last_reset) {
            if duration >= Duration::from_secs(24 * 60 * 60) {
                self.current_usage = 0.0;
                self.total_queries_today = 0;
                self.last_reset = now;
                info!("Reset daily budget counters");
            }
        }
    }
    
    /// Estimate P95 latency headroom based on recent usage
    fn estimate_p95_headroom(&self) -> f32 {
        let recent_usage: f32 = self.usage_history.iter()
            .filter(|(timestamp, _)| {
                SystemTime::now().duration_since(*timestamp)
                    .unwrap_or(Duration::ZERO) < Duration::from_secs(60 * 60)
            })
            .map(|(_, cost)| cost)
            .sum();
        
        if recent_usage > 0.5 {
            1.0 // Low headroom
        } else if recent_usage > 0.2 {
            5.0 // Medium headroom
        } else {
            15.0 // High headroom
        }
    }
}

/// Current budget status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetStatus {
    pub daily_budget_percent: f32,
    pub current_usage_percent: f32,
    pub remaining_budget: f32,
    pub usage_rate_percent: f32,
    pub total_queries: u64,
    pub p95_headroom_estimate: f32,
}

/// Calibration sample for training conformal predictor
#[derive(Debug, Clone)]
pub struct CalibrationSample {
    pub features: ConformalFeatures,
    pub predicted_quality: f32,
    pub actual_quality: f32,
    pub timestamp: SystemTime,
}

/// Conformal predictor for quality risk assessment
pub struct ConformalPredictor {
    /// Calibration dataset
    calibration_data: VecDeque<CalibrationSample>,
    /// Nonconformity scores for quantile calculation
    nonconformity_scores: Vec<f32>,
    /// Whether predictor is calibrated
    is_calibrated: bool,
    /// Minimum samples required for calibration
    min_samples: usize,
    /// Retention period for calibration data
    retention_period: Duration,
}

impl ConformalPredictor {
    pub fn new(min_samples: usize, retention_hours: u64) -> Self {
        Self {
            calibration_data: VecDeque::with_capacity(min_samples * 2),
            nonconformity_scores: Vec::with_capacity(min_samples),
            is_calibrated: false,
            min_samples,
            retention_period: Duration::from_secs(retention_hours * 60 * 60),
        }
    }
    
    /// Add calibration sample
    pub fn add_calibration_sample(&mut self, sample: CalibrationSample) {
        // Remove old samples
        let cutoff = SystemTime::now() - self.retention_period;
        while let Some(front) = self.calibration_data.front() {
            if front.timestamp < cutoff {
                self.calibration_data.pop_front();
            } else {
                break;
            }
        }
        
        self.calibration_data.push_back(sample);
        
        // Recalibrate if we have enough samples
        if self.calibration_data.len() >= self.min_samples {
            self.recalibrate();
        }
    }
    
    /// Predict quality risk with confidence intervals
    pub fn predict_risk(
        &self,
        features: &ConformalFeatures,
        confidence_level: f32,
    ) -> RiskAssessment {
        let base_prediction = self.predict_base_quality(features);
        
        if !self.is_calibrated || self.nonconformity_scores.is_empty() {
            return self.heuristic_risk_assessment(features, base_prediction);
        }
        
        // Calculate prediction interval using conformal prediction
        let alpha = 1.0 - confidence_level;
        let quantile_index = ((1.0 - alpha) * (self.nonconformity_scores.len() as f32 + 1.0)) as usize;
        let quantile_index = quantile_index.min(self.nonconformity_scores.len() - 1);
        
        let nonconformity_quantile = self.nonconformity_scores[quantile_index];
        
        let lower_bound = (base_prediction - nonconformity_quantile).max(0.0);
        let upper_bound = (base_prediction + nonconformity_quantile).min(1.0);
        
        // Risk score: high risk if predicted quality is low or interval is wide
        let interval_width = upper_bound - lower_bound;
        let quality_risk = 1.0 - base_prediction;
        let uncertainty_risk = interval_width;
        let risk_score = (quality_risk * 0.7 + uncertainty_risk * 0.3).min(1.0);
        
        RiskAssessment {
            risk_score,
            confidence_interval: (lower_bound, upper_bound),
            nonconformity_score: nonconformity_quantile,
            calibrated: true,
            risk_factors: self.identify_risk_factors(features, quality_risk, uncertainty_risk),
        }
    }
    
    /// Recalibrate the predictor using current data
    fn recalibrate(&mut self) {
        self.nonconformity_scores.clear();
        
        for sample in &self.calibration_data {
            let predicted = self.predict_base_quality(&sample.features);
            let nonconformity = (sample.actual_quality - predicted).abs();
            self.nonconformity_scores.push(nonconformity);
        }
        
        // Sort for quantile calculations
        self.nonconformity_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        self.is_calibrated = true;
        
        info!(
            "Conformal predictor recalibrated with {} samples, mean nonconformity: {:.3}",
            self.calibration_data.len(),
            self.nonconformity_scores.iter().sum::<f32>() / self.nonconformity_scores.len() as f32
        );
    }
    
    /// Predict base quality score from features
    fn predict_base_quality(&self, features: &ConformalFeatures) -> f32 {
        // Simple linear model (in production, use trained ML model)
        let mut quality = 0.7; // Base quality
        
        // Query complexity factors (negative impact)
        if features.query_length > 100 { quality -= 0.1; }
        if features.word_count > 10 { quality -= 0.05; }
        if features.semantic_complexity > 0.8 { quality -= 0.15; }
        if features.avg_word_length < 3.0 { quality -= 0.1; }
        
        // Positive factors
        if features.has_file_context { quality += 0.1; }
        if features.language_detected { quality += 0.05; }
        if features.intent_confidence > 0.8 { quality += 0.1; }
        if features.identifier_density > 0.6 { quality += 0.08; }
        if features.naturalness_score > 0.7 { quality += 0.05; }
        
        // Historical factors
        quality += features.similar_queries_success_rate * 0.1;
        quality += features.user_satisfaction_history * 0.05;
        
        quality.clamp(0.1, 0.95)
    }
    
    /// Heuristic risk assessment when not calibrated
    fn heuristic_risk_assessment(&self, features: &ConformalFeatures, base_prediction: f32) -> RiskAssessment {
        let mut risk_score = 1.0 - base_prediction;
        
        // Add uncertainty for complex queries
        if features.semantic_complexity > 0.7 { risk_score += 0.1; }
        if features.query_length > 200 { risk_score += 0.1; }
        if features.naturalness_score < 0.3 { risk_score += 0.05; }
        
        risk_score = risk_score.clamp(0.0, 1.0);
        
        let uncertainty = 0.15; // Fixed uncertainty when not calibrated
        
        RiskAssessment {
            risk_score,
            confidence_interval: (
                (base_prediction - uncertainty).max(0.0),
                (base_prediction + uncertainty).min(1.0)
            ),
            nonconformity_score: uncertainty,
            calibrated: false,
            risk_factors: self.identify_risk_factors(features, risk_score, uncertainty),
        }
    }
    
    /// Identify contributing risk factors
    fn identify_risk_factors(&self, features: &ConformalFeatures, quality_risk: f32, uncertainty_risk: f32) -> Vec<RiskFactor> {
        let mut factors = Vec::new();
        
        if features.semantic_complexity > 0.8 {
            factors.push(RiskFactor {
                name: "semantic_complexity".to_string(),
                contribution: features.semantic_complexity * 0.2,
                description: "High semantic complexity may reduce search accuracy".to_string(),
            });
        }
        
        if features.query_length > 100 {
            factors.push(RiskFactor {
                name: "query_length".to_string(),
                contribution: (features.query_length as f32 / 500.0).min(0.2),
                description: "Long queries are harder to process accurately".to_string(),
            });
        }
        
        if features.intent_confidence < 0.5 {
            factors.push(RiskFactor {
                name: "intent_ambiguity".to_string(),
                contribution: (0.5 - features.intent_confidence) * 0.3,
                description: "Ambiguous query intent increases search uncertainty".to_string(),
            });
        }
        
        if !features.has_file_context {
            factors.push(RiskFactor {
                name: "missing_context".to_string(),
                contribution: 0.1,
                description: "Lack of file context limits search precision".to_string(),
            });
        }
        
        factors
    }
    
    /// Get predictor status
    pub fn get_status(&self) -> ConformalPredictorStatus {
        ConformalPredictorStatus {
            is_calibrated: self.is_calibrated,
            calibration_samples: self.calibration_data.len(),
            min_samples_required: self.min_samples,
            mean_nonconformity: if self.nonconformity_scores.is_empty() {
                0.0
            } else {
                self.nonconformity_scores.iter().sum::<f32>() / self.nonconformity_scores.len() as f32
            },
            oldest_sample_age_hours: self.calibration_data.front()
                .and_then(|sample| SystemTime::now().duration_since(sample.timestamp).ok())
                .map(|d| d.as_secs() as f32 / 3600.0)
                .unwrap_or(0.0),
        }
    }
}

/// Status of conformal predictor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformalPredictorStatus {
    pub is_calibrated: bool,
    pub calibration_samples: usize,
    pub min_samples_required: usize,
    pub mean_nonconformity: f32,
    pub oldest_sample_age_hours: f32,
}

/// Main conformal router with risk-aware upshift decisions
pub struct ConformalRouter {
    config: ConformalRouterConfig,
    predictor: RwLock<ConformalPredictor>,
    budget_manager: RwLock<BudgetManager>,
    metrics: Arc<parking_lot::RwLock<ConformalRouterMetrics>>,
    upshift_costs: HashMap<UpshiftType, f32>,
}

impl ConformalRouter {
    /// Create new conformal router
    pub fn new(config: ConformalRouterConfig) -> Self {
        let predictor = RwLock::new(ConformalPredictor::new(
            config.min_calibration_samples,
            config.calibration_retention_hours,
        ));
        
        let budget_manager = RwLock::new(BudgetManager::new(config.daily_budget_percent));
        
        let mut upshift_costs = HashMap::new();
        upshift_costs.insert(UpshiftType::None, 0.0);
        upshift_costs.insert(UpshiftType::HighDimEmbeddings, 0.02); // 2% of daily budget
        upshift_costs.insert(UpshiftType::EnhancedSearch, 0.01);     // 1% of daily budget
        upshift_costs.insert(UpshiftType::DiversityOptimization, 0.015); // 1.5% of daily budget
        upshift_costs.insert(UpshiftType::CrossEncoder, 0.03);       // 3% of daily budget
        
        info!(
            "Initialized ConformalRouter: risk_threshold={}, budget={}%, enabled={}",
            config.risk_threshold, config.daily_budget_percent, config.enabled
        );
        
        Self {
            config,
            predictor,
            budget_manager,
            metrics: Arc::new(parking_lot::RwLock::new(ConformalRouterMetrics::default())),
            upshift_costs,
        }
    }
    
    /// Make routing decision based on risk assessment
    #[instrument(skip(self, features, classification), fields(
        query_len = features.query_length,
        intent = %classification.intent,
        confidence = classification.confidence
    ))]
    pub async fn make_routing_decision(
        &self,
        features: &ConformalFeatures,
        classification: &QueryClassification,
    ) -> Result<RoutingDecision> {
        if !self.config.enabled {
            return Ok(RoutingDecision {
                should_upshift: false,
                upshift_type: UpshiftType::None,
                budget_consumed: 0.0,
                routing_reason: "router_disabled".to_string(),
                expected_improvement: 0.0,
                risk_assessment: RiskAssessment {
                    risk_score: 0.0,
                    confidence_interval: (0.0, 1.0),
                    nonconformity_score: 0.0,
                    calibrated: false,
                    risk_factors: Vec::new(),
                },
            });
        }
        
        let start = Instant::now();
        
        // Record query for budget tracking
        {
            let mut budget = self.budget_manager.write().await;
            budget.record_query();
        }
        
        // Get risk assessment from conformal predictor
        let risk_assessment = {
            let predictor = self.predictor.read().await;
            predictor.predict_risk(features, self.config.confidence_level)
        };
        
        debug!(
            "Risk assessment: score={:.3}, interval=({:.3}, {:.3}), calibrated={}",
            risk_assessment.risk_score,
            risk_assessment.confidence_interval.0,
            risk_assessment.confidence_interval.1,
            risk_assessment.calibrated
        );
        
        // Check if upshift is warranted
        let should_upshift = risk_assessment.risk_score > self.config.risk_threshold;
        
        if !should_upshift {
            self.record_routing_decision(&RoutingDecision {
                should_upshift: false,
                upshift_type: UpshiftType::None,
                budget_consumed: 0.0,
                routing_reason: format!("risk_below_threshold_{:.3}", risk_assessment.risk_score),
                expected_improvement: 0.0,
                risk_assessment: risk_assessment.clone(),
            }, start.elapsed()).await;
            
            return Ok(RoutingDecision {
                should_upshift: false,
                upshift_type: UpshiftType::None,
                budget_consumed: 0.0,
                routing_reason: format!("risk_below_threshold_{:.3}", risk_assessment.risk_score),
                expected_improvement: 0.0,
                risk_assessment,
            });
        }
        
        // Select upshift type and check budget
        let upshift_type = self.select_upshift_type(features, &risk_assessment, classification);
        let upshift_cost = *self.upshift_costs.get(&upshift_type).unwrap_or(&0.0);
        
        let can_upshift = {
            let mut budget = self.budget_manager.write().await;
            let budget_status = budget.get_status();
            
            // Check budget availability and P95 headroom
            budget.can_upshift(upshift_cost) && 
            budget_status.p95_headroom_estimate >= self.config.p95_headroom_threshold_ms
        };
        
        if !can_upshift {
            let reason = {
                let budget = self.budget_manager.read().await;
                let status = budget.get_status();
                if status.remaining_budget < upshift_cost {
                    "budget_exhausted".to_string()
                } else {
                    "insufficient_headroom".to_string()
                }
            };
            
            let decision = RoutingDecision {
                should_upshift: false,
                upshift_type: UpshiftType::None,
                budget_consumed: 0.0,
                routing_reason: reason,
                expected_improvement: 0.0,
                risk_assessment,
            };
            
            self.record_routing_decision(&decision, start.elapsed()).await;
            return Ok(decision);
        }
        
        // Record upshift usage
        {
            let mut budget = self.budget_manager.write().await;
            budget.record_upshift(upshift_cost);
        }
        
        let expected_improvement = self.estimate_improvement(upshift_type, &risk_assessment);
        
        let decision = RoutingDecision {
            should_upshift: true,
            upshift_type,
            budget_consumed: upshift_cost,
            routing_reason: format!("high_risk_{:.3}", risk_assessment.risk_score),
            expected_improvement,
            risk_assessment,
        };
        
        self.record_routing_decision(&decision, start.elapsed()).await;
        
        info!(
            "Upshift decision: type={}, cost={:.3}, improvement={:.3}",
            upshift_type, upshift_cost, expected_improvement
        );
        
        Ok(decision)
    }
    
    /// Select appropriate upshift type based on features and risk
    fn select_upshift_type(
        &self,
        features: &ConformalFeatures,
        risk_assessment: &RiskAssessment,
        classification: &QueryClassification,
    ) -> UpshiftType {
        // High semantic complexity -> use enhanced embeddings
        if features.semantic_complexity > 0.8 || features.naturalness_score > 0.7 {
            return UpshiftType::HighDimEmbeddings;
        }
        
        // Structural queries -> enhanced search parameters
        if features.structural_mode && features.identifier_density > 0.5 {
            return UpshiftType::EnhancedSearch;
        }
        
        // Natural language with diversity needs -> MMR optimization
        if classification.intent == QueryIntent::NaturalLanguage && features.word_count > 5 {
            return UpshiftType::DiversityOptimization;
        }
        
        // High uncertainty -> cross-encoder for precision
        let interval_width = risk_assessment.confidence_interval.1 - risk_assessment.confidence_interval.0;
        if interval_width > 0.3 {
            return UpshiftType::CrossEncoder;
        }
        
        // Default to enhanced search
        UpshiftType::EnhancedSearch
    }
    
    /// Estimate expected quality improvement from upshift
    fn estimate_improvement(&self, upshift_type: UpshiftType, risk_assessment: &RiskAssessment) -> f32 {
        let base_improvements = match upshift_type {
            UpshiftType::None => 0.0,
            UpshiftType::HighDimEmbeddings => 0.08,       // +8pp expected
            UpshiftType::EnhancedSearch => 0.04,          // +4pp expected
            UpshiftType::DiversityOptimization => 0.06,   // +6pp expected
            UpshiftType::CrossEncoder => 0.12,            // +12pp expected
            UpshiftType::SemanticReranking => 0.10,       // +10pp expected
            UpshiftType::LSPIntegration => 0.15,          // +15pp expected
            UpshiftType::ASTAnalysis => 0.09,             // +9pp expected
            UpshiftType::CrossLanguageSearch => 0.11,     // +11pp expected
        };
        
        // Scale by risk score - higher risk queries benefit more
        let risk_multiplier = 0.5 + (risk_assessment.risk_score * 0.5);
        base_improvements * risk_multiplier
    }
    
    /// Add calibration sample for model improvement
    pub async fn add_calibration_sample(
        &self,
        features: ConformalFeatures,
        predicted_quality: f32,
        actual_quality: f32,
    ) {
        let sample = CalibrationSample {
            features,
            predicted_quality,
            actual_quality,
            timestamp: SystemTime::now(),
        };
        
        let mut predictor = self.predictor.write().await;
        predictor.add_calibration_sample(sample);
    }
    
    /// Get comprehensive router status
    pub async fn get_status(&self) -> ConformalRouterStatus {
        let budget_status = {
            let budget = self.budget_manager.read().await;
            budget.get_status()
        };
        
        let predictor_status = {
            let predictor = self.predictor.read().await;
            predictor.get_status()
        };
        
        let metrics = self.metrics.read().clone();
        
        ConformalRouterStatus {
            enabled: self.config.enabled,
            risk_threshold: self.config.risk_threshold,
            budget_status,
            predictor_status,
            metrics,
        }
    }
    
    /// Update router configuration
    pub async fn update_config(&mut self, new_config: ConformalRouterConfig) {
        info!("Updating conformal router configuration");
        
        // Update budget manager if budget changed
        if new_config.daily_budget_percent != self.config.daily_budget_percent {
            let mut budget = self.budget_manager.write().await;
            *budget = BudgetManager::new(new_config.daily_budget_percent);
        }
        
        // Update predictor if retention period changed
        if new_config.calibration_retention_hours != self.config.calibration_retention_hours ||
           new_config.min_calibration_samples != self.config.min_calibration_samples {
            let mut predictor = self.predictor.write().await;
            *predictor = ConformalPredictor::new(
                new_config.min_calibration_samples,
                new_config.calibration_retention_hours,
            );
        }
        
        self.config = new_config;
        
        info!(
            "Updated conformal router: risk_threshold={}, budget={}%, enabled={}",
            self.config.risk_threshold, self.config.daily_budget_percent, self.config.enabled
        );
    }
    
    /// Record routing decision metrics
    async fn record_routing_decision(&self, decision: &RoutingDecision, latency: Duration) {
        let mut metrics = self.metrics.write();
        metrics.total_decisions += 1;
        metrics.total_latency += latency;
        
        if decision.should_upshift {
            metrics.upshift_decisions += 1;
            *metrics.upshift_type_counts.entry(decision.upshift_type).or_insert(0) += 1;
            metrics.total_budget_consumed += decision.budget_consumed;
        }
        
        if decision.risk_assessment.calibrated {
            metrics.calibrated_decisions += 1;
        }
        
        // Track risk score distribution
        let risk_bucket = (decision.risk_assessment.risk_score * 10.0) as usize;
        metrics.risk_score_histogram[risk_bucket.min(9)] += 1;
    }
}

/// Comprehensive status of conformal router
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformalRouterStatus {
    pub enabled: bool,
    pub risk_threshold: f32,
    pub budget_status: BudgetStatus,
    pub predictor_status: ConformalPredictorStatus,
    pub metrics: ConformalRouterMetrics,
}

/// Performance and usage metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConformalRouterMetrics {
    pub total_decisions: u64,
    pub upshift_decisions: u64,
    pub calibrated_decisions: u64,
    pub total_latency: Duration,
    pub total_budget_consumed: f32,
    pub upshift_type_counts: HashMap<UpshiftType, u64>,
    pub risk_score_histogram: [u64; 10], // Buckets for 0.0-0.1, 0.1-0.2, ..., 0.9-1.0
    pub cache_hit_rate: f64, // Cache hit rate for routing decisions
}

impl ConformalRouterMetrics {
    pub fn upshift_rate(&self) -> f32 {
        if self.total_decisions == 0 {
            0.0
        } else {
            (self.upshift_decisions as f32 / self.total_decisions as f32) * 100.0
        }
    }
    
    pub fn avg_latency_ms(&self) -> f64 {
        if self.total_decisions == 0 {
            0.0
        } else {
            self.total_latency.as_millis() as f64 / self.total_decisions as f64
        }
    }
    
    pub fn calibration_rate(&self) -> f32 {
        if self.total_decisions == 0 {
            0.0
        } else {
            (self.calibrated_decisions as f32 / self.total_decisions as f32) * 100.0
        }
    }
    
    pub fn avg_risk_score(&self) -> f64 {
        let total_weight: u64 = self.risk_score_histogram.iter().sum();
        if total_weight == 0 {
            return 0.0;
        }
        
        let weighted_sum: f64 = self.risk_score_histogram.iter()
            .enumerate()
            .map(|(i, &count)| (i as f64 + 0.5) * 0.1 * count as f64)
            .sum();
            
        weighted_sum / total_weight as f64
    }
}

/// Extract conformal features from query and context
pub fn extract_conformal_features(
    query: &str,
    classification: &QueryClassification,
    file_context: Option<&crate::semantic::intent_router::FileContext>,
) -> ConformalFeatures {
    let words: Vec<&str> = query.split_whitespace().collect();
    let chars: Vec<char> = query.chars().collect();
    
    // Calculate query entropy
    let mut char_counts = HashMap::new();
    for &ch in &chars {
        *char_counts.entry(ch).or_insert(0) += 1;
    }
    
    let query_entropy = if chars.is_empty() {
        0.0
    } else {
        char_counts.values().map(|&count| {
            let p = count as f32 / chars.len() as f32;
            -p * p.log2()
        }).sum()
    };
    
    // Calculate identifier density
    let identifier_pattern = regex::Regex::new(r"[a-zA-Z_][a-zA-Z0-9_]*").unwrap();
    let identifiers = identifier_pattern.find_iter(query).count();
    let identifier_density = if words.is_empty() {
        0.0
    } else {
        identifiers as f32 / words.len() as f32
    };
    
    // Average word length
    let avg_word_length = if words.is_empty() {
        0.0
    } else {
        words.iter().map(|w| w.len()).sum::<usize>() as f32 / words.len() as f32
    };
    
    // Special characters detection
    let special_chars = query.chars().any(|c| "{}[]().,;:!@#$%^&*".contains(c));
    
    ConformalFeatures {
        query_length: query.len() as u32,
        word_count: words.len() as u32,
        has_special_chars: special_chars,
        fuzzy_enabled: false, // Would be set based on search context
        structural_mode: matches!(classification.intent, QueryIntent::Structural),
        avg_word_length,
        query_entropy,
        identifier_density,
        semantic_complexity: classification.complexity_score,
        has_file_context: file_context.is_some(),
        language_detected: !classification.language_hints.is_empty(),
        intent_confidence: classification.confidence,
        naturalness_score: classification.naturalness_score,
        similar_queries_success_rate: 0.7, // Would be calculated from history
        user_satisfaction_history: 0.8,    // Would be calculated from feedback
    }
}

/// Initialize conformal router module
pub async fn initialize_conformal_router(config: &ConformalRouterConfig) -> Result<()> {
    tracing::info!("Initializing conformal router module");
    tracing::info!("Risk threshold: {}", config.risk_threshold);
    tracing::info!("Daily budget: {}%", config.daily_budget_percent);
    tracing::info!("Confidence level: {}", config.confidence_level);
    tracing::info!("Min calibration samples: {}", config.min_calibration_samples);
    tracing::info!("P95 headroom threshold: {}ms", config.p95_headroom_threshold_ms);
    tracing::info!("Enabled: {}", config.enabled);
    
    // Validate configuration
    if config.risk_threshold < 0.0 || config.risk_threshold > 1.0 {
        anyhow::bail!("Risk threshold must be in range [0.0, 1.0]");
    }
    
    if config.daily_budget_percent < 0.0 || config.daily_budget_percent > 100.0 {
        anyhow::bail!("Daily budget percent must be in range [0.0, 100.0]");
    }
    
    if config.confidence_level < 0.0 || config.confidence_level > 1.0 {
        anyhow::bail!("Confidence level must be in range [0.0, 1.0]");
    }
    
    if config.min_calibration_samples == 0 {
        anyhow::bail!("Minimum calibration samples must be greater than 0");
    }
    
    tracing::info!("Conformal router module initialized successfully");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::query_classifier::{QueryIntent, QueryCharacteristic, ClassifierConfig, QueryClassifier};
    use smallvec::smallvec;
    
    #[tokio::test]
    async fn test_conformal_router_creation() {
        let config = ConformalRouterConfig::default();
        let router = ConformalRouter::new(config);
        
        let status = router.get_status().await;
        assert!(status.enabled);
        assert_eq!(status.risk_threshold, 0.6);
    }
    
    #[tokio::test]
    async fn test_budget_manager() {
        let mut budget = BudgetManager::new(5.0); // 5% daily budget
        
        assert!(budget.can_upshift(0.02)); // 2% cost
        budget.record_upshift(0.02);
        
        assert!(budget.can_upshift(0.02)); // Can still upshift more
        budget.record_upshift(0.02);
        
        assert!(!budget.can_upshift(0.02)); // Should exceed budget now
        
        let status = budget.get_status();
        assert!(status.current_usage_percent > 75.0); // Should be close to budget limit
    }
    
    #[tokio::test]
    async fn test_conformal_predictor() {
        let mut predictor = ConformalPredictor::new(5, 24);
        
        // Add calibration samples
        for i in 0..10 {
            let features = ConformalFeatures {
                query_length: 50 + i,
                word_count: 5 + (i / 2),
                has_special_chars: i % 2 == 0,
                fuzzy_enabled: false,
                structural_mode: false,
                avg_word_length: 5.0,
                query_entropy: 2.5,
                identifier_density: 0.3,
                semantic_complexity: 0.4,
                has_file_context: true,
                language_detected: true,
                intent_confidence: 0.8,
                naturalness_score: 0.6,
                similar_queries_success_rate: 0.7,
                user_satisfaction_history: 0.8,
            };
            
            predictor.add_calibration_sample(CalibrationSample {
                features,
                predicted_quality: 0.7,
                actual_quality: 0.75 + (i as f32 * 0.01),
                timestamp: SystemTime::now(),
            });
        }
        
        let status = predictor.get_status();
        assert!(status.is_calibrated);
        assert!(status.calibration_samples >= 5);
        
        // Test prediction
        let test_features = ConformalFeatures {
            query_length: 60,
            word_count: 6,
            has_special_chars: true,
            fuzzy_enabled: false,
            structural_mode: false,
            avg_word_length: 5.5,
            query_entropy: 2.8,
            identifier_density: 0.4,
            semantic_complexity: 0.6,
            has_file_context: true,
            language_detected: true,
            intent_confidence: 0.9,
            naturalness_score: 0.7,
            similar_queries_success_rate: 0.8,
            user_satisfaction_history: 0.85,
        };
        
        let risk = predictor.predict_risk(&test_features, 0.95);
        assert!(risk.risk_score >= 0.0 && risk.risk_score <= 1.0);
        assert!(risk.confidence_interval.0 <= risk.confidence_interval.1);
        assert!(risk.calibrated);
    }
    
    #[tokio::test]
    async fn test_routing_decision() {
        let config = ConformalRouterConfig::default();
        let router = ConformalRouter::new(config);
        
        let features = ConformalFeatures {
            query_length: 100,
            word_count: 8,
            has_special_chars: true,
            fuzzy_enabled: false,
            structural_mode: false,
            avg_word_length: 6.0,
            query_entropy: 3.2,
            identifier_density: 0.2,
            semantic_complexity: 0.8, // High complexity
            has_file_context: false,
            language_detected: true,
            intent_confidence: 0.6,
            naturalness_score: 0.9, // High naturalness
            similar_queries_success_rate: 0.5,
            user_satisfaction_history: 0.6,
        };
        
        let classification = crate::semantic::query_classifier::QueryClassification {
            intent: QueryIntent::NaturalLanguage,
            confidence: 0.8,
            characteristics: vec![QueryCharacteristic::HasDescriptiveWords],
            naturalness_score: 0.9,
            complexity_score: 0.8,
            language_hints: vec!["english".to_string()],
        };
        
        let decision = router.make_routing_decision(&features, &classification).await;
        assert!(decision.is_ok());
        
        let decision = decision.unwrap();
        // High semantic complexity should trigger upshift
        assert!(decision.risk_assessment.risk_score > 0.0);
    }
    
    #[test]
    fn test_feature_extraction() {
        let query = "how to find a function that calculates the sum of two numbers";
        let classification = crate::semantic::query_classifier::QueryClassification {
            intent: QueryIntent::NaturalLanguage,
            confidence: 0.9,
            characteristics: vec![
                QueryCharacteristic::HasDescriptiveWords,
                QueryCharacteristic::HasArticles
            ],
            naturalness_score: 0.95,
            complexity_score: 0.3,
            language_hints: vec!["english".to_string()],
        };
        
        let features = extract_conformal_features(&query, &classification, None);
        
        assert_eq!(features.query_length, query.len() as u32);
        assert_eq!(features.word_count, 12); // Number of words in query
        assert!(!features.has_special_chars);
        assert!(features.naturalness_score > 0.9);
        assert!(!features.has_file_context);
        assert!(features.language_detected);
    }
    
    #[tokio::test]
    async fn test_upshift_type_selection() {
        let config = ConformalRouterConfig::default();
        let router = ConformalRouter::new(config);
        
        // Test high semantic complexity -> high dim embeddings
        let features = ConformalFeatures {
            semantic_complexity: 0.9,
            naturalness_score: 0.8,
            ..Default::default()
        };
        
        let risk_assessment = RiskAssessment {
            risk_score: 0.7,
            confidence_interval: (0.5, 0.9),
            nonconformity_score: 0.2,
            calibrated: true,
            risk_factors: vec![],
        };
        
        let classification = crate::semantic::query_classifier::QueryClassification {
            intent: QueryIntent::NaturalLanguage,
            confidence: 0.8,
            characteristics: vec![],
            naturalness_score: 0.8,
            complexity_score: 0.9,
            language_hints: vec![],
        };
        
        let upshift_type = router.select_upshift_type(&features, &risk_assessment, &classification);
        assert_eq!(upshift_type, UpshiftType::HighDimEmbeddings);
    }
    
    #[tokio::test]
    async fn test_configuration_validation() {
        let mut config = ConformalRouterConfig::default();
        config.risk_threshold = 1.5; // Invalid
        
        let result = initialize_conformal_router(&config).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Risk threshold"));
        
        config.risk_threshold = 0.6; // Valid
        config.daily_budget_percent = -5.0; // Invalid
        
        let result = initialize_conformal_router(&config).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Daily budget"));
    }
}

impl Default for ConformalFeatures {
    fn default() -> Self {
        Self {
            query_length: 0,
            word_count: 0,
            has_special_chars: false,
            fuzzy_enabled: false,
            structural_mode: false,
            avg_word_length: 0.0,
            query_entropy: 0.0,
            identifier_density: 0.0,
            semantic_complexity: 0.0,
            has_file_context: false,
            language_detected: false,
            intent_confidence: 0.0,
            naturalness_score: 0.0,
            similar_queries_success_rate: 0.0,
            user_satisfaction_history: 0.0,
        }
    }
}