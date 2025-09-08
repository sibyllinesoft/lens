//! # Calibration Preservation System
//!
//! Ensures semantic features don't cause calibration shock:
//! - Cap dense features in log-odds space to prevent extreme predictions
//! - Maintain Expected Calibration Error (ECE) ≤ 0.005 drift from baseline
//! - Temperature scaling and isotonic calibration as safety mechanisms
//! - Monitor calibration across query types and languages

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Calibration preservation system for semantic search
pub struct CalibrationSystem {
    config: CalibrationConfig,
    /// Baseline calibration measurements (pre-semantic)
    baseline_calibration: Arc<RwLock<Option<CalibrationMeasurement>>>,
    /// Current calibration state
    current_calibration: Arc<RwLock<CalibrationMeasurement>>,
    /// Temperature scaling models per query type/language
    temperature_models: Arc<RwLock<HashMap<String, TemperatureModel>>>,
    /// Feature scaling parameters
    feature_scalers: Arc<RwLock<FeatureScalerSet>>,
    /// Calibration history for drift detection
    calibration_history: Arc<RwLock<Vec<CalibrationSnapshot>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// Maximum ECE drift allowed from baseline
    pub max_ece_drift: f32,
    /// Cap for dense features in log-odds space
    pub log_odds_cap: f32,
    /// Default temperature scaling factor
    pub temperature: f32,
    /// Minimum samples for calibration measurement
    pub min_samples_for_calibration: usize,
    /// Calibration measurement window size
    pub measurement_window_size: usize,
    /// Enable automatic temperature adjustment
    pub auto_temperature_adjustment: bool,
}

/// Calibration measurement with confidence intervals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationMeasurement {
    /// Expected Calibration Error
    pub ece: f32,
    /// Calibration error confidence interval
    pub ece_ci_lower: f32,
    pub ece_ci_upper: f32,
    /// Maximum Calibration Error
    pub mce: f32,
    /// Reliability diagram data points
    pub reliability_data: Vec<ReliabilityPoint>,
    /// Sample count for measurement
    pub sample_count: usize,
    /// Measurement timestamp
    pub timestamp: std::time::SystemTime,
    /// Query type breakdown
    pub by_query_type: HashMap<String, f32>,
    /// Language breakdown
    pub by_language: HashMap<String, f32>,
}

/// Point in reliability diagram
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityPoint {
    /// Predicted probability bin
    pub predicted_prob: f32,
    /// Actual positive rate in this bin
    pub actual_positive_rate: f32,
    /// Sample count in bin
    pub bin_count: usize,
    /// Confidence interval for actual rate
    pub confidence_interval: (f32, f32),
}

/// Temperature scaling model for post-hoc calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureModel {
    /// Learned temperature parameter
    pub temperature: f32,
    /// Bias term
    pub bias: f32,
    /// Model performance metrics
    pub ece_before: f32,
    pub ece_after: f32,
    /// Training sample count
    pub training_samples: usize,
    /// Model scope (query_type:language)
    pub scope: String,
}

/// Feature scaling to prevent calibration shock
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureScalerSet {
    /// Log-odds caps for semantic features
    pub semantic_feature_caps: HashMap<String, f32>,
    /// Linear scaling factors
    pub linear_scalers: HashMap<String, LinearScaler>,
    /// Quantile-based robust scalers
    pub robust_scalers: HashMap<String, RobustScaler>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearScaler {
    pub mean: f32,
    pub std: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustScaler {
    pub median: f32,
    pub iqr: f32, // Interquartile range
}

/// Snapshot of calibration at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationSnapshot {
    pub timestamp: std::time::SystemTime,
    pub ece: f32,
    pub mce: f32,
    pub sample_count: usize,
    pub semantic_features_active: bool,
    pub temperature: f32,
}

/// Prediction with confidence for calibration analysis
#[derive(Debug, Clone)]
pub struct CalibratedPrediction {
    pub raw_score: f32,
    pub calibrated_score: f32,
    pub confidence: f32,
    pub query_type: String,
    pub language: Option<String>,
    pub features_used: Vec<String>,
}

/// Training sample for calibration
#[derive(Debug, Clone)]
pub struct CalibrationSample {
    pub prediction: f32,
    pub actual_relevance: f32, // 0.0 or 1.0 for binary relevance
    pub query_type: String,
    pub language: Option<String>,
}

impl CalibrationSystem {
    /// Create new calibration system
    pub async fn new(config: CalibrationConfig) -> Result<Self> {
        info!("Creating calibration preservation system");
        info!("ECE drift limit: {:.4}, log-odds cap: {:.2}, temperature: {:.2}", 
              config.max_ece_drift, config.log_odds_cap, config.temperature);
        
        Ok(Self {
            config,
            baseline_calibration: Arc::new(RwLock::new(None)),
            current_calibration: Arc::new(RwLock::new(CalibrationMeasurement::default())),
            temperature_models: Arc::new(RwLock::new(HashMap::new())),
            feature_scalers: Arc::new(RwLock::new(FeatureScalerSet::default())),
            calibration_history: Arc::new(RwLock::new(Vec::new())),
        })
    }
    
    /// Establish baseline calibration before semantic features
    pub async fn establish_baseline(&self, samples: &[CalibrationSample]) -> Result<()> {
        info!("Establishing baseline calibration with {} samples", samples.len());
        
        if samples.len() < self.config.min_samples_for_calibration {
            anyhow::bail!("Insufficient samples for baseline: {} < {}", 
                         samples.len(), self.config.min_samples_for_calibration);
        }
        
        // Calculate baseline calibration
        let measurement = self.calculate_calibration_measurement(samples).await?;
        
        info!("Baseline calibration established: ECE = {:.4}, MCE = {:.4}", 
              measurement.ece, measurement.mce);
        
        *self.baseline_calibration.write().await = Some(measurement.clone());
        // Add to history before moving measurement
        self.add_calibration_snapshot(measurement.ece, measurement.mce, false, self.config.temperature).await;
        
        // Store current measurement
        *self.current_calibration.write().await = measurement;
        
        Ok(())
    }
    
    /// Apply calibration-aware scaling to features
    pub async fn scale_features(&self, features: &HashMap<String, f32>) -> Result<HashMap<String, f32>> {
        let scalers = self.feature_scalers.read().await;
        let mut scaled_features = HashMap::new();
        
        for (name, value) in features {
            let scaled_value = if name.contains("semantic") {
                // Apply log-odds capping for semantic features
                let capped_value = if let Some(&cap) = scalers.semantic_feature_caps.get(name) {
                    value.clamp(-cap, cap)
                } else {
                    value.clamp(-self.config.log_odds_cap, self.config.log_odds_cap)
                };
                
                // Apply robust scaling if available
                if let Some(robust_scaler) = scalers.robust_scalers.get(name) {
                    (capped_value - robust_scaler.median) / (robust_scaler.iqr + 1e-8)
                } else {
                    capped_value
                }
            } else {
                // Standard scaling for non-semantic features
                if let Some(linear_scaler) = scalers.linear_scalers.get(name) {
                    (value - linear_scaler.mean) / (linear_scaler.std + 1e-8)
                } else {
                    *value
                }
            };
            
            scaled_features.insert(name.clone(), scaled_value);
        }
        
        debug!("Scaled {} features with calibration preservation", scaled_features.len());
        Ok(scaled_features)
    }
    
    /// Apply temperature scaling to prediction
    pub async fn apply_temperature_scaling(&self, prediction: f32, query_type: &str, language: Option<&str>) -> Result<f32> {
        let models = self.temperature_models.read().await;
        
        // Try to find specific model for query type + language
        let scope_key = match language {
            Some(lang) => format!("{}:{}", query_type, lang),
            None => query_type.to_string(),
        };
        
        let temperature = if let Some(model) = models.get(&scope_key) {
            model.temperature
        } else if let Some(model) = models.get(query_type) {
            model.temperature
        } else {
            self.config.temperature
        };
        
        // Apply temperature scaling: p_calibrated = sigmoid(logit(p) / T)
        let logit_p = (prediction / (1.0 - prediction + 1e-8)).ln();
        let scaled_logit = logit_p / temperature;
        let calibrated_p = 1.0 / (1.0 + (-scaled_logit).exp());
        
        Ok(calibrated_p.clamp(0.001, 0.999)) // Avoid extreme probabilities
    }
    
    /// Update calibration measurements with new samples
    pub async fn update_calibration(&self, samples: &[CalibrationSample]) -> Result<CalibrationStatus> {
        if samples.len() < self.config.min_samples_for_calibration / 4 {
            return Ok(CalibrationStatus::InsufficientData);
        }
        
        // Calculate current calibration
        let current_measurement = self.calculate_calibration_measurement(samples).await?;
        
        // Check for drift from baseline
        let baseline_guard = self.baseline_calibration.read().await;
        let drift_status = if let Some(baseline) = baseline_guard.as_ref() {
            let ece_drift = (current_measurement.ece - baseline.ece).abs();
            
            if ece_drift > self.config.max_ece_drift {
                warn!("Calibration drift detected: ECE drift {:.4} > limit {:.4}", 
                      ece_drift, self.config.max_ece_drift);
                CalibrationStatus::DriftDetected { 
                    ece_drift,
                    current_ece: current_measurement.ece,
                    baseline_ece: baseline.ece,
                }
            } else {
                CalibrationStatus::WithinLimits {
                    ece_drift,
                    current_ece: current_measurement.ece,
                }
            }
        } else {
            CalibrationStatus::NoBaseline
        };
        
        // Update current measurement
        *self.current_calibration.write().await = current_measurement.clone();
        
        // Add to history
        self.add_calibration_snapshot(current_measurement.ece, current_measurement.mce, true, self.config.temperature).await;
        
        // Trigger automatic temperature adjustment if needed
        if matches!(drift_status, CalibrationStatus::DriftDetected { .. }) && self.config.auto_temperature_adjustment {
            self.adjust_temperature_models(samples).await?;
        }
        
        info!("Calibration updated: ECE = {:.4}, status = {:?}", 
              current_measurement.ece, drift_status);
        
        Ok(drift_status)
    }
    
    /// Train temperature scaling models
    pub async fn train_temperature_models(&self, samples: &[CalibrationSample]) -> Result<()> {
        info!("Training temperature scaling models with {} samples", samples.len());
        
        // Group samples by query type and language
        let mut grouped_samples: HashMap<String, Vec<&CalibrationSample>> = HashMap::new();
        
        for sample in samples {
            let key = match &sample.language {
                Some(lang) => format!("{}:{}", sample.query_type, lang),
                None => sample.query_type.clone(),
            };
            grouped_samples.entry(key.clone()).or_default().push(sample);
            
            // Also add to query type only group
            grouped_samples.entry(sample.query_type.clone()).or_default().push(sample);
        }
        
        let mut models = self.temperature_models.write().await;
        
        for (scope, group_samples) in grouped_samples {
            if group_samples.len() >= self.config.min_samples_for_calibration / 4 {
                let model = self.fit_temperature_model(group_samples, &scope).await?;
                models.insert(scope, model);
            }
        }
        
        info!("Trained {} temperature models", models.len());
        Ok(())
    }
    
    /// Get current calibration metrics
    pub async fn get_calibration_metrics(&self) -> CalibrationMetrics {
        let current = self.current_calibration.read().await;
        let baseline = self.baseline_calibration.read().await;
        let history = self.calibration_history.read().await;
        
        let ece_drift = if let Some(baseline_measurement) = baseline.as_ref() {
            (current.ece - baseline_measurement.ece).abs()
        } else {
            0.0
        };
        
        let recent_trend = if history.len() >= 5 {
            let recent_eces: Vec<f32> = history.iter().rev().take(5).map(|s| s.ece).collect();
            let first = recent_eces[4];
            let last = recent_eces[0];
            (last - first) / 5.0 // Average change per measurement
        } else {
            0.0
        };
        
        CalibrationMetrics {
            current_ece: current.ece,
            baseline_ece: baseline.as_ref().map(|b| b.ece).unwrap_or(0.0),
            ece_drift,
            max_allowed_drift: self.config.max_ece_drift,
            within_limits: ece_drift <= self.config.max_ece_drift,
            recent_trend,
            measurement_count: history.len(),
            sample_count: current.sample_count,
        }
    }
    
    // Private implementation methods
    
    fn calculate_calibration_measurement<'a>(&'a self, samples: &'a [CalibrationSample]) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<CalibrationMeasurement>> + Send + 'a>> {
        Box::pin(async move {
        let num_bins = 10;
        let mut bins = vec![Vec::new(); num_bins];
        
        // Assign samples to bins based on predicted probability
        for sample in samples {
            let bin_idx = ((sample.prediction * num_bins as f32) as usize).min(num_bins - 1);
            bins[bin_idx].push(sample);
        }
        
        // Calculate reliability points
        let mut reliability_data = Vec::new();
        let mut total_ece = 0.0;
        let mut total_samples = 0;
        let mut max_calibration_error = 0.0;
        
        for (i, bin) in bins.iter().enumerate() {
            if bin.is_empty() {
                continue;
            }
            
            let bin_center = (i as f32 + 0.5) / num_bins as f32;
            let actual_positive_rate = bin.iter()
                .map(|s| s.actual_relevance)
                .sum::<f32>() / bin.len() as f32;
            
            let bin_error = (bin_center - actual_positive_rate).abs();
            let bin_weight = bin.len() as f32 / samples.len() as f32;
            
            total_ece += bin_error * bin_weight;
            max_calibration_error = f32::max(max_calibration_error, bin_error);
            total_samples += bin.len();
            
            // Calculate confidence interval (Wilson score interval)
            let (ci_lower, ci_upper) = self.calculate_wilson_confidence_interval(
                actual_positive_rate, bin.len()
            );
            
            reliability_data.push(ReliabilityPoint {
                predicted_prob: bin_center,
                actual_positive_rate,
                bin_count: bin.len(),
                confidence_interval: (ci_lower, ci_upper),
            });
        }
        
        // Calculate ECE confidence interval
        let ece_std_error = (total_ece * (1.0 - total_ece) / samples.len() as f32).sqrt();
        let ece_ci_lower = (total_ece - 1.96 * ece_std_error).max(0.0);
        let ece_ci_upper = (total_ece + 1.96 * ece_std_error).min(1.0);
        
        // Calculate breakdown by query type and language
        let by_query_type = self.calculate_ece_by_category(samples, |s| &s.query_type).await?;
        let by_language = self.calculate_ece_by_language(samples).await?;
        
        Ok(CalibrationMeasurement {
            ece: total_ece,
            ece_ci_lower,
            ece_ci_upper,
            mce: max_calibration_error,
            reliability_data,
            sample_count: samples.len(),
            timestamp: std::time::SystemTime::now(),
            by_query_type,
            by_language,
        })
        })
    }
    
    fn calculate_wilson_confidence_interval(&self, p: f32, n: usize) -> (f32, f32) {
        if n == 0 {
            return (0.0, 1.0);
        }
        
        let z = 1.96; // 95% confidence
        let n_f = n as f32;
        
        let center = p + z * z / (2.0 * n_f);
        let half_width = z * ((p * (1.0 - p) / n_f) + (z * z / (4.0 * n_f * n_f))).sqrt();
        let denominator = 1.0 + z * z / n_f;
        
        let lower = ((center - half_width) / denominator).max(0.0);
        let upper = ((center + half_width) / denominator).min(1.0);
        
        (lower, upper)
    }
    
    async fn calculate_ece_by_category<F>(&self, samples: &[CalibrationSample], category_fn: F) -> Result<HashMap<String, f32>>
    where
        F: Fn(&CalibrationSample) -> &String,
    {
        let mut category_groups: HashMap<String, Vec<&CalibrationSample>> = HashMap::new();
        
        for sample in samples {
            let category = category_fn(sample);
            category_groups.entry(category.clone()).or_default().push(sample);
        }
        
        let mut category_eces = HashMap::new();
        
        for (category, group) in category_groups {
            if group.len() >= 10 { // Minimum samples for reliable ECE
                let group_samples: Vec<CalibrationSample> = group.into_iter().cloned().collect();
                let measurement = self.calculate_calibration_measurement(&group_samples).await?;
                category_eces.insert(category, measurement.ece);
            }
        }
        
        Ok(category_eces)
    }
    
    async fn calculate_ece_by_language(&self, samples: &[CalibrationSample]) -> Result<HashMap<String, f32>> {
        let mut language_groups: HashMap<String, Vec<&CalibrationSample>> = HashMap::new();
        
        for sample in samples {
            let language = sample.language.as_deref().unwrap_or("unknown");
            language_groups.entry(language.to_string()).or_default().push(sample);
        }
        
        let mut language_eces = HashMap::new();
        
        for (language, group) in language_groups {
            if group.len() >= 10 {
                let group_samples: Vec<CalibrationSample> = group.into_iter().cloned().collect();
                let measurement = self.calculate_calibration_measurement(&group_samples).await?;
                language_eces.insert(language, measurement.ece);
            }
        }
        
        Ok(language_eces)
    }
    
    async fn fit_temperature_model(&self, samples: Vec<&CalibrationSample>, scope: &str) -> Result<TemperatureModel> {
        // Calculate ECE before temperature scaling
        let samples_owned: Vec<CalibrationSample> = samples.into_iter().cloned().collect();
        let before_measurement = self.calculate_calibration_measurement(&samples_owned).await?;
        
        // Use simple grid search to find optimal temperature
        let mut best_temperature = 1.0;
        let mut best_ece = before_measurement.ece;
        
        for temp_candidate in (5..=200).step_by(5) {
            let temperature = temp_candidate as f32 / 100.0; // 0.05 to 2.0
            
            // Apply temperature scaling and calculate ECE
            let calibrated_samples: Vec<CalibrationSample> = samples_owned.iter()
                .map(|s| {
                    let logit_p = (s.prediction / (1.0 - s.prediction + 1e-8)).ln();
                    let scaled_logit = logit_p / temperature;
                    let calibrated_p = 1.0 / (1.0 + (-scaled_logit).exp());
                    
                    CalibrationSample {
                        prediction: calibrated_p.clamp(0.001, 0.999),
                        actual_relevance: s.actual_relevance,
                        query_type: s.query_type.clone(),
                        language: s.language.clone(),
                    }
                })
                .collect();
            
            let temp_measurement = self.calculate_calibration_measurement(&calibrated_samples).await?;
            
            if temp_measurement.ece < best_ece {
                best_ece = temp_measurement.ece;
                best_temperature = temperature;
            }
        }
        
        Ok(TemperatureModel {
            temperature: best_temperature,
            bias: 0.0, // Not used in current implementation
            ece_before: before_measurement.ece,
            ece_after: best_ece,
            training_samples: samples_owned.len(),
            scope: scope.to_string(),
        })
    }
    
    async fn adjust_temperature_models(&self, samples: &[CalibrationSample]) -> Result<()> {
        info!("Adjusting temperature models due to calibration drift");
        
        // Retrain temperature models with current data
        self.train_temperature_models(samples).await?;
        
        Ok(())
    }
    
    async fn add_calibration_snapshot(&self, ece: f32, mce: f32, semantic_active: bool, temperature: f32) {
        let snapshot = CalibrationSnapshot {
            timestamp: std::time::SystemTime::now(),
            ece,
            mce,
            sample_count: 0, // Would be filled in real implementation
            semantic_features_active: semantic_active,
            temperature,
        };
        
        let mut history = self.calibration_history.write().await;
        history.push(snapshot);
        
        // Keep only recent history
        let history_len = history.len();
        if history_len > self.config.measurement_window_size {
            history.drain(0..history_len - self.config.measurement_window_size);
        }
    }
}

// Default implementations

impl Default for CalibrationMeasurement {
    fn default() -> Self {
        Self {
            ece: 0.0,
            ece_ci_lower: 0.0,
            ece_ci_upper: 0.0,
            mce: 0.0,
            reliability_data: Vec::new(),
            sample_count: 0,
            timestamp: std::time::SystemTime::now(),
            by_query_type: HashMap::new(),
            by_language: HashMap::new(),
        }
    }
}

impl Default for FeatureScalerSet {
    fn default() -> Self {
        Self {
            semantic_feature_caps: HashMap::new(),
            linear_scalers: HashMap::new(),
            robust_scalers: HashMap::new(),
        }
    }
}

/// Calibration drift status
#[derive(Debug, Clone)]
pub enum CalibrationStatus {
    WithinLimits { ece_drift: f32, current_ece: f32 },
    DriftDetected { ece_drift: f32, current_ece: f32, baseline_ece: f32 },
    NoBaseline,
    InsufficientData,
}

/// Calibration system metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationMetrics {
    pub current_ece: f32,
    pub baseline_ece: f32,
    pub ece_drift: f32,
    pub max_allowed_drift: f32,
    pub within_limits: bool,
    pub recent_trend: f32,
    pub measurement_count: usize,
    pub sample_count: usize,
}

/// Initialize calibration system
pub async fn initialize_calibration(config: &CalibrationConfig) -> Result<()> {
    info!("Initializing calibration preservation system");
    info!("ECE drift limit: {:.4}, log-odds cap: {:.2}", 
          config.max_ece_drift, config.log_odds_cap);
    
    if config.max_ece_drift > 0.01 {
        warn!("ECE drift limit {:.4} may be too permissive", config.max_ece_drift);
    }
    
    if config.log_odds_cap < 3.0 {
        warn!("Log-odds cap {:.2} may be too restrictive", config.log_odds_cap);
    }
    
    info!("Calibration system initialized");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_calibration_system_creation() {
        let config = CalibrationConfig {
            max_ece_drift: 0.005,
            log_odds_cap: 5.0,
            temperature: 1.0,
            min_samples_for_calibration: 100,
            measurement_window_size: 50,
            auto_temperature_adjustment: true,
        };
        
        let system = CalibrationSystem::new(config).await.unwrap();
        let metrics = system.get_calibration_metrics().await;
        assert_eq!(metrics.measurement_count, 0);
    }

    #[tokio::test]
    async fn test_feature_scaling() {
        let config = CalibrationConfig {
            max_ece_drift: 0.005,
            log_odds_cap: 3.0,
            temperature: 1.0,
            min_samples_for_calibration: 100,
            measurement_window_size: 50,
            auto_temperature_adjustment: false,
        };
        
        let system = CalibrationSystem::new(config).await.unwrap();
        
        let mut features = HashMap::new();
        features.insert("semantic_similarity".to_string(), 8.0); // High value
        features.insert("lexical_score".to_string(), 0.7);
        
        let scaled = system.scale_features(&features).await.unwrap();
        
        // Semantic feature should be capped
        assert!(scaled["semantic_similarity"].abs() <= 3.0);
        // Non-semantic feature should pass through (no scaler configured)
        assert_eq!(scaled["lexical_score"], 0.7);
    }

    #[tokio::test]
    async fn test_temperature_scaling() {
        let config = CalibrationConfig {
            max_ece_drift: 0.005,
            log_odds_cap: 5.0,
            temperature: 2.0, // Higher temperature for smoothing
            min_samples_for_calibration: 100,
            measurement_window_size: 50,
            auto_temperature_adjustment: false,
        };
        
        let system = CalibrationSystem::new(config).await.unwrap();
        
        let extreme_prediction = 0.95;
        let calibrated = system.apply_temperature_scaling(extreme_prediction, "natural_language", None).await.unwrap();
        
        // Temperature scaling should reduce extreme predictions
        assert!(calibrated < extreme_prediction);
        assert!(calibrated > 0.5); // Should still be above 0.5 for high confidence
    }

    #[test]
    fn test_wilson_confidence_interval() {
        let config = CalibrationConfig {
            max_ece_drift: 0.005,
            log_odds_cap: 5.0,
            temperature: 1.0,
            min_samples_for_calibration: 100,
            measurement_window_size: 50,
            auto_temperature_adjustment: false,
        };
        
        let system = CalibrationSystem {
            config,
            baseline_calibration: Arc::new(RwLock::new(None)),
            current_calibration: Arc::new(RwLock::new(CalibrationMeasurement::default())),
            temperature_models: Arc::new(RwLock::new(HashMap::new())),
            feature_scalers: Arc::new(RwLock::new(FeatureScalerSet::default())),
            calibration_history: Arc::new(RwLock::new(Vec::new())),
        };
        
        let (lower, upper) = system.calculate_wilson_confidence_interval(0.5, 100);
        
        // Should have reasonable confidence interval
        assert!(lower < 0.5);
        assert!(upper > 0.5);
        assert!(upper - lower < 0.2); // Not too wide for n=100
    }

    #[tokio::test]
    async fn test_calibration_measurement() {
        let config = CalibrationConfig {
            max_ece_drift: 0.005,
            log_odds_cap: 5.0,
            temperature: 1.0,
            min_samples_for_calibration: 10,
            measurement_window_size: 50,
            auto_temperature_adjustment: false,
        };
        
        let system = CalibrationSystem::new(config).await.unwrap();
        
        // Create perfectly calibrated samples
        let samples = vec![
            CalibrationSample { prediction: 0.1, actual_relevance: 0.0, query_type: "test".to_string(), language: None },
            CalibrationSample { prediction: 0.3, actual_relevance: 0.0, query_type: "test".to_string(), language: None },
            CalibrationSample { prediction: 0.5, actual_relevance: 1.0, query_type: "test".to_string(), language: None },
            CalibrationSample { prediction: 0.7, actual_relevance: 1.0, query_type: "test".to_string(), language: None },
            CalibrationSample { prediction: 0.9, actual_relevance: 1.0, query_type: "test".to_string(), language: None },
        ];
        
        let measurement = system.calculate_calibration_measurement(&samples).await.unwrap();
        
        assert!(measurement.ece >= 0.0);
        assert!(measurement.ece <= 1.0);
        assert!(measurement.mce >= 0.0);
        assert_eq!(measurement.sample_count, 5);
    }
}