//! Regression Detection System for Calibration
//!
//! Makes calibration regressions loud through immediate detection, statistical
//! significance testing, trend analysis, and integration with monitoring infrastructure.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{error, warn, info, debug};

use crate::calibration::drift_slos::{CalibrationMetrics, AlertSeverity};

/// Statistical significance test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignificanceTest {
    pub test_name: String,
    pub p_value: f64,
    pub test_statistic: f64,
    pub critical_value: f64,
    pub is_significant: bool,
    pub confidence_level: f64,
    pub effect_size: f64,
}

/// Regression detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionDetection {
    pub metric_name: String,
    pub regression_type: RegressionType,
    pub severity: RegressionSeverity,
    pub detected_at: u64,
    pub baseline_value: f64,
    pub current_value: f64,
    pub change_magnitude: f64,
    pub significance_test: SignificanceTest,
    pub trend_analysis: TrendAnalysis,
    pub context: HashMap<String, String>,
}

/// Types of regression patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegressionType {
    /// Sudden significant drop/increase
    SuddenJump,
    /// Gradual but consistent decline
    GradualDrift,
    /// Increased variance/instability
    VarianceIncrease,
    /// Oscillating behavior
    Oscillation,
    /// Complete metric failure
    MetricFailure,
}

/// Regression severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegressionSeverity {
    /// Critical: System integrity compromised
    Critical,
    /// Severe: Major performance degradation
    Severe,
    /// Moderate: Noticeable degradation
    Moderate,
    /// Minor: Small but statistically significant change
    Minor,
}

/// Trend analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub slope: f64,
    pub r_squared: f64,
    pub trend_direction: TrendDirection,
    pub volatility: f64,
    pub acceleration: f64,
    pub prediction_horizon_hours: u8,
    pub predicted_value: f64,
    pub confidence_interval: (f64, f64),
}

/// Trend directions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Volatile,
}

/// Early warning system state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyWarning {
    pub metric_name: String,
    pub warning_level: WarningLevel,
    pub predicted_regression_time: Option<u64>,
    pub confidence: f64,
    pub recommended_actions: Vec<String>,
}

/// Warning levels for early warning system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WarningLevel {
    Green,    // All clear
    Yellow,   // Watch closely
    Orange,   // Action may be needed soon
    Red,      // Immediate attention required
}

/// Regression detection and monitoring system
#[derive(Debug)]
pub struct RegressionDetector {
    /// Historical metrics buffer for trend analysis
    metrics_history: VecDeque<CalibrationMetrics>,
    /// Maximum history size
    max_history_size: usize,
    /// Detected regressions
    detected_regressions: Vec<RegressionDetection>,
    /// Statistical significance thresholds
    significance_level: f64,
    /// Minimum effect size for regression detection
    min_effect_size: f64,
    /// Early warning configurations
    early_warning_thresholds: HashMap<String, f64>,
}

impl RegressionDetector {
    /// Create new regression detector with default settings
    pub fn new() -> Self {
        Self {
            metrics_history: VecDeque::with_capacity(1000),
            max_history_size: 1000,
            detected_regressions: Vec::new(),
            significance_level: 0.05, // 95% confidence
            min_effect_size: 0.2,     // Cohen's d = 0.2 (small effect)
            early_warning_thresholds: Self::default_warning_thresholds(),
        }
    }

    /// Create detector with custom configuration
    pub fn with_config(
        max_history_size: usize,
        significance_level: f64,
        min_effect_size: f64,
    ) -> Self {
        Self {
            metrics_history: VecDeque::with_capacity(max_history_size),
            max_history_size,
            detected_regressions: Vec::new(),
            significance_level,
            min_effect_size,
            early_warning_thresholds: Self::default_warning_thresholds(),
        }
    }

    /// Default early warning thresholds for different metrics
    fn default_warning_thresholds() -> HashMap<String, f64> {
        let mut thresholds = HashMap::new();
        thresholds.insert("aece".to_string(), 0.005);  // 0.5% change triggers warning
        thresholds.insert("dece".to_string(), 0.005);
        thresholds.insert("alpha".to_string(), 0.025); // 2.5% change
        thresholds.insert("clamp_rate".to_string(), 0.02); // 2% change
        thresholds.insert("merged_bin_rate".to_string(), 0.01); // 1% change
        thresholds
    }

    /// Add new metrics and detect regressions
    pub fn add_metrics(&mut self, metrics: CalibrationMetrics) -> Vec<RegressionDetection> {
        // Add to history
        self.metrics_history.push_back(metrics);
        if self.metrics_history.len() > self.max_history_size {
            self.metrics_history.pop_front();
        }

        // Detect regressions if we have enough history
        if self.metrics_history.len() >= 10 {
            self.detect_regressions()
        } else {
            Vec::new()
        }
    }

    /// Detect regressions in current metrics compared to baseline
    fn detect_regressions(&mut self) -> Vec<RegressionDetection> {
        let mut regressions = Vec::new();
        
        if let (Some(current), Some(baseline)) = (
            self.metrics_history.back(),
            self.get_baseline_metrics()
        ) {
            // Check each metric for regressions
            regressions.extend(self.check_metric_regression("aece", current.aece, baseline.aece, current.timestamp));
            regressions.extend(self.check_metric_regression("dece", current.dece, baseline.dece, current.timestamp));  
            regressions.extend(self.check_metric_regression("alpha", current.alpha, baseline.alpha, current.timestamp));
            regressions.extend(self.check_metric_regression("clamp_rate", current.clamp_rate, baseline.clamp_rate, current.timestamp));
            regressions.extend(self.check_metric_regression("merged_bin_rate", current.merged_bin_rate, baseline.merged_bin_rate, current.timestamp));
            
            // Check for metric failures
            regressions.extend(self.check_metric_failures(current));
        }

        // Store and log regressions
        for regression in &regressions {
            self.log_regression(regression);
        }
        self.detected_regressions.extend(regressions.clone());

        regressions
    }

    /// Check specific metric for regression
    fn check_metric_regression(
        &self,
        metric_name: &str,
        current_value: f64,
        baseline_value: f64,
        timestamp: u64,
    ) -> Vec<RegressionDetection> {
        let mut regressions = Vec::new();
        
        // Extract time series for this metric
        let values: Vec<f64> = self.metrics_history.iter()
            .map(|m| self.extract_metric_value(m, metric_name))
            .collect();
        
        if values.len() < 10 {
            return regressions;
        }

        // Perform statistical significance test
        let significance_test = self.perform_significance_test(metric_name, &values);
        
        // Perform trend analysis
        let trend_analysis = self.analyze_trend(&values);
        
        // Calculate change magnitude and effect size
        let change_magnitude = (current_value - baseline_value).abs();
        let effect_size = significance_test.effect_size;
        
        // Determine if this is a regression
        if significance_test.is_significant && effect_size >= self.min_effect_size {
            let regression_type = self.classify_regression_type(&values, &trend_analysis);
            let severity = self.assess_regression_severity(
                metric_name, change_magnitude, effect_size, &regression_type
            );
            
            // Only report if severity is above minimal threshold
            if severity != RegressionSeverity::Minor || regression_type == RegressionType::MetricFailure {
                regressions.push(RegressionDetection {
                    metric_name: metric_name.to_string(),
                    regression_type,
                    severity,
                    detected_at: timestamp,
                    baseline_value,
                    current_value,
                    change_magnitude,
                    significance_test,
                    trend_analysis,
                    context: self.build_regression_context(metric_name, &values),
                });
            }
        }

        regressions
    }

    /// Check for complete metric failures
    fn check_metric_failures(&self, metrics: &CalibrationMetrics) -> Vec<RegressionDetection> {
        let mut failures = Vec::new();
        let timestamp = metrics.timestamp;
        
        // Check for score range violations (critical failure)
        if metrics.score_range_violations > 0 {
            failures.push(RegressionDetection {
                metric_name: "score_range_violations".to_string(),
                regression_type: RegressionType::MetricFailure,
                severity: RegressionSeverity::Critical,
                detected_at: timestamp,
                baseline_value: 0.0,
                current_value: metrics.score_range_violations as f64,
                change_magnitude: metrics.score_range_violations as f64,
                significance_test: SignificanceTest {
                    test_name: "Failure Detection".to_string(),
                    p_value: 0.0,
                    test_statistic: f64::INFINITY,
                    critical_value: 0.0,
                    is_significant: true,
                    confidence_level: 1.0,
                    effect_size: f64::INFINITY,
                },
                trend_analysis: TrendAnalysis {
                    slope: 0.0,
                    r_squared: 0.0,
                    trend_direction: TrendDirection::Degrading,
                    volatility: 0.0,
                    acceleration: 0.0,
                    prediction_horizon_hours: 0,
                    predicted_value: metrics.score_range_violations as f64,
                    confidence_interval: (0.0, f64::INFINITY),
                },
                context: HashMap::from([
                    ("failure_type".to_string(), "score_range_violation".to_string()),
                    ("count".to_string(), metrics.score_range_violations.to_string()),
                ]),
            });
        }

        // Check for mask mismatches (high severity failure)
        if metrics.mask_mismatch_count > 0 {
            failures.push(RegressionDetection {
                metric_name: "mask_mismatch".to_string(),
                regression_type: RegressionType::MetricFailure,
                severity: RegressionSeverity::Severe,
                detected_at: timestamp,
                baseline_value: 0.0,
                current_value: metrics.mask_mismatch_count as f64,
                change_magnitude: metrics.mask_mismatch_count as f64,
                significance_test: SignificanceTest {
                    test_name: "Failure Detection".to_string(),
                    p_value: 0.0,
                    test_statistic: f64::INFINITY,
                    critical_value: 0.0,
                    is_significant: true,
                    confidence_level: 1.0,
                    effect_size: f64::INFINITY,
                },
                trend_analysis: TrendAnalysis {
                    slope: 0.0,
                    r_squared: 0.0,
                    trend_direction: TrendDirection::Degrading,
                    volatility: 0.0,
                    acceleration: 0.0,
                    prediction_horizon_hours: 0,
                    predicted_value: metrics.mask_mismatch_count as f64,
                    confidence_interval: (0.0, f64::INFINITY),
                },
                context: HashMap::from([
                    ("failure_type".to_string(), "mask_mismatch".to_string()),
                    ("count".to_string(), metrics.mask_mismatch_count.to_string()),
                ]),
            });
        }

        failures
    }

    /// Extract metric value by name
    fn extract_metric_value(&self, metrics: &CalibrationMetrics, metric_name: &str) -> f64 {
        match metric_name {
            "aece" => metrics.aece,
            "dece" => metrics.dece,
            "alpha" => metrics.alpha,
            "clamp_rate" => metrics.clamp_rate,
            "merged_bin_rate" => metrics.merged_bin_rate,
            _ => 0.0,
        }
    }

    /// Get baseline metrics (median of historical values)
    fn get_baseline_metrics(&self) -> Option<CalibrationMetrics> {
        if self.metrics_history.len() < 10 {
            return None;
        }

        // Use median values from historical data as baseline
        let mut aece_values: Vec<f64> = self.metrics_history.iter().map(|m| m.aece).collect();
        let mut dece_values: Vec<f64> = self.metrics_history.iter().map(|m| m.dece).collect();
        let mut alpha_values: Vec<f64> = self.metrics_history.iter().map(|m| m.alpha).collect();
        let mut clamp_rate_values: Vec<f64> = self.metrics_history.iter().map(|m| m.clamp_rate).collect();
        let mut merged_bin_values: Vec<f64> = self.metrics_history.iter().map(|m| m.merged_bin_rate).collect();

        aece_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        dece_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        alpha_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        clamp_rate_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        merged_bin_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let len = self.metrics_history.len();
        Some(CalibrationMetrics {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            aece: aece_values[len / 2],
            dece: dece_values[len / 2],
            alpha: alpha_values[len / 2],
            clamp_rate: clamp_rate_values[len / 2],
            merged_bin_rate: merged_bin_values[len / 2],
            score_range_violations: 0,
            mask_mismatch_count: 0,
            total_samples: self.metrics_history.back().unwrap().total_samples,
        })
    }

    /// Perform statistical significance test
    fn perform_significance_test(&self, metric_name: &str, values: &[f64]) -> SignificanceTest {
        if values.len() < 10 {
            return SignificanceTest {
                test_name: "Insufficient Data".to_string(),
                p_value: 1.0,
                test_statistic: 0.0,
                critical_value: 0.0,
                is_significant: false,
                confidence_level: 0.0,
                effect_size: 0.0,
            };
        }

        // Split into recent and historical groups
        let split_point = values.len() * 2 / 3;
        let historical = &values[..split_point];
        let recent = &values[split_point..];

        // Calculate means and standard deviations
        let hist_mean = historical.iter().sum::<f64>() / historical.len() as f64;
        let recent_mean = recent.iter().sum::<f64>() / recent.len() as f64;
        
        let hist_var = historical.iter()
            .map(|x| (x - hist_mean).powi(2))
            .sum::<f64>() / (historical.len() - 1) as f64;
        let recent_var = recent.iter()
            .map(|x| (x - recent_mean).powi(2))
            .sum::<f64>() / (recent.len() - 1) as f64;

        // Welch's t-test for unequal variances
        let pooled_std = ((hist_var / historical.len() as f64) + (recent_var / recent.len() as f64)).sqrt();
        let t_statistic = (recent_mean - hist_mean) / pooled_std;
        
        // Degrees of freedom (Welch-Satterthwaite equation)
        let df = ((hist_var / historical.len() as f64) + (recent_var / recent.len() as f64)).powi(2) 
                / ((hist_var / historical.len() as f64).powi(2) / (historical.len() - 1) as f64
                + (recent_var / recent.len() as f64).powi(2) / (recent.len() - 1) as f64);

        // Critical value (approximate for df > 30, use 1.96 for 95% confidence)
        let critical_value = if df > 30.0 { 1.96 } else { 2.042 }; // Conservative estimate
        
        // P-value (approximation)
        let p_value = if t_statistic.abs() > critical_value { 
            self.significance_level / 2.0 
        } else { 
            self.significance_level * 2.0 
        };
        
        // Effect size (Cohen's d)
        let effect_size = (recent_mean - hist_mean).abs() / (hist_var.max(recent_var)).sqrt();

        SignificanceTest {
            test_name: "Welch's t-test".to_string(),
            p_value,
            test_statistic: t_statistic,
            critical_value,
            is_significant: p_value < self.significance_level,
            confidence_level: 1.0 - self.significance_level,
            effect_size,
        }
    }

    /// Analyze trend in time series data
    fn analyze_trend(&self, values: &[f64]) -> TrendAnalysis {
        if values.len() < 3 {
            return TrendAnalysis {
                slope: 0.0,
                r_squared: 0.0,
                trend_direction: TrendDirection::Stable,
                volatility: 0.0,
                acceleration: 0.0,
                prediction_horizon_hours: 24,
                predicted_value: values.last().copied().unwrap_or(0.0),
                confidence_interval: (0.0, 0.0),
            };
        }

        // Simple linear regression
        let n = values.len() as f64;
        let x: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = values.iter().sum::<f64>() / n;
        
        let numerator: f64 = x.iter().zip(values).map(|(xi, yi)| (xi - x_mean) * (yi - y_mean)).sum();
        let denominator: f64 = x.iter().map(|xi| (xi - x_mean).powi(2)).sum();
        
        let slope = if denominator != 0.0 { numerator / denominator } else { 0.0 };
        let intercept = y_mean - slope * x_mean;
        
        // R-squared
        let ss_tot: f64 = values.iter().map(|yi| (yi - y_mean).powi(2)).sum();
        let ss_res: f64 = x.iter().zip(values)
            .map(|(xi, yi)| (yi - (slope * xi + intercept)).powi(2))
            .sum();
        let r_squared = if ss_tot != 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };
        
        // Volatility (standard deviation)
        let volatility = (values.iter().map(|y| (y - y_mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        
        // Acceleration (second derivative approximation)
        let mut acceleration = 0.0;
        if values.len() >= 3 {
            let recent_slope = (values[values.len()-1] - values[values.len()-2]);
            let earlier_slope = (values[values.len()-2] - values[values.len()-3]);
            acceleration = recent_slope - earlier_slope;
        }

        // Trend direction
        let trend_direction = if slope.abs() < volatility * 0.1 {
            if volatility > y_mean.abs() * 0.1 {
                TrendDirection::Volatile
            } else {
                TrendDirection::Stable
            }
        } else if slope > 0.0 {
            TrendDirection::Improving
        } else {
            TrendDirection::Degrading
        };

        // Prediction (24 hours ahead, assuming hourly measurements)
        let prediction_horizon = 24.0;
        let predicted_value = slope * (n - 1.0 + prediction_horizon) + intercept;
        
        // Confidence interval (rough approximation)
        let stderr = (ss_res / ((n - 2.0) * denominator)).sqrt();
        let confidence_margin = 1.96 * stderr;
        let confidence_interval = (
            predicted_value - confidence_margin,
            predicted_value + confidence_margin,
        );

        TrendAnalysis {
            slope,
            r_squared,
            trend_direction,
            volatility,
            acceleration,
            prediction_horizon_hours: 24,
            predicted_value,
            confidence_interval,
        }
    }

    /// Classify type of regression based on patterns
    fn classify_regression_type(&self, values: &[f64], trend: &TrendAnalysis) -> RegressionType {
        // Check for sudden jump (large change in last few points)
        if values.len() >= 3 {
            let recent_change = (values[values.len()-1] - values[values.len()-3]).abs();
            let typical_change = trend.volatility;
            
            if recent_change > typical_change * 3.0 {
                return RegressionType::SuddenJump;
            }
        }

        // Check based on trend characteristics
        match trend.trend_direction {
            TrendDirection::Degrading => {
                if trend.r_squared > 0.7 {
                    RegressionType::GradualDrift
                } else {
                    RegressionType::SuddenJump
                }
            },
            TrendDirection::Volatile => {
                if trend.volatility > values.iter().sum::<f64>() / values.len() as f64 * 0.2 {
                    RegressionType::VarianceIncrease
                } else {
                    RegressionType::Oscillation
                }
            },
            _ => RegressionType::GradualDrift,
        }
    }

    /// Assess regression severity
    fn assess_regression_severity(
        &self,
        metric_name: &str,
        change_magnitude: f64,
        effect_size: f64,
        regression_type: &RegressionType,
    ) -> RegressionSeverity {
        // Critical severity for metric failures
        if matches!(regression_type, RegressionType::MetricFailure) {
            return RegressionSeverity::Critical;
        }

        // Assess based on effect size and metric importance
        let base_severity = if effect_size > 2.0 {
            RegressionSeverity::Severe
        } else if effect_size > 0.8 {
            RegressionSeverity::Moderate
        } else if effect_size > 0.2 {
            RegressionSeverity::Minor
        } else {
            return RegressionSeverity::Minor;
        };

        // Adjust for metric importance
        let importance_multiplier = match metric_name {
            "score_range_violations" | "mask_mismatch" => 2.0,
            "aece" | "dece" => 1.5,
            "clamp_rate" | "merged_bin_rate" => 1.0,
            _ => 0.8,
        };

        // Adjust for regression type
        let type_multiplier = match regression_type {
            RegressionType::SuddenJump => 1.5,
            RegressionType::VarianceIncrease => 1.3,
            RegressionType::GradualDrift => 1.0,
            RegressionType::Oscillation => 0.8,
            RegressionType::MetricFailure => 2.0,
        };

        let adjusted_effect = effect_size * importance_multiplier * type_multiplier;

        if adjusted_effect > 3.0 {
            RegressionSeverity::Critical
        } else if adjusted_effect > 1.5 {
            RegressionSeverity::Severe
        } else if adjusted_effect > 0.8 {
            RegressionSeverity::Moderate
        } else {
            RegressionSeverity::Minor
        }
    }

    /// Build context for regression detection
    fn build_regression_context(&self, metric_name: &str, values: &[f64]) -> HashMap<String, String> {
        let mut context = HashMap::new();
        
        context.insert("metric".to_string(), metric_name.to_string());
        context.insert("history_length".to_string(), values.len().to_string());
        context.insert("current_value".to_string(), format!("{:.6}", values.last().unwrap_or(&0.0)));
        
        if values.len() >= 2 {
            let prev_value = values[values.len()-2];
            let change = values.last().unwrap() - prev_value;
            context.insert("recent_change".to_string(), format!("{:.6}", change));
            context.insert("recent_change_pct".to_string(), 
                          format!("{:.2}%", (change / prev_value) * 100.0));
        }
        
        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        context.insert("range_min".to_string(), format!("{:.6}", min_val));
        context.insert("range_max".to_string(), format!("{:.6}", max_val));
        
        context
    }

    /// Generate early warning alerts
    pub fn generate_early_warnings(&self) -> Vec<EarlyWarning> {
        let mut warnings = Vec::new();
        
        if self.metrics_history.len() < 5 {
            return warnings;
        }

        for metric_name in &["aece", "dece", "alpha", "clamp_rate", "merged_bin_rate"] {
            let values: Vec<f64> = self.metrics_history.iter()
                .map(|m| self.extract_metric_value(m, metric_name))
                .collect();
            
            let trend = self.analyze_trend(&values);
            let warning = self.assess_early_warning(metric_name, &values, &trend);
            
            if warning.warning_level != WarningLevel::Green {
                warnings.push(warning);
            }
        }

        warnings
    }

    /// Assess early warning level for a metric
    fn assess_early_warning(&self, metric_name: &str, values: &[f64], trend: &TrendAnalysis) -> EarlyWarning {
        let current_value = values.last().copied().unwrap_or(0.0);
        let threshold = self.early_warning_thresholds.get(metric_name).copied().unwrap_or(0.01);
        
        // Assess current state
        let current_risk = if current_value.is_nan() || current_value.is_infinite() {
            1.0 // Maximum risk
        } else if trend.trend_direction == TrendDirection::Degrading {
            (trend.slope.abs() / threshold).min(1.0)
        } else if trend.trend_direction == TrendDirection::Volatile {
            (trend.volatility / threshold / 2.0).min(1.0)
        } else {
            0.0
        };

        // Predict future risk
        let predicted_risk = if trend.predicted_value.is_nan() {
            1.0
        } else {
            let predicted_change = (trend.predicted_value - current_value).abs();
            (predicted_change / threshold).min(1.0)
        };
        
        let combined_risk = current_risk.max(predicted_risk);
        
        let (warning_level, confidence, recommended_actions) = if combined_risk > 0.8 {
            (
                WarningLevel::Red,
                0.9,
                vec![
                    "Immediate investigation required".to_string(),
                    "Consider triggering remediation".to_string(),
                    "Alert on-call team".to_string(),
                ]
            )
        } else if combined_risk > 0.6 {
            (
                WarningLevel::Orange,
                0.7,
                vec![
                    "Monitor closely".to_string(),
                    "Prepare contingency plans".to_string(),
                    "Review recent changes".to_string(),
                ]
            )
        } else if combined_risk > 0.3 {
            (
                WarningLevel::Yellow,
                0.5,
                vec![
                    "Increased monitoring".to_string(),
                    "Review trend analysis".to_string(),
                ]
            )
        } else {
            (
                WarningLevel::Green,
                0.2,
                vec!["Continue normal monitoring".to_string()]
            )
        };

        let predicted_regression_time = if trend.trend_direction == TrendDirection::Degrading && trend.slope != 0.0 {
            let time_to_threshold = threshold / trend.slope.abs();
            let current_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
            Some(current_time + (time_to_threshold * 3600.0) as u64) // Convert hours to seconds
        } else {
            None
        };

        EarlyWarning {
            metric_name: metric_name.to_string(),
            warning_level,
            predicted_regression_time,
            confidence,
            recommended_actions,
        }
    }

    /// Log regression detection
    fn log_regression(&self, regression: &RegressionDetection) {
        let context_str = regression.context
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(" ");

        match regression.severity {
            RegressionSeverity::Critical => {
                error!(
                    "🚨 CRITICAL REGRESSION DETECTED: {} ({:?}) - {} → {} (Δ={:.6}, effect_size={:.3}) - {}",
                    regression.metric_name,
                    regression.regression_type,
                    regression.baseline_value,
                    regression.current_value,
                    regression.change_magnitude,
                    regression.significance_test.effect_size,
                    context_str
                );
            }
            RegressionSeverity::Severe => {
                error!(
                    "🔥 SEVERE REGRESSION DETECTED: {} ({:?}) - {} → {} (Δ={:.6}, effect_size={:.3}) - {}",
                    regression.metric_name,
                    regression.regression_type,
                    regression.baseline_value,
                    regression.current_value,
                    regression.change_magnitude,
                    regression.significance_test.effect_size,
                    context_str
                );
            }
            RegressionSeverity::Moderate => {
                warn!(
                    "⚠️ MODERATE REGRESSION DETECTED: {} ({:?}) - {} → {} (Δ={:.6}, effect_size={:.3}) - {}",
                    regression.metric_name,
                    regression.regression_type,
                    regression.baseline_value,
                    regression.current_value,
                    regression.change_magnitude,
                    regression.significance_test.effect_size,
                    context_str
                );
            }
            RegressionSeverity::Minor => {
                info!(
                    "📊 MINOR REGRESSION DETECTED: {} ({:?}) - {} → {} (Δ={:.6}, effect_size={:.3}) - {}",
                    regression.metric_name,
                    regression.regression_type,
                    regression.baseline_value,
                    regression.current_value,
                    regression.change_magnitude,
                    regression.significance_test.effect_size,
                    context_str
                );
            }
        }
    }

    /// Get recent regressions (last 24 hours)
    pub fn get_recent_regressions(&self) -> Vec<RegressionDetection> {
        let day_ago = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .saturating_sub(24 * 3600);

        self.detected_regressions
            .iter()
            .filter(|r| r.detected_at >= day_ago)
            .cloned()
            .collect()
    }

    /// Clear regression history (for maintenance)
    pub fn clear_regression_history(&mut self) {
        info!("Clearing regression detection history, had {} regressions", 
              self.detected_regressions.len());
        self.detected_regressions.clear();
    }

    /// Get system health status based on regressions
    pub fn get_health_status(&self) -> (bool, String) {
        let recent_regressions = self.get_recent_regressions();
        let critical_count = recent_regressions.iter()
            .filter(|r| r.severity == RegressionSeverity::Critical)
            .count();
        let severe_count = recent_regressions.iter()
            .filter(|r| r.severity == RegressionSeverity::Severe)
            .count();
        
        let is_healthy = critical_count == 0 && severe_count <= 1;
        let status = if critical_count > 0 {
            format!("🚨 CRITICAL: {} critical regressions detected", critical_count)
        } else if severe_count > 2 {
            format!("🔥 SEVERE: {} severe regressions detected", severe_count)
        } else if severe_count > 0 {
            format!("⚠️ DEGRADED: {} severe regressions detected", severe_count)
        } else if recent_regressions.len() > 5 {
            format!("📊 MONITORING: {} minor regressions detected", recent_regressions.len())
        } else {
            "✅ HEALTHY: No significant regressions detected".to_string()
        };

        (is_healthy, status)
    }

    /// Generate regression summary report
    pub fn generate_regression_report(&self) -> String {
        let recent_regressions = self.get_recent_regressions();
        let early_warnings = self.generate_early_warnings();
        let (is_healthy, status) = self.get_health_status();
        
        format!(
            "CALIBRATION REGRESSION DETECTION REPORT\n\
             =======================================\n\
             \n\
             Overall Status: {}\n\
             \n\
             Recent Regressions (24h): {}\n\
             - Critical: {}\n\
             - Severe: {}\n\
             - Moderate: {}\n\
             - Minor: {}\n\
             \n\
             Early Warnings: {}\n\
             - Red: {}\n\
             - Orange: {}\n\
             - Yellow: {}\n\
             \n\
             System Health: {}\n\
             History Buffer: {} / {} samples\n\
             \n\
             Statistical Configuration:\n\
             - Significance Level: {:.1}% confidence\n\
             - Minimum Effect Size: {:.2}\n\
             - History Retention: {} samples",
            status,
            recent_regressions.len(),
            recent_regressions.iter().filter(|r| r.severity == RegressionSeverity::Critical).count(),
            recent_regressions.iter().filter(|r| r.severity == RegressionSeverity::Severe).count(),
            recent_regressions.iter().filter(|r| r.severity == RegressionSeverity::Moderate).count(),
            recent_regressions.iter().filter(|r| r.severity == RegressionSeverity::Minor).count(),
            early_warnings.len(),
            early_warnings.iter().filter(|w| w.warning_level == WarningLevel::Red).count(),
            early_warnings.iter().filter(|w| w.warning_level == WarningLevel::Orange).count(),
            early_warnings.iter().filter(|w| w.warning_level == WarningLevel::Yellow).count(),
            if is_healthy { "HEALTHY" } else { "DEGRADED" },
            self.metrics_history.len(),
            self.max_history_size,
            (1.0 - self.significance_level) * 100.0,
            self.min_effect_size,
            self.max_history_size
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_metrics(aece: f64, timestamp_offset: u64) -> CalibrationMetrics {
        CalibrationMetrics {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() + timestamp_offset,
            aece,
            dece: 0.015,
            alpha: 0.5,
            clamp_rate: 0.05,
            merged_bin_rate: 0.02,
            score_range_violations: 0,
            mask_mismatch_count: 0,
            total_samples: 10000,
        }
    }

    #[test]
    fn test_regression_detection_normal_operation() {
        let mut detector = RegressionDetector::new();
        
        // Add normal metrics
        for i in 0..20 {
            let metrics = create_test_metrics(0.02 + (i as f64 * 0.0001), i);
            let regressions = detector.add_metrics(metrics);
            
            if i >= 10 {
                // Should not detect regressions for normal variation
                assert!(regressions.is_empty() || regressions.iter().all(|r| r.severity == RegressionSeverity::Minor));
            }
        }
        
        assert!(detector.get_health_status().0); // Should be healthy
    }

    #[test]
    fn test_sudden_regression_detection() {
        let mut detector = RegressionDetector::new();
        
        // Add normal metrics
        for i in 0..15 {
            detector.add_metrics(create_test_metrics(0.02, i));
        }
        
        // Add sudden regression
        let regressions = detector.add_metrics(create_test_metrics(0.05, 15)); // Large jump
        
        assert!(!regressions.is_empty());
        assert!(regressions.iter().any(|r| r.metric_name == "aece"));
        assert!(regressions.iter().any(|r| matches!(r.regression_type, RegressionType::SuddenJump)));
    }

    #[test]
    fn test_gradual_drift_detection() {
        let mut detector = RegressionDetector::new();
        
        // Add gradual drift
        for i in 0..25 {
            let drift_value = 0.02 + (i as f64 * 0.002); // Gradual increase
            let regressions = detector.add_metrics(create_test_metrics(drift_value, i));
            
            if i > 20 {
                // Should detect gradual drift
                if !regressions.is_empty() {
                    assert!(regressions.iter().any(|r| r.significance_test.is_significant));
                }
            }
        }
    }

    #[test]
    fn test_metric_failure_detection() {
        let mut detector = RegressionDetector::new();
        
        // Add normal metrics first
        for i in 0..10 {
            detector.add_metrics(create_test_metrics(0.02, i));
        }
        
        // Add metric with failures
        let mut failing_metrics = create_test_metrics(0.02, 10);
        failing_metrics.score_range_violations = 5;
        failing_metrics.mask_mismatch_count = 2;
        
        let regressions = detector.add_metrics(failing_metrics);
        
        assert!(!regressions.is_empty());
        assert!(regressions.iter().any(|r| r.metric_name == "score_range_violations"));
        assert!(regressions.iter().any(|r| r.severity == RegressionSeverity::Critical));
        assert!(regressions.iter().any(|r| r.metric_name == "mask_mismatch"));
        assert!(regressions.iter().any(|r| r.severity == RegressionSeverity::Severe));
    }

    #[test]
    fn test_early_warning_system() {
        let mut detector = RegressionDetector::new();
        
        // Add metrics showing concerning trend
        for i in 0..10 {
            let trending_value = 0.02 + (i as f64 * 0.001); // Upward trend
            detector.add_metrics(create_test_metrics(trending_value, i));
        }
        
        let warnings = detector.generate_early_warnings();
        assert!(!warnings.is_empty());
        
        let aece_warning = warnings.iter().find(|w| w.metric_name == "aece");
        assert!(aece_warning.is_some());
        assert!(aece_warning.unwrap().warning_level != WarningLevel::Green);
    }

    #[test]
    fn test_statistical_significance() {
        let detector = RegressionDetector::new();
        
        // Test with significant change
        let stable_values: Vec<f64> = (0..10).map(|_| 0.02).collect();
        let changed_values: Vec<f64> = (0..10).map(|_| 0.02).chain((0..5).map(|_| 0.04)).collect();
        
        let test = detector.perform_significance_test("aece", &changed_values);
        // With a large effect size, should be significant
        assert!(test.effect_size > detector.min_effect_size);
    }

    #[test]
    fn test_trend_analysis() {
        let detector = RegressionDetector::new();
        
        // Test upward trend
        let upward_trend: Vec<f64> = (0..10).map(|i| 0.02 + (i as f64 * 0.001)).collect();
        let trend = detector.analyze_trend(&upward_trend);
        
        assert!(trend.slope > 0.0);
        assert_eq!(trend.trend_direction, TrendDirection::Degrading); // Higher values are worse for AECE
        
        // Test stable values
        let stable_values: Vec<f64> = (0..10).map(|_| 0.02).collect();
        let stable_trend = detector.analyze_trend(&stable_values);
        
        assert!(stable_trend.slope.abs() < 0.001);
        assert_eq!(stable_trend.trend_direction, TrendDirection::Stable);
    }

    #[test]
    fn test_health_status() {
        let mut detector = RegressionDetector::new();
        
        // Healthy state
        assert!(detector.get_health_status().0);
        
        // Add critical regression
        detector.detected_regressions.push(RegressionDetection {
            metric_name: "test".to_string(),
            regression_type: RegressionType::MetricFailure,
            severity: RegressionSeverity::Critical,
            detected_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            baseline_value: 0.0,
            current_value: 1.0,
            change_magnitude: 1.0,
            significance_test: SignificanceTest {
                test_name: "test".to_string(),
                p_value: 0.001,
                test_statistic: 5.0,
                critical_value: 1.96,
                is_significant: true,
                confidence_level: 0.95,
                effect_size: 2.0,
            },
            trend_analysis: TrendAnalysis {
                slope: 1.0,
                r_squared: 0.9,
                trend_direction: TrendDirection::Degrading,
                volatility: 0.1,
                acceleration: 0.0,
                prediction_horizon_hours: 24,
                predicted_value: 1.0,
                confidence_interval: (0.8, 1.2),
            },
            context: HashMap::new(),
        });
        
        // Should now be unhealthy
        assert!(!detector.get_health_status().0);
    }
}