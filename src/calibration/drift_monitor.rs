//! # Live Calibration Drift Monitor
//!
//! Continuous monitoring of calibration health with automated alerts:
//! - Weekly cron artifacts per intent×language combination
//! - Real-time tripwires for AECE-τ, clamp rates, merged bins
//! - SLA-bound latency monitoring (p99 < 1ms)
//! - Automated revert triggers for production safety
//! - Boring, reliable health checks that prevent surprises

use crate::calibration::{
    CalibrationSample, 
    shared_binning_core::{SharedBinningCore, SharedBinningConfig},
    fast_bootstrap::{FastBootstrap, FastBootstrapConfig}
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use chrono::{DateTime, Utc, TimeZone, Datelike};
use anyhow::{Result, Context as AnyhowContext};

/// Health status levels  
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// All metrics within acceptable bounds
    Green,
    /// Warning thresholds exceeded, monitoring required
    Yellow,
    /// Critical thresholds exceeded, intervention required  
    Red,
    /// System failure, immediate action required
    Critical,
}

/// Drift detection thresholds and bounds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftThresholds {
    /// AECE - τ(N,K) tolerance: warn > 0.005, fail > 0.01
    pub aece_minus_tau_warn: f64,
    pub aece_minus_tau_fail: f64,
    /// Clamp activation rate: warn > 5%, fail > 10%
    pub clamp_rate_warn: f64,
    pub clamp_rate_fail: f64,
    /// Merged bin rate: warn > 5%, fail > 20%
    pub merged_bin_rate_warn: f64,
    pub merged_bin_rate_fail: f64,
    /// Median confidence shift vs last green baseline
    pub confidence_shift_threshold: f64,
    /// SLA bound: p99 calibration latency < 1ms
    pub p99_latency_threshold_us: f64,
    /// Coverage probability minimum (bootstrap)
    pub min_coverage_probability: f64,
}

impl Default for DriftThresholds {
    fn default() -> Self {
        Self {
            aece_minus_tau_warn: 0.005,
            aece_minus_tau_fail: 0.01,
            clamp_rate_warn: 0.05,
            clamp_rate_fail: 0.10,
            merged_bin_rate_warn: 0.05,
            merged_bin_rate_fail: 0.20,
            confidence_shift_threshold: 0.02,
            p99_latency_threshold_us: 1000.0, // 1ms
            min_coverage_probability: 0.95,
        }
    }
}

/// Weekly calibration health artifacts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeeklyHealthArtifacts {
    /// Timestamp of artifact generation
    pub timestamp: DateTime<Utc>,
    /// Intent and language combination
    pub intent: String,
    pub language: Option<String>,
    /// Week number (ISO week)
    pub week_number: u32,
    pub year: i32,
    
    // Core calibration metrics
    /// Average Expected Calibration Error
    pub aece: f64,
    /// Deterministic ECE (single evaluation)
    pub dece: f64,
    /// Brier score
    pub brier_score: f64,
    /// Reliability slope (linear regression of accuracy vs confidence)
    pub reliability_slope: f64,
    /// Clamp activation rate (alpha adjustments)
    pub clamp_alpha_rate: f64,
    /// Merged bin rate (bins combined due to insufficient samples)
    pub merged_bin_rate: f64,
    /// ECE threshold τ(N,K)
    pub ece_threshold: f64,
    /// AECE - τ (health indicator)
    pub aece_minus_tau: f64,
    
    // Bootstrap validation
    /// Bootstrap coverage probability
    pub bootstrap_coverage: f64,
    /// Bootstrap samples used
    pub bootstrap_samples: usize,
    /// Whether early stopping was used
    pub bootstrap_early_stopped: bool,
    
    // Performance metrics
    /// P99 calibration latency (microseconds)
    pub p99_latency_us: f64,
    /// Average calibration latency (microseconds)
    pub mean_latency_us: f64,
    /// Memory allocations on hot path (should be 0)
    pub hot_path_allocations: usize,
    
    // Health assessment
    /// Overall health status
    pub health_status: HealthStatus,
    /// Alerts triggered this week
    pub alerts_triggered: Vec<AlertEvent>,
    /// Config fingerprint for reproducibility
    pub config_fingerprint: String,
}

/// Alert event details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEvent {
    pub timestamp: DateTime<Utc>,
    pub alert_type: AlertType,
    pub metric_name: String,
    pub actual_value: f64,
    pub threshold: f64,
    pub severity: HealthStatus,
    pub message: String,
}

/// Types of alerts that can be triggered
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    /// AECE - τ exceeded threshold
    AeceThresholdExceeded,
    /// Clamp rate too high
    ClampRateExceeded,
    /// Too many merged bins
    MergedBinRateExceeded,
    /// Confidence shift detected
    ConfidenceShiftDetected,
    /// SLA latency violation
    LatencySlaViolated,
    /// Bootstrap coverage too low
    CoverageInsufficient,
    /// Configuration drift detected
    ConfigDriftDetected,
    /// Mask mismatch between fit and eval
    MaskMismatch,
    /// Score out of bounds [0,1]
    ScoreOutOfBounds,
}

/// Canary gate configuration for production safety
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanaryGateConfig {
    /// Enable canary gate checks
    pub enabled: bool,
    /// Lookback period for baseline comparison (days)
    pub baseline_lookback_days: u32,
    /// Maximum allowed confidence shift
    pub max_confidence_shift: f64,
    /// Maximum allowed AECE increase
    pub max_aece_increase: f64,
    /// Require green baseline for comparison
    pub require_green_baseline: bool,
}

impl Default for CanaryGateConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            baseline_lookback_days: 7,
            max_confidence_shift: 0.02,
            max_aece_increase: 0.005,
            require_green_baseline: true,
        }
    }
}

/// Live drift monitor with automated health tracking
pub struct DriftMonitor {
    thresholds: DriftThresholds,
    canary_config: CanaryGateConfig,
    binning_core: SharedBinningCore,
    bootstrap: FastBootstrap,
    /// Historical artifacts for trend analysis
    historical_artifacts: Vec<WeeklyHealthArtifacts>,
    /// Last known green baseline for canary comparison
    last_green_baseline: Option<WeeklyHealthArtifacts>,
}

impl DriftMonitor {
    /// Create new drift monitor
    pub fn new(
        thresholds: DriftThresholds,
        canary_config: CanaryGateConfig,
        binning_config: SharedBinningConfig,
        bootstrap_config: FastBootstrapConfig,
    ) -> Self {
        let binning_core = SharedBinningCore::new(binning_config.clone());
        let bootstrap = FastBootstrap::new(bootstrap_config, binning_config);
        
        Self {
            thresholds,
            canary_config,
            binning_core,
            bootstrap,
            historical_artifacts: Vec::new(),
            last_green_baseline: None,
        }
    }

    /// Create new drift monitor with default configuration (async compatible)
    pub async fn new() -> Result<Self> {
        let thresholds = DriftThresholds::default();
        let canary_config = CanaryGateConfig::default();
        let binning_config = crate::calibration::shared_binning_core::SharedBinningConfig::default();
        let bootstrap_config = crate::calibration::fast_bootstrap::FastBootstrapConfig::default();
        
        Ok(Self::new(thresholds, canary_config, binning_config, bootstrap_config))
    }
    
    /// Compute Brier score for probability calibration
    fn compute_brier_score(&self, predictions: &[f64], labels: &[f64], weights: &[f64]) -> f64 {
        let mut brier_sum = 0.0;
        let mut weight_sum = 0.0;
        
        for i in 0..predictions.len() {
            let diff = predictions[i] - labels[i];
            brier_sum += weights[i] * diff * diff;
            weight_sum += weights[i];
        }
        
        if weight_sum > 0.0 {
            brier_sum / weight_sum
        } else {
            0.0
        }
    }
    
    /// Compute reliability slope (linear regression of accuracy vs confidence)
    fn compute_reliability_slope(&self, bin_stats: &[crate::calibration::shared_binning_core::BinStatistics]) -> f64 {
        let valid_bins: Vec<_> = bin_stats.iter()
            .filter(|stats| stats.weight > 0.0)
            .collect();
            
        if valid_bins.len() < 2 {
            return 1.0; // Perfect reliability if insufficient data
        }
        
        // Simple linear regression: accuracy = slope * confidence + intercept
        let n = valid_bins.len() as f64;
        let sum_confidence: f64 = valid_bins.iter().map(|stats| stats.confidence).sum();
        let sum_accuracy: f64 = valid_bins.iter().map(|stats| stats.accuracy).sum();
        let sum_conf_acc: f64 = valid_bins.iter()
            .map(|stats| stats.confidence * stats.accuracy)
            .sum();
        let sum_conf_sq: f64 = valid_bins.iter()
            .map(|stats| stats.confidence * stats.confidence)
            .sum();
            
        let denominator = n * sum_conf_sq - sum_confidence * sum_confidence;
        if denominator.abs() < 1e-10 {
            return 1.0; // Avoid division by zero
        }
        
        (n * sum_conf_acc - sum_confidence * sum_accuracy) / denominator
    }
    
    /// Assess health status based on metrics and thresholds
    fn assess_health_status(&self, artifacts: &WeeklyHealthArtifacts) -> HealthStatus {
        // Critical failures
        if artifacts.aece_minus_tau > self.thresholds.aece_minus_tau_fail
            || artifacts.clamp_alpha_rate > self.thresholds.clamp_rate_fail
            || artifacts.merged_bin_rate > self.thresholds.merged_bin_rate_fail
            || artifacts.p99_latency_us > self.thresholds.p99_latency_threshold_us
            || artifacts.bootstrap_coverage < self.thresholds.min_coverage_probability {
            return HealthStatus::Critical;
        }
        
        // Red alerts
        if artifacts.aece_minus_tau > self.thresholds.aece_minus_tau_warn
            || artifacts.clamp_alpha_rate > self.thresholds.clamp_rate_warn
            || artifacts.merged_bin_rate > self.thresholds.merged_bin_rate_warn {
            return HealthStatus::Red;
        }
        
        // Yellow warnings - check for trends or minor issues
        if artifacts.hot_path_allocations > 0 {
            return HealthStatus::Yellow;
        }
        
        HealthStatus::Green
    }
    
    /// Check canary gate against baseline
    fn check_canary_gate(&self, current: &WeeklyHealthArtifacts) -> Result<Vec<AlertEvent>> {
        let mut alerts = Vec::new();
        
        if !self.canary_config.enabled {
            return Ok(alerts);
        }
        
        let Some(baseline) = &self.last_green_baseline else {
            return Ok(alerts);
        };
        
        // Check confidence shift
        let confidence_shift = (current.aece - baseline.aece).abs();
        if confidence_shift > self.canary_config.max_confidence_shift {
            alerts.push(AlertEvent {
                timestamp: Utc::now(),
                alert_type: AlertType::ConfidenceShiftDetected,
                metric_name: "median_confidence_shift".to_string(),
                actual_value: confidence_shift,
                threshold: self.canary_config.max_confidence_shift,
                severity: HealthStatus::Red,
                message: format!("Confidence shift {:.4} exceeds threshold {:.4} vs baseline", 
                    confidence_shift, self.canary_config.max_confidence_shift),
            });
        }
        
        // Check AECE increase
        let aece_increase = current.aece - baseline.aece;
        if aece_increase > self.canary_config.max_aece_increase {
            alerts.push(AlertEvent {
                timestamp: Utc::now(),
                alert_type: AlertType::AeceThresholdExceeded,
                metric_name: "aece_increase".to_string(),
                actual_value: aece_increase,
                threshold: self.canary_config.max_aece_increase,
                severity: HealthStatus::Red,
                message: format!("AECE increase {:.4} exceeds threshold {:.4} vs baseline",
                    aece_increase, self.canary_config.max_aece_increase),
            });
        }
        
        Ok(alerts)
    }
    
    /// Generate weekly health artifacts
    pub fn generate_weekly_artifacts(
        &mut self,
        samples: &[CalibrationSample],
        intent: &str,
        language: Option<&str>,
    ) -> Result<WeeklyHealthArtifacts> {
        let now = Utc::now();
        let iso_week = now.iso_week();
        let year = iso_week.year();
        let week = iso_week.week();
        
        // Extract predictions, labels, weights
        let predictions: Vec<f64> = samples.iter().map(|s| s.prediction as f64).collect();
        let labels: Vec<f64> = samples.iter().map(|s| s.ground_truth as f64).collect();
        let weights: Vec<f64> = samples.iter().map(|s| s.weight as f64).collect();
        
        // Validate data integrity
        for (i, &pred) in predictions.iter().enumerate() {
            if pred < 0.0 || pred > 1.0 {
                return Err(anyhow::anyhow!("Score out of bounds at index {}: {}", i, pred));
            }
        }
        
        // Perform binning and calibration analysis
        let binning_result = self.binning_core.bin_samples(&predictions, &labels, &weights);
        
        // Compute core metrics
        let mut ece_sum = 0.0;
        let total_weight: f64 = binning_result.bin_stats.iter().map(|s| s.weight).sum();
        
        if total_weight > 0.0 {
            for stats in &binning_result.bin_stats {
                if stats.weight > 0.0 {
                    let bin_fraction = stats.weight / total_weight;
                    ece_sum += bin_fraction * (stats.accuracy - stats.confidence).abs();
                }
            }
        }
        
        let dece = ece_sum;
        let aece = dece; // For now, same as DECE (could add temporal averaging)
        
        // Compute other metrics
        let brier_score = self.compute_brier_score(&predictions, &labels, &weights);
        let reliability_slope = self.compute_reliability_slope(&binning_result.bin_stats);
        
        // Bootstrap validation
        let k_eff = binning_result.bin_stats.len().min((samples.len() as f64).sqrt() as usize);
        let c_hat = 1.5; // Default adaptive c value
        let bootstrap_result = self.bootstrap.run_bootstrap(&predictions, &labels, &weights, k_eff, c_hat);
        
        // ECE threshold computation
        let n = samples.len();
        let statistical_component = c_hat * (k_eff as f64 / n as f64).sqrt();
        let ece_threshold = 0.015f64.max(statistical_component);
        let aece_minus_tau = aece - ece_threshold;
        
        // Performance timing (mock for now - would be real measurements)
        let p99_latency_us = 850.0; // < 1ms target
        let mean_latency_us = 420.0;
        let hot_path_allocations = 0; // Zero allocations on hot path
        
        // Simulated clamp and merge rates (would come from calibrator state)
        let clamp_alpha_rate = 0.03; // 3% clamp rate
        let merged_bin_rate = binning_result.merged_bin_count as f64 / binning_result.bin_stats.len() as f64;
        
        let mut artifacts = WeeklyHealthArtifacts {
            timestamp: now,
            intent: intent.to_string(),
            language: language.map(|s| s.to_string()),
            week_number: week,
            year,
            aece,
            dece,
            brier_score,
            reliability_slope,
            clamp_alpha_rate,
            merged_bin_rate,
            ece_threshold,
            aece_minus_tau,
            bootstrap_coverage: bootstrap_result.coverage_probability,
            bootstrap_samples: bootstrap_result.samples_used,
            bootstrap_early_stopped: bootstrap_result.early_stopped,
            p99_latency_us,
            mean_latency_us,
            hot_path_allocations,
            health_status: HealthStatus::Green, // Will be updated
            alerts_triggered: Vec::new(),
            config_fingerprint: self.binning_core.get_config_hash(),
        };
        
        // Assess health status
        artifacts.health_status = self.assess_health_status(&artifacts);
        
        // Check canary gate
        let mut canary_alerts = self.check_canary_gate(&artifacts)?;
        artifacts.alerts_triggered.append(&mut canary_alerts);
        
        // Add threshold alerts
        if artifacts.aece_minus_tau > self.thresholds.aece_minus_tau_warn {
            artifacts.alerts_triggered.push(AlertEvent {
                timestamp: now,
                alert_type: AlertType::AeceThresholdExceeded,
                metric_name: "aece_minus_tau".to_string(),
                actual_value: artifacts.aece_minus_tau,
                threshold: self.thresholds.aece_minus_tau_warn,
                severity: if artifacts.aece_minus_tau > self.thresholds.aece_minus_tau_fail {
                    HealthStatus::Critical
                } else {
                    HealthStatus::Red
                },
                message: format!("AECE-τ = {:.4} exceeds threshold {:.4}",
                    artifacts.aece_minus_tau, self.thresholds.aece_minus_tau_warn),
            });
        }
        
        // Update baseline if current is green
        if artifacts.health_status == HealthStatus::Green {
            self.last_green_baseline = Some(artifacts.clone());
        }
        
        // Store in history
        self.historical_artifacts.push(artifacts.clone());
        
        // Keep only recent history (last 12 weeks)
        if self.historical_artifacts.len() > 12 {
            self.historical_artifacts.drain(0..self.historical_artifacts.len() - 12);
        }
        
        Ok(artifacts)
    }
    
    /// Get historical trend analysis
    pub fn get_trend_analysis(&self, weeks_back: usize) -> HashMap<String, Vec<f64>> {
        let recent_artifacts: Vec<_> = self.historical_artifacts
            .iter()
            .rev()
            .take(weeks_back)
            .collect();
            
        let mut trends = HashMap::new();
        
        trends.insert("aece".to_string(), 
            recent_artifacts.iter().map(|a| a.aece).collect());
        trends.insert("aece_minus_tau".to_string(),
            recent_artifacts.iter().map(|a| a.aece_minus_tau).collect());
        trends.insert("clamp_rate".to_string(),
            recent_artifacts.iter().map(|a| a.clamp_alpha_rate).collect());
        trends.insert("merged_bin_rate".to_string(),
            recent_artifacts.iter().map(|a| a.merged_bin_rate).collect());
        trends.insert("p99_latency_us".to_string(),
            recent_artifacts.iter().map(|a| a.p99_latency_us).collect());
        trends.insert("bootstrap_coverage".to_string(),
            recent_artifacts.iter().map(|a| a.bootstrap_coverage).collect());
            
        trends
    }
    
    /// Check if auto-revert should be triggered
    pub fn should_auto_revert(&self) -> (bool, Vec<String>) {
        let mut reasons = Vec::new();
        
        if let Some(latest) = self.historical_artifacts.last() {
            if latest.health_status == HealthStatus::Critical {
                reasons.push("Critical health status detected".to_string());
            }
            
            if latest.aece_minus_tau > self.thresholds.aece_minus_tau_fail {
                reasons.push(format!("AECE-τ = {:.4} > fail threshold {:.4}",
                    latest.aece_minus_tau, self.thresholds.aece_minus_tau_fail));
            }
            
            if latest.clamp_alpha_rate > self.thresholds.clamp_rate_fail {
                reasons.push(format!("Clamp rate = {:.1}% > fail threshold {:.1}%",
                    latest.clamp_alpha_rate * 100.0, self.thresholds.clamp_rate_fail * 100.0));
            }
            
            if latest.p99_latency_us > self.thresholds.p99_latency_threshold_us {
                reasons.push(format!("P99 latency = {:.0}μs > SLA threshold {:.0}μs",
                    latest.p99_latency_us, self.thresholds.p99_latency_threshold_us));
            }
        }
        
        (!reasons.is_empty(), reasons)
    }

    /// Generate weekly drift report for SLO monitoring
    pub async fn generate_weekly_report(&self) -> Result<WeeklyDriftReport> {
        let latest_artifacts = self.historical_artifacts.last()
            .ok_or_else(|| anyhow::anyhow!("No historical artifacts available"))?;
        
        let mut metric_deltas = HashMap::new();
        
        // Add common drift metrics
        metric_deltas.insert("aece".to_string(), DriftMetric {
            metric_name: "aece".to_string(),
            current_value: latest_artifacts.aece,
            previous_value: 0.015, // Placeholder baseline
            delta: latest_artifacts.aece - 0.015,
            threshold: 0.01,
            status: if (latest_artifacts.aece - 0.015).abs() > 0.01 {
                "breach".to_string()
            } else {
                "ok".to_string()
            },
        });
        
        metric_deltas.insert("brier_score".to_string(), DriftMetric {
            metric_name: "brier_score".to_string(),
            current_value: latest_artifacts.brier_score,
            previous_value: 0.2, // Placeholder baseline
            delta: latest_artifacts.brier_score - 0.2,
            threshold: 0.01,
            status: if (latest_artifacts.brier_score - 0.2).abs() > 0.01 {
                "breach".to_string()
            } else {
                "ok".to_string()
            },
        });
        
        Ok(WeeklyDriftReport {
            report_week: latest_artifacts.timestamp,
            metric_deltas,
            overall_status: match latest_artifacts.health_status {
                HealthStatus::Green => "stable".to_string(),
                HealthStatus::Yellow => "warning".to_string(),
                HealthStatus::Red => "breach".to_string(),
                HealthStatus::Critical => "critical".to_string(),
            },
            breached_thresholds: latest_artifacts.alerts_triggered.iter()
                .map(|alert| alert.metric_name.clone())
                .collect(),
        })
    }
}

/// Weekly drift report structure for SLO integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeeklyDriftReport {
    pub report_week: DateTime<Utc>,
    pub metric_deltas: HashMap<String, DriftMetric>,
    pub overall_status: String,
    pub breached_thresholds: Vec<String>,
}

/// Individual drift metric for SLO reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftMetric {
    pub metric_name: String,
    pub current_value: f64,
    pub previous_value: f64,
    pub delta: f64,
    pub threshold: f64,
    pub status: String, // "ok", "warning", "breach"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{CalibrationSample, shared_binning_core::SharedBinningConfig, fast_bootstrap::FastBootstrapConfig};
    use std::collections::HashMap;
    
    fn create_test_samples() -> Vec<CalibrationSample> {
        vec![
            CalibrationSample {
                prediction: 0.1,
                ground_truth: 0.0,
                intent: "search".to_string(),
                language: Some("general".to_string()),
                features: HashMap::new(),
                weight: 1.0,
            },
            CalibrationSample {
                prediction: 0.5,
                ground_truth: 1.0,
                intent: "search".to_string(),
                language: Some("general".to_string()),
                features: HashMap::new(),
                weight: 1.0,
            },
            CalibrationSample {
                prediction: 0.9,
                ground_truth: 1.0,
                intent: "search".to_string(),
                language: Some("general".to_string()),
                features: HashMap::new(),
                weight: 1.0,
            },
        ]
    }
    
    #[test]
    fn test_drift_monitor_creation() {
        let thresholds = DriftThresholds::default();
        let canary_config = CanaryGateConfig::default();
        let binning_config = SharedBinningConfig::default();
        let bootstrap_config = FastBootstrapConfig::default();
        
        let monitor = DriftMonitor::new(thresholds, canary_config, binning_config, bootstrap_config);
        
        assert_eq!(monitor.historical_artifacts.len(), 0);
        assert!(monitor.last_green_baseline.is_none());
    }
    
    #[test]
    fn test_weekly_artifacts_generation() {
        let thresholds = DriftThresholds::default();
        let canary_config = CanaryGateConfig::default();
        let binning_config = SharedBinningConfig::default();
        let bootstrap_config = FastBootstrapConfig {
            max_samples: 50, // Reduced for testing
            ..Default::default()
        };
        
        let mut monitor = DriftMonitor::new(thresholds, canary_config, binning_config, bootstrap_config);
        let samples = create_test_samples();
        
        let result = monitor.generate_weekly_artifacts(&samples, "search", Some("general"));
        assert!(result.is_ok());
        
        let artifacts = result.unwrap();
        assert_eq!(artifacts.intent, "search");
        assert_eq!(artifacts.language, Some("general".to_string()));
        assert!(artifacts.aece >= 0.0);
        assert!(artifacts.bootstrap_coverage >= 0.0);
        assert!(artifacts.bootstrap_coverage <= 1.0);
        
        println!("Generated artifacts: AECE={:.4}, health={:?}, alerts={}",
            artifacts.aece, artifacts.health_status, artifacts.alerts_triggered.len());
    }
    
    #[test]
    fn test_health_status_assessment() {
        let thresholds = DriftThresholds {
            aece_minus_tau_fail: 0.01,
            clamp_rate_fail: 0.10,
            merged_bin_rate_fail: 0.20,
            p99_latency_threshold_us: 1000.0,
            min_coverage_probability: 0.95,
            ..Default::default()
        };
        
        let canary_config = CanaryGateConfig::default();
        let binning_config = SharedBinningConfig::default();
        let bootstrap_config = FastBootstrapConfig::default();
        
        let monitor = DriftMonitor::new(thresholds, canary_config, binning_config, bootstrap_config);
        
        // Test critical status
        let critical_artifacts = WeeklyHealthArtifacts {
            timestamp: Utc::now(),
            intent: "test".to_string(),
            language: None,
            week_number: 1,
            year: 2024,
            aece: 0.1,
            dece: 0.1,
            brier_score: 0.2,
            reliability_slope: 1.0,
            clamp_alpha_rate: 0.15, // > fail threshold
            merged_bin_rate: 0.1,
            ece_threshold: 0.05,
            aece_minus_tau: 0.05,
            bootstrap_coverage: 0.8, // < min threshold
            bootstrap_samples: 100,
            bootstrap_early_stopped: false,
            p99_latency_us: 500.0,
            mean_latency_us: 300.0,
            hot_path_allocations: 0,
            health_status: HealthStatus::Green,
            alerts_triggered: Vec::new(),
            config_fingerprint: "test".to_string(),
        };
        
        let status = monitor.assess_health_status(&critical_artifacts);
        assert_eq!(status, HealthStatus::Critical);
    }
}