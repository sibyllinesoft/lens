//! # Real-Time Calibration Monitoring & Alerting
//!
//! Continuous ECE monitoring and alerting system for PHASE 4 calibration.
//! Ensures ECE ≤ 0.015 is maintained in production with automatic alerts.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, info, warn, error};

use super::{AlertConfig, CalibrationResult};

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Target ECE threshold for monitoring
    pub target_ece: f32,
    /// Alert configuration
    pub alert_config: AlertConfig,
    /// Real-time monitoring enabled
    pub realtime_enabled: bool,
}

/// Real-time calibration monitor
#[derive(Debug, Clone)]
pub struct CalibrationMonitor {
    config: MonitoringConfig,
    /// Recent ECE measurements per slice
    slice_measurements: Arc<RwLock<HashMap<String, VecDeque<ECEMeasurement>>>>,
    /// Alert state tracking
    alert_state: Arc<RwLock<AlertState>>,
    /// System-wide ECE statistics
    global_stats: Arc<RwLock<GlobalECEStats>>,
    /// Monitoring start time
    start_time: SystemTime,
}

/// ECE measurement with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ECEMeasurement {
    /// Measured ECE value
    pub ece: f32,
    /// Number of samples in measurement
    pub sample_count: usize,
    /// Slice identifier (intent:language)
    pub slice: String,
    /// Measurement timestamp
    pub timestamp: SystemTime,
    /// Calibration method used
    pub method: String,
    /// Confidence in measurement
    pub confidence: f32,
}

/// Alert state tracking
#[derive(Debug, Clone)]
pub struct AlertState {
    /// Recent alerts with cooldown tracking
    recent_alerts: VecDeque<ECEAlert>,
    /// Last alert time per slice
    last_alert_per_slice: HashMap<String, SystemTime>,
    /// Alert count in current hour
    alerts_this_hour: usize,
    /// Hour start time for rate limiting
    hour_start: SystemTime,
}

/// ECE threshold violation alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ECEAlert {
    /// Alert ID
    pub id: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Slice that triggered alert
    pub slice: String,
    /// Current ECE value
    pub current_ece: f32,
    /// Threshold that was exceeded
    pub threshold: f32,
    /// Alert message
    pub message: String,
    /// Alert timestamp
    pub timestamp: SystemTime,
    /// Suggested actions
    pub suggested_actions: Vec<String>,
    /// Additional context
    pub context: HashMap<String, String>,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// ECE slightly above threshold
    Warning,
    /// ECE significantly above threshold
    Critical,
    /// ECE extremely high, immediate action needed
    Emergency,
}

/// Global ECE statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalECEStats {
    /// Overall system ECE
    pub overall_ece: f32,
    /// ECE by slice
    pub slice_eces: HashMap<String, f32>,
    /// ECE trend (positive = getting worse)
    pub ece_trend: f32,
    /// Number of slices exceeding threshold
    pub slices_exceeding_threshold: usize,
    /// Total measurements processed
    pub total_measurements: usize,
    /// Last update timestamp
    pub last_updated: SystemTime,
}

/// ECE monitoring report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringReport {
    /// Report timestamp
    pub timestamp: SystemTime,
    /// Overall compliance status
    pub compliant: bool,
    /// Global statistics
    pub global_stats: GlobalECEStats,
    /// Recent alerts
    pub recent_alerts: Vec<ECEAlert>,
    /// Slice-by-slice analysis
    pub slice_analysis: HashMap<String, SliceAnalysis>,
    /// System health score [0, 1]
    pub health_score: f32,
}

/// Analysis for a specific slice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliceAnalysis {
    /// Current ECE
    pub current_ece: f32,
    /// ECE trend over time
    pub trend: f32,
    /// Compliance status
    pub compliant: bool,
    /// Recent measurement count
    pub measurement_count: usize,
    /// Confidence in measurements
    pub confidence: f32,
    /// Time since last measurement
    pub time_since_last_measurement: Duration,
}

impl CalibrationMonitor {
    /// Create new calibration monitor
    pub async fn new(config: MonitoringConfig) -> Result<Self> {
        info!("Creating calibration monitor");
        info!("Target ECE: ≤{:.4}, Real-time: {}", config.target_ece, config.realtime_enabled);
        info!("Alert threshold: {:.4}, Cooldown: {}s", 
              config.alert_config.ece_alert_threshold,
              config.alert_config.alert_cooldown_seconds);

        Ok(Self {
            config,
            slice_measurements: Arc::new(RwLock::new(HashMap::new())),
            alert_state: Arc::new(RwLock::new(AlertState::new())),
            global_stats: Arc::new(RwLock::new(GlobalECEStats::default())),
            start_time: SystemTime::now(),
        })
    }

    /// Check ECE threshold for a calibration result
    pub async fn check_ece_threshold(&self, result: &CalibrationResult) -> Result<Option<ECEAlert>> {
        if !self.config.realtime_enabled {
            return Ok(None);
        }

        // Record measurement
        self.record_measurement(result).await?;

        // Check if ECE exceeds threshold
        if result.slice_ece > self.config.alert_config.ece_alert_threshold {
            let alert = self.generate_alert(result).await?;
            
            // Check if we should actually send this alert (rate limiting, cooldown)
            if self.should_send_alert(&alert).await? {
                self.record_alert(&alert).await?;
                warn!("ECE alert generated: {}", alert.message);
                return Ok(Some(alert));
            }
        }

        Ok(None)
    }

    /// Record a calibration measurement
    pub async fn record_measurement(&self, result: &CalibrationResult) -> Result<()> {
        let slice_key = self.make_slice_key(&result.intent, result.language.as_deref());
        
        let measurement = ECEMeasurement {
            ece: result.slice_ece,
            sample_count: 1, // Would be actual sample count in production
            slice: slice_key.clone(),
            timestamp: SystemTime::now(),
            method: format!("{:?}", result.method_used),
            confidence: result.calibration_confidence,
        };

        // Record in slice-specific measurements
        let mut slice_measurements = self.slice_measurements.write().await;
        let slice_queue = slice_measurements.entry(slice_key).or_insert_with(VecDeque::new);
        
        slice_queue.push_back(measurement);
        
        // Keep only recent measurements (last 1000 per slice)
        if slice_queue.len() > 1000 {
            slice_queue.pop_front();
        }

        // Update global statistics
        self.update_global_stats().await?;

        debug!("Recorded ECE measurement: slice={}, ECE={:.4}, method={:?}", 
               &result.intent, result.slice_ece, result.method_used);

        Ok(())
    }

    /// Get current monitoring report
    pub async fn get_monitoring_report(&self) -> Result<MonitoringReport> {
        let global_stats = self.global_stats.read().await.clone();
        let alert_state = self.alert_state.read().await;
        let recent_alerts: Vec<ECEAlert> = alert_state.recent_alerts
            .iter()
            .rev()
            .take(10)
            .cloned()
            .collect();

        // Generate slice analysis
        let slice_analysis = self.generate_slice_analysis().await?;
        
        // Calculate health score
        let health_score = self.calculate_health_score(&global_stats, &slice_analysis).await?;
        
        // Check overall compliance
        let compliant = global_stats.overall_ece <= self.config.target_ece &&
                       global_stats.slices_exceeding_threshold == 0;

        Ok(MonitoringReport {
            timestamp: SystemTime::now(),
            compliant,
            global_stats,
            recent_alerts,
            slice_analysis,
            health_score,
        })
    }

    /// Get uptime duration
    pub fn get_uptime(&self) -> Duration {
        SystemTime::now().duration_since(self.start_time).unwrap_or(Duration::ZERO)
    }

    /// Reset monitoring state (for testing/debugging)
    pub async fn reset_monitoring_state(&self) -> Result<()> {
        info!("Resetting calibration monitoring state");
        
        let mut slice_measurements = self.slice_measurements.write().await;
        slice_measurements.clear();
        
        let mut alert_state = self.alert_state.write().await;
        *alert_state = AlertState::new();
        
        let mut global_stats = self.global_stats.write().await;
        *global_stats = GlobalECEStats::default();
        
        Ok(())
    }

    /// Check system health
    pub async fn check_system_health(&self) -> Result<bool> {
        let report = self.get_monitoring_report().await?;
        
        // System is healthy if:
        // 1. Overall ECE is within target
        // 2. No slices exceed threshold
        // 3. Health score is above 0.8
        // 4. No recent emergency alerts
        
        let ece_healthy = report.global_stats.overall_ece <= self.config.target_ece;
        let slices_healthy = report.global_stats.slices_exceeding_threshold == 0;
        let score_healthy = report.health_score > 0.8;
        
        let recent_emergencies = report.recent_alerts.iter()
            .any(|alert| matches!(alert.severity, AlertSeverity::Emergency));
        
        let healthy = ece_healthy && slices_healthy && score_healthy && !recent_emergencies;
        
        if !healthy {
            warn!("System health check failed: ECE={:.4}, exceeding_slices={}, health={:.3}", 
                  report.global_stats.overall_ece,
                  report.global_stats.slices_exceeding_threshold,
                  report.health_score);
        }
        
        Ok(healthy)
    }

    // Private implementation methods

    fn make_slice_key(&self, intent: &str, language: Option<&str>) -> String {
        match language {
            Some(lang) => format!("{}:{}", intent, lang),
            None => intent.to_string(),
        }
    }

    async fn generate_alert(&self, result: &CalibrationResult) -> Result<ECEAlert> {
        let severity = if result.slice_ece > self.config.target_ece * 3.0 {
            AlertSeverity::Emergency
        } else if result.slice_ece > self.config.target_ece * 2.0 {
            AlertSeverity::Critical
        } else {
            AlertSeverity::Warning
        };

        let slice_key = self.make_slice_key(&result.intent, result.language.as_deref());
        
        let id = format!("ece_alert_{}_{}", 
                        slice_key.replace(":", "_"),
                        SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis());

        let message = format!(
            "ECE threshold violation: slice '{}' has ECE {:.4} > threshold {:.4}",
            slice_key, result.slice_ece, self.config.alert_config.ece_alert_threshold
        );

        let suggested_actions = match severity {
            AlertSeverity::Emergency => vec![
                "Immediately investigate calibration system".to_string(),
                "Consider rolling back recent changes".to_string(),
                "Enable emergency fallback calibration".to_string(),
            ],
            AlertSeverity::Critical => vec![
                "Review calibration model for this slice".to_string(),
                "Check for data drift or quality issues".to_string(),
                "Consider retraining slice-specific calibrator".to_string(),
            ],
            AlertSeverity::Warning => vec![
                "Monitor trend for continued degradation".to_string(),
                "Check recent calibration samples".to_string(),
            ],
        };

        let mut context = HashMap::new();
        context.insert("method".to_string(), format!("{:?}", result.method_used));
        context.insert("confidence".to_string(), result.calibration_confidence.to_string());
        context.insert("input_score".to_string(), result.input_score.to_string());
        context.insert("calibrated_score".to_string(), result.calibrated_score.to_string());

        Ok(ECEAlert {
            id,
            severity,
            slice: slice_key,
            current_ece: result.slice_ece,
            threshold: self.config.alert_config.ece_alert_threshold,
            message,
            timestamp: SystemTime::now(),
            suggested_actions,
            context,
        })
    }

    async fn should_send_alert(&self, alert: &ECEAlert) -> Result<bool> {
        let mut alert_state = self.alert_state.write().await;
        
        // Check rate limiting
        let now = SystemTime::now();
        
        // Reset hourly counter if needed
        if now.duration_since(alert_state.hour_start)? > Duration::from_secs(3600) {
            alert_state.alerts_this_hour = 0;
            alert_state.hour_start = now;
        }
        
        // Check if we've exceeded max alerts per hour
        if alert_state.alerts_this_hour >= self.config.alert_config.max_alerts_per_hour {
            debug!("Alert rate limit exceeded, suppressing alert for slice {}", alert.slice);
            return Ok(false);
        }
        
        // Check cooldown for this slice
        if let Some(last_alert_time) = alert_state.last_alert_per_slice.get(&alert.slice) {
            let cooldown = Duration::from_secs(self.config.alert_config.alert_cooldown_seconds);
            if now.duration_since(*last_alert_time)? < cooldown {
                debug!("Alert cooldown active for slice {}, suppressing alert", alert.slice);
                return Ok(false);
            }
        }
        
        Ok(true)
    }

    async fn record_alert(&self, alert: &ECEAlert) -> Result<()> {
        let mut alert_state = self.alert_state.write().await;
        
        // Record the alert
        alert_state.recent_alerts.push_back(alert.clone());
        
        // Keep only recent alerts (last 100)
        if alert_state.recent_alerts.len() > 100 {
            alert_state.recent_alerts.pop_front();
        }
        
        // Update per-slice tracking
        alert_state.last_alert_per_slice.insert(alert.slice.clone(), alert.timestamp);
        alert_state.alerts_this_hour += 1;
        
        Ok(())
    }

    async fn update_global_stats(&self) -> Result<()> {
        let slice_measurements = self.slice_measurements.read().await;
        let mut global_stats = self.global_stats.write().await;
        
        // Calculate overall ECE and slice-specific ECEs
        let mut all_eces = Vec::new();
        let mut slice_eces = HashMap::new();
        let mut exceeding_count = 0;
        let mut total_measurements = 0;
        
        for (slice, measurements) in slice_measurements.iter() {
            if let Some(latest) = measurements.back() {
                let ece = latest.ece;
                all_eces.push(ece);
                slice_eces.insert(slice.clone(), ece);
                
                if ece > self.config.target_ece {
                    exceeding_count += 1;
                }
            }
            total_measurements += measurements.len();
        }
        
        let overall_ece = if all_eces.is_empty() {
            0.0
        } else {
            all_eces.iter().sum::<f32>() / all_eces.len() as f32
        };
        
        // Calculate trend (simplified - would use proper time series analysis)
        let ece_trend = if global_stats.overall_ece > 0.0 {
            overall_ece - global_stats.overall_ece
        } else {
            0.0
        };
        
        *global_stats = GlobalECEStats {
            overall_ece,
            slice_eces,
            ece_trend,
            slices_exceeding_threshold: exceeding_count,
            total_measurements,
            last_updated: SystemTime::now(),
        };
        
        Ok(())
    }

    async fn generate_slice_analysis(&self) -> Result<HashMap<String, SliceAnalysis>> {
        let slice_measurements = self.slice_measurements.read().await;
        let mut analysis = HashMap::new();
        
        for (slice, measurements) in slice_measurements.iter() {
            if measurements.is_empty() {
                continue;
            }
            
            let latest = measurements.back().unwrap();
            let current_ece = latest.ece;
            
            // Calculate trend (simple: compare first and last measurements)
            let trend = if measurements.len() > 1 {
                let first = measurements.front().unwrap();
                current_ece - first.ece
            } else {
                0.0
            };
            
            let compliant = current_ece <= self.config.target_ece;
            let measurement_count = measurements.len();
            
            // Calculate average confidence
            let confidence = measurements.iter()
                .map(|m| m.confidence)
                .sum::<f32>() / measurements.len() as f32;
            
            let time_since_last = SystemTime::now()
                .duration_since(latest.timestamp)
                .unwrap_or(Duration::ZERO);
            
            analysis.insert(slice.clone(), SliceAnalysis {
                current_ece,
                trend,
                compliant,
                measurement_count,
                confidence,
                time_since_last_measurement: time_since_last,
            });
        }
        
        Ok(analysis)
    }

    async fn calculate_health_score(
        &self,
        global_stats: &GlobalECEStats,
        slice_analysis: &HashMap<String, SliceAnalysis>,
    ) -> Result<f32> {
        let mut score = 1.0;
        
        // Penalize overall ECE being above target
        if global_stats.overall_ece > self.config.target_ece {
            let penalty = (global_stats.overall_ece / self.config.target_ece - 1.0) * 0.5;
            score -= penalty.min(0.5);
        }
        
        // Penalize slices exceeding threshold
        let slice_count = slice_analysis.len().max(1) as f32;
        let exceeding_ratio = global_stats.slices_exceeding_threshold as f32 / slice_count;
        score -= exceeding_ratio * 0.3;
        
        // Penalize negative trends
        if global_stats.ece_trend > 0.0 {
            score -= (global_stats.ece_trend * 10.0).min(0.2);
        }
        
        // Penalize low confidence measurements
        let avg_confidence = slice_analysis.values()
            .map(|s| s.confidence)
            .sum::<f32>() / slice_analysis.len().max(1) as f32;
        
        if avg_confidence < 0.7 {
            score -= (0.7 - avg_confidence) * 0.2;
        }
        
        Ok(score.clamp(0.0, 1.0))
    }
}

impl AlertState {
    fn new() -> Self {
        Self {
            recent_alerts: VecDeque::new(),
            last_alert_per_slice: HashMap::new(),
            alerts_this_hour: 0,
            hour_start: SystemTime::now(),
        }
    }
}

impl Default for GlobalECEStats {
    fn default() -> Self {
        Self {
            overall_ece: 0.0,
            slice_eces: HashMap::new(),
            ece_trend: 0.0,
            slices_exceeding_threshold: 0,
            total_measurements: 0,
            last_updated: SystemTime::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{CalibrationMethod, CalibrationResult};

    fn create_test_result(ece: f32, intent: &str, language: Option<&str>) -> CalibrationResult {
        CalibrationResult {
            input_score: 0.5,
            calibrated_score: 0.5,
            method_used: CalibrationMethod::IsotonicRegression { slope: 1.0 },
            intent: intent.to_string(),
            language: language.map(|s| s.to_string()),
            slice_ece: ece,
            calibration_confidence: 0.8,
        }
    }

    #[tokio::test]
    async fn test_monitor_creation() {
        let config = MonitoringConfig {
            target_ece: 0.015,
            alert_config: AlertConfig::default(),
            realtime_enabled: true,
        };
        
        let monitor = CalibrationMonitor::new(config).await.unwrap();
        assert!(monitor.get_uptime() >= Duration::ZERO);
        
        let healthy = monitor.check_system_health().await.unwrap();
        assert!(healthy); // Should start healthy
    }

    #[tokio::test]
    async fn test_ece_threshold_monitoring() {
        let config = MonitoringConfig {
            target_ece: 0.015,
            alert_config: AlertConfig {
                ece_alert_threshold: 0.02,
                max_alerts_per_hour: 10,
                alert_cooldown_seconds: 60,
                ..Default::default()
            },
            realtime_enabled: true,
        };
        
        let monitor = CalibrationMonitor::new(config).await.unwrap();
        
        // Test normal ECE - should not alert
        let normal_result = create_test_result(0.01, "exact_match", Some("rust"));
        let alert = monitor.check_ece_threshold(&normal_result).await.unwrap();
        assert!(alert.is_none());
        
        // Test high ECE - should alert
        let high_ece_result = create_test_result(0.025, "semantic", Some("python"));
        let alert = monitor.check_ece_threshold(&high_ece_result).await.unwrap();
        assert!(alert.is_some());
        
        let alert = alert.unwrap();
        assert_eq!(alert.slice, "semantic:python");
        assert_eq!(alert.current_ece, 0.025);
    }

    #[tokio::test]
    async fn test_alert_rate_limiting() {
        let config = MonitoringConfig {
            target_ece: 0.015,
            alert_config: AlertConfig {
                ece_alert_threshold: 0.02,
                max_alerts_per_hour: 2, // Very low limit
                alert_cooldown_seconds: 10,
                ..Default::default()
            },
            realtime_enabled: true,
        };
        
        let monitor = CalibrationMonitor::new(config).await.unwrap();
        let high_ece_result = create_test_result(0.03, "test", Some("rust"));
        
        // First alert should succeed
        let alert1 = monitor.check_ece_threshold(&high_ece_result).await.unwrap();
        assert!(alert1.is_some());
        
        // Second alert should succeed
        let alert2 = monitor.check_ece_threshold(&high_ece_result).await.unwrap();
        assert!(alert2.is_some());
        
        // Third alert should be rate-limited
        let alert3 = monitor.check_ece_threshold(&high_ece_result).await.unwrap();
        assert!(alert3.is_none());
    }

    #[tokio::test]
    async fn test_alert_cooldown() {
        let config = MonitoringConfig {
            target_ece: 0.015,
            alert_config: AlertConfig {
                ece_alert_threshold: 0.02,
                max_alerts_per_hour: 100, // High limit
                alert_cooldown_seconds: 2, // Short cooldown for testing
                ..Default::default()
            },
            realtime_enabled: true,
        };
        
        let monitor = CalibrationMonitor::new(config).await.unwrap();
        let high_ece_result = create_test_result(0.025, "test", Some("rust"));
        
        // First alert should succeed
        let alert1 = monitor.check_ece_threshold(&high_ece_result).await.unwrap();
        assert!(alert1.is_some());
        
        // Immediate second alert should be blocked by cooldown
        let alert2 = monitor.check_ece_threshold(&high_ece_result).await.unwrap();
        assert!(alert2.is_none());
        
        // Wait for cooldown and try again
        tokio::time::sleep(Duration::from_secs(3)).await;
        let alert3 = monitor.check_ece_threshold(&high_ece_result).await.unwrap();
        assert!(alert3.is_some());
    }

    #[tokio::test]
    async fn test_monitoring_report() {
        let config = MonitoringConfig {
            target_ece: 0.015,
            alert_config: AlertConfig::default(),
            realtime_enabled: true,
        };
        
        let monitor = CalibrationMonitor::new(config).await.unwrap();
        
        // Record some measurements
        let result1 = create_test_result(0.01, "exact_match", Some("rust"));
        let result2 = create_test_result(0.02, "semantic", Some("python"));
        
        monitor.record_measurement(&result1).await.unwrap();
        monitor.record_measurement(&result2).await.unwrap();
        
        let report = monitor.get_monitoring_report().await.unwrap();
        
        // Should have measurements
        assert!(report.global_stats.total_measurements > 0);
        assert!(report.slice_analysis.len() > 0);
        assert!(report.health_score > 0.0);
        
        // Check slice analysis
        assert!(report.slice_analysis.contains_key("exact_match:rust"));
        assert!(report.slice_analysis.contains_key("semantic:python"));
    }

    #[tokio::test]
    async fn test_system_health_check() {
        let config = MonitoringConfig {
            target_ece: 0.015,
            alert_config: AlertConfig {
                ece_alert_threshold: 0.02,
                ..Default::default()
            },
            realtime_enabled: true,
        };
        
        let monitor = CalibrationMonitor::new(config).await.unwrap();
        
        // Should start healthy
        assert!(monitor.check_system_health().await.unwrap());
        
        // Add measurements that keep system healthy
        let good_result = create_test_result(0.01, "exact_match", Some("rust"));
        monitor.record_measurement(&good_result).await.unwrap();
        assert!(monitor.check_system_health().await.unwrap());
        
        // Add measurement that makes system unhealthy
        let bad_result = create_test_result(0.05, "semantic", Some("python"));
        monitor.record_measurement(&bad_result).await.unwrap();
        
        // System should now be unhealthy due to high ECE
        let healthy = monitor.check_system_health().await.unwrap();
        assert!(!healthy);
    }

    #[tokio::test]
    async fn test_alert_severity_classification() {
        let config = MonitoringConfig {
            target_ece: 0.015,
            alert_config: AlertConfig {
                ece_alert_threshold: 0.02,
                ..Default::default()
            },
            realtime_enabled: true,
        };
        
        let monitor = CalibrationMonitor::new(config).await.unwrap();
        
        // Warning level (just above threshold)
        let warning_result = create_test_result(0.025, "test", Some("rust"));
        let alert = monitor.generate_alert(&warning_result).await.unwrap();
        assert!(matches!(alert.severity, AlertSeverity::Warning));
        
        // Critical level (2x target ECE)
        let critical_result = create_test_result(0.035, "test", Some("rust"));
        let alert = monitor.generate_alert(&critical_result).await.unwrap();
        assert!(matches!(alert.severity, AlertSeverity::Critical));
        
        // Emergency level (3x target ECE)
        let emergency_result = create_test_result(0.05, "test", Some("rust"));
        let alert = monitor.generate_alert(&emergency_result).await.unwrap();
        assert!(matches!(alert.severity, AlertSeverity::Emergency));
    }

    #[tokio::test]
    async fn test_state_reset() {
        let config = MonitoringConfig {
            target_ece: 0.015,
            alert_config: AlertConfig::default(),
            realtime_enabled: true,
        };
        
        let monitor = CalibrationMonitor::new(config).await.unwrap();
        
        // Add some measurements and alerts
        let result = create_test_result(0.03, "test", Some("rust"));
        monitor.check_ece_threshold(&result).await.unwrap();
        
        let report_before = monitor.get_monitoring_report().await.unwrap();
        assert!(report_before.global_stats.total_measurements > 0);
        
        // Reset state
        monitor.reset_monitoring_state().await.unwrap();
        
        let report_after = monitor.get_monitoring_report().await.unwrap();
        assert_eq!(report_after.global_stats.total_measurements, 0);
        assert_eq!(report_after.recent_alerts.len(), 0);
    }
}