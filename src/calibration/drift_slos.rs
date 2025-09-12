//! Weekly Drift SLO Monitoring System
//! 
//! Implements comprehensive SLO monitoring for calibration drift with automated
//! alerting and severity classification. Ensures invisible utility operation
//! through proactive monitoring and intervention.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{error, warn, info, debug};

/// Weekly drift SLO thresholds for calibration metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftSlos {
    /// Absolute Expected Calibration Error threshold: |ΔAECE| < 0.01
    pub aece_threshold: f64,
    /// Distributional Expected Calibration Error threshold: |ΔDECE| < 0.01
    pub dece_threshold: f64,
    /// Alpha parameter drift threshold: |Δα| < 0.05
    pub alpha_threshold: f64,
    /// Clamp rate threshold: ≤ 10%
    pub clamp_rate_threshold: f64,
    /// Merged bin warning threshold: ≤ 5%
    pub merged_bin_warn_threshold: f64,
    /// Merged bin failure threshold: > 20%
    pub merged_bin_fail_threshold: f64,
}

impl Default for DriftSlos {
    fn default() -> Self {
        Self {
            aece_threshold: 0.01,
            dece_threshold: 0.01,
            alpha_threshold: 0.05,
            clamp_rate_threshold: 0.10,
            merged_bin_warn_threshold: 0.05,
            merged_bin_fail_threshold: 0.20,
        }
    }
}

/// Severity levels for SLO violations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Critical: Immediate action required, system degraded
    Critical,
    /// High: Urgent attention needed, approaching failure
    High,
    /// Medium: Warning condition, monitoring required
    Medium,
    /// Low: Information only, no action needed
    Low,
}

/// Calibration drift metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationMetrics {
    pub timestamp: u64,
    pub aece: f64,
    pub dece: f64,
    pub alpha: f64,
    pub clamp_rate: f64,
    pub merged_bin_rate: f64,
    pub score_range_violations: u64,
    pub mask_mismatch_count: u64,
    pub total_samples: u64,
}

/// SLO violation details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SloViolation {
    pub metric_name: String,
    pub current_value: f64,
    pub threshold: f64,
    pub drift: f64,
    pub severity: AlertSeverity,
    pub timestamp: u64,
    pub context: HashMap<String, String>,
}

/// Weekly drift monitoring and SLO enforcement system
#[derive(Debug)]
pub struct WeeklyDriftMonitor {
    slos: DriftSlos,
    baseline_metrics: Option<CalibrationMetrics>,
    violations: Vec<SloViolation>,
    last_check_time: SystemTime,
}

impl WeeklyDriftMonitor {
    /// Create a new weekly drift monitor with default SLOs
    pub fn new() -> Self {
        Self {
            slos: DriftSlos::default(),
            baseline_metrics: None,
            violations: Vec::new(),
            last_check_time: SystemTime::now(),
        }
    }

    /// Create monitor with custom SLO thresholds
    pub fn with_slos(slos: DriftSlos) -> Self {
        Self {
            slos,
            baseline_metrics: None,
            violations: Vec::new(),
            last_check_time: SystemTime::now(),
        }
    }

    /// Establish baseline metrics for drift comparison
    pub fn set_baseline(&mut self, metrics: CalibrationMetrics) {
        info!(
            "Establishing calibration baseline: AECE={:.4}, DECE={:.4}, α={:.4}",
            metrics.aece, metrics.dece, metrics.alpha
        );
        self.baseline_metrics = Some(metrics);
    }

    /// Check current metrics against SLOs and generate alerts
    pub fn check_slos(&mut self, current_metrics: CalibrationMetrics) -> Vec<SloViolation> {
        let mut violations = Vec::new();
        let timestamp = current_metrics.timestamp;

        // Validate score range (scores must be ∈ [0,1])
        if current_metrics.score_range_violations > 0 {
            violations.push(SloViolation {
                metric_name: "score_range_violations".to_string(),
                current_value: current_metrics.score_range_violations as f64,
                threshold: 0.0,
                drift: current_metrics.score_range_violations as f64,
                severity: AlertSeverity::Critical,
                timestamp,
                context: self.build_context("Score range validation failed", &current_metrics),
            });
        }

        // Validate mask mismatch detection
        if current_metrics.mask_mismatch_count > 0 {
            violations.push(SloViolation {
                metric_name: "mask_mismatch_count".to_string(),
                current_value: current_metrics.mask_mismatch_count as f64,
                threshold: 0.0,
                drift: current_metrics.mask_mismatch_count as f64,
                severity: AlertSeverity::High,
                timestamp,
                context: self.build_context("Mask mismatch detected", &current_metrics),
            });
        }

        // Check clamp rate SLO
        if current_metrics.clamp_rate > self.slos.clamp_rate_threshold {
            violations.push(SloViolation {
                metric_name: "clamp_rate".to_string(),
                current_value: current_metrics.clamp_rate,
                threshold: self.slos.clamp_rate_threshold,
                drift: current_metrics.clamp_rate - self.slos.clamp_rate_threshold,
                severity: if current_metrics.clamp_rate > 0.15 { 
                    AlertSeverity::High 
                } else { 
                    AlertSeverity::Medium 
                },
                timestamp,
                context: self.build_context("Clamp rate exceeded threshold", &current_metrics),
            });
        }

        // Check merged bin rate SLOs
        if current_metrics.merged_bin_rate > self.slos.merged_bin_fail_threshold {
            violations.push(SloViolation {
                metric_name: "merged_bin_rate".to_string(),
                current_value: current_metrics.merged_bin_rate,
                threshold: self.slos.merged_bin_fail_threshold,
                drift: current_metrics.merged_bin_rate - self.slos.merged_bin_fail_threshold,
                severity: AlertSeverity::Critical,
                timestamp,
                context: self.build_context("Merged bin rate failure threshold exceeded", &current_metrics),
            });
        } else if current_metrics.merged_bin_rate > self.slos.merged_bin_warn_threshold {
            violations.push(SloViolation {
                metric_name: "merged_bin_rate".to_string(),
                current_value: current_metrics.merged_bin_rate,
                threshold: self.slos.merged_bin_warn_threshold,
                drift: current_metrics.merged_bin_rate - self.slos.merged_bin_warn_threshold,
                severity: AlertSeverity::Medium,
                timestamp,
                context: self.build_context("Merged bin rate warning threshold exceeded", &current_metrics),
            });
        }

        // Check drift metrics against baseline if available
        if let Some(baseline) = &self.baseline_metrics {
            self.check_drift_slos(baseline, &current_metrics, &mut violations);
        }

        // Store violations and log alerts
        for violation in &violations {
            self.log_violation(violation);
        }
        self.violations.extend(violations.clone());
        self.last_check_time = SystemTime::now();

        violations
    }

    /// Check drift-specific SLOs against baseline
    fn check_drift_slos(
        &self,
        baseline: &CalibrationMetrics,
        current: &CalibrationMetrics,
        violations: &mut Vec<SloViolation>,
    ) {
        let timestamp = current.timestamp;

        // Check AECE drift: |ΔAECE| < 0.01
        let aece_drift = (current.aece - baseline.aece).abs();
        if aece_drift > self.slos.aece_threshold {
            violations.push(SloViolation {
                metric_name: "aece_drift".to_string(),
                current_value: current.aece,
                threshold: self.slos.aece_threshold,
                drift: aece_drift,
                severity: self.classify_drift_severity(aece_drift, self.slos.aece_threshold),
                timestamp,
                context: self.build_drift_context("AECE drift exceeded", baseline, current),
            });
        }

        // Check DECE drift: |ΔDECE| < 0.01
        let dece_drift = (current.dece - baseline.dece).abs();
        if dece_drift > self.slos.dece_threshold {
            violations.push(SloViolation {
                metric_name: "dece_drift".to_string(),
                current_value: current.dece,
                threshold: self.slos.dece_threshold,
                drift: dece_drift,
                severity: self.classify_drift_severity(dece_drift, self.slos.dece_threshold),
                timestamp,
                context: self.build_drift_context("DECE drift exceeded", baseline, current),
            });
        }

        // Check Alpha drift: |Δα| < 0.05
        let alpha_drift = (current.alpha - baseline.alpha).abs();
        if alpha_drift > self.slos.alpha_threshold {
            violations.push(SloViolation {
                metric_name: "alpha_drift".to_string(),
                current_value: current.alpha,
                threshold: self.slos.alpha_threshold,
                drift: alpha_drift,
                severity: self.classify_drift_severity(alpha_drift, self.slos.alpha_threshold),
                timestamp,
                context: self.build_drift_context("Alpha drift exceeded", baseline, current),
            });
        }
    }

    /// Classify drift severity based on threshold exceedance
    fn classify_drift_severity(&self, drift: f64, threshold: f64) -> AlertSeverity {
        let ratio = drift / threshold;
        if ratio > 3.0 {
            AlertSeverity::Critical
        } else if ratio > 2.0 {
            AlertSeverity::High
        } else if ratio > 1.5 {
            AlertSeverity::Medium
        } else {
            AlertSeverity::Low
        }
    }

    /// Build context information for SLO violations
    fn build_context(&self, message: &str, metrics: &CalibrationMetrics) -> HashMap<String, String> {
        let mut context = HashMap::new();
        context.insert("message".to_string(), message.to_string());
        context.insert("timestamp".to_string(), metrics.timestamp.to_string());
        context.insert("total_samples".to_string(), metrics.total_samples.to_string());
        context.insert("aece".to_string(), format!("{:.6}", metrics.aece));
        context.insert("dece".to_string(), format!("{:.6}", metrics.dece));
        context.insert("alpha".to_string(), format!("{:.6}", metrics.alpha));
        context.insert("clamp_rate".to_string(), format!("{:.4}", metrics.clamp_rate));
        context.insert("merged_bin_rate".to_string(), format!("{:.4}", metrics.merged_bin_rate));
        context
    }

    /// Build drift-specific context information
    fn build_drift_context(
        &self, 
        message: &str, 
        baseline: &CalibrationMetrics, 
        current: &CalibrationMetrics
    ) -> HashMap<String, String> {
        let mut context = self.build_context(message, current);
        context.insert("baseline_timestamp".to_string(), baseline.timestamp.to_string());
        context.insert("baseline_aece".to_string(), format!("{:.6}", baseline.aece));
        context.insert("baseline_dece".to_string(), format!("{:.6}", baseline.dece));
        context.insert("baseline_alpha".to_string(), format!("{:.6}", baseline.alpha));
        
        let time_diff = current.timestamp.saturating_sub(baseline.timestamp);
        context.insert("time_since_baseline_hours".to_string(), 
                      (time_diff / 3600).to_string());
        context
    }

    /// Log SLO violation with appropriate severity
    fn log_violation(&self, violation: &SloViolation) {
        let context_str = violation.context
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(" ");

        match violation.severity {
            AlertSeverity::Critical => {
                error!(
                    "CRITICAL SLO VIOLATION: {} = {:.6} (threshold: {:.6}, drift: {:.6}) - {}",
                    violation.metric_name,
                    violation.current_value,
                    violation.threshold,
                    violation.drift,
                    context_str
                );
            }
            AlertSeverity::High => {
                warn!(
                    "HIGH SLO VIOLATION: {} = {:.6} (threshold: {:.6}, drift: {:.6}) - {}",
                    violation.metric_name,
                    violation.current_value,
                    violation.threshold,
                    violation.drift,
                    context_str
                );
            }
            AlertSeverity::Medium => {
                warn!(
                    "MEDIUM SLO VIOLATION: {} = {:.6} (threshold: {:.6}, drift: {:.6}) - {}",
                    violation.metric_name,
                    violation.current_value,
                    violation.threshold,
                    violation.drift,
                    context_str
                );
            }
            AlertSeverity::Low => {
                info!(
                    "LOW SLO VIOLATION: {} = {:.6} (threshold: {:.6}, drift: {:.6}) - {}",
                    violation.metric_name,
                    violation.current_value,
                    violation.threshold,
                    violation.drift,
                    context_str
                );
            }
        }
    }

    /// Get all violations within the last week
    pub fn get_recent_violations(&self) -> Vec<SloViolation> {
        let week_ago = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .saturating_sub(7 * 24 * 3600);

        self.violations
            .iter()
            .filter(|v| v.timestamp >= week_ago)
            .cloned()
            .collect()
    }

    /// Generate weekly SLO report
    pub fn generate_weekly_report(&self) -> String {
        let violations = self.get_recent_violations();
        let critical_count = violations.iter().filter(|v| v.severity == AlertSeverity::Critical).count();
        let high_count = violations.iter().filter(|v| v.severity == AlertSeverity::High).count();
        let medium_count = violations.iter().filter(|v| v.severity == AlertSeverity::Medium).count();

        format!(
            "WEEKLY CALIBRATION SLO REPORT\n\
             ===============================\n\
             Total violations: {}\n\
             Critical: {}\n\
             High: {}\n\
             Medium: {}\n\
             Low: {}\n\n\
             SLO Thresholds:\n\
             - AECE drift: |Δ| < {:.3}\n\
             - DECE drift: |Δ| < {:.3}\n\
             - Alpha drift: |Δ| < {:.3}\n\
             - Clamp rate: ≤ {:.1}%\n\
             - Merged bins: warn ≤ {:.1}%, fail > {:.1}%\n\n\
             Status: {}",
            violations.len(),
            critical_count,
            high_count,
            medium_count,
            violations.len().saturating_sub(critical_count + high_count + medium_count),
            self.slos.aece_threshold,
            self.slos.dece_threshold,
            self.slos.alpha_threshold,
            self.slos.clamp_rate_threshold * 100.0,
            self.slos.merged_bin_warn_threshold * 100.0,
            self.slos.merged_bin_fail_threshold * 100.0,
            if critical_count > 0 || high_count > 3 {
                "🚨 ACTION REQUIRED"
            } else if high_count > 0 || medium_count > 5 {
                "⚠️ MONITORING REQUIRED"
            } else {
                "✅ OPERATING WITHIN SLO"
            }
        )
    }

    /// Reset violation history (typically done weekly)
    pub fn reset_violations(&mut self) {
        info!("Resetting SLO violation history, had {} violations", self.violations.len());
        self.violations.clear();
    }

    /// Update SLO thresholds (for operational tuning)
    pub fn update_slos(&mut self, new_slos: DriftSlos) {
        info!("Updating SLO thresholds: AECE {:.3}→{:.3}, DECE {:.3}→{:.3}, Alpha {:.3}→{:.3}",
              self.slos.aece_threshold, new_slos.aece_threshold,
              self.slos.dece_threshold, new_slos.dece_threshold,
              self.slos.alpha_threshold, new_slos.alpha_threshold);
        self.slos = new_slos;
    }

    /// Check if system is operating within all SLOs
    pub fn is_healthy(&self) -> bool {
        let recent_violations = self.get_recent_violations();
        let critical_count = recent_violations.iter().filter(|v| v.severity == AlertSeverity::Critical).count();
        let high_count = recent_violations.iter().filter(|v| v.severity == AlertSeverity::High).count();
        
        critical_count == 0 && high_count <= 2
    }

    /// Get current SLO status summary
    pub fn get_status_summary(&self) -> HashMap<String, String> {
        let violations = self.get_recent_violations();
        let mut summary = HashMap::new();
        
        summary.insert("healthy".to_string(), self.is_healthy().to_string());
        summary.insert("total_violations".to_string(), violations.len().to_string());
        summary.insert("critical_violations".to_string(), 
                      violations.iter().filter(|v| v.severity == AlertSeverity::Critical).count().to_string());
        summary.insert("high_violations".to_string(),
                      violations.iter().filter(|v| v.severity == AlertSeverity::High).count().to_string());
        summary.insert("last_check".to_string(), 
                      format!("{:.0}", self.last_check_time.duration_since(UNIX_EPOCH).unwrap().as_secs()));
        
        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_metrics(aece: f64, dece: f64, alpha: f64, clamp_rate: f64, merged_bin_rate: f64) -> CalibrationMetrics {
        CalibrationMetrics {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            aece,
            dece,
            alpha,
            clamp_rate,
            merged_bin_rate,
            score_range_violations: 0,
            mask_mismatch_count: 0,
            total_samples: 10000,
        }
    }

    #[test]
    fn test_slo_monitoring_healthy_state() {
        let mut monitor = WeeklyDriftMonitor::new();
        let baseline = create_test_metrics(0.02, 0.015, 0.5, 0.05, 0.02);
        monitor.set_baseline(baseline.clone());
        
        let current = create_test_metrics(0.021, 0.016, 0.52, 0.06, 0.03);
        let violations = monitor.check_slos(current);
        
        assert!(violations.is_empty(), "Should have no violations for healthy metrics");
        assert!(monitor.is_healthy());
    }

    #[test]
    fn test_aece_drift_violation() {
        let mut monitor = WeeklyDriftMonitor::new();
        let baseline = create_test_metrics(0.02, 0.015, 0.5, 0.05, 0.02);
        monitor.set_baseline(baseline.clone());
        
        // AECE drift of 0.015 > 0.01 threshold
        let current = create_test_metrics(0.035, 0.016, 0.52, 0.06, 0.03);
        let violations = monitor.check_slos(current);
        
        assert!(!violations.is_empty());
        assert!(violations.iter().any(|v| v.metric_name == "aece_drift"));
    }

    #[test]
    fn test_clamp_rate_violation() {
        let mut monitor = WeeklyDriftMonitor::new();
        
        // Clamp rate of 15% > 10% threshold
        let current = create_test_metrics(0.02, 0.015, 0.5, 0.15, 0.02);
        let violations = monitor.check_slos(current);
        
        assert!(!violations.is_empty());
        assert!(violations.iter().any(|v| v.metric_name == "clamp_rate"));
        assert_eq!(violations.iter().find(|v| v.metric_name == "clamp_rate").unwrap().severity, 
                   AlertSeverity::High);
    }

    #[test]
    fn test_merged_bin_rate_violations() {
        let mut monitor = WeeklyDriftMonitor::new();
        
        // Warning threshold: 7% > 5%
        let current = create_test_metrics(0.02, 0.015, 0.5, 0.05, 0.07);
        let violations = monitor.check_slos(current);
        assert!(violations.iter().any(|v| v.metric_name == "merged_bin_rate" && v.severity == AlertSeverity::Medium));
        
        // Failure threshold: 25% > 20%  
        let current = create_test_metrics(0.02, 0.015, 0.5, 0.05, 0.25);
        let violations = monitor.check_slos(current);
        assert!(violations.iter().any(|v| v.metric_name == "merged_bin_rate" && v.severity == AlertSeverity::Critical));
    }

    #[test]
    fn test_score_range_violation() {
        let mut monitor = WeeklyDriftMonitor::new();
        
        let mut current = create_test_metrics(0.02, 0.015, 0.5, 0.05, 0.02);
        current.score_range_violations = 5;
        
        let violations = monitor.check_slos(current);
        assert!(violations.iter().any(|v| v.metric_name == "score_range_violations"));
        assert_eq!(violations.iter().find(|v| v.metric_name == "score_range_violations").unwrap().severity,
                   AlertSeverity::Critical);
    }

    #[test]
    fn test_weekly_report_generation() {
        let mut monitor = WeeklyDriftMonitor::new();
        let current = create_test_metrics(0.02, 0.015, 0.5, 0.15, 0.25);
        monitor.check_slos(current);
        
        let report = monitor.generate_weekly_report();
        assert!(report.contains("WEEKLY CALIBRATION SLO REPORT"));
        assert!(report.contains("Total violations:"));
        assert!(report.contains("🚨 ACTION REQUIRED") || report.contains("⚠️ MONITORING REQUIRED"));
    }
}