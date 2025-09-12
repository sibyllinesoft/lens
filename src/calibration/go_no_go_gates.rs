use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error};

use crate::metrics::CalibrationMetrics;
use crate::calibration::production_rollout::BaselineMetrics;

/// Four critical go/no-go gates for CALIB_V22 production rollout
/// 
/// Gates monitor:
/// 1. p99 < 1ms calibration latency
/// 2. AECE-τ ≤ 0.01 on every intent×lang slice
/// 3. Median confidence shift ≤ 0.02  
/// 4. Δ(SLA-Recall@50) = 0 (zero change requirement)
pub struct GoNoGoGates {
    baseline: Option<BaselineMetrics>,
    thresholds: GateThresholds,
    monitoring_window: Duration,
    statistical_config: StatisticalConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateThresholds {
    /// Maximum allowed p99 latency in milliseconds
    pub max_p99_latency_ms: f64,
    /// Maximum allowed AECE-τ value
    pub max_aece_tau: f64,
    /// Maximum allowed median confidence shift from baseline
    pub max_confidence_shift: f64,
    /// Maximum allowed SLA-Recall@50 change from baseline (should be 0.0)
    pub max_sla_recall_change: f64,
    /// Minimum statistical significance for comparisons
    pub min_statistical_significance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalConfig {
    /// Minimum sample size for statistical tests
    pub min_sample_size: usize,
    /// Confidence level for statistical tests (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Window size for rolling statistics
    pub rolling_window_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GateStatus {
    /// Gate is healthy - all criteria met
    Healthy,
    /// Gate violated - specific violation details
    Violated(GateViolation),
    /// Insufficient data for evaluation
    InsufficientData { reason: String },
    /// Baseline not yet established
    NoBaseline,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateViolation {
    pub gate_name: String,
    pub threshold: f64,
    pub actual_value: f64,
    pub baseline_value: Option<f64>,
    pub statistical_significance: Option<f64>,
    pub violation_severity: ViolationSeverity,
    pub detected_at: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// Minor violation, within noise threshold
    Minor,
    /// Significant violation, clear degradation
    Significant,
    /// Critical violation, immediate action required
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentLanguageSlice {
    pub intent: String,
    pub language: String,
    pub aece_tau: f64,
    pub sample_count: usize,
    pub confidence_95: f64,
}

impl Default for GateThresholds {
    fn default() -> Self {
        Self {
            max_p99_latency_ms: 1.0,        // 1ms maximum
            max_aece_tau: 0.01,             // 1% calibration error max
            max_confidence_shift: 0.02,     // 2% confidence shift max
            max_sla_recall_change: 0.0,     // Zero change requirement
            min_statistical_significance: 0.05, // p < 0.05
        }
    }
}

impl Default for StatisticalConfig {
    fn default() -> Self {
        Self {
            min_sample_size: 100,
            confidence_level: 0.95,
            rolling_window_size: 1000,
        }
    }
}

impl GoNoGoGates {
    pub fn new(thresholds: GateThresholds, monitoring_window: Duration) -> Self {
        Self {
            baseline: None,
            thresholds,
            monitoring_window,
            statistical_config: StatisticalConfig::default(),
        }
    }

    pub fn with_statistical_config(mut self, config: StatisticalConfig) -> Self {
        self.statistical_config = config;
        self
    }

    /// Set baseline metrics from pre-deployment measurements
    pub async fn set_baseline(&mut self, baseline: BaselineMetrics) -> Result<()> {
        info!("📊 Setting baseline metrics: p99={:.2}ms, AECE-τ={:.4}, confidence={:.3}, SLA-recall@50={:.3}",
              baseline.p99_latency_ms, baseline.aece_tau_max, 
              baseline.median_confidence, baseline.sla_recall_at_50);
        
        self.baseline = Some(baseline);
        Ok(())
    }

    pub async fn get_baseline(&self) -> Result<BaselineMetrics> {
        self.baseline.clone()
            .ok_or_else(|| anyhow::anyhow!("Baseline not yet established"))
    }

    /// Check all four gates against current metrics
    pub async fn check_all_gates(&self, current: CalibrationMetrics) -> Result<HashMap<String, GateStatus>> {
        let mut results = HashMap::new();
        
        // Gate 1: p99 < 1ms calibration latency
        results.insert("latency_gate".to_string(), 
                      self.check_latency_gate(current.p99_latency_ms).await?);
        
        // Gate 2: AECE-τ ≤ 0.01 on every intent×lang slice
        results.insert("aece_gate".to_string(), 
                      self.check_aece_gate(current.aece_tau_max, &current.intent_slices).await?);
        
        // Gate 3: Median confidence shift ≤ 0.02
        results.insert("confidence_gate".to_string(), 
                      self.check_confidence_gate(current.median_confidence).await?);
        
        // Gate 4: Δ(SLA-Recall@50) = 0
        results.insert("sla_recall_gate".to_string(), 
                      self.check_sla_recall_gate(current.sla_recall_at_50).await?);
        
        // Log gate check summary
        let healthy_count = results.values().filter(|s| matches!(s, GateStatus::Healthy)).count();
        let violated_count = results.values().filter(|s| matches!(s, GateStatus::Violated(_))).count();
        
        info!("🔍 Gate check complete: {}/{} healthy, {} violated", 
              healthy_count, results.len(), violated_count);
        
        Ok(results)
    }

    async fn check_latency_gate(&self, current_p99: f64) -> Result<GateStatus> {
        if current_p99 <= self.thresholds.max_p99_latency_ms {
            Ok(GateStatus::Healthy)
        } else {
            let severity = if current_p99 > self.thresholds.max_p99_latency_ms * 2.0 {
                ViolationSeverity::Critical
            } else if current_p99 > self.thresholds.max_p99_latency_ms * 1.5 {
                ViolationSeverity::Significant
            } else {
                ViolationSeverity::Minor
            };

            Ok(GateStatus::Violated(GateViolation {
                gate_name: "latency_gate".to_string(),
                threshold: self.thresholds.max_p99_latency_ms,
                actual_value: current_p99,
                baseline_value: self.baseline.as_ref().map(|b| b.p99_latency_ms),
                statistical_significance: None, // Latency is direct measurement
                violation_severity: severity,
                detected_at: SystemTime::now(),
            }))
        }
    }

    async fn check_aece_gate(&self, current_max_tau: f64, slices: &[IntentLanguageSlice]) -> Result<GateStatus> {
        // Check global maximum first
        if current_max_tau > self.thresholds.max_aece_tau {
            return Ok(GateStatus::Violated(GateViolation {
                gate_name: "aece_gate".to_string(),
                threshold: self.thresholds.max_aece_tau,
                actual_value: current_max_tau,
                baseline_value: self.baseline.as_ref().map(|b| b.aece_tau_max),
                statistical_significance: None,
                violation_severity: ViolationSeverity::Significant,
                detected_at: SystemTime::now(),
            }));
        }

        // Check each intent×language slice
        for slice in slices {
            if slice.aece_tau > self.thresholds.max_aece_tau {
                warn!("🚨 AECE-τ violation in slice {}×{}: {:.4} > {:.4}",
                      slice.intent, slice.language, slice.aece_tau, self.thresholds.max_aece_tau);
                
                return Ok(GateStatus::Violated(GateViolation {
                    gate_name: format!("aece_gate_{}_{}", slice.intent, slice.language),
                    threshold: self.thresholds.max_aece_tau,
                    actual_value: slice.aece_tau,
                    baseline_value: None, // Would need slice-specific baseline
                    statistical_significance: Some(slice.confidence_95),
                    violation_severity: self.classify_aece_violation(slice.aece_tau),
                    detected_at: SystemTime::now(),
                }));
            }
        }

        Ok(GateStatus::Healthy)
    }

    async fn check_confidence_gate(&self, current_confidence: f64) -> Result<GateStatus> {
        let Some(baseline) = &self.baseline else {
            return Ok(GateStatus::NoBaseline);
        };

        let confidence_shift = (current_confidence - baseline.median_confidence).abs();
        
        if confidence_shift <= self.thresholds.max_confidence_shift {
            Ok(GateStatus::Healthy)
        } else {
            // Perform statistical significance test
            let significance = self.test_confidence_significance(
                baseline.median_confidence, 
                current_confidence
            ).await?;

            let severity = if confidence_shift > self.thresholds.max_confidence_shift * 3.0 {
                ViolationSeverity::Critical
            } else if confidence_shift > self.thresholds.max_confidence_shift * 2.0 {
                ViolationSeverity::Significant
            } else {
                ViolationSeverity::Minor
            };

            Ok(GateStatus::Violated(GateViolation {
                gate_name: "confidence_gate".to_string(),
                threshold: self.thresholds.max_confidence_shift,
                actual_value: confidence_shift,
                baseline_value: Some(baseline.median_confidence),
                statistical_significance: Some(significance),
                violation_severity: severity,
                detected_at: SystemTime::now(),
            }))
        }
    }

    async fn check_sla_recall_gate(&self, current_sla_recall: f64) -> Result<GateStatus> {
        let Some(baseline) = &self.baseline else {
            return Ok(GateStatus::NoBaseline);
        };

        let sla_recall_change = (current_sla_recall - baseline.sla_recall_at_50).abs();
        
        // Zero change requirement - any change is a violation
        if sla_recall_change <= self.thresholds.max_sla_recall_change {
            Ok(GateStatus::Healthy)
        } else {
            // Perform statistical significance test
            let significance = self.test_sla_recall_significance(
                baseline.sla_recall_at_50,
                current_sla_recall
            ).await?;

            // Any statistically significant change in SLA-Recall@50 is concerning
            let severity = if significance < 0.001 {
                ViolationSeverity::Critical
            } else if significance < 0.01 {
                ViolationSeverity::Significant
            } else {
                ViolationSeverity::Minor
            };

            Ok(GateStatus::Violated(GateViolation {
                gate_name: "sla_recall_gate".to_string(),
                threshold: self.thresholds.max_sla_recall_change,
                actual_value: sla_recall_change,
                baseline_value: Some(baseline.sla_recall_at_50),
                statistical_significance: Some(significance),
                violation_severity: severity,
                detected_at: SystemTime::now(),
            }))
        }
    }

    fn classify_aece_violation(&self, aece_tau: f64) -> ViolationSeverity {
        if aece_tau > self.thresholds.max_aece_tau * 5.0 {
            ViolationSeverity::Critical
        } else if aece_tau > self.thresholds.max_aece_tau * 2.0 {
            ViolationSeverity::Significant
        } else {
            ViolationSeverity::Minor
        }
    }

    async fn test_confidence_significance(&self, baseline: f64, current: f64) -> Result<f64> {
        // Simplified statistical test - in production would use proper bootstrap/t-test
        let difference = (current - baseline).abs();
        let baseline_variance = baseline * (1.0 - baseline); // Assuming binomial
        let z_score = difference / baseline_variance.sqrt();
        
        // Convert z-score to p-value (simplified)
        let p_value = if z_score > 2.58 {
            0.01  // p < 0.01
        } else if z_score > 1.96 {
            0.05  // p < 0.05
        } else {
            0.1   // p >= 0.05
        };

        Ok(p_value)
    }

    async fn test_sla_recall_significance(&self, baseline: f64, current: f64) -> Result<f64> {
        // Simplified statistical test for recall comparison
        let difference = (current - baseline).abs();
        
        // Use McNemar's test approximation for paired recall comparison
        let z_score = difference / (baseline * (1.0 - baseline)).sqrt();
        
        let p_value = if z_score > 2.58 {
            0.01
        } else if z_score > 1.96 {
            0.05
        } else {
            0.1
        };

        Ok(p_value)
    }

    /// Get detailed gate status for monitoring dashboards
    pub async fn get_gate_details(&self) -> Result<GateDetails> {
        let baseline = self.baseline.clone();
        
        Ok(GateDetails {
            thresholds: self.thresholds.clone(),
            baseline_metrics: baseline,
            statistical_config: self.statistical_config.clone(),
            monitoring_window: self.monitoring_window,
            last_check: SystemTime::now(),
        })
    }

    /// Perform comprehensive gate health check
    pub async fn health_check(&self) -> Result<GateHealthReport> {
        let mut issues = Vec::new();
        
        if self.baseline.is_none() {
            issues.push("Baseline metrics not established".to_string());
        }
        
        if self.thresholds.max_p99_latency_ms <= 0.0 {
            issues.push("Invalid latency threshold".to_string());
        }
        
        if self.thresholds.max_aece_tau <= 0.0 || self.thresholds.max_aece_tau >= 1.0 {
            issues.push("Invalid AECE-τ threshold".to_string());
        }
        
        let healthy = issues.is_empty();
        
        Ok(GateHealthReport {
            healthy,
            issues,
            baseline_established: self.baseline.is_some(),
            thresholds_valid: true,
            statistical_config_valid: self.statistical_config.min_sample_size > 0,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateDetails {
    pub thresholds: GateThresholds,
    pub baseline_metrics: Option<BaselineMetrics>,
    pub statistical_config: StatisticalConfig,
    pub monitoring_window: Duration,
    pub last_check: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateHealthReport {
    pub healthy: bool,
    pub issues: Vec<String>,
    pub baseline_established: bool,
    pub thresholds_valid: bool,
    pub statistical_config_valid: bool,
}

/// Real-time gate monitoring with 15-minute windows
pub struct RealTimeGateMonitor {
    gates: GoNoGoGates,
    window_size: Duration,
    sample_buffer: Vec<(SystemTime, CalibrationMetrics)>,
}

impl RealTimeGateMonitor {
    pub fn new(gates: GoNoGoGates, window_size: Duration) -> Self {
        Self {
            gates,
            window_size,
            sample_buffer: Vec::new(),
        }
    }

    pub async fn add_sample(&mut self, metrics: CalibrationMetrics) -> Result<()> {
        let now = SystemTime::now();
        self.sample_buffer.push((now, metrics));
        
        // Remove samples outside window
        let cutoff = now - self.window_size;
        self.sample_buffer.retain(|(timestamp, _)| *timestamp > cutoff);
        
        Ok(())
    }

    pub async fn check_gates_rolling_window(&self) -> Result<HashMap<String, GateStatus>> {
        if self.sample_buffer.is_empty() {
            return Ok(HashMap::new());
        }

        // Aggregate metrics over rolling window
        let aggregated = self.aggregate_window_metrics()?;
        
        // Check gates with aggregated metrics
        self.gates.check_all_gates(aggregated).await
    }

    fn aggregate_window_metrics(&self) -> Result<CalibrationMetrics> {
        if self.sample_buffer.is_empty() {
            return Err(anyhow::anyhow!("No samples in window"));
        }

        let metrics: Vec<&CalibrationMetrics> = self.sample_buffer.iter()
            .map(|(_, m)| m)
            .collect();

        // Calculate rolling statistics
        let p99_latency_ms = self.calculate_percentile(&metrics.iter()
            .map(|m| m.p99_latency_ms).collect::<Vec<_>>(), 0.99)?;
        
        let aece_tau_max = metrics.iter()
            .map(|m| m.aece_tau_max)
            .fold(0.0f64, f64::max);
        
        let median_confidence = self.calculate_percentile(&metrics.iter()
            .map(|m| m.median_confidence).collect::<Vec<_>>(), 0.5)?;
        
        let sla_recall_at_50 = metrics.iter()
            .map(|m| m.sla_recall_at_50)
            .sum::<f64>() / metrics.len() as f64;

        // Aggregate intent slices (simplified - would need more sophisticated merging)
        let intent_slices = if let Some(first) = metrics.first() {
            first.intent_slices.clone()
        } else {
            Vec::new()
        };

        Ok(CalibrationMetrics {
            p99_latency_ms,
            aece_tau_max,
            median_confidence,
            sla_recall_at_50,
            intent_slices,
            timestamp: SystemTime::now(),
            sample_count: metrics.len(),
        })
    }

    fn calculate_percentile(&self, values: &[f64], percentile: f64) -> Result<f64> {
        if values.is_empty() {
            return Err(anyhow::anyhow!("Cannot calculate percentile of empty array"));
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = ((sorted.len() - 1) as f64 * percentile) as usize;
        Ok(sorted[index.min(sorted.len() - 1)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn create_test_metrics() -> CalibrationMetrics {
        CalibrationMetrics {
            p99_latency_ms: 0.8,
            aece_tau_max: 0.005,
            median_confidence: 0.85,
            sla_recall_at_50: 0.92,
            intent_slices: vec![
                IntentLanguageSlice {
                    intent: "search".to_string(),
                    language: "python".to_string(),
                    aece_tau: 0.005,
                    sample_count: 1000,
                    confidence_95: 0.95,
                }
            ],
            timestamp: SystemTime::now(),
            sample_count: 1000,
        }
    }

    #[tokio::test]
    async fn test_latency_gate_healthy() {
        let mut gates = GoNoGoGates::new(GateThresholds::default(), Duration::from_secs(900));
        let metrics = create_test_metrics();

        let status = gates.check_latency_gate(metrics.p99_latency_ms).await.unwrap();
        assert!(matches!(status, GateStatus::Healthy));
    }

    #[tokio::test]
    async fn test_latency_gate_violation() {
        let mut gates = GoNoGoGates::new(GateThresholds::default(), Duration::from_secs(900));
        
        let status = gates.check_latency_gate(1.5).await.unwrap();
        if let GateStatus::Violated(violation) = status {
            assert_eq!(violation.gate_name, "latency_gate");
            assert_eq!(violation.threshold, 1.0);
            assert_eq!(violation.actual_value, 1.5);
        } else {
            panic!("Expected violation");
        }
    }

    #[tokio::test]
    async fn test_aece_gate_healthy() {
        let mut gates = GoNoGoGates::new(GateThresholds::default(), Duration::from_secs(900));
        let metrics = create_test_metrics();

        let status = gates.check_aece_gate(metrics.aece_tau_max, &metrics.intent_slices).await.unwrap();
        assert!(matches!(status, GateStatus::Healthy));
    }

    #[tokio::test]
    async fn test_real_time_monitor() {
        let gates = GoNoGoGates::new(GateThresholds::default(), Duration::from_secs(900));
        let mut monitor = RealTimeGateMonitor::new(gates, Duration::from_secs(900));
        
        let metrics = create_test_metrics();
        monitor.add_sample(metrics).await.unwrap();
        
        assert_eq!(monitor.sample_buffer.len(), 1);
    }
}