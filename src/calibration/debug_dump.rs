use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::time::{SystemTime, UNIX_EPOCH};
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::calibration::isotonic_calibration::IsotonicCalibratorV2;
use crate::metrics::CalibrationMetrics;
use crate::calibration::go_no_go_gates::GateViolation;

/// One-screen debug dump generator for P1 incident response
/// 
/// Generates comprehensive diagnostics in a single screen format:
/// - Key metrics: N, K_eff, α, clamp_rate, τ, ĉ, per-bin table
/// - Gate violations and baseline comparisons
/// - System health indicators
/// - Actionable diagnostics for rapid incident response
pub struct DebugDumper {
    calibrator: Option<IsotonicCalibratorV2>,
    dump_directory: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugDump {
    pub incident_id: String,
    pub timestamp: SystemTime,
    pub summary: DebugSummary,
    pub calibration_state: CalibrationState,
    pub gate_violations: Vec<GateViolation>,
    pub system_health: SystemHealth,
    pub actionable_diagnostics: Vec<ActionableItem>,
    pub formatted_display: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugSummary {
    pub overall_status: String,
    pub primary_concern: String,
    pub confidence_score: f64,
    pub time_to_resolution_estimate: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationState {
    /// Total number of predictions in current window
    pub n_predictions: u64,
    /// Effective calibration dataset size
    pub k_effective: u64,
    /// Smoothing parameter
    pub alpha: f64,
    /// Percentage of predictions being clamped
    pub clamp_rate: f64,
    /// Current AECE-τ value
    pub tau: f64,
    /// Current mean calibrated confidence
    pub mean_calibrated_confidence: f64,
    /// Per-bin calibration table
    pub calibration_bins: Vec<CalibrationBin>,
    /// Recent calibration history
    pub recent_history: Vec<HistoryPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationBin {
    pub bin_id: u8,
    pub confidence_range: (f64, f64),
    pub count: u64,
    pub mean_confidence: f64,
    pub mean_accuracy: f64,
    pub calibration_delta: f64,
    pub is_problematic: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryPoint {
    pub timestamp: SystemTime,
    pub tau: f64,
    pub confidence: f64,
    pub n_predictions: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub memory_usage_mb: f64,
    pub cpu_utilization: f64,
    pub active_requests: u64,
    pub cache_hit_rate: f64,
    pub error_rate: f64,
    pub upstream_services: Vec<ServiceStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceStatus {
    pub service_name: String,
    pub status: String,
    pub response_time_ms: f64,
    pub error_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionableItem {
    pub priority: String,
    pub action: String,
    pub expected_impact: String,
    pub estimated_effort: String,
    pub owner: String,
}

impl DebugDumper {
    pub fn new(dump_directory: String) -> Self {
        Self {
            calibrator: None,
            dump_directory,
        }
    }

    pub fn with_calibrator(mut self, calibrator: IsotonicCalibratorV2) -> Self {
        self.calibrator = Some(calibrator);
        self
    }

    /// Generate comprehensive debug dump for P1 incident
    pub async fn generate_debug_dump(&self, incident_id: &str) -> Result<String> {
        info!("🚨 Generating debug dump for incident: {}", incident_id);

        let timestamp = SystemTime::now();
        
        // Collect all diagnostic data
        let calibration_state = self.collect_calibration_state().await?;
        let system_health = self.collect_system_health().await?;
        let gate_violations = self.collect_recent_violations().await?;
        
        // Generate summary and diagnostics
        let summary = self.generate_summary(&calibration_state, &gate_violations, &system_health);
        let actionable_diagnostics = self.generate_actionable_diagnostics(
            &calibration_state, &gate_violations, &system_health
        );
        
        // Create formatted one-screen display
        let formatted_display = self.format_one_screen_display(
            incident_id,
            &summary,
            &calibration_state,
            &gate_violations,
            &system_health,
            &actionable_diagnostics,
        );
        
        let debug_dump = DebugDump {
            incident_id: incident_id.to_string(),
            timestamp,
            summary,
            calibration_state,
            gate_violations,
            system_health,
            actionable_diagnostics,
            formatted_display: formatted_display.clone(),
        };
        
        // Save to file
        let dump_path = self.save_debug_dump(&debug_dump).await?;
        
        // Also log the formatted display for immediate visibility
        info!("📊 Debug Dump for {}:\n{}", incident_id, formatted_display);
        
        Ok(dump_path)
    }

    async fn collect_calibration_state(&self) -> Result<CalibrationState> {
        // In production, this would collect from the actual calibrator
        let calibration_bins = self.get_calibration_bins().await?;
        let recent_history = self.get_recent_history().await?;
        
        Ok(CalibrationState {
            n_predictions: 150_000,  // Would be from actual metrics
            k_effective: 10_000,
            alpha: 0.1,
            clamp_rate: 0.15,       // 15% of predictions clamped
            tau: 0.0084,            // Current AECE-τ value
            mean_calibrated_confidence: 0.847,
            calibration_bins,
            recent_history,
        })
    }

    async fn get_calibration_bins(&self) -> Result<Vec<CalibrationBin>> {
        // Generate example calibration bins - in production would be from calibrator
        let mut bins = Vec::new();
        
        for i in 0..10 {
            let range_start = i as f64 / 10.0;
            let range_end = (i + 1) as f64 / 10.0;
            let mean_conf = (range_start + range_end) / 2.0;
            
            // Simulate some miscalibration in higher confidence bins
            let mean_accuracy = if i >= 8 {
                mean_conf - 0.05  // Overconfident in high bins
            } else {
                mean_conf + 0.01  // Slightly underconfident in low bins
            };
            
            let delta = (mean_conf - mean_accuracy).abs();
            
            bins.push(CalibrationBin {
                bin_id: i,
                confidence_range: (range_start, range_end),
                count: 15000 - i as u64 * 1000, // More predictions in lower confidence
                mean_confidence: mean_conf,
                mean_accuracy,
                calibration_delta: delta,
                is_problematic: delta > 0.03,
            });
        }
        
        Ok(bins)
    }

    async fn get_recent_history(&self) -> Result<Vec<HistoryPoint>> {
        let mut history = Vec::new();
        let now = SystemTime::now();
        
        // Generate last 24 hours of history points (every hour)
        for i in 0..24 {
            let timestamp = now - std::time::Duration::from_secs(i * 3600);
            
            // Simulate degradation over time leading to incident
            let tau = 0.005 + (i as f64 * 0.0003); // Gradual increase in calibration error
            let confidence = 0.85 - (i as f64 * 0.001); // Slight confidence drift
            let n_predictions = 6000 + (i as u64 * 50); // Increasing load
            
            history.push(HistoryPoint {
                timestamp,
                tau,
                confidence,
                n_predictions,
            });
        }
        
        history.reverse(); // Oldest first
        Ok(history)
    }

    async fn collect_system_health(&self) -> Result<SystemHealth> {
        // In production, would collect from actual monitoring systems
        Ok(SystemHealth {
            memory_usage_mb: 2048.5,
            cpu_utilization: 0.78,
            active_requests: 1250,
            cache_hit_rate: 0.94,
            error_rate: 0.0023,
            upstream_services: vec![
                ServiceStatus {
                    service_name: "embedding-service".to_string(),
                    status: "healthy".to_string(),
                    response_time_ms: 45.2,
                    error_rate: 0.001,
                },
                ServiceStatus {
                    service_name: "calibration-db".to_string(),
                    status: "degraded".to_string(),
                    response_time_ms: 125.8,
                    error_rate: 0.005,
                },
                ServiceStatus {
                    service_name: "metrics-collector".to_string(),
                    status: "healthy".to_string(),
                    response_time_ms: 23.1,
                    error_rate: 0.0,
                },
            ],
        })
    }

    async fn collect_recent_violations(&self) -> Result<Vec<GateViolation>> {
        // In production, would query actual violation history
        Ok(vec![
            GateViolation {
                gate_name: "aece_gate".to_string(),
                threshold: 0.01,
                actual_value: 0.0084,
                baseline_value: Some(0.0052),
                statistical_significance: Some(0.02),
                violation_severity: crate::calibration::go_no_go_gates::ViolationSeverity::Significant,
                detected_at: SystemTime::now() - std::time::Duration::from_secs(300),
            },
        ])
    }

    fn generate_summary(
        &self,
        calibration: &CalibrationState,
        violations: &[GateViolation],
        health: &SystemHealth,
    ) -> DebugSummary {
        let overall_status = if !violations.is_empty() {
            "DEGRADED"
        } else if calibration.tau > 0.008 {
            "MARGINAL"
        } else {
            "HEALTHY"
        };

        let primary_concern = if !violations.is_empty() {
            format!("Gate violations: {}", 
                violations.iter()
                    .map(|v| v.gate_name.as_str())
                    .collect::<Vec<_>>()
                    .join(", "))
        } else if calibration.clamp_rate > 0.2 {
            "High clamp rate affecting calibration quality".to_string()
        } else if health.upstream_services.iter().any(|s| s.status == "degraded") {
            "Upstream service degradation detected".to_string()
        } else {
            "System operating normally".to_string()
        };

        let confidence_score = if overall_status == "HEALTHY" {
            0.95
        } else if overall_status == "MARGINAL" {
            0.75
        } else {
            0.45
        };

        let time_to_resolution = if overall_status == "DEGRADED" {
            "15-30 minutes"
        } else {
            "5-10 minutes"
        };

        DebugSummary {
            overall_status: overall_status.to_string(),
            primary_concern,
            confidence_score,
            time_to_resolution_estimate: time_to_resolution.to_string(),
        }
    }

    fn generate_actionable_diagnostics(
        &self,
        calibration: &CalibrationState,
        violations: &[GateViolation],
        health: &SystemHealth,
    ) -> Vec<ActionableItem> {
        let mut actions = Vec::new();

        // Gate violation actions
        if !violations.is_empty() {
            actions.push(ActionableItem {
                priority: "P0".to_string(),
                action: "Disable CALIB_V22 feature flag immediately".to_string(),
                expected_impact: "Stop degradation, return to baseline".to_string(),
                estimated_effort: "2 minutes".to_string(),
                owner: "on-call-sre".to_string(),
            });
        }

        // High clamp rate
        if calibration.clamp_rate > 0.2 {
            actions.push(ActionableItem {
                priority: "P1".to_string(),
                action: "Investigate confidence distribution shift".to_string(),
                expected_impact: "Identify root cause of calibration drift".to_string(),
                estimated_effort: "15 minutes".to_string(),
                owner: "ml-engineer".to_string(),
            });
        }

        // High AECE-τ
        if calibration.tau > 0.008 {
            actions.push(ActionableItem {
                priority: "P1".to_string(),
                action: "Review recent model updates and data quality".to_string(),
                expected_impact: "Address calibration quality degradation".to_string(),
                estimated_effort: "30 minutes".to_string(),
                owner: "ml-engineer".to_string(),
            });
        }

        // Upstream service issues
        if health.upstream_services.iter().any(|s| s.status == "degraded") {
            let degraded_services: Vec<&str> = health.upstream_services.iter()
                .filter(|s| s.status == "degraded")
                .map(|s| s.service_name.as_str())
                .collect();
            
            actions.push(ActionableItem {
                priority: "P1".to_string(),
                action: format!("Check upstream services: {}", degraded_services.join(", ")),
                expected_impact: "Resolve upstream dependency issues".to_string(),
                estimated_effort: "10 minutes".to_string(),
                owner: "platform-team".to_string(),
            });
        }

        // Memory pressure
        if health.memory_usage_mb > 3000.0 {
            actions.push(ActionableItem {
                priority: "P2".to_string(),
                action: "Scale up memory or investigate memory leaks".to_string(),
                expected_impact: "Prevent OOM and service degradation".to_string(),
                estimated_effort: "20 minutes".to_string(),
                owner: "platform-team".to_string(),
            });
        }

        actions
    }

    fn format_one_screen_display(
        &self,
        incident_id: &str,
        summary: &DebugSummary,
        calibration: &CalibrationState,
        violations: &[GateViolation],
        health: &SystemHealth,
        actions: &[ActionableItem],
    ) -> String {
        let mut output = String::new();
        
        // Header
        writeln!(&mut output, "╔═══════════════════════════════════════════════════════════════════════════════╗").unwrap();
        writeln!(&mut output, "║                          🚨 CALIB_V2 DEBUG DUMP 🚨                          ║").unwrap();
        writeln!(&mut output, "║ Incident: {:65} ║", format!("{:<65}", incident_id)).unwrap();
        writeln!(&mut output, "║ Status: {:67} ║", format!("{:<67}", summary.overall_status)).unwrap();
        writeln!(&mut output, "╠═══════════════════════════════════════════════════════════════════════════════╣").unwrap();
        
        // Key Metrics Row
        writeln!(&mut output, "║ KEY METRICS                                                                   ║").unwrap();
        writeln!(&mut output, "║ N={:>7} │ K_eff={:>6} │ α={:>4.2} │ Clamp={:>5.1}% │ τ={:>6.4} │ ĉ={:>5.3} ║", 
                 calibration.n_predictions,
                 calibration.k_effective,
                 calibration.alpha,
                 calibration.clamp_rate * 100.0,
                 calibration.tau,
                 calibration.mean_calibrated_confidence).unwrap();
        
        // Gate Violations
        writeln!(&mut output, "╠═══════════════════════════════════════════════════════════════════════════════╣").unwrap();
        writeln!(&mut output, "║ GATE VIOLATIONS                                                               ║").unwrap();
        if violations.is_empty() {
            writeln!(&mut output, "║ ✅ All gates healthy                                                          ║").unwrap();
        } else {
            for violation in violations.iter().take(3) { // Limit to 3 for screen space
                writeln!(&mut output, "║ ❌ {:<20} {:>8.4} > {:>8.4} ({:>6}%)                      ║",
                        violation.gate_name,
                        violation.actual_value,
                        violation.threshold,
                        ((violation.actual_value - violation.threshold) / violation.threshold * 100.0) as i32).unwrap();
            }
        }
        
        // Calibration Bins (Top 5 problematic)
        writeln!(&mut output, "╠═══════════════════════════════════════════════════════════════════════════════╣").unwrap();
        writeln!(&mut output, "║ CALIBRATION BINS (Problematic)                                               ║").unwrap();
        writeln!(&mut output, "║ Bin │  Range   │ Count │ Conf │ Acc  │  Δ   │ Status                       ║").unwrap();
        
        let problematic_bins: Vec<&CalibrationBin> = calibration.calibration_bins.iter()
            .filter(|b| b.is_problematic)
            .take(5)
            .collect();
        
        if problematic_bins.is_empty() {
            writeln!(&mut output, "║ ✅ All calibration bins within tolerance                                     ║").unwrap();
        } else {
            for bin in problematic_bins {
                let status = if bin.calibration_delta > 0.05 { "CRITICAL" } else { "WARNING" };
                writeln!(&mut output, "║ {:>3} │ {:>3.1}-{:<3.1} │ {:>5}k│ {:.2} │ {:.2} │{:>5.3}│ {:>8}              ║",
                        bin.bin_id,
                        bin.confidence_range.0,
                        bin.confidence_range.1,
                        bin.count / 1000,
                        bin.mean_confidence,
                        bin.mean_accuracy,
                        bin.calibration_delta,
                        status).unwrap();
            }
        }
        
        // System Health
        writeln!(&mut output, "╠═══════════════════════════════════════════════════════════════════════════════╣").unwrap();
        writeln!(&mut output, "║ SYSTEM HEALTH                                                                 ║").unwrap();
        writeln!(&mut output, "║ Memory: {:>7.1}MB │ CPU: {:>5.1}% │ Requests: {:>5} │ Cache: {:>5.1}% │ Errors: {:>5.3}% ║",
                 health.memory_usage_mb,
                 health.cpu_utilization * 100.0,
                 health.active_requests,
                 health.cache_hit_rate * 100.0,
                 health.error_rate * 100.0).unwrap();
        
        // Upstream Services
        for service in &health.upstream_services {
            let status_icon = match service.status.as_str() {
                "healthy" => "✅",
                "degraded" => "⚠️",
                "down" => "❌",
                _ => "❓",
            };
            writeln!(&mut output, "║ {} {:>15}: {:>6.1}ms │ {:>5.2}% errors                                    ║",
                    status_icon,
                    service.service_name,
                    service.response_time_ms,
                    service.error_rate * 100.0).unwrap();
        }
        
        // Immediate Actions
        writeln!(&mut output, "╠═══════════════════════════════════════════════════════════════════════════════╣").unwrap();
        writeln!(&mut output, "║ IMMEDIATE ACTIONS                                                             ║").unwrap();
        
        for (i, action) in actions.iter().enumerate().take(3) { // Top 3 priority actions
            let priority_icon = match action.priority.as_str() {
                "P0" => "🚨",
                "P1" => "⚠️",
                "P2" => "📋",
                _ => "➡️",
            };
            writeln!(&mut output, "║ {} [{}] {} ({})                ║",
                    priority_icon,
                    action.priority,
                    format!("{:<50}", action.action),
                    action.estimated_effort).unwrap();
        }
        
        // Footer
        writeln!(&mut output, "╠═══════════════════════════════════════════════════════════════════════════════╣").unwrap();
        writeln!(&mut output, "║ Primary: {:69} ║", format!("{:<69}", summary.primary_concern)).unwrap();
        writeln!(&mut output, "║ Confidence: {:>5.1}% │ ETA Resolution: {:>20}                      ║",
                 summary.confidence_score * 100.0,
                 summary.time_to_resolution_estimate).unwrap();
        writeln!(&mut output, "╚═══════════════════════════════════════════════════════════════════════════════╝").unwrap();
        
        output
    }

    async fn save_debug_dump(&self, dump: &DebugDump) -> Result<String> {
        let timestamp = dump.timestamp.duration_since(UNIX_EPOCH)?;
        let filename = format!("debug-dump-{}-{}.json", 
                             dump.incident_id, 
                             timestamp.as_secs());
        let filepath = format!("{}/{}", self.dump_directory, filename);
        
        // Ensure directory exists
        tokio::fs::create_dir_all(&self.dump_directory).await?;
        
        // Save JSON dump
        let json_content = serde_json::to_string_pretty(dump)?;
        tokio::fs::write(&filepath, json_content).await?;
        
        // Also save formatted display as text file
        let text_filepath = format!("{}/{}", self.dump_directory, 
                                  filename.replace(".json", ".txt"));
        tokio::fs::write(text_filepath, &dump.formatted_display).await?;
        
        info!("💾 Debug dump saved to: {}", filepath);
        Ok(filepath)
    }

    /// Generate quick health summary for monitoring dashboards
    pub async fn generate_health_summary(&self) -> Result<String> {
        let calibration = self.collect_calibration_state().await?;
        let health = self.collect_system_health().await?;
        
        let mut summary = String::new();
        writeln!(&mut summary, "🔍 CALIB_V2 Health Check")?;
        writeln!(&mut summary, "τ: {:.4} | Conf: {:.3} | Clamp: {:.1}%", 
                calibration.tau, 
                calibration.mean_calibrated_confidence,
                calibration.clamp_rate * 100.0)?;
        writeln!(&mut summary, "Memory: {:.0}MB | CPU: {:.1}% | Errors: {:.3}%",
                health.memory_usage_mb,
                health.cpu_utilization * 100.0,
                health.error_rate * 100.0)?;
        
        Ok(summary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_debug_dump_generation() {
        let temp_dir = TempDir::new().unwrap();
        let dumper = DebugDumper::new(temp_dir.path().to_string_lossy().to_string());
        
        let dump_path = dumper.generate_debug_dump("TEST-001").await.unwrap();
        
        // Verify file was created
        assert!(tokio::fs::metadata(&dump_path).await.is_ok());
        
        // Verify content
        let content = tokio::fs::read_to_string(&dump_path).await.unwrap();
        assert!(content.contains("TEST-001"));
    }

    #[tokio::test]
    async fn test_calibration_state_collection() {
        let temp_dir = TempDir::new().unwrap();
        let dumper = DebugDumper::new(temp_dir.path().to_string_lossy().to_string());
        
        let state = dumper.collect_calibration_state().await.unwrap();
        
        assert!(state.n_predictions > 0);
        assert!(state.calibration_bins.len() == 10);
        assert!(!state.recent_history.is_empty());
    }

    #[test]
    fn test_one_screen_formatting() {
        let temp_dir = TempDir::new().unwrap();
        let dumper = DebugDumper::new(temp_dir.path().to_string_lossy().to_string());
        
        let summary = DebugSummary {
            overall_status: "DEGRADED".to_string(),
            primary_concern: "High calibration error".to_string(),
            confidence_score: 0.75,
            time_to_resolution_estimate: "15 minutes".to_string(),
        };
        
        let calibration = CalibrationState {
            n_predictions: 150000,
            k_effective: 10000,
            alpha: 0.1,
            clamp_rate: 0.15,
            tau: 0.0084,
            mean_calibrated_confidence: 0.847,
            calibration_bins: Vec::new(),
            recent_history: Vec::new(),
        };
        
        let output = dumper.format_one_screen_display(
            "TEST-001",
            &summary,
            &calibration,
            &vec![],
            &SystemHealth {
                memory_usage_mb: 2048.5,
                cpu_utilization: 0.78,
                active_requests: 1250,
                cache_hit_rate: 0.94,
                error_rate: 0.0023,
                upstream_services: Vec::new(),
            },
            &vec![],
        );
        
        // Verify it contains expected elements
        assert!(output.contains("TEST-001"));
        assert!(output.contains("DEGRADED"));
        assert!(output.contains("150000"));
        assert!(output.len() > 1000); // Should be substantial output
        
        // Verify it fits in reasonable screen space (under 50 lines)
        assert!(output.lines().count() < 50);
    }
}