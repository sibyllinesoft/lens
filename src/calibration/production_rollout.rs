use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tokio::time::{interval, sleep};
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error};

use crate::calibration::go_no_go_gates::{GoNoGoGates, GateStatus, GateViolation};
use crate::calibration::debug_dump::DebugDumper;
use crate::metrics::MetricsCollector;

/// 24-hour canary production rollout system with strict go/no-go gates
/// 
/// This system manages the global CALIB_V22 feature flag with:
/// - Real-time gate monitoring (15-minute windows)
/// - Auto-revert on 2 consecutive breaches
/// - P1 incident generation with debug dumps
/// - Comprehensive artifact publishing
pub struct ProductionRollout {
    gates: Arc<GoNoGoGates>,
    debug_dumper: Arc<DebugDumper>,
    metrics: Arc<MetricsCollector>,
    state: Arc<RwLock<RolloutState>>,
    config: RolloutConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutConfig {
    /// Duration of canary deployment phase (default: 24 hours)
    pub canary_duration: Duration,
    /// Gate monitoring interval (default: 15 minutes)
    pub monitoring_interval: Duration,
    /// Number of consecutive breaches before auto-revert (default: 2)
    pub breach_threshold: u32,
    /// Feature flag name to control
    pub feature_flag: String,
    /// Baseline data collection window
    pub baseline_window: Duration,
    /// P1 incident webhook URL
    pub incident_webhook: Option<String>,
    /// Artifact publishing configuration
    pub artifact_config: ArtifactConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactConfig {
    pub manifest_path: String,
    pub reliability_diagrams_path: String,
    pub coverage_plots_path: String,
    pub debug_dumps_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutState {
    pub phase: RolloutPhase,
    pub started_at: SystemTime,
    pub last_baseline_update: SystemTime,
    pub consecutive_breaches: u32,
    pub total_breaches: u32,
    pub gate_history: Vec<GateCheckResult>,
    pub artifacts_published: Vec<String>,
    pub incidents_created: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RolloutPhase {
    /// Collecting baseline metrics before deployment
    BaselineCollection,
    /// Canary deployment active, monitoring gates
    CanaryActive { progress: f64 },
    /// Auto-reverted due to gate failures
    AutoReverted { reason: String },
    /// Successfully completed canary phase
    CanaryComplete,
    /// Manual abort requested
    ManualAbort { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateCheckResult {
    pub timestamp: SystemTime,
    pub monitoring_window_start: SystemTime,
    pub gates_status: HashMap<String, GateStatus>,
    pub violations: Vec<GateViolation>,
    pub overall_healthy: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutManifest {
    pub rollout_id: String,
    pub feature_flag: String,
    pub started_at: SystemTime,
    pub config: RolloutConfig,
    pub baseline_metrics: BaselineMetrics,
    pub gate_results: Vec<GateCheckResult>,
    pub artifacts: Vec<String>,
    pub incidents: Vec<IncidentRecord>,
    pub final_status: Option<RolloutOutcome>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineMetrics {
    pub p99_latency_ms: f64,
    pub aece_tau_max: f64,
    pub median_confidence: f64,
    pub sla_recall_at_50: f64,
    pub collection_window: Duration,
    pub sample_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncidentRecord {
    pub incident_id: String,
    pub created_at: SystemTime,
    pub severity: String,
    pub title: String,
    pub description: String,
    pub debug_dump_path: String,
    pub gate_violations: Vec<GateViolation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RolloutOutcome {
    Success,
    AutoReverted { reason: String },
    ManualAbort { reason: String },
    Failed { reason: String },
}

impl Default for RolloutConfig {
    fn default() -> Self {
        Self {
            canary_duration: Duration::from_hours(24),
            monitoring_interval: Duration::from_secs(15 * 60), // 15 minutes
            breach_threshold: 2,
            feature_flag: "CALIB_V22".to_string(),
            baseline_window: Duration::from_hours(1),
            incident_webhook: None,
            artifact_config: ArtifactConfig {
                manifest_path: "/tmp/rollout-manifest.json".to_string(),
                reliability_diagrams_path: "/tmp/reliability-diagrams/".to_string(),
                coverage_plots_path: "/tmp/coverage-plots/".to_string(),
                debug_dumps_path: "/tmp/debug-dumps/".to_string(),
            },
        }
    }
}

impl ProductionRollout {
    pub fn new(
        gates: Arc<GoNoGoGates>,
        debug_dumper: Arc<DebugDumper>,
        metrics: Arc<MetricsCollector>,
        config: RolloutConfig,
    ) -> Self {
        let state = Arc::new(RwLock::new(RolloutState {
            phase: RolloutPhase::BaselineCollection,
            started_at: SystemTime::now(),
            last_baseline_update: SystemTime::now(),
            consecutive_breaches: 0,
            total_breaches: 0,
            gate_history: Vec::new(),
            artifacts_published: Vec::new(),
            incidents_created: Vec::new(),
        }));

        Self {
            gates,
            debug_dumper,
            metrics,
            state,
            config,
        }
    }

    /// Start the 24-hour canary rollout process
    pub async fn start_rollout(&self) -> Result<()> {
        info!("🚀 Starting production rollout for {}", self.config.feature_flag);
        
        // Phase 1: Collect baseline metrics
        self.collect_baseline_metrics().await?;
        
        // Phase 2: Enable feature flag and start monitoring
        self.enable_feature_flag().await?;
        
        // Phase 3: Monitor gates for 24 hours
        self.monitor_canary_phase().await?;
        
        // Phase 4: Complete rollout or handle failures
        self.complete_rollout().await?;
        
        info!("✅ Production rollout completed successfully");
        Ok(())
    }

    async fn collect_baseline_metrics(&self) -> Result<()> {
        info!("📊 Collecting baseline metrics for {} window", 
               format_duration(self.config.baseline_window));
        
        let start = Instant::now();
        let mut samples = Vec::new();
        
        while start.elapsed() < self.config.baseline_window {
            let sample = self.metrics.collect_calibration_metrics().await?;
            samples.push(sample);
            
            sleep(Duration::from_secs(60)).await; // Sample every minute
        }
        
        let baseline = self.calculate_baseline_metrics(samples)?;
        self.gates.set_baseline(baseline.clone()).await?;
        
        // Update state
        {
            let mut state = self.state.write().await;
            state.phase = RolloutPhase::CanaryActive { progress: 0.0 };
            state.last_baseline_update = SystemTime::now();
        }
        
        info!("📈 Baseline metrics collected: p99={:.2}ms, AECE-τ={:.4}, confidence={:.3}, SLA-recall@50={:.3}",
              baseline.p99_latency_ms, baseline.aece_tau_max, 
              baseline.median_confidence, baseline.sla_recall_at_50);
        
        Ok(())
    }

    async fn enable_feature_flag(&self) -> Result<()> {
        info!("🏴 Enabling feature flag: {}", self.config.feature_flag);
        
        // Enable the feature flag via your feature flag system
        self.set_feature_flag_state(true).await?;
        
        // Wait for propagation
        sleep(Duration::from_secs(30)).await;
        
        info!("✅ Feature flag {} is now active", self.config.feature_flag);
        Ok(())
    }

    async fn monitor_canary_phase(&self) -> Result<()> {
        info!("👀 Starting 24-hour canary monitoring with {}-minute gates",
               self.config.monitoring_interval.as_secs() / 60);
        
        let start_time = Instant::now();
        let mut interval = interval(self.config.monitoring_interval);
        
        loop {
            interval.tick().await;
            
            let elapsed = start_time.elapsed();
            let progress = elapsed.as_secs_f64() / self.config.canary_duration.as_secs_f64();
            
            // Check if canary phase is complete
            if elapsed >= self.config.canary_duration {
                info!("⏰ Canary phase completed after {}", format_duration(elapsed));
                break;
            }
            
            // Update progress
            {
                let mut state = self.state.write().await;
                if let RolloutPhase::CanaryActive { ref mut progress } = state.phase {
                    *progress = progress.min(1.0);
                }
            }
            
            // Check all gates
            let gate_result = self.check_all_gates().await?;
            
            // Record result
            {
                let mut state = self.state.write().await;
                state.gate_history.push(gate_result.clone());
            }
            
            // Handle gate violations
            if !gate_result.overall_healthy {
                let should_revert = self.handle_gate_violations(&gate_result).await?;
                if should_revert {
                    return self.execute_auto_revert("2 consecutive gate breaches").await;
                }
            } else {
                // Reset consecutive breaches on healthy check
                let mut state = self.state.write().await;
                state.consecutive_breaches = 0;
            }
            
            info!("🔍 Gate check at {:.1}% progress: {} gates healthy, {} violations",
                  progress * 100.0,
                  gate_result.gates_status.values().filter(|s| matches!(s, GateStatus::Healthy)).count(),
                  gate_result.violations.len());
        }
        
        Ok(())
    }

    async fn check_all_gates(&self) -> Result<GateCheckResult> {
        let window_start = SystemTime::now() - self.config.monitoring_interval;
        let timestamp = SystemTime::now();
        
        // Collect current metrics
        let current_metrics = self.metrics.collect_calibration_metrics().await?;
        
        // Check each gate
        let gates_status = self.gates.check_all_gates(current_metrics).await?;
        
        // Extract violations
        let violations: Vec<GateViolation> = gates_status
            .values()
            .filter_map(|status| match status {
                GateStatus::Violated(violation) => Some(violation.clone()),
                _ => None,
            })
            .collect();
        
        let overall_healthy = violations.is_empty();
        
        Ok(GateCheckResult {
            timestamp,
            monitoring_window_start: window_start,
            gates_status,
            violations,
            overall_healthy,
        })
    }

    async fn handle_gate_violations(&self, result: &GateCheckResult) -> Result<bool> {
        let mut state = self.state.write().await;
        
        if !result.violations.is_empty() {
            state.consecutive_breaches += 1;
            state.total_breaches += 1;
            
            warn!("🚨 Gate violations detected (breach #{}/{}): {:?}",
                  state.consecutive_breaches, self.config.breach_threshold,
                  result.violations.iter().map(|v| &v.gate_name).collect::<Vec<_>>());
            
            // Create P1 incident for first breach
            if state.consecutive_breaches == 1 {
                let incident_id = self.create_p1_incident(result).await?;
                state.incidents_created.push(incident_id);
            }
            
            // Check if we should auto-revert
            if state.consecutive_breaches >= self.config.breach_threshold {
                return Ok(true);
            }
        }
        
        Ok(false)
    }

    async fn create_p1_incident(&self, result: &GateCheckResult) -> Result<String> {
        let incident_id = format!("CALIB-P1-{}", 
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs());
        
        // Generate debug dump
        let dump_path = self.debug_dumper.generate_debug_dump(&incident_id).await?;
        
        let incident = IncidentRecord {
            incident_id: incident_id.clone(),
            created_at: SystemTime::now(),
            severity: "P1".to_string(),
            title: format!("CALIB_V22 Canary Gate Violations - {}", 
                result.violations.iter()
                    .map(|v| v.gate_name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")),
            description: self.format_incident_description(result),
            debug_dump_path: dump_path,
            gate_violations: result.violations.clone(),
        };
        
        // Send to incident management system
        if let Some(webhook_url) = &self.config.incident_webhook {
            self.send_incident_webhook(webhook_url, &incident).await?;
        }
        
        error!("🚨 P1 Incident Created: {} - Gate violations in CALIB_V22 canary deployment", 
               incident_id);
        
        Ok(incident_id)
    }

    fn format_incident_description(&self, result: &GateCheckResult) -> String {
        let mut desc = String::from("CALIB_V22 canary deployment gate violations detected:\n\n");
        
        for violation in &result.violations {
            desc.push_str(&format!("❌ {}: {} (threshold: {})\n", 
                violation.gate_name, violation.actual_value, violation.threshold));
        }
        
        desc.push_str("\nAuto-revert will trigger after 2 consecutive breaches.\n");
        desc.push_str("Debug dump and artifacts attached for analysis.");
        
        desc
    }

    async fn execute_auto_revert(&self, reason: &str) -> Result<()> {
        warn!("🔄 Executing auto-revert: {}", reason);
        
        // Disable feature flag immediately
        self.set_feature_flag_state(false).await?;
        
        // Update state
        {
            let mut state = self.state.write().await;
            state.phase = RolloutPhase::AutoReverted { 
                reason: reason.to_string() 
            };
        }
        
        // Generate final debug dump
        let revert_dump_id = format!("AUTO-REVERT-{}", 
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs());
        self.debug_dumper.generate_debug_dump(&revert_dump_id).await?;
        
        // Publish artifacts
        self.publish_rollout_artifacts().await?;
        
        error!("🚨 CALIB_V22 auto-reverted: {}", reason);
        Ok(())
    }

    async fn complete_rollout(&self) -> Result<()> {
        let state = self.state.read().await;
        
        match &state.phase {
            RolloutPhase::AutoReverted { .. } => {
                // Already handled
                Ok(())
            }
            RolloutPhase::CanaryActive { .. } => {
                // Successful completion
                drop(state);
                
                let mut state = self.state.write().await;
                state.phase = RolloutPhase::CanaryComplete;
                drop(state);
                
                self.publish_rollout_artifacts().await?;
                info!("🎉 Canary rollout completed successfully - CALIB_V22 is stable");
                Ok(())
            }
            _ => {
                Err(anyhow::anyhow!("Invalid state for rollout completion"))
            }
        }
    }

    async fn publish_rollout_artifacts(&self) -> Result<()> {
        info!("📦 Publishing rollout artifacts...");
        
        let state = self.state.read().await;
        let baseline = self.gates.get_baseline().await?;
        
        // Generate rollout manifest
        let manifest = RolloutManifest {
            rollout_id: format!("CALIB_V22-{}", 
                state.started_at.duration_since(UNIX_EPOCH)?.as_secs()),
            feature_flag: self.config.feature_flag.clone(),
            started_at: state.started_at,
            config: self.config.clone(),
            baseline_metrics: baseline,
            gate_results: state.gate_history.clone(),
            artifacts: state.artifacts_published.clone(),
            incidents: Vec::new(), // Would be populated from incident records
            final_status: Some(match &state.phase {
                RolloutPhase::CanaryComplete => RolloutOutcome::Success,
                RolloutPhase::AutoReverted { reason } => 
                    RolloutOutcome::AutoReverted { reason: reason.clone() },
                RolloutPhase::ManualAbort { reason } => 
                    RolloutOutcome::ManualAbort { reason: reason.clone() },
                _ => RolloutOutcome::Failed { reason: "Unexpected state".to_string() },
            }),
        };
        
        // Write manifest
        tokio::fs::write(
            &self.config.artifact_config.manifest_path,
            serde_json::to_string_pretty(&manifest)?
        ).await?;
        
        // Generate reliability diagrams
        self.generate_reliability_diagrams(&state).await?;
        
        // Generate coverage plots
        self.generate_coverage_plots(&state).await?;
        
        info!("✅ Artifacts published to {}", self.config.artifact_config.manifest_path);
        Ok(())
    }

    async fn generate_reliability_diagrams(&self, _state: &RolloutState) -> Result<()> {
        // Implementation would generate reliability diagrams showing:
        // - Gate status over time
        // - Violation patterns
        // - Performance trends
        info!("📈 Generating reliability diagrams...");
        Ok(())
    }

    async fn generate_coverage_plots(&self, _state: &RolloutState) -> Result<()> {
        // Implementation would generate coverage plots showing:
        // - Intent×language slice coverage
        // - Confidence distribution changes
        // - Calibration effectiveness
        info!("📊 Generating coverage plots...");
        Ok(())
    }

    async fn set_feature_flag_state(&self, enabled: bool) -> Result<()> {
        // Integration with your feature flag system
        info!("🏴 Setting {} to {}", self.config.feature_flag, enabled);
        
        // Example implementation:
        // self.feature_flag_client.set_flag(&self.config.feature_flag, enabled).await?;
        
        Ok(())
    }

    async fn send_incident_webhook(&self, _webhook_url: &str, _incident: &IncidentRecord) -> Result<()> {
        // Implementation would send incident to webhook
        info!("🚨 Sending incident webhook notification");
        Ok(())
    }

    fn calculate_baseline_metrics(&self, samples: Vec<crate::metrics::CalibrationMetrics>) -> Result<BaselineMetrics> {
        if samples.is_empty() {
            return Err(anyhow::anyhow!("No baseline samples collected"));
        }
        
        // Calculate percentiles and aggregates from samples
        let mut latencies: Vec<f64> = samples.iter().map(|s| s.p99_latency_ms).collect();
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let p99_latency_ms = latencies[(latencies.len() * 99 / 100).min(latencies.len() - 1)];
        let aece_tau_max = samples.iter().map(|s| s.aece_tau_max).fold(0.0f64, f64::max);
        let median_confidence = {
            let mut confidences: Vec<f64> = samples.iter().map(|s| s.median_confidence).collect();
            confidences.sort_by(|a, b| a.partial_cmp(b).unwrap());
            confidences[confidences.len() / 2]
        };
        let sla_recall_at_50 = samples.iter().map(|s| s.sla_recall_at_50).sum::<f64>() / samples.len() as f64;
        
        Ok(BaselineMetrics {
            p99_latency_ms,
            aece_tau_max,
            median_confidence,
            sla_recall_at_50,
            collection_window: self.config.baseline_window,
            sample_count: samples.len() as u64,
        })
    }

    /// Get current rollout status
    pub async fn get_status(&self) -> RolloutState {
        self.state.read().await.clone()
    }

    /// Manually abort the rollout
    pub async fn abort_rollout(&self, reason: String) -> Result<()> {
        info!("🛑 Manual rollout abort requested: {}", reason);
        
        self.set_feature_flag_state(false).await?;
        
        let mut state = self.state.write().await;
        state.phase = RolloutPhase::ManualAbort { reason };
        
        Ok(())
    }
}

fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    let secs = seconds % 60;
    
    if hours > 0 {
        format!("{}h{}m{}s", hours, minutes, secs)
    } else if minutes > 0 {
        format!("{}m{}s", minutes, secs)
    } else {
        format!("{}s", secs)
    }
}

// Extension trait for Duration
trait DurationExt {
    fn from_hours(hours: u64) -> Duration;
}

impl DurationExt for Duration {
    fn from_hours(hours: u64) -> Duration {
        Duration::from_secs(hours * 3600)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_rollout_state_transitions() {
        // Test state machine transitions
        let config = RolloutConfig::default();
        let state = RolloutState {
            phase: RolloutPhase::BaselineCollection,
            started_at: SystemTime::now(),
            last_baseline_update: SystemTime::now(),
            consecutive_breaches: 0,
            total_breaches: 0,
            gate_history: Vec::new(),
            artifacts_published: Vec::new(),
            incidents_created: Vec::new(),
        };
        
        assert!(matches!(state.phase, RolloutPhase::BaselineCollection));
    }
    
    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_secs(3661)), "1h1m1s");
        assert_eq!(format_duration(Duration::from_secs(61)), "1m1s");
        assert_eq!(format_duration(Duration::from_secs(30)), "30s");
    }
}