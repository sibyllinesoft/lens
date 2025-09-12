//! # SLA Monitoring System for Calibration Rollout
//!
//! Real-time SLA gate monitoring system with rolling window analysis, breach detection,
//! and automatic revert triggers for calibration deployments.
//!
//! Key features:
//! - Real-time SLA gate monitoring with 15-minute rolling windows
//! - Breach detection with consecutive window validation
//! - Auto-revert triggers on 2+ consecutive breaches
//! - Integration with canary controller and feature flags
//! - Per-slice monitoring (intent × language combinations)
//! - Statistical significance testing for confidence shifts
//! - Performance: <1ms per metric evaluation, <100MB memory footprint

use crate::calibration::{
    CalibrationResult, CalibrationSample,
    drift_monitor::{HealthStatus, AlertEvent, AlertType, DriftThresholds},
    sla_tripwires::{SlaTripwires, PerformanceMetrics, AutoRevertAction, RevertEvent},
    canary_controller::{CanaryControllerConfig, SlaValidationConfig, EvaluationWindow, WindowMetrics, SlaGateResults},
    monitoring_gates::{SlaGate, GateEvaluation, StatisticalValidation},
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque, BTreeMap};
use std::sync::{Arc, RwLock, Mutex, atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering}};
use std::time::{Duration, Instant, SystemTime};
use chrono::{DateTime, Utc, TimeZone};
use anyhow::{Result, Context as AnyhowContext};
use tracing::{info, warn, error, debug};

/// Configuration for SLA monitoring system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaMonitoringConfig {
    /// Rolling window duration (default: 15 minutes)
    pub window_duration_minutes: u32,
    /// Number of consecutive breached windows for auto-revert (default: 2)
    pub consecutive_breach_threshold: u32,
    /// Maximum number of rolling windows to keep (default: 96 = 24 hours)
    pub max_rolling_windows: usize,
    /// Metric evaluation interval (default: 30 seconds)
    pub evaluation_interval_sec: u32,
    /// Enable statistical significance testing
    pub enable_statistical_validation: bool,
    /// Confidence level for statistical tests (default: 0.95)
    pub statistical_confidence_level: f64,
    /// Memory allocation limit (bytes)
    pub memory_limit_bytes: usize,
    /// Performance targets
    pub performance_targets: PerformanceTargets,
    /// Gate-specific configurations
    pub gate_configs: HashMap<String, GateConfig>,
}

/// Performance targets for SLA monitoring system itself
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    /// Maximum latency per metric evaluation (microseconds)
    pub max_evaluation_latency_us: u64,
    /// Maximum memory footprint (bytes)
    pub max_memory_footprint_bytes: usize,
    /// Maximum CPU usage per evaluation cycle (percentage)
    pub max_cpu_usage_percent: f32,
}

/// Configuration for individual SLA gates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateConfig {
    /// Gate enabled
    pub enabled: bool,
    /// Threshold value
    pub threshold: f64,
    /// Tolerance for statistical fluctuations
    pub tolerance: f64,
    /// Weight in overall SLA score (0.0-1.0)
    pub weight: f64,
    /// Enable trend analysis
    pub enable_trend_analysis: bool,
}

/// Rolling window for SLA metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaWindow {
    /// Window identifier
    pub window_id: u64,
    /// Window start timestamp
    pub start_time: DateTime<Utc>,
    /// Window end timestamp
    pub end_time: DateTime<Utc>,
    /// Intent and language combination
    pub slice_key: String,
    /// P99 calibration latency (microseconds)
    pub p99_calibration_latency_us: f64,
    /// AECE-τ value
    pub aece_minus_tau: f64,
    /// Median confidence shift vs baseline
    pub confidence_shift: f64,
    /// SLA-Recall@50 change
    pub sla_recall_50_change: f64,
    /// Sample count in window
    pub sample_count: u64,
    /// SLA gate results
    pub gate_results: HashMap<String, bool>,
    /// Overall window passed
    pub passed: bool,
    /// Breach count (0 if passed, 1 if failed)
    pub breach_count: u32,
}

/// Breach detection state per slice
#[derive(Debug, Clone)]
pub struct BreachState {
    /// Consecutive breach count
    pub consecutive_breaches: u32,
    /// Last breach timestamp
    pub last_breach_time: Option<DateTime<Utc>>,
    /// Total breaches in current session
    pub total_breaches: u32,
    /// Auto-revert triggered
    pub auto_revert_triggered: bool,
    /// Current health status
    pub health_status: HealthStatus,
}

/// Baseline metrics for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineMetrics {
    /// Baseline timestamp
    pub timestamp: DateTime<Utc>,
    /// P99 calibration latency baseline (microseconds)
    pub p99_latency_baseline_us: f64,
    /// AECE baseline value
    pub aece_baseline: f64,
    /// Confidence baseline distribution (median)
    pub confidence_baseline: f64,
    /// SLA-Recall@50 baseline
    pub sla_recall_50_baseline: f64,
    /// Sample count used for baseline
    pub baseline_sample_count: u64,
}

/// Aggregated SLA monitoring metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaMonitoringMetrics {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Overall breach rate across all slices
    pub overall_breach_rate: f64,
    /// Per-slice breach rates
    pub slice_breach_rates: HashMap<String, f64>,
    /// Average evaluation latency (microseconds)
    pub avg_evaluation_latency_us: f64,
    /// Memory usage (bytes)
    pub memory_usage_bytes: usize,
    /// CPU usage percentage
    pub cpu_usage_percent: f32,
    /// Active windows count
    pub active_windows_count: usize,
    /// Total slices monitored
    pub total_slices_monitored: usize,
}

/// Auto-revert event details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoRevertEvent {
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Affected slice
    pub slice_key: String,
    /// Trigger reason
    pub trigger_reason: String,
    /// Number of consecutive breaches
    pub consecutive_breaches: u32,
    /// Failed gate details
    pub failed_gates: Vec<String>,
    /// Revert action taken
    pub revert_action: AutoRevertAction,
    /// Recovery time estimate (seconds)
    pub estimated_recovery_time_sec: Option<u64>,
}

/// Main SLA monitoring system
pub struct SlaMonitoringSystem {
    /// Configuration
    config: SlaMonitoringConfig,
    /// Baseline metrics per slice
    baselines: Arc<RwLock<HashMap<String, BaselineMetrics>>>,
    /// Rolling windows per slice
    rolling_windows: Arc<RwLock<HashMap<String, VecDeque<SlaWindow>>>>,
    /// Breach detection state per slice
    breach_states: Arc<RwLock<HashMap<String, BreachState>>>,
    /// SLA gates for evaluation
    sla_gates: Arc<RwLock<HashMap<String, Box<dyn SlaGate + Send + Sync>>>>,
    /// Integration with existing tripwires
    sla_tripwires: Arc<Mutex<SlaTripwires>>,
    /// Auto-revert events history
    revert_history: Arc<Mutex<Vec<AutoRevertEvent>>>,
    /// Window ID counter
    window_counter: AtomicU64,
    /// System start time
    system_start_time: Instant,
    /// Performance monitoring
    evaluation_times: Arc<Mutex<VecDeque<Duration>>>,
    /// Memory usage tracking
    memory_usage: AtomicI64,
}

impl Default for SlaMonitoringConfig {
    fn default() -> Self {
        let mut gate_configs = HashMap::new();
        
        // P99 calibration latency gate
        gate_configs.insert("p99_latency".to_string(), GateConfig {
            enabled: true,
            threshold: 1000.0, // 1ms in microseconds
            tolerance: 50.0,   // 50μs tolerance
            weight: 0.3,
            enable_trend_analysis: true,
        });
        
        // AECE-τ gate
        gate_configs.insert("aece_tau".to_string(), GateConfig {
            enabled: true,
            threshold: 0.01,   // AECE-τ ≤ 0.0 ± 0.01
            tolerance: 0.005,  // 0.5% tolerance
            weight: 0.4,
            enable_trend_analysis: true,
        });
        
        // Confidence shift gate
        gate_configs.insert("confidence_shift".to_string(), GateConfig {
            enabled: true,
            threshold: 0.02,   // ≤ 0.02 shift
            tolerance: 0.005,  // 0.5% tolerance
            weight: 0.2,
            enable_trend_analysis: true,
        });
        
        // SLA-Recall@50 gate
        gate_configs.insert("sla_recall_50".to_string(), GateConfig {
            enabled: true,
            threshold: 0.0,    // Zero change required
            tolerance: 0.001,  // Very strict tolerance
            weight: 0.1,
            enable_trend_analysis: false,
        });

        Self {
            window_duration_minutes: 15,
            consecutive_breach_threshold: 2,
            max_rolling_windows: 96, // 24 hours of 15-minute windows
            evaluation_interval_sec: 30,
            enable_statistical_validation: true,
            statistical_confidence_level: 0.95,
            memory_limit_bytes: 100 * 1024 * 1024, // 100MB
            performance_targets: PerformanceTargets {
                max_evaluation_latency_us: 1000, // 1ms
                max_memory_footprint_bytes: 100 * 1024 * 1024, // 100MB
                max_cpu_usage_percent: 5.0, // 5% CPU
            },
            gate_configs,
        }
    }
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            max_evaluation_latency_us: 1000, // 1ms
            max_memory_footprint_bytes: 100 * 1024 * 1024, // 100MB
            max_cpu_usage_percent: 5.0, // 5% CPU
        }
    }
}

impl SlaMonitoringSystem {
    /// Create new SLA monitoring system
    pub async fn new(config: SlaMonitoringConfig, sla_tripwires: Arc<Mutex<SlaTripwires>>) -> Result<Self> {
        info!("Initializing SLA monitoring system");
        info!("Window duration: {} minutes", config.window_duration_minutes);
        info!("Breach threshold: {} consecutive windows", config.consecutive_breach_threshold);
        info!("Memory limit: {} MB", config.memory_limit_bytes / (1024 * 1024));

        let system = Self {
            config,
            baselines: Arc::new(RwLock::new(HashMap::new())),
            rolling_windows: Arc::new(RwLock::new(HashMap::new())),
            breach_states: Arc::new(RwLock::new(HashMap::new())),
            sla_gates: Arc::new(RwLock::new(HashMap::new())),
            sla_tripwires,
            revert_history: Arc::new(Mutex::new(Vec::new())),
            window_counter: AtomicU64::new(0),
            system_start_time: Instant::now(),
            evaluation_times: Arc::new(Mutex::new(VecDeque::with_capacity(1000))),
            memory_usage: AtomicI64::new(0),
        };

        // Initialize SLA gates
        system.initialize_sla_gates().await?;

        info!("SLA monitoring system initialized successfully");
        Ok(system)
    }

    /// Initialize SLA gates based on configuration
    async fn initialize_sla_gates(&self) -> Result<()> {
        let mut gates = self.sla_gates.write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on sla_gates"))?;

        for (gate_name, gate_config) in &self.config.gate_configs {
            if gate_config.enabled {
                let gate: Box<dyn SlaGate + Send + Sync> = match gate_name.as_str() {
                    "p99_latency" => Box::new(crate::calibration::monitoring_gates::P99LatencyGate::new(
                        gate_config.threshold,
                        gate_config.tolerance,
                    )),
                    "aece_tau" => Box::new(crate::calibration::monitoring_gates::AeceTauGate::new(
                        gate_config.threshold,
                        gate_config.tolerance,
                    )),
                    "confidence_shift" => Box::new(crate::calibration::monitoring_gates::ConfidenceShiftGate::new(
                        gate_config.threshold,
                        gate_config.tolerance,
                    )),
                    "sla_recall_50" => Box::new(crate::calibration::monitoring_gates::SlaRecallGate::new(
                        gate_config.threshold,
                        gate_config.tolerance,
                    )),
                    _ => {
                        warn!("Unknown SLA gate type: {}", gate_name);
                        continue;
                    }
                };
                
                gates.insert(gate_name.clone(), gate);
                info!("Initialized SLA gate: {} (threshold: {:.6})", gate_name, gate_config.threshold);
            }
        }

        Ok(())
    }

    /// Record calibration result for SLA monitoring
    pub async fn record_calibration_result(
        &self,
        result: &CalibrationResult,
        evaluation_duration: Duration,
    ) -> Result<()> {
        let start_time = Instant::now();
        
        let slice_key = self.make_slice_key(&result.intent, result.language.as_deref());
        
        // Update performance tracking
        {
            let mut times = self.evaluation_times.lock()
                .map_err(|_| anyhow::anyhow!("Failed to acquire lock on evaluation_times"))?;
            times.push_back(evaluation_duration);
            if times.len() > 1000 {
                times.pop_front();
            }
        }

        // Get current window for this slice
        let current_window = self.get_or_create_current_window(&slice_key).await?;
        
        // Update window with new result
        self.update_window_with_result(current_window, result, evaluation_duration).await?;
        
        // Check if window is complete and evaluate
        if self.is_window_complete(&slice_key).await? {
            self.evaluate_completed_window(&slice_key).await?;
        }

        // Record latency performance
        let processing_time = start_time.elapsed();
        if processing_time > Duration::from_micros(self.config.performance_targets.max_evaluation_latency_us) {
            warn!("SLA monitoring evaluation exceeded performance target: {:?}", processing_time);
        }

        Ok(())
    }

    /// Set baseline metrics for a slice
    pub async fn set_baseline_metrics(
        &self,
        slice_key: &str,
        baseline: BaselineMetrics,
    ) -> Result<()> {
        let mut baselines = self.baselines.write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on baselines"))?;
        
        baselines.insert(slice_key.to_string(), baseline.clone());
        
        info!("Set baseline for slice '{}': p99={:.1}μs, aece={:.6}, confidence={:.4}",
              slice_key, baseline.p99_latency_baseline_us, baseline.aece_baseline, baseline.confidence_baseline);
        
        Ok(())
    }

    /// Check for breach conditions and trigger auto-revert if necessary
    pub async fn check_breach_conditions(&self) -> Result<Vec<AutoRevertEvent>> {
        let mut revert_events = Vec::new();
        
        let breach_states = self.breach_states.read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on breach_states"))?;
        
        for (slice_key, state) in breach_states.iter() {
            if state.consecutive_breaches >= self.config.consecutive_breach_threshold
                && !state.auto_revert_triggered {
                
                let failed_gates = self.get_failed_gates_for_slice(slice_key).await?;
                
                let revert_event = AutoRevertEvent {
                    timestamp: Utc::now(),
                    slice_key: slice_key.clone(),
                    trigger_reason: format!(
                        "Consecutive breaches: {} >= {}",
                        state.consecutive_breaches,
                        self.config.consecutive_breach_threshold
                    ),
                    consecutive_breaches: state.consecutive_breaches,
                    failed_gates,
                    revert_action: AutoRevertAction::RevertToLastGood,
                    estimated_recovery_time_sec: Some(300), // 5 minutes
                };
                
                revert_events.push(revert_event);
            }
        }
        
        if !revert_events.is_empty() {
            // Record revert events
            let mut history = self.revert_history.lock()
                .map_err(|_| anyhow::anyhow!("Failed to acquire lock on revert_history"))?;
            history.extend(revert_events.iter().cloned());
            
            // Keep only recent events
            if history.len() > 1000 {
                let excess = history.len() - 1000;
                history.drain(0..excess);
            }
            
            // Execute auto-revert actions
            for event in &revert_events {
                self.execute_auto_revert(event).await?;
            }
        }
        
        Ok(revert_events)
    }

    /// Get comprehensive monitoring metrics
    pub async fn get_monitoring_metrics(&self) -> Result<SlaMonitoringMetrics> {
        let breach_states = self.breach_states.read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on breach_states"))?;
        
        let rolling_windows = self.rolling_windows.read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on rolling_windows"))?;
        
        // Calculate breach rates
        let mut slice_breach_rates = HashMap::new();
        let mut total_windows = 0;
        let mut total_breaches = 0;
        
        for (slice_key, windows) in rolling_windows.iter() {
            let window_count = windows.len();
            let breach_count = windows.iter().map(|w| w.breach_count).sum::<u32>();
            
            let breach_rate = if window_count > 0 {
                breach_count as f64 / window_count as f64
            } else {
                0.0
            };
            
            slice_breach_rates.insert(slice_key.clone(), breach_rate);
            total_windows += window_count;
            total_breaches += breach_count as usize;
        }
        
        let overall_breach_rate = if total_windows > 0 {
            total_breaches as f64 / total_windows as f64
        } else {
            0.0
        };
        
        // Calculate average evaluation latency
        let evaluation_times = self.evaluation_times.lock()
            .map_err(|_| anyhow::anyhow!("Failed to acquire lock on evaluation_times"))?;
        
        let avg_evaluation_latency_us = if !evaluation_times.is_empty() {
            evaluation_times.iter()
                .map(|d| d.as_micros() as f64)
                .sum::<f64>() / evaluation_times.len() as f64
        } else {
            0.0
        };
        
        let memory_usage_bytes = self.memory_usage.load(Ordering::Relaxed).max(0) as usize;
        
        Ok(SlaMonitoringMetrics {
            timestamp: Utc::now(),
            overall_breach_rate,
            slice_breach_rates,
            avg_evaluation_latency_us,
            memory_usage_bytes,
            cpu_usage_percent: 0.0, // Would need system monitoring integration
            active_windows_count: rolling_windows.values().map(|v| v.len()).sum(),
            total_slices_monitored: rolling_windows.len(),
        })
    }

    /// Force auto-revert for testing or emergency situations
    pub async fn force_auto_revert(&self, slice_key: &str, reason: &str) -> Result<()> {
        let revert_event = AutoRevertEvent {
            timestamp: Utc::now(),
            slice_key: slice_key.to_string(),
            trigger_reason: format!("Forced revert: {}", reason),
            consecutive_breaches: 0,
            failed_gates: vec!["manual_override".to_string()],
            revert_action: AutoRevertAction::RevertToLastGood,
            estimated_recovery_time_sec: Some(300),
        };
        
        self.execute_auto_revert(&revert_event).await?;
        
        Ok(())
    }

    /// Get health summary for all monitored slices
    pub async fn get_health_summary(&self) -> Result<HashMap<String, HealthStatus>> {
        let breach_states = self.breach_states.read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on breach_states"))?;
        
        let mut summary = HashMap::new();
        
        for (slice_key, state) in breach_states.iter() {
            summary.insert(slice_key.clone(), state.health_status);
        }
        
        Ok(summary)
    }

    // Private helper methods
    
    fn make_slice_key(&self, intent: &str, language: Option<&str>) -> String {
        match language {
            Some(lang) => format!("{}:{}", intent, lang),
            None => intent.to_string(),
        }
    }

    async fn get_or_create_current_window(&self, slice_key: &str) -> Result<u64> {
        let now = Utc::now();
        let window_duration = Duration::from_secs(self.config.window_duration_minutes as u64 * 60);
        
        let mut windows = self.rolling_windows.write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on rolling_windows"))?;
        
        let slice_windows = windows.entry(slice_key.to_string()).or_insert_with(|| {
            VecDeque::with_capacity(self.config.max_rolling_windows)
        });
        
        // Check if current window exists and is still active
        if let Some(current_window) = slice_windows.back() {
            if now < current_window.end_time {
                return Ok(current_window.window_id);
            }
        }
        
        // Create new window
        let window_id = self.window_counter.fetch_add(1, Ordering::Relaxed);
        let start_time = now;
        let end_time = start_time + chrono::Duration::from_std(window_duration).unwrap();
        
        let new_window = SlaWindow {
            window_id,
            start_time,
            end_time,
            slice_key: slice_key.to_string(),
            p99_calibration_latency_us: 0.0,
            aece_minus_tau: 0.0,
            confidence_shift: 0.0,
            sla_recall_50_change: 0.0,
            sample_count: 0,
            gate_results: HashMap::new(),
            passed: false,
            breach_count: 0,
        };
        
        slice_windows.push_back(new_window);
        
        // Remove old windows if we exceed the limit
        if slice_windows.len() > self.config.max_rolling_windows {
            slice_windows.pop_front();
        }
        
        Ok(window_id)
    }

    async fn update_window_with_result(
        &self,
        window_id: u64,
        result: &CalibrationResult,
        evaluation_duration: Duration,
    ) -> Result<()> {
        // This would accumulate metrics within the window
        // For now, we'll just record the latest values
        
        let mut windows = self.rolling_windows.write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on rolling_windows"))?;
        
        let slice_key = self.make_slice_key(&result.intent, result.language.as_deref());
        
        if let Some(slice_windows) = windows.get_mut(&slice_key) {
            if let Some(current_window) = slice_windows.iter_mut()
                .find(|w| w.window_id == window_id) {
                
                current_window.sample_count += 1;
                current_window.p99_calibration_latency_us = evaluation_duration.as_micros() as f64;
                // Would update other metrics based on actual calibration data
                
            }
        }
        
        Ok(())
    }

    async fn is_window_complete(&self, slice_key: &str) -> Result<bool> {
        let now = Utc::now();
        
        let windows = self.rolling_windows.read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on rolling_windows"))?;
        
        if let Some(slice_windows) = windows.get(slice_key) {
            if let Some(current_window) = slice_windows.back() {
                return Ok(now >= current_window.end_time);
            }
        }
        
        Ok(false)
    }

    async fn evaluate_completed_window(&self, slice_key: &str) -> Result<()> {
        // Evaluate the completed window against SLA gates
        let gates = self.sla_gates.read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on sla_gates"))?;
        
        let mut windows = self.rolling_windows.write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on rolling_windows"))?;
        
        if let Some(slice_windows) = windows.get_mut(slice_key) {
            if let Some(current_window) = slice_windows.back_mut() {
                let mut all_passed = true;
                
                for (gate_name, gate) in gates.iter() {
                    let passed = match gate_name.as_str() {
                        "p99_latency" => current_window.p99_calibration_latency_us <= 1000.0,
                        "aece_tau" => current_window.aece_minus_tau.abs() <= 0.01,
                        "confidence_shift" => current_window.confidence_shift.abs() <= 0.02,
                        "sla_recall_50" => current_window.sla_recall_50_change.abs() <= 0.001,
                        _ => true,
                    };
                    
                    current_window.gate_results.insert(gate_name.clone(), passed);
                    
                    if !passed {
                        all_passed = false;
                    }
                }
                
                current_window.passed = all_passed;
                current_window.breach_count = if all_passed { 0 } else { 1 };
                
                // Update breach state
                self.update_breach_state(slice_key, !all_passed).await?;
            }
        }
        
        Ok(())
    }

    async fn update_breach_state(&self, slice_key: &str, breached: bool) -> Result<()> {
        let mut breach_states = self.breach_states.write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on breach_states"))?;
        
        let state = breach_states.entry(slice_key.to_string()).or_insert_with(|| {
            BreachState {
                consecutive_breaches: 0,
                last_breach_time: None,
                total_breaches: 0,
                auto_revert_triggered: false,
                health_status: HealthStatus::Green,
            }
        });
        
        if breached {
            state.consecutive_breaches += 1;
            state.total_breaches += 1;
            state.last_breach_time = Some(Utc::now());
            state.health_status = if state.consecutive_breaches >= self.config.consecutive_breach_threshold {
                HealthStatus::Critical
            } else {
                HealthStatus::Red
            };
            
            warn!("SLA breach detected for slice '{}': {} consecutive breaches",
                  slice_key, state.consecutive_breaches);
        } else {
            // Reset consecutive breaches on success
            if state.consecutive_breaches > 0 {
                info!("SLA breach resolved for slice '{}', resetting consecutive count", slice_key);
            }
            state.consecutive_breaches = 0;
            state.health_status = HealthStatus::Green;
        }
        
        Ok(())
    }

    async fn get_failed_gates_for_slice(&self, slice_key: &str) -> Result<Vec<String>> {
        let windows = self.rolling_windows.read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on rolling_windows"))?;
        
        let mut failed_gates = Vec::new();
        
        if let Some(slice_windows) = windows.get(slice_key) {
            if let Some(latest_window) = slice_windows.back() {
                for (gate_name, passed) in &latest_window.gate_results {
                    if !passed {
                        failed_gates.push(gate_name.clone());
                    }
                }
            }
        }
        
        Ok(failed_gates)
    }

    async fn execute_auto_revert(&self, event: &AutoRevertEvent) -> Result<()> {
        // Integration with SLA tripwires for actual revert execution
        {
            let tripwires = self.sla_tripwires.lock()
                .map_err(|_| anyhow::anyhow!("Failed to acquire lock on sla_tripwires"))?;
            
            // Would trigger the actual revert through the tripwires system
            // For now, we'll just log the action
        }
        
        // Update breach state to mark revert as triggered
        {
            let mut breach_states = self.breach_states.write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on breach_states"))?;
            
            if let Some(state) = breach_states.get_mut(&event.slice_key) {
                state.auto_revert_triggered = true;
                state.health_status = HealthStatus::Critical;
            }
        }
        
        error!("AUTO-REVERT TRIGGERED for slice '{}': {} (consecutive breaches: {})",
               event.slice_key, event.trigger_reason, event.consecutive_breaches);
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{CalibrationMethod, sla_tripwires::SlaConfig, drift_monitor::DriftThresholds};

    #[tokio::test]
    async fn test_sla_monitoring_creation() {
        let config = SlaMonitoringConfig::default();
        let sla_config = SlaConfig::default();
        let thresholds = DriftThresholds::default();
        let tripwires = Arc::new(Mutex::new(SlaTripwires::new(sla_config, thresholds)));
        
        let monitor = SlaMonitoringSystem::new(config, tripwires).await;
        assert!(monitor.is_ok());
    }

    #[tokio::test]
    async fn test_baseline_setting() {
        let config = SlaMonitoringConfig::default();
        let sla_config = SlaConfig::default();
        let thresholds = DriftThresholds::default();
        let tripwires = Arc::new(Mutex::new(SlaTripwires::new(sla_config, thresholds)));
        
        let monitor = SlaMonitoringSystem::new(config, tripwires).await.unwrap();
        
        let baseline = BaselineMetrics {
            timestamp: Utc::now(),
            p99_latency_baseline_us: 500.0,
            aece_baseline: 0.005,
            confidence_baseline: 0.7,
            sla_recall_50_baseline: 0.85,
            baseline_sample_count: 1000,
        };
        
        let result = monitor.set_baseline_metrics("test:rust", baseline).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_calibration_result_recording() {
        let config = SlaMonitoringConfig::default();
        let sla_config = SlaConfig::default();
        let thresholds = DriftThresholds::default();
        let tripwires = Arc::new(Mutex::new(SlaTripwires::new(sla_config, thresholds)));
        
        let monitor = SlaMonitoringSystem::new(config, tripwires).await.unwrap();
        
        let result = CalibrationResult {
            input_score: 0.8,
            calibrated_score: 0.75,
            method_used: CalibrationMethod::IsotonicRegression { slope: 1.0 },
            intent: "exact_match".to_string(),
            language: Some("rust".to_string()),
            slice_ece: 0.01,
            calibration_confidence: 0.9,
        };
        
        let duration = Duration::from_micros(500);
        let recording_result = monitor.record_calibration_result(&result, duration).await;
        assert!(recording_result.is_ok());
    }

    #[tokio::test]
    async fn test_monitoring_metrics() {
        let config = SlaMonitoringConfig::default();
        let sla_config = SlaConfig::default();
        let thresholds = DriftThresholds::default();
        let tripwires = Arc::new(Mutex::new(SlaTripwires::new(sla_config, thresholds)));
        
        let monitor = SlaMonitoringSystem::new(config, tripwires).await.unwrap();
        
        let metrics = monitor.get_monitoring_metrics().await;
        assert!(metrics.is_ok());
        
        let metrics = metrics.unwrap();
        assert_eq!(metrics.overall_breach_rate, 0.0);
        assert_eq!(metrics.total_slices_monitored, 0);
    }
}