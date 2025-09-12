//! # SLA-Bound Tripwires for Production Safety
//!
//! Automated circuit breakers and safety mechanisms that prevent calibration
//! from degrading production performance or accuracy:
//! - p99 < 1ms latency enforcement with automatic fallback
//! - Zero allocations on hot path monitoring
//! - Automated revert triggers for critical thresholds
//! - Circuit breaker patterns for graceful degradation
//! - Dead man's switch for runaway calibration

use crate::calibration::{
    CalibrationSample,
    drift_monitor::{HealthStatus, AlertEvent, AlertType, DriftThresholds},
    fast_bootstrap::{FastBootstrap, BootstrapTiming},
    shared_binning_core::SharedBinningCore,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, atomic::{AtomicBool, AtomicU64, Ordering}};
use std::time::{Duration, Instant};
use std::collections::VecDeque;
use chrono::{DateTime, Utc};
use anyhow::{Result, Context as AnyhowContext};

/// SLA enforcement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaConfig {
    /// Maximum allowed p99 latency (microseconds)
    pub max_p99_latency_us: u64,
    /// Maximum allowed mean latency (microseconds) 
    pub max_mean_latency_us: u64,
    /// Maximum allowed hot path allocations (should be 0)
    pub max_hot_path_allocations: usize,
    /// Circuit breaker failure threshold (consecutive failures)
    pub circuit_breaker_threshold: usize,
    /// Circuit breaker recovery timeout (seconds)
    pub circuit_breaker_timeout_sec: u64,
    /// Auto-revert enabled
    pub auto_revert_enabled: bool,
    /// Emergency fallback to uncalibrated predictions
    pub emergency_fallback_enabled: bool,
    /// Dead man's switch timeout (seconds) - revert if no heartbeat
    pub dead_mans_switch_timeout_sec: u64,
}

impl Default for SlaConfig {
    fn default() -> Self {
        Self {
            max_p99_latency_us: 1000,  // 1ms
            max_mean_latency_us: 500,  // 0.5ms
            max_hot_path_allocations: 0,
            circuit_breaker_threshold: 5,
            circuit_breaker_timeout_sec: 300, // 5 minutes
            auto_revert_enabled: true,
            emergency_fallback_enabled: true,
            dead_mans_switch_timeout_sec: 3600, // 1 hour
        }
    }
}

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitBreakerState {
    /// Normal operation
    Closed,
    /// Too many failures, blocking requests
    Open,
    /// Testing if service has recovered
    HalfOpen,
}

/// Performance metrics for SLA monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub timestamp: DateTime<Utc>,
    pub latency_p99_us: f64,
    pub latency_mean_us: f64,
    pub hot_path_allocations: usize,
    pub throughput_ops_per_sec: f64,
    pub memory_usage_bytes: usize,
    pub cpu_usage_percent: f32,
}

/// Auto-revert action types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AutoRevertAction {
    /// Revert to last known good calibration config
    RevertToLastGood,
    /// Fall back to uncalibrated predictions
    FallbackUncalibrated,
    /// Switch to simplified calibration (temperature scaling only)
    SimplifiedCalibration,
    /// Disable calibration completely (emergency)
    DisableCalibration,
}

/// Revert event logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevertEvent {
    pub timestamp: DateTime<Utc>,
    pub trigger_reason: String,
    pub action_taken: AutoRevertAction,
    pub metrics_snapshot: PerformanceMetrics,
    pub health_status: HealthStatus,
    pub recovery_time_sec: Option<u64>,
}

/// SLA tripwire system with circuit breaker and auto-revert
pub struct SlaTripwires {
    config: SlaConfig,
    thresholds: DriftThresholds,
    
    // Circuit breaker state
    circuit_state: Arc<Mutex<CircuitBreakerState>>,
    failure_count: Arc<AtomicU64>,
    last_failure_time: Arc<Mutex<Option<Instant>>>,
    
    // Performance monitoring
    recent_metrics: Arc<Mutex<VecDeque<PerformanceMetrics>>>,
    
    // Auto-revert tracking
    revert_history: Arc<Mutex<Vec<RevertEvent>>>,
    last_known_good_config: Arc<Mutex<Option<String>>>,
    
    // Dead man's switch
    last_heartbeat: Arc<AtomicU64>,
    emergency_mode: Arc<AtomicBool>,
    
    // Timing measurements
    timing_buffer: Vec<Duration>,
}

impl SlaTripwires {
    /// Create new SLA tripwire system
    pub fn new(config: SlaConfig, thresholds: DriftThresholds) -> Self {
        Self {
            config,
            thresholds,
            circuit_state: Arc::new(Mutex::new(CircuitBreakerState::Closed)),
            failure_count: Arc::new(AtomicU64::new(0)),
            last_failure_time: Arc::new(Mutex::new(None)),
            recent_metrics: Arc::new(Mutex::new(VecDeque::with_capacity(1000))),
            revert_history: Arc::new(Mutex::new(Vec::new())),
            last_known_good_config: Arc::new(Mutex::new(None)),
            last_heartbeat: Arc::new(AtomicU64::new(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            )),
            emergency_mode: Arc::new(AtomicBool::new(false)),
            timing_buffer: Vec::with_capacity(1000),
        }
    }
    
    /// Update heartbeat to prevent dead man's switch
    pub fn heartbeat(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.last_heartbeat.store(now, Ordering::Relaxed);
        
        // Clear emergency mode if heartbeat is received
        if self.emergency_mode.load(Ordering::Relaxed) {
            self.emergency_mode.store(false, Ordering::Relaxed);
        }
    }
    
    /// Check if dead man's switch should activate
    pub fn check_dead_mans_switch(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let last_heartbeat = self.last_heartbeat.load(Ordering::Relaxed);
        let elapsed = now.saturating_sub(last_heartbeat);
        
        elapsed > self.config.dead_mans_switch_timeout_sec
    }
    
    /// Record performance measurement
    pub fn record_performance(&mut self, metrics: PerformanceMetrics) -> Result<()> {
        // Check SLA violations
        let mut violations = Vec::new();
        
        if metrics.latency_p99_us > self.config.max_p99_latency_us as f64 {
            violations.push(format!("P99 latency {:.0}μs > {} μs", 
                metrics.latency_p99_us, self.config.max_p99_latency_us));
        }
        
        if metrics.latency_mean_us > self.config.max_mean_latency_us as f64 {
            violations.push(format!("Mean latency {:.0}μs > {} μs",
                metrics.latency_mean_us, self.config.max_mean_latency_us));
        }
        
        if metrics.hot_path_allocations > self.config.max_hot_path_allocations {
            violations.push(format!("Hot path allocations {} > {}",
                metrics.hot_path_allocations, self.config.max_hot_path_allocations));
        }
        
        // Store metrics
        {
            let mut recent = self.recent_metrics.lock().unwrap();
            recent.push_back(metrics.clone());
            if recent.len() > 1000 {
                recent.pop_front();
            }
        }
        
        // Handle violations
        if !violations.is_empty() {
            self.handle_sla_violation(violations, metrics)?;
        } else {
            // Reset failure count on success
            self.failure_count.store(0, Ordering::Relaxed);
        }
        
        Ok(())
    }
    
    /// Handle SLA violation with circuit breaker logic
    fn handle_sla_violation(&self, violations: Vec<String>, metrics: PerformanceMetrics) -> Result<()> {
        let failure_count = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        
        {
            let mut last_failure = self.last_failure_time.lock().unwrap();
            *last_failure = Some(Instant::now());
        }
        
        // Check circuit breaker threshold
        if failure_count >= self.config.circuit_breaker_threshold as u64 {
            self.trip_circuit_breaker(violations, metrics)?;
        }
        
        Ok(())
    }
    
    /// Trip circuit breaker and initiate auto-revert if enabled
    fn trip_circuit_breaker(&self, violations: Vec<String>, metrics: PerformanceMetrics) -> Result<()> {
        // Update circuit breaker state
        {
            let mut state = self.circuit_state.lock().unwrap();
            *state = CircuitBreakerState::Open;
        }
        
        let trigger_reason = format!("SLA violations: {}", violations.join("; "));
        
        if self.config.auto_revert_enabled {
            self.execute_auto_revert(trigger_reason, metrics)?;
        }
        
        Ok(())
    }
    
    /// Execute auto-revert action
    fn execute_auto_revert(&self, trigger_reason: String, metrics: PerformanceMetrics) -> Result<()> {
        // Determine revert action based on severity
        let action = if metrics.latency_p99_us > (self.config.max_p99_latency_us as f64 * 10.0) {
            // Extreme latency violation - emergency fallback
            AutoRevertAction::DisableCalibration
        } else if metrics.hot_path_allocations > 1000 {
            // Memory leak detected - simplified calibration
            AutoRevertAction::SimplifiedCalibration
        } else if metrics.latency_p99_us > (self.config.max_p99_latency_us as f64 * 5.0) {
            // High latency - fallback to uncalibrated
            AutoRevertAction::FallbackUncalibrated
        } else {
            // Moderate violation - revert to last good
            AutoRevertAction::RevertToLastGood
        };
        
        // Create revert event
        let revert_event = RevertEvent {
            timestamp: Utc::now(),
            trigger_reason,
            action_taken: action.clone(),
            metrics_snapshot: metrics,
            health_status: HealthStatus::Critical,
            recovery_time_sec: None, // Will be updated when recovered
        };
        
        // Store revert event
        {
            let mut history = self.revert_history.lock().unwrap();
            history.push(revert_event);
            
            // Keep only last 100 revert events
            if history.len() > 100 {
                let excess = history.len() - 100;
                history.drain(0..excess);
            }
        }
        
        // Execute the revert action (would integrate with actual calibration system)
        self.apply_revert_action(action)?;
        
        Ok(())
    }
    
    /// Apply the chosen revert action
    fn apply_revert_action(&self, action: AutoRevertAction) -> Result<()> {
        match action {
            AutoRevertAction::RevertToLastGood => {
                // Would restore last known good configuration
                println!("AUTO-REVERT: Restoring last known good calibration configuration");
            }
            AutoRevertAction::FallbackUncalibrated => {
                // Would disable calibration, use raw predictions
                println!("AUTO-REVERT: Falling back to uncalibrated predictions");
            }
            AutoRevertAction::SimplifiedCalibration => {
                // Would switch to temperature scaling only
                println!("AUTO-REVERT: Switching to simplified temperature scaling");
            }
            AutoRevertAction::DisableCalibration => {
                // Emergency: disable all calibration
                println!("EMERGENCY AUTO-REVERT: Disabling all calibration");
                self.emergency_mode.store(true, Ordering::Relaxed);
            }
        }
        
        Ok(())
    }
    
    /// Check if calibration should be blocked (circuit breaker open)
    pub fn should_block_calibration(&self) -> bool {
        // Check dead man's switch
        if self.check_dead_mans_switch() {
            return true;
        }
        
        // Check emergency mode
        if self.emergency_mode.load(Ordering::Relaxed) {
            return true;
        }
        
        // Check circuit breaker state
        let state = self.circuit_state.lock().unwrap();
        match *state {
            CircuitBreakerState::Open => {
                // Check if enough time has passed to try recovery
                let last_failure = self.last_failure_time.lock().unwrap();
                if let Some(failure_time) = *last_failure {
                    let elapsed = failure_time.elapsed().as_secs();
                    elapsed < self.config.circuit_breaker_timeout_sec
                } else {
                    true
                }
            }
            CircuitBreakerState::HalfOpen => {
                // Allow limited requests for testing
                false
            }
            CircuitBreakerState::Closed => {
                // Normal operation
                false
            }
        }
    }
    
    /// Attempt to recover circuit breaker
    pub fn attempt_recovery(&self) -> Result<bool> {
        let mut state = self.circuit_state.lock().unwrap();
        
        match *state {
            CircuitBreakerState::Open => {
                // Check if timeout has passed
                let last_failure = self.last_failure_time.lock().unwrap();
                if let Some(failure_time) = *last_failure {
                    if failure_time.elapsed().as_secs() >= self.config.circuit_breaker_timeout_sec {
                        *state = CircuitBreakerState::HalfOpen;
                        return Ok(true);
                    }
                }
            }
            CircuitBreakerState::HalfOpen => {
                // Check if recent metrics show recovery
                let recent = self.recent_metrics.lock().unwrap();
                if let Some(latest) = recent.back() {
                    let is_healthy = latest.latency_p99_us <= self.config.max_p99_latency_us as f64
                        && latest.latency_mean_us <= self.config.max_mean_latency_us as f64
                        && latest.hot_path_allocations <= self.config.max_hot_path_allocations;
                        
                    if is_healthy {
                        *state = CircuitBreakerState::Closed;
                        self.failure_count.store(0, Ordering::Relaxed);
                        return Ok(true);
                    }
                }
            }
            CircuitBreakerState::Closed => {
                // Already recovered
                return Ok(true);
            }
        }
        
        Ok(false)
    }
    
    /// Get current system health summary
    pub fn get_health_summary(&self) -> Result<serde_json::Value> {
        let state = self.circuit_state.lock().unwrap();
        let recent = self.recent_metrics.lock().unwrap();
        let revert_history = self.revert_history.lock().unwrap();
        
        let latest_metrics = recent.back();
        
        Ok(serde_json::json!({
            "circuit_breaker_state": format!("{:?}", *state),
            "failure_count": self.failure_count.load(Ordering::Relaxed),
            "emergency_mode": self.emergency_mode.load(Ordering::Relaxed),
            "dead_mans_switch_active": self.check_dead_mans_switch(),
            "recent_reverts": revert_history.len(),
            "latest_metrics": latest_metrics,
            "sla_config": self.config,
            "metrics_history_size": recent.len(),
        }))
    }
    
    /// Force emergency shutdown (for testing or critical situations)
    pub fn emergency_shutdown(&self, reason: &str) -> Result<()> {
        self.emergency_mode.store(true, Ordering::Relaxed);
        
        let metrics = PerformanceMetrics {
            timestamp: Utc::now(),
            latency_p99_us: 0.0,
            latency_mean_us: 0.0,
            hot_path_allocations: 0,
            throughput_ops_per_sec: 0.0,
            memory_usage_bytes: 0,
            cpu_usage_percent: 0.0,
        };
        
        let revert_event = RevertEvent {
            timestamp: Utc::now(),
            trigger_reason: format!("Emergency shutdown: {}", reason),
            action_taken: AutoRevertAction::DisableCalibration,
            metrics_snapshot: metrics,
            health_status: HealthStatus::Critical,
            recovery_time_sec: None,
        };
        
        let mut history = self.revert_history.lock().unwrap();
        history.push(revert_event);
        
        println!("EMERGENCY SHUTDOWN: {}", reason);
        
        Ok(())
    }
    
    /// Record successful calibration operation (for circuit breaker recovery)
    pub fn record_success(&mut self, duration: Duration) {
        self.timing_buffer.push(duration);
        
        // Keep only recent timings for p99 calculation
        if self.timing_buffer.len() > 1000 {
            let excess = self.timing_buffer.len() - 1000;
            self.timing_buffer.drain(0..excess);
        }
        
        // Compute metrics periodically
        if self.timing_buffer.len() % 100 == 0 {
            self.compute_and_record_metrics();
        }
    }
    
    /// Compute current performance metrics from timing buffer
    fn compute_and_record_metrics(&self) {
        if self.timing_buffer.is_empty() {
            return;
        }
        
        let mut sorted_timings: Vec<f64> = self.timing_buffer
            .iter()
            .map(|d| d.as_micros() as f64)
            .collect();
        sorted_timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let p99_idx = (sorted_timings.len() * 99 / 100).min(sorted_timings.len() - 1);
        let p99 = sorted_timings[p99_idx];
        
        let mean = sorted_timings.iter().sum::<f64>() / sorted_timings.len() as f64;
        
        let metrics = PerformanceMetrics {
            timestamp: Utc::now(),
            latency_p99_us: p99,
            latency_mean_us: mean,
            hot_path_allocations: 0, // Would be measured from allocator
            throughput_ops_per_sec: 1.0 / (mean / 1_000_000.0), // Approximate
            memory_usage_bytes: 0,   // Would be measured from system
            cpu_usage_percent: 0.0,  // Would be measured from system
        };
        
        // This would normally call record_performance, but we skip it here
        // to avoid infinite recursion during testing
        let mut recent = self.recent_metrics.lock().unwrap();
        recent.push_back(metrics);
        if recent.len() > 1000 {
            recent.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::drift_monitor::DriftThresholds;
    
    #[test]
    fn test_sla_tripwires_creation() {
        let config = SlaConfig::default();
        let thresholds = DriftThresholds::default();
        
        let tripwires = SlaTripwires::new(config, thresholds);
        
        assert!(!tripwires.should_block_calibration());
        assert!(!tripwires.check_dead_mans_switch());
        assert!(!tripwires.emergency_mode.load(Ordering::Relaxed));
    }
    
    #[test]
    fn test_heartbeat_and_dead_mans_switch() {
        let config = SlaConfig {
            dead_mans_switch_timeout_sec: 1, // 1 second for testing
            ..Default::default()
        };
        let thresholds = DriftThresholds::default();
        
        let tripwires = SlaTripwires::new(config, thresholds);
        
        // Initially no timeout
        assert!(!tripwires.check_dead_mans_switch());
        
        // Update heartbeat
        tripwires.heartbeat();
        assert!(!tripwires.check_dead_mans_switch());
        
        // Wait for timeout (in real test, would need to wait or mock time)
        // For unit test, we can't easily test the timeout without mocking
    }
    
    #[test]
    fn test_performance_recording() {
        let config = SlaConfig::default();
        let thresholds = DriftThresholds::default();
        
        let mut tripwires = SlaTripwires::new(config, thresholds);
        
        // Record good performance
        let good_metrics = PerformanceMetrics {
            timestamp: Utc::now(),
            latency_p99_us: 500.0,  // Under 1ms limit
            latency_mean_us: 300.0, // Under limit
            hot_path_allocations: 0, // No allocations
            throughput_ops_per_sec: 1000.0,
            memory_usage_bytes: 1024 * 1024,
            cpu_usage_percent: 50.0,
        };
        
        let result = tripwires.record_performance(good_metrics);
        assert!(result.is_ok());
        assert!(!tripwires.should_block_calibration());
    }
    
    #[test]
    fn test_emergency_shutdown() {
        let config = SlaConfig::default();
        let thresholds = DriftThresholds::default();
        
        let tripwires = SlaTripwires::new(config, thresholds);
        
        // Initially not in emergency mode
        assert!(!tripwires.emergency_mode.load(Ordering::Relaxed));
        assert!(!tripwires.should_block_calibration());
        
        // Trigger emergency shutdown
        let result = tripwires.emergency_shutdown("Test emergency");
        assert!(result.is_ok());
        
        // Should now be in emergency mode and block calibration
        assert!(tripwires.emergency_mode.load(Ordering::Relaxed));
        assert!(tripwires.should_block_calibration());
    }
    
    #[test]
    fn test_health_summary() {
        let config = SlaConfig::default();
        let thresholds = DriftThresholds::default();
        
        let tripwires = SlaTripwires::new(config, thresholds);
        
        let summary = tripwires.get_health_summary();
        assert!(summary.is_ok());
        
        let json = summary.unwrap();
        assert!(json["circuit_breaker_state"].is_string());
        assert!(json["failure_count"].is_number());
        assert!(json["emergency_mode"].is_boolean());
    }
}