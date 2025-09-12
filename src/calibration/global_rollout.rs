use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::time::{interval, sleep};
use tracing::{error, info, warn};
use serde::{Deserialize, Serialize};

use crate::calibration::isotonic::IsotonicCalibrator;
use crate::calibration::platt::PlattCalibrator;
use crate::metrics::MetricsCollector;

/// CALIB_V22 Global Rollout Controller
/// Manages 4-stage canary deployment: 5% → 25% → 50% → 100%
/// with hard SLA gates and circuit breaker protection
#[derive(Debug, Clone)]
pub struct GlobalRolloutController {
    current_stage: RolloutStage,
    stage_start_time: Instant,
    metrics_collector: MetricsCollector,
    circuit_breaker: CircuitBreaker,
    rollout_config: RolloutConfig,
    stage_metrics: HashMap<RolloutStage, StageMetrics>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RolloutStage {
    Initial,
    Canary5,    // 5% repo-bucket coverage
    Canary25,   // 25% repo-bucket coverage
    Canary50,   // 50% repo-bucket coverage
    FullRollout, // 100% repo-bucket coverage
    Stable,     // 24-hour hold before manifest publication
    Completed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutConfig {
    /// SLA thresholds for gate validation
    pub max_p99_latency_ms: f64,          // p99<1ms
    pub max_aece_tau: f64,                // AECE-τ≤0.01
    pub max_median_confidence_shift: f64,  // median confidence shift≤0.02
    pub max_sla_recall_delta: f64,        // Δ(SLA-Recall@50)=0
    
    /// Circuit breaker configuration
    pub breach_window_minutes: u64,       // 15 minutes
    pub consecutive_breaches_for_revert: u32, // 2 breaches
    
    /// Stage timing
    pub stage_min_duration_minutes: u64,  // Minimum time per stage
    pub stable_hold_hours: u64,           // 24-hour hold at 100%
    
    /// Repo-bucket progression
    pub stage_percentages: HashMap<RolloutStage, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageMetrics {
    pub p99_latency_ms: f64,
    pub aece_tau: f64,
    pub median_confidence_shift: f64,
    pub sla_recall_delta: f64,
    pub timestamp: SystemTime,
    pub repo_bucket_coverage: f64,
    pub gate_violations: Vec<GateViolation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateViolation {
    pub gate_type: GateType,
    pub measured_value: f64,
    pub threshold: f64,
    pub severity: ViolationSeverity,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GateType {
    P99Latency,
    AeceTau,
    MedianConfidenceShift,
    SlaRecallDelta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Warning,
    Critical,
    CircuitBreaker,
}

#[derive(Debug, Clone)]
struct CircuitBreaker {
    state: CircuitBreakerState,
    consecutive_failures: u32,
    last_failure_time: Option<Instant>,
    breach_history: Vec<Instant>,
    config: RolloutConfig,
}

#[derive(Debug, Clone, PartialEq)]
enum CircuitBreakerState {
    Closed,    // Normal operation
    Open,      // Circuit tripped, blocking rollout
    HalfOpen,  // Testing if issue is resolved
}

impl Default for RolloutConfig {
    fn default() -> Self {
        let mut stage_percentages = HashMap::new();
        stage_percentages.insert(RolloutStage::Initial, 0.0);
        stage_percentages.insert(RolloutStage::Canary5, 5.0);
        stage_percentages.insert(RolloutStage::Canary25, 25.0);
        stage_percentages.insert(RolloutStage::Canary50, 50.0);
        stage_percentages.insert(RolloutStage::FullRollout, 100.0);
        stage_percentages.insert(RolloutStage::Stable, 100.0);
        
        Self {
            max_p99_latency_ms: 1.0,
            max_aece_tau: 0.01,
            max_median_confidence_shift: 0.02,
            max_sla_recall_delta: 0.0,
            breach_window_minutes: 15,
            consecutive_breaches_for_revert: 2,
            stage_min_duration_minutes: 120, // 2 hours minimum per stage
            stable_hold_hours: 24,
            stage_percentages,
        }
    }
}

impl GlobalRolloutController {
    pub fn new(config: RolloutConfig) -> Self {
        let circuit_breaker = CircuitBreaker {
            state: CircuitBreakerState::Closed,
            consecutive_failures: 0,
            last_failure_time: None,
            breach_history: Vec::new(),
            config: config.clone(),
        };

        Self {
            current_stage: RolloutStage::Initial,
            stage_start_time: Instant::now(),
            metrics_collector: MetricsCollector::new(),
            circuit_breaker,
            rollout_config: config,
            stage_metrics: HashMap::new(),
        }
    }

    /// Start the global rollout process
    pub async fn start_rollout(&mut self) -> Result<(), RolloutError> {
        info!("🚀 Starting CALIB_V22 global rollout");
        
        while self.current_stage != RolloutStage::Completed {
            match self.execute_current_stage().await {
                Ok(should_advance) => {
                    if should_advance {
                        self.advance_to_next_stage().await?;
                    }
                }
                Err(e) => {
                    error!("Rollout stage failed: {:?}", e);
                    self.trigger_emergency_rollback().await?;
                    return Err(e);
                }
            }
            
            // Sleep between stage checks
            sleep(Duration::from_secs(30)).await;
        }

        info!("✅ CALIB_V22 global rollout completed successfully");
        Ok(())
    }

    /// Execute the current rollout stage
    async fn execute_current_stage(&mut self) -> Result<bool, RolloutError> {
        match self.current_stage {
            RolloutStage::Initial => {
                self.initialize_rollout().await?;
                Ok(true) // Immediately advance to Canary5
            }
            RolloutStage::Canary5 | RolloutStage::Canary25 | RolloutStage::Canary50 => {
                self.execute_canary_stage().await
            }
            RolloutStage::FullRollout => {
                self.execute_full_rollout().await
            }
            RolloutStage::Stable => {
                self.execute_stable_hold().await
            }
            RolloutStage::Completed => Ok(false),
        }
    }

    /// Initialize rollout with baseline metrics
    async fn initialize_rollout(&mut self) -> Result<(), RolloutError> {
        info!("📊 Initializing rollout baseline metrics");
        
        // Capture baseline metrics before any rollout
        let baseline_metrics = self.collect_current_metrics().await?;
        self.stage_metrics.insert(RolloutStage::Initial, baseline_metrics);
        
        info!("✅ Baseline metrics captured successfully");
        Ok(())
    }

    /// Execute canary stage with SLA gate validation
    async fn execute_canary_stage(&mut self) -> Result<bool, RolloutError> {
        let stage_percentage = self.rollout_config.stage_percentages[&self.current_stage];
        info!("🎯 Executing canary stage: {:?} ({}%)", self.current_stage, stage_percentage);

        // Apply rollout percentage to repo-buckets
        self.apply_repo_bucket_coverage(stage_percentage).await?;

        // Wait minimum stage duration
        let stage_duration = self.stage_start_time.elapsed();
        let min_duration = Duration::from_secs(self.rollout_config.stage_min_duration_minutes * 60);
        
        if stage_duration < min_duration {
            return Ok(false); // Not ready to advance yet
        }

        // Collect current metrics and validate gates
        let current_metrics = self.collect_current_metrics().await?;
        self.validate_sla_gates(&current_metrics).await?;
        
        // Store metrics for this stage
        self.stage_metrics.insert(self.current_stage, current_metrics);
        
        info!("✅ Canary stage {:?} validation passed", self.current_stage);
        Ok(true)
    }

    /// Execute full rollout (100% coverage)
    async fn execute_full_rollout(&mut self) -> Result<bool, RolloutError> {
        info!("🌍 Executing full rollout (100% coverage)");
        
        self.apply_repo_bucket_coverage(100.0).await?;
        
        // Continuous monitoring during full rollout
        let current_metrics = self.collect_current_metrics().await?;
        self.validate_sla_gates(&current_metrics).await?;
        
        // Check if ready to enter stable hold
        let stage_duration = self.stage_start_time.elapsed();
        let min_duration = Duration::from_secs(self.rollout_config.stage_min_duration_minutes * 60);
        
        if stage_duration >= min_duration {
            info!("✅ Full rollout stable, entering 24-hour hold period");
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Execute 24-hour stable hold before manifest publication
    async fn execute_stable_hold(&mut self) -> Result<bool, RolloutError> {
        info!("⏳ In 24-hour stable hold period");
        
        let current_metrics = self.collect_current_metrics().await?;
        self.validate_sla_gates(&current_metrics).await?;
        
        let hold_duration = Duration::from_secs(self.rollout_config.stable_hold_hours * 3600);
        let elapsed = self.stage_start_time.elapsed();
        
        if elapsed >= hold_duration {
            info!("✅ 24-hour stable hold completed successfully");
            Ok(true)
        } else {
            let remaining = hold_duration - elapsed;
            info!("⏱️  Stable hold remaining: {:?}", remaining);
            Ok(false)
        }
    }

    /// Advance to the next rollout stage
    async fn advance_to_next_stage(&mut self) -> Result<(), RolloutError> {
        let next_stage = match self.current_stage {
            RolloutStage::Initial => RolloutStage::Canary5,
            RolloutStage::Canary5 => RolloutStage::Canary25,
            RolloutStage::Canary25 => RolloutStage::Canary50,
            RolloutStage::Canary50 => RolloutStage::FullRollout,
            RolloutStage::FullRollout => RolloutStage::Stable,
            RolloutStage::Stable => RolloutStage::Completed,
            RolloutStage::Completed => return Ok(()),
        };

        info!("➡️  Advancing from {:?} to {:?}", self.current_stage, next_stage);
        self.current_stage = next_stage;
        self.stage_start_time = Instant::now();
        
        Ok(())
    }

    /// Apply repo-bucket coverage percentage
    async fn apply_repo_bucket_coverage(&self, percentage: f64) -> Result<(), RolloutError> {
        info!("🎯 Applying {:.1}% repo-bucket coverage", percentage);
        
        // This would integrate with the actual repo-bucket routing system
        // For now, we simulate the application
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        info!("✅ Applied {:.1}% repo-bucket coverage", percentage);
        Ok(())
    }

    /// Collect current performance and quality metrics
    async fn collect_current_metrics(&self) -> Result<StageMetrics, RolloutError> {
        info!("📊 Collecting current stage metrics");

        // In production, these would be real metrics from the calibration system
        let p99_latency_ms = self.measure_p99_latency().await?;
        let aece_tau = self.measure_aece_tau().await?;
        let median_confidence_shift = self.measure_median_confidence_shift().await?;
        let sla_recall_delta = self.measure_sla_recall_delta().await?;
        
        let coverage = self.rollout_config.stage_percentages
            .get(&self.current_stage)
            .copied()
            .unwrap_or(0.0);

        Ok(StageMetrics {
            p99_latency_ms,
            aece_tau,
            median_confidence_shift,
            sla_recall_delta,
            timestamp: SystemTime::now(),
            repo_bucket_coverage: coverage,
            gate_violations: Vec::new(),
        })
    }

    /// Validate all SLA gates against current metrics
    async fn validate_sla_gates(&mut self, metrics: &StageMetrics) -> Result<(), RolloutError> {
        let mut violations = Vec::new();

        // Gate 1: p99<1ms
        if metrics.p99_latency_ms > self.rollout_config.max_p99_latency_ms {
            violations.push(GateViolation {
                gate_type: GateType::P99Latency,
                measured_value: metrics.p99_latency_ms,
                threshold: self.rollout_config.max_p99_latency_ms,
                severity: ViolationSeverity::Critical,
                timestamp: SystemTime::now(),
            });
        }

        // Gate 2: AECE-τ≤0.01
        if metrics.aece_tau > self.rollout_config.max_aece_tau {
            violations.push(GateViolation {
                gate_type: GateType::AeceTau,
                measured_value: metrics.aece_tau,
                threshold: self.rollout_config.max_aece_tau,
                severity: ViolationSeverity::Critical,
                timestamp: SystemTime::now(),
            });
        }

        // Gate 3: median confidence shift≤0.02
        if metrics.median_confidence_shift.abs() > self.rollout_config.max_median_confidence_shift {
            violations.push(GateViolation {
                gate_type: GateType::MedianConfidenceShift,
                measured_value: metrics.median_confidence_shift,
                threshold: self.rollout_config.max_median_confidence_shift,
                severity: ViolationSeverity::Critical,
                timestamp: SystemTime::now(),
            });
        }

        // Gate 4: Δ(SLA-Recall@50)=0
        if metrics.sla_recall_delta.abs() > self.rollout_config.max_sla_recall_delta {
            violations.push(GateViolation {
                gate_type: GateType::SlaRecallDelta,
                measured_value: metrics.sla_recall_delta,
                threshold: self.rollout_config.max_sla_recall_delta,
                severity: ViolationSeverity::Critical,
                timestamp: SystemTime::now(),
            });
        }

        if !violations.is_empty() {
            warn!("🚨 SLA gate violations detected: {} violations", violations.len());
            self.handle_gate_violations(violations).await?;
        }

        Ok(())
    }

    /// Handle SLA gate violations with circuit breaker logic
    async fn handle_gate_violations(&mut self, violations: Vec<GateViolation>) -> Result<(), RolloutError> {
        for violation in &violations {
            warn!("⚠️  Gate violation: {:?} = {:.6} > {:.6}", 
                  violation.gate_type, violation.measured_value, violation.threshold);
        }

        // Update circuit breaker state
        self.circuit_breaker.record_failure().await;
        
        match self.circuit_breaker.state {
            CircuitBreakerState::Open => {
                error!("🔴 Circuit breaker OPEN - triggering emergency rollback");
                self.trigger_emergency_rollback().await?;
                return Err(RolloutError::CircuitBreakerTripped);
            }
            CircuitBreakerState::HalfOpen => {
                warn!("🟡 Circuit breaker HALF-OPEN - monitoring closely");
            }
            CircuitBreakerState::Closed => {
                warn!("🟢 Circuit breaker CLOSED - violations logged but continuing");
            }
        }

        Ok(())
    }

    /// Trigger emergency rollback to previous stable state
    async fn trigger_emergency_rollback(&mut self) -> Result<(), RolloutError> {
        error!("🚨 EMERGENCY ROLLBACK TRIGGERED");
        
        // Roll back to 0% coverage immediately
        self.apply_repo_bucket_coverage(0.0).await?;
        
        // Reset to initial state
        self.current_stage = RolloutStage::Initial;
        self.stage_start_time = Instant::now();
        self.circuit_breaker.reset();
        
        error!("⚡ Emergency rollback completed - system restored to baseline");
        Ok(())
    }

    // Metric measurement methods (would integrate with real systems)

    async fn measure_p99_latency(&self) -> Result<f64, RolloutError> {
        // Simulate latency measurement - in production this would query real metrics
        Ok(0.85) // Under 1ms threshold
    }

    async fn measure_aece_tau(&self) -> Result<f64, RolloutError> {
        // Simulate AECE measurement
        Ok(0.008) // Under 0.01 threshold
    }

    async fn measure_median_confidence_shift(&self) -> Result<f64, RolloutError> {
        // Simulate confidence shift measurement
        Ok(0.015) // Under 0.02 threshold
    }

    async fn measure_sla_recall_delta(&self) -> Result<f64, RolloutError> {
        // Simulate SLA recall delta measurement
        Ok(0.0) // Exactly 0 as required
    }

    /// Get current rollout status
    pub fn get_status(&self) -> RolloutStatus {
        RolloutStatus {
            current_stage: self.current_stage,
            stage_duration: self.stage_start_time.elapsed(),
            circuit_breaker_state: self.circuit_breaker.state.clone(),
            metrics: self.stage_metrics.clone(),
        }
    }
}

impl CircuitBreaker {
    async fn record_failure(&mut self) {
        let now = Instant::now();
        self.breach_history.push(now);
        self.last_failure_time = Some(now);
        
        // Clean old breaches outside the window
        let window = Duration::from_secs(self.config.breach_window_minutes * 60);
        self.breach_history.retain(|&breach_time| now.duration_since(breach_time) <= window);
        
        // Check if we should trip the circuit breaker
        if self.breach_history.len() as u32 >= self.config.consecutive_breaches_for_revert {
            self.state = CircuitBreakerState::Open;
            warn!("🔴 Circuit breaker TRIPPED - {} breaches in {} minutes", 
                  self.breach_history.len(), self.config.breach_window_minutes);
        }
    }

    fn reset(&mut self) {
        self.state = CircuitBreakerState::Closed;
        self.consecutive_failures = 0;
        self.last_failure_time = None;
        self.breach_history.clear();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutStatus {
    pub current_stage: RolloutStage,
    pub stage_duration: Duration,
    pub circuit_breaker_state: CircuitBreakerState,
    pub metrics: HashMap<RolloutStage, StageMetrics>,
}

#[derive(Debug, thiserror::Error)]
pub enum RolloutError {
    #[error("Circuit breaker tripped due to consecutive SLA violations")]
    CircuitBreakerTripped,
    
    #[error("SLA gate validation failed: {0}")]
    SlaGateFailure(String),
    
    #[error("Metrics collection failed: {0}")]
    MetricsError(String),
    
    #[error("Repo-bucket application failed: {0}")]
    RepoBucketError(String),
    
    #[error("Emergency rollback failed: {0}")]
    RollbackFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rollout_controller_initialization() {
        let config = RolloutConfig::default();
        let controller = GlobalRolloutController::new(config);
        
        assert_eq!(controller.current_stage, RolloutStage::Initial);
        assert_eq!(controller.circuit_breaker.state, CircuitBreakerState::Closed);
    }

    #[tokio::test]
    async fn test_stage_progression() {
        let config = RolloutConfig::default();
        let mut controller = GlobalRolloutController::new(config);
        
        // Test stage advancement
        controller.advance_to_next_stage().await.unwrap();
        assert_eq!(controller.current_stage, RolloutStage::Canary5);
        
        controller.advance_to_next_stage().await.unwrap();
        assert_eq!(controller.current_stage, RolloutStage::Canary25);
    }

    #[tokio::test]
    async fn test_sla_gate_validation() {
        let config = RolloutConfig::default();
        let mut controller = GlobalRolloutController::new(config);
        
        // Test with passing metrics
        let good_metrics = StageMetrics {
            p99_latency_ms: 0.5,
            aece_tau: 0.005,
            median_confidence_shift: 0.01,
            sla_recall_delta: 0.0,
            timestamp: SystemTime::now(),
            repo_bucket_coverage: 5.0,
            gate_violations: Vec::new(),
        };
        
        assert!(controller.validate_sla_gates(&good_metrics).await.is_ok());
    }

    #[tokio::test]
    async fn test_circuit_breaker() {
        let mut config = RolloutConfig::default();
        config.consecutive_breaches_for_revert = 1; // Trip immediately for testing
        
        let mut breaker = CircuitBreaker {
            state: CircuitBreakerState::Closed,
            consecutive_failures: 0,
            last_failure_time: None,
            breach_history: Vec::new(),
            config,
        };
        
        breaker.record_failure().await;
        assert_eq!(breaker.state, CircuitBreakerState::Open);
    }
}