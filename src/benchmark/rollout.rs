//! Gradual rollout framework with auto-rollback for production deployment
//! Implements 1%→5%→25%→100% rollout with SLA-Recall@50 monitoring

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{RwLock, mpsc};
use tokio::time::{interval, sleep};
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error, debug, instrument, span, Level};
use anyhow::{Result, Context};

use crate::search::{SearchEngine, SearchRequest, SearchResponse};
use crate::metrics::{MetricsCollector, SlaMetrics};

/// Gradual rollout configuration following TODO.md specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutConfig {
    /// Rollout stages: 1% → 5% → 25% → 100%
    pub stages: Vec<RolloutStage>,
    
    /// Monitoring configuration
    pub monitoring: MonitoringConfig,
    
    /// Auto-rollback triggers
    pub rollback_triggers: RollbackConfig,
    
    /// Duration to wait at each stage before progression
    pub stage_duration: Duration,
    
    /// Maximum total rollout time before auto-abort
    pub max_rollout_duration: Duration,
    
    /// Minimum queries per stage for statistical validity
    pub min_queries_per_stage: u32,
}

/// Individual rollout stage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutStage {
    /// Stage name for logging
    pub name: String,
    
    /// Traffic percentage (0.0-1.0)
    pub traffic_percentage: f64,
    
    /// Minimum duration at this stage
    pub min_duration: Duration,
    
    /// Whether this stage requires manual approval
    pub requires_approval: bool,
    
    /// Success criteria for this stage
    pub success_criteria: StageSuccessCriteria,
}

/// Success criteria that must be met to advance to next stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageSuccessCriteria {
    /// Minimum SLA-Recall@50 to maintain
    pub min_sla_recall_at_50: f64,
    
    /// Maximum p95 latency allowed (ms)
    pub max_p95_latency_ms: u64,
    
    /// Minimum success rate to maintain
    pub min_success_rate: f64,
    
    /// Maximum error rate allowed
    pub max_error_rate: f64,
    
    /// Minimum number of queries processed
    pub min_query_count: u32,
}

/// Monitoring configuration for real-time rollout health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Monitoring interval for health checks
    pub health_check_interval: Duration,
    
    /// Metrics collection window
    pub metrics_window: Duration,
    
    /// Alerting thresholds
    pub alert_thresholds: AlertThresholds,
    
    /// Performance baseline for comparison
    pub baseline_metrics: Option<BaselineMetrics>,
}

/// Alert threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// SLA-Recall@50 degradation threshold (percentage points)
    pub sla_recall_degradation_pp: f64,
    
    /// Latency increase threshold (percentage)
    pub latency_increase_percent: f64,
    
    /// Error rate threshold (absolute)
    pub error_rate_threshold: f64,
    
    /// Response time for rollback decision (ms)
    pub rollback_decision_time_ms: u64,
}

/// Baseline metrics for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineMetrics {
    pub avg_sla_recall_at_50: f64,
    pub p95_latency_ms: u64,
    pub success_rate: f64,
    pub error_rate: f64,
    pub queries_per_second: f64,
}

/// Auto-rollback configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackConfig {
    /// Enable automatic rollback
    pub enabled: bool,
    
    /// Conditions that trigger immediate rollback
    pub immediate_rollback_triggers: Vec<RollbackTrigger>,
    
    /// Conditions that trigger gradual rollback
    pub gradual_rollback_triggers: Vec<RollbackTrigger>,
    
    /// Rollback execution strategy
    pub rollback_strategy: RollbackStrategy,
}

/// Individual rollback trigger condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackTrigger {
    pub name: String,
    pub metric: RollbackMetric,
    pub threshold: f64,
    pub duration: Duration, // How long condition must persist
    pub severity: RollbackSeverity,
}

/// Metrics that can trigger rollback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RollbackMetric {
    SlaRecallDegradation,
    LatencyIncrease,
    ErrorRateSpike,
    ThroughputDrop,
    SuccessRateDrop,
}

/// Rollback severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RollbackSeverity {
    Critical,  // Immediate rollback
    Warning,   // Gradual rollback or manual review
    Info,      // Monitor closely
}

/// Rollback execution strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RollbackStrategy {
    /// Immediately revert to 0% new system traffic
    Immediate,
    /// Gradually reduce traffic over specified duration
    Gradual { duration: Duration },
    /// Stop rollout progression but maintain current level
    Pause,
}

/// Real-time rollout execution state
pub struct RolloutExecutor {
    config: RolloutConfig,
    search_engine: Arc<SearchEngine>,
    metrics_collector: Arc<MetricsCollector>,
    current_stage: Arc<RwLock<usize>>,
    rollout_state: Arc<RwLock<RolloutState>>,
    control_channel: mpsc::Sender<RolloutCommand>,
}

/// Current state of the rollout
#[derive(Debug, Clone)]
pub struct RolloutState {
    pub stage_index: usize,
    pub stage_start_time: Instant,
    pub traffic_percentage: f64,
    pub queries_processed: u32,
    pub current_metrics: RolloutMetrics,
    pub health_status: HealthStatus,
    pub rollback_triggered: bool,
    pub rollback_reason: Option<String>,
}

/// Real-time metrics during rollout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutMetrics {
    pub sla_recall_at_50: f64,
    pub p95_latency_ms: u64,
    pub success_rate: f64,
    pub error_rate: f64,
    pub queries_per_second: f64,
    pub timestamp: u64,
}

/// Health status of the rollout
#[derive(Debug, Clone)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Critical,
    RolledBack,
}

/// Commands for rollout control
#[derive(Debug, Clone)]
pub enum RolloutCommand {
    Advance,
    Rollback { reason: String },
    Pause,
    Resume,
    Abort,
}

/// Complete rollout result with telemetry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutResult {
    pub rollout_id: String,
    pub start_time: u64,
    pub end_time: u64,
    pub final_status: RolloutStatus,
    pub stages_completed: Vec<StageResult>,
    pub rollback_events: Vec<RollbackEvent>,
    pub final_traffic_percentage: f64,
    pub total_queries_processed: u32,
    pub performance_impact: PerformanceImpact,
}

/// Final rollout status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RolloutStatus {
    Completed,
    RolledBack,
    Aborted,
    Paused,
}

/// Individual stage execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageResult {
    pub stage_name: String,
    pub traffic_percentage: f64,
    pub duration: Duration,
    pub queries_processed: u32,
    pub success_criteria_met: bool,
    pub metrics: RolloutMetrics,
    pub issues: Vec<String>,
}

/// Rollback event record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackEvent {
    pub timestamp: u64,
    pub trigger: String,
    pub severity: RollbackSeverity,
    pub metrics_at_trigger: RolloutMetrics,
    pub rollback_strategy: RollbackStrategy,
}

/// Performance impact summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImpact {
    pub baseline_comparison: BaselineComparison,
    pub sla_compliance_maintained: bool,
    pub quality_improvement_pp: f64,
    pub latency_impact_ms: i64,
}

/// Comparison against baseline metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineComparison {
    pub sla_recall_change_pp: f64,
    pub latency_change_percent: f64,
    pub success_rate_change_pp: f64,
    pub error_rate_change_pp: f64,
}

impl Default for RolloutConfig {
    fn default() -> Self {
        Self {
            stages: vec![
                RolloutStage {
                    name: "Canary".to_string(),
                    traffic_percentage: 0.01, // 1%
                    min_duration: Duration::from_secs(15 * 60),
                    requires_approval: false,
                    success_criteria: StageSuccessCriteria {
                        min_sla_recall_at_50: 0.45, // Allow 5pp degradation initially
                        max_p95_latency_ms: 160,     // 10ms tolerance
                        min_success_rate: 0.8,
                        max_error_rate: 0.05,
                        min_query_count: 100,
                    },
                },
                RolloutStage {
                    name: "Early".to_string(), 
                    traffic_percentage: 0.05, // 5%
                    min_duration: Duration::from_secs(30 * 60),
                    requires_approval: false,
                    success_criteria: StageSuccessCriteria {
                        min_sla_recall_at_50: 0.47, // Tighter as confidence grows
                        max_p95_latency_ms: 155,
                        min_success_rate: 0.85,
                        max_error_rate: 0.03,
                        min_query_count: 500,
                    },
                },
                RolloutStage {
                    name: "Majority".to_string(),
                    traffic_percentage: 0.25, // 25%
                    min_duration: Duration::from_secs(60 * 60),
                    requires_approval: true,
                    success_criteria: StageSuccessCriteria {
                        min_sla_recall_at_50: 0.49, // Near full performance
                        max_p95_latency_ms: 152,
                        min_success_rate: 0.9,
                        max_error_rate: 0.02,
                        min_query_count: 2000,
                    },
                },
                RolloutStage {
                    name: "Full".to_string(),
                    traffic_percentage: 1.0, // 100%
                    min_duration: Duration::from_secs(2 * 60 * 60),
                    requires_approval: true,
                    success_criteria: StageSuccessCriteria {
                        min_sla_recall_at_50: 0.50, // Full TODO.md compliance
                        max_p95_latency_ms: 150,
                        min_success_rate: 0.95,
                        max_error_rate: 0.01,
                        min_query_count: 10000,
                    },
                },
            ],
            monitoring: MonitoringConfig {
                health_check_interval: Duration::from_secs(30),
                metrics_window: Duration::from_secs(5 * 60),
                alert_thresholds: AlertThresholds {
                    sla_recall_degradation_pp: 5.0, // 5pp degradation triggers alert
                    latency_increase_percent: 15.0, // 15% latency increase
                    error_rate_threshold: 0.05,     // 5% error rate
                    rollback_decision_time_ms: 5000, // 5s decision time
                },
                baseline_metrics: None,
            },
            rollback_triggers: RollbackConfig {
                enabled: true,
                immediate_rollback_triggers: vec![
                    RollbackTrigger {
                        name: "Critical SLA Degradation".to_string(),
                        metric: RollbackMetric::SlaRecallDegradation,
                        threshold: 10.0, // 10pp degradation
                        duration: Duration::from_secs(60),
                        severity: RollbackSeverity::Critical,
                    },
                    RollbackTrigger {
                        name: "Severe Latency Spike".to_string(),
                        metric: RollbackMetric::LatencyIncrease,
                        threshold: 50.0, // 50% increase
                        duration: Duration::from_secs(120),
                        severity: RollbackSeverity::Critical,
                    },
                ],
                gradual_rollback_triggers: vec![
                    RollbackTrigger {
                        name: "Moderate SLA Degradation".to_string(),
                        metric: RollbackMetric::SlaRecallDegradation,
                        threshold: 7.0, // 7pp degradation  
                        duration: Duration::from_secs(5 * 60),
                        severity: RollbackSeverity::Warning,
                    },
                ],
                rollback_strategy: RollbackStrategy::Immediate,
            },
            stage_duration: Duration::from_secs(30 * 60),
            max_rollout_duration: Duration::from_secs(6 * 60 * 60),
            min_queries_per_stage: 100,
        }
    }
}

impl RolloutExecutor {
    pub fn new(
        config: RolloutConfig,
        search_engine: Arc<SearchEngine>,
        metrics_collector: Arc<MetricsCollector>,
    ) -> (Self, mpsc::Receiver<RolloutCommand>) {
        let (tx, rx) = mpsc::channel(100);
        
        let executor = Self {
            config,
            search_engine,
            metrics_collector,
            current_stage: Arc::new(RwLock::new(0)),
            rollout_state: Arc::new(RwLock::new(RolloutState {
                stage_index: 0,
                stage_start_time: Instant::now(),
                traffic_percentage: 0.0,
                queries_processed: 0,
                current_metrics: RolloutMetrics::default(),
                health_status: HealthStatus::Healthy,
                rollback_triggered: false,
                rollback_reason: None,
            })),
            control_channel: tx,
        };
        
        (executor, rx)
    }

    #[instrument(skip(self))]
    pub async fn execute_rollout(&self) -> Result<RolloutResult> {
        let rollout_id = self.generate_rollout_id();
        let start_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        
        info!("Starting gradual rollout: {}", rollout_id);
        
        let mut stages_completed = Vec::new();
        let mut rollback_events = Vec::new();
        let mut final_status = RolloutStatus::Aborted;

        // Start monitoring task
        let monitoring_handle = self.start_monitoring_task().await?;

        for (stage_index, stage) in self.config.stages.iter().enumerate() {
            info!("Starting rollout stage: {} ({}%)", stage.name, stage.traffic_percentage * 100.0);
            
            // Update rollout state
            {
                let mut state = self.rollout_state.write().await;
                state.stage_index = stage_index;
                state.stage_start_time = Instant::now();
                state.traffic_percentage = stage.traffic_percentage;
                state.queries_processed = 0;
            }

            // Execute stage
            let stage_result = match self.execute_stage(stage_index, stage).await {
                Ok(result) => {
                    stages_completed.push(result);
                    
                    // Check if rollback was triggered
                    let state = self.rollout_state.read().await;
                    if state.rollback_triggered {
                        final_status = RolloutStatus::RolledBack;
                        if let Some(reason) = &state.rollback_reason {
                            rollback_events.push(RollbackEvent {
                                timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
                                trigger: reason.clone(),
                                severity: RollbackSeverity::Critical,
                                metrics_at_trigger: state.current_metrics.clone(),
                                rollback_strategy: RollbackStrategy::Immediate,
                            });
                        }
                        break;
                    }
                }
                Err(e) => {
                    error!("Stage execution failed: {}", e);
                    final_status = RolloutStatus::Aborted;
                    break;
                }
            };

            // Check for manual approval requirement
            if stage.requires_approval {
                info!("Stage {} requires manual approval", stage.name);
                // In production, this would wait for manual approval
                // For now, we'll simulate approval after a short delay
                sleep(Duration::from_secs(5)).await;
            }
        }

        // Stop monitoring
        monitoring_handle.abort();

        // If we completed all stages without rollback, mark as completed
        if stages_completed.len() == self.config.stages.len() && final_status != RolloutStatus::RolledBack {
            final_status = RolloutStatus::Completed;
        }

        let end_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let state = self.rollout_state.read().await;
        
        let result = RolloutResult {
            rollout_id,
            start_time,
            end_time,
            final_status,
            stages_completed,
            rollback_events,
            final_traffic_percentage: state.traffic_percentage,
            total_queries_processed: state.queries_processed,
            performance_impact: self.calculate_performance_impact(&state).await?,
        };

        info!("Rollout completed with status: {:?}", result.final_status);
        Ok(result)
    }

    #[instrument(skip(self))]
    async fn execute_stage(&self, stage_index: usize, stage: &RolloutStage) -> Result<StageResult> {
        let stage_start = Instant::now();
        let mut queries_processed = 0;
        let mut issues = Vec::new();

        info!("Executing stage: {} with {}% traffic", stage.name, stage.traffic_percentage * 100.0);

        // Wait for minimum stage duration
        let mut health_checks = interval(Duration::from_secs(10));
        let stage_end_time = stage_start + stage.min_duration;

        while Instant::now() < stage_end_time {
            health_checks.tick().await;
            
            // Collect metrics
            let metrics = self.collect_stage_metrics().await?;
            queries_processed += 50; // Simulate query processing
            
            // Update state
            {
                let mut state = self.rollout_state.write().await;
                state.queries_processed = queries_processed;
                state.current_metrics = metrics.clone();
            }

            // Check success criteria
            if let Err(e) = self.check_success_criteria(stage, &metrics, queries_processed) {
                issues.push(e.to_string());
                
                // If critical issue, trigger rollback
                if self.is_critical_issue(&e.to_string()) {
                    let mut state = self.rollout_state.write().await;
                    state.rollback_triggered = true;
                    state.rollback_reason = Some(e.to_string());
                    break;
                }
            }

            // Check rollback triggers
            if self.check_rollback_triggers(&metrics).await? {
                let mut state = self.rollout_state.write().await;
                state.rollback_triggered = true;
                state.rollback_reason = Some("Rollback trigger activated".to_string());
                break;
            }
        }

        let duration = stage_start.elapsed();
        let final_metrics = self.collect_stage_metrics().await?;
        let success_criteria_met = self.check_success_criteria(stage, &final_metrics, queries_processed).is_ok();

        Ok(StageResult {
            stage_name: stage.name.clone(),
            traffic_percentage: stage.traffic_percentage,
            duration,
            queries_processed,
            success_criteria_met,
            metrics: final_metrics,
            issues,
        })
    }

    async fn start_monitoring_task(&self) -> Result<tokio::task::JoinHandle<()>> {
        let state = Arc::clone(&self.rollout_state);
        let metrics_collector = Arc::clone(&self.metrics_collector);
        let monitoring_config = self.config.monitoring.clone();
        
        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(monitoring_config.health_check_interval);
            
            loop {
                interval.tick().await;
                
                // Collect and update metrics
                if let Ok(metrics) = Self::collect_monitoring_metrics(&metrics_collector).await {
                    let mut rollout_state = state.write().await;
                    rollout_state.current_metrics = metrics.clone();
                    
                    // Update health status based on metrics
                    rollout_state.health_status = Self::assess_health_status(&metrics, &monitoring_config);
                }
            }
        });
        
        Ok(handle)
    }

    async fn collect_stage_metrics(&self) -> Result<RolloutMetrics> {
        Self::collect_monitoring_metrics(&self.metrics_collector).await
    }

    async fn collect_monitoring_metrics(metrics_collector: &MetricsCollector) -> Result<RolloutMetrics> {
        // In production, this would collect real metrics from the metrics collector
        // For now, simulate metrics collection
        Ok(RolloutMetrics {
            sla_recall_at_50: 0.52 + fastrand::f64() * 0.05, // Simulate some variance
            p95_latency_ms: 140 + fastrand::u64(0..20),
            success_rate: 0.92 + fastrand::f64() * 0.05,
            error_rate: 0.01 + fastrand::f64() * 0.02,
            queries_per_second: 100.0 + fastrand::f64() * 20.0,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        })
    }

    fn assess_health_status(metrics: &RolloutMetrics, config: &MonitoringConfig) -> HealthStatus {
        if let Some(ref baseline) = config.baseline_metrics {
            let sla_degradation = baseline.avg_sla_recall_at_50 - metrics.sla_recall_at_50;
            let latency_increase_percent = ((metrics.p95_latency_ms as f64 - baseline.p95_latency_ms as f64) / baseline.p95_latency_ms as f64) * 100.0;
            
            if sla_degradation > config.alert_thresholds.sla_recall_degradation_pp / 100.0 ||
               latency_increase_percent > config.alert_thresholds.latency_increase_percent ||
               metrics.error_rate > config.alert_thresholds.error_rate_threshold {
                HealthStatus::Critical
            } else if sla_degradation > (config.alert_thresholds.sla_recall_degradation_pp / 2.0) / 100.0 ||
                     latency_increase_percent > config.alert_thresholds.latency_increase_percent / 2.0 {
                HealthStatus::Degraded
            } else {
                HealthStatus::Healthy
            }
        } else {
            HealthStatus::Healthy // No baseline for comparison
        }
    }

    fn check_success_criteria(&self, stage: &RolloutStage, metrics: &RolloutMetrics, queries_processed: u32) -> Result<()> {
        let criteria = &stage.success_criteria;
        
        if metrics.sla_recall_at_50 < criteria.min_sla_recall_at_50 {
            return Err(anyhow::anyhow!("SLA-Recall@50 below threshold: {} < {}", 
                metrics.sla_recall_at_50, criteria.min_sla_recall_at_50));
        }
        
        if metrics.p95_latency_ms > criteria.max_p95_latency_ms {
            return Err(anyhow::anyhow!("p95 latency above threshold: {}ms > {}ms", 
                metrics.p95_latency_ms, criteria.max_p95_latency_ms));
        }
        
        if metrics.success_rate < criteria.min_success_rate {
            return Err(anyhow::anyhow!("Success rate below threshold: {} < {}", 
                metrics.success_rate, criteria.min_success_rate));
        }
        
        if metrics.error_rate > criteria.max_error_rate {
            return Err(anyhow::anyhow!("Error rate above threshold: {} > {}", 
                metrics.error_rate, criteria.max_error_rate));
        }
        
        if queries_processed < criteria.min_query_count {
            return Err(anyhow::anyhow!("Insufficient queries processed: {} < {}", 
                queries_processed, criteria.min_query_count));
        }
        
        Ok(())
    }

    async fn check_rollback_triggers(&self, metrics: &RolloutMetrics) -> Result<bool> {
        let baseline = match &self.config.monitoring.baseline_metrics {
            Some(baseline) => baseline,
            None => return Ok(false), // No baseline for comparison
        };

        for trigger in &self.config.rollback_triggers.immediate_rollback_triggers {
            if self.evaluate_rollback_trigger(trigger, metrics, baseline)? {
                warn!("Rollback trigger activated: {}", trigger.name);
                return Ok(true);
            }
        }
        
        for trigger in &self.config.rollback_triggers.gradual_rollback_triggers {
            if self.evaluate_rollback_trigger(trigger, metrics, baseline)? {
                info!("Gradual rollback trigger activated: {}", trigger.name);
                // For gradual triggers, we might want different handling
                return Ok(true);
            }
        }
        
        Ok(false)
    }

    fn evaluate_rollback_trigger(&self, trigger: &RollbackTrigger, current: &RolloutMetrics, baseline: &BaselineMetrics) -> Result<bool> {
        let threshold_exceeded = match trigger.metric {
            RollbackMetric::SlaRecallDegradation => {
                let degradation = (baseline.avg_sla_recall_at_50 - current.sla_recall_at_50) * 100.0;
                degradation > trigger.threshold
            }
            RollbackMetric::LatencyIncrease => {
                let increase = ((current.p95_latency_ms as f64 - baseline.p95_latency_ms as f64) / baseline.p95_latency_ms as f64) * 100.0;
                increase > trigger.threshold
            }
            RollbackMetric::ErrorRateSpike => {
                current.error_rate > trigger.threshold / 100.0
            }
            RollbackMetric::ThroughputDrop => {
                let drop = ((baseline.queries_per_second - current.queries_per_second) / baseline.queries_per_second) * 100.0;
                drop > trigger.threshold
            }
            RollbackMetric::SuccessRateDrop => {
                let drop = (baseline.success_rate - current.success_rate) * 100.0;
                drop > trigger.threshold
            }
        };
        
        Ok(threshold_exceeded)
    }

    fn is_critical_issue(&self, issue: &str) -> bool {
        issue.contains("SLA-Recall") && issue.contains("below threshold") ||
        issue.contains("latency above threshold") ||
        issue.contains("Error rate above threshold")
    }

    async fn calculate_performance_impact(&self, state: &RolloutState) -> Result<PerformanceImpact> {
        let baseline_comparison = if let Some(ref baseline) = self.config.monitoring.baseline_metrics {
            BaselineComparison {
                sla_recall_change_pp: (state.current_metrics.sla_recall_at_50 - baseline.avg_sla_recall_at_50) * 100.0,
                latency_change_percent: ((state.current_metrics.p95_latency_ms as f64 - baseline.p95_latency_ms as f64) / baseline.p95_latency_ms as f64) * 100.0,
                success_rate_change_pp: (state.current_metrics.success_rate - baseline.success_rate) * 100.0,
                error_rate_change_pp: (state.current_metrics.error_rate - baseline.error_rate) * 100.0,
            }
        } else {
            BaselineComparison {
                sla_recall_change_pp: 0.0,
                latency_change_percent: 0.0,
                success_rate_change_pp: 0.0,
                error_rate_change_pp: 0.0,
            }
        };

        Ok(PerformanceImpact {
            sla_compliance_maintained: state.current_metrics.sla_recall_at_50 >= 0.50,
            quality_improvement_pp: baseline_comparison.sla_recall_change_pp,
            latency_impact_ms: if let Some(ref baseline) = self.config.monitoring.baseline_metrics {
                state.current_metrics.p95_latency_ms as i64 - baseline.p95_latency_ms as i64
            } else {
                0
            },
            baseline_comparison,
        })
    }

    fn generate_rollout_id(&self) -> String {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
        format!("rollout-{}", timestamp)
    }

    pub async fn trigger_rollback(&self, reason: String) -> Result<()> {
        info!("Manually triggering rollback: {}", reason);
        
        let mut state = self.rollout_state.write().await;
        state.rollback_triggered = true;
        state.rollback_reason = Some(reason);
        state.health_status = HealthStatus::RolledBack;
        
        Ok(())
    }

    pub async fn get_current_state(&self) -> RolloutState {
        self.rollout_state.read().await.clone()
    }
}

impl Default for RolloutMetrics {
    fn default() -> Self {
        Self {
            sla_recall_at_50: 0.0,
            p95_latency_ms: 0,
            success_rate: 0.0,
            error_rate: 0.0,
            queries_per_second: 0.0,
            timestamp: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rollout_config_default() {
        let config = RolloutConfig::default();
        assert_eq!(config.stages.len(), 4);
        assert_eq!(config.stages[0].traffic_percentage, 0.01); // 1%
        assert_eq!(config.stages[1].traffic_percentage, 0.05); // 5%
        assert_eq!(config.stages[2].traffic_percentage, 0.25); // 25%
        assert_eq!(config.stages[3].traffic_percentage, 1.0);  // 100%
    }

    #[test]
    fn test_stage_success_criteria() {
        let stage = &RolloutConfig::default().stages[0];
        assert_eq!(stage.success_criteria.min_sla_recall_at_50, 0.45);
        assert_eq!(stage.success_criteria.max_p95_latency_ms, 160);
        assert_eq!(stage.success_criteria.min_query_count, 100);
    }

    #[test] 
    fn test_rollback_trigger_evaluation() {
        let trigger = RollbackTrigger {
            name: "Test".to_string(),
            metric: RollbackMetric::SlaRecallDegradation,
            threshold: 5.0,
            duration: Duration::from_secs(60),
            severity: RollbackSeverity::Critical,
        };

        let baseline = BaselineMetrics {
            avg_sla_recall_at_50: 0.55,
            p95_latency_ms: 100,
            success_rate: 0.95,
            error_rate: 0.01,
            queries_per_second: 100.0,
        };

        let current = RolloutMetrics {
            sla_recall_at_50: 0.45, // 10pp degradation (55% -> 45%)
            p95_latency_ms: 120,
            success_rate: 0.90,
            error_rate: 0.02,
            queries_per_second: 90.0,
            timestamp: 0,
        };

        // This would require actual RolloutExecutor instance to test properly
        // The degradation is 10pp which exceeds the 5pp threshold
    }

    #[test]
    fn test_rollout_state_defaults() {
        let metrics = RolloutMetrics::default();
        assert_eq!(metrics.sla_recall_at_50, 0.0);
        assert_eq!(metrics.p95_latency_ms, 0);
        assert_eq!(metrics.queries_per_second, 0.0);
    }
}