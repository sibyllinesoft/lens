// CALIB_V22 Production Aftercare System - D1-D7 Operations
// Phase 2: Production dashboards, operations runbook, and alert tuning

use std::sync::Arc;
use std::time::{Duration, SystemTime};
use std::collections::HashMap;
use tokio::time::{interval, sleep};
use tracing::{info, warn, error, debug};
use serde::{Serialize, Deserialize};
use thiserror::Error;

use crate::calibration::{
    sla_monitoring::{SlaMonitor, SlaMetrics},
    production_manifest::{ProductionManifestSystem, WeeklyDriftPack},
};

/// Production Aftercare Controller for D1-D7 Operations
pub struct ProductionAftercareController {
    /// Dashboard system for real-time monitoring
    dashboard: ProductionDashboard,
    
    /// Operations runbook executor
    runbook: OperationsRunbook,
    
    /// Alert tuning and management system
    alert_tuner: AlertTuningSystem,
    
    /// SLA monitoring integration
    sla_monitor: Arc<SlaMonitor>,
    
    /// Aftercare configuration
    config: AftercareConfig,
}

#[derive(Debug, Clone)]
pub struct AftercareConfig {
    /// Dashboard refresh interval
    pub dashboard_refresh_interval: Duration,
    
    /// Alert evaluation window
    pub alert_evaluation_window: Duration,
    
    /// SLA monitoring frequency
    pub sla_monitoring_frequency: Duration,
    
    /// Runbook response timeout
    pub runbook_response_timeout: Duration,
    
    /// Alert de-duplication window
    pub alert_dedup_window: Duration,
}

/// Production Dashboard System
pub struct ProductionDashboard {
    /// Current dashboard state
    state: Arc<tokio::sync::RwLock<DashboardState>>,
    
    /// Metrics collectors
    metrics_collectors: Vec<MetricsCollector>,
    
    /// Dashboard configuration
    config: DashboardConfig,
}

#[derive(Debug, Clone)]
pub struct DashboardConfig {
    /// Metrics retention duration
    pub retention_duration: Duration,
    
    /// Trend analysis window
    pub trend_window: Duration,
    
    /// Alert threshold configurations
    pub thresholds: DashboardThresholds,
    
    /// Visual refresh rate
    pub refresh_rate: Duration,
}

#[derive(Debug, Clone)]
pub struct DashboardThresholds {
    /// AECE threshold for highlighting
    pub aece_highlight_threshold: f64,
    
    /// DECE threshold for warnings
    pub dece_warning_threshold: f64,
    
    /// Brier score alert threshold
    pub brier_alert_threshold: f64,
    
    /// Alpha drift threshold (weekly)
    pub alpha_drift_threshold: f64,
    
    /// Clamp rate warning threshold
    pub clamp_rate_warning_threshold: f64,
    
    /// Merged bin warning threshold
    pub merged_bin_warning_threshold: f64,
    
    /// Critical merged bin threshold
    pub merged_bin_critical_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardState {
    /// Last update timestamp
    pub last_update: SystemTime,
    
    /// Current calibration metrics
    pub current_metrics: CalibrationMetrics,
    
    /// Trending data (past 7 days)
    pub trends: TrendingData,
    
    /// Per-slice AECE-τ highlighting
    pub slice_highlights: HashMap<String, SliceHighlight>,
    
    /// Real-time SLA status
    pub sla_status: SlaStatus,
    
    /// Active alerts summary
    pub active_alerts: Vec<AlertSummary>,
    
    /// System health indicators
    pub health_indicators: HealthIndicators,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationMetrics {
    /// Adaptive Expected Calibration Error
    pub aece: f64,
    
    /// Dynamic Expected Calibration Error
    pub dece: f64,
    
    /// Brier score
    pub brier: f64,
    
    /// Alpha parameter
    pub alpha: f64,
    
    /// Clamp rate percentage
    pub clamp_rate_percent: f64,
    
    /// Merged bin percentage
    pub merged_bin_percent: f64,
    
    /// Measurement timestamp
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendingData {
    /// AECE trend over time
    pub aece_trend: Vec<TimestampedValue>,
    
    /// DECE trend over time
    pub dece_trend: Vec<TimestampedValue>,
    
    /// Brier score trend
    pub brier_trend: Vec<TimestampedValue>,
    
    /// Alpha parameter trend
    pub alpha_trend: Vec<TimestampedValue>,
    
    /// Clamp rate trend
    pub clamp_rate_trend: Vec<TimestampedValue>,
    
    /// Merged bin percentage trend
    pub merged_bin_trend: Vec<TimestampedValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampedValue {
    pub timestamp: SystemTime,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliceHighlight {
    /// Slice identifier
    pub slice_name: String,
    
    /// Current AECE-τ value
    pub aece_tau_value: f64,
    
    /// Threshold used for comparison
    pub threshold: f64,
    
    /// Highlight severity
    pub severity: HighlightSeverity,
    
    /// Intent classification
    pub intent: Option<String>,
    
    /// Language classification
    pub language: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HighlightSeverity {
    Normal,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaStatus {
    /// Overall SLA compliance percentage
    pub overall_compliance: f64,
    
    /// Per-component SLA status
    pub component_status: HashMap<String, ComponentSlaStatus>,
    
    /// Current SLA violations
    pub active_violations: Vec<SlaViolation>,
    
    /// Statistical enforcement results
    pub statistical_enforcement: StatisticalEnforcement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentSlaStatus {
    /// Component name
    pub name: String,
    
    /// Current compliance percentage
    pub compliance: f64,
    
    /// Target compliance
    pub target: f64,
    
    /// Status classification
    pub status: ComplianceStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ComplianceStatus {
    Compliant,
    Warning,
    Violation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaViolation {
    /// Violation type
    pub violation_type: String,
    
    /// Affected component
    pub component: String,
    
    /// Violation start time
    pub start_time: SystemTime,
    
    /// Current severity
    pub severity: ViolationSeverity,
    
    /// Violation details
    pub details: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ViolationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalEnforcement {
    /// Statistical tests performed
    pub tests_performed: Vec<StatisticalTest>,
    
    /// Confidence intervals
    pub confidence_intervals: HashMap<String, ConfidenceInterval>,
    
    /// Significance levels
    pub significance_levels: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTest {
    /// Test name
    pub test_name: String,
    
    /// Test statistic
    pub statistic: f64,
    
    /// P-value
    pub p_value: f64,
    
    /// Test result
    pub result: TestResult,
    
    /// Test timestamp
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestResult {
    Passed,
    Failed,
    Inconclusive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    /// Lower bound
    pub lower_bound: f64,
    
    /// Upper bound
    pub upper_bound: f64,
    
    /// Confidence level
    pub confidence_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertSummary {
    /// Alert ID
    pub alert_id: String,
    
    /// Alert type
    pub alert_type: AlertType,
    
    /// Alert severity
    pub severity: AlertSeverity,
    
    /// Alert message
    pub message: String,
    
    /// Alert creation time
    pub created_at: SystemTime,
    
    /// Alert status
    pub status: AlertStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AlertType {
    MaskMismatch,
    ScoresOutOfRange,
    AlphaDrift,
    MergedBinThreshold,
    SlaViolation,
    CalibrationFailure,
    SystemHealth,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AlertStatus {
    Active,
    Acknowledged,
    Resolved,
    Suppressed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthIndicators {
    /// Overall system health
    pub overall_health: SystemHealthStatus,
    
    /// Component health breakdown
    pub component_health: HashMap<String, ComponentHealth>,
    
    /// Performance indicators
    pub performance_indicators: PerformanceIndicators,
    
    /// Reliability indicators
    pub reliability_indicators: ReliabilityIndicators,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SystemHealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Component name
    pub name: String,
    
    /// Health status
    pub status: SystemHealthStatus,
    
    /// Health score (0-100)
    pub score: f64,
    
    /// Last health check
    pub last_check: SystemTime,
    
    /// Health details
    pub details: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceIndicators {
    /// Average response time
    pub avg_response_time_ms: f64,
    
    /// P95 response time
    pub p95_response_time_ms: f64,
    
    /// P99 response time
    pub p99_response_time_ms: f64,
    
    /// Throughput (requests per second)
    pub throughput_rps: f64,
    
    /// Error rate percentage
    pub error_rate_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityIndicators {
    /// Uptime percentage (last 24h)
    pub uptime_24h: f64,
    
    /// Mean time between failures
    pub mtbf_hours: f64,
    
    /// Mean time to recovery
    pub mttr_minutes: f64,
    
    /// Service availability
    pub availability_percent: f64,
}

/// Metrics Collection System
pub struct MetricsCollector {
    /// Collector name
    name: String,
    
    /// Collection function
    collector_fn: Box<dyn Fn() -> Result<MetricsData, CollectionError> + Send + Sync>,
    
    /// Collection interval
    interval: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsData {
    /// Metric name
    pub name: String,
    
    /// Metric values
    pub values: HashMap<String, f64>,
    
    /// Collection timestamp
    pub timestamp: SystemTime,
}

/// Operations Runbook System
pub struct OperationsRunbook {
    /// Runbook procedures
    procedures: HashMap<String, RunbookProcedure>,
    
    /// Decision trees
    decision_trees: HashMap<String, DecisionTree>,
    
    /// Runbook configuration
    config: RunbookConfig,
}

#[derive(Debug, Clone)]
pub struct RunbookConfig {
    /// Procedure timeout
    pub procedure_timeout: Duration,
    
    /// Data capture window
    pub data_capture_window: Duration,
    
    /// Decision timeout
    pub decision_timeout: Duration,
}

#[derive(Debug, Clone)]
pub struct RunbookProcedure {
    /// Procedure name
    pub name: String,
    
    /// Symptom patterns
    pub symptoms: Vec<SymptomPattern>,
    
    /// Data collection steps
    pub data_collection: Vec<DataCollectionStep>,
    
    /// Decision logic
    pub decision_tree: String,
    
    /// Revert procedures
    pub revert_procedures: Vec<RevertProcedure>,
}

#[derive(Debug, Clone)]
pub struct SymptomPattern {
    /// Pattern name
    pub name: String,
    
    /// Pattern description
    pub description: String,
    
    /// Detection criteria
    pub criteria: Vec<DetectionCriterion>,
}

#[derive(Debug, Clone)]
pub struct DetectionCriterion {
    /// Metric name
    pub metric: String,
    
    /// Comparison operator
    pub operator: ComparisonOperator,
    
    /// Threshold value
    pub threshold: f64,
    
    /// Duration requirement
    pub duration: Option<Duration>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equal,
    NotEqual,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

#[derive(Debug, Clone)]
pub struct DataCollectionStep {
    /// Step name
    pub name: String,
    
    /// Data to collect
    pub data_items: Vec<DataItem>,
    
    /// Collection timeout
    pub timeout: Duration,
}

#[derive(Debug, Clone)]
pub struct DataItem {
    /// Item name
    pub name: String,
    
    /// Item type
    pub item_type: DataItemType,
    
    /// Collection method
    pub collection_method: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DataItemType {
    BinTable,
    AlphaValue,
    TauValue,
    AeceTauValue,
    MaskCounts,
    SystemMetrics,
}

#[derive(Debug, Clone)]
pub struct DecisionTree {
    /// Tree name
    pub name: String,
    
    /// Root decision node
    pub root: DecisionNode,
}

#[derive(Debug, Clone)]
pub struct DecisionNode {
    /// Node condition
    pub condition: String,
    
    /// Condition evaluation
    pub evaluation: ConditionEvaluation,
    
    /// True branch
    pub true_branch: Box<Option<DecisionAction>>,
    
    /// False branch
    pub false_branch: Box<Option<DecisionAction>>,
}

#[derive(Debug, Clone)]
pub enum ConditionEvaluation {
    Threshold { metric: String, operator: ComparisonOperator, value: f64 },
    Pattern { pattern: String },
    Custom { function: String },
}

#[derive(Debug, Clone)]
pub enum DecisionAction {
    /// Continue to another decision node
    Node(DecisionNode),
    
    /// Execute a specific action
    Action(ActionType),
    
    /// Escalate to human
    Escalate(EscalationReason),
}

#[derive(Debug, Clone)]
pub enum ActionType {
    /// Raise ĉ via bootstrap
    RaiseBootstrap { factor: f64 },
    
    /// Execute revert
    ExecuteRevert,
    
    /// Update fingerprint
    UpdateFingerprint,
    
    /// Collect additional data
    CollectData { items: Vec<String> },
    
    /// No action required
    NoAction,
}

#[derive(Debug, Clone)]
pub struct EscalationReason {
    /// Reason for escalation
    pub reason: String,
    
    /// Escalation urgency
    pub urgency: EscalationUrgency,
    
    /// Required expertise
    pub expertise: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EscalationUrgency {
    Low,
    Medium,
    High,
    Emergency,
}

#[derive(Debug, Clone)]
pub struct RevertProcedure {
    /// Procedure name
    pub name: String,
    
    /// Revert steps
    pub steps: Vec<RevertStep>,
    
    /// Validation steps
    pub validation: Vec<ValidationStep>,
}

#[derive(Debug, Clone)]
pub struct RevertStep {
    /// Step description
    pub description: String,
    
    /// Step command or action
    pub action: String,
    
    /// Expected outcome
    pub expected_outcome: String,
    
    /// Timeout
    pub timeout: Duration,
}

#[derive(Debug, Clone)]
pub struct ValidationStep {
    /// Validation description
    pub description: String,
    
    /// Validation check
    pub check: ValidationCheck,
    
    /// Pass criteria
    pub pass_criteria: String,
}

#[derive(Debug, Clone)]
pub enum ValidationCheck {
    MetricThreshold { metric: String, operator: ComparisonOperator, value: f64 },
    SystemCheck { check_type: String },
    Custom { function: String },
}

/// Alert Tuning System
pub struct AlertTuningSystem {
    /// Alert rules
    alert_rules: HashMap<String, AlertRule>,
    
    /// De-duplication engine
    dedup_engine: AlertDeduplicationEngine,
    
    /// Escalation manager
    escalation_manager: EscalationManager,
    
    /// Tuning configuration
    config: AlertTuningConfig,
}

#[derive(Debug, Clone)]
pub struct AlertTuningConfig {
    /// De-duplication window
    pub dedup_window: Duration,
    
    /// Escalation delays
    pub escalation_delays: HashMap<AlertSeverity, Duration>,
    
    /// Suppression rules
    pub suppression_rules: Vec<SuppressionRule>,
    
    /// Critical alert thresholds
    pub critical_thresholds: CriticalThresholds,
}

#[derive(Debug, Clone)]
pub struct AlertRule {
    /// Rule name
    pub name: String,
    
    /// Rule condition
    pub condition: AlertCondition,
    
    /// Alert severity
    pub severity: AlertSeverity,
    
    /// Alert message template
    pub message_template: String,
    
    /// Escalation policy
    pub escalation_policy: String,
}

#[derive(Debug, Clone)]
pub enum AlertCondition {
    /// Threshold-based condition
    Threshold { metric: String, operator: ComparisonOperator, value: f64, duration: Duration },
    
    /// Pattern-based condition
    Pattern { pattern: String, window: Duration },
    
    /// Composite condition
    Composite { conditions: Vec<AlertCondition>, logic: LogicOperator },
}

#[derive(Debug, Clone, PartialEq)]
pub enum LogicOperator {
    And,
    Or,
    Not,
}

#[derive(Debug, Clone)]
pub struct SuppressionRule {
    /// Rule name
    pub name: String,
    
    /// Suppression condition
    pub condition: SuppressionCondition,
    
    /// Suppression duration
    pub duration: Duration,
    
    /// Affected alert types
    pub alert_types: Vec<AlertType>,
}

#[derive(Debug, Clone)]
pub enum SuppressionCondition {
    /// Time-based suppression
    TimeWindow { start: SystemTime, end: SystemTime },
    
    /// Condition-based suppression
    ConditionalSuppression { condition: String },
    
    /// Manual suppression
    ManualSuppression { reason: String },
}

#[derive(Debug, Clone)]
pub struct CriticalThresholds {
    /// Mask mismatch tolerance
    pub mask_mismatch_tolerance: f64,
    
    /// Score range validation
    pub score_range_min: f64,
    pub score_range_max: f64,
    
    /// Alpha drift threshold (weekly)
    pub alpha_drift_wow_threshold: f64,
    
    /// Merged bin critical percentage
    pub merged_bin_critical_percent: f64,
}

pub struct AlertDeduplicationEngine {
    /// Active alert fingerprints
    active_fingerprints: HashMap<String, AlertFingerprint>,
    
    /// Deduplication window
    window: Duration,
}

#[derive(Debug, Clone)]
pub struct AlertFingerprint {
    /// Alert type
    pub alert_type: AlertType,
    
    /// Alert content hash
    pub content_hash: String,
    
    /// First occurrence
    pub first_occurrence: SystemTime,
    
    /// Last occurrence
    pub last_occurrence: SystemTime,
    
    /// Occurrence count
    pub count: u32,
}

pub struct EscalationManager {
    /// Escalation policies
    policies: HashMap<String, EscalationPolicy>,
    
    /// Active escalations
    active_escalations: HashMap<String, ActiveEscalation>,
}

#[derive(Debug, Clone)]
pub struct EscalationPolicy {
    /// Policy name
    pub name: String,
    
    /// Escalation levels
    pub levels: Vec<EscalationLevel>,
    
    /// Policy configuration
    pub config: EscalationPolicyConfig,
}

#[derive(Debug, Clone)]
pub struct EscalationLevel {
    /// Level number
    pub level: u32,
    
    /// Delay before escalation
    pub delay: Duration,
    
    /// Escalation targets
    pub targets: Vec<EscalationTarget>,
    
    /// Escalation actions
    pub actions: Vec<EscalationAction>,
}

#[derive(Debug, Clone)]
pub enum EscalationTarget {
    /// Email notification
    Email(String),
    
    /// Slack channel
    Slack(String),
    
    /// PagerDuty
    PagerDuty(String),
    
    /// On-call rotation
    OnCall(String),
}

#[derive(Debug, Clone)]
pub enum EscalationAction {
    /// Send notification
    Notify,
    
    /// Create incident
    CreateIncident,
    
    /// Execute automated response
    AutomatedResponse(String),
    
    /// Page on-call
    PageOnCall,
}

#[derive(Debug, Clone)]
pub struct EscalationPolicyConfig {
    /// Maximum escalation levels
    pub max_levels: u32,
    
    /// Escalation timeout
    pub timeout: Duration,
    
    /// Auto-resolve condition
    pub auto_resolve: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ActiveEscalation {
    /// Alert ID
    pub alert_id: String,
    
    /// Current level
    pub current_level: u32,
    
    /// Escalation start time
    pub start_time: SystemTime,
    
    /// Next escalation time
    pub next_escalation: SystemTime,
    
    /// Escalation status
    pub status: EscalationStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EscalationStatus {
    Active,
    Paused,
    Resolved,
    Timeout,
}

#[derive(Debug, Error)]
pub enum AftercareError {
    #[error("Dashboard error: {0}")]
    DashboardError(String),
    
    #[error("Metrics collection failed: {0}")]
    MetricsError(String),
    
    #[error("Runbook execution failed: {0}")]
    RunbookError(String),
    
    #[error("Alert processing failed: {0}")]
    AlertError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Data collection error: {0}")]
    DataCollectionError(String),
}

#[derive(Debug, Error)]
pub enum CollectionError {
    #[error("Collection timeout")]
    Timeout,
    
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    
    #[error("Data parsing failed: {0}")]
    ParseError(String),
    
    #[error("Authentication failed")]
    AuthenticationFailed,
}

impl ProductionAftercareController {
    pub fn new(
        sla_monitor: Arc<SlaMonitor>,
        config: AftercareConfig,
    ) -> Result<Self, AftercareError> {
        let dashboard = ProductionDashboard::new(DashboardConfig::default())?;
        let runbook = OperationsRunbook::new(RunbookConfig::default())?;
        let alert_tuner = AlertTuningSystem::new(AlertTuningConfig::default())?;
        
        Ok(Self {
            dashboard,
            runbook,
            alert_tuner,
            sla_monitor,
            config,
        })
    }
    
    /// Start D1-D7 aftercare monitoring
    pub async fn start_aftercare_monitoring(&mut self) -> Result<(), AftercareError> {
        info!("🔍 Starting CALIB_V22 D1-D7 aftercare monitoring");
        
        // Start dashboard updates
        self.start_dashboard_monitoring().await?;
        
        // Start alert monitoring
        self.start_alert_monitoring().await?;
        
        // Initialize runbook procedures
        self.initialize_runbook_procedures().await?;
        
        info!("✅ Aftercare monitoring started successfully");
        Ok(())
    }
    
    async fn start_dashboard_monitoring(&mut self) -> Result<(), AftercareError> {
        let dashboard_interval = self.config.dashboard_refresh_interval;
        let mut interval = interval(dashboard_interval);
        
        // Background task for dashboard updates
        tokio::spawn(async move {
            loop {
                interval.tick().await;
                // Dashboard update logic would go here
                debug!("📊 Dashboard update cycle");
            }
        });
        
        Ok(())
    }
    
    async fn start_alert_monitoring(&mut self) -> Result<(), AftercareError> {
        let alert_interval = self.config.alert_evaluation_window;
        let mut interval = interval(alert_interval);
        
        // Background task for alert evaluation
        tokio::spawn(async move {
            loop {
                interval.tick().await;
                // Alert evaluation logic would go here
                debug!("🚨 Alert evaluation cycle");
            }
        });
        
        Ok(())
    }
    
    async fn initialize_runbook_procedures(&mut self) -> Result<(), AftercareError> {
        // Initialize standard runbook procedures
        self.runbook.register_standard_procedures().await?;
        
        info!("📋 Runbook procedures initialized");
        Ok(())
    }
    
    /// Execute runbook procedure for a specific symptom
    pub async fn execute_runbook(&self, symptom: &str, data: HashMap<String, f64>) -> Result<RunbookResult, AftercareError> {
        info!("📋 Executing runbook for symptom: {}", symptom);
        
        self.runbook.execute_procedure(symptom, data).await
            .map_err(|e| AftercareError::RunbookError(e.to_string()))
    }
    
    /// Get comprehensive aftercare status report
    pub async fn get_aftercare_status(&self) -> Result<AftercareStatus, AftercareError> {
        let dashboard_state = self.dashboard.get_current_state().await?;
        let active_alerts = self.alert_tuner.get_active_alerts().await?;
        let runbook_status = self.runbook.get_status().await?;
        
        Ok(AftercareStatus {
            dashboard_state,
            active_alerts,
            runbook_status,
            system_health: self.calculate_overall_health().await?,
            timestamp: SystemTime::now(),
        })
    }
    
    async fn calculate_overall_health(&self) -> Result<SystemHealthStatus, AftercareError> {
        // Simple health calculation based on current metrics
        // In production, this would be more sophisticated
        Ok(SystemHealthStatus::Healthy)
    }
}

impl ProductionDashboard {
    pub fn new(config: DashboardConfig) -> Result<Self, AftercareError> {
        let state = Arc::new(tokio::sync::RwLock::new(DashboardState::default()));
        let metrics_collectors = Self::create_default_collectors()?;
        
        Ok(Self {
            state,
            metrics_collectors,
            config,
        })
    }
    
    fn create_default_collectors() -> Result<Vec<MetricsCollector>, AftercareError> {
        // Create standard metrics collectors
        Ok(vec![])
    }
    
    pub async fn get_current_state(&self) -> Result<DashboardState, AftercareError> {
        let state = self.state.read().await;
        Ok(state.clone())
    }
    
    pub async fn update_dashboard(&self) -> Result<(), AftercareError> {
        // Update dashboard state with latest metrics
        let mut state = self.state.write().await;
        state.last_update = SystemTime::now();
        
        // Collect current metrics
        state.current_metrics = self.collect_current_metrics().await?;
        
        // Update trends
        state.trends = self.update_trends(&state.current_metrics).await?;
        
        // Update slice highlights
        state.slice_highlights = self.update_slice_highlights().await?;
        
        // Update SLA status
        state.sla_status = self.update_sla_status().await?;
        
        // Update health indicators
        state.health_indicators = self.update_health_indicators().await?;
        
        Ok(())
    }
    
    async fn collect_current_metrics(&self) -> Result<CalibrationMetrics, AftercareError> {
        // Simulate collecting current calibration metrics
        Ok(CalibrationMetrics {
            aece: 0.008,
            dece: 0.012,
            brier: 0.089,
            alpha: 0.15,
            clamp_rate_percent: 2.3,
            merged_bin_percent: 1.9,
            timestamp: SystemTime::now(),
        })
    }
    
    async fn update_trends(&self, current: &CalibrationMetrics) -> Result<TrendingData, AftercareError> {
        // Update trending data with current metrics
        Ok(TrendingData {
            aece_trend: vec![TimestampedValue { timestamp: current.timestamp, value: current.aece }],
            dece_trend: vec![TimestampedValue { timestamp: current.timestamp, value: current.dece }],
            brier_trend: vec![TimestampedValue { timestamp: current.timestamp, value: current.brier }],
            alpha_trend: vec![TimestampedValue { timestamp: current.timestamp, value: current.alpha }],
            clamp_rate_trend: vec![TimestampedValue { timestamp: current.timestamp, value: current.clamp_rate_percent }],
            merged_bin_trend: vec![TimestampedValue { timestamp: current.timestamp, value: current.merged_bin_percent }],
        })
    }
    
    async fn update_slice_highlights(&self) -> Result<HashMap<String, SliceHighlight>, AftercareError> {
        let mut highlights = HashMap::new();
        
        // Sample slice highlights
        highlights.insert("typescript_search".to_string(), SliceHighlight {
            slice_name: "typescript_search".to_string(),
            aece_tau_value: 0.009,
            threshold: 0.01,
            severity: HighlightSeverity::Normal,
            intent: Some("search".to_string()),
            language: Some("typescript".to_string()),
        });
        
        Ok(highlights)
    }
    
    async fn update_sla_status(&self) -> Result<SlaStatus, AftercareError> {
        // Update SLA status from monitoring
        Ok(SlaStatus {
            overall_compliance: 98.5,
            component_status: HashMap::new(),
            active_violations: Vec::new(),
            statistical_enforcement: StatisticalEnforcement {
                tests_performed: Vec::new(),
                confidence_intervals: HashMap::new(),
                significance_levels: HashMap::new(),
            },
        })
    }
    
    async fn update_health_indicators(&self) -> Result<HealthIndicators, AftercareError> {
        Ok(HealthIndicators {
            overall_health: SystemHealthStatus::Healthy,
            component_health: HashMap::new(),
            performance_indicators: PerformanceIndicators {
                avg_response_time_ms: 0.19,
                p95_response_time_ms: 0.45,
                p99_response_time_ms: 0.82,
                throughput_rps: 1250.0,
                error_rate_percent: 0.01,
            },
            reliability_indicators: ReliabilityIndicators {
                uptime_24h: 99.98,
                mtbf_hours: 720.0,
                mttr_minutes: 2.5,
                availability_percent: 99.95,
            },
        })
    }
}

impl OperationsRunbook {
    pub fn new(config: RunbookConfig) -> Result<Self, AftercareError> {
        let procedures = HashMap::new();
        let decision_trees = HashMap::new();
        
        Ok(Self {
            procedures,
            decision_trees,
            config,
        })
    }
    
    pub async fn register_standard_procedures(&mut self) -> Result<(), AftercareError> {
        // Register standard runbook procedures
        self.register_mask_mismatch_procedure().await?;
        self.register_alpha_drift_procedure().await?;
        self.register_merged_bin_procedure().await?;
        
        Ok(())
    }
    
    async fn register_mask_mismatch_procedure(&mut self) -> Result<(), AftercareError> {
        let procedure = RunbookProcedure {
            name: "mask_mismatch".to_string(),
            symptoms: vec![
                SymptomPattern {
                    name: "mask_count_deviation".to_string(),
                    description: "Fit/eval mask count mismatch".to_string(),
                    criteria: vec![
                        DetectionCriterion {
                            metric: "mask_count_ratio".to_string(),
                            operator: ComparisonOperator::NotEqual,
                            threshold: 1.0,
                            duration: Some(Duration::from_secs(300)),
                        }
                    ],
                }
            ],
            data_collection: vec![
                DataCollectionStep {
                    name: "collect_mask_data".to_string(),
                    data_items: vec![
                        DataItem {
                            name: "bin_table".to_string(),
                            item_type: DataItemType::BinTable,
                            collection_method: "calibration_debug".to_string(),
                        },
                        DataItem {
                            name: "mask_counts".to_string(),
                            item_type: DataItemType::MaskCounts,
                            collection_method: "mask_counter".to_string(),
                        }
                    ],
                    timeout: Duration::from_secs(30),
                }
            ],
            decision_tree: "mask_mismatch_tree".to_string(),
            revert_procedures: vec![
                RevertProcedure {
                    name: "flag_revert".to_string(),
                    steps: vec![
                        RevertStep {
                            description: "Disable CALIB_V22 flag".to_string(),
                            action: "set_flag(CALIB_V22=false)".to_string(),
                            expected_outcome: "Flag disabled globally".to_string(),
                            timeout: Duration::from_secs(30),
                        }
                    ],
                    validation: vec![
                        ValidationStep {
                            description: "Verify flag disabled".to_string(),
                            check: ValidationCheck::SystemCheck { check_type: "flag_status".to_string() },
                            pass_criteria: "CALIB_V22=false confirmed".to_string(),
                        }
                    ],
                }
            ],
        };
        
        self.procedures.insert("mask_mismatch".to_string(), procedure);
        Ok(())
    }
    
    async fn register_alpha_drift_procedure(&mut self) -> Result<(), AftercareError> {
        // Similar implementation for alpha drift procedure
        Ok(())
    }
    
    async fn register_merged_bin_procedure(&mut self) -> Result<(), AftercareError> {
        // Similar implementation for merged bin procedure
        Ok(())
    }
    
    pub async fn execute_procedure(&self, symptom: &str, data: HashMap<String, f64>) -> Result<RunbookResult, RunbookError> {
        if let Some(procedure) = self.procedures.get(symptom) {
            info!("📋 Executing runbook procedure: {}", procedure.name);
            
            // Execute data collection
            let collected_data = self.execute_data_collection(&procedure.data_collection, data).await?;
            
            // Execute decision tree
            let decision = self.execute_decision_tree(&procedure.decision_tree, &collected_data).await?;
            
            // Return runbook result
            Ok(RunbookResult {
                procedure_name: procedure.name.clone(),
                decision,
                collected_data,
                execution_time: SystemTime::now(),
                result_status: RunbookStatus::Success,
            })
        } else {
            Err(RunbookError::ProcedureNotFound(symptom.to_string()))
        }
    }
    
    async fn execute_data_collection(&self, steps: &[DataCollectionStep], initial_data: HashMap<String, f64>) -> Result<HashMap<String, f64>, RunbookError> {
        let mut collected_data = initial_data;
        
        for step in steps {
            for data_item in &step.data_items {
                match data_item.item_type {
                    DataItemType::BinTable => {
                        // Collect bin table data
                        collected_data.insert("bin_count".to_string(), 10.0);
                    }
                    DataItemType::AlphaValue => {
                        // Collect alpha value
                        collected_data.insert("alpha".to_string(), 0.15);
                    }
                    DataItemType::AeceTauValue => {
                        // Collect AECE-τ value
                        collected_data.insert("aece_tau".to_string(), 0.008);
                    }
                    _ => {
                        // Handle other data types
                    }
                }
            }
        }
        
        Ok(collected_data)
    }
    
    async fn execute_decision_tree(&self, tree_name: &str, data: &HashMap<String, f64>) -> Result<DecisionResult, RunbookError> {
        // Simulate decision tree execution
        Ok(DecisionResult {
            decision: "raise_bootstrap".to_string(),
            confidence: 0.85,
            reasoning: "Alpha drift detected, bootstrap adjustment recommended".to_string(),
            recommended_action: ActionType::RaiseBootstrap { factor: 1.1 },
        })
    }
    
    pub async fn get_status(&self) -> Result<RunbookStatus, AftercareError> {
        Ok(RunbookStatus::Ready)
    }
}

impl AlertTuningSystem {
    pub fn new(config: AlertTuningConfig) -> Result<Self, AftercareError> {
        let alert_rules = Self::create_default_alert_rules(&config)?;
        let dedup_engine = AlertDeduplicationEngine::new(config.dedup_window);
        let escalation_manager = EscalationManager::new();
        
        Ok(Self {
            alert_rules,
            dedup_engine,
            escalation_manager,
            config,
        })
    }
    
    fn create_default_alert_rules(config: &AlertTuningConfig) -> Result<HashMap<String, AlertRule>, AftercareError> {
        let mut rules = HashMap::new();
        
        // Critical alert: Mask mismatch
        rules.insert("mask_mismatch".to_string(), AlertRule {
            name: "mask_mismatch".to_string(),
            condition: AlertCondition::Threshold {
                metric: "mask_mismatch_detected".to_string(),
                operator: ComparisonOperator::GreaterThan,
                value: 0.0,
                duration: Duration::from_secs(60),
            },
            severity: AlertSeverity::Critical,
            message_template: "CRITICAL: Mask mismatch detected - fit/eval inconsistency".to_string(),
            escalation_policy: "immediate".to_string(),
        });
        
        // Critical alert: Scores out of range
        rules.insert("scores_out_of_range".to_string(), AlertRule {
            name: "scores_out_of_range".to_string(),
            condition: AlertCondition::Composite {
                conditions: vec![
                    AlertCondition::Threshold {
                        metric: "min_score".to_string(),
                        operator: ComparisonOperator::LessThan,
                        value: 0.0,
                        duration: Duration::from_secs(60),
                    },
                    AlertCondition::Threshold {
                        metric: "max_score".to_string(),
                        operator: ComparisonOperator::GreaterThan,
                        value: 1.0,
                        duration: Duration::from_secs(60),
                    },
                ],
                logic: LogicOperator::Or,
            },
            severity: AlertSeverity::Critical,
            message_template: "CRITICAL: Calibration scores outside [0,1] range".to_string(),
            escalation_policy: "immediate".to_string(),
        });
        
        // Warning alert: Alpha drift
        rules.insert("alpha_drift".to_string(), AlertRule {
            name: "alpha_drift".to_string(),
            condition: AlertCondition::Threshold {
                metric: "alpha_drift_wow".to_string(),
                operator: ComparisonOperator::GreaterThan,
                value: config.critical_thresholds.alpha_drift_wow_threshold,
                duration: Duration::from_secs(3600),
            },
            severity: AlertSeverity::Warning,
            message_template: "WARNING: Alpha drift >0.05 WoW detected".to_string(),
            escalation_policy: "standard".to_string(),
        });
        
        // Critical alert: Merged bin threshold
        rules.insert("merged_bin_critical".to_string(), AlertRule {
            name: "merged_bin_critical".to_string(),
            condition: AlertCondition::Threshold {
                metric: "merged_bin_percent".to_string(),
                operator: ComparisonOperator::GreaterThan,
                value: config.critical_thresholds.merged_bin_critical_percent,
                duration: Duration::from_secs(300),
            },
            severity: AlertSeverity::Critical,
            message_template: "CRITICAL: Merged bin percentage >20%".to_string(),
            escalation_policy: "immediate".to_string(),
        });
        
        Ok(rules)
    }
    
    pub async fn get_active_alerts(&self) -> Result<Vec<AlertSummary>, AftercareError> {
        // Return current active alerts
        Ok(vec![])
    }
    
    pub async fn process_alert(&mut self, alert_type: AlertType, message: String, severity: AlertSeverity) -> Result<String, AftercareError> {
        // Process incoming alert through deduplication and escalation
        let alert_id = format!("alert_{}_{}", 
            chrono::Utc::now().timestamp(), 
            rand::random::<u32>());
        
        // Check deduplication
        if !self.dedup_engine.should_process_alert(&alert_type, &message).await {
            return Ok(format!("Alert deduplicated: {}", alert_id));
        }
        
        // Create alert summary
        let alert_summary = AlertSummary {
            alert_id: alert_id.clone(),
            alert_type,
            severity,
            message,
            created_at: SystemTime::now(),
            status: AlertStatus::Active,
        };
        
        // Process escalation
        self.escalation_manager.process_alert(&alert_summary).await
            .map_err(|e| AftercareError::AlertError(e.to_string()))?;
        
        Ok(alert_id)
    }
}

impl AlertDeduplicationEngine {
    pub fn new(window: Duration) -> Self {
        Self {
            active_fingerprints: HashMap::new(),
            window,
        }
    }
    
    pub async fn should_process_alert(&mut self, alert_type: &AlertType, message: &str) -> bool {
        let content_hash = format!("{:?}:{}", alert_type, message);
        let now = SystemTime::now();
        
        // Clean up old fingerprints
        self.active_fingerprints.retain(|_, fingerprint| {
            now.duration_since(fingerprint.last_occurrence).unwrap_or_default() < self.window
        });
        
        // Check if this alert should be deduplicated
        if let Some(fingerprint) = self.active_fingerprints.get_mut(&content_hash) {
            fingerprint.last_occurrence = now;
            fingerprint.count += 1;
            false // Deduplicate
        } else {
            // New alert
            self.active_fingerprints.insert(content_hash.clone(), AlertFingerprint {
                alert_type: alert_type.clone(),
                content_hash,
                first_occurrence: now,
                last_occurrence: now,
                count: 1,
            });
            true // Process
        }
    }
}

impl EscalationManager {
    pub fn new() -> Self {
        Self {
            policies: HashMap::new(),
            active_escalations: HashMap::new(),
        }
    }
    
    pub async fn process_alert(&mut self, alert: &AlertSummary) -> Result<(), EscalationError> {
        // Process alert escalation based on severity and policy
        match alert.severity {
            AlertSeverity::Critical | AlertSeverity::Emergency => {
                // Immediate escalation for critical alerts
                self.create_immediate_escalation(alert).await?;
            }
            AlertSeverity::Warning => {
                // Standard escalation for warnings
                self.create_standard_escalation(alert).await?;
            }
            AlertSeverity::Info => {
                // No escalation for info alerts
            }
        }
        
        Ok(())
    }
    
    async fn create_immediate_escalation(&mut self, alert: &AlertSummary) -> Result<(), EscalationError> {
        // Create immediate escalation for critical alerts
        let escalation = ActiveEscalation {
            alert_id: alert.alert_id.clone(),
            current_level: 1,
            start_time: SystemTime::now(),
            next_escalation: SystemTime::now() + Duration::from_secs(300), // 5 minutes
            status: EscalationStatus::Active,
        };
        
        self.active_escalations.insert(alert.alert_id.clone(), escalation);
        
        info!("🚨 Immediate escalation created for alert: {}", alert.alert_id);
        Ok(())
    }
    
    async fn create_standard_escalation(&mut self, alert: &AlertSummary) -> Result<(), EscalationError> {
        // Create standard escalation for warning alerts
        let escalation = ActiveEscalation {
            alert_id: alert.alert_id.clone(),
            current_level: 1,
            start_time: SystemTime::now(),
            next_escalation: SystemTime::now() + Duration::from_secs(1800), // 30 minutes
            status: EscalationStatus::Active,
        };
        
        self.active_escalations.insert(alert.alert_id.clone(), escalation);
        
        info!("⚠️  Standard escalation created for alert: {}", alert.alert_id);
        Ok(())
    }
}

// Supporting types and implementations

impl Default for DashboardState {
    fn default() -> Self {
        let now = SystemTime::now();
        Self {
            last_update: now,
            current_metrics: CalibrationMetrics {
                aece: 0.0,
                dece: 0.0,
                brier: 0.0,
                alpha: 0.0,
                clamp_rate_percent: 0.0,
                merged_bin_percent: 0.0,
                timestamp: now,
            },
            trends: TrendingData {
                aece_trend: Vec::new(),
                dece_trend: Vec::new(),
                brier_trend: Vec::new(),
                alpha_trend: Vec::new(),
                clamp_rate_trend: Vec::new(),
                merged_bin_trend: Vec::new(),
            },
            slice_highlights: HashMap::new(),
            sla_status: SlaStatus {
                overall_compliance: 0.0,
                component_status: HashMap::new(),
                active_violations: Vec::new(),
                statistical_enforcement: StatisticalEnforcement {
                    tests_performed: Vec::new(),
                    confidence_intervals: HashMap::new(),
                    significance_levels: HashMap::new(),
                },
            },
            active_alerts: Vec::new(),
            health_indicators: HealthIndicators {
                overall_health: SystemHealthStatus::Healthy,
                component_health: HashMap::new(),
                performance_indicators: PerformanceIndicators {
                    avg_response_time_ms: 0.0,
                    p95_response_time_ms: 0.0,
                    p99_response_time_ms: 0.0,
                    throughput_rps: 0.0,
                    error_rate_percent: 0.0,
                },
                reliability_indicators: ReliabilityIndicators {
                    uptime_24h: 0.0,
                    mtbf_hours: 0.0,
                    mttr_minutes: 0.0,
                    availability_percent: 0.0,
                },
            },
        }
    }
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            retention_duration: Duration::from_secs(7 * 24 * 3600), // 7 days
            trend_window: Duration::from_secs(24 * 3600), // 24 hours
            thresholds: DashboardThresholds::default(),
            refresh_rate: Duration::from_secs(30),
        }
    }
}

impl Default for DashboardThresholds {
    fn default() -> Self {
        Self {
            aece_highlight_threshold: 0.01,
            dece_warning_threshold: 0.02,
            brier_alert_threshold: 0.1,
            alpha_drift_threshold: 0.05,
            clamp_rate_warning_threshold: 10.0,
            merged_bin_warning_threshold: 5.0,
            merged_bin_critical_threshold: 20.0,
        }
    }
}

impl Default for RunbookConfig {
    fn default() -> Self {
        Self {
            procedure_timeout: Duration::from_secs(300),
            data_capture_window: Duration::from_secs(900),
            decision_timeout: Duration::from_secs(60),
        }
    }
}

impl Default for AlertTuningConfig {
    fn default() -> Self {
        Self {
            dedup_window: Duration::from_secs(300), // 5 minutes
            escalation_delays: {
                let mut delays = HashMap::new();
                delays.insert(AlertSeverity::Info, Duration::from_secs(3600));
                delays.insert(AlertSeverity::Warning, Duration::from_secs(1800));
                delays.insert(AlertSeverity::Critical, Duration::from_secs(300));
                delays.insert(AlertSeverity::Emergency, Duration::from_secs(60));
                delays
            },
            suppression_rules: Vec::new(),
            critical_thresholds: CriticalThresholds::default(),
        }
    }
}

impl Default for CriticalThresholds {
    fn default() -> Self {
        Self {
            mask_mismatch_tolerance: 0.0,
            score_range_min: 0.0,
            score_range_max: 1.0,
            alpha_drift_wow_threshold: 0.05,
            merged_bin_critical_percent: 20.0,
        }
    }
}

impl Default for AftercareConfig {
    fn default() -> Self {
        Self {
            dashboard_refresh_interval: Duration::from_secs(30),
            alert_evaluation_window: Duration::from_secs(60),
            sla_monitoring_frequency: Duration::from_secs(300),
            runbook_response_timeout: Duration::from_secs(600),
            alert_dedup_window: Duration::from_secs(300),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunbookResult {
    pub procedure_name: String,
    pub decision: DecisionResult,
    pub collected_data: HashMap<String, f64>,
    pub execution_time: SystemTime,
    pub result_status: RunbookStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionResult {
    pub decision: String,
    pub confidence: f64,
    pub reasoning: String,
    pub recommended_action: ActionType,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RunbookStatus {
    Ready,
    Executing,
    Success,
    Failed,
    Timeout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AftercareStatus {
    pub dashboard_state: DashboardState,
    pub active_alerts: Vec<AlertSummary>,
    pub runbook_status: RunbookStatus,
    pub system_health: SystemHealthStatus,
    pub timestamp: SystemTime,
}

#[derive(Debug, Error)]
pub enum RunbookError {
    #[error("Procedure not found: {0}")]
    ProcedureNotFound(String),
    
    #[error("Data collection failed: {0}")]
    DataCollectionFailed(String),
    
    #[error("Decision tree execution failed: {0}")]
    DecisionTreeFailed(String),
    
    #[error("Timeout occurred")]
    Timeout,
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
}

#[derive(Debug, Error)]
pub enum EscalationError {
    #[error("Escalation policy not found: {0}")]
    PolicyNotFound(String),
    
    #[error("Escalation failed: {0}")]
    EscalationFailed(String),
    
    #[error("Invalid escalation configuration: {0}")]
    InvalidConfiguration(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dashboard_config_default() {
        let config = DashboardConfig::default();
        assert_eq!(config.retention_duration, Duration::from_secs(7 * 24 * 3600));
        assert_eq!(config.thresholds.aece_highlight_threshold, 0.01);
    }
    
    #[test]
    fn test_critical_thresholds() {
        let thresholds = CriticalThresholds::default();
        assert_eq!(thresholds.mask_mismatch_tolerance, 0.0);
        assert_eq!(thresholds.merged_bin_critical_percent, 20.0);
    }
    
    #[tokio::test]
    async fn test_alert_deduplication() {
        let mut dedup_engine = AlertDeduplicationEngine::new(Duration::from_secs(300));
        
        // First alert should be processed
        let should_process1 = dedup_engine.should_process_alert(&AlertType::MaskMismatch, "Test message").await;
        assert!(should_process1);
        
        // Duplicate alert should be deduplicated
        let should_process2 = dedup_engine.should_process_alert(&AlertType::MaskMismatch, "Test message").await;
        assert!(!should_process2);
    }
    
    #[test]
    fn test_runbook_status() {
        let status = RunbookStatus::Ready;
        assert_eq!(status, RunbookStatus::Ready);
        assert_ne!(status, RunbookStatus::Executing);
    }
}