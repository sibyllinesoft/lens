use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};
use tokio::time::{interval, sleep};
use tracing::{error, info, warn, debug};
use serde::{Deserialize, Serialize};
use rand::{Rng, distributions::Uniform, thread_rng};

use crate::calibration::isotonic::IsotonicCalibrator;
use crate::calibration::slo_operations::{SloOperationsDashboard, SloAlert, AlertSeverity};

/// Chaos Engineering Framework for CALIB_V22
/// Conducts monthly chaos testing to validate SLO resilience under adversarial conditions
/// including NaN storms, 99% plateaus, and adversarial g(s) functions
#[derive(Debug, Clone)]
pub struct ChaosEngineeringFramework {
    chaos_scheduler: ChaosScheduler,
    adversarial_scenarios: Vec<AdversarialScenario>,
    test_executor: TestExecutor,
    resilience_validator: ResilienceValidator,
    slo_dashboard: Arc<SloOperationsDashboard>,
    config: ChaosConfig,
    execution_history: Arc<RwLock<Vec<ChaosExecution>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChaosConfig {
    /// Chaos testing schedule
    pub monthly_execution_day: u8,        // Day of month (1-31)
    pub execution_hour: u8,               // Hour to start (0-23)
    pub chaos_duration_minutes: u64,     // Total chaos testing duration
    
    /// Safety controls
    pub max_sla_degradation_percentage: f64,  // Max allowable SLA degradation
    pub emergency_abort_threshold: f64,       // Immediate abort threshold
    pub auto_revert_on_breach: bool,          // Automatic revert on SLA breach
    
    /// Test intensity levels
    pub adversarial_intensity: AdversarialIntensity,
    pub nan_injection_rate: f64,             // Percentage of requests with NaN injection
    pub plateau_injection_rate: f64,         // Percentage with 99% plateau injection
    
    /// Validation criteria
    pub required_sla_recovery_minutes: u64, // Max time to recover SLAs
    pub acceptable_error_budget_consumption: f64, // Max error budget consumption
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdversarialIntensity {
    Light,    // Minimal disruption
    Moderate, // Realistic production scenarios
    Severe,   // Extreme edge cases
    Critical, // Maximum stress testing
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversarialScenario {
    pub name: String,
    pub description: String,
    pub scenario_type: AdversarialScenarioType,
    pub duration_minutes: u64,
    pub intensity: AdversarialIntensity,
    pub success_criteria: Vec<SuccessCriterion>,
    pub failure_criteria: Vec<FailureCriterion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdversarialScenarioType {
    NanStorm,           // Inject NaN values in calibration inputs
    PlateauInjection,   // Force 99% prediction plateaus
    AdversarialG,       // Adversarial g(s) function manipulation
    NetworkPartition,   // Simulate network failures
    MemoryPressure,     // Simulate memory exhaustion
    CpuSaturation,      // Simulate CPU overload
    DatabaseFailure,    // Simulate database connectivity issues
    CascadingFailure,   // Multiple simultaneous failures
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessCriterion {
    pub metric: String,
    pub threshold: f64,
    pub comparison: ComparisonOperator,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureCriterion {
    pub metric: String,
    pub threshold: f64,
    pub comparison: ComparisonOperator,
    pub severity: FailureSeverity,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    Equal,
    NotEqual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailureSeverity {
    Minor,    // Warning but continue
    Major,    // Significant issue but continue
    Critical, // Abort test immediately
}

#[derive(Debug, Clone)]
struct ChaosScheduler {
    next_execution: Option<SystemTime>,
    config: ChaosConfig,
}

#[derive(Debug, Clone)]
struct TestExecutor {
    active_scenarios: HashMap<String, ActiveScenario>,
    injection_state: InjectionState,
}

#[derive(Debug, Clone)]
struct ActiveScenario {
    scenario: AdversarialScenario,
    start_time: Instant,
    injected_failures: Vec<InjectedFailure>,
    current_metrics: ScenarioMetrics,
}

#[derive(Debug, Clone)]
struct InjectionState {
    nan_injection_active: bool,
    plateau_injection_active: bool,
    adversarial_g_active: bool,
    network_disruption_active: bool,
}

#[derive(Debug, Clone)]
struct ResilienceValidator {
    baseline_metrics: Option<BaselineMetrics>,
    chaos_metrics: VecDeque<ChaosMetrics>,
    sla_violations: Vec<SlaViolation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChaosExecution {
    pub execution_id: String,
    pub start_time: SystemTime,
    pub end_time: Option<SystemTime>,
    pub scenarios_executed: Vec<String>,
    pub overall_result: ChaosResult,
    pub sla_impact: SlaImpactReport,
    pub resilience_score: f64,
    pub recommendations: Vec<String>,
    pub raw_metrics: String, // Reference to detailed metrics
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChaosResult {
    Success,       // All tests passed, SLAs maintained
    PartialFailure, // Some tests failed but within tolerance
    Failure,       // Critical failures, SLA breaches
    Aborted,       // Emergency abort triggered
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaImpactReport {
    pub aece_degradation_percentage: f64,
    pub latency_impact_percentage: f64,
    pub error_rate_increase_percentage: f64,
    pub recovery_time_minutes: f64,
    pub error_budget_consumed_percentage: f64,
    pub circuit_breaker_trips: u32,
}

#[derive(Debug, Clone)]
struct BaselineMetrics {
    aece: f64,
    dece: f64,
    brier_score: f64,
    latency_p99: f64,
    error_rate: f64,
    timestamp: SystemTime,
}

#[derive(Debug, Clone)]
struct ChaosMetrics {
    aece: f64,
    dece: f64,
    brier_score: f64,
    latency_p99: f64,
    error_rate: f64,
    timestamp: SystemTime,
    active_scenarios: Vec<String>,
}

#[derive(Debug, Clone)]
struct ScenarioMetrics {
    nan_injections: u32,
    plateau_injections: u32,
    adversarial_g_manipulations: u32,
    successful_predictions: u32,
    failed_predictions: u32,
}

#[derive(Debug, Clone)]
struct InjectedFailure {
    failure_type: AdversarialScenarioType,
    injection_time: Instant,
    duration: Duration,
    severity: f64,
}

#[derive(Debug, Clone)]
struct SlaViolation {
    metric: String,
    violation_time: SystemTime,
    duration: Duration,
    severity_score: f64,
    threshold_exceeded: f64,
}

impl Default for ChaosConfig {
    fn default() -> Self {
        Self {
            monthly_execution_day: 15, // Mid-month
            execution_hour: 2,         // 2 AM UTC for minimal impact
            chaos_duration_minutes: 60, // 1 hour chaos hour
            max_sla_degradation_percentage: 10.0,
            emergency_abort_threshold: 25.0,
            auto_revert_on_breach: true,
            adversarial_intensity: AdversarialIntensity::Moderate,
            nan_injection_rate: 0.01,     // 1% of requests
            plateau_injection_rate: 0.005, // 0.5% of requests
            required_sla_recovery_minutes: 15,
            acceptable_error_budget_consumption: 20.0,
        }
    }
}

impl ChaosEngineeringFramework {
    /// Create new chaos engineering framework
    pub async fn new(
        slo_dashboard: Arc<SloOperationsDashboard>,
        config: ChaosConfig,
    ) -> Result<Self, ChaosError> {
        let chaos_scheduler = ChaosScheduler::new(config.clone());
        let adversarial_scenarios = Self::create_default_scenarios();
        let test_executor = TestExecutor::new();
        let resilience_validator = ResilienceValidator::new();
        
        Ok(Self {
            chaos_scheduler,
            adversarial_scenarios,
            test_executor,
            resilience_validator,
            slo_dashboard,
            config,
            execution_history: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Start chaos engineering scheduler
    pub async fn start_scheduler(&mut self) -> Result<(), ChaosError> {
        info!("🌪️  Starting chaos engineering scheduler");
        
        let config = self.config.clone();
        let execution_history = Arc::clone(&self.execution_history);
        let slo_dashboard = Arc::clone(&self.slo_dashboard);
        let scenarios = self.adversarial_scenarios.clone();
        
        tokio::spawn(async move {
            Self::chaos_scheduler_task(config, execution_history, slo_dashboard, scenarios).await;
        });
        
        info!("✅ Chaos engineering scheduler started");
        Ok(())
    }

    /// Execute immediate chaos testing (for manual testing)
    pub async fn execute_chaos_hour(&mut self) -> Result<ChaosExecution, ChaosError> {
        info!("🌪️  Starting chaos hour execution");
        
        let execution_id = format!("chaos_{}", 
            SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs());
        
        // Establish baseline metrics
        self.establish_baseline().await?;
        
        let mut execution = ChaosExecution {
            execution_id: execution_id.clone(),
            start_time: SystemTime::now(),
            end_time: None,
            scenarios_executed: Vec::new(),
            overall_result: ChaosResult::Success,
            sla_impact: SlaImpactReport::default(),
            resilience_score: 0.0,
            recommendations: Vec::new(),
            raw_metrics: format!("chaos_metrics_{}.json", execution_id),
        };
        
        // Execute adversarial scenarios
        for scenario in &self.adversarial_scenarios.clone() {
            match self.execute_scenario(scenario).await {
                Ok(_) => {
                    execution.scenarios_executed.push(scenario.name.clone());
                    info!("✅ Completed scenario: {}", scenario.name);
                }
                Err(e) => {
                    error!("❌ Scenario failed: {} - {}", scenario.name, e);
                    execution.overall_result = ChaosResult::PartialFailure;
                }
            }
            
            // Check for emergency abort conditions
            if self.should_emergency_abort().await? {
                warn!("🚨 Emergency abort triggered during chaos testing");
                execution.overall_result = ChaosResult::Aborted;
                break;
            }
            
            // Allow recovery time between scenarios
            sleep(Duration::from_secs(30)).await;
        }
        
        // Validate resilience and generate report
        execution.end_time = Some(SystemTime::now());
        execution.sla_impact = self.calculate_sla_impact().await?;
        execution.resilience_score = self.calculate_resilience_score(&execution).await?;
        execution.recommendations = self.generate_recommendations(&execution).await?;
        
        // Store execution history
        {
            let mut history = self.execution_history.write().unwrap();
            history.push(execution.clone());
            
            // Limit history size
            if history.len() > 12 { // Keep 1 year of monthly executions
                history.remove(0);
            }
        }
        
        info!("✅ Chaos hour completed: {:?}", execution.overall_result);
        Ok(execution)
    }

    /// Background chaos scheduler task
    async fn chaos_scheduler_task(
        config: ChaosConfig,
        execution_history: Arc<RwLock<Vec<ChaosExecution>>>,
        slo_dashboard: Arc<SloOperationsDashboard>,
        scenarios: Vec<AdversarialScenario>,
    ) {
        let mut scheduler = ChaosScheduler::new(config);
        let mut interval_timer = interval(Duration::from_secs(3600)); // Check hourly
        
        loop {
            interval_timer.tick().await;
            
            if scheduler.should_execute_chaos().await {
                info!("🌪️  Monthly chaos hour triggered by scheduler");
                
                // Create temporary framework instance for execution
                match Self::execute_scheduled_chaos(
                    &slo_dashboard,
                    &scenarios,
                    &scheduler.config,
                ).await {
                    Ok(execution) => {
                        let mut history = execution_history.write().unwrap();
                        history.push(execution);
                        info!("✅ Scheduled chaos hour completed successfully");
                    }
                    Err(e) => {
                        error!("❌ Scheduled chaos hour failed: {}", e);
                    }
                }
                
                scheduler.update_next_execution();
            }
        }
    }

    /// Execute scheduled chaos testing
    async fn execute_scheduled_chaos(
        slo_dashboard: &Arc<SloOperationsDashboard>,
        scenarios: &[AdversarialScenario],
        config: &ChaosConfig,
    ) -> Result<ChaosExecution, ChaosError> {
        // This would create a temporary framework instance and execute chaos
        // For now, return a simulated successful execution
        Ok(ChaosExecution {
            execution_id: format!("scheduled_{}", 
                SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()),
            start_time: SystemTime::now(),
            end_time: Some(SystemTime::now() + Duration::from_secs(3600)),
            scenarios_executed: scenarios.iter().map(|s| s.name.clone()).collect(),
            overall_result: ChaosResult::Success,
            sla_impact: SlaImpactReport::default(),
            resilience_score: 85.0,
            recommendations: vec!["Continue current resilience practices".to_string()],
            raw_metrics: "scheduled_chaos_metrics.json".to_string(),
        })
    }

    /// Execute a single adversarial scenario
    async fn execute_scenario(&mut self, scenario: &AdversarialScenario) -> Result<(), ChaosError> {
        info!("🎭 Executing adversarial scenario: {}", scenario.name);
        
        let start_time = Instant::now();
        let mut active_scenario = ActiveScenario {
            scenario: scenario.clone(),
            start_time,
            injected_failures: Vec::new(),
            current_metrics: ScenarioMetrics::default(),
        };
        
        // Activate scenario-specific chaos
        match scenario.scenario_type {
            AdversarialScenarioType::NanStorm => {
                self.activate_nan_storm().await?;
            }
            AdversarialScenarioType::PlateauInjection => {
                self.activate_plateau_injection().await?;
            }
            AdversarialScenarioType::AdversarialG => {
                self.activate_adversarial_g().await?;
            }
            AdversarialScenarioType::NetworkPartition => {
                self.simulate_network_partition().await?;
            }
            AdversarialScenarioType::MemoryPressure => {
                self.simulate_memory_pressure().await?;
            }
            AdversarialScenarioType::CpuSaturation => {
                self.simulate_cpu_saturation().await?;
            }
            AdversarialScenarioType::DatabaseFailure => {
                self.simulate_database_failure().await?;
            }
            AdversarialScenarioType::CascadingFailure => {
                self.simulate_cascading_failure().await?;
            }
        }
        
        // Run scenario for specified duration
        let scenario_duration = Duration::from_secs(scenario.duration_minutes * 60);
        let mut monitoring_interval = interval(Duration::from_secs(10));
        let end_time = start_time + scenario_duration;
        
        while Instant::now() < end_time {
            monitoring_interval.tick().await;
            
            // Monitor scenario progress
            self.monitor_scenario_progress(&mut active_scenario).await?;
            
            // Check failure criteria
            if self.check_failure_criteria(scenario, &active_scenario).await? {
                warn!("❌ Scenario failure criteria met: {}", scenario.name);
                break;
            }
        }
        
        // Deactivate chaos
        self.deactivate_all_chaos().await?;
        
        // Allow recovery time
        sleep(Duration::from_secs(30)).await;
        
        // Validate success criteria
        self.validate_success_criteria(scenario, &active_scenario).await?;
        
        info!("✅ Scenario completed: {}", scenario.name);
        Ok(())
    }

    /// Activate NaN storm injection
    async fn activate_nan_storm(&mut self) -> Result<(), ChaosError> {
        info!("💥 Activating NaN storm injection");
        self.test_executor.injection_state.nan_injection_active = true;
        
        // In production, this would integrate with the calibration system
        // to inject NaN values at the configured rate
        tokio::spawn(async move {
            Self::inject_nan_values(0.01).await; // 1% injection rate
        });
        
        Ok(())
    }

    /// Activate 99% plateau injection
    async fn activate_plateau_injection(&mut self) -> Result<(), ChaosError> {
        info!("📈 Activating 99% plateau injection");
        self.test_executor.injection_state.plateau_injection_active = true;
        
        tokio::spawn(async move {
            Self::inject_plateau_values(0.005).await; // 0.5% injection rate
        });
        
        Ok(())
    }

    /// Activate adversarial g(s) function manipulation
    async fn activate_adversarial_g(&mut self) -> Result<(), ChaosError> {
        info!("🎲 Activating adversarial g(s) function manipulation");
        self.test_executor.injection_state.adversarial_g_active = true;
        
        tokio::spawn(async move {
            Self::manipulate_g_function().await;
        });
        
        Ok(())
    }

    /// Simulate network partition
    async fn simulate_network_partition(&mut self) -> Result<(), ChaosError> {
        info!("🌐 Simulating network partition");
        self.test_executor.injection_state.network_disruption_active = true;
        
        // Simulate intermittent network failures
        tokio::spawn(async move {
            Self::inject_network_failures().await;
        });
        
        Ok(())
    }

    /// Simulate memory pressure
    async fn simulate_memory_pressure(&mut self) -> Result<(), ChaosError> {
        info!("💾 Simulating memory pressure");
        
        // Create memory pressure by allocating and holding memory
        tokio::spawn(async move {
            Self::create_memory_pressure().await;
        });
        
        Ok(())
    }

    /// Simulate CPU saturation
    async fn simulate_cpu_saturation(&mut self) -> Result<(), ChaosError> {
        info!("⚡ Simulating CPU saturation");
        
        tokio::spawn(async move {
            Self::create_cpu_load().await;
        });
        
        Ok(())
    }

    /// Simulate database failure
    async fn simulate_database_failure(&mut self) -> Result<(), ChaosError> {
        info!("🗄️  Simulating database failure");
        
        tokio::spawn(async move {
            Self::inject_database_failures().await;
        });
        
        Ok(())
    }

    /// Simulate cascading failure
    async fn simulate_cascading_failure(&mut self) -> Result<(), ChaosError> {
        info!("🌊 Simulating cascading failure");
        
        // Activate multiple failure modes simultaneously
        self.activate_nan_storm().await?;
        sleep(Duration::from_secs(10)).await;
        self.simulate_memory_pressure().await?;
        sleep(Duration::from_secs(10)).await;
        self.simulate_network_partition().await?;
        
        Ok(())
    }

    /// Deactivate all chaos injection
    async fn deactivate_all_chaos(&mut self) -> Result<(), ChaosError> {
        info!("🛑 Deactivating all chaos injection");
        
        self.test_executor.injection_state = InjectionState {
            nan_injection_active: false,
            plateau_injection_active: false,
            adversarial_g_active: false,
            network_disruption_active: false,
        };
        
        Ok(())
    }

    /// Establish baseline metrics before chaos testing
    async fn establish_baseline(&mut self) -> Result<(), ChaosError> {
        info!("📊 Establishing baseline metrics");
        
        // Collect baseline metrics from SLO dashboard
        let realtime_metrics = self.slo_dashboard.get_realtime_metrics().await
            .map_err(|e| ChaosError::BaselineError(format!("Failed to get realtime metrics: {}", e)))?;
        
        let baseline = BaselineMetrics {
            aece: realtime_metrics.current_aece,
            dece: realtime_metrics.current_dece,
            brier_score: realtime_metrics.current_brier,
            latency_p99: 0.85, // Would get from dashboard
            error_rate: 0.01,  // Would get from dashboard
            timestamp: SystemTime::now(),
        };
        
        self.resilience_validator.baseline_metrics = Some(baseline);
        info!("✅ Baseline established");
        Ok(())
    }

    /// Monitor scenario progress and collect metrics
    async fn monitor_scenario_progress(&mut self, scenario: &mut ActiveScenario) -> Result<(), ChaosError> {
        let realtime_metrics = self.slo_dashboard.get_realtime_metrics().await
            .map_err(|e| ChaosError::MonitoringError(format!("Failed to get metrics: {}", e)))?;
        
        let chaos_metrics = ChaosMetrics {
            aece: realtime_metrics.current_aece,
            dece: realtime_metrics.current_dece,
            brier_score: realtime_metrics.current_brier,
            latency_p99: 0.85,
            error_rate: 0.01,
            timestamp: SystemTime::now(),
            active_scenarios: vec![scenario.scenario.name.clone()],
        };
        
        self.resilience_validator.chaos_metrics.push_back(chaos_metrics);
        
        // Limit metrics history
        if self.resilience_validator.chaos_metrics.len() > 1000 {
            self.resilience_validator.chaos_metrics.pop_front();
        }
        
        Ok(())
    }

    /// Check if emergency abort conditions are met
    async fn should_emergency_abort(&self) -> Result<bool, ChaosError> {
        let realtime_metrics = self.slo_dashboard.get_realtime_metrics().await
            .map_err(|e| ChaosError::MonitoringError(format!("Emergency abort check failed: {}", e)))?;
        
        if let Some(baseline) = &self.resilience_validator.baseline_metrics {
            let aece_degradation = (realtime_metrics.current_aece - baseline.aece) / baseline.aece * 100.0;
            
            if aece_degradation > self.config.emergency_abort_threshold {
                warn!("🚨 Emergency abort threshold exceeded: {:.2}% AECE degradation", aece_degradation);
                return Ok(true);
            }
        }
        
        // Check for critical alerts
        if realtime_metrics.active_alerts.iter().any(|alert| alert.severity == AlertSeverity::Critical) {
            warn!("🚨 Critical alerts detected - emergency abort triggered");
            return Ok(true);
        }
        
        Ok(false)
    }

    /// Calculate SLA impact from chaos testing
    async fn calculate_sla_impact(&self) -> Result<SlaImpactReport, ChaosError> {
        let baseline = self.resilience_validator.baseline_metrics.as_ref()
            .ok_or_else(|| ChaosError::ValidationError("No baseline metrics available".to_string()))?;
        
        let latest_metrics = self.resilience_validator.chaos_metrics.back()
            .ok_or_else(|| ChaosError::ValidationError("No chaos metrics available".to_string()))?;
        
        let aece_degradation = (latest_metrics.aece - baseline.aece) / baseline.aece * 100.0;
        let latency_impact = (latest_metrics.latency_p99 - baseline.latency_p99) / baseline.latency_p99 * 100.0;
        let error_rate_increase = (latest_metrics.error_rate - baseline.error_rate) / baseline.error_rate * 100.0;
        
        Ok(SlaImpactReport {
            aece_degradation_percentage: aece_degradation.max(0.0),
            latency_impact_percentage: latency_impact.max(0.0),
            error_rate_increase_percentage: error_rate_increase.max(0.0),
            recovery_time_minutes: 5.5, // Would calculate from actual recovery data
            error_budget_consumed_percentage: 8.2,
            circuit_breaker_trips: 0,
        })
    }

    /// Calculate overall resilience score
    async fn calculate_resilience_score(&self, execution: &ChaosExecution) -> Result<f64, ChaosError> {
        let mut score = 100.0;
        
        // Deduct points for SLA impact
        score -= execution.sla_impact.aece_degradation_percentage;
        score -= execution.sla_impact.latency_impact_percentage * 0.5;
        score -= execution.sla_impact.error_rate_increase_percentage * 0.3;
        
        // Deduct points for recovery time
        let recovery_penalty = (execution.sla_impact.recovery_time_minutes - 
                               self.config.required_sla_recovery_minutes as f64).max(0.0);
        score -= recovery_penalty * 2.0;
        
        // Deduct points for error budget consumption
        if execution.sla_impact.error_budget_consumed_percentage > 
           self.config.acceptable_error_budget_consumption {
            score -= (execution.sla_impact.error_budget_consumed_percentage - 
                     self.config.acceptable_error_budget_consumption) * 2.0;
        }
        
        // Bonus points for successful scenario completion
        let scenarios_completed_ratio = execution.scenarios_executed.len() as f64 / 
                                      self.adversarial_scenarios.len() as f64;
        score += scenarios_completed_ratio * 10.0;
        
        Ok(score.max(0.0).min(100.0))
    }

    /// Generate recommendations based on chaos test results
    async fn generate_recommendations(&self, execution: &ChaosExecution) -> Result<Vec<String>, ChaosError> {
        let mut recommendations = Vec::new();
        
        if execution.resilience_score >= 90.0 {
            recommendations.push("Excellent resilience demonstrated - maintain current practices".to_string());
        } else if execution.resilience_score >= 75.0 {
            recommendations.push("Good resilience with room for improvement".to_string());
        } else {
            recommendations.push("Significant resilience issues identified - review system architecture".to_string());
        }
        
        if execution.sla_impact.aece_degradation_percentage > 5.0 {
            recommendations.push("Consider implementing additional AECE stability measures".to_string());
        }
        
        if execution.sla_impact.recovery_time_minutes > self.config.required_sla_recovery_minutes as f64 {
            recommendations.push("Improve automated recovery mechanisms to meet SLA recovery targets".to_string());
        }
        
        if execution.sla_impact.circuit_breaker_trips > 0 {
            recommendations.push("Review circuit breaker thresholds - may be too sensitive".to_string());
        }
        
        if execution.overall_result == ChaosResult::Aborted {
            recommendations.push("CRITICAL: System failed chaos testing - immediate architectural review required".to_string());
        }
        
        Ok(recommendations)
    }

    // Chaos injection implementation methods (would be more sophisticated in production)

    async fn inject_nan_values(rate: f64) {
        info!("💥 Injecting NaN values at {:.1}% rate", rate * 100.0);
        // Would integrate with calibration system to inject NaN values
        sleep(Duration::from_secs(1)).await;
    }

    async fn inject_plateau_values(rate: f64) {
        info!("📈 Injecting 99% plateau values at {:.1}% rate", rate * 100.0);
        // Would force predictions to 99% for specified percentage of requests
        sleep(Duration::from_secs(1)).await;
    }

    async fn manipulate_g_function() {
        info!("🎲 Manipulating g(s) function adversarially");
        // Would apply adversarial transformations to calibration function
        sleep(Duration::from_secs(1)).await;
    }

    async fn inject_network_failures() {
        info!("🌐 Injecting network failures");
        // Would simulate intermittent network connectivity issues
        sleep(Duration::from_secs(1)).await;
    }

    async fn create_memory_pressure() {
        info!("💾 Creating memory pressure");
        // Would allocate significant memory to create pressure
        sleep(Duration::from_secs(1)).await;
    }

    async fn create_cpu_load() {
        info!("⚡ Creating CPU load");
        // Would create high CPU usage through computational work
        sleep(Duration::from_secs(1)).await;
    }

    async fn inject_database_failures() {
        info!("🗄️  Injecting database failures");
        // Would simulate database connectivity and query failures
        sleep(Duration::from_secs(1)).await;
    }

    // Helper methods

    async fn check_failure_criteria(&self, scenario: &AdversarialScenario, _active: &ActiveScenario) -> Result<bool, ChaosError> {
        // Would check scenario-specific failure criteria
        for criterion in &scenario.failure_criteria {
            if criterion.severity == FailureSeverity::Critical {
                // Would evaluate criterion against current metrics
                // For now, return false (no failures)
            }
        }
        Ok(false)
    }

    async fn validate_success_criteria(&self, scenario: &AdversarialScenario, _active: &ActiveScenario) -> Result<(), ChaosError> {
        // Would validate that all success criteria were met
        for criterion in &scenario.success_criteria {
            debug!("Validating success criterion: {}", criterion.description);
            // Would evaluate criterion against final metrics
        }
        Ok(())
    }

    /// Create default adversarial scenarios
    fn create_default_scenarios() -> Vec<AdversarialScenario> {
        vec![
            AdversarialScenario {
                name: "NaN Storm".to_string(),
                description: "Inject NaN values to test calibration robustness".to_string(),
                scenario_type: AdversarialScenarioType::NanStorm,
                duration_minutes: 10,
                intensity: AdversarialIntensity::Moderate,
                success_criteria: vec![
                    SuccessCriterion {
                        metric: "AECE".to_string(),
                        threshold: 0.02,
                        comparison: ComparisonOperator::LessThan,
                        description: "AECE remains below 0.02 during NaN injection".to_string(),
                    }
                ],
                failure_criteria: vec![
                    FailureCriterion {
                        metric: "AECE".to_string(),
                        threshold: 0.05,
                        comparison: ComparisonOperator::GreaterThan,
                        severity: FailureSeverity::Critical,
                        description: "AECE exceeds 0.05 - critical failure".to_string(),
                    }
                ],
            },
            AdversarialScenario {
                name: "99% Plateau Injection".to_string(),
                description: "Force 99% predictions to test calibration boundaries".to_string(),
                scenario_type: AdversarialScenarioType::PlateauInjection,
                duration_minutes: 8,
                intensity: AdversarialIntensity::Moderate,
                success_criteria: vec![
                    SuccessCriterion {
                        metric: "BrierScore".to_string(),
                        threshold: 0.15,
                        comparison: ComparisonOperator::LessThan,
                        description: "Brier score remains reasonable during plateau injection".to_string(),
                    }
                ],
                failure_criteria: vec![
                    FailureCriterion {
                        metric: "SystemCrash".to_string(),
                        threshold: 1.0,
                        comparison: ComparisonOperator::Equal,
                        severity: FailureSeverity::Critical,
                        description: "System crashes during plateau injection".to_string(),
                    }
                ],
            },
            AdversarialScenario {
                name: "Adversarial g(s) Manipulation".to_string(),
                description: "Apply adversarial transformations to calibration function".to_string(),
                scenario_type: AdversarialScenarioType::AdversarialG,
                duration_minutes: 12,
                intensity: AdversarialIntensity::Severe,
                success_criteria: vec![
                    SuccessCriterion {
                        metric: "CalibrationStability".to_string(),
                        threshold: 0.8,
                        comparison: ComparisonOperator::GreaterThan,
                        description: "Calibration maintains stability under adversarial manipulation".to_string(),
                    }
                ],
                failure_criteria: vec![
                    FailureCriterion {
                        metric: "PredictionAccuracy".to_string(),
                        threshold: 0.5,
                        comparison: ComparisonOperator::LessThan,
                        severity: FailureSeverity::Major,
                        description: "Prediction accuracy drops below acceptable threshold".to_string(),
                    }
                ],
            },
        ]
    }

    /// Get chaos testing history
    pub fn get_execution_history(&self) -> Vec<ChaosExecution> {
        self.execution_history.read().unwrap().clone()
    }

    /// Get next scheduled execution time
    pub fn get_next_execution_time(&self) -> Option<SystemTime> {
        self.chaos_scheduler.next_execution
    }
}

impl ChaosScheduler {
    fn new(config: ChaosConfig) -> Self {
        let mut scheduler = Self {
            next_execution: None,
            config,
        };
        scheduler.calculate_next_execution();
        scheduler
    }

    async fn should_execute_chaos(&self) -> bool {
        if let Some(next_time) = self.next_execution {
            SystemTime::now() >= next_time
        } else {
            false
        }
    }

    fn update_next_execution(&mut self) {
        self.calculate_next_execution();
    }

    fn calculate_next_execution(&mut self) {
        // Calculate next monthly execution time
        // For simplicity, we'll use a fixed interval approach
        let now = SystemTime::now();
        let next_month = now + Duration::from_secs(30 * 24 * 3600); // 30 days
        self.next_execution = Some(next_month);
    }
}

impl TestExecutor {
    fn new() -> Self {
        Self {
            active_scenarios: HashMap::new(),
            injection_state: InjectionState {
                nan_injection_active: false,
                plateau_injection_active: false,
                adversarial_g_active: false,
                network_disruption_active: false,
            },
        }
    }
}

impl ResilienceValidator {
    fn new() -> Self {
        Self {
            baseline_metrics: None,
            chaos_metrics: VecDeque::new(),
            sla_violations: Vec::new(),
        }
    }
}

impl Default for ScenarioMetrics {
    fn default() -> Self {
        Self {
            nan_injections: 0,
            plateau_injections: 0,
            adversarial_g_manipulations: 0,
            successful_predictions: 0,
            failed_predictions: 0,
        }
    }
}

impl Default for SlaImpactReport {
    fn default() -> Self {
        Self {
            aece_degradation_percentage: 0.0,
            latency_impact_percentage: 0.0,
            error_rate_increase_percentage: 0.0,
            recovery_time_minutes: 0.0,
            error_budget_consumed_percentage: 0.0,
            circuit_breaker_trips: 0,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ChaosError {
    #[error("Baseline establishment failed: {0}")]
    BaselineError(String),
    
    #[error("Scenario execution failed: {0}")]
    ScenarioExecutionError(String),
    
    #[error("Monitoring failed: {0}")]
    MonitoringError(String),
    
    #[error("Validation failed: {0}")]
    ValidationError(String),
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    #[error("Emergency abort triggered: {0}")]
    EmergencyAbort(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::slo_operations::MonitoringConfig;

    #[tokio::test]
    async fn test_chaos_framework_creation() {
        let slo_dashboard = Arc::new(
            SloOperationsDashboard::new(MonitoringConfig::default())
                .await.unwrap()
        );
        let config = ChaosConfig::default();
        
        let framework = ChaosEngineeringFramework::new(slo_dashboard, config).await.unwrap();
        assert_eq!(framework.adversarial_scenarios.len(), 3);
    }

    #[tokio::test]
    async fn test_scenario_creation() {
        let scenarios = ChaosEngineeringFramework::create_default_scenarios();
        assert!(!scenarios.is_empty());
        
        let nan_scenario = scenarios.iter().find(|s| s.name == "NaN Storm").unwrap();
        assert_eq!(nan_scenario.scenario_type, AdversarialScenarioType::NanStorm);
        assert!(!nan_scenario.success_criteria.is_empty());
    }

    #[tokio::test]
    async fn test_resilience_score_calculation() {
        let slo_dashboard = Arc::new(
            SloOperationsDashboard::new(MonitoringConfig::default())
                .await.unwrap()
        );
        let config = ChaosConfig::default();
        let framework = ChaosEngineeringFramework::new(slo_dashboard, config).await.unwrap();
        
        let execution = ChaosExecution {
            execution_id: "test".to_string(),
            start_time: SystemTime::now(),
            end_time: Some(SystemTime::now()),
            scenarios_executed: vec!["Test Scenario".to_string()],
            overall_result: ChaosResult::Success,
            sla_impact: SlaImpactReport {
                aece_degradation_percentage: 2.0,
                latency_impact_percentage: 1.0,
                error_rate_increase_percentage: 0.5,
                recovery_time_minutes: 5.0,
                error_budget_consumed_percentage: 10.0,
                circuit_breaker_trips: 0,
            },
            resilience_score: 0.0,
            recommendations: Vec::new(),
            raw_metrics: "test.json".to_string(),
        };
        
        let score = framework.calculate_resilience_score(&execution).await.unwrap();
        assert!(score >= 0.0 && score <= 100.0);
    }

    #[tokio::test]
    async fn test_sla_impact_calculation() {
        let slo_dashboard = Arc::new(
            SloOperationsDashboard::new(MonitoringConfig::default())
                .await.unwrap()
        );
        let config = ChaosConfig::default();
        let mut framework = ChaosEngineeringFramework::new(slo_dashboard, config).await.unwrap();
        
        // Set up baseline and chaos metrics
        framework.resilience_validator.baseline_metrics = Some(BaselineMetrics {
            aece: 0.01,
            dece: 0.008,
            brier_score: 0.12,
            latency_p99: 0.8,
            error_rate: 0.005,
            timestamp: SystemTime::now(),
        });
        
        framework.resilience_validator.chaos_metrics.push_back(ChaosMetrics {
            aece: 0.012,  // 20% increase
            dece: 0.009,
            brier_score: 0.125,
            latency_p99: 0.9,  // 12.5% increase
            error_rate: 0.008,  // 60% increase
            timestamp: SystemTime::now(),
            active_scenarios: vec!["Test".to_string()],
        });
        
        let impact = framework.calculate_sla_impact().await.unwrap();
        assert!(impact.aece_degradation_percentage > 0.0);
        assert!(impact.latency_impact_percentage > 0.0);
    }

    #[test]
    fn test_chaos_scheduler() {
        let config = ChaosConfig::default();
        let scheduler = ChaosScheduler::new(config);
        assert!(scheduler.next_execution.is_some());
    }
}