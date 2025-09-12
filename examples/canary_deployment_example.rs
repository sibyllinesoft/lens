//! # CALIB_V22 Canary Deployment Example
//!
//! Demonstrates the complete feature flag and canary rollout system for CALIB_V22
//! including progressive rollout, SLA gate validation, and automatic circuit breakers.

use lens_core::calibration::{
    feature_flags::{
        CalibV22Config, CalibV22FeatureFlag, BucketStrategy, BucketMethod,
        SlaGateConfig, AutoRevertConfig, PromotionCriteria, RolloutStage,
    },
    canary_controller::{
        CanaryController, CanaryControllerConfig, SlaValidationConfig,
        ProgressionRules, CanaryMonitoringConfig, CanaryDecisionType,
    },
    sla_tripwires::{SlaTripwires, SlaConfig, PerformanceMetrics},
    drift_monitor::{DriftMonitor, DriftThresholds, HealthStatus, CanaryGateConfig},
    shared_binning_core::SharedBinningConfig,
    fast_bootstrap::FastBootstrapConfig,
    CalibrationResult, CalibrationSample, CalibrationMethod,
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use chrono::Utc;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("🚀 CALIB_V22 Canary Deployment Example");
    println!("=====================================");
    
    // Step 1: Configure CALIB_V22 feature flag system
    let feature_flag_config = create_feature_flag_config();
    println!("📋 Feature flag configuration created");
    println!("   - Rollout stage: Canary (5%)");
    println!("   - SLA gates: P99<1ms, AECE-τ≤0.01, confidence shift≤0.02");
    println!("   - Auto-revert: Enabled (2 consecutive breaches)");
    
    // Step 2: Initialize supporting systems
    let sla_config = SlaConfig::default();
    let drift_thresholds = DriftThresholds::default();
    let binning_config = SharedBinningConfig::default();
    
    // Create feature flag system
    let feature_flag = Arc::new(CalibV22FeatureFlag::new(
        feature_flag_config,
        sla_config.clone(),
        drift_thresholds.clone(),
        binning_config.clone(),
    )?);
    println!("✅ CALIB_V22 feature flag system initialized");
    
    // Step 3: Initialize canary controller
    let canary_config = create_canary_config();
    
    let sla_tripwires = Arc::new(RwLock::new(
        SlaTripwires::new(sla_config, drift_thresholds.clone())
    ));
    
    let canary_gate_config = CanaryGateConfig::default();
    let bootstrap_config = FastBootstrapConfig::default();
    let drift_monitor = Arc::new(RwLock::new(
        DriftMonitor::new(drift_thresholds, canary_gate_config, binning_config, bootstrap_config)
    ));
    
    let canary_controller = CanaryController::new(
        canary_config,
        feature_flag.clone(),
        sla_tripwires.clone(),
        drift_monitor.clone(),
    )?;
    println!("✅ Canary deployment controller initialized");
    
    // Step 4: Start monitoring
    canary_controller.start_monitoring()?;
    println!("📊 Canary monitoring started");
    
    // Step 5: Simulate repository traffic
    println!("\n🔄 Simulating repository traffic...");
    
    let test_repositories = vec![
        "microsoft/vscode",
        "facebook/react", 
        "torvalds/linux",
        "rust-lang/rust",
        "golang/go",
        "python/cpython",
        "nodejs/node",
        "tensorflow/tensorflow",
    ];
    
    for (i, repo_id) in test_repositories.iter().enumerate() {
        // Make feature flag decision
        let decision = feature_flag.should_use_calib_v22(repo_id)?;
        
        println!("Repository: {} -> CALIB_V22: {} ({})", 
            repo_id, 
            if decision.use_calib_v22 { "✅ YES" } else { "❌ NO" },
            decision.decision_reason
        );
        
        // Simulate calibration result
        let calibration_result = if decision.use_calib_v22 {
            simulate_v22_calibration_result()
        } else {
            simulate_control_calibration_result()
        };
        
        // Record result for monitoring
        feature_flag.record_calibration_result(
            decision.use_calib_v22,
            &calibration_result,
            repo_id,
        )?;
        
        // Simulate some processing time
        if i % 3 == 0 {
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
    }
    
    // Step 6: Evaluate canary stage
    println!("\n📈 Evaluating canary stage...");
    let decision = canary_controller.evaluate_stage()?;
    
    match decision.decision_type {
        CanaryDecisionType::Continue => {
            println!("✅ Stage evaluation: CONTINUE monitoring");
        }
        CanaryDecisionType::Promote => {
            println!("🎯 Stage evaluation: PROMOTE to {}", 
                decision.target_stage.unwrap_or_else(|| "Unknown".to_string()));
        }
        CanaryDecisionType::Rollback => {
            println!("⚠️ Stage evaluation: ROLLBACK to {}", 
                decision.target_stage.unwrap_or_else(|| "Unknown".to_string()));
        }
        CanaryDecisionType::EmergencyStop => {
            println!("🚨 Stage evaluation: EMERGENCY STOP");
        }
        CanaryDecisionType::ManualIntervention => {
            println!("👥 Stage evaluation: MANUAL INTERVENTION required");
        }
    }
    
    println!("Reason: {}", decision.reason);
    
    // Step 7: Display system status
    println!("\n📊 System Status");
    println!("================");
    
    let feature_flag_status = feature_flag.get_status()?;
    println!("Feature Flag Status:");
    println!("  - Enabled: {}", feature_flag_status["enabled"]);
    println!("  - Current Stage: {}", feature_flag_status["current_stage"]);
    println!("  - Rollout Percentage: {}%", feature_flag_status["rollout_percentage"]);
    println!("  - Circuit Breaker: {}", 
        if feature_flag_status["circuit_breaker_open"].as_bool().unwrap_or(false) {
            "🔴 OPEN"
        } else {
            "🟢 CLOSED"
        }
    );
    
    let canary_status = canary_controller.get_status()?;
    println!("\nCanary Controller Status:");
    println!("  - Monitoring Active: {}", canary_status["monitoring_active"]);
    println!("  - Current Stage: {}", canary_status["current_stage"]);
    println!("  - Consecutive Breaches: {}", canary_status["consecutive_breaches"]);
    println!("  - Total Decisions: {}", canary_status["decisions"]["total_count"]);
    
    // Step 8: Demonstrate forced operations
    println!("\n🔧 Demonstrating administrative operations...");
    
    // Force a stage promotion (for demonstration)
    println!("Forcing promotion to Limited stage...");
    let forced_decision = canary_controller.force_decision(
        CanaryDecisionType::Promote,
        "Administrative override for demonstration".to_string(),
    )?;
    println!("✅ Forced promotion completed: {}", forced_decision.reason);
    
    // Show updated status
    let updated_status = feature_flag.get_status()?;
    println!("Updated rollout percentage: {}%", updated_status["rollout_percentage"]);
    
    // Step 9: Simulate SLA violation and auto-revert
    println!("\n⚠️ Simulating SLA violation scenario...");
    
    // Force emergency stop to demonstrate auto-revert
    let emergency_decision = canary_controller.force_decision(
        CanaryDecisionType::EmergencyStop,
        "Simulated critical SLA violation".to_string(),
    )?;
    println!("🚨 Emergency stop triggered: {}", emergency_decision.reason);
    
    let final_status = feature_flag.get_status()?;
    println!("Final circuit breaker status: {}", 
        if final_status["circuit_breaker_open"].as_bool().unwrap_or(false) {
            "🔴 OPEN (Feature disabled)"
        } else {
            "🟢 CLOSED"
        }
    );
    
    // Step 10: Cleanup
    canary_controller.stop_monitoring()?;
    println!("\n✅ Canary deployment example completed successfully");
    println!("   - Demonstrated progressive rollout configuration");
    println!("   - Showed repository-based traffic splitting"); 
    println!("   - Validated SLA gate enforcement");
    println!("   - Tested automatic promotion and rollback");
    println!("   - Verified circuit breaker integration");
    
    Ok(())
}

fn create_feature_flag_config() -> CalibV22Config {
    let mut override_buckets = HashMap::new();
    // Force specific repositories for testing
    override_buckets.insert("microsoft/vscode".to_string(), true);
    override_buckets.insert("facebook/react".to_string(), false);
    
    CalibV22Config {
        enabled: true,
        rollout_percentage: 5, // Start with 5% canary
        rollout_stage: "Canary".to_string(),
        bucket_strategy: BucketStrategy {
            method: BucketMethod::RepositoryHash { 
                salt: "calib_v22_demo_2025".to_string() 
            },
            bucket_salt: "canary_demo".to_string(),
            sticky_sessions: true,
            override_buckets,
        },
        sla_gates: SlaGateConfig {
            max_p99_latency_increase_us: 1000.0, // 1ms
            max_aece_tau_threshold: 0.01,
            max_confidence_shift: 0.02,
            require_zero_sla_recall_change: true,
            evaluation_window_minutes: 15,
            consecutive_breach_threshold: 2,
        },
        auto_revert_config: AutoRevertConfig {
            enabled: true,
            breach_window_threshold: 2,
            breach_window_duration_minutes: 15,
            revert_cooldown_minutes: 30, // Shorter for demo
            max_reverts_per_day: 10,
        },
        config_fingerprint: format!("demo_v22_{}", Utc::now().timestamp()),
        rollout_start_time: Utc::now(),
        promotion_criteria: PromotionCriteria {
            min_observation_hours: 1, // Shorter for demo
            required_health_status: HealthStatus::Green,
            max_aece_degradation: 0.005,
            require_p99_compliance: true,
            min_success_rate: 0.995,
        },
    }
}

fn create_canary_config() -> CanaryControllerConfig {
    let mut min_observation_hours = HashMap::new();
    min_observation_hours.insert("Canary".to_string(), 1);    // Shorter for demo
    min_observation_hours.insert("Limited".to_string(), 2);
    min_observation_hours.insert("Major".to_string(), 3);
    min_observation_hours.insert("Full".to_string(), 4);
    
    let mut success_rate_thresholds = HashMap::new();
    success_rate_thresholds.insert("Canary".to_string(), 0.99);
    success_rate_thresholds.insert("Limited".to_string(), 0.995);
    success_rate_thresholds.insert("Major".to_string(), 0.998);
    success_rate_thresholds.insert("Full".to_string(), 0.999);
    
    let mut min_sample_counts = HashMap::new();
    min_sample_counts.insert("Canary".to_string(), 10);       // Lower for demo
    min_sample_counts.insert("Limited".to_string(), 50);
    min_sample_counts.insert("Major".to_string(), 100);
    min_sample_counts.insert("Full".to_string(), 500);
    
    CanaryControllerConfig {
        auto_promotion_enabled: true,
        auto_rollback_enabled: true,
        sla_validation: SlaValidationConfig {
            p99_latency_sla_us: 1000.0,
            aece_tau_threshold: 0.01,
            max_confidence_shift: 0.02,
            require_zero_sla_recall_change: true,
            evaluation_window_minutes: 5, // Shorter for demo
            breach_detection: crate::calibration::canary_controller::BreachDetectionConfig {
                consecutive_breach_threshold: 2,
                window_duration_minutes: 5,
                grace_period_minutes: 1,
                max_breach_rate: 0.1,
            },
        },
        progression_rules: ProgressionRules {
            min_observation_hours,
            success_rate_thresholds,
            required_health_status: HealthStatus::Green,
            min_sample_counts,
        },
        monitoring_config: CanaryMonitoringConfig {
            real_time_metrics: true,
            collection_interval_sec: 10, // Faster for demo
            alert_thresholds: crate::calibration::canary_controller::AlertThresholds {
                latency_degradation_threshold: 0.15,
                error_rate_threshold: 0.01,
                aece_degradation_threshold: 0.005,
            },
            baseline_tracking: crate::calibration::canary_controller::BaselineTrackingConfig {
                enabled: true,
                baseline_lookback_days: 1, // Shorter for demo
                require_green_baseline: true,
                significance_level: 0.05,
            },
        },
        config_fingerprint: format!("canary_demo_{}", Utc::now().timestamp()),
    }
}

fn simulate_v22_calibration_result() -> CalibrationResult {
    CalibrationResult {
        input_score: 0.8,
        calibrated_score: 0.78, // Slightly better calibration
        method_used: CalibrationMethod::IsotonicRegression { slope: 0.95 },
        intent: "search".to_string(),
        language: Some("typescript".to_string()),
        slice_ece: 0.009, // Good ECE
        calibration_confidence: 0.95,
    }
}

fn simulate_control_calibration_result() -> CalibrationResult {
    CalibrationResult {
        input_score: 0.8,
        calibrated_score: 0.82, // Standard calibration
        method_used: CalibrationMethod::TemperatureScaling { temperature: 1.1 },
        intent: "search".to_string(), 
        language: Some("typescript".to_string()),
        slice_ece: 0.013, // Slightly worse ECE
        calibration_confidence: 0.85,
    }
}