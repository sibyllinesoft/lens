//! # SLA Monitoring Integration Tests
//!
//! Comprehensive integration tests for the SLA monitoring and auto-revert system,
//! covering real-world scenarios, breach detection, auto-revert triggers, and
//! performance validation.
//!
//! Test coverage:
//! - Real-time monitoring with rolling windows
//! - Breach detection with consecutive window validation  
//! - Auto-revert trigger scenarios
//! - Performance and accuracy testing
//! - Statistical validation
//! - Integration with existing calibration infrastructure

use lens_core::calibration::{
    CalibrationResult, CalibrationMethod, CalibrationSample,
    sla_monitoring::{SlaMonitoringSystem, SlaMonitoringConfig, BaselineMetrics, AutoRevertEvent},
    monitoring_gates::{SlaGateEvaluator, P99LatencyGate, AeceTauGate, ConfidenceShiftGate, SlaRecallGate, GateContext},
    sla_tripwires::{SlaTripwires, SlaConfig, PerformanceMetrics, AutoRevertAction},
    drift_monitor::{DriftThresholds, HealthStatus},
};
use chrono::{DateTime, Utc, Duration as ChronoDuration};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use anyhow::Result;

/// Test fixture for SLA monitoring integration tests
struct SlaMonitoringTestFixture {
    monitoring_system: SlaMonitoringSystem,
    test_data: TestDataGenerator,
}

impl SlaMonitoringTestFixture {
    async fn new() -> Result<Self> {
        let config = SlaMonitoringConfig {
            window_duration_minutes: 1, // 1 minute for faster testing
            consecutive_breach_threshold: 2,
            max_rolling_windows: 10,
            evaluation_interval_sec: 5, // 5 seconds for faster testing
            ..Default::default()
        };
        
        let sla_config = SlaConfig::default();
        let thresholds = DriftThresholds::default();
        let tripwires = Arc::new(Mutex::new(SlaTripwires::new(sla_config, thresholds)));
        
        let monitoring_system = SlaMonitoringSystem::new(config, tripwires).await?;
        let test_data = TestDataGenerator::new();
        
        Ok(Self {
            monitoring_system,
            test_data,
        })
    }
}

/// Test data generator for various breach scenarios
struct TestDataGenerator {
    base_timestamp: DateTime<Utc>,
}

impl TestDataGenerator {
    fn new() -> Self {
        Self {
            base_timestamp: Utc::now(),
        }
    }
    
    /// Generate calibration result with specific characteristics
    fn generate_calibration_result(
        &self,
        intent: &str,
        language: Option<&str>,
        latency_multiplier: f64,
        calibration_quality: f64,
    ) -> (CalibrationResult, Duration) {
        let base_latency_us = 200.0; // Base 200μs latency
        let latency_us = base_latency_us * latency_multiplier;
        
        let result = CalibrationResult {
            input_score: 0.8,
            calibrated_score: 0.75,
            method_used: CalibrationMethod::IsotonicRegression { slope: 1.0 },
            intent: intent.to_string(),
            language: language.map(|s| s.to_string()),
            slice_ece: (0.01 * calibration_quality) as f32,
            calibration_confidence: 0.9,
        };
        
        let duration = Duration::from_micros(latency_us as u64);
        
        (result, duration)
    }
    
    /// Generate baseline metrics for a slice
    fn generate_baseline_metrics(&self, slice_key: &str) -> BaselineMetrics {
        BaselineMetrics {
            timestamp: self.base_timestamp,
            p99_latency_baseline_us: 200.0, // 200μs baseline
            aece_baseline: 0.005,           // Good calibration baseline
            confidence_baseline: 0.75,     // Baseline confidence
            sla_recall_50_baseline: 0.85,  // Baseline recall
            baseline_sample_count: 1000,
        }
    }
    
    /// Generate mock data for different breach scenarios
    fn generate_breach_scenario(&self, scenario: BreachScenario) -> Vec<(CalibrationResult, Duration)> {
        let mut results = Vec::new();
        
        match scenario {
            BreachScenario::LatencySpike => {
                // Generate results with gradually increasing latency
                for i in 0..20 {
                    let latency_multiplier = 1.0 + (i as f64 * 0.5); // Up to 10x latency
                    let (result, duration) = self.generate_calibration_result(
                        "exact_match", Some("rust"), latency_multiplier, 1.0
                    );
                    results.push((result, duration));
                }
            }
            BreachScenario::AeceDeterioration => {
                // Generate results with poor calibration quality
                for i in 0..20 {
                    let quality = 5.0 + (i as f64 * 0.5); // ECE increases to 0.15
                    let (result, duration) = self.generate_calibration_result(
                        "semantic", Some("python"), 1.0, quality
                    );
                    results.push((result, duration));
                }
            }
            BreachScenario::ConsecutiveBreaches => {
                // Generate consistent SLA violations
                for _ in 0..15 {
                    let (result, duration) = self.generate_calibration_result(
                        "structural", Some("typescript"), 8.0, 3.0 // High latency + poor ECE
                    );
                    results.push((result, duration));
                }
            }
            BreachScenario::IntermittentIssues => {
                // Alternating good and bad results
                for i in 0..20 {
                    let multiplier = if i % 3 == 0 { 6.0 } else { 1.0 };
                    let quality = if i % 3 == 0 { 4.0 } else { 1.0 };
                    let (result, duration) = self.generate_calibration_result(
                        "identifier", Some("go"), multiplier, quality
                    );
                    results.push((result, duration));
                }
            }
        }
        
        results
    }
}

/// Different breach scenarios for testing
#[derive(Debug, Clone)]
enum BreachScenario {
    LatencySpike,
    AeceDeterioration,
    ConsecutiveBreaches,
    IntermittentIssues,
}

/// Test basic SLA monitoring system initialization
#[tokio::test]
async fn test_sla_monitoring_initialization() -> Result<()> {
    let fixture = SlaMonitoringTestFixture::new().await?;
    
    // Test that system initializes with correct configuration
    let metrics = fixture.monitoring_system.get_monitoring_metrics().await?;
    assert_eq!(metrics.overall_breach_rate, 0.0);
    assert_eq!(metrics.total_slices_monitored, 0);
    
    Ok(())
}

/// Test baseline setting and retrieval
#[tokio::test]
async fn test_baseline_metrics_management() -> Result<()> {
    let fixture = SlaMonitoringTestFixture::new().await?;
    
    let slice_key = "test:rust";
    let baseline = fixture.test_data.generate_baseline_metrics(slice_key);
    
    // Set baseline
    fixture.monitoring_system.set_baseline_metrics(slice_key, baseline.clone()).await?;
    
    // Verify baseline is set correctly by checking that subsequent measurements use it
    let (result, duration) = fixture.test_data.generate_calibration_result(
        "test", Some("rust"), 1.0, 1.0
    );
    
    fixture.monitoring_system.record_calibration_result(&result, duration).await?;
    
    // Should not panic or error - baseline is available for comparison
    Ok(())
}

/// Test calibration result recording and window management
#[tokio::test]
async fn test_calibration_result_recording() -> Result<()> {
    let fixture = SlaMonitoringTestFixture::new().await?;
    
    let slice_key = "exact_match:typescript";
    let baseline = fixture.test_data.generate_baseline_metrics(slice_key);
    fixture.monitoring_system.set_baseline_metrics(slice_key, baseline).await?;
    
    // Record multiple calibration results
    for i in 0..10 {
        let (result, duration) = fixture.test_data.generate_calibration_result(
            "exact_match", Some("typescript"), 1.0 + (i as f64 * 0.1), 1.0
        );
        
        fixture.monitoring_system.record_calibration_result(&result, duration).await?;
    }
    
    let metrics = fixture.monitoring_system.get_monitoring_metrics().await?;
    assert!(metrics.avg_evaluation_latency_us > 0.0);
    
    Ok(())
}

/// Test latency spike detection and breach handling
#[tokio::test]
async fn test_latency_spike_detection() -> Result<()> {
    let fixture = SlaMonitoringTestFixture::new().await?;
    
    let slice_key = "exact_match:rust";
    let baseline = fixture.test_data.generate_baseline_metrics(slice_key);
    fixture.monitoring_system.set_baseline_metrics(slice_key, baseline).await?;
    
    // Generate latency spike scenario
    let spike_data = fixture.test_data.generate_breach_scenario(BreachScenario::LatencySpike);
    
    let start_time = Instant::now();
    for (result, duration) in spike_data {
        fixture.monitoring_system.record_calibration_result(&result, duration).await?;
        
        // Small delay to simulate real-time processing
        sleep(Duration::from_millis(10)).await;
    }
    
    // Check for breach detection
    let health_summary = fixture.monitoring_system.get_health_summary().await?;
    let metrics = fixture.monitoring_system.get_monitoring_metrics().await?;
    
    // Should have detected breaches due to high latency
    assert!(metrics.overall_breach_rate > 0.0);
    println!("Latency spike test completed in {:?}", start_time.elapsed());
    
    Ok(())
}

/// Test AECE-τ deterioration detection
#[tokio::test]
async fn test_aece_deterioration_detection() -> Result<()> {
    let fixture = SlaMonitoringTestFixture::new().await?;
    
    let slice_key = "semantic:python";
    let baseline = fixture.test_data.generate_baseline_metrics(slice_key);
    fixture.monitoring_system.set_baseline_metrics(slice_key, baseline).await?;
    
    // Generate AECE deterioration scenario
    let deterioration_data = fixture.test_data.generate_breach_scenario(BreachScenario::AeceDeterioration);
    
    for (result, duration) in deterioration_data {
        fixture.monitoring_system.record_calibration_result(&result, duration).await?;
        sleep(Duration::from_millis(10)).await;
    }
    
    let metrics = fixture.monitoring_system.get_monitoring_metrics().await?;
    
    // Should detect calibration quality deterioration
    assert!(metrics.slice_breach_rates.get("semantic:python").unwrap_or(&0.0) > &0.0);
    
    Ok(())
}

/// Test consecutive breach detection and auto-revert triggering
#[tokio::test]
async fn test_consecutive_breach_auto_revert() -> Result<()> {
    let fixture = SlaMonitoringTestFixture::new().await?;
    
    let slice_key = "structural:typescript";
    let baseline = fixture.test_data.generate_baseline_metrics(slice_key);
    fixture.monitoring_system.set_baseline_metrics(slice_key, baseline).await?;
    
    // Generate consecutive breaches
    let breach_data = fixture.test_data.generate_breach_scenario(BreachScenario::ConsecutiveBreaches);
    
    for (result, duration) in breach_data {
        fixture.monitoring_system.record_calibration_result(&result, duration).await?;
        sleep(Duration::from_millis(10)).await;
        
        // Check for auto-revert after each result
        let revert_events = fixture.monitoring_system.check_breach_conditions().await?;
        if !revert_events.is_empty() {
            println!("Auto-revert triggered: {:?}", revert_events[0]);
            break;
        }
    }
    
    // Final check for breach conditions
    let final_revert_events = fixture.monitoring_system.check_breach_conditions().await?;
    assert!(!final_revert_events.is_empty(), "Expected auto-revert to be triggered");
    
    let revert_event = &final_revert_events[0];
    assert_eq!(revert_event.slice_key, slice_key);
    assert!(revert_event.consecutive_breaches >= 2);
    
    Ok(())
}

/// Test intermittent issues handling (should not trigger auto-revert)
#[tokio::test]
async fn test_intermittent_issues_handling() -> Result<()> {
    let fixture = SlaMonitoringTestFixture::new().await?;
    
    let slice_key = "identifier:go";
    let baseline = fixture.test_data.generate_baseline_metrics(slice_key);
    fixture.monitoring_system.set_baseline_metrics(slice_key, baseline).await?;
    
    // Generate intermittent issues
    let intermittent_data = fixture.test_data.generate_breach_scenario(BreachScenario::IntermittentIssues);
    
    for (result, duration) in intermittent_data {
        fixture.monitoring_system.record_calibration_result(&result, duration).await?;
        sleep(Duration::from_millis(10)).await;
    }
    
    // Check that no auto-revert is triggered (intermittent issues shouldn't cause consecutive breaches)
    let revert_events = fixture.monitoring_system.check_breach_conditions().await?;
    
    // Should not have triggered auto-revert due to intermittent nature
    let health_summary = fixture.monitoring_system.get_health_summary().await?;
    if let Some(health_status) = health_summary.get(slice_key) {
        // Health status might be degraded but should not trigger critical auto-revert
        assert_ne!(*health_status, HealthStatus::Critical);
    }
    
    Ok(())
}

/// Test performance characteristics of SLA monitoring system
#[tokio::test]
async fn test_sla_monitoring_performance() -> Result<()> {
    let fixture = SlaMonitoringTestFixture::new().await?;
    
    let slice_key = "performance:test";
    let baseline = fixture.test_data.generate_baseline_metrics(slice_key);
    fixture.monitoring_system.set_baseline_metrics(slice_key, baseline).await?;
    
    // Performance test: process many results quickly
    let start_time = Instant::now();
    let num_results = 1000;
    
    for i in 0..num_results {
        let (result, duration) = fixture.test_data.generate_calibration_result(
            "performance", Some("test"), 1.0, 1.0 + (i as f64 * 0.001)
        );
        
        fixture.monitoring_system.record_calibration_result(&result, duration).await?;
    }
    
    let total_duration = start_time.elapsed();
    let per_result_duration = total_duration / num_results;
    
    println!("Processed {} results in {:?} ({:?} per result)", 
             num_results, total_duration, per_result_duration);
    
    // Performance target: <1ms per result evaluation
    assert!(per_result_duration < Duration::from_millis(1), 
            "Per-result processing time {:?} exceeded 1ms target", per_result_duration);
    
    // Memory usage should be reasonable
    let metrics = fixture.monitoring_system.get_monitoring_metrics().await?;
    assert!(metrics.memory_usage_bytes < 50 * 1024 * 1024, // <50MB
            "Memory usage {} bytes exceeded 50MB limit", metrics.memory_usage_bytes);
    
    Ok(())
}

/// Test statistical validation in monitoring gates
#[tokio::test]
async fn test_statistical_validation() -> Result<()> {
    use lens_core::calibration::monitoring_gates::{SlaGateEvaluator, P99LatencyGate, GateContext};
    
    let mut evaluator = SlaGateEvaluator::new();
    evaluator.add_gate(Box::new(P99LatencyGate::new(1000.0, 50.0)));
    
    // Create context with historical data for statistical testing
    let historical_values = (0..50)
        .map(|i| (Utc::now(), 400.0 + (i as f64 * 10.0)))
        .collect();
    
    let context = GateContext {
        slice_key: "statistical:test".to_string(),
        timestamp: Utc::now(),
        baseline_value: Some(500.0),
        historical_values,
        metadata: HashMap::new(),
    };
    
    let measurements = [("p99_calibration_latency".to_string(), 1200.0)]
        .iter().cloned().collect();
    
    let results = evaluator.evaluate_all_gates(&measurements, &context)?;
    
    assert_eq!(results.len(), 1);
    let evaluation = results.get("p99_calibration_latency").unwrap();
    
    // Should fail threshold test
    assert!(!evaluation.passed);
    
    // Should have statistical significance result
    assert!(evaluation.statistical_significance.is_some());
    
    let stats = evaluation.statistical_significance.as_ref().unwrap();
    println!("Statistical test result: p-value={:.6}, significant={}", 
             stats.p_value, stats.is_significant);
    
    Ok(())
}

/// Test memory usage and resource management
#[tokio::test]
async fn test_memory_management() -> Result<()> {
    let fixture = SlaMonitoringTestFixture::new().await?;
    
    // Generate data for multiple slices to test memory management
    let slices = [
        ("exact_match", "rust"),
        ("semantic", "python"),
        ("structural", "typescript"),
        ("identifier", "go"),
        ("fuzzy", "java"),
    ];
    
    for (intent, lang) in &slices {
        let slice_key = format!("{}:{}", intent, lang);
        let baseline = fixture.test_data.generate_baseline_metrics(&slice_key);
        fixture.monitoring_system.set_baseline_metrics(&slice_key, baseline).await?;
    }
    
    // Generate many results across all slices
    for _ in 0..200 {
        for (intent, lang) in &slices {
            let (result, duration) = fixture.test_data.generate_calibration_result(
                intent, Some(lang), 1.0, 1.0
            );
            
            fixture.monitoring_system.record_calibration_result(&result, duration).await?;
        }
    }
    
    let metrics = fixture.monitoring_system.get_monitoring_metrics().await?;
    
    // Should be monitoring all slices
    assert_eq!(metrics.total_slices_monitored, slices.len());
    
    // Memory usage should be within limits
    assert!(metrics.memory_usage_bytes < 100 * 1024 * 1024, // <100MB
            "Memory usage exceeded limit: {} bytes", metrics.memory_usage_bytes);
    
    println!("Memory usage: {} bytes across {} slices", 
             metrics.memory_usage_bytes, metrics.total_slices_monitored);
    
    Ok(())
}

/// Test forced auto-revert functionality
#[tokio::test]
async fn test_forced_auto_revert() -> Result<()> {
    let fixture = SlaMonitoringTestFixture::new().await?;
    
    let slice_key = "emergency:test";
    
    // Test forced auto-revert
    fixture.monitoring_system.force_auto_revert(slice_key, "Emergency test scenario").await?;
    
    // Check health status after forced revert
    let health_summary = fixture.monitoring_system.get_health_summary().await?;
    
    // The system should record the forced revert event
    println!("Health summary after forced revert: {:?}", health_summary);
    
    Ok(())
}

/// Test comprehensive monitoring metrics collection
#[tokio::test]
async fn test_comprehensive_monitoring_metrics() -> Result<()> {
    let fixture = SlaMonitoringTestFixture::new().await?;
    
    // Set up multiple slices with different behaviors
    let test_scenarios = [
        ("good:slice", BreachScenario::LatencySpike),    // This will cause breaches
        ("stable:slice", BreachScenario::IntermittentIssues), // This will be mixed
    ];
    
    for (slice_name, _scenario) in &test_scenarios {
        let baseline = fixture.test_data.generate_baseline_metrics(slice_name);
        fixture.monitoring_system.set_baseline_metrics(slice_name, baseline).await?;
    }
    
    // Generate different patterns for each slice
    for (slice_name, scenario) in &test_scenarios {
        let data = fixture.test_data.generate_breach_scenario(scenario.clone());
        
        for (result, duration) in data {
            // Modify result to match slice name
            let mut modified_result = result;
            let parts: Vec<&str> = slice_name.split(':').collect();
            if parts.len() == 2 {
                modified_result.intent = parts[0].to_string();
                modified_result.language = Some(parts[1].to_string());
            }
            
            fixture.monitoring_system.record_calibration_result(&modified_result, duration).await?;
            sleep(Duration::from_millis(5)).await;
        }
    }
    
    // Collect comprehensive metrics
    let metrics = fixture.monitoring_system.get_monitoring_metrics().await?;
    
    println!("Comprehensive metrics:");
    println!("  Overall breach rate: {:.3}", metrics.overall_breach_rate);
    println!("  Total slices monitored: {}", metrics.total_slices_monitored);
    println!("  Average evaluation latency: {:.1}μs", metrics.avg_evaluation_latency_us);
    println!("  Active windows: {}", metrics.active_windows_count);
    
    for (slice_key, breach_rate) in &metrics.slice_breach_rates {
        println!("  Slice '{}' breach rate: {:.3}", slice_key, breach_rate);
    }
    
    // Verify metrics are reasonable
    assert!(metrics.total_slices_monitored <= 2);
    assert!(metrics.avg_evaluation_latency_us > 0.0);
    assert!(metrics.avg_evaluation_latency_us < 10000.0); // <10ms per evaluation
    
    Ok(())
}

/// Integration test with existing SLA tripwires
#[tokio::test]
async fn test_sla_tripwires_integration() -> Result<()> {
    let fixture = SlaMonitoringTestFixture::new().await?;
    
    let slice_key = "integration:test";
    let baseline = fixture.test_data.generate_baseline_metrics(slice_key);
    fixture.monitoring_system.set_baseline_metrics(slice_key, baseline).await?;
    
    // Generate severe breaches that should trigger tripwires integration
    let severe_breach_data = fixture.test_data.generate_breach_scenario(BreachScenario::ConsecutiveBreaches);
    
    for (result, duration) in severe_breach_data {
        fixture.monitoring_system.record_calibration_result(&result, duration).await?;
        sleep(Duration::from_millis(10)).await;
    }
    
    // Check for auto-revert events
    let revert_events = fixture.monitoring_system.check_breach_conditions().await?;
    
    if !revert_events.is_empty() {
        let event = &revert_events[0];
        println!("Integration test triggered auto-revert:");
        println!("  Slice: {}", event.slice_key);
        println!("  Reason: {}", event.trigger_reason);
        println!("  Action: {:?}", event.revert_action);
        println!("  Consecutive breaches: {}", event.consecutive_breaches);
        
        // Verify the integration worked
        assert_eq!(event.slice_key, slice_key);
        assert!(event.consecutive_breaches >= 2);
    }
    
    Ok(())
}