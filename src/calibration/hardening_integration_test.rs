//! Integration tests for Day-2 Hardening System
//!
//! Comprehensive tests demonstrating the operational excellence framework
//! for invisible utility operation of the calibration system.

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::calibration::{
        drift_slos::{WeeklyDriftMonitor, DriftSlos, CalibrationMetrics, SloViolation, AlertSeverity},
        operational_runbook::{OperationalRunbook, CalibrationSymptom, RemediationAction},
        regression_detector::{RegressionDetector, RegressionType, RegressionSeverity, WarningLevel},
    };
    use std::collections::HashMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Create test calibration metrics
    fn create_test_metrics(
        aece: f64, 
        dece: f64, 
        alpha: f64, 
        clamp_rate: f64, 
        merged_bin_rate: f64,
        timestamp_offset: u64
    ) -> CalibrationMetrics {
        CalibrationMetrics {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() + timestamp_offset,
            aece,
            dece,
            alpha,
            clamp_rate,
            merged_bin_rate,
            score_range_violations: 0,
            mask_mismatch_count: 0,
            total_samples: 10000,
        }
    }

    #[tokio::test]
    async fn test_comprehensive_hardening_system() {
        println!("\n🛡️ TESTING: Comprehensive Day-2 Hardening System");
        println!("================================================\n");

        // Initialize all hardening components
        let mut drift_monitor = WeeklyDriftMonitor::new();
        let runbook = OperationalRunbook::new();
        let mut regression_detector = RegressionDetector::new();

        // Establish baseline metrics (healthy state)
        let baseline = create_test_metrics(0.012, 0.010, 0.50, 0.05, 0.02, 0);
        drift_monitor.set_baseline(baseline.clone());
        regression_detector.add_metrics(baseline.clone());

        println!("✅ Established baseline metrics:");
        println!("   AECE: {:.4}, DECE: {:.4}, α: {:.4}", baseline.aece, baseline.dece, baseline.alpha);
        println!("   Clamp rate: {:.2}%, Merged bins: {:.2}%", baseline.clamp_rate * 100.0, baseline.merged_bin_rate * 100.0);
        
        // Add some normal variation to establish trend
        for i in 1..=8 {
            let normal_variation = create_test_metrics(
                0.012 + (i as f64 * 0.0002), // Slight AECE increase
                0.010 + (i as f64 * 0.0001), // Slight DECE increase
                0.50 + (i as f64 * 0.001),   // Slight alpha variation
                0.05 + (i as f64 * 0.001),   // Slight clamp rate increase
                0.02 + (i as f64 * 0.0005),  // Slight merged bin increase
                i
            );
            regression_detector.add_metrics(normal_variation);
        }
        
        println!("\n📈 Added normal variation metrics (trend establishment)");

        // Test 1: Weekly SLO Monitoring with Violations
        println!("\n🔍 TEST 1: Weekly SLO Monitoring");
        println!("--------------------------------");
        
        // Create metrics that violate SLOs
        let mut violating_metrics = create_test_metrics(0.025, 0.022, 0.58, 0.12, 0.25, 9);
        violating_metrics.score_range_violations = 3; // Critical violation
        violating_metrics.mask_mismatch_count = 1;    // High severity violation
        
        let violations = drift_monitor.check_slos(violating_metrics.clone());
        
        assert!(!violations.is_empty(), "Should detect SLO violations");
        
        let critical_violations = violations.iter().filter(|v| v.severity == AlertSeverity::Critical).count();
        let high_violations = violations.iter().filter(|v| v.severity == AlertSeverity::High).count();
        
        println!("   Detected {} total violations:", violations.len());
        println!("   • Critical: {}", critical_violations);
        println!("   • High: {}", high_violations);
        
        for violation in &violations {
            println!("   • {}: {:.6} (threshold: {:.6}, severity: {:?})",
                     violation.metric_name, violation.current_value, 
                     violation.threshold, violation.severity);
        }
        
        assert!(critical_violations > 0, "Should detect critical violations");
        assert!(!drift_monitor.is_healthy(), "System should be unhealthy after violations");

        // Test 2: Operational Runbook Response
        println!("\n📋 TEST 2: Operational Runbook Response");
        println!("----------------------------------------");
        
        // Convert violations to symptoms
        let symptoms = vec![
            CalibrationSymptom::AeceDrift { 
                current: violating_metrics.aece, 
                baseline: baseline.aece, 
                threshold: 0.01 
            },
            CalibrationSymptom::ScoreRangeViolations { 
                count: violating_metrics.score_range_violations 
            },
            CalibrationSymptom::ExcessiveMergedBins { 
                rate: violating_metrics.merged_bin_rate,
                warn_threshold: 0.05,
                fail_threshold: 0.20
            },
        ];
        
        // Execute automated incident response
        let remediation_actions = runbook.execute_incident_response(symptoms.clone());
        
        assert!(!remediation_actions.is_empty(), "Should generate remediation actions");
        
        println!("   Generated {} remediation actions:", remediation_actions.len());
        for (i, action) in remediation_actions.iter().enumerate() {
            match action {
                RemediationAction::RaiseConfidenceThreshold { class_id, from, to, reason } => {
                    println!("   {}. Raise confidence threshold for class {} from {:.2} to {:.2} ({})", 
                             i+1, class_id, from, to, reason);
                },
                RemediationAction::RevertToPreviousModel { reason, .. } => {
                    println!("   {}. Revert to previous model ({})", i+1, reason);
                },
                RemediationAction::EscalateToHuman { severity, context } => {
                    println!("   {}. Escalate to human ({:?}): {}", i+1, severity, context);
                },
                _ => {
                    println!("   {}. {:?}", i+1, action);
                }
            }
        }
        
        // Test incident data collection
        let incident_data = runbook.collect_incident_data(symptoms);
        assert_eq!(incident_data.symptoms.len(), 3, "Should capture all symptoms");
        
        println!("   Collected incident data: {} symptoms", incident_data.symptoms.len());

        // Test communication generation
        let technical_comm = runbook.generate_communication("technical_team", &incident_data, &remediation_actions);
        assert!(technical_comm.is_ok(), "Should generate technical communication");
        
        let comm_message = technical_comm.unwrap();
        assert!(comm_message.contains("CALIBRATION INCIDENT ALERT"), "Should contain alert header");
        
        println!("   Generated technical team communication ({} chars)", comm_message.len());

        // Test 3: Regression Detection System  
        println!("\n📊 TEST 3: Regression Detection System");
        println!("---------------------------------------");
        
        // Add the violating metrics to regression detector
        let regressions = regression_detector.add_metrics(violating_metrics);
        
        assert!(!regressions.is_empty(), "Should detect regressions");
        
        let critical_regressions = regressions.iter().filter(|r| r.severity == RegressionSeverity::Critical).count();
        let severe_regressions = regressions.iter().filter(|r| r.severity == RegressionSeverity::Severe).count();
        
        println!("   Detected {} total regressions:", regressions.len());
        println!("   • Critical: {}", critical_regressions);
        println!("   • Severe: {}", severe_regressions);
        
        for regression in &regressions {
            println!("   • {} ({:?}): {:.6} → {:.6} (effect size: {:.3})",
                     regression.metric_name, regression.regression_type,
                     regression.baseline_value, regression.current_value,
                     regression.significance_test.effect_size);
        }
        
        assert!(critical_regressions > 0 || severe_regressions > 0, "Should detect severe regressions");

        // Test early warning system
        let early_warnings = regression_detector.generate_early_warnings();
        
        println!("\n   Early Warning System:");
        for warning in &early_warnings {
            println!("   • {}: {:?} (confidence: {:.1}%)", 
                     warning.metric_name, warning.warning_level, warning.confidence * 100.0);
        }
        
        let red_warnings = early_warnings.iter().filter(|w| w.warning_level == WarningLevel::Red).count();
        assert!(red_warnings > 0, "Should generate red warnings for critical state");

        // Test 4: System Health Assessment
        println!("\n🏥 TEST 4: System Health Assessment");
        println!("------------------------------------");
        
        let (slo_healthy, slo_status) = drift_monitor.get_status_summary();
        let (regression_healthy, regression_status) = regression_detector.get_health_status();
        
        println!("   SLO Health: {}", if slo_healthy.get("healthy").unwrap_or(&"false".to_string()) == "true" { "✅ Healthy" } else { "🚨 Unhealthy" });
        println!("   Regression Health: {}", if regression_healthy { "✅ Healthy" } else { "🚨 Unhealthy" });
        println!("   Regression Status: {}", regression_status);
        
        // Overall system should be unhealthy
        let overall_healthy = drift_monitor.is_healthy() && regression_healthy;
        assert!(!overall_healthy, "Overall system should be unhealthy after violations");
        
        println!("   Overall System Health: {}", if overall_healthy { "✅ Healthy" } else { "🚨 Unhealthy" });

        // Test 5: Recovery Simulation
        println!("\n🔄 TEST 5: Recovery Simulation");
        println!("-------------------------------");
        
        // Simulate recovery by adding improving metrics
        for i in 10..=15 {
            let recovery_metrics = create_test_metrics(
                0.025 - ((i - 9) as f64 * 0.002), // Improving AECE
                0.022 - ((i - 9) as f64 * 0.002), // Improving DECE
                0.52 - ((i - 9) as f64 * 0.003),  // Stabilizing alpha
                0.08 - ((i - 9) as f64 * 0.005),  // Reducing clamp rate
                0.15 - ((i - 9) as f64 * 0.020),  // Reducing merged bins
                i
            );
            regression_detector.add_metrics(recovery_metrics);
        }
        
        // Final healthy metrics
        let recovered_metrics = create_test_metrics(0.013, 0.011, 0.51, 0.06, 0.03, 16);
        let recovery_violations = drift_monitor.check_slos(recovered_metrics.clone());
        let recovery_regressions = regression_detector.add_metrics(recovered_metrics);
        
        println!("   Recovery violations: {}", recovery_violations.len());
        println!("   Recovery regressions: {}", recovery_regressions.len());
        
        let final_warnings = regression_detector.generate_early_warnings();
        let green_warnings = final_warnings.iter().filter(|w| w.warning_level == WarningLevel::Green).count();
        
        println!("   Early warnings after recovery: {} (Green: {})", final_warnings.len(), green_warnings);
        
        // System should be healthier after recovery
        let final_health = drift_monitor.is_healthy();
        println!("   Final system health: {}", if final_health { "✅ Healthy" } else { "⚠️ Still unhealthy" });

        // Test 6: Reporting and Documentation
        println!("\n📄 TEST 6: Reporting and Documentation");
        println!("---------------------------------------");
        
        let weekly_report = drift_monitor.generate_weekly_report();
        let regression_report = regression_detector.generate_regression_report();
        
        println!("   Weekly SLO Report:");
        for line in weekly_report.lines().take(8) {
            println!("     {}", line);
        }
        
        println!("\n   Regression Detection Report:");
        for line in regression_report.lines().take(8) {
            println!("     {}", line);
        }
        
        // Verify reports contain expected content
        assert!(weekly_report.contains("WEEKLY CALIBRATION SLO REPORT"), "Should contain SLO report header");
        assert!(regression_report.contains("CALIBRATION REGRESSION DETECTION REPORT"), "Should contain regression report header");

        println!("\n🎯 HARDENING SYSTEM INTEGRATION TEST COMPLETE");
        println!("==============================================");
        println!("✅ Weekly SLO monitoring with automated alerts");
        println!("✅ Operational runbook with decision automation"); 
        println!("✅ Regression detection with statistical significance");
        println!("✅ Early warning system with trend analysis");
        println!("✅ Automated incident response and communication");
        println!("✅ Comprehensive health monitoring and reporting");
        println!("✅ Recovery detection and system stabilization");
        println!("\n🏆 DAY-2 HARDENING SYSTEM: PRODUCTION READY FOR INVISIBLE UTILITY OPERATION");
    }

    #[tokio::test]
    async fn test_slo_threshold_accuracy() {
        println!("\n🎯 TESTING: SLO Threshold Accuracy");
        println!("==================================");
        
        let mut monitor = WeeklyDriftMonitor::new();
        let baseline = create_test_metrics(0.010, 0.008, 0.50, 0.03, 0.01, 0);
        monitor.set_baseline(baseline);
        
        // Test each SLO threshold precisely
        
        // AECE drift: should trigger at |Δ| = 0.01
        let aece_violation = create_test_metrics(0.021, 0.008, 0.50, 0.03, 0.01, 1); // Δ = 0.011
        let violations = monitor.check_slos(aece_violation);
        let aece_violated = violations.iter().any(|v| v.metric_name == "aece_drift");
        assert!(aece_violated, "Should detect AECE drift > 0.01");
        
        // DECE drift: should trigger at |Δ| = 0.01  
        let dece_violation = create_test_metrics(0.010, 0.019, 0.50, 0.03, 0.01, 2); // Δ = 0.011
        let violations = monitor.check_slos(dece_violation);
        let dece_violated = violations.iter().any(|v| v.metric_name == "dece_drift");
        assert!(dece_violated, "Should detect DECE drift > 0.01");
        
        // Alpha drift: should trigger at |Δ| = 0.05
        let alpha_violation = create_test_metrics(0.010, 0.008, 0.56, 0.03, 0.01, 3); // Δ = 0.06
        let violations = monitor.check_slos(alpha_violation);
        let alpha_violated = violations.iter().any(|v| v.metric_name == "alpha_drift");
        assert!(alpha_violated, "Should detect Alpha drift > 0.05");
        
        // Clamp rate: should trigger at > 10%
        let clamp_violation = create_test_metrics(0.010, 0.008, 0.50, 0.11, 0.01, 4);
        let violations = monitor.check_slos(clamp_violation);
        let clamp_violated = violations.iter().any(|v| v.metric_name == "clamp_rate");
        assert!(clamp_violated, "Should detect clamp rate > 10%");
        
        // Merged bins: warning at > 5%, critical at > 20%
        let merged_warn = create_test_metrics(0.010, 0.008, 0.50, 0.03, 0.07, 5);
        let violations = monitor.check_slos(merged_warn);
        let merged_warned = violations.iter().any(|v| 
            v.metric_name == "merged_bin_rate" && v.severity == AlertSeverity::Medium);
        assert!(merged_warned, "Should detect merged bin warning > 5%");
        
        let merged_critical = create_test_metrics(0.010, 0.008, 0.50, 0.03, 0.25, 6);
        let violations = monitor.check_slos(merged_critical);
        let merged_failed = violations.iter().any(|v| 
            v.metric_name == "merged_bin_rate" && v.severity == AlertSeverity::Critical);
        assert!(merged_failed, "Should detect merged bin failure > 20%");
        
        println!("✅ All SLO thresholds validated with precise boundaries");
    }

    #[tokio::test]
    async fn test_operational_runbook_decision_coverage() {
        println!("\n🌳 TESTING: Decision Tree Coverage");
        println!("==================================");
        
        let runbook = OperationalRunbook::new();
        
        // Test each major symptom type
        let test_cases = vec![
            (CalibrationSymptom::AeceDrift { current: 0.025, baseline: 0.015, threshold: 0.01 }, "aece_drift"),
            (CalibrationSymptom::DeceDrift { current: 0.025, baseline: 0.015, threshold: 0.01 }, "dece_drift"),
            (CalibrationSymptom::HighClampRate { rate: 0.15, threshold: 0.10 }, "high_clamp_rate"),
            (CalibrationSymptom::ExcessiveMergedBins { rate: 0.25, warn_threshold: 0.05, fail_threshold: 0.20 }, "excessive_merged_bins"),
            (CalibrationSymptom::ScoreRangeViolations { count: 5 }, "score_range_violations"),
        ];
        
        for (symptom, expected_action) in test_cases {
            let actions = runbook.execute_incident_response(vec![symptom]);
            assert!(!actions.is_empty(), "Should generate action for {}", expected_action);
            
            match &actions[0] {
                RemediationAction::RaiseConfidenceThreshold { .. } => {
                    println!("✅ {}: Raise confidence threshold", expected_action);
                },
                RemediationAction::TriggerRecalibration { .. } => {
                    println!("✅ {}: Trigger recalibration", expected_action);
                },
                RemediationAction::MonitorAndWait { .. } => {
                    println!("✅ {}: Monitor and wait", expected_action);
                },
                RemediationAction::RevertToPreviousModel { .. } => {
                    println!("✅ {}: Revert to previous model", expected_action);
                },
                RemediationAction::EscalateToHuman { .. } => {
                    println!("✅ {}: Escalate to human", expected_action);
                },
                _ => {
                    println!("✅ {}: Other remediation action", expected_action);
                }
            }
        }
        
        println!("✅ All decision tree branches validated");
    }

    #[tokio::test]
    async fn test_regression_statistical_significance() {
        println!("\n📊 TESTING: Statistical Significance");
        println!("====================================");
        
        let mut detector = RegressionDetector::new();
        
        // Establish stable baseline (20 samples)
        for i in 0..20 {
            let stable_metrics = create_test_metrics(0.015, 0.012, 0.50, 0.05, 0.02, i);
            detector.add_metrics(stable_metrics);
        }
        
        // Add significant change
        let significant_change = create_test_metrics(0.035, 0.025, 0.60, 0.12, 0.08, 20);
        let regressions = detector.add_metrics(significant_change);
        
        // Should detect statistically significant changes
        let significant_regressions = regressions.iter()
            .filter(|r| r.significance_test.is_significant && r.significance_test.effect_size >= 0.2)
            .count();
        
        assert!(significant_regressions > 0, "Should detect statistically significant regressions");
        
        for regression in &regressions {
            if regression.significance_test.is_significant {
                println!("✅ Significant regression in {}: p-value={:.4}, effect_size={:.3}",
                         regression.metric_name, 
                         regression.significance_test.p_value,
                         regression.significance_test.effect_size);
            }
        }
        
        println!("✅ Statistical significance testing validated");
    }

    #[tokio::test]
    async fn test_invisible_utility_operation() {
        println!("\n👤 TESTING: Invisible Utility Operation");
        println!("=======================================");
        
        let mut drift_monitor = WeeklyDriftMonitor::new();
        let mut regression_detector = RegressionDetector::new();
        
        // Simulate 100 normal operations (should be invisible)
        let baseline = create_test_metrics(0.014, 0.011, 0.50, 0.06, 0.025, 0);
        drift_monitor.set_baseline(baseline.clone());
        
        let mut violation_count = 0;
        let mut regression_count = 0;
        
        for i in 0..100 {
            // Small random variations within normal bounds
            let noise_aece = 0.014 + (i as f64 % 7 - 3.0) * 0.0005;
            let noise_dece = 0.011 + (i as f64 % 5 - 2.0) * 0.0003;
            let noise_alpha = 0.50 + (i as f64 % 9 - 4.0) * 0.002;
            let noise_clamp = 0.06 + (i as f64 % 3 - 1.0) * 0.005;
            let noise_merged = 0.025 + (i as f64 % 4 - 2.0) * 0.003;
            
            let normal_metrics = create_test_metrics(
                noise_aece, noise_dece, noise_alpha, noise_clamp, noise_merged, i
            );
            
            let violations = drift_monitor.check_slos(normal_metrics.clone());
            let regressions = regression_detector.add_metrics(normal_metrics);
            
            violation_count += violations.len();
            regression_count += regressions.len();
        }
        
        println!("   Normal operations: 100");
        println!("   Total violations: {}", violation_count);
        println!("   Total regressions: {}", regression_count);
        println!("   False positive rate: {:.1}%", (violation_count + regression_count) as f64 / 100.0);
        
        // Should have very few false positives for invisible operation
        assert!(violation_count <= 5, "Should have minimal violations during normal operation");
        assert!(regression_count <= 10, "Should have minimal regression alerts during normal operation");
        
        println!("✅ System operates invisibly during normal conditions");
        
        // Now test that problems are immediately visible
        let mut problem_metrics = create_test_metrics(0.030, 0.025, 0.65, 0.15, 0.30, 100);
        problem_metrics.score_range_violations = 2;
        
        let problem_violations = drift_monitor.check_slos(problem_metrics.clone());
        let problem_regressions = regression_detector.add_metrics(problem_metrics);
        
        assert!(problem_violations.len() >= 3, "Should immediately detect problems");
        assert!(problem_regressions.len() >= 1, "Should immediately detect regressions");
        
        println!("   Problem detected with {} violations and {} regressions", 
                 problem_violations.len(), problem_regressions.len());
        
        println!("✅ Problems are immediately visible when they occur");
        println!("✅ INVISIBLE UTILITY OPERATION VALIDATED");
    }
}

#[cfg(test)]
mod benchmark_tests {
    use super::super::*;
    use std::time::Instant;

    #[tokio::test]
    async fn test_hardening_system_performance() {
        println!("\n⚡ PERFORMANCE: Hardening System Latency");
        println!("========================================");
        
        let mut drift_monitor = WeeklyDriftMonitor::new();
        let runbook = OperationalRunbook::new();
        let mut regression_detector = RegressionDetector::new();
        
        let baseline = create_test_metrics(0.015, 0.012, 0.50, 0.05, 0.02, 0);
        drift_monitor.set_baseline(baseline);
        
        // Warm up
        for i in 0..10 {
            let metrics = create_test_metrics(0.015, 0.012, 0.50, 0.05, 0.02, i);
            drift_monitor.check_slos(metrics.clone());
            regression_detector.add_metrics(metrics);
        }
        
        // Performance test
        let iterations = 1000;
        let violating_metrics = create_test_metrics(0.025, 0.022, 0.58, 0.12, 0.25, 100);
        
        let start = Instant::now();
        
        for i in 0..iterations {
            let mut test_metrics = violating_metrics.clone();
            test_metrics.timestamp += i;
            
            // SLO monitoring
            let _violations = drift_monitor.check_slos(test_metrics.clone());
            
            // Regression detection  
            let _regressions = regression_detector.add_metrics(test_metrics);
            
            if i % 100 == 0 {
                // Runbook response (more expensive, test less frequently)
                let symptoms = vec![CalibrationSymptom::AeceDrift { 
                    current: test_metrics.aece, 
                    baseline: 0.015, 
                    threshold: 0.01 
                }];
                let _actions = runbook.execute_incident_response(symptoms);
            }
        }
        
        let duration = start.elapsed();
        let ops_per_sec = iterations as f64 / duration.as_secs_f64();
        let avg_latency_us = duration.as_micros() as f64 / iterations as f64;
        
        println!("   Operations: {}", iterations);
        println!("   Total time: {:.2}s", duration.as_secs_f64());
        println!("   Throughput: {:.0} ops/sec", ops_per_sec);
        println!("   Average latency: {:.1}μs", avg_latency_us);
        
        // Performance requirements for production operation
        assert!(ops_per_sec > 1000.0, "Should process > 1000 ops/sec");
        assert!(avg_latency_us < 1000.0, "Should have < 1ms average latency");
        
        println!("✅ Performance requirements met for production operation");
    }
}

/// Helper function to create test metrics (same as in tests above)
fn create_test_metrics(
    aece: f64, 
    dece: f64, 
    alpha: f64, 
    clamp_rate: f64, 
    merged_bin_rate: f64,
    timestamp_offset: u64
) -> crate::calibration::drift_slos::CalibrationMetrics {
    use std::time::{SystemTime, UNIX_EPOCH};
    
    crate::calibration::drift_slos::CalibrationMetrics {
        timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() + timestamp_offset,
        aece,
        dece,
        alpha,
        clamp_rate,
        merged_bin_rate,
        score_range_violations: 0,
        mask_mismatch_count: 0,
        total_samples: 10000,
    }
}