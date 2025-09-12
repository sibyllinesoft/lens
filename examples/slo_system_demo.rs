//! # SLO System Demonstration
//!
//! This example demonstrates the comprehensive SLO system and dashboard
//! for calibration governance with real-time monitoring and alerting.

use lens::calibration::{
    SloSystem, SloDashboard, CalibrationGovernance, GovernanceConfig,
    CalibrationMonitor, MonitoringConfig, AlertConfig, SloType,
    drift_monitor::DriftMonitor,
    slo_dashboard::ExportFormat,
};
use std::sync::Arc;
use tokio::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    tracing_subscriber::init();
    
    println!("🔧 SLO System & Dashboard Demonstration");
    println!("========================================");
    
    // Step 1: Create the governance system
    println!("\n📋 Step 1: Creating Calibration Governance System...");
    let governance_config = GovernanceConfig::default();
    let governance = Arc::new(CalibrationGovernance::new(governance_config).await?);
    println!("✅ Governance system created successfully");
    
    // Step 2: Create the monitoring system
    println!("\n📊 Step 2: Creating Calibration Monitor...");
    let monitor_config = MonitoringConfig {
        target_ece: 0.015,
        alert_config: AlertConfig::default(),
        realtime_enabled: true,
    };
    let monitor = Arc::new(CalibrationMonitor::new(monitor_config).await?);
    println!("✅ Monitor created successfully");
    
    // Step 3: Create the drift monitor
    println!("\n📈 Step 3: Creating Drift Monitor...");
    let drift_monitor = Arc::new(DriftMonitor::new().await?);
    println!("✅ Drift monitor created successfully");
    
    // Step 4: Create the SLO system
    println!("\n🎯 Step 4: Creating SLO System...");
    let slo_system = Arc::new(SloSystem::new(governance, monitor, drift_monitor));
    println!("✅ SLO system created successfully");
    
    // Step 5: Create the dashboard
    println!("\n📱 Step 5: Creating SLO Dashboard...");
    let dashboard = SloDashboard::new(slo_system.clone());
    println!("✅ Dashboard created successfully");
    
    // Step 6: Start the SLO monitoring system
    println!("\n🚀 Step 6: Starting SLO Monitoring...");
    slo_system.start_monitoring().await?;
    println!("✅ SLO monitoring started");
    
    // Step 7: Start the dashboard
    println!("\n📺 Step 7: Starting Dashboard...");
    dashboard.start().await?;
    println!("✅ Dashboard started");
    
    // Step 8: Wait a moment for measurements to be taken
    println!("\n⏳ Step 8: Allowing time for measurements...");
    tokio::time::sleep(Duration::from_secs(5)).await;
    
    // Step 9: Get current SLO state
    println!("\n📋 Step 9: Retrieving SLO Status...");
    let slo_state = slo_system.get_slo_state().await;
    println!("System Health: {:?}", slo_state.overall_health);
    println!("Total SLOs Monitored: {}", slo_state.slo_status.len());
    
    for (slo_type, status) in &slo_state.slo_status {
        println!("  📊 {}: {} (target: {:.3}, current: {:.3})", 
                slo_type.name(),
                match status.status {
                    lens::calibration::slo_system::SloHealthStatus::Meeting => "✅ MEETING",
                    lens::calibration::slo_system::SloHealthStatus::Warning => "⚠️  WARNING", 
                    lens::calibration::slo_system::SloHealthStatus::Breached => "❌ BREACHED",
                    lens::calibration::slo_system::SloHealthStatus::Error => "🔥 ERROR",
                    lens::calibration::slo_system::SloHealthStatus::Unknown => "❓ UNKNOWN",
                },
                status.target_value,
                status.current_value
        );
    }
    
    // Step 10: Generate comprehensive report
    println!("\n📄 Step 10: Generating SLO Report...");
    let report = slo_system.generate_report().await?;
    println!("Report timestamp: {}", report.timestamp);
    println!("Overall health: {:?}", report.overall_health);
    println!("Active alerts: {}", report.active_alerts.len());
    
    // Step 11: Test dashboard export
    println!("\n💾 Step 11: Testing Dashboard Export...");
    let json_export = dashboard.export(ExportFormat::Json, None).await?;
    println!("JSON export size: {} bytes", json_export.len());
    
    // Step 12: Display SLO coverage
    println!("\n🔍 Step 12: SLO Coverage Summary");
    println!("The following SLOs are monitored:");
    println!("  1. 📊 ECE Bound: ≤ max(0.015, ĉ√(K/N))");
    println!("  2. 🔒 Clamp Activation: ≤10% across all slices");
    println!("  3. 📦 Merged Bin Warning: ≤5% (warning), >20% (failure)");
    println!("  4. ⚡ P99 Latency: <1ms for calibration operations");
    println!("  5. 📈 Weekly Drift Deltas: All metrics <0.01 absolute change");
    println!("  6. ✅ Score Range Validation: All scores ∈ [0,1]");
    println!("  7. 🔧 Clamp Alpha Changes: <0.05 week-over-week");
    
    // Step 13: Show dashboard capabilities
    println!("\n🎛️  Step 13: Dashboard Capabilities");
    let dashboard_state = dashboard.get_dashboard_state().await;
    println!("Dashboard last updated: {}", dashboard_state.last_updated);
    
    let visualizations = dashboard.get_visualizations().await;
    println!("Available visualizations:");
    println!("  • SLO Overview (system health, status cards, metrics summary)");
    println!("  • Reliability Charts (time-series compliance, breach markers)");
    println!("  • Calibration Tables (per-bin analysis, mask mismatch detection)");
    println!("  • Drift Analysis (weekly deltas, trend analysis, predictions)");
    println!("  • Alert Panel (active alerts, statistics, escalation tracking)");
    
    println!("\n🎉 SLO System & Dashboard Demo Complete!");
    println!("=====================================");
    println!("The system is now running and monitoring {} SLOs with comprehensive", slo_state.slo_status.len());
    println!("real-time dashboards, alerting, and trend analysis capabilities.");
    println!("\nKey Features Demonstrated:");
    println!("✅ Production-ready SLO monitoring with 7 comprehensive SLOs");
    println!("✅ Real-time dashboard with interactive visualizations");
    println!("✅ Automated alerting and escalation policies");
    println!("✅ Weekly drift detection and trend analysis");
    println!("✅ Comprehensive reporting and export capabilities");
    println!("✅ Integration with existing governance and monitoring systems");
    
    Ok(())
}