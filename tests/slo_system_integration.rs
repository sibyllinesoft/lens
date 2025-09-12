//! # SLO System Integration Tests
//!
//! Comprehensive integration tests for the SLO system and dashboard.
//! Tests the integration between SLO monitoring, governance, and dashboard components.

use lens::calibration::{
    SloSystem, SloConfig, SloDashboard, DashboardConfig,
    CalibrationGovernance, GovernanceConfig, CalibrationMonitor, MonitoringConfig,
    DriftMonitor, Phase4Config, AlertConfig, SloType,
};
use std::sync::Arc;
use tokio::time::Duration;

#[tokio::test]
async fn test_slo_system_basic_creation() {
    // Create governance system
    let governance_config = GovernanceConfig::default();
    let governance = Arc::new(CalibrationGovernance::new(governance_config).await.unwrap());
    
    // Create monitoring system
    let monitor_config = MonitoringConfig {
        target_ece: 0.015,
        alert_config: AlertConfig::default(),
        realtime_enabled: true,
    };
    let monitor = Arc::new(CalibrationMonitor::new(monitor_config).await.unwrap());
    
    // Create drift monitor
    let drift_monitor = Arc::new(DriftMonitor::new().await.unwrap());
    
    // Create SLO system
    let slo_system = SloSystem::new(governance, monitor, drift_monitor);
    
    // Verify system state
    let state = slo_system.get_slo_state().await;
    assert!(!state.slo_status.is_empty());
    
    // Test should pass if system initializes correctly
}

#[tokio::test]
async fn test_slo_dashboard_integration() {
    // Create governance system
    let governance_config = GovernanceConfig::default();
    let governance = Arc::new(CalibrationGovernance::new(governance_config).await.unwrap());
    
    // Create monitoring system
    let monitor_config = MonitoringConfig {
        target_ece: 0.015,
        alert_config: AlertConfig::default(),
        realtime_enabled: true,
    };
    let monitor = Arc::new(CalibrationMonitor::new(monitor_config).await.unwrap());
    
    // Create drift monitor
    let drift_monitor = Arc::new(DriftMonitor::new().await.unwrap());
    
    // Create SLO system
    let slo_system = Arc::new(SloSystem::new(governance, monitor, drift_monitor));
    
    // Create dashboard
    let dashboard = SloDashboard::new(slo_system);
    
    // Test dashboard state
    let dashboard_state = dashboard.get_dashboard_state().await;
    assert!(!dashboard_state.last_updated.timestamp().is_zero());
    
    // Test should pass if dashboard initializes with SLO system
}

#[tokio::test]
async fn test_slo_types_coverage() {
    // Verify all required SLO types are implemented
    let slo_types = vec![
        SloType::EceBound,
        SloType::ClampActivation,
        SloType::MergedBinWarning,
        SloType::P99Latency,
        SloType::WeeklyDriftDeltas,
        SloType::ScoreRangeValidation,
        SloType::ClampAlphaChanges,
    ];
    
    // All SLO types should have names
    for slo_type in slo_types {
        let name = slo_type.name();
        assert!(!name.is_empty());
        println!("SLO Type: {:?} -> Name: {}", slo_type, name);
    }
}

#[tokio::test]
async fn test_slo_configuration_validation() {
    // Test that default SLO configuration is valid
    let governance_config = GovernanceConfig::default();
    let governance = Arc::new(CalibrationGovernance::new(governance_config).await.unwrap());
    
    let monitor_config = MonitoringConfig {
        target_ece: 0.015,
        alert_config: AlertConfig::default(),
        realtime_enabled: true,
    };
    let monitor = Arc::new(CalibrationMonitor::new(monitor_config).await.unwrap());
    
    let drift_monitor = Arc::new(DriftMonitor::new().await.unwrap());
    
    // Create SLO system with default config
    let slo_system = SloSystem::new(governance, monitor, drift_monitor);
    
    // Should initialize successfully with valid configuration
    let state = slo_system.get_slo_state().await;
    
    // Check that all required SLOs are configured
    assert!(state.slo_status.contains_key(&SloType::EceBound));
    assert!(state.slo_status.contains_key(&SloType::ClampActivation));
    assert!(state.slo_status.contains_key(&SloType::P99Latency));
}

#[tokio::test] 
async fn test_dashboard_export_functionality() {
    let governance_config = GovernanceConfig::default();
    let governance = Arc::new(CalibrationGovernance::new(governance_config).await.unwrap());
    
    let monitor_config = MonitoringConfig {
        target_ece: 0.015,
        alert_config: AlertConfig::default(),
        realtime_enabled: true,
    };
    let monitor = Arc::new(CalibrationMonitor::new(monitor_config).await.unwrap());
    
    let drift_monitor = Arc::new(DriftMonitor::new().await.unwrap());
    
    let slo_system = Arc::new(SloSystem::new(governance, monitor, drift_monitor));
    let dashboard = SloDashboard::new(slo_system);
    
    // Test JSON export
    let json_export = dashboard.export(
        lens::calibration::slo_dashboard::ExportFormat::Json, 
        None
    ).await;
    
    assert!(json_export.is_ok());
    let json_data = json_export.unwrap();
    assert!(!json_data.is_empty());
    
    // Verify it's valid JSON
    let parsed: serde_json::Value = serde_json::from_slice(&json_data).unwrap();
    assert!(parsed.is_object());
}