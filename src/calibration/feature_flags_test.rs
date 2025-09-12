//! Simple test to verify feature flag system works

#[cfg(test)]
mod tests {
    use super::super::{
        feature_flags::{CalibV22Config, CalibV22FeatureFlag, RolloutStage},
        sla_tripwires::SlaConfig,
        drift_monitor::DriftThresholds,
        shared_binning_core::SharedBinningConfig,
    };
    
    #[tokio::test]
    async fn test_feature_flag_basic_functionality() {
        let config = CalibV22Config::default();
        let sla_config = SlaConfig::default();
        let drift_thresholds = DriftThresholds::default();
        let binning_config = SharedBinningConfig::default();
        
        let feature_flag = CalibV22FeatureFlag::new(
            config, sla_config, drift_thresholds, binning_config
        ).expect("Failed to create feature flag");
        
        // Test repository decision
        let decision = feature_flag.should_use_calib_v22("test_repo").expect("Failed to get decision");
        
        // Since default config has 0% rollout, should always be false
        assert!(!decision.use_calib_v22);
        assert!(decision.decision_reason.contains("disabled") || decision.rollout_stage == "Disabled");
    }
    
    #[tokio::test] 
    async fn test_rollout_stage_progression() {
        // Test stage percentage mapping
        assert_eq!(RolloutStage::Disabled.get_percentage(), 0);
        assert_eq!(RolloutStage::Canary.get_percentage(), 5);
        assert_eq!(RolloutStage::Limited.get_percentage(), 25);
        assert_eq!(RolloutStage::Major.get_percentage(), 50);
        assert_eq!(RolloutStage::Full.get_percentage(), 100);
        
        // Test stage progression
        assert_eq!(RolloutStage::Canary.next_stage(), Some(RolloutStage::Limited));
        assert_eq!(RolloutStage::Limited.next_stage(), Some(RolloutStage::Major));
        assert_eq!(RolloutStage::Major.next_stage(), Some(RolloutStage::Full));
        assert_eq!(RolloutStage::Full.next_stage(), None);
        assert_eq!(RolloutStage::Disabled.next_stage(), Some(RolloutStage::Canary));
    }
    
    #[tokio::test]
    async fn test_feature_flag_status() {
        let config = CalibV22Config::default();
        let sla_config = SlaConfig::default();
        let drift_thresholds = DriftThresholds::default();
        let binning_config = SharedBinningConfig::default();
        
        let feature_flag = CalibV22FeatureFlag::new(
            config, sla_config, drift_thresholds, binning_config
        ).expect("Failed to create feature flag");
        
        let status = feature_flag.get_status().expect("Failed to get status");
        
        // Verify status structure
        assert!(status["enabled"].is_boolean());
        assert!(status["current_stage"].is_string());
        assert!(status["rollout_percentage"].is_number());
        assert!(status["circuit_breaker_open"].is_boolean());
        assert!(status["metrics"].is_object());
    }
}