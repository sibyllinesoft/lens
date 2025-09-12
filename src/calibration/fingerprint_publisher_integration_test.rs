//! Integration tests for green fingerprint publishing system

#[cfg(test)]
mod tests {
    use super::super::{
        FingerprintPublisher, FingerprintPublisherConfig, FingerprintStatus,
        AttestationService, SloSystem, RegressionDetector, CalibrationManifest,
        ConfigurationFingerprint, Phase4Config,
        feature_flags::CalibV22Config,
    };
    use chrono::{DateTime, Utc};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_fingerprint_publishing_workflow() {
        // Mock services (in real implementation these would be injected)
        let attestation_service = Arc::new(create_mock_attestation_service());
        let slo_system = Arc::new(create_mock_slo_system());
        let regression_detector = Arc::new(create_mock_regression_detector());

        let config = FingerprintPublisherConfig::default();
        let publisher = FingerprintPublisher::new(
            config,
            attestation_service,
            slo_system,
            regression_detector,
        );

        // Create test manifest
        let manifest = create_test_manifest();

        // Start validation session
        let session_id = publisher.start_validation_session(manifest)
            .await
            .expect("Failed to start validation session");

        assert!(!session_id.is_empty());

        // Simulate metric collection
        for i in 0..10 {
            publisher.collect_validation_metrics(
                &session_id,
                0.010 + (i as f64) * 0.0001, // Gradual ECE increase
                0.0005, // Stable drift rate
            ).await.expect("Failed to collect metrics");
        }

        // Check that session is still active (not enough time elapsed)
        let completion = publisher.check_validation_completion(&session_id)
            .await
            .expect("Failed to check completion");
        
        assert!(completion.is_none(), "Session should not be complete yet");
    }

    #[tokio::test]
    async fn test_fingerprint_status_validation() {
        let status = FingerprintStatus::Validating {
            start_time: Utc::now(),
            validation_duration: chrono::Duration::hours(24),
        };

        assert!(!status.is_green());
        assert_eq!(status.stability_score(), 0.0);

        let green_status = FingerprintStatus::Green {
            validated_at: Utc::now(),
            stability_metrics: create_test_stability_metrics(),
        };

        assert!(green_status.is_green());
        assert!(green_status.stability_score() > 0.9);
    }

    fn create_mock_attestation_service() -> AttestationService {
        // Create a minimal mock attestation service
        AttestationService::new(
            crate::calibration::attestation::AttestationConfig {
                authority_name: "test-authority".to_string(),
                key_storage_path: std::path::PathBuf::from("/tmp/test-keys"),
                certificate_chain_path: std::path::PathBuf::from("/tmp/test-certs"),
                hsm_enabled: false,
                key_rotation_days: 365,
                signature_algorithm: "Ed25519".to_string(),
            },
        ).expect("Failed to create mock attestation service")
    }

    fn create_mock_slo_system() -> SloSystem {
        SloSystem::new(crate::calibration::SloConfig::default())
            .expect("Failed to create mock SLO system")
    }

    fn create_mock_regression_detector() -> RegressionDetector {
        RegressionDetector::new(Default::default())
    }

    fn create_test_manifest() -> CalibrationManifest {
        use std::collections::HashMap;
        
        CalibrationManifest {
            manifest_version: "test-1.0.0".to_string(),
            generated_at: Utc::now(),
            environment: "test".to_string(),
            deployment_id: "test-deployment-001".to_string(),
            component_versions: HashMap::new(),
            compatibility_matrix: HashMap::new(),
            config_fingerprint: ConfigurationFingerprint {
                config_hash: "test-hash-12345".to_string(),
                rust_lib_hash: "rust-hash-12345".to_string(),
                wasm_hash: "wasm-hash-12345".to_string(),
                typescript_glue_hash: "ts-hash-12345".to_string(),
                quantile_policy_hash: "quantile-hash-12345".to_string(),
                float_rounding_hash: "float-hash-12345".to_string(),
                bootstrap_settings_hash: "bootstrap-hash-12345".to_string(),
                generated_at: Utc::now(),
                algorithm: "SHA-256".to_string(),
            },
            phase4_config: Phase4Config::default(),
            feature_flags: CalibV22Config::default(),
            float_rounding: crate::calibration::FloatRoundingConfig::default(),
            quantile_policy: crate::calibration::QuantilePolicyConfig::default(),
            bootstrap_config: crate::calibration::BootstrapConfig::default(),
            sbom: Vec::new(),
            sbom_tool_version: "test-sbom-1.0".to_string(),
            sbom_signature: None,
            integrity_status: crate::calibration::IntegrityStatus::Valid,
            validation_results: HashMap::new(),
            security_summary: crate::calibration::SecuritySummary {
                overall_score: 95.0,
                total_vulnerabilities: 0,
                high_severity_count: 0,
                medium_severity_count: 0,
                low_severity_count: 0,
                last_scan_time: Utc::now(),
                scanner_version: "test-scanner-1.0".to_string(),
            },
            change_log: Vec::new(),
            approval_chain: Vec::new(),
            attestations: HashMap::new(),
        }
    }

    fn create_test_stability_metrics() -> crate::calibration::StabilityMetrics {
        crate::calibration::StabilityMetrics {
            mean_ece: 0.012,
            max_ece: 0.014,
            ece_std: 0.001,
            mean_drift_rate: 0.0003,
            max_drift: 0.0008,
            slo_breaches: 0,
            alert_count: 1,
            regression_count: 0,
            stability_score: 0.95,
            validation_period: (
                Utc::now() - chrono::Duration::hours(24),
                Utc::now(),
            ),
        }
    }
}