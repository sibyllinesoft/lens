//! # Calibration Manifest System Demonstration
//!
//! This example demonstrates how to use the calibration manifest system for deployment
//! governance, including configuration fingerprinting, SBOM generation, and integrity validation.

use lens_core::calibration::{
    CalibrationManifest, SbomEntry, SbomComponentType, Phase4Config,
    feature_flags::{CalibV22Config, BucketStrategy, BucketMethod, SlaGateConfig, AutoRevertConfig, PromotionCriteria},
    drift_monitor::HealthStatus,
    SecuritySummary, ComplianceStatus, ConfigurationChange, ApprovalEntry, ApprovalLevel,
};
use chrono::Utc;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🔧 Calibration Manifest System Demo");
    println!("==================================\n");

    // 1. Create base configurations
    let phase4_config = Phase4Config::default();
    let feature_flags = create_sample_feature_flags();
    
    println!("✅ Created base configurations");
    println!("   Phase 4 Target ECE: {:.4}", phase4_config.target_ece);
    println!("   Feature Flags Enabled: {}", feature_flags.enabled);
    println!("   Rollout Percentage: {}%\n", feature_flags.rollout_percentage);

    // 2. Create a new calibration manifest
    let mut manifest = CalibrationManifest::new(
        "production".to_string(),
        "lens-prod-2025-09-11-001".to_string(),
        phase4_config,
        feature_flags,
    )?;
    
    println!("✅ Created calibration manifest");
    println!("   Deployment ID: {}", manifest.deployment_id);
    println!("   Environment: {}", manifest.environment);
    println!("   Config Hash: {}...", &manifest.config_fingerprint.config_hash[..16]);
    println!("   Generated At: {}\n", manifest.generated_at);

    // 3. Add SBOM entries
    add_sample_sbom_entries(&mut manifest)?;
    println!("✅ Added {} SBOM entries\n", manifest.sbom.len());

    // 4. Add security scan results
    let security_summary = SecuritySummary {
        overall_score: 92.5,
        total_vulnerabilities: 3,
        high_severity_count: 0,
        medium_severity_count: 2,
        low_severity_count: 1,
        last_scan: Utc::now(),
        compliance_status: ComplianceStatus::MinorIssues { issue_count: 2 },
    };
    manifest.update_security_summary(security_summary);
    println!("✅ Updated security summary");
    println!("   Security Score: {:.1}/100", manifest.security_summary.overall_score);
    println!("   Vulnerabilities: {} total", manifest.security_summary.total_vulnerabilities);
    println!("   Compliance: {}\n", manifest.security_summary.compliance_status);

    // 5. Add configuration change history
    let change = ConfigurationChange {
        timestamp: Utc::now(),
        changed_by: "deployment-system".to_string(),
        description: "Updated ECE target threshold".to_string(),
        section: "phase4_config".to_string(),
        previous_hash: Some("abc123".to_string()),
        new_hash: "def456".to_string(),
        justification: "Improving calibration accuracy based on latest research".to_string(),
        ticket_reference: Some("LENS-1234".to_string()),
    };
    manifest.add_configuration_change(change);
    println!("✅ Added configuration change history\n");

    // 6. Add approval chain
    let approval = ApprovalEntry {
        approver: "senior-engineer@company.com".to_string(),
        approved_at: Utc::now(),
        approval_level: ApprovalLevel::Technical,
        signature: "tech-approval-signature-abc123".to_string(),
        comments: Some("Configuration changes reviewed and approved".to_string()),
    };
    manifest.add_approval(approval)?;
    println!("✅ Added technical approval\n");

    // 7. Validate manifest integrity
    manifest.validate_integrity()?;
    println!("✅ Validated manifest integrity");
    println!("   Status: {}", manifest.integrity_status);
    println!("   Validation Results: {}/{}",
             manifest.validation_results.values().filter(|v| v.passed).count(),
             manifest.validation_results.len());
    
    // Show detailed validation results
    for (component, result) in &manifest.validation_results {
        let status = if result.passed { "✅" } else { "❌" };
        println!("   {} {}: {}", status, component, result.message);
    }
    println!();

    // 8. Generate manifest summary
    let summary = manifest.get_summary();
    println!("✅ Generated manifest summary");
    println!("   Components: {}", summary.component_count);
    println!("   SBOM Entries: {}", summary.sbom_entry_count);
    println!("   Validation Status: {}", if summary.validation_status { "PASSED" } else { "FAILED" });
    println!("   Approvals: {}\n", summary.approval_count);

    // 9. Export to JSON
    let json_manifest = manifest.to_json()?;
    println!("✅ Exported manifest to JSON");
    println!("   JSON size: {} bytes", json_manifest.len());
    println!("   First 200 chars: {}...\n", &json_manifest[..200.min(json_manifest.len())]);

    // 10. Demonstrate round-trip serialization
    let imported_manifest = CalibrationManifest::from_json(&json_manifest)?;
    assert_eq!(manifest.deployment_id, imported_manifest.deployment_id);
    assert_eq!(manifest.config_fingerprint.config_hash, imported_manifest.config_fingerprint.config_hash);
    println!("✅ Verified round-trip JSON serialization\n");

    // 11. Show configuration fingerprint details
    println!("🔍 Configuration Fingerprint Details:");
    println!("   Algorithm: {}", manifest.config_fingerprint.algorithm);
    println!("   Config Hash: {}", manifest.config_fingerprint.config_hash);
    println!("   Rust Lib Hash: {}", manifest.config_fingerprint.rust_lib_hash);
    println!("   WASM Hash: {}", manifest.config_fingerprint.wasm_hash);
    println!("   Quantile Policy Hash: {}", manifest.config_fingerprint.quantile_policy_hash);
    println!("   Bootstrap Settings Hash: {}", manifest.config_fingerprint.bootstrap_settings_hash);
    println!();

    println!("🎉 Calibration Manifest Demo Complete!");
    println!("The manifest system provides comprehensive deployment governance with:");
    println!("   • Complete configuration fingerprinting");
    println!("   • Software Bill of Materials (SBOM) tracking");
    println!("   • Security vulnerability management"); 
    println!("   • Approval workflow enforcement");
    println!("   • Audit trail maintenance");
    println!("   • Integrity validation");
    
    Ok(())
}

fn create_sample_feature_flags() -> CalibV22Config {
    CalibV22Config {
        enabled: true,
        rollout_percentage: 75,
        rollout_stage: "major".to_string(),
        bucket_strategy: BucketStrategy {
            method: BucketMethod::RepositoryHash { salt: "prod-salt-2025".to_string() },
            bucket_salt: "prod-salt-2025".to_string(),
            sticky_sessions: true,
            override_buckets: HashMap::new(),
        },
        sla_gates: SlaGateConfig {
            max_p99_latency_increase_us: 500.0,
            max_aece_tau_threshold: 0.015,
            max_confidence_shift: 0.05,
            require_zero_sla_recall_change: true,
            evaluation_window_minutes: 30,
            consecutive_breach_threshold: 2,
        },
        auto_revert_config: AutoRevertConfig {
            enabled: true,
            breach_window_threshold: 3,
            breach_window_duration_minutes: 45,
            revert_cooldown_minutes: 120,
            max_reverts_per_day: 2,
        },
        config_fingerprint: "prod-config-2025-09-11".to_string(),
        rollout_start_time: Utc::now(),
        promotion_criteria: PromotionCriteria {
            min_observation_hours: 48,
            required_health_status: HealthStatus::Green,
            max_aece_degradation: 0.003,
            require_p99_compliance: true,
            min_success_rate: 0.995,
        },
    }
}

fn add_sample_sbom_entries(manifest: &mut CalibrationManifest) -> anyhow::Result<()> {
    let entries = vec![
        SbomEntry {
            name: "tokio".to_string(),
            version: "1.40.0".to_string(),
            license: Some("MIT".to_string()),
            source_url: Some("https://github.com/tokio-rs/tokio".to_string()),
            hash: CalibrationManifest::sha256_hash("tokio-1.40.0-production-build"),
            component_type: SbomComponentType::RustLibrary,
            security_scan: None,
        },
        SbomEntry {
            name: "serde".to_string(),
            version: "1.0.210".to_string(),
            license: Some("MIT OR Apache-2.0".to_string()),
            source_url: Some("https://github.com/serde-rs/serde".to_string()),
            hash: CalibrationManifest::sha256_hash("serde-1.0.210-production-build"),
            component_type: SbomComponentType::RustLibrary,
            security_scan: None,
        },
        SbomEntry {
            name: "tantivy".to_string(),
            version: "0.22.0".to_string(),
            license: Some("MIT".to_string()),
            source_url: Some("https://github.com/quickwit-oss/tantivy".to_string()),
            hash: CalibrationManifest::sha256_hash("tantivy-0.22.0-production-build"),
            component_type: SbomComponentType::RustLibrary,
            security_scan: None,
        },
        SbomEntry {
            name: "lens-calibration-wasm".to_string(),
            version: "1.0.0".to_string(),
            license: Some("MIT".to_string()),
            source_url: Some("https://github.com/sibyllinesoft/lens".to_string()),
            hash: CalibrationManifest::sha256_hash("lens-calibration-wasm-1.0.0"),
            component_type: SbomComponentType::WasmModule,
            security_scan: None,
        },
        SbomEntry {
            name: "lens-ts-glue".to_string(),
            version: "1.0.0".to_string(),
            license: Some("MIT".to_string()),
            source_url: Some("https://github.com/sibyllinesoft/lens".to_string()),
            hash: CalibrationManifest::sha256_hash("lens-ts-glue-1.0.0"),
            component_type: SbomComponentType::TypeScriptGlue,
            security_scan: None,
        },
    ];

    for entry in entries {
        manifest.add_sbom_entry(entry)?;
    }

    Ok(())
}