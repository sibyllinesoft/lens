//! Demonstration of the green fingerprint publishing and public SLO manifest system
//! 
//! This example shows how to:
//! 1. Generate public calibration SLO documentation
//! 2. Create publisher-ready fingerprints with attestation
//! 3. Validate stability metrics and publish green fingerprints

use lens_core::calibration::{
    PublicCalibrationSlo, PublicManifestGenerator, CalibrationManifest, ConfigurationFingerprint,
    Phase4Config, feature_flags::CalibV22Config, StabilityMetrics,
    FingerprintStatus, FloatRoundingConfig, QuantilePolicyConfig, BootstrapConfig,
    IntegrityStatus, SecuritySummary,
};
use std::collections::HashMap;
use chrono::Utc;

fn main() -> anyhow::Result<()> {
    println!("🔐 Green Fingerprint Publishing & Public SLO Manifest Demo");
    println!("=========================================================\n");

    // Create a sample calibration manifest
    let manifest = create_sample_manifest();
    println!("✅ Created calibration manifest: {}", manifest.manifest_version);

    // Create sample stability metrics from 24-hour validation
    let stability_metrics = create_sample_stability_metrics();
    println!("📊 Stability metrics: ECE={:.4}, Drift={:.6}, Score={:.3}",
        stability_metrics.mean_ece,
        stability_metrics.mean_drift_rate,
        stability_metrics.stability_score,
    );

    // Generate public SLO manifest
    let generator = PublicManifestGenerator::new();
    let public_slo = generator.generate_public_manifest(&manifest, Some(&stability_metrics))?;

    println!("\n📋 Generated Public SLO Contract:");
    println!("   Version: {}", public_slo.contract_version);
    println!("   ECE Threshold: {:.4}", public_slo.performance_guarantees.ece_guarantees.base_threshold);
    println!("   τ Formula: {}", public_slo.performance_guarantees.ece_guarantees.adaptive_threshold_formula);

    // Generate human-readable documentation
    let documentation = generator.generate_documentation(&public_slo)?;
    println!("\n📖 Generated Documentation Preview:");
    println!("{}", &documentation[0..500]);
    println!("   ... (truncated)");

    // Create green fingerprint status
    let green_status = FingerprintStatus::Green {
        validated_at: Utc::now(),
        stability_metrics,
    };

    println!("\n🟢 Green Fingerprint Status:");
    println!("   Is Green: {}", green_status.is_green());
    println!("   Stability Score: {:.3}", green_status.stability_score());

    // Show τ formula calculation example
    demonstrate_tau_formula();

    println!("\n🎯 Integration Points:");
    println!("   • Fingerprint Publisher: Validates 24h stability → publishes green fingerprints");
    println!("   • Public Manifest: Generates transparent SLO documentation for consumers");
    println!("   • Attestation System: Cryptographic signatures ensure integrity");
    println!("   • Repository Publishing: Automated distribution to public repositories");

    Ok(())
}

fn create_sample_manifest() -> CalibrationManifest {
    CalibrationManifest {
        manifest_version: "v2.1.0-stable".to_string(),
        created_at: Utc::now(),
        config_fingerprint: ConfigurationFingerprint {
            config_hash: "a7f8c9d2e1b5f3a8c6d4e9b2f7a1c5e8d3b9f4a6c2e7d1b8f5a3c9e4d2b6f1a4".to_string(),
            rust_lib_hash: "b2f3c8e1a9d6f4b7c5e2d8f1a4c7e3b6d9f2a5c8e4b1d7f3a6c9e2b5f8a1c4".to_string(),
            wasm_hash: "c1d4f7a2b5c8e3d6f9a4b7c2e5d8f1a3b6c9d2f5a8e1b4c7d3f6a9b2c5d8f1a4".to_string(),
            typescript_glue_hash: "d9f2a5c8e1b4d7f3a6c9e2b5f8a1c4d7f3a6c9e2b5f8a1c4d7f3a6c9e2b5f8".to_string(),
            quantile_policy_hash: "e3b6d9f2a5c8e1b4d7f3a6c9e2b5f8a1c4d7f3a6c9e2b5f8a1c4d7f3a6c9e2".to_string(),
            float_rounding_hash: "f1a4c7e3b6d9f2a5c8e1b4d7f3a6c9e2b5f8a1c4d7f3a6c9e2b5f8a1c4d7f3".to_string(),
            bootstrap_settings_hash: "a8e1b4c7d3f6a9b2c5d8f1a4c7e3b6d9f2a5c8e1b4d7f3a6c9e2b5f8a1c4d7".to_string(),
            generated_at: Utc::now(),
            algorithm: "SHA-256".to_string(),
        },
        generated_at: Utc::now(),
        environment: "demo".to_string(),
        deployment_id: "demo-deployment-001".to_string(),
        component_versions: HashMap::new(),
        compatibility_matrix: HashMap::new(),
        phase4_config: Phase4Config::default(),
        feature_flags: CalibV22Config::default(),
        float_rounding: FloatRoundingConfig::default(),
        quantile_policy: QuantilePolicyConfig::default(),
        bootstrap_config: BootstrapConfig::default(),
        sbom: Vec::new(),
        sbom_tool_version: "demo-sbom-1.0".to_string(),
        sbom_signature: None,
        integrity_status: IntegrityStatus::Valid,
        validation_results: HashMap::new(),
        security_summary: SecuritySummary {
            overall_score: 95.0,
            total_vulnerabilities: 0,
            high_severity_count: 0,
            medium_severity_count: 0,
            low_severity_count: 0,
            last_scan_time: Utc::now(),
            scanner_version: "demo-scanner-1.0".to_string(),
        },
        change_log: Vec::new(),
        approval_chain: Vec::new(),
        attestations: HashMap::new(),
    }
}

fn create_sample_stability_metrics() -> StabilityMetrics {
    StabilityMetrics {
        mean_ece: 0.0118,   // Well below 0.015 threshold
        max_ece: 0.0134,    // Peak ECE still within acceptable range
        ece_std: 0.0012,    // Low variability indicates stability
        mean_drift_rate: 0.0003,  // Minimal drift
        max_drift: 0.0007,  // Controlled maximum drift
        slo_breaches: 0,    // No SLO violations during validation
        alert_count: 2,     // Minimal alerts (within threshold)
        regression_count: 0, // No regressions detected
        stability_score: 0.934, // High stability score
        validation_period: (
            Utc::now() - chrono::Duration::hours(24),
            Utc::now(),
        ),
    }
}

fn demonstrate_tau_formula() {
    println!("\n🧮 τ Formula Demonstration: max(0.015, ĉ√(K/N))");
    
    let scenarios = vec![
        ("Small dataset", 1.96, 10, 1000, 0.015),
        ("Medium dataset", 1.96, 10, 10000, 0.015),
        ("Large dataset", 1.96, 10, 100000, 0.015),
        ("Few bins", 1.96, 5, 10000, 0.015),
        ("Many bins", 1.96, 20, 10000, 0.015),
    ];

    for (scenario, c_hat, k, n, base) in scenarios {
        let sqrt_term = ((k as f64) / (n as f64)).sqrt();
        let adaptive_term = c_hat * sqrt_term;
        let tau = base.max(adaptive_term);
        
        println!("   {}: K={}, N={} → τ = max({:.3}, {:.4}) = {:.4}",
            scenario, k, n, base, adaptive_term, tau);
    }
    
    println!("\n   📝 Key Insights:");
    println!("      • Larger datasets (↑N) → lower adaptive threshold");
    println!("      • More bins (↑K) → higher adaptive threshold");
    println!("      • Base threshold (0.015) ensures minimum quality standard");
    println!("      • Formula balances statistical reliability with practical precision");
}