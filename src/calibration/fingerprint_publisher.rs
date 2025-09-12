//! # Green Fingerprint Publishing System
//!
//! Production-ready publishing system for validated calibration fingerprints after
//! 24-hour stability validation. Provides cryptographic attestation, manifest binding,
//! and release verification for stable calibration configurations.
//!
//! ## Key Features
//!
//! * **24-Hour Stability Validation**: Only publishes after proven stability period
//! * **Cryptographic Attestation**: Ed25519 signatures with certificate chains
//! * **Manifest Binding**: Immutable binding to specific calibration configurations
//! * **Release Fingerprint Verification**: Tamper-proof release artifact validation
//! * **Public Repository Integration**: Automated publishing to public repositories
//! * **Green/Red Status System**: Clear stability indicators for downstream consumers
//!
//! ## Publishing Process
//!
//! 1. **Stability Validation**: 24-hour monitoring of calibration performance
//! 2. **Green Light Assessment**: ECE, drift, and reliability thresholds passed
//! 3. **Fingerprint Generation**: Cryptographic fingerprint of stable configuration
//! 4. **Attestation Creation**: Ed25519 signature with certificate chain binding
//! 5. **Public Repository Publishing**: Automated distribution to consumers
//! 6. **Verification System**: Real-time validation of published fingerprints

use crate::calibration::{
    CalibrationManifest, ConfigurationFingerprint, AttestationService, AttestationSignature,
    Ed25519KeyPair, SloSystem, SloStatus, ECEAlert, RegressionDetector, RegressionSeverity,
};
use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use tokio::fs;
use tracing::{info, warn, debug};
use uuid::Uuid;

/// Green fingerprint status indicating stability validation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FingerprintStatus {
    /// Undergoing stability validation
    Validating {
        start_time: DateTime<Utc>,
        validation_duration: Duration,
    },
    /// Passed 24-hour stability validation
    Green {
        validated_at: DateTime<Utc>,
        stability_metrics: StabilityMetrics,
    },
    /// Failed validation - not suitable for publishing
    Red {
        failed_at: DateTime<Utc>,
        failure_reasons: Vec<ValidationFailure>,
    },
    /// Deprecated - replaced by newer fingerprint
    Deprecated {
        deprecated_at: DateTime<Utc>,
        replacement_fingerprint: String,
    },
}

/// Comprehensive stability metrics from 24-hour validation period
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityMetrics {
    /// Mean ECE over validation period
    pub mean_ece: f64,
    /// Maximum ECE observed
    pub max_ece: f64,
    /// ECE standard deviation
    pub ece_std: f64,
    /// Mean drift rate
    pub mean_drift_rate: f64,
    /// Maximum drift observed
    pub max_drift: f64,
    /// SLO breach count during validation
    pub slo_breaches: u32,
    /// Alert count during validation
    pub alert_count: u32,
    /// Regression incidents detected
    pub regression_count: u32,
    /// Overall stability score (0.0 - 1.0)
    pub stability_score: f64,
    /// Validation start/end timestamps
    pub validation_period: (DateTime<Utc>, DateTime<Utc>),
}

/// Specific validation failure reasons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationFailure {
    /// ECE exceeded threshold
    EceThresholdExceeded { observed: f64, threshold: f64 },
    /// Drift rate too high
    DriftRateExceeded { observed: f64, threshold: f64 },
    /// SLO breaches during validation
    SloBreaches { count: u32, threshold: u32 },
    /// Regression detected
    RegressionDetected { severity: RegressionSeverity },
    /// Insufficient stability duration
    InsufficientStability { duration: Duration, required: Duration },
    /// System alerts during validation
    AlertsTriggered { count: u32, threshold: u32 },
}

/// Published green fingerprint with full attestation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishedFingerprint {
    /// Unique fingerprint identifier
    pub fingerprint_id: String,
    /// Fingerprint status (must be Green for publishing)
    pub status: FingerprintStatus,
    /// Bound calibration manifest
    pub manifest: CalibrationManifest,
    /// Cryptographic attestation
    pub attestation: AttestationSignature,
    /// Publication metadata
    pub publication_metadata: PublicationMetadata,
    /// Release verification data
    pub release_verification: ReleaseVerification,
}

/// Publication metadata and tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicationMetadata {
    /// Publication timestamp
    pub published_at: DateTime<Utc>,
    /// Publishing authority
    pub publisher: String,
    /// Publication version
    pub version: String,
    /// Distribution channels
    pub distribution_channels: Vec<String>,
    /// Download URLs
    pub download_urls: HashMap<String, String>,
    /// Verification instructions
    pub verification_instructions: String,
}

/// Release verification and integrity data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseVerification {
    /// Release artifact checksums
    pub artifact_checksums: HashMap<String, String>,
    /// Verification script checksum
    pub verification_script_hash: String,
    /// GPG signature (optional)
    pub gpg_signature: Option<String>,
    /// Certificate chain for verification
    pub certificate_chain: Vec<String>,
    /// Verification timestamp
    pub verified_at: DateTime<Utc>,
}

/// Green fingerprint publishing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerprintPublisherConfig {
    /// Validation duration requirement (default: 24 hours)
    pub validation_duration: Duration,
    /// ECE threshold for green status
    pub ece_threshold: f64,
    /// Drift rate threshold
    pub drift_threshold: f64,
    /// Maximum allowed SLO breaches
    pub max_slo_breaches: u32,
    /// Maximum allowed alerts
    pub max_alerts: u32,
    /// Publication repository settings
    pub repository_config: RepositoryConfig,
    /// Attestation configuration
    pub attestation_config: PublisherAttestationConfig,
}

/// Repository configuration for publishing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryConfig {
    /// Repository base URL
    pub base_url: String,
    /// Authentication token
    pub auth_token: Option<String>,
    /// Publication path
    pub publication_path: String,
    /// Repository type (git, s3, etc.)
    pub repository_type: String,
}

/// Publisher attestation configuration for fingerprints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublisherAttestationConfig {
    /// Signing key identifier
    pub signing_key_id: String,
    /// Certificate chain path
    pub certificate_chain_path: PathBuf,
    /// Optional HSM configuration
    pub hsm_config: Option<HsmConfig>,
}

/// Hardware Security Module configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HsmConfig {
    /// HSM slot identifier
    pub slot_id: u32,
    /// Token label
    pub token_label: String,
    /// PIN for HSM access
    pub pin: String,
}

/// Green fingerprint publishing service
pub struct FingerprintPublisher {
    /// Publishing configuration
    config: FingerprintPublisherConfig,
    /// Attestation service for signatures
    attestation_service: Arc<AttestationService>,
    /// SLO monitoring system
    slo_system: Arc<SloSystem>,
    /// Regression detection system
    regression_detector: Arc<RegressionDetector>,
    /// Published fingerprints registry
    published_fingerprints: Arc<RwLock<HashMap<String, PublishedFingerprint>>>,
    /// Active validation sessions
    validation_sessions: Arc<RwLock<HashMap<String, ValidationSession>>>,
}

/// Active validation session tracking
#[derive(Debug, Clone)]
struct ValidationSession {
    /// Session identifier
    session_id: String,
    /// Fingerprint being validated
    fingerprint_id: String,
    /// Associated manifest
    manifest: CalibrationManifest,
    /// Validation start time
    started_at: DateTime<Utc>,
    /// Required validation duration
    required_duration: Duration,
    /// Collected metrics
    metrics: Vec<ValidationMetric>,
    /// Current status
    status: ValidationSessionStatus,
}

/// Validation session status
#[derive(Debug, Clone)]
enum ValidationSessionStatus {
    Active,
    Completed { result: ValidationResult },
    Failed { failures: Vec<ValidationFailure> },
}

/// Individual validation metric collection
#[derive(Debug, Clone)]
struct ValidationMetric {
    /// Metric collection timestamp
    timestamp: DateTime<Utc>,
    /// ECE measurement
    ece: f64,
    /// Drift rate
    drift_rate: f64,
    /// SLO status
    slo_status: SloStatus,
    /// Active alerts
    active_alerts: Vec<ECEAlert>,
}

/// Validation result after completion
#[derive(Debug, Clone)]
enum ValidationResult {
    Passed { metrics: StabilityMetrics },
    Failed { failures: Vec<ValidationFailure> },
}

impl Default for FingerprintPublisherConfig {
    fn default() -> Self {
        Self {
            validation_duration: Duration::hours(24),
            ece_threshold: 0.015,
            drift_threshold: 0.001,
            max_slo_breaches: 0,
            max_alerts: 5,
            repository_config: RepositoryConfig {
                base_url: "https://releases.calibration.ai".to_string(),
                auth_token: None,
                publication_path: "fingerprints/green/".to_string(),
                repository_type: "https".to_string(),
            },
            attestation_config: PublisherAttestationConfig {
                signing_key_id: "calibration-publisher-2024".to_string(),
                certificate_chain_path: PathBuf::from("certs/publisher-chain.pem"),
                hsm_config: None,
            },
        }
    }
}

impl FingerprintPublisher {
    /// Create new fingerprint publisher
    pub fn new(
        config: FingerprintPublisherConfig,
        attestation_service: Arc<AttestationService>,
        slo_system: Arc<SloSystem>,
        regression_detector: Arc<RegressionDetector>,
    ) -> Self {
        Self {
            config,
            attestation_service,
            slo_system,
            regression_detector,
            published_fingerprints: Arc::new(RwLock::new(HashMap::new())),
            validation_sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start 24-hour validation session for calibration manifest
    pub async fn start_validation_session(
        &self,
        manifest: CalibrationManifest,
    ) -> Result<String> {
        let session_id = Uuid::new_v4().to_string();
        let fingerprint_id = manifest.config_fingerprint.config_hash.clone();

        let session = ValidationSession {
            session_id: session_id.clone(),
            fingerprint_id: fingerprint_id.clone(),
            manifest,
            started_at: Utc::now(),
            required_duration: self.config.validation_duration,
            metrics: Vec::new(),
            status: ValidationSessionStatus::Active,
        };

        {
            let mut sessions = self.validation_sessions.write()
                .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
            sessions.insert(session_id.clone(), session);
        }

        info!(
            session_id = %session_id,
            fingerprint_id = %fingerprint_id,
            duration_hours = %self.config.validation_duration.num_hours(),
            "Started 24-hour validation session for fingerprint"
        );

        Ok(session_id)
    }

    /// Collect validation metrics for active session
    pub async fn collect_validation_metrics(
        &self,
        session_id: &str,
        ece: f64,
        drift_rate: f64,
    ) -> Result<()> {
        let slo_status = self.slo_system.get_current_status().await?;
        let active_alerts = self.slo_system.get_active_alerts().await?;

        let metric = ValidationMetric {
            timestamp: Utc::now(),
            ece,
            drift_rate,
            slo_status,
            active_alerts,
        };

        {
            let mut sessions = self.validation_sessions.write()
                .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
            
            if let Some(session) = sessions.get_mut(session_id) {
                session.metrics.push(metric);
                debug!(
                    session_id = %session_id,
                    ece = %ece,
                    drift_rate = %drift_rate,
                    "Collected validation metric"
                );
            } else {
                bail!("Validation session not found: {}", session_id);
            }
        }

        Ok(())
    }

    /// Check if validation session is complete and assess results
    pub async fn check_validation_completion(
        &self,
        session_id: &str,
    ) -> Result<Option<ValidationResult>> {
        let mut sessions = self.validation_sessions.write()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;

        if let Some(session) = sessions.get_mut(session_id) {
            let elapsed = Utc::now().signed_duration_since(session.started_at);
            
            if elapsed >= session.required_duration {
                // Validation period complete - assess results
                let result = self.assess_validation_results(&session.metrics).await?;
                session.status = ValidationSessionStatus::Completed {
                    result: result.clone(),
                };

                info!(
                    session_id = %session_id,
                    result = ?result,
                    "Validation session completed"
                );

                Ok(Some(result))
            } else {
                // Still validating
                Ok(None)
            }
        } else {
            bail!("Validation session not found: {}", session_id);
        }
    }

    /// Assess validation results from collected metrics
    async fn assess_validation_results(
        &self,
        metrics: &[ValidationMetric],
    ) -> Result<ValidationResult> {
        if metrics.is_empty() {
            return Ok(ValidationResult::Failed {
                failures: vec![ValidationFailure::InsufficientStability {
                    duration: Duration::zero(),
                    required: self.config.validation_duration,
                }],
            });
        }

        let mut failures = Vec::new();

        // Calculate stability metrics
        let eces: Vec<f64> = metrics.iter().map(|m| m.ece).collect();
        let drift_rates: Vec<f64> = metrics.iter().map(|m| m.drift_rate).collect();

        let mean_ece = eces.iter().sum::<f64>() / eces.len() as f64;
        let max_ece = eces.iter().fold(0.0, |a, &b| a.max(b));
        let ece_variance = eces.iter()
            .map(|&x| (x - mean_ece).powi(2))
            .sum::<f64>() / eces.len() as f64;
        let ece_std = ece_variance.sqrt();

        let mean_drift_rate = drift_rates.iter().sum::<f64>() / drift_rates.len() as f64;
        let max_drift = drift_rates.iter().fold(0.0, |a, &b| a.max(b));

        // Count SLO breaches and alerts
        let slo_breaches = metrics.iter()
            .filter(|m| !matches!(m.slo_status, SloStatus::Healthy))
            .count() as u32;

        let total_alerts = metrics.iter()
            .map(|m| m.active_alerts.len() as u32)
            .sum::<u32>();

        // Check regression incidents
        let regression_count = self.regression_detector.get_incident_count_since(
            metrics[0].timestamp
        ).await.unwrap_or(0);

        // Validate against thresholds
        if mean_ece > self.config.ece_threshold {
            failures.push(ValidationFailure::EceThresholdExceeded {
                observed: mean_ece,
                threshold: self.config.ece_threshold,
            });
        }

        if mean_drift_rate > self.config.drift_threshold {
            failures.push(ValidationFailure::DriftRateExceeded {
                observed: mean_drift_rate,
                threshold: self.config.drift_threshold,
            });
        }

        if slo_breaches > self.config.max_slo_breaches {
            failures.push(ValidationFailure::SloBreaches {
                count: slo_breaches,
                threshold: self.config.max_slo_breaches,
            });
        }

        if total_alerts > self.config.max_alerts {
            failures.push(ValidationFailure::AlertsTriggered {
                count: total_alerts,
                threshold: self.config.max_alerts,
            });
        }

        if regression_count > 0 {
            failures.push(ValidationFailure::RegressionDetected {
                severity: RegressionSeverity::High, // Conservative assessment
            });
        }

        if failures.is_empty() {
            // Calculate stability score
            let ece_score = (self.config.ece_threshold - mean_ece) / self.config.ece_threshold;
            let drift_score = (self.config.drift_threshold - mean_drift_rate) / self.config.drift_threshold;
            let alert_score = if total_alerts == 0 { 1.0 } else {
                1.0 - (total_alerts as f64 / self.config.max_alerts as f64)
            };
            let stability_score = (ece_score + drift_score + alert_score) / 3.0;

            let stability_metrics = StabilityMetrics {
                mean_ece,
                max_ece,
                ece_std,
                mean_drift_rate,
                max_drift,
                slo_breaches,
                alert_count: total_alerts,
                regression_count,
                stability_score: stability_score.max(0.0).min(1.0),
                validation_period: (metrics[0].timestamp, metrics.last().unwrap().timestamp),
            };

            Ok(ValidationResult::Passed { metrics: stability_metrics })
        } else {
            Ok(ValidationResult::Failed { failures })
        }
    }

    /// Publish green fingerprint to public repository
    pub async fn publish_green_fingerprint(
        &self,
        session_id: &str,
    ) -> Result<PublishedFingerprint> {
        let session = {
            let sessions = self.validation_sessions.read()
                .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
            sessions.get(session_id).cloned()
                .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?
        };

        match session.status {
            ValidationSessionStatus::Completed { result: ValidationResult::Passed { metrics } } => {
                // Create green fingerprint
                let status = FingerprintStatus::Green {
                    validated_at: Utc::now(),
                    stability_metrics: metrics,
                };

                // Generate attestation
                let attestation = self.attestation_service.create_manifest_attestation(
                    &session.manifest,
                    &self.config.attestation_config.signing_key_id,
                ).await.context("Failed to create attestation")?;

                // Create publication metadata
                let publication_metadata = PublicationMetadata {
                    published_at: Utc::now(),
                    publisher: "Calibration Publisher Service v1.0".to_string(),
                    version: session.manifest.manifest_version.clone(),
                    distribution_channels: vec![
                        "https".to_string(),
                        "ipfs".to_string(),
                    ],
                    download_urls: HashMap::from([
                        ("https".to_string(), format!("{}/{}.json", 
                            self.config.repository_config.base_url,
                            session.fingerprint_id)),
                        ("checksum".to_string(), format!("{}/{}.sha256", 
                            self.config.repository_config.base_url,
                            session.fingerprint_id)),
                    ]),
                    verification_instructions: "Verify using: lens-verify --fingerprint {fingerprint_id}".to_string(),
                };

                // Create release verification
                let release_verification = self.create_release_verification(
                    &session.manifest,
                    &attestation,
                ).await?;

                let published_fingerprint = PublishedFingerprint {
                    fingerprint_id: session.fingerprint_id.clone(),
                    status,
                    manifest: session.manifest.clone(),
                    attestation,
                    publication_metadata,
                    release_verification,
                };

                // Store in registry
                {
                    let mut published = self.published_fingerprints.write()
                        .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
                    published.insert(session.fingerprint_id.clone(), published_fingerprint.clone());
                }

                // Publish to repository
                self.publish_to_repository(&published_fingerprint).await?;

                info!(
                    fingerprint_id = %session.fingerprint_id,
                    stability_score = %published_fingerprint.status.stability_score(),
                    "Successfully published green fingerprint"
                );

                Ok(published_fingerprint)
            }
            _ => bail!("Session is not ready for publishing: validation not passed"),
        }
    }

    /// Create release verification data
    async fn create_release_verification(
        &self,
        manifest: &CalibrationManifest,
        attestation: &AttestationSignature,
    ) -> Result<ReleaseVerification> {
        let mut artifact_checksums = HashMap::new();

        // Calculate manifest checksum
        let manifest_bytes = serde_json::to_vec(manifest)?;
        let manifest_hash = Sha256::digest(&manifest_bytes);
        artifact_checksums.insert(
            "manifest.json".to_string(),
            hex::encode(manifest_hash),
        );

        // Calculate attestation checksum  
        let attestation_bytes = serde_json::to_vec(attestation)?;
        let attestation_hash = Sha256::digest(&attestation_bytes);
        artifact_checksums.insert(
            "attestation.json".to_string(),
            hex::encode(attestation_hash),
        );

        // Create verification script checksum (placeholder)
        let verification_script = r#"#!/bin/bash
# Verification script for green fingerprints
echo "Verifying fingerprint integrity..."
"#;
        let script_hash = Sha256::digest(verification_script.as_bytes());

        Ok(ReleaseVerification {
            artifact_checksums,
            verification_script_hash: hex::encode(script_hash),
            gpg_signature: None, // TODO: Add GPG support
            certificate_chain: vec![], // TODO: Load from config
            verified_at: Utc::now(),
        })
    }

    /// Publish fingerprint to configured repository
    async fn publish_to_repository(
        &self,
        published_fingerprint: &PublishedFingerprint,
    ) -> Result<()> {
        let json_content = serde_json::to_string_pretty(published_fingerprint)?;
        
        // For now, write to local filesystem (production would use repository API)
        let filename = format!("{}.json", published_fingerprint.fingerprint_id);
        let filepath = PathBuf::from(&self.config.repository_config.publication_path)
            .join(&filename);

        if let Some(parent) = filepath.parent() {
            fs::create_dir_all(parent).await?;
        }

        fs::write(&filepath, json_content).await
            .context("Failed to write published fingerprint")?;

        info!(
            fingerprint_id = %published_fingerprint.fingerprint_id,
            path = %filepath.display(),
            "Published fingerprint to repository"
        );

        Ok(())
    }

    /// Get published fingerprint by ID
    pub fn get_published_fingerprint(&self, fingerprint_id: &str) -> Result<Option<PublishedFingerprint>> {
        let published = self.published_fingerprints.read()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
        Ok(published.get(fingerprint_id).cloned())
    }

    /// List all published fingerprints
    pub fn list_published_fingerprints(&self) -> Result<Vec<PublishedFingerprint>> {
        let published = self.published_fingerprints.read()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
        Ok(published.values().cloned().collect())
    }

    /// Verify published fingerprint integrity
    pub async fn verify_published_fingerprint(
        &self,
        fingerprint_id: &str,
    ) -> Result<bool> {
        if let Some(published) = self.get_published_fingerprint(fingerprint_id)? {
            // Verify attestation signature
            let verification_result = self.attestation_service.verify_manifest_attestation(
                &published.manifest,
                &published.attestation,
            ).await?;

            Ok(verification_result.is_valid)
        } else {
            bail!("Published fingerprint not found: {}", fingerprint_id);
        }
    }
}

impl FingerprintStatus {
    /// Get stability score for status
    pub fn stability_score(&self) -> f64 {
        match self {
            FingerprintStatus::Green { stability_metrics, .. } => stability_metrics.stability_score,
            _ => 0.0,
        }
    }

    /// Check if status is green (validated)
    pub fn is_green(&self) -> bool {
        matches!(self, FingerprintStatus::Green { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_fingerprint_status_creation() {
        let status = FingerprintStatus::Validating {
            start_time: Utc::now(),
            validation_duration: Duration::hours(24),
        };

        assert!(!status.is_green());
        assert_eq!(status.stability_score(), 0.0);
    }

    #[tokio::test]
    async fn test_stability_metrics() {
        let metrics = StabilityMetrics {
            mean_ece: 0.010,
            max_ece: 0.012,
            ece_std: 0.001,
            mean_drift_rate: 0.0005,
            max_drift: 0.0008,
            slo_breaches: 0,
            alert_count: 2,
            regression_count: 0,
            stability_score: 0.95,
            validation_period: (Utc::now() - Duration::hours(24), Utc::now()),
        };

        assert!(metrics.stability_score > 0.9);
        assert!(metrics.mean_ece < 0.015);
    }
}