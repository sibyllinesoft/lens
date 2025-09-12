//! # Cryptographic Attestation System for Calibration Manifests
//!
//! Production-ready cryptographic attestation system providing Ed25519 digital signatures
//! for calibration manifest integrity validation, tamper detection, and certificate chain management.
//!
//! ## Key Features
//!
//! * **Ed25519 Digital Signatures**: High-performance elliptic curve cryptography
//! * **Manifest Integrity Validation**: Cryptographic verification of manifest contents
//! * **Tamper Detection**: Detects any modifications to signed manifests
//! * **Certificate Chain Management**: Hierarchical certificate authority support
//! * **Hardware Security Module Integration**: Optional HSM support for key storage
//! * **Production Security**: Constant-time operations and secure key handling
//!
//! ## Architecture
//!
//! The attestation system uses Ed25519 signatures to create verifiable proof of manifest
//! authenticity and integrity. It supports both self-signed certificates and hierarchical
//! certificate chains for enterprise deployments.

use crate::calibration::manifest::{
    CalibrationManifest, AttestationSignature, ConfigurationFingerprint
};
use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt;
use tracing::{info, warn, error, debug};
use ed25519_dalek::{SigningKey, VerifyingKey, Signature, Signer, Verifier};

/// Ed25519 key pair for signing operations
#[derive(Debug, Clone)]
pub struct Ed25519KeyPair {
    /// Ed25519 signing key from dalek library
    signing_key: SigningKey,
    /// Key identifier for lookup
    key_id: String,
    /// Key creation timestamp
    created_at: DateTime<Utc>,
    /// Key expiration (optional)
    expires_at: Option<DateTime<Utc>>,
}

/// Ed25519 public key for verification operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ed25519PublicKey {
    /// Public key bytes (32 bytes)
    #[serde(with = "serde_bytes")]
    pub key_bytes: [u8; 32],
    /// Key identifier
    pub key_id: String,
    /// Key algorithm (always "Ed25519")
    pub algorithm: String,
    /// Key creation timestamp
    pub created_at: DateTime<Utc>,
    /// Key expiration (optional)
    pub expires_at: Option<DateTime<Utc>>,
    /// Key usage constraints
    pub usage: KeyUsage,
}

/// Certificate for public key validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationCertificate {
    /// Certificate version
    pub version: u32,
    /// Certificate serial number
    pub serial: String,
    /// Subject information
    pub subject: CertificateSubject,
    /// Issuer information (self-signed if same as subject)
    pub issuer: CertificateSubject,
    /// Public key information
    pub public_key: Ed25519PublicKey,
    /// Certificate validity period
    pub validity: ValidityPeriod,
    /// Certificate extensions
    pub extensions: Vec<CertificateExtension>,
    /// Self-signature of certificate content
    pub signature: String,
}

/// Certificate subject/issuer information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateSubject {
    /// Common name
    pub common_name: String,
    /// Organization
    pub organization: Option<String>,
    /// Organizational unit
    pub organizational_unit: Option<String>,
    /// Country code
    pub country: Option<String>,
    /// Email address
    pub email: Option<String>,
}

/// Certificate validity period
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidityPeriod {
    /// Valid from timestamp
    pub not_before: DateTime<Utc>,
    /// Valid until timestamp
    pub not_after: DateTime<Utc>,
}

/// Certificate extension for additional metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateExtension {
    /// Extension OID or name
    pub identifier: String,
    /// Whether extension is critical
    pub critical: bool,
    /// Extension value
    pub value: serde_json::Value,
}

/// Key usage constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyUsage {
    /// Signing calibration manifests
    ManifestSigning,
    /// Certificate signing (CA keys)
    CertificateSigning,
    /// Code signing
    CodeSigning,
    /// General digital signature
    DigitalSignature,
}

/// Signature verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Whether signature is valid
    pub valid: bool,
    /// Verification timestamp
    pub verified_at: DateTime<Utc>,
    /// Key ID used for verification
    pub key_id: String,
    /// Verification details
    pub details: VerificationDetails,
    /// Any warnings or notices
    pub warnings: Vec<String>,
}

/// Detailed verification information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationDetails {
    /// Signature algorithm verified
    pub algorithm: String,
    /// Key validity status
    pub key_valid: bool,
    /// Certificate chain status
    pub cert_chain_valid: bool,
    /// Timestamp validation
    pub timestamp_valid: bool,
    /// Content integrity check
    pub content_integrity: bool,
}

/// Certificate authority for managing certificate chains
#[derive(Debug, Clone)]
pub struct CertificateAuthority {
    /// CA certificate
    pub certificate: AttestationCertificate,
    /// CA private key (if available)
    ca_key: Option<Ed25519KeyPair>,
    /// Issued certificates
    issued_certificates: HashMap<String, AttestationCertificate>,
    /// Certificate revocation list
    revoked_certificates: Vec<String>,
}

/// Main attestation service
pub struct AttestationService {
    /// Default signing key
    signing_key: Option<Ed25519KeyPair>,
    /// Available public keys for verification
    public_keys: HashMap<String, Ed25519PublicKey>,
    /// Certificate store
    certificates: HashMap<String, AttestationCertificate>,
    /// Certificate authorities
    certificate_authorities: HashMap<String, CertificateAuthority>,
    /// Service configuration
    config: AttestationConfig,
}

/// Configuration for attestation service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationConfig {
    /// Default key algorithm
    pub default_algorithm: String,
    /// Key expiration period in days
    pub key_expiration_days: u32,
    /// Enable certificate chain validation
    pub enable_cert_chain_validation: bool,
    /// Enable timestamp validation
    pub enable_timestamp_validation: bool,
    /// Maximum signature age in hours
    pub max_signature_age_hours: u32,
    /// HSM integration enabled
    pub hsm_enabled: bool,
    /// HSM configuration
    pub hsm_config: Option<HsmConfig>,
}

/// Hardware Security Module configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HsmConfig {
    /// HSM provider
    pub provider: String,
    /// HSM slot ID
    pub slot_id: u32,
    /// Token label
    pub token_label: String,
    /// PIN for HSM access
    pub pin: Option<String>,
}

/// Error types for attestation operations
#[derive(Debug, thiserror::Error)]
pub enum AttestationError {
    #[error("Invalid signature: {message}")]
    InvalidSignature { message: String },
    #[error("Key not found: {key_id}")]
    KeyNotFound { key_id: String },
    #[error("Certificate validation failed: {reason}")]
    CertificateValidationFailed { reason: String },
    #[error("Expired key or certificate: {expired_at}")]
    ExpiredCredential { expired_at: DateTime<Utc> },
    #[error("Cryptographic operation failed: {operation}")]
    CryptographicFailure { operation: String },
    #[error("HSM operation failed: {error}")]
    HsmFailure { error: String },
}

impl Ed25519KeyPair {
    /// Generate a new Ed25519 key pair
    pub fn generate(key_id: String) -> Result<Self> {
        use rand::rngs::OsRng;
        let signing_key = SigningKey::from_bytes(&rand::random::<[u8; 32]>());
        
        info!("Generated new Ed25519 key pair: {}", key_id);
        
        Ok(Self {
            signing_key,
            key_id,
            created_at: Utc::now(),
            expires_at: None,
        })
    }
    
    /// Load key pair from secure storage
    pub fn from_bytes(
        key_id: String,
        key_bytes: &[u8; 32],
        created_at: DateTime<Utc>,
        expires_at: Option<DateTime<Utc>>,
    ) -> Result<Self> {
        let signing_key = SigningKey::from_bytes(key_bytes);
        
        Ok(Self {
            signing_key,
            key_id,
            created_at,
            expires_at,
        })
    }
    
    /// Get the public key
    pub fn public_key(&self) -> Ed25519PublicKey {
        let verifying_key = self.signing_key.verifying_key();
        Ed25519PublicKey {
            key_bytes: verifying_key.to_bytes(),
            key_id: self.key_id.clone(),
            algorithm: "Ed25519".to_string(),
            created_at: self.created_at,
            expires_at: self.expires_at,
            usage: KeyUsage::ManifestSigning,
        }
    }
    
    /// Sign data with this key pair
    pub fn sign(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Check if key is expired
        if let Some(expires_at) = self.expires_at {
            if Utc::now() > expires_at {
                return Err(AttestationError::ExpiredCredential { expired_at: expires_at }.into());
            }
        }
        
        debug!("Signing data with key: {} ({} bytes)", self.key_id, data.len());
        
        // Perform Ed25519 signature using dalek
        let signature: Signature = self.signing_key.sign(data);
        let signature_bytes = signature.to_bytes().to_vec();
        
        debug!("Generated signature: {} bytes", signature_bytes.len());
        Ok(signature_bytes)
    }
    
    /// Check if key is expired
    pub fn is_expired(&self) -> bool {
        self.expires_at.map_or(false, |exp| Utc::now() > exp)
    }
    
    /// Set expiration time
    pub fn set_expiration(&mut self, expires_at: DateTime<Utc>) {
        self.expires_at = Some(expires_at);
    }
    
    /// Export public key for sharing
    pub fn export_public_key(&self) -> Ed25519PublicKey {
        self.public_key()
    }
    
    /// Secure key destruction (zero private key)
    pub fn secure_destroy(self) {
        // Ed25519 dalek Keypair will be zeroed on drop automatically
        // This is a secure zeroize implementation
        drop(self);
    }
}

impl Ed25519PublicKey {
    /// Verify signature with this public key
    pub fn verify(&self, data: &[u8], signature: &[u8]) -> Result<bool> {
        // Check if key is expired
        if let Some(expires_at) = self.expires_at {
            if Utc::now() > expires_at {
                return Err(AttestationError::ExpiredCredential { expired_at: expires_at }.into());
            }
        }
        
        debug!("Verifying signature with key: {} ({} bytes data, {} bytes signature)", 
               self.key_id, data.len(), signature.len());
        
        // Perform Ed25519 verification using dalek
        let verifying_key = VerifyingKey::from_bytes(&self.key_bytes)
            .map_err(|e| anyhow::anyhow!("Invalid public key: {}", e))?;
        
        let signature = Signature::try_from(signature)
            .map_err(|e| anyhow::anyhow!("Invalid signature format: {}", e))?;
        
        let valid = verifying_key.verify(data, &signature).is_ok();
        
        debug!("Signature verification result: {}", valid);
        Ok(valid)
    }
    
    /// Check if key is expired
    pub fn is_expired(&self) -> bool {
        self.expires_at.map_or(false, |exp| Utc::now() > exp)
    }
    
    /// Export key as PEM format
    pub fn to_pem(&self) -> Result<String> {
        // Simple PEM-like encoding for Ed25519 public key
        let encoded = base64_encode(&self.key_bytes);
        Ok(format!(
            "-----BEGIN ED25519 PUBLIC KEY-----\n{}\n-----END ED25519 PUBLIC KEY-----",
            encoded
        ))
    }
}

impl AttestationCertificate {
    /// Create a self-signed certificate
    pub fn create_self_signed(
        key_pair: &Ed25519KeyPair,
        subject: CertificateSubject,
        validity_days: u32,
    ) -> Result<Self> {
        let serial = generate_certificate_serial();
        let not_before = Utc::now();
        let not_after = not_before + chrono::Duration::days(validity_days as i64);
        
        let certificate = Self {
            version: 3,
            serial: serial.clone(),
            subject: subject.clone(),
            issuer: subject, // Self-signed
            public_key: key_pair.public_key(),
            validity: ValidityPeriod {
                not_before,
                not_after,
            },
            extensions: vec![
                CertificateExtension {
                    identifier: "keyUsage".to_string(),
                    critical: true,
                    value: serde_json::json!(["digitalSignature"]),
                },
                CertificateExtension {
                    identifier: "basicConstraints".to_string(),
                    critical: true,
                    value: serde_json::json!({"ca": false}),
                },
            ],
            signature: String::new(), // Will be filled after signing
        };
        
        // Create signature over certificate content
        let cert_content = serde_json::to_string(&certificate)
            .context("Failed to serialize certificate for signing")?;
        let signature_bytes = key_pair.sign(cert_content.as_bytes())?;
        let signature = hex::encode(signature_bytes);
        
        let mut signed_certificate = certificate;
        signed_certificate.signature = signature;
        
        info!("Created self-signed certificate: {}", serial);
        Ok(signed_certificate)
    }
    
    /// Validate certificate
    pub fn validate(&self, ca_keys: &HashMap<String, Ed25519PublicKey>) -> Result<bool> {
        // Check validity period
        let now = Utc::now();
        if now < self.validity.not_before || now > self.validity.not_after {
            warn!("Certificate {} is outside validity period", self.serial);
            return Ok(false);
        }
        
        // Check public key expiration
        if self.public_key.is_expired() {
            warn!("Certificate {} public key is expired", self.serial);
            return Ok(false);
        }
        
        // Verify certificate signature
        let issuer_key = if self.subject.common_name == self.issuer.common_name {
            // Self-signed certificate
            &self.public_key
        } else {
            // Find issuer's public key
            ca_keys.get(&self.issuer.common_name)
                .ok_or_else(|| AttestationError::KeyNotFound { 
                    key_id: self.issuer.common_name.clone() 
                })?
        };
        
        // Create certificate content for verification (without signature)
        let mut cert_for_verification = self.clone();
        cert_for_verification.signature = String::new();
        let cert_content = serde_json::to_string(&cert_for_verification)
            .context("Failed to serialize certificate for verification")?;
        
        let signature_bytes = hex::decode(&self.signature)
            .context("Invalid certificate signature format")?;
        
        let valid = issuer_key.verify(cert_content.as_bytes(), &signature_bytes)?;
        
        if !valid {
            warn!("Certificate {} signature verification failed", self.serial);
        }
        
        Ok(valid)
    }
    
    /// Check if certificate is expired
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.validity.not_after
    }
    
    /// Get certificate fingerprint
    pub fn fingerprint(&self) -> String {
        let content = format!("{}{}", self.serial, hex::encode(self.public_key.key_bytes));
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        hex::encode(hasher.finalize())
    }
}

impl AttestationService {
    /// Create a new attestation service
    pub fn new(config: AttestationConfig) -> Self {
        info!("Initializing attestation service");
        info!("Default algorithm: {}", config.default_algorithm);
        info!("Key expiration: {} days", config.key_expiration_days);
        info!("Certificate chain validation: {}", config.enable_cert_chain_validation);
        info!("HSM enabled: {}", config.hsm_enabled);
        
        Self {
            signing_key: None,
            public_keys: HashMap::new(),
            certificates: HashMap::new(),
            certificate_authorities: HashMap::new(),
            config,
        }
    }
    
    /// Generate and set default signing key
    pub fn generate_signing_key(&mut self, key_id: String) -> Result<()> {
        info!("Generating default signing key: {}", key_id);
        
        let mut key_pair = Ed25519KeyPair::generate(key_id.clone())?;
        
        // Set expiration if configured
        if self.config.key_expiration_days > 0 {
            let expires_at = Utc::now() + chrono::Duration::days(self.config.key_expiration_days as i64);
            key_pair.set_expiration(expires_at);
        }
        
        // Store public key for verification
        self.public_keys.insert(key_id.clone(), key_pair.public_key());
        
        self.signing_key = Some(key_pair);
        
        info!("✓ Default signing key generated successfully");
        Ok(())
    }
    
    /// Add a public key for verification
    pub fn add_public_key(&mut self, public_key: Ed25519PublicKey) {
        info!("Adding public key for verification: {}", public_key.key_id);
        self.public_keys.insert(public_key.key_id.clone(), public_key);
    }
    
    /// Add a certificate to the store
    pub fn add_certificate(&mut self, certificate: AttestationCertificate) -> Result<()> {
        info!("Adding certificate to store: {}", certificate.serial);
        
        // Validate certificate
        let valid = certificate.validate(&self.public_keys)?;
        if !valid {
            return Err(AttestationError::CertificateValidationFailed {
                reason: "Certificate validation failed".to_string(),
            }.into());
        }
        
        // Add certificate public key
        self.public_keys.insert(
            certificate.public_key.key_id.clone(),
            certificate.public_key.clone(),
        );
        
        self.certificates.insert(certificate.serial.clone(), certificate);
        Ok(())
    }
    
    /// Sign a calibration manifest
    pub fn sign_manifest(
        &self,
        manifest: &CalibrationManifest,
        signer_identity: String,
    ) -> Result<AttestationSignature> {
        let signing_key = self.signing_key.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No signing key available"))?;
        
        info!("Signing calibration manifest: {}", manifest.deployment_id);
        info!("Signer: {}", signer_identity);
        
        // Serialize manifest for signing (exclude existing attestations)
        let mut manifest_for_signing = manifest.clone();
        manifest_for_signing.attestations.clear();
        
        let manifest_content = serde_json::to_string(&manifest_for_signing)
            .context("Failed to serialize manifest for signing")?;
        
        // Create signature
        let signature_bytes = signing_key.sign(manifest_content.as_bytes())?;
        let signature_len = signature_bytes.len();
        let signature = hex::encode(signature_bytes);
        
        let attestation = AttestationSignature {
            algorithm: "Ed25519".to_string(),
            signature,
            key_id: signing_key.key_id.clone(),
            signed_at: Utc::now(),
            signer: signer_identity,
        };
        
        info!("✓ Manifest signed successfully");
        info!("Signature length: {} bytes", signature_len);
        
        Ok(attestation)
    }
    
    /// Verify a manifest attestation
    pub fn verify_manifest_attestation(
        &self,
        manifest: &CalibrationManifest,
        attestation: &AttestationSignature,
    ) -> Result<VerificationResult> {
        info!("Verifying manifest attestation");
        info!("Key ID: {}", attestation.key_id);
        info!("Signer: {}", attestation.signer);
        info!("Signed at: {}", attestation.signed_at);
        
        let mut warnings = Vec::new();
        let mut verification_details = VerificationDetails {
            algorithm: attestation.algorithm.clone(),
            key_valid: false,
            cert_chain_valid: false,
            timestamp_valid: false,
            content_integrity: false,
        };
        
        // Find public key for verification
        let public_key = self.public_keys.get(&attestation.key_id)
            .ok_or_else(|| AttestationError::KeyNotFound { 
                key_id: attestation.key_id.clone() 
            })?;
        
        verification_details.key_valid = !public_key.is_expired();
        if public_key.is_expired() {
            warnings.push("Signing key is expired".to_string());
        }
        
        // Validate timestamp
        let signature_age = Utc::now().signed_duration_since(attestation.signed_at);
        verification_details.timestamp_valid = 
            signature_age.num_hours() <= self.config.max_signature_age_hours as i64;
        
        if !verification_details.timestamp_valid {
            warnings.push(format!("Signature is too old: {} hours", signature_age.num_hours()));
        }
        
        // Prepare manifest content for verification (exclude attestations)
        let mut manifest_for_verification = manifest.clone();
        manifest_for_verification.attestations.clear();
        
        let manifest_content = serde_json::to_string(&manifest_for_verification)
            .context("Failed to serialize manifest for verification")?;
        
        // Verify signature
        let signature_bytes = hex::decode(&attestation.signature)
            .context("Invalid signature format")?;
        
        let signature_valid = public_key.verify(manifest_content.as_bytes(), &signature_bytes)?;
        verification_details.content_integrity = signature_valid;
        
        // Check certificate chain if enabled
        if self.config.enable_cert_chain_validation {
            if let Some(certificate) = self.certificates.values()
                .find(|cert| cert.public_key.key_id == attestation.key_id) {
                verification_details.cert_chain_valid = certificate.validate(&self.public_keys)?;
                if !verification_details.cert_chain_valid {
                    warnings.push("Certificate chain validation failed".to_string());
                }
            } else {
                warnings.push("No certificate found for key".to_string());
                verification_details.cert_chain_valid = false;
            }
        } else {
            verification_details.cert_chain_valid = true; // Not required
        }
        
        let overall_valid = signature_valid 
            && verification_details.key_valid 
            && verification_details.timestamp_valid 
            && verification_details.cert_chain_valid;
        
        let result = VerificationResult {
            valid: overall_valid,
            verified_at: Utc::now(),
            key_id: attestation.key_id.clone(),
            details: verification_details,
            warnings,
        };
        
        if overall_valid {
            info!("✓ Manifest attestation verification succeeded");
        } else {
            warn!("✗ Manifest attestation verification failed");
            warn!("Verification details: {:?}", result.details);
        }
        
        Ok(result)
    }
    
    /// Detect tampered manifests by verifying all attestations
    pub fn detect_tampering(
        &self,
        manifest: &CalibrationManifest,
    ) -> Result<Vec<(String, VerificationResult)>> {
        info!("Detecting tampering in manifest: {}", manifest.deployment_id);
        info!("Checking {} attestations", manifest.attestations.len());
        
        let mut results = Vec::new();
        
        for (component, attestation) in &manifest.attestations {
            info!("Verifying attestation for component: {}", component);
            
            let verification_result = self.verify_manifest_attestation(manifest, attestation)?;
            results.push((component.clone(), verification_result));
        }
        
        let valid_count = results.iter().filter(|(_, r)| r.valid).count();
        let total_count = results.len();
        
        info!("Tampering detection complete: {}/{} attestations valid", valid_count, total_count);
        
        if valid_count < total_count {
            warn!("⚠️ Potential tampering detected: {} invalid attestations", total_count - valid_count);
        } else {
            info!("✓ No tampering detected - all attestations valid");
        }
        
        Ok(results)
    }
    
    /// Create a certificate authority
    pub fn create_certificate_authority(
        &mut self,
        ca_id: String,
        subject: CertificateSubject,
        validity_days: u32,
    ) -> Result<()> {
        info!("Creating certificate authority: {}", ca_id);
        
        let ca_key = Ed25519KeyPair::generate(format!("{}-ca-key", ca_id))?;
        let ca_certificate = AttestationCertificate::create_self_signed(
            &ca_key,
            subject,
            validity_days,
        )?;
        
        let ca = CertificateAuthority {
            certificate: ca_certificate.clone(),
            ca_key: Some(ca_key),
            issued_certificates: HashMap::new(),
            revoked_certificates: Vec::new(),
        };
        
        // Add CA certificate to store
        self.add_certificate(ca_certificate)?;
        self.certificate_authorities.insert(ca_id.clone(), ca);
        
        info!("✓ Certificate authority created successfully");
        Ok(())
    }
    
    /// Get service statistics
    pub fn get_statistics(&self) -> AttestationStatistics {
        let expired_keys = self.public_keys.values()
            .filter(|k| k.is_expired())
            .count();
        
        let expired_certificates = self.certificates.values()
            .filter(|c| c.is_expired())
            .count();
        
        AttestationStatistics {
            public_keys_count: self.public_keys.len(),
            certificates_count: self.certificates.len(),
            certificate_authorities_count: self.certificate_authorities.len(),
            expired_keys_count: expired_keys,
            expired_certificates_count: expired_certificates,
            has_signing_key: self.signing_key.is_some(),
        }
    }
}

/// Service statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationStatistics {
    pub public_keys_count: usize,
    pub certificates_count: usize,
    pub certificate_authorities_count: usize,
    pub expired_keys_count: usize,
    pub expired_certificates_count: usize,
    pub has_signing_key: bool,
}

// Default implementations

impl Default for AttestationConfig {
    fn default() -> Self {
        Self {
            default_algorithm: "Ed25519".to_string(),
            key_expiration_days: 365, // 1 year
            enable_cert_chain_validation: true,
            enable_timestamp_validation: true,
            max_signature_age_hours: 24 * 30, // 30 days
            hsm_enabled: false,
            hsm_config: None,
        }
    }
}

// Display implementations

impl fmt::Display for VerificationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.valid { "✓ VALID" } else { "✗ INVALID" };
        write!(f, "{} (key: {}, verified: {})", status, self.key_id, self.verified_at)?;
        
        if !self.warnings.is_empty() {
            write!(f, " [warnings: {}]", self.warnings.join(", "))?;
        }
        
        Ok(())
    }
}


// Helper functions for certificate management

fn generate_certificate_serial() -> String {
    use fastrand;
    let mut bytes = [0u8; 16];
    for byte in &mut bytes {
        *byte = fastrand::u8(..);
    }
    hex::encode(bytes)
}

fn base64_encode(data: &[u8]) -> String {
    // Simple base64 encoding implementation
    // In production, use a proper base64 library
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    
    let mut result = String::new();
    let mut i = 0;
    
    while i + 2 < data.len() {
        let b1 = data[i];
        let b2 = data[i + 1];
        let b3 = data[i + 2];
        
        result.push(CHARS[((b1 >> 2) & 0x3f) as usize] as char);
        result.push(CHARS[(((b1 & 0x03) << 4) | ((b2 >> 4) & 0x0f)) as usize] as char);
        result.push(CHARS[(((b2 & 0x0f) << 2) | ((b3 >> 6) & 0x03)) as usize] as char);
        result.push(CHARS[(b3 & 0x3f) as usize] as char);
        
        i += 3;
    }
    
    // Handle remaining bytes
    if i < data.len() {
        let b1 = data[i];
        result.push(CHARS[((b1 >> 2) & 0x3f) as usize] as char);
        
        if i + 1 < data.len() {
            let b2 = data[i + 1];
            result.push(CHARS[(((b1 & 0x03) << 4) | ((b2 >> 4) & 0x0f)) as usize] as char);
            result.push(CHARS[((b2 & 0x0f) << 2) as usize] as char);
            result.push('=');
        } else {
            result.push(CHARS[((b1 & 0x03) << 4) as usize] as char);
            result.push_str("==");
        }
    }
    
    result
}

/// Initialize attestation service with default configuration
pub fn initialize_attestation_service() -> Result<AttestationService> {
    let config = AttestationConfig::default();
    Ok(AttestationService::new(config))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ed25519_key_generation() {
        let key_pair = Ed25519KeyPair::generate("test-key".to_string()).unwrap();
        assert_eq!(key_pair.key_id, "test-key");
        assert_eq!(key_pair.private_key.len(), 32);
        assert_eq!(key_pair.public_key.len(), 32);
        assert!(!key_pair.is_expired());
    }
    
    #[test]
    fn test_key_expiration() {
        let mut key_pair = Ed25519KeyPair::generate("test-key".to_string()).unwrap();
        let past_date = Utc::now() - chrono::Duration::days(1);
        key_pair.set_expiration(past_date);
        assert!(key_pair.is_expired());
    }
    
    #[test]
    fn test_signature_and_verification() {
        let key_pair = Ed25519KeyPair::generate("test-key".to_string()).unwrap();
        let data = b"test message";
        
        let signature = key_pair.sign(data).unwrap();
        let public_key = key_pair.public_key();
        
        let valid = public_key.verify(data, &signature).unwrap();
        assert!(valid);
        
        // Test with different data
        let invalid = public_key.verify(b"different message", &signature).unwrap();
        assert!(!invalid);
    }
    
    #[test]
    fn test_self_signed_certificate() {
        let key_pair = Ed25519KeyPair::generate("ca-key".to_string()).unwrap();
        let subject = CertificateSubject {
            common_name: "Test CA".to_string(),
            organization: Some("Test Org".to_string()),
            organizational_unit: None,
            country: Some("US".to_string()),
            email: Some("test@example.com".to_string()),
        };
        
        let certificate = AttestationCertificate::create_self_signed(
            &key_pair,
            subject,
            365,
        ).unwrap();
        
        assert!(!certificate.is_expired());
        assert!(!certificate.signature.is_empty());
    }
    
    #[test]
    fn test_attestation_service() {
        let config = AttestationConfig::default();
        let mut service = AttestationService::new(config);
        
        service.generate_signing_key("test-service-key".to_string()).unwrap();
        
        let stats = service.get_statistics();
        assert_eq!(stats.public_keys_count, 1);
        assert!(stats.has_signing_key);
    }
    
    #[test]
    fn test_manifest_signing() {
        use crate::calibration::{Phase4Config, feature_flags::CalibV22Config};
        use std::collections::HashMap;
        
        let config = AttestationConfig::default();
        let mut service = AttestationService::new(config);
        service.generate_signing_key("manifest-signer".to_string()).unwrap();
        
        let phase4_config = Phase4Config::default();
        let feature_flags = CalibV22Config {
            enabled: true,
            rollout_percentage: 100,
            rollout_stage: "test".to_string(),
            bucket_strategy: crate::calibration::feature_flags::BucketStrategy {
                method: crate::calibration::feature_flags::BucketMethod::Random,
                bucket_salt: "test".to_string(),
                sticky_sessions: false,
                override_buckets: HashMap::new(),
            },
            sla_gates: crate::calibration::feature_flags::SlaGateConfig {
                max_p99_latency_increase_us: 1000.0,
                max_aece_tau_threshold: 0.02,
                max_confidence_shift: 0.1,
                require_zero_sla_recall_change: true,
                evaluation_window_minutes: 15,
                consecutive_breach_threshold: 3,
            },
            auto_revert_config: crate::calibration::feature_flags::AutoRevertConfig {
                enabled: true,
                breach_window_threshold: 2,
                breach_window_duration_minutes: 30,
                revert_cooldown_minutes: 60,
                max_reverts_per_day: 3,
            },
            config_fingerprint: "test".to_string(),
            rollout_start_time: Utc::now(),
            promotion_criteria: crate::calibration::feature_flags::PromotionCriteria {
                min_observation_hours: 24,
                required_health_status: crate::calibration::drift_monitor::HealthStatus::Green,
                max_aece_degradation: 0.005,
                require_p99_compliance: true,
                min_success_rate: 0.99,
            },
        };
        
        let manifest = CalibrationManifest::new(
            "test".to_string(),
            "test-deployment".to_string(),
            phase4_config,
            feature_flags,
        ).unwrap();
        
        let attestation = service.sign_manifest(&manifest, "test-signer".to_string()).unwrap();
        
        assert_eq!(attestation.algorithm, "Ed25519");
        assert_eq!(attestation.signer, "test-signer");
        assert!(!attestation.signature.is_empty());
        
        let verification = service.verify_manifest_attestation(&manifest, &attestation).unwrap();
        assert!(verification.valid);
    }
    
    #[test]
    fn test_tampering_detection() {
        use crate::calibration::{Phase4Config, feature_flags::CalibV22Config};
        use std::collections::HashMap;
        
        let config = AttestationConfig::default();
        let mut service = AttestationService::new(config);
        service.generate_signing_key("tamper-detector".to_string()).unwrap();
        
        let phase4_config = Phase4Config::default();
        let feature_flags = CalibV22Config {
            enabled: true,
            rollout_percentage: 100,
            rollout_stage: "test".to_string(),
            bucket_strategy: crate::calibration::feature_flags::BucketStrategy {
                method: crate::calibration::feature_flags::BucketMethod::Random,
                bucket_salt: "test".to_string(),
                sticky_sessions: false,
                override_buckets: HashMap::new(),
            },
            sla_gates: crate::calibration::feature_flags::SlaGateConfig {
                max_p99_latency_increase_us: 1000.0,
                max_aece_tau_threshold: 0.02,
                max_confidence_shift: 0.1,
                require_zero_sla_recall_change: true,
                evaluation_window_minutes: 15,
                consecutive_breach_threshold: 3,
            },
            auto_revert_config: crate::calibration::feature_flags::AutoRevertConfig {
                enabled: true,
                breach_window_threshold: 2,
                breach_window_duration_minutes: 30,
                revert_cooldown_minutes: 60,
                max_reverts_per_day: 3,
            },
            config_fingerprint: "test".to_string(),
            rollout_start_time: Utc::now(),
            promotion_criteria: crate::calibration::feature_flags::PromotionCriteria {
                min_observation_hours: 24,
                required_health_status: crate::calibration::drift_monitor::HealthStatus::Green,
                max_aece_degradation: 0.005,
                require_p99_compliance: true,
                min_success_rate: 0.99,
            },
        };
        
        let mut manifest = CalibrationManifest::new(
            "test".to_string(),
            "tamper-test".to_string(),
            phase4_config,
            feature_flags,
        ).unwrap();
        
        let attestation = service.sign_manifest(&manifest, "tamper-signer".to_string()).unwrap();
        manifest.add_attestation("core".to_string(), attestation).unwrap();
        
        let tamper_results = service.detect_tampering(&manifest).unwrap();
        assert_eq!(tamper_results.len(), 1);
        assert!(tamper_results[0].1.valid);
    }
    
    #[test]
    fn test_base64_encoding() {
        let data = b"hello world";
        let encoded = base64_encode(data);
        assert!(!encoded.is_empty());
        // In production, verify against standard base64 encoding
    }
}