//! Anti-fraud attestation and tripwire system

use anyhow::{Result, bail};
use sha2::{Sha256, Digest};
use crate::built_info;

/// Verify we're running in real mode, not mock
pub fn verify_real_mode() -> Result<()> {
    let mode = std::env::var("LENS_MODE").unwrap_or_else(|_| "real".to_string());
    
    if mode != "real" {
        bail!("TRIPWIRE VIOLATION: LENS_MODE must be 'real', got '{}'", mode);
    }
    
    Ok(())
}

/// Generate build information for handshake
pub fn get_build_info() -> crate::proto::BuildInfoResponse {
    crate::proto::BuildInfoResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
        commit: built_info::GIT_VERSION.unwrap_or("unknown").to_string(),
        build_timestamp: env!("BUILD_TIMESTAMP", "unknown").to_string(),
        features: "default,lsp,benchmarks,attestation".to_string(), // Static for now
    }
}

/// Perform anti-fraud handshake with nonce/response
pub fn perform_handshake(nonce: &str) -> Result<String> {
    let build_sha = built_info::GIT_VERSION.unwrap_or("unknown");
    let challenge_input = format!("{}{}", nonce, build_sha);
    
    let mut hasher = Sha256::new();
    hasher.update(challenge_input.as_bytes());
    let response = format!("{:x}", hasher.finalize());
    
    Ok(response)
}

/// Check for banned patterns in input
pub fn check_banned_patterns(text: &str) -> Vec<String> {
    let banned = [
        "generateMock", "simulate", "MOCK_RESULT", "mock_file_", "fake"
    ];
    
    let mut violations = Vec::new();
    for pattern in &banned {
        if text.to_lowercase().contains(&pattern.to_lowercase()) {
            violations.push(format!("Banned pattern detected: {}", pattern));
        }
    }
    
    violations
}

/// Attestation manager for system integrity verification
pub struct AttestationManager {
    enabled: bool,
    build_hash: String,
    runtime_checks: Vec<String>,
}

impl AttestationManager {
    /// Create new attestation manager
    pub fn new(enabled: bool) -> Result<Self> {
        let build_hash = Self::compute_build_hash()?;
        
        Ok(Self {
            enabled,
            build_hash,
            runtime_checks: Vec::new(),
        })
    }

    /// Verify system integrity
    pub fn verify_integrity(&self) -> Result<AttestationReport> {
        let mut violations = Vec::new();
        let mut checks_passed = 0;
        let mut total_checks = 0;

        // Verify real mode
        total_checks += 1;
        if let Err(e) = verify_real_mode() {
            violations.push(e.to_string());
        } else {
            checks_passed += 1;
        }

        // Verify build integrity
        total_checks += 1;
        if let Err(e) = self.verify_build_integrity() {
            violations.push(e.to_string());
        } else {
            checks_passed += 1;
        }

        // Verify environment
        total_checks += 1;
        if let Err(e) = self.verify_environment() {
            violations.push(e.to_string());
        } else {
            checks_passed += 1;
        }

        Ok(AttestationReport {
            timestamp: chrono::Utc::now(),
            total_checks,
            checks_passed,
            violations,
            build_hash: self.build_hash.clone(),
            integrity_score: (checks_passed as f64 / total_checks as f64) * 100.0,
        })
    }

    /// Perform handshake with client
    pub fn handshake(&self, nonce: &str) -> Result<HandshakeResponse> {
        if !self.enabled {
            bail!("Attestation is disabled");
        }

        let response_hash = perform_handshake(nonce)?;
        let build_info = get_build_info();

        Ok(HandshakeResponse {
            response_hash,
            build_info,
            attestation_enabled: self.enabled,
        })
    }

    fn compute_build_hash() -> Result<String> {
        let build_data = format!(
            "{}{}{}{}",
            built_info::GIT_VERSION.unwrap_or("unknown"),
            env!("BUILD_TIMESTAMP", "unknown"),
            built_info::RUSTC_VERSION,
            built_info::CFG_TARGET_ARCH
        );

        let mut hasher = Sha256::new();
        hasher.update(build_data.as_bytes());
        Ok(format!("{:x}", hasher.finalize()))
    }

    fn verify_build_integrity(&self) -> Result<()> {
        let current_hash = Self::compute_build_hash()?;
        if current_hash != self.build_hash {
            bail!("Build integrity check failed: hash mismatch");
        }
        Ok(())
    }

    fn verify_environment(&self) -> Result<()> {
        // Check for suspicious environment variables
        let suspicious_vars = ["LENS_MOCK", "LENS_SIMULATE", "LENS_FAKE"];
        for var in suspicious_vars {
            if std::env::var(var).is_ok() {
                bail!("Suspicious environment variable detected: {}", var);
            }
        }
        Ok(())
    }
}

/// Attestation report
#[derive(Debug, Clone)]
pub struct AttestationReport {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub total_checks: usize,
    pub checks_passed: usize,
    pub violations: Vec<String>,
    pub build_hash: String,
    pub integrity_score: f64,
}

impl AttestationReport {
    pub fn is_valid(&self) -> bool {
        self.violations.is_empty() && self.integrity_score >= 100.0
    }
}

/// Handshake response
#[derive(Debug, Clone)]
pub struct HandshakeResponse {
    pub response_hash: String,
    pub build_info: crate::proto::BuildInfoResponse,
    pub attestation_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::sync::Mutex;
    
    // Shared mutex to prevent race conditions with environment variables across all attestation tests
    static ATTESTATION_ENV_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn test_verify_real_mode_default() {
        let _lock = ATTESTATION_ENV_MUTEX.lock().unwrap();
        
        // Default mode should be "real"
        env::remove_var("LENS_MODE");
        assert!(verify_real_mode().is_ok());
    }

    #[test]
    fn test_verify_real_mode_explicit() {
        let _lock = ATTESTATION_ENV_MUTEX.lock().unwrap();
        
        env::set_var("LENS_MODE", "real");
        assert!(verify_real_mode().is_ok());
        // Clean up
        env::remove_var("LENS_MODE");
    }

    #[test]
    fn test_verify_real_mode_rejects_mock() {
        let _lock = ATTESTATION_ENV_MUTEX.lock().unwrap();
        
        env::set_var("LENS_MODE", "mock");
        let result = verify_real_mode();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("TRIPWIRE VIOLATION"));
        
        // Clean up
        env::remove_var("LENS_MODE");
    }

    #[test]
    fn test_verify_real_mode_rejects_invalid() {
        let _lock = ATTESTATION_ENV_MUTEX.lock().unwrap();
        
        env::set_var("LENS_MODE", "test");
        let result = verify_real_mode();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("TRIPWIRE VIOLATION"));
        
        // Clean up
        env::remove_var("LENS_MODE");
    }

    #[test]
    fn test_get_build_info() {
        let info = get_build_info();
        assert!(!info.version.is_empty());
        assert!(!info.commit.is_empty());
        assert!(!info.build_timestamp.is_empty());
        assert!(!info.features.is_empty());
    }

    #[test]
    fn test_perform_handshake() {
        let nonce = "test_nonce_123";
        let response = perform_handshake(nonce);
        assert!(response.is_ok());
        
        let response_hash = response.unwrap();
        assert_eq!(response_hash.len(), 64); // SHA256 hex string length
        
        // Same nonce should produce same response
        let response2 = perform_handshake(nonce).unwrap();
        assert_eq!(response_hash, response2);
    }

    #[test]
    fn test_perform_handshake_different_nonces() {
        let response1 = perform_handshake("nonce1").unwrap();
        let response2 = perform_handshake("nonce2").unwrap();
        assert_ne!(response1, response2);
    }

    #[test]
    fn test_check_banned_patterns_clean() {
        let clean_text = "This is a normal search query";
        let violations = check_banned_patterns(clean_text);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_check_banned_patterns_violations() {
        let violations = check_banned_patterns("generateMock data");
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("generateMock"));
        
        let violations = check_banned_patterns("MOCK_RESULT and fake data");
        assert_eq!(violations.len(), 2);
    }

    #[test]
    fn test_check_banned_patterns_case_insensitive() {
        let violations = check_banned_patterns("GenerateMOCK and FAKE");
        assert_eq!(violations.len(), 2);
    }

    #[test]
    fn test_attestation_manager_creation() {
        let manager = AttestationManager::new(true);
        assert!(manager.is_ok());
        
        let manager = manager.unwrap();
        assert!(manager.enabled);
        assert!(!manager.build_hash.is_empty());
        assert_eq!(manager.build_hash.len(), 64); // SHA256 hex
    }

    #[test]
    fn test_attestation_manager_disabled() {
        let manager = AttestationManager::new(false).unwrap();
        assert!(!manager.enabled);
    }

    #[test]
    fn test_verify_integrity_clean_environment() {
        let _lock = ATTESTATION_ENV_MUTEX.lock().unwrap();
        
        // Ensure clean environment
        let suspicious_vars = ["LENS_MOCK", "LENS_SIMULATE", "LENS_FAKE", "LENS_MODE"];
        for var in suspicious_vars {
            env::remove_var(var);
        }
        
        let manager = AttestationManager::new(true).unwrap();
        let report = manager.verify_integrity();
        assert!(report.is_ok());
        
        let report = report.unwrap();
        assert_eq!(report.total_checks, 3);
        assert_eq!(report.checks_passed, 3);
        assert!(report.violations.is_empty());
        assert_eq!(report.integrity_score, 100.0);
        assert!(report.is_valid());
    }

    #[test]
    fn test_verify_integrity_with_violations() {
        let _lock = ATTESTATION_ENV_MUTEX.lock().unwrap();
        
        // Set up violation condition
        env::set_var("LENS_MOCK", "true");
        
        let manager = AttestationManager::new(true).unwrap();
        let report = manager.verify_integrity().unwrap();
        
        assert!(report.checks_passed < report.total_checks);
        assert!(!report.violations.is_empty());
        assert!(report.integrity_score < 100.0);
        assert!(!report.is_valid());
        
        // Clean up
        env::remove_var("LENS_MOCK");
    }

    #[test]
    fn test_handshake_enabled() {
        let manager = AttestationManager::new(true).unwrap();
        let response = manager.handshake("test_nonce");
        assert!(response.is_ok());
        
        let response = response.unwrap();
        assert!(!response.response_hash.is_empty());
        assert_eq!(response.response_hash.len(), 64);
        assert!(response.attestation_enabled);
        assert!(!response.build_info.version.is_empty());
    }

    #[test]
    fn test_handshake_disabled() {
        let manager = AttestationManager::new(false).unwrap();
        let response = manager.handshake("test_nonce");
        assert!(response.is_err());
        assert!(response.unwrap_err().to_string().contains("Attestation is disabled"));
    }

    #[test]
    fn test_compute_build_hash_consistency() {
        let hash1 = AttestationManager::compute_build_hash().unwrap();
        let hash2 = AttestationManager::compute_build_hash().unwrap();
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 64); // SHA256 hex string
    }

    #[test]
    fn test_verify_build_integrity() {
        let manager = AttestationManager::new(true).unwrap();
        assert!(manager.verify_build_integrity().is_ok());
    }

    #[test]
    fn test_verify_environment_clean() {
        let _lock = ATTESTATION_ENV_MUTEX.lock().unwrap();
        
        // Clean environment
        let suspicious_vars = ["LENS_MOCK", "LENS_SIMULATE", "LENS_FAKE"];
        for var in suspicious_vars {
            env::remove_var(var);
        }
        
        let manager = AttestationManager::new(true).unwrap();
        assert!(manager.verify_environment().is_ok());
    }

    #[test]
    fn test_verify_environment_suspicious() {
        let _lock = ATTESTATION_ENV_MUTEX.lock().unwrap();
        
        env::set_var("LENS_MOCK", "true");
        
        let manager = AttestationManager::new(true).unwrap();
        let result = manager.verify_environment();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("LENS_MOCK"));
        
        // Clean up
        env::remove_var("LENS_MOCK");
    }

    #[test]
    fn test_attestation_report_validity() {
        use chrono::Utc;
        
        let valid_report = AttestationReport {
            timestamp: Utc::now(),
            total_checks: 3,
            checks_passed: 3,
            violations: Vec::new(),
            build_hash: "test_hash".to_string(),
            integrity_score: 100.0,
        };
        assert!(valid_report.is_valid());
        
        let invalid_report = AttestationReport {
            timestamp: Utc::now(),
            total_checks: 3,
            checks_passed: 2,
            violations: vec!["test violation".to_string()],
            build_hash: "test_hash".to_string(),
            integrity_score: 66.67,
        };
        assert!(!invalid_report.is_valid());
    }

    #[test] 
    fn test_handshake_response_structure() {
        let manager = AttestationManager::new(true).unwrap();
        let response = manager.handshake("test").unwrap();
        
        // Verify all fields are properly set
        assert!(!response.response_hash.is_empty());
        assert!(response.attestation_enabled);
        assert!(!response.build_info.version.is_empty());
        assert!(!response.build_info.commit.is_empty());
        assert!(!response.build_info.build_timestamp.is_empty());
    }
}