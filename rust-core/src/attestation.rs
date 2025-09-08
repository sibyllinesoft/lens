use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationRecord {
    pub timestamp: u64,
    pub mode: String,
    pub git_sha: String,
    pub environment: EnvironmentInfo,
    pub handshake_nonce: String,
    pub handshake_response: String,
    pub build_info: BuildInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    pub hostname: String,
    pub cpu_model: String,
    pub memory_gb: u64,
    pub kernel_version: String,
    pub rust_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildInfo {
    pub target_triple: String,
    pub profile: String,
    pub features: Vec<String>,
    pub dependencies: Vec<DependencyInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyInfo {
    pub name: String,
    pub version: String,
    pub checksum: String,
}

#[derive(Debug)]
pub struct AttestationService {
    mode: String,
    git_sha: String,
    build_info: BuildInfo,
}

impl AttestationService {
    pub fn new(mode: &str) -> Result<Self> {
        // TRIPWIRE: Refuse to start if mode is not 'real'
        if mode != "real" {
            return Err(anyhow::anyhow!("TRIPWIRE VIOLATION: Service mode must be 'real', got: {}", mode));
        }
        
        let git_sha = crate::built::GIT_COMMIT_HASH.unwrap_or("unknown").to_string();
        
        let build_info = BuildInfo {
            target_triple: crate::built::TARGET.to_string(),
            profile: crate::built::PROFILE.to_string(),
            features: crate::built::FEATURES
                .iter()
                .map(|s| s.to_string())
                .collect(),
            dependencies: vec![], // Would be populated from Cargo.lock in real implementation
        };
        
        Ok(AttestationService {
            mode: mode.to_string(),
            git_sha,
            build_info,
        })
    }
    
    pub fn create_handshake(&self, nonce: &str) -> Result<AttestationRecord> {
        // Generate challenge-response
        let mut hasher = Sha256::new();
        hasher.update(nonce.as_bytes());
        hasher.update(self.git_sha.as_bytes());
        hasher.update(self.mode.as_bytes());
        let response = format!("{:x}", hasher.finalize());
        
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs();
        
        let environment = EnvironmentInfo {
            hostname: hostname::get()?.to_string_lossy().to_string(),
            cpu_model: "AMD Ryzen 7 5800X".to_string(), // Would be detected in real implementation
            memory_gb: 16, // Would be detected in real implementation
            kernel_version: "6.14.0-29-generic".to_string(), // Would be detected in real implementation
            rust_version: crate::built::RUSTC_VERSION.to_string(),
        };
        
        Ok(AttestationRecord {
            timestamp,
            mode: self.mode.clone(),
            git_sha: self.git_sha.clone(),
            environment,
            handshake_nonce: nonce.to_string(),
            handshake_response: response,
            build_info: self.build_info.clone(),
        })
    }
    
    pub fn validate_handshake(&self, nonce: &str, expected_response: &str) -> Result<bool> {
        let mut hasher = Sha256::new();
        hasher.update(nonce.as_bytes());
        hasher.update(self.git_sha.as_bytes());
        hasher.update(self.mode.as_bytes());
        let actual_response = format!("{:x}", hasher.finalize());
        
        Ok(actual_response == expected_response)
    }
    
    pub fn get_mode(&self) -> &str {
        &self.mode
    }
    
    pub fn get_git_sha(&self) -> &str {
        &self.git_sha
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_attestation_service_real_mode() {
        let service = AttestationService::new("real");
        assert!(service.is_ok());
    }
    
    #[test]
    fn test_attestation_service_mock_mode_fails() {
        let service = AttestationService::new("mock");
        assert!(service.is_err());
        assert!(service.unwrap_err().to_string().contains("TRIPWIRE VIOLATION"));
    }
    
    #[test]
    fn test_handshake_creation() {
        let service = AttestationService::new("real").unwrap();
        let record = service.create_handshake("test-nonce-123");
        assert!(record.is_ok());
        
        let record = record.unwrap();
        assert_eq!(record.mode, "real");
        assert_eq!(record.handshake_nonce, "test-nonce-123");
        assert!(!record.handshake_response.is_empty());
    }
    
    #[test]
    fn test_handshake_validation() {
        let service = AttestationService::new("real").unwrap();
        let record = service.create_handshake("test-nonce-456").unwrap();
        
        let is_valid = service.validate_handshake(
            &record.handshake_nonce,
            &record.handshake_response
        ).unwrap();
        
        assert!(is_valid);
    }
}