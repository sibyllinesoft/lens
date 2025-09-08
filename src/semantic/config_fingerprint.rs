//! Configuration Fingerprint and Attestation System  
//!
//! Implements cryptographic fingerprinting of training configurations to ensure
//! reproducible training runs and enable policy attestation.
//!
//! **TODO.md Requirements:**
//! - SHA256 fingerprints of all configuration parameters
//! - Attestation binding between configs, models, and artifacts
//! - Baseline policy fingerprints: `policy://lexical_struct_only@<fingerprint>`
//! - Reproducible training with deterministic configuration hashing
//! - Git commit SHA binding for source code versioning

use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use std::collections::{BTreeMap, HashMap};
use anyhow::{Result, Context};
use chrono::{DateTime, Utc};
use tracing::{info, warn};

/// Configuration fingerprint with full attestation data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigFingerprint {
    pub fingerprint_id: String, // SHA256 hex
    pub config_type: ConfigType,
    pub parameters: BTreeMap<String, serde_json::Value>, // Sorted for deterministic hashing
    pub git_commit_sha: String,
    pub creation_timestamp: DateTime<Utc>,
    pub dependencies: Vec<DependencyInfo>,
    pub attestation: AttestationInfo,
}

/// Type of configuration being fingerprinted
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConfigType {
    LTRTraining,
    FeatureExtraction,
    HardNegatives,
    IsotonicCalibration,
    SLAEvaluation,
    BaselinePolicy,
    SearchConfiguration,
}

/// Dependency information for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyInfo {
    pub name: String,
    pub version: String,
    pub source: String, // e.g., "crates.io", "git", "path"
    pub checksum: Option<String>,
}

/// Attestation information for verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationInfo {
    pub attester: String, // System or user that created the config
    pub environment: EnvironmentInfo,
    pub validation_status: ValidationStatus,
    pub signature: Option<String>, // Optional cryptographic signature
}

/// Environment information for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    pub hostname: String,
    pub rust_version: String,
    pub platform: String,
    pub architecture: String,
    pub environment_variables: HashMap<String, String>, // Relevant env vars only
}

/// Validation status of the configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStatus {
    Valid,
    Warning(Vec<String>),
    Invalid(Vec<String>),
}

/// Policy reference with fingerprint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyReference {
    pub policy_name: String,
    pub policy_type: PolicyType,
    pub config_fingerprint: String,
    pub model_fingerprint: Option<String>, // For trained models
    pub uri: String, // e.g., "policy://lexical_struct_only@abc123"
}

/// Type of policy being referenced
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyType {
    Baseline,
    Trained,
    Experimental,
}

/// Main configuration fingerprint system
pub struct ConfigFingerprintSystem {
    git_commit_sha: String,
    environment_info: EnvironmentInfo,
    attester: String,
}

impl ConfigFingerprintSystem {
    /// Create new fingerprint system
    pub fn new() -> Result<Self> {
        let git_commit_sha = Self::get_git_commit_sha()
            .unwrap_or_else(|_| "unknown".to_string());
        
        let environment_info = Self::collect_environment_info()?;
        let attester = format!("lens-training-system@{}", environment_info.hostname);

        Ok(Self {
            git_commit_sha,
            environment_info,
            attester,
        })
    }

    /// Create fingerprint for LTR training configuration
    pub fn fingerprint_ltr_config(
        &self,
        config: &crate::semantic::ltr_trainer::LTRConfig,
    ) -> Result<ConfigFingerprint> {
        let mut parameters = BTreeMap::new();
        
        // Serialize all configuration parameters in deterministic order
        parameters.insert("objective".to_string(), serde_json::to_value(&config.objective)?);
        parameters.insert("max_log_odds_delta".to_string(), serde_json::to_value(&config.max_log_odds_delta)?);
        parameters.insert("monotonic_increasing".to_string(), serde_json::to_value(&config.monotonic_increasing)?);
        parameters.insert("hard_negative_ratio".to_string(), serde_json::to_value(&config.hard_negative_ratio)?);
        parameters.insert("cross_validation_folds".to_string(), serde_json::to_value(&config.cross_validation_folds)?);
        parameters.insert("max_iterations".to_string(), serde_json::to_value(&config.max_iterations)?);
        parameters.insert("learning_rate".to_string(), serde_json::to_value(&config.learning_rate)?);
        parameters.insert("regularization".to_string(), serde_json::to_value(&config.regularization)?);

        self.create_fingerprint(ConfigType::LTRTraining, parameters)
    }

    /// Create fingerprint for feature extraction configuration
    pub fn fingerprint_feature_config(
        &self,
        config: &crate::semantic::feature_extractor::FeatureExtractionConfig,
    ) -> Result<ConfigFingerprint> {
        let mut parameters = BTreeMap::new();
        
        parameters.insert("lexical_features".to_string(), serde_json::to_value(&config.lexical_features)?);
        parameters.insert("structural_features".to_string(), serde_json::to_value(&config.structural_features)?);
        parameters.insert("raptor_features".to_string(), serde_json::to_value(&config.raptor_features)?);
        parameters.insert("centrality_features".to_string(), serde_json::to_value(&config.centrality_features)?);
        parameters.insert("ann_features".to_string(), serde_json::to_value(&config.ann_features)?);
        parameters.insert("path_prior_features".to_string(), serde_json::to_value(&config.path_prior_features)?);

        self.create_fingerprint(ConfigType::FeatureExtraction, parameters)
    }

    /// Create fingerprint for hard negatives configuration  
    pub fn fingerprint_hard_negatives_config(
        &self,
        config: &crate::semantic::hard_negatives::HardNegativesConfig,
    ) -> Result<ConfigFingerprint> {
        let mut parameters = BTreeMap::new();
        
        parameters.insert("negatives_per_positive".to_string(), serde_json::to_value(&config.negatives_per_positive)?);
        parameters.insert("min_distance".to_string(), serde_json::to_value(&config.min_distance)?);
        parameters.insert("max_attempts".to_string(), serde_json::to_value(&config.max_attempts)?);
        parameters.insert("use_symbol_graph".to_string(), serde_json::to_value(&config.use_symbol_graph)?);

        self.create_fingerprint(ConfigType::HardNegatives, parameters)
    }

    /// Create fingerprint for isotonic calibration configuration
    pub fn fingerprint_calibration_config(
        &self,
        config: &crate::semantic::isotonic_calibration::IsotonicCalibrationConfig,
    ) -> Result<ConfigFingerprint> {
        let mut parameters = BTreeMap::new();
        
        parameters.insert("min_slope".to_string(), serde_json::to_value(&config.min_slope)?);
        parameters.insert("max_slope".to_string(), serde_json::to_value(&config.max_slope)?);
        parameters.insert("max_ece".to_string(), serde_json::to_value(&config.max_ece)?);
        parameters.insert("min_bin_size".to_string(), serde_json::to_value(&config.min_bin_size)?);

        self.create_fingerprint(ConfigType::IsotonicCalibration, parameters)
    }

    /// Create fingerprint for SLA evaluation configuration
    pub fn fingerprint_evaluation_config(
        &self,
        config: &crate::semantic::sla_bounded_evaluation::SLAEvaluationConfig,
    ) -> Result<ConfigFingerprint> {
        let mut parameters = BTreeMap::new();
        
        parameters.insert("sla_timeout_ms".to_string(), serde_json::to_value(&config.sla_timeout_ms)?);
        parameters.insert("max_ece_threshold".to_string(), serde_json::to_value(&config.max_ece_threshold)?);
        parameters.insert("bootstrap_samples".to_string(), serde_json::to_value(&config.bootstrap_samples)?);
        parameters.insert("significance_alpha".to_string(), serde_json::to_value(&config.significance_alpha)?);
        parameters.insert("ndcg_cutoff".to_string(), serde_json::to_value(&config.ndcg_cutoff)?);
        parameters.insert("calibration_bins".to_string(), serde_json::to_value(&config.calibration_bins)?);
        parameters.insert("baseline_policy_fingerprint".to_string(), serde_json::to_value(&config.baseline_policy_fingerprint)?);

        self.create_fingerprint(ConfigType::SLAEvaluation, parameters)
    }

    /// Create baseline policy fingerprint
    pub fn create_baseline_policy_fingerprint(&self) -> Result<ConfigFingerprint> {
        let mut parameters = BTreeMap::new();
        
        // Baseline lexical+structural policy parameters
        parameters.insert("policy_name".to_string(), serde_json::Value::String("lexical_struct_only".to_string()));
        parameters.insert("lexical_weight".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(0.7).unwrap()));
        parameters.insert("structural_weight".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(0.3).unwrap()));
        parameters.insert("semantic_weight".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(0.0).unwrap()));
        parameters.insert("use_reranking".to_string(), serde_json::Value::Bool(false));
        parameters.insert("max_results".to_string(), serde_json::Value::Number(serde_json::Number::from(50)));

        self.create_fingerprint(ConfigType::BaselinePolicy, parameters)
    }

    /// Create combined configuration fingerprint
    pub fn create_training_pipeline_fingerprint(
        &self,
        ltr_config: &crate::semantic::ltr_trainer::LTRConfig,
        feature_config: &crate::semantic::feature_extractor::FeatureExtractionConfig,
        negatives_config: &crate::semantic::hard_negatives::HardNegativesConfig,
        calibration_config: &crate::semantic::isotonic_calibration::IsotonicCalibrationConfig,
    ) -> Result<ConfigFingerprint> {
        let mut parameters = BTreeMap::new();
        
        // Create sub-fingerprints
        let ltr_fp = self.fingerprint_ltr_config(ltr_config)?;
        let feature_fp = self.fingerprint_feature_config(feature_config)?;
        let negatives_fp = self.fingerprint_hard_negatives_config(negatives_config)?;
        let calibration_fp = self.fingerprint_calibration_config(calibration_config)?;

        // Combine fingerprints
        parameters.insert("ltr_config_fingerprint".to_string(), serde_json::Value::String(ltr_fp.fingerprint_id));
        parameters.insert("feature_config_fingerprint".to_string(), serde_json::Value::String(feature_fp.fingerprint_id));
        parameters.insert("negatives_config_fingerprint".to_string(), serde_json::Value::String(negatives_fp.fingerprint_id));
        parameters.insert("calibration_config_fingerprint".to_string(), serde_json::Value::String(calibration_fp.fingerprint_id));

        // Add pipeline-level parameters
        parameters.insert("pipeline_version".to_string(), serde_json::Value::String("1.0".to_string()));
        parameters.insert("training_mode".to_string(), serde_json::Value::String("production".to_string()));

        self.create_fingerprint(ConfigType::LTRTraining, parameters)
    }

    /// Create policy reference from fingerprint
    pub fn create_policy_reference(
        &self,
        policy_name: &str,
        policy_type: PolicyType,
        config_fingerprint: &ConfigFingerprint,
        model_fingerprint: Option<String>,
    ) -> PolicyReference {
        let uri = match policy_type {
            PolicyType::Baseline => format!("policy://{}@{}", policy_name, &config_fingerprint.fingerprint_id[..8]),
            _ => format!("policy://{}@{}", policy_name, &config_fingerprint.fingerprint_id[..8]),
        };

        PolicyReference {
            policy_name: policy_name.to_string(),
            policy_type,
            config_fingerprint: config_fingerprint.fingerprint_id.clone(),
            model_fingerprint,
            uri,
        }
    }

    /// Create fingerprint from parameters
    fn create_fingerprint(
        &self,
        config_type: ConfigType,
        parameters: BTreeMap<String, serde_json::Value>,
    ) -> Result<ConfigFingerprint> {
        // Create deterministic hash input
        let mut hash_input = Vec::new();
        hash_input.extend_from_slice(format!("{:?}", config_type).as_bytes());
        hash_input.extend_from_slice(self.git_commit_sha.as_bytes());
        
        // Add sorted parameters
        let parameters_json = serde_json::to_string(&parameters)
            .context("Failed to serialize parameters")?;
        hash_input.extend_from_slice(parameters_json.as_bytes());

        // Add environment fingerprint
        let env_json = serde_json::to_string(&self.environment_info)
            .context("Failed to serialize environment info")?;
        hash_input.extend_from_slice(env_json.as_bytes());

        // Compute SHA256
        let mut hasher = Sha256::new();
        hasher.update(&hash_input);
        let fingerprint_id = format!("{:x}", hasher.finalize());

        // Collect dependencies
        let dependencies = Self::collect_dependencies()?;

        // Validate configuration
        let validation_status = self.validate_config(&config_type, &parameters)?;

        let attestation = AttestationInfo {
            attester: self.attester.clone(),
            environment: self.environment_info.clone(),
            validation_status,
            signature: None, // Could add cryptographic signatures later
        };

        Ok(ConfigFingerprint {
            fingerprint_id,
            config_type,
            parameters,
            git_commit_sha: self.git_commit_sha.clone(),
            creation_timestamp: Utc::now(),
            dependencies,
            attestation,
        })
    }

    /// Collect environment information
    fn collect_environment_info() -> Result<EnvironmentInfo> {
        let hostname = std::env::var("HOSTNAME")
            .or_else(|_| std::env::var("COMPUTERNAME"))
            .unwrap_or_else(|_| "unknown".to_string());

        let rust_version = std::env::var("RUSTC_VERSION")
            .unwrap_or_else(|_| "unknown".to_string());

        let platform = std::env::consts::OS.to_string();
        let architecture = std::env::consts::ARCH.to_string();

        // Collect relevant environment variables
        let mut environment_variables = HashMap::new();
        for (key, value) in std::env::vars() {
            if key.starts_with("LENS_") || key.starts_with("RUST_") || key == "PATH" {
                environment_variables.insert(key, value);
            }
        }

        Ok(EnvironmentInfo {
            hostname,
            rust_version,
            platform,
            architecture,
            environment_variables,
        })
    }

    /// Get Git commit SHA
    fn get_git_commit_sha() -> Result<String> {
        use std::process::Command;
        
        let output = Command::new("git")
            .args(&["rev-parse", "HEAD"])
            .output()
            .context("Failed to execute git command")?;

        if output.status.success() {
            let sha = String::from_utf8(output.stdout)?
                .trim()
                .to_string();
            Ok(sha)
        } else {
            anyhow::bail!("Git command failed: {}", String::from_utf8_lossy(&output.stderr));
        }
    }

    /// Collect dependency information
    fn collect_dependencies() -> Result<Vec<DependencyInfo>> {
        // For now, return basic Rust dependencies
        // In a real implementation, this would parse Cargo.toml and Cargo.lock
        let dependencies = vec![
            DependencyInfo {
                name: "serde".to_string(),
                version: "1.0".to_string(),
                source: "crates.io".to_string(),
                checksum: None,
            },
            DependencyInfo {
                name: "tokio".to_string(),
                version: "1.0".to_string(),
                source: "crates.io".to_string(),
                checksum: None,
            },
            DependencyInfo {
                name: "anyhow".to_string(),
                version: "1.0".to_string(),
                source: "crates.io".to_string(),
                checksum: None,
            },
        ];

        Ok(dependencies)
    }

    /// Validate configuration parameters
    fn validate_config(
        &self,
        config_type: &ConfigType,
        parameters: &BTreeMap<String, serde_json::Value>,
    ) -> Result<ValidationStatus> {
        let mut warnings = Vec::new();
        let mut errors = Vec::new();

        match config_type {
            ConfigType::LTRTraining => {
                // Validate LTR training parameters
                if let Some(max_log_odds_delta) = parameters.get("max_log_odds_delta") {
                    if let Some(delta) = max_log_odds_delta.as_f64() {
                        if delta != 0.4 {
                            warnings.push(format!("max_log_odds_delta is {}, TODO.md specifies 0.4", delta));
                        }
                    }
                }

                if let Some(hard_negative_ratio) = parameters.get("hard_negative_ratio") {
                    if let Some(ratio) = hard_negative_ratio.as_f64() {
                        if ratio != 4.0 {
                            warnings.push(format!("hard_negative_ratio is {}, TODO.md specifies 4.0", ratio));
                        }
                    }
                }
            }
            ConfigType::IsotonicCalibration => {
                // Validate isotonic calibration parameters
                if let Some(min_slope) = parameters.get("min_slope") {
                    if let Some(slope) = min_slope.as_f64() {
                        if slope != 0.9 {
                            errors.push(format!("min_slope is {}, TODO.md requires 0.9", slope));
                        }
                    }
                }

                if let Some(max_slope) = parameters.get("max_slope") {
                    if let Some(slope) = max_slope.as_f64() {
                        if slope != 1.1 {
                            errors.push(format!("max_slope is {}, TODO.md requires 1.1", slope));
                        }
                    }
                }
            }
            _ => {
                // Basic validation for other config types
                if parameters.is_empty() {
                    warnings.push("Configuration has no parameters".to_string());
                }
            }
        }

        let status = if !errors.is_empty() {
            ValidationStatus::Invalid(errors)
        } else if !warnings.is_empty() {
            ValidationStatus::Warning(warnings)
        } else {
            ValidationStatus::Valid
        };

        Ok(status)
    }

    /// Save fingerprint to artifact storage
    pub async fn save_fingerprint(&self, fingerprint: &ConfigFingerprint) -> Result<String> {
        let artifact_dir = std::path::Path::new("artifact").join("config");
        tokio::fs::create_dir_all(&artifact_dir).await
            .context("Failed to create config artifact directory")?;

        let filename = format!("config_{}_{}.json", 
            fingerprint.config_type.to_string().to_lowercase(),
            &fingerprint.fingerprint_id[..8]
        );
        
        let filepath = artifact_dir.join(&filename);
        let content = serde_json::to_string_pretty(fingerprint)
            .context("Failed to serialize fingerprint")?;

        tokio::fs::write(&filepath, content).await
            .context("Failed to write fingerprint artifact")?;

        let artifact_path = format!("artifact://config/{}", filename);
        info!("Saved config fingerprint: {} -> {}", fingerprint.fingerprint_id, artifact_path);

        Ok(artifact_path)
    }

    /// Load fingerprint from artifact storage
    pub async fn load_fingerprint(&self, fingerprint_id: &str) -> Result<ConfigFingerprint> {
        let artifact_dir = std::path::Path::new("artifact").join("config");
        
        // Try to find the fingerprint file
        let mut dir_entries = tokio::fs::read_dir(&artifact_dir).await
            .context("Failed to read config artifact directory")?;

        while let Some(entry) = dir_entries.next_entry().await? {
            let filename = entry.file_name().to_string_lossy().to_string();
            if filename.contains(&fingerprint_id[..8]) {
                let filepath = entry.path();
                let content = tokio::fs::read_to_string(&filepath).await
                    .context("Failed to read fingerprint file")?;
                
                let fingerprint: ConfigFingerprint = serde_json::from_str(&content)
                    .context("Failed to deserialize fingerprint")?;
                
                return Ok(fingerprint);
            }
        }

        anyhow::bail!("Fingerprint not found: {}", fingerprint_id);
    }

    /// Verify fingerprint integrity
    pub fn verify_fingerprint(&self, fingerprint: &ConfigFingerprint) -> Result<bool> {
        // Recreate the fingerprint with the same parameters
        let recreated = self.create_fingerprint(
            fingerprint.config_type.clone(),
            fingerprint.parameters.clone(),
        )?;

        Ok(recreated.fingerprint_id == fingerprint.fingerprint_id)
    }
}

impl ConfigType {
    fn to_string(&self) -> &'static str {
        match self {
            ConfigType::LTRTraining => "ltr_training",
            ConfigType::FeatureExtraction => "feature_extraction", 
            ConfigType::HardNegatives => "hard_negatives",
            ConfigType::IsotonicCalibration => "isotonic_calibration",
            ConfigType::SLAEvaluation => "sla_evaluation",
            ConfigType::BaselinePolicy => "baseline_policy",
            ConfigType::SearchConfiguration => "search_configuration",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_baseline_policy_fingerprint() {
        let system = ConfigFingerprintSystem::new().unwrap();
        let fingerprint = system.create_baseline_policy_fingerprint().unwrap();
        
        assert_eq!(fingerprint.config_type, ConfigType::BaselinePolicy);
        assert!(fingerprint.fingerprint_id.len() == 64); // SHA256 hex
        assert!(fingerprint.parameters.contains_key("policy_name"));
        
        let policy_ref = system.create_policy_reference(
            "lexical_struct_only",
            PolicyType::Baseline,
            &fingerprint,
            None,
        );
        
        assert!(policy_ref.uri.starts_with("policy://lexical_struct_only@"));
    }

    #[test] 
    fn test_deterministic_fingerprinting() {
        let system = ConfigFingerprintSystem::new().unwrap();
        let mut parameters = BTreeMap::new();
        parameters.insert("test_param".to_string(), serde_json::Value::String("test_value".to_string()));

        let fp1 = system.create_fingerprint(ConfigType::LTRTraining, parameters.clone()).unwrap();
        let fp2 = system.create_fingerprint(ConfigType::LTRTraining, parameters.clone()).unwrap();

        // Fingerprints should be identical for same parameters
        assert_eq!(fp1.fingerprint_id, fp2.fingerprint_id);
    }

    #[test]
    fn test_fingerprint_verification() {
        let system = ConfigFingerprintSystem::new().unwrap();
        let fingerprint = system.create_baseline_policy_fingerprint().unwrap();
        
        let is_valid = system.verify_fingerprint(&fingerprint).unwrap();
        assert!(is_valid);
    }
}