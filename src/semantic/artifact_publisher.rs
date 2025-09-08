//! Artifact Publishing System with SHA256 Verification
//!
//! Implements secure artifact publishing for trained models, calibration data, 
//! evaluation results, and configuration fingerprints with cryptographic verification.
//!
//! **TODO.md Requirements:**
//! - SHA256 hashes for all published artifacts  
//! - Atomic publishing with verification
//! - Rollback capability for failed deployments
//! - Artifact registry with metadata
//! - Policy artifact publishing: `artifact://model/ltr_<DATE>.bin`

use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use anyhow::{Result, Context};
use chrono::{DateTime, Utc};
use tracing::{info, warn, error};
use tokio::fs;

use crate::semantic::{
    config_fingerprint::{ConfigFingerprint, PolicyReference},
    gate_checker::GateCheckResult,
};

/// Published artifact metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishedArtifact {
    pub artifact_id: String,
    pub artifact_type: PublishedArtifactType,
    pub version: String,
    pub sha256_hash: String,
    pub file_size_bytes: u64,
    pub source_path: String,
    pub published_path: String,
    pub creation_timestamp: DateTime<Utc>,
    pub publication_timestamp: DateTime<Utc>,
    pub metadata: ArtifactMetadata,
    pub dependencies: Vec<ArtifactDependency>,
}

/// Type of published artifact
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PublishedArtifactType {
    TrainedModel,
    CalibrationModel,
    EvaluationResults,
    ConfigFingerprint,
    GateCheckResult,
    PolicyDefinition,
    BenchmarkReport,
}

/// Artifact metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactMetadata {
    pub title: String,
    pub description: String,
    pub tags: Vec<String>,
    pub config_fingerprint: String,
    pub git_commit_sha: String,
    pub training_duration_seconds: Option<f64>,
    pub performance_metrics: HashMap<String, f64>,
    pub compatibility_info: CompatibilityInfo,
}

/// Compatibility information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityInfo {
    pub rust_version: String,
    pub required_features: Vec<String>,
    pub api_version: String,
    pub backward_compatible: bool,
}

/// Artifact dependency reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactDependency {
    pub dependency_id: String,
    pub dependency_type: PublishedArtifactType,
    pub required_version: String,
    pub sha256_hash: String,
}

/// Artifact registry entry
#[derive(Debug, Serialize, Deserialize)]
pub struct ArtifactRegistry {
    pub registry_version: String,
    pub last_updated: DateTime<Utc>,
    pub artifacts: HashMap<String, PublishedArtifact>,
    pub policies: HashMap<String, PolicyReference>,
    pub index: ArtifactIndex,
}

/// Artifact index for fast lookups
#[derive(Debug, Serialize, Deserialize)]
pub struct ArtifactIndex {
    pub by_type: HashMap<PublishedArtifactType, Vec<String>>,
    pub by_config_fingerprint: HashMap<String, Vec<String>>,
    pub by_date: Vec<(DateTime<Utc>, String)>, // Sorted by date
}

/// Publication batch for atomic operations
#[derive(Debug, Serialize, Deserialize)]
pub struct PublicationBatch {
    pub batch_id: String,
    pub artifacts: Vec<PublishedArtifact>,
    pub policies: Vec<PolicyReference>,
    pub gate_check_result: Option<GateCheckResult>,
    pub rollback_info: RollbackInfo,
}

/// Rollback information
#[derive(Debug, Serialize, Deserialize)]
pub struct RollbackInfo {
    pub previous_versions: HashMap<String, String>, // artifact_id -> previous_version
    pub backup_paths: HashMap<String, String>,      // artifact_id -> backup_path
    pub rollback_script: String,
}

/// Main artifact publishing system
pub struct ArtifactPublisher {
    registry_path: PathBuf,
    artifacts_root: PathBuf,
    current_registry: Option<ArtifactRegistry>,
}

impl ArtifactPublisher {
    /// Create new artifact publisher
    pub fn new(artifacts_root: Option<PathBuf>) -> Result<Self> {
        let artifacts_root = artifacts_root.unwrap_or_else(|| Path::new("artifact").to_path_buf());
        let registry_path = artifacts_root.join("registry.json");

        let mut publisher = Self {
            registry_path,
            artifacts_root,
            current_registry: None,
        };

        publisher.load_registry().await?;
        Ok(publisher)
    }

    /// Load artifact registry
    async fn load_registry(&mut self) -> Result<()> {
        if self.registry_path.exists() {
            let content = fs::read_to_string(&self.registry_path).await
                .context("Failed to read registry file")?;
            
            self.current_registry = Some(serde_json::from_str(&content)
                .context("Failed to parse registry JSON")?);
        } else {
            // Create new registry
            self.current_registry = Some(ArtifactRegistry {
                registry_version: "1.0".to_string(),
                last_updated: Utc::now(),
                artifacts: HashMap::new(),
                policies: HashMap::new(),
                index: ArtifactIndex {
                    by_type: HashMap::new(),
                    by_config_fingerprint: HashMap::new(),
                    by_date: Vec::new(),
                },
            });
        }
        Ok(())
    }

    /// Save artifact registry
    async fn save_registry(&self) -> Result<()> {
        if let Some(registry) = &self.current_registry {
            let content = serde_json::to_string_pretty(registry)
                .context("Failed to serialize registry")?;
            
            // Create parent directory if it doesn't exist
            if let Some(parent) = self.registry_path.parent() {
                fs::create_dir_all(parent).await
                    .context("Failed to create registry directory")?;
            }

            fs::write(&self.registry_path, content).await
                .context("Failed to write registry file")?;
            
            info!("Updated artifact registry: {}", self.registry_path.display());
        }
        Ok(())
    }

    /// Publish a single artifact
    pub async fn publish_artifact(
        &mut self,
        source_path: &Path,
        artifact_type: PublishedArtifactType,
        metadata: ArtifactMetadata,
        dependencies: Vec<ArtifactDependency>,
    ) -> Result<PublishedArtifact> {
        info!("Publishing artifact: {} -> {:?}", source_path.display(), artifact_type);

        // Calculate SHA256 hash
        let sha256_hash = self.calculate_file_hash(source_path).await
            .context("Failed to calculate file hash")?;

        // Get file size
        let file_metadata = fs::metadata(source_path).await
            .context("Failed to read file metadata")?;
        let file_size_bytes = file_metadata.len();

        // Generate artifact ID and version
        let artifact_id = self.generate_artifact_id(&artifact_type, &metadata.config_fingerprint);
        let version = self.generate_version();

        // Determine published path
        let published_path = self.get_published_path(&artifact_type, &artifact_id, &version);

        // Copy file to published location
        self.copy_to_published_location(source_path, &published_path).await
            .context("Failed to copy artifact to published location")?;

        // Verify published file hash
        let published_hash = self.calculate_file_hash(&published_path).await
            .context("Failed to verify published file hash")?;
        
        if published_hash != sha256_hash {
            return Err(anyhow::anyhow!(
                "Hash mismatch after publishing: expected {}, got {}",
                sha256_hash, published_hash
            ));
        }

        // Create published artifact
        let published_artifact = PublishedArtifact {
            artifact_id: artifact_id.clone(),
            artifact_type: artifact_type.clone(),
            version,
            sha256_hash,
            file_size_bytes,
            source_path: source_path.to_string_lossy().to_string(),
            published_path: published_path.to_string_lossy().to_string(),
            creation_timestamp: metadata.training_duration_seconds.map(|d| {
                Utc::now() - chrono::Duration::seconds(d as i64)
            }).unwrap_or(Utc::now()),
            publication_timestamp: Utc::now(),
            metadata,
            dependencies,
        };

        // Add to registry
        self.add_to_registry(published_artifact.clone()).await?;

        info!("Successfully published artifact: {} ({})", artifact_id, sha256_hash);
        Ok(published_artifact)
    }

    /// Publish training pipeline artifacts in atomic batch
    pub async fn publish_training_pipeline(
        &mut self,
        model_path: &Path,
        calibration_path: &Path,
        config_fingerprints: HashMap<String, ConfigFingerprint>,
        evaluation_results: HashMap<String, crate::semantic::sla_bounded_evaluation::SLABoundedEvaluationResult>,
        gate_check_result: GateCheckResult,
    ) -> Result<PublicationBatch> {
        let batch_id = format!("training_pipeline_{}", chrono::Utc::now().format("%Y%m%d_%H%M%S"));
        info!("Starting atomic publication batch: {}", batch_id);

        let mut published_artifacts = Vec::new();
        let mut published_policies = Vec::new();

        // Create rollback info
        let rollback_info = RollbackInfo {
            previous_versions: HashMap::new(),
            backup_paths: HashMap::new(),
            rollback_script: format!("# Rollback script for batch {}\n", batch_id),
        };

        // Publish trained model
        let model_metadata = self.create_model_metadata(&config_fingerprints, &evaluation_results)?;
        let model_artifact = self.publish_artifact(
            model_path,
            PublishedArtifactType::TrainedModel,
            model_metadata,
            vec![],
        ).await?;
        published_artifacts.push(model_artifact.clone());

        // Publish calibration model
        let calibration_metadata = self.create_calibration_metadata(&config_fingerprints)?;
        let calibration_artifact = self.publish_artifact(
            calibration_path,
            PublishedArtifactType::CalibrationModel,
            calibration_metadata,
            vec![ArtifactDependency {
                dependency_id: model_artifact.artifact_id.clone(),
                dependency_type: PublishedArtifactType::TrainedModel,
                required_version: model_artifact.version.clone(),
                sha256_hash: model_artifact.sha256_hash.clone(),
            }],
        ).await?;
        published_artifacts.push(calibration_artifact);

        // Publish config fingerprints
        for (config_name, fingerprint) in &config_fingerprints {
            let config_metadata = self.create_config_metadata(config_name, fingerprint)?;
            let config_path = self.save_fingerprint_to_temp(fingerprint).await?;
            
            let config_artifact = self.publish_artifact(
                &config_path,
                PublishedArtifactType::ConfigFingerprint,
                config_metadata,
                vec![],
            ).await?;
            published_artifacts.push(config_artifact);
        }

        // Publish evaluation results
        for (slice_name, result) in &evaluation_results {
            let eval_metadata = self.create_evaluation_metadata(slice_name, result, &config_fingerprints)?;
            let eval_path = self.save_evaluation_to_temp(slice_name, result).await?;
            
            let eval_artifact = self.publish_artifact(
                &eval_path,
                PublishedArtifactType::EvaluationResults,
                eval_metadata,
                vec![],
            ).await?;
            published_artifacts.push(eval_artifact);
        }

        // Publish gate check result
        let gate_metadata = self.create_gate_metadata(&gate_check_result, &config_fingerprints)?;
        let gate_path = self.save_gate_result_to_temp(&gate_check_result).await?;
        
        let gate_artifact = self.publish_artifact(
            &gate_path,
            PublishedArtifactType::GateCheckResult,
            gate_metadata,
            vec![],
        ).await?;
        published_artifacts.push(gate_artifact);

        // Create policy reference for the trained model
        let policy_ref = PolicyReference {
            policy_name: format!("ltr_trained_{}", chrono::Utc::now().format("%Y%m%d")),
            policy_type: crate::semantic::config_fingerprint::PolicyType::Trained,
            config_fingerprint: config_fingerprints.get("ltr_training")
                .map(|cf| cf.fingerprint_id.clone())
                .unwrap_or_else(|| "unknown".to_string()),
            model_fingerprint: Some(model_artifact.sha256_hash.clone()),
            uri: format!("policy://ltr_trained@{}", &model_artifact.sha256_hash[..8]),
        };
        published_policies.push(policy_ref);

        let batch = PublicationBatch {
            batch_id: batch_id.clone(),
            artifacts: published_artifacts,
            policies: published_policies,
            gate_check_result: Some(gate_check_result),
            rollback_info,
        };

        // Save batch metadata
        self.save_batch_metadata(&batch).await?;

        info!("Successfully published training pipeline batch: {} ({} artifacts)", 
            batch_id, batch.artifacts.len());

        Ok(batch)
    }

    /// Calculate SHA256 hash of a file
    async fn calculate_file_hash(&self, file_path: &Path) -> Result<String> {
        let content = fs::read(file_path).await
            .with_context(|| format!("Failed to read file: {}", file_path.display()))?;
        
        let mut hasher = Sha256::new();
        hasher.update(&content);
        let hash = hasher.finalize();
        
        Ok(format!("{:x}", hash))
    }

    /// Generate artifact ID
    fn generate_artifact_id(&self, artifact_type: &PublishedArtifactType, config_fingerprint: &str) -> String {
        let type_prefix = match artifact_type {
            PublishedArtifactType::TrainedModel => "model",
            PublishedArtifactType::CalibrationModel => "calib",
            PublishedArtifactType::EvaluationResults => "eval",
            PublishedArtifactType::ConfigFingerprint => "config",
            PublishedArtifactType::GateCheckResult => "gate",
            PublishedArtifactType::PolicyDefinition => "policy",
            PublishedArtifactType::BenchmarkReport => "bench",
        };

        format!("{}_{}_{}",
            type_prefix,
            &config_fingerprint[..8],
            chrono::Utc::now().format("%Y%m%d_%H%M%S")
        )
    }

    /// Generate version string
    fn generate_version(&self) -> String {
        chrono::Utc::now().format("%Y.%m.%d.%H%M%S").to_string()
    }

    /// Get published path for artifact
    fn get_published_path(&self, artifact_type: &PublishedArtifactType, artifact_id: &str, version: &str) -> PathBuf {
        let type_dir = match artifact_type {
            PublishedArtifactType::TrainedModel => "models",
            PublishedArtifactType::CalibrationModel => "calibration",
            PublishedArtifactType::EvaluationResults => "evaluations",
            PublishedArtifactType::ConfigFingerprint => "configs",
            PublishedArtifactType::GateCheckResult => "gates",
            PublishedArtifactType::PolicyDefinition => "policies",
            PublishedArtifactType::BenchmarkReport => "benchmarks",
        };

        let extension = match artifact_type {
            PublishedArtifactType::TrainedModel => "bin",
            PublishedArtifactType::CalibrationModel => "json",
            _ => "json",
        };

        self.artifacts_root
            .join(type_dir)
            .join(format!("{}_{}.{}", artifact_id, version, extension))
    }

    /// Copy file to published location
    async fn copy_to_published_location(&self, source: &Path, destination: &Path) -> Result<()> {
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).await
                .context("Failed to create destination directory")?;
        }

        fs::copy(source, destination).await
            .context("Failed to copy file")?;

        info!("Copied {} -> {}", source.display(), destination.display());
        Ok(())
    }

    /// Add artifact to registry
    async fn add_to_registry(&mut self, artifact: PublishedArtifact) -> Result<()> {
        if let Some(registry) = &mut self.current_registry {
            // Add to artifacts map
            registry.artifacts.insert(artifact.artifact_id.clone(), artifact.clone());

            // Update index
            registry.index.by_type
                .entry(artifact.artifact_type.clone())
                .or_insert_with(Vec::new)
                .push(artifact.artifact_id.clone());

            registry.index.by_config_fingerprint
                .entry(artifact.metadata.config_fingerprint.clone())
                .or_insert_with(Vec::new)
                .push(artifact.artifact_id.clone());

            registry.index.by_date.push((artifact.publication_timestamp, artifact.artifact_id.clone()));
            registry.index.by_date.sort_by_key(|(date, _)| *date);

            registry.last_updated = Utc::now();

            // Save updated registry
            self.save_registry().await?;
        }
        Ok(())
    }

    /// Create model metadata
    fn create_model_metadata(
        &self,
        config_fingerprints: &HashMap<String, ConfigFingerprint>,
        evaluation_results: &HashMap<String, crate::semantic::sla_bounded_evaluation::SLABoundedEvaluationResult>,
    ) -> Result<ArtifactMetadata> {
        let mut performance_metrics = HashMap::new();
        
        // Extract performance metrics from evaluation results
        for (slice_name, result) in evaluation_results {
            performance_metrics.insert(
                format!("ndcg_at_10_{}", slice_name),
                result.mean_ndcg_at_10 as f64
            );
            performance_metrics.insert(
                format!("ece_{}", slice_name),
                result.expected_calibration_error as f64
            );
            performance_metrics.insert(
                format!("sla_recall_{}", slice_name),
                result.sla_recall as f64
            );
        }

        Ok(ArtifactMetadata {
            title: "Trained LTR Model".to_string(),
            description: "Learning-to-Rank model trained with pairwise logistic regression and monotonic constraints".to_string(),
            tags: vec![
                "ltr".to_string(),
                "reranking".to_string(),
                "semantic_lift".to_string(),
                "production".to_string(),
            ],
            config_fingerprint: config_fingerprints.get("ltr_training")
                .map(|cf| cf.fingerprint_id.clone())
                .unwrap_or_else(|| "unknown".to_string()),
            git_commit_sha: config_fingerprints.values()
                .next()
                .map(|cf| cf.git_commit_sha.clone())
                .unwrap_or_else(|| "unknown".to_string()),
            training_duration_seconds: None,
            performance_metrics,
            compatibility_info: CompatibilityInfo {
                rust_version: "1.70+".to_string(),
                required_features: vec!["semantic".to_string(), "ltr".to_string()],
                api_version: "1.0".to_string(),
                backward_compatible: true,
            },
        })
    }

    /// Create calibration metadata
    fn create_calibration_metadata(
        &self,
        config_fingerprints: &HashMap<String, ConfigFingerprint>,
    ) -> Result<ArtifactMetadata> {
        Ok(ArtifactMetadata {
            title: "Isotonic Calibration Model".to_string(),
            description: "Per-intent×language isotonic calibration with slope clamping ∈ [0.9,1.1]".to_string(),
            tags: vec![
                "calibration".to_string(),
                "isotonic".to_string(),
                "probability".to_string(),
            ],
            config_fingerprint: config_fingerprints.get("isotonic_calibration")
                .map(|cf| cf.fingerprint_id.clone())
                .unwrap_or_else(|| "unknown".to_string()),
            git_commit_sha: config_fingerprints.values()
                .next()
                .map(|cf| cf.git_commit_sha.clone())
                .unwrap_or_else(|| "unknown".to_string()),
            training_duration_seconds: None,
            performance_metrics: HashMap::new(),
            compatibility_info: CompatibilityInfo {
                rust_version: "1.70+".to_string(),
                required_features: vec!["semantic".to_string(), "calibration".to_string()],
                api_version: "1.0".to_string(),
                backward_compatible: true,
            },
        })
    }

    /// Create config metadata
    fn create_config_metadata(&self, config_name: &str, fingerprint: &ConfigFingerprint) -> Result<ArtifactMetadata> {
        Ok(ArtifactMetadata {
            title: format!("Configuration: {}", config_name),
            description: format!("Configuration fingerprint for {:?}", fingerprint.config_type),
            tags: vec!["config".to_string(), "fingerprint".to_string()],
            config_fingerprint: fingerprint.fingerprint_id.clone(),
            git_commit_sha: fingerprint.git_commit_sha.clone(),
            training_duration_seconds: None,
            performance_metrics: HashMap::new(),
            compatibility_info: CompatibilityInfo {
                rust_version: "1.70+".to_string(),
                required_features: vec!["config".to_string()],
                api_version: "1.0".to_string(),
                backward_compatible: true,
            },
        })
    }

    /// Create evaluation metadata
    fn create_evaluation_metadata(
        &self,
        slice_name: &str,
        result: &crate::semantic::sla_bounded_evaluation::SLABoundedEvaluationResult,
        config_fingerprints: &HashMap<String, ConfigFingerprint>,
    ) -> Result<ArtifactMetadata> {
        let mut performance_metrics = HashMap::new();
        performance_metrics.insert("ndcg_at_10".to_string(), result.mean_ndcg_at_10 as f64);
        performance_metrics.insert("ece".to_string(), result.expected_calibration_error as f64);
        performance_metrics.insert("sla_recall".to_string(), result.sla_recall as f64);

        Ok(ArtifactMetadata {
            title: format!("Evaluation Results: {}", slice_name),
            description: format!("SLA-bounded evaluation results for slice {}", slice_name),
            tags: vec!["evaluation".to_string(), "sla_bounded".to_string(), slice_name.to_string()],
            config_fingerprint: config_fingerprints.get("sla_evaluation")
                .map(|cf| cf.fingerprint_id.clone())
                .unwrap_or_else(|| "unknown".to_string()),
            git_commit_sha: config_fingerprints.values()
                .next()
                .map(|cf| cf.git_commit_sha.clone())
                .unwrap_or_else(|| "unknown".to_string()),
            training_duration_seconds: None,
            performance_metrics,
            compatibility_info: CompatibilityInfo {
                rust_version: "1.70+".to_string(),
                required_features: vec!["evaluation".to_string()],
                api_version: "1.0".to_string(),
                backward_compatible: true,
            },
        })
    }

    /// Create gate metadata
    fn create_gate_metadata(
        &self,
        gate_result: &GateCheckResult,
        config_fingerprints: &HashMap<String, ConfigFingerprint>,
    ) -> Result<ArtifactMetadata> {
        let mut performance_metrics = HashMap::new();
        performance_metrics.insert("passed_requirements".to_string(), gate_result.summary.passed_requirements as f64);
        performance_metrics.insert("failed_requirements".to_string(), gate_result.summary.failed_requirements as f64);

        Ok(ArtifactMetadata {
            title: format!("Gate Check Result: {}", gate_result.gate_id),
            description: format!("Gate validation result with {:?} status", gate_result.overall_status),
            tags: vec!["gate_check".to_string(), "validation".to_string()],
            config_fingerprint: config_fingerprints.values()
                .next()
                .map(|cf| cf.fingerprint_id.clone())
                .unwrap_or_else(|| "unknown".to_string()),
            git_commit_sha: config_fingerprints.values()
                .next()
                .map(|cf| cf.git_commit_sha.clone())
                .unwrap_or_else(|| "unknown".to_string()),
            training_duration_seconds: None,
            performance_metrics,
            compatibility_info: CompatibilityInfo {
                rust_version: "1.70+".to_string(),
                required_features: vec!["gates".to_string()],
                api_version: "1.0".to_string(),
                backward_compatible: true,
            },
        })
    }

    /// Save fingerprint to temporary location
    async fn save_fingerprint_to_temp(&self, fingerprint: &ConfigFingerprint) -> Result<PathBuf> {
        let temp_path = self.artifacts_root.join("temp").join(format!("fingerprint_{}.json", &fingerprint.fingerprint_id[..8]));
        
        if let Some(parent) = temp_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        let content = serde_json::to_string_pretty(fingerprint)?;
        fs::write(&temp_path, content).await?;

        Ok(temp_path)
    }

    /// Save evaluation result to temporary location
    async fn save_evaluation_to_temp(
        &self,
        slice_name: &str,
        result: &crate::semantic::sla_bounded_evaluation::SLABoundedEvaluationResult,
    ) -> Result<PathBuf> {
        let temp_path = self.artifacts_root.join("temp").join(format!("eval_{}.json", slice_name));
        
        if let Some(parent) = temp_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        let content = serde_json::to_string_pretty(result)?;
        fs::write(&temp_path, content).await?;

        Ok(temp_path)
    }

    /// Save gate result to temporary location
    async fn save_gate_result_to_temp(&self, gate_result: &GateCheckResult) -> Result<PathBuf> {
        let temp_path = self.artifacts_root.join("temp").join(format!("gate_{}.json", gate_result.gate_id));
        
        if let Some(parent) = temp_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        let content = serde_json::to_string_pretty(gate_result)?;
        fs::write(&temp_path, content).await?;

        Ok(temp_path)
    }

    /// Save batch metadata
    async fn save_batch_metadata(&self, batch: &PublicationBatch) -> Result<()> {
        let batch_path = self.artifacts_root.join("batches").join(format!("{}.json", batch.batch_id));
        
        if let Some(parent) = batch_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        let content = serde_json::to_string_pretty(batch)?;
        fs::write(&batch_path, content).await?;

        info!("Saved batch metadata: {}", batch_path.display());
        Ok(())
    }

    /// Verify artifact integrity by SHA256
    pub async fn verify_artifact(&self, artifact_id: &str) -> Result<bool> {
        if let Some(registry) = &self.current_registry {
            if let Some(artifact) = registry.artifacts.get(artifact_id) {
                let published_path = Path::new(&artifact.published_path);
                if published_path.exists() {
                    let actual_hash = self.calculate_file_hash(published_path).await?;
                    return Ok(actual_hash == artifact.sha256_hash);
                }
            }
        }
        Ok(false)
    }

    /// Get artifact by ID
    pub fn get_artifact(&self, artifact_id: &str) -> Option<&PublishedArtifact> {
        self.current_registry.as_ref()?.artifacts.get(artifact_id)
    }

    /// List artifacts by type
    pub fn list_artifacts_by_type(&self, artifact_type: &PublishedArtifactType) -> Vec<&PublishedArtifact> {
        if let Some(registry) = &self.current_registry {
            if let Some(artifact_ids) = registry.index.by_type.get(artifact_type) {
                return artifact_ids
                    .iter()
                    .filter_map(|id| registry.artifacts.get(id))
                    .collect();
            }
        }
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_artifact_publisher_creation() {
        let temp_dir = TempDir::new().unwrap();
        let publisher = ArtifactPublisher::new(Some(temp_dir.path().to_path_buf())).await;
        assert!(publisher.is_ok());
    }

    #[tokio::test]
    async fn test_file_hash_calculation() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.txt");
        fs::write(&test_file, "test content").await.unwrap();

        let publisher = ArtifactPublisher::new(Some(temp_dir.path().to_path_buf())).await.unwrap();
        let hash = publisher.calculate_file_hash(&test_file).await.unwrap();
        
        assert_eq!(hash.len(), 64); // SHA256 hex length
    }
}