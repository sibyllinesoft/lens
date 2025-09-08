//! # Docker Image Manifest Generator
//!
//! Creates reproducible Docker image manifests with digests
//! for the replication pack as specified in TODO.md Step 2(b).
//!
//! Generates:
//! - Docker image digests for exact reproduction
//! - Container environment specifications
//! - Build reproducibility metadata

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Stdio;
use tokio::process::Command;
use tracing::{info, warn};

/// Complete Docker environment manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerManifest {
    pub metadata: DockerMetadata,
    pub base_images: Vec<DockerImage>,
    pub application_images: Vec<ApplicationImage>,
    pub environment_spec: EnvironmentSpec,
    pub build_configuration: BuildConfiguration,
    pub reproduction_commands: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerMetadata {
    pub version: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub docker_version: String,
    pub docker_compose_version: Option<String>,
    pub host_platform: String,
    pub architecture: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerImage {
    pub name: String,
    pub tag: String,
    pub digest: String,
    pub size_bytes: u64,
    pub architecture: String,
    pub os: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub layers: Vec<LayerInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplicationImage {
    pub name: String,
    pub tag: String,
    pub digest: String,
    pub size_bytes: u64,
    pub dockerfile_path: String,
    pub build_context: String,
    pub build_args: HashMap<String, String>,
    pub environment_variables: HashMap<String, String>,
    pub exposed_ports: Vec<u16>,
    pub volumes: Vec<String>,
    pub base_image: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    pub digest: String,
    pub size_bytes: u64,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub created_by: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentSpec {
    pub cpu_architecture: String,
    pub memory_limit: Option<String>,
    pub cpu_limit: Option<String>,
    pub network_mode: String,
    pub dns_servers: Vec<String>,
    pub environment_variables: HashMap<String, String>,
    pub user_id: Option<u32>,
    pub group_id: Option<u32>,
    pub working_directory: String,
    pub entrypoint: Vec<String>,
    pub command: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildConfiguration {
    pub dockerfile_content: String,
    pub dockerignore_content: Option<String>,
    pub build_timestamp: chrono::DateTime<chrono::Utc>,
    pub build_platform: String,
    pub build_tools: BuildTools,
    pub reproducibility_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildTools {
    pub docker_version: String,
    pub buildkit_version: Option<String>,
    pub buildx_version: Option<String>,
    pub compose_version: Option<String>,
}

/// Docker manifest generator
pub struct DockerManifestGenerator {
    project_root: std::path::PathBuf,
    image_names: Vec<String>,
}

impl DockerManifestGenerator {
    /// Create new Docker manifest generator
    pub fn new(project_root: impl AsRef<std::path::Path>) -> Self {
        Self {
            project_root: project_root.as_ref().to_path_buf(),
            image_names: vec![
                "lens:latest".to_string(),
                "postgres:15".to_string(),
                "redis:7-alpine".to_string(),
                "nginx:alpine".to_string(),
            ],
        }
    }

    /// Add custom image to manifest
    pub fn add_image(&mut self, image_name: String) {
        self.image_names.push(image_name);
    }

    /// Generate complete Docker manifest
    pub async fn generate_manifest(&self) -> Result<DockerManifest> {
        info!("Generating Docker manifest for {} images", self.image_names.len());

        // Get Docker system info
        let metadata = self.get_docker_metadata().await?;
        
        // Get base images (external dependencies)
        let mut base_images = Vec::new();
        let mut application_images = Vec::new();

        for image_name in &self.image_names {
            if self.is_application_image(image_name) {
                let app_image = self.get_application_image_info(image_name).await?;
                application_images.push(app_image);
            } else {
                let base_image = self.get_base_image_info(image_name).await?;
                base_images.push(base_image);
            }
        }

        // Get environment specification
        let environment_spec = self.get_environment_spec().await?;

        // Get build configuration
        let build_configuration = self.get_build_configuration().await?;

        // Generate reproduction commands
        let reproduction_commands = self.generate_reproduction_commands(&base_images, &application_images);

        Ok(DockerManifest {
            metadata,
            base_images,
            application_images,
            environment_spec,
            build_configuration,
            reproduction_commands,
        })
    }

    /// Get Docker system metadata
    async fn get_docker_metadata(&self) -> Result<DockerMetadata> {
        // Get Docker version
        let docker_version = self.run_command("docker", &["--version"]).await?;
        let docker_version = self.extract_version(&docker_version, "Docker version");

        // Get Docker Compose version (optional)
        let compose_version = self.run_command("docker-compose", &["--version"]).await.ok()
            .and_then(|output| self.extract_version(&output, "docker-compose version"));

        // Get host platform info
        let platform_info = self.run_command("docker", &["version", "--format", "{{.Server.Platform.Name}}"]).await?;
        
        // Get architecture
        let arch_info = self.run_command("docker", &["version", "--format", "{{.Server.Arch}}"]).await?;

        Ok(DockerMetadata {
            version: "1.0.0".to_string(),
            created_at: chrono::Utc::now(),
            docker_version,
            docker_compose_version: compose_version,
            host_platform: platform_info.trim().to_string(),
            architecture: arch_info.trim().to_string(),
        })
    }

    /// Get base image information
    async fn get_base_image_info(&self, image_name: &str) -> Result<DockerImage> {
        info!("Getting base image info for: {}", image_name);

        // Pull image to ensure we have the latest
        self.run_command("docker", &["pull", image_name]).await?;

        // Get image digest
        let digest_output = self.run_command("docker", &[
            "images", "--digests", "--format", "{{.Digest}}", image_name
        ]).await?;
        let digest = digest_output.trim().to_string();

        // Get image inspect info
        let inspect_output = self.run_command("docker", &[
            "inspect", "--format", "{{json .}}", image_name
        ]).await?;
        
        let image_info: serde_json::Value = serde_json::from_str(&inspect_output)
            .context("Failed to parse Docker inspect output")?;

        // Extract image details
        let config = &image_info["Config"];
        let size = image_info["Size"].as_u64().unwrap_or(0);
        let architecture = image_info["Architecture"].as_str().unwrap_or("unknown").to_string();
        let os = image_info["Os"].as_str().unwrap_or("unknown").to_string();
        
        let created_str = image_info["Created"].as_str().unwrap_or("");
        let created_at = chrono::DateTime::parse_from_rfc3339(created_str)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now());

        // Get layer information
        let layers = self.get_layer_info(image_name).await?;

        // Parse name and tag
        let (name, tag) = self.parse_image_name(image_name);

        Ok(DockerImage {
            name,
            tag,
            digest,
            size_bytes: size,
            architecture,
            os,
            created_at,
            layers,
        })
    }

    /// Get application image information
    async fn get_application_image_info(&self, image_name: &str) -> Result<ApplicationImage> {
        info!("Getting application image info for: {}", image_name);

        // Get image inspect info
        let inspect_output = self.run_command("docker", &[
            "inspect", "--format", "{{json .}}", image_name
        ]).await?;
        
        let image_info: serde_json::Value = serde_json::from_str(&inspect_output)
            .context("Failed to parse Docker inspect output")?;

        let config = &image_info["Config"];
        let size = image_info["Size"].as_u64().unwrap_or(0);

        // Get digest
        let digest_output = self.run_command("docker", &[
            "images", "--digests", "--format", "{{.Digest}}", image_name
        ]).await?;
        let digest = digest_output.trim().to_string();

        // Extract configuration details
        let env_vars = self.extract_env_vars(&config["Env"]);
        let exposed_ports = self.extract_exposed_ports(&config["ExposedPorts"]);
        let volumes = self.extract_volumes(&config["Volumes"]);

        // Get Dockerfile path (assume standard location)
        let dockerfile_path = self.find_dockerfile_path(image_name);

        // Parse name and tag
        let (name, tag) = self.parse_image_name(image_name);

        Ok(ApplicationImage {
            name,
            tag,
            digest,
            size_bytes: size,
            dockerfile_path,
            build_context: ".".to_string(),
            build_args: HashMap::new(), // Would be extracted from build history
            environment_variables: env_vars,
            exposed_ports,
            volumes,
            base_image: "node:18-alpine".to_string(), // Would be extracted from Dockerfile
        })
    }

    /// Get layer information for an image
    async fn get_layer_info(&self, image_name: &str) -> Result<Vec<LayerInfo>> {
        let history_output = self.run_command("docker", &[
            "history", "--format", "{{json .}}", "--no-trunc", image_name
        ]).await?;

        let mut layers = Vec::new();
        for line in history_output.lines() {
            if let Ok(layer_data) = serde_json::from_str::<serde_json::Value>(line) {
                let digest = layer_data["ID"].as_str().unwrap_or("").to_string();
                let size = layer_data["Size"].as_str().unwrap_or("0B");
                let created_by = layer_data["CreatedBy"].as_str().unwrap_or("").to_string();
                let created_since = layer_data["CreatedSince"].as_str().unwrap_or("");

                // Parse size (simplified - would need better parsing in production)
                let size_bytes = self.parse_size_string(size);

                layers.push(LayerInfo {
                    digest,
                    size_bytes,
                    created_at: chrono::Utc::now(), // Would parse from history
                    created_by: created_by.chars().take(100).collect(), // Truncate long commands
                });
            }
        }

        Ok(layers)
    }

    /// Get environment specification
    async fn get_environment_spec(&self) -> Result<EnvironmentSpec> {
        Ok(EnvironmentSpec {
            cpu_architecture: "x86_64".to_string(),
            memory_limit: Some("4G".to_string()),
            cpu_limit: Some("2.0".to_string()),
            network_mode: "bridge".to_string(),
            dns_servers: vec!["8.8.8.8".to_string(), "8.8.4.4".to_string()],
            environment_variables: HashMap::new(),
            user_id: Some(1000),
            group_id: Some(1000),
            working_directory: "/app".to_string(),
            entrypoint: vec!["/app/entrypoint.sh".to_string()],
            command: vec!["npm".to_string(), "start".to_string()],
        })
    }

    /// Get build configuration
    async fn get_build_configuration(&self) -> Result<BuildConfiguration> {
        // Read Dockerfile
        let dockerfile_path = self.project_root.join("Dockerfile");
        let dockerfile_content = tokio::fs::read_to_string(&dockerfile_path).await
            .unwrap_or_else(|_| "# Dockerfile not found".to_string());

        // Read .dockerignore
        let dockerignore_path = self.project_root.join(".dockerignore");
        let dockerignore_content = tokio::fs::read_to_string(&dockerignore_path).await.ok();

        // Get build tools versions
        let docker_version = self.run_command("docker", &["--version"]).await?;
        let docker_version = self.extract_version(&docker_version, "Docker version");

        let build_tools = BuildTools {
            docker_version,
            buildkit_version: None, // Would be extracted if available
            buildx_version: None,   // Would be extracted if available
            compose_version: None,  // Would be extracted if available
        };

        // Calculate reproducibility hash
        let reproducibility_hash = self.calculate_reproducibility_hash(&dockerfile_content);

        Ok(BuildConfiguration {
            dockerfile_content,
            dockerignore_content,
            build_timestamp: chrono::Utc::now(),
            build_platform: "linux/amd64".to_string(),
            build_tools,
            reproducibility_hash,
        })
    }

    /// Generate reproduction commands
    fn generate_reproduction_commands(&self, base_images: &[DockerImage], app_images: &[ApplicationImage]) -> Vec<String> {
        let mut commands = Vec::new();

        // Pull base images with specific digests
        for image in base_images {
            commands.push(format!("docker pull {}@{}", image.name, image.digest));
        }

        // Build application images
        for image in app_images {
            let build_cmd = format!(
                "docker build -t {}:{} --platform linux/amd64 .",
                image.name, image.tag
            );
            commands.push(build_cmd);
        }

        // Docker compose up command
        commands.push("docker-compose up -d".to_string());
        
        // Health check commands
        commands.push("docker ps --format 'table {{.Names}}\\t{{.Status}}\\t{{.Ports}}'".to_string());

        commands
    }

    /// Helper functions
    async fn run_command(&self, cmd: &str, args: &[&str]) -> Result<String> {
        let output = Command::new(cmd)
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .with_context(|| format!("Failed to run command: {} {}", cmd, args.join(" ")))?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Command failed: {} {} - {}", cmd, args.join(" "), stderr);
        }
    }

    fn extract_version(&self, output: &str, prefix: &str) -> String {
        output.lines()
            .find(|line| line.contains(prefix))
            .and_then(|line| {
                line.split_whitespace()
                    .find(|part| part.chars().next().map(|c| c.is_digit(10)).unwrap_or(false))
            })
            .unwrap_or("unknown")
            .to_string()
    }

    fn parse_image_name(&self, image_name: &str) -> (String, String) {
        if let Some((name, tag)) = image_name.rsplit_once(':') {
            (name.to_string(), tag.to_string())
        } else {
            (image_name.to_string(), "latest".to_string())
        }
    }

    fn is_application_image(&self, image_name: &str) -> bool {
        image_name.starts_with("lens:") || image_name.starts_with("local/")
    }

    fn extract_env_vars(&self, env_json: &serde_json::Value) -> HashMap<String, String> {
        let mut env_vars = HashMap::new();
        if let Some(env_array) = env_json.as_array() {
            for env_var in env_array {
                if let Some(env_str) = env_var.as_str() {
                    if let Some((key, value)) = env_str.split_once('=') {
                        env_vars.insert(key.to_string(), value.to_string());
                    }
                }
            }
        }
        env_vars
    }

    fn extract_exposed_ports(&self, ports_json: &serde_json::Value) -> Vec<u16> {
        let mut ports = Vec::new();
        if let Some(ports_obj) = ports_json.as_object() {
            for key in ports_obj.keys() {
                if let Some(port_str) = key.split('/').next() {
                    if let Ok(port) = port_str.parse::<u16>() {
                        ports.push(port);
                    }
                }
            }
        }
        ports
    }

    fn extract_volumes(&self, volumes_json: &serde_json::Value) -> Vec<String> {
        let mut volumes = Vec::new();
        if let Some(volumes_obj) = volumes_json.as_object() {
            for key in volumes_obj.keys() {
                volumes.push(key.clone());
            }
        }
        volumes
    }

    fn find_dockerfile_path(&self, _image_name: &str) -> String {
        // Simplified - would search for Dockerfile based on image name
        "Dockerfile".to_string()
    }

    fn parse_size_string(&self, size_str: &str) -> u64 {
        // Simplified size parsing - would need proper implementation
        if size_str.ends_with("MB") {
            size_str.trim_end_matches("MB").parse::<u64>().unwrap_or(0) * 1024 * 1024
        } else if size_str.ends_with("KB") {
            size_str.trim_end_matches("KB").parse::<u64>().unwrap_or(0) * 1024
        } else if size_str.ends_with("GB") {
            size_str.trim_end_matches("GB").parse::<u64>().unwrap_or(0) * 1024 * 1024 * 1024
        } else {
            size_str.chars().filter(|c| c.is_digit(10)).collect::<String>().parse().unwrap_or(0)
        }
    }

    fn calculate_reproducibility_hash(&self, dockerfile_content: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(dockerfile_content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Save manifest to file
    pub async fn save_manifest(&self, manifest: &DockerManifest, output_path: &std::path::Path) -> Result<()> {
        let json = serde_json::to_string_pretty(manifest)
            .context("Failed to serialize Docker manifest")?;
        
        if let Some(parent) = output_path.parent() {
            tokio::fs::create_dir_all(parent).await
                .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
        }
        
        tokio::fs::write(output_path, json).await
            .with_context(|| format!("Failed to write Docker manifest: {}", output_path.display()))?;
        
        info!("Docker manifest saved to: {}", output_path.display());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_parse_image_name() {
        let generator = DockerManifestGenerator::new(".");
        
        let (name, tag) = generator.parse_image_name("postgres:15");
        assert_eq!(name, "postgres");
        assert_eq!(tag, "15");
        
        let (name, tag) = generator.parse_image_name("redis");
        assert_eq!(name, "redis");
        assert_eq!(tag, "latest");
    }

    #[tokio::test]
    async fn test_size_parsing() {
        let generator = DockerManifestGenerator::new(".");
        
        assert_eq!(generator.parse_size_string("100MB"), 100 * 1024 * 1024);
        assert_eq!(generator.parse_size_string("5KB"), 5 * 1024);
        assert_eq!(generator.parse_size_string("2GB"), 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_reproducibility_hash() {
        let generator = DockerManifestGenerator::new(".");
        let dockerfile1 = "FROM node:18\nCOPY . /app";
        let dockerfile2 = "FROM node:18\nCOPY . /app";
        let dockerfile3 = "FROM node:18\nCOPY . /different";
        
        let hash1 = generator.calculate_reproducibility_hash(dockerfile1);
        let hash2 = generator.calculate_reproducibility_hash(dockerfile2);
        let hash3 = generator.calculate_reproducibility_hash(dockerfile3);
        
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }
}