//! Pinned Dataset Loader Implementation
//!
//! Handles loading and validation of pinned golden datasets for consistent benchmarking.
//! Supports the format described in CLAUDE.md with 390 golden queries in version
//! "08653c1e-2025-09-01T21-51-35-302Z".

use super::types::{
    PinnedDataset, DatasetMetadata, DatasetVersion, ValidationResult, 
    InconsistentQuery, LoadingError, BenchmarkConfig, GoldenQuery
};
use anyhow::{anyhow, Result};
use serde_json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};
use chrono::{DateTime, Utc};

/// Pinned dataset loader with caching and validation capabilities
pub struct PinnedDatasetLoader {
    config: BenchmarkConfig,
    dataset_cache: Arc<RwLock<HashMap<String, Arc<PinnedDataset>>>>,
    available_versions: Arc<RwLock<Vec<DatasetVersion>>>,
}

impl PinnedDatasetLoader {
    /// Create new loader with default configuration
    pub async fn new() -> Result<Self> {
        let config = BenchmarkConfig::default();
        Self::with_config(config).await
    }

    /// Create loader with custom configuration
    pub async fn with_config(config: BenchmarkConfig) -> Result<Self> {
        info!("🔧 Initializing pinned dataset loader");
        debug!("Dataset path: {}", config.dataset_path);

        let loader = Self {
            config,
            dataset_cache: Arc::new(RwLock::new(HashMap::new())),
            available_versions: Arc::new(RwLock::new(Vec::new())),
        };

        // Discover available datasets if enabled
        if loader.config.auto_discover_datasets {
            if let Err(e) = loader.discover_available_versions().await {
                warn!("Failed to discover datasets: {}", e);
                // Continue without discovery - we'll try direct loading
            }
        }

        info!("✅ Pinned dataset loader initialized");
        Ok(loader)
    }

    /// Load the current pinned dataset (golden-pinned-current.json or default version)
    pub async fn load_current_pinned_dataset(&self) -> Result<PinnedDataset> {
        debug!("🔍 Loading current pinned dataset");

        // Try to load current dataset symlink first
        let current_path = PathBuf::from(&self.config.dataset_path)
            .join("golden-pinned-current.json");

        if current_path.exists() {
            info!("📄 Loading current dataset from: {}", current_path.display());
            return self.load_dataset_from_file(&current_path).await;
        }

        // Fall back to default version
        if let Some(ref default_version) = self.config.default_version {
            info!("📄 Loading default version: {}", default_version);
            return self.load_pinned_dataset_version(default_version).await;
        }

        // Try to find any available pinned dataset
        let versions = self.list_available_versions().await?;
        if !versions.is_empty() {
            let latest = &versions[0]; // Assume first is latest
            info!("📄 Loading latest available version: {}", latest.version);
            return self.load_dataset_from_file(&latest.file_path).await;
        }

        // Last resort: create a mock dataset for development
        warn!("⚠️ No pinned datasets found, creating mock dataset for development");
        Ok(self.create_mock_dataset())
    }

    /// Load specific pinned dataset version
    pub async fn load_pinned_dataset_version(&self, version: &str) -> Result<PinnedDataset> {
        debug!("🔍 Loading pinned dataset version: {}", version);

        // Check cache first if caching is enabled
        if self.config.enable_caching {
            let cache = self.dataset_cache.read().await;
            if let Some(cached) = cache.get(version) {
                info!("💾 Loaded dataset from cache: {}", version);
                return Ok((**cached).clone());
            }
        }

        // Construct file path for the version
        let filename = format!("golden-pinned-{}.json", version);
        let file_path = PathBuf::from(&self.config.dataset_path).join(&filename);

        if !file_path.exists() {
            return Err(anyhow!(LoadingError::VersionNotFound {
                version: version.to_string()
            }));
        }

        info!("📄 Loading dataset from: {}", file_path.display());
        let dataset = self.load_dataset_from_file(&file_path).await?;

        // Cache the loaded dataset if caching is enabled
        if self.config.enable_caching {
            self.cache_dataset(version, &dataset).await;
        }

        Ok(dataset)
    }

    /// Load dataset from a specific file path
    async fn load_dataset_from_file(&self, file_path: &Path) -> Result<PinnedDataset> {
        let start_time = std::time::Instant::now();

        let content = fs::read_to_string(file_path).await
            .map_err(|e| anyhow!(LoadingError::IoError { source: e }))?;

        // Try to parse as PinnedDataset format first
        if let Ok(dataset) = serde_json::from_str::<PinnedDataset>(&content) {
            let duration = start_time.elapsed();
            info!("✅ Loaded pinned dataset: {} queries in {:?}", 
                  dataset.queries.len(), duration);
            return Ok(dataset);
        }

        // Fall back to simple JSON array format (legacy compatibility)
        match serde_json::from_str::<Vec<GoldenQuery>>(&content) {
            Ok(queries) => {
                info!("📝 Converting legacy dataset format to PinnedDataset");
                let dataset = self.convert_legacy_to_pinned(queries, file_path)?;
                
                let duration = start_time.elapsed();
                info!("✅ Converted and loaded dataset: {} queries in {:?}", 
                      dataset.queries.len(), duration);
                Ok(dataset)
            }
            Err(e) => {
                error!("❌ Failed to parse dataset from {}: {}", file_path.display(), e);
                Err(anyhow!(LoadingError::JsonParseError { source: e }))
            }
        }
    }

    /// Convert legacy query array format to full PinnedDataset
    fn convert_legacy_to_pinned(&self, queries: Vec<GoldenQuery>, file_path: &Path) -> Result<PinnedDataset> {
        let version = file_path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .replace("golden-pinned-", "");

        let mut query_distribution = HashMap::new();
        let mut languages = std::collections::HashSet::new();

        for query in &queries {
            *query_distribution.entry(query.query_type.clone()).or_insert(0) += 1;
            if let Some(ref lang) = query.language {
                languages.insert(lang.clone());
            }
        }

        let metadata = DatasetMetadata {
            version: version.clone(),
            name: "storyviz-pinned".to_string(),
            created_at: Utc::now(),
            total_queries: queries.len(),
            query_distribution,
            languages: languages.into_iter().collect(),
            git_sha: None,
            additional_metadata: HashMap::new(),
        };

        // Create basic slices
        let mut slices = HashMap::new();
        let all_indices: Vec<usize> = (0..queries.len()).collect();
        slices.insert("ALL".to_string(), all_indices.clone());
        
        // Create SMOKE_DEFAULT slice (first 40 queries as mentioned in CLAUDE.md)
        let smoke_size = std::cmp::min(40, queries.len());
        slices.insert("SMOKE_DEFAULT".to_string(), (0..smoke_size).collect());

        Ok(PinnedDataset {
            metadata,
            queries,
            slices,
            corpus_info: None,
        })
    }

    /// Validate dataset consistency against corpus
    pub async fn validate_dataset_consistency(&self, dataset: &PinnedDataset) -> Result<ValidationResult> {
        info!("🔍 Validating corpus consistency for {} queries", dataset.queries.len());
        
        let mut valid_queries = 0;
        let mut inconsistent_queries = Vec::new();

        // Check for indexed-content or similar corpus directory
        let corpus_paths = [
            "indexed-content",
            "benchmark-corpus", 
            "src",
            "rust-core/src"
        ];

        let corpus_path = corpus_paths.iter()
            .find(|path| std::path::Path::new(path).exists())
            .map(|&s| s.to_string());

        if corpus_path.is_none() {
            warn!("⚠️ No corpus directory found for validation");
            return Ok(ValidationResult {
                is_consistent: false,
                valid_queries: 0,
                total_queries: dataset.queries.len(),
                inconsistent_queries,
                validated_at: Utc::now(),
            });
        }

        let corpus_path = corpus_path.unwrap();
        debug!("📁 Using corpus path: {}", corpus_path);

        // Validate each query
        for query in &dataset.queries {
            let mut query_valid = true;
            let mut missing_files = Vec::new();

            // Check if expected files exist in corpus
            for expected_file in &query.expected_files {
                // Handle different path formats
                let file_paths_to_check = vec![
                    PathBuf::from(&corpus_path).join(expected_file),
                    PathBuf::from(expected_file),
                    PathBuf::from(&corpus_path).join(
                        expected_file.strip_prefix("indexed-content/").unwrap_or(expected_file)
                    ),
                ];

                let file_exists = file_paths_to_check.iter().any(|path| path.exists());
                
                if !file_exists {
                    query_valid = false;
                    missing_files.push(expected_file.clone());
                }
            }

            if query_valid {
                valid_queries += 1;
            } else {
                inconsistent_queries.push(InconsistentQuery {
                    query: query.clone(),
                    failure_reason: "Expected files not found in corpus".to_string(),
                    missing_files,
                });
            }
        }

        let is_consistent = valid_queries == dataset.queries.len();
        let consistency_rate = valid_queries as f64 / dataset.queries.len() as f64 * 100.0;

        if is_consistent {
            info!("✅ Perfect corpus consistency: {}/{} queries (100%)", 
                  valid_queries, dataset.queries.len());
        } else {
            warn!("⚠️ Partial corpus consistency: {}/{} queries ({:.1}%)", 
                  valid_queries, dataset.queries.len(), consistency_rate);
        }

        Ok(ValidationResult {
            is_consistent,
            valid_queries,
            total_queries: dataset.queries.len(),
            inconsistent_queries,
            validated_at: Utc::now(),
        })
    }

    /// Discover available dataset versions in the dataset directory
    async fn discover_available_versions(&self) -> Result<()> {
        debug!("🔍 Discovering available dataset versions");
        
        let dataset_dir = Path::new(&self.config.dataset_path);
        if !dataset_dir.exists() {
            warn!("Dataset directory does not exist: {}", dataset_dir.display());
            return Ok(());
        }

        let mut entries = fs::read_dir(dataset_dir).await?;
        let mut versions = Vec::new();

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            let filename = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");

            // Look for pinned dataset files
            if filename.starts_with("golden-pinned-") && filename.ends_with(".json") {
                let version = filename
                    .strip_prefix("golden-pinned-")
                    .and_then(|s| s.strip_suffix(".json"))
                    .unwrap_or("unknown")
                    .to_string();

                if version != "current" { // Skip the symlink
                    let metadata = entry.metadata().await?;
                    let created_at = metadata.created().ok()
                        .and_then(|t| DateTime::from_timestamp(
                            t.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs() as i64, 0
                        ))
                        .unwrap_or_else(|| Utc::now());

                    // Try to get query count by loading the dataset
                    let query_count = match self.load_dataset_from_file(&path).await {
                        Ok(dataset) => dataset.queries.len(),
                        Err(_) => 0,
                    };

                    versions.push(DatasetVersion {
                        version: version.clone(),
                        created_at,
                        query_count,
                        file_path: path,
                        is_current: version == self.config.default_version.as_deref().unwrap_or(""),
                    });
                }
            }
        }

        // Sort by creation time (newest first)
        versions.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        let count = versions.len();
        *self.available_versions.write().await = versions;
        
        info!("📊 Discovered {} dataset versions", count);
        Ok(())
    }

    /// List available dataset versions
    pub async fn list_available_versions(&self) -> Result<Vec<DatasetVersion>> {
        Ok(self.available_versions.read().await.clone())
    }

    /// Cache a dataset in memory
    async fn cache_dataset(&self, version: &str, dataset: &PinnedDataset) {
        if !self.config.enable_caching {
            return;
        }

        let mut cache = self.dataset_cache.write().await;
        
        // Enforce cache size limit
        if cache.len() >= self.config.max_cache_size {
            // Remove oldest entry (simple LRU approximation)
            if let Some(oldest_key) = cache.keys().next().cloned() {
                cache.remove(&oldest_key);
            }
        }

        cache.insert(version.to_string(), Arc::new(dataset.clone()));
        debug!("💾 Cached dataset version: {}", version);
    }

    /// Create a mock dataset for development when no pinned datasets exist
    fn create_mock_dataset(&self) -> PinnedDataset {
        let mock_queries = vec![
            GoldenQuery {
                query: "function".to_string(),
                expected_files: vec!["src/main.rs".to_string()],
                query_type: super::types::QueryType::Identifier,
                metadata: HashMap::new(),
                language: Some("rust".to_string()),
                confidence: Some(0.8),
            },
            GoldenQuery {
                query: "struct SearchEngine".to_string(),
                expected_files: vec!["src/search.rs".to_string()],
                query_type: super::types::QueryType::Identifier,
                metadata: HashMap::new(),
                language: Some("rust".to_string()),
                confidence: Some(0.9),
            },
        ];

        let mut query_distribution = HashMap::new();
        query_distribution.insert(super::types::QueryType::Identifier, mock_queries.len());

        let metadata = DatasetMetadata {
            version: "mock-dev".to_string(),
            name: "mock-development-dataset".to_string(),
            created_at: Utc::now(),
            total_queries: mock_queries.len(),
            query_distribution,
            languages: vec!["rust".to_string()],
            git_sha: None,
            additional_metadata: HashMap::new(),
        };

        let mut slices = HashMap::new();
        slices.insert("ALL".to_string(), vec![0, 1]);
        slices.insert("SMOKE_DEFAULT".to_string(), vec![0, 1]);

        PinnedDataset {
            metadata,
            queries: mock_queries,
            slices,
            corpus_info: None,
        }
    }

    /// Get dataset slice by name
    pub fn get_dataset_slice(&self, dataset: &PinnedDataset, slice_name: &str) -> Option<Vec<GoldenQuery>> {
        if let Some(indices) = dataset.slices.get(slice_name) {
            let slice_queries = indices.iter()
                .filter_map(|&idx| dataset.queries.get(idx).cloned())
                .collect();
            Some(slice_queries)
        } else {
            None
        }
    }

    /// Get SMOKE test dataset (subset for quick validation)
    pub fn get_smoke_dataset(&self, dataset: &PinnedDataset) -> Vec<GoldenQuery> {
        self.get_dataset_slice(dataset, "SMOKE_DEFAULT")
            .unwrap_or_else(|| {
                // Fallback: take first 40 queries
                let smoke_size = std::cmp::min(40, dataset.queries.len());
                dataset.queries.iter().take(smoke_size).cloned().collect()
            })
    }

    /// Clear dataset cache
    pub async fn clear_cache(&self) {
        let mut cache = self.dataset_cache.write().await;
        cache.clear();
        info!("🧹 Cleared dataset cache");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tokio::fs::File;
    use tokio::io::AsyncWriteExt;

    async fn create_test_dataset_file(dir: &Path, filename: &str, queries: Vec<GoldenQuery>) -> Result<PathBuf> {
        let file_path = dir.join(filename);
        let mut file = File::create(&file_path).await?;
        
        // Extract version from filename (e.g., "golden-pinned-test-version.json" -> "test-version")
        let version = filename
            .strip_prefix("golden-pinned-")
            .and_then(|s| s.strip_suffix(".json"))
            .unwrap_or("test");
        
        let dataset = PinnedDataset {
            metadata: DatasetMetadata {
                version: version.to_string(),
                name: "test-dataset".to_string(),
                created_at: Utc::now(),
                total_queries: queries.len(),
                query_distribution: HashMap::new(),
                languages: vec!["rust".to_string()],
                git_sha: None,
                additional_metadata: HashMap::new(),
            },
            queries,
            slices: HashMap::new(),
            corpus_info: None,
        };

        let json = serde_json::to_string_pretty(&dataset)?;
        file.write_all(json.as_bytes()).await?;
        
        Ok(file_path)
    }

    #[tokio::test]
    async fn test_pinned_loader_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = BenchmarkConfig {
            dataset_path: temp_dir.path().to_string_lossy().to_string(),
            auto_discover_datasets: false,
            ..Default::default()
        };

        let loader = PinnedDatasetLoader::with_config(config).await.unwrap();
        assert!(!loader.config.dataset_path.is_empty());
    }

    #[tokio::test]
    async fn test_mock_dataset_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = BenchmarkConfig {
            dataset_path: temp_dir.path().to_string_lossy().to_string(),
            auto_discover_datasets: false,
            default_version: None,  // No default version in temp dir, should fall back to mock
            ..Default::default()
        };

        let loader = PinnedDatasetLoader::with_config(config).await.unwrap();
        let dataset = loader.load_current_pinned_dataset().await.unwrap();
        
        assert!(!dataset.queries.is_empty());
        assert_eq!(dataset.metadata.version, "mock-dev");
    }

    #[tokio::test]
    async fn test_dataset_loading_and_caching() {
        let temp_dir = TempDir::new().unwrap();
        let config = BenchmarkConfig {
            dataset_path: temp_dir.path().to_string_lossy().to_string(),
            enable_caching: true,
            auto_discover_datasets: false,
            default_version: None,  // No default version in temp dir
            ..Default::default()
        };

        // Create a test dataset file
        let test_queries = vec![
            GoldenQuery {
                query: "test query".to_string(),
                expected_files: vec!["test.rs".to_string()],
                query_type: super::super::types::QueryType::Identifier,
                metadata: HashMap::new(),
                language: None,
                confidence: None,
            }
        ];

        let _file_path = create_test_dataset_file(
            temp_dir.path(), 
            "golden-pinned-test-version.json", 
            test_queries
        ).await.unwrap();

        let loader = PinnedDatasetLoader::with_config(config).await.unwrap();
        
        // Load dataset twice - second should come from cache
        let dataset1 = loader.load_pinned_dataset_version("test-version").await.unwrap();
        let dataset2 = loader.load_pinned_dataset_version("test-version").await.unwrap();
        
        assert_eq!(dataset1.queries.len(), dataset2.queries.len());
        assert_eq!(dataset1.metadata.version, "test-version");
    }
}