//! Configuration management for Lens
//!
//! This module provides unified configuration management across all components

use anyhow::Result;
use config::{Config, ConfigError, Environment, File, FileFormat};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LensConfig {
    /// Search engine configuration
    pub search: SearchConfig,
    /// LSP server configuration
    pub lsp: LspConfig,
    /// HTTP server configuration
    pub http: HttpConfig,
    /// General application settings
    pub app: AppConfig,
}

/// Search engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Path to the search index directory
    pub index_path: PathBuf,
    /// Maximum number of search results to return
    pub max_results: usize,
    /// Enable fuzzy search by default
    pub enable_fuzzy: bool,
    /// Fuzzy search distance
    pub fuzzy_distance: u8,
    /// Heap size for indexing (MB)
    pub heap_size_mb: usize,
    /// Commit interval for indexing (milliseconds)
    pub commit_interval_ms: u64,
    /// Supported file extensions
    pub supported_extensions: Vec<String>,
    /// Enable query caching
    pub enable_cache: bool,
    /// Cache size (number of queries)
    pub cache_size: usize,
}

/// LSP server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspConfig {
    /// Maximum number of search results for LSP operations
    pub max_search_results: usize,
    /// Enable fuzzy search in LSP
    pub enable_fuzzy_search: bool,
    /// Enable semantic search features
    pub enable_semantic_search: bool,
    /// Debounce delay for search requests (milliseconds)
    pub search_debounce_ms: u64,
    /// Enable result caching in LSP
    pub enable_result_caching: bool,
}

/// HTTP server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpConfig {
    /// Server bind address
    pub bind: String,
    /// Server port
    pub port: u16,
    /// Enable CORS
    pub enable_cors: bool,
    /// Request timeout (seconds)
    pub request_timeout_secs: u64,
    /// Maximum request body size (bytes)
    pub max_body_size: usize,
}

/// General application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Log level (trace, debug, info, warn, error)
    pub log_level: String,
    /// Enable file watching for auto-indexing
    pub enable_file_watching: bool,
    /// Number of worker threads
    pub worker_threads: Option<usize>,
}

impl Default for LensConfig {
    fn default() -> Self {
        Self {
            search: SearchConfig::default(),
            lsp: LspConfig::default(),
            http: HttpConfig::default(),
            app: AppConfig::default(),
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            index_path: PathBuf::from("./index"),
            max_results: 50,
            enable_fuzzy: true,
            fuzzy_distance: 2,
            heap_size_mb: 128,
            commit_interval_ms: 5000,
            supported_extensions: vec![
                ".rs".to_string(),
                ".py".to_string(),
                ".ts".to_string(),
                ".js".to_string(),
                ".go".to_string(),
                ".java".to_string(),
                ".cpp".to_string(),
                ".c".to_string(),
                ".h".to_string(),
                ".hpp".to_string(),
                ".rb".to_string(),
                ".php".to_string(),
                ".swift".to_string(),
                ".kt".to_string(),
                ".scala".to_string(),
                ".clj".to_string(),
                ".ex".to_string(),
                ".exs".to_string(),
                ".md".to_string(),
                ".txt".to_string(),
            ],
            enable_cache: true,
            cache_size: 1000,
        }
    }
}

impl Default for LspConfig {
    fn default() -> Self {
        Self {
            max_search_results: 50,
            enable_fuzzy_search: true,
            enable_semantic_search: false,
            search_debounce_ms: 300,
            enable_result_caching: true,
        }
    }
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1".to_string(),
            port: 3000,
            enable_cors: false,
            request_timeout_secs: 30,
            max_body_size: 1024 * 1024, // 1MB
        }
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            log_level: "info".to_string(),
            enable_file_watching: true,
            worker_threads: None, // Use default runtime threads
        }
    }
}

impl LensConfig {
    /// Load configuration from file and environment variables
    pub fn load() -> Result<Self, ConfigError> {
        let mut builder = Config::builder()
            // Start with default values
            .add_source(config::Config::try_from(&LensConfig::default())?);

        // Add configuration file if it exists
        let mut config_paths = vec![
            PathBuf::from("lens.toml"),
            PathBuf::from("config/lens.toml"),
            PathBuf::from("/etc/lens/lens.toml"),
        ];

        if let Some(config_dir) = dirs::config_dir() {
            config_paths.push(config_dir.join("lens/lens.toml"));
        }

        for path in config_paths {
            if path.exists() {
                builder = builder.add_source(File::from(path).format(FileFormat::Toml));
                break;
            }
        }

        // Add environment variables with LENS_ prefix
        builder = builder.add_source(
            Environment::with_prefix("LENS")
                .prefix_separator("_")
                .separator("__"),
        );

        // Build and deserialize
        let config = builder.build()?;
        config.try_deserialize()
    }

    /// Load configuration from a specific file
    pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, ConfigError> {
        let config = Config::builder()
            .add_source(config::Config::try_from(&LensConfig::default())?)
            .add_source(File::from(path.as_ref()).format(FileFormat::Toml))
            .add_source(
                Environment::with_prefix("LENS")
                    .prefix_separator("_")
                    .separator("__"),
            )
            .build()?;

        config.try_deserialize()
    }

    /// Save configuration to a file
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let toml_string = toml::to_string_pretty(self)?;
        std::fs::write(path, toml_string)?;
        Ok(())
    }

    /// Create a sample configuration file
    pub fn create_sample_config<P: AsRef<std::path::Path>>(path: P) -> Result<()> {
        let config = LensConfig::default();
        config.save_to_file(path)?;
        Ok(())
    }

    /// Get search engine configuration compatible with lens-search-engine
    #[allow(dead_code)]
    pub fn to_search_engine_config(&self) -> lens_search_engine::SearchConfig {
        lens_search_engine::SearchConfig {
            index_path: self.search.index_path.clone(),
            max_results: self.search.max_results,
            cache_size: self.search.cache_size,
            enable_fuzzy: self.search.enable_fuzzy,
            fuzzy_distance: self.search.fuzzy_distance,
            heap_size_mb: self.search.heap_size_mb,
            commit_interval_ms: self.search.commit_interval_ms,
            supported_extensions: self.search.supported_extensions.clone(),
        }
    }

    /// Get LSP server configuration compatible with lens-lsp-server
    #[allow(dead_code)]
    pub fn to_lsp_config(&self) -> lens_lsp_server::LspServerConfig {
        lens_lsp_server::LspServerConfig {
            max_search_results: self.lsp.max_search_results,
            enable_fuzzy_search: self.lsp.enable_fuzzy_search,
            enable_semantic_search: self.lsp.enable_semantic_search,
            search_debounce_ms: self.lsp.search_debounce_ms,
            enable_result_caching: self.lsp.enable_result_caching,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = LensConfig::default();
        assert_eq!(config.search.max_results, 50);
        assert_eq!(config.lsp.max_search_results, 50);
        assert_eq!(config.http.port, 3000);
        assert_eq!(config.app.log_level, "info");
    }

    #[test]
    fn test_config_serialization() {
        let config = LensConfig::default();
        let toml_string = toml::to_string(&config).unwrap();
        assert!(toml_string.contains("max_results"));
        assert!(toml_string.contains("index_path"));
    }

    #[test]
    fn test_config_file_roundtrip() {
        let config = LensConfig::default();
        let temp_file = NamedTempFile::new().unwrap();

        // Save config
        config.save_to_file(temp_file.path()).unwrap();

        // Load config
        let loaded_config = LensConfig::load_from_file(temp_file.path()).unwrap();

        // Compare (we can't use PartialEq directly due to PathBuf)
        assert_eq!(loaded_config.search.max_results, config.search.max_results);
        assert_eq!(
            loaded_config.lsp.max_search_results,
            config.lsp.max_search_results
        );
    }
}
