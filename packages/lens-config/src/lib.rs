//! Shared configuration loader for Lens components.
//!
//! This crate hosts the `LensConfig` structure and helpers for loading
//! configuration from files and environment variables. It reuses the concrete
//! configuration types from the search engine and LSP crates so there is a
//! single source of truth for tunable settings.

use std::path::{Path, PathBuf};

use anyhow::Result;
use config::{Config, ConfigError, Environment, File, FileFormat};
use lens_lsp_server::LspServerConfig;
use lens_search_engine::SearchConfig;
use serde::{Deserialize, Serialize};

/// Top-level configuration consumed by the Lens application.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LensConfig {
    /// Search engine configuration
    pub search: SearchConfig,
    /// LSP server configuration
    pub lsp: LspServerConfig,
    /// HTTP server configuration
    pub http: HttpConfig,
    /// Telemetry and observability configuration
    pub telemetry: TelemetryConfig,
    /// General application settings
    pub app: AppConfig,
}

/// HTTP server configuration options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpConfig {
    /// Server bind address
    pub bind: String,
    /// Server port
    pub port: u16,
    /// Enable CORS handling
    pub enable_cors: bool,
    /// Request timeout (seconds)
    pub request_timeout_secs: u64,
    /// Maximum request body size (bytes)
    pub max_body_size: usize,
    /// Authentication controls for the public HTTP API
    pub auth: HttpAuthConfig,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1".to_string(),
            port: 3000,
            enable_cors: false,
            request_timeout_secs: 30,
            max_body_size: 1024 * 1024, // 1MB
            auth: HttpAuthConfig::default(),
        }
    }
}

/// HTTP authentication settings for protecting the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpAuthConfig {
    /// When true, all endpoints require an authentication token
    pub enabled: bool,
    /// Header that carries the credential, defaults to `Authorization`
    pub header_name: String,
    /// Optional prefix stripped from the header value (e.g. `Bearer `)
    pub bearer_prefix: Option<String>,
    /// Static tokens accepted for access
    pub tokens: Vec<String>,
}

impl Default for HttpAuthConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            header_name: "Authorization".to_string(),
            bearer_prefix: Some("Bearer ".to_string()),
            tokens: Vec::new(),
        }
    }
}

/// Telemetry configuration shared across Lens components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    /// Enable OpenTelemetry trace export
    pub enabled: bool,
    /// Optional OTLP endpoint (http(s)://host:port). Uses default agent if unset.
    pub otlp_endpoint: Option<String>,
    /// Service name used when reporting traces
    pub service_name: String,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            otlp_endpoint: None,
            service_name: "lens".to_string(),
        }
    }
}

/// Application-wide configuration toggles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Log level (trace, debug, info, warn, error)
    pub log_level: String,
    /// Enable file watching for auto-indexing
    pub enable_file_watching: bool,
    /// Optional override for Tokio worker threads
    pub worker_threads: Option<usize>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            log_level: "info".to_string(),
            enable_file_watching: true,
            worker_threads: None,
        }
    }
}

impl LensConfig {
    /// Load configuration from well-known locations and environment variables.
    pub fn load() -> Result<Self, ConfigError> {
        let mut builder =
            Config::builder().add_source(config::Config::try_from(&LensConfig::default())?);

        let mut config_paths = vec![
            PathBuf::from("lens.toml"),
            PathBuf::from("config/lens.toml"),
            PathBuf::from("/etc/lens/lens.toml"),
        ];

        if let Some(dir) = dirs::config_dir() {
            config_paths.push(dir.join("lens/lens.toml"));
        }

        for path in config_paths {
            if path.exists() {
                builder = builder.add_source(File::from(path).format(FileFormat::Toml));
                break;
            }
        }

        builder = builder.add_source(
            Environment::with_prefix("LENS")
                .prefix_separator("_")
                .separator("__"),
        );

        let config = builder.build()?;
        config.try_deserialize()
    }

    /// Load configuration from an explicit file path.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
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

    /// Persist configuration to a TOML file.
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let toml_string = toml::to_string_pretty(self)?;
        std::fs::write(path, toml_string)?;
        Ok(())
    }

    /// Write a sample configuration file containing default values.
    pub fn create_sample_config<P: AsRef<Path>>(path: P) -> Result<()> {
        let config = LensConfig::default();
        config.save_to_file(path)?;
        Ok(())
    }

    /// Helper to produce a search engine configuration, optionally overriding the index path.
    pub fn search_engine_config(&self, index_path: Option<PathBuf>) -> SearchConfig {
        let mut config = self.search.clone();
        if let Some(path) = index_path {
            config.index_path = path;
        }
        config
    }

    /// Helper to produce an LSP server configuration clone.
    pub fn lsp_server_config(&self) -> LspServerConfig {
        self.lsp.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use tempfile::NamedTempFile;

    #[test]
    fn default_config_has_reasonable_values() {
        let config = LensConfig::default();
        assert_eq!(config.search.max_results, 50);
        assert_eq!(config.http.port, 3000);
        assert_eq!(config.app.log_level, "info");
        assert!(config.lsp.max_search_results > 0);
        assert!(!config.http.auth.enabled);
        assert_eq!(config.http.auth.header_name, "Authorization");
        assert!(!config.telemetry.enabled);
    }

    #[test]
    fn config_roundtrip() -> Result<()> {
        let temp = NamedTempFile::new()?;
        let original = LensConfig::default();
        original.save_to_file(temp.path())?;

        let loaded = LensConfig::load_from_file(temp.path())?;
        assert_eq!(loaded.search.max_results, original.search.max_results);
        assert_eq!(
            loaded.lsp.max_search_results,
            original.lsp.max_search_results
        );
        Ok(())
    }
}
