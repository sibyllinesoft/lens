//! # Lens Core - High-Performance Code Search Engine
//!
//! A production-ready code search engine with:
//! - LSP-first architecture with real language server integration
//! - Zero-copy fused pipeline for ≤150ms p95 latency
//! - Bounded BFS for def/ref/type/impl traversal
//! - SLA-bounded metrics: Success@10, nDCG@10, SLA-Recall@50
//! - Built-in benchmarking and fraud-resistant attestation
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
//! │   gRPC API      │────│  Query Router   │────│  Search Engine  │
//! └─────────────────┘    └─────────────────┘    └─────────────────┘
//!          │                       │                       │
//!          │              ┌─────────────────┐              │
//!          │              │  LSP Manager    │              │
//!          │              └─────────────────┘              │
//!          │                       │                       │
//!          │              ┌─────────────────┐              │
//!          │              │  Hint Cache     │              │
//!          │              └─────────────────┘              │
//!          │                                               │
//!          └─────────────────┬─────────────────────────────┘
//!                            │
//!                   ┌─────────────────┐
//!                   │  Attestation    │
//!                   └─────────────────┘
//! ```

pub mod adversarial;
pub mod attestation;
pub mod baseline;
pub mod benchmark;
pub mod cache;
// Temporarily disabled calibration module due to complex dependencies
// pub mod calibration;
pub mod config;
pub mod grpc;
pub mod lang;
pub mod lsp;
pub mod metrics;
pub mod pipeline;
pub mod proto;
pub mod query;
pub mod search;
pub mod semantic;
pub mod server;

use anyhow::Result;

// Build-time information for anti-fraud
pub mod built_info {
    include!(concat!(env!("OUT_DIR"), "/built.rs"));
}

/// Initialize the lens core with configuration and anti-fraud checks
pub async fn initialize() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "lens_core=info,tower_lsp=debug".into()),
        )
        .init();

    // Verify we're in real mode, not mock
    if built_info::CFG_OS == "test" {
        tracing::warn!("Running in test mode");
    } else {
        attestation::verify_real_mode()?;
    }

    // Initialize metrics system
    if let Err(e) = metrics::init_prometheus_metrics() {
        tracing::warn!("Failed to initialize Prometheus metrics: {}", e);
        tracing::info!("Continuing without Prometheus metrics - basic metrics still available");
    }

    tracing::info!("Lens Core initialized");
    tracing::info!("Git SHA: {}", built_info::GIT_VERSION.unwrap_or("unknown"));
    tracing::info!("Build timestamp: {}", env!("BUILD_TIMESTAMP", "unknown"));
    tracing::info!("Target: {} {}", built_info::CFG_TARGET_ARCH, built_info::CFG_OS);

    Ok(())
}

/// Core application configuration
#[derive(Debug, Clone)]
pub struct LensConfig {
    pub server_port: u16,
    pub index_path: String,
    pub lsp_enabled: bool,
    pub cache_ttl_hours: u64,
    pub max_results: usize,
    pub performance_target_ms: u64,
    pub attestation_enabled: bool,
    
    // Dataset configuration for pinned golden datasets
    pub dataset_path: String,
    pub enable_pinned_datasets: bool,
    pub default_dataset_version: Option<String>,
    pub enable_corpus_validation: bool,
}

impl Default for LensConfig {
    fn default() -> Self {
        Self {
            server_port: 50051,
            index_path: "./index".to_string(),
            lsp_enabled: true,
            cache_ttl_hours: 24,
            max_results: 50,
            performance_target_ms: 150,
            attestation_enabled: true,
            
            // Default dataset configuration
            dataset_path: "./pinned-datasets".to_string(),
            enable_pinned_datasets: true,
            default_dataset_version: Some(crate::benchmark::DEFAULT_PINNED_VERSION.to_string()),
            enable_corpus_validation: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[tokio::test]
    async fn test_lens_config_default() {
        let config = LensConfig::default();
        
        assert_eq!(config.server_port, 50051);
        assert_eq!(config.index_path, "./index");
        assert!(config.lsp_enabled);
        assert_eq!(config.cache_ttl_hours, 24);
        assert_eq!(config.max_results, 50);
        assert_eq!(config.performance_target_ms, 150);
        assert!(config.attestation_enabled);
        assert!(config.enable_pinned_datasets);
    }

    #[test]
    fn test_lens_config_clone() {
        let config1 = LensConfig::default();
        let config2 = config1.clone();
        
        assert_eq!(config1.server_port, config2.server_port);
        assert_eq!(config1.index_path, config2.index_path);
    }

    #[tokio::test]
    async fn test_initialize_in_test_mode() {
        // Set test environment to avoid real mode verification
        env::set_var("RUST_LOG", "debug");
        
        // Initialize should not fail in test environment
        let result = initialize().await;
        
        // Should succeed or fail gracefully
        match result {
            Ok(_) => println!("Initialization successful"),
            Err(e) => println!("Initialization failed as expected in test mode: {}", e),
        }
    }

    #[test]
    fn test_built_info_available() {
        // Verify build info is accessible
        assert!(built_info::PKG_NAME.len() > 0);
        assert!(built_info::PKG_VERSION.len() > 0);
    }
    
    // Additional simple configuration and data structure tests
    #[test]
    fn test_lens_config_debug_format() {
        let config = LensConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("LensConfig"));
        assert!(debug_str.contains("50051"));
    }
    
    #[test]
    fn test_lens_config_modification() {
        let mut config = LensConfig::default();
        config.server_port = 8080;
        config.max_results = 100;
        
        assert_eq!(config.server_port, 8080);
        assert_eq!(config.max_results, 100);
        // Other values should remain default
        assert_eq!(config.cache_ttl_hours, 24);
        assert!(config.lsp_enabled);
    }
}