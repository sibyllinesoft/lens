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
pub mod calibration;
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

    // Initialize metrics (TODO: implement init_prometheus_metrics)
    // metrics::init_prometheus_metrics();

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
        }
    }
}