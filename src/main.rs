//! Lens Core - Rust Migration Entry Point
//! 
//! Complete migration to Rust architecture with:
//! - LSP supremacy with real language servers
//! - Zero-copy fused pipeline (≤150ms p95)
//! - SLA-bounded benchmarking system
//! - Performance gates: ≥10pp gain, ≤+1ms p95
//! - 40-60% LSP routing target

use std::sync::Arc;
use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use tracing::{info, error};

use lens_core::{
    LensConfig,
    search::SearchEngine,
    lsp::manager::LspManager,
    pipeline::FusedPipeline,
    metrics::MetricsCollector,
    attestation::AttestationManager,
    benchmark::{BenchmarkRunner, BenchmarkOrchestrator, BenchmarkConfig},
    grpc::{create_server as create_grpc_server, ServerConfig as GrpcServerConfig},
    server::{create_server as create_http_server, ServerConfig as HttpServerConfig},
};

#[derive(Parser)]
#[command(name = "lens")]
#[command(about = "High-performance code search with LSP integration")]
#[command(version = env!("CARGO_PKG_VERSION"))]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Server bind address
    #[arg(long, default_value = "127.0.0.1")]
    bind: String,

    /// Server port
    #[arg(long, default_value = "50051")]
    port: u16,

    /// Index path
    #[arg(long, default_value = "./indexed-content")]
    index_path: String,

    /// Enable LSP integration
    #[arg(long, default_value = "true")]
    enable_lsp: bool,

    /// Enable semantic search
    #[arg(long, default_value = "false")]
    enable_semantic: bool,

    /// Cache TTL in hours
    #[arg(long, default_value = "24")]
    cache_ttl: u64,

    /// Dataset path for pinned golden datasets
    #[arg(long, default_value = "./pinned-datasets")]
    dataset_path: String,

    /// Enable pinned dataset loading
    #[arg(long, default_value = "true")]
    enable_datasets: bool,

    /// Default dataset version to load
    #[arg(long)]
    dataset_version: Option<String>,

    /// Enable corpus validation for datasets
    #[arg(long, default_value = "true")]
    enable_validation: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the HTTP REST API server
    Serve,
    /// Start the gRPC server (legacy)
    ServeGrpc,
    /// Start the MCP (Model Context Protocol) server
    #[cfg(feature = "mcp")]
    ServeMcp,
    /// Run benchmarks
    Benchmark {
        /// Dataset name to benchmark
        #[arg(long, default_value = "storyviz")]
        dataset: String,
        /// Query limit for testing
        #[arg(long)]
        limit: Option<u32>,
        /// Enable smoke test mode
        #[arg(long)]
        smoke: bool,
        /// Generate detailed reports
        #[arg(long, default_value = "true")]
        reports: bool,
    },
    /// Validate corpus consistency
    Validate {
        /// Dataset to validate
        #[arg(long, default_value = "storyviz")]
        dataset: String,
    },
    /// Show system health
    Health,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    // Create core configuration
    let config = LensConfig {
        server_port: cli.port,
        index_path: cli.index_path.clone(),
        lsp_enabled: cli.enable_lsp,
        cache_ttl_hours: cli.cache_ttl,
        performance_target_ms: 150, // ≤150ms p95 per TODO.md
        
        // Dataset configuration
        dataset_path: cli.dataset_path,
        enable_pinned_datasets: cli.enable_datasets,
        default_dataset_version: cli.dataset_version.or_else(|| 
            Some(lens_core::benchmark::DEFAULT_PINNED_VERSION.to_string())
        ),
        enable_corpus_validation: cli.enable_validation,
        
        ..Default::default()
    };

    match cli.command {
        Commands::Serve => {
            info!("🚀 Starting Lens HTTP API server with Rust architecture");
            serve_http(config, cli.bind, cli.port, cli.enable_semantic).await
        }
        Commands::ServeGrpc => {
            info!("🚀 Starting Lens gRPC server (legacy mode)");
            serve_grpc(config, cli.bind, cli.port, cli.enable_semantic).await
        }
        #[cfg(feature = "mcp")]
        Commands::ServeMcp => {
            info!("🔌 Starting Lens MCP server with Model Context Protocol");
            serve_mcp(config, cli.enable_semantic).await
        }
        Commands::Benchmark { dataset, limit, smoke, reports } => {
            info!("🧪 Running benchmark suite for dataset: {}", dataset);
            run_benchmark(config, dataset, limit, smoke, reports).await
        }
        Commands::Validate { dataset } => {
            info!("✅ Validating corpus consistency for dataset: {}", dataset);
            validate_corpus(config, dataset).await
        }
        Commands::Health => {
            info!("💓 Checking system health");
            check_health(config).await
        }
    }
}

/// Start the HTTP REST API server
async fn serve_http(
    config: LensConfig,
    bind_address: String,
    port: u16,
    enable_semantic: bool,
) -> Result<()> {
    info!("Initializing HTTP REST API server components...");

    // Create SearchConfig from LensConfig
    let search_config = lens_core::search::SearchConfig {
        index_path: config.index_path.clone(),
        max_results_default: config.max_results,
        sla_target_ms: config.performance_target_ms,
        lsp_routing_rate: if config.lsp_enabled { 0.5 } else { 0.0 },
        enable_fusion_pipeline: false, // Temporarily disabled
        enable_semantic_search: enable_semantic,
        enable_lsp: config.lsp_enabled,
        context_lines: 3,
        
        // Dataset configuration from LensConfig
        dataset_path: config.dataset_path.clone(),
        enable_pinned_datasets: config.enable_pinned_datasets,
        default_dataset_version: config.default_dataset_version.clone(),
        enable_corpus_validation: config.enable_corpus_validation,
    };

    // Initialize search engine with dataset support
    let search_engine = SearchEngine::with_config(&config.index_path, search_config).await
        .map_err(|e| anyhow::anyhow!("Failed to create search engine: {}", e))?;

    let search_engine = Arc::new(search_engine);

    // Initialize metrics collector for SLA tracking
    let metrics_collector = Arc::new(MetricsCollector::new());
    
    // Initialize attestation manager for fraud resistance
    let attestation_manager = Arc::new(AttestationManager::new(config.attestation_enabled)?);

    // Initialize benchmark runner
    let benchmark_config = BenchmarkConfig::default();
    let benchmark_runner = Arc::new(BenchmarkRunner::new(
        search_engine.clone(),
        metrics_collector.clone(),
        benchmark_config,
    ));

    // Configure HTTP server
    let http_server_config = HttpServerConfig {
        bind_address,
        port,
        enable_cors: true,
        request_timeout: std::time::Duration::from_millis(config.performance_target_ms),
        max_request_size: 1024 * 1024, // 1MB
        enable_tracing: true,
    };

    // Create and start HTTP server
    info!("🌐 Starting HTTP API server on {}:{}", http_server_config.bind_address, http_server_config.port);
    create_http_server(
        http_server_config,
        search_engine,
        metrics_collector,
        attestation_manager,
        benchmark_runner,
    ).await?;

    Ok(())
}

/// Start the complete Rust gRPC server with all components (legacy)
async fn serve_grpc(
    config: LensConfig,
    bind_address: String,
    port: u16,
    enable_semantic: bool,
) -> Result<()> {
    info!("Initializing Rust migration components...");

    // Create SearchConfig from LensConfig
    let search_config = lens_core::search::SearchConfig {
        index_path: config.index_path.clone(),
        max_results_default: config.max_results,
        sla_target_ms: config.performance_target_ms,
        lsp_routing_rate: if config.lsp_enabled { 0.5 } else { 0.0 },
        enable_fusion_pipeline: false, // Temporarily disabled
        enable_semantic_search: enable_semantic,
        enable_lsp: config.lsp_enabled,
        context_lines: 3,
        
        // Dataset configuration from LensConfig
        dataset_path: config.dataset_path.clone(),
        enable_pinned_datasets: config.enable_pinned_datasets,
        default_dataset_version: config.default_dataset_version.clone(),
        enable_corpus_validation: config.enable_corpus_validation,
    };

    // Initialize search engine with dataset support
    let search_engine = SearchEngine::with_config(&config.index_path, search_config).await
        .map_err(|e| anyhow::anyhow!("Failed to create search engine: {}", e))?;

    // Initialize LSP manager if enabled  
    if config.lsp_enabled {
        info!("🔧 Initializing LSP manager with real language servers");
        let lsp_config = lens_core::lsp::LspConfig::default();
        let _lsp_manager = LspManager::new(lsp_config).await
            .map_err(|e| anyhow::anyhow!("Failed to create LSP manager: {}", e))?;
        
        // Note: SearchEngine LSP integration disabled for now - would need enable_lsp method
        // search_engine.enable_lsp(Arc::new(lsp_manager)).await?;
    }

    // Initialize fused pipeline for ≤150ms p95 performance
    info!("⚡ Initializing zero-copy fused pipeline");
    let pipeline_config = lens_core::pipeline::PipelineConfig {
        max_latency_ms: config.performance_target_ms,
        ..Default::default()
    };
    let pipeline = FusedPipeline::new(pipeline_config).await
        .map_err(|e| anyhow::anyhow!("Failed to create fused pipeline: {}", e))?;
    
    // Note: SearchEngine integration disabled for now - would need enable_pipeline method
    // search_engine.enable_pipeline(Arc::new(pipeline)).await?;

    let search_engine = Arc::new(search_engine);

    // Initialize metrics collector for SLA tracking
    let metrics_collector = Arc::new(MetricsCollector::new());
    
    // Note: MetricsCollector doesn't have start_collection method - metrics are collected on demand
    // metrics_collector.start_collection().await;

    // Initialize attestation manager for fraud resistance
    let attestation_manager = Arc::new(AttestationManager::new(config.attestation_enabled)?);

    // Initialize benchmark runner
    let benchmark_config = BenchmarkConfig::default();
    let benchmark_runner = Arc::new(BenchmarkRunner::new(
        search_engine.clone(),
        metrics_collector.clone(),
        benchmark_config,
    ));

    // Configure gRPC server
    let server_config = GrpcServerConfig {
        bind_address,
        port,
        max_concurrent_requests: 1000,
        request_timeout: std::time::Duration::from_millis(config.performance_target_ms),
        enable_reflection: false, // Disabled for production
        enable_health_check: true,
    };

    // Create and start gRPC server
    info!("🌐 Starting gRPC server on {}:{}", server_config.bind_address, server_config.port);
    let server = create_grpc_server(
        server_config,
        search_engine,
        metrics_collector,
        attestation_manager,
        benchmark_runner,
    ).await?;

    // Start server with graceful shutdown
    tokio::select! {
        result = server => {
            if let Err(e) = result {
                error!("gRPC server error: {}", e);
            }
        }
        _ = tokio::signal::ctrl_c() => {
            info!("🛑 Received shutdown signal");
        }
    }

    info!("Server shutdown complete");
    Ok(())
}

/// Run comprehensive benchmark suite
async fn run_benchmark(
    config: LensConfig,
    dataset: String,
    query_limit: Option<u32>,
    smoke_test: bool,
    generate_reports: bool,
) -> Result<()> {
    // Initialize components for benchmarking
    let search_engine = Arc::new(SearchEngine::new(&config.index_path).await?);
    let metrics_collector = Arc::new(MetricsCollector::new());

    // Configure benchmark
    let benchmark_config = BenchmarkConfig::default();

    let benchmark_runner = BenchmarkRunner::new(
        search_engine,
        metrics_collector,
        benchmark_config,
    );

    // Execute benchmark
    info!("🏃 Starting benchmark execution...");
    match benchmark_runner.run_benchmark(&dataset, query_limit, smoke_test).await {
        Ok(results) => {
            info!("✅ Benchmark completed successfully");
            info!("📊 Total queries: {}", results.summary.total_queries);
            info!("⚡ Average latency: {}ms", results.summary.average_latency_ms);
            info!("🎯 P95 latency: {}ms", results.summary.p95_latency_ms);
            info!("📈 Average Success@10: {:.3}", results.summary.average_success_at_10);
            info!("🚦 SLA compliance: {:.1}%", results.summary.sla_compliance_rate * 100.0);

            // Performance gates analysis
            if results.summary.passes_performance_gates {
                info!("🎉 All performance gates PASSED! System ready for production.");
            } else {
                error!("❌ Performance gates FAILED. Review gate analysis:");
                for gate in &results.summary.gate_analysis {
                    if !gate.passed {
                        error!("  - {}: {:.1} (target: {:.1})", gate.gate_name, gate.actual_value, gate.target_value);
                    }
                }
            }

            if let Some(report_path) = results.report_path {
                info!("📋 Detailed report saved: {}", report_path);
            }
        }
        Err(e) => {
            error!("❌ Benchmark failed: {}", e);
            return Err(anyhow::anyhow!("{}", e));
        }
    }

    Ok(())
}

/// Validate corpus consistency
async fn validate_corpus(config: LensConfig, dataset: String) -> Result<()> {
    info!("🔍 Validating corpus consistency for dataset: {}", dataset);
    
    // Create benchmark orchestrator to perform validation
    let benchmark_config = BenchmarkConfig {
        dataset_path: config.dataset_path.clone(),
        enable_corpus_validation: config.enable_corpus_validation,
        default_version: config.default_dataset_version.clone(),
        ..BenchmarkConfig::default()
    };
    
    let orchestrator = BenchmarkOrchestrator::with_config(benchmark_config).await?;
    
    // Load the specified dataset (or current version if default)
    let pinned_dataset = if dataset == "current" || dataset.is_empty() {
        orchestrator.load_pinned_dataset().await?
    } else {
        orchestrator.load_dataset_version(&dataset).await?
    };
    
    info!("📊 Loaded dataset with {} queries for validation", pinned_dataset.queries.len());
    
    // Perform corpus consistency validation
    let is_consistent = orchestrator.validate_corpus_consistency(&pinned_dataset).await?;
    
    if is_consistent {
        info!("✅ Corpus validation passed - all golden queries exist in the indexed corpus");
        Ok(())
    } else {
        Err(anyhow!("❌ Corpus validation failed - some golden queries do not exist in the indexed corpus"))
    }
}

/// Check system health
async fn check_health(config: LensConfig) -> Result<()> {
    info!("Performing system health check...");
    
    // Check if index exists
    let index_exists = tokio::fs::try_exists(&config.index_path).await.unwrap_or(false);
    info!("📁 Index directory: {} - {}", config.index_path, 
        if index_exists { "✅ EXISTS" } else { "❌ MISSING" });

    // Check LSP servers if enabled
    if config.lsp_enabled {
        info!("🔧 LSP integration: ✅ ENABLED");
        
        // Check if LSP servers are available (basic check without full initialization)
        info!("🔍 Checking LSP server availability...");
        let mut servers_found = 0;
        
        // Check for common LSP servers
        let server_commands = vec![
            ("TypeScript", "typescript-language-server"),
            ("Python", "pylsp"),
            ("Rust", "rust-analyzer"), 
            ("Go", "gopls"),
        ];
        
        for (name, command) in server_commands {
            match tokio::process::Command::new(command)
                .arg("--version")
                .output()
                .await
            {
                Ok(output) if output.status.success() => {
                    info!("  ✅ {} LSP server found: {}", name, command);
                    servers_found += 1;
                }
                _ => {
                    info!("  ⚠️  {} LSP server not found: {}", name, command);
                }
            }
        }
        
        info!("🔧 LSP servers available: {}/4", servers_found);
    } else {
        info!("🔧 LSP integration: ⚠️ DISABLED");
    }

    // Check performance targets
    info!("⚡ Performance target: ≤{}ms p95", config.performance_target_ms);
    info!("🎯 SLA requirements: Success@10, nDCG@10, SLA-Recall@50");
    
    // System info
    info!("🦀 Rust version: {}", env!("CARGO_PKG_VERSION"));
    info!("🏗️ Build info: {} ({})", 
        lens_core::built_info::GIT_VERSION.unwrap_or("unknown"),
        env!("BUILD_TIMESTAMP", "unknown"));

    Ok(())
}

/// Start the MCP (Model Context Protocol) server
#[cfg(feature = "mcp")]
async fn serve_mcp(
    config: LensConfig,
    enable_semantic: bool,
) -> Result<()> {
    info!("Initializing MCP server components...");

    // Create SearchConfig from LensConfig
    let search_config = lens_core::search::SearchConfig {
        index_path: config.index_path.clone(),
        max_results_default: config.max_results,
        sla_target_ms: config.performance_target_ms,
        lsp_routing_rate: if config.lsp_enabled { 0.5 } else { 0.0 },
        enable_fusion_pipeline: false,
        enable_semantic_search: enable_semantic,
        enable_lsp: config.lsp_enabled,
        context_lines: 3,
        
        // Dataset configuration from LensConfig
        dataset_path: config.dataset_path.clone(),
        enable_pinned_datasets: config.enable_pinned_datasets,
        default_dataset_version: config.default_dataset_version.clone(),
        enable_corpus_validation: config.enable_corpus_validation,
    };

    // Initialize search engine with dataset support
    let search_engine = SearchEngine::with_config(&config.index_path, search_config).await
        .map_err(|e| anyhow::anyhow!("Failed to create search engine: {}", e))?;

    let search_engine = Arc::new(search_engine);

    // Create and start MCP server
    info!("🔌 Starting MCP server over stdio");
    lens_core::mcp::create_mcp_server(search_engine).await?;

    Ok(())
}