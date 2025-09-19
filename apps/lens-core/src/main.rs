//! Lens - High-Performance Code Search Engine
//!
//! This is the main application binary that brings together all the real
//! components to create a production-ready code search system.
//!
//! Features:
//! - Real Tantivy-based search engine (no simulation)
//! - Real LSP server implementation
//! - Real HTTP API for web integration
//! - Real indexing with language detection
//! - Real CLI interface

mod cli;
mod config;
mod http_server;

use anyhow::Result;
use clap::{Parser, Subcommand};
use config::LensConfig;
use lens_search_engine::SearchEngine;
use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{
    propagation::TraceContextPropagator, runtime::Tokio, trace as sdktrace, Resource,
};
use std::{path::PathBuf, sync::Arc};
use tracing::info;
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

#[derive(Parser)]
#[command(name = "lens")]
#[command(about = "High-performance code search with LSP integration")]
#[command(version = env!("CARGO_PKG_VERSION"))]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable debug logging
    #[arg(long, short, global = true)]
    debug: bool,

    /// Index directory path
    #[arg(long, global = true)]
    index_path: Option<PathBuf>,

    /// Enable verbose output
    #[arg(long, short, global = true)]
    verbose: bool,

    /// Configuration file path
    #[arg(long, global = true)]
    config: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the LSP server
    Lsp {
        /// Whether to run over stdio (default) or TCP
        #[arg(long)]
        tcp: bool,
        /// TCP port (only used with --tcp)
        #[arg(long, default_value = "9999")]
        port: u16,
    },
    /// Start the HTTP API server
    Serve {
        /// Server bind address
        #[arg(long, default_value = "127.0.0.1")]
        bind: String,
        /// Server port
        #[arg(long, default_value = "3000")]
        port: u16,
        /// Enable CORS
        #[arg(long)]
        cors: bool,
    },
    /// Index files from a directory
    Index {
        /// Directory to index
        directory: PathBuf,
        /// Force re-indexing of existing files
        #[arg(long)]
        force: bool,
        /// Show progress
        #[arg(long)]
        progress: bool,
    },
    /// Search the index
    Search {
        /// Search query (supports inline tokens like `lang:rust path:src/ limit:20 offset:10`)
        query: String,
        /// Maximum number of results
        #[arg(long, default_value = "10")]
        limit: usize,
        /// Number of results to skip
        #[arg(long, default_value = "0")]
        offset: usize,
        /// Use fuzzy search
        #[arg(long)]
        fuzzy: bool,
        /// Search for symbols only
        #[arg(long)]
        symbols: bool,
        /// Filter by language
        #[arg(long)]
        language: Option<String>,
        /// Filter results to matching file paths
        #[arg(long = "file-pattern")]
        file_pattern: Option<String>,
    },
    /// Show index statistics
    Stats,
    /// Optimize the search index
    Optimize,
    /// Clear the search index
    Clear {
        /// Skip confirmation prompt
        #[arg(long)]
        yes: bool,
    },
    /// Configuration management
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Show current configuration
    Show,
    /// Create a sample configuration file
    Init {
        /// Output file path
        #[arg(default_value = "lens.toml")]
        output: PathBuf,
    },
    /// Validate configuration
    Validate {
        /// Configuration file to validate
        file: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let config = if let Some(config_path) = &cli.config {
        LensConfig::load_from_file(config_path)
            .map_err(|e| anyhow::anyhow!("Failed to load config from {:?}: {}", config_path, e))?
    } else {
        LensConfig::load().unwrap_or_else(|_| {
            tracing::warn!("Could not load configuration file, using defaults");
            LensConfig::default()
        })
    };

    let log_level = if cli.debug {
        "debug"
    } else if cli.verbose {
        "info"
    } else {
        &config.app.log_level
    };

    let telemetry_enabled = init_tracing(&config, log_level)?;

    info!("Starting Lens v{}", env!("CARGO_PKG_VERSION"));
    info!(
        "Using configuration: search.index_path={:?}, lsp.max_search_results={}, http.port={}",
        config.search.index_path, config.lsp.max_search_results, config.http.port
    );

    let effective_index_path = cli
        .index_path
        .clone()
        .unwrap_or_else(|| config.search.index_path.clone());

    let command_result = match cli.command {
        Commands::Lsp { tcp, port } => {
            let search_config = config.search_engine_config(Some(effective_index_path.clone()));
            let lsp_config = config.lsp_server_config();
            if tcp {
                cli::start_lsp_tcp_server(search_config, lsp_config, port).await
            } else {
                cli::start_lsp_stdio_server(search_config, lsp_config).await
            }
        }
        Commands::Serve { bind, port, cors } => {
            let search_config = config.search_engine_config(Some(effective_index_path.clone()));
            let mut http_cfg = config.http.clone();
            http_cfg.bind = bind;
            http_cfg.port = port;
            if cors {
                http_cfg.enable_cors = true;
            }
            cli::start_http_server(search_config, http_cfg).await
        }
        Commands::Index {
            directory,
            force,
            progress,
        } => {
            let search_config = config.search_engine_config(Some(effective_index_path.clone()));
            cli::index_directory(search_config, directory, force, progress).await
        }
        Commands::Search {
            query,
            limit,
            offset,
            fuzzy,
            symbols,
            language,
            file_pattern,
        } => {
            let search_config = config.search_engine_config(Some(effective_index_path.clone()));
            let search_engine = Arc::new(SearchEngine::with_config(search_config).await?);
            cli::search_index(
                Arc::clone(&search_engine),
                query,
                cli::SearchCommandOptions {
                    limit,
                    offset,
                    fuzzy,
                    symbols,
                    language,
                    file_pattern,
                },
            )
            .await
        }
        Commands::Stats => {
            let search_config = config.search_engine_config(Some(effective_index_path.clone()));
            let index_path = search_config.index_path.clone();
            let search_engine = Arc::new(SearchEngine::with_config(search_config).await?);
            cli::show_stats(search_engine, &index_path).await
        }
        Commands::Optimize => {
            let search_config = config.search_engine_config(Some(effective_index_path.clone()));
            let index_path = search_config.index_path.clone();
            let search_engine = Arc::new(SearchEngine::with_config(search_config).await?);
            cli::optimize_index(search_engine, &index_path).await
        }
        Commands::Clear { yes } => {
            let search_config = config.search_engine_config(Some(effective_index_path.clone()));
            let index_path = search_config.index_path.clone();
            let engine = if index_path.exists() {
                Some(Arc::new(SearchEngine::with_config(search_config).await?))
            } else {
                None
            };
            cli::clear_index(engine, &index_path, yes).await
        }
        Commands::Config { action } => handle_config_action(action, &config).await,
    };

    if telemetry_enabled {
        global::shutdown_tracer_provider();
    }

    command_result
}

fn init_tracing(config: &LensConfig, log_level: &str) -> Result<bool> {
    let env_filter = std::env::var("RUST_LOG").unwrap_or_else(|_| {
        format!(
            "lens={},lens_search_engine={},lens_lsp_server={}",
            log_level, log_level, log_level
        )
    });
    let env_filter = EnvFilter::try_new(env_filter)?;

    let fmt_layer = tracing_subscriber::fmt::layer().with_target(true);

    if config.telemetry.enabled {
        global::set_text_map_propagator(TraceContextPropagator::new());

        let exporter = {
            let mut builder = opentelemetry_otlp::new_exporter().tonic();
            if let Some(endpoint) = &config.telemetry.otlp_endpoint {
                builder = builder.with_endpoint(endpoint.clone());
            }
            builder
        };

        let tracer = opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(exporter)
            .with_trace_config(
                sdktrace::Config::default().with_resource(Resource::new(vec![KeyValue::new(
                    "service.name",
                    config.telemetry.service_name.clone(),
                )])),
            )
            .install_batch(Tokio)?;

        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .with(OpenTelemetryLayer::new(tracer))
            .init();

        Ok(true)
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .init();

        Ok(false)
    }
}

async fn handle_config_action(action: ConfigAction, config: &LensConfig) -> Result<()> {
    match action {
        ConfigAction::Show => {
            println!("Current Configuration:");
            println!();

            // App settings
            println!("[app]");
            println!("log_level = \"{}\"", config.app.log_level);
            println!("enable_file_watching = {}", config.app.enable_file_watching);
            if let Some(threads) = config.app.worker_threads {
                println!("worker_threads = {}", threads);
            }
            println!();

            // Search settings
            println!("[search]");
            println!("index_path = \"{}\"", config.search.index_path.display());
            println!("max_results = {}", config.search.max_results);
            println!("enable_fuzzy = {}", config.search.enable_fuzzy);
            println!("fuzzy_distance = {}", config.search.fuzzy_distance);
            println!("heap_size_mb = {}", config.search.heap_size_mb);
            println!("commit_interval_ms = {}", config.search.commit_interval_ms);
            println!("enable_cache = {}", config.search.enable_cache);
            println!("cache_size = {}", config.search.cache_size);
            println!(
                "supported_extensions = {:?}",
                config.search.supported_extensions
            );
            println!(
                "ignored_directories = {:?}",
                config.search.ignored_directories
            );
            println!(
                "ignored_file_patterns = {:?}",
                config.search.ignored_file_patterns
            );
            println!();

            // LSP settings
            println!("[lsp]");
            println!("max_search_results = {}", config.lsp.max_search_results);
            println!("enable_fuzzy_search = {}", config.lsp.enable_fuzzy_search);
            println!(
                "enable_semantic_search = {}",
                config.lsp.enable_semantic_search
            );
            println!("search_debounce_ms = {}", config.lsp.search_debounce_ms);
            println!(
                "enable_result_caching = {}",
                config.lsp.enable_result_caching
            );
            println!(
                "workspace_exclude_patterns = {:?}",
                config.lsp.workspace_exclude_patterns
            );
            println!();

            // HTTP settings
            println!("[http]");
            println!("bind = \"{}\"", config.http.bind);
            println!("port = {}", config.http.port);
            println!("enable_cors = {}", config.http.enable_cors);
            println!(
                "request_timeout_secs = {}",
                config.http.request_timeout_secs
            );
            println!("max_body_size = {}", config.http.max_body_size);
            println!("auth.enabled = {}", config.http.auth.enabled);
            println!("auth.header_name = \"{}\"", config.http.auth.header_name);
            if let Some(prefix) = &config.http.auth.bearer_prefix {
                println!("auth.bearer_prefix = \"{}\"", prefix);
            }
            println!("auth.tokens = {}", config.http.auth.tokens.len());
            println!();

            // Telemetry settings
            println!("[telemetry]");
            println!("enabled = {}", config.telemetry.enabled);
            if let Some(endpoint) = &config.telemetry.otlp_endpoint {
                println!("otlp_endpoint = \"{}\"", endpoint);
            }
            println!("service_name = \"{}\"", config.telemetry.service_name);
        }
        ConfigAction::Init { output } => {
            if output.exists() {
                return Err(anyhow::anyhow!(
                    "Configuration file already exists: {:?}",
                    output
                ));
            }

            LensConfig::create_sample_config(&output)?;
            println!("Created sample configuration file: {:?}", output);
            println!("You can now edit this file to customize your settings.");
        }
        ConfigAction::Validate { file } => {
            let config_to_validate = if let Some(config_path) = file {
                match LensConfig::load_from_file(&config_path) {
                    Ok(config) => {
                        println!("✅ Configuration file is valid: {:?}", config_path);
                        config
                    }
                    Err(e) => {
                        println!("❌ Configuration file is invalid: {:?}", config_path);
                        println!("Error: {}", e);
                        return Err(anyhow::anyhow!("Invalid configuration file"));
                    }
                }
            } else {
                match LensConfig::load() {
                    Ok(config) => {
                        println!("✅ Default configuration is valid");
                        config
                    }
                    Err(e) => {
                        println!("❌ Default configuration is invalid");
                        println!("Error: {}", e);
                        return Err(anyhow::anyhow!("Invalid default configuration"));
                    }
                }
            };

            // Validate specific constraints
            if config_to_validate.search.max_results == 0 {
                println!("⚠️  Warning: search.max_results is 0, which may not be useful");
            }
            if config_to_validate.search.heap_size_mb < 32 {
                println!(
                    "⚠️  Warning: search.heap_size_mb is very low (< 32MB), indexing may be slow"
                );
            }
            if config_to_validate.lsp.max_search_results == 0 {
                println!("⚠️  Warning: lsp.max_search_results is 0, LSP may not return results");
            }
            if !config_to_validate
                .search
                .index_path
                .parent()
                .map(|p| p.exists())
                .unwrap_or(true)
            {
                println!(
                    "⚠️  Warning: index_path parent directory does not exist: {:?}",
                    config_to_validate.search.index_path.parent()
                );
            }

            println!("Configuration validation complete.");
        }
    }
    Ok(())
}
