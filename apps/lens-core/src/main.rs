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
use std::path::PathBuf;
use tracing::info;

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
    #[arg(long, global = true, default_value = "./index")]
    index_path: PathBuf,

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

    // Load configuration
    let config = if let Some(config_path) = &cli.config {
        LensConfig::load_from_file(config_path)
            .map_err(|e| anyhow::anyhow!("Failed to load config from {:?}: {}", config_path, e))?
    } else {
        LensConfig::load().unwrap_or_else(|_| {
            tracing::warn!("Could not load configuration file, using defaults");
            LensConfig::default()
        })
    };

    // Initialize logging with config
    let log_level = if cli.debug {
        "debug"
    } else if cli.verbose {
        "info"
    } else {
        &config.app.log_level
    };

    tracing_subscriber::fmt()
        .with_env_filter(format!(
            "lens={},lens_search_engine={}",
            log_level, log_level
        ))
        .init();

    info!("Starting Lens v{}", env!("CARGO_PKG_VERSION"));
    info!(
        "Using configuration: search.index_path={:?}, lsp.max_search_results={}, http.port={}",
        config.search.index_path, config.lsp.max_search_results, config.http.port
    );

    // Use config values, but allow CLI overrides
    let effective_index_path = cli.index_path.clone();

    // Execute the selected command
    match cli.command {
        Commands::Lsp { tcp, port } => {
            if tcp {
                cli::start_lsp_tcp_server(effective_index_path, port).await
            } else {
                cli::start_lsp_stdio_server(effective_index_path).await
            }
        }
        Commands::Serve { bind, port, cors } => {
            cli::start_http_server(effective_index_path, bind, port, cors).await
        }
        Commands::Index {
            directory,
            force,
            progress,
        } => cli::index_directory(effective_index_path, directory, force, progress).await,
        Commands::Search {
            query,
            limit,
            offset,
            fuzzy,
            symbols,
            language,
            file_pattern,
        } => {
            cli::search_index(
                effective_index_path,
                query,
                limit,
                offset,
                fuzzy,
                symbols,
                language,
                file_pattern,
            )
            .await
        }
        Commands::Stats => cli::show_stats(effective_index_path).await,
        Commands::Optimize => cli::optimize_index(effective_index_path).await,
        Commands::Clear { yes } => cli::clear_index(effective_index_path, yes).await,
        Commands::Config { action } => handle_config_command(action, &config).await,
    }
}

async fn handle_config_command(action: ConfigAction, config: &LensConfig) -> Result<()> {
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
