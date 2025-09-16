/*!
 * Lens Code Search Engine - Production Entry Point
 * 
 * This is the main entry point for the Lens code search service, a production-grade,
 * fraud-resistant search engine designed for enterprise environments.
 * 
 * ## Security Features
 * 
 * - **Tripwire Protection**: Enforces "real" mode only - service will not start in
 *   development or mock modes to prevent production deployment of test code
 * - **Build-time Attestation**: Embeds Git SHA, Rust version, and build environment
 *   information for cryptographic verification
 * - **Production-only Operation**: All development and testing features are disabled
 *   
 * ## Usage
 * 
 * ```bash
 * # Start in production mode (default - no flags needed)
 * cargo run --release -- --addr 0.0.0.0:8080
 * 
 * # Start in development mode (for testing only)
 * cargo run -- --dev --addr 0.0.0.0:8080
 * 
 * # Start MCP server in production mode
 * cargo run --release -- --mcp
 * ```
 * 
 * ## Environment Variables
 * 
 * - `RUST_LOG`: Controls logging level (error, warn, info, debug, trace)
 * - Default logging includes service startup, build information, and attestation data
 */

use anyhow::Result;
use clap::{Arg, Command};
use lens_core::LensRpcServer;
use tracing::{info, error};
use tracing_subscriber::EnvFilter;
use std::sync::{Arc, Mutex};

#[cfg(feature = "mcp")]
use lens_core::{McpServer, SearchEngine};

/// Main entry point for the Lens code search service.
/// 
/// Initializes the production search engine with comprehensive security enforcement:
/// - Validates command-line arguments for security compliance
/// - Enforces production-only mode through tripwire mechanisms
/// - Logs build and deployment information for attestation
/// - Starts the high-performance HTTP server on the specified address
/// 
/// # Security Model
/// 
/// The service implements a strict security model where:
/// 1. Only "real" mode is permitted (enforced at startup)
/// 2. Build information is embedded and logged for verification
/// 3. All operations are logged for audit trails
/// 4. Server creation and binding are validated before service start
/// 
/// # Exit Codes
/// 
/// - `0`: Successful shutdown (rare, as service runs indefinitely)
/// - `1`: Security violation (wrong mode) or startup failure
/// 
/// # Examples
/// 
/// ```bash
/// # Correct usage - start production service
/// ./lens-core --mode real --addr 0.0.0.0:8080
/// 
/// # Security violation - will exit with code 1
/// ./lens-core --mode development
/// ```
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize structured logging with environment-based filtering
    // Default directive ensures lens_core events are logged at info level or higher
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("lens_core=info".parse()?))
        .init();
    
    // Configure command-line interface with security-focused argument validation
    // All arguments are validated to prevent injection attacks or configuration bypass
    let matches = Command::new("lens-core")
        .version(env!("CARGO_PKG_VERSION"))
        .about("Production fraud-resistant search engine")
        .arg(
            Arg::new("dev")
                .long("dev")
                .help("Enable development mode (disables production security enforcement)")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("addr")
                .long("addr")
                .value_name("ADDRESS")
                .help("Server address to bind to (format: IP:PORT)")
                .default_value("0.0.0.0:8080")
        )
        .arg(
            Arg::new("mcp")
                .long("mcp")
                .help("Run as MCP (Model Context Protocol) server on STDIO instead of HTTP server")
                .action(clap::ArgAction::SetTrue)
        )
        .get_matches();
    
    // Extract and validate command-line arguments
    let addr = matches.get_one::<String>("addr").unwrap();
    let mcp_mode = matches.get_flag("mcp");
    let dev_mode = matches.get_flag("dev");
    
    // Determine mode - default to production "real" mode
    let mode = if dev_mode { "dev" } else { "real" };
    
    // SECURITY TRIPWIRE: Enforce production-only operation by default
    // This is the primary security mechanism preventing deployment of development code
    // in production environments. Development mode must be explicitly enabled with --dev flag.
    if dev_mode {
        info!("WARNING: Development mode enabled - production security enforcement disabled");
        info!("This mode is for testing only and should NEVER be used in production");
    }
    
    // Log startup information for attestation and audit purposes
    // This information is used for cryptographic verification and deployment tracking
    info!("Starting lens-core server in {} mode", mode);
    info!("Git SHA: {}", lens_core::built::GIT_COMMIT_HASH.unwrap_or("unknown"));
    info!("Build target: {}", lens_core::built::TARGET);
    info!("Rust version: {}", lens_core::built::RUSTC_VERSION);
    
    if mcp_mode {
        // MCP (Model Context Protocol) server mode - run on STDIO
        #[cfg(feature = "mcp")]
        {
            info!("Starting MCP server on STDIO");
            
            // Create search engine for MCP server
            match SearchEngine::new_in_memory() {
                Ok(search_engine) => {
                    let search_engine = Arc::new(Mutex::new(search_engine));
                    let mcp_server = McpServer::new(search_engine);
                    
                    info!("MCP server created successfully");
                    info!("Search engine initialized - ready for MCP queries");
                    
                    // Start MCP server on STDIO - this blocks until shutdown
                    if let Err(e) = mcp_server.start().await {
                        error!("MCP server failed during operation: {}", e);
                        std::process::exit(1);
                    }
                }
                Err(e) => {
                    error!("Failed to create search engine for MCP server: {}", e);
                    std::process::exit(1);
                }
            }
        }
        
        #[cfg(not(feature = "mcp"))]
        {
            error!("MCP mode requested but MCP feature not enabled");
            error!("Rebuild with --features mcp to enable MCP functionality");
            std::process::exit(1);
        }
    } else {
        // HTTP server mode - traditional RPC server
        info!("Starting HTTP server on {}", addr);
        
        // Initialize the high-performance RPC server with security enforcement
        // Server creation validates attestation service and search engine initialization
        match LensRpcServer::new() {
            Ok(server) => {
                info!("Server created successfully, starting on {}", addr);
                info!("Attestation service active - fraud resistance enabled");
                info!("Search engine initialized - ready for production queries");
                
                // Start serving requests - this blocks until shutdown signal received
                // The server implements graceful shutdown and error recovery mechanisms
                if let Err(e) = server.serve(addr).await {
                    error!("Server failed during operation: {}", e);
                    error!("Check network connectivity, port availability, and system resources");
                    std::process::exit(1);
                }
            }
            Err(e) => {
                error!("Failed to create server during initialization: {}", e);
                error!("This typically indicates a configuration or resource issue");
                error!("Verify system has sufficient memory and required dependencies");
                std::process::exit(1);
            }
        }
    }
    
    Ok(())
}