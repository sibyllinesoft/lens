use anyhow::Result;
use clap::{Arg, Command};
use lens_core::LensRpcServer;
use tracing::{info, error};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("lens_core=info".parse()?))
        .init();
    
    let matches = Command::new("lens-core")
        .version(env!("CARGO_PKG_VERSION"))
        .about("Production fraud-resistant search engine")
        .arg(
            Arg::new("mode")
                .long("mode")
                .value_name("MODE")
                .help("Service mode (must be 'real')")
                .default_value("real")
                .required(true)
        )
        .arg(
            Arg::new("addr")
                .long("addr")
                .value_name("ADDRESS")
                .help("Server address to bind to")
                .default_value("0.0.0.0:8080")
        )
        .get_matches();
    
    let mode = matches.get_one::<String>("mode").unwrap();
    let addr = matches.get_one::<String>("addr").unwrap();
    
    // TRIPWIRE: Refuse to start if mode is not 'real'
    if mode != "real" {
        error!("TRIPWIRE VIOLATION: Service mode must be 'real', got: {}", mode);
        std::process::exit(1);
    }
    
    info!("Starting lens-core server in {} mode", mode);
    info!("Git SHA: {}", lens_core::built::GIT_COMMIT_HASH.unwrap_or("unknown"));
    info!("Build target: {}", lens_core::built::TARGET);
    info!("Rust version: {}", lens_core::built::RUSTC_VERSION);
    
    // Create and start server
    match LensRpcServer::new() {
        Ok(server) => {
            info!("Server created successfully, starting on {}", addr);
            if let Err(e) = server.serve(addr).await {
                error!("Server failed: {}", e);
                std::process::exit(1);
            }
        }
        Err(e) => {
            error!("Failed to create server: {}", e);
            std::process::exit(1);
        }
    }
    
    Ok(())
}