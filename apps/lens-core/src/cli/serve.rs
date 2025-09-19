//! HTTP server command implementation

use crate::http_server;
use anyhow::Result;
use std::path::PathBuf;

/// Start the HTTP API server
pub async fn start_http_server(
    index_path: PathBuf,
    bind: String,
    port: u16,
    enable_cors: bool,
) -> Result<()> {
    let config = http_server::ServerConfig {
        bind,
        port,
        enable_cors,
        index_path,
    };

    http_server::start_server(config).await
}
