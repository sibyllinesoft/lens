//! HTTP server command implementation

use crate::http_server;
use anyhow::Result;
use lens_search_engine::SearchConfig as EngineSearchConfig;

/// Start the HTTP API server
pub async fn start_http_server(
    search_config: EngineSearchConfig,
    bind: String,
    port: u16,
    enable_cors: bool,
) -> Result<()> {
    let index_path = search_config.index_path.clone();
    let config = http_server::ServerConfig {
        bind,
        port,
        enable_cors,
        index_path,
        search_config,
    };

    http_server::start_server(config).await
}
