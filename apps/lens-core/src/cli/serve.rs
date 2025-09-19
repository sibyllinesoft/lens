//! HTTP server command implementation

use crate::http_server;
use anyhow::Result;
use lens_config::HttpConfig;
use lens_search_engine::SearchConfig as EngineSearchConfig;

/// Start the HTTP API server
pub async fn start_http_server(
    search_config: EngineSearchConfig,
    http_config: HttpConfig,
) -> Result<()> {
    let index_path = search_config.index_path.clone();
    let config = http_server::ServerConfig {
        bind: http_config.bind.clone(),
        port: http_config.port,
        enable_cors: http_config.enable_cors,
        index_path,
        search_config,
        auth: http_config.auth,
    };

    http_server::start_server(config).await
}
