//! LSP server command implementations

use anyhow::Result;
use lens_lsp_server::LspServerConfig;
use lens_search_engine::{SearchConfig as EngineSearchConfig, SearchEngine};
use std::sync::Arc;
use tracing::info;

/// Start the LSP server over stdio
pub async fn start_lsp_stdio_server(
    search_config: EngineSearchConfig,
    lsp_config: LspServerConfig,
) -> Result<()> {
    let index_path = search_config.index_path.clone();
    info!(
        "Starting LSP server over stdio with index at: {:?}",
        index_path
    );

    // Create search engine
    let search_engine = Arc::new(SearchEngine::with_config(search_config).await?);

    // Start LSP server
    lens_lsp_server::start_lsp_server(search_engine, lsp_config).await?;

    Ok(())
}

/// Start the LSP server over TCP (for debugging)
pub async fn start_lsp_tcp_server(
    search_config: EngineSearchConfig,
    lsp_config: LspServerConfig,
    port: u16,
) -> Result<()> {
    let index_path = search_config.index_path.clone();
    info!(
        "Starting LSP server over TCP on port {} with index at: {:?}",
        port, index_path
    );

    // Create search engine
    let search_engine = Arc::new(SearchEngine::with_config(search_config).await?);

    // Start TCP LSP server
    lens_lsp_server::start_lsp_tcp_server(search_engine, lsp_config, port).await?;

    Ok(())
}
