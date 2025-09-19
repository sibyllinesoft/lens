//! LSP server command implementations

use anyhow::Result;
use lens_search_engine::SearchEngine;
use std::path::PathBuf;
use tracing::info;

/// Start the LSP server over stdio
pub async fn start_lsp_stdio_server(index_path: PathBuf) -> Result<()> {
    info!(
        "Starting LSP server over stdio with index at: {:?}",
        index_path
    );

    // Create search engine
    let search_engine = std::sync::Arc::new(SearchEngine::new(&index_path).await?);

    // Start LSP server
    lens_lsp_server::start_lsp_server(search_engine).await?;

    Ok(())
}

/// Start the LSP server over TCP (for debugging)
pub async fn start_lsp_tcp_server(index_path: PathBuf, port: u16) -> Result<()> {
    info!(
        "Starting LSP server over TCP on port {} with index at: {:?}",
        port, index_path
    );

    // Create search engine
    let search_engine = std::sync::Arc::new(SearchEngine::new(&index_path).await?);

    // Start TCP LSP server
    lens_lsp_server::start_lsp_tcp_server(search_engine, port).await?;

    Ok(())
}
