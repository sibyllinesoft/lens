//! Index command implementation

use anyhow::{anyhow, Result};
use lens_search_engine::SearchEngine;
use std::path::PathBuf;
use tracing::info;

/// Index a directory
pub async fn index_directory(
    index_path: PathBuf,
    directory: PathBuf,
    _force: bool,
    show_progress: bool,
) -> Result<()> {
    info!("Indexing directory: {:?} -> {:?}", directory, index_path);

    if !directory.exists() {
        return Err(anyhow!("Directory does not exist: {:?}", directory));
    }

    // Create search engine
    let search_engine = SearchEngine::new(&index_path).await?;

    // Show progress if requested
    if show_progress {
        println!("Indexing files from {:?}...", directory);
    }

    // Index the directory
    let stats = search_engine.index_directory(&directory).await?;

    // Show results
    println!("Indexing complete:");
    println!("  Files indexed: {}", stats.files_indexed);
    println!("  Files failed: {}", stats.files_failed);
    println!("  Lines indexed: {}", stats.lines_indexed);
    println!("  Symbols extracted: {}", stats.symbols_extracted);
    println!("  Duration: {:?}", stats.indexing_duration);
    println!("  Success rate: {:.1}%", stats.success_rate() * 100.0);
    println!("  Files/sec: {:.2}", stats.files_per_second());

    Ok(())
}
