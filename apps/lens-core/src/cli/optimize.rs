//! Optimize command implementation

use anyhow::Result;
use lens_search_engine::SearchEngine;
use std::path::PathBuf;

/// Optimize the search index
pub async fn optimize_index(index_path: PathBuf) -> Result<()> {
    let search_engine = SearchEngine::new(&index_path).await?;

    println!("Optimizing search index...");
    let start = std::time::Instant::now();

    search_engine.optimize().await?;

    let duration = start.elapsed();
    println!("Index optimization completed in {:?}", duration);

    Ok(())
}
