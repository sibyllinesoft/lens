//! Optimize command implementation

use anyhow::Result;
use lens_search_engine::SearchEngine;
use std::{path::Path, sync::Arc};

/// Optimize the search index
pub async fn optimize_index(search_engine: Arc<SearchEngine>, index_path: &Path) -> Result<()> {
    println!("Optimizing search index at {:?}...", index_path);
    let start = std::time::Instant::now();

    search_engine.optimize().await?;

    let duration = start.elapsed();
    println!("Index optimization completed in {:?}", duration);

    Ok(())
}
