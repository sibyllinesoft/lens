//! Optimize command implementation

use anyhow::Result;
use lens_search_engine::{SearchConfig as EngineSearchConfig, SearchEngine};

/// Optimize the search index
pub async fn optimize_index(search_config: EngineSearchConfig) -> Result<()> {
    let index_path = search_config.index_path.clone();
    let search_engine = SearchEngine::with_config(search_config).await?;

    println!("Optimizing search index at {:?}...", index_path);
    let start = std::time::Instant::now();

    search_engine.optimize().await?;

    let duration = start.elapsed();
    println!("Index optimization completed in {:?}", duration);

    Ok(())
}
