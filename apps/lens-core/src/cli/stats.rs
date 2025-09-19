//! Stats command implementation

use anyhow::Result;
use lens_search_engine::{SearchConfig as EngineSearchConfig, SearchEngine};

/// Show index statistics
pub async fn show_stats(search_config: EngineSearchConfig) -> Result<()> {
    let index_path = search_config.index_path.clone();
    let search_engine = SearchEngine::with_config(search_config).await?;
    let stats = search_engine.get_stats().await?;

    println!("Index Statistics ({:?}):", index_path);
    println!("  Total documents: {}", stats.total_documents);
    println!("  Index size: {}", stats.human_readable_size());
    println!("  Supported languages: {}", stats.supported_languages);
    println!("  Last updated: {:?}", stats.last_updated);

    if let Some(avg_size) = Some(stats.average_document_size()) {
        if avg_size > 0.0 {
            println!("  Average document size: {:.1} bytes", avg_size);
        }
    }

    Ok(())
}
