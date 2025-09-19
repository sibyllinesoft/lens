//! Clear command implementation

use anyhow::Result;
use lens_search_engine::SearchEngine;
use std::path::PathBuf;

/// Clear the search index
pub async fn clear_index(index_path: PathBuf, skip_confirmation: bool) -> Result<()> {
    if !skip_confirmation {
        println!(
            "This will delete the entire search index at: {:?}",
            index_path
        );
        println!("Are you sure? (y/N)");

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;

        if !input.trim().to_lowercase().starts_with('y') {
            println!("Cancelled.");
            return Ok(());
        }
    }

    // Create search engine and use its proper clear_index method
    if index_path.exists() {
        let search_engine = SearchEngine::new(&index_path).await?;
        search_engine.clear_index().await?;
        println!("Index cleared and properly reinitialized: {:?}", index_path);
    } else {
        println!("Index directory does not exist: {:?}", index_path);
    }

    Ok(())
}
