//! Clear command implementation

use anyhow::Result;
use lens_search_engine::SearchEngine;
use std::{path::Path, sync::Arc};

/// Clear the search index
pub async fn clear_index(
    search_engine: Option<Arc<SearchEngine>>,
    index_path: &Path,
    skip_confirmation: bool,
) -> Result<()> {
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

    match search_engine {
        Some(engine) => {
            engine.clear_index().await?;
            println!("Index cleared and properly reinitialized: {:?}", index_path);
        }
        None => {
            println!("Index directory does not exist: {:?}", index_path);
        }
    }

    Ok(())
}
