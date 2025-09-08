use std::fs;
use std::path::Path;
use walkdir::WalkDir;
use anyhow::Result;

// Import from the lens project
mod search;
use search::{SearchEngine, SearchEngineConfig, SearchDocument};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    
    println!("🔧 Populating empty search index with project source files...");
    
    // Remove the empty index and let SearchEngine recreate it
    let index_path = "./indexed-content";
    if Path::new(index_path).exists() {
        fs::remove_dir_all(index_path)?;
        println!("🗑️  Removed empty index directory");
    }
    
    // Create search engine with default config
    let config = SearchEngineConfig {
        sla_target_ms: 150,
        lsp_routing_rate: 0.0, // Disable LSP for indexing
        ..Default::default()
    };
    
    let search_engine = SearchEngine::new(config, index_path).await?;
    println!("✅ Created search engine with fresh index");
    
    // Index project source files
    let mut total_documents = 0;
    let source_dirs = ["src", "rust-core/src"];
    
    for source_dir in &source_dirs {
        if !Path::new(source_dir).exists() {
            println!("⚠️  Directory {} not found, skipping", source_dir);
            continue;
        }
        
        for entry in WalkDir::new(source_dir) {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_file() && should_index_file(path) {
                println!("📄 Indexing: {}", path.display());
                
                let content = fs::read_to_string(path)?;
                if content.is_empty() {
                    continue;
                }
                
                // Index the file content - could be done per-line or per-file
                // For now, let's index per-file to get basic search working
                let doc = SearchDocument {
                    file_path: path.to_string_lossy().to_string(),
                    content: content.clone(),
                    line_number: 1, // Use 1 for whole-file indexing
                    language: get_language_from_path(path),
                };
                
                search_engine.index_document(&doc).await?;
                total_documents += 1;
                
                // Also index per-line for better granularity
                for (line_num, line_content) in content.lines().enumerate() {
                    if line_content.trim().is_empty() {
                        continue;
                    }
                    
                    let line_doc = SearchDocument {
                        file_path: path.to_string_lossy().to_string(),
                        content: line_content.to_string(),
                        line_number: (line_num + 1) as u32,
                        language: get_language_from_path(path),
                    };
                    
                    search_engine.index_document(&line_doc).await?;
                    total_documents += 1;
                }
            }
        }
    }
    
    println!("🎉 Successfully indexed {} documents!", total_documents);
    println!("🔍 Index populated at: {}", index_path);
    
    // Test the search to verify it works
    println!("🧪 Testing search functionality...");
    let (results, _metrics) = search_engine.search("function", 5).await?;
    println!("✅ Search test returned {} results", results.len());
    
    if !results.is_empty() {
        println!("📋 Sample result: {} from {}", 
                 results[0].content.chars().take(50).collect::<String>(),
                 results[0].file_path);
    }
    
    Ok(())
}

fn should_index_file(path: &Path) -> bool {
    if let Some(extension) = path.extension() {
        matches!(extension.to_str(), Some("rs" | "ts" | "js" | "py" | "go" | "java" | "cpp" | "c" | "h"))
    } else {
        false
    }
}

fn get_language_from_path(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| match ext {
            "rs" => "rust",
            "ts" => "typescript", 
            "js" => "javascript",
            "py" => "python",
            "go" => "go",
            "java" => "java",
            "cpp" | "cc" | "cxx" => "cpp",
            "c" => "c",
            "h" => "header",
            _ => ext,
        })
        .map(String::from)
}