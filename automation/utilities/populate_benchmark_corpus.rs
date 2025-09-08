use std::fs;
use std::path::Path;
use walkdir::WalkDir;
use anyhow::Result;
use std::collections::HashMap;

// Import from the lens project
mod search;
use search::{SearchEngine, SearchEngineConfig, SearchDocument};

/// Benchmark types supported by the corpus system
#[derive(Debug, Clone, PartialEq)]
pub enum BenchmarkType {
    SWEBench,
    CodeSearchNet,
    CoIR,
    CoSQA,
}

impl BenchmarkType {
    fn corpus_directory(&self) -> &str {
        match self {
            BenchmarkType::SWEBench => "benchmark-corpus/swe-bench",
            BenchmarkType::CodeSearchNet => "benchmark-corpus/codesearchnet",
            BenchmarkType::CoIR => "benchmark-corpus/coir", 
            BenchmarkType::CoSQA => "benchmark-corpus/cosqa",
        }
    }
    
    fn index_directory(&self) -> &str {
        match self {
            BenchmarkType::SWEBench => "indexed-content/swe-bench",
            BenchmarkType::CodeSearchNet => "indexed-content/codesearchnet",
            BenchmarkType::CoIR => "indexed-content/coir",
            BenchmarkType::CoSQA => "indexed-content/cosqa",
        }
    }
    
    fn from_query_id(query_id: &str) -> Option<BenchmarkType> {
        if query_id.starts_with("swe_bench_") || query_id.contains("__") {
            Some(BenchmarkType::SWEBench)
        } else if query_id.starts_with("codesearchnet_") || query_id.starts_with("csn_") {
            Some(BenchmarkType::CodeSearchNet)
        } else if query_id.starts_with("coir_") {
            Some(BenchmarkType::CoIR)
        } else if query_id.starts_with("cosqa_") {
            Some(BenchmarkType::CoSQA)
        } else {
            // Default to CodeSearchNet for general queries
            Some(BenchmarkType::CodeSearchNet)
        }
    }
}

struct BenchmarkCorpusIndexer {
    search_engines: HashMap<BenchmarkType, SearchEngine>,
}

impl BenchmarkCorpusIndexer {
    async fn new() -> Result<Self> {
        let mut search_engines = HashMap::new();
        
        let benchmark_types = vec![
            BenchmarkType::SWEBench,
            BenchmarkType::CodeSearchNet, 
            BenchmarkType::CoIR,
            BenchmarkType::CoSQA,
        ];
        
        for benchmark_type in benchmark_types {
            let config = SearchEngineConfig {
                sla_target_ms: 150,
                lsp_routing_rate: 0.0, // Disable LSP for indexing
                ..Default::default()
            };
            
            let index_path = benchmark_type.index_directory();
            
            // Remove existing index
            if Path::new(index_path).exists() {
                fs::remove_dir_all(index_path)?;
                println!("🗑️  Removed existing index: {}", index_path);
            }
            
            let search_engine = SearchEngine::new(config, index_path).await?;
            search_engines.insert(benchmark_type.clone(), search_engine);
            println!("✅ Created search engine for {:?} at {}", benchmark_type, index_path);
        }
        
        Ok(BenchmarkCorpusIndexer { search_engines })
    }
    
    async fn index_benchmark_corpus(&mut self, benchmark_type: BenchmarkType) -> Result<u64> {
        let corpus_dir = benchmark_type.corpus_directory();
        
        if !Path::new(corpus_dir).exists() {
            println!("⚠️  Corpus directory {} not found, skipping {:?}", corpus_dir, benchmark_type);
            return Ok(0);
        }
        
        let search_engine = self.search_engines.get(&benchmark_type)
            .ok_or_else(|| anyhow::anyhow!("No search engine for {:?}", benchmark_type))?;
        
        let mut total_documents = 0;
        
        println!("📁 Indexing {:?} corpus from {}", benchmark_type, corpus_dir);
        
        for entry in WalkDir::new(corpus_dir) {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_file() && should_index_file(path) {
                println!("📄 Indexing: {}", path.display());
                
                let content = fs::read_to_string(path)?;
                if content.is_empty() {
                    continue;
                }
                
                // Different indexing strategies based on benchmark type
                match benchmark_type {
                    BenchmarkType::SWEBench => {
                        // For SWE-bench, index both file-level and line-level content
                        // This matches how real GitHub issues reference code
                        total_documents += self.index_source_file(search_engine, path, &content).await?;
                    },
                    BenchmarkType::CodeSearchNet => {
                        // For CodeSearchNet, focus on function/class level indexing
                        total_documents += self.index_code_structures(search_engine, path, &content).await?;
                    },
                    BenchmarkType::CoIR => {
                        // For CoIR, index document sections and paragraphs
                        total_documents += self.index_document_sections(search_engine, path, &content).await?;
                    },
                    BenchmarkType::CoSQA => {
                        // For CoSQA, index Q&A pairs with context
                        total_documents += self.index_qa_content(search_engine, path, &content).await?;
                    },
                }
            }
        }
        
        println!("🎉 Indexed {} documents for {:?}", total_documents, benchmark_type);
        Ok(total_documents)
    }
    
    async fn index_source_file(&self, search_engine: &SearchEngine, path: &Path, content: &str) -> Result<u64> {
        let mut doc_count = 0;
        
        // Index whole file (useful for file-level queries)
        let doc = SearchDocument {
            file_path: path.to_string_lossy().to_string(),
            content: content.to_string(),
            line_number: 1,
            language: get_language_from_path(path),
        };
        search_engine.index_document(&doc).await?;
        doc_count += 1;
        
        // Index per-line (useful for specific code locations)
        for (line_num, line_content) in content.lines().enumerate() {
            let trimmed_line = line_content.trim();
            if trimmed_line.is_empty() || trimmed_line.starts_with('#') || trimmed_line.starts_with("//") {
                continue;
            }
            
            let line_doc = SearchDocument {
                file_path: path.to_string_lossy().to_string(),
                content: line_content.to_string(),
                line_number: (line_num + 1) as u32,
                language: get_language_from_path(path),
            };
            
            search_engine.index_document(&line_doc).await?;
            doc_count += 1;
        }
        
        Ok(doc_count)
    }
    
    async fn index_code_structures(&self, search_engine: &SearchEngine, path: &Path, content: &str) -> Result<u64> {
        let mut doc_count = 0;
        let language = get_language_from_path(path);
        
        // Simple structure detection (could be enhanced with proper parsing)
        let lines: Vec<&str> = content.lines().collect();
        let mut current_function = String::new();
        let mut function_start_line = 0;
        
        for (line_num, line) in lines.iter().enumerate() {
            let trimmed_line = line.trim();
            
            // Detect function/class/method definitions (simplified)
            let is_function_def = match language.as_ref().map(|s| s.as_str()) {
                Some("python") => trimmed_line.starts_with("def ") || trimmed_line.starts_with("class "),
                Some("javascript") | Some("typescript") => {
                    trimmed_line.starts_with("function ") || 
                    trimmed_line.starts_with("class ") ||
                    trimmed_line.contains("= function") ||
                    trimmed_line.contains("=> ")
                },
                Some("java") => trimmed_line.contains("public ") || trimmed_line.contains("private "),
                Some("rust") => trimmed_line.starts_with("fn ") || trimmed_line.starts_with("impl "),
                _ => trimmed_line.contains("function") || trimmed_line.contains("class"),
            };
            
            if is_function_def {
                // Index previous function if exists
                if !current_function.is_empty() {
                    let func_doc = SearchDocument {
                        file_path: path.to_string_lossy().to_string(),
                        content: current_function.clone(),
                        line_number: function_start_line as u32,
                        language: language.clone(),
                    };
                    search_engine.index_document(&func_doc).await?;
                    doc_count += 1;
                }
                
                // Start new function
                current_function = String::new();
                function_start_line = line_num + 1;
            }
            
            current_function.push_str(line);
            current_function.push('\n');
        }
        
        // Index final function
        if !current_function.is_empty() {
            let func_doc = SearchDocument {
                file_path: path.to_string_lossy().to_string(),
                content: current_function,
                line_number: function_start_line as u32,
                language: language.clone(),
            };
            search_engine.index_document(&func_doc).await?;
            doc_count += 1;
        }
        
        Ok(doc_count)
    }
    
    async fn index_document_sections(&self, search_engine: &SearchEngine, path: &Path, content: &str) -> Result<u64> {
        let mut doc_count = 0;
        
        // Split content into sections (by headers or paragraphs)
        let sections: Vec<&str> = if content.contains("## ") {
            content.split("## ").collect()
        } else {
            content.split("\n\n").collect()
        };
        
        for (section_num, section_content) in sections.iter().enumerate() {
            if section_content.trim().len() < 20 { // Skip very short sections
                continue;
            }
            
            let section_doc = SearchDocument {
                file_path: path.to_string_lossy().to_string(),
                content: section_content.to_string(),
                line_number: (section_num + 1) as u32,
                language: get_language_from_path(path),
            };
            
            search_engine.index_document(&section_doc).await?;
            doc_count += 1;
        }
        
        Ok(doc_count)
    }
    
    async fn index_qa_content(&self, search_engine: &SearchEngine, path: &Path, content: &str) -> Result<u64> {
        let mut doc_count = 0;
        
        // Index Q&A pairs separately
        let sections: Vec<&str> = content.split("## ").collect();
        
        for (section_num, section_content) in sections.iter().enumerate() {
            if section_content.trim().len() < 10 {
                continue;
            }
            
            // Index the whole Q&A pair
            let qa_doc = SearchDocument {
                file_path: path.to_string_lossy().to_string(),
                content: section_content.to_string(),
                line_number: (section_num + 1) as u32,
                language: Some("markdown".to_string()),
            };
            
            search_engine.index_document(&qa_doc).await?;
            doc_count += 1;
            
            // Also index question and answer separately if they're clearly marked
            if section_content.contains("Question") && section_content.contains("Answer") {
                let parts: Vec<&str> = section_content.split("Answer").collect();
                if parts.len() >= 2 {
                    // Index question
                    let question_doc = SearchDocument {
                        file_path: format!("{}_question", path.to_string_lossy()),
                        content: parts[0].to_string(),
                        line_number: (section_num * 2 + 1) as u32,
                        language: Some("markdown".to_string()),
                    };
                    search_engine.index_document(&question_doc).await?;
                    doc_count += 1;
                    
                    // Index answer
                    let answer_doc = SearchDocument {
                        file_path: format!("{}_answer", path.to_string_lossy()),
                        content: parts[1].to_string(),
                        line_number: (section_num * 2 + 2) as u32,
                        language: Some("markdown".to_string()),
                    };
                    search_engine.index_document(&answer_doc).await?;
                    doc_count += 1;
                }
            }
        }
        
        Ok(doc_count)
    }
    
    async fn test_search(&self, benchmark_type: BenchmarkType, query: &str) -> Result<()> {
        let search_engine = self.search_engines.get(&benchmark_type)
            .ok_or_else(|| anyhow::anyhow!("No search engine for {:?}", benchmark_type))?;
        
        println!("🧪 Testing {:?} search with query: '{}'", benchmark_type, query);
        let (results, _metrics) = search_engine.search(query, 3).await?;
        println!("✅ {:?} search returned {} results", benchmark_type, results.len());
        
        for (i, result) in results.iter().enumerate() {
            let preview = result.content.chars().take(80).collect::<String>();
            println!("  {}. {} (score: {:.2})", i + 1, preview, result.score);
        }
        
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    
    println!("🔧 Populating benchmark-specific corpus indexes...");
    
    let mut indexer = BenchmarkCorpusIndexer::new().await?;
    
    let mut total_documents = 0;
    
    // Index all benchmark corpuses
    for benchmark_type in [
        BenchmarkType::SWEBench,
        BenchmarkType::CodeSearchNet,
        BenchmarkType::CoIR, 
        BenchmarkType::CoSQA,
    ] {
        let doc_count = indexer.index_benchmark_corpus(benchmark_type.clone()).await?;
        total_documents += doc_count;
        
        // Test each corpus with a relevant query
        let test_query = match benchmark_type {
            BenchmarkType::SWEBench => "function bug fix",
            BenchmarkType::CodeSearchNet => "sort array",
            BenchmarkType::CoIR => "database connection",
            BenchmarkType::CoSQA => "how to",
        };
        
        indexer.test_search(benchmark_type, test_query).await?;
        println!();
    }
    
    println!("🎉 Successfully indexed {} total documents across all benchmarks!", total_documents);
    println!();
    println!("📊 Corpus Summary:");
    println!("├─ SWE-bench: Real GitHub repository code for issue resolution");
    println!("├─ CodeSearchNet: Diverse code examples for general search");
    println!("├─ CoIR: Technical documentation for information retrieval");
    println!("└─ CoSQA: Q&A pairs for code question answering");
    println!();
    println!("🔄 Next steps:");
    println!("1. Run semantic search benchmarks with domain-specific corpuses");
    println!("2. Compare results against previous generic corpus approach");
    println!("3. Measure semantic lift improvement across benchmark types");
    
    Ok(())
}

fn should_index_file(path: &Path) -> bool {
    if let Some(extension) = path.extension() {
        matches!(
            extension.to_str(), 
            Some("rs" | "ts" | "js" | "jsx" | "tsx" | "py" | "go" | "java" | "cpp" | "c" | "h" | "cs" | "rb" | "php" | "scala" | "kt" | "swift" | "md" | "txt" | "json" | "yaml" | "yml" | "toml")
        )
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
            "tsx" => "typescript",
            "js" => "javascript",
            "jsx" => "javascript",
            "py" => "python",
            "go" => "go",
            "java" => "java",
            "cpp" | "cc" | "cxx" => "cpp",
            "c" => "c",
            "h" => "header",
            "cs" => "csharp",
            "rb" => "ruby",
            "php" => "php",
            "scala" => "scala",
            "kt" => "kotlin",
            "swift" => "swift",
            "md" => "markdown",
            "txt" => "text",
            "json" => "json",
            "yaml" | "yml" => "yaml",
            "toml" => "toml",
            _ => ext,
        })
        .map(String::from)
}