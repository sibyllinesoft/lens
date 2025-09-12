//! General Dataset Loading Utilities
//!
//! Provides utilities for loading various dataset formats and managing
//! corpus indexing for benchmarking.

use super::types::{GoldenQuery, QueryType, LoadingError};
use anyhow::{anyhow, Result};
use serde_json;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use tokio::fs;
use tracing::{info, warn, debug};

/// General dataset loader for various formats
pub struct DatasetLoader {
    base_path: PathBuf,
}

impl DatasetLoader {
    /// Create new dataset loader
    pub fn new<P: AsRef<Path>>(base_path: P) -> Self {
        Self {
            base_path: base_path.as_ref().to_path_buf(),
        }
    }

    /// Load golden queries from JSON file (legacy format)
    pub async fn load_golden_queries<P: AsRef<Path>>(&self, file_path: P) -> Result<Vec<GoldenQuery>> {
        let full_path = self.base_path.join(file_path);
        let content = fs::read_to_string(&full_path).await
            .map_err(|e| anyhow!(LoadingError::IoError { source: e }))?;

        // Try different JSON formats
        
        // Format 1: Array of GoldenQuery objects
        if let Ok(queries) = serde_json::from_str::<Vec<GoldenQuery>>(&content) {
            info!("📄 Loaded {} golden queries from {}", queries.len(), full_path.display());
            return Ok(queries);
        }

        // Format 2: Simple array of query objects (legacy)
        if let Ok(simple_queries) = serde_json::from_str::<Vec<serde_json::Value>>(&content) {
            info!("📄 Converting {} legacy queries from {}", simple_queries.len(), full_path.display());
            return self.convert_legacy_queries(simple_queries);
        }

        Err(anyhow!(LoadingError::InvalidFormat { 
            reason: format!("Unrecognized JSON format in {}", full_path.display())
        }))
    }

    /// Convert legacy query format to GoldenQuery
    fn convert_legacy_queries(&self, legacy_queries: Vec<serde_json::Value>) -> Result<Vec<GoldenQuery>> {
        let mut queries = Vec::new();

        for (index, legacy_query) in legacy_queries.into_iter().enumerate() {
            let query_obj = legacy_query.as_object()
                .ok_or_else(|| anyhow!(LoadingError::InvalidFormat {
                    reason: format!("Query {} is not an object", index)
                }))?;

            let query_str = query_obj.get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let expected_files = query_obj.get("expected_files")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter()
                    .filter_map(|f| f.as_str().map(|s| s.to_string()))
                    .collect::<Vec<_>>())
                .unwrap_or_else(Vec::new);

            let query_type = query_obj.get("query_type")
                .and_then(|v| v.as_str())
                .and_then(|s| self.parse_query_type(s))
                .unwrap_or(QueryType::Identifier);

            let language = query_obj.get("language")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            queries.push(GoldenQuery {
                query: query_str,
                expected_files,
                query_type,
                metadata: HashMap::new(),
                language,
                confidence: None,
            });
        }

        Ok(queries)
    }

    /// Parse query type from string
    fn parse_query_type(&self, type_str: &str) -> Option<QueryType> {
        match type_str.to_lowercase().as_str() {
            "exact_match" | "exact" => Some(QueryType::ExactMatch),
            "identifier" | "id" => Some(QueryType::Identifier),
            "structural" | "struct" => Some(QueryType::Structural),
            "semantic" | "nlp" => Some(QueryType::Semantic),
            "reference" | "ref" => Some(QueryType::Reference),
            "definition" | "def" => Some(QueryType::Definition),
            _ => None,
        }
    }

    /// Discover corpus files for indexing
    pub async fn discover_corpus_files<P: AsRef<Path>>(&self, corpus_path: P) -> Result<Vec<PathBuf>> {
        let corpus_dir = self.base_path.join(corpus_path);
        
        if !corpus_dir.exists() {
            return Err(anyhow!(LoadingError::DatasetNotFound {
                path: corpus_dir.to_string_lossy().to_string()
            }));
        }

        let mut files = Vec::new();
        self.walk_directory(&corpus_dir, &mut files).await?;
        
        // Filter for relevant source files
        let source_files: Vec<PathBuf> = files.into_iter()
            .filter(|path| self.is_source_file(path))
            .collect();

        info!("📁 Discovered {} source files in {}", source_files.len(), corpus_dir.display());
        Ok(source_files)
    }

    /// Recursively walk directory to find files
    fn walk_directory<'a>(&'a self, dir: &'a Path, files: &'a mut Vec<PathBuf>) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let mut entries = fs::read_dir(dir).await?;
            
            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();
                
                if path.is_dir() {
                    // Skip common build/cache directories
                    let dir_name = path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("");
                    
                    if !self.should_skip_directory(dir_name) {
                        self.walk_directory(&path, files).await?;
                    }
                } else {
                    files.push(path);
                }
            }
            
            Ok(())
        })
    }

    /// Check if directory should be skipped during corpus discovery
    fn should_skip_directory(&self, dir_name: &str) -> bool {
        matches!(dir_name, 
            "target" | "node_modules" | ".git" | ".svn" | ".hg" | 
            "build" | "dist" | "out" | ".next" | ".nuxt" |
            "__pycache__" | ".pytest_cache" | "coverage" |
            ".DS_Store" | "Thumbs.db"
        )
    }

    /// Check if file is a source file worth indexing
    fn is_source_file(&self, path: &Path) -> bool {
        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        matches!(extension.as_str(),
            "rs" | "py" | "js" | "ts" | "jsx" | "tsx" | 
            "java" | "kt" | "scala" | "go" | "c" | "cpp" | 
            "cc" | "cxx" | "h" | "hpp" | "cs" | "php" | 
            "rb" | "swift" | "m" | "mm" | "dart" | "elm" |
            "clj" | "cljs" | "hs" | "ml" | "fs" | "pl" | 
            "r" | "jl" | "lua" | "sh" | "bash" | "zsh" |
            "sql" | "graphql" | "proto" | "thrift"
        )
    }

    /// Generate corpus statistics
    pub async fn generate_corpus_stats<P: AsRef<Path>>(&self, corpus_files: &[PathBuf]) -> Result<CorpusStats> {
        let mut stats = CorpusStats {
            total_files: corpus_files.len(),
            total_lines: 0,
            total_size_bytes: 0,
            language_distribution: HashMap::new(),
        };

        for file_path in corpus_files {
            if let Ok(metadata) = fs::metadata(file_path).await {
                stats.total_size_bytes += metadata.len();
            }

            if let Ok(content) = fs::read_to_string(file_path).await {
                let line_count = content.lines().count();
                stats.total_lines += line_count;

                // Count by language (based on extension)
                if let Some(ext) = file_path.extension().and_then(|e| e.to_str()) {
                    *stats.language_distribution.entry(ext.to_string()).or_insert(0) += 1;
                }
            }
        }

        Ok(stats)
    }

    /// Validate corpus file exists and is accessible
    pub async fn validate_corpus_file<P: AsRef<Path>>(&self, file_path: P) -> bool {
        let full_path = self.base_path.join(file_path);
        
        match fs::metadata(&full_path).await {
            Ok(metadata) => {
                if metadata.is_file() && metadata.len() > 0 {
                    debug!("✅ Corpus file validated: {}", full_path.display());
                    true
                } else {
                    debug!("❌ Invalid corpus file: {}", full_path.display());
                    false
                }
            }
            Err(_) => {
                debug!("❌ Corpus file not found: {}", full_path.display());
                false
            }
        }
    }

    /// Get file content for corpus validation
    pub async fn get_file_content<P: AsRef<Path>>(&self, file_path: P) -> Result<String> {
        let full_path = self.base_path.join(file_path);
        
        fs::read_to_string(&full_path).await
            .map_err(|e| anyhow!(LoadingError::IoError { source: e }))
    }
}

/// Corpus statistics
#[derive(Debug, Clone)]
pub struct CorpusStats {
    pub total_files: usize,
    pub total_lines: usize,
    pub total_size_bytes: u64,
    pub language_distribution: HashMap<String, usize>,
}

impl CorpusStats {
    /// Get formatted size string
    pub fn formatted_size(&self) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = self.total_size_bytes as f64;
        let mut unit_index = 0;
        
        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }
        
        format!("{:.1} {}", size, UNITS[unit_index])
    }

    /// Get most common languages (top N)
    pub fn top_languages(&self, n: usize) -> Vec<(String, usize)> {
        let mut sorted_langs: Vec<_> = self.language_distribution.iter()
            .map(|(lang, count)| (lang.clone(), *count))
            .collect();
        
        sorted_langs.sort_by(|a, b| b.1.cmp(&a.1));
        sorted_langs.into_iter().take(n).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tokio::fs::File;
    use tokio::io::AsyncWriteExt;

    #[tokio::test]
    async fn test_dataset_loader_creation() {
        let temp_dir = TempDir::new().unwrap();
        let loader = DatasetLoader::new(temp_dir.path());
        
        assert_eq!(loader.base_path, temp_dir.path());
    }

    #[tokio::test]
    async fn test_query_type_parsing() {
        let loader = DatasetLoader::new(".");
        
        assert_eq!(loader.parse_query_type("identifier"), Some(QueryType::Identifier));
        assert_eq!(loader.parse_query_type("exact_match"), Some(QueryType::ExactMatch));
        assert_eq!(loader.parse_query_type("structural"), Some(QueryType::Structural));
        assert_eq!(loader.parse_query_type("invalid"), None);
    }

    #[tokio::test]
    async fn test_source_file_detection() {
        let loader = DatasetLoader::new(".");
        
        assert!(loader.is_source_file(Path::new("test.rs")));
        assert!(loader.is_source_file(Path::new("script.py")));
        assert!(loader.is_source_file(Path::new("component.tsx")));
        assert!(!loader.is_source_file(Path::new("data.json")));
        assert!(!loader.is_source_file(Path::new("image.png")));
    }

    #[tokio::test]
    async fn test_directory_skip_logic() {
        let loader = DatasetLoader::new(".");
        
        assert!(loader.should_skip_directory("target"));
        assert!(loader.should_skip_directory("node_modules"));
        assert!(loader.should_skip_directory(".git"));
        assert!(!loader.should_skip_directory("src"));
        assert!(!loader.should_skip_directory("lib"));
    }

    #[tokio::test]
    async fn test_legacy_query_conversion() {
        let temp_dir = TempDir::new().unwrap();
        let loader = DatasetLoader::new(temp_dir.path());

        // Create test data - each item should be a JSON object, not an array
        let legacy_data = serde_json::json!({
            "query": "function test",
            "expected_files": ["test.js"],
            "query_type": "identifier"
        });

        let legacy_queries = vec![legacy_data];
        let converted = loader.convert_legacy_queries(legacy_queries).unwrap();
        
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].query, "function test");
        assert_eq!(converted[0].query_type, QueryType::Identifier);
    }

    #[tokio::test]
    async fn test_corpus_stats_formatting() {
        let stats = CorpusStats {
            total_files: 100,
            total_lines: 10000,
            total_size_bytes: 2_048_576, // 2MB
            language_distribution: {
                let mut map = HashMap::new();
                map.insert("rs".to_string(), 50);
                map.insert("py".to_string(), 30);
                map.insert("js".to_string(), 20);
                map
            },
        };

        assert_eq!(stats.formatted_size(), "2.0 MB");
        
        let top_langs = stats.top_languages(2);
        assert_eq!(top_langs.len(), 2);
        assert_eq!(top_langs[0], ("rs".to_string(), 50));
        assert_eq!(top_langs[1], ("py".to_string(), 30));
    }
}