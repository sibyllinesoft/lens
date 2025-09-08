use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub file: String,
    pub line: u32,
    pub col: u32,
    pub lang: String,
    pub snippet: String,
    pub score: f64,
    pub why: Vec<String>,
    pub ast_path: Option<String>,
    pub symbol_kind: Option<String>,
    pub byte_offset: Option<u64>,
    pub span_len: Option<u32>,
}

#[derive(Debug, Clone)]
struct DocumentIndex {
    file_path: String,
    lines: Vec<String>,
}

pub struct SearchEngine {
    // Simple in-memory search index for initial implementation
    documents: BTreeMap<String, DocumentIndex>,
}

impl SearchEngine {
    pub fn new_in_memory() -> Result<Self> {
        Ok(SearchEngine {
            documents: BTreeMap::new(),
        })
    }

    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let query_lower = query.to_lowercase();
        let mut results = Vec::new();
        
        for (file_path, doc) in &self.documents {
            for (line_num, line_content) in doc.lines.iter().enumerate() {
                if line_content.to_lowercase().contains(&query_lower) {
                    let score = self.calculate_score(&query_lower, line_content);
                    
                    results.push(SearchResult {
                        file: file_path.clone(),
                        line: line_num as u32 + 1,
                        col: line_content.to_lowercase().find(&query_lower).unwrap_or(0) as u32,
                        lang: self.detect_language(file_path),
                        snippet: line_content.clone(),
                        score,
                        why: vec!["exact".to_string()],
                        ast_path: None,
                        symbol_kind: None,
                        byte_offset: None,
                        span_len: Some(query.len() as u32),
                    });
                    
                    if results.len() >= limit {
                        break;
                    }
                }
            }
            
            if results.len() >= limit {
                break;
            }
        }
        
        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        
        Ok(results)
    }
    
    pub fn index_document(&mut self, file_path: &str, content: &str) -> Result<()> {
        if content.is_empty() {
            return Err(anyhow!("Cannot index empty content"));
        }
        
        let lines: Vec<String> = content.lines().map(|line| line.to_string()).collect();
        
        self.documents.insert(file_path.to_string(), DocumentIndex {
            file_path: file_path.to_string(),
            lines,
        });
        
        Ok(())
    }
    
    fn calculate_score(&self, query: &str, line: &str) -> f64 {
        let line_lower = line.to_lowercase();
        let matches = line_lower.matches(query).count() as f64;
        let line_length = line.len() as f64;
        
        // Simple scoring: more matches and shorter lines get higher scores
        matches / (line_length / 100.0).max(1.0)
    }
    
    fn detect_language(&self, file_path: &str) -> String {
        if file_path.ends_with(".py") {
            "python".to_string()
        } else if file_path.ends_with(".rs") {
            "rust".to_string()
        } else if file_path.ends_with(".ts") || file_path.ends_with(".tsx") {
            "typescript".to_string()
        } else if file_path.ends_with(".js") || file_path.ends_with(".jsx") {
            "javascript".to_string()
        } else {
            "text".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_engine_creation() {
        let engine = SearchEngine::new_in_memory();
        assert!(engine.is_ok());
    }
    
    #[test] 
    fn test_index_and_search() {
        let mut engine = SearchEngine::new_in_memory().unwrap();
        
        // Index a sample document
        let result = engine.index_document("test.py", "def hello():\n    return 'world'");
        assert!(result.is_ok());
        
        // Search for the content
        let results = engine.search("hello", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].file, "test.py");
        assert_eq!(results[0].lang, "python");
        assert_eq!(results[0].line, 1);
    }
    
    #[test]
    fn test_empty_content_fails() {
        let mut engine = SearchEngine::new_in_memory().unwrap();
        let result = engine.index_document("empty.py", "");
        assert!(result.is_err());
    }
}