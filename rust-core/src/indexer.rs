use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingStats {
    pub files_indexed: u64,
    pub total_lines: u64,
    pub total_bytes: u64,
    pub duration_ms: u64,
}

pub struct Indexer {
    stats: IndexingStats,
}

impl Indexer {
    pub fn new_in_memory() -> Result<Self> {
        Ok(Indexer {
            stats: IndexingStats {
                files_indexed: 0,
                total_lines: 0,
                total_bytes: 0,
                duration_ms: 0,
            }
        })
    }
    
    pub fn index_document(&mut self, _file_path: &str, content: &str) -> Result<()> {
        // In a real implementation, this would write to the search index
        // For now, just validate the content
        if content.is_empty() {
            return Err(anyhow::anyhow!("Cannot index empty content"));
        }
        
        // Simulate indexing work
        let _lines = content.lines().count();
        let _bytes = content.len();
        
        Ok(())
    }
    
    pub fn index_directory<P: AsRef<Path>>(&mut self, dir_path: P) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        for entry in walkdir::WalkDir::new(dir_path).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            
            if path.is_file() && self.should_index_file(path) {
                let content = fs::read_to_string(path)?;
                let lines = content.lines().count() as u64;
                let bytes = content.len() as u64;
                
                self.index_document(path.to_string_lossy().as_ref(), &content)?;
                
                self.stats.files_indexed += 1;
                self.stats.total_lines += lines;
                self.stats.total_bytes += bytes;
            }
        }
        
        self.stats.duration_ms = start_time.elapsed().as_millis() as u64;
        Ok(())
    }
    
    pub fn get_stats(&self) -> &IndexingStats {
        &self.stats
    }
    
    fn should_index_file(&self, path: &Path) -> bool {
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            matches!(extension, "py" | "rs" | "ts" | "tsx" | "js" | "jsx" | "go" | "java" | "cpp" | "c" | "h")
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_indexer_creation() {
        let indexer = Indexer::new_in_memory();
        assert!(indexer.is_ok());
    }
    
    #[test]
    fn test_index_document() {
        let mut indexer = Indexer::new_in_memory().unwrap();
        let result = indexer.index_document("test.py", "def hello(): return 'world'");
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_empty_content_fails() {
        let mut indexer = Indexer::new_in_memory().unwrap();
        let result = indexer.index_document("empty.py", "");
        assert!(result.is_err());
    }
}