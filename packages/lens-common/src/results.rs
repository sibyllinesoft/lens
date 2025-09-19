//! Search results and statistics
//!
//! Shared types for search results and index statistics

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::language::ProgrammingLanguage;

/// A single search result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    /// Path to the file containing this result
    pub file_path: String,
    /// Line number (1-based)
    pub line_number: u32,
    /// Column number (1-based, 0 if not available)
    pub column: u32,
    /// The matching content/line
    pub content: String,
    /// Search relevance score
    pub score: f64,
    /// Programming language of the file
    pub language: Option<ProgrammingLanguage>,
    /// Type of search result
    pub result_type: SearchResultType,
    /// Context lines around the match
    pub context_lines: Option<Vec<String>>,
}

/// Type of search result
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SearchResultType {
    /// Regular text match
    Text,
    /// Function definition
    Function,
    /// Class/struct definition
    Class,
    /// Variable/field definition
    Variable,
    /// Import/include statement
    Import,
    /// Comment
    Comment,
    /// Symbol reference
    Symbol,
}

impl SearchResult {
    /// Create a new search result
    pub fn new(file_path: String, line_number: u32, content: String, score: f64) -> Self {
        Self {
            file_path,
            line_number,
            column: 0,
            content,
            score,
            language: None,
            result_type: SearchResultType::Text,
            context_lines: None,
        }
    }

    /// Set the language
    pub fn with_language(mut self, language: ProgrammingLanguage) -> Self {
        self.language = Some(language);
        self
    }

    /// Set the result type
    pub fn with_type(mut self, result_type: SearchResultType) -> Self {
        self.result_type = result_type;
        self
    }

    /// Set the column
    pub fn with_column(mut self, column: u32) -> Self {
        self.column = column;
        self
    }

    /// Set context lines
    pub fn with_context(mut self, context_lines: Vec<String>) -> Self {
        self.context_lines = Some(context_lines);
        self
    }

    /// Extract file name from path
    pub fn file_name(&self) -> String {
        std::path::Path::new(&self.file_path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(&self.file_path)
            .to_string()
    }

    /// Get file extension
    pub fn file_extension(&self) -> Option<String> {
        std::path::Path::new(&self.file_path)
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|s| s.to_string())
    }

    /// Check if this result is a definition (function, class, etc.)
    pub fn is_definition(&self) -> bool {
        matches!(
            self.result_type,
            SearchResultType::Function | SearchResultType::Class | SearchResultType::Variable
        )
    }

    /// Get a unique identifier for this result
    pub fn unique_id(&self) -> String {
        format!("{}:{}:{}", self.file_path, self.line_number, self.column)
    }
}

/// Collection of search results with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResults {
    /// The actual search results
    pub results: Vec<SearchResult>,
    /// Total number of matches found
    pub total_matches: usize,
    /// Time taken to perform the search
    pub search_duration: Duration,
    /// Whether results came from cache
    pub from_cache: bool,
}

impl SearchResults {
    /// Create new search results
    pub fn new(
        results: Vec<SearchResult>,
        total_matches: usize,
        search_duration: Duration,
    ) -> Self {
        Self {
            results,
            total_matches,
            search_duration,
            from_cache: false,
        }
    }

    /// Create empty search results
    pub fn empty() -> Self {
        Self {
            results: Vec::new(),
            total_matches: 0,
            search_duration: Duration::from_millis(0),
            from_cache: false,
        }
    }

    /// Mark results as coming from cache
    pub fn into_cached(mut self) -> Self {
        self.from_cache = true;
        self
    }

    /// Get search duration in milliseconds
    pub fn duration_ms(&self) -> u64 {
        self.search_duration.as_millis() as u64
    }

    /// Check if results are empty
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Get unique file paths
    pub fn unique_files(&self) -> Vec<String> {
        let mut files: Vec<String> = self
            .results
            .iter()
            .map(|result| result.file_path.clone())
            .collect();
        files.sort();
        files.dedup();
        files
    }

    /// Calculate language distribution
    pub fn language_distribution(&self) -> std::collections::HashMap<ProgrammingLanguage, usize> {
        let mut distribution = std::collections::HashMap::new();

        for result in &self.results {
            if let Some(ref language) = result.language {
                *distribution.entry(language.clone()).or_insert(0) += 1;
            }
        }

        distribution
    }

    /// Get the top N results
    pub fn top_n(&self, n: usize) -> Vec<&SearchResult> {
        self.results.iter().take(n).collect()
    }

    /// Check if search was fast (under threshold)
    pub fn is_fast(&self, threshold_ms: u64) -> bool {
        self.search_duration.as_millis() <= threshold_ms as u128
    }

    /// Get results by type
    pub fn by_type(&self, result_type: SearchResultType) -> Vec<&SearchResult> {
        self.results
            .iter()
            .filter(|r| r.result_type == result_type)
            .collect()
    }

    /// Get results by language
    pub fn by_language(&self, language: ProgrammingLanguage) -> Vec<&SearchResult> {
        self.results
            .iter()
            .filter(|r| r.language.as_ref() == Some(&language))
            .collect()
    }

    /// Sort results by score (descending)
    pub fn sort_by_score(&mut self) {
        self.results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Take the top N results
    pub fn take(mut self, n: usize) -> Self {
        self.results.truncate(n);
        self.total_matches = self.results.len();
        self
    }
}

/// Index statistics and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    /// Total number of indexed documents/files
    pub total_documents: usize,
    /// Total size of the index in bytes
    pub index_size_bytes: u64,
    /// Number of supported programming languages
    pub supported_languages: usize,
    /// When the index was last updated
    pub last_updated: DateTime<Utc>,
    /// Total lines indexed
    pub total_lines: usize,
    /// Total symbols extracted
    pub total_symbols: usize,
}

impl IndexStats {
    /// Create new index stats
    pub fn new() -> Self {
        Self {
            total_documents: 0,
            index_size_bytes: 0,
            supported_languages: 0,
            last_updated: Utc::now(),
            total_lines: 0,
            total_symbols: 0,
        }
    }

    /// Get human-readable size
    pub fn human_readable_size(&self) -> String {
        let bytes = self.index_size_bytes as f64;
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];

        if bytes == 0.0 {
            return "0 B".to_string();
        }

        let base: f64 = 1024.0;
        let i = (bytes.ln() / base.ln()).floor() as usize;
        let i = i.min(UNITS.len() - 1);

        let size = bytes / base.powi(i as i32);

        if i == 0 {
            format!("{} {}", size as u64, UNITS[i])
        } else {
            format!("{:.1} {}", size, UNITS[i])
        }
    }

    /// Calculate average document size
    pub fn average_document_size(&self) -> f64 {
        if self.total_documents == 0 {
            0.0
        } else {
            self.index_size_bytes as f64 / self.total_documents as f64
        }
    }

    /// Calculate average lines per document
    pub fn average_lines_per_document(&self) -> f64 {
        if self.total_documents == 0 {
            0.0
        } else {
            self.total_lines as f64 / self.total_documents as f64
        }
    }

    /// Calculate average symbols per document
    pub fn average_symbols_per_document(&self) -> f64 {
        if self.total_documents == 0 {
            0.0
        } else {
            self.total_symbols as f64 / self.total_documents as f64
        }
    }
}

impl Default for IndexStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for indexing operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexingStats {
    /// Number of files successfully indexed
    pub files_indexed: usize,
    /// Number of files that failed to index
    pub files_failed: usize,
    /// Total lines indexed across all files
    pub lines_indexed: usize,
    /// Total symbols extracted
    pub symbols_extracted: usize,
    /// Time taken for indexing
    pub indexing_duration: Duration,
}

impl IndexingStats {
    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        let total_files = self.files_indexed + self.files_failed;
        if total_files == 0 {
            1.0
        } else {
            self.files_indexed as f64 / total_files as f64
        }
    }

    /// Calculate indexing speed (files per second)
    pub fn files_per_second(&self) -> f64 {
        let duration_secs = self.indexing_duration.as_secs_f64();
        if duration_secs > 0.0 {
            self.files_indexed as f64 / duration_secs
        } else {
            0.0
        }
    }

    /// Calculate lines per second
    pub fn lines_per_second(&self) -> f64 {
        let duration_secs = self.indexing_duration.as_secs_f64();
        if duration_secs > 0.0 {
            self.lines_indexed as f64 / duration_secs
        } else {
            0.0
        }
    }
}

/// Statistics for individual file indexing
#[derive(Debug, Clone, Default)]
pub struct FileIndexingStats {
    /// Number of lines indexed in this file
    pub lines_indexed: usize,
    /// Number of symbols extracted from this file
    pub symbols_extracted: usize,
}

/// Performance metrics for search operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchMetrics {
    /// Query processing time
    pub query_time_ms: u32,
    /// Index search time
    pub search_time_ms: u32,
    /// Result processing time
    pub processing_time_ms: u32,
    /// Total time including network/serialization
    pub total_time_ms: u32,
    /// Cache hit/miss status
    pub cache_hit: bool,
    /// Number of documents examined
    pub documents_examined: u32,
    /// Number of results returned
    pub results_returned: u32,
}

impl SearchMetrics {
    /// Check if search meets SLA requirements
    pub fn meets_sla(&self, sla_ms: u32) -> bool {
        self.total_time_ms <= sla_ms
    }

    /// Calculate search efficiency (results per ms)
    pub fn efficiency(&self) -> f64 {
        if self.total_time_ms > 0 {
            self.results_returned as f64 / self.total_time_ms as f64
        } else {
            0.0
        }
    }
}
