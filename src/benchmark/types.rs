//! Type definitions for benchmark dataset management
//!
//! Provides Rust data structures compatible with the existing JSON-based
//! pinned dataset format described in CLAUDE.md

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use thiserror::Error;
use chrono::{DateTime, Utc};

/// Represents a single golden query for benchmarking
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GoldenQuery {
    /// The search query string
    pub query: String,
    
    /// Expected result file paths for this query
    pub expected_files: Vec<String>,
    
    /// Type/category of this query
    pub query_type: QueryType,
    
    /// Optional metadata for the query
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    
    /// Optional language context for the query
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    
    /// Optional confidence score for expected results
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
}

/// Query classification types based on the dataset
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum QueryType {
    /// Exact string match queries
    ExactMatch,
    
    /// Identifier/symbol queries
    Identifier,
    
    /// Structural code pattern queries
    Structural,
    
    /// Semantic/natural language queries
    Semantic,
    
    /// Cross-reference queries
    Reference,
    
    /// Definition lookup queries
    Definition,
}

impl Default for QueryType {
    fn default() -> Self {
        QueryType::Identifier
    }
}

/// Complete pinned dataset with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinnedDataset {
    /// Dataset metadata and versioning info
    pub metadata: DatasetMetadata,
    
    /// Collection of golden queries
    pub queries: Vec<GoldenQuery>,
    
    /// Dataset slices for different benchmark types
    #[serde(default)]
    pub slices: HashMap<String, Vec<usize>>, // slice_name -> query indices
    
    /// Corpus information for validation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub corpus_info: Option<CorpusInfo>,
}

/// Metadata for dataset versioning and tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Unique version identifier (e.g., "08653c1e-2025-09-01T21-51-35-302Z")
    pub version: String,
    
    /// Human-readable dataset name
    #[serde(default = "default_dataset_name")]
    pub name: String,
    
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    
    /// Total number of queries in dataset
    pub total_queries: usize,
    
    /// Query type distribution
    #[serde(default)]
    pub query_distribution: HashMap<QueryType, usize>,
    
    /// Languages covered in the dataset
    #[serde(default)]
    pub languages: Vec<String>,
    
    /// Git SHA for reproducibility (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_sha: Option<String>,
    
    /// Additional metadata
    #[serde(default)]
    pub additional_metadata: HashMap<String, serde_json::Value>,
}

fn default_dataset_name() -> String {
    "storyviz-pinned".to_string()
}

/// Information about the corpus for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusInfo {
    /// Total files in corpus
    pub total_files: usize,
    
    /// Corpus directory path
    pub corpus_path: String,
    
    /// File extensions included
    pub included_extensions: Vec<String>,
    
    /// Corpus statistics
    #[serde(default)]
    pub statistics: CorpusStatistics,
}

/// Detailed corpus statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CorpusStatistics {
    /// Total lines of code
    pub total_lines: usize,
    
    /// Total file size in bytes
    pub total_size_bytes: u64,
    
    /// Language distribution
    #[serde(default)]
    pub language_distribution: HashMap<String, usize>,
}

/// Dataset version identifier with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetVersion {
    /// Version string
    pub version: String,
    
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    
    /// Number of queries in this version
    pub query_count: usize,
    
    /// File path to dataset
    pub file_path: PathBuf,
    
    /// Whether this is the current active version
    pub is_current: bool,
}

/// Configuration for benchmark dataset operations
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Base path for dataset files
    pub dataset_path: String,
    
    /// Enable corpus consistency validation
    pub enable_corpus_validation: bool,
    
    /// Auto-discover available datasets
    pub auto_discover_datasets: bool,
    
    /// Default dataset version to load
    pub default_version: Option<String>,
    
    /// Timeout for dataset loading operations (seconds)
    pub loading_timeout_secs: u64,
    
    /// Cache loaded datasets in memory
    pub enable_caching: bool,
    
    /// Maximum cache size (number of datasets)
    pub max_cache_size: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            dataset_path: "./pinned-datasets".to_string(),
            enable_corpus_validation: true,
            auto_discover_datasets: true,
            default_version: Some(super::DEFAULT_PINNED_VERSION.to_string()),
            loading_timeout_secs: 30,
            enable_caching: true,
            max_cache_size: 5,
        }
    }
}

/// Result of dataset consistency validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the dataset is consistent with corpus
    pub is_consistent: bool,
    
    /// Number of valid queries (found in corpus)
    pub valid_queries: usize,
    
    /// Total queries checked
    pub total_queries: usize,
    
    /// Details of inconsistent queries
    pub inconsistent_queries: Vec<InconsistentQuery>,
    
    /// Validation timestamp
    pub validated_at: DateTime<Utc>,
}

/// Details about a query that failed corpus validation
#[derive(Debug, Clone)]
pub struct InconsistentQuery {
    /// Query that failed validation
    pub query: GoldenQuery,
    
    /// Reason for failure
    pub failure_reason: String,
    
    /// Expected files that were not found
    pub missing_files: Vec<String>,
}

/// Errors that can occur during dataset loading and validation
#[derive(Debug, Error)]
pub enum LoadingError {
    #[error("Dataset file not found: {path}")]
    DatasetNotFound { path: String },
    
    #[error("Failed to parse dataset JSON: {source}")]
    JsonParseError { source: serde_json::Error },
    
    #[error("IO error accessing dataset: {source}")]
    IoError { source: std::io::Error },
    
    #[error("Dataset version not found: {version}")]
    VersionNotFound { version: String },
    
    #[error("Corpus validation failed: {message}")]
    ValidationFailed { message: String },
    
    #[error("Dataset format is invalid: {reason}")]
    InvalidFormat { reason: String },
    
    #[error("Operation timeout: {operation}")]
    Timeout { operation: String },
    
    #[error("Cache error: {message}")]
    CacheError { message: String },
}

impl From<std::io::Error> for LoadingError {
    fn from(error: std::io::Error) -> Self {
        LoadingError::IoError { source: error }
    }
}

impl From<serde_json::Error> for LoadingError {
    fn from(error: serde_json::Error) -> Self {
        LoadingError::JsonParseError { source: error }
    }
}

/// Dataset slice configurations for different benchmark types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSlices {
    /// SMOKE test slice (subset for quick validation)
    #[serde(default)]
    pub smoke_default: Vec<usize>,
    
    /// All queries slice
    #[serde(default)]
    pub all: Vec<usize>,
    
    /// Language-specific slices
    #[serde(default)]
    pub by_language: HashMap<String, Vec<usize>>,
    
    /// Query type slices
    #[serde(default)]
    pub by_query_type: HashMap<QueryType, Vec<usize>>,
}

impl Default for DatasetSlices {
    fn default() -> Self {
        Self {
            smoke_default: vec![],
            all: vec![],
            by_language: HashMap::new(),
            by_query_type: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_golden_query_serialization() {
        let query = GoldenQuery {
            query: "function test".to_string(),
            expected_files: vec!["test.js".to_string()],
            query_type: QueryType::Identifier,
            metadata: HashMap::new(),
            language: Some("javascript".to_string()),
            confidence: Some(0.95),
        };

        let json = serde_json::to_string(&query).unwrap();
        let deserialized: GoldenQuery = serde_json::from_str(&json).unwrap();
        
        assert_eq!(query, deserialized);
    }

    #[test]
    fn test_query_type_variants() {
        let types = vec![
            QueryType::ExactMatch,
            QueryType::Identifier,
            QueryType::Structural,
            QueryType::Semantic,
            QueryType::Reference,
            QueryType::Definition,
        ];

        for query_type in types {
            let json = serde_json::to_string(&query_type).unwrap();
            let deserialized: QueryType = serde_json::from_str(&json).unwrap();
            assert_eq!(query_type, deserialized);
        }
    }

    #[test]
    fn test_pinned_dataset_structure() {
        let metadata = DatasetMetadata {
            version: "test-version".to_string(),
            name: "test-dataset".to_string(),
            created_at: Utc::now(),
            total_queries: 1,
            query_distribution: HashMap::new(),
            languages: vec!["rust".to_string()],
            git_sha: Some("abc123".to_string()),
            additional_metadata: HashMap::new(),
        };

        let query = GoldenQuery {
            query: "test query".to_string(),
            expected_files: vec!["test.rs".to_string()],
            query_type: QueryType::Identifier,
            metadata: HashMap::new(),
            language: None,
            confidence: None,
        };

        let dataset = PinnedDataset {
            metadata,
            queries: vec![query],
            slices: HashMap::new(),
            corpus_info: None,
        };

        // Should serialize and deserialize without errors
        let json = serde_json::to_string_pretty(&dataset).unwrap();
        let deserialized: PinnedDataset = serde_json::from_str(&json).unwrap();
        
        assert_eq!(dataset.queries.len(), deserialized.queries.len());
        assert_eq!(dataset.metadata.version, deserialized.metadata.version);
    }

    #[test]
    fn test_benchmark_config_defaults() {
        let config = BenchmarkConfig::default();
        
        assert!(!config.dataset_path.is_empty());
        assert!(config.enable_corpus_validation);
        assert!(config.auto_discover_datasets);
        assert_eq!(config.loading_timeout_secs, 30);
    }
}