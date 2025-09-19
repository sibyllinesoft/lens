//! Real file indexing functionality
//!
//! This module provides the actual indexing implementation using Tantivy,
//! replacing any simulation or mock indexing code.

use anyhow::Result;
use ignore::WalkBuilder;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tantivy::IndexWriter;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::language::LanguageDetector;
use crate::SchemaFields;
use lens_common::results::{FileIndexingStats, IndexingStats};
use lens_common::ProgrammingLanguage;

/// Configuration for the indexer
#[derive(Debug, Clone)]
pub struct IndexerConfig {
    /// Maximum number of concurrent indexing tasks
    pub max_concurrent_files: usize,
    /// Buffer size for the indexing channel
    pub channel_buffer_size: usize,
    /// Maximum file size to index (bytes)
    pub max_file_size: usize,
    /// File extensions to index
    pub allowed_extensions: Vec<String>,
    /// Directories to ignore
    pub ignored_directories: Vec<String>,
    /// File patterns to ignore
    pub ignored_patterns: Vec<String>,
    /// Whether to follow symbolic links
    pub follow_symlinks: bool,
    /// Whether to respect .gitignore files
    pub respect_gitignore: bool,
}

impl Default for IndexerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_files: 10,
            channel_buffer_size: 1000,
            max_file_size: 10 * 1024 * 1024, // 10 MB
            allowed_extensions: vec![
                "rs".to_string(),
                "py".to_string(),
                "ts".to_string(),
                "js".to_string(),
                "go".to_string(),
                "java".to_string(),
                "cpp".to_string(),
                "c".to_string(),
                "h".to_string(),
                "hpp".to_string(),
                "rb".to_string(),
                "php".to_string(),
                "swift".to_string(),
                "kt".to_string(),
                "scala".to_string(),
                "clj".to_string(),
                "ex".to_string(),
                "exs".to_string(),
                "md".to_string(),
                "txt".to_string(),
            ],
            ignored_directories: vec![
                ".git".to_string(),
                "node_modules".to_string(),
                "target".to_string(),
                "dist".to_string(),
                "build".to_string(),
                "__pycache__".to_string(),
                ".pytest_cache".to_string(),
                "coverage".to_string(),
                "vendor".to_string(),
                ".venv".to_string(),
                "venv".to_string(),
                "env".to_string(),
            ],
            ignored_patterns: vec![
                "*.min.js".to_string(),
                "*.min.css".to_string(),
                "*.map".to_string(),
                "*.lock".to_string(),
                "package-lock.json".to_string(),
                "yarn.lock".to_string(),
                "Cargo.lock".to_string(),
                "*.log".to_string(),
                "*.tmp".to_string(),
                "*.temp".to_string(),
            ],
            follow_symlinks: false,
            respect_gitignore: true,
        }
    }
}

/// File to be indexed
#[derive(Debug, Clone)]
struct IndexingTask {
    path: PathBuf,
    language: ProgrammingLanguage,
    size: usize,
}

/// Result of indexing a single file
#[derive(Debug, Clone)]
struct IndexingResult {
    success: bool,
    stats: Option<FileIndexingStats>,
}

/// Real file indexer using Tantivy
pub struct Indexer {
    config: IndexerConfig,
    language_detector: Arc<LanguageDetector>,
    stats: Arc<RwLock<IndexingStats>>,
}

impl Indexer {
    /// Create a new indexer with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(IndexerConfig::default())
    }

    /// Create a new indexer with custom configuration
    pub fn with_config(config: IndexerConfig) -> Result<Self> {
        let language_detector = Arc::new(LanguageDetector::new()?);
        let stats = Arc::new(RwLock::new(IndexingStats::default()));

        Ok(Self {
            config,
            language_detector,
            stats,
        })
    }

    /// Index all files in a directory using real parallel processing
    pub async fn index_directory<P: AsRef<Path>>(
        &self,
        directory: P,
        writer: &IndexWriter,
        fields: &SchemaFields,
    ) -> Result<IndexingStats> {
        let start = Instant::now();
        let directory = directory.as_ref();

        info!("Starting real indexing of directory: {:?}", directory);

        // Reset stats
        {
            let mut stats = self.stats.write().await;
            *stats = IndexingStats::default();
        }

        // Discover files to index
        let files_to_index = self.discover_files(directory).await?;
        info!("Discovered {} files to index", files_to_index.len());

        if files_to_index.is_empty() {
            warn!("No files found to index in directory: {:?}", directory);
            return Ok(IndexingStats::default());
        }

        // Process files sequentially (IndexWriter doesn't support parallel writes)
        let _indexing_results = self
            .process_files_sequentially(files_to_index, writer, *fields)
            .await?;

        // Update final statistics
        let mut final_stats = self.stats.write().await;
        final_stats.indexing_duration = start.elapsed();

        // Log results
        info!("Indexing completed:");
        info!("  Files indexed: {}", final_stats.files_indexed);
        info!("  Files failed: {}", final_stats.files_failed);
        info!("  Lines indexed: {}", final_stats.lines_indexed);
        info!("  Symbols extracted: {}", final_stats.symbols_extracted);
        info!("  Duration: {:?}", final_stats.indexing_duration);
        info!("  Success rate: {:.1}%", final_stats.success_rate() * 100.0);
        info!("  Files/sec: {:.2}", final_stats.files_per_second());

        Ok(final_stats.clone())
    }

    /// Get current indexing statistics
    pub async fn get_stats(&self) -> IndexingStats {
        self.stats.read().await.clone()
    }

    /// Discover all files to index in a directory
    async fn discover_files<P: AsRef<Path>>(&self, directory: P) -> Result<Vec<IndexingTask>> {
        let directory = directory.as_ref();
        let mut files = Vec::new();

        // Build walker with ignore rules
        let mut builder = WalkBuilder::new(directory);
        builder
            .follow_links(self.config.follow_symlinks)
            .git_ignore(self.config.respect_gitignore)
            .max_filesize(Some(self.config.max_file_size as u64));

        // Add ignored directories
        for ignored_dir in &self.config.ignored_directories {
            builder.add_ignore(format!("/{}", ignored_dir));
        }

        let walker = builder.build();

        // Walk directory and collect files
        for entry in walker {
            match entry {
                Ok(entry) => {
                    if !entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                        continue;
                    }

                    let path = entry.path();
                    if self.should_index_file(path).await? {
                        let metadata = match tokio::fs::metadata(path).await {
                            Ok(metadata) => metadata,
                            Err(e) => {
                                debug!("Failed to get metadata for {:?}: {}", path, e);
                                continue;
                            }
                        };

                        let size = metadata.len() as usize;
                        if size > self.config.max_file_size {
                            debug!("Skipping large file: {:?} ({} bytes)", path, size);
                            continue;
                        }

                        // Detect language
                        let content_sample = match self.read_file_sample(path).await {
                            Ok(sample) => sample,
                            Err(e) => {
                                debug!("Failed to read file sample for {:?}: {}", path, e);
                                continue;
                            }
                        };

                        let language = match self
                            .language_detector
                            .detect_language(path, &content_sample)
                            .await
                        {
                            Ok(lang) => lang,
                            Err(e) => {
                                debug!("Failed to detect language for {:?}: {}", path, e);
                                ProgrammingLanguage::Unknown
                            }
                        };

                        files.push(IndexingTask {
                            path: path.to_path_buf(),
                            language,
                            size,
                        });
                    }
                }
                Err(e) => {
                    warn!("Error walking directory: {}", e);
                }
            }
        }

        // Sort by size (smaller files first for better load balancing)
        files.sort_by_key(|task| task.size);

        Ok(files)
    }

    /// Check if a file should be indexed
    async fn should_index_file<P: AsRef<Path>>(&self, path: P) -> Result<bool> {
        let path = path.as_ref();

        // Check if file extension is allowed
        if let Some(extension) = path.extension() {
            if let Some(ext_str) = extension.to_str() {
                if !self
                    .config
                    .allowed_extensions
                    .contains(&ext_str.to_string())
                {
                    return Ok(false);
                }
            } else {
                return Ok(false);
            }
        } else {
            return Ok(false);
        }

        // Check ignored patterns
        if let Some(file_name) = path.file_name() {
            if let Some(name_str) = file_name.to_str() {
                for pattern in &self.config.ignored_patterns {
                    if globset::Glob::new(pattern)?
                        .compile_matcher()
                        .is_match(name_str)
                    {
                        return Ok(false);
                    }
                }
            }
        }

        // Check if file is readable
        match tokio::fs::metadata(path).await {
            Ok(metadata) => Ok(metadata.is_file()),
            Err(_) => Ok(false),
        }
    }

    /// Read a small sample of the file for language detection
    async fn read_file_sample<P: AsRef<Path>>(&self, path: P) -> Result<String> {
        let content = tokio::fs::read_to_string(path).await?;
        // Take first 1KB for language detection
        if content.len() > 1024 {
            Ok(content.chars().take(1024).collect())
        } else {
            Ok(content)
        }
    }

    /// Process files sequentially (IndexWriter does not support parallel access)
    async fn process_files_sequentially(
        &self,
        files: Vec<IndexingTask>,
        writer: &IndexWriter,
        fields: SchemaFields,
    ) -> Result<Vec<IndexingResult>> {
        let total_files = files.len();
        let mut results = Vec::new();

        for (index, task) in files.into_iter().enumerate() {
            let result =
                Self::index_single_file(&task, writer, &fields, &self.language_detector).await;

            // Update stats
            {
                let mut stats_guard = self.stats.write().await;
                match &result {
                    IndexingResult {
                        success: true,
                        stats: Some(file_stats),
                        ..
                    } => {
                        stats_guard.files_indexed += 1;
                        stats_guard.lines_indexed += file_stats.lines_indexed;
                        stats_guard.symbols_extracted += file_stats.symbols_extracted;
                    }
                    IndexingResult { success: false, .. } => {
                        stats_guard.files_failed += 1;
                    }
                    _ => {}
                }
            }

            // Progress tracking could be added here with a callback field

            results.push(result);

            // Log progress
            if (index + 1) % 100 == 0 || (index + 1) == total_files {
                info!("Processed {}/{} files", index + 1, total_files);
            }
        }

        Ok(results)
    }

    /// Index a single file - real implementation
    async fn index_single_file(
        task: &IndexingTask,
        writer: &IndexWriter,
        fields: &SchemaFields,
        language_detector: &LanguageDetector,
    ) -> IndexingResult {
        let path = &task.path;

        // Read file content
        let content = match tokio::fs::read_to_string(path).await {
            Ok(content) => content,
            Err(e) => {
                error!("Failed to read file {:?}: {}", path, e);
                return IndexingResult {
                    success: false,
                    stats: None,
                };
            }
        };

        // Parse content for symbols and structure
        let parsed_content = match language_detector
            .parse_content(&content, &task.language)
            .await
        {
            Ok(parsed) => parsed,
            Err(e) => {
                debug!("Failed to parse content for {:?}: {}", path, e);
                return IndexingResult {
                    success: false,
                    stats: None,
                };
            }
        };

        let mut file_stats = FileIndexingStats::default();

        // Index each line with extracted metadata
        for (line_number, line_content) in content.lines().enumerate() {
            if line_content.trim().is_empty() {
                continue;
            }

            let mut doc = tantivy::doc!();

            // Add basic fields
            doc.add_text(fields.content, line_content);
            doc.add_text(fields.file_path, path.to_string_lossy().as_ref());
            doc.add_text(fields.language, task.language.to_string());
            doc.add_u64(fields.line_number, (line_number + 1) as u64);

            // Add extracted function name if present
            if let Some(function_name) = parsed_content.functions_at_line(line_number) {
                doc.add_text(fields.function_name, &function_name);
                file_stats.symbols_extracted += 1;
            }

            // Add extracted class name if present
            if let Some(class_name) = parsed_content.classes_at_line(line_number) {
                doc.add_text(fields.class_name, &class_name);
                file_stats.symbols_extracted += 1;
            }

            // Add import information if this is an import line
            if parsed_content.is_import_line(line_number) {
                doc.add_text(fields.imports, line_content);
            }

            // Add extracted symbols
            let symbols = parsed_content.symbols_at_line(line_number);
            if !symbols.is_empty() {
                doc.add_text(fields.symbols, symbols.join(" "));
                file_stats.symbols_extracted += symbols.len();
            }

            // Add document to index
            if let Err(e) = writer.add_document(doc) {
                error!(
                    "Failed to add document for {}:{}: {}",
                    path.display(),
                    line_number + 1,
                    e
                );
                return IndexingResult {
                    success: false,
                    stats: None,
                };
            }

            file_stats.lines_indexed += 1;
        }

        debug!(
            "Indexed file: {:?} ({} lines, {} symbols)",
            path, file_stats.lines_indexed, file_stats.symbols_extracted
        );

        IndexingResult {
            success: true,
            stats: Some(file_stats),
        }
    }

    /// Reset indexing statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = IndexingStats::default();
    }
}

/// Utility functions for indexing
impl Indexer {
    /// Estimate indexing time based on file count and sizes
    pub fn estimate_indexing_time(
        &self,
        file_count: usize,
        total_size_bytes: u64,
    ) -> std::time::Duration {
        // Rough estimates based on typical performance
        let files_per_second = 50.0; // Conservative estimate
        let bytes_per_second = 5_000_000.0; // 5 MB/s

        let time_by_files = file_count as f64 / files_per_second;
        let time_by_size = total_size_bytes as f64 / bytes_per_second;

        // Take the maximum of the two estimates
        let estimated_seconds = time_by_files.max(time_by_size);
        std::time::Duration::from_secs_f64(estimated_seconds)
    }

    /// Check if a directory is suitable for indexing
    pub async fn validate_directory<P: AsRef<Path>>(
        &self,
        directory: P,
    ) -> Result<DirectoryValidation> {
        let directory = directory.as_ref();

        if !directory.exists() {
            return Ok(DirectoryValidation {
                valid: false,
                issues: vec!["Directory does not exist".to_string()],
                estimated_files: 0,
                estimated_size: 0,
            });
        }

        if !directory.is_dir() {
            return Ok(DirectoryValidation {
                valid: false,
                issues: vec!["Path is not a directory".to_string()],
                estimated_files: 0,
                estimated_size: 0,
            });
        }

        let mut issues = Vec::new();
        let mut file_count = 0;
        let mut total_size = 0u64;

        // Quick scan to estimate work
        for entry in (std::fs::read_dir(directory)?).flatten() {
            if entry.file_type()?.is_file() {
                if let Ok(metadata) = entry.metadata() {
                    file_count += 1;
                    total_size += metadata.len();
                }
            }
        }

        if file_count == 0 {
            issues.push("No files found in directory".to_string());
        }

        if total_size > 1_000_000_000 {
            issues
                .push("Directory is very large (>1GB), indexing may take a long time".to_string());
        }

        Ok(DirectoryValidation {
            valid: issues.is_empty(),
            issues,
            estimated_files: file_count,
            estimated_size: total_size,
        })
    }
}

/// Result of directory validation
#[derive(Debug, Clone)]
pub struct DirectoryValidation {
    pub valid: bool,
    pub issues: Vec<String>,
    pub estimated_files: usize,
    pub estimated_size: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_indexer_creation() {
        let indexer = Indexer::new();
        assert!(indexer.is_ok());
    }

    #[tokio::test]
    async fn test_file_discovery() {
        let temp_dir = TempDir::new().unwrap();
        let indexer = Indexer::new().unwrap();

        // Create test files
        let rust_file = temp_dir.path().join("test.rs");
        tokio::fs::write(&rust_file, "fn main() {}").await.unwrap();

        let python_file = temp_dir.path().join("test.py");
        tokio::fs::write(&python_file, "def main(): pass")
            .await
            .unwrap();

        // Create ignored file
        let ignored_file = temp_dir.path().join("test.min.js");
        tokio::fs::write(&ignored_file, "var x=1;").await.unwrap();

        let files = indexer.discover_files(temp_dir.path()).await.unwrap();

        // Should find 2 files (rust and python), ignore the minified JS
        assert_eq!(files.len(), 2);

        let file_names: Vec<String> = files
            .iter()
            .map(|f| f.path.file_name().unwrap().to_string_lossy().to_string())
            .collect();

        assert!(file_names.contains(&"test.rs".to_string()));
        assert!(file_names.contains(&"test.py".to_string()));
        assert!(!file_names.contains(&"test.min.js".to_string()));
    }

    #[tokio::test]
    async fn test_directory_validation() {
        let temp_dir = TempDir::new().unwrap();
        let indexer = Indexer::new().unwrap();

        // Test valid directory
        let validation = indexer.validate_directory(temp_dir.path()).await.unwrap();
        assert!(
            validation.valid
                || validation
                    .issues
                    .contains(&"No files found in directory".to_string())
        );

        // Test non-existent directory
        let fake_path = temp_dir.path().join("non_existent");
        let validation = indexer.validate_directory(&fake_path).await.unwrap();
        assert!(!validation.valid);
        assert!(validation
            .issues
            .iter()
            .any(|issue| issue.contains("does not exist")));
    }
}
