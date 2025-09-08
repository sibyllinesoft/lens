//! # Corpus Manifest Generator
//! 
//! Creates reproducible corpus manifests with cryptographic hashes
//! for the replication pack as specified in TODO.md Step 2.
//!
//! Generates:
//! - (a) corpus manifest + hashes
//! - Content fingerprints for verification
//! - File integrity validation

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::{info, warn};

/// Complete corpus manifest with integrity verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusManifest {
    /// Manifest metadata
    pub metadata: ManifestMetadata,
    /// File entries with hashes
    pub files: Vec<FileEntry>,
    /// Directory structure
    pub directories: Vec<DirectoryEntry>,
    /// Corpus statistics
    pub statistics: CorpusStatistics,
    /// Manifest hash for integrity
    pub manifest_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestMetadata {
    pub version: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub corpus_name: String,
    pub corpus_version: String,
    pub generator_version: String,
    pub total_files: usize,
    pub total_size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    /// Relative path from corpus root
    pub relative_path: String,
    /// File size in bytes
    pub size_bytes: u64,
    /// SHA256 hash of file contents
    pub sha256_hash: String,
    /// File modification time
    pub modified_at: chrono::DateTime<chrono::Utc>,
    /// File type/language detected
    pub file_type: String,
    /// Line count for text files
    pub line_count: Option<usize>,
    /// Character count for text files
    pub char_count: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectoryEntry {
    pub relative_path: String,
    pub file_count: usize,
    pub total_size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusStatistics {
    pub total_files: usize,
    pub total_size_bytes: u64,
    pub total_lines: usize,
    pub total_characters: usize,
    pub by_language: HashMap<String, LanguageStats>,
    pub by_directory: HashMap<String, DirectoryStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageStats {
    pub file_count: usize,
    pub size_bytes: u64,
    pub line_count: usize,
    pub char_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectoryStats {
    pub file_count: usize,
    pub size_bytes: u64,
    pub subdirectories: usize,
}

/// Corpus manifest generator
pub struct CorpusManifestGenerator {
    corpus_root: PathBuf,
    include_patterns: Vec<String>,
    exclude_patterns: Vec<String>,
}

impl CorpusManifestGenerator {
    /// Create new manifest generator
    pub fn new(corpus_root: impl AsRef<Path>) -> Self {
        Self {
            corpus_root: corpus_root.as_ref().to_path_buf(),
            include_patterns: vec![
                "*.py".to_string(),
                "*.ts".to_string(),
                "*.js".to_string(),
                "*.rs".to_string(),
                "*.go".to_string(),
                "*.java".to_string(),
                "*.cpp".to_string(),
                "*.c".to_string(),
                "*.h".to_string(),
                "*.hpp".to_string(),
                "*.md".to_string(),
                "*.json".to_string(),
                "*.yaml".to_string(),
                "*.yml".to_string(),
                "*.toml".to_string(),
                "*.xml".to_string(),
                "*.html".to_string(),
                "*.css".to_string(),
                "*.sql".to_string(),
                "*.sh".to_string(),
                "*.txt".to_string(),
            ],
            exclude_patterns: vec![
                "node_modules/**".to_string(),
                "target/**".to_string(),
                ".git/**".to_string(),
                "dist/**".to_string(),
                "build/**".to_string(),
                "*.min.js".to_string(),
                "*.map".to_string(),
                "*.pyc".to_string(),
                "__pycache__/**".to_string(),
                ".venv/**".to_string(),
                "venv/**".to_string(),
                ".pytest_cache/**".to_string(),
                "coverage/**".to_string(),
                ".coverage".to_string(),
            ],
        }
    }

    /// Generate complete corpus manifest
    pub async fn generate_manifest(&self) -> Result<CorpusManifest> {
        info!("Generating corpus manifest for: {}", self.corpus_root.display());

        if !self.corpus_root.exists() {
            anyhow::bail!("Corpus root does not exist: {}", self.corpus_root.display());
        }

        let mut files = Vec::new();
        let mut directories = HashMap::new();
        let mut language_stats = HashMap::new();

        // Walk directory tree
        self.walk_directory(&self.corpus_root, &mut files, &mut directories, &mut language_stats).await?;

        info!("Found {} files across {} directories", files.len(), directories.len());

        // Calculate corpus statistics
        let statistics = self.calculate_statistics(&files, &directories, &language_stats);

        // Create directory entries
        let directory_entries: Vec<DirectoryEntry> = directories.into_iter()
            .map(|(path, stats)| DirectoryEntry {
                relative_path: path,
                file_count: stats.file_count,
                total_size_bytes: stats.size_bytes,
            })
            .collect();

        // Create manifest
        let mut manifest = CorpusManifest {
            metadata: ManifestMetadata {
                version: "1.0.0".to_string(),
                created_at: chrono::Utc::now(),
                corpus_name: self.corpus_root.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string(),
                corpus_version: self.get_corpus_version().await?,
                generator_version: env!("CARGO_PKG_VERSION").to_string(),
                total_files: files.len(),
                total_size_bytes: statistics.total_size_bytes,
            },
            files,
            directories: directory_entries,
            statistics,
            manifest_hash: String::new(), // Will be calculated after serialization
        };

        // Calculate manifest hash
        manifest.manifest_hash = self.calculate_manifest_hash(&manifest)?;

        Ok(manifest)
    }

    /// Walk directory tree and collect file information
    async fn walk_directory(
        &self,
        dir: &Path,
        files: &mut Vec<FileEntry>,
        directories: &mut HashMap<String, DirectoryStats>,
        language_stats: &mut HashMap<String, LanguageStats>,
    ) -> Result<()> {
        let mut entries = fs::read_dir(dir).await
            .with_context(|| format!("Failed to read directory: {}", dir.display()))?;

        let relative_dir = dir.strip_prefix(&self.corpus_root)
            .unwrap_or(Path::new(""))
            .to_string_lossy()
            .to_string();

        let mut dir_stats = DirectoryStats {
            file_count: 0,
            size_bytes: 0,
            subdirectories: 0,
        };

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            let metadata = entry.metadata().await?;

            if metadata.is_file() {
                if self.should_include_file(&path) {
                    let file_entry = self.create_file_entry(&path).await?;
                    
                    // Update directory stats
                    dir_stats.file_count += 1;
                    dir_stats.size_bytes += file_entry.size_bytes;
                    
                    // Update language stats
                    let lang_stats = language_stats.entry(file_entry.file_type.clone())
                        .or_insert_with(|| LanguageStats {
                            file_count: 0,
                            size_bytes: 0,
                            line_count: 0,
                            char_count: 0,
                        });
                    
                    lang_stats.file_count += 1;
                    lang_stats.size_bytes += file_entry.size_bytes;
                    if let Some(lines) = file_entry.line_count {
                        lang_stats.line_count += lines;
                    }
                    if let Some(chars) = file_entry.char_count {
                        lang_stats.char_count += chars;
                    }
                    
                    files.push(file_entry);
                }
            } else if metadata.is_dir() && !self.should_exclude_directory(&path) {
                dir_stats.subdirectories += 1;
                self.walk_directory(&path, files, directories, language_stats).await?;
            }
        }

        directories.insert(relative_dir, dir_stats);
        Ok(())
    }

    /// Create file entry with hash and metadata
    async fn create_file_entry(&self, path: &Path) -> Result<FileEntry> {
        let metadata = fs::metadata(path).await
            .with_context(|| format!("Failed to get metadata for: {}", path.display()))?;

        let relative_path = path.strip_prefix(&self.corpus_root)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();

        // Read file content for hash calculation
        let content = fs::read(path).await
            .with_context(|| format!("Failed to read file: {}", path.display()))?;

        // Calculate SHA256 hash
        let mut hasher = Sha256::new();
        hasher.update(&content);
        let sha256_hash = format!("{:x}", hasher.finalize());

        // Detect file type
        let file_type = self.detect_file_type(path);

        // Calculate line/char counts for text files
        let (line_count, char_count) = if self.is_text_file(&file_type) {
            let text = String::from_utf8_lossy(&content);
            let lines = text.lines().count();
            let chars = text.chars().count();
            (Some(lines), Some(chars))
        } else {
            (None, None)
        };

        Ok(FileEntry {
            relative_path,
            size_bytes: metadata.len(),
            sha256_hash,
            modified_at: chrono::DateTime::<chrono::Utc>::from(metadata.modified()?),
            file_type,
            line_count,
            char_count,
        })
    }

    /// Detect file type from extension
    fn detect_file_type(&self, path: &Path) -> String {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase())
            .unwrap_or_else(|| "unknown".to_string())
    }

    /// Check if file type is text-based
    fn is_text_file(&self, file_type: &str) -> bool {
        matches!(file_type, 
            "py" | "ts" | "js" | "rs" | "go" | "java" | "cpp" | "c" | "h" | "hpp" |
            "md" | "txt" | "json" | "yaml" | "yml" | "toml" | "xml" | "html" | 
            "css" | "sql" | "sh" | "rb" | "php" | "swift" | "kt"
        )
    }

    /// Check if file should be included
    fn should_include_file(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy().to_lowercase();
        
        // Check exclusions first
        for pattern in &self.exclude_patterns {
            if self.matches_pattern(&path_str, pattern) {
                return false;
            }
        }
        
        // Check inclusions
        for pattern in &self.include_patterns {
            if self.matches_pattern(&path_str, pattern) {
                return true;
            }
        }
        
        false
    }

    /// Check if directory should be excluded
    fn should_exclude_directory(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy().to_lowercase();
        
        for pattern in &self.exclude_patterns {
            if self.matches_pattern(&path_str, pattern) {
                return true;
            }
        }
        
        false
    }

    /// Simple pattern matching (supports * wildcards)
    fn matches_pattern(&self, text: &str, pattern: &str) -> bool {
        if pattern.contains('*') {
            let parts: Vec<&str> = pattern.split('*').collect();
            if parts.len() == 2 {
                text.starts_with(parts[0]) && text.ends_with(parts[1])
            } else {
                // Complex pattern - use simple contains check
                text.contains(&pattern.replace("*", ""))
            }
        } else {
            text.ends_with(pattern) || text.contains(pattern)
        }
    }

    /// Calculate comprehensive statistics
    fn calculate_statistics(
        &self,
        files: &[FileEntry],
        _directories: &HashMap<String, DirectoryStats>,
        language_stats: &HashMap<String, LanguageStats>,
    ) -> CorpusStatistics {
        let total_files = files.len();
        let total_size_bytes = files.iter().map(|f| f.size_bytes).sum();
        let total_lines = files.iter().map(|f| f.line_count.unwrap_or(0)).sum();
        let total_characters = files.iter().map(|f| f.char_count.unwrap_or(0)).sum();

        // Group by directory for directory stats
        let mut by_directory = HashMap::new();
        for file in files {
            let dir = Path::new(&file.relative_path)
                .parent()
                .unwrap_or(Path::new(""))
                .to_string_lossy()
                .to_string();
            
            let dir_stats = by_directory.entry(dir).or_insert_with(|| DirectoryStats {
                file_count: 0,
                size_bytes: 0,
                subdirectories: 0,
            });
            
            dir_stats.file_count += 1;
            dir_stats.size_bytes += file.size_bytes;
        }

        CorpusStatistics {
            total_files,
            total_size_bytes,
            total_lines,
            total_characters,
            by_language: language_stats.clone(),
            by_directory,
        }
    }

    /// Get corpus version (from git or timestamp)
    async fn get_corpus_version(&self) -> Result<String> {
        // Try to get git commit hash
        let git_dir = self.corpus_root.join(".git");
        if git_dir.exists() {
            if let Ok(head_content) = fs::read_to_string(git_dir.join("HEAD")).await {
                if head_content.starts_with("ref: ") {
                    let ref_path = head_content.trim_start_matches("ref: ").trim();
                    if let Ok(commit_hash) = fs::read_to_string(git_dir.join(ref_path)).await {
                        return Ok(commit_hash.trim().to_string());
                    }
                }
            }
        }
        
        // Fallback to timestamp
        Ok(chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string())
    }

    /// Calculate manifest hash for integrity
    fn calculate_manifest_hash(&self, manifest: &CorpusManifest) -> Result<String> {
        // Create a copy without the hash field
        let mut manifest_for_hash = manifest.clone();
        manifest_for_hash.manifest_hash = String::new();
        
        let json = serde_json::to_string(&manifest_for_hash)
            .context("Failed to serialize manifest for hashing")?;
        
        let mut hasher = Sha256::new();
        hasher.update(json.as_bytes());
        Ok(format!("{:x}", hasher.finalize()))
    }

    /// Save manifest to file
    pub async fn save_manifest(&self, manifest: &CorpusManifest, output_path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(manifest)
            .context("Failed to serialize manifest")?;
        
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).await
                .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
        }
        
        fs::write(output_path, json).await
            .with_context(|| format!("Failed to write manifest: {}", output_path.display()))?;
        
        info!("Corpus manifest saved to: {}", output_path.display());
        Ok(())
    }

    /// Verify corpus against manifest
    pub async fn verify_corpus(&self, manifest: &CorpusManifest) -> Result<VerificationResult> {
        info!("Verifying corpus against manifest");
        
        let mut verification = VerificationResult {
            total_files_checked: 0,
            hash_matches: 0,
            hash_mismatches: Vec::new(),
            missing_files: Vec::new(),
            extra_files: Vec::new(),
            verification_passed: false,
        };

        // Check each file in manifest
        for file_entry in &manifest.files {
            let file_path = self.corpus_root.join(&file_entry.relative_path);
            verification.total_files_checked += 1;
            
            if !file_path.exists() {
                verification.missing_files.push(file_entry.relative_path.clone());
                continue;
            }
            
            // Calculate current hash
            let content = fs::read(&file_path).await
                .with_context(|| format!("Failed to read file: {}", file_path.display()))?;
            
            let mut hasher = Sha256::new();
            hasher.update(&content);
            let current_hash = format!("{:x}", hasher.finalize());
            
            if current_hash == file_entry.sha256_hash {
                verification.hash_matches += 1;
            } else {
                verification.hash_mismatches.push(HashMismatch {
                    file_path: file_entry.relative_path.clone(),
                    expected_hash: file_entry.sha256_hash.clone(),
                    actual_hash: current_hash,
                });
            }
        }

        verification.verification_passed = verification.hash_mismatches.is_empty() 
            && verification.missing_files.is_empty();

        if verification.verification_passed {
            info!("Corpus verification PASSED: {}/{} files verified", 
                  verification.hash_matches, verification.total_files_checked);
        } else {
            warn!("Corpus verification FAILED: {} hash mismatches, {} missing files",
                  verification.hash_mismatches.len(), verification.missing_files.len());
        }

        Ok(verification)
    }
}

/// Verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub total_files_checked: usize,
    pub hash_matches: usize,
    pub hash_mismatches: Vec<HashMismatch>,
    pub missing_files: Vec<String>,
    pub extra_files: Vec<String>,
    pub verification_passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashMismatch {
    pub file_path: String,
    pub expected_hash: String,
    pub actual_hash: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_manifest_generation() {
        let temp_dir = TempDir::new().unwrap();
        let corpus_root = temp_dir.path();
        
        // Create test files
        let test_file = corpus_root.join("test.py");
        fs::write(&test_file, "print('hello world')").await.unwrap();
        
        let generator = CorpusManifestGenerator::new(corpus_root);
        let manifest = generator.generate_manifest().await.unwrap();
        
        assert_eq!(manifest.files.len(), 1);
        assert_eq!(manifest.files[0].relative_path, "test.py");
        assert_eq!(manifest.files[0].file_type, "py");
        assert!(!manifest.manifest_hash.is_empty());
    }

    #[tokio::test]
    async fn test_corpus_verification() {
        let temp_dir = TempDir::new().unwrap();
        let corpus_root = temp_dir.path();
        
        // Create test file
        let test_file = corpus_root.join("test.py");
        fs::write(&test_file, "print('hello world')").await.unwrap();
        
        let generator = CorpusManifestGenerator::new(corpus_root);
        let manifest = generator.generate_manifest().await.unwrap();
        
        // Verify against unchanged corpus
        let result = generator.verify_corpus(&manifest).await.unwrap();
        assert!(result.verification_passed);
        assert_eq!(result.hash_matches, 1);
        assert!(result.hash_mismatches.is_empty());
    }

    #[tokio::test]
    async fn test_hash_mismatch_detection() {
        let temp_dir = TempDir::new().unwrap();
        let corpus_root = temp_dir.path();
        
        // Create test file
        let test_file = corpus_root.join("test.py");
        fs::write(&test_file, "print('hello world')").await.unwrap();
        
        let generator = CorpusManifestGenerator::new(corpus_root);
        let manifest = generator.generate_manifest().await.unwrap();
        
        // Modify file
        fs::write(&test_file, "print('modified content')").await.unwrap();
        
        // Verify should detect mismatch
        let result = generator.verify_corpus(&manifest).await.unwrap();
        assert!(!result.verification_passed);
        assert_eq!(result.hash_mismatches.len(), 1);
    }
}