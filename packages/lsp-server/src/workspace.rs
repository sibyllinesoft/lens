//! Workspace management for the LSP server
//!
//! This module handles real workspace operations, file tracking,
//! and indexing coordination, replacing any workspace simulation.

use anyhow::Result;
use globset::{Glob, GlobSet, GlobSetBuilder};
use lsp_types::*;
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Represents a workspace folder being managed by the LSP server
#[derive(Debug, Clone)]
pub struct WorkspaceFolder {
    pub uri: Url,
    pub name: String,
    pub indexed: bool,
    pub file_count: usize,
    pub last_indexed: Option<std::time::SystemTime>,
}

impl From<lsp_types::WorkspaceFolder> for WorkspaceFolder {
    fn from(folder: lsp_types::WorkspaceFolder) -> Self {
        Self {
            uri: folder.uri,
            name: folder.name,
            indexed: false,
            file_count: 0,
            last_indexed: None,
        }
    }
}

/// Represents a tracked file in the workspace
#[derive(Debug, Clone)]
pub struct TrackedFile {
    pub uri: Url,
    pub language_id: String,
    pub version: i32,
    pub indexed: bool,
    pub last_modified: Option<std::time::SystemTime>,
    pub size: usize,
}

/// Workspace management for the LSP server
#[derive(Debug)]
pub struct Workspace {
    /// Workspace folders being tracked
    pub folders: Vec<WorkspaceFolder>,
    /// Individual files being tracked
    pub files: HashMap<Url, TrackedFile>,
    /// Workspace configuration
    pub config: WorkspaceConfig,
}

/// Configuration for workspace management
#[derive(Debug)]
pub struct WorkspaceConfig {
    /// Whether to automatically index new folders
    pub auto_index: bool,
    /// Maximum number of files to track
    pub max_files: usize,
    /// File size limit for indexing (bytes)
    pub max_file_size: usize,
    /// Whether to watch for file changes
    pub watch_files: bool,
    /// Raw exclude patterns (for reference)
    pub exclude_patterns: Vec<String>,
    /// Compiled glob set for efficient pattern matching
    pub exclude_globset: GlobSet,
}

impl Default for WorkspaceConfig {
    fn default() -> Self {
        let exclude_patterns = vec![
            "**/node_modules/**".to_string(),
            "**/target/**".to_string(),
            "**/.git/**".to_string(),
            "**/dist/**".to_string(),
            "**/build/**".to_string(),
            "**/__pycache__/**".to_string(),
            "**/coverage/**".to_string(),
            "**/*.min.js".to_string(),
            "**/*.min.css".to_string(),
            "**/*.map".to_string(),
        ];

        // Build the compiled glob set
        let exclude_globset = Self::build_globset(&exclude_patterns).unwrap_or_else(|e| {
            warn!("Failed to build exclude glob set: {}, using empty set", e);
            GlobSet::empty()
        });

        Self {
            auto_index: true,
            max_files: 50000,
            max_file_size: 10 * 1024 * 1024, // 10 MB
            watch_files: true,
            exclude_patterns,
            exclude_globset,
        }
    }
}

impl WorkspaceConfig {
    /// Create a new config with custom exclude patterns
    pub fn with_exclude_patterns(mut self, patterns: Vec<String>) -> Result<Self> {
        self.exclude_patterns = patterns;
        self.exclude_globset = Self::build_globset(&self.exclude_patterns)?;
        Ok(self)
    }

    /// Build a GlobSet from patterns
    fn build_globset(patterns: &[String]) -> Result<GlobSet> {
        let mut builder = GlobSetBuilder::new();

        for pattern in patterns {
            match Glob::new(pattern) {
                Ok(glob) => {
                    builder.add(glob);
                }
                Err(e) => {
                    warn!("Invalid glob pattern '{}': {}", pattern, e);
                    // Continue with other patterns instead of failing completely
                }
            }
        }

        Ok(builder.build()?)
    }
}

impl Workspace {
    /// Create a new workspace manager
    pub fn new() -> Self {
        Self {
            folders: Vec::new(),
            files: HashMap::new(),
            config: WorkspaceConfig::default(),
        }
    }

    /// Create a workspace manager with custom configuration
    pub fn with_config(config: WorkspaceConfig) -> Self {
        Self {
            folders: Vec::new(),
            files: HashMap::new(),
            config,
        }
    }

    /// Add a workspace folder
    pub async fn add_folder(&mut self, folder: lsp_types::WorkspaceFolder) {
        info!("Adding workspace folder: {} ({})", folder.name, folder.uri);

        let workspace_folder = WorkspaceFolder::from(folder);
        self.folders.push(workspace_folder);
    }

    /// Add a root URI as a workspace folder
    pub async fn add_root_uri(&mut self, uri: Url) {
        let name = uri.path().rsplit('/').next().unwrap_or("root").to_string();
        let folder = lsp_types::WorkspaceFolder { name, uri };
        self.add_folder(folder).await;
    }

    /// Remove a workspace folder
    pub async fn remove_folder(&mut self, uri: &Url) -> bool {
        let initial_len = self.folders.len();
        self.folders.retain(|folder| &folder.uri != uri);

        // Also remove tracked files from this folder
        let folder_path = uri.path();
        self.files
            .retain(|file_uri, _| !file_uri.path().starts_with(folder_path));

        let removed = self.folders.len() < initial_len;
        if removed {
            info!("Removed workspace folder: {}", uri);
        }

        removed
    }

    /// Track a file in the workspace
    pub async fn track_file(&mut self, uri: Url, language_id: String, version: i32) -> Result<()> {
        debug!("Tracking file: {} ({})", uri, language_id);

        // Check if file is in a workspace folder
        let in_workspace = self
            .folders
            .iter()
            .any(|folder| uri.path().starts_with(folder.uri.path()));

        if !in_workspace {
            warn!("File not in any workspace folder: {}", uri);
        }

        // Get file metadata if possible
        let (size, last_modified) = if let Ok(path) = uri.to_file_path() {
            match tokio::fs::metadata(&path).await {
                Ok(metadata) => (metadata.len() as usize, metadata.modified().ok()),
                Err(_) => (0, None),
            }
        } else {
            (0, None)
        };

        // Check file size limit
        if size > self.config.max_file_size {
            warn!("File too large to track: {} ({} bytes)", uri, size);
            return Ok(());
        }

        // Check file count limit
        if self.files.len() >= self.config.max_files {
            warn!("Maximum file count reached, not tracking: {}", uri);
            return Ok(());
        }

        let tracked_file = TrackedFile {
            uri: uri.clone(),
            language_id,
            version,
            indexed: false,
            last_modified,
            size,
        };

        self.files.insert(uri, tracked_file);
        Ok(())
    }

    /// Untrack a file
    pub async fn untrack_file(&mut self, uri: &Url) -> bool {
        let removed = self.files.remove(uri).is_some();
        if removed {
            debug!("Untracked file: {}", uri);
        }
        removed
    }

    /// Update file version
    pub async fn update_file_version(&mut self, uri: &Url, version: i32) {
        if let Some(file) = self.files.get_mut(uri) {
            file.version = version;
            file.indexed = false; // Mark as needing re-indexing
            debug!("Updated file version: {} -> {}", uri, version);
        }
    }

    /// Mark a file as indexed
    pub async fn mark_file_indexed(&mut self, uri: &Url) {
        if let Some(file) = self.files.get_mut(uri) {
            file.indexed = true;
            debug!("Marked file as indexed: {}", uri);
        }
    }

    /// Mark a folder as indexed
    pub async fn mark_folder_indexed(&mut self, uri: &Url, file_count: usize) {
        for folder in &mut self.folders {
            if folder.uri == *uri {
                folder.indexed = true;
                folder.file_count = file_count;
                folder.last_indexed = Some(std::time::SystemTime::now());
                info!("Marked folder as indexed: {} ({} files)", uri, file_count);
                break;
            }
        }
    }

    /// Get files that need indexing
    pub async fn get_files_needing_indexing(&self) -> Vec<&TrackedFile> {
        self.files.values().filter(|file| !file.indexed).collect()
    }

    /// Get folders that need indexing
    pub async fn get_folders_needing_indexing(&self) -> Vec<&WorkspaceFolder> {
        self.folders
            .iter()
            .filter(|folder| !folder.indexed)
            .collect()
    }

    /// Check if a file should be indexed based on patterns
    pub fn should_index_file(&self, uri: &Url) -> bool {
        let path = uri.path();

        // Check exclude patterns using compiled glob set
        if self.config.exclude_globset.is_match(path) {
            return false;
        }

        // Check file extension
        if let Some(extension) = std::path::Path::new(path).extension() {
            if let Some(ext_str) = extension.to_str() {
                return is_indexable_extension(ext_str);
            }
        }

        false
    }

    /// Get workspace statistics
    pub async fn get_stats(&self) -> WorkspaceStats {
        let total_files = self.files.len();
        let indexed_files = self.files.values().filter(|f| f.indexed).count();
        let total_folders = self.folders.len();
        let indexed_folders = self.folders.iter().filter(|f| f.indexed).count();

        let total_size: usize = self.files.values().map(|f| f.size).sum();

        WorkspaceStats {
            total_folders,
            indexed_folders,
            total_files,
            indexed_files,
            total_size_bytes: total_size as u64,
            languages: self.get_language_distribution(),
        }
    }

    /// Get distribution of languages in the workspace
    fn get_language_distribution(&self) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();

        for file in self.files.values() {
            *distribution.entry(file.language_id.clone()).or_insert(0) += 1;
        }

        distribution
    }

    /// Find files by language
    pub async fn find_files_by_language(&self, language_id: &str) -> Vec<&TrackedFile> {
        self.files
            .values()
            .filter(|file| file.language_id == language_id)
            .collect()
    }

    /// Find files by pattern
    pub async fn find_files_by_pattern(&self, pattern: &str) -> Vec<&TrackedFile> {
        // Compile the pattern for this specific search
        match Glob::new(pattern) {
            Ok(glob) => {
                let matcher = glob.compile_matcher();
                self.files
                    .values()
                    .filter(|file| matcher.is_match(file.uri.path()))
                    .collect()
            }
            Err(e) => {
                warn!("Invalid glob pattern '{}': {}", pattern, e);
                Vec::new()
            }
        }
    }

    /// Get recently modified files
    pub async fn get_recently_modified_files(
        &self,
        since: std::time::SystemTime,
    ) -> Vec<&TrackedFile> {
        self.files
            .values()
            .filter(|file| {
                file.last_modified
                    .map(|modified| modified > since)
                    .unwrap_or(false)
            })
            .collect()
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new()
    }
}

/// Workspace statistics
#[derive(Debug, Clone)]
pub struct WorkspaceStats {
    pub total_folders: usize,
    pub indexed_folders: usize,
    pub total_files: usize,
    pub indexed_files: usize,
    pub total_size_bytes: u64,
    pub languages: HashMap<String, usize>,
}

impl WorkspaceStats {
    /// Get indexing progress as a percentage
    pub fn indexing_progress(&self) -> f64 {
        if self.total_files > 0 {
            (self.indexed_files as f64 / self.total_files as f64) * 100.0
        } else {
            100.0
        }
    }

    /// Get human-readable size
    pub fn human_readable_size(&self) -> String {
        let size = self.total_size_bytes as f64;

        if size < 1024.0 {
            format!("{:.0} B", size)
        } else if size < 1024.0 * 1024.0 {
            format!("{:.1} KB", size / 1024.0)
        } else if size < 1024.0 * 1024.0 * 1024.0 {
            format!("{:.1} MB", size / (1024.0 * 1024.0))
        } else {
            format!("{:.1} GB", size / (1024.0 * 1024.0 * 1024.0))
        }
    }
}

/// Check if file extension is indexable
fn is_indexable_extension(extension: &str) -> bool {
    matches!(
        extension,
        "rs" | "py"
            | "ts"
            | "js"
            | "go"
            | "java"
            | "cpp"
            | "c"
            | "h"
            | "hpp"
            | "rb"
            | "php"
            | "swift"
            | "kt"
            | "scala"
            | "clj"
            | "ex"
            | "exs"
            | "md"
            | "txt"
            | "json"
            | "yaml"
            | "yml"
            | "toml"
            | "xml"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_workspace_creation() {
        let workspace = Workspace::new();
        assert_eq!(workspace.folders.len(), 0);
        assert_eq!(workspace.files.len(), 0);
        assert!(workspace.config.auto_index);
    }

    #[tokio::test]
    async fn test_add_workspace_folder() {
        let mut workspace = Workspace::new();
        let uri = Url::parse("file:///test/workspace").unwrap();
        let folder = lsp_types::WorkspaceFolder {
            uri: uri.clone(),
            name: "test".to_string(),
        };

        workspace.add_folder(folder).await;
        assert_eq!(workspace.folders.len(), 1);
        assert_eq!(workspace.folders[0].uri, uri);
        assert_eq!(workspace.folders[0].name, "test");
    }

    #[tokio::test]
    async fn test_track_file() {
        let mut workspace = Workspace::new();
        let uri = Url::parse("file:///test/file.rs").unwrap();

        let result = workspace
            .track_file(uri.clone(), "rust".to_string(), 1)
            .await;
        assert!(result.is_ok());
        assert_eq!(workspace.files.len(), 1);
        assert!(workspace.files.contains_key(&uri));
    }

    #[test]
    fn test_glob_matching() {
        // Test the workspace's exclude patterns work correctly
        let workspace = Workspace::new();

        // Test that node_modules files are excluded
        let node_modules_file = "/path/to/node_modules/file.js";
        assert!(workspace.config.exclude_globset.is_match(node_modules_file));

        // Test that min.js files are excluded
        assert!(workspace.config.exclude_globset.is_match("app.min.js"));

        // Test that normal rust files are not excluded
        assert!(!workspace.config.exclude_globset.is_match("src/main.rs"));

        // Test individual glob patterns
        let glob = Glob::new("*.min.js").unwrap();
        let matcher = glob.compile_matcher();
        assert!(matcher.is_match("app.min.js"));
        assert!(!matcher.is_match("app.js"));

        let glob = Glob::new("exact_match").unwrap();
        let matcher = glob.compile_matcher();
        assert!(matcher.is_match("exact_match"));
        assert!(!matcher.is_match("other_match"));
    }

    #[test]
    fn test_indexable_extensions() {
        assert!(is_indexable_extension("rs"));
        assert!(is_indexable_extension("py"));
        assert!(is_indexable_extension("ts"));
        assert!(is_indexable_extension("js"));
        assert!(!is_indexable_extension("exe"));
        assert!(!is_indexable_extension("bin"));
    }

    #[test]
    fn test_should_index_file() {
        let workspace = Workspace::new();

        let rust_file = Url::parse("file:///src/main.rs").unwrap();
        assert!(workspace.should_index_file(&rust_file));

        let node_modules = Url::parse("file:///node_modules/package/index.js").unwrap();
        assert!(!workspace.should_index_file(&node_modules));

        let minified = Url::parse("file:///dist/app.min.js").unwrap();
        assert!(!workspace.should_index_file(&minified));
    }

    #[tokio::test]
    async fn test_workspace_stats() {
        let mut workspace = Workspace::new();

        // Add some test files
        let uri1 = Url::parse("file:///test/file1.rs").unwrap();
        let uri2 = Url::parse("file:///test/file2.py").unwrap();

        workspace
            .track_file(uri1.clone(), "rust".to_string(), 1)
            .await
            .unwrap();
        workspace
            .track_file(uri2.clone(), "python".to_string(), 1)
            .await
            .unwrap();
        workspace.mark_file_indexed(&uri1).await;

        let stats = workspace.get_stats().await;
        assert_eq!(stats.total_files, 2);
        assert_eq!(stats.indexed_files, 1);
        assert_eq!(stats.indexing_progress(), 50.0);
        assert_eq!(stats.languages.get("rust"), Some(&1));
        assert_eq!(stats.languages.get("python"), Some(&1));
    }
}
