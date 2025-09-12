//! LSP Hint Caching System
//!
//! Implements 24h TTL caching with invalidation for LSP results
//! Features:
//! - Time-based expiration (24h default)
//! - Size-based LRU eviction
//! - File change invalidation
//! - Concurrent access with dashmap
//! - Metrics tracking

use super::{LspSearchResult, LspServerType};
use anyhow::Result;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tokio::time::interval;
use tracing::{debug, info, warn};

/// Type of LSP hint for categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HintType {
    Definition,
    References,
    TypeDefinition,
    Implementation,
    Declaration,
    Symbol,
    Hover,
    Completion,
}

impl HintType {
    pub fn as_str(&self) -> &'static str {
        match self {
            HintType::Definition => "definition",
            HintType::References => "references",
            HintType::TypeDefinition => "type_definition",
            HintType::Implementation => "implementation",
            HintType::Declaration => "declaration",
            HintType::Symbol => "symbol",
            HintType::Hover => "hover",
            HintType::Completion => "completion",
        }
    }
}

/// Cached LSP hint entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedHint {
    pub results: Vec<LspSearchResult>,
    pub created_at: u64,
    pub expires_at: u64,
    pub access_count: u64,
    pub last_accessed: u64,
    pub cache_key: String,
}

impl CachedHint {
    pub fn new(results: Vec<LspSearchResult>, ttl_seconds: u64, cache_key: String) -> Self {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        Self {
            results,
            created_at: now,
            expires_at: now + ttl_seconds,
            access_count: 0,
            last_accessed: now,
            cache_key,
        }
    }

    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        now >= self.expires_at
    }

    pub fn touch(&mut self) {
        self.access_count += 1;
        self.last_accessed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    }

    pub fn age_seconds(&self) -> u64 {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        now.saturating_sub(self.created_at)
    }
}

/// File modification tracking for cache invalidation
#[derive(Debug, Clone)]
pub struct FileTracker {
    pub path: PathBuf,
    pub last_modified: SystemTime,
    pub size: u64,
}

impl FileTracker {
    pub async fn new(path: PathBuf) -> Result<Self> {
        let metadata = tokio::fs::metadata(&path).await?;
        Ok(Self {
            path,
            last_modified: metadata.modified()?,
            size: metadata.len(),
        })
    }

    pub async fn has_changed(&self) -> bool {
        match tokio::fs::metadata(&self.path).await {
            Ok(metadata) => {
                let new_modified = metadata.modified().unwrap_or(UNIX_EPOCH);
                let new_size = metadata.len();
                new_modified != self.last_modified || new_size != self.size
            }
            Err(_) => true, // File doesn't exist anymore, consider it changed
        }
    }
}

/// Cache statistics
#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub invalidations: u64,
    pub size: usize,
    pub memory_usage_bytes: u64,
    pub avg_lookup_time_ns: u64,
    pub avg_storage_time_ns: u64,
    pub hit_rate: f64,
}

impl CacheStats {
    pub fn update_hit_rate(&mut self) {
        let total_requests = self.hits + self.misses;
        if total_requests > 0 {
            self.hit_rate = self.hits as f64 / total_requests as f64;
        }
    }
    
    /// Check if cache meets TODO.md performance targets: ≤+1ms p95 overhead
    pub fn meets_performance_targets(&self) -> bool {
        const MAX_LOOKUP_TIME_NS: u64 = 1_000_000; // 1ms in nanoseconds
        const MIN_HIT_RATE: f64 = 0.7; // 70% minimum hit rate for effectiveness
        
        self.avg_lookup_time_ns <= MAX_LOOKUP_TIME_NS && self.hit_rate >= MIN_HIT_RATE
    }
}

impl CacheStats {
    /// Calculate current hit rate
    pub fn calculate_hit_rate(&self) -> f64 {
        if self.hits + self.misses == 0 {
            0.0
        } else {
            self.hits as f64 / (self.hits + self.misses) as f64
        }
    }
    
    /// Update timing metrics with exponential moving average
    pub fn update_lookup_timing(&mut self, lookup_time_ns: u64) {
        const ALPHA: f64 = 0.1; // EMA smoothing factor
        if self.avg_lookup_time_ns == 0 {
            self.avg_lookup_time_ns = lookup_time_ns;
        } else {
            self.avg_lookup_time_ns = ((1.0 - ALPHA) * self.avg_lookup_time_ns as f64 
                                    + ALPHA * lookup_time_ns as f64) as u64;
        }
    }
    
    /// Update storage timing metrics
    pub fn update_storage_timing(&mut self, storage_time_ns: u64) {
        const ALPHA: f64 = 0.1; // EMA smoothing factor
        if self.avg_storage_time_ns == 0 {
            self.avg_storage_time_ns = storage_time_ns;
        } else {
            self.avg_storage_time_ns = ((1.0 - ALPHA) * self.avg_storage_time_ns as f64 
                                      + ALPHA * storage_time_ns as f64) as u64;
        }
    }
}

/// High-performance LSP hint cache
pub struct HintCache {
    // Main cache storage
    cache: Arc<DashMap<String, CachedHint>>,
    
    // File modification tracking
    file_trackers: Arc<RwLock<DashMap<PathBuf, FileTracker>>>,
    
    // Configuration
    max_size: usize,
    default_ttl_seconds: u64,
    
    // Statistics
    stats: Arc<RwLock<CacheStats>>,
    
    // Background tasks
    cleanup_handle: Option<tokio::task::JoinHandle<()>>,
    invalidation_handle: Option<tokio::task::JoinHandle<()>>,
}

impl HintCache {
    /// Create a new hint cache
    pub async fn new(ttl_hours: u64) -> Result<Self> {
        let cache = Arc::new(DashMap::new());
        let file_trackers = Arc::new(RwLock::new(DashMap::new()));
        let stats = Arc::new(RwLock::new(CacheStats::default()));
        
        let mut hint_cache = Self {
            cache: cache.clone(),
            file_trackers,
            max_size: 10000, // Default max entries
            default_ttl_seconds: ttl_hours * 3600,
            stats,
            cleanup_handle: None,
            invalidation_handle: None,
        };

        // Start background cleanup task
        hint_cache.start_cleanup_task().await;
        
        // Start file invalidation task
        hint_cache.start_invalidation_task().await;
        
        info!("LSP hint cache initialized with {}h TTL", ttl_hours);
        Ok(hint_cache)
    }

    /// Get cached hints with performance timing
    pub async fn get(&self, key: &str) -> Result<Option<Vec<LspSearchResult>>> {
        let start_time = std::time::Instant::now();
        let result = self.get_internal(key).await;
        let elapsed_ns = start_time.elapsed().as_nanos() as u64;
        
        // Update timing statistics
        {
            let mut stats = self.stats.write().await;
            stats.update_lookup_timing(elapsed_ns);
            stats.hit_rate = stats.calculate_hit_rate();
        }
        
        result
    }
    
    /// Internal get method without timing overhead
    async fn get_internal(&self, key: &str) -> Result<Option<Vec<LspSearchResult>>> {
        let mut stats = self.stats.write().await;
        
        match self.cache.get_mut(key) {
            Some(mut entry) => {
                if entry.is_expired() {
                    // Remove expired entry
                    drop(entry);
                    self.cache.remove(key);
                    stats.misses += 1;
                    debug!("Cache expired for key: {}", key);
                    Ok(None)
                } else {
                    // Update access info and return results
                    entry.touch();
                    let results = entry.results.clone();
                    stats.hits += 1;
                    debug!("Cache hit for key: {}", key);
                    Ok(Some(results))
                }
            }
            None => {
                stats.misses += 1;
                debug!("Cache miss for key: {}", key);
                Ok(None)
            }
        }
    }

    /// Set cached hints with performance timing
    pub async fn set(&self, key: String, results: Vec<LspSearchResult>, ttl_seconds: u64) -> Result<()> {
        let start_time = std::time::Instant::now();
        let result = self.set_internal(key, results, ttl_seconds).await;
        let elapsed_ns = start_time.elapsed().as_nanos() as u64;
        
        // Update timing statistics
        {
            let mut stats = self.stats.write().await;
            stats.update_storage_timing(elapsed_ns);
        }
        
        result
    }
    
    /// Internal set method without timing overhead
    async fn set_internal(&self, key: String, results: Vec<LspSearchResult>, ttl_seconds: u64) -> Result<()> {
        // Check size limits
        if self.cache.len() >= self.max_size {
            self.evict_lru().await;
        }

        let cached_hint = CachedHint::new(results, ttl_seconds, key.clone());
        
        // Track files mentioned in results for invalidation
        self.track_files_from_results(&cached_hint.results).await;
        
        self.cache.insert(key.clone(), cached_hint);
        
        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.size = self.cache.len();
            stats.memory_usage_bytes = self.estimate_memory_usage();
        }
        
        debug!("Cached hints for key: {}", key);
        Ok(())
    }

    /// Invalidate cache entries for a specific file
    pub async fn invalidate_file(&self, file_path: &PathBuf) -> Result<u64> {
        let mut invalidated = 0;
        let file_path_str = file_path.to_string_lossy();
        
        // Remove entries that reference this file
        self.cache.retain(|_key, hint| {
            let should_keep = !hint.results.iter().any(|result| {
                result.file_path.contains(&*file_path_str)
            });
            
            if !should_keep {
                invalidated += 1;
            }
            
            should_keep
        });

        // Update stats
        if invalidated > 0 {
            let mut stats = self.stats.write().await;
            stats.invalidations += invalidated;
            stats.size = self.cache.len();
            debug!("Invalidated {} cache entries for file: {:?}", invalidated, file_path);
        }

        Ok(invalidated)
    }

    /// Clear all cache entries
    pub async fn clear(&self) -> Result<()> {
        let size = self.cache.len();
        self.cache.clear();
        
        {
            let mut stats = self.stats.write().await;
            stats.size = 0;
            stats.memory_usage_bytes = 0;
        }
        
        info!("Cleared {} cache entries", size);
        Ok(())
    }

    /// Get cache statistics
    pub async fn stats(&self) -> CacheStats {
        self.stats.read().await.clone()
    }

    /// Track files from search results for invalidation
    async fn track_files_from_results(&self, results: &[LspSearchResult]) {
        let trackers = self.file_trackers.read().await;
        
        for result in results {
            let path = PathBuf::from(&result.file_path);
            
            if !trackers.contains_key(&path) {
                if let Ok(tracker) = FileTracker::new(path.clone()).await {
                    drop(trackers); // Release read lock
                    let mut trackers_write = self.file_trackers.write().await;
                    trackers_write.insert(path, tracker);
                    return; // Re-acquire read lock on next iteration
                }
            }
        }
    }

    /// Evict least recently used entries
    async fn evict_lru(&self) {
        if self.cache.is_empty() {
            return;
        }

        // Find oldest entry by last_accessed time
        let mut oldest_key: Option<String> = None;
        let mut oldest_time = u64::MAX;

        for entry in self.cache.iter() {
            if entry.value().last_accessed < oldest_time {
                oldest_time = entry.value().last_accessed;
                oldest_key = Some(entry.key().clone());
            }
        }

        if let Some(key) = oldest_key {
            self.cache.remove(&key);
            
            let mut stats = self.stats.write().await;
            stats.evictions += 1;
            stats.size = self.cache.len();
            
            debug!("Evicted LRU cache entry: {}", key);
        }
    }

    /// Start background cleanup task for expired entries
    async fn start_cleanup_task(&mut self) {
        let cache = self.cache.clone();
        let stats = self.stats.clone();
        
        let handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(300)); // 5 minutes
            
            loop {
                interval.tick().await;
                
                let mut expired_keys = Vec::new();
                
                // Find expired entries
                for entry in cache.iter() {
                    if entry.value().is_expired() {
                        expired_keys.push(entry.key().clone());
                    }
                }
                
                // Remove expired entries
                let mut removed_count = 0;
                for key in expired_keys {
                    if cache.remove(&key).is_some() {
                        removed_count += 1;
                    }
                }
                
                if removed_count > 0 {
                    let mut stats_lock = stats.write().await;
                    stats_lock.size = cache.len();
                    debug!("Cleaned up {} expired cache entries", removed_count);
                }
            }
        });
        
        self.cleanup_handle = Some(handle);
    }

    /// Start background file invalidation task
    async fn start_invalidation_task(&mut self) {
        let cache = self.cache.clone();
        let file_trackers = self.file_trackers.clone();
        let stats = self.stats.clone();
        
        let handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60)); // 1 minute
            
            loop {
                interval.tick().await;
                
                let trackers = file_trackers.read().await;
                let mut changed_files = Vec::new();
                
                // Check for file changes
                for tracker in trackers.iter() {
                    if tracker.value().has_changed().await {
                        changed_files.push(tracker.key().clone());
                    }
                }
                
                if !changed_files.is_empty() {
                    drop(trackers); // Release read lock
                    
                    let mut total_invalidated = 0;
                    
                    for file_path in changed_files {
                        let file_path_str = file_path.to_string_lossy();
                        
                        // Invalidate cache entries for this file
                        cache.retain(|_key, hint| {
                            let should_keep = !hint.results.iter().any(|result| {
                                result.file_path.contains(&*file_path_str)
                            });
                            
                            if !should_keep {
                                total_invalidated += 1;
                            }
                            
                            should_keep
                        });
                        
                        // Update file tracker
                        if let Ok(new_tracker) = FileTracker::new(file_path.clone()).await {
                            let mut trackers_write = file_trackers.write().await;
                            trackers_write.insert(file_path, new_tracker);
                        }
                    }
                    
                    if total_invalidated > 0 {
                        let mut stats_lock = stats.write().await;
                        stats_lock.invalidations += total_invalidated;
                        stats_lock.size = cache.len();
                        
                        debug!("Invalidated {} cache entries due to file changes", total_invalidated);
                    }
                }
            }
        });
        
        self.invalidation_handle = Some(handle);
    }

    /// Estimate memory usage (rough calculation)
    fn estimate_memory_usage(&self) -> u64 {
        // Rough estimate: each cache entry ~1KB
        self.cache.len() as u64 * 1024
    }

    /// Shutdown the cache and background tasks
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down LSP hint cache");
        
        if let Some(handle) = &self.cleanup_handle {
            handle.abort();
        }
        
        if let Some(handle) = &self.invalidation_handle {
            handle.abort();
        }
        
        self.cache.clear();
        
        Ok(())
    }
}

impl Drop for HintCache {
    fn drop(&mut self) {
        debug!("LSP hint cache dropped");
    }
}

/// Symbol hint for caching and quick lookups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolHint {
    pub symbol_name: String,
    pub symbol_kind: String,
    pub file_path: String,
    pub line_number: u32,
    pub column: u32,
    pub documentation: Option<String>,
    pub signature: Option<String>,
    pub confidence: f64,
}

impl SymbolHint {
    pub fn new(
        symbol_name: String,
        symbol_kind: String,
        file_path: String,
        line_number: u32,
        column: u32,
        confidence: f64,
    ) -> Self {
        Self {
            symbol_name,
            symbol_kind,
            file_path,
            line_number,
            column,
            documentation: None,
            signature: None,
            confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_cache_basic_operations() {
        let cache = HintCache::new(1).await.unwrap(); // 1 hour TTL
        let key = "test_key".to_string();
        
        // Test miss
        assert!(cache.get(&key).await.unwrap().is_none());
        
        // Test set and get
        let results = vec![LspSearchResult {
            file_path: "/test/file.rs".to_string(),
            line_number: 10,
            column: 5,
            content: "test content".to_string(),
            hint_type: HintType::Definition,
            server_type: LspServerType::Rust,
            confidence: 0.9,
            context_lines: None,
        }];
        
        cache.set(key.clone(), results.clone(), 3600).await.unwrap();
        
        let cached = cache.get(&key).await.unwrap().unwrap();
        assert_eq!(cached.len(), 1);
        assert_eq!(cached[0].file_path, results[0].file_path);
        
        // Test stats
        let stats = cache.stats().await;
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.size, 1);
    }

    #[tokio::test] 
    async fn test_cache_expiration() {
        let cache = HintCache::new(1).await.unwrap();
        let key = "expiry_test".to_string();
        
        let results = vec![LspSearchResult {
            file_path: "/test/file.rs".to_string(),
            line_number: 10,
            column: 5,
            content: "test content".to_string(),
            hint_type: HintType::Definition,
            server_type: LspServerType::Rust,
            confidence: 0.9,
            context_lines: None,
        }];
        
        // Set with 1 second TTL
        cache.set(key.clone(), results, 1).await.unwrap();
        
        // Should be available immediately
        assert!(cache.get(&key).await.unwrap().is_some());
        
        // Wait for expiration
        sleep(Duration::from_secs(2)).await;
        
        // Should be expired now
        assert!(cache.get(&key).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_lru_eviction() {
        let mut cache = HintCache::new(1).await.unwrap();
        cache.max_size = 2; // Set small limit for testing
        
        let results = vec![LspSearchResult {
            file_path: "/test/file.rs".to_string(),
            line_number: 10,
            column: 5,
            content: "test content".to_string(),
            hint_type: HintType::Definition,
            server_type: LspServerType::Rust,
            confidence: 0.9,
            context_lines: None,
        }];
        
        // Fill cache to capacity
        cache.set("key1".to_string(), results.clone(), 3600).await.unwrap();
        sleep(Duration::from_millis(100)).await; // Ensure different timestamps
        cache.set("key2".to_string(), results.clone(), 3600).await.unwrap();
        
        // Access key1 to make key2 the LRU
        sleep(Duration::from_millis(100)).await; // Ensure different timestamps
        let _val = cache.get("key1").await.unwrap();
        assert!(_val.is_some()); // key1 should exist and be accessed
        
        // Add another entry, should evict key2
        sleep(Duration::from_millis(100)).await; // Ensure different timestamps
        cache.set("key3".to_string(), results, 3600).await.unwrap();
        
        // key1 and key3 should exist, key2 should be evicted
        assert!(cache.get("key1").await.unwrap().is_some());
        assert!(cache.get("key2").await.unwrap().is_none());
        assert!(cache.get("key3").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_hint_type_as_str() {
        assert_eq!(HintType::Definition.as_str(), "definition");
        assert_eq!(HintType::References.as_str(), "references");
        assert_eq!(HintType::TypeDefinition.as_str(), "type_definition");
        assert_eq!(HintType::Implementation.as_str(), "implementation");
        assert_eq!(HintType::Declaration.as_str(), "declaration");
        assert_eq!(HintType::Symbol.as_str(), "symbol");
        assert_eq!(HintType::Hover.as_str(), "hover");
        assert_eq!(HintType::Completion.as_str(), "completion");
    }

    #[tokio::test]
    async fn test_hint_type_equality() {
        assert_eq!(HintType::Definition, HintType::Definition);
        assert_ne!(HintType::Definition, HintType::References);
    }

    #[tokio::test]
    async fn test_cached_hint_creation() {
        let results = vec![LspSearchResult {
            file_path: "/test/file.rs".to_string(),
            line_number: 10,
            column: 5,
            content: "test content".to_string(),
            hint_type: HintType::Definition,
            server_type: LspServerType::Rust,
            confidence: 0.9,
            context_lines: None,
        }];
        
        let cache_key = "test_key".to_string();
        let ttl_seconds = 3600;
        
        let cached_hint = CachedHint::new(results.clone(), ttl_seconds, cache_key.clone());
        
        assert_eq!(cached_hint.results.len(), 1);
        assert_eq!(cached_hint.cache_key, cache_key);
        assert_eq!(cached_hint.access_count, 0);
        assert!(!cached_hint.is_expired());
        assert!(cached_hint.age_seconds() < 5); // Should be very recent
    }

    #[tokio::test]
    async fn test_cached_hint_expiration() {
        let results = vec![];
        let cache_key = "test_key".to_string();
        
        // Create hint that expires immediately
        let mut cached_hint = CachedHint::new(results, 0, cache_key);
        cached_hint.expires_at = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() - 1;
        
        assert!(cached_hint.is_expired());
    }

    #[tokio::test]
    async fn test_cached_hint_touch() {
        let results = vec![];
        let cache_key = "test_key".to_string();
        let mut cached_hint = CachedHint::new(results, 3600, cache_key);
        
        let initial_access_count = cached_hint.access_count;
        let initial_last_accessed = cached_hint.last_accessed;
        
        // Small delay to ensure different timestamp
        sleep(Duration::from_millis(10)).await;
        cached_hint.touch();
        
        assert_eq!(cached_hint.access_count, initial_access_count + 1);
        assert!(cached_hint.last_accessed >= initial_last_accessed);
    }

    #[tokio::test]
    async fn test_cached_hint_age() {
        let results = vec![];
        let cache_key = "test_key".to_string();
        let cached_hint = CachedHint::new(results, 3600, cache_key);
        
        let age = cached_hint.age_seconds();
        assert!(age < 5); // Should be very young
        
        // Test with manually set created_at
        let mut old_hint = cached_hint;
        old_hint.created_at = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() - 100;
        
        let old_age = old_hint.age_seconds();
        assert!(old_age >= 90); // Should be around 100 seconds old
    }

    #[tokio::test]
    async fn test_cache_stats_hit_rate() {
        let mut stats = CacheStats::default();
        
        // No requests yet
        stats.update_hit_rate();
        assert_eq!(stats.hit_rate, 0.0);
        
        // All misses
        stats.misses = 5;
        stats.update_hit_rate();
        assert_eq!(stats.hit_rate, 0.0);
        
        // Mixed hits and misses
        stats.hits = 3;
        stats.misses = 7;
        stats.update_hit_rate();
        assert_eq!(stats.hit_rate, 0.3);
        
        // All hits
        stats.hits = 10;
        stats.misses = 0;
        stats.update_hit_rate();
        assert_eq!(stats.hit_rate, 1.0);
    }

    #[tokio::test]
    async fn test_cache_stats_clone() {
        let mut stats = CacheStats::default();
        stats.hits = 5;
        stats.misses = 3;
        stats.size = 10;
        
        let cloned = stats.clone();
        assert_eq!(cloned.hits, stats.hits);
        assert_eq!(cloned.misses, stats.misses);
        assert_eq!(cloned.size, stats.size);
    }

    #[tokio::test]
    async fn test_file_tracker_creation() {
        use tempfile::NamedTempFile;
        use tokio::io::AsyncWriteExt;
        
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().to_path_buf();
        
        // Write some content
        {
            let mut file = tokio::fs::File::create(&temp_path).await.unwrap();
            file.write_all(b"test content").await.unwrap();
        }
        
        let tracker = FileTracker::new(temp_path.clone()).await.unwrap();
        
        assert_eq!(tracker.path, temp_path);
        assert!(tracker.size > 0);
        assert!(!tracker.has_changed().await);
    }

    #[tokio::test]
    async fn test_file_tracker_change_detection() {
        use tempfile::NamedTempFile;
        use tokio::io::AsyncWriteExt;
        
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().to_path_buf();
        
        // Create initial file
        {
            let mut file = tokio::fs::File::create(&temp_path).await.unwrap();
            file.write_all(b"initial content").await.unwrap();
        }
        
        let tracker = FileTracker::new(temp_path.clone()).await.unwrap();
        assert!(!tracker.has_changed().await);
        
        // Modify the file
        sleep(Duration::from_millis(100)).await; // Ensure different timestamp
        {
            let mut file = tokio::fs::File::create(&temp_path).await.unwrap();
            file.write_all(b"modified content").await.unwrap();
        }
        
        assert!(tracker.has_changed().await);
    }

    #[tokio::test]
    async fn test_file_tracker_missing_file() {
        let non_existent_path = PathBuf::from("/definitely/does/not/exist.txt");
        let tracker = FileTracker::new(non_existent_path.clone()).await;
        
        // Should fail to create tracker for non-existent file
        assert!(tracker.is_err());
        
        // For existing tracker, missing file should be detected as changed
        use tempfile::NamedTempFile;
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().to_path_buf();
        
        let tracker = FileTracker::new(temp_path.clone()).await.unwrap();
        
        // Remove the file
        std::fs::remove_file(&temp_path).unwrap();
        
        // Should detect as changed
        assert!(tracker.has_changed().await);
    }

    #[tokio::test]
    async fn test_cache_clear() {
        let cache = HintCache::new(1).await.unwrap();
        let results = vec![LspSearchResult {
            file_path: "/test/file.rs".to_string(),
            line_number: 10,
            column: 5,
            content: "test content".to_string(),
            hint_type: HintType::Definition,
            server_type: LspServerType::Rust,
            confidence: 0.9,
            context_lines: None,
        }];
        
        // Add some entries
        cache.set("key1".to_string(), results.clone(), 3600).await.unwrap();
        cache.set("key2".to_string(), results, 3600).await.unwrap();
        
        let stats = cache.stats().await;
        assert_eq!(stats.size, 2);
        
        // Clear cache
        cache.clear().await.unwrap();
        
        let stats_after = cache.stats().await;
        assert_eq!(stats_after.size, 0);
        assert_eq!(stats_after.memory_usage_bytes, 0);
        
        // Entries should be gone
        assert!(cache.get("key1").await.unwrap().is_none());
        assert!(cache.get("key2").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_cache_invalidate_file() {
        let cache = HintCache::new(1).await.unwrap();
        
        let results_file1 = vec![LspSearchResult {
            file_path: "/test/file1.rs".to_string(),
            line_number: 10,
            column: 5,
            content: "test content 1".to_string(),
            hint_type: HintType::Definition,
            server_type: LspServerType::Rust,
            confidence: 0.9,
            context_lines: None,
        }];
        
        let results_file2 = vec![LspSearchResult {
            file_path: "/test/file2.rs".to_string(),
            line_number: 20,
            column: 10,
            content: "test content 2".to_string(),
            hint_type: HintType::References,
            server_type: LspServerType::Rust,
            confidence: 0.8,
            context_lines: None,
        }];
        
        // Add entries for different files
        cache.set("key1".to_string(), results_file1, 3600).await.unwrap();
        cache.set("key2".to_string(), results_file2, 3600).await.unwrap();
        
        // Invalidate file1
        let file1_path = PathBuf::from("/test/file1.rs");
        let invalidated = cache.invalidate_file(&file1_path).await.unwrap();
        
        assert_eq!(invalidated, 1);
        
        // key1 should be gone, key2 should remain
        assert!(cache.get("key1").await.unwrap().is_none());
        assert!(cache.get("key2").await.unwrap().is_some());
        
        let stats = cache.stats().await;
        assert_eq!(stats.invalidations, 1);
        assert_eq!(stats.size, 1);
    }

    #[tokio::test]
    async fn test_cache_memory_usage_estimation() {
        let cache = HintCache::new(1).await.unwrap();
        
        let initial_usage = cache.estimate_memory_usage();
        assert_eq!(initial_usage, 0);
        
        let results = vec![LspSearchResult {
            file_path: "/test/file.rs".to_string(),
            line_number: 10,
            column: 5,
            content: "test content".to_string(),
            hint_type: HintType::Definition,
            server_type: LspServerType::Rust,
            confidence: 0.9,
            context_lines: None,
        }];
        
        cache.set("key1".to_string(), results, 3600).await.unwrap();
        
        let usage_after = cache.estimate_memory_usage();
        assert_eq!(usage_after, 1024); // 1 entry * 1KB estimate
    }

    #[tokio::test]
    async fn test_cache_concurrent_access() {
        let cache = Arc::new(HintCache::new(1).await.unwrap());
        
        let results = vec![LspSearchResult {
            file_path: "/test/file.rs".to_string(),
            line_number: 10,
            column: 5,
            content: "test content".to_string(),
            hint_type: HintType::Definition,
            server_type: LspServerType::Rust,
            confidence: 0.9,
            context_lines: None,
        }];
        
        // Concurrent writes
        let mut handles = vec![];
        for i in 0..10 {
            let cache = cache.clone();
            let results = results.clone();
            let handle = tokio::spawn(async move {
                cache.set(format!("key{}", i), results, 3600).await
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.await.unwrap().unwrap();
        }
        
        let stats = cache.stats().await;
        assert_eq!(stats.size, 10);
        
        // Concurrent reads
        let mut handles = vec![];
        for i in 0..10 {
            let cache = cache.clone();
            let handle = tokio::spawn(async move {
                cache.get(&format!("key{}", i)).await
            });
            handles.push(handle);
        }
        
        for handle in handles {
            let result = handle.await.unwrap().unwrap();
            assert!(result.is_some());
        }
        
        let final_stats = cache.stats().await;
        assert_eq!(final_stats.hits, 10);
    }

    #[tokio::test]
    async fn test_symbol_hint_creation() {
        let symbol_hint = SymbolHint::new(
            "test_function".to_string(),
            "function".to_string(),
            "/test/file.rs".to_string(),
            42,
            10,
            0.95,
        );
        
        assert_eq!(symbol_hint.symbol_name, "test_function");
        assert_eq!(symbol_hint.symbol_kind, "function");
        assert_eq!(symbol_hint.file_path, "/test/file.rs");
        assert_eq!(symbol_hint.line_number, 42);
        assert_eq!(symbol_hint.column, 10);
        assert_eq!(symbol_hint.confidence, 0.95);
        assert!(symbol_hint.documentation.is_none());
        assert!(symbol_hint.signature.is_none());
    }

    #[tokio::test]
    async fn test_symbol_hint_with_optional_fields() {
        let mut symbol_hint = SymbolHint::new(
            "test_function".to_string(),
            "function".to_string(),
            "/test/file.rs".to_string(),
            42,
            10,
            0.95,
        );
        
        symbol_hint.documentation = Some("Test function documentation".to_string());
        symbol_hint.signature = Some("fn test_function() -> bool".to_string());
        
        assert!(symbol_hint.documentation.is_some());
        assert!(symbol_hint.signature.is_some());
        assert_eq!(symbol_hint.documentation.unwrap(), "Test function documentation");
    }

    #[tokio::test]
    async fn test_cache_shutdown() {
        let cache = HintCache::new(1).await.unwrap();
        
        // Add some data
        let results = vec![LspSearchResult {
            file_path: "/test/file.rs".to_string(),
            line_number: 10,
            column: 5,
            content: "test content".to_string(),
            hint_type: HintType::Definition,
            server_type: LspServerType::Rust,
            confidence: 0.9,
            context_lines: None,
        }];
        
        cache.set("key1".to_string(), results, 3600).await.unwrap();
        
        let stats_before = cache.stats().await;
        assert_eq!(stats_before.size, 1);
        
        // Shutdown should clear cache
        cache.shutdown().await.unwrap();
        
        // Cache should be empty after shutdown
        assert_eq!(cache.cache.len(), 0);
    }

    #[tokio::test]
    async fn test_cache_drop() {
        let cache = HintCache::new(1).await.unwrap();
        
        // Test that drop doesn't panic
        drop(cache);
    }

    #[tokio::test]
    async fn test_cache_evict_lru_empty_cache() {
        let cache = HintCache::new(1).await.unwrap();
        
        // Should not panic with empty cache
        cache.evict_lru().await;
        
        let stats = cache.stats().await;
        assert_eq!(stats.size, 0);
        assert_eq!(stats.evictions, 0);
    }

    #[tokio::test]
    async fn test_large_cache_performance() {
        let cache = HintCache::new(1).await.unwrap();
        let start = Instant::now();
        
        let results = vec![LspSearchResult {
            file_path: "/test/file.rs".to_string(),
            line_number: 10,
            column: 5,
            content: "test content".to_string(),
            hint_type: HintType::Definition,
            server_type: LspServerType::Rust,
            confidence: 0.9,
            context_lines: None,
        }];
        
        // Add many entries
        for i in 0..1000 {
            cache.set(format!("key{}", i), results.clone(), 3600).await.unwrap();
        }
        
        let insert_time = start.elapsed();
        assert!(insert_time < Duration::from_secs(5)); // Should be reasonably fast
        
        // Test retrieval performance
        let start = Instant::now();
        for i in 0..1000 {
            let _result = cache.get(&format!("key{}", i)).await.unwrap();
        }
        
        let retrieval_time = start.elapsed();
        assert!(retrieval_time < Duration::from_secs(2)); // Should be fast
        
        let stats = cache.stats().await;
        assert_eq!(stats.hits, 1000);
        assert_eq!(stats.size, 1000);
    }

    #[tokio::test]
    async fn test_cache_key_with_special_characters() {
        let cache = HintCache::new(1).await.unwrap();
        
        let results = vec![LspSearchResult {
            file_path: "/test/file.rs".to_string(),
            line_number: 10,
            column: 5,
            content: "test content".to_string(),
            hint_type: HintType::Definition,
            server_type: LspServerType::Rust,
            confidence: 0.9,
            context_lines: None,
        }];
        
        let special_keys = vec![
            "key with spaces".to_string(),
            "key:with:colons".to_string(),
            "key/with/slashes".to_string(),
            "key-with-dashes".to_string(),
            "key_with_underscores".to_string(),
            "key.with.dots".to_string(),
            "key@with@symbols".to_string(),
        ];
        
        for key in special_keys {
            cache.set(key.clone(), results.clone(), 3600).await.unwrap();
            let retrieved = cache.get(&key).await.unwrap();
            assert!(retrieved.is_some());
        }
    }

    #[tokio::test]
    async fn test_empty_results_caching() {
        let cache = HintCache::new(1).await.unwrap();
        let key = "empty_results".to_string();
        
        // Cache empty results
        let empty_results: Vec<LspSearchResult> = vec![];
        cache.set(key.clone(), empty_results, 3600).await.unwrap();
        
        let retrieved = cache.get(&key).await.unwrap().unwrap();
        assert!(retrieved.is_empty());
        
        let stats = cache.stats().await;
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.size, 1);
    }
}