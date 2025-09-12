use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::lsp::hint::SymbolHint;

/// LSP hint cache for 24-hour persistence
#[derive(Debug, Clone)]
pub struct HintCache {
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    max_entries: usize,
    ttl: Duration,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    hints: Vec<SymbolHint>,
    created_at: Instant,
    file_hash: u64,
    access_count: u64,
}

impl HintCache {
    /// Create a new hint cache
    pub fn new(max_entries: usize, ttl_hours: u64) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_entries,
            ttl: Duration::from_secs(ttl_hours * 3600),
        }
    }

    /// Create cache with 24-hour TTL as specified in TODO.md
    pub fn with_24h_ttl(max_entries: usize) -> Self {
        Self::new(max_entries, 24)
    }

    /// Get hints from cache if valid
    pub async fn get_hints(&self, file_path: &str, file_hash: u64) -> Option<Vec<SymbolHint>> {
        let mut cache = self.cache.write().await;
        
        if let Some(entry) = cache.get_mut(file_path) {
            // Check if entry is still valid
            if entry.created_at.elapsed() < self.ttl && entry.file_hash == file_hash {
                entry.access_count += 1;
                debug!("Cache hit for {}: {} hints", file_path, entry.hints.len());
                return Some(entry.hints.clone());
            } else {
                // Remove stale entry
                cache.remove(file_path);
                debug!("Cache miss (stale) for {}", file_path);
            }
        } else {
            debug!("Cache miss for {}", file_path);
        }

        None
    }

    /// Store hints in cache
    pub async fn store_hints(&self, file_path: String, file_hash: u64, hints: Vec<SymbolHint>) {
        let mut cache = self.cache.write().await;

        // Evict old entries if at capacity
        if cache.len() >= self.max_entries {
            self.evict_lru(&mut cache).await;
        }

        let entry = CacheEntry {
            hints,
            created_at: Instant::now(),
            file_hash,
            access_count: 1,
        };

        cache.insert(file_path.clone(), entry);
        debug!("Cached hints for {}: {} entries", file_path, cache.len());
    }

    /// Invalidate cache entry for a file (e.g., on file change)
    pub async fn invalidate(&self, file_path: &str) {
        let mut cache = self.cache.write().await;
        if cache.remove(file_path).is_some() {
            debug!("Invalidated cache for {}", file_path);
        }
    }

    /// Clear all cache entries
    pub async fn clear(&self) {
        let mut cache = self.cache.write().await;
        let count = cache.len();
        cache.clear();
        info!("Cleared cache: {} entries removed", count);
    }

    /// Get cache statistics
    pub async fn stats(&self) -> CacheStats {
        let cache = self.cache.read().await;
        let now = Instant::now();
        
        let mut total_access_count = 0;
        let mut valid_entries = 0;
        let mut stale_entries = 0;

        for entry in cache.values() {
            total_access_count += entry.access_count;
            if entry.created_at.elapsed() < self.ttl {
                valid_entries += 1;
            } else {
                stale_entries += 1;
            }
        }

        CacheStats {
            total_entries: cache.len(),
            valid_entries,
            stale_entries,
            total_access_count,
        }
    }

    /// Evict least recently used entry
    async fn evict_lru(&self, cache: &mut HashMap<String, CacheEntry>) {
        if let Some(lru_key) = cache
            .iter()
            .min_by_key(|(_, entry)| entry.access_count)
            .map(|(k, _)| k.clone())
        {
            cache.remove(&lru_key);
            debug!("Evicted LRU entry: {}", lru_key);
        }
    }

    /// Cleanup stale entries
    pub async fn cleanup_stale(&self) {
        let mut cache = self.cache.write().await;
        let initial_count = cache.len();
        
        cache.retain(|_, entry| entry.created_at.elapsed() < self.ttl);
        
        let removed_count = initial_count - cache.len();
        if removed_count > 0 {
            info!("Cleaned up {} stale cache entries", removed_count);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_entries: usize,
    pub valid_entries: usize,
    pub stale_entries: usize,
    pub total_access_count: u64,
}

impl Default for HintCache {
    fn default() -> Self {
        Self::with_24h_ttl(10000) // Default 10k entries with 24h TTL
    }
}

/// File hash calculator for cache invalidation
pub struct FileHasher;

impl FileHasher {
    /// Calculate a simple hash for file content validation
    pub fn hash_file_content(content: &[u8]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        hasher.finish()
    }

    /// Calculate hash from file metadata (size + mtime)
    pub async fn hash_file_metadata(path: &std::path::Path) -> Result<u64, std::io::Error> {
        let metadata = tokio::fs::metadata(path).await?;
        
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        use std::hash::{Hash, Hasher};
        
        metadata.len().hash(&mut hasher);
        if let Ok(modified) = metadata.modified() {
            if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
                duration.as_secs().hash(&mut hasher);
            }
        }
        
        Ok(hasher.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lsp::hint::SymbolHint;
    use std::path::PathBuf;
    use tokio::time::{sleep, Duration};

    fn create_test_hint(name: &str) -> SymbolHint {
        SymbolHint::new(
            name.to_string(),
            "function".to_string(),
            "/test/file.rs".to_string(),
            1,
            1,
            0.8,
        )
    }

    #[tokio::test]
    async fn test_hint_cache_new() {
        let cache = HintCache::new(100, 1);
        
        assert_eq!(cache.max_entries, 100);
        assert_eq!(cache.ttl, Duration::from_secs(3600));
    }

    #[tokio::test]
    async fn test_hint_cache_24h_ttl() {
        let cache = HintCache::with_24h_ttl(50);
        
        assert_eq!(cache.max_entries, 50);
        assert_eq!(cache.ttl, Duration::from_secs(24 * 3600));
    }

    #[tokio::test]
    async fn test_cache_store_and_get() {
        let cache = HintCache::new(10, 1);
        let hints = vec![
            create_test_hint("function1"),
            create_test_hint("function2"),
        ];
        
        // Store hints
        cache.store_hints("test.rs".to_string(), 12345, hints.clone()).await;
        
        // Retrieve hints with same hash
        let retrieved = cache.get_hints("test.rs", 12345).await;
        assert!(retrieved.is_some());
        
        let retrieved_hints = retrieved.unwrap();
        assert_eq!(retrieved_hints.len(), 2);
        assert_eq!(retrieved_hints[0].symbol_name, "function1");
        assert_eq!(retrieved_hints[1].symbol_name, "function2");
    }

    #[tokio::test]
    async fn test_cache_miss_different_hash() {
        let cache = HintCache::new(10, 1);
        let hints = vec![create_test_hint("function1")];
        
        cache.store_hints("test.rs".to_string(), 12345, hints).await;
        
        // Try to get with different hash
        let retrieved = cache.get_hints("test.rs", 67890).await;
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_cache_miss_different_file() {
        let cache = HintCache::new(10, 1);
        let hints = vec![create_test_hint("function1")];
        
        cache.store_hints("test.rs".to_string(), 12345, hints).await;
        
        // Try to get different file
        let retrieved = cache.get_hints("other.rs", 12345).await;
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_cache_eviction_by_size() {
        let cache = HintCache::new(2, 1); // Small cache
        let hints1 = vec![create_test_hint("func1")];
        let hints2 = vec![create_test_hint("func2")];
        let hints3 = vec![create_test_hint("func3")];
        
        // Fill cache to capacity
        cache.store_hints("file1.rs".to_string(), 1, hints1).await;
        cache.store_hints("file2.rs".to_string(), 2, hints2).await;
        
        // This should trigger eviction
        cache.store_hints("file3.rs".to_string(), 3, hints3).await;
        
        // Verify cache size is maintained
        let stats = cache.stats().await;
        assert!(stats.total_entries <= 2);
    }

    #[tokio::test]
    async fn test_cache_clear() {
        let cache = HintCache::new(10, 1);
        let hints = vec![create_test_hint("function1")];
        
        cache.store_hints("test.rs".to_string(), 12345, hints).await;
        
        // Verify it exists
        let retrieved = cache.get_hints("test.rs", 12345).await;
        assert!(retrieved.is_some());
        
        // Clear cache
        cache.clear().await;
        
        // Should be gone
        let cleared = cache.get_hints("test.rs", 12345).await;
        assert!(cleared.is_none());
    }

    #[tokio::test]
    async fn test_cache_invalidate_file() {
        let cache = HintCache::new(10, 1);
        let hints1 = vec![create_test_hint("func1")];
        let hints2 = vec![create_test_hint("func2")];
        
        cache.store_hints("file1.rs".to_string(), 1, hints1).await;
        cache.store_hints("file2.rs".to_string(), 2, hints2).await;
        
        // Invalidate specific file
        cache.invalidate("file1.rs").await;
        
        // file1 should be gone, file2 should remain
        assert!(cache.get_hints("file1.rs", 1).await.is_none());
        assert!(cache.get_hints("file2.rs", 2).await.is_some());
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let cache = HintCache::new(10, 1);
        let hints = vec![create_test_hint("function1")];
        
        // Initial stats
        let initial_stats = cache.stats().await;
        assert_eq!(initial_stats.total_entries, 0);
        
        // Add entry
        cache.store_hints("test.rs".to_string(), 12345, hints).await;
        
        // Check stats after store
        let after_store = cache.stats().await;
        assert_eq!(after_store.total_entries, 1);
    }

    #[tokio::test]
    async fn test_concurrent_cache_access() {
        let cache = Arc::new(HintCache::new(10, 1));
        let hints = vec![create_test_hint("function1")];
        
        cache.store_hints("test.rs".to_string(), 12345, hints).await;
        
        // Spawn multiple concurrent readers
        let mut handles = vec![];
        for i in 0..10 {
            let cache_clone = cache.clone();
            let handle = tokio::spawn(async move {
                let result = cache_clone.get_hints("test.rs", 12345).await;
                assert!(result.is_some(), "Read {} should succeed", i);
            });
            handles.push(handle);
        }
        
        // Wait for all reads to complete
        for handle in handles {
            handle.await.expect("Task should complete successfully");
        }
    }

    #[test]
    fn test_file_hasher_consistency() {
        let content1 = b"hello world";
        let content2 = b"hello world";
        let content3 = b"hello world!";
        
        // Same content should produce same hash
        assert_eq!(
            FileHasher::hash_file_content(content1),
            FileHasher::hash_file_content(content2)
        );
        
        // Different content should produce different hash
        assert_ne!(
            FileHasher::hash_file_content(content1),
            FileHasher::hash_file_content(content3)
        );
    }

    #[test]
    fn test_file_hasher_empty_content() {
        let empty = b"";
        let hash = FileHasher::hash_file_content(empty);
        
        // Empty content should still produce a valid hash
        assert!(hash != 0); // Just verify it's not zero (implementation detail)
    }

    #[test]
    fn test_cache_stats_default() {
        let stats = CacheStats {
            total_entries: 0,
            valid_entries: 0,
            stale_entries: 0,
            total_access_count: 0,
        };
        
        assert_eq!(stats.total_entries, 0);
        assert_eq!(stats.valid_entries, 0);
        assert_eq!(stats.stale_entries, 0);
        assert_eq!(stats.total_access_count, 0);
    }

    #[test]
    fn test_cache_entry_creation() {
        let hints = vec![create_test_hint("test_func")];
        let entry = CacheEntry {
            hints: hints.clone(),
            created_at: Instant::now(),
            file_hash: 12345,
            access_count: 0,
        };
        
        assert_eq!(entry.hints.len(), 1);
        assert_eq!(entry.file_hash, 12345);
        assert_eq!(entry.access_count, 0);
        assert_eq!(entry.hints[0].symbol_name, "test_func");
    }

    #[tokio::test]
    async fn test_default_cache_configuration() {
        let cache = HintCache::default();
        
        // Should use 24-hour TTL and 10k entries as per TODO.md
        assert_eq!(cache.ttl, Duration::from_secs(24 * 3600));
        assert_eq!(cache.max_entries, 10000);
    }
}