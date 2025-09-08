//! Zero-Copy Memory Management
//!
//! Implements zero-copy buffer management for the fused pipeline:
//! - Memory-mapped buffers for large datasets
//! - Segment views without data copying
//! - Reference-counted buffer sharing
//! - Efficient memory pool management

use anyhow::{anyhow, Result};
use bytes::{Bytes, BytesMut};
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::collections::VecDeque;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, warn};

/// Zero-copy buffer with reference counting and segment views
#[derive(Debug)]
pub struct ZeroCopyBuffer {
    data: Bytes,
    size: usize,
    ref_count: Arc<AtomicUsize>,
    id: BufferId,
}

/// Unique buffer identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(u64);

impl BufferId {
    fn new() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed) as u64)
    }
}

impl ZeroCopyBuffer {
    /// Create a new zero-copy buffer with specified capacity
    pub fn new(capacity: usize) -> Self {
        let data = BytesMut::with_capacity(capacity).freeze();
        
        Self {
            data,
            size: 0,
            ref_count: Arc::new(AtomicUsize::new(1)),
            id: BufferId::new(),
        }
    }

    /// Create buffer from existing data without copying
    pub fn from_bytes(bytes: Bytes) -> Self {
        let size = bytes.len();
        
        Self {
            data: bytes,
            size,
            ref_count: Arc::new(AtomicUsize::new(1)),
            id: BufferId::new(),
        }
    }

    /// Create a segment view of this buffer
    pub fn create_view(&self, offset: usize, length: usize) -> Result<SegmentView> {
        if offset + length > self.data.len() {
            return Err(anyhow!("Segment view bounds exceed buffer size"));
        }

        // Increment reference count for the view
        self.ref_count.fetch_add(1, Ordering::Relaxed);

        Ok(SegmentView {
            buffer_id: self.id,
            data: self.data.slice(offset..offset + length),
            offset,
            length,
            buffer_ref: self.ref_count.clone(),
            view_type: ViewType::Data,
        })
    }

    /// Create a view of the entire buffer
    pub fn create_full_view(&self) -> SegmentView {
        self.ref_count.fetch_add(1, Ordering::Relaxed);
        
        SegmentView {
            buffer_id: self.id,
            data: self.data.clone(),
            offset: 0,
            length: self.data.len(),
            buffer_ref: self.ref_count.clone(),
            view_type: ViewType::Full,
        }
    }

    /// Get buffer size
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get current reference count
    pub fn ref_count(&self) -> usize {
        self.ref_count.load(Ordering::Relaxed)
    }

    /// Get buffer ID
    pub fn id(&self) -> BufferId {
        self.id
    }
}

impl Drop for ZeroCopyBuffer {
    fn drop(&mut self) {
        let remaining_refs = self.ref_count.fetch_sub(1, Ordering::Relaxed);
        debug!("Buffer {:?} dropped, {} references remaining", self.id, remaining_refs - 1);
    }
}

/// View type for different kinds of segment views
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ViewType {
    /// View of data segment
    Data,
    /// View of metadata segment  
    Metadata,
    /// View of search results
    Results,
    /// Full buffer view
    Full,
    /// Compressed data view
    Compressed,
}

/// Zero-copy segment view into a buffer
#[derive(Debug, Clone)]
pub struct SegmentView {
    buffer_id: BufferId,
    data: Bytes,
    offset: usize,
    length: usize,
    buffer_ref: Arc<AtomicUsize>,
    view_type: ViewType,
}

impl SegmentView {
    /// Get the data as bytes (zero-copy)
    pub fn as_bytes(&self) -> &Bytes {
        &self.data
    }

    /// Get data as slice
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Get view length
    pub fn len(&self) -> usize {
        self.length
    }

    /// Check if view is empty
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Get offset within parent buffer
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get view type
    pub fn view_type(&self) -> ViewType {
        self.view_type
    }

    /// Get buffer ID this view belongs to
    pub fn buffer_id(&self) -> BufferId {
        self.buffer_id
    }

    /// Create a sub-view of this view (zero-copy)
    pub fn slice(&self, range: Range<usize>) -> Result<SegmentView> {
        if range.end > self.length {
            return Err(anyhow!("Slice range exceeds view length"));
        }

        // Increment reference count for new view
        self.buffer_ref.fetch_add(1, Ordering::Relaxed);

        Ok(SegmentView {
            buffer_id: self.buffer_id,
            data: self.data.slice(range.start..range.end),
            offset: self.offset + range.start,
            length: range.end - range.start,
            buffer_ref: self.buffer_ref.clone(),
            view_type: self.view_type,
        })
    }

    /// Convert to string (for text data)
    pub fn to_string_lossy(&self) -> String {
        String::from_utf8_lossy(&self.data).to_string()
    }

    /// Parse as JSON
    pub fn parse_json<T>(&self) -> Result<T> 
    where 
        T: serde::de::DeserializeOwned 
    {
        let text = std::str::from_utf8(&self.data)?;
        Ok(serde_json::from_str(text)?)
    }
}

impl Drop for SegmentView {
    fn drop(&mut self) {
        let remaining_refs = self.buffer_ref.fetch_sub(1, Ordering::Relaxed);
        if remaining_refs == 1 {
            debug!("Last reference to buffer {:?} dropped", self.buffer_id);
        }
    }
}

/// Memory pool for efficient buffer management
pub struct BufferPool {
    small_buffers: Mutex<VecDeque<ZeroCopyBuffer>>,  // < 1KB
    medium_buffers: Mutex<VecDeque<ZeroCopyBuffer>>, // 1KB - 100KB  
    large_buffers: Mutex<VecDeque<ZeroCopyBuffer>>,  // > 100KB
    stats: RwLock<PoolStats>,
    config: PoolConfig,
}

#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub small_buffer_size: usize,
    pub medium_buffer_size: usize,
    pub large_buffer_size: usize,
    pub max_small_buffers: usize,
    pub max_medium_buffers: usize,
    pub max_large_buffers: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            small_buffer_size: 1024,       // 1KB
            medium_buffer_size: 100_1024,  // 100KB
            large_buffer_size: 1024_1024,  // 1MB
            max_small_buffers: 100,
            max_medium_buffers: 50,
            max_large_buffers: 10,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct PoolStats {
    pub total_allocations: usize,
    pub pool_hits: usize,
    pub pool_misses: usize,
    pub small_buffers_active: usize,
    pub medium_buffers_active: usize,
    pub large_buffers_active: usize,
    pub total_memory_bytes: usize,
}

impl PoolStats {
    pub fn hit_rate(&self) -> f64 {
        if self.total_allocations == 0 {
            0.0
        } else {
            self.pool_hits as f64 / self.total_allocations as f64
        }
    }
}

impl BufferPool {
    /// Create a new buffer pool
    pub fn new(config: PoolConfig) -> Self {
        Self {
            small_buffers: Mutex::new(VecDeque::new()),
            medium_buffers: Mutex::new(VecDeque::new()),
            large_buffers: Mutex::new(VecDeque::new()),
            stats: RwLock::new(PoolStats::default()),
            config,
        }
    }

    /// Acquire a buffer from the pool or allocate new one
    pub async fn acquire(&self, size: usize) -> Result<ZeroCopyBuffer> {
        let mut stats = self.stats.write().await;
        stats.total_allocations += 1;

        let buffer = if size <= self.config.small_buffer_size {
            self.acquire_small_buffer(&mut stats).await
        } else if size <= self.config.medium_buffer_size {
            self.acquire_medium_buffer(&mut stats).await
        } else {
            self.acquire_large_buffer(&mut stats, size).await
        };

        match buffer {
            Some(buf) => {
                stats.pool_hits += 1;
                Ok(buf)
            }
            None => {
                stats.pool_misses += 1;
                Ok(ZeroCopyBuffer::new(size))
            }
        }
    }

    async fn acquire_small_buffer(&self, stats: &mut PoolStats) -> Option<ZeroCopyBuffer> {
        let mut pool = self.small_buffers.lock().await;
        let buffer = pool.pop_front();
        
        if buffer.is_some() {
            stats.small_buffers_active += 1;
        }
        
        buffer
    }

    async fn acquire_medium_buffer(&self, stats: &mut PoolStats) -> Option<ZeroCopyBuffer> {
        let mut pool = self.medium_buffers.lock().await;
        let buffer = pool.pop_front();
        
        if buffer.is_some() {
            stats.medium_buffers_active += 1;
        }
        
        buffer
    }

    async fn acquire_large_buffer(&self, stats: &mut PoolStats, size: usize) -> Option<ZeroCopyBuffer> {
        let mut pool = self.large_buffers.lock().await;
        
        // Find a large buffer that's big enough
        let pos = pool.iter().position(|buf| buf.len() >= size);
        
        if let Some(idx) = pos {
            let buffer = pool.remove(idx);
            stats.large_buffers_active += 1;
            buffer
        } else {
            None
        }
    }

    /// Return a buffer to the pool for reuse
    pub async fn release(&self, buffer: ZeroCopyBuffer) -> Result<()> {
        // Only pool buffers that have no other references
        if buffer.ref_count() > 1 {
            return Ok(()); // Let it drop naturally
        }

        let size = buffer.len();
        let mut stats = self.stats.write().await;

        if size <= self.config.small_buffer_size {
            let mut pool = self.small_buffers.lock().await;
            if pool.len() < self.config.max_small_buffers {
                pool.push_back(buffer);
                if stats.small_buffers_active > 0 {
                    stats.small_buffers_active -= 1;
                }
            }
        } else if size <= self.config.medium_buffer_size {
            let mut pool = self.medium_buffers.lock().await;
            if pool.len() < self.config.max_medium_buffers {
                pool.push_back(buffer);
                if stats.medium_buffers_active > 0 {
                    stats.medium_buffers_active -= 1;
                }
            }
        } else {
            let mut pool = self.large_buffers.lock().await;
            if pool.len() < self.config.max_large_buffers {
                pool.push_back(buffer);
                if stats.large_buffers_active > 0 {
                    stats.large_buffers_active -= 1;
                }
            }
        }

        Ok(())
    }

    /// Get pool statistics
    pub async fn stats(&self) -> PoolStats {
        let stats = self.stats.read().await;
        stats.clone()
    }

    /// Clear all pooled buffers
    pub async fn clear(&self) -> Result<()> {
        let mut small = self.small_buffers.lock().await;
        let mut medium = self.medium_buffers.lock().await;
        let mut large = self.large_buffers.lock().await;

        small.clear();
        medium.clear();
        large.clear();

        let mut stats = self.stats.write().await;
        stats.small_buffers_active = 0;
        stats.medium_buffers_active = 0;
        stats.large_buffers_active = 0;

        Ok(())
    }
}

/// Memory manager for the entire pipeline
pub struct PipelineMemoryManager {
    buffer_pool: BufferPool,
    active_buffers: RwLock<std::collections::HashMap<BufferId, Arc<ZeroCopyBuffer>>>,
    memory_limit: usize,
    current_usage: AtomicUsize,
}

impl PipelineMemoryManager {
    /// Create a new memory manager
    pub fn new(memory_limit_mb: usize) -> Self {
        let config = PoolConfig::default();
        let buffer_pool = BufferPool::new(config);
        
        Self {
            buffer_pool,
            active_buffers: RwLock::new(std::collections::HashMap::new()),
            memory_limit: memory_limit_mb * 1024 * 1024, // Convert to bytes
            current_usage: AtomicUsize::new(0),
        }
    }

    /// Allocate a new buffer
    pub async fn allocate(&self, size: usize) -> Result<Arc<ZeroCopyBuffer>> {
        // Check memory limit
        let current = self.current_usage.load(Ordering::Relaxed);
        if current + size > self.memory_limit {
            return Err(anyhow!("Memory limit exceeded: {} + {} > {}", 
                             current, size, self.memory_limit));
        }

        let buffer = Arc::new(self.buffer_pool.acquire(size).await?);
        let buffer_id = buffer.id();

        // Track the buffer
        {
            let mut active = self.active_buffers.write().await;
            active.insert(buffer_id, buffer.clone());
        }

        self.current_usage.fetch_add(size, Ordering::Relaxed);
        
        debug!("Allocated buffer {:?} of {} bytes", buffer_id, size);
        Ok(buffer)
    }

    /// Deallocate a buffer
    pub async fn deallocate(&self, buffer_id: BufferId) -> Result<()> {
        let buffer = {
            let mut active = self.active_buffers.write().await;
            active.remove(&buffer_id)
        };

        if let Some(buffer) = buffer {
            let size = buffer.len();
            
            // Try to return to pool
            if Arc::strong_count(&buffer) == 1 {
                // We have the only reference, safe to extract and pool
                if let Ok(owned_buffer) = Arc::try_unwrap(buffer) {
                    self.buffer_pool.release(owned_buffer).await?;
                }
            }
            
            self.current_usage.fetch_sub(size, Ordering::Relaxed);
            debug!("Deallocated buffer {:?} of {} bytes", buffer_id, size);
        }

        Ok(())
    }

    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        self.current_usage.load(Ordering::Relaxed)
    }

    /// Get memory utilization percentage
    pub fn utilization(&self) -> f64 {
        self.current_usage() as f64 / self.memory_limit as f64
    }

    /// Get buffer pool statistics
    pub async fn pool_stats(&self) -> PoolStats {
        self.buffer_pool.stats().await
    }

    /// Force garbage collection of unused buffers
    pub async fn gc(&self) -> Result<usize> {
        let mut collected = 0;
        let mut to_remove = Vec::new();

        {
            let active = self.active_buffers.read().await;
            for (id, buffer) in active.iter() {
                // If we have the only reference, it can be collected
                if Arc::strong_count(buffer) == 1 {
                    to_remove.push(*id);
                }
            }
        }

        for buffer_id in to_remove {
            self.deallocate(buffer_id).await?;
            collected += 1;
        }

        if collected > 0 {
            debug!("Garbage collected {} unused buffers", collected);
        }

        Ok(collected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_copy_buffer() {
        let buffer = ZeroCopyBuffer::new(1024);
        assert_eq!(buffer.len(), 0); // Empty until data is added
        assert_eq!(buffer.ref_count(), 1);
        
        let view = buffer.create_view(0, 0).unwrap();
        assert_eq!(buffer.ref_count(), 2); // Buffer + view
        
        drop(view);
        assert_eq!(buffer.ref_count(), 1); // Just buffer
    }

    #[test]
    fn test_segment_view() {
        let data = Bytes::from("Hello, World!");
        let buffer = ZeroCopyBuffer::from_bytes(data);
        
        let view = buffer.create_view(0, 5).unwrap();
        assert_eq!(view.len(), 5);
        assert_eq!(view.to_string_lossy(), "Hello");
        
        let sub_view = view.slice(1..4).unwrap();
        assert_eq!(sub_view.len(), 3);
        assert_eq!(sub_view.to_string_lossy(), "ell");
    }

    #[tokio::test]
    async fn test_buffer_pool() {
        let config = PoolConfig::default();
        let pool = BufferPool::new(config);
        
        // First acquisition should be a miss
        let buffer1 = pool.acquire(512).await.unwrap();
        let stats = pool.stats().await;
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.pool_misses, 1);
        
        // Release and reacquire should be a hit
        pool.release(buffer1).await.unwrap();
        let _buffer2 = pool.acquire(512).await.unwrap();
        
        let stats = pool.stats().await;
        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.pool_hits, 1);
    }

    #[tokio::test]
    async fn test_memory_manager() {
        let manager = PipelineMemoryManager::new(1); // 1MB limit
        
        let buffer = manager.allocate(1024).await.unwrap();
        assert_eq!(manager.current_usage(), 1024); // Buffer allocated, usage increases
        
        // Test memory limit
        let large_buffer = manager.allocate(2 * 1024 * 1024).await;
        assert!(large_buffer.is_err()); // Should exceed 1MB limit
        
        manager.deallocate(buffer.id()).await.unwrap();
    }

    #[tokio::test]
    async fn test_garbage_collection() {
        let manager = PipelineMemoryManager::new(10); // 10MB limit
        
        let buffer = manager.allocate(1024).await.unwrap();
        let buffer_id = buffer.id();
        
        // Buffer is still referenced, shouldn't be collected
        let collected = manager.gc().await.unwrap();
        assert_eq!(collected, 0);
        
        drop(buffer);
        
        // Now buffer can be collected
        let collected = manager.gc().await.unwrap();
        assert_eq!(collected, 1);
    }
}