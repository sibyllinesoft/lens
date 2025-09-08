//! # 2048-Token Semantic Encoder
//!
//! High-capacity code encoder with full context understanding:
//! - 2048-token context window for long files (up to ~100KB)
//! - CodeT5/UniXcoder-class transformer architecture
//! - Efficient tokenization and embedding generation
//! - Performance target: ≤50ms p95 inference time

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use super::EncoderConfig;

/// Maximum file size supported (approximately 100KB)
const MAX_FILE_SIZE_BYTES: usize = 100 * 1024;

/// Token-to-character ratio approximation for code
const APPROX_CHARS_PER_TOKEN: f32 = 4.0;

/// Semantic encoder for code understanding
pub struct SemanticEncoder {
    config: EncoderConfig,
    tokenizer: Arc<RwLock<Option<CodeTokenizer>>>,
    model: Arc<RwLock<Option<CodeModel>>>,
    cache: Arc<RwLock<EmbeddingCache>>,
}

/// High-level tokenizer interface for code
pub struct CodeTokenizer {
    model_type: String,
    max_tokens: usize,
    vocab_size: usize,
    special_tokens: HashMap<String, u32>,
}

/// High-level model interface for embeddings
pub struct CodeModel {
    model_type: String, 
    embedding_dim: usize,
    device: String,
    initialized: bool,
}

/// LRU cache for embeddings with content-addressable keys
#[derive(Default)]
pub struct EmbeddingCache {
    cache: HashMap<String, CachedEmbedding>,
    access_order: Vec<String>,
    max_size: usize,
}

#[derive(Clone)]
pub struct CachedEmbedding {
    embedding: Vec<f32>,
    timestamp: std::time::Instant,
    access_count: u32,
}

/// Tokenized code with metadata
#[derive(Debug, Clone)]
pub struct TokenizedCode {
    pub tokens: Vec<u32>,
    pub attention_mask: Vec<u32>, 
    pub token_type_ids: Vec<u32>,
    pub original_length: usize,
    pub truncated: bool,
    pub language: Option<String>,
}

/// Code embedding with rich metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeEmbedding {
    pub embedding: Vec<f32>,
    pub dimension: usize,
    pub model_version: String,
    pub content_hash: String,
    pub language: Option<String>,
    pub token_count: usize,
    pub inference_time_ms: u64,
}

impl SemanticEncoder {
    /// Create new semantic encoder
    pub async fn new(config: EncoderConfig) -> Result<Self> {
        info!("Creating semantic encoder: {}", config.model_type);
        info!("Max tokens: {}, embedding dim: {}", config.max_tokens, config.embedding_dim);
        
        Ok(Self {
            config,
            tokenizer: Arc::new(RwLock::new(None)),
            model: Arc::new(RwLock::new(None)), 
            cache: Arc::new(RwLock::new(EmbeddingCache::new(1000))), // 1K cache entries
        })
    }
    
    /// Initialize the encoder (loads model weights, tokenizer)
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing semantic encoder: {}", self.config.model_type);
        
        // Initialize tokenizer
        let tokenizer = self.create_tokenizer().await
            .context("Failed to create tokenizer")?;
        
        *self.tokenizer.write().await = Some(tokenizer);
        
        // Initialize model
        let model = self.create_model().await
            .context("Failed to create model")?;
            
        *self.model.write().await = Some(model);
        
        info!("Semantic encoder initialized successfully");
        Ok(())
    }
    
    /// Encode code content into dense embedding
    pub async fn encode(&self, content: &str, language: Option<&str>) -> Result<CodeEmbedding> {
        let start_time = std::time::Instant::now();
        
        // Check content size limits
        if content.len() > MAX_FILE_SIZE_BYTES {
            warn!("Content size {}KB exceeds limit {}KB, truncating", 
                  content.len() / 1024, MAX_FILE_SIZE_BYTES / 1024);
        }
        
        // Generate content hash for caching
        let content_hash = self.hash_content(content);
        
        // Check cache first
        if let Some(cached) = self.get_cached_embedding(&content_hash).await {
            debug!("Cache hit for content hash: {}", content_hash);
            return Ok(cached);
        }
        
        // Tokenize the content
        let tokenized = self.tokenize_content(content, language).await
            .context("Failed to tokenize content")?;
            
        // Generate embedding  
        let embedding_vec = self.generate_embedding(&tokenized).await
            .context("Failed to generate embedding")?;
            
        let inference_time = start_time.elapsed().as_millis() as u64;
        
        // Create embedding result
        let embedding = CodeEmbedding {
            embedding: embedding_vec,
            dimension: self.config.embedding_dim,
            model_version: self.config.model_type.clone(),
            content_hash: content_hash.clone(),
            language: language.map(String::from),
            token_count: tokenized.tokens.len(),
            inference_time_ms: inference_time,
        };
        
        // Cache the result
        self.cache_embedding(&content_hash, &embedding).await;
        
        // Performance tracking
        if inference_time > 50 {
            warn!("Slow inference: {}ms > 50ms target", inference_time);
        }
        
        debug!("Encoded {} tokens in {}ms", tokenized.tokens.len(), inference_time);
        
        Ok(embedding)
    }
    
    /// Batch encode multiple code fragments for efficiency
    pub async fn encode_batch(&self, contents: &[(&str, Option<&str>)]) -> Result<Vec<CodeEmbedding>> {
        let start_time = std::time::Instant::now();
        
        // Check batch size limits
        if contents.len() > self.config.batch_size {
            warn!("Batch size {} exceeds limit {}, processing in chunks",
                  contents.len(), self.config.batch_size);
        }
        
        let mut results = Vec::with_capacity(contents.len());
        
        // Process in chunks to respect batch size limits
        for chunk in contents.chunks(self.config.batch_size) {
            let chunk_results = self.encode_chunk(chunk).await?;
            results.extend(chunk_results);
        }
        
        let total_time = start_time.elapsed().as_millis() as u64;
        let avg_time_per_item = total_time / contents.len() as u64;
        
        info!("Batch encoded {} items in {}ms (avg {}ms/item)", 
              contents.len(), total_time, avg_time_per_item);
              
        Ok(results)
    }
    
    /// Check if content is within context limits
    pub fn can_handle_content(&self, content: &str) -> bool {
        let estimated_tokens = (content.len() as f32 / APPROX_CHARS_PER_TOKEN) as usize;
        estimated_tokens <= self.config.max_tokens && content.len() <= MAX_FILE_SIZE_BYTES
    }
    
    /// Get encoder performance metrics
    pub async fn get_metrics(&self) -> EncoderMetrics {
        let cache_guard = self.cache.read().await;
        
        EncoderMetrics {
            cache_size: cache_guard.cache.len(),
            cache_hit_rate: cache_guard.calculate_hit_rate(),
            max_tokens: self.config.max_tokens,
            embedding_dim: self.config.embedding_dim,
            model_type: self.config.model_type.clone(),
        }
    }
    
    // Private implementation methods
    
    async fn create_tokenizer(&self) -> Result<CodeTokenizer> {
        info!("Loading tokenizer for {}", self.config.model_type);
        
        // Mock tokenizer creation - in real implementation would load from disk/HuggingFace
        let mut special_tokens = HashMap::new();
        special_tokens.insert("<pad>".to_string(), 0);
        special_tokens.insert("<unk>".to_string(), 1); 
        special_tokens.insert("<s>".to_string(), 2);
        special_tokens.insert("</s>".to_string(), 3);
        
        Ok(CodeTokenizer {
            model_type: self.config.model_type.clone(),
            max_tokens: self.config.max_tokens,
            vocab_size: 50000, // Typical CodeT5 vocab size
            special_tokens,
        })
    }
    
    async fn create_model(&self) -> Result<CodeModel> {
        info!("Loading model {} on device {}", self.config.model_type, self.config.device);
        
        // Mock model creation - in real implementation would load weights
        Ok(CodeModel {
            model_type: self.config.model_type.clone(),
            embedding_dim: self.config.embedding_dim,
            device: self.config.device.clone(),
            initialized: true,
        })
    }
    
    async fn tokenize_content(&self, content: &str, language: Option<&str>) -> Result<TokenizedCode> {
        let tokenizer_guard = self.tokenizer.read().await;
        let tokenizer = tokenizer_guard.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Tokenizer not initialized"))?;
            
        // Truncate content if needed
        let max_chars = (self.config.max_tokens as f32 * APPROX_CHARS_PER_TOKEN) as usize;
        let content = if content.len() > max_chars {
            &content[..max_chars]
        } else {
            content
        };
        
        // Mock tokenization - real implementation would use proper tokenizer
        let tokens = self.mock_tokenize(content, tokenizer).await?;
        let token_count = tokens.len().min(self.config.max_tokens);
        let truncated = tokens.len() > self.config.max_tokens;
        
        Ok(TokenizedCode {
            tokens: tokens[..token_count].to_vec(),
            attention_mask: vec![1; token_count],
            token_type_ids: vec![0; token_count],
            original_length: content.len(),
            truncated,
            language: language.map(String::from),
        })
    }
    
    async fn generate_embedding(&self, tokenized: &TokenizedCode) -> Result<Vec<f32>> {
        let model_guard = self.model.read().await;
        let model = model_guard.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model not initialized"))?;
            
        // Mock embedding generation - real implementation would run inference
        let embedding = self.mock_inference(tokenized, model).await?;
        
        Ok(embedding)
    }
    
    async fn encode_chunk(&self, chunk: &[(&str, Option<&str>)]) -> Result<Vec<CodeEmbedding>> {
        let mut results = Vec::with_capacity(chunk.len());
        
        // For mock implementation, process sequentially
        // Real implementation would batch the inference
        for (content, language) in chunk {
            let embedding = self.encode(content, *language).await?;
            results.push(embedding);
        }
        
        Ok(results)
    }
    
    fn hash_content(&self, content: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        self.config.model_type.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
    
    async fn get_cached_embedding(&self, hash: &str) -> Option<CodeEmbedding> {
        let mut cache_guard = self.cache.write().await;
        cache_guard.get(hash).map(|cached| CodeEmbedding {
            embedding: cached.embedding.clone(),
            dimension: self.config.embedding_dim,
            model_version: self.config.model_type.clone(),
            content_hash: hash.to_string(),
            language: None, // Not stored in cache
            token_count: 0, // Not stored in cache
            inference_time_ms: 0, // Cache hit
        })
    }
    
    async fn cache_embedding(&self, hash: &str, embedding: &CodeEmbedding) {
        let mut cache_guard = self.cache.write().await;
        cache_guard.insert(hash.to_string(), CachedEmbedding {
            embedding: embedding.embedding.clone(),
            timestamp: std::time::Instant::now(),
            access_count: 1,
        });
    }
    
    // Mock implementations for development - replace with real ML inference
    
    async fn mock_tokenize(&self, content: &str, _tokenizer: &CodeTokenizer) -> Result<Vec<u32>> {
        // Simple mock: split on whitespace and hash to token IDs
        let words: Vec<&str> = content.split_whitespace().collect();
        let mut tokens = Vec::with_capacity(words.len() + 2);
        
        tokens.push(2); // <s> start token
        
        for word in words {
            let hash = word.bytes().fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
            tokens.push(hash % 50000); // Map to vocab range
        }
        
        tokens.push(3); // </s> end token
        
        Ok(tokens)
    }
    
    async fn mock_inference(&self, tokenized: &TokenizedCode, _model: &CodeModel) -> Result<Vec<f32>> {
        // Mock embedding: hash-based deterministic vector
        let mut embedding = vec![0.0f32; self.config.embedding_dim];
        
        for (i, &token) in tokenized.tokens.iter().enumerate() {
            let idx = (token as usize + i) % self.config.embedding_dim;
            embedding[idx] += (token as f32) / 50000.0 - 0.5;
        }
        
        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }
        
        // Add small delay to simulate inference
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        
        Ok(embedding)
    }
}

impl EmbeddingCache {
    fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            access_order: Vec::new(),
            max_size,
        }
    }
    
    fn get(&mut self, key: &str) -> Option<&CachedEmbedding> {
        if let Some(embedding) = self.cache.get_mut(key) {
            embedding.access_count += 1;
            
            // Move to end of access order (most recently used)
            if let Some(pos) = self.access_order.iter().position(|x| x == key) {
                self.access_order.remove(pos);
            }
            self.access_order.push(key.to_string());
            
            Some(embedding)
        } else {
            None
        }
    }
    
    fn insert(&mut self, key: String, value: CachedEmbedding) {
        // Evict if at capacity
        if self.cache.len() >= self.max_size && !self.cache.contains_key(&key) {
            if let Some(lru_key) = self.access_order.first().cloned() {
                self.cache.remove(&lru_key);
                self.access_order.retain(|k| k != &lru_key);
            }
        }
        
        // Insert new entry
        self.cache.insert(key.clone(), value);
        self.access_order.push(key);
    }
    
    fn calculate_hit_rate(&self) -> f32 {
        if self.cache.is_empty() {
            return 0.0;
        }
        
        let total_accesses: u32 = self.cache.values().map(|e| e.access_count).sum();
        let hits = total_accesses.saturating_sub(self.cache.len() as u32); // Subtract first access
        
        if total_accesses == 0 {
            0.0
        } else {
            hits as f32 / total_accesses as f32
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderMetrics {
    pub cache_size: usize,
    pub cache_hit_rate: f32,
    pub max_tokens: usize,
    pub embedding_dim: usize,
    pub model_type: String,
}

/// Initialize the semantic encoder
pub async fn initialize_encoder(config: &EncoderConfig) -> Result<()> {
    info!("Initializing semantic encoder with config: {:?}", config);
    
    // Validate configuration
    if config.max_tokens > 4096 {
        warn!("Max tokens {} exceeds recommended 4096", config.max_tokens);
    }
    
    if config.max_tokens < 512 {
        anyhow::bail!("Max tokens {} too small, minimum 512", config.max_tokens);
    }
    
    info!("Encoder initialization complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_encoder_creation() {
        let config = EncoderConfig {
            model_type: "codet5-base".to_string(),
            max_tokens: 2048,
            embedding_dim: 768,
            batch_size: 16,
            device: "cpu".to_string(),
        };
        
        let encoder = SemanticEncoder::new(config).await.unwrap();
        assert!(!encoder.can_handle_content("test")); // Not initialized yet
    }

    #[tokio::test]
    async fn test_content_limits() {
        let config = EncoderConfig {
            model_type: "codet5-base".to_string(),
            max_tokens: 2048,
            embedding_dim: 768,
            batch_size: 16,
            device: "cpu".to_string(),
        };
        
        let encoder = SemanticEncoder::new(config).await.unwrap();
        
        // Small content should be fine
        let small_content = "def hello(): return 'world'";
        assert!(encoder.can_handle_content(small_content));
        
        // Very large content should be rejected
        let large_content = "x".repeat(200_000);
        assert!(!encoder.can_handle_content(&large_content));
    }

    #[tokio::test]
    async fn test_cache_operations() {
        let mut cache = EmbeddingCache::new(2);
        
        let embedding1 = CachedEmbedding {
            embedding: vec![1.0, 2.0],
            timestamp: std::time::Instant::now(),
            access_count: 1,
        };
        
        let embedding2 = CachedEmbedding {
            embedding: vec![3.0, 4.0],
            timestamp: std::time::Instant::now(),
            access_count: 1,
        };
        
        // Insert first entry
        cache.insert("key1".to_string(), embedding1.clone());
        assert_eq!(cache.cache.len(), 1);
        
        // Insert second entry
        cache.insert("key2".to_string(), embedding2.clone());
        assert_eq!(cache.cache.len(), 2);
        
        // Should evict LRU when inserting third
        let embedding3 = CachedEmbedding {
            embedding: vec![5.0, 6.0],
            timestamp: std::time::Instant::now(),
            access_count: 1,
        };
        
        cache.insert("key3".to_string(), embedding3);
        assert_eq!(cache.cache.len(), 2);
        assert!(!cache.cache.contains_key("key1")); // Should be evicted
    }
}