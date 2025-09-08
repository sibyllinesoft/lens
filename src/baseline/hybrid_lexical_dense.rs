//! # Hybrid Lexical + Dense Baseline Competitor
//!
//! Implements a hybrid search system combining traditional lexical search (BM25)
//! with dense semantic search (embeddings) optimized for code search.
//!
//! Features:
//! - BM25 lexical scoring for exact term matching
//! - Dense embeddings for semantic similarity
//! - Intelligent weight balancing between lexical and semantic
//! - Code-specific embedding models and techniques

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use super::{BaselineSearcher, SearchResult, ResultMetadata, ScoringBreakdown, BaselineSystemConfig, SystemStatistics, IndexConfig, ScoringConfig};

/// Hybrid search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridConfig {
    /// Weight for lexical (BM25) component
    pub lexical_weight: f32,
    /// Weight for semantic (dense) component  
    pub semantic_weight: f32,
    /// BM25 parameters
    pub bm25_k1: f32,
    pub bm25_b: f32,
    /// Embedding model configuration
    pub embedding_model: String,
    pub embedding_dim: usize,
    /// Semantic search parameters
    pub semantic_top_k: usize,
    pub min_semantic_similarity: f32,
    /// Query processing
    pub query_expansion: bool,
    pub query_rewriting: bool,
    /// Result fusion method
    pub fusion_method: FusionMethod,
    /// Language-specific weights
    pub language_weights: HashMap<String, LanguageWeights>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageWeights {
    pub lexical_boost: f32,
    pub semantic_boost: f32,
    pub keyword_boost: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionMethod {
    LinearCombination,
    RankBasedFusion,
    LearnedFusion,
    AdaptiveWeighting,
}

impl HybridConfig {
    /// Balanced hybrid configuration
    pub fn balanced_hybrid() -> Self {
        let mut language_weights = HashMap::new();
        language_weights.insert("python".to_string(), LanguageWeights {
            lexical_boost: 1.0,
            semantic_boost: 1.2,
            keyword_boost: 1.3,
        });
        language_weights.insert("typescript".to_string(), LanguageWeights {
            lexical_boost: 1.1,
            semantic_boost: 1.0,
            keyword_boost: 1.2,
        });
        language_weights.insert("rust".to_string(), LanguageWeights {
            lexical_boost: 1.2,
            semantic_boost: 0.9,
            keyword_boost: 1.4,
        });
        
        Self {
            lexical_weight: 0.6,
            semantic_weight: 0.4,
            bm25_k1: 1.2,
            bm25_b: 0.75,
            embedding_model: "code-search-net".to_string(),
            embedding_dim: 768,
            semantic_top_k: 1000,
            min_semantic_similarity: 0.1,
            query_expansion: true,
            query_rewriting: false,
            fusion_method: FusionMethod::LinearCombination,
            language_weights,
        }
    }
    
    /// Code-optimized configuration with higher semantic weight
    pub fn code_optimized() -> Self {
        let mut config = Self::balanced_hybrid();
        config.lexical_weight = 0.5;
        config.semantic_weight = 0.5;
        config.query_expansion = true;
        config.query_rewriting = true;
        config.fusion_method = FusionMethod::AdaptiveWeighting;
        config.embedding_model = "unixcoder-base".to_string();
        config
    }
    
    /// High recall configuration
    pub fn high_recall() -> Self {
        let mut config = Self::balanced_hybrid();
        config.lexical_weight = 0.7;
        config.semantic_weight = 0.3;
        config.semantic_top_k = 2000;
        config.min_semantic_similarity = 0.05;
        config.query_expansion = true;
        config
    }
}

/// Document with both lexical and semantic representations
#[derive(Debug, Clone)]
struct HybridDocument {
    file_path: String,
    content: String,
    language: String,
    
    // Lexical representation
    tokens: Vec<String>,
    term_frequencies: HashMap<String, usize>,
    document_length: usize,
    
    // Semantic representation  
    embedding: Vec<f32>,
    function_embeddings: HashMap<String, Vec<f32>>,
    class_embeddings: HashMap<String, Vec<f32>>,
    
    // Metadata
    metadata: HybridDocumentMetadata,
}

#[derive(Debug, Clone)]
struct HybridDocumentMetadata {
    file_size: usize,
    last_modified: Option<chrono::DateTime<chrono::Utc>>,
    functions: Vec<String>,
    classes: Vec<String>,
    complexity_score: f32,
    popularity_score: f32,
}

/// Hybrid search index
struct HybridIndex {
    documents: HashMap<String, HybridDocument>,
    
    // Lexical index (BM25)
    inverted_index: HashMap<String, Vec<LexicalPosting>>,
    document_frequencies: HashMap<String, usize>,
    total_documents: usize,
    average_document_length: f32,
    
    // Semantic index (dense vectors)
    document_embeddings: HashMap<String, Vec<f32>>,
    function_embeddings: HashMap<String, HashMap<String, Vec<f32>>>, // doc_id -> function_name -> embedding
    
    // Index statistics
    index_size_mb: f32,
    build_time_seconds: f32,
}

#[derive(Debug, Clone)]
struct LexicalPosting {
    document_id: String,
    term_frequency: usize,
    positions: Vec<usize>,
}

/// Hybrid search result combining lexical and semantic scores
#[derive(Debug, Clone)]
struct HybridSearchResult {
    document_id: String,
    lexical_score: f32,
    semantic_score: f32,
    combined_score: f32,
    fusion_explanation: String,
    matched_terms: Vec<String>,
    matched_functions: Vec<String>,
    snippet: String,
}

/// Mock embedding service (in production, would use actual embedding models)
struct EmbeddingService {
    model_name: String,
    embedding_dim: usize,
}

impl EmbeddingService {
    fn new(model_name: String, embedding_dim: usize) -> Self {
        Self {
            model_name,
            embedding_dim,
        }
    }
    
    /// Generate embedding for text (mock implementation)
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        // Mock embedding generation - would use actual model in production
        let mut embedding = vec![0.0; self.embedding_dim];
        
        // Simple hash-based mock embedding
        let hash = self.simple_hash(text);
        for (i, val) in embedding.iter_mut().enumerate() {
            *val = ((hash.wrapping_add(i * 17)) % 1000) as f32 / 1000.0 - 0.5;
        }
        
        // Normalize
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }
        
        Ok(embedding)
    }
    
    /// Generate embeddings for code functions
    async fn embed_function(&self, function_signature: &str, function_body: &str) -> Result<Vec<f32>> {
        let combined_text = format!("{} {}", function_signature, function_body);
        self.embed_text(&combined_text).await
    }
    
    /// Calculate cosine similarity between embeddings
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        
        dot_product / (norm_a * norm_b)
    }
    
    fn simple_hash(&self, text: &str) -> u32 {
        text.chars().fold(0u32, |acc, c| acc.wrapping_mul(31).wrapping_add(c as u32))
    }
}

/// Hybrid lexical + dense searcher implementation
pub struct HybridSearcher {
    config: HybridConfig,
    index: Arc<RwLock<HybridIndex>>,
    embedding_service: EmbeddingService,
    statistics: Arc<RwLock<SystemStatistics>>,
    system_name: String,
}

impl HybridSearcher {
    /// Create new hybrid searcher
    pub async fn new(config: HybridConfig) -> Result<Self> {
        let system_name = format!(
            "Hybrid-L{:.1}-S{:.1}-{}", 
            config.lexical_weight, 
            config.semantic_weight,
            config.embedding_model
        );
        
        info!("Creating hybrid searcher: {}", system_name);
        info!("Configuration: lexical_weight={:.1}, semantic_weight={:.1}, model={}", 
              config.lexical_weight, config.semantic_weight, config.embedding_model);
        
        let embedding_service = EmbeddingService::new(
            config.embedding_model.clone(),
            config.embedding_dim
        );
        
        let index = HybridIndex {
            documents: HashMap::new(),
            inverted_index: HashMap::new(),
            document_frequencies: HashMap::new(),
            total_documents: 0,
            average_document_length: 0.0,
            document_embeddings: HashMap::new(),
            function_embeddings: HashMap::new(),
            index_size_mb: 0.0,
            build_time_seconds: 0.0,
        };
        
        let statistics = SystemStatistics {
            queries_processed: 0,
            average_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            cache_hit_rate: 0.0,
            index_size_mb: 0.0,
            memory_usage_mb: 0.0,
        };
        
        Ok(Self {
            config,
            index: Arc::new(RwLock::new(index)),
            embedding_service,
            statistics: Arc::new(RwLock::new(statistics)),
            system_name,
        })
    }
    
    /// Index documents with both lexical and semantic indexing
    pub async fn index_documents(&self, documents: Vec<DocumentToIndex>) -> Result<()> {
        info!("Indexing {} documents with hybrid approach", documents.len());
        let start_time = std::time::Instant::now();
        
        let mut index = self.index.write().await;
        
        for doc_input in documents {
            let hybrid_doc = self.create_hybrid_document(doc_input).await?;
            self.add_to_hybrid_index(&mut index, hybrid_doc).await?;
        }
        
        // Update index statistics
        if !index.documents.is_empty() {
            let total_length: usize = index.documents.values()
                .map(|doc| doc.document_length)
                .sum();
            index.average_document_length = total_length as f32 / index.documents.len() as f32;
        }
        
        index.build_time_seconds = start_time.elapsed().as_secs_f32();
        
        info!("Hybrid indexing complete: {} documents in {:.1}s", 
              index.total_documents, index.build_time_seconds);
        Ok(())
    }
    
    /// Create hybrid document with both lexical and semantic representations
    async fn create_hybrid_document(&self, doc_input: DocumentToIndex) -> Result<HybridDocument> {
        // Tokenize for lexical representation
        let tokens = self.tokenize_content(&doc_input.content).await?;
        let term_frequencies = self.calculate_term_frequencies(&tokens);
        
        // Generate embeddings for semantic representation
        let embedding = self.embedding_service.embed_text(&doc_input.content).await
            .context("Failed to generate document embedding")?;
        
        // Extract and embed functions
        let functions = self.extract_functions(&doc_input.content, &doc_input.language);
        let mut function_embeddings = HashMap::new();
        
        for function_name in &functions {
            if let Some(function_body) = self.extract_function_body(&doc_input.content, function_name) {
                let func_embedding = self.embedding_service.embed_function(function_name, &function_body).await
                    .context("Failed to generate function embedding")?;
                function_embeddings.insert(function_name.clone(), func_embedding);
            }
        }
        
        // Extract and embed classes  
        let classes = self.extract_classes(&doc_input.content, &doc_input.language);
        let mut class_embeddings = HashMap::new();
        
        for class_name in &classes {
            if let Some(class_body) = self.extract_class_body(&doc_input.content, class_name) {
                let class_embedding = self.embedding_service.embed_text(&class_body).await
                    .context("Failed to generate class embedding")?;
                class_embeddings.insert(class_name.clone(), class_embedding);
            }
        }
        
        // Calculate metadata
        let metadata = HybridDocumentMetadata {
            file_size: doc_input.content.len(),
            last_modified: doc_input.last_modified,
            functions: functions,
            classes: classes,
            complexity_score: self.calculate_complexity_score(&tokens),
            popularity_score: 1.0, // Would be calculated from usage statistics
        };
        
        Ok(HybridDocument {
            file_path: doc_input.file_path,
            content: doc_input.content,
            language: doc_input.language,
            tokens,
            term_frequencies,
            document_length: tokens.len(),
            embedding,
            function_embeddings,
            class_embeddings,
            metadata,
        })
    }
    
    /// Add document to hybrid index
    async fn add_to_hybrid_index(&self, index: &mut HybridIndex, document: HybridDocument) -> Result<()> {
        let doc_id = document.file_path.clone();
        
        // Add to lexical index
        for (term, &frequency) in &document.term_frequencies {
            let positions = self.find_term_positions(&document.tokens, term);
            
            let posting = LexicalPosting {
                document_id: doc_id.clone(),
                term_frequency: frequency,
                positions,
            };
            
            index.inverted_index.entry(term.clone()).or_insert_with(Vec::new).push(posting);
            *index.document_frequencies.entry(term.clone()).or_insert(0) += 1;
        }
        
        // Add to semantic index
        index.document_embeddings.insert(doc_id.clone(), document.embedding.clone());
        
        if !document.function_embeddings.is_empty() {
            index.function_embeddings.insert(doc_id.clone(), document.function_embeddings.clone());
        }
        
        // Add to document collection
        index.documents.insert(doc_id, document);
        index.total_documents += 1;
        
        Ok(())
    }
    
    /// Execute hybrid search combining lexical and semantic results
    async fn execute_hybrid_search(&self, query: &str, intent: &str, language: &str, max_results: usize) -> Result<Vec<HybridSearchResult>> {
        let index = self.index.read().await;
        
        // Process query
        let processed_query = self.process_query(query).await?;
        debug!("Processed query: {:?}", processed_query);
        
        // Get lexical results (BM25)
        let lexical_results = self.get_lexical_results(&index, &processed_query, max_results * 2).await?;
        debug!("Found {} lexical results", lexical_results.len());
        
        // Get semantic results (embedding similarity)  
        let semantic_results = self.get_semantic_results(&index, &processed_query.original, intent, language, max_results * 2).await?;
        debug!("Found {} semantic results", semantic_results.len());
        
        // Fuse results using configured method
        let fused_results = self.fuse_results(lexical_results, semantic_results, &processed_query).await?;
        
        // Limit to requested number of results
        let mut final_results = fused_results;
        final_results.truncate(max_results);
        
        Ok(final_results)
    }
    
    /// Process and expand query
    async fn process_query(&self, query: &str) -> Result<ProcessedQuery> {
        let mut processed = ProcessedQuery {
            original: query.to_string(),
            terms: Vec::new(),
            expanded_terms: Vec::new(),
            embedding: Vec::new(),
        };
        
        // Tokenize query
        processed.terms = query.to_lowercase()
            .split_whitespace()
            .filter(|term| term.len() >= 2)
            .map(|term| term.to_string())
            .collect();
        
        // Query expansion (if enabled)
        if self.config.query_expansion {
            processed.expanded_terms = self.expand_query_terms(&processed.terms).await?;
        }
        
        // Generate query embedding
        processed.embedding = self.embedding_service.embed_text(query).await
            .context("Failed to generate query embedding")?;
        
        Ok(processed)
    }
    
    /// Get lexical search results using BM25
    async fn get_lexical_results(&self, index: &HybridIndex, query: &ProcessedQuery, max_results: usize) -> Result<Vec<LexicalResult>> {
        let mut lexical_results = Vec::new();
        
        // Find candidate documents
        let mut candidates: HashMap<String, f32> = HashMap::new();
        
        let all_terms = if self.config.query_expansion && !query.expanded_terms.is_empty() {
            let mut terms = query.terms.clone();
            terms.extend(query.expanded_terms.clone());
            terms
        } else {
            query.terms.clone()
        };
        
        for term in &all_terms {
            if let Some(postings) = index.inverted_index.get(term) {
                for posting in postings {
                    let doc_id = &posting.document_id;
                    
                    if let Some(document) = index.documents.get(doc_id) {
                        // Calculate BM25 score for this term
                        let tf = posting.term_frequency as f32;
                        let df = index.document_frequencies.get(term).copied().unwrap_or(1) as f32;
                        
                        let idf = ((index.total_documents as f32 - df + 0.5) / (df + 0.5)).ln();
                        let tf_component = tf * (self.config.bm25_k1 + 1.0) / 
                            (tf + self.config.bm25_k1 * (1.0 - self.config.bm25_b + 
                             self.config.bm25_b * document.document_length as f32 / index.average_document_length));
                        
                        let term_score = idf * tf_component;
                        *candidates.entry(doc_id.clone()).or_insert(0.0) += term_score;
                    }
                }
            }
        }
        
        // Convert to results and sort
        for (doc_id, score) in candidates {
            lexical_results.push(LexicalResult {
                document_id: doc_id,
                score,
            });
        }
        
        lexical_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        lexical_results.truncate(max_results);
        
        Ok(lexical_results)
    }
    
    /// Get semantic search results using embedding similarity
    async fn get_semantic_results(&self, index: &HybridIndex, query: &str, _intent: &str, _language: &str, max_results: usize) -> Result<Vec<SemanticResult>> {
        let query_embedding = &self.embedding_service.embed_text(query).await?;
        let mut semantic_results = Vec::new();
        
        // Calculate similarities with all document embeddings
        for (doc_id, doc_embedding) in &index.document_embeddings {
            let similarity = self.embedding_service.cosine_similarity(query_embedding, doc_embedding);
            
            if similarity >= self.config.min_semantic_similarity {
                semantic_results.push(SemanticResult {
                    document_id: doc_id.clone(),
                    similarity,
                });
            }
        }
        
        // Also check function-level similarities for more precise matching
        for (doc_id, function_embeddings) in &index.function_embeddings {
            for (_function_name, function_embedding) in function_embeddings {
                let similarity = self.embedding_service.cosine_similarity(query_embedding, function_embedding);
                
                if similarity >= self.config.min_semantic_similarity {
                    // Boost function-level matches
                    let boosted_similarity = similarity * 1.2;
                    
                    if let Some(existing) = semantic_results.iter_mut().find(|r| r.document_id == *doc_id) {
                        existing.similarity = existing.similarity.max(boosted_similarity);
                    } else {
                        semantic_results.push(SemanticResult {
                            document_id: doc_id.clone(),
                            similarity: boosted_similarity,
                        });
                    }
                }
            }
        }
        
        // Sort by similarity and limit results
        semantic_results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        semantic_results.truncate(max_results);
        
        Ok(semantic_results)
    }
    
    /// Fuse lexical and semantic results
    async fn fuse_results(&self, lexical: Vec<LexicalResult>, semantic: Vec<SemanticResult>, query: &ProcessedQuery) -> Result<Vec<HybridSearchResult>> {
        let mut fused = Vec::new();
        let index = self.index.read().await;
        
        // Collect all unique documents
        let mut all_docs = HashMap::new();
        
        // Add lexical results
        for lexical_result in lexical {
            all_docs.insert(lexical_result.document_id.clone(), (Some(lexical_result.score), None));
        }
        
        // Add semantic results
        for semantic_result in semantic {
            let entry = all_docs.entry(semantic_result.document_id.clone()).or_insert((None, None));
            entry.1 = Some(semantic_result.similarity);
        }
        
        // Calculate combined scores using fusion method
        for (doc_id, (lexical_score, semantic_score)) in all_docs {
            let final_lexical = lexical_score.unwrap_or(0.0);
            let final_semantic = semantic_score.unwrap_or(0.0);
            
            let combined_score = match self.config.fusion_method {
                FusionMethod::LinearCombination => {
                    final_lexical * self.config.lexical_weight + 
                    final_semantic * self.config.semantic_weight
                },
                FusionMethod::AdaptiveWeighting => {
                    // Adapt weights based on query characteristics
                    let semantic_weight = if query.terms.len() > 3 { 0.6 } else { 0.4 };
                    let lexical_weight = 1.0 - semantic_weight;
                    final_lexical * lexical_weight + final_semantic * semantic_weight
                },
                _ => {
                    // Default to linear combination
                    final_lexical * self.config.lexical_weight + 
                    final_semantic * self.config.semantic_weight
                }
            };
            
            if let Some(document) = index.documents.get(&doc_id) {
                let snippet = self.generate_snippet(document, query);
                let matched_terms = self.find_matched_terms(document, &query.terms);
                let matched_functions = self.find_matched_functions(document, query);
                let fusion_explanation = format!("L:{:.3} S:{:.3}", final_lexical, final_semantic);
                
                fused.push(HybridSearchResult {
                    document_id: doc_id,
                    lexical_score: final_lexical,
                    semantic_score: final_semantic,
                    combined_score,
                    fusion_explanation,
                    matched_terms,
                    matched_functions,
                    snippet,
                });
            }
        }
        
        // Sort by combined score
        fused.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());
        
        Ok(fused)
    }
    
    // Helper methods
    
    async fn tokenize_content(&self, content: &str) -> Result<Vec<String>> {
        let tokens = content.to_lowercase()
            .split_whitespace()
            .filter(|token| token.len() >= 2)
            .map(|token| {
                token.chars()
                    .filter(|c| c.is_alphanumeric() || *c == '_')
                    .collect()
            })
            .filter(|token: &String| !token.is_empty())
            .collect();
        
        Ok(tokens)
    }
    
    fn calculate_term_frequencies(&self, tokens: &[String]) -> HashMap<String, usize> {
        let mut frequencies = HashMap::new();
        for token in tokens {
            *frequencies.entry(token.clone()).or_insert(0) += 1;
        }
        frequencies
    }
    
    fn find_term_positions(&self, tokens: &[String], term: &str) -> Vec<usize> {
        tokens.iter()
            .enumerate()
            .filter_map(|(i, token)| if token == term { Some(i) } else { None })
            .collect()
    }
    
    fn extract_functions(&self, content: &str, language: &str) -> Vec<String> {
        let mut functions = Vec::new();
        
        // Simple regex-based function extraction (would use proper parsers in production)
        let function_patterns = match language {
            "rust" => vec![r"fn\s+(\w+)"],
            "typescript" | "javascript" => vec![r"function\s+(\w+)", r"(\w+)\s*\(.*\)\s*=>"],
            "python" => vec![r"def\s+(\w+)"],
            _ => vec![r"function\s+(\w+)", r"def\s+(\w+)", r"fn\s+(\w+)"],
        };
        
        for pattern in function_patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                for captures in regex.captures_iter(content) {
                    if let Some(function_name) = captures.get(1) {
                        functions.push(function_name.as_str().to_string());
                    }
                }
            }
        }
        
        functions
    }
    
    fn extract_classes(&self, content: &str, language: &str) -> Vec<String> {
        let mut classes = Vec::new();
        
        let class_patterns = match language {
            "rust" => vec![r"struct\s+(\w+)", r"enum\s+(\w+)", r"trait\s+(\w+)"],
            "typescript" | "javascript" => vec![r"class\s+(\w+)", r"interface\s+(\w+)"],
            "python" => vec![r"class\s+(\w+)"],
            _ => vec![r"class\s+(\w+)", r"struct\s+(\w+)", r"interface\s+(\w+)"],
        };
        
        for pattern in class_patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                for captures in regex.captures_iter(content) {
                    if let Some(class_name) = captures.get(1) {
                        classes.push(class_name.as_str().to_string());
                    }
                }
            }
        }
        
        classes
    }
    
    fn extract_function_body(&self, content: &str, _function_name: &str) -> Option<String> {
        // Simplified - would need proper parsing for accurate extraction
        Some(content.lines().take(10).collect::<Vec<_>>().join("\\n"))
    }
    
    fn extract_class_body(&self, content: &str, _class_name: &str) -> Option<String> {
        // Simplified - would need proper parsing for accurate extraction
        Some(content.lines().take(20).collect::<Vec<_>>().join("\\n"))
    }
    
    fn calculate_complexity_score(&self, tokens: &[String]) -> f32 {
        // Simple complexity metric based on token diversity
        let unique_tokens = tokens.iter().collect::<std::collections::HashSet<_>>().len();
        (unique_tokens as f32) / (tokens.len() as f32).max(1.0)
    }
    
    async fn expand_query_terms(&self, terms: &[String]) -> Result<Vec<String>> {
        // Simple query expansion - would use proper expansion techniques
        let mut expanded = Vec::new();
        
        for term in terms {
            // Add common programming synonyms
            match term.as_str() {
                "function" => expanded.extend(vec!["method".to_string(), "procedure".to_string()]),
                "class" => expanded.extend(vec!["struct".to_string(), "type".to_string()]),
                "variable" => expanded.extend(vec!["var".to_string(), "field".to_string()]),
                _ => {}
            }
        }
        
        Ok(expanded)
    }
    
    fn generate_snippet(&self, document: &HybridDocument, query: &ProcessedQuery) -> String {
        let lines: Vec<&str> = document.content.lines().collect();
        
        // Find best snippet around query terms
        for (i, line) in lines.iter().enumerate() {
            let line_lower = line.to_lowercase();
            if query.terms.iter().any(|term| line_lower.contains(term)) {
                let start = i.saturating_sub(2);
                let end = (i + 3).min(lines.len());
                return lines[start..end].join("\\n");
            }
        }
        
        // Fallback: return first few lines
        lines.iter().take(3).cloned().collect::<Vec<_>>().join("\\n")
    }
    
    fn find_matched_terms(&self, document: &HybridDocument, query_terms: &[String]) -> Vec<String> {
        let mut matched = Vec::new();
        for term in query_terms {
            if document.term_frequencies.contains_key(term) {
                matched.push(term.clone());
            }
        }
        matched
    }
    
    fn find_matched_functions(&self, document: &HybridDocument, query: &ProcessedQuery) -> Vec<String> {
        let mut matched = Vec::new();
        for function_name in &document.metadata.functions {
            if query.terms.iter().any(|term| function_name.to_lowercase().contains(term)) {
                matched.push(function_name.clone());
            }
        }
        matched
    }
}

// Supporting types

#[derive(Debug)]
struct ProcessedQuery {
    original: String,
    terms: Vec<String>,
    expanded_terms: Vec<String>,
    embedding: Vec<f32>,
}

#[derive(Debug)]
struct LexicalResult {
    document_id: String,
    score: f32,
}

#[derive(Debug)]
struct SemanticResult {
    document_id: String,
    similarity: f32,
}

#[async_trait::async_trait]
impl BaselineSearcher for HybridSearcher {
    fn system_name(&self) -> &str {
        &self.system_name
    }
    
    async fn search(&self, query: &str, intent: &str, language: &str, max_results: usize) -> Result<Vec<SearchResult>> {
        let start_time = std::time::Instant::now();
        
        let hybrid_results = self.execute_hybrid_search(query, intent, language, max_results).await
            .context("Hybrid search execution failed")?;
        
        let latency_ms = start_time.elapsed().as_millis() as f32;
        
        // Update statistics
        {
            let mut stats = self.statistics.write().await;
            stats.queries_processed += 1;
            
            let n = stats.queries_processed as f32;
            stats.average_latency_ms = ((n - 1.0) * stats.average_latency_ms + latency_ms) / n;
            
            if latency_ms > stats.p95_latency_ms {
                stats.p95_latency_ms = latency_ms;
            }
            if latency_ms > stats.p99_latency_ms {
                stats.p99_latency_ms = latency_ms;
            }
        }
        
        // Convert to standard search results
        let index = self.index.read().await;
        let mut results = Vec::new();
        
        for (rank, result) in hybrid_results.into_iter().enumerate() {
            if let Some(doc) = index.documents.get(&result.document_id) {
                let metadata = ResultMetadata {
                    line_number: None,
                    function_name: result.matched_functions.first().cloned(),
                    class_name: doc.metadata.classes.first().cloned(),
                    language: doc.language.clone(),
                    file_size: doc.metadata.file_size,
                    last_modified: doc.metadata.last_modified,
                    scoring_breakdown: ScoringBreakdown {
                        lexical_score: result.lexical_score,
                        semantic_score: Some(result.semantic_score),
                        proximity_score: None,
                        recency_score: None,
                        popularity_score: Some(doc.metadata.popularity_score),
                        final_score: result.combined_score,
                    },
                };
                
                results.push(SearchResult {
                    file_path: result.document_id,
                    score: result.combined_score,
                    snippet: result.snippet,
                    rank: rank + 1,
                    metadata,
                });
            }
        }
        
        debug!("Hybrid search completed: {} results in {:.1}ms", results.len(), latency_ms);
        Ok(results)
    }
    
    fn get_config(&self) -> BaselineSystemConfig {
        let mut parameters = HashMap::new();
        parameters.insert("lexical_weight".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(self.config.lexical_weight as f64).unwrap()));
        parameters.insert("semantic_weight".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(self.config.semantic_weight as f64).unwrap()));
        parameters.insert("embedding_model".to_string(), serde_json::Value::String(self.config.embedding_model.clone()));
        parameters.insert("fusion_method".to_string(), serde_json::Value::String(format!("{:?}", self.config.fusion_method)));
        
        BaselineSystemConfig {
            system_name: self.system_name.clone(),
            version: "1.0.0".to_string(),
            parameters,
            index_config: IndexConfig {
                tokenizer: "hybrid_lexical_semantic".to_string(),
                stemming: false,
                stop_words: false,
                n_grams: vec![1],
                case_sensitive: false,
                special_characters: true,
            },
            scoring_config: ScoringConfig {
                bm25_k1: self.config.bm25_k1,
                bm25_b: self.config.bm25_b,
                proximity_weight: None,
                semantic_weight: Some(self.config.semantic_weight),
                recency_weight: None,
                normalization: "hybrid_fusion".to_string(),
            },
        }
    }
    
    async fn get_statistics(&self) -> Result<SystemStatistics> {
        Ok(self.statistics.read().await.clone())
    }
}

/// Document input for indexing
#[derive(Debug, Clone)]
pub struct DocumentToIndex {
    pub file_path: String,
    pub content: String,
    pub language: String,
    pub last_modified: Option<chrono::DateTime<chrono::Utc>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hybrid_searcher_creation() {
        let config = HybridConfig::balanced_hybrid();
        let searcher = HybridSearcher::new(config).await.unwrap();
        
        assert!(searcher.system_name().contains("Hybrid"));
        assert_eq!(searcher.config.lexical_weight, 0.6);
        assert_eq!(searcher.config.semantic_weight, 0.4);
    }

    #[tokio::test]
    async fn test_embedding_service() {
        let embedding_service = EmbeddingService::new("test-model".to_string(), 128);
        
        let embedding1 = embedding_service.embed_text("async function handler").await.unwrap();
        let embedding2 = embedding_service.embed_text("async function handler").await.unwrap();
        let embedding3 = embedding_service.embed_text("different text").await.unwrap();
        
        assert_eq!(embedding1.len(), 128);
        assert_eq!(embedding1, embedding2); // Same text = same embedding
        assert_ne!(embedding1, embedding3); // Different text = different embedding
        
        let similarity = embedding_service.cosine_similarity(&embedding1, &embedding2);
        assert!((similarity - 1.0).abs() < 1e-6); // Self-similarity should be ~1.0
    }

    #[tokio::test]
    async fn test_query_processing() {
        let config = HybridConfig::balanced_hybrid();
        let searcher = HybridSearcher::new(config).await.unwrap();
        
        let query = "async function handler";
        let processed = searcher.process_query(query).await.unwrap();
        
        assert_eq!(processed.original, "async function handler");
        assert_eq!(processed.terms.len(), 3);
        assert!(processed.terms.contains(&"async".to_string()));
        assert!(processed.terms.contains(&"function".to_string()));
        assert!(processed.terms.contains(&"handler".to_string()));
        assert_eq!(processed.embedding.len(), searcher.config.embedding_dim);
    }

    #[tokio::test]
    async fn test_function_extraction() {
        let config = HybridConfig::balanced_hybrid();
        let searcher = HybridSearcher::new(config).await.unwrap();
        
        let content = "function handleRequest() {\\n  return response;\\n}\\n\\nconst process = async () => {\\n  await fetch();\\n}";
        let functions = searcher.extract_functions(content, "typescript");
        
        assert!(!functions.is_empty());
        assert!(functions.contains(&"handleRequest".to_string()) || functions.contains(&"process".to_string()));
    }

    #[test]
    fn test_config_variants() {
        let balanced = HybridConfig::balanced_hybrid();
        assert_eq!(balanced.lexical_weight, 0.6);
        assert_eq!(balanced.semantic_weight, 0.4);
        
        let code_optimized = HybridConfig::code_optimized();
        assert_eq!(code_optimized.lexical_weight, 0.5);
        assert_eq!(code_optimized.semantic_weight, 0.5);
        assert!(code_optimized.query_expansion);
        assert!(code_optimized.query_rewriting);
        
        let high_recall = HybridConfig::high_recall();
        assert_eq!(high_recall.lexical_weight, 0.7);
        assert_eq!(high_recall.semantic_weight, 0.3);
        assert_eq!(high_recall.semantic_top_k, 2000);
    }
}