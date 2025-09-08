//! # BM25 + Proximity Baseline Competitor
//!
//! Implements a strong BM25 baseline with proximity-aware scoring
//! optimized for code search scenarios.
//!
//! Features:
//! - Traditional BM25 scoring with code-specific tuning
//! - Term proximity awareness for better phrase matching
//! - Language-specific tokenization and weighting
//! - Optimized for ≥3pp margin requirement

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

use super::{BaselineSearcher, SearchResult, ResultMetadata, ScoringBreakdown, BaselineSystemConfig, SystemStatistics, IndexConfig, ScoringConfig};

/// BM25 + Proximity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProximityConfig {
    /// BM25 k1 parameter (term frequency saturation)
    pub bm25_k1: f32,
    /// BM25 b parameter (length normalization)
    pub bm25_b: f32,
    /// Proximity boost weight
    pub proximity_weight: f32,
    /// Maximum proximity distance to consider
    pub max_proximity_distance: usize,
    /// Exact phrase match bonus
    pub phrase_match_bonus: f32,
    /// Language-specific boosts
    pub language_boosts: HashMap<String, f32>,
    /// Special token weights (keywords, operators, etc.)
    pub special_token_weights: HashMap<String, f32>,
    /// Minimum term length for indexing
    pub min_term_length: usize,
    /// Enable positional indexing
    pub positional_indexing: bool,
}

impl ProximityConfig {
    /// Optimized configuration for code search
    pub fn optimized_for_code() -> Self {
        let mut language_boosts = HashMap::new();
        language_boosts.insert("function".to_string(), 1.5);
        language_boosts.insert("class".to_string(), 1.4);
        language_boosts.insert("interface".to_string(), 1.3);
        language_boosts.insert("struct".to_string(), 1.3);
        language_boosts.insert("trait".to_string(), 1.2);
        language_boosts.insert("impl".to_string(), 1.2);
        
        let mut special_tokens = HashMap::new();
        special_tokens.insert("async".to_string(), 1.3);
        special_tokens.insert("await".to_string(), 1.3);
        special_tokens.insert("const".to_string(), 1.2);
        special_tokens.insert("static".to_string(), 1.2);
        special_tokens.insert("public".to_string(), 1.1);
        special_tokens.insert("private".to_string(), 1.1);
        special_tokens.insert("protected".to_string(), 1.1);
        
        Self {
            bm25_k1: 1.2, // Standard BM25 parameter
            bm25_b: 0.75, // Standard length normalization
            proximity_weight: 0.3, // Moderate proximity influence
            max_proximity_distance: 50, // Within ~50 tokens
            phrase_match_bonus: 1.5, // 50% bonus for exact phrases
            language_boosts,
            special_token_weights: special_tokens,
            min_term_length: 2,
            positional_indexing: true,
        }
    }
    
    /// High precision configuration
    pub fn high_precision() -> Self {
        let mut config = Self::optimized_for_code();
        config.bm25_k1 = 1.0; // Less term frequency saturation
        config.proximity_weight = 0.4; // Higher proximity weight
        config.phrase_match_bonus = 2.0; // Higher phrase bonus
        config.max_proximity_distance = 30; // Stricter proximity
        config
    }
}

/// Document in the BM25 index
#[derive(Debug, Clone)]
struct IndexedDocument {
    file_path: String,
    content: String,
    tokens: Vec<Token>,
    term_frequencies: HashMap<String, usize>,
    document_length: usize,
    language: String,
    metadata: DocumentMetadata,
}

#[derive(Debug, Clone)]
struct Token {
    term: String,
    position: usize,
    line_number: usize,
    in_function: Option<String>,
    in_class: Option<String>,
    token_type: TokenType,
}

#[derive(Debug, Clone)]
enum TokenType {
    Keyword,
    Identifier,
    Literal,
    Operator,
    Comment,
    String,
}

#[derive(Debug, Clone)]
struct DocumentMetadata {
    file_size: usize,
    last_modified: Option<chrono::DateTime<chrono::Utc>>,
    language: String,
    functions: Vec<String>,
    classes: Vec<String>,
}

/// BM25 + Proximity index
struct ProximityIndex {
    documents: HashMap<String, IndexedDocument>,
    inverted_index: HashMap<String, Vec<PostingListEntry>>,
    document_frequencies: HashMap<String, usize>,
    total_documents: usize,
    average_document_length: f32,
}

#[derive(Debug, Clone)]
struct PostingListEntry {
    document_id: String,
    term_frequency: usize,
    positions: Vec<usize>,
    score_boost: f32,
}

/// BM25 + Proximity search result
struct ProximitySearchResult {
    document_id: String,
    bm25_score: f32,
    proximity_score: f32,
    combined_score: f32,
    matched_positions: Vec<usize>,
    snippet: String,
}

/// BM25 + Proximity searcher implementation
pub struct BM25ProximitySearcher {
    config: ProximityConfig,
    index: Arc<RwLock<ProximityIndex>>,
    statistics: Arc<RwLock<SystemStatistics>>,
    system_name: String,
}

impl BM25ProximitySearcher {
    /// Create new BM25 + Proximity searcher
    pub async fn new(config: ProximityConfig) -> Result<Self> {
        let system_name = format!("BM25+Proximity-k1:{}-prox:{}", config.bm25_k1, config.proximity_weight);
        
        info!("Creating BM25+Proximity searcher: {}", system_name);
        info!("Configuration: k1={}, b={}, proximity_weight={}", 
              config.bm25_k1, config.bm25_b, config.proximity_weight);
        
        let index = ProximityIndex {
            documents: HashMap::new(),
            inverted_index: HashMap::new(),
            document_frequencies: HashMap::new(),
            total_documents: 0,
            average_document_length: 0.0,
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
            statistics: Arc::new(RwLock::new(statistics)),
            system_name,
        })
    }
    
    /// Index documents for searching
    pub async fn index_documents(&self, documents: Vec<DocumentToIndex>) -> Result<()> {
        info!("Indexing {} documents", documents.len());
        let mut index = self.index.write().await;
        
        for doc_input in documents {
            let indexed_doc = self.process_document(doc_input).await?;
            self.add_document_to_index(&mut index, indexed_doc).await?;
        }
        
        // Calculate average document length
        if !index.documents.is_empty() {
            let total_length: usize = index.documents.values()
                .map(|doc| doc.document_length)
                .sum();
            index.average_document_length = total_length as f32 / index.documents.len() as f32;
        }
        
        info!("Indexing complete: {} documents, avg length: {:.1} tokens", 
              index.total_documents, index.average_document_length);
        Ok(())
    }
    
    /// Process document into indexed format
    async fn process_document(&self, doc_input: DocumentToIndex) -> Result<IndexedDocument> {
        // Tokenize document content
        let tokens = self.tokenize_content(&doc_input.content, &doc_input.language).await?;
        
        // Calculate term frequencies
        let mut term_frequencies = HashMap::new();
        for token in &tokens {
            *term_frequencies.entry(token.term.clone()).or_insert(0) += 1;
        }
        
        // Extract metadata
        let metadata = DocumentMetadata {
            file_size: doc_input.content.len(),
            last_modified: doc_input.last_modified,
            language: doc_input.language.clone(),
            functions: self.extract_functions(&tokens),
            classes: self.extract_classes(&tokens),
        };
        
        Ok(IndexedDocument {
            file_path: doc_input.file_path,
            content: doc_input.content,
            document_length: tokens.len(),
            tokens,
            term_frequencies,
            language: doc_input.language,
            metadata,
        })
    }
    
    /// Tokenize content with language awareness
    async fn tokenize_content(&self, content: &str, language: &str) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();
        let mut position = 0;
        
        // Simple tokenization - would use proper language-specific tokenizer
        let lines: Vec<&str> = content.lines().collect();
        
        for (line_idx, line) in lines.iter().enumerate() {
            let words: Vec<&str> = line.split_whitespace().collect();
            
            for word in words {
                if word.len() >= self.config.min_term_length {
                    let cleaned = self.clean_token(word);
                    if !cleaned.is_empty() {
                        let token_type = self.classify_token(&cleaned, language);
                        
                        tokens.push(Token {
                            term: cleaned.to_lowercase(),
                            position,
                            line_number: line_idx + 1,
                            in_function: None, // Would be extracted with proper parsing
                            in_class: None,    // Would be extracted with proper parsing
                            token_type,
                        });
                    }
                }
                position += 1;
            }
        }
        
        Ok(tokens)
    }
    
    /// Clean and normalize tokens
    fn clean_token(&self, token: &str) -> String {
        token.chars()
            .filter(|c| c.is_alphanumeric() || *c == '_')
            .collect()
    }
    
    /// Classify token type for scoring
    fn classify_token(&self, token: &str, language: &str) -> TokenType {
        // Language-specific keyword detection
        let keywords = match language {
            "rust" => vec!["fn", "struct", "impl", "trait", "enum", "mod", "pub", "async", "await"],
            "typescript" | "javascript" => vec!["function", "class", "interface", "async", "await", "const", "let"],
            "python" => vec!["def", "class", "async", "await", "import", "from"],
            _ => vec!["function", "class", "async", "await"],
        };
        
        if keywords.contains(&token) {
            TokenType::Keyword
        } else if token.chars().all(|c| c.is_digit(10) || c == '.') {
            TokenType::Literal
        } else if "+-*/=<>!&|".contains(token) {
            TokenType::Operator
        } else {
            TokenType::Identifier
        }
    }
    
    /// Extract function names from tokens
    fn extract_functions(&self, tokens: &[Token]) -> Vec<String> {
        let mut functions = Vec::new();
        
        for window in tokens.windows(2) {
            if matches!(window[0].token_type, TokenType::Keyword) && 
               (window[0].term == "fn" || window[0].term == "function" || window[0].term == "def") {
                if matches!(window[1].token_type, TokenType::Identifier) {
                    functions.push(window[1].term.clone());
                }
            }
        }
        
        functions
    }
    
    /// Extract class names from tokens
    fn extract_classes(&self, tokens: &[Token]) -> Vec<String> {
        let mut classes = Vec::new();
        
        for window in tokens.windows(2) {
            if matches!(window[0].token_type, TokenType::Keyword) && 
               (window[0].term == "class" || window[0].term == "struct" || window[0].term == "interface") {
                if matches!(window[1].token_type, TokenType::Identifier) {
                    classes.push(window[1].term.clone());
                }
            }
        }
        
        classes
    }
    
    /// Add document to inverted index
    async fn add_document_to_index(&self, index: &mut ProximityIndex, document: IndexedDocument) -> Result<()> {
        let doc_id = document.file_path.clone();
        
        // Add to document collection
        index.documents.insert(doc_id.clone(), document.clone());
        index.total_documents += 1;
        
        // Update inverted index
        for (term, &frequency) in &document.term_frequencies {
            // Collect positions for this term
            let positions: Vec<usize> = document.tokens.iter()
                .enumerate()
                .filter_map(|(i, token)| {
                    if token.term == *term {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect();
            
            // Calculate score boost based on special tokens and language context
            let score_boost = self.calculate_score_boost(term, &document.language);
            
            let posting = PostingListEntry {
                document_id: doc_id.clone(),
                term_frequency: frequency,
                positions,
                score_boost,
            };
            
            index.inverted_index.entry(term.clone()).or_insert_with(Vec::new).push(posting);
            *index.document_frequencies.entry(term.clone()).or_insert(0) += 1;
        }
        
        Ok(())
    }
    
    /// Calculate score boost for special tokens
    fn calculate_score_boost(&self, term: &str, language: &str) -> f32 {
        let mut boost = 1.0;
        
        // Special token weights
        if let Some(&weight) = self.config.special_token_weights.get(term) {
            boost *= weight;
        }
        
        // Language-specific boosts
        if let Some(&lang_boost) = self.config.language_boosts.get(term) {
            boost *= lang_boost;
        }
        
        boost
    }
    
    /// Execute BM25 + Proximity search
    async fn execute_search(&self, query: &str, max_results: usize) -> Result<Vec<ProximitySearchResult>> {
        let index = self.index.read().await;
        
        // Parse and process query
        let query_terms = self.parse_query(query).await?;
        debug!("Query terms: {:?}", query_terms);
        
        if query_terms.is_empty() {
            return Ok(Vec::new());
        }
        
        // Find candidate documents
        let candidates = self.find_candidate_documents(&index, &query_terms).await?;
        
        // Score documents with BM25 + Proximity
        let mut scored_results = Vec::new();
        
        for (doc_id, doc) in candidates {
            let bm25_score = self.calculate_bm25_score(&index, &doc, &query_terms);
            let proximity_score = self.calculate_proximity_score(&doc, &query_terms);
            let combined_score = bm25_score + (proximity_score * self.config.proximity_weight);
            
            let snippet = self.generate_snippet(&doc, &query_terms);
            let matched_positions = self.find_matched_positions(&doc, &query_terms);
            
            scored_results.push(ProximitySearchResult {
                document_id: doc_id,
                bm25_score,
                proximity_score,
                combined_score,
                matched_positions,
                snippet,
            });
        }
        
        // Sort by combined score and take top results
        scored_results.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());
        scored_results.truncate(max_results);
        
        Ok(scored_results)
    }
    
    /// Parse query into terms
    async fn parse_query(&self, query: &str) -> Result<Vec<String>> {
        let terms = query.to_lowercase()
            .split_whitespace()
            .filter(|term| term.len() >= self.config.min_term_length)
            .map(|term| self.clean_token(term))
            .filter(|term| !term.is_empty())
            .collect();
        
        Ok(terms)
    }
    
    /// Find documents containing query terms
    async fn find_candidate_documents(
        &self, 
        index: &ProximityIndex, 
        query_terms: &[String]
    ) -> Result<HashMap<String, &IndexedDocument>> {
        let mut candidates = HashMap::new();
        
        for term in query_terms {
            if let Some(posting_list) = index.inverted_index.get(term) {
                for posting in posting_list {
                    if let Some(doc) = index.documents.get(&posting.document_id) {
                        candidates.insert(posting.document_id.clone(), doc);
                    }
                }
            }
        }
        
        Ok(candidates)
    }
    
    /// Calculate BM25 score for document
    fn calculate_bm25_score(&self, index: &ProximityIndex, document: &IndexedDocument, query_terms: &[String]) -> f32 {
        let mut score = 0.0;
        
        for term in query_terms {
            if let Some(tf) = document.term_frequencies.get(term) {
                let tf = *tf as f32;
                let df = index.document_frequencies.get(term).copied().unwrap_or(0) as f32;
                
                // IDF calculation
                let idf = ((index.total_documents as f32 - df + 0.5) / (df + 0.5)).ln();
                
                // BM25 term score
                let numerator = tf * (self.config.bm25_k1 + 1.0);
                let denominator = tf + self.config.bm25_k1 * (1.0 - self.config.bm25_b + 
                    self.config.bm25_b * document.document_length as f32 / index.average_document_length);
                
                let term_score = idf * (numerator / denominator);
                
                // Apply score boost for special tokens
                let boost = self.calculate_score_boost(term, &document.language);
                
                score += term_score * boost;
            }
        }
        
        score
    }
    
    /// Calculate proximity score based on term positions
    fn calculate_proximity_score(&self, document: &IndexedDocument, query_terms: &[String]) -> f32 {
        if query_terms.len() < 2 {
            return 0.0;
        }
        
        let mut proximity_score = 0.0;
        let mut phrase_matches = 0;
        
        // Find positions of all query terms
        let mut term_positions: HashMap<String, Vec<usize>> = HashMap::new();
        for token in &document.tokens {
            if query_terms.contains(&token.term) {
                term_positions.entry(token.term.clone()).or_insert_with(Vec::new).push(token.position);
            }
        }
        
        // Calculate pairwise proximity scores
        for i in 0..query_terms.len() {
            for j in i + 1..query_terms.len() {
                let term1 = &query_terms[i];
                let term2 = &query_terms[j];
                
                if let (Some(positions1), Some(positions2)) = (term_positions.get(term1), term_positions.get(term2)) {
                    for &pos1 in positions1 {
                        for &pos2 in positions2 {
                            let distance = (pos1 as i32 - pos2 as i32).abs() as usize;
                            
                            if distance <= self.config.max_proximity_distance {
                                // Closer terms get higher proximity scores
                                let proximity = 1.0 - (distance as f32 / self.config.max_proximity_distance as f32);
                                proximity_score += proximity;
                                
                                // Exact phrase detection (consecutive terms)
                                if distance == 1 {
                                    phrase_matches += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Add phrase match bonus
        if phrase_matches > 0 {
            proximity_score += phrase_matches as f32 * self.config.phrase_match_bonus;
        }
        
        proximity_score
    }
    
    /// Generate result snippet
    fn generate_snippet(&self, document: &IndexedDocument, query_terms: &[String]) -> String {
        let lines: Vec<&str> = document.content.lines().collect();
        
        // Find best snippet around matched terms
        for token in &document.tokens {
            if query_terms.contains(&token.term) {
                let line_idx = token.line_number.saturating_sub(1);
                if line_idx < lines.len() {
                    // Return context around the match
                    let start_line = line_idx.saturating_sub(2);
                    let end_line = (line_idx + 3).min(lines.len());
                    
                    return lines[start_line..end_line].join("\\n");
                }
            }
        }
        
        // Fallback: return first few lines
        lines.iter().take(3).cloned().collect::<Vec<_>>().join("\\n")
    }
    
    /// Find positions of matched terms
    fn find_matched_positions(&self, document: &IndexedDocument, query_terms: &[String]) -> Vec<usize> {
        document.tokens.iter()
            .enumerate()
            .filter_map(|(i, token)| {
                if query_terms.contains(&token.term) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }
}

#[async_trait::async_trait]
impl BaselineSearcher for BM25ProximitySearcher {
    fn system_name(&self) -> &str {
        &self.system_name
    }
    
    async fn search(&self, query: &str, _intent: &str, _language: &str, max_results: usize) -> Result<Vec<SearchResult>> {
        let start_time = std::time::Instant::now();
        
        let proximity_results = self.execute_search(query, max_results).await
            .context("BM25+Proximity search failed")?;
        
        let latency_ms = start_time.elapsed().as_millis() as f32;
        
        // Update statistics
        {
            let mut stats = self.statistics.write().await;
            stats.queries_processed += 1;
            
            // Update running averages
            let n = stats.queries_processed as f32;
            stats.average_latency_ms = ((n - 1.0) * stats.average_latency_ms + latency_ms) / n;
            
            // Simplified percentile updates
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
        
        for (rank, result) in proximity_results.into_iter().enumerate() {
            if let Some(doc) = index.documents.get(&result.document_id) {
                let metadata = ResultMetadata {
                    line_number: result.matched_positions.first().map(|_| 1), // Simplified
                    function_name: doc.metadata.functions.first().cloned(),
                    class_name: doc.metadata.classes.first().cloned(),
                    language: doc.language.clone(),
                    file_size: doc.metadata.file_size,
                    last_modified: doc.metadata.last_modified,
                    scoring_breakdown: ScoringBreakdown {
                        lexical_score: result.bm25_score,
                        semantic_score: None,
                        proximity_score: Some(result.proximity_score),
                        recency_score: None,
                        popularity_score: None,
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
        
        debug!("BM25+Proximity search completed: {} results in {:.1}ms", results.len(), latency_ms);
        Ok(results)
    }
    
    fn get_config(&self) -> BaselineSystemConfig {
        let mut parameters = HashMap::new();
        parameters.insert("bm25_k1".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(self.config.bm25_k1 as f64).unwrap()));
        parameters.insert("bm25_b".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(self.config.bm25_b as f64).unwrap()));
        parameters.insert("proximity_weight".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(self.config.proximity_weight as f64).unwrap()));
        parameters.insert("max_proximity_distance".to_string(), serde_json::Value::Number(serde_json::Number::from(self.config.max_proximity_distance)));
        
        BaselineSystemConfig {
            system_name: self.system_name.clone(),
            version: "1.0.0".to_string(),
            parameters,
            index_config: IndexConfig {
                tokenizer: "whitespace_language_aware".to_string(),
                stemming: false,
                stop_words: false,
                n_grams: vec![1],
                case_sensitive: false,
                special_characters: true,
            },
            scoring_config: ScoringConfig {
                bm25_k1: self.config.bm25_k1,
                bm25_b: self.config.bm25_b,
                proximity_weight: Some(self.config.proximity_weight),
                semantic_weight: None,
                recency_weight: None,
                normalization: "bm25".to_string(),
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
    async fn test_bm25_proximity_searcher_creation() {
        let config = ProximityConfig::optimized_for_code();
        let searcher = BM25ProximitySearcher::new(config).await.unwrap();
        
        assert!(searcher.system_name().contains("BM25+Proximity"));
        assert_eq!(searcher.config.bm25_k1, 1.2);
        assert_eq!(searcher.config.proximity_weight, 0.3);
    }

    #[tokio::test]
    async fn test_tokenization() {
        let config = ProximityConfig::optimized_for_code();
        let searcher = BM25ProximitySearcher::new(config).await.unwrap();
        
        let content = "async function processData() {\n  return await fetch('/api/data');\n}";
        let tokens = searcher.tokenize_content(content, "typescript").await.unwrap();
        
        assert!(!tokens.is_empty());
        assert!(tokens.iter().any(|t| t.term == "async"));
        assert!(tokens.iter().any(|t| t.term == "function"));
        assert!(tokens.iter().any(|t| t.term == "processdata"));
    }

    #[tokio::test]
    async fn test_document_processing() {
        let config = ProximityConfig::optimized_for_code();
        let searcher = BM25ProximitySearcher::new(config).await.unwrap();
        
        let doc_input = DocumentToIndex {
            file_path: "src/test.ts".to_string(),
            content: "class TestClass {\n  async method() {\n    return true;\n  }\n}".to_string(),
            language: "typescript".to_string(),
            last_modified: None,
        };
        
        let indexed_doc = searcher.process_document(doc_input).await.unwrap();
        
        assert_eq!(indexed_doc.file_path, "src/test.ts");
        assert_eq!(indexed_doc.language, "typescript");
        assert!(!indexed_doc.tokens.is_empty());
        assert!(!indexed_doc.metadata.classes.is_empty());
        assert_eq!(indexed_doc.metadata.classes[0], "testclass");
    }

    #[tokio::test]
    async fn test_query_parsing() {
        let config = ProximityConfig::optimized_for_code();
        let searcher = BM25ProximitySearcher::new(config).await.unwrap();
        
        let query = "async function handler";
        let terms = searcher.parse_query(query).await.unwrap();
        
        assert_eq!(terms.len(), 3);
        assert!(terms.contains(&"async".to_string()));
        assert!(terms.contains(&"function".to_string()));
        assert!(terms.contains(&"handler".to_string()));
    }

    #[test]
    fn test_score_boost_calculation() {
        let config = ProximityConfig::optimized_for_code();
        let searcher = BM25ProximitySearcher {
            config: config.clone(),
            index: Arc::new(RwLock::new(ProximityIndex {
                documents: HashMap::new(),
                inverted_index: HashMap::new(),
                document_frequencies: HashMap::new(),
                total_documents: 0,
                average_document_length: 0.0,
            })),
            statistics: Arc::new(RwLock::new(SystemStatistics {
                queries_processed: 0,
                average_latency_ms: 0.0,
                p95_latency_ms: 0.0,
                p99_latency_ms: 0.0,
                cache_hit_rate: 0.0,
                index_size_mb: 0.0,
                memory_usage_mb: 0.0,
            })),
            system_name: "test".to_string(),
        };
        
        // Test special token boost
        let boost_async = searcher.calculate_score_boost("async", "typescript");
        assert!(boost_async > 1.0);
        
        // Test language keyword boost
        let boost_function = searcher.calculate_score_boost("function", "typescript");
        assert!(boost_function > 1.0);
        
        // Test regular token (no boost)
        let boost_regular = searcher.calculate_score_boost("variable", "typescript");
        assert_eq!(boost_regular, 1.0);
    }
}