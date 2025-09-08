//! # Comprehensive Feature Extractor for LTR
//!
//! Extracts all required features as specified in TODO.md:
//! - Lexical: BM25, exact matches, term frequency
//! - Structural/LSP: symbol matches, AST features, code structure
//! - RAPTOR: topic hierarchy, semantic clusters
//! - Centrality: PageRank, degree, betweenness
//! - ANN scores: dense vector similarities
//! - Path priors: file type, directory structure

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{debug, warn};

/// Complete feature vector for LTR training
#[derive(Debug, Clone)]
pub struct FeatureVector {
    pub features: Vec<f32>,
    pub feature_names: Vec<String>,
    pub query_id: String,
    pub doc_id: String,
}

/// Main feature extractor combining all feature types
pub struct ComprehensiveFeatureExtractor {
    lexical_extractor: LexicalFeatureExtractor,
    structural_extractor: StructuralFeatureExtractor,
    raptor_extractor: RaptorFeatureExtractor,
    centrality_extractor: CentralityFeatureExtractor,
    ann_extractor: ANNFeatureExtractor,
    path_extractor: PathPriorExtractor,
}

/// Document representation for feature extraction
#[derive(Debug, Clone)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub file_path: String,
    pub language: String,
    pub metadata: HashMap<String, String>,
    
    // Pre-computed scores
    pub lexical_score: f32,
    pub semantic_score: Option<f32>,
    pub lsp_scores: HashMap<String, f32>,
    
    // Structural information
    pub ast_nodes: Vec<ASTNode>,
    pub symbols: Vec<Symbol>,
    
    // Graph features
    pub centrality_scores: HashMap<String, f32>,
    pub raptor_topics: Vec<f32>,
    
    // ANN features
    pub dense_embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct ASTNode {
    pub node_type: String,
    pub start_line: usize,
    pub end_line: usize,
    pub depth: usize,
}

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub symbol_type: String, // function, class, variable, etc.
    pub location: (usize, usize), // (line, column)
    pub references: Vec<(String, usize, usize)>, // (file, line, col)
}

impl ComprehensiveFeatureExtractor {
    /// Create new comprehensive feature extractor
    pub fn new() -> Self {
        Self {
            lexical_extractor: LexicalFeatureExtractor::new(),
            structural_extractor: StructuralFeatureExtractor::new(),
            raptor_extractor: RaptorFeatureExtractor::new(),
            centrality_extractor: CentralityFeatureExtractor::new(),
            ann_extractor: ANNFeatureExtractor::new(),
            path_extractor: PathPriorExtractor::new(),
        }
    }

    /// Extract all features for a query-document pair
    pub async fn extract_features(&self, query: &str, document: &Document) -> Result<FeatureVector> {
        debug!("Extracting features for query='{}', doc='{}'", query, document.id);

        let mut all_features = Vec::new();
        let mut all_names = Vec::new();

        // Extract lexical features
        let lexical = self.lexical_extractor.extract(query, document)?;
        all_features.extend(lexical.features);
        all_names.extend(lexical.names);

        // Extract structural/LSP features
        let structural = self.structural_extractor.extract(query, document)?;
        all_features.extend(structural.features);
        all_names.extend(structural.names);

        // Extract RAPTOR features
        let raptor = self.raptor_extractor.extract(query, document).await?;
        all_features.extend(raptor.features);
        all_names.extend(raptor.names);

        // Extract centrality features
        let centrality = self.centrality_extractor.extract(query, document)?;
        all_features.extend(centrality.features);
        all_names.extend(centrality.names);

        // Extract ANN features
        let ann = self.ann_extractor.extract(query, document).await?;
        all_features.extend(ann.features);
        all_names.extend(ann.names);

        // Extract path prior features
        let path = self.path_extractor.extract(query, document)?;
        all_features.extend(path.features);
        all_names.extend(path.names);

        // Validate feature bounds (all features should be in [0, 1])
        for (i, feature) in all_features.iter().enumerate() {
            if !feature.is_finite() || *feature < 0.0 || *feature > 1.0 {
                warn!("Feature {} ({}) out of bounds: {}", i, all_names.get(i).unwrap_or(&"unknown".to_string()), feature);
            }
        }

        debug!("Extracted {} features for doc '{}'", all_features.len(), document.id);

        Ok(FeatureVector {
            features: all_features,
            feature_names: all_names,
            query_id: query.to_string(),
            doc_id: document.id.clone(),
        })
    }

    /// Get all feature names
    pub fn get_feature_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        names.extend(self.lexical_extractor.get_feature_names());
        names.extend(self.structural_extractor.get_feature_names());
        names.extend(self.raptor_extractor.get_feature_names());
        names.extend(self.centrality_extractor.get_feature_names());
        names.extend(self.ann_extractor.get_feature_names());
        names.extend(self.path_extractor.get_feature_names());
        names
    }

    /// Get feature count
    pub fn get_feature_count(&self) -> usize {
        self.get_feature_names().len()
    }
}

/// Extracted features with names
#[derive(Debug, Clone)]
struct ExtractedFeatures {
    features: Vec<f32>,
    names: Vec<String>,
}

/// Lexical feature extractor (BM25, exact matches, term frequency)
struct LexicalFeatureExtractor;

impl LexicalFeatureExtractor {
    fn new() -> Self {
        Self
    }

    fn extract(&self, query: &str, document: &Document) -> Result<ExtractedFeatures> {
        let mut features = Vec::new();
        let mut names = Vec::new();

        // BM25-like lexical score (already computed)
        features.push(document.lexical_score.clamp(0.0, 1.0));
        names.push("lexical_bm25".to_string());

        // Exact match features
        let query_terms = Self::tokenize(query);
        let content_lower = document.content.to_lowercase();
        
        // Exact term matches
        let exact_matches = query_terms.iter()
            .filter(|term| content_lower.contains(&term.to_lowercase()))
            .count() as f32 / query_terms.len().max(1) as f32;
        features.push(exact_matches);
        names.push("exact_match".to_string());

        // Query coverage (what fraction of query is in document)
        let unique_query_terms: HashSet<String> = query_terms.iter().cloned().collect();
        let covered_terms = unique_query_terms.iter()
            .filter(|term| content_lower.contains(&term.to_lowercase()))
            .count() as f32 / unique_query_terms.len().max(1) as f32;
        features.push(covered_terms);
        names.push("query_coverage".to_string());

        // Term frequency features
        let content_terms = Self::tokenize(&document.content);
        let tf_score = if !content_terms.is_empty() {
            query_terms.iter()
                .map(|qterm| {
                    let count = content_terms.iter().filter(|cterm| cterm.eq_ignore_ascii_case(qterm)).count();
                    count as f32 / content_terms.len() as f32
                })
                .sum::<f32>() / query_terms.len().max(1) as f32
        } else {
            0.0
        };
        features.push(tf_score);
        names.push("term_frequency".to_string());

        // Document length penalty (normalized)
        let length_penalty = 1.0 - (document.content.len() as f32 / 10000.0).min(1.0);
        features.push(length_penalty);
        names.push("length_penalty".to_string());

        // Query-document length ratio
        let length_ratio = if document.content.len() > 0 {
            (query.len() as f32 / document.content.len() as f32).min(1.0)
        } else {
            0.0
        };
        features.push(length_ratio);
        names.push("length_ratio".to_string());

        Ok(ExtractedFeatures { features, names })
    }

    fn get_feature_names(&self) -> Vec<String> {
        vec![
            "lexical_bm25".to_string(),
            "exact_match".to_string(),
            "query_coverage".to_string(),
            "term_frequency".to_string(),
            "length_penalty".to_string(),
            "length_ratio".to_string(),
        ]
    }

    fn tokenize(text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|s| !s.is_empty())
            .map(|s| s.to_lowercase())
            .collect()
    }
}

/// Structural/LSP feature extractor (symbols, AST, code structure)
struct StructuralFeatureExtractor;

impl StructuralFeatureExtractor {
    fn new() -> Self {
        Self
    }

    fn extract(&self, query: &str, document: &Document) -> Result<ExtractedFeatures> {
        let mut features = Vec::new();
        let mut names = Vec::new();

        // Symbol match features
        let query_tokens = LexicalFeatureExtractor::tokenize(query);
        let symbol_matches = self.compute_symbol_matches(&query_tokens, &document.symbols);
        features.push(symbol_matches);
        names.push("struct_hit".to_string());

        // AST structure features
        let ast_depth_score = self.compute_ast_depth_score(&document.ast_nodes);
        features.push(ast_depth_score);
        names.push("ast_depth".to_string());

        // Function/class density
        let code_density = self.compute_code_density(&document.symbols, &document.content);
        features.push(code_density);
        names.push("code_density".to_string());

        // LSP-based features
        let lsp_symbol_score = document.lsp_scores.get("symbol_match").unwrap_or(&0.0).clamp(0.0, 1.0);
        features.push(lsp_symbol_score);
        names.push("lsp_symbol_match".to_string());

        let lsp_reference_score = document.lsp_scores.get("reference_density").unwrap_or(&0.0).clamp(0.0, 1.0);
        features.push(lsp_reference_score);
        names.push("lsp_reference_density".to_string());

        // Import/dependency features
        let import_relevance = self.compute_import_relevance(&query_tokens, &document.content);
        features.push(import_relevance);
        names.push("import_relevance".to_string());

        Ok(ExtractedFeatures { features, names })
    }

    fn get_feature_names(&self) -> Vec<String> {
        vec![
            "struct_hit".to_string(),
            "ast_depth".to_string(),
            "code_density".to_string(),
            "lsp_symbol_match".to_string(),
            "lsp_reference_density".to_string(),
            "import_relevance".to_string(),
        ]
    }

    fn compute_symbol_matches(&self, query_tokens: &[String], symbols: &[Symbol]) -> f32 {
        if query_tokens.is_empty() || symbols.is_empty() {
            return 0.0;
        }

        let matches = query_tokens.iter()
            .filter(|token| {
                symbols.iter().any(|symbol| {
                    symbol.name.to_lowercase().contains(token) ||
                    token.contains(&symbol.name.to_lowercase())
                })
            })
            .count();

        matches as f32 / query_tokens.len() as f32
    }

    fn compute_ast_depth_score(&self, ast_nodes: &[ASTNode]) -> f32 {
        if ast_nodes.is_empty() {
            return 0.0;
        }

        let max_depth = ast_nodes.iter().map(|node| node.depth).max().unwrap_or(0);
        let avg_depth = ast_nodes.iter().map(|node| node.depth).sum::<usize>() as f32 / ast_nodes.len() as f32;
        
        // Normalize to [0, 1] assuming max reasonable depth is 10
        (avg_depth / 10.0).min(1.0)
    }

    fn compute_code_density(&self, symbols: &[Symbol], content: &str) -> f32 {
        let line_count = content.lines().count().max(1);
        let symbol_count = symbols.len();
        
        (symbol_count as f32 / line_count as f32).min(1.0)
    }

    fn compute_import_relevance(&self, query_tokens: &[String], content: &str) -> f32 {
        let import_patterns = ["import ", "from ", "use ", "#include", "require("];
        let import_lines: Vec<&str> = content.lines()
            .filter(|line| import_patterns.iter().any(|pattern| line.contains(pattern)))
            .collect();

        if import_lines.is_empty() || query_tokens.is_empty() {
            return 0.0;
        }

        let relevant_imports = query_tokens.iter()
            .filter(|token| {
                import_lines.iter().any(|line| line.to_lowercase().contains(&token.to_lowercase()))
            })
            .count();

        relevant_imports as f32 / query_tokens.len() as f32
    }
}

/// RAPTOR feature extractor (topic hierarchy, semantic clusters)
struct RaptorFeatureExtractor;

impl RaptorFeatureExtractor {
    fn new() -> Self {
        Self
    }

    async fn extract(&self, query: &str, document: &Document) -> Result<ExtractedFeatures> {
        let mut features = Vec::new();
        let mut names = Vec::new();

        // Topic relevance from RAPTOR hierarchy
        let topic_relevance = self.compute_topic_relevance(query, &document.raptor_topics).await?;
        features.push(topic_relevance);
        names.push("raptor_topic_relevance".to_string());

        // Document cluster membership strength
        let cluster_strength = self.compute_cluster_strength(&document.raptor_topics);
        features.push(cluster_strength);
        names.push("raptor_cluster_strength".to_string());

        // Hierarchical level features
        let hierarchy_level = self.compute_hierarchy_level(&document.raptor_topics);
        features.push(hierarchy_level);
        names.push("raptor_hierarchy_level".to_string());

        // Topic diversity (how many topics document spans)
        let topic_diversity = self.compute_topic_diversity(&document.raptor_topics);
        features.push(topic_diversity);
        names.push("raptor_topic_diversity".to_string());

        Ok(ExtractedFeatures { features, names })
    }

    fn get_feature_names(&self) -> Vec<String> {
        vec![
            "raptor_topic_relevance".to_string(),
            "raptor_cluster_strength".to_string(),
            "raptor_hierarchy_level".to_string(),
            "raptor_topic_diversity".to_string(),
        ]
    }

    async fn compute_topic_relevance(&self, query: &str, topics: &[f32]) -> Result<f32> {
        // Placeholder: compute semantic similarity between query and document topics
        // In practice, would use the actual RAPTOR topic embeddings
        if topics.is_empty() {
            return Ok(0.0);
        }

        // Simple heuristic based on topic strength
        let max_topic_score = topics.iter().cloned().fold(0.0f32, f32::max);
        Ok(max_topic_score.clamp(0.0, 1.0))
    }

    fn compute_cluster_strength(&self, topics: &[f32]) -> f32 {
        if topics.is_empty() {
            return 0.0;
        }

        // Measure how strongly the document belongs to its primary cluster
        let max_score = topics.iter().cloned().fold(0.0f32, f32::max);
        let avg_score = topics.iter().sum::<f32>() / topics.len() as f32;
        
        if avg_score > 0.0 {
            (max_score / avg_score).min(1.0) / 10.0 // Normalize to [0, 1]
        } else {
            0.0
        }
    }

    fn compute_hierarchy_level(&self, topics: &[f32]) -> f32 {
        // Placeholder: in practice would use actual RAPTOR hierarchy depth
        if topics.is_empty() {
            return 0.0;
        }

        // Simple heuristic: average topic activation indicates hierarchy level
        let avg_activation = topics.iter().sum::<f32>() / topics.len() as f32;
        avg_activation.clamp(0.0, 1.0)
    }

    fn compute_topic_diversity(&self, topics: &[f32]) -> f32 {
        if topics.is_empty() {
            return 0.0;
        }

        // Count how many topics have significant activation
        let active_topics = topics.iter().filter(|&&score| score > 0.1).count();
        (active_topics as f32 / topics.len() as f32).clamp(0.0, 1.0)
    }
}

/// Centrality feature extractor (PageRank, degree, betweenness)
struct CentralityFeatureExtractor;

impl CentralityFeatureExtractor {
    fn new() -> Self {
        Self
    }

    fn extract(&self, _query: &str, document: &Document) -> Result<ExtractedFeatures> {
        let mut features = Vec::new();
        let mut names = Vec::new();

        // PageRank centrality
        let pagerank = document.centrality_scores.get("pagerank").unwrap_or(&0.0).clamp(0.0, 1.0);
        features.push(pagerank);
        names.push("centrality_pagerank".to_string());

        // Degree centrality
        let degree = document.centrality_scores.get("degree").unwrap_or(&0.0).clamp(0.0, 1.0);
        features.push(degree);
        names.push("centrality_degree".to_string());

        // Betweenness centrality
        let betweenness = document.centrality_scores.get("betweenness").unwrap_or(&0.0).clamp(0.0, 1.0);
        features.push(betweenness);
        names.push("centrality_betweenness".to_string());

        // Closeness centrality
        let closeness = document.centrality_scores.get("closeness").unwrap_or(&0.0).clamp(0.0, 1.0);
        features.push(closeness);
        names.push("centrality_closeness".to_string());

        Ok(ExtractedFeatures { features, names })
    }

    fn get_feature_names(&self) -> Vec<String> {
        vec![
            "centrality_pagerank".to_string(),
            "centrality_degree".to_string(),
            "centrality_betweenness".to_string(),
            "centrality_closeness".to_string(),
        ]
    }
}

/// ANN feature extractor (dense vector similarities)
struct ANNFeatureExtractor;

impl ANNFeatureExtractor {
    fn new() -> Self {
        Self
    }

    async fn extract(&self, query: &str, document: &Document) -> Result<ExtractedFeatures> {
        let mut features = Vec::new();
        let mut names = Vec::new();

        // Dense embedding similarity (if available)
        let embedding_similarity = if let Some(ref embedding) = document.dense_embedding {
            self.compute_embedding_similarity(query, embedding).await?
        } else {
            document.semantic_score.unwrap_or(0.0)
        };
        features.push(embedding_similarity.clamp(0.0, 1.0));
        names.push("ann_embedding_similarity".to_string());

        // Vector norm (indicates document representation quality)
        let vector_norm = if let Some(ref embedding) = document.dense_embedding {
            let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            (norm / embedding.len() as f32).min(1.0)
        } else {
            0.5 // Default if no embedding
        };
        features.push(vector_norm);
        names.push("ann_vector_norm".to_string());

        // Embedding confidence (placeholder - would be computed by ANN system)
        let embedding_confidence = 0.7; // Placeholder
        features.push(embedding_confidence);
        names.push("ann_confidence".to_string());

        Ok(ExtractedFeatures { features, names })
    }

    fn get_feature_names(&self) -> Vec<String> {
        vec![
            "ann_embedding_similarity".to_string(),
            "ann_vector_norm".to_string(),
            "ann_confidence".to_string(),
        ]
    }

    async fn compute_embedding_similarity(&self, _query: &str, _embedding: &[f32]) -> Result<f32> {
        // Placeholder: would compute actual query-document embedding similarity
        // In practice, would encode query and compute cosine similarity
        Ok(0.5)
    }
}

/// Path prior extractor (file type, directory structure)
struct PathPriorExtractor;

impl PathPriorExtractor {
    fn new() -> Self {
        Self
    }

    fn extract(&self, query: &str, document: &Document) -> Result<ExtractedFeatures> {
        let mut features = Vec::new();
        let mut names = Vec::new();

        // File type priors
        let file_type_score = self.compute_file_type_prior(&document.file_path, &document.language);
        features.push(file_type_score);
        names.push("path_file_type_prior".to_string());

        // Directory depth (deeper files are often more specific)
        let depth_score = self.compute_directory_depth(&document.file_path);
        features.push(depth_score);
        names.push("path_directory_depth".to_string());

        // Directory name relevance
        let dir_relevance = self.compute_directory_relevance(query, &document.file_path);
        features.push(dir_relevance);
        names.push("path_directory_relevance".to_string());

        // Filename relevance
        let filename_relevance = self.compute_filename_relevance(query, &document.file_path);
        features.push(filename_relevance);
        names.push("path_filename_relevance".to_string());

        Ok(ExtractedFeatures { features, names })
    }

    fn get_feature_names(&self) -> Vec<String> {
        vec![
            "path_file_type_prior".to_string(),
            "path_directory_depth".to_string(),
            "path_directory_relevance".to_string(),
            "path_filename_relevance".to_string(),
        ]
    }

    fn compute_file_type_prior(&self, file_path: &str, language: &str) -> f32 {
        // Higher priors for source code files
        let source_extensions = [".rs", ".py", ".ts", ".js", ".cpp", ".h", ".java", ".go"];
        let config_extensions = [".json", ".yaml", ".toml", ".cfg"];
        let doc_extensions = [".md", ".txt", ".rst"];

        if source_extensions.iter().any(|ext| file_path.ends_with(ext)) {
            0.8
        } else if config_extensions.iter().any(|ext| file_path.ends_with(ext)) {
            0.4
        } else if doc_extensions.iter().any(|ext| file_path.ends_with(ext)) {
            0.6
        } else {
            0.3
        }
    }

    fn compute_directory_depth(&self, file_path: &str) -> f32 {
        let depth = file_path.split('/').filter(|part| !part.is_empty()).count();
        // Normalize depth, assuming typical range is 1-10
        (depth as f32 / 10.0).min(1.0)
    }

    fn compute_directory_relevance(&self, query: &str, file_path: &str) -> f32 {
        let query_tokens = LexicalFeatureExtractor::tokenize(query);
        if query_tokens.is_empty() {
            return 0.0;
        }

        let path_parts: Vec<&str> = file_path.split('/').collect();
        let relevant_parts = query_tokens.iter()
            .filter(|token| {
                path_parts.iter().any(|part| part.to_lowercase().contains(&token.to_lowercase()))
            })
            .count();

        relevant_parts as f32 / query_tokens.len() as f32
    }

    fn compute_filename_relevance(&self, query: &str, file_path: &str) -> f32 {
        let filename = file_path.split('/').last().unwrap_or("");
        let filename_without_ext = filename.split('.').next().unwrap_or("");
        
        let query_tokens = LexicalFeatureExtractor::tokenize(query);
        if query_tokens.is_empty() {
            return 0.0;
        }

        let matches = query_tokens.iter()
            .filter(|token| filename_without_ext.to_lowercase().contains(&token.to_lowercase()))
            .count();

        matches as f32 / query_tokens.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_document() -> Document {
        Document {
            id: "test_doc".to_string(),
            content: "def hello_world(): return 'Hello, World!'".to_string(),
            file_path: "src/utils/hello.py".to_string(),
            language: "python".to_string(),
            metadata: HashMap::new(),
            lexical_score: 0.8,
            semantic_score: Some(0.7),
            lsp_scores: {
                let mut scores = HashMap::new();
                scores.insert("symbol_match".to_string(), 0.6);
                scores.insert("reference_density".to_string(), 0.4);
                scores
            },
            ast_nodes: vec![
                ASTNode {
                    node_type: "function_def".to_string(),
                    start_line: 1,
                    end_line: 1,
                    depth: 0,
                }
            ],
            symbols: vec![
                Symbol {
                    name: "hello_world".to_string(),
                    symbol_type: "function".to_string(),
                    location: (1, 4),
                    references: vec![],
                }
            ],
            centrality_scores: {
                let mut scores = HashMap::new();
                scores.insert("pagerank".to_string(), 0.5);
                scores.insert("degree".to_string(), 0.3);
                scores.insert("betweenness".to_string(), 0.2);
                scores.insert("closeness".to_string(), 0.4);
                scores
            },
            raptor_topics: vec![0.8, 0.2, 0.0, 0.1],
            dense_embedding: Some(vec![0.1, 0.2, 0.3, 0.4, 0.5]),
        }
    }

    #[tokio::test]
    async fn test_feature_extraction() {
        let extractor = ComprehensiveFeatureExtractor::new();
        let document = create_test_document();
        let query = "hello world function";

        let features = extractor.extract_features(query, &document).await.unwrap();
        
        assert!(!features.features.is_empty());
        assert_eq!(features.features.len(), features.feature_names.len());
        assert_eq!(features.query_id, query);
        assert_eq!(features.doc_id, "test_doc");

        // Check that all features are bounded
        for (i, feature) in features.features.iter().enumerate() {
            assert!(
                feature.is_finite() && *feature >= 0.0 && *feature <= 1.0,
                "Feature {} ({}) out of bounds: {}",
                i,
                features.feature_names.get(i).unwrap_or(&"unknown".to_string()),
                feature
            );
        }
    }

    #[test]
    fn test_lexical_extractor() {
        let extractor = LexicalFeatureExtractor::new();
        let document = create_test_document();
        let query = "hello world";

        let features = extractor.extract(query, &document).unwrap();
        
        assert_eq!(features.features.len(), 6);
        assert_eq!(features.names.len(), 6);
        
        // Check exact match feature
        assert!(features.features[1] > 0.0); // Should have some exact matches
    }

    #[test]
    fn test_path_extractor() {
        let extractor = PathPriorExtractor::new();
        let document = create_test_document();
        let query = "hello utils";

        let features = extractor.extract(query, &document).unwrap();
        
        assert_eq!(features.features.len(), 4);
        assert_eq!(features.names.len(), 4);
        
        // Check file type prior (Python file should have high prior)
        assert!(features.features[0] > 0.5);
        
        // Check directory relevance ("utils" should match)
        assert!(features.features[2] > 0.0);
    }

    #[test]
    fn test_structural_extractor() {
        let extractor = StructuralFeatureExtractor::new();
        let document = create_test_document();
        let query = "hello_world function";

        let features = extractor.extract(query, &document).unwrap();
        
        assert_eq!(features.features.len(), 6);
        
        // Check symbol match (should match "hello_world")
        assert!(features.features[0] > 0.0);
    }
}