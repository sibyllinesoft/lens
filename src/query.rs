use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use anyhow::Result;
use uuid;

/// Query processing and optimization module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    pub id: String,
    pub original_query: String,
    pub processed_query: String,
    pub query_type: QueryType,
    pub filters: QueryFilters,
    pub options: QueryOptions,
    pub metadata: HashMap<String, String>,
}

/// Classification of query types for optimization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum QueryType {
    /// Exact string match
    Exact,
    /// Fuzzy/approximate matching
    #[default]
    Fuzzy,
    /// Structural code patterns
    Structural,
    /// Semantic understanding
    Semantic,
    /// Symbol/identifier search
    Symbol,
    /// Hybrid approach using multiple strategies
    Hybrid,
    /// Regular expression pattern
    Regex,
}

/// Query filters for scoping search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFilters {
    pub file_extensions: Vec<String>,
    pub exclude_paths: Vec<String>,
    pub include_paths: Vec<String>,
    pub language_filter: Option<String>,
    pub size_limit: Option<usize>,
    pub modified_since: Option<chrono::DateTime<chrono::Utc>>,
    pub file_type: Option<FileType>,
}

/// File type classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum FileType {
    #[default]
    Source,
    Test,
    Configuration,
    Documentation,
    Build,
    Data,
}

/// Query execution options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryOptions {
    pub max_results: usize,
    pub timeout_ms: u64,
    pub systems: Vec<String>,
    pub include_snippets: bool,
    pub highlight_matches: bool,
    pub sort_by: SortOrder,
    pub boost_factors: HashMap<String, f64>,
}

/// Result sorting options
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SortOrder {
    #[default]
    Relevance,
    Modified,
    Created,
    Size,
    Name,
    Language,
}

/// Query processing pipeline
pub struct QueryProcessor {
    analyzers: Vec<Box<dyn QueryAnalyzer + Send + Sync>>,
    optimizers: Vec<Box<dyn QueryOptimizer + Send + Sync>>,
    validators: Vec<Box<dyn QueryValidator + Send + Sync>>,
}

/// Trait for query analysis
pub trait QueryAnalyzer: Send + Sync {
    fn name(&self) -> &str;
    fn analyze(&self, query: &str) -> Result<QueryAnalysis>;
}

/// Trait for query optimization
pub trait QueryOptimizer: Send + Sync {
    fn name(&self) -> &str;
    fn optimize(&self, query: &Query, context: &QueryContext) -> Result<Query>;
}

/// Trait for query validation
pub trait QueryValidator: Send + Sync {
    fn name(&self) -> &str;
    fn validate(&self, query: &Query) -> Result<ValidationResult>;
}

/// Query analysis results
#[derive(Debug, Clone)]
pub struct QueryAnalysis {
    pub complexity_score: f64,
    pub estimated_results: usize,
    pub suggested_systems: Vec<String>,
    pub optimization_hints: Vec<String>,
    pub language_hints: Vec<String>,
    pub pattern_type: PatternType,
}

/// Pattern type classification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatternType {
    Literal,
    Wildcard,
    Regex,
    Structural,
    Semantic,
}

/// Query execution context
#[derive(Debug, Clone)]
pub struct QueryContext {
    pub user_id: Option<String>,
    pub workspace_path: String,
    pub recent_files: Vec<String>,
    pub language_preference: Option<String>,
    pub performance_profile: PerformanceProfile,
}

/// Performance profile for query optimization
#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceProfile {
    Speed,      // Optimize for fastest response
    Accuracy,   // Optimize for best results
    Balanced,   // Balance speed and accuracy
}

/// Query validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub valid: bool,
    pub issues: Vec<ValidationIssue>,
    pub suggestions: Vec<String>,
}

/// Validation issue
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    pub severity: IssueSeverity,
    pub message: String,
    pub suggestion: Option<String>,
}

/// Issue severity levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IssueSeverity {
    Error,
    Warning,
    Info,
}

impl QueryProcessor {
    /// Create new query processor with default analyzers and optimizers
    pub fn new() -> Self {
        let mut processor = Self {
            analyzers: Vec::new(),
            optimizers: Vec::new(),
            validators: Vec::new(),
        };

        // Add default components
        processor.add_analyzer(Box::new(BasicQueryAnalyzer::new()));
        processor.add_analyzer(Box::new(PatternAnalyzer::new()));
        processor.add_optimizer(Box::new(PerformanceOptimizer::new()));
        processor.add_optimizer(Box::new(SystemSelectionOptimizer::new()));
        processor.add_validator(Box::new(BasicQueryValidator::new()));

        processor
    }

    /// Add query analyzer
    pub fn add_analyzer(&mut self, analyzer: Box<dyn QueryAnalyzer + Send + Sync>) {
        self.analyzers.push(analyzer);
    }

    /// Add query optimizer
    pub fn add_optimizer(&mut self, optimizer: Box<dyn QueryOptimizer + Send + Sync>) {
        self.optimizers.push(optimizer);
    }

    /// Add query validator
    pub fn add_validator(&mut self, validator: Box<dyn QueryValidator + Send + Sync>) {
        self.validators.push(validator);
    }

    /// Process raw query into optimized Query object
    pub fn process_query(
        &self,
        raw_query: &str,
        context: &QueryContext,
    ) -> Result<ProcessedQuery> {
        let mut query = self.parse_raw_query(raw_query)?;

        // Analyze the query
        let mut analysis = QueryAnalysis {
            complexity_score: 0.0,
            estimated_results: 0,
            suggested_systems: Vec::new(),
            optimization_hints: Vec::new(),
            language_hints: Vec::new(),
            pattern_type: PatternType::Literal,
        };

        for analyzer in &self.analyzers {
            let result = analyzer.analyze(&query.original_query)?;
            analysis = self.merge_analysis(analysis, result);
        }

        // Optimize the query
        for optimizer in &self.optimizers {
            query = optimizer.optimize(&query, context)?;
        }

        // Validate the query
        let mut validation_results = Vec::new();
        for validator in &self.validators {
            validation_results.push(validator.validate(&query)?);
        }

        let overall_validation = self.merge_validation_results(validation_results);

        Ok(ProcessedQuery {
            query,
            analysis,
            validation: overall_validation,
            processing_time_ms: 0.0, // Would track actual time
        })
    }

    /// Parse raw query string into structured Query
    fn parse_raw_query(&self, raw_query: &str) -> Result<Query> {
        let query_id = uuid::Uuid::new_v4().to_string();
        
        // Extract filters from query
        let (processed_query, filters) = self.extract_filters(raw_query);
        
        // Classify query type
        let query_type = self.classify_query(&processed_query);

        Ok(Query {
            id: query_id,
            original_query: raw_query.to_string(),
            processed_query,
            query_type,
            filters,
            options: QueryOptions::default(),
            metadata: HashMap::new(),
        })
    }

    /// Extract filters from query string
    fn extract_filters(&self, query: &str) -> (String, QueryFilters) {
        let mut processed = query.to_string();
        let mut filters = QueryFilters::default();

        // Extract file extensions (ext:ts, ext:py)
        if let Ok(ext_regex) = regex::Regex::new(r"ext:(\w+)") {
            for cap in ext_regex.captures_iter(query) {
                if let Some(ext) = cap.get(1) {
                    filters.file_extensions.push(format!(".{}", ext.as_str()));
                    processed = processed.replace(cap.get(0).unwrap().as_str(), "");
                }
            }
        }

        // Extract language filter (lang:typescript, lang:python)
        if let Ok(lang_regex) = regex::Regex::new(r"lang:(\w+)") {
            for cap in lang_regex.captures_iter(query) {
                if let Some(lang) = cap.get(1) {
                    filters.language_filter = Some(lang.as_str().to_string());
                    processed = processed.replace(cap.get(0).unwrap().as_str(), "");
                }
            }
        }

        // Extract path filters (path:src/, -path:node_modules/)
        if let Ok(path_regex) = regex::Regex::new(r"(-?)path:([^\s]+)") {
            for cap in path_regex.captures_iter(query) {
                if let Some(path) = cap.get(2) {
                    let path_str = path.as_str().to_string();
                    if cap.get(1).map(|m| m.as_str()) == Some("-") {
                        filters.exclude_paths.push(path_str);
                    } else {
                        filters.include_paths.push(path_str);
                    }
                    processed = processed.replace(cap.get(0).unwrap().as_str(), "");
                }
            }
        }

        (processed.trim().to_string(), filters)
    }

    /// Classify query type based on patterns
    fn classify_query(&self, query: &str) -> QueryType {
        // Exact match (quoted strings)
        if query.starts_with('"') && query.ends_with('"') {
            return QueryType::Exact;
        }

        // Regex patterns
        if query.contains('*') || query.contains('?') || query.contains('[') || query.contains('^') {
            return QueryType::Regex;
        }

        // Structural patterns
        if query.contains("class ") || query.contains("function ") || 
           query.contains("interface ") || query.contains("struct ") {
            return QueryType::Structural;
        }

        // Symbol search (single identifier)
        if query.split_whitespace().count() == 1 && 
           query.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return QueryType::Symbol;
        }

        // Semantic search (natural language)
        if query.split_whitespace().count() > 3 {
            return QueryType::Semantic;
        }

        // Default to fuzzy
        QueryType::Fuzzy
    }

    /// Merge multiple analysis results
    fn merge_analysis(&self, base: QueryAnalysis, new: QueryAnalysis) -> QueryAnalysis {
        QueryAnalysis {
            complexity_score: (base.complexity_score + new.complexity_score) / 2.0,
            estimated_results: std::cmp::max(base.estimated_results, new.estimated_results),
            suggested_systems: {
                let mut systems = base.suggested_systems;
                systems.extend(new.suggested_systems);
                systems.sort();
                systems.dedup();
                systems
            },
            optimization_hints: {
                let mut hints = base.optimization_hints;
                hints.extend(new.optimization_hints);
                hints
            },
            language_hints: {
                let mut hints = base.language_hints;
                hints.extend(new.language_hints);
                hints.sort();
                hints.dedup();
                hints
            },
            pattern_type: new.pattern_type, // Use latest classification
        }
    }

    /// Merge validation results
    fn merge_validation_results(&self, results: Vec<ValidationResult>) -> ValidationResult {
        let valid = results.iter().all(|r| r.valid);
        let mut all_issues = Vec::new();
        let mut all_suggestions = Vec::new();

        for result in results {
            all_issues.extend(result.issues);
            all_suggestions.extend(result.suggestions);
        }

        ValidationResult {
            valid,
            issues: all_issues,
            suggestions: all_suggestions,
        }
    }
}

/// Complete processed query result
#[derive(Debug, Clone)]
pub struct ProcessedQuery {
    pub query: Query,
    pub analysis: QueryAnalysis,
    pub validation: ValidationResult,
    pub processing_time_ms: f64,
}

// Default implementations

impl Default for QueryFilters {
    fn default() -> Self {
        Self {
            file_extensions: Vec::new(),
            exclude_paths: Vec::new(),
            include_paths: Vec::new(),
            language_filter: None,
            size_limit: None,
            modified_since: None,
            file_type: None,
        }
    }
}

impl Default for QueryOptions {
    fn default() -> Self {
        Self {
            max_results: 50,
            timeout_ms: 2000,
            systems: Vec::new(),
            include_snippets: true,
            highlight_matches: true,
            sort_by: SortOrder::Relevance,
            boost_factors: HashMap::new(),
        }
    }
}

impl Default for Query {
    fn default() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            original_query: String::new(),
            processed_query: String::new(),
            query_type: QueryType::default(),
            filters: QueryFilters::default(),
            options: QueryOptions::default(),
            metadata: HashMap::new(),
        }
    }
}

// Basic implementations of analyzer/optimizer/validator traits

pub struct BasicQueryAnalyzer;

impl BasicQueryAnalyzer {
    pub fn new() -> Self {
        Self
    }
}

impl QueryAnalyzer for BasicQueryAnalyzer {
    fn name(&self) -> &str {
        "basic_analyzer"
    }

    fn analyze(&self, query: &str) -> Result<QueryAnalysis> {
        let complexity = if query.len() > 100 { 0.8 } else { 0.4 };
        let estimated_results = query.len() * 10; // Rough estimate

        Ok(QueryAnalysis {
            complexity_score: complexity,
            estimated_results,
            suggested_systems: vec!["lex".to_string()],
            optimization_hints: Vec::new(),
            language_hints: Vec::new(),
            pattern_type: PatternType::Literal,
        })
    }
}

pub struct PatternAnalyzer;

impl PatternAnalyzer {
    pub fn new() -> Self {
        Self
    }
}

impl QueryAnalyzer for PatternAnalyzer {
    fn name(&self) -> &str {
        "pattern_analyzer"
    }

    fn analyze(&self, query: &str) -> Result<QueryAnalysis> {
        let pattern_type = if query.contains('*') || query.contains('?') {
            PatternType::Wildcard
        } else if query.contains("class ") || query.contains("function ") {
            PatternType::Structural
        } else {
            PatternType::Literal
        };

        Ok(QueryAnalysis {
            complexity_score: 0.5,
            estimated_results: 20,
            suggested_systems: vec!["symbols".to_string()],
            optimization_hints: Vec::new(),
            language_hints: Vec::new(),
            pattern_type,
        })
    }
}

pub struct PerformanceOptimizer;

impl PerformanceOptimizer {
    pub fn new() -> Self {
        Self
    }
}

impl QueryOptimizer for PerformanceOptimizer {
    fn name(&self) -> &str {
        "performance_optimizer"
    }

    fn optimize(&self, query: &Query, context: &QueryContext) -> Result<Query> {
        let mut optimized = query.clone();

        // Adjust timeout based on performance profile
        match context.performance_profile {
            PerformanceProfile::Speed => {
                optimized.options.timeout_ms = std::cmp::min(optimized.options.timeout_ms, 1000);
                optimized.options.max_results = std::cmp::min(optimized.options.max_results, 20);
            }
            PerformanceProfile::Accuracy => {
                optimized.options.timeout_ms = std::cmp::max(optimized.options.timeout_ms, 5000);
                optimized.options.max_results = std::cmp::max(optimized.options.max_results, 100);
            }
            PerformanceProfile::Balanced => {
                // Keep defaults
            }
        }

        Ok(optimized)
    }
}

pub struct SystemSelectionOptimizer;

impl SystemSelectionOptimizer {
    pub fn new() -> Self {
        Self
    }
}

impl QueryOptimizer for SystemSelectionOptimizer {
    fn name(&self) -> &str {
        "system_selection_optimizer"
    }

    fn optimize(&self, query: &Query, _context: &QueryContext) -> Result<Query> {
        let mut optimized = query.clone();

        // Select optimal systems based on query type
        optimized.options.systems = match query.query_type {
            QueryType::Symbol => vec!["symbols".to_string()],
            QueryType::Structural => vec!["symbols".to_string(), "lex".to_string()],
            QueryType::Semantic => vec!["semantic".to_string(), "lex".to_string()],
            _ => vec!["lex".to_string(), "symbols".to_string()],
        };

        Ok(optimized)
    }
}

pub struct BasicQueryValidator;

impl BasicQueryValidator {
    pub fn new() -> Self {
        Self
    }
}

impl QueryValidator for BasicQueryValidator {
    fn name(&self) -> &str {
        "basic_validator"
    }

    fn validate(&self, query: &Query) -> Result<ValidationResult> {
        let mut issues = Vec::new();
        let mut suggestions = Vec::new();

        // Check for empty query
        if query.processed_query.trim().is_empty() {
            issues.push(ValidationIssue {
                severity: IssueSeverity::Error,
                message: "Query cannot be empty".to_string(),
                suggestion: Some("Enter a search term".to_string()),
            });
        }

        // Check query length
        if query.processed_query.len() > 1000 {
            issues.push(ValidationIssue {
                severity: IssueSeverity::Warning,
                message: "Very long query may be slow".to_string(),
                suggestion: Some("Consider shortening your query".to_string()),
            });
        }

        // Check timeout
        if query.options.timeout_ms < 100 {
            suggestions.push("Consider increasing timeout for better results".to_string());
        }

        Ok(ValidationResult {
            valid: !issues.iter().any(|i| i.severity == IssueSeverity::Error),
            issues,
            suggestions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_creation() {
        let query = Query {
            id: "test-id".to_string(),
            original_query: "test query".to_string(),
            processed_query: "test query".to_string(),
            query_type: QueryType::Fuzzy,
            filters: QueryFilters::default(),
            options: QueryOptions::default(),
            metadata: HashMap::new(),
        };
        
        assert_eq!(query.id, "test-id");
        assert_eq!(query.original_query, "test query");
        assert_eq!(query.query_type, QueryType::Fuzzy);
    }

    #[test]
    fn test_query_type_variants() {
        assert_eq!(QueryType::Exact, QueryType::Exact);
        assert_ne!(QueryType::Exact, QueryType::Fuzzy);
        
        // Test all variants exist
        let _types = [
            QueryType::Exact,
            QueryType::Fuzzy,
            QueryType::Structural,
            QueryType::Semantic,
            QueryType::Symbol,
            QueryType::Hybrid,
            QueryType::Regex,
        ];
    }

    #[test]
    fn test_query_filters_default() {
        let filters = QueryFilters::default();
        
        assert!(filters.file_extensions.is_empty());
        assert!(filters.exclude_paths.is_empty());
        assert!(filters.include_paths.is_empty());
        assert_eq!(filters.language_filter, None);
        assert_eq!(filters.size_limit, None);
        assert_eq!(filters.modified_since, None);
        assert_eq!(filters.file_type, None);
    }

    #[test]
    fn test_file_type_variants() {
        let types = [
            FileType::Source,
            FileType::Test,
            FileType::Configuration,
            FileType::Documentation,
            FileType::Build,
            FileType::Data,
        ];
        
        assert_eq!(types.len(), 6);
        assert_eq!(FileType::Source, FileType::Source);
        assert_ne!(FileType::Source, FileType::Test);
    }

    #[test]
    fn test_query_options_default() {
        let options = QueryOptions::default();
        
        assert_eq!(options.max_results, 50);
        assert_eq!(options.timeout_ms, 2000);
        assert!(options.systems.is_empty());
        assert!(options.include_snippets);
        assert!(options.highlight_matches);
        assert_eq!(options.sort_by, SortOrder::Relevance);
        assert!(options.boost_factors.is_empty());
    }

    #[test]
    fn test_sort_order_variants() {
        let orders = [
            SortOrder::Relevance,
            SortOrder::Modified,
            SortOrder::Created,
            SortOrder::Size,
            SortOrder::Name,
            SortOrder::Language,
        ];
        
        assert_eq!(orders.len(), 6);
        assert_eq!(SortOrder::Relevance, SortOrder::Relevance);
        assert_ne!(SortOrder::Relevance, SortOrder::Modified);
    }

    #[test]
    fn test_query_processor_new() {
        let processor = QueryProcessor::new();
        
        // Should start with default analyzers, optimizers, validators
        assert_eq!(processor.analyzers.len(), 2); // BasicQueryAnalyzer + PatternAnalyzer
        assert_eq!(processor.optimizers.len(), 2); // PerformanceOptimizer + SystemSelectionOptimizer  
        assert_eq!(processor.validators.len(), 1); // BasicQueryValidator
    }

    #[test]
    fn test_query_processor_add_analyzer() {
        let mut processor = QueryProcessor::new();
        let analyzer = BasicQueryAnalyzer::new();
        
        processor.add_analyzer(Box::new(analyzer));
        assert_eq!(processor.analyzers.len(), 3); // 2 defaults + 1 added
    }

    #[test]
    fn test_query_processor_add_optimizer() {
        let mut processor = QueryProcessor::new();
        let optimizer = PerformanceOptimizer::new();
        
        processor.add_optimizer(Box::new(optimizer));
        assert_eq!(processor.optimizers.len(), 3); // 2 defaults + 1 added
    }

    #[test]
    fn test_query_processor_add_validator() {
        let mut processor = QueryProcessor::new();
        let validator = BasicQueryValidator::new();
        
        processor.add_validator(Box::new(validator));
        assert_eq!(processor.validators.len(), 2); // 1 default + 1 added
    }

    #[test]
    fn test_extract_filters_file_extensions() {
        let processor = QueryProcessor::new();
        
        let (processed, filters) = processor.extract_filters("search code ext:ts ext:py");
        
        assert_eq!(processed, "search code");
        assert_eq!(filters.file_extensions, vec![".ts", ".py"]);
    }

    #[test]
    fn test_extract_filters_language() {
        let processor = QueryProcessor::new();
        
        let (processed, filters) = processor.extract_filters("function handler lang:typescript");
        
        assert_eq!(processed, "function handler");
        assert_eq!(filters.language_filter, Some("typescript".to_string()));
    }

    #[test]
    fn test_extract_filters_paths() {
        let processor = QueryProcessor::new();
        
        let (processed, filters) = processor.extract_filters("test path:src/ -path:node_modules/");
        
        assert_eq!(processed, "test");
        assert_eq!(filters.include_paths, vec!["src/"]);
        assert_eq!(filters.exclude_paths, vec!["node_modules/"]);
    }

    #[test]
    fn test_classify_query_exact() {
        let processor = QueryProcessor::new();
        
        let query_type = processor.classify_query("\"exact match\"");
        assert_eq!(query_type, QueryType::Exact);
    }

    #[test]
    fn test_classify_query_regex() {
        let processor = QueryProcessor::new();
        
        let query_type = processor.classify_query("test*pattern");
        assert_eq!(query_type, QueryType::Regex);
        
        let query_type = processor.classify_query("test?pattern");
        assert_eq!(query_type, QueryType::Regex);
        
        let query_type = processor.classify_query("^start");
        assert_eq!(query_type, QueryType::Regex);
    }

    #[test]
    fn test_classify_query_structural() {
        let processor = QueryProcessor::new();
        
        let query_type = processor.classify_query("class MyClass");
        assert_eq!(query_type, QueryType::Structural);
        
        let query_type = processor.classify_query("function handler");
        assert_eq!(query_type, QueryType::Structural);
        
        let query_type = processor.classify_query("interface Config");
        assert_eq!(query_type, QueryType::Structural);
    }

    #[test]
    fn test_classify_query_symbol() {
        let processor = QueryProcessor::new();
        
        let query_type = processor.classify_query("handleRequest");
        assert_eq!(query_type, QueryType::Symbol);
        
        let query_type = processor.classify_query("my_variable");
        assert_eq!(query_type, QueryType::Symbol);
    }

    #[test]
    fn test_classify_query_semantic() {
        let processor = QueryProcessor::new();
        
        let query_type = processor.classify_query("how to handle user authentication in web apps");
        assert_eq!(query_type, QueryType::Semantic);
    }

    #[test]
    fn test_classify_query_fuzzy_default() {
        let processor = QueryProcessor::new();
        
        let query_type = processor.classify_query("test query");
        assert_eq!(query_type, QueryType::Fuzzy);
    }

    #[test]
    fn test_basic_query_analyzer() {
        let analyzer = BasicQueryAnalyzer::new();
        
        assert_eq!(analyzer.name(), "basic_analyzer");
        
        let result = analyzer.analyze("test query").unwrap();
        assert!(result.complexity_score >= 0.0);
        assert!(result.estimated_results > 0); // Remove useless comparison with 0 for usize
        assert!(!result.suggested_systems.is_empty());
    }

    #[test]
    fn test_basic_query_analyzer_complex_query() {
        let analyzer = BasicQueryAnalyzer::new();
        
        // Use a query > 100 characters to trigger complex classification
        let complex_query = "this is a very complex query with many terms and conditions that should be long enough to trigger the complexity threshold of over 100 characters";
        let result = analyzer.analyze(complex_query).unwrap();
        assert!(result.complexity_score > 0.5); // Should be marked as complex (0.8)
        assert_eq!(result.complexity_score, 0.8); // Verify exact value for long queries
    }

    #[test]
    fn test_pattern_analyzer() {
        let analyzer = PatternAnalyzer::new();
        
        assert_eq!(analyzer.name(), "pattern_analyzer");
        
        let result = analyzer.analyze("class.*implements").unwrap();
        assert_eq!(result.pattern_type, PatternType::Wildcard); // Contains *, classified as wildcard
        
        let result = analyzer.analyze("simple text").unwrap();
        assert_eq!(result.pattern_type, PatternType::Literal);
    }

    #[test]
    fn test_performance_optimizer_speed() {
        let optimizer = PerformanceOptimizer::new();
        let context = QueryContext {
            user_id: Some("test".to_string()),
            workspace_path: "/test/workspace".to_string(),
            recent_files: vec![],
            language_preference: None,
            performance_profile: PerformanceProfile::Speed,
        };
        
        let mut query = Query {
            id: "test".to_string(),
            original_query: "test".to_string(),
            processed_query: "test".to_string(),
            query_type: QueryType::Fuzzy,
            filters: QueryFilters::default(),
            options: QueryOptions {
                max_results: 100,
                timeout_ms: 5000,
                ..QueryOptions::default()
            },
            metadata: HashMap::new(),
        };
        
        let optimized = optimizer.optimize(&query, &context).unwrap();
        
        // Should reduce timeout and max_results for speed
        assert!(optimized.options.timeout_ms <= 1000);
        assert!(optimized.options.max_results <= 20);
    }

    #[test]
    fn test_performance_optimizer_accuracy() {
        let optimizer = PerformanceOptimizer::new();
        let context = QueryContext {
            user_id: Some("test".to_string()),
            workspace_path: "/test/workspace".to_string(),
            recent_files: vec![],
            language_preference: None,
            performance_profile: PerformanceProfile::Accuracy,
        };
        
        let query = Query {
            id: "test".to_string(),
            original_query: "test".to_string(),
            processed_query: "test".to_string(),
            query_type: QueryType::Fuzzy,
            filters: QueryFilters::default(),
            options: QueryOptions {
                max_results: 10,
                timeout_ms: 1000,
                ..QueryOptions::default()
            },
            metadata: HashMap::new(),
        };
        
        let optimized = optimizer.optimize(&query, &context).unwrap();
        
        // Should increase timeout and max_results for accuracy
        assert!(optimized.options.timeout_ms >= 5000);
        assert!(optimized.options.max_results >= 100);
    }

    #[test]
    fn test_system_selection_optimizer() {
        let optimizer = SystemSelectionOptimizer::new();
        let context = QueryContext {
            user_id: Some("test".to_string()),
            workspace_path: "/test/workspace".to_string(),
            recent_files: vec![],
            language_preference: None,
            performance_profile: PerformanceProfile::Balanced,
        };
        
        // Test symbol query
        let query = Query {
            query_type: QueryType::Symbol,
            ..Query::default()
        };
        let optimized = optimizer.optimize(&query, &context).unwrap();
        assert_eq!(optimized.options.systems, vec!["symbols"]);
        
        // Test structural query
        let query = Query {
            query_type: QueryType::Structural,
            ..Query::default()
        };
        let optimized = optimizer.optimize(&query, &context).unwrap();
        assert_eq!(optimized.options.systems, vec!["symbols", "lex"]);
        
        // Test semantic query
        let query = Query {
            query_type: QueryType::Semantic,
            ..Query::default()
        };
        let optimized = optimizer.optimize(&query, &context).unwrap();
        assert_eq!(optimized.options.systems, vec!["semantic", "lex"]);
    }

    #[test]
    fn test_basic_query_validator_empty() {
        let validator = BasicQueryValidator::new();
        let query = Query {
            processed_query: "".to_string(),
            ..Query::default()
        };
        
        let result = validator.validate(&query).unwrap();
        assert!(!result.valid); // Empty query should be invalid
        assert_eq!(result.issues.len(), 1);
        assert_eq!(result.issues[0].severity, IssueSeverity::Error);
    }

    #[test]
    fn test_basic_query_validator_valid() {
        let validator = BasicQueryValidator::new();
        let query = Query {
            processed_query: "valid query".to_string(),
            options: QueryOptions {
                timeout_ms: 1000,
                ..QueryOptions::default()
            },
            ..Query::default()
        };
        
        let result = validator.validate(&query).unwrap();
        assert!(result.valid);
        assert!(result.issues.is_empty());
    }

    #[test]
    fn test_basic_query_validator_long_query() {
        let validator = BasicQueryValidator::new();
        let long_query = "a".repeat(1001); // Over 1000 characters
        let query = Query {
            processed_query: long_query,
            ..Query::default()
        };
        
        let result = validator.validate(&query).unwrap();
        assert!(result.valid); // Still valid, just a warning
        assert_eq!(result.issues.len(), 1);
        assert_eq!(result.issues[0].severity, IssueSeverity::Warning);
    }

    #[test]
    fn test_basic_query_validator_low_timeout() {
        let validator = BasicQueryValidator::new();
        let query = Query {
            processed_query: "test".to_string(),
            options: QueryOptions {
                timeout_ms: 50, // Very low timeout
                ..QueryOptions::default()
            },
            ..Query::default()
        };
        
        let result = validator.validate(&query).unwrap();
        assert!(result.valid);
        assert!(!result.suggestions.is_empty()); // Should suggest increasing timeout
    }

    #[test]
    fn test_pattern_type_variants() {
        let types = [
            PatternType::Literal,
            PatternType::Regex,
            PatternType::Wildcard,
            PatternType::Structural,
        ];
        
        assert_eq!(types.len(), 4);
        assert_eq!(PatternType::Literal, PatternType::Literal);
        assert_ne!(PatternType::Literal, PatternType::Regex);
    }

    #[test]
    fn test_performance_profile_variants() {
        let profiles = [
            PerformanceProfile::Speed,
            PerformanceProfile::Accuracy,
            PerformanceProfile::Balanced,
        ];
        
        assert_eq!(profiles.len(), 3);
        assert_eq!(PerformanceProfile::Speed, PerformanceProfile::Speed);
        assert_ne!(PerformanceProfile::Speed, PerformanceProfile::Accuracy);
    }

    #[test]
    fn test_issue_severity_variants() {
        let severities = [
            IssueSeverity::Error,
            IssueSeverity::Warning,
            IssueSeverity::Info,
        ];
        
        assert_eq!(severities.len(), 3);
        assert_eq!(IssueSeverity::Error, IssueSeverity::Error);
        assert_ne!(IssueSeverity::Error, IssueSeverity::Warning);
    }

    #[test]
    fn test_query_context_creation() {
        let context = QueryContext {
            user_id: Some("test".to_string()),
            workspace_path: "/test/workspace".to_string(),
            recent_files: vec!["file1.rs".to_string(), "file2.rs".to_string()],
            language_preference: Some("rust".to_string()),
            performance_profile: PerformanceProfile::Balanced,
        };
        
        assert_eq!(context.user_id, Some("test".to_string()));
        assert_eq!(context.workspace_path, "/test/workspace");
        assert_eq!(context.recent_files.len(), 2);
        assert_eq!(context.language_preference, Some("rust".to_string()));
        assert_eq!(context.performance_profile, PerformanceProfile::Balanced);
    }

    #[test]
    fn test_validation_result_structure() {
        let result = ValidationResult {
            valid: true,
            issues: vec![ValidationIssue {
                severity: IssueSeverity::Info,
                message: "Info message".to_string(),
                suggestion: Some("Suggestion".to_string()),
            }],
            suggestions: vec!["General suggestion".to_string()],
        };
        
        assert!(result.valid);
        assert_eq!(result.issues.len(), 1);
        assert_eq!(result.suggestions.len(), 1);
        assert_eq!(result.issues[0].severity, IssueSeverity::Info);
        assert_eq!(result.issues[0].suggestion, Some("Suggestion".to_string()));
    }

    #[test]
    fn test_query_default() {
        let query = Query::default();
        
        assert!(!query.id.is_empty());
        assert_eq!(query.original_query, "");
        assert_eq!(query.processed_query, "");
        assert_eq!(query.query_type, QueryType::Fuzzy);
        assert!(query.metadata.is_empty());
    }
}