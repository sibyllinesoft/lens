//! Comprehensive tests for query processing functionality
//! 
//! These tests focus on real user behavior and business logic validation
//! following Kent Beck's TDD principles: test behavior, not implementation.

use crate::query::*;
use std::collections::HashMap;

/// Test query creation with realistic user input scenarios
#[test]
fn test_query_creation_with_realistic_scenarios() {
    // Test common user search patterns
    let test_cases = vec![
        ("function authenticate", QueryType::Symbol),
        ("class DatabaseConnection", QueryType::Symbol),
        ("password", QueryType::Fuzzy),
        ("fn main() {", QueryType::Structural),
        ("error handling pattern", QueryType::Semantic),
        ("^pub struct.*Config", QueryType::Regex),
    ];
    
    for (input, expected_type) in test_cases {
        let query = Query {
            id: uuid::Uuid::new_v4().to_string(),
            original_query: input.to_string(),
            processed_query: input.to_string(),
            query_type: expected_type.clone(),
            filters: QueryFilters::default(),
            options: QueryOptions::default(),
            metadata: HashMap::new(),
        };
        
        assert_eq!(query.original_query, input);
        assert_eq!(query.query_type, expected_type);
        assert!(!query.id.is_empty());
    }
}

/// Test query filters behavior with realistic file filtering scenarios
#[test]
fn test_query_filters_realistic_scenarios() {
    let mut filters = QueryFilters::default();
    
    // Common filtering scenarios developers use
    filters.file_extensions = vec!["rs".to_string(), "toml".to_string()];
    filters.exclude_paths = vec!["target/".to_string(), "node_modules/".to_string()];
    filters.include_paths = vec!["src/".to_string(), "tests/".to_string()];
    filters.language_filter = Some("rust".to_string());
    filters.size_limit = Some(1024 * 1024); // 1MB limit
    
    // Verify filters work as expected
    assert_eq!(filters.file_extensions.len(), 2);
    assert!(filters.file_extensions.contains(&"rs".to_string()));
    assert!(filters.exclude_paths.contains(&"target/".to_string()));
    assert_eq!(filters.language_filter, Some("rust".to_string()));
    assert_eq!(filters.size_limit, Some(1024 * 1024));
}

/// Test query options with performance-oriented configurations
#[test]
fn test_query_options_performance_profiles() {
    let speed_options = QueryOptions {
        max_results: 20,
        timeout_ms: 500,
        systems: vec!["lex".to_string()],
        include_snippets: false,
        include_metadata: false,
        scoring_method: ScoringMethod::BM25,
        enable_caching: true,
        cache_ttl_seconds: 300,
        parallel_execution: true,
    };
    
    let accuracy_options = QueryOptions {
        max_results: 100,
        timeout_ms: 5000,
        systems: vec!["lex".to_string(), "semantic".to_string(), "symbols".to_string()],
        include_snippets: true,
        include_metadata: true,
        scoring_method: ScoringMethod::Hybrid,
        enable_caching: true,
        cache_ttl_seconds: 600,
        parallel_execution: true,
    };
    
    // Speed profile should prioritize fast response
    assert!(speed_options.timeout_ms < accuracy_options.timeout_ms);
    assert!(speed_options.max_results < accuracy_options.max_results);
    assert!(speed_options.systems.len() < accuracy_options.systems.len());
    
    // Accuracy profile should include more comprehensive search
    assert!(accuracy_options.include_snippets);
    assert!(accuracy_options.include_metadata);
    assert_eq!(accuracy_options.scoring_method, ScoringMethod::Hybrid);
}

/// Test query validation with real error scenarios developers encounter
#[test]
fn test_query_validation_real_scenarios() {
    let validator = BasicQueryValidator;
    
    // Test empty query (common mistake)
    let empty_query = Query {
        id: "test-1".to_string(),
        original_query: "".to_string(),
        processed_query: "".to_string(),
        query_type: QueryType::Fuzzy,
        filters: QueryFilters::default(),
        options: QueryOptions::default(),
        metadata: HashMap::new(),
    };
    
    let validation_result = validator.validate(&empty_query).unwrap();
    assert!(!validation_result.valid);
    assert!(!validation_result.issues.is_empty());
    assert_eq!(validation_result.issues[0].severity, IssueSeverity::Error);
    
    // Test very long query (performance concern)
    let long_query = Query {
        id: "test-2".to_string(),
        original_query: "x".repeat(1500),
        processed_query: "x".repeat(1500),
        query_type: QueryType::Fuzzy,
        filters: QueryFilters::default(),
        options: QueryOptions::default(),
        metadata: HashMap::new(),
    };
    
    let validation_result = validator.validate(&long_query).unwrap();
    // Should be valid but with warnings
    assert!(validation_result.valid);
    assert!(validation_result.issues.iter().any(|i| i.severity == IssueSeverity::Warning));
    
    // Test valid query
    let valid_query = Query {
        id: "test-3".to_string(),
        original_query: "function main".to_string(),
        processed_query: "function main".to_string(),
        query_type: QueryType::Symbol,
        filters: QueryFilters::default(),
        options: QueryOptions::default(),
        metadata: HashMap::new(),
    };
    
    let validation_result = validator.validate(&valid_query).unwrap();
    assert!(validation_result.valid);
    assert!(validation_result.issues.is_empty());
}

/// Test query analysis with realistic complexity scenarios
#[test]
fn test_query_analysis_complexity_scenarios() {
    let analyzer = BasicQueryAnalyzer;
    
    let test_cases = vec![
        (
            "simple",
            QueryType::Exact,
            0.1, // Low complexity
            "Simple exact match should have low complexity"
        ),
        (
            "function.*async.*await",
            QueryType::Regex,
            0.7, // High complexity
            "Regex with multiple patterns should have high complexity"
        ),
        (
            "authentication security pattern",
            QueryType::Semantic,
            0.8, // Very high complexity
            "Semantic search should have high complexity"
        ),
        (
            "class Config",
            QueryType::Symbol,
            0.3, // Medium complexity
            "Symbol search should have medium complexity"
        ),
    ];
    
    for (query_text, query_type, expected_min_complexity, description) in test_cases {
        let query = Query {
            id: uuid::Uuid::new_v4().to_string(),
            original_query: query_text.to_string(),
            processed_query: query_text.to_string(),
            query_type,
            filters: QueryFilters::default(),
            options: QueryOptions::default(),
            metadata: HashMap::new(),
        };
        
        let context = QueryContext {
            user_id: Some("test-user".to_string()),
            workspace_path: "/test/workspace".to_string(),
            recent_files: vec!["main.rs".to_string(), "lib.rs".to_string()],
            language_preference: Some("rust".to_string()),
            performance_profile: PerformanceProfile::Balanced,
        };
        
        let analysis = analyzer.analyze(&query, &context).unwrap();
        
        assert!(
            analysis.complexity_score >= expected_min_complexity,
            "{}: Expected complexity >= {}, got {}",
            description,
            expected_min_complexity,
            analysis.complexity_score
        );
        
        assert!(analysis.complexity_score <= 1.0, "Complexity score should not exceed 1.0");
        assert!(!analysis.suggested_systems.is_empty(), "Should suggest at least one system");
    }
}

/// Test query processor integration with realistic workflow
#[test]
fn test_query_processor_integration_workflow() {
    let processor = QueryProcessor::new();
    
    // Simulate realistic user search workflow
    let user_input = "authentication middleware";
    
    let initial_query = Query {
        id: uuid::Uuid::new_v4().to_string(),
        original_query: user_input.to_string(),
        processed_query: user_input.to_string(),
        query_type: QueryType::Fuzzy,
        filters: QueryFilters {
            file_extensions: vec!["rs".to_string()],
            exclude_paths: vec!["target/".to_string()],
            include_paths: vec!["src/".to_string()],
            language_filter: Some("rust".to_string()),
            size_limit: None,
            modified_since: None,
            file_type: Some(FileType::Source),
        },
        options: QueryOptions::default(),
        metadata: HashMap::new(),
    };
    
    let context = QueryContext {
        user_id: Some("dev-user".to_string()),
        workspace_path: "/workspace/rust-project".to_string(),
        recent_files: vec!["auth.rs".to_string(), "middleware.rs".to_string()],
        language_preference: Some("rust".to_string()),
        performance_profile: PerformanceProfile::Balanced,
    };
    
    // Process the query
    let processed_query = processor.process_query(initial_query, &context).unwrap();
    
    // Verify processing results
    assert_eq!(processed_query.original_query, user_input);
    assert!(!processed_query.processed_query.is_empty());
    assert!(!processed_query.id.is_empty());
    
    // Validate the processed query
    let validation_result = processor.validate_query(&processed_query).unwrap();
    assert!(validation_result.valid, "Processed query should be valid");
    
    // Analyze the query
    let analysis = processor.analyze_query(&processed_query, &context).unwrap();
    assert!(analysis.complexity_score > 0.0);
    assert!(analysis.complexity_score <= 1.0);
    assert!(!analysis.suggested_systems.is_empty());
}

/// Test performance profile behavior differences
#[test]
fn test_performance_profiles_behavior() {
    let speed_context = QueryContext {
        user_id: Some("speed-user".to_string()),
        workspace_path: "/workspace".to_string(),
        recent_files: vec![],
        language_preference: None,
        performance_profile: PerformanceProfile::Speed,
    };
    
    let accuracy_context = QueryContext {
        user_id: Some("accuracy-user".to_string()),
        workspace_path: "/workspace".to_string(),
        recent_files: vec![],
        language_preference: None,
        performance_profile: PerformanceProfile::Accuracy,
    };
    
    let balanced_context = QueryContext {
        user_id: Some("balanced-user".to_string()),
        workspace_path: "/workspace".to_string(),
        recent_files: vec![],
        language_preference: None,
        performance_profile: PerformanceProfile::Balanced,
    };
    
    // Verify profiles are distinct
    assert_eq!(speed_context.performance_profile, PerformanceProfile::Speed);
    assert_eq!(accuracy_context.performance_profile, PerformanceProfile::Accuracy);
    assert_eq!(balanced_context.performance_profile, PerformanceProfile::Balanced);
    
    // All profiles should be usable for query processing
    assert!(speed_context.workspace_path.starts_with("/"));
    assert!(accuracy_context.workspace_path.starts_with("/"));
    assert!(balanced_context.workspace_path.starts_with("/"));
}

/// Test edge cases and error handling
#[test]
fn test_query_edge_cases_and_error_handling() {
    // Test with special characters that might break processing
    let special_chars_query = Query {
        id: uuid::Uuid::new_v4().to_string(),
        original_query: "function_name::module && <T: Clone>".to_string(),
        processed_query: "function_name::module && <T: Clone>".to_string(),
        query_type: QueryType::Structural,
        filters: QueryFilters::default(),
        options: QueryOptions::default(),
        metadata: HashMap::new(),
    };
    
    let validator = BasicQueryValidator;
    let validation_result = validator.validate(&special_chars_query).unwrap();
    
    // Should handle special characters gracefully
    assert!(validation_result.valid);
    
    // Test with Unicode characters
    let unicode_query = Query {
        id: uuid::Uuid::new_v4().to_string(),
        original_query: "函数 função función function".to_string(),
        processed_query: "函数 função función function".to_string(),
        query_type: QueryType::Fuzzy,
        filters: QueryFilters::default(),
        options: QueryOptions::default(),
        metadata: HashMap::new(),
    };
    
    let validation_result = validator.validate(&unicode_query).unwrap();
    assert!(validation_result.valid, "Should handle Unicode gracefully");
}

/// Test query metadata handling
#[test]
fn test_query_metadata_handling() {
    let mut metadata = HashMap::new();
    metadata.insert("source".to_string(), "ide".to_string());
    metadata.insert("session_id".to_string(), "abc123".to_string());
    metadata.insert("timestamp".to_string(), "2024-01-01T00:00:00Z".to_string());
    
    let query = Query {
        id: uuid::Uuid::new_v4().to_string(),
        original_query: "search term".to_string(),
        processed_query: "search term".to_string(),
        query_type: QueryType::Fuzzy,
        filters: QueryFilters::default(),
        options: QueryOptions::default(),
        metadata: metadata.clone(),
    };
    
    // Verify metadata is preserved
    assert_eq!(query.metadata.get("source"), Some(&"ide".to_string()));
    assert_eq!(query.metadata.get("session_id"), Some(&"abc123".to_string()));
    assert_eq!(query.metadata.len(), 3);
    
    // Test that metadata doesn't affect validation
    let validator = BasicQueryValidator;
    let validation_result = validator.validate(&query).unwrap();
    assert!(validation_result.valid);
}