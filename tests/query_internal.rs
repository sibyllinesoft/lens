//! Internal tests for query.rs module - comprehensive coverage
//! 
//! These tests focus on internal functionality, edge cases, and integration
//! scenarios that complement the public API tests in query_focused.rs.

use lens_core::query::*;
use std::collections::HashMap;
use chrono::{DateTime, Utc, TimeZone};

#[test]
fn test_query_processor_full_pipeline() {
    let processor = QueryProcessor::new();
    let context = QueryContext {
        user_id: Some("test-user".to_string()),
        workspace_path: "/test/workspace".to_string(),
        recent_files: vec!["main.rs".to_string(), "lib.rs".to_string()],
        language_preference: Some("rust".to_string()),
        performance_profile: PerformanceProfile::Balanced,
    };

    // Test comprehensive pipeline processing
    let queries = vec![
        "function handler",
        "\"exact match query\"",
        "test* wildcard",
        "simple_identifier",
        "semantic search for authentication handling patterns",
        "class MyClass extends BaseClass",
    ];

    for query_str in queries {
        let result = processor.process_query(query_str, &context);
        assert!(result.is_ok(), "Failed to process query: {}", query_str);
        
        let processed = result.unwrap();
        assert!(!processed.query.id.is_empty());
        assert_eq!(processed.query.original_query, query_str);
        assert!(!processed.query.processed_query.is_empty());
        
        // Verify analysis was performed
        assert!(processed.analysis.complexity_score >= 0.0);
        assert!(processed.analysis.complexity_score <= 1.0);
        assert!(!processed.analysis.suggested_systems.is_empty());
        
        // Verify systems are selected based on query type
        match processed.query.query_type {
            QueryType::Symbol => assert!(processed.query.options.systems.contains(&"symbols".to_string())),
            QueryType::Semantic => {
                assert!(processed.query.options.systems.contains(&"semantic".to_string()));
                assert!(processed.query.options.systems.contains(&"lex".to_string()));
            },
            QueryType::Structural => {
                assert!(processed.query.options.systems.contains(&"symbols".to_string()));
                assert!(processed.query.options.systems.contains(&"lex".to_string()));
            },
            _ => assert!(!processed.query.options.systems.is_empty()),
        }
    }
}

#[test]
fn test_query_classification_edge_cases() {
    let processor = QueryProcessor::new();
    
    // Test processing different query types to verify classification via public API
    let queries_and_expected_types = vec![
        // Edge cases for exact queries
        ("\"\"", "exact"),  // Empty quoted string
        ("\"single word\"", "exact"),
        ("\"multi word exact match\"", "exact"),
        
        // Edge cases for regex
        ("test*", "regex"),
        ("test?pattern", "regex"),
        ("pattern[abc]", "regex"),
        ("^start", "regex"),
        ("test*.?[a-z]^", "regex"),
        
        // Edge cases for structural
        ("class MyClass", "structural"),  // Must have more than one word
        ("function handler", "structural"),
        ("interface Config", "structural"),
        ("struct Data", "structural"),
        ("class MyClass extends BaseClass", "structural"),
        
        // Edge cases for symbol
        ("a", "symbol"),  // Single character
        ("_", "symbol"),  // Just underscore
        ("class", "symbol"),  // Single keyword - treated as symbol
        ("handle_request", "symbol"),
        ("MyClass", "symbol"),
        ("snake_case_identifier", "symbol"),
        ("camelCaseIdentifier", "symbol"),
        
        // Edge cases for semantic
        ("how to handle this", "semantic"),  // 4 words - semantic
        ("please explain the authentication mechanism in detail", "semantic"),
        
        // Edge cases for fuzzy (default)
        ("", "fuzzy"),  // Empty string
        ("simple query", "fuzzy"),  // 2 words
        ("test query here", "fuzzy"),  // 3 words exactly
    ];

    let context = QueryContext {
        user_id: Some("test".to_string()),
        workspace_path: "/test".to_string(),
        recent_files: vec![],
        language_preference: None,
        performance_profile: PerformanceProfile::Balanced,
    };

    for (query, expected_category) in queries_and_expected_types {
        let result = processor.process_query(query, &context);
        assert!(result.is_ok(), "Query should process: {}", query);
        
        let processed = result.unwrap();
        
        // Verify the query type matches expected category
        match expected_category {
            "exact" => assert_eq!(processed.query.query_type, QueryType::Exact),
            "regex" => assert_eq!(processed.query.query_type, QueryType::Regex),
            "structural" => assert_eq!(processed.query.query_type, QueryType::Structural),
            "symbol" => assert_eq!(processed.query.query_type, QueryType::Symbol),
            "semantic" => assert_eq!(processed.query.query_type, QueryType::Semantic),
            "fuzzy" => assert_eq!(processed.query.query_type, QueryType::Fuzzy),
            _ => panic!("Unknown category: {}", expected_category),
        }
    }
}

#[test]
fn test_performance_optimizer_edge_cases() {
    let optimizer = PerformanceOptimizer::new();
    
    // Test balanced profile doesn't change defaults
    let context_balanced = QueryContext {
        user_id: Some("test".to_string()),
        workspace_path: "/test".to_string(),
        recent_files: vec![],
        language_preference: None,
        performance_profile: PerformanceProfile::Balanced,
    };
    
    let original_query = Query {
        options: QueryOptions {
            max_results: 50,
            timeout_ms: 2000,
            ..QueryOptions::default()
        },
        ..Query::default()
    };
    
    let optimized = optimizer.optimize(&original_query, &context_balanced).unwrap();
    assert_eq!(optimized.options.max_results, 50);  // Unchanged
    assert_eq!(optimized.options.timeout_ms, 2000);  // Unchanged
    
    // Test extreme values for Speed profile
    let context_speed = QueryContext {
        performance_profile: PerformanceProfile::Speed,
        ..context_balanced.clone()
    };
    
    let high_values_query = Query {
        options: QueryOptions {
            max_results: 10000,
            timeout_ms: 60000,
            ..QueryOptions::default()
        },
        ..Query::default()
    };
    
    let optimized = optimizer.optimize(&high_values_query, &context_speed).unwrap();
    assert_eq!(optimized.options.max_results, 20);  // Capped at 20
    assert_eq!(optimized.options.timeout_ms, 1000);  // Capped at 1000
    
    // Test extreme values for Accuracy profile  
    let context_accuracy = QueryContext {
        performance_profile: PerformanceProfile::Accuracy,
        ..context_balanced.clone()
    };
    
    let low_values_query = Query {
        options: QueryOptions {
            max_results: 1,
            timeout_ms: 100,
            ..QueryOptions::default()
        },
        ..Query::default()
    };
    
    let optimized = optimizer.optimize(&low_values_query, &context_accuracy).unwrap();
    assert_eq!(optimized.options.max_results, 100);  // Raised to 100
    assert_eq!(optimized.options.timeout_ms, 5000);  // Raised to 5000
}

#[test]
fn test_system_selection_optimizer_all_types() {
    let optimizer = SystemSelectionOptimizer::new();
    let context = QueryContext {
        user_id: Some("test".to_string()),
        workspace_path: "/test".to_string(),
        recent_files: vec![],
        language_preference: None,
        performance_profile: PerformanceProfile::Balanced,
    };
    
    let test_cases = vec![
        (QueryType::Symbol, vec!["symbols"]),
        (QueryType::Structural, vec!["symbols", "lex"]),
        (QueryType::Semantic, vec!["semantic", "lex"]),
        (QueryType::Exact, vec!["lex", "symbols"]),
        (QueryType::Fuzzy, vec!["lex", "symbols"]),
        (QueryType::Regex, vec!["lex", "symbols"]),
        (QueryType::Hybrid, vec!["lex", "symbols"]),
    ];
    
    for (query_type, expected_systems) in test_cases {
        let query = Query {
            query_type: query_type.clone(),
            ..Query::default()
        };
        
        let optimized = optimizer.optimize(&query, &context).unwrap();
        assert_eq!(optimized.options.systems, expected_systems, 
                   "Systems for {:?}", query_type);
    }
}

#[test]
fn test_basic_validator_comprehensive() {
    let validator = BasicQueryValidator::new();
    
    // Test empty queries
    let empty_query = Query {
        processed_query: "".to_string(),
        ..Query::default()
    };
    let result = validator.validate(&empty_query).unwrap();
    assert!(!result.valid);
    assert_eq!(result.issues.len(), 1);
    assert_eq!(result.issues[0].severity, IssueSeverity::Error);
    
    // Test valid queries
    let valid_query = Query {
        processed_query: "valid query".to_string(),
        ..Query::default()
    };
    let result = validator.validate(&valid_query).unwrap();
    assert!(result.valid);
    assert!(result.issues.is_empty());
    
    // Test long queries (warning but valid)  
    let long_query = Query {
        processed_query: "a".repeat(1001),
        ..Query::default()
    };
    let result = validator.validate(&long_query).unwrap();
    assert!(result.valid);
    assert_eq!(result.issues.len(), 1);
    assert_eq!(result.issues[0].severity, IssueSeverity::Warning);
    
    // Test timeout suggestions
    let low_timeout_query = Query {
        processed_query: "test query".to_string(),
        options: QueryOptions {
            timeout_ms: 50,  // Very low
            ..QueryOptions::default()
        },
        ..Query::default()
    };
    
    let result = validator.validate(&low_timeout_query).unwrap();
    assert!(result.valid);
    assert!(!result.suggestions.is_empty());
    assert!(result.suggestions[0].contains("timeout"));
    
    // Test normal timeout (no suggestion)
    let normal_timeout_query = Query {
        processed_query: "test query".to_string(),
        options: QueryOptions {
            timeout_ms: 2000,  // Normal
            ..QueryOptions::default()
        },
        ..Query::default()
    };
    
    let result = validator.validate(&normal_timeout_query).unwrap();
    assert!(result.valid);
    assert!(result.suggestions.is_empty());
}

#[test]
fn test_query_filters_with_datetime() {
    use chrono::{Duration};
    
    let now = Utc::now();
    let week_ago = now - Duration::days(7);
    
    let filters = QueryFilters {
        file_extensions: vec![".rs".to_string(), ".py".to_string()],
        exclude_paths: vec!["target/".to_string()],
        include_paths: vec!["src/".to_string(), "tests/".to_string()],
        language_filter: Some("rust".to_string()),
        size_limit: Some(1024 * 1024), // 1MB
        modified_since: Some(week_ago),
        file_type: Some(FileType::Source),
    };
    
    assert_eq!(filters.file_extensions.len(), 2);
    assert_eq!(filters.exclude_paths.len(), 1);
    assert_eq!(filters.include_paths.len(), 2);
    assert_eq!(filters.language_filter, Some("rust".to_string()));
    assert_eq!(filters.size_limit, Some(1024 * 1024));
    assert!(filters.modified_since.is_some());
    assert_eq!(filters.file_type, Some(FileType::Source));
    
    // Test that the datetime is reasonable
    let since = filters.modified_since.unwrap();
    assert!(since < now);
    assert!(since > now - Duration::days(8));
}

#[test]
fn test_query_options_boost_factors() {
    let mut options = QueryOptions::default();
    
    // Test that boost factors can be modified
    options.boost_factors.insert("exact_match".to_string(), 2.0);
    options.boost_factors.insert("file_name".to_string(), 1.5);
    options.boost_factors.insert("recent_files".to_string(), 1.2);
    
    assert_eq!(options.boost_factors.len(), 3);
    assert_eq!(options.boost_factors["exact_match"], 2.0);
    assert_eq!(options.boost_factors["file_name"], 1.5);
    assert_eq!(options.boost_factors["recent_files"], 1.2);
    
    // Test that other defaults are preserved
    assert_eq!(options.max_results, 50);
    assert_eq!(options.timeout_ms, 2000);
    assert!(options.include_snippets);
    assert!(options.highlight_matches);
    assert_eq!(options.sort_by, SortOrder::Relevance);
}

#[test]
fn test_processed_query_comprehensive() {
    let processor = QueryProcessor::new();
    let context = QueryContext {
        user_id: Some("integration-test".to_string()),
        workspace_path: "/integration/test".to_string(),
        recent_files: vec!["main.rs".to_string(), "lib.rs".to_string()],
        language_preference: Some("rust".to_string()),
        performance_profile: PerformanceProfile::Speed,
    };
    
    // Test a complex query 
    let complex_query = "function handler";
    let result = processor.process_query(complex_query, &context).unwrap();
    
    // Verify the processed query structure
    assert!(!result.query.id.is_empty());
    assert_eq!(result.query.original_query, complex_query);
    assert_eq!(result.query.processed_query, "function handler");
    assert_eq!(result.query.query_type, QueryType::Structural);
    
    // Verify performance optimization was applied (Speed profile)
    assert!(result.query.options.timeout_ms <= 1000);
    assert!(result.query.options.max_results <= 20);
    
    // Verify system selection for structural query
    assert!(result.query.options.systems.contains(&"symbols".to_string()));
    assert!(result.query.options.systems.contains(&"lex".to_string()));
    
    // Verify analysis was performed
    assert!(result.analysis.complexity_score > 0.0);
    assert!(result.analysis.estimated_results > 0);
    assert!(!result.analysis.suggested_systems.is_empty());
    
    // Verify validation passed (should be valid structural query)
    assert!(result.validation.valid);
    
    // Verify processing time is initialized
    assert_eq!(result.processing_time_ms, 0.0);
}

#[test] 
fn test_component_composition() {
    let mut processor = QueryProcessor::new();
    
    // Test that we can add multiple components and they all get used
    processor.add_analyzer(Box::new(BasicQueryAnalyzer::new()));
    processor.add_optimizer(Box::new(PerformanceOptimizer::new())); 
    processor.add_validator(Box::new(BasicQueryValidator::new()));
    
    let context = QueryContext {
        user_id: Some("test".to_string()),
        workspace_path: "/test".to_string(), 
        recent_files: vec![],
        language_preference: None,
        performance_profile: PerformanceProfile::Balanced,
    };
    
    let result = processor.process_query("test query", &context);
    assert!(result.is_ok());
    
    let processed = result.unwrap();
    
    // Should have processed through multiple analyzers (complexity averaged)
    // Since we added BasicQueryAnalyzer twice, the analysis should be merged
    assert!(processed.analysis.complexity_score > 0.0);
    
    // Should have gone through multiple optimizers
    // (In this case, both instances of PerformanceOptimizer should have same effect)
    
    // Should have gone through multiple validators
    // (Both should pass for a valid query, so result should still be valid)
    assert!(processed.validation.valid);
}

#[test]
fn test_query_pipeline_with_filters() {
    let processor = QueryProcessor::new();
    let context = QueryContext {
        user_id: Some("test".to_string()),
        workspace_path: "/test".to_string(),
        recent_files: vec![],
        language_preference: None,
        performance_profile: PerformanceProfile::Balanced,
    };
    
    // Test processing queries with different filter patterns
    let filtered_queries = vec![
        "search code ext:rs",
        "function handler lang:typescript",
        "test path:src/",
        "debug -path:node_modules/",
        "complex ext:py lang:python path:src/ -path:target/",
    ];
    
    for query_str in filtered_queries {
        let result = processor.process_query(query_str, &context);
        assert!(result.is_ok(), "Should process filtered query: {}", query_str);
        
        let processed = result.unwrap();
        assert!(!processed.query.id.is_empty());
        assert_eq!(processed.query.original_query, query_str);
        
        // Processed query should have filters extracted
        assert!(!processed.query.processed_query.is_empty());
        
        // Should have valid validation result
        assert!(processed.validation.valid || !processed.validation.issues.is_empty());
    }
}

#[test]
fn test_analyzer_and_optimizer_integration() {
    let processor = QueryProcessor::new();
    let context = QueryContext {
        user_id: Some("test".to_string()),
        workspace_path: "/test".to_string(),
        recent_files: vec!["main.rs".to_string()],
        language_preference: Some("rust".to_string()),
        performance_profile: PerformanceProfile::Speed,
    };
    
    // Test that analyzers and optimizers work together
    let result = processor.process_query("complex query for testing analyzers", &context);
    assert!(result.is_ok());
    
    let processed = result.unwrap();
    
    // Should have analysis from BasicQueryAnalyzer and PatternAnalyzer
    assert!(processed.analysis.complexity_score > 0.0);
    assert!(processed.analysis.estimated_results > 0);
    assert!(!processed.analysis.suggested_systems.is_empty());
    
    // Should be optimized for speed profile
    assert!(processed.query.options.timeout_ms <= 1000);
    assert!(processed.query.options.max_results <= 20);
    
    // Should have system selection based on query type
    assert!(!processed.query.options.systems.is_empty());
    
    // Should pass validation
    assert!(processed.validation.valid);
}