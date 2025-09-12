//! Comprehensive tests for query.rs module
//! 
//! These tests cover:
//! - Query parsing and validation
//! - Query transformation and optimization  
//! - Query execution and result processing
//! - Error handling and edge cases
//! - Performance and configuration aspects

use lens_core::query::*;
use std::collections::HashMap;
use std::time::Instant;

#[test]
fn test_query_creation_comprehensive() {
    let query = Query {
        id: "comprehensive-test-id".to_string(),
        original_query: "comprehensive test query".to_string(),
        processed_query: "comprehensive test query".to_string(),
        query_type: QueryType::Semantic,
        filters: QueryFilters::default(),
        options: QueryOptions::default(),
        metadata: HashMap::new(),
    };
    
    assert_eq!(query.id, "comprehensive-test-id");
    assert_eq!(query.original_query, "comprehensive test query");
    assert_eq!(query.query_type, QueryType::Semantic);
    assert!(query.metadata.is_empty());
}

#[test]
fn test_query_type_serialization() {
    use serde_json;
    
    let query_type = QueryType::Structural;
    let serialized = serde_json::to_string(&query_type).unwrap();
    assert_eq!(serialized, "\"structural\"");
    
    let deserialized: QueryType = serde_json::from_str(&serialized).unwrap();
    assert_eq!(deserialized, QueryType::Structural);
}

#[test]
fn test_all_query_type_variants() {
    let types = [
        QueryType::Exact,
        QueryType::Fuzzy,
        QueryType::Structural,
        QueryType::Semantic,
        QueryType::Symbol,
        QueryType::Hybrid,
        QueryType::Regex,
    ];
    
    // Test default
    assert_eq!(QueryType::default(), QueryType::Fuzzy);
    
    // Test all variants are unique
    for (i, type1) in types.iter().enumerate() {
        for (j, type2) in types.iter().enumerate() {
            if i != j {
                assert_ne!(type1, type2);
            } else {
                assert_eq!(type1, type2);
            }
        }
    }
}

#[test]
fn test_query_filters_comprehensive() {
    let mut filters = QueryFilters::default();
    
    // Test initial state
    assert!(filters.file_extensions.is_empty());
    assert!(filters.exclude_paths.is_empty());
    assert!(filters.include_paths.is_empty());
    assert_eq!(filters.language_filter, None);
    assert_eq!(filters.size_limit, None);
    assert_eq!(filters.modified_since, None);
    assert_eq!(filters.file_type, None);
    
    // Test population
    filters.file_extensions = vec![".rs".to_string(), ".ts".to_string()];
    filters.exclude_paths = vec!["target/".to_string(), "node_modules/".to_string()];
    filters.include_paths = vec!["src/".to_string(), "tests/".to_string()];
    filters.language_filter = Some("rust".to_string());
    filters.size_limit = Some(1024 * 1024); // 1MB
    filters.file_type = Some(FileType::Source);
    
    assert_eq!(filters.file_extensions.len(), 2);
    assert_eq!(filters.exclude_paths.len(), 2);
    assert_eq!(filters.include_paths.len(), 2);
    assert_eq!(filters.language_filter, Some("rust".to_string()));
    assert_eq!(filters.size_limit, Some(1024 * 1024));
    assert_eq!(filters.file_type, Some(FileType::Source));
}

#[test]
fn test_file_type_comprehensive() {
    let types = [
        FileType::Source,
        FileType::Test,
        FileType::Configuration,
        FileType::Documentation,
        FileType::Build,
        FileType::Data,
    ];
    
    assert_eq!(FileType::default(), FileType::Source);
    
    // Test serialization
    use serde_json;
    let file_type = FileType::Documentation;
    let serialized = serde_json::to_string(&file_type).unwrap();
    assert_eq!(serialized, "\"documentation\"");
}

#[test]
fn test_query_options_comprehensive() {
    let mut options = QueryOptions::default();
    
    // Test defaults
    assert_eq!(options.max_results, 50);
    assert_eq!(options.timeout_ms, 2000);
    assert!(options.systems.is_empty());
    assert!(options.include_snippets);
    assert!(options.highlight_matches);
    assert_eq!(options.sort_by, SortOrder::Relevance);
    assert!(options.boost_factors.is_empty());
    
    // Test modification
    options.max_results = 100;
    options.timeout_ms = 5000;
    options.systems = vec!["lex".to_string(), "semantic".to_string()];
    options.include_snippets = false;
    options.highlight_matches = false;
    options.sort_by = SortOrder::Modified;
    options.boost_factors.insert("language:rust".to_string(), 2.0);
    
    assert_eq!(options.max_results, 100);
    assert_eq!(options.timeout_ms, 5000);
    assert_eq!(options.systems.len(), 2);
    assert!(!options.include_snippets);
    assert!(!options.highlight_matches);
    assert_eq!(options.sort_by, SortOrder::Modified);
    assert_eq!(options.boost_factors.get("language:rust"), Some(&2.0));
}

#[test]
fn test_sort_order_comprehensive() {
    let orders = [
        SortOrder::Relevance,
        SortOrder::Modified,
        SortOrder::Created,
        SortOrder::Size,
        SortOrder::Name,
        SortOrder::Language,
    ];
    
    assert_eq!(SortOrder::default(), SortOrder::Relevance);
    
    // Test serialization
    use serde_json;
    let sort_order = SortOrder::Modified;
    let serialized = serde_json::to_string(&sort_order).unwrap();
    assert_eq!(serialized, "\"modified\"");
}

#[test]
fn test_query_processor_creation() {
    let processor = QueryProcessor::new();
    
    // Processor should be created successfully
    // Test via public API
    let context = QueryContext {
        user_id: Some("test".to_string()),
        workspace_path: "/test".to_string(),
        recent_files: vec![],
        language_preference: None,
        performance_profile: PerformanceProfile::Balanced,
    };
    
    let result = processor.process_query("test query", &context);
    assert!(result.is_ok());
}

#[test]
fn test_query_processor_component_addition() {
    let mut processor = QueryProcessor::new();
    
    // Add components (test that it doesn't panic)
    processor.add_analyzer(Box::new(BasicQueryAnalyzer::new()));
    processor.add_optimizer(Box::new(PerformanceOptimizer::new()));
    processor.add_validator(Box::new(BasicQueryValidator::new()));
    
    // Test that processor still works after adding components
    let context = QueryContext {
        user_id: Some("test".to_string()),
        workspace_path: "/test".to_string(),
        recent_files: vec![],
        language_preference: None,
        performance_profile: PerformanceProfile::Balanced,
    };
    
    let result = processor.process_query("test query", &context);
    assert!(result.is_ok());
}

#[test]
fn test_process_query_comprehensive() {
    let processor = QueryProcessor::new();
    let context = QueryContext {
        user_id: Some("test".to_string()),
        workspace_path: "/test".to_string(),
        recent_files: vec![],
        language_preference: None,
        performance_profile: PerformanceProfile::Balanced,
    };
    
    // Test processing various query types
    let result = processor.process_query("search function", &context);
    assert!(result.is_ok());
    
    let processed_query = result.unwrap();
    assert!(!processed_query.query.id.is_empty());
    assert_eq!(processed_query.query.original_query, "search function");
    assert!(processed_query.validation.valid || !processed_query.validation.issues.is_empty());
}

#[test]
fn test_filter_extraction_edge_cases() {
    let processor = QueryProcessor::new();
    let context = QueryContext {
        user_id: None,
        workspace_path: "/test".to_string(),
        recent_files: vec![],
        language_preference: None,
        performance_profile: PerformanceProfile::Balanced,
    };
    
    // Empty query
    let result = processor.process_query("", &context).unwrap();
    let processed = &result.query.processed_query;
    let filters = &result.query.filters;
    assert_eq!(processed, "");
    assert!(filters.file_extensions.is_empty());
    
    // Only filters
    let result = processor.process_query("ext:py lang:python", &context).unwrap();
    let processed = &result.query.processed_query;
    let filters = &result.query.filters;
    assert_eq!(processed, "");
    assert_eq!(filters.file_extensions, vec![".py"]);
    assert_eq!(filters.language_filter, Some("python".to_string()));
    
    // Malformed filters (should be ignored)
    let result = processor.process_query("search ext: lang: path:", &context).unwrap();
    let processed = &result.query.processed_query;
    let filters = &result.query.filters;
    assert_eq!(processed, "search ext: lang: path:");
    assert!(filters.file_extensions.is_empty());
    assert_eq!(filters.language_filter, None);
}

#[test]
fn test_query_classification_comprehensive() {
    let processor = QueryProcessor::new();
    let context = QueryContext {
        user_id: None,
        workspace_path: "/test".to_string(),
        recent_files: vec![],
        language_preference: None,
        performance_profile: PerformanceProfile::Balanced,
    };
    
    // Exact match variations
    assert_eq!(processor.process_query("\"exact match\"", &context).unwrap().query.query_type, QueryType::Exact);
    assert_eq!(processor.process_query("\"\"", &context).unwrap().query.query_type, QueryType::Exact); // Empty quotes
    
    // Regex variations
    assert_eq!(processor.process_query("test*pattern", &context).unwrap().query.query_type, QueryType::Regex);
    assert_eq!(processor.process_query("test?pattern", &context).unwrap().query.query_type, QueryType::Regex);
    assert_eq!(processor.process_query("test[0-9]", &context).unwrap().query.query_type, QueryType::Regex);
    assert_eq!(processor.process_query("^start", &context).unwrap().query.query_type, QueryType::Regex);
    assert_eq!(processor.process_query("end$", &context).unwrap().query.query_type, QueryType::Regex);
    
    // Structural variations
    assert_eq!(processor.process_query("class MyClass", &context).unwrap().query.query_type, QueryType::Structural);
    assert_eq!(processor.process_query("function handler", &context).unwrap().query.query_type, QueryType::Structural);
    assert_eq!(processor.process_query("interface Config", &context).unwrap().query.query_type, QueryType::Structural);
    assert_eq!(processor.process_query("struct Data", &context).unwrap().query.query_type, QueryType::Structural);
    
    // Symbol variations
    assert_eq!(processor.process_query("handleRequest", &context).unwrap().query.query_type, QueryType::Symbol);
    assert_eq!(processor.process_query("my_variable", &context).unwrap().query.query_type, QueryType::Symbol);
    assert_eq!(processor.process_query("MY_CONSTANT", &context).unwrap().query.query_type, QueryType::Symbol);
    assert_eq!(processor.process_query("camelCase123", &context).unwrap().query.query_type, QueryType::Symbol);
    
    // Semantic variations  
    assert_eq!(processor.process_query("how to handle user authentication in web applications", &context).unwrap().query.query_type, QueryType::Semantic);
    assert_eq!(processor.process_query("what is the best practice for error handling", &context).unwrap().query.query_type, QueryType::Semantic);
    
    // Fuzzy (default) variations
    assert_eq!(processor.process_query("test query", &context).unwrap().query.query_type, QueryType::Fuzzy);
    assert_eq!(processor.process_query("simple text", &context).unwrap().query.query_type, QueryType::Fuzzy);
    assert_eq!(processor.process_query("two words", &context).unwrap().query.query_type, QueryType::Fuzzy);
}

#[test]
fn test_query_analysis_comprehensive() {
    let analyzer = BasicQueryAnalyzer::new();
    
    // Test simple query
    let result = analyzer.analyze("simple query").unwrap();
    assert!(result.complexity_score >= 0.0);
    assert!(result.complexity_score <= 1.0);
    assert!(result.estimated_results > 0);
    assert!(!result.suggested_systems.is_empty());
    
    // Test complex query (>100 chars)
    let complex_query = "a".repeat(150);
    let result = analyzer.analyze(&complex_query).unwrap();
    assert_eq!(result.complexity_score, 0.8);
    
    // Test very short query
    let result = analyzer.analyze("a").unwrap();
    assert_eq!(result.complexity_score, 0.4);
}

#[test]
fn test_pattern_analyzer_comprehensive() {
    let analyzer = PatternAnalyzer::new();
    
    // Test wildcard patterns
    let result = analyzer.analyze("test*pattern").unwrap();
    assert_eq!(result.pattern_type, PatternType::Wildcard);
    
    let result = analyzer.analyze("test?pattern").unwrap();
    assert_eq!(result.pattern_type, PatternType::Wildcard);
    
    // Test structural patterns
    let result = analyzer.analyze("class Test").unwrap();
    assert_eq!(result.pattern_type, PatternType::Structural);
    
    let result = analyzer.analyze("function handler").unwrap();
    assert_eq!(result.pattern_type, PatternType::Structural);
    
    // Test literal patterns
    let result = analyzer.analyze("simple text").unwrap();
    assert_eq!(result.pattern_type, PatternType::Literal);
}

#[test]
fn test_performance_optimizer_comprehensive() {
    let optimizer = PerformanceOptimizer::new();
    
    // Test Speed profile
    let context = QueryContext {
        user_id: Some("test".to_string()),
        workspace_path: "/test/workspace".to_string(),
        recent_files: vec![],
        language_preference: None,
        performance_profile: PerformanceProfile::Speed,
    };
    
    let query = Query {
        options: QueryOptions {
            max_results: 1000,
            timeout_ms: 10000,
            ..QueryOptions::default()
        },
        ..Query::default()
    };
    
    let optimized = optimizer.optimize(&query, &context).unwrap();
    assert!(optimized.options.timeout_ms <= 1000);
    assert!(optimized.options.max_results <= 20);
    
    // Test Accuracy profile
    let context = QueryContext {
        performance_profile: PerformanceProfile::Accuracy,
        ..context
    };
    
    let query = Query {
        options: QueryOptions {
            max_results: 5,
            timeout_ms: 500,
            ..QueryOptions::default()
        },
        ..Query::default()
    };
    
    let optimized = optimizer.optimize(&query, &context).unwrap();
    assert!(optimized.options.timeout_ms >= 5000);
    assert!(optimized.options.max_results >= 100);
    
    // Test Balanced profile (should keep defaults)
    let context = QueryContext {
        performance_profile: PerformanceProfile::Balanced,
        ..context
    };
    
    let original_timeout = query.options.timeout_ms;
    let original_max_results = query.options.max_results;
    let optimized = optimizer.optimize(&query, &context).unwrap();
    assert_eq!(optimized.options.timeout_ms, original_timeout);
    assert_eq!(optimized.options.max_results, original_max_results);
}

#[test]
fn test_system_selection_optimizer_comprehensive() {
    let optimizer = SystemSelectionOptimizer::new();
    let context = QueryContext {
        user_id: Some("test".to_string()),
        workspace_path: "/test/workspace".to_string(),
        recent_files: vec![],
        language_preference: None,
        performance_profile: PerformanceProfile::Balanced,
    };
    
    // Test all query types
    let test_cases = [
        (QueryType::Symbol, vec!["symbols"]),
        (QueryType::Structural, vec!["symbols", "lex"]),
        (QueryType::Semantic, vec!["semantic", "lex"]),
        (QueryType::Exact, vec!["lex", "symbols"]),
        (QueryType::Fuzzy, vec!["lex", "symbols"]),
        (QueryType::Hybrid, vec!["lex", "symbols"]),
        (QueryType::Regex, vec!["lex", "symbols"]),
    ];
    
    for (query_type, expected_systems) in test_cases.iter() {
        let query = Query {
            query_type: query_type.clone(),
            ..Query::default()
        };
        
        let optimized = optimizer.optimize(&query, &context).unwrap();
        assert_eq!(optimized.options.systems, *expected_systems);
    }
}

#[test]
fn test_basic_query_validator_comprehensive() {
    let validator = BasicQueryValidator::new();
    
    // Test empty query
    let query = Query {
        processed_query: "".to_string(),
        ..Query::default()
    };
    let result = validator.validate(&query).unwrap();
    assert!(!result.valid);
    assert_eq!(result.issues.len(), 1);
    assert_eq!(result.issues[0].severity, IssueSeverity::Error);
    assert!(result.issues[0].message.contains("empty"));
    
    // Test whitespace-only query
    let query = Query {
        processed_query: "   \t\n  ".to_string(),
        ..Query::default()
    };
    let result = validator.validate(&query).unwrap();
    assert!(!result.valid);
    
    // Test very long query
    let long_query = "a".repeat(1001);
    let query = Query {
        processed_query: long_query,
        ..Query::default()
    };
    let result = validator.validate(&query).unwrap();
    assert!(result.valid); // Still valid, just warning
    assert_eq!(result.issues.len(), 1);
    assert_eq!(result.issues[0].severity, IssueSeverity::Warning);
    
    // Test exactly 1000 chars (boundary)
    let boundary_query = "a".repeat(1000);
    let query = Query {
        processed_query: boundary_query,
        ..Query::default()
    };
    let result = validator.validate(&query).unwrap();
    assert!(result.valid);
    assert!(result.issues.is_empty());
    
    // Test low timeout
    let query = Query {
        processed_query: "test".to_string(),
        options: QueryOptions {
            timeout_ms: 50,
            ..QueryOptions::default()
        },
        ..Query::default()
    };
    let result = validator.validate(&query).unwrap();
    assert!(result.valid);
    assert!(!result.suggestions.is_empty());
    assert!(result.suggestions[0].contains("timeout"));
    
    // Test boundary timeout (100ms)
    let query = Query {
        processed_query: "test".to_string(),
        options: QueryOptions {
            timeout_ms: 100,
            ..QueryOptions::default()
        },
        ..Query::default()
    };
    let result = validator.validate(&query).unwrap();
    assert!(result.valid);
    assert!(result.suggestions.is_empty());
    
    // Test valid query
    let query = Query {
        processed_query: "valid query".to_string(),
        options: QueryOptions {
            timeout_ms: 2000,
            ..QueryOptions::default()
        },
        ..Query::default()
    };
    let result = validator.validate(&query).unwrap();
    assert!(result.valid);
    assert!(result.issues.is_empty());
}

#[test]
fn test_query_context_comprehensive() {
    let context = QueryContext {
        user_id: Some("test-user".to_string()),
        workspace_path: "/home/user/project".to_string(),
        recent_files: vec![
            "src/main.rs".to_string(),
            "src/lib.rs".to_string(),
            "tests/integration.rs".to_string(),
        ],
        language_preference: Some("rust".to_string()),
        performance_profile: PerformanceProfile::Balanced,
    };
    
    assert_eq!(context.user_id, Some("test-user".to_string()));
    assert_eq!(context.workspace_path, "/home/user/project");
    assert_eq!(context.recent_files.len(), 3);
    assert_eq!(context.language_preference, Some("rust".to_string()));
    assert_eq!(context.performance_profile, PerformanceProfile::Balanced);
    
    // Test clone
    let cloned = context.clone();
    assert_eq!(cloned.user_id, context.user_id);
    assert_eq!(cloned.workspace_path, context.workspace_path);
}

#[test]
fn test_validation_issue_comprehensive() {
    let issue = ValidationIssue {
        severity: IssueSeverity::Warning,
        message: "Test warning message".to_string(),
        suggestion: Some("Test suggestion".to_string()),
    };
    
    assert_eq!(issue.severity, IssueSeverity::Warning);
    assert_eq!(issue.message, "Test warning message");
    assert_eq!(issue.suggestion, Some("Test suggestion".to_string()));
    
    // Test issue without suggestion
    let issue = ValidationIssue {
        severity: IssueSeverity::Error,
        message: "Test error message".to_string(),
        suggestion: None,
    };
    
    assert_eq!(issue.severity, IssueSeverity::Error);
    assert_eq!(issue.suggestion, None);
}

#[test]
fn test_pattern_type_comprehensive() {
    let types = [
        PatternType::Literal,
        PatternType::Wildcard,
        PatternType::Regex,
        PatternType::Structural,
        PatternType::Semantic,
    ];
    
    // Test all variants are unique
    for (i, type1) in types.iter().enumerate() {
        for (j, type2) in types.iter().enumerate() {
            if i != j {
                assert_ne!(type1, type2);
            } else {
                assert_eq!(type1, type2);
            }
        }
    }
    
    // Test clone and debug
    let pattern = PatternType::Structural;
    let cloned = pattern.clone();
    assert_eq!(pattern, cloned);
    
    let debug_str = format!("{:?}", pattern);
    assert_eq!(debug_str, "Structural");
}

#[test]
fn test_performance_profile_comprehensive() {
    let profiles = [
        PerformanceProfile::Speed,
        PerformanceProfile::Accuracy,
        PerformanceProfile::Balanced,
    ];
    
    // Test all variants are unique
    for (i, profile1) in profiles.iter().enumerate() {
        for (j, profile2) in profiles.iter().enumerate() {
            if i != j {
                assert_ne!(profile1, profile2);
            } else {
                assert_eq!(profile1, profile2);
            }
        }
    }
    
    // Test clone and debug
    let profile = PerformanceProfile::Speed;
    let cloned = profile.clone();
    assert_eq!(profile, cloned);
    
    let debug_str = format!("{:?}", profile);
    assert_eq!(debug_str, "Speed");
}

#[test]
fn test_issue_severity_comprehensive() {
    let severities = [
        IssueSeverity::Error,
        IssueSeverity::Warning,
        IssueSeverity::Info,
    ];
    
    // Test all variants are unique
    for (i, severity1) in severities.iter().enumerate() {
        for (j, severity2) in severities.iter().enumerate() {
            if i != j {
                assert_ne!(severity1, severity2);
            } else {
                assert_eq!(severity1, severity2);
            }
        }
    }
    
    // Test clone and debug
    let severity = IssueSeverity::Warning;
    let cloned = severity.clone();
    assert_eq!(severity, cloned);
    
    let debug_str = format!("{:?}", severity);
    assert_eq!(debug_str, "Warning");
}

#[test]
fn test_processed_query_comprehensive() {
    let query = Query::default();
    let analysis = QueryAnalysis {
        complexity_score: 0.6,
        estimated_results: 42,
        suggested_systems: vec!["lex".to_string(), "symbols".to_string()],
        optimization_hints: vec!["Consider using filters".to_string()],
        language_hints: vec!["rust".to_string()],
        pattern_type: PatternType::Literal,
    };
    let validation = ValidationResult {
        valid: true,
        issues: vec![],
        suggestions: vec!["Good query".to_string()],
    };
    
    let processed = ProcessedQuery {
        query,
        analysis,
        validation,
        processing_time_ms: 125.5,
    };
    
    assert!(processed.query.id.len() > 0); // UUID should be generated
    assert_eq!(processed.analysis.complexity_score, 0.6);
    assert_eq!(processed.analysis.estimated_results, 42);
    assert_eq!(processed.analysis.suggested_systems.len(), 2);
    assert_eq!(processed.validation.valid, true);
    assert_eq!(processed.processing_time_ms, 125.5);
}

#[test]
fn test_query_default_comprehensive() {
    let query = Query::default();
    
    assert!(!query.id.is_empty()); // UUID should be generated
    assert_eq!(query.original_query, "");
    assert_eq!(query.processed_query, "");
    assert_eq!(query.query_type, QueryType::Fuzzy); // Default variant
    assert!(query.metadata.is_empty());
    
    // Test filters are default
    assert!(query.filters.file_extensions.is_empty());
    assert_eq!(query.filters.language_filter, None);
    
    // Test options are default
    assert_eq!(query.options.max_results, 50);
    assert_eq!(query.options.timeout_ms, 2000);
    assert_eq!(query.options.sort_by, SortOrder::Relevance);
}

#[test]
fn test_edge_case_empty_inputs() {
    let processor = QueryProcessor::new();
    let context = QueryContext {
        user_id: None,
        workspace_path: "/test".to_string(),
        recent_files: vec![],
        language_preference: None,
        performance_profile: PerformanceProfile::Balanced,
    };
    
    // Empty string classification
    let result = processor.process_query("", &context).unwrap();
    assert_eq!(result.query.query_type, QueryType::Fuzzy);
    
    // Whitespace-only classification
    let result = processor.process_query("   \t\n   ", &context).unwrap();
    assert_eq!(result.query.query_type, QueryType::Fuzzy);
    
    // Single character
    let result = processor.process_query("a", &context).unwrap();
    assert_eq!(result.query.query_type, QueryType::Symbol);
}

#[test]
fn test_edge_case_special_characters() {
    let processor = QueryProcessor::new();
    
    // Special characters that might cause issues
    let special_queries = [
        "query with spaces",
        "query\twith\ttabs",
        "query\nwith\nnewlines",
        "query with unicode: 🦀 Rust",
        "query with punctuation: hello, world!",
        "query with quotes: 'single' and \"double\"",
    ];
    
    let context = QueryContext {
        user_id: None,
        workspace_path: "/test".to_string(),
        recent_files: vec![],
        language_preference: None,
        performance_profile: PerformanceProfile::Balanced,
    };
    
    for query in &special_queries {
        let result = processor.process_query(query, &context).unwrap();
        let processed = &result.query.processed_query;
        let query_type = result.query.query_type;
        
        // Should not panic and should produce reasonable results
        assert!(!processed.is_empty() || query.trim().is_empty());
        // Query type should be one of the valid variants
        match query_type {
            QueryType::Exact | QueryType::Fuzzy | QueryType::Structural |
            QueryType::Semantic | QueryType::Symbol | QueryType::Hybrid |
            QueryType::Regex => {
                // Valid variant
            }
        }
    }
}

#[test]
#[cfg(feature = "stress-tests")]
fn test_memory_efficiency() {
    // Test that large numbers of queries don't cause memory issues
    let processor = QueryProcessor::new();
    
    let context = QueryContext {
        user_id: None,
        workspace_path: "/test".to_string(),
        recent_files: vec![],
        language_preference: None,
        performance_profile: PerformanceProfile::Balanced,
    };
    
    for i in 0..1000 {
        let query_text = format!("test query {}", i);
        let result = processor.process_query(&query_text, &context).unwrap();
        let _processed = &result.query.processed_query;
        let _query_type = result.query.query_type;
    }
    
    // If we get here without OOM, the test passes
    assert!(true);
}

#[test]
fn test_concurrent_safety() {
    // Test that query processing is thread-safe for read operations
    use std::sync::Arc;
    use std::thread;
    
    let processor = Arc::new(QueryProcessor::new());
    let mut handles = vec![];
    
    for i in 0..10 {
        let processor_clone = Arc::clone(&processor);
        let handle = thread::spawn(move || {
            let query_text = format!("test query {}", i);
            let context = QueryContext {
                user_id: None,
                workspace_path: "/test".to_string(),
                recent_files: vec![],
                language_preference: None,
                performance_profile: PerformanceProfile::Balanced,
            };
            let result = processor_clone.process_query(&query_text, &context).unwrap();
            let _processed = &result.query.processed_query;
            let _query_type = result.query.query_type;
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    // If we get here without deadlocks or panics, the test passes
    assert!(true);
}