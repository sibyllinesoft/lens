//! Focused tests for query.rs module - public API only
//! 
//! These tests focus on the public API and components that can be tested
//! without accessing private fields or methods.

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
fn test_query_type_variants() {
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
fn test_query_type_serialization() {
    use serde_json;
    
    let query_type = QueryType::Structural;
    let serialized = serde_json::to_string(&query_type).unwrap();
    assert_eq!(serialized, "\"structural\"");
    
    let deserialized: QueryType = serde_json::from_str(&serialized).unwrap();
    assert_eq!(deserialized, QueryType::Structural);
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
fn test_file_type_variants() {
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
fn test_sort_order_variants() {
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
fn test_basic_query_analyzer() {
    let analyzer = BasicQueryAnalyzer::new();
    
    assert_eq!(analyzer.name(), "basic_analyzer");
    
    let result = analyzer.analyze("test query").unwrap();
    assert!(result.complexity_score >= 0.0);
    assert!(result.complexity_score <= 1.0);
    assert!(result.estimated_results > 0);
    assert!(!result.suggested_systems.is_empty());
}

#[test]
fn test_basic_query_analyzer_complex_query() {
    let analyzer = BasicQueryAnalyzer::new();
    
    // Use a query > 100 characters to trigger complex classification
    let complex_query = "a".repeat(150);
    let result = analyzer.analyze(&complex_query).unwrap();
    assert_eq!(result.complexity_score, 0.8);
}

#[test]
fn test_pattern_analyzer() {
    let analyzer = PatternAnalyzer::new();
    
    assert_eq!(analyzer.name(), "pattern_analyzer");
    
    let result = analyzer.analyze("class.*implements").unwrap();
    assert_eq!(result.pattern_type, PatternType::Wildcard);
    
    let result = analyzer.analyze("simple text").unwrap();
    assert_eq!(result.pattern_type, PatternType::Literal);
}

#[test]
fn test_performance_optimizer() {
    let optimizer = PerformanceOptimizer::new();
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
    
    // Should reduce timeout and max_results for speed
    assert!(optimized.options.timeout_ms <= 1000);
    assert!(optimized.options.max_results <= 20);
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
    
    // Test semantic query
    let query = Query {
        query_type: QueryType::Semantic,
        ..Query::default()
    };
    let optimized = optimizer.optimize(&query, &context).unwrap();
    assert_eq!(optimized.options.systems, vec!["semantic", "lex"]);
}

#[test]
fn test_basic_query_validator() {
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
fn test_query_context_creation() {
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
}

#[test]
fn test_validation_issue_creation() {
    let issue = ValidationIssue {
        severity: IssueSeverity::Warning,
        message: "Test warning message".to_string(),
        suggestion: Some("Test suggestion".to_string()),
    };
    
    assert_eq!(issue.severity, IssueSeverity::Warning);
    assert_eq!(issue.message, "Test warning message");
    assert_eq!(issue.suggestion, Some("Test suggestion".to_string()));
}

#[test]
fn test_pattern_type_variants() {
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
}

#[test]
fn test_performance_profile_variants() {
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
}

#[test]
fn test_issue_severity_variants() {
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
}

#[test]
fn test_processed_query_structure() {
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
fn test_query_default() {
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