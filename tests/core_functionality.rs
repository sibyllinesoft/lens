//! Core functionality tests for Rust migration validation
//! 
//! These tests focus on the essential components that are compiling correctly
//! to establish baseline test coverage for the Rust migration.

use lens_core::search::{SearchConfig, SearchRequest, SearchMethod, SearchResultType};

#[test]
fn test_search_config_default_values() {
    let config = SearchConfig::default();
    
    // Test all default configuration values
    assert_eq!(config.index_path, "./index");
    assert_eq!(config.max_results_default, 50);
    assert_eq!(config.sla_target_ms, 150); // ≤150ms SLA per TODO.md
    assert_eq!(config.lsp_routing_rate, 0.5); // 50% LSP routing
    assert!(config.enable_lsp); // LSP enabled by default
    assert!(config.enable_fusion_pipeline);
    assert!(!config.enable_semantic_search); // Future enhancement
    assert_eq!(config.context_lines, 3);
    
    // Test pinned dataset configuration
    assert_eq!(config.dataset_path, "./pinned-datasets");
    assert!(config.enable_pinned_datasets);
    assert!(config.enable_corpus_validation);
    assert_eq!(config.default_dataset_version, Some("default".to_string()));
}

#[test]
fn test_search_config_custom_creation() {
    let mut config = SearchConfig::default();
    
    // Customize configuration values
    config.index_path = "/custom/index".to_string();
    config.max_results_default = 100;
    config.sla_target_ms = 200;
    config.enable_lsp = false;
    
    // Verify customizations
    assert_eq!(config.index_path, "/custom/index");
    assert_eq!(config.max_results_default, 100);
    assert_eq!(config.sla_target_ms, 200);
    assert!(!config.enable_lsp);
}

#[test]
fn test_search_request_default_values() {
    let request = SearchRequest::default();
    
    // Test default request values
    assert_eq!(request.query, "");
    assert!(request.file_path.is_none());
    assert!(request.language.is_none());
    assert_eq!(request.max_results, 50);
    assert!(request.include_context);
    assert_eq!(request.timeout_ms, 150); // ≤150ms SLA
    assert!(request.enable_lsp);
    
    // Test default search types
    let expected_types = vec![
        SearchResultType::TextMatch,
        SearchResultType::Definition,
        SearchResultType::Reference,
        SearchResultType::Symbol,
    ];
    assert_eq!(request.search_types, expected_types);
    
    // Test default search method
    assert_eq!(request.search_method, Some(SearchMethod::Hybrid));
}

#[test]
fn test_search_request_custom_creation() {
    let request = SearchRequest {
        query: "test query".to_string(),
        file_path: Some("/path/to/file.rs".to_string()),
        language: Some("rust".to_string()),
        max_results: 25,
        include_context: false,
        timeout_ms: 100,
        enable_lsp: false,
        search_types: vec![SearchResultType::Definition],
        search_method: Some(SearchMethod::Semantic),
    };
    
    // Verify custom values
    assert_eq!(request.query, "test query");
    assert_eq!(request.file_path, Some("/path/to/file.rs".to_string()));
    assert_eq!(request.language, Some("rust".to_string()));
    assert_eq!(request.max_results, 25);
    assert!(!request.include_context);
    assert_eq!(request.timeout_ms, 100);
    assert!(!request.enable_lsp);
    assert_eq!(request.search_types, vec![SearchResultType::Definition]);
    assert_eq!(request.search_method, Some(SearchMethod::Semantic));
}

#[test]
fn test_search_method_enum() {
    // Test SearchMethod enum values
    assert_eq!(SearchMethod::default(), SearchMethod::Hybrid);
    
    // Test all enum variants exist
    let methods = vec![
        SearchMethod::Lexical,
        SearchMethod::Structural,
        SearchMethod::Semantic,
        SearchMethod::Hybrid,
        SearchMethod::ForceSemantic,
    ];
    
    // Verify we can create all methods
    assert_eq!(methods.len(), 5);
}

#[test]
fn test_search_result_type_enum() {
    // Test SearchResultType enum variants
    let result_types = vec![
        SearchResultType::TextMatch,
        SearchResultType::Definition,
        SearchResultType::Reference,
        SearchResultType::TypeInfo,
        SearchResultType::Implementation,
        SearchResultType::Symbol,
        SearchResultType::Semantic,
    ];
    
    // Verify all variants can be created
    assert_eq!(result_types.len(), 7);
    
    // Test PartialEq and Hash traits work
    assert_eq!(SearchResultType::Definition, SearchResultType::Definition);
    assert_ne!(SearchResultType::Definition, SearchResultType::Reference);
}

#[test]
fn test_search_config_sla_compliance() {
    let config = SearchConfig::default();
    
    // Verify SLA target meets TODO.md requirement of ≤150ms
    assert!(config.sla_target_ms <= 150, "SLA target must be ≤150ms per TODO.md");
    
    // Test that default request timeout matches SLA
    let request = SearchRequest::default();
    assert_eq!(request.timeout_ms, config.sla_target_ms);
}

#[test]
fn test_lsp_configuration_consistency() {
    let config = SearchConfig::default();
    let request = SearchRequest::default();
    
    // Both should have LSP enabled by default
    assert!(config.enable_lsp);
    assert!(request.enable_lsp);
    
    // LSP routing rate should be reasonable (0.0-1.0)
    assert!(config.lsp_routing_rate >= 0.0);
    assert!(config.lsp_routing_rate <= 1.0);
}

#[test]
fn test_search_result_types_completeness() {
    // Verify we have all necessary result types for LSP integration
    let lsp_types = vec![
        SearchResultType::Definition,
        SearchResultType::Reference,
        SearchResultType::TypeInfo,
        SearchResultType::Implementation,
        SearchResultType::Symbol,
    ];
    
    // All LSP types should be distinct
    let mut unique_types = lsp_types.clone();
    unique_types.sort();
    unique_types.dedup();
    assert_eq!(unique_types.len(), lsp_types.len());
}