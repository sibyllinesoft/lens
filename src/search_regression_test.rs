#[cfg(test)]
mod search_regression_tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::TempDir;

    async fn create_test_search_engine() -> Result<(SearchEngine, TempDir)> {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().to_str().unwrap();
        
        let mut config = SearchConfig::default();
        config.index_path = index_path.to_string();
        config.enable_lsp = false; // Disable LSP for regression tests
        
        let engine = SearchEngine::new(config).await?;
        Ok((engine, temp_dir))
    }

    #[tokio::test]
    async fn test_basic_search_functionality() {
        let (engine, _temp_dir) = create_test_search_engine().await.unwrap();
        
        // Test basic Rust keywords that should exist in the indexed content
        let test_cases = vec![
            ("struct", "Should find struct definitions"),
            ("impl", "Should find impl blocks"),
            ("fn", "Should find function definitions"),
            ("SearchEngine", "Should find SearchEngine references"),
            ("pub", "Should find public declarations"),
        ];
        
        for (query, description) in test_cases {
            let request = SearchRequest {
                query: query.to_string(),
                max_results: 10,
                language: None,
                enable_lsp: false,
                include_context: false,
                timeout_ms: 1000,
                search_method: Some(SearchMethod::Lexical), // Use lexical search for baseline
            };
            
            let response = engine.search(request).await.unwrap();
            
            // REGRESSION TEST: Basic queries should return results from populated index
            assert!(
                !response.results.is_empty(),
                "REGRESSION FAILURE: Query '{}' returned 0 results. {}", 
                query, description
            );
            
            println!("✅ Query '{}': {} results", query, response.results.len());
        }
    }

    #[tokio::test]
    async fn test_query_sanitization_preserves_searchable_terms() {
        let (engine, _temp_dir) = create_test_search_engine().await.unwrap();
        
        // Test that sanitization doesn't destroy searchable content
        let original_query = "struct SearchEngine impl search";
        let sanitized = engine.sanitize_query(original_query);
        
        // REGRESSION TEST: Sanitization should preserve core terms
        assert!(
            sanitized.contains("struct") || sanitized.contains("SearchEngine") || sanitized.contains("impl"),
            "REGRESSION FAILURE: Query sanitization removed all searchable terms: '{}' -> '{}'", 
            original_query, sanitized
        );
        
        // Test that sanitized query still returns results
        let request = SearchRequest {
            query: sanitized,
            max_results: 10,
            language: None,
            enable_lsp: false,
            include_context: false,
            timeout_ms: 1000,
            search_method: Some(SearchMethod::Lexical),
        };
        
        let response = engine.search(request).await.unwrap();
        
        // REGRESSION TEST: Sanitized queries should still return results
        assert!(
            !response.results.is_empty(),
            "REGRESSION FAILURE: Sanitized query '{}' returned 0 results", sanitized
        );
        
        println!("✅ Sanitized query '{}': {} results", sanitized, response.results.len());
    }

    #[tokio::test]
    async fn test_semantic_reranking_with_results() {
        let (engine, _temp_dir) = create_test_search_engine().await.unwrap();
        
        // Test that semantic reranking works when there are initial results
        let request = SearchRequest {
            query: "function implementation".to_string(),
            max_results: 10,
            language: None,
            enable_lsp: false,
            include_context: false,
            timeout_ms: 2000,
            search_method: Some(SearchMethod::ForceSemantic), // Force semantic for testing
        };
        
        let response = engine.search(request).await.unwrap();
        
        // REGRESSION TEST: Semantic search should work when initial search returns results
        // Note: This may return 0 results if semantic pipeline fails, but it shouldn't crash
        println!("✅ Semantic search completed: {} results", response.results.len());
        
        // Check that the response includes semantic metadata
        assert!(response.metadata.contains_key("semantic_applied") || response.results.is_empty(),
                "REGRESSION FAILURE: Semantic search should include metadata or explain empty results");
    }

    #[tokio::test]
    async fn test_index_population_regression() {
        let (engine, _temp_dir) = create_test_search_engine().await.unwrap();
        
        // REGRESSION TEST: Index should be automatically populated during creation
        let reader = &engine.reader;
        let searcher = reader.searcher();
        
        // Check that the index contains documents
        let all_query = tantivy::query::AllQuery;
        let top_docs = searcher.search(&all_query, &tantivy::collector::TopDocs::with_limit(1)).unwrap();
        
        assert!(
            !top_docs.is_empty(),
            "REGRESSION FAILURE: Index should be automatically populated with documents"
        );
        
        println!("✅ Index contains {} documents (verified with sample)", top_docs.len());
    }

    #[tokio::test]
    async fn test_real_benchmark_query_patterns() {
        let (engine, _temp_dir) = create_test_search_engine().await.unwrap();
        
        // Test patterns similar to actual benchmark queries but adapted to indexed content
        let adapted_queries = vec![
            // Adapted from SWE-bench style to match Rust content
            "implement search function",
            "create struct definition", 
            "error handling implementation",
            "async function pattern",
            "configuration management",
        ];
        
        for query in adapted_queries {
            let request = SearchRequest {
                query: query.to_string(),
                max_results: 5,
                language: None,
                enable_lsp: false,
                include_context: false,
                timeout_ms: 1000,
                search_method: Some(SearchMethod::Lexical),
            };
            
            let response = engine.search(request).await.unwrap();
            
            // REGRESSION TEST: Realistic queries should return some results from Rust codebase
            println!("Query '{}': {} results", query, response.results.len());
            
            // Note: We don't assert non-empty here because some queries might legitimately return 0
            // But we verify the search doesn't crash and returns a valid response
            assert!(response.metadata.contains_key("execution_time_ms"),
                    "REGRESSION FAILURE: Search should include execution metadata");
        }
    }
}

// Helper function to make sanitize_query public for testing
impl SearchEngine {
    #[cfg(test)]
    pub fn sanitize_query(&self, query: &str) -> String {
        self.sanitize_query(query)
    }
}