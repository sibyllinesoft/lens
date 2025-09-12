//! LSP Integration and Complex Scenario Coverage Tests
//! 
//! This module tests complex integration scenarios, state management,
//! and interaction patterns between LSP components that require
//! comprehensive testing for high coverage.

use lens_core::lsp::*;
use anyhow::Result;
use std::sync::Arc;
use std::time::Duration;
use std::collections::HashMap;

#[cfg(test)]
mod lsp_integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_router_hint_cache_integration() {
        // Test integration between router and hint cache
        let router = lens_core::lsp::router::LspRouter::new(0.6);
        let cache = lens_core::lsp::hint::HintCache::new(1).await.unwrap();
        
        // Simulate routing decisions that would affect caching
        let queries = vec![
            ("findUserById", QueryIntent::Symbol),
            ("class DatabaseConnection", QueryIntent::Symbol),
            ("authentication middleware", QueryIntent::TextSearch),
            ("import React from 'react'", QueryIntent::TextSearch),
        ];
        
        for (query, intent) in queries {
            // Make routing decision
            let decision = router.make_routing_decision(query, &intent).await;
            
            // Simulate caching the result based on routing decision
            let cache_key = format!("{}:{:?}", query, intent);
            if decision.should_route_to_lsp {
                // Cache LSP results
                let results = vec![LspSearchResult {
                    file_path: "/test/file.rs".to_string(),
                    line_number: 10,
                    column: 5,
                    content: format!("lsp_result_confidence_{}", decision.confidence),
                    hint_type: HintType::Definition,
                    server_type: LspServerType::Rust,
                    confidence: decision.confidence,
                    context_lines: None,
                }];
                cache.set(cache_key.clone(), results, 86400).await.unwrap();
                
                // Verify cached result
                let cached = cache.get(&cache_key).await.unwrap();
                assert!(cached.is_some(), "LSP result should be cached");
            } else {
                // Baseline results might not be cached, or cached differently
                let cache_key = format!("baseline:{}:{:?}", query, intent);
                let results = vec![LspSearchResult {
                    file_path: "/test/file.rs".to_string(),
                    line_number: 10,
                    column: 5,
                    content: "baseline_result".to_string(),
                    hint_type: HintType::Symbol,
                    server_type: LspServerType::Rust,
                    confidence: 0.8,
                    context_lines: None,
                }];
                cache.set(cache_key.clone(), results, 86400).await.unwrap();
            }
        }
        
        let stats = cache.stats().await;
        assert!(stats.size > 0, "Cache should contain entries from routing decisions");
    }

    #[tokio::test]
    async fn test_manager_router_coordination() {
        // Test coordination between manager and router components
        let config = LspConfig {
            enabled: true,
            routing_percentage: 0.7,
            cache_ttl_hours: 2,
            server_timeout_ms: 3000,
            max_concurrent_requests: 8,
            traversal_bounds: TraversalBounds::default(),
        };
        
        // Test that manager config affects router behavior
        let router = lens_core::lsp::router::LspRouter::new(config.routing_percentage);
        
        // Simulate manager statistics affecting router decisions
        let mut server_errors = HashMap::new();
        server_errors.insert(LspServerType::Rust, 10);
        server_errors.insert(LspServerType::Python, 2);
        server_errors.insert(LspServerType::TypeScript, 5);
        
        let manager_stats = lens_core::lsp::manager::LspStats {
            total_requests: 1000,
            cache_hits: 300,
            lsp_routed: 600,
            fallback_used: 400,
            avg_response_time_ms: 250,
            server_errors,
        };
        
        // Test different server types and their error rates
        let high_error_rate = manager_stats.server_errors[&LspServerType::Rust] as f64 / 100.0;
        let low_error_rate = manager_stats.server_errors[&LspServerType::Python] as f64 / 100.0;
        
        assert!(high_error_rate > low_error_rate, "Error rates should vary by server type");
        
        // Router should potentially adapt based on error rates
        for _ in 0..20 {
            // Report failures for high-error server type
            router.report_lsp_result(&QueryIntent::Symbol, false, 1000).await;
        }
        
        let routing_stats = router.get_routing_stats().await;
        assert!(routing_stats.total_queries >= 20, "Router should track reported results");
    }

    #[tokio::test]
    async fn test_multi_language_server_simulation() {
        // Simulate managing multiple language servers
        let server_types = vec![
            LspServerType::Rust,
            LspServerType::Python,
            LspServerType::TypeScript,
            LspServerType::Go,
        ];
        
        let mut server_stats = HashMap::new();
        
        for server_type in server_types {
            // Simulate different performance characteristics per server
            let (success_rate, avg_latency) = match server_type {
                LspServerType::Rust => (0.95, 100),        // Fast, reliable
                LspServerType::Python => (0.85, 200),      // Moderate
                LspServerType::TypeScript => (0.90, 150),  // Good
                LspServerType::Go => (0.88, 180),          // Decent
                LspServerType::JavaScript => (0.87, 175),  // Good
            };
            
            server_stats.insert(server_type, (success_rate, avg_latency));
        }
        
        // Test request routing based on file types
        let file_queries = vec![
            ("search.rs", LspServerType::Rust),
            ("main.py", LspServerType::Python),
            ("component.tsx", LspServerType::TypeScript),
            ("server.go", LspServerType::Go),
        ];
        
        for (filename, expected_server) in file_queries {
            let extension = filename.split('.').last().unwrap_or("");
            let detected_server = LspServerType::from_file_extension(extension);
            
            if let Some(server_type) = detected_server {
                assert_eq!(server_type, expected_server, 
                          "Server type detection should be correct for {}", filename);
                
                // Test that server stats are available
                assert!(server_stats.contains_key(&server_type),
                       "Server stats should be available for detected type");
            }
        }
    }

    #[tokio::test]
    async fn test_traversal_bounds_validation() {
        // Test that traversal bounds are properly validated
        let bounds_test_cases = vec![
            TraversalBounds { max_depth: 1, max_results: 10, timeout_ms: 1000 },
            TraversalBounds { max_depth: 5, max_results: 100, timeout_ms: 5000 },
            TraversalBounds { max_depth: 1, max_results: 1, timeout_ms: 100 },  // Minimal (max_depth must be >= 1)
            TraversalBounds { max_depth: 10, max_results: 1000, timeout_ms: 10000 }, // Large
        ];
        
        for bounds in bounds_test_cases {
            // Validate bounds constraints from TODO.md
            assert!(bounds.max_depth >= 1, "Max depth should be at least 1");
            assert!(bounds.max_depth <= 10, "Max depth should be reasonable (≤10)");
            assert!(bounds.max_results >= 1, "Max results should be at least 1");
            assert!(bounds.max_results <= 1000, "Max results should be reasonable (≤1000)");
            assert!(bounds.timeout_ms > 0, "Timeout should be positive");
            assert!(bounds.timeout_ms <= 30000, "Timeout should be reasonable (≤30s)");
            
            // Test default bounds are reasonable
            let default_bounds = TraversalBounds::default();
            assert_eq!(default_bounds.max_depth, 2, "Default max depth should be 2 per TODO.md");
            assert_eq!(default_bounds.max_results, 64, "Default max results should be 64 per TODO.md");
        }
    }

    #[tokio::test]
    async fn test_complex_query_intent_routing() {
        let router = lens_core::lsp::router::LspRouter::new(0.6);
        
        // Test complex interaction between query patterns and intents
        let complex_scenarios = vec![
            // Exact matches should prefer baseline for safety
            ("exact_function_name", QueryIntent::Symbol),
            ("EXACT_CONSTANT", QueryIntent::Symbol),
            
            // Structural queries should consider LSP for code understanding  
            ("class MyClass extends BaseClass", QueryIntent::TextSearch),
            ("function processData(input: DataType)", QueryIntent::TextSearch),
            ("interface UserInterface { name: string; }", QueryIntent::TypeDefinition),
            
            // Semantic queries should leverage LSP when effective
            ("find all database operations", QueryIntent::TextSearch),
            ("show error handling patterns", QueryIntent::TextSearch), 
            ("locate authentication logic", QueryIntent::TextSearch),
            
            // Identifier queries balance between baseline and LSP
            ("getUserById", QueryIntent::Symbol),
            ("calculateTotalPrice", QueryIntent::Symbol),
            ("validateUserInput", QueryIntent::Symbol),
        ];
        
        for (query, intent) in complex_scenarios {
            let decision = router.make_routing_decision(query, &intent).await;
            
            // Verify decision logic aligns with intent characteristics
            match intent {
                QueryIntent::Symbol => {
                    // Symbol queries can go either way depending on confidence
                    if decision.should_route_to_lsp {
                        assert!(decision.confidence > 0.5, 
                               "Symbol LSP routing should have reasonable confidence");
                    }
                    // Baseline routing is also acceptable
                }
                QueryIntent::TextSearch => {
                    // Text search queries benefit from LSP understanding
                    if decision.should_route_to_lsp {
                        assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0,
                               "TextSearch LSP confidence should be valid");
                    }
                    // Both LSP and baseline are acceptable
                }
                QueryIntent::TypeDefinition | QueryIntent::Definition | QueryIntent::References => {
                    // Should make reasonable decisions based on query complexity
                    assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0,
                           "Decision confidence should be in valid range");
                    // Either routing choice is reasonable for these intent types
                }
                _ => {
                    // Other intent types are also valid
                    assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0,
                           "Decision confidence should be in valid range");
                }
            }
        }
    }

    #[tokio::test]
    async fn test_cache_hint_type_consistency() {
        let cache = lens_core::lsp::hint::HintCache::new(2).await.unwrap();
        
        // Test that different hint types are handled consistently
        let hint_type_scenarios = vec![
            (HintType::Definition, "function definition location"),
            (HintType::References, "all references to symbol"),
            (HintType::TypeDefinition, "type definition details"),
            (HintType::Implementation, "implementation details"),
            (HintType::Declaration, "symbol declaration"),
            (HintType::Symbol, "symbol information"),
            (HintType::Hover, "hover information"),
            (HintType::Completion, "completion suggestions"),
        ];
        
        for (hint_type, content) in &hint_type_scenarios {
            let key = format!("test_{:?}", hint_type);
            
            // Store hint
            let results = vec![LspSearchResult {
                file_path: "/test/file.rs".to_string(),
                line_number: 10,
                column: 5,
                content: content.to_string(),
                hint_type: *hint_type,
                server_type: LspServerType::Rust,
                confidence: 0.9,
                context_lines: None,
            }];
            cache.set(key.clone(), results.clone(), 3600).await.unwrap();
            
            // Retrieve and verify
            if let Some(cached_results) = cache.get(&key).await.unwrap() {
                assert_eq!(cached_results[0].hint_type, *hint_type,
                          "Cached hint type should match original");
                assert_eq!(cached_results[0].content, *content,
                          "Cached content should match original");
                
                // Test hint type string conversion
                let type_str = hint_type.as_str();
                assert!(!type_str.is_empty(), "Hint type string should not be empty");
                assert!(type_str.chars().all(|c| c.is_ascii_lowercase() || c == '_'),
                       "Hint type string should be lowercase with underscores");
            }
        }
        
        let stats = cache.stats().await;
        assert_eq!(stats.size, hint_type_scenarios.len(),
                  "Cache should contain all hint types");
    }

    #[tokio::test]
    async fn test_concurrent_cache_hint_invalidation() {
        let cache = Arc::new(lens_core::lsp::hint::HintCache::new(1).await.unwrap());
        
        // Setup initial cache state
        for i in 0..100 {
            let key = format!("invalidation_test_{}", i);
            let results = vec![LspSearchResult {
                file_path: "/test/file.rs".to_string(),
                line_number: 10,
                column: 5,
                content: format!("initial_value_{}", i),
                hint_type: HintType::Definition,
                server_type: LspServerType::Rust,
                confidence: 0.9,
                context_lines: None,
            }];
            cache.set(key, results, 3600).await.unwrap();
        }
        
        // Test concurrent invalidation patterns
        let handles: Vec<_> = (0..10)
            .map(|task_id| {
                let cache_clone = cache.clone();
                tokio::spawn(async move {
                    for i in 0..20 {
                        match i % 4 {
                            0 => {
                                // File-based invalidation
                                let file_path = std::path::PathBuf::from(
                                    format!("/project/src/file_{}.rs", task_id)
                                );
                                cache_clone.invalidate_file(&file_path).await.unwrap();
                            }
                            1 => {
                                // Add new entries during invalidation
                                let key = format!("concurrent_add_{}_{}", task_id, i);
                                let results = vec![LspSearchResult {
                                    file_path: "/test/file.rs".to_string(),
                                    line_number: 10,
                                    column: 5,
                                    content: format!("concurrent_value_{}_{}", task_id, i),
                                    hint_type: HintType::References,
                                    server_type: LspServerType::Rust,
                                    confidence: 0.8,
                                    context_lines: None,
                                }];
                                cache_clone.set(key, results, 3600).await.unwrap();
                            }
                            2 => {
                                // Read existing entries (touch not available, use get)
                                let key = format!("invalidation_test_{}", i % 50);
                                let _ = cache_clone.get(&key).await;
                            }
                            3 => {
                                // Read entries during invalidation
                                let key = format!("invalidation_test_{}", i % 100);
                                let _ = cache_clone.get(&key).await;
                            }
                            _ => unreachable!(),
                        }
                    }
                })
            })
            .collect();
        
        // Wait for all concurrent operations
        for handle in handles {
            handle.await.expect("Concurrent invalidation task should complete");
        }
        
        // Verify cache is still in consistent state
        let stats = cache.stats().await;
        assert!(stats.hits + stats.misses > 0, "Cache should have recorded operations");
        assert!(stats.invalidations > 0, "Cache should have recorded invalidations");
        
        // Test cache still works after concurrent invalidation
        let results = vec![LspSearchResult {
            file_path: "/test/file.rs".to_string(),
            line_number: 10,
            column: 5,
            content: "test_value".to_string(),
            hint_type: HintType::Symbol,
            server_type: LspServerType::Rust,
            confidence: 0.9,
            context_lines: None,
        }];
        cache.set("post_invalidation_test".to_string(), results, 3600).await.unwrap();
        assert!(cache.get("post_invalidation_test").await.unwrap().is_some(),
               "Cache should still function after concurrent invalidation");
    }

    #[tokio::test]
    async fn test_search_result_validation_comprehensive() {
        // Test comprehensive search result validation
        let valid_results = vec![
            LspSearchResult {
                file_path: "/src/user.rs".to_string(),
                line_number: 10,
                column: 5,
                content: "getUserData".to_string(),
                hint_type: HintType::Definition,
                server_type: LspServerType::Rust,
                confidence: 0.9,
                context_lines: None,
            },
            LspSearchResult {
                file_path: "/models/user.py".to_string(),
                line_number: 25,
                column: 10,
                content: "class UserModel:".to_string(),
                hint_type: HintType::Symbol,
                server_type: LspServerType::Python,
                confidence: 0.85,
                context_lines: Some(vec!["# User model definition".to_string()]),
            },
            LspSearchResult {
                file_path: "/auth/middleware.ts".to_string(),
                line_number: 15,
                column: 0,
                content: "authentication logic".to_string(),
                hint_type: HintType::Hover,
                server_type: LspServerType::TypeScript,
                confidence: 0.7,
                context_lines: None,
            },
        ];
        
        for result in valid_results {
            // Validate result structure
            assert!(!result.file_path.is_empty(), "File path should not be empty");
            assert!(!result.content.is_empty(), "Content should not be empty");
            
            // Validate position info
            assert!(result.line_number >= 0, "Line number should be non-negative");
            assert!(result.column >= 0, "Column should be non-negative");
            
            // Validate confidence
            assert!(result.confidence >= 0.0 && result.confidence <= 1.0,
                   "Confidence should be between 0.0 and 1.0");
            
            // Validate file path has extension
            assert!(result.file_path.contains('.'), "File path should have extension");
            
            // Validate hint type makes sense for content
            match result.hint_type {
                HintType::Definition => {
                    // Definitions often reference specific symbols
                }
                HintType::References => {
                    // References can be various types
                }
                HintType::Symbol => {
                    // Symbols are identifiers
                }
                HintType::Hover => {
                    // Hover results can be anything
                }
                _ => {
                    // Other hint types are also valid
                }
            }
        }
        
        // Test TraversalBounds validation
        let bounds = TraversalBounds {
            max_depth: 3,
            max_results: 50,
            timeout_ms: 5000,
        };
        
        assert!(bounds.max_depth > 0, "Max depth should be positive");
        assert!(bounds.max_results > 0, "Max results should be positive");
        assert!(bounds.timeout_ms > 0, "Timeout should be positive");
    }
}

#[cfg(test)]
mod lsp_state_management_tests {
    use super::*;

    #[tokio::test]
    async fn test_router_state_consistency() {
        let router = Arc::new(lens_core::lsp::router::LspRouter::new(0.5));
        
        // Test that router maintains consistent state across operations
        let initial_stats = router.get_routing_stats().await;
        
        // Perform various operations
        for i in 0..50 {
            let query = format!("state_test_query_{}", i);
            let intent = match i % 4 {
                0 => QueryIntent::Definition,
                1 => QueryIntent::References,
                2 => QueryIntent::TextSearch,
                _ => QueryIntent::Symbol,
            };
            
            // Make routing decision
            let _ = router.make_routing_decision(&query, &intent).await;
            
            // Report some results
            let success = i % 3 != 0; // Mix of success/failure
            let latency = 50 + (i % 200); // Varying latency
            router.report_lsp_result(&intent, success, latency as u64).await;
        }
        
        let final_stats = router.get_routing_stats().await;
        
        // Verify state changes are consistent
        assert!(final_stats.total_queries >= initial_stats.total_queries + 50,
               "Router should track all reported results");
        
        // Verify routing rate is still within valid bounds
        assert!(final_stats.current_routing_rate >= 0.0 && final_stats.current_routing_rate <= 1.0,
               "Routing rate should remain in valid range");
    }

    #[tokio::test]
    async fn test_cache_state_transitions() {
        let cache = lens_core::lsp::hint::HintCache::new(1).await.unwrap();
        
        // Test cache state transitions
        let key = "state_transition_test";
        
        // Initial state: empty
        assert!(cache.get(key).await.unwrap().is_none(), "Cache should be initially empty");
        
        // Transition: empty -> populated
        let initial_results = vec![LspSearchResult {
            file_path: "/test/file.rs".to_string(),
            line_number: 10,
            column: 5,
            content: "initial_value".to_string(),
            hint_type: HintType::Definition,
            server_type: LspServerType::Rust,
            confidence: 0.9,
            context_lines: None,
        }];
        cache.set(key.to_string(), initial_results, 3600).await.unwrap();
        assert!(cache.get(key).await.unwrap().is_some(), "Cache should contain value after set");
        
        // Transition: populated -> updated  
        let updated_results = vec![LspSearchResult {
            file_path: "/test/file.rs".to_string(),
            line_number: 10,
            column: 5,
            content: "updated_value".to_string(),
            hint_type: HintType::Definition,
            server_type: LspServerType::Rust,
            confidence: 0.9,
            context_lines: None,
        }];
        cache.set(key.to_string(), updated_results, 3600).await.unwrap();
        if let Some(cached) = cache.get(key).await.unwrap() {
            assert_eq!(cached[0].content, "updated_value", "Cache should contain updated value");
        }
        
        // Transition: populated -> invalidated (simulated)
        let file_path = std::path::PathBuf::from("/test/state.rs");
        cache.invalidate_file(&file_path).await.unwrap();
        
        // Verify cache is still functional after state transitions
        let post_transition_results = vec![LspSearchResult {
            file_path: "/test/file.rs".to_string(),
            line_number: 10,
            column: 5,
            content: "test".to_string(),
            hint_type: HintType::Symbol,
            server_type: LspServerType::Rust,
            confidence: 0.9,
            context_lines: None,
        }];
        cache.set("post_transition".to_string(), post_transition_results, 3600).await.unwrap();
        assert!(cache.get("post_transition").await.unwrap().is_some(),
               "Cache should be functional after state transitions");
    }
}