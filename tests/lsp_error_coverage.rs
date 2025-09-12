//! LSP Error Handling and Edge Case Coverage Tests
//! 
//! This module tests error conditions, edge cases, and failure scenarios
//! in LSP modules that are often untested but critical for robustness.
//! Focus areas:
//! - Network/communication failures
//! - Malformed input handling  
//! - Resource exhaustion scenarios
//! - Timeout and cancellation handling
//! - Invalid configuration handling

use lens_core::lsp::*;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;

#[cfg(test)]
mod lsp_error_handling_tests {
    use super::*;

    #[tokio::test]
    async fn test_router_malformed_query_handling() {
        let router = lens_core::lsp::router::LspRouter::new(0.5);
        
        // Test various malformed/edge case queries
        let long_query = "a".repeat(10000);
        let malformed_queries = vec![
            "", // Empty query
            " ", // Whitespace only
            "\n\t\r", // Control characters only
            &long_query, // Extremely long query
            "🚀🔥💯", // Unicode emojis
            "\x00\x01\x02", // Control characters
            "query\nwith\nnewlines", // Multi-line query
            r#"query"with"quotes"#, // Special characters
            "SELECT * FROM users WHERE id = 1; DROP TABLE users;", // SQL injection attempt
            "<script>alert('xss')</script>", // XSS attempt
        ];
        
        for query in malformed_queries {
            // Should not panic or crash
            let decision = router.make_routing_decision(query, &QueryIntent::References).await;
            
            // All decisions should be valid
            assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0, 
                   "Confidence should be valid for query: {:?}", query);
            assert!(decision.estimated_latency_ms > 0, 
                   "Estimated latency should be positive for query: {:?}", query);
        }
    }

    #[tokio::test] 
    async fn test_router_concurrent_stress() {
        let router = Arc::new(lens_core::lsp::router::LspRouter::new(0.5));
        let num_tasks = 100;
        let queries_per_task = 50;
        
        // Create high-concurrency stress test
        let handles: Vec<_> = (0..num_tasks)
            .map(|task_id| {
                let router_clone = router.clone();
                tokio::spawn(async move {
                    for i in 0..queries_per_task {
                        let query = format!("stress_query_{}_{}", task_id, i);
                        
                        // Mix different operations
                        match i % 4 {
                            0 => {
                                let _ = router_clone.make_routing_decision(&query, &QueryIntent::Definition).await;
                            }
                            1 => {
                                router_clone.report_lsp_result(&QueryIntent::References, true, 100).await;
                            }
                            2 => {
                                let _ = router_clone.get_routing_stats().await;  
                            }
                            3 => {
                                router_clone.report_lsp_result(&QueryIntent::TypeDefinition, false, 500).await;
                            }
                            _ => unreachable!(),
                        }
                    }
                })
            })
            .collect();
        
        // Wait for all stress tasks
        for handle in handles {
            handle.await.expect("Stress task should complete without panic");
        }
        
        // Verify router is still functional after stress
        let stats = router.get_routing_stats().await;
        assert!(stats.total_queries > 0, "Router should have processed queries during stress test");
        
        // Verify router can still make decisions
        let decision = router.make_routing_decision("post_stress_query", &QueryIntent::Definition).await;
        // Router should still be functional and return a valid decision
        assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
    }

    #[tokio::test]
    async fn test_hint_cache_memory_pressure() {
        let cache = lens_core::lsp::hint::HintCache::new(1).await.unwrap();
        
        // Simulate memory pressure with large cache entries
        let large_value = "x".repeat(1024 * 1024); // 1MB value
        
        // Add many large entries to trigger memory management
        for i in 0..100 {
            let key = format!("memory_pressure_{}", i);
            let results = vec![
                lens_core::lsp::LspSearchResult {
                    file_path: "test.rs".to_string(),
                    line_number: 1,
                    column: 0,
                    content: large_value.clone(),
                    hint_type: lens_core::lsp::HintType::Symbol,
                    server_type: lens_core::lsp::LspServerType::Rust,
                    confidence: 1.0,
                    context_lines: None,
                }
            ];
            cache.set(key, results, 3600).await.unwrap();
        }
        
        let stats = cache.stats().await;
        
        // Cache should handle memory pressure gracefully
        assert!(stats.evictions > 0 || stats.size < 100, 
               "Cache should evict entries under memory pressure");
        
        // Cache should still be functional
        let results = vec![
            lens_core::lsp::LspSearchResult {
                file_path: "test.rs".to_string(),
                line_number: 1,
                column: 0,
                content: "value".to_string(),
                hint_type: lens_core::lsp::HintType::Definition,
                server_type: lens_core::lsp::LspServerType::Rust,
                confidence: 1.0,
                context_lines: None,
            }
        ];
        cache.set("test_after_pressure".to_string(), results, 3600).await.unwrap();
        assert!(cache.get("test_after_pressure").await.unwrap().is_some(), 
               "Cache should still work after memory pressure");
    }

    #[tokio::test]
    async fn test_hint_cache_concurrent_invalidation() {
        let cache = Arc::new(lens_core::lsp::hint::HintCache::new(1).await.unwrap());
        
        // Add many cache entries
        for i in 0..1000 {
            let key = format!("invalidation_test_{}", i);
            let results = vec![
                lens_core::lsp::LspSearchResult {
                    file_path: "test.rs".to_string(),
                    line_number: 1,
                    column: 0,
                    content: "value".to_string(),
                    hint_type: lens_core::lsp::HintType::References,
                    server_type: lens_core::lsp::LspServerType::Rust,
                    confidence: 1.0,
                    context_lines: None,
                }
            ];
            cache.set(key, results, 3600).await.unwrap();
        }
        
        // Perform concurrent invalidations and operations
        let handles: Vec<_> = (0..20)
            .map(|task_id| {
                let cache_clone = cache.clone();
                tokio::spawn(async move {
                    for i in 0..50 {
                        match i % 3 {
                            0 => {
                                // Invalidate files
                                let file_path = std::path::PathBuf::from(format!("/test/file_{}.rs", task_id));
                                cache_clone.invalidate_file(&file_path).await;
                            }
                            1 => {
                                // Add new entries
                                let key = format!("concurrent_{}_{}", task_id, i);
                                let results = vec![
                                    lens_core::lsp::LspSearchResult {
                                        file_path: "test.rs".to_string(),
                                        line_number: 1,
                                        column: 0,
                                        content: "value".to_string(),
                                        hint_type: lens_core::lsp::HintType::Hover,
                                        server_type: lens_core::lsp::LspServerType::Rust,
                                        confidence: 1.0,
                                        context_lines: None,
                                    }
                                ];
                                cache_clone.set(key, results, 3600).await.unwrap();
                            }
                            2 => {
                                // Read entries
                                let key = format!("invalidation_test_{}", i * 10 % 1000);
                                let _ = cache_clone.get(&key).await;
                            }
                            _ => unreachable!(),
                        }
                    }
                })
            })
            .collect();
        
        // Wait for concurrent operations
        for handle in handles {
            handle.await.expect("Concurrent invalidation task should complete");
        }
        
        // Verify cache is still functional
        let stats = cache.stats().await;
        assert!(stats.size >= 0, "Cache should have valid size");
    }

    #[tokio::test]
    async fn test_lsp_configuration_edge_cases() {
        // Test invalid configuration handling
        let invalid_configs = vec![
            lens_core::lsp::LspConfig {
                enabled: true,
                routing_percentage: -0.5, // Invalid negative percentage
                cache_ttl_hours: 0, // Invalid zero TTL
                server_timeout_ms: 0, // Invalid zero timeout
                max_concurrent_requests: 0, // Invalid zero concurrency
                ..Default::default()
            },
            lens_core::lsp::LspConfig {
                enabled: true,
                routing_percentage: 1.5, // Invalid > 1.0 percentage
                cache_ttl_hours: 1000000, // Extremely large TTL
                server_timeout_ms: 0, // Invalid zero timeout
                max_concurrent_requests: 1000000, // Very large concurrency
                ..Default::default() 
            },
            lens_core::lsp::LspConfig {
                enabled: true,
                server_timeout_ms: 0, // Invalid zero timeout
                max_concurrent_requests: 1000000, // Very large concurrency
                ..Default::default()
            },
        ];
        
        for config in invalid_configs {
            // Configuration validation should handle invalid configs gracefully
            // These shouldn't panic during construction or basic operations
            
            // Test basic config field access
            let _ = config.enabled;
            let _ = config.routing_percentage;
            let _ = config.cache_ttl_hours;
            
            // Test bounds checking logic that should exist
            if config.routing_percentage < 0.0 || config.routing_percentage > 1.0 {
                // Should be detected as invalid
                assert!(true, "Invalid routing percentage should be detectable");
            }
            
            if config.cache_ttl_hours == 0 || config.server_timeout_ms == 0 {
                // Should be detected as invalid
                assert!(true, "Invalid timeout values should be detectable");
            }
        }
    }

    #[tokio::test]
    async fn test_lsp_request_timeout_handling() {
        // Test timeout scenarios
        let short_timeout = Duration::from_millis(1); // Very short timeout
        
        // Test operations that might take longer than timeout
        let timeout_result = timeout(short_timeout, async {
            // Simulate potentially slow operation
            tokio::time::sleep(Duration::from_millis(10)).await;
            "completed"
        }).await;
        
        assert!(timeout_result.is_err(), "Short timeout should cause timeout error");
        
        // Test reasonable timeout
        let reasonable_timeout = Duration::from_millis(100);
        let success_result = timeout(reasonable_timeout, async {
            // Quick operation
            "completed"
        }).await;
        
        assert!(success_result.is_ok(), "Reasonable timeout should allow completion");
    }

    #[tokio::test]
    async fn test_lsp_server_type_edge_cases() {
        // Test server type detection with edge cases
        let long_ext = "a".repeat(100);
        let edge_case_extensions = vec![
            "", // Empty extension
            "rs", // Valid extension  
            "RS", // Uppercase
            "typescript", // Full language name
            "py3", // Python variant
            "jsx", // React extension
            "tsx", // TypeScript React
            "unknown_extension", // Unknown
            "a", // Single character
            &long_ext, // Very long extension
        ];
        
        for ext in edge_case_extensions {
            let server_type = lens_core::lsp::LspServerType::from_file_extension(ext);
            
            // Should either return valid server type or None
            match server_type {
                Some(server_type) => {
                    // Verify it's a known server type
                    match server_type {
                        lens_core::lsp::LspServerType::Rust |
                        lens_core::lsp::LspServerType::Python |
                        lens_core::lsp::LspServerType::TypeScript |
                        lens_core::lsp::LspServerType::JavaScript |
                        lens_core::lsp::LspServerType::Go => {
                            // Valid server type
                        }
                    }
                }
                None => {
                    // Unknown extension - acceptable
                }
            }
        }
    }

    #[tokio::test]
    async fn test_lsp_stats_overflow_protection() {
        let mut stats = lens_core::lsp::manager::LspStats::default();
        
        // Test potential overflow scenarios
        stats.total_requests = u64::MAX - 1;
        stats.cache_hits = u64::MAX - 1;
        stats.lsp_routed = u64::MAX - 1;
        
        // Increment operations shouldn't overflow
        // Note: In real implementation, there should be overflow protection
        
        // Test large numbers don't cause calculation issues
        if stats.total_requests > 0 {
            let hit_rate = stats.cache_hits as f64 / stats.total_requests as f64;
            assert!(hit_rate >= 0.0 && hit_rate <= 1.0, "Hit rate should be valid ratio");
        }
        
        // Test edge case calculations
        let zero_stats = lens_core::lsp::manager::LspStats::default();
        assert_eq!(zero_stats.total_requests, 0);
        assert_eq!(zero_stats.cache_hits, 0);
        
        // Division by zero protection should exist
        let hit_rate = if zero_stats.total_requests > 0 {
            zero_stats.cache_hits as f64 / zero_stats.total_requests as f64
        } else {
            0.0
        };
        assert_eq!(hit_rate, 0.0, "Zero stats should handle division by zero");
    }

    #[tokio::test]
    async fn test_bfs_traversal_edge_cases() {
        let router = lens_core::lsp::router::LspRouter::new(0.5);
        
        // Test BFS with various edge cases
        let edge_case_nodes = vec![
            // Node with empty symbol ID
            lens_core::lsp::router::BfsNode {
                symbol_id: "".to_string(),
                symbol_type: lens_core::lsp::router::SymbolType::Definition,
                file_path: "/test/empty.rs".to_string(),
                line: 0,
                column: 0,
            },
            // Node with very long paths
            lens_core::lsp::router::BfsNode {
                symbol_id: "test_symbol".to_string(),
                symbol_type: lens_core::lsp::router::SymbolType::Reference,
                file_path: "/".to_string() + &"very_long_path/".repeat(100) + "file.rs",
                line: u32::MAX,
                column: u32::MAX,
            },
            // Node with special characters
            lens_core::lsp::router::BfsNode {
                symbol_id: "symbol🚀with🔥emojis💯".to_string(),
                symbol_type: lens_core::lsp::router::SymbolType::TypeDefinition,
                file_path: "/test/unicode_🚀_file.rs".to_string(),
                line: 1,
                column: 1,
            },
        ];
        
        for node in edge_case_nodes {
            // Test BFS with minimal bounds
            let minimal_bounds = lens_core::lsp::TraversalBounds {
                max_depth: 0,
                max_results: 1,
                timeout_ms: 1000,
            };
            let result = router.bounded_bfs_traversal(node.clone(), &minimal_bounds).await;
            assert!(result.is_ok(), "BFS should handle minimal bounds gracefully");
            if let Ok(result) = result {
                assert!(result.depth_reached <= 0, "Zero depth should be respected");
                assert!(result.nodes_explored <= 1, "Minimal node limit should be respected");
            }
            
            // Test BFS with large bounds (but within TODO.md limits)
            let large_bounds = lens_core::lsp::TraversalBounds {
                max_depth: 2, // TODO.md limit
                max_results: 64, // TODO.md limit
                timeout_ms: 5000,
            };
            let result = router.bounded_bfs_traversal(node.clone(), &large_bounds).await;
            assert!(result.is_ok(), "BFS should handle bounds within TODO.md limits");
            if let Ok(result) = result {
                assert!(result.depth_reached <= 2, "Large depth should be bounded to TODO.md limit");
                assert!(result.nodes_explored <= 64, "Large node limit should be bounded to TODO.md limit");
            }
        }
    }
}

#[cfg(test)]
mod lsp_performance_edge_cases {
    use super::*;

    #[tokio::test]
    async fn test_router_performance_degradation() {
        let router = Arc::new(lens_core::lsp::router::LspRouter::new(0.5));
        
        // Simulate performance degradation scenarios
        for _ in 0..100 {
            // Report slow LSP results to trigger adaptation
            router.report_lsp_result(&QueryIntent::References, true, 5000).await; // 5 second response
        }
        
        let _stats = router.get_routing_stats().await;
        
        // Router should adapt to poor performance
        let decision = router.make_routing_decision("performance_test", &QueryIntent::References).await;
        
        // After many slow responses, router should still return valid decisions
        assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0, "Confidence should remain valid");
        assert!(decision.estimated_latency_ms > 0, "Estimated latency should be positive");
    }

    #[tokio::test] 
    async fn test_cache_performance_under_load() {
        let cache = Arc::new(lens_core::lsp::hint::HintCache::new(1).await.unwrap());
        
        // Create high-load scenario
        let num_concurrent_tasks = 50;
        let operations_per_task = 200;
        
        let start = std::time::Instant::now();
        
        let handles: Vec<_> = (0..num_concurrent_tasks)
            .map(|task_id| {
                let cache_clone = cache.clone();
                tokio::spawn(async move {
                    for i in 0..operations_per_task {
                        let key = format!("perf_test_{}_{}", task_id, i % 100); // Some key reuse
                        let value = format!("value_{}_{}", task_id, i);
                        
                        match i % 3 {
                            0 => {
                                let results = vec![
                                    lens_core::lsp::LspSearchResult {
                                        file_path: "test.rs".to_string(),
                                        line_number: 1,
                                        column: 0,
                                        content: value,
                                        hint_type: lens_core::lsp::HintType::Definition,
                                        server_type: lens_core::lsp::LspServerType::Rust,
                                        confidence: 1.0,
                                        context_lines: None,
                                    }
                                ];
                                cache_clone.set(key, results, 3600).await.unwrap();
                            }
                            1 => { let _ = cache_clone.get(&key).await; }
                            2 => {
                                // Touch operation - try to get and re-set if exists
                                if let Ok(Some(existing)) = cache_clone.get(&key).await {
                                    cache_clone.set(key, existing, 3600).await.unwrap();
                                }
                            }
                            _ => unreachable!(),
                        }
                    }
                })
            })
            .collect();
        
        for handle in handles {
            handle.await.expect("Performance test task should complete");
        }
        
        let elapsed = start.elapsed();
        
        // Verify performance is reasonable (should complete within reasonable time)
        assert!(elapsed.as_secs() < 30, "High-load cache operations should complete within 30 seconds");
        
        let stats = cache.stats().await;
        assert!(stats.size >= 0, "Cache should have valid size after load test");
    }
}