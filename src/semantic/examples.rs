//! # Semantic Integration Examples
//!
//! This module provides examples of how to integrate the new Rust-based
//! semantic processing system with the existing search engine.

use crate::search::{SearchEngine, SearchRequest, SearchConfig};
use super::integration::{SemanticSearchIntegration, SemanticSearchRequest, SearchEngineSemanticExt};
use super::{SemanticConfig, initialize_semantic_integration};
use anyhow::Result;
use tracing::info;

/// Example: Basic semantic search integration
pub async fn example_basic_semantic_search() -> Result<()> {
    info!("Running basic semantic search integration example");
    
    // 1. Initialize search engine with minimal configuration
    let search_config = SearchConfig {
        lsp_routing_rate: 0.0, // Disable LSP for simplicity
        enable_semantic_search: true,
        enable_pinned_datasets: false,
        ..Default::default()
    };
    
    let search_engine = SearchEngine::with_config("./example_index", search_config).await?;
    
    // 2. Initialize semantic integration
    let semantic_config = SemanticConfig::default();
    let semantic_integration = initialize_semantic_integration(&semantic_config).await?;
    
    // 3. Perform semantic search
    let query = "how to implement binary search algorithm";
    let semantic_response = search_engine.search_auto_semantic(query, &semantic_integration).await?;
    
    info!("Semantic search completed:");
    info!("- Query: {}", query);
    info!("- Results found: {}", semantic_response.base_response.results.len());
    info!("- Semantic enhanced: {}", semantic_response.semantic_enhanced);
    info!("- Processing time: {}ms", semantic_response.semantic_metrics.total_processing_time_ms);
    
    if let Some(classification) = &semantic_response.classification {
        info!("- Classified as: {:?} (confidence: {:.3})", 
              classification.intent, classification.confidence);
    }
    
    Ok(())
}

/// Example: Advanced semantic search with custom configuration
pub async fn example_advanced_semantic_search() -> Result<()> {
    info!("Running advanced semantic search integration example");
    
    // 1. Initialize search engine
    let search_engine = SearchEngine::new("./example_index").await?;
    
    // 2. Initialize semantic integration with custom config
    let integration_config = super::integration::SemanticIntegrationConfig {
        enabled: true,
        nl_upshift_threshold: 0.6, // Lower threshold for more semantic processing
        max_processing_time_ms: 200, // More generous time budget
        enable_conformal_routing: true,
        fallback_on_error: true,
        enable_result_caching: true,
        similarity_threshold: 0.4, // Lower threshold for more inclusive similarity
    };
    
    let semantic_integration = SemanticSearchIntegration::new(integration_config).await?;
    
    // 3. Test different query types
    let test_queries = vec![
        ("how to optimize search performance", "Natural Language"),
        ("SearchEngine::search", "Symbol Search"),  
        ("impl Iterator for", "Structural Search"),
        ("rust async await patterns", "Mixed Query"),
    ];
    
    for (query, query_type) in test_queries {
        info!("\n--- Testing {} Query ---", query_type);
        info!("Query: {}", query);
        
        let semantic_request = SemanticSearchRequest {
            base_request: SearchRequest {
                query: query.to_string(),
                max_results: 5,
                timeout_ms: 500,
                ..Default::default()
            },
            force_semantic: query_type == "Natural Language", // Force semantic for NL queries
            ..Default::default()
        };
        
        match semantic_integration.process_search(&search_engine, semantic_request).await {
            Ok(response) => {
                info!("✅ Results: {}", response.base_response.results.len());
                info!("✅ Semantic enhanced: {}", response.semantic_enhanced);
                info!("✅ Processing time: {}ms", response.semantic_metrics.total_processing_time_ms);
                
                if let Some(classification) = &response.classification {
                    info!("✅ Classification: {:?} (confidence: {:.3})", 
                          classification.intent, classification.confidence);
                }
                
                if let Some(routing) = &response.routing_decision {
                    info!("✅ Routing: {:?} (expected improvement: {:.3})", 
                          routing.upshift_type, routing.expected_improvement);
                }
            }
            Err(e) => {
                info!("❌ Search failed: {}", e);
            }
        }
    }
    
    // 4. Display overall integration metrics
    let metrics = semantic_integration.get_metrics().await;
    info!("\n--- Integration Metrics ---");
    info!("Total requests: {}", metrics.total_requests);
    info!("Successful enhancements: {}", metrics.successful_enhancements);
    info!("Fallback count: {}", metrics.fallback_count);
    info!("Average processing time: {:.2}ms", metrics.avg_processing_time_ms);
    
    Ok(())
}

/// Example: Health monitoring and diagnostics
pub async fn example_health_monitoring() -> Result<()> {
    info!("Running health monitoring example");
    
    // Initialize semantic integration
    let semantic_config = SemanticConfig::default();
    let semantic_integration = initialize_semantic_integration(&semantic_config).await?;
    
    // Perform health check
    let health_status = semantic_integration.health_check().await?;
    
    info!("=== Semantic System Health Status ===");
    info!("Overall healthy: {}", health_status.overall_healthy);
    info!("Encoder healthy: {}", health_status.encoder_healthy);
    info!("Classifier healthy: {}", health_status.classifier_healthy);
    info!("Intent router healthy: {}", health_status.intent_router_healthy);
    info!("Conformal router healthy: {}", health_status.conformal_router_healthy);
    info!("Last check: {}", health_status.last_check);
    
    if !health_status.overall_healthy {
        info!("⚠️ Some components are unhealthy - check individual status");
    } else {
        info!("✅ All semantic components are healthy");
    }
    
    Ok(())
}

/// Example: Performance benchmarking
pub async fn example_performance_benchmarking() -> Result<()> {
    info!("Running performance benchmarking example");
    
    let search_engine = SearchEngine::new("./example_index").await?;
    let semantic_config = SemanticConfig::default();
    let semantic_integration = initialize_semantic_integration(&semantic_config).await?;
    
    // Benchmark semantic vs non-semantic search
    let test_query = "rust error handling best practices";
    let iterations = 10;
    
    info!("Benchmarking query: '{}'", test_query);
    info!("Iterations: {}", iterations);
    
    // Benchmark non-semantic search
    let mut non_semantic_times = Vec::new();
    for _ in 0..iterations {
        let start = std::time::Instant::now();
        let _response = search_engine.search(test_query, 10).await?;
        non_semantic_times.push(start.elapsed().as_millis() as u64);
    }
    
    // Benchmark semantic search
    let mut semantic_times = Vec::new();
    for _ in 0..iterations {
        let start = std::time::Instant::now();
        let _response = search_engine.search_auto_semantic(test_query, &semantic_integration).await?;
        semantic_times.push(start.elapsed().as_millis() as u64);
    }
    
    // Calculate statistics
    let avg_non_semantic = non_semantic_times.iter().sum::<u64>() as f64 / iterations as f64;
    let avg_semantic = semantic_times.iter().sum::<u64>() as f64 / iterations as f64;
    
    let min_non_semantic = *non_semantic_times.iter().min().unwrap();
    let max_non_semantic = *non_semantic_times.iter().max().unwrap();
    let min_semantic = *semantic_times.iter().min().unwrap();
    let max_semantic = *semantic_times.iter().max().unwrap();
    
    info!("\n=== Performance Results ===");
    info!("Non-semantic search:");
    info!("  Average: {:.2}ms", avg_non_semantic);
    info!("  Min: {}ms, Max: {}ms", min_non_semantic, max_non_semantic);
    
    info!("Semantic search:");
    info!("  Average: {:.2}ms", avg_semantic);
    info!("  Min: {}ms, Max: {}ms", min_semantic, max_semantic);
    
    let overhead = avg_semantic - avg_non_semantic;
    let overhead_percentage = (overhead / avg_non_semantic) * 100.0;
    
    info!("Semantic overhead: {:.2}ms ({:.1}%)", overhead, overhead_percentage);
    
    if overhead_percentage < 50.0 {
        info!("✅ Semantic processing overhead is acceptable");
    } else {
        info!("⚠️ Semantic processing overhead is high - consider optimization");
    }
    
    Ok(())
}

/// Run all examples
pub async fn run_all_examples() -> Result<()> {
    info!("🚀 Running all semantic integration examples");
    
    if let Err(e) = example_basic_semantic_search().await {
        info!("❌ Basic example failed: {}", e);
    }
    
    if let Err(e) = example_advanced_semantic_search().await {
        info!("❌ Advanced example failed: {}", e);
    }
    
    if let Err(e) = example_health_monitoring().await {
        info!("❌ Health monitoring example failed: {}", e);
    }
    
    if let Err(e) = example_performance_benchmarking().await {
        info!("❌ Performance benchmarking example failed: {}", e);
    }
    
    info!("✅ All semantic integration examples completed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    #[ignore] // Ignore by default since it requires index setup
    async fn test_basic_semantic_integration() {
        let result = example_basic_semantic_search().await;
        assert!(result.is_ok(), "Basic semantic integration example should succeed");
    }
    
    #[tokio::test]
    async fn test_health_monitoring() {
        let result = example_health_monitoring().await;
        assert!(result.is_ok(), "Health monitoring example should succeed");
    }
}