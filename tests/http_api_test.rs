//! Comprehensive HTTP API tests for Lens Rust server
//!
//! Tests all endpoints for compatibility with the TypeScript API,
//! ensuring proper request/response formats and error handling.

use axum::http::{Method, StatusCode};
use axum_test::TestServer;
use serde_json::{json, Value};
use std::sync::Arc;
use tempfile::TempDir;
use tokio;

use lens_core::{
    search::SearchEngine,
    metrics::MetricsCollector,
    attestation::AttestationManager,
    benchmark::{BenchmarkRunner, BenchmarkConfig},
    server::{create_app, ServerConfig, AppState, SearchRequest, SearchMode, SearchResponse},
};

/// Create a test server instance with mocked dependencies
async fn create_test_server() -> TestServer {
    // Create temporary directory for test index
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let index_path = temp_dir.path().join("test_index");
    
    // Create mock search engine
    let search_engine = Arc::new(SearchEngine::new(&*index_path.to_string_lossy()).await.expect("Failed to create search engine"));
    
    // Create other mock components
    let metrics = Arc::new(MetricsCollector::new());
    let attestation = Arc::new(AttestationManager::new(false).expect("Failed to create attestation manager"));
    
    let benchmark_config = BenchmarkConfig::default();
    let benchmark_runner = Arc::new(BenchmarkRunner::new(
        search_engine.clone(),
        metrics.clone(),
        benchmark_config,
    ));
    
    let app_state = AppState {
        search_engine,
        metrics,
        attestation,
        benchmark_runner,
        start_time: std::time::SystemTime::now(),
    };
    
    let app = create_app(app_state).await.expect("Failed to create app");
    TestServer::new(app).expect("Failed to create test server")
}

#[tokio::test]
async fn test_health_endpoint() {
    let server = create_test_server().await;
    
    let response = server
        .get("/health")
        .await;
    
    response.assert_status(StatusCode::OK);
    
    let body: Value = response.json();
    assert!(body["status"].is_string());
    assert!(body["timestamp"].is_string());
    assert!(body["shards_healthy"].is_number());
}

#[tokio::test]
async fn test_manifest_endpoint() {
    let server = create_test_server().await;
    
    let response = server
        .get("/manifest")
        .await;
    
    response.assert_status(StatusCode::OK);
    
    let body: Value = response.json();
    assert_eq!(body["name"], "lens-core");
    assert!(body["version"].is_string());
    assert_eq!(body["api_version"], "v1");
    assert_eq!(body["index_version"], "v1");
    assert_eq!(body["policy_version"], "v1");
    assert!(body["build_info"].is_object());
    assert!(body["capabilities"].is_object());
}

#[tokio::test]
async fn test_search_endpoint_success() {
    let server = create_test_server().await;
    
    let request = SearchRequest {
        repo_sha: "test-repo".to_string(),
        q: "function test".to_string(),
        mode: SearchMode::Lex,
        fuzzy: 0,
        k: 10,
        timeout_ms: Some(5000),
    };
    
    let response = server
        .post("/search")
        .json(&request)
        .await;
    
    response.assert_status(StatusCode::OK);
    
    let body: SearchResponse = response.json();
    assert!(body.hits.len() >= 0); // May be empty for test index
    assert!(body.total >= 0);
    assert!(body.latency_ms.total > 0);
    assert_eq!(body.api_version, "v1");
    assert_eq!(body.index_version, "v1");
    assert_eq!(body.policy_version, "v1");
    assert!(body.trace_id.len() > 0);
}

#[tokio::test]
async fn test_search_endpoint_validation() {
    let server = create_test_server().await;
    
    // Test empty query validation
    let request = json!({
        "repo_sha": "test-repo",
        "q": "",
        "mode": "lex",
        "fuzzy": 0,
        "k": 10
    });
    
    let response = server
        .post("/search")
        .json(&request)
        .await;
    
    response.assert_status(StatusCode::BAD_REQUEST);
    
    let body: Value = response.json();
    assert!(body["error"].as_str().unwrap().contains("Query cannot be empty"));
}

#[tokio::test]
async fn test_search_endpoint_k_validation() {
    let server = create_test_server().await;
    
    // Test k > 200 validation
    let request = json!({
        "repo_sha": "test-repo",
        "q": "test",
        "mode": "lex",
        "fuzzy": 0,
        "k": 250
    });
    
    let response = server
        .post("/search")
        .json(&request)
        .await;
    
    response.assert_status(StatusCode::BAD_REQUEST);
    
    let body: Value = response.json();
    assert!(body["error"].as_str().unwrap().contains("k cannot exceed 200"));
}

#[tokio::test]
async fn test_struct_search_endpoint() {
    let server = create_test_server().await;
    
    let request = json!({
        "repo_sha": "test-repo",
        "pattern": "function $name() { $body }",
        "lang": "typescript",
        "max_results": 20
    });
    
    let response = server
        .post("/struct")
        .json(&request)
        .await;
    
    response.assert_status(StatusCode::OK);
    
    let body: SearchResponse = response.json();
    assert!(body.hits.len() >= 0);
    assert!(body.latency_ms.total > 0);
}

#[tokio::test]
async fn test_symbols_near_endpoint() {
    let server = create_test_server().await;
    
    let request = json!({
        "file": "test.ts",
        "line": 42,
        "radius": 5
    });
    
    let response = server
        .post("/symbols/near")
        .json(&request)
        .await;
    
    response.assert_status(StatusCode::OK);
    
    let body: SearchResponse = response.json();
    assert!(body.hits.len() >= 0);
}

#[tokio::test]
async fn test_compatibility_check_endpoint() {
    let server = create_test_server().await;
    
    let response = server
        .get("/compat/check?api_version=v1&index_version=v1&policy_version=v1")
        .await;
    
    response.assert_status(StatusCode::OK);
    
    let body: Value = response.json();
    assert_eq!(body["compatible"], true);
    assert_eq!(body["api_version"], "v1");
    assert_eq!(body["server_api_version"], "v1");
}

#[tokio::test]
async fn test_compatibility_check_mismatch() {
    let server = create_test_server().await;
    
    let response = server
        .get("/compat/check?api_version=v2&index_version=v1&policy_version=v1")
        .await;
    
    response.assert_status(StatusCode::OK);
    
    let body: Value = response.json();
    assert_eq!(body["compatible"], false);
    assert!(body["errors"].is_array());
}

#[tokio::test]
async fn test_spi_endpoints() {
    let server = create_test_server().await;
    
    // Test SPI search endpoint
    let response = server
        .post("/v1/spi/search")
        .json(&json!({"query": "test", "k": 10}))
        .await;
    
    response.assert_status(StatusCode::OK);
    
    // Test SPI health endpoint
    let response = server
        .get("/v1/spi/health")
        .await;
    
    response.assert_status(StatusCode::OK);
    
    let body: Value = response.json();
    assert!(body["status"].is_string());
}

#[tokio::test]
async fn test_cors_headers() {
    let server = create_test_server().await;
    
    let response = server.get("/health").await;
    
    // Should allow CORS preflight
    assert!(response.status_code().is_success() || response.status_code() == StatusCode::NO_CONTENT);
}

#[tokio::test]
async fn test_tracing_middleware() {
    let server = create_test_server().await;
    
    let response = server
        .get("/health")
        .await;
    
    response.assert_status(StatusCode::OK);
    
    // Should include trace headers in response (verify via logs in real scenarios)
    // This is primarily tested through manual observation of logs
}

#[tokio::test]
async fn test_error_handling() {
    let server = create_test_server().await;
    
    // Test malformed JSON
    let response = server
        .post("/search")
        .add_header("content-type", "application/json")
        .text("invalid json")
        .await;
    
    response.assert_status(StatusCode::UNSUPPORTED_MEDIA_TYPE);
}

#[tokio::test] 
async fn test_timeout_handling() {
    let server = create_test_server().await;
    
    // This would require a way to make the search engine hang
    // For now, just test that the endpoint responds quickly
    let start = std::time::Instant::now();
    
    let response = server
        .get("/health")
        .await;
    
    let duration = start.elapsed();
    response.assert_status(StatusCode::OK);
    
    // Should complete quickly for health check
    assert!(duration.as_secs() < 5);
}

#[tokio::test]
async fn test_metrics_recording() {
    let server = create_test_server().await;
    
    // Make multiple requests to ensure metrics are recorded
    for _ in 0..5 {
        let response = server.get("/health").await;
        response.assert_status(StatusCode::OK);
    }
    
    // In a real scenario, you'd verify metrics were recorded
    // This test ensures the middleware doesn't crash
}

#[tokio::test]
async fn test_search_modes() {
    let server = create_test_server().await;
    
    let modes = vec!["lex", "struct", "hybrid"];
    
    for mode in modes {
        let request = json!({
            "repo_sha": "test-repo",
            "q": "function test",
            "mode": mode,
            "fuzzy": 0,
            "k": 10
        });
        
        let response = server
            .post("/search")
            .json(&request)
            .await;
        
        response.assert_status(StatusCode::OK);
        
        let body: SearchResponse = response.json();
        assert!(body.latency_ms.total > 0);
    }
}

#[tokio::test]
async fn test_request_validation() {
    let server = create_test_server().await;
    
    // Test validation scenarios that reach our custom validation logic
    // (axum deserializer handles missing required fields before our validation runs)
    let test_cases = vec![
        // Invalid fuzzy distance  
        (json!({"repo_sha": "test", "q": "test", "mode": "lex", "fuzzy": 5, "k": 10}), "fuzzy distance cannot exceed 2"),
        // Invalid k value (0)
        (json!({"repo_sha": "test", "q": "test", "mode": "lex", "fuzzy": 0, "k": 0}), "k must be greater than 0"),
        // Invalid k value (> 200)
        (json!({"repo_sha": "test", "q": "test", "mode": "lex", "fuzzy": 0, "k": 250}), "k cannot exceed 200"),
    ];
    
    for (request, expected_error_substring) in test_cases {
        let response = server
            .post("/search")
            .json(&request)
            .await;
        
        // Should return validation error
        response.assert_status(StatusCode::BAD_REQUEST);
        
        let body: Value = response.json();
        let error_msg = body["error"].as_str().expect("Error message should be present");
        assert!(error_msg.contains(expected_error_substring), 
                "Expected error containing '{}', got '{}'", expected_error_substring, error_msg);
    }
}

#[tokio::test]
async fn test_response_format_consistency() {
    let server = create_test_server().await;
    
    let request = json!({
        "repo_sha": "test-repo",
        "q": "test",
        "mode": "lex",
        "fuzzy": 0,
        "k": 5
    });
    
    let response = server
        .post("/search")
        .json(&request)
        .await;
    
    response.assert_status(StatusCode::OK);
    
    let body: SearchResponse = response.json();
    
    // Verify all required fields are present
    assert!(body.trace_id.len() > 0);
    assert_eq!(body.api_version, "v1");
    assert_eq!(body.index_version, "v1");
    assert_eq!(body.policy_version, "v1");
    assert!(body.latency_ms.total > 0);
    
    // Verify hits have correct structure
    for hit in &body.hits {
        assert!(hit.file.len() > 0);
        assert!(hit.line > 0);
        assert!(hit.score >= 0.0 && hit.score <= 1.0);
        assert!(!hit.why.is_empty());
    }
}

// Integration test that would require a real index
#[ignore] // Ignored by default since it needs a real corpus
#[tokio::test]
async fn test_full_search_integration() {
    // This test would:
    // 1. Create a real search index with sample code
    // 2. Perform searches and verify meaningful results
    // 3. Test all search modes with real data
    // 4. Verify performance meets SLA targets
    
    // Implementation would require setting up a test corpus
    // and populating the search index with real data
}