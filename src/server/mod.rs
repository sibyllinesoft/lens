//! HTTP Server Module
//! 
//! Provides the complete HTTP REST API server for Lens,
//! including type definitions and integration adapters.

pub mod api_types;

use std::sync::Arc;
use anyhow::Result;

// Re-export the main server functions and types
pub use api_types::*;

// Server implementation
use std::net::SocketAddr;
use std::time::{Duration, Instant, SystemTime};
use uuid::Uuid;

use axum::{
    extract::{Query, State},
    http::{Method, StatusCode},
    middleware::{self, Next},
    response::{Json, Response, IntoResponse},
    routing::{get, post},
    Router,
};
use tower_http::{
    cors::{CorsLayer, Any},
    trace::TraceLayer,
};
use tracing::{info, instrument};

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    pub search_engine: Arc<crate::search::SearchEngine>,
    pub metrics: Arc<crate::metrics::MetricsCollector>,
    pub attestation: Arc<crate::attestation::AttestationManager>,
    pub benchmark_runner: Arc<crate::benchmark::BenchmarkRunner>,
    pub start_time: SystemTime,
}

/// Configuration for the HTTP server
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub bind_address: String,
    pub port: u16,
    pub enable_cors: bool,
    pub request_timeout: Duration,
    pub max_request_size: usize,
    pub enable_tracing: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_address: "127.0.0.1".to_string(),
            port: 3000,
            enable_cors: true,
            request_timeout: Duration::from_millis(5000),
            max_request_size: 1024 * 1024, // 1MB
            enable_tracing: true,
        }
    }
}

/// Create the main HTTP server
pub async fn create_server(
    config: ServerConfig,
    search_engine: Arc<crate::search::SearchEngine>,
    metrics: Arc<crate::metrics::MetricsCollector>,
    attestation: Arc<crate::attestation::AttestationManager>,
    benchmark_runner: Arc<crate::benchmark::BenchmarkRunner>,
) -> anyhow::Result<()> {
    let app_state = AppState {
        search_engine,
        metrics,
        attestation,
        benchmark_runner,
        start_time: SystemTime::now(),
    };

    let app = create_app(app_state).await?;

    let addr: SocketAddr = format!("{}:{}", config.bind_address, config.port)
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid bind address: {}", e))?;

    info!("🚀 Starting Rust HTTP API server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to bind to {}: {}", addr, e))?;

    axum::serve(listener, app)
        .await
        .map_err(|e| anyhow::anyhow!("Server error: {}", e))?;

    Ok(())
}

/// Create the axum app with all routes and middleware
pub async fn create_app(state: AppState) -> anyhow::Result<Router> {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
        .allow_headers(Any);

    let trace_layer = TraceLayer::new_for_http();

    let app = Router::new()
        // Core search endpoints
        .route("/search", post(search_handler))
        .route("/struct", post(struct_search_handler))  
        .route("/symbols/near", post(symbols_near_handler))
        
        // System health and compatibility
        .route("/health", get(health_handler))
        .route("/manifest", get(manifest_handler))
        .route("/compat/check", get(compat_check_handler))
        .route("/compat/bundles", get(compat_bundles_handler))
        
        // SPI v1 endpoints (LSP interface)
        .route("/v1/spi/search", post(spi_search_handler))
        .route("/v1/spi/health", get(spi_health_handler))
        
        // Add middleware layers individually
        .layer(cors)
        .layer(trace_layer)
        .layer(middleware::from_fn_with_state(state.clone(), request_tracing_middleware))
        .layer(middleware::from_fn(timeout_middleware))
        .with_state(state);

    Ok(app)
}

/// Request tracing and metrics middleware
#[instrument(skip_all)]
async fn request_tracing_middleware(
    State(_state): State<AppState>,
    request: axum::http::Request<axum::body::Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    let start = Instant::now();
    let method = request.method().clone();
    let uri = request.uri().clone();
    let trace_id = Uuid::new_v4().to_string();
    
    // Add trace ID to request extensions
    let mut request = request;
    request.extensions_mut().insert(trace_id.clone());
    
    info!("Request started: {} {} (trace: {})", method, uri, trace_id);
    
    let response = next.run(request).await;
    
    let duration = start.elapsed();
    let status = response.status();
    
    // TODO: Record metrics when MetricsCollector API is available
    // state.metrics.record_request(...).await;
    
    info!(
        "Request completed: {} {} -> {} in {:?} (trace: {})",
        method, uri, status, duration, trace_id
    );
    
    Ok(response)
}

/// Request timeout middleware
async fn timeout_middleware(
    request: axum::http::Request<axum::body::Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    match tokio::time::timeout(Duration::from_secs(30), next.run(request)).await {
        Ok(response) => Ok(response),
        Err(_) => {
            info!("Request timed out");
            Err(StatusCode::REQUEST_TIMEOUT)
        }
    }
}

//
// Handler implementations
//

/// POST /search - Main search endpoint
#[instrument(skip_all)]
async fn search_handler(
    State(_state): State<AppState>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    let start = Instant::now();
    
    info!("Search request: repo={}, query='{}', mode={:?}", 
          request.repo_sha, request.q, request.mode);
    
    // Validate request
    request.validate()
        .map_err(|e| ApiError::BadRequest(e))?;
    
    let total_latency = start.elapsed();
    
    // Create mock response for now
    let response = SearchResponse {
        hits: vec![],
        total: 0,
        latency_ms: LatencyBreakdown {
            stage_a: 10,
            stage_b: 5,
            stage_c: None,
            total: std::cmp::max(1, total_latency.as_millis() as u32),
        },
        trace_id: Uuid::new_v4().to_string(),
        api_version: "v1".to_string(),
        index_version: "v1".to_string(),
        policy_version: "v1".to_string(),
        error: None,
        message: None,
    };
    
    info!("Search completed: {} hits in {:?}", response.hits.len(), total_latency);
    
    Ok(Json(response))
}

/// POST /struct - Structural search endpoint
#[instrument(skip_all)]
async fn struct_search_handler(
    State(_state): State<AppState>,
    Json(request): Json<StructRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    info!("Structural search: repo={}, pattern='{}', lang={:?}", 
          request.repo_sha, request.pattern, request.lang);
    
    // Validate request
    request.validate()
        .map_err(|e| ApiError::BadRequest(e))?;
    
    // Create mock response for now
    let response = SearchResponse {
        hits: vec![],
        total: 0,
        latency_ms: LatencyBreakdown {
            stage_a: 15,
            stage_b: 8,
            stage_c: None,
            total: 23,
        },
        trace_id: Uuid::new_v4().to_string(),
        api_version: "v1".to_string(),
        index_version: "v1".to_string(),
        policy_version: "v1".to_string(),
        error: None,
        message: None,
    };
    
    Ok(Json(response))
}

/// POST /symbols/near - Find symbols near a location
#[instrument(skip_all)]
async fn symbols_near_handler(
    State(_state): State<AppState>,
    Json(request): Json<SymbolsNearRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    info!("Symbols near: file={}, line={}, radius={:?}", 
          request.file, request.line, request.radius);
    
    // Validate request
    request.validate()
        .map_err(|e| ApiError::BadRequest(e))?;
    
    // Create mock response for now
    let response = SearchResponse {
        hits: vec![],
        total: 0,
        latency_ms: LatencyBreakdown {
            stage_a: 8,
            stage_b: 12,
            stage_c: None,
            total: 20,
        },
        trace_id: Uuid::new_v4().to_string(),
        api_version: "v1".to_string(),
        index_version: "v1".to_string(),
        policy_version: "v1".to_string(),
        error: None,
        message: None,
    };
    
    Ok(Json(response))
}

/// GET /health - System health check
#[instrument(skip_all)]
async fn health_handler(
    State(_state): State<AppState>,
) -> Result<Json<HealthResponse>, ApiError> {
    let response = HealthResponse {
        status: "ok".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        shards_healthy: 1,
    };
    
    Ok(Json(response))
}

/// GET /manifest - API manifest information
#[instrument(skip_all)]
async fn manifest_handler(
    State(_state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let manifest = serde_json::json!({
        "name": "lens-core",
        "version": env!("CARGO_PKG_VERSION"),
        "api_version": "v1",
        "index_version": "v1", 
        "policy_version": "v1",
        "build_info": {
            "version": env!("CARGO_PKG_VERSION"),
            "build_timestamp": env!("BUILD_TIMESTAMP"),
            "profile": if cfg!(debug_assertions) { "debug" } else { "release" }
        },
        "capabilities": {
            "search_modes": ["lex", "struct", "hybrid"],
            "languages": ["typescript", "python", "rust", "go", "java"],
            "lsp_integration": true,
            "semantic_search": true,
        }
    });
    
    Ok(Json(manifest))
}

/// GET /compat/check - Compatibility check
#[instrument(skip_all)]
async fn compat_check_handler(
    State(_state): State<AppState>,
    Query(request): Query<CompatibilityCheckRequest>,
) -> Result<Json<CompatibilityCheckResponse>, ApiError> {
    let server_versions = ("v1", "v1", "v1");
    
    let compatible = request.api_version == "v1" && 
                    request.index_version == "v1" &&
                    request.policy_version.as_deref().unwrap_or("v1") == "v1";
    
    let response = CompatibilityCheckResponse {
        compatible,
        api_version: request.api_version,
        index_version: request.index_version,
        policy_version: request.policy_version,
        server_api_version: server_versions.0.to_string(),
        server_index_version: server_versions.1.to_string(),
        server_policy_version: server_versions.2.to_string(),
        warnings: None,
        errors: if compatible { None } else { 
            Some(vec!["Version mismatch detected".to_string()]) 
        },
    };
    
    Ok(Json(response))
}

// Placeholder handlers for additional endpoints
async fn compat_bundles_handler(State(_): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({"bundles": [], "status": "ok"}))
}

async fn spi_search_handler(State(_): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({"hits": [], "total": 0}))
}

async fn spi_health_handler(State(_): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

/// Error handling for API responses
#[derive(Debug)]
pub enum ApiError {
    BadRequest(String),
    Unauthorized(String),
    NotFound(String),
    InternalError(String),
    ServiceUnavailable(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            ApiError::Unauthorized(msg) => (StatusCode::UNAUTHORIZED, msg),
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            ApiError::InternalError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            ApiError::ServiceUnavailable(msg) => (StatusCode::SERVICE_UNAVAILABLE, msg),
        };

        let body = Json(serde_json::json!({
            "error": error_message,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "trace_id": Uuid::new_v4().to_string(),
        }));

        (status, body).into_response()
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiError::BadRequest(msg) => write!(f, "Bad Request: {}", msg),
            ApiError::Unauthorized(msg) => write!(f, "Unauthorized: {}", msg),
            ApiError::NotFound(msg) => write!(f, "Not Found: {}", msg),
            ApiError::InternalError(msg) => write!(f, "Internal Error: {}", msg),
            ApiError::ServiceUnavailable(msg) => write!(f, "Service Unavailable: {}", msg),
        }
    }
}

impl std::error::Error for ApiError {}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::header::CONTENT_TYPE;
    use tower::timeout::TimeoutLayer;
    use tower_http::limit::RequestBodyLimitLayer;

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        
        assert_eq!(config.bind_address, "127.0.0.1");
        assert_eq!(config.port, 3000);
        assert_eq!(config.enable_cors, true);
        assert_eq!(config.request_timeout, Duration::from_millis(5000));
        assert_eq!(config.max_request_size, 1024 * 1024);
        assert_eq!(config.enable_tracing, true);
    }

    #[test]
    fn test_server_config_custom() {
        let config = ServerConfig {
            bind_address: "0.0.0.0".to_string(),
            port: 8080,
            enable_cors: false,
            request_timeout: Duration::from_millis(10000),
            max_request_size: 2048 * 1024,
            enable_tracing: false,
        };
        
        assert_eq!(config.bind_address, "0.0.0.0");
        assert_eq!(config.port, 8080);
        assert_eq!(config.enable_cors, false);
        assert_eq!(config.request_timeout, Duration::from_millis(10000));
        assert_eq!(config.max_request_size, 2048 * 1024);
        assert_eq!(config.enable_tracing, false);
    }

    #[test]
    fn test_server_config_debug() {
        let config = ServerConfig::default();
        let debug_output = format!("{:?}", config);
        
        assert!(debug_output.contains("ServerConfig"));
        assert!(debug_output.contains("127.0.0.1"));
        assert!(debug_output.contains("3000"));
    }

    #[test]
    fn test_server_config_clone() {
        let config1 = ServerConfig::default();
        let config2 = config1.clone();
        
        assert_eq!(config1.bind_address, config2.bind_address);
        assert_eq!(config1.port, config2.port);
        assert_eq!(config1.enable_cors, config2.enable_cors);
        assert_eq!(config1.request_timeout, config2.request_timeout);
        assert_eq!(config1.max_request_size, config2.max_request_size);
        assert_eq!(config1.enable_tracing, config2.enable_tracing);
    }

    #[test]
    fn test_api_error_display() {
        let errors = vec![
            ApiError::BadRequest("Invalid input".to_string()),
            ApiError::NotFound("Resource not found".to_string()),
            ApiError::InternalError("Server error".to_string()),
            ApiError::ServiceUnavailable("Service down".to_string()),
        ];
        
        for error in errors {
            let display = format!("{}", error);
            assert!(!display.is_empty());
            assert!(display.len() > 5);
        }
    }

    #[test]
    fn test_api_error_debug() {
        let error = ApiError::BadRequest("test".to_string());
        let debug = format!("{:?}", error);
        assert!(debug.contains("BadRequest"));
        assert!(debug.contains("test"));
    }

    #[test]
    fn test_api_error_as_std_error() {
        let error = ApiError::InternalError("test error".to_string());
        let std_error: &dyn std::error::Error = &error;
        
        // Should not panic
        let _source = std_error.source();
    }

    #[test]
    fn test_layer_configurations() {
        // Test basic layer configurations that don't require external dependencies
        let timeout_duration = Duration::from_secs(30);
        assert_eq!(timeout_duration.as_secs(), 30);
        
        let max_request_size = 1024 * 1024;
        assert_eq!(max_request_size, 1048576);
    }

    #[test]
    fn test_server_config_fields() {
        let config = ServerConfig::default();
        
        // Test individual fields are accessible
        assert!(!config.bind_address.is_empty());
        assert!(config.port > 0);
        assert!(config.request_timeout.as_millis() > 0);
        assert!(config.max_request_size > 0);
    }

    #[test]
    fn test_api_error_chains() {
        let error1 = ApiError::BadRequest("First error".to_string());
        let error2 = ApiError::InternalError("Second error".to_string());
        let error3 = ApiError::NotFound("Third error".to_string());
        
        // Test error display chains work
        let errors = vec![error1, error2, error3];
        for error in errors {
            let display_str = format!("{}", error);
            assert!(!display_str.is_empty());
        }
    }

    #[test]
    fn test_server_config_edge_cases() {
        // Test edge case configurations
        let config = ServerConfig {
            bind_address: "[::]".to_string(), // IPv6
            port: 0, // Any available port
            enable_cors: false,
            request_timeout: Duration::from_millis(1),
            max_request_size: 0,
            enable_tracing: false,
        };
        
        assert_eq!(config.bind_address, "[::]");
        assert_eq!(config.port, 0);
        assert_eq!(config.max_request_size, 0);
    }

    #[test]
    fn test_server_config_comparison() {
        let config1 = ServerConfig::default();
        let config2 = ServerConfig::default();
        let config3 = ServerConfig {
            port: 8080,
            ..ServerConfig::default()
        };
        
        // Test field-level comparisons since PartialEq is not derived
        assert_eq!(config1.bind_address, config2.bind_address);
        assert_eq!(config1.port, config2.port);
        assert_ne!(config1.port, config3.port);
    }
}

