//! HTTP Server Implementation for Lens Search
//!
//! This module provides a real HTTP API server that uses the Tantivy-based
//! search engine instead of simulations. It replaces the JavaScript simulation
//! layers with actual production functionality.

use anyhow::Result;
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use lens_common::{LensError, ProgrammingLanguage};
use lens_search_engine::{
    parse_full_query, QueryBuilder, QueryType, SearchConfig as EngineSearchConfig, SearchEngine,
};
use serde::{Deserialize, Serialize};
use std::{path::PathBuf, sync::Arc, time::Instant};
use tower_http::cors::CorsLayer;
use tracing::{error, info, warn};

/// HTTP server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub bind: String,
    pub port: u16,
    pub enable_cors: bool,
    pub index_path: PathBuf,
    pub search_config: EngineSearchConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        let search_config = EngineSearchConfig::default();
        Self {
            bind: "127.0.0.1".to_string(),
            port: 3000,
            enable_cors: true,
            index_path: search_config.index_path.clone(),
            search_config,
        }
    }
}

/// Server state containing the search engine
#[derive(Clone)]
pub struct ServerState {
    pub search_engine: Arc<SearchEngine>,
    pub started_at: Instant,
    pub index_path: PathBuf,
}

/// Start the HTTP server with real search functionality
pub async fn start_server(config: ServerConfig) -> Result<()> {
    let ServerConfig {
        bind,
        port,
        enable_cors,
        index_path,
        search_config,
    } = config;

    info!(
        "Starting Lens HTTP server at {}:{} with index at: {:?}",
        bind, port, index_path
    );

    let search_engine = Arc::new(SearchEngine::with_config(search_config).await?);
    let started_at = Instant::now();

    // Create server state
    let state = ServerState {
        search_engine,
        started_at,
        index_path: index_path.clone(),
    };

    // Create the router with real endpoints
    let app = create_router(state, enable_cors);

    // Start the server
    let listener = tokio::net::TcpListener::bind(format!("{}:{}", bind, port)).await?;
    info!("Lens HTTP server listening on http://{}:{}", bind, port);

    let shutdown_signal = async {
        if let Err(err) = tokio::signal::ctrl_c().await {
            warn!("Failed to listen for shutdown signal: {}", err);
        } else {
            info!("Shutdown signal received, shutting down HTTP server");
        }
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal)
        .await?;

    info!("HTTP server shutdown complete");

    Ok(())
}

/// Create the router with all API endpoints
fn create_router(state: ServerState, enable_cors: bool) -> Router {
    let mut app = Router::new()
        // Core search endpoints (replaces simulation)
        .route("/search", get(search_handler))
        .route("/search", post(search_handler))
        .route("/api/search", get(search_handler))
        .route("/api/search", post(search_handler))
        // Index management endpoints (real functionality)
        .route("/index", post(index_handler))
        .route("/api/index", post(index_handler))
        .route("/optimize", post(optimize_handler))
        .route("/api/optimize", post(optimize_handler))
        .route("/clear", post(clear_handler))
        .route("/api/clear", post(clear_handler))
        // Stats and health endpoints (real data)
        .route("/stats", get(stats_handler))
        .route("/api/stats", get(stats_handler))
        .route("/health", get(health_handler))
        .route("/api/health", get(health_handler))
        // Advanced search endpoints (real functionality)
        .route("/search/fuzzy", get(fuzzy_search_handler))
        .route("/search/symbol", get(symbol_search_handler))
        .route("/search/exact", get(exact_search_handler))
        .route("/api/search/fuzzy", get(fuzzy_search_handler))
        .route("/api/search/symbol", get(symbol_search_handler))
        .route("/api/search/exact", get(exact_search_handler))
        .with_state(state);

    // Add CORS if enabled
    if enable_cors {
        app = app.layer(CorsLayer::permissive());
    }

    app
}

// ============================================================================
// Request/Response Types (replacing simulation types)
// ============================================================================

#[derive(Deserialize)]
pub struct SearchParams {
    pub q: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: Option<usize>,
    #[serde(default)]
    pub fuzzy: bool,
    #[serde(default)]
    pub symbols: bool,
    pub language: Option<String>,
    pub file_pattern: Option<String>,
}

fn default_limit() -> usize {
    10
}

const MAX_SEARCH_LIMIT: usize = 100;
const MAX_SEARCH_OFFSET: usize = 100_000;

#[derive(Serialize, Debug)]
pub struct SearchResponse {
    pub query: String,
    pub total: usize,
    pub limit: usize,
    pub offset: usize,
    pub duration_ms: u64,
    pub query_type: String,
    pub from_cache: bool,
    pub results: Vec<SearchResultResponse>,
    pub index_stats: IndexStatsResponse,
}

#[derive(Serialize, Debug)]
pub struct SearchResultResponse {
    pub file_path: String,
    pub line_number: u32,
    pub content: String,
    pub score: f64,
    pub language: Option<String>,
    pub matched_terms: Vec<String>,
    pub context_lines: Option<Vec<String>>,
}

#[derive(Deserialize)]
pub struct IndexRequest {
    pub directory: String,
    #[serde(default)]
    pub force: bool,
    #[serde(default)]
    pub recursive: bool,
}

#[derive(Serialize, Debug)]
pub struct IndexResponse {
    pub success: bool,
    pub message: String,
    pub files_indexed: usize,
    pub files_failed: usize,
    pub lines_indexed: usize,
    pub symbols_extracted: usize,
    pub duration_ms: u64,
}

#[derive(Serialize, Debug)]
pub struct IndexStatsResponse {
    pub total_documents: usize,
    pub index_size_bytes: u64,
    pub index_size_human: String,
    pub supported_languages: usize,
    pub last_updated: Option<String>,
    pub average_document_size: f64,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub search_engine_ready: bool,
    pub index_path: String,
}

// ============================================================================
// Handler Functions (real implementations, not simulations)
// ============================================================================

/// Main search handler - replaces JavaScript simulation entirely
async fn search_handler(
    State(state): State<ServerState>,
    Query(params): Query<SearchParams>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<LensError>)> {
    if params.q.trim().is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            LensError::search_with_query("Query cannot be empty", params.q.clone()),
        ));
    }

    if params.limit == 0 || params.limit > MAX_SEARCH_LIMIT {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            LensError::config_with_field(
                format!(
                    "Invalid limit parameter: must be between 1 and {}",
                    MAX_SEARCH_LIMIT
                ),
                "limit",
            ),
        ));
    }

    let limit = params.limit;
    let offset = params.offset.unwrap_or(0);
    if offset > MAX_SEARCH_OFFSET {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            LensError::config_with_field(
                format!(
                    "Invalid offset parameter: must be between 0 and {}",
                    MAX_SEARCH_OFFSET
                ),
                "offset",
            ),
        ));
    }

    let parsed_query =
        parse_full_query(&params.q).unwrap_or_else(|_| QueryBuilder::new(&params.q).build());
    let mut query_builder = QueryBuilder::new(&parsed_query.text);
    let core_query_text = parsed_query.text.clone();

    let query_type = if params.fuzzy {
        query_builder = query_builder.fuzzy();
        "fuzzy"
    } else if params.symbols {
        query_builder = query_builder.symbol();
        "symbol"
    } else {
        query_builder = match parsed_query.query_type {
            QueryType::Exact => query_builder.exact(),
            QueryType::Fuzzy => query_builder.fuzzy(),
            QueryType::Symbol => query_builder.symbol(),
            QueryType::Text => query_builder,
        };
        match parsed_query.query_type {
            QueryType::Exact => "exact",
            QueryType::Fuzzy => "fuzzy",
            QueryType::Symbol => "symbol",
            QueryType::Text => "text",
        }
    };

    query_builder = query_builder.limit(limit).offset(offset);

    if let Some(ref language) = params.language {
        if let Some(lang) = map_language_string(language) {
            query_builder = query_builder.language(lang);
        } else {
            warn!("Unknown language filter: {}", language);
        }
    } else if let Some(language) = parsed_query.language_filter.clone() {
        query_builder = query_builder.language(language);
    }

    if let Some(ref pattern) = params.file_pattern {
        query_builder = query_builder.file_pattern(pattern);
    } else if let Some(pattern) = parsed_query.file_filter.clone() {
        query_builder = query_builder.file_pattern(pattern);
    }

    let search_query = query_builder.build();

    // Execute the real search (not simulation)
    match state.search_engine.search(&search_query).await {
        Ok(results) => {
            let duration_ms = results.duration_ms();
            let matched_terms = extract_matched_terms(&core_query_text);
            let response_results: Vec<SearchResultResponse> = results
                .results
                .into_iter()
                .map(|result| {
                    let language = result.language.map(|l| l.to_string());
                    SearchResultResponse {
                        file_path: result.file_path,
                        line_number: result.line_number,
                        content: result.content,
                        score: result.score,
                        language,
                        matched_terms: matched_terms.clone(),
                        context_lines: result.context_lines,
                    }
                })
                .collect();

            // Get real index stats
            let stats = state.search_engine.get_stats().await.map_err(|e| {
                error!("Failed to get index stats: {}", e);
                api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    LensError::index_with_details("Failed to get index statistics", e.to_string()),
                )
            })?;

            Ok(Json(SearchResponse {
                query: params.q,
                total: results.total_matches,
                limit,
                offset,
                duration_ms,
                query_type: query_type.to_string(),
                from_cache: results.from_cache,
                results: response_results,
                index_stats: IndexStatsResponse {
                    total_documents: stats.total_documents,
                    index_size_bytes: stats.index_size_bytes,
                    index_size_human: stats.human_readable_size(),
                    supported_languages: stats.supported_languages,
                    last_updated: Some(format!("{:?}", stats.last_updated)),
                    average_document_size: stats.average_document_size(),
                },
            }))
        }
        Err(e) => {
            error!("Search failed: {}", e);
            Err(api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                LensError::search_with_query(
                    format!("Search execution failed: {}", e),
                    core_query_text.clone(),
                ),
            ))
        }
    }
}

/// Fuzzy search handler
async fn fuzzy_search_handler(
    State(state): State<ServerState>,
    Query(mut params): Query<SearchParams>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<LensError>)> {
    params.fuzzy = true;
    search_handler(State(state), Query(params)).await
}

/// Symbol search handler
async fn symbol_search_handler(
    State(state): State<ServerState>,
    Query(mut params): Query<SearchParams>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<LensError>)> {
    params.symbols = true;
    search_handler(State(state), Query(params)).await
}

/// Exact search handler
async fn exact_search_handler(
    State(state): State<ServerState>,
    Query(params): Query<SearchParams>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<LensError>)> {
    // For exact search, we use text search with no fuzzy matching
    search_handler(State(state), Query(params)).await
}

/// Index directory handler - real indexing, not simulation
async fn index_handler(
    State(state): State<ServerState>,
    Json(request): Json<IndexRequest>,
) -> Result<Json<IndexResponse>, (StatusCode, Json<LensError>)> {
    let directory = PathBuf::from(&request.directory);

    if !directory.exists() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            LensError::io_with_path("Directory does not exist", directory.display().to_string()),
        ));
    }

    info!("Indexing directory: {:?}", directory);

    if request.force {
        info!("Force reindex requested; clearing existing index first");
        if let Err(e) = state.search_engine.clear_index().await {
            error!("Failed to clear index before reindexing: {}", e);
            return Err(api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                LensError::index_with_details(
                    "Failed to clear index before reindexing",
                    e.to_string(),
                ),
            ));
        }
    }

    if !request.recursive {
        warn!("Recursive indexing disabled in request, but current implementation indexes recursively");
    }

    match state.search_engine.index_directory(&directory).await {
        Ok(stats) => Ok(Json(IndexResponse {
            success: true,
            message: "Directory indexed successfully".to_string(),
            files_indexed: stats.files_indexed,
            files_failed: stats.files_failed,
            lines_indexed: stats.lines_indexed,
            symbols_extracted: stats.symbols_extracted,
            duration_ms: stats.indexing_duration.as_millis() as u64,
        })),
        Err(e) => {
            error!("Indexing failed: {}", e);
            Err(api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                LensError::index_with_details("Indexing failed", e.to_string()),
            ))
        }
    }
}

/// Optimize index handler
async fn optimize_handler(
    State(state): State<ServerState>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<LensError>)> {
    match state.search_engine.optimize().await {
        Ok(_) => Ok(Json(serde_json::json!({
            "success": true,
            "message": "Index optimized successfully"
        }))),
        Err(e) => {
            error!("Optimization failed: {}", e);
            Err(api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                LensError::index_with_details("Index optimization failed", e.to_string()),
            ))
        }
    }
}

/// Clear index handler
async fn clear_handler(
    State(state): State<ServerState>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<LensError>)> {
    info!("Clear index endpoint called");

    match state.search_engine.clear_index().await {
        Ok(()) => {
            info!("Index cleared successfully");
            Ok(Json(serde_json::json!({
                "success": true,
                "message": "Index cleared successfully"
            })))
        }
        Err(e) => {
            error!("Failed to clear index: {}", e);
            Err(api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                LensError::index_with_details("Failed to clear index", e.to_string()),
            ))
        }
    }
}

/// Stats handler - real statistics, not simulation
async fn stats_handler(
    State(state): State<ServerState>,
) -> Result<Json<IndexStatsResponse>, (StatusCode, Json<LensError>)> {
    match state.search_engine.get_stats().await {
        Ok(stats) => Ok(Json(IndexStatsResponse {
            total_documents: stats.total_documents,
            index_size_bytes: stats.index_size_bytes,
            index_size_human: stats.human_readable_size(),
            supported_languages: stats.supported_languages,
            last_updated: Some(format!("{:?}", stats.last_updated)),
            average_document_size: stats.average_document_size(),
        })),
        Err(e) => {
            error!("Failed to get stats: {}", e);
            Err(api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                LensError::index_with_details("Failed to retrieve index statistics", e.to_string()),
            ))
        }
    }
}

/// Health check handler - real system status
async fn health_handler(State(state): State<ServerState>) -> Json<HealthResponse> {
    // Test if search engine is responsive
    let search_engine_ready = state.search_engine.get_stats().await.is_ok();

    Json(HealthResponse {
        status: if search_engine_ready {
            "healthy"
        } else {
            "degraded"
        }
        .to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: state.started_at.elapsed().as_secs(),
        search_engine_ready,
        index_path: state.index_path.display().to_string(),
    })
}

// ============================================================================
// Helper Functions
// ============================================================================
fn api_error(status: StatusCode, error: LensError) -> (StatusCode, Json<LensError>) {
    (status, Json(error))
}

/// Map language string to programming language enum
fn map_language_string(language: &str) -> Option<ProgrammingLanguage> {
    match language.to_lowercase().as_str() {
        "rust" | "rs" => Some(ProgrammingLanguage::Rust),
        "python" | "py" => Some(ProgrammingLanguage::Python),
        "typescript" | "ts" => Some(ProgrammingLanguage::TypeScript),
        "javascript" | "js" => Some(ProgrammingLanguage::JavaScript),
        "go" => Some(ProgrammingLanguage::Go),
        "java" => Some(ProgrammingLanguage::Java),
        "cpp" | "c++" => Some(ProgrammingLanguage::Cpp),
        "c" => Some(ProgrammingLanguage::C),
        _ => None,
    }
}

/// Extract matched terms from a query string
fn extract_matched_terms(query: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut current_term = String::new();
    let mut in_quotes = false;

    for char in query.chars() {
        match char {
            '"' => {
                in_quotes = !in_quotes;
            }
            ' ' if !in_quotes => {
                if !current_term.is_empty() {
                    terms.push(current_term.trim().to_string());
                    current_term.clear();
                }
            }
            _ => {
                current_term.push(char);
            }
        }
    }

    if !current_term.is_empty() {
        terms.push(current_term.trim().to_string());
    }

    terms
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::{Query, State};
    use lens_search_engine::SearchEngine;
    use std::sync::Arc;
    use tempfile::TempDir;

    #[test]
    fn test_language_mapping() {
        assert!(map_language_string("rust").is_some());
        assert!(map_language_string("python").is_some());
        assert!(map_language_string("typescript").is_some());
        assert!(map_language_string("invalid").is_none());
    }

    #[test]
    fn test_extract_matched_terms() {
        let terms = extract_matched_terms("hello world");
        assert_eq!(terms, vec!["hello", "world"]);

        let terms = extract_matched_terms("\"exact phrase\" single");
        assert_eq!(terms, vec!["exact phrase", "single"]);
    }

    #[test]
    fn test_default_limit() {
        assert_eq!(default_limit(), 10);
    }

    #[tokio::test]
    async fn test_search_handler_rejects_invalid_limit() {
        let index_dir = TempDir::new().unwrap();
        let engine = SearchEngine::new(index_dir.path()).await.unwrap();
        let state = ServerState {
            search_engine: Arc::new(engine),
            started_at: Instant::now(),
            index_path: index_dir.path().to_path_buf(),
        };

        let params = SearchParams {
            q: "foo".to_string(),
            limit: 0,
            offset: None,
            fuzzy: false,
            symbols: false,
            language: None,
            file_pattern: None,
        };

        let result = search_handler(State(state.clone()), Query(params)).await;
        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_search_handler_rejects_invalid_offset() {
        let index_dir = TempDir::new().unwrap();
        let engine = SearchEngine::new(index_dir.path()).await.unwrap();
        let state = ServerState {
            search_engine: Arc::new(engine),
            started_at: Instant::now(),
            index_path: index_dir.path().to_path_buf(),
        };

        let params = SearchParams {
            q: "foo".to_string(),
            limit: 10,
            offset: Some(MAX_SEARCH_OFFSET + 1),
            fuzzy: false,
            symbols: false,
            language: None,
            file_pattern: None,
        };

        let result = search_handler(State(state.clone()), Query(params)).await;
        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_search_handler_applies_limit_and_offset() {
        let index_dir = TempDir::new().unwrap();
        let data_dir = TempDir::new().unwrap();

        let files = ["alpha.rs", "beta.rs", "gamma.rs"];
        for (idx, name) in files.iter().enumerate() {
            let path = data_dir.path().join(name);
            std::fs::write(&path, format!("fn pagination_token_{}() {{}}", idx)).unwrap();
        }

        let engine = SearchEngine::new(index_dir.path()).await.unwrap();
        engine
            .index_directory(data_dir.path())
            .await
            .expect("indexing should succeed");

        let state = ServerState {
            search_engine: Arc::new(engine),
            started_at: Instant::now(),
            index_path: index_dir.path().to_path_buf(),
        };

        let params_first = SearchParams {
            q: "pagination_token".to_string(),
            limit: 1,
            offset: Some(0),
            fuzzy: false,
            symbols: false,
            language: None,
            file_pattern: None,
        };

        let Json(first_response) = search_handler(State(state.clone()), Query(params_first))
            .await
            .expect("first page should succeed");
        assert_eq!(first_response.limit, 1);
        assert_eq!(first_response.offset, 0);
        assert_eq!(first_response.results.len(), 1);
        let first_path = first_response.results[0].file_path.clone();

        let params_second = SearchParams {
            q: "pagination_token".to_string(),
            limit: 1,
            offset: Some(1),
            fuzzy: false,
            symbols: false,
            language: None,
            file_pattern: None,
        };

        let Json(second_response) = search_handler(State(state), Query(params_second))
            .await
            .expect("second page should succeed");
        assert_eq!(second_response.limit, 1);
        assert_eq!(second_response.offset, 1);
        assert_eq!(second_response.results.len(), 1);
        let second_path = second_response.results[0].file_path.clone();

        assert_ne!(
            first_path, second_path,
            "offset should advance to a different document"
        );
    }
}
