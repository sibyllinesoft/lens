//! HTTP Server Implementation for Lens Search
//!
//! This module provides a real HTTP API server that uses the Tantivy-based
//! search engine instead of simulations. It replaces the JavaScript simulation
//! layers with actual production functionality.

use anyhow::Result;
use axum::{
    body::Body,
    extract::{Query, State},
    http::{
        header::{HeaderName, CONTENT_TYPE},
        HeaderMap, Method, Request, StatusCode,
    },
    middleware::{from_fn, Next},
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use lens_common::{LensError, ProgrammingLanguage};
use lens_config::HttpAuthConfig;
use lens_search_engine::{
    parse_full_query, QueryBuilder, QueryType, SearchConfig as EngineSearchConfig, SearchEngine,
};
use prometheus::{
    exponential_buckets, Encoder, HistogramOpts, HistogramVec, IntCounterVec, IntGauge, Opts,
    Registry, TextEncoder,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};
use tower_http::{
    cors::CorsLayer,
    trace::{DefaultMakeSpan, DefaultOnResponse, TraceLayer},
};
use tracing::{error, info, warn, Level};

/// HTTP server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub bind: String,
    pub port: u16,
    pub enable_cors: bool,
    pub index_path: PathBuf,
    pub search_config: EngineSearchConfig,
    pub auth: HttpAuthConfig,
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
            auth: HttpAuthConfig::default(),
        }
    }
}

/// Server state containing the search engine
#[derive(Clone)]
pub struct ServerState {
    pub search_engine: Arc<SearchEngine>,
    pub started_at: Instant,
    pub index_path: PathBuf,
    pub metrics: Arc<ServerMetrics>,
}

pub struct ServerMetrics {
    registry: Registry,
    http_requests_total: IntCounterVec,
    http_request_duration_seconds: HistogramVec,
    search_queries_total: IntCounterVec,
    search_duration_seconds: HistogramVec,
    index_operations_total: IntCounterVec,
    index_operation_duration_seconds: HistogramVec,
    index_documents_total: IntGauge,
}

impl ServerMetrics {
    fn new() -> Result<Self> {
        let registry = Registry::new();

        let http_requests_total = IntCounterVec::new(
            Opts::new(
                "http_requests_total",
                "Total number of HTTP requests handled by the Lens HTTP server",
            ),
            &["method", "path", "status"],
        )?;
        registry.register(Box::new(http_requests_total.clone()))?;

        let http_duration_buckets = exponential_buckets(0.005, 2.0, 12)?;
        let http_request_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "http_request_duration_seconds",
                "Latency distribution for HTTP requests handled by the server",
            )
            .buckets(http_duration_buckets.clone()),
            &["method", "path"],
        )?;
        registry.register(Box::new(http_request_duration_seconds.clone()))?;

        let search_queries_total = IntCounterVec::new(
            Opts::new(
                "search_queries_total",
                "Count of search queries processed by the Lens search engine",
            ),
            &["outcome"],
        )?;
        registry.register(Box::new(search_queries_total.clone()))?;

        let search_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "search_queries_duration_seconds",
                "Latency distribution for search queries processed by the Lens search engine",
            )
            .buckets(http_duration_buckets.clone()),
            &["outcome"],
        )?;
        registry.register(Box::new(search_duration_seconds.clone()))?;

        let index_operations_total = IntCounterVec::new(
            Opts::new(
                "index_operations_total",
                "Count of index maintenance operations executed by the Lens server",
            ),
            &["operation", "outcome"],
        )?;
        registry.register(Box::new(index_operations_total.clone()))?;

        let index_operation_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "index_operation_duration_seconds",
                "Latency distribution for index operations (index, clear, optimize)",
            )
            .buckets(http_duration_buckets),
            &["operation", "outcome"],
        )?;
        registry.register(Box::new(index_operation_duration_seconds.clone()))?;

        let index_documents_total = IntGauge::new(
            "index_documents_total",
            "Current number of documents tracked in the Lens search index",
        )?;
        registry.register(Box::new(index_documents_total.clone()))?;

        Ok(Self {
            registry,
            http_requests_total,
            http_request_duration_seconds,
            search_queries_total,
            search_duration_seconds,
            index_operations_total,
            index_operation_duration_seconds,
            index_documents_total,
        })
    }

    fn observe_http(&self, method: &Method, path: &str, status: StatusCode, duration: Duration) {
        let status_label = status.as_u16().to_string();
        let method_label = method.as_str();
        let path_label = sanitize_path(path);

        self.http_requests_total
            .with_label_values(&[method_label, path_label.as_str(), status_label.as_str()])
            .inc();
        self.http_request_duration_seconds
            .with_label_values(&[method_label, path_label.as_str()])
            .observe(duration.as_secs_f64());
    }

    fn observe_search(&self, outcome: &str, duration: Duration) {
        self.search_queries_total
            .with_label_values(&[outcome])
            .inc();
        self.search_duration_seconds
            .with_label_values(&[outcome])
            .observe(duration.as_secs_f64());
    }

    fn record_index_operation(&self, operation: &str, outcome: &str, duration: Duration) {
        self.index_operations_total
            .with_label_values(&[operation, outcome])
            .inc();
        self.index_operation_duration_seconds
            .with_label_values(&[operation, outcome])
            .observe(duration.as_secs_f64());
    }

    fn set_index_documents(&self, count: usize) {
        self.index_documents_total.set(count as i64);
    }

    fn gather(&self) -> Result<String> {
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        TextEncoder::default().encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }
}

fn sanitize_path(path: &str) -> String {
    let base = path.split('?').next().unwrap_or("/").trim();
    if base.is_empty() || base == "/" {
        return "/".to_string();
    }

    let normalized = base.trim_end_matches('/');
    if normalized.starts_with('/') {
        normalized.to_string()
    } else {
        format!("/{}", normalized)
    }
}

#[derive(Clone, Debug)]
struct AuthState {
    header_name: HeaderName,
    bearer_prefix: Option<String>,
    tokens: HashSet<String>,
}

impl AuthState {
    fn is_authorized(&self, headers: &HeaderMap) -> bool {
        let raw_value = match headers.get(&self.header_name) {
            Some(value) => match value.to_str() {
                Ok(v) => v.trim(),
                Err(_) => return false,
            },
            None => return false,
        };

        let candidate = if let Some(prefix) = &self.bearer_prefix {
            raw_value
                .strip_prefix(prefix)
                .map(str::trim)
                .unwrap_or(raw_value)
        } else {
            raw_value
        };

        self.tokens.contains(candidate)
    }
}

fn build_auth_state(config: HttpAuthConfig) -> Result<Option<Arc<AuthState>>> {
    if !config.enabled {
        return Ok(None);
    }

    let HttpAuthConfig {
        enabled: _,
        header_name,
        bearer_prefix,
        tokens,
    } = config;

    let header_name_str = header_name.trim().to_string();
    let header_name = header_name_str.parse::<HeaderName>().map_err(|err| {
        anyhow::anyhow!(
            "Invalid http.auth.header_name '{}': {}",
            header_name_str,
            err
        )
    })?;

    let normalized_tokens: HashSet<String> = tokens
        .into_iter()
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
        .collect();

    if normalized_tokens.is_empty() {
        anyhow::bail!("HTTP authentication is enabled but no tokens are configured");
    }

    let normalized_prefix = bearer_prefix.and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    });

    Ok(Some(Arc::new(AuthState {
        header_name,
        bearer_prefix: normalized_prefix,
        tokens: normalized_tokens,
    })))
}

/// Start the HTTP server with real search functionality
pub async fn start_server(config: ServerConfig) -> Result<()> {
    let ServerConfig {
        bind,
        port,
        enable_cors,
        index_path,
        search_config,
        auth,
    } = config;

    info!(
        "Starting Lens HTTP server at {}:{} with index at: {:?}",
        bind, port, index_path
    );

    let search_engine = Arc::new(SearchEngine::with_config(search_config).await?);
    let started_at = Instant::now();
    let metrics = Arc::new(ServerMetrics::new()?);

    let auth_state = build_auth_state(auth)?;
    if let Some(auth) = auth_state.as_ref() {
        info!(header = %auth.header_name, "HTTP authentication enabled");
    } else {
        info!("HTTP authentication disabled");
    }

    // Create server state
    let state = ServerState {
        search_engine,
        started_at,
        index_path: index_path.clone(),
        metrics: metrics.clone(),
    };

    // Create the router with real endpoints
    let app = create_router(state, enable_cors, auth_state, metrics);

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
fn create_router(
    state: ServerState,
    enable_cors: bool,
    auth_state: Option<Arc<AuthState>>,
    metrics: Arc<ServerMetrics>,
) -> Router {
    let mut app = Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/api/metrics", get(metrics_handler))
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

    let trace_layer = TraceLayer::new_for_http()
        .make_span_with(DefaultMakeSpan::new().level(Level::INFO))
        .on_response(DefaultOnResponse::new().level(Level::INFO));

    app = app.layer(trace_layer);

    // Add CORS if enabled
    if enable_cors {
        app = app.layer(CorsLayer::permissive());
    }

    if let Some(auth) = auth_state {
        let auth_for_layer = auth.clone();
        app = app.layer(from_fn(move |request, next| {
            let auth = auth_for_layer.clone();
            async move { auth_middleware(auth, request, next).await }
        }));
    }

    let metrics_for_layer = metrics.clone();
    app = app.layer(from_fn(move |request, next| {
        let metrics = metrics_for_layer.clone();
        async move { metrics_middleware(metrics, request, next).await }
    }));

    app
}

async fn metrics_middleware(
    metrics: Arc<ServerMetrics>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let start = Instant::now();

    let response = next.run(request).await;
    let status = response.status();

    metrics.observe_http(&method, &path, status, start.elapsed());

    response
}

async fn auth_middleware(
    auth_state: Arc<AuthState>,
    request: Request<Body>,
    next: Next,
) -> std::result::Result<Response, StatusCode> {
    let method = request.method().clone();
    let uri = request.uri().clone();

    if method == Method::OPTIONS {
        return Ok(next.run(request).await);
    }

    if !auth_state.is_authorized(request.headers()) {
        warn!(method = %method, path = %uri, "Rejected unauthorized request");
        return Err(StatusCode::UNAUTHORIZED);
    }

    Ok(next.run(request).await)
}

async fn metrics_handler(State(state): State<ServerState>) -> Response {
    match state.metrics.gather() {
        Ok(payload) => (
            [(CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
            payload,
        )
            .into_response(),
        Err(error) => {
            error!("Failed to gather Prometheus metrics: {}", error);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to collect metrics",
            )
                .into_response()
        }
    }
}

// ============================================================================
// Request/Response Types (replacing simulation types)
// ============================================================================

#[derive(Deserialize, Debug)]
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
#[tracing::instrument(
    name = "http.search",
    skip(state, params),
    fields(
        query = %params.q,
        limit = params.limit,
        offset = params.offset.unwrap_or(0),
        fuzzy = params.fuzzy,
        symbols = params.symbols
    )
)]
async fn search_handler(
    State(state): State<ServerState>,
    Query(params): Query<SearchParams>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<LensError>)> {
    let metrics = state.metrics.clone();
    let start = Instant::now();

    if params.q.trim().is_empty() {
        metrics.observe_search("client_error", start.elapsed());
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            LensError::search_with_query("Query cannot be empty", params.q.clone()),
        ));
    }

    if params.limit == 0 || params.limit > MAX_SEARCH_LIMIT {
        metrics.observe_search("client_error", start.elapsed());
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
        metrics.observe_search("client_error", start.elapsed());
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
                metrics.observe_search("server_error", start.elapsed());
                api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    LensError::index_with_details("Failed to get index statistics", e.to_string()),
                )
            })?;

            metrics.set_index_documents(stats.total_documents);
            metrics.observe_search("success", start.elapsed());

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
            metrics.observe_search("server_error", start.elapsed());
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
#[tracing::instrument(name = "http.search.fuzzy", skip(state, params))]
async fn fuzzy_search_handler(
    State(state): State<ServerState>,
    Query(mut params): Query<SearchParams>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<LensError>)> {
    params.fuzzy = true;
    search_handler(State(state), Query(params)).await
}

/// Symbol search handler
#[tracing::instrument(name = "http.search.symbol", skip(state, params))]
async fn symbol_search_handler(
    State(state): State<ServerState>,
    Query(mut params): Query<SearchParams>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<LensError>)> {
    params.symbols = true;
    search_handler(State(state), Query(params)).await
}

/// Exact search handler
#[tracing::instrument(name = "http.search.exact", skip(state, params))]
async fn exact_search_handler(
    State(state): State<ServerState>,
    Query(params): Query<SearchParams>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<LensError>)> {
    // For exact search, we use text search with no fuzzy matching
    search_handler(State(state), Query(params)).await
}

/// Index directory handler - real indexing, not simulation
#[tracing::instrument(
    name = "http.index",
    skip(state, request),
    fields(directory = %request.directory, force = request.force, recursive = request.recursive)
)]
async fn index_handler(
    State(state): State<ServerState>,
    Json(request): Json<IndexRequest>,
) -> Result<Json<IndexResponse>, (StatusCode, Json<LensError>)> {
    let metrics = state.metrics.clone();
    let start = Instant::now();
    let directory = PathBuf::from(&request.directory);

    if !directory.exists() {
        metrics.record_index_operation("bulk_index", "client_error", start.elapsed());
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
            metrics.record_index_operation("bulk_index", "error", start.elapsed());
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
        Ok(stats) => {
            if let Ok(latest) = state.search_engine.get_stats().await {
                metrics.set_index_documents(latest.total_documents);
            }
            metrics.record_index_operation("bulk_index", "success", start.elapsed());

            Ok(Json(IndexResponse {
                success: true,
                message: "Directory indexed successfully".to_string(),
                files_indexed: stats.files_indexed,
                files_failed: stats.files_failed,
                lines_indexed: stats.lines_indexed,
                symbols_extracted: stats.symbols_extracted,
                duration_ms: stats.indexing_duration.as_millis() as u64,
            }))
        }
        Err(e) => {
            error!("Indexing failed: {}", e);
            metrics.record_index_operation("bulk_index", "error", start.elapsed());
            Err(api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                LensError::index_with_details("Indexing failed", e.to_string()),
            ))
        }
    }
}

/// Optimize index handler
#[tracing::instrument(name = "http.optimize", skip(state))]
async fn optimize_handler(
    State(state): State<ServerState>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<LensError>)> {
    let metrics = state.metrics.clone();
    let start = Instant::now();

    match state.search_engine.optimize().await {
        Ok(_) => {
            metrics.record_index_operation("optimize", "success", start.elapsed());
            Ok(Json(serde_json::json!({
                "success": true,
                "message": "Index optimized successfully"
            })))
        }
        Err(e) => {
            error!("Optimization failed: {}", e);
            metrics.record_index_operation("optimize", "error", start.elapsed());
            Err(api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                LensError::index_with_details("Index optimization failed", e.to_string()),
            ))
        }
    }
}

/// Clear index handler
#[tracing::instrument(name = "http.clear", skip(state))]
async fn clear_handler(
    State(state): State<ServerState>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<LensError>)> {
    info!("Clear index endpoint called");

    let metrics = state.metrics.clone();
    let start = Instant::now();

    match state.search_engine.clear_index().await {
        Ok(()) => {
            info!("Index cleared successfully");
            metrics.set_index_documents(0);
            metrics.record_index_operation("clear", "success", start.elapsed());
            Ok(Json(serde_json::json!({
                "success": true,
                "message": "Index cleared successfully"
            })))
        }
        Err(e) => {
            error!("Failed to clear index: {}", e);
            metrics.record_index_operation("clear", "error", start.elapsed());
            Err(api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                LensError::index_with_details("Failed to clear index", e.to_string()),
            ))
        }
    }
}

/// Stats handler - real statistics, not simulation
#[tracing::instrument(name = "http.stats", skip(state))]
async fn stats_handler(
    State(state): State<ServerState>,
) -> Result<Json<IndexStatsResponse>, (StatusCode, Json<LensError>)> {
    match state.search_engine.get_stats().await {
        Ok(stats) => {
            state.metrics.set_index_documents(stats.total_documents);
            Ok(Json(IndexStatsResponse {
                total_documents: stats.total_documents,
                index_size_bytes: stats.index_size_bytes,
                index_size_human: stats.human_readable_size(),
                supported_languages: stats.supported_languages,
                last_updated: Some(format!("{:?}", stats.last_updated)),
                average_document_size: stats.average_document_size(),
            }))
        }
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
#[tracing::instrument(name = "http.health", skip(state))]
async fn health_handler(State(state): State<ServerState>) -> Json<HealthResponse> {
    // Test if search engine is responsive
    let stats_result = state.search_engine.get_stats().await;
    let search_engine_ready = stats_result.is_ok();

    if let Ok(stats) = stats_result {
        state.metrics.set_index_documents(stats.total_documents);
    }

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
        let metrics = Arc::new(ServerMetrics::new().unwrap());
        let state = ServerState {
            search_engine: Arc::new(engine),
            started_at: Instant::now(),
            index_path: index_dir.path().to_path_buf(),
            metrics,
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
        let metrics = Arc::new(ServerMetrics::new().unwrap());
        let state = ServerState {
            search_engine: Arc::new(engine),
            started_at: Instant::now(),
            index_path: index_dir.path().to_path_buf(),
            metrics,
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

        let metrics = Arc::new(ServerMetrics::new().unwrap());
        let state = ServerState {
            search_engine: Arc::new(engine),
            started_at: Instant::now(),
            index_path: index_dir.path().to_path_buf(),
            metrics,
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
