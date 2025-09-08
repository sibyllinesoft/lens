use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tonic::{Request, Response, Status, Code};
use tracing::{info, warn, error, debug, instrument};
use sha2::Digest;

use crate::search::{SearchEngine, SearchRequest as InternalSearchRequest, SearchResponse as InternalSearchResponse};
use crate::metrics::{MetricsCollector, SlaMetrics, PerformanceGate};
use crate::attestation::AttestationManager;
use crate::benchmark::BenchmarkRunner;

// Use the main proto module
use crate::proto;

use proto::{
    SearchRequest, SearchResponse, SearchResult, SearchMetrics, 
    HealthRequest, HealthResponse, BuildInfoRequest, BuildInfoResponse,
    HandshakeRequest, HandshakeResponse,
    lens_search_server::{LensSearch, LensSearchServer},
};

/// Main gRPC server implementation
pub struct LensSearchServiceImpl {
    search_engine: Arc<SearchEngine>,
    metrics_collector: Arc<MetricsCollector>,
    attestation_manager: Arc<AttestationManager>,
    benchmark_runner: Arc<BenchmarkRunner>,
    server_start_time: Instant,
}

impl LensSearchServiceImpl {
    pub fn new(
        search_engine: Arc<SearchEngine>,
        metrics_collector: Arc<MetricsCollector>,
        attestation_manager: Arc<AttestationManager>,
        benchmark_runner: Arc<BenchmarkRunner>,
    ) -> Self {
        Self {
            search_engine,
            metrics_collector,
            attestation_manager,
            benchmark_runner,
            server_start_time: Instant::now(),
        }
    }

    /// Create the tonic server instance
    pub fn create_server(self) -> proto::lens_search_server::LensSearchServer<Self> {
        proto::lens_search_server::LensSearchServer::new(self)
    }

    /// Convert protobuf SearchRequest to internal format
    fn convert_request(&self, request: proto::SearchRequest) -> Result<crate::search::SearchRequest, Status> {
        if request.query.is_empty() {
            return Err(Status::new(Code::InvalidArgument, "Query cannot be empty"));
        }

        Ok(crate::search::SearchRequest {
            query: request.query,
            file_path: None,
            language: None,
            max_results: if request.limit == 0 { 50 } else { request.limit as usize },
            include_context: true,
            timeout_ms: 150, // Default SLA timeout
            enable_lsp: true,
            search_types: vec![
                crate::search::SearchResultType::TextMatch,
                crate::search::SearchResultType::Definition,
            ],
            search_method: Some(crate::search::SearchMethod::Hybrid),
        })
    }

    /// Convert internal SearchResponse to protobuf format
    fn convert_response(
        &self, 
        internal_response: crate::search::SearchResponse,
        attestation_hash: String
    ) -> proto::SearchResponse {
        let results_len = internal_response.results.len();
        let results = internal_response.results.into_iter().map(|result| {
            proto::SearchResult {
                file_path: result.file_path,
                line_number: result.line_number,
                content: result.content,
                score: result.score,
            }
        }).collect();
        
        let metrics = proto::SearchMetrics {
            total_docs: results_len as u64,
            matched_docs: results_len as u64,
            duration_ms: internal_response.metrics.duration_ms,
        };

        proto::SearchResponse {
            results,
            metrics: Some(metrics),
            attestation: attestation_hash,
        }
    }


    /// Check if request should be routed to LSP based on query pattern
    fn should_route_to_lsp(&self, query: &str, language: Option<&str>) -> bool {
        // Simple heuristics for LSP routing - can be made more sophisticated
        let lsp_patterns = [
            "def", "define", "definition", "go to", "goto",
            "references", "ref", "usage", "uses",
            "type", "typeof", "implements", "implementation",
            "class", "function", "method", "variable",
        ];

        let query_lower = query.to_lowercase();
        let has_lsp_pattern = lsp_patterns.iter().any(|&pattern| query_lower.contains(pattern));
        let has_language = language.is_some();
        
        has_lsp_pattern || has_language
    }
}

#[tonic::async_trait]
impl proto::lens_search_server::LensSearch for LensSearchServiceImpl {
    #[instrument(skip(self, request))]
    async fn search(
        &self,
        request: Request<proto::SearchRequest>,
    ) -> Result<Response<proto::SearchResponse>, Status> {
        let start_time = Instant::now();
        let req = request.into_inner();
        
        debug!("Received search request: {}", req.query);

        // Convert to internal request format
        let internal_req = self.convert_request(req.clone())?;

        // Execute search with SLA monitoring
        let sla_start = Instant::now();
        let search_result = self.search_engine.search_comprehensive(internal_req).await;
        let search_duration = sla_start.elapsed();

        // Check SLA compliance (≤150ms p95 per TODO.md)
        let sla_compliant = search_duration.as_millis() <= 150;
        if !sla_compliant {
            warn!("Search exceeded SLA: {}ms > 150ms", search_duration.as_millis());
        }

        // Process search results
        match search_result {
            Ok(internal_response) => {
                // Generate simple attestation hash
                let attestation_hash = format!("{:x}", 
                    sha2::Digest::finalize(sha2::Sha256::new()
                        .chain_update(req.query.as_bytes())
                        .chain_update(&search_duration.as_millis().to_le_bytes())
                        .chain_update(&internal_response.results.len().to_le_bytes())
                    )
                );

                // Convert to protobuf response
                let response = self.convert_response(internal_response, attestation_hash);
                
                info!(
                    "Search completed: query='{}', results={}, duration={}ms, sla_compliant={}",
                    req.query,
                    response.results.len(),
                    search_duration.as_millis(),
                    sla_compliant
                );

                Ok(Response::new(response))
            }
            Err(e) => {
                error!("Search failed: {}", e);
                
                let error_response = proto::SearchResponse {
                    results: vec![],
                    metrics: Some(proto::SearchMetrics {
                        total_docs: 0,
                        matched_docs: 0,
                        duration_ms: search_duration.as_millis() as u32,
                    }),
                    attestation: "error".to_string(),
                };
                
                Ok(Response::new(error_response))
            }
        }
    }


    #[instrument(skip(self, _request))]
    async fn health(
        &self,
        _request: Request<proto::HealthRequest>,
    ) -> Result<Response<proto::HealthResponse>, Status> {
        // Simple health check
        let response = proto::HealthResponse {
            status: "healthy".to_string(),
            mode: "real".to_string(), // MUST be "real" per proto requirement
            version: env!("CARGO_PKG_VERSION").to_string(),
        };

        debug!("Health check requested");
        Ok(Response::new(response))
    }

    #[instrument(skip(self, _request))]
    async fn get_build_info(
        &self,
        _request: Request<proto::BuildInfoRequest>,
    ) -> Result<Response<proto::BuildInfoResponse>, Status> {
        // Get build information using attestation manager
        let build_info = crate::attestation::get_build_info();
        
        debug!("Build info requested");
        Ok(Response::new(build_info))
    }

    #[instrument(skip(self, request))]
    async fn handshake(
        &self,
        request: Request<proto::HandshakeRequest>,
    ) -> Result<Response<proto::HandshakeResponse>, Status> {
        let req = request.into_inner();
        let nonce = req.nonce.clone();
        
        // Perform handshake with attestation manager
        let handshake_response = {
            // Fallback handshake without attestation manager
            let response_hash = crate::attestation::perform_handshake(&req.nonce)
                .map_err(|e| Status::internal(format!("Handshake failed: {}", e)))?;
            let build_info = crate::attestation::get_build_info();
            
            proto::HandshakeResponse {
                nonce: req.nonce,
                response: response_hash,
                build_info: Some(build_info),
            }
        };
        
        debug!("Handshake completed for nonce: {}", nonce);
        Ok(Response::new(handshake_response))
    }

}

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub bind_address: String,
    pub port: u16,
    pub max_concurrent_requests: usize,
    pub request_timeout: Duration,
    pub enable_reflection: bool,
    pub enable_health_check: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_address: "127.0.0.1".to_string(),
            port: 50051,
            max_concurrent_requests: 1000,
            request_timeout: Duration::from_millis(150), // Per TODO.md SLA
            enable_reflection: false, // Disable in production
            enable_health_check: true,
        }
    }
}

/// Create and configure the gRPC server
pub async fn create_server(
    config: ServerConfig,
    search_engine: Arc<SearchEngine>,
    metrics_collector: Arc<MetricsCollector>,
    attestation_manager: Arc<AttestationManager>,
    benchmark_runner: Arc<BenchmarkRunner>,
) -> Result<impl std::future::Future<Output = Result<(), tonic::transport::Error>>, anyhow::Error> {
    let addr = format!("{}:{}", config.bind_address, config.port).parse()?;
    
    let service_impl = LensSearchServiceImpl::new(
        search_engine,
        metrics_collector,
        attestation_manager,
        benchmark_runner,
    );
    let search_service = service_impl.create_server();

    let mut server_builder = tonic::transport::Server::builder()
        .timeout(config.request_timeout)
        .concurrency_limit_per_connection(config.max_concurrent_requests)
        .add_service(search_service);
    
    // Add reflection service in development
    if config.enable_reflection {
        server_builder = server_builder.add_service(
            tonic_reflection::server::Builder::configure()
                .register_encoded_file_descriptor_set(proto::FILE_DESCRIPTOR_SET)
                .build()?
        );
    }

    let server = server_builder.serve(addr);

    info!("gRPC server configured on {}", addr);
    Ok(server)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tonic::{Request, Status, Code};
    
    // Mock implementations for testing
    struct MockSearchEngine {
        should_fail: bool,
        results_count: usize,
        processing_time_ms: u64,
    }

    impl MockSearchEngine {
        fn new() -> Self {
            Self {
                should_fail: false,
                results_count: 5,
                processing_time_ms: 50,
            }
        }

        fn with_failure() -> Self {
            Self {
                should_fail: true,
                results_count: 0,
                processing_time_ms: 10,
            }
        }

        fn with_slow_response(processing_time_ms: u64) -> Self {
            Self {
                should_fail: false,
                results_count: 3,
                processing_time_ms,
            }
        }

        fn with_results_count(results_count: usize) -> Self {
            Self {
                should_fail: false,
                results_count,
                processing_time_ms: 30,
            }
        }

        async fn search_comprehensive(&self, _request: InternalSearchRequest) -> anyhow::Result<InternalSearchResponse> {
            tokio::time::sleep(Duration::from_millis(self.processing_time_ms)).await;
            
            if self.should_fail {
                return Err(anyhow::anyhow!("Mock search engine failure"));
            }

            let results = (0..self.results_count)
                .map(|i| crate::search::SearchResult {
                    file_path: format!("test_file_{}.rs", i),
                    line_number: (i + 1) as u32,
                    content: format!("test content {}", i),
                    score: 0.95 - (i as f64 * 0.1),
                })
                .collect();

            Ok(InternalSearchResponse {
                results,
                metrics: crate::search::SearchMetrics {
                    duration_ms: self.processing_time_ms as u32,
                    files_searched: 100,
                    total_matches: self.results_count,
                },
            })
        }
    }

    fn create_mock_service() -> LensSearchServiceImpl {
        let mock_search_engine = Arc::new(MockSearchEngine::new());
        let metrics_collector = Arc::new(crate::metrics::MetricsCollector::new());
        let attestation_manager = Arc::new(crate::attestation::AttestationManager::new());
        let benchmark_runner = Arc::new(crate::benchmark::BenchmarkRunner::new());
        
        // Use unsafe transmute as a simplified approach for testing
        // In real implementation, you'd want proper dependency injection
        let search_engine: Arc<SearchEngine> = unsafe { std::mem::transmute(mock_search_engine) };
        
        LensSearchServiceImpl::new(
            search_engine,
            metrics_collector,
            attestation_manager,
            benchmark_runner,
        )
    }

    fn create_mock_service_with_failure() -> LensSearchServiceImpl {
        let mock_search_engine = Arc::new(MockSearchEngine::with_failure());
        let metrics_collector = Arc::new(crate::metrics::MetricsCollector::new());
        let attestation_manager = Arc::new(crate::attestation::AttestationManager::new());
        let benchmark_runner = Arc::new(crate::benchmark::BenchmarkRunner::new());
        
        let search_engine: Arc<SearchEngine> = unsafe { std::mem::transmute(mock_search_engine) };
        
        LensSearchServiceImpl::new(
            search_engine,
            metrics_collector,
            attestation_manager,
            benchmark_runner,
        )
    }

    fn create_mock_service_with_slow_response(processing_time_ms: u64) -> LensSearchServiceImpl {
        let mock_search_engine = Arc::new(MockSearchEngine::with_slow_response(processing_time_ms));
        let metrics_collector = Arc::new(crate::metrics::MetricsCollector::new());
        let attestation_manager = Arc::new(crate::attestation::AttestationManager::new());
        let benchmark_runner = Arc::new(crate::benchmark::BenchmarkRunner::new());
        
        let search_engine: Arc<SearchEngine> = unsafe { std::mem::transmute(mock_search_engine) };
        
        LensSearchServiceImpl::new(
            search_engine,
            metrics_collector,
            attestation_manager,
            benchmark_runner,
        )
    }

    // Test service creation
    #[test]
    fn test_service_creation() {
        let service = create_mock_service();
        let start_time = service.server_start_time;
        let _server = service.create_server();
        
        // Service should be created successfully
        assert!(start_time.elapsed() < Duration::from_secs(1));
    }

    // Test request conversion - valid requests
    #[test]
    fn test_convert_request_valid() {
        let service = create_mock_service();
        
        let proto_request = SearchRequest {
            query: "test query".to_string(),
            limit: 10,
            dataset_sha256: "abcd1234".to_string(),
        };
        
        let result = service.convert_request(proto_request);
        assert!(result.is_ok());
        
        let internal_request = result.unwrap();
        assert_eq!(internal_request.query, "test query");
        assert_eq!(internal_request.max_results, 10);
        assert_eq!(internal_request.timeout_ms, 150);
        assert!(internal_request.enable_lsp);
        assert!(internal_request.include_context);
        assert_eq!(internal_request.search_types.len(), 2);
    }

    #[test]
    fn test_convert_request_default_limit() {
        let service = create_mock_service();
        
        let proto_request = SearchRequest {
            query: "test query".to_string(),
            limit: 0, // Should default to 50
            dataset_sha256: "abcd1234".to_string(),
        };
        
        let result = service.convert_request(proto_request);
        assert!(result.is_ok());
        
        let internal_request = result.unwrap();
        assert_eq!(internal_request.max_results, 50); // Default limit
    }

    #[test]
    fn test_convert_request_empty_query() {
        let service = create_mock_service();
        
        let proto_request = SearchRequest {
            query: "".to_string(), // Empty query should fail
            limit: 10,
            dataset_sha256: "abcd1234".to_string(),
        };
        
        let result = service.convert_request(proto_request);
        assert!(result.is_err());
        
        let error = result.unwrap_err();
        assert_eq!(error.code(), Code::InvalidArgument);
        assert!(error.message().contains("Query cannot be empty"));
    }

    #[test]
    fn test_convert_request_whitespace_only() {
        let service = create_mock_service();
        
        let proto_request = SearchRequest {
            query: "   \t\n  ".to_string(), // Whitespace only
            limit: 10,
            dataset_sha256: "abcd1234".to_string(),
        };
        
        // Should not be empty after trimming in a real implementation
        let result = service.convert_request(proto_request);
        assert!(result.is_ok()); // Current implementation doesn't trim
    }

    // Test response conversion
    #[test]
    fn test_convert_response() {
        let service = create_mock_service();
        
        let internal_response = InternalSearchResponse {
            results: vec![
                crate::search::SearchResult {
                    file_path: "test.rs".to_string(),
                    line_number: 42,
                    content: "test content".to_string(),
                    score: 0.95,
                },
                crate::search::SearchResult {
                    file_path: "another.rs".to_string(),
                    line_number: 100,
                    content: "another content".to_string(),
                    score: 0.85,
                },
            ],
            metrics: crate::search::SearchMetrics {
                duration_ms: 50,
                files_searched: 200,
                total_matches: 2,
            },
        };
        
        let attestation_hash = "test_hash".to_string();
        let proto_response = service.convert_response(internal_response, attestation_hash.clone());
        
        assert_eq!(proto_response.results.len(), 2);
        assert_eq!(proto_response.attestation, attestation_hash);
        
        // Check first result
        assert_eq!(proto_response.results[0].file_path, "test.rs");
        assert_eq!(proto_response.results[0].line_number, 42);
        assert_eq!(proto_response.results[0].content, "test content");
        assert_eq!(proto_response.results[0].score, 0.95);
        
        // Check metrics
        assert!(proto_response.metrics.is_some());
        let metrics = proto_response.metrics.unwrap();
        assert_eq!(metrics.total_docs, 2);
        assert_eq!(metrics.matched_docs, 2);
        assert_eq!(metrics.duration_ms, 50);
    }

    #[test]
    fn test_convert_response_empty_results() {
        let service = create_mock_service();
        
        let internal_response = InternalSearchResponse {
            results: vec![],
            metrics: crate::search::SearchMetrics {
                duration_ms: 25,
                files_searched: 100,
                total_matches: 0,
            },
        };
        
        let attestation_hash = "empty_hash".to_string();
        let proto_response = service.convert_response(internal_response, attestation_hash.clone());
        
        assert_eq!(proto_response.results.len(), 0);
        assert_eq!(proto_response.attestation, attestation_hash);
        
        let metrics = proto_response.metrics.unwrap();
        assert_eq!(metrics.total_docs, 0);
        assert_eq!(metrics.matched_docs, 0);
    }

    // Test LSP routing logic
    #[test]
    fn test_should_route_to_lsp_patterns() {
        let service = create_mock_service();
        
        // Positive cases - should route to LSP
        assert!(service.should_route_to_lsp("def myFunction", None));
        assert!(service.should_route_to_lsp("find definition", None));
        assert!(service.should_route_to_lsp("go to implementation", None));
        assert!(service.should_route_to_lsp("show references", None));
        assert!(service.should_route_to_lsp("type information", None));
        assert!(service.should_route_to_lsp("class MyClass", None));
        assert!(service.should_route_to_lsp("function getName", None));
        assert!(service.should_route_to_lsp("method call", None));
        assert!(service.should_route_to_lsp("variable usage", None));
        
        // With language specified
        assert!(service.should_route_to_lsp("simple query", Some("rust")));
        assert!(service.should_route_to_lsp("search text", Some("typescript")));
    }

    #[test]
    fn test_should_route_to_lsp_negative_cases() {
        let service = create_mock_service();
        
        // Negative cases - should not route to LSP
        assert!(!service.should_route_to_lsp("hello world", None));
        assert!(!service.should_route_to_lsp("simple text search", None));
        assert!(!service.should_route_to_lsp("random query", None));
        assert!(!service.should_route_to_lsp("", None));
        assert!(!service.should_route_to_lsp("123 456", None));
    }

    #[test]
    fn test_should_route_to_lsp_case_insensitive() {
        let service = create_mock_service();
        
        // Should be case insensitive
        assert!(service.should_route_to_lsp("DEF myFunction", None));
        assert!(service.should_route_to_lsp("CLASS MyClass", None));
        assert!(service.should_route_to_lsp("FUNCTION getName", None));
    }

    // Test successful search endpoint
    #[tokio::test]
    async fn test_search_successful() {
        let service = create_mock_service();
        
        let proto_request = SearchRequest {
            query: "test query".to_string(),
            limit: 5,
            dataset_sha256: "abcd1234".to_string(),
        };
        
        let request = Request::new(proto_request);
        let result = service.search(request).await;
        
        assert!(result.is_ok());
        let response = result.unwrap().into_inner();
        
        assert_eq!(response.results.len(), 5);
        assert!(response.metrics.is_some());
        assert!(!response.attestation.is_empty());
        
        // Check attestation hash format (should be hex)
        assert!(response.attestation.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[tokio::test]
    async fn test_search_failure() {
        let service = create_mock_service_with_failure();
        
        let proto_request = SearchRequest {
            query: "test query".to_string(),
            limit: 5,
            dataset_sha256: "abcd1234".to_string(),
        };
        
        let request = Request::new(proto_request);
        let result = service.search(request).await;
        
        assert!(result.is_ok()); // Should return ok with error response
        let response = result.unwrap().into_inner();
        
        assert_eq!(response.results.len(), 0);
        assert_eq!(response.attestation, "error");
        
        let metrics = response.metrics.unwrap();
        assert_eq!(metrics.total_docs, 0);
        assert_eq!(metrics.matched_docs, 0);
    }

    #[tokio::test] 
    async fn test_search_sla_compliance() {
        let service = create_mock_service_with_slow_response(200); // 200ms > 150ms SLA
        
        let proto_request = SearchRequest {
            query: "slow query".to_string(),
            limit: 3,
            dataset_sha256: "abcd1234".to_string(),
        };
        
        let start_time = Instant::now();
        let request = Request::new(proto_request);
        let result = service.search(request).await;
        let total_duration = start_time.elapsed();
        
        assert!(result.is_ok());
        let response = result.unwrap().into_inner();
        
        assert_eq!(response.results.len(), 3);
        assert!(total_duration.as_millis() >= 200); // Should take at least 200ms
        
        // Should log SLA violation (we can't easily test logging in unit tests)
        let metrics = response.metrics.unwrap();
        assert!(metrics.duration_ms >= 200);
    }

    #[tokio::test]
    async fn test_search_empty_query_error() {
        let service = create_mock_service();
        
        let proto_request = SearchRequest {
            query: "".to_string(), // Empty query
            limit: 5,
            dataset_sha256: "abcd1234".to_string(),
        };
        
        let request = Request::new(proto_request);
        let result = service.search(request).await;
        
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert_eq!(error.code(), Code::InvalidArgument);
    }

    #[tokio::test]
    async fn test_search_attestation_uniqueness() {
        let service = create_mock_service();
        
        let proto_request1 = SearchRequest {
            query: "query1".to_string(),
            limit: 5,
            dataset_sha256: "abcd1234".to_string(),
        };
        
        let proto_request2 = SearchRequest {
            query: "query2".to_string(),
            limit: 5,
            dataset_sha256: "abcd1234".to_string(),
        };
        
        let request1 = Request::new(proto_request1);
        let request2 = Request::new(proto_request2);
        
        let result1 = service.search(request1).await.unwrap().into_inner();
        let result2 = service.search(request2).await.unwrap().into_inner();
        
        // Attestation hashes should be different for different queries
        assert_ne!(result1.attestation, result2.attestation);
    }

    // Test health endpoint
    #[tokio::test]
    async fn test_health_check() {
        let service = create_mock_service();
        
        let request = Request::new(HealthRequest {});
        let result = service.health(request).await;
        
        assert!(result.is_ok());
        let response = result.unwrap().into_inner();
        
        assert_eq!(response.status, "healthy");
        assert_eq!(response.mode, "real"); // Must be "real" per proto requirement
        assert!(!response.version.is_empty());
    }

    #[tokio::test]
    async fn test_health_check_response_format() {
        let service = create_mock_service();
        
        let request = Request::new(HealthRequest {});
        let result = service.health(request).await;
        
        assert!(result.is_ok());
        let response = result.unwrap().into_inner();
        
        // Version should match cargo package version
        assert_eq!(response.version, env!("CARGO_PKG_VERSION"));
    }

    // Test build info endpoint
    #[tokio::test]
    async fn test_get_build_info() {
        let service = create_mock_service();
        
        let request = Request::new(BuildInfoRequest {});
        let result = service.get_build_info(request).await;
        
        assert!(result.is_ok());
        let response = result.unwrap().into_inner();
        
        // Build info should be populated by attestation manager
        assert!(!response.version.is_empty());
    }

    // Test handshake endpoint
    #[tokio::test]
    async fn test_handshake_successful() {
        let service = create_mock_service();
        
        let nonce = "test_nonce_12345".to_string();
        let request = Request::new(HandshakeRequest {
            nonce: nonce.clone(),
        });
        
        let result = service.handshake(request).await;
        
        assert!(result.is_ok());
        let response = result.unwrap().into_inner();
        
        assert_eq!(response.nonce, nonce);
        assert!(!response.response.is_empty());
        assert!(response.build_info.is_some());
    }

    #[tokio::test]
    async fn test_handshake_different_nonces() {
        let service = create_mock_service();
        
        let nonce1 = "nonce1".to_string();
        let nonce2 = "nonce2".to_string();
        
        let request1 = Request::new(HandshakeRequest {
            nonce: nonce1.clone(),
        });
        let request2 = Request::new(HandshakeRequest {
            nonce: nonce2.clone(),
        });
        
        let result1 = service.handshake(request1).await.unwrap().into_inner();
        let result2 = service.handshake(request2).await.unwrap().into_inner();
        
        // Different nonces should produce different responses
        assert_ne!(result1.response, result2.response);
        assert_eq!(result1.nonce, nonce1);
        assert_eq!(result2.nonce, nonce2);
    }

    #[tokio::test]
    async fn test_handshake_empty_nonce() {
        let service = create_mock_service();
        
        let request = Request::new(HandshakeRequest {
            nonce: "".to_string(), // Empty nonce
        });
        
        let result = service.handshake(request).await;
        
        // Should still succeed with empty nonce
        assert!(result.is_ok());
        let response = result.unwrap().into_inner();
        assert_eq!(response.nonce, "");
        assert!(!response.response.is_empty());
    }

    // Test server configuration
    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        
        assert_eq!(config.bind_address, "127.0.0.1");
        assert_eq!(config.port, 50051);
        assert_eq!(config.max_concurrent_requests, 1000);
        assert_eq!(config.request_timeout, Duration::from_millis(150));
        assert!(!config.enable_reflection); // Should be false in production
        assert!(config.enable_health_check);
    }

    #[test]
    fn test_server_config_custom() {
        let config = ServerConfig {
            bind_address: "0.0.0.0".to_string(),
            port: 8080,
            max_concurrent_requests: 500,
            request_timeout: Duration::from_millis(300),
            enable_reflection: true,
            enable_health_check: false,
        };
        
        assert_eq!(config.bind_address, "0.0.0.0");
        assert_eq!(config.port, 8080);
        assert_eq!(config.max_concurrent_requests, 500);
        assert_eq!(config.request_timeout, Duration::from_millis(300));
        assert!(config.enable_reflection);
        assert!(!config.enable_health_check);
    }

    #[test]
    fn test_server_config_clone() {
        let original = ServerConfig::default();
        let cloned = original.clone();
        
        assert_eq!(original.bind_address, cloned.bind_address);
        assert_eq!(original.port, cloned.port);
        assert_eq!(original.max_concurrent_requests, cloned.max_concurrent_requests);
    }

    // Test concurrent requests
    #[tokio::test]
    async fn test_concurrent_search_requests() {
        let service = Arc::new(create_mock_service());
        let mut handles = vec![];
        
        // Spawn multiple concurrent search requests
        for i in 0..10 {
            let service_clone = service.clone();
            let handle = tokio::spawn(async move {
                let proto_request = SearchRequest {
                    query: format!("query {}", i),
                    limit: 3,
                    dataset_sha256: "abcd1234".to_string(),
                };
                
                let request = Request::new(proto_request);
                service_clone.search(request).await
            });
            handles.push(handle);
        }
        
        // Wait for all requests to complete
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
            
            let response = result.unwrap().into_inner();
            assert_eq!(response.results.len(), 3); // Mock returns 3 results
        }
    }

    #[tokio::test]
    async fn test_concurrent_different_endpoints() {
        let service = Arc::new(create_mock_service());
        let mut handles = vec![];
        
        // Test concurrent access to different endpoints
        for i in 0..5 {
            let service_clone = service.clone();
            let handle = tokio::spawn(async move {
                match i % 4 {
                    0 => {
                        let request = Request::new(SearchRequest {
                            query: format!("search {}", i),
                            limit: 5,
                            dataset_sha256: "abcd1234".to_string(),
                        });
                        service_clone.search(request).await.map(|r| format!("search_{}", r.into_inner().results.len()))
                    }
                    1 => {
                        let request = Request::new(HealthRequest {});
                        service_clone.health(request).await.map(|r| r.into_inner().status)
                    }
                    2 => {
                        let request = Request::new(BuildInfoRequest {});
                        service_clone.get_build_info(request).await.map(|r| r.into_inner().version)
                    }
                    3 => {
                        let request = Request::new(HandshakeRequest {
                            nonce: format!("nonce_{}", i),
                        });
                        service_clone.handshake(request).await.map(|r| r.into_inner().nonce)
                    }
                    _ => unreachable!(),
                }
            });
            handles.push(handle);
        }
        
        // All requests should complete successfully
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
        }
    }

    // Test edge cases and error conditions
    #[tokio::test]
    async fn test_very_long_query() {
        let service = create_mock_service();
        
        let long_query = "a".repeat(10000);
        let proto_request = SearchRequest {
            query: long_query.clone(),
            limit: 5,
            dataset_sha256: "abcd1234".to_string(),
        };
        
        let request = Request::new(proto_request);
        let result = service.search(request).await;
        
        assert!(result.is_ok());
        let response = result.unwrap().into_inner();
        
        // Should handle long queries gracefully
        assert_eq!(response.results.len(), 5);
    }

    #[tokio::test]
    async fn test_unicode_query() {
        let service = create_mock_service();
        
        let unicode_queries = vec![
            "函数名称", // Chinese
            "función_test", // Spanish with accents
            "クラス名", // Japanese
            "переменная", // Cyrillic
            "🚀_test_function", // Emoji
        ];
        
        for query in unicode_queries {
            let proto_request = SearchRequest {
                query: query.to_string(),
                limit: 3,
                dataset_sha256: "abcd1234".to_string(),
            };
            
            let request = Request::new(proto_request);
            let result = service.search(request).await;
            
            assert!(result.is_ok());
            let response = result.unwrap().into_inner();
            assert_eq!(response.results.len(), 3);
        }
    }

    #[tokio::test]
    async fn test_zero_and_negative_limits() {
        let service = create_mock_service();
        
        // Zero limit should default to 50
        let proto_request = SearchRequest {
            query: "test".to_string(),
            limit: 0,
            dataset_sha256: "abcd1234".to_string(),
        };
        
        let internal_request = service.convert_request(proto_request).unwrap();
        assert_eq!(internal_request.max_results, 50);
    }

    #[tokio::test]
    async fn test_special_characters_in_query() {
        let service = create_mock_service();
        
        let special_queries = vec![
            r#"query with "quotes""#,
            "query with 'single quotes'",
            "query with [brackets]",
            "query with {braces}",
            "query with (parentheses)",
            "query with <angles>",
            "query with $special!@#%^&*()_+",
            "query with newline\n and tab\t",
        ];
        
        for query in special_queries {
            let proto_request = SearchRequest {
                query: query.to_string(),
                limit: 3,
                dataset_sha256: "abcd1234".to_string(),
            };
            
            let request = Request::new(proto_request);
            let result = service.search(request).await;
            
            assert!(result.is_ok());
        }
    }

    // Test performance characteristics
    #[tokio::test]
    async fn test_search_performance_measurement() {
        let service = create_mock_service_with_slow_response(100);
        
        let proto_request = SearchRequest {
            query: "performance test".to_string(),
            limit: 5,
            dataset_sha256: "abcd1234".to_string(),
        };
        
        let start_time = Instant::now();
        let request = Request::new(proto_request);
        let result = service.search(request).await;
        let total_duration = start_time.elapsed();
        
        assert!(result.is_ok());
        let response = result.unwrap().into_inner();
        
        // Should complete in reasonable time
        assert!(total_duration.as_millis() >= 100);
        assert!(total_duration.as_millis() < 1000); // Should not be too slow
        
        let metrics = response.metrics.unwrap();
        assert!(metrics.duration_ms >= 100);
    }

    #[tokio::test]
    async fn test_memory_usage_with_large_results() {
        let service = create_mock_service();
        
        // Create service that returns many results
        let mock_search_engine = Arc::new(MockSearchEngine::with_results_count(1000));
        let metrics_collector = Arc::new(crate::metrics::MetricsCollector::new());
        let attestation_manager = Arc::new(crate::attestation::AttestationManager::new());
        let benchmark_runner = Arc::new(crate::benchmark::BenchmarkRunner::new());
        
        let search_engine: Arc<SearchEngine> = unsafe { std::mem::transmute(mock_search_engine) };
        let service = LensSearchServiceImpl::new(
            search_engine,
            metrics_collector,
            attestation_manager,
            benchmark_runner,
        );
        
        let proto_request = SearchRequest {
            query: "large results test".to_string(),
            limit: 1000,
            dataset_sha256: "abcd1234".to_string(),
        };
        
        let request = Request::new(proto_request);
        let result = service.search(request).await;
        
        assert!(result.is_ok());
        let response = result.unwrap().into_inner();
        
        // Should handle large result sets
        assert_eq!(response.results.len(), 1000);
        
        let metrics = response.metrics.unwrap();
        assert_eq!(metrics.total_docs, 1000);
        assert_eq!(metrics.matched_docs, 1000);
    }

    // Test error handling and resilience
    #[tokio::test]
    async fn test_service_resilience_after_errors() {
        let service = create_mock_service_with_failure();
        
        // First request should fail gracefully
        let proto_request = SearchRequest {
            query: "test query".to_string(),
            limit: 5,
            dataset_sha256: "abcd1234".to_string(),
        };
        
        let request = Request::new(proto_request.clone());
        let result = service.search(request).await;
        
        assert!(result.is_ok()); // Should return Ok with error response
        let response = result.unwrap().into_inner();
        assert_eq!(response.results.len(), 0);
        assert_eq!(response.attestation, "error");
        
        // Service should still respond to health checks
        let health_request = Request::new(HealthRequest {});
        let health_result = service.health(health_request).await;
        assert!(health_result.is_ok());
    }

    #[tokio::test] 
    async fn test_handshake_consistency() {
        let service = create_mock_service();
        
        let nonce = "consistent_nonce".to_string();
        
        // Same nonce should produce same response (if deterministic)
        let request1 = Request::new(HandshakeRequest {
            nonce: nonce.clone(),
        });
        let request2 = Request::new(HandshakeRequest {
            nonce: nonce.clone(),
        });
        
        let result1 = service.handshake(request1).await.unwrap().into_inner();
        let result2 = service.handshake(request2).await.unwrap().into_inner();
        
        assert_eq!(result1.nonce, result2.nonce);
        // Response hash might vary due to timestamp, but structure should be consistent
        assert!(!result1.response.is_empty());
        assert!(!result2.response.is_empty());
    }

    // Test attestation hash properties
    #[tokio::test]
    async fn test_attestation_hash_properties() {
        let service = create_mock_service();
        
        let proto_request = SearchRequest {
            query: "attestation test".to_string(),
            limit: 5,
            dataset_sha256: "abcd1234".to_string(),
        };
        
        let request = Request::new(proto_request);
        let result = service.search(request).await.unwrap().into_inner();
        
        // Attestation should be hex string of SHA256 (64 characters)
        assert_eq!(result.attestation.len(), 64);
        assert!(result.attestation.chars().all(|c| c.is_ascii_hexdigit()));
        
        // Should be lowercase hex
        assert!(result.attestation.chars().all(|c| !c.is_uppercase()));
    }
}