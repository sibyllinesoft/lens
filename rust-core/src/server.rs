use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

use crate::{AttestationService, SearchEngine};
use crate::search::SearchResult;

#[derive(Debug, Serialize, Deserialize)]
pub struct HandshakeRequest {
    pub nonce: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HandshakeResponse {
    pub success: bool,
    pub attestation: Option<crate::attestation::AttestationRecord>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub total: usize,
    pub query_time_ms: u64,
    pub attestation_hash: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ManifestResponse {
    pub service: String,
    pub version: String,
    pub mode: String,
    pub git_sha: String,
    pub build_target: String,
    pub attestation_enabled: bool,
}

#[derive(Clone)]
pub struct AppState {
    pub search_engine: Arc<Mutex<SearchEngine>>,
    pub attestation_service: Arc<AttestationService>,
}

pub struct LensRpcServer {
    app: Router,
    state: AppState,
}

impl LensRpcServer {
    pub fn new() -> Result<Self> {
        // TRIPWIRE: Enforce real mode only
        let attestation_service = Arc::new(AttestationService::new("real")?);
        let search_engine = Arc::new(Mutex::new(SearchEngine::new_in_memory()?));
        
        let state = AppState {
            search_engine,
            attestation_service,
        };
        
        let app = Router::new()
            .route("/manifest", get(manifest_handler))
            .route("/handshake", post(handshake_handler))
            .route("/search", post(search_handler))
            .route("/health", get(health_handler))
            // Note: CORS and tracing middleware removed for initial build compatibility
            .with_state(state.clone());
        
        Ok(LensRpcServer { app, state })
    }
    
    pub fn router(self) -> Router {
        self.app
    }
    
    pub async fn serve(self, addr: &str) -> Result<()> {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        info!("Lens RPC server listening on {}", addr);
        
        axum::serve(listener, self.app).await?;
        Ok(())
    }
}

async fn manifest_handler(State(state): State<AppState>) -> Json<ManifestResponse> {
    Json(ManifestResponse {
        service: "lens-rpc".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        mode: state.attestation_service.get_mode().to_string(),
        git_sha: state.attestation_service.get_git_sha().to_string(),
        build_target: crate::built::TARGET.to_string(),
        attestation_enabled: true,
    })
}

async fn handshake_handler(
    State(state): State<AppState>,
    Json(request): Json<HandshakeRequest>,
) -> Result<Json<HandshakeResponse>, StatusCode> {
    match state.attestation_service.create_handshake(&request.nonce) {
        Ok(attestation) => Ok(Json(HandshakeResponse {
            success: true,
            attestation: Some(attestation),
            error: None,
        })),
        Err(e) => {
            warn!("Handshake failed: {}", e);
            Ok(Json(HandshakeResponse {
                success: false,
                attestation: None,
                error: Some(e.to_string()),
            }))
        }
    }
}

async fn search_handler(
    State(state): State<AppState>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, StatusCode> {
    let start_time = std::time::Instant::now();
    let limit = request.limit.unwrap_or(10);
    
    let search_results = {
        let search_engine = state.search_engine.lock().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        search_engine.search(&request.query, limit).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    };
    
    match Ok::<Vec<SearchResult>, anyhow::Error>(search_results) {
        Ok(results) => {
            let query_time = start_time.elapsed().as_millis() as u64;
            
            // Generate attestation hash for this search
            use sha2::{Sha256, Digest};
            let mut hasher = Sha256::new();
            hasher.update(request.query.as_bytes());
            hasher.update(state.attestation_service.get_git_sha().as_bytes());
            hasher.update(format!("{}", query_time).as_bytes());
            let attestation_hash = format!("{:x}", hasher.finalize());
            
            Ok(Json(SearchResponse {
                total: results.len(),
                results,
                query_time_ms: query_time,
                attestation_hash,
            }))
        }
        Err(e) => {
            warn!("Search failed: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn health_handler(State(state): State<AppState>) -> Result<Json<serde_json::Value>, StatusCode> {
    // TRIPWIRE: Health check must verify mode
    if state.attestation_service.get_mode() != "real" {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }
    
    Ok(Json(serde_json::json!({
        "status": "healthy",
        "mode": state.attestation_service.get_mode(),
        "git_sha": state.attestation_service.get_git_sha(),
        "timestamp": chrono::Utc::now().to_rfc3339()
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum_test::TestServer;
    
    #[tokio::test]
    async fn test_server_creation() {
        let server = LensRpcServer::new();
        assert!(server.is_ok());
    }
    
    #[tokio::test]
    async fn test_manifest_endpoint() {
        let server = LensRpcServer::new().unwrap();
        let test_server = TestServer::new(server.router()).unwrap();
        
        let response = test_server.get("/manifest").await;
        response.assert_status_ok();
        
        let manifest: ManifestResponse = response.json();
        assert_eq!(manifest.service, "lens-rpc");
        assert_eq!(manifest.mode, "real");
        assert!(manifest.attestation_enabled);
    }
    
    #[tokio::test]
    async fn test_health_endpoint() {
        let server = LensRpcServer::new().unwrap();
        let test_server = TestServer::new(server.router()).unwrap();
        
        let response = test_server.get("/health").await;
        response.assert_status_ok();
    }
    
    #[tokio::test]
    async fn test_handshake_endpoint() {
        let server = LensRpcServer::new().unwrap();
        let test_server = TestServer::new(server.router()).unwrap();
        
        let handshake_request = HandshakeRequest {
            nonce: "test-nonce-789".to_string(),
        };
        
        let response = test_server.post("/handshake").json(&handshake_request).await;
        response.assert_status_ok();
        
        let handshake_response: HandshakeResponse = response.json();
        assert!(handshake_response.success);
        assert!(handshake_response.attestation.is_some());
    }
}