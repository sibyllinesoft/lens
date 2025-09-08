use std::env;
use std::process::Command;
use std::time::Instant;
use warp::Filter;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Deserialize)]
struct SearchRequest {
    query: String,
    #[serde(default = "default_max_results")]
    max_results: usize,
    #[serde(default)]
    case_sensitive: bool,
    #[serde(default)]
    regex: bool,
}

fn default_max_results() -> usize { 50 }

#[derive(Debug, Serialize)]
struct SearchResult {
    query_id: String,
    system: String,
    version: String,
    latency_ms: f64,
    total_hits: usize,
    results: Vec<SearchHit>,
    sla_violated: bool,
}

#[derive(Debug, Serialize)]
struct SearchHit {
    file_path: String,
    line_number: u32,
    line_content: String,
    byte_offset: u64,
    match_start: usize,
    match_end: usize,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    system: String,
    version: String,
    corpus_path: String,
    corpus_files: usize,
}

async fn search_handler(req: SearchRequest) -> Result<impl warp::Reply, warp::Rejection> {
    let query_id = Uuid::new_v4().to_string();
    let start_time = Instant::now();
    
    let corpus_path = env::var("CORPUS_PATH").unwrap_or_else(|_| "/datasets".to_string());
    
    // Build ripgrep command
    let mut cmd = Command::new("rg");
    cmd.arg("--json")
       .arg("--max-count").arg(req.max_results.to_string());
    
    if !req.case_sensitive {
        cmd.arg("--ignore-case");
    }
    
    if !req.regex {
        cmd.arg("--fixed-strings");
    }
    
    cmd.arg(&req.query)
       .arg(&corpus_path);
    
    // Execute ripgrep
    let output = match cmd.output() {
        Ok(output) => output,
        Err(e) => {
            tracing::error!("Failed to execute ripgrep: {}", e);
            return Err(warp::reject::custom(SearchError::ExecutionError));
        }
    };
    
    let latency_ms = start_time.elapsed().as_secs_f64() * 1000.0;
    let sla_violated = latency_ms > 150.0;
    
    // Parse ripgrep JSON output
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut results = Vec::new();
    
    for line in stdout.lines() {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(line) {
            if json.get("type") == Some(&serde_json::Value::String("match".to_string())) {
                if let Some(data) = json.get("data") {
                    if let (Some(path), Some(line_num), Some(line_text), Some(byte_offset)) = (
                        data.get("path").and_then(|v| v.get("text")).and_then(|v| v.as_str()),
                        data.get("line_number").and_then(|v| v.as_u64()),
                        data.get("lines").and_then(|v| v.get("text")).and_then(|v| v.as_str()),
                        data.get("absolute_offset").and_then(|v| v.as_u64()),
                    ) {
                        // Extract match positions
                        let (match_start, match_end) = if let Some(submatches) = data.get("submatches") {
                            if let Some(submatch) = submatches.get(0) {
                                let start = submatch.get("start").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                                let end = submatch.get("end").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                                (start, end)
                            } else {
                                (0, 0)
                            }
                        } else {
                            (0, 0)
                        };
                        
                        results.push(SearchHit {
                            file_path: path.to_string(),
                            line_number: line_num as u32,
                            line_content: line_text.to_string(),
                            byte_offset,
                            match_start,
                            match_end,
                        });
                    }
                }
            }
        }
    }
    
    let response = SearchResult {
        query_id,
        system: "ripgrep".to_string(),
        version: get_ripgrep_version(),
        latency_ms,
        total_hits: results.len(),
        results,
        sla_violated,
    };
    
    tracing::info!("Search completed: query='{}', hits={}, latency={}ms, sla_violated={}", 
                   req.query, response.total_hits, latency_ms, sla_violated);
    
    Ok(warp::reply::json(&response))
}

async fn health_handler() -> Result<impl warp::Reply, warp::Rejection> {
    let corpus_path = env::var("CORPUS_PATH").unwrap_or_else(|_| "/datasets".to_string());
    
    // Count corpus files
    let corpus_files = match std::fs::read_dir(&corpus_path) {
        Ok(entries) => entries.count(),
        Err(_) => 0,
    };
    
    let response = HealthResponse {
        status: "healthy".to_string(),
        system: "ripgrep".to_string(),
        version: get_ripgrep_version(),
        corpus_path,
        corpus_files,
    };
    
    Ok(warp::reply::json(&response))
}

fn get_ripgrep_version() -> String {
    match Command::new("rg").arg("--version").output() {
        Ok(output) => {
            let version_output = String::from_utf8_lossy(&output.stdout);
            version_output.lines()
                .next()
                .unwrap_or("unknown")
                .trim()
                .to_string()
        }
        Err(_) => "unknown".to_string(),
    }
}

#[derive(Debug)]
struct SearchError {
    message: String,
}

impl SearchError {
    const fn new(message: &'static str) -> Self {
        Self { message: message.to_string() }
    }
    
    const ExecutionError: SearchError = SearchError::new("Failed to execute ripgrep");
}

impl warp::reject::Reject for SearchError {}

#[tokio::main]
async fn main() {
    tracing_subscriber::init();
    
    let port = env::var("RIPGREP_SERVER_PORT")
        .unwrap_or_else(|_| "8080".to_string())
        .parse::<u16>()
        .unwrap_or(8080);
    
    tracing::info!("Starting ripgrep server on port {}", port);
    
    let search = warp::path("search")
        .and(warp::post())
        .and(warp::body::json())
        .and_then(search_handler);
    
    let health = warp::path("health")
        .and(warp::get())
        .and_then(health_handler);
    
    let cors = warp::cors()
        .allow_any_origin()
        .allow_headers(vec!["content-type"])
        .allow_methods(vec!["GET", "POST"]);
    
    let routes = search
        .or(health)
        .with(cors)
        .with(warp::log("ripgrep_server"));
    
    warp::serve(routes)
        .run(([0, 0, 0, 0], port))
        .await;
}