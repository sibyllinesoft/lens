use std::env;
use std::process::Command;
use std::time::Instant;
use warp::Filter;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use tempfile::NamedTempFile;
use std::io::Write;

#[derive(Debug, Deserialize)]
struct SearchRequest {
    pattern: String,
    language: Option<String>,
    #[serde(default = "default_max_results")]
    max_results: usize,
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
    start_line: u32,
    end_line: u32,
    start_column: u32,
    end_column: u32,
    matched_text: String,
    context: String,
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
    
    // Create temporary configuration file for ast-grep
    let config_content = create_ast_grep_config(&req.pattern, &req.language);
    let mut temp_config = match NamedTempFile::new() {
        Ok(file) => file,
        Err(e) => {
            tracing::error!("Failed to create temp config file: {}", e);
            return Err(warp::reject::custom(SearchError::ConfigError));
        }
    };
    
    if let Err(e) = temp_config.write_all(config_content.as_bytes()) {
        tracing::error!("Failed to write config file: {}", e);
        return Err(warp::reject::custom(SearchError::ConfigError));
    }
    
    // Build ast-grep command
    let mut cmd = Command::new("ast-grep");
    cmd.arg("scan")
       .arg("--json")
       .arg("--config").arg(temp_config.path())
       .arg(&corpus_path);
    
    // Execute ast-grep
    let output = match cmd.output() {
        Ok(output) => output,
        Err(e) => {
            tracing::error!("Failed to execute ast-grep: {}", e);
            return Err(warp::reject::custom(SearchError::ExecutionError));
        }
    };
    
    let latency_ms = start_time.elapsed().as_secs_f64() * 1000.0;
    let sla_violated = latency_ms > 150.0;
    
    // Parse ast-grep JSON output
    let stdout = String::from_utf8_lossy(&output.stdout);
    let results = parse_ast_grep_output(&stdout, req.max_results);
    
    let response = SearchResult {
        query_id,
        system: "ast-grep".to_string(),
        version: get_ast_grep_version(),
        latency_ms,
        total_hits: results.len(),
        results,
        sla_violated,
    };
    
    tracing::info!("Search completed: pattern='{}', hits={}, latency={}ms, sla_violated={}", 
                   req.pattern, response.total_hits, latency_ms, sla_violated);
    
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
        system: "ast-grep".to_string(),
        version: get_ast_grep_version(),
        corpus_path,
        corpus_files,
    };
    
    Ok(warp::reply::json(&response))
}

fn create_ast_grep_config(pattern: &str, language: &Option<String>) -> String {
    let lang = language.as_deref().unwrap_or("javascript");
    
    format!(r#"
rules:
  - id: search-pattern
    language: {}
    pattern: |
      {}
    message: "Found matching pattern"
"#, lang, pattern)
}

fn parse_ast_grep_output(stdout: &str, max_results: usize) -> Vec<SearchHit> {
    let mut results = Vec::new();
    
    for line in stdout.lines() {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(line) {
            if let (
                Some(file_path),
                Some(start_line),
                Some(end_line),
                Some(start_col),
                Some(end_col),
                Some(text),
            ) = (
                json.get("file").and_then(|v| v.as_str()),
                json.get("range").and_then(|r| r.get("start")).and_then(|s| s.get("line")).and_then(|v| v.as_u64()),
                json.get("range").and_then(|r| r.get("end")).and_then(|e| e.get("line")).and_then(|v| v.as_u64()),
                json.get("range").and_then(|r| r.get("start")).and_then(|s| s.get("column")).and_then(|v| v.as_u64()),
                json.get("range").and_then(|r| r.get("end")).and_then(|e| e.get("column")).and_then(|v| v.as_u64()),
                json.get("text").and_then(|v| v.as_str()),
            ) {
                let context = json.get("context").and_then(|v| v.as_str()).unwrap_or("").to_string();
                
                results.push(SearchHit {
                    file_path: file_path.to_string(),
                    start_line: start_line as u32,
                    end_line: end_line as u32,
                    start_column: start_col as u32,
                    end_column: end_col as u32,
                    matched_text: text.to_string(),
                    context,
                });
                
                if results.len() >= max_results {
                    break;
                }
            }
        }
    }
    
    results
}

fn get_ast_grep_version() -> String {
    match Command::new("ast-grep").arg("--version").output() {
        Ok(output) => {
            String::from_utf8_lossy(&output.stdout)
                .lines()
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
    
    const ExecutionError: SearchError = SearchError::new("Failed to execute ast-grep");
    const ConfigError: SearchError = SearchError::new("Failed to create config file");
}

impl warp::reject::Reject for SearchError {}

#[tokio::main]
async fn main() {
    tracing_subscriber::init();
    
    let port = env::var("AST_GREP_SERVER_PORT")
        .unwrap_or_else(|_| "8082".to_string())
        .parse::<u16>()
        .unwrap_or(8082);
    
    tracing::info!("Starting ast-grep server on port {}", port);
    
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
        .with(warp::log("ast_grep_server"));
    
    warp::serve(routes)
        .run(([0, 0, 0, 0], port))
        .await;
}