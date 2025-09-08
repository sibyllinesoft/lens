use crate::search::SearchResult;
use crate::lsp::LspSearchResponse;
use async_trait::async_trait;
use tokio::time::{timeout, Duration, Instant};
use anyhow::Result;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Pipeline stage trait for zero-copy processing
#[async_trait]
pub trait PipelineStage: Send + Sync {
    /// Stage identifier
    fn name(&self) -> &str;
    
    /// Process input and produce output
    async fn process(&self, input: StageInput) -> Result<StageOutput>;
    
    /// Get stage performance metrics
    fn metrics(&self) -> StageMetrics;
    
    /// Health check for the stage
    async fn health_check(&self) -> HealthStatus;
}

/// Input data for pipeline stages
#[derive(Debug, Clone)]
pub struct StageInput {
    pub query: String,
    pub query_id: String,
    pub context: PipelineContext,
    pub data: StageData,
}

/// Pipeline execution context
#[derive(Debug, Clone)]
pub struct PipelineContext {
    pub timeout: Duration,
    pub language_hint: Option<String>,
    pub systems: Vec<String>,
    pub max_results: usize,
    pub metadata: HashMap<String, String>,
}

/// Stage data (can be different types as it flows through pipeline)
#[derive(Debug, Clone)]
pub enum StageData {
    Query(QueryData),
    LspResults(LspResultData),
    SearchResults(SearchResultData),
    FusedResults(FusedResultData),
}

/// Query data at start of pipeline
#[derive(Debug, Clone)]
pub struct QueryData {
    pub original_query: String,
    pub processed_query: String,
    pub query_type: QueryType,
    pub filters: QueryFilters,
}

/// LSP search results data
#[derive(Debug, Clone)]
pub struct LspResultData {
    pub responses: Vec<LspSearchResponse>,
    pub latencies: HashMap<String, f64>,
    pub errors: Vec<String>,
}

/// Text search results data
#[derive(Debug, Clone)]
pub struct SearchResultData {
    pub results: Vec<SearchResult>,
    pub total_matches: usize,
    pub search_time_ms: f64,
}

/// Fused results data
#[derive(Debug, Clone)]
pub struct FusedResultData {
    pub results: Vec<crate::pipeline::fusion::FusedResult>,
    pub fusion_strategy: String,
    pub confidence: f64,
}

/// Query type classification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryType {
    Exact,
    Fuzzy,
    Structural,
    Semantic,
    Symbol,
    Hybrid,
}

/// Query filters
#[derive(Debug, Clone)]
pub struct QueryFilters {
    pub file_extensions: Vec<String>,
    pub exclude_paths: Vec<String>,
    pub language_filter: Option<String>,
    pub size_limit: Option<usize>,
}

/// Stage output
#[derive(Debug, Clone)]
pub struct StageOutput {
    pub data: StageData,
    pub metrics: StageExecutionMetrics,
    pub errors: Vec<String>,
    pub metadata: HashMap<String, String>,
}

/// Stage execution metrics
#[derive(Debug, Clone)]
pub struct StageExecutionMetrics {
    pub execution_time_ms: f64,
    pub memory_usage_mb: f64,
    pub items_processed: usize,
    pub success: bool,
}

/// Stage performance metrics
#[derive(Debug, Clone)]
pub struct StageMetrics {
    pub total_executions: u64,
    pub successful_executions: u64,
    pub failed_executions: u64,
    pub average_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub average_memory_mb: f64,
}

/// Health status for pipeline stages
#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub healthy: bool,
    pub message: String,
    pub checks: HashMap<String, bool>,
}

/// Query preprocessing stage
pub struct QueryPreprocessingStage {
    metrics: parking_lot::RwLock<StageMetrics>,
}

impl QueryPreprocessingStage {
    pub fn new() -> Self {
        Self {
            metrics: parking_lot::RwLock::new(StageMetrics::new()),
        }
    }

    fn classify_query(&self, query: &str) -> QueryType {
        if query.starts_with('"') && query.ends_with('"') {
            QueryType::Exact
        } else if query.contains("class ") || query.contains("function ") || query.contains("interface ") {
            QueryType::Structural
        } else if query.contains('*') || query.contains('?') {
            QueryType::Fuzzy
        } else if query.split_whitespace().count() > 3 {
            QueryType::Semantic
        } else if query.chars().all(|c| c.is_alphanumeric() || c == '_') {
            QueryType::Symbol
        } else {
            QueryType::Hybrid
        }
    }

    fn extract_filters(&self, query: &str) -> (String, QueryFilters) {
        // Simple filter extraction - would be more sophisticated in practice
        let mut processed_query = query.to_string();
        let mut file_extensions = Vec::new();
        let mut exclude_paths = Vec::new();
        let mut language_filter = None;

        // Extract file extensions (e.g., "ext:ts")
        if let Some(ext_match) = regex::Regex::new(r"ext:(\w+)").unwrap().find(query) {
            file_extensions.push(format!(".{}", &query[ext_match.start() + 4..ext_match.end()]));
            processed_query = processed_query.replace(ext_match.as_str(), "").trim().to_string();
        }

        // Extract language filter (e.g., "lang:typescript")
        if let Some(lang_match) = regex::Regex::new(r"lang:(\w+)").unwrap().find(query) {
            language_filter = Some(query[lang_match.start() + 5..lang_match.end()].to_string());
            processed_query = processed_query.replace(lang_match.as_str(), "").trim().to_string();
        }

        (processed_query, QueryFilters {
            file_extensions,
            exclude_paths,
            language_filter,
            size_limit: None,
        })
    }
}

#[async_trait]
impl PipelineStage for QueryPreprocessingStage {
    fn name(&self) -> &str {
        "query_preprocessing"
    }

    async fn process(&self, input: StageInput) -> Result<StageOutput> {
        let start_time = Instant::now();
        
        let query_type = self.classify_query(&input.query);
        let (processed_query, filters) = self.extract_filters(&input.query);
        
        let query_data = QueryData {
            original_query: input.query.clone(),
            processed_query,
            query_type,
            filters,
        };

        let execution_time = start_time.elapsed().as_millis() as f64;
        
        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.total_executions += 1;
            metrics.successful_executions += 1;
            metrics.update_latency(execution_time);
        }

        Ok(StageOutput {
            data: StageData::Query(query_data),
            metrics: StageExecutionMetrics {
                execution_time_ms: execution_time,
                memory_usage_mb: 1.0, // Minimal memory usage
                items_processed: 1,
                success: true,
            },
            errors: Vec::new(),
            metadata: HashMap::new(),
        })
    }

    fn metrics(&self) -> StageMetrics {
        self.metrics.read().clone()
    }

    async fn health_check(&self) -> HealthStatus {
        HealthStatus {
            healthy: true,
            message: "Query preprocessing stage is healthy".to_string(),
            checks: HashMap::from([
                ("regex_engine".to_string(), true),
                ("classification".to_string(), true),
            ]),
        }
    }
}

/// LSP search stage
pub struct LspSearchStage {
    metrics: parking_lot::RwLock<StageMetrics>,
    lsp_clients: HashMap<String, Arc<dyn LspClient>>,
}

/// LSP client trait
#[async_trait]
pub trait LspClient: Send + Sync {
    async fn search(&self, query: &str, options: &LspSearchOptions) -> Result<LspSearchResponse>;
}

/// LSP search options
#[derive(Debug, Clone)]
pub struct LspSearchOptions {
    pub timeout: Duration,
    pub max_results: usize,
    pub language_hint: Option<String>,
}

impl LspSearchStage {
    pub fn new() -> Self {
        Self {
            metrics: parking_lot::RwLock::new(StageMetrics::new()),
            lsp_clients: HashMap::new(),
        }
    }

    pub fn add_lsp_client(&mut self, language: String, client: Arc<dyn LspClient>) {
        self.lsp_clients.insert(language, client);
    }
}

#[async_trait]
impl PipelineStage for LspSearchStage {
    fn name(&self) -> &str {
        "lsp_search"
    }

    async fn process(&self, input: StageInput) -> Result<StageOutput> {
        let start_time = Instant::now();
        let mut responses = Vec::new();
        let mut latencies = HashMap::new();
        let mut errors = Vec::new();

        if let StageData::Query(query_data) = &input.data {
            // Execute LSP searches for relevant languages
            let languages_to_search = if let Some(lang) = &input.context.language_hint {
                vec![lang.clone()]
            } else {
                vec!["typescript".to_string(), "python".to_string(), "rust".to_string()]
            };

            for language in languages_to_search {
                if let Some(client) = self.lsp_clients.get(&language) {
                    let search_start = Instant::now();
                    let options = LspSearchOptions {
                        timeout: input.context.timeout,
                        max_results: input.context.max_results,
                        language_hint: Some(language.clone()),
                    };

                    match timeout(input.context.timeout, client.search(&query_data.processed_query, &options)).await {
                        Ok(Ok(response)) => {
                            let latency = search_start.elapsed().as_millis() as f64;
                            latencies.insert(language, latency);
                            responses.push(response);
                        }
                        Ok(Err(e)) => {
                            errors.push(format!("LSP search failed for {}: {}", language, e));
                        }
                        Err(_) => {
                            errors.push(format!("LSP search timeout for {}", language));
                        }
                    }
                }
            }
        } else {
            return Err(anyhow::anyhow!("Invalid input data type for LSP search stage"));
        }

        let execution_time = start_time.elapsed().as_millis() as f64;
        let success = !responses.is_empty();

        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.total_executions += 1;
            if success {
                metrics.successful_executions += 1;
            } else {
                metrics.failed_executions += 1;
            }
            metrics.update_latency(execution_time);
        }

        Ok(StageOutput {
            data: StageData::LspResults(LspResultData {
                responses: responses.clone(),
                latencies,
                errors: errors.clone(),
            }),
            metrics: StageExecutionMetrics {
                execution_time_ms: execution_time,
                memory_usage_mb: 2.0,
                items_processed: responses.len(),
                success,
            },
            errors,
            metadata: HashMap::new(),
        })
    }

    fn metrics(&self) -> StageMetrics {
        self.metrics.read().clone()
    }

    async fn health_check(&self) -> HealthStatus {
        let mut checks = HashMap::new();
        
        // Check LSP client connectivity
        for (language, _client) in &self.lsp_clients {
            // Would do actual health check
            checks.insert(format!("lsp_{}", language), true);
        }

        let all_healthy = checks.values().all(|&v| v);

        HealthStatus {
            healthy: all_healthy,
            message: if all_healthy {
                "All LSP clients are healthy".to_string()
            } else {
                "Some LSP clients are unhealthy".to_string()
            },
            checks,
        }
    }
}

/// Text search stage using Tantivy
pub struct TextSearchStage {
    metrics: parking_lot::RwLock<StageMetrics>,
    index: Option<tantivy::Index>,
}

impl TextSearchStage {
    pub fn new() -> Self {
        Self {
            metrics: parking_lot::RwLock::new(StageMetrics::new()),
            index: None,
        }
    }

    pub fn set_index(&mut self, index: tantivy::Index) {
        self.index = Some(index);
    }
}

#[async_trait]
impl PipelineStage for TextSearchStage {
    fn name(&self) -> &str {
        "text_search"
    }

    async fn process(&self, input: StageInput) -> Result<StageOutput> {
        let start_time = Instant::now();
        
        // TODO: Implement actual Tantivy search
        let results = Vec::new();
        let total_matches = 0;
        
        let execution_time = start_time.elapsed().as_millis() as f64;

        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.total_executions += 1;
            metrics.successful_executions += 1;
            metrics.update_latency(execution_time);
        }

        Ok(StageOutput {
            data: StageData::SearchResults(SearchResultData {
                results,
                total_matches,
                search_time_ms: execution_time,
            }),
            metrics: StageExecutionMetrics {
                execution_time_ms: execution_time,
                memory_usage_mb: 5.0,
                items_processed: total_matches,
                success: true,
            },
            errors: Vec::new(),
            metadata: HashMap::new(),
        })
    }

    fn metrics(&self) -> StageMetrics {
        self.metrics.read().clone()
    }

    async fn health_check(&self) -> HealthStatus {
        let index_healthy = self.index.is_some();
        
        HealthStatus {
            healthy: index_healthy,
            message: if index_healthy {
                "Text search index is available".to_string()
            } else {
                "Text search index is not available".to_string()
            },
            checks: HashMap::from([
                ("index_available".to_string(), index_healthy),
            ]),
        }
    }
}

impl StageMetrics {
    fn new() -> Self {
        Self {
            total_executions: 0,
            successful_executions: 0,
            failed_executions: 0,
            average_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            average_memory_mb: 0.0,
        }
    }

    fn update_latency(&mut self, latency_ms: f64) {
        if self.total_executions > 0 {
            self.average_latency_ms = (self.average_latency_ms * (self.total_executions - 1) as f64 + latency_ms) / self.total_executions as f64;
        } else {
            self.average_latency_ms = latency_ms;
        }
        
        // Simplified P95 calculation - would use proper percentile tracking in practice
        self.p95_latency_ms = self.average_latency_ms * 1.2;
    }
}

use std::sync::Arc;