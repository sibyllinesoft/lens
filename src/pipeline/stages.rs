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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
        } else if query.contains('*') || query.contains('?') {
            QueryType::Fuzzy
        } else if query.split_whitespace().count() > 3 {
            QueryType::Semantic  // Check semantic first (long queries)
        } else if query.contains("class ") || query.contains("function ") || query.contains("interface ") {
            QueryType::Structural
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
            processed_query = processed_query.replace(ext_match.as_str(), "");
        }

        // Extract language filter (e.g., "lang:typescript")
        if let Some(lang_match) = regex::Regex::new(r"lang:(\w+)").unwrap().find(query) {
            language_filter = Some(query[lang_match.start() + 5..lang_match.end()].to_string());
            processed_query = processed_query.replace(lang_match.as_str(), "");
        }

        // Clean up extra spaces
        processed_query = processed_query.split_whitespace().collect::<Vec<_>>().join(" ");

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tokio::time::Duration;

    fn create_test_context() -> PipelineContext {
        PipelineContext {
            timeout: Duration::from_secs(30),
            language_hint: Some("rust".to_string()),
            systems: vec!["test".to_string()],
            max_results: 100,
            metadata: HashMap::new(),
        }
    }

    fn create_test_query_data() -> QueryData {
        QueryData {
            original_query: "test query".to_string(),
            processed_query: "test".to_string(),
            query_type: QueryType::Exact,
            filters: QueryFilters {
                file_extensions: vec![".rs".to_string()],
                exclude_paths: vec![],
                language_filter: Some("rust".to_string()),
                size_limit: Some(1000),
            },
        }
    }

    #[test]
    fn test_stage_input_creation() {
        let context = create_test_context();
        let query_data = create_test_query_data();
        
        let input = StageInput {
            query: "test query".to_string(),
            query_id: "test-123".to_string(),
            context: context.clone(),
            data: StageData::Query(query_data),
        };

        assert_eq!(input.query, "test query");
        assert_eq!(input.query_id, "test-123");
        assert_eq!(input.context.timeout, Duration::from_secs(30));
        assert_eq!(input.context.max_results, 100);
    }

    #[test]
    fn test_pipeline_context_creation() {
        let mut metadata = HashMap::new();
        metadata.insert("test_key".to_string(), "test_value".to_string());

        let context = PipelineContext {
            timeout: Duration::from_millis(5000),
            language_hint: Some("typescript".to_string()),
            systems: vec!["lsp".to_string(), "search".to_string()],
            max_results: 50,
            metadata,
        };

        assert_eq!(context.timeout, Duration::from_millis(5000));
        assert_eq!(context.language_hint, Some("typescript".to_string()));
        assert_eq!(context.systems.len(), 2);
        assert_eq!(context.max_results, 50);
        assert_eq!(context.metadata.get("test_key"), Some(&"test_value".to_string()));
    }

    #[test]
    fn test_query_type_variants() {
        let exact = QueryType::Exact;
        let fuzzy = QueryType::Fuzzy;
        let structural = QueryType::Structural;
        let semantic = QueryType::Semantic;
        let symbol = QueryType::Symbol;
        let hybrid = QueryType::Hybrid;

        // Test serialization/deserialization
        assert_eq!(serde_json::to_string(&exact).unwrap(), "\"exact\"");
        assert_eq!(serde_json::to_string(&fuzzy).unwrap(), "\"fuzzy\"");
        assert_eq!(serde_json::to_string(&structural).unwrap(), "\"structural\"");
        assert_eq!(serde_json::to_string(&semantic).unwrap(), "\"semantic\"");
        assert_eq!(serde_json::to_string(&symbol).unwrap(), "\"symbol\"");
        assert_eq!(serde_json::to_string(&hybrid).unwrap(), "\"hybrid\"");
    }

    #[test]
    fn test_query_filters_creation() {
        let filters = QueryFilters {
            file_extensions: vec![".rs".to_string(), ".ts".to_string()],
            exclude_paths: vec!["target/".to_string(), "node_modules/".to_string()],
            language_filter: Some("rust".to_string()),
            size_limit: Some(10000),
        };

        assert_eq!(filters.file_extensions.len(), 2);
        assert_eq!(filters.exclude_paths.len(), 2);
        assert_eq!(filters.language_filter, Some("rust".to_string()));
        assert_eq!(filters.size_limit, Some(10000));
    }

    #[test]
    fn test_stage_data_variants() {
        let query_data = create_test_query_data();
        let stage_data_query = StageData::Query(query_data);

        match stage_data_query {
            StageData::Query(data) => {
                assert_eq!(data.original_query, "test query");
                assert_eq!(data.query_type, QueryType::Exact);
            }
            _ => panic!("Expected Query variant"),
        }

        let lsp_data = LspResultData {
            responses: vec![],
            latencies: HashMap::new(),
            errors: vec!["test error".to_string()],
        };
        let stage_data_lsp = StageData::LspResults(lsp_data);

        match stage_data_lsp {
            StageData::LspResults(data) => {
                assert_eq!(data.responses.len(), 0);
                assert_eq!(data.errors.len(), 1);
            }
            _ => panic!("Expected LspResults variant"),
        }
    }

    #[test]
    fn test_stage_execution_metrics() {
        let metrics = StageExecutionMetrics {
            execution_time_ms: 150.5,
            memory_usage_mb: 25.3,
            items_processed: 42,
            success: true,
        };

        assert_eq!(metrics.execution_time_ms, 150.5);
        assert_eq!(metrics.memory_usage_mb, 25.3);
        assert_eq!(metrics.items_processed, 42);
        assert!(metrics.success);
    }

    #[test]
    fn test_stage_metrics_new() {
        let metrics = StageMetrics::new();
        
        assert_eq!(metrics.total_executions, 0);
        assert_eq!(metrics.successful_executions, 0);
        assert_eq!(metrics.failed_executions, 0);
        assert_eq!(metrics.average_latency_ms, 0.0);
        assert_eq!(metrics.p95_latency_ms, 0.0);
        assert_eq!(metrics.average_memory_mb, 0.0);
    }

    #[test]
    fn test_stage_metrics_update_latency() {
        let mut metrics = StageMetrics::new();
        
        // First update (total_executions = 0, so should set average directly)
        metrics.total_executions = 1;
        metrics.update_latency(100.0);
        assert_eq!(metrics.average_latency_ms, 100.0);
        assert_eq!(metrics.p95_latency_ms, 120.0); // 100.0 * 1.2

        // Second update (total_executions > 0, so should calculate average)
        metrics.total_executions = 2;
        metrics.update_latency(200.0);
        assert_eq!(metrics.average_latency_ms, 150.0); // (100.0 * 1 + 200.0) / 2
        assert_eq!(metrics.p95_latency_ms, 180.0); // 150.0 * 1.2
    }

    #[test]
    fn test_health_status() {
        let mut checks = HashMap::new();
        checks.insert("database".to_string(), true);
        checks.insert("cache".to_string(), false);

        let health = HealthStatus {
            healthy: false,
            message: "Cache is down".to_string(),
            checks,
        };

        assert!(!health.healthy);
        assert_eq!(health.message, "Cache is down");
        assert_eq!(health.checks.len(), 2);
        assert_eq!(health.checks.get("database"), Some(&true));
        assert_eq!(health.checks.get("cache"), Some(&false));
    }

    #[tokio::test]
    async fn test_query_preprocessing_stage_creation() {
        let stage = QueryPreprocessingStage::new();
        assert_eq!(stage.name(), "query_preprocessing");
        
        let metrics = stage.metrics();
        assert_eq!(metrics.total_executions, 0);
    }

    #[tokio::test] 
    async fn test_query_preprocessing_stage_classify_query() {
        let stage = QueryPreprocessingStage::new();
        
        // Test exact query
        assert_eq!(stage.classify_query("\"exact match\""), QueryType::Exact);
        
        // Test structural query
        assert_eq!(stage.classify_query("class MyClass"), QueryType::Structural);
        assert_eq!(stage.classify_query("function test"), QueryType::Structural);
        assert_eq!(stage.classify_query("interface ITest"), QueryType::Structural);
        
        // Test fuzzy query
        assert_eq!(stage.classify_query("test*"), QueryType::Fuzzy);
        assert_eq!(stage.classify_query("test?query"), QueryType::Fuzzy);
        
        // Test semantic query (more than 3 words)
        assert_eq!(stage.classify_query("find all function definitions that return string"), QueryType::Semantic);
        
        // Test symbol query (alphanumeric + underscore only)
        assert_eq!(stage.classify_query("test_function"), QueryType::Symbol);
        
        // Test hybrid query (default)
        assert_eq!(stage.classify_query("test-query"), QueryType::Hybrid);
    }

    #[tokio::test]
    async fn test_query_preprocessing_stage_extract_filters() {
        let stage = QueryPreprocessingStage::new();
        
        // Test extension filter
        let (query, filters) = stage.extract_filters("test query ext:ts");
        assert_eq!(query, "test query");
        assert_eq!(filters.file_extensions, vec![".ts"]);
        
        // Test language filter
        let (query, filters) = stage.extract_filters("test query lang:typescript");
        assert_eq!(query, "test query");
        assert_eq!(filters.language_filter, Some("typescript".to_string()));
        
        // Test both filters
        let (query, filters) = stage.extract_filters("test ext:rs lang:rust query");
        assert_eq!(query, "test query");
        assert_eq!(filters.file_extensions, vec![".rs"]);
        assert_eq!(filters.language_filter, Some("rust".to_string()));
        
        // Test no filters
        let (query, filters) = stage.extract_filters("just a regular query");
        assert_eq!(query, "just a regular query");
        assert_eq!(filters.file_extensions.len(), 0);
        assert_eq!(filters.language_filter, None);
    }

    #[tokio::test]
    async fn test_query_preprocessing_stage_process() {
        let stage = QueryPreprocessingStage::new();
        let context = create_test_context();
        
        let input = StageInput {
            query: "test ext:rs function".to_string(),
            query_id: "test-123".to_string(),
            context,
            data: StageData::Query(create_test_query_data()),
        };

        let output = stage.process(input).await.unwrap();
        
        match output.data {
            StageData::Query(query_data) => {
                assert_eq!(query_data.original_query, "test ext:rs function");
                assert_eq!(query_data.processed_query, "test function");
                assert_eq!(query_data.query_type, QueryType::Hybrid); // Contains "function" but not "function " pattern
                assert_eq!(query_data.filters.file_extensions, vec![".rs"]);
            }
            _ => panic!("Expected Query data"),
        }

        assert!(output.metrics.success);
        assert!(output.metrics.execution_time_ms > 0.0);
        assert_eq!(output.errors.len(), 0);
    }

    #[tokio::test]
    async fn test_query_preprocessing_stage_health_check() {
        let stage = QueryPreprocessingStage::new();
        let health = stage.health_check().await;
        
        assert!(health.healthy);
        assert_eq!(health.message, "Query preprocessing stage is healthy");
        assert_eq!(health.checks.len(), 2);
        assert_eq!(health.checks.get("regex_engine"), Some(&true));
        assert_eq!(health.checks.get("classification"), Some(&true));
    }

    #[tokio::test]
    async fn test_lsp_search_stage_creation() {
        let stage = LspSearchStage::new();
        assert_eq!(stage.name(), "lsp_search");
        
        let metrics = stage.metrics();
        assert_eq!(metrics.total_executions, 0);
    }

    #[tokio::test]
    async fn test_lsp_search_stage_health_check() {
        let stage = LspSearchStage::new();
        let health = stage.health_check().await;
        
        // Should be healthy even with no clients (empty checks)
        assert!(health.healthy);
        assert_eq!(health.message, "All LSP clients are healthy");
    }

    #[tokio::test]
    async fn test_text_search_stage_creation() {
        let stage = TextSearchStage::new();
        assert_eq!(stage.name(), "text_search");
        
        let metrics = stage.metrics();
        assert_eq!(metrics.total_executions, 0);
    }

    #[tokio::test]
    async fn test_text_search_stage_health_check() {
        let stage = TextSearchStage::new();
        let health = stage.health_check().await;
        
        // Should be unhealthy without index
        assert!(!health.healthy);
        assert_eq!(health.message, "Text search index is not available");
        assert_eq!(health.checks.get("index_available"), Some(&false));
    }

    #[tokio::test]
    async fn test_text_search_stage_process() {
        let stage = TextSearchStage::new();
        let context = create_test_context();
        
        let input = StageInput {
            query: "test".to_string(),
            query_id: "test-123".to_string(),
            context,
            data: StageData::Query(create_test_query_data()),
        };

        let output = stage.process(input).await.unwrap();
        
        match output.data {
            StageData::SearchResults(search_data) => {
                assert_eq!(search_data.results.len(), 0); // TODO implementation
                assert_eq!(search_data.total_matches, 0);
                assert!(search_data.search_time_ms >= 0.0); // Can be 0 for TODO implementation
            }
            _ => panic!("Expected SearchResults data"),
        }

        assert!(output.metrics.success);
        assert_eq!(output.errors.len(), 0);
    }

    #[test]
    fn test_lsp_search_options() {
        let options = LspSearchOptions {
            timeout: Duration::from_secs(10),
            max_results: 50,
            language_hint: Some("rust".to_string()),
        };

        assert_eq!(options.timeout, Duration::from_secs(10));
        assert_eq!(options.max_results, 50);
        assert_eq!(options.language_hint, Some("rust".to_string()));
    }
}