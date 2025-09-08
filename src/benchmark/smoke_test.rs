use std::time::{Duration, Instant};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error, debug, instrument};
use tokio::time::timeout;

use crate::search::{SearchEngine, SearchRequest, SearchResponse};
use super::{BenchmarkConfig, BenchmarkResult, GoldenQuery, QuerySlice, MetricsCalculator};

/// SMOKE test suite for quick validation
pub struct SmokeTestSuite {
    config: BenchmarkConfig,
    search_engine: SearchEngine,
}

impl SmokeTestSuite {
    /// Create a new SMOKE test suite
    pub fn new(config: BenchmarkConfig, search_engine: SearchEngine) -> Self {
        Self {
            config,
            search_engine,
        }
    }

    /// Run SMOKE tests with default configuration
    #[instrument(skip(self))]
    pub async fn run_smoke_tests(&self) -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
        info!("🚬 Starting SMOKE test suite");
        
        // Load SMOKE test queries (subset of golden dataset)
        let smoke_queries = self.load_smoke_queries().await?;
        info!("📋 Loaded {} SMOKE test queries", smoke_queries.len());

        let mut results = Vec::new();

        // Test each system configuration
        for system_config in &self.config.systems {
            info!("🔧 Testing system: {}", system_config.name);
            
            let system_results = self.run_system_smoke_tests(system_config, &smoke_queries).await?;
            results.extend(system_results);
        }

        info!("✅ SMOKE tests completed: {} total results", results.len());
        Ok(results)
    }

    /// Run SMOKE tests for a specific system configuration
    #[instrument(skip(self, queries))]
    async fn run_system_smoke_tests(
        &self,
        system_config: &super::SystemConfig,
        queries: &[GoldenQuery],
    ) -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();

        for query in queries {
            let start_time = Instant::now();
            
            // Configure search engine for this system
            let search_request = SearchRequest {
                query: query.query.clone(),
                file_path: None,
                language: None,
                max_results: 50,
                include_context: true,
                timeout_ms: 2000, // 2 second SLA timeout
                enable_lsp: system_config.enable_lsp,
                search_types: vec![crate::search::SearchResultType::TextMatch],
                search_method: Some(crate::search::SearchMethod::Hybrid),
            };

            // Execute search with timeout
            let search_result = match timeout(
                Duration::from_millis(2000),
                self.search_engine.search_comprehensive(search_request)
            ).await {
                Ok(Ok(response)) => {
                    let elapsed = start_time.elapsed();
                    let latency_ms = elapsed.as_millis() as u64;
                    
                    // Calculate metrics
                    let predicted_files: Vec<String> = response.results
                        .iter()
                        .map(|r| r.file_path.clone())
                        .collect();
                    
                    let success_at_10 = MetricsCalculator::calculate_success_at_10(
                        &predicted_files,
                        &query.expected_files
                    );
                    
                    let ndcg_at_10 = MetricsCalculator::calculate_ndcg_at_10(
                        &predicted_files,
                        &query.expected_files
                    );
                    
                    let sla_compliant = latency_ms <= 2000;
                    let sla_recall_at_50 = MetricsCalculator::calculate_sla_recall_at_50(
                        &predicted_files,
                        &query.expected_files,
                        sla_compliant
                    );

                    BenchmarkResult {
                        system_name: system_config.name.clone(),
                        query_id: query.id.clone(),
                        query_text: query.query.clone(),
                        success_at_10,
                        ndcg_at_10,
                        sla_recall_at_50,
                        latency_ms,
                        sla_compliant,
                        lsp_routed: system_config.enable_lsp && response.lsp_response.is_some(),
                        results_count: response.results.len() as u32,
                        error: None,
                    }
                },
                Ok(Err(e)) => {
                    warn!("Search error for query {}: {}", query.id, e);
                    BenchmarkResult {
                        system_name: system_config.name.clone(),
                        query_id: query.id.clone(),
                        query_text: query.query.clone(),
                        success_at_10: 0.0,
                        ndcg_at_10: 0.0,
                        sla_recall_at_50: 0.0,
                        latency_ms: start_time.elapsed().as_millis() as u64,
                        sla_compliant: false,
                        lsp_routed: false,
                        results_count: 0,
                        error: Some(e.to_string()),
                    }
                },
                Err(_) => {
                    warn!("Query {} timed out", query.id);
                    BenchmarkResult {
                        system_name: system_config.name.clone(),
                        query_id: query.id.clone(),
                        query_text: query.query.clone(),
                        success_at_10: 0.0,
                        ndcg_at_10: 0.0,
                        sla_recall_at_50: 0.0,
                        latency_ms: 2000, // Timeout value
                        sla_compliant: false,
                        lsp_routed: false,
                        results_count: 0,
                        error: Some("Query timeout".to_string()),
                    }
                },
            };

            results.push(search_result);
        }

        Ok(results)
    }

    /// Load SMOKE test queries (stratified sample)
    #[instrument(skip(self))]
    async fn load_smoke_queries(&self) -> Result<Vec<GoldenQuery>, Box<dyn std::error::Error>> {
        // For SMOKE tests, we want a representative sample
        // In a real implementation, this would load from the golden dataset
        // and apply stratified sampling
        
        Ok(vec![
            GoldenQuery {
                id: "smoke_001".to_string(),
                query: "function".to_string(),
                expected_files: vec!["test-file.js".to_string()],
                expected_symbols: vec!["testFunction".to_string()],
                query_type: super::QueryType::Identifier,
                language: Some("javascript".to_string()),
                difficulty: super::QueryDifficulty::Easy,
                slice: QuerySlice::SmokeDefault,
            },
            GoldenQuery {
                id: "smoke_002".to_string(),
                query: "class".to_string(),
                expected_files: vec!["test-file.js".to_string()],
                expected_symbols: vec!["TestClass".to_string()],
                query_type: super::QueryType::Identifier,
                language: Some("javascript".to_string()),
                difficulty: super::QueryDifficulty::Easy,
                slice: QuerySlice::SmokeDefault,
            },
            GoldenQuery {
                id: "smoke_003".to_string(),
                query: "test".to_string(),
                expected_files: vec!["test-file.js".to_string()],
                expected_symbols: vec![],
                query_type: super::QueryType::ExactMatch,
                language: None,
                difficulty: super::QueryDifficulty::Easy,
                slice: QuerySlice::SmokeDefault,
            },
        ])
    }

    /// Run quick health check
    #[instrument(skip(self))]
    pub async fn health_check(&self) -> Result<bool, Box<dyn std::error::Error>> {
        info!("🏥 Running SMOKE health check");

        // Simple search to verify the system is operational
        let search_request = SearchRequest {
            query: "test".to_string(),
            file_path: None,
            language: None,
            max_results: 5,
            include_context: true,
            timeout_ms: 1000,
            enable_lsp: false,
            search_types: vec![crate::search::SearchResultType::TextMatch],
            search_method: Some(crate::search::SearchMethod::Hybrid),
        };

        let start_time = Instant::now();
        match timeout(Duration::from_millis(1000), self.search_engine.search_comprehensive(search_request)).await {
            Ok(Ok(_response)) => {
                let elapsed = start_time.elapsed();
                info!("✅ Health check passed: {}ms", elapsed.as_millis());
                Ok(true)
            },
            Ok(Err(e)) => {
                error!("❌ Health check failed: {}", e);
                Ok(false)
            },
            Err(_) => {
                error!("❌ Health check timed out");
                Ok(false)
            }
        }
    }

    /// Validate SMOKE test configuration
    pub fn validate_config(&self) -> Result<(), String> {
        if self.config.systems.is_empty() {
            return Err("No systems configured for testing".to_string());
        }

        if !self.config.systems.iter().any(|s| s.baseline) {
            return Err("No baseline system configured".to_string());
        }

        Ok(())
    }
}

/// SMOKE test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmokeTestConfig {
    pub max_queries: u32,
    pub timeout_ms: u64,
    pub stratified_sample: bool,
    pub quick_mode: bool,
}

impl Default for SmokeTestConfig {
    fn default() -> Self {
        Self {
            max_queries: 10,         // Small sample for quick testing
            timeout_ms: 2000,        // 2 second SLA
            stratified_sample: true, // Representative sample
            quick_mode: false,       // Full metrics calculation
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::BenchmarkConfig;

    #[tokio::test]
    async fn test_smoke_config_validation() {
        let config = BenchmarkConfig::default();
        let temp_dir = tempfile::TempDir::new().unwrap();
        let search_engine = SearchEngine::new(temp_dir.path()).await.unwrap();
        let smoke_suite = SmokeTestSuite::new(config, search_engine);
        
        assert!(smoke_suite.validate_config().is_ok());
    }

    #[test]
    fn test_smoke_config_default() {
        let config = SmokeTestConfig::default();
        assert_eq!(config.max_queries, 10);
        assert_eq!(config.timeout_ms, 2000);
        assert!(config.stratified_sample);
        assert!(!config.quick_mode);
    }
}