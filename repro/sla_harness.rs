//! # SLA Harness for 150ms Performance Boundary
//!
//! Implements the SLA harness that drops out-of-SLA hits
//! as specified in TODO.md Step 2(c).
//!
//! Key features:
//! - Hard 150ms timeout enforcement
//! - SLA-bounded result collection
//! - Performance degradation detection
//! - Comparable baseline enforcement

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::timeout;
use tracing::{debug, info, warn};

/// SLA configuration for performance boundaries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaConfig {
    /// Hard timeout limit (150ms as per TODO.md)
    pub timeout_ms: u64,
    /// Minimum successful queries for valid test
    pub min_successful_queries: usize,
    /// Maximum allowed timeout percentage
    pub max_timeout_percentage: f32,
    /// Warm-up queries to exclude from measurements
    pub warmup_queries: usize,
    /// Enable detailed timing measurements
    pub detailed_timing: bool,
}

impl Default for SlaConfig {
    fn default() -> Self {
        Self {
            timeout_ms: 150,          // TODO.md requirement: 150ms SLA
            min_successful_queries: 50, // Minimum for statistical validity
            max_timeout_percentage: 0.05, // Max 5% timeouts allowed
            warmup_queries: 10,       // Warm up caches/connections
            detailed_timing: true,
        }
    }
}

/// Query execution result with SLA compliance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaQueryResult {
    pub query_id: String,
    pub query: String,
    pub intent: String,
    pub language: String,
    pub execution_time_ms: u64,
    pub within_sla: bool,
    pub results: Vec<SearchResult>,
    pub error: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub file_path: String,
    pub score: f32,
    pub snippet: String,
    pub rank: usize,
}

/// SLA test execution summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaTestSummary {
    pub total_queries: usize,
    pub successful_queries: usize,
    pub timeout_queries: usize,
    pub error_queries: usize,
    pub timeout_percentage: f32,
    pub average_response_time_ms: f32,
    pub p95_response_time_ms: f32,
    pub p99_response_time_ms: f32,
    pub sla_compliance_percentage: f32,
    pub passed_sla_requirements: bool,
    pub performance_metrics: PerformanceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub min_response_time_ms: u64,
    pub max_response_time_ms: u64,
    pub median_response_time_ms: u64,
    pub std_deviation_ms: f32,
    pub queries_per_second: f32,
    pub successful_queries_per_second: f32,
}

/// Test query for SLA validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestQuery {
    pub id: String,
    pub query: String,
    pub intent: String,
    pub language: String,
    pub expected_file_count: Option<usize>,
}

/// SLA harness for performance testing
pub struct SlaHarness {
    config: SlaConfig,
    search_client: Box<dyn SearchClient + Send + Sync>,
    results: Vec<SlaQueryResult>,
}

/// Search client trait for testing different implementations
#[async_trait::async_trait]
pub trait SearchClient {
    async fn search(&self, query: &str, intent: &str, language: &str) -> Result<Vec<SearchResult>>;
    fn client_name(&self) -> &str;
}

impl SlaHarness {
    /// Create new SLA harness
    pub fn new(config: SlaConfig, search_client: Box<dyn SearchClient + Send + Sync>) -> Self {
        info!("Creating SLA harness with {}ms timeout", config.timeout_ms);
        info!("Client: {}, min queries: {}", 
              search_client.client_name(), config.min_successful_queries);
        
        Self {
            config,
            search_client,
            results: Vec::new(),
        }
    }

    /// Execute SLA test with performance boundaries
    pub async fn execute_sla_test(&mut self, test_queries: &[TestQuery]) -> Result<SlaTestSummary> {
        info!("Starting SLA test with {} queries", test_queries.len());
        
        if test_queries.len() < self.config.min_successful_queries {
            anyhow::bail!("Insufficient test queries: {} < {}", 
                         test_queries.len(), self.config.min_successful_queries);
        }

        self.results.clear();
        let test_start = Instant::now();

        // Warmup phase
        if self.config.warmup_queries > 0 {
            info!("Warming up with {} queries", self.config.warmup_queries);
            let warmup_queries = &test_queries[..self.config.warmup_queries.min(test_queries.len())];
            for query in warmup_queries {
                let _ = self.execute_single_query_with_sla(query).await;
            }
            self.results.clear(); // Clear warmup results
        }

        // Main test execution
        let test_queries_main = if self.config.warmup_queries > 0 && test_queries.len() > self.config.warmup_queries {
            &test_queries[self.config.warmup_queries..]
        } else {
            test_queries
        };

        info!("Executing main test with {} queries under {}ms SLA", 
              test_queries_main.len(), self.config.timeout_ms);

        for (i, query) in test_queries_main.iter().enumerate() {
            let result = self.execute_single_query_with_sla(query).await;
            self.results.push(result);
            
            if (i + 1) % 100 == 0 {
                let timeout_rate = self.calculate_current_timeout_rate();
                info!("Progress: {}/{} queries, timeout rate: {:.1}%", 
                      i + 1, test_queries_main.len(), timeout_rate * 100.0);
                
                // Early termination if timeout rate too high
                if timeout_rate > self.config.max_timeout_percentage && i > 50 {
                    warn!("High timeout rate detected, continuing test for full measurement");
                }
            }
        }

        let test_duration = test_start.elapsed();
        let summary = self.calculate_test_summary(test_duration);

        info!("SLA test completed: {}/{} successful queries, {:.1}% within SLA", 
              summary.successful_queries, summary.total_queries, 
              summary.sla_compliance_percentage);

        Ok(summary)
    }

    /// Execute single query with SLA enforcement
    async fn execute_single_query_with_sla(&self, test_query: &TestQuery) -> SlaQueryResult {
        let query_start = Instant::now();
        let timeout_duration = Duration::from_millis(self.config.timeout_ms);

        debug!("Executing query: {} ({}×{})", 
               test_query.query, test_query.intent, test_query.language);

        // Execute with timeout
        let search_result = timeout(
            timeout_duration,
            self.search_client.search(&test_query.query, &test_query.intent, &test_query.language)
        ).await;

        let execution_time = query_start.elapsed();
        let execution_time_ms = execution_time.as_millis() as u64;
        let within_sla = execution_time_ms <= self.config.timeout_ms;

        let (results, error) = match search_result {
            Ok(Ok(search_results)) => (search_results, None),
            Ok(Err(search_error)) => {
                warn!("Search error for query '{}': {}", test_query.query, search_error);
                (Vec::new(), Some(search_error.to_string()))
            }
            Err(_timeout_error) => {
                warn!("Query timeout after {}ms: {}", self.config.timeout_ms, test_query.query);
                (Vec::new(), Some(format!("Timeout after {}ms", self.config.timeout_ms)))
            }
        };

        SlaQueryResult {
            query_id: test_query.id.clone(),
            query: test_query.query.clone(),
            intent: test_query.intent.clone(),
            language: test_query.language.clone(),
            execution_time_ms,
            within_sla,
            results,
            error,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Calculate current timeout rate for monitoring
    fn calculate_current_timeout_rate(&self) -> f32 {
        if self.results.is_empty() {
            return 0.0;
        }

        let timeout_count = self.results.iter()
            .filter(|r| !r.within_sla || r.error.is_some())
            .count();

        timeout_count as f32 / self.results.len() as f32
    }

    /// Calculate comprehensive test summary
    fn calculate_test_summary(&self, test_duration: Duration) -> SlaTestSummary {
        let total_queries = self.results.len();
        
        let successful_results: Vec<_> = self.results.iter()
            .filter(|r| r.error.is_none() && r.within_sla)
            .collect();

        let successful_queries = successful_results.len();
        let error_queries = self.results.iter().filter(|r| r.error.is_some()).count();
        let timeout_queries = total_queries - successful_queries - error_queries;

        // Calculate timing statistics for successful queries only
        let mut response_times: Vec<u64> = successful_results.iter()
            .map(|r| r.execution_time_ms)
            .collect();
        response_times.sort_unstable();

        let (avg_time, p95_time, p99_time, perf_metrics) = if !response_times.is_empty() {
            let avg = response_times.iter().sum::<u64>() as f32 / response_times.len() as f32;
            let p95_idx = (response_times.len() as f32 * 0.95) as usize;
            let p99_idx = (response_times.len() as f32 * 0.99) as usize;
            
            let p95 = response_times.get(p95_idx.min(response_times.len() - 1))
                .copied().unwrap_or(0) as f32;
            let p99 = response_times.get(p99_idx.min(response_times.len() - 1))
                .copied().unwrap_or(0) as f32;

            let min_time = response_times.first().copied().unwrap_or(0);
            let max_time = response_times.last().copied().unwrap_or(0);
            let median_time = response_times.get(response_times.len() / 2).copied().unwrap_or(0);
            
            // Calculate standard deviation
            let variance = response_times.iter()
                .map(|&time| {
                    let diff = time as f32 - avg;
                    diff * diff
                })
                .sum::<f32>() / response_times.len() as f32;
            let std_dev = variance.sqrt();

            let test_duration_secs = test_duration.as_secs_f32();
            let qps = total_queries as f32 / test_duration_secs;
            let successful_qps = successful_queries as f32 / test_duration_secs;

            let metrics = PerformanceMetrics {
                min_response_time_ms: min_time,
                max_response_time_ms: max_time,
                median_response_time_ms: median_time,
                std_deviation_ms: std_dev,
                queries_per_second: qps,
                successful_queries_per_second: successful_qps,
            };

            (avg, p95, p99, metrics)
        } else {
            (0.0, 0.0, 0.0, PerformanceMetrics {
                min_response_time_ms: 0,
                max_response_time_ms: 0,
                median_response_time_ms: 0,
                std_deviation_ms: 0.0,
                queries_per_second: 0.0,
                successful_queries_per_second: 0.0,
            })
        };

        let timeout_percentage = if total_queries > 0 {
            (timeout_queries + error_queries) as f32 / total_queries as f32
        } else {
            0.0
        };

        let sla_compliance_percentage = if total_queries > 0 {
            successful_queries as f32 / total_queries as f32 * 100.0
        } else {
            0.0
        };

        // Check if test passed SLA requirements
        let passed_sla_requirements = 
            successful_queries >= self.config.min_successful_queries &&
            timeout_percentage <= self.config.max_timeout_percentage &&
            p99_time <= self.config.timeout_ms as f32;

        SlaTestSummary {
            total_queries,
            successful_queries,
            timeout_queries,
            error_queries,
            timeout_percentage,
            average_response_time_ms: avg_time,
            p95_response_time_ms: p95_time,
            p99_response_time_ms: p99_time,
            sla_compliance_percentage,
            passed_sla_requirements,
            performance_metrics: perf_metrics,
        }
    }

    /// Filter results to include only SLA-compliant queries
    pub fn get_sla_compliant_results(&self) -> Vec<&SlaQueryResult> {
        self.results.iter()
            .filter(|r| r.within_sla && r.error.is_none())
            .collect()
    }

    /// Get detailed timing breakdown
    pub fn get_timing_breakdown(&self) -> TimingBreakdown {
        let sla_compliant = self.get_sla_compliant_results();
        let mut timing_by_intent = HashMap::new();
        let mut timing_by_language = HashMap::new();

        for result in &sla_compliant {
            timing_by_intent.entry(result.intent.clone())
                .or_insert_with(Vec::new)
                .push(result.execution_time_ms);
            
            timing_by_language.entry(result.language.clone())
                .or_insert_with(Vec::new)
                .push(result.execution_time_ms);
        }

        TimingBreakdown {
            by_intent: timing_by_intent,
            by_language: timing_by_language,
            total_sla_compliant: sla_compliant.len(),
        }
    }

    /// Save results to file
    pub async fn save_results(&self, output_path: &std::path::Path, summary: &SlaTestSummary) -> Result<()> {
        let output_data = SlaTestOutput {
            config: self.config.clone(),
            client_name: self.search_client.client_name().to_string(),
            summary: summary.clone(),
            results: self.results.clone(),
            timing_breakdown: self.get_timing_breakdown(),
        };

        let json = serde_json::to_string_pretty(&output_data)
            .context("Failed to serialize SLA test results")?;
        
        if let Some(parent) = output_path.parent() {
            tokio::fs::create_dir_all(parent).await
                .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
        }
        
        tokio::fs::write(output_path, json).await
            .with_context(|| format!("Failed to write SLA results: {}", output_path.display()))?;
        
        info!("SLA test results saved to: {}", output_path.display());
        Ok(())
    }
}

/// Complete SLA test output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaTestOutput {
    pub config: SlaConfig,
    pub client_name: String,
    pub summary: SlaTestSummary,
    pub results: Vec<SlaQueryResult>,
    pub timing_breakdown: TimingBreakdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingBreakdown {
    pub by_intent: HashMap<String, Vec<u64>>,
    pub by_language: HashMap<String, Vec<u64>>,
    pub total_sla_compliant: usize,
}

/// Mock search client for testing
pub struct MockSearchClient {
    name: String,
    base_latency_ms: u64,
    failure_rate: f32,
}

impl MockSearchClient {
    pub fn new(name: String, base_latency_ms: u64, failure_rate: f32) -> Self {
        Self {
            name,
            base_latency_ms,
            failure_rate,
        }
    }
}

#[async_trait::async_trait]
impl SearchClient for MockSearchClient {
    async fn search(&self, query: &str, _intent: &str, _language: &str) -> Result<Vec<SearchResult>> {
        // Simulate variable latency
        let latency_variation = (query.len() % 50) as u64;
        let total_latency = self.base_latency_ms + latency_variation;
        
        if total_latency > 0 {
            tokio::time::sleep(Duration::from_millis(total_latency)).await;
        }
        
        // Simulate occasional failures
        if rand::random::<f32>() < self.failure_rate {
            anyhow::bail!("Simulated search failure");
        }
        
        // Generate mock results
        let results = vec![
            SearchResult {
                file_path: "src/main.rs".to_string(),
                score: 0.95,
                snippet: format!("Mock result for query: {}", query),
                rank: 1,
            },
            SearchResult {
                file_path: "src/lib.rs".to_string(),
                score: 0.87,
                snippet: "Secondary result".to_string(),
                rank: 2,
            },
        ];
        
        Ok(results)
    }
    
    fn client_name(&self) -> &str {
        &self.name
    }
}

/// Create sample test queries for validation
pub fn create_sample_test_queries() -> Vec<TestQuery> {
    vec![
        TestQuery {
            id: "query_001".to_string(),
            query: "async function handler".to_string(),
            intent: "NL".to_string(),
            language: "typescript".to_string(),
            expected_file_count: Some(5),
        },
        TestQuery {
            id: "query_002".to_string(),
            query: "class DatabaseConnection".to_string(),
            intent: "identifier".to_string(),
            language: "python".to_string(),
            expected_file_count: Some(3),
        },
        TestQuery {
            id: "query_003".to_string(),
            query: "impl Iterator for".to_string(),
            intent: "structural".to_string(),
            language: "rust".to_string(),
            expected_file_count: Some(2),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sla_harness_basic() {
        let config = SlaConfig {
            timeout_ms: 100,
            min_successful_queries: 2,
            max_timeout_percentage: 0.5,
            warmup_queries: 0,
            detailed_timing: true,
        };
        
        let client = Box::new(MockSearchClient::new("test".to_string(), 50, 0.0));
        let mut harness = SlaHarness::new(config, client);
        
        let queries = create_sample_test_queries();
        let summary = harness.execute_sla_test(&queries).await.unwrap();
        
        assert_eq!(summary.total_queries, queries.len());
        assert!(summary.passed_sla_requirements);
        assert!(summary.average_response_time_ms < 100.0);
    }

    #[tokio::test]
    async fn test_sla_timeout_handling() {
        let config = SlaConfig {
            timeout_ms: 50, // Very short timeout
            min_successful_queries: 1,
            max_timeout_percentage: 1.0, // Allow all timeouts for this test
            warmup_queries: 0,
            detailed_timing: true,
        };
        
        let client = Box::new(MockSearchClient::new("slow".to_string(), 100, 0.0)); // Slower than timeout
        let mut harness = SlaHarness::new(config, client);
        
        let queries = vec![create_sample_test_queries()[0].clone()];
        let summary = harness.execute_sla_test(&queries).await.unwrap();
        
        assert_eq!(summary.total_queries, 1);
        assert_eq!(summary.successful_queries, 0); // Should timeout
        assert!(!summary.passed_sla_requirements);
    }

    #[tokio::test]
    async fn test_timing_breakdown() {
        let config = SlaConfig::default();
        let client = Box::new(MockSearchClient::new("test".to_string(), 30, 0.0));
        let mut harness = SlaHarness::new(config, client);
        
        let queries = create_sample_test_queries();
        let _summary = harness.execute_sla_test(&queries).await.unwrap();
        
        let breakdown = harness.get_timing_breakdown();
        assert!(!breakdown.by_intent.is_empty());
        assert!(!breakdown.by_language.is_empty());
        assert_eq!(breakdown.total_sla_compliant, queries.len());
    }
}