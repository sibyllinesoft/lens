use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::fs;
use tokio::sync::Semaphore;
use serde_json;
use tracing::{info, warn, error, debug, instrument};
use futures::stream::{FuturesUnordered, StreamExt};

use super::{
    BenchmarkConfig, BenchmarkResult, BenchmarkResults, BenchmarkSummary,
    SystemSummary, GoldenQuery, MetricsCalculator, GateResult,
    PerformanceGates, SystemConfig,
};
use crate::search::{SearchEngine, SearchRequest};
use crate::metrics::MetricsCollector;

/// Main benchmark orchestrator
pub struct BenchmarkRunner {
    search_engine: Arc<SearchEngine>,
    metrics_collector: Arc<MetricsCollector>,
    config: BenchmarkConfig,
    concurrency_limit: Arc<Semaphore>,
}

impl BenchmarkRunner {
    pub fn new(
        search_engine: Arc<SearchEngine>,
        metrics_collector: Arc<MetricsCollector>,
        config: BenchmarkConfig,
    ) -> Self {
        let concurrency_limit = Arc::new(Semaphore::new(10)); // Limit concurrent queries
        
        Self {
            search_engine,
            metrics_collector,
            config,
            concurrency_limit,
        }
    }

    /// Run a complete benchmark suite
    #[instrument(skip(self))]
    pub async fn run_benchmark(
        &self,
        dataset_name: &str,
        query_limit: Option<u32>,
        enable_smoke_test: bool,
    ) -> Result<BenchmarkResults, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        info!("Starting benchmark for dataset: {}", dataset_name);

        // Load dataset configuration
        let dataset_config = self.config.datasets.get(dataset_name)
            .ok_or_else(|| format!("Dataset '{}' not found in configuration", dataset_name))?;

        // Load golden queries
        let golden_queries = self.load_golden_queries(&dataset_config.golden_queries_path).await?;
        info!("Loaded {} golden queries", golden_queries.len());

        // Apply query limit and stratified sampling
        let test_queries = self.select_test_queries(golden_queries, query_limit, enable_smoke_test);
        info!("Selected {} queries for testing", test_queries.len());

        // Validate corpus consistency if enabled
        if self.config.validate_corpus {
            self.validate_corpus_consistency(&test_queries, &dataset_config.corpus_path).await?;
        }

        // Run benchmarks for all configured systems
        let mut all_results = Vec::new();
        let mut system_summaries = Vec::new();

        for system_config in &self.config.systems {
            info!("Testing system: {}", system_config.name);
            
            let system_results = self.run_system_benchmark(system_config, &test_queries).await?;
            let system_summary = self.calculate_system_summary(system_config, &system_results);
            
            all_results.extend(system_results);
            system_summaries.push(system_summary);
            
            info!(
                "System '{}' completed: avg_success@10={:.3}, p95_latency={}ms",
                system_config.name,
                system_summaries.last().unwrap().avg_success_at_10,
                system_summaries.last().unwrap().p95_latency_ms
            );
        }

        // Calculate overall summary and performance gate evaluation
        let overall_summary = self.calculate_overall_summary(&all_results, system_summaries)?;
        
        // Generate benchmark report
        let report_path = if self.config.generate_reports {
            Some(self.generate_report(&overall_summary, dataset_name).await?)
        } else {
            None
        };

        // Save results artifacts
        self.save_benchmark_artifacts(&all_results, &overall_summary, dataset_name).await?;

        let total_duration = start_time.elapsed();
        info!(
            "Benchmark completed in {:.2}s: queries={}, systems={}, gates_passed={}",
            total_duration.as_secs_f64(),
            test_queries.len(),
            self.config.systems.len(),
            overall_summary.passes_performance_gates
        );

        Ok(BenchmarkResults {
            results: all_results,
            summary: overall_summary,
            report_path,
            config_fingerprint: self.generate_config_fingerprint(),
            timestamp: chrono::Utc::now(),
        })
    }

    /// Load golden queries from JSON file
    async fn load_golden_queries(&self, path: &str) -> Result<Vec<GoldenQuery>, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path).await
            .map_err(|e| format!("Failed to read golden queries from '{}': {}", path, e))?;
        
        let queries: Vec<GoldenQuery> = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse golden queries: {}", e))?;
        
        debug!("Loaded {} golden queries from {}", queries.len(), path);
        Ok(queries)
    }

    /// Select test queries based on criteria and sampling strategy
    fn select_test_queries(
        &self,
        mut queries: Vec<GoldenQuery>,
        query_limit: Option<u32>,
        enable_smoke_test: bool,
    ) -> Vec<GoldenQuery> {
        // Filter for smoke test if requested
        if enable_smoke_test {
            queries.retain(|q| matches!(q.slice, super::QuerySlice::SmokeDefault));
        }

        // Apply query limit with stratified sampling
        if let Some(limit) = query_limit {
            let limit = limit as usize;
            if queries.len() > limit {
                if self.config.datasets.values().any(|d| d.stratified_sampling) {
                    queries = self.stratified_sample(queries, limit);
                } else {
                    queries.truncate(limit);
                }
            }
        }

        queries
    }

    /// Perform stratified sampling to ensure representative test coverage
    fn stratified_sample(&self, queries: Vec<GoldenQuery>, target_size: usize) -> Vec<GoldenQuery> {
        let mut by_type: HashMap<String, Vec<GoldenQuery>> = HashMap::new();
        
        // Group by query type and language
        for query in queries {
            let key = match (&query.query_type, &query.language) {
                (qt, Some(lang)) => format!("{:?}_{}", qt, lang),
                (qt, None) => format!("{:?}", qt),
            };
            by_type.entry(key).or_default().push(query);
        }

        let mut sampled = Vec::new();
        let strata_count = by_type.len();
        let per_stratum = target_size / strata_count;
        let remainder = target_size % strata_count;

        for (i, (_, mut stratum_queries)) in by_type.into_iter().enumerate() {
            let take_count = per_stratum + if i < remainder { 1 } else { 0 };
            let take_count = take_count.min(stratum_queries.len());
            
            stratum_queries.truncate(take_count);
            sampled.extend(stratum_queries);
        }

        debug!("Stratified sampling: {} strata, {} total queries", strata_count, sampled.len());
        sampled
    }

    /// Validate that corpus files exist for golden queries
    async fn validate_corpus_consistency(
        &self,
        queries: &[GoldenQuery],
        corpus_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut missing_files = Vec::new();
        
        for query in queries {
            for expected_file in &query.expected_files {
                let file_path = format!("{}/{}", corpus_path, expected_file);
                if !tokio::fs::try_exists(&file_path).await.unwrap_or(false) {
                    missing_files.push(expected_file.clone());
                }
            }
        }

        if !missing_files.is_empty() {
            warn!("Found {} missing corpus files", missing_files.len());
            // Don't fail the benchmark, just warn about consistency issues
        } else {
            info!("Corpus consistency validation passed");
        }

        Ok(())
    }

    /// Run benchmark for a specific system configuration
    #[instrument(skip(self, queries))]
    async fn run_system_benchmark(
        &self,
        system_config: &SystemConfig,
        queries: &[GoldenQuery],
    ) -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        
        // Create futures for all queries with concurrency control
        let futures: FuturesUnordered<_> = queries
            .iter()
            .map(|query| self.run_single_query(system_config, query))
            .collect();

        // Execute all queries and collect results
        let results: Vec<BenchmarkResult> = futures.collect().await;
        
        let duration = start_time.elapsed();
        info!(
            "System '{}' benchmark completed: {} queries in {:.2}s",
            system_config.name,
            results.len(),
            duration.as_secs_f64()
        );

        Ok(results)
    }

    /// Run a single query against the search engine
    async fn run_single_query(
        &self,
        system_config: &SystemConfig,
        golden_query: &GoldenQuery,
    ) -> BenchmarkResult {
        // Acquire semaphore permit for concurrency control
        let _permit = self.concurrency_limit.acquire().await.unwrap();
        
        let start_time = Instant::now();
        
        // Create search request with system configuration
        let search_request = SearchRequest {
            query: golden_query.query.clone(),
            file_path: None,
            language: golden_query.language.clone(),
            max_results: 50, // SLA-Recall@50
            include_context: true,
            timeout_ms: 150, // ≤150ms SLA per TODO.md
            enable_lsp: system_config.enable_lsp,
            search_types: vec![crate::search::SearchResultType::TextMatch],
            search_method: Some(crate::search::SearchMethod::Hybrid),
        };

        // Execute search
        match self.search_engine.search_comprehensive(search_request).await {
            Ok(response) => {
                let latency = start_time.elapsed();
                let sla_compliant = latency.as_millis() <= 150; // ≤150ms per TODO.md

                // Extract result file paths
                let predicted_files: Vec<String> = response.results
                    .iter()
                    .map(|r| r.file_path.clone())
                    .collect();

                // Calculate metrics
                let success_at_10 = MetricsCalculator::calculate_success_at_10(
                    &predicted_files,
                    &golden_query.expected_files
                );
                
                let ndcg_at_10 = MetricsCalculator::calculate_ndcg_at_10(
                    &predicted_files,
                    &golden_query.expected_files
                );

                let sla_recall_at_50 = MetricsCalculator::calculate_sla_recall_at_50(
                    &predicted_files,
                    &golden_query.expected_files,
                    sla_compliant
                );

                // Determine if LSP was used
                let lsp_routed = response.lsp_response.is_some();

                BenchmarkResult {
                    system_name: system_config.name.clone(),
                    query_id: golden_query.id.clone(),
                    query_text: golden_query.query.clone(),
                    success_at_10,
                    ndcg_at_10,
                    sla_recall_at_50,
                    latency_ms: latency.as_millis() as u64,
                    sla_compliant,
                    lsp_routed,
                    results_count: response.results.len() as u32,
                    error: None,
                }
            }
            Err(e) => {
                let latency = start_time.elapsed();
                error!("Query '{}' failed: {}", golden_query.query, e);

                BenchmarkResult {
                    system_name: system_config.name.clone(),
                    query_id: golden_query.id.clone(),
                    query_text: golden_query.query.clone(),
                    success_at_10: 0.0,
                    ndcg_at_10: 0.0,
                    sla_recall_at_50: 0.0,
                    latency_ms: latency.as_millis() as u64,
                    sla_compliant: false,
                    lsp_routed: false,
                    results_count: 0,
                    error: Some(e.to_string()),
                }
            }
        }
    }

    /// Calculate summary metrics for a system
    fn calculate_system_summary(
        &self,
        system_config: &SystemConfig,
        results: &[BenchmarkResult],
    ) -> SystemSummary {
        if results.is_empty() {
            return SystemSummary {
                system_name: system_config.name.clone(),
                total_queries: 0,
                successful_queries: 0,
                performance_gain_pp: 0.0,
                p95_latency_ms: 0,
                meets_sla: false,
                lsp_routing_percentage: 0.0,
                avg_success_at_10: 0.0,
                avg_ndcg_at_10: 0.0,
                avg_sla_recall_at_50: 0.0,
            };
        }

        let total_queries = results.len() as u32;
        let successful_queries = results.iter().filter(|r| r.error.is_none()).count() as u32;

        // Calculate averages
        let valid_results: Vec<_> = results.iter().filter(|r| r.error.is_none()).collect();
        let avg_success_at_10 = if valid_results.is_empty() {
            0.0
        } else {
            valid_results.iter().map(|r| r.success_at_10).sum::<f64>() / valid_results.len() as f64
        };
        
        let avg_ndcg_at_10 = if valid_results.is_empty() {
            0.0
        } else {
            valid_results.iter().map(|r| r.ndcg_at_10).sum::<f64>() / valid_results.len() as f64
        };

        let avg_sla_recall_at_50 = if valid_results.is_empty() {
            0.0
        } else {
            valid_results.iter().map(|r| r.sla_recall_at_50).sum::<f64>() / valid_results.len() as f64
        };

        // Calculate p95 latency
        let latencies: Vec<u64> = results.iter().map(|r| r.latency_ms).collect();
        let p95_latency_ms = MetricsCalculator::calculate_p95_latency(&latencies);

        // Calculate LSP routing percentage
        let lsp_queries = results.iter().filter(|r| r.lsp_routed).count();
        let lsp_routing_percentage = if total_queries > 0 {
            (lsp_queries as f64 / total_queries as f64) * 100.0
        } else {
            0.0
        };

        // Check SLA compliance
        let sla_compliant_count = results.iter().filter(|r| r.sla_compliant).count();
        let meets_sla = p95_latency_ms <= 150 && sla_compliant_count as f64 / total_queries as f64 >= 0.95;

        SystemSummary {
            system_name: system_config.name.clone(),
            total_queries,
            successful_queries,
            performance_gain_pp: 0.0, // Will be calculated relative to baseline
            p95_latency_ms,
            meets_sla,
            lsp_routing_percentage,
            avg_success_at_10,
            avg_ndcg_at_10,
            avg_sla_recall_at_50,
        }
    }

    /// Calculate overall benchmark summary with performance gate evaluation
    fn calculate_overall_summary(
        &self,
        results: &[BenchmarkResult],
        mut system_summaries: Vec<SystemSummary>,
    ) -> Result<BenchmarkSummary, Box<dyn std::error::Error>> {
        // Find baseline system for comparison
        let baseline_summary = system_summaries
            .iter()
            .find(|s| s.system_name == "baseline")
            .ok_or("No baseline system found for comparison")?
            .clone();

        // Calculate performance gains relative to baseline
        for summary in &mut system_summaries {
            if summary.system_name != "baseline" {
                summary.performance_gain_pp = MetricsCalculator::calculate_performance_gain_pp(
                    baseline_summary.avg_success_at_10,
                    summary.avg_success_at_10,
                );
            }
        }

        // Find the best performing system for gate evaluation
        let best_system = system_summaries
            .iter()
            .filter(|s| s.system_name != "baseline")
            .max_by(|a, b| a.performance_gain_pp.partial_cmp(&b.performance_gain_pp).unwrap())
            .unwrap_or(&baseline_summary);

        // Evaluate performance gates
        let gate_results = MetricsCalculator::evaluate_performance_gates(
            &self.config.performance_gates,
            &baseline_summary,
            best_system,
        );

        let passes_performance_gates = gate_results.iter().all(|g| g.passed);

        // Calculate overall metrics
        let total_queries = results.len() as u32;
        let successful_queries = results.iter().filter(|r| r.error.is_none()).count() as u32;

        let valid_results: Vec<_> = results.iter().filter(|r| r.error.is_none()).collect();
        let average_success_at_10 = if valid_results.is_empty() {
            0.0
        } else {
            valid_results.iter().map(|r| r.success_at_10).sum::<f64>() / valid_results.len() as f64
        };

        let average_ndcg_at_10 = if valid_results.is_empty() {
            0.0
        } else {
            valid_results.iter().map(|r| r.ndcg_at_10).sum::<f64>() / valid_results.len() as f64
        };

        let average_sla_recall_at_50 = if valid_results.is_empty() {
            0.0
        } else {
            valid_results.iter().map(|r| r.sla_recall_at_50).sum::<f64>() / valid_results.len() as f64
        };

        let all_latencies: Vec<u64> = results.iter().map(|r| r.latency_ms).collect();
        let average_latency_ms = if all_latencies.is_empty() {
            0
        } else {
            all_latencies.iter().sum::<u64>() / all_latencies.len() as u64
        };
        let p95_latency_ms = MetricsCalculator::calculate_p95_latency(&all_latencies);

        let sla_compliant_count = results.iter().filter(|r| r.sla_compliant).count();
        let sla_compliance_rate = if total_queries > 0 {
            sla_compliant_count as f64 / total_queries as f64
        } else {
            0.0
        };

        Ok(BenchmarkSummary {
            total_queries,
            successful_queries,
            average_success_at_10,
            average_ndcg_at_10,
            average_sla_recall_at_50,
            average_latency_ms,
            p95_latency_ms,
            sla_compliance_rate,
            system_summaries,
            passes_performance_gates,
            gate_analysis: gate_results,
        })
    }

    /// Generate human-readable benchmark report
    async fn generate_report(
        &self,
        summary: &BenchmarkSummary,
        dataset_name: &str,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let report_path = format!("{}/benchmark_report_{}_{}.md", self.config.output_path, dataset_name, timestamp);

        // Ensure output directory exists
        if let Some(parent) = std::path::Path::new(&report_path).parent() {
            fs::create_dir_all(parent).await?;
        }

        let mut report = String::new();
        report.push_str(&format!("# Benchmark Report: {}\n\n", dataset_name));
        report.push_str(&format!("**Generated**: {}\n", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
        report.push_str(&format!("**Dataset**: {}\n", dataset_name));
        report.push_str(&format!("**Total Queries**: {}\n\n", summary.total_queries));

        // Performance Gates Summary
        report.push_str("## Performance Gates\n\n");
        for gate in &summary.gate_analysis {
            let status = if gate.passed { "✅ PASS" } else { "❌ FAIL" };
            report.push_str(&format!(
                "- **{}**: {} (Target: {:.1}, Actual: {:.1}, Margin: {:.1})\n",
                gate.gate_name, status, gate.target_value, gate.actual_value, gate.margin
            ));
        }
        let overall_status = if summary.passes_performance_gates { "✅ PASS" } else { "❌ FAIL" };
        report.push_str(&format!("\n**Overall Gate Status**: {}\n\n", overall_status));

        // System Performance Summary
        report.push_str("## System Performance\n\n");
        report.push_str("| System | Success@10 | nDCG@10 | SLA-Recall@50 | P95 Latency | LSP Routing | Gain (pp) |\n");
        report.push_str("|--------|------------|---------|---------------|-------------|-------------|-----------||\n");
        
        for system in &summary.system_summaries {
            report.push_str(&format!(
                "| {} | {:.3} | {:.3} | {:.3} | {}ms | {:.1}% | {:.1} |\n",
                system.system_name,
                system.avg_success_at_10,
                system.avg_ndcg_at_10,
                system.avg_sla_recall_at_50,
                system.p95_latency_ms,
                system.lsp_routing_percentage,
                system.performance_gain_pp
            ));
        }

        // Overall Statistics
        report.push_str("\n## Overall Statistics\n\n");
        report.push_str(&format!("- **Total Queries**: {}\n", summary.total_queries));
        report.push_str(&format!("- **Successful Queries**: {}\n", summary.successful_queries));
        report.push_str(&format!("- **Average Success@10**: {:.3}\n", summary.average_success_at_10));
        report.push_str(&format!("- **Average nDCG@10**: {:.3}\n", summary.average_ndcg_at_10));
        report.push_str(&format!("- **Average SLA-Recall@50**: {:.3}\n", summary.average_sla_recall_at_50));
        report.push_str(&format!("- **Average Latency**: {}ms\n", summary.average_latency_ms));
        report.push_str(&format!("- **P95 Latency**: {}ms\n", summary.p95_latency_ms));
        report.push_str(&format!("- **SLA Compliance Rate**: {:.1}%\n", summary.sla_compliance_rate * 100.0));

        // Performance Analysis
        report.push_str("\n## Performance Analysis\n\n");
        if summary.passes_performance_gates {
            report.push_str("🎉 **All performance gates passed!** The system meets TODO.md requirements.\n\n");
        } else {
            report.push_str("⚠️ **Performance gates failed.** Review the following issues:\n\n");
            for gate in &summary.gate_analysis {
                if !gate.passed {
                    report.push_str(&format!("- {}: {} (deficit: {:.1})\n", gate.gate_name, gate.description, -gate.margin));
                }
            }
        }

        // Save report to file
        fs::write(&report_path, &report).await?;
        info!("Benchmark report generated: {}", report_path);
        
        Ok(report_path)
    }

    /// Save benchmark artifacts (metrics, errors, config)
    async fn save_benchmark_artifacts(
        &self,
        results: &[BenchmarkResult],
        summary: &BenchmarkSummary,
        dataset_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        
        // Ensure output directory exists
        fs::create_dir_all(&self.config.output_path).await?;

        // Save detailed results as JSON
        let results_path = format!("{}/benchmark_results_{}_{}.json", self.config.output_path, dataset_name, timestamp);
        let results_json = serde_json::to_string_pretty(results)?;
        fs::write(&results_path, results_json).await?;

        // Save summary as JSON
        let summary_path = format!("{}/benchmark_summary_{}_{}.json", self.config.output_path, dataset_name, timestamp);
        let summary_json = serde_json::to_string_pretty(summary)?;
        fs::write(&summary_path, summary_json).await?;

        // Save configuration fingerprint
        let config_path = format!("{}/benchmark_config_{}_{}.json", self.config.output_path, dataset_name, timestamp);
        let config_json = serde_json::to_string_pretty(&self.config)?;
        fs::write(&config_path, config_json).await?;

        info!("Benchmark artifacts saved to {}", self.config.output_path);
        Ok(())
    }

    /// Generate a fingerprint of the current configuration for reproducibility
    fn generate_config_fingerprint(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let config_str = serde_json::to_string(&self.config).unwrap_or_default();
        let mut hasher = DefaultHasher::new();
        config_str.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}