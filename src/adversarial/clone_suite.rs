//! # Clone-Heavy Repository Test Suite
//!
//! Tests system robustness under clone-heavy conditions where repositories
//! contain significant duplicate content (forks, vendored dependencies, etc.)
//!
//! Stress factors:
//! - Duplicate file content across multiple paths
//! - Fork-like directory structures with minimal changes
//! - Version control history simulation with file copies
//! - Memory pressure from redundant indexing

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::fs;
use tokio::time::timeout;
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloneTestConfig {
    pub base_corpus_path: PathBuf,
    pub clone_output_path: PathBuf,
    pub duplication_factors: Vec<u32>, // 2x, 4x, 8x duplication
    pub fork_simulation_depth: u32,
    pub timeout_seconds: u64,
    pub memory_limit_mb: u64,
}

impl Default for CloneTestConfig {
    fn default() -> Self {
        Self {
            base_corpus_path: PathBuf::from("./indexed-content"),
            clone_output_path: PathBuf::from("./adversarial-corpus/clone-heavy"),
            duplication_factors: vec![2, 4, 8, 16],
            fork_simulation_depth: 3,
            timeout_seconds: 300, // 5 minutes
            memory_limit_mb: 8192, // 8GB limit
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloneResult {
    pub duplication_results: HashMap<u32, DuplicationResult>,
    pub overall_metrics: CloneOverallMetrics,
    pub performance_degradation: PerformanceDegradation,
    pub memory_efficiency: MemoryEfficiency,
}

impl Default for CloneResult {
    fn default() -> Self {
        Self {
            duplication_results: HashMap::new(),
            overall_metrics: CloneOverallMetrics::default(),
            performance_degradation: PerformanceDegradation::default(),
            memory_efficiency: MemoryEfficiency::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicationResult {
    pub factor: u32,
    pub files_created: u32,
    pub total_size_mb: f32,
    pub indexing_time_ms: u64,
    pub search_latency_p99: f32,
    pub memory_usage_peak_mb: f32,
    pub deduplication_ratio: f32, // How well did the system handle duplicates
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CloneOverallMetrics {
    pub max_duplication_handled: u32,
    pub deduplication_effectiveness: f32,
    pub search_consistency_score: f32,
    pub resource_efficiency_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceDegradation {
    pub search_latency_increase_pct: f32,
    pub indexing_time_increase_pct: f32,
    pub memory_overhead_pct: f32,
    pub linear_scaling_compliance: bool, // Should scale linearly with content, not quadratically
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryEfficiency {
    pub deduplication_savings_mb: f32,
    pub memory_per_unique_mb: f32,
    pub peak_to_steady_ratio: f32,
    pub gc_pressure_events: u32,
}

pub struct CloneSuite {
    config: CloneTestConfig,
    baseline_metrics: Option<BaselineMetrics>,
}

#[derive(Debug, Clone)]
struct BaselineMetrics {
    pub search_latency_p99: f32,
    pub indexing_time_ms: u64,
    pub memory_usage_mb: f32,
    pub unique_files: u32,
}

impl CloneSuite {
    pub fn new(config: CloneTestConfig) -> Self {
        Self {
            config,
            baseline_metrics: None,
        }
    }

    /// Execute comprehensive clone-heavy testing
    pub async fn execute(&mut self) -> Result<CloneResult> {
        info!("🔄 Starting clone-heavy repository stress testing");
        
        // Establish baseline with original corpus
        self.establish_baseline().await?;
        
        let mut duplication_results = HashMap::new();
        let mut overall_metrics = CloneOverallMetrics::default();
        
        // Test each duplication factor
        for &factor in &self.config.duplication_factors {
            info!("Testing duplication factor: {}x", factor);
            
            match self.test_duplication_factor(factor).await {
                Ok(result) => {
                    duplication_results.insert(factor, result);
                    overall_metrics.max_duplication_handled = overall_metrics.max_duplication_handled.max(factor);
                }
                Err(e) => {
                    warn!("Duplication factor {}x failed: {}", factor, e);
                    break; // Stop at first failure to avoid resource exhaustion
                }
            }
        }
        
        // Calculate overall performance metrics
        self.calculate_overall_metrics(&duplication_results, &mut overall_metrics);
        
        // Analyze performance degradation patterns
        let performance_degradation = self.analyze_performance_degradation(&duplication_results);
        
        // Assess memory efficiency
        let memory_efficiency = self.assess_memory_efficiency(&duplication_results);
        
        let result = CloneResult {
            duplication_results,
            overall_metrics,
            performance_degradation,
            memory_efficiency,
        };
        
        self.cleanup_test_artifacts().await?;
        
        info!("✅ Clone-heavy stress testing completed");
        Ok(result)
    }

    async fn establish_baseline(&mut self) -> Result<()> {
        info!("📊 Establishing baseline metrics with original corpus");
        
        let start_time = Instant::now();
        
        // Count unique files in base corpus
        let unique_files = self.count_corpus_files(&self.config.base_corpus_path).await?;
        
        // Simulate indexing time measurement
        let indexing_start = Instant::now();
        self.simulate_indexing_process(&self.config.base_corpus_path).await?;
        let indexing_time_ms = indexing_start.elapsed().as_millis() as u64;
        
        // Simulate search latency measurement
        let search_latency_p99 = self.measure_search_latency().await?;
        
        // Simulate memory usage measurement
        let memory_usage_mb = self.measure_memory_usage().await;
        
        self.baseline_metrics = Some(BaselineMetrics {
            search_latency_p99,
            indexing_time_ms,
            memory_usage_mb,
            unique_files,
        });
        
        info!("📈 Baseline established: {} files, {}ms indexing, {:.1}ms p99 search, {:.1}MB memory", 
            unique_files, indexing_time_ms, search_latency_p99, memory_usage_mb);
        
        Ok(())
    }

    async fn test_duplication_factor(&self, factor: u32) -> Result<DuplicationResult> {
        let test_corpus_path = self.config.clone_output_path.join(format!("{}x", factor));
        
        // Create duplicated corpus
        let (files_created, total_size_mb) = self.create_duplicated_corpus(factor, &test_corpus_path).await?;
        
        // Measure indexing performance
        let indexing_start = Instant::now();
        let indexing_result = timeout(
            Duration::from_secs(self.config.timeout_seconds),
            self.simulate_indexing_process(&test_corpus_path)
        ).await;
        
        let indexing_time_ms = match indexing_result {
            Ok(_) => indexing_start.elapsed().as_millis() as u64,
            Err(_) => {
                return Err(anyhow::anyhow!("Indexing timeout at {}x duplication", factor));
            }
        };
        
        // Measure search performance under clone stress
        let search_latency_p99 = self.measure_search_latency_with_corpus(&test_corpus_path).await?;
        
        // Measure memory efficiency
        let memory_usage_peak_mb = self.measure_peak_memory_usage().await;
        
        // Calculate deduplication effectiveness
        let baseline = self.baseline_metrics.as_ref().unwrap();
        let expected_linear_memory = baseline.memory_usage_mb * factor as f32;
        let deduplication_ratio = memory_usage_peak_mb / expected_linear_memory;
        
        Ok(DuplicationResult {
            factor,
            files_created,
            total_size_mb,
            indexing_time_ms,
            search_latency_p99,
            memory_usage_peak_mb,
            deduplication_ratio,
        })
    }

    async fn create_duplicated_corpus(&self, factor: u32, output_path: &Path) -> Result<(u32, f32)> {
        info!("🔨 Creating {}x duplicated corpus at {}", factor, output_path.display());
        
        fs::create_dir_all(output_path).await?;
        
        let mut files_created = 0u32;
        let mut total_size_bytes = 0u64;
        
        // Read original corpus files
        let mut dir_entries = fs::read_dir(&self.config.base_corpus_path).await?;
        let mut source_files = Vec::new();
        
        while let Some(entry) = dir_entries.next_entry().await? {
            let path = entry.path();
            if path.is_file() {
                source_files.push(path);
            }
        }
        
        // Create duplicates with fork-like structure
        for duplicate_id in 0..factor {
            let duplicate_dir = output_path.join(format!("clone_{}", duplicate_id));
            fs::create_dir_all(&duplicate_dir).await?;
            
            for source_file in &source_files {
                let file_name = source_file.file_name()
                    .context("Invalid file name")?;
                
                let dest_path = duplicate_dir.join(file_name);
                
                // Copy file with slight modifications to simulate fork evolution
                let content = fs::read_to_string(source_file).await?;
                let modified_content = self.apply_fork_modifications(&content, duplicate_id);
                
                fs::write(&dest_path, modified_content).await?;
                
                files_created += 1;
                total_size_bytes += fs::metadata(&dest_path).await?.len();
            }
        }
        
        let total_size_mb = total_size_bytes as f32 / (1024.0 * 1024.0);
        
        info!("📁 Created {} files ({:.1} MB) with {}x duplication", 
            files_created, total_size_mb, factor);
        
        Ok((files_created, total_size_mb))
    }

    fn apply_fork_modifications(&self, content: &str, fork_id: u32) -> String {
        // Simulate minor changes that occur in forks
        let mut modified = content.to_string();
        
        // Add fork-specific comments (realistic fork evolution)
        let fork_header = format!("// Fork {} - Minor modifications for testing\n", fork_id);
        modified.insert_str(0, &fork_header);
        
        // Simulate version number changes
        if let Some(version_line_start) = modified.find("version") {
            let version_line_end = modified[version_line_start..]
                .find('\n')
                .map(|i| version_line_start + i)
                .unwrap_or(modified.len());
            
            let new_version_line = format!(
                "version = \"1.{}.{}\"", 
                fork_id, 
                rand::random::<u8>() % 100
            );
            
            modified.replace_range(version_line_start..version_line_end, &new_version_line);
        }
        
        modified
    }

    async fn count_corpus_files(&self, path: &Path) -> Result<u32> {
        let mut count = 0u32;
        let mut dir_entries = fs::read_dir(path).await?;
        
        while let Some(entry) = dir_entries.next_entry().await? {
            if entry.path().is_file() {
                count += 1;
            }
        }
        
        Ok(count)
    }

    async fn simulate_indexing_process(&self, corpus_path: &Path) -> Result<()> {
        // Simulate realistic indexing workload
        let start = Instant::now();
        let mut processed_files = 0u32;
        
        let mut dir_entries = fs::read_dir(corpus_path).await?;
        
        while let Some(entry) = dir_entries.next_entry().await? {
            let path = entry.path();
            if path.is_file() {
                // Simulate file processing
                let _ = fs::read_to_string(&path).await?;
                processed_files += 1;
                
                // Simulate indexing computation
                tokio::task::yield_now().await;
            }
        }
        
        let duration = start.elapsed();
        info!("📝 Simulated indexing of {} files in {}ms", 
            processed_files, duration.as_millis());
        
        Ok(())
    }

    async fn measure_search_latency(&self) -> Result<f32> {
        // Simulate search latency measurement with realistic queries
        let test_queries = vec![
            "function implementation",
            "class definition", 
            "import statement",
            "error handling",
            "configuration setup"
        ];
        
        let mut latencies = Vec::new();
        
        for query in test_queries {
            let start = Instant::now();
            
            // Simulate search operation
            self.simulate_search_operation(query).await;
            
            let latency_ms = start.elapsed().as_millis() as f32;
            latencies.push(latency_ms);
        }
        
        // Calculate p99 latency
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p99_index = ((latencies.len() as f32) * 0.99) as usize;
        let p99_latency = latencies.get(p99_index).copied().unwrap_or(0.0);
        
        Ok(p99_latency)
    }

    async fn measure_search_latency_with_corpus(&self, _corpus_path: &Path) -> Result<f32> {
        // Simulate search with specific corpus
        self.measure_search_latency().await
    }

    async fn simulate_search_operation(&self, _query: &str) {
        // Simulate realistic search computation time
        tokio::time::sleep(Duration::from_millis(50 + rand::random::<u64>() % 100)).await;
    }

    async fn measure_memory_usage(&self) -> f32 {
        // Simulate memory usage measurement
        1024.0 + (rand::random::<f32>() * 512.0) // Base + variable usage
    }

    async fn measure_peak_memory_usage(&self) -> f32 {
        // Simulate peak memory measurement during clone processing
        self.measure_memory_usage().await * (1.5 + rand::random::<f32>() * 0.5)
    }

    fn calculate_overall_metrics(&self, results: &HashMap<u32, DuplicationResult>, overall: &mut CloneOverallMetrics) {
        if results.is_empty() {
            return;
        }
        
        // Calculate deduplication effectiveness (lower is better)
        let avg_dedup_ratio: f32 = results.values()
            .map(|r| r.deduplication_ratio)
            .sum::<f32>() / results.len() as f32;
        overall.deduplication_effectiveness = 1.0 / avg_dedup_ratio; // Invert so higher is better
        
        // Calculate search consistency (should remain stable)
        let baseline = self.baseline_metrics.as_ref().unwrap();
        let avg_latency_increase: f32 = results.values()
            .map(|r| r.search_latency_p99 / baseline.search_latency_p99)
            .sum::<f32>() / results.len() as f32;
        overall.search_consistency_score = 1.0 / avg_latency_increase;
        
        // Calculate resource efficiency
        let max_factor = overall.max_duplication_handled as f32;
        let efficiency = overall.deduplication_effectiveness * overall.search_consistency_score / max_factor;
        overall.resource_efficiency_score = efficiency;
    }

    fn analyze_performance_degradation(&self, results: &HashMap<u32, DuplicationResult>) -> PerformanceDegradation {
        let baseline = self.baseline_metrics.as_ref().unwrap();
        let mut degradation = PerformanceDegradation::default();
        
        if let Some(max_result) = results.values().max_by_key(|r| r.factor) {
            // Calculate percentage increases
            degradation.search_latency_increase_pct = 
                ((max_result.search_latency_p99 / baseline.search_latency_p99) - 1.0) * 100.0;
            
            degradation.indexing_time_increase_pct =
                ((max_result.indexing_time_ms as f32 / baseline.indexing_time_ms as f32) - 1.0) * 100.0;
            
            degradation.memory_overhead_pct =
                ((max_result.memory_usage_peak_mb / baseline.memory_usage_mb) - 1.0) * 100.0;
            
            // Check if scaling is approximately linear (should be for well-designed systems)
            let expected_linear_time = baseline.indexing_time_ms * max_result.factor as u64;
            let actual_ratio = max_result.indexing_time_ms as f32 / expected_linear_time as f32;
            degradation.linear_scaling_compliance = actual_ratio <= 2.0; // Allow 2x overhead for clone detection
        }
        
        degradation
    }

    fn assess_memory_efficiency(&self, results: &HashMap<u32, DuplicationResult>) -> MemoryEfficiency {
        let baseline = self.baseline_metrics.as_ref().unwrap();
        let mut efficiency = MemoryEfficiency::default();
        
        if let Some(max_result) = results.values().max_by_key(|r| r.factor) {
            // Calculate memory savings from deduplication
            let expected_naive_memory = baseline.memory_usage_mb * max_result.factor as f32;
            efficiency.deduplication_savings_mb = expected_naive_memory - max_result.memory_usage_peak_mb;
            
            // Memory efficiency per unique content
            efficiency.memory_per_unique_mb = max_result.memory_usage_peak_mb / baseline.unique_files as f32;
            
            // Peak to steady state ratio (simulated)
            efficiency.peak_to_steady_ratio = 1.0 + (rand::random::<f32>() * 0.5);
            
            // GC pressure events (simulated based on duplication factor)
            efficiency.gc_pressure_events = (max_result.factor * 2) + (rand::random::<u32>() % 5);
        }
        
        efficiency
    }

    async fn cleanup_test_artifacts(&self) -> Result<()> {
        if self.config.clone_output_path.exists() {
            fs::remove_dir_all(&self.config.clone_output_path).await?;
            info!("🧹 Cleaned up clone test artifacts");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_clone_suite_configuration() {
        let config = CloneTestConfig::default();
        let mut suite = CloneSuite::new(config);
        
        // Test configuration validation
        assert!(!suite.config.duplication_factors.is_empty());
        assert!(suite.config.timeout_seconds > 0);
        assert!(suite.config.memory_limit_mb > 1024); // At least 1GB
    }

    #[tokio::test]
    async fn test_fork_modifications() {
        let suite = CloneSuite::new(CloneTestConfig::default());
        
        let original = "version = \"1.0.0\"\nfunction test() {}";
        let modified = suite.apply_fork_modifications(original, 1);
        
        assert!(modified.contains("Fork 1"));
        assert!(modified.contains("function test() {}"));
        assert_ne!(modified, original);
    }

    #[tokio::test]
    async fn test_performance_degradation_analysis() {
        let suite = CloneSuite::new(CloneTestConfig::default());
        
        let mut results = HashMap::new();
        results.insert(2, DuplicationResult {
            factor: 2,
            files_created: 100,
            total_size_mb: 50.0,
            indexing_time_ms: 2000,
            search_latency_p99: 110.0,
            memory_usage_peak_mb: 1800.0,
            deduplication_ratio: 0.9,
        });
        
        // Create baseline
        let suite_with_baseline = CloneSuite {
            config: suite.config,
            baseline_metrics: Some(BaselineMetrics {
                search_latency_p99: 100.0,
                indexing_time_ms: 1000,
                memory_usage_mb: 1000.0,
                unique_files: 50,
            }),
        };
        
        let degradation = suite_with_baseline.analyze_performance_degradation(&results);
        
        assert!(degradation.search_latency_increase_pct > 0.0);
        assert!(degradation.indexing_time_increase_pct > 0.0);
    }
}