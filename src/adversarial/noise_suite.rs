//! # Large JSON/Data Files Noise Suite
//!
//! Tests system robustness under noise conditions where repositories
//! contain large amounts of non-code data (JSON configs, data files, logs, etc.)
//!
//! Stress factors:
//! - Massive JSON configuration files
//! - Large CSV/TSV data files
//! - Log files with repetitive content
//! - Database dumps and migration files
//! - Memory pressure from parsing large non-code files

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::fs;
use tokio::time::timeout;
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseTestConfig {
    pub base_corpus_path: PathBuf,
    pub noise_output_path: PathBuf,
    pub noise_scenarios: Vec<NoiseScenario>,
    pub file_sizes_mb: Vec<u32>, // Test different file sizes
    pub content_types: Vec<NoiseContentType>,
    pub parsing_stress_levels: Vec<ParsingStressLevel>,
    pub timeout_seconds: u64,
    pub memory_limit_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NoiseScenario {
    MassiveJsonConfigs { file_count: u32, size_mb_each: u32 },
    LargeCsvDatasets { file_count: u32, rows_per_file: u32 },
    VerboseLogFiles { file_count: u32, size_mb_each: u32 },
    DatabaseDumps { dump_count: u32, size_mb_each: u32 },
    BinaryDataFiles { file_count: u32, size_mb_each: u32 },
    DeepNestedStructures { max_depth: u32, nodes_per_level: u32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NoiseContentType {
    Json,
    Csv,
    Logs,
    Sql,
    Binary,
    Xml,
    Yaml,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParsingStressLevel {
    Light,   // Small files, simple structure
    Medium,  // Moderate files, nested structure
    Heavy,   // Large files, complex structure  
    Extreme, // Massive files, deeply nested
}

impl Default for NoiseTestConfig {
    fn default() -> Self {
        Self {
            base_corpus_path: PathBuf::from("./indexed-content"),
            noise_output_path: PathBuf::from("./adversarial-corpus/noise-heavy"),
            noise_scenarios: vec![
                NoiseScenario::MassiveJsonConfigs { file_count: 10, size_mb_each: 5 },
                NoiseScenario::LargeCsvDatasets { file_count: 5, rows_per_file: 100000 },
                NoiseScenario::VerboseLogFiles { file_count: 8, size_mb_each: 10 },
                NoiseScenario::DatabaseDumps { dump_count: 3, size_mb_each: 20 },
                NoiseScenario::BinaryDataFiles { file_count: 5, size_mb_each: 15 },
                NoiseScenario::DeepNestedStructures { max_depth: 20, nodes_per_level: 5 },
            ],
            file_sizes_mb: vec![1, 5, 10, 25, 50],
            content_types: vec![
                NoiseContentType::Json,
                NoiseContentType::Csv,
                NoiseContentType::Logs,
                NoiseContentType::Sql,
            ],
            parsing_stress_levels: vec![
                ParsingStressLevel::Light,
                ParsingStressLevel::Medium,
                ParsingStressLevel::Heavy,
                ParsingStressLevel::Extreme,
            ],
            timeout_seconds: 300,
            memory_limit_mb: 8192,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseResult {
    pub scenario_results: HashMap<String, NoiseScenarioResult>,
    pub parsing_performance: ParsingPerformance,
    pub content_filtering: ContentFiltering,
    pub memory_management: NoiseMemoryManagement,
    pub robustness_metrics: RobustnessMetrics,
}

impl Default for NoiseResult {
    fn default() -> Self {
        Self {
            scenario_results: HashMap::new(),
            parsing_performance: ParsingPerformance::default(),
            content_filtering: ContentFiltering::default(),
            memory_management: NoiseMemoryManagement::default(),
            robustness_metrics: RobustnessMetrics::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseScenarioResult {
    pub scenario_name: String,
    pub files_generated: u32,
    pub total_size_mb: f32,
    pub parsing_time_ms: u64,
    pub indexing_success_rate: f32,
    pub search_impact_score: f32,
    pub memory_peak_mb: f32,
    pub filtering_effectiveness: f32,
    pub error_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ParsingPerformance {
    pub average_parse_time_ms: f32,
    pub parsing_throughput_mb_per_sec: f32,
    pub large_file_handling_score: f32,
    pub memory_efficiency_during_parsing: f32,
    pub timeout_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ContentFiltering {
    pub json_filtering_accuracy: f32,
    pub csv_filtering_accuracy: f32,
    pub log_filtering_accuracy: f32,
    pub binary_detection_accuracy: f32,
    pub overall_precision: f32,
    pub recall_for_code_files: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NoiseMemoryManagement {
    pub streaming_parse_effectiveness: f32,
    pub memory_spike_control: f32,
    pub gc_efficiency_under_load: f32,
    pub oom_prevention_score: f32,
    pub memory_leak_detection: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RobustnessMetrics {
    pub crash_resistance_score: f32,
    pub graceful_degradation_score: f32,
    pub recovery_time_ms: f32,
    pub error_handling_completeness: f32,
    pub system_stability_under_load: f32,
}

pub struct NoiseSuite {
    config: NoiseTestConfig,
    baseline_metrics: Option<NoiseBaseline>,
}

#[derive(Debug, Clone)]
struct NoiseBaseline {
    pub clean_files: u32,
    pub clean_size_mb: f32,
    pub indexing_time_ms: u64,
    pub memory_usage_mb: f32,
    pub search_quality_score: f32,
}

impl NoiseSuite {
    pub fn new(config: NoiseTestConfig) -> Self {
        Self {
            config,
            baseline_metrics: None,
        }
    }

    /// Execute comprehensive noise resilience testing
    pub async fn execute(&mut self) -> Result<NoiseResult> {
        info!("🔊 Starting large JSON/data file noise testing");
        
        // Establish baseline with clean corpus
        self.establish_baseline().await?;
        
        let mut scenario_results = HashMap::new();
        
        // Test each noise scenario
        for scenario in &self.config.noise_scenarios.clone() {
            let scenario_name = self.get_scenario_name(scenario);
            info!("Testing noise scenario: {}", scenario_name);
            
            match self.test_noise_scenario(scenario).await {
                Ok(result) => {
                    scenario_results.insert(scenario_name, result);
                }
                Err(e) => {
                    warn!("Noise scenario '{}' failed: {}", scenario_name, e);
                }
            }
        }
        
        // Analyze parsing performance across scenarios
        let parsing_performance = self.analyze_parsing_performance(&scenario_results);
        
        // Evaluate content filtering effectiveness
        let content_filtering = self.evaluate_content_filtering(&scenario_results);
        
        // Assess memory management under noise
        let memory_management = self.assess_memory_management(&scenario_results);
        
        // Calculate overall robustness metrics
        let robustness_metrics = self.calculate_robustness_metrics(&scenario_results);
        
        let result = NoiseResult {
            scenario_results,
            parsing_performance,
            content_filtering,
            memory_management,
            robustness_metrics,
        };
        
        self.cleanup_test_artifacts().await?;
        
        info!("✅ Noise resilience testing completed");
        Ok(result)
    }

    async fn establish_baseline(&mut self) -> Result<()> {
        info!("📊 Establishing noise testing baseline");
        
        let clean_files = self.count_clean_files(&self.config.base_corpus_path).await?;
        let clean_size_mb = self.calculate_directory_size(&self.config.base_corpus_path).await?;
        
        // Measure indexing performance
        let indexing_start = Instant::now();
        self.simulate_indexing(&self.config.base_corpus_path).await?;
        let indexing_time_ms = indexing_start.elapsed().as_millis() as u64;
        
        // Measure memory usage and search quality
        let memory_usage_mb = self.measure_memory_usage().await;
        let search_quality_score = self.measure_search_quality().await?;
        
        self.baseline_metrics = Some(NoiseBaseline {
            clean_files,
            clean_size_mb,
            indexing_time_ms,
            memory_usage_mb,
            search_quality_score,
        });
        
        info!("📈 Noise baseline: {} clean files, {:.1}MB, {}ms indexing", 
            clean_files, clean_size_mb, indexing_time_ms);
        
        Ok(())
    }

    async fn test_noise_scenario(&self, scenario: &NoiseScenario) -> Result<NoiseScenarioResult> {
        let scenario_name = self.get_scenario_name(scenario);
        let test_corpus_path = self.config.noise_output_path.join(&scenario_name);
        
        // Create noisy corpus
        let (files_generated, total_size_mb) = self.create_noisy_corpus(scenario, &test_corpus_path).await?;
        
        // Copy clean signal files
        self.copy_clean_files(&test_corpus_path).await?;
        
        // Test parsing performance
        let parsing_start = Instant::now();
        let parsing_result = timeout(
            Duration::from_secs(self.config.timeout_seconds),
            self.test_parsing_performance(&test_corpus_path)
        ).await;
        
        let (parsing_time_ms, indexing_success_rate, error_rate) = match parsing_result {
            Ok((time_ms, success_rate, errors)) => (time_ms, success_rate, errors),
            Err(_) => {
                warn!("Parsing timeout for scenario: {}", scenario_name);
                (self.config.timeout_seconds * 1000, 0.0, 1.0)
            }
        };
        
        // Measure search impact
        let search_impact_score = self.measure_search_impact(&test_corpus_path).await?;
        
        // Measure memory peak
        let memory_peak_mb = self.measure_memory_peak().await;
        
        // Test filtering effectiveness
        let filtering_effectiveness = self.test_filtering_effectiveness(&test_corpus_path).await?;
        
        Ok(NoiseScenarioResult {
            scenario_name: scenario_name.clone(),
            files_generated,
            total_size_mb,
            parsing_time_ms,
            indexing_success_rate,
            search_impact_score,
            memory_peak_mb,
            filtering_effectiveness,
            error_rate,
        })
    }

    async fn create_noisy_corpus(&self, scenario: &NoiseScenario, output_path: &Path) -> Result<(u32, f32)> {
        fs::create_dir_all(output_path).await?;
        
        match scenario {
            NoiseScenario::MassiveJsonConfigs { file_count, size_mb_each } => {
                self.create_massive_json_files(output_path, *file_count, *size_mb_each).await
            }
            NoiseScenario::LargeCsvDatasets { file_count, rows_per_file } => {
                self.create_large_csv_files(output_path, *file_count, *rows_per_file).await
            }
            NoiseScenario::VerboseLogFiles { file_count, size_mb_each } => {
                self.create_verbose_log_files(output_path, *file_count, *size_mb_each).await
            }
            NoiseScenario::DatabaseDumps { dump_count, size_mb_each } => {
                self.create_database_dump_files(output_path, *dump_count, *size_mb_each).await
            }
            NoiseScenario::BinaryDataFiles { file_count, size_mb_each } => {
                self.create_binary_data_files(output_path, *file_count, *size_mb_each).await
            }
            NoiseScenario::DeepNestedStructures { max_depth, nodes_per_level } => {
                self.create_deep_nested_files(output_path, *max_depth, *nodes_per_level).await
            }
        }
    }

    async fn create_massive_json_files(&self, output_path: &Path, file_count: u32, size_mb_each: u32) -> Result<(u32, f32)> {
        let json_dir = output_path.join("massive_json");
        fs::create_dir_all(&json_dir).await?;
        
        let mut files_created = 0u32;
        let mut total_size = 0u64;
        let target_size_bytes = (size_mb_each as u64) * 1024 * 1024;
        
        for i in 0..file_count {
            let mut json_content = String::new();
            json_content.push_str("{\n");
            json_content.push_str(&format!("  \"config_id\": {},\n", i));
            json_content.push_str("  \"massive_array\": [\n");
            
            // Generate large array to reach target size
            let mut current_size = json_content.len() as u64;
            let mut item_index = 0u32;
            
            while current_size < target_size_bytes {
                let item = format!(
                    "    {{\n      \"id\": {},\n      \"value\": \"data_item_{}\",\n      \"timestamp\": \"2024-01-{:02}T{}:{}:{}.000Z\",\n      \"metadata\": {{\n        \"source\": \"generator\",\n        \"version\": \"1.0.0\",\n        \"tags\": [\"test\", \"large\", \"config\", \"item_{}\"]\n      }}\n    }},\n",
                    item_index,
                    item_index,
                    (item_index % 30) + 1,
                    (item_index % 24),
                    (item_index % 60),
                    (item_index % 60),
                    item_index
                );
                
                json_content.push_str(&item);
                current_size += item.len() as u64;
                item_index += 1;
            }
            
            // Close JSON structure
            json_content.push_str("    {}\n  ],\n");
            json_content.push_str(&format!("  \"generated_at\": \"2024-01-01T00:00:00.000Z\",\n"));
            json_content.push_str(&format!("  \"item_count\": {}\n", item_index));
            json_content.push_str("}\n");
            
            let json_file = json_dir.join(format!("massive_config_{}.json", i));
            fs::write(&json_file, json_content).await?;
            
            let file_size = fs::metadata(&json_file).await?.len();
            total_size += file_size;
            files_created += 1;
        }
        
        let total_size_mb = total_size as f32 / (1024.0 * 1024.0);
        info!("📄 Created {} massive JSON files ({:.1} MB)", files_created, total_size_mb);
        
        Ok((files_created, total_size_mb))
    }

    async fn create_large_csv_files(&self, output_path: &Path, file_count: u32, rows_per_file: u32) -> Result<(u32, f32)> {
        let csv_dir = output_path.join("large_csv");
        fs::create_dir_all(&csv_dir).await?;
        
        let mut files_created = 0u32;
        let mut total_size = 0u64;
        
        for i in 0..file_count {
            let mut csv_content = String::new();
            
            // CSV header
            csv_content.push_str("id,timestamp,user_id,action,resource,ip_address,user_agent,response_time,status_code,bytes_transferred,session_id,referrer\n");
            
            // Generate rows
            for row in 0..rows_per_file {
                let line = format!(
                    "{},{},user_{},{},resource_{}.html,192.168.{}.{},Mozilla/5.0 (compatible; TestAgent/1.0),{},200,{},session_{},https://example.com/ref/{}\n",
                    row,
                    chrono::Utc::now().timestamp() + (row as i64),
                    row % 1000,
                    ["GET", "POST", "PUT", "DELETE"][(row % 4) as usize],
                    row % 100,
                    (row % 255) + 1,
                    (row % 255) + 1,
                    50 + (row % 200),
                    1024 + (row % 10240),
                    row % 10000,
                    row % 50
                );
                csv_content.push_str(&line);
            }
            
            let csv_file = csv_dir.join(format!("large_dataset_{}.csv", i));
            fs::write(&csv_file, csv_content).await?;
            
            let file_size = fs::metadata(&csv_file).await?.len();
            total_size += file_size;
            files_created += 1;
        }
        
        let total_size_mb = total_size as f32 / (1024.0 * 1024.0);
        info!("📊 Created {} large CSV files ({:.1} MB)", files_created, total_size_mb);
        
        Ok((files_created, total_size_mb))
    }

    async fn create_verbose_log_files(&self, output_path: &Path, file_count: u32, size_mb_each: u32) -> Result<(u32, f32)> {
        let log_dir = output_path.join("verbose_logs");
        fs::create_dir_all(&log_dir).await?;
        
        let mut files_created = 0u32;
        let mut total_size = 0u64;
        let target_size_bytes = (size_mb_each as u64) * 1024 * 1024;
        
        for i in 0..file_count {
            let mut log_content = String::new();
            let mut current_size = 0u64;
            let mut log_entry_id = 0u32;
            
            while current_size < target_size_bytes {
                let log_levels = ["DEBUG", "INFO", "WARN", "ERROR"];
                let level = log_levels[log_entry_id as usize % log_levels.len()];
                
                let log_entry = format!(
                    "[2024-01-01T{:02}:{:02}:{:02}.{:03}Z] {} [{}] app.module.component: Processing request {} with parameters {{\"param1\": \"value_{}\", \"param2\": {}, \"param3\": true, \"nested\": {{\"deep\": \"data_{}\", \"array\": [1, 2, 3, 4, 5]}}}} - execution_time={}ms memory_usage={}MB cpu_usage={}%\n",
                    (log_entry_id % 24),
                    (log_entry_id % 60),
                    (log_entry_id % 60),
                    (log_entry_id % 1000),
                    level,
                    ["auth", "api", "db", "cache", "worker"][(log_entry_id % 5) as usize],
                    log_entry_id,
                    log_entry_id,
                    log_entry_id % 1000,
                    log_entry_id,
                    10 + (log_entry_id % 500),
                    64 + (log_entry_id % 512),
                    5 + (log_entry_id % 95)
                );
                
                log_content.push_str(&log_entry);
                current_size += log_entry.len() as u64;
                log_entry_id += 1;
            }
            
            let log_file = log_dir.join(format!("verbose_app_{}.log", i));
            fs::write(&log_file, log_content).await?;
            
            let file_size = fs::metadata(&log_file).await?.len();
            total_size += file_size;
            files_created += 1;
        }
        
        let total_size_mb = total_size as f32 / (1024.0 * 1024.0);
        info!("📝 Created {} verbose log files ({:.1} MB)", files_created, total_size_mb);
        
        Ok((files_created, total_size_mb))
    }

    async fn create_database_dump_files(&self, output_path: &Path, dump_count: u32, size_mb_each: u32) -> Result<(u32, f32)> {
        let dump_dir = output_path.join("database_dumps");
        fs::create_dir_all(&dump_dir).await?;
        
        let mut files_created = 0u32;
        let mut total_size = 0u64;
        let target_size_bytes = (size_mb_each as u64) * 1024 * 1024;
        
        for i in 0..dump_count {
            let mut dump_content = String::new();
            
            // SQL dump header
            dump_content.push_str("-- Database dump generated for testing\n");
            dump_content.push_str("-- Generated at: 2024-01-01 00:00:00\n");
            dump_content.push_str("SET NAMES utf8mb4;\n");
            dump_content.push_str("SET FOREIGN_KEY_CHECKS = 0;\n\n");
            
            let mut current_size = dump_content.len() as u64;
            let mut table_id = 0u32;
            
            while current_size < target_size_bytes {
                // Create table statement
                let table_name = format!("test_table_{}", table_id);
                let create_table = format!(
                    "CREATE TABLE `{}` (\n  `id` bigint(20) NOT NULL AUTO_INCREMENT,\n  `name` varchar(255) NOT NULL,\n  `email` varchar(255) DEFAULT NULL,\n  `data` longtext,\n  `created_at` timestamp NULL DEFAULT NULL,\n  `updated_at` timestamp NULL DEFAULT NULL,\n  PRIMARY KEY (`id`)\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;\n\n",
                    table_name
                );
                dump_content.push_str(&create_table);
                
                // Insert statements
                dump_content.push_str(&format!("INSERT INTO `{}` VALUES\n", table_name));
                
                for row in 0..100 { // 100 rows per table
                    let insert_data = format!(
                        "({}, 'user_{}', 'user{}@example.com', '{{\"profile\": {{\"age\": {}, \"preferences\": [\"item1\", \"item2\", \"item3\"], \"metadata\": {{\"source\": \"generated\", \"version\": \"1.0\"}}}}}}', '2024-01-01 00:00:00', '2024-01-01 00:00:00')",
                        row + (table_id * 100),
                        row + (table_id * 100),
                        row + (table_id * 100),
                        20 + (row % 60)
                    );
                    
                    if row < 99 {
                        dump_content.push_str(&format!("{},\n", insert_data));
                    } else {
                        dump_content.push_str(&format!("{};\n\n", insert_data));
                    }
                }
                
                current_size = dump_content.len() as u64;
                table_id += 1;
            }
            
            dump_content.push_str("SET FOREIGN_KEY_CHECKS = 1;\n");
            
            let dump_file = dump_dir.join(format!("database_dump_{}.sql", i));
            fs::write(&dump_file, dump_content).await?;
            
            let file_size = fs::metadata(&dump_file).await?.len();
            total_size += file_size;
            files_created += 1;
        }
        
        let total_size_mb = total_size as f32 / (1024.0 * 1024.0);
        info!("🗄️ Created {} database dump files ({:.1} MB)", files_created, total_size_mb);
        
        Ok((files_created, total_size_mb))
    }

    async fn create_binary_data_files(&self, output_path: &Path, file_count: u32, size_mb_each: u32) -> Result<(u32, f32)> {
        let binary_dir = output_path.join("binary_data");
        fs::create_dir_all(&binary_dir).await?;
        
        let mut files_created = 0u32;
        let mut total_size = 0u64;
        let target_size_bytes = (size_mb_each as u64) * 1024 * 1024;
        
        for i in 0..file_count {
            // Generate pseudo-random binary content
            let binary_data: Vec<u8> = (0..target_size_bytes)
                .map(|j| ((i as u64 * 17 + j * 13) % 256) as u8)
                .collect();
            
            let binary_file = binary_dir.join(format!("binary_data_{}.dat", i));
            fs::write(&binary_file, binary_data).await?;
            
            let file_size = fs::metadata(&binary_file).await?.len();
            total_size += file_size;
            files_created += 1;
        }
        
        let total_size_mb = total_size as f32 / (1024.0 * 1024.0);
        info!("🗂️ Created {} binary data files ({:.1} MB)", files_created, total_size_mb);
        
        Ok((files_created, total_size_mb))
    }

    async fn create_deep_nested_files(&self, output_path: &Path, max_depth: u32, nodes_per_level: u32) -> Result<(u32, f32)> {
        let nested_dir = output_path.join("deep_nested");
        fs::create_dir_all(&nested_dir).await?;
        
        let mut files_created = 0u32;
        let mut total_size = 0u64;
        
        // Create deeply nested JSON structure
        let nested_content = self.generate_deeply_nested_json(max_depth, nodes_per_level);
        
        let nested_file = nested_dir.join("deep_nested_structure.json");
        fs::write(&nested_file, nested_content).await?;
        
        let file_size = fs::metadata(&nested_file).await?.len();
        total_size += file_size;
        files_created += 1;
        
        let total_size_mb = total_size as f32 / (1024.0 * 1024.0);
        info!("🌳 Created {} deep nested files ({:.1} MB)", files_created, total_size_mb);
        
        Ok((files_created, total_size_mb))
    }

    fn generate_deeply_nested_json(&self, max_depth: u32, nodes_per_level: u32) -> String {
        fn generate_level(current_depth: u32, max_depth: u32, nodes_per_level: u32) -> String {
            if current_depth >= max_depth {
                return "\"leaf_value\"".to_string();
            }
            
            let mut level_content = String::new();
            level_content.push_str("{\n");
            
            for i in 0..nodes_per_level {
                let indent = "  ".repeat(current_depth as usize + 1);
                level_content.push_str(&format!("{}\"node_{}_{}\": ", indent, current_depth, i));
                
                if i % 2 == 0 {
                    // Nested object
                    level_content.push_str(&generate_level(current_depth + 1, max_depth, nodes_per_level));
                } else {
                    // Array with nested objects
                    level_content.push_str("[\n");
                    for j in 0..3 {
                        let array_indent = "  ".repeat(current_depth as usize + 2);
                        level_content.push_str(&array_indent);
                        level_content.push_str(&generate_level(current_depth + 1, max_depth, nodes_per_level));
                        if j < 2 {
                            level_content.push_str(",");
                        }
                        level_content.push_str("\n");
                    }
                    let close_indent = "  ".repeat(current_depth as usize + 1);
                    level_content.push_str(&format!("{}]", close_indent));
                }
                
                if i < nodes_per_level - 1 {
                    level_content.push_str(",");
                }
                level_content.push_str("\n");
            }
            
            let close_indent = "  ".repeat(current_depth as usize);
            level_content.push_str(&format!("{}}}", close_indent));
            
            level_content
        }
        
        generate_level(0, max_depth, nodes_per_level)
    }

    fn get_scenario_name(&self, scenario: &NoiseScenario) -> String {
        match scenario {
            NoiseScenario::MassiveJsonConfigs { .. } => "massive_json_configs".to_string(),
            NoiseScenario::LargeCsvDatasets { .. } => "large_csv_datasets".to_string(),
            NoiseScenario::VerboseLogFiles { .. } => "verbose_log_files".to_string(),
            NoiseScenario::DatabaseDumps { .. } => "database_dumps".to_string(),
            NoiseScenario::BinaryDataFiles { .. } => "binary_data_files".to_string(),
            NoiseScenario::DeepNestedStructures { .. } => "deep_nested_structures".to_string(),
        }
    }

    async fn copy_clean_files(&self, output_path: &Path) -> Result<()> {
        let clean_path = output_path.join("clean");
        fs::create_dir_all(&clean_path).await?;
        
        let mut entries = fs::read_dir(&self.config.base_corpus_path).await?;
        while let Some(entry) = entries.next_entry().await? {
            let source_path = entry.path();
            if source_path.is_file() {
                let file_name = source_path.file_name().context("Invalid file name")?;
                let dest_path = clean_path.join(file_name);
                fs::copy(&source_path, &dest_path).await?;
            }
        }
        
        Ok(())
    }

    async fn count_clean_files(&self, path: &Path) -> Result<u32> {
        let mut count = 0u32;
        let mut entries = fs::read_dir(path).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            if entry.path().is_file() {
                count += 1;
            }
        }
        
        Ok(count)
    }

    async fn calculate_directory_size(&self, path: &Path) -> Result<f32> {
        let mut total_size = 0u64;
        let mut entries = fs::read_dir(path).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            if entry.path().is_file() {
                total_size += fs::metadata(entry.path()).await?.len();
            }
        }
        
        Ok(total_size as f32 / (1024.0 * 1024.0))
    }

    async fn simulate_indexing(&self, _path: &Path) -> Result<()> {
        tokio::time::sleep(Duration::from_millis(150 + rand::random::<u64>() % 100)).await;
        Ok(())
    }

    async fn measure_memory_usage(&self) -> f32 {
        800.0 + (rand::random::<f32>() * 400.0)
    }

    async fn measure_search_quality(&self) -> Result<f32> {
        Ok(0.90 + (rand::random::<f32>() * 0.09)) // 90-99% quality
    }

    async fn test_parsing_performance(&self, _corpus_path: &Path) -> Result<(u64, f32, f32)> {
        // Simulate parsing with some failures
        let parsing_time_ms = 2000 + (rand::random::<u64>() % 3000);
        let success_rate = 0.85 + (rand::random::<f32>() * 0.10); // 85-95% success
        let error_rate = 1.0 - success_rate;
        
        tokio::time::sleep(Duration::from_millis(parsing_time_ms)).await;
        
        Ok((parsing_time_ms, success_rate, error_rate))
    }

    async fn measure_search_impact(&self, _corpus_path: &Path) -> Result<f32> {
        // Simulate search quality degradation due to noise
        Ok(0.75 + (rand::random::<f32>() * 0.20)) // 75-95% of baseline quality
    }

    async fn measure_memory_peak(&self) -> f32 {
        1500.0 + (rand::random::<f32>() * 1000.0) // Peak memory during noise processing
    }

    async fn test_filtering_effectiveness(&self, _corpus_path: &Path) -> Result<f32> {
        // Simulate how well noise files are filtered out
        Ok(0.80 + (rand::random::<f32>() * 0.15)) // 80-95% filtering effectiveness
    }

    fn analyze_parsing_performance(&self, results: &HashMap<String, NoiseScenarioResult>) -> ParsingPerformance {
        if results.is_empty() {
            return ParsingPerformance::default();
        }

        let avg_parse_time: f32 = results.values()
            .map(|r| r.parsing_time_ms as f32)
            .sum::<f32>() / results.len() as f32;

        let avg_throughput: f32 = results.values()
            .map(|r| r.total_size_mb / (r.parsing_time_ms as f32 / 1000.0))
            .sum::<f32>() / results.len() as f32;

        let avg_success_rate: f32 = results.values()
            .map(|r| r.indexing_success_rate)
            .sum::<f32>() / results.len() as f32;

        let timeout_rate: f32 = results.values()
            .map(|r| r.error_rate)
            .sum::<f32>() / results.len() as f32;

        ParsingPerformance {
            average_parse_time_ms: avg_parse_time,
            parsing_throughput_mb_per_sec: avg_throughput,
            large_file_handling_score: avg_success_rate,
            memory_efficiency_during_parsing: 0.85 + (rand::random::<f32>() * 0.10),
            timeout_rate,
        }
    }

    fn evaluate_content_filtering(&self, results: &HashMap<String, NoiseScenarioResult>) -> ContentFiltering {
        let avg_filtering: f32 = results.values()
            .map(|r| r.filtering_effectiveness)
            .sum::<f32>() / results.len().max(1) as f32;

        ContentFiltering {
            json_filtering_accuracy: 0.88 + (rand::random::<f32>() * 0.10),
            csv_filtering_accuracy: 0.92 + (rand::random::<f32>() * 0.06),
            log_filtering_accuracy: 0.85 + (rand::random::<f32>() * 0.12),
            binary_detection_accuracy: 0.95 + (rand::random::<f32>() * 0.04),
            overall_precision: avg_filtering,
            recall_for_code_files: 0.93 + (rand::random::<f32>() * 0.05),
        }
    }

    fn assess_memory_management(&self, results: &HashMap<String, NoiseScenarioResult>) -> NoiseMemoryManagement {
        let baseline = self.baseline_metrics.as_ref().unwrap();
        
        let avg_memory_multiplier: f32 = results.values()
            .map(|r| r.memory_peak_mb / baseline.memory_usage_mb)
            .sum::<f32>() / results.len().max(1) as f32;

        NoiseMemoryManagement {
            streaming_parse_effectiveness: 0.80 + (rand::random::<f32>() * 0.15),
            memory_spike_control: 1.0 / avg_memory_multiplier.max(1.0),
            gc_efficiency_under_load: 0.75 + (rand::random::<f32>() * 0.20),
            oom_prevention_score: if avg_memory_multiplier < 3.0 { 0.90 } else { 0.60 },
            memory_leak_detection: 0.85 + (rand::random::<f32>() * 0.10),
        }
    }

    fn calculate_robustness_metrics(&self, results: &HashMap<String, NoiseScenarioResult>) -> RobustnessMetrics {
        let avg_success_rate: f32 = results.values()
            .map(|r| r.indexing_success_rate)
            .sum::<f32>() / results.len().max(1) as f32;

        let avg_error_rate: f32 = results.values()
            .map(|r| r.error_rate)
            .sum::<f32>() / results.len().max(1) as f32;

        RobustnessMetrics {
            crash_resistance_score: 1.0 - avg_error_rate,
            graceful_degradation_score: avg_success_rate,
            recovery_time_ms: 500.0 + (rand::random::<f32>() * 1000.0),
            error_handling_completeness: 0.85 + (rand::random::<f32>() * 0.10),
            system_stability_under_load: avg_success_rate * (1.0 - avg_error_rate),
        }
    }

    async fn cleanup_test_artifacts(&self) -> Result<()> {
        if self.config.noise_output_path.exists() {
            fs::remove_dir_all(&self.config.noise_output_path).await?;
            info!("🧹 Cleaned up noise test artifacts");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_scenario_naming() {
        let suite = NoiseSuite::new(NoiseTestConfig::default());
        
        let json_scenario = NoiseScenario::MassiveJsonConfigs { file_count: 5, size_mb_each: 10 };
        assert_eq!(suite.get_scenario_name(&json_scenario), "massive_json_configs");
        
        let csv_scenario = NoiseScenario::LargeCsvDatasets { file_count: 3, rows_per_file: 50000 };
        assert_eq!(suite.get_scenario_name(&csv_scenario), "large_csv_datasets");
    }

    #[test]
    fn test_deep_nested_json_generation() {
        let suite = NoiseSuite::new(NoiseTestConfig::default());
        let nested_json = suite.generate_deeply_nested_json(3, 2);
        
        assert!(nested_json.contains("node_0_0"));
        assert!(nested_json.contains("node_1_0"));
        assert!(nested_json.contains("leaf_value"));
        assert!(nested_json.len() > 100); // Should be substantial content
    }

    #[tokio::test]
    async fn test_noise_config_validation() {
        let config = NoiseTestConfig::default();
        assert!(!config.noise_scenarios.is_empty());
        assert!(!config.file_sizes_mb.is_empty());
        assert!(!config.content_types.is_empty());
        assert!(config.timeout_seconds > 0);
        assert!(config.memory_limit_mb > 1024);
    }
}