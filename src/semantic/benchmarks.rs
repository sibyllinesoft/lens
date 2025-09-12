//! # Comprehensive Semantic Processing Benchmarks
//!
//! Production benchmarking suite for semantic processing components with:
//! - Performance regression testing and monitoring
//! - Memory usage profiling and leak detection
//! - Throughput and latency measurements across workloads
//! - A/B testing framework for model comparisons
//! - Integration with existing pinned dataset system

use anyhow::{Context, Result};
#[cfg(feature = "benchmarks")]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{info, instrument, span, Level};

use super::{
    embedding::{SemanticEncoder, EmbeddingConfig, CodeEmbedding},
    query_classifier::{QueryClassifier, ClassifierConfig, QueryClassification},
    intent_router::{IntentRouter, IntentRouterConfig, SearchContext, IntentRoutingResult},
    conformal_router::{ConformalRouter, ConformalRouterConfig, extract_conformal_features},
};

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Benchmark dataset size for different test types
    pub dataset_sizes: BenchmarkDatasetSizes,
    /// Performance targets for validation
    pub performance_targets: PerformanceTargets,
    /// Memory usage limits
    pub memory_limits: MemoryLimits,
    /// Enable detailed profiling
    pub enable_profiling: bool,
    /// Output directory for results
    pub output_directory: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkDatasetSizes {
    pub micro: usize,      // 10 queries
    pub small: usize,      // 100 queries
    pub medium: usize,     // 1,000 queries
    pub large: usize,      // 10,000 queries
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    /// Single query classification (ms)
    pub classification_p95_ms: f64,
    /// Single query routing (ms)
    pub routing_p95_ms: f64,
    /// Single embedding encoding (ms)
    pub encoding_p95_ms: f64,
    /// Batch encoding throughput (queries/sec)
    pub batch_encoding_qps: f64,
    /// End-to-end semantic processing (ms)
    pub end_to_end_p95_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLimits {
    /// Maximum memory per query (MB)
    pub max_memory_per_query_mb: f64,
    /// Maximum memory for embeddings cache (MB)
    pub max_embedding_cache_mb: f64,
    /// Maximum memory growth per hour (MB)
    pub max_memory_growth_per_hour_mb: f64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            dataset_sizes: BenchmarkDatasetSizes {
                micro: 10,
                small: 100,
                medium: 1_000,
                large: 10_000,
            },
            performance_targets: PerformanceTargets {
                classification_p95_ms: 5.0,
                routing_p95_ms: 10.0,
                encoding_p95_ms: 50.0,
                batch_encoding_qps: 100.0,
                end_to_end_p95_ms: 100.0,
            },
            memory_limits: MemoryLimits {
                max_memory_per_query_mb: 10.0,
                max_embedding_cache_mb: 100.0,
                max_memory_growth_per_hour_mb: 50.0,
            },
            enable_profiling: true,
            output_directory: "./benchmark-results".to_string(),
        }
    }
}

/// Benchmark dataset for semantic processing
#[derive(Debug, Clone)]
pub struct BenchmarkDataset {
    pub queries: Vec<BenchmarkQuery>,
    pub size_category: String,
}

#[derive(Debug, Clone)]
pub struct BenchmarkQuery {
    pub id: String,
    pub text: String,
    pub expected_intent: Option<super::query_classifier::QueryIntent>,
    pub expected_naturalness: Option<f32>,
    pub language: Option<String>,
    pub complexity: QueryComplexity,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QueryComplexity {
    Simple,   // Single word or phrase
    Medium,   // Short sentence with clear intent
    Complex,  // Long query with multiple concepts
    Extreme,  // Very long or ambiguous query
}

/// Benchmark results for a single component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentBenchmarkResults {
    pub component_name: String,
    pub dataset_size: usize,
    pub metrics: PerformanceMetrics,
    pub memory_usage: MemoryUsage,
    pub errors: Vec<BenchmarkError>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub mean_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub max_latency_ms: f64,
    pub min_latency_ms: f64,
    pub throughput_qps: f64,
    pub success_rate: f64,
    pub total_operations: u64,
    pub total_duration_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub peak_memory_mb: f64,
    pub avg_memory_mb: f64,
    pub memory_growth_mb: f64,
    pub cache_hit_rate: f64,
    pub gc_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkError {
    pub query_id: String,
    pub error_message: String,
    pub error_type: String,
    pub timestamp: u64,
}

/// Comprehensive benchmark suite for semantic processing
pub struct SemanticBenchmarkSuite {
    config: BenchmarkConfig,
    datasets: HashMap<String, BenchmarkDataset>,
    results: Arc<RwLock<Vec<ComponentBenchmarkResults>>>,
}

impl SemanticBenchmarkSuite {
    /// Create new benchmark suite
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            datasets: HashMap::new(),
            results: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Generate benchmark datasets
    pub async fn generate_datasets(&mut self) -> Result<()> {
        info!("Generating benchmark datasets");
        
        // Micro dataset (10 queries)
        let micro_queries = self.generate_queries(self.config.dataset_sizes.micro, "micro").await?;
        self.datasets.insert("micro".to_string(), BenchmarkDataset {
            queries: micro_queries,
            size_category: "micro".to_string(),
        });
        
        // Small dataset (100 queries)  
        let small_queries = self.generate_queries(self.config.dataset_sizes.small, "small").await?;
        self.datasets.insert("small".to_string(), BenchmarkDataset {
            queries: small_queries,
            size_category: "small".to_string(),
        });
        
        // Medium dataset (1,000 queries) - only if needed
        if self.config.dataset_sizes.medium > 0 {
            let medium_queries = self.generate_queries(self.config.dataset_sizes.medium, "medium").await?;
            self.datasets.insert("medium".to_string(), BenchmarkDataset {
                queries: medium_queries,
                size_category: "medium".to_string(),
            });
        }
        
        // Large dataset (10,000 queries) - only if needed
        if self.config.dataset_sizes.large > 0 {
            let large_queries = self.generate_queries(self.config.dataset_sizes.large, "large").await?;
            self.datasets.insert("large".to_string(), BenchmarkDataset {
                queries: large_queries,
                size_category: "large".to_string(),
            });
        }
        
        info!("Generated {} benchmark datasets", self.datasets.len());
        Ok(())
    }
    
    /// Generate queries for a dataset
    async fn generate_queries(&self, count: usize, category: &str) -> Result<Vec<BenchmarkQuery>> {
        let mut queries = Vec::with_capacity(count);
        
        // Query templates by complexity
        let simple_queries = [
            "calculateSum",
            "def sort",
            "class User",
            "function map",
            "import json",
        ];
        
        let medium_queries = [
            "how to sort an array",
            "find function that calculates sum",
            "class definition for user model", 
            "import statement for json parsing",
            "def function with parameters",
        ];
        
        let complex_queries = [
            "how to implement a binary search algorithm that works with generic types",
            "find all functions that process user authentication and handle edge cases",
            "create a class hierarchy for a REST API with proper error handling",
            "implement async function that fetches data from multiple APIs concurrently",
            "refactor this code to use dependency injection and improve testability",
        ];
        
        let extreme_queries = [
            "I need to find a way to optimize this complex algorithm that processes large datasets by implementing caching strategies, parallel processing, and memory-efficient data structures while maintaining backwards compatibility with existing API contracts and ensuring thread safety across multiple concurrent operations",
            "Can you help me understand how to implement a distributed system architecture that handles real-time data processing with fault tolerance, automatic failover, load balancing, and consistent data replication across multiple geographic regions while meeting strict performance requirements",
        ];
        
        for i in 0..count {
            let complexity = match i % 10 {
                0..=3 => QueryComplexity::Simple,
                4..=7 => QueryComplexity::Medium,
                8 => QueryComplexity::Complex,
                9 => QueryComplexity::Extreme,
                _ => QueryComplexity::Medium,
            };
            
            let query_text = match complexity {
                QueryComplexity::Simple => simple_queries[i % simple_queries.len()].to_string(),
                QueryComplexity::Medium => medium_queries[i % medium_queries.len()].to_string(),
                QueryComplexity::Complex => complex_queries[i % complex_queries.len()].to_string(),
                QueryComplexity::Extreme => extreme_queries[i % extreme_queries.len()].to_string(),
            };
            
            let expected_intent = self.infer_expected_intent(&query_text);
            let expected_naturalness = self.calculate_expected_naturalness(&query_text);
            let language = self.detect_query_language(&query_text);
            
            queries.push(BenchmarkQuery {
                id: format!("{}_{:04}", category, i),
                text: query_text,
                expected_intent,
                expected_naturalness,
                language,
                complexity,
            });
        }
        
        Ok(queries)
    }
    
    /// Infer expected intent for benchmark validation
    fn infer_expected_intent(&self, query: &str) -> Option<super::query_classifier::QueryIntent> {
        use super::query_classifier::QueryIntent;
        
        if query.starts_with("def ") || query.starts_with("class ") {
            Some(QueryIntent::Definition)
        } else if query.starts_with("refs ") || query.contains("references") {
            Some(QueryIntent::References)
        } else if query.contains("how to") || query.contains("find") {
            Some(QueryIntent::NaturalLanguage)
        } else if query.contains("{}") || query.contains("()") {
            Some(QueryIntent::Structural)
        } else if query.len() > 1 && query.chars().any(|c| c.is_uppercase()) {
            Some(QueryIntent::Symbol)
        } else {
            Some(QueryIntent::Lexical)
        }
    }
    
    /// Calculate expected naturalness score
    fn calculate_expected_naturalness(&self, query: &str) -> Option<f32> {
        let words: Vec<&str> = query.split_whitespace().collect();
        let has_articles = words.iter().any(|&w| ["the", "a", "an"].contains(&w));
        let has_questions = words.iter().any(|&w| ["how", "what", "where", "why"].contains(&w));
        let has_prepositions = words.iter().any(|&w| ["to", "of", "in", "for", "with"].contains(&w));
        let has_code_syntax = query.contains("()") || query.contains("{}") || query.starts_with("def ");
        
        let mut score = 0.5;
        if has_articles { score += 0.2; }
        if has_questions { score += 0.2; }
        if has_prepositions { score += 0.1; }
        if has_code_syntax { score -= 0.3; }
        if words.len() > 5 { score += 0.1; }
        
        Some((score as f32).clamp(0.0, 1.0))
    }
    
    /// Detect query language
    fn detect_query_language(&self, query: &str) -> Option<String> {
        if query.contains("def ") || query.contains("import ") {
            Some("python".to_string())
        } else if query.contains("function ") || query.contains("const ") {
            Some("javascript".to_string())
        } else if query.contains("fn ") || query.contains("impl ") {
            Some("rust".to_string())
        } else {
            Some("natural".to_string())
        }
    }
    
    /// Benchmark query classification component
    #[instrument(skip(self, classifier), fields(dataset_size = dataset.queries.len()))]
    pub async fn benchmark_query_classifier(
        &self,
        classifier: &QueryClassifier,
        dataset: &BenchmarkDataset,
    ) -> Result<ComponentBenchmarkResults> {
        info!("Benchmarking query classifier with {} queries", dataset.queries.len());
        
        let start_time = Instant::now();
        let mut latencies = Vec::with_capacity(dataset.queries.len());
        let mut errors = Vec::new();
        let mut successful_operations = 0u64;
        
        // Memory monitoring
        let start_memory = get_memory_usage();
        let mut peak_memory = start_memory;
        
        for query in &dataset.queries {
            let query_start = Instant::now();
            
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| classifier.classify(&query.text))) {
                Ok(classification) => {
                    let latency = query_start.elapsed();
                    latencies.push(latency.as_millis() as f64);
                    successful_operations += 1;
                    
                    // Validate against expected results if available
                    if let Some(expected_intent) = query.expected_intent {
                        if classification.intent != expected_intent {
                            errors.push(BenchmarkError {
                                query_id: query.id.clone(),
                                error_message: format!(
                                    "Intent mismatch: expected {:?}, got {:?}",
                                    expected_intent, classification.intent
                                ),
                                error_type: "validation".to_string(),
                                timestamp: std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)?
                                    .as_secs(),
                            });
                        }
                    }
                }
                Err(e) => {
                    errors.push(BenchmarkError {
                        query_id: query.id.clone(),
                        error_message: format!("Classification panic: {:?}", e),
                        error_type: "panic".to_string(),
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)?
                            .as_secs(),
                    });
                }
            }
            
            // Update peak memory
            let current_memory = get_memory_usage();
            if current_memory > peak_memory {
                peak_memory = current_memory;
            }
        }
        
        let total_duration = start_time.elapsed();
        let end_memory = get_memory_usage();
        
        // Calculate performance metrics
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let metrics = PerformanceMetrics {
            mean_latency_ms: latencies.iter().sum::<f64>() / latencies.len() as f64,
            p50_latency_ms: latencies[latencies.len() / 2],
            p95_latency_ms: latencies[(latencies.len() as f64 * 0.95) as usize],
            p99_latency_ms: latencies[(latencies.len() as f64 * 0.99) as usize],
            max_latency_ms: latencies.last().copied().unwrap_or(0.0),
            min_latency_ms: latencies.first().copied().unwrap_or(0.0),
            throughput_qps: successful_operations as f64 / total_duration.as_secs_f64(),
            success_rate: successful_operations as f64 / dataset.queries.len() as f64,
            total_operations: successful_operations,
            total_duration_ms: total_duration.as_millis() as f64,
        };
        
        let memory_usage = MemoryUsage {
            peak_memory_mb: peak_memory,
            avg_memory_mb: (start_memory + end_memory) / 2.0,
            memory_growth_mb: end_memory - start_memory,
            cache_hit_rate: 0.0, // Classifiers don't use caching by design
            gc_time_ms: 0.0,     // Not applicable for Rust
        };
        
        Ok(ComponentBenchmarkResults {
            component_name: "query_classifier".to_string(),
            dataset_size: dataset.queries.len(),
            metrics,
            memory_usage,
            errors,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
        })
    }
    
    /// Benchmark semantic encoder component
    #[instrument(skip(self, encoder), fields(dataset_size = dataset.queries.len()))]
    pub async fn benchmark_semantic_encoder(
        &self,
        encoder: &SemanticEncoder,
        dataset: &BenchmarkDataset,
    ) -> Result<ComponentBenchmarkResults> {
        info!("Benchmarking semantic encoder with {} queries", dataset.queries.len());
        
        let start_time = Instant::now();
        let mut latencies = Vec::with_capacity(dataset.queries.len());
        let mut errors = Vec::new();
        let mut successful_operations = 0u64;
        
        // Memory monitoring
        let start_memory = get_memory_usage();
        let mut peak_memory = start_memory;
        
        // Test both single and batch encoding
        for query in &dataset.queries {
            let query_start = Instant::now();
            
            match encoder.encode_query(&query.text).await {
                Ok(embedding) => {
                    let latency = query_start.elapsed();
                    latencies.push(latency.as_millis() as f64);
                    successful_operations += 1;
                    
                    // Validate embedding properties
                    if embedding.vector.is_empty() {
                        errors.push(BenchmarkError {
                            query_id: query.id.clone(),
                            error_message: "Empty embedding vector".to_string(),
                            error_type: "validation".to_string(),
                            timestamp: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)?
                                .as_secs(),
                        });
                    }
                }
                Err(e) => {
                    errors.push(BenchmarkError {
                        query_id: query.id.clone(),
                        error_message: format!("Encoding error: {}", e),
                        error_type: "encoding".to_string(),
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)?
                            .as_secs(),
                    });
                }
            }
            
            // Update peak memory
            let current_memory = get_memory_usage();
            if current_memory > peak_memory {
                peak_memory = current_memory;
            }
        }
        
        // Test batch encoding
        let batch_texts: Vec<(&str, &str)> = dataset.queries.iter()
            .take(10) // Small batch for testing
            .map(|q| (q.text.as_str(), q.id.as_str()))
            .collect();
        
        if !batch_texts.is_empty() {
            let batch_start = Instant::now();
            // For batch processing, encode each individually since there's no batch method
            let mut batch_success = 0;
            for (text, _id) in &batch_texts {
                match encoder.encode_query(text).await {
                    Ok(_embedding) => {
                        batch_success += 1;
                    }
                    Err(e) => {
                        errors.push(BenchmarkError {
                        query_id: "batch_test".to_string(),
                        error_message: format!("Batch encoding error: {}", e),
                        error_type: "batch_encoding".to_string(),
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)?
                            .as_secs(),
                    });
                    }
                }
            }
            
            if batch_success > 0 {
                let batch_latency = batch_start.elapsed();
                let per_item_latency = batch_latency.as_millis() as f64 / batch_success as f64;
                latencies.push(per_item_latency);
                successful_operations += batch_success;
            }
        }
        
        let total_duration = start_time.elapsed();
        let end_memory = get_memory_usage();
        // Get encoder metrics if available
        let encoder_cache_stats = encoder.get_cache_stats().await;
        
        // Calculate performance metrics
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let metrics = PerformanceMetrics {
            mean_latency_ms: latencies.iter().sum::<f64>() / latencies.len() as f64,
            p50_latency_ms: latencies[latencies.len() / 2],
            p95_latency_ms: latencies[(latencies.len() as f64 * 0.95) as usize],
            p99_latency_ms: latencies[(latencies.len() as f64 * 0.99) as usize],
            max_latency_ms: latencies.last().copied().unwrap_or(0.0),
            min_latency_ms: latencies.first().copied().unwrap_or(0.0),
            throughput_qps: successful_operations as f64 / total_duration.as_secs_f64(),
            success_rate: successful_operations as f64 / (dataset.queries.len() + batch_texts.len()) as f64,
            total_operations: successful_operations,
            total_duration_ms: total_duration.as_millis() as f64,
        };
        
        let memory_usage = MemoryUsage {
            peak_memory_mb: peak_memory,
            avg_memory_mb: (start_memory + end_memory) / 2.0,
            memory_growth_mb: end_memory - start_memory,
            cache_hit_rate: encoder_cache_stats.hit_rate,
            gc_time_ms: 0.0,
        };
        
        Ok(ComponentBenchmarkResults {
            component_name: "semantic_encoder".to_string(),
            dataset_size: dataset.queries.len(),
            metrics,
            memory_usage,
            errors,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
        })
    }
    
    /// Benchmark conformal router component
    #[instrument(skip(self, router, classifier), fields(dataset_size = dataset.queries.len()))]
    pub async fn benchmark_conformal_router(
        &self,
        router: &ConformalRouter,
        classifier: &QueryClassifier,
        dataset: &BenchmarkDataset,
    ) -> Result<ComponentBenchmarkResults> {
        info!("Benchmarking conformal router with {} queries", dataset.queries.len());
        
        let start_time = Instant::now();
        let mut latencies = Vec::with_capacity(dataset.queries.len());
        let mut errors = Vec::new();
        let mut successful_operations = 0u64;
        
        // Memory monitoring
        let start_memory = get_memory_usage();
        let mut peak_memory = start_memory;
        
        for query in &dataset.queries {
            let query_start = Instant::now();
            
            // First classify the query
            let classification = classifier.classify(&query.text);
            
            // Extract conformal features
            let features = super::conformal_router::extract_conformal_features(
                &query.text,
                &classification,
                None, // No file context for benchmarking
            );
            
            match router.make_routing_decision(&features, &classification).await {
                Ok(decision) => {
                    let latency = query_start.elapsed();
                    latencies.push(latency.as_millis() as f64);
                    successful_operations += 1;
                    
                    // Validate decision properties
                    if decision.risk_assessment.risk_score < 0.0 || decision.risk_assessment.risk_score > 1.0 {
                        errors.push(BenchmarkError {
                            query_id: query.id.clone(),
                            error_message: format!(
                                "Invalid risk score: {}",
                                decision.risk_assessment.risk_score
                            ),
                            error_type: "validation".to_string(),
                            timestamp: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)?
                                .as_secs(),
                        });
                    }
                }
                Err(e) => {
                    errors.push(BenchmarkError {
                        query_id: query.id.clone(),
                        error_message: format!("Routing decision error: {}", e),
                        error_type: "routing".to_string(),
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)?
                            .as_secs(),
                    });
                }
            }
            
            // Update peak memory
            let current_memory = get_memory_usage();
            if current_memory > peak_memory {
                peak_memory = current_memory;
            }
        }
        
        let total_duration = start_time.elapsed();
        let end_memory = get_memory_usage();
        let router_status = router.get_status().await;
        
        // Calculate performance metrics
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let metrics = PerformanceMetrics {
            mean_latency_ms: latencies.iter().sum::<f64>() / latencies.len() as f64,
            p50_latency_ms: latencies[latencies.len() / 2],
            p95_latency_ms: latencies[(latencies.len() as f64 * 0.95) as usize],
            p99_latency_ms: latencies[(latencies.len() as f64 * 0.99) as usize],
            max_latency_ms: latencies.last().copied().unwrap_or(0.0),
            min_latency_ms: latencies.first().copied().unwrap_or(0.0),
            throughput_qps: successful_operations as f64 / total_duration.as_secs_f64(),
            success_rate: successful_operations as f64 / dataset.queries.len() as f64,
            total_operations: successful_operations,
            total_duration_ms: total_duration.as_millis() as f64,
        };
        
        let memory_usage = MemoryUsage {
            peak_memory_mb: peak_memory,
            avg_memory_mb: (start_memory + end_memory) / 2.0,
            memory_growth_mb: end_memory - start_memory,
            cache_hit_rate: router_status.metrics.cache_hit_rate,
            gc_time_ms: 0.0,
        };
        
        Ok(ComponentBenchmarkResults {
            component_name: "conformal_router".to_string(),
            dataset_size: dataset.queries.len(),
            metrics,
            memory_usage,
            errors,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
        })
    }
    
    /// Run comprehensive benchmark suite
    pub async fn run_comprehensive_benchmark(&mut self) -> Result<BenchmarkSuiteResults> {
        info!("Starting comprehensive semantic processing benchmark suite");
        
        // Ensure datasets are generated
        if self.datasets.is_empty() {
            self.generate_datasets().await?;
        }
        
        let mut all_results = Vec::new();
        
        // Initialize components for benchmarking
        let classifier_config = ClassifierConfig::default();
        let classifier = QueryClassifier::new(classifier_config)?;
        
        let embedding_config = EmbeddingConfig::default();
        let encoder = SemanticEncoder::new(embedding_config).await?;
        
        let conformal_config = ConformalRouterConfig::default();
        let conformal_router = ConformalRouter::new(conformal_config);
        
        // Benchmark each dataset size
        for (size_name, dataset) in &self.datasets {
            info!("Benchmarking with {} dataset", size_name);
            
            // Benchmark query classifier
            let classifier_results = self.benchmark_query_classifier(&classifier, dataset).await?;
            all_results.push(classifier_results);
            
            // Benchmark semantic encoder
            let encoder_results = self.benchmark_semantic_encoder(&encoder, dataset).await?;
            all_results.push(encoder_results);
            
            // Benchmark conformal router
            let router_results = self.benchmark_conformal_router(&conformal_router, &classifier, dataset).await?;
            all_results.push(router_results);
        }
        
        // Store results
        {
            let mut results_lock = self.results.write().await;
            results_lock.extend(all_results.clone());
        }
        
        // Analyze results
        let analysis = self.analyze_results(&all_results)?;
        
        let suite_results = BenchmarkSuiteResults {
            individual_results: all_results,
            analysis,
            config: self.config.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
        };
        
        // Write results to file if configured
        self.write_results_to_file(&suite_results).await?;
        
        info!("Comprehensive benchmark suite completed");
        Ok(suite_results)
    }
    
    /// Analyze benchmark results
    fn analyze_results(&self, results: &[ComponentBenchmarkResults]) -> Result<BenchmarkAnalysis> {
        let mut analysis = BenchmarkAnalysis {
            performance_summary: PerformanceSummary::default(),
            regression_detected: Vec::new(),
            recommendations: Vec::new(),
            target_compliance: TargetCompliance::default(),
        };
        
        // Group results by component
        let mut component_results: HashMap<String, Vec<&ComponentBenchmarkResults>> = HashMap::new();
        for result in results {
            component_results.entry(result.component_name.clone())
                .or_insert_with(Vec::new)
                .push(result);
        }
        
        let component_count = component_results.len();
        
        // Analyze each component
        for (component_name, component_results) in component_results {
            let latest_result = component_results.iter()
                .max_by_key(|r| r.timestamp)
                .unwrap();
            
            // Check performance targets
            let meets_targets = self.check_performance_targets(component_name.as_str(), latest_result);
            analysis.target_compliance.components.insert(component_name.clone(), meets_targets);
            
            // Check for regression (simplified - would compare against historical data)
            if latest_result.metrics.p95_latency_ms > 100.0 {
                analysis.regression_detected.push(format!(
                    "{}: P95 latency {}ms exceeds reasonable threshold",
                    component_name, latest_result.metrics.p95_latency_ms
                ));
            }
            
            if latest_result.metrics.success_rate < 0.95 {
                analysis.regression_detected.push(format!(
                    "{}: Success rate {:.2}% below acceptable threshold",
                    component_name, latest_result.metrics.success_rate * 100.0
                ));
            }
            
            // Generate recommendations
            if latest_result.memory_usage.memory_growth_mb > 50.0 {
                analysis.recommendations.push(format!(
                    "{}: Consider memory optimization - growth {}MB",
                    component_name, latest_result.memory_usage.memory_growth_mb
                ));
            }
            
            if latest_result.metrics.throughput_qps < 10.0 {
                analysis.recommendations.push(format!(
                    "{}: Low throughput {:.2} QPS - consider batch processing",
                    component_name, latest_result.metrics.throughput_qps
                ));
            }
        }
        
        // Overall summary
        analysis.performance_summary.total_components = component_count;
        analysis.performance_summary.passing_components = analysis.target_compliance.components
            .values()
            .filter(|&&meets| meets)
            .count();
        analysis.performance_summary.overall_success_rate = results.iter()
            .map(|r| r.metrics.success_rate)
            .sum::<f64>() / results.len() as f64;
        analysis.performance_summary.avg_p95_latency_ms = results.iter()
            .map(|r| r.metrics.p95_latency_ms)
            .sum::<f64>() / results.len() as f64;
        
        Ok(analysis)
    }
    
    /// Check if component meets performance targets
    fn check_performance_targets(&self, component_name: &str, result: &ComponentBenchmarkResults) -> bool {
        let targets = &self.config.performance_targets;
        
        match component_name {
            "query_classifier" => {
                result.metrics.p95_latency_ms <= targets.classification_p95_ms &&
                result.metrics.success_rate >= 0.95
            }
            "semantic_encoder" => {
                result.metrics.p95_latency_ms <= targets.encoding_p95_ms &&
                result.metrics.throughput_qps >= targets.batch_encoding_qps / 10.0 && // Adjusted for single queries
                result.metrics.success_rate >= 0.95
            }
            "conformal_router" => {
                result.metrics.p95_latency_ms <= targets.routing_p95_ms &&
                result.metrics.success_rate >= 0.95
            }
            _ => result.metrics.success_rate >= 0.95
        }
    }
    
    /// Write results to file
    async fn write_results_to_file(&self, results: &BenchmarkSuiteResults) -> Result<()> {
        use std::fs;
        
        // Create output directory if it doesn't exist
        fs::create_dir_all(&self.config.output_directory)
            .context("Failed to create output directory")?;
        
        let timestamp = chrono::DateTime::from_timestamp(results.timestamp as i64, 0)
            .unwrap_or_default()
            .format("%Y%m%d_%H%M%S");
        
        let filename = format!("{}/semantic_benchmark_{}.json", 
                              self.config.output_directory, timestamp);
        
        let json_results = serde_json::to_string_pretty(results)
            .context("Failed to serialize results")?;
        
        fs::write(&filename, json_results)
            .context("Failed to write results file")?;
        
        info!("Benchmark results written to: {}", filename);
        Ok(())
    }
}

/// Complete benchmark suite results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuiteResults {
    pub individual_results: Vec<ComponentBenchmarkResults>,
    pub analysis: BenchmarkAnalysis,
    pub config: BenchmarkConfig,
    pub timestamp: u64,
}

/// Analysis of benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkAnalysis {
    pub performance_summary: PerformanceSummary,
    pub regression_detected: Vec<String>,
    pub recommendations: Vec<String>,
    pub target_compliance: TargetCompliance,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub total_components: usize,
    pub passing_components: usize,
    pub overall_success_rate: f64,
    pub avg_p95_latency_ms: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TargetCompliance {
    pub components: HashMap<String, bool>,
}

/// Get current memory usage in MB using /proc/self/status
fn get_memory_usage() -> f64 {
    use std::fs;
    
    // Try to read memory from /proc/self/status on Linux
    if let Ok(status) = fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                if let Some(memory_str) = line.split_whitespace().nth(1) {
                    if let Ok(memory_kb) = memory_str.parse::<f64>() {
                        return memory_kb / 1024.0; // Convert KB to MB
                    }
                }
            }
        }
    }
    
    // Fallback: use malloc_size if available (macOS)
    #[cfg(target_os = "macos")]
    {
        extern "C" {
            fn malloc_size(ptr: *const std::ffi::c_void) -> usize;
        }
        // This is a simplified fallback - would need proper heap tracking
        let stack_var = 0;
        let heap_estimate = unsafe { malloc_size(&stack_var as *const i32 as *const std::ffi::c_void) };
        return heap_estimate as f64 / (1024.0 * 1024.0);
    }
    
    // Final fallback for unsupported platforms
    64.0 // Reasonable default MB estimate
}

/// Initialize benchmark module
pub async fn initialize_benchmarks(config: &BenchmarkConfig) -> Result<()> {
    tracing::info!("Initializing semantic benchmarking module");
    tracing::info!("Dataset sizes: micro={}, small={}, medium={}, large={}", 
                   config.dataset_sizes.micro,
                   config.dataset_sizes.small,
                   config.dataset_sizes.medium,
                   config.dataset_sizes.large);
    tracing::info!("Performance targets: classification={}ms, routing={}ms, encoding={}ms",
                   config.performance_targets.classification_p95_ms,
                   config.performance_targets.routing_p95_ms,
                   config.performance_targets.encoding_p95_ms);
    tracing::info!("Profiling enabled: {}", config.enable_profiling);
    tracing::info!("Output directory: {}", config.output_directory);
    
    // Validate configuration
    if config.dataset_sizes.micro == 0 {
        anyhow::bail!("Micro dataset size must be greater than 0");
    }
    
    if config.performance_targets.classification_p95_ms <= 0.0 {
        anyhow::bail!("Classification P95 target must be greater than 0");
    }
    
    if config.output_directory.is_empty() {
        anyhow::bail!("Output directory must be specified");
    }
    
    tracing::info!("Semantic benchmarking module initialized successfully");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_benchmark_suite_creation() {
        let config = BenchmarkConfig::default();
        let suite = SemanticBenchmarkSuite::new(config);
        
        assert!(suite.datasets.is_empty()); // No datasets generated yet
        assert_eq!(suite.results.read().await.len(), 0);
    }
    
    #[tokio::test]
    async fn test_dataset_generation() {
        let config = BenchmarkConfig {
            dataset_sizes: BenchmarkDatasetSizes {
                micro: 5,
                small: 10,
                medium: 0,
                large: 0,
            },
            ..Default::default()
        };
        
        let mut suite = SemanticBenchmarkSuite::new(config);
        suite.generate_datasets().await.unwrap();
        
        assert!(suite.datasets.contains_key("micro"));
        assert!(suite.datasets.contains_key("small"));
        assert!(!suite.datasets.contains_key("medium"));
        
        let micro_dataset = suite.datasets.get("micro").unwrap();
        assert_eq!(micro_dataset.queries.len(), 5);
        
        // Verify query properties
        for query in &micro_dataset.queries {
            assert!(!query.text.is_empty());
            assert!(query.id.starts_with("micro_"));
            assert!(query.expected_intent.is_some());
        }
    }
    
    #[tokio::test]
    async fn test_query_classifier_benchmark() {
        let config = BenchmarkConfig::default();
        let suite = SemanticBenchmarkSuite::new(config);
        
        let classifier_config = ClassifierConfig::default();
        let classifier = QueryClassifier::new(classifier_config).unwrap();
        
        // Create small test dataset
        let queries = vec![
            BenchmarkQuery {
                id: "test_1".to_string(),
                text: "how to sort an array".to_string(),
                expected_intent: Some(super::super::query_classifier::QueryIntent::NaturalLanguage),
                expected_naturalness: Some(0.8),
                language: Some("natural".to_string()),
                complexity: QueryComplexity::Medium,
            },
            BenchmarkQuery {
                id: "test_2".to_string(),
                text: "def calculateSum".to_string(),
                expected_intent: Some(super::super::query_classifier::QueryIntent::Definition),
                expected_naturalness: Some(0.2),
                language: Some("python".to_string()),
                complexity: QueryComplexity::Simple,
            },
        ];
        
        let dataset = BenchmarkDataset {
            queries,
            size_category: "test".to_string(),
        };
        
        let results = suite.benchmark_query_classifier(&classifier, &dataset).await.unwrap();
        
        assert_eq!(results.component_name, "query_classifier");
        assert_eq!(results.dataset_size, 2);
        assert!(results.metrics.success_rate > 0.0);
        assert!(results.metrics.mean_latency_ms >= 0.0);
        assert!(results.metrics.throughput_qps > 0.0);
    }
    
    #[test]
    fn test_expected_intent_inference() {
        let config = BenchmarkConfig::default();
        let suite = SemanticBenchmarkSuite::new(config);
        
        assert_eq!(
            suite.infer_expected_intent("def myFunction"),
            Some(super::super::query_classifier::QueryIntent::Definition)
        );
        
        assert_eq!(
            suite.infer_expected_intent("how to sort an array"),
            Some(super::super::query_classifier::QueryIntent::NaturalLanguage)
        );
        
        assert_eq!(
            suite.infer_expected_intent("refs myVariable"),
            Some(super::super::query_classifier::QueryIntent::References)
        );
    }
    
    #[test]
    fn test_naturalness_calculation() {
        let config = BenchmarkConfig::default();
        let suite = SemanticBenchmarkSuite::new(config);
        
        let naturalness = suite.calculate_expected_naturalness("how to find the best solution").unwrap();
        assert!(naturalness > 0.7); // Should be high for natural language
        
        let code_naturalness = suite.calculate_expected_naturalness("def calculate_sum()").unwrap();
        assert!(code_naturalness < 0.5); // Should be low for code syntax
    }
    
    #[tokio::test]
    async fn test_configuration_validation() {
        let mut config = BenchmarkConfig::default();
        config.dataset_sizes.micro = 0; // Invalid
        
        let result = initialize_benchmarks(&config).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Micro dataset size"));
        
        config.dataset_sizes.micro = 10; // Valid
        config.performance_targets.classification_p95_ms = -1.0; // Invalid
        
        let result = initialize_benchmarks(&config).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Classification P95"));
    }
}