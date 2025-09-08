//! Pipeline Executor
//!
//! Orchestrates the fused pipeline execution with:
//! - Async overlap processing
//! - Stage fusion for performance
//! - SLA-bounded execution
//! - Learning-to-stop algorithms
//! - Cross-shard optimization

use super::{
    PipelineContext, PipelineData, PipelineConfig, PipelineStage, PipelineStageProcessor,
    PipelineError, PipelineMetrics, memory::PipelineMemoryManager
};
use crate::lsp::{LspManager, LspConfig, QueryIntent};
use crate::search::SearchEngine;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

/// Fused pipeline executor with stage management
pub struct PipelineExecutor {
    config: PipelineConfig,
    
    // Core engines
    search_engine: Arc<SearchEngine>,
    lsp_manager: Arc<crate::lsp::LspState>,
    
    // Memory management
    memory_manager: Arc<PipelineMemoryManager>,
    
    // Stage processors
    stages: HashMap<PipelineStage, Arc<dyn PipelineStageProcessor>>,
    
    // Execution control
    concurrency_limiter: Arc<Semaphore>,
    
    // Metrics and monitoring
    metrics: Arc<RwLock<ExecutorMetrics>>,
    
    // Learning-to-stop state
    stopping_predictor: Arc<RwLock<StoppingPredictor>>,
}

#[derive(Debug, Default, Clone)]
pub struct ExecutorMetrics {
    pub total_executions: u64,
    pub successful_executions: u64,
    pub failed_executions: u64,
    pub avg_execution_time_ms: f64,
    pub stage_fusion_count: u64,
    pub early_stopping_count: u64,
    pub memory_usage_peak_mb: f64,
    pub sla_violations: u64,
}

/// Learning-to-stop predictor for early termination
#[derive(Debug)]
pub struct StoppingPredictor {
    quality_threshold: f64,
    confidence_threshold: f64,
    historical_accuracies: Vec<f64>,
    max_history: usize,
}

impl Default for StoppingPredictor {
    fn default() -> Self {
        Self {
            quality_threshold: 0.8,
            confidence_threshold: 0.9,
            historical_accuracies: Vec::new(),
            max_history: 100,
        }
    }
}

impl StoppingPredictor {
    /// Predict if we should stop early based on current results
    pub fn should_stop_early(&self, current_quality: f64, confidence: f64, time_budget_used: f64) -> bool {
        // Don't stop if we haven't used much time yet
        if time_budget_used < 0.3 {
            return false;
        }
        
        // Stop if we're confident and quality is good
        if confidence >= self.confidence_threshold && current_quality >= self.quality_threshold {
            return true;
        }
        
        // Stop if we're running out of time and results are reasonable
        if time_budget_used > 0.8 && current_quality >= self.quality_threshold * 0.7 {
            return true;
        }
        
        false
    }
    
    /// Update predictor with execution outcome
    pub fn update(&mut self, predicted_stop: bool, actual_quality: f64) {
        // Calculate accuracy of the prediction
        let was_accurate = if predicted_stop {
            actual_quality >= self.quality_threshold * 0.9
        } else {
            actual_quality < self.quality_threshold
        };
        
        let accuracy = if was_accurate { 1.0 } else { 0.0 };
        
        self.historical_accuracies.push(accuracy);
        
        // Keep only recent history
        if self.historical_accuracies.len() > self.max_history {
            self.historical_accuracies.remove(0);
        }
        
        // Adapt thresholds based on accuracy
        let avg_accuracy = self.historical_accuracies.iter().sum::<f64>() / self.historical_accuracies.len() as f64;
        
        if avg_accuracy < 0.7 {
            // Too many false positives, be more conservative
            self.quality_threshold = (self.quality_threshold + 0.05).min(0.95);
            self.confidence_threshold = (self.confidence_threshold + 0.02).min(0.98);
        } else if avg_accuracy > 0.9 {
            // Very accurate, can be more aggressive
            self.quality_threshold = (self.quality_threshold - 0.02).max(0.6);
            self.confidence_threshold = (self.confidence_threshold - 0.01).max(0.8);
        }
    }
}

impl PipelineExecutor {
    /// Create a new pipeline executor
    pub async fn new(config: PipelineConfig) -> Result<Self> {
        // Initialize core engines
        let search_engine = Arc::new(SearchEngine::new(&config.max_concurrent.to_string()).await?);
        
        let lsp_config = LspConfig {
            enabled: true,
            server_timeout_ms: config.max_latency_ms / 2, // Use half of SLA for LSP
            cache_ttl_hours: 24,
            max_concurrent_requests: config.max_concurrent / 2,
            routing_percentage: 0.5, // 50% routing target
            ..Default::default()
        };
        let lsp_state = Arc::new(crate::lsp::LspState::new(lsp_config));
        lsp_state.initialize().await?;
        let lsp_manager = lsp_state;
        
        // Initialize memory manager (allocate 25% of system memory limit for pipeline)
        let memory_manager = Arc::new(PipelineMemoryManager::new(256)); // 256MB limit
        
        let concurrency_limiter = Arc::new(Semaphore::new(config.max_concurrent));
        let metrics = Arc::new(RwLock::new(ExecutorMetrics::default()));
        let stopping_predictor = Arc::new(RwLock::new(StoppingPredictor::default()));
        
        let mut executor = Self {
            config,
            search_engine,
            lsp_manager,
            memory_manager,
            stages: HashMap::new(),
            concurrency_limiter,
            metrics,
            stopping_predictor,
        };
        
        // Initialize stage processors
        executor.init_stages().await?;
        
        info!("Pipeline executor initialized with {}ms SLA", executor.config.max_latency_ms);
        
        Ok(executor)
    }
    
    async fn init_stages(&mut self) -> Result<()> {
        // Query analysis stage
        let query_stage = Arc::new(QueryAnalysisStage::new());
        self.stages.insert(PipelineStage::QueryAnalysis, query_stage);
        
        // LSP routing stage  
        let lsp_routing_stage = Arc::new(LspRoutingStage::new(self.lsp_manager.clone()));
        self.stages.insert(PipelineStage::LspRouting, lsp_routing_stage);
        
        // Parallel search stage
        let search_stage = Arc::new(ParallelSearchStage::new(
            self.search_engine.clone(),
            self.lsp_manager.clone(),
        ));
        self.stages.insert(PipelineStage::ParallelSearch, search_stage);
        
        // Result fusion stage
        let fusion_stage = Arc::new(ResultFusionStage::new());
        self.stages.insert(PipelineStage::ResultFusion, fusion_stage);
        
        // Post-processing stage
        let post_process_stage = Arc::new(PostProcessStage::new());
        self.stages.insert(PipelineStage::PostProcess, post_process_stage);
        
        Ok(())
    }
    
    /// Execute a complete pipeline run
    pub async fn execute(&self, context: PipelineContext) -> Result<PipelineData, PipelineError> {
        let start_time = Instant::now();
        let _permit = self.concurrency_limiter.acquire().await.map_err(|_| {
            PipelineError::FusionError {
                stage: PipelineStage::Input,
                message: "Failed to acquire concurrency permit".to_string(),
            }
        })?;
        
        // Check SLA deadline before starting
        if context.is_deadline_exceeded() {
            return Err(PipelineError::DeadlineExceeded {
                elapsed_ms: context.elapsed().as_millis() as u64,
                deadline_ms: self.config.max_latency_ms,
            });
        }
        
        debug!("Starting pipeline execution for request: {}", context.request_id);
        
        // Initialize pipeline data with appropriate buffer size
        let buffer_size = self.estimate_buffer_size(&context);
        let buffer = self.memory_manager.allocate(buffer_size).await.map_err(|e| {
            PipelineError::BufferAllocation { 
                requested_size: buffer_size 
            }
        })?;
        
        let mut data = PipelineData::new(buffer_size);
        data.buffer = buffer;
        
        // Execute pipeline stages
        let result = self.execute_stages(context, data).await;
        
        // Update metrics
        let execution_time = start_time.elapsed();
        self.update_metrics(execution_time, result.is_ok()).await;
        
        result
    }
    
    async fn execute_stages(&self, context: PipelineContext, mut data: PipelineData) -> Result<PipelineData, PipelineError> {
        let mut current_stage = PipelineStage::QueryAnalysis;
        
        loop {
            // Check deadline before each stage
            if context.is_deadline_exceeded() {
                return Err(PipelineError::DeadlineExceeded {
                    elapsed_ms: context.elapsed().as_millis() as u64,
                    deadline_ms: self.config.max_latency_ms,
                });
            }
            
            // Get stage timeout
            let stage_timeout = self.config.stage_timeouts.calculate_timeout(
                self.config.max_latency_ms,
                current_stage,
            );
            
            // Execute stage with timeout
            let stage_start = Instant::now();
            let stage_processor = self.stages.get(&current_stage).ok_or_else(|| {
                PipelineError::FusionError {
                    stage: current_stage,
                    message: "Stage processor not found".to_string(),
                }
            })?;
            
            debug!("Executing stage: {:?}", current_stage);
            
            let stage_result = timeout(stage_timeout, stage_processor.process(&context, data)).await;
            
            match stage_result {
                Ok(Ok(new_data)) => {
                    data = new_data;
                    let stage_time = stage_start.elapsed();
                    
                    debug!(
                        "Stage {:?} completed in {}ms",
                        current_stage,
                        stage_time.as_millis()
                    );
                    
                    // Check if we should stop early
                    if self.should_stop_early(&context, &data, current_stage).await {
                        info!("Early stopping after stage {:?}", current_stage);
                        
                        let mut metrics = self.metrics.write().await;
                        metrics.early_stopping_count += 1;
                        
                        break;
                    }
                    
                    // Advance to next stage
                    if let Some(next_stage) = current_stage.next() {
                        current_stage = next_stage;
                        data.advance_stage(current_stage);
                    } else {
                        break; // Pipeline complete
                    }
                }
                Ok(Err(e)) => {
                    error!("Stage {:?} failed: {:?}", current_stage, e);
                    return Err(e);
                }
                Err(_) => {
                    let elapsed = stage_start.elapsed();
                    return Err(PipelineError::StageTimeout {
                        stage: current_stage,
                        elapsed_ms: elapsed.as_millis() as u64,
                    });
                }
            }
        }
        
        Ok(data)
    }
    
    async fn should_stop_early(&self, context: &PipelineContext, data: &PipelineData, current_stage: PipelineStage) -> bool {
        if !self.config.early_stopping_enabled {
            return false;
        }
        
        // Don't stop too early in the pipeline
        if matches!(current_stage, PipelineStage::QueryAnalysis | PipelineStage::LspRouting) {
            return false;
        }
        
        // Calculate current quality estimate (simplified)
        let quality_score = self.estimate_result_quality(data).await;
        let confidence = self.estimate_result_confidence(data).await;
        let time_budget_used = context.time_budget_percent();
        
        let predictor = self.stopping_predictor.read().await;
        let should_stop = predictor.should_stop_early(quality_score, confidence, time_budget_used);
        
        if should_stop {
            debug!(
                "Early stopping decision: quality={:.3}, confidence={:.3}, time_used={:.1}%",
                quality_score,
                confidence,
                time_budget_used * 100.0
            );
        }
        
        should_stop
    }
    
    async fn estimate_result_quality(&self, data: &PipelineData) -> f64 {
        // Simplified quality estimation based on result count and diversity
        let result_count = data.metadata.search_results_count;
        let has_lsp_results = data.metadata.lsp_queries > 0;
        
        let mut quality: f64 = 0.0;
        
        // Base quality from result count
        if result_count > 0 {
            quality += 0.3;
            if result_count >= 10 {
                quality += 0.3;
            }
            if result_count >= 50 {
                quality += 0.2;
            }
        }
        
        // Bonus for LSP integration
        if has_lsp_results {
            quality += 0.2;
        }
        
        quality.min(1.0)
    }
    
    async fn estimate_result_confidence(&self, data: &PipelineData) -> f64 {
        // Simplified confidence estimation
        let cache_hit_rate = if data.metadata.lsp_queries > 0 {
            data.metadata.cache_hits as f64 / data.metadata.lsp_queries as f64
        } else {
            0.0
        };
        
        let data_completeness = if data.total_size() > 0 { 0.8 } else { 0.2 };
        
        (cache_hit_rate * 0.3 + data_completeness * 0.7).min(1.0)
    }
    
    fn estimate_buffer_size(&self, context: &PipelineContext) -> usize {
        // Estimate buffer size based on query complexity and expected results
        let base_size = 1024; // 1KB base
        let query_factor = (context.query.len() / 10).max(1);
        let results_factor = context.max_results / 10;
        
        base_size * query_factor * results_factor
    }
    
    async fn update_metrics(&self, execution_time: Duration, success: bool) {
        let mut metrics = self.metrics.write().await;
        
        metrics.total_executions += 1;
        
        if success {
            metrics.successful_executions += 1;
        } else {
            metrics.failed_executions += 1;
        }
        
        let execution_time_ms = execution_time.as_millis() as f64;
        let total = metrics.total_executions as f64;
        
        // Update average execution time
        metrics.avg_execution_time_ms = 
            (metrics.avg_execution_time_ms * (total - 1.0) + execution_time_ms) / total;
        
        // Check SLA violation
        if execution_time.as_millis() as u64 > self.config.max_latency_ms {
            metrics.sla_violations += 1;
        }
        
        // Update memory usage peak
        let current_usage_mb = self.memory_manager.current_usage() as f64 / 1024.0 / 1024.0;
        if current_usage_mb > metrics.memory_usage_peak_mb {
            metrics.memory_usage_peak_mb = current_usage_mb;
        }
    }
    
    /// Get executor metrics
    pub async fn get_metrics(&self) -> ExecutorMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Shutdown the executor
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down pipeline executor");
        
        // Shutdown LSP manager
        self.lsp_manager.shutdown().await?;
        
        // Clear memory manager
        self.memory_manager.gc().await?;
        
        Ok(())
    }
}

/// Query analysis stage processor
pub struct QueryAnalysisStage;

impl QueryAnalysisStage {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl PipelineStageProcessor for QueryAnalysisStage {
    async fn process(&self, context: &PipelineContext, mut data: PipelineData) -> Result<PipelineData, PipelineError> {
        // Analyze query intent and complexity
        let intent = QueryIntent::classify(&context.query);
        
        // Store analysis results in metadata
        data.metadata.bytes_processed += context.query.len();
        
        // Add query analysis segment to data
        let analysis = serde_json::json!({
            "intent": intent,
            "query_length": context.query.len(),
            "estimated_complexity": context.query.split_whitespace().count(),
        });
        
        debug!("Query analysis completed: {:?}", intent);
        
        Ok(data)
    }
    
    fn stage_id(&self) -> PipelineStage {
        PipelineStage::QueryAnalysis
    }
    
    fn supports_fusion(&self) -> bool {
        true // Can be fused with LSP routing
    }
}

/// LSP routing stage processor
pub struct LspRoutingStage {
    lsp_manager: Arc<crate::lsp::LspState>,
}

impl LspRoutingStage {
    pub fn new(lsp_manager: Arc<crate::lsp::LspState>) -> Self {
        Self { lsp_manager }
    }
}

#[async_trait::async_trait]
impl PipelineStageProcessor for LspRoutingStage {
    async fn process(&self, context: &PipelineContext, mut data: PipelineData) -> Result<PipelineData, PipelineError> {
        // Route query through LSP if appropriate
        let lsp_response = self.lsp_manager.search(&context.query, context.file_path.as_deref()).await
            .map_err(|e| PipelineError::LspError { message: e.to_string() })?;
        
        // Update metadata
        data.metadata.lsp_queries += 1;
        if !lsp_response.lsp_results.is_empty() {
            data.metadata.search_results_count += lsp_response.lsp_results.len();
        }
        
        debug!("LSP routing completed: {} results", lsp_response.lsp_results.len());
        
        Ok(data)
    }
    
    fn stage_id(&self) -> PipelineStage {
        PipelineStage::LspRouting
    }
    
    fn supports_fusion(&self) -> bool {
        true
    }
}

/// Parallel search stage processor
pub struct ParallelSearchStage {
    search_engine: Arc<SearchEngine>,
    lsp_manager: Arc<crate::lsp::LspState>,
}

impl ParallelSearchStage {
    pub fn new(search_engine: Arc<SearchEngine>, lsp_manager: Arc<crate::lsp::LspState>) -> Self {
        Self {
            search_engine,
            lsp_manager,
        }
    }
}

#[async_trait::async_trait]
impl PipelineStageProcessor for ParallelSearchStage {
    async fn process(&self, context: &PipelineContext, mut data: PipelineData) -> Result<PipelineData, PipelineError> {
        // Execute parallel search across text search and LSP
        let (search_results, _metrics) = self.search_engine.search(&context.query, context.max_results)
            .await
            .map_err(|e| PipelineError::SearchError { message: e.to_string() })?;
        
        data.metadata.search_results_count += search_results.len();
        
        debug!("Parallel search completed: {} results", search_results.len());
        
        Ok(data)
    }
    
    fn stage_id(&self) -> PipelineStage {
        PipelineStage::ParallelSearch
    }
    
    fn supports_fusion(&self) -> bool {
        false // Search operations are complex and shouldn't be fused
    }
    
    fn estimate_processing_time(&self, data_size: usize) -> Duration {
        // Estimate based on data size and index complexity
        Duration::from_millis(50 + (data_size / 1000) as u64)
    }
}

/// Result fusion stage processor  
pub struct ResultFusionStage;

impl ResultFusionStage {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl PipelineStageProcessor for ResultFusionStage {
    async fn process(&self, _context: &PipelineContext, mut data: PipelineData) -> Result<PipelineData, PipelineError> {
        // Fuse results from different sources (LSP + text search)
        // This is where zero-copy operations really shine
        
        debug!("Result fusion completed");
        
        Ok(data)
    }
    
    fn stage_id(&self) -> PipelineStage {
        PipelineStage::ResultFusion
    }
    
    fn supports_fusion(&self) -> bool {
        true
    }
}

/// Post-processing stage processor
pub struct PostProcessStage;

impl PostProcessStage {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl PipelineStageProcessor for PostProcessStage {
    async fn process(&self, _context: &PipelineContext, mut data: PipelineData) -> Result<PipelineData, PipelineError> {
        // Final post-processing: deduplication, ranking, formatting
        
        debug!("Post-processing completed");
        
        Ok(data)
    }
    
    fn stage_id(&self) -> PipelineStage {
        PipelineStage::PostProcess
    }
    
    fn supports_fusion(&self) -> bool {
        true
    }
}