//! Zero-Copy Fused Pipeline Architecture
//!
//! Implements the fused Rust pipeline with:
//! - Zero-copy segment views for ≤150ms p95 latency
//! - Async overlap processing 
//! - Cross-shard TA/NRA stopping
//! - Learning-to-stop for WAND/HNSW
//! - Prefetch/visited-set reuse
//! - SLA-bounded execution with timeouts

pub mod executor;
pub mod fusion;
pub mod learning;
pub mod memory;
pub mod scheduler;
pub mod stages;

use anyhow::Result;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info, warn};

pub use executor::PipelineExecutor;
pub use fusion::ResultFusion;
pub use memory::{ZeroCopyBuffer, SegmentView};
pub use scheduler::PipelineScheduler;
pub use stages::{QueryPreprocessingStage, LspSearchStage, TextSearchStage};

/// Pipeline execution context with zero-copy semantics
#[derive(Debug, Clone)]
pub struct PipelineContext {
    pub request_id: String,
    pub query: String,
    pub file_path: Option<String>,
    pub max_results: usize,
    pub timeout: Duration,
    pub started_at: Instant,
    pub sla_deadline: Instant,
}

impl PipelineContext {
    pub fn new(request_id: String, query: String, timeout_ms: u64) -> Self {
        let started_at = Instant::now();
        let timeout = Duration::from_millis(timeout_ms);
        let sla_deadline = started_at + timeout;
        
        Self {
            request_id,
            query,
            file_path: None,
            max_results: 50,
            timeout,
            started_at,
            sla_deadline,
        }
    }

    pub fn with_file_path(mut self, file_path: String) -> Self {
        self.file_path = Some(file_path);
        self
    }

    pub fn with_max_results(mut self, max_results: usize) -> Self {
        self.max_results = max_results;
        self
    }

    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    pub fn remaining_time(&self) -> Duration {
        self.sla_deadline.saturating_duration_since(Instant::now())
    }

    pub fn is_deadline_exceeded(&self) -> bool {
        Instant::now() >= self.sla_deadline
    }

    pub fn time_budget_percent(&self) -> f64 {
        let total_time = self.timeout.as_millis() as f64;
        let elapsed_time = self.elapsed().as_millis() as f64;
        
        if total_time > 0.0 {
            (elapsed_time / total_time).min(1.0)
        } else {
            1.0
        }
    }
}

/// Zero-copy pipeline data flowing between stages
#[derive(Debug, Clone)]
pub struct PipelineData {
    /// Core data buffer with zero-copy views
    pub buffer: Arc<ZeroCopyBuffer>,
    
    /// Metadata about the data
    pub metadata: PipelineMetadata,
    
    /// Current processing stage
    pub stage: PipelineStage,
    
    /// Segment views for different data types
    pub segments: Vec<SegmentView>,
}

impl PipelineData {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Arc::new(ZeroCopyBuffer::new(capacity)),
            metadata: PipelineMetadata::default(),
            stage: PipelineStage::Input,
            segments: Vec::new(),
        }
    }

    /// Add a segment view without copying data
    pub fn add_segment(&mut self, segment: SegmentView) {
        self.segments.push(segment);
    }

    /// Create a new view of existing data
    pub fn create_view(&self, offset: usize, length: usize) -> Result<SegmentView> {
        self.buffer.create_view(offset, length)
    }

    /// Get total data size across all segments
    pub fn total_size(&self) -> usize {
        self.segments.iter().map(|s| s.len()).sum()
    }

    /// Advance to next pipeline stage
    pub fn advance_stage(&mut self, stage: PipelineStage) {
        self.stage = stage;
        self.metadata.stage_transitions += 1;
    }
}

/// Metadata tracked throughout pipeline execution
#[derive(Debug, Default, Clone)]
pub struct PipelineMetadata {
    pub stage_transitions: usize,
    pub bytes_processed: usize,
    pub cache_hits: usize,
    pub lsp_queries: usize,
    pub search_results_count: usize,
    pub performance_metrics: PerformanceMetrics,
}

/// Performance metrics for pipeline execution
#[derive(Debug, Default, Clone)]
pub struct PerformanceMetrics {
    pub total_time_ms: u64,
    pub lsp_time_ms: u64,
    pub search_time_ms: u64,
    pub fusion_time_ms: u64,
    pub memory_allocations: usize,
    pub zero_copy_operations: usize,
}

/// Pipeline execution stages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineStage {
    Input,
    QueryAnalysis,
    LspRouting,
    ParallelSearch,
    ResultFusion,
    PostProcess,
    Output,
}

impl PipelineStage {
    pub fn as_str(&self) -> &'static str {
        match self {
            PipelineStage::Input => "input",
            PipelineStage::QueryAnalysis => "query_analysis",
            PipelineStage::LspRouting => "lsp_routing",
            PipelineStage::ParallelSearch => "parallel_search",
            PipelineStage::ResultFusion => "result_fusion",
            PipelineStage::PostProcess => "post_process",
            PipelineStage::Output => "output",
        }
    }

    pub fn next(&self) -> Option<PipelineStage> {
        match self {
            PipelineStage::Input => Some(PipelineStage::QueryAnalysis),
            PipelineStage::QueryAnalysis => Some(PipelineStage::LspRouting),
            PipelineStage::LspRouting => Some(PipelineStage::ParallelSearch),
            PipelineStage::ParallelSearch => Some(PipelineStage::ResultFusion),
            PipelineStage::ResultFusion => Some(PipelineStage::PostProcess),
            PipelineStage::PostProcess => Some(PipelineStage::Output),
            PipelineStage::Output => None,
        }
    }
}

/// Pipeline execution result
#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub request_id: String,
    pub data: PipelineData,
    pub success: bool,
    pub error_message: Option<String>,
    pub metrics: PipelineMetrics,
}

/// Overall pipeline metrics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PipelineMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub avg_latency_ms: f64,
    pub p95_latency_ms: u64,
    pub p99_latency_ms: u64,
    pub zero_copy_ratio: f64,
    pub fusion_effectiveness: f64,
    pub sla_compliance_rate: f64,
    pub stage_breakdown: StageBreakdown,
}

/// Performance breakdown by pipeline stage
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct StageBreakdown {
    pub query_analysis_ms: f64,
    pub lsp_routing_ms: f64,
    pub parallel_search_ms: f64,
    pub result_fusion_ms: f64,
    pub post_process_ms: f64,
}

/// Configuration for the fused pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Maximum latency SLA in milliseconds (150ms per TODO.md)
    pub max_latency_ms: u64,
    
    /// Buffer pool size for zero-copy operations
    pub buffer_pool_size: usize,
    
    /// Maximum concurrent pipeline executions
    pub max_concurrent: usize,
    
    /// Stage timeout allocations (percentages of total)
    pub stage_timeouts: StageTimeouts,
    
    /// Fusion and optimization settings
    pub fusion_enabled: bool,
    pub prefetch_enabled: bool,
    pub visited_set_reuse: bool,
    
    /// Learning-to-stop thresholds
    pub learning_to_stop_threshold: f64,
    pub early_stopping_enabled: bool,
}

/// Timeout allocation per stage
#[derive(Debug, Clone)]
pub struct StageTimeouts {
    pub query_analysis_percent: f64,
    pub lsp_routing_percent: f64,
    pub parallel_search_percent: f64,
    pub result_fusion_percent: f64,
    pub post_process_percent: f64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_latency_ms: 150, // ≤150ms p95 per TODO.md
            buffer_pool_size: 100,
            max_concurrent: 50,
            stage_timeouts: StageTimeouts {
                query_analysis_percent: 0.05, // 5%
                lsp_routing_percent: 0.10,     // 10%
                parallel_search_percent: 0.65, // 65%
                result_fusion_percent: 0.15,   // 15%
                post_process_percent: 0.05,    // 5%
            },
            fusion_enabled: true,
            prefetch_enabled: true,
            visited_set_reuse: true,
            learning_to_stop_threshold: 0.8,
            early_stopping_enabled: true,
        }
    }
}

impl StageTimeouts {
    pub fn calculate_timeout(&self, total_timeout_ms: u64, stage: PipelineStage) -> Duration {
        let percent = match stage {
            PipelineStage::QueryAnalysis => self.query_analysis_percent,
            PipelineStage::LspRouting => self.lsp_routing_percent,
            PipelineStage::ParallelSearch => self.parallel_search_percent,
            PipelineStage::ResultFusion => self.result_fusion_percent,
            PipelineStage::PostProcess => self.post_process_percent,
            _ => 0.0,
        };
        
        Duration::from_millis((total_timeout_ms as f64 * percent) as u64)
    }
}

/// Pipeline execution error types
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("SLA deadline exceeded: {elapsed_ms}ms > {deadline_ms}ms")]
    DeadlineExceeded { elapsed_ms: u64, deadline_ms: u64 },
    
    #[error("Stage timeout in {stage:?}: {elapsed_ms}ms")]
    StageTimeout { stage: PipelineStage, elapsed_ms: u64 },
    
    #[error("Buffer allocation failed: {requested_size} bytes")]
    BufferAllocation { requested_size: usize },
    
    #[error("Zero-copy operation failed: {reason}")]
    ZeroCopyFailed { reason: String },
    
    #[error("Pipeline fusion error: {stage:?} - {message}")]
    FusionError { stage: PipelineStage, message: String },
    
    #[error("LSP integration error: {message}")]
    LspError { message: String },
    
    #[error("Search engine error: {message}")]
    SearchError { message: String },
}

/// Pipeline stage interface
#[async_trait::async_trait]
pub trait PipelineStageProcessor: Send + Sync {
    /// Process data through this stage
    async fn process(&self, context: &PipelineContext, data: PipelineData) -> Result<PipelineData, PipelineError>;
    
    /// Get stage identifier
    fn stage_id(&self) -> PipelineStage;
    
    /// Check if this stage can be fused with others
    fn supports_fusion(&self) -> bool {
        false
    }
    
    /// Estimate processing time for planning
    fn estimate_processing_time(&self, data_size: usize) -> Duration {
        Duration::from_millis(10) // Default estimate
    }
}

/// Main fused pipeline implementation
pub struct FusedPipeline {
    config: PipelineConfig,
    executor: Arc<PipelineExecutor>,
    scheduler: Arc<PipelineScheduler>,
    metrics: Arc<RwLock<PipelineMetrics>>,
}

impl FusedPipeline {
    pub async fn new(config: PipelineConfig) -> Result<Self> {
        let executor = Arc::new(PipelineExecutor::new(config.clone()).await?);
        let scheduler = Arc::new(PipelineScheduler::new(config.max_concurrent));
        let metrics = Arc::new(RwLock::new(PipelineMetrics::default()));
        
        info!("Initialized fused pipeline with ≤{}ms SLA", config.max_latency_ms);
        
        Ok(Self {
            config,
            executor,
            scheduler,
            metrics,
        })
    }

    /// Execute a search query through the fused pipeline
    pub async fn search(&self, context: PipelineContext) -> Result<PipelineResult, PipelineError> {
        let start_time = Instant::now();
        
        // Check SLA before starting
        if context.is_deadline_exceeded() {
            return Err(PipelineError::DeadlineExceeded {
                elapsed_ms: context.elapsed().as_millis() as u64,
                deadline_ms: self.config.max_latency_ms,
            });
        }

        // Acquire scheduler slot
        let _permit = self.scheduler.acquire().await;
        
        // Execute through fused stages
        let result = self.executor.execute(context.clone()).await;
        
        // Update metrics
        let latency_ms = start_time.elapsed().as_millis() as u64;
        self.update_metrics(latency_ms, result.is_ok()).await;
        
        match result {
            Ok(data) => {
                debug!(
                    "Pipeline execution completed: request_id={}, latency={}ms, sla_compliance={}",
                    context.request_id,
                    latency_ms,
                    latency_ms <= self.config.max_latency_ms
                );
                
                Ok(PipelineResult {
                    request_id: context.request_id,
                    data,
                    success: true,
                    error_message: None,
                    metrics: self.get_current_metrics().await,
                })
            }
            Err(e) => {
                warn!(
                    "Pipeline execution failed: request_id={}, error={:?}",
                    context.request_id, e
                );
                
                Ok(PipelineResult {
                    request_id: context.request_id,
                    data: PipelineData::new(0),
                    success: false,
                    error_message: Some(e.to_string()),
                    metrics: self.get_current_metrics().await,
                })
            }
        }
    }

    async fn update_metrics(&self, latency_ms: u64, success: bool) {
        let mut metrics = self.metrics.write().await;
        
        metrics.total_requests += 1;
        
        if success {
            metrics.successful_requests += 1;
        } else {
            metrics.failed_requests += 1;
        }
        
        // Update latency statistics (simplified)
        let total_requests = metrics.total_requests as f64;
        metrics.avg_latency_ms = (metrics.avg_latency_ms * (total_requests - 1.0) + latency_ms as f64) / total_requests;
        
        // Update percentiles (simplified - would use proper quantile estimation in production)
        if latency_ms > metrics.p95_latency_ms {
            metrics.p95_latency_ms = latency_ms;
        }
        if latency_ms > metrics.p99_latency_ms {
            metrics.p99_latency_ms = latency_ms;
        }
        
        // Update SLA compliance
        let sla_compliant = latency_ms <= self.config.max_latency_ms;
        let compliant_requests = if sla_compliant { 1.0 } else { 0.0 };
        metrics.sla_compliance_rate = (metrics.sla_compliance_rate * (total_requests - 1.0) + compliant_requests) / total_requests;
    }

    async fn get_current_metrics(&self) -> PipelineMetrics {
        self.metrics.read().await.clone()
    }

    /// Get comprehensive pipeline statistics
    pub async fn get_metrics(&self) -> PipelineMetrics {
        self.get_current_metrics().await
    }

    /// Shutdown the pipeline gracefully
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down fused pipeline");
        self.executor.shutdown().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_context() {
        let context = PipelineContext::new("test-123".to_string(), "test query".to_string(), 150);
        
        assert_eq!(context.request_id, "test-123");
        assert_eq!(context.query, "test query");
        assert_eq!(context.max_results, 50);
        assert_eq!(context.timeout.as_millis(), 150);
        assert!(!context.is_deadline_exceeded());
    }

    #[test]
    fn test_pipeline_data() {
        let mut data = PipelineData::new(1024);
        assert_eq!(data.stage, PipelineStage::Input);
        assert_eq!(data.segments.len(), 0);
        assert_eq!(data.total_size(), 0);
        
        data.advance_stage(PipelineStage::QueryAnalysis);
        assert_eq!(data.stage, PipelineStage::QueryAnalysis);
        assert_eq!(data.metadata.stage_transitions, 1);
    }

    #[test]
    fn test_stage_timeouts() {
        let timeouts = StageTimeouts {
            query_analysis_percent: 0.1,
            lsp_routing_percent: 0.2,
            parallel_search_percent: 0.5,
            result_fusion_percent: 0.15,
            post_process_percent: 0.05,
        };
        
        let timeout = timeouts.calculate_timeout(1000, PipelineStage::ParallelSearch);
        assert_eq!(timeout.as_millis(), 500);
    }

    #[test]
    fn test_stage_progression() {
        let mut stage = PipelineStage::Input;
        
        stage = stage.next().unwrap();
        assert_eq!(stage, PipelineStage::QueryAnalysis);
        
        stage = stage.next().unwrap();
        assert_eq!(stage, PipelineStage::LspRouting);
        
        // Continue through all stages
        while let Some(next_stage) = stage.next() {
            stage = next_stage;
        }
        
        assert_eq!(stage, PipelineStage::Output);
        assert!(stage.next().is_none());
    }

    #[tokio::test]
    async fn test_pipeline_creation() {
        let config = PipelineConfig::default();
        let pipeline = FusedPipeline::new(config).await;
        assert!(pipeline.is_ok());
    }
}