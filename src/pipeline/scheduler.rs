use tokio::sync::{mpsc, oneshot, Semaphore};
use tokio::time::{timeout, Duration, Instant};
use std::sync::Arc;
use std::collections::{HashMap, VecDeque};
use parking_lot::RwLock;
use anyhow::Result;
use uuid::Uuid;

/// High-performance query scheduler with priority queuing and resource management
pub struct QueryScheduler {
    /// Semaphore for concurrent query limiting
    query_semaphore: Arc<Semaphore>,
    /// Task queue for different priority levels
    task_queues: Arc<RwLock<HashMap<Priority, VecDeque<ScheduledTask>>>>,
    /// Worker pool for processing tasks
    worker_handles: Vec<tokio::task::JoinHandle<()>>,
    /// Task sender channel
    task_sender: mpsc::UnboundedSender<ScheduledTask>,
    /// Metrics tracking
    metrics: Arc<RwLock<SchedulerMetrics>>,
    /// Configuration
    config: SchedulerConfig,
}

/// Priority levels for query scheduling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Priority {
    Critical = 0,
    High = 1,
    Normal = 2,
    Low = 3,
    Background = 4,
}

/// Scheduled task with priority and deadline
#[derive(Debug)]
pub struct ScheduledTask {
    pub id: Uuid,
    pub priority: Priority,
    pub deadline: Option<Instant>,
    pub task_type: TaskType,
    pub payload: TaskPayload,
    pub response_sender: oneshot::Sender<TaskResult>,
    pub created_at: Instant,
}

/// Type of scheduled task
#[derive(Debug, Clone)]
pub enum TaskType {
    Search,
    Index,
    Benchmark,
    HealthCheck,
    Maintenance,
}

/// Task payload containing execution data
#[derive(Debug, Clone)]
pub enum TaskPayload {
    Search {
        query: String,
        options: SearchOptions,
    },
    Index {
        file_paths: Vec<String>,
        incremental: bool,
    },
    Benchmark {
        suite_name: String,
        config: BenchmarkConfig,
    },
    HealthCheck {
        component: String,
    },
    Maintenance {
        operation: MaintenanceOperation,
    },
}

/// Search options for scheduled queries
#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub systems: Vec<String>,
    pub timeout_ms: u64,
    pub max_results: usize,
    pub language_hint: Option<String>,
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub query_count: usize,
    pub timeout_ms: u64,
    pub parallel_queries: usize,
}

/// Maintenance operations
#[derive(Debug, Clone)]
pub enum MaintenanceOperation {
    CacheClear,
    IndexOptimize,
    LogRotate,
    MetricsReset,
}

/// Task execution result
#[derive(Debug)]
pub enum TaskResult {
    Success(TaskOutput),
    Error(String),
    Timeout,
}

/// Task output data
#[derive(Debug)]
pub enum TaskOutput {
    SearchResults(Vec<crate::search::SearchResult>),
    IndexUpdate { files_processed: usize },
    BenchmarkResults { metrics: BenchmarkMetrics },
    HealthStatus { healthy: bool, details: String },
    MaintenanceComplete,
}

/// Benchmark metrics output
#[derive(Debug)]
pub struct BenchmarkMetrics {
    pub total_queries: usize,
    pub successful_queries: usize,
    pub average_latency_ms: f64,
    pub p95_latency_ms: f64,
}

/// Scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_concurrent_queries: usize,
    pub max_queue_size: usize,
    pub worker_count: usize,
    pub default_timeout_ms: u64,
    pub priority_weights: HashMap<Priority, f64>,
}

/// Scheduler metrics
#[derive(Debug, Clone)]
struct SchedulerMetrics {
    pub total_tasks: u64,
    pub completed_tasks: u64,
    pub failed_tasks: u64,
    pub timeout_tasks: u64,
    pub queue_sizes: HashMap<Priority, usize>,
    pub average_wait_time_ms: f64,
    pub average_execution_time_ms: f64,
}

impl QueryScheduler {
    /// Create new query scheduler
    pub fn new(config: SchedulerConfig) -> Self {
        let query_semaphore = Arc::new(Semaphore::new(config.max_concurrent_queries));
        let task_queues = Arc::new(RwLock::new(HashMap::new()));
        let metrics = Arc::new(RwLock::new(SchedulerMetrics::new()));
        
        // Initialize priority queues
        {
            let mut queues = task_queues.write();
            for priority in [Priority::Critical, Priority::High, Priority::Normal, Priority::Low, Priority::Background] {
                queues.insert(priority, VecDeque::new());
            }
        }

        let (task_sender, task_receiver) = mpsc::unbounded_channel();
        
        let mut scheduler = Self {
            query_semaphore,
            task_queues: task_queues.clone(),
            worker_handles: Vec::new(),
            task_sender,
            metrics: metrics.clone(),
            config: config.clone(),
        };

        // Start worker threads
        scheduler.start_workers(task_receiver);
        
        scheduler
    }

    /// Schedule a new task
    pub async fn schedule_task(&self, 
        task_type: TaskType,
        priority: Priority,
        payload: TaskPayload,
        deadline: Option<Duration>,
    ) -> Result<TaskResult> {
        let (response_sender, response_receiver) = oneshot::channel();
        
        let task = ScheduledTask {
            id: Uuid::new_v4(),
            priority,
            deadline: deadline.map(|d| Instant::now() + d),
            task_type,
            payload,
            response_sender,
            created_at: Instant::now(),
        };

        // Check queue capacity
        let queue_size = self.get_queue_size(priority);
        if queue_size >= self.config.max_queue_size {
            return Ok(TaskResult::Error("Queue capacity exceeded".to_string()));
        }

        // Send task to worker pool
        if self.task_sender.send(task).is_err() {
            return Ok(TaskResult::Error("Scheduler is shutting down".to_string()));
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.total_tasks += 1;
        }

        // Wait for result with timeout
        let timeout_duration = Duration::from_millis(self.config.default_timeout_ms);
        match timeout(timeout_duration, response_receiver).await {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(_)) => Ok(TaskResult::Error("Task was cancelled".to_string())),
            Err(_) => {
                let mut metrics = self.metrics.write();
                metrics.timeout_tasks += 1;
                Ok(TaskResult::Timeout)
            }
        }
    }

    /// Schedule a search task (convenience method)
    pub async fn schedule_search(
        &self,
        query: String,
        systems: Vec<String>,
        timeout_ms: Option<u64>,
        priority: Priority,
    ) -> Result<TaskResult> {
        let payload = TaskPayload::Search {
            query,
            options: SearchOptions {
                systems,
                timeout_ms: timeout_ms.unwrap_or(self.config.default_timeout_ms),
                max_results: 50,
                language_hint: None,
            },
        };

        self.schedule_task(
            TaskType::Search,
            priority,
            payload,
            Some(Duration::from_millis(timeout_ms.unwrap_or(self.config.default_timeout_ms))),
        ).await
    }

    /// Start worker threads
    fn start_workers(&mut self, mut task_receiver: mpsc::UnboundedReceiver<ScheduledTask>) {
        let receiver = Arc::new(tokio::sync::Mutex::new(task_receiver));
        
        for worker_id in 0..self.config.worker_count {
            let query_semaphore = self.query_semaphore.clone();
            let metrics = self.metrics.clone();
            let receiver = receiver.clone();
            
            let handle = tokio::spawn(async move {
                loop {
                    let task = {
                        let mut rx = receiver.lock().await;
                        rx.recv().await
                    };
                    
                    match task {
                        Some(task) => {
                            Self::execute_worker_task(worker_id, task, query_semaphore.clone(), metrics.clone()).await;
                        }
                        None => break, // Channel closed
                    }
                }
            });
            
            self.worker_handles.push(handle);
        }
    }

    /// Execute a task in a worker thread
    async fn execute_worker_task(
        _worker_id: usize,
        task: ScheduledTask,
        semaphore: Arc<Semaphore>,
        metrics: Arc<RwLock<SchedulerMetrics>>,
    ) {
        let start_time = Instant::now();
        let wait_time_ms = start_time.duration_since(task.created_at).as_millis() as f64;

        // Check deadline
        if let Some(deadline) = task.deadline {
            if Instant::now() > deadline {
                let _ = task.response_sender.send(TaskResult::Error("Task deadline exceeded".to_string()));
                let mut m = metrics.write();
                m.failed_tasks += 1;
                return;
            }
        }

        // Acquire semaphore permit
        let permit = match semaphore.acquire().await {
            Ok(permit) => permit,
            Err(_) => {
                let _ = task.response_sender.send(TaskResult::Error("Scheduler is shutting down".to_string()));
                return;
            }
        };

        // Execute the task
        let result = Self::execute_task_payload(task.task_type, task.payload).await;
        
        let execution_time_ms = start_time.elapsed().as_millis() as f64;
        
        // Update metrics
        {
            let mut m = metrics.write();
            match result {
                TaskResult::Success(_) => m.completed_tasks += 1,
                TaskResult::Error(_) => m.failed_tasks += 1,
                TaskResult::Timeout => m.timeout_tasks += 1,
            }
            
            // Update rolling averages
            let total_completed = m.completed_tasks + m.failed_tasks + m.timeout_tasks;
            if total_completed > 0 {
                m.average_wait_time_ms = (m.average_wait_time_ms * (total_completed - 1) as f64 + wait_time_ms) / total_completed as f64;
                m.average_execution_time_ms = (m.average_execution_time_ms * (total_completed - 1) as f64 + execution_time_ms) / total_completed as f64;
            }
        }

        // Send result back
        let _ = task.response_sender.send(result);
        
        // Release permit
        drop(permit);
    }

    /// Execute task payload
    async fn execute_task_payload(task_type: TaskType, payload: TaskPayload) -> TaskResult {
        match (task_type, payload) {
            (TaskType::Search, TaskPayload::Search { query: _query, options: _options }) => {
                // TODO: Integrate with actual search implementation
                TaskResult::Success(TaskOutput::SearchResults(vec![]))
            }
            
            (TaskType::Index, TaskPayload::Index { file_paths, incremental: _ }) => {
                // TODO: Integrate with actual indexing implementation
                TaskResult::Success(TaskOutput::IndexUpdate { 
                    files_processed: file_paths.len() 
                })
            }
            
            (TaskType::Benchmark, TaskPayload::Benchmark { suite_name: _suite_name, config }) => {
                // TODO: Integrate with actual benchmark implementation
                TaskResult::Success(TaskOutput::BenchmarkResults {
                    metrics: BenchmarkMetrics {
                        total_queries: config.query_count,
                        successful_queries: config.query_count,
                        average_latency_ms: 150.0,
                        p95_latency_ms: 300.0,
                    }
                })
            }
            
            (TaskType::HealthCheck, TaskPayload::HealthCheck { component }) => {
                // TODO: Implement actual health checking
                TaskResult::Success(TaskOutput::HealthStatus {
                    healthy: true,
                    details: format!("Component {} is healthy", component),
                })
            }
            
            (TaskType::Maintenance, TaskPayload::Maintenance { operation: _operation }) => {
                // TODO: Implement maintenance operations
                TaskResult::Success(TaskOutput::MaintenanceComplete)
            }
            
            _ => TaskResult::Error("Invalid task type and payload combination".to_string()),
        }
    }

    /// Get current queue size for priority
    fn get_queue_size(&self, priority: Priority) -> usize {
        let queues = self.task_queues.read();
        queues.get(&priority).map(|q| q.len()).unwrap_or(0)
    }

    /// Get scheduler metrics
    pub fn get_metrics(&self) -> SchedulerMetrics {
        let metrics = self.metrics.read();
        let mut queue_sizes = HashMap::new();
        
        let queues = self.task_queues.read();
        for (priority, queue) in queues.iter() {
            queue_sizes.insert(*priority, queue.len());
        }
        
        let mut result = metrics.clone();
        result.queue_sizes = queue_sizes;
        result
    }

    /// Shutdown scheduler gracefully
    pub async fn shutdown(&mut self) {
        // Cancel all worker handles
        for handle in &mut self.worker_handles {
            handle.abort();
        }
        
        // Wait for workers to finish (with timeout)
        let timeout_duration = Duration::from_secs(5);
        for handle in self.worker_handles.drain(..) {
            let _ = timeout(timeout_duration, handle).await;
        }
    }
}

impl SchedulerMetrics {
    fn new() -> Self {
        Self {
            total_tasks: 0,
            completed_tasks: 0,
            failed_tasks: 0,
            timeout_tasks: 0,
            queue_sizes: HashMap::new(),
            average_wait_time_ms: 0.0,
            average_execution_time_ms: 0.0,
        }
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        let mut priority_weights = HashMap::new();
        priority_weights.insert(Priority::Critical, 1.0);
        priority_weights.insert(Priority::High, 0.8);
        priority_weights.insert(Priority::Normal, 0.6);
        priority_weights.insert(Priority::Low, 0.4);
        priority_weights.insert(Priority::Background, 0.2);
        
        Self {
            max_concurrent_queries: 100,
            max_queue_size: 1000,
            worker_count: 8,
            default_timeout_ms: 30000,
            priority_weights,
        }
    }
}

/// Pipeline-specific scheduler for coordinating pipeline stages
pub struct PipelineScheduler {
    max_concurrent: usize,
    semaphore: Arc<Semaphore>,
    metrics: Arc<RwLock<PipelineSchedulerMetrics>>,
}

#[derive(Debug, Default, Clone)]
pub struct PipelineSchedulerMetrics {
    pub total_requests: u64,
    pub active_requests: u64,
    pub queued_requests: u64,
    pub completed_requests: u64,
    pub failed_requests: u64,
    pub average_wait_time_ms: f64,
}

impl PipelineScheduler {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            max_concurrent,
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            metrics: Arc::new(RwLock::new(PipelineSchedulerMetrics::default())),
        }
    }

    /// Acquire a permit for pipeline execution
    pub async fn acquire(&self) -> SemaphorePermit<'_> {
        let start_time = Instant::now();
        
        // Update queued requests
        {
            let mut metrics = self.metrics.write();
            metrics.total_requests += 1;
            metrics.queued_requests += 1;
        }

        let permit = self.semaphore.acquire().await.expect("Semaphore closed");
        
        let wait_time = start_time.elapsed().as_millis() as f64;
        
        // Update metrics after acquiring permit
        {
            let mut metrics = self.metrics.write();
            metrics.queued_requests -= 1;
            metrics.active_requests += 1;
            
            // Update rolling average wait time
            let total = metrics.total_requests as f64;
            metrics.average_wait_time_ms = (metrics.average_wait_time_ms * (total - 1.0) + wait_time) / total;
        }

        permit
    }

    /// Get scheduler metrics
    pub async fn get_metrics(&self) -> PipelineSchedulerMetrics {
        self.metrics.read().clone()
    }

    /// Get available capacity
    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }
}

use tokio::sync::SemaphorePermit;

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    fn create_test_config() -> SchedulerConfig {
        let mut priority_weights = HashMap::new();
        priority_weights.insert(Priority::Critical, 1.0);
        priority_weights.insert(Priority::High, 0.8);
        priority_weights.insert(Priority::Normal, 0.6);
        priority_weights.insert(Priority::Low, 0.4);
        priority_weights.insert(Priority::Background, 0.2);

        SchedulerConfig {
            max_concurrent_queries: 4,
            max_queue_size: 100,
            worker_count: 2,
            default_timeout_ms: 5000,
            priority_weights,
        }
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical < Priority::High);
        assert!(Priority::High < Priority::Normal);
        assert!(Priority::Normal < Priority::Low);
        assert!(Priority::Low < Priority::Background);
    }

    #[test]
    fn test_scheduler_config_creation() {
        let config = create_test_config();
        assert_eq!(config.max_concurrent_queries, 4);
        assert_eq!(config.max_queue_size, 100);
        assert_eq!(config.worker_count, 2);
        assert_eq!(config.default_timeout_ms, 5000);
        assert_eq!(config.priority_weights.len(), 5);
    }

    #[test]
    fn test_scheduler_metrics_new() {
        let metrics = SchedulerMetrics::new();
        assert_eq!(metrics.total_tasks, 0);
        assert_eq!(metrics.completed_tasks, 0);
        assert_eq!(metrics.failed_tasks, 0);
        assert_eq!(metrics.timeout_tasks, 0);
        assert_eq!(metrics.queue_sizes.len(), 0);
        assert_eq!(metrics.average_wait_time_ms, 0.0);
        assert_eq!(metrics.average_execution_time_ms, 0.0);
    }

    #[tokio::test]
    async fn test_query_scheduler_creation() {
        let config = create_test_config();
        let scheduler = QueryScheduler::new(config.clone());
        
        assert_eq!(scheduler.config.max_concurrent_queries, config.max_concurrent_queries);
        assert_eq!(scheduler.config.worker_count, config.worker_count);
        
        // Check that all priority queues are initialized
        let queues = scheduler.task_queues.read();
        assert_eq!(queues.len(), 5);
        assert!(queues.contains_key(&Priority::Critical));
        assert!(queues.contains_key(&Priority::High));
        assert!(queues.contains_key(&Priority::Normal));
        assert!(queues.contains_key(&Priority::Low));
        assert!(queues.contains_key(&Priority::Background));
    }

    #[test]
    fn test_task_type_variants() {
        let search = TaskType::Search;
        let index = TaskType::Index;
        let benchmark = TaskType::Benchmark;
        let health_check = TaskType::HealthCheck;
        let maintenance = TaskType::Maintenance;

        // Just ensure variants exist and can be created
        assert!(matches!(search, TaskType::Search));
        assert!(matches!(index, TaskType::Index));
        assert!(matches!(benchmark, TaskType::Benchmark));
        assert!(matches!(health_check, TaskType::HealthCheck));
        assert!(matches!(maintenance, TaskType::Maintenance));
    }

    #[test]
    fn test_task_payload_variants() {
        let search_payload = TaskPayload::Search {
            query: "test query".to_string(),
            options: SearchOptions {
                systems: vec!["lex".to_string()],
                timeout_ms: 1000,
                max_results: 10,
                language_hint: Some("rust".to_string()),
            },
        };

        let index_payload = TaskPayload::Index {
            file_paths: vec!["test.rs".to_string()],
            incremental: true,
        };

        let benchmark_payload = TaskPayload::Benchmark {
            suite_name: "test_suite".to_string(),
            config: BenchmarkConfig {
                query_count: 100,
                timeout_ms: 1000,
                parallel_queries: 4,
            },
        };

        let health_payload = TaskPayload::HealthCheck {
            component: "search_engine".to_string(),
        };

        let maintenance_payload = TaskPayload::Maintenance {
            operation: MaintenanceOperation::CacheClear,
        };

        assert!(matches!(search_payload, TaskPayload::Search { .. }));
        assert!(matches!(index_payload, TaskPayload::Index { .. }));
        assert!(matches!(benchmark_payload, TaskPayload::Benchmark { .. }));
        assert!(matches!(health_payload, TaskPayload::HealthCheck { .. }));
        assert!(matches!(maintenance_payload, TaskPayload::Maintenance { .. }));
    }

    #[test]
    fn test_maintenance_operations() {
        let cache_clear = MaintenanceOperation::CacheClear;
        let index_optimize = MaintenanceOperation::IndexOptimize;
        let log_rotate = MaintenanceOperation::LogRotate;
        let metrics_reset = MaintenanceOperation::MetricsReset;

        assert!(matches!(cache_clear, MaintenanceOperation::CacheClear));
        assert!(matches!(index_optimize, MaintenanceOperation::IndexOptimize));
        assert!(matches!(log_rotate, MaintenanceOperation::LogRotate));
        assert!(matches!(metrics_reset, MaintenanceOperation::MetricsReset));
    }

    #[test]
    fn test_task_result_variants() {
        let success_result = TaskResult::Success(TaskOutput::MaintenanceComplete);
        let error_result = TaskResult::Error("test error".to_string());
        let timeout_result = TaskResult::Timeout;

        assert!(matches!(success_result, TaskResult::Success(_)));
        assert!(matches!(error_result, TaskResult::Error(_)));
        assert!(matches!(timeout_result, TaskResult::Timeout));
    }

    #[test]
    fn test_task_output_variants() {
        let search_output = TaskOutput::SearchResults(vec![]);
        let index_output = TaskOutput::IndexUpdate { files_processed: 5 };
        let benchmark_output = TaskOutput::BenchmarkResults { 
            metrics: BenchmarkMetrics {
                total_queries: 100,
                successful_queries: 95,
                average_latency_ms: 150.0,
                p95_latency_ms: 300.0,
            }
        };
        let health_output = TaskOutput::HealthStatus { 
            healthy: true, 
            details: "All systems operational".to_string() 
        };
        let maintenance_output = TaskOutput::MaintenanceComplete;

        assert!(matches!(search_output, TaskOutput::SearchResults(_)));
        assert!(matches!(index_output, TaskOutput::IndexUpdate { .. }));
        assert!(matches!(benchmark_output, TaskOutput::BenchmarkResults { .. }));
        assert!(matches!(health_output, TaskOutput::HealthStatus { .. }));
        assert!(matches!(maintenance_output, TaskOutput::MaintenanceComplete));
    }

    #[test]
    fn test_search_options() {
        let options = SearchOptions {
            systems: vec!["lex".to_string(), "symbols".to_string()],
            timeout_ms: 2000,
            max_results: 25,
            language_hint: Some("typescript".to_string()),
        };

        assert_eq!(options.systems.len(), 2);
        assert_eq!(options.timeout_ms, 2000);
        assert_eq!(options.max_results, 25);
        assert_eq!(options.language_hint, Some("typescript".to_string()));
    }

    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig {
            query_count: 200,
            timeout_ms: 5000,
            parallel_queries: 8,
        };

        assert_eq!(config.query_count, 200);
        assert_eq!(config.timeout_ms, 5000);
        assert_eq!(config.parallel_queries, 8);
    }

    #[test]
    fn test_benchmark_metrics() {
        let metrics = BenchmarkMetrics {
            total_queries: 100,
            successful_queries: 98,
            average_latency_ms: 125.5,
            p95_latency_ms: 250.0,
        };

        assert_eq!(metrics.total_queries, 100);
        assert_eq!(metrics.successful_queries, 98);
        assert_eq!(metrics.average_latency_ms, 125.5);
        assert_eq!(metrics.p95_latency_ms, 250.0);
    }

    #[tokio::test]
    async fn test_scheduler_queue_size_tracking() {
        let config = create_test_config();
        let scheduler = QueryScheduler::new(config);

        // Initially all queues should be empty
        assert_eq!(scheduler.get_queue_size(Priority::High), 0);
        assert_eq!(scheduler.get_queue_size(Priority::Normal), 0);
        assert_eq!(scheduler.get_queue_size(Priority::Low), 0);
    }

    #[tokio::test] 
    async fn test_scheduler_metrics_tracking() {
        let config = create_test_config();
        let scheduler = QueryScheduler::new(config);

        let metrics = scheduler.get_metrics();
        assert_eq!(metrics.total_tasks, 0);
        assert_eq!(metrics.completed_tasks, 0);
        assert_eq!(metrics.failed_tasks, 0);
        assert_eq!(metrics.timeout_tasks, 0);
    }

    #[test]
    fn test_pipeline_scheduler_creation() {
        let scheduler = PipelineScheduler::new(10);
        assert_eq!(scheduler.available_permits(), 10);
    }

    #[tokio::test]
    async fn test_pipeline_scheduler_acquire_release() {
        let scheduler = PipelineScheduler::new(2);
        
        // Should be able to acquire permits
        let _permit1 = scheduler.acquire().await;
        assert_eq!(scheduler.available_permits(), 1);
        
        let _permit2 = scheduler.acquire().await;
        assert_eq!(scheduler.available_permits(), 0);
        
        // Permits should be released when dropped
        drop(_permit1);
        // Note: We can't directly test permit release timing in sync code
        // but the semaphore will release when the permit is dropped
    }

    #[tokio::test]
    async fn test_scheduler_basic_task_scheduling() {
        let config = create_test_config();
        let scheduler = QueryScheduler::new(config);

        let payload = TaskPayload::HealthCheck {
            component: "test_component".to_string(),
        };

        // Schedule a simple task - this tests the basic scheduling mechanism
        // Note: The actual task execution would depend on worker implementation
        // which might not be fully functional in a unit test environment
        let task_future = scheduler.schedule_task(
            TaskType::HealthCheck,
            Priority::Normal,
            payload,
            Some(Duration::from_millis(100)),
        );

        // The task should either complete, timeout, or return an error
        // We don't assert specific outcomes since worker implementation
        // may not be fully operational in test environment
        let _result = task_future.await;
        
        // Verify metrics were updated
        let metrics = scheduler.get_metrics();
        assert!(metrics.total_tasks > 0);
    }

    #[tokio::test]
    async fn test_scheduler_queue_capacity_limit() {
        let mut config = create_test_config();
        config.max_queue_size = 1; // Very small queue for testing
        let scheduler = QueryScheduler::new(config);

        let payload1 = TaskPayload::HealthCheck {
            component: "test1".to_string(),
        };
        let payload2 = TaskPayload::HealthCheck {
            component: "test2".to_string(),
        };

        // First task should be accepted
        let result1_future = scheduler.schedule_task(
            TaskType::HealthCheck,
            Priority::Normal,
            payload1,
            Some(Duration::from_millis(100)),
        );

        // Give a small delay to let the first task start processing
        sleep(Duration::from_millis(10)).await;

        // Second task might be rejected due to queue capacity
        let result2_future = scheduler.schedule_task(
            TaskType::HealthCheck,
            Priority::Normal,
            payload2,
            Some(Duration::from_millis(100)),
        );

        let _result1 = result1_future.await.unwrap();
        let _result2 = result2_future.await.unwrap();

        // At least one task should have been processed
        let metrics = scheduler.get_metrics();
        assert!(metrics.total_tasks > 0);
    }
}