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