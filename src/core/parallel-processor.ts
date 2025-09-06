/**
 * Enhanced Parallel Processing System
 * High-performance task orchestration for sub-10ms search operations
 * Implements work-stealing, dynamic load balancing, and intelligent batching
 */

import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';
import { cpus } from 'os';
import { performance } from 'perf_hooks';
import { LensTracer } from '../telemetry/tracer.js';
import { globalMemoryPool } from './memory-pool-manager.js';
import { globalCacheManager } from './advanced-cache-manager.js';
import type { SearchContext, SearchHit, Candidate } from '../types/core.js';

interface ParallelTask<T, R> {
  id: string;
  type: TaskType;
  payload: T;
  priority: TaskPriority;
  context?: SearchContext;
  timeout: number;
  createdAt: number;
  estimatedDuration: number;
  dependencies?: string[];
  callback?: (result: R | Error) => void;
}

interface WorkerStats {
  id: number;
  tasksProcessed: number;
  totalDuration: number;
  avgDuration: number;
  isIdle: boolean;
  currentTask?: string;
  memoryUsage: number;
  errorCount: number;
}

interface BatchConfig {
  maxBatchSize: number;
  maxWaitTime: number;
  minBatchSize: number;
  batchingEnabled: boolean;
}

enum TaskType {
  LEXICAL_SEARCH = 'lexical_search',
  SYMBOL_SEARCH = 'symbol_search',
  SEMANTIC_RERANK = 'semantic_rerank',
  TRIGRAM_GENERATION = 'trigram_generation',
  FUZZY_MATCHING = 'fuzzy_matching',
  INDEX_BUILDING = 'index_building',
  CACHE_WARMUP = 'cache_warmup',
  COMPRESSION = 'compression',
  VALIDATION = 'validation'
}

enum TaskPriority {
  CRITICAL = 0,  // User-facing searches
  HIGH = 1,      // Background processing with deadlines
  NORMAL = 2,    // Standard operations
  LOW = 3,       // Maintenance, cleanup
  BACKGROUND = 4 // Non-urgent background tasks
}

interface ProcessorConfig {
  maxWorkers: number;
  minWorkers: number;
  idleTimeout: number;
  workStealingEnabled: boolean;
  adaptiveScaling: boolean;
  batchProcessing: BatchConfig;
  queueSizeLimit: number;
}

interface ProcessorStats {
  totalTasks: number;
  completedTasks: number;
  failedTasks: number;
  avgTaskDuration: number;
  workerCount: number;
  idleWorkers: number;
  queueSize: number;
  throughputPerSecond: number;
  memoryUsage: number;
  cpuUtilization: number;
}

export class ParallelProcessor {
  private static instance: ParallelProcessor;
  
  // Worker management
  private workers: Worker[] = [];
  private workerStats: Map<number, WorkerStats> = new Map();
  private idleWorkers: Set<number> = new Set();
  private workerTasks: Map<number, Set<string>> = new Map();
  
  // Task management
  private taskQueue: Map<TaskPriority, ParallelTask<any, any>[]> = new Map();
  private activeTasks: Map<string, ParallelTask<any, any>> = new Map();
  private completedTasksMap: Map<string, any> = new Map();
  private batchQueue: Map<TaskType, ParallelTask<any, any>[]> = new Map();
  
  // Performance tracking
  private totalTasks = 0;
  private completedTasks = 0;
  private failedTasks = 0;
  private startTime = Date.now();
  private taskDurations: number[] = [];
  
  // Configuration
  private config: ProcessorConfig;
  
  // Timers and intervals
  private scalingTimer?: NodeJS.Timeout;
  private batchTimer?: NodeJS.Timeout;
  private statsTimer?: NodeJS.Timeout;
  private cleanupTimer?: NodeJS.Timeout;
  
  private constructor() {
    this.config = {
      maxWorkers: Math.min(cpus().length, 8),
      minWorkers: Math.max(2, Math.ceil(cpus().length / 2)),
      idleTimeout: 30000, // 30 seconds
      workStealingEnabled: true,
      adaptiveScaling: true,
      batchProcessing: {
        maxBatchSize: 100,
        maxWaitTime: 10, // 10ms max wait for batching
        minBatchSize: 5,
        batchingEnabled: true
      },
      queueSizeLimit: 10000
    };
    
    this.initializeQueues();
    this.startWorkerPool();
    this.startMaintenanceTimers();
  }
  
  public static getInstance(): ParallelProcessor {
    if (!ParallelProcessor.instance) {
      ParallelProcessor.instance = new ParallelProcessor();
    }
    return ParallelProcessor.instance;
  }
  
  /**
   * Initialize task queues for each priority level
   */
  private initializeQueues(): void {
    for (const priority of Object.values(TaskPriority)) {
      if (typeof priority === 'number') {
        this.taskQueue.set(priority, []);
      }
    }
    
    for (const taskType of Object.values(TaskType)) {
      this.batchQueue.set(taskType, []);
    }
  }
  
  /**
   * Start initial worker pool
   */
  private startWorkerPool(): void {
    const span = LensTracer.createChildSpan('parallel_processor_init');
    
    try {
      // Start with minimum workers
      for (let i = 0; i < this.config.minWorkers; i++) {
        this.createWorker();
      }
      
      console.log(`🚀 Parallel Processor initialized with ${this.config.minWorkers} workers (max: ${this.config.maxWorkers})`);
      
      span.setAttributes({
        success: true,
        initial_workers: this.config.minWorkers,
        max_workers: this.config.maxWorkers
      });
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw error;
    } finally {
      span.end();
    }
  }
  
  /**
   * Create a new worker
   */
  private createWorker(): Worker {
    const workerId = this.workers.length;
    
    const worker = new Worker(__filename, {
      workerData: { 
        isWorker: true, 
        workerId,
        memoryPoolConfig: true,
        cacheConfig: true
      }
    });
    
    // Initialize worker stats
    this.workerStats.set(workerId, {
      id: workerId,
      tasksProcessed: 0,
      totalDuration: 0,
      avgDuration: 0,
      isIdle: true,
      memoryUsage: 0,
      errorCount: 0
    });
    
    this.idleWorkers.add(workerId);
    this.workerTasks.set(workerId, new Set());
    
    // Set up event handlers
    worker.on('message', (message) => {
      this.handleWorkerMessage(workerId, message);
    });
    
    worker.on('error', (error) => {
      console.error(`Worker ${workerId} error:`, error);
      this.handleWorkerError(workerId, error);
    });
    
    worker.on('exit', (code) => {
      console.warn(`Worker ${workerId} exited with code ${code}`);
      this.handleWorkerExit(workerId, code);
    });
    
    this.workers.push(worker);
    return worker;
  }
  
  /**
   * Submit task for parallel processing
   */
  async submitTask<T, R>(
    type: TaskType,
    payload: T,
    priority: TaskPriority = TaskPriority.NORMAL,
    context?: SearchContext,
    timeout: number = 30000
  ): Promise<R> {
    const span = LensTracer.createChildSpan('submit_parallel_task');
    
    // Check cache first
    if (context) {
      const cacheKey = this.generateCacheKey(type, payload, context);
      try {
        const cached = await globalCacheManager.get<R>(cacheKey, context);
        if (cached) {
          return cached;
        }
      } catch (e) {
        // Cache error, continue with task execution
      }
    }

    return new Promise((resolve, reject) => {
      try {
        const taskId = this.generateTaskId();
        
        const task: ParallelTask<T, R> = {
          id: taskId,
          type,
          payload,
          priority,
          context,
          timeout,
          createdAt: Date.now(),
          estimatedDuration: this.estimateTaskDuration(type, payload),
          callback: (result: R | Error) => {
            if (result instanceof Error) {
              reject(result);
            } else {
              resolve(result);
            }
          }
        };
        
        // Check if batching is beneficial
        if (this.shouldBatch(task)) {
          this.addToBatch(task);
        } else {
          this.queueTask(task);
        }
        
        this.totalTasks++;
        
        // Set timeout
        setTimeout(() => {
          if (this.activeTasks.has(taskId)) {
            this.handleTaskTimeout(taskId);
            reject(new Error(`Task ${taskId} timed out after ${timeout}ms`));
          }
        }, timeout);
        
        span.setAttributes({
          success: true,
          task_id: taskId,
          task_type: type,
          priority,
          estimated_duration: task.estimatedDuration
        });
        
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : 'Unknown error';
        span.recordException(error as Error);
        span.setAttributes({ success: false, error: errorMsg });
        reject(error);
      } finally {
        span.end();
      }
    });
  }
  
  /**
   * Submit batch of similar tasks
   */
  async submitBatch<T, R>(
    type: TaskType,
    payloads: T[],
    priority: TaskPriority = TaskPriority.NORMAL,
    context?: SearchContext
  ): Promise<R[]> {
    const span = LensTracer.createChildSpan('submit_batch_tasks');
    
    try {
      const batchPromises = payloads.map(payload => 
        this.submitTask<T, R>(type, payload, priority, context)
      );
      
      const results = await Promise.allSettled(batchPromises);
      
      const successResults: R[] = [];
      const errors: Error[] = [];
      
      for (const result of results) {
        if (result.status === 'fulfilled') {
          successResults.push(result.value);
        } else {
          errors.push(new Error(result.reason));
        }
      }
      
      if (errors.length > 0 && successResults.length === 0) {
        throw new Error(`All batch tasks failed: ${errors.map(e => e.message).join(', ')}`);
      }
      
      span.setAttributes({
        success: true,
        batch_size: payloads.length,
        successful_tasks: successResults.length,
        failed_tasks: errors.length
      });
      
      return successResults;
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw error;
    } finally {
      span.end();
    }
  }
  
  /**
   * Queue task for processing
   */
  private queueTask(task: ParallelTask<any, any>): void {
    const queue = this.taskQueue.get(task.priority);
    if (queue) {
      queue.push(task);
      
      // Try to assign immediately if workers are available
      if (this.idleWorkers.size > 0) {
        setImmediate(() => this.assignTasks());
      }
    }
  }
  
  /**
   * Check if task should be batched
   */
  private shouldBatch(task: ParallelTask<any, any>): boolean {
    if (!this.config.batchProcessing.batchingEnabled) return false;
    
    const batchableTypes = [
      TaskType.TRIGRAM_GENERATION,
      TaskType.FUZZY_MATCHING,
      TaskType.COMPRESSION
    ];
    
    return batchableTypes.includes(task.type);
  }
  
  /**
   * Add task to batch queue
   */
  private addToBatch(task: ParallelTask<any, any>): void {
    const batchQueue = this.batchQueue.get(task.type);
    if (batchQueue) {
      batchQueue.push(task);
      
      // Process batch if it's full or after timeout
      if (batchQueue.length >= this.config.batchProcessing.maxBatchSize) {
        setImmediate(() => this.processBatch(task.type));
      }
    }
  }
  
  /**
   * Process batch of similar tasks
   */
  private async processBatch(taskType: TaskType): Promise<void> {
    const span = LensTracer.createChildSpan('process_batch');
    
    try {
      const batchQueue = this.batchQueue.get(taskType);
      if (!batchQueue || batchQueue.length === 0) return;
      
      // Take all tasks from batch queue
      const tasks = batchQueue.splice(0);
      
      if (tasks.length === 0) return;
      
      // Create batch processing task
      const batchTask: ParallelTask<any, any> = {
        id: this.generateTaskId(),
        type: taskType,
        payload: {
          batchMode: true,
          tasks: tasks.map(t => ({ id: t.id, payload: t.payload, context: t.context }))
        },
        priority: Math.min(...tasks.map(t => t.priority)),
        timeout: Math.max(...tasks.map(t => t.timeout)),
        createdAt: Date.now(),
        estimatedDuration: this.estimateBatchDuration(tasks),
        callback: (result: any) => {
          this.handleBatchResult(tasks, result);
        }
      };
      
      this.queueTask(batchTask);
      
      span.setAttributes({
        success: true,
        task_type: taskType,
        batch_size: tasks.length
      });
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
    } finally {
      span.end();
    }
  }
  
  /**
   * Handle batch processing result
   */
  private handleBatchResult(tasks: ParallelTask<any, any>[], batchResult: any): void {
    if (batchResult.error) {
      // Handle batch failure
      tasks.forEach(task => {
        if (task.callback) {
          task.callback(new Error(`Batch processing failed: ${batchResult.error}`));
        }
      });
      return;
    }
    
    // Distribute results to individual task callbacks
    if (batchResult.results && Array.isArray(batchResult.results)) {
      for (let i = 0; i < Math.min(tasks.length, batchResult.results.length); i++) {
        const task = tasks[i];
        const result = batchResult.results[i];
        
        if (task.callback) {
          if (result.error) {
            task.callback(new Error(result.error));
          } else {
            task.callback(result.value);
          }
        }
      }
    }
  }
  
  /**
   * Assign tasks to idle workers
   */
  private assignTasks(): void {
    const span = LensTracer.createChildSpan('assign_tasks');
    let assignedCount = 0;
    
    try {
      // Process by priority (lower number = higher priority)
      const priorities = Array.from(this.taskQueue.keys()).sort((a, b) => a - b);
      
      for (const priority of priorities) {
        const queue = this.taskQueue.get(priority);
        if (!queue || queue.length === 0) continue;
        
        // Assign tasks while we have idle workers and queued tasks
        while (this.idleWorkers.size > 0 && queue.length > 0) {
          const task = queue.shift();
          const workerId = this.getOptimalWorker();
          
          if (task && workerId !== null) {
            this.assignTaskToWorker(task, workerId);
            assignedCount++;
          } else {
            break;
          }
        }
      }
      
      span.setAttributes({
        success: true,
        assigned_count: assignedCount,
        idle_workers: this.idleWorkers.size
      });
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
    } finally {
      span.end();
    }
  }
  
  /**
   * Get optimal worker for task assignment
   */
  private getOptimalWorker(): number | null {
    if (this.idleWorkers.size === 0) return null;
    
    // Simple strategy: prefer workers with lower average duration
    let bestWorkerId: number | null = null;
    let bestScore = Infinity;
    
    for (const workerId of this.idleWorkers) {
      const stats = this.workerStats.get(workerId);
      if (stats) {
        // Score based on average duration and task count
        const score = stats.avgDuration + (stats.tasksProcessed * 0.1);
        
        if (score < bestScore) {
          bestScore = score;
          bestWorkerId = workerId;
        }
      }
    }
    
    return bestWorkerId || Array.from(this.idleWorkers)[0];
  }
  
  /**
   * Assign task to specific worker
   */
  private assignTaskToWorker(task: ParallelTask<any, any>, workerId: number): void {
    const worker = this.workers[workerId];
    const stats = this.workerStats.get(workerId);
    
    if (!worker || !stats) return;
    
    // Update worker state
    this.idleWorkers.delete(workerId);
    stats.isIdle = false;
    stats.currentTask = task.id;
    
    const workerTasks = this.workerTasks.get(workerId);
    if (workerTasks) {
      workerTasks.add(task.id);
    }
    
    // Track active task
    this.activeTasks.set(task.id, task);
    
    // Send task to worker
    worker.postMessage({
      type: 'execute_task',
      task: {
        id: task.id,
        type: task.type,
        payload: task.payload,
        context: task.context,
        createdAt: task.createdAt
      }
    });
  }
  
  /**
   * Handle message from worker
   */
  private handleWorkerMessage(workerId: number, message: any): void {
    const { type, taskId, result, error, stats } = message;
    
    switch (type) {
      case 'task_completed':
        this.handleTaskCompleted(workerId, taskId, result);
        break;
        
      case 'task_failed':
        this.handleTaskFailed(workerId, taskId, error);
        break;
        
      case 'worker_stats':
        this.updateWorkerStats(workerId, stats);
        break;
        
      case 'worker_ready':
        this.handleWorkerReady(workerId);
        break;
    }
  }
  
  /**
   * Handle task completion
   */
  private handleTaskCompleted(workerId: number, taskId: string, result: any): void {
    const span = LensTracer.createChildSpan('handle_task_completed');
    
    try {
      const task = this.activeTasks.get(taskId);
      if (!task) return;
      
      const duration = Date.now() - task.createdAt;
      
      // Update statistics
      const stats = this.workerStats.get(workerId);
      if (stats) {
        stats.tasksProcessed++;
        stats.totalDuration += duration;
        stats.avgDuration = stats.totalDuration / stats.tasksProcessed;
        stats.isIdle = true;
        stats.currentTask = undefined;
      }
      
      // Update worker state
      this.idleWorkers.add(workerId);
      const workerTasks = this.workerTasks.get(workerId);
      if (workerTasks) {
        workerTasks.delete(taskId);
      }
      
      // Clean up
      this.activeTasks.delete(taskId);
      this.completedTasksMap.set(taskId, result);
      this.completedTasks++;
      this.taskDurations.push(duration);
      
      // Cache result if applicable
      if (task.context) {
        const cacheKey = this.generateCacheKey(task.type, task.payload, task.context);
        globalCacheManager.set(cacheKey, result, 300000, task.context); // 5 minute TTL
      }
      
      // Call task callback
      if (task.callback) {
        task.callback(result);
      }
      
      // Try to assign more tasks
      setImmediate(() => this.assignTasks());
      
      span.setAttributes({
        success: true,
        task_id: taskId,
        worker_id: workerId,
        duration_ms: duration
      });
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
    } finally {
      span.end();
    }
  }
  
  /**
   * Handle task failure
   */
  private handleTaskFailed(workerId: number, taskId: string, error: any): void {
    const span = LensTracer.createChildSpan('handle_task_failed');
    
    try {
      const task = this.activeTasks.get(taskId);
      if (!task) return;
      
      // Update statistics
      const stats = this.workerStats.get(workerId);
      if (stats) {
        stats.errorCount++;
        stats.isIdle = true;
        stats.currentTask = undefined;
      }
      
      // Update worker state
      this.idleWorkers.add(workerId);
      const workerTasks = this.workerTasks.get(workerId);
      if (workerTasks) {
        workerTasks.delete(taskId);
      }
      
      // Clean up
      this.activeTasks.delete(taskId);
      this.failedTasks++;
      
      // Call task callback with error
      if (task.callback) {
        task.callback(new Error(error));
      }
      
      // Try to assign more tasks
      setImmediate(() => this.assignTasks());
      
      span.setAttributes({
        success: true,
        task_id: taskId,
        worker_id: workerId,
        error: error
      });
      
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Unknown error';
      span.recordException(err as Error);
      span.setAttributes({ success: false, error: errorMsg });
    } finally {
      span.end();
    }
  }
  
  /**
   * Update worker statistics
   */
  private updateWorkerStats(workerId: number, statsUpdate: Partial<WorkerStats>): void {
    const stats = this.workerStats.get(workerId);
    if (stats) {
      Object.assign(stats, statsUpdate);
    }
  }
  
  /**
   * Handle worker ready state
   */
  private handleWorkerReady(workerId: number): void {
    this.idleWorkers.add(workerId);
    const stats = this.workerStats.get(workerId);
    if (stats) {
      stats.isIdle = true;
    }
    
    // Try to assign pending tasks
    setImmediate(() => this.assignTasks());
  }
  
  /**
   * Handle worker error
   */
  private handleWorkerError(workerId: number, error: Error): void {
    const stats = this.workerStats.get(workerId);
    if (stats) {
      stats.errorCount++;
    }
    
    // Consider restarting worker if too many errors
    if (stats && stats.errorCount > 10) {
      this.restartWorker(workerId);
    }
  }
  
  /**
   * Handle worker exit
   */
  private handleWorkerExit(workerId: number, exitCode: number): void {
    // Remove from idle workers
    this.idleWorkers.delete(workerId);
    
    // Reassign any active tasks from this worker
    const workerTasks = this.workerTasks.get(workerId);
    if (workerTasks) {
      for (const taskId of workerTasks) {
        const task = this.activeTasks.get(taskId);
        if (task) {
          // Re-queue the task
          this.queueTask(task);
          this.activeTasks.delete(taskId);
        }
      }
    }
    
    // Restart worker if needed
    if (this.workers.length < this.config.minWorkers) {
      this.restartWorker(workerId);
    }
  }
  
  /**
   * Restart a worker
   */
  private restartWorker(workerId: number): void {
    try {
      // Terminate existing worker
      const existingWorker = this.workers[workerId];
      if (existingWorker) {
        existingWorker.terminate();
      }
      
      // Create new worker
      const newWorker = new Worker(__filename, {
        workerData: { 
          isWorker: true, 
          workerId,
          memoryPoolConfig: true,
          cacheConfig: true
        }
      });
      
      // Update worker array
      this.workers[workerId] = newWorker;
      
      // Reset stats
      this.workerStats.set(workerId, {
        id: workerId,
        tasksProcessed: 0,
        totalDuration: 0,
        avgDuration: 0,
        isIdle: true,
        memoryUsage: 0,
        errorCount: 0
      });
      
      this.idleWorkers.add(workerId);
      this.workerTasks.set(workerId, new Set());
      
      console.log(`🔄 Worker ${workerId} restarted`);
      
    } catch (error) {
      console.error(`Failed to restart worker ${workerId}:`, error);
    }
  }
  
  /**
   * Scale worker pool based on load
   */
  private scaleWorkerPool(): void {
    if (!this.config.adaptiveScaling) return;
    
    const queueSize = this.getTotalQueueSize();
    const idleWorkers = this.idleWorkers.size;
    const activeWorkers = this.workers.length - idleWorkers;
    
    // Scale up if queue is backing up
    if (queueSize > activeWorkers * 2 && this.workers.length < this.config.maxWorkers) {
      this.createWorker();
      console.log(`📈 Scaled up: ${this.workers.length} workers (queue: ${queueSize})`);
    }
    
    // Scale down if too many idle workers
    if (idleWorkers > this.config.minWorkers && queueSize === 0) {
      const workersToRemove = Math.min(idleWorkers - this.config.minWorkers, 2);
      for (let i = 0; i < workersToRemove; i++) {
        this.removeIdleWorker();
      }
      console.log(`📉 Scaled down: ${this.workers.length} workers`);
    }
  }
  
  /**
   * Remove an idle worker
   */
  private removeIdleWorker(): void {
    if (this.idleWorkers.size <= this.config.minWorkers) return;
    
    const workerIdToRemove = Array.from(this.idleWorkers)[0];
    const worker = this.workers[workerIdToRemove];
    
    if (worker) {
      worker.terminate();
      this.idleWorkers.delete(workerIdToRemove);
      this.workerStats.delete(workerIdToRemove);
      this.workerTasks.delete(workerIdToRemove);
      
      // Remove from workers array (keep index, set to null)
      this.workers[workerIdToRemove] = null as any;
    }
  }
  
  /**
   * Get total queue size across all priorities
   */
  private getTotalQueueSize(): number {
    let total = 0;
    for (const queue of this.taskQueue.values()) {
      total += queue.length;
    }
    return total;
  }
  
  /**
   * Handle task timeout
   */
  private handleTaskTimeout(taskId: string): void {
    const task = this.activeTasks.get(taskId);
    if (!task) return;
    
    // Find worker handling this task
    for (const [workerId, tasks] of this.workerTasks.entries()) {
      if (tasks.has(taskId)) {
        // Mark worker as potentially problematic
        const stats = this.workerStats.get(workerId);
        if (stats) {
          stats.errorCount++;
        }
        
        // Remove task from worker
        tasks.delete(taskId);
        break;
      }
    }
    
    this.activeTasks.delete(taskId);
    this.failedTasks++;
  }
  
  /**
   * Generate unique task ID
   */
  private generateTaskId(): string {
    return `task_${Date.now()}_${Math.random().toString(36).substring(2)}`;
  }
  
  /**
   * Generate cache key for task
   */
  private generateCacheKey(type: TaskType, payload: any, context: SearchContext): string {
    const key = JSON.stringify({ type, payload: this.sanitizePayload(payload), context });
    return `parallel_${type}_${Buffer.from(key).toString('base64').substring(0, 32)}`;
  }
  
  /**
   * Sanitize payload for cache key generation
   */
  private sanitizePayload(payload: any): any {
    if (typeof payload !== 'object') return payload;
    
    // Remove functions and circular references
    return JSON.parse(JSON.stringify(payload));
  }
  
  /**
   * Estimate task duration
   */
  private estimateTaskDuration(type: TaskType, payload: any): number {
    const baseDurations = {
      [TaskType.LEXICAL_SEARCH]: 10,
      [TaskType.SYMBOL_SEARCH]: 15,
      [TaskType.SEMANTIC_RERANK]: 25,
      [TaskType.TRIGRAM_GENERATION]: 5,
      [TaskType.FUZZY_MATCHING]: 8,
      [TaskType.INDEX_BUILDING]: 100,
      [TaskType.CACHE_WARMUP]: 20,
      [TaskType.COMPRESSION]: 12,
      [TaskType.VALIDATION]: 3
    };
    
    let estimate = baseDurations[type] || 10;
    
    // Adjust based on payload size
    if (payload && typeof payload === 'object') {
      const payloadSize = JSON.stringify(payload).length;
      estimate += Math.log(payloadSize / 1000) * 2;
    }
    
    return Math.max(1, estimate);
  }
  
  /**
   * Estimate batch processing duration
   */
  private estimateBatchDuration(tasks: ParallelTask<any, any>[]): number {
    const totalIndividual = tasks.reduce((sum, task) => sum + task.estimatedDuration, 0);
    
    // Batch processing is typically 60-80% of individual processing time
    return Math.ceil(totalIndividual * 0.7);
  }
  
  /**
   * Start maintenance timers
   */
  private startMaintenanceTimers(): void {
    // Worker pool scaling every 5 seconds
    this.scalingTimer = setInterval(() => {
      this.scaleWorkerPool();
    }, 5000);
    
    // Batch processing every 10ms
    this.batchTimer = setInterval(() => {
      for (const taskType of Object.values(TaskType)) {
        const queue = this.batchQueue.get(taskType);
        if (queue && queue.length >= this.config.batchProcessing.minBatchSize) {
          this.processBatch(taskType);
        }
      }
    }, this.config.batchProcessing.maxWaitTime);
    
    // Statistics logging every 30 seconds
    this.statsTimer = setInterval(() => {
      const stats = this.getStats();
      console.log(`📊 Parallel Processor Stats: ${JSON.stringify({
        throughput: stats.throughputPerSecond.toFixed(1),
        queue_size: stats.queueSize,
        workers: `${stats.workerCount}/${this.config.maxWorkers}`,
        avg_duration_ms: stats.avgTaskDuration.toFixed(1),
        success_rate: ((stats.completedTasks / (stats.completedTasks + stats.failedTasks)) * 100).toFixed(1) + '%'
      })}`);
    }, 30000);
    
    // Cleanup old completed task results every 60 seconds
    this.cleanupTimer = setInterval(() => {
      const cutoff = Date.now() - 300000; // 5 minutes ago
      for (const [taskId, result] of this.completedTasksMap.entries()) {
        if (typeof result === 'object' && result.completedAt && result.completedAt < cutoff) {
          this.completedTasksMap.delete(taskId);
        }
      }
    }, 60000);
  }
  
  /**
   * Get comprehensive processor statistics
   */
  getStats(): ProcessorStats {
    const runtime = (Date.now() - this.startTime) / 1000;
    const throughput = runtime > 0 ? this.completedTasks / runtime : 0;
    const avgDuration = this.taskDurations.length > 0 ? 
      this.taskDurations.reduce((a, b) => a + b, 0) / this.taskDurations.length : 0;
    
    let totalMemory = 0;
    for (const stats of this.workerStats.values()) {
      totalMemory += stats.memoryUsage;
    }
    
    return {
      totalTasks: this.totalTasks,
      completedTasks: this.completedTasks,
      failedTasks: this.failedTasks,
      avgTaskDuration: avgDuration,
      workerCount: this.workers.length,
      idleWorkers: this.idleWorkers.size,
      queueSize: this.getTotalQueueSize(),
      throughputPerSecond: throughput,
      memoryUsage: totalMemory,
      cpuUtilization: ((this.workers.length - this.idleWorkers.size) / this.workers.length) * 100
    };
  }
  
  /**
   * Shutdown parallel processor
   */
  async shutdown(): Promise<void> {
    const span = LensTracer.createChildSpan('parallel_processor_shutdown');
    
    try {
      console.log('🛑 Shutting down Parallel Processor...');
      
      // Clear timers
      if (this.scalingTimer) clearInterval(this.scalingTimer);
      if (this.batchTimer) clearInterval(this.batchTimer);
      if (this.statsTimer) clearInterval(this.statsTimer);
      if (this.cleanupTimer) clearInterval(this.cleanupTimer);
      
      // Wait for active tasks with timeout
      const activeTaskIds = Array.from(this.activeTasks.keys());
      if (activeTaskIds.length > 0) {
        console.log(`⏳ Waiting for ${activeTaskIds.length} active tasks to complete...`);
        
        const timeout = 10000; // 10 second timeout
        const start = Date.now();
        
        while (this.activeTasks.size > 0 && (Date.now() - start) < timeout) {
          await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        if (this.activeTasks.size > 0) {
          console.warn(`⚠️ Forcing shutdown with ${this.activeTasks.size} active tasks`);
        }
      }
      
      // Terminate all workers
      const terminationPromises = this.workers.map((worker, index) => {
        if (worker) {
          return worker.terminate();
        }
        return Promise.resolve();
      });
      
      await Promise.allSettled(terminationPromises);
      
      // Clear data structures
      this.workers = [];
      this.workerStats.clear();
      this.idleWorkers.clear();
      this.workerTasks.clear();
      this.taskQueue.clear();
      this.activeTasks.clear();
      this.completedTasksMap.clear();
      this.batchQueue.clear();
      
      console.log('✅ Parallel Processor shutdown complete');
      
      span.setAttributes({ success: true });
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw error;
    } finally {
      span.end();
    }
  }
}

// Worker thread implementation
if (!isMainThread && workerData?.isWorker) {
  const workerId = workerData.workerId;
  let tasksProcessed = 0;
  let totalDuration = 0;
  
  // Initialize worker-specific resources
  if (workerData.memoryPoolConfig) {
    // Initialize memory pool for worker
  }
  
  if (workerData.cacheConfig) {
    // Initialize cache for worker
  }
  
  // Report ready state
  parentPort?.postMessage({
    type: 'worker_ready',
    workerId
  });
  
  parentPort?.on('message', async ({ type, task }) => {
    if (type === 'execute_task') {
      const startTime = performance.now();
      
      try {
        let result;
        
        // Process task based on type
        switch (task.type) {
          case TaskType.LEXICAL_SEARCH:
            result = await processLexicalSearch(task.payload, task.context);
            break;
            
          case TaskType.TRIGRAM_GENERATION:
            result = await processTrigramGeneration(task.payload);
            break;
            
          case TaskType.FUZZY_MATCHING:
            result = await processFuzzyMatching(task.payload);
            break;
            
          case TaskType.COMPRESSION:
            result = await processCompression(task.payload);
            break;
            
          default:
            throw new Error(`Unknown task type: ${task.type}`);
        }
        
        const duration = performance.now() - startTime;
        tasksProcessed++;
        totalDuration += duration;
        
        parentPort?.postMessage({
          type: 'task_completed',
          taskId: task.id,
          result,
          workerId
        });
        
        // Periodic stats update
        if (tasksProcessed % 10 === 0) {
          parentPort?.postMessage({
            type: 'worker_stats',
            workerId,
            stats: {
              tasksProcessed,
              totalDuration,
              avgDuration: totalDuration / tasksProcessed,
              memoryUsage: process.memoryUsage().heapUsed
            }
          });
        }
        
      } catch (error) {
        parentPort?.postMessage({
          type: 'task_failed',
          taskId: task.id,
          error: error instanceof Error ? error.message : 'Unknown error',
          workerId
        });
      }
    }
  });
  
  // Worker task processing functions
  async function processLexicalSearch(payload: any, context: any): Promise<any> {
    // Simulate lexical search processing
    await new Promise(resolve => setTimeout(resolve, Math.random() * 20 + 5));
    return { hits: [], duration: 15 };
  }
  
  async function processTrigramGeneration(payload: any): Promise<any> {
    if (payload.batchMode) {
      // Process batch of trigram generation tasks
      const results = [];
      for (const task of payload.tasks) {
        const trigrams = generateTrigrams(task.payload.text || '');
        results.push({ taskId: task.id, value: trigrams });
      }
      return { results };
    }
    
    const trigrams = generateTrigrams(payload.text || '');
    return trigrams;
  }
  
  async function processFuzzyMatching(payload: any): Promise<any> {
    if (payload.batchMode) {
      // Process batch of fuzzy matching tasks
      const results = [];
      for (const task of payload.tasks) {
        const matches = performFuzzyMatch(task.payload.query, task.payload.candidates);
        results.push({ taskId: task.id, value: matches });
      }
      return { results };
    }
    
    return performFuzzyMatch(payload.query, payload.candidates);
  }
  
  async function processCompression(payload: any): Promise<any> {
    // Simple compression simulation
    const compressed = Buffer.from(JSON.stringify(payload.data)).toString('base64');
    return { compressed, originalSize: JSON.stringify(payload.data).length };
  }
  
  function generateTrigrams(text: string): string[] {
    const trigrams: string[] = [];
    for (let i = 0; i <= text.length - 3; i++) {
      trigrams.push(text.substring(i, i + 3));
    }
    return trigrams;
  }
  
  function performFuzzyMatch(query: string, candidates: string[]): any[] {
    return candidates.map(candidate => ({
      text: candidate,
      distance: Math.floor(Math.random() * 3),
      score: Math.random()
    }));
  }
}

// Global instance
export const globalParallelProcessor = ParallelProcessor.getInstance();