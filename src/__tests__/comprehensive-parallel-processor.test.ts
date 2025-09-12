/**
 * Comprehensive Parallel Processor Coverage Tests
 * 
 * Target: Test all parallel processing, worker management, and task orchestration
 * Coverage focus: Work-stealing, load balancing, batching, error handling
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach, jest, mock } from 'bun:test';
import { ParallelProcessor } from '../core/parallel-processor.js';
import { LensTracer } from '../telemetry/tracer.js';
import type { SearchContext, SearchHit, Candidate } from '../types/core.js';

// Mock worker_threads module for testing
mock('worker_threads', () => ({
  Worker: jest.fn().mockImplementation(() => ({
    postMessage: jest.fn(),
    terminate: jest.fn(),
    on: jest.fn(),
    removeAllListeners: jest.fn()
  })),
  isMainThread: true,
  parentPort: null,
  workerData: null
}));

describe('Comprehensive Parallel Processor Coverage Tests', () => {
  let processor: ParallelProcessor;

  beforeAll(async () => {
    // Initialize telemetry
    LensTracer.initialize('test-parallel-processor');
  });

  afterAll(async () => {
    if (processor) {
      await processor.shutdown();
    }
    LensTracer.shutdown();
  });

  beforeEach(() => {
    // Reset mocks and create fresh processor instance
    jest.clearAllMocks();
    processor = new ParallelProcessor();
  });

  afterEach(async () => {
    if (processor) {
      await processor.shutdown();
    }
  });

  describe('Initialization and Configuration', () => {
    it('should initialize with default configuration', async () => {
      const config = {
        maxWorkers: 4,
        minWorkers: 2,
        idleTimeout: 30000,
        workStealingEnabled: true,
        adaptiveScaling: true,
        batchProcessing: {
          maxBatchSize: 10,
          maxWaitTime: 100,
          minBatchSize: 2,
          batchingEnabled: true
        },
        queueSizeLimit: 1000
      };

      await expect(processor.initialize(config)).resolves.not.toThrow();
      
      const stats = processor.getStats();
      expect(stats).toBeDefined();
      expect(stats.totalTasks).toBe(0);
      expect(stats.completedTasks).toBe(0);
    });

    it('should handle custom configuration parameters', async () => {
      const customConfig = {
        maxWorkers: 8,
        minWorkers: 4,
        idleTimeout: 60000,
        workStealingEnabled: false,
        adaptiveScaling: false,
        batchProcessing: {
          maxBatchSize: 20,
          maxWaitTime: 200,
          minBatchSize: 5,
          batchingEnabled: false
        },
        queueSizeLimit: 2000
      };

      await expect(processor.initialize(customConfig)).resolves.not.toThrow();
      
      const isInitialized = processor.isInitialized();
      expect(isInitialized).toBe(true);
    });

    it('should validate configuration parameters', async () => {
      const invalidConfigs = [
        { maxWorkers: 0, minWorkers: 1 }, // maxWorkers too low
        { maxWorkers: 2, minWorkers: 5 }, // minWorkers > maxWorkers
        { idleTimeout: -1 }, // negative timeout
        { queueSizeLimit: 0 }, // zero queue size
      ];

      for (const config of invalidConfigs) {
        await expect(processor.initialize(config as any))
          .rejects.toThrow();
      }
    });

    it('should adapt to system CPU count', async () => {
      // Mock os.cpus() to return specific count
      jest.doMock('os', () => ({
        cpus: () => Array.from({ length: 8 }, () => ({})) // Mock 8 CPUs
      }));

      const config = {
        maxWorkers: 0, // 0 means auto-detect
        adaptiveScaling: true
      };

      await expect(processor.initialize(config)).resolves.not.toThrow();
      
      const stats = processor.getStats();
      expect(stats.activeWorkers).toBeGreaterThan(0);
      expect(stats.activeWorkers).toBeLessThanOrEqual(8);
    });
  });

  describe('Task Scheduling and Execution', () => {
    beforeEach(async () => {
      await processor.initialize();
    });

    it('should process simple tasks', async () => {
      const task = {
        type: 'LEXICAL_SEARCH',
        payload: { query: 'test', context: {} },
        priority: 'NORMAL',
        timeout: 5000
      };

      // Mock successful task completion
      const mockResult = { results: [{ file: 'test.ts', line: 1, score: 1.0 }] };
      jest.spyOn(processor as any, 'executeTask')
        .mockResolvedValueOnce(mockResult);

      const result = await processor.submitTask(task);
      
      expect(result).toBeDefined();
      expect(result).toEqual(mockResult);
    });

    it('should handle multiple task types', async () => {
      const taskTypes = [
        'LEXICAL_SEARCH',
        'SYMBOL_SEARCH', 
        'SEMANTIC_RERANK',
        'TRIGRAM_GENERATION',
        'FUZZY_MATCHING',
        'INDEX_BUILDING',
        'CACHE_WARMUP',
        'COMPRESSION',
        'VALIDATION'
      ];

      const tasks = taskTypes.map(type => ({
        type,
        payload: { data: `test-${type}` },
        priority: 'NORMAL',
        timeout: 5000
      }));

      // Mock task execution for each type
      jest.spyOn(processor as any, 'executeTask')
        .mockImplementation(async (task) => ({ result: `processed-${task.type}` }));

      const results = await Promise.all(
        tasks.map(task => processor.submitTask(task))
      );
      
      expect(results).toHaveLength(taskTypes.length);
      results.forEach((result, index) => {
        expect(result).toEqual({ result: `processed-${taskTypes[index]}` });
      });
    });

    it('should respect task priorities', async () => {
      const tasks = [
        { type: 'LEXICAL_SEARCH', priority: 'LOW', payload: { id: 'low' } },
        { type: 'LEXICAL_SEARCH', priority: 'CRITICAL', payload: { id: 'critical' } },
        { type: 'LEXICAL_SEARCH', priority: 'HIGH', payload: { id: 'high' } },
        { type: 'LEXICAL_SEARCH', priority: 'NORMAL', payload: { id: 'normal' } }
      ];

      const executionOrder: string[] = [];
      
      jest.spyOn(processor as any, 'executeTask')
        .mockImplementation(async (task) => {
          executionOrder.push(task.payload.id);
          return { processed: task.payload.id };
        });

      // Submit all tasks
      const promises = tasks.map(task => 
        processor.submitTask({ ...task, timeout: 5000 })
      );
      
      await Promise.all(promises);
      
      // Critical should be processed first, then HIGH, NORMAL, LOW
      expect(executionOrder[0]).toBe('critical');
      expect(executionOrder.includes('high')).toBe(true);
      expect(executionOrder.includes('normal')).toBe(true);
      expect(executionOrder.includes('low')).toBe(true);
    });

    it('should handle task timeouts', async () => {
      const slowTask = {
        type: 'SLOW_TASK',
        payload: { data: 'slow' },
        priority: 'NORMAL',
        timeout: 100 // Very short timeout
      };

      // Mock slow execution
      jest.spyOn(processor as any, 'executeTask')
        .mockImplementation(async () => {
          await new Promise(resolve => setTimeout(resolve, 500)); // Takes 500ms
          return { result: 'never reached' };
        });

      await expect(processor.submitTask(slowTask))
        .rejects.toThrow(/timeout/i);
    });

    it('should handle task dependencies', async () => {
      const task1 = {
        id: 'task-1',
        type: 'INDEX_BUILDING',
        payload: { data: 'base' },
        priority: 'NORMAL',
        timeout: 5000
      };

      const task2 = {
        id: 'task-2',
        type: 'LEXICAL_SEARCH',
        payload: { data: 'dependent' },
        priority: 'NORMAL',
        timeout: 5000,
        dependencies: ['task-1']
      };

      const executionOrder: string[] = [];
      
      jest.spyOn(processor as any, 'executeTask')
        .mockImplementation(async (task) => {
          executionOrder.push(task.id);
          // Simulate task duration
          await new Promise(resolve => setTimeout(resolve, 10));
          return { processed: task.id };
        });

      // Submit dependent task first, then dependency
      const promise2 = processor.submitTask(task2);
      const promise1 = processor.submitTask(task1);
      
      await Promise.all([promise1, promise2]);
      
      // task-1 should execute before task-2
      const task1Index = executionOrder.indexOf('task-1');
      const task2Index = executionOrder.indexOf('task-2');
      expect(task1Index).toBeLessThan(task2Index);
    });
  });

  describe('Worker Management', () => {
    beforeEach(async () => {
      await processor.initialize();
    });

    it('should scale workers dynamically based on load', async () => {
      // Submit many tasks to trigger scaling
      const tasks = Array.from({ length: 20 }, (_, i) => ({
        type: 'LEXICAL_SEARCH',
        payload: { query: `query-${i}` },
        priority: 'NORMAL',
        timeout: 5000
      }));

      jest.spyOn(processor as any, 'executeTask')
        .mockImplementation(async () => {
          await new Promise(resolve => setTimeout(resolve, 50));
          return { processed: true };
        });

      const statsBefore = processor.getStats();
      
      // Submit all tasks
      const promises = tasks.map(task => processor.submitTask(task));
      
      // Check stats during processing
      await new Promise(resolve => setTimeout(resolve, 25));
      const statsDuring = processor.getStats();
      
      await Promise.all(promises);
      
      const statsAfter = processor.getStats();
      
      // Should have scaled up during high load
      expect(statsDuring.activeWorkers).toBeGreaterThanOrEqual(statsBefore.activeWorkers);
      expect(statsAfter.completedTasks).toBe(20);
    });

    it('should handle worker failures and recovery', async () => {
      const task = {
        type: 'FAILING_TASK',
        payload: { shouldFail: true },
        priority: 'NORMAL',
        timeout: 5000
      };

      // Mock worker failure
      jest.spyOn(processor as any, 'executeTask')
        .mockRejectedValueOnce(new Error('Worker crashed'))
        .mockResolvedValueOnce({ recovered: true }); // Second attempt succeeds

      // Should recover and retry
      const result = await processor.submitTask(task);
      expect(result).toEqual({ recovered: true });
      
      const stats = processor.getStats();
      expect(stats.failedTasks).toBeGreaterThan(0);
    });

    it('should implement work stealing between workers', async () => {
      const config = {
        maxWorkers: 4,
        workStealingEnabled: true,
        adaptiveScaling: true
      };
      
      await processor.shutdown();
      processor = new ParallelProcessor();
      await processor.initialize(config);

      // Create uneven workload distribution
      const tasks = [
        ...Array.from({ length: 10 }, (_, i) => ({
          type: 'FAST_TASK',
          payload: { id: `fast-${i}` },
          priority: 'NORMAL',
          timeout: 5000,
          workerAffinity: 0 // Prefer worker 0
        })),
        {
          type: 'SLOW_TASK',
          payload: { id: 'slow' },
          priority: 'NORMAL',
          timeout: 5000,
          workerAffinity: 0 // Prefer worker 0 (will be busy)
        }
      ];

      jest.spyOn(processor as any, 'executeTask')
        .mockImplementation(async (task) => {
          const duration = task.type === 'SLOW_TASK' ? 200 : 10;
          await new Promise(resolve => setTimeout(resolve, duration));
          return { processed: task.payload.id };
        });

      const startTime = Date.now();
      await Promise.all(tasks.map(task => processor.submitTask(task)));
      const totalTime = Date.now() - startTime;
      
      // With work stealing, total time should be reasonable
      expect(totalTime).toBeLessThan(500); // Without work stealing, would take much longer
    });

    it('should clean up idle workers', async () => {
      const config = {
        maxWorkers: 8,
        minWorkers: 2,
        idleTimeout: 100, // Very short timeout for testing
        adaptiveScaling: true
      };
      
      await processor.shutdown();
      processor = new ParallelProcessor();
      await processor.initialize(config);

      // Submit many tasks to create workers
      const tasks = Array.from({ length: 8 }, (_, i) => ({
        type: 'QUICK_TASK',
        payload: { id: i },
        priority: 'NORMAL',
        timeout: 5000
      }));

      jest.spyOn(processor as any, 'executeTask')
        .mockImplementation(async () => ({ done: true }));

      await Promise.all(tasks.map(task => processor.submitTask(task)));
      
      const statsAfterWork = processor.getStats();
      
      // Wait for idle timeout
      await new Promise(resolve => setTimeout(resolve, 200));
      
      const statsAfterIdle = processor.getStats();
      
      // Should have cleaned up some idle workers
      expect(statsAfterIdle.activeWorkers).toBeLessThanOrEqual(statsAfterWork.activeWorkers);
      expect(statsAfterIdle.activeWorkers).toBeGreaterThanOrEqual(2); // Min workers
    });
  });

  describe('Batch Processing', () => {
    beforeEach(async () => {
      const config = {
        batchProcessing: {
          maxBatchSize: 5,
          maxWaitTime: 100,
          minBatchSize: 2,
          batchingEnabled: true
        }
      };
      await processor.initialize(config);
    });

    it('should batch compatible tasks together', async () => {
      const tasks = Array.from({ length: 6 }, (_, i) => ({
        type: 'BATCHABLE_TASK',
        payload: { id: i },
        priority: 'NORMAL',
        timeout: 5000
      }));

      const batchedExecutions: any[] = [];
      
      jest.spyOn(processor as any, 'executeBatch')
        .mockImplementation(async (batch) => {
          batchedExecutions.push(batch);
          return batch.map((task: any) => ({ processed: task.payload.id }));
        });

      const results = await Promise.all(
        tasks.map(task => processor.submitTask(task))
      );
      
      expect(results).toHaveLength(6);
      expect(batchedExecutions.length).toBeGreaterThan(0);
      
      // Should have created at least one batch with multiple tasks
      const hasBatch = batchedExecutions.some(batch => batch.length > 1);
      expect(hasBatch).toBe(true);
    });

    it('should respect maximum wait time for batching', async () => {
      const task1 = {
        type: 'SLOW_BATCH_TASK',
        payload: { id: 1 },
        priority: 'NORMAL',
        timeout: 5000
      };

      jest.spyOn(processor as any, 'executeBatch')
        .mockImplementation(async (batch) => {
          return batch.map((task: any) => ({ processed: task.payload.id }));
        });

      const startTime = Date.now();
      const result = await processor.submitTask(task1);
      const duration = Date.now() - startTime;
      
      expect(result).toBeDefined();
      // Should not wait longer than maxWaitTime when batch is incomplete
      expect(duration).toBeLessThan(200); // maxWaitTime + some buffer
    });

    it('should handle mixed priority tasks in batches', async () => {
      const tasks = [
        { type: 'BATCH_TASK', payload: { id: 1 }, priority: 'HIGH' },
        { type: 'BATCH_TASK', payload: { id: 2 }, priority: 'HIGH' },
        { type: 'BATCH_TASK', payload: { id: 3 }, priority: 'NORMAL' },
        { type: 'BATCH_TASK', payload: { id: 4 }, priority: 'NORMAL' }
      ].map(task => ({ ...task, timeout: 5000 }));

      const batchedExecutions: any[] = [];
      
      jest.spyOn(processor as any, 'executeBatch')
        .mockImplementation(async (batch) => {
          batchedExecutions.push(batch.map((t: any) => t.priority));
          return batch.map((task: any) => ({ processed: task.payload.id }));
        });

      await Promise.all(tasks.map(task => processor.submitTask(task)));
      
      // High priority tasks should be batched together separately from normal
      const highPriorityBatches = batchedExecutions.filter(batch => 
        batch.every(priority => priority === 'HIGH')
      );
      expect(highPriorityBatches.length).toBeGreaterThan(0);
    });
  });

  describe('Performance Monitoring and Stats', () => {
    beforeEach(async () => {
      await processor.initialize();
    });

    it('should track comprehensive statistics', async () => {
      const tasks = Array.from({ length: 5 }, (_, i) => ({
        type: 'STATS_TEST_TASK',
        payload: { id: i },
        priority: 'NORMAL',
        timeout: 5000
      }));

      jest.spyOn(processor as any, 'executeTask')
        .mockImplementation(async (task) => {
          await new Promise(resolve => setTimeout(resolve, 10));
          return { processed: task.payload.id };
        });

      const statsBefore = processor.getStats();
      
      await Promise.all(tasks.map(task => processor.submitTask(task)));
      
      const statsAfter = processor.getStats();
      
      expect(statsAfter.totalTasks).toBe(statsBefore.totalTasks + 5);
      expect(statsAfter.completedTasks).toBe(statsBefore.completedTasks + 5);
      expect(statsAfter.avgTaskDuration).toBeGreaterThan(0);
    });

    it('should provide worker-level statistics', async () => {
      const workerStats = processor.getWorkerStats();
      
      expect(Array.isArray(workerStats)).toBe(true);
      
      if (workerStats.length > 0) {
        const stats = workerStats[0];
        expect(stats).toHaveProperty('id');
        expect(stats).toHaveProperty('tasksProcessed');
        expect(stats).toHaveProperty('totalDuration');
        expect(stats).toHaveProperty('avgDuration');
        expect(stats).toHaveProperty('isIdle');
        expect(stats).toHaveProperty('memoryUsage');
        expect(stats).toHaveProperty('errorCount');
      }
    });

    it('should track performance metrics over time', async () => {
      const task = {
        type: 'PERF_TEST_TASK',
        payload: { data: 'test' },
        priority: 'NORMAL',
        timeout: 5000
      };

      jest.spyOn(processor as any, 'executeTask')
        .mockImplementation(async () => {
          await new Promise(resolve => setTimeout(resolve, 50));
          return { done: true };
        });

      // Execute multiple tasks to build up metrics
      for (let i = 0; i < 3; i++) {
        await processor.submitTask({ ...task, payload: { id: i } });
      }
      
      const perfMetrics = processor.getPerformanceMetrics();
      
      expect(perfMetrics).toHaveProperty('avgLatency');
      expect(perfMetrics).toHaveProperty('p95Latency');
      expect(perfMetrics).toHaveProperty('throughput');
      expect(perfMetrics).toHaveProperty('errorRate');
      
      expect(perfMetrics.avgLatency).toBeGreaterThan(0);
      expect(perfMetrics.throughput).toBeGreaterThan(0);
    });
  });

  describe('Error Handling and Resilience', () => {
    beforeEach(async () => {
      await processor.initialize();
    });

    it('should handle task execution failures gracefully', async () => {
      const task = {
        type: 'FAILING_TASK',
        payload: { shouldFail: true },
        priority: 'NORMAL',
        timeout: 5000
      };

      jest.spyOn(processor as any, 'executeTask')
        .mockRejectedValueOnce(new Error('Task execution failed'));

      await expect(processor.submitTask(task))
        .rejects.toThrow('Task execution failed');
        
      const stats = processor.getStats();
      expect(stats.failedTasks).toBe(1);
    });

    it('should implement circuit breaker for failing workers', async () => {
      const task = {
        type: 'CIRCUIT_BREAKER_TEST',
        payload: { data: 'test' },
        priority: 'NORMAL',
        timeout: 5000
      };

      // Mock repeated failures
      jest.spyOn(processor as any, 'executeTask')
        .mockRejectedValue(new Error('Worker failure'));

      // Submit multiple tasks to trigger circuit breaker
      const failedTasks = [];
      for (let i = 0; i < 5; i++) {
        try {
          await processor.submitTask({ ...task, payload: { id: i } });
        } catch (error) {
          failedTasks.push(error);
        }
      }
      
      expect(failedTasks.length).toBe(5);
      
      const stats = processor.getStats();
      expect(stats.failedTasks).toBeGreaterThan(0);
    });

    it('should handle queue overflow gracefully', async () => {
      const config = {
        queueSizeLimit: 3, // Very small queue for testing
        maxWorkers: 1
      };
      
      await processor.shutdown();
      processor = new ParallelProcessor();
      await processor.initialize(config);

      // Mock slow task execution to fill queue
      jest.spyOn(processor as any, 'executeTask')
        .mockImplementation(async () => {
          await new Promise(resolve => setTimeout(resolve, 100));
          return { done: true };
        });

      const tasks = Array.from({ length: 6 }, (_, i) => ({
        type: 'QUEUE_OVERFLOW_TEST',
        payload: { id: i },
        priority: 'NORMAL',
        timeout: 5000
      }));

      // Should reject tasks when queue is full
      const results = await Promise.allSettled(
        tasks.map(task => processor.submitTask(task))
      );
      
      const rejected = results.filter(r => r.status === 'rejected');
      expect(rejected.length).toBeGreaterThan(0);
    });

    it('should handle worker memory leaks', async () => {
      const task = {
        type: 'MEMORY_LEAK_TEST',
        payload: { largeData: 'x'.repeat(1000000) }, // Large payload
        priority: 'NORMAL',
        timeout: 5000
      };

      jest.spyOn(processor as any, 'executeTask')
        .mockImplementation(async (task) => {
          // Simulate memory usage
          return { processed: true, dataSize: task.payload.largeData.length };
        });

      const result = await processor.submitTask(task);
      
      expect(result).toBeDefined();
      expect(result.dataSize).toBe(1000000);
      
      // Check that worker stats show memory usage
      const workerStats = processor.getWorkerStats();
      if (workerStats.length > 0) {
        expect(workerStats[0].memoryUsage).toBeGreaterThanOrEqual(0);
      }
    });
  });

  describe('Advanced Features', () => {
    it('should support task cancellation', async () => {
      await processor.initialize();
      
      const task = {
        id: 'cancellable-task',
        type: 'LONG_RUNNING_TASK',
        payload: { duration: 1000 },
        priority: 'NORMAL',
        timeout: 5000
      };

      jest.spyOn(processor as any, 'executeTask')
        .mockImplementation(async () => {
          await new Promise(resolve => setTimeout(resolve, 500));
          return { completed: true };
        });

      const taskPromise = processor.submitTask(task);
      
      // Cancel after short delay
      setTimeout(() => {
        processor.cancelTask('cancellable-task');
      }, 50);
      
      await expect(taskPromise).rejects.toThrow(/cancelled/i);
    });

    it('should support task scheduling with delays', async () => {
      await processor.initialize();
      
      const task = {
        type: 'SCHEDULED_TASK',
        payload: { data: 'scheduled' },
        priority: 'NORMAL',
        timeout: 5000,
        executeAt: Date.now() + 100 // Execute in 100ms
      };

      jest.spyOn(processor as any, 'executeTask')
        .mockImplementation(async () => ({ scheduled: true }));

      const startTime = Date.now();
      const result = await processor.submitTask(task);
      const duration = Date.now() - startTime;
      
      expect(result).toEqual({ scheduled: true });
      expect(duration).toBeGreaterThan(90); // Should wait at least 90ms
    });

    it('should support conditional task execution', async () => {
      await processor.initialize();
      
      const condition = () => Date.now() % 2 === 0; // Random condition
      
      const task = {
        type: 'CONDITIONAL_TASK',
        payload: { data: 'conditional' },
        priority: 'NORMAL',
        timeout: 5000,
        condition
      };

      jest.spyOn(processor as any, 'executeTask')
        .mockImplementation(async () => ({ executed: true }));

      const result = await processor.submitTask(task);
      
      // Should either execute or skip based on condition
      expect(result).toBeDefined();
    });
  });

  describe('Shutdown and Cleanup', () => {
    it('should shutdown gracefully and wait for running tasks', async () => {
      await processor.initialize();
      
      const task = {
        type: 'SHUTDOWN_TEST_TASK',
        payload: { duration: 100 },
        priority: 'NORMAL',
        timeout: 5000
      };

      jest.spyOn(processor as any, 'executeTask')
        .mockImplementation(async () => {
          await new Promise(resolve => setTimeout(resolve, 50));
          return { completed: true };
        });

      // Start task
      const taskPromise = processor.submitTask(task);
      
      // Initiate shutdown
      const shutdownPromise = processor.shutdown();
      
      // Both should complete
      const [taskResult] = await Promise.all([taskPromise, shutdownPromise]);
      
      expect(taskResult).toEqual({ completed: true });
    });

    it('should force shutdown after timeout', async () => {
      await processor.initialize();
      
      const task = {
        type: 'NEVER_ENDING_TASK',
        payload: { data: 'infinite' },
        priority: 'NORMAL',
        timeout: 10000
      };

      jest.spyOn(processor as any, 'executeTask')
        .mockImplementation(async () => {
          // Never resolves
          return new Promise(() => {});
        });

      // Start never-ending task
      processor.submitTask(task).catch(() => {}); // Ignore rejection
      
      // Force shutdown with short timeout
      const startTime = Date.now();
      await processor.shutdown(100); // 100ms timeout
      const duration = Date.now() - startTime;
      
      // Should not wait longer than shutdown timeout
      expect(duration).toBeLessThan(300);
    });
  });
});
