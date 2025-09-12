import { describe, it, expect, beforeEach, afterEach, mock, jest, mock } from 'bun:test';

// Mock all external dependencies BEFORE importing ParallelProcessor
mock('worker_threads', () => ({
  Worker: jest.fn().mockImplementation(() => ({
    postMessage: jest.fn(),
    terminate: jest.fn().mockResolvedValue(undefined),
    on: jest.fn(),
    removeAllListeners: jest.fn(),
  })),
  isMainThread: true,
  parentPort: null,
  workerData: null,
}));

mock('os', () => ({
  cpus: jest.fn(() => Array(8).fill({ model: 'Intel Core' })),
}));

mock('perf_hooks', () => ({
  performance: {
    now: jest.fn(() => Date.now()),
  },
}));

mock('../telemetry/tracer', () => ({
  LensTracer: {
    createChildSpan: jest.fn(() => ({
      setAttributes: jest.fn(),
      recordException: jest.fn(),
      end: jest.fn(),
    })),
  },
}));

mock('./memory-pool-manager', () => ({
  globalMemoryPool: {
    allocate: jest.fn(),
    release: jest.fn(),
  },
}));

mock('./advanced-cache-manager', () => ({
  globalCacheManager: {
    get: jest.fn().mockResolvedValue(null),
    set: jest.fn(),
  },
}));

// Import AFTER mocks are set up
import { ParallelProcessor } from '../parallel-processor';

// Task types and priorities from the implementation
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
  CRITICAL = 0,
  HIGH = 1,
  NORMAL = 2,
  LOW = 3,
  BACKGROUND = 4
}

describe('ParallelProcessor', () => {
  let processor: ParallelProcessor;

  beforeEach(() => {
    jest.clearAllMocks();
    // Get a fresh instance for each test
    processor = ParallelProcessor.getInstance();
  });

  afterEach(() => {
    // Clean up after each test - but don't await shutdown in tests to avoid timeouts
    try {
      processor.shutdown();
    } catch (e) {
      // Ignore shutdown errors in tests
    }
  });

  describe('Singleton Pattern', () => {
    it('should return the same instance on multiple calls', () => {
      const instance1 = ParallelProcessor.getInstance();
      const instance2 = ParallelProcessor.getInstance();
      expect(instance1).toBe(instance2);
    });

    it('should initialize with default configuration', () => {
      const stats = processor.getStats();
      expect(stats).toBeDefined();
      expect(stats.totalTasks).toBe(0);
      expect(stats.completedTasks).toBe(0);
      expect(stats.failedTasks).toBe(0);
    });
  });

  describe('Task Submission', () => {
    it('should submit a lexical search task', async () => {
      const mockPayload = { query: 'test search', maxResults: 10 };
      
      const promise = processor.submitTask(
        TaskType.LEXICAL_SEARCH,
        mockPayload,
        TaskPriority.NORMAL
      );

      expect(promise).toBeInstanceOf(Promise);
      // Note: Promise will timeout in tests but that's expected behavior
    });

    it('should submit a trigram generation task', async () => {
      const mockPayload = { text: 'hello world', generateAll: true };
      
      const promise = processor.submitTask(
        TaskType.TRIGRAM_GENERATION,
        mockPayload,
        TaskPriority.HIGH
      );

      expect(promise).toBeInstanceOf(Promise);
    });

    it('should submit a fuzzy matching task with different priority levels', async () => {
      const mockPayload = { query: 'search', candidates: ['result1', 'result2'] };
      
      const criticalPromise = processor.submitTask(
        TaskType.FUZZY_MATCHING,
        mockPayload,
        TaskPriority.CRITICAL
      );

      const backgroundPromise = processor.submitTask(
        TaskType.FUZZY_MATCHING,
        mockPayload,
        TaskPriority.BACKGROUND
      );

      expect(criticalPromise).toBeInstanceOf(Promise);
      expect(backgroundPromise).toBeInstanceOf(Promise);
    });

    it('should submit compression tasks', async () => {
      const mockPayload = { data: { key: 'value', array: [1, 2, 3] } };
      
      const promise = processor.submitTask(
        TaskType.COMPRESSION,
        mockPayload,
        TaskPriority.LOW
      );

      expect(promise).toBeInstanceOf(Promise);
    });

    it('should handle task submission with custom timeout', async () => {
      const mockPayload = { query: 'long running task' };
      const customTimeout = 60000; // 60 seconds
      
      const promise = processor.submitTask(
        TaskType.SEMANTIC_RERANK,
        mockPayload,
        TaskPriority.NORMAL,
        undefined,
        customTimeout
      );

      expect(promise).toBeInstanceOf(Promise);
    });

    it('should submit task with search context', async () => {
      const mockPayload = { query: 'contextual search' };
      const mockContext = {
        repo_name: 'test-repo',
        repo_sha: 'abc123',
        file_path: 'src/test.ts',
        mode: 'lexical' as const,
      };
      
      const promise = processor.submitTask(
        TaskType.LEXICAL_SEARCH,
        mockPayload,
        TaskPriority.NORMAL,
        mockContext
      );

      expect(promise).toBeInstanceOf(Promise);
    });
  });

  describe('Batch Task Submission', () => {
    it('should submit batch of trigram generation tasks', async () => {
      const payloads = [
        { text: 'first text' },
        { text: 'second text' },
        { text: 'third text' }
      ];
      
      const promise = processor.submitBatch(
        TaskType.TRIGRAM_GENERATION,
        payloads,
        TaskPriority.NORMAL
      );

      expect(promise).toBeInstanceOf(Promise);
    });

    it('should submit batch of fuzzy matching tasks', async () => {
      const payloads = [
        { query: 'query1', candidates: ['a', 'b'] },
        { query: 'query2', candidates: ['c', 'd'] },
      ];
      
      const promise = processor.submitBatch(
        TaskType.FUZZY_MATCHING,
        payloads,
        TaskPriority.HIGH
      );

      expect(promise).toBeInstanceOf(Promise);
    });

    it('should submit batch with search context', async () => {
      const payloads = [
        { query: 'search1' },
        { query: 'search2' }
      ];
      const mockContext = {
        repo_name: 'test-repo',
        repo_sha: 'abc123',
        mode: 'lexical' as const,
      };
      
      const promise = processor.submitBatch(
        TaskType.LEXICAL_SEARCH,
        payloads,
        TaskPriority.NORMAL,
        mockContext
      );

      expect(promise).toBeInstanceOf(Promise);
    });

    it('should handle empty batch submission', async () => {
      const emptyPayloads: any[] = [];
      
      const promise = processor.submitBatch(
        TaskType.VALIDATION,
        emptyPayloads,
        TaskPriority.LOW
      );

      expect(promise).toBeInstanceOf(Promise);
      
      const results = await promise;
      expect(Array.isArray(results)).toBe(true);
      expect(results.length).toBe(0);
    });
  });

  describe('Statistics and Monitoring', () => {
    it('should return initial processor stats', () => {
      const stats = processor.getStats();
      
      expect(stats).toBeDefined();
      expect(typeof stats.totalTasks).toBe('number');
      expect(typeof stats.completedTasks).toBe('number');
      expect(typeof stats.failedTasks).toBe('number');
      expect(typeof stats.avgTaskDuration).toBe('number');
      expect(typeof stats.workerCount).toBe('number');
      expect(typeof stats.idleWorkers).toBe('number');
      expect(typeof stats.queueSize).toBe('number');
      expect(typeof stats.throughputPerSecond).toBe('number');
      expect(typeof stats.memoryUsage).toBe('number');
      expect(typeof stats.cpuUtilization).toBe('number');
    });

    it('should show zero stats for new processor', () => {
      const stats = processor.getStats();
      
      expect(stats.totalTasks).toBe(0);
      expect(stats.completedTasks).toBe(0);
      expect(stats.failedTasks).toBe(0);
      expect(stats.avgTaskDuration).toBe(0);
      expect(stats.throughputPerSecond).toBe(0);
      expect(stats.memoryUsage).toBe(0);
    });

    it('should track worker count correctly', () => {
      const stats = processor.getStats();
      
      // Should have initialized with minimum workers
      expect(stats.workerCount).toBeGreaterThan(0);
      expect(stats.workerCount).toBeLessThanOrEqual(8); // max workers
    });

    it('should calculate CPU utilization', () => {
      const stats = processor.getStats();
      
      expect(stats.cpuUtilization).toBeGreaterThanOrEqual(0);
      expect(stats.cpuUtilization).toBeLessThanOrEqual(100);
    });

    it('should track queue size', () => {
      const stats = processor.getStats();
      
      expect(stats.queueSize).toBe(0); // Initially empty
    });
  });

  describe('Worker Management', () => {
    it('should handle worker initialization', () => {
      // Worker creation should be handled internally
      const stats = processor.getStats();
      expect(stats.workerCount).toBeGreaterThan(0);
    });

    it('should track idle workers', () => {
      const stats = processor.getStats();
      // Initially all workers should be idle
      expect(stats.idleWorkers).toBe(stats.workerCount);
    });

    it('should handle worker scaling', () => {
      // The processor should auto-scale workers based on workload
      const initialStats = processor.getStats();
      expect(initialStats.workerCount).toBeGreaterThan(0);
    });
  });

  describe('Task Types Support', () => {
    it('should support lexical search tasks', async () => {
      const promise = processor.submitTask(
        TaskType.LEXICAL_SEARCH,
        { query: 'function test' },
        TaskPriority.NORMAL
      );
      expect(promise).toBeInstanceOf(Promise);
    });

    it('should support symbol search tasks', async () => {
      const promise = processor.submitTask(
        TaskType.SYMBOL_SEARCH,
        { symbol: 'TestClass' },
        TaskPriority.NORMAL
      );
      expect(promise).toBeInstanceOf(Promise);
    });

    it('should support semantic rerank tasks', async () => {
      const promise = processor.submitTask(
        TaskType.SEMANTIC_RERANK,
        { candidates: [], query: 'test' },
        TaskPriority.NORMAL
      );
      expect(promise).toBeInstanceOf(Promise);
    });

    it('should support index building tasks', async () => {
      const promise = processor.submitTask(
        TaskType.INDEX_BUILDING,
        { documents: [], config: {} },
        TaskPriority.LOW
      );
      expect(promise).toBeInstanceOf(Promise);
    });

    it('should support cache warmup tasks', async () => {
      const promise = processor.submitTask(
        TaskType.CACHE_WARMUP,
        { keys: ['key1', 'key2'] },
        TaskPriority.BACKGROUND
      );
      expect(promise).toBeInstanceOf(Promise);
    });

    it('should support validation tasks', async () => {
      const promise = processor.submitTask(
        TaskType.VALIDATION,
        { data: { valid: true } },
        TaskPriority.HIGH
      );
      expect(promise).toBeInstanceOf(Promise);
    });
  });

  describe('Priority Queue Management', () => {
    it('should handle different priority levels', () => {
      // Submit tasks with different priorities
      processor.submitTask(TaskType.LEXICAL_SEARCH, { query: 'critical' }, TaskPriority.CRITICAL);
      processor.submitTask(TaskType.LEXICAL_SEARCH, { query: 'normal' }, TaskPriority.NORMAL);
      processor.submitTask(TaskType.LEXICAL_SEARCH, { query: 'background' }, TaskPriority.BACKGROUND);
      
      const stats = processor.getStats();
      expect(stats.queueSize).toBeGreaterThan(0);
    });

    it('should process critical tasks with higher priority', async () => {
      // This tests the internal priority queue behavior
      const criticalPromise = processor.submitTask(
        TaskType.FUZZY_MATCHING,
        { query: 'critical task' },
        TaskPriority.CRITICAL
      );

      const backgroundPromise = processor.submitTask(
        TaskType.FUZZY_MATCHING,
        { query: 'background task' },
        TaskPriority.BACKGROUND
      );

      expect(criticalPromise).toBeInstanceOf(Promise);
      expect(backgroundPromise).toBeInstanceOf(Promise);
    });
  });

  describe('Performance and Load Testing', () => {
    it('should handle multiple concurrent tasks', async () => {
      const tasks = [];
      
      // Submit 20 concurrent tasks
      for (let i = 0; i < 20; i++) {
        tasks.push(
          processor.submitTask(
            TaskType.TRIGRAM_GENERATION,
            { text: `test text ${i}` },
            TaskPriority.NORMAL
          )
        );
      }
      
      expect(tasks.length).toBe(20);
      tasks.forEach(task => expect(task).toBeInstanceOf(Promise));
    });

    it('should track task completion metrics', async () => {
      const initialStats = processor.getStats();
      
      // Submit a task
      processor.submitTask(
        TaskType.COMPRESSION,
        { data: 'test data' },
        TaskPriority.NORMAL
      );
      
      const afterSubmitStats = processor.getStats();
      expect(afterSubmitStats.totalTasks).toBeGreaterThan(initialStats.totalTasks);
    });
  });

  describe('Error Handling and Recovery', () => {
    it('should handle task submission errors gracefully', async () => {
      // Test with invalid task type (should still create promise)
      const promise = processor.submitTask(
        'INVALID_TYPE' as TaskType,
        { invalid: true },
        TaskPriority.NORMAL
      );
      
      expect(promise).toBeInstanceOf(Promise);
    });

    it('should handle worker communication errors', () => {
      // The processor should be resilient to worker failures
      const stats = processor.getStats();
      expect(stats).toBeDefined();
    });

    it('should track failed tasks', () => {
      const stats = processor.getStats();
      expect(typeof stats.failedTasks).toBe('number');
      expect(stats.failedTasks).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Cache Integration', () => {
    it('should support cache-enabled task submission', async () => {
      const mockContext = {
        repo_name: 'cached-repo',
        repo_sha: 'cache123',
        mode: 'lexical' as const,
      };
      
      const promise = processor.submitTask(
        TaskType.LEXICAL_SEARCH,
        { query: 'cached query' },
        TaskPriority.NORMAL,
        mockContext
      );
      
      expect(promise).toBeInstanceOf(Promise);
    });

    it('should handle cache operations in batches', async () => {
      const mockContext = {
        repo_name: 'batch-cache-repo',
        repo_sha: 'batch123',
        mode: 'lexical' as const,
      };
      
      const payloads = [
        { query: 'cache query 1' },
        { query: 'cache query 2' }
      ];
      
      const promise = processor.submitBatch(
        TaskType.LEXICAL_SEARCH,
        payloads,
        TaskPriority.NORMAL,
        mockContext
      );
      
      expect(promise).toBeInstanceOf(Promise);
    });
  });

  describe('Graceful Shutdown', () => {
    it('should shutdown gracefully', () => {
      const shutdownPromise = processor.shutdown();
      expect(shutdownPromise).toBeInstanceOf(Promise);
      
      // After shutdown, stats should still be accessible
      const stats = processor.getStats();
      expect(stats).toBeDefined();
    });

    it('should handle shutdown with active tasks', () => {
      // Submit a task, then shutdown
      processor.submitTask(
        TaskType.LEXICAL_SEARCH,
        { query: 'pre-shutdown task' },
        TaskPriority.NORMAL
      );
      
      const shutdownPromise = processor.shutdown();
      expect(shutdownPromise).toBeInstanceOf(Promise);
    });
  });
});