/**
 * Performance tests for LSP components
 * Validates ≤+3ms Stage-B p95 latency constraint and other performance bounds
 * Tests throughput, memory usage, and concurrent request handling
 */

import { describe, test, expect, vi, beforeEach, afterEach } from 'vitest';
import { LSPStageBEnhancer } from '../core/lsp-stage-b.js';
import { LSPStageCEnhancer } from '../core/lsp-stage-c.js';
import { IntentRouter } from '../core/intent-router.js';
import { LSPSidecar } from '../core/lsp-sidecar.js';
import { WorkspaceConfig } from '../core/workspace-config.js';
import { SearchEngine } from '../api/search-engine.js';
import type {
  Candidate,
  SearchContext,
  QueryIntent,
  LSPHint
} from '../types/core.js';

// Mock external dependencies
vi.mock('fs', () => ({
  readFileSync: vi.fn(),
  writeFileSync: vi.fn(),
  existsSync: vi.fn().mockReturnValue(true),
  promises: {
    readFile: vi.fn(),
    writeFile: vi.fn(),
    access: vi.fn().mockResolvedValue(true),
  },
}));

vi.mock('child_process', () => ({
  spawn: vi.fn(() => ({
    stdout: { on: vi.fn(), setEncoding: vi.fn() },
    stderr: { on: vi.fn(), setEncoding: vi.fn() },
    stdin: { write: vi.fn(), end: vi.fn() },
    on: vi.fn(),
    kill: vi.fn(),
    pid: 12345,
  })),
}));

// Mock telemetry
vi.mock('../telemetry/tracer.js', () => ({
  LensTracer: {
    createChildSpan: vi.fn(() => ({
      setAttributes: vi.fn(),
      recordException: vi.fn(),
      end: vi.fn(),
    })),
  },
}));

// Performance measurement utilities
class PerformanceTracker {
  private measurements: number[] = [];

  measure<T>(fn: () => T | Promise<T>): T | Promise<T> {
    const start = performance.now();
    const result = fn();
    
    if (result instanceof Promise) {
      return result.then(value => {
        const end = performance.now();
        this.measurements.push(end - start);
        return value;
      });
    } else {
      const end = performance.now();
      this.measurements.push(end - start);
      return result;
    }
  }

  getStats() {
    if (this.measurements.length === 0) {
      return { avg: 0, p50: 0, p95: 0, p99: 0, min: 0, max: 0, count: 0 };
    }

    const sorted = [...this.measurements].sort((a, b) => a - b);
    const count = sorted.length;
    
    return {
      avg: this.measurements.reduce((a, b) => a + b, 0) / count,
      p50: sorted[Math.floor(count * 0.5)],
      p95: sorted[Math.floor(count * 0.95)],
      p99: sorted[Math.floor(count * 0.99)],
      min: sorted[0],
      max: sorted[count - 1],
      count,
    };
  }

  clear() {
    this.measurements = [];
  }
}

describe('LSP Performance Tests', () => {
  let lspSidecar: LSPSidecar;
  let lspStageBEnhancer: LSPStageBEnhancer;
  let lspStageCEnhancer: LSPStageCEnhancer;
  let intentRouter: IntentRouter;
  let workspaceConfig: WorkspaceConfig;
  let searchEngine: SearchEngine;

  const mockRepoPath = '/test/repo';
  const mockWorkspaceFiles = [
    '/test/repo/src/utils.ts',
    '/test/repo/src/services/user.ts',
    '/test/repo/src/models/user.ts',
    '/test/repo/src/controllers/auth.ts',
    '/test/repo/src/middleware/validation.ts',
    '/test/repo/package.json',
    '/test/repo/tsconfig.json',
  ];

  const createMockCandidate = (
    filePath: string,
    line: number,
    matchReasons: string[] = ['lexical'],
    similarity = 0.7
  ): Candidate => ({
    file_path: filePath,
    line,
    col: 0,
    content: `performance test content at ${filePath}:${line}`,
    symbol: 'perfSymbol',
    match_reasons: matchReasons,
    similarity,
    stage_b_score: similarity,
    stage_c_features: {
      is_definition: false,
      is_reference: false,
      has_documentation: false,
      complexity_score: 0.5,
      recency_score: 0.5,
    },
  });

  const createSearchContext = (query: string): SearchContext => ({
    trace_id: `perf_test_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    repo_sha: 'perf_test_repo_sha',
    query,
    mode: 'hybrid',
    k: 50,
    fuzzy_distance: 2,
    started_at: new Date(),
    stages: [],
  });

  const generateMockHints = (count: number): LSPHint[] => {
    return Array.from({ length: count }, (_, i) => ({
      symbol_id: `perf_symbol_${i}`,
      name: `perfSymbol${i}`,
      file_path: `/test/repo/src/file${i % 5}.ts`,
      line: 10 + (i % 20),
      col: 0,
      kind: ['function', 'class', 'interface', 'variable', 'method'][i % 5] as any,
      container: `container${i % 3}`,
      detail: `performance test symbol ${i}`,
      range: {
        start: { line: 10 + (i % 20), character: 0 },
        end: { line: 15 + (i % 20), character: 1 },
      },
    }));
  };

  beforeEach(async () => {
    vi.clearAllMocks();

    // Mock workspace configuration
    vi.mocked(require('fs').readFileSync).mockImplementation((path: string) => {
      if (path.includes('tsconfig.json')) {
        return JSON.stringify({
          compilerOptions: { baseUrl: './src' },
          include: ['src/**/*'],
        });
      }
      if (path.includes('package.json')) {
        return JSON.stringify({ name: 'perf-test-repo' });
      }
      return '{}';
    });

    // Initialize components
    workspaceConfig = new WorkspaceConfig(mockRepoPath);
    await workspaceConfig.parseWorkspaceConfig();

    lspSidecar = new LSPSidecar(workspaceConfig);
    lspStageBEnhancer = new LSPStageBEnhancer(lspSidecar);
    lspStageCEnhancer = new LSPStageCEnhancer(lspSidecar);
    intentRouter = new IntentRouter(lspSidecar);

    searchEngine = new SearchEngine(
      mockWorkspaceFiles,
      mockRepoPath,
      true,
      lspSidecar,
      lspStageBEnhancer,
      lspStageCEnhancer,
      intentRouter
    );

    // Mock LSP sidecar for performance tests
    vi.spyOn(lspSidecar, 'initializeLanguageServers').mockResolvedValue(undefined);
    vi.spyOn(lspSidecar, 'generateHints').mockResolvedValue(generateMockHints(100));
    vi.spyOn(lspSidecar, 'shutdown').mockResolvedValue(undefined);
  });

  afterEach(async () => {
    try {
      await searchEngine.shutdown();
    } catch (error) {
      // Ignore shutdown errors in performance tests
    }
    vi.restoreAllMocks();
  });

  describe('LSP Stage-B performance constraints', () => {
    test('maintains ≤+3ms p95 latency constraint under normal load', async () => {
      const tracker = new PerformanceTracker();
      const query = 'performance test function';
      const context = createSearchContext(query);
      
      const baseCandidates = Array.from({ length: 20 }, (_, i) =>
        createMockCandidate(`/test/repo/src/file${i}.ts`, 10 + i, ['lexical'], 0.7 - i * 0.01)
      );

      // Run multiple Stage-B enhancements to measure p95 latency
      const iterations = 100;
      for (let i = 0; i < iterations; i++) {
        await tracker.measure(async () => {
          const result = lspStageBEnhancer.enhanceStageB(
            query,
            createSearchContext(`${query} ${i}`),
            baseCandidates,
            50
          );
          return result;
        });
      }

      const stats = tracker.getStats();
      
      expect(stats.count).toBe(iterations);
      expect(stats.p95).toBeLessThanOrEqual(3.0); // ≤+3ms constraint
      expect(stats.avg).toBeLessThan(2.0); // Average should be well under limit
      expect(stats.min).toBeGreaterThan(0);

      console.log(`Stage-B Performance Stats: avg=${stats.avg.toFixed(2)}ms, p95=${stats.p95.toFixed(2)}ms, p99=${stats.p99.toFixed(2)}ms`);
    });

    test('maintains performance under high candidate volume', async () => {
      const tracker = new PerformanceTracker();
      const query = 'high volume test';
      
      // Test with large candidate sets
      const candidateCounts = [50, 100, 200, 500];
      
      for (const count of candidateCounts) {
        const candidates = Array.from({ length: count }, (_, i) =>
          createMockCandidate(`/test/repo/src/file${i % 10}.ts`, 10 + i, ['lexical'], 0.8 - i * 0.001)
        );

        const context = createSearchContext(`${query} ${count} candidates`);

        await tracker.measure(async () => {
          return lspStageBEnhancer.enhanceStageB(query, context, candidates, count);
        });
      }

      const stats = tracker.getStats();
      
      // Even with varying candidate counts, p95 should stay within bounds
      expect(stats.p95).toBeLessThanOrEqual(5.0); // Slightly relaxed for high volume
      expect(stats.max).toBeLessThan(10.0); // No single request should exceed 10ms
    });

    test('handles concurrent Stage-B requests efficiently', async () => {
      const concurrencyLevels = [5, 10, 20];
      
      for (const concurrency of concurrencyLevels) {
        const tracker = new PerformanceTracker();
        const candidates = Array.from({ length: 30 }, (_, i) =>
          createMockCandidate(`/test/repo/src/concurrent${i}.ts`, i + 5)
        );

        // Execute concurrent requests
        const promises = Array.from({ length: concurrency }, (_, i) => {
          const query = `concurrent test ${i}`;
          const context = createSearchContext(query);
          
          return tracker.measure(async () => {
            return lspStageBEnhancer.enhanceStageB(query, context, candidates, 30);
          });
        });

        await Promise.all(promises);
        const stats = tracker.getStats();

        expect(stats.count).toBe(concurrency);
        expect(stats.p95).toBeLessThanOrEqual(4.0); // Slightly relaxed for concurrent load
        expect(stats.avg).toBeLessThan(3.0);

        console.log(`Concurrency ${concurrency}: avg=${stats.avg.toFixed(2)}ms, p95=${stats.p95.toFixed(2)}ms`);
      }
    });

    test('maintains performance with different query complexities', async () => {
      const queryTypes = [
        { query: 'simple', complexity: 'simple' },
        { query: 'function calculateTotalWithValidation', complexity: 'medium' },
        { query: 'async function that validates user input and returns structured response', complexity: 'complex' },
        { query: 'interface User extends BaseEntity with validation', complexity: 'complex' },
      ];

      const tracker = new PerformanceTracker();
      const candidates = Array.from({ length: 25 }, (_, i) =>
        createMockCandidate(`/test/repo/src/query_test${i}.ts`, i + 1)
      );

      for (const { query, complexity } of queryTypes) {
        const iterations = complexity === 'complex' ? 20 : 30;
        
        for (let i = 0; i < iterations; i++) {
          const context = createSearchContext(`${query} ${i}`);
          
          await tracker.measure(async () => {
            return lspStageBEnhancer.enhanceStageB(query, context, candidates, 25);
          });
        }
      }

      const stats = tracker.getStats();
      
      expect(stats.p95).toBeLessThanOrEqual(3.5); // Slightly relaxed for complex queries
      expect(stats.avg).toBeLessThan(2.5);
    });
  });

  describe('LSP Stage-C performance validation', () => {
    test('validates bounded contribution computation performance', async () => {
      const tracker = new PerformanceTracker();
      
      const candidates = Array.from({ length: 50 }, (_, i) =>
        createMockCandidate(`/test/repo/src/stage_c_test${i}.ts`, i + 1, ['lsp_hint'], 0.8)
      );

      // Test Stage-C performance under normal conditions
      const iterations = 50;
      for (let i = 0; i < iterations; i++) {
        const query = `stage c performance test ${i}`;
        const context = createSearchContext(query);

        await tracker.measure(async () => {
          const result = lspStageCEnhancer.enhanceStageC(candidates, query, context);
          return result;
        });
      }

      const stats = tracker.getStats();
      
      expect(stats.count).toBe(iterations);
      expect(stats.p95).toBeLessThanOrEqual(50.0); // Stage-C can be slower than Stage-B
      expect(stats.avg).toBeLessThan(30.0);
      
      console.log(`Stage-C Performance Stats: avg=${stats.avg.toFixed(2)}ms, p95=${stats.p95.toFixed(2)}ms`);
    });

    test('validates feature extraction performance scales linearly', async () => {
      const candidateCounts = [10, 25, 50, 100];
      const results: { count: number; avgTime: number }[] = [];

      for (const count of candidateCounts) {
        const tracker = new PerformanceTracker();
        const candidates = Array.from({ length: count }, (_, i) =>
          createMockCandidate(`/test/repo/src/scaling${i}.ts`, i + 1, ['lsp_hint'], 0.8)
        );

        const iterations = 10;
        for (let i = 0; i < iterations; i++) {
          const query = `scaling test ${count} candidates`;
          const context = createSearchContext(query);

          await tracker.measure(async () => {
            return lspStageCEnhancer.enhanceStageC(candidates, query, context);
          });
        }

        const stats = tracker.getStats();
        results.push({ count, avgTime: stats.avg });
      }

      // Verify roughly linear scaling
      const firstResult = results[0];
      const lastResult = results[results.length - 1];
      const scalingFactor = lastResult.avgTime / firstResult.avgTime;
      const candidateRatio = lastResult.count / firstResult.count;

      // Scaling should be roughly proportional to candidate count
      expect(scalingFactor).toBeLessThan(candidateRatio * 1.5); // Allow 50% overhead
      expect(scalingFactor).toBeGreaterThan(candidateRatio * 0.5); // At least 50% of expected scaling
    });
  });

  describe('Intent Router performance validation', () => {
    test('validates intent classification performance', async () => {
      const tracker = new PerformanceTracker();
      
      const queryTypes = [
        'function calculateTotal',
        'refs UserService',
        'interface User',
        'class AuthController',
        'method authenticate',
        'variable CONFIG_PORT',
        'if (user && user.active)',
        'try { await database.connect() }',
        'function that validates user input',
        'how to implement authentication',
      ];

      // Mock baseline search handler
      const mockSearchHandler = vi.fn().mockResolvedValue([
        createMockCandidate('/test/repo/src/mock.ts', 10)
      ]);

      const iterations = 20;
      for (const queryType of queryTypes) {
        for (let i = 0; i < iterations; i++) {
          const query = `${queryType} ${i}`;
          const context = createSearchContext(query);

          await tracker.measure(async () => {
            return intentRouter.routeQuery(query, context, undefined, mockSearchHandler);
          });
        }
      }

      const stats = tracker.getStats();
      
      expect(stats.count).toBe(queryTypes.length * iterations);
      expect(stats.p95).toBeLessThanOrEqual(100.0); // Intent routing can take up to 100ms
      expect(stats.avg).toBeLessThan(50.0);
      
      console.log(`Intent Router Performance: avg=${stats.avg.toFixed(2)}ms, p95=${stats.p95.toFixed(2)}ms`);
    });

    test('validates intent router scales with query complexity', async () => {
      const complexityLevels = [
        { queries: ['simple', 'test', 'user'], label: 'simple' },
        { queries: ['function calculateTotal', 'class UserService', 'interface Model'], label: 'medium' },
        { queries: [
          'function that validates user input and returns structured error response',
          'complex async method that processes payments with error handling',
          'interface that extends multiple base classes with generic constraints'
        ], label: 'complex' },
      ];

      const mockSearchHandler = vi.fn().mockResolvedValue([
        createMockCandidate('/test/repo/src/complexity.ts', 15)
      ]);

      for (const { queries, label } of complexityLevels) {
        const tracker = new PerformanceTracker();
        
        for (const query of queries) {
          const iterations = 15;
          for (let i = 0; i < iterations; i++) {
            const context = createSearchContext(`${query} ${i}`);

            await tracker.measure(async () => {
              return intentRouter.routeQuery(query, context, undefined, mockSearchHandler);
            });
          }
        }

        const stats = tracker.getStats();
        console.log(`Intent Router ${label}: avg=${stats.avg.toFixed(2)}ms, p95=${stats.p95.toFixed(2)}ms`);
        
        // Complex queries can take longer but should still be reasonable
        if (label === 'complex') {
          expect(stats.p95).toBeLessThanOrEqual(150.0);
        } else {
          expect(stats.p95).toBeLessThanOrEqual(75.0);
        }
      }
    });
  });

  describe('end-to-end performance validation', () => {
    test('validates complete search pipeline performance', async () => {
      await searchEngine.initializeLSP();
      
      const tracker = new PerformanceTracker();
      
      const testQueries = [
        'function calculateTotal',
        'refs UserService',
        'interface User',
        'class AuthController',
        'method authenticate',
      ];

      // Mock LSP components for consistent performance testing
      vi.spyOn(lspStageBEnhancer, 'enhanceStageB').mockImplementation(async () => {
        await new Promise(resolve => setTimeout(resolve, 1)); // 1ms processing time
        return {
          candidates: [createMockCandidate('/test/repo/src/enhanced.ts', 10, ['lsp_enhanced'], 0.9)],
          metrics: { enhanced_count: 1, processing_time_ms: 1 },
        };
      });

      vi.spyOn(lspStageCEnhancer, 'enhanceStageC').mockImplementation(async () => {
        await new Promise(resolve => setTimeout(resolve, 2)); // 2ms processing time
        return {
          enhanced_candidates: [createMockCandidate('/test/repo/src/enhanced.ts', 10, ['stage_c'], 0.95)],
          feature_stats: { total_features: 8, active_features: 6 },
          bounded_contribution: 0.35,
        };
      });

      vi.spyOn(intentRouter, 'routeQuery').mockImplementation(async () => {
        await new Promise(resolve => setTimeout(resolve, 5)); // 5ms processing time
        return {
          primary_candidates: [createMockCandidate('/test/repo/src/routed.ts', 10, ['intent_routed'], 0.85)],
          intent_classification: { intent: 'def' as QueryIntent, confidence: 0.9 },
          routing_stats: { router_time_ms: 5 },
        };
      });

      const iterations = 25;
      for (const query of testQueries) {
        for (let i = 0; i < iterations; i++) {
          const context = createSearchContext(`${query} ${i}`);

          await tracker.measure(async () => {
            return searchEngine.search(query, context);
          });
        }
      }

      const stats = tracker.getStats();
      
      expect(stats.count).toBe(testQueries.length * iterations);
      expect(stats.p95).toBeLessThanOrEqual(500.0); // End-to-end should be under 500ms
      expect(stats.avg).toBeLessThan(100.0); // Average should be much faster
      
      console.log(`End-to-End Performance: avg=${stats.avg.toFixed(2)}ms, p95=${stats.p95.toFixed(2)}ms, p99=${stats.p99.toFixed(2)}ms`);
    });

    test('validates memory usage stays within bounds during sustained load', async () => {
      const initialMemory = process.memoryUsage();
      
      // Run sustained load test
      const iterations = 100;
      const queries = [
        'sustained load test',
        'memory usage validation',
        'performance boundary check',
        'resource utilization test',
      ];

      for (let i = 0; i < iterations; i++) {
        const query = queries[i % queries.length];
        const context = createSearchContext(`${query} ${i}`);
        
        await searchEngine.search(query, context);
        
        // Periodic memory checks
        if (i % 25 === 0) {
          const currentMemory = process.memoryUsage();
          const heapIncrease = (currentMemory.heapUsed - initialMemory.heapUsed) / 1024 / 1024;
          
          // Memory growth should be reasonable
          expect(heapIncrease).toBeLessThan(50); // Less than 50MB increase
        }
      }

      const finalMemory = process.memoryUsage();
      const totalHeapIncrease = (finalMemory.heapUsed - initialMemory.heapUsed) / 1024 / 1024;
      
      console.log(`Memory usage increase after ${iterations} searches: ${totalHeapIncrease.toFixed(2)}MB`);
      expect(totalHeapIncrease).toBeLessThan(100); // Total increase should be reasonable
    });

    test('validates throughput under concurrent load', async () => {
      const concurrencyLevels = [1, 5, 10, 20];
      
      for (const concurrency of concurrencyLevels) {
        const startTime = performance.now();
        const requestsPerWorker = 10;
        
        const promises = Array.from({ length: concurrency }, async (_, workerId) => {
          for (let i = 0; i < requestsPerWorker; i++) {
            const query = `throughput test worker${workerId} request${i}`;
            const context = createSearchContext(query);
            await searchEngine.search(query, context);
          }
        });

        await Promise.all(promises);
        
        const endTime = performance.now();
        const totalTime = endTime - startTime;
        const totalRequests = concurrency * requestsPerWorker;
        const throughput = totalRequests / (totalTime / 1000); // requests per second

        console.log(`Concurrency ${concurrency}: ${throughput.toFixed(2)} req/sec`);
        
        expect(throughput).toBeGreaterThan(1); // At least 1 request per second
        expect(totalTime / totalRequests).toBeLessThan(1000); // Average under 1 second per request
      }
    });
  });

  describe('performance regression detection', () => {
    test('establishes performance baseline for regression detection', async () => {
      // This test establishes baseline metrics that can be used to detect
      // performance regressions in future runs
      
      const baselineMetrics = {
        stageBP95: 3.0, // milliseconds
        stageCAvg: 30.0, // milliseconds
        intentRouterP95: 100.0, // milliseconds
        endToEndP95: 500.0, // milliseconds
      };

      // Run quick performance validation
      const tracker = new PerformanceTracker();
      const candidates = Array.from({ length: 20 }, (_, i) =>
        createMockCandidate(`/test/repo/src/baseline${i}.ts`, i)
      );

      for (let i = 0; i < 20; i++) {
        const query = `baseline test ${i}`;
        const context = createSearchContext(query);

        await tracker.measure(async () => {
          return lspStageBEnhancer.enhanceStageB(query, context, candidates, 20);
        });
      }

      const stats = tracker.getStats();
      
      // Validate against baseline
      expect(stats.p95).toBeLessThanOrEqual(baselineMetrics.stageBP95);
      
      console.log('Performance Baseline Validation:');
      console.log(`Stage-B p95: ${stats.p95.toFixed(2)}ms (baseline: ≤${baselineMetrics.stageBP95}ms)`);
      console.log(`Stage-B avg: ${stats.avg.toFixed(2)}ms`);
    });

    test('validates performance does not degrade with increased hint count', async () => {
      const hintCounts = [50, 100, 200, 500];
      const performanceResults: { hintCount: number; p95: number }[] = [];

      for (const hintCount of hintCounts) {
        // Mock LSP sidecar with variable hint counts
        vi.spyOn(lspSidecar, 'generateHints')
          .mockResolvedValue(generateMockHints(hintCount));

        const tracker = new PerformanceTracker();
        const candidates = Array.from({ length: 25 }, (_, i) =>
          createMockCandidate(`/test/repo/src/hints${i}.ts`, i)
        );

        const iterations = 15;
        for (let i = 0; i < iterations; i++) {
          const query = `hint scaling test ${hintCount} hints`;
          const context = createSearchContext(query);

          await tracker.measure(async () => {
            return lspStageBEnhancer.enhanceStageB(query, context, candidates, 25);
          });
        }

        const stats = tracker.getStats();
        performanceResults.push({ hintCount, p95: stats.p95 });
      }

      // Validate that performance doesn't degrade significantly with more hints
      const firstResult = performanceResults[0];
      const lastResult = performanceResults[performanceResults.length - 1];
      
      const performanceDegradation = lastResult.p95 / firstResult.p95;
      
      // Performance should not degrade by more than 3x with 10x more hints
      expect(performanceDegradation).toBeLessThan(3.0);
      
      console.log('Hint Count Scaling Results:');
      performanceResults.forEach(({ hintCount, p95 }) => {
        console.log(`${hintCount} hints: ${p95.toFixed(2)}ms p95`);
      });
    });
  });
});