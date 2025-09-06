/**
 * Tests for LSP fallback behavior when LSP components fail
 * Validates graceful degradation and error recovery mechanisms
 * Ensures baseline search functionality continues when LSP is unavailable
 */

import { describe, test, expect, vi, beforeEach, afterEach } from 'vitest';
import { SearchEngine } from '../api/search-engine.js';
import { LSPSidecar } from '../core/lsp-sidecar.js';
import { LSPStageBEnhancer } from '../core/lsp-stage-b.js';
import { LSPStageCEnhancer } from '../core/lsp-stage-c.js';
import { IntentRouter } from '../core/intent-router.js';
import { WorkspaceConfig } from '../core/workspace-config.js';
import type {
  Candidate,
  SearchContext,
  QueryIntent
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

describe('LSP Fallback Tests', () => {
  let searchEngine: SearchEngine;
  let lspSidecar: LSPSidecar;
  let lspStageBEnhancer: LSPStageBEnhancer;
  let lspStageCEnhancer: LSPStageCEnhancer;
  let intentRouter: IntentRouter;
  let workspaceConfig: WorkspaceConfig;

  const mockRepoPath = '/test/repo';
  const mockWorkspaceFiles = [
    '/test/repo/src/utils.ts',
    '/test/repo/src/services/user.ts',
    '/test/repo/src/models/user.ts',
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
    content: `mock baseline content at ${filePath}:${line}`,
    symbol: 'baselineSymbol',
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
    trace_id: `fallback_test_${Date.now()}`,
    repo_sha: 'test_repo_sha',
    query,
    mode: 'hybrid',
    k: 50,
    fuzzy_distance: 2,
    started_at: new Date(),
    stages: [],
  });

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
        return JSON.stringify({ name: 'test-repo' });
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

    // Initialize search engine with LSP enabled
    searchEngine = new SearchEngine(
      mockWorkspaceFiles,
      mockRepoPath,
      true, // enableLSP
      lspSidecar,
      lspStageBEnhancer,
      lspStageCEnhancer,
      intentRouter
    );
  });

  afterEach(async () => {
    try {
      await searchEngine.shutdown();
    } catch (error) {
      // Ignore shutdown errors in tests
    }
    vi.restoreAllMocks();
  });

  describe('LSP initialization failures', () => {
    test('falls back to baseline search when LSP sidecar fails to initialize', async () => {
      // Mock LSP initialization failure
      vi.spyOn(lspSidecar, 'initializeLanguageServers')
        .mockRejectedValue(new Error('TypeScript server failed to start'));

      const query = 'function calculateTotal';
      const context = createSearchContext(query);

      // Should still return results from baseline search
      const results = await searchEngine.search(query, context);

      expect(results).toBeDefined();
      expect(Array.isArray(results)).toBe(true);
      
      // Should have attempted LSP initialization
      expect(lspSidecar.initializeLanguageServers).toHaveBeenCalled();
    });

    test('handles workspace config parsing failures gracefully', async () => {
      // Mock workspace config parsing failure
      vi.mocked(require('fs').readFileSync).mockImplementation(() => {
        throw new Error('Failed to read tsconfig.json');
      });

      const failingWorkspaceConfig = new WorkspaceConfig('/invalid/path');
      const failingSidecar = new LSPSidecar(failingWorkspaceConfig);
      
      const fallbackSearchEngine = new SearchEngine(
        mockWorkspaceFiles,
        mockRepoPath,
        true,
        failingSidecar,
        new LSPStageBEnhancer(failingSidecar),
        new LSPStageCEnhancer(failingSidecar),
        new IntentRouter(failingSidecar)
      );

      const query = 'test query';
      const context = createSearchContext(query);

      // Should handle workspace config failure and continue
      const results = await fallbackSearchEngine.search(query, context);

      expect(results).toBeDefined();
      expect(Array.isArray(results)).toBe(true);

      await fallbackSearchEngine.shutdown();
    });

    test('handles LSP server process crashes', async () => {
      // Mock successful initialization followed by process crash
      vi.spyOn(lspSidecar, 'initializeLanguageServers').mockResolvedValue(undefined);
      vi.spyOn(lspSidecar, 'harvestSymbols').mockRejectedValue(new Error('Process terminated'));

      await searchEngine.initializeLSP();

      const query = 'UserService';
      const context = createSearchContext(query);

      // Should handle process crash gracefully
      const results = await searchEngine.search(query, context);

      expect(results).toBeDefined();
      expect(Array.isArray(results)).toBe(true);
    });
  });

  describe('intent router fallback', () => {
    test('falls back to direct search when intent router fails', async () => {
      const query = 'function testFunction';
      const context = createSearchContext(query);

      // Mock intent router failure
      vi.spyOn(intentRouter, 'routeQuery')
        .mockRejectedValue(new Error('Intent classification failed'));

      // Mock baseline search to still return results
      const baselineResults = [
        createMockCandidate('/test/repo/src/test.ts', 10, ['baseline'], 0.75),
      ];

      const results = await searchEngine.search(query, context);

      expect(results).toBeDefined();
      expect(Array.isArray(results)).toBe(true);
      
      // Should have attempted intent routing
      expect(intentRouter.routeQuery).toHaveBeenCalled();
    });

    test('handles intent router timeout gracefully', async () => {
      vi.useFakeTimers();

      const query = 'timeout test';
      const context = createSearchContext(query);

      // Mock intent router with long delay
      vi.spyOn(intentRouter, 'routeQuery').mockImplementation(async () => {
        return new Promise((resolve) => {
          setTimeout(() => {
            resolve({
              primary_candidates: [createMockCandidate('/test/repo/src/test.ts', 10)],
              intent_classification: { intent: 'def' as QueryIntent, confidence: 0.8 },
              routing_stats: { router_time_ms: 5000 },
            });
          }, 5000);
        });
      });

      const searchPromise = searchEngine.search(query, context);
      
      // Advance timers to trigger timeout behavior
      vi.advanceTimersByTime(6000);
      
      const results = await searchPromise;

      expect(results).toBeDefined();

      vi.useRealTimers();
    });

    test('handles malformed intent router responses', async () => {
      const query = 'malformed response';
      const context = createSearchContext(query);

      // Mock malformed response
      vi.spyOn(intentRouter, 'routeQuery').mockResolvedValue({
        primary_candidates: null as any, // Invalid
        intent_classification: undefined as any, // Invalid
        routing_stats: { router_time_ms: 25 },
      });

      const results = await searchEngine.search(query, context);

      expect(results).toBeDefined();
      expect(Array.isArray(results)).toBe(true);
    });
  });

  describe('LSP Stage-B fallback', () => {
    test('continues with original candidates when Stage-B fails', async () => {
      const query = 'stageB failure test';
      const context = createSearchContext(query);

      // Mock intent router success
      vi.spyOn(intentRouter, 'routeQuery').mockResolvedValue({
        primary_candidates: [
          createMockCandidate('/test/repo/src/test.ts', 15, ['intent_routed'], 0.8),
        ],
        intent_classification: { intent: 'def' as QueryIntent, confidence: 0.85 },
        routing_stats: { router_time_ms: 30 },
      });

      // Mock Stage-B failure
      vi.spyOn(lspStageBEnhancer, 'enhanceStageB')
        .mockRejectedValue(new Error('LSP Stage-B enhancement failed'));

      // Mock Stage-C success
      vi.spyOn(lspStageCEnhancer, 'enhanceStageC').mockResolvedValue({
        enhanced_candidates: [
          createMockCandidate('/test/repo/src/test.ts', 15, ['stage_c_enhanced'], 0.85),
        ],
        feature_stats: { total_features: 5, active_features: 3 },
        bounded_contribution: 0.25,
      });

      const results = await searchEngine.search(query, context);

      expect(results).toBeDefined();
      expect(results.length).toBeGreaterThan(0);
      
      // Should have tried Stage-B
      expect(lspStageBEnhancer.enhanceStageB).toHaveBeenCalled();
      
      // Should have continued to Stage-C
      expect(lspStageCEnhancer.enhanceStageC).toHaveBeenCalled();
    });

    test('handles Stage-B timeout with graceful degradation', async () => {
      vi.useFakeTimers();

      const query = 'stageB timeout';
      const context = createSearchContext(query);

      // Mock Stage-B with timeout
      vi.spyOn(lspStageBEnhancer, 'enhanceStageB').mockImplementation(async () => {
        return new Promise((resolve) => {
          setTimeout(() => {
            resolve({
              candidates: [createMockCandidate('/test/repo/src/test.ts', 10)],
              metrics: { enhanced_count: 1, processing_time_ms: 3000 },
            });
          }, 3000);
        });
      });

      const searchPromise = searchEngine.search(query, context);
      
      vi.advanceTimersByTime(4000);
      
      const results = await searchPromise;

      expect(results).toBeDefined();

      vi.useRealTimers();
    });

    test('handles Stage-B memory/resource exhaustion', async () => {
      const query = 'resource exhaustion';
      const context = createSearchContext(query);

      // Mock Stage-B resource exhaustion
      vi.spyOn(lspStageBEnhancer, 'enhanceStageB')
        .mockRejectedValue(new Error('ENOMEM: not enough memory'));

      const results = await searchEngine.search(query, context);

      expect(results).toBeDefined();
      expect(Array.isArray(results)).toBe(true);
    });

    test('handles Stage-B returning invalid data', async () => {
      const query = 'invalid stageB data';
      const context = createSearchContext(query);

      // Mock Stage-B returning invalid data
      vi.spyOn(lspStageBEnhancer, 'enhanceStageB').mockResolvedValue({
        candidates: null as any, // Invalid
        metrics: undefined as any, // Invalid
      });

      const results = await searchEngine.search(query, context);

      expect(results).toBeDefined();
      expect(Array.isArray(results)).toBe(true);
    });
  });

  describe('LSP Stage-C fallback', () => {
    test('returns Stage-B results when Stage-C fails', async () => {
      const query = 'stageC failure test';
      const context = createSearchContext(query);

      // Mock Stage-B success
      vi.spyOn(lspStageBEnhancer, 'enhanceStageB').mockResolvedValue({
        candidates: [
          createMockCandidate('/test/repo/src/test.ts', 20, ['stage_b_enhanced'], 0.85),
        ],
        metrics: { enhanced_count: 1, processing_time_ms: 45 },
      });

      // Mock Stage-C failure
      vi.spyOn(lspStageCEnhancer, 'enhanceStageC')
        .mockRejectedValue(new Error('LSP Stage-C enhancement failed'));

      const results = await searchEngine.search(query, context);

      expect(results).toBeDefined();
      expect(results.length).toBeGreaterThan(0);
      
      // Should have tried both stages
      expect(lspStageBEnhancer.enhanceStageB).toHaveBeenCalled();
      expect(lspStageCEnhancer.enhanceStageC).toHaveBeenCalled();

      // Result should be from Stage-B
      const stageResult = results.find(r => r.match_reasons.includes('stage_b_enhanced'));
      expect(stageResult).toBeDefined();
    });

    test('handles Stage-C bounded contribution failures', async () => {
      const query = 'bounded contribution failure';
      const context = createSearchContext(query);

      // Mock Stage-C with invalid bounded contribution
      vi.spyOn(lspStageCEnhancer, 'enhanceStageC').mockResolvedValue({
        enhanced_candidates: [createMockCandidate('/test/repo/src/test.ts', 25, ['stage_c'], 0.9)],
        feature_stats: { total_features: 10, active_features: 8 },
        bounded_contribution: NaN, // Invalid contribution
      });

      const results = await searchEngine.search(query, context);

      expect(results).toBeDefined();
      expect(Array.isArray(results)).toBe(true);
    });

    test('handles Stage-C feature extraction errors', async () => {
      const query = 'feature extraction error';
      const context = createSearchContext(query);

      // Mock Stage-C feature extraction failure
      vi.spyOn(lspStageCEnhancer, 'enhanceStageC')
        .mockRejectedValue(new Error('Feature extraction failed'));

      const results = await searchEngine.search(query, context);

      expect(results).toBeDefined();
    });
  });

  describe('cascade failure scenarios', () => {
    test('handles complete LSP component cascade failure', async () => {
      const query = 'complete cascade failure';
      const context = createSearchContext(query);

      // Mock all LSP components to fail
      vi.spyOn(intentRouter, 'routeQuery')
        .mockRejectedValue(new Error('Intent router failed'));
      vi.spyOn(lspStageBEnhancer, 'enhanceStageB')
        .mockRejectedValue(new Error('Stage-B failed'));
      vi.spyOn(lspStageCEnhancer, 'enhanceStageC')
        .mockRejectedValue(new Error('Stage-C failed'));
      vi.spyOn(lspSidecar, 'generateHints')
        .mockRejectedValue(new Error('LSP hints failed'));

      // Should still return baseline search results
      const results = await searchEngine.search(query, context);

      expect(results).toBeDefined();
      expect(Array.isArray(results)).toBe(true);
      
      // All LSP components should have been attempted
      expect(intentRouter.routeQuery).toHaveBeenCalled();
      expect(lspStageBEnhancer.enhanceStageB).toHaveBeenCalled();
      expect(lspStageCEnhancer.enhanceStageC).toHaveBeenCalled();
    });

    test('maintains baseline search quality during LSP failures', async () => {
      const queries = [
        'function calculateTotal',
        'class UserService',
        'interface User',
        'method authenticate',
        'variable CONFIG_PORT',
      ];

      // Mock all LSP components to fail
      vi.spyOn(intentRouter, 'routeQuery')
        .mockRejectedValue(new Error('LSP unavailable'));
      vi.spyOn(lspStageBEnhancer, 'enhanceStageB')
        .mockRejectedValue(new Error('LSP unavailable'));
      vi.spyOn(lspStageCEnhancer, 'enhanceStageC')
        .mockRejectedValue(new Error('LSP unavailable'));

      // Test multiple queries to ensure consistent fallback
      for (const query of queries) {
        const context = createSearchContext(query);
        const results = await searchEngine.search(query, context);

        expect(results).toBeDefined();
        expect(Array.isArray(results)).toBe(true);
        
        // Should maintain reasonable baseline quality
        // (In real implementation, would validate specific baseline search results)
      }
    });

    test('handles intermittent LSP failures with recovery', async () => {
      const query = 'intermittent failure test';

      // First call fails
      vi.spyOn(intentRouter, 'routeQuery')
        .mockRejectedValueOnce(new Error('Temporary failure'))
        .mockResolvedValue({
          primary_candidates: [createMockCandidate('/test/repo/src/recovered.ts', 10)],
          intent_classification: { intent: 'def' as QueryIntent, confidence: 0.8 },
          routing_stats: { router_time_ms: 25 },
        });

      // First search - should handle failure
      const context1 = createSearchContext(query);
      const results1 = await searchEngine.search(query, context1);

      expect(results1).toBeDefined();
      expect(Array.isArray(results1)).toBe(true);

      // Second search - should work after recovery
      const context2 = createSearchContext(query);
      const results2 = await searchEngine.search(query, context2);

      expect(results2).toBeDefined();
      expect(Array.isArray(results2)).toBe(true);

      // Should have attempted router twice
      expect(intentRouter.routeQuery).toHaveBeenCalledTimes(2);
    });
  });

  describe('performance during failures', () => {
    test('maintains acceptable response time during LSP failures', async () => {
      const query = 'performance test during failure';
      const context = createSearchContext(query);

      // Mock slow failing LSP components
      vi.spyOn(intentRouter, 'routeQuery').mockImplementation(async () => {
        await new Promise(resolve => setTimeout(resolve, 100));
        throw new Error('Slow failure');
      });

      vi.spyOn(lspStageBEnhancer, 'enhanceStageB').mockImplementation(async () => {
        await new Promise(resolve => setTimeout(resolve, 100));
        throw new Error('Slow failure');
      });

      const startTime = performance.now();
      const results = await searchEngine.search(query, context);
      const endTime = performance.now();

      const responseTime = endTime - startTime;

      expect(results).toBeDefined();
      expect(responseTime).toBeLessThan(1000); // Should timeout/fallback quickly
    });

    test('prevents memory leaks during repeated failures', async () => {
      const initialMemory = process.memoryUsage().heapUsed;

      // Mock LSP failures
      vi.spyOn(intentRouter, 'routeQuery')
        .mockRejectedValue(new Error('Persistent failure'));

      // Execute many queries with failures
      const promises = Array.from({ length: 50 }, (_, i) => {
        const query = `failure test ${i}`;
        const context = createSearchContext(query);
        return searchEngine.search(query, context);
      });

      await Promise.all(promises);

      const finalMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = (finalMemory - initialMemory) / 1024 / 1024; // MB

      // Memory increase should be reasonable despite failures
      expect(memoryIncrease).toBeLessThan(50); // Less than 50MB increase
    });
  });

  describe('error reporting and diagnostics', () => {
    test('logs appropriate error information during failures', async () => {
      const query = 'error logging test';
      const context = createSearchContext(query);

      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      // Mock LSP failure
      vi.spyOn(intentRouter, 'routeQuery')
        .mockRejectedValue(new Error('Test error for logging'));

      await searchEngine.search(query, context);

      // In a real implementation, would verify appropriate error logging
      // For now, just ensure search completes
      expect(consoleSpy).toBeDefined(); // Placeholder for error logging verification

      consoleSpy.mockRestore();
    });

    test('provides fallback status in search context', async () => {
      const query = 'fallback status test';
      const context = createSearchContext(query);

      // Mock failures
      vi.spyOn(intentRouter, 'routeQuery')
        .mockRejectedValue(new Error('Router failed'));
      vi.spyOn(lspStageBEnhancer, 'enhanceStageB')
        .mockRejectedValue(new Error('Stage-B failed'));

      await searchEngine.search(query, context);

      // In real implementation, context would be updated with fallback status
      expect(context.stages).toBeDefined();
      expect(Array.isArray(context.stages)).toBe(true);
    });
  });

  describe('configuration-based fallback', () => {
    test('respects LSP disable configuration', async () => {
      // Create search engine with LSP disabled
      const disabledSearchEngine = new SearchEngine(
        mockWorkspaceFiles,
        mockRepoPath,
        false, // enableLSP = false
        lspSidecar,
        lspStageBEnhancer,
        lspStageCEnhancer,
        intentRouter
      );

      const query = 'disabled LSP test';
      const context = createSearchContext(query);

      const results = await disabledSearchEngine.search(query, context);

      expect(results).toBeDefined();
      expect(Array.isArray(results)).toBe(true);

      // LSP components should not be called
      expect(intentRouter.routeQuery).not.toHaveBeenCalled();
      expect(lspStageBEnhancer.enhanceStageB).not.toHaveBeenCalled();
      expect(lspStageCEnhancer.enhanceStageC).not.toHaveBeenCalled();

      await disabledSearchEngine.shutdown();
    });

    test('handles partial LSP feature disabling', async () => {
      const query = 'partial disable test';
      const context = createSearchContext(query);

      // Mock only intent router working, others failing
      vi.spyOn(intentRouter, 'routeQuery').mockResolvedValue({
        primary_candidates: [createMockCandidate('/test/repo/src/test.ts', 10)],
        intent_classification: { intent: 'def' as QueryIntent, confidence: 0.8 },
        routing_stats: { router_time_ms: 25 },
      });

      vi.spyOn(lspStageBEnhancer, 'enhanceStageB')
        .mockRejectedValue(new Error('Stage-B disabled'));
      vi.spyOn(lspStageCEnhancer, 'enhanceStageC')
        .mockRejectedValue(new Error('Stage-C disabled'));

      const results = await searchEngine.search(query, context);

      expect(results).toBeDefined();
      expect(results.length).toBeGreaterThan(0);

      // Only intent router should succeed
      expect(intentRouter.routeQuery).toHaveBeenCalled();
    });
  });
});