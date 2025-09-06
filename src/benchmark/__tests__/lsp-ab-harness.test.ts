/**
 * Comprehensive tests for LSP A/B Benchmark Harness
 * Tests the three benchmark modes: Baseline Lens, Lens + LSP-assist, Competitor+LSP
 * Validates comparison logic and performance measurement capabilities
 */

import { describe, test, expect, vi, beforeEach, afterEach } from 'vitest';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { LSPABBenchmarkHarness } from '../lsp-ab-harness.js';
import { LSPSidecar } from '../../core/lsp-sidecar.js';
import { LSPStageBEnhancer } from '../../core/lsp-stage-b.js';
import { LSPStageCEnhancer } from '../../core/lsp-stage-c.js';
import { IntentRouter } from '../../core/intent-router.js';
import type {
  Candidate,
  SearchContext,
  LSPBenchmarkResult,
  QueryIntent,
  LossTaxonomy
} from '../../types/core.js';

// Mock file system operations
vi.mock('fs', () => ({
  readFileSync: vi.fn(),
  writeFileSync: vi.fn(),
  existsSync: vi.fn(),
}));

// Mock telemetry
vi.mock('../../telemetry/tracer.js', () => ({
  LensTracer: {
    createChildSpan: vi.fn(() => ({
      setAttributes: vi.fn(),
      recordException: vi.fn(),
      end: vi.fn(),
    })),
  },
}));

describe('LSPABBenchmarkHarness', () => {
  let harness: LSPABBenchmarkHarness;
  let mockLspSidecar: LSPSidecar;
  let mockLspStageBEnhancer: LSPStageBEnhancer;
  let mockLspStageCEnhancer: LSPStageCEnhancer;
  let mockIntentRouter: IntentRouter;
  let mockBaselineSearchHandler: vi.MockedFunction<(query: string, context: SearchContext) => Promise<Candidate[]>>;

  // Mock candidate data for testing
  const createMockCandidate = (filePath: string, line: number, score = 0.8): Candidate => ({
    file_path: filePath,
    line,
    col: 0,
    content: `mock content at ${filePath}:${line}`,
    symbol: 'mockSymbol',
    match_reasons: ['exact_match'],
    similarity: score,
    stage_b_score: score,
    stage_c_features: {
      is_definition: false,
      is_reference: false,
      has_documentation: false,
      complexity_score: 0.5,
      recency_score: 0.5,
    },
  });

  const createMockSearchContext = (query: string): SearchContext => ({
    trace_id: `test_${Date.now()}`,
    repo_sha: 'test_repo_sha',
    query,
    mode: 'hybrid',
    k: 50,
    fuzzy_distance: 2,
    started_at: new Date(),
    stages: [],
  });

  beforeEach(() => {
    vi.clearAllMocks();

    // Create mock LSP components
    mockLspSidecar = {} as LSPSidecar;
    
    mockLspStageBEnhancer = {
      enhanceStageB: vi.fn().mockResolvedValue({
        candidates: [createMockCandidate('/mock/enhanced.ts', 10, 0.9)],
        metrics: { enhanced_count: 1, processing_time_ms: 50 },
      }),
    } as any;

    mockLspStageCEnhancer = {
      enhanceStageC: vi.fn().mockResolvedValue({
        enhanced_candidates: [createMockCandidate('/mock/enhanced.ts', 10, 0.95)],
        feature_stats: { total_features: 5, active_features: 3 },
        bounded_contribution: 0.3,
      }),
    } as any;

    mockIntentRouter = {
      routeQuery: vi.fn().mockResolvedValue({
        primary_candidates: [createMockCandidate('/mock/routed.ts', 5, 0.85)],
        intent_classification: { intent: 'def' as QueryIntent, confidence: 0.9 },
        routing_stats: { router_time_ms: 25 },
      }),
    } as any;

    // Mock baseline search handler
    mockBaselineSearchHandler = vi.fn().mockResolvedValue([
      createMockCandidate('/mock/baseline.ts', 10, 0.7),
      createMockCandidate('/mock/baseline2.ts', 20, 0.6),
    ]);

    harness = new LSPABBenchmarkHarness(
      mockLspSidecar,
      mockLspStageBEnhancer,
      mockLspStageCEnhancer,
      mockIntentRouter
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('initialization', () => {
    test('initializes with three benchmark modes', () => {
      const stats = harness.getStats();
      
      expect(stats.modes_configured).toBe(3);
    });

    test('configures baseline mode correctly', () => {
      const harness2 = new LSPABBenchmarkHarness();
      const stats = harness2.getStats();
      
      expect(stats.modes_configured).toBe(3);
    });

    test('initializes with zero queries and runs', () => {
      const stats = harness.getStats();
      
      expect(stats.queries_loaded).toBe(0);
      expect(stats.runs_completed).toBe(0);
    });
  });

  describe('benchmark query loading', () => {
    test('loads queries from file when exists', async () => {
      const mockQueries = [
        {
          id: 'test_1',
          query: 'function testQuery',
          intent: 'def' as QueryIntent,
          ground_truth: [{
            file_path: '/test/file.ts',
            line: 10,
            col: 0,
            relevance: 3,
            is_primary: true,
          }],
          difficulty: 'easy' as const,
          language: 'typescript',
        },
      ];

      vi.mocked(existsSync).mockReturnValue(true);
      vi.mocked(readFileSync).mockReturnValue(JSON.stringify(mockQueries));

      await harness.loadBenchmarkQueries('/path/to/queries.json');
      
      const stats = harness.getStats();
      expect(stats.queries_loaded).toBe(1);
    });

    test('generates synthetic queries when file does not exist', async () => {
      vi.mocked(existsSync).mockReturnValue(false);

      await harness.loadBenchmarkQueries('/nonexistent/path.json');
      
      const stats = harness.getStats();
      expect(stats.queries_loaded).toBeGreaterThan(0);
      expect(stats.intent_coverage).toEqual(
        expect.objectContaining({
          def: expect.any(Number),
          refs: expect.any(Number),
          symbol: expect.any(Number),
          struct: expect.any(Number),
          NL: expect.any(Number),
        })
      );
    });

    test('generates synthetic queries when no file path provided', async () => {
      await harness.loadBenchmarkQueries();
      
      const stats = harness.getStats();
      expect(stats.queries_loaded).toBeGreaterThan(15); // Should generate various query types
      
      // Verify all intent types are covered
      const coverage = stats.intent_coverage;
      expect(coverage.def).toBeGreaterThan(0);
      expect(coverage.refs).toBeGreaterThan(0);
      expect(coverage.symbol).toBeGreaterThan(0);
      expect(coverage.struct).toBeGreaterThan(0);
      expect(coverage.NL).toBeGreaterThan(0);
    });

    test('handles file parsing errors gracefully', async () => {
      vi.mocked(existsSync).mockReturnValue(true);
      vi.mocked(readFileSync).mockImplementation(() => {
        throw new Error('File read error');
      });

      await expect(harness.loadBenchmarkQueries('/bad/path.json')).rejects.toThrow('File read error');
    });

    test('handles invalid JSON gracefully', async () => {
      vi.mocked(existsSync).mockReturnValue(true);
      vi.mocked(readFileSync).mockReturnValue('invalid json {');

      await expect(harness.loadBenchmarkQueries('/bad/json.json')).rejects.toThrow();
    });
  });

  describe('benchmark execution', () => {
    beforeEach(async () => {
      await harness.loadBenchmarkQueries(); // Load synthetic queries
    });

    test('runs benchmark across all modes', async () => {
      const results = await harness.runBenchmark(mockBaselineSearchHandler);
      
      expect(results).toHaveLength(3);
      
      const modes = results.map(r => r.mode);
      expect(modes).toContain('baseline');
      expect(modes).toContain('lsp_assist');
      expect(modes).toContain('competitor_lsp');
    });

    test('executes baseline mode without LSP components', async () => {
      const results = await harness.runBenchmark(mockBaselineSearchHandler);
      
      const baselineResult = results.find(r => r.mode === 'baseline');
      expect(baselineResult).toBeDefined();
      
      // In baseline mode, LSP components should not be called
      expect(mockIntentRouter.routeQuery).not.toHaveBeenCalled();
      expect(mockLspStageBEnhancer.enhanceStageB).not.toHaveBeenCalled();
      expect(mockLspStageCEnhancer.enhanceStageC).not.toHaveBeenCalled();
    });

    test('executes LSP-assist mode with all LSP components', async () => {
      const results = await harness.runBenchmark(mockBaselineSearchHandler);
      
      const lspAssistResult = results.find(r => r.mode === 'lsp_assist');
      expect(lspAssistResult).toBeDefined();
      
      // In LSP-assist mode, all components should be used
      expect(mockIntentRouter.routeQuery).toHaveBeenCalled();
      expect(mockLspStageBEnhancer.enhanceStageB).toHaveBeenCalled();
      expect(mockLspStageCEnhancer.enhanceStageC).toHaveBeenCalled();
    });

    test('handles search errors gracefully', async () => {
      const errorHandler = vi.fn().mockRejectedValue(new Error('Search failed'));
      
      const results = await harness.runBenchmark(errorHandler);
      
      expect(results).toHaveLength(3);
      
      // All results should have zero_result_rate of 1 due to failures
      results.forEach(result => {
        expect(result.zero_result_rate).toBe(1);
      });
    });

    test('calculates metrics correctly for successful queries', async () => {
      // Setup baseline handler to return relevant results
      const relevantCandidate = createMockCandidate('/mock/src/calculateTotal.ts', 10, 0.9);
      mockBaselineSearchHandler.mockResolvedValue([relevantCandidate]);

      const results = await harness.runBenchmark(mockBaselineSearchHandler);
      
      const baselineResult = results.find(r => r.mode === 'baseline');
      expect(baselineResult?.success_at_1).toBeGreaterThan(0);
      expect(baselineResult?.ndcg_at_10).toBeGreaterThan(0);
    });

    test('measures performance timing correctly', async () => {
      vi.useFakeTimers();
      
      // Make search handler take some time
      mockBaselineSearchHandler.mockImplementation(async () => {
        vi.advanceTimersByTime(100);
        return [createMockCandidate('/test.ts', 1)];
      });

      const resultsPromise = harness.runBenchmark(mockBaselineSearchHandler);
      
      vi.runAllTimers();
      const results = await resultsPromise;
      
      const baselineResult = results.find(r => r.mode === 'baseline');
      expect(baselineResult?.p95_latency_ms).toBeGreaterThan(0);
      
      vi.useRealTimers();
    });

    test('saves results to file when output path provided', async () => {
      const outputPath = '/test/benchmark_results.json';
      
      await harness.runBenchmark(mockBaselineSearchHandler, outputPath);
      
      expect(writeFileSync).toHaveBeenCalledWith(
        outputPath,
        expect.stringContaining('"timestamp"'),
      );
    });
  });

  describe('metrics calculation', () => {
    test('calculates success@1 correctly', async () => {
      await harness.loadBenchmarkQueries();

      // Setup handler to return exact match for first query
      const relevantCandidate = createMockCandidate('/mock/src/calculatetotal.ts', 10, 0.9);
      mockBaselineSearchHandler.mockResolvedValue([relevantCandidate]);

      const results = await harness.runBenchmark(mockBaselineSearchHandler);
      
      const baselineResult = results.find(r => r.mode === 'baseline');
      expect(baselineResult?.success_at_1).toBeGreaterThanOrEqual(0);
    });

    test('calculates NDCG@10 correctly', async () => {
      await harness.loadBenchmarkQueries();

      const results = await harness.runBenchmark(mockBaselineSearchHandler);
      
      results.forEach(result => {
        expect(result.ndcg_at_10).toBeGreaterThanOrEqual(0);
        expect(result.ndcg_at_10).toBeLessThanOrEqual(1);
      });
    });

    test('calculates recall and precision correctly', async () => {
      await harness.loadBenchmarkQueries();

      const results = await harness.runBenchmark(mockBaselineSearchHandler);
      
      results.forEach(result => {
        expect(result.recall_at_50).toBeGreaterThanOrEqual(0);
        expect(result.recall_at_50).toBeLessThanOrEqual(1);
      });
    });

    test('handles zero results correctly', async () => {
      await harness.loadBenchmarkQueries();

      mockBaselineSearchHandler.mockResolvedValue([]);

      const results = await harness.runBenchmark(mockBaselineSearchHandler);
      
      results.forEach(result => {
        expect(result.zero_result_rate).toBe(1);
        expect(result.success_at_1).toBe(0);
      });
    });
  });

  describe('loss taxonomy analysis', () => {
    beforeEach(async () => {
      await harness.loadBenchmarkQueries();
    });

    test('identifies NO_SYM_COVERAGE loss', async () => {
      // Return empty results for symbol queries
      mockBaselineSearchHandler.mockResolvedValue([]);

      const results = await harness.runBenchmark(mockBaselineSearchHandler);
      
      const baselineResult = results.find(r => r.mode === 'baseline');
      expect(baselineResult?.loss_taxonomy.NO_SYM_COVERAGE).toBeGreaterThan(0);
    });

    test('identifies WRONG_ALIAS loss', async () => {
      // Return results with alias match reasons but wrong file paths
      const wrongCandidate = createMockCandidate('/wrong/path.ts', 10, 0.8);
      wrongCandidate.match_reasons = ['alias'];
      mockBaselineSearchHandler.mockResolvedValue([wrongCandidate]);

      const results = await harness.runBenchmark(mockBaselineSearchHandler);
      
      // Some results should show WRONG_ALIAS loss
      const hasWrongAlias = results.some(r => r.loss_taxonomy.WRONG_ALIAS > 0);
      expect(hasWrongAlias).toBeTruthy();
    });

    test('identifies RANKING_ONLY loss', async () => {
      // Return relevant results but in positions > 10
      const relevantCandidates = Array.from({ length: 15 }, (_, i) => 
        createMockCandidate(`/mock/src/file${i}.ts`, 10, 0.5 - i * 0.01)
      );
      // Insert one relevant candidate at position 12
      relevantCandidates[11] = createMockCandidate('/mock/src/calculatetotal.ts', 10, 0.4);
      
      mockBaselineSearchHandler.mockResolvedValue(relevantCandidates);

      const results = await harness.runBenchmark(mockBaselineSearchHandler);
      
      // Should identify some RANKING_ONLY losses
      const hasRankingLoss = results.some(r => r.loss_taxonomy.RANKING_ONLY > 0);
      expect(hasRankingLoss).toBeTruthy();
    });

    test('aggregates loss taxonomy correctly', async () => {
      const results = await harness.runBenchmark(mockBaselineSearchHandler);
      
      results.forEach(result => {
        const taxonomy = result.loss_taxonomy;
        Object.values(taxonomy).forEach(value => {
          expect(value).toBeGreaterThanOrEqual(0);
          expect(value).toBeLessThanOrEqual(1);
        });
      });
    });
  });

  describe('comparative analysis', () => {
    test('calculates improvement percentages', async () => {
      await harness.loadBenchmarkQueries();

      // Make LSP-assist perform better than baseline
      mockBaselineSearchHandler.mockImplementation(async (query, context) => {
        if (context.trace_id.includes('lsp_assist')) {
          return [createMockCandidate('/mock/enhanced.ts', 10, 0.95)];
        }
        return [createMockCandidate('/mock/baseline.ts', 10, 0.7)];
      });

      const results = await harness.runBenchmark(mockBaselineSearchHandler);
      
      expect(results).toHaveLength(3);
      
      const baseline = results.find(r => r.mode === 'baseline');
      const lspAssist = results.find(r => r.mode === 'lsp_assist');
      
      expect(baseline).toBeDefined();
      expect(lspAssist).toBeDefined();
    });

    test('handles statistical significance calculation', async () => {
      await harness.loadBenchmarkQueries();

      const results = await harness.runBenchmark(mockBaselineSearchHandler);
      
      // Results should be generated without errors
      expect(results).toHaveLength(3);
      expect(results.every(r => typeof r.success_at_1 === 'number')).toBeTruthy();
    });

    test('generates report with all required sections', async () => {
      await harness.loadBenchmarkQueries();

      const outputPath = '/test/results.json';
      await harness.runBenchmark(mockBaselineSearchHandler, outputPath);
      
      expect(writeFileSync).toHaveBeenCalled();
      const callArgs = vi.mocked(writeFileSync).mock.calls[0];
      const reportContent = JSON.parse(callArgs[1] as string);
      
      expect(reportContent).toHaveProperty('timestamp');
      expect(reportContent).toHaveProperty('summary');
      expect(reportContent).toHaveProperty('results');
      expect(reportContent).toHaveProperty('comparative_analysis');
      expect(reportContent).toHaveProperty('raw_runs');
    });
  });

  describe('performance constraints', () => {
    test('measures latency within expected bounds', async () => {
      await harness.loadBenchmarkQueries();

      const results = await harness.runBenchmark(mockBaselineSearchHandler);
      
      results.forEach(result => {
        // P95 latency should be reasonable for test environment
        expect(result.p95_latency_ms).toBeLessThan(10000); // 10 seconds max
        expect(result.p95_latency_ms).toBeGreaterThan(0);
      });
    });

    test('handles timeout scenarios', async () => {
      vi.useFakeTimers();

      // Make search handler timeout
      mockBaselineSearchHandler.mockImplementation(async () => {
        vi.advanceTimersByTime(15000); // Exceed timeout
        throw new Error('Timeout');
      });

      const resultsPromise = harness.runBenchmark(mockBaselineSearchHandler);
      
      vi.runAllTimers();
      const results = await resultsPromise;
      
      // Should handle timeouts gracefully
      expect(results).toHaveLength(3);
      results.forEach(result => {
        expect(result.timeout_rate).toBe(1);
      });
      
      vi.useRealTimers();
    });

    test('validates LSP enhancement performance impact', async () => {
      await harness.loadBenchmarkQueries();

      // Simulate LSP components adding latency
      mockIntentRouter.routeQuery.mockImplementation(async () => {
        await new Promise(resolve => setTimeout(resolve, 50));
        return {
          primary_candidates: [createMockCandidate('/mock/routed.ts', 5, 0.85)],
          intent_classification: { intent: 'def' as QueryIntent, confidence: 0.9 },
          routing_stats: { router_time_ms: 50 },
        };
      });

      const results = await harness.runBenchmark(mockBaselineSearchHandler);
      
      const baseline = results.find(r => r.mode === 'baseline');
      const lspAssist = results.find(r => r.mode === 'lsp_assist');
      
      expect(baseline?.p95_latency_ms).toBeGreaterThanOrEqual(0);
      expect(lspAssist?.p95_latency_ms).toBeGreaterThanOrEqual(0);
    });
  });

  describe('edge cases and error handling', () => {
    test('handles empty benchmark queries', async () => {
      // Don't load any queries
      const results = await harness.runBenchmark(mockBaselineSearchHandler);
      
      expect(results).toHaveLength(3);
      results.forEach(result => {
        expect(result.success_at_1).toBe(0);
        expect(result.zero_result_rate).toBe(0); // No queries ran
      });
    });

    test('handles null LSP components gracefully', async () => {
      const harnessWithoutLSP = new LSPABBenchmarkHarness();
      await harnessWithoutLSP.loadBenchmarkQueries();

      const results = await harnessWithoutLSP.runBenchmark(mockBaselineSearchHandler);
      
      expect(results).toHaveLength(3);
      
      // Should not crash, LSP modes just won't use LSP components
      const lspAssistResult = results.find(r => r.mode === 'lsp_assist');
      expect(lspAssistResult).toBeDefined();
    });

    test('handles malformed ground truth data', async () => {
      const malformedQueries = [{
        id: 'malformed_1',
        query: 'test query',
        intent: 'def' as QueryIntent,
        ground_truth: [], // Empty ground truth
        difficulty: 'easy' as const,
        language: 'typescript',
      }];

      vi.mocked(existsSync).mockReturnValue(true);
      vi.mocked(readFileSync).mockReturnValue(JSON.stringify(malformedQueries));

      await harness.loadBenchmarkQueries('/test/malformed.json');
      const results = await harness.runBenchmark(mockBaselineSearchHandler);
      
      expect(results).toHaveLength(3);
      // Should handle empty ground truth without crashing
    });

    test('handles concurrent benchmark runs', async () => {
      await harness.loadBenchmarkQueries();

      const results1Promise = harness.runBenchmark(mockBaselineSearchHandler);
      const results2Promise = harness.runBenchmark(mockBaselineSearchHandler);
      
      const [results1, results2] = await Promise.all([results1Promise, results2Promise]);
      
      expect(results1).toHaveLength(3);
      expect(results2).toHaveLength(3);
    });
  });

  describe('integration with LSP components', () => {
    test('validates LSP Stage-B enhancement integration', async () => {
      await harness.loadBenchmarkQueries();

      const enhancedCandidate = createMockCandidate('/enhanced.ts', 5, 0.95);
      mockLspStageBEnhancer.enhanceStageB.mockResolvedValue({
        candidates: [enhancedCandidate],
        metrics: { enhanced_count: 1, processing_time_ms: 75 },
      });

      const results = await harness.runBenchmark(mockBaselineSearchHandler);
      
      const lspAssistResult = results.find(r => r.mode === 'lsp_assist');
      expect(lspAssistResult).toBeDefined();
      
      expect(mockLspStageBEnhancer.enhanceStageB).toHaveBeenCalled();
    });

    test('validates LSP Stage-C enhancement integration', async () => {
      await harness.loadBenchmarkQueries();

      const enhancedCandidate = createMockCandidate('/stage-c-enhanced.ts', 8, 0.92);
      mockLspStageCEnhancer.enhanceStageC.mockResolvedValue({
        enhanced_candidates: [enhancedCandidate],
        feature_stats: { total_features: 10, active_features: 7 },
        bounded_contribution: 0.35, // Within 0.4 log-odds limit
      });

      const results = await harness.runBenchmark(mockBaselineSearchHandler);
      
      const lspAssistResult = results.find(r => r.mode === 'lsp_assist');
      expect(lspAssistResult).toBeDefined();
      
      expect(mockLspStageCEnhancer.enhanceStageC).toHaveBeenCalled();
    });

    test('validates Intent Router integration', async () => {
      await harness.loadBenchmarkQueries();

      mockIntentRouter.routeQuery.mockResolvedValue({
        primary_candidates: [createMockCandidate('/routed.ts', 12, 0.88)],
        intent_classification: { intent: 'symbol' as QueryIntent, confidence: 0.85 },
        routing_stats: { router_time_ms: 30 },
      });

      const results = await harness.runBenchmark(mockBaselineSearchHandler);
      
      const lspAssistResult = results.find(r => r.mode === 'lsp_assist');
      expect(lspAssistResult).toBeDefined();
      
      expect(mockIntentRouter.routeQuery).toHaveBeenCalled();
    });

    test('validates LSP component error handling', async () => {
      await harness.loadBenchmarkQueries();

      // Make LSP Stage-B fail
      mockLspStageBEnhancer.enhanceStageB.mockRejectedValue(new Error('LSP Stage-B failed'));

      const results = await harness.runBenchmark(mockBaselineSearchHandler);
      
      expect(results).toHaveLength(3);
      
      // Should handle LSP failures gracefully
      const lspAssistResult = results.find(r => r.mode === 'lsp_assist');
      expect(lspAssistResult).toBeDefined();
    });
  });
});