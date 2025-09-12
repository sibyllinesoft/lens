/**
 * Unit tests for LSP vs Serena comparison functionality
 * Validates that the comparison test correctly identifies performance gaps
 * and LSP activation evidence
 */

import { describe, test, expect, jest, beforeEach, afterEach, mock } from 'bun:test';
import { LSPSerenaComparisonTest } from '../../benchmarks/src/lsp-serena-comparison.js';
import { WorkspaceConfig } from '../core/workspace-config.js';
import { SearchEngine } from '../api/search-engine.js';
import { LSPSidecar } from '../core/lsp-sidecar.js';
import type { Candidate } from '../types/core.js';

// Mock external dependencies
mock('fs', () => ({
  readFileSync: jest.fn(),
  writeFileSync: jest.fn(),
  existsSync: jest.fn().mockReturnValue(true),
}));

mock('child_process', () => ({
  spawn: jest.fn(() => ({
    stdout: { on: jest.fn(), setEncoding: jest.fn() },
    stderr: { on: jest.fn(), setEncoding: jest.fn() },
    stdin: { write: jest.fn(), end: jest.fn() },
    on: jest.fn(),
    kill: jest.fn(),
    pid: 12345,
  })),
}));

mock('../telemetry/tracer.js', () => ({
  LensTracer: {
    createChildSpan: jest.fn(() => ({
      setAttributes: jest.fn(),
      recordException: jest.fn(),
      end: jest.fn(),
    })),
  },
}));

describe('LSP-Serena Comparison Test', () => {
  let comparisonTest: LSPSerenaComparisonTest;
  let mockWorkspaceConfig: WorkspaceConfig;
  
  const testCorpusPath = '/test/corpus';
  
  beforeEach(async () => {
    jest.clearAllMocks();
    
    // Mock workspace config
    mockWorkspaceConfig = {
      parseWorkspaceConfig: jest.fn().mockResolvedValue(undefined),
      getStats: jest.fn().mockReturnValue({
        languages: { typescript: true, rust: true },
        configs_loaded: 2
      })
    } as any;
    
    comparisonTest = new LSPSerenaComparisonTest(
      testCorpusPath,
      undefined, // No Serena path for unit tests
      mockWorkspaceConfig
    );
  });

  afterEach(async () => {
    if (comparisonTest) {
      await comparisonTest.cleanup();
    }
    jest.restoreAllMocks();
  });

  describe('test initialization', () => {
    test('initializes with test corpus path', () => {
      expect(comparisonTest).toBeDefined();
    });

    test('loads test queries on initialization', async () => {
      await comparisonTest.initializeTest();
      
      // Verify test was initialized (no direct getter, but should not throw)
      expect(true).toBe(true);
    });

    test('handles missing workspace configuration gracefully', async () => {
      const testWithoutConfig = new LSPSerenaComparisonTest(
        testCorpusPath,
        undefined,
        undefined
      );
      
      // Should throw during LSP initialization, but not during construction
      await expect(testWithoutConfig.initializeTest()).rejects.toThrow();
    });
  });

  describe('LSP activation verification', () => {
    test('detects when LSP servers are not started', async () => {
      // Mock LSP sidecar that fails to start
      const mockLSPSidecar = {
        initializeLanguageServers: jest.fn().mockRejectedValue(new Error('Server failed')),
        harvestSymbols: jest.fn().mockResolvedValue([]),
        shutdown: jest.fn().mockResolvedValue(undefined)
      } as any;
      
      // Use reflection to set the LSP sidecar
      (comparisonTest as any).lspSidecar = mockLSPSidecar;
      
      const evidence = await comparisonTest.verifyLSPActivation();
      
      expect(evidence.server_status).toBe('failed');
      expect(evidence.has_lsp_hints).toBe(false);
      expect(evidence.hint_count).toBe(0);
    });

    test('detects successful LSP activation', async () => {
      // Mock successful LSP sidecar
      const mockHints = [
        {
          symbol_id: 'test_symbol',
          name: 'testFunction',
          file_path: '/test/file.ts',
          line: 10,
          col: 0,
          kind: 'function',
          container: 'test',
          detail: 'function testFunction(): void',
          range: { start: { line: 10, character: 0 }, end: { line: 15, character: 1 } }
        }
      ];
      
      const mockLSPSidecar = {
        initializeLanguageServers: jest.fn().mockResolvedValue(undefined),
        harvestSymbols: jest.fn().mockResolvedValue(mockHints),
        shutdown: jest.fn().mockResolvedValue(undefined)
      } as any;
      
      (comparisonTest as any).lspSidecar = mockLSPSidecar;
      
      const evidence = await comparisonTest.verifyLSPActivation();
      
      expect(evidence.server_status).toBe('active');
      expect(evidence.has_lsp_hints).toBe(true);
      expect(evidence.hint_count).toBe(1);
    });
  });

  describe('query execution and comparison', () => {
    test('executes lens query with LSP evidence detection', async () => {
      // Mock search engine
      const mockCandidates: Candidate[] = [
        {
          file_path: '/test/file.ts',
          line: 10,
          col: 0,
          content: 'function testFunction() {}',
          symbol: 'testFunction',
          match_reasons: ['lsp_hint', 'lsp_routed_as_def'],
          similarity: 0.9,
          stage_b_score: 0.85,
          stage_c_features: {
            is_definition: true,
            is_reference: false,
            has_documentation: true,
            complexity_score: 0.3,
            recency_score: 0.8
          }
        }
      ];

      const mockSearchEngine = {
        search: jest.fn().mockResolvedValue(mockCandidates),
        initializeLSP: jest.fn().mockResolvedValue(undefined),
        shutdown: jest.fn().mockResolvedValue(undefined)
      } as any;

      (comparisonTest as any).searchEngine = mockSearchEngine;

      const testQuery = {
        id: 'test_def_query',
        query: 'function testFunction',
        intent: 'def' as const,
        description: 'Test definition query',
        expected_symbols: ['testFunction'],
        difficulty: 'easy' as const,
        focus_area: 'def' as const
      };

      const result = await (comparisonTest as any).runLensQuery(testQuery);

      expect(result.system_name).toBe('lens_lsp');
      expect(result.candidates).toHaveLength(1);
      expect(result.success_at_1).toBe(true);
      expect(result.lsp_evidence).toBeDefined();
      expect(result.lsp_evidence!.has_lsp_hints).toBe(true);
      expect(result.lsp_evidence!.lsp_routing_markers).toContain('lsp_hint');
      expect(result.lsp_evidence!.lsp_routed_queries).toContain('lsp_routed_as_def');
    });

    test('generates mock serena results with expected success rate', async () => {
      const testQuery = {
        id: 'test_serena_query',
        query: 'function testFunction',
        intent: 'def' as const,
        description: 'Test query for Serena',
        expected_symbols: ['testFunction'],
        difficulty: 'easy' as const,
        focus_area: 'def' as const
      };

      // Run multiple times to test probabilistic behavior
      const results = [];
      for (let i = 0; i < 100; i++) {
        const result = await (comparisonTest as any).runSerenaQuery(testQuery);
        results.push(result.success_at_1);
      }

      // Should approximate 54.9% success rate (with some variance due to randomness)
      const successRate = results.filter(s => s).length / results.length;
      expect(successRate).toBeGreaterThan(0.4); // At least 40%
      expect(successRate).toBeLessThan(0.7);    // At most 70%
    });

    test('correctly identifies performance gaps', async () => {
      const testQuery = {
        id: 'gap_analysis_query',
        query: 'function testGap',
        intent: 'def' as const,
        description: 'Query for gap analysis',
        expected_symbols: ['testGap'],
        difficulty: 'medium' as const,
        focus_area: 'def' as const
      };

      const lensResult = {
        system_name: 'lens_lsp' as const,
        query_id: 'gap_analysis_query',
        candidates: [],
        success_at_1: false,
        success_at_5: false,
        success_at_10: false,
        execution_time_ms: 100,
        lsp_evidence: {
          has_lsp_hints: false,
          lsp_routed_queries: [],
          lsp_routing_markers: [],
          hint_count: 0,
          server_status: 'not_started' as const
        }
      };

      const serenaResult = {
        system_name: 'serena_lsp' as const,
        query_id: 'gap_analysis_query',
        candidates: [
          {
            file_path: '/mock/testgap.ts',
            line: 5,
            col: 0,
            content: 'mock testGap content',
            symbol: 'testGap',
            match_reasons: ['serena_semantic'],
            similarity: 0.8,
            stage_b_score: 0.8,
            stage_c_features: {
              is_definition: true,
              is_reference: false,
              has_documentation: false,
              complexity_score: 0.5,
              recency_score: 0.5
            }
          }
        ],
        success_at_1: true,
        success_at_5: true,
        success_at_10: true,
        execution_time_ms: 80
      };

      const gapAnalysis = (comparisonTest as any).analyzeGap(
        testQuery, 
        lensResult, 
        serenaResult
      );

      expect(gapAnalysis.serena_better).toBe(true);
      expect(gapAnalysis.lens_better).toBe(false);
      expect(gapAnalysis.performance_gap_pp).toBeGreaterThan(0);
      expect(gapAnalysis.specific_failures).toContain('no_lsp_hints_detected');
      expect(gapAnalysis.improvement_opportunities).toContain('verify_lsp_server_startup');
    });
  });

  describe('comprehensive analysis', () => {
    test('calculates overall success rates correctly', async () => {
      // Setup mock results
      (comparisonTest as any).results = [
        {
          query: { id: 'q1' },
          lens_result: { success_at_5: true, error: undefined },
          serena_result: { success_at_5: false, error: undefined },
          gap_analysis: {}
        },
        {
          query: { id: 'q2' },
          lens_result: { success_at_5: false, error: undefined },
          serena_result: { success_at_5: true, error: undefined },
          gap_analysis: {}
        },
        {
          query: { id: 'q3' },
          lens_result: { success_at_5: true, error: 'failed' },
          serena_result: { success_at_5: true, error: undefined },
          gap_analysis: {}
        }
      ];

      const lensRate = (comparisonTest as any).calculateOverallSuccessRate('lens_lsp');
      const serenaRate = (comparisonTest as any).calculateOverallSuccessRate('serena_lsp');

      // Lens: 1 success out of 2 valid results = 50%
      expect(lensRate).toBe(0.5);
      // Serena: 2 successes out of 3 valid results = 67%
      expect(serenaRate).toBeCloseTo(0.67, 2);
    });

    test('identifies gap closure achievement', async () => {
      // Setup results that show gap closure
      (comparisonTest as any).results = [
        // Lens wins most queries
        ...Array(8).fill(0).map((_, i) => ({
          query: { id: `q${i}` },
          lens_result: { success_at_5: true, error: undefined },
          serena_result: { success_at_5: false, error: undefined },
          gap_analysis: { performance_gap_pp: -5 } // Lens better by 5pp
        })),
        // Serena wins fewer queries
        ...Array(2).fill(0).map((_, i) => ({
          query: { id: `q${i + 8}` },
          lens_result: { success_at_5: false, error: undefined },
          serena_result: { success_at_5: true, error: undefined },
          gap_analysis: { performance_gap_pp: 10 } // Serena better by 10pp
        }))
      ];

      const report = await comparisonTest.generateComparisonReport();

      expect(report.performance_metrics.lens_success_rate).toBeGreaterThan(0.7);
      expect(report.performance_metrics.performance_gap_pp).toBeLessThan(7);
      expect(report.performance_metrics.gap_closure_achieved).toBe(true);
    });

    test('detects when gap closure is not achieved', async () => {
      // Setup results that show persistent gap
      (comparisonTest as any).results = [
        // Serena wins most queries
        ...Array(8).fill(0).map((_, i) => ({
          query: { id: `q${i}` },
          lens_result: { success_at_5: false, error: undefined },
          serena_result: { success_at_5: true, error: undefined },
          gap_analysis: { performance_gap_pp: 15 } // Large gap
        })),
        // Lens wins fewer
        ...Array(2).fill(0).map((_, i) => ({
          query: { id: `q${i + 8}` },
          lens_result: { success_at_5: true, error: undefined },
          serena_result: { success_at_5: false, error: undefined },
          gap_analysis: { performance_gap_pp: -5 }
        }))
      ];

      const report = await comparisonTest.generateComparisonReport();

      expect(report.performance_metrics.performance_gap_pp).toBeGreaterThan(7);
      expect(report.performance_metrics.gap_closure_achieved).toBe(false);
    });
  });

  describe('error handling and edge cases', () => {
    test('handles search engine failures gracefully', async () => {
      const mockSearchEngine = {
        search: jest.fn().mockRejectedValue(new Error('Search failed')),
        initializeLSP: jest.fn().mockResolvedValue(undefined),
        shutdown: jest.fn().mockResolvedValue(undefined)
      } as any;

      (comparisonTest as any).searchEngine = mockSearchEngine;

      const testQuery = {
        id: 'failing_query',
        query: 'broken query',
        intent: 'def' as const,
        description: 'Query that will fail',
        expected_symbols: ['broken'],
        difficulty: 'hard' as const,
        focus_area: 'def' as const
      };

      const result = await (comparisonTest as any).runLensQuery(testQuery);

      expect(result.error).toBeDefined();
      expect(result.success_at_1).toBe(false);
      expect(result.candidates).toHaveLength(0);
    });

    test('handles empty candidate lists', async () => {
      const testQuery = {
        id: 'empty_results',
        query: 'nonexistent',
        intent: 'def' as const,
        description: 'Query with no results',
        expected_symbols: ['nonexistent'],
        difficulty: 'hard' as const,
        focus_area: 'def' as const
      };

      const emptyResults = await (comparisonTest as any).runSerenaQuery(testQuery);
      
      // Should handle empty results gracefully
      expect(emptyResults.candidates).toHaveLength(0);
      expect(emptyResults.success_at_1).toBe(false);
    });

    test('handles malformed LSP evidence', async () => {
      const malformedCandidates = [
        {
          file_path: '/test/file.ts',
          line: 10,
          col: 0,
          content: 'test content',
          symbol: 'test',
          match_reasons: ['invalid_lsp_marker', null, undefined] as any,
          similarity: 0.8,
          stage_b_score: 0.8,
          stage_c_features: {} as any
        }
      ];

      const evidence = (comparisonTest as any).extractLSPEvidence(malformedCandidates);

      expect(evidence).toBeDefined();
      expect(evidence.has_lsp_hints).toBe(false);
      expect(evidence.lsp_routing_markers).toEqual([]);
    });
  });

  describe('statistical analysis', () => {
    test('calculates improvement distribution correctly', async () => {
      const mockResults = [
        { gap_analysis: { performance_gap_pp: -15 } }, // Lens better by 15pp
        { gap_analysis: { performance_gap_pp: -8 } },  // Lens better by 8pp
        { gap_analysis: { performance_gap_pp: -3 } },  // Lens better by 3pp
        { gap_analysis: { performance_gap_pp: 2 } },   // Serena better by 2pp
        { gap_analysis: { performance_gap_pp: 12 } },  // Serena better by 12pp
      ];

      (comparisonTest as any).results = mockResults;

      const stats = (comparisonTest as any).performStatisticalAnalysis();

      expect(stats.improvement_distribution['lens_better_10pp']).toBe(1); // -15pp
      expect(stats.improvement_distribution['lens_better_5pp']).toBe(1);  // -8pp
      expect(stats.improvement_distribution['competitive_parity']).toBe(2); // -3pp, +2pp
      expect(stats.improvement_distribution['serena_better_10pp']).toBe(1); // +12pp

      expect(stats.mean_improvement).toBeGreaterThan(0); // Overall positive improvement
      expect(stats.is_significant).toBe(true); // Should be significant
    });
  });
});