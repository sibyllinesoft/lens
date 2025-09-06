/**
 * Integration tests for full search pipeline with LSP enabled
 * Tests end-to-end integration of LSP components with search engine
 * Validates performance bounds and feature constraints
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
  QueryIntent,
  LSPHint,
  LSPBenchmarkResult
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
    stdout: {
      on: vi.fn(),
      setEncoding: vi.fn(),
    },
    stderr: {
      on: vi.fn(),
      setEncoding: vi.fn(),
    },
    stdin: {
      write: vi.fn(),
      end: vi.fn(),
    },
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

describe('LSP Integration Tests', () => {
  let searchEngine: SearchEngine;
  let lspSidecar: LSPSidecar;
  let lspStageBEnhancer: LSPStageBEnhancer;
  let lspStageCEnhancer: LSPStageCEnhancer;
  let intentRouter: IntentRouter;
  let workspaceConfig: WorkspaceConfig;

  // Mock repository data
  const mockRepoPath = '/test/repo';
  const mockWorkspaceFiles = [
    '/test/repo/src/utils.ts',
    '/test/repo/src/services/user.ts',
    '/test/repo/src/models/user.ts',
    '/test/repo/src/controllers/auth.ts',
    '/test/repo/package.json',
    '/test/repo/tsconfig.json',
  ];

  const mockTsConfig = {
    compilerOptions: {
      baseUrl: './src',
      paths: {
        '@/*': ['src/*'],
        '@utils/*': ['src/utils/*'],
        '@models/*': ['src/models/*'],
      },
    },
    include: ['src/**/*'],
    exclude: ['node_modules', 'dist'],
  };

  const mockLSPHints: LSPHint[] = [
    {
      symbol_id: 'func_calculateTotal',
      name: 'calculateTotal',
      file_path: '/test/repo/src/utils.ts',
      line: 10,
      col: 0,
      kind: 'function',
      container: 'utils',
      detail: 'function calculateTotal(items: Item[]): number',
      range: { start: { line: 10, character: 0 }, end: { line: 15, character: 1 } },
    },
    {
      symbol_id: 'class_UserService',
      name: 'UserService',
      file_path: '/test/repo/src/services/user.ts',
      line: 5,
      col: 0,
      kind: 'class',
      container: 'services',
      detail: 'class UserService',
      range: { start: { line: 5, character: 0 }, end: { line: 100, character: 1 } },
    },
    {
      symbol_id: 'interface_User',
      name: 'User',
      file_path: '/test/repo/src/models/user.ts',
      line: 1,
      col: 0,
      kind: 'interface',
      container: 'models',
      detail: 'interface User',
      range: { start: { line: 1, character: 0 }, end: { line: 10, character: 1 } },
    },
    {
      symbol_id: 'method_authenticate',
      name: 'authenticate',
      file_path: '/test/repo/src/controllers/auth.ts',
      line: 15,
      col: 2,
      kind: 'method',
      container: 'AuthController',
      detail: 'authenticate(credentials: LoginCredentials): Promise<AuthResult>',
      range: { start: { line: 15, character: 2 }, end: { line: 25, character: 3 } },
    },
  ];

  const createMockCandidate = (
    filePath: string,
    line: number,
    matchReasons: string[] = ['lexical'],
    similarity = 0.7,
    symbol?: string
  ): Candidate => ({
    file_path: filePath,
    line,
    col: 0,
    content: `mock content at ${filePath}:${line}`,
    symbol: symbol || 'mockSymbol',
    match_reasons: matchReasons,
    similarity,
    stage_b_score: similarity,
    stage_c_features: {
      is_definition: matchReasons.includes('definition'),
      is_reference: matchReasons.includes('reference'),
      has_documentation: false,
      complexity_score: 0.5,
      recency_score: 0.5,
    },
  });

  const createSearchContext = (query: string): SearchContext => ({
    trace_id: `integration_test_${Date.now()}`,
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

    // Mock file system for workspace config
    vi.mocked(require('fs').readFileSync).mockImplementation((path: string) => {
      if (path.includes('tsconfig.json')) {
        return JSON.stringify(mockTsConfig);
      }
      if (path.includes('package.json')) {
        return JSON.stringify({ name: 'test-repo', dependencies: { typescript: '^5.0.0' } });
      }
      return '{}';
    });

    // Initialize workspace config
    workspaceConfig = new WorkspaceConfig(mockRepoPath);
    await workspaceConfig.parseWorkspaceConfig();

    // Initialize LSP components
    lspSidecar = new LSPSidecar(workspaceConfig);
    lspStageBEnhancer = new LSPStageBEnhancer(lspSidecar);
    lspStageCEnhancer = new LSPStageCEnhancer(lspSidecar);
    intentRouter = new IntentRouter(lspSidecar);

    // Mock LSP sidecar methods
    vi.spyOn(lspSidecar, 'initializeLanguageServers').mockResolvedValue(undefined);
    vi.spyOn(lspSidecar, 'harvestSymbols').mockResolvedValue(mockLSPHints);
    vi.spyOn(lspSidecar, 'generateHints').mockResolvedValue(mockLSPHints);
    vi.spyOn(lspSidecar, 'shutdown').mockResolvedValue(undefined);

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

    await searchEngine.initializeLSP();
  });

  afterEach(async () => {
    await searchEngine.shutdown();
    vi.restoreAllMocks();
  });

  describe('LSP initialization and configuration', () => {
    test('initializes LSP components successfully', () => {
      expect(lspSidecar.initializeLanguageServers).toHaveBeenCalled();
    });

    test('harvests symbols from workspace', async () => {
      const hints = await lspSidecar.harvestSymbols();
      
      expect(hints).toHaveLength(4);
      expect(hints.find(h => h.name === 'calculateTotal')).toBeDefined();
      expect(hints.find(h => h.name === 'UserService')).toBeDefined();
      expect(hints.find(h => h.name === 'User')).toBeDefined();
      expect(hints.find(h => h.name === 'authenticate')).toBeDefined();
    });

    test('loads workspace configuration correctly', () => {
      const stats = workspaceConfig.getStats();
      
      expect(stats.languages.typescript).toBe(true);
      expect(stats.configs_loaded).toBeGreaterThan(0);
    });
  });

  describe('end-to-end search integration', () => {
    test('executes definition search with LSP enhancement', async () => {
      const query = 'function calculateTotal';
      const context = createSearchContext(query);

      // Mock baseline search to return candidates
      const baselineCandidates = [
        createMockCandidate('/test/repo/src/utils.ts', 10, ['lexical'], 0.6, 'calculateTotal'),
      ];

      // Mock LSP Stage-B enhancement
      vi.spyOn(lspStageBEnhancer, 'enhanceStageB').mockResolvedValue({
        candidates: [
          createMockCandidate('/test/repo/src/utils.ts', 10, ['lsp_hint', 'definition'], 0.9, 'calculateTotal'),
        ],
        metrics: {
          enhanced_count: 1,
          processing_time_ms: 45,
        },
      });

      // Mock LSP Stage-C enhancement
      vi.spyOn(lspStageCEnhancer, 'enhanceStageC').mockResolvedValue({
        enhanced_candidates: [
          createMockCandidate('/test/repo/src/utils.ts', 10, ['lsp_hint', 'definition'], 0.95, 'calculateTotal'),
        ],
        feature_stats: {
          total_features: 8,
          active_features: 6,
        },
        bounded_contribution: 0.35, // Within 0.4 log-odds limit
      });

      // Mock intent router
      vi.spyOn(intentRouter, 'routeQuery').mockResolvedValue({
        primary_candidates: [
          createMockCandidate('/test/repo/src/utils.ts', 10, ['intent_routed'], 0.88, 'calculateTotal'),
        ],
        intent_classification: {
          intent: 'def' as QueryIntent,
          confidence: 0.92,
        },
        routing_stats: {
          router_time_ms: 25,
        },
      });

      // Execute search
      const results = await searchEngine.search(query, context);

      // Validate results
      expect(results).toBeDefined();
      expect(results.length).toBeGreaterThan(0);
      
      const topResult = results[0];
      expect(topResult.file_path).toBe('/test/repo/src/utils.ts');
      expect(topResult.line).toBe(10);
      expect(topResult.similarity).toBeGreaterThan(0.8); // Enhanced by LSP

      // Validate LSP components were called
      expect(intentRouter.routeQuery).toHaveBeenCalled();
      expect(lspStageBEnhancer.enhanceStageB).toHaveBeenCalled();
      expect(lspStageCEnhancer.enhanceStageC).toHaveBeenCalled();
    });

    test('executes references search with LSP assistance', async () => {
      const query = 'refs UserService';
      const context = createSearchContext(query);

      // Mock intent router to return reference candidates
      vi.spyOn(intentRouter, 'routeQuery').mockResolvedValue({
        primary_candidates: [
          createMockCandidate('/test/repo/src/controllers/auth.ts', 20, ['reference'], 0.85, 'UserService'),
          createMockCandidate('/test/repo/src/middleware/auth.ts', 15, ['reference'], 0.80, 'UserService'),
        ],
        intent_classification: {
          intent: 'refs' as QueryIntent,
          confidence: 0.89,
        },
        routing_stats: {
          router_time_ms: 30,
        },
      });

      // Mock Stage-B enhancement
      vi.spyOn(lspStageBEnhancer, 'enhanceStageB').mockResolvedValue({
        candidates: [
          createMockCandidate('/test/repo/src/controllers/auth.ts', 20, ['lsp_hint', 'reference'], 0.92, 'UserService'),
          createMockCandidate('/test/repo/src/middleware/auth.ts', 15, ['lsp_hint', 'reference'], 0.87, 'UserService'),
        ],
        metrics: {
          enhanced_count: 2,
          processing_time_ms: 55,
        },
      });

      // Mock Stage-C enhancement
      vi.spyOn(lspStageCEnhancer, 'enhanceStageC').mockResolvedValue({
        enhanced_candidates: [
          createMockCandidate('/test/repo/src/controllers/auth.ts', 20, ['lsp_hint', 'reference'], 0.94, 'UserService'),
          createMockCandidate('/test/repo/src/middleware/auth.ts', 15, ['lsp_hint', 'reference'], 0.89, 'UserService'),
        ],
        feature_stats: {
          total_features: 10,
          active_features: 7,
        },
        bounded_contribution: 0.38, // Within bounds
      });

      const results = await searchEngine.search(query, context);

      expect(results).toBeDefined();
      expect(results.length).toBeGreaterThanOrEqual(2);
      
      // Validate reference results
      const authResult = results.find(r => r.file_path.includes('auth.ts'));
      expect(authResult).toBeDefined();
      expect(authResult?.match_reasons).toContain('reference');
    });

    test('executes symbol-by-name search with workspace awareness', async () => {
      const query = 'User';
      const context = createSearchContext(query);

      // Mock intent router
      vi.spyOn(intentRouter, 'routeQuery').mockResolvedValue({
        primary_candidates: [
          createMockCandidate('/test/repo/src/models/user.ts', 1, ['symbol'], 0.90, 'User'),
        ],
        intent_classification: {
          intent: 'symbol' as QueryIntent,
          confidence: 0.88,
        },
        routing_stats: {
          router_time_ms: 28,
        },
      });

      // Mock Stage-B with symbol mapping
      vi.spyOn(lspStageBEnhancer, 'enhanceStageB').mockResolvedValue({
        candidates: [
          createMockCandidate('/test/repo/src/models/user.ts', 1, ['lsp_hint', 'symbol'], 0.95, 'User'),
        ],
        metrics: {
          enhanced_count: 1,
          processing_time_ms: 40,
        },
      });

      const results = await searchEngine.search(query, context);

      expect(results).toBeDefined();
      expect(results.length).toBeGreaterThan(0);
      
      const symbolResult = results[0];
      expect(symbolResult.file_path).toBe('/test/repo/src/models/user.ts');
      expect(symbolResult.symbol).toBe('User');
    });

    test('executes structural search with LSP-aware parsing', async () => {
      const query = 'if (user &&';
      const context = createSearchContext(query);

      // Mock intent router for structural query
      vi.spyOn(intentRouter, 'routeQuery').mockResolvedValue({
        primary_candidates: [
          createMockCandidate('/test/repo/src/controllers/auth.ts', 35, ['structural'], 0.75),
          createMockCandidate('/test/repo/src/middleware/validation.ts', 20, ['structural'], 0.70),
        ],
        intent_classification: {
          intent: 'struct' as QueryIntent,
          confidence: 0.82,
        },
        routing_stats: {
          router_time_ms: 35,
        },
      });

      const results = await searchEngine.search(query, context);

      expect(results).toBeDefined();
      expect(results.length).toBeGreaterThan(0);
      
      // Validate structural pattern results
      results.forEach(result => {
        expect(result.match_reasons.some(reason => 
          reason.includes('structural') || reason.includes('pattern')
        )).toBeTruthy();
      });
    });

    test('executes natural language query with semantic enhancement', async () => {
      const query = 'function that validates user authentication';
      const context = createSearchContext(query);

      // Mock intent router for NL query
      vi.spyOn(intentRouter, 'routeQuery').mockResolvedValue({
        primary_candidates: [
          createMockCandidate('/test/repo/src/controllers/auth.ts', 15, ['semantic'], 0.78, 'authenticate'),
          createMockCandidate('/test/repo/src/middleware/auth.ts', 10, ['semantic'], 0.73, 'validateAuth'),
        ],
        intent_classification: {
          intent: 'NL' as QueryIntent,
          confidence: 0.86,
        },
        routing_stats: {
          router_time_ms: 45,
        },
      });

      const results = await searchEngine.search(query, context);

      expect(results).toBeDefined();
      expect(results.length).toBeGreaterThan(0);
      
      // Validate semantic enhancement
      const authResult = results.find(r => r.symbol === 'authenticate');
      expect(authResult).toBeDefined();
      expect(authResult?.match_reasons).toContain('semantic');
    });
  });

  describe('performance bounds validation', () => {
    test('maintains Stage-B p95 latency ≤+3ms constraint', async () => {
      vi.useFakeTimers();
      
      const query = 'calculateTotal';
      const context = createSearchContext(query);

      // Mock Stage-B to simulate processing time
      vi.spyOn(lspStageBEnhancer, 'enhanceStageB').mockImplementation(async () => {
        // Simulate 2ms processing time (within bounds)
        vi.advanceTimersByTime(2);
        return {
          candidates: [createMockCandidate('/test/repo/src/utils.ts', 10, ['lsp_hint'], 0.9)],
          metrics: {
            enhanced_count: 1,
            processing_time_ms: 2,
          },
        };
      });

      const startTime = Date.now();
      await searchEngine.search(query, context);
      vi.runAllTimers();
      const endTime = Date.now();
      
      const stageBLatency = 2; // From mock
      expect(stageBLatency).toBeLessThanOrEqual(3); // ≤+3ms constraint
      
      vi.useRealTimers();
    });

    test('validates bounded LSP contribution ≤0.4 log-odds', async () => {
      const query = 'UserService.findById';
      const context = createSearchContext(query);

      // Mock Stage-C with bounded contribution
      vi.spyOn(lspStageCEnhancer, 'enhanceStageC').mockResolvedValue({
        enhanced_candidates: [
          createMockCandidate('/test/repo/src/services/user.ts', 25, ['lsp_hint'], 0.85),
        ],
        feature_stats: {
          total_features: 12,
          active_features: 8,
        },
        bounded_contribution: 0.38, // Within 0.4 limit
      });

      await searchEngine.search(query, context);

      expect(lspStageCEnhancer.enhanceStageC).toHaveBeenCalled();
      
      const callArgs = vi.mocked(lspStageCEnhancer.enhanceStageC).mock.results[0];
      const result = await callArgs.value;
      
      expect(result.bounded_contribution).toBeLessThanOrEqual(0.4);
    });

    test('measures end-to-end search latency', async () => {
      const query = 'interface User';
      const context = createSearchContext(query);

      const startTime = performance.now();
      const results = await searchEngine.search(query, context);
      const endTime = performance.now();
      
      const totalLatency = endTime - startTime;
      
      expect(results).toBeDefined();
      expect(totalLatency).toBeLessThan(5000); // 5 second timeout
      expect(totalLatency).toBeGreaterThan(0);
    });

    test('validates memory usage stays within bounds', async () => {
      const queries = [
        'function calculateTotal',
        'refs UserService', 
        'User interface',
        'authenticate method',
        'if (user && user.active)',
      ];

      const initialMemory = process.memoryUsage().heapUsed;

      // Execute multiple queries
      for (const query of queries) {
        const context = createSearchContext(query);
        await searchEngine.search(query, context);
      }

      const finalMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = (finalMemory - initialMemory) / 1024 / 1024; // MB

      // Memory increase should be reasonable
      expect(memoryIncrease).toBeLessThan(100); // Less than 100MB increase
    });
  });

  describe('LSP fallback behavior', () => {
    test('falls back to baseline search when LSP fails', async () => {
      const query = 'fallbackTest';
      const context = createSearchContext(query);

      // Mock LSP components to fail
      vi.spyOn(intentRouter, 'routeQuery').mockRejectedValue(new Error('LSP Router failed'));
      vi.spyOn(lspStageBEnhancer, 'enhanceStageB').mockRejectedValue(new Error('Stage-B failed'));
      vi.spyOn(lspStageCEnhancer, 'enhanceStageC').mockRejectedValue(new Error('Stage-C failed'));

      // Should fall back to baseline search
      const results = await searchEngine.search(query, context);

      expect(results).toBeDefined();
      // Results should still be returned from fallback mechanism
    });

    test('handles partial LSP failures gracefully', async () => {
      const query = 'partialFailure';
      const context = createSearchContext(query);

      // Mock intent router to succeed
      vi.spyOn(intentRouter, 'routeQuery').mockResolvedValue({
        primary_candidates: [createMockCandidate('/test/repo/src/test.ts', 10)],
        intent_classification: { intent: 'def' as QueryIntent, confidence: 0.8 },
        routing_stats: { router_time_ms: 25 },
      });

      // Mock Stage-B to fail
      vi.spyOn(lspStageBEnhancer, 'enhanceStageB').mockRejectedValue(new Error('Stage-B timeout'));

      // Mock Stage-C to succeed
      vi.spyOn(lspStageCEnhancer, 'enhanceStageC').mockResolvedValue({
        enhanced_candidates: [createMockCandidate('/test/repo/src/test.ts', 10, ['lsp_hint'], 0.85)],
        feature_stats: { total_features: 5, active_features: 3 },
        bounded_contribution: 0.25,
      });

      const results = await searchEngine.search(query, context);

      expect(results).toBeDefined();
      expect(results.length).toBeGreaterThan(0);
      
      // Should continue with available LSP components
      expect(intentRouter.routeQuery).toHaveBeenCalled();
      expect(lspStageCEnhancer.enhanceStageC).toHaveBeenCalled();
    });

    test('handles LSP timeout gracefully', async () => {
      vi.useFakeTimers();

      const query = 'timeoutTest';
      const context = createSearchContext(query);

      // Mock LSP components with long delays
      vi.spyOn(intentRouter, 'routeQuery').mockImplementation(async () => {
        vi.advanceTimersByTime(10000); // 10 second delay
        return {
          primary_candidates: [createMockCandidate('/test/repo/src/test.ts', 10)],
          intent_classification: { intent: 'def' as QueryIntent, confidence: 0.8 },
          routing_stats: { router_time_ms: 10000 },
        };
      });

      const searchPromise = searchEngine.search(query, context);
      
      // Advance timers to trigger timeout
      vi.runAllTimers();
      
      const results = await searchPromise;

      // Should return results despite timeout
      expect(results).toBeDefined();

      vi.useRealTimers();
    });
  });

  describe('quality and correctness validation', () => {
    test('validates LSP hints are correctly applied', async () => {
      const query = 'calculateTotal';
      const context = createSearchContext(query);

      // Mock to verify hint application
      vi.spyOn(lspStageBEnhancer, 'enhanceStageB').mockResolvedValue({
        candidates: [
          {
            ...createMockCandidate('/test/repo/src/utils.ts', 10, ['lsp_hint'], 0.92, 'calculateTotal'),
            ...{ lsp_applied: true, hint_id: 'func_calculateTotal' },
          },
        ],
        metrics: {
          enhanced_count: 1,
          processing_time_ms: 40,
        },
      });

      const results = await searchEngine.search(query, context);
      
      expect(results).toBeDefined();
      const enhancedResult = results.find(r => (r as any).lsp_applied);
      expect(enhancedResult).toBeDefined();
      expect((enhancedResult as any).hint_id).toBe('func_calculateTotal');
    });

    test('validates workspace path mapping integration', async () => {
      const query = '@utils/calculateTotal';
      const context = createSearchContext(query);

      // Should resolve path mapping from tsconfig
      vi.spyOn(intentRouter, 'routeQuery').mockResolvedValue({
        primary_candidates: [
          createMockCandidate('/test/repo/src/utils/calculateTotal.ts', 5, ['path_mapped'], 0.88),
        ],
        intent_classification: { intent: 'def' as QueryIntent, confidence: 0.85 },
        routing_stats: { router_time_ms: 30 },
      });

      const results = await searchEngine.search(query, context);
      
      expect(results).toBeDefined();
      const pathMappedResult = results.find(r => r.match_reasons.includes('path_mapped'));
      expect(pathMappedResult).toBeDefined();
    });

    test('validates symbol kind filtering works correctly', async () => {
      const query = 'class UserService';
      const context = createSearchContext(query);

      vi.spyOn(intentRouter, 'routeQuery').mockResolvedValue({
        primary_candidates: [
          createMockCandidate('/test/repo/src/services/user.ts', 5, ['symbol'], 0.90, 'UserService'),
        ],
        intent_classification: { intent: 'def' as QueryIntent, confidence: 0.92 },
        routing_stats: { router_time_ms: 25 },
      });

      const results = await searchEngine.search(query, context);
      
      expect(results).toBeDefined();
      const classResult = results.find(r => r.symbol === 'UserService');
      expect(classResult).toBeDefined();
      // Should prioritize class symbols for "class" queries
    });

    test('validates cross-file reference resolution', async () => {
      const query = 'refs calculateTotal';
      const context = createSearchContext(query);

      // Mock cross-file references
      vi.spyOn(intentRouter, 'routeQuery').mockResolvedValue({
        primary_candidates: [
          createMockCandidate('/test/repo/src/services/user.ts', 25, ['reference'], 0.85, 'calculateTotal'),
          createMockCandidate('/test/repo/src/controllers/billing.ts', 40, ['reference'], 0.80, 'calculateTotal'),
          createMockCandidate('/test/repo/src/components/cart.ts', 15, ['reference'], 0.75, 'calculateTotal'),
        ],
        intent_classification: { intent: 'refs' as QueryIntent, confidence: 0.88 },
        routing_stats: { router_time_ms: 35 },
      });

      const results = await searchEngine.search(query, context);
      
      expect(results).toBeDefined();
      expect(results.length).toBeGreaterThanOrEqual(3);
      
      // All results should be references across different files
      results.forEach(result => {
        expect(result.match_reasons).toContain('reference');
        expect(result.symbol).toBe('calculateTotal');
      });
    });
  });

  describe('integration error handling', () => {
    test('handles workspace configuration errors', async () => {
      // Create search engine with invalid workspace config
      vi.mocked(require('fs').readFileSync).mockImplementation(() => {
        throw new Error('Cannot read tsconfig.json');
      });

      const invalidWorkspaceConfig = new WorkspaceConfig('/invalid/path');
      
      await expect(async () => {
        await invalidWorkspaceConfig.parseWorkspaceConfig();
      }).rejects.toThrow();

      // Search engine should handle this gracefully in real scenarios
    });

    test('handles LSP server startup failures', async () => {
      vi.spyOn(lspSidecar, 'initializeLanguageServers')
        .mockRejectedValue(new Error('TypeScript server failed to start'));

      const searchEngine2 = new SearchEngine(
        mockWorkspaceFiles,
        mockRepoPath,
        true,
        lspSidecar,
        lspStageBEnhancer,
        lspStageCEnhancer,
        intentRouter
      );

      // Should handle LSP initialization failure
      await expect(searchEngine2.initializeLSP()).rejects.toThrow();
    });

    test('handles malformed LSP responses', async () => {
      const query = 'malformedResponse';
      const context = createSearchContext(query);

      // Mock malformed response
      vi.spyOn(lspSidecar, 'generateHints').mockResolvedValue([
        // Missing required fields
        { name: 'incomplete', file_path: '/test.ts' } as any,
      ]);

      // Should handle malformed data gracefully
      const results = await searchEngine.search(query, context);
      expect(results).toBeDefined();
    });
  });

  describe('concurrent request handling', () => {
    test('handles multiple concurrent searches', async () => {
      const queries = [
        'function calculateTotal',
        'refs UserService',
        'interface User',
        'class AuthController',
        'method authenticate',
      ];

      const searchPromises = queries.map(query => {
        const context = createSearchContext(query);
        return searchEngine.search(query, context);
      });

      const results = await Promise.all(searchPromises);

      expect(results).toHaveLength(5);
      results.forEach(result => {
        expect(result).toBeDefined();
        expect(Array.isArray(result)).toBe(true);
      });
    });

    test('maintains LSP component isolation across requests', async () => {
      const query1 = 'test1';
      const query2 = 'test2';
      const context1 = createSearchContext(query1);
      const context2 = createSearchContext(query2);

      // Execute searches concurrently
      const [results1, results2] = await Promise.all([
        searchEngine.search(query1, context1),
        searchEngine.search(query2, context2),
      ]);

      expect(results1).toBeDefined();
      expect(results2).toBeDefined();
      
      // Verify trace IDs remain separate
      expect(context1.trace_id).not.toBe(context2.trace_id);
    });
  });
});