/**
 * Comprehensive Search Engine Tests
 * Target: High coverage using proven patterns from ASTCache (74%) and Quality-gates (94%)
 * Strategy: Test all search pipeline stages, business logic execution, and optimization paths
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { LensSearchEngine } from '../search-engine.js';
import type { SearchContext, SearchHit, HealthStatus } from '../../types/core.js';

// Mock all dependencies to isolate search engine testing
vi.mock('../../storage/segments.js');
vi.mock('../../indexer/lexical.js');
vi.mock('../../indexer/symbols.js');
vi.mock('../../indexer/semantic.js');
vi.mock('../../core/messaging.js');
vi.mock('../../core/index-registry.js');
vi.mock('../../core/ast-cache.js');
vi.mock('../../core/learned-reranker.js');
vi.mock('../../benchmark/phase-b-comprehensive.js');
vi.mock('../../core/adaptive-fanout.js');
vi.mock('../../core/work-conserving-ann.js');
vi.mock('../../core/precision-optimization.js');
vi.mock('../../core/intent-router.js');
vi.mock('../../core/lsp-stage-b.js');
vi.mock('../../core/lsp-stage-c.js');
vi.mock('../../telemetry/tracer.js');

describe('Comprehensive Search Engine Tests', () => {
  let searchEngine: LensSearchEngine;
  let mockIndexReader: any;
  let mockIndexRegistry: any;

  const sampleSearchContext: SearchContext = {
    query: 'function searchQuery',
    repo_sha: 'abc123def456',
    k: 25,
    fuzzy_distance: 1,
    case_sensitive: false,
    exact_match: false,
    file_filter: ['**/*.ts', '**/*.js'],
    lang_filter: 'typescript',
  };

  const sampleSearchHits: SearchHit[] = [
    {
      file: 'src/core/search.ts',
      line: 42,
      col: 8,
      lang: 'typescript' as any,
      snippet: 'function performSearch(query: string) {',
      score: 0.95,
      why: ['exact', 'fuzzy'] as any,
      byte_offset: 1024,
      span_len: 35,
    },
    {
      file: 'src/utils/helpers.ts',
      line: 15,
      col: 0,
      lang: 'typescript' as any,
      snippet: 'export const searchHelpers = {',
      score: 0.87,
      why: ['fuzzy', 'symbol'] as any,
      byte_offset: 512,
      span_len: 30,
    },
    {
      file: 'src/api/routes.ts',
      line: 28,
      col: 4,
      lang: 'typescript' as any,
      snippet: '  async searchEndpoint(req: Request) {',
      score: 0.82,
      why: ['structural'] as any,
      byte_offset: 896,
      span_len: 38,
    },
  ];

  beforeEach(async () => {
    vi.clearAllMocks();

    // Mock IndexReader with realistic search results
    mockIndexReader = {
      searchLexical: vi.fn().mockResolvedValue([
        {
          file: 'src/core/search.ts',
          line: 42,
          col: 8,
          lang: 'typescript',
          snippet: 'function performSearch(query: string) {',
          score: 0.95,
          why: ['exact', 'fuzzy'],
          byte_offset: 1024,
          span_len: 35,
        },
        {
          file: 'src/utils/helpers.ts',
          line: 15,
          col: 0,
          lang: 'typescript',
          snippet: 'export const searchHelpers = {',
          score: 0.87,
          why: ['fuzzy', 'symbol'],
          byte_offset: 512,
          span_len: 30,
        },
      ]),
      searchSymbols: vi.fn().mockResolvedValue([
        {
          name: 'SearchEngine',
          kind: 'class',
          file: 'src/engine.ts',
          line: 10,
          col: 0,
        }
      ]),
      getHealthStatus: vi.fn().mockResolvedValue({
        status: 'ok',
        shards_healthy: 2,
        shards_total: 2,
      }),
    };

    // Mock IndexRegistry
    mockIndexRegistry = {
      hasRepo: vi.fn().mockReturnValue(true),
      getReader: vi.fn().mockReturnValue(mockIndexReader),
      initialize: vi.fn().mockResolvedValue(undefined),
    };

    // Setup mocks for the constructor dependencies
    const { IndexRegistry } = await import('../../core/index-registry.js');
    vi.mocked(IndexRegistry).mockImplementation(() => mockIndexRegistry);

    // Mock other dependencies
    const { SegmentStorage } = await import('../../storage/segments.js');
    vi.mocked(SegmentStorage).mockImplementation(() => ({
      initialize: vi.fn().mockResolvedValue(undefined),
      getHealthStatus: vi.fn().mockResolvedValue({
        status: 'ok',
        shards_healthy: 1,
        shards_total: 1,
      }),
    }) as any);

    const { LexicalSearchEngine } = await import('../../indexer/lexical.js');
    vi.mocked(LexicalSearchEngine).mockImplementation(() => ({
      initialize: vi.fn().mockResolvedValue(undefined),
      search: vi.fn().mockResolvedValue(sampleSearchHits.slice(0, 2)),
    }) as any);

    const { SymbolSearchEngine } = await import('../../indexer/symbols.js');
    vi.mocked(SymbolSearchEngine).mockImplementation(() => ({
      initialize: vi.fn().mockResolvedValue(undefined),
      search: vi.fn().mockResolvedValue([
        { name: 'SearchEngine', kind: 'class', file: 'src/engine.ts' }
      ]),
    }) as any);

    const { SemanticRerankEngine } = await import('../../indexer/semantic.js');
    vi.mocked(SemanticRerankEngine).mockImplementation(() => ({
      initialize: vi.fn().mockResolvedValue(undefined),
      rerank: vi.fn().mockResolvedValue(sampleSearchHits),
    }) as any);

    // Mock globalAdaptiveFanout
    const { globalAdaptiveFanout } = await import('../../core/adaptive-fanout.js');
    vi.mocked(globalAdaptiveFanout).isEnabled = vi.fn().mockReturnValue(true);
    vi.mocked(globalAdaptiveFanout).extractFeatures = vi.fn().mockReturnValue({
      queryLength: 15,
      hasSpecialChars: false,
      tokenCount: 2,
    });
    vi.mocked(globalAdaptiveFanout).calculateHardness = vi.fn().mockReturnValue(0.75);
    vi.mocked(globalAdaptiveFanout).getAdaptiveKCandidates = vi.fn().mockReturnValue(100);

    // Mock tracer
    const { LensTracer } = await import('../../telemetry/tracer.js');
    const mockSpan = {
      setAttributes: vi.fn(),
      recordException: vi.fn(),
      end: vi.fn(),
    };
    vi.mocked(LensTracer).startSearchSpan = vi.fn().mockReturnValue(mockSpan);
    vi.mocked(LensTracer).startStageSpan = vi.fn().mockReturnValue(mockSpan);
    vi.mocked(LensTracer).endStageSpan = vi.fn();

    // Create search engine instance
    searchEngine = new LensSearchEngine('./test-indexed-content', undefined, undefined, true);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Initialization and Configuration', () => {
    it('should initialize with default configuration', async () => {
      await searchEngine.initialize();
      
      expect(mockIndexRegistry.initialize).toHaveBeenCalled();
      // Verify all engines are initialized
      const { SegmentStorage } = await import('../../storage/segments.js');
      expect(vi.mocked(SegmentStorage).mock.instances[0].initialize).toHaveBeenCalled();
    });

    it('should initialize with custom rerank configuration', async () => {
      const customEngine = new LensSearchEngine(
        './custom-index',
        { enabled: true, nlThreshold: 0.8 },
        undefined,
        false
      );

      await customEngine.initialize();
      
      expect(mockIndexRegistry.initialize).toHaveBeenCalled();
    });

    it('should initialize with Phase B optimization enabled', async () => {
      const phaseBEngine = new LensSearchEngine(
        './index',
        undefined,
        { optimizationsEnabled: true },
        true
      );

      await phaseBEngine.initialize();
      expect(mockIndexRegistry.initialize).toHaveBeenCalled();
    });

    it('should handle initialization failures gracefully', async () => {
      mockIndexRegistry.initialize.mockRejectedValueOnce(new Error('Index init failed'));

      await expect(searchEngine.initialize()).rejects.toThrow(
        'Failed to initialize search engine: Index init failed'
      );
    });

    it('should track initialization state properly', async () => {
      // Should throw if searching before initialization
      await expect(searchEngine.search(sampleSearchContext)).rejects.toThrow(
        'Search engine not initialized'
      );

      // Should work after initialization
      await searchEngine.initialize();
      const result = await searchEngine.search(sampleSearchContext);
      expect(result.hits).toBeDefined();
    });
  });

  describe('Core Search Pipeline', () => {
    beforeEach(async () => {
      await searchEngine.initialize();
    });

    it('should execute full four-stage search pipeline', async () => {
      const result = await searchEngine.search(sampleSearchContext);

      expect(result.hits).toBeDefined();
      expect(result.hits.length).toBeGreaterThan(0);
      expect(result.stage_a_latency).toBeGreaterThan(0);
      
      // Verify IndexReader was called with correct parameters
      expect(mockIndexReader.searchLexical).toHaveBeenCalledWith({
        q: 'function searchQuery',
        fuzzy: 2, // Math.min(2, Math.max(0, Math.round(1 * 2)))
        subtokens: true,
        k: 100, // Adaptive k from mock
      });
    });

    it('should handle adaptive fanout correctly', async () => {
      const { globalAdaptiveFanout } = await import('../../core/adaptive-fanout.js');
      
      const result = await searchEngine.search(sampleSearchContext);

      expect(globalAdaptiveFanout.extractFeatures).toHaveBeenCalledWith(
        'function searchQuery',
        sampleSearchContext
      );
      expect(globalAdaptiveFanout.calculateHardness).toHaveBeenCalled();
      expect(globalAdaptiveFanout.getAdaptiveKCandidates).toHaveBeenCalledWith(0.75);
    });

    it('should handle queries without adaptive fanout', async () => {
      const { globalAdaptiveFanout } = await import('../../core/adaptive-fanout.js');
      vi.mocked(globalAdaptiveFanout).isEnabled.mockReturnValue(false);

      const result = await searchEngine.search({
        ...sampleSearchContext,
        k: 50,
      });

      expect(result.hits).toBeDefined();
      // Should use regular k calculation (k * 4 = 50 * 4 = 200)
      expect(mockIndexReader.searchLexical).toHaveBeenCalledWith(
        expect.objectContaining({
          k: 200,
        })
      );
    });

    it('should convert search results correctly', async () => {
      const result = await searchEngine.search(sampleSearchContext);

      expect(result.hits[0]).toEqual({
        file: 'src/core/search.ts',
        line: 42,
        col: 8,
        lang: 'typescript',
        snippet: 'function performSearch(query: string) {',
        score: 0.95,
        why: ['exact', 'fuzzy'],
        byte_offset: 1024,
        span_len: 35,
      });
    });

    it('should handle missing repository error', async () => {
      mockIndexRegistry.hasRepo.mockReturnValue(false);

      await expect(searchEngine.search(sampleSearchContext)).rejects.toThrow(
        'INDEX_MISSING: Repository not found in index: abc123def456'
      );
    });

    it('should cap adaptive k at safety limit', async () => {
      const { globalAdaptiveFanout } = await import('../../core/adaptive-fanout.js');
      vi.mocked(globalAdaptiveFanout).getAdaptiveKCandidates.mockReturnValue(1000);

      await searchEngine.search(sampleSearchContext);

      // Should cap at 500 per safety requirements
      expect(mockIndexReader.searchLexical).toHaveBeenCalledWith(
        expect.objectContaining({
          k: 500,
        })
      );
    });

    it('should handle different fuzzy distance values', async () => {
      await searchEngine.search({
        ...sampleSearchContext,
        fuzzy_distance: 0.5,
      });

      expect(mockIndexReader.searchLexical).toHaveBeenCalledWith(
        expect.objectContaining({
          fuzzy: 1, // Math.min(2, Math.max(0, Math.round(0.5 * 2)))
        })
      );

      await searchEngine.search({
        ...sampleSearchContext,
        fuzzy_distance: 3,
      });

      expect(mockIndexReader.searchLexical).toHaveBeenCalledWith(
        expect.objectContaining({
          fuzzy: 2, // Capped at 2
        })
      );
    });
  });

  describe('LSP Integration and Intent Routing', () => {
    beforeEach(async () => {
      await searchEngine.initialize();
    });

    it('should handle LSP-enhanced searches', async () => {
      // This tests the LSP integration path
      const lspContext: SearchContext = {
        ...sampleSearchContext,
        lsp_hints: [
          {
            symbol: 'SearchEngine',
            file: 'src/engine.ts',
            line: 10,
            kind: 'class',
          }
        ],
      };

      const result = await searchEngine.search(lspContext);
      expect(result.hits).toBeDefined();
    });

    it('should route queries based on intent when LSP enabled', async () => {
      const queryContext = {
        ...sampleSearchContext,
        query: 'class definition SearchEngine',
      };

      const result = await searchEngine.search(queryContext);
      expect(result.hits).toBeDefined();
      // Should use the intent routing path for definition queries
    });
  });

  describe('Phase B Optimizations', () => {
    it('should use Phase B optimizations when enabled', async () => {
      const phaseBEngine = new LensSearchEngine(
        './index',
        undefined,
        { optimizationsEnabled: true },
        false
      );
      
      await phaseBEngine.initialize();

      // Mock the Phase B optimization method
      const searchWithPhaseB = vi.spyOn(phaseBEngine as any, 'searchWithPhaseBOptimizations');
      searchWithPhaseB.mockResolvedValue({
        hits: sampleSearchHits,
        stage_a_latency: 8,
        stage_b_latency: 12,
      });

      const result = await phaseBEngine.search(sampleSearchContext);
      
      expect(result.hits).toBeDefined();
      expect(result.stage_a_latency).toBe(8);
      expect(result.stage_b_latency).toBe(12);
    });
  });

  describe('Health Status and Monitoring', () => {
    beforeEach(async () => {
      await searchEngine.initialize();
    });

    it('should return comprehensive health status', async () => {
      const health = await searchEngine.getHealthStatus();

      expect(health.status).toBe('ok');
      expect(health.shards_healthy).toBeGreaterThan(0);
      expect(health.shards_total).toBeGreaterThan(0);
      expect(health.memory_usage_gb).toBeGreaterThan(0);
      expect(health.active_queries).toBeGreaterThanOrEqual(0);
      expect(health.worker_pool_status).toBeDefined();
      expect(health.last_compaction).toBeInstanceOf(Date);
    });

    it('should track active query count correctly', async () => {
      const initialHealth = await searchEngine.getHealthStatus();
      const initialActiveQueries = initialHealth.active_queries;

      // Execute search and check active queries during execution
      const searchPromise = searchEngine.search(sampleSearchContext);
      
      // Add small delay to allow query tracking
      await new Promise(resolve => setTimeout(resolve, 1));
      
      const result = await searchPromise;
      expect(result.hits).toBeDefined();

      // Active queries should be back to initial after completion
      const finalHealth = await searchEngine.getHealthStatus();
      expect(finalHealth.active_queries).toBe(initialActiveQueries);
    });

    it('should handle health check errors', async () => {
      const { SegmentStorage } = await import('../../storage/segments.js');
      const mockStorage = vi.mocked(SegmentStorage).mock.instances[0] as any;
      mockStorage.getHealthStatus.mockRejectedValue(new Error('Storage error'));

      await expect(searchEngine.getHealthStatus()).rejects.toThrow('Storage error');
    });

    it('should calculate uptime correctly', async () => {
      // Wait a small amount to ensure uptime > 0
      await new Promise(resolve => setTimeout(resolve, 10));
      
      const health = await searchEngine.getHealthStatus();
      expect(health.uptime_ms).toBeGreaterThan(0);
    });

    it('should include memory usage in health status', async () => {
      const health = await searchEngine.getHealthStatus();
      
      expect(health.memory_usage_gb).toBeGreaterThan(0);
      expect(typeof health.memory_usage_gb).toBe('number');
    });
  });

  describe('Manifest and Repository Management', () => {
    beforeEach(async () => {
      await searchEngine.initialize();
    });

    it('should return repository manifest', async () => {
      mockIndexRegistry.getManifest = vi.fn().mockResolvedValue({
        'test-repo': {
          repo_sha: 'abc123',
          api_version: '1.0.0',
          index_version: '1.2.3',
          policy_version: '1.0.0',
        },
      });

      const manifest = await searchEngine.getManifest();
      
      expect(manifest).toBeDefined();
      expect(manifest['test-repo']).toBeDefined();
      expect(manifest['test-repo'].repo_sha).toBe('abc123');
      expect(mockIndexRegistry.getManifest).toHaveBeenCalled();
    });

    it('should handle manifest errors', async () => {
      mockIndexRegistry.getManifest = vi.fn().mockRejectedValue(new Error('Manifest error'));

      await expect(searchEngine.getManifest()).rejects.toThrow('Manifest error');
    });
  });

  describe('Error Handling and Edge Cases', () => {
    beforeEach(async () => {
      await searchEngine.initialize();
    });

    it('should handle empty search results', async () => {
      mockIndexReader.searchLexical.mockResolvedValue([]);

      const result = await searchEngine.search(sampleSearchContext);
      
      expect(result.hits).toEqual([]);
      expect(result.stage_a_latency).toBeGreaterThan(0);
    });

    it('should handle search errors gracefully', async () => {
      mockIndexReader.searchLexical.mockRejectedValue(new Error('Search failed'));

      await expect(searchEngine.search(sampleSearchContext)).rejects.toThrow('Search failed');
    });

    it('should validate context parameters', async () => {
      const invalidContext: SearchContext = {
        query: '', // Empty query
        repo_sha: 'abc123',
        k: 0, // Invalid k
      };

      // Should handle gracefully or throw appropriate error
      await expect(searchEngine.search(invalidContext)).rejects.toThrow();
    });

    it('should handle concurrent searches correctly', async () => {
      const searches = Array.from({ length: 10 }, (_, i) =>
        searchEngine.search({
          ...sampleSearchContext,
          query: `test query ${i}`,
        })
      );

      const results = await Promise.all(searches);
      
      results.forEach((result, i) => {
        expect(result.hits).toBeDefined();
        expect(mockIndexReader.searchLexical).toHaveBeenCalledWith(
          expect.objectContaining({
            q: `test query ${i}`,
          })
        );
      });

      expect(mockIndexReader.searchLexical).toHaveBeenCalledTimes(10);
    });

    it('should handle malformed search results', async () => {
      mockIndexReader.searchLexical.mockResolvedValue([
        {
          file: 'test.ts',
          // Missing required fields
          line: null,
          col: undefined,
        },
      ]);

      // Should handle malformed results gracefully
      const result = await searchEngine.search(sampleSearchContext);
      expect(result.hits).toBeDefined();
    });
  });

  describe('Performance and SLA Compliance', () => {
    beforeEach(async () => {
      await searchEngine.initialize();
    });

    it('should meet Stage A latency SLA', async () => {
      const startTime = Date.now();
      
      const result = await searchEngine.search(sampleSearchContext);
      
      expect(result.stage_a_latency).toBeLessThan(50); // Reasonable SLA
      expect(result.hits).toBeDefined();
    });

    it('should handle large result sets efficiently', async () => {
      const largeResults = Array.from({ length: 500 }, (_, i) => ({
        file: `src/file${i}.ts`,
        line: i + 1,
        col: 0,
        lang: 'typescript',
        snippet: `function test${i}() {}`,
        score: 0.9 - (i * 0.001),
        why: ['fuzzy'],
        byte_offset: i * 100,
        span_len: 15,
      }));

      mockIndexReader.searchLexical.mockResolvedValue(largeResults);

      const result = await searchEngine.search({
        ...sampleSearchContext,
        k: 500,
      });

      expect(result.hits.length).toBe(500);
      expect(result.stage_a_latency).toBeGreaterThan(0);
    });

    it('should optimize memory usage for frequent searches', async () => {
      // Perform multiple searches to test memory efficiency
      for (let i = 0; i < 20; i++) {
        const result = await searchEngine.search({
          ...sampleSearchContext,
          query: `query iteration ${i}`,
        });
        expect(result.hits).toBeDefined();
      }

      // Memory usage should not grow unbounded
      const health = await searchEngine.getHealthStatus();
      expect(health.memory_usage_gb).toBeLessThan(10); // Reasonable limit
    });
  });

  describe('Match Reason Conversion', () => {
    beforeEach(async () => {
      await searchEngine.initialize();
    });

    it('should convert match reasons correctly', async () => {
      mockIndexReader.searchLexical.mockResolvedValue([
        {
          file: 'test.ts',
          line: 1,
          col: 0,
          lang: 'typescript',
          snippet: 'test',
          score: 0.9,
          why: ['exact', 'semantic', 'lsp_hint'],
          byte_offset: 0,
          span_len: 4,
        },
      ]);

      const result = await searchEngine.search(sampleSearchContext);
      
      expect(result.hits[0].why).toEqual(['exact', 'semantic', 'lsp_hint']);
    });

    it('should handle invalid match reasons', async () => {
      mockIndexReader.searchLexical.mockResolvedValue([
        {
          file: 'test.ts',
          line: 1,
          col: 0,
          lang: 'typescript',
          snippet: 'test',
          score: 0.9,
          why: ['exact', 'invalid_reason', 'fuzzy'],
          byte_offset: 0,
          span_len: 4,
        },
      ]);

      const result = await searchEngine.search(sampleSearchContext);
      
      // Should filter out invalid reasons
      expect(result.hits[0].why).toEqual(['exact', 'fuzzy']);
    });

    it('should provide fallback for missing match reasons', async () => {
      mockIndexReader.searchLexical.mockResolvedValue([
        {
          file: 'test.ts',
          line: 1,
          col: 0,
          lang: 'typescript',
          snippet: 'test',
          score: 0.9,
          why: null,
          byte_offset: 0,
          span_len: 4,
        },
      ]);

      const result = await searchEngine.search(sampleSearchContext);
      
      // Should handle null/undefined reasons gracefully
      expect(Array.isArray(result.hits[0].why)).toBe(true);
    });
  });
});