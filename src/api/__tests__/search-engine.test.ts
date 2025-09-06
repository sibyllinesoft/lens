/**
 * Comprehensive tests for LensSearchEngine
 * Priority: HIGH - Core search functionality with 60 complexity, 885 LOC, no existing tests
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { LensSearchEngine } from '../search-engine.js';
import type { SearchContext } from '../../types/core.js';

// Mock all external dependencies
vi.mock('../../storage/segments.js', () => ({
  SegmentStorage: vi.fn().mockImplementation(() => ({
    listSegments: vi.fn().mockReturnValue([]),
    shutdown: vi.fn().mockResolvedValue(undefined),
  })),
}));

vi.mock('../../indexer/lexical.js', () => ({
  LexicalSearchEngine: vi.fn().mockImplementation(() => ({
    updateConfig: vi.fn().mockResolvedValue(undefined),
  })),
}));

vi.mock('../../indexer/symbols.js', () => ({
  SymbolSearchEngine: vi.fn().mockImplementation(() => ({
    initialize: vi.fn().mockResolvedValue(undefined),
    shutdown: vi.fn().mockResolvedValue(undefined),
  })),
}));

vi.mock('../../indexer/semantic.js', () => ({
  SemanticRerankEngine: vi.fn().mockImplementation(() => ({
    initialize: vi.fn().mockResolvedValue(undefined),
    shutdown: vi.fn().mockResolvedValue(undefined),
    rerankCandidates: vi.fn().mockResolvedValue([]),
    updateConfig: vi.fn().mockResolvedValue(undefined),
  })),
}));

vi.mock('../../core/messaging.js', () => ({
  MessagingSystem: vi.fn().mockImplementation(() => ({
    initialize: vi.fn().mockResolvedValue(undefined),
    shutdown: vi.fn().mockResolvedValue(undefined),
    getWorkerStatus: vi.fn().mockResolvedValue({
      ingest_active: 0,
      query_active: 0,
      maintenance_active: 0,
    }),
  })),
}));

vi.mock('../../core/index-registry.js', () => ({
  IndexRegistry: vi.fn().mockImplementation(() => ({
    refresh: vi.fn().mockResolvedValue(undefined),
    stats: vi.fn().mockReturnValue({ totalRepos: 1, loadedRepos: 1 }),
    hasRepo: vi.fn().mockReturnValue(true),
    getReader: vi.fn().mockImplementation(() => ({
      searchLexical: vi.fn().mockResolvedValue([
        {
          file: 'test.ts',
          line: 1,
          col: 0,
          lang: 'typescript',
          snippet: 'function test() {}',
          score: 0.95,
          why: ['exact_match'],
          byte_offset: 0,
          span_len: 17,
        },
      ]),
      searchStructural: vi.fn().mockResolvedValue([
        {
          file: 'test.ts',
          line: 2,
          col: 0,
          lang: 'typescript',
          snippet: 'class TestClass {}',
          score: 0.90,
          why: ['structural_match'],
          byte_offset: 18,
          span_len: 18,
          pattern_type: 'class',
          symbol_name: 'TestClass',
          signature: 'class TestClass',
        },
      ]),
    })),
    shutdown: vi.fn().mockResolvedValue(undefined),
    getManifests: vi.fn().mockReturnValue([
      {
        repo_ref: 'test-repo',
        repo_sha: 'abc123',
        api_version: '1.0.0',
        index_version: '1.0.0', 
        policy_version: '1.0.0',
        shard_paths: ['test.ts', 'another.ts'],
      },
    ]),
  })),
}));

vi.mock('../../core/ast-cache.js', () => ({
  ASTCache: vi.fn().mockImplementation(() => ({
    getStats: vi.fn().mockReturnValue({
      hits: 10,
      misses: 2,
      size: 5,
      maxSize: 50,
    }),
    getCoverageStats: vi.fn().mockReturnValue({
      coverage_percent: 25.5,
      cached_files: 5,
      total_files: 20,
    }),
  })),
}));

vi.mock('../../core/learned-reranker.js', () => ({
  LearnedReranker: vi.fn().mockImplementation(() => ({
    rerank: vi.fn().mockImplementation(async (hits) => hits),
    updateConfig: vi.fn(),
  })),
}));

vi.mock('../../benchmark/phase-b-comprehensive.js', () => ({
  PhaseBComprehensiveOptimizer: vi.fn().mockImplementation(() => ({
    executeOptimizedSearch: vi.fn().mockResolvedValue({
      hits: [],
      stage_a_latency: 5,
      stage_b_latency: 3,
      stage_c_latency: 7,
    }),
    runComprehensiveBenchmark: vi.fn().mockResolvedValue({
      overall_status: 'PASS',
      stage_a_p95_ms: 8.5,
      meets_performance_targets: true,
      meets_quality_targets: true,
    }),
    generateCalibrationPlotData: vi.fn().mockResolvedValue({
      calibration_error: 0.05,
      reliability_score: 0.92,
      bins: Array(10).fill(0).map((_, i) => ({ predicted: i * 0.1, actual: i * 0.1 })),
    }),
  })),
}));

vi.mock('../../telemetry/tracer.js', () => ({
  LensTracer: {
    createChildSpan: vi.fn().mockReturnValue({
      setAttributes: vi.fn(),
      recordException: vi.fn(),
      end: vi.fn(),
    }),
    startSearchSpan: vi.fn().mockReturnValue({}),
    endSearchSpan: vi.fn(),
    startStageSpan: vi.fn().mockReturnValue({}),
    endStageSpan: vi.fn(),
  },
}));

// Mock global optimization engines
vi.mock('../../core/adaptive-fanout.js', () => ({
  globalAdaptiveFanout: {
    isEnabled: vi.fn().mockReturnValue(false),
    extractFeatures: vi.fn(),
    calculateHardness: vi.fn(),
    getAdaptiveKCandidates: vi.fn(),
    getAdaptiveParameters: vi.fn(),
  },
}));

vi.mock('../../core/work-conserving-ann.js', () => ({
  globalWorkConservingANN: {
    isEnabled: vi.fn().mockReturnValue(false),
    search: vi.fn(),
    updateConfig: vi.fn(),
    setEnabled: vi.fn(),
  },
}));

vi.mock('../../core/precision-optimization.js', () => ({
  globalPrecisionEngine: {
    applyBlockA: vi.fn().mockImplementation(async (hits) => hits),
    applyBlockB: vi.fn().mockImplementation(async (hits) => hits),
    applyBlockC: vi.fn().mockImplementation(async (hits) => hits),
    setBlockEnabled: vi.fn(),
    getOptimizationStatus: vi.fn().mockReturnValue({}),
  },
}));

vi.mock('../../core/span_resolver/index.js', () => ({
  resolveLexicalMatches: vi.fn(),
  resolveSymbolMatches: vi.fn(),
  resolveSemanticMatches: vi.fn().mockImplementation(async (candidates) => 
    candidates.map(c => ({
      file: c.file || 'test.ts',
      line: c.line || 1,
      col: 0,
      lang: 'typescript',
      snippet: c.snippet || 'test code',
      score: c.score || 0.5,
      why: c.why || ['semantic'],
      byte_offset: 0,
      span_len: 10,
    }))
  ),
  prepareSemanticCandidates: vi.fn().mockImplementation((hits, scores) => 
    hits.map((hit, i) => ({ ...hit, score: scores[i] || hit.score }))
  ),
}));

describe('LensSearchEngine', () => {
  let searchEngine: LensSearchEngine;
  let mockSearchContext: SearchContext;

  beforeEach(() => {
    // Reset all mocks
    vi.clearAllMocks();
    
    // Create fresh instance for each test
    searchEngine = new LensSearchEngine();
    
    // Standard test search context
    mockSearchContext = {
      query: 'test function',
      repo_sha: 'abc123',
      mode: 'hybrid',
      k: 10,
      fuzzy_distance: 1,
    };
  });

  afterEach(async () => {
    if (searchEngine) {
      try {
        await searchEngine.shutdown();
      } catch {
        // Ignore shutdown errors in tests
      }
    }
  });

  describe('Initialization', () => {
    it('should initialize successfully with valid repositories', async () => {
      await expect(searchEngine.initialize()).resolves.not.toThrow();
    });

    it('should throw error when no repositories found', async () => {
      const mockIndexRegistry = {
        refresh: vi.fn().mockResolvedValue(undefined),
        stats: vi.fn().mockReturnValue({ totalRepos: 0, loadedRepos: 0 }),
        hasRepo: vi.fn().mockReturnValue(false),
        shutdown: vi.fn().mockResolvedValue(undefined),
        getManifests: vi.fn().mockReturnValue([]),
      };
      
      // Override the mock temporarily
      const originalIndexRegistry = searchEngine['indexRegistry'];
      searchEngine['indexRegistry'] = mockIndexRegistry as any;
      
      await expect(searchEngine.initialize())
        .rejects.toThrow('Failed to initialize search engine: No repositories found in index - cannot start search engine');
        
      // Restore original
      searchEngine['indexRegistry'] = originalIndexRegistry;
    });

    it('should handle initialization failures gracefully', async () => {
      const mockMessaging = {
        initialize: vi.fn().mockRejectedValue(new Error('Messaging failed')),
        shutdown: vi.fn().mockResolvedValue(undefined),
        getWorkerStatus: vi.fn().mockResolvedValue({}),
      };
      
      searchEngine['messaging'] = mockMessaging as any;
      
      await expect(searchEngine.initialize())
        .rejects.toThrow('Failed to initialize search engine: Messaging failed');
    });
  });

  describe('Search Pipeline', () => {
    beforeEach(async () => {
      await searchEngine.initialize();
    });

    it('should throw error when not initialized', async () => {
      const uninitializedEngine = new LensSearchEngine();
      
      await expect(uninitializedEngine.search(mockSearchContext))
        .rejects.toThrow('Search engine not initialized');
    });

    it('should perform lexical search successfully', async () => {
      const result = await searchEngine.search(mockSearchContext);
      
      expect(result).toHaveProperty('hits');
      expect(result).toHaveProperty('stage_a_latency');
      expect(result.hits.length).toBeGreaterThan(0); // May merge lexical and structural results
      expect(result.hits[0]).toMatchObject({
        file: 'test.ts',
        line: 1,
        score: expect.any(Number),
        why: expect.any(Array),
      });
    });

    it('should handle hybrid mode with structural search', async () => {
      const hybridContext = { ...mockSearchContext, mode: 'hybrid' as const };
      const result = await searchEngine.search(hybridContext);
      
      expect(result.hits).toHaveLength(2); // Lexical + structural merged
      expect(result).toHaveProperty('stage_b_latency');
    });

    it('should skip structural search in lexical mode', async () => {
      const lexicalContext = { ...mockSearchContext, mode: 'lex' as const };
      const result = await searchEngine.search(lexicalContext);
      
      expect(result.hits).toHaveLength(1); // Only lexical
      expect(result.stage_b_latency).toBe(0);
    });

    it('should handle repository not found error', async () => {
      const mockIndexRegistry = {
        hasRepo: vi.fn().mockReturnValue(false),
        getReader: vi.fn(),
        refresh: vi.fn().mockResolvedValue(undefined),
        stats: vi.fn().mockReturnValue({ totalRepos: 1, loadedRepos: 1 }),
        shutdown: vi.fn().mockResolvedValue(undefined),
        getManifests: vi.fn().mockReturnValue([]),
      };
      
      searchEngine['indexRegistry'] = mockIndexRegistry as any;
      
      await expect(searchEngine.search(mockSearchContext))
        .rejects.toThrow('INDEX_MISSING: Repository not found in index: abc123');
    });

    it('should limit results to requested k value', async () => {
      const smallKContext = { ...mockSearchContext, k: 1 };
      const result = await searchEngine.search(smallKContext);
      
      expect(result.hits.length).toBeLessThanOrEqual(1);
    });

    it('should handle search errors gracefully', async () => {
      const mockReader = {
        searchLexical: vi.fn().mockRejectedValue(new Error('Search failed')),
        searchStructural: vi.fn(),
      };
      
      const mockIndexRegistry = {
        hasRepo: vi.fn().mockReturnValue(true),
        getReader: vi.fn().mockReturnValue(mockReader),
        refresh: vi.fn().mockResolvedValue(undefined),
        stats: vi.fn().mockReturnValue({ totalRepos: 1, loadedRepos: 1 }),
        shutdown: vi.fn().mockResolvedValue(undefined),
        getManifests: vi.fn().mockReturnValue([]),
      };
      
      searchEngine['indexRegistry'] = mockIndexRegistry as any;
      
      await expect(searchEngine.search(mockSearchContext))
        .rejects.toThrow('Search failed');
    });
  });

  describe('Phase B Optimizations', () => {
    beforeEach(async () => {
      await searchEngine.initialize();
    });

    it('should enable/disable Phase B optimizations', () => {
      expect(() => searchEngine.setPhaseBOptimizationsEnabled(true)).not.toThrow();
      expect(() => searchEngine.setPhaseBOptimizationsEnabled(false)).not.toThrow();
    });

    it('should use optimized search when Phase B enabled', async () => {
      searchEngine.setPhaseBOptimizationsEnabled(true);
      
      const result = await searchEngine.search(mockSearchContext);
      
      expect(result).toHaveProperty('hits');
      expect(result).toHaveProperty('stage_a_latency');
      expect(result).toHaveProperty('stage_b_latency');
      expect(result).toHaveProperty('stage_c_latency');
    });

    it('should run Phase B benchmark successfully', async () => {
      const result = await searchEngine.runPhaseBBenchmark();
      
      expect(result).toMatchObject({
        overall_status: 'PASS',
        stage_a_p95_ms: 8.5,
        meets_performance_targets: true,
        meets_quality_targets: true,
      });
    });

    it('should generate calibration plot data', async () => {
      const result = await searchEngine.generateCalibrationPlot();
      
      expect(result).toHaveProperty('calibration_error');
      expect(result).toHaveProperty('reliability_score');
      expect(result).toHaveProperty('bins');
      expect(Array.isArray(result.bins)).toBe(true);
    });
  });

  describe('Configuration Updates', () => {
    beforeEach(async () => {
      await searchEngine.initialize();
    });

    it('should update Stage-A configuration', async () => {
      const config = {
        rare_term_fuzzy: true,
        synonyms_when_identifier_density_below: 0.8,
        prefilter_enabled: true,
        wand_enabled: false,
        per_file_span_cap: 100,
        native_scanner: 'auto' as const,
      };
      
      await expect(searchEngine.updateStageAConfig(config))
        .resolves.not.toThrow();
    });

    it('should update semantic configuration', async () => {
      const config = {
        nl_threshold: 0.4,
        min_candidates: 15,
        efSearch: 200,
        confidence_cutoff: 0.7,
        ann_k: 50,
        adaptive_gates_enabled: true,
      };
      
      await expect(searchEngine.updateSemanticConfig(config))
        .resolves.not.toThrow();
    });

    it('should handle configuration update errors', async () => {
      const mockLexicalEngine = {
        updateConfig: vi.fn().mockRejectedValue(new Error('Config update failed')),
      };
      
      searchEngine['lexicalEngine'] = mockLexicalEngine as any;
      
      await expect(searchEngine.updateStageAConfig({}))
        .rejects.toThrow('Failed to update Stage-A config: Config update failed');
    });
  });

  describe('Learned Reranking', () => {
    beforeEach(async () => {
      await searchEngine.initialize();
    });

    it('should enable/disable learned reranking', () => {
      expect(() => searchEngine.setRerankingEnabled(true)).not.toThrow();
      expect(() => searchEngine.setRerankingEnabled(false)).not.toThrow();
    });
  });

  describe('System Health and Status', () => {
    beforeEach(async () => {
      await searchEngine.initialize();
    });

    it('should return health status', async () => {
      const health = await searchEngine.getHealthStatus();
      
      expect(health).toHaveProperty('status');
      expect(health).toHaveProperty('shards_healthy');
      expect(health).toHaveProperty('shards_total');
      expect(health).toHaveProperty('memory_usage_gb');
      expect(health).toHaveProperty('active_queries');
      expect(health).toHaveProperty('worker_pool_status');
      expect(health).toHaveProperty('last_compaction');
    });

    it('should return manifest mapping', async () => {
      const manifest = await searchEngine.getManifest();
      
      expect(manifest).toHaveProperty('test-repo');
      expect(manifest['test-repo']).toMatchObject({
        repo_sha: 'abc123',
        api_version: '1.0.0',
        index_version: '1.0.0',
        policy_version: '1.0.0',
      });
    });

    it('should return AST cache coverage stats', () => {
      const stats = searchEngine.getASTCoverageStats();
      
      expect(stats).toHaveProperty('coverage');
      expect(stats).toHaveProperty('stats');
      expect(stats.stats).toMatchObject({
        hits: 10,
        misses: 2,
        size: 5,
        maxSize: 50,
      });
    });
  });

  describe('Precision Optimizations', () => {
    beforeEach(async () => {
      await searchEngine.initialize();
    });

    it('should enable/disable precision optimization blocks', () => {
      expect(() => searchEngine.setPrecisionOptimizationEnabled('A', true)).not.toThrow();
      expect(() => searchEngine.setPrecisionOptimizationEnabled('B', false)).not.toThrow();
      expect(() => searchEngine.setPrecisionOptimizationEnabled('C', true)).not.toThrow();
    });

    it('should return precision optimization status', () => {
      const status = searchEngine.getPrecisionOptimizationStatus();
      expect(status).toBeDefined();
    });
  });

  describe('Error Handling', () => {
    beforeEach(async () => {
      await searchEngine.initialize();
    });

    it('should handle shutdown with active queries', async () => {
      // Simulate active queries
      searchEngine['activeQueries'] = 2;
      
      await expect(searchEngine.shutdown()).resolves.not.toThrow();
    });

    it('should handle shutdown failures', async () => {
      const mockMessaging = {
        initialize: vi.fn().mockResolvedValue(undefined),
        shutdown: vi.fn().mockRejectedValue(new Error('Shutdown failed')),
        getWorkerStatus: vi.fn().mockResolvedValue({}),
      };
      
      searchEngine['messaging'] = mockMessaging as any;
      
      await expect(searchEngine.shutdown())
        .rejects.toThrow('Failed to shutdown search engine: Shutdown failed');
    });
  });

  describe('Search Hit Merging', () => {
    beforeEach(async () => {
      await searchEngine.initialize();
    });

    it('should merge lexical and structural hits correctly', () => {
      const lexicalHits = [
        {
          file: 'test.ts', line: 1, col: 0, lang: 'typescript', 
          snippet: 'function test() {}', score: 0.9, why: ['exact_match'],
          byte_offset: 0, span_len: 17
        }
      ];
      
      const structuralHits = [
        {
          file: 'test.ts', line: 2, col: 0, lang: 'typescript',
          snippet: 'class Test {}', score: 0.8, why: ['structural_match'],
          byte_offset: 18, span_len: 13
        }
      ];
      
      const merged = (searchEngine as any).mergeSearchHits(lexicalHits, structuralHits, 10);
      
      expect(merged).toHaveLength(2);
      expect(merged[0].score).toBeGreaterThanOrEqual(merged[1].score);
    });

    it('should deduplicate identical hits', () => {
      const hit = {
        file: 'test.ts', line: 1, col: 0, lang: 'typescript',
        snippet: 'function test() {}', score: 0.9, why: ['exact_match'],
        byte_offset: 0, span_len: 17
      };
      
      const merged = (searchEngine as any).mergeSearchHits([hit], [hit], 10);
      
      expect(merged).toHaveLength(1);
      expect(merged[0].why).toContain('exact_match');
    });
  });

  describe('Candidate Conversion', () => {
    beforeEach(async () => {
      await searchEngine.initialize();
    });

    it('should convert SearchHits to Candidates correctly', () => {
      const hits = [
        {
          file: 'test.ts', line: 1, col: 0, lang: 'typescript',
          snippet: 'function test() {}', score: 0.9, why: ['exact_match'],
          byte_offset: 0, span_len: 17
        }
      ];
      
      const candidates = (searchEngine as any).convertHitsToCandidates(hits);
      
      expect(candidates).toHaveLength(1);
      expect(candidates[0]).toMatchObject({
        file_path: 'test.ts',
        line: 1,
        col: 0,
        score: 0.9,
        snippet: 'function test() {}',
      });
    });
  });
});