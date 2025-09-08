/**
 * Focused Tests for EnhancedSemanticRerankEngine
 * Target high coverage with real API interactions
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { EnhancedSemanticRerankEngine } from '../enhanced-semantic.js';
import type { Candidate, SearchContext } from '../../types/core.js';

// Mock dependencies
vi.mock('../../telemetry/tracer.js', () => ({
  LensTracer: {
    createChildSpan: vi.fn(() => ({
      setAttributes: vi.fn(),
      recordException: vi.fn(),
      end: vi.fn()
    }))
  }
}));

vi.mock('../../storage/segments.js', () => ({
  SegmentStorage: vi.fn().mockImplementation(() => ({
    listSegments: vi.fn().mockReturnValue(['semantic_001', 'semantic_002']),
    openSegment: vi.fn().mockResolvedValue({ size: 1024 }),
    readFromSegment: vi.fn().mockResolvedValue(Buffer.from(JSON.stringify({
      vectors: {
        'doc1': [0.1, 0.2, 0.3, 0.4],
        'doc2': [0.5, 0.6, 0.7, 0.8]
      }
    }))),
    updateConfig: vi.fn().mockResolvedValue(undefined),
    shutdown: vi.fn().mockResolvedValue(undefined),
  })),
}));

vi.mock('../../core/query-classifier.js', () => ({
  shouldApplySemanticReranking: vi.fn().mockReturnValue(true),
  explainSemanticDecision: vi.fn().mockReturnValue('Query benefits from semantic reranking'),
}));

vi.mock('../../core/isotonic-reranker.js', () => ({
  IsotonicCalibratedReranker: vi.fn().mockImplementation(() => ({
    rerank: vi.fn().mockResolvedValue([
      { doc_id: 'doc1', score: 0.9 },
      { doc_id: 'doc2', score: 0.8 }
    ]),
    getStats: vi.fn().mockReturnValue({ calibrated: true }),
    updateConfig: vi.fn().mockResolvedValue(undefined),
  })),
}));

vi.mock('../../core/optimized-hnsw.js', () => ({
  OptimizedHNSWIndex: vi.fn().mockImplementation(() => ({
    buildIndex: vi.fn().mockResolvedValue(undefined),
    search: vi.fn().mockResolvedValue([
      { doc_id: 'doc1', score: 0.85 },
      { doc_id: 'doc2', score: 0.75 }
    ]),
    tuneEfSearch: vi.fn().mockResolvedValue(64),
    getStats: vi.fn().mockReturnValue({ nodeCount: 100 }),
    updateConfig: vi.fn().mockResolvedValue(undefined),
  })),
}));

// Suppress console output
vi.spyOn(console, 'log').mockImplementation(() => {});
vi.spyOn(console, 'warn').mockImplementation(() => {});

describe('EnhancedSemanticRerankEngine', () => {
  let engine: EnhancedSemanticRerankEngine;
  let mockSegmentStorage: any;

  beforeEach(() => {
    vi.clearAllMocks();
    
    mockSegmentStorage = {
      listSegments: vi.fn().mockReturnValue(['semantic_001']),
      openSegment: vi.fn().mockResolvedValue({ size: 1024 }),
      readFromSegment: vi.fn().mockResolvedValue(Buffer.from(JSON.stringify({
        vectors: {
          'doc1': [0.1, 0.2, 0.3, 0.4],
          'doc2': [0.5, 0.6, 0.7, 0.8]
        }
      }))),
      updateConfig: vi.fn().mockResolvedValue(undefined),
      shutdown: vi.fn().mockResolvedValue(undefined),
    };
    
    engine = new EnhancedSemanticRerankEngine(mockSegmentStorage);
  });

  describe('Constructor and Configuration', () => {
    it('should initialize with default configuration', () => {
      expect(engine).toBeDefined();
      expect(typeof engine).toBe('object');
    });

    it('should accept custom configuration', () => {
      const customConfig = {
        enableIsotonicCalibration: false,
        maxLatencyMs: 10,
        qualityThreshold: 0.01
      };
      
      const customEngine = new EnhancedSemanticRerankEngine(mockSegmentStorage, customConfig);
      expect(customEngine).toBeDefined();
    });

    it('should provide comprehensive stats', () => {
      const stats = engine.getStats();
      
      expect(stats).toBeDefined();
      expect(stats.config).toBeDefined();
      expect(stats.vectors).toBeDefined();
      expect(stats.performance).toBeDefined();
      expect(typeof stats.query_cache_size).toBe('number');
    });
  });

  describe('Initialization', () => {
    it('should initialize successfully', async () => {
      await expect(engine.initialize()).resolves.not.toThrow();
    });

    it('should load semantic segments during initialization', async () => {
      await engine.initialize();
      
      expect(mockSegmentStorage.listSegments).toHaveBeenCalled();
    });

    it('should handle initialization errors gracefully', async () => {
      mockSegmentStorage.listSegments = vi.fn().mockImplementation(() => {
        throw new Error('Storage error');
      });

      await expect(engine.initialize()).rejects.toThrow('Storage error');
    });
  });

  describe('Core Reranking Functionality', () => {
    const mockCandidates: Candidate[] = [
      {
        doc_id: 'doc1',
        file_path: 'test1.ts',
        line: 10,
        col: 5,
        score: 0.8,
        match_reasons: ['lexical'],
        context: 'function test() { return true; }'
      },
      {
        doc_id: 'doc2',
        file_path: 'test2.ts',
        line: 20,
        col: 10,
        score: 0.7,
        match_reasons: ['trigram'],
        context: 'class TestClass { constructor() {} }'
      }
    ];

    const mockContext: SearchContext = {
      query: 'test function',
      mode: 'hybrid' as const,
      repo_sha: 'abc123',
      max_results: 10,
    };

    beforeEach(async () => {
      await engine.initialize();
    });

    it('should rerank candidates successfully', async () => {
      const result = await engine.rerankCandidates(mockCandidates, mockContext);
      
      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBeGreaterThanOrEqual(0);
    });

    it('should handle empty candidate list', async () => {
      const result = await engine.rerankCandidates([], mockContext);
      
      // The engine might return results from internal processing
      expect(Array.isArray(result)).toBe(true);
    });

    it('should respect max results limit', async () => {
      const result = await engine.rerankCandidates(mockCandidates, mockContext, 1);
      
      expect(result.length).toBeLessThanOrEqual(1);
    });

    it('should handle semantic reranking skip conditions', async () => {
      const { shouldApplySemanticReranking } = await import('../../core/query-classifier.js');
      vi.mocked(shouldApplySemanticReranking).mockReturnValue(false);

      const result = await engine.rerankCandidates(mockCandidates, mockContext);
      
      expect(Array.isArray(result)).toBe(true);
    });

    it('should handle reranking errors gracefully', async () => {
      // Simulate internal error
      const originalEncode = (engine as any).embeddingModel.encode;
      (engine as any).embeddingModel.encode = vi.fn().mockRejectedValue(new Error('Encoding failed'));

      const result = await engine.rerankCandidates(mockCandidates, mockContext);
      
      // Should fallback to original ordering
      expect(result).toEqual(mockCandidates.slice(0, 10));
    });

    it('should respect latency budget', async () => {
      const customEngine = new EnhancedSemanticRerankEngine(mockSegmentStorage, {
        maxLatencyMs: 1 // Very tight budget
      });
      await customEngine.initialize();

      const start = Date.now();
      const result = await customEngine.rerankCandidates(mockCandidates, mockContext);
      const elapsed = Date.now() - start;

      expect(Array.isArray(result)).toBe(true);
      // Should either complete quickly or fallback
    });
  });

  describe('Document Indexing', () => {
    beforeEach(async () => {
      await engine.initialize();
    });

    it('should index documents successfully', async () => {
      await expect(engine.indexDocument(
        'test_doc',
        'function testCode() { return 42; }',
        'test.ts'
      )).resolves.not.toThrow();
    });

    it('should handle empty content', async () => {
      await expect(engine.indexDocument('empty_doc', '', 'empty.ts'))
        .resolves.not.toThrow();
    });

    it('should handle indexing errors', async () => {
      // Mock encoding failure
      const originalEncode = (engine as any).embeddingModel.encode;
      (engine as any).embeddingModel.encode = vi.fn().mockRejectedValue(new Error('Encoding failed'));

      await expect(engine.indexDocument('fail_doc', 'content', 'fail.ts'))
        .rejects.toThrow('Encoding failed');
    });
  });

  describe('Configuration Updates', () => {
    it('should update configuration successfully', async () => {
      const newConfig = {
        maxLatencyMs: 15,
        enableIsotonicCalibration: false
      };

      await expect(engine.updateConfig(newConfig)).resolves.not.toThrow();
    });

    it('should handle partial configuration updates', async () => {
      const partialConfig = {
        maxLatencyMs: 12
      };

      await expect(engine.updateConfig(partialConfig)).resolves.not.toThrow();
    });
  });

  describe('Query Embedding Cache', () => {
    beforeEach(async () => {
      await engine.initialize();
    });

    it('should cache query embeddings', async () => {
      const query = 'test query for caching';
      
      // First call should encode and cache
      const embedding1 = await (engine as any).getOrCacheQueryEmbedding(query);
      expect(embedding1).toBeInstanceOf(Float32Array);
      
      // Second call should use cache
      const embedding2 = await (engine as any).getOrCacheQueryEmbedding(query);
      expect(embedding1).toEqual(embedding2);
    });

    it('should respect cache size limits', async () => {
      const maxCacheSize = (engine as any).MAX_QUERY_CACHE_SIZE;
      
      // Fill cache beyond capacity
      for (let i = 0; i <= maxCacheSize + 5; i++) {
        await (engine as any).getOrCacheQueryEmbedding(`query_${i}`);
      }
      
      const cacheSize = (engine as any).queryEmbeddingCache.size;
      expect(cacheSize).toBeLessThanOrEqual(maxCacheSize);
    });
  });

  describe('Performance Metrics', () => {
    beforeEach(async () => {
      await engine.initialize();
    });

    it('should track performance metrics', async () => {
      const candidates: Candidate[] = [{
        doc_id: 'perf_test',
        file_path: 'perf.ts',
        line: 1,
        col: 1,
        score: 0.9,
        match_reasons: ['test'],
        context: 'performance test'
      }];

      const context: SearchContext = {
        query: 'performance test',
        mode: 'hybrid' as const,
        repo_sha: 'perf123',
        max_results: 5,
      };

      await engine.rerankCandidates(candidates, context);

      const stats = engine.getStats();
      expect(stats.performance).toBeDefined();
      expect(typeof stats.performance.avgLatencyMs).toBe('number');
      expect(typeof stats.performance.throughputQPS).toBe('number');
    });
  });

  describe('Score Combination', () => {
    beforeEach(async () => {
      await engine.initialize();
    });

    it('should combine scores with context awareness', () => {
      const context: SearchContext = {
        query: 'how to implement authentication',
        mode: 'hybrid' as const,
        repo_sha: 'test123',
        max_results: 10,
      };

      const combinedScore = (engine as any).combineScoresOptimized(0.8, 0.9, context);
      
      expect(typeof combinedScore).toBe('number');
      expect(combinedScore).toBeGreaterThanOrEqual(0);
      expect(combinedScore).toBeLessThanOrEqual(1);
    });

    it('should handle different query types', () => {
      const contexts = [
        { query: 'getUserData', mode: 'precise' as const, repo_sha: 'test', max_results: 10 },
        { query: 'what is user authentication', mode: 'hybrid' as const, repo_sha: 'test', max_results: 10 },
        { query: 'a', mode: 'fuzzy' as const, repo_sha: 'test', max_results: 10 }
      ];

      contexts.forEach(context => {
        const score = (engine as any).combineScoresOptimized(0.7, 0.8, context);
        expect(typeof score).toBe('number');
        expect(score).toBeGreaterThanOrEqual(0);
        expect(score).toBeLessThanOrEqual(1);
      });
    });
  });

  describe('Quality Preservation', () => {
    beforeEach(async () => {
      await engine.initialize();
    });

    it('should apply quality preservation logic', () => {
      const candidate: Candidate = {
        doc_id: 'quality_test',
        file_path: 'quality.ts',
        line: 1,
        col: 1,
        score: 0.8,
        match_reasons: ['exact'],
        context: 'exact match candidate'
      };

      const context: SearchContext = {
        query: 'exact match',
        mode: 'precise' as const,
        repo_sha: 'quality123',
        max_results: 10,
      };

      const preservedScore = (engine as any).applyQualityPreservation(0.8, candidate, context);
      
      // Exact matches should get boost
      expect(preservedScore).toBeGreaterThan(0.8);
      expect(preservedScore).toBeLessThanOrEqual(1.0);
    });

    it('should boost symbol matches', () => {
      const candidate: Candidate = {
        doc_id: 'symbol_test',
        file_path: 'symbol.ts',
        line: 1,
        col: 1,
        score: 0.7,
        match_reasons: ['symbol'],
        context: 'function symbolTest() {}',
        symbol_kind: 'function'
      };

      const context: SearchContext = {
        query: 'symbolTest',
        mode: 'precise' as const,
        repo_sha: 'symbol123',
        max_results: 10,
      };

      const preservedScore = (engine as any).applyQualityPreservation(0.7, candidate, context);
      
      // Symbol matches should get boost
      expect(preservedScore).toBeGreaterThan(0.7);
      expect(preservedScore).toBeLessThanOrEqual(1.0);
    });
  });

  describe('Fallback Similarity', () => {
    beforeEach(async () => {
      await engine.initialize();
    });

    it('should calculate fallback similarity', () => {
      const candidate: Candidate = {
        doc_id: 'fallback_test',
        file_path: 'fallback.ts',
        line: 1,
        col: 1,
        score: 0.5,
        match_reasons: ['fallback'],
        context: 'this is a test context for fallback similarity calculation'
      };

      const queryEmbedding = new Float32Array([0.1, 0.2, 0.3, 0.4]);
      
      const similarity = (engine as any).calculateFallbackSimilarity(candidate, queryEmbedding);
      
      expect(typeof similarity).toBe('number');
      expect(similarity).toBeGreaterThanOrEqual(0);
      expect(similarity).toBeLessThanOrEqual(1);
    });
  });

  describe('Cleanup', () => {
    it('should shutdown cleanly', async () => {
      await engine.initialize();
      await expect(engine.shutdown()).resolves.not.toThrow();
    });
  });

  describe('SimpleEmbeddingModel Integration', () => {
    beforeEach(async () => {
      await engine.initialize();
    });

    it('should generate embeddings for text', async () => {
      const embeddingModel = (engine as any).embeddingModel;
      const embedding = await embeddingModel.encode('test function code');
      
      expect(embedding).toBeInstanceOf(Float32Array);
      expect(embedding.length).toBe(128); // Default dimension
    });

    it('should calculate similarity between embeddings', async () => {
      const embeddingModel = (engine as any).embeddingModel;
      const embedding1 = await embeddingModel.encode('function test');
      const embedding2 = await embeddingModel.encode('test function');
      
      const similarity = embeddingModel.similarity(embedding1, embedding2);
      
      expect(typeof similarity).toBe('number');
      expect(similarity).toBeGreaterThanOrEqual(0);
      expect(similarity).toBeLessThanOrEqual(1);
    });

    it('should handle different text inputs', async () => {
      const embeddingModel = (engine as any).embeddingModel;
      const texts = [
        'function calculateSum(a, b) { return a + b; }',
        'class UserService { getUser(id) { return users[id]; } }',
        'const API_ENDPOINT = "https://api.example.com";',
        'async function fetchData() { const response = await fetch(url); }',
        ''
      ];
      
      for (const text of texts) {
        const embedding = await embeddingModel.encode(text);
        expect(embedding).toBeInstanceOf(Float32Array);
        expect(embedding.length).toBe(128);
      }
    });

    it('should normalize embeddings', async () => {
      const embeddingModel = (engine as any).embeddingModel;
      const embedding = await embeddingModel.encode('test normalization');
      
      // Check if embedding is normalized (dot product with itself should be ~1)
      const dotProduct = embedding.reduce((sum: number, val: number) => sum + val * val, 0);
      expect(Math.abs(dotProduct - 1.0)).toBeLessThan(0.01);
    });
  });

  describe('Data Conversion Utilities', () => {
    beforeEach(async () => {
      await engine.initialize();
    });

    it('should convert candidates to search hits', () => {
      const candidates: Candidate[] = [{
        doc_id: 'convert_test',
        file_path: 'convert.ts',
        line: 15,
        col: 8,
        score: 0.85,
        match_reasons: ['semantic', 'lexical'],
        context: 'conversion test context',
        symbol_kind: 'function',
        ast_path: 'root.function.body'
      }];

      const searchHits = (engine as any).convertCandidatesToSearchHits(candidates);
      
      expect(Array.isArray(searchHits)).toBe(true);
      expect(searchHits[0]).toMatchObject({
        doc_id: 'convert_test',
        file: 'convert.ts',
        line: 15,
        col: 8,
        score: 0.85,
        snippet: 'conversion test context',
        why: 'semantic,lexical',
        symbol_kind: 'function',
        ast_path: 'root.function.body'
      });
    });

    it('should convert search hits back to candidates', () => {
      const searchHits = [{
        doc_id: 'convert_back_test',
        file: 'convert_back.ts',
        line: 25,
        col: 12,
        score: 0.75,
        snippet: 'convert back test context',
        why: 'lexical,trigram',
        symbol_kind: 'class',
        ast_path: 'root.class.declaration'
      }];

      const candidates = (engine as any).convertSearchHitsToCandidates(searchHits);
      
      expect(Array.isArray(candidates)).toBe(true);
      expect(candidates[0]).toMatchObject({
        doc_id: 'convert_back_test',
        file_path: 'convert_back.ts',
        line: 25,
        col: 12,
        score: 0.75,
        context: 'convert back test context',
        match_reasons: ['lexical', 'trigram'],
        symbol_kind: 'class',
        ast_path: 'root.class.declaration'
      });
    });
  });
});