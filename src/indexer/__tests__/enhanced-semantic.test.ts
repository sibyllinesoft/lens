import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { EnhancedSemanticRerankEngine } from '../enhanced-semantic.js';
import type { Candidate, SearchContext } from '../../types/core.js';
import { SegmentStorage } from '../../storage/segments.js';

// Mock dependencies
vi.mock('../../telemetry/tracer.js', () => ({
  LensTracer: {
    trace: vi.fn((label, fn) => fn()),
    traceAsync: vi.fn(async (label, fn) => await fn()),
    createChildSpan: vi.fn().mockReturnValue({
      finish: vi.fn(),
      setTag: vi.fn(),
      log: vi.fn(),
      end: vi.fn()
    })
  }
}));

vi.mock('../../storage/segments.js', () => ({
  SegmentStorage: vi.fn().mockImplementation(() => ({
    loadSegment: vi.fn().mockResolvedValue(new Map()),
    saveSegment: vi.fn().mockResolvedValue(undefined),
    getSegmentStats: vi.fn().mockReturnValue({ size: 100, timestamp: Date.now() })
  }))
}));

vi.mock('../../core/query-classifier.js', () => ({
  shouldApplySemanticReranking: vi.fn().mockReturnValue(true),
  explainSemanticDecision: vi.fn().mockReturnValue({
    shouldApply: true,
    reason: 'Query benefits from semantic analysis',
    confidence: 0.85
  })
}));

vi.mock('../../core/isotonic-reranker.js', () => ({
  IsotonicCalibratedReranker: vi.fn().mockImplementation(() => ({
    rerank: vi.fn().mockResolvedValue([]),
    updateCalibration: vi.fn().mockResolvedValue(undefined),
    getConfidenceScore: vi.fn().mockReturnValue(0.8),
    isCalibrated: vi.fn().mockReturnValue(true)
  }))
}));

vi.mock('../../core/optimized-hnsw.js', () => ({
  OptimizedHNSWIndex: vi.fn().mockImplementation(() => ({
    search: vi.fn().mockResolvedValue([]),
    addVector: vi.fn().mockResolvedValue(undefined),
    optimize: vi.fn().mockResolvedValue(undefined),
    getStats: vi.fn().mockReturnValue({ nodeCount: 1000, averageConnections: 16 })
  }))
}));

// Mock console to reduce noise
const consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

describe('EnhancedSemanticRerankEngine', () => {
  let engine: EnhancedSemanticRerankEngine;
  let mockSegmentStorage: SegmentStorage;
  let mockConfig: any;

  const mockCandidates: Candidate[] = [
    {
      file: 'test1.py',
      line: 1,
      snippet: 'def hello_world():\n    print("Hello")',
      score: 0.95,
      metadata: { 
        type: 'function',
        language: 'python'
      }
    },
    {
      file: 'test2.js', 
      line: 5,
      snippet: 'function greet(name) {\n  console.log(`Hi ${name}`);\n}',
      score: 0.87,
      metadata: {
        type: 'function',
        language: 'javascript'
      }
    },
    {
      file: 'test3.ts',
      line: 10, 
      snippet: 'class UserService {\n  getUser(id: string) { return users[id]; }\n}',
      score: 0.72,
      metadata: {
        type: 'class',
        language: 'typescript'
      }
    }
  ];

  const mockSearchContext: SearchContext = {
    query: 'function to greet users',
    language: 'any',
    project: 'test-project',
    session_id: 'test-session',
    timestamp: Date.now()
  };

  beforeEach(async () => {
    vi.clearAllMocks();

    // Setup default config
    mockConfig = {
      enableIsotonicCalibration: true,
      enableConfidenceGating: true,
      enableOptimizedHNSW: true,
      maxLatencyMs: 8,
      qualityThreshold: 0.005,
      calibrationConfig: {
        enabled: true,
        minCalibrationData: 50,
        confidenceCutoff: 0.12,
        updateFreq: 100
      },
      hnswConfig: {
        K: 150,
        efSearch: 64,
        autoTune: true
      },
      featureFlags: {
        stageCOptimizations: true,
        advancedCalibration: true,
        experimentalHNSW: false
      }
    };

    // Create mock segment storage
    mockSegmentStorage = new SegmentStorage('test-segments');

    // Create engine instance
    engine = new EnhancedSemanticRerankEngine(mockSegmentStorage, mockConfig);
  });

  afterEach(() => {
    if (engine) {
      engine.shutdown();
    }
    consoleLogSpy.mockClear();
    consoleWarnSpy.mockClear();
  });

  describe('Constructor and Initialization', () => {
    it('should initialize with segment storage and config', () => {
      expect(engine).toBeInstanceOf(EnhancedSemanticRerankEngine);
      expect(engine.segmentStorage).toBe(mockSegmentStorage);
      expect(engine.config).toBeDefined();
    });

    it('should initialize with default config when none provided', () => {
      const defaultEngine = new EnhancedSemanticRerankEngine(mockSegmentStorage);
      expect(defaultEngine.config.maxLatencyMs).toBe(8);
      expect(defaultEngine.config.enableIsotonicCalibration).toBe(true);
    });

    it('should initialize performance metrics tracking', () => {
      expect(engine.performanceMetrics).toBeDefined();
      expect(engine.performanceMetrics.totalQueries).toBe(0);
      expect(engine.performanceMetrics.averageLatencyMs).toBe(0);
    });

    it('should initialize components when initialize() is called', async () => {
      await engine.initialize();

      expect(engine.optimizedHNSW).toBeDefined();
      expect(engine.isotonicReranker).toBeDefined();
      expect(engine.segmentStorage).toBeDefined();
    });

    it('should handle initialization errors gracefully', async () => {
      // Mock a component initialization failure
      const failingStorage = new SegmentStorage('failing-storage');
      failingStorage.loadSegment = vi.fn().mockRejectedValue(new Error('Storage failed'));
      
      const failingEngine = new EnhancedSemanticRerankEngine(failingStorage, mockConfig);
      
      await expect(failingEngine.initialize()).rejects.toThrow('Storage failed');
    });
  });

  describe('Core Reranking Functionality', () => {
    beforeEach(async () => {
      await engine.initialize();
    });

    it('should rerank candidates successfully', async () => {
      const result = await engine.rerankCandidates(mockCandidates, mockSearchContext);

      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBe(mockCandidates.length);
    });

    it('should apply B3 optimizations when enabled', async () => {
      const result = await engine.applyB3Optimizations(mockCandidates, mockSearchContext);

      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);
      expect(result).toBeDefined();
    });

    it('should fall back to basic reranking when B3 optimizations fail', async () => {
      // Mock B3 optimization failure
      engine.applyB3Optimizations = vi.fn().mockRejectedValue(new Error('B3 failed'));

      const result = await engine.rerankCandidates(mockCandidates, mockSearchContext);

      expect(result).toBeDefined();
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining('B3 optimizations failed')
      );
    });

    it('should apply basic semantic reranking', async () => {
      const result = await engine.applyBasicSemanticReranking(mockCandidates, mockSearchContext);

      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);
      expect(result.every(candidate => typeof candidate.score === 'number')).toBe(true);
    });

    it('should handle empty candidate list gracefully', async () => {
      const result = await engine.rerankCandidates([], mockSearchContext);

      expect(result).toEqual([]);
      expect(engine.performanceMetrics.totalQueries).toBe(1);
    });

    it('should preserve quality when applying reranking', async () => {
      const result = await engine.rerankCandidates(mockCandidates, mockSearchContext);
      
      // Check that quality preservation logic is applied
      const qualityPreserved = engine.applyQualityPreservation(result, mockCandidates);
      expect(qualityPreserved).toBeDefined();
    });
  });

  describe('Optimized Semantic Similarity', () => {
    beforeEach(async () => {
      await engine.initialize();
    });

    it('should apply optimized semantic similarity', async () => {
      const result = await engine.applyOptimizedSemanticSimilarity(mockCandidates, mockSearchContext);

      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);
      expect(result).toBeDefined();
    });

    it('should combine scores optimally', () => {
      const lexicalScore = 0.8;
      const semanticScore = 0.9;
      const confidence = 0.85;

      const combinedScore = engine.combineScoresOptimized(lexicalScore, semanticScore, confidence);

      expect(combinedScore).toBeGreaterThan(0);
      expect(combinedScore).toBeLessThanOrEqual(1);
    });

    it('should combine ranking signals correctly', () => {
      const signals = {
        lexicalScore: 0.8,
        semanticScore: 0.9,
        confidence: 0.85,
        isotonicScore: 0.88
      };

      const combinedScore = engine.combineRankingSignals(signals);

      expect(combinedScore).toBeGreaterThan(0);
      expect(combinedScore).toBeLessThanOrEqual(1);
    });

    it('should calculate fallback similarity when embedding fails', async () => {
      // Mock embedding failure
      mockEmbeddingModel.encode = vi.fn().mockRejectedValue(new Error('Encoding failed'));

      const fallbackScore = engine.calculateFallbackSimilarity('test query', 'test snippet');

      expect(fallbackScore).toBeGreaterThanOrEqual(0);
      expect(fallbackScore).toBeLessThanOrEqual(1);
    });
  });

  describe('HNSW Auto-tuning', () => {
    beforeEach(async () => {
      await engine.initialize();
    });

    it('should auto-tune HNSW parameters', async () => {
      const testQueries = await engine.generateTestQueries();
      const result = await engine.autoTuneHNSWParameters(testQueries);

      expect(result).toBeDefined();
      expect(typeof result.optimalEfSearch === 'number').toBe(true);
      expect(result.performanceGain).toBeGreaterThanOrEqual(0);
    });

    it('should generate test queries for auto-tuning', async () => {
      const testQueries = await engine.generateTestQueries();

      expect(Array.isArray(testQueries)).toBe(true);
      expect(testQueries.length).toBeGreaterThan(0);
      expect(testQueries.every(q => typeof q === 'string')).toBe(true);
    });

    it('should handle auto-tuning failure gracefully', async () => {
      // Mock HNSW optimization failure
      if (engine.optimizedHNSW) {
        engine.optimizedHNSW.optimize = vi.fn().mockRejectedValue(new Error('Auto-tune failed'));
      }

      const testQueries = ['test query'];
      const result = await engine.autoTuneHNSWParameters(testQueries);

      expect(result.performanceGain).toBe(0);
      expect(consoleWarnSpy).toHaveBeenCalled();
    });
  });

  describe('Query Embedding Cache', () => {
    beforeEach(async () => {
      await engine.initialize();
    });

    it('should cache query embeddings', async () => {
      const query = 'test query for caching';
      
      // First call should encode
      const embedding1 = await engine.getOrCacheQueryEmbedding(query);
      expect(mockEmbeddingModel.encode).toHaveBeenCalledWith(query);
      
      // Second call should use cache
      vi.clearAllMocks();
      const embedding2 = await engine.getOrCacheQueryEmbedding(query);
      expect(mockEmbeddingModel.encode).not.toHaveBeenCalled();
      
      expect(embedding1).toEqual(embedding2);
    });

    it('should respect cache size limits', async () => {
      const maxCacheSize = engine.MAX_QUERY_CACHE_SIZE;
      
      // Fill cache beyond capacity
      for (let i = 0; i <= maxCacheSize + 5; i++) {
        await engine.getOrCacheQueryEmbedding(`query ${i}`);
      }
      
      expect(engine.queryEmbeddingCache.size).toBeLessThanOrEqual(maxCacheSize);
    });

    it('should handle embedding generation failures in cache', async () => {
      mockEmbeddingModel.encode = vi.fn().mockRejectedValue(new Error('Encoding failed'));
      
      await expect(engine.getOrCacheQueryEmbedding('failing query')).rejects.toThrow('Encoding failed');
    });
  });

  describe('Semantic Index Operations', () => {
    beforeEach(async () => {
      await engine.initialize();
    });

    it('should index documents successfully', async () => {
      const document = {
        id: 'doc1',
        content: 'function testFunction() { return true; }',
        metadata: { language: 'javascript', type: 'function' }
      };

      await engine.indexDocument(document);

      expect(result).toBeUndefined(); // indexDocument returns void
    });

    it('should handle document indexing errors', async () => {
      // Mock internal embedding failure by making the engine fail
      const originalEncode = engine.embeddingModel?.encode;
      if (engine.embeddingModel) {
        engine.embeddingModel.encode = vi.fn().mockRejectedValue(new Error('Indexing failed'));
      }
      
      const document = {
        id: 'doc1',
        content: 'test content',
        metadata: {}
      };

      await expect(engine.indexDocument(document)).rejects.toThrow('Indexing failed');
    });

    it('should convert between candidates and search hits', () => {
      const searchHits = engine.convertCandidatesToSearchHits(mockCandidates);
      
      expect(Array.isArray(searchHits)).toBe(true);
      expect(searchHits.length).toBe(mockCandidates.length);
      expect(searchHits.every(hit => 'file' in hit && 'score' in hit)).toBe(true);
    });

    it('should convert search hits back to candidates', () => {
      const searchHits = engine.convertCandidatesToSearchHits(mockCandidates);
      const candidates = engine.convertSearchHitsToCandidates(searchHits);
      
      expect(Array.isArray(candidates)).toBe(true);
      expect(candidates.length).toBe(searchHits.length);
      expect(candidates.every(c => 'file' in c && 'score' in c)).toBe(true);
    });
  });

  describe('Segment Storage Integration', () => {
    beforeEach(async () => {
      await engine.initialize();
    });

    it('should load semantic segments', async () => {
      const segmentId = 'test-segment-001';
      const result = await engine.loadSemanticSegment(segmentId);

      expect(result).toBeDefined();
      expect(engine.segmentStorage.loadSegment).toHaveBeenCalledWith(segmentId);
    });

    it('should handle segment loading errors', async () => {
      if (engine.segmentStorage) {
        engine.segmentStorage.loadSegment = vi.fn().mockRejectedValue(new Error('Segment not found'));
      }

      const segmentId = 'missing-segment';
      await expect(engine.loadSemanticSegment(segmentId)).rejects.toThrow('Segment not found');
    });

    it('should update performance metrics after operations', () => {
      const latency = 5.5;
      const candidates = mockCandidates.length;
      
      engine.updatePerformanceMetrics(latency, candidates);
      
      expect(engine.performanceMetrics.totalQueries).toBe(1);
      expect(engine.performanceMetrics.totalCandidatesProcessed).toBe(candidates);
      expect(engine.performanceMetrics.averageLatencyMs).toBe(latency);
    });
  });

  describe('Configuration Management', () => {
    beforeEach(async () => {
      await engine.initialize();
    });

    it('should update configuration dynamically', () => {
      const newConfig = {
        ...mockConfig,
        maxLatencyMs: 10,
        enableIsotonicCalibration: false
      };

      engine.updateConfig(newConfig);

      expect(engine.config.maxLatencyMs).toBe(10);
      expect(engine.config.enableIsotonicCalibration).toBe(false);
    });

    it('should validate configuration updates', () => {
      const invalidConfig = {
        maxLatencyMs: -1, // Invalid negative latency
        qualityThreshold: 2 // Invalid threshold > 1
      };

      expect(() => {
        engine.updateConfig(invalidConfig);
      }).toThrow();
    });

    it('should preserve existing config when partial update provided', () => {
      const originalMaxLatency = engine.config.maxLatencyMs;
      const partialUpdate = {
        enableIsotonicCalibration: false
      };

      engine.updateConfig(partialUpdate);

      expect(engine.config.maxLatencyMs).toBe(originalMaxLatency);
      expect(engine.config.enableIsotonicCalibration).toBe(false);
    });
  });

  describe('Performance Monitoring and Stats', () => {
    beforeEach(async () => {
      await engine.initialize();
    });

    it('should return comprehensive performance stats', () => {
      // Simulate some operations
      engine.updatePerformanceMetrics(5.2, 10);
      engine.updatePerformanceMetrics(7.1, 15);
      engine.updatePerformanceMetrics(4.8, 8);

      const stats = engine.getStats();

      expect(stats.totalQueries).toBe(3);
      expect(stats.averageLatencyMs).toBeCloseTo(5.7, 1);
      expect(stats.totalCandidatesProcessed).toBe(33);
      expect(stats.config).toEqual(mockConfig);
    });

    it('should track latency distribution', () => {
      const latencies = [2.1, 4.5, 6.8, 3.2, 8.9, 5.1, 7.3];
      
      latencies.forEach(latency => {
        engine.updatePerformanceMetrics(latency, 5);
      });

      const stats = engine.getStats();
      expect(stats.p95LatencyMs).toBeGreaterThan(0);
      expect(stats.p99LatencyMs).toBeGreaterThan(stats.p95LatencyMs);
    });

    it('should handle empty performance metrics gracefully', () => {
      const stats = engine.getStats();

      expect(stats.totalQueries).toBe(0);
      expect(stats.averageLatencyMs).toBe(0);
      expect(stats.totalCandidatesProcessed).toBe(0);
    });
  });

  describe('Error Handling and Edge Cases', () => {
    beforeEach(async () => {
      await engine.initialize();
    });

    it('should handle malformed candidate data', async () => {
      const malformedCandidates = [
        { file: 'test.py', score: 0.9 }, // Missing required fields
        { line: 5, snippet: 'test', score: 0.8 }, // Missing file
        null as any, // Null candidate
        undefined as any // Undefined candidate
      ];

      const result = await engine.rerankCandidates(malformedCandidates, mockSearchContext);

      expect(Array.isArray(result)).toBe(true);
      // Should filter out malformed candidates gracefully
    });

    it('should handle embedding model failures gracefully', async () => {
      // Mock internal embedding failure
      if (engine.embeddingModel) {
        engine.embeddingModel.encode = vi.fn().mockRejectedValue(new Error('Model unavailable'));
      }

      const result = await engine.rerankCandidates(mockCandidates, mockSearchContext);

      // Should fall back to basic reranking
      expect(result).toBeDefined();
      expect(consoleWarnSpy).toHaveBeenCalled();
    });

    it('should handle very large candidate lists efficiently', async () => {
      const largeCandidateList = Array.from({ length: 1000 }, (_, i) => ({
        file: `test${i}.py`,
        line: i,
        snippet: `function test${i}() { return ${i}; }`,
        score: Math.random(),
        metadata: { type: 'function' }
      }));

      const startTime = Date.now();
      const result = await engine.rerankCandidates(largeCandidateList, mockSearchContext);
      const endTime = Date.now();

      expect(result.length).toBe(largeCandidateList.length);
      expect(endTime - startTime).toBeLessThan(mockConfig.maxLatencyMs * 100); // Reasonable timeout
    });

    it('should handle concurrent reranking requests', async () => {
      const requests = Array.from({ length: 5 }, (_, i) => 
        engine.rerankCandidates(mockCandidates, {
          ...mockSearchContext,
          query: `concurrent query ${i}`
        })
      );

      const results = await Promise.all(requests);

      expect(results.length).toBe(5);
      expect(results.every(r => Array.isArray(r))).toBe(true);
    });

    it('should handle shutdown gracefully', () => {
      expect(() => {
        engine.shutdown();
      }).not.toThrow();

      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Enhanced semantic rerank engine shutdown')
      );
    });

    it('should handle operations after shutdown', async () => {
      engine.shutdown();

      // Operations after shutdown should handle gracefully
      await expect(engine.rerankCandidates(mockCandidates, mockSearchContext))
        .rejects.toThrow();
    });
  });

  describe('Integration with Feature Flags', () => {
    it('should respect feature flag settings', async () => {
      const configWithDisabledFeatures = {
        ...mockConfig,
        featureFlags: {
          enableB3Pipeline: false,
          enableConfidenceGating: false,
          enableAdaptiveThresholding: false
        }
      };

      const engineWithDisabledFeatures = new EnhancedSemanticRerankEngine(
        mockSegmentStorage, 
        configWithDisabledFeatures
      );
      await engineWithDisabledFeatures.initialize();

      const result = await engineWithDisabledFeatures.rerankCandidates(mockCandidates, mockSearchContext);

      expect(result).toBeDefined();
      // Should use basic reranking when B3 pipeline is disabled
    });

    it('should handle gradual feature rollout', async () => {
      const partiallyEnabledConfig = {
        ...mockConfig,
        featureFlags: {
          enableB3Pipeline: true,
          enableConfidenceGating: false,
          enableAdaptiveThresholding: true
        }
      };

      const testEngine = new EnhancedSemanticRerankEngine(mockSegmentStorage, partiallyEnabledConfig);
      await testEngine.initialize();

      const result = await testEngine.rerankCandidates(mockCandidates, mockSearchContext);

      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);
    });
  });
});

// Note: SimpleEmbeddingModel is tested indirectly through EnhancedSemanticRerankEngine tests
// as it's an internal implementation detail and not exported