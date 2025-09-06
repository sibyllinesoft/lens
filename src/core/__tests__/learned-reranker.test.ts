/**
 * Tests for Learned Reranker
 * Covers pairwise logistic regression, feature extraction, reranking logic, and performance optimization
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import {
  LearnedReranker,
  RerankingFeatures,
  RerankingConfig,
  extractRerankingFeatures,
  computeRerankingScore,
  DEFAULT_LEARNED_WEIGHTS,
  LEARNED_WEIGHTS,
} from '../learned-reranker.js';
import type { SearchHit, SearchContext } from '../types/core.js';

// Mock the tracer
vi.mock('../../telemetry/tracer.js', () => ({
  LensTracer: {
    createChildSpan: vi.fn(() => ({
      setAttributes: vi.fn(),
      setStatus: vi.fn(),
      end: vi.fn(),
    })),
  },
}));

describe('Learned Reranker', () => {
  let reranker: LearnedReranker;
  let mockConfig: RerankingConfig;
  let mockHits: SearchHit[];
  let mockContext: SearchContext;

  beforeEach(() => {
    vi.clearAllMocks();
    
    mockConfig = {
      enabled: true,
      nlThreshold: 0.5,
      minCandidates: 3,
      maxLatencyMs: 12,
    };
    
    reranker = new LearnedReranker(mockConfig);
    
    mockHits = [
      {
        file: 'src/auth/authenticate.ts',
        start_line: 10,
        end_line: 15,
        snippet: 'export function authenticate(user: User): boolean { return user.isValid(); }',
        score: 0.9,
        span_id: 'auth_span_1',
        exact_match: true,
        symbol_name: 'authenticate',
      },
      {
        file: 'lib/utils/helper.js',
        start_line: 25,
        end_line: 30,
        snippet: 'function helper() { /* utility function */ }',
        score: 0.7,
        span_id: 'utils_span_1',
        exact_match: false,
        symbol_name: 'helper',
      },
      {
        file: 'tests/auth.test.ts',
        start_line: 5,
        end_line: 12,
        snippet: 'describe("authentication", () => { it("should authenticate user", () => {}); });',
        score: 0.6,
        span_id: 'test_span_1',
        exact_match: false,
        symbol_name: 'describe',
      },
      {
        file: 'docs/README.md',
        start_line: 1,
        end_line: 3,
        snippet: '# Authentication Guide\nThis guide explains user authentication.',
        score: 0.4,
        span_id: 'docs_span_1',
        exact_match: false,
      },
    ];
    
    mockContext = {
      query: 'user authentication function',
      mode: 'hybrid',
      max_results: 10,
      include_snippets: true,
      filters: {},
      nl_likelihood: 0.8, // High natural language likelihood
    };
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Feature Extraction', () => {
    it('should extract exactness features correctly', () => {
      const exactHit = { ...mockHits[0], exact_match: true };
      const fuzzyHit = { ...mockHits[1], exact_match: false };
      
      const exactFeatures = extractRerankingFeatures(exactHit, mockContext);
      const fuzzyFeatures = extractRerankingFeatures(fuzzyHit, mockContext);
      
      expect(exactFeatures.exactness).toBeGreaterThan(fuzzyFeatures.exactness);
      expect(exactFeatures.exactness).toBeGreaterThanOrEqual(0.8);
      expect(fuzzyFeatures.exactness).toBeLessThan(0.8);
    });

    it('should calculate symbol proximity features', () => {
      const authHit = mockHits[0]; // Contains 'authenticate' - close to query
      const helperHit = mockHits[1]; // Contains 'helper' - less relevant
      
      const authFeatures = extractRerankingFeatures(authHit, mockContext);
      const helperFeatures = extractRerankingFeatures(helperHit, mockContext);
      
      expect(authFeatures.symbolProximity).toBeGreaterThan(helperFeatures.symbolProximity);
      expect(authFeatures.symbolProximity).toBeGreaterThan(0.7);
      expect(helperFeatures.symbolProximity).toBeLessThan(0.5);
    });

    it('should detect structural hits', () => {
      const structuralHit = {
        ...mockHits[0],
        snippet: 'class Authentication { login() { } logout() { } }',
        symbol_name: 'Authentication',
      };
      
      const features = extractRerankingFeatures(structuralHit, mockContext);
      
      expect(features.structHit).toBeGreaterThan(0.7); // Should detect class structure
    });

    it('should calculate path prior scores', () => {
      const srcHit = mockHits[0]; // src/auth/
      const libHit = mockHits[1]; // lib/utils/
      const testHit = mockHits[2]; // tests/
      const docHit = mockHits[3]; // docs/
      
      const srcFeatures = extractRerankingFeatures(srcHit, mockContext);
      const libFeatures = extractRerankingFeatures(libHit, mockContext);
      const testFeatures = extractRerankingFeatures(testHit, mockContext);
      const docFeatures = extractRerankingFeatures(docHit, mockContext);
      
      // src/ should have highest priority
      expect(srcFeatures.pathPrior).toBeGreaterThanOrEqual(libFeatures.pathPrior);
      expect(libFeatures.pathPrior).toBeGreaterThanOrEqual(testFeatures.pathPrior);
      expect(testFeatures.pathPrior).toBeGreaterThan(docFeatures.pathPrior);
    });

    it('should normalize snippet length appropriately', () => {
      const shortHit = {
        ...mockHits[0],
        snippet: 'auth()',
      };
      
      const longHit = {
        ...mockHits[1],
        snippet: 'function authenticate(user: User, options: AuthOptions): Promise<AuthResult> { /* very long implementation with detailed logic */ }',
      };
      
      const shortFeatures = extractRerankingFeatures(shortHit, mockContext);
      const longFeatures = extractRerankingFeatures(longHit, mockContext);
      
      expect(shortFeatures.snippetLength).toBeLessThan(longFeatures.snippetLength);
      expect(shortFeatures.snippetLength).toBeGreaterThanOrEqual(0);
      expect(longFeatures.snippetLength).toBeLessThanOrEqual(1);
    });

    it('should use context natural language likelihood', () => {
      const features = extractRerankingFeatures(mockHits[0], mockContext);
      
      expect(features.nlLikelihood).toBe(mockContext.nl_likelihood);
      expect(features.nlLikelihood).toBe(0.8);
    });

    it('should handle missing or malformed hit data gracefully', () => {
      const malformedHit = {
        file: 'test.js',
        start_line: 1,
        end_line: 2,
        snippet: '',
        score: NaN,
        span_id: 'test_span',
      } as any;
      
      expect(() => extractRerankingFeatures(malformedHit, mockContext)).not.toThrow();
      
      const features = extractRerankingFeatures(malformedHit, mockContext);
      expect(features.exactness).toBeGreaterThanOrEqual(0);
      expect(features.symbolProximity).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Reranking Score Computation', () => {
    it('should compute weighted feature scores', () => {
      const features: RerankingFeatures = {
        exactness: 0.9,
        symbolProximity: 0.8,
        structHit: 0.7,
        pathPrior: 0.6,
        snippetLength: 0.5,
        nlLikelihood: 0.8,
      };
      
      const score = computeRerankingScore(features, LEARNED_WEIGHTS);
      
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
      
      // Should be high given all good features
      expect(score).toBeGreaterThan(0.7);
    });

    it('should handle zero features appropriately', () => {
      const zeroFeatures: RerankingFeatures = {
        exactness: 0,
        symbolProximity: 0,
        structHit: 0,
        pathPrior: 0,
        snippetLength: 0,
        nlLikelihood: 0,
      };
      
      const score = computeRerankingScore(zeroFeatures, LEARNED_WEIGHTS);
      
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThan(0.5); // Should be low
    });

    it('should weight exactness most heavily by default', () => {
      const exactnessFeatures: RerankingFeatures = {
        exactness: 1.0,
        symbolProximity: 0,
        structHit: 0,
        pathPrior: 0,
        snippetLength: 0,
        nlLikelihood: 0.5,
      };
      
      const proximityFeatures: RerankingFeatures = {
        exactness: 0,
        symbolProximity: 1.0,
        structHit: 0,
        pathPrior: 0,
        snippetLength: 0,
        nlLikelihood: 0.5,
      };
      
      const exactnessScore = computeRerankingScore(exactnessFeatures, LEARNED_WEIGHTS);
      const proximityScore = computeRerankingScore(proximityFeatures, LEARNED_WEIGHTS);
      
      expect(exactnessScore).toBeGreaterThan(proximityScore);
    });

    it('should be deterministic for same inputs', () => {
      const features: RerankingFeatures = {
        exactness: 0.7,
        symbolProximity: 0.6,
        structHit: 0.5,
        pathPrior: 0.4,
        snippetLength: 0.3,
        nlLikelihood: 0.8,
      };
      
      const score1 = computeRerankingScore(features, LEARNED_WEIGHTS);
      const score2 = computeRerankingScore(features, LEARNED_WEIGHTS);
      
      expect(score1).toBe(score2);
    });
  });

  describe('Reranking Logic', () => {
    it('should rerank hits when conditions are met', async () => {
      const result = await reranker.rerank(mockHits, mockContext);
      
      expect(result.rerank_applied).toBe(true);
      expect(result.hits).toHaveLength(mockHits.length);
      expect(result.processing_time_ms).toBeDefined();
      expect(result.processing_time_ms).toBeGreaterThan(0);
      
      // Hits should be reordered based on learned features
      const originalOrder = mockHits.map(hit => hit.span_id);
      const rerankedOrder = result.hits.map(hit => hit.span_id);
      
      // At least some reordering should occur with good features
      const orderChanged = !originalOrder.every((id, index) => id === rerankedOrder[index]);
      expect(orderChanged).toBe(true);
    });

    it('should preserve original scores but add rerank scores', async () => {
      const result = await reranker.rerank(mockHits, mockContext);
      
      result.hits.forEach((hit, index) => {
        expect(hit.original_score).toBe(mockHits[index].score);
        expect(hit.rerank_score).toBeDefined();
        expect(hit.rerank_score).toBeGreaterThanOrEqual(0);
        expect(hit.rerank_score).toBeLessThanOrEqual(1);
      });
    });

    it('should skip reranking when NL threshold not met', async () => {
      const lowNlContext = {
        ...mockContext,
        nl_likelihood: 0.3, // Below threshold of 0.5
      };
      
      const result = await reranker.rerank(mockHits, lowNlContext);
      
      expect(result.rerank_applied).toBe(false);
      expect(result.hits).toEqual(mockHits);
      expect(result.skip_reason).toMatch(/natural language threshold/i);
    });

    it('should skip reranking with insufficient candidates', async () => {
      const fewHits = mockHits.slice(0, 2); // Only 2 hits, below minCandidates of 3
      
      const result = await reranker.rerank(fewHits, mockContext);
      
      expect(result.rerank_applied).toBe(false);
      expect(result.hits).toEqual(fewHits);
      expect(result.skip_reason).toMatch(/insufficient candidates/i);
    });

    it('should skip reranking when disabled', async () => {
      const disabledReranker = new LearnedReranker({ ...mockConfig, enabled: false });
      
      const result = await disabledReranker.rerank(mockHits, mockContext);
      
      expect(result.rerank_applied).toBe(false);
      expect(result.hits).toEqual(mockHits);
      expect(result.skip_reason).toMatch(/disabled/i);
    });

    it('should respect latency budget', async () => {
      const strictLatencyConfig = { ...mockConfig, maxLatencyMs: 1 }; // Very strict
      const fastReranker = new LearnedReranker(strictLatencyConfig);
      
      const start = performance.now();
      const result = await fastReranker.rerank(mockHits, mockContext);
      const duration = performance.now() - start;
      
      // Should either complete fast or skip due to budget
      if (result.rerank_applied) {
        expect(duration).toBeLessThan(strictLatencyConfig.maxLatencyMs * 2); // Allow some margin
      } else {
        expect(result.skip_reason).toMatch(/latency/i);
      }
    });

    it('should handle empty hit list', async () => {
      const result = await reranker.rerank([], mockContext);
      
      expect(result.hits).toEqual([]);
      expect(result.rerank_applied).toBe(false);
      expect(result.processing_time_ms).toBeDefined();
    });

    it('should handle single hit gracefully', async () => {
      const singleHit = [mockHits[0]];
      const result = await reranker.rerank(singleHit, mockContext);
      
      expect(result.rerank_applied).toBe(false); // Below minCandidates
      expect(result.hits).toEqual(singleHit);
    });
  });

  describe('Training and Learning', () => {
    it('should accept training data for relevance learning', () => {
      const features: RerankingFeatures = {
        exactness: 0.9,
        symbolProximity: 0.8,
        structHit: 0.7,
        pathPrior: 0.6,
        snippetLength: 0.5,
        nlLikelihood: 0.8,
      };
      
      expect(() => reranker.addTrainingData(features, 1.0)).not.toThrow();
      expect(() => reranker.addTrainingData(features, 0.0)).not.toThrow();
      
      expect(reranker.getTrainingDataSize()).toBe(2);
    });

    it('should limit training data size to prevent memory bloat', () => {
      // Add more training data than the limit
      for (let i = 0; i < 6000; i++) {
        reranker.addTrainingData({
          exactness: Math.random(),
          symbolProximity: Math.random(),
          structHit: Math.random(),
          pathPrior: Math.random(),
          snippetLength: Math.random(),
          nlLikelihood: Math.random(),
        }, Math.random());
      }
      
      // Should be trimmed to keep memory bounded
      expect(reranker.getTrainingDataSize()).toBeLessThanOrEqual(5000);
    });

    it('should retrain model periodically with new data', async () => {
      // Add some training data
      const trainingData = [
        { features: { exactness: 0.9, symbolProximity: 0.8, structHit: 0.7, pathPrior: 0.6, snippetLength: 0.5, nlLikelihood: 0.8 }, relevance: 1.0 },
        { features: { exactness: 0.3, symbolProximity: 0.2, structHit: 0.1, pathPrior: 0.4, snippetLength: 0.3, nlLikelihood: 0.6 }, relevance: 0.2 },
        { features: { exactness: 0.8, symbolProximity: 0.7, structHit: 0.6, pathPrior: 0.5, snippetLength: 0.4, nlLikelihood: 0.7 }, relevance: 0.9 },
      ];
      
      trainingData.forEach(({ features, relevance }) => 
        reranker.addTrainingData(features, relevance)
      );
      
      // Get initial reranking result
      const initialResult = await reranker.rerank(mockHits, mockContext);
      
      // Force retraining by adding more data
      for (let i = 0; i < 100; i++) {
        reranker.addTrainingData({
          exactness: 0.1, // Low exactness
          symbolProximity: 0.1,
          structHit: 0.1,
          pathPrior: 0.1,
          snippetLength: 0.1,
          nlLikelihood: 0.8,
        }, 0.05); // Low relevance
      }
      
      // Get result after retraining
      const retrainedResult = await reranker.rerank(mockHits, mockContext);
      
      // Results should potentially be different due to updated model
      // (though this is hard to test definitively without knowing internal state)
      expect(retrainedResult.hits).toHaveLength(mockHits.length);
    });

    it('should provide model statistics and diagnostics', () => {
      // Add some training data
      for (let i = 0; i < 50; i++) {
        reranker.addTrainingData({
          exactness: Math.random(),
          symbolProximity: Math.random(),
          structHit: Math.random(),
          pathPrior: Math.random(),
          snippetLength: Math.random(),
          nlLikelihood: Math.random(),
        }, Math.random());
      }
      
      const stats = reranker.getModelStatistics();
      
      expect(stats.training_samples).toBe(50);
      expect(stats.model_version).toBeDefined();
      expect(stats.last_trained).toBeDefined();
      expect(stats.feature_importance).toBeDefined();
      
      // Feature importance should reflect learned weights
      expect(stats.feature_importance.exactness).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Performance Optimization', () => {
    it('should meet latency targets for typical workloads', async () => {
      // Create larger hit set
      const largeHitSet = Array.from({ length: 100 }, (_, i) => ({
        file: `src/module${i}.ts`,
        start_line: i * 10,
        end_line: i * 10 + 5,
        snippet: `function func${i}() { return ${i}; }`,
        score: 0.9 - i * 0.005,
        span_id: `span_${i}`,
        exact_match: i % 10 === 0, // Every 10th is exact match
        symbol_name: `func${i}`,
      }));
      
      const start = performance.now();
      const result = await reranker.rerank(largeHitSet, mockContext);
      const duration = performance.now() - start;
      
      // Should complete within latency budget
      expect(duration).toBeLessThan(mockConfig.maxLatencyMs * 3); // Allow some margin
      
      if (result.rerank_applied) {
        expect(result.hits).toHaveLength(largeHitSet.length);
      }
    });

    it('should cache feature computations for efficiency', async () => {
      // First rerank
      const start1 = performance.now();
      const result1 = await reranker.rerank(mockHits, mockContext);
      const duration1 = performance.now() - start1;
      
      // Second rerank with same hits (should use cached features)
      const start2 = performance.now();
      const result2 = await reranker.rerank(mockHits, mockContext);
      const duration2 = performance.now() - start2;
      
      // Second run should be faster or similar due to caching
      expect(duration2).toBeLessThanOrEqual(duration1 * 1.2); // Allow 20% variance
      
      if (result1.rerank_applied && result2.rerank_applied) {
        expect(result2.hits).toEqual(result1.hits);
      }
    });

    it('should handle concurrent reranking requests', async () => {
      const promises = Array.from({ length: 5 }, () => 
        reranker.rerank(mockHits, mockContext)
      );
      
      const results = await Promise.all(promises);
      
      // All results should be consistent
      results.forEach(result => {
        expect(result.hits).toHaveLength(mockHits.length);
        expect(result.processing_time_ms).toBeDefined();
      });
      
      // If reranking was applied, results should be identical
      const appliedResults = results.filter(r => r.rerank_applied);
      if (appliedResults.length > 1) {
        const firstResult = appliedResults[0];
        appliedResults.slice(1).forEach(result => {
          expect(result.hits.map(h => h.span_id)).toEqual(
            firstResult.hits.map(h => h.span_id)
          );
        });
      }
    });

    it('should minimize memory allocation during reranking', async () => {
      const initialMemory = process.memoryUsage().heapUsed;
      
      // Perform multiple reranking operations
      for (let i = 0; i < 100; i++) {
        await reranker.rerank(mockHits, mockContext);
      }
      
      const finalMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = finalMemory - initialMemory;
      
      // Memory increase should be minimal (less than 10MB for 100 operations)
      expect(memoryIncrease).toBeLessThan(10 * 1024 * 1024);
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle malformed search hits gracefully', async () => {
      const malformedHits = [
        { ...mockHits[0], score: NaN },
        { ...mockHits[1], snippet: null },
        { ...mockHits[2], span_id: undefined },
      ] as any;
      
      const result = await reranker.rerank(malformedHits, mockContext);
      
      expect(result).toBeDefined();
      expect(result.hits).toHaveLength(malformedHits.length);
      
      // Should sanitize or handle malformed data
      result.hits.forEach(hit => {
        expect(hit.span_id).toBeDefined();
        expect(typeof hit.score).toBe('number');
        expect(hit.score).not.toBeNaN();
      });
    });

    it('should handle missing context properties', async () => {
      const incompleteContext = {
        query: 'test query',
        mode: 'hybrid',
        // Missing other properties
      } as any;
      
      expect(() => reranker.rerank(mockHits, incompleteContext)).not.toThrow();
      
      const result = await reranker.rerank(mockHits, incompleteContext);
      expect(result).toBeDefined();
    });

    it('should handle extreme feature values', async () => {
      const extremeHit = {
        file: 'test.js',
        start_line: 1,
        end_line: 1000000, // Very large line range
        snippet: 'x'.repeat(100000), // Very long snippet
        score: Infinity,
        span_id: 'extreme_span',
        exact_match: true,
        symbol_name: 'x'.repeat(1000), // Very long symbol name
      };
      
      const result = await reranker.rerank([extremeHit] as any, mockContext);
      
      expect(result).toBeDefined();
      expect(result.hits).toHaveLength(1);
      
      const features = extractRerankingFeatures(extremeHit as any, mockContext);
      expect(features.snippetLength).toBeLessThanOrEqual(1);
      expect(features.symbolProximity).toBeLessThanOrEqual(1);
    });

    it('should maintain consistent behavior with different query patterns', async () => {
      const queryVariations = [
        'authenticate user login',
        'user-authentication',
        'AUTH_USER_LOGIN',
        'UserAuthenticationService',
        'how to authenticate a user',
      ];
      
      const results = await Promise.all(
        queryVariations.map(query => 
          reranker.rerank(mockHits, { ...mockContext, query })
        )
      );
      
      results.forEach(result => {
        expect(result).toBeDefined();
        expect(result.hits).toHaveLength(mockHits.length);
      });
      
      // Should handle all query patterns without errors
      expect(results.every(r => r.processing_time_ms > 0)).toBe(true);
    });

    it('should recover from internal computation errors', async () => {
      // Mock internal function to throw error
      const originalExtract = extractRerankingFeatures;
      
      vi.mocked(extractRerankingFeatures as any) = vi.fn()
        .mockImplementationOnce(() => { throw new Error('Feature extraction failed'); })
        .mockImplementation(originalExtract);
      
      const result = await reranker.rerank(mockHits, mockContext);
      
      // Should fallback gracefully
      expect(result.rerank_applied).toBe(false);
      expect(result.hits).toEqual(mockHits);
      expect(result.error).toMatch(/feature extraction|computation error/i);
    });
  });

  describe('Feature Weight Configuration', () => {
    it('should accept custom feature weights', () => {
      const customWeights = {
        exactness: 0.5,
        symbolProximity: 0.3,
        structHit: 0.15,
        pathPrior: 0.05,
        snippetLength: 0.0,
        nlLikelihood: 0.0,
      };
      
      const customReranker = new LearnedReranker({
        ...mockConfig,
        customWeights,
      });
      
      expect(customReranker.getFeatureWeights()).toEqual(customWeights);
    });

    it('should validate feature weights sum to 1', () => {
      const invalidWeights = {
        exactness: 0.5,
        symbolProximity: 0.3,
        structHit: 0.3, // Sum > 1
        pathPrior: 0.1,
        snippetLength: 0.0,
        nlLikelihood: 0.0,
      };
      
      expect(() => new LearnedReranker({
        ...mockConfig,
        customWeights: invalidWeights,
      })).toThrow(/weights.*sum/i);
    });

    it('should normalize feature weights if requested', () => {
      const unnormalizedWeights = {
        exactness: 10,
        symbolProximity: 6,
        structHit: 4,
        pathPrior: 2,
        snippetLength: 1,
        nlLikelihood: 0,
      };
      
      const rerankerWithNormalization = new LearnedReranker({
        ...mockConfig,
        customWeights: unnormalizedWeights,
        normalizeWeights: true,
      });
      
      const normalizedWeights = rerankerWithNormalization.getFeatureWeights();
      const sum = Object.values(normalizedWeights).reduce((a, b) => a + b, 0);
      
      expect(Math.abs(sum - 1.0)).toBeLessThan(0.001);
    });
  });
});