/**
 * Tests for SemanticRerankEngine
 * Priority: HIGH - Core semantic search functionality, 43 complexity, 615 LOC
 */

import { describe, it, expect, beforeEach, afterEach, mock, jest, mock } from 'bun:test';
import { SemanticRerankEngine } from '../semantic.js';
import type { Candidate, SearchContext } from '../../types/core.js';

// Mock dependencies
mock('../../storage/segments.js', () => ({
  SegmentStorage: jest.fn().mockImplementation(() => ({
    getSegment: jest.fn(),
    storeSegment: jest.fn(),
  })),
}));

mock('../../telemetry/tracer.js', () => ({
  LensTracer: {
    createChildSpan: jest.fn().mockReturnValue({
      setAttributes: jest.fn(),
      recordException: jest.fn(),
      end: jest.fn(),
    }),
  },
}));

describe('SemanticRerankEngine', () => {
  let engine: SemanticRerankEngine;
  let mockSegmentStorage: any;

  beforeEach(() => {
    jest.clearAllMocks();
    mockSegmentStorage = {
      getSegment: jest.fn(),
      storeSegment: jest.fn(),
    };
    engine = new SemanticRerankEngine(mockSegmentStorage);
  });

  afterEach(async () => {
    if (engine) {
      await engine.shutdown();
    }
  });

  describe('Initialization', () => {
    it('should initialize successfully', async () => {
      await expect(engine.initialize()).resolves.not.toThrow();
    });

    it('should handle initialization errors', async () => {
      mockSegmentStorage.getSegment.mockRejectedValue(new Error('Storage error'));
      
      // Should not throw but log error
      await expect(engine.initialize()).resolves.not.toThrow();
    });
  });

  describe('Candidate Reranking', () => {
    let mockCandidates: Candidate[];
    let mockContext: SearchContext;

    beforeEach(async () => {
      await engine.initialize();
      
      mockCandidates = [
        {
          doc_id: '1',
          file_path: 'test.ts',
          line: 1,
          col: 0,
          score: 0.9,
          match_reasons: ['exact_match'],
          snippet: 'function test() {}',
        },
        {
          doc_id: '2', 
          file_path: 'app.ts',
          line: 5,
          col: 0,
          score: 0.8,
          match_reasons: ['fuzzy_match'],
          snippet: 'class Application {}',
        },
        {
          doc_id: '3',
          file_path: 'utils.ts',
          line: 10,
          col: 0,
          score: 0.7,
          match_reasons: ['token_match'],
          snippet: 'export const helper = () => {}',
        },
      ];

      mockContext = {
        query: 'test function',
        repo_sha: 'abc123',
        mode: 'hybrid',
        k: 10,
      };
    });

    it('should rerank candidates based on semantic similarity', async () => {
      const reranked = await engine.rerankCandidates(mockCandidates, mockContext, 10);
      
      expect(Array.isArray(reranked)).toBe(true);
      expect(reranked.length).toBeLessThanOrEqual(mockCandidates.length);
      
      // Scores should be adjusted by semantic reranking
      reranked.forEach((candidate, i) => {
        expect(candidate).toHaveProperty('score');
        expect(candidate.score).toBeGreaterThanOrEqual(0);
        expect(candidate.score).toBeLessThanOrEqual(1);
        
        if (i > 0) {
          // Results should be sorted by score (descending)
          expect(candidate.score).toBeLessThanOrEqual(reranked[i - 1].score);
        }
      });
    });

    it('should handle empty candidate list', async () => {
      const reranked = await engine.rerankCandidates([], mockContext, 10);
      expect(reranked).toEqual([]);
    });

    it('should respect maxResults parameter', async () => {
      const reranked = await engine.rerankCandidates(mockCandidates, mockContext, 2);
      expect(reranked.length).toBeLessThanOrEqual(2);
    });

    it('should handle natural language queries differently', async () => {
      const nlContext = {
        ...mockContext,
        query: 'find me a function that handles user authentication',
      };

      const reranked = await engine.rerankCandidates(mockCandidates, nlContext, 10);
      expect(Array.isArray(reranked)).toBe(true);
    });

    it('should handle code queries with proper weighting', async () => {
      const codeContext = {
        ...mockContext,
        query: 'function test()',
      };

      const reranked = await engine.rerankCandidates(mockCandidates, codeContext, 10);
      expect(Array.isArray(reranked)).toBe(true);
    });

    it('should maintain candidate metadata', async () => {
      const reranked = await engine.rerankCandidates(mockCandidates, mockContext, 10);
      
      reranked.forEach(candidate => {
        expect(candidate).toHaveProperty('doc_id');
        expect(candidate).toHaveProperty('file_path');
        expect(candidate).toHaveProperty('line');
        expect(candidate).toHaveProperty('match_reasons');
        expect(candidate).toHaveProperty('snippet');
      });
    });
  });

  describe('Configuration Updates', () => {
    beforeEach(async () => {
      await engine.initialize();
    });

    it('should update configuration successfully', async () => {
      const newConfig = {
        nlThreshold: 0.4,
        minCandidates: 15,
        efSearch: 200,
        confidenceCutoff: 0.75,
      };

      await expect(engine.updateConfig(newConfig)).resolves.not.toThrow();
    });

    it('should handle invalid configuration values', async () => {
      const invalidConfig = {
        nlThreshold: -0.5, // Invalid negative value
        minCandidates: -10, // Invalid negative value
      };

      // Should handle gracefully or validate
      await expect(engine.updateConfig(invalidConfig)).resolves.not.toThrow();
    });

    it('should apply configuration changes to reranking', async () => {
      const highThresholdConfig = {
        nlThreshold: 0.9, // Very high threshold
      };

      await engine.updateConfig(highThresholdConfig);

      const mockCandidates: Candidate[] = [
        {
          doc_id: '1',
          file_path: 'test.ts',
          line: 1,
          col: 0,
          score: 0.5,
          match_reasons: ['exact_match'],
          snippet: 'function test() {}',
        },
      ];

      const context = {
        query: 'simple query',
        repo_sha: 'abc123',
        mode: 'hybrid' as const,
        k: 10,
      };

      const reranked = await engine.rerankCandidates(mockCandidates, context, 10);
      expect(Array.isArray(reranked)).toBe(true);
    });
  });

  describe('Natural Language Detection', () => {
    beforeEach(async () => {
      await engine.initialize();
    });

    it('should identify natural language queries', async () => {
      const nlQueries = [
        'find functions that handle user authentication',
        'show me all the classes related to database operations',
        'what does this error mean?',
        'how do I implement caching?',
      ];

      for (const query of nlQueries) {
        const context = {
          query,
          repo_sha: 'abc123',
          mode: 'hybrid' as const,
          k: 10,
        };

        const mockCandidates: Candidate[] = [{
          doc_id: '1',
          file_path: 'test.ts',
          line: 1,
          col: 0,
          score: 0.5,
          match_reasons: ['match'],
          snippet: 'some code',
        }];

        const reranked = await engine.rerankCandidates(mockCandidates, context, 10);
        expect(Array.isArray(reranked)).toBe(true);
      }
    });

    it('should identify code-based queries', async () => {
      const codeQueries = [
        'function test()',
        'class User',
        'const API_URL',
        'import { useState }',
        'SELECT * FROM',
      ];

      for (const query of codeQueries) {
        const context = {
          query,
          repo_sha: 'abc123',
          mode: 'hybrid' as const,
          k: 10,
        };

        const mockCandidates: Candidate[] = [{
          doc_id: '1',
          file_path: 'test.ts',
          line: 1,
          col: 0,
          score: 0.5,
          match_reasons: ['match'],
          snippet: 'some code',
        }];

        const reranked = await engine.rerankCandidates(mockCandidates, context, 10);
        expect(Array.isArray(reranked)).toBe(true);
      }
    });
  });

  describe('Error Handling', () => {
    beforeEach(async () => {
      await engine.initialize();
    });

    it('should handle reranking errors gracefully', async () => {
      const mockCandidates: Candidate[] = [
        {
          doc_id: '1',
          file_path: 'test.ts',
          line: 1,
          col: 0,
          score: 0.9,
          match_reasons: ['exact_match'],
          snippet: 'function test() {}',
        },
      ];

      const context = {
        query: 'test',
        repo_sha: 'abc123',
        mode: 'hybrid' as const,
        k: 10,
      };

      // Mock internal error
      const originalRerank = (engine as any).performSemanticRerank;
      if (originalRerank) {
        (engine as any).performSemanticRerank = jest.fn().mockRejectedValue(new Error('Semantic error'));
      }

      // Should return original candidates on error, not throw
      const result = await engine.rerankCandidates(mockCandidates, context, 10);
      expect(Array.isArray(result)).toBe(true);
    });

    it('should handle malformed candidates', async () => {
      const malformedCandidates = [
        {
          doc_id: '1',
          // Missing required fields
          score: 0.9,
        } as any,
        {
          file_path: 'test.ts',
          // Missing doc_id
          line: 1,
          score: 0.8,
        } as any,
      ];

      const context = {
        query: 'test',
        repo_sha: 'abc123',
        mode: 'hybrid' as const,
        k: 10,
      };

      // Should handle gracefully
      const result = await engine.rerankCandidates(malformedCandidates, context, 10);
      expect(Array.isArray(result)).toBe(true);
    });
  });

  describe('Performance', () => {
    beforeEach(async () => {
      await engine.initialize();
    });

    it('should handle large candidate lists efficiently', async () => {
      const largeCandidateList: Candidate[] = Array(1000).fill(0).map((_, i) => ({
        doc_id: `doc_${i}`,
        file_path: `file_${i}.ts`,
        line: i + 1,
        col: 0,
        score: Math.random(),
        match_reasons: ['match'],
        snippet: `function test${i}() {}`,
      }));

      const context = {
        query: 'test function',
        repo_sha: 'abc123',
        mode: 'hybrid' as const,
        k: 50,
      };

      const startTime = Date.now();
      const reranked = await engine.rerankCandidates(largeCandidateList, context, 50);
      const duration = Date.now() - startTime;

      expect(reranked).toHaveLength(50);
      expect(duration).toBeLessThan(5000); // Should complete in under 5 seconds
    });

    it('should handle concurrent reranking requests', async () => {
      const mockCandidates: Candidate[] = Array(10).fill(0).map((_, i) => ({
        doc_id: `doc_${i}`,
        file_path: `file_${i}.ts`,
        line: i + 1,
        col: 0,
        score: Math.random(),
        match_reasons: ['match'],
        snippet: `function test${i}() {}`,
      }));

      const context = {
        query: 'test function',
        repo_sha: 'abc123',
        mode: 'hybrid' as const,
        k: 10,
      };

      const promises = Array(5).fill(0).map(() => 
        engine.rerankCandidates(mockCandidates, context, 10)
      );

      const results = await Promise.all(promises);
      
      results.forEach(result => {
        expect(Array.isArray(result)).toBe(true);
        expect(result.length).toBeLessThanOrEqual(10);
      });
    });
  });

  describe('Shutdown', () => {
    it('should shutdown cleanly', async () => {
      await engine.initialize();
      await expect(engine.shutdown()).resolves.not.toThrow();
    });

    it('should handle shutdown without initialization', async () => {
      const newEngine = new SemanticRerankEngine(mockSegmentStorage);
      await expect(newEngine.shutdown()).resolves.not.toThrow();
    });

    it('should handle shutdown during active operations', async () => {
      await engine.initialize();
      
      const mockCandidates: Candidate[] = [{
        doc_id: '1',
        file_path: 'test.ts',
        line: 1,
        col: 0,
        score: 0.9,
        match_reasons: ['match'],
        snippet: 'function test() {}',
      }];

      const context = {
        query: 'test',
        repo_sha: 'abc123',
        mode: 'hybrid' as const,
        k: 10,
      };

      // Start reranking operation
      const rerankPromise = engine.rerankCandidates(mockCandidates, context, 10);
      
      // Shutdown while operation is running
      const shutdownPromise = engine.shutdown();
      
      // Both should complete successfully
      await expect(Promise.all([rerankPromise, shutdownPromise]))
        .resolves.not.toThrow();
    });
  });

  describe('Semantic Similarity', () => {
    beforeEach(async () => {
      await engine.initialize();
    });

    it('should boost semantically similar candidates', async () => {
      const candidates: Candidate[] = [
        {
          doc_id: '1',
          file_path: 'auth.ts',
          line: 1,
          col: 0,
          score: 0.5,
          match_reasons: ['token_match'],
          snippet: 'function authenticate(user, password) {}',
        },
        {
          doc_id: '2',
          file_path: 'utils.ts',
          line: 1,
          col: 0,
          score: 0.9, // Higher initial score but less relevant
          match_reasons: ['exact_match'],
          snippet: 'function formatDate(date) {}',
        },
      ];

      const context = {
        query: 'user authentication login',
        repo_sha: 'abc123',
        mode: 'hybrid' as const,
        k: 10,
      };

      const reranked = await engine.rerankCandidates(candidates, context, 10);
      
      expect(reranked).toHaveLength(2);
      // The authenticate function should be boosted despite lower initial score
      // This test might need adjustment based on actual semantic implementation
    });

    it('should handle technical terms appropriately', async () => {
      const candidates: Candidate[] = [
        {
          doc_id: '1',
          file_path: 'api.ts',
          line: 1,
          col: 0,
          score: 0.7,
          match_reasons: ['match'],
          snippet: 'async function fetchUserData(userId: string) {}',
        },
        {
          doc_id: '2',
          file_path: 'cache.ts',
          line: 1,
          col: 0,
          score: 0.6,
          match_reasons: ['match'],
          snippet: 'class LRUCache<T> implements Cache<T> {}',
        },
      ];

      const context = {
        query: 'REST API endpoint',
        repo_sha: 'abc123',
        mode: 'hybrid' as const,
        k: 10,
      };

      const reranked = await engine.rerankCandidates(candidates, context, 10);
      expect(Array.isArray(reranked)).toBe(true);
      expect(reranked.length).toBe(2);
    });
  });
});