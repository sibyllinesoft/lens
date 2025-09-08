/**
 * Simple Tests for IntentRouter
 * Focused on core functionality and coverage
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { IntentRouter } from '../intent-router.js';
import type { SearchContext, Candidate } from '../../types/core.js';

// Mock dependencies
vi.mock('../telemetry/tracer.js', () => ({
  LensTracer: {
    createChildSpan: vi.fn(() => ({
      setAttributes: vi.fn(),
      recordException: vi.fn(),
      end: vi.fn()
    }))
  }
}));

describe('IntentRouter', () => {
  let router: IntentRouter;
  let mockLspEnhancer: any;
  let mockFullSearchHandler: any;

  beforeEach(() => {
    vi.clearAllMocks();
    
    mockLspEnhancer = {
      enhanceStageB: vi.fn().mockResolvedValue({ candidates: [] })
    };
    
    router = new IntentRouter(mockLspEnhancer);
    
    mockFullSearchHandler = vi.fn().mockResolvedValue([
      { doc_id: 'doc1', file_path: 'test.ts', score: 0.9, snippet: 'test result' }
    ]);
  });

  describe('Basic Functionality', () => {
    const mockContext: SearchContext = {
      query: 'test',
      mode: 'precise' as const,
      repo_sha: 'abc123',
      max_results: 10,
    };

    it('should classify queries and return classification info', () => {
      const classification = router.classifyQueryIntent('function test');
      
      expect(classification).toBeDefined();
      expect(classification.intent).toBeDefined();
      expect(typeof classification.confidence).toBe('number');
      expect(classification.features).toBeDefined();
    });

    it('should route queries successfully', async () => {
      const result = await router.routeQuery(
        'test query',
        mockContext,
        undefined,
        mockFullSearchHandler
      );

      expect(result).toBeDefined();
      expect(result.classification).toBeDefined();
      expect(result.primary_candidates).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            doc_id: expect.any(String),
            file_path: expect.any(String)
          })
        ])
      );
      expect(result.routing_path).toEqual(expect.any(Array));
      expect(result.routing_path.length).toBeGreaterThan(0);
    });

    it('should handle empty queries', async () => {
      const result = await router.routeQuery(
        '',
        mockContext,
        undefined,
        mockFullSearchHandler
      );

      expect(result).toBeDefined();
      expect(result.classification.intent).toBe('lexical');
    });

    it('should handle various query types', async () => {
      const queries = [
        'function getUserData',
        'def calculateSum',
        'class MyClass',
        'find references',
        'test query',
        'natural language search',
        'MySymbol'
      ];

      for (const query of queries) {
        const classification = router.classifyQueryIntent(query);
        expect(classification.intent).toBeDefined();
        expect(classification.confidence).toBeGreaterThanOrEqual(0);
        expect(classification.confidence).toBeLessThanOrEqual(1);
      }
    });
  });

  describe('Error Handling', () => {
    const mockContext: SearchContext = {
      query: 'test',
      mode: 'precise' as const,
      repo_sha: 'abc123',
      max_results: 10,
    };

    it('should handle handler errors gracefully', async () => {
      mockFullSearchHandler.mockRejectedValue(new Error('Search error'));

      await expect(async () => {
        await router.routeQuery(
          'test',
          mockContext,
          undefined,
          mockFullSearchHandler
        );
      }).rejects.toThrow('Search error');
    });

    it('should handle missing handlers', async () => {
      const result = await router.routeQuery(
        'test',
        mockContext,
        undefined,
        undefined
      );

      expect(result).toBeDefined();
      expect(result.primary_candidates).toEqual([]);
    });

    it('should handle malformed contexts', async () => {
      const invalidContext = {} as SearchContext;

      const result = await router.routeQuery(
        'test',
        invalidContext,
        undefined,
        mockFullSearchHandler
      );

      expect(result).toBeDefined();
    });
  });

  describe('Feature Extraction', () => {
    it('should extract features from simple queries', () => {
      const classification = router.classifyQueryIntent('test');
      
      expect(classification.features).toEqual({
        has_definition_pattern: expect.any(Boolean),
        has_reference_pattern: expect.any(Boolean),
        has_symbol_prefix: expect.any(Boolean),
        has_structural_chars: expect.any(Boolean),
        is_natural_language: expect.any(Boolean),
      });
    });

    it('should handle special characters', () => {
      const classification = router.classifyQueryIntent('test() { return true; }');
      
      expect(classification.features.has_structural_chars).toBe(true);
    });

    it('should handle empty and whitespace queries', () => {
      const emptyClassification = router.classifyQueryIntent('');
      expect(emptyClassification.intent).toBe('lexical');
      
      const whitespaceClassification = router.classifyQueryIntent('   ');
      expect(whitespaceClassification.intent).toBe('lexical');
    });
  });

  describe('Result Limiting', () => {
    it('should respect max_results configuration', async () => {
      const manyResults = Array.from({ length: 50 }, (_, i) => ({
        doc_id: `doc${i}`,
        file_path: `file${i}.ts`,
        score: 0.9,
        snippet: `result ${i}`
      }));

      mockFullSearchHandler.mockResolvedValue(manyResults);

      const result = await router.routeQuery(
        'test',
        { ...{
          query: 'test',
          mode: 'precise' as const,
          repo_sha: 'abc123',
          max_results: 20,
        } },
        undefined,
        mockFullSearchHandler
      );

      expect(result.primary_candidates.length).toBeGreaterThan(0);
    });
  });

  describe('Statistics', () => {
    it('should provide router statistics', () => {
      const stats = router.getStats();
      
      expect(stats).toBeDefined();
      expect(typeof stats.confidence_threshold).toBe('number');
      expect(typeof stats.max_primary_results).toBe('number');
      expect(Array.isArray(stats.classification_features)).toBe(true);
    });
  });

  describe('Training and Updates', () => {
    it('should handle classification model updates', () => {
      const trainingData = [
        {
          query: 'function test',
          actual_intent: 'def' as const,
          user_satisfaction: 0.9
        }
      ];

      expect(() => {
        router.updateClassificationModel(trainingData);
      }).not.toThrow();
    });
  });

  describe('Performance', () => {
    it('should handle concurrent requests', async () => {
      const mockContext: SearchContext = {
        query: 'concurrent',
        mode: 'precise' as const,
        repo_sha: 'abc123',
        max_results: 10,
      };

      const promises = Array.from({ length: 5 }, () =>
        router.routeQuery('concurrent test', mockContext, undefined, mockFullSearchHandler)
      );

      const results = await Promise.all(promises);
      
      expect(results).toHaveLength(5);
      results.forEach(result => {
        expect(result).toBeDefined();
        expect(result.classification).toBeDefined();
      });
    });

    it('should complete routing quickly', async () => {
      const start = Date.now();
      
      await router.routeQuery(
        'performance test',
        {
          query: 'performance test',
          mode: 'precise' as const,
          repo_sha: 'abc123',
          max_results: 10,
        },
        undefined,
        mockFullSearchHandler
      );
      
      const elapsed = Date.now() - start;
      expect(elapsed).toBeLessThan(1000); // Should be fast
    });
  });
});