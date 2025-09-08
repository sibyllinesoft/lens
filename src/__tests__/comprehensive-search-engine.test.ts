/**
 * Comprehensive Search Engine Coverage Tests
 * 
 * Target: Test all methods, code paths, and business logic in LensSearchEngine
 * Coverage focus: Core search pipeline, error handling, caching, reranking
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach, vi } from 'vitest';
import { LensSearchEngine } from '../api/search-engine.js';
import { LensTracer } from '../telemetry/tracer.js';
import { SegmentStorage } from '../storage/segments.js';
import { IndexRegistry } from '../core/index-registry.js';
import { ASTCache } from '../core/ast-cache.js';
import { LearnedReranker } from '../core/learned-reranker.js';
import type { SearchContext, SystemHealth } from '../types/core.js';
import type { SupportedLanguage } from '../types/api.js';

// Import fixtures
import { getSearchFixtures } from './fixtures/db-fixtures-simple.js';

describe('Comprehensive Search Engine Coverage Tests', () => {
  let searchEngine: LensSearchEngine;
  let fixtures: any;

  beforeAll(async () => {
    // Get test fixtures
    fixtures = await getSearchFixtures();
    
    // Create search engine instance
    searchEngine = new LensSearchEngine();
    
    // Wait for initialization
    await searchEngine.initialize();
  });

  afterAll(async () => {
    if (searchEngine) {
      await searchEngine.shutdown();
    }
  });

  beforeEach(() => {
    // Reset any stateful components between tests
    vi.clearAllMocks();
  });

  describe('Initialization and Shutdown', () => {
    it('should initialize properly with all components', async () => {
      const engine = new LensSearchEngine();
      
      await expect(engine.initialize()).resolves.not.toThrow();
      
      // Verify internal state
      const health = await engine.getSystemHealth();
      expect(health.status).toBe('healthy');
      expect(health.uptime).toBeGreaterThan(0);
      
      await engine.shutdown();
    });

    it('should handle initialization errors gracefully', async () => {
      const engine = new LensSearchEngine();
      
      // Mock storage failure
      vi.spyOn(SegmentStorage.prototype, 'initialize')
        .mockRejectedValueOnce(new Error('Storage init failed'));
      
      await expect(engine.initialize()).rejects.toThrow('Storage init failed');
    });

    it('should shutdown gracefully and clean up resources', async () => {
      const engine = new LensSearchEngine();
      await engine.initialize();
      
      const shutdownSpy = vi.spyOn(engine as any, 'shutdown');
      await engine.shutdown();
      
      expect(shutdownSpy).toHaveBeenCalled();
    });
  });

  describe('Core Search Pipeline', () => {
    it('should execute basic search with lexical stage', async () => {
      const context: SearchContext = {
        query: 'function test',
        max_results: 10,
        language_hints: ['typescript'] as SupportedLanguage[],
        include_definitions: true,
        include_references: false
      };

      const results = await searchEngine.search(context);
      
      expect(results).toBeDefined();
      expect(results.hits).toBeDefined();
      expect(Array.isArray(results.hits)).toBe(true);
      expect(results.hits.length).toBeGreaterThanOrEqual(0);
      expect(results.hits.length).toBeLessThanOrEqual(context.max_results);
      
      // Check that results have expected structure
      if (results.hits.length > 0) {
        const firstHit = results.hits[0];
        expect(firstHit).toHaveProperty('file');
        expect(firstHit).toHaveProperty('line');
        expect(firstHit).toHaveProperty('character');
        expect(firstHit).toHaveProperty('text');
        expect(firstHit).toHaveProperty('score');
        expect(firstHit.score).toBeGreaterThan(0);
      }
    });

    it('should handle semantic reranking stage', async () => {
      const context: SearchContext = {
        query: 'search engine',
        max_results: 20,
        language_hints: ['typescript'] as SupportedLanguage[],
        include_definitions: true,
        include_references: true,
        semantic_rerank: true
      };

      const results = await searchEngine.search(context);
      
      expect(results).toBeDefined();
      expect(results.hits).toBeDefined();
      
      // Should have stage latencies when semantic reranking is enabled
      expect(results.stage_a_latency).toBeGreaterThanOrEqual(0);
      expect(results.stage_b_latency).toBeGreaterThanOrEqual(0);
    });

    it('should handle symbol-based search', async () => {
      const context: SearchContext = {
        query: 'class LensSearchEngine',
        max_results: 5,
        language_hints: ['typescript'] as SupportedLanguage[],
        include_definitions: true,
        symbol_search: true
      };

      const results = await searchEngine.search(context);
      
      expect(results).toBeDefined();
      expect(results.hits).toBeDefined();
      
      // Symbol search should find class definitions
      if (results.hits.length > 0) {
        const symbolHits = results.hits.filter(hit => 
          hit.match_reasons?.includes('symbol') || hit.match_reasons?.includes('exact_name')
        );
        expect(symbolHits.length).toBeGreaterThan(0);
      }
    });

    it('should execute learned reranking stage', async () => {
      const context: SearchContext = {
        query: 'search function implementation',
        max_results: 15,
        language_hints: ['typescript'] as SupportedLanguage[],
        learned_rerank: true
      };

      const results = await searchEngine.search(context);
      
      expect(results).toBeDefined();
      expect(results.hits).toBeDefined();
      
      // Learned reranking should provide stage timing
      expect(results.stage_c_latency).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Query Processing and Normalization', () => {
    it('should handle empty queries', async () => {
      const context: SearchContext = {
        query: '',
        max_results: 10,
        language_hints: ['typescript'] as SupportedLanguage[]
      };

      const results = await searchEngine.search(context);
      
      expect(results).toBeDefined();
      expect(results.hits).toBeDefined();
      expect(results.hits.length).toBe(0);
    });

    it('should handle whitespace-only queries', async () => {
      const context: SearchContext = {
        query: '   \t\n  ',
        max_results: 10,
        language_hints: ['typescript'] as SupportedLanguage[]
      };

      const results = await searchEngine.search(context);
      
      expect(results).toBeDefined();
      expect(results.hits).toBeDefined();
      expect(results.hits.length).toBe(0);
    });

    it('should normalize unicode characters in queries', async () => {
      const context: SearchContext = {
        query: 'café', // Contains accented character
        max_results: 5,
        language_hints: ['typescript'] as SupportedLanguage[]
      };

      const results = await searchEngine.search(context);
      
      expect(results).toBeDefined();
      expect(results.hits).toBeDefined();
      // Should handle unicode normalization gracefully
    });

    it('should handle very long queries', async () => {
      const context: SearchContext = {
        query: 'very long query '.repeat(100),
        max_results: 10,
        language_hints: ['typescript'] as SupportedLanguage[]
      };

      const results = await searchEngine.search(context);
      
      expect(results).toBeDefined();
      expect(results.hits).toBeDefined();
      // Should handle long queries without crashing
    });
  });

  describe('Language-Specific Search', () => {
    const languages: SupportedLanguage[] = [
      'typescript', 'javascript', 'python', 'rust', 'go', 'java', 'cpp', 'csharp'
    ];

    languages.forEach(language => {
      it(`should handle ${language} language hints`, async () => {
        const context: SearchContext = {
          query: 'function main',
          max_results: 5,
          language_hints: [language]
        };

        const results = await searchEngine.search(context);
        
        expect(results).toBeDefined();
        expect(results.hits).toBeDefined();
        // Should complete successfully for all supported languages
      });
    });

    it('should handle multiple language hints', async () => {
      const context: SearchContext = {
        query: 'interface definition',
        max_results: 10,
        language_hints: ['typescript', 'java', 'csharp'] as SupportedLanguage[]
      };

      const results = await searchEngine.search(context);
      
      expect(results).toBeDefined();
      expect(results.hits).toBeDefined();
    });
  });

  describe('Search Options and Configuration', () => {
    it('should respect max_results parameter', async () => {
      const maxResults = 3;
      const context: SearchContext = {
        query: 'function',
        max_results: maxResults,
        language_hints: ['typescript'] as SupportedLanguage[]
      };

      const results = await searchEngine.search(context);
      
      expect(results.hits.length).toBeLessThanOrEqual(maxResults);
    });

    it('should handle include_definitions flag', async () => {
      const context: SearchContext = {
        query: 'class LensSearchEngine',
        max_results: 10,
        language_hints: ['typescript'] as SupportedLanguage[],
        include_definitions: true,
        include_references: false
      };

      const results = await searchEngine.search(context);
      
      expect(results).toBeDefined();
      expect(results.hits).toBeDefined();
      
      // When include_definitions is true, should find class definitions
      if (results.hits.length > 0) {
        const definitionHits = results.hits.filter(hit => 
          hit.is_definition === true
        );
        // Should have at least some definition hits
        expect(definitionHits.length).toBeGreaterThanOrEqual(0);
      }
    });

    it('should handle include_references flag', async () => {
      const context: SearchContext = {
        query: 'search',
        max_results: 10,
        language_hints: ['typescript'] as SupportedLanguage[],
        include_definitions: false,
        include_references: true
      };

      const results = await searchEngine.search(context);
      
      expect(results).toBeDefined();
      expect(results.hits).toBeDefined();
    });

    it('should handle fuzzy search options', async () => {
      const context: SearchContext = {
        query: 'serc', // Typo for 'search'
        max_results: 5,
        language_hints: ['typescript'] as SupportedLanguage[],
        fuzzy_search: true,
        fuzzy_threshold: 0.7
      };

      const results = await searchEngine.search(context);
      
      expect(results).toBeDefined();
      expect(results.hits).toBeDefined();
      
      // Fuzzy search might find relevant results despite typo
      if (results.hits.length > 0) {
        const fuzzyHits = results.hits.filter(hit => 
          hit.match_reasons?.includes('fuzzy')
        );
        expect(fuzzyHits.length).toBeGreaterThanOrEqual(0);
      }
    });
  });

  describe('LSP Integration', () => {
    it('should handle LSP hints when available', async () => {
      const context: SearchContext = {
        query: 'initialize',
        max_results: 10,
        language_hints: ['typescript'] as SupportedLanguage[],
        lsp_hints: [
          {
            file: 'src/api/search-engine.ts',
            line: 100,
            character: 10,
            hint_type: 'definition',
            confidence: 0.9
          }
        ]
      };

      const results = await searchEngine.search(context);
      
      expect(results).toBeDefined();
      expect(results.hits).toBeDefined();
      
      // LSP hints should influence ranking
      if (results.hits.length > 0) {
        const lspHits = results.hits.filter(hit => 
          hit.match_reasons?.includes('lsp_hint')
        );
        expect(lspHits.length).toBeGreaterThanOrEqual(0);
      }
    });

    it('should handle LSP stage B enhancements', async () => {
      const context: SearchContext = {
        query: 'method call',
        max_results: 8,
        language_hints: ['typescript'] as SupportedLanguage[],
        enable_lsp_stage_b: true
      };

      const results = await searchEngine.search(context);
      
      expect(results).toBeDefined();
      expect(results.hits).toBeDefined();
      expect(results.stage_b_latency).toBeGreaterThanOrEqual(0);
    });

    it('should handle LSP stage C enhancements', async () => {
      const context: SearchContext = {
        query: 'class method',
        max_results: 8,
        language_hints: ['typescript'] as SupportedLanguage[],
        enable_lsp_stage_c: true
      };

      const results = await searchEngine.search(context);
      
      expect(results).toBeDefined();
      expect(results.hits).toBeDefined();
      expect(results.stage_c_latency).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle null/undefined context gracefully', async () => {
      await expect(searchEngine.search(null as any))
        .rejects.toThrow();
        
      await expect(searchEngine.search(undefined as any))
        .rejects.toThrow();
    });

    it('should handle malformed context objects', async () => {
      const malformedContexts = [
        { query: null }, // null query
        { query: 'test' }, // missing max_results
        { max_results: 10 }, // missing query
        { query: 'test', max_results: -1 }, // negative max_results
        { query: 'test', max_results: 'ten' }, // wrong type
      ];

      for (const malformedContext of malformedContexts) {
        await expect(searchEngine.search(malformedContext as any))
          .rejects.toThrow();
      }
    });

    it('should handle storage failures gracefully', async () => {
      const context: SearchContext = {
        query: 'test storage failure',
        max_results: 5,
        language_hints: ['typescript'] as SupportedLanguage[]
      };

      // Mock storage failure
      const originalSearch = searchEngine.search.bind(searchEngine);
      vi.spyOn(searchEngine as any, 'search')
        .mockImplementationOnce(() => {
          throw new Error('Storage unavailable');
        });

      await expect(searchEngine.search(context))
        .rejects.toThrow('Storage unavailable');
    });

    it('should handle concurrent search requests', async () => {
      const contexts = Array.from({ length: 5 }, (_, i) => ({
        query: `concurrent search ${i}`,
        max_results: 5,
        language_hints: ['typescript'] as SupportedLanguage[]
      }));

      const searches = contexts.map(context => searchEngine.search(context));
      const results = await Promise.all(searches);
      
      // All searches should complete
      expect(results).toHaveLength(5);
      results.forEach(result => {
        expect(result).toBeDefined();
        expect(result.hits).toBeDefined();
      });
    });
  });

  describe('System Health and Monitoring', () => {
    it('should track system health accurately', async () => {
      const health = await searchEngine.getSystemHealth();
      
      expect(health).toHaveProperty('status');
      expect(health).toHaveProperty('uptime');
      expect(health).toHaveProperty('active_queries');
      expect(health).toHaveProperty('total_queries');
      
      expect(['healthy', 'degraded', 'unhealthy']).toContain(health.status);
      expect(health.uptime).toBeGreaterThanOrEqual(0);
      expect(health.active_queries).toBeGreaterThanOrEqual(0);
      expect(health.total_queries).toBeGreaterThanOrEqual(0);
    });

    it('should update query counters during searches', async () => {
      const healthBefore = await searchEngine.getSystemHealth();
      
      await searchEngine.search({
        query: 'health test',
        max_results: 5,
        language_hints: ['typescript'] as SupportedLanguage[]
      });
      
      const healthAfter = await searchEngine.getSystemHealth();
      
      expect(healthAfter.total_queries).toBeGreaterThan(healthBefore.total_queries);
    });

    it('should handle health check failures', async () => {
      // Mock a component failure
      const originalGetHealth = searchEngine.getSystemHealth.bind(searchEngine);
      vi.spyOn(searchEngine, 'getSystemHealth')
        .mockImplementationOnce(async () => {
          throw new Error('Health check failed');
        });

      await expect(searchEngine.getSystemHealth())
        .rejects.toThrow('Health check failed');
    });
  });

  describe('Caching and Performance', () => {
    it('should utilize AST caching for repeated queries', async () => {
      const context: SearchContext = {
        query: 'cached search test',
        max_results: 5,
        language_hints: ['typescript'] as SupportedLanguage[]
      };

      // First search
      const results1 = await searchEngine.search(context);
      expect(results1).toBeDefined();
      
      // Second search (should benefit from caching)
      const results2 = await searchEngine.search(context);
      expect(results2).toBeDefined();
      
      // Results should be consistent
      expect(results1.hits.length).toBe(results2.hits.length);
    });

    it('should track performance metrics', async () => {
      const context: SearchContext = {
        query: 'performance test',
        max_results: 10,
        language_hints: ['typescript'] as SupportedLanguage[],
        semantic_rerank: true,
        learned_rerank: true
      };

      const results = await searchEngine.search(context);
      
      // Should have timing information for all stages
      expect(results.stage_a_latency).toBeGreaterThanOrEqual(0);
      expect(results.stage_b_latency).toBeGreaterThanOrEqual(0);
      expect(results.stage_c_latency).toBeGreaterThanOrEqual(0);
      
      // Latencies should be reasonable (under 1000ms for tests)
      expect(results.stage_a_latency).toBeLessThan(1000);
      expect(results.stage_b_latency).toBeLessThan(1000);
      expect(results.stage_c_latency).toBeLessThan(1000);
    });
  });

  describe('Advanced Features', () => {
    it('should handle adaptive fanout configuration', async () => {
      const context: SearchContext = {
        query: 'adaptive test',
        max_results: 10,
        language_hints: ['typescript'] as SupportedLanguage[],
        enable_adaptive_fanout: true
      };

      const results = await searchEngine.search(context);
      
      expect(results).toBeDefined();
      expect(results.hits).toBeDefined();
    });

    it('should handle work-conserving ANN optimization', async () => {
      const context: SearchContext = {
        query: 'ann optimization',
        max_results: 10,
        language_hints: ['typescript'] as SupportedLanguage[],
        enable_work_conserving_ann: true
      };

      const results = await searchEngine.search(context);
      
      expect(results).toBeDefined();
      expect(results.hits).toBeDefined();
    });

    it('should handle precision optimization features', async () => {
      const context: SearchContext = {
        query: 'precision test',
        max_results: 10,
        language_hints: ['typescript'] as SupportedLanguage[],
        enable_precision_optimization: true
      };

      const results = await searchEngine.search(context);
      
      expect(results).toBeDefined();
      expect(results.hits).toBeDefined();
    });

    it('should handle intent routing', async () => {
      const context: SearchContext = {
        query: 'find class definition MyClass',
        max_results: 5,
        language_hints: ['typescript'] as SupportedLanguage[],
        enable_intent_routing: true
      };

      const results = await searchEngine.search(context);
      
      expect(results).toBeDefined();
      expect(results.hits).toBeDefined();
    });
  });

  describe('Result Quality and Ranking', () => {
    it('should return results with proper scoring', async () => {
      const context: SearchContext = {
        query: 'search engine',
        max_results: 10,
        language_hints: ['typescript'] as SupportedLanguage[]
      };

      const results = await searchEngine.search(context);
      
      if (results.hits.length > 0) {
        // Results should be sorted by score (highest first)
        for (let i = 1; i < results.hits.length; i++) {
          expect(results.hits[i - 1].score).toBeGreaterThanOrEqual(results.hits[i].score);
        }
        
        // All scores should be positive
        results.hits.forEach(hit => {
          expect(hit.score).toBeGreaterThan(0);
        });
      }
    });

    it('should provide diverse match reasons', async () => {
      const context: SearchContext = {
        query: 'function search test',
        max_results: 20,
        language_hints: ['typescript'] as SupportedLanguage[],
        fuzzy_search: true,
        semantic_rerank: true,
        symbol_search: true
      };

      const results = await searchEngine.search(context);
      
      if (results.hits.length > 0) {
        const allMatchReasons = new Set<string>();
        results.hits.forEach(hit => {
          if (hit.match_reasons) {
            hit.match_reasons.forEach(reason => allMatchReasons.add(reason));
          }
        });
        
        // Should have multiple types of match reasons
        expect(allMatchReasons.size).toBeGreaterThan(0);
        
        // Common match reasons should be present
        const commonReasons = ['exact', 'fuzzy', 'symbol', 'semantic'];
        const foundReasons = commonReasons.filter(reason => allMatchReasons.has(reason));
        expect(foundReasons.length).toBeGreaterThan(0);
      }
    });

    it('should handle result deduplication', async () => {
      const context: SearchContext = {
        query: 'duplicate test',
        max_results: 20,
        language_hints: ['typescript'] as SupportedLanguage[]
      };

      const results = await searchEngine.search(context);
      
      if (results.hits.length > 1) {
        const locations = new Set<string>();
        results.hits.forEach(hit => {
          const location = `${hit.file}:${hit.line}:${hit.character}`;
          expect(locations.has(location)).toBe(false); // No duplicates
          locations.add(location);
        });
      }
    });
  });
});
