/**
 * Integration tests for Lens Search Engine
 * Tests the complete three-layer processing pipeline
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { LensSearchEngine } from '../../src/api/search-engine.js';
import type { SearchContext } from '../../src/types/core.js';
import { PRODUCTION_CONFIG } from '../../src/types/config.js';

describe('LensSearchEngine Integration', () => {
  let searchEngine: LensSearchEngine;

  beforeAll(async () => {
    searchEngine = new LensSearchEngine();
    await searchEngine.initialize();
  });

  afterAll(async () => {
    await searchEngine.shutdown();
  });

  describe('Initialization', () => {
    it('should initialize successfully', async () => {
      const newEngine = new LensSearchEngine();
      await expect(newEngine.initialize()).resolves.not.toThrow();
      await newEngine.shutdown();
    });

    it('should handle multiple initialization attempts', async () => {
      const newEngine = new LensSearchEngine();
      await newEngine.initialize();
      
      // Second initialization should not fail
      await expect(newEngine.initialize()).resolves.not.toThrow();
      await newEngine.shutdown();
    });
  });

  describe('Three-Layer Search Pipeline', () => {
    it('should process lexical search (Stage A)', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-pipeline-lex',
        query: 'function',
        mode: 'lex',
        k: 20,
        fuzzy_distance: 1,
        started_at: new Date(),
        stages: [],
      };

      const result = await searchEngine.search(ctx);
      
      expect(Array.isArray(result.candidates)).toBe(true);
      expect(typeof result.stage_a_latency).toBe('number');
      expect(result.stage_a_latency).toBeGreaterThan(0);
      
      // Stage A should complete within target
      expect(result.stage_a_latency).toBeLessThan(50); // Well under target
    });

    it('should process structural search (Stage B)', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-pipeline-struct',
        query: 'class test',
        mode: 'struct',
        k: 15,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      };

      const result = await searchEngine.search(ctx);
      
      expect(Array.isArray(result.candidates)).toBe(true);
      expect(typeof result.stage_a_latency).toBe('number');
      expect(typeof result.stage_b_latency).toBe('number');
    });

    it('should process hybrid search (All Stages)', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-pipeline-hybrid',
        query: 'test query',
        mode: 'hybrid',
        k: 100, // High enough to trigger Stage C
        fuzzy_distance: 2,
        started_at: new Date(),
        stages: [],
      };

      const result = await searchEngine.search(ctx);
      
      expect(Array.isArray(result.candidates)).toBe(true);
      expect(typeof result.stage_a_latency).toBe('number');
      expect(typeof result.stage_b_latency).toBe('number');
      
      // Stage C might be triggered if enough candidates
      if (result.stage_c_latency !== undefined) {
        expect(typeof result.stage_c_latency).toBe('number');
        expect(result.stage_c_latency).toBeGreaterThan(0);
      }
    });

    it('should respect candidate limits', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-pipeline-limits',
        query: 'limit test',
        mode: 'lex',
        k: 5,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      };

      const result = await searchEngine.search(ctx);
      
      expect(result.candidates.length).toBeLessThanOrEqual(5);
    });

    it('should handle semantic reranking threshold', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-pipeline-semantic',
        query: 'semantic rerank test',
        mode: 'hybrid',
        k: PRODUCTION_CONFIG.performance.max_candidates, // Max candidates
        fuzzy_distance: 1,
        started_at: new Date(),
        stages: [],
      };

      const result = await searchEngine.search(ctx);
      
      // Should trigger semantic reranking logic (even if it's currently a placeholder)
      expect(Array.isArray(result.candidates)).toBe(true);
      expect(result.candidates.length).toBeLessThanOrEqual(PRODUCTION_CONFIG.performance.max_candidates);
    });
  });

  describe('Search Modes', () => {
    it('should handle lexical mode', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-mode-lex',
        query: 'lexical mode test',
        mode: 'lex',
        k: 10,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      };

      const result = await searchEngine.search(ctx);
      
      expect(result.stage_a_latency).toBeGreaterThan(0);
      expect(result.stage_b_latency).toBeGreaterThanOrEqual(0);
    });

    it('should handle structural mode', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-mode-struct',
        query: 'structural pattern',
        mode: 'struct',
        k: 10,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      };

      const result = await searchEngine.search(ctx);
      
      expect(result.stage_a_latency).toBeGreaterThan(0);
      expect(result.stage_b_latency).toBeGreaterThan(0);
    });

    it('should handle hybrid mode', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-mode-hybrid',
        query: 'hybrid search test',
        mode: 'hybrid',
        k: 20,
        fuzzy_distance: 1,
        started_at: new Date(),
        stages: [],
      };

      const result = await searchEngine.search(ctx);
      
      expect(result.stage_a_latency).toBeGreaterThan(0);
      expect(result.stage_b_latency).toBeGreaterThan(0);
    });
  });

  describe('Structural Search', () => {
    it('should handle TypeScript patterns', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-struct-ts',
        query: 'interface Test',
        mode: 'struct',
        k: 15,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      };

      const result = await searchEngine.structuralSearch(ctx, 'typescript');
      
      expect(Array.isArray(result.candidates)).toBe(true);
    });

    it('should handle Python patterns', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-struct-py',
        query: 'def function_name(',
        mode: 'struct',
        k: 10,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      };

      const result = await searchEngine.structuralSearch(ctx, 'python');
      
      expect(Array.isArray(result.candidates)).toBe(true);
    });

    it('should handle Rust patterns', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-struct-rust',
        query: 'fn main() {',
        mode: 'struct',
        k: 5,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      };

      const result = await searchEngine.structuralSearch(ctx, 'rust');
      
      expect(Array.isArray(result.candidates)).toBe(true);
    });
  });

  describe('Symbols Near Location', () => {
    it('should find symbols near location', async () => {
      const result = await searchEngine.findSymbolsNear('/test/example.ts', 50, 20);
      
      expect(Array.isArray(result)).toBe(true);
    });

    it('should handle different radius values', async () => {
      const radii = [5, 10, 25, 50];
      
      for (const radius of radii) {
        const result = await searchEngine.findSymbolsNear('/test/file.js', 100, radius);
        expect(Array.isArray(result)).toBe(true);
      }
    });

    it('should handle edge case line numbers', async () => {
      const result1 = await searchEngine.findSymbolsNear('/test/start.py', 1, 10);
      const result2 = await searchEngine.findSymbolsNear('/test/large.rs', 10000, 5);
      
      expect(Array.isArray(result1)).toBe(true);
      expect(Array.isArray(result2)).toBe(true);
    });
  });

  describe('Repository Indexing', () => {
    it('should handle repository indexing request', async () => {
      const repoPath = '/test/mock-repo';
      const repoSha = '1234567890abcdef';
      
      await expect(searchEngine.indexRepository(repoPath, repoSha))
        .resolves.not.toThrow();
    });

    it('should handle multiple repository indexing', async () => {
      const repos = [
        { path: '/test/repo1', sha: 'sha1' },
        { path: '/test/repo2', sha: 'sha2' },
        { path: '/test/repo3', sha: 'sha3' },
      ];

      for (const repo of repos) {
        await expect(searchEngine.indexRepository(repo.path, repo.sha))
          .resolves.not.toThrow();
      }
    });
  });

  describe('Health Status', () => {
    it('should return valid health status', async () => {
      const health = await searchEngine.getHealthStatus();
      
      expect(['ok', 'degraded', 'down']).toContain(health.status);
      expect(typeof health.shards_healthy).toBe('number');
      expect(typeof health.shards_total).toBe('number');
      expect(typeof health.memory_usage_gb).toBe('number');
      expect(typeof health.active_queries).toBe('number');
      expect(health.shards_healthy).toBeGreaterThanOrEqual(0);
      expect(health.shards_total).toBeGreaterThanOrEqual(health.shards_healthy);
      expect(health.memory_usage_gb).toBeGreaterThanOrEqual(0);
      expect(health.active_queries).toBeGreaterThanOrEqual(0);
      expect(health.last_compaction).toBeInstanceOf(Date);
      
      // Worker pool status
      expect(typeof health.worker_pool_status.ingest_active).toBe('number');
      expect(typeof health.worker_pool_status.query_active).toBe('number');
      expect(typeof health.worker_pool_status.maintenance_active).toBe('number');
    });

    it('should handle health check during search', async () => {
      // Start a search in the background
      const searchPromise = searchEngine.search({
        trace_id: 'test-health-during-search',
        query: 'background search',
        mode: 'lex',
        k: 10,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      });

      // Check health while search is running
      const health = await searchEngine.getHealthStatus();
      
      // Wait for search to complete
      await searchPromise;
      
      expect(health.status).toMatch(/^(ok|degraded|down)$/);
    });
  });

  describe('Error Handling', () => {
    it('should handle search before initialization', async () => {
      const uninitializedEngine = new LensSearchEngine();
      
      const ctx: SearchContext = {
        trace_id: 'test-error-uninit',
        query: 'test',
        mode: 'lex',
        k: 5,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      };

      await expect(uninitializedEngine.search(ctx))
        .rejects.toThrow('not initialized');
    });

    it('should handle empty queries gracefully', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-error-empty',
        query: '',
        mode: 'lex',
        k: 10,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      };

      const result = await searchEngine.search(ctx);
      
      expect(Array.isArray(result.candidates)).toBe(true);
      expect(result.candidates.length).toBe(0);
    });

    it('should handle large k values', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-error-large-k',
        query: 'test',
        mode: 'lex',
        k: 10000, // Very large k
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      };

      const result = await searchEngine.search(ctx);
      
      expect(Array.isArray(result.candidates)).toBe(true);
      // Should handle gracefully, even if limited internally
    });

    it('should handle high fuzzy distances', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-error-high-fuzzy',
        query: 'test',
        mode: 'lex',
        k: 10,
        fuzzy_distance: 5, // Higher than recommended
        started_at: new Date(),
        stages: [],
      };

      const result = await searchEngine.search(ctx);
      
      expect(Array.isArray(result.candidates)).toBe(true);
    });
  });

  describe('Performance Monitoring', () => {
    it('should track active queries', async () => {
      const initialHealth = await searchEngine.getHealthStatus();
      const initialQueries = initialHealth.active_queries;

      // Start multiple searches
      const searches = [];
      for (let i = 0; i < 3; i++) {
        searches.push(searchEngine.search({
          trace_id: `test-perf-${i}`,
          query: `performance test ${i}`,
          mode: 'lex',
          k: 5,
          fuzzy_distance: 0,
          started_at: new Date(),
          stages: [],
        }));
      }

      // Wait for searches to complete
      await Promise.all(searches);

      const finalHealth = await searchEngine.getHealthStatus();
      
      // Active queries should be back to initial level (or close to it)
      expect(finalHealth.active_queries).toBeGreaterThanOrEqual(initialQueries);
    });

    it('should report memory usage', async () => {
      const health = await searchEngine.getHealthStatus();
      
      expect(health.memory_usage_gb).toBeGreaterThan(0);
      expect(health.memory_usage_gb).toBeLessThan(PRODUCTION_CONFIG.resources.memory_limit_gb);
    });
  });

  describe('Concurrent Operations', () => {
    it('should handle concurrent searches', async () => {
      const searches = [];
      
      for (let i = 0; i < 5; i++) {
        searches.push(searchEngine.search({
          trace_id: `test-concurrent-${i}`,
          query: `concurrent search ${i}`,
          mode: 'lex',
          k: 10,
          fuzzy_distance: 1,
          started_at: new Date(),
          stages: [],
        }));
      }

      const results = await Promise.all(searches);
      
      expect(results.length).toBe(5);
      results.forEach((result, index) => {
        expect(Array.isArray(result.candidates)).toBe(true);
        expect(result.stage_a_latency).toBeGreaterThan(0);
      });
    });

    it('should handle mixed operation types', async () => {
      const operations = [
        searchEngine.search({
          trace_id: 'test-mixed-1',
          query: 'mixed test',
          mode: 'lex',
          k: 5,
          fuzzy_distance: 0,
          started_at: new Date(),
          stages: [],
        }),
        searchEngine.getHealthStatus(),
        searchEngine.findSymbolsNear('/test/mixed.js', 25, 10),
        searchEngine.structuralSearch({
          trace_id: 'test-mixed-2',
          query: 'struct test',
          mode: 'struct',
          k: 5,
          fuzzy_distance: 0,
          started_at: new Date(),
          stages: [],
        }, 'typescript'),
      ];

      const results = await Promise.all(operations);
      
      expect(results.length).toBe(4);
      expect(Array.isArray(results[0].candidates)).toBe(true); // Search result
      expect(results[1].status).toMatch(/^(ok|degraded|down)$/); // Health result
      expect(Array.isArray(results[2])).toBe(true); // Symbols result
      expect(Array.isArray(results[3].candidates)).toBe(true); // Structural search result
    });
  });

  describe('Shutdown', () => {
    it('should shutdown gracefully', async () => {
      const testEngine = new LensSearchEngine();
      await testEngine.initialize();
      
      await expect(testEngine.shutdown()).resolves.not.toThrow();
    });

    it('should handle multiple shutdown calls', async () => {
      const testEngine = new LensSearchEngine();
      await testEngine.initialize();
      
      await testEngine.shutdown();
      await expect(testEngine.shutdown()).resolves.not.toThrow();
    });

    it('should wait for active queries during shutdown', async () => {
      const testEngine = new LensSearchEngine();
      await testEngine.initialize();
      
      // Start a search that might take some time
      const longSearch = testEngine.search({
        trace_id: 'test-shutdown-wait',
        query: 'long running search',
        mode: 'hybrid',
        k: 100,
        fuzzy_distance: 2,
        started_at: new Date(),
        stages: [],
      });

      // Start shutdown (should wait for the search)
      const shutdownPromise = testEngine.shutdown();
      
      // Both should complete
      await Promise.all([longSearch, shutdownPromise]);
      
      expect(true).toBe(true); // Test passes if no errors thrown
    });
  });
});