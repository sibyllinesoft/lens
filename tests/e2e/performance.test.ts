/**
 * End-to-end performance tests for Lens
 * Validates SLA targets: Stage-A: 2-8ms, Stage-B: 3-10ms, Stage-C: 5-15ms, Overall: <20ms
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { LensSearchEngine } from '../../src/api/search-engine.js';
import { fastify } from '../../src/api/server.js';
import type { SearchRequest, SearchResponse } from '../../src/types/api.js';
import { PRODUCTION_CONFIG } from '../../src/types/config.js';

describe('Lens Performance SLA Validation', () => {
  let searchEngine: LensSearchEngine;
  let serverUrl: string;

  beforeAll(async () => {
    // Initialize search engine and server
    searchEngine = new LensSearchEngine();
    await searchEngine.initialize();
    
    // Use dynamic port to avoid conflicts (port 0 = OS assigns available port)
    await fastify.listen({ port: 0, host: '127.0.0.1' });
    
    // Get the actual assigned port from the server
    const address = fastify.server.address();
    const port = typeof address === 'string' ? 3002 : address?.port || 3002;
    serverUrl = `http://127.0.0.1:${port}`;

    // Pre-warm the system with some test data
    await warmupSystem();
  });

  afterAll(async () => {
    await searchEngine.shutdown();
    await fastify.close();
  });

  /**
   * Pre-warm system with sample data to ensure realistic performance testing
   */
  async function warmupSystem() {
    // This would normally load test data into the search engine
    // For now, we'll just run a few warm-up queries
    const warmupQueries = [
      'function', 'class', 'interface', 'const', 'let', 'var',
      'import', 'export', 'async', 'await', 'return', 'if'
    ];

    for (const query of warmupQueries) {
      const searchRequest: SearchRequest = {
        q: query,
        mode: 'lex',
        fuzzy: 1,
        k: 10,
      };

      await fastify.inject({
        method: 'POST',
        url: '/search',
        payload: searchRequest,
      });
    }
  }

  describe('Stage A Performance (Lexical+Fuzzy) - Target: 2-8ms', () => {
    it('should complete lexical search within Stage A SLA', async () => {
      const testQueries = [
        'function',
        'calculateSum',
        'myVariable',
        'TestClass',
        'methodName',
      ];

      const results: number[] = [];

      for (const query of testQueries) {
        const searchRequest: SearchRequest = {
          q: query,
          mode: 'lex',
          fuzzy: 0, // Pure lexical for Stage A
          k: 20,
        };

        const response = await fastify.inject({
          method: 'POST',
          url: '/search',
          payload: searchRequest,
        });

        expect(response.statusCode).toBe(200);
        
        const searchResponse: SearchResponse = JSON.parse(response.payload);
        const stageALatency = searchResponse.latency_ms.stage_a;
        
        results.push(stageALatency);
      }

      // Calculate statistics
      const avgLatency = results.reduce((a, b) => a + b, 0) / results.length;
      const maxLatency = Math.max(...results);
      const minLatency = Math.min(...results);

      console.log(`Stage A Performance Stats:`);
      console.log(`  Min: ${minLatency}ms`);
      console.log(`  Max: ${maxLatency}ms`);
      console.log(`  Avg: ${avgLatency.toFixed(2)}ms`);
      console.log(`  Target: ${PRODUCTION_CONFIG.performance.stage_a_target_ms}ms`);

      // Validate against SLA targets
      expect(maxLatency).toBeLessThanOrEqual(PRODUCTION_CONFIG.performance.stage_a_target_ms);
      expect(avgLatency).toBeLessThan(PRODUCTION_CONFIG.performance.stage_a_target_ms);
      
      // At least 80% of queries should be under target
      const underTarget = results.filter(r => r <= PRODUCTION_CONFIG.performance.stage_a_target_ms);
      expect(underTarget.length / results.length).toBeGreaterThanOrEqual(0.8);
    });

    it('should handle fuzzy search within Stage A SLA', async () => {
      const fuzzyQueries = [
        'functoin', // function with typo
        'calcualte', // calculate with typo
        'varaible', // variable with typo
      ];

      for (const query of fuzzyQueries) {
        const searchRequest: SearchRequest = {
          q: query,
          mode: 'lex',
          fuzzy: 2, // Allow fuzzy matching
          k: 15,
        };

        const response = await fastify.inject({
          method: 'POST',
          url: '/search',
          payload: searchRequest,
        });

        expect(response.statusCode).toBe(200);
        
        const searchResponse: SearchResponse = JSON.parse(response.payload);
        const stageALatency = searchResponse.latency_ms.stage_a;
        
        // Fuzzy search might be slightly slower but should still be within bounds
        expect(stageALatency).toBeLessThanOrEqual(PRODUCTION_CONFIG.performance.stage_a_target_ms * 1.5);
      }
    });
  });

  describe('Stage B Performance (Symbol/AST) - Target: 3-10ms', () => {
    it('should complete structural search within Stage B SLA', async () => {
      const structQueries = [
        'function test',
        'class MyClass',
        'interface Config',
        'const value',
        'async function',
      ];

      const results: number[] = [];

      for (const query of structQueries) {
        const searchRequest: SearchRequest = {
          q: query,
          mode: 'struct',
          fuzzy: 0,
          k: 15,
        };

        const response = await fastify.inject({
          method: 'POST',
          url: '/search',
          payload: searchRequest,
        });

        expect(response.statusCode).toBe(200);
        
        const searchResponse: SearchResponse = JSON.parse(response.payload);
        const stageBLatency = searchResponse.latency_ms.stage_b;
        
        results.push(stageBLatency);
      }

      const avgLatency = results.reduce((a, b) => a + b, 0) / results.length;
      const maxLatency = Math.max(...results);

      console.log(`Stage B Performance Stats:`);
      console.log(`  Max: ${maxLatency}ms`);
      console.log(`  Avg: ${avgLatency.toFixed(2)}ms`);
      console.log(`  Target: ${PRODUCTION_CONFIG.performance.stage_b_target_ms}ms`);

      // Validate against SLA targets
      expect(maxLatency).toBeLessThanOrEqual(PRODUCTION_CONFIG.performance.stage_b_target_ms);
      expect(avgLatency).toBeLessThan(PRODUCTION_CONFIG.performance.stage_b_target_ms);
    });

    it('should handle symbols near lookup within SLA', async () => {
      const symbolsRequest = {
        file: '/test/large-file.ts',
        line: 500,
        radius: 25,
      };

      const start = Date.now();
      
      const response = await fastify.inject({
        method: 'POST',
        url: '/symbols/near',
        payload: symbolsRequest,
      });

      const latency = Date.now() - start;
      
      expect(response.statusCode).toBe(200);
      expect(latency).toBeLessThanOrEqual(15); // SLA target for symbols/near
    });
  });

  describe('Stage C Performance (Semantic Rerank) - Target: 5-15ms', () => {
    it('should complete semantic rerank within Stage C SLA when triggered', async () => {
      // Use query that should generate many candidates to trigger Stage C
      const searchRequest: SearchRequest = {
        q: 'test',
        mode: 'hybrid',
        fuzzy: 1,
        k: PRODUCTION_CONFIG.performance.max_candidates, // Trigger rerank
      };

      const response = await fastify.inject({
        method: 'POST',
        url: '/search',
        payload: searchRequest,
      });

      expect(response.statusCode).toBe(200);
      
      const searchResponse: SearchResponse = JSON.parse(response.payload);
      const stageCLatency = searchResponse.latency_ms.stage_c;
      
      if (stageCLatency !== undefined) {
        console.log(`Stage C Performance: ${stageCLatency}ms`);
        console.log(`Target: ${PRODUCTION_CONFIG.performance.stage_c_target_ms}ms`);
        
        expect(stageCLatency).toBeLessThanOrEqual(PRODUCTION_CONFIG.performance.stage_c_target_ms);
      }
    });
  });

  describe('Overall Performance - Target: <20ms p95', () => {
    it('should meet overall p95 SLA target', async () => {
      const queries = [
        { q: 'function', mode: 'lex' as const, fuzzy: 0, k: 20 },
        { q: 'class Test', mode: 'struct' as const, fuzzy: 0, k: 15 },
        { q: 'variable', mode: 'hybrid' as const, fuzzy: 1, k: 30 },
        { q: 'interface Config', mode: 'struct' as const, fuzzy: 0, k: 10 },
        { q: 'async function', mode: 'lex' as const, fuzzy: 1, k: 25 },
        { q: 'const value', mode: 'hybrid' as const, fuzzy: 0, k: 20 },
        { q: 'import statement', mode: 'lex' as const, fuzzy: 1, k: 15 },
        { q: 'export default', mode: 'struct' as const, fuzzy: 0, k: 12 },
      ];

      const latencies: number[] = [];

      // Run multiple iterations to get statistical significance
      for (let iteration = 0; iteration < 3; iteration++) {
        for (const searchRequest of queries) {
          const response = await fastify.inject({
            method: 'POST',
            url: '/search',
            payload: searchRequest,
          });

          expect(response.statusCode).toBe(200);
          
          const searchResponse: SearchResponse = JSON.parse(response.payload);
          latencies.push(searchResponse.latency_ms.total);
        }
      }

      // Calculate percentiles
      latencies.sort((a, b) => a - b);
      const p50 = latencies[Math.floor(latencies.length * 0.5)];
      const p95 = latencies[Math.floor(latencies.length * 0.95)];
      const p99 = latencies[Math.floor(latencies.length * 0.99)];
      const max = Math.max(...latencies);
      const avg = latencies.reduce((a, b) => a + b, 0) / latencies.length;

      console.log(`Overall Performance Stats (${latencies.length} queries):`);
      console.log(`  p50: ${p50}ms`);
      console.log(`  p95: ${p95}ms`);
      console.log(`  p99: ${p99}ms`);
      console.log(`  Max: ${max}ms`);
      console.log(`  Avg: ${avg.toFixed(2)}ms`);
      console.log(`  Target p95: ${PRODUCTION_CONFIG.performance.overall_p95_ms}ms`);

      // Validate against SLA targets
      expect(p95).toBeLessThanOrEqual(PRODUCTION_CONFIG.performance.overall_p95_ms);
      expect(avg).toBeLessThan(PRODUCTION_CONFIG.performance.overall_p95_ms * 0.5);
      
      // At least 95% of queries should meet SLA
      const withinSLA = latencies.filter(l => l <= PRODUCTION_CONFIG.performance.overall_p95_ms);
      expect(withinSLA.length / latencies.length).toBeGreaterThanOrEqual(0.95);
    });

    it('should maintain performance under concurrent load', async () => {
      const concurrentQueries = 10;
      const searchRequest: SearchRequest = {
        q: 'concurrent load test',
        mode: 'hybrid',
        fuzzy: 1,
        k: 20,
      };

      const startTime = Date.now();
      
      // Run concurrent searches
      const promises = Array(concurrentQueries).fill(null).map((_, i) => 
        fastify.inject({
          method: 'POST',
          url: '/search',
          payload: { ...searchRequest, q: `${searchRequest.q} ${i}` },
          headers: { 'x-trace-id': `concurrent-test-${i}` },
        })
      );

      const responses = await Promise.all(promises);
      const totalTime = Date.now() - startTime;

      // All requests should succeed
      responses.forEach(response => {
        expect(response.statusCode).toBe(200);
      });

      // Parse latencies
      const latencies = responses.map(response => {
        const searchResponse: SearchResponse = JSON.parse(response.payload);
        return searchResponse.latency_ms.total;
      });

      const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
      const maxLatency = Math.max(...latencies);

      console.log(`Concurrent Load Performance:`);
      console.log(`  Concurrent requests: ${concurrentQueries}`);
      console.log(`  Total time: ${totalTime}ms`);
      console.log(`  Avg latency: ${avgLatency.toFixed(2)}ms`);
      console.log(`  Max latency: ${maxLatency}ms`);

      // Performance should not degrade significantly under load
      expect(maxLatency).toBeLessThanOrEqual(PRODUCTION_CONFIG.performance.overall_p95_ms * 2);
      expect(avgLatency).toBeLessThan(PRODUCTION_CONFIG.performance.overall_p95_ms);
    });
  });

  describe('Resource Utilization', () => {
    it('should stay within memory limits', async () => {
      const health = await searchEngine.getHealthStatus();
      
      expect(health.memory_usage_gb).toBeLessThan(PRODUCTION_CONFIG.resources.memory_limit_gb);
      expect(health.memory_usage_gb).toBeGreaterThan(0);
      
      console.log(`Memory Usage: ${health.memory_usage_gb.toFixed(2)}GB / ${PRODUCTION_CONFIG.resources.memory_limit_gb}GB`);
    });

    it('should handle concurrent query limits', async () => {
      const health = await searchEngine.getHealthStatus();
      
      expect(health.active_queries).toBeLessThanOrEqual(PRODUCTION_CONFIG.resources.max_concurrent_queries);
      
      console.log(`Active Queries: ${health.active_queries} / ${PRODUCTION_CONFIG.resources.max_concurrent_queries}`);
    });
  });

  describe('Health Check Performance', () => {
    it('should complete health check within 5ms SLA', async () => {
      const iterations = 10;
      const latencies: number[] = [];

      for (let i = 0; i < iterations; i++) {
        const start = Date.now();
        
        const response = await fastify.inject({
          method: 'GET',
          url: '/health',
        });

        const latency = Date.now() - start;
        
        expect(response.statusCode).toBe(200);
        latencies.push(latency);
      }

      const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
      const maxLatency = Math.max(...latencies);

      console.log(`Health Check Performance:`);
      console.log(`  Max: ${maxLatency}ms`);
      console.log(`  Avg: ${avgLatency.toFixed(2)}ms`);
      console.log(`  Target: 5ms`);

      expect(maxLatency).toBeLessThanOrEqual(5); // Health check SLA
      expect(avgLatency).toBeLessThan(3); // Should be well under target
    });
  });

  describe('Performance Regression Detection', () => {
    it('should not regress beyond acceptable thresholds', async () => {
      // Baseline performance test
      const baselineQueries = [
        'function test',
        'class Example',
        'const variable',
        'interface Type',
        'async method',
      ];

      const baselineLatencies: number[] = [];
      
      for (const query of baselineQueries) {
        const searchRequest: SearchRequest = {
          q: query,
          mode: 'hybrid',
          fuzzy: 1,
          k: 20,
        };

        const response = await fastify.inject({
          method: 'POST',
          url: '/search',
          payload: searchRequest,
        });

        expect(response.statusCode).toBe(200);
        
        const searchResponse: SearchResponse = JSON.parse(response.payload);
        baselineLatencies.push(searchResponse.latency_ms.total);
      }

      const avgBaseline = baselineLatencies.reduce((a, b) => a + b, 0) / baselineLatencies.length;

      console.log(`Baseline Average Latency: ${avgBaseline.toFixed(2)}ms`);
      
      // Performance should be stable and predictable
      const standardDeviation = Math.sqrt(
        baselineLatencies.reduce((sum, latency) => sum + Math.pow(latency - avgBaseline, 2), 0) / baselineLatencies.length
      );

      console.log(`Standard Deviation: ${standardDeviation.toFixed(2)}ms`);
      
      // Low variability indicates stable performance
      expect(standardDeviation).toBeLessThan(avgBaseline * 0.3); // Within 30% variation
    });
  });
});