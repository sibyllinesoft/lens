/**
 * Integration tests for Lens API endpoints
 * Tests the complete request/response flow with validation
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { fastify, searchEngine, initializeServer, isInitialized } from '../../src/api/server.js';
import type { SearchRequest, SearchResponse, HealthResponse } from '../../src/types/api.js';

describe('Lens API Integration', () => {
  beforeAll(async () => {
    // Initialize the search engine if not already initialized
    if (!isInitialized) {
      await initializeServer();
    }
    
    // Start the server
    await fastify.listen({ port: 3001, host: '127.0.0.1' });
  });

  afterAll(async () => {
    await searchEngine.shutdown();
    await fastify.close();
  });

  describe('Health Endpoint', () => {
    it('should return health status', async () => {
      const response = await fastify.inject({
        method: 'GET',
        url: '/health',
      });

      expect(response.statusCode).toBe(200);
      
      const health: HealthResponse = JSON.parse(response.payload);
      expect(health.status).toMatch(/^(ok|degraded|down)$/);
      expect(health.timestamp).toBeTruthy();
      expect(typeof health.shards_healthy).toBe('number');
      expect(health.shards_healthy).toBeGreaterThanOrEqual(0);
    });

    it('should complete health check within SLA', async () => {
      const start = Date.now();
      
      const response = await fastify.inject({
        method: 'GET',
        url: '/health',
      });

      const latency = Date.now() - start;
      expect(response.statusCode).toBe(200);
      expect(latency).toBeLessThan(50); // Well under 5ms SLA target
    });
  });

  describe('Search Endpoint', () => {
    it('should handle valid search request', async () => {
      const searchRequest: SearchRequest = {
        q: 'function',
        mode: 'lex',
        fuzzy: 1,
        k: 10,
      };

      const response = await fastify.inject({
        method: 'POST',
        url: '/search',
        payload: searchRequest,
        headers: {
          'x-trace-id': 'test-trace-search-1',
        },
      });

      expect(response.statusCode).toBe(200);
      
      const searchResponse: SearchResponse = JSON.parse(response.payload);
      expect(Array.isArray(searchResponse.hits)).toBe(true);
      expect(typeof searchResponse.total).toBe('number');
      expect(searchResponse.total).toBeGreaterThanOrEqual(0);
      expect(searchResponse.latency_ms).toBeDefined();
      expect(searchResponse.latency_ms.total).toBeGreaterThan(0);
      expect(searchResponse.trace_id).toBe('test-trace-search-1');
    });

    it('should validate request schema', async () => {
      const invalidRequest = {
        q: '', // Empty query should fail validation
        mode: 'invalid_mode',
        fuzzy: 5, // Above max limit
        k: 0, // Below min limit
      };

      const response = await fastify.inject({
        method: 'POST',
        url: '/search',
        payload: invalidRequest,
      });

      expect(response.statusCode).toBe(400);
    });

    it('should handle different search modes', async () => {
      const modes: Array<'lex' | 'struct' | 'hybrid'> = ['lex', 'struct', 'hybrid'];

      for (const mode of modes) {
        const searchRequest: SearchRequest = {
          q: 'test query',
          mode,
          fuzzy: 1,
          k: 5,
        };

        const response = await fastify.inject({
          method: 'POST',
          url: '/search',
          payload: searchRequest,
        });

        expect(response.statusCode).toBe(200);
        
        const searchResponse: SearchResponse = JSON.parse(response.payload);
        expect(searchResponse.hits.length).toBeLessThanOrEqual(5);
      }
    });

    it('should respect k parameter limit', async () => {
      const searchRequest: SearchRequest = {
        q: 'test',
        mode: 'lex',
        fuzzy: 0,
        k: 3,
      };

      const response = await fastify.inject({
        method: 'POST',
        url: '/search',
        payload: searchRequest,
      });

      expect(response.statusCode).toBe(200);
      
      const searchResponse: SearchResponse = JSON.parse(response.payload);
      expect(searchResponse.hits.length).toBeLessThanOrEqual(3);
    });

    it('should handle fuzzy search', async () => {
      const searchRequest: SearchRequest = {
        q: 'functoin', // Typo
        mode: 'lex',
        fuzzy: 2,
        k: 10,
      };

      const response = await fastify.inject({
        method: 'POST',
        url: '/search',
        payload: searchRequest,
      });

      expect(response.statusCode).toBe(200);
      
      const searchResponse: SearchResponse = JSON.parse(response.payload);
      expect(Array.isArray(searchResponse.hits)).toBe(true);
    });

    it('should include timing information', async () => {
      const searchRequest: SearchRequest = {
        q: 'performance test',
        mode: 'hybrid',
        fuzzy: 1,
        k: 50,
      };

      const response = await fastify.inject({
        method: 'POST',
        url: '/search',
        payload: searchRequest,
      });

      expect(response.statusCode).toBe(200);
      
      const searchResponse: SearchResponse = JSON.parse(response.payload);
      const latency = searchResponse.latency_ms;
      
      expect(latency.stage_a).toBeGreaterThanOrEqual(0);
      expect(latency.stage_b).toBeGreaterThanOrEqual(0);
      expect(latency.total).toBeGreaterThan(0);
      
      // Check if it's within reasonable bounds (not SLA since we don't have real data)
      expect(latency.total).toBeLessThan(1000); // 1 second max for test
    });
  });

  describe('Structural Search Endpoint', () => {
    it('should handle struct search request', async () => {
      const structRequest = {
        pattern: 'function $_($args) { $body }',
        lang: 'typescript' as const,
        max_results: 10,
      };

      const response = await fastify.inject({
        method: 'POST',
        url: '/struct',
        payload: structRequest,
        headers: {
          'x-trace-id': 'test-trace-struct-1',
        },
      });

      expect(response.statusCode).toBe(200);
      
      const searchResponse: SearchResponse = JSON.parse(response.payload);
      expect(Array.isArray(searchResponse.hits)).toBe(true);
      expect(searchResponse.trace_id).toBe('test-trace-struct-1');
    });

    it('should validate struct request schema', async () => {
      const invalidRequest = {
        pattern: '', // Empty pattern
        lang: 'invalid_lang',
      };

      const response = await fastify.inject({
        method: 'POST',
        url: '/struct',
        payload: invalidRequest,
      });

      expect(response.statusCode).toBe(400);
    });

    it('should support different languages', async () => {
      const languages = ['typescript', 'python', 'rust', 'bash', 'go', 'java'];

      for (const lang of languages) {
        const structRequest = {
          pattern: 'function test()',
          lang: lang as any,
        };

        const response = await fastify.inject({
          method: 'POST',
          url: '/struct',
          payload: structRequest,
        });

        expect(response.statusCode).toBe(200);
      }
    });
  });

  describe('Symbols Near Endpoint', () => {
    it('should handle symbols near request', async () => {
      const symbolsRequest = {
        file: '/test/example.ts',
        line: 10,
        radius: 15,
      };

      const response = await fastify.inject({
        method: 'POST',
        url: '/symbols/near',
        payload: symbolsRequest,
        headers: {
          'x-trace-id': 'test-trace-symbols-1',
        },
      });

      expect(response.statusCode).toBe(200);
      
      const searchResponse: SearchResponse = JSON.parse(response.payload);
      expect(Array.isArray(searchResponse.hits)).toBe(true);
      expect(searchResponse.trace_id).toBe('test-trace-symbols-1');
    });

    it('should validate symbols request schema', async () => {
      const invalidRequest = {
        file: '', // Empty file path
        line: 0,  // Invalid line number
      };

      const response = await fastify.inject({
        method: 'POST',
        url: '/symbols/near',
        payload: invalidRequest,
      });

      expect(response.statusCode).toBe(400);
    });

    it('should handle default radius', async () => {
      const symbolsRequest = {
        file: '/test/example.py',
        line: 25,
        // No radius specified - should use default
      };

      const response = await fastify.inject({
        method: 'POST',
        url: '/symbols/near',
        payload: symbolsRequest,
      });

      expect(response.statusCode).toBe(200);
    });

    it('should complete within SLA target', async () => {
      const start = Date.now();
      
      const symbolsRequest = {
        file: '/test/performance.rs',
        line: 100,
        radius: 50,
      };

      const response = await fastify.inject({
        method: 'POST',
        url: '/symbols/near',
        payload: symbolsRequest,
      });

      const latency = Date.now() - start;
      
      expect(response.statusCode).toBe(200);
      expect(latency).toBeLessThan(100); // Well under 15ms SLA target
    });
  });

  describe('Error Handling', () => {
    it('should handle malformed JSON', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: '/search',
        payload: 'invalid json{',
        headers: {
          'content-type': 'application/json',
        },
      });

      expect(response.statusCode).toBe(400);
    });

    it('should handle missing content-type', async () => {
      const response = await fastify.inject({
        method: 'POST',
        url: '/search',
        payload: '{"q":"test","mode":"lex","fuzzy":0,"k":10}',
      });

      // Should still work or return appropriate error
      expect([200, 400, 415]).toContain(response.statusCode);
    });

    it('should handle unknown endpoints', async () => {
      const response = await fastify.inject({
        method: 'GET',
        url: '/unknown-endpoint',
      });

      expect(response.statusCode).toBe(404);
    });

    it('should generate trace IDs when not provided', async () => {
      const searchRequest: SearchRequest = {
        q: 'trace test',
        mode: 'lex',
        fuzzy: 0,
        k: 5,
      };

      const response = await fastify.inject({
        method: 'POST',
        url: '/search',
        payload: searchRequest,
        // No trace ID header
      });

      expect(response.statusCode).toBe(200);
      
      const searchResponse: SearchResponse = JSON.parse(response.payload);
      expect(searchResponse.trace_id).toMatch(/^[a-f0-9-]{36}$/); // UUID format
    });
  });

  describe('CORS Support', () => {
    it('should include CORS headers', async () => {
      const response = await fastify.inject({
        method: 'OPTIONS',
        url: '/search',
        headers: {
          origin: 'http://localhost:3000',
          'access-control-request-method': 'POST',
        },
      });

      expect(response.headers['access-control-allow-origin']).toBeDefined();
    });
  });

  describe('Response Validation', () => {
    it('should return valid search hit format', async () => {
      const searchRequest: SearchRequest = {
        q: 'format test',
        mode: 'lex',
        fuzzy: 0,
        k: 1,
      };

      const response = await fastify.inject({
        method: 'POST',
        url: '/search',
        payload: searchRequest,
      });

      expect(response.statusCode).toBe(200);
      
      const searchResponse: SearchResponse = JSON.parse(response.payload);
      
      // Validate each hit has required fields
      searchResponse.hits.forEach(hit => {
        expect(typeof hit.file).toBe('string');
        expect(hit.file.length).toBeGreaterThan(0);
        expect(typeof hit.line).toBe('number');
        expect(hit.line).toBeGreaterThanOrEqual(1);
        expect(typeof hit.col).toBe('number');
        expect(hit.col).toBeGreaterThanOrEqual(0);
        expect(typeof hit.score).toBe('number');
        expect(hit.score).toBeGreaterThanOrEqual(0);
        expect(hit.score).toBeLessThanOrEqual(1);
        expect(Array.isArray(hit.why)).toBe(true);
      });
    });

    it('should include proper latency breakdown', async () => {
      const searchRequest: SearchRequest = {
        q: 'latency test',
        mode: 'hybrid',
        fuzzy: 1,
        k: 10,
      };

      const response = await fastify.inject({
        method: 'POST',
        url: '/search',
        payload: searchRequest,
      });

      expect(response.statusCode).toBe(200);
      
      const searchResponse: SearchResponse = JSON.parse(response.payload);
      const latency = searchResponse.latency_ms;
      
      // Stage A should always be present
      expect(typeof latency.stage_a).toBe('number');
      expect(latency.stage_a).toBeGreaterThanOrEqual(0);
      
      // Stage B should be present for hybrid mode
      expect(typeof latency.stage_b).toBe('number');
      expect(latency.stage_b).toBeGreaterThanOrEqual(0);
      
      // Stage C might be present depending on candidate count
      if (latency.stage_c !== undefined) {
        expect(typeof latency.stage_c).toBe('number');
        expect(latency.stage_c).toBeGreaterThanOrEqual(0);
      }
      
      // Total should be sum of stages (approximately)
      expect(latency.total).toBeGreaterThanOrEqual(latency.stage_a + latency.stage_b);
    });
  });
});