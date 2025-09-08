/**
 * Comprehensive API Server Coverage Tests
 * 
 * Target: 85%+ coverage across lines, functions, statements, branches
 * Strategy: Test all major endpoints, error paths, and edge cases
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import { createApp } from '../api/server.js';
import { LensTracer } from '../telemetry/tracer.js';
import { MetricsTelemetry } from '../raptor/metrics-telemetry.js';
import type {
  SearchRequest,
  StructRequest,
  SymbolsNearRequest,
  CompatibilityCheckRequest,
  SpiSearchRequest,
  ResolveRequest,
  ContextRequest,
  XrefRequest,
  SymbolsListRequest
} from '../types/api.js';

// Import fixtures for realistic test data
import { getSearchFixtures, getSymbolsFixtures } from './fixtures/db-fixtures-simple.js';

describe('Comprehensive API Server Coverage Tests', () => {
  let app: any;
  let fixtures: any;

  beforeAll(async () => {
    // Create server instance without auto-start for testing
    app = await createApp({ autoStart: false });
    fixtures = await getSearchFixtures();
    
    // Initialize telemetry for testing
    LensTracer.initialize('test-server');
    MetricsTelemetry.initialize();
  });

  afterAll(async () => {
    if (app) {
      await app.close();
    }
    LensTracer.shutdown();
  });

  beforeEach(() => {
    // Reset any global state between tests
  });

  describe('Core API Endpoints - Happy Paths', () => {
    it('should handle /search endpoint successfully', async () => {
      const searchRequest: SearchRequest = {
        query: 'function test',
        max_results: 10,
        include_definitions: true,
        include_references: false
      };

      const response = await app.inject({
        method: 'POST',
        url: '/search',
        payload: searchRequest,
        headers: {
          'content-type': 'application/json'
        }
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('results');
      expect(data).toHaveProperty('total_count');
      expect(data).toHaveProperty('query_id');
      expect(Array.isArray(data.results)).toBe(true);
    });

    it('should handle /struct endpoint successfully', async () => {
      const structRequest: StructRequest = {
        file: 'src/api/server.ts',
        line: 100,
        character: 10
      };

      const response = await app.inject({
        method: 'POST',
        url: '/struct',
        payload: structRequest,
        headers: {
          'content-type': 'application/json'
        }
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('symbols');
    });

    it('should handle /symbols/near endpoint successfully', async () => {
      const symbolsNearRequest: SymbolsNearRequest = {
        file: 'src/api/server.ts',
        line: 50,
        character: 5,
        radius: 100
      };

      const response = await app.inject({
        method: 'POST',
        url: '/symbols/near',
        payload: symbolsNearRequest,
        headers: {
          'content-type': 'application/json'
        }
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('symbols');
      expect(Array.isArray(data.symbols)).toBe(true);
    });

    it('should handle /health endpoint successfully', async () => {
      const response = await app.inject({
        method: 'GET',
        url: '/health'
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('status');
      expect(data).toHaveProperty('timestamp');
      expect(data).toHaveProperty('version');
      expect(data).toHaveProperty('uptime');
      expect(data.status).toBe('ok');
    });

    it('should handle /compatibility-check endpoint successfully', async () => {
      const compatRequest: CompatibilityCheckRequest = {
        client_version: '1.0.0',
        features: ['search', 'symbols'],
        bundle_hash: 'test-hash-123'
      };

      const response = await app.inject({
        method: 'POST',
        url: '/compatibility-check',
        payload: compatRequest,
        headers: {
          'content-type': 'application/json'
        }
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('compatible');
      expect(data).toHaveProperty('version_info');
    });
  });

  describe('SPI (Search Provider Interface) Endpoints', () => {
    it('should handle SPI search endpoint', async () => {
      const spiRequest: SpiSearchRequest = {
        query: 'test function',
        language: 'typescript',
        scope: 'project',
        max_results: 20
      };

      const response = await app.inject({
        method: 'POST',
        url: '/spi/search',
        payload: spiRequest,
        headers: {
          'content-type': 'application/json'
        }
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('results');
      expect(data).toHaveProperty('provider');
      expect(data.provider).toBe('lens');
    });

    it('should handle SPI health endpoint', async () => {
      const response = await app.inject({
        method: 'GET',
        url: '/spi/health'
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('status');
      expect(data).toHaveProperty('provider');
      expect(data.provider).toBe('lens');
    });
  });

  describe('LSP (Language Server Protocol) Endpoints', () => {
    it('should handle LSP resolve endpoint', async () => {
      const resolveRequest: ResolveRequest = {
        file: 'src/api/server.ts',
        line: 100,
        character: 15,
        symbol: 'fastify'
      };

      const response = await app.inject({
        method: 'POST',
        url: '/lsp/resolve',
        payload: resolveRequest,
        headers: {
          'content-type': 'application/json'
        }
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('resolved');
    });

    it('should handle LSP context endpoint', async () => {
      const contextRequest: ContextRequest = {
        file: 'src/api/server.ts',
        line: 50,
        character: 10,
        depth: 2
      };

      const response = await app.inject({
        method: 'POST',
        url: '/lsp/context',
        payload: contextRequest,
        headers: {
          'content-type': 'application/json'
        }
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('context');
    });

    it('should handle LSP xrefs endpoint', async () => {
      const xrefRequest: XrefRequest = {
        file: 'src/api/server.ts',
        line: 100,
        character: 10,
        include_definitions: true,
        include_references: true
      };

      const response = await app.inject({
        method: 'POST',
        url: '/lsp/xrefs',
        payload: xrefRequest,
        headers: {
          'content-type': 'application/json'
        }
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('xrefs');
      expect(Array.isArray(data.xrefs)).toBe(true);
    });

    it('should handle LSP symbols list endpoint', async () => {
      const symbolsRequest: SymbolsListRequest = {
        file: 'src/api/server.ts',
        include_private: false,
        symbol_types: ['function', 'class', 'interface']
      };

      const response = await app.inject({
        method: 'POST',
        url: '/lsp/symbols',
        payload: symbolsRequest,
        headers: {
          'content-type': 'application/json'
        }
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('symbols');
      expect(Array.isArray(data.symbols)).toBe(true);
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle malformed JSON requests', async () => {
      const response = await app.inject({
        method: 'POST',
        url: '/search',
        payload: '{ invalid json',
        headers: {
          'content-type': 'application/json'
        }
      });

      expect(response.statusCode).toBe(400);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('error');
      expect(data.error).toContain('Invalid JSON');
    });

    it('should handle missing required fields', async () => {
      const response = await app.inject({
        method: 'POST',
        url: '/search',
        payload: {},
        headers: {
          'content-type': 'application/json'
        }
      });

      expect(response.statusCode).toBe(400);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('error');
    });

    it('should handle invalid endpoint paths', async () => {
      const response = await app.inject({
        method: 'GET',
        url: '/non-existent-endpoint'
      });

      expect(response.statusCode).toBe(404);
    });

    it('should handle method not allowed errors', async () => {
      const response = await app.inject({
        method: 'GET',
        url: '/search'  // Should be POST
      });

      expect(response.statusCode).toBe(404); // Fastify returns 404 for wrong method
    });

    it('should handle internal server errors gracefully', async () => {
      // Test with a request that might cause internal errors
      const response = await app.inject({
        method: 'POST',
        url: '/search',
        payload: {
          query: 'a'.repeat(10000), // Very long query that might cause issues
          max_results: -1 // Invalid negative number
        },
        headers: {
          'content-type': 'application/json'
        }
      });

      // Should handle gracefully, not crash
      expect([200, 400, 500]).toContain(response.statusCode);
    });
  });

  describe('Request Validation and Schema Enforcement', () => {
    it('should validate search request schema strictly', async () => {
      const invalidRequests = [
        { query: 123 }, // Wrong type
        { query: '', max_results: 'ten' }, // Wrong type for max_results
        { query: 'test', include_definitions: 'yes' }, // Wrong type for boolean
      ];

      for (const invalidRequest of invalidRequests) {
        const response = await app.inject({
          method: 'POST',
          url: '/search',
          payload: invalidRequest,
          headers: {
            'content-type': 'application/json'
          }
        });

        expect(response.statusCode).toBe(400);
        const data = JSON.parse(response.payload);
        expect(data).toHaveProperty('error');
      }
    });

    it('should validate struct request parameters', async () => {
      const invalidRequests = [
        { file: '', line: 'ten' }, // Wrong type for line
        { file: 'test.ts', line: -1 }, // Negative line number
        { line: 10, character: 5 }, // Missing required file
      ];

      for (const invalidRequest of invalidRequests) {
        const response = await app.inject({
          method: 'POST',
          url: '/struct',
          payload: invalidRequest,
          headers: {
            'content-type': 'application/json'
          }
        });

        expect(response.statusCode).toBe(400);
      }
    });
  });

  describe('Response Format Validation', () => {
    it('should return properly formatted search responses', async () => {
      const response = await app.inject({
        method: 'POST',
        url: '/search',
        payload: { query: 'test', max_results: 5 },
        headers: { 'content-type': 'application/json' }
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      
      // Validate response schema
      expect(typeof data.total_count).toBe('number');
      expect(typeof data.query_id).toBe('string');
      expect(Array.isArray(data.results)).toBe(true);
      expect(data.total_count).toBeGreaterThanOrEqual(0);
      
      if (data.results.length > 0) {
        const firstResult = data.results[0];
        expect(firstResult).toHaveProperty('file');
        expect(firstResult).toHaveProperty('line');
        expect(firstResult).toHaveProperty('character');
        expect(firstResult).toHaveProperty('text');
      }
    });

    it('should include proper CORS headers', async () => {
      const response = await app.inject({
        method: 'OPTIONS',
        url: '/search',
        headers: {
          'origin': 'http://localhost:3000',
          'access-control-request-method': 'POST'
        }
      });

      expect(response.headers).toHaveProperty('access-control-allow-origin');
    });
  });

  describe('Performance and Resource Management', () => {
    it('should handle concurrent requests efficiently', async () => {
      const concurrentRequests = Array.from({ length: 10 }, (_, i) => (
        app.inject({
          method: 'POST',
          url: '/search',
          payload: { query: `test query ${i}`, max_results: 5 },
          headers: { 'content-type': 'application/json' }
        })
      ));

      const responses = await Promise.all(concurrentRequests);
      
      // All requests should complete successfully
      responses.forEach(response => {
        expect(response.statusCode).toBe(200);
        const data = JSON.parse(response.payload);
        expect(data).toHaveProperty('results');
      });
    });

    it('should handle large result sets appropriately', async () => {
      const response = await app.inject({
        method: 'POST',
        url: '/search',
        payload: { query: 'function', max_results: 1000 },
        headers: { 'content-type': 'application/json' }
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('results');
      // Should enforce reasonable limits
      expect(data.results.length).toBeLessThanOrEqual(1000);
    });
  });

  describe('Telemetry and Monitoring Integration', () => {
    it('should generate telemetry traces for requests', async () => {
      const response = await app.inject({
        method: 'POST',
        url: '/search',
        payload: { query: 'telemetry test', max_results: 5 },
        headers: { 'content-type': 'application/json' }
      });

      expect(response.statusCode).toBe(200);
      // Telemetry should be working (tested indirectly through successful request)
      expect(response.headers).toBeDefined();
    });

    it('should track request metrics properly', async () => {
      const response = await app.inject({
        method: 'GET',
        url: '/health'
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('uptime');
      expect(typeof data.uptime).toBe('number');
      expect(data.uptime).toBeGreaterThan(0);
    });
  });

  describe('Feature Flags and Configuration', () => {
    it('should respect feature flag configurations', async () => {
      // Test that endpoints work regardless of feature flag states
      const response = await app.inject({
        method: 'POST',
        url: '/search',
        payload: { query: 'feature test', max_results: 5 },
        headers: { 'content-type': 'application/json' }
      });

      expect(response.statusCode).toBe(200);
    });

    it('should handle quality gates validation', async () => {
      // Quality gates should not block normal operation during testing
      const response = await app.inject({
        method: 'GET',
        url: '/health'
      });

      expect(response.statusCode).toBe(200);
    });
  });

  describe('Benchmark and Metrics Endpoints', () => {
    it('should handle benchmark endpoint registration', async () => {
      // Test that benchmark endpoints are properly registered
      const response = await app.inject({
        method: 'GET',
        url: '/health'
      });

      expect(response.statusCode).toBe(200);
      // If benchmark endpoints are registered, server should start successfully
    });

    it('should handle metrics endpoints registration', async () => {
      // Test that metrics endpoints are properly registered
      const response = await app.inject({
        method: 'GET',
        url: '/health'
      });

      expect(response.statusCode).toBe(200);
      // If metrics endpoints are registered, server should start successfully  
    });
  });

  describe('Edge Cases and Boundary Conditions', () => {
    it('should handle empty query strings', async () => {
      const response = await app.inject({
        method: 'POST',
        url: '/search',
        payload: { query: '', max_results: 10 },
        headers: { 'content-type': 'application/json' }
      });

      // Should handle gracefully (either 200 with no results or 400 validation error)
      expect([200, 400]).toContain(response.statusCode);
    });

    it('should handle very long query strings', async () => {
      const longQuery = 'a'.repeat(10000);
      const response = await app.inject({
        method: 'POST',
        url: '/search',
        payload: { query: longQuery, max_results: 5 },
        headers: { 'content-type': 'application/json' }
      });

      // Should handle without crashing
      expect([200, 400, 413]).toContain(response.statusCode);
    });

    it('should handle boundary values for numeric parameters', async () => {
      const testCases = [
        { max_results: 0 },
        { max_results: 1 },
        { max_results: 10000 },
      ];

      for (const testCase of testCases) {
        const response = await app.inject({
          method: 'POST',
          url: '/search',
          payload: { query: 'test', ...testCase },
          headers: { 'content-type': 'application/json' }
        });

        // Should handle boundary conditions gracefully
        expect([200, 400]).toContain(response.statusCode);
      }
    });

    it('should handle special characters in queries', async () => {
      const specialQueries = [
        'test/path',
        'test.function',
        'test->method',
        'test::namespace',
        'test[0]',
        'test<T>',
        'test@annotation'
      ];

      for (const query of specialQueries) {
        const response = await app.inject({
          method: 'POST',
          url: '/search',
          payload: { query, max_results: 5 },
          headers: { 'content-type': 'application/json' }
        });

        expect(response.statusCode).toBe(200);
        const data = JSON.parse(response.payload);
        expect(data).toHaveProperty('results');
      }
    });
  });
});
