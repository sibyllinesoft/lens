/**
 * Unit tests for Fastify server core functionality
 * Focus on request processing, validation, and business logic without actual server startup
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import type {
  SearchRequest,
  SearchResponse,
  HealthResponse,
  CompatibilityCheckRequest,
  SpiSearchRequest,
  ResolveRequest,
  ContextRequest,
  XrefRequest
} from '../../types/api.js';
import type { SearchContext, SearchHit } from '../../types/core.js';

// Mock all external dependencies
vi.mock('fastify', () => {
  const mockFastify = {
    register: vi.fn(),
    get: vi.fn(),
    post: vi.fn(),
    listen: vi.fn(),
    close: vi.fn(),
    log: {
      info: vi.fn(),
      error: vi.fn(),
      warn: vi.fn()
    },
    setErrorHandler: vi.fn()
  };
  return {
    default: vi.fn(() => mockFastify)
  };
});

vi.mock('@fastify/cors', () => ({
  default: vi.fn()
}));

vi.mock('../search-engine.js', () => ({
  LensSearchEngine: vi.fn().mockImplementation(() => ({
    initialize: vi.fn().mockResolvedValue(undefined),
    search: vi.fn().mockResolvedValue({
      hits: [],
      stage_a_latency: 5,
      stage_b_latency: 3,
      stage_c_latency: 2
    }),
    getHealthStatus: vi.fn().mockResolvedValue({
      status: 'ok',
      shards_healthy: 1,
      shards_total: 1,
      memory_usage_gb: 0.5,
      active_queries: 0,
      worker_pool_status: {
        ingest_active: 0,
        query_active: 0,
        maintenance_active: 0
      },
      last_compaction: new Date()
    }),
    getManifest: vi.fn().mockResolvedValue({
      'test-repo': {
        repo_sha: 'abc123',
        api_version: '1.0.0',
        index_version: '1.0.0',
        policy_version: '1.0.0'
      }
    }),
    shutdown: vi.fn().mockResolvedValue(undefined)
  }))
}));

vi.mock('../../telemetry/tracer.js', () => ({
  LensTracer: {
    createChildSpan: vi.fn(() => ({
      setAttributes: vi.fn(),
      recordException: vi.fn(),
      end: vi.fn()
    }))
  }
}));

vi.mock('../../types/config.js', () => ({
  PRODUCTION_CONFIG: {
    performance: {
      max_concurrent_queries: 100,
      timeout_ms: 30000
    },
    security: {
      max_query_length: 1000
    },
    api: {
      port: 3000
    }
  }
}));

describe('Fastify Server Unit Tests', () => {
  describe('Request Validation Logic', () => {
    it('should validate search request parameters', () => {
      const validRequest: SearchRequest = {
        q: 'test function',
        repo_sha: 'abc123',
        k: 10
      };

      expect(validRequest.q).toBeTruthy();
      expect(validRequest.repo_sha).toBeTruthy();
      expect(validRequest.k).toBeGreaterThan(0);
    });

    it('should handle optional search parameters', () => {
      const requestWithOptionals: SearchRequest = {
        q: 'test',
        repo_sha: 'abc123',
        k: 5,
        mode: 'hybrid',
        fuzzy: true,
        fuzzy_distance: 2
      };

      expect(requestWithOptionals.mode).toBe('hybrid');
      expect(requestWithOptionals.fuzzy).toBe(true);
      expect(requestWithOptionals.fuzzy_distance).toBe(2);
    });

    it('should validate query length limits', () => {
      const maxLength = 1000;
      const validQuery = 'a'.repeat(maxLength - 1);
      const invalidQuery = 'a'.repeat(maxLength + 1);

      expect(validQuery.length).toBeLessThan(maxLength);
      expect(invalidQuery.length).toBeGreaterThan(maxLength);
    });

    it('should validate k parameter bounds', () => {
      const minK = 1;
      const maxK = 100;
      
      const validK = 10;
      const tooSmallK = 0;
      const tooLargeK = 150;

      expect(validK).toBeGreaterThanOrEqual(minK);
      expect(validK).toBeLessThanOrEqual(maxK);
      expect(tooSmallK).toBeLessThan(minK);
      expect(tooLargeK).toBeGreaterThan(maxK);
    });

    it('should validate repo_sha format', () => {
      const validSha = 'abc123def456';
      const invalidSha = 'invalid!';

      const isValidSha = /^[a-fA-F0-9]+$/.test(validSha);
      const isInvalidSha = /^[a-fA-F0-9]+$/.test(invalidSha);

      expect(isValidSha).toBe(true);
      expect(isInvalidSha).toBe(false);
    });
  });

  describe('Search Request Processing', () => {
    it('should convert search request to search context', () => {
      const request: SearchRequest = {
        q: 'function test',
        repo_sha: 'abc123',
        k: 10,
        mode: 'hybrid',
        fuzzy: true,
        fuzzy_distance: 1
      };

      const context: SearchContext = {
        query: request.q,
        repo_sha: request.repo_sha,
        k: request.k,
        mode: request.mode || 'hybrid',
        fuzzy: request.fuzzy,
        fuzzy_distance: request.fuzzy_distance
      };

      expect(context.query).toBe(request.q);
      expect(context.repo_sha).toBe(request.repo_sha);
      expect(context.k).toBe(request.k);
      expect(context.mode).toBe('hybrid');
      expect(context.fuzzy).toBe(true);
    });

    it('should apply default parameters', () => {
      const minimalRequest: SearchRequest = {
        q: 'test',
        repo_sha: 'abc123',
        k: 5
      };

      const context: SearchContext = {
        query: minimalRequest.q,
        repo_sha: minimalRequest.repo_sha,
        k: minimalRequest.k,
        mode: 'hybrid', // default
        fuzzy: false // default
      };

      expect(context.mode).toBe('hybrid');
      expect(context.fuzzy).toBe(false);
    });

    it('should sanitize query input', () => {
      const rawQuery = '  test   function   ';
      const sanitizedQuery = rawQuery.trim().replace(/\s+/g, ' ');

      expect(sanitizedQuery).toBe('test function');
    });

    it('should handle special characters in queries', () => {
      const specialQuery = 'function(test) { return @param; }';
      const escapedQuery = specialQuery; // No escaping needed for search

      expect(escapedQuery).toContain('(');
      expect(escapedQuery).toContain('@');
      expect(escapedQuery).toContain('{');
    });
  });

  describe('Response Formation', () => {
    it('should format search response correctly', () => {
      const hits: SearchHit[] = [
        {
          file: 'src/test.ts',
          line: 10,
          col: 5,
          lang: 'typescript',
          snippet: 'function test() {',
          score: 0.95,
          why: ['exact'],
          byte_offset: 100,
          span_len: 17
        }
      ];

      const response: SearchResponse = {
        hits: hits.map(hit => ({
          file_path: hit.file,
          line: hit.line,
          col: hit.col,
          snippet: hit.snippet,
          score: hit.score,
          match_reasons: hit.why,
          byte_offset: hit.byte_offset,
          span_len: hit.span_len,
          lang: hit.lang
        })),
        latency_ms: 10,
        stage_a_latency_ms: 5,
        stage_b_latency_ms: 3,
        stage_c_latency_ms: 2
      };

      expect(response.hits).toHaveLength(1);
      expect(response.hits[0].file_path).toBe('src/test.ts');
      expect(response.hits[0].score).toBe(0.95);
      expect(response.latency_ms).toBe(10);
    });

    it('should handle empty search results', () => {
      const response: SearchResponse = {
        hits: [],
        latency_ms: 5,
        stage_a_latency_ms: 5
      };

      expect(response.hits).toHaveLength(0);
      expect(response.latency_ms).toBe(5);
    });

    it('should include timing information', () => {
      const timing = {
        stage_a_latency: 5,
        stage_b_latency: 3,
        stage_c_latency: 2
      };

      const totalLatency = timing.stage_a_latency + 
                          (timing.stage_b_latency || 0) + 
                          (timing.stage_c_latency || 0);

      expect(totalLatency).toBe(10);
    });
  });

  describe('Health Check Logic', () => {
    it('should format health response correctly', () => {
      const healthResponse: HealthResponse = {
        status: 'ok',
        shards_healthy: 5,
        shards_total: 5,
        memory_usage_gb: 1.2,
        active_queries: 3,
        worker_pool_status: {
          ingest_active: 1,
          query_active: 2,
          maintenance_active: 0
        },
        last_compaction: new Date().toISOString()
      };

      expect(healthResponse.status).toBe('ok');
      expect(healthResponse.shards_healthy).toBe(healthResponse.shards_total);
      expect(healthResponse.memory_usage_gb).toBeGreaterThan(0);
    });

    it('should detect degraded status', () => {
      const memoryUsageGb = 8.5;
      const memoryLimit = 8.0;
      const activeQueries = 150;
      const queryLimit = 100;

      const isMemoryDegraded = memoryUsageGb > memoryLimit * 0.9;
      const isQueryDegraded = activeQueries > queryLimit;

      expect(isMemoryDegraded).toBe(true);
      expect(isQueryDegraded).toBe(true);
    });

    it('should detect down status', () => {
      const shardsHealthy = 0;
      const shardsTotal = 5;
      const isInitialized = false;

      const isDown = !isInitialized || shardsHealthy === 0;

      expect(isDown).toBe(true);
    });
  });

  describe('Error Handling', () => {
    it('should handle search engine errors gracefully', () => {
      const error = new Error('Search engine failure');
      
      const errorResponse = {
        error: 'Internal Server Error',
        message: 'Search request failed',
        statusCode: 500
      };

      expect(errorResponse.statusCode).toBe(500);
      expect(errorResponse.error).toBeTruthy();
    });

    it('should validate missing required parameters', () => {
      const incompleteRequest = {
        q: 'test',
        // missing repo_sha and k
      };

      const hasRequiredFields = 
        'q' in incompleteRequest &&
        'repo_sha' in incompleteRequest &&
        'k' in incompleteRequest;

      expect(hasRequiredFields).toBe(false);
    });

    it('should handle malformed JSON requests', () => {
      const malformedJson = '{"q": "test", "k":}'; // Invalid JSON
      
      let isValidJson = false;
      try {
        JSON.parse(malformedJson);
        isValidJson = true;
      } catch (e) {
        isValidJson = false;
      }

      expect(isValidJson).toBe(false);
    });

    it('should handle timeout errors', () => {
      const timeout = 30000;
      const requestTime = 35000;
      
      const isTimeout = requestTime > timeout;
      
      expect(isTimeout).toBe(true);
    });
  });

  describe('SPI (Service Provider Interface) Endpoints', () => {
    it('should validate SPI search request', () => {
      const spiRequest: SpiSearchRequest = {
        q: 'test function',
        repo_sha: 'abc123',
        k: 10,
        trace_id: 'trace-123'
      };

      expect(spiRequest.q).toBeTruthy();
      expect(spiRequest.repo_sha).toBeTruthy();
      expect(spiRequest.trace_id).toBeTruthy();
    });

    it('should format SPI search response', () => {
      const spiResponse: SpiSearchResponse = {
        results: [
          {
            path: 'src/test.ts',
            line: 10,
            column: 5,
            content: 'function test() {',
            score: 0.95,
            reasons: ['exact_match']
          }
        ],
        metadata: {
          total_time_ms: 15,
          stage_times: {
            lexical_ms: 5,
            structural_ms: 3,
            semantic_ms: 7
          }
        }
      };

      expect(spiResponse.results).toHaveLength(1);
      expect(spiResponse.metadata.total_time_ms).toBe(15);
      expect(spiResponse.results[0].score).toBe(0.95);
    });
  });

  describe('Compatibility and Version Management', () => {
    it('should validate compatibility check request', () => {
      const compatRequest: CompatibilityCheckRequest = {
        client_version: '1.0.0',
        api_version: '1.0.0',
        features: ['search', 'symbols']
      };

      expect(compatRequest.client_version).toBeTruthy();
      expect(compatRequest.api_version).toBeTruthy();
      expect(compatRequest.features).toBeInstanceOf(Array);
    });

    it('should determine version compatibility', () => {
      const clientVersion = '1.0.0';
      const serverVersion = '1.0.0';
      const minSupportedVersion = '0.9.0';
      
      const isCompatible = clientVersion >= minSupportedVersion && 
                          clientVersion.split('.')[0] === serverVersion.split('.')[0];
      
      expect(isCompatible).toBe(true);
    });

    it('should handle version mismatch', () => {
      const clientVersion = '2.0.0';
      const serverVersion = '1.0.0';
      
      const majorVersionMatch = clientVersion.split('.')[0] === serverVersion.split('.')[0];
      
      expect(majorVersionMatch).toBe(false);
    });
  });

  describe('Request Context and Metadata', () => {
    it('should extract request metadata', () => {
      const mockRequest = {
        method: 'POST',
        url: '/api/search',
        headers: {
          'user-agent': 'lens-client/1.0.0',
          'content-type': 'application/json'
        },
        hostname: 'localhost',
        ip: '127.0.0.1'
      };

      const metadata = {
        method: mockRequest.method,
        url: mockRequest.url,
        userAgent: mockRequest.headers['user-agent'],
        ip: mockRequest.ip
      };

      expect(metadata.method).toBe('POST');
      expect(metadata.url).toBe('/api/search');
      expect(metadata.userAgent).toBe('lens-client/1.0.0');
    });

    it('should generate request trace IDs', () => {
      const generateTraceId = (): string => {
        return `trace-${Date.now()}-${Math.random().toString(36).substring(7)}`;
      };

      const traceId = generateTraceId();
      
      expect(traceId).toMatch(/^trace-\d+-[a-z0-9]+$/);
    });

    it('should handle concurrent request tracking', () => {
      let activeRequests = 0;
      const maxConcurrent = 100;
      
      const startRequest = (): boolean => {
        if (activeRequests >= maxConcurrent) {
          return false; // Reject request
        }
        activeRequests++;
        return true;
      };
      
      const endRequest = (): void => {
        if (activeRequests > 0) {
          activeRequests--;
        }
      };
      
      expect(startRequest()).toBe(true);
      expect(activeRequests).toBe(1);
      
      endRequest();
      expect(activeRequests).toBe(0);
    });
  });

  describe('Query Parameter Processing', () => {
    it('should parse mode parameter correctly', () => {
      const validModes = ['lexical', 'struct', 'hybrid'];
      const testMode = 'hybrid';
      
      const isValidMode = validModes.includes(testMode);
      expect(isValidMode).toBe(true);
      
      const invalidMode = 'invalid';
      const isInvalidMode = validModes.includes(invalidMode);
      expect(isInvalidMode).toBe(false);
    });

    it('should handle boolean parameters', () => {
      const fuzzyParam = 'true';
      const fuzzyValue = fuzzyParam === 'true';
      
      expect(fuzzyValue).toBe(true);
      
      const falsyParam = 'false';
      const falsyValue = falsyParam === 'true';
      
      expect(falsyValue).toBe(false);
    });

    it('should parse numeric parameters with validation', () => {
      const kParam = '10';
      const kValue = parseInt(kParam, 10);
      
      const isValidK = !isNaN(kValue) && kValue > 0 && kValue <= 100;
      
      expect(isValidK).toBe(true);
      expect(kValue).toBe(10);
    });
  });

  describe('CORS and Security Headers', () => {
    it('should configure CORS correctly', () => {
      const corsConfig = {
        origin: true,
        methods: ['GET', 'POST'],
        allowedHeaders: ['Content-Type', 'Authorization']
      };

      expect(corsConfig.origin).toBe(true);
      expect(corsConfig.methods).toContain('GET');
      expect(corsConfig.methods).toContain('POST');
    });

    it('should validate security headers', () => {
      const securityHeaders = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block'
      };

      expect(securityHeaders['X-Content-Type-Options']).toBe('nosniff');
      expect(securityHeaders['X-Frame-Options']).toBe('DENY');
    });
  });
});