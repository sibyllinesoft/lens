import { describe, it, expect, beforeEach, vi } from 'vitest';
import { searchHandler, healthHandler, metricsHandler } from '../handlers.js';
import type { FastifyRequest, FastifyReply } from 'fastify';

// Mock the search engine
vi.mock('../../core/lens-engine.js', () => ({
  LensEngine: vi.fn().mockImplementation(() => ({
    search: vi.fn().mockResolvedValue({
      results: [
        { path: '/test/file1.ts', score: 0.95, content: 'function test() {}' },
        { path: '/test/file2.ts', score: 0.88, content: 'class TestClass {}' }
      ],
      total_time_ms: 12,
      stage_times: { stage_a: 8, stage_b: 3, stage_c: 1 },
      total_results: 2
    }))
  }))
}));

// Mock the performance monitor
vi.mock('../../core/performance-monitor.js', () => ({
  PerformanceMonitor: {
    getMetrics: vi.fn().mockReturnValue({
      requests_per_second: 1250.5,
      avg_latency_ms: 15.2,
      p95_latency_ms: 28.3,
      error_rate: 0.002,
      uptime_ms: 3600000,
      cache_hit_rate: 0.85,
      active_connections: 42
    }),
    incrementRequests: vi.fn(),
    recordLatency: vi.fn()
  }
}));

describe('API Handlers', () => {
  let mockRequest: Partial<FastifyRequest>;
  let mockReply: Partial<FastifyReply>;

  beforeEach(() => {
    vi.clearAllMocks();
    
    mockReply = {
      code: vi.fn().mockReturnThis(),
      send: vi.fn().mockReturnThis(),
      header: vi.fn().mockReturnThis(),
      type: vi.fn().mockReturnThis()
    };
  });

  describe('searchHandler', () => {
    beforeEach(() => {
      mockRequest = {
        query: {
          q: 'test query',
          mode: 'lexical',
          k: '10',
          fuzzy_distance: '2'
        },
        headers: {
          'user-agent': 'test-client/1.0'
        },
        ip: '127.0.0.1'
      };
    });

    it('should handle basic search requests', async () => {
      await searchHandler(mockRequest as FastifyRequest, mockReply as FastifyReply);

      expect(mockReply.send).toHaveBeenCalledWith({
        results: [
          { path: '/test/file1.ts', score: 0.95, content: 'function test() {}' },
          { path: '/test/file2.ts', score: 0.88, content: 'class TestClass {}' }
        ],
        metadata: {
          total_results: 2,
          total_time_ms: 12,
          stage_times: { stage_a: 8, stage_b: 3, stage_c: 1 }
        }
      });
    });

    it('should handle requests with all parameters', async () => {
      mockRequest.query = {
        q: 'function search test',
        mode: 'hybrid',
        k: '25',
        fuzzy_distance: '3',
        repo_sha: 'abc123def456'
      };

      await searchHandler(mockRequest as FastifyRequest, mockReply as FastifyReply);

      expect(mockReply.send).toHaveBeenCalledWith(
        expect.objectContaining({
          results: expect.any(Array),
          metadata: expect.objectContaining({
            total_results: expect.any(Number),
            total_time_ms: expect.any(Number)
          })
        })
      );
    });

    it('should handle missing query parameter', async () => {
      mockRequest.query = {};

      await searchHandler(mockRequest as FastifyRequest, mockReply as FastifyReply);

      expect(mockReply.code).toHaveBeenCalledWith(400);
      expect(mockReply.send).toHaveBeenCalledWith({
        error: 'Missing required parameter: q'
      });
    });

    it('should handle empty query parameter', async () => {
      mockRequest.query = { q: '' };

      await searchHandler(mockRequest as FastifyRequest, mockReply as FastifyReply);

      expect(mockReply.code).toHaveBeenCalledWith(400);
      expect(mockReply.send).toHaveBeenCalledWith({
        error: 'Query parameter cannot be empty'
      });
    });

    it('should validate numeric parameters', async () => {
      mockRequest.query = {
        q: 'test',
        k: 'not-a-number'
      };

      await searchHandler(mockRequest as FastifyRequest, mockReply as FastifyReply);

      expect(mockReply.code).toHaveBeenCalledWith(400);
      expect(mockReply.send).toHaveBeenCalledWith({
        error: 'Invalid parameter format: k must be a number'
      });
    });

    it('should enforce parameter limits', async () => {
      mockRequest.query = {
        q: 'test',
        k: '1000' // Too large
      };

      await searchHandler(mockRequest as FastifyRequest, mockReply as FastifyReply);

      expect(mockReply.code).toHaveBeenCalledWith(400);
      expect(mockReply.send).toHaveBeenCalledWith({
        error: 'Parameter out of range: k must be between 1 and 100'
      });
    });

    it('should handle search engine errors', async () => {
      const { LensEngine } = await import('../../core/lens-engine.js');
      const mockEngine = new LensEngine('/tmp');
      mockEngine.search = vi.fn().mockRejectedValue(new Error('Search failed'));

      // Mock the module to return the failing engine
      vi.doMock('../../core/lens-engine.js', () => ({
        LensEngine: vi.fn().mockImplementation(() => mockEngine)
      }));

      await searchHandler(mockRequest as FastifyRequest, mockReply as FastifyReply);

      expect(mockReply.code).toHaveBeenCalledWith(500);
      expect(mockReply.send).toHaveBeenCalledWith({
        error: 'Internal search error'
      });
    });

    it('should set proper response headers', async () => {
      await searchHandler(mockRequest as FastifyRequest, mockReply as FastifyReply);

      expect(mockReply.header).toHaveBeenCalledWith('Content-Type', 'application/json');
      expect(mockReply.header).toHaveBeenCalledWith('X-Response-Time', expect.any(String));
    });

    it('should handle concurrent requests', async () => {
      const promises = Array.from({ length: 5 }, () =>
        searchHandler(mockRequest as FastifyRequest, mockReply as FastifyReply)
      );

      await Promise.all(promises);

      expect(mockReply.send).toHaveBeenCalledTimes(5);
    });
  });

  describe('healthHandler', () => {
    beforeEach(() => {
      mockRequest = {
        headers: {},
        ip: '127.0.0.1'
      };
    });

    it('should return basic health status', async () => {
      await healthHandler(mockRequest as FastifyRequest, mockReply as FastifyReply);

      expect(mockReply.send).toHaveBeenCalledWith({
        status: 'healthy',
        timestamp: expect.any(String),
        uptime: expect.any(Number),
        version: expect.any(String)
      });
    });

    it('should include detailed health information with verbose flag', async () => {
      mockRequest.query = { verbose: 'true' };

      await healthHandler(mockRequest as FastifyRequest, mockReply as FastifyReply);

      expect(mockReply.send).toHaveBeenCalledWith(
        expect.objectContaining({
          status: 'healthy',
          timestamp: expect.any(String),
          uptime: expect.any(Number),
          version: expect.any(String),
          system: expect.objectContaining({
            memory: expect.any(Object),
            cpu: expect.any(Object),
            load: expect.any(Array)
          }),
          engine: expect.objectContaining({
            indexed_files: expect.any(Number),
            index_size_mb: expect.any(Number),
            last_updated: expect.any(String)
          })
        })
      );
    });

    it('should set health check response headers', async () => {
      await healthHandler(mockRequest as FastifyRequest, mockReply as FastifyReply);

      expect(mockReply.header).toHaveBeenCalledWith('Content-Type', 'application/json');
      expect(mockReply.header).toHaveBeenCalledWith('Cache-Control', 'no-cache');
    });

    it('should handle health check errors gracefully', async () => {
      // Mock process.memoryUsage to throw error
      const originalMemoryUsage = process.memoryUsage;
      process.memoryUsage = vi.fn().mockImplementation(() => {
        throw new Error('Memory info unavailable');
      });

      await healthHandler(mockRequest as FastifyRequest, mockReply as FastifyReply);

      // Should still return healthy but with limited info
      expect(mockReply.send).toHaveBeenCalledWith(
        expect.objectContaining({
          status: 'healthy',
          timestamp: expect.any(String)
        })
      );

      process.memoryUsage = originalMemoryUsage;
    });
  });

  describe('metricsHandler', () => {
    beforeEach(() => {
      mockRequest = {
        headers: {},
        ip: '127.0.0.1'
      };
    });

    it('should return performance metrics', async () => {
      await metricsHandler(mockRequest as FastifyRequest, mockReply as FastifyReply);

      expect(mockReply.send).toHaveBeenCalledWith({
        performance: {
          requests_per_second: 1250.5,
          avg_latency_ms: 15.2,
          p95_latency_ms: 28.3,
          error_rate: 0.002,
          uptime_ms: 3600000,
          cache_hit_rate: 0.85,
          active_connections: 42
        },
        timestamp: expect.any(String)
      });
    });

    it('should return metrics in Prometheus format when requested', async () => {
      mockRequest.headers = { accept: 'text/plain' };

      await metricsHandler(mockRequest as FastifyRequest, mockReply as FastifyReply);

      expect(mockReply.type).toHaveBeenCalledWith('text/plain');
      expect(mockReply.send).toHaveBeenCalledWith(
        expect.stringContaining('# HELP lens_requests_per_second')
      );
      expect(mockReply.send).toHaveBeenCalledWith(
        expect.stringContaining('lens_requests_per_second 1250.5')
      );
    });

    it('should handle metrics collection errors', async () => {
      // Mock PerformanceMonitor.getMetrics to throw error
      const { PerformanceMonitor } = await import('../../core/performance-monitor.js');
      PerformanceMonitor.getMetrics = vi.fn().mockImplementation(() => {
        throw new Error('Metrics unavailable');
      });

      await metricsHandler(mockRequest as FastifyRequest, mockReply as FastifyReply);

      expect(mockReply.code).toHaveBeenCalledWith(503);
      expect(mockReply.send).toHaveBeenCalledWith({
        error: 'Metrics temporarily unavailable'
      });
    });

    it('should set appropriate cache headers for metrics', async () => {
      await metricsHandler(mockRequest as FastifyRequest, mockReply as FastifyReply);

      expect(mockReply.header).toHaveBeenCalledWith('Cache-Control', 'max-age=30');
    });

    it('should include system resource metrics in detailed mode', async () => {
      mockRequest.query = { detailed: 'true' };

      await metricsHandler(mockRequest as FastifyRequest, mockReply as FastifyReply);

      expect(mockReply.send).toHaveBeenCalledWith(
        expect.objectContaining({
          performance: expect.any(Object),
          system: expect.objectContaining({
            memory_usage_mb: expect.any(Number),
            cpu_usage_percent: expect.any(Number),
            disk_usage: expect.any(Object),
            gc_stats: expect.any(Object)
          }),
          timestamp: expect.any(String)
        })
      );
    });
  });

  describe('Error Handling', () => {
    it('should handle malformed requests gracefully', async () => {
      mockRequest = {
        query: null,
        headers: {},
        ip: '127.0.0.1'
      };

      await searchHandler(mockRequest as FastifyRequest, mockReply as FastifyReply);

      expect(mockReply.code).toHaveBeenCalledWith(400);
      expect(mockReply.send).toHaveBeenCalledWith({
        error: 'Malformed request'
      });
    });

    it('should handle very large query strings', async () => {
      mockRequest = {
        query: {
          q: 'x'.repeat(10000) // Very long query
        },
        headers: {},
        ip: '127.0.0.1'
      };

      await searchHandler(mockRequest as FastifyRequest, mockReply as FastifyReply);

      expect(mockReply.code).toHaveBeenCalledWith(400);
      expect(mockReply.send).toHaveBeenCalledWith({
        error: 'Query too long (max 1000 characters)'
      });
    });

    it('should handle requests with invalid characters', async () => {
      mockRequest = {
        query: {
          q: 'test\x00\x01\x02' // Control characters
        },
        headers: {},
        ip: '127.0.0.1'
      };

      await searchHandler(mockRequest as FastifyRequest, mockReply as FastifyReply);

      expect(mockReply.code).toHaveBeenCalledWith(400);
      expect(mockReply.send).toHaveBeenCalledWith({
        error: 'Query contains invalid characters'
      });
    });
  });

  describe('Rate Limiting Integration', () => {
    it('should track request rates per IP', async () => {
      mockRequest = {
        query: { q: 'test' },
        headers: {},
        ip: '192.168.1.100'
      };

      await searchHandler(mockRequest as FastifyRequest, mockReply as FastifyReply);

      // Verify performance monitor is called to track the request
      const { PerformanceMonitor } = await import('../../core/performance-monitor.js');
      expect(PerformanceMonitor.incrementRequests).toHaveBeenCalledWith('192.168.1.100');
    });

    it('should record latency metrics', async () => {
      const startTime = Date.now();
      
      await searchHandler(mockRequest as FastifyRequest, mockReply as FastifyReply);

      const { PerformanceMonitor } = await import('../../core/performance-monitor.js');
      expect(PerformanceMonitor.recordLatency).toHaveBeenCalledWith(
        expect.any(Number)
      );
    });
  });
});