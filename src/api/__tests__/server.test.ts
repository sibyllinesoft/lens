/**
 * Tests for Lens API Server
 * Priority: CRITICAL - Highest complexity (136), 2336 LOC, no existing tests
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import fastify from 'fastify';
import { initializeServer } from '../server.js';

// Mock the search engine first
const mockSearchEngine = {
  initialize: vi.fn().mockResolvedValue(undefined),
  search: vi.fn().mockResolvedValue({
    hits: [
      {
        file: 'test.ts',
        line: 1,
        col: 0,
        lang: 'typescript',
        snippet: 'function test() {}',
        score: 0.95,
        why: ['exact_match'],
        byte_offset: 0,
        span_len: 17,
      },
    ],
    stage_a_latency: 5,
    stage_b_latency: 3,
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
      maintenance_active: 0,
    },
    last_compaction: new Date(),
  }),
  getManifest: vi.fn().mockResolvedValue({
    'test-repo': {
      repo_sha: 'abc123',
      api_version: '1.0.0',
      index_version: '1.0.0',
      policy_version: '1.0.0',
    },
  }),
  setRerankingEnabled: vi.fn(),
  setPhaseBOptimizationsEnabled: vi.fn(),
  runPhaseBBenchmark: vi.fn().mockResolvedValue({
    overall_status: 'PASS',
    stage_a_p95_ms: 8.5,
    meets_performance_targets: true,
    meets_quality_targets: true,
  }),
  generateCalibrationPlot: vi.fn().mockResolvedValue({
    calibration_error: 0.05,
    reliability_score: 0.92,
    bins: [],
  }),
  updateStageAConfig: vi.fn().mockResolvedValue(undefined),
  updateSemanticConfig: vi.fn().mockResolvedValue(undefined),
  getASTCoverageStats: vi.fn().mockReturnValue({
    coverage: { coverage_percent: 25.5 },
    stats: { hits: 10, misses: 2 },
  }),
  shutdown: vi.fn().mockResolvedValue(undefined),
};

// Mock external dependencies after mockSearchEngine definition
vi.mock('../search-engine.js', () => ({
  LensSearchEngine: vi.fn().mockImplementation(() => mockSearchEngine),
}));

vi.mock('../../core/feature-flags.js', () => ({
  FeatureFlagManager: vi.fn().mockImplementation(() => ({
    isEnabled: vi.fn().mockReturnValue(false),
    getFlags: vi.fn().mockReturnValue({}),
    updateFlag: vi.fn(),
  })),
}));

vi.mock('../../deployment/canary-rollout-system.js', () => ({
  CanaryRolloutSystem: vi.fn().mockImplementation(() => ({
    getStatus: vi.fn().mockResolvedValue({
      status: 'stable',
      traffic_percentage: 0,
      experiments_active: 0,
    }),
    progressCanary: vi.fn().mockResolvedValue({ success: true }),
    killSwitch: vi.fn().mockResolvedValue({ success: true }),
  })),
}));

vi.mock('../../core/signoff-manager.js', () => ({
  SignoffManager: vi.fn().mockImplementation(() => ({
    getSignoffReport: vi.fn().mockResolvedValue({
      ready_for_production: true,
      gates_passed: 5,
      gates_total: 5,
    }),
    getValidationStatus: vi.fn().mockResolvedValue({
      overall_status: 'PASS',
      last_validation: new Date(),
    }),
    runNightlyValidation: vi.fn().mockResolvedValue({
      status: 'SUCCESS',
      duration_minutes: 15,
    }),
    runQualityGates: vi.fn().mockResolvedValue({
      status: 'PASS',
      gates_passed: 5,
    }),
  })),
}));

vi.mock('../../monitoring/phase-d-dashboards.js', () => ({
  PhaseDDashboards: vi.fn().mockImplementation(() => ({
    getDashboardData: vi.fn().mockResolvedValue({
      system_health: { status: 'healthy' },
      performance_metrics: { p95_latency: 15 },
      active_experiments: [],
    }),
  })),
}));

// Mock various other dependencies that might be imported
vi.mock('../../core/compatibility-checker.js', () => ({
  checkBundleCompatibility: vi.fn().mockResolvedValue({
    compatible: true,
    issues: [],
  }),
}));

describe('Lens API Server', () => {
  let app: any;

  beforeEach(async () => {
    vi.clearAllMocks();
    app = await initializeServer();
  });

  afterEach(async () => {
    if (app) {
      await app.close();
    }
  });

  describe('Server Initialization', () => {
    it('should initialize server successfully', () => {
      expect(app).toBeDefined();
      expect(app.server).toBeDefined();
    });

    it('should register CORS plugin', async () => {
      const response = await app.inject({
        method: 'OPTIONS',
        url: '/health',
        headers: {
          origin: 'http://localhost:3000',
          'access-control-request-method': 'GET',
        },
      });

      expect(response.statusCode).toBe(204);
    });
  });

  describe('Health Endpoint', () => {
    it('should return health status', async () => {
      const response = await app.inject({
        method: 'GET',
        url: '/health',
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('status');
      expect(data).toHaveProperty('shards_healthy');
      expect(data).toHaveProperty('shards_total');
    });

    it('should handle health check errors', async () => {
      mockSearchEngine.getHealthStatus.mockRejectedValueOnce(new Error('Health check failed'));

      const response = await app.inject({
        method: 'GET',
        url: '/health',
      });

      expect(response.statusCode).toBe(500);
    });
  });

  describe('Search Endpoint', () => {
    it('should perform search successfully', async () => {
      const searchPayload = {
        q: 'test function',
        repo_sha: 'abc123',
        k: 10,
        mode: 'hybrid',
      };

      const response = await app.inject({
        method: 'POST',
        url: '/search',
        payload: searchPayload,
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('hits');
      expect(data.hits).toHaveLength(1);
      expect(data.hits[0]).toMatchObject({
        file: 'test.ts',
        score: 0.95,
      });
    });

    it('should validate search request parameters', async () => {
      const invalidPayload = {
        // Missing required fields
        k: 10,
      };

      const response = await app.inject({
        method: 'POST',
        url: '/search',
        payload: invalidPayload,
      });

      expect(response.statusCode).toBe(400);
    });

    it('should handle search errors', async () => {
      mockSearchEngine.search.mockRejectedValueOnce(new Error('Search failed'));

      const searchPayload = {
        q: 'test function',
        repo_sha: 'abc123',
        k: 10,
        mode: 'hybrid',
      };

      const response = await app.inject({
        method: 'POST',
        url: '/search',
        payload: searchPayload,
      });

      expect(response.statusCode).toBe(500);
    });

    it('should handle large k values appropriately', async () => {
      const searchPayload = {
        q: 'test function',
        repo_sha: 'abc123',
        k: 10000, // Very large k
        mode: 'hybrid',
      };

      const response = await app.inject({
        method: 'POST',
        url: '/search',
        payload: searchPayload,
      });

      // Should either succeed or return 400 for too large k
      expect([200, 400]).toContain(response.statusCode);
    });
  });

  describe('Manifest Endpoint', () => {
    it('should return repository manifest', async () => {
      const response = await app.inject({
        method: 'GET',
        url: '/manifest',
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('test-repo');
      expect(data['test-repo']).toMatchObject({
        repo_sha: 'abc123',
        api_version: '1.0.0',
      });
    });
  });

  describe('Configuration Endpoints', () => {
    it('should enable reranker', async () => {
      const response = await app.inject({
        method: 'POST',
        url: '/reranker/enable',
        payload: { enabled: true },
      });

      expect(response.statusCode).toBe(200);
      expect(mockSearchEngine.setRerankingEnabled).toHaveBeenCalledWith(true);
    });

    it('should enable Phase B optimizations', async () => {
      const response = await app.inject({
        method: 'POST',
        url: '/policy/phaseB/enable',
        payload: { enabled: true },
      });

      expect(response.statusCode).toBe(200);
      expect(mockSearchEngine.setPhaseBOptimizationsEnabled).toHaveBeenCalledWith(true);
    });

    it('should update Stage A configuration', async () => {
      const config = {
        rare_term_fuzzy: true,
        synonyms_when_identifier_density_below: 0.8,
        prefilter_enabled: true,
      };

      const response = await app.inject({
        method: 'PATCH',
        url: '/policy/stageA',
        payload: config,
      });

      expect(response.statusCode).toBe(200);
      expect(mockSearchEngine.updateStageAConfig).toHaveBeenCalledWith(config);
    });

    it('should update semantic configuration', async () => {
      const config = {
        nl_threshold: 0.4,
        min_candidates: 15,
        efSearch: 200,
      };

      const response = await app.inject({
        method: 'PATCH',
        url: '/policy/stageC',
        payload: config,
      });

      expect(response.statusCode).toBe(200);
      expect(mockSearchEngine.updateSemanticConfig).toHaveBeenCalledWith(config);
    });
  });

  describe('Benchmark Endpoints', () => {
    it('should run Phase B benchmark', async () => {
      const response = await app.inject({
        method: 'POST',
        url: '/bench/phaseB',
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      expect(data).toMatchObject({
        overall_status: 'PASS',
        stage_a_p95_ms: 8.5,
      });
    });

    it('should generate calibration plot', async () => {
      const response = await app.inject({
        method: 'GET',
        url: '/reports/calibration-plot',
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('calibration_error');
      expect(data).toHaveProperty('reliability_score');
    });
  });

  describe('Canary System Endpoints', () => {
    it('should return canary status', async () => {
      const response = await app.inject({
        method: 'GET',
        url: '/canary/status',
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('status');
    });

    it('should progress canary rollout', async () => {
      const response = await app.inject({
        method: 'POST',
        url: '/canary/progress',
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('success');
    });

    it('should trigger kill switch', async () => {
      const response = await app.inject({
        method: 'POST',
        url: '/canary/killswitch',
        payload: { reason: 'Test kill switch' },
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('success');
    });
  });

  describe('Validation Endpoints', () => {
    it('should return signoff report', async () => {
      const response = await app.inject({
        method: 'GET',
        url: '/validation/signoff-report',
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('ready_for_production');
    });

    it('should return validation status', async () => {
      const response = await app.inject({
        method: 'GET',
        url: '/validation/status',
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('overall_status');
    });

    it('should run nightly validation', async () => {
      const response = await app.inject({
        method: 'POST',
        url: '/validation/nightly',
        payload: { duration_minutes: 30 },
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('status');
    });

    it('should run quality gates', async () => {
      const response = await app.inject({
        method: 'POST',
        url: '/validation/quality-gates',
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('status');
    });
  });

  describe('Coverage Endpoints', () => {
    it('should return AST coverage stats', async () => {
      const response = await app.inject({
        method: 'GET',
        url: '/coverage/ast',
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('coverage');
      expect(data).toHaveProperty('stats');
    });
  });

  describe('Monitoring Dashboard', () => {
    it('should return dashboard data', async () => {
      const response = await app.inject({
        method: 'GET',
        url: '/monitoring/dashboard',
      });

      expect(response.statusCode).toBe(200);
      const data = JSON.parse(response.payload);
      expect(data).toHaveProperty('system_health');
    });
  });

  describe('Error Handling', () => {
    it('should handle 404 for unknown endpoints', async () => {
      const response = await app.inject({
        method: 'GET',
        url: '/unknown-endpoint',
      });

      expect(response.statusCode).toBe(404);
    });

    it('should handle malformed JSON', async () => {
      const response = await app.inject({
        method: 'POST',
        url: '/search',
        payload: '{ invalid json',
        headers: {
          'content-type': 'application/json',
        },
      });

      expect(response.statusCode).toBe(400);
    });

    it('should have custom error handler', async () => {
      // Force an error by calling a method that doesn't exist
      mockSearchEngine.getHealthStatus.mockImplementation(() => {
        throw new Error('Simulated error');
      });

      const response = await app.inject({
        method: 'GET',
        url: '/health',
      });

      expect(response.statusCode).toBe(500);
    });
  });

  describe('Request Validation', () => {
    it('should require valid content-type for POST requests', async () => {
      const response = await app.inject({
        method: 'POST',
        url: '/search',
        payload: JSON.stringify({
          q: 'test',
          repo_sha: 'abc123',
          k: 10,
        }),
        headers: {
          'content-type': 'text/plain', // Wrong content type
        },
      });

      expect(response.statusCode).toBe(400);
    });

    it('should handle empty request bodies', async () => {
      const response = await app.inject({
        method: 'POST',
        url: '/search',
        payload: '',
        headers: {
          'content-type': 'application/json',
        },
      });

      expect(response.statusCode).toBe(400);
    });
  });

  describe('Performance', () => {
    it('should handle concurrent requests', async () => {
      const requests = Array(10).fill(0).map(() => 
        app.inject({
          method: 'GET',
          url: '/health',
        })
      );

      const responses = await Promise.all(requests);
      
      responses.forEach(response => {
        expect(response.statusCode).toBe(200);
      });
    });
  });
});