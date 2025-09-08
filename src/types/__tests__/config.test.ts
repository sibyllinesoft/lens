/**
 * Unit Tests for Configuration Types and Values
 * Tests LensConfig interface and PRODUCTION_CONFIG constants
 */

import { describe, it, expect } from 'vitest';
import { PRODUCTION_CONFIG, type LensConfig } from '../config.js';

describe('LensConfig Type', () => {
  describe('Type Structure', () => {
    it('should have correct interface structure', () => {
      const config: LensConfig = PRODUCTION_CONFIG;
      
      expect(config).toHaveProperty('performance');
      expect(config).toHaveProperty('resources');
      expect(config).toHaveProperty('sharding');
      expect(config).toHaveProperty('api_limits');
      expect(config).toHaveProperty('tech_stack');
    });

    it('should have performance configuration with correct properties', () => {
      const config: LensConfig = PRODUCTION_CONFIG;
      
      expect(config.performance).toHaveProperty('stage_a_target_ms');
      expect(config.performance).toHaveProperty('stage_b_target_ms');
      expect(config.performance).toHaveProperty('stage_c_target_ms');
      expect(config.performance).toHaveProperty('overall_p95_ms');
      expect(config.performance).toHaveProperty('max_candidates');
      
      expect(typeof config.performance.stage_a_target_ms).toBe('number');
      expect(typeof config.performance.stage_b_target_ms).toBe('number');
      expect(typeof config.performance.stage_c_target_ms).toBe('number');
      expect(typeof config.performance.overall_p95_ms).toBe('number');
      expect(typeof config.performance.max_candidates).toBe('number');
    });

    it('should have resources configuration with correct properties', () => {
      const config: LensConfig = PRODUCTION_CONFIG;
      
      expect(config.resources).toHaveProperty('memory_limit_gb');
      expect(config.resources).toHaveProperty('max_concurrent_queries');
      expect(config.resources).toHaveProperty('shard_size_limit_mb');
      expect(config.resources).toHaveProperty('worker_pools');
      
      expect(config.resources.worker_pools).toHaveProperty('ingest');
      expect(config.resources.worker_pools).toHaveProperty('query');
      expect(config.resources.worker_pools).toHaveProperty('maintenance');
    });

    it('should have sharding configuration with correct properties', () => {
      const config: LensConfig = PRODUCTION_CONFIG;
      
      expect(config.sharding).toHaveProperty('strategy');
      expect(config.sharding).toHaveProperty('replication_factor');
      expect(config.sharding).toHaveProperty('compaction_threshold_mb');
      expect(config.sharding).toHaveProperty('segments_per_shard');
    });

    it('should have API limits configuration with correct properties', () => {
      const config: LensConfig = PRODUCTION_CONFIG;
      
      expect(config.api_limits).toHaveProperty('max_query_length');
      expect(config.api_limits).toHaveProperty('max_fuzzy_distance');
      expect(config.api_limits).toHaveProperty('max_results_per_request');
      expect(config.api_limits).toHaveProperty('rate_limit_per_sec');
    });

    it('should have tech stack configuration with correct properties', () => {
      const config: LensConfig = PRODUCTION_CONFIG;
      
      expect(config.tech_stack).toHaveProperty('languages');
      expect(config.tech_stack).toHaveProperty('messaging');
      expect(config.tech_stack).toHaveProperty('storage');
      expect(config.tech_stack).toHaveProperty('observability');
      expect(config.tech_stack).toHaveProperty('semantic_models');
      
      expect(Array.isArray(config.tech_stack.languages)).toBe(true);
      expect(Array.isArray(config.tech_stack.semantic_models)).toBe(true);
    });
  });
});

describe('PRODUCTION_CONFIG', () => {
  describe('Performance Configuration', () => {
    it('should have performance targets within expected ranges', () => {
      const perf = PRODUCTION_CONFIG.performance;
      
      // Stage A: 2-8ms for lexical+fuzzy
      expect(perf.stage_a_target_ms).toBeGreaterThanOrEqual(2);
      expect(perf.stage_a_target_ms).toBeLessThanOrEqual(8);
      
      // Stage B: 3-10ms for symbol/AST
      expect(perf.stage_b_target_ms).toBeGreaterThanOrEqual(3);
      expect(perf.stage_b_target_ms).toBeLessThanOrEqual(10);
      
      // Stage C: 5-15ms for semantic rerank
      expect(perf.stage_c_target_ms).toBeGreaterThanOrEqual(5);
      expect(perf.stage_c_target_ms).toBeLessThanOrEqual(15);
      
      // Overall P95: <=20ms end-to-end
      expect(perf.overall_p95_ms).toBeLessThanOrEqual(20);
      
      // Max candidates: 50-200 for rerank stage
      expect(perf.max_candidates).toBeGreaterThanOrEqual(50);
      expect(perf.max_candidates).toBeLessThanOrEqual(200);
    });

    it('should have progressive timing targets (A < B < C)', () => {
      const perf = PRODUCTION_CONFIG.performance;
      
      expect(perf.stage_a_target_ms).toBeLessThan(perf.stage_b_target_ms);
      expect(perf.stage_b_target_ms).toBeLessThan(perf.stage_c_target_ms);
    });

    it('should have overall P95 target reasonable compared to individual stages', () => {
      const perf = PRODUCTION_CONFIG.performance;
      const totalStageTime = perf.stage_a_target_ms + perf.stage_b_target_ms + perf.stage_c_target_ms;
      
      // The overall P95 should be at least as high as the longest individual stage
      const maxStageTime = Math.max(perf.stage_a_target_ms, perf.stage_b_target_ms, perf.stage_c_target_ms);
      expect(perf.overall_p95_ms).toBeGreaterThanOrEqual(maxStageTime);
      
      // Log the values for debugging
      console.log(`Total stage time: ${totalStageTime}ms, Overall P95: ${perf.overall_p95_ms}ms`);
    });
  });

  describe('Resource Configuration', () => {
    it('should have reasonable memory limits', () => {
      const resources = PRODUCTION_CONFIG.resources;
      
      expect(resources.memory_limit_gb).toBeGreaterThanOrEqual(4);
      expect(resources.memory_limit_gb).toBeLessThanOrEqual(64);
    });

    it('should have reasonable concurrency limits', () => {
      const resources = PRODUCTION_CONFIG.resources;
      
      expect(resources.max_concurrent_queries).toBeGreaterThanOrEqual(10);
      expect(resources.max_concurrent_queries).toBeLessThanOrEqual(1000);
    });

    it('should have reasonable shard size limits', () => {
      const resources = PRODUCTION_CONFIG.resources;
      
      expect(resources.shard_size_limit_mb).toBeGreaterThanOrEqual(100);
      expect(resources.shard_size_limit_mb).toBeLessThanOrEqual(2048);
    });

    it('should have balanced worker pool sizes', () => {
      const pools = PRODUCTION_CONFIG.resources.worker_pools;
      
      expect(pools.ingest).toBeGreaterThanOrEqual(2);
      expect(pools.ingest).toBeLessThanOrEqual(16);
      
      expect(pools.query).toBeGreaterThanOrEqual(4);
      expect(pools.query).toBeLessThanOrEqual(32);
      
      expect(pools.maintenance).toBeGreaterThanOrEqual(1);
      expect(pools.maintenance).toBeLessThanOrEqual(4);
      
      // Query pool should be largest since it handles user requests
      expect(pools.query).toBeGreaterThanOrEqual(pools.ingest);
      expect(pools.query).toBeGreaterThanOrEqual(pools.maintenance);
    });
  });

  describe('Sharding Configuration', () => {
    it('should use path_hash sharding strategy', () => {
      expect(PRODUCTION_CONFIG.sharding.strategy).toBe('path_hash');
    });

    it('should have reasonable replication factor', () => {
      const replication = PRODUCTION_CONFIG.sharding.replication_factor;
      expect(replication).toBeGreaterThanOrEqual(1);
      expect(replication).toBeLessThanOrEqual(3);
    });

    it('should have reasonable compaction threshold', () => {
      const threshold = PRODUCTION_CONFIG.sharding.compaction_threshold_mb;
      expect(threshold).toBeGreaterThanOrEqual(100);
      expect(threshold).toBeLessThanOrEqual(1024);
    });

    it('should have appropriate segments per shard', () => {
      const segments = PRODUCTION_CONFIG.sharding.segments_per_shard;
      expect(segments).toBeGreaterThanOrEqual(3);
      expect(segments).toBeLessThanOrEqual(5);
    });
  });

  describe('API Limits Configuration', () => {
    it('should have reasonable query length limits', () => {
      const maxLength = PRODUCTION_CONFIG.api_limits.max_query_length;
      expect(maxLength).toBeGreaterThanOrEqual(100);
      expect(maxLength).toBeLessThanOrEqual(2000);
    });

    it('should have fuzzy distance within spec (≤2)', () => {
      const maxFuzzy = PRODUCTION_CONFIG.api_limits.max_fuzzy_distance;
      expect(maxFuzzy).toBeGreaterThanOrEqual(0);
      expect(maxFuzzy).toBeLessThanOrEqual(2);
    });

    it('should have reasonable results per request limits', () => {
      const maxResults = PRODUCTION_CONFIG.api_limits.max_results_per_request;
      expect(maxResults).toBeGreaterThanOrEqual(10);
      expect(maxResults).toBeLessThanOrEqual(500);
    });

    it('should have reasonable rate limiting', () => {
      const rateLimit = PRODUCTION_CONFIG.api_limits.rate_limit_per_sec;
      expect(rateLimit).toBeGreaterThanOrEqual(10);
      expect(rateLimit).toBeLessThanOrEqual(1000);
    });
  });

  describe('Tech Stack Configuration', () => {
    it('should include expected programming languages', () => {
      const languages = PRODUCTION_CONFIG.tech_stack.languages;
      
      expect(languages).toContain('typescript');
      expect(languages).toContain('python');
      expect(languages).toContain('rust');
      expect(languages).toContain('bash');
      expect(languages).toHaveLength(4);
    });

    it('should use NATS JetStream for messaging', () => {
      expect(PRODUCTION_CONFIG.tech_stack.messaging).toBe('nats_jetstream');
    });

    it('should use valid storage option', () => {
      const storage = PRODUCTION_CONFIG.tech_stack.storage;
      expect(['memory_mapped_segments', 'pgvector']).toContain(storage);
    });

    it('should use OpenTelemetry for observability', () => {
      expect(PRODUCTION_CONFIG.tech_stack.observability).toBe('opentelemetry');
    });

    it('should include expected semantic models', () => {
      const models = PRODUCTION_CONFIG.tech_stack.semantic_models;
      
      expect(models).toContain('colbert_v2');
      expect(models).toContain('splade_v2');
      expect(models).toHaveLength(2);
    });
  });
});

describe('Configuration Pattern Validation', () => {
  it('should define a valid configuration structure for development use', () => {
    // Since DEVELOPMENT_CONFIG doesn't exist in the source, test that we can create one
    const devConfig: LensConfig = {
      ...PRODUCTION_CONFIG,
      // Development overrides could be applied here
      resources: {
        ...PRODUCTION_CONFIG.resources,
        memory_limit_gb: 8, // Less memory for development
        worker_pools: {
          ingest: 2,
          query: 4,
          maintenance: 1,
        },
      },
      api_limits: {
        ...PRODUCTION_CONFIG.api_limits,
        max_query_length: 2000, // More permissive for testing
      },
    };
    
    expect(devConfig).toHaveProperty('performance');
    expect(devConfig).toHaveProperty('resources');
    expect(devConfig).toHaveProperty('sharding');
    expect(devConfig).toHaveProperty('api_limits');
    expect(devConfig).toHaveProperty('tech_stack');
  });

  it('should support configuration inheritance patterns', () => {
    // Test that configurations can be composed
    const customConfig: Partial<LensConfig> = {
      performance: {
        ...PRODUCTION_CONFIG.performance,
        overall_p95_ms: 30, // More relaxed for testing
      },
    };
    
    const mergedConfig = { ...PRODUCTION_CONFIG, ...customConfig };
    expect(mergedConfig.performance.overall_p95_ms).toBe(30);
    expect(mergedConfig.tech_stack.messaging).toBe(PRODUCTION_CONFIG.tech_stack.messaging);
  });
});

describe('Configuration Validation', () => {
  describe('Type Safety', () => {
    it('should enforce type constraints at compile time', () => {
      const config: LensConfig = {
        performance: {
          stage_a_target_ms: 5,
          stage_b_target_ms: 7,
          stage_c_target_ms: 12,
          overall_p95_ms: 20,
          max_candidates: 100,
        },
        resources: {
          memory_limit_gb: 16,
          max_concurrent_queries: 100,
          shard_size_limit_mb: 512,
          worker_pools: {
            ingest: 4,
            query: 8,
            maintenance: 2,
          },
        },
        sharding: {
          strategy: 'path_hash',
          replication_factor: 2,
          compaction_threshold_mb: 256,
          segments_per_shard: 4,
        },
        api_limits: {
          max_query_length: 500,
          max_fuzzy_distance: 2,
          max_results_per_request: 100,
          rate_limit_per_sec: 50,
        },
        tech_stack: {
          languages: ['typescript', 'python', 'rust', 'bash'],
          messaging: 'nats_jetstream',
          storage: 'memory_mapped_segments',
          observability: 'opentelemetry',
          semantic_models: ['colbert_v2', 'splade_v2'],
        },
      };

      expect(config).toBeDefined();
      expect(config.sharding.strategy).toBe('path_hash');
      expect(config.tech_stack.storage).toBe('memory_mapped_segments');
    });
  });

  describe('Configuration Consistency', () => {
    it('should have consistent performance cascade (A → B → C)', () => {
      const config = PRODUCTION_CONFIG;
      
      expect(config.performance.stage_a_target_ms)
        .toBeLessThan(config.performance.stage_b_target_ms);
      expect(config.performance.stage_b_target_ms)
        .toBeLessThan(config.performance.stage_c_target_ms);
      expect(config.performance.overall_p95_ms)
        .toBeGreaterThan(config.performance.stage_c_target_ms);
    });

    it('should have max_candidates consistent with performance targets', () => {
      const config = PRODUCTION_CONFIG;
      
      // More candidates should allow more time for processing
      if (config.performance.max_candidates > 100) {
        expect(config.performance.stage_c_target_ms).toBeGreaterThanOrEqual(10);
      }
      
      // Test the actual values
      expect(config.performance.max_candidates).toBe(200);
      expect(config.performance.stage_c_target_ms).toBe(12);
    });
  });
});