/**
 * Unit Tests for Benchmark Types and Schemas
 * Tests all Zod schemas, type validations, and constants
 */

import { describe, it, expect } from 'vitest';
import {
  BenchmarkConfigSchema,
  RepoSnapshotSchema,
  GoldenDataItemSchema,
  BenchmarkRunSchema,
  ABTestResultSchema,
  MetamorphicTestSchema,
  RobustnessTestSchema,
  BenchmarkPlanMessageSchema,
  BenchmarkRunMessageSchema,
  BenchmarkResultMessageSchema,
  PROMOTION_GATE_CRITERIA,
  type BenchmarkConfig,
  type RepoSnapshot,
  type GoldenDataItem,
  type BenchmarkRun,
  type ABTestResult,
  type MetamorphicTest,
  type RobustnessTest,
  type BenchmarkPlanMessage,
  type BenchmarkRunMessage,
  type BenchmarkResultMessage,
  type ConfigFingerprint,
  type BenchmarkOrchestrationConfig,
  type ReportData,
} from '../../benchmarks/src.js';

describe('Benchmark Types - Schema Validation', () => {
  describe('BenchmarkConfigSchema', () => {
    const validConfig: BenchmarkConfig = {
      trace_id: '123e4567-e89b-12d3-a456-426614174000',
      suite: ['codesearch', 'structural'],
      systems: ['lex', '+symbols', '+symbols+semantic'],
      slices: 'SMOKE_DEFAULT',
      seeds: 3,
      api_base_url: 'https://api.example.com',
      k: 50,
      include_cold_start: true,
      batch_size: 25,
      cache_mode: 'warm',
      robustness: false,
      metamorphic: true,
      k_candidates: 200,
      top_n: 50,
      fuzzy: 2,
      subtokens: true,
      semantic_gating: {
        nl_likelihood_threshold: 0.7,
        min_candidates: 15
      },
      latency_budgets: {
        stage_a_ms: 150,
        stage_b_ms: 250,
        stage_c_ms: 350
      }
    };

    it('should validate a complete valid configuration', () => {
      const result = BenchmarkConfigSchema.safeParse(validConfig);
      expect(result.success).toBe(true);
    });

    it('should require trace_id to be a valid UUID', () => {
      const invalid = { ...validConfig, trace_id: 'invalid-uuid' };
      const result = BenchmarkConfigSchema.safeParse(invalid);
      expect(result.success).toBe(false);
    });

    it('should validate suite enum values', () => {
      const invalidSuite = { ...validConfig, suite: ['invalid_suite'] };
      const result = BenchmarkConfigSchema.safeParse(invalidSuite);
      expect(result.success).toBe(false);
      
      const validSuite = { ...validConfig, suite: ['docs'] };
      const result2 = BenchmarkConfigSchema.safeParse(validSuite);
      expect(result2.success).toBe(true);
    });

    it('should validate seeds range (1-5)', () => {
      const tooLow = { ...validConfig, seeds: 0 };
      const tooHigh = { ...validConfig, seeds: 6 };
      
      expect(BenchmarkConfigSchema.safeParse(tooLow).success).toBe(false);
      expect(BenchmarkConfigSchema.safeParse(tooHigh).success).toBe(false);
    });

    it('should validate URL format for api_base_url', () => {
      const invalidUrl = { ...validConfig, api_base_url: 'not-a-url' };
      expect(BenchmarkConfigSchema.safeParse(invalidUrl).success).toBe(false);
    });

    it('should validate k parameter range (1-200)', () => {
      const valid = { ...validConfig, k: 100 };
      const tooLow = { ...validConfig, k: 0 };
      const tooHigh = { ...validConfig, k: 201 };
      
      expect(BenchmarkConfigSchema.safeParse(valid).success).toBe(true);
      expect(BenchmarkConfigSchema.safeParse(tooLow).success).toBe(false);
      expect(BenchmarkConfigSchema.safeParse(tooHigh).success).toBe(false);
    });

    it('should validate cache_mode options', () => {
      const singleMode = { ...validConfig, cache_mode: 'cold' };
      const multiMode = { ...validConfig, cache_mode: ['warm', 'cold'] };
      const invalid = { ...validConfig, cache_mode: 'invalid' };
      
      expect(BenchmarkConfigSchema.safeParse(singleMode).success).toBe(true);
      expect(BenchmarkConfigSchema.safeParse(multiMode).success).toBe(true);
      expect(BenchmarkConfigSchema.safeParse(invalid).success).toBe(false);
    });

    it('should apply default values correctly', () => {
      const minimal = {
        trace_id: '123e4567-e89b-12d3-a456-426614174000',
        suite: ['codesearch'] as const,
        systems: ['lex'],
        slices: 'ALL' as const
      };
      
      const result = BenchmarkConfigSchema.parse(minimal);
      
      expect(result.seeds).toBe(1); // default
      expect(result.cache_mode).toBe('warm'); // default
      expect(result.robustness).toBe(false); // default
      expect(result.k_candidates).toBe(200); // default
    });

    it('should validate semantic_gating configuration', () => {
      const validGating = {
        ...validConfig,
        semantic_gating: {
          nl_likelihood_threshold: 0.8,
          min_candidates: 20
        }
      };
      
      const invalidThreshold = {
        ...validConfig,
        semantic_gating: {
          nl_likelihood_threshold: 1.5, // > 1
          min_candidates: 10
        }
      };
      
      expect(BenchmarkConfigSchema.safeParse(validGating).success).toBe(true);
      expect(BenchmarkConfigSchema.safeParse(invalidThreshold).success).toBe(false);
    });
  });

  describe('RepoSnapshotSchema', () => {
    const validSnapshot: RepoSnapshot = {
      repo_ref: 'main',
      repo_sha: 'abcd1234567890abcd1234567890abcd12345678',
      manifest: {
        'src/index.ts': { lines: 100, modified: '2024-01-01' },
        'README.md': { lines: 50, modified: '2024-01-02' }
      },
      timestamp: '2024-01-15T10:30:00Z',
      language_distribution: {
        'typescript': 0.7,
        'javascript': 0.2,
        'markdown': 0.1
      },
      total_files: 150,
      total_lines: 5000
    };

    it('should validate a complete valid repo snapshot', () => {
      const result = RepoSnapshotSchema.safeParse(validSnapshot);
      expect(result.success).toBe(true);
    });

    it('should require repo_sha to be exactly 40 characters', () => {
      const tooShort = { ...validSnapshot, repo_sha: 'abc123' };
      const tooLong = { ...validSnapshot, repo_sha: 'a'.repeat(41) };
      
      expect(RepoSnapshotSchema.safeParse(tooShort).success).toBe(false);
      expect(RepoSnapshotSchema.safeParse(tooLong).success).toBe(false);
    });

    it('should validate datetime format for timestamp', () => {
      const invalidDate = { ...validSnapshot, timestamp: 'invalid-date' };
      expect(RepoSnapshotSchema.safeParse(invalidDate).success).toBe(false);
    });

    it('should validate non-negative integers for totals', () => {
      const negative = { ...validSnapshot, total_files: -1 };
      expect(RepoSnapshotSchema.safeParse(negative).success).toBe(false);
    });
  });

  describe('GoldenDataItemSchema', () => {
    const validGoldenItem: GoldenDataItem = {
      id: 'test-query-001',
      query: 'function processData',
      query_class: 'identifier',
      language: 'ts',
      source: 'pr-derived',
      snapshot_sha: 'abcd1234567890abcd1234567890abcd12345678',
      slice_tags: ['core', 'utilities'],
      expected_results: [
        {
          file: 'src/process.ts',
          line: 15,
          col: 0,
          relevance_score: 0.95,
          match_type: 'exact',
          why: 'Direct function name match'
        }
      ],
      metadata: {
        created_by: 'test-engineer',
        reviewed: true
      }
    };

    it('should validate a complete golden data item', () => {
      const result = GoldenDataItemSchema.safeParse(validGoldenItem);
      expect(result.success).toBe(true);
    });

    it('should validate query_class enum', () => {
      const invalid = { ...validGoldenItem, query_class: 'invalid-class' };
      expect(GoldenDataItemSchema.safeParse(invalid).success).toBe(false);
      
      const validClasses = ['identifier', 'regex', 'nl-ish', 'structural', 'docs'] as const;
      validClasses.forEach(cls => {
        const valid = { ...validGoldenItem, query_class: cls };
        expect(GoldenDataItemSchema.safeParse(valid).success).toBe(true);
      });
    });

    it('should validate language enum', () => {
      const invalid = { ...validGoldenItem, language: 'invalid-lang' };
      expect(GoldenDataItemSchema.safeParse(invalid).success).toBe(false);
      
      const validLangs = ['ts', 'py', 'rust', 'bash', 'go', 'java'] as const;
      validLangs.forEach(lang => {
        const valid = { ...validGoldenItem, language: lang };
        expect(GoldenDataItemSchema.safeParse(valid).success).toBe(true);
      });
    });

    it('should validate expected_results structure', () => {
      const invalidResults = {
        ...validGoldenItem,
        expected_results: [
          {
            file: 'test.ts',
            line: 0, // line must be >= 1
            col: 0,
            relevance_score: 1.5, // must be <= 1
            match_type: 'exact'
          }
        ]
      };
      
      expect(GoldenDataItemSchema.safeParse(invalidResults).success).toBe(false);
    });

    it('should validate relevance_score range (0-1)', () => {
      const invalidScore = {
        ...validGoldenItem,
        expected_results: [
          {
            ...validGoldenItem.expected_results[0],
            relevance_score: 2.0
          }
        ]
      };
      
      expect(GoldenDataItemSchema.safeParse(invalidScore).success).toBe(false);
    });
  });

  describe('BenchmarkRunSchema', () => {
    const validRun: BenchmarkRun = {
      trace_id: '123e4567-e89b-12d3-a456-426614174000',
      config_fingerprint: 'sha256:abc123',
      timestamp: '2024-01-15T10:30:00Z',
      status: 'completed',
      system: '+symbols+semantic',
      total_queries: 100,
      completed_queries: 98,
      failed_queries: 2,
      metrics: {
        recall_at_10: 0.85,
        recall_at_50: 0.92,
        ndcg_at_10: 0.78,
        mrr: 0.65,
        first_relevant_tokens: 15.5,
        stage_latencies: {
          stage_a_p50: 45.2,
          stage_a_p95: 78.1,
          stage_b_p50: 123.4,
          stage_b_p95: 245.6,
          stage_c_p50: 67.8,
          stage_c_p95: 134.5,
          e2e_p50: 235.6,
          e2e_p95: 458.2
        },
        fan_out_sizes: {
          stage_a: 1000,
          stage_b: 200,
          stage_c: 50
        },
        why_attributions: {
          'exact': 45,
          'fuzzy': 23,
          'semantic': 30
        },
        cbu_score: 0.82,
        ece_score: 0.03,
        kv_reuse_rate: 0.67
      },
      errors: [
        {
          query_id: 'query-123',
          error_type: 'timeout',
          message: 'Query exceeded time limit',
          stage: 'stage_c'
        }
      ]
    };

    it('should validate a complete benchmark run', () => {
      const result = BenchmarkRunSchema.safeParse(validRun);
      expect(result.success).toBe(true);
    });

    it('should validate status enum values', () => {
      const statuses = ['running', 'completed', 'failed', 'cancelled'] as const;
      
      statuses.forEach(status => {
        const valid = { ...validRun, status };
        expect(BenchmarkRunSchema.safeParse(valid).success).toBe(true);
      });
      
      const invalid = { ...validRun, status: 'invalid-status' };
      expect(BenchmarkRunSchema.safeParse(invalid).success).toBe(false);
    });

    it('should validate metric ranges (0-1) for recall and ndcg', () => {
      const invalidMetrics = {
        ...validRun,
        metrics: {
          ...validRun.metrics,
          recall_at_10: 1.5 // > 1
        }
      };
      
      expect(BenchmarkRunSchema.safeParse(invalidMetrics).success).toBe(false);
    });

    it('should validate non-negative latencies', () => {
      const negativeLatency = {
        ...validRun,
        metrics: {
          ...validRun.metrics,
          stage_latencies: {
            ...validRun.metrics.stage_latencies,
            stage_a_p50: -10 // negative
          }
        }
      };
      
      expect(BenchmarkRunSchema.safeParse(negativeLatency).success).toBe(false);
    });

    it('should handle optional stage_c metrics', () => {
      const withoutStageC = {
        ...validRun,
        metrics: {
          ...validRun.metrics,
          stage_latencies: {
            stage_a_p50: 45.2,
            stage_a_p95: 78.1,
            stage_b_p50: 123.4,
            stage_b_p95: 245.6,
            e2e_p50: 235.6,
            e2e_p95: 458.2
            // stage_c metrics omitted
          },
          fan_out_sizes: {
            stage_a: 1000,
            stage_b: 200
            // stage_c omitted
          }
        }
      };
      
      expect(BenchmarkRunSchema.safeParse(withoutStageC).success).toBe(true);
    });
  });

  describe('ABTestResultSchema', () => {
    const validABTest: ABTestResult = {
      metric: 'recall_at_10',
      baseline_mean: 0.75,
      treatment_mean: 0.78,
      delta: 0.03,
      delta_percent: 4.0,
      ci_lower: 0.01,
      ci_upper: 0.05,
      p_value: 0.023,
      is_significant: true,
      sample_size: 1000,
      effect_size: 0.15
    };

    it('should validate a complete A/B test result', () => {
      const result = ABTestResultSchema.safeParse(validABTest);
      expect(result.success).toBe(true);
    });

    it('should require positive sample size', () => {
      const invalidSize = { ...validABTest, sample_size: 0 };
      expect(ABTestResultSchema.safeParse(invalidSize).success).toBe(false);
    });

    it('should handle optional effect_size', () => {
      const { effect_size, ...withoutEffectSize } = validABTest;
      expect(ABTestResultSchema.safeParse(withoutEffectSize).success).toBe(true);
    });
  });

  describe('MetamorphicTestSchema', () => {
    const validMetamorphic: MetamorphicTest = {
      test_id: 'metamorphic-001',
      transform_type: 'rename_symbol',
      repo_snapshot: 'snapshot-sha-123',
      original_query: 'function calculateSum',
      transformed_query: 'function computeSum',
      expected_invariant: 'Results should be identical after symbol rename',
      tolerance: {
        rank_delta_max: 2,
        recall_drop_max: 0.02
      },
      metadata: {
        created_by: 'test-suite',
        version: '1.0'
      }
    };

    it('should validate a complete metamorphic test', () => {
      const result = MetamorphicTestSchema.safeParse(validMetamorphic);
      expect(result.success).toBe(true);
    });

    it('should validate transform_type enum', () => {
      const validTypes = ['rename_symbol', 'move_file', 'reformat', 'inject_decoys', 'plant_canaries'] as const;
      
      validTypes.forEach(type => {
        const valid = { ...validMetamorphic, transform_type: type };
        expect(MetamorphicTestSchema.safeParse(valid).success).toBe(true);
      });
      
      const invalid = { ...validMetamorphic, transform_type: 'invalid-type' };
      expect(MetamorphicTestSchema.safeParse(invalid).success).toBe(false);
    });

    it('should apply default tolerance values', () => {
      const withoutTolerance = {
        test_id: 'test-001',
        transform_type: 'reformat' as const,
        repo_snapshot: 'snap-123',
        original_query: 'test query',
        expected_invariant: 'test invariant'
      };
      
      const result = MetamorphicTestSchema.parse(withoutTolerance);
      expect(result.tolerance.rank_delta_max).toBe(3);
      expect(result.tolerance.recall_drop_max).toBe(0.05);
    });

    it('should handle optional transformed_query and metadata', () => {
      const minimal = {
        test_id: 'test-002',
        transform_type: 'move_file' as const,
        repo_snapshot: 'snap-456',
        original_query: 'test query',
        expected_invariant: 'test invariant'
      };
      
      expect(MetamorphicTestSchema.safeParse(minimal).success).toBe(true);
    });
  });

  describe('RobustnessTestSchema', () => {
    const validRobustness: RobustnessTest = {
      test_type: 'concurrency',
      parameters: {
        concurrent_users: 50,
        duration_seconds: 300
      },
      success_criteria: {
        error_rate_max: 0.01,
        latency_p95_max: 500
      },
      timeout_ms: 45000
    };

    it('should validate a complete robustness test', () => {
      const result = RobustnessTestSchema.safeParse(validRobustness);
      expect(result.success).toBe(true);
    });

    it('should validate test_type enum', () => {
      const validTypes = ['concurrency', 'cold_start', 'incremental_rebuild', 'compaction_under_load', 'fault_injection'] as const;
      
      validTypes.forEach(type => {
        const valid = { ...validRobustness, test_type: type };
        expect(RobustnessTestSchema.safeParse(valid).success).toBe(true);
      });
    });

    it('should require timeout_ms >= 1000', () => {
      const tooShort = { ...validRobustness, timeout_ms: 500 };
      expect(RobustnessTestSchema.safeParse(tooShort).success).toBe(false);
    });

    it('should apply default timeout_ms', () => {
      const withoutTimeout = {
        test_type: 'cold_start' as const,
        parameters: {},
        success_criteria: {}
      };
      
      const result = RobustnessTestSchema.parse(withoutTimeout);
      expect(result.timeout_ms).toBe(30000);
    });
  });
});

describe('Benchmark Types - NATS Message Schemas', () => {
  describe('BenchmarkPlanMessageSchema', () => {
    const validPlanMessage: BenchmarkPlanMessage = {
      trace_id: '123e4567-e89b-12d3-a456-426614174000',
      timestamp: '2024-01-15T10:30:00Z',
      config: {
        trace_id: '123e4567-e89b-12d3-a456-426614174000',
        suite: ['codesearch'],
        systems: ['lex'],
        slices: 'SMOKE_DEFAULT'
      },
      estimated_duration_ms: 300000,
      total_queries: 50
    };

    it('should validate a complete benchmark plan message', () => {
      const result = BenchmarkPlanMessageSchema.safeParse(validPlanMessage);
      expect(result.success).toBe(true);
    });

    it('should validate nested config schema', () => {
      const invalidConfig = {
        ...validPlanMessage,
        config: {
          ...validPlanMessage.config,
          trace_id: 'invalid-uuid'
        }
      };
      
      expect(BenchmarkPlanMessageSchema.safeParse(invalidConfig).success).toBe(false);
    });
  });

  describe('BenchmarkRunMessageSchema', () => {
    const validRunMessage: BenchmarkRunMessage = {
      trace_id: '123e4567-e89b-12d3-a456-426614174000',
      timestamp: '2024-01-15T10:30:00Z',
      status: 'query_completed',
      query_id: 'query-123',
      stage: 'stage_b',
      latency_ms: 145.6,
      candidates_count: 200
    };

    it('should validate a complete benchmark run message', () => {
      const result = BenchmarkRunMessageSchema.safeParse(validRunMessage);
      expect(result.success).toBe(true);
    });

    it('should validate status enum', () => {
      const validStatuses = ['started', 'query_completed', 'stage_completed', 'failed'] as const;
      
      validStatuses.forEach(status => {
        const valid = { ...validRunMessage, status };
        expect(BenchmarkRunMessageSchema.safeParse(valid).success).toBe(true);
      });
    });

    it('should handle optional fields', () => {
      const minimal = {
        trace_id: '123e4567-e89b-12d3-a456-426614174000',
        timestamp: '2024-01-15T10:30:00Z',
        status: 'started' as const
      };
      
      expect(BenchmarkRunMessageSchema.safeParse(minimal).success).toBe(true);
    });
  });

  describe('BenchmarkResultMessageSchema', () => {
    const validResultMessage: BenchmarkResultMessage = {
      trace_id: '123e4567-e89b-12d3-a456-426614174000',
      timestamp: '2024-01-15T10:30:00Z',
      final_metrics: {
        recall_at_10: 0.85,
        recall_at_50: 0.92,
        ndcg_at_10: 0.78,
        mrr: 0.65,
        first_relevant_tokens: 15.5,
        stage_latencies: {
          stage_a_p50: 45.2,
          stage_a_p95: 78.1,
          stage_b_p50: 123.4,
          stage_b_p95: 245.6,
          e2e_p50: 235.6,
          e2e_p95: 458.2
        },
        fan_out_sizes: {
          stage_a: 1000,
          stage_b: 200
        },
        why_attributions: {
          'exact': 45,
          'fuzzy': 23,
          'semantic': 30
        },
        cbu_score: 0.82,
        ece_score: 0.03
      },
      artifacts: {
        metrics_parquet: '/tmp/metrics.parquet',
        errors_ndjson: '/tmp/errors.ndjson',
        traces_ndjson: '/tmp/traces.ndjson',
        report_pdf: '/tmp/report.pdf',
        config_fingerprint_json: '/tmp/config.json'
      },
      duration_ms: 300000,
      promotion_gate_result: {
        passed: true,
        ndcg_delta: 0.03,
        recall_50_maintained: true,
        latency_p95_acceptable: true,
        regressions: []
      }
    };

    it('should validate a complete benchmark result message', () => {
      const result = BenchmarkResultMessageSchema.safeParse(validResultMessage);
      expect(result.success).toBe(true);
    });

    it('should validate promotion gate results', () => {
      const withRegressions = {
        ...validResultMessage,
        promotion_gate_result: {
          passed: false,
          ndcg_delta: -0.02,
          recall_50_maintained: false,
          latency_p95_acceptable: true,
          regressions: ['recall degradation', 'latency spike']
        }
      };
      
      expect(BenchmarkResultMessageSchema.safeParse(withRegressions).success).toBe(true);
    });
  });
});

describe('Benchmark Types - Constants and Interfaces', () => {
  describe('PROMOTION_GATE_CRITERIA', () => {
    it('should have all required criteria defined', () => {
      expect(PROMOTION_GATE_CRITERIA.cbu_improvement_min).toBe(0.05);
      expect(PROMOTION_GATE_CRITERIA.ece_max).toBe(0.05);
      expect(PROMOTION_GATE_CRITERIA.cpu_p95_max_ms).toBe(150);
      expect(PROMOTION_GATE_CRITERIA.kv_reuse_improvement_min).toBe(0.10);
      expect(PROMOTION_GATE_CRITERIA.ndcg_improvement_min).toBe(0.02);
      expect(PROMOTION_GATE_CRITERIA.recall_50_non_degrading).toBe(true);
      expect(PROMOTION_GATE_CRITERIA.latency_p95_max_increase).toBe(0.10);
      expect(PROMOTION_GATE_CRITERIA.significance_level).toBe(0.05);
      expect(PROMOTION_GATE_CRITERIA.max_slice_regression_pp).toBe(2);
      expect(PROMOTION_GATE_CRITERIA.bootstrap_samples).toBe(1000);
      expect(PROMOTION_GATE_CRITERIA.ci_level).toBe(0.95);
    });

    it('should have reasonable threshold values', () => {
      // Core quality gates
      expect(PROMOTION_GATE_CRITERIA.cbu_improvement_min).toBeGreaterThan(0);
      expect(PROMOTION_GATE_CRITERIA.ece_max).toBeLessThanOrEqual(0.1);
      expect(PROMOTION_GATE_CRITERIA.cpu_p95_max_ms).toBeGreaterThan(0);
      
      // Statistical parameters
      expect(PROMOTION_GATE_CRITERIA.significance_level).toBeLessThan(0.1);
      expect(PROMOTION_GATE_CRITERIA.ci_level).toBeGreaterThan(0.9);
      expect(PROMOTION_GATE_CRITERIA.bootstrap_samples).toBeGreaterThanOrEqual(100);
    });
  });

  describe('ConfigFingerprint Interface', () => {
    it('should define all required properties', () => {
      const fingerprint: ConfigFingerprint = {
        bench_schema: 'v1.0',
        seed: 42,
        pool_sha: 'abc123',
        oracle_sha: 'def456',
        cbu_coefficients: {
          gamma: 0.5,
          delta: 0.3,
          beta: 0.2
        },
        code_hash: 'code123',
        config_hash: 'config456',
        snapshot_shas: {
          'repo1': 'sha1',
          'repo2': 'sha2'
        },
        shard_layout: {
          'shard1': { files: 100 }
        },
        timestamp: '2024-01-15T10:30:00Z',
        seed_set: [1, 2, 3, 4, 5],
        contract_hash: 'contract123',
        fixed_layout: true,
        dedup_enabled: false,
        causal_musts: true,
        kv_budget_cap: 1000
      };

      // Verify all properties are defined
      expect(fingerprint.bench_schema).toBeDefined();
      expect(fingerprint.cbu_coefficients.gamma).toBeDefined();
      expect(fingerprint.cbu_coefficients.delta).toBeDefined();
      expect(fingerprint.cbu_coefficients.beta).toBeDefined();
    });
  });

  describe('BenchmarkOrchestrationConfig Interface', () => {
    it('should define orchestration configuration', () => {
      const config: BenchmarkOrchestrationConfig = {
        workingDir: '/t../../benchmarks/src',
        outputDir: '/tmp/output',
        natsUrl: 'nats://localhost:4222',
        repositories: [
          { name: 'repo1', path: '/path/to/repo1' },
          { name: 'repo2', path: '/path/to/repo2' }
        ]
      };

      expect(config.workingDir).toBe('/t../../benchmarks/src');
      expect(config.repositories).toHaveLength(2);
      expect(config.repositories[0].name).toBe('repo1');
    });

    it('should handle optional natsUrl', () => {
      const config: BenchmarkOrchestrationConfig = {
        workingDir: '/t../../benchmarks/src',
        outputDir: '/tmp/output',
        repositories: []
      };

      expect(config.natsUrl).toBeUndefined();
      expect(config.repositories).toEqual([]);
    });
  });

  describe('ReportData Interface', () => {
    it('should define report data structure', () => {
      const reportData: ReportData = {
        title: 'Benchmark Report',
        config: {
          trace_id: '123e4567-e89b-12d3-a456-426614174000',
          suite: ['codesearch'],
          systems: ['lex'],
          slices: 'SMOKE_DEFAULT'
        },
        benchmarkRuns: [],
        abTestResults: [],
        metamorphicResults: [],
        robustnessResults: [],
        configFingerprint: {
          bench_schema: 'v1.0',
          seed: 42,
          pool_sha: 'abc123',
          oracle_sha: 'def456',
          cbu_coefficients: { gamma: 0.5, delta: 0.3, beta: 0.2 },
          code_hash: 'code123',
          config_hash: 'config456',
          snapshot_shas: {},
          shard_layout: {},
          timestamp: '2024-01-15T10:30:00Z',
          seed_set: [1, 2, 3],
          contract_hash: 'contract123',
          fixed_layout: true,
          dedup_enabled: false,
          causal_musts: true,
          kv_budget_cap: 1000
        },
        metadata: {
          generated_at: '2024-01-15T10:30:00Z',
          total_duration_ms: 300000,
          systems_tested: ['lex', '+symbols'],
          queries_executed: 100
        }
      };

      expect(reportData.title).toBe('Benchmark Report');
      expect(reportData.metadata.queries_executed).toBe(100);
      expect(reportData.configFingerprint.seed).toBe(42);
    });
  });
});

describe('Benchmark Types - Type Compatibility', () => {
  it('should ensure BenchmarkConfig type matches schema', () => {
    // This test ensures the TypeScript type inferred from the schema
    // is compatible with our expected usage
    const config: BenchmarkConfig = {
      trace_id: '123e4567-e89b-12d3-a456-426614174000',
      suite: ['codesearch', 'structural', 'docs'],
      systems: ['lex', '+symbols', '+symbols+semantic'],
      slices: ['slice1', 'slice2'],
      seeds: 3
    };

    // Should compile without errors
    expect(config.trace_id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/);
    expect(config.suite).toContain('codesearch');
    expect(config.systems).toContain('lex');
  });

  it('should handle union types correctly', () => {
    const config1: BenchmarkConfig = {
      trace_id: '123e4567-e89b-12d3-a456-426614174000',
      suite: ['codesearch'],
      systems: ['lex'],
      slices: 'SMOKE_DEFAULT' // string literal
    };

    const config2: BenchmarkConfig = {
      trace_id: '123e4567-e89b-12d3-a456-426614174000',
      suite: ['structural'],
      systems: ['lex'],
      slices: ['custom-slice'] // array of strings
    };

    expect(typeof config1.slices).toBe('string');
    expect(Array.isArray(config2.slices)).toBe(true);
  });

  it('should handle optional fields correctly', () => {
    const minimal: BenchmarkConfig = {
      trace_id: '123e4567-e89b-12d3-a456-426614174000',
      suite: ['codesearch'],
      systems: ['lex'],
      slices: 'ALL'
    };

    // Optional fields should be undefined when not provided
    expect(minimal.api_base_url).toBeUndefined();
    expect(minimal.k).toBeUndefined();
    expect(minimal.include_cold_start).toBeUndefined();
  });
});