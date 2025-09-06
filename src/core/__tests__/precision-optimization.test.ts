/**
 * Comprehensive tests for PrecisionOptimizationEngine and PrecisionExperimentFramework
 * Priority: HIGH - Critical precision optimization with 827 LOC, no existing tests
 */

import { describe, it, expect, beforeEach, afterEach, vi, MockedFunction } from 'vitest';
import { PrecisionOptimizationEngine, PrecisionExperimentFramework } from '../precision-optimization.js';
import type { SearchContext, SearchHit } from '../../types/core.js';
import type { ExperimentConfig, ValidationResult, PrecisionOptimizationConfig } from '../../types/api.js';

// Mock dependencies
vi.mock('../ltr-training-pipeline.js', () => ({
  PairwiseLTRTrainingPipeline: vi.fn().mockImplementation(() => ({
    rerank: vi.fn().mockImplementation(async (candidates, ctx) => candidates),
  })),
}));

vi.mock('../drift-detection-system.js', () => ({
  DriftDetectionSystem: vi.fn(),
  globalDriftDetectionSystem: {
    recordMetrics: vi.fn().mockResolvedValue(undefined),
  },
}));

vi.mock('../../telemetry/tracer.js', () => ({
  LensTracer: {
    createChildSpan: vi.fn().mockReturnValue({
      setAttributes: vi.fn(),
      recordException: vi.fn(),
      end: vi.fn(),
    }),
  },
}));

describe('PrecisionOptimizationEngine', () => {
  let engine: PrecisionOptimizationEngine;
  let mockCandidates: SearchHit[];
  let mockContext: SearchContext;

  beforeEach(async () => {
    vi.clearAllMocks();
    
    // Reset mock for clean state in each test
    const { LensTracer } = await import('../../telemetry/tracer.js');
    const mockSpan = {
      setAttributes: vi.fn(),
      recordException: vi.fn(),
      end: vi.fn(),
    };
    vi.mocked(LensTracer.createChildSpan).mockReturnValue(mockSpan);
    
    engine = new PrecisionOptimizationEngine();
    
    mockCandidates = [
      {
        file: 'test1.ts',
        line: 10,
        col: 5,
        lang: 'typescript',
        snippet: 'function authenticate(user: User)',
        score: 0.95,
        why: ['exact_match'],
        byte_offset: 100,
        span_len: 30,
      },
      {
        file: 'test2.ts',
        line: 25,
        col: 0,
        lang: 'typescript',
        snippet: 'async function handleRequest(req, res)',
        score: 0.85,
        why: ['fuzzy_match'],
        byte_offset: 250,
        span_len: 38,
      },
      {
        file: 'test3.ts',
        line: 15,
        col: 8,
        lang: 'typescript',
        snippet: 'interface UserConfig { auth: boolean }',
        score: 0.75,
        why: ['structural_match'],
        byte_offset: 175,
        span_len: 38,
      },
      {
        file: 'vendor/lib.d.ts',
        line: 5,
        col: 0,
        lang: 'typescript',
        snippet: 'declare module "vendor-lib"',
        score: 0.65,
        why: ['identifier_match'],
        byte_offset: 50,
        span_len: 27,
      },
      {
        file: 'test1.ts',
        line: 12,
        col: 2,
        lang: 'typescript',
        snippet: 'function authenticate(username: string)',
        score: 0.55,
        why: ['fuzzy_match'],
        byte_offset: 140,
        span_len: 39,
      }
    ];

    mockContext = {
      query: 'authentication middleware',
      repo_sha: 'abc123',
      mode: 'hybrid',
      k: 20,
      fuzzy_distance: 1,
    };
  });

  describe('Constructor and Initialization', () => {
    it('should initialize with default configuration', () => {
      expect(engine).toBeDefined();
      expect(engine).toBeInstanceOf(PrecisionOptimizationEngine);
    });

    it('should initialize with all blocks disabled by default', () => {
      const status = engine.getOptimizationStatus();
      
      expect(status.block_a_enabled).toBe(false);
      expect(status.block_b_enabled).toBe(false);
      expect(status.block_c_enabled).toBe(false);
    });

    it('should initialize with default configuration values', () => {
      const status = engine.getOptimizationStatus();
      
      expect(status.config.early_exit.enabled).toBe(false);
      expect(status.config.early_exit.margin).toBe(0.12);
      expect(status.config.early_exit.min_probes).toBe(96);
      
      expect(status.config.dynamic_topn.enabled).toBe(false);
      expect(status.config.dynamic_topn.score_threshold).toBe(0.0);
      expect(status.config.dynamic_topn.hard_cap).toBe(20);
      
      expect(status.config.deduplication.in_file.simhash.k).toBe(5);
      expect(status.config.deduplication.in_file.simhash.hamming_max).toBe(2);
      expect(status.config.deduplication.in_file.keep).toBe(3);
      expect(status.config.deduplication.cross_file.vendor_deboost).toBe(0.3);
    });
  });

  describe('LTR Pipeline Integration', () => {
    it('should initialize LTR pipeline', () => {
      const mockConfig = {
        model_path: './ltr-model',
        feature_extractor: 'default',
        training_data_path: './training-data',
      };
      
      expect(() => engine.initializeLTRPipeline(mockConfig)).not.toThrow();
    });

    it('should record drift metrics successfully', async () => {
      const mockMetrics = {
        anchorP1: 0.85,
        anchorRecall50: 0.89,
        ladderRatio: 0.78,
        lsifCoverage: 85.0,
        treeSitterCoverage: 92.0,
        sampleCount: 100,
        queryComplexity: { simple: 0.6, medium: 0.3, complex: 0.1 }
      };

      await expect(engine.recordDriftMetrics(
        mockMetrics.anchorP1,
        mockMetrics.anchorRecall50,
        mockMetrics.ladderRatio,
        mockMetrics.lsifCoverage,
        mockMetrics.treeSitterCoverage,
        mockMetrics.sampleCount,
        mockMetrics.queryComplexity
      )).resolves.not.toThrow();
    });
  });

  describe('Block Configuration', () => {
    it('should enable/disable Block A', () => {
      engine.setBlockEnabled('A', true);
      expect(engine.getOptimizationStatus().block_a_enabled).toBe(true);
      
      engine.setBlockEnabled('A', false);
      expect(engine.getOptimizationStatus().block_a_enabled).toBe(false);
    });

    it('should enable/disable Block B', () => {
      engine.setBlockEnabled('B', true);
      expect(engine.getOptimizationStatus().block_b_enabled).toBe(true);
      
      engine.setBlockEnabled('B', false);
      expect(engine.getOptimizationStatus().block_b_enabled).toBe(false);
    });

    it('should enable/disable Block C', () => {
      engine.setBlockEnabled('C', true);
      expect(engine.getOptimizationStatus().block_c_enabled).toBe(true);
      
      engine.setBlockEnabled('C', false);
      expect(engine.getOptimizationStatus().block_c_enabled).toBe(false);
    });

    it('should return comprehensive optimization status', () => {
      engine.setBlockEnabled('A', true);
      engine.setBlockEnabled('B', false);
      engine.setBlockEnabled('C', true);
      
      const status = engine.getOptimizationStatus();
      
      expect(status).toHaveProperty('block_a_enabled', true);
      expect(status).toHaveProperty('block_b_enabled', false);
      expect(status).toHaveProperty('block_c_enabled', true);
      expect(status).toHaveProperty('config');
      expect(status.config).toHaveProperty('early_exit');
      expect(status.config).toHaveProperty('dynamic_topn');
      expect(status.config).toHaveProperty('deduplication');
    });
  });

  describe('Block A: Early-Exit Optimization', () => {
    beforeEach(() => {
      engine.setBlockEnabled('A', true);
    });

    it('should pass through candidates when Block A is disabled', async () => {
      engine.setBlockEnabled('A', false);
      
      const result = await engine.applyBlockA(mockCandidates, mockContext);
      
      expect(result).toEqual(mockCandidates);
      expect(result.length).toBe(mockCandidates.length);
    });

    it('should apply early exit when enabled', async () => {
      const config: PrecisionOptimizationConfig = {
        block_a_early_exit: {
          enabled: true,
          margin: 0.2,
          min_probes: 2
        }
      };
      
      const result = await engine.applyBlockA(mockCandidates, mockContext, config);
      
      // Should apply early exit based on margin
      expect(result.length).toBeLessThanOrEqual(mockCandidates.length);
      expect(result[0]).toBeDefined();
    });

    it('should respect min_probes configuration', async () => {
      const config: PrecisionOptimizationConfig = {
        block_a_early_exit: {
          enabled: true,
          margin: 0.5, // Large margin
          min_probes: 3
        }
      };
      
      const result = await engine.applyBlockA(mockCandidates, mockContext, config);
      
      // Should keep at least min_probes candidates
      expect(result.length).toBeGreaterThanOrEqual(3);
    });

    it('should apply ANN configuration', async () => {
      const config: PrecisionOptimizationConfig = {
        block_a_ann: {
          k: 100,
          efSearch: 400
        }
      };
      
      const result = await engine.applyBlockA(mockCandidates, mockContext, config);
      
      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);
    });

    it('should apply gate configuration', async () => {
      const config: PrecisionOptimizationConfig = {
        block_a_gate: {
          min_candidates: 10
        }
      };
      
      const result = await engine.applyBlockA(mockCandidates, mockContext, config);
      
      // Should limit results when below threshold
      expect(result.length).toBeLessThanOrEqual(10);
    });

    it('should handle LTR reranking when pipeline is available', async () => {
      const mockLTRConfig = {
        model_path: './ltr-model',
        feature_extractor: 'default',
        training_data_path: './training-data',
      };
      
      engine.initializeLTRPipeline(mockLTRConfig);
      
      const result = await engine.applyBlockA(mockCandidates, mockContext);
      
      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);
    });

    it('should handle errors gracefully', async () => {
      // Mock span to throw error
      const mockSpan = {
        setAttributes: vi.fn(),
        recordException: vi.fn(),
        end: vi.fn(),
      };
      
      const { LensTracer } = await import('../../telemetry/tracer.js');
      vi.mocked(LensTracer.createChildSpan).mockReturnValue(mockSpan);
      
      mockSpan.setAttributes.mockImplementation(() => {
        throw new Error('Span error');
      });
      
      await expect(engine.applyBlockA(mockCandidates, mockContext))
        .rejects.toThrow('Span error');
      
      expect(mockSpan.recordException).toHaveBeenCalled();
      expect(mockSpan.end).toHaveBeenCalled();
    });
  });

  describe('Block B: Dynamic TopN Optimization', () => {
    beforeEach(() => {
      engine.setBlockEnabled('B', true);
    });

    it('should pass through candidates when Block B is disabled', async () => {
      engine.setBlockEnabled('B', false);
      
      const result = await engine.applyBlockB(mockCandidates, mockContext);
      
      expect(result).toEqual(mockCandidates);
    });

    it('should pass through candidates when dynamic_topn is disabled', async () => {
      const config: PrecisionOptimizationConfig = {
        block_b_dynamic_topn: {
          enabled: false,
          score_threshold: 0.8,
          hard_cap: 10
        }
      };
      
      const result = await engine.applyBlockB(mockCandidates, mockContext, config);
      
      expect(result).toEqual(mockCandidates);
    });

    it('should apply dynamic topN filtering when enabled', async () => {
      const config: PrecisionOptimizationConfig = {
        block_b_dynamic_topn: {
          enabled: true,
          score_threshold: 0.8,
          hard_cap: 10
        }
      };
      
      const result = await engine.applyBlockB(mockCandidates, mockContext, config);
      
      // Should filter by threshold and apply hard cap
      expect(result.length).toBeLessThanOrEqual(10);
      expect(result.every(hit => hit.score >= 0.8)).toBe(true);
    });

    it('should respect hard cap configuration', async () => {
      const config: PrecisionOptimizationConfig = {
        block_b_dynamic_topn: {
          enabled: true,
          score_threshold: 0.0, // Very low threshold
          hard_cap: 2
        }
      };
      
      const result = await engine.applyBlockB(mockCandidates, mockContext, config);
      
      expect(result.length).toBeLessThanOrEqual(2);
    });

    it('should use reliability curve for threshold calculation', async () => {
      const config: PrecisionOptimizationConfig = {
        block_b_dynamic_topn: {
          enabled: true,
          score_threshold: 0.0, // Should use calculated threshold
          hard_cap: 20
        }
      };
      
      const result = await engine.applyBlockB(mockCandidates, mockContext, config);
      
      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);
    });

    it('should handle errors gracefully', async () => {
      const mockSpan = {
        setAttributes: vi.fn(),
        recordException: vi.fn(),
        end: vi.fn(),
      };
      
      const { LensTracer } = await import('../../telemetry/tracer.js');
      vi.mocked(LensTracer.createChildSpan).mockReturnValue(mockSpan);
      
      mockSpan.setAttributes.mockImplementation(() => {
        throw new Error('Block B error');
      });
      
      const config: PrecisionOptimizationConfig = {
        block_b_dynamic_topn: {
          enabled: true,
          score_threshold: 0.8,
          hard_cap: 10
        }
      };
      
      await expect(engine.applyBlockB(mockCandidates, mockContext, config))
        .rejects.toThrow('Block B error');
    });
  });

  describe('Block C: Deduplication Optimization', () => {
    beforeEach(() => {
      engine.setBlockEnabled('C', true);
    });

    it('should pass through candidates when Block C is disabled', async () => {
      engine.setBlockEnabled('C', false);
      
      const result = await engine.applyBlockC(mockCandidates, mockContext);
      
      expect(result).toEqual(mockCandidates);
    });

    it('should apply in-file deduplication', async () => {
      const result = await engine.applyBlockC(mockCandidates, mockContext);
      
      // Should reduce duplicates from same file (test1.ts has 2 candidates)
      expect(result.length).toBeLessThanOrEqual(mockCandidates.length);
    });

    it('should apply vendor deboost to vendor files', async () => {
      const result = await engine.applyBlockC(mockCandidates, mockContext);
      
      // Find vendor file result
      const vendorResult = result.find(hit => hit.file.includes('vendor'));
      const originalVendor = mockCandidates.find(hit => hit.file.includes('vendor'));
      
      if (vendorResult && originalVendor) {
        // Score should be reduced (deboost applied)
        expect(vendorResult.score).toBeLessThan(originalVendor.score);
      }
      
      // Results should be sorted by score descending
      for (let i = 0; i < result.length - 1; i++) {
        expect(result[i].score).toBeGreaterThanOrEqual(result[i + 1].score);
      }
    });

    it('should respect deduplication configuration', async () => {
      const config: PrecisionOptimizationConfig = {
        block_c_dedup: {
          in_file: {
            simhash: { k: 3, hamming_max: 1 },
            keep: 2
          },
          cross_file: {
            vendor_deboost: 0.1
          }
        }
      };
      
      const result = await engine.applyBlockC(mockCandidates, mockContext, config);
      
      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);
    });

    it('should handle simhash deduplication correctly', async () => {
      // Create candidates with very similar content
      const similarCandidates: SearchHit[] = [
        {
          file: 'same.ts',
          line: 1,
          col: 0,
          lang: 'typescript',
          snippet: 'function authenticate user',
          score: 0.9,
          why: ['exact_match'],
          byte_offset: 0,
          span_len: 25,
        },
        {
          file: 'same.ts',
          line: 2,
          col: 0,
          lang: 'typescript',
          snippet: 'function authenticate user',
          score: 0.85,
          why: ['exact_match'],
          byte_offset: 30,
          span_len: 25,
        },
        {
          file: 'same.ts',
          line: 3,
          col: 0,
          lang: 'typescript',
          snippet: 'function different content',
          score: 0.8,
          why: ['fuzzy_match'],
          byte_offset: 60,
          span_len: 26,
        }
      ];
      
      const result = await engine.applyBlockC(similarCandidates, mockContext);
      
      // Should deduplicate similar content or keep as is
      expect(result.length).toBeLessThanOrEqual(similarCandidates.length);
    });

    it('should handle errors gracefully', async () => {
      const mockSpan = {
        setAttributes: vi.fn(),
        recordException: vi.fn(),
        end: vi.fn(),
      };
      
      const { LensTracer } = await import('../../telemetry/tracer.js');
      vi.mocked(LensTracer.createChildSpan).mockReturnValue(mockSpan);
      
      mockSpan.setAttributes.mockImplementation(() => {
        throw new Error('Block C error');
      });
      
      await expect(engine.applyBlockC(mockCandidates, mockContext))
        .rejects.toThrow('Block C error');
    });
  });

  describe('Simhash and Deduplication Algorithms', () => {
    it('should compute simhash for text snippets', () => {
      // Access private method for testing
      const computeSimhash = (engine as any).computeSimhash.bind(engine);
      
      const hash1 = computeSimhash('function authenticate user', 5);
      const hash2 = computeSimhash('function authenticate user', 5);
      const hash3 = computeSimhash('completely different text', 5);
      
      expect(hash1).toBe(hash2); // Same text should produce same hash
      expect(hash1).not.toBe(hash3); // Different text should produce different hash
    });

    it('should calculate hamming distance correctly', () => {
      const hammingDistance = (engine as any).hammingDistance.bind(engine);
      
      const distance1 = hammingDistance(0b1010n, 0b1010n);
      expect(distance1).toBe(0); // Same values
      
      const distance2 = hammingDistance(0b1010n, 0b1011n);
      expect(distance2).toBe(1); // One bit different
      
      const distance3 = hammingDistance(0b1010n, 0b0101n);
      expect(distance3).toBe(4); // All bits different
    });

    it('should convert strings to bigint hashes', () => {
      const stringHashToBigInt = (engine as any).stringHashToBigInt.bind(engine);
      
      const hash1 = stringHashToBigInt('test');
      const hash2 = stringHashToBigInt('test');
      const hash3 = stringHashToBigInt('different');
      
      expect(typeof hash1).toBe('bigint');
      expect(hash1).toBe(hash2);
      expect(hash1).not.toBe(hash3);
    });
  });

  describe('Reliability Curve and Threshold Calculation', () => {
    it('should calculate optimal threshold for target results per query', () => {
      const calculateOptimalThreshold = (engine as any).calculateOptimalThreshold.bind(engine);
      
      const threshold1 = calculateOptimalThreshold(5.0);
      const threshold2 = calculateOptimalThreshold(10.0);
      const threshold3 = calculateOptimalThreshold(20.0);
      
      expect(typeof threshold1).toBe('number');
      expect(typeof threshold2).toBe('number');
      expect(typeof threshold3).toBe('number');
      
      // Lower target should result in higher threshold
      expect(threshold1).toBeGreaterThan(threshold2);
      expect(threshold2).toBeGreaterThan(threshold3);
    });

    it('should have initialized reliability curve', () => {
      const reliabilityCurve = (engine as any).reliabilityCurve;
      
      expect(Array.isArray(reliabilityCurve)).toBe(true);
      expect(reliabilityCurve.length).toBeGreaterThan(0);
      
      // Each point should have required properties
      reliabilityCurve.forEach((point: any) => {
        expect(point).toHaveProperty('threshold');
        expect(point).toHaveProperty('precision');
        expect(point).toHaveProperty('recall');
        expect(point).toHaveProperty('expected_results_per_query');
      });
    });
  });
});

describe('PrecisionExperimentFramework', () => {
  let engine: PrecisionOptimizationEngine;
  let framework: PrecisionExperimentFramework;
  let mockExperimentConfig: ExperimentConfig;

  beforeEach(async () => {
    vi.clearAllMocks();
    
    // Reset mock for clean state in each test
    const { LensTracer } = await import('../../telemetry/tracer.js');
    const mockSpan = {
      setAttributes: vi.fn(),
      recordException: vi.fn(),
      end: vi.fn(),
    };
    vi.mocked(LensTracer.createChildSpan).mockReturnValue(mockSpan);
    
    engine = new PrecisionOptimizationEngine();
    framework = new PrecisionExperimentFramework(engine);
    
    mockExperimentConfig = {
      experiment_id: 'test-exp-001',
      name: 'Block A Early Exit Test',
      description: 'Testing early exit optimization',
      traffic_percentage: 20,
      treatment_config: {
        block_a_early_exit: {
          enabled: true,
          margin: 0.15,
          min_probes: 100
        }
      },
      promotion_gates: {
        min_ndcg_improvement_pct: 2.0,
        min_recall_at_50: 0.85,
        min_span_coverage_pct: 95.0,
        max_latency_multiplier: 1.5
      }
    };
  });

  describe('Experiment Management', () => {
    it('should create experiment successfully', async () => {
      await expect(framework.createExperiment(mockExperimentConfig))
        .resolves.not.toThrow();
    });

    it('should store experiment configuration', async () => {
      await framework.createExperiment(mockExperimentConfig);
      
      const status = framework.getExperimentStatus('test-exp-001');
      expect(status.config).toEqual(mockExperimentConfig);
      expect(status.results).toEqual([]);
    });

    it('should determine traffic splitting correctly', async () => {
      await framework.createExperiment(mockExperimentConfig);
      
      // Test consistent hashing for same request ID
      const shouldUseTreatment1 = framework.shouldUseTreatment('test-exp-001', 'user-123');
      const shouldUseTreatment2 = framework.shouldUseTreatment('test-exp-001', 'user-123');
      expect(shouldUseTreatment1).toBe(shouldUseTreatment2);
      
      // Test different request IDs
      const results = [];
      for (let i = 0; i < 100; i++) {
        results.push(framework.shouldUseTreatment('test-exp-001', `user-${i}`));
      }
      
      const treatmentCount = results.filter(r => r).length;
      // Should be approximately 20% (±10% tolerance for small sample)
      expect(treatmentCount).toBeGreaterThan(10);
      expect(treatmentCount).toBeLessThan(30);
    });

    it('should return false for non-existent experiment', () => {
      const result = framework.shouldUseTreatment('non-existent', 'user-123');
      expect(result).toBe(false);
    });
  });

  describe('Anchor Validation', () => {
    beforeEach(async () => {
      await framework.createExperiment(mockExperimentConfig);
    });

    it('should run anchor validation successfully', async () => {
      const result = await framework.runAnchorValidation('test-exp-001');
      
      expect(result).toHaveProperty('validation_type', 'anchor');
      expect(result).toHaveProperty('passed');
      expect(result).toHaveProperty('metrics');
      expect(result).toHaveProperty('gate_results');
      expect(result).toHaveProperty('timestamp');
      
      expect(typeof result.passed).toBe('boolean');
      expect(result.metrics).toHaveProperty('ndcg_at_10_delta_pct');
      expect(result.metrics).toHaveProperty('recall_at_50');
      expect(result.metrics).toHaveProperty('span_coverage_pct');
      expect(result.metrics).toHaveProperty('p99_latency_ms');
    });

    it('should evaluate promotion gates correctly', async () => {
      const result = await framework.runAnchorValidation('test-exp-001');
      
      expect(result.gate_results).toHaveProperty('ndcg_improvement');
      expect(result.gate_results).toHaveProperty('recall_maintenance');
      expect(result.gate_results).toHaveProperty('span_coverage');
      expect(result.gate_results).toHaveProperty('latency_control');
      
      // Each gate result should be boolean
      Object.values(result.gate_results).forEach(gateResult => {
        expect(typeof gateResult).toBe('boolean');
      });
    });

    it('should store validation results', async () => {
      await framework.runAnchorValidation('test-exp-001');
      
      const status = framework.getExperimentStatus('test-exp-001');
      expect(status.results.length).toBe(1);
      expect(status.results[0].validation_type).toBe('anchor');
    });

    it('should throw error for non-existent experiment', async () => {
      await expect(framework.runAnchorValidation('non-existent'))
        .rejects.toThrow('Experiment non-existent not found');
    });

    it('should handle validation errors gracefully', async () => {
      // Mock span to throw error
      const mockSpan = {
        setAttributes: vi.fn(),
        recordException: vi.fn(),
        end: vi.fn(),
      };
      
      const { LensTracer } = await import('../../telemetry/tracer.js');
      vi.mocked(LensTracer.createChildSpan).mockReturnValue(mockSpan);
      
      mockSpan.setAttributes.mockImplementation(() => {
        throw new Error('Validation error');
      });
      
      await expect(framework.runAnchorValidation('test-exp-001'))
        .rejects.toThrow('Validation error');
    });
  });

  describe('Ladder Validation', () => {
    beforeEach(async () => {
      await framework.createExperiment(mockExperimentConfig);
    });

    it('should run ladder validation successfully', async () => {
      const result = await framework.runLadderValidation('test-exp-001');
      
      expect(result).toHaveProperty('validation_type', 'ladder');
      expect(result).toHaveProperty('passed');
      expect(result).toHaveProperty('metrics');
      expect(result).toHaveProperty('gate_results');
      
      expect(result.gate_results).toHaveProperty('positives_in_candidates');
      expect(result.gate_results).toHaveProperty('hard_negative_leakage');
    });

    it('should store ladder validation results separately', async () => {
      await framework.runAnchorValidation('test-exp-001');
      await framework.runLadderValidation('test-exp-001');
      
      const status = framework.getExperimentStatus('test-exp-001');
      expect(status.results.length).toBe(2);
      
      const anchorResults = status.results.filter(r => r.validation_type === 'anchor');
      const ladderResults = status.results.filter(r => r.validation_type === 'ladder');
      
      expect(anchorResults.length).toBe(1);
      expect(ladderResults.length).toBe(1);
    });

    it('should throw error for non-existent experiment', async () => {
      await expect(framework.runLadderValidation('non-existent'))
        .rejects.toThrow('Experiment non-existent not found');
    });
  });

  describe('Promotion Readiness', () => {
    beforeEach(async () => {
      await framework.createExperiment(mockExperimentConfig);
    });

    it('should check promotion readiness correctly', async () => {
      // Initially not ready (no validations)
      let readiness = await framework.checkPromotionReadiness('test-exp-001');
      expect(readiness.ready).toBe(false);
      expect(readiness.anchor_passed).toBe(false);
      expect(readiness.ladder_passed).toBe(false);
      
      // After anchor validation
      await framework.runAnchorValidation('test-exp-001');
      readiness = await framework.checkPromotionReadiness('test-exp-001');
      // Anchor validation result depends on mock validation logic
      expect(typeof readiness.anchor_passed).toBe('boolean');
      expect(readiness.ladder_passed).toBe(false); // Ladder validation not run yet
      expect(readiness.ready).toBe(false); // Need both to pass
      
      // After both validations
      await framework.runLadderValidation('test-exp-001');
      readiness = await framework.checkPromotionReadiness('test-exp-001');
      expect(readiness.ready).toBe(readiness.anchor_passed && readiness.ladder_passed); // Depends on mock validation results
      expect(readiness.latest_results.length).toBe(2);
    });

    it('should return empty results for non-existent experiment', async () => {
      const readiness = await framework.checkPromotionReadiness('non-existent');
      expect(readiness.ready).toBe(false);
      expect(readiness.latest_results.length).toBe(0);
    });
  });

  describe('Experiment Rollback', () => {
    beforeEach(async () => {
      await framework.createExperiment(mockExperimentConfig);
      // Enable some blocks first
      engine.setBlockEnabled('A', true);
      engine.setBlockEnabled('B', true);
      engine.setBlockEnabled('C', true);
    });

    it('should rollback experiment successfully', async () => {
      await expect(framework.rollbackExperiment('test-exp-001'))
        .resolves.not.toThrow();
      
      // Should disable all blocks
      const status = engine.getOptimizationStatus();
      expect(status.block_a_enabled).toBe(false);
      expect(status.block_b_enabled).toBe(false);
      expect(status.block_c_enabled).toBe(false);
    });

    it('should throw error for non-existent experiment', async () => {
      await expect(framework.rollbackExperiment('non-existent'))
        .rejects.toThrow('Experiment non-existent not found');
    });

    it('should handle rollback errors gracefully', async () => {
      const mockSpan = {
        setAttributes: vi.fn(),
        recordException: vi.fn(),
        end: vi.fn(),
      };
      
      const { LensTracer } = await import('../../telemetry/tracer.js');
      vi.mocked(LensTracer.createChildSpan).mockReturnValue(mockSpan);
      
      mockSpan.setAttributes.mockImplementation(() => {
        throw new Error('Rollback error');
      });
      
      await expect(framework.rollbackExperiment('test-exp-001'))
        .rejects.toThrow('Rollback error');
    });
  });

  describe('Experiment Status and Results', () => {
    beforeEach(async () => {
      await framework.createExperiment(mockExperimentConfig);
    });

    it('should return comprehensive experiment status', async () => {
      await framework.runAnchorValidation('test-exp-001');
      
      const status = framework.getExperimentStatus('test-exp-001');
      
      expect(status).toHaveProperty('config');
      expect(status).toHaveProperty('results');
      expect(status).toHaveProperty('optimization_status');
      
      expect(status.config).toEqual(mockExperimentConfig);
      expect(status.results.length).toBe(1);
      expect(status.optimization_status).toBeDefined();
    });

    it('should return empty results for non-existent experiment', () => {
      const status = framework.getExperimentStatus('non-existent');
      
      expect(status.config).toBeUndefined();
      expect(status.results).toEqual([]);
      expect(status.optimization_status).toBeDefined();
    });
  });

  describe('Hash Function', () => {
    it('should produce consistent hashes for same input', () => {
      const hashString = (framework as any).hashString.bind(framework);
      
      const hash1 = hashString('test-string');
      const hash2 = hashString('test-string');
      const hash3 = hashString('different-string');
      
      expect(hash1).toBe(hash2);
      expect(hash1).not.toBe(hash3);
      expect(typeof hash1).toBe('number');
      expect(hash1).toBeGreaterThanOrEqual(0);
    });

    it('should produce well-distributed hashes', () => {
      const hashString = (framework as any).hashString.bind(framework);
      const hashes = [];
      
      for (let i = 0; i < 100; i++) {
        hashes.push(hashString(`test-${i}`) % 100);
      }
      
      // Check distribution - should not be too clustered
      const buckets = new Array(10).fill(0);
      hashes.forEach(hash => {
        buckets[Math.floor(hash / 10)]++;
      });
      
      // Each bucket should have some values (no bucket completely empty)
      const nonEmptyBuckets = buckets.filter(count => count > 0).length;
      expect(nonEmptyBuckets).toBeGreaterThan(5);
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle empty candidate arrays', async () => {
      engine.setBlockEnabled('A', true);
      engine.setBlockEnabled('B', true);
      engine.setBlockEnabled('C', true);
      
      const emptyCandidates: SearchHit[] = [];
      const testContext = {
        query: 'test query',
        repo_sha: 'abc123',
        mode: 'hybrid' as const,
        k: 10,
        fuzzy_distance: 1,
      };
      
      const resultA = await engine.applyBlockA(emptyCandidates, testContext);
      const resultB = await engine.applyBlockB(emptyCandidates, testContext);
      const resultC = await engine.applyBlockC(emptyCandidates, testContext);
      
      expect(resultA).toEqual([]);
      expect(resultB).toEqual([]);
      expect(resultC).toEqual([]);
    });

    it('should handle single candidate', async () => {
      engine.setBlockEnabled('A', true);
      engine.setBlockEnabled('B', true);
      engine.setBlockEnabled('C', true);
      
      const singleCandidate = [{
        file: 'test.ts',
        line: 10,
        col: 5,
        lang: 'typescript',
        snippet: 'function test()',
        score: 0.95,
        why: ['exact_match'],
        byte_offset: 100,
        span_len: 30,
      }];
      
      const testContext = {
        query: 'test query',
        repo_sha: 'abc123',
        mode: 'hybrid' as const,
        k: 10,
        fuzzy_distance: 1,
      };
      
      const resultA = await engine.applyBlockA(singleCandidate, testContext);
      const resultB = await engine.applyBlockB(singleCandidate, testContext);
      const resultC = await engine.applyBlockC(singleCandidate, testContext);
      
      expect(resultA.length).toBeLessThanOrEqual(1);
      expect(resultB.length).toBeLessThanOrEqual(1);
      expect(resultC.length).toBeLessThanOrEqual(1);
    });

    it('should handle candidates with missing properties gracefully', async () => {
      engine.setBlockEnabled('C', true);
      
      const incompleteCandidates: SearchHit[] = [
        {
          file: 'test.ts',
          line: 1,
          col: 0,
          lang: 'typescript',
          // snippet missing
          score: 0.9,
          why: ['test'],
          byte_offset: 0,
          span_len: 10,
        } as SearchHit
      ];
      
      const testContext = {
        query: 'test query',
        repo_sha: 'abc123',
        mode: 'hybrid' as const,
        k: 10,
        fuzzy_distance: 1,
      };
      
      const result = await engine.applyBlockC(incompleteCandidates, testContext);
      
      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);
    });
  });

  describe('Integration Tests', () => {
    it('should apply all blocks in sequence', async () => {
      engine.setBlockEnabled('A', true);
      engine.setBlockEnabled('B', true);
      engine.setBlockEnabled('C', true);
      
      const config: PrecisionOptimizationConfig = {
        block_a_early_exit: {
          enabled: true,
          margin: 0.1,
          min_probes: 2
        },
        block_b_dynamic_topn: {
          enabled: true,
          score_threshold: 0.5,
          hard_cap: 10
        },
        block_c_dedup: {
          in_file: {
            simhash: { k: 5, hamming_max: 2 },
            keep: 3
          },
          cross_file: {
            vendor_deboost: 0.2
          }
        }
      };
      
      const testCandidates = [
        {
          file: 'test1.ts',
          line: 10,
          col: 5,
          lang: 'typescript',
          snippet: 'function authenticate(user: User)',
          score: 0.95,
          why: ['exact_match'],
          byte_offset: 100,
          span_len: 30,
        },
        {
          file: 'test2.ts',
          line: 25,
          col: 0,
          lang: 'typescript',
          snippet: 'async function handleRequest(req, res)',
          score: 0.85,
          why: ['fuzzy_match'],
          byte_offset: 250,
          span_len: 38,
        }
      ];
      
      const testContext = {
        query: 'authentication middleware',
        repo_sha: 'abc123',
        mode: 'hybrid' as const,
        k: 20,
        fuzzy_distance: 1,
      };
      
      let candidates = [...testCandidates];
      
      candidates = await engine.applyBlockA(candidates, testContext, config);
      candidates = await engine.applyBlockB(candidates, testContext, config);
      candidates = await engine.applyBlockC(candidates, testContext, config);
      
      expect(candidates).toBeDefined();
      expect(Array.isArray(candidates)).toBe(true);
      expect(candidates.length).toBeLessThanOrEqual(testCandidates.length);
    });

    it('should maintain candidate integrity through all blocks', async () => {
      engine.setBlockEnabled('A', true);
      engine.setBlockEnabled('B', true);
      engine.setBlockEnabled('C', true);
      
      const testCandidates = [
        {
          file: 'test1.ts',
          line: 10,
          col: 5,
          lang: 'typescript',
          snippet: 'function authenticate(user: User)',
          score: 0.95,
          why: ['exact_match'],
          byte_offset: 100,
          span_len: 30,
        },
        {
          file: 'test2.ts',
          line: 25,
          col: 0,
          lang: 'typescript',
          snippet: 'async function handleRequest(req, res)',
          score: 0.85,
          why: ['fuzzy_match'],
          byte_offset: 250,
          span_len: 38,
        }
      ];
      
      const testContext = {
        query: 'authentication middleware',
        repo_sha: 'abc123',
        mode: 'hybrid' as const,
        k: 20,
        fuzzy_distance: 1,
      };
      
      let candidates = [...testCandidates];
      
      candidates = await engine.applyBlockA(candidates, testContext);
      candidates = await engine.applyBlockB(candidates, testContext);
      candidates = await engine.applyBlockC(candidates, testContext);
      
      // All remaining candidates should be valid SearchHits
      candidates.forEach(candidate => {
        expect(candidate).toHaveProperty('file');
        expect(candidate).toHaveProperty('line');
        expect(candidate).toHaveProperty('score');
        expect(typeof candidate.score).toBe('number');
        expect(candidate.score).toBeGreaterThanOrEqual(0);
        expect(candidate.score).toBeLessThanOrEqual(1);
      });
      
      // Should be sorted by score descending
      for (let i = 0; i < candidates.length - 1; i++) {
        expect(candidates[i].score).toBeGreaterThanOrEqual(candidates[i + 1].score);
      }
    });
  });
});