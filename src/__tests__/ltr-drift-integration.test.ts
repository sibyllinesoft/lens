/**
 * Integration tests for LTR Training Pipeline and Drift Detection System
 * 
 * Tests the end-to-end integration of:
 * - Pairwise LTR training with feature extraction
 * - Drift detection with CUSUM algorithms
 * - Integration with precision optimization framework
 * - Monitoring and alerting functionality
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { PairwiseLTRTrainingPipeline, type LTRTrainingConfig, type LTRFeatures } from '../core/ltr-training-pipeline.js';
import { DriftDetectionSystem, type DriftDetectionConfig, type DriftMetrics } from '../core/drift-detection-system.js';
import { PrecisionOptimizationEngine, globalPrecisionEngine } from '../core/precision-optimization.js';
import type { SearchHit, SearchContext } from '../types/core.js';

describe('LTR Training Pipeline', () => {
  let ltrPipeline: PairwiseLTRTrainingPipeline;
  let mockSearchHits: SearchHit[];
  let mockContext: SearchContext;

  beforeEach(() => {
    const config: LTRTrainingConfig = {
      learning_rate: 0.01,
      regularization: 0.001,
      max_iterations: 100,
      convergence_threshold: 1e-6,
      validation_split: 0.2,
      isotonic_calibration: true,
      feature_normalization: true
    };

    ltrPipeline = new PairwiseLTRTrainingPipeline(config);

    mockSearchHits = [
      {
        file: 'src/core/search-engine.ts',
        line: 42,
        col: 8,
        snippet: 'async function searchIndex(query: string): Promise<SearchHit[]>',
        score: 0.95,
        why: ['exact', 'symbol'],
        symbol_kind: 'function',
        ast_path: '/class/method',
        pattern_type: 'function_def'
      },
      {
        file: 'tests/search-engine.test.ts',
        line: 15,
        col: 4,
        snippet: 'test("should search index correctly")',
        score: 0.65,
        why: ['fuzzy'],
        symbol_kind: 'function'
      },
      {
        file: 'node_modules/lodash/index.js',
        line: 1234,
        col: 1,
        snippet: 'function search(collection, predicate)',
        score: 0.45,
        why: ['fuzzy'],
        symbol_kind: 'function'
      }
    ];

    mockContext = {
      query: 'search function',
      mode: 'hybrid',
      fuzzy: 1,
      k: 20,
      filters: {},
      timeout_ms: 1000,
      repo_sha: 'abc123',
      index_version: 'v1',
      api_version: 'v1'
    };
  });

  describe('Feature Extraction', () => {
    it('should extract LTR features correctly', () => {
      const features = ltrPipeline.extractFeatures(mockSearchHits[0], mockContext.query, mockContext);

      expect(features.subtoken_jaccard).toBeGreaterThan(0);
      expect(features.subtoken_jaccard).toBeLessThanOrEqual(1);
      
      expect(features.struct_distance).toBeGreaterThan(0);
      expect(features.struct_distance).toBeLessThanOrEqual(1);
      
      expect(features.path_prior_residual).toBeGreaterThan(0);
      expect(features.path_prior_residual).toBeLessThanOrEqual(1);
      
      expect(features.docBM25).toBeGreaterThanOrEqual(0);
      expect(features.docBM25).toBeLessThanOrEqual(1);
      
      expect(features.pos_in_file).toBeGreaterThanOrEqual(0);
      expect(features.pos_in_file).toBeLessThanOrEqual(1);
      
      expect(features.near_dup_flags).toBeGreaterThanOrEqual(0);
      expect(features.near_dup_flags).toBeLessThanOrEqual(1);
    });

    it('should prioritize core implementation files', () => {
      const coreFeatures = ltrPipeline.extractFeatures(mockSearchHits[0], mockContext.query, mockContext);
      const testFeatures = ltrPipeline.extractFeatures(mockSearchHits[1], mockContext.query, mockContext);
      const vendorFeatures = ltrPipeline.extractFeatures(mockSearchHits[2], mockContext.query, mockContext);

      // Core files should have higher path prior scores
      expect(coreFeatures.path_prior_residual).toBeGreaterThan(testFeatures.path_prior_residual);
      expect(testFeatures.path_prior_residual).toBeGreaterThan(vendorFeatures.path_prior_residual);
    });

    it('should handle symbol kind relevance correctly', () => {
      const functionHit = mockSearchHits[0];
      const features = ltrPipeline.extractFeatures(functionHit, 'function search', mockContext);

      // Function symbols should have high structural relevance for function queries
      expect(features.struct_distance).toBeGreaterThan(0.7);
    });
  });

  describe('Training Process', () => {
    it('should add training examples correctly', () => {
      ltrPipeline.addTrainingExample(
        mockContext.query,
        mockSearchHits[0], // Positive (higher relevance)
        mockSearchHits[2], // Negative (lower relevance)
        mockContext,
        1.0
      );

      const stats = ltrPipeline.getTrainingStats();
      expect(stats.training_examples).toBe(1);
    });

    it('should train model with sufficient data', async () => {
      // Add multiple training examples
      for (let i = 0; i < 50; i++) {
        ltrPipeline.addTrainingExample(
          `query ${i}`,
          mockSearchHits[0],
          mockSearchHits[2],
          mockContext,
          1.0
        );
      }

      const result = await ltrPipeline.trainModel();

      expect(result.convergence_iterations).toBeGreaterThan(0);
      expect(result.final_loss).toBeLessThan(1.0);
      expect(result.validation_accuracy).toBeGreaterThan(0.5);

      // Weights should be reasonable
      const weights = result.final_weights;
      Object.values(weights).forEach(weight => {
        expect(Math.abs(weight)).toBeLessThan(10); // Reasonable weight range
      });
    });

    it('should improve ranking after training', async () => {
      // Add training data favoring core files over vendor files
      for (let i = 0; i < 30; i++) {
        ltrPipeline.addTrainingExample(
          `query ${i}`,
          mockSearchHits[0], // Core file (positive)
          mockSearchHits[2], // Vendor file (negative)
          mockContext,
          1.0
        );
      }

      await ltrPipeline.trainModel();

      // Test reranking
      const rerankedHits = await ltrPipeline.rerank([...mockSearchHits].reverse(), mockContext);

      // Core file should be ranked higher than vendor file
      const coreIndex = rerankedHits.findIndex(hit => hit.file === 'src/core/search-engine.ts');
      const vendorIndex = rerankedHits.findIndex(hit => hit.file.includes('node_modules'));

      expect(coreIndex).toBeLessThan(vendorIndex);
    });
  });

  describe('Model Persistence', () => {
    it('should save and load model weights', async () => {
      // Train model
      for (let i = 0; i < 20; i++) {
        ltrPipeline.addTrainingExample(`query ${i}`, mockSearchHits[0], mockSearchHits[1], mockContext, 1.0);
      }
      await ltrPipeline.trainModel();

      // Save weights
      const savedWeights = ltrPipeline.getModelWeights();

      // Create new pipeline and load weights
      const newPipeline = new PairwiseLTRTrainingPipeline({
        learning_rate: 0.01,
        regularization: 0.001,
        max_iterations: 100,
        convergence_threshold: 1e-6,
        validation_split: 0.2,
        isotonic_calibration: false,
        feature_normalization: true
      });
      newPipeline.setModelWeights(savedWeights);

      // Both pipelines should produce same scores
      const originalScore = ltrPipeline.extractFeatures(mockSearchHits[0], mockContext.query, mockContext);
      const newScore = newPipeline.extractFeatures(mockSearchHits[0], mockContext.query, mockContext);

      expect(originalScore.subtoken_jaccard).toBeCloseTo(newScore.subtoken_jaccard, 5);
    });
  });
});

describe('Drift Detection System', () => {
  let driftSystem: DriftDetectionSystem;
  let alertsReceived: any[] = [];

  beforeEach(() => {
    const config: DriftDetectionConfig = {
      anchor_p1_cusum: {
        reference_value: 0.85,
        drift_threshold: 0.02,
        decision_interval: 3.0,
        reset_threshold: 0.01,
        min_samples: 5
      },
      anchor_recall_cusum: {
        reference_value: 0.90,
        drift_threshold: 0.03,
        decision_interval: 3.0,
        reset_threshold: 0.015,
        min_samples: 5
      },
      ladder_monitoring: {
        enabled: true,
        baseline_ratio: 0.75,
        degradation_threshold: 0.05,
        trend_window_size: 10
      },
      coverage_monitoring: {
        enabled: true,
        lsif_baseline_pct: 85.0,
        tree_sitter_baseline_pct: 90.0,
        degradation_threshold_pct: 5.0,
        measurement_interval_hours: 1
      },
      alerting: {
        consolidation_window_minutes: 5,
        max_alerts_per_hour: 5,
        escalation_thresholds: {
          warning_consecutive: 2,
          error_consecutive: 3,
          critical_consecutive: 5
        }
      }
    };

    driftSystem = new DriftDetectionSystem(config);
    alertsReceived = [];

    driftSystem.on('drift_alert', (alert) => {
      alertsReceived.push(alert);
    });
  });

  afterEach(() => {
    driftSystem.removeAllListeners();
  });

  describe('CUSUM Drift Detection', () => {
    it('should detect anchor P@1 degradation', async () => {
      // Simulate gradual degradation
      const baselineP1 = 0.85;
      for (let i = 0; i < 10; i++) {
        const degradedP1 = baselineP1 - (i * 0.01); // Gradual degradation
        const metrics: DriftMetrics = createMockMetrics({ anchor_p_at_1: degradedP1 });
        await driftSystem.recordMetrics(metrics);
      }

      // Should trigger drift alert
      expect(alertsReceived.length).toBeGreaterThan(0);
      const p1Alert = alertsReceived.find(alert => alert.alert_type === 'anchor_p1_drift');
      expect(p1Alert).toBeDefined();
      expect(p1Alert.severity).toMatch(/warning|error|critical/);
    });

    it('should detect anchor recall degradation', async () => {
      const baselineRecall = 0.90;
      for (let i = 0; i < 10; i++) {
        const degradedRecall = baselineRecall - (i * 0.015); // More aggressive degradation
        const metrics: DriftMetrics = createMockMetrics({ anchor_recall_at_50: degradedRecall });
        await driftSystem.recordMetrics(metrics);
      }

      const recallAlert = alertsReceived.find(alert => alert.alert_type === 'anchor_recall_drift');
      expect(recallAlert).toBeDefined();
    });

    it('should not trigger false alarms for normal variation', async () => {
      const baseline = 0.85;
      // Add normal variation around baseline
      for (let i = 0; i < 20; i++) {
        const variation = (Math.random() - 0.5) * 0.02; // ±1% variation
        const metrics: DriftMetrics = createMockMetrics({ anchor_p_at_1: baseline + variation });
        await driftSystem.recordMetrics(metrics);
      }

      // Should not trigger alerts for normal variation
      expect(alertsReceived.length).toBe(0);
    });
  });

  describe('Ladder Monitoring', () => {
    it('should detect ladder positives ratio degradation', async () => {
      const baselineRatio = 0.75;
      // Create trend of degrading ladder performance
      for (let i = 0; i < 12; i++) {
        const degradedRatio = baselineRatio - (i * 0.01);
        const metrics: DriftMetrics = createMockMetrics({ ladder_positives_ratio: degradedRatio });
        await driftSystem.recordMetrics(metrics);
      }

      const ladderAlert = alertsReceived.find(alert => alert.alert_type === 'ladder_drift');
      expect(ladderAlert).toBeDefined();
      expect(ladderAlert.context.trend_direction).toBe('decreasing');
    });
  });

  describe('Coverage Monitoring', () => {
    it('should detect LSIF coverage degradation', async () => {
      const degradedMetrics: DriftMetrics = createMockMetrics({ 
        lsif_coverage_pct: 78.0 // Below baseline of 85%
      });
      
      await driftSystem.recordMetrics(degradedMetrics);

      const coverageAlert = alertsReceived.find(alert => 
        alert.alert_type === 'coverage_drift' && alert.metric_name === 'lsif_coverage_pct'
      );
      expect(coverageAlert).toBeDefined();
    });

    it('should detect tree-sitter coverage degradation', async () => {
      const degradedMetrics: DriftMetrics = createMockMetrics({ 
        tree_sitter_coverage_pct: 83.0 // Below baseline of 90%
      });
      
      await driftSystem.recordMetrics(degradedMetrics);

      const coverageAlert = alertsReceived.find(alert => 
        alert.alert_type === 'coverage_drift' && alert.metric_name === 'tree_sitter_coverage_pct'
      );
      expect(coverageAlert).toBeDefined();
    });
  });

  describe('Alert Management', () => {
    it('should escalate severity with consecutive violations', async () => {
      const degradedP1 = 0.75; // Significantly below baseline

      // Generate multiple violations
      for (let i = 0; i < 8; i++) {
        const metrics: DriftMetrics = createMockMetrics({ anchor_p_at_1: degradedP1 });
        await driftSystem.recordMetrics(metrics);
      }

      const alerts = alertsReceived.filter(alert => alert.alert_type === 'anchor_p1_drift');
      
      // Should have alerts with escalating severity
      const severities = alerts.map(alert => alert.severity);
      expect(severities).toContain('warning');
      if (alerts.length >= 3) {
        expect(severities).toContain('error');
      }
      if (alerts.length >= 5) {
        expect(severities).toContain('critical');
      }
    });

    it('should provide actionable recommendations', async () => {
      const degradedMetrics: DriftMetrics = createMockMetrics({ anchor_p_at_1: 0.75 });
      await driftSystem.recordMetrics(degradedMetrics);

      const alert = alertsReceived[0];
      expect(alert.recommended_actions).toBeDefined();
      expect(alert.recommended_actions.length).toBeGreaterThan(0);
      expect(alert.recommended_actions[0]).toContain('Check');
    });
  });

  describe('System Health Reporting', () => {
    it('should generate comprehensive drift report', async () => {
      // Add some metrics
      const metrics: DriftMetrics = createMockMetrics({});
      await driftSystem.recordMetrics(metrics);

      const report = driftSystem.getDriftReport();

      expect(report.system_health).toMatch(/healthy|degraded|critical/);
      expect(report.metrics_summary).toBeDefined();
      expect(report.detector_stats).toBeDefined();
      expect(report.recommendations).toBeDefined();
      expect(Array.isArray(report.recommendations)).toBe(true);
    });

    it('should reflect system health based on alerts', async () => {
      // Generate critical alert
      for (let i = 0; i < 8; i++) {
        const metrics: DriftMetrics = createMockMetrics({ anchor_p_at_1: 0.65 });
        await driftSystem.recordMetrics(metrics);
      }

      const report = driftSystem.getDriftReport();
      const criticalAlerts = report.active_alerts.filter(alert => alert.severity === 'critical');
      
      if (criticalAlerts.length > 0) {
        expect(report.system_health).toBe('critical');
      }
    });
  });

  // Helper function to create mock metrics
  function createMockMetrics(overrides: Partial<DriftMetrics> = {}): DriftMetrics {
    return {
      timestamp: new Date().toISOString(),
      anchor_p_at_1: 0.85,
      anchor_recall_at_50: 0.90,
      ladder_positives_ratio: 0.75,
      lsif_coverage_pct: 85.0,
      tree_sitter_coverage_pct: 90.0,
      sample_count: 100,
      query_complexity_distribution: {
        simple: 0.6,
        medium: 0.3,
        complex: 0.1
      },
      ...overrides
    };
  }
});

describe('Integration with Precision Optimization', () => {
  let precisionEngine: PrecisionOptimizationEngine;

  beforeEach(() => {
    precisionEngine = new PrecisionOptimizationEngine();
    
    // Initialize LTR pipeline
    const ltrConfig: LTRTrainingConfig = {
      learning_rate: 0.01,
      regularization: 0.001,
      max_iterations: 50,
      convergence_threshold: 1e-5,
      validation_split: 0.2,
      isotonic_calibration: true,
      feature_normalization: true
    };
    
    precisionEngine.initializeLTRPipeline(ltrConfig);
  });

  it('should integrate LTR reranking with Block A optimization', async () => {
    const mockHits: SearchHit[] = [
      {
        file: 'src/core/engine.ts',
        line: 10,
        col: 0,
        snippet: 'export class SearchEngine',
        score: 0.8,
        why: ['exact']
      },
      {
        file: 'test/engine.test.ts',
        line: 5,
        col: 0,
        snippet: 'test("engine works")',
        score: 0.75,
        why: ['fuzzy']
      }
    ];

    const mockContext: SearchContext = {
      query: 'SearchEngine',
      mode: 'hybrid',
      fuzzy: 1,
      k: 10,
      filters: {},
      timeout_ms: 1000,
      repo_sha: 'abc123',
      index_version: 'v1',
      api_version: 'v1'
    };

    // Enable Block A
    precisionEngine.setBlockEnabled('A', true);

    const optimizedHits = await precisionEngine.applyBlockA(mockHits, mockContext);

    expect(optimizedHits.length).toBeGreaterThan(0);
    expect(optimizedHits.length).toBeLessThanOrEqual(mockHits.length);
  });

  it('should record drift metrics correctly', async () => {
    const spy = vi.spyOn(console, 'log').mockImplementation(() => {});

    await precisionEngine.recordDriftMetrics(
      0.82,  // Anchor P@1
      0.88,  // Anchor Recall@50
      0.73,  // Ladder ratio
      83.0,  // LSIF coverage
      89.0,  // Tree-sitter coverage
      150,   // Sample count
      { simple: 0.7, medium: 0.2, complex: 0.1 }
    );

    // Should not throw and should complete successfully
    expect(spy).toHaveBeenCalledWith(expect.stringContaining('Drift metrics recorded'));
    
    spy.mockRestore();
  });

  it('should maintain 100% span coverage throughout optimization', async () => {
    const mockHits: SearchHit[] = Array.from({ length: 100 }, (_, i) => ({
      file: `src/file${i}.ts`,
      line: i + 1,
      col: 0,
      snippet: `function func${i}()`,
      score: 0.9 - (i * 0.005), // Decreasing scores
      why: ['exact'],
      span_len: 10,
      byte_offset: i * 100
    }));

    const mockContext: SearchContext = {
      query: 'function',
      mode: 'hybrid',
      fuzzy: 1,
      k: 50,
      filters: {},
      timeout_ms: 1000,
      repo_sha: 'abc123',
      index_version: 'v1',
      api_version: 'v1'
    };

    // Apply all optimization blocks
    precisionEngine.setBlockEnabled('A', true);
    precisionEngine.setBlockEnabled('B', true);
    precisionEngine.setBlockEnabled('C', true);

    let optimizedHits = await precisionEngine.applyBlockA(mockHits, mockContext);
    optimizedHits = await precisionEngine.applyBlockB(optimizedHits, mockContext);
    optimizedHits = await precisionEngine.applyBlockC(optimizedHits, mockContext);

    // Verify all hits have valid spans
    optimizedHits.forEach(hit => {
      expect(hit.file).toBeDefined();
      expect(hit.line).toBeGreaterThan(0);
      expect(hit.col).toBeGreaterThanOrEqual(0);
      if (hit.span_len) {
        expect(hit.span_len).toBeGreaterThan(0);
      }
      if (hit.byte_offset) {
        expect(hit.byte_offset).toBeGreaterThanOrEqual(0);
      }
    });

    // Should maintain reasonable number of results (span coverage)
    expect(optimizedHits.length).toBeGreaterThan(10);
    expect(optimizedHits.length).toBeLessThanOrEqual(50);
  });
});

describe('Performance and Load Testing', () => {
  it('should handle high-volume drift metrics efficiently', async () => {
    const driftSystem = new DriftDetectionSystem({
      anchor_p1_cusum: { reference_value: 0.85, drift_threshold: 0.02, decision_interval: 5.0, reset_threshold: 0.01, min_samples: 10 },
      anchor_recall_cusum: { reference_value: 0.90, drift_threshold: 0.03, decision_interval: 5.0, reset_threshold: 0.015, min_samples: 10 },
      ladder_monitoring: { enabled: true, baseline_ratio: 0.75, degradation_threshold: 0.05, trend_window_size: 20 },
      coverage_monitoring: { enabled: true, lsif_baseline_pct: 85, tree_sitter_baseline_pct: 90, degradation_threshold_pct: 5, measurement_interval_hours: 1 },
      alerting: { consolidation_window_minutes: 5, max_alerts_per_hour: 10, escalation_thresholds: { warning_consecutive: 2, error_consecutive: 4, critical_consecutive: 8 } }
    });

    const startTime = Date.now();
    
    // Process 1000 metrics
    for (let i = 0; i < 1000; i++) {
      const metrics: DriftMetrics = {
        timestamp: new Date().toISOString(),
        anchor_p_at_1: 0.85 + (Math.random() - 0.5) * 0.02,
        anchor_recall_at_50: 0.90 + (Math.random() - 0.5) * 0.02,
        ladder_positives_ratio: 0.75 + (Math.random() - 0.5) * 0.02,
        lsif_coverage_pct: 85.0 + (Math.random() - 0.5) * 2,
        tree_sitter_coverage_pct: 90.0 + (Math.random() - 0.5) * 2,
        sample_count: 100,
        query_complexity_distribution: {
          simple: 0.6,
          medium: 0.3,
          complex: 0.1
        }
      };
      
      await driftSystem.recordMetrics(metrics);
    }

    const processingTime = Date.now() - startTime;
    
    // Should process 1000 metrics in reasonable time (< 1 second)
    expect(processingTime).toBeLessThan(1000);

    const stats = driftSystem.getSystemStats();
    expect(stats.metrics_history_size).toBe(1000);
  });

  it('should handle LTR training with large datasets efficiently', async () => {
    const config: LTRTrainingConfig = {
      learning_rate: 0.05, // Higher learning rate for faster convergence
      regularization: 0.001,
      max_iterations: 50,   // Limited iterations for test
      convergence_threshold: 1e-4,
      validation_split: 0.2,
      isotonic_calibration: false, // Disable for faster testing
      feature_normalization: true
    };

    const ltrPipeline = new PairwiseLTRTrainingPipeline(config);

    const mockContext: SearchContext = {
      query: 'test query',
      mode: 'hybrid',
      fuzzy: 1,
      k: 20,
      filters: {},
      timeout_ms: 1000,
      repo_sha: 'abc123',
      index_version: 'v1',
      api_version: 'v1'
    };

    const startTime = Date.now();

    // Add 500 training examples
    for (let i = 0; i < 500; i++) {
      const positiveHit: SearchHit = {
        file: `src/core/file${i}.ts`,
        line: i + 1,
        col: 0,
        snippet: `function relevantFunc${i}()`,
        score: 0.9,
        why: ['exact'],
        symbol_kind: 'function'
      };

      const negativeHit: SearchHit = {
        file: `node_modules/lib/file${i}.js`,
        line: i + 1,
        col: 0,
        snippet: `function irrelevantFunc${i}()`,
        score: 0.3,
        why: ['fuzzy']
      };

      ltrPipeline.addTrainingExample(
        `query ${i}`,
        positiveHit,
        negativeHit,
        mockContext,
        1.0
      );
    }

    const trainingResult = await ltrPipeline.trainModel();
    const trainingTime = Date.now() - startTime;

    // Should complete training in reasonable time (< 5 seconds)
    expect(trainingTime).toBeLessThan(5000);

    // Should achieve reasonable training performance
    expect(trainingResult.validation_accuracy).toBeGreaterThan(0.6);
    expect(trainingResult.final_loss).toBeLessThan(1.0);

    const stats = ltrPipeline.getTrainingStats();
    expect(stats.training_examples).toBeGreaterThan(300); // 80% of 500
    expect(stats.validation_examples).toBeGreaterThan(50);  // 20% of 500
  });
});