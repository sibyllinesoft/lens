/**
 * Comprehensive test suite for Phase B3 optimizations
 * Tests isotonic calibration, confidence gating, optimized HNSW, and performance validation
 * Target: Maintain ≥85% test coverage and validate performance improvements
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { IsotonicCalibrator, IsotonicCalibratedReranker } from '../../src/core/isotonic-reranker.js';
import { OptimizedHNSWIndex } from '../../src/core/optimized-hnsw.js';
import { EnhancedSemanticRerankEngine } from '../../src/indexer/enhanced-semantic.js';
import { FeatureFlagManager } from '../../src/core/feature-flags.js';
import { SegmentStorage } from '../../src/storage/segments.js';
import type { SearchContext, Candidate } from '../../src/types/core.js';

describe('Phase B3 Optimizations', () => {
  describe('IsotonicCalibrator', () => {
    let calibrator: IsotonicCalibrator;

    beforeEach(() => {
      calibrator = new IsotonicCalibrator({
        enabled: true,
        minCalibrationData: 10,
        confidenceCutoff: 0.12,
        updateFreq: 50
      });
    });

    describe('Calibration Data Management', () => {
      it('should add calibration points', () => {
        calibrator.addCalibrationPoint(0.8, 0.9);
        calibrator.addCalibrationPoint(0.6, 0.7);
        calibrator.addCalibrationPoint(0.4, 0.5);

        const stats = calibrator.getStats();
        expect(stats.calibration_points).toBe(3);
        expect(stats.fitted).toBe(false); // Not enough data yet
      });

      it('should maintain bounded calibration data', () => {
        // Add more than maximum to test bounded behavior
        for (let i = 0; i < 5100; i++) {
          calibrator.addCalibrationPoint(Math.random(), Math.random());
        }

        const stats = calibrator.getStats();
        expect(stats.calibration_points).toBeLessThanOrEqual(3000);
      });

      it('should check for update needs correctly', () => {
        expect(calibrator.needsUpdate()).toBe(true); // No calibration yet

        // Add minimum data and fit
        for (let i = 0; i < 15; i++) {
          calibrator.addCalibrationPoint(i * 0.1, i * 0.1 + 0.05);
        }
        calibrator.fitCalibration();

        expect(calibrator.needsUpdate()).toBe(false);

        // Add more data beyond update frequency
        for (let i = 0; i < 60; i++) {
          calibrator.addCalibrationPoint(Math.random(), Math.random());
        }

        expect(calibrator.needsUpdate()).toBe(true);
      });
    });

    describe('PAVA Algorithm', () => {
      it('should fit isotonic calibration with sufficient data', () => {
        // Add monotonic training data
        const trainingData = [
          [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5],
          [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9],
          [0.9, 0.95], [1.0, 1.0]
        ];

        for (const [pred, actual] of trainingData) {
          calibrator.addCalibrationPoint(pred, actual);
        }

        const success = calibrator.fitCalibration();
        expect(success).toBe(true);

        const stats = calibrator.getStats();
        expect(stats.fitted).toBe(true);
        expect(stats.map_size).toBeGreaterThan(0);
      });

      it('should handle non-monotonic data with PAVA', () => {
        // Add non-monotonic data that PAVA should fix
        const trainingData = [
          [0.1, 0.8], [0.2, 0.2], [0.3, 0.9], [0.4, 0.3],
          [0.5, 0.7], [0.6, 0.4], [0.7, 0.8], [0.8, 0.5],
          [0.9, 0.9], [1.0, 0.6]
        ];

        for (const [pred, actual] of trainingData) {
          calibrator.addCalibrationPoint(pred, actual);
        }

        const success = calibrator.fitCalibration();
        expect(success).toBe(true);

        // Test that calibrated scores are monotonic
        const scores = [0.1, 0.3, 0.5, 0.7, 0.9];
        const calibratedScores = scores.map(s => calibrator.calibrateScore(s));

        for (let i = 1; i < calibratedScores.length; i++) {
          expect(calibratedScores[i]).toBeGreaterThanOrEqual(calibratedScores[i-1]);
        }
      });

      it('should interpolate between calibration points', () => {
        // Add known calibration points
        calibrator.addCalibrationPoint(0.0, 0.0);
        calibrator.addCalibrationPoint(0.5, 0.6);
        calibrator.addCalibrationPoint(1.0, 1.0);

        // Add more points to reach minimum
        for (let i = 0; i < 10; i++) {
          calibrator.addCalibrationPoint(Math.random(), Math.random());
        }

        calibrator.fitCalibration();

        // Test interpolation
        const calibratedMid = calibrator.calibrateScore(0.25);
        expect(calibratedMid).toBeGreaterThan(0.0);
        expect(calibratedMid).toBeLessThan(0.6);
      });

      it('should handle edge cases in calibration', () => {
        // Add minimum data
        for (let i = 0; i < 15; i++) {
          calibrator.addCalibrationPoint(i * 0.1, i * 0.1);
        }

        calibrator.fitCalibration();

        // Test edge cases
        expect(calibrator.calibrateScore(-0.1)).toBeGreaterThanOrEqual(0);
        expect(calibrator.calibrateScore(1.1)).toBeLessThanOrEqual(1);
        expect(calibrator.calibrateScore(0.5)).toBeGreaterThan(0);
      });
    });

    describe('Score Calibration', () => {
      beforeEach(() => {
        // Setup calibrated model
        const trainingData = [
          [0.1, 0.15], [0.2, 0.25], [0.3, 0.35], [0.4, 0.45],
          [0.5, 0.55], [0.6, 0.65], [0.7, 0.75], [0.8, 0.85],
          [0.9, 0.92], [1.0, 0.98]
        ];

        for (const [pred, actual] of trainingData) {
          calibrator.addCalibrationPoint(pred, actual);
        }

        calibrator.fitCalibration();
      });

      it('should return original score when not fitted', () => {
        const unfittedCalibrator = new IsotonicCalibrator({
          enabled: true,
          minCalibrationData: 10,
          confidenceCutoff: 0.12,
          updateFreq: 50
        });

        expect(unfittedCalibrator.calibrateScore(0.5)).toBe(0.5);
      });

      it('should calibrate scores correctly', () => {
        const testScores = [0.1, 0.3, 0.5, 0.7, 0.9];
        
        for (const score of testScores) {
          const calibrated = calibrator.calibrateScore(score);
          expect(calibrated).toBeGreaterThan(0);
          expect(calibrated).toBeLessThanOrEqual(1);
          
          // Should generally increase the score for this training data
          expect(calibrated).toBeGreaterThanOrEqual(score * 0.95);
        }
      });
    });
  });

  describe('IsotonicCalibratedReranker', () => {
    let reranker: IsotonicCalibratedReranker;

    beforeEach(() => {
      reranker = new IsotonicCalibratedReranker({
        enabled: true,
        minCalibrationData: 10,
        confidenceCutoff: 0.12,
        maxLatencyMs: 8,
        calibrationUpdateFreq: 20
      });
    });

    afterEach(() => {
      // Clean up any timers or async operations
    });

    describe('Configuration and Initialization', () => {
      it('should initialize with correct default config', () => {
        const stats = reranker.getStats();
        expect(stats.config.enabled).toBe(true);
        expect(stats.config.maxLatencyMs).toBe(8);
        expect(stats.config.confidenceCutoff).toBe(0.12);
      });

      it('should allow config updates', () => {
        reranker.updateConfig({
          confidenceCutoff: 0.2,
          maxLatencyMs: 10
        });

        const stats = reranker.getStats();
        expect(stats.config.confidenceCutoff).toBe(0.2);
        expect(stats.config.maxLatencyMs).toBe(10);
      });
    });

    describe('Reranking Logic', () => {
      let mockSearchHits: any[];
      let mockContext: SearchContext;

      beforeEach(() => {
        mockSearchHits = [
          {
            doc_id: 'doc1',
            file: '/test1.js',
            line: 1,
            col: 1,
            score: 0.8,
            snippet: 'function test() { return 1; }',
            why: 'exact',
            symbol_kind: 'function'
          },
          {
            doc_id: 'doc2',
            file: '/test2.js',
            line: 5,
            col: 10,
            score: 0.6,
            snippet: 'class TestClass { method() {} }',
            why: 'symbol',
            symbol_kind: 'class'
          },
          {
            doc_id: 'doc3',
            file: '/test3.js',
            line: 3,
            col: 5,
            score: 0.4,
            snippet: 'const value = 42;',
            why: 'fuzzy',
            symbol_kind: 'variable'
          }
        ];

        mockContext = {
          trace_id: 'test-rerank',
          query: 'test function',
          mode: 'hybrid',
          k: 10,
          fuzzy_distance: 0,
          started_at: new Date(),
          stages: []
        } as SearchContext;
      });

      it('should rerank hits successfully', async () => {
        const reranked = await reranker.rerank(mockSearchHits, mockContext);
        
        expect(reranked).toBeDefined();
        expect(reranked.length).toBe(mockSearchHits.length);
        
        // All hits should have scores
        reranked.forEach(hit => {
          expect(hit.score).toBeGreaterThan(0);
          expect(hit.score).toBeLessThanOrEqual(1);
        });
      });

      it('should respect latency budget', async () => {
        const fastReranker = new IsotonicCalibratedReranker({
          enabled: true,
          maxLatencyMs: 1 // Very tight budget
        });

        const startTime = Date.now();
        const reranked = await fastReranker.rerank(mockSearchHits, mockContext);
        const latency = Date.now() - startTime;

        // Should complete quickly or fallback gracefully
        expect(reranked).toBeDefined();
        expect(latency).toBeLessThan(20); // Allow some overhead
      });

      it('should apply confidence gating', async () => {
        const gatingSensitiveReranker = new IsotonicCalibratedReranker({
          enabled: true,
          confidenceCutoff: 0.8 // High threshold
        });

        // Test with low-confidence query
        const lowConfidenceContext = {
          ...mockContext,
          query: 'x' // Very short, low-confidence query
        };

        const reranked = await gatingSensitiveReranker.rerank(mockSearchHits, lowConfidenceContext);
        
        // Should return original results due to low confidence
        expect(reranked).toBeDefined();
        expect(reranked.length).toBe(mockSearchHits.length);
      });

      it('should handle empty input gracefully', async () => {
        const reranked = await reranker.rerank([], mockContext);
        expect(reranked).toEqual([]);
      });

      it('should handle errors gracefully', async () => {
        // Create context that might cause issues
        const problematicContext = {
          ...mockContext,
          query: null as any // Invalid query
        };

        const reranked = await reranker.rerank(mockSearchHits, problematicContext);
        
        // Should fallback to original hits
        expect(reranked).toBeDefined();
        expect(reranked.length).toBe(mockSearchHits.length);
      });
    });

    describe('Performance Tracking', () => {
      it('should track and provide statistics', () => {
        const stats = reranker.getStats();
        
        expect(stats).toHaveProperty('config');
        expect(stats).toHaveProperty('base_reranker');
        expect(stats).toHaveProperty('calibrator');
        
        expect(stats.config).toHaveProperty('enabled');
        expect(stats.config).toHaveProperty('maxLatencyMs');
      });

      it('should record calibration examples', () => {
        const mockHit = mockSearchHits[0];
        const predictedScore = 0.7;
        const actualRelevance = 0.8;

        reranker.recordCalibrationExample(mockHit, predictedScore, actualRelevance);

        const stats = reranker.getStats();
        expect(stats.calibrator.calibration_points).toBeGreaterThan(0);
      });
    });
  });

  describe('OptimizedHNSWIndex', () => {
    let hnswIndex: OptimizedHNSWIndex;

    beforeEach(() => {
      hnswIndex = new OptimizedHNSWIndex({
        K: 150, // Fixed per requirements
        efSearch: 64,
        efConstruction: 128,
        qualityThreshold: 0.005,
        performanceTarget: 0.4
      });
    });

    describe('Index Construction', () => {
      it('should enforce K=150 requirement', () => {
        const indexWithWrongK = new OptimizedHNSWIndex({
          K: 100 // Should be corrected to 150
        });

        const stats = indexWithWrongK.getStats();
        expect(stats.config.K).toBe(150);
      });

      it('should build index from vectors', async () => {
        const vectors = new Map<string, Float32Array>();
        
        // Add test vectors
        for (let i = 0; i < 20; i++) {
          const vector = new Float32Array(128);
          for (let j = 0; j < 128; j++) {
            vector[j] = Math.random();
          }
          vectors.set(`doc${i}`, vector);
        }

        await hnswIndex.buildIndex(vectors);

        const stats = hnswIndex.getStats();
        expect(stats.index.nodes).toBe(20);
        expect(stats.index.layers).toBeGreaterThan(0);
      });

      it('should handle empty vector set', async () => {
        const emptyVectors = new Map<string, Float32Array>();
        
        await hnswIndex.buildIndex(emptyVectors);

        const stats = hnswIndex.getStats();
        expect(stats.index.nodes).toBe(0);
        expect(stats.index.layers).toBe(0);
      });

      it('should track build progress with callback', async () => {
        const vectors = new Map<string, Float32Array>();
        
        for (let i = 0; i < 10; i++) {
          const vector = new Float32Array(64);
          vector.fill(i * 0.1);
          vectors.set(`doc${i}`, vector);
        }

        let progressCalls = 0;
        const progressCallback = (progress: number) => {
          expect(progress).toBeGreaterThanOrEqual(0);
          expect(progress).toBeLessThanOrEqual(1);
          progressCalls++;
        };

        await hnswIndex.buildIndex(vectors, progressCallback);
        expect(progressCalls).toBeGreaterThan(0);
      });
    });

    describe('Search Operations', () => {
      beforeEach(async () => {
        // Build a test index
        const vectors = new Map<string, Float32Array>();
        
        for (let i = 0; i < 50; i++) {
          const vector = new Float32Array(64);
          for (let j = 0; j < 64; j++) {
            vector[j] = Math.random() + i * 0.01; // Slight bias per document
          }
          vectors.set(`doc${i}`, vector);
        }

        await hnswIndex.buildIndex(vectors);
      });

      it('should search and return results', async () => {
        const queryVector = new Float32Array(64);
        queryVector.fill(0.5);

        const results = await hnswIndex.search(queryVector, 10);

        expect(results).toBeDefined();
        expect(results.length).toBeLessThanOrEqual(10);
        
        results.forEach(result => {
          expect(result).toHaveProperty('doc_id');
          expect(result).toHaveProperty('score');
          expect(result).toHaveProperty('distance');
          expect(result.score).toBeGreaterThanOrEqual(0);
          expect(result.score).toBeLessThanOrEqual(1);
        });

        // Results should be sorted by distance (ascending)
        for (let i = 1; i < results.length; i++) {
          expect(results[i].distance).toBeGreaterThanOrEqual(results[i-1].distance);
        }
      });

      it('should handle different k values', async () => {
        const queryVector = new Float32Array(64);
        queryVector.fill(0.3);

        const results5 = await hnswIndex.search(queryVector, 5);
        const results15 = await hnswIndex.search(queryVector, 15);

        expect(results5.length).toBeLessThanOrEqual(5);
        expect(results15.length).toBeLessThanOrEqual(15);
        expect(results15.length).toBeGreaterThanOrEqual(results5.length);
      });

      it('should handle custom efSearch parameter', async () => {
        const queryVector = new Float32Array(64);
        queryVector.fill(0.7);

        const defaultResults = await hnswIndex.search(queryVector, 10);
        const customResults = await hnswIndex.search(queryVector, 10, 32);

        expect(defaultResults.length).toBeLessThanOrEqual(10);
        expect(customResults.length).toBeLessThanOrEqual(10);
      });

      it('should handle edge case searches', async () => {
        const zeroVector = new Float32Array(64); // All zeros
        const results = await hnswIndex.search(zeroVector, 5);

        expect(results).toBeDefined();
        expect(results.length).toBeLessThanOrEqual(5);
      });
    });

    describe('Performance Tuning', () => {
      beforeEach(async () => {
        // Build index for tuning tests
        const vectors = new Map<string, Float32Array>();
        
        for (let i = 0; i < 30; i++) {
          const vector = new Float32Array(32);
          for (let j = 0; j < 32; j++) {
            vector[j] = Math.sin(i * 0.1 + j * 0.02); // Deterministic but varied
          }
          vectors.set(`test_doc_${i}`, vector);
        }

        await hnswIndex.buildIndex(vectors);
      });

      it('should tune efSearch parameter', async () => {
        const testQueries: Float32Array[] = [];
        
        // Generate test queries
        for (let i = 0; i < 5; i++) {
          const query = new Float32Array(32);
          for (let j = 0; j < 32; j++) {
            query[j] = Math.cos(i * 0.15 + j * 0.03);
          }
          testQueries.push(query);
        }

        const originalEfSearch = hnswIndex.getStats().config.efSearch;
        const tunedEfSearch = await hnswIndex.tuneEfSearch(testQueries, [], 10);

        expect(tunedEfSearch).toBeGreaterThan(0);
        expect(tunedEfSearch).toBeLessThanOrEqual(256);
        
        const newStats = hnswIndex.getStats();
        expect(newStats.config.efSearch).toBe(tunedEfSearch);
      });

      it('should provide performance statistics', () => {
        const stats = hnswIndex.getStats();

        expect(stats).toHaveProperty('config');
        expect(stats).toHaveProperty('index');
        expect(stats).toHaveProperty('performance');

        expect(stats.config.K).toBe(150);
        expect(stats.index.nodes).toBeGreaterThan(0);
      });

      it('should update configuration correctly', () => {
        const originalEfSearch = hnswIndex.getStats().config.efSearch;
        
        hnswIndex.updateConfig({
          efSearch: 128,
          qualityThreshold: 0.01
        });

        const stats = hnswIndex.getStats();
        expect(stats.config.efSearch).toBe(128);
        expect(stats.config.qualityThreshold).toBe(0.01);
        expect(stats.config.K).toBe(150); // Should remain fixed
      });
    });
  });

  describe('FeatureFlagManager', () => {
    let flagManager: FeatureFlagManager;

    beforeEach(() => {
      flagManager = new FeatureFlagManager({
        stageCOptimizations: false,
        isotonicCalibration: false,
        performanceMonitoring: true,
        abTesting: {
          enabled: false,
          experimentId: 'test-experiment',
          trafficPercentage: 20,
          controlGroup: 'control',
          treatmentGroup: 'treatment'
        }
      });
    });

    describe('Basic Flag Operations', () => {
      it('should check flag status correctly', () => {
        expect(flagManager.isEnabled('stageCOptimizations')).toBe(false);
        expect(flagManager.isEnabled('performanceMonitoring')).toBe(true);
      });

      it('should set and respect overrides', () => {
        expect(flagManager.isEnabled('stageCOptimizations')).toBe(false);

        flagManager.setOverride('stageCOptimizations', true, 'Testing override');
        expect(flagManager.isEnabled('stageCOptimizations')).toBe(true);

        flagManager.removeOverride('stageCOptimizations');
        expect(flagManager.isEnabled('stageCOptimizations')).toBe(false);
      });

      it('should handle expired overrides', () => {
        const pastDate = new Date(Date.now() - 1000);
        
        flagManager.setOverride('stageCOptimizations', true, 'Expired test', {
          expiresAt: pastDate
        });

        // Should return default value since override is expired
        expect(flagManager.isEnabled('stageCOptimizations')).toBe(false);
      });

      it('should respect emergency disable', () => {
        // Set some flags to true
        flagManager.setOverride('stageCOptimizations', true, 'Test');
        flagManager.setOverride('isotonicCalibration', true, 'Test');

        flagManager.emergencyDisableAll('Test emergency');

        // All experimental flags should be disabled
        expect(flagManager.isEnabled('stageCOptimizations')).toBe(false);
        expect(flagManager.isEnabled('isotonicCalibration')).toBe(false);
      });
    });

    describe('A/B Testing', () => {
      beforeEach(() => {
        flagManager = new FeatureFlagManager({
          stageCOptimizations: false,
          abTesting: {
            enabled: true,
            experimentId: 'test-ab',
            trafficPercentage: 50,
            controlGroup: 'control',
            treatmentGroup: 'treatment'
          }
        });
      });

      it('should assign users to A/B groups consistently', () => {
        const user1 = 'user_123';
        const user2 = 'user_456';

        // Same user should get same result
        const result1a = flagManager.isEnabled('stageCOptimizations', { userId: user1 });
        const result1b = flagManager.isEnabled('stageCOptimizations', { userId: user1 });
        expect(result1a).toBe(result1b);

        // Different users may get different results
        const result2 = flagManager.isEnabled('stageCOptimizations', { userId: user2 });
        // This might be the same or different, but we can't assert which
        expect(typeof result2).toBe('boolean');
      });

      it('should respect traffic percentage', () => {
        // Test with 0% traffic
        flagManager.updateConfig({
          abTesting: {
            enabled: true,
            experimentId: 'zero-traffic',
            trafficPercentage: 0,
            controlGroup: 'control',
            treatmentGroup: 'treatment'
          }
        });

        // No users should be in experiment
        for (let i = 0; i < 10; i++) {
          const result = flagManager.isEnabled('stageCOptimizations', { userId: `user_${i}` });
          expect(result).toBe(false); // Default value
        }
      });
    });

    describe('Performance Monitoring and Rollback', () => {
      beforeEach(() => {
        flagManager.setOverride('stageCOptimizations', true, 'Performance test');
      });

      it('should record performance metrics', () => {
        flagManager.recordPerformanceMetrics('stageCOptimizations', {
          latencyMs: 5.0,
          errorRate: 0.01,
          qualityScore: 0.98
        });

        const status = flagManager.getStatus();
        const metrics = status.metrics.stageCOptimizations;
        
        expect(metrics).toBeDefined();
        expect(metrics.performanceImpact.avgLatencyMs).toBeGreaterThan(0);
        expect(metrics.performanceImpact.errorRate).toBeGreaterThanOrEqual(0);
        expect(metrics.performanceImpact.qualityScore).toBeLessThanOrEqual(1);
      });

      it('should trigger automatic rollback on performance degradation', () => {
        // Record good metrics first
        flagManager.recordPerformanceMetrics('stageCOptimizations', {
          latencyMs: 5.0,
          errorRate: 0.01,
          qualityScore: 0.98
        });

        expect(flagManager.isEnabled('stageCOptimizations')).toBe(true);

        // Record bad metrics that should trigger rollback
        flagManager.recordPerformanceMetrics('stageCOptimizations', {
          latencyMs: 20.0, // Over threshold
          errorRate: 0.01,
          qualityScore: 0.98
        });

        // Flag should be disabled due to rollback
        expect(flagManager.isEnabled('stageCOptimizations')).toBe(false);
      });

      it('should trigger rollback on quality degradation', () => {
        flagManager.setOverride('stageCOptimizations', true, 'Quality test');

        // Record metrics with poor quality
        flagManager.recordPerformanceMetrics('stageCOptimizations', {
          latencyMs: 5.0,
          errorRate: 0.01,
          qualityScore: 0.85 // Below threshold
        });

        expect(flagManager.isEnabled('stageCOptimizations')).toBe(false);
      });

      it('should trigger rollback on high error rate', () => {
        flagManager.setOverride('stageCOptimizations', true, 'Error rate test');

        // Record metrics with high error rate
        flagManager.recordPerformanceMetrics('stageCOptimizations', {
          latencyMs: 5.0,
          errorRate: 0.10, // Above threshold
          qualityScore: 0.98
        });

        expect(flagManager.isEnabled('stageCOptimizations')).toBe(false);
      });
    });

    describe('Safety Checks', () => {
      it('should apply safety checks for B3 flags', () => {
        // Mock a flag with rollback history
        flagManager.recordPerformanceMetrics('stageCOptimizations', {
          latencyMs: 15.0, // Trigger rollback
          errorRate: 0.01,
          qualityScore: 0.98
        });

        // Set flag to true, but safety checks should prevent it
        flagManager.updateConfig({
          stageCOptimizations: true
        });

        // Should still be disabled due to safety checks
        const enabled = flagManager.isEnabled('stageCOptimizations');
        expect(enabled).toBe(false);
      });

      it('should provide comprehensive status', () => {
        const status = flagManager.getStatus();

        expect(status).toHaveProperty('config');
        expect(status).toHaveProperty('overrides');
        expect(status).toHaveProperty('metrics');
        expect(status).toHaveProperty('rollbackHistory');

        expect(status.config).toHaveProperty('stageCOptimizations');
        expect(status.config).toHaveProperty('rollbackThresholds');
      });
    });

    describe('Configuration Updates', () => {
      it('should update configuration correctly', () => {
        flagManager.updateConfig({
          stageCOptimizations: true,
          rollbackThresholds: {
            maxLatencyMs: 15,
            minQualityScore: 0.90,
            maxErrorRate: 0.03
          }
        });

        const status = flagManager.getStatus();
        expect(status.config.stageCOptimizations).toBe(true);
        expect(status.config.rollbackThresholds.maxLatencyMs).toBe(15);
      });

      it('should clear emergency disable state', () => {
        flagManager.emergencyDisableAll('Test emergency');
        expect(flagManager.isEnabled('stageCOptimizations')).toBe(false);

        flagManager.clearEmergencyDisable();
        // Emergency flag should be cleared (though overrides may still exist)
        const status = flagManager.getStatus();
        expect(status.config.emergencyDisable).toBe(false);
      });
    });
  });

  describe('Integration Tests', () => {
    let enhancedEngine: EnhancedSemanticRerankEngine;
    let segmentStorage: SegmentStorage;

    beforeEach(async () => {
      segmentStorage = new SegmentStorage('./test-segments-enhanced');
      enhancedEngine = new EnhancedSemanticRerankEngine(segmentStorage, {
        enableIsotonicCalibration: true,
        enableConfidenceGating: true,
        enableOptimizedHNSW: true,
        maxLatencyMs: 10,
        featureFlags: {
          stageCOptimizations: true,
          advancedCalibration: true,
          experimentalHNSW: false
        }
      });
      await enhancedEngine.initialize();
    });

    afterEach(async () => {
      await enhancedEngine.shutdown();
      await segmentStorage.shutdown();
    });

    describe('End-to-End B3 Optimization Flow', () => {
      let mockCandidates: Candidate[];
      let mockContext: SearchContext;

      beforeEach(async () => {
        // Index some test documents
        await enhancedEngine.indexDocument('test1', 'function calculateSum(a, b) { return a + b; }', '/math.js');
        await enhancedEngine.indexDocument('test2', 'class Calculator { multiply(x, y) { return x * y; } }', '/calc.js');
        await enhancedEngine.indexDocument('test3', 'const utils = { divide: (a, b) => a / b };', '/utils.js');

        mockCandidates = [
          {
            doc_id: 'test1:1:1',
            file_path: '/math.js',
            line: 1,
            col: 1,
            score: 0.8,
            match_reasons: ['exact'],
            context: 'function calculateSum(a, b) { return a + b; }'
          },
          {
            doc_id: 'test2:1:1',
            file_path: '/calc.js',
            line: 1,
            col: 1,
            score: 0.6,
            match_reasons: ['symbol'],
            context: 'class Calculator { multiply(x, y) { return x * y; } }'
          },
          {
            doc_id: 'test3:1:1',
            file_path: '/utils.js',
            line: 1,
            col: 1,
            score: 0.4,
            match_reasons: ['fuzzy'],
            context: 'const utils = { divide: (a, b) => a / b };'
          }
        ];

        mockContext = {
          trace_id: 'integration-test',
          query: 'calculate sum addition',
          mode: 'hybrid',
          k: 10,
          fuzzy_distance: 0,
          started_at: new Date(),
          stages: []
        } as SearchContext;
      });

      it('should apply complete B3 optimization pipeline', async () => {
        const startTime = Date.now();
        const reranked = await enhancedEngine.rerankCandidates(mockCandidates, mockContext, 10);
        const latency = Date.now() - startTime;

        // Performance validation
        expect(latency).toBeLessThan(15); // Should be well under Stage-C budget
        
        // Quality validation
        expect(reranked).toBeDefined();
        expect(reranked.length).toBe(mockCandidates.length);
        
        reranked.forEach(candidate => {
          expect(candidate.score).toBeGreaterThan(0);
          expect(candidate.score).toBeLessThanOrEqual(1);
          expect(candidate).toHaveProperty('file_path');
          expect(candidate).toHaveProperty('match_reasons');
        });

        // The most relevant document (math/sum) should rank highly
        const mathCandidate = reranked.find(c => c.context?.includes('calculateSum'));
        expect(mathCandidate).toBeDefined();
        expect(mathCandidate!.score).toBeGreaterThan(0.7);
      });

      it('should handle feature flag control', async () => {
        // Test with optimizations disabled
        await enhancedEngine.updateConfig({
          featureFlags: {
            stageCOptimizations: false,
            advancedCalibration: false,
            experimentalHNSW: false
          }
        });

        const baselineResults = await enhancedEngine.rerankCandidates(mockCandidates, mockContext, 10);

        // Test with optimizations enabled
        await enhancedEngine.updateConfig({
          featureFlags: {
            stageCOptimizations: true,
            advancedCalibration: true,
            experimentalHNSW: false
          }
        });

        const optimizedResults = await enhancedEngine.rerankCandidates(mockCandidates, mockContext, 10);

        // Both should work, but may have different scores
        expect(baselineResults.length).toBe(optimizedResults.length);
        expect(baselineResults).toBeDefined();
        expect(optimizedResults).toBeDefined();
      });

      it('should maintain quality under performance pressure', async () => {
        // Set aggressive performance constraints
        await enhancedEngine.updateConfig({
          maxLatencyMs: 5 // Very tight constraint
        });

        const results = await enhancedEngine.rerankCandidates(mockCandidates, mockContext, 10);

        // Should still return valid results, even with tight constraints
        expect(results).toBeDefined();
        expect(results.length).toBeGreaterThan(0);
        
        results.forEach(candidate => {
          expect(candidate.score).toBeGreaterThan(0);
        });
      });

      it('should provide comprehensive statistics', () => {
        const stats = enhancedEngine.getStats();

        expect(stats).toHaveProperty('config');
        expect(stats).toHaveProperty('vectors');
        expect(stats).toHaveProperty('performance');
        expect(stats).toHaveProperty('isotonic_reranker');
        expect(stats).toHaveProperty('optimized_hnsw');

        expect(stats.vectors).toBe(3); // 3 indexed documents
        expect(stats.performance).toHaveProperty('avgLatencyMs');
        expect(stats.config).toHaveProperty('enableIsotonicCalibration');
      });
    });

    describe('Performance Regression Testing', () => {
      it('should maintain target latency improvements', async () => {
        const testCases = [
          { query: 'function test', candidates: 5 },
          { query: 'calculate sum addition', candidates: 10 },
          { query: 'class method property', candidates: 15 },
          { query: 'how to implement sorting algorithm', candidates: 20 }
        ];

        const latencies: number[] = [];

        for (const testCase of testCases) {
          const candidates = Array.from({ length: testCase.candidates }, (_, i) => ({
            doc_id: `perf_test_${i}`,
            file_path: `/test${i}.js`,
            line: 1,
            col: 1,
            score: Math.random(),
            match_reasons: ['symbol'] as any[],
            context: `function test${i}() { return ${i}; }`
          }));

          const context = {
            trace_id: `perf-test-${testCase.candidates}`,
            query: testCase.query,
            mode: 'hybrid',
            k: 10,
            fuzzy_distance: 0,
            started_at: new Date(),
            stages: []
          } as SearchContext;

          const startTime = Date.now();
          await enhancedEngine.rerankCandidates(candidates, context, 10);
          const latency = Date.now() - startTime;

          latencies.push(latency);

          // Individual test should meet Stage-C budget
          expect(latency).toBeLessThan(12); // 12ms Stage-C budget
        }

        // Average latency should show improvement target
        const avgLatency = latencies.reduce((sum, lat) => sum + lat, 0) / latencies.length;
        expect(avgLatency).toBeLessThan(10); // Target: 6-8ms, allowing some margin

        console.log(`Average B3 optimization latency: ${avgLatency.toFixed(1)}ms`);
      });

      it('should maintain quality metrics under load', async () => {
        const batchSize = 20;
        const batches = 5;
        const qualityScores: number[] = [];

        for (let batch = 0; batch < batches; batch++) {
          const candidates = Array.from({ length: batchSize }, (_, i) => ({
            doc_id: `batch_${batch}_doc_${i}`,
            file_path: `/batch${batch}/test${i}.js`,
            line: 1,
            col: 1,
            score: Math.random() * 0.8 + 0.1, // 0.1 to 0.9
            match_reasons: ['symbol'] as any[],
            context: `function testBatch${batch}_${i}() { return "test"; }`
          }));

          const context = {
            trace_id: `quality-batch-${batch}`,
            query: 'test function implementation',
            mode: 'hybrid',
            k: 10,
            fuzzy_distance: 0,
            started_at: new Date(),
            stages: []
          } as SearchContext;

          const results = await enhancedEngine.rerankCandidates(candidates, context, 10);

          // Calculate quality score (higher scores should be ranked higher)
          let qualityScore = 0;
          for (let i = 0; i < results.length - 1; i++) {
            if (results[i].score >= results[i + 1].score) {
              qualityScore++;
            }
          }
          qualityScore = results.length > 1 ? qualityScore / (results.length - 1) : 1;
          qualityScores.push(qualityScore);
        }

        const avgQuality = qualityScores.reduce((sum, score) => sum + score, 0) / qualityScores.length;
        
        // Should maintain high ranking quality
        expect(avgQuality).toBeGreaterThan(0.95); // 95% correct ranking
      });
    });
  });
});