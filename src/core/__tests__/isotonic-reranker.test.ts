/**
 * Tests for Isotonic Calibrated Reranker
 * Covers isotonic regression, PAVA algorithm, score calibration, and performance optimization
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import {
  IsotonicCalibrator,
  IsotonicReranker,
  CalibrationPoint,
  IsotonicConfig,
  poolAdjacentViolators,
  interpolateCalibration,
  DEFAULT_ISOTONIC_CONFIG,
} from '../isotonic-reranker.js';
import type { SearchHit, SearchContext } from '../types/core.js';

// Mock the tracer
vi.mock('../../telemetry/tracer.js', () => ({
  LensTracer: {
    createChildSpan: vi.fn(() => ({
      setAttributes: vi.fn(),
      setStatus: vi.fn(),
      end: vi.fn(),
    })),
  },
}));

// Mock the learned reranker
vi.mock('./learned-reranker.js', () => ({
  LearnedReranker: vi.fn().mockImplementation(() => ({
    rerank: vi.fn().mockImplementation((hits) => 
      hits.map((hit: any, index: number) => ({
        ...hit,
        score: 0.8 - index * 0.1, // Declining scores
        rerank_confidence: 0.9,
      }))
    ),
  })),
}));

describe('Isotonic Calibrated Reranker', () => {
  let calibrator: IsotonicCalibrator;
  let reranker: IsotonicReranker;
  let mockConfig: IsotonicConfig;

  beforeEach(() => {
    vi.clearAllMocks();
    
    mockConfig = {
      enabled: true,
      minCalibrationData: 10,
      confidenceCutoff: 0.5,
      maxLatencyMs: 6,
      calibrationUpdateFreq: 100,
    };
    
    calibrator = new IsotonicCalibrator(mockConfig);
    reranker = new IsotonicReranker(mockConfig);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Isotonic Calibrator', () => {
    describe('Calibration Data Management', () => {
      it('should add calibration points correctly', () => {
        calibrator.addCalibrationPoint(0.8, 1.0);
        calibrator.addCalibrationPoint(0.6, 0.0);
        calibrator.addCalibrationPoint(0.9, 1.0);
        
        expect(calibrator.getCalibrationDataSize()).toBe(3);
      });

      it('should limit calibration data size to prevent memory growth', () => {
        // Add more than the memory limit (5000)
        for (let i = 0; i < 5100; i++) {
          calibrator.addCalibrationPoint(
            Math.random(),
            Math.round(Math.random())
          );
        }
        
        // Should be trimmed to 3000 most recent points
        expect(calibrator.getCalibrationDataSize()).toBe(3000);
      });

      it('should maintain chronological order of calibration data', () => {
        const points: CalibrationPoint[] = [
          { predicted_score: 0.9, actual_relevance: 1 },
          { predicted_score: 0.5, actual_relevance: 0 },
          { predicted_score: 0.7, actual_relevance: 1 },
        ];
        
        points.forEach(point => 
          calibrator.addCalibrationPoint(point.predicted_score, point.actual_relevance)
        );
        
        const data = calibrator.getCalibrationData();
        expect(data).toHaveLength(3);
        expect(data[0].predicted_score).toBe(0.9);
        expect(data[2].predicted_score).toBe(0.7);
      });
    });

    describe('Isotonic Regression (PAVA)', () => {
      it('should fit calibration when sufficient data available', () => {
        // Add monotonic calibration data
        const trainingData = [
          [0.1, 0.0], [0.2, 0.1], [0.4, 0.3], [0.6, 0.5],
          [0.7, 0.6], [0.8, 0.8], [0.9, 0.9], [0.95, 1.0],
          [0.3, 0.2], [0.5, 0.4], [0.65, 0.6], [0.85, 0.85]
        ];
        
        trainingData.forEach(([pred, actual]) => 
          calibrator.addCalibrationPoint(pred, actual)
        );
        
        const success = calibrator.fitCalibration();
        
        expect(success).toBe(true);
        expect(calibrator.isFitted()).toBe(true);
      });

      it('should not fit with insufficient data', () => {
        // Add only a few points (less than minCalibrationData)
        calibrator.addCalibrationPoint(0.8, 1);
        calibrator.addCalibrationPoint(0.6, 0);
        
        const success = calibrator.fitCalibration();
        
        expect(success).toBe(false);
        expect(calibrator.isFitted()).toBe(false);
      });

      it('should produce monotonic calibration mapping', () => {
        // Add training data with some violations
        const trainingData = [
          [0.1, 0.2], [0.2, 0.1], // Violation: lower predicted has higher actual
          [0.3, 0.3], [0.4, 0.5], [0.5, 0.4], // Another violation
          [0.6, 0.6], [0.7, 0.8], [0.8, 0.7], // Another violation
          [0.9, 0.9], [0.95, 1.0], [0.85, 0.85], [0.75, 0.75]
        ];
        
        trainingData.forEach(([pred, actual]) => 
          calibrator.addCalibrationPoint(pred, actual)
        );
        
        calibrator.fitCalibration();
        
        // Test that calibrated scores are monotonic
        const testScores = [0.2, 0.4, 0.6, 0.8];
        const calibratedScores = testScores.map(score => 
          calibrator.calibrateScore(score)
        );
        
        for (let i = 1; i < calibratedScores.length; i++) {
          expect(calibratedScores[i]).toBeGreaterThanOrEqual(calibratedScores[i - 1]);
        }
      });

      it('should handle edge cases in PAVA algorithm', () => {
        // Test with constant predictions
        const constantPredictions = Array(12).fill(null).map((_, i) => 
          [0.5, i % 2] // Same prediction, alternating relevance
        );
        
        constantPredictions.forEach(([pred, actual]) => 
          calibrator.addCalibrationPoint(pred as number, actual as number)
        );
        
        expect(() => calibrator.fitCalibration()).not.toThrow();
        expect(calibrator.isFitted()).toBe(true);
      });

      it('should handle perfect calibration data', () => {
        // Data that's already perfectly calibrated
        const perfectData = [
          [0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3],
          [0.4, 0.4], [0.5, 0.5], [0.6, 0.6], [0.7, 0.7],
          [0.8, 0.8], [0.9, 0.9], [1.0, 1.0]
        ];
        
        perfectData.forEach(([pred, actual]) => 
          calibrator.addCalibrationPoint(pred, actual)
        );
        
        calibrator.fitCalibration();
        
        // Calibrated scores should be close to original for perfect data
        const testScore = 0.75;
        const calibrated = calibrator.calibrateScore(testScore);
        
        expect(Math.abs(calibrated - testScore)).toBeLessThan(0.1);
      });
    });

    describe('Score Calibration', () => {
      beforeEach(() => {
        // Set up calibrator with training data
        const trainingData = [
          [0.1, 0.05], [0.2, 0.15], [0.3, 0.25], [0.4, 0.4],
          [0.5, 0.5], [0.6, 0.65], [0.7, 0.75], [0.8, 0.85],
          [0.9, 0.95], [0.15, 0.1], [0.35, 0.3], [0.55, 0.6]
        ];
        
        trainingData.forEach(([pred, actual]) => 
          calibrator.addCalibrationPoint(pred, actual)
        );
        
        calibrator.fitCalibration();
      });

      it('should calibrate scores within training range', () => {
        const testScores = [0.2, 0.5, 0.8];
        
        testScores.forEach(score => {
          const calibrated = calibrator.calibrateScore(score);
          
          expect(calibrated).toBeGreaterThanOrEqual(0);
          expect(calibrated).toBeLessThanOrEqual(1);
          expect(typeof calibrated).toBe('number');
          expect(calibrated).not.toBeNaN();
        });
      });

      it('should extrapolate for scores outside training range', () => {
        const belowRange = calibrator.calibrateScore(0.05);
        const aboveRange = calibrator.calibrateScore(0.95);
        
        expect(belowRange).toBeGreaterThanOrEqual(0);
        expect(aboveRange).toBeLessThanOrEqual(1);
        expect(belowRange).toBeLessThan(aboveRange); // Monotonicity preserved
      });

      it('should handle boundary values correctly', () => {
        const minScore = calibrator.calibrateScore(0.0);
        const maxScore = calibrator.calibrateScore(1.0);
        
        expect(minScore).toBeGreaterThanOrEqual(0);
        expect(maxScore).toBeLessThanOrEqual(1);
        expect(minScore).toBeLessThanOrEqual(maxScore);
      });

      it('should return original score when not fitted', () => {
        const untrainedCalibrator = new IsotonicCalibrator(mockConfig);
        const originalScore = 0.75;
        const calibrated = untrainedCalibrator.calibrateScore(originalScore);
        
        expect(calibrated).toBe(originalScore);
      });

      it('should interpolate between calibration points', () => {
        // Test interpolation between known points
        const midpoint = 0.55; // Between training points
        const calibrated = calibrator.calibrateScore(midpoint);
        
        // Should be between the calibrated values of neighboring points
        const below = calibrator.calibrateScore(0.5);
        const above = calibrator.calibrateScore(0.6);
        
        expect(calibrated).toBeGreaterThanOrEqual(below);
        expect(calibrated).toBeLessThanOrEqual(above);
      });
    });

    describe('Calibration Updates and Maintenance', () => {
      it('should update calibration periodically', () => {
        const initialData = [
          [0.3, 0.3], [0.5, 0.5], [0.7, 0.7], [0.4, 0.4],
          [0.6, 0.6], [0.8, 0.8], [0.2, 0.2], [0.9, 0.9],
          [0.1, 0.1], [0.35, 0.35], [0.45, 0.45], [0.55, 0.55]
        ];
        
        initialData.forEach(([pred, actual]) => 
          calibrator.addCalibrationPoint(pred, actual)
        );
        
        calibrator.fitCalibration();
        const initialCalibrated = calibrator.calibrateScore(0.5);
        
        // Add more data that changes the calibration
        const newData = [
          [0.5, 0.3], [0.5, 0.4], [0.5, 0.35] // Lower actual for same predicted
        ];
        
        newData.forEach(([pred, actual]) => 
          calibrator.addCalibrationPoint(pred, actual)
        );
        
        calibrator.fitCalibration();
        const updatedCalibrated = calibrator.calibrateScore(0.5);
        
        expect(updatedCalibrated).not.toEqual(initialCalibrated);
      });

      it('should maintain calibration quality metrics', () => {
        const trainingData = [
          [0.1, 0.05], [0.2, 0.18], [0.3, 0.28], [0.4, 0.42],
          [0.5, 0.51], [0.6, 0.63], [0.7, 0.72], [0.8, 0.84],
          [0.9, 0.91], [0.15, 0.12], [0.25, 0.23], [0.35, 0.33]
        ];
        
        trainingData.forEach(([pred, actual]) => 
          calibrator.addCalibrationPoint(pred, actual)
        );
        
        calibrator.fitCalibration();
        const metrics = calibrator.getCalibrationMetrics();
        
        expect(metrics.reliability_score).toBeGreaterThan(0.8);
        expect(metrics.monotonicity_violations).toBe(0);
        expect(metrics.calibration_error).toBeLessThan(0.1);
      });
    });
  });

  describe('Isotonic Reranker Integration', () => {
    let mockHits: SearchHit[];
    let mockContext: SearchContext;

    beforeEach(() => {
      mockHits = [
        {
          file: 'src/auth.ts',
          start_line: 10,
          end_line: 15,
          snippet: 'function authenticate(user) { return true; }',
          score: 0.9,
          span_id: 'span1',
        },
        {
          file: 'src/user.ts', 
          start_line: 20,
          end_line: 25,
          snippet: 'class User { login() {} }',
          score: 0.7,
          span_id: 'span2',
        },
        {
          file: 'src/api.ts',
          start_line: 5,
          end_line: 10,
          snippet: 'export const api = {};',
          score: 0.5,
          span_id: 'span3',
        },
      ];
      
      mockContext = {
        query: 'user authentication',
        mode: 'hybrid',
        max_results: 10,
        include_snippets: true,
        filters: {},
      };
    });

    describe('Reranking with Calibration', () => {
      it('should rerank hits using calibrated scores', async () => {
        // Train the calibrator first
        const trainingData = [
          [0.9, 0.95], [0.8, 0.75], [0.7, 0.65], [0.6, 0.5],
          [0.5, 0.4], [0.4, 0.3], [0.3, 0.2], [0.85, 0.8],
          [0.75, 0.7], [0.65, 0.6], [0.55, 0.45], [0.45, 0.35]
        ];
        
        trainingData.forEach(([pred, actual]) => 
          reranker.addTrainingData(pred, actual)
        );
        
        const result = await reranker.rerank(mockHits, mockContext);
        
        expect(result).toBeDefined();
        expect(result.hits).toHaveLength(mockHits.length);
        expect(result.rerank_applied).toBe(true);
        expect(result.calibration_applied).toBe(true);
        
        // Scores should be calibrated (potentially different from original)
        const originalScores = mockHits.map(hit => hit.score);
        const rerankedScores = result.hits.map(hit => hit.score);
        
        // At least some scores should be different due to calibration
        const scoresDifferent = originalScores.some((score, i) => 
          Math.abs(score - rerankedScores[i]) > 0.01
        );
        expect(scoresDifferent).toBe(true);
      });

      it('should maintain hit ordering consistency', async () => {
        const result = await reranker.rerank(mockHits, mockContext);
        
        // Hits should be ordered by calibrated score (descending)
        for (let i = 1; i < result.hits.length; i++) {
          expect(result.hits[i-1].score).toBeGreaterThanOrEqual(result.hits[i].score);
        }
      });

      it('should skip reranking when confidence is too low', async () => {
        const lowConfidenceConfig = {
          ...mockConfig,
          confidenceCutoff: 0.95, // Very high threshold
        };
        
        const lowConfidenceReranker = new IsotonicReranker(lowConfidenceConfig);
        const result = await lowConfidenceReranker.rerank(mockHits, mockContext);
        
        expect(result.rerank_applied).toBe(false);
        expect(result.hits).toEqual(mockHits); // Original order preserved
      });

      it('should respect latency budget', async () => {
        const strictLatencyConfig = {
          ...mockConfig,
          maxLatencyMs: 1, // Very strict budget
        };
        
        const fastReranker = new IsotonicReranker(strictLatencyConfig);
        
        const start = performance.now();
        const result = await fastReranker.rerank(mockHits, mockContext);
        const duration = performance.now() - start;
        
        // Should complete within budget or fallback gracefully
        expect(duration).toBeLessThan(10); // Allow some overhead
        expect(result).toBeDefined();
      });

      it('should handle empty hit list gracefully', async () => {
        const result = await reranker.rerank([], mockContext);
        
        expect(result.hits).toEqual([]);
        expect(result.rerank_applied).toBe(false);
        expect(result.processing_time_ms).toBeDefined();
      });

      it('should handle single hit gracefully', async () => {
        const singleHit = [mockHits[0]];
        const result = await reranker.rerank(singleHit, mockContext);
        
        expect(result.hits).toHaveLength(1);
        expect(result.hits[0]).toBe(singleHit[0]);
      });
    });

    describe('Performance Optimization', () => {
      it('should meet latency targets for typical workloads', async () => {
        // Create larger hit set for realistic testing
        const largeHitSet = Array.from({ length: 50 }, (_, i) => ({
          file: `src/file${i}.ts`,
          start_line: i * 10,
          end_line: i * 10 + 5,
          snippet: `function func${i}() { return ${i}; }`,
          score: 0.9 - i * 0.01,
          span_id: `span${i}`,
        }));
        
        const start = performance.now();
        const result = await reranker.rerank(largeHitSet, mockContext);
        const duration = performance.now() - start;
        
        // Should meet target latency of 6-8ms
        expect(duration).toBeLessThan(mockConfig.maxLatencyMs * 2); // Allow some margin
        expect(result.hits).toHaveLength(largeHitSet.length);
      });

      it('should cache calibration computations', async () => {
        // Train calibrator
        const trainingData = Array.from({ length: 50 }, (_, i) => 
          [i / 50, Math.random()]
        );
        
        trainingData.forEach(([pred, actual]) => 
          reranker.addTrainingData(pred, actual)
        );
        
        // First rerank
        const start1 = performance.now();
        await reranker.rerank(mockHits, mockContext);
        const duration1 = performance.now() - start1;
        
        // Second rerank (should use cache)
        const start2 = performance.now();
        await reranker.rerank(mockHits, mockContext);
        const duration2 = performance.now() - start2;
        
        // Second run should be faster due to caching
        expect(duration2).toBeLessThanOrEqual(duration1 * 1.1); // Allow 10% variance
      });

      it('should handle concurrent reranking requests', async () => {
        const promises = Array.from({ length: 10 }, () => 
          reranker.rerank(mockHits, mockContext)
        );
        
        const results = await Promise.all(promises);
        
        // All results should be consistent
        results.forEach(result => {
          expect(result.hits).toHaveLength(mockHits.length);
          expect(result.processing_time_ms).toBeDefined();
        });
      });
    });

    describe('Quality Validation', () => {
      it('should provide reranking diagnostics', async () => {
        const result = await reranker.rerank(mockHits, mockContext);
        
        expect(result.diagnostics).toBeDefined();
        expect(result.diagnostics.calibration_confidence).toBeGreaterThanOrEqual(0);
        expect(result.diagnostics.score_adjustments).toBeDefined();
        expect(result.diagnostics.monotonicity_preserved).toBe(true);
      });

      it('should detect and report calibration issues', async () => {
        // Add problematic training data
        const problematicData = [
          [0.9, 0.1], [0.8, 0.9], [0.7, 0.2], // Inconsistent patterns
          [0.6, 0.8], [0.5, 0.3], [0.4, 0.7],
          [0.3, 0.4], [0.2, 0.6], [0.1, 0.5],
          [0.95, 0.05], [0.85, 0.95], [0.75, 0.25]
        ];
        
        problematicData.forEach(([pred, actual]) => 
          reranker.addTrainingData(pred, actual)
        );
        
        const result = await reranker.rerank(mockHits, mockContext);
        
        expect(result.diagnostics.calibration_warnings).toBeDefined();
        if (result.diagnostics.calibration_warnings.length > 0) {
          expect(result.diagnostics.calibration_warnings[0]).toMatch(/calibration|quality|reliability/i);
        }
      });

      it('should validate score distribution properties', async () => {
        // Add realistic training data
        const realisticData = [
          [0.9, 0.85], [0.8, 0.75], [0.7, 0.65], [0.6, 0.55],
          [0.5, 0.45], [0.4, 0.35], [0.3, 0.25], [0.2, 0.15],
          [0.85, 0.8], [0.75, 0.7], [0.65, 0.6], [0.45, 0.4]
        ];
        
        realisticData.forEach(([pred, actual]) => 
          reranker.addTrainingData(pred, actual)
        );
        
        const result = await reranker.rerank(mockHits, mockContext);
        
        // Check that score distribution is reasonable
        const scores = result.hits.map(hit => hit.score);
        const minScore = Math.min(...scores);
        const maxScore = Math.max(...scores);
        
        expect(minScore).toBeGreaterThanOrEqual(0);
        expect(maxScore).toBeLessThanOrEqual(1);
        expect(maxScore - minScore).toBeGreaterThan(0.1); // Some spread in scores
      });
    });

    describe('Error Handling and Edge Cases', () => {
      it('should handle malformed hits gracefully', async () => {
        const malformedHits = [
          { ...mockHits[0], score: NaN },
          { ...mockHits[1], score: Infinity },
          { ...mockHits[2], score: -1 },
        ];
        
        const result = await reranker.rerank(malformedHits as any, mockContext);
        
        // Should sanitize or filter bad scores
        const validScores = result.hits.filter(hit => 
          isFinite(hit.score) && hit.score >= 0 && hit.score <= 1
        );
        
        expect(validScores.length).toBeGreaterThan(0);
      });

      it('should handle calibration training failures', async () => {
        // Add insufficient or invalid training data
        reranker.addTrainingData(0.5, NaN);
        reranker.addTrainingData(Infinity, 0.5);
        
        const result = await reranker.rerank(mockHits, mockContext);
        
        // Should fallback gracefully without calibration
        expect(result.calibration_applied).toBe(false);
        expect(result.hits).toHaveLength(mockHits.length);
      });

      it('should handle disabled configuration', async () => {
        const disabledConfig = { ...mockConfig, enabled: false };
        const disabledReranker = new IsotonicReranker(disabledConfig);
        
        const result = await disabledReranker.rerank(mockHits, mockContext);
        
        expect(result.rerank_applied).toBe(false);
        expect(result.hits).toEqual(mockHits);
      });
    });
  });

  describe('Pool-Adjacent-Violators Algorithm (PAVA)', () => {
    it('should implement PAVA correctly for monotonic regression', () => {
      const data = [
        { x: 0.1, y: 0.2 }, { x: 0.2, y: 0.1 }, // Violation
        { x: 0.3, y: 0.4 }, { x: 0.4, y: 0.3 }, // Violation
        { x: 0.5, y: 0.6 }, { x: 0.6, y: 0.8 }
      ];
      
      const result = poolAdjacentViolators(data);
      
      // Result should be monotonic
      for (let i = 1; i < result.length; i++) {
        expect(result[i].y).toBeGreaterThanOrEqual(result[i-1].y);
      }
      
      // Should have same or fewer points due to pooling
      expect(result.length).toBeLessThanOrEqual(data.length);
    });

    it('should handle already monotonic data', () => {
      const monotonicData = [
        { x: 0.1, y: 0.1 }, { x: 0.2, y: 0.2 },
        { x: 0.3, y: 0.3 }, { x: 0.4, y: 0.4 }
      ];
      
      const result = poolAdjacentViolators(monotonicData);
      
      expect(result).toEqual(monotonicData);
    });

    it('should handle constant values correctly', () => {
      const constantData = [
        { x: 0.1, y: 0.5 }, { x: 0.2, y: 0.5 },
        { x: 0.3, y: 0.5 }, { x: 0.4, y: 0.5 }
      ];
      
      const result = poolAdjacentViolators(constantData);
      
      // Should pool into single point or maintain constant values
      expect(result.every(point => point.y === 0.5)).toBe(true);
    });
  });

  describe('Calibration Interpolation', () => {
    it('should interpolate between calibration points', () => {
      const calibrationMap = new Map([
        [0.2, 0.15],
        [0.4, 0.35],
        [0.6, 0.55],
        [0.8, 0.75]
      ]);
      
      // Test interpolation between points
      const interpolated = interpolateCalibration(0.5, calibrationMap);
      
      // Should be between 0.35 and 0.55
      expect(interpolated).toBeGreaterThan(0.35);
      expect(interpolated).toBeLessThan(0.55);
    });

    it('should extrapolate beyond calibration range', () => {
      const calibrationMap = new Map([[0.5, 0.4], [0.7, 0.6]]);
      
      const below = interpolateCalibration(0.3, calibrationMap);
      const above = interpolateCalibration(0.9, calibrationMap);
      
      expect(below).toBeLessThan(0.4);
      expect(above).toBeGreaterThan(0.6);
      expect(below).toBeGreaterThanOrEqual(0);
      expect(above).toBeLessThanOrEqual(1);
    });

    it('should handle exact calibration points', () => {
      const calibrationMap = new Map([[0.5, 0.4], [0.7, 0.6]]);
      
      const exact = interpolateCalibration(0.5, calibrationMap);
      expect(exact).toBe(0.4);
    });
  });
});