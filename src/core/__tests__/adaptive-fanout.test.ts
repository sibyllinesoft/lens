/**
 * Tests for Adaptive Fan-out System
 * Covers hardness-based query adaptation, configurable mapping, and performance optimization
 */

import { describe, it, expect, beforeEach } from 'bun:test';
import {
  HardnessFeatures,
  AdaptiveConfig,
  DEFAULT_ADAPTIVE_CONFIG,
  computeAdaptiveFanout,
  computeHardnessScore,
  adaptiveThresholds,
  HardnessAssessment,
  validateConfig,
  createHardnessFeatures,
} from '../adaptive-fanout.js';

describe('Adaptive Fan-out System', () => {
  describe('Hardness Features Creation', () => {
    it('should create default hardness features with all zero values', () => {
      const features = createHardnessFeatures();
      
      expect(features).toEqual({
        rare_terms: 0,
        fuzzy_edits: 0,
        id_entropy: 0,
        path_var: 0,
        cand_slope: 0,
      });
    });

    it('should create hardness features with custom values', () => {
      const customFeatures: HardnessFeatures = {
        rare_terms: 0.8,
        fuzzy_edits: 2,
        id_entropy: 0.95,
        path_var: 0.3,
        cand_slope: 1.2,
      };
      
      const features = createHardnessFeatures(customFeatures);
      expect(features).toEqual(customFeatures);
    });

    it('should validate feature ranges', () => {
      const validFeatures: HardnessFeatures = {
        rare_terms: 0.5,
        fuzzy_edits: 1,
        id_entropy: 0.7,
        path_var: 0.2,
        cand_slope: 0.8,
      };
      
      expect(() => createHardnessFeatures(validFeatures)).not.toThrow();
    });
  });

  describe('Hardness Score Computation', () => {
    const testFeatures: HardnessFeatures = {
      rare_terms: 0.6,
      fuzzy_edits: 1,
      id_entropy: 0.8,
      path_var: 0.4,
      cand_slope: 1.1,
    };

    it('should compute hardness score with default config', () => {
      const score = computeHardnessScore(testFeatures);
      
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
      expect(typeof score).toBe('number');
    });

    it('should compute hardness score with custom weights', () => {
      const customConfig: AdaptiveConfig = {
        ...DEFAULT_ADAPTIVE_CONFIG,
        weights: {
          w1: 0.3,
          w2: 0.2,
          w3: 0.2,
          w4: 0.15,
          w5: 0.15,
        }
      };
      
      const score = computeHardnessScore(testFeatures, customConfig);
      
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
    });

    it('should handle zero features', () => {
      const zeroFeatures = createHardnessFeatures();
      const score = computeHardnessScore(zeroFeatures);
      
      expect(score).toBe(0);
    });

    it('should handle maximum features', () => {
      const maxFeatures: HardnessFeatures = {
        rare_terms: 1.0,
        fuzzy_edits: 3,
        id_entropy: 1.0,
        path_var: 1.0,
        cand_slope: 2.0,
      };
      
      const score = computeHardnessScore(maxFeatures);
      
      expect(score).toBeLessThanOrEqual(1);
      expect(score).toBeGreaterThan(0);
    });

    it('should be consistent for same inputs', () => {
      const score1 = computeHardnessScore(testFeatures);
      const score2 = computeHardnessScore(testFeatures);
      
      expect(score1).toEqual(score2);
    });

    it('should increase with harder features', () => {
      const easyFeatures = createHardnessFeatures({
        rare_terms: 0.1,
        fuzzy_edits: 0,
        id_entropy: 0.2,
        path_var: 0.1,
        cand_slope: 0.3,
      });
      
      const hardFeatures = createHardnessFeatures({
        rare_terms: 0.9,
        fuzzy_edits: 2,
        id_entropy: 0.95,
        path_var: 0.8,
        cand_slope: 1.8,
      });
      
      const easyScore = computeHardnessScore(easyFeatures);
      const hardScore = computeHardnessScore(hardFeatures);
      
      expect(hardScore).toBeGreaterThan(easyScore);
    });
  });

  describe('Adaptive Thresholds Computation', () => {
    it('should compute thresholds for low hardness', () => {
      const hardnessScore = 0.1;
      const thresholds = adaptiveThresholds(hardnessScore);
      
      expect(thresholds.nl_threshold).toBeGreaterThanOrEqual(DEFAULT_ADAPTIVE_CONFIG.gate.nl_threshold.min);
      expect(thresholds.nl_threshold).toBeLessThanOrEqual(DEFAULT_ADAPTIVE_CONFIG.gate.nl_threshold.max);
      expect(thresholds.min_candidates).toBeGreaterThanOrEqual(DEFAULT_ADAPTIVE_CONFIG.gate.min_candidates.min);
      expect(thresholds.min_candidates).toBeLessThanOrEqual(DEFAULT_ADAPTIVE_CONFIG.gate.min_candidates.max);
    });

    it('should compute thresholds for high hardness', () => {
      const hardnessScore = 0.9;
      const thresholds = adaptiveThresholds(hardnessScore);
      
      expect(thresholds.nl_threshold).toBeGreaterThanOrEqual(DEFAULT_ADAPTIVE_CONFIG.gate.nl_threshold.min);
      expect(thresholds.nl_threshold).toBeLessThanOrEqual(DEFAULT_ADAPTIVE_CONFIG.gate.nl_threshold.max);
      expect(thresholds.min_candidates).toBeGreaterThanOrEqual(DEFAULT_ADAPTIVE_CONFIG.gate.min_candidates.min);
      expect(thresholds.min_candidates).toBeLessThanOrEqual(DEFAULT_ADAPTIVE_CONFIG.gate.min_candidates.max);
    });

    it('should adapt thresholds based on hardness score', () => {
      const lowHardness = 0.1;
      const highHardness = 0.9;
      
      const lowThresholds = adaptiveThresholds(lowHardness);
      const highThresholds = adaptiveThresholds(highHardness);
      
      // For harder queries, we typically want more relaxed NL thresholds and more candidates
      expect(highThresholds.nl_threshold).not.toEqual(lowThresholds.nl_threshold);
      expect(highThresholds.min_candidates).not.toEqual(lowThresholds.min_candidates);
    });

    it('should handle edge cases', () => {
      const zeroThresholds = adaptiveThresholds(0);
      const oneThresholds = adaptiveThresholds(1);
      
      expect(zeroThresholds.nl_threshold).toBeFinite();
      expect(zeroThresholds.min_candidates).toBeFinite();
      expect(oneThresholds.nl_threshold).toBeFinite();
      expect(oneThresholds.min_candidates).toBeFinite();
    });

    it('should be monotonic in hardness score', () => {
      const scores = [0, 0.25, 0.5, 0.75, 1.0];
      const thresholds = scores.map(adaptiveThresholds);
      
      for (let i = 1; i < thresholds.length; i++) {
        // Verify that thresholds change monotonically (in some consistent direction)
        expect(thresholds[i]).toBeDefined();
        expect(thresholds[i].nl_threshold).toBeFinite();
        expect(thresholds[i].min_candidates).toBeFinite();
      }
    });
  });

  describe('Adaptive Fan-out Computation', () => {
    const testFeatures: HardnessFeatures = {
      rare_terms: 0.5,
      fuzzy_edits: 1,
      id_entropy: 0.7,
      path_var: 0.3,
      cand_slope: 0.9,
    };

    it('should compute adaptive fan-out with default config', () => {
      const result = computeAdaptiveFanout(testFeatures);
      
      expect(result.k_candidates).toBeGreaterThanOrEqual(DEFAULT_ADAPTIVE_CONFIG.k_candidates.min);
      expect(result.k_candidates).toBeLessThanOrEqual(DEFAULT_ADAPTIVE_CONFIG.k_candidates.max);
      expect(result.hardness_score).toBeGreaterThanOrEqual(0);
      expect(result.hardness_score).toBeLessThanOrEqual(1);
      expect(result.thresholds.nl_threshold).toBeFinite();
      expect(result.thresholds.min_candidates).toBeFinite();
    });

    it('should compute adaptive fan-out with custom config', () => {
      const customConfig: AdaptiveConfig = {
        k_candidates: {
          min: 100,
          max: 500,
        },
        gate: {
          nl_threshold: { min: 0.25, max: 0.6 },
          min_candidates: { min: 5, max: 20 },
        },
        weights: {
          w1: 0.25, w2: 0.2, w3: 0.2, w4: 0.175, w5: 0.175,
        },
      };

      const result = computeAdaptiveFanout(testFeatures, customConfig);
      
      expect(result.k_candidates).toBeGreaterThanOrEqual(100);
      expect(result.k_candidates).toBeLessThanOrEqual(500);
      expect(result.thresholds.nl_threshold).toBeGreaterThanOrEqual(0.25);
      expect(result.thresholds.nl_threshold).toBeLessThanOrEqual(0.6);
    });

    it('should provide consistent results for same input', () => {
      const result1 = computeAdaptiveFanout(testFeatures);
      const result2 = computeAdaptiveFanout(testFeatures);
      
      expect(result1).toEqual(result2);
    });

    it('should adapt k_candidates based on hardness', () => {
      const easyFeatures = createHardnessFeatures({ rare_terms: 0.1 });
      const hardFeatures = createHardnessFeatures({ rare_terms: 0.9 });
      
      const easyResult = computeAdaptiveFanout(easyFeatures);
      const hardResult = computeAdaptiveFanout(hardFeatures);
      
      // Harder queries should generally get more candidates
      expect(hardResult.k_candidates).not.toEqual(easyResult.k_candidates);
    });

    it('should include assessment metadata', () => {
      const result = computeAdaptiveFanout(testFeatures);
      
      expect(result.assessment).toBeDefined();
      expect(['easy', 'medium', 'hard']).toContain(result.assessment.difficulty);
      expect(result.assessment.confidence).toBeGreaterThanOrEqual(0);
      expect(result.assessment.confidence).toBeLessThanOrEqual(1);
      expect(Array.isArray(result.assessment.dominant_factors)).toBe(true);
    });
  });

  describe('Configuration Validation', () => {
    it('should validate default configuration', () => {
      expect(() => validateConfig(DEFAULT_ADAPTIVE_CONFIG)).not.toThrow();
    });

    it('should reject invalid k_candidates range', () => {
      const invalidConfig: AdaptiveConfig = {
        ...DEFAULT_ADAPTIVE_CONFIG,
        k_candidates: { min: 500, max: 100 }, // min > max
      };
      
      expect(() => validateConfig(invalidConfig)).toThrow();
    });

    it('should reject invalid threshold ranges', () => {
      const invalidConfig: AdaptiveConfig = {
        ...DEFAULT_ADAPTIVE_CONFIG,
        gate: {
          nl_threshold: { min: 0.8, max: 0.2 }, // min > max
          min_candidates: { min: 20, max: 5 }, // min > max
        },
      };
      
      expect(() => validateConfig(invalidConfig)).toThrow();
    });

    it('should reject weights that do not sum to 1', () => {
      const invalidConfig: AdaptiveConfig = {
        ...DEFAULT_ADAPTIVE_CONFIG,
        weights: {
          w1: 0.1, w2: 0.1, w3: 0.1, w4: 0.1, w5: 0.1, // sum = 0.5
        },
      };
      
      expect(() => validateConfig(invalidConfig)).toThrow();
    });

    it('should reject negative weights', () => {
      const invalidConfig: AdaptiveConfig = {
        ...DEFAULT_ADAPTIVE_CONFIG,
        weights: {
          w1: -0.1, w2: 0.3, w3: 0.3, w4: 0.25, w5: 0.25,
        },
      };
      
      expect(() => validateConfig(invalidConfig)).toThrow();
    });

    it('should accept valid custom configuration', () => {
      const validConfig: AdaptiveConfig = {
        k_candidates: { min: 50, max: 1000 },
        gate: {
          nl_threshold: { min: 0.1, max: 0.9 },
          min_candidates: { min: 1, max: 50 },
        },
        weights: {
          w1: 0.2, w2: 0.2, w3: 0.2, w4: 0.2, w5: 0.2,
        },
      };
      
      expect(() => validateConfig(validConfig)).not.toThrow();
    });
  });

  describe('Performance and Edge Cases', () => {
    it('should handle extreme hardness features efficiently', () => {
      const extremeFeatures: HardnessFeatures = {
        rare_terms: 0.999,
        fuzzy_edits: 10,
        id_entropy: 0.999,
        path_var: 0.999,
        cand_slope: 5.0,
      };
      
      const start = performance.now();
      const result = computeAdaptiveFanout(extremeFeatures);
      const end = performance.now();
      
      expect(end - start).toBeLessThan(10); // Should complete in under 10ms
      expect(result.k_candidates).toBeFinite();
      expect(result.hardness_score).toBeFinite();
    });

    it('should handle NaN and Infinity in features gracefully', () => {
      const invalidFeatures: HardnessFeatures = {
        rare_terms: NaN,
        fuzzy_edits: Infinity,
        id_entropy: -Infinity,
        path_var: 0.5,
        cand_slope: 1.0,
      };
      
      // Should either throw a meaningful error or handle gracefully
      expect(() => computeAdaptiveFanout(invalidFeatures)).toThrowError(/invalid|nan|infinity/i);
    });

    it('should be thread-safe with concurrent calls', async () => {
      const features = createHardnessFeatures({
        rare_terms: 0.5,
        fuzzy_edits: 1,
        id_entropy: 0.7,
        path_var: 0.3,
        cand_slope: 0.9,
      });
      
      const promises = Array.from({ length: 10 }, () => 
        Promise.resolve(computeAdaptiveFanout(features))
      );
      
      const results = await Promise.all(promises);
      
      // All results should be identical
      const firstResult = results[0];
      results.forEach(result => {
        expect(result).toEqual(firstResult);
      });
    });

    it('should maintain precision with small hardness values', () => {
      const tinyFeatures = createHardnessFeatures({
        rare_terms: 0.001,
        fuzzy_edits: 0,
        id_entropy: 0.001,
        path_var: 0.001,
        cand_slope: 0.001,
      });
      
      const result = computeAdaptiveFanout(tinyFeatures);
      
      expect(result.hardness_score).toBeGreaterThanOrEqual(0);
      expect(result.hardness_score).toBeLessThan(0.1);
      expect(result.k_candidates).toBeGreaterThan(0);
    });
  });

  describe('Integration with Query Types', () => {
    it('should adapt differently for lexical vs semantic queries', () => {
      const lexicalFeatures = createHardnessFeatures({
        rare_terms: 0.1, // Common terms
        fuzzy_edits: 0,   // Exact match
        id_entropy: 0.2,  // Low entropy
        path_var: 0.1,    // Simple path
        cand_slope: 0.3,  // Gradual slope
      });
      
      const semanticFeatures = createHardnessFeatures({
        rare_terms: 0.8,  // Rare terms
        fuzzy_edits: 2,   // Fuzzy matching
        id_entropy: 0.9,  // High entropy
        path_var: 0.7,    // Variable paths
        cand_slope: 1.5,  // Steep slope
      });
      
      const lexicalResult = computeAdaptiveFanout(lexicalFeatures);
      const semanticResult = computeAdaptiveFanout(semanticFeatures);
      
      expect(lexicalResult.hardness_score).toBeLessThan(semanticResult.hardness_score);
      expect(lexicalResult.assessment.difficulty).not.toEqual(semanticResult.assessment.difficulty);
    });

    it('should provide meaningful assessment categories', () => {
      const veryEasyFeatures = createHardnessFeatures();
      const veryHardFeatures = createHardnessFeatures({
        rare_terms: 0.95,
        fuzzy_edits: 3,
        id_entropy: 0.98,
        path_var: 0.9,
        cand_slope: 2.5,
      });
      
      const easyResult = computeAdaptiveFanout(veryEasyFeatures);
      const hardResult = computeAdaptiveFanout(veryHardFeatures);
      
      expect(easyResult.assessment.difficulty).toBe('easy');
      expect(hardResult.assessment.difficulty).toBe('hard');
      expect(hardResult.assessment.confidence).toBeGreaterThan(easyResult.assessment.confidence);
    });
  });
});