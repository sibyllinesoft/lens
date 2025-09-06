/**
 * Tests for Feature Flag Manager
 * Covers feature flag logic, environment overrides, and A/B testing
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { FeatureFlagManager, featureFlags, type FeatureFlags } from '../features.js';

describe('Feature Flag Manager', () => {
  let originalEnv: NodeJS.ProcessEnv;
  
  beforeEach(() => {
    // Save original environment
    originalEnv = { ...process.env };
  });
  
  afterEach(() => {
    // Restore original environment
    process.env = originalEnv;
  });

  describe('Constructor and Defaults', () => {
    it('should initialize with default configuration', () => {
      const manager = new FeatureFlagManager();
      const features = manager.getFeatures();
      
      expect(features.bitmapTrigramIndex.enabled).toBe(true);
      expect(features.bitmapTrigramIndex.rolloutPercentage).toBe(100);
      expect(features.bitmapTrigramIndex.minDocumentThreshold).toBe(100);
      expect(features.bitmapTrigramIndex.enablePerformanceLogging).toBe(true);
      
      expect(features.prefilter.enabled).toBe(true);
      expect(features.prefilter.maxCandidatesBeforeFilter).toBe(10000);
      
      expect(features.experimental.advancedFST).toBe(false);
      expect(features.experimental.semanticRerank).toBe(false);
    });

    it('should apply constructor overrides', () => {
      const overrides: Partial<FeatureFlags> = {
        bitmapTrigramIndex: {
          enabled: false,
          rolloutPercentage: 50,
          minDocumentThreshold: 200,
          enablePerformanceLogging: false,
        },
        experimental: {
          advancedFST: true,
          semanticRerank: true,
        },
      };
      
      const manager = new FeatureFlagManager(overrides);
      const features = manager.getFeatures();
      
      expect(features.bitmapTrigramIndex.enabled).toBe(false);
      expect(features.bitmapTrigramIndex.rolloutPercentage).toBe(50);
      expect(features.bitmapTrigramIndex.minDocumentThreshold).toBe(200);
      expect(features.bitmapTrigramIndex.enablePerformanceLogging).toBe(false);
      expect(features.experimental.advancedFST).toBe(true);
      expect(features.experimental.semanticRerank).toBe(true);
      
      // Prefilter should remain default since not overridden
      expect(features.prefilter.enabled).toBe(true);
      expect(features.prefilter.maxCandidatesBeforeFilter).toBe(10000);
    });

    it('should apply partial constructor overrides', () => {
      const overrides: Partial<FeatureFlags> = {
        bitmapTrigramIndex: {
          enabled: false,
          // Only override enabled, others should remain default
        } as any,
      };
      
      const manager = new FeatureFlagManager(overrides);
      const features = manager.getFeatures();
      
      expect(features.bitmapTrigramIndex.enabled).toBe(false);
      // Others should remain default
      expect(features.bitmapTrigramIndex.rolloutPercentage).toBe(100);
      expect(features.bitmapTrigramIndex.minDocumentThreshold).toBe(100);
    });

    it('should return immutable feature copies', () => {
      const manager = new FeatureFlagManager();
      const features1 = manager.getFeatures();
      const features2 = manager.getFeatures();
      
      expect(features1).not.toBe(features2);
      expect(features1).toEqual(features2);
      
      // Modifying returned object should not affect internal state
      features1.bitmapTrigramIndex.enabled = false;
      const features3 = manager.getFeatures();
      expect(features3.bitmapTrigramIndex.enabled).toBe(true);
    });
  });

  describe('Environment Variable Overrides', () => {
    it('should apply bitmap index environment overrides', () => {
      process.env['LENS_BITMAP_INDEX_ENABLED'] = 'false';
      process.env['LENS_BITMAP_ROLLOUT_PERCENTAGE'] = '75';
      process.env['LENS_BITMAP_MIN_DOCS'] = '500';
      
      const manager = new FeatureFlagManager();
      const features = manager.getFeatures();
      
      expect(features.bitmapTrigramIndex.enabled).toBe(false);
      expect(features.bitmapTrigramIndex.rolloutPercentage).toBe(75);
      expect(features.bitmapTrigramIndex.minDocumentThreshold).toBe(500);
    });

    it('should apply prefilter environment overrides', () => {
      process.env['LENS_PREFILTER_ENABLED'] = 'false';
      
      const manager = new FeatureFlagManager();
      const features = manager.getFeatures();
      
      expect(features.prefilter.enabled).toBe(false);
    });

    it('should apply experimental feature environment overrides', () => {
      process.env['LENS_EXPERIMENTAL_FST'] = 'true';
      
      const manager = new FeatureFlagManager();
      const features = manager.getFeatures();
      
      expect(features.experimental.advancedFST).toBe(true);
    });

    it('should ignore invalid environment values', () => {
      process.env['LENS_BITMAP_ROLLOUT_PERCENTAGE'] = 'invalid';
      process.env['LENS_BITMAP_MIN_DOCS'] = 'not-a-number';
      
      const manager = new FeatureFlagManager();
      const features = manager.getFeatures();
      
      // Should remain defaults
      expect(features.bitmapTrigramIndex.rolloutPercentage).toBe(100);
      expect(features.bitmapTrigramIndex.minDocumentThreshold).toBe(100);
    });

    it('should ignore out-of-range percentage values', () => {
      process.env['LENS_BITMAP_ROLLOUT_PERCENTAGE'] = '150';
      
      const manager = new FeatureFlagManager();
      const features = manager.getFeatures();
      
      // Should remain default
      expect(features.bitmapTrigramIndex.rolloutPercentage).toBe(100);
    });

    it('should ignore negative document thresholds', () => {
      process.env['LENS_BITMAP_MIN_DOCS'] = '-100';
      
      const manager = new FeatureFlagManager();
      const features = manager.getFeatures();
      
      // Should remain default
      expect(features.bitmapTrigramIndex.minDocumentThreshold).toBe(100);
    });

    it('should handle edge case environment values', () => {
      process.env['LENS_BITMAP_ROLLOUT_PERCENTAGE'] = '0';
      process.env['LENS_BITMAP_MIN_DOCS'] = '0';
      
      const manager = new FeatureFlagManager();
      const features = manager.getFeatures();
      
      expect(features.bitmapTrigramIndex.rolloutPercentage).toBe(0);
      expect(features.bitmapTrigramIndex.minDocumentThreshold).toBe(0);
    });
  });

  describe('Bitmap Index Decision Logic', () => {
    let manager: FeatureFlagManager;
    
    beforeEach(() => {
      manager = new FeatureFlagManager();
    });

    it('should reject when bitmap index is disabled', () => {
      manager.updateFeatures({
        bitmapTrigramIndex: { enabled: false } as any,
      });
      
      expect(manager.shouldUseBitmapIndex(1000)).toBe(false);
    });

    it('should reject when document count is below threshold', () => {
      expect(manager.shouldUseBitmapIndex(50)).toBe(false);
      expect(manager.shouldUseBitmapIndex(99)).toBe(false);
      expect(manager.shouldUseBitmapIndex(100)).toBe(true);
      expect(manager.shouldUseBitmapIndex(101)).toBe(true);
    });

    it('should accept when rollout is 100%', () => {
      expect(manager.shouldUseBitmapIndex(1000)).toBe(true);
      expect(manager.shouldUseBitmapIndex(1000, 'any-user-hash')).toBe(true);
    });

    it('should use user hash for consistent rollout decisions', () => {
      manager.updateFeatures({
        bitmapTrigramIndex: { rolloutPercentage: 50 } as any,
      });
      
      const userHash1 = 'user1';
      const userHash2 = 'user2';
      
      // Same user should get consistent results
      const result1a = manager.shouldUseBitmapIndex(1000, userHash1);
      const result1b = manager.shouldUseBitmapIndex(1000, userHash1);
      expect(result1a).toBe(result1b);
      
      // Different users might get different results
      const result2 = manager.shouldUseBitmapIndex(1000, userHash2);
      // Can't assert specific result due to hash variability, but should be consistent
      const result2b = manager.shouldUseBitmapIndex(1000, userHash2);
      expect(result2).toBe(result2b);
    });

    it('should use random selection when no user hash provided', () => {
      manager.updateFeatures({
        bitmapTrigramIndex: { rolloutPercentage: 0 } as any,
      });
      
      // Mock Math.random to control randomness
      const originalRandom = Math.random;
      Math.random = vi.fn(() => 0.5); // 50%
      
      const result = manager.shouldUseBitmapIndex(1000);
      expect(result).toBe(false); // 50% > 0% rollout
      
      Math.random = originalRandom;
    });

    it('should handle edge case rollout percentages', () => {
      // Test 0% rollout
      manager.updateFeatures({
        bitmapTrigramIndex: { rolloutPercentage: 0 } as any,
      });
      
      const originalRandom = Math.random;
      Math.random = vi.fn(() => 0.001); // Very small random value
      
      expect(manager.shouldUseBitmapIndex(1000)).toBe(false);
      
      Math.random = originalRandom;
    });
  });

  describe('Feature Flag Queries', () => {
    let manager: FeatureFlagManager;
    
    beforeEach(() => {
      manager = new FeatureFlagManager();
    });

    it('should return correct prefilter status', () => {
      expect(manager.isPrefilterEnabled()).toBe(true);
      
      manager.updateFeatures({
        prefilter: { enabled: false } as any,
      });
      
      expect(manager.isPrefilterEnabled()).toBe(false);
    });

    it('should return correct prefilter threshold', () => {
      expect(manager.getPrefilterThreshold()).toBe(10000);
      
      manager.updateFeatures({
        prefilter: { maxCandidatesBeforeFilter: 5000 } as any,
      });
      
      expect(manager.getPrefilterThreshold()).toBe(5000);
    });

    it('should return correct bitmap performance logging status', () => {
      expect(manager.isBitmapPerformanceLoggingEnabled()).toBe(true);
      
      manager.updateFeatures({
        bitmapTrigramIndex: { enablePerformanceLogging: false } as any,
      });
      
      expect(manager.isBitmapPerformanceLoggingEnabled()).toBe(false);
    });
  });

  describe('Runtime Feature Updates', () => {
    let manager: FeatureFlagManager;
    
    beforeEach(() => {
      manager = new FeatureFlagManager();
    });

    it('should update features at runtime', () => {
      const updates: Partial<FeatureFlags> = {
        experimental: {
          advancedFST: true,
          semanticRerank: true,
        },
      };
      
      manager.updateFeatures(updates);
      const features = manager.getFeatures();
      
      expect(features.experimental.advancedFST).toBe(true);
      expect(features.experimental.semanticRerank).toBe(true);
      
      // Other features should remain unchanged
      expect(features.bitmapTrigramIndex.enabled).toBe(true);
      expect(features.prefilter.enabled).toBe(true);
    });

    it('should handle partial updates', () => {
      manager.updateFeatures({
        bitmapTrigramIndex: {
          enabled: false,
        } as any,
      });
      
      const features = manager.getFeatures();
      
      expect(features.bitmapTrigramIndex.enabled).toBe(false);
      // Other bitmap properties should remain unchanged
      expect(features.bitmapTrigramIndex.rolloutPercentage).toBe(100);
      expect(features.bitmapTrigramIndex.minDocumentThreshold).toBe(100);
    });

    it('should handle multiple sequential updates', () => {
      manager.updateFeatures({
        experimental: { advancedFST: true } as any,
      });
      
      manager.updateFeatures({
        experimental: { semanticRerank: true } as any,
      });
      
      const features = manager.getFeatures();
      
      expect(features.experimental.advancedFST).toBe(true);
      expect(features.experimental.semanticRerank).toBe(true);
    });
  });

  describe('String Hashing', () => {
    let manager: FeatureFlagManager;
    
    beforeEach(() => {
      manager = new FeatureFlagManager({
        bitmapTrigramIndex: { rolloutPercentage: 50 } as any,
      });
    });

    it('should produce consistent hashes for same input', () => {
      const userHash = 'consistent-user';
      const result1 = manager.shouldUseBitmapIndex(1000, userHash);
      const result2 = manager.shouldUseBitmapIndex(1000, userHash);
      
      expect(result1).toBe(result2);
    });

    it('should produce different results for different inputs', () => {
      // Test with many different inputs to increase chance of different results
      const results = new Set();
      for (let i = 0; i < 100; i++) {
        results.add(manager.shouldUseBitmapIndex(1000, `user-${i}`));
      }
      
      // With 50% rollout and 100 different users, we should see both true and false
      // (This is probabilistic, but very likely to pass)
      expect(results.size).toBe(2); // Should contain both true and false
    });

    it('should handle edge case string inputs', () => {
      // Empty string
      const result1 = manager.shouldUseBitmapIndex(1000, '');
      const result2 = manager.shouldUseBitmapIndex(1000, '');
      expect(result1).toBe(result2);
      
      // Very long string
      const longString = 'a'.repeat(10000);
      const result3 = manager.shouldUseBitmapIndex(1000, longString);
      const result4 = manager.shouldUseBitmapIndex(1000, longString);
      expect(result3).toBe(result4);
      
      // String with special characters
      const specialString = '!@#$%^&*()_+-=[]{}|;:,.<>?/~`"\'\\';
      const result5 = manager.shouldUseBitmapIndex(1000, specialString);
      const result6 = manager.shouldUseBitmapIndex(1000, specialString);
      expect(result5).toBe(result6);
      
      // Unicode characters
      const unicodeString = '🚀🎯💡🔍✨';
      const result7 = manager.shouldUseBitmapIndex(1000, unicodeString);
      const result8 = manager.shouldUseBitmapIndex(1000, unicodeString);
      expect(result7).toBe(result8);
    });

    it('should normalize hash output to 0-1 range', () => {
      // Test multiple strings to ensure all hashes are in valid range
      const testStrings = [
        'test1', 'test2', 'test3', 'very-long-string-to-test-hash-normalization',
        '12345', 'special!@#$%^&*()', '🚀🎯💡', ''
      ];
      
      for (const testString of testStrings) {
        // We can't directly test the private hash method, but we can test its effect
        // through the rollout logic with different percentages
        manager.updateFeatures({
          bitmapTrigramIndex: { rolloutPercentage: 0 } as any,
        });
        const result0 = manager.shouldUseBitmapIndex(1000, testString);
        
        manager.updateFeatures({
          bitmapTrigramIndex: { rolloutPercentage: 100 } as any,
        });
        const result100 = manager.shouldUseBitmapIndex(1000, testString);
        
        // 0% rollout should always be false, 100% should always be true
        expect(result0).toBe(false);
        expect(result100).toBe(true);
      }
    });
  });

  describe('Global Instance', () => {
    it('should export a global feature flags instance', () => {
      expect(featureFlags).toBeInstanceOf(FeatureFlagManager);
    });

    it('should have default configuration in global instance', () => {
      const features = featureFlags.getFeatures();
      
      expect(features.bitmapTrigramIndex.enabled).toBe(true);
      expect(features.prefilter.enabled).toBe(true);
      expect(features.experimental.advancedFST).toBe(false);
      expect(features.experimental.semanticRerank).toBe(false);
    });
  });

  describe('Complex Integration Scenarios', () => {
    it('should handle environment + constructor + runtime overrides', () => {
      // Set environment variables
      process.env['LENS_BITMAP_INDEX_ENABLED'] = 'false';
      process.env['LENS_BITMAP_ROLLOUT_PERCENTAGE'] = '25';
      
      // Constructor overrides
      const constructorOverrides: Partial<FeatureFlags> = {
        bitmapTrigramIndex: {
          minDocumentThreshold: 500,
        } as any,
        experimental: {
          advancedFST: true,
        } as any,
      };
      
      const manager = new FeatureFlagManager(constructorOverrides);
      
      // Runtime updates
      manager.updateFeatures({
        prefilter: {
          maxCandidatesBeforeFilter: 2000,
        } as any,
      });
      
      const features = manager.getFeatures();
      
      // Environment should win for bitmap enabled/rollout
      expect(features.bitmapTrigramIndex.enabled).toBe(false);
      expect(features.bitmapTrigramIndex.rolloutPercentage).toBe(25);
      
      // Constructor should set threshold and experimental
      expect(features.bitmapTrigramIndex.minDocumentThreshold).toBe(500);
      expect(features.experimental.advancedFST).toBe(true);
      
      // Runtime should set prefilter
      expect(features.prefilter.maxCandidatesBeforeFilter).toBe(2000);
    });

    it('should handle A/B testing scenarios correctly', () => {
      const manager = new FeatureFlagManager({
        bitmapTrigramIndex: {
          rolloutPercentage: 30, // 30% rollout
        } as any,
      });
      
      // Test with multiple users to verify distribution
      let enabledCount = 0;
      const totalUsers = 1000;
      
      for (let i = 0; i < totalUsers; i++) {
        if (manager.shouldUseBitmapIndex(1000, `user-${i}`)) {
          enabledCount++;
        }
      }
      
      // Should be approximately 30% (allowing for some variance in hash distribution)
      const percentage = (enabledCount / totalUsers) * 100;
      expect(percentage).toBeGreaterThan(20); // At least 20%
      expect(percentage).toBeLessThan(40); // At most 40%
    });

    it('should maintain feature flag integrity across updates', () => {
      const manager = new FeatureFlagManager();
      
      // Get initial state
      const initial = manager.getFeatures();
      
      // Make multiple updates
      manager.updateFeatures({
        bitmapTrigramIndex: { enabled: false } as any,
      });
      
      manager.updateFeatures({
        experimental: { advancedFST: true } as any,
      });
      
      manager.updateFeatures({
        prefilter: { enabled: false } as any,
      });
      
      const final = manager.getFeatures();
      
      // Updated fields should be changed
      expect(final.bitmapTrigramIndex.enabled).toBe(false);
      expect(final.experimental.advancedFST).toBe(true);
      expect(final.prefilter.enabled).toBe(false);
      
      // Unchanged fields should remain as initial defaults
      expect(final.bitmapTrigramIndex.rolloutPercentage).toBe(initial.bitmapTrigramIndex.rolloutPercentage);
      expect(final.bitmapTrigramIndex.minDocumentThreshold).toBe(initial.bitmapTrigramIndex.minDocumentThreshold);
      expect(final.prefilter.maxCandidatesBeforeFilter).toBe(initial.prefilter.maxCandidatesBeforeFilter);
      expect(final.experimental.semanticRerank).toBe(initial.experimental.semanticRerank);
    });
  });
});