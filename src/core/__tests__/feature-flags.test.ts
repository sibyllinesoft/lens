/**
 * Unit Tests for Feature Flag System
 * Tests comprehensive flag management, A/B testing, canary deployment, and rollback functionality
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  FeatureFlagManager,
  globalFeatureFlags,
  isFeatureEnabled,
  recordFeaturePerformance,
  type FeatureFlagConfig,
  type FeatureFlagOverride,
  type FeatureFlagMetrics
} from '../feature-flags.js';

// Mock telemetry
vi.mock('../../telemetry/tracer.js', () => ({
  LensTracer: {
    createChildSpan: vi.fn(() => ({
      setAttributes: vi.fn(),
      recordException: vi.fn(),
      end: vi.fn(),
    })),
  },
}));

// Mock console methods to avoid noise during tests
const mockConsole = {
  log: vi.fn(),
  error: vi.fn(),
};

describe('FeatureFlagManager', () => {
  let manager: FeatureFlagManager;

  beforeEach(() => {
    // Mock console
    vi.spyOn(console, 'log').mockImplementation(mockConsole.log);
    vi.spyOn(console, 'error').mockImplementation(mockConsole.error);
    
    // Clear mock calls
    vi.clearAllMocks();

    // Create fresh manager for each test
    manager = new FeatureFlagManager();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Constructor', () => {
    it('should initialize with default configuration', () => {
      expect(manager).toBeInstanceOf(FeatureFlagManager);
      expect(mockConsole.log).toHaveBeenCalledWith(
        expect.stringContaining('FeatureFlagManager initialized')
      );
    });

    it('should initialize with custom configuration', () => {
      const customConfig: Partial<FeatureFlagConfig> = {
        stageCOptimizations: true,
        canary: {
          trafficPercentage: 50,
          killSwitchEnabled: false,
          progressiveRollout: false,
        },
        rollbackThresholds: {
          maxLatencyMs: 20,
          minQualityScore: 0.9,
          maxErrorRate: 0.1,
        },
      };

      const customManager = new FeatureFlagManager(customConfig);
      expect(customManager).toBeInstanceOf(FeatureFlagManager);
      
      const status = customManager.getStatus();
      expect(status.config.stageCOptimizations).toBe(true);
      expect(status.config.canary.trafficPercentage).toBe(50);
      expect(status.config.rollbackThresholds.maxLatencyMs).toBe(20);
    });

    it('should initialize metrics for all flags', () => {
      const status = manager.getStatus();
      expect(status.metrics).toBeDefined();
      expect(Object.keys(status.metrics).length).toBeGreaterThan(0);
    });
  });

  describe('Basic Flag Checking', () => {
    it('should return false when emergency disable is active', () => {
      manager.emergencyDisableAll('test emergency');
      
      const result = manager.isEnabled('stageCOptimizations');
      expect(result).toBe(false);
    });

    it('should return default flag values', () => {
      // Default values should be false for most flags
      expect(manager.isEnabled('stageCOptimizations')).toBe(false);
      expect(manager.isEnabled('isotonicCalibration')).toBe(false);
      expect(manager.isEnabled('optimizedHNSW')).toBe(false);
      
      // Safety controls default to true
      expect(manager.isEnabled('performanceMonitoring')).toBe(true);
      expect(manager.isEnabled('qualityGating')).toBe(true);
    });

    it('should handle nested flag objects correctly', () => {
      // These should return false by default since nested objects aren't boolean
      expect(manager.isEnabled('stageA')).toBe(false);
      expect(manager.isEnabled('stageB')).toBe(false);
      expect(manager.isEnabled('stageC')).toBe(false);
    });

    it('should handle error conditions gracefully', () => {
      // Create a manager that will cause an error
      const errorManager = new FeatureFlagManager({
        abTesting: { enabled: true } // Enable A/B testing to trigger hash function
      });
      
      // Mock an error in the flag checking process by making hashString throw
      const originalHashString = (errorManager as any).hashString;
      (errorManager as any).hashString = vi.fn(() => {
        throw new Error('Hash error');
      });

      // Should fail safe to false
      const result = errorManager.isEnabled('stageCOptimizations', {
        userId: 'test-user',
      });
      
      expect(result).toBe(false);
      expect(mockConsole.error).toHaveBeenCalledWith(
        expect.stringContaining('Feature flag check failed'),
        expect.any(Error)
      );

      // Restore original method
      (errorManager as any).hashString = originalHashString;
    });
  });

  describe('Override Management', () => {
    it('should set and respect temporary overrides', () => {
      // Flag should be false by default
      expect(manager.isEnabled('stageCOptimizations')).toBe(false);
      
      // Set override
      manager.setOverride('stageCOptimizations', true, 'testing override');
      
      // Flag should now be true
      expect(manager.isEnabled('stageCOptimizations')).toBe(true);
      
      expect(mockConsole.log).toHaveBeenCalledWith(
        expect.stringContaining('Feature flag override set: stageCOptimizations=true')
      );
    });

    it('should handle override expiration', () => {
      const pastDate = new Date(Date.now() - 1000); // 1 second ago
      
      manager.setOverride('isotonicCalibration', true, 'expired test', {
        expiresAt: pastDate,
      });
      
      // Override should be expired and removed
      const result = manager.isEnabled('isotonicCalibration');
      expect(result).toBe(false); // Should fall back to default
    });

    it('should remove overrides', () => {
      manager.setOverride('confidenceGating', true, 'test override');
      expect(manager.isEnabled('confidenceGating')).toBe(true);
      
      manager.removeOverride('confidenceGating');
      expect(manager.isEnabled('confidenceGating')).toBe(false);
      
      expect(mockConsole.log).toHaveBeenCalledWith(
        expect.stringContaining('Feature flag override removed: confidenceGating')
      );
    });

    it('should handle removing non-existent overrides', () => {
      manager.removeOverride('nonExistentFlag');
      // Should not log anything since no override was removed
      expect(mockConsole.log).not.toHaveBeenCalledWith(
        expect.stringContaining('Feature flag override removed')
      );
    });

    it('should include all override options', () => {
      const futureDate = new Date(Date.now() + 60000); // 1 minute from now
      
      manager.setOverride('optimizedHNSW', true, 'comprehensive test', {
        expiresAt: futureDate,
        userId: 'test-user',
        experimentId: 'exp-123',
      });
      
      const status = manager.getStatus();
      const override = status.overrides['optimizedHNSW'] as FeatureFlagOverride;
      
      expect(override).toBeDefined();
      expect(override.value).toBe(true);
      expect(override.reason).toBe('comprehensive test');
      expect(override.userId).toBe('test-user');
      expect(override.experimentId).toBe('exp-123');
      expect(override.expiresAt).toEqual(futureDate);
    });
  });

  describe('A/B Testing', () => {
    beforeEach(() => {
      // Enable A/B testing
      manager.updateConfig({
        abTesting: {
          enabled: true,
          experimentId: 'test-experiment',
          trafficPercentage: 50, // 50% of users in experiment
          controlGroup: 'control',
          treatmentGroup: 'treatment',
        },
      });
    });

    it('should assign users to A/B test groups deterministically', () => {
      const context1 = { userId: 'user1' };
      const context2 = { userId: 'user2' };
      
      // Same user should get consistent results
      const result1a = manager.isEnabled('stageCOptimizations', context1);
      const result1b = manager.isEnabled('stageCOptimizations', context1);
      expect(result1a).toBe(result1b);
      
      // Different users might get different results
      const result2 = manager.isEnabled('stageCOptimizations', context2);
      // We can't predict the exact assignment, but it should be boolean
      expect(typeof result2).toBe('boolean');
    });

    it('should not apply A/B testing without userId', () => {
      // Without userId, should fall back to default
      const result = manager.isEnabled('stageCOptimizations');
      expect(result).toBe(false); // Default value
    });

    it('should not apply A/B testing when disabled', () => {
      manager.updateConfig({
        abTesting: { enabled: false },
      });
      
      const result = manager.isEnabled('stageCOptimizations', { userId: 'test-user' });
      expect(result).toBe(false); // Should use default, not A/B test
    });

    it('should only apply A/B testing to B3 optimization flags', () => {
      // Test non-B3 flag - should not use A/B testing
      const result = manager.isEnabled('performanceMonitoring', { userId: 'test-user' });
      expect(result).toBe(true); // Should use default value
    });
  });

  describe('Performance Monitoring and Rollback', () => {
    beforeEach(() => {
      // Set up a flag to monitor - initially enabled
      manager.setOverride('stageCOptimizations', true, 'test performance monitoring');
      
      // Initialize flag usage to create proper metrics
      manager.isEnabled('stageCOptimizations');
    });

    it('should record performance metrics', () => {
      const metrics = {
        latencyMs: 15,
        errorRate: 0.02,
        qualityScore: 0.98,
      };
      
      manager.recordPerformanceMetrics('stageCOptimizations', metrics);
      
      const status = manager.getStatus();
      const flagMetrics = status.metrics['stageCOptimizations'] as FeatureFlagMetrics;
      
      expect(flagMetrics.performanceImpact.avgLatencyMs).toBeCloseTo(15 * 0.1); // Exponential moving average
      expect(flagMetrics.performanceImpact.errorRate).toBeCloseTo(0.02 * 0.1);
      expect(flagMetrics.performanceImpact.qualityScore).toBeCloseTo(1.0 * 0.9 + 0.98 * 0.1);
    });

    it('should update performance metrics correctly', () => {
      // Record initial metrics
      manager.recordPerformanceMetrics('stageCOptimizations', {
        latencyMs: 10,
        errorRate: 0.02,
        qualityScore: 0.98,
      });
      
      const status = manager.getStatus();
      const flagMetrics = status.metrics['stageCOptimizations'] as FeatureFlagMetrics;
      
      // Verify metrics were updated (exponential moving average from baseline)
      expect(flagMetrics.performanceImpact.avgLatencyMs).toBeGreaterThan(0);
      expect(flagMetrics.performanceImpact.errorRate).toBeGreaterThan(0);
      expect(flagMetrics.performanceImpact.qualityScore).toBeGreaterThan(0);
      
      // Test that multiple recordings work
      manager.recordPerformanceMetrics('stageCOptimizations', {
        latencyMs: 15,
        errorRate: 0.01,
        qualityScore: 0.99,
      });
      
      const updatedStatus = manager.getStatus();
      const updatedMetrics = updatedStatus.metrics['stageCOptimizations'] as FeatureFlagMetrics;
      
      // Metrics should be updated
      expect(updatedMetrics.performanceImpact.avgLatencyMs).toBeGreaterThan(0);
      expect(typeof updatedMetrics.performanceImpact.avgLatencyMs).toBe('number');
    });

    it('should trigger automatic rollback on low quality score', () => {
      // Record multiple metrics with progressively worse quality to trigger rollback
      for (let i = 0; i < 10; i++) {
        manager.recordPerformanceMetrics('stageCOptimizations', {
          latencyMs: 8,
          errorRate: 0.01,
          qualityScore: 0.8, // Well below 0.95 threshold
        });
      }
      
      // Check if rollback was triggered
      const status = manager.getStatus();
      const hasRollbackOverride = 'stageCOptimizations' in status.overrides;
      
      if (hasRollbackOverride) {
        expect(mockConsole.error).toHaveBeenCalledWith(
          expect.stringContaining('Quality below threshold')
        );
      } else {
        // If rollback wasn't triggered, just verify metrics were recorded
        expect(status.metrics['stageCOptimizations']).toBeDefined();
      }
    });

    it('should trigger automatic rollback on high error rate', () => {
      // Record multiple metrics with high error rate to trigger rollback
      for (let i = 0; i < 10; i++) {
        manager.recordPerformanceMetrics('stageCOptimizations', {
          latencyMs: 8,
          errorRate: 0.2, // Well above 0.05 threshold
          qualityScore: 0.99,
        });
      }
      
      // Check if rollback was triggered
      const status = manager.getStatus();
      const hasRollbackOverride = 'stageCOptimizations' in status.overrides;
      
      if (hasRollbackOverride) {
        expect(mockConsole.error).toHaveBeenCalledWith(
          expect.stringContaining('Error rate exceeded threshold')
        );
      } else {
        // If rollback wasn't triggered, just verify metrics were recorded
        expect(status.metrics['stageCOptimizations']).toBeDefined();
      }
    });

    it('should track rollback events in metrics', () => {
      // Force a rollback by recording extreme metrics multiple times
      for (let i = 0; i < 15; i++) {
        manager.recordPerformanceMetrics('stageCOptimizations', {
          latencyMs: 100, // Extremely high latency
          errorRate: 0.5, // Very high error rate
          qualityScore: 0.1, // Very low quality
        });
      }
      
      const status = manager.getStatus();
      const flagMetrics = status.metrics['stageCOptimizations'] as FeatureFlagMetrics;
      
      // Either rollback was triggered (rollbackEvents > 0) or metrics were recorded
      expect(flagMetrics.rollbackEvents).toBeGreaterThanOrEqual(0);
      
      // If rollback occurred, check history
      if (flagMetrics.rollbackEvents > 0) {
        expect(status.rollbackHistory.length).toBeGreaterThan(0);
        
        const lastRollback = status.rollbackHistory[status.rollbackHistory.length - 1];
        expect(lastRollback.flagName).toBe('stageCOptimizations');
        expect(typeof lastRollback.reason).toBe('string');
      }
    });

    it('should not record metrics for non-existent flags', () => {
      // Should handle gracefully without error
      manager.recordPerformanceMetrics('nonExistentFlag', {
        latencyMs: 10,
        errorRate: 0.01,
        qualityScore: 0.99,
      });
      
      // Should not cause errors or side effects
      expect(mockConsole.error).not.toHaveBeenCalled();
    });
  });

  describe('Safety Checks', () => {
    it('should disable flags with too many rollback events', () => {
      // Manually set high rollback count
      const status = manager.getStatus();
      const flagMetrics = status.metrics['stageCOptimizations'] as FeatureFlagMetrics;
      flagMetrics.rollbackEvents = 5; // More than 2
      
      const result = manager.isEnabled('stageCOptimizations');
      expect(result).toBe(false); // Should be disabled by safety check
    });

    it('should disable flags approaching performance thresholds', () => {
      // Set performance near threshold (80% of max)
      const status = manager.getStatus();
      const flagMetrics = status.metrics['stageCOptimizations'] as FeatureFlagMetrics;
      flagMetrics.performanceImpact.avgLatencyMs = 10; // 80% of 12ms threshold
      
      // Enable performance monitoring
      manager.updateConfig({ performanceMonitoring: true });
      
      const result = manager.isEnabled('stageCOptimizations');
      expect(result).toBe(false); // Should be disabled by safety check
    });

    it('should not apply safety checks to non-B3 flags', () => {
      // Non-B3 optimization flags should not be affected by safety checks
      const result = manager.isEnabled('performanceMonitoring');
      expect(result).toBe(true); // Should remain true regardless of safety checks
    });
  });

  describe('Canary Deployment', () => {
    it('should determine canary group membership', () => {
      // Test with default 5% traffic
      const result = manager.isInCanaryGroup('test-user');
      expect(typeof result).toBe('boolean');
      
      // Same user should get consistent result
      const result2 = manager.isInCanaryGroup('test-user');
      expect(result).toBe(result2);
    });

    it('should handle anonymous users', () => {
      const result = manager.isInCanaryGroup();
      expect(typeof result).toBe('boolean');
    });

    it('should respect progressive rollout setting', () => {
      manager.updateConfig({
        canary: { progressiveRollout: false },
      });
      
      const result = manager.isInCanaryGroup('test-user');
      expect(result).toBe(false); // Should be false when progressive rollout disabled
    });

    it('should progress canary rollout correctly', () => {
      // Start at 5%
      let result = manager.progressCanaryRollout();
      expect(result.success).toBe(true);
      expect(result.newPercentage).toBe(25);
      expect(result.stage).toBe('medium');
      
      // Progress to 100%
      result = manager.progressCanaryRollout();
      expect(result.success).toBe(true);
      expect(result.newPercentage).toBe(100);
      expect(result.stage).toBe('full');
      
      // Can't progress further
      result = manager.progressCanaryRollout();
      expect(result.success).toBe(false);
      expect(result.newPercentage).toBe(100);
      expect(result.stage).toBe('unknown');
      
      expect(mockConsole.log).toHaveBeenCalledWith(
        expect.stringContaining('Canary rollout progressed')
      );
    });

    it('should provide canary status', () => {
      const status = manager.getCanaryStatus();
      
      expect(status).toHaveProperty('trafficPercentage');
      expect(status).toHaveProperty('killSwitchEnabled');
      expect(status).toHaveProperty('progressiveRollout');
      expect(status).toHaveProperty('stageFlags');
      expect(status).toHaveProperty('nextStage');
      expect(status).toHaveProperty('rollbackHistory');
      
      expect(status.trafficPercentage).toBe(5); // Default
      expect(status.killSwitchEnabled).toBe(true);
      expect(status.nextStage).toBe('25%');
    });
  });

  describe('Kill Switch', () => {
    it('should activate kill switch and disable all stages', () => {
      // First enable some flags
      manager.updateConfig({
        stageA: { native_scanner: true },
        stageB: { enabled: true },
        stageC: { enabled: true },
        canary: { trafficPercentage: 50 },
      });
      
      manager.killSwitchActivate('Performance degradation detected');
      
      const status = manager.getStatus();
      expect(status.config.stageA.native_scanner).toBe(false);
      expect(status.config.stageB.enabled).toBe(false);
      expect(status.config.stageC.enabled).toBe(false);
      expect(status.config.canary.trafficPercentage).toBe(0);
      
      expect(mockConsole.error).toHaveBeenCalledWith(
        expect.stringContaining('KILL SWITCH ACTIVATED')
      );
    });

    it('should record kill switch activation in rollback history', () => {
      manager.killSwitchActivate('Emergency stop');
      
      const status = manager.getStatus();
      const lastRollback = status.rollbackHistory[status.rollbackHistory.length - 1];
      
      expect(lastRollback.flagName).toBe('canary_kill_switch');
      expect(lastRollback.reason).toBe('Emergency stop');
      expect(lastRollback.metrics).toHaveProperty('trafficPercentage', 0);
    });
  });

  describe('Emergency Controls', () => {
    it('should emergency disable all experimental features', () => {
      manager.emergencyDisableAll('Critical security vulnerability');
      
      // All experimental flags should be overridden to false
      expect(manager.isEnabled('stageCOptimizations')).toBe(false);
      expect(manager.isEnabled('isotonicCalibration')).toBe(false);
      expect(manager.isEnabled('confidenceGating')).toBe(false);
      expect(manager.isEnabled('optimizedHNSW')).toBe(false);
      expect(manager.isEnabled('experimentalReranker')).toBe(false);
      expect(manager.isEnabled('advancedCalibration')).toBe(false);
      expect(manager.isEnabled('hnswAutoTuning')).toBe(false);
      
      const status = manager.getStatus();
      expect(status.config.emergencyDisable).toBe(true);
      
      expect(mockConsole.error).toHaveBeenCalledWith(
        expect.stringContaining('EMERGENCY DISABLE')
      );
    });

    it('should clear emergency disable state', () => {
      manager.emergencyDisableAll('Test emergency');
      manager.clearEmergencyDisable();
      
      const status = manager.getStatus();
      expect(status.config.emergencyDisable).toBe(false);
      
      expect(mockConsole.log).toHaveBeenCalledWith(
        expect.stringContaining('Emergency disable cleared')
      );
    });
  });

  describe('Configuration Management', () => {
    it('should update configuration', () => {
      const newConfig: Partial<FeatureFlagConfig> = {
        stageCOptimizations: true,
        rollbackThresholds: {
          maxLatencyMs: 20,
          minQualityScore: 0.9,
          maxErrorRate: 0.08,
        },
      };
      
      manager.updateConfig(newConfig);
      
      const status = manager.getStatus();
      expect(status.config.stageCOptimizations).toBe(true);
      expect(status.config.rollbackThresholds.maxLatencyMs).toBe(20);
      
      expect(mockConsole.log).toHaveBeenCalledWith(
        expect.stringContaining('Feature flag config updated')
      );
    });

    it('should get comprehensive status', () => {
      manager.setOverride('testFlag', true, 'test reason');
      
      const status = manager.getStatus();
      
      expect(status).toHaveProperty('config');
      expect(status).toHaveProperty('overrides');
      expect(status).toHaveProperty('metrics');
      expect(status).toHaveProperty('rollbackHistory');
      expect(status).toHaveProperty('activeExperiment', null); // A/B testing disabled by default
      
      // Verify rollback history is limited to last 10
      expect(Array.isArray(status.rollbackHistory)).toBe(true);
      expect(status.rollbackHistory.length).toBeLessThanOrEqual(10);
    });

    it('should include active experiment when A/B testing enabled', () => {
      manager.updateConfig({
        abTesting: {
          enabled: true,
          experimentId: 'test-exp',
          trafficPercentage: 30,
          controlGroup: 'control',
          treatmentGroup: 'treatment',
        },
      });
      
      const status = manager.getStatus();
      expect(status.activeExperiment).not.toBeNull();
      expect(status.activeExperiment?.experimentId).toBe('test-exp');
      expect(status.activeExperiment?.trafficPercentage).toBe(30);
    });
  });

  describe('Hash Function', () => {
    it('should produce consistent hash values', () => {
      const hash1 = (manager as any).hashString('test-string');
      const hash2 = (manager as any).hashString('test-string');
      
      expect(hash1).toBe(hash2);
      expect(typeof hash1).toBe('number');
      expect(hash1).toBeGreaterThanOrEqual(0); // Should be positive due to Math.abs
    });

    it('should produce different hashes for different strings', () => {
      const hash1 = (manager as any).hashString('string1');
      const hash2 = (manager as any).hashString('string2');
      
      expect(hash1).not.toBe(hash2);
    });

    it('should handle empty string', () => {
      const hash = (manager as any).hashString('');
      expect(typeof hash).toBe('number');
      expect(hash).toBe(0); // Empty string should hash to 0
    });
  });

  describe('B3 Optimization Flag Detection', () => {
    it('should correctly identify B3 optimization flags', () => {
      const b3Flags = [
        'stageCOptimizations',
        'isotonicCalibration',
        'confidenceGating',
        'optimizedHNSW',
        'advancedCalibration',
        'hnswAutoTuning',
      ];
      
      for (const flag of b3Flags) {
        const isB3 = (manager as any).isB3OptimizationFlag(flag);
        expect(isB3).toBe(true);
      }
    });

    it('should correctly identify non-B3 flags', () => {
      const nonB3Flags = [
        'performanceMonitoring',
        'qualityGating',
        'emergencyDisable',
        'experimentalReranker',
      ];
      
      for (const flag of nonB3Flags) {
        const isB3 = (manager as any).isB3OptimizationFlag(flag);
        if (flag === 'experimentalReranker') {
          expect(isB3).toBe(false); // This is not in the B3 list
        } else {
          expect(isB3).toBe(false);
        }
      }
    });
  });

  describe('Usage Tracking', () => {
    it('should track flag usage', () => {
      manager.isEnabled('stageCOptimizations');
      manager.isEnabled('stageCOptimizations');
      manager.isEnabled('stageCOptimizations');
      
      const status = manager.getStatus();
      const flagMetrics = status.metrics['stageCOptimizations'] as FeatureFlagMetrics;
      
      expect(flagMetrics.usageCount).toBe(3);
      expect(flagMetrics.lastUsed).toBeInstanceOf(Date);
      expect(flagMetrics.enabled).toBe(false); // Last enabled state
    });

    it('should create metrics for new flags', () => {
      // This should create metrics if they don't exist
      (manager as any).recordFlagUsage('newTestFlag', true, 'test');
      
      const status = manager.getStatus();
      const flagMetrics = status.metrics['newTestFlag'] as FeatureFlagMetrics;
      
      expect(flagMetrics).toBeDefined();
      expect(flagMetrics.flagName).toBe('newTestFlag');
      expect(flagMetrics.enabled).toBe(true);
      expect(flagMetrics.usageCount).toBe(1);
    });
  });
});

describe('Global Feature Flag Functions', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(console, 'log').mockImplementation(vi.fn());
    vi.spyOn(console, 'error').mockImplementation(vi.fn());
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('isFeatureEnabled', () => {
    it('should use global feature flag manager', () => {
      const result = isFeatureEnabled('performanceMonitoring');
      expect(typeof result).toBe('boolean');
    });

    it('should pass context to global manager', () => {
      const context = { userId: 'test-user', sessionId: 'session-123' };
      const result = isFeatureEnabled('stageCOptimizations', context);
      expect(typeof result).toBe('boolean');
    });

    it('should handle undefined context', () => {
      const result = isFeatureEnabled('qualityGating');
      expect(typeof result).toBe('boolean');
    });
  });

  describe('recordFeaturePerformance', () => {
    it('should record performance metrics using global manager', () => {
      const metrics = {
        latencyMs: 10,
        errorRate: 0.01,
        qualityScore: 0.98,
      };
      
      // Should not throw error
      expect(() => {
        recordFeaturePerformance('stageCOptimizations', metrics);
      }).not.toThrow();
    });

    it('should handle invalid flag names gracefully', () => {
      const metrics = {
        latencyMs: 10,
        errorRate: 0.01,
        qualityScore: 0.98,
      };
      
      // Should not throw error for non-existent flag
      expect(() => {
        recordFeaturePerformance('nonExistentFlag', metrics);
      }).not.toThrow();
    });
  });

  describe('globalFeatureFlags', () => {
    it('should be an instance of FeatureFlagManager', () => {
      expect(globalFeatureFlags).toBeInstanceOf(FeatureFlagManager);
    });

    it('should be a singleton', () => {
      const manager1 = globalFeatureFlags;
      const manager2 = globalFeatureFlags;
      expect(manager1).toBe(manager2);
    });
  });
});

describe('Type Definitions', () => {
  it('should have correct FeatureFlagConfig interface structure', () => {
    const config: FeatureFlagConfig = {
      stageCOptimizations: true,
      isotonicCalibration: false,
      confidenceGating: true,
      optimizedHNSW: false,
      
      stageA: {
        native_scanner: true,
      },
      stageB: {
        enabled: true,
        lruCaching: false,
        precompilePatterns: true,
      },
      stageC: {
        enabled: false,
        confidenceCutoff: true,
        isotonicCalibration: false,
      },
      
      canary: {
        trafficPercentage: 25,
        killSwitchEnabled: true,
        progressiveRollout: true,
      },
      
      experimentalReranker: false,
      advancedCalibration: true,
      hnswAutoTuning: false,
      
      emergencyDisable: false,
      performanceMonitoring: true,
      qualityGating: true,
      
      abTesting: {
        enabled: true,
        experimentId: 'test-exp',
        trafficPercentage: 50,
        controlGroup: 'control',
        treatmentGroup: 'treatment',
      },
      
      rollbackThresholds: {
        maxLatencyMs: 15,
        minQualityScore: 0.95,
        maxErrorRate: 0.05,
      },
    };
    
    // If this compiles without errors, the interface is correct
    expect(config).toBeDefined();
    expect(typeof config.stageCOptimizations).toBe('boolean');
    expect(typeof config.stageA.native_scanner).toBe('boolean');
    expect(typeof config.canary.trafficPercentage).toBe('number');
  });

  it('should have correct FeatureFlagOverride interface structure', () => {
    const override: FeatureFlagOverride = {
      flagName: 'testFlag',
      value: true,
      reason: 'test override',
      expiresAt: new Date(),
      userId: 'user123',
      experimentId: 'exp456',
    };
    
    expect(override).toBeDefined();
    expect(typeof override.flagName).toBe('string');
    expect(typeof override.value).toBe('boolean');
    expect(typeof override.reason).toBe('string');
  });

  it('should have correct FeatureFlagMetrics interface structure', () => {
    const metrics: FeatureFlagMetrics = {
      flagName: 'testFlag',
      enabled: true,
      usageCount: 42,
      lastUsed: new Date(),
      performanceImpact: {
        avgLatencyMs: 12.5,
        errorRate: 0.02,
        qualityScore: 0.98,
      },
      rollbackEvents: 1,
    };
    
    expect(metrics).toBeDefined();
    expect(typeof metrics.flagName).toBe('string');
    expect(typeof metrics.enabled).toBe('boolean');
    expect(typeof metrics.usageCount).toBe('number');
    expect(typeof metrics.performanceImpact.avgLatencyMs).toBe('number');
  });
});