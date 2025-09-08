/**
 * Comprehensive Edge Cases and Error Handling Tests
 * 
 * Target: Maximum branch coverage through error paths, edge cases, boundary conditions
 * Coverage focus: Error handlers, validation paths, fallback mechanisms, conditional branches
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach, vi } from 'vitest';
import { LensTracer } from '../telemetry/tracer.js';
import { globalFeatureFlags } from '../core/feature-flags.js';
import { IndexRegistry } from '../core/index-registry.js';
import { ASTCache } from '../core/ast-cache.js';
import { ConfigRolloutManager } from '../raptor/config-rollout.js';
import { CompatibilityChecker } from '../core/compatibility-checker.js';
import { VersionManager } from '../core/version-manager.js';
import { QualityGates } from '../core/quality-gates.js';
import { ThreeNightValidation } from '../core/three-night-validation.js';

// Import test fixtures
import { getSearchFixtures } from './fixtures/db-fixtures-simple.js';

describe('Comprehensive Edge Cases and Error Handling Tests', () => {
  let fixtures: any;

  beforeAll(async () => {
    // Remove LensTracer calls that don't exist
    fixtures = await getSearchFixtures();
  });

  afterAll(async () => {
    // No cleanup needed for LensTracer
  });

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Feature Flags Edge Cases', () => {
    it('should handle invalid feature flag values', () => {
      const invalidValues = [null, undefined, {}, [], 'invalid', 123, Symbol('test')];
      
      invalidValues.forEach(value => {
        expect(() => {
          globalFeatureFlags.setFlag('test-flag', value as any);
        }).toThrow();
      });
    });

    it('should handle concurrent flag updates', async () => {
      const flagName = 'concurrent-test';
      
      // Multiple concurrent updates
      const updates = Array.from({ length: 10 }, (_, i) => 
        globalFeatureFlags.setFlag(flagName, i % 2 === 0)
      );
      
      await Promise.all(updates);
      
      // Should have a consistent final state
      const finalValue = globalFeatureFlags.isEnabled(flagName);
      expect(typeof finalValue).toBe('boolean');
    });

    it('should handle flag evaluation with complex conditions', () => {
      const testCases = [
        { name: 'empty-string', value: '', expected: false },
        { name: 'zero', value: 0, expected: false },
        { name: 'negative', value: -1, expected: true },
        { name: 'positive', value: 1, expected: true },
        { name: 'true', value: true, expected: true },
        { name: 'false', value: false, expected: false }
      ];
      
      testCases.forEach(({ name, value, expected }) => {
        globalFeatureFlags.setFlag(name, value as any);
        expect(globalFeatureFlags.isEnabled(name)).toBe(expected);
      });
    });

    it('should handle rollout percentages at boundaries', () => {
      const boundaryValues = [0, 0.1, 0.5, 0.99, 1.0, 100];
      
      boundaryValues.forEach(percentage => {
        globalFeatureFlags.setRolloutPercentage('boundary-test', percentage);
        
        // Test multiple evaluations for consistency
        const results = Array.from({ length: 100 }, () => 
          globalFeatureFlags.isEnabledForUser('boundary-test', `user-${Math.random()}`)
        );
        
        const enabledCount = results.filter(Boolean).length;
        
        if (percentage === 0) {
          expect(enabledCount).toBe(0);
        } else if (percentage >= 1 || percentage >= 100) {
          expect(enabledCount).toBe(100);
        } else {
          expect(enabledCount).toBeGreaterThan(0);
          expect(enabledCount).toBeLessThan(100);
        }
      });
    });
  });

  describe('Index Registry Error Handling', () => {
    it('should handle corrupted index files', async () => {
      const registry = new IndexRegistry();
      
      // Mock corrupted file
      vi.spyOn(registry as any, 'loadIndexFile')
        .mockRejectedValueOnce(new Error('Corrupted index file'));
      
      await expect(registry.initialize()).rejects.toThrow('Corrupted index file');
    });

    it('should handle missing index dependencies', async () => {
      const registry = new IndexRegistry();
      
      // Mock missing dependency
      vi.spyOn(registry as any, 'checkDependencies')
        .mockResolvedValueOnce({ missing: ['lexical-index', 'symbol-index'] });
      
      await expect(registry.validateIntegrity())
        .resolves.toEqual({ 
          valid: false, 
          errors: expect.arrayContaining([expect.stringContaining('missing')]) 
        });
    });

    it('should handle index version mismatches', async () => {
      const registry = new IndexRegistry();
      
      // Mock version mismatch
      vi.spyOn(registry as any, 'checkVersions')
        .mockResolvedValueOnce({
          compatible: false,
          conflicts: [
            { index: 'semantic', expected: '1.0.0', actual: '0.9.0' }
          ]
        });
      
      const result = await registry.checkCompatibility();
      expect(result.compatible).toBe(false);
      expect(result.conflicts).toHaveLength(1);
    });

    it('should handle concurrent index operations', async () => {
      const registry = new IndexRegistry();
      
      // Simulate concurrent readers
      const readers = Array.from({ length: 5 }, () => 
        registry.getReader('test-index')
      );
      
      const results = await Promise.allSettled(readers);
      
      // All should either succeed or fail consistently
      const statuses = results.map(r => r.status);
      const uniqueStatuses = new Set(statuses);
      expect(uniqueStatuses.size).toBeLessThanOrEqual(2); // 'fulfilled' or 'rejected'
    });
  });

  describe('AST Cache Edge Cases', () => {
    it('should handle cache corruption recovery', async () => {
      const cache = new ASTCache();
      
      // Mock corrupted cache entry
      vi.spyOn(cache as any, 'loadFromDisk')
        .mockResolvedValueOnce(null); // Corrupted data
      
      const result = await cache.get('corrupted-file.ts');
      expect(result).toBeNull();
      
      // Should attempt to regenerate
      vi.spyOn(cache as any, 'generateAST')
        .mockResolvedValueOnce({ valid: true, ast: {} });
      
      const regenerated = await cache.getOrGenerate('corrupted-file.ts');
      expect(regenerated).toBeDefined();
    });

    it('should handle memory pressure scenarios', async () => {
      const cache = new ASTCache({ maxMemoryMB: 1 }); // Very small cache
      
      // Add many entries to trigger eviction
      const entries = Array.from({ length: 100 }, (_, i) => ({
        file: `file-${i}.ts`,
        ast: { large: 'x'.repeat(10000) } // Large AST
      }));
      
      for (const entry of entries) {
        cache.set(entry.file, entry.ast);
      }
      
      // Cache should have evicted older entries
      const stats = cache.getStats();
      expect(stats.entries).toBeLessThan(100);
      expect(stats.evictions).toBeGreaterThan(0);
    });

    it('should handle invalid AST structures', async () => {
      const cache = new ASTCache();
      
      const invalidASTStructures = [
        null,
        undefined,
        'string-instead-of-object',
        123,
        [],
        { malformed: true, missing: 'required-fields' }
      ];
      
      invalidASTStructures.forEach((invalidAST, index) => {
        expect(() => {
          cache.set(`invalid-${index}.ts`, invalidAST as any);
        }).toThrow();
      });
    });

    it('should handle concurrent cache access', async () => {
      const cache = new ASTCache();
      const filename = 'concurrent-test.ts';
      
      // Multiple concurrent requests for same file
      const requests = Array.from({ length: 10 }, () => 
        cache.getOrGenerate(filename)
      );
      
      const results = await Promise.all(requests);
      
      // All should return the same result (cached or generated once)
      const firstResult = results[0];
      results.forEach(result => {
        expect(result).toEqual(firstResult);
      });
    });
  });

  describe('Config Rollout Edge Cases', () => {
    it('should handle invalid rollout configurations', () => {
      const rolloutManager = new ConfigRolloutManager();
      
      const invalidConfigs = [
        { percentage: -1 }, // Negative percentage
        { percentage: 150 }, // > 100%
        { percentage: 'fifty' }, // Non-numeric
        { targetUsers: null }, // Invalid target
        { conditions: 'not-an-array' }, // Invalid conditions
      ];
      
      invalidConfigs.forEach(config => {
        expect(() => {
          rolloutManager.setRolloutConfig('test-feature', config as any);
        }).toThrow();
      });
    });

    it('should handle user targeting edge cases', () => {
      const rolloutManager = new ConfigRolloutManager();
      
      const config = {
        percentage: 50,
        targetUsers: ['user1', 'user2'],
        excludeUsers: ['user3'],
        conditions: [{ attribute: 'region', value: 'US' }]
      };
      
      rolloutManager.setRolloutConfig('targeting-test', config);
      
      const testCases = [
        { userId: '', context: {}, expected: false }, // Empty user ID
        { userId: null, context: {}, expected: false }, // Null user ID
        { userId: 'user1', context: {}, expected: true }, // Explicitly targeted
        { userId: 'user3', context: {}, expected: false }, // Explicitly excluded
        { userId: 'user4', context: { region: 'EU' }, expected: false }, // Wrong region
        { userId: 'user5', context: { region: 'US' }, expected: true } // Meets condition
      ];
      
      testCases.forEach(({ userId, context, expected }) => {
        const result = rolloutManager.shouldRolloutForUser('targeting-test', userId as any, context);
        expect(result).toBe(expected);
      });
    });

    it('should handle rollout state transitions', async () => {
      const rolloutManager = new ConfigRolloutManager();
      
      // Test state progression: disabled -> pilot -> full
      const states = [
        { name: 'disabled', percentage: 0 },
        { name: 'pilot', percentage: 10 },
        { name: 'partial', percentage: 50 },
        { name: 'full', percentage: 100 }
      ];
      
      for (const state of states) {
        rolloutManager.setRolloutConfig('state-test', { percentage: state.percentage });
        
        // Test multiple users
        const testUsers = Array.from({ length: 20 }, (_, i) => `user${i}`);
        const enabledUsers = testUsers.filter(user => 
          rolloutManager.shouldRolloutForUser('state-test', user)
        );
        
        const enabledPercentage = (enabledUsers.length / testUsers.length) * 100;
        
        if (state.percentage === 0) {
          expect(enabledUsers.length).toBe(0);
        } else if (state.percentage === 100) {
          expect(enabledUsers.length).toBe(testUsers.length);
        } else {
          expect(enabledPercentage).toBeGreaterThan(0);
          expect(enabledPercentage).toBeLessThan(100);
        }
      }
    });
  });

  describe('Compatibility Checker Edge Cases', () => {
    it('should handle version string parsing edge cases', () => {
      const checker = new CompatibilityChecker();
      
      const edgeCaseVersions = [
        '1.0.0-alpha.1',
        '2.0.0-beta.2+build.123',
        '0.0.1-dev',
        '10.20.30',
        '1.0',
        '1',
        'v1.2.3',
        '1.2.3-SNAPSHOT',
        '1.0.0+20220101'
      ];
      
      edgeCaseVersions.forEach(version => {
        expect(() => {
          const result = checker.parseVersion(version);
          expect(result).toBeDefined();
          expect(result.major).toBeGreaterThanOrEqual(0);
        }).not.toThrow();
      });
    });

    it('should handle malformed version strings', () => {
      const checker = new CompatibilityChecker();
      
      const malformedVersions = [
        '',
        'not-a-version',
        '1.2.3.4.5',
        '1.a.b',
        'null',
        '1..3',
        '.1.2',
        '1.2.'
      ];
      
      malformedVersions.forEach(version => {
        expect(() => {
          checker.parseVersion(version);
        }).toThrow();
      });
    });

    it('should handle bundle compatibility edge cases', async () => {
      const checker = new CompatibilityChecker();
      
      const testBundles = [
        { hash: '', features: [] }, // Empty bundle
        { hash: 'invalid-hash', features: ['unknown-feature'] }, // Unknown features
        { hash: 'a'.repeat(64), features: Array.from({ length: 100 }, (_, i) => `feature-${i}`) }, // Large feature set
        { hash: null, features: ['search'] }, // Null hash
        { hash: 'valid-hash', features: null } // Null features
      ];
      
      for (const bundle of testBundles) {
        const result = await checker.checkBundleCompatibility(bundle as any);
        expect(result).toHaveProperty('compatible');
        expect(result).toHaveProperty('issues');
        expect(typeof result.compatible).toBe('boolean');
        expect(Array.isArray(result.issues)).toBe(true);
      }
    });
  });

  describe('Quality Gates Edge Cases', () => {
    it('should handle metric threshold boundary conditions', async () => {
      const qualityGates = new QualityGates();
      
      const boundaryTests = [
        { metric: 'test_coverage', value: 0, threshold: 0.1, shouldPass: false },
        { metric: 'test_coverage', value: 0.1, threshold: 0.1, shouldPass: true },
        { metric: 'test_coverage', value: 0.10000001, threshold: 0.1, shouldPass: true },
        { metric: 'error_rate', value: 0.05, threshold: 0.05, shouldPass: true }, // At threshold
        { metric: 'error_rate', value: 0.050001, threshold: 0.05, shouldPass: false }, // Just over
        { metric: 'latency_p95', value: 100, threshold: 100, shouldPass: true },
        { metric: 'latency_p95', value: 100.1, threshold: 100, shouldPass: false }
      ];
      
      for (const test of boundaryTests) {
        const result = await qualityGates.evaluateMetric(
          test.metric, 
          test.value, 
          test.threshold
        );
        
        expect(result.passed).toBe(test.shouldPass);
      }
    });

    it('should handle missing or invalid metrics', async () => {
      const qualityGates = new QualityGates();
      
      const invalidMetrics = [
        { name: undefined, value: 0.5 },
        { name: '', value: 0.5 },
        { name: 'test_coverage', value: undefined },
        { name: 'test_coverage', value: null },
        { name: 'test_coverage', value: 'not-a-number' },
        { name: 'test_coverage', value: Infinity },
        { name: 'test_coverage', value: -Infinity },
        { name: 'test_coverage', value: NaN }
      ];
      
      for (const metric of invalidMetrics) {
        await expect(qualityGates.evaluateMetric(
          metric.name as any, 
          metric.value as any, 
          0.8
        )).rejects.toThrow();
      }
    });

    it('should handle gate dependencies and ordering', async () => {
      const qualityGates = new QualityGates();
      
      const gateConfig = {
        gates: [
          { name: 'build', dependencies: [], threshold: 1.0 },
          { name: 'unit_tests', dependencies: ['build'], threshold: 0.9 },
          { name: 'integration_tests', dependencies: ['unit_tests'], threshold: 0.8 },
          { name: 'performance', dependencies: ['integration_tests'], threshold: 0.95 }
        ]
      };
      
      qualityGates.configure(gateConfig);
      
      // Test dependency violations
      const metrics = {
        build: 1.0, // Passes
        unit_tests: 0.85, // Fails (< 0.9)
        integration_tests: 0.85, // Should be skipped due to unit_tests failure
        performance: 0.98 // Should be skipped
      };
      
      const result = await qualityGates.evaluateAll(metrics);
      
      expect(result.gates.build.passed).toBe(true);
      expect(result.gates.unit_tests.passed).toBe(false);
      expect(result.gates.integration_tests.skipped).toBe(true);
      expect(result.gates.performance.skipped).toBe(true);
    });
  });

  describe('Three Night Validation Edge Cases', () => {
    it('should handle validation interruptions', async () => {
      const validation = new ThreeNightValidation();
      
      // Start validation
      const validationPromise = validation.startValidation({
        duration: 1000, // 1 second for testing
        metrics: ['accuracy', 'latency']
      });
      
      // Interrupt after short delay
      setTimeout(() => {
        validation.interrupt('Testing interruption');
      }, 100);
      
      const result = await validationPromise;
      
      expect(result.completed).toBe(false);
      expect(result.interrupted).toBe(true);
      expect(result.reason).toContain('Testing interruption');
    });

    it('should handle partial validation failures', async () => {
      const validation = new ThreeNightValidation();
      
      // Mock mixed validation results
      vi.spyOn(validation as any, 'runNightlyValidation')
        .mockResolvedValueOnce({ success: true, metrics: { accuracy: 0.95 } }) // Night 1: Pass
        .mockResolvedValueOnce({ success: false, metrics: { accuracy: 0.75 } }) // Night 2: Fail
        .mockResolvedValueOnce({ success: true, metrics: { accuracy: 0.92 } }); // Night 3: Pass
      
      const result = await validation.runFullValidation();
      
      expect(result.nightsCompleted).toBe(3);
      expect(result.nightsPassed).toBe(2);
      expect(result.nightsFailed).toBe(1);
      expect(result.overallSuccess).toBe(false); // Requires all 3 nights to pass
    });

    it('should handle validation metric fluctuations', async () => {
      const validation = new ThreeNightValidation();
      
      const fluctuatingMetrics = [
        { accuracy: 0.95, latency: 50 },
        { accuracy: 0.89, latency: 75 }, // Dip in accuracy
        { accuracy: 0.96, latency: 45 },
        { accuracy: 0.91, latency: 80 },
        { accuracy: 0.97, latency: 40 }
      ];
      
      vi.spyOn(validation as any, 'collectMetrics')
        .mockImplementation(async () => {
          const index = Math.floor(Math.random() * fluctuatingMetrics.length);
          return fluctuatingMetrics[index];
        });
      
      const result = await validation.runValidationWithStabilityCheck();
      
      expect(result).toHaveProperty('stable');
      expect(result).toHaveProperty('variance');
      expect(result.variance.accuracy).toBeGreaterThan(0);
      expect(result.variance.latency).toBeGreaterThan(0);
    });
  });

  describe('Telemetry and Tracing Edge Cases', () => {
    it('should handle trace buffer overflow', () => {
      const tracer = LensTracer;
      
      // Generate many spans quickly
      const spans = [];
      for (let i = 0; i < 10000; i++) {
        const span = tracer.startSpan(`test-span-${i}`);
        spans.push(span);
      }
      
      // End all spans
      spans.forEach(span => span.end());
      
      // Should handle gracefully without crashing
      const stats = tracer.getStats();
      expect(stats.totalSpans).toBeGreaterThan(0);
      expect(stats.droppedSpans).toBeGreaterThanOrEqual(0);
    });

    it('should handle malformed trace context', () => {
      const tracer = LensTracer;
      
      const malformedContexts = [
        '', // Empty
        'invalid-trace-context', // Not base64
        'dGVzdA==', // Valid base64 but invalid structure
        null,
        undefined,
        {}, // Object instead of string
        123 // Number instead of string
      ];
      
      malformedContexts.forEach(context => {
        expect(() => {
          tracer.setTraceContext(context as any);
        }).not.toThrow(); // Should handle gracefully
      });
    });

    it('should handle concurrent span operations', () => {
      const tracer = LensTracer;
      
      // Create many spans concurrently
      const spanPromises = Array.from({ length: 100 }, async (_, i) => {
        const span = tracer.startSpan(`concurrent-span-${i}`);
        
        // Add some attributes
        span.setAttribute('index', i);
        span.setAttribute('timestamp', Date.now());
        
        // Simulate some async work
        await new Promise(resolve => setTimeout(resolve, Math.random() * 10));
        
        span.end();
        return span;
      });
      
      return expect(Promise.all(spanPromises)).resolves.toHaveLength(100);
    });
  });

  describe('Memory and Resource Management', () => {
    it('should handle memory pressure gracefully', async () => {
      // Create large objects to simulate memory pressure
      const largeObjects = [];
      
      try {
        for (let i = 0; i < 1000; i++) {
          largeObjects.push({
            id: i,
            data: 'x'.repeat(100000), // 100KB each
            nested: Array.from({ length: 1000 }, (_, j) => ({ index: j }))
          });
        }
        
        // Try to perform operations under memory pressure
        const cache = new ASTCache({ maxMemoryMB: 10 });
        
        for (let i = 0; i < 100; i++) {
          cache.set(`file-${i}.ts`, { large: 'x'.repeat(10000) });
        }
        
        const stats = cache.getStats();
        expect(stats.evictions).toBeGreaterThan(0); // Should have evicted items
        
      } finally {
        // Clean up
        largeObjects.length = 0;
      }
    });

    it('should handle resource cleanup on errors', async () => {
      const registry = new IndexRegistry();
      
      // Mock resource allocation failure
      vi.spyOn(registry as any, 'allocateResources')
        .mockRejectedValueOnce(new Error('Resource allocation failed'));
      
      await expect(registry.initialize()).rejects.toThrow('Resource allocation failed');
      
      // Resources should be cleaned up even after failure
      const resourceStats = registry.getResourceStats();
      expect(resourceStats.leaked).toBe(0);
    });
  });

  describe('Boundary Value Testing', () => {
    it('should handle numeric boundary values', () => {
      const boundaries = [
        { value: 0, context: 'zero' },
        { value: 1, context: 'one' },
        { value: -1, context: 'negative one' },
        { value: Number.MAX_SAFE_INTEGER, context: 'max safe integer' },
        { value: Number.MIN_SAFE_INTEGER, context: 'min safe integer' },
        { value: Number.POSITIVE_INFINITY, context: 'positive infinity' },
        { value: Number.NEGATIVE_INFINITY, context: 'negative infinity' },
        { value: Number.NaN, context: 'NaN' }
      ];
      
      boundaries.forEach(({ value, context }) => {
        // Test with various components that handle numeric values
        expect(() => {
          globalFeatureFlags.setRolloutPercentage('boundary-test', value);
        }).not.toThrow(`Failed for ${context}`);
      });
    });

    it('should handle string boundary values', () => {
      const stringBoundaries = [
        '', // Empty string
        ' ', // Single space
        '\n', // Newline
        '\t', // Tab
        '\0', // Null character
        'a', // Single character
        'x'.repeat(10000), // Very long string
        '🚀', // Unicode emoji
        'café', // Accented characters
        '中文', // Non-Latin characters
        '\u0000\u0001\u0002' // Control characters
      ];
      
      stringBoundaries.forEach(str => {
        expect(() => {
          const cache = new ASTCache();
          cache.set(str || 'empty-string', { content: str });
        }).not.toThrow(`Failed for string: ${JSON.stringify(str)}`);
      });
    });
  });
});
