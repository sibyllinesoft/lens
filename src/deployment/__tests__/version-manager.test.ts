import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { VersionManager, type ConfigFingerprint } from '../version-manager.js';
import { existsSync, rmSync } from 'fs';

// Mock fs module
vi.mock('fs', () => ({
  writeFileSync: vi.fn(),
  readFileSync: vi.fn(),
  existsSync: vi.fn(),
  mkdirSync: vi.fn(),
  rmSync: vi.fn()
}));

// Mock crypto module
vi.mock('crypto', () => ({
  createHash: vi.fn(() => ({
    update: vi.fn().mockReturnThis(),
    digest: vi.fn(() => 'mock-hash-123456')
  }))
}));

describe('VersionManager', () => {
  let versionManager: VersionManager;
  let mockConfigFingerprint: ConfigFingerprint;
  const testVersionPath = '/tmp/test_versions';

  beforeEach(() => {
    vi.clearAllMocks();
    
    versionManager = new VersionManager(testVersionPath);

    mockConfigFingerprint = {
      version: '1.2.3',
      timestamp: '2024-09-06T10:00:00Z',
      git_commit: 'abc123def456',
      api_version: '2.1.0',
      index_version: '1.5.2',
      policy_version: '0.8.1',
      api_config: {
        port: 3000,
        timeout: 30000,
        max_connections: 100
      },
      tau_value: 0.85,
      ltr_model_hash: 'model_hash_789',
      dedup_params: {
        k: 50,
        hamming_max: 3,
        keep: 10,
        simhash_bits: 64
      },
      early_exit_config: {
        enabled: true,
        precision_threshold: 0.95,
        recall_threshold: 0.90,
        max_stages: 3
      },
      feature_schema: {
        query_features: ['length', 'complexity', 'semantic_score'],
        document_features: ['relevance', 'popularity', 'freshness'],
        interaction_features: ['click_rate', 'dwell_time']
      },
      reliability_curve: [
        { precision: 0.8, coverage: 0.9 },
        { precision: 0.9, coverage: 0.7 },
        { precision: 0.95, coverage: 0.5 }
      ],
      baseline_metrics: {
        p95_latency_ms: 25,
        throughput_qps: 1000,
        error_rate: 0.001,
        recall_at_10: 0.85,
        ndcg_at_10: 0.78
      },
      promotion_gates: {
        quality_gate: {
          min_recall: 0.80,
          min_precision: 0.75,
          min_ndcg: 0.70
        },
        performance_gate: {
          max_p95_latency: 30,
          min_throughput: 800,
          max_error_rate: 0.005
        },
        stability_gate: {
          min_uptime_percent: 99.5,
          max_memory_mb: 512,
          max_cpu_percent: 70
        }
      },
      drift_thresholds: {
        quality_drift_max: 0.05,
        performance_drift_max: 0.10,
        feature_drift_max: 0.15
      }
    };
  });

  afterEach(() => {
    // Clean up test files if they exist
    if (existsSync(testVersionPath)) {
      rmSync(testVersionPath, { recursive: true, force: true });
    }
  });

  describe('Initialization', () => {
    it('should initialize version manager with default path', () => {
      const defaultManager = new VersionManager();
      expect(defaultManager).toBeDefined();
    });

    it('should initialize version manager with custom path', () => {
      const customManager = new VersionManager('/custom/versions');
      expect(customManager).toBeDefined();
    });

    it('should create version directory if it does not exist', () => {
      const { mkdirSync, existsSync } = require('fs');
      existsSync.mockReturnValue(false);

      new VersionManager('/new/version/path');

      expect(mkdirSync).toHaveBeenCalledWith('/new/version/path', { recursive: true });
    });

    it('should not create directory if it already exists', () => {
      const { mkdirSync, existsSync } = require('fs');
      existsSync.mockReturnValue(true);

      new VersionManager('/existing/version/path');

      expect(mkdirSync).not.toHaveBeenCalled();
    });
  });

  describe('Configuration Fingerprinting', () => {
    it('should create configuration fingerprint', () => {
      const configData = {
        api: { port: 3000 },
        search: { k: 50 },
        features: ['semantic', 'lexical']
      };

      const fingerprint = versionManager.createFingerprint(configData, '2.0.0');

      expect(fingerprint.version).toBe('2.0.0');
      expect(fingerprint.timestamp).toBeDefined();
      expect(fingerprint.api_version).toBeDefined();
      expect(fingerprint.index_version).toBeDefined();
      expect(fingerprint.policy_version).toBeDefined();
    });

    it('should include git commit if provided', () => {
      const configData = { test: true };
      const gitCommit = 'commit_hash_123';

      const fingerprint = versionManager.createFingerprint(configData, '1.0.0', gitCommit);

      expect(fingerprint.git_commit).toBe(gitCommit);
    });

    it('should generate unique fingerprints for different configs', () => {
      const config1 = { setting: 'value1' };
      const config2 = { setting: 'value2' };

      const fingerprint1 = versionManager.createFingerprint(config1, '1.0.0');
      const fingerprint2 = versionManager.createFingerprint(config2, '1.0.0');

      expect(fingerprint1.timestamp).not.toBe(fingerprint2.timestamp);
    });

    it('should validate semantic versioning', () => {
      const config = { test: true };

      expect(() => {
        versionManager.createFingerprint(config, 'invalid-version');
      }).toThrow('Invalid semantic version');

      expect(() => {
        versionManager.createFingerprint(config, '1.0');
      }).toThrow('Invalid semantic version');

      expect(() => {
        versionManager.createFingerprint(config, 'v1.0.0');
      }).toThrow('Invalid semantic version');
    });

    it('should handle complex configuration structures', () => {
      const complexConfig = {
        nested: {
          deeply: {
            embedded: {
              values: [1, 2, 3],
              object: { a: 1, b: 2 }
            }
          }
        },
        array: ['item1', 'item2'],
        boolean: true,
        number: 42
      };

      const fingerprint = versionManager.createFingerprint(complexConfig, '1.0.0');

      expect(fingerprint).toBeDefined();
      expect(fingerprint.version).toBe('1.0.0');
    });
  });

  describe('Version Tagging and Storage', () => {
    it('should tag and freeze configuration version', () => {
      const { writeFileSync } = require('fs');
      
      versionManager.tagVersion(mockConfigFingerprint);

      expect(writeFileSync).toHaveBeenCalledWith(
        expect.stringContaining('1.2.3.json'),
        expect.any(String)
      );
    });

    it('should store version with correct filename format', () => {
      const { writeFileSync } = require('fs');
      
      versionManager.tagVersion(mockConfigFingerprint);

      expect(writeFileSync).toHaveBeenCalledWith(
        expect.stringMatching(/1\.2\.3\.json$/),
        expect.any(String)
      );
    });

    it('should serialize configuration properly', () => {
      const { writeFileSync } = require('fs');
      
      versionManager.tagVersion(mockConfigFingerprint);

      const writtenData = writeFileSync.mock.calls[0][1];
      const parsedData = JSON.parse(writtenData);

      expect(parsedData.version).toBe('1.2.3');
      expect(parsedData.tau_value).toBe(0.85);
      expect(parsedData.dedup_params.k).toBe(50);
    });

    it('should handle version overwriting', () => {
      const { writeFileSync, existsSync } = require('fs');
      existsSync.mockReturnValue(true);
      
      versionManager.tagVersion(mockConfigFingerprint);

      expect(writeFileSync).toHaveBeenCalled();
    });

    it('should validate fingerprint before tagging', () => {
      const invalidFingerprint = {
        // Missing required fields
        version: '1.0.0'
      } as any;

      expect(() => {
        versionManager.tagVersion(invalidFingerprint);
      }).toThrow('Invalid fingerprint');
    });
  });

  describe('Version Loading and Retrieval', () => {
    it('should load existing version configuration', () => {
      const { readFileSync, existsSync } = require('fs');
      existsSync.mockReturnValue(true);
      readFileSync.mockReturnValue(JSON.stringify(mockConfigFingerprint));

      const loaded = versionManager.loadVersion('1.2.3');

      expect(loaded).toEqual(mockConfigFingerprint);
      expect(readFileSync).toHaveBeenCalledWith(
        expect.stringContaining('1.2.3.json'),
        'utf8'
      );
    });

    it('should throw error when loading non-existent version', () => {
      const { existsSync } = require('fs');
      existsSync.mockReturnValue(false);

      expect(() => {
        versionManager.loadVersion('9.9.9');
      }).toThrow('Version 9.9.9 not found');
    });

    it('should handle corrupted version files', () => {
      const { readFileSync, existsSync } = require('fs');
      existsSync.mockReturnValue(true);
      readFileSync.mockReturnValue('invalid json content');

      expect(() => {
        versionManager.loadVersion('1.2.3');
      }).toThrow('Failed to parse version');
    });

    it('should get latest version', () => {
      const versions = ['1.0.0', '1.1.0', '2.0.0', '1.2.3'];
      vi.spyOn(versionManager, 'listVersions').mockReturnValue(versions);

      const latest = versionManager.getLatestVersion();

      expect(latest).toBe('2.0.0');
    });

    it('should handle empty version list', () => {
      vi.spyOn(versionManager, 'listVersions').mockReturnValue([]);

      expect(() => {
        versionManager.getLatestVersion();
      }).toThrow('No versions found');
    });
  });

  describe('Version Comparison and History', () => {
    it('should compare two configuration fingerprints', () => {
      const fingerprint1 = { ...mockConfigFingerprint };
      const fingerprint2 = { 
        ...mockConfigFingerprint, 
        tau_value: 0.90,
        version: '1.2.4'
      };

      const diff = versionManager.compareVersions(fingerprint1, fingerprint2);

      expect(diff.changed_fields).toContain('tau_value');
      expect(diff.changed_fields).toContain('version');
      expect(diff.changes.tau_value).toEqual({ from: 0.85, to: 0.90 });
    });

    it('should detect deep object changes', () => {
      const fingerprint1 = { ...mockConfigFingerprint };
      const fingerprint2 = { 
        ...mockConfigFingerprint,
        dedup_params: {
          ...mockConfigFingerprint.dedup_params,
          k: 75
        }
      };

      const diff = versionManager.compareVersions(fingerprint1, fingerprint2);

      expect(diff.changed_fields).toContain('dedup_params.k');
      expect(diff.changes['dedup_params.k']).toEqual({ from: 50, to: 75 });
    });

    it('should detect array changes', () => {
      const fingerprint1 = { ...mockConfigFingerprint };
      const fingerprint2 = { 
        ...mockConfigFingerprint,
        reliability_curve: [
          { precision: 0.8, coverage: 0.9 },
          { precision: 0.9, coverage: 0.8 } // Changed coverage
        ]
      };

      const diff = versionManager.compareVersions(fingerprint1, fingerprint2);

      expect(diff.changed_fields.length).toBeGreaterThan(0);
      expect(diff.summary).toContain('2 field(s) changed');
    });

    it('should handle identical fingerprints', () => {
      const diff = versionManager.compareVersions(mockConfigFingerprint, mockConfigFingerprint);

      expect(diff.changed_fields).toHaveLength(0);
      expect(diff.summary).toBe('No changes detected');
    });

    it('should get version history', () => {
      const versions = ['1.0.0', '1.1.0', '1.2.0', '1.2.3'];
      vi.spyOn(versionManager, 'listVersions').mockReturnValue(versions);

      const history = versionManager.getVersionHistory();

      expect(history).toEqual(versions.reverse()); // Should be in descending order
    });

    it('should get version history with limit', () => {
      const versions = ['1.0.0', '1.1.0', '1.2.0', '1.2.3', '2.0.0'];
      vi.spyOn(versionManager, 'listVersions').mockReturnValue(versions);

      const history = versionManager.getVersionHistory(3);

      expect(history).toHaveLength(3);
      expect(history[0]).toBe('2.0.0'); // Latest first
    });
  });

  describe('Rollback Capability', () => {
    it('should create rollback plan', () => {
      const targetVersion = '1.1.0';
      const rollbackPlan = versionManager.createRollbackPlan('1.2.3', targetVersion);

      expect(rollbackPlan.from_version).toBe('1.2.3');
      expect(rollbackPlan.to_version).toBe(targetVersion);
      expect(rollbackPlan.steps).toBeDefined();
      expect(rollbackPlan.estimated_downtime_seconds).toBeDefined();
      expect(rollbackPlan.risk_level).toBeDefined();
    });

    it('should calculate rollback risk based on version distance', () => {
      const majorRollback = versionManager.createRollbackPlan('2.0.0', '1.0.0');
      const minorRollback = versionManager.createRollbackPlan('1.2.3', '1.2.0');

      expect(majorRollback.risk_level).toBe('high');
      expect(minorRollback.risk_level).toBe('low');
    });

    it('should validate rollback target exists', () => {
      vi.spyOn(versionManager, 'versionExists').mockReturnValue(false);

      expect(() => {
        versionManager.createRollbackPlan('1.2.3', '9.9.9');
      }).toThrow('Target version 9.9.9 does not exist');
    });

    it('should execute rollback plan', async () => {
      const rollbackPlan = {
        from_version: '1.2.3',
        to_version: '1.1.0',
        steps: [
          { action: 'stop_services', description: 'Stop running services' },
          { action: 'restore_config', description: 'Restore configuration' },
          { action: 'restart_services', description: 'Restart with old config' }
        ],
        estimated_downtime_seconds: 30,
        risk_level: 'medium' as const
      };

      const result = await versionManager.executeRollback(rollbackPlan);

      expect(result.success).toBe(true);
      expect(result.steps_completed).toBe(3);
      expect(result.actual_downtime_seconds).toBeDefined();
    });

    it('should handle rollback execution failures', async () => {
      const failingRollbackPlan = {
        from_version: '1.2.3',
        to_version: '1.1.0',
        steps: [
          { action: 'failing_step', description: 'This step will fail' }
        ],
        estimated_downtime_seconds: 10,
        risk_level: 'low' as const
      };

      // Mock a failing step
      vi.spyOn(versionManager as any, 'executeRollbackStep')
        .mockRejectedValue(new Error('Step failed'));

      const result = await versionManager.executeRollback(failingRollbackPlan);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Step failed');
      expect(result.steps_completed).toBe(0);
    });
  });

  describe('Version Validation and Health Checks', () => {
    it('should validate fingerprint integrity', () => {
      const validationResult = versionManager.validateFingerprint(mockConfigFingerprint);

      expect(validationResult.valid).toBe(true);
      expect(validationResult.errors).toHaveLength(0);
    });

    it('should detect missing required fields', () => {
      const incompleteFingerprint = {
        version: '1.0.0',
        timestamp: '2024-09-06T10:00:00Z'
        // Missing other required fields
      } as any;

      const validationResult = versionManager.validateFingerprint(incompleteFingerprint);

      expect(validationResult.valid).toBe(false);
      expect(validationResult.errors.length).toBeGreaterThan(0);
      expect(validationResult.errors).toContain(
        expect.stringMatching(/missing.*api_version/)
      );
    });

    it('should validate version format', () => {
      const invalidVersionFingerprint = {
        ...mockConfigFingerprint,
        version: 'not-a-version'
      };

      const validationResult = versionManager.validateFingerprint(invalidVersionFingerprint);

      expect(validationResult.valid).toBe(false);
      expect(validationResult.errors).toContain(
        expect.stringMatching(/invalid.*version.*format/)
      );
    });

    it('should validate numeric ranges', () => {
      const invalidRangeFingerprint = {
        ...mockConfigFingerprint,
        tau_value: 1.5, // Invalid: should be 0-1
        baseline_metrics: {
          ...mockConfigFingerprint.baseline_metrics,
          error_rate: -0.1 // Invalid: negative error rate
        }
      };

      const validationResult = versionManager.validateFingerprint(invalidRangeFingerprint);

      expect(validationResult.valid).toBe(false);
      expect(validationResult.errors.length).toBeGreaterThan(0);
    });

    it('should check version compatibility', () => {
      const currentVersion = '1.2.3';
      const targetVersions = ['1.2.4', '1.3.0', '2.0.0', '0.9.0'];

      targetVersions.forEach(targetVersion => {
        const compatibility = versionManager.checkCompatibility(currentVersion, targetVersion);
        expect(compatibility).toBeDefined();
        expect(['compatible', 'warning', 'incompatible']).toContain(compatibility.status);
      });
    });
  });

  describe('Performance and Scalability', () => {
    it('should handle large configuration objects efficiently', () => {
      const largeConfig = {
        ...mockConfigFingerprint,
        large_data: Array.from({ length: 10000 }, (_, i) => ({
          id: i,
          value: `item_${i}`,
          metadata: { created: new Date(), score: Math.random() }
        }))
      };

      const startTime = Date.now();
      const fingerprint = versionManager.createFingerprint(largeConfig, '1.0.0');
      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(100); // Should complete quickly
      expect(fingerprint).toBeDefined();
    });

    it('should manage version storage efficiently', () => {
      // Test with many versions
      for (let i = 0; i < 100; i++) {
        const version = `1.${i}.0`;
        const fingerprint = { ...mockConfigFingerprint, version };
        versionManager.tagVersion(fingerprint);
      }

      const { writeFileSync } = require('fs');
      expect(writeFileSync).toHaveBeenCalledTimes(100);
    });

    it('should handle concurrent version operations', async () => {
      const concurrentOps = Array.from({ length: 10 }, (_, i) =>
        versionManager.createFingerprint({ test: i }, `1.0.${i}`)
      );

      const results = await Promise.all(
        concurrentOps.map(fp => Promise.resolve(fp))
      );

      expect(results).toHaveLength(10);
      results.forEach((result, index) => {
        expect(result.version).toBe(`1.0.${index}`);
      });
    });
  });

  describe('Integration and Configuration Management', () => {
    it('should integrate with deployment system', () => {
      const deploymentConfig = {
        canary_percentage: 10,
        rollout_strategy: 'blue_green',
        health_checks: ['api', 'search', 'metrics']
      };

      const fingerprint = versionManager.createFingerprint(
        deploymentConfig, 
        '1.0.0',
        'deployment_commit_123'
      );

      expect(fingerprint.version).toBe('1.0.0');
      expect(fingerprint.git_commit).toBe('deployment_commit_123');
    });

    it('should support feature flag configurations', () => {
      const featureFlagConfig = {
        feature_flags: {
          semantic_search: { enabled: true, rollout_percentage: 50 },
          new_ui: { enabled: false, rollout_percentage: 0 },
          performance_mode: { enabled: true, rollout_percentage: 100 }
        }
      };

      const fingerprint = versionManager.createFingerprint(featureFlagConfig, '2.1.0');

      expect(fingerprint.version).toBe('2.1.0');
    });

    it('should handle environment-specific configurations', () => {
      const environments = ['development', 'staging', 'production'];
      
      environments.forEach(env => {
        const envConfig = {
          environment: env,
          database_url: `postgresql://${env}.example.com`,
          log_level: env === 'production' ? 'warn' : 'debug'
        };

        const fingerprint = versionManager.createFingerprint(
          envConfig, 
          '1.0.0', 
          `${env}_commit`
        );

        expect(fingerprint.git_commit).toBe(`${env}_commit`);
      });
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle file system errors gracefully', () => {
      const { writeFileSync } = require('fs');
      writeFileSync.mockImplementation(() => {
        throw new Error('Disk full');
      });

      expect(() => {
        versionManager.tagVersion(mockConfigFingerprint);
      }).toThrow('Failed to save version');
    });

    it('should handle permission errors', () => {
      const { mkdirSync } = require('fs');
      mkdirSync.mockImplementation(() => {
        throw new Error('Permission denied');
      });

      expect(() => {
        new VersionManager('/protected/versions');
      }).toThrow('Permission denied');
    });

    it('should handle corrupted version directory', () => {
      vi.spyOn(versionManager, 'listVersions').mockImplementation(() => {
        throw new Error('Cannot read directory');
      });

      expect(() => {
        versionManager.getVersionHistory();
      }).toThrow('Cannot read directory');
    });

    it('should handle malformed version files during listing', () => {
      const { readFileSync } = require('fs');
      
      // Mock file system to return malformed files
      vi.spyOn(versionManager as any, 'getVersionFiles')
        .mockReturnValue(['valid_1.0.0.json', 'invalid.txt', 'malformed_version.json']);

      vi.spyOn(versionManager, 'listVersions').mockImplementation(() => {
        return ['1.0.0']; // Only valid versions
      });

      const versions = versionManager.listVersions();
      expect(versions).toEqual(['1.0.0']);
    });

    it('should validate input parameters', () => {
      expect(() => {
        versionManager.createFingerprint(null as any, '1.0.0');
      }).toThrow('Configuration cannot be null');

      expect(() => {
        versionManager.createFingerprint({}, '');
      }).toThrow('Version cannot be empty');

      expect(() => {
        versionManager.loadVersion('');
      }).toThrow('Version cannot be empty');
    });
  });
});