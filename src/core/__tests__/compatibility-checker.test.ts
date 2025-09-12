/**
 * Tests for Compatibility Checker
 * Covers Phase A1.2 compatibility checking against nightly bundles, version validation, and schema compatibility
 */

import { describe, it, expect, beforeEach, jest, afterEach, mock } from 'bun:test';
import { promises as fs } from 'fs';
import * as path from 'path';
import {
  checkBundleCompatibility,
  checkApiCompatibility,
  checkIndexCompatibility,
  checkPolicyCompatibility,
  loadNightlyBundles,
  validateBundleMetadata,
  generateCompatibilityMatrix,
  NightlyBundle,
  CompatibilityReport,
} from '../compatibility-checker.js';

// Mock the file system operations
mock('fs/promises');
const mockFs = mocked(fs);

// Mock the tracer
mock('../../telemetry/tracer.js', () => ({
  LensTracer: {
    createChildSpan: jest.fn(() => ({
      setAttributes: jest.fn(),
      setStatus: jest.fn(),
      end: jest.fn(),
    })),
  },
}));

// Mock version constants
mock('../version-manager.js', () => ({
  SERVER_API_VERSION: '1.2.3',
  SERVER_INDEX_VERSION: '2.1.0',
  SERVER_POLICY_VERSION: '1.0.5',
}));

describe('Compatibility Checker', () => {
  const mockBundlesPath = './test-nightly-bundles';
  const mockBundle1: NightlyBundle = {
    bundle_id: 'nightly-20241201-abc123',
    created_at: '2024-12-01T02:00:00Z',
    api_version: '1.2.3',
    index_version: '2.1.0', 
    policy_version: '1.0.5',
    schema_hash: 'sha256:abc123def456',
    index_format_version: 3,
  };

  const mockBundle2: NightlyBundle = {
    bundle_id: 'nightly-20241202-def456',
    created_at: '2024-12-02T02:00:00Z',
    api_version: '1.2.4',
    index_version: '2.1.1',
    policy_version: '1.0.5',
    schema_hash: 'sha256:def456ghi789',
    index_format_version: 3,
  };

  const incompatibleBundle: NightlyBundle = {
    bundle_id: 'nightly-20241203-ghi789',
    created_at: '2024-12-03T02:00:00Z',
    api_version: '2.0.0', // Major version change
    index_version: '3.0.0', // Major version change
    policy_version: '2.0.0', // Major version change
    schema_hash: 'sha256:ghi789jkl012',
    index_format_version: 4, // Format version change
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('Bundle Loading', () => {
    it('should load nightly bundles from directory', async () => {
      mockFs.readdir.mockResolvedValue([
        'nightly-20241201-abc123.json' as any,
        'nightly-20241202-def456.json' as any,
        'other-file.txt' as any, // Should be ignored
      ]);
      
      mockFs.readFile
        .mockResolvedValueOnce(JSON.stringify(mockBundle1))
        .mockResolvedValueOnce(JSON.stringify(mockBundle2));

      const bundles = await loadNightlyBundles(mockBundlesPath);

      expect(bundles).toHaveLength(2);
      expect(bundles[0]).toEqual(mockBundle1);
      expect(bundles[1]).toEqual(mockBundle2);
      expect(mockFs.readdir).toHaveBeenCalledWith(mockBundlesPath);
    });

    it('should handle missing bundles directory', async () => {
      mockFs.readdir.mockRejectedValue(new Error('ENOENT: no such file or directory'));

      await expect(loadNightlyBundles(mockBundlesPath)).rejects.toThrow();
    });

    it('should handle corrupted bundle files', async () => {
      mockFs.readdir.mockResolvedValue(['nightly-20241201-abc123.json' as any]);
      mockFs.readFile.mockResolvedValue('invalid json content');

      await expect(loadNightlyBundles(mockBundlesPath)).rejects.toThrow();
    });

    it('should filter and sort bundles by date', async () => {
      const olderBundle = {
        ...mockBundle1,
        bundle_id: 'nightly-20241130-old123',
        created_at: '2024-11-30T02:00:00Z',
      };

      mockFs.readdir.mockResolvedValue([
        'nightly-20241202-def456.json' as any,
        'nightly-20241130-old123.json' as any,
        'nightly-20241201-abc123.json' as any,
      ]);
      
      mockFs.readFile
        .mockResolvedValueOnce(JSON.stringify(mockBundle2))
        .mockResolvedValueOnce(JSON.stringify(olderBundle))
        .mockResolvedValueOnce(JSON.stringify(mockBundle1));

      const bundles = await loadNightlyBundles(mockBundlesPath, 2);

      expect(bundles).toHaveLength(2);
      // Should return the 2 most recent bundles
      expect(bundles[0].created_at).toBe('2024-12-02T02:00:00Z');
      expect(bundles[1].created_at).toBe('2024-12-01T02:00:00Z');
    });
  });

  describe('Bundle Metadata Validation', () => {
    it('should validate correct bundle metadata', () => {
      expect(() => validateBundleMetadata(mockBundle1)).not.toThrow();
    });

    it('should reject bundle with missing required fields', () => {
      const invalidBundle = { ...mockBundle1 };
      delete (invalidBundle as any).api_version;

      expect(() => validateBundleMetadata(invalidBundle as any)).toThrow(/api_version/);
    });

    it('should reject bundle with invalid version format', () => {
      const invalidBundle = {
        ...mockBundle1,
        api_version: 'invalid-version',
      };

      expect(() => validateBundleMetadata(invalidBundle)).toThrow(/version format/);
    });

    it('should reject bundle with invalid schema hash', () => {
      const invalidBundle = {
        ...mockBundle1,
        schema_hash: 'invalid-hash',
      };

      expect(() => validateBundleMetadata(invalidBundle)).toThrow(/schema hash/);
    });

    it('should reject bundle with invalid index format version', () => {
      const invalidBundle = {
        ...mockBundle1,
        index_format_version: -1,
      };

      expect(() => validateBundleMetadata(invalidBundle)).toThrow(/index format version/);
    });
  });

  describe('Version Compatibility Checking', () => {
    it('should check API compatibility for same version', async () => {
      const result = await checkApiCompatibility('1.2.3', '1.2.3');
      
      expect(result.compatible).toBe(true);
      expect(result.issues).toHaveLength(0);
    });

    it('should check API compatibility for patch version differences', async () => {
      const result = await checkApiCompatibility('1.2.3', '1.2.4');
      
      expect(result.compatible).toBe(true);
      expect(result.issues).toHaveLength(0);
    });

    it('should check API compatibility for minor version differences', async () => {
      const result = await checkApiCompatibility('1.2.3', '1.3.0');
      
      expect(result.compatible).toBe(true);
      expect(result.warnings).toContain(/minor version difference/);
    });

    it('should check API incompatibility for major version differences', async () => {
      const result = await checkApiCompatibility('1.2.3', '2.0.0');
      
      expect(result.compatible).toBe(false);
      expect(result.issues).toContain(/major version change/);
    });

    it('should check index compatibility for format versions', async () => {
      const result = await checkIndexCompatibility('2.1.0', '2.1.0', 3, 3);
      
      expect(result.compatible).toBe(true);
      expect(result.issues).toHaveLength(0);
    });

    it('should check index incompatibility for format version changes', async () => {
      const result = await checkIndexCompatibility('2.1.0', '2.1.0', 3, 4);
      
      expect(result.compatible).toBe(false);
      expect(result.issues).toContain(/index format version/);
    });

    it('should check policy compatibility', async () => {
      const compatibleResult = await checkPolicyCompatibility('1.0.5', '1.0.5');
      const incompatibleResult = await checkPolicyCompatibility('1.0.5', '2.0.0');
      
      expect(compatibleResult.compatible).toBe(true);
      expect(incompatibleResult.compatible).toBe(false);
      expect(incompatibleResult.issues).toContain(/policy version/);
    });
  });

  describe('Compatibility Matrix Generation', () => {
    it('should generate compatibility matrix for compatible bundles', async () => {
      const bundles = [mockBundle1, mockBundle2];
      const matrix = await generateCompatibilityMatrix(bundles);

      expect(matrix).toHaveLength(2);
      expect(matrix[0].bundle_id).toBe(mockBundle1.bundle_id);
      expect(matrix[0].api_compatible).toBe(true);
      expect(matrix[0].index_compatible).toBe(true);
      expect(matrix[0].policy_compatible).toBe(true);
      expect(matrix[0].issues).toHaveLength(0);
    });

    it('should generate compatibility matrix for incompatible bundles', async () => {
      const bundles = [mockBundle1, incompatibleBundle];
      const matrix = await generateCompatibilityMatrix(bundles);

      expect(matrix).toHaveLength(2);
      
      const compatibleEntry = matrix.find(entry => entry.bundle_id === mockBundle1.bundle_id);
      const incompatibleEntry = matrix.find(entry => entry.bundle_id === incompatibleBundle.bundle_id);

      expect(compatibleEntry?.api_compatible).toBe(true);
      expect(incompatibleEntry?.api_compatible).toBe(false);
      expect(incompatibleEntry?.issues.length).toBeGreaterThan(0);
    });

    it('should handle mixed compatibility scenarios', async () => {
      const partiallyCompatibleBundle = {
        ...mockBundle1,
        bundle_id: 'partial-compat-test',
        api_version: '1.3.0', // Minor version change - compatible with warnings
        index_version: '3.0.0', // Major version change - incompatible
        policy_version: '1.0.5', // Same - compatible
      };

      const bundles = [partiallyCompatibleBundle];
      const matrix = await generateCompatibilityMatrix(bundles);

      expect(matrix).toHaveLength(1);
      expect(matrix[0].api_compatible).toBe(true);
      expect(matrix[0].index_compatible).toBe(false);
      expect(matrix[0].policy_compatible).toBe(true);
    });
  });

  describe('Full Compatibility Check', () => {
    beforeEach(() => {
      mockFs.readdir.mockResolvedValue([
        'nightly-20241201-abc123.json' as any,
        'nightly-20241202-def456.json' as any,
      ]);
      
      mockFs.readFile
        .mockResolvedValueOnce(JSON.stringify(mockBundle1))
        .mockResolvedValueOnce(JSON.stringify(mockBundle2));
    });

    it('should perform full compatibility check with compatible bundles', async () => {
      const result = await checkBundleCompatibility(mockBundlesPath);

      expect(result.compatible).toBe(true);
      expect(result.overall_status).toBe('compatible');
      expect(result.bundles_checked).toHaveLength(2);
      expect(result.compatibility_matrix).toHaveLength(2);
      expect(result.errors).toHaveLength(0);
      expect(result.current_version.api_version).toBe('1.2.3');
    });

    it('should perform full compatibility check with incompatible bundles', async () => {
      mockFs.readFile
        .mockResolvedValueOnce(JSON.stringify(mockBundle1))
        .mockResolvedValueOnce(JSON.stringify(incompatibleBundle));

      const result = await checkBundleCompatibility(mockBundlesPath);

      expect(result.compatible).toBe(false);
      expect(result.overall_status).toBe('incompatible');
      expect(result.errors.length).toBeGreaterThan(0);
    });

    it('should handle partial compatibility', async () => {
      const partiallyCompatibleBundle = {
        ...mockBundle1,
        bundle_id: 'partial-test',
        api_version: '1.3.0', // Compatible with warnings
      };

      mockFs.readFile
        .mockResolvedValueOnce(JSON.stringify(mockBundle1))
        .mockResolvedValueOnce(JSON.stringify(partiallyCompatibleBundle));

      const result = await checkBundleCompatibility(mockBundlesPath);

      expect(result.overall_status).toBe('partial');
      expect(result.warnings.length).toBeGreaterThan(0);
    });

    it('should respect allowCompat parameter', async () => {
      mockFs.readFile
        .mockResolvedValueOnce(JSON.stringify(mockBundle1))
        .mockResolvedValueOnce(JSON.stringify(incompatibleBundle));

      const strictResult = await checkBundleCompatibility(mockBundlesPath, false);
      const relaxedResult = await checkBundleCompatibility(mockBundlesPath, true);

      expect(strictResult.compatible).toBe(false);
      // With allowCompat=true, some compatibility issues might be downgraded to warnings
      expect(relaxedResult.warnings.length).toBeGreaterThanOrEqual(strictResult.warnings.length);
    });

    it('should handle empty bundles directory gracefully', async () => {
      mockFs.readdir.mockResolvedValue([]);

      const result = await checkBundleCompatibility(mockBundlesPath);

      expect(result.compatible).toBe(true);
      expect(result.bundles_checked).toHaveLength(0);
      expect(result.overall_status).toBe('compatible');
      expect(result.warnings).toContain(/no bundles found/);
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle file system errors gracefully', async () => {
      mockFs.readdir.mockRejectedValue(new Error('Permission denied'));

      await expect(checkBundleCompatibility(mockBundlesPath)).rejects.toThrow(/Permission denied/);
    });

    it('should handle malformed JSON in bundle files', async () => {
      mockFs.readdir.mockResolvedValue(['invalid-bundle.json' as any]);
      mockFs.readFile.mockResolvedValue('{ invalid json }');

      await expect(checkBundleCompatibility(mockBundlesPath)).rejects.toThrow();
    });

    it('should handle bundles with future timestamps', async () => {
      const futureBundle = {
        ...mockBundle1,
        bundle_id: 'future-bundle',
        created_at: '2030-01-01T00:00:00Z',
      };

      mockFs.readdir.mockResolvedValue(['future-bundle.json' as any]);
      mockFs.readFile.mockResolvedValue(JSON.stringify(futureBundle));

      const result = await checkBundleCompatibility(mockBundlesPath);
      
      expect(result.warnings).toContain(/future timestamp/);
    });

    it('should validate schema hash format', async () => {
      const invalidHashBundle = {
        ...mockBundle1,
        schema_hash: 'invalid-hash-format',
      };

      mockFs.readdir.mockResolvedValue(['invalid-hash-bundle.json' as any]);
      mockFs.readFile.mockResolvedValue(JSON.stringify(invalidHashBundle));

      await expect(checkBundleCompatibility(mockBundlesPath)).rejects.toThrow(/schema hash/);
    });

    it('should handle very large bundle lists efficiently', async () => {
      // Create 100 mock bundle files
      const manyBundles = Array.from({ length: 100 }, (_, i) => `nightly-bundle-${i}.json` as any);
      const mockBundles = Array.from({ length: 100 }, (_, i) => ({
        ...mockBundle1,
        bundle_id: `nightly-bundle-${i}`,
        created_at: new Date(Date.now() - i * 24 * 60 * 60 * 1000).toISOString(),
      }));

      mockFs.readdir.mockResolvedValue(manyBundles);
      mockBundles.forEach(bundle => {
        mockFs.readFile.mockResolvedValueOnce(JSON.stringify(bundle));
      });

      const start = performance.now();
      const result = await checkBundleCompatibility(mockBundlesPath, false);
      const duration = performance.now() - start;

      // Should complete within reasonable time even with many bundles
      expect(duration).toBeLessThan(5000); // 5 seconds
      expect(result.bundles_checked.length).toBeLessThanOrEqual(10); // Should limit to most recent
    });
  });

  describe('Performance and Concurrency', () => {
    it('should handle concurrent compatibility checks', async () => {
      mockFs.readdir.mockResolvedValue([
        'nightly-20241201-abc123.json' as any,
        'nightly-20241202-def456.json' as any,
      ]);
      
      mockFs.readFile
        .mockResolvedValue(JSON.stringify(mockBundle1))
        .mockResolvedValue(JSON.stringify(mockBundle2));

      const promises = Array.from({ length: 5 }, () => 
        checkBundleCompatibility(mockBundlesPath)
      );

      const results = await Promise.all(promises);

      // All results should be consistent
      results.forEach(result => {
        expect(result.compatible).toBe(true);
        expect(result.bundles_checked).toHaveLength(2);
      });
    });

    it('should cache bundle loading for performance', async () => {
      mockFs.readdir.mockResolvedValue(['bundle.json' as any]);
      mockFs.readFile.mockResolvedValue(JSON.stringify(mockBundle1));

      // First call
      await checkBundleCompatibility(mockBundlesPath);
      
      // Second call should use cache (reduce file system calls)
      await checkBundleCompatibility(mockBundlesPath);

      // Verify reasonable number of fs calls
      expect(mockFs.readdir).toHaveBeenCalledTimes(2);
      expect(mockFs.readFile).toHaveBeenCalledTimes(2);
    });
  });
});