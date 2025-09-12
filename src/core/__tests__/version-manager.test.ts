/**
 * Tests for Version Manager
 * Covers compatibility checking, version validation, and version info
 */

import { describe, it, expect } from 'bun:test';
import {
  checkCompatibility,
  validateVersionCompatibility,
  getVersionInfo,
  SERVER_API_VERSION,
  SERVER_INDEX_VERSION,
  SERVER_POLICY_VERSION,
} from '../version-manager.js';
import type { ApiVersion, IndexVersion, PolicyVersion } from '../../types/api.js';

describe('Version Manager', () => {
  describe('Server Version Constants', () => {
    it('should export valid server versions', () => {
      expect(SERVER_API_VERSION).toBe('v1');
      expect(SERVER_INDEX_VERSION).toBe('v1');
      expect(SERVER_POLICY_VERSION).toBe('v1');
    });
  });

  describe('Version Compatibility Checking', () => {
    it('should return compatible for matching versions', () => {
      const result = checkCompatibility('v1' as ApiVersion, 'v1' as IndexVersion);
      
      expect(result.compatible).toBe(true);
      expect(result.api_version).toBe('v1');
      expect(result.index_version).toBe('v1');
      expect(result.server_api_version).toBe('v1');
      expect(result.server_index_version).toBe('v1');
      expect(result.server_policy_version).toBe('v1');
      expect(result.warnings).toBeUndefined();
      expect(result.errors).toBeUndefined();
    });

    it('should return compatible for matching versions with policy', () => {
      const result = checkCompatibility('v1' as ApiVersion, 'v1' as IndexVersion, false, 'v1' as PolicyVersion);
      
      expect(result.compatible).toBe(true);
      expect(result.policy_version).toBe('v1');
      expect(result.warnings).toBeUndefined();
      expect(result.errors).toBeUndefined();
    });

    it('should return incompatible for mismatched API version', () => {
      const result = checkCompatibility('v2' as ApiVersion, 'v1' as IndexVersion);
      
      expect(result.compatible).toBe(false);
      expect(result.errors).toContain('API version mismatch: client v2, server v1');
      expect(result.warnings).toBeUndefined();
    });

    it('should return incompatible for mismatched index version', () => {
      const result = checkCompatibility('v1' as ApiVersion, 'v2' as IndexVersion);
      
      expect(result.compatible).toBe(false);
      expect(result.errors).toContain('Index version mismatch: client v2, server v1');
      expect(result.warnings).toBeUndefined();
    });

    it('should return incompatible for mismatched policy version', () => {
      const result = checkCompatibility('v1' as ApiVersion, 'v1' as IndexVersion, false, 'v2' as PolicyVersion);
      
      expect(result.compatible).toBe(false);
      expect(result.errors).toContain('Policy version mismatch: client v2, server v1');
      expect(result.warnings).toBeUndefined();
    });

    it('should handle multiple version mismatches', () => {
      const result = checkCompatibility('v2' as ApiVersion, 'v3' as IndexVersion, false, 'v4' as PolicyVersion);
      
      expect(result.compatible).toBe(false);
      expect(result.errors).toHaveLength(3);
      expect(result.errors).toContain('API version mismatch: client v2, server v1');
      expect(result.errors).toContain('Index version mismatch: client v3, server v1');
      expect(result.errors).toContain('Policy version mismatch: client v4, server v1');
    });

    it('should allow compatibility with allowCompat flag', () => {
      const result = checkCompatibility('v2' as ApiVersion, 'v3' as IndexVersion, true, 'v4' as PolicyVersion);
      
      expect(result.compatible).toBe(true);
      expect(result.warnings).toHaveLength(3);
      expect(result.warnings).toContain('API version mismatch allowed by --allow-compat flag: client v2, server v1');
      expect(result.warnings).toContain('Index version mismatch allowed by --allow-compat flag: client v3, server v1');
      expect(result.warnings).toContain('Policy version mismatch allowed by --allow-compat flag: client v4, server v1');
      expect(result.errors).toBeUndefined();
    });

    it('should partially allow compatibility with allowCompat flag for mixed matches', () => {
      const result = checkCompatibility('v1' as ApiVersion, 'v2' as IndexVersion, true);
      
      expect(result.compatible).toBe(true);
      expect(result.warnings).toHaveLength(1);
      expect(result.warnings).toContain('Index version mismatch allowed by --allow-compat flag: client v2, server v1');
      expect(result.errors).toBeUndefined();
    });

    it('should handle unknown versions gracefully', () => {
      const result = checkCompatibility('unknown' as ApiVersion, 'unknown' as IndexVersion);
      
      expect(result.compatible).toBe(false);
      expect(result.errors).toContain('API version mismatch: client unknown, server v1');
      expect(result.errors).toContain('Index version mismatch: client unknown, server v1');
    });

    it('should handle missing policy version', () => {
      const result = checkCompatibility('v1' as ApiVersion, 'v1' as IndexVersion);
      
      expect(result.compatible).toBe(true);
      expect(result.policy_version).toBeUndefined();
      expect(result.server_policy_version).toBe('v1');
    });
  });

  describe('Version Validation', () => {
    it('should not throw for compatible versions', () => {
      expect(() => {
        validateVersionCompatibility('v1' as ApiVersion, 'v1' as IndexVersion);
      }).not.toThrow();
    });

    it('should not throw for compatible versions with policy', () => {
      expect(() => {
        validateVersionCompatibility('v1' as ApiVersion, 'v1' as IndexVersion, false, 'v1' as PolicyVersion);
      }).not.toThrow();
    });

    it('should throw for incompatible API version', () => {
      expect(() => {
        validateVersionCompatibility('v2' as ApiVersion, 'v1' as IndexVersion);
      }).toThrow('Version compatibility error: API version mismatch: client v2, server v1');
    });

    it('should throw for incompatible index version', () => {
      expect(() => {
        validateVersionCompatibility('v1' as ApiVersion, 'v2' as IndexVersion);
      }).toThrow('Version compatibility error: Index version mismatch: client v2, server v1');
    });

    it('should throw for incompatible policy version', () => {
      expect(() => {
        validateVersionCompatibility('v1' as ApiVersion, 'v1' as IndexVersion, false, 'v2' as PolicyVersion);
      }).toThrow('Version compatibility error: Policy version mismatch: client v2, server v1');
    });

    it('should throw with multiple errors combined', () => {
      expect(() => {
        validateVersionCompatibility('v2' as ApiVersion, 'v3' as IndexVersion);
      }).toThrow('Version compatibility error: API version mismatch: client v2, server v1; Index version mismatch: client v3, server v1');
    });

    it('should not throw when allowCompat is true', () => {
      expect(() => {
        validateVersionCompatibility('v2' as ApiVersion, 'v3' as IndexVersion, true, 'v4' as PolicyVersion);
      }).not.toThrow();
    });

    it('should throw for unknown versions', () => {
      expect(() => {
        validateVersionCompatibility('unknown' as ApiVersion, 'unknown' as IndexVersion);
      }).toThrow('Version compatibility error');
    });
  });

  describe('Version Info', () => {
    it('should return current server version info', () => {
      const versionInfo = getVersionInfo();
      
      expect(versionInfo).toEqual({
        api_version: 'v1',
        index_version: 'v1',
        policy_version: 'v1',
      });
    });

    it('should return consistent versions', () => {
      const versionInfo = getVersionInfo();
      
      expect(versionInfo.api_version).toBe(SERVER_API_VERSION);
      expect(versionInfo.index_version).toBe(SERVER_INDEX_VERSION);
      expect(versionInfo.policy_version).toBe(SERVER_POLICY_VERSION);
    });
  });

  describe('Edge Cases', () => {
    it('should handle undefined client versions', () => {
      const result = checkCompatibility(undefined as any, undefined as any);
      
      expect(result.compatible).toBe(false);
      expect(result.errors).toContain('API version mismatch: client undefined, server v1');
      expect(result.errors).toContain('Index version mismatch: client undefined, server v1');
    });

    it('should handle null client versions', () => {
      const result = checkCompatibility(null as any, null as any);
      
      expect(result.compatible).toBe(false);
      expect(result.errors).toContain('API version mismatch: client null, server v1');
      expect(result.errors).toContain('Index version mismatch: client null, server v1');
    });

    it('should handle empty string versions', () => {
      const result = checkCompatibility('' as ApiVersion, '' as IndexVersion);
      
      expect(result.compatible).toBe(false);
      expect(result.errors).toContain('API version mismatch: client , server v1');
      expect(result.errors).toContain('Index version mismatch: client , server v1');
    });
  });
});