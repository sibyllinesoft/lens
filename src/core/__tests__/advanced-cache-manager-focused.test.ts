import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

// Mock all external dependencies BEFORE importing
vi.mock('../telemetry/tracer', () => ({
  LensTracer: {
    createChildSpan: vi.fn(() => ({
      setAttributes: vi.fn(),
      recordException: vi.fn(),
      end: vi.fn(),
    })),
  },
}));

vi.mock('./memory-pool-manager', () => ({
  globalMemoryPool: {
    allocate: vi.fn(),
    release: vi.fn(),
  },
}));

// Import AFTER mocks are set up
import { AdvancedCacheManager } from '../advanced-cache-manager';

describe('AdvancedCacheManager', () => {
  let cacheManager: AdvancedCacheManager;

  beforeEach(() => {
    vi.clearAllMocks();
    cacheManager = AdvancedCacheManager.getInstance();
    cacheManager.clear(); // Start with clean state
  });

  afterEach(() => {
    try {
      cacheManager.shutdown();
    } catch (e) {
      // Ignore shutdown errors in tests
    }
  });

  describe('Singleton Pattern', () => {
    it('should return the same instance on multiple calls', () => {
      const instance1 = AdvancedCacheManager.getInstance();
      const instance2 = AdvancedCacheManager.getInstance();
      expect(instance1).toBe(instance2);
    });

    it('should initialize with default configuration', () => {
      const stats = cacheManager.getStats();
      expect(stats).toBeDefined();
      expect(stats.totalEntries).toBe(0);
      expect(stats.totalMemoryMB).toBe(0);
      expect(stats.hitRate).toBe(0);
    });
  });

  describe('Basic Cache Operations', () => {
    it('should set and get string values', async () => {
      const key = 'test-string';
      const value = 'Hello, World!';
      
      await cacheManager.set(key, value);
      const retrieved = await cacheManager.get<string>(key);
      
      expect(retrieved).toBe(value);
    });

    it('should set and get number values', async () => {
      const key = 'test-number';
      const value = 42;
      
      await cacheManager.set(key, value);
      const retrieved = await cacheManager.get<number>(key);
      
      expect(retrieved).toBe(value);
    });

    it('should set and get object values', async () => {
      const key = 'test-object';
      const value = { 
        id: 1, 
        name: 'Test Object', 
        data: [1, 2, 3],
        nested: { prop: 'value' }
      };
      
      await cacheManager.set(key, value);
      const retrieved = await cacheManager.get<typeof value>(key);
      
      expect(retrieved).toEqual(value);
    });

    it('should set and get array values', async () => {
      const key = 'test-array';
      const value = [
        { id: 1, name: 'Item 1' },
        { id: 2, name: 'Item 2' },
        { id: 3, name: 'Item 3' }
      ];
      
      await cacheManager.set(key, value);
      const retrieved = await cacheManager.get<typeof value>(key);
      
      expect(retrieved).toEqual(value);
    });

    it('should return null for non-existent keys', async () => {
      const retrieved = await cacheManager.get('non-existent-key');
      expect(retrieved).toBeNull();
    });

    it('should handle undefined and null values', async () => {
      await cacheManager.set('undefined-key', undefined);
      await cacheManager.set('null-key', null);
      
      const undefinedValue = await cacheManager.get('undefined-key');
      const nullValue = await cacheManager.get('null-key');
      
      expect(undefinedValue).toBeUndefined();
      expect(nullValue).toBeNull();
    });
  });

  describe('TTL (Time To Live) Functionality', () => {
    it('should respect default TTL', async () => {
      const key = 'ttl-test';
      const value = 'expires soon';
      
      await cacheManager.set(key, value, 100); // 100ms TTL
      
      // Should be available immediately
      let retrieved = await cacheManager.get(key);
      expect(retrieved).toBe(value);
      
      // Wait for TTL expiration
      await new Promise(resolve => setTimeout(resolve, 150));
      
      // Should be null after expiration
      retrieved = await cacheManager.get(key);
      expect(retrieved).toBeNull();
    });

    it('should handle very short TTL values', async () => {
      const key = 'short-ttl';
      const value = 'very short lived';
      
      await cacheManager.set(key, value, 1); // 1ms TTL
      
      // Wait for guaranteed expiration
      await new Promise(resolve => setTimeout(resolve, 10));
      
      const retrieved = await cacheManager.get(key);
      expect(retrieved).toBeNull();
    });

    it('should handle long TTL values', async () => {
      const key = 'long-ttl';
      const value = 'long lived';
      
      await cacheManager.set(key, value, 60000); // 1 minute TTL
      
      const retrieved = await cacheManager.get(key);
      expect(retrieved).toBe(value);
    });

    it('should update TTL on re-set', async () => {
      const key = 'update-ttl';
      const value1 = 'first value';
      const value2 = 'second value';
      
      // Set with short TTL
      await cacheManager.set(key, value1, 100);
      
      // Immediately update with longer TTL
      await cacheManager.set(key, value2, 60000);
      
      // Wait past original TTL
      await new Promise(resolve => setTimeout(resolve, 150));
      
      // Should still be available with new value
      const retrieved = await cacheManager.get(key);
      expect(retrieved).toBe(value2);
    });
  });

  describe('Cache Tiers', () => {
    it('should handle hot tier operations', async () => {
      // Simulate hot tier usage with frequent access
      const key = 'hot-data';
      const value = { type: 'hot', data: 'frequently accessed' };
      
      await cacheManager.set(key, value);
      
      // Access multiple times to promote to hot tier
      for (let i = 0; i < 10; i++) {
        await cacheManager.get(key);
      }
      
      const stats = cacheManager.getStats();
      expect(stats.totalEntries).toBeGreaterThan(0);
      expect(stats.hitRate).toBeGreaterThan(0);
    });

    it('should handle warm tier operations', async () => {
      const key = 'warm-data';
      const value = { type: 'warm', data: 'moderately accessed' };
      
      await cacheManager.set(key, value);
      
      // Access a moderate number of times
      for (let i = 0; i < 3; i++) {
        await cacheManager.get(key);
      }
      
      const retrieved = await cacheManager.get(key);
      expect(retrieved).toEqual(value);
    });

    it('should handle cold tier operations', async () => {
      const key = 'cold-data';
      const value = { type: 'cold', data: 'rarely accessed' };
      
      await cacheManager.set(key, value);
      
      // Access only once
      const retrieved = await cacheManager.get(key);
      expect(retrieved).toEqual(value);
    });

    it('should track tier-specific statistics', async () => {
      // Add entries to different tiers
      await cacheManager.set('hot-key', 'hot-value');
      await cacheManager.set('warm-key', 'warm-value');
      await cacheManager.set('cold-key', 'cold-value');
      
      const stats = cacheManager.getStats();
      expect(stats.totalEntries).toBe(3);
      expect(stats.hotTierEntries + stats.warmTierEntries + stats.coldTierEntries).toBe(3);
    });
  });

  describe('Search Context Integration', () => {
    it('should handle cache operations with search context', async () => {
      const context = {
        repo_name: 'test-repo',
        repo_sha: 'abc123',
        mode: 'lexical' as const,
      };
      
      const key = 'context-key';
      const value = { search: 'results', count: 5 };
      
      await cacheManager.set(key, value, undefined, context);
      const retrieved = await cacheManager.get(key, context);
      
      expect(retrieved).toEqual(value);
    });

    it('should handle different search modes', async () => {
      const contexts = [
        { repo_name: 'test', repo_sha: 'abc', mode: 'lexical' as const },
        { repo_name: 'test', repo_sha: 'abc', mode: 'semantic' as const },
        { repo_name: 'test', repo_sha: 'abc', mode: 'hybrid' as const },
      ];
      
      for (let i = 0; i < contexts.length; i++) {
        const key = `mode-key-${i}`;
        const value = { mode: contexts[i].mode, results: [] };
        
        await cacheManager.set(key, value, undefined, contexts[i]);
        const retrieved = await cacheManager.get(key, contexts[i]);
        
        expect(retrieved).toEqual(value);
      }
    });

    it('should handle file-specific context', async () => {
      const context = {
        repo_name: 'test-repo',
        repo_sha: 'abc123',
        file_path: 'src/main.ts',
        mode: 'structural' as const,
      };
      
      const key = 'file-specific-key';
      const value = { file: 'src/main.ts', symbols: ['function1', 'class1'] };
      
      await cacheManager.set(key, value, undefined, context);
      const retrieved = await cacheManager.get(key, context);
      
      expect(retrieved).toEqual(value);
    });
  });

  describe('Performance and Statistics', () => {
    it('should track hit rates accurately', async () => {
      const key = 'hit-rate-test';
      const value = 'test value';
      
      await cacheManager.set(key, value);
      
      // Generate hits
      for (let i = 0; i < 5; i++) {
        await cacheManager.get(key);
      }
      
      // Generate misses
      for (let i = 0; i < 3; i++) {
        await cacheManager.get(`non-existent-${i}`);
      }
      
      const stats = cacheManager.getStats();
      expect(stats.hitRate).toBeGreaterThan(0);
      expect(stats.missRate).toBeGreaterThan(0);
      expect(stats.hitRate + stats.missRate).toBeCloseTo(100, 1);
    });

    it('should track memory usage', async () => {
      const initialStats = cacheManager.getStats();
      const initialMemory = initialStats.totalMemoryMB;
      
      // Add several entries
      for (let i = 0; i < 10; i++) {
        await cacheManager.set(`memory-key-${i}`, {
          id: i,
          data: 'x'.repeat(1000), // ~1KB of data
          timestamp: Date.now()
        });
      }
      
      const afterStats = cacheManager.getStats();
      expect(afterStats.totalMemoryMB).toBeGreaterThan(initialMemory);
      expect(afterStats.totalEntries).toBe(10);
    });

    it('should track access patterns', async () => {
      const keys = ['access-1', 'access-2', 'access-3'];
      const values = ['value-1', 'value-2', 'value-3'];
      
      // Set all keys
      for (let i = 0; i < keys.length; i++) {
        await cacheManager.set(keys[i], values[i]);
      }
      
      // Access with different frequencies
      for (let i = 0; i < 10; i++) {
        await cacheManager.get('access-1'); // High frequency
      }
      
      for (let i = 0; i < 3; i++) {
        await cacheManager.get('access-2'); // Medium frequency
      }
      
      await cacheManager.get('access-3'); // Low frequency
      
      const stats = cacheManager.getStats();
      expect(stats.hitRate).toBeGreaterThan(0);
      expect(stats.avgAccessTime).toBeGreaterThanOrEqual(0);
    });

    it('should provide comprehensive statistics', async () => {
      // Add various types of data
      await cacheManager.set('string-key', 'string value');
      await cacheManager.set('number-key', 123);
      await cacheManager.set('object-key', { complex: 'object' });
      
      // Access some entries
      await cacheManager.get('string-key');
      await cacheManager.get('object-key');
      await cacheManager.get('non-existent');
      
      const stats = cacheManager.getStats();
      
      expect(typeof stats.totalEntries).toBe('number');
      expect(typeof stats.hotTierEntries).toBe('number');
      expect(typeof stats.warmTierEntries).toBe('number');
      expect(typeof stats.coldTierEntries).toBe('number');
      expect(typeof stats.totalMemoryMB).toBe('number');
      expect(typeof stats.hitRate).toBe('number');
      expect(typeof stats.missRate).toBe('number');
      expect(typeof stats.evictionRate).toBe('number');
      expect(typeof stats.compressionRatio).toBe('number');
      expect(typeof stats.avgAccessTime).toBe('number');
    });
  });

  describe('Cache Management', () => {
    it('should clear all cache entries', async () => {
      // Add several entries
      for (let i = 0; i < 5; i++) {
        await cacheManager.set(`clear-key-${i}`, `value-${i}`);
      }
      
      let stats = cacheManager.getStats();
      expect(stats.totalEntries).toBe(5);
      
      // Clear the cache
      cacheManager.clear();
      
      stats = cacheManager.getStats();
      expect(stats.totalEntries).toBe(0);
      expect(stats.totalMemoryMB).toBe(0);
    });

    it('should handle shutdown gracefully', async () => {
      // Add some entries
      await cacheManager.set('shutdown-key', 'shutdown-value');
      
      let stats = cacheManager.getStats();
      expect(stats.totalEntries).toBe(1);
      
      // Should not throw
      expect(() => cacheManager.shutdown()).not.toThrow();
      
      // Should clear all data
      stats = cacheManager.getStats();
      expect(stats.totalEntries).toBe(0);
    });

    it('should reset statistics on clear', async () => {
      // Generate some activity
      await cacheManager.set('stats-key', 'stats-value');
      await cacheManager.get('stats-key');
      await cacheManager.get('non-existent');
      
      let stats = cacheManager.getStats();
      expect(stats.hitRate).toBeGreaterThan(0);
      
      // Clear and check reset
      cacheManager.clear();
      
      stats = cacheManager.getStats();
      expect(stats.hitRate).toBe(0);
      expect(stats.missRate).toBe(0);
    });
  });

  describe('Edge Cases and Error Handling', () => {
    it('should handle very large values', async () => {
      const key = 'large-value';
      const largeObject = {
        id: 'large-test',
        data: 'x'.repeat(100000), // ~100KB
        metadata: {
          created: Date.now(),
          tags: Array.from({ length: 1000 }, (_, i) => `tag-${i}`)
        }
      };
      
      await cacheManager.set(key, largeObject);
      const retrieved = await cacheManager.get(key);
      
      expect(retrieved).toEqual(largeObject);
    });

    it('should handle special key characters', async () => {
      const specialKeys = [
        'key-with-dashes',
        'key_with_underscores',
        'key.with.dots',
        'key:with:colons',
        'key/with/slashes',
        'key with spaces'
      ];
      
      for (const key of specialKeys) {
        const value = `value for ${key}`;
        await cacheManager.set(key, value);
        const retrieved = await cacheManager.get(key);
        expect(retrieved).toBe(value);
      }
    });

    it('should handle concurrent operations', async () => {
      const promises = [];
      
      // Concurrent set operations
      for (let i = 0; i < 20; i++) {
        promises.push(cacheManager.set(`concurrent-${i}`, `value-${i}`));
      }
      
      await Promise.all(promises);
      
      // Verify all values were set correctly
      for (let i = 0; i < 20; i++) {
        const retrieved = await cacheManager.get(`concurrent-${i}`);
        expect(retrieved).toBe(`value-${i}`);
      }
    });

    it('should handle rapid set/get cycles', async () => {
      const key = 'rapid-test';
      const iterations = 50;
      
      for (let i = 0; i < iterations; i++) {
        await cacheManager.set(key, `value-${i}`);
        const retrieved = await cacheManager.get(key);
        expect(retrieved).toBe(`value-${i}`);
      }
    });

    it('should handle empty string keys and values', async () => {
      await cacheManager.set('', 'empty key');
      await cacheManager.set('empty-value', '');
      
      const emptyKeyValue = await cacheManager.get('');
      const emptyValue = await cacheManager.get('empty-value');
      
      expect(emptyKeyValue).toBe('empty key');
      expect(emptyValue).toBe('');
    });
  });

  describe('Complex Data Structures', () => {
    it('should handle nested objects with functions', async () => {
      const complexObject = {
        metadata: {
          created: new Date(),
          version: '1.0.0'
        },
        results: [
          { id: 1, score: 0.95 },
          { id: 2, score: 0.87 }
        ],
        // Note: Functions won't serialize, but we test object handling
        config: {
          maxResults: 100,
          includeMetadata: true
        }
      };
      
      await cacheManager.set('complex-object', complexObject);
      const retrieved = await cacheManager.get('complex-object');
      
      // Functions won't survive serialization, but the rest should
      expect(retrieved).toBeDefined();
      expect(retrieved?.metadata).toBeDefined();
      expect(retrieved?.results).toHaveLength(2);
    });

    it('should handle Map-like structures', async () => {
      const mapLikeObject = {
        entries: new Map([
          ['key1', 'value1'],
          ['key2', { nested: 'value2' }],
          ['key3', [1, 2, 3]]
        ])
      };
      
      await cacheManager.set('map-like', mapLikeObject);
      const retrieved = await cacheManager.get('map-like');
      
      expect(retrieved).toBeDefined();
    });

    it('should handle circular reference protection', async () => {
      // Create object with potential circular reference
      const obj: any = {
        id: 'circular-test',
        data: 'some data'
      };
      obj.self = obj; // Circular reference
      
      // Should handle gracefully (might stringify without circular reference)
      await cacheManager.set('circular', obj);
      const retrieved = await cacheManager.get('circular');
      
      expect(retrieved).toBeDefined();
      expect(retrieved?.id).toBe('circular-test');
    });
  });
});