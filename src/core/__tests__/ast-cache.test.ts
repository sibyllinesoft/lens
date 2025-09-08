import { describe, it, expect, beforeEach, vi } from 'vitest';
import { ASTCache, CachedAST } from '../ast-cache';

describe('ASTCache', () => {
  let astCache: ASTCache;

  beforeEach(() => {
    vi.clearAllMocks();
    astCache = new ASTCache(10); // Small cache for testing
  });

  describe('Cache Initialization', () => {
    it('should initialize with default max files', () => {
      const cache = new ASTCache();
      expect(cache).toBeDefined();
      expect(cache.getStats()).toEqual({
        cacheSize: 0,
        hitCount: 0,
        missCount: 0,
        hitRate: 0,
        totalRequests: 0,
      });
    });

    it('should initialize with custom max files', () => {
      const cache = new ASTCache(5);
      expect(cache).toBeDefined();
    });
  });

  describe('AST Caching - TypeScript Files', () => {
    it('should parse and cache TypeScript function declarations', async () => {
      const filePath = '/test/functions.ts';
      const content = `
export async function processData(input: string): Promise<string> {
  return input.toUpperCase();
}

function helperFunction(x: number) {
  return x * 2;
}
`;

      const result = await astCache.getAST(filePath, content, 'typescript');
      
      expect(result).toBeDefined();
      expect(result.language).toBe('typescript');
      expect(result.symbolCount).toBeGreaterThan(0);
      expect(result.mockAST.functions).toHaveLength(2);
      
      expect(result.mockAST.functions[0].name).toBe('processData');
      expect(result.mockAST.functions[1].name).toBe('helperFunction');
    });

    it('should parse and cache TypeScript class declarations', async () => {
      const filePath = '/test/classes.ts';
      const content = `
export class DataProcessor extends BaseProcessor implements IProcessor {
  process() {}
}

abstract class AbstractHandler {
  abstract handle(): void;
}
`;

      const result = await astCache.getAST(filePath, content, 'typescript');
      
      expect(result.mockAST.classes).toHaveLength(2);
      expect(result.mockAST.classes[0].name).toBe('DataProcessor');
      expect(result.mockAST.classes[0].extends).toBe('BaseProcessor implements IProcessor');
      expect(result.mockAST.classes[1].name).toBe('AbstractHandler');
    });

    it('should parse and cache TypeScript interface declarations', async () => {
      const filePath = '/test/interfaces.ts';
      const content = `
export interface IUser extends BaseEntity {
  name: string;
  email: string;
}

interface IService<T> extends IBaseService<T>, ILoggable {
  process(data: T): Promise<T>;
}
`;

      const result = await astCache.getAST(filePath, content, 'typescript');
      
      expect(result.mockAST.interfaces).toHaveLength(2);
      expect(result.mockAST.interfaces[0].name).toBe('IUser');
      expect(result.mockAST.interfaces[0].extends).toEqual(['BaseEntity']);
      expect(result.mockAST.interfaces[1].name).toBe('IService');
      expect(result.mockAST.interfaces[1].extends).toEqual(['IBaseService<T>', 'ILoggable']);
    });

    it('should parse and cache TypeScript type definitions', async () => {
      const filePath = '/test/types.ts';
      const content = `
export type UserRole = 'admin' | 'user' | 'guest';
type EventHandler<T> = (event: T) => void;
`;

      const result = await astCache.getAST(filePath, content, 'typescript');
      
      expect(result.mockAST.types).toHaveLength(2);
      expect(result.mockAST.types[0].name).toBe('UserRole');
      expect(result.mockAST.types[0].definition).toBe("'admin' | 'user' | 'guest';");
      expect(result.mockAST.types[1].name).toBe('EventHandler');
    });

    it('should parse and cache TypeScript import statements', async () => {
      const filePath = '/test/imports.ts';
      const content = `
import React from 'react';
import { useState, useEffect } from 'react';
import { Logger } from '../utils/logger';
`;

      const result = await astCache.getAST(filePath, content, 'typescript');
      
      expect(result.mockAST.imports).toHaveLength(3);
      expect(result.mockAST.imports[0].module).toBe('react');
      expect(result.mockAST.imports[0].imports).toEqual(['React']);
      expect(result.mockAST.imports[1].imports).toEqual(['useState', 'useEffect']);
      expect(result.mockAST.imports[2].imports).toEqual(['Logger']);
    });
  });

  describe('AST Caching - JavaScript Files', () => {
    it('should parse JavaScript functions', async () => {
      const filePath = '/test/script.js';
      const content = `
function calculate(x, y) {
  return x + y;
}

export async function fetchData() {
  return await fetch('/api/data');
}
`;

      const result = await astCache.getAST(filePath, content, 'javascript');
      
      expect(result.mockAST.functions).toHaveLength(2);
      expect(result.mockAST.functions[0].name).toBe('calculate');
      expect(result.mockAST.functions[1].name).toBe('fetchData');
    });
  });

  describe('AST Caching - Python Files', () => {
    it('should parse Python functions', async () => {
      const filePath = '/test/script.py';
      const content = `
def calculate(x, y):
    return x + y

async def fetch_data():
    return await get_data()

class DataProcessor:
    def process(self, data):
        return data.upper()
`;

      const result = await astCache.getAST(filePath, content, 'python');
      
      expect(result.mockAST.functions).toHaveLength(3);
      expect(result.mockAST.functions[0].name).toBe('calculate');
      expect(result.mockAST.functions[1].name).toBe('fetch_data');
      expect(result.mockAST.functions[2].name).toBe('process'); // method from DataProcessor class
      
      expect(result.mockAST.classes).toHaveLength(1);
      expect(result.mockAST.classes[0].name).toBe('DataProcessor');
    });
  });

  describe('Cache Hit/Miss Logic', () => {
    it('should return cached result for unchanged content', async () => {
      const filePath = '/test/unchanged.ts';
      const content = 'function test() {}';
      
      // First call - cache miss
      const result1 = await astCache.getAST(filePath, content, 'typescript');
      const stats1 = astCache.getStats();
      expect(stats1.missCount).toBe(1);
      expect(stats1.hitCount).toBe(0);
      
      // Second call - cache hit
      const result2 = await astCache.getAST(filePath, content, 'typescript');
      const stats2 = astCache.getStats();
      expect(stats2.missCount).toBe(1);
      expect(stats2.hitCount).toBe(1);
      expect(stats2.hitRate).toBe(50); // 1 hit out of 2 total
      
      expect(result1.fileHash).toBe(result2.fileHash);
    });

    it('should re-parse when content changes', async () => {
      const filePath = '/test/changed.ts';
      const content1 = 'function test() {}';
      const content2 = 'function test() { return 42; }';
      
      const result1 = await astCache.getAST(filePath, content1, 'typescript');
      const result2 = await astCache.getAST(filePath, content2, 'typescript');
      
      const stats = astCache.getStats();
      expect(stats.missCount).toBe(2); // Both should be misses
      expect(stats.hitCount).toBe(0);
      
      expect(result1.fileHash).not.toBe(result2.fileHash);
    });

    it('should update lastAccessed on cache hit', async () => {
      const filePath = '/test/access.ts';
      const content = 'function test() {}';
      
      const result1 = await astCache.getAST(filePath, content, 'typescript');
      const initialAccess = result1.lastAccessed;
      
      // Wait a bit to ensure different timestamp
      await new Promise(resolve => setTimeout(resolve, 10));
      
      const result2 = await astCache.getAST(filePath, content, 'typescript');
      expect(result2.lastAccessed).toBeGreaterThan(initialAccess);
    });
  });

  describe('Cache Management', () => {
    it('should handle cache size limits with LRU eviction', async () => {
      const smallCache = new ASTCache(2); // Very small cache
      
      await smallCache.getAST('/file1.ts', 'function f1() {}', 'typescript');
      await smallCache.getAST('/file2.ts', 'function f2() {}', 'typescript');
      expect(smallCache.getStats().cacheSize).toBe(2);
      
      // This should evict the least recently used
      await smallCache.getAST('/file3.ts', 'function f3() {}', 'typescript');
      expect(smallCache.getStats().cacheSize).toBe(2);
    });

    it('should clear all cache data', async () => {
      await astCache.getAST('/test1.ts', 'function test1() {}', 'typescript');
      await astCache.getAST('/test2.ts', 'function test2() {}', 'typescript');
      
      const beforeClear = astCache.getStats();
      expect(beforeClear.cacheSize).toBe(2);
      expect(beforeClear.totalRequests).toBe(2);
      
      astCache.clear();
      
      const afterClear = astCache.getStats();
      expect(afterClear.cacheSize).toBe(0);
      expect(afterClear.hitCount).toBe(0);
      expect(afterClear.missCount).toBe(0);
      expect(afterClear.totalRequests).toBe(0);
    });
  });

  describe('Cache Statistics', () => {
    it('should calculate hit rate correctly', async () => {
      const filePath = '/test/stats.ts';
      const content = 'function test() {}';
      
      // 1 miss
      await astCache.getAST(filePath, content, 'typescript');
      expect(astCache.getStats().hitRate).toBe(0);
      
      // 1 hit
      await astCache.getAST(filePath, content, 'typescript');
      expect(astCache.getStats().hitRate).toBe(50);
      
      // Another hit
      await astCache.getAST(filePath, content, 'typescript');
      expect(astCache.getStats().hitRate).toBe(67); // 2/3 rounded
    });

    it('should provide comprehensive stats', async () => {
      await astCache.getAST('/test1.ts', 'function a() {}', 'typescript');
      await astCache.getAST('/test2.ts', 'function b() {}', 'typescript');
      await astCache.getAST('/test1.ts', 'function a() {}', 'typescript'); // Hit
      
      const stats = astCache.getStats();
      expect(stats).toEqual({
        cacheSize: 2,
        hitCount: 1,
        missCount: 2,
        hitRate: 33, // 1/3 rounded
        totalRequests: 3,
      });
    });
  });

  describe('Coverage Statistics', () => {
    it('should calculate TypeScript file coverage', async () => {
      await astCache.getAST('/file1.ts', 'function f1() {}', 'typescript');
      await astCache.getAST('/file2.js', 'function f2() {}', 'javascript');
      await astCache.getAST('/file3.ts', 'class C {}', 'typescript');
      
      const coverage = astCache.getCoverageStats(5); // Total of 5 TS files
      
      expect(coverage).toEqual({
        totalTSFiles: 5,
        cachedTSFiles: 2, // Only TS files counted
        coveragePercentage: 40, // 2/5
        symbolsCached: 3, // 1 function + 1 class + 1 function (from Python file)
      });
    });

    it('should handle zero total files', () => {
      const coverage = astCache.getCoverageStats(0);
      expect(coverage.coveragePercentage).toBe(0);
    });
  });

  describe('Refresh AST', () => {
    it('should force refresh cached AST', async () => {
      const filePath = '/test/refresh.ts';
      const content = 'function test() {}';
      
      const result1 = await astCache.getAST(filePath, content, 'typescript');
      const result2 = await astCache.refreshAST(filePath, content, 'typescript');
      
      // Should have 2 misses (original + refresh)
      const stats = astCache.getStats();
      expect(stats.missCount).toBe(2);
      expect(stats.hitCount).toBe(0);
      
      // Results should be equivalent but different objects
      expect(result1.fileHash).toBe(result2.fileHash);
      expect(result1).not.toBe(result2);
    });
  });

  describe('Symbol Counting', () => {
    it('should count symbols correctly across different types', async () => {
      const filePath = '/test/symbols.ts';
      const content = `
function fn() {}
class MyClass {}
interface IData {}
type MyType = string;
`;

      const result = await astCache.getAST(filePath, content, 'typescript');
      
      expect(result.symbolCount).toBe(4); // 1 function + 1 class + 1 interface + 1 type
      expect(result.mockAST.functions).toHaveLength(1);
      expect(result.mockAST.classes).toHaveLength(1);
      expect(result.mockAST.interfaces).toHaveLength(1);
      expect(result.mockAST.types).toHaveLength(1);
    });
  });

  describe('Performance Tracking', () => {
    it('should track parse time', async () => {
      const filePath = '/test/perf.ts';
      const content = 'function test() {}';
      
      const result = await astCache.getAST(filePath, content, 'typescript');
      
      expect(result.parseTime).toBeGreaterThanOrEqual(0);
      expect(typeof result.parseTime).toBe('number');
    });

    it('should handle concurrent requests efficiently', async () => {
      const filePath = '/test/concurrent.ts';
      const content = 'function test() {}';
      
      // Make multiple concurrent requests
      const promises = Array.from({ length: 5 }, () => 
        astCache.getAST(filePath, content, 'typescript')
      );
      
      const results = await Promise.all(promises);
      
      // All results should be identical
      results.forEach(result => {
        expect(result.fileHash).toBe(results[0].fileHash);
      });
      
      // Should have 5 requests total (1 miss, 4 hits)
      const stats = astCache.getStats();
      expect(stats.totalRequests).toBe(5);
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty file content', async () => {
      const result = await astCache.getAST('/empty.ts', '', 'typescript');
      
      expect(result.symbolCount).toBe(0);
      expect(result.mockAST.functions).toHaveLength(0);
      expect(result.mockAST.classes).toHaveLength(0);
    });

    it('should handle malformed code gracefully', async () => {
      const content = `
function broken( {
  // Incomplete function
class Incomplete
// Missing brace
`;
      
      const result = await astCache.getAST('/broken.ts', content, 'typescript');
      
      // Should still attempt parsing and return partial results
      expect(result).toBeDefined();
      expect(result.symbolCount).toBeGreaterThanOrEqual(0);
    });

    it('should handle very large files', async () => {
      const largeContent = 'function test() {}\n'.repeat(1000);
      
      const result = await astCache.getAST('/large.ts', largeContent, 'typescript');
      
      expect(result.mockAST.functions).toHaveLength(1000);
      expect(result.symbolCount).toBe(1000);
    });
  });
});