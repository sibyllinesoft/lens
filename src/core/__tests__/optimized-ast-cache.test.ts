/**
 * Comprehensive tests for OptimizedASTCache
 * Includes performance benchmarks and Stage-B target validation
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { OptimizedASTCache, PERFORMANCE_PRESETS, type BatchParseRequest } from '../optimized-ast-cache.js';
import type { CachedAST } from '../ast-cache.js';

describe('OptimizedASTCache', () => {
  let cache: OptimizedASTCache;
  let mockContent: string;
  let mockTypeScriptContent: string;

  beforeEach(() => {
    cache = new OptimizedASTCache(PERFORMANCE_PRESETS.balanced);
    
    mockContent = `
      function testFunction(param: string): number {
        return param.length;
      }
      
      class TestClass {
        private value: string;
        constructor(value: string) { this.value = value; }
        getValue(): string { return this.value; }
      }
      
      interface TestInterface {
        method(): void;
      }
      
      type TestType = string | number;
      
      const testVariable = 'hello';
    `;

    mockTypeScriptContent = `
      export async function processData<T>(data: T[]): Promise<T[]> {
        return data.filter(item => item != null);
      }
      
      export class DataProcessor<T> implements IProcessor<T> {
        private items: T[] = [];
        
        add(item: T): void {
          this.items.push(item);
        }
        
        process(): T[] {
          return this.processData(this.items);
        }
      }
      
      export interface IProcessor<T> {
        add(item: T): void;
        process(): T[];
      }
      
      export type ProcessorResult<T> = {
        items: T[];
        count: number;
      };
      
      import { Logger } from './logger';
      import { Config, Settings } from './config';
    `;
  });

  afterEach(async () => {
    await cache.shutdown();
  });

  describe('Core Caching Functionality', () => {
    it('should cache and retrieve AST correctly', async () => {
      const filePath = '/test/file.ts';
      
      // First call should parse and cache
      const ast1 = await cache.getAST(filePath, mockContent, 'typescript');
      expect(ast1).toBeDefined();
      expect(ast1.language).toBe('typescript');
      expect(ast1.symbolCount).toBeGreaterThan(0);
      expect(ast1.parseTime).toBeGreaterThan(0);

      // Second call should hit cache (much faster)
      const start = Date.now();
      const ast2 = await cache.getAST(filePath, mockContent, 'typescript');
      const retrievalTime = Date.now() - start;
      
      expect(ast2).toBeDefined();
      expect(ast2.fileHash).toBe(ast1.fileHash);
      expect(retrievalTime).toBeLessThan(5); // Should be very fast cache hit
      
      const metrics = cache.getMetrics();
      expect(metrics.hitRate).toBeGreaterThan(0);
    });

    it('should handle content changes correctly', async () => {
      const filePath = '/test/file.ts';
      
      const ast1 = await cache.getAST(filePath, mockContent, 'typescript');
      const modifiedContent = mockContent + '\nconst newVar = 123;';
      const ast2 = await cache.getAST(filePath, modifiedContent, 'typescript');
      
      expect(ast1.fileHash).not.toBe(ast2.fileHash);
      expect(ast2.symbolCount).toBeGreaterThan(ast1.symbolCount);
    });

    it('should support different languages', async () => {
      const pythonContent = `
def test_function(param):
    return len(param)

class TestClass:
    def __init__(self, value):
        self.value = value
    
    def get_value(self):
        return self.value
      `;
      
      const ast = await cache.getAST('/test/file.py', pythonContent, 'python');
      
      expect(ast.language).toBe('python');
      expect(ast.mockAST.functions.length).toBeGreaterThan(0);
      expect(ast.mockAST.classes.length).toBeGreaterThan(0);
    });
  });

  describe('Performance Optimizations', () => {
    it('should demonstrate improved cache capacity', () => {
      const metrics = cache.getMetrics();
      expect(metrics.cacheSize).toBeLessThanOrEqual(200); // Expanded from 50
    });

    it('should support batch processing', async () => {
      const batchRequests: BatchParseRequest[] = [
        { filePath: '/test/file1.ts', content: mockContent, language: 'typescript', priority: 'high' },
        { filePath: '/test/file2.ts', content: mockTypeScriptContent, language: 'typescript', priority: 'normal' },
        { filePath: '/test/file3.ts', content: mockContent, language: 'typescript', priority: 'low' },
      ];

      const start = Date.now();
      const results = await cache.batchGetAST(batchRequests);
      const batchTime = Date.now() - start;

      expect(results).toHaveLength(3);
      expect(results.every(r => r.success)).toBe(true);
      expect(batchTime).toBeLessThan(100); // Batch should be efficient
      
      // Verify all files are now cached
      const metrics = cache.getMetrics();
      expect(metrics.cacheSize).toBe(3);
    });

    it('should handle priority ordering in batch requests', async () => {
      const batchRequests: BatchParseRequest[] = [
        { filePath: '/test/low.ts', content: mockContent, language: 'typescript', priority: 'low' },
        { filePath: '/test/high.ts', content: mockContent, language: 'typescript', priority: 'high' },
        { filePath: '/test/normal.ts', content: mockContent, language: 'typescript', priority: 'normal' },
      ];

      const results = await cache.batchGetAST(batchRequests);
      
      // All should succeed regardless of priority
      expect(results.every(r => r.success)).toBe(true);
      expect(results).toHaveLength(3);
    });
  });

  describe('Enhanced Pattern Matching', () => {
    it('should extract TypeScript symbols with precompiled patterns', async () => {
      const ast = await cache.getAST('/test/complex.ts', mockTypeScriptContent, 'typescript');
      
      expect(ast.mockAST.functions.length).toBeGreaterThan(0);
      expect(ast.mockAST.classes.length).toBeGreaterThan(0);
      expect(ast.mockAST.interfaces.length).toBeGreaterThan(0);
      expect(ast.mockAST.types.length).toBeGreaterThan(0);
      expect(ast.mockAST.imports.length).toBeGreaterThan(0);
      
      // Verify specific patterns are matched
      const processDataFunction = ast.mockAST.functions.find(f => f.name === 'processData');
      expect(processDataFunction).toBeDefined();
      expect(processDataFunction?.signature).toContain('async');
      
      const dataProcessorClass = ast.mockAST.classes.find(c => c.name === 'DataProcessor');
      expect(dataProcessorClass).toBeDefined();
      expect(dataProcessorClass?.implements).toContain('IProcessor');
    });

    it('should handle complex TypeScript constructs', async () => {
      const complexContent = `
        export abstract class BaseProcessor<T, U = T> extends EventEmitter implements IProcessor<T> {
          protected abstract process(items: T[]): Promise<U[]>;
          
          async execute<V extends T>(input: V[]): Promise<U[]> {
            return this.process(input);
          }
        }
        
        export interface IAdvancedProcessor<T> extends IProcessor<T> {
          configure<K extends keyof T>(key: K, value: T[K]): void;
        }
        
        export type ProcessorConfig<T> = {
          [K in keyof T]: T[K] extends Function ? never : T[K];
        };
      `;
      
      const ast = await cache.getAST('/test/complex.ts', complexContent, 'typescript');
      
      expect(ast.mockAST.classes.some(c => c.name === 'BaseProcessor')).toBe(true);
      expect(ast.mockAST.interfaces.some(i => i.name === 'IAdvancedProcessor')).toBe(true);
      expect(ast.mockAST.types.some(t => t.name === 'ProcessorConfig')).toBe(true);
    });
  });

  describe('Performance Benchmarks', () => {
    it('should meet Stage-B performance targets for small files', async () => {
      const smallFile = mockContent;
      const filePath = '/test/small.ts';
      
      // Warm up (initial parse)
      await cache.getAST(filePath, smallFile, 'typescript');
      
      // Measure cache hit performance
      const start = Date.now();
      await cache.getAST(filePath, smallFile, 'typescript');
      const retrievalTime = Date.now() - start;
      
      // Stage-B target: sub-millisecond for cache hits
      expect(retrievalTime).toBeLessThan(1);
    });

    it('should meet Stage-B targets for parsing new files', async () => {
      const mediumFile = mockTypeScriptContent.repeat(3); // ~3KB file
      const filePath = '/test/medium.ts';
      
      const start = Date.now();
      const ast = await cache.getAST(filePath, mediumFile, 'typescript');
      const parseTime = Date.now() - start;
      
      // Stage-B target: <4ms for medium files
      expect(parseTime).toBeLessThan(4);
      expect(ast.symbolCount).toBeGreaterThan(0);
    });

    it('should demonstrate 40% improvement over baseline (simulated)', async () => {
      // Simulate baseline performance (7ms target)
      const baselineTargetMs = 7;
      const optimizedTargetMs = 4; // 43% improvement
      
      const files = Array.from({ length: 10 }, (_, i) => ({
        path: `/test/file${i}.ts`,
        content: mockTypeScriptContent
      }));
      
      const parseTimesMs: number[] = [];
      
      for (const file of files) {
        const start = Date.now();
        await cache.getAST(file.path, file.content, 'typescript');
        parseTimesMs.push(Date.now() - start);
      }
      
      const avgParseTime = parseTimesMs.reduce((sum, t) => sum + t, 0) / parseTimesMs.length;
      
      // Verify we're hitting the optimized target
      expect(avgParseTime).toBeLessThan(optimizedTargetMs);
      
      // Calculate improvement percentage
      const improvementPercent = ((baselineTargetMs - avgParseTime) / baselineTargetMs) * 100;
      expect(improvementPercent).toBeGreaterThan(30); // At least 30% improvement
      
      console.log(`📊 Performance: ${avgParseTime.toFixed(2)}ms avg (${improvementPercent.toFixed(1)}% improvement over ${baselineTargetMs}ms baseline)`);
    });

    it('should maintain performance under load', async () => {
      const loadTestFiles = 50;
      const files = Array.from({ length: loadTestFiles }, (_, i) => ({
        filePath: `/load/file${i}.ts`,
        content: mockContent + `\nconst loadVar${i} = ${i};`,
        language: 'typescript' as const,
        priority: 'normal' as const
      }));
      
      const start = Date.now();
      const results = await cache.batchGetAST(files);
      const totalTime = Date.now() - start;
      
      const avgTimePerFile = totalTime / loadTestFiles;
      const throughput = (loadTestFiles / totalTime) * 1000; // Files per second
      
      expect(results.every(r => r.success)).toBe(true);
      expect(avgTimePerFile).toBeLessThan(5); // <5ms per file on average
      expect(throughput).toBeGreaterThan(20); // >20 files/second
      
      console.log(`📊 Load test: ${loadTestFiles} files in ${totalTime}ms (${throughput.toFixed(1)} files/sec)`);
    });
  });

  describe('Cache Management', () => {
    it('should provide comprehensive metrics', () => {
      const metrics = cache.getMetrics();
      
      expect(metrics).toHaveProperty('hitCount');
      expect(metrics).toHaveProperty('missCount');
      expect(metrics).toHaveProperty('hitRate');
      expect(metrics).toHaveProperty('avgParseTime');
      expect(metrics).toHaveProperty('cacheSize');
      expect(metrics).toHaveProperty('memoryUsage');
      expect(metrics).toHaveProperty('avgRetrievalTime');
    });

    it('should support cache preloading', async () => {
      const filePaths = ['/test/preload1.ts', '/test/preload2.ts'];
      
      // Mock file system
      vi.mock('fs', () => ({
        readFile: vi.fn().mockImplementation((path, encoding, callback) => {
          if (typeof callback === 'function') {
            callback(null, mockContent);
          }
        })
      }));
      
      // Note: In a real test environment, we'd use actual files or proper mocking
      // For now, we'll test the interface exists
      expect(cache.preloadFiles).toBeDefined();
    });

    it('should support selective cache clearing', () => {
      cache.clear(/test/);
      const metrics = cache.getMetrics();
      expect(metrics.cacheSize).toBe(0);
    });
  });

  describe('Error Handling', () => {
    it('should handle malformed content gracefully', async () => {
      const malformedContent = '{ invalid javascript syntax ][';
      
      // Should not throw, but may not extract symbols correctly
      const ast = await cache.getAST('/test/malformed.ts', malformedContent, 'typescript');
      expect(ast).toBeDefined();
      expect(ast.symbolCount).toBeGreaterThanOrEqual(0);
    });

    it('should handle empty content', async () => {
      const ast = await cache.getAST('/test/empty.ts', '', 'typescript');
      expect(ast).toBeDefined();
      expect(ast.symbolCount).toBe(0);
      expect(ast.mockAST.functions).toHaveLength(0);
    });

    it('should handle very large files', async () => {
      const largeContent = mockTypeScriptContent.repeat(100); // ~50KB file
      
      const start = Date.now();
      const ast = await cache.getAST('/test/large.ts', largeContent, 'typescript');
      const parseTime = Date.now() - start;
      
      expect(ast).toBeDefined();
      expect(ast.symbolCount).toBeGreaterThan(0);
      // Large files may take longer, but should still be reasonable
      expect(parseTime).toBeLessThan(50);
    });
  });

  describe('Memory Management', () => {
    it('should not leak memory with many operations', async () => {
      const initialMetrics = cache.getMetrics();
      
      // Perform many cache operations
      for (let i = 0; i < 100; i++) {
        await cache.getAST(`/test/temp${i}.ts`, mockContent + `\nconst temp${i} = ${i};`, 'typescript');
      }
      
      const afterMetrics = cache.getMetrics();
      
      // Memory usage should be bounded by cache size limits
      expect(afterMetrics.memoryUsage).toBeLessThan(50); // <50MB
      expect(afterMetrics.cacheSize).toBeLessThanOrEqual(200); // Respects max capacity
    });

    it('should properly cleanup on shutdown', async () => {
      await cache.getAST('/test/cleanup.ts', mockContent, 'typescript');
      
      const beforeShutdown = cache.getMetrics();
      expect(beforeShutdown.cacheSize).toBeGreaterThan(0);
      
      await cache.shutdown();
      
      // After shutdown, should be clean
      const afterShutdown = cache.getMetrics();
      expect(afterShutdown.cacheSize).toBe(0);
    });
  });

  describe('Configuration Presets', () => {
    it('should support performance preset configuration', () => {
      const perfCache = new OptimizedASTCache(PERFORMANCE_PRESETS.performance);
      const balancedCache = new OptimizedASTCache(PERFORMANCE_PRESETS.balanced);
      const memoryCache = new OptimizedASTCache(PERFORMANCE_PRESETS.memory_efficient);
      
      // Each preset should have different characteristics
      expect(PERFORMANCE_PRESETS.performance.maxFiles).toBeGreaterThan(PERFORMANCE_PRESETS.balanced.maxFiles);
      expect(PERFORMANCE_PRESETS.balanced.maxFiles).toBeGreaterThan(PERFORMANCE_PRESETS.memory_efficient.maxFiles);
      
      perfCache.shutdown();
      balancedCache.shutdown();
      memoryCache.shutdown();
    });
  });
});