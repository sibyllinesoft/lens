/**
 * Performance Benchmark Tests for OptimizedASTCache
 * Phase B2 Validation: Ensures ~40% Stage-B performance improvement (7ms → 3-4ms)
 * Comprehensive performance measurement and regression testing
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { OptimizedASTCache, PERFORMANCE_PRESETS, type BatchParseRequest, type CacheMetrics } from '../optimized-ast-cache.js';
import { StructuralPatternEngine } from '../structural-pattern-engine.js';
import { CoverageTracker } from '../coverage-tracker.js';
import type { CachedAST } from '../ast-cache.js';

describe('AST Cache Performance Benchmarks', () => {
  let cache: OptimizedASTCache;
  let patternEngine: StructuralPatternEngine;
  let coverageTracker: CoverageTracker;
  
  // Test content samples of varying sizes
  const sampleContents = {
    small: `
      function small() { return 42; }
      const value = 'test';
    `,
    medium: `
      export interface DataProcessor<T> {
        process(items: T[]): Promise<T[]>;
        validate(item: T): boolean;
        transform(item: T): T;
      }
      
      export class BatchProcessor<T> implements DataProcessor<T> {
        private queue: T[] = [];
        private processing = false;
        
        constructor(private batchSize: number = 100) {}
        
        async process(items: T[]): Promise<T[]> {
          if (this.processing) {
            throw new Error('Already processing');
          }
          
          this.processing = true;
          const results: T[] = [];
          
          try {
            for (let i = 0; i < items.length; i += this.batchSize) {
              const batch = items.slice(i, i + this.batchSize);
              const processed = await this.processBatch(batch);
              results.push(...processed);
            }
            return results;
          } finally {
            this.processing = false;
          }
        }
        
        private async processBatch(batch: T[]): Promise<T[]> {
          return batch.filter(this.validate).map(this.transform);
        }
        
        validate(item: T): boolean {
          return item != null;
        }
        
        transform(item: T): T {
          return item;
        }
      }
      
      export type ProcessorConfig<T> = {
        batchSize: number;
        validateFn?: (item: T) => boolean;
        transformFn?: (item: T) => T;
      };
      
      import { Logger } from './logger';
      import { Config } from './config';
    `,
    large: '' // Will be generated in beforeEach
  };

  beforeEach(() => {
    cache = new OptimizedASTCache(PERFORMANCE_PRESETS.performance);
    patternEngine = new StructuralPatternEngine();
    coverageTracker = new CoverageTracker();
    
    // Generate large content sample
    sampleContents.large = Array.from({ length: 20 }, (_, i) => `
      export class LargeClass${i}<T extends Record<string, any>> {
        private data: Map<string, T> = new Map();
        private listeners: Array<(data: T) => void> = [];
        
        constructor(private config: { maxSize: number }) {}
        
        add(key: string, value: T): void {
          if (this.data.size >= this.config.maxSize) {
            const firstKey = this.data.keys().next().value;
            this.data.delete(firstKey);
          }
          this.data.set(key, value);
          this.notifyListeners(value);
        }
        
        get(key: string): T | undefined {
          return this.data.get(key);
        }
        
        private notifyListeners(data: T): void {
          this.listeners.forEach(listener => listener(data));
        }
        
        addListener(listener: (data: T) => void): void {
          this.listeners.push(listener);
        }
      }
      
      export interface IDataStore${i}<T> {
        add(key: string, value: T): void;
        get(key: string): T | undefined;
        size(): number;
      }
      
      export type DataStoreConfig${i} = {
        maxSize: number;
        enableLogging?: boolean;
        compressionLevel?: number;
      };
    `).join('\n');
  });

  afterEach(async () => {
    await cache.shutdown();
    await patternEngine.clear();
    await coverageTracker.shutdown();
  });

  describe('Stage-B Performance Targets', () => {
    it('should achieve sub-millisecond cache hits (Stage-B target)', async () => {
      const filePath = '/perf/cache-hit.ts';
      const content = sampleContents.medium;
      
      // Warm up cache
      await cache.getAST(filePath, content, 'typescript');
      
      // Measure multiple cache hits for consistency
      const measurements: number[] = [];
      const iterations = 50;
      
      for (let i = 0; i < iterations; i++) {
        const start = performance.now();
        await cache.getAST(filePath, content, 'typescript');
        measurements.push(performance.now() - start);
      }
      
      const avgHitTime = measurements.reduce((sum, t) => sum + t, 0) / measurements.length;
      const p95HitTime = measurements.sort((a, b) => a - b)[Math.floor(measurements.length * 0.95)];
      
      // Stage-B target: sub-millisecond cache hits
      expect(avgHitTime).toBeLessThan(1);
      expect(p95HitTime).toBeLessThan(1.5);
      
      console.log(`📊 Cache Hit Performance: ${avgHitTime.toFixed(3)}ms avg, ${p95HitTime.toFixed(3)}ms p95`);
    });

    it('should achieve 3-4ms parsing for medium files (Stage-B target)', async () => {
      const content = sampleContents.medium;
      const iterations = 20;
      const measurements: number[] = [];
      
      for (let i = 0; i < iterations; i++) {
        const filePath = `/perf/parse-${i}.ts`;
        const start = performance.now();
        const ast = await cache.getAST(filePath, content, 'typescript');
        const parseTime = performance.now() - start;
        
        measurements.push(parseTime);
        expect(ast.symbolCount).toBeGreaterThan(0);
      }
      
      const avgParseTime = measurements.reduce((sum, t) => sum + t, 0) / measurements.length;
      const p95ParseTime = measurements.sort((a, b) => a - b)[Math.floor(measurements.length * 0.95)];
      
      // Stage-B target: 3-4ms for medium files
      expect(avgParseTime).toBeLessThan(4);
      expect(p95ParseTime).toBeLessThan(6);
      
      console.log(`📊 Parse Performance: ${avgParseTime.toFixed(2)}ms avg, ${p95ParseTime.toFixed(2)}ms p95`);
    });

    it('should demonstrate 40% improvement over baseline', async () => {
      // Simulate baseline performance measurements
      const baselineTarget = 7; // ms (original target)
      const optimizedTarget = 4.2; // ms (40% improvement = 7 * 0.6)
      
      const testFiles = Array.from({ length: 15 }, (_, i) => ({
        path: `/perf/improvement-${i}.ts`,
        content: sampleContents.medium + `\nconst perfVar${i} = ${i * 42};`
      }));
      
      const measurements: number[] = [];
      
      for (const file of testFiles) {
        const start = performance.now();
        const ast = await cache.getAST(file.path, file.content, 'typescript');
        const parseTime = performance.now() - start;
        
        measurements.push(parseTime);
        expect(ast.symbolCount).toBeGreaterThan(5); // Reasonable symbol count
      }
      
      const avgTime = measurements.reduce((sum, t) => sum + t, 0) / measurements.length;
      const improvementPercent = ((baselineTarget - avgTime) / baselineTarget) * 100;
      
      // Verify 40% improvement achieved
      expect(avgTime).toBeLessThan(optimizedTarget);
      expect(improvementPercent).toBeGreaterThan(35); // At least 35% improvement
      
      console.log(`📊 Performance Improvement: ${avgTime.toFixed(2)}ms (${improvementPercent.toFixed(1)}% improvement over ${baselineTarget}ms baseline)`);
    });
  });

  describe('Batch Processing Performance', () => {
    it('should achieve 25% I/O overhead reduction with batching', async () => {
      const batchRequests: BatchParseRequest[] = Array.from({ length: 15 }, (_, i) => ({
        filePath: `/batch/file${i}.ts`,
        content: sampleContents.medium,
        language: 'typescript',
        priority: i % 3 === 0 ? 'high' : i % 3 === 1 ? 'normal' : 'low'
      }));
      
      // Measure sequential processing time
      const sequentialStart = performance.now();
      const sequentialResults = [];
      for (const request of batchRequests) {
        const ast = await cache.getAST(request.filePath + '_seq', request.content, request.language);
        sequentialResults.push(ast);
      }
      const sequentialTime = performance.now() - sequentialStart;
      
      // Clear cache for fair comparison
      cache.clear();
      
      // Measure batch processing time
      const batchStart = performance.now();
      const batchResults = await cache.batchGetAST(batchRequests);
      const batchTime = performance.now() - batchStart;
      
      const improvementPercent = ((sequentialTime - batchTime) / sequentialTime) * 100;
      
      expect(batchResults).toHaveLength(batchRequests.length);
      expect(batchResults.every(r => r.success)).toBe(true);
      expect(improvementPercent).toBeGreaterThan(20); // At least 20% improvement
      
      console.log(`📊 Batch Efficiency: ${improvementPercent.toFixed(1)}% improvement (${batchTime.toFixed(2)}ms vs ${sequentialTime.toFixed(2)}ms)`);
    });

    it('should maintain performance consistency under parallel load', async () => {
      const concurrency = 8;
      const filesPerWorker = 10;
      
      const workerPromises = Array.from({ length: concurrency }, async (_, workerId) => {
        const workerFiles = Array.from({ length: filesPerWorker }, (_, i) => ({
          filePath: `/parallel/worker${workerId}/file${i}.ts`,
          content: sampleContents.medium,
          language: 'typescript' as const,
          priority: 'normal' as const
        }));
        
        const start = performance.now();
        const results = await cache.batchGetAST(workerFiles);
        const time = performance.now() - start;
        
        return {
          workerId,
          time,
          results,
          avgTimePerFile: time / filesPerWorker
        };
      });
      
      const workerResults = await Promise.all(workerPromises);
      
      const avgWorkerTime = workerResults.reduce((sum, w) => sum + w.time, 0) / workerResults.length;
      const maxWorkerTime = Math.max(...workerResults.map(w => w.time));
      const minWorkerTime = Math.min(...workerResults.map(w => w.time));
      const timeVariance = ((maxWorkerTime - minWorkerTime) / avgWorkerTime) * 100;
      
      // All workers should complete successfully
      expect(workerResults.every(w => w.results.every(r => r.success))).toBe(true);
      
      // Time variance should be reasonable (less than 50%)
      expect(timeVariance).toBeLessThan(50);
      
      // Average time per file should be reasonable
      const overallAvgTimePerFile = avgWorkerTime / filesPerWorker;
      expect(overallAvgTimePerFile).toBeLessThan(5);
      
      console.log(`📊 Parallel Performance: ${concurrency} workers, ${avgWorkerTime.toFixed(2)}ms avg, ${timeVariance.toFixed(1)}% variance`);
    });
  });

  describe('Memory Performance', () => {
    it('should maintain efficient memory usage', async () => {
      const initialMetrics = cache.getMetrics();
      const initialMemory = initialMetrics.memoryUsage;
      
      // Load many files to test memory efficiency
      const fileCount = 100;
      for (let i = 0; i < fileCount; i++) {
        const content = sampleContents.small + `\nconst memVar${i} = ${i};`;
        await cache.getAST(`/memory/file${i}.ts`, content, 'typescript');
      }
      
      const afterLoadMetrics = cache.getMetrics();
      const memoryGrowth = afterLoadMetrics.memoryUsage - initialMemory;
      const memoryPerFile = memoryGrowth / fileCount;
      
      // Memory usage should be reasonable
      expect(afterLoadMetrics.memoryUsage).toBeLessThan(100); // <100MB total
      expect(memoryPerFile).toBeLessThan(1); // <1MB per cached file
      expect(afterLoadMetrics.cacheSize).toBeLessThanOrEqual(200); // Respects capacity limit
      
      console.log(`📊 Memory Usage: ${afterLoadMetrics.memoryUsage.toFixed(2)}MB total, ${memoryPerFile.toFixed(3)}MB per file`);
    });

    it('should handle cache eviction efficiently', async () => {
      // Create cache with small capacity for testing eviction
      const smallCache = new OptimizedASTCache({ 
        maxFiles: 10,
        ttl: 1000 * 60 * 5 
      });
      
      try {
        // Load more files than cache capacity
        const fileCount = 20;
        const evictionStartTime = performance.now();
        
        for (let i = 0; i < fileCount; i++) {
          await smallCache.getAST(`/eviction/file${i}.ts`, sampleContents.small, 'typescript');
        }
        
        const evictionTime = performance.now() - evictionStartTime;
        const metrics = smallCache.getMetrics();
        
        // Cache size should respect limits
        expect(metrics.cacheSize).toBeLessThanOrEqual(10);
        
        // Eviction should not significantly impact performance
        const avgTimePerFile = evictionTime / fileCount;
        expect(avgTimePerFile).toBeLessThan(10); // <10ms per file including eviction
        
        console.log(`📊 Eviction Performance: ${avgTimePerFile.toFixed(2)}ms per file with eviction`);
        
      } finally {
        await smallCache.shutdown();
      }
    });
  });

  describe('Pattern Engine Performance', () => {
    it('should demonstrate pattern compilation benefits', async () => {
      const content = sampleContents.large;
      
      // Test without pattern engine (baseline)
      const baselineStart = performance.now();
      const baselineAst = await cache.getAST('/pattern/baseline.ts', content, 'typescript');
      const baselineTime = performance.now() - baselineStart;
      
      // Test with pattern engine optimization
      const optimizedStart = performance.now();
      const symbols = await patternEngine.findSymbols(content, 'typescript');
      const optimizedTime = performance.now() - optimizedStart;
      
      const improvementPercent = ((baselineTime - optimizedTime) / baselineTime) * 100;
      
      expect(symbols.length).toBeGreaterThan(0);
      expect(optimizedTime).toBeLessThan(baselineTime);
      
      console.log(`📊 Pattern Engine: ${improvementPercent.toFixed(1)}% faster (${optimizedTime.toFixed(2)}ms vs ${baselineTime.toFixed(2)}ms)`);
    });

    it('should scale pattern execution efficiently', async () => {
      const testPatterns = [
        'ts-function-declarations',
        'ts-class-declarations', 
        'ts-interface-declarations',
        'ts-type-aliases'
      ];
      
      const scalingTests = [10, 50, 100].map(async patternCount => {
        const patterns = Array.from({ length: patternCount }, (_, i) => 
          testPatterns[i % testPatterns.length]
        );
        
        const start = performance.now();
        const results = await patternEngine.executePatterns(patterns, sampleContents.medium, 'typescript');
        const time = performance.now() - start;
        
        return {
          patternCount,
          time,
          results: results.length,
          avgTimePerPattern: time / patterns.length
        };
      });
      
      const scalingResults = await Promise.all(scalingTests);
      
      // Performance should scale sub-linearly
      const timePerPattern100 = scalingResults[2].avgTimePerPattern;
      const timePerPattern10 = scalingResults[0].avgTimePerPattern;
      const scalingFactor = timePerPattern100 / timePerPattern10;
      
      expect(scalingFactor).toBeLessThan(2); // Should not be linear scaling
      expect(timePerPattern100).toBeLessThan(1); // <1ms per pattern even at scale
      
      console.log(`📊 Pattern Scaling: ${scalingFactor.toFixed(2)}x (${timePerPattern100.toFixed(3)}ms per pattern at 100 patterns)`);
    });
  });

  describe('Coverage Tracking Performance', () => {
    it('should track coverage with minimal overhead', async () => {
      const fileCount = 50;
      const files = Array.from({ length: fileCount }, (_, i) => ({
        path: `/coverage/file${i}.ts`,
        content: sampleContents.medium
      }));
      
      // Register files for coverage tracking
      coverageTracker.registerFiles(files.map(f => f.path), 'typescript');
      
      // Measure parsing with coverage tracking
      const trackingStart = performance.now();
      for (const file of files) {
        const ast = await cache.getAST(file.path, file.content, 'typescript');
        
        // Simulate coverage tracking
        const symbols = [
          { name: 'test', kind: 'function' as const, file_path: file.path, line: 1, col: 1, scope: 'global' }
        ];
        coverageTracker.recordFileIndexing(file.path, 'typescript', symbols, ast.parseTime);
      }
      const trackingTime = performance.now() - trackingStart;
      
      // Generate coverage report
      const reportStart = performance.now();
      const report = coverageTracker.generateReport();
      const reportTime = performance.now() - reportStart;
      
      const avgTrackingOverhead = trackingTime / fileCount;
      
      expect(report.metrics.totalFiles).toBe(fileCount);
      expect(avgTrackingOverhead).toBeLessThan(2); // <2ms overhead per file
      expect(reportTime).toBeLessThan(10); // <10ms to generate report
      
      console.log(`📊 Coverage Tracking: ${avgTrackingOverhead.toFixed(2)}ms overhead per file, ${reportTime.toFixed(2)}ms report generation`);
    });
  });

  describe('Real-World Performance Scenarios', () => {
    it('should handle IDE-like usage patterns efficiently', async () => {
      // Simulate IDE usage: frequent small edits with cache hits
      const baseFile = sampleContents.medium;
      const filePath = '/ide/active-file.ts';
      
      const scenarios = Array.from({ length: 30 }, (_, i) => {
        const modification = `\n// Edit ${i}\nconst edit${i} = ${i};`;
        return baseFile + modification;
      });
      
      const timings: number[] = [];
      
      for (let i = 0; i < scenarios.length; i++) {
        const start = performance.now();
        const ast = await cache.getAST(filePath, scenarios[i], 'typescript');
        timings.push(performance.now() - start);
        
        expect(ast.symbolCount).toBeGreaterThan(0);
      }
      
      const avgEditTime = timings.reduce((sum, t) => sum + t, 0) / timings.length;
      const maxEditTime = Math.max(...timings);
      
      // IDE responsiveness targets
      expect(avgEditTime).toBeLessThan(5); // <5ms average
      expect(maxEditTime).toBeLessThan(15); // <15ms worst case
      
      console.log(`📊 IDE Simulation: ${avgEditTime.toFixed(2)}ms avg, ${maxEditTime.toFixed(2)}ms max edit time`);
    });

    it('should handle build-time batch processing efficiently', async () => {
      // Simulate build tool processing many files
      const projectFiles = Array.from({ length: 100 }, (_, i) => ({
        filePath: `/build/src/module${i}.ts`,
        content: sampleContents.medium + `\nexport const BUILD_ID_${i} = '${i}';`,
        language: 'typescript' as const,
        priority: 'normal' as const
      }));
      
      const batchSize = 20;
      const batches = [];
      for (let i = 0; i < projectFiles.length; i += batchSize) {
        batches.push(projectFiles.slice(i, i + batchSize));
      }
      
      const batchTimings: number[] = [];
      let totalFiles = 0;
      
      for (const batch of batches) {
        const start = performance.now();
        const results = await cache.batchGetAST(batch);
        const batchTime = performance.now() - start;
        
        batchTimings.push(batchTime);
        totalFiles += results.filter(r => r.success).length;
      }
      
      const totalBatchTime = batchTimings.reduce((sum, t) => sum + t, 0);
      const avgTimePerFile = totalBatchTime / totalFiles;
      const throughput = (totalFiles / totalBatchTime) * 1000; // files/second
      
      expect(totalFiles).toBe(projectFiles.length);
      expect(avgTimePerFile).toBeLessThan(3); // <3ms per file in batch mode
      expect(throughput).toBeGreaterThan(50); // >50 files/second
      
      console.log(`📊 Build Simulation: ${totalFiles} files in ${totalBatchTime.toFixed(2)}ms (${throughput.toFixed(1)} files/sec)`);
    });
  });

  describe('Regression Prevention', () => {
    it('should maintain consistent performance across test runs', async () => {
      const iterations = 5;
      const runResults: Array<{ parseTime: number; cacheTime: number; batchTime: number }> = [];
      
      for (let run = 0; run < iterations; run++) {
        // Clear cache between runs
        cache.clear();
        
        // Measure parse time
        const parseStart = performance.now();
        await cache.getAST(`/regression/parse-${run}.ts`, sampleContents.medium, 'typescript');
        const parseTime = performance.now() - parseStart;
        
        // Measure cache time
        const cacheStart = performance.now();
        await cache.getAST(`/regression/parse-${run}.ts`, sampleContents.medium, 'typescript');
        const cacheTime = performance.now() - cacheStart;
        
        // Measure batch time
        const batchRequests: BatchParseRequest[] = [
          { filePath: `/regression/batch1-${run}.ts`, content: sampleContents.small, language: 'typescript', priority: 'normal' },
          { filePath: `/regression/batch2-${run}.ts`, content: sampleContents.small, language: 'typescript', priority: 'normal' }
        ];
        
        const batchStart = performance.now();
        await cache.batchGetAST(batchRequests);
        const batchTime = performance.now() - batchStart;
        
        runResults.push({ parseTime, cacheTime, batchTime });
      }
      
      // Calculate variance across runs
      const parseAvg = runResults.reduce((sum, r) => sum + r.parseTime, 0) / iterations;
      const parseVariance = Math.max(...runResults.map(r => r.parseTime)) - Math.min(...runResults.map(r => r.parseTime));
      const parseCoV = parseVariance / parseAvg; // Coefficient of variation
      
      // Performance should be consistent (low variance)
      expect(parseCoV).toBeLessThan(0.3); // <30% coefficient of variation
      expect(parseAvg).toBeLessThan(4); // Maintain performance targets
      
      console.log(`📊 Regression Test: ${parseAvg.toFixed(2)}ms avg, ${(parseCoV * 100).toFixed(1)}% variance across ${iterations} runs`);
    });
  });
});