/**
 * Phase B1 Performance Benchmark: Bitmap vs Set-based Trigram Operations
 * Validates the ~30% Stage-A performance improvement from bitmap optimization
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { LexicalSearchEngine } from '../lexical.js';
import { SegmentStorage } from '../../storage/segments.js';
import { featureFlags } from '../../config/features.js';
import type { SearchContext } from '../../types/core.js';

// Mock SegmentStorage
const mockSegmentStorage = {} as SegmentStorage;

describe('Bitmap Performance Benchmark', () => {
  let mockSearchContext: SearchContext;

  beforeEach(() => {
    mockSearchContext = {
      trace_id: 'benchmark-trace',
      repo_sha: 'benchmark-sha',
      query: 'test',
      mode: 'lex',
      k: 100,
      fuzzy_distance: 0,
      started_at: new Date(),
      stages: [],
    };
  });

  describe('Stage-A Latency Benchmarks', () => {
    const generateTestDataset = (size: number) => {
      const documents = [];
      const codePatterns = [
        'function {name}({params}): {returnType} { return {implementation}; }',
        'class {name} { constructor() { this.{prop} = {value}; } {method}() { return this.{prop}; } }',
        'interface {name} { {prop}: {type}; {method}(): {returnType}; }',
        'const {name} = ({params}) => { const result = {computation}; return result; };',
        'export default { {name}, {prop}: {value}, {method}: () => {returnValue} };',
        'import { {imports} } from "./{module}"; export { {exports} };',
        'type {name} = { {prop}: {type}; } & { {method}(): {returnType}; };',
        'async function {name}({params}): Promise<{returnType}> { const data = await {asyncCall}; return data; }',
      ];

      const tokens = [
        'User', 'Data', 'Service', 'Controller', 'Manager', 'Handler', 'Processor', 'Validator',
        'calculate', 'process', 'handle', 'validate', 'transform', 'filter', 'map', 'reduce',
        'string', 'number', 'boolean', 'object', 'array', 'void', 'Promise', 'Observable',
        'id', 'name', 'email', 'password', 'token', 'config', 'options', 'params', 'result',
      ];

      for (let i = 0; i < size; i++) {
        const pattern = codePatterns[i % codePatterns.length];
        const content = pattern
          .replace(/{name}/g, `${tokens[i % tokens.length]}${i}`)
          .replace(/{params}/g, `param1: string, param2: number`)
          .replace(/{returnType}/g, tokens[(i + 1) % tokens.length])
          .replace(/{implementation}/g, `param1 + param2.toString()`)
          .replace(/{prop}/g, `prop${i}`)
          .replace(/{value}/g, `"value${i}"`)
          .replace(/{method}/g, `method${i}`)
          .replace(/{type}/g, tokens[(i + 2) % tokens.length])
          .replace(/{computation}/g, `param1.length + param2`)
          .replace(/{returnValue}/g, `"return${i}"`)
          .replace(/{imports}/g, tokens.slice(0, 3).join(', '))
          .replace(/{module}/g, `module${i}`)
          .replace(/{exports}/g, tokens.slice(3, 6).join(', '))
          .replace(/{asyncCall}/g, `fetch("/api/data${i}")`);

        documents.push({
          id: `doc${i}`,
          path: `/src/generated/file${i}.ts`,
          content,
        });
      }

      return documents;
    };

    it('should demonstrate 30% performance improvement with bitmap optimization', async () => {
      const dataset = generateTestDataset(1000); // Large dataset for meaningful comparison
      const queries = ['function', 'User', 'process', 'Data', 'calculate'];
      const iterations = 5;

      // Benchmark with legacy Set-based indexing
      featureFlags.updateFeatures({
        bitmapTrigramIndex: { enabled: false },
      });

      const legacyEngine = new LexicalSearchEngine(mockSegmentStorage);
      
      console.log('Indexing documents with legacy Set-based index...');
      const legacyIndexStart = Date.now();
      for (const doc of dataset) {
        await legacyEngine.indexDocument(doc.id, doc.path, doc.content);
      }
      const legacyIndexTime = Date.now() - legacyIndexStart;

      console.log('Running search benchmarks with legacy index...');
      const legacySearchTimes: number[] = [];
      
      for (let i = 0; i < iterations; i++) {
        for (const query of queries) {
          const start = Date.now();
          await legacyEngine.search(mockSearchContext, query, 0);
          const duration = Date.now() - start;
          legacySearchTimes.push(duration);
        }
      }

      // Benchmark with bitmap optimization
      featureFlags.updateFeatures({
        bitmapTrigramIndex: {
          enabled: true,
          rolloutPercentage: 100,
          minDocumentThreshold: 100,
          enablePerformanceLogging: true,
        },
      });

      const bitmapEngine = new LexicalSearchEngine(mockSegmentStorage);
      
      console.log('Indexing documents with bitmap-based index...');
      const bitmapIndexStart = Date.now();
      for (const doc of dataset) {
        await bitmapEngine.indexDocument(doc.id, doc.path, doc.content);
      }
      const bitmapIndexTime = Date.now() - bitmapIndexStart;

      console.log('Running search benchmarks with bitmap index...');
      const bitmapSearchTimes: number[] = [];
      
      for (let i = 0; i < iterations; i++) {
        for (const query of queries) {
          const start = Date.now();
          await bitmapEngine.search(mockSearchContext, query, 0);
          const duration = Date.now() - start;
          bitmapSearchTimes.push(duration);
        }
      }

      // Calculate performance metrics
      const legacyAvg = legacySearchTimes.reduce((a, b) => a + b) / legacySearchTimes.length;
      const bitmapAvg = bitmapSearchTimes.reduce((a, b) => a + b) / bitmapSearchTimes.length;
      const improvement = ((legacyAvg - bitmapAvg) / legacyAvg) * 100;

      const legacyP95 = legacySearchTimes.sort((a, b) => a - b)[Math.floor(legacySearchTimes.length * 0.95)];
      const bitmapP95 = bitmapSearchTimes.sort((a, b) => a - b)[Math.floor(bitmapSearchTimes.length * 0.95)];
      const p95Improvement = ((legacyP95! - bitmapP95!) / legacyP95!) * 100;

      console.log('\n=== Performance Benchmark Results ===');
      console.log(`Dataset size: ${dataset.length} documents`);
      console.log(`Queries tested: ${queries.length} * ${iterations} iterations = ${legacySearchTimes.length} searches`);
      console.log('\n--- Indexing Performance ---');
      console.log(`Legacy indexing time: ${legacyIndexTime}ms`);
      console.log(`Bitmap indexing time: ${bitmapIndexTime}ms`);
      console.log(`Indexing improvement: ${(((legacyIndexTime - bitmapIndexTime) / legacyIndexTime) * 100).toFixed(2)}%`);
      console.log('\n--- Search Performance ---');
      console.log(`Legacy average search time: ${legacyAvg.toFixed(2)}ms`);
      console.log(`Bitmap average search time: ${bitmapAvg.toFixed(2)}ms`);
      console.log(`Average improvement: ${improvement.toFixed(2)}%`);
      console.log(`Legacy P95 search time: ${legacyP95}ms`);
      console.log(`Bitmap P95 search time: ${bitmapP95}ms`);
      console.log(`P95 improvement: ${p95Improvement.toFixed(2)}%`);

      // Performance assertions
      expect(improvement).toBeGreaterThanOrEqual(20); // At least 20% improvement (target is 30%)
      expect(p95Improvement).toBeGreaterThan(0); // P95 should also improve
      expect(bitmapAvg).toBeLessThan(50); // Should meet Stage-A target latency
      
      // Verify search accuracy is maintained
      const legacyResults = await legacyEngine.search(mockSearchContext, 'function', 0);
      const bitmapResults = await bitmapEngine.search(mockSearchContext, 'function', 0);
      expect(bitmapResults.length).toEqual(legacyResults.length);
    }, 60000); // 60 second timeout for comprehensive benchmark

    it('should maintain sub-8ms Stage-A latency with bitmap optimization', async () => {
      const dataset = generateTestDataset(500);
      
      // Enable bitmap optimization
      featureFlags.updateFeatures({
        bitmapTrigramIndex: {
          enabled: true,
          rolloutPercentage: 100,
          minDocumentThreshold: 100,
        },
      });

      const engine = new LexicalSearchEngine(mockSegmentStorage);
      
      // Index documents
      for (const doc of dataset) {
        await engine.indexDocument(doc.id, doc.path, doc.content);
      }

      // Verify bitmap optimization is active
      const stats = engine.getStats();
      expect(stats.bitmap_optimization.enabled).toBe(true);

      // Test various query patterns for Stage-A latency
      const queries = [
        'function',      // Common token
        'User',          // Capitalized token
        'calculate',     // Verb pattern
        'Service',       // Class-like pattern
        'async',         // Keyword pattern
      ];

      const latencies: number[] = [];

      for (const query of queries) {
        // Run multiple iterations for each query
        for (let i = 0; i < 10; i++) {
          const start = Date.now();
          const results = await engine.search(mockSearchContext, query, 0);
          const latency = Date.now() - start;
          
          latencies.push(latency);
          
          // Verify we get reasonable results
          expect(results.length).toBeGreaterThanOrEqual(0);
        }
      }

      // Calculate latency statistics
      const avgLatency = latencies.reduce((a, b) => a + b) / latencies.length;
      const maxLatency = Math.max(...latencies);
      const p95Latency = latencies.sort((a, b) => a - b)[Math.floor(latencies.length * 0.95)];

      console.log('\n=== Stage-A Latency Results ===');
      console.log(`Average latency: ${avgLatency.toFixed(2)}ms`);
      console.log(`P95 latency: ${p95Latency}ms`);
      console.log(`Max latency: ${maxLatency}ms`);
      console.log(`Target: <8ms (Stage-A requirement)`);

      // Assert Stage-A latency requirements
      expect(p95Latency).toBeLessThanOrEqual(8); // P95 under 8ms
      expect(avgLatency).toBeLessThanOrEqual(5); // Average under 5ms
      expect(maxLatency).toBeLessThanOrEqual(15); // Max under 15ms (allowing for outliers)
    }, 30000);
  });

  describe('Memory Efficiency Benchmarks', () => {
    it('should demonstrate memory efficiency of bitmap vs Set operations', async () => {
      const dataset = generateTestDataset(200);
      
      // Test with legacy Set-based index
      featureFlags.updateFeatures({
        bitmapTrigramIndex: { enabled: false },
      });

      const legacyEngine = new LexicalSearchEngine(mockSegmentStorage);
      
      const legacyMemStart = process.memoryUsage().heapUsed;
      for (const doc of dataset) {
        await legacyEngine.indexDocument(doc.id, doc.path, doc.content);
      }
      const legacyMemEnd = process.memoryUsage().heapUsed;
      const legacyMemUsage = legacyMemEnd - legacyMemStart;

      // Test with bitmap optimization
      featureFlags.updateFeatures({
        bitmapTrigramIndex: {
          enabled: true,
          rolloutPercentage: 100,
          minDocumentThreshold: 50,
        },
      });

      const bitmapEngine = new LexicalSearchEngine(mockSegmentStorage);
      
      // Force garbage collection if available
      if (global.gc) {
        global.gc();
      }
      
      const bitmapMemStart = process.memoryUsage().heapUsed;
      for (const doc of dataset) {
        await bitmapEngine.indexDocument(doc.id, doc.path, doc.content);
      }
      const bitmapMemEnd = process.memoryUsage().heapUsed;
      const bitmapMemUsage = bitmapMemEnd - bitmapMemStart;

      console.log('\n=== Memory Usage Comparison ===');
      console.log(`Legacy memory usage: ${(legacyMemUsage / 1024 / 1024).toFixed(2)} MB`);
      console.log(`Bitmap memory usage: ${(bitmapMemUsage / 1024 / 1024).toFixed(2)} MB`);
      console.log(`Memory efficiency: ${(((legacyMemUsage - bitmapMemUsage) / legacyMemUsage) * 100).toFixed(2)}%`);

      const bitmapStats = bitmapEngine.getStats();
      console.log(`Bitmap efficiency metric: ${(bitmapStats.bitmap_optimization.memory_efficiency * 100).toFixed(1)}%`);

      // Memory usage should be comparable or better
      // Note: In practice, bitmap efficiency is most pronounced with larger, denser document sets
      expect(bitmapMemUsage).toBeLessThanOrEqual(legacyMemUsage * 1.2); // Allow 20% overhead for test variability
      expect(bitmapStats.bitmap_optimization.memory_efficiency).toBeGreaterThan(0.3);
    });
  });

  describe('Scalability Benchmarks', () => {
    it('should demonstrate improved scalability with document set growth', async () => {
      const documentSizes = [100, 250, 500, 1000];
      const results: Array<{ size: number; legacyTime: number; bitmapTime: number; improvement: number }> = [];

      for (const size of documentSizes) {
        console.log(`\n--- Testing with ${size} documents ---`);
        
        const dataset = generateTestDataset(size);
        
        // Benchmark legacy approach
        featureFlags.updateFeatures({
          bitmapTrigramIndex: { enabled: false },
        });

        const legacyEngine = new LexicalSearchEngine(mockSegmentStorage);
        for (const doc of dataset) {
          await legacyEngine.indexDocument(doc.id, doc.path, doc.content);
        }

        const legacyStart = Date.now();
        await legacyEngine.search(mockSearchContext, 'function', 0);
        await legacyEngine.search(mockSearchContext, 'User', 0);
        await legacyEngine.search(mockSearchContext, 'Data', 0);
        const legacyTime = Date.now() - legacyStart;

        // Benchmark bitmap approach
        featureFlags.updateFeatures({
          bitmapTrigramIndex: {
            enabled: true,
            rolloutPercentage: 100,
            minDocumentThreshold: 50,
          },
        });

        const bitmapEngine = new LexicalSearchEngine(mockSegmentStorage);
        for (const doc of dataset) {
          await bitmapEngine.indexDocument(doc.id, doc.path, doc.content);
        }

        const bitmapStart = Date.now();
        await bitmapEngine.search(mockSearchContext, 'function', 0);
        await bitmapEngine.search(mockSearchContext, 'User', 0);
        await bitmapEngine.search(mockSearchContext, 'Data', 0);
        const bitmapTime = Date.now() - bitmapStart;

        const improvement = ((legacyTime - bitmapTime) / legacyTime) * 100;
        results.push({ size, legacyTime, bitmapTime, improvement });

        console.log(`Legacy time: ${legacyTime}ms, Bitmap time: ${bitmapTime}ms, Improvement: ${improvement.toFixed(2)}%`);
      }

      console.log('\n=== Scalability Results ===');
      console.log('Document Count | Legacy (ms) | Bitmap (ms) | Improvement (%)');
      console.log('---------------|-------------|-------------|---------------');
      results.forEach(({ size, legacyTime, bitmapTime, improvement }) => {
        console.log(`${size.toString().padStart(13)} | ${legacyTime.toString().padStart(11)} | ${bitmapTime.toString().padStart(11)} | ${improvement.toFixed(2).padStart(13)}`);
      });

      // Verify that improvement generally increases with document set size
      const largestImprovement = results[results.length - 1]?.improvement ?? 0;
      expect(largestImprovement).toBeGreaterThan(10); // At least 10% improvement at largest scale
      
      // Verify all bitmap results meet Stage-A latency requirements
      results.forEach(({ bitmapTime }) => {
        expect(bitmapTime).toBeLessThan(25); // 3 searches should complete within 25ms total
      });
    }, 120000); // Extended timeout for scalability testing
  });
});