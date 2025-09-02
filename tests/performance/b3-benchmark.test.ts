/**
 * Performance Benchmark Suite for Phase B3 Optimizations
 * Validates the target 12ms → 6-8ms (~40% improvement) for Stage-C
 * Tests isotonic calibration, confidence gating, and optimized HNSW performance
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { IsotonicCalibratedReranker } from '../../src/core/isotonic-reranker.js';
import { OptimizedHNSWIndex } from '../../src/core/optimized-hnsw.js';
import { EnhancedSemanticRerankEngine } from '../../src/indexer/enhanced-semantic.js';
import { SemanticRerankEngine } from '../../src/indexer/semantic.js'; // Baseline
import { SegmentStorage } from '../../src/storage/segments.js';
import type { SearchContext, Candidate } from '../../src/types/core.js';

interface BenchmarkResult {
  name: string;
  latencyMs: number;
  throughputQps: number;
  qualityScore: number;
  memoryMb: number;
  iterations: number;
}

interface PerformanceMetrics {
  avg: number;
  p50: number;
  p95: number;
  p99: number;
  min: number;
  max: number;
}

function calculatePercentiles(values: number[]): PerformanceMetrics {
  const sorted = values.sort((a, b) => a - b);
  const len = sorted.length;
  
  return {
    avg: values.reduce((sum, v) => sum + v, 0) / len,
    p50: sorted[Math.floor(len * 0.5)]!,
    p95: sorted[Math.floor(len * 0.95)]!,
    p99: sorted[Math.floor(len * 0.99)]!,
    min: sorted[0]!,
    max: sorted[len - 1]!
  };
}

function generateTestVectors(count: number, dimension: number): Map<string, Float32Array> {
  const vectors = new Map<string, Float32Array>();
  
  for (let i = 0; i < count; i++) {
    const vector = new Float32Array(dimension);
    for (let j = 0; j < dimension; j++) {
      vector[j] = Math.random() * 2 - 1; // -1 to 1
    }
    
    // Normalize vector
    const norm = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    if (norm > 0) {
      for (let j = 0; j < dimension; j++) {
        vector[j] /= norm;
      }
    }
    
    vectors.set(`doc_${i}`, vector);
  }
  
  return vectors;
}

function generateTestCandidates(count: number, baseScore: number = 0.5): Candidate[] {
  return Array.from({ length: count }, (_, i) => ({
    doc_id: `candidate_${i}`,
    file_path: `/test/file_${i}.js`,
    line: Math.floor(Math.random() * 100) + 1,
    col: Math.floor(Math.random() * 80) + 1,
    score: Math.max(0.1, Math.min(1.0, baseScore + (Math.random() - 0.5) * 0.4)),
    match_reasons: ['symbol', 'fuzzy'][Math.floor(Math.random() * 2)] as any,
    context: `function test_${i}() { return "test_${i}"; }`,
    symbol_kind: ['function', 'class', 'variable'][Math.floor(Math.random() * 3)] as any
  }));
}

describe('B3 Performance Benchmarks - Stage-C Optimization', () => {
  const DIMENSION = 128;
  const VECTOR_COUNT = 500;
  const BENCHMARK_ITERATIONS = 100;
  const TARGET_LATENCY_MS = 8; // B3 target: 6-8ms
  const BASELINE_LATENCY_MS = 12; // Current baseline
  const IMPROVEMENT_TARGET = 0.4; // 40% improvement

  let testVectors: Map<string, Float32Array>;
  let testQueries: Float32Array[];
  let baselineEngine: SemanticRerankEngine;
  let enhancedEngine: EnhancedSemanticRerankEngine;
  let segmentStorage: SegmentStorage;

  beforeAll(async () => {
    console.log('🚀 Setting up B3 performance benchmark suite...');
    
    // Generate test data
    testVectors = generateTestVectors(VECTOR_COUNT, DIMENSION);
    
    testQueries = Array.from({ length: 20 }, (_, i) => {
      const query = new Float32Array(DIMENSION);
      for (let j = 0; j < DIMENSION; j++) {
        query[j] = Math.sin(i * 0.1 + j * 0.05);
      }
      return query;
    });

    // Initialize engines
    segmentStorage = new SegmentStorage('./benchmark-segments');
    
    baselineEngine = new SemanticRerankEngine(segmentStorage);
    await baselineEngine.initialize();

    enhancedEngine = new EnhancedSemanticRerankEngine(segmentStorage, {
      enableIsotonicCalibration: true,
      enableConfidenceGating: true,
      enableOptimizedHNSW: true,
      maxLatencyMs: TARGET_LATENCY_MS,
      featureFlags: {
        stageCOptimizations: true,
        advancedCalibration: true,
        experimentalHNSW: false
      }
    });
    await enhancedEngine.initialize();

    // Index test documents for both engines
    let docIndex = 0;
    for (const [docId, vector] of testVectors) {
      const content = `function test_${docIndex}() { return "content_${docIndex}"; }`;
      const filePath = `/test/doc_${docIndex}.js`;
      
      await baselineEngine.indexDocument(docId, content, filePath);
      await enhancedEngine.indexDocument(docId, content, filePath);
      docIndex++;
    }

    console.log(`📊 Indexed ${VECTOR_COUNT} documents, ready for benchmarking`);
  });

  afterAll(async () => {
    await baselineEngine?.shutdown();
    await enhancedEngine?.shutdown();
    await segmentStorage?.shutdown();
  });

  describe('Isotonic Calibration Performance', () => {
    let reranker: IsotonicCalibratedReranker;

    beforeAll(() => {
      reranker = new IsotonicCalibratedReranker({
        enabled: true,
        minCalibrationData: 50,
        confidenceCutoff: 0.12,
        maxLatencyMs: TARGET_LATENCY_MS,
        calibrationUpdateFreq: 100
      });

      // Pre-train with calibration data
      for (let i = 0; i < 100; i++) {
        const predictedScore = Math.random();
        const actualRelevance = Math.min(1, predictedScore + (Math.random() - 0.5) * 0.2);
        reranker.recordCalibrationExample({} as any, predictedScore, actualRelevance);
      }
    });

    it('should rerank within latency budget', async () => {
      const candidatesVariations = [10, 25, 50, 100];
      const results: BenchmarkResult[] = [];

      for (const candidateCount of candidatesVariations) {
        const latencies: number[] = [];
        const testCandidates = generateTestCandidates(candidateCount);
        
        const testContext: SearchContext = {
          trace_id: `isotonic-bench-${candidateCount}`,
          query: 'test function implementation',
          mode: 'hybrid',
          k: Math.min(10, candidateCount),
          fuzzy_distance: 0,
          started_at: new Date(),
          stages: []
        } as SearchContext;

        // Warm up
        for (let i = 0; i < 5; i++) {
          await reranker.rerank(testCandidates.map(c => ({ ...c, doc_id: c.doc_id, file: c.file_path, line: c.line, col: c.col, score: c.score, snippet: c.context, why: c.match_reasons.join(',') })), testContext);
        }

        // Benchmark
        for (let i = 0; i < BENCHMARK_ITERATIONS; i++) {
          const startTime = performance.now();
          
          const searchHits = testCandidates.map(c => ({
            doc_id: c.doc_id,
            file: c.file_path,
            line: c.line,
            col: c.col,
            score: c.score,
            snippet: c.context,
            why: c.match_reasons.join(','),
            symbol_kind: c.symbol_kind
          }));

          await reranker.rerank(searchHits, testContext);
          
          const latency = performance.now() - startTime;
          latencies.push(latency);
        }

        const metrics = calculatePercentiles(latencies);
        const throughput = 1000 / metrics.avg;

        results.push({
          name: `Isotonic-${candidateCount}`,
          latencyMs: metrics.avg,
          throughputQps: throughput,
          qualityScore: 0.95, // Estimated
          memoryMb: process.memoryUsage().heapUsed / 1024 / 1024,
          iterations: BENCHMARK_ITERATIONS
        });

        // Validate performance targets
        expect(metrics.p95).toBeLessThan(TARGET_LATENCY_MS);
        expect(metrics.avg).toBeLessThan(TARGET_LATENCY_MS * 0.8); // 80% of budget on average

        console.log(`📈 Isotonic reranker (${candidateCount} candidates): avg=${metrics.avg.toFixed(1)}ms, p95=${metrics.p95.toFixed(1)}ms, throughput=${throughput.toFixed(0)} QPS`);
      }

      // Overall performance should be significantly better than baseline
      const avgLatency = results.reduce((sum, r) => sum + r.latencyMs, 0) / results.length;
      expect(avgLatency).toBeLessThan(BASELINE_LATENCY_MS * (1 - IMPROVEMENT_TARGET)); // 40% improvement
    });
  });

  describe('Optimized HNSW Performance', () => {
    let hnswIndex: OptimizedHNSWIndex;

    beforeAll(async () => {
      hnswIndex = new OptimizedHNSWIndex({
        K: 150, // Fixed per B3 requirements
        efSearch: 64,
        efConstruction: 200,
        qualityThreshold: 0.005,
        performanceTarget: IMPROVEMENT_TARGET
      });

      await hnswIndex.buildIndex(testVectors);
    });

    it('should search within performance targets', async () => {
      const kValues = [10, 20, 50, 100];
      const results: BenchmarkResult[] = [];

      for (const k of kValues) {
        const latencies: number[] = [];

        // Warm up
        for (let i = 0; i < 5; i++) {
          await hnswIndex.search(testQueries[0]!, k);
        }

        // Benchmark across different queries
        for (const query of testQueries) {
          const startTime = performance.now();
          const searchResults = await hnswIndex.search(query, k);
          const latency = performance.now() - startTime;
          
          latencies.push(latency);
          
          // Validate search quality
          expect(searchResults.length).toBeLessThanOrEqual(k);
          expect(searchResults.every(r => r.score >= 0 && r.score <= 1)).toBe(true);
        }

        const metrics = calculatePercentiles(latencies);
        const throughput = 1000 / metrics.avg;

        results.push({
          name: `HNSW-k${k}`,
          latencyMs: metrics.avg,
          throughputQps: throughput,
          qualityScore: 0.97, // Estimated based on HNSW quality
          memoryMb: process.memoryUsage().heapUsed / 1024 / 1024,
          iterations: testQueries.length
        });

        // Performance validation
        expect(metrics.p95).toBeLessThan(5.0); // HNSW should be very fast
        expect(metrics.avg).toBeLessThan(3.0); // Average should be even better

        console.log(`🔍 HNSW search (k=${k}): avg=${metrics.avg.toFixed(2)}ms, p95=${metrics.p95.toFixed(2)}ms, throughput=${throughput.toFixed(0)} QPS`);
      }

      // Overall HNSW performance should be excellent
      const avgLatency = results.reduce((sum, r) => sum + r.latencyMs, 0) / results.length;
      expect(avgLatency).toBeLessThan(2.5); // HNSW component should be <2.5ms
    });

    it('should tune efSearch for optimal performance', async () => {
      const tuningStart = performance.now();
      
      const optimalEfSearch = await hnswIndex.tuneEfSearch(testQueries.slice(0, 10), []);
      
      const tuningLatency = performance.now() - tuningStart;

      expect(optimalEfSearch).toBeGreaterThan(0);
      expect(optimalEfSearch).toBeLessThanOrEqual(256);
      expect(tuningLatency).toBeLessThan(5000); // Tuning should complete quickly

      console.log(`🔧 HNSW tuning completed in ${tuningLatency.toFixed(0)}ms, optimal efSearch: ${optimalEfSearch}`);

      // Test performance with tuned parameters
      const latencies: number[] = [];
      for (const query of testQueries.slice(0, 5)) {
        const startTime = performance.now();
        await hnswIndex.search(query, 20, optimalEfSearch);
        const latency = performance.now() - startTime;
        latencies.push(latency);
      }

      const tunedMetrics = calculatePercentiles(latencies);
      expect(tunedMetrics.avg).toBeLessThan(3.0);
      
      console.log(`⚡ Tuned HNSW performance: avg=${tunedMetrics.avg.toFixed(2)}ms, p95=${tunedMetrics.p95.toFixed(2)}ms`);
    });
  });

  describe('End-to-End B3 vs Baseline Comparison', () => {
    const testScenarios = [
      { name: 'Small', candidates: 25, query: 'test function' },
      { name: 'Medium', candidates: 50, query: 'calculate sum addition' },
      { name: 'Large', candidates: 100, query: 'how to implement sorting algorithm efficiently' },
      { name: 'XLarge', candidates: 200, query: 'find all methods in class that handle user authentication and authorization' }
    ];

    it('should demonstrate 40% performance improvement over baseline', async () => {
      const baselineResults: BenchmarkResult[] = [];
      const enhancedResults: BenchmarkResult[] = [];

      for (const scenario of testScenarios) {
        const testCandidates = generateTestCandidates(scenario.candidates, 0.6);
        
        const testContext: SearchContext = {
          trace_id: `e2e-bench-${scenario.name}`,
          query: scenario.query,
          mode: 'hybrid',
          k: Math.min(20, scenario.candidates),
          fuzzy_distance: 0,
          started_at: new Date(),
          stages: []
        } as SearchContext;

        // Baseline performance test
        const baselineLatencies: number[] = [];
        for (let i = 0; i < 50; i++) { // Fewer iterations for e2e tests
          const startTime = performance.now();
          await baselineEngine.rerankCandidates(testCandidates, testContext, 20);
          const latency = performance.now() - startTime;
          baselineLatencies.push(latency);
        }

        // Enhanced B3 performance test
        const enhancedLatencies: number[] = [];
        for (let i = 0; i < 50; i++) {
          const startTime = performance.now();
          await enhancedEngine.rerankCandidates(testCandidates, testContext, 20);
          const latency = performance.now() - startTime;
          enhancedLatencies.push(latency);
        }

        const baselineMetrics = calculatePercentiles(baselineLatencies);
        const enhancedMetrics = calculatePercentiles(enhancedLatencies);

        baselineResults.push({
          name: `Baseline-${scenario.name}`,
          latencyMs: baselineMetrics.avg,
          throughputQps: 1000 / baselineMetrics.avg,
          qualityScore: 0.90, // Estimated baseline quality
          memoryMb: process.memoryUsage().heapUsed / 1024 / 1024,
          iterations: 50
        });

        enhancedResults.push({
          name: `Enhanced-${scenario.name}`,
          latencyMs: enhancedMetrics.avg,
          throughputQps: 1000 / enhancedMetrics.avg,
          qualityScore: 0.92, // Slightly better due to isotonic calibration
          memoryMb: process.memoryUsage().heapUsed / 1024 / 1024,
          iterations: 50
        });

        // Individual scenario validation
        const improvement = (baselineMetrics.avg - enhancedMetrics.avg) / baselineMetrics.avg;
        
        console.log(`📊 ${scenario.name} (${scenario.candidates} candidates):`);
        console.log(`   Baseline: avg=${baselineMetrics.avg.toFixed(1)}ms, p95=${baselineMetrics.p95.toFixed(1)}ms`);
        console.log(`   Enhanced: avg=${enhancedMetrics.avg.toFixed(1)}ms, p95=${enhancedMetrics.p95.toFixed(1)}ms`);
        console.log(`   Improvement: ${(improvement * 100).toFixed(1)}%`);

        // B3 optimizations should show improvement
        expect(enhancedMetrics.avg).toBeLessThan(baselineMetrics.avg);
        expect(enhancedMetrics.p95).toBeLessThan(TARGET_LATENCY_MS);
      }

      // Overall improvement validation
      const avgBaselineLatency = baselineResults.reduce((sum, r) => sum + r.latencyMs, 0) / baselineResults.length;
      const avgEnhancedLatency = enhancedResults.reduce((sum, r) => sum + r.latencyMs, 0) / enhancedResults.length;
      const overallImprovement = (avgBaselineLatency - avgEnhancedLatency) / avgBaselineLatency;

      console.log(`\n🎯 OVERALL B3 PERFORMANCE RESULTS:`);
      console.log(`   Average Baseline Latency: ${avgBaselineLatency.toFixed(1)}ms`);
      console.log(`   Average Enhanced Latency: ${avgEnhancedLatency.toFixed(1)}ms`);
      console.log(`   Overall Improvement: ${(overallImprovement * 100).toFixed(1)}%`);
      console.log(`   Target Achievement: ${avgEnhancedLatency <= TARGET_LATENCY_MS ? '✅' : '❌'}`);

      // Main B3 validation criteria
      expect(overallImprovement).toBeGreaterThanOrEqual(IMPROVEMENT_TARGET); // 40% improvement target
      expect(avgEnhancedLatency).toBeLessThanOrEqual(TARGET_LATENCY_MS); // 6-8ms target
    });

    it('should maintain quality while improving performance', async () => {
      const testCandidates = generateTestCandidates(75, 0.5);
      
      // Create a context with a query that should favor certain candidates
      const testContext: SearchContext = {
        trace_id: 'quality-test',
        query: 'calculate sum addition math',
        mode: 'hybrid',
        k: 15,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: []
      } as SearchContext;

      // Boost some candidates to test quality preservation
      testCandidates[5].context = 'function calculateSum(a, b) { return a + b; }'; // Should rank high
      testCandidates[5].score = 0.75;
      testCandidates[15].context = 'function addNumbers(x, y) { return x + y; }'; // Should rank high
      testCandidates[15].score = 0.7;

      const baselineResults = await baselineEngine.rerankCandidates(testCandidates, testContext, 15);
      const enhancedResults = await enhancedEngine.rerankCandidates(testCandidates, testContext, 15);

      // Both should identify the relevant candidates highly
      const baselineRelevantRank = baselineResults.findIndex(c => c.context?.includes('calculateSum'));
      const enhancedRelevantRank = enhancedResults.findIndex(c => c.context?.includes('calculateSum'));

      expect(baselineRelevantRank).toBeLessThan(10); // Should be in top 10
      expect(enhancedRelevantRank).toBeLessThan(10); // Should be in top 10

      // Enhanced should maintain or improve quality
      expect(enhancedRelevantRank).toBeLessThanOrEqual(baselineRelevantRank + 2); // Allow small quality trade-off

      console.log(`📈 Quality preservation: baseline rank ${baselineRelevantRank}, enhanced rank ${enhancedRelevantRank}`);
    });
  });

  describe('Memory and Resource Efficiency', () => {
    it('should not significantly increase memory usage', async () => {
      const initialMemory = process.memoryUsage();
      
      // Perform intensive operations
      const heavyTestCandidates = generateTestCandidates(300, 0.5);
      const testContext: SearchContext = {
        trace_id: 'memory-test',
        query: 'intensive memory test query with multiple terms',
        mode: 'hybrid',
        k: 50,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: []
      } as SearchContext;

      // Run multiple iterations
      for (let i = 0; i < 20; i++) {
        await enhancedEngine.rerankCandidates(heavyTestCandidates, testContext, 50);
      }

      const finalMemory = process.memoryUsage();
      const memoryIncrease = (finalMemory.heapUsed - initialMemory.heapUsed) / 1024 / 1024;

      console.log(`💾 Memory usage: initial=${(initialMemory.heapUsed/1024/1024).toFixed(1)}MB, final=${(finalMemory.heapUsed/1024/1024).toFixed(1)}MB, increase=${memoryIncrease.toFixed(1)}MB`);

      // Memory increase should be reasonable
      expect(memoryIncrease).toBeLessThan(50); // Less than 50MB increase
    });

    it('should handle concurrent requests efficiently', async () => {
      const concurrentQueries = 10;
      const candidatesPerQuery = 50;
      
      const promises = Array.from({ length: concurrentQueries }, async (_, i) => {
        const candidates = generateTestCandidates(candidatesPerQuery, 0.6);
        const context: SearchContext = {
          trace_id: `concurrent-${i}`,
          query: `concurrent test query ${i}`,
          mode: 'hybrid',
          k: 20,
          fuzzy_distance: 0,
          started_at: new Date(),
          stages: []
        } as SearchContext;

        const startTime = performance.now();
        const results = await enhancedEngine.rerankCandidates(candidates, context, 20);
        const latency = performance.now() - startTime;

        return { results, latency, queryId: i };
      });

      const startTime = performance.now();
      const concurrentResults = await Promise.all(promises);
      const totalTime = performance.now() - startTime;

      const avgLatency = concurrentResults.reduce((sum, r) => sum + r.latency, 0) / concurrentQueries;
      const maxLatency = Math.max(...concurrentResults.map(r => r.latency));

      console.log(`🔄 Concurrent performance: ${concurrentQueries} queries, total=${totalTime.toFixed(0)}ms, avg=${avgLatency.toFixed(1)}ms, max=${maxLatency.toFixed(1)}ms`);

      // Concurrent performance should remain good
      expect(avgLatency).toBeLessThan(TARGET_LATENCY_MS * 1.5); // Allow 50% overhead for concurrency
      expect(maxLatency).toBeLessThan(TARGET_LATENCY_MS * 2); // Max should be reasonable

      // All queries should return valid results
      concurrentResults.forEach(result => {
        expect(result.results.length).toBeGreaterThan(0);
        expect(result.results.every(c => c.score > 0)).toBe(true);
      });
    });
  });

  describe('Feature Flag Performance Impact', () => {
    it('should measure performance impact of individual optimizations', async () => {
      const testCandidates = generateTestCandidates(60, 0.5);
      const testContext: SearchContext = {
        trace_id: 'feature-flag-test',
        query: 'feature flag performance test',
        mode: 'hybrid',
        k: 15,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: []
      } as SearchContext;

      const configurations = [
        { name: 'All Disabled', stageCOptimizations: false, isotonicCalibration: false, optimizedHNSW: false },
        { name: 'Isotonic Only', stageCOptimizations: false, isotonicCalibration: true, optimizedHNSW: false },
        { name: 'HNSW Only', stageCOptimizations: false, isotonicCalibration: false, optimizedHNSW: true },
        { name: 'All Enabled', stageCOptimizations: true, isotonicCalibration: true, optimizedHNSW: true }
      ];

      const results: Record<string, number> = {};

      for (const config of configurations) {
        await enhancedEngine.updateConfig({
          featureFlags: {
            stageCOptimizations: config.stageCOptimizations,
            advancedCalibration: config.isotonicCalibration,
            experimentalHNSW: config.optimizedHNSW
          }
        });

        // Warm up
        for (let i = 0; i < 3; i++) {
          await enhancedEngine.rerankCandidates(testCandidates, testContext, 15);
        }

        // Benchmark
        const latencies: number[] = [];
        for (let i = 0; i < 25; i++) {
          const startTime = performance.now();
          await enhancedEngine.rerankCandidates(testCandidates, testContext, 15);
          const latency = performance.now() - startTime;
          latencies.push(latency);
        }

        const avgLatency = latencies.reduce((sum, l) => sum + l, 0) / latencies.length;
        results[config.name] = avgLatency;

        console.log(`🚩 ${config.name}: ${avgLatency.toFixed(1)}ms`);
      }

      // All enabled should be fastest
      expect(results['All Enabled']).toBeLessThan(results['All Disabled']);
      expect(results['All Enabled']).toBeLessThanOrEqual(TARGET_LATENCY_MS);

      // Individual optimizations should show measurable impact
      expect(results['Isotonic Only']).toBeLessThan(results['All Disabled']);
    });
  });
});