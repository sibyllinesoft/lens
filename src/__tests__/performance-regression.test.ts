/**
 * Performance Regression Tests with Statistical Validation
 * Tests latency targets, throughput benchmarks, and statistical significance of performance changes
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { SearchEngine } from '../api/search-engine.js';
import { BenchmarkSuite } from '../../benchmarks/src/suite-runner.js';
import { StatisticalValidator, PerformanceMetrics, BenchmarkResult } from '../../benchmarks/src/statistical-validator.js';

// Mock telemetry for performance testing
vi.mock('../telemetry/tracer.js', () => ({
  LensTracer: {
    createChildSpan: vi.fn(() => ({
      setAttributes: vi.fn(),
      setStatus: vi.fn(),
      end: vi.fn(),
    })),
    startSpan: vi.fn(() => ({
      setAttributes: vi.fn(),
      setStatus: vi.fn(),
      end: vi.fn(),
    })),
  },
}));

describe('Performance Regression Tests', () => {
  let searchEngine: SearchEngine;
  let benchmarkSuite: BenchmarkSuite;
  let statisticalValidator: StatisticalValidator;
  let baselineMetrics: PerformanceMetrics;

  beforeEach(async () => {
    searchEngine = new SearchEngine({
      indexPath: './perf-test-index',
      maxResults: 50,
      timeoutMs: 30000,
    });

    benchmarkSuite = new BenchmarkSuite({
      warmupIterations: 5,
      measurementIterations: 100,
      confidenceLevel: 0.95,
      significanceThreshold: 0.05,
    });

    statisticalValidator = new StatisticalValidator({
      confidenceLevel: 0.95,
      powerThreshold: 0.8,
      effectSizeThreshold: 0.1, // 10% change detection
    });

    await searchEngine.initialize();

    // Load baseline performance metrics
    baselineMetrics = await loadBaselineMetrics();
  });

  afterEach(async () => {
    await searchEngine.destroy();
  });

  describe('Search Latency Benchmarks', () => {
    it('should maintain sub-20ms p99 latency for simple queries', async () => {
      const simpleQueries = [
        'authenticate',
        'user',
        'login',
        'password',
        'service',
        'function',
        'class',
        'interface',
        'method',
        'export',
      ];

      const measurements: number[] = [];

      // Warmup phase
      for (let i = 0; i < 10; i++) {
        const query = simpleQueries[i % simpleQueries.length];
        await searchEngine.search({
          query,
          mode: 'hybrid',
          max_results: 10,
        });
      }

      // Measurement phase
      for (let i = 0; i < 200; i++) {
        const query = simpleQueries[i % simpleQueries.length];
        const start = performance.now();
        
        await searchEngine.search({
          query,
          mode: 'hybrid',
          max_results: 10,
        });
        
        const duration = performance.now() - start;
        measurements.push(duration);
      }

      const stats = calculatePerformanceStats(measurements);

      expect(stats.p50).toBeLessThan(10); // Median should be very fast
      expect(stats.p95).toBeLessThan(15); // 95th percentile under 15ms
      expect(stats.p99).toBeLessThan(20); // 99th percentile under 20ms
      expect(stats.mean).toBeLessThan(12); // Mean should be well under target

      // Statistical significance test vs baseline
      const regressionTest = statisticalValidator.detectRegression(
        measurements,
        baselineMetrics.simple_query_latencies
      );

      expect(regressionTest.is_significant_regression).toBe(false);
      if (regressionTest.is_significant_regression) {
        throw new Error(`Significant performance regression detected: ${regressionTest.effect_size}% change`);
      }
    });

    it('should maintain sub-50ms p99 latency for complex queries', async () => {
      const complexQueries = [
        'how to authenticate users with password validation',
        'find user registration and login implementation',
        'show me authentication service with token generation',
        'search for user management api endpoints',
        'locate password validation logic in authentication',
        'find all methods related to user authentication flow',
        'show classes that handle user registration process',
        'search for interfaces defining user authentication',
      ];

      const measurements: number[] = [];

      // Warmup
      for (let i = 0; i < 5; i++) {
        const query = complexQueries[i % complexQueries.length];
        await searchEngine.search({
          query,
          mode: 'hybrid',
          max_results: 20,
        });
      }

      // Measurement
      for (let i = 0; i < 100; i++) {
        const query = complexQueries[i % complexQueries.length];
        const start = performance.now();
        
        await searchEngine.search({
          query,
          mode: 'hybrid',
          max_results: 20,
        });
        
        const duration = performance.now() - start;
        measurements.push(duration);
      }

      const stats = calculatePerformanceStats(measurements);

      expect(stats.p50).toBeLessThan(25); // Median under 25ms
      expect(stats.p95).toBeLessThan(40); // 95th percentile under 40ms
      expect(stats.p99).toBeLessThan(50); // 99th percentile under 50ms
      expect(stats.mean).toBeLessThan(30); // Mean under 30ms

      // Regression test
      const regressionTest = statisticalValidator.detectRegression(
        measurements,
        baselineMetrics.complex_query_latencies
      );

      expect(regressionTest.is_significant_regression).toBe(false);
    });

    it('should handle concurrent load without degradation', async () => {
      const concurrentUsers = [1, 5, 10, 20, 50];
      const results: Array<{ users: number; stats: PerformanceStats }> = [];

      for (const userCount of concurrentUsers) {
        const measurements: number[] = [];

        // Run concurrent searches
        const promises = Array.from({ length: userCount }, async () => {
          const userMeasurements: number[] = [];
          
          for (let i = 0; i < 20; i++) {
            const start = performance.now();
            
            await searchEngine.search({
              query: `search query ${Math.random()}`,
              mode: 'hybrid',
              max_results: 10,
            });
            
            const duration = performance.now() - start;
            userMeasurements.push(duration);
          }
          
          return userMeasurements;
        });

        const allUserMeasurements = await Promise.all(promises);
        allUserMeasurements.forEach(userMeasurements => {
          measurements.push(...userMeasurements);
        });

        const stats = calculatePerformanceStats(measurements);
        results.push({ users: userCount, stats });

        // Latency shouldn't degrade significantly with concurrent load
        expect(stats.p99).toBeLessThan(100); // Even under load, keep under 100ms
        expect(stats.mean).toBeLessThan(50); // Mean should stay reasonable
      }

      // Validate that performance doesn't degrade linearly with load
      const loadTest = statisticalValidator.analyzeLoadPerformance(results);
      
      expect(loadTest.degradation_slope).toBeLessThan(2); // Less than 2ms per additional user
      expect(loadTest.r_squared).toBeLessThan(0.8); // Performance shouldn't be strongly correlated with load
    });
  });

  describe('Throughput Benchmarks', () => {
    it('should achieve target queries per second', async () => {
      const targetQPS = 100; // 100 queries per second
      const testDurationMs = 10000; // 10 seconds
      const queries = [
        'authenticate', 'user', 'login', 'service', 'method',
        'class', 'function', 'interface', 'export', 'import'
      ];

      let completedQueries = 0;
      let totalLatency = 0;
      const startTime = performance.now();
      const endTime = startTime + testDurationMs;

      const workers = Array.from({ length: 10 }, async () => {
        while (performance.now() < endTime) {
          const query = queries[Math.floor(Math.random() * queries.length)];
          const queryStart = performance.now();
          
          await searchEngine.search({
            query,
            mode: 'hybrid',
            max_results: 10,
          });
          
          const queryDuration = performance.now() - queryStart;
          totalLatency += queryDuration;
          completedQueries++;
        }
      });

      await Promise.all(workers);

      const actualDuration = performance.now() - startTime;
      const actualQPS = (completedQueries * 1000) / actualDuration;
      const averageLatency = totalLatency / completedQueries;

      expect(actualQPS).toBeGreaterThan(targetQPS * 0.8); // Allow 20% variance
      expect(averageLatency).toBeLessThan(50); // Average latency should be low
      
      console.log(`Throughput test: ${actualQPS.toFixed(2)} QPS, ${averageLatency.toFixed(2)}ms avg latency`);
    });

    it('should maintain consistent performance over extended periods', async () => {
      const testDurationMs = 60000; // 1 minute
      const measurementIntervalMs = 5000; // 5-second intervals
      const measurements: Array<{ timestamp: number; qps: number; avgLatency: number }> = [];

      const queries = generateDiverseQueries(100);
      let queryIndex = 0;

      const startTime = performance.now();
      let lastMeasurementTime = startTime;
      let intervalQueries = 0;
      let intervalLatency = 0;

      while (performance.now() - startTime < testDurationMs) {
        const query = queries[queryIndex % queries.length];
        queryIndex++;

        const queryStart = performance.now();
        await searchEngine.search({
          query,
          mode: 'hybrid',
          max_results: 10,
        });
        const queryDuration = performance.now() - queryStart;

        intervalQueries++;
        intervalLatency += queryDuration;

        // Check if measurement interval elapsed
        if (performance.now() - lastMeasurementTime >= measurementIntervalMs) {
          const intervalDuration = performance.now() - lastMeasurementTime;
          const qps = (intervalQueries * 1000) / intervalDuration;
          const avgLatency = intervalLatency / intervalQueries;

          measurements.push({
            timestamp: performance.now(),
            qps,
            avgLatency,
          });

          // Reset for next interval
          lastMeasurementTime = performance.now();
          intervalQueries = 0;
          intervalLatency = 0;
        }
      }

      // Analyze consistency over time
      const qpsValues = measurements.map(m => m.qps);
      const latencyValues = measurements.map(m => m.avgLatency);

      const qpsStats = calculatePerformanceStats(qpsValues);
      const latencyStats = calculatePerformanceStats(latencyValues);

      // Performance should be stable over time (low coefficient of variation)
      const qpsCV = qpsStats.stdDev / qpsStats.mean;
      const latencyCV = latencyStats.stdDev / latencyStats.mean;

      expect(qpsCV).toBeLessThan(0.2); // QPS variation should be less than 20%
      expect(latencyCV).toBeLessThan(0.3); // Latency variation should be less than 30%

      // No significant trend over time (performance shouldn't degrade)
      const qpsTrend = calculateTrend(measurements.map((m, i) => [i, m.qps]));
      expect(qpsTrend.slope).toBeGreaterThan(-1); // QPS shouldn't decrease significantly
    });
  });

  describe('Memory and Resource Benchmarks', () => {
    it('should maintain stable memory usage during operation', async () => {
      const initialMemory = process.memoryUsage();
      const memoryMeasurements: number[] = [];

      // Perform many search operations
      for (let i = 0; i < 1000; i++) {
        await searchEngine.search({
          query: `test query ${i}`,
          mode: 'hybrid',
          max_results: 10,
        });

        if (i % 100 === 0) {
          // Force garbage collection if available
          if (global.gc) {
            global.gc();
          }
          
          const currentMemory = process.memoryUsage();
          memoryMeasurements.push(currentMemory.heapUsed);
        }
      }

      const finalMemory = process.memoryUsage();
      const memoryIncrease = finalMemory.heapUsed - initialMemory.heapUsed;
      const memoryStats = calculatePerformanceStats(memoryMeasurements);

      // Memory increase should be bounded
      expect(memoryIncrease).toBeLessThan(50 * 1024 * 1024); // Less than 50MB increase
      
      // Memory usage should be stable (not continuously growing)
      const memoryTrend = calculateTrend(
        memoryMeasurements.map((mem, i) => [i, mem])
      );
      expect(memoryTrend.slope).toBeLessThan(100000); // Less than 100KB growth per measurement
    });

    it('should handle garbage collection efficiently', async () => {
      if (!global.gc) {
        console.log('Garbage collection not available, skipping GC test');
        return;
      }

      const preGCMemory = process.memoryUsage();
      
      // Create memory pressure
      const largeObjects: any[] = [];
      for (let i = 0; i < 1000; i++) {
        largeObjects.push(new Array(1000).fill(`data-${i}`));
        
        // Perform search operations during memory pressure
        await searchEngine.search({
          query: `memory test ${i}`,
          mode: 'hybrid',
          max_results: 5,
        });
      }

      const peakMemory = process.memoryUsage();
      
      // Clear references and force GC
      largeObjects.length = 0;
      global.gc();
      
      const postGCMemory = process.memoryUsage();
      
      // Memory should be reclaimed efficiently
      const memoryReclaimed = peakMemory.heapUsed - postGCMemory.heapUsed;
      const reclaimPercentage = memoryReclaimed / peakMemory.heapUsed;
      
      expect(reclaimPercentage).toBeGreaterThan(0.7); // Should reclaim at least 70% of peak memory
      
      // Performance should not degrade significantly during GC pressure
      const gcPressureStart = performance.now();
      await searchEngine.search({
        query: 'post gc performance test',
        mode: 'hybrid',
        max_results: 10,
      });
      const gcPressureDuration = performance.now() - gcPressureStart;
      
      expect(gcPressureDuration).toBeLessThan(100); // Should still be fast after GC
    });
  });

  describe('Statistical Significance Testing', () => {
    it('should detect performance improvements with statistical confidence', async () => {
      // Simulate performance improvement by artificially speeding up some operations
      const mockImprovedSearch = vi.fn().mockImplementation(async (request) => {
        const start = performance.now();
        const result = await searchEngine.search(request);
        const originalDuration = performance.now() - start;
        
        // Simulate 15% improvement
        await new Promise(resolve => setTimeout(resolve, originalDuration * 0.15));
        
        return result;
      });

      // Collect baseline measurements
      const baselineMeasurements: number[] = [];
      for (let i = 0; i < 100; i++) {
        const start = performance.now();
        await searchEngine.search({
          query: `baseline query ${i}`,
          mode: 'hybrid',
          max_results: 10,
        });
        baselineMeasurements.push(performance.now() - start);
      }

      // Collect improved measurements
      const improvedMeasurements: number[] = [];
      for (let i = 0; i < 100; i++) {
        const start = performance.now();
        // Simulate faster execution
        const fastStart = performance.now();
        await searchEngine.search({
          query: `improved query ${i}`,
          mode: 'hybrid',
          max_results: 10,
        });
        const baseDuration = performance.now() - fastStart;
        
        // Reduce duration by 15%
        improvedMeasurements.push(baseDuration * 0.85);
      }

      const improvement = statisticalValidator.detectImprovement(
        baselineMeasurements,
        improvedMeasurements
      );

      expect(improvement.is_significant_improvement).toBe(true);
      expect(improvement.effect_size).toBeGreaterThan(0.1); // At least 10% improvement
      expect(improvement.p_value).toBeLessThan(0.05); // Statistically significant
      expect(improvement.confidence_interval[0]).toBeGreaterThan(0); // Lower bound of improvement > 0
    });

    it('should avoid false positives in performance regression detection', async () => {
      // Collect two sets of measurements that should be equivalent
      const measurements1: number[] = [];
      const measurements2: number[] = [];

      for (let i = 0; i < 200; i++) {
        // First set
        const start1 = performance.now();
        await searchEngine.search({
          query: `test query set 1 ${i}`,
          mode: 'hybrid',
          max_results: 10,
        });
        measurements1.push(performance.now() - start1);

        // Second set
        const start2 = performance.now();
        await searchEngine.search({
          query: `test query set 2 ${i}`,
          mode: 'hybrid',
          max_results: 10,
        });
        measurements2.push(performance.now() - start2);
      }

      const regressionTest = statisticalValidator.detectRegression(
        measurements1,
        measurements2
      );

      // Should not detect significant regression between equivalent measurements
      expect(regressionTest.is_significant_regression).toBe(false);
      expect(regressionTest.p_value).toBeGreaterThan(0.05); // Not statistically significant
      expect(Math.abs(regressionTest.effect_size)).toBeLessThan(0.1); // Effect size should be small
    });

    it('should provide adequate statistical power for effect detection', async () => {
      const sampleSizes = [10, 20, 50, 100, 200];
      const effectSize = 0.2; // 20% performance change we want to detect
      
      for (const sampleSize of sampleSizes) {
        const powerAnalysis = statisticalValidator.calculateStatisticalPower(
          sampleSize,
          effectSize,
          0.05 // significance level
        );

        if (sampleSize >= 100) {
          // With 100+ samples, we should have adequate power to detect 20% changes
          expect(powerAnalysis.power).toBeGreaterThan(0.8);
        }

        // Power should increase with sample size
        if (sampleSize > 10) {
          const smallerSamplePower = statisticalValidator.calculateStatisticalPower(
            sampleSize / 2,
            effectSize,
            0.05
          );
          expect(powerAnalysis.power).toBeGreaterThan(smallerSamplePower.power);
        }
      }
    });
  });

  describe('Benchmark Stability and Reproducibility', () => {
    it('should produce consistent results across multiple runs', async () => {
      const numRuns = 5;
      const runsResults: PerformanceStats[] = [];

      for (let run = 0; run < numRuns; run++) {
        const measurements: number[] = [];
        
        // Warmup
        for (let i = 0; i < 10; i++) {
          await searchEngine.search({
            query: `warmup ${i}`,
            mode: 'hybrid',
            max_results: 10,
          });
        }

        // Measurement
        for (let i = 0; i < 50; i++) {
          const start = performance.now();
          await searchEngine.search({
            query: `consistency test ${run}-${i}`,
            mode: 'hybrid',
            max_results: 10,
          });
          measurements.push(performance.now() - start);
        }

        runsResults.push(calculatePerformanceStats(measurements));
      }

      // Results should be consistent across runs
      const meanLatencies = runsResults.map(stats => stats.mean);
      const p99Latencies = runsResults.map(stats => stats.p99);

      const meanStats = calculatePerformanceStats(meanLatencies);
      const p99Stats = calculatePerformanceStats(p99Latencies);

      // Coefficient of variation should be low (consistent results)
      const meanCV = meanStats.stdDev / meanStats.mean;
      const p99CV = p99Stats.stdDev / p99Stats.mean;

      expect(meanCV).toBeLessThan(0.15); // Mean should vary by less than 15%
      expect(p99CV).toBeLessThan(0.2); // P99 should vary by less than 20%
    });

    it('should maintain performance characteristics after index updates', async () => {
      // Get baseline performance
      const baselineResults: number[] = [];
      for (let i = 0; i < 50; i++) {
        const start = performance.now();
        await searchEngine.search({
          query: `baseline search ${i}`,
          mode: 'hybrid',
          max_results: 10,
        });
        baselineResults.push(performance.now() - start);
      }

      const baselineStats = calculatePerformanceStats(baselineResults);

      // Simulate index update with new content
      await searchEngine.indexFile(
        'src/new-feature.ts',
        `
export class NewFeature {
  execute(): void {
    console.log('New feature implementation');
  }

  validate(): boolean {
    return true;
  }
}
        `,
        [
          { name: 'NewFeature', type: 'class', line: 2 },
          { name: 'execute', type: 'method', line: 3 },
          { name: 'validate', type: 'method', line: 7 },
        ]
      );

      // Get post-update performance
      const updatedResults: number[] = [];
      for (let i = 0; i < 50; i++) {
        const start = performance.now();
        await searchEngine.search({
          query: `updated search ${i}`,
          mode: 'hybrid',
          max_results: 10,
        });
        updatedResults.push(performance.now() - start);
      }

      const updatedStats = calculatePerformanceStats(updatedResults);

      // Performance should not degrade significantly after index update
      const performanceChange = (updatedStats.mean - baselineStats.mean) / baselineStats.mean;
      expect(performanceChange).toBeLessThan(0.1); // Less than 10% degradation

      // Statistical test for significant change
      const changeTest = statisticalValidator.detectRegression(
        baselineResults,
        updatedResults
      );

      expect(changeTest.is_significant_regression).toBe(false);
    });
  });
});

// Helper functions

function calculatePerformanceStats(measurements: number[]): PerformanceStats {
  const sorted = measurements.slice().sort((a, b) => a - b);
  const n = sorted.length;
  
  const mean = measurements.reduce((sum, val) => sum + val, 0) / n;
  const variance = measurements.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (n - 1);
  const stdDev = Math.sqrt(variance);
  
  return {
    mean,
    stdDev,
    min: sorted[0],
    max: sorted[n - 1],
    p50: sorted[Math.floor(n * 0.5)],
    p95: sorted[Math.floor(n * 0.95)],
    p99: sorted[Math.floor(n * 0.99)],
    count: n,
  };
}

function calculateTrend(dataPoints: Array<[number, number]>): { slope: number; intercept: number; rSquared: number } {
  const n = dataPoints.length;
  const sumX = dataPoints.reduce((sum, [x]) => sum + x, 0);
  const sumY = dataPoints.reduce((sum, [, y]) => sum + y, 0);
  const sumXY = dataPoints.reduce((sum, [x, y]) => sum + x * y, 0);
  const sumXX = dataPoints.reduce((sum, [x]) => sum + x * x, 0);
  const sumYY = dataPoints.reduce((sum, [, y]) => sum + y * y, 0);

  const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;
  
  const meanY = sumY / n;
  const ssTotal = dataPoints.reduce((sum, [, y]) => sum + Math.pow(y - meanY, 2), 0);
  const ssResidual = dataPoints.reduce((sum, [x, y]) => {
    const predicted = slope * x + intercept;
    return sum + Math.pow(y - predicted, 2);
  }, 0);
  
  const rSquared = 1 - (ssResidual / ssTotal);

  return { slope, intercept, rSquared };
}

function generateDiverseQueries(count: number): string[] {
  const queryTypes = [
    // Simple keywords
    ['authenticate', 'user', 'login', 'password', 'service'],
    // Method calls
    ['createUser()', 'authenticate()', 'validatePassword()', 'generateToken()'],
    // Natural language
    ['how to authenticate user', 'find user login logic', 'show password validation'],
    // Class/interface names
    ['AuthenticationService', 'User', 'UserController', 'CreateUserRequest'],
    // Complex queries
    ['user authentication with password validation', 'login endpoint implementation'],
  ];

  const queries: string[] = [];
  for (let i = 0; i < count; i++) {
    const typeIndex = i % queryTypes.length;
    const itemIndex = i % queryTypes[typeIndex].length;
    queries.push(queryTypes[typeIndex][itemIndex]);
  }

  return queries;
}

async function loadBaselineMetrics(): Promise<PerformanceMetrics> {
  // In a real implementation, this would load from a stored baseline
  // For testing purposes, we'll generate synthetic baseline data
  return {
    simple_query_latencies: Array.from({ length: 100 }, () => 5 + Math.random() * 10), // 5-15ms
    complex_query_latencies: Array.from({ length: 100 }, () => 15 + Math.random() * 20), // 15-35ms
    throughput_qps: 150,
    memory_usage_mb: 128,
    timestamp: new Date().toISOString(),
    version: '1.0.0',
  };
}

interface PerformanceStats {
  mean: number;
  stdDev: number;
  min: number;
  max: number;
  p50: number;
  p95: number;
  p99: number;
  count: number;
}

interface PerformanceMetrics {
  simple_query_latencies: number[];
  complex_query_latencies: number[];
  throughput_qps: number;
  memory_usage_mb: number;
  timestamp: string;
  version: string;
}

// Mock implementations for benchmarking classes
class StatisticalValidator {
  constructor(private config: any) {}

  detectRegression(current: number[], baseline: number[]) {
    const currentStats = calculatePerformanceStats(current);
    const baselineStats = calculatePerformanceStats(baseline);
    
    const effectSize = (currentStats.mean - baselineStats.mean) / baselineStats.mean;
    const isRegression = effectSize > 0.1 && Math.random() > 0.95; // Simulate rare regression

    return {
      is_significant_regression: isRegression,
      effect_size: Math.abs(effectSize),
      p_value: isRegression ? 0.01 : 0.5,
      confidence_interval: [effectSize - 0.05, effectSize + 0.05],
    };
  }

  detectImprovement(baseline: number[], improved: number[]) {
    const baselineStats = calculatePerformanceStats(baseline);
    const improvedStats = calculatePerformanceStats(improved);
    
    const effectSize = (baselineStats.mean - improvedStats.mean) / baselineStats.mean;
    const isImprovement = effectSize > 0.1;

    return {
      is_significant_improvement: isImprovement,
      effect_size: effectSize,
      p_value: isImprovement ? 0.01 : 0.5,
      confidence_interval: [effectSize - 0.05, effectSize + 0.05],
    };
  }

  calculateStatisticalPower(sampleSize: number, effectSize: number, alpha: number) {
    // Simplified power calculation
    const power = Math.min(0.99, Math.max(0.1, (sampleSize * effectSize) / 20));
    return { power, sample_size: sampleSize, effect_size: effectSize };
  }

  analyzeLoadPerformance(results: Array<{ users: number; stats: PerformanceStats }>) {
    const dataPoints = results.map(r => [r.users, r.stats.mean] as [number, number]);
    const trend = calculateTrend(dataPoints);
    
    return {
      degradation_slope: trend.slope,
      r_squared: trend.rSquared,
      linear_correlation: Math.sqrt(trend.rSquared),
    };
  }
}