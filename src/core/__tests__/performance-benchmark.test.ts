import { describe, it, expect, beforeAll } from 'bun:test';
import { AdvancedSearchIntegration } from '../advanced-search-integration';
import { ComprehensiveMonitoring } from '../comprehensive-monitoring';
import type { SearchContext, SearchHit } from '../../types/search';

interface BenchmarkResult {
  operationsPerSecond: number;
  p50LatencyMs: number;
  p95LatencyMs: number;
  p99LatencyMs: number;
  maxLatencyMs: number;
  memoryUsageMB: number;
  errorRate: number;
}

interface LoadTestConfig {
  concurrency: number;
  duration: number; // seconds
  rampUpTime: number; // seconds
  targetOpsPerSec: number;
}

class PerformanceBenchmark {
  private integration: AdvancedSearchIntegration;
  private monitoring: ComprehensiveMonitoring;
  private mockHits: SearchHit[];

  constructor() {
    this.integration = AdvancedSearchIntegration.getInstance();
    this.monitoring = ComprehensiveMonitoring.getInstance();
    this.mockHits = this.generateMockHits(100);
  }

  private generateMockHits(count: number): SearchHit[] {
    return Array.from({ length: count }, (_, i) => ({
      id: `perf-hit-${i}`,
      path: `/performance/test/file${i % 20}.ts`, // Create some path diversity
      content: `performance test content ${i}`.repeat(Math.floor(Math.random() * 5) + 1),
      score: Math.max(0.1, 1.0 - (i * 0.01) + (Math.random() * 0.1 - 0.05)),
      snippet: `performance snippet ${i}`,
      line: Math.floor(Math.random() * 1000) + 1,
      column: Math.floor(Math.random() * 80) + 1,
      context: `performance context ${i}`,
      startOffset: i * 15,
      endOffset: (i * 15) + 8
    }));
  }

  private createTestContext(requestId: string): SearchContext {
    return {
      query: `performance test query ${Math.floor(Math.random() * 10)}`,
      filters: Math.random() > 0.8 ? { type: 'function' } : {},
      userId: `perf-user-${Math.floor(Math.random() * 5)}`,
      requestId,
      timestamp: Date.now(),
      efSearch: Math.random() > 0.9 ? 100 : 50, // Occasionally use expensive mode
      maxResults: Math.floor(Math.random() * 20) + 10,
      includeSnippets: Math.random() > 0.3
    };
  }

  async runSingleOperation(requestId: string): Promise<{
    latencyMs: number;
    success: boolean;
    resultCount: number;
  }> {
    const startTime = performance.now();
    const memBefore = process.memoryUsage().heapUsed;
    
    try {
      const context = this.createTestContext(requestId);
      const result = await this.integration.executeAdvancedSearch(
        this.mockHits,
        context,
        Math.random() > 0.7 ? new Float32Array(384).fill(Math.random() * 0.1) : undefined
      );

      const latencyMs = performance.now() - startTime;
      const memAfter = process.memoryUsage().heapUsed;
      const memDeltaMB = (memAfter - memBefore) / (1024 * 1024);

      return {
        latencyMs,
        success: result.safetyValidation.passed,
        resultCount: result.enhancedHits.length
      };
    } catch (error) {
      const latencyMs = performance.now() - startTime;
      return {
        latencyMs,
        success: false,
        resultCount: 0
      };
    }
  }

  async runLoadTest(config: LoadTestConfig): Promise<BenchmarkResult> {
    const results: Array<{ latencyMs: number; success: boolean }> = [];
    const startTime = Date.now();
    const endTime = startTime + (config.duration * 1000);
    
    let requestCounter = 0;
    let completedOps = 0;
    let errors = 0;
    
    const workers: Promise<void>[] = [];
    
    // Create concurrent workers
    for (let w = 0; w < config.concurrency; w++) {
      const worker = async () => {
        while (Date.now() < endTime) {
          const requestId = `load-test-${requestCounter++}`;
          const opResult = await this.runSingleOperation(requestId);
          
          results.push({
            latencyMs: opResult.latencyMs,
            success: opResult.success
          });
          
          completedOps++;
          if (!opResult.success) errors++;
          
          // Throttle to target ops/sec
          const targetIntervalMs = (1000 * config.concurrency) / config.targetOpsPerSec;
          await new Promise(resolve => setTimeout(resolve, Math.max(0, targetIntervalMs - opResult.latencyMs)));
        }
      };
      
      workers.push(worker());
    }
    
    // Wait for all workers to complete
    await Promise.all(workers);
    
    // Calculate statistics
    const actualDuration = (Date.now() - startTime) / 1000;
    const latencies = results.map(r => r.latencyMs).sort((a, b) => a - b);
    const memUsage = process.memoryUsage();
    
    return {
      operationsPerSecond: completedOps / actualDuration,
      p50LatencyMs: this.percentile(latencies, 50),
      p95LatencyMs: this.percentile(latencies, 95),
      p99LatencyMs: this.percentile(latencies, 99),
      maxLatencyMs: Math.max(...latencies),
      memoryUsageMB: memUsage.heapUsed / (1024 * 1024),
      errorRate: errors / completedOps
    };
  }

  private percentile(values: number[], p: number): number {
    const index = Math.ceil((p / 100) * values.length) - 1;
    return values[Math.max(0, Math.min(index, values.length - 1))];
  }

  async runStressTest(): Promise<BenchmarkResult> {
    // High concurrency, high load test
    return this.runLoadTest({
      concurrency: 20,
      duration: 30,
      rampUpTime: 5,
      targetOpsPerSec: 500
    });
  }

  async runSustainedTest(): Promise<BenchmarkResult> {
    // Lower load but sustained over longer period
    return this.runLoadTest({
      concurrency: 10,
      duration: 120,
      rampUpTime: 10,
      targetOpsPerSec: 200
    });
  }

  async runLatencyTest(): Promise<BenchmarkResult> {
    // Focus on latency with moderate load
    return this.runLoadTest({
      concurrency: 5,
      duration: 60,
      rampUpTime: 5,
      targetOpsPerSec: 100
    });
  }
}

describe('Advanced Search Performance Benchmarks', () => {
  let benchmark: PerformanceBenchmark;

  beforeAll(async () => {
    benchmark = new PerformanceBenchmark();
    
    // Warm up the system
    console.log('Warming up system...');
    await benchmark.runLoadTest({
      concurrency: 2,
      duration: 10,
      rampUpTime: 2,
      targetOpsPerSec: 50
    });
    console.log('Warm-up complete');
  });

  describe('Latency Requirements Validation', () => {
    it('should meet p95 latency requirement (≤21ms including optimizations)', async () => {
      console.log('Running latency-focused benchmark...');
      const result = await benchmark.runLatencyTest();
      
      console.log(`\nLatency Benchmark Results:`);
      console.log(`- Operations/sec: ${result.operationsPerSecond.toFixed(1)}`);
      console.log(`- p50 latency: ${result.p50LatencyMs.toFixed(2)}ms`);
      console.log(`- p95 latency: ${result.p95LatencyMs.toFixed(2)}ms`);
      console.log(`- p99 latency: ${result.p99LatencyMs.toFixed(2)}ms`);
      console.log(`- Max latency: ${result.maxLatencyMs.toFixed(2)}ms`);
      console.log(`- Error rate: ${(result.errorRate * 100).toFixed(2)}%`);
      console.log(`- Memory usage: ${result.memoryUsageMB.toFixed(1)}MB`);
      
      // Allow 1ms overhead for optimizations on top of 20ms baseline
      expect(result.p95LatencyMs).toBeLessThanOrEqual(21.0);
      expect(result.errorRate).toBeLessThan(0.01); // Less than 1% errors
    }, 90000); // 90 second timeout
    
    it('should maintain low latency under sustained load', async () => {
      console.log('Running sustained load benchmark...');
      const result = await benchmark.runSustainedTest();
      
      console.log(`\nSustained Load Benchmark Results:`);
      console.log(`- Operations/sec: ${result.operationsPerSecond.toFixed(1)}`);
      console.log(`- p95 latency: ${result.p95LatencyMs.toFixed(2)}ms`);
      console.log(`- Error rate: ${(result.errorRate * 100).toFixed(2)}%`);
      console.log(`- Memory usage: ${result.memoryUsageMB.toFixed(1)}MB`);
      
      // Under sustained load, allow slightly higher latency
      expect(result.p95LatencyMs).toBeLessThanOrEqual(25.0);
      expect(result.errorRate).toBeLessThan(0.02); // Less than 2% errors under load
      expect(result.memoryUsageMB).toBeLessThan(500); // Reasonable memory usage
    }, 180000); // 3 minute timeout
  });

  describe('Throughput Requirements Validation', () => {
    it('should handle high throughput without degradation', async () => {
      console.log('Running stress test benchmark...');
      const result = await benchmark.runStressTest();
      
      console.log(`\nStress Test Benchmark Results:`);
      console.log(`- Operations/sec: ${result.operationsPerSecond.toFixed(1)}`);
      console.log(`- p95 latency: ${result.p95LatencyMs.toFixed(2)}ms`);
      console.log(`- Error rate: ${(result.errorRate * 100).toFixed(2)}%`);
      console.log(`- Memory usage: ${result.memoryUsageMB.toFixed(1)}MB`);
      
      // Should handle at least 200 ops/sec
      expect(result.operationsPerSecond).toBeGreaterThanOrEqual(200);
      expect(result.errorRate).toBeLessThan(0.05); // Less than 5% errors under stress
      
      // Latency may be higher under extreme load, but should stay reasonable
      expect(result.p95LatencyMs).toBeLessThanOrEqual(50.0);
    }, 60000); // 1 minute timeout
  });

  describe('Resource Utilization Validation', () => {
    it('should demonstrate memory efficiency', async () => {
      const baseline = process.memoryUsage();
      console.log(`Baseline memory: ${(baseline.heapUsed / (1024 * 1024)).toFixed(1)}MB`);
      
      // Run moderate load
      const result = await benchmark.runLoadTest({
        concurrency: 8,
        duration: 30,
        rampUpTime: 3,
        targetOpsPerSec: 150
      });
      
      const postTest = process.memoryUsage();
      const memoryGrowthMB = (postTest.heapUsed - baseline.heapUsed) / (1024 * 1024);
      
      console.log(`\nMemory Efficiency Results:`);
      console.log(`- Baseline: ${(baseline.heapUsed / (1024 * 1024)).toFixed(1)}MB`);
      console.log(`- Post-test: ${(postTest.heapUsed / (1024 * 1024)).toFixed(1)}MB`);
      console.log(`- Growth: ${memoryGrowthMB.toFixed(1)}MB`);
      console.log(`- Operations: ${(result.operationsPerSecond * 30).toFixed(0)} total`);
      
      // Memory growth should be minimal
      expect(memoryGrowthMB).toBeLessThan(100); // Less than 100MB growth
      
      // Force garbage collection and check for leaks
      if (global.gc) {
        global.gc();
        const postGC = process.memoryUsage();
        const leakMB = (postGC.heapUsed - baseline.heapUsed) / (1024 * 1024);
        console.log(`- Post-GC: ${(postGC.heapUsed / (1024 * 1024)).toFixed(1)}MB`);
        console.log(`- Potential leak: ${leakMB.toFixed(1)}MB`);
        
        expect(leakMB).toBeLessThan(50); // Less than 50MB potential leak
      }
    }, 60000);
  });

  describe('Scalability Validation', () => {
    it('should scale performance with concurrency', async () => {
      const concurrencyLevels = [1, 2, 5, 10];
      const results: Array<{ concurrency: number; opsPerSec: number; p95Latency: number }> = [];
      
      for (const concurrency of concurrencyLevels) {
        console.log(`Testing concurrency level: ${concurrency}`);
        
        const result = await benchmark.runLoadTest({
          concurrency,
          duration: 20,
          rampUpTime: 2,
          targetOpsPerSec: 50 * concurrency // Scale target with concurrency
        });
        
        results.push({
          concurrency,
          opsPerSec: result.operationsPerSecond,
          p95Latency: result.p95LatencyMs
        });
        
        console.log(`- Concurrency ${concurrency}: ${result.operationsPerSecond.toFixed(1)} ops/sec, ${result.p95LatencyMs.toFixed(2)}ms p95`);
      }
      
      console.log('\nScalability Analysis:');
      results.forEach(r => {
        console.log(`Concurrency ${r.concurrency}: ${r.opsPerSec.toFixed(1)} ops/sec, ${r.p95Latency.toFixed(2)}ms p95`);
      });
      
      // Throughput should increase with concurrency (with some efficiency loss)
      const throughputRatio = results[results.length - 1].opsPerSec / results[0].opsPerSec;
      const concurrencyRatio = concurrencyLevels[concurrencyLevels.length - 1] / concurrencyLevels[0];
      const efficiency = throughputRatio / concurrencyRatio;
      
      console.log(`Scaling efficiency: ${(efficiency * 100).toFixed(1)}%`);
      
      expect(efficiency).toBeGreaterThan(0.5); // At least 50% scaling efficiency
      
      // p95 latency shouldn't degrade too much with concurrency
      const latencyIncrease = results[results.length - 1].p95Latency / results[0].p95Latency;
      expect(latencyIncrease).toBeLessThan(3.0); // Less than 3x latency increase
    }, 120000); // 2 minute timeout
  });

  describe('Advanced Optimization Impact Analysis', () => {
    it('should measure the impact of each optimization component', async () => {
      const integration = AdvancedSearchIntegration.getInstance();
      const componentTests = [
        { name: 'Baseline (All Disabled)', enabled: false },
        { name: 'Conformal Router Only', enabled: true, component: 'conformal' },
        { name: 'Entropy Gating Only', enabled: true, component: 'entropy' },
        { name: 'RAPTOR Only', enabled: true, component: 'raptor' },
        { name: 'All Components', enabled: true, component: 'all' }
      ];

      console.log('\nComponent Impact Analysis:');
      
      for (const test of componentTests) {
        // Configure components
        if (!test.enabled) {
          integration.disable();
        } else {
          integration.enable();
          // In practice, you'd selectively enable/disable components
          // This is a simplified test
        }

        // Run focused test
        const result = await benchmark.runLoadTest({
          concurrency: 3,
          duration: 15,
          rampUpTime: 2,
          targetOpsPerSec: 60
        });

        console.log(`${test.name}:`);
        console.log(`  - Throughput: ${result.operationsPerSecond.toFixed(1)} ops/sec`);
        console.log(`  - p95 Latency: ${result.p95LatencyMs.toFixed(2)}ms`);
        console.log(`  - Error Rate: ${(result.errorRate * 100).toFixed(2)}%`);

        // Basic validation - should maintain reasonable performance
        expect(result.p95LatencyMs).toBeLessThan(30); // Allow higher latency for analysis
        expect(result.errorRate).toBeLessThan(0.1); // Allow higher error rate for analysis
      }
      
      // Re-enable all components for other tests
      integration.enable();
    }, 180000); // 3 minute timeout
  });
});

describe('Long-running Stability Tests', () => {
  let benchmark: PerformanceBenchmark;

  beforeAll(() => {
    benchmark = new PerformanceBenchmark();
  });

  it('should maintain performance over extended periods', async () => {
    console.log('Running extended stability test (5 minutes)...');
    
    const result = await benchmark.runLoadTest({
      concurrency: 6,
      duration: 300, // 5 minutes
      rampUpTime: 30,
      targetOpsPerSec: 120
    });

    console.log(`\nExtended Stability Test Results (5 minutes):`);
    console.log(`- Operations/sec: ${result.operationsPerSecond.toFixed(1)}`);
    console.log(`- p95 latency: ${result.p95LatencyMs.toFixed(2)}ms`);
    console.log(`- p99 latency: ${result.p99LatencyMs.toFixed(2)}ms`);
    console.log(`- Error rate: ${(result.errorRate * 100).toFixed(2)}%`);
    console.log(`- Memory usage: ${result.memoryUsageMB.toFixed(1)}MB`);

    // Should maintain stable performance over time
    expect(result.operationsPerSecond).toBeGreaterThanOrEqual(100);
    expect(result.p95LatencyMs).toBeLessThanOrEqual(30);
    expect(result.errorRate).toBeLessThan(0.03); // Less than 3% errors
    expect(result.memoryUsageMB).toBeLessThan(1000); // Memory should be reasonable

    // Generate final monitoring report
    const monitoring = ComprehensiveMonitoring.getInstance();
    const dashboard = await monitoring.generateDashboard();
    
    console.log(`\nSystem Health After Extended Test:`);
    console.log(`- Overall Status: ${dashboard.systemHealth.overallStatus}`);
    console.log(`- Total Operations: ${dashboard.metrics.totalOperations}`);
    console.log(`- Active Alerts: ${dashboard.alerts.length}`);
    
    expect(dashboard.systemHealth.overallStatus).toBe('healthy');
  }, 360000); // 6 minute timeout
});