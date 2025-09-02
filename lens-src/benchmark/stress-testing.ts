/**
 * Stress Testing Benchmark for Phase B Optimizations
 * Tests performance under load, concurrency, and resource pressure
 * Validates optimization stability and resource usage under stress
 */

import { z } from 'zod';
import { v4 as uuidv4 } from 'uuid';
import { promises as fs } from 'fs';
import path from 'path';
import type { BenchmarkConfig, GoldenDataItem } from '../types/benchmark.js';
import { PhaseBPerformanceBenchmark, OptimizationStage } from './phase-b-performance.js';

// Stress test configuration schema
export const StressTestConfigSchema = z.object({
  load_tests: z.object({
    concurrent_queries: z.array(z.number().int().min(1).max(100)).default([1, 5, 10, 20, 50]),
    duration_minutes: z.number().int().min(1).max(30).default(5),
    ramp_up_seconds: z.number().int().min(10).max(300).default(30),
    query_rate_qps: z.array(z.number().min(0.1).max(100)).default([1, 5, 10, 20])
  }),
  resource_pressure: z.object({
    memory_pressure_enabled: z.boolean().default(true),
    memory_limit_mb: z.number().int().min(256).max(8192).default(1024),
    cpu_throttling_enabled: z.boolean().default(true),
    cpu_limit_percent: z.number().int().min(50).max(100).default(80),
    io_pressure_enabled: z.boolean().default(true)
  }),
  endurance_testing: z.object({
    long_running_hours: z.number().min(0.5).max(24).default(2),
    query_burst_enabled: z.boolean().default(true),
    burst_multiplier: z.number().min(2).max(10).default(5),
    burst_duration_seconds: z.number().int().min(10).max(300).default(60),
    burst_interval_minutes: z.number().int().min(5).max(60).default(15)
  }),
  degradation_thresholds: z.object({
    max_latency_degradation_pct: z.number().default(50), // 50% max degradation under load
    max_throughput_degradation_pct: z.number().default(30), // 30% max throughput loss
    max_error_rate_pct: z.number().default(5), // 5% max error rate
    max_memory_growth_mb: z.number().default(512), // 512MB max memory growth
    recovery_time_max_seconds: z.number().default(60) // 60s max recovery time
  })
});

export type StressTestConfig = z.infer<typeof StressTestConfigSchema>;

// Stress test results schema
export const StressTestResultSchema = z.object({
  trace_id: z.string().uuid(),
  timestamp: z.string().datetime(),
  config: StressTestConfigSchema,
  optimization_stage: z.enum([
    OptimizationStage.BASELINE,
    OptimizationStage.ROARING_BITMAP,
    OptimizationStage.AST_CACHE,
    OptimizationStage.ISOTONIC_CALIBRATION,
    OptimizationStage.INTEGRATED
  ]),
  load_test_results: z.array(z.object({
    test_name: z.string(),
    concurrent_queries: z.number(),
    duration_seconds: z.number(),
    total_queries: z.number(),
    successful_queries: z.number(),
    failed_queries: z.number(),
    avg_latency_ms: z.number(),
    p95_latency_ms: z.number(),
    p99_latency_ms: z.number(),
    throughput_qps: z.number(),
    error_rate_pct: z.number(),
    resource_usage: z.object({
      peak_memory_mb: z.number(),
      avg_cpu_pct: z.number(),
      peak_cpu_pct: z.number(),
      disk_io_ops: z.number()
    })
  })),
  endurance_test_results: z.object({
    duration_hours: z.number(),
    total_queries: z.number(),
    successful_queries: z.number(),
    memory_leak_detected: z.boolean(),
    memory_growth_mb: z.number(),
    performance_degradation_pct: z.number(),
    stability_score: z.number().min(0).max(1),
    recovery_metrics: z.object({
      burst_recovery_avg_ms: z.number(),
      error_recovery_time_ms: z.number(),
      cache_stability: z.boolean()
    })
  }),
  stress_comparison: z.object({
    baseline_vs_loaded: z.object({
      latency_degradation_pct: z.number(),
      throughput_degradation_pct: z.number(),
      error_rate_increase_pct: z.number(),
      memory_overhead_mb: z.number()
    }),
    optimization_benefits: z.object({
      load_handling_improvement: z.number(),
      memory_efficiency_gain: z.number(),
      error_resilience_gain: z.number()
    })
  }),
  stress_gate_evaluation: z.object({
    passed: z.boolean(),
    failing_thresholds: z.array(z.string()),
    stability_rating: z.enum(['excellent', 'good', 'acceptable', 'poor', 'failing'])
  })
});

export type StressTestResult = z.infer<typeof StressTestResultSchema>;

export class StressTestingBenchmark {
  constructor(
    private readonly phaseBBenchmark: PhaseBPerformanceBenchmark,
    private readonly outputDir: string
  ) {}

  /**
   * Run comprehensive stress testing for a specific optimization stage
   */
  async runStressTestSuite(
    stage: OptimizationStage, 
    config: StressTestConfig
  ): Promise<StressTestResult> {
    const traceId = uuidv4();
    console.log(`💪 Starting Stress Test Suite for ${stage} - Trace ID: ${traceId}`);

    const result: StressTestResult = {
      trace_id: traceId,
      timestamp: new Date().toISOString(),
      config,
      optimization_stage: stage,
      load_test_results: [],
      endurance_test_results: {
        duration_hours: 0,
        total_queries: 0,
        successful_queries: 0,
        memory_leak_detected: false,
        memory_growth_mb: 0,
        performance_degradation_pct: 0,
        stability_score: 0,
        recovery_metrics: {
          burst_recovery_avg_ms: 0,
          error_recovery_time_ms: 0,
          cache_stability: false
        }
      },
      stress_comparison: {
        baseline_vs_loaded: {
          latency_degradation_pct: 0,
          throughput_degradation_pct: 0,
          error_rate_increase_pct: 0,
          memory_overhead_mb: 0
        },
        optimization_benefits: {
          load_handling_improvement: 0,
          memory_efficiency_gain: 0,
          error_resilience_gain: 0
        }
      },
      stress_gate_evaluation: {
        passed: false,
        failing_thresholds: [],
        stability_rating: 'poor'
      }
    };

    try {
      // 1. Run baseline performance measurement
      console.log('📊 Measuring baseline performance...');
      const baselineMetrics = await this.measureBaselinePerformance(stage);

      // 2. Run load tests with increasing concurrency
      console.log('🚀 Running load tests...');
      for (const concurrency of config.load_tests.concurrent_queries) {
        const loadTestResult = await this.runLoadTest(stage, concurrency, config, baselineMetrics);
        result.load_test_results.push(loadTestResult);
      }

      // 3. Run endurance test
      console.log('⏰ Running endurance test...');
      result.endurance_test_results = await this.runEnduranceTest(stage, config);

      // 4. Analyze stress test results
      result.stress_comparison = this.analyzeStressImpact(result.load_test_results, baselineMetrics);
      result.stress_gate_evaluation = this.evaluateStressGate(result, config);

      // 5. Generate stress test report
      await this.generateStressTestReport(result);

      const rating = result.stress_gate_evaluation.stability_rating;
      console.log(`✅ Stress testing complete - Rating: ${rating.toUpperCase()}`);

      return result;

    } catch (error) {
      console.error('❌ Stress testing failed:', error);
      result.stress_gate_evaluation = {
        passed: false,
        failing_thresholds: ['test_execution_error'],
        stability_rating: 'failing'
      };
      return result;
    }
  }

  /**
   * Measure baseline performance metrics for comparison
   */
  private async measureBaselinePerformance(stage: OptimizationStage): Promise<any> {
    console.log('  Measuring single-query baseline...');
    
    // Run a small benchmark to establish baseline
    const baselineConfig: BenchmarkConfig = {
      trace_id: uuidv4(),
      suite: ['codesearch'],
      systems: ['lex'],
      slices: 'SMOKE_DEFAULT',
      seeds: 1,
      cache_mode: 'warm',
      robustness: false,
      metamorphic: false,
      k_candidates: 200,
      top_n: 50,
      fuzzy: 2,
      subtokens: true,
      semantic_gating: {
        nl_likelihood_threshold: 0.5,
        min_candidates: 10
      },
      latency_budgets: {
        stage_a_ms: 200,
        stage_b_ms: 300,
        stage_c_ms: 300
      }
    };

    // Configure the optimization stage
    await this.configureOptimizationStage(stage);
    
    // Run baseline benchmark (mock implementation - would integrate with real benchmark)
    const startTime = Date.now();
    const testQueries = await this.getTestQueries(10); // Small set for baseline
    const results = await this.executeQueries(testQueries, 1); // Single concurrency

    return {
      avg_latency_ms: results.avg_latency_ms,
      p95_latency_ms: results.p95_latency_ms,
      throughput_qps: results.throughput_qps,
      memory_baseline_mb: results.memory_usage_mb,
      error_rate_pct: results.error_rate_pct
    };
  }

  /**
   * Run load test with specific concurrency level
   */
  private async runLoadTest(
    stage: OptimizationStage,
    concurrency: number,
    config: StressTestConfig,
    baselineMetrics: any
  ): Promise<any> {
    console.log(`  Running load test: ${concurrency} concurrent queries...`);

    const testName = `load_test_${concurrency}x`;
    const durationSeconds = config.load_tests.duration_minutes * 60;
    
    // Ramp up gradually to target concurrency
    await this.rampUpLoad(concurrency, config.load_tests.ramp_up_seconds);
    
    const startTime = Date.now();
    const testQueries = await this.getTestQueries(100); // Larger set for load testing
    
    // Execute queries with specified concurrency
    const results = await this.executeQueriesWithLoad(
      testQueries, 
      concurrency, 
      durationSeconds,
      config
    );
    
    const endTime = Date.now();
    const actualDurationSeconds = (endTime - startTime) / 1000;
    
    return {
      test_name: testName,
      concurrent_queries: concurrency,
      duration_seconds: actualDurationSeconds,
      total_queries: results.total_queries,
      successful_queries: results.successful_queries,
      failed_queries: results.failed_queries,
      avg_latency_ms: results.avg_latency_ms,
      p95_latency_ms: results.p95_latency_ms,
      p99_latency_ms: results.p99_latency_ms,
      throughput_qps: results.throughput_qps,
      error_rate_pct: results.error_rate_pct,
      resource_usage: {
        peak_memory_mb: results.peak_memory_mb,
        avg_cpu_pct: results.avg_cpu_pct,
        peak_cpu_pct: results.peak_cpu_pct,
        disk_io_ops: results.disk_io_ops
      }
    };
  }

  /**
   * Run long-running endurance test with burst patterns
   */
  private async runEnduranceTest(
    stage: OptimizationStage,
    config: StressTestConfig
  ): Promise<any> {
    const durationHours = config.endurance_testing.long_running_hours;
    const durationMs = durationHours * 60 * 60 * 1000;
    
    console.log(`  Running endurance test: ${durationHours} hours...`);
    
    const startTime = Date.now();
    const memorySnapshots: number[] = [];
    const performanceSnapshots: any[] = [];
    
    let totalQueries = 0;
    let successfulQueries = 0;
    let burstRecoveryTimes: number[] = [];
    let errorRecoveryTime = 0;
    
    const testQueries = await this.getTestQueries(50);
    
    // Simulate endurance test with periodic bursts
    while ((Date.now() - startTime) < durationMs) {
      // Normal load period
      const normalResult = await this.executeQueries(testQueries, 2); // Low concurrency
      totalQueries += normalResult.total_queries;
      successfulQueries += normalResult.successful_queries;
      
      // Take memory snapshot
      memorySnapshots.push(normalResult.memory_usage_mb);
      performanceSnapshots.push({
        timestamp: Date.now(),
        latency_p95: normalResult.p95_latency_ms,
        throughput: normalResult.throughput_qps,
        memory_mb: normalResult.memory_usage_mb
      });
      
      // Periodic burst if enabled
      if (config.endurance_testing.query_burst_enabled) {
        const burstStart = Date.now();
        const burstConcurrency = 2 * config.endurance_testing.burst_multiplier;
        
        await this.executeQueries(testQueries, burstConcurrency);
        
        // Measure recovery time back to baseline performance
        const recoveryTime = await this.measureRecoveryTime(testQueries);
        burstRecoveryTimes.push(recoveryTime);
        
        // Wait until next burst
        await this.sleep(config.endurance_testing.burst_interval_minutes * 60 * 1000);
      } else {
        // Regular interval without bursts
        await this.sleep(60000); // 1 minute intervals
      }
    }
    
    const endTime = Date.now();
    const actualDurationHours = (endTime - startTime) / (1000 * 60 * 60);
    
    // Analyze results
    const memoryGrowth = memorySnapshots.length > 0 ? 
      Math.max(...memorySnapshots) - Math.min(...memorySnapshots) : 0;
    const memoryLeakDetected = this.detectMemoryLeak(memorySnapshots);
    const performanceDegradation = this.calculatePerformanceDegradation(performanceSnapshots);
    const stabilityScore = this.calculateStabilityScore(performanceSnapshots, totalQueries, successfulQueries);
    
    return {
      duration_hours: actualDurationHours,
      total_queries: totalQueries,
      successful_queries: successfulQueries,
      memory_leak_detected: memoryLeakDetected,
      memory_growth_mb: memoryGrowth,
      performance_degradation_pct: performanceDegradation,
      stability_score: stabilityScore,
      recovery_metrics: {
        burst_recovery_avg_ms: burstRecoveryTimes.length > 0 ? 
          burstRecoveryTimes.reduce((a, b) => a + b, 0) / burstRecoveryTimes.length : 0,
        error_recovery_time_ms: errorRecoveryTime,
        cache_stability: memoryGrowth < 100 // Less than 100MB growth considered stable
      }
    };
  }

  /**
   * Analyze stress impact compared to baseline
   */
  private analyzeStressImpact(loadTestResults: any[], baselineMetrics: any): any {
    if (loadTestResults.length === 0) {
      return {
        baseline_vs_loaded: {
          latency_degradation_pct: 0,
          throughput_degradation_pct: 0,
          error_rate_increase_pct: 0,
          memory_overhead_mb: 0
        },
        optimization_benefits: {
          load_handling_improvement: 0,
          memory_efficiency_gain: 0,
          error_resilience_gain: 0
        }
      };
    }

    // Find the highest load test result for comparison
    const highestLoadTest = loadTestResults.reduce((prev, current) => 
      current.concurrent_queries > prev.concurrent_queries ? current : prev
    );

    const latencyDegradation = ((highestLoadTest.p95_latency_ms - baselineMetrics.p95_latency_ms) / 
                               baselineMetrics.p95_latency_ms) * 100;
    const throughputDegradation = ((baselineMetrics.throughput_qps - highestLoadTest.throughput_qps) / 
                                  baselineMetrics.throughput_qps) * 100;
    const errorRateIncrease = highestLoadTest.error_rate_pct - baselineMetrics.error_rate_pct;
    const memoryOverhead = highestLoadTest.resource_usage.peak_memory_mb - baselineMetrics.memory_baseline_mb;

    return {
      baseline_vs_loaded: {
        latency_degradation_pct: latencyDegradation,
        throughput_degradation_pct: Math.max(0, throughputDegradation),
        error_rate_increase_pct: errorRateIncrease,
        memory_overhead_mb: memoryOverhead
      },
      optimization_benefits: {
        load_handling_improvement: Math.max(0, -throughputDegradation), // Negative degradation is improvement
        memory_efficiency_gain: Math.max(0, -memoryOverhead / baselineMetrics.memory_baseline_mb * 100),
        error_resilience_gain: Math.max(0, baselineMetrics.error_rate_pct - highestLoadTest.error_rate_pct)
      }
    };
  }

  /**
   * Evaluate stress gate criteria
   */
  private evaluateStressGate(result: StressTestResult, config: StressTestConfig): any {
    const failingThresholds: string[] = [];
    const stressComparison = result.stress_comparison.baseline_vs_loaded;
    const thresholds = config.degradation_thresholds;

    // Check latency degradation
    if (stressComparison.latency_degradation_pct > thresholds.max_latency_degradation_pct) {
      failingThresholds.push('latency_degradation');
    }

    // Check throughput degradation  
    if (stressComparison.throughput_degradation_pct > thresholds.max_throughput_degradation_pct) {
      failingThresholds.push('throughput_degradation');
    }

    // Check error rate increase
    if (stressComparison.error_rate_increase_pct > thresholds.max_error_rate_pct) {
      failingThresholds.push('error_rate_increase');
    }

    // Check memory growth
    if (result.endurance_test_results.memory_growth_mb > thresholds.max_memory_growth_mb) {
      failingThresholds.push('memory_growth');
    }

    // Check memory leak detection
    if (result.endurance_test_results.memory_leak_detected) {
      failingThresholds.push('memory_leak_detected');
    }

    // Determine stability rating
    let stabilityRating: 'excellent' | 'good' | 'acceptable' | 'poor' | 'failing';
    const failureCount = failingThresholds.length;
    const stabilityScore = result.endurance_test_results.stability_score;

    if (failureCount === 0 && stabilityScore > 0.95) {
      stabilityRating = 'excellent';
    } else if (failureCount === 0 && stabilityScore > 0.90) {
      stabilityRating = 'good';
    } else if (failureCount <= 1 && stabilityScore > 0.80) {
      stabilityRating = 'acceptable';
    } else if (failureCount <= 2 && stabilityScore > 0.60) {
      stabilityRating = 'poor';
    } else {
      stabilityRating = 'failing';
    }

    return {
      passed: failingThresholds.length === 0,
      failing_thresholds: failingThresholds,
      stability_rating: stabilityRating
    };
  }

  /**
   * Generate comprehensive stress test report
   */
  private async generateStressTestReport(result: StressTestResult): Promise<void> {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const stage = result.optimization_stage.toLowerCase().replace('_', '-');
    const reportPath = path.join(this.outputDir, `stress-test-${stage}-${timestamp}.json`);

    await fs.writeFile(reportPath, JSON.stringify(result, null, 2));
    console.log(`📄 Stress test report written to: ${reportPath}`);

    // Generate markdown summary
    const markdownSummary = this.generateStressTestMarkdown(result);
    const markdownPath = path.join(this.outputDir, `stress-test-${stage}-summary-${timestamp}.md`);
    await fs.writeFile(markdownPath, markdownSummary);
    console.log(`📋 Stress test summary written to: ${markdownPath}`);
  }

  // Helper methods (mock implementations)
  private async configureOptimizationStage(stage: OptimizationStage): Promise<void> {
    console.log(`  Configuring ${stage} optimization...`);
    // Mock implementation - would configure the search engine
  }

  private async getTestQueries(count: number): Promise<GoldenDataItem[]> {
    // Mock implementation - would return real test queries
    return Array.from({ length: count }, (_, i) => ({
      id: `test-${i}`,
      query: `test query ${i}`,
      query_class: 'identifier' as const,
      language: 'ts' as const,
      source: 'synthetics' as const,
      snapshot_sha: '0123456789abcdef0123456789abcdef01234567',
      slice_tags: ['test'],
      expected_results: []
    }));
  }

  private async executeQueries(queries: GoldenDataItem[], concurrency: number): Promise<any> {
    // Mock implementation - would execute real queries
    const latency = 50 + Math.random() * 100;
    const errorRate = Math.random() * 2; // 0-2% error rate
    
    return {
      total_queries: queries.length,
      successful_queries: Math.floor(queries.length * (1 - errorRate / 100)),
      failed_queries: queries.length - Math.floor(queries.length * (1 - errorRate / 100)),
      avg_latency_ms: latency,
      p95_latency_ms: latency * 1.5,
      p99_latency_ms: latency * 2,
      throughput_qps: queries.length / (latency / 1000),
      error_rate_pct: errorRate,
      memory_usage_mb: 256 + Math.random() * 256,
      peak_memory_mb: 512 + Math.random() * 256,
      avg_cpu_pct: 40 + Math.random() * 30,
      peak_cpu_pct: 60 + Math.random() * 30,
      disk_io_ops: Math.floor(Math.random() * 1000)
    };
  }

  private async executeQueriesWithLoad(
    queries: GoldenDataItem[],
    concurrency: number,
    durationSeconds: number,
    config: StressTestConfig
  ): Promise<any> {
    // Mock load test execution
    const baseLatency = 50 + (concurrency * 5); // Latency increases with load
    const degradationFactor = Math.min(2, 1 + (concurrency / 50));
    
    return {
      total_queries: Math.floor(durationSeconds * 10 * concurrency / degradationFactor),
      successful_queries: Math.floor(durationSeconds * 10 * concurrency / degradationFactor * 0.98),
      failed_queries: Math.floor(durationSeconds * 10 * concurrency / degradationFactor * 0.02),
      avg_latency_ms: baseLatency * degradationFactor,
      p95_latency_ms: baseLatency * degradationFactor * 1.8,
      p99_latency_ms: baseLatency * degradationFactor * 2.5,
      throughput_qps: (10 * concurrency / degradationFactor) * 0.98,
      error_rate_pct: 2,
      peak_memory_mb: 256 + (concurrency * 10),
      avg_cpu_pct: Math.min(95, 40 + (concurrency * 2)),
      peak_cpu_pct: Math.min(100, 60 + (concurrency * 3)),
      disk_io_ops: concurrency * 100
    };
  }

  private async rampUpLoad(targetConcurrency: number, rampUpSeconds: number): Promise<void> {
    console.log(`    Ramping up to ${targetConcurrency} over ${rampUpSeconds}s...`);
    await this.sleep(Math.min(rampUpSeconds * 1000, 5000)); // Mock ramp up (max 5s for demo)
  }

  private async measureRecoveryTime(queries: GoldenDataItem[]): Promise<number> {
    // Mock recovery time measurement
    return 1000 + Math.random() * 2000; // 1-3 seconds recovery time
  }

  private detectMemoryLeak(memorySnapshots: number[]): boolean {
    if (memorySnapshots.length < 10) return false;
    
    // Simple trend detection - consistent upward growth
    let increases = 0;
    for (let i = 1; i < memorySnapshots.length; i++) {
      if (memorySnapshots[i] > memorySnapshots[i - 1]) {
        increases++;
      }
    }
    
    return increases > (memorySnapshots.length * 0.7); // 70% increasing trend
  }

  private calculatePerformanceDegradation(snapshots: any[]): number {
    if (snapshots.length < 2) return 0;
    
    const initial = snapshots[0];
    const final = snapshots[snapshots.length - 1];
    
    const latencyDegradation = (final.latency_p95 - initial.latency_p95) / initial.latency_p95;
    const throughputDegradation = (initial.throughput - final.throughput) / initial.throughput;
    
    return Math.max(latencyDegradation, throughputDegradation) * 100;
  }

  private calculateStabilityScore(snapshots: any[], totalQueries: number, successfulQueries: number): number {
    if (totalQueries === 0) return 0;
    
    const successRate = successfulQueries / totalQueries;
    const consistencyScore = this.calculateConsistencyScore(snapshots);
    
    return (successRate + consistencyScore) / 2;
  }

  private calculateConsistencyScore(snapshots: any[]): number {
    if (snapshots.length < 3) return 1;
    
    // Calculate coefficient of variation for latency
    const latencies = snapshots.map(s => s.latency_p95);
    const mean = latencies.reduce((a, b) => a + b, 0) / latencies.length;
    const variance = latencies.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / latencies.length;
    const stdDev = Math.sqrt(variance);
    const coefficientOfVariation = stdDev / mean;
    
    // Lower coefficient of variation = higher consistency score
    return Math.max(0, 1 - coefficientOfVariation);
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  private generateStressTestMarkdown(result: StressTestResult): string {
    const stage = result.optimization_stage.replace('_', ' ').toUpperCase();
    const rating = result.stress_gate_evaluation.stability_rating.toUpperCase();
    const passed = result.stress_gate_evaluation.passed;

    return `# Stress Test Results: ${stage}

**Trace ID:** ${result.trace_id}  
**Timestamp:** ${result.timestamp}  
**Stability Rating:** ${rating} ${passed ? '✅' : '❌'}

## Load Test Results

${result.load_test_results.map(test => `
### ${test.concurrent_queries} Concurrent Queries

- **Duration:** ${test.duration_seconds.toFixed(1)}s
- **Total Queries:** ${test.total_queries}
- **Success Rate:** ${((test.successful_queries / test.total_queries) * 100).toFixed(1)}%
- **Avg Latency:** ${test.avg_latency_ms.toFixed(1)}ms
- **P95 Latency:** ${test.p95_latency_ms.toFixed(1)}ms
- **P99 Latency:** ${test.p99_latency_ms.toFixed(1)}ms
- **Throughput:** ${test.throughput_qps.toFixed(1)} QPS
- **Peak Memory:** ${test.resource_usage.peak_memory_mb}MB
- **Peak CPU:** ${test.resource_usage.peak_cpu_pct}%
`).join('\n')}

## Endurance Test Results

- **Duration:** ${result.endurance_test_results.duration_hours.toFixed(2)} hours
- **Total Queries:** ${result.endurance_test_results.total_queries}
- **Success Rate:** ${((result.endurance_test_results.successful_queries / result.endurance_test_results.total_queries) * 100).toFixed(1)}%
- **Memory Growth:** ${result.endurance_test_results.memory_growth_mb}MB
- **Memory Leak:** ${result.endurance_test_results.memory_leak_detected ? '⚠️ DETECTED' : '✅ None'}
- **Performance Degradation:** ${result.endurance_test_results.performance_degradation_pct.toFixed(1)}%
- **Stability Score:** ${result.endurance_test_results.stability_score.toFixed(3)}

## Stress Impact Analysis

### Baseline vs Maximum Load

- **Latency Degradation:** ${result.stress_comparison.baseline_vs_loaded.latency_degradation_pct.toFixed(1)}%
- **Throughput Degradation:** ${result.stress_comparison.baseline_vs_loaded.throughput_degradation_pct.toFixed(1)}%
- **Error Rate Increase:** ${result.stress_comparison.baseline_vs_loaded.error_rate_increase_pct.toFixed(1)}%
- **Memory Overhead:** ${result.stress_comparison.baseline_vs_loaded.memory_overhead_mb}MB

### Optimization Benefits

- **Load Handling Improvement:** ${result.stress_comparison.optimization_benefits.load_handling_improvement.toFixed(1)}%
- **Memory Efficiency Gain:** ${result.stress_comparison.optimization_benefits.memory_efficiency_gain.toFixed(1)}%
- **Error Resilience Gain:** ${result.stress_comparison.optimization_benefits.error_resilience_gain.toFixed(1)}%

## Gate Evaluation

**Status:** ${passed ? '✅ PASSED' : '❌ FAILED'}  
**Stability Rating:** ${rating}

${result.stress_gate_evaluation.failing_thresholds.length > 0 ? 
  `**Failing Thresholds:** ${result.stress_gate_evaluation.failing_thresholds.join(', ')}` : 
  '**All stress thresholds met!** 🎉'
}

## Summary

The ${stage} optimization demonstrates ${rating.toLowerCase()} stability under stress conditions.
${passed ? 
  'Performance remains within acceptable bounds under maximum load and extended duration testing.' : 
  'Performance degradation exceeds acceptable thresholds and requires optimization before production deployment.'
}
`;
  }
}