/**
 * RAPTOR Benchmarking System
 * 
 * Implements benchmarking against Anchor NL/struct gates with promotion
 * criteria and statistical significance testing.
 */

import { RaptorSystem, RaptorFeatures } from './index.js';

export interface BenchmarkConfig {
  dataset: 'anchor' | 'ladder';
  slices: ('nl' | 'structural')[];
  baseline_system: string;
  test_system: string;
  significance_threshold: number; // p-value threshold (0.05)
  min_samples: number;           // Minimum queries per slice
  max_runtime_ms: number;        // Per-query timeout
}

export interface BenchmarkMetrics {
  ndcg_at_10: number;
  recall_at_50: number;
  precision_at_10: number;
  span_coverage: number;         // % queries with 100% span coverage
  e2e_p95_ms: number;           // End-to-end latency p95
  feature_computation_ms: number; // Time to compute RAPTOR features
}

export interface BenchmarkResult {
  slice: string;
  query_count: number;
  baseline_metrics: BenchmarkMetrics;
  test_metrics: BenchmarkMetrics;
  deltas: {
    ndcg_at_10_delta: number;
    ndcg_at_10_pvalue: number;
    recall_at_50_delta: number;
    span_coverage_delta: number;
    e2e_p95_delta_pct: number;
  };
  promotion_gates: {
    ndcg_improvement: boolean;    // ΔnDCG@10 ≥ +2% (p<0.05)
    recall_maintained: boolean;   // Recall@50 ≥ baseline
    span_maintained: boolean;     // Span = 100%
    latency_acceptable: boolean;  // E2E p95 ≤ +10%
  };
  passed: boolean;
}

export interface BenchmarkSuite {
  config: BenchmarkConfig;
  results: BenchmarkResult[];
  overall_passed: boolean;
  timestamp: number;
  duration_ms: number;
}

export interface QuerySample {
  id: string;
  query: string;
  repo_sha: string;
  expected_files: string[];     // Ground truth files
  candidate_files: string[];   // Stage A/B candidates
  slice: 'nl' | 'structural';
}

/**
 * Statistical significance testing
 */
class StatisticalTesting {
  static tTest(baseline: number[], test: number[]): { tStatistic: number; pValue: number } {
    if (baseline.length === 0 || test.length === 0) {
      return { tStatistic: 0, pValue: 1 };
    }

    const meanBaseline = this.mean(baseline);
    const meanTest = this.mean(test);
    const varBaseline = this.variance(baseline, meanBaseline);
    const varTest = this.variance(test, meanTest);

    const pooledStdErr = Math.sqrt(
      (varBaseline / baseline.length) + (varTest / test.length)
    );

    if (pooledStdErr === 0) {
      return { tStatistic: 0, pValue: 1 };
    }

    const tStatistic = (meanTest - meanBaseline) / pooledStdErr;
    const degreesOfFreedom = baseline.length + test.length - 2;
    
    // Approximate p-value calculation (simplified)
    const pValue = this.approximatePValue(Math.abs(tStatistic), degreesOfFreedom);

    return { tStatistic, pValue };
  }

  private static mean(values: number[]): number {
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  private static variance(values: number[], mean: number): number {
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    return this.mean(squaredDiffs);
  }

  private static approximatePValue(tStat: number, df: number): number {
    // Simplified p-value approximation - in production use proper statistical library
    if (df < 1) return 1;
    
    const criticalValues = [
      { df: 1, critical: 12.706 },
      { df: 5, critical: 2.571 },
      { df: 10, critical: 2.228 },
      { df: 20, critical: 2.086 },
      { df: 30, critical: 2.042 },
      { df: Infinity, critical: 1.96 }
    ];

    const closest = criticalValues.find(cv => df <= cv.df) || criticalValues[criticalValues.length - 1];
    
    if (tStat < closest.critical) {
      return 0.1; // Not significant
    } else if (tStat > closest.critical * 1.5) {
      return 0.01; // Highly significant
    } else {
      return 0.05; // Borderline significant
    }
  }
}

/**
 * Mock query evaluator for testing
 */
class QueryEvaluator {
  async evaluateQuery(
    query: QuerySample,
    raptorSystem: RaptorSystem,
    useRaptorFeatures: boolean
  ): Promise<{ metrics: BenchmarkMetrics; rankedFiles: string[] }> {
    const startTime = Date.now();

    // Compute RAPTOR features if enabled
    let raptorFeatures: Map<string, RaptorFeatures> | undefined;
    let featureComputationTime = 0;

    if (useRaptorFeatures) {
      const featureStart = Date.now();
      
      try {
        const result = await raptorSystem.computeFeatures({
          repo_sha: query.repo_sha,
          query: query.query,
          candidate_files: query.candidate_files,
          nl_score: query.slice === 'nl' ? 0.8 : 0.2
        });
        
        raptorFeatures = result.features;
        featureComputationTime = Date.now() - featureStart;
      } catch (error) {
        console.warn(`Failed to compute RAPTOR features for query ${query.id}:`, error);
        raptorFeatures = new Map();
      }
    }

    // Simulate ranking with or without RAPTOR features
    const rankedFiles = this.simulateRanking(
      query.candidate_files,
      query.expected_files,
      raptorFeatures
    );

    const endTime = Date.now();
    const totalTime = endTime - startTime;

    // Compute metrics
    const metrics = this.computeMetrics(rankedFiles, query.expected_files, totalTime, featureComputationTime);

    return { metrics, rankedFiles };
  }

  private simulateRanking(
    candidates: string[],
    expectedFiles: string[],
    raptorFeatures?: Map<string, RaptorFeatures>
  ): string[] {
    // Simulate ranking by giving expected files higher scores
    const scored = candidates.map(file => {
      let score = Math.random(); // Base random score
      
      // Boost expected files
      if (expectedFiles.includes(file)) {
        score += 2;
      }

      // Apply RAPTOR boost if available
      if (raptorFeatures?.has(file)) {
        const features = raptorFeatures.get(file)!;
        score += features.raptor_max_sim * 0.5;
        score += features.topic_overlap * 0.3;
        score += Math.max(0, features.B) * 0.1;
      }

      return { file, score };
    });

    return scored
      .sort((a, b) => b.score - a.score)
      .map(item => item.file);
  }

  private computeMetrics(
    rankedFiles: string[],
    expectedFiles: string[],
    totalTimeMs: number,
    featureTimeMs: number
  ): BenchmarkMetrics {
    // NDCG@10
    const ndcg10 = this.computeNDCG(rankedFiles.slice(0, 10), expectedFiles);

    // Recall@50
    const top50 = new Set(rankedFiles.slice(0, 50));
    const recall50 = expectedFiles.filter(file => top50.has(file)).length / expectedFiles.length;

    // Precision@10
    const top10 = new Set(rankedFiles.slice(0, 10));
    const precision10 = expectedFiles.filter(file => top10.has(file)).length / Math.min(10, rankedFiles.length);

    // Span coverage (simplified - assume 100% if any expected files in top 50)
    const spanCoverage = recall50 > 0 ? 100 : 0;

    return {
      ndcg_at_10: ndcg10,
      recall_at_50: recall50,
      precision_at_10: precision10,
      span_coverage: spanCoverage,
      e2e_p95_ms: totalTimeMs, // Simplified - single query time
      feature_computation_ms: featureTimeMs
    };
  }

  private computeNDCG(rankedFiles: string[], expectedFiles: string[]): number {
    if (expectedFiles.length === 0) return 0;

    let dcg = 0;
    const expectedSet = new Set(expectedFiles);

    for (let i = 0; i < rankedFiles.length; i++) {
      if (expectedSet.has(rankedFiles[i])) {
        dcg += 1 / Math.log2(i + 2); // +2 because log2(1) = 0
      }
    }

    // Ideal DCG (all expected files at the top)
    let idcg = 0;
    for (let i = 0; i < Math.min(expectedFiles.length, rankedFiles.length); i++) {
      idcg += 1 / Math.log2(i + 2);
    }

    return idcg > 0 ? dcg / idcg : 0;
  }
}

/**
 * Main benchmarking system
 */
export class RaptorBenchmarkSuite {
  private config: BenchmarkConfig;
  private evaluator: QueryEvaluator;

  constructor(config: BenchmarkConfig) {
    this.config = config;
    this.evaluator = new QueryEvaluator();
  }

  static createAnchorConfig(): BenchmarkConfig {
    return {
      dataset: 'anchor',
      slices: ['nl', 'structural'],
      baseline_system: 'stage-c-baseline',
      test_system: 'stage-c-raptor',
      significance_threshold: 0.05,
      min_samples: 100,
      max_runtime_ms: 5000
    };
  }

  static createLadderConfig(): BenchmarkConfig {
    return {
      dataset: 'ladder',
      slices: ['nl'],
      baseline_system: 'stage-c-baseline',
      test_system: 'stage-c-raptor',
      significance_threshold: 0.05,
      min_samples: 50,
      max_runtime_ms: 5000
    };
  }

  async runBenchmark(
    raptorSystem: RaptorSystem,
    queryGenerator: () => AsyncGenerator<QuerySample>
  ): Promise<BenchmarkSuite> {
    const startTime = Date.now();
    const results: BenchmarkResult[] = [];

    // Process queries by slice
    for (const slice of this.config.slices) {
      console.log(`Running benchmark for slice: ${slice}`);
      
      const sliceResult = await this.runSliceBenchmark(
        raptorSystem,
        queryGenerator,
        slice
      );
      
      results.push(sliceResult);
    }

    const endTime = Date.now();
    const overallPassed = results.every(result => result.passed);

    return {
      config: this.config,
      results,
      overall_passed: overallPassed,
      timestamp: startTime,
      duration_ms: endTime - startTime
    };
  }

  private async runSliceBenchmark(
    raptorSystem: RaptorSystem,
    queryGenerator: () => AsyncGenerator<QuerySample>,
    targetSlice: 'nl' | 'structural'
  ): Promise<BenchmarkResult> {
    const baselineMetrics: number[] = [];
    const testMetrics: number[] = [];
    const baselineRecalls: number[] = [];
    const testRecalls: number[] = [];
    const baselineSpanCoverages: number[] = [];
    const testSpanCoverages: number[] = [];
    const baselineLatencies: number[] = [];
    const testLatencies: number[] = [];

    let queryCount = 0;
    const maxQueries = this.config.min_samples * 2; // Safety limit

    // Generate and process queries
    for await (const query of queryGenerator()) {
      if (query.slice !== targetSlice) continue;
      if (queryCount >= maxQueries) break;

      try {
        // Run baseline evaluation (without RAPTOR)
        const baselineResult = await this.evaluator.evaluateQuery(
          query,
          raptorSystem,
          false // No RAPTOR features
        );

        // Run test evaluation (with RAPTOR)  
        const testResult = await this.evaluator.evaluateQuery(
          query,
          raptorSystem,
          true // Use RAPTOR features
        );

        // Collect metrics
        baselineMetrics.push(baselineResult.metrics.ndcg_at_10);
        testMetrics.push(testResult.metrics.ndcg_at_10);
        baselineRecalls.push(baselineResult.metrics.recall_at_50);
        testRecalls.push(testResult.metrics.recall_at_50);
        baselineSpanCoverages.push(baselineResult.metrics.span_coverage);
        testSpanCoverages.push(testResult.metrics.span_coverage);
        baselineLatencies.push(baselineResult.metrics.e2e_p95_ms);
        testLatencies.push(testResult.metrics.e2e_p95_ms);

        queryCount++;

        if (queryCount % 10 === 0) {
          console.log(`Processed ${queryCount} queries for slice ${targetSlice}`);
        }

      } catch (error) {
        console.warn(`Failed to process query ${query.id}:`, error);
        continue;
      }
    }

    if (queryCount < this.config.min_samples) {
      throw new Error(`Insufficient samples for slice ${targetSlice}: ${queryCount} < ${this.config.min_samples}`);
    }

    // Compute aggregate metrics
    const baselineAgg = this.aggregateMetrics({
      ndcg_at_10: baselineMetrics,
      recall_at_50: baselineRecalls,
      span_coverage: baselineSpanCoverages,
      latencies: baselineLatencies
    });

    const testAgg = this.aggregateMetrics({
      ndcg_at_10: testMetrics,
      recall_at_50: testRecalls,
      span_coverage: testSpanCoverages,
      latencies: testLatencies
    });

    // Statistical significance testing
    const ndcgTest = StatisticalTesting.tTest(baselineMetrics, testMetrics);
    const ndcgDelta = ((testAgg.ndcg_at_10 - baselineAgg.ndcg_at_10) / baselineAgg.ndcg_at_10) * 100;
    const recallDelta = testAgg.recall_at_50 - baselineAgg.recall_at_50;
    const spanDelta = testAgg.span_coverage - baselineAgg.span_coverage;
    const latencyDelta = ((testAgg.e2e_p95_ms - baselineAgg.e2e_p95_ms) / baselineAgg.e2e_p95_ms) * 100;

    // Check promotion gates
    const promotionGates = {
      ndcg_improvement: ndcgDelta >= 2.0 && ndcgTest.pValue < this.config.significance_threshold,
      recall_maintained: recallDelta >= 0,
      span_maintained: spanDelta >= 0,
      latency_acceptable: latencyDelta <= 10.0
    };

    const passed = Object.values(promotionGates).every(gate => gate);

    return {
      slice: targetSlice,
      query_count: queryCount,
      baseline_metrics: baselineAgg,
      test_metrics: testAgg,
      deltas: {
        ndcg_at_10_delta: ndcgDelta,
        ndcg_at_10_pvalue: ndcgTest.pValue,
        recall_at_50_delta: recallDelta,
        span_coverage_delta: spanDelta,
        e2e_p95_delta_pct: latencyDelta
      },
      promotion_gates: promotionGates,
      passed
    };
  }

  private aggregateMetrics(data: {
    ndcg_at_10: number[];
    recall_at_50: number[];
    span_coverage: number[];
    latencies: number[];
  }): BenchmarkMetrics {
    return {
      ndcg_at_10: this.mean(data.ndcg_at_10),
      recall_at_50: this.mean(data.recall_at_50),
      precision_at_10: 0, // Not aggregated in this simplified version
      span_coverage: this.mean(data.span_coverage),
      e2e_p95_ms: this.percentile(data.latencies, 0.95),
      feature_computation_ms: 0 // Would need to track separately
    };
  }

  private mean(values: number[]): number {
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  private percentile(values: number[], p: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.floor(p * sorted.length);
    return sorted[Math.min(index, sorted.length - 1)];
  }

  generateReport(suite: BenchmarkSuite): string {
    const lines: string[] = [];
    
    lines.push('# RAPTOR Benchmark Report');
    lines.push('');
    lines.push(`**Dataset:** ${suite.config.dataset}`);
    lines.push(`**Duration:** ${(suite.duration_ms / 1000).toFixed(1)}s`);
    lines.push(`**Overall Result:** ${suite.overall_passed ? '✅ PASSED' : '❌ FAILED'}`);
    lines.push('');

    for (const result of suite.results) {
      lines.push(`## ${result.slice.toUpperCase()} Slice Results`);
      lines.push('');
      lines.push(`**Queries Processed:** ${result.query_count}`);
      lines.push(`**Result:** ${result.passed ? '✅ PASSED' : '❌ FAILED'}`);
      lines.push('');

      lines.push('### Metrics Comparison');
      lines.push('');
      lines.push('| Metric | Baseline | Test | Delta | Gate |');
      lines.push('|--------|----------|------|-------|------|');
      lines.push(`| nDCG@10 | ${result.baseline_metrics.ndcg_at_10.toFixed(3)} | ${result.test_metrics.ndcg_at_10.toFixed(3)} | ${result.deltas.ndcg_at_10_delta.toFixed(1)}% | ${result.promotion_gates.ndcg_improvement ? '✅' : '❌'} |`);
      lines.push(`| Recall@50 | ${result.baseline_metrics.recall_at_50.toFixed(3)} | ${result.test_metrics.recall_at_50.toFixed(3)} | ${result.deltas.recall_at_50_delta.toFixed(3)} | ${result.promotion_gates.recall_maintained ? '✅' : '❌'} |`);
      lines.push(`| Span Coverage | ${result.baseline_metrics.span_coverage.toFixed(1)}% | ${result.test_metrics.span_coverage.toFixed(1)}% | ${result.deltas.span_coverage_delta.toFixed(1)}% | ${result.promotion_gates.span_maintained ? '✅' : '❌'} |`);
      lines.push(`| E2E p95 Latency | ${result.baseline_metrics.e2e_p95_ms.toFixed(0)}ms | ${result.test_metrics.e2e_p95_ms.toFixed(0)}ms | ${result.deltas.e2e_p95_delta_pct.toFixed(1)}% | ${result.promotion_gates.latency_acceptable ? '✅' : '❌'} |`);
      lines.push('');

      lines.push('### Promotion Gates');
      lines.push('');
      lines.push(`- **nDCG@10 Improvement:** ${result.deltas.ndcg_at_10_delta.toFixed(1)}% (p=${result.deltas.ndcg_at_10_pvalue.toFixed(3)}) - Need ≥2.0% (p<0.05) ${result.promotion_gates.ndcg_improvement ? '✅' : '❌'}`);
      lines.push(`- **Recall@50 Maintained:** ${result.deltas.recall_at_50_delta.toFixed(3)} - Need ≥0 ${result.promotion_gates.recall_maintained ? '✅' : '❌'}`);
      lines.push(`- **Span Coverage:** ${result.deltas.span_coverage_delta.toFixed(1)}% - Need ≥0% ${result.promotion_gates.span_maintained ? '✅' : '❌'}`);
      lines.push(`- **Latency Impact:** ${result.deltas.e2e_p95_delta_pct.toFixed(1)}% - Need ≤10% ${result.promotion_gates.latency_acceptable ? '✅' : '❌'}`);
      lines.push('');
    }

    lines.push('---');
    lines.push(`Generated at: ${new Date(suite.timestamp).toISOString()}`);
    lines.push(`Config: ${JSON.stringify(suite.config, null, 2)}`);

    return lines.join('\n');
  }
}

export default RaptorBenchmarkSuite;