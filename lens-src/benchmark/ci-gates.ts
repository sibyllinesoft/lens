/**
 * CI/CD Quality Gates for Lens Benchmarking
 * Phase 5: Benchmark Hardening & CI Gates
 * 
 * Implements preflight checks, performance tripwires, and CI integration
 * to prevent quality regressions from reaching production.
 */

import { z } from 'zod';
import { promises as fs } from 'fs';
import path from 'path';
import type { BenchmarkRun, GoldenDataItem } from '../types/benchmark.js';

// Performance tripwire thresholds
export const PERFORMANCE_TRIPWIRES = {
  // Ranking failure detection
  recall_convergence_threshold: 0.005, // Fail if Recall@50 ≈ Recall@10 within 0.5%
  
  // Coverage quality
  span_coverage_min: 0.98, // Fail if span coverage < 98%
  
  // Quality degradation thresholds  
  ndcg_regression_max: 0.02, // Fail if nDCG@10 drops >2%
  recall_regression_max: 0.01, // Fail if Recall@50 drops >1%
  
  // Latency SLA thresholds
  stage_a_p95_max: 250, // ms
  stage_b_p95_max: 350, // ms 
  stage_c_p95_max: 350, // ms
  e2e_p95_max: 800, // ms
  
  // Stability thresholds
  error_rate_max: 0.02, // 2% maximum error rate
  timeout_rate_max: 0.01, // 1% maximum timeout rate
} as const;

// Gate result schemas
export const PreflightResultSchema = z.object({
  passed: z.boolean(),
  consistency_check: z.object({
    passed: z.boolean(),
    total_golden_items: z.number(),
    valid_results: z.number(),
    inconsistent_results: z.number(),
    pass_rate: z.number(),
    corpus_file_count: z.number()
  }),
  blocking_issues: z.array(z.string()),
  warnings: z.array(z.string()),
  artifacts: z.object({
    inconsistency_report: z.string().optional()
  })
});

export const PerformanceGateResultSchema = z.object({
  passed: z.boolean(),
  tripwires_triggered: z.array(z.object({
    name: z.string(),
    threshold: z.number(),
    actual: z.number(),
    severity: z.enum(['error', 'warning'])
  })),
  baseline_comparison: z.object({
    ndcg_delta: z.number(),
    recall_delta: z.number(),
    latency_delta_percent: z.number()
  }),
  coverage_analysis: z.object({
    span_coverage: z.number(),
    candidate_coverage: z.number(),
    ranking_quality: z.number()
  })
});

export type PreflightResult = z.infer<typeof PreflightResultSchema>;
export type PerformanceGateResult = z.infer<typeof PerformanceGateResultSchema>;

export class CIGatesOrchestrator {
  constructor(
    private readonly outputDir: string,
    private readonly goldenItems: GoldenDataItem[]
  ) {}

  /**
   * Run preflight consistency checks - MUST pass before any benchmarks
   */
  async runPreflightChecks(): Promise<PreflightResult> {
    console.log('🔍 Running preflight consistency checks...');
    
    const inconsistencies: any[] = [];
    let validItems = 0;
    const blockingIssues: string[] = [];
    const warnings: string[] = [];
    
    // Get corpus file inventory
    const corpusFiles = await this.getCorpusInventory();
    console.log(`📊 Corpus contains ${corpusFiles.size} indexed files`);
    
    // Check each golden item against corpus
    for (const item of this.goldenItems) {
      for (const expectedResult of item.expected_results) {
        const filePath = expectedResult.file;
        const exists = this.checkFileInCorpus(filePath, corpusFiles);
        
        if (!exists) {
          inconsistencies.push({
            golden_item_id: item.id,
            query: item.query,
            expected_file: filePath,
            line: expectedResult.line,
            col: expectedResult.col,
            issue: 'file_not_in_corpus',
            severity: this.categorizeMissingFile(filePath)
          });
        } else {
          validItems++;
        }
      }
    }
    
    const totalExpected = this.goldenItems.reduce((sum, item) => sum + item.expected_results.length, 0);
    const passRate = validItems / Math.max(totalExpected, 1);
    
    // Determine blocking vs warning inconsistencies
    const criticalInconsistencies = inconsistencies.filter(inc => inc.severity === 'critical');
    const warningInconsistencies = inconsistencies.filter(inc => inc.severity === 'warning');
    
    if (criticalInconsistencies.length > 0) {
      blockingIssues.push(`${criticalInconsistencies.length} critical golden items reference missing files`);
    }
    
    if (warningInconsistencies.length > 0) {
      warnings.push(`${warningInconsistencies.length} golden items reference files that may be stale`);
    }
    
    if (passRate < 0.95) {
      blockingIssues.push(`Golden-corpus alignment too low: ${(passRate * 100).toFixed(1)}% (min 95%)`);
    }
    
    // Write inconsistency report
    const artifacts: any = {};
    if (inconsistencies.length > 0) {
      const reportPath = path.join(this.outputDir, 'inconsistency.ndjson');
      const ndjsonLines = inconsistencies.map(inc => JSON.stringify(inc));
      await fs.writeFile(reportPath, ndjsonLines.join('\n'));
      artifacts.inconsistency_report = reportPath;
      
      console.log(`📄 Inconsistency report: ${reportPath}`);
    }
    
    const passed = blockingIssues.length === 0;
    const result: PreflightResult = {
      passed,
      consistency_check: {
        passed: criticalInconsistencies.length === 0,
        total_golden_items: this.goldenItems.length,
        valid_results: validItems,
        inconsistent_results: inconsistencies.length,
        pass_rate: passRate,
        corpus_file_count: corpusFiles.size
      },
      blocking_issues: blockingIssues,
      warnings,
      artifacts
    };
    
    if (passed) {
      console.log(`✅ Preflight checks PASSED: ${validItems}/${totalExpected} golden items aligned`);
    } else {
      console.log(`❌ Preflight checks FAILED: ${blockingIssues.length} blocking issues`);
      blockingIssues.forEach(issue => console.log(`  🚫 ${issue}`));
    }
    
    return result;
  }

  /**
   * Run performance tripwires - detect ranking failures and quality degradation
   */
  async runPerformanceTripwires(
    currentResults: BenchmarkRun[],
    baselineResults: BenchmarkRun[] = []
  ): Promise<PerformanceGateResult> {
    console.log('⚡ Running performance tripwires...');
    
    const tripwiresTriggered: any[] = [];
    
    // Analyze each benchmark run
    for (const run of currentResults) {
      await this.checkRankingQuality(run, tripwiresTriggered);
      await this.checkCoverageQuality(run, tripwiresTriggered);  
      await this.checkLatencyThresholds(run, tripwiresTriggered);
      await this.checkStabilityMetrics(run, tripwiresTriggered);
    }
    
    // Compare against baseline if provided
    let baselineComparison = {
      ndcg_delta: 0,
      recall_delta: 0,
      latency_delta_percent: 0
    };
    
    if (baselineResults.length > 0 && currentResults.length > 0) {
      baselineComparison = await this.compareAgainstBaseline(currentResults, baselineResults, tripwiresTriggered);
    }
    
    // Coverage analysis
    const coverageAnalysis = await this.analyzeCoverage(currentResults);
    
    const errorTripwires = tripwiresTriggered.filter(t => t.severity === 'error');
    const passed = errorTripwires.length === 0;
    
    const result: PerformanceGateResult = {
      passed,
      tripwires_triggered: tripwiresTriggered,
      baseline_comparison: baselineComparison,
      coverage_analysis: coverageAnalysis
    };
    
    if (passed) {
      console.log(`✅ Performance gates PASSED: ${tripwiresTriggered.length} tripwires checked`);
    } else {
      console.log(`❌ Performance gates FAILED: ${errorTripwires.length} error tripwires triggered`);
      errorTripwires.forEach(tripwire => {
        console.log(`  🚨 ${tripwire.name}: ${tripwire.actual} (threshold: ${tripwire.threshold})`);
      });
    }
    
    return result;
  }

  /**
   * Generate comprehensive test report with all required artifacts
   */
  async generateTestReport(
    preflightResult: PreflightResult,
    performanceResult: PerformanceGateResult,
    benchmarkRuns: BenchmarkRun[],
    testType: 'smoke' | 'full'
  ): Promise<{
    artifacts: {
      metrics_parquet: string;
      errors_ndjson: string;
      traces_ndjson: string;
      report_pdf: string;
      summary_json: string;
    }
  }> {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const baseFilename = `${testType}_${timestamp}`;
    
    const artifacts = {
      metrics_parquet: path.join(this.outputDir, `${baseFilename}_metrics.parquet`),
      errors_ndjson: path.join(this.outputDir, `${baseFilename}_errors.ndjson`),
      traces_ndjson: path.join(this.outputDir, `${baseFilename}_traces.ndjson`),
      report_pdf: path.join(this.outputDir, `${baseFilename}_report.pdf`),
      summary_json: path.join(this.outputDir, `${baseFilename}_summary.json`)
    };
    
    // Generate metrics data (Parquet format - for now save as JSON)
    const metricsData = {
      test_metadata: {
        test_type: testType,
        timestamp,
        total_runs: benchmarkRuns.length,
        gate_results: {
          preflight_passed: preflightResult.passed,
          performance_passed: performanceResult.passed
        }
      },
      benchmark_runs: benchmarkRuns.map(run => ({
        trace_id: run.trace_id,
        system: run.system,
        metrics: run.metrics,
        status: run.status,
        error_count: run.errors.length
      })),
      quality_gates: {
        preflight: preflightResult,
        performance: performanceResult
      }
    };
    
    await fs.writeFile(artifacts.metrics_parquet + '.json', JSON.stringify(metricsData, null, 2));
    
    // Generate consolidated errors file
    const allErrors = benchmarkRuns.flatMap(run => 
      run.errors.map(error => ({
        ...error,
        trace_id: run.trace_id,
        system: run.system,
        timestamp: run.timestamp
      }))
    );
    
    const errorsNdjson = allErrors.map(error => JSON.stringify(error)).join('\n');
    await fs.writeFile(artifacts.errors_ndjson, errorsNdjson);
    
    // Generate traces file
    const traces = this.generateTracesData(benchmarkRuns);
    const tracesNdjson = traces.map(trace => JSON.stringify(trace)).join('\n');
    await fs.writeFile(artifacts.traces_ndjson, tracesNdjson);
    
    // Generate summary report
    const summary = {
      test_type: testType,
      timestamp,
      results: {
        preflight: preflightResult,
        performance: performanceResult,
        overall_passed: preflightResult.passed && performanceResult.passed
      },
      benchmark_summary: {
        total_runs: benchmarkRuns.length,
        completed_runs: benchmarkRuns.filter(r => r.status === 'completed').length,
        failed_runs: benchmarkRuns.filter(r => r.status === 'failed').length,
        total_queries: benchmarkRuns.reduce((sum, r) => sum + r.total_queries, 0),
        total_errors: allErrors.length
      },
      quality_metrics: this.calculateQualityMetrics(benchmarkRuns),
      artifacts
    };
    
    await fs.writeFile(artifacts.summary_json, JSON.stringify(summary, null, 2));
    
    // Generate PDF report placeholder (would use actual PDF library)
    await fs.writeFile(artifacts.report_pdf, `${testType.toUpperCase()} Test Report - ${timestamp}\n\nGenerated by Lens CI Gates`);
    
    console.log(`📋 Test report generated:`);
    console.log(`  📊 Metrics: ${artifacts.metrics_parquet}.json`);
    console.log(`  🚨 Errors: ${artifacts.errors_ndjson}`);  
    console.log(`  📈 Traces: ${artifacts.traces_ndjson}`);
    console.log(`  📄 Summary: ${artifacts.summary_json}`);
    
    return { artifacts };
  }

  // Private helper methods

  private async getCorpusInventory(): Promise<Set<string>> {
    const indexedFiles = new Set<string>();
    
    try {
      const indexedDir = path.join(process.cwd(), 'indexed-content');
      const files = await fs.readdir(indexedDir);
      
      for (const file of files) {
        if (file.endsWith('.py') || file.endsWith('.ts') || file.endsWith('.js')) {
          // Add both original and flattened filename formats
          const originalPath = file.replace(/[_]/g, '/');
          indexedFiles.add(originalPath);
          indexedFiles.add(file);
        }
      }
    } catch (error) {
      console.warn('Could not read indexed-content directory:', error);
    }
    
    return indexedFiles;
  }

  private checkFileInCorpus(filePath: string, corpusFiles: Set<string>): boolean {
    return corpusFiles.has(filePath) || 
           corpusFiles.has(path.basename(filePath)) ||
           corpusFiles.has(filePath.replace(/\//g, '_'));
  }

  private categorizeMissingFile(filePath: string): 'critical' | 'warning' {
    // Critical: Core source files that should definitely be in corpus
    if (filePath.includes('src/') && (filePath.endsWith('.ts') || filePath.endsWith('.js') || filePath.endsWith('.py'))) {
      return 'critical';
    }
    
    // Warning: Test files, build artifacts, etc.
    if (filePath.includes('test') || filePath.includes('dist/') || filePath.includes('node_modules/')) {
      return 'warning';
    }
    
    return 'critical';
  }

  private async checkRankingQuality(run: BenchmarkRun, tripwires: any[]): Promise<void> {
    const { recall_at_10, recall_at_50 } = run.metrics;
    
    // Check for ranking failure: Recall@50 ≈ Recall@10 within threshold
    const recallConvergence = Math.abs(recall_at_50 - recall_at_10);
    
    if (recallConvergence < PERFORMANCE_TRIPWIRES.recall_convergence_threshold) {
      tripwires.push({
        name: 'ranking_failure_detected',
        threshold: PERFORMANCE_TRIPWIRES.recall_convergence_threshold,
        actual: recallConvergence,
        severity: 'error'
      });
    }
  }

  private async checkCoverageQuality(run: BenchmarkRun, tripwires: any[]): Promise<void> {
    // Calculate span coverage from fan-out data
    const { stage_a, stage_b, stage_c } = run.metrics.fan_out_sizes;
    const totalCandidates = stage_a + stage_b + (stage_c || 0);
    const actualResults = run.completed_queries;
    
    const spanCoverage = actualResults > 0 ? Math.min(totalCandidates / (actualResults * 100), 1) : 0;
    
    if (spanCoverage < PERFORMANCE_TRIPWIRES.span_coverage_min) {
      tripwires.push({
        name: 'span_coverage_low',
        threshold: PERFORMANCE_TRIPWIRES.span_coverage_min,
        actual: spanCoverage,
        severity: 'error'
      });
    }
  }

  private async checkLatencyThresholds(run: BenchmarkRun, tripwires: any[]): Promise<void> {
    const latencies = run.metrics.stage_latencies;
    
    const checks = [
      { name: 'stage_a_p95_latency', actual: latencies.stage_a_p95, threshold: PERFORMANCE_TRIPWIRES.stage_a_p95_max },
      { name: 'stage_b_p95_latency', actual: latencies.stage_b_p95, threshold: PERFORMANCE_TRIPWIRES.stage_b_p95_max },
      { name: 'stage_c_p95_latency', actual: latencies.stage_c_p95 || 0, threshold: PERFORMANCE_TRIPWIRES.stage_c_p95_max },
      { name: 'e2e_p95_latency', actual: latencies.e2e_p95, threshold: PERFORMANCE_TRIPWIRES.e2e_p95_max }
    ];
    
    for (const check of checks) {
      if (check.actual > check.threshold) {
        tripwires.push({
          name: check.name,
          threshold: check.threshold,
          actual: check.actual,
          severity: 'error'
        });
      }
    }
  }

  private async checkStabilityMetrics(run: BenchmarkRun, tripwires: any[]): Promise<void> {
    const errorRate = run.failed_queries / Math.max(run.total_queries, 1);
    
    if (errorRate > PERFORMANCE_TRIPWIRES.error_rate_max) {
      tripwires.push({
        name: 'error_rate_high',
        threshold: PERFORMANCE_TRIPWIRES.error_rate_max,
        actual: errorRate,
        severity: 'error'
      });
    }
  }

  private async compareAgainstBaseline(
    currentResults: BenchmarkRun[],
    baselineResults: BenchmarkRun[],
    tripwires: any[]
  ): Promise<{ ndcg_delta: number; recall_delta: number; latency_delta_percent: number }> {
    
    // Average metrics across runs
    const currentAvg = this.averageMetrics(currentResults);
    const baselineAvg = this.averageMetrics(baselineResults);
    
    const ndcgDelta = currentAvg.ndcg_at_10 - baselineAvg.ndcg_at_10;
    const recallDelta = currentAvg.recall_at_50 - baselineAvg.recall_at_50;
    const latencyDeltaPercent = (currentAvg.e2e_p95 - baselineAvg.e2e_p95) / baselineAvg.e2e_p95;
    
    // Check for regressions
    if (ndcgDelta < -PERFORMANCE_TRIPWIRES.ndcg_regression_max) {
      tripwires.push({
        name: 'ndcg_regression',
        threshold: -PERFORMANCE_TRIPWIRES.ndcg_regression_max,
        actual: ndcgDelta,
        severity: 'error'
      });
    }
    
    if (recallDelta < -PERFORMANCE_TRIPWIRES.recall_regression_max) {
      tripwires.push({
        name: 'recall_regression',
        threshold: -PERFORMANCE_TRIPWIRES.recall_regression_max,
        actual: recallDelta,
        severity: 'error'
      });
    }
    
    return {
      ndcg_delta: ndcgDelta,
      recall_delta: recallDelta,
      latency_delta_percent: latencyDeltaPercent
    };
  }

  private async analyzeCoverage(results: BenchmarkRun[]): Promise<{
    span_coverage: number;
    candidate_coverage: number;
    ranking_quality: number;
  }> {
    const avgMetrics = this.averageMetrics(results);
    
    // Calculate coverage metrics
    const spanCoverage = avgMetrics.recall_at_50; // Proxy for span coverage
    const candidateCoverage = avgMetrics.recall_at_10; // Proxy for candidate coverage
    const rankingQuality = avgMetrics.ndcg_at_10; // Direct ranking quality measure
    
    return {
      span_coverage: spanCoverage,
      candidate_coverage: candidateCoverage,
      ranking_quality: rankingQuality
    };
  }

  private averageMetrics(results: BenchmarkRun[]): any {
    if (results.length === 0) return {};
    
    const totals = results.reduce((acc, run) => {
      acc.ndcg_at_10 += run.metrics.ndcg_at_10;
      acc.recall_at_10 += run.metrics.recall_at_10;
      acc.recall_at_50 += run.metrics.recall_at_50;
      acc.e2e_p95 += run.metrics.stage_latencies.e2e_p95;
      return acc;
    }, { ndcg_at_10: 0, recall_at_10: 0, recall_at_50: 0, e2e_p95: 0 });
    
    const count = results.length;
    return {
      ndcg_at_10: totals.ndcg_at_10 / count,
      recall_at_10: totals.recall_at_10 / count,
      recall_at_50: totals.recall_at_50 / count,
      e2e_p95: totals.e2e_p95 / count
    };
  }

  private generateTracesData(runs: BenchmarkRun[]): any[] {
    const traces: any[] = [];
    
    for (const run of runs) {
      traces.push({
        trace_id: run.trace_id,
        system: run.system,
        timestamp: run.timestamp,
        total_queries: run.total_queries,
        latency_profile: run.metrics.stage_latencies,
        error_count: run.errors.length
      });
    }
    
    return traces;
  }

  private calculateQualityMetrics(runs: BenchmarkRun[]): any {
    const avgMetrics = this.averageMetrics(runs);
    const totalErrors = runs.reduce((sum, r) => sum + r.errors.length, 0);
    const totalQueries = runs.reduce((sum, r) => sum + r.total_queries, 0);
    
    return {
      average_ndcg_at_10: avgMetrics.ndcg_at_10,
      average_recall_at_50: avgMetrics.recall_at_50,
      overall_error_rate: totalErrors / Math.max(totalQueries, 1),
      stability_score: 1 - (totalErrors / Math.max(totalQueries, 1)),
      performance_score: Math.min(avgMetrics.ndcg_at_10, avgMetrics.recall_at_50)
    };
  }
}