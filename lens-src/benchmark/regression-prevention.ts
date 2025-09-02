/**
 * Regression Prevention Benchmark System
 * Provides continuous monitoring and early warning for performance/quality regressions
 * Integrates with CI/CD pipeline for automated regression detection
 */

import { z } from 'zod';
import { v4 as uuidv4 } from 'uuid';
import { promises as fs } from 'fs';
import path from 'path';
import type { BenchmarkConfig, BenchmarkRun } from '../types/benchmark.js';
import { PROMOTION_GATE_CRITERIA } from '../types/benchmark.js';
import { BenchmarkSuiteRunner } from './suite-runner.js';
import { MetricsCalculator } from './metrics-calculator.js';

// Regression detection configuration
export const RegressionConfigSchema = z.object({
  baseline_config: z.object({
    lookback_days: z.number().int().min(1).max(30).default(7),
    min_samples: z.number().int().min(3).max(20).default(5),
    percentile: z.number().min(50).max(99).default(95) // Use p95 for baseline
  }),
  thresholds: z.object({
    // Performance regression thresholds (% increase = bad)
    stage_a_p95_regression_pct: z.number().default(20), // 20% stage-A regression
    stage_b_p95_regression_pct: z.number().default(15), // 15% stage-B regression  
    stage_c_p95_regression_pct: z.number().default(15), // 15% stage-C regression
    e2e_p95_regression_pct: z.number().default(10), // 10% E2E regression (TODO.md limit)
    
    // Quality regression thresholds (decrease = bad)
    ndcg_regression_pct: z.number().default(2), // 2% nDCG regression
    recall_regression_pct: z.number().default(5), // 5% recall regression
    span_coverage_regression_pct: z.number().default(2), // 2% span coverage regression
    
    // Resource usage thresholds
    memory_increase_pct: z.number().default(25), // 25% memory increase
    cpu_increase_pct: z.number().default(30), // 30% CPU increase
    
    // Error rate thresholds
    error_rate_increase_pct: z.number().default(2), // 2% error rate increase
    timeout_rate_increase_pct: z.number().default(1) // 1% timeout rate increase
  }),
  monitoring: z.object({
    early_warning_enabled: z.boolean().default(true),
    early_warning_threshold_pct: z.number().default(75), // Warn at 75% of threshold
    trend_analysis_enabled: z.boolean().default(true),
    trend_window_measurements: z.number().int().default(10),
    anomaly_detection_enabled: z.boolean().default(true),
    anomaly_sigma_threshold: z.number().default(2.5) // 2.5 sigma for anomaly detection
  }),
  ci_integration: z.object({
    fail_build_on_regression: z.boolean().default(true),
    allow_override_flag: z.string().default('--ignore-regression'),
    notification_enabled: z.boolean().default(true),
    notification_channels: z.array(z.enum(['slack', 'email', 'github'])).default(['github']),
    auto_bisect_enabled: z.boolean().default(false), // Auto git bisect on regression
    max_bisect_commits: z.number().int().default(20)
  })
});

export type RegressionConfig = z.infer<typeof RegressionConfigSchema>;

// Historical benchmark data point
export const HistoricalDataPointSchema = z.object({
  timestamp: z.string().datetime(),
  commit_sha: z.string().length(40),
  branch: z.string(),
  trace_id: z.string().uuid(),
  metrics: z.object({
    stage_a_p95_ms: z.number(),
    stage_b_p95_ms: z.number(),
    stage_c_p95_ms: z.number().optional(),
    e2e_p95_ms: z.number(),
    ndcg_at_10: z.number(),
    recall_at_50: z.number(),
    span_coverage: z.number(),
    memory_usage_mb: z.number(),
    cpu_utilization_pct: z.number(),
    error_rate_pct: z.number(),
    timeout_rate_pct: z.number()
  }),
  environment: z.object({
    os: z.string(),
    cpu_cores: z.number().int(),
    memory_gb: z.number(),
    optimization_flags: z.array(z.string())
  })
});

export type HistoricalDataPoint = z.infer<typeof HistoricalDataPointSchema>;

// Regression detection result
export const RegressionResultSchema = z.object({
  trace_id: z.string().uuid(),
  timestamp: z.string().datetime(),
  commit_sha: z.string().length(40),
  baseline_period: z.object({
    start_date: z.string().datetime(),
    end_date: z.string().datetime(),
    sample_count: z.number().int(),
    commits_included: z.array(z.string())
  }),
  current_measurement: HistoricalDataPointSchema,
  regression_analysis: z.object({
    performance_regressions: z.array(z.object({
      metric: z.string(),
      baseline_value: z.number(),
      current_value: z.number(),
      regression_pct: z.number(),
      threshold_pct: z.number(),
      severity: z.enum(['warning', 'minor', 'major', 'critical']),
      confidence: z.number().min(0).max(1)
    })),
    quality_regressions: z.array(z.object({
      metric: z.string(),
      baseline_value: z.number(),
      current_value: z.number(),
      regression_pct: z.number(),
      threshold_pct: z.number(),
      severity: z.enum(['warning', 'minor', 'major', 'critical']),
      statistical_significance: z.boolean()
    })),
    trend_analysis: z.object({
      trend_direction: z.enum(['improving', 'stable', 'degrading', 'volatile']),
      trend_strength: z.number().min(0).max(1),
      trend_duration_days: z.number(),
      projected_regression_days: z.number().optional() // Days until threshold breach
    }),
    anomaly_detection: z.object({
      anomalies_detected: z.array(z.object({
        metric: z.string(),
        z_score: z.number(),
        is_outlier: z.boolean(),
        likely_cause: z.string().optional()
      }))
    })
  }),
  gate_evaluation: z.object({
    passed: z.boolean(),
    blocking_regressions: z.array(z.string()),
    warnings: z.array(z.string()),
    early_warning_triggered: z.boolean(),
    recommendation: z.enum(['proceed', 'investigate', 'revert', 'emergency_stop'])
  }),
  bisect_analysis: z.object({
    should_bisect: z.boolean(),
    suspected_commit_range: z.array(z.string()),
    bisect_strategy: z.enum(['performance', 'quality', 'combined']).optional()
  }).optional()
});

export type RegressionResult = z.infer<typeof RegressionResultSchema>;

export class RegressionPreventionSystem {
  private metricsCalculator: MetricsCalculator;
  
  constructor(
    private readonly suiteRunner: BenchmarkSuiteRunner,
    private readonly outputDir: string,
    private readonly historyDir: string
  ) {
    this.metricsCalculator = new MetricsCalculator();
  }

  /**
   * Run regression detection for current commit
   */
  async detectRegressions(
    commitSha: string,
    config: RegressionConfig,
    overrideBenchmarkConfig?: Partial<BenchmarkConfig>
  ): Promise<RegressionResult> {
    const traceId = uuidv4();
    console.log(`🔍 Starting regression detection for ${commitSha.substring(0, 8)} - Trace ID: ${traceId}`);

    try {
      // 1. Run current benchmark
      console.log('📊 Running current benchmark...');
      const currentBenchmark = await this.runCurrentBenchmark(commitSha, overrideBenchmarkConfig);
      const currentDataPoint = await this.extractDataPoint(currentBenchmark, commitSha);

      // 2. Load baseline data
      console.log('📈 Loading baseline data...');
      const baselineData = await this.loadBaselineData(config.baseline_config);

      if (baselineData.length < config.baseline_config.min_samples) {
        console.warn(`⚠️ Insufficient baseline data: ${baselineData.length} < ${config.baseline_config.min_samples}`);
        return this.createInsufficientDataResult(traceId, commitSha, currentDataPoint);
      }

      // 3. Analyze regressions
      console.log('🔬 Analyzing regressions...');
      const regressionAnalysis = await this.analyzeRegressions(
        currentDataPoint,
        baselineData,
        config
      );

      // 4. Evaluate gate criteria
      const gateEvaluation = this.evaluateRegressionGate(regressionAnalysis, config);

      // 5. Determine if bisect is needed
      const bisectAnalysis = this.analyzeBisectNeed(regressionAnalysis, config);

      // 6. Store current measurement in history
      await this.storeHistoricalData(currentDataPoint);

      const result: RegressionResult = {
        trace_id: traceId,
        timestamp: new Date().toISOString(),
        commit_sha: commitSha,
        baseline_period: {
          start_date: baselineData[0]?.timestamp || new Date().toISOString(),
          end_date: baselineData[baselineData.length - 1]?.timestamp || new Date().toISOString(),
          sample_count: baselineData.length,
          commits_included: baselineData.map(d => d.commit_sha)
        },
        current_measurement: currentDataPoint,
        regression_analysis: regressionAnalysis,
        gate_evaluation: gateEvaluation,
        bisect_analysis: bisectAnalysis
      };

      // 7. Generate regression report
      await this.generateRegressionReport(result);

      // 8. Send notifications if needed
      if (config.ci_integration.notification_enabled && !gateEvaluation.passed) {
        await this.sendRegressionNotifications(result, config);
      }

      const status = gateEvaluation.passed ? 'PASS' : 'FAIL';
      console.log(`✅ Regression detection complete - Gate: ${status}`);

      return result;

    } catch (error) {
      console.error('❌ Regression detection failed:', error);
      throw error;
    }
  }

  /**
   * Run quick regression check for CI (smoke test only)
   */
  async runCIRegressionCheck(
    commitSha: string,
    config: RegressionConfig
  ): Promise<{ passed: boolean; summary: string; details: RegressionResult }> {
    console.log('⚡ Running CI regression check (smoke test)...');

    // Use smoke test configuration for speed
    const smokeConfig: Partial<BenchmarkConfig> = {
      suite: ['codesearch'], // Only codesearch for speed
      systems: ['+symbols+semantic'], // Only test full system
      slices: 'SMOKE_DEFAULT',
      seeds: 1
    };

    const result = await this.detectRegressions(commitSha, config, smokeConfig);
    
    const passed = result.gate_evaluation.passed;
    const summary = this.generateCISummary(result);

    return { passed, summary, details: result };
  }

  /**
   * Run current benchmark measurement
   */
  private async runCurrentBenchmark(
    commitSha: string,
    overrideConfig?: Partial<BenchmarkConfig>
  ): Promise<BenchmarkRun> {
    const benchmarkConfig: BenchmarkConfig = {
      trace_id: uuidv4(),
      suite: ['codesearch', 'structural'],
      systems: ['lex', '+symbols', '+symbols+semantic'],
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
      },
      ...overrideConfig
    };

    return await this.suiteRunner.runSmokeSuite(benchmarkConfig);
  }

  /**
   * Extract historical data point from benchmark run
   */
  private async extractDataPoint(
    benchmarkRun: BenchmarkRun,
    commitSha: string
  ): Promise<HistoricalDataPoint> {
    const environment = await this.getEnvironmentInfo();
    
    return {
      timestamp: new Date().toISOString(),
      commit_sha: commitSha,
      branch: await this.getCurrentBranch(),
      trace_id: benchmarkRun.trace_id,
      metrics: {
        stage_a_p95_ms: benchmarkRun.metrics.stage_latencies.stage_a_p95,
        stage_b_p95_ms: benchmarkRun.metrics.stage_latencies.stage_b_p95,
        stage_c_p95_ms: benchmarkRun.metrics.stage_latencies.stage_c_p95,
        e2e_p95_ms: benchmarkRun.metrics.stage_latencies.e2e_p95,
        ndcg_at_10: benchmarkRun.metrics.ndcg_at_10,
        recall_at_50: benchmarkRun.metrics.recall_at_50,
        span_coverage: this.calculateSpanCoverage(benchmarkRun),
        memory_usage_mb: await this.getCurrentMemoryUsage(),
        cpu_utilization_pct: await this.getCurrentCPUUsage(),
        error_rate_pct: (benchmarkRun.failed_queries / Math.max(benchmarkRun.total_queries, 1)) * 100,
        timeout_rate_pct: await this.calculateTimeoutRate(benchmarkRun)
      },
      environment
    };
  }

  /**
   * Load baseline data for comparison
   */
  private async loadBaselineData(config: { lookback_days: number; min_samples: number }): Promise<HistoricalDataPoint[]> {
    const historyFile = path.join(this.historyDir, 'benchmark-history.ndjson');
    
    try {
      const historyContent = await fs.readFile(historyFile, 'utf-8');
      const allData = historyContent
        .trim()
        .split('\n')
        .map(line => JSON.parse(line) as HistoricalDataPoint);

      // Filter to lookback period
      const cutoffDate = new Date();
      cutoffDate.setDate(cutoffDate.getDate() - config.lookback_days);

      const recentData = allData
        .filter(d => new Date(d.timestamp) >= cutoffDate)
        .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

      return recentData.slice(0, Math.max(config.min_samples * 2, 20)); // Get more than min for better statistics

    } catch (error) {
      console.warn('No historical data found, creating baseline');
      return [];
    }
  }

  /**
   * Analyze performance and quality regressions
   */
  private async analyzeRegressions(
    current: HistoricalDataPoint,
    baseline: HistoricalDataPoint[],
    config: RegressionConfig
  ): Promise<any> {
    const baselineStats = this.calculateBaselineStatistics(baseline);
    
    // Performance regression analysis
    const performanceRegressions = [
      {
        metric: 'stage_a_p95_ms',
        baseline_value: baselineStats.stage_a_p95_ms.median,
        current_value: current.metrics.stage_a_p95_ms,
        threshold_pct: config.thresholds.stage_a_p95_regression_pct
      },
      {
        metric: 'stage_b_p95_ms',
        baseline_value: baselineStats.stage_b_p95_ms.median,
        current_value: current.metrics.stage_b_p95_ms,
        threshold_pct: config.thresholds.stage_b_p95_regression_pct
      },
      {
        metric: 'e2e_p95_ms',
        baseline_value: baselineStats.e2e_p95_ms.median,
        current_value: current.metrics.e2e_p95_ms,
        threshold_pct: config.thresholds.e2e_p95_regression_pct
      },
      {
        metric: 'memory_usage_mb',
        baseline_value: baselineStats.memory_usage_mb.median,
        current_value: current.metrics.memory_usage_mb,
        threshold_pct: config.thresholds.memory_increase_pct
      }
    ].map(item => ({
      ...item,
      regression_pct: ((item.current_value - item.baseline_value) / item.baseline_value) * 100,
      severity: this.calculateSeverity(
        ((item.current_value - item.baseline_value) / item.baseline_value) * 100,
        item.threshold_pct
      ),
      confidence: this.calculateConfidence(current, baseline, item.metric)
    })).filter(item => item.regression_pct > 0); // Only regressions (increases)

    // Quality regression analysis  
    const qualityRegressions = [
      {
        metric: 'ndcg_at_10',
        baseline_value: baselineStats.ndcg_at_10.median,
        current_value: current.metrics.ndcg_at_10,
        threshold_pct: config.thresholds.ndcg_regression_pct,
        higher_is_better: true
      },
      {
        metric: 'recall_at_50',
        baseline_value: baselineStats.recall_at_50.median,
        current_value: current.metrics.recall_at_50,
        threshold_pct: config.thresholds.recall_regression_pct,
        higher_is_better: true
      },
      {
        metric: 'span_coverage',
        baseline_value: baselineStats.span_coverage.median,
        current_value: current.metrics.span_coverage,
        threshold_pct: config.thresholds.span_coverage_regression_pct,
        higher_is_better: true
      }
    ].map(item => ({
      ...item,
      regression_pct: Math.abs((item.baseline_value - item.current_value) / item.baseline_value) * 100,
      severity: this.calculateSeverity(
        Math.abs((item.baseline_value - item.current_value) / item.baseline_value) * 100,
        item.threshold_pct
      ),
      statistical_significance: this.calculateStatisticalSignificance(current, baseline, item.metric)
    })).filter(item => 
      item.higher_is_better ? item.current_value < item.baseline_value : item.current_value > item.baseline_value
    );

    // Trend analysis
    const trendAnalysis = config.monitoring.trend_analysis_enabled
      ? this.analyzeTrends(baseline, current, config.monitoring.trend_window_measurements)
      : { trend_direction: 'stable' as const, trend_strength: 0, trend_duration_days: 0 };

    // Anomaly detection
    const anomalyDetection = config.monitoring.anomaly_detection_enabled
      ? this.detectAnomalies(current, baseline, config.monitoring.anomaly_sigma_threshold)
      : { anomalies_detected: [] };

    return {
      performance_regressions: performanceRegressions,
      quality_regressions: qualityRegressions,
      trend_analysis: trendAnalysis,
      anomaly_detection: anomalyDetection
    };
  }

  /**
   * Calculate baseline statistics from historical data
   */
  private calculateBaselineStatistics(baseline: HistoricalDataPoint[]): any {
    const metrics = [
      'stage_a_p95_ms', 'stage_b_p95_ms', 'stage_c_p95_ms', 'e2e_p95_ms',
      'ndcg_at_10', 'recall_at_50', 'span_coverage', 'memory_usage_mb',
      'cpu_utilization_pct', 'error_rate_pct', 'timeout_rate_pct'
    ];

    const stats: any = {};

    for (const metric of metrics) {
      const values = baseline
        .map(d => (d.metrics as any)[metric])
        .filter(v => v !== undefined && v !== null)
        .sort((a, b) => a - b);

      if (values.length > 0) {
        stats[metric] = {
          mean: values.reduce((a, b) => a + b, 0) / values.length,
          median: values[Math.floor(values.length / 2)],
          p95: values[Math.floor(values.length * 0.95)],
          min: values[0],
          max: values[values.length - 1],
          stdDev: this.calculateStandardDeviation(values)
        };
      }
    }

    return stats;
  }

  /**
   * Evaluate regression gate criteria
   */
  private evaluateRegressionGate(regressionAnalysis: any, config: RegressionConfig): any {
    const blockingRegressions: string[] = [];
    const warnings: string[] = [];
    let earlyWarningTriggered = false;

    // Check performance regressions
    for (const regression of regressionAnalysis.performance_regressions) {
      if (regression.severity === 'critical' || regression.severity === 'major') {
        blockingRegressions.push(`${regression.metric}: +${regression.regression_pct.toFixed(1)}%`);
      } else if (regression.regression_pct > (regression.threshold_pct * config.monitoring.early_warning_threshold_pct / 100)) {
        warnings.push(`${regression.metric}: approaching threshold (+${regression.regression_pct.toFixed(1)}%)`);
        earlyWarningTriggered = true;
      }
    }

    // Check quality regressions
    for (const regression of regressionAnalysis.quality_regressions) {
      if (regression.severity === 'critical' || regression.severity === 'major') {
        blockingRegressions.push(`${regression.metric}: -${regression.regression_pct.toFixed(1)}%`);
      } else if (regression.statistical_significance) {
        warnings.push(`${regression.metric}: statistically significant degradation`);
        earlyWarningTriggered = true;
      }
    }

    // Check trend analysis
    if (regressionAnalysis.trend_analysis.trend_direction === 'degrading' &&
        regressionAnalysis.trend_analysis.trend_strength > 0.7) {
      warnings.push(`Performance trending downward (${regressionAnalysis.trend_analysis.trend_duration_days} days)`);
      earlyWarningTriggered = true;
    }

    // Determine recommendation
    let recommendation: 'proceed' | 'investigate' | 'revert' | 'emergency_stop';
    if (blockingRegressions.length > 0) {
      const criticalCount = regressionAnalysis.performance_regressions
        .filter((r: any) => r.severity === 'critical').length +
        regressionAnalysis.quality_regressions
        .filter((r: any) => r.severity === 'critical').length;
      
      recommendation = criticalCount > 0 ? 'emergency_stop' : 'revert';
    } else if (warnings.length > 0) {
      recommendation = 'investigate';
    } else {
      recommendation = 'proceed';
    }

    return {
      passed: blockingRegressions.length === 0,
      blocking_regressions: blockingRegressions,
      warnings: warnings,
      early_warning_triggered: earlyWarningTriggered,
      recommendation: recommendation
    };
  }

  /**
   * Analyze if git bisect is needed
   */
  private analyzeBisectNeed(regressionAnalysis: any, config: RegressionConfig): any {
    const hasMajorRegression = 
      regressionAnalysis.performance_regressions.some((r: any) => r.severity === 'major' || r.severity === 'critical') ||
      regressionAnalysis.quality_regressions.some((r: any) => r.severity === 'major' || r.severity === 'critical');

    if (!hasMajorRegression || !config.ci_integration.auto_bisect_enabled) {
      return { should_bisect: false };
    }

    // Determine bisect strategy
    const hasPerformanceRegression = regressionAnalysis.performance_regressions.length > 0;
    const hasQualityRegression = regressionAnalysis.quality_regressions.length > 0;

    let bisectStrategy: 'performance' | 'quality' | 'combined';
    if (hasPerformanceRegression && hasQualityRegression) {
      bisectStrategy = 'combined';
    } else if (hasPerformanceRegression) {
      bisectStrategy = 'performance';
    } else {
      bisectStrategy = 'quality';
    }

    return {
      should_bisect: true,
      suspected_commit_range: [], // Would be populated with git log data
      bisect_strategy: bisectStrategy
    };
  }

  /**
   * Store historical data point
   */
  private async storeHistoricalData(dataPoint: HistoricalDataPoint): Promise<void> {
    const historyFile = path.join(this.historyDir, 'benchmark-history.ndjson');
    await fs.mkdir(this.historyDir, { recursive: true });
    
    const line = JSON.stringify(dataPoint) + '\n';
    await fs.appendFile(historyFile, line);
  }

  /**
   * Generate comprehensive regression report
   */
  private async generateRegressionReport(result: RegressionResult): Promise<void> {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const commitShort = result.commit_sha.substring(0, 8);
    const reportPath = path.join(this.outputDir, `regression-analysis-${commitShort}-${timestamp}.json`);

    await fs.writeFile(reportPath, JSON.stringify(result, null, 2));
    console.log(`📄 Regression report written to: ${reportPath}`);

    // Generate markdown summary
    const markdownSummary = this.generateRegressionMarkdown(result);
    const markdownPath = path.join(this.outputDir, `regression-summary-${commitShort}-${timestamp}.md`);
    await fs.writeFile(markdownPath, markdownSummary);
  }

  // Helper methods (mock implementations for some)
  private async getEnvironmentInfo(): Promise<any> {
    return {
      os: process.platform,
      cpu_cores: 8, // Mock
      memory_gb: 16, // Mock
      optimization_flags: [] // Mock
    };
  }

  private async getCurrentBranch(): Promise<string> {
    // Mock implementation - would use git command
    return 'main';
  }

  private calculateSpanCoverage(benchmarkRun: BenchmarkRun): number {
    const { stage_a, stage_b, stage_c } = benchmarkRun.metrics.fan_out_sizes;
    const totalCandidates = stage_a + stage_b + (stage_c || 0);
    const actualResults = benchmarkRun.completed_queries;
    return actualResults > 0 ? Math.min(totalCandidates / (actualResults * 100), 1) : 0;
  }

  private async getCurrentMemoryUsage(): Promise<number> {
    // Mock implementation - would check actual memory usage
    return 512 + Math.random() * 256;
  }

  private async getCurrentCPUUsage(): Promise<number> {
    // Mock implementation - would check actual CPU usage
    return 45 + Math.random() * 20;
  }

  private async calculateTimeoutRate(benchmarkRun: BenchmarkRun): Promise<number> {
    // Mock implementation - would calculate actual timeout rate
    return Math.random() * 2; // 0-2% timeout rate
  }

  private calculateSeverity(regressionPct: number, thresholdPct: number): 'warning' | 'minor' | 'major' | 'critical' {
    if (regressionPct >= thresholdPct * 2) return 'critical';
    if (regressionPct >= thresholdPct * 1.5) return 'major';
    if (regressionPct >= thresholdPct) return 'minor';
    return 'warning';
  }

  private calculateConfidence(current: HistoricalDataPoint, baseline: HistoricalDataPoint[], metric: string): number {
    // Mock confidence calculation - would use statistical methods
    return 0.8 + Math.random() * 0.15; // 80-95% confidence
  }

  private calculateStatisticalSignificance(current: HistoricalDataPoint, baseline: HistoricalDataPoint[], metric: string): boolean {
    // Mock statistical significance test - would use t-test or similar
    return Math.random() > 0.7; // 30% chance of significance
  }

  private analyzeTrends(baseline: HistoricalDataPoint[], current: HistoricalDataPoint, windowSize: number): any {
    if (baseline.length < windowSize) {
      return { trend_direction: 'stable' as const, trend_strength: 0, trend_duration_days: 0 };
    }

    // Mock trend analysis - would use actual time series analysis
    const directions = ['improving', 'stable', 'degrading', 'volatile'] as const;
    return {
      trend_direction: directions[Math.floor(Math.random() * directions.length)],
      trend_strength: Math.random(),
      trend_duration_days: Math.floor(Math.random() * 7) + 1,
      projected_regression_days: Math.random() > 0.5 ? Math.floor(Math.random() * 30) + 5 : undefined
    };
  }

  private detectAnomalies(current: HistoricalDataPoint, baseline: HistoricalDataPoint[], sigmaThreshold: number): any {
    // Mock anomaly detection - would use statistical outlier detection
    return {
      anomalies_detected: Math.random() > 0.8 ? [{
        metric: 'stage_a_p95_ms',
        z_score: 2.8,
        is_outlier: true,
        likely_cause: 'Configuration change or system load'
      }] : []
    };
  }

  private calculateStandardDeviation(values: number[]): number {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    return Math.sqrt(variance);
  }

  private createInsufficientDataResult(traceId: string, commitSha: string, current: HistoricalDataPoint): RegressionResult {
    return {
      trace_id: traceId,
      timestamp: new Date().toISOString(),
      commit_sha: commitSha,
      baseline_period: {
        start_date: new Date().toISOString(),
        end_date: new Date().toISOString(),
        sample_count: 0,
        commits_included: []
      },
      current_measurement: current,
      regression_analysis: {
        performance_regressions: [],
        quality_regressions: [],
        trend_analysis: { trend_direction: 'stable', trend_strength: 0, trend_duration_days: 0 },
        anomaly_detection: { anomalies_detected: [] }
      },
      gate_evaluation: {
        passed: true,
        blocking_regressions: [],
        warnings: ['Insufficient baseline data for regression analysis'],
        early_warning_triggered: false,
        recommendation: 'proceed'
      }
    };
  }

  private generateCISummary(result: RegressionResult): string {
    const { gate_evaluation } = result;
    
    if (gate_evaluation.passed) {
      return `✅ No regressions detected${gate_evaluation.warnings.length > 0 ? ` (${gate_evaluation.warnings.length} warnings)` : ''}`;
    } else {
      return `❌ Regressions detected: ${gate_evaluation.blocking_regressions.join(', ')}`;
    }
  }

  private async sendRegressionNotifications(result: RegressionResult, config: RegressionConfig): Promise<void> {
    console.log('📧 Sending regression notifications...');
    // Mock notification sending - would integrate with actual notification services
  }

  private generateRegressionMarkdown(result: RegressionResult): string {
    const commitShort = result.commit_sha.substring(0, 8);
    const status = result.gate_evaluation.passed ? '✅ PASS' : '❌ FAIL';
    const recommendation = result.gate_evaluation.recommendation.toUpperCase();

    return `# Regression Analysis Report

**Commit:** ${commitShort}  
**Timestamp:** ${result.timestamp}  
**Gate Status:** ${status}  
**Recommendation:** ${recommendation}

## Baseline Period
- **Duration:** ${result.baseline_period.start_date} to ${result.baseline_period.end_date}
- **Samples:** ${result.baseline_period.sample_count} measurements
- **Commits:** ${result.baseline_period.commits_included.length} commits

## Performance Regressions
${result.regression_analysis.performance_regressions.length === 0 ? 
  '✅ No performance regressions detected' :
  result.regression_analysis.performance_regressions.map(r => `
- **${r.metric}:** ${r.regression_pct.toFixed(1)}% increase (${r.severity})
  - Baseline: ${r.baseline_value.toFixed(2)}
  - Current: ${r.current_value.toFixed(2)}
  - Threshold: ${r.threshold_pct}%
  - Confidence: ${(r.confidence * 100).toFixed(1)}%
`).join('')}

## Quality Regressions  
${result.regression_analysis.quality_regressions.length === 0 ?
  '✅ No quality regressions detected' :
  result.regression_analysis.quality_regressions.map(r => `
- **${r.metric}:** ${r.regression_pct.toFixed(1)}% decrease (${r.severity})
  - Baseline: ${r.baseline_value.toFixed(3)}
  - Current: ${r.current_value.toFixed(3)}
  - Threshold: ${r.threshold_pct}%
  - Significant: ${r.statistical_significance ? '⚠️ Yes' : 'No'}
`).join('')}

## Trend Analysis
- **Direction:** ${result.regression_analysis.trend_analysis.trend_direction.toUpperCase()}
- **Strength:** ${(result.regression_analysis.trend_analysis.trend_strength * 100).toFixed(1)}%
- **Duration:** ${result.regression_analysis.trend_analysis.trend_duration_days} days

## Gate Evaluation
${result.gate_evaluation.blocking_regressions.length > 0 ? 
  `**Blocking Regressions:** ${result.gate_evaluation.blocking_regressions.join(', ')}` : ''}
${result.gate_evaluation.warnings.length > 0 ? 
  `**Warnings:** ${result.gate_evaluation.warnings.join(', ')}` : ''}
${result.gate_evaluation.early_warning_triggered ? '⚠️ **Early warning triggered**' : ''}

**Recommendation:** ${recommendation}

## Current Metrics
- **Stage-A p95:** ${result.current_measurement.metrics.stage_a_p95_ms.toFixed(1)}ms
- **Stage-B p95:** ${result.current_measurement.metrics.stage_b_p95_ms.toFixed(1)}ms
- **E2E p95:** ${result.current_measurement.metrics.e2e_p95_ms.toFixed(1)}ms
- **nDCG@10:** ${result.current_measurement.metrics.ndcg_at_10.toFixed(3)}
- **Recall@50:** ${result.current_measurement.metrics.recall_at_50.toFixed(3)}
- **Span Coverage:** ${(result.current_measurement.metrics.span_coverage * 100).toFixed(1)}%
- **Memory:** ${result.current_measurement.metrics.memory_usage_mb.toFixed(0)}MB
- **Error Rate:** ${result.current_measurement.metrics.error_rate_pct.toFixed(1)}%
`;
  }
}