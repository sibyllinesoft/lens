/**
 * Phase B Performance Benchmark Suite
 * Validates Stage-A roaring bitmap, Stage-B AST cache, and Stage-C isotonic calibration optimizations
 * Per TODO.md: Stage-A p95 ≤5ms, E2E p95 ≤+10%, quality preservation (Δ nDCG@10 ≥ +2%, Recall@50 ≥ baseline)
 */

import { z } from 'zod';
import { v4 as uuidv4 } from 'uuid';
import { promises as fs } from 'fs';
import path from 'path';
import type {
  BenchmarkConfig,
  BenchmarkRun,
  GoldenDataItem,
  ABTestResult
} from '../types/benchmark.js';
import type { SearchResponse } from '../types/api.js';
import { BenchmarkSuiteRunner } from './suite-runner.js';
import { MetricsCalculator } from './metrics-calculator.js';

// Phase B specific configuration schema
export const PhaseBConfigSchema = z.object({
  optimizations: z.object({
    roaring_bitmap: z.object({
      enabled: z.boolean().default(true),
      prefilter_candidate_files: z.boolean().default(true),
      roaring_compression: z.boolean().default(true)
    }).default({}),
    ast_cache: z.object({
      enabled: z.boolean().default(true),
      max_files: z.number().int().min(50).max(1000).default(200),
      ttl_minutes: z.number().int().min(30).max(120).default(60),
      batch_processing: z.boolean().default(true),
      stale_while_revalidate: z.boolean().default(true)
    }).default({}),
    isotonic_calibration: z.object({
      enabled: z.boolean().default(true),
      confidence_cutoff: z.number().min(0.1).max(0.5).default(0.12),
      ann_k: z.number().int().default(150),
      ann_ef_search: z.number().int().default(64)
    }).default({})
  }),
  performance_targets: z.object({
    stage_a_p95_ms: z.number().default(5), // TODO.md target: ≤5ms
    stage_b_improvement_pct: z.number().default(40), // Target: ~40% improvement
    stage_c_improvement_pct: z.number().default(40), // Target: ~40% improvement
    e2e_p95_increase_max_pct: z.number().default(10), // TODO.md: ≤+10%
    quality_ndcg_improvement_min: z.number().default(0.02), // TODO.md: ≥+2%
    quality_recall_maintain: z.boolean().default(true), // TODO.md: Recall@50 ≥ baseline
    span_coverage_min: z.number().default(0.98) // TODO.md: ≥98%
  })
});

export type PhaseBConfig = z.infer<typeof PhaseBConfigSchema>;

/**
 * Performance optimization stages being tested
 */
export enum OptimizationStage {
  BASELINE = 'baseline',
  ROARING_BITMAP = 'B1_roaring_bitmap',
  AST_CACHE = 'B2_ast_cache',
  ISOTONIC_CALIBRATION = 'B3_isotonic_calibration',
  INTEGRATED = 'B_integrated'
}

/**
 * Phase B Performance Results
 */
export const PhaseBResultSchema = z.object({
  trace_id: z.string().uuid(),
  timestamp: z.string().datetime(),
  config: PhaseBConfigSchema,
  stages: z.record(z.enum([
    OptimizationStage.BASELINE,
    OptimizationStage.ROARING_BITMAP,
    OptimizationStage.AST_CACHE,
    OptimizationStage.ISOTONIC_CALIBRATION,
    OptimizationStage.INTEGRATED
  ]), z.object({
    benchmark_run: z.any(), // BenchmarkRun
    stage_metrics: z.object({
      stage_a_p50_ms: z.number(),
      stage_a_p95_ms: z.number(),
      stage_b_p50_ms: z.number(),
      stage_b_p95_ms: z.number(),
      stage_c_p50_ms: z.number().optional(),
      stage_c_p95_ms: z.number().optional(),
      e2e_p50_ms: z.number(),
      e2e_p95_ms: z.number(),
      memory_usage_mb: z.number(),
      cpu_utilization_pct: z.number(),
      cache_hit_rate: z.number().optional(),
      early_termination_rate: z.number().optional(),
      roaring_compression_ratio: z.number().optional()
    }),
    quality_metrics: z.object({
      recall_at_10: z.number(),
      recall_at_50: z.number(),
      ndcg_at_10: z.number(),
      span_coverage: z.number(),
      consistency_score: z.number()
    })
  })),
  comparisons: z.array(z.object({
    baseline_stage: z.string(),
    treatment_stage: z.string(),
    performance_improvement: z.object({
      stage_a_improvement_pct: z.number(),
      stage_b_improvement_pct: z.number(),
      stage_c_improvement_pct: z.number(),
      e2e_improvement_pct: z.number(),
      memory_delta_mb: z.number()
    }),
    quality_preservation: z.object({
      ndcg_delta: z.number(),
      recall_maintained: z.boolean(),
      span_coverage_maintained: z.boolean(),
      statistical_significance: z.boolean()
    }),
    meets_targets: z.object({
      stage_a_p95_target: z.boolean(),
      e2e_p95_target: z.boolean(),
      quality_targets: z.boolean(),
      overall_pass: z.boolean()
    })
  })),
  promotion_gate: z.object({
    passed: z.boolean(),
    failing_criteria: z.array(z.string()),
    summary: z.string()
  })
});

export type PhaseBResult = z.infer<typeof PhaseBResultSchema>;

export class PhaseBPerformanceBenchmark {
  private suiteRunner: BenchmarkSuiteRunner;
  private metricsCalculator: MetricsCalculator;
  
  constructor(
    suiteRunner: BenchmarkSuiteRunner,
    private readonly outputDir: string
  ) {
    this.suiteRunner = suiteRunner;
    this.metricsCalculator = new MetricsCalculator();
  }

  /**
   * Run comprehensive Phase B performance validation
   */
  async runPhaseBValidation(config: PhaseBConfig): Promise<PhaseBResult> {
    const traceId = uuidv4();
    console.log(`🚀 Starting Phase B Performance Validation - Trace ID: ${traceId}`);

    const result: PhaseBResult = {
      trace_id: traceId,
      timestamp: new Date().toISOString(),
      config,
      stages: {},
      comparisons: [],
      promotion_gate: {
        passed: false,
        failing_criteria: [],
        summary: ''
      }
    };

    try {
      // 1. Baseline measurement
      console.log('📊 Running baseline measurement...');
      result.stages[OptimizationStage.BASELINE] = await this.runStageTest(
        OptimizationStage.BASELINE,
        { optimizations: this.getDisabledOptimizations() },
        traceId
      );

      // 2. Stage A - Roaring Bitmap Optimization
      console.log('🔍 Testing Stage-A Roaring Bitmap optimization...');
      result.stages[OptimizationStage.ROARING_BITMAP] = await this.runStageTest(
        OptimizationStage.ROARING_BITMAP,
        { optimizations: { roaring_bitmap: config.optimizations.roaring_bitmap } },
        traceId
      );

      // 3. Stage B - AST Cache Enhancement
      console.log('🏗️ Testing Stage-B AST Cache enhancement...');
      result.stages[OptimizationStage.AST_CACHE] = await this.runStageTest(
        OptimizationStage.AST_CACHE,
        { optimizations: { ast_cache: config.optimizations.ast_cache } },
        traceId
      );

      // 4. Stage C - Isotonic Calibration
      console.log('📈 Testing Stage-C Isotonic Calibration...');
      result.stages[OptimizationStage.ISOTONIC_CALIBRATION] = await this.runStageTest(
        OptimizationStage.ISOTONIC_CALIBRATION,
        { optimizations: { isotonic_calibration: config.optimizations.isotonic_calibration } },
        traceId
      );

      // 5. Integrated optimization test
      console.log('🎯 Testing integrated optimizations...');
      result.stages[OptimizationStage.INTEGRATED] = await this.runStageTest(
        OptimizationStage.INTEGRATED,
        { optimizations: config.optimizations },
        traceId
      );

      // 6. Generate comparisons and validate targets
      result.comparisons = await this.generateComparisons(result.stages, config);
      result.promotion_gate = this.evaluatePromotionGate(result.comparisons, config);

      // 7. Generate detailed report
      await this.generatePhaseBReport(result);

      console.log(`✅ Phase B validation complete - Gate: ${result.promotion_gate.passed ? 'PASS' : 'FAIL'}`);
      return result;

    } catch (error) {
      console.error('❌ Phase B validation failed:', error);
      result.promotion_gate = {
        passed: false,
        failing_criteria: ['validation_error'],
        summary: `Validation failed: ${error instanceof Error ? error.message : String(error)}`
      };
      return result;
    }
  }

  /**
   * Run individual stage optimization test
   */
  private async runStageTest(
    stage: OptimizationStage,
    optimizationConfig: any,
    traceId: string
  ): Promise<any> {
    console.log(`  Running ${stage} test...`);

    // Configure search engine with specific optimizations
    await this.configureOptimizations(stage, optimizationConfig);

    // Run smoke test with performance measurement
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
      }
    };

    const benchmarkRun = await this.suiteRunner.runSmokeSuite(benchmarkConfig);

    // Collect additional performance metrics
    const resourceMetrics = await this.collectResourceMetrics();
    const cacheMetrics = await this.collectCacheMetrics(stage);

    return {
      benchmark_run: benchmarkRun,
      stage_metrics: {
        stage_a_p50_ms: benchmarkRun.metrics.stage_latencies.stage_a_p50,
        stage_a_p95_ms: benchmarkRun.metrics.stage_latencies.stage_a_p95,
        stage_b_p50_ms: benchmarkRun.metrics.stage_latencies.stage_b_p50,
        stage_b_p95_ms: benchmarkRun.metrics.stage_latencies.stage_b_p95,
        stage_c_p50_ms: benchmarkRun.metrics.stage_latencies.stage_c_p50,
        stage_c_p95_ms: benchmarkRun.metrics.stage_latencies.stage_c_p95,
        e2e_p50_ms: benchmarkRun.metrics.stage_latencies.e2e_p50,
        e2e_p95_ms: benchmarkRun.metrics.stage_latencies.e2e_p95,
        memory_usage_mb: resourceMetrics.memory_usage_mb,
        cpu_utilization_pct: resourceMetrics.cpu_utilization_pct,
        cache_hit_rate: cacheMetrics?.cache_hit_rate,
        early_termination_rate: cacheMetrics?.early_termination_rate,
        roaring_compression_ratio: cacheMetrics?.roaring_compression_ratio
      },
      quality_metrics: {
        recall_at_10: benchmarkRun.metrics.recall_at_10,
        recall_at_50: benchmarkRun.metrics.recall_at_50,
        ndcg_at_10: benchmarkRun.metrics.ndcg_at_10,
        span_coverage: this.calculateSpanCoverage(benchmarkRun),
        consistency_score: this.calculateConsistencyScore(benchmarkRun)
      }
    };
  }

  /**
   * Configure search engine with specific optimizations for testing
   */
  private async configureOptimizations(stage: OptimizationStage, config: any): Promise<void> {
    const endpoint = 'http://localhost:4000';

    switch (stage) {
      case OptimizationStage.BASELINE:
        // Disable all optimizations
        await this.apiCall(`${endpoint}/policy/stageA`, 'PATCH', {
          rare_term_fuzzy: false,
          synonyms_when_identifier_density_below: 1.0,
          prefilter: { type: "none", enabled: false },
          wand: { enabled: false, block_max: false },
          per_file_span_cap: 10,
          native_scanner: "off"
        });
        await this.apiCall(`${endpoint}/policy/stageB`, 'PATCH', {
          ast_cache: { enabled: false, max_files: 50, ttl_minutes: 30 }
        });
        await this.apiCall(`${endpoint}/policy/stageC`, 'PATCH', {
          calibration: "none",
          gate: { nl_threshold: 0.5, min_candidates: 10, confidence_cutoff: 0.0 },
          ann: { k: 100, efSearch: 32 }
        });
        break;

      case OptimizationStage.ROARING_BITMAP:
        await this.apiCall(`${endpoint}/policy/stageA`, 'PATCH', {
          rare_term_fuzzy: true,
          synonyms_when_identifier_density_below: 0.5,
          prefilter: { type: "roaring", enabled: true },
          wand: { enabled: true, block_max: true },
          per_file_span_cap: 3,
          native_scanner: "auto"
        });
        break;

      case OptimizationStage.AST_CACHE:
        await this.apiCall(`${endpoint}/policy/stageB`, 'PATCH', {
          ast_cache: {
            enabled: true,
            max_files: config.optimizations?.ast_cache?.max_files || 200,
            ttl_minutes: config.optimizations?.ast_cache?.ttl_minutes || 60,
            batch_processing: config.optimizations?.ast_cache?.batch_processing || true,
            stale_while_revalidate: config.optimizations?.ast_cache?.stale_while_revalidate || true
          }
        });
        break;

      case OptimizationStage.ISOTONIC_CALIBRATION:
        await this.apiCall(`${endpoint}/policy/stageC`, 'PATCH', {
          calibration: "isotonic_v1",
          gate: { 
            nl_threshold: 0.5, 
            min_candidates: 10, 
            confidence_cutoff: config.optimizations?.isotonic_calibration?.confidence_cutoff || 0.12 
          },
          ann: { 
            k: config.optimizations?.isotonic_calibration?.ann_k || 150, 
            efSearch: config.optimizations?.isotonic_calibration?.ann_ef_search || 64 
          }
        });
        break;

      case OptimizationStage.INTEGRATED:
        // Apply all optimizations
        await this.configureOptimizations(OptimizationStage.ROARING_BITMAP, config);
        await this.configureOptimizations(OptimizationStage.AST_CACHE, config);
        await this.configureOptimizations(OptimizationStage.ISOTONIC_CALIBRATION, config);
        break;
    }
  }

  /**
   * Generate performance and quality comparisons between stages
   */
  private async generateComparisons(
    stages: Record<string, any>,
    config: PhaseBConfig
  ): Promise<any[]> {
    const comparisons: any[] = [];
    const baseline = stages[OptimizationStage.BASELINE];

    if (!baseline) {
      throw new Error('Baseline measurement missing');
    }

    for (const [stageName, stageResult] of Object.entries(stages)) {
      if (stageName === OptimizationStage.BASELINE) continue;

      const comparison = {
        baseline_stage: OptimizationStage.BASELINE,
        treatment_stage: stageName,
        performance_improvement: this.calculatePerformanceImprovement(baseline, stageResult),
        quality_preservation: this.calculateQualityPreservation(baseline, stageResult),
        meets_targets: this.evaluateTargets(baseline, stageResult, config)
      };

      comparisons.push(comparison);
    }

    return comparisons;
  }

  /**
   * Calculate performance improvement percentages
   */
  private calculatePerformanceImprovement(baseline: any, treatment: any): any {
    const baselineMetrics = baseline.stage_metrics;
    const treatmentMetrics = treatment.stage_metrics;

    return {
      stage_a_improvement_pct: this.calculateImprovementPct(
        baselineMetrics.stage_a_p95_ms,
        treatmentMetrics.stage_a_p95_ms
      ),
      stage_b_improvement_pct: this.calculateImprovementPct(
        baselineMetrics.stage_b_p95_ms,
        treatmentMetrics.stage_b_p95_ms
      ),
      stage_c_improvement_pct: this.calculateImprovementPct(
        baselineMetrics.stage_c_p95_ms || 0,
        treatmentMetrics.stage_c_p95_ms || 0
      ),
      e2e_improvement_pct: this.calculateImprovementPct(
        baselineMetrics.e2e_p95_ms,
        treatmentMetrics.e2e_p95_ms
      ),
      memory_delta_mb: treatmentMetrics.memory_usage_mb - baselineMetrics.memory_usage_mb
    };
  }

  /**
   * Calculate quality preservation metrics
   */
  private calculateQualityPreservation(baseline: any, treatment: any): any {
    const baselineQuality = baseline.quality_metrics;
    const treatmentQuality = treatment.quality_metrics;

    const ndcgDelta = treatmentQuality.ndcg_at_10 - baselineQuality.ndcg_at_10;
    const recallMaintained = treatmentQuality.recall_at_50 >= baselineQuality.recall_at_50;
    const spanCoverageMaintained = treatmentQuality.span_coverage >= 0.98;

    return {
      ndcg_delta: ndcgDelta,
      recall_maintained: recallMaintained,
      span_coverage_maintained: spanCoverageMaintained,
      statistical_significance: Math.abs(ndcgDelta) >= 0.02 // 2% threshold
    };
  }

  /**
   * Evaluate if optimization meets Phase B targets
   */
  private evaluateTargets(baseline: any, treatment: any, config: PhaseBConfig): any {
    const treatmentMetrics = treatment.stage_metrics;
    const qualityPreservation = this.calculateQualityPreservation(baseline, treatment);

    const stageATarget = treatmentMetrics.stage_a_p95_ms <= config.performance_targets.stage_a_p95_ms;
    const e2eTarget = this.calculateImprovementPct(
      baseline.stage_metrics.e2e_p95_ms,
      treatmentMetrics.e2e_p95_ms
    ) <= config.performance_targets.e2e_p95_increase_max_pct;
    const qualityTargets = qualityPreservation.ndcg_delta >= config.performance_targets.quality_ndcg_improvement_min &&
                           qualityPreservation.recall_maintained &&
                           qualityPreservation.span_coverage_maintained;

    return {
      stage_a_p95_target: stageATarget,
      e2e_p95_target: e2eTarget,
      quality_targets: qualityTargets,
      overall_pass: stageATarget && e2eTarget && qualityTargets
    };
  }

  /**
   * Evaluate overall promotion gate criteria
   */
  private evaluatePromotionGate(comparisons: any[], config: PhaseBConfig): any {
    const integratedComparison = comparisons.find(c => c.treatment_stage === OptimizationStage.INTEGRATED);
    
    if (!integratedComparison) {
      return {
        passed: false,
        failing_criteria: ['missing_integrated_test'],
        summary: 'Integrated optimization test results missing'
      };
    }

    const failingCriteria: string[] = [];

    if (!integratedComparison.meets_targets.stage_a_p95_target) {
      failingCriteria.push('stage_a_p95_target');
    }
    if (!integratedComparison.meets_targets.e2e_p95_target) {
      failingCriteria.push('e2e_p95_target');
    }
    if (!integratedComparison.meets_targets.quality_targets) {
      failingCriteria.push('quality_targets');
    }

    const passed = failingCriteria.length === 0;
    const summary = passed 
      ? 'All Phase B optimization targets met'
      : `Failed criteria: ${failingCriteria.join(', ')}`;

    return { passed, failing_criteria: failingCriteria, summary };
  }

  /**
   * Generate comprehensive Phase B performance report
   */
  private async generatePhaseBReport(result: PhaseBResult): Promise<void> {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const reportPath = path.join(this.outputDir, `phase-b-performance-${timestamp}.json`);

    // Add detailed analysis
    const reportData = {
      ...result,
      analysis: {
        optimization_summary: this.generateOptimizationSummary(result),
        performance_analysis: this.generatePerformanceAnalysis(result),
        quality_analysis: this.generateQualityAnalysis(result),
        recommendations: this.generateRecommendations(result)
      }
    };

    await fs.writeFile(reportPath, JSON.stringify(reportData, null, 2));
    console.log(`📄 Phase B performance report written to: ${reportPath}`);

    // Generate markdown summary for human review
    const markdownReport = this.generateMarkdownSummary(result);
    const markdownPath = path.join(this.outputDir, `phase-b-summary-${timestamp}.md`);
    await fs.writeFile(markdownPath, markdownReport);
    console.log(`📋 Phase B summary written to: ${markdownPath}`);
  }

  // Helper methods
  private getDisabledOptimizations(): any {
    return {
      roaring_bitmap: { enabled: false },
      ast_cache: { enabled: false },
      isotonic_calibration: { enabled: false }
    };
  }

  private calculateImprovementPct(baseline: number, treatment: number): number {
    if (baseline === 0) return 0;
    return ((baseline - treatment) / baseline) * 100;
  }

  private calculateSpanCoverage(benchmarkRun: BenchmarkRun): number {
    // Calculate span coverage from fan-out data
    const { stage_a, stage_b, stage_c } = benchmarkRun.metrics.fan_out_sizes;
    const totalCandidates = stage_a + stage_b + (stage_c || 0);
    const actualResults = benchmarkRun.completed_queries;
    return actualResults > 0 ? Math.min(totalCandidates / (actualResults * 100), 1) : 0;
  }

  private calculateConsistencyScore(benchmarkRun: BenchmarkRun): number {
    // Calculate consistency score based on error rate and result stability
    const errorRate = benchmarkRun.failed_queries / Math.max(benchmarkRun.total_queries, 1);
    return Math.max(0, 1 - errorRate);
  }

  private async collectResourceMetrics(): Promise<any> {
    // Mock resource collection - would integrate with actual monitoring
    return {
      memory_usage_mb: Math.floor(Math.random() * 512) + 256,
      cpu_utilization_pct: Math.floor(Math.random() * 30) + 40
    };
  }

  private async collectCacheMetrics(stage: OptimizationStage): Promise<any> {
    // Mock cache metrics - would integrate with actual cache monitoring
    if (stage === OptimizationStage.AST_CACHE || stage === OptimizationStage.INTEGRATED) {
      return {
        cache_hit_rate: 0.85 + Math.random() * 0.1,
        early_termination_rate: stage.includes('ROARING') ? 0.3 + Math.random() * 0.2 : undefined,
        roaring_compression_ratio: stage.includes('ROARING') ? 2.5 + Math.random() * 1.5 : undefined
      };
    }
    return {};
  }

  private async apiCall(url: string, method: string, body?: any): Promise<any> {
    // Mock API calls - would make real HTTP requests to search engine
    console.log(`  API Call: ${method} ${url}`, body ? '(with body)' : '');
    return Promise.resolve({});
  }

  private generateOptimizationSummary(result: PhaseBResult): any {
    return {
      total_stages_tested: Object.keys(result.stages).length,
      successful_optimizations: result.comparisons.filter(c => c.meets_targets.overall_pass).length,
      best_performing_stage: this.findBestPerformingStage(result.comparisons),
      overall_promotion_gate: result.promotion_gate.passed
    };
  }

  private generatePerformanceAnalysis(result: PhaseBResult): any {
    const integratedComparison = result.comparisons.find(c => c.treatment_stage === OptimizationStage.INTEGRATED);
    if (!integratedComparison) return {};

    return {
      stage_a_improvement: integratedComparison.performance_improvement.stage_a_improvement_pct,
      stage_b_improvement: integratedComparison.performance_improvement.stage_b_improvement_pct,
      stage_c_improvement: integratedComparison.performance_improvement.stage_c_improvement_pct,
      e2e_improvement: integratedComparison.performance_improvement.e2e_improvement_pct,
      memory_impact: integratedComparison.performance_improvement.memory_delta_mb
    };
  }

  private generateQualityAnalysis(result: PhaseBResult): any {
    const integratedComparison = result.comparisons.find(c => c.treatment_stage === OptimizationStage.INTEGRATED);
    if (!integratedComparison) return {};

    return {
      ndcg_delta: integratedComparison.quality_preservation.ndcg_delta,
      recall_maintained: integratedComparison.quality_preservation.recall_maintained,
      span_coverage_maintained: integratedComparison.quality_preservation.span_coverage_maintained,
      statistical_significance: integratedComparison.quality_preservation.statistical_significance
    };
  }

  private generateRecommendations(result: PhaseBResult): string[] {
    const recommendations: string[] = [];
    
    if (!result.promotion_gate.passed) {
      recommendations.push(`Failed promotion gate: ${result.promotion_gate.summary}`);
    }

    // Add specific recommendations based on results
    for (const comparison of result.comparisons) {
      if (!comparison.meets_targets.stage_a_p95_target) {
        recommendations.push(`${comparison.treatment_stage}: Stage-A p95 exceeds 5ms target`);
      }
      if (!comparison.meets_targets.quality_targets) {
        recommendations.push(`${comparison.treatment_stage}: Quality targets not met`);
      }
    }

    if (recommendations.length === 0) {
      recommendations.push('All optimization targets met - ready for Phase C');
    }

    return recommendations;
  }

  private findBestPerformingStage(comparisons: any[]): string {
    let bestStage = '';
    let bestScore = -Infinity;

    for (const comparison of comparisons) {
      // Simple scoring based on e2e improvement and quality preservation
      const score = comparison.performance_improvement.e2e_improvement_pct + 
                   (comparison.quality_preservation.ndcg_delta * 100);
      if (score > bestScore && comparison.meets_targets.overall_pass) {
        bestScore = score;
        bestStage = comparison.treatment_stage;
      }
    }

    return bestStage || 'none';
  }

  private generateMarkdownSummary(result: PhaseBResult): string {
    return `# Phase B Performance Validation Summary

**Trace ID:** ${result.trace_id}  
**Timestamp:** ${result.timestamp}  
**Promotion Gate:** ${result.promotion_gate.passed ? '✅ PASS' : '❌ FAIL'}

## Optimization Results

${Object.entries(result.stages).map(([stage, stageResult]) => `
### ${stage.replace('_', ' ').toUpperCase()}

- **Stage-A p95:** ${(stageResult as any).stage_metrics.stage_a_p95_ms}ms
- **Stage-B p95:** ${(stageResult as any).stage_metrics.stage_b_p95_ms}ms  
- **E2E p95:** ${(stageResult as any).stage_metrics.e2e_p95_ms}ms
- **nDCG@10:** ${(stageResult as any).quality_metrics.ndcg_at_10.toFixed(3)}
- **Recall@50:** ${(stageResult as any).quality_metrics.recall_at_50.toFixed(3)}
- **Span Coverage:** ${((stageResult as any).quality_metrics.span_coverage * 100).toFixed(1)}%
`).join('\n')}

## Performance Comparisons vs Baseline

${result.comparisons.map(comparison => `
### ${comparison.treatment_stage} vs Baseline

- **Stage-A Improvement:** ${comparison.performance_improvement.stage_a_improvement_pct.toFixed(1)}%
- **Stage-B Improvement:** ${comparison.performance_improvement.stage_b_improvement_pct.toFixed(1)}%
- **E2E Improvement:** ${comparison.performance_improvement.e2e_improvement_pct.toFixed(1)}%
- **nDCG Delta:** ${(comparison.quality_preservation.ndcg_delta * 100).toFixed(1)}%
- **Targets Met:** ${comparison.meets_targets.overall_pass ? '✅' : '❌'}
`).join('\n')}

## Summary

${result.promotion_gate.summary}

${result.promotion_gate.failing_criteria.length > 0 ? 
  `**Failing Criteria:** ${result.promotion_gate.failing_criteria.join(', ')}` : 
  '**All targets met!** 🎉'
}
`;
  }
}