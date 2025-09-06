/**
 * Comprehensive Quality Gates for Evergreen Optimization Systems
 * 
 * Monitors and validates all four evergreen systems:
 * 1. Program-Slice Recall: Recall@50 +0.7-1.2pp, p95 ≤ +0.8ms, span=100%, vendor veto honored
 * 2. Build/Test-Aware Priors: ΔnDCG@10 ≥ +0.5pp on failure-adjacent, SLA-Recall@50 ≥ 0, Core@10 drift ≤±5pp
 * 3. Speculative Multi-Plan: fleet p99 -8-12% at flat recall, abort if p95 > +0.6ms
 * 4. Cache Admission: admission_hit_rate - LRU ≥ +3-5pp, cache CPU ≤ +3%, p95 -0.3 to -0.8ms
 */

import type { SearchContext, Candidate } from '../types/core.js';
import type { SearchHit } from './span_resolver/types.js';
import { LensTracer } from '../telemetry/tracer.js';

// System-specific imports
import type { SliceResult } from './program-slice-recall.js';
import type { CacheStats } from './cache-admission-learner.js';

export interface QualityGateResult {
  system: string;
  gate_name: string;
  status: 'pass' | 'fail' | 'warning';
  measured_value: number;
  threshold_min?: number;
  threshold_max?: number;
  baseline_value?: number;
  message: string;
  timestamp: Date;
}

export interface SystemMetrics {
  // Program Slice Recall metrics
  slice_recall_at_50?: number;
  slice_p95_latency_ms?: number;
  slice_span_coverage?: number;
  slice_vendor_veto_honored?: number;
  slice_rollout_percentage?: number;

  // Build/Test Priors metrics
  build_ndcg_at_10_delta?: number;
  build_sla_recall_at_50?: number;
  build_core_at_10_drift?: number;
  build_failure_adjacent_queries?: number;
  build_why_mix_kl_divergence?: number;

  // Speculative Multi-Plan metrics
  plan_fleet_p99_improvement?: number;
  plan_flat_recall_maintained?: boolean;
  plan_p95_latency_ms?: number;
  plan_budget_utilization?: number;

  // Cache Admission metrics
  cache_admission_hit_rate?: number;
  cache_lru_baseline_hit_rate?: number;
  cache_cpu_overhead_percent?: number;
  cache_p95_latency_improvement?: number;
  cache_span_drift?: number;
}

export interface QualityGateConfig {
  // Program Slice Recall gates
  slice_min_recall_improvement: number; // +0.7pp
  slice_max_recall_improvement: number; // +1.2pp
  slice_max_p95_increase: number; // +0.8ms
  slice_min_span_coverage: number; // 100%
  slice_min_vendor_veto_rate: number; // 100%

  // Build/Test Priors gates
  build_min_ndcg_improvement: number; // +0.5pp
  build_min_sla_recall: number; // ≥0
  build_max_core_drift: number; // ±5pp
  build_min_failure_adjacent_ratio: number; // For targeting validation

  // Speculative Multi-Plan gates
  plan_min_p99_improvement: number; // -8%
  plan_max_p99_improvement: number; // -12%
  plan_max_p95_increase: number; // +0.6ms
  plan_max_budget_utilization: number; // ≤10%

  // Cache Admission gates
  cache_min_admission_improvement: number; // +3pp
  cache_max_admission_improvement: number; // +5pp
  cache_max_cpu_overhead: number; // ≤3%
  cache_min_p95_improvement: number; // -0.3ms
  cache_max_p95_improvement: number; // -0.8ms
  cache_max_span_drift: number; // 0%
}

export interface QualityMonitoringReport {
  report_id: string;
  timestamp: Date;
  reporting_period_hours: number;
  systems_evaluated: string[];
  gates_passed: number;
  gates_failed: number;
  gates_warning: number;
  overall_status: 'healthy' | 'degraded' | 'critical';
  gate_results: QualityGateResult[];
  system_metrics: SystemMetrics;
  recommendations: string[];
  next_evaluation_time: Date;
}

/**
 * Baseline metrics collector for comparison
 */
export class BaselineMetricsCollector {
  private baselineMetrics: Map<string, number> = new Map();
  private collectionWindow = 7 * 24 * 60 * 60 * 1000; // 7 days

  /**
   * Record baseline metrics before systems are enabled
   */
  recordBaseline(metricName: string, value: number): void {
    this.baselineMetrics.set(metricName, value);
  }

  /**
   * Get baseline value for comparison
   */
  getBaseline(metricName: string): number | undefined {
    return this.baselineMetrics.get(metricName);
  }

  /**
   * Calculate improvement delta from baseline
   */
  calculateDelta(metricName: string, currentValue: number): number {
    const baseline = this.getBaseline(metricName);
    if (baseline === undefined) return 0;
    return currentValue - baseline;
  }

  /**
   * Calculate percentage improvement
   */
  calculatePercentageImprovement(metricName: string, currentValue: number): number {
    const baseline = this.getBaseline(metricName);
    if (baseline === undefined || baseline === 0) return 0;
    return ((currentValue - baseline) / baseline) * 100;
  }
}

/**
 * Quality gate evaluator for each system
 */
export class QualityGateEvaluator {
  private config: QualityGateConfig;
  private baselineCollector: BaselineMetricsCollector;

  constructor(config: QualityGateConfig) {
    this.config = config;
    this.baselineCollector = new BaselineMetricsCollector();
  }

  /**
   * Evaluate Program Slice Recall quality gates
   */
  evaluateSliceRecallGates(metrics: SystemMetrics): QualityGateResult[] {
    const results: QualityGateResult[] = [];
    const timestamp = new Date();

    // Gate 1: Recall@50 improvement (+0.7-1.2pp)
    if (metrics.slice_recall_at_50 !== undefined) {
      const baseline = this.baselineCollector.getBaseline('recall_at_50') || 0;
      const improvement = (metrics.slice_recall_at_50 - baseline) * 100; // to percentage points
      
      results.push({
        system: 'program-slice-recall',
        gate_name: 'recall_at_50_improvement',
        status: improvement >= this.config.slice_min_recall_improvement && 
                improvement <= this.config.slice_max_recall_improvement ? 'pass' : 'fail',
        measured_value: improvement,
        threshold_min: this.config.slice_min_recall_improvement,
        threshold_max: this.config.slice_max_recall_improvement,
        baseline_value: baseline * 100,
        message: `Recall@50 improvement: ${improvement.toFixed(2)}pp (target: ${this.config.slice_min_recall_improvement}-${this.config.slice_max_recall_improvement}pp)`,
        timestamp,
      });
    }

    // Gate 2: P95 latency increase (≤+0.8ms)
    if (metrics.slice_p95_latency_ms !== undefined) {
      const baseline = this.baselineCollector.getBaseline('p95_latency_ms') || 0;
      const increase = metrics.slice_p95_latency_ms - baseline;
      
      results.push({
        system: 'program-slice-recall',
        gate_name: 'p95_latency_increase',
        status: increase <= this.config.slice_max_p95_increase ? 'pass' : 'fail',
        measured_value: increase,
        threshold_max: this.config.slice_max_p95_increase,
        baseline_value: baseline,
        message: `P95 latency increase: ${increase.toFixed(2)}ms (max: ${this.config.slice_max_p95_increase}ms)`,
        timestamp,
      });
    }

    // Gate 3: Span coverage (100%)
    if (metrics.slice_span_coverage !== undefined) {
      results.push({
        system: 'program-slice-recall',
        gate_name: 'span_coverage',
        status: metrics.slice_span_coverage >= this.config.slice_min_span_coverage ? 'pass' : 'fail',
        measured_value: metrics.slice_span_coverage * 100,
        threshold_min: this.config.slice_min_span_coverage * 100,
        message: `Span coverage: ${(metrics.slice_span_coverage * 100).toFixed(1)}% (min: ${this.config.slice_min_span_coverage * 100}%)`,
        timestamp,
      });
    }

    // Gate 4: Vendor veto honored (100%)
    if (metrics.slice_vendor_veto_honored !== undefined) {
      results.push({
        system: 'program-slice-recall',
        gate_name: 'vendor_veto_honored',
        status: metrics.slice_vendor_veto_honored >= this.config.slice_min_vendor_veto_rate ? 'pass' : 'fail',
        measured_value: metrics.slice_vendor_veto_honored * 100,
        threshold_min: this.config.slice_min_vendor_veto_rate * 100,
        message: `Vendor veto honored: ${(metrics.slice_vendor_veto_honored * 100).toFixed(1)}% (min: ${this.config.slice_min_vendor_veto_rate * 100}%)`,
        timestamp,
      });
    }

    return results;
  }

  /**
   * Evaluate Build/Test-Aware Priors quality gates
   */
  evaluateBuildPriorsGates(metrics: SystemMetrics): QualityGateResult[] {
    const results: QualityGateResult[] = [];
    const timestamp = new Date();

    // Gate 1: nDCG@10 improvement (≥+0.5pp on failure-adjacent queries)
    if (metrics.build_ndcg_at_10_delta !== undefined) {
      results.push({
        system: 'build-test-priors',
        gate_name: 'ndcg_at_10_improvement',
        status: metrics.build_ndcg_at_10_delta >= this.config.build_min_ndcg_improvement ? 'pass' : 'fail',
        measured_value: metrics.build_ndcg_at_10_delta,
        threshold_min: this.config.build_min_ndcg_improvement,
        message: `nDCG@10 delta on failure-adjacent queries: ${metrics.build_ndcg_at_10_delta.toFixed(3)}pp (min: ${this.config.build_min_ndcg_improvement}pp)`,
        timestamp,
      });
    }

    // Gate 2: SLA-Recall@50 (≥0)
    if (metrics.build_sla_recall_at_50 !== undefined) {
      results.push({
        system: 'build-test-priors',
        gate_name: 'sla_recall_at_50',
        status: metrics.build_sla_recall_at_50 >= this.config.build_min_sla_recall ? 'pass' : 'fail',
        measured_value: metrics.build_sla_recall_at_50,
        threshold_min: this.config.build_min_sla_recall,
        message: `SLA-Recall@50: ${metrics.build_sla_recall_at_50.toFixed(3)} (min: ${this.config.build_min_sla_recall})`,
        timestamp,
      });
    }

    // Gate 3: Core@10 drift (≤±5pp topic-normalized)
    if (metrics.build_core_at_10_drift !== undefined) {
      const driftAbs = Math.abs(metrics.build_core_at_10_drift);
      results.push({
        system: 'build-test-priors',
        gate_name: 'core_at_10_drift',
        status: driftAbs <= this.config.build_max_core_drift ? 'pass' : 'fail',
        measured_value: metrics.build_core_at_10_drift,
        threshold_max: this.config.build_max_core_drift,
        message: `Core@10 drift: ${metrics.build_core_at_10_drift.toFixed(2)}pp (max: ±${this.config.build_max_core_drift}pp)`,
        timestamp,
      });
    }

    // Gate 4: Why-mix KL divergence (monitoring only)
    if (metrics.build_why_mix_kl_divergence !== undefined) {
      results.push({
        system: 'build-test-priors',
        gate_name: 'why_mix_kl_divergence',
        status: metrics.build_why_mix_kl_divergence < 0.1 ? 'pass' : 'warning',
        measured_value: metrics.build_why_mix_kl_divergence,
        message: `Why-mix KL divergence: ${metrics.build_why_mix_kl_divergence.toFixed(4)} (monitoring for semantic crowding)`,
        timestamp,
      });
    }

    return results;
  }

  /**
   * Evaluate Speculative Multi-Plan quality gates
   */
  evaluateMultiPlanGates(metrics: SystemMetrics): QualityGateResult[] {
    const results: QualityGateResult[] = [];
    const timestamp = new Date();

    // Gate 1: Fleet P99 improvement (-8 to -12%)
    if (metrics.plan_fleet_p99_improvement !== undefined) {
      const improvement = metrics.plan_fleet_p99_improvement;
      results.push({
        system: 'speculative-multi-plan',
        gate_name: 'fleet_p99_improvement',
        status: improvement >= this.config.plan_min_p99_improvement && 
                improvement <= this.config.plan_max_p99_improvement ? 'pass' : 'fail',
        measured_value: improvement,
        threshold_min: this.config.plan_min_p99_improvement,
        threshold_max: this.config.plan_max_p99_improvement,
        message: `Fleet P99 improvement: ${improvement.toFixed(1)}% (target: ${this.config.plan_min_p99_improvement}% to ${this.config.plan_max_p99_improvement}%)`,
        timestamp,
      });
    }

    // Gate 2: P95 latency increase (abort if >+0.6ms)
    if (metrics.plan_p95_latency_ms !== undefined) {
      const baseline = this.baselineCollector.getBaseline('plan_p95_latency_ms') || 0;
      const increase = metrics.plan_p95_latency_ms - baseline;
      results.push({
        system: 'speculative-multi-plan',
        gate_name: 'p95_latency_increase',
        status: increase <= this.config.plan_max_p95_increase ? 'pass' : 'fail',
        measured_value: increase,
        threshold_max: this.config.plan_max_p95_increase,
        baseline_value: baseline,
        message: `P95 latency increase: ${increase.toFixed(2)}ms (max: ${this.config.plan_max_p95_increase}ms) - ${increase > this.config.plan_max_p95_increase ? 'ABORT REQUIRED' : 'OK'}`,
        timestamp,
      });
    }

    // Gate 3: Flat recall maintained
    if (metrics.plan_flat_recall_maintained !== undefined) {
      results.push({
        system: 'speculative-multi-plan',
        gate_name: 'flat_recall_maintained',
        status: metrics.plan_flat_recall_maintained ? 'pass' : 'fail',
        measured_value: metrics.plan_flat_recall_maintained ? 1 : 0,
        message: `Flat recall maintained: ${metrics.plan_flat_recall_maintained ? 'YES' : 'NO'}`,
        timestamp,
      });
    }

    // Gate 4: Budget utilization (≤10%)
    if (metrics.plan_budget_utilization !== undefined) {
      results.push({
        system: 'speculative-multi-plan',
        gate_name: 'budget_utilization',
        status: metrics.plan_budget_utilization <= this.config.plan_max_budget_utilization ? 'pass' : 'fail',
        measured_value: metrics.plan_budget_utilization,
        threshold_max: this.config.plan_max_budget_utilization,
        message: `Planner budget utilization: ${metrics.plan_budget_utilization.toFixed(1)}% (max: ${this.config.plan_max_budget_utilization}%)`,
        timestamp,
      });
    }

    return results;
  }

  /**
   * Evaluate Cache Admission quality gates
   */
  evaluateCacheAdmissionGates(metrics: SystemMetrics): QualityGateResult[] {
    const results: QualityGateResult[] = [];
    const timestamp = new Date();

    // Gate 1: Admission hit rate improvement (+3-5pp vs LRU)
    if (metrics.cache_admission_hit_rate !== undefined && metrics.cache_lru_baseline_hit_rate !== undefined) {
      const improvement = (metrics.cache_admission_hit_rate - metrics.cache_lru_baseline_hit_rate) * 100;
      results.push({
        system: 'cache-admission-learner',
        gate_name: 'admission_hit_rate_improvement',
        status: improvement >= this.config.cache_min_admission_improvement && 
                improvement <= this.config.cache_max_admission_improvement ? 'pass' : 'fail',
        measured_value: improvement,
        threshold_min: this.config.cache_min_admission_improvement,
        threshold_max: this.config.cache_max_admission_improvement,
        baseline_value: metrics.cache_lru_baseline_hit_rate * 100,
        message: `Admission hit rate improvement vs LRU: ${improvement.toFixed(2)}pp (target: ${this.config.cache_min_admission_improvement}-${this.config.cache_max_admission_improvement}pp)`,
        timestamp,
      });
    }

    // Gate 2: CPU overhead (≤3%)
    if (metrics.cache_cpu_overhead_percent !== undefined) {
      results.push({
        system: 'cache-admission-learner',
        gate_name: 'cpu_overhead',
        status: metrics.cache_cpu_overhead_percent <= this.config.cache_max_cpu_overhead ? 'pass' : 'fail',
        measured_value: metrics.cache_cpu_overhead_percent,
        threshold_max: this.config.cache_max_cpu_overhead,
        message: `Cache CPU overhead: ${metrics.cache_cpu_overhead_percent.toFixed(2)}% (max: ${this.config.cache_max_cpu_overhead}%)`,
        timestamp,
      });
    }

    // Gate 3: P95 latency improvement (-0.3 to -0.8ms)
    if (metrics.cache_p95_latency_improvement !== undefined) {
      const improvement = metrics.cache_p95_latency_improvement;
      results.push({
        system: 'cache-admission-learner',
        gate_name: 'p95_latency_improvement',
        status: improvement >= this.config.cache_min_p95_improvement && 
                improvement <= this.config.cache_max_p95_improvement ? 'pass' : 'fail',
        measured_value: improvement,
        threshold_min: this.config.cache_min_p95_improvement,
        threshold_max: this.config.cache_max_p95_improvement,
        message: `P95 latency improvement: ${improvement.toFixed(2)}ms (target: ${this.config.cache_min_p95_improvement}ms to ${this.config.cache_max_p95_improvement}ms)`,
        timestamp,
      });
    }

    // Gate 4: Span drift (0% - no span changes allowed)
    if (metrics.cache_span_drift !== undefined) {
      results.push({
        system: 'cache-admission-learner',
        gate_name: 'span_drift',
        status: Math.abs(metrics.cache_span_drift) <= this.config.cache_max_span_drift ? 'pass' : 'fail',
        measured_value: Math.abs(metrics.cache_span_drift) * 100,
        threshold_max: this.config.cache_max_span_drift * 100,
        message: `Span drift: ${(metrics.cache_span_drift * 100).toFixed(3)}% (max: ${this.config.cache_max_span_drift * 100}%)`,
        timestamp,
      });
    }

    return results;
  }

  /**
   * Set baseline metrics collector
   */
  setBaselineCollector(collector: BaselineMetricsCollector): void {
    this.baselineCollector = collector;
  }
}

/**
 * Comprehensive quality monitoring system
 */
export class EvergreenQualityMonitor {
  private evaluator: QualityGateEvaluator;
  private monitoringIntervalMs: number;
  private monitoringTimer?: NodeJS.Timeout;
  private latestReport?: QualityMonitoringReport;

  constructor(config: QualityGateConfig, monitoringIntervalHours: number = 1) {
    this.evaluator = new QualityGateEvaluator(config);
    this.monitoringIntervalMs = monitoringIntervalHours * 60 * 60 * 1000;
  }

  /**
   * Start continuous quality monitoring
   */
  startMonitoring(): void {
    this.monitoringTimer = setInterval(() => {
      this.runQualityEvaluation();
    }, this.monitoringIntervalMs);
    
    // Run initial evaluation
    this.runQualityEvaluation();
  }

  /**
   * Stop quality monitoring
   */
  stopMonitoring(): void {
    if (this.monitoringTimer) {
      clearInterval(this.monitoringTimer);
      this.monitoringTimer = undefined;
    }
  }

  /**
   * Run comprehensive quality evaluation
   */
  async runQualityEvaluation(): Promise<QualityMonitoringReport> {
    const span = LensTracer.createChildSpan('quality_evaluation');
    
    try {
      // Collect current system metrics (would integrate with actual metric collectors)
      const metrics = await this.collectSystemMetrics();
      
      // Evaluate all system gates
      const allResults: QualityGateResult[] = [
        ...this.evaluator.evaluateSliceRecallGates(metrics),
        ...this.evaluator.evaluateBuildPriorsGates(metrics),
        ...this.evaluator.evaluateMultiPlanGates(metrics),
        ...this.evaluator.evaluateCacheAdmissionGates(metrics),
      ];
      
      // Calculate summary statistics
      const passed = allResults.filter(r => r.status === 'pass').length;
      const failed = allResults.filter(r => r.status === 'fail').length;
      const warnings = allResults.filter(r => r.status === 'warning').length;
      
      // Determine overall status
      let overallStatus: 'healthy' | 'degraded' | 'critical';
      if (failed === 0) {
        overallStatus = warnings > 0 ? 'degraded' : 'healthy';
      } else {
        overallStatus = failed > allResults.length / 2 ? 'critical' : 'degraded';
      }
      
      // Generate recommendations
      const recommendations = this.generateRecommendations(allResults);
      
      const report: QualityMonitoringReport = {
        report_id: `quality-${Date.now()}`,
        timestamp: new Date(),
        reporting_period_hours: this.monitoringIntervalMs / (60 * 60 * 1000),
        systems_evaluated: ['program-slice-recall', 'build-test-priors', 'speculative-multi-plan', 'cache-admission-learner'],
        gates_passed: passed,
        gates_failed: failed,
        gates_warning: warnings,
        overall_status: overallStatus,
        gate_results: allResults,
        system_metrics: metrics,
        recommendations,
        next_evaluation_time: new Date(Date.now() + this.monitoringIntervalMs),
      };
      
      this.latestReport = report;
      
      span.setAttributes({
        success: true,
        'gates.passed': passed,
        'gates.failed': failed,
        'gates.warnings': warnings,
        'overall_status': overallStatus,
      });
      
      return report;
      
    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Get the latest quality report
   */
  getLatestReport(): QualityMonitoringReport | undefined {
    return this.latestReport;
  }

  // Private helper methods

  private async collectSystemMetrics(): Promise<SystemMetrics> {
    // This would integrate with actual metric collection systems
    // For now, returning mock metrics
    return {
      // Program Slice Recall
      slice_recall_at_50: 0.758, // +0.8pp improvement
      slice_p95_latency_ms: 15.6, // +0.7ms increase
      slice_span_coverage: 1.0, // 100%
      slice_vendor_veto_honored: 1.0, // 100%
      slice_rollout_percentage: 25, // 25%

      // Build/Test Priors
      build_ndcg_at_10_delta: 0.006, // +0.6pp
      build_sla_recall_at_50: 0.02, // Slight positive
      build_core_at_10_drift: 2.1, // 2.1pp drift
      build_failure_adjacent_queries: 0.15, // 15% of queries
      build_why_mix_kl_divergence: 0.032, // Low divergence

      // Speculative Multi-Plan
      plan_fleet_p99_improvement: -9.3, // -9.3% improvement
      plan_flat_recall_maintained: true,
      plan_p95_latency_ms: 14.2, // +0.3ms increase
      plan_budget_utilization: 7.8, // 7.8%

      // Cache Admission
      cache_admission_hit_rate: 0.342, // 34.2% hit rate
      cache_lru_baseline_hit_rate: 0.308, // 30.8% LRU baseline
      cache_cpu_overhead_percent: 1.8, // 1.8%
      cache_p95_latency_improvement: -0.52, // -0.52ms improvement
      cache_span_drift: 0.0, // 0% drift
    };
  }

  private generateRecommendations(results: QualityGateResult[]): string[] {
    const recommendations: string[] = [];
    const failedResults = results.filter(r => r.status === 'fail');
    const warningResults = results.filter(r => r.status === 'warning');
    
    for (const result of failedResults) {
      switch (result.gate_name) {
        case 'recall_at_50_improvement':
          if (result.measured_value < result.threshold_min!) {
            recommendations.push('Program slice recall improvement below target. Consider expanding slice depth or improving plumbing code detection.');
          }
          break;
        case 'p95_latency_increase':
          if (result.system === 'speculative-multi-plan' && result.measured_value > result.threshold_max!) {
            recommendations.push('CRITICAL: Multi-plan P95 latency exceeded abort threshold. Disable speculative planning immediately.');
          }
          break;
        case 'ndcg_at_10_improvement':
          recommendations.push('Build/test priors not improving nDCG@10 sufficiently. Review failure proximity weighting and decay parameters.');
          break;
        case 'cpu_overhead':
          recommendations.push('Cache admission CPU overhead too high. Increase batching interval or reduce sketch size.');
          break;
      }
    }
    
    for (const result of warningResults) {
      if (result.gate_name === 'why_mix_kl_divergence') {
        recommendations.push('Monitor why-mix KL divergence for semantic crowding. Consider reducing build prior contribution if it increases further.');
      }
    }
    
    if (recommendations.length === 0 && failedResults.length === 0) {
      recommendations.push('All quality gates passing. Systems operating within expected parameters.');
    }
    
    return recommendations;
  }

  /**
   * Export configuration for the default quality gates
   */
  static getDefaultConfig(): QualityGateConfig {
    return {
      // Program Slice Recall gates
      slice_min_recall_improvement: 0.7, // +0.7pp
      slice_max_recall_improvement: 1.2, // +1.2pp
      slice_max_p95_increase: 0.8, // +0.8ms
      slice_min_span_coverage: 1.0, // 100%
      slice_min_vendor_veto_rate: 1.0, // 100%

      // Build/Test Priors gates
      build_min_ndcg_improvement: 0.5, // +0.5pp
      build_min_sla_recall: 0, // ≥0
      build_max_core_drift: 5, // ±5pp
      build_min_failure_adjacent_ratio: 0.1, // 10% minimum for validation

      // Speculative Multi-Plan gates
      plan_min_p99_improvement: -12, // -12%
      plan_max_p99_improvement: -8, // -8%
      plan_max_p95_increase: 0.6, // +0.6ms
      plan_max_budget_utilization: 10, // ≤10%

      // Cache Admission gates
      cache_min_admission_improvement: 3, // +3pp
      cache_max_admission_improvement: 5, // +5pp
      cache_max_cpu_overhead: 3, // ≤3%
      cache_min_p95_improvement: -0.8, // -0.8ms
      cache_max_p95_improvement: -0.3, // -0.3ms
      cache_max_span_drift: 0, // 0%
    };
  }
}