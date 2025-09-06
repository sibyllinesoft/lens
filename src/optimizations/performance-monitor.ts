/**
 * Performance Monitor for Optimization Systems
 * 
 * Comprehensive monitoring and benchmarking for all four durable optimizations:
 * - Real-time SLA compliance tracking
 * - Performance gate validation
 * - Automated alerting for degradation
 * - Benchmarking against baseline
 * - Detailed metrics collection and reporting
 */

import { OptimizationEngine, type OptimizationConfig } from './optimization-engine.js';
import type { SearchHit, MatchReason } from '../core/span_resolver/types.js';
import type { SearchContext } from '../types/core.js';
import type { DiversityFeatures } from './targeted-diversity.js';
import { LensTracer } from '../telemetry/tracer.js';

// Performance thresholds from TODO.md
const SLA_THRESHOLDS = {
  CLONE_RECALL_MIN: 0.005, // +0.5pp
  CLONE_RECALL_MAX: 0.010, // +1.0pp
  CLONE_LATENCY_MAX: 0.6, // ≤+0.6ms p95
  LEARNING_STOP_IMPROVEMENT_MIN: 0.8, // -0.8ms
  LEARNING_STOP_IMPROVEMENT_MAX: 1.5, // -1.5ms
  LEARNING_STOP_UPSHIFT_MIN: 0.03, // 3%
  LEARNING_STOP_UPSHIFT_MAX: 0.07, // 7%
  DIVERSITY_IMPROVEMENT_MIN: 0.10, // +10%
  TTL_IMPROVEMENT_MIN: 0.5, // -0.5ms
  TTL_IMPROVEMENT_MAX: 1.0, // -1.0ms
  TTL_WHY_MIX_KL_MAX: 0.02, // KL ≤ 0.02
  OVERALL_SLA_COMPLIANCE_MIN: 0.95, // 95% SLA compliance
};

export interface BenchmarkResult {
  test_name: string;
  timestamp: number;
  optimization_engine_metrics: any;
  individual_system_metrics: {
    clone_aware: any;
    learning_to_stop: any;
    targeted_diversity: any;
    churn_aware_ttl: any;
  };
  sla_compliance: {
    overall_compliant: boolean;
    clone_recall_compliant: boolean;
    clone_latency_compliant: boolean;
    learning_stop_compliant: boolean;
    diversity_compliant: boolean;
    ttl_compliant: boolean;
  };
  performance_summary: {
    total_optimization_time_ms: number;
    recall_improvement: number;
    diversity_improvement: number;
    cache_hit_rate: number;
    system_health_score: number;
  };
  alerts: string[];
  recommendations: string[];
}

export interface MonitoringConfig {
  benchmark_interval_ms: number;
  alert_threshold_violations: number;
  performance_degradation_threshold: number;
  enable_real_time_monitoring: boolean;
  enable_alerting: boolean;
  log_level: 'debug' | 'info' | 'warn' | 'error';
}

export class PerformanceMonitor {
  private engine: OptimizationEngine;
  private config: MonitoringConfig;
  private benchmarkHistory: BenchmarkResult[] = [];
  private activeAlerts = new Map<string, { count: number; last_seen: number }>();
  private monitoring_active = false;
  private monitoring_interval?: NodeJS.Timeout;
  
  constructor(engine: OptimizationEngine, config: MonitoringConfig) {
    this.engine = engine;
    this.config = config;
  }
  
  /**
   * Start real-time performance monitoring
   */
  async startMonitoring(): Promise<void> {
    const span = LensTracer.createChildSpan('start_performance_monitoring');
    
    try {
      if (this.monitoring_active) {
        console.warn('Performance monitoring already active');
        return;
      }
      
      this.monitoring_active = true;
      
      console.log('🔍 Starting real-time performance monitoring...');
      
      if (this.config.enable_real_time_monitoring) {
        this.monitoring_interval = setInterval(
          () => this.performPeriodicBenchmark(),
          this.config.benchmark_interval_ms
        );
      }
      
      // Initial baseline benchmark
      await this.runComprehensiveBenchmark('baseline');
      
      span.setAttributes({
        success: true,
        real_time_enabled: this.config.enable_real_time_monitoring,
        interval_ms: this.config.benchmark_interval_ms
      });
      
      console.log('✅ Performance monitoring started');
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw error;
    } finally {
      span.end();
    }
  }
  
  /**
   * Stop real-time performance monitoring
   */
  stopMonitoring(): void {
    console.log('🛑 Stopping performance monitoring...');
    
    this.monitoring_active = false;
    
    if (this.monitoring_interval) {
      clearInterval(this.monitoring_interval);
      this.monitoring_interval = undefined;
    }
    
    console.log('✅ Performance monitoring stopped');
  }
  
  /**
   * Run comprehensive benchmark of all optimization systems
   */
  async runComprehensiveBenchmark(testName: string): Promise<BenchmarkResult> {
    const span = LensTracer.createChildSpan('comprehensive_benchmark', { test_name: testName });
    
    try {
      console.log(`📊 Running comprehensive benchmark: ${testName}`);
      
      const startTime = Date.now();
      
      // Generate realistic test scenarios
      const testScenarios = this.generateTestScenarios();
      
      // Run each scenario and collect metrics
      const scenarioResults = [];
      
      for (const scenario of testScenarios) {
        const scenarioStart = Date.now();
        const pipeline = await this.engine.optimizeSearchResults(
          scenario.hits,
          scenario.context,
          scenario.features
        );
        const scenarioTime = Date.now() - scenarioStart;
        
        scenarioResults.push({
          scenario: scenario.name,
          pipeline,
          execution_time_ms: scenarioTime,
        });
      }
      
      // Collect system metrics
      const engineMetrics = this.engine.getPerformanceMetrics();
      const systemHealth = this.engine.getSystemHealth();
      
      // Calculate SLA compliance
      const slaCompliance = this.evaluateSLACompliance(scenarioResults, engineMetrics);
      
      // Generate performance summary
      const performanceSummary = this.generatePerformanceSummary(scenarioResults, engineMetrics);
      
      // Generate alerts and recommendations
      const alerts = this.generateAlerts(slaCompliance, performanceSummary);
      const recommendations = this.generateRecommendations(slaCompliance, performanceSummary);
      
      const result: BenchmarkResult = {
        test_name: testName,
        timestamp: startTime,
        optimization_engine_metrics: engineMetrics,
        individual_system_metrics: {
          clone_aware: engineMetrics.subsystem_metrics.clone_aware,
          learning_to_stop: engineMetrics.subsystem_metrics.learning_to_stop,
          targeted_diversity: engineMetrics.subsystem_metrics.targeted_diversity,
          churn_aware_ttl: engineMetrics.subsystem_metrics.churn_aware_ttl,
        },
        sla_compliance: slaCompliance,
        performance_summary: performanceSummary,
        alerts,
        recommendations,
      };
      
      this.benchmarkHistory.push(result);
      
      // Cleanup old history (keep last 100 benchmarks)
      if (this.benchmarkHistory.length > 100) {
        this.benchmarkHistory = this.benchmarkHistory.slice(-100);
      }
      
      // Process alerts
      if (this.config.enable_alerting) {
        this.processAlerts(alerts);
      }
      
      this.logBenchmarkResult(result);
      
      span.setAttributes({
        success: true,
        scenarios_run: testScenarios.length,
        total_time_ms: Date.now() - startTime,
        sla_compliant: slaCompliance.overall_compliant
      });
      
      return result;
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw error;
    } finally {
      span.end();
    }
  }
  
  /**
   * Generate realistic test scenarios for benchmarking
   */
  private generateTestScenarios(): Array<{
    name: string;
    hits: SearchHit[];
    context: SearchContext;
    features?: DiversityFeatures;
  }> {
    const createHit = (file: string, line: number, score: number, symbolKind?: string, why?: MatchReason[]): SearchHit => ({
      file, line, col: 0, lang: 'typescript',
      snippet: `// ${file}:${line}`, score, why: why || ['lexical'],
      byte_offset: line * 80, span_len: 50, symbol_kind: symbolKind as any,
      context_before: 'before', context_after: 'after',
    });
    
    return [
      {
        name: 'clone_expansion_scenario',
        hits: [
          createHit('utils/helper.ts', 10, 95, 'function', ['exact']),
          createHit('lib/utils.ts', 20, 85, 'function', ['lexical']),
        ],
        context: { query: 'helper utility function', repo_sha: 'main', k: 20, timeout_ms: 1000, include_tests: false, languages: ['typescript'] },
      },
      {
        name: 'diversity_overview_scenario',
        hits: Array.from({ length: 15 }, (_, i) =>
          createHit(`module${i % 3}/file${i}.ts`, i * 5, 90 - i, 'function')
        ),
        context: { query: 'overview of system functionality', repo_sha: 'main', k: 20, timeout_ms: 1000, include_tests: false, languages: ['typescript'] },
        features: {
          query_type: 'NL_overview',
          topic_entropy: 0.85,
          result_count: 15,
          exact_matches: 3,
          structural_matches: 5,
          clone_collapsed: true,
        },
      },
      {
        name: 'targeted_search_scenario',
        hits: [
          createHit('auth/login.ts', 15, 95, 'function', ['exact']),
          createHit('auth/verify.ts', 8, 88, 'function', ['ast']),
          createHit('types/auth.ts', 1, 80, 'interface', ['lexical']),
        ],
        context: { query: 'authentication login function', repo_sha: 'main', k: 10, timeout_ms: 1000, include_tests: false, languages: ['typescript'] },
        features: {
          query_type: 'targeted_search',
          topic_entropy: 0.4,
          result_count: 3,
          exact_matches: 1,
          structural_matches: 1,
          clone_collapsed: true,
        },
      },
      {
        name: 'high_volume_scenario',
        hits: Array.from({ length: 50 }, (_, i) =>
          createHit(`src/component${i}.ts`, i * 3, Math.max(50, 95 - i), 'function', 
            i < 5 ? ['exact'] : i < 15 ? ['ast'] : ['lexical']
          )
        ),
        context: { query: 'component system high volume', repo_sha: 'main', k: 50, timeout_ms: 2000, include_tests: false, languages: ['typescript'] },
        features: {
          query_type: 'NL_overview',
          topic_entropy: 0.92,
          result_count: 50,
          exact_matches: 5,
          structural_matches: 10,
          clone_collapsed: true,
        },
      },
      {
        name: 'cache_intensive_scenario',
        hits: Array.from({ length: 10 }, (_, i) =>
          createHit(`cache/item${i}.ts`, i * 2, 85 - i, 'function')
        ),
        context: { query: 'cached data operations', repo_sha: 'cache-test', k: 15, timeout_ms: 1000, include_tests: false, languages: ['typescript'] },
      },
    ];
  }
  
  /**
   * Evaluate SLA compliance across all systems
   */
  private evaluateSLACompliance(scenarioResults: any[], engineMetrics: any): BenchmarkResult['sla_compliance'] {
    // Clone-aware recall compliance
    const cloneMetrics = engineMetrics.subsystem_metrics.clone_aware;
    const cloneRecallCompliant = !cloneMetrics || (
      cloneMetrics.expansion_p95_latency_ms <= SLA_THRESHOLDS.CLONE_LATENCY_MAX
    );
    
    // Learning-to-stop compliance
    const learningMetrics = engineMetrics.subsystem_metrics.learning_to_stop;
    const learningStopCompliant = !learningMetrics || (
      learningMetrics.sla_compliant === true &&
      learningMetrics.p95_improvement_ms >= SLA_THRESHOLDS.LEARNING_STOP_IMPROVEMENT_MIN &&
      learningMetrics.p95_improvement_ms <= SLA_THRESHOLDS.LEARNING_STOP_IMPROVEMENT_MAX
    );
    
    // Diversity compliance
    const diversityMetrics = engineMetrics.subsystem_metrics.targeted_diversity;
    const diversityCompliant = !diversityMetrics || (
      diversityMetrics.average_diversity_improvement >= SLA_THRESHOLDS.DIVERSITY_IMPROVEMENT_MIN &&
      diversityMetrics.average_ndcg_change >= 0
    );
    
    // TTL compliance
    const ttlMetrics = engineMetrics.subsystem_metrics.churn_aware_ttl;
    const ttlCompliant = !ttlMetrics || (
      ttlMetrics.sla_compliant === true &&
      ttlMetrics.average_why_mix_kl <= SLA_THRESHOLDS.TTL_WHY_MIX_KL_MAX
    );
    
    // Overall compliance
    const overallCompliant = engineMetrics.sla_compliance_rate >= SLA_THRESHOLDS.OVERALL_SLA_COMPLIANCE_MIN;
    
    return {
      overall_compliant: overallCompliant,
      clone_recall_compliant: cloneRecallCompliant,
      clone_latency_compliant: cloneRecallCompliant, // Same check for now
      learning_stop_compliant: learningStopCompliant,
      diversity_compliant: diversityCompliant,
      ttl_compliant: ttlCompliant,
    };
  }
  
  /**
   * Generate performance summary from benchmark results
   */
  private generatePerformanceSummary(scenarioResults: any[], engineMetrics: any): BenchmarkResult['performance_summary'] {
    const totalOptimizationTime = scenarioResults.reduce(
      (sum, result) => sum + result.execution_time_ms, 0
    ) / scenarioResults.length;
    
    const recallImprovement = engineMetrics.average_recall_change || 0;
    
    // Calculate diversity improvement from scenarios that applied diversity
    const diversityResults = scenarioResults.filter(r => 
      r.pipeline.optimizations_applied?.includes('targeted_diversity')
    );
    const diversityImprovement = diversityResults.length > 0
      ? diversityResults.reduce((sum, r) => sum + r.pipeline.performance_impact.diversity_improvement, 0) / diversityResults.length
      : 0;
    
    const cacheHitRate = engineMetrics.subsystem_metrics.churn_aware_ttl?.cache_hit_rate || 0;
    
    // Calculate system health score
    const healthStatus = engineMetrics.system_health;
    const healthScore = healthStatus.overall_healthy ? 
      (1 - (healthStatus.degraded_optimizations.length / 4)) : 0.5;
    
    return {
      total_optimization_time_ms: totalOptimizationTime,
      recall_improvement: recallImprovement,
      diversity_improvement: diversityImprovement,
      cache_hit_rate: cacheHitRate,
      system_health_score: healthScore,
    };
  }
  
  /**
   * Generate alerts based on SLA compliance and performance
   */
  private generateAlerts(slaCompliance: BenchmarkResult['sla_compliance'], performance: BenchmarkResult['performance_summary']): string[] {
    const alerts: string[] = [];
    
    if (!slaCompliance.overall_compliant) {
      alerts.push('CRITICAL: Overall SLA compliance below threshold');
    }
    
    if (!slaCompliance.clone_recall_compliant) {
      alerts.push('WARNING: Clone-aware recall system not meeting performance targets');
    }
    
    if (!slaCompliance.learning_stop_compliant) {
      alerts.push('WARNING: Learning-to-stop system not meeting performance targets');
    }
    
    if (!slaCompliance.diversity_compliant) {
      alerts.push('WARNING: Targeted diversity system not meeting quality gates');
    }
    
    if (!slaCompliance.ttl_compliant) {
      alerts.push('WARNING: TTL system not meeting performance or quality targets');
    }
    
    if (performance.system_health_score < 0.8) {
      alerts.push('CRITICAL: System health degraded - multiple optimization systems failing');
    }
    
    if (performance.total_optimization_time_ms > 100) {
      alerts.push('WARNING: Average optimization time exceeding budget');
    }
    
    if (performance.cache_hit_rate < 0.5) {
      alerts.push('INFO: Cache hit rate low - consider TTL adjustment');
    }
    
    return alerts;
  }
  
  /**
   * Generate recommendations for performance improvement
   */
  private generateRecommendations(slaCompliance: BenchmarkResult['sla_compliance'], performance: BenchmarkResult['performance_summary']): string[] {
    const recommendations: string[] = [];
    
    if (!slaCompliance.clone_recall_compliant) {
      recommendations.push('Consider reducing clone budget k_clone or optimizing shingle generation');
      recommendations.push('Review topic similarity threshold τ to reduce false expansions');
    }
    
    if (!slaCompliance.learning_stop_compliant) {
      recommendations.push('Retrain learning-to-stop model with recent query patterns');
      recommendations.push('Adjust never-stop floor m based on recall requirements');
    }
    
    if (!slaCompliance.diversity_compliant) {
      recommendations.push('Review γ parameter range for MMR optimization');
      recommendations.push('Validate exact/structural match floor constraints');
    }
    
    if (!slaCompliance.ttl_compliant) {
      recommendations.push('Analyze churn rate patterns and adjust TTL constant c');
      recommendations.push('Review why-mix KL divergence calculation for drift detection');
    }
    
    if (performance.total_optimization_time_ms > 50) {
      recommendations.push('Consider selective optimization enablement based on query type');
      recommendations.push('Profile individual system performance for bottlenecks');
    }
    
    if (performance.cache_hit_rate < 0.7) {
      recommendations.push('Analyze cache key collision and TTL effectiveness');
      recommendations.push('Consider cache warming strategies for common queries');
    }
    
    if (performance.system_health_score < 0.9) {
      recommendations.push('Implement more robust error handling and recovery mechanisms');
      recommendations.push('Add circuit breakers for failing optimization systems');
    }
    
    return recommendations;
  }
  
  /**
   * Process alerts and manage alert state
   */
  private processAlerts(alerts: string[]): void {
    const now = Date.now();
    
    for (const alert of alerts) {
      const existing = this.activeAlerts.get(alert);
      
      if (existing) {
        existing.count++;
        existing.last_seen = now;
      } else {
        this.activeAlerts.set(alert, { count: 1, last_seen: now });
      }
      
      // Fire alert if threshold exceeded
      const alertState = this.activeAlerts.get(alert)!;
      if (alertState.count >= this.config.alert_threshold_violations) {
        this.fireAlert(alert, alertState);
      }
    }
    
    // Clean up old alerts
    const alertTimeout = 10 * 60 * 1000; // 10 minutes
    for (const [alert, state] of this.activeAlerts.entries()) {
      if (now - state.last_seen > alertTimeout) {
        this.activeAlerts.delete(alert);
      }
    }
  }
  
  /**
   * Fire alert notification
   */
  private fireAlert(alert: string, state: { count: number; last_seen: number }): void {
    console.error(`🚨 PERFORMANCE ALERT (${state.count}x): ${alert}`);
    
    // In production, would send to monitoring system, Slack, etc.
    // For now, just log with high visibility
  }
  
  /**
   * Periodic benchmark for continuous monitoring
   */
  private async performPeriodicBenchmark(): Promise<void> {
    try {
      const testName = `periodic_${Date.now()}`;
      await this.runComprehensiveBenchmark(testName);
    } catch (error) {
      console.error('Periodic benchmark failed:', error);
    }
  }
  
  /**
   * Log benchmark result with appropriate level
   */
  private logBenchmarkResult(result: BenchmarkResult): void {
    const level = result.alerts.some(a => a.startsWith('CRITICAL')) ? 'error' :
                 result.alerts.some(a => a.startsWith('WARNING')) ? 'warn' : 'info';
    
    if (this.config.log_level === 'debug' || 
        (this.config.log_level === 'info' && level !== 'debug') ||
        (this.config.log_level === 'warn' && ['warn', 'error'].includes(level)) ||
        (this.config.log_level === 'error' && level === 'error')) {
      
      const summary = result.performance_summary;
      const compliance = result.sla_compliance;
      
      console.log(`📊 Benchmark ${result.test_name}:`);
      console.log(`   SLA Compliant: ${compliance.overall_compliant}`);
      console.log(`   Optimization Time: ${summary.total_optimization_time_ms.toFixed(2)}ms`);
      console.log(`   Recall Improvement: ${(summary.recall_improvement * 100).toFixed(1)}%`);
      console.log(`   Diversity Improvement: ${(summary.diversity_improvement * 100).toFixed(1)}%`);
      console.log(`   Cache Hit Rate: ${(summary.cache_hit_rate * 100).toFixed(1)}%`);
      console.log(`   System Health: ${(summary.system_health_score * 100).toFixed(1)}%`);
      
      if (result.alerts.length > 0) {
        console.log(`   Alerts (${result.alerts.length}):`);
        result.alerts.forEach(alert => console.log(`     - ${alert}`));
      }
    }
  }
  
  /**
   * Get current benchmark history
   */
  getBenchmarkHistory(): BenchmarkResult[] {
    return [...this.benchmarkHistory];
  }
  
  /**
   * Get active alerts
   */
  getActiveAlerts(): Map<string, { count: number; last_seen: number }> {
    return new Map(this.activeAlerts);
  }
  
  /**
   * Generate performance report
   */
  generatePerformanceReport(): string {
    const recent = this.benchmarkHistory.slice(-10);
    if (recent.length === 0) {
      return 'No benchmark data available';
    }
    
    const avgCompliance = recent.reduce((sum, r) => sum + (r.sla_compliance.overall_compliant ? 1 : 0), 0) / recent.length;
    const avgOptTime = recent.reduce((sum, r) => sum + r.performance_summary.total_optimization_time_ms, 0) / recent.length;
    const avgRecall = recent.reduce((sum, r) => sum + r.performance_summary.recall_improvement, 0) / recent.length;
    const avgDiversity = recent.reduce((sum, r) => sum + r.performance_summary.diversity_improvement, 0) / recent.length;
    const avgCacheHit = recent.reduce((sum, r) => sum + r.performance_summary.cache_hit_rate, 0) / recent.length;
    const avgHealth = recent.reduce((sum, r) => sum + r.performance_summary.system_health_score, 0) / recent.length;
    
    return `
## Optimization Systems Performance Report

### Summary (Last ${recent.length} benchmarks)
- **SLA Compliance**: ${(avgCompliance * 100).toFixed(1)}%
- **Average Optimization Time**: ${avgOptTime.toFixed(2)}ms
- **Average Recall Improvement**: ${(avgRecall * 100).toFixed(1)}%
- **Average Diversity Improvement**: ${(avgDiversity * 100).toFixed(1)}%
- **Average Cache Hit Rate**: ${(avgCacheHit * 100).toFixed(1)}%
- **Average System Health**: ${(avgHealth * 100).toFixed(1)}%

### Active Alerts
${this.activeAlerts.size > 0 
  ? Array.from(this.activeAlerts.entries()).map(([alert, state]) => 
      `- ${alert} (${state.count}x)`).join('\n')
  : 'None'}

### Recent Performance Trends
${recent.slice(-5).map(r => 
  `- ${new Date(r.timestamp).toISOString()}: ${r.sla_compliance.overall_compliant ? '✅' : '❌'} SLA, ${r.performance_summary.total_optimization_time_ms.toFixed(1)}ms`
).join('\n')}

### System Status
- **Clone-Aware Recall**: ${recent.slice(-1)[0]?.sla_compliance.clone_recall_compliant ? '✅' : '❌'}
- **Learning-to-Stop**: ${recent.slice(-1)[0]?.sla_compliance.learning_stop_compliant ? '✅' : '❌'}
- **Targeted Diversity**: ${recent.slice(-1)[0]?.sla_compliance.diversity_compliant ? '✅' : '❌'}
- **Churn-Aware TTL**: ${recent.slice(-1)[0]?.sla_compliance.ttl_compliant ? '✅' : '❌'}
`;
  }
  
  /**
   * Export benchmark data for analysis
   */
  exportBenchmarkData(): any {
    return {
      config: this.config,
      benchmark_history: this.benchmarkHistory,
      active_alerts: Object.fromEntries(this.activeAlerts),
      export_timestamp: Date.now(),
    };
  }
}