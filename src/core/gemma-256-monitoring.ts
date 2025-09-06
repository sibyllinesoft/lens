/**
 * Gemma-256 Production Monitoring Dashboards
 * 
 * Pin four hero SLIs: p95/p99 latency, QPS@150ms, SLA-Recall@50, router upshift rate
 * Plus failure taxonomy {Z0,T,P,F} and per-language slices for complete observability.
 */

import type { SearchHit, SearchContext } from './span_resolver/types.js';
import type { RoutingDecision } from './gemma-256-hybrid-router.js';
import type { CanaryMetrics } from './gemma-256-canary-rollout.js';
import { LensTracer } from '../telemetry/tracer.js';

export interface MonitoringConfig {
  enabled: boolean;
  // Hero SLI collection
  sli_collection_interval_seconds: number;
  sli_retention_hours: number;
  // Failure taxonomy tracking
  failure_tracking_enabled: boolean;
  failure_retention_hours: number;
  // Language slice monitoring
  language_slice_tracking: boolean;
  supported_languages: string[];
  // Alerting configuration
  alerting: {
    enabled: boolean;
    sli_breach_thresholds: HeroSLIThresholds;
    failure_rate_thresholds: FailureTaxonomyThresholds;
    alert_cooldown_minutes: number;
  };
  // Dashboard configuration
  dashboard: {
    refresh_interval_seconds: number;
    historical_window_hours: number;
    real_time_window_minutes: number;
  };
}

export interface HeroSLIThresholds {
  p95_latency_ms: { warning: number; critical: number };
  p99_latency_ms: { warning: number; critical: number };
  qps_at_150ms: { warning: number; critical: number };
  sla_recall_at_50: { warning: number; critical: number };
  router_upshift_rate: { warning: number; critical: number };
}

export interface FailureTaxonomyThresholds {
  z0_zero_results_pct: { warning: number; critical: number };
  t_timeout_pct: { warning: number; critical: number };
  p_precision_degradation_pct: { warning: number; critical: number };
  f_functional_failure_pct: { warning: number; critical: number };
}

// Hero SLI Data Structures

export interface HeroSLISnapshot {
  timestamp: Date;
  window_minutes: number;
  
  // The Four Hero SLIs
  p95_latency_ms: number;
  p99_latency_ms: number;
  qps_at_150ms: number;
  sla_recall_at_50: number;
  router_upshift_rate: number;
  
  // Contextual data
  total_queries: number;
  successful_queries: number;
  upshifted_queries: number;
  
  // SLI breach flags
  breaches: {
    p95_latency: 'ok' | 'warning' | 'critical';
    p99_latency: 'ok' | 'warning' | 'critical';
    qps_at_150ms: 'ok' | 'warning' | 'critical';
    sla_recall_at_50: 'ok' | 'warning' | 'critical';
    router_upshift_rate: 'ok' | 'warning' | 'critical';
  };
}

// Failure Taxonomy {Z0, T, P, F}

export type FailureType = 'Z0' | 'T' | 'P' | 'F';

export interface FailureEvent {
  timestamp: Date;
  query: string;
  failure_type: FailureType;
  failure_reason: string;
  context: SearchContext;
  latency_ms: number;
  language?: string;
  routing_decision?: RoutingDecision;
  // Additional context
  candidates_count: number;
  upshift_attempted: boolean;
  system_state: 'baseline_768d' | 'hybrid_256d' | 'canary_rollout';
}

export interface FailureTaxonomySnapshot {
  timestamp: Date;
  window_minutes: number;
  
  // Failure counts by type
  z0_zero_results: number;
  t_timeout: number;
  p_precision_degradation: number;
  f_functional_failure: number;
  
  // Failure rates (percentage of total queries)
  z0_rate_pct: number;
  t_rate_pct: number;
  p_rate_pct: number;
  f_rate_pct: number;
  
  total_failures: number;
  total_queries: number;
  overall_failure_rate_pct: number;
}

// Language Slice Monitoring

export interface LanguageSliceMetrics {
  language: string;
  timestamp: Date;
  window_minutes: number;
  
  // Per-language SLIs
  queries_count: number;
  p95_latency_ms: number;
  recall_at_50: number;
  upshift_rate: number;
  
  // Per-language failure rates
  failure_rate_pct: number;
  z0_rate_pct: number;
  
  // Language-specific insights
  avg_query_length: number;
  symbol_query_pct: number;
  nl_query_pct: number;
}

/**
 * Hero SLI Collector - Tracks the four critical metrics
 */
export class HeroSLICollector {
  private recentQueries: Array<{
    timestamp: Date;
    latency_ms: number;
    success: boolean;
    recall_at_50: number;
    upshifted: boolean;
    qps_bucket: '0-150ms' | '150ms+';
  }> = [];
  
  private sliHistory: HeroSLISnapshot[] = [];
  
  constructor(private config: MonitoringConfig) {}
  
  /**
   * Record query metrics for SLI calculation
   */
  recordQuery(
    latencyMs: number,
    success: boolean,
    recallAt50: number,
    upshifted: boolean
  ): void {
    const qpsBucket = latencyMs <= 150 ? '0-150ms' : '150ms+';
    
    this.recentQueries.push({
      timestamp: new Date(),
      latency_ms: latencyMs,
      success,
      recall_at_50: recallAt50,
      upshifted,
      qps_bucket: qpsBucket
    });
    
    // Keep bounded to last hour of data
    const oneHourAgo = Date.now() - (60 * 60 * 1000);
    this.recentQueries = this.recentQueries.filter(q => 
      q.timestamp.getTime() > oneHourAgo
    );
  }
  
  /**
   * Calculate current Hero SLI snapshot
   */
  calculateCurrentSLIs(windowMinutes = 5): HeroSLISnapshot {
    const windowStart = Date.now() - (windowMinutes * 60 * 1000);
    const windowQueries = this.recentQueries.filter(q => 
      q.timestamp.getTime() >= windowStart
    );
    
    if (windowQueries.length === 0) {
      return this.getEmptySLISnapshot(windowMinutes);
    }
    
    // Calculate latency percentiles
    const latencies = windowQueries.map(q => q.latency_ms).sort((a, b) => a - b);
    const p95Index = Math.floor(latencies.length * 0.95);
    const p99Index = Math.floor(latencies.length * 0.99);
    
    const p95Latency = latencies[p95Index] || 0;
    const p99Latency = latencies[p99Index] || 0;
    
    // Calculate QPS@150ms
    const queriesWithin150ms = windowQueries.filter(q => q.qps_bucket === '0-150ms').length;
    const qpsAt150ms = (queriesWithin150ms / windowMinutes) * 60; // Queries per minute * 60 = QPS
    
    // Calculate SLA Recall@50
    const recallValues = windowQueries.map(q => q.recall_at_50);
    const avgRecall = recallValues.reduce((sum, r) => sum + r, 0) / recallValues.length;
    
    // Calculate router upshift rate
    const upshiftedCount = windowQueries.filter(q => q.upshifted).length;
    const upshiftRate = upshiftedCount / windowQueries.length;
    
    const snapshot: HeroSLISnapshot = {
      timestamp: new Date(),
      window_minutes: windowMinutes,
      p95_latency_ms: p95Latency,
      p99_latency_ms: p99Latency,
      qps_at_150ms: qpsAt150ms,
      sla_recall_at_50: avgRecall,
      router_upshift_rate: upshiftRate,
      total_queries: windowQueries.length,
      successful_queries: windowQueries.filter(q => q.success).length,
      upshifted_queries: upshiftedCount,
      breaches: this.calculateBreaches(p95Latency, p99Latency, qpsAt150ms, avgRecall, upshiftRate)
    };
    
    this.sliHistory.push(snapshot);
    
    // Keep history bounded
    const retentionCutoff = Date.now() - (this.config.sli_retention_hours * 60 * 60 * 1000);
    this.sliHistory = this.sliHistory.filter(s => s.timestamp.getTime() > retentionCutoff);
    
    return snapshot;
  }
  
  /**
   * Calculate SLI breach levels
   */
  private calculateBreaches(
    p95: number, 
    p99: number, 
    qps: number, 
    recall: number, 
    upshift: number
  ): HeroSLISnapshot['breaches'] {
    const thresholds = this.config.alerting.sli_breach_thresholds;
    
    return {
      p95_latency: this.getBreachLevel(p95, thresholds.p95_latency_ms),
      p99_latency: this.getBreachLevel(p99, thresholds.p99_latency_ms),
      qps_at_150ms: this.getBreachLevel(qps, thresholds.qps_at_150ms, true), // Higher is better
      sla_recall_at_50: this.getBreachLevel(recall, thresholds.sla_recall_at_50, true), // Higher is better
      router_upshift_rate: this.getBreachLevel(upshift, thresholds.router_upshift_rate)
    };
  }
  
  private getBreachLevel(
    value: number, 
    thresholds: { warning: number; critical: number },
    higherIsBetter = false
  ): 'ok' | 'warning' | 'critical' {
    if (higherIsBetter) {
      if (value < thresholds.critical) return 'critical';
      if (value < thresholds.warning) return 'warning';
      return 'ok';
    } else {
      if (value > thresholds.critical) return 'critical';
      if (value > thresholds.warning) return 'warning';
      return 'ok';
    }
  }
  
  private getEmptySLISnapshot(windowMinutes: number): HeroSLISnapshot {
    return {
      timestamp: new Date(),
      window_minutes: windowMinutes,
      p95_latency_ms: 0,
      p99_latency_ms: 0,
      qps_at_150ms: 0,
      sla_recall_at_50: 0,
      router_upshift_rate: 0,
      total_queries: 0,
      successful_queries: 0,
      upshifted_queries: 0,
      breaches: {
        p95_latency: 'ok',
        p99_latency: 'ok',
        qps_at_150ms: 'ok',
        sla_recall_at_50: 'ok',
        router_upshift_rate: 'ok'
      }
    };
  }
  
  /**
   * Get SLI history for dashboard
   */
  getSLIHistory(hours = 24): HeroSLISnapshot[] {
    const cutoff = Date.now() - (hours * 60 * 60 * 1000);
    return this.sliHistory.filter(s => s.timestamp.getTime() > cutoff);
  }
  
  getStats() {
    return {
      recent_queries: this.recentQueries.length,
      sli_history: this.sliHistory.length,
      latest_sli: this.sliHistory[this.sliHistory.length - 1]
    };
  }
}

/**
 * Failure Taxonomy Tracker - Categorizes failures as {Z0, T, P, F}
 */
export class FailureTaxonomyTracker {
  private failures: FailureEvent[] = [];
  private taxonomyHistory: FailureTaxonomySnapshot[] = [];
  
  constructor(private config: MonitoringConfig) {}
  
  /**
   * Record a failure event with taxonomy classification
   */
  recordFailure(
    query: string,
    context: SearchContext,
    latencyMs: number,
    results: SearchHit[],
    routingDecision?: RoutingDecision
  ): void {
    if (!this.config.failure_tracking_enabled) {
      return;
    }
    
    const failureType = this.classifyFailure(results, latencyMs, context);
    if (!failureType) {
      return; // Not a failure
    }
    
    const failure: FailureEvent = {
      timestamp: new Date(),
      query,
      failure_type: failureType,
      failure_reason: this.getFailureReason(failureType, results, latencyMs),
      context,
      latency_ms: latencyMs,
      language: this.detectLanguage(query, context),
      routing_decision: routingDecision,
      candidates_count: results.length,
      upshift_attempted: routingDecision?.use_768d || false,
      system_state: this.getSystemState(routingDecision)
    };
    
    this.failures.push(failure);
    
    // Keep history bounded
    const retentionCutoff = Date.now() - (this.config.failure_retention_hours * 60 * 60 * 1000);
    this.failures = this.failures.filter(f => f.timestamp.getTime() > retentionCutoff);
    
    console.warn(`🚨 Failure recorded: ${failureType} - ${failure.failure_reason} (${query})`);
  }
  
  /**
   * Classify failure into taxonomy {Z0, T, P, F}
   */
  private classifyFailure(
    results: SearchHit[],
    latencyMs: number,
    context: SearchContext
  ): FailureType | null {
    // Z0: Zero results
    if (results.length === 0) {
      return 'Z0';
    }
    
    // T: Timeout (using SLA threshold)
    if (latencyMs > 25) { // p95 SLA threshold
      return 'T';
    }
    
    // P: Precision degradation (low-quality results)
    const avgScore = results.reduce((sum, r) => sum + r.score, 0) / results.length;
    if (avgScore < 0.1) { // Very low scores indicate precision issues
      return 'P';
    }
    
    // F: Functional failure (system error, malformed results)
    const hasValidResults = results.every(r => 
      r.score >= 0 && 
      r.score <= 1 && 
      typeof r.score === 'number' &&
      !isNaN(r.score)
    );
    
    if (!hasValidResults) {
      return 'F';
    }
    
    // Not a failure by our taxonomy
    return null;
  }
  
  private getFailureReason(
    type: FailureType,
    results: SearchHit[],
    latencyMs: number
  ): string {
    switch (type) {
      case 'Z0':
        return `No results returned for query`;
      case 'T':
        return `Query timeout: ${latencyMs}ms > 25ms SLA`;
      case 'P':
        const avgScore = results.reduce((sum, r) => sum + r.score, 0) / results.length;
        return `Low precision: avg score ${avgScore.toFixed(3)} < 0.1`;
      case 'F':
        return `Functional failure: invalid result format or system error`;
      default:
        return 'Unknown failure type';
    }
  }
  
  private detectLanguage(query: string, context: SearchContext): string | undefined {
    // Simple language detection based on query patterns
    // In production, would use more sophisticated detection
    
    if (/\b(class|def|function|import|from)\b/.test(query)) {
      return 'python';
    }
    if (/\b(const|let|var|function|class|interface)\b/.test(query)) {
      return 'typescript';
    }
    if (/\b(fn|struct|impl|use|mod)\b/.test(query)) {
      return 'rust';
    }
    if (/\b(func|type|interface|package|import)\b/.test(query)) {
      return 'go';
    }
    
    return 'unknown';
  }
  
  private getSystemState(routingDecision?: RoutingDecision): 'baseline_768d' | 'hybrid_256d' | 'canary_rollout' {
    if (!routingDecision) {
      return 'baseline_768d';
    }
    
    // Simplified system state detection
    return routingDecision.use_768d ? 'hybrid_256d' : 'canary_rollout';
  }
  
  /**
   * Calculate failure taxonomy snapshot
   */
  calculateTaxonomySnapshot(windowMinutes = 5): FailureTaxonomySnapshot {
    const windowStart = Date.now() - (windowMinutes * 60 * 1000);
    const windowFailures = this.failures.filter(f => 
      f.timestamp.getTime() >= windowStart
    );
    
    const z0Count = windowFailures.filter(f => f.failure_type === 'Z0').length;
    const tCount = windowFailures.filter(f => f.failure_type === 'T').length;
    const pCount = windowFailures.filter(f => f.failure_type === 'P').length;
    const fCount = windowFailures.filter(f => f.failure_type === 'F').length;
    
    const totalFailures = windowFailures.length;
    
    // Estimate total queries from failure rate (simplified)
    const estimatedTotalQueries = Math.max(totalFailures * 10, totalFailures); // Assume 10% failure rate
    
    const snapshot: FailureTaxonomySnapshot = {
      timestamp: new Date(),
      window_minutes: windowMinutes,
      z0_zero_results: z0Count,
      t_timeout: tCount,
      p_precision_degradation: pCount,
      f_functional_failure: fCount,
      z0_rate_pct: (z0Count / estimatedTotalQueries) * 100,
      t_rate_pct: (tCount / estimatedTotalQueries) * 100,
      p_rate_pct: (pCount / estimatedTotalQueries) * 100,
      f_rate_pct: (fCount / estimatedTotalQueries) * 100,
      total_failures: totalFailures,
      total_queries: estimatedTotalQueries,
      overall_failure_rate_pct: (totalFailures / estimatedTotalQueries) * 100
    };
    
    this.taxonomyHistory.push(snapshot);
    
    // Keep history bounded
    const retentionCutoff = Date.now() - (this.config.failure_retention_hours * 60 * 60 * 1000);
    this.taxonomyHistory = this.taxonomyHistory.filter(s => s.timestamp.getTime() > retentionCutoff);
    
    return snapshot;
  }
  
  /**
   * Get failure taxonomy history
   */
  getTaxonomyHistory(hours = 24): FailureTaxonomySnapshot[] {
    const cutoff = Date.now() - (hours * 60 * 60 * 1000);
    return this.taxonomyHistory.filter(s => s.timestamp.getTime() > cutoff);
  }
  
  /**
   * Get recent failures for investigation
   */
  getRecentFailures(hours = 1): FailureEvent[] {
    const cutoff = Date.now() - (hours * 60 * 60 * 1000);
    return this.failures.filter(f => f.timestamp.getTime() > cutoff);
  }
  
  getStats() {
    return {
      total_failures: this.failures.length,
      taxonomy_snapshots: this.taxonomyHistory.length,
      recent_failures_by_type: {
        Z0: this.failures.filter(f => f.failure_type === 'Z0').length,
        T: this.failures.filter(f => f.failure_type === 'T').length,
        P: this.failures.filter(f => f.failure_type === 'P').length,
        F: this.failures.filter(f => f.failure_type === 'F').length
      }
    };
  }
}

/**
 * Language Slice Monitor - Tracks per-language performance
 */
export class LanguageSliceMonitor {
  private languageMetrics: Map<string, Array<{
    timestamp: Date;
    query: string;
    latency_ms: number;
    recall_at_50: number;
    upshifted: boolean;
    failed: boolean;
    failure_type?: FailureType;
  }>> = new Map();
  
  private sliceHistory: LanguageSliceMetrics[] = [];
  
  constructor(private config: MonitoringConfig) {
    // Initialize tracking for supported languages
    for (const lang of config.supported_languages) {
      this.languageMetrics.set(lang, []);
    }
  }
  
  /**
   * Record query metrics for language slice analysis
   */
  recordLanguageQuery(
    language: string,
    query: string,
    latencyMs: number,
    recallAt50: number,
    upshifted: boolean,
    failed = false,
    failureType?: FailureType
  ): void {
    if (!this.config.language_slice_tracking) {
      return;
    }
    
    const metrics = this.languageMetrics.get(language) || [];
    metrics.push({
      timestamp: new Date(),
      query,
      latency_ms: latencyMs,
      recall_at_50: recallAt50,
      upshifted,
      failed,
      failure_type: failureType
    });
    
    this.languageMetrics.set(language, metrics);
    
    // Keep bounded
    const oneHourAgo = Date.now() - (60 * 60 * 1000);
    const filtered = metrics.filter(m => m.timestamp.getTime() > oneHourAgo);
    this.languageMetrics.set(language, filtered);
  }
  
  /**
   * Calculate language slice metrics
   */
  calculateLanguageSlices(windowMinutes = 5): LanguageSliceMetrics[] {
    const windowStart = Date.now() - (windowMinutes * 60 * 1000);
    const slices: LanguageSliceMetrics[] = [];
    
    for (const [language, metrics] of this.languageMetrics) {
      const windowMetrics = metrics.filter(m => m.timestamp.getTime() >= windowStart);
      
      if (windowMetrics.length === 0) {
        continue;
      }
      
      // Calculate per-language statistics
      const latencies = windowMetrics.map(m => m.latency_ms);
      latencies.sort((a, b) => a - b);
      const p95Index = Math.floor(latencies.length * 0.95);
      const p95Latency = latencies[p95Index] || 0;
      
      const avgRecall = windowMetrics.reduce((sum, m) => sum + m.recall_at_50, 0) / windowMetrics.length;
      const upshiftRate = windowMetrics.filter(m => m.upshifted).length / windowMetrics.length;
      const failureRate = windowMetrics.filter(m => m.failed).length / windowMetrics.length;
      const z0Rate = windowMetrics.filter(m => m.failure_type === 'Z0').length / windowMetrics.length;
      
      // Language-specific insights
      const avgQueryLength = windowMetrics.reduce((sum, m) => sum + m.query.length, 0) / windowMetrics.length;
      const symbolQueryCount = windowMetrics.filter(m => /^[a-zA-Z_][a-zA-Z0-9_]*$/.test(m.query)).length;
      const nlQueryCount = windowMetrics.filter(m => m.query.split(' ').length > 3).length;
      
      const slice: LanguageSliceMetrics = {
        language,
        timestamp: new Date(),
        window_minutes: windowMinutes,
        queries_count: windowMetrics.length,
        p95_latency_ms: p95Latency,
        recall_at_50: avgRecall,
        upshift_rate: upshiftRate,
        failure_rate_pct: failureRate * 100,
        z0_rate_pct: z0Rate * 100,
        avg_query_length: avgQueryLength,
        symbol_query_pct: (symbolQueryCount / windowMetrics.length) * 100,
        nl_query_pct: (nlQueryCount / windowMetrics.length) * 100
      };
      
      slices.push(slice);
    }
    
    // Store historical data
    this.sliceHistory.push(...slices);
    
    // Keep history bounded
    const retentionCutoff = Date.now() - (this.config.sli_retention_hours * 60 * 60 * 1000);
    this.sliceHistory = this.sliceHistory.filter(s => s.timestamp.getTime() > retentionCutoff);
    
    return slices;
  }
  
  /**
   * Get language slice history
   */
  getLanguageHistory(language: string, hours = 24): LanguageSliceMetrics[] {
    const cutoff = Date.now() - (hours * 60 * 60 * 1000);
    return this.sliceHistory.filter(s => 
      s.language === language && s.timestamp.getTime() > cutoff
    );
  }
  
  getStats() {
    const languageStats = new Map<string, number>();
    for (const [language, metrics] of this.languageMetrics) {
      languageStats.set(language, metrics.length);
    }
    
    return {
      tracked_languages: Array.from(this.languageMetrics.keys()),
      recent_queries_per_language: Object.fromEntries(languageStats),
      slice_history: this.sliceHistory.length
    };
  }
}

/**
 * Alert Manager - Handles threshold breaches and notifications
 */
export class AlertManager {
  private activeAlerts = new Map<string, Date>(); // alertKey -> last triggered
  private alertHistory: Array<{
    timestamp: Date;
    alert_type: string;
    severity: 'warning' | 'critical';
    message: string;
    resolved: boolean;
  }> = [];
  
  constructor(private config: MonitoringConfig) {}
  
  /**
   * Check SLI thresholds and trigger alerts
   */
  checkSLIAlerts(snapshot: HeroSLISnapshot): void {
    if (!this.config.alerting.enabled) {
      return;
    }
    
    const alerts = [];
    
    // Check each SLI breach
    if (snapshot.breaches.p95_latency !== 'ok') {
      alerts.push({
        key: 'p95_latency_breach',
        severity: snapshot.breaches.p95_latency,
        message: `P95 latency ${snapshot.p95_latency_ms.toFixed(1)}ms exceeds threshold`
      });
    }
    
    if (snapshot.breaches.p99_latency !== 'ok') {
      alerts.push({
        key: 'p99_latency_breach',
        severity: snapshot.breaches.p99_latency,
        message: `P99 latency ${snapshot.p99_latency_ms.toFixed(1)}ms exceeds threshold`
      });
    }
    
    if (snapshot.breaches.qps_at_150ms !== 'ok') {
      alerts.push({
        key: 'qps_performance_degradation',
        severity: snapshot.breaches.qps_at_150ms,
        message: `QPS@150ms ${snapshot.qps_at_150ms.toFixed(1)} below threshold`
      });
    }
    
    if (snapshot.breaches.sla_recall_at_50 !== 'ok') {
      alerts.push({
        key: 'recall_degradation',
        severity: snapshot.breaches.sla_recall_at_50,
        message: `SLA Recall@50 ${snapshot.sla_recall_at_50.toFixed(3)} below threshold`
      });
    }
    
    if (snapshot.breaches.router_upshift_rate !== 'ok') {
      alerts.push({
        key: 'upshift_rate_anomaly',
        severity: snapshot.breaches.router_upshift_rate,
        message: `Router upshift rate ${(snapshot.router_upshift_rate * 100).toFixed(1)}% outside expected range`
      });
    }
    
    // Trigger alerts with cooldown
    for (const alert of alerts) {
      this.triggerAlert(alert.key, alert.severity as 'warning' | 'critical', alert.message);
    }
  }
  
  /**
   * Check failure taxonomy thresholds
   */
  checkFailureAlerts(snapshot: FailureTaxonomySnapshot): void {
    if (!this.config.alerting.enabled) {
      return;
    }
    
    const thresholds = this.config.alerting.failure_rate_thresholds;
    
    if (snapshot.z0_rate_pct > thresholds.z0_zero_results_pct.critical) {
      this.triggerAlert('z0_critical', 'critical', 
        `Zero results rate ${snapshot.z0_rate_pct.toFixed(1)}% exceeds critical threshold`);
    } else if (snapshot.z0_rate_pct > thresholds.z0_zero_results_pct.warning) {
      this.triggerAlert('z0_warning', 'warning',
        `Zero results rate ${snapshot.z0_rate_pct.toFixed(1)}% exceeds warning threshold`);
    }
    
    if (snapshot.t_rate_pct > thresholds.t_timeout_pct.critical) {
      this.triggerAlert('timeout_critical', 'critical',
        `Timeout rate ${snapshot.t_rate_pct.toFixed(1)}% exceeds critical threshold`);
    }
    
    if (snapshot.p_rate_pct > thresholds.p_precision_degradation_pct.critical) {
      this.triggerAlert('precision_critical', 'critical',
        `Precision degradation rate ${snapshot.p_rate_pct.toFixed(1)}% exceeds critical threshold`);
    }
    
    if (snapshot.f_rate_pct > thresholds.f_functional_failure_pct.critical) {
      this.triggerAlert('functional_critical', 'critical',
        `Functional failure rate ${snapshot.f_rate_pct.toFixed(1)}% exceeds critical threshold`);
    }
  }
  
  /**
   * Trigger alert with cooldown
   */
  private triggerAlert(key: string, severity: 'warning' | 'critical', message: string): void {
    const now = new Date();
    const lastTriggered = this.activeAlerts.get(key);
    
    // Check cooldown
    if (lastTriggered) {
      const cooldownMs = this.config.alerting.alert_cooldown_minutes * 60 * 1000;
      if (now.getTime() - lastTriggered.getTime() < cooldownMs) {
        return; // Still in cooldown
      }
    }
    
    this.activeAlerts.set(key, now);
    
    const alert = {
      timestamp: now,
      alert_type: key,
      severity,
      message,
      resolved: false
    };
    
    this.alertHistory.push(alert);
    
    // Keep alert history bounded
    if (this.alertHistory.length > 1000) {
      this.alertHistory = this.alertHistory.slice(-500);
    }
    
    console.warn(`🚨 ${severity.toUpperCase()} ALERT: ${key} - ${message}`);
    
    // In production, would send to alerting system (PagerDuty, Slack, etc.)
  }
  
  /**
   * Resolve alert
   */
  resolveAlert(key: string): void {
    this.activeAlerts.delete(key);
    
    // Mark as resolved in history
    const recentAlert = this.alertHistory
      .slice(-50) // Check recent alerts
      .find(a => a.alert_type === key && !a.resolved);
    
    if (recentAlert) {
      recentAlert.resolved = true;
      console.log(`✅ Alert resolved: ${key}`);
    }
  }
  
  getStats() {
    return {
      active_alerts: this.activeAlerts.size,
      alert_history: this.alertHistory.length,
      recent_alerts: this.alertHistory.slice(-10)
    };
  }
}

/**
 * Production Monitoring Dashboard
 * Coordinates all monitoring components and provides unified interface
 */
export class Gemma256MonitoringDashboard {
  private sliCollector: HeroSLICollector;
  private failureTracker: FailureTaxonomyTracker;
  private languageMonitor: LanguageSliceMonitor;
  private alertManager: AlertManager;
  private isProduction: boolean;
  
  // Dashboard state
  private dashboardData: {
    last_update: Date;
    hero_slis: HeroSLISnapshot | null;
    failure_taxonomy: FailureTaxonomySnapshot | null;
    language_slices: LanguageSliceMetrics[];
    active_alerts: number;
    system_health: 'healthy' | 'degraded' | 'critical';
  } = {
    last_update: new Date(),
    hero_slis: null,
    failure_taxonomy: null,
    language_slices: [],
    active_alerts: 0,
    system_health: 'healthy'
  };
  
  constructor(config: Partial<MonitoringConfig> = {}, isProduction = true) {
    this.isProduction = isProduction;
    
    // Production monitoring configuration
    const productionConfig: MonitoringConfig = {
      enabled: true,
      sli_collection_interval_seconds: 30,
      sli_retention_hours: 72,               // 3 days of SLI data
      failure_tracking_enabled: true,
      failure_retention_hours: 48,           // 2 days of failure data
      language_slice_tracking: true,
      supported_languages: ['typescript', 'python', 'rust', 'go', 'javascript', 'java', 'cpp'],
      
      alerting: {
        enabled: true,
        sli_breach_thresholds: {
          p95_latency_ms: { warning: 20, critical: 25 },
          p99_latency_ms: { warning: 40, critical: 50 },
          qps_at_150ms: { warning: 100, critical: 50 },
          sla_recall_at_50: { warning: 0.85, critical: 0.80 },
          router_upshift_rate: { warning: 0.08, critical: 0.10 } // Warning at 8%, critical at 10%
        },
        failure_rate_thresholds: {
          z0_zero_results_pct: { warning: 5, critical: 10 },
          t_timeout_pct: { warning: 2, critical: 5 },
          p_precision_degradation_pct: { warning: 3, critical: 7 },
          f_functional_failure_pct: { warning: 1, critical: 3 }
        },
        alert_cooldown_minutes: 10
      },
      
      dashboard: {
        refresh_interval_seconds: 30,
        historical_window_hours: 24,
        real_time_window_minutes: 15
      },
      
      ...config
    };
    
    this.sliCollector = new HeroSLICollector(productionConfig);
    this.failureTracker = new FailureTaxonomyTracker(productionConfig);
    this.languageMonitor = new LanguageSliceMonitor(productionConfig);
    this.alertManager = new AlertManager(productionConfig);
    
    console.log(`📊 Gemma-256 Monitoring Dashboard initialized (production=${isProduction})`);
    console.log(`   Hero SLIs: p95/p99 latency, QPS@150ms, SLA-Recall@50, upshift rate`);
    console.log(`   Failure taxonomy: {Z0, T, P, F}`);
    console.log(`   Language slices: ${productionConfig.supported_languages.join(', ')}`);
  }
  
  /**
   * Record comprehensive query metrics
   */
  recordQuery(
    query: string,
    context: SearchContext,
    results: SearchHit[],
    latencyMs: number,
    recallAt50: number,
    routingDecision?: RoutingDecision,
    language?: string
  ): void {
    const success = results.length > 0 && latencyMs <= 25; // Within SLA
    const upshifted = routingDecision?.use_768d || false;
    
    // Record for Hero SLIs
    this.sliCollector.recordQuery(latencyMs, success, recallAt50, upshifted);
    
    // Record potential failures
    if (!success) {
      this.failureTracker.recordFailure(query, context, latencyMs, results, routingDecision);
    }
    
    // Record language-specific metrics
    if (language && this.languageMonitor) {
      this.languageMonitor.recordLanguageQuery(
        language,
        query,
        latencyMs,
        recallAt50,
        upshifted,
        !success
      );
    }
  }
  
  /**
   * Update dashboard data and check alerts
   */
  async updateDashboard(): Promise<void> {
    const span = LensTracer.createChildSpan('dashboard_update', {
      'production': this.isProduction
    });
    
    try {
      // Calculate current metrics
      const heroSLIs = this.sliCollector.calculateCurrentSLIs(5); // 5-minute window
      const failureTaxonomy = this.failureTracker.calculateTaxonomySnapshot(5);
      const languageSlices = this.languageMonitor.calculateLanguageSlices(5);
      
      // Update dashboard state
      this.dashboardData = {
        last_update: new Date(),
        hero_slis: heroSLIs,
        failure_taxonomy: failureTaxonomy,
        language_slices: languageSlices,
        active_alerts: this.alertManager.getStats().active_alerts,
        system_health: this.calculateSystemHealth(heroSLIs, failureTaxonomy)
      };
      
      // Check alerts
      this.alertManager.checkSLIAlerts(heroSLIs);
      this.alertManager.checkFailureAlerts(failureTaxonomy);
      
      span.setAttributes({
        success: true,
        system_health: this.dashboardData.system_health,
        total_queries: heroSLIs.total_queries,
        upshift_rate: heroSLIs.router_upshift_rate,
        failure_rate: failureTaxonomy.overall_failure_rate_pct
      });
      
    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      console.error('❌ Dashboard update failed:', error);
    } finally {
      span.end();
    }
  }
  
  /**
   * Calculate overall system health
   */
  private calculateSystemHealth(
    slis: HeroSLISnapshot,
    failures: FailureTaxonomySnapshot
  ): 'healthy' | 'degraded' | 'critical' {
    // Check for critical breaches
    const criticalBreaches = Object.values(slis.breaches).filter(b => b === 'critical').length;
    if (criticalBreaches > 0 || failures.overall_failure_rate_pct > 10) {
      return 'critical';
    }
    
    // Check for warning breaches
    const warningBreaches = Object.values(slis.breaches).filter(b => b === 'warning').length;
    if (warningBreaches > 1 || failures.overall_failure_rate_pct > 5) {
      return 'degraded';
    }
    
    return 'healthy';
  }
  
  /**
   * Get real-time dashboard data
   */
  getDashboardData() {
    return {
      ...this.dashboardData,
      production_mode: this.isProduction
    };
  }
  
  /**
   * Get historical data for charts
   */
  getHistoricalData(hours = 24) {
    return {
      hero_slis: this.sliCollector.getSLIHistory(hours),
      failure_taxonomy: this.failureTracker.getTaxonomyHistory(hours),
      recent_failures: this.failureTracker.getRecentFailures(1),
      alerts: this.alertManager.getStats()
    };
  }
  
  /**
   * Get comprehensive system statistics
   */
  getStats() {
    return {
      production_mode: this.isProduction,
      dashboard: this.dashboardData,
      components: {
        sli_collector: this.sliCollector.getStats(),
        failure_tracker: this.failureTracker.getStats(),
        language_monitor: this.languageMonitor.getStats(),
        alert_manager: this.alertManager.getStats()
      }
    };
  }
  
  /**
   * Generate monitoring report for stakeholders
   */
  generateMonitoringReport(): {
    summary: {
      system_health: string;
      uptime_pct: number;
      avg_latency_ms: number;
      success_rate_pct: number;
    };
    hero_slis: HeroSLISnapshot;
    failure_breakdown: FailureTaxonomySnapshot;
    top_issues: string[];
    recommendations: string[];
  } {
    const slis = this.dashboardData.hero_slis!;
    const failures = this.dashboardData.failure_taxonomy!;
    
    const successRate = ((slis.successful_queries / slis.total_queries) || 1) * 100;
    const uptime = 100 - failures.overall_failure_rate_pct;
    
    const topIssues: string[] = [];
    const recommendations: string[] = [];
    
    // Identify top issues
    if (failures.z0_rate_pct > 5) {
      topIssues.push(`High zero-result rate: ${failures.z0_rate_pct.toFixed(1)}%`);
      recommendations.push('Review index quality and query understanding');
    }
    
    if (slis.p95_latency_ms > 20) {
      topIssues.push(`P95 latency above target: ${slis.p95_latency_ms.toFixed(1)}ms`);
      recommendations.push('Investigate performance bottlenecks and optimize hot paths');
    }
    
    if (slis.router_upshift_rate > 0.08) {
      topIssues.push(`High upshift rate: ${(slis.router_upshift_rate * 100).toFixed(1)}%`);
      recommendations.push('Review routing thresholds and 256d model quality');
    }
    
    return {
      summary: {
        system_health: this.dashboardData.system_health,
        uptime_pct: uptime,
        avg_latency_ms: slis.p95_latency_ms,
        success_rate_pct: successRate
      },
      hero_slis: slis,
      failure_breakdown: failures,
      top_issues: topIssues,
      recommendations: recommendations
    };
  }
}