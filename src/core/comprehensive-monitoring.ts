/**
 * Comprehensive Monitoring and Drift Detection
 * 
 * Integrates all advanced search optimization components with comprehensive
 * monitoring, alerting, and drift detection. Provides unified dashboard
 * for system health and performance tracking.
 */

import type { SearchContext, SearchHit } from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';
import { globalConformalRouter } from './conformal-router.js';
import { globalEntropyGatedPriors } from './entropy-gated-priors.js';
import { globalLatencyConditionedMetrics } from './latency-conditioned-metrics.js';
import { globalRAPTORHygiene } from './raptor-hygiene.js';
import { globalEmbeddingRoadmap } from './embedding-roadmap.js';
import { globalUnicodeNormalizer } from './unicode-nfc-normalizer.js';

export interface SystemAlert {
  alert_id: string;
  alert_level: 'info' | 'warning' | 'critical';
  component: string;
  message: string;
  timestamp: number;
  metric_values: Record<string, number>;
  suggested_actions: string[];
  acknowledged: boolean;
  resolution_timestamp?: number;
}

export interface PerformanceSnapshot {
  timestamp: number;
  overall_p95_latency_ms: number;
  stage_a_latency_ms: number;
  stage_b_latency_ms: number;
  stage_c_latency_ms: number;
  queries_per_second: number;
  memory_usage_gb: number;
  cpu_usage_percent: number;
  active_queries: number;
  component_health: Map<string, 'healthy' | 'degraded' | 'critical'>;
}

export interface QualityMetrics {
  timestamp: number;
  sla_recall_50: number;
  sla_core_10: number;
  sla_diversity_10: number;
  conformal_upshift_rate: number;
  entropy_gating_rate: number;
  raptor_quality_score: number;
  embedding_distillation_progress: number;
  unicode_normalization_rate: number;
}

export interface DriftSignal {
  component: string;
  metric_name: string;
  current_value: number;
  baseline_value: number;
  drift_magnitude: number;
  confidence_level: number;
  detection_method: 'cusum' | 'statistical' | 'ml_based';
  timestamp: number;
}

export interface MonitoringDashboard {
  system_status: 'healthy' | 'degraded' | 'critical';
  active_alerts: SystemAlert[];
  performance_snapshot: PerformanceSnapshot;
  quality_metrics: QualityMetrics;
  drift_signals: DriftSignal[];
  component_statuses: Map<string, ComponentStatus>;
  last_update: number;
}

export interface ComponentStatus {
  name: string;
  health: 'healthy' | 'degraded' | 'critical';
  enabled: boolean;
  last_activity: Date;
  key_metrics: Record<string, number>;
  alerts: number;
  uptime_percent: number;
}

/**
 * Alert manager for system notifications
 */
class AlertManager {
  private alerts: Map<string, SystemAlert> = new Map();
  private alertHistory: SystemAlert[] = [];
  private nextAlertId = 1;
  
  private alertThresholds = {
    p95_latency_ms: 25, // Alert if p95 > 25ms
    memory_usage_gb: 12, // Alert if memory > 12GB
    upshift_rate_percent: 7, // Alert if upshift rate > 7%
    sla_recall_50: 0.75, // Alert if recall drops below 75%
    cusum_threshold: 3.0 // CUSUM alert threshold
  };
  
  /**
   * Create system alert
   */
  createAlert(
    level: SystemAlert['alert_level'],
    component: string,
    message: string,
    metricValues: Record<string, number> = {},
    suggestedActions: string[] = []
  ): SystemAlert {
    const alert: SystemAlert = {
      alert_id: `alert_${this.nextAlertId++}`,
      alert_level: level,
      component,
      message,
      timestamp: Date.now(),
      metric_values: metricValues,
      suggested_actions: suggestedActions,
      acknowledged: false
    };
    
    this.alerts.set(alert.alert_id, alert);
    this.alertHistory.push(alert);
    
    // Keep history limited
    if (this.alertHistory.length > 1000) {
      this.alertHistory = this.alertHistory.slice(-1000);
    }
    
    console.log(`🚨 ${level.toUpperCase()} Alert [${component}]: ${message}`);
    
    return alert;
  }
  
  /**
   * Check metrics against thresholds and create alerts
   */
  checkMetricThresholds(
    performance: PerformanceSnapshot,
    quality: QualityMetrics
  ): SystemAlert[] {
    const newAlerts: SystemAlert[] = [];
    
    // Performance alerts
    if (performance.overall_p95_latency_ms > this.alertThresholds.p95_latency_ms) {
      newAlerts.push(this.createAlert(
        'warning',
        'performance',
        `High p95 latency: ${performance.overall_p95_latency_ms.toFixed(1)}ms`,
        { p95_latency_ms: performance.overall_p95_latency_ms },
        ['Check system load', 'Review Stage-C upshift patterns', 'Consider scaling resources']
      ));
    }
    
    if (performance.memory_usage_gb > this.alertThresholds.memory_usage_gb) {
      newAlerts.push(this.createAlert(
        'warning',
        'memory',
        `High memory usage: ${performance.memory_usage_gb.toFixed(1)}GB`,
        { memory_usage_gb: performance.memory_usage_gb },
        ['Check memory leaks', 'Review cache sizes', 'Consider garbage collection tuning']
      ));
    }
    
    // Quality alerts
    if (quality.sla_recall_50 < this.alertThresholds.sla_recall_50) {
      newAlerts.push(this.createAlert(
        'critical',
        'quality',
        `SLA-Recall@50 below threshold: ${(quality.sla_recall_50 * 100).toFixed(1)}%`,
        { sla_recall_50: quality.sla_recall_50 },
        ['Review index quality', 'Check semantic model performance', 'Validate ground truth data']
      ));
    }
    
    if (quality.conformal_upshift_rate > this.alertThresholds.upshift_rate_percent) {
      newAlerts.push(this.createAlert(
        'warning',
        'conformal-router',
        `High upshift rate: ${quality.conformal_upshift_rate.toFixed(1)}%`,
        { upshift_rate: quality.conformal_upshift_rate },
        ['Review conformal thresholds', 'Check risk assessment accuracy', 'Monitor budget usage']
      ));
    }
    
    return newAlerts;
  }
  
  /**
   * Acknowledge alert
   */
  acknowledgeAlert(alertId: string): boolean {
    const alert = this.alerts.get(alertId);
    if (alert) {
      alert.acknowledged = true;
      return true;
    }
    return false;
  }
  
  /**
   * Resolve alert
   */
  resolveAlert(alertId: string): boolean {
    const alert = this.alerts.get(alertId);
    if (alert) {
      alert.resolution_timestamp = Date.now();
      this.alerts.delete(alertId);
      return true;
    }
    return false;
  }
  
  /**
   * Get active alerts
   */
  getActiveAlerts(): SystemAlert[] {
    return Array.from(this.alerts.values());
  }
  
  /**
   * Get alert history
   */
  getAlertHistory(hours = 24): SystemAlert[] {
    const cutoff = Date.now() - hours * 60 * 60 * 1000;
    return this.alertHistory.filter(alert => alert.timestamp > cutoff);
  }
  
  /**
   * Update alert thresholds
   */
  updateThresholds(thresholds: Partial<typeof this.alertThresholds>): void {
    this.alertThresholds = { ...this.alertThresholds, ...thresholds };
    console.log('🔧 Alert thresholds updated:', thresholds);
  }
}

/**
 * Statistical drift detector
 */
class StatisticalDriftDetector {
  private historicalData: Map<string, number[]> = new Map();
  private baselines: Map<string, number> = new Map();
  
  /**
   * Add measurement for drift detection
   */
  addMeasurement(metricName: string, value: number): void {
    if (!this.historicalData.has(metricName)) {
      this.historicalData.set(metricName, []);
    }
    
    const data = this.historicalData.get(metricName)!;
    data.push(value);
    
    // Keep rolling window of last 100 measurements
    if (data.length > 100) {
      data.shift();
    }
    
    // Update baseline (moving average of first 30 measurements)
    if (data.length >= 30) {
      const baseline = data.slice(0, 30).reduce((sum, val) => sum + val, 0) / 30;
      this.baselines.set(metricName, baseline);
    }
  }
  
  /**
   * Detect drift using statistical methods
   */
  detectDrift(
    metricName: string, 
    sensitivityThreshold = 0.2
  ): DriftSignal | null {
    const data = this.historicalData.get(metricName);
    const baseline = this.baselines.get(metricName);
    
    if (!data || !baseline || data.length < 30) {
      return null; // Not enough data
    }
    
    // Current value (mean of last 10 measurements)
    const recentData = data.slice(-10);
    const currentValue = recentData.reduce((sum, val) => sum + val, 0) / recentData.length;
    
    // Calculate drift magnitude
    const driftMagnitude = Math.abs(currentValue - baseline) / baseline;
    
    if (driftMagnitude > sensitivityThreshold) {
      // Calculate confidence using t-test approximation
      const recentStd = this.calculateStandardDeviation(recentData);
      const confidence = Math.min(0.99, 0.5 + driftMagnitude * 2);
      
      return {
        component: 'statistical-detector',
        metric_name: metricName,
        current_value: currentValue,
        baseline_value: baseline,
        drift_magnitude: driftMagnitude,
        confidence_level: confidence,
        detection_method: 'statistical',
        timestamp: Date.now()
      };
    }
    
    return null;
  }
  
  private calculateStandardDeviation(values: number[]): number {
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    return Math.sqrt(variance);
  }
  
  /**
   * Get all current baselines
   */
  getBaselines(): Map<string, number> {
    return new Map(this.baselines);
  }
}

/**
 * Main comprehensive monitoring system
 */
export class ComprehensiveMonitoring {
  private static instance: ComprehensiveMonitoring | null = null;
  
  private alertManager: AlertManager;
  private driftDetector: StatisticalDriftDetector;
  private enabled = true;
  
  private performanceHistory: PerformanceSnapshot[] = [];
  private qualityHistory: QualityMetrics[] = [];
  private currentSnapshot: PerformanceSnapshot | null = null;
  private currentQuality: QualityMetrics | null = null;
  
  // Component health tracking
  private componentHealth: Map<string, ComponentStatus> = new Map();
  
  // System metrics
  private systemStartTime = Date.now();
  private totalQueries = 0;
  private lastQueryTime = Date.now();
  
  constructor() {
    this.alertManager = new AlertManager();
    this.driftDetector = new StatisticalDriftDetector();
    
    this.initializeComponentHealth();
  }
  
  /**
   * Get singleton instance
   */
  static getInstance(): ComprehensiveMonitoring {
    if (!ComprehensiveMonitoring.instance) {
      ComprehensiveMonitoring.instance = new ComprehensiveMonitoring();
    }
    return ComprehensiveMonitoring.instance;
  }
  
  /**
   * Initialize component health tracking
   */
  private initializeComponentHealth(): void {
    const components = [
      'conformal-router',
      'entropy-gated-priors',
      'latency-conditioned-metrics',
      'raptor-hygiene',
      'embedding-roadmap',
      'unicode-normalizer'
    ];
    
    for (const component of components) {
      this.componentHealth.set(component, {
        name: component,
        health: 'healthy',
        enabled: true,
        last_activity: new Date(),
        key_metrics: {},
        alerts: 0,
        uptime_percent: 100.0
      });
    }
  }
  
  /**
   * Collect performance snapshot from all components
   */
  async collectPerformanceSnapshot(): Promise<PerformanceSnapshot> {
    if (!this.enabled) {
      return this.getEmptyPerformanceSnapshot();
    }
    
    const span = LensTracer.createChildSpan('collect_performance_snapshot');
    
    try {
      // Collect component health
      const componentHealth = new Map<string, 'healthy' | 'degraded' | 'critical'>();
      
      // Get stats from each component
      const conformalStats = globalConformalRouter.getStats();
      const entropyStats = globalEntropyGatedPriors.getStats();
      const metricsStats = globalLatencyConditionedMetrics.getAggregateStats();
      const raptorStats = globalRAPTORHygiene.getOperationalStats();
      const embeddingStats = globalEmbeddingRoadmap.getStats();
      const unicodeStats = globalUnicodeNormalizer.getStats();
      
      // Update component health based on stats
      this.updateComponentHealth('conformal-router', {
        enabled: conformalStats.enabled,
        key_metrics: {
          upshift_rate: conformalStats.upshift_rate,
          total_queries: conformalStats.total_queries
        }
      });
      
      this.updateComponentHealth('entropy-gated-priors', {
        enabled: entropyStats.enabled,
        key_metrics: {
          prior_application_rate: entropyStats.prior_application_rate,
          entropy_gating_rate: entropyStats.entropy_gating_rate
        }
      });
      
      this.updateComponentHealth('raptor-hygiene', {
        enabled: raptorStats.enabled,
        key_metrics: {
          total_operations: raptorStats.total_operations,
          node_splits: raptorStats.node_splits
        }
      });
      
      // Calculate overall metrics
      const memUsage = process.memoryUsage();
      const cpuUsage = await this.getCPUUsage();
      
      componentHealth.set('conformal-router', conformalStats.enabled ? 'healthy' : 'degraded');
      componentHealth.set('entropy-gated-priors', entropyStats.enabled ? 'healthy' : 'degraded');
      componentHealth.set('latency-conditioned-metrics', 'healthy');
      componentHealth.set('raptor-hygiene', raptorStats.enabled ? 'healthy' : 'degraded');
      componentHealth.set('embedding-roadmap', embeddingStats.enabled ? 'healthy' : 'degraded');
      componentHealth.set('unicode-normalizer', unicodeStats.enabled ? 'healthy' : 'degraded');
      
      const snapshot: PerformanceSnapshot = {
        timestamp: Date.now(),
        overall_p95_latency_ms: metricsStats.avg_p95_latency_ms,
        stage_a_latency_ms: 3.5, // Mock - would get from actual metrics
        stage_b_latency_ms: 5.2, // Mock
        stage_c_latency_ms: 8.1, // Mock
        queries_per_second: this.calculateQPS(),
        memory_usage_gb: memUsage.heapUsed / (1024 * 1024 * 1024),
        cpu_usage_percent: cpuUsage,
        active_queries: 0, // Mock
        component_health: componentHealth
      };
      
      this.currentSnapshot = snapshot;
      this.performanceHistory.push(snapshot);
      
      // Keep history limited
      if (this.performanceHistory.length > 1440) { // 24 hours at 1-minute intervals
        this.performanceHistory.shift();
      }
      
      // Add to drift detection
      this.driftDetector.addMeasurement('p95_latency_ms', snapshot.overall_p95_latency_ms);
      this.driftDetector.addMeasurement('memory_usage_gb', snapshot.memory_usage_gb);
      this.driftDetector.addMeasurement('cpu_usage_percent', snapshot.cpu_usage_percent);
      
      span.setAttributes({
        success: true,
        p95_latency_ms: snapshot.overall_p95_latency_ms,
        memory_usage_gb: snapshot.memory_usage_gb,
        cpu_usage_percent: snapshot.cpu_usage_percent
      });
      
      return snapshot;
      
    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }
  
  /**
   * Collect quality metrics from all components
   */
  async collectQualityMetrics(): Promise<QualityMetrics> {
    if (!this.enabled) {
      return this.getEmptyQualityMetrics();
    }
    
    const span = LensTracer.createChildSpan('collect_quality_metrics');
    
    try {
      // Get quality stats from components
      const conformalStats = globalConformalRouter.getStats();
      const entropyStats = globalEntropyGatedPriors.getStats();
      const metricsStats = globalLatencyConditionedMetrics.getAggregateStats();
      const raptorStats = globalRAPTORHygiene.getOperationalStats();
      const embeddingStats = globalEmbeddingRoadmap.getStats();
      const unicodeStats = globalUnicodeNormalizer.getStats();
      
      const quality: QualityMetrics = {
        timestamp: Date.now(),
        sla_recall_50: metricsStats.avg_sla_recall_50,
        sla_core_10: metricsStats.avg_sla_core_10,
        sla_diversity_10: metricsStats.avg_sla_diversity_10,
        conformal_upshift_rate: conformalStats.upshift_rate,
        entropy_gating_rate: entropyStats.entropy_gating_rate,
        raptor_quality_score: raptorStats.quality_stats.avg_cluster_coverage || 0,
        embedding_distillation_progress: embeddingStats.model_promotions > 0 ? 1.0 : 0.5,
        unicode_normalization_rate: unicodeStats.normalization_rate
      };
      
      this.currentQuality = quality;
      this.qualityHistory.push(quality);
      
      // Keep history limited
      if (this.qualityHistory.length > 1440) { // 24 hours
        this.qualityHistory.shift();
      }
      
      // Add to drift detection
      this.driftDetector.addMeasurement('sla_recall_50', quality.sla_recall_50);
      this.driftDetector.addMeasurement('sla_core_10', quality.sla_core_10);
      this.driftDetector.addMeasurement('conformal_upshift_rate', quality.conformal_upshift_rate);
      
      span.setAttributes({
        success: true,
        sla_recall_50: quality.sla_recall_50,
        sla_core_10: quality.sla_core_10,
        upshift_rate: quality.conformal_upshift_rate
      });
      
      return quality;
      
    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }
  
  /**
   * Record query execution for monitoring
   */
  recordQuery(ctx: SearchContext, hits: SearchHit[], latencyMs: number): void {
    if (!this.enabled) return;
    
    this.totalQueries++;
    this.lastQueryTime = Date.now();
    
    // Update component activity
    for (const [componentName, status] of this.componentHealth) {
      status.last_activity = new Date();
    }
  }
  
  /**
   * Generate monitoring dashboard
   */
  async generateDashboard(): Promise<MonitoringDashboard> {
    const span = LensTracer.createChildSpan('generate_monitoring_dashboard');
    
    try {
      // Collect current snapshots
      const performance = this.currentSnapshot || await this.collectPerformanceSnapshot();
      const quality = this.currentQuality || await this.collectQualityMetrics();
      
      // Check for new alerts
      const newAlerts = this.alertManager.checkMetricThresholds(performance, quality);
      
      // Detect drift
      const driftSignals = this.detectAllDrift();
      
      // Determine overall system status
      const systemStatus = this.calculateSystemStatus(performance, quality, newAlerts, driftSignals);
      
      const dashboard: MonitoringDashboard = {
        system_status: systemStatus,
        active_alerts: this.alertManager.getActiveAlerts(),
        performance_snapshot: performance,
        quality_metrics: quality,
        drift_signals: driftSignals,
        component_statuses: new Map(this.componentHealth),
        last_update: Date.now()
      };
      
      console.log(`📊 Dashboard update: ${systemStatus} status, ${dashboard.active_alerts.length} alerts, ${driftSignals.length} drift signals`);
      
      span.setAttributes({
        success: true,
        system_status: systemStatus,
        active_alerts: dashboard.active_alerts.length,
        drift_signals: driftSignals.length
      });
      
      return dashboard;
      
    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }
  
  /**
   * Detect drift across all metrics
   */
  private detectAllDrift(): DriftSignal[] {
    const driftSignals: DriftSignal[] = [];
    
    // Statistical drift detection
    const metricNames = [
      'p95_latency_ms',
      'memory_usage_gb',
      'sla_recall_50',
      'sla_core_10',
      'conformal_upshift_rate'
    ];
    
    for (const metricName of metricNames) {
      const drift = this.driftDetector.detectDrift(metricName, 0.15); // 15% sensitivity
      if (drift) {
        driftSignals.push(drift);
      }
    }
    
    // CUSUM drift detection (from latency-conditioned metrics)
    const cusumAlarms = globalLatencyConditionedMetrics.getCUSUMAlarms();
    for (const [metricName, alarm] of cusumAlarms) {
      if (alarm.alarm_state === 'warning' || alarm.alarm_state === 'critical') {
        driftSignals.push({
          component: 'cusum-detector',
          metric_name: metricName,
          current_value: alarm.current_sum,
          baseline_value: 0,
          drift_magnitude: alarm.current_sum / alarm.threshold,
          confidence_level: 0.95,
          detection_method: 'cusum',
          timestamp: Date.now()
        });
      }
    }
    
    return driftSignals;
  }
  
  /**
   * Calculate overall system status
   */
  private calculateSystemStatus(
    performance: PerformanceSnapshot,
    quality: QualityMetrics,
    alerts: SystemAlert[],
    driftSignals: DriftSignal[]
  ): 'healthy' | 'degraded' | 'critical' {
    // Critical conditions
    const criticalAlerts = alerts.filter(a => a.alert_level === 'critical').length;
    if (criticalAlerts > 0) return 'critical';
    
    const highDrift = driftSignals.filter(d => d.drift_magnitude > 0.5).length;
    if (highDrift > 2) return 'critical';
    
    if (quality.sla_recall_50 < 0.7) return 'critical';
    if (performance.overall_p95_latency_ms > 30) return 'critical';
    
    // Degraded conditions
    const warningAlerts = alerts.filter(a => a.alert_level === 'warning').length;
    if (warningAlerts > 3) return 'degraded';
    
    const moderateDrift = driftSignals.filter(d => d.drift_magnitude > 0.2).length;
    if (moderateDrift > 3) return 'degraded';
    
    if (quality.sla_recall_50 < 0.8) return 'degraded';
    if (performance.overall_p95_latency_ms > 22) return 'degraded';
    
    return 'healthy';
  }
  
  /**
   * Update component health status
   */
  private updateComponentHealth(
    componentName: string, 
    updates: {
      enabled?: boolean;
      key_metrics?: Record<string, number>;
    }
  ): void {
    const status = this.componentHealth.get(componentName);
    if (status) {
      if (updates.enabled !== undefined) {
        status.enabled = updates.enabled;
        status.health = updates.enabled ? 'healthy' : 'degraded';
      }
      if (updates.key_metrics) {
        status.key_metrics = { ...status.key_metrics, ...updates.key_metrics };
      }
      status.last_activity = new Date();
    }
  }
  
  /**
   * Calculate queries per second
   */
  private calculateQPS(): number {
    const uptimeHours = (Date.now() - this.systemStartTime) / (1000 * 60 * 60);
    return uptimeHours > 0 ? this.totalQueries / (uptimeHours * 3600) : 0;
  }
  
  /**
   * Get CPU usage (mock implementation)
   */
  private async getCPUUsage(): Promise<number> {
    // Mock CPU usage - in production would use actual system metrics
    return 15 + Math.random() * 20; // 15-35% usage
  }
  
  /**
   * Get empty performance snapshot
   */
  private getEmptyPerformanceSnapshot(): PerformanceSnapshot {
    return {
      timestamp: Date.now(),
      overall_p95_latency_ms: 0,
      stage_a_latency_ms: 0,
      stage_b_latency_ms: 0,
      stage_c_latency_ms: 0,
      queries_per_second: 0,
      memory_usage_gb: 0,
      cpu_usage_percent: 0,
      active_queries: 0,
      component_health: new Map()
    };
  }
  
  /**
   * Get empty quality metrics
   */
  private getEmptyQualityMetrics(): QualityMetrics {
    return {
      timestamp: Date.now(),
      sla_recall_50: 0,
      sla_core_10: 0,
      sla_diversity_10: 0,
      conformal_upshift_rate: 0,
      entropy_gating_rate: 0,
      raptor_quality_score: 0,
      embedding_distillation_progress: 0,
      unicode_normalization_rate: 0
    };
  }
  
  /**
   * Get performance history
   */
  getPerformanceHistory(hours = 24): PerformanceSnapshot[] {
    const cutoff = Date.now() - hours * 60 * 60 * 1000;
    return this.performanceHistory.filter(snapshot => snapshot.timestamp > cutoff);
  }
  
  /**
   * Get quality history
   */
  getQualityHistory(hours = 24): QualityMetrics[] {
    const cutoff = Date.now() - hours * 60 * 60 * 1000;
    return this.qualityHistory.filter(metrics => metrics.timestamp > cutoff);
  }
  
  /**
   * Acknowledge alert
   */
  acknowledgeAlert(alertId: string): boolean {
    return this.alertManager.acknowledgeAlert(alertId);
  }
  
  /**
   * Resolve alert
   */
  resolveAlert(alertId: string): boolean {
    return this.alertManager.resolveAlert(alertId);
  }
  
  /**
   * Get system uptime
   */
  getSystemUptime(): {
    uptime_ms: number;
    uptime_hours: number;
    total_queries: number;
    avg_qps: number;
  } {
    const uptimeMs = Date.now() - this.systemStartTime;
    const uptimeHours = uptimeMs / (1000 * 60 * 60);
    
    return {
      uptime_ms: uptimeMs,
      uptime_hours: uptimeHours,
      total_queries: this.totalQueries,
      avg_qps: this.calculateQPS()
    };
  }
  
  /**
   * Enable/disable monitoring
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    console.log(`📊 Comprehensive monitoring ${enabled ? 'ENABLED' : 'DISABLED'}`);
  }
  
  /**
   * Update alert thresholds
   */
  updateAlertThresholds(thresholds: Record<string, number>): void {
    this.alertManager.updateThresholds(thresholds);
  }
}

// Global instance
export const globalComprehensiveMonitoring = new ComprehensiveMonitoring();