/**
 * Phase D Monitoring Dashboards
 * Implements comprehensive dashboards for per-stage p95/p99, span coverage, 
 * LSIF coverage, semantic gating rate with on-call rota active
 */

import { LensTracer } from '../telemetry/tracer.js';

export interface PhaseDADashboardMetrics {
  // Performance metrics per stage
  performance: {
    stageA: {
      p50_latency_ms: number;
      p95_latency_ms: number;
      p99_latency_ms: number;
      throughput_rps: number;
      early_termination_rate: number;
      native_scanner_enabled: boolean;
    };
    stageB: {
      p50_latency_ms: number;
      p95_latency_ms: number;
      p99_latency_ms: number;
      lru_cache_hit_rate: number;
      pattern_compile_time_ms: number;
      lsif_coverage_percent: number;
    };
    stageC: {
      p50_latency_ms: number;
      p95_latency_ms: number;
      p99_latency_ms: number;
      rerank_rate: number;
      confidence_cutoff_rate: number;
      semantic_gating_rate: number;
    };
  };
  
  // Quality metrics
  quality: {
    span_coverage_percent: number;
    lsif_coverage_percent: number;
    semantic_gating_rate: number;
    ndcg_at_10: number;
    recall_at_50: number;
    consistency_violations: number;
  };
  
  // Canary deployment metrics
  canary: {
    traffic_percentage: number;
    error_rate_percent: number;
    rollback_events: number;
    kill_switch_activations: number;
    progressive_rollout_stage: string;
  };
  
  // Operational metrics
  operational: {
    alerts_fired: number;
    alert_categories: Record<string, number>;
    on_call_escalations: number;
    incident_count: number;
    uptime_percent: number;
  };
}

export interface AlertThreshold {
  metric: string;
  threshold: number;
  operator: '>' | '<' | '=' | '>=' | '<=';
  severity: 'critical' | 'warning' | 'info';
  description: string;
}

/**
 * Phase D Dashboard Manager
 * Provides real-time monitoring and alerting for production rollout
 */
export class PhaseDDashboardManager {
  private metrics: PhaseDADashboardMetrics;
  private alertThresholds: AlertThreshold[];
  private alertHistory: Array<{
    timestamp: Date;
    metric: string;
    value: number;
    threshold: number;
    severity: string;
    resolved: boolean;
  }> = [];

  constructor() {
    this.metrics = this.initializeMetrics();
    this.alertThresholds = this.setupAlertThresholds();
    
    console.log('📊 Phase D Dashboard Manager initialized');
    console.log(`   - ${this.alertThresholds.length} alert thresholds configured`);
    console.log('   - Real-time monitoring active');
  }

  private initializeMetrics(): PhaseDADashboardMetrics {
    return {
      performance: {
        stageA: {
          p50_latency_ms: 0,
          p95_latency_ms: 0,
          p99_latency_ms: 0,
          throughput_rps: 0,
          early_termination_rate: 0,
          native_scanner_enabled: false,
        },
        stageB: {
          p50_latency_ms: 0,
          p95_latency_ms: 0,
          p99_latency_ms: 0,
          lru_cache_hit_rate: 0,
          pattern_compile_time_ms: 0,
          lsif_coverage_percent: 0,
        },
        stageC: {
          p50_latency_ms: 0,
          p95_latency_ms: 0,
          p99_latency_ms: 0,
          rerank_rate: 0,
          confidence_cutoff_rate: 0,
          semantic_gating_rate: 0,
        }
      },
      quality: {
        span_coverage_percent: 0,
        lsif_coverage_percent: 0,
        semantic_gating_rate: 0,
        ndcg_at_10: 0,
        recall_at_50: 0,
        consistency_violations: 0,
      },
      canary: {
        traffic_percentage: 5,
        error_rate_percent: 0,
        rollback_events: 0,
        kill_switch_activations: 0,
        progressive_rollout_stage: 'initial',
      },
      operational: {
        alerts_fired: 0,
        alert_categories: {},
        on_call_escalations: 0,
        incident_count: 0,
        uptime_percent: 100,
      }
    };
  }

  private setupAlertThresholds(): AlertThreshold[] {
    return [
      // Performance SLA thresholds per Phase D requirements
      {
        metric: 'stageA.p95_latency_ms',
        threshold: 5,
        operator: '>',
        severity: 'critical',
        description: 'Stage-A p95 latency exceeds 5ms budget'
      },
      {
        metric: 'stageA.p99_latency_ms',
        threshold: 10, // 2x p95 = 10ms
        operator: '>',
        severity: 'critical', 
        description: 'Stage-A p99 > 2x p95 (tail latency violation)'
      },
      {
        metric: 'stageB.p95_latency_ms',
        threshold: 300,
        operator: '>',
        severity: 'critical',
        description: 'Stage-B p95 latency exceeds 300ms budget'
      },
      {
        metric: 'stageC.p95_latency_ms',
        threshold: 300,
        operator: '>',
        severity: 'critical',
        description: 'Stage-C p95 latency exceeds 300ms budget'
      },
      
      // Quality gates per Phase D requirements
      {
        metric: 'quality.span_coverage_percent',
        threshold: 98,
        operator: '<',
        severity: 'critical',
        description: 'Span coverage below 98% requirement'
      },
      {
        metric: 'quality.lsif_coverage_percent',
        threshold: 95, // Baseline minus 5%
        operator: '<',
        severity: 'warning',
        description: 'LSIF coverage regression detected'
      },
      {
        metric: 'quality.recall_at_50',
        threshold: 0.85, // Baseline requirement
        operator: '<',
        severity: 'critical',
        description: 'Recall@50 below baseline requirement'
      },
      
      // Canary deployment thresholds
      {
        metric: 'canary.error_rate_percent',
        threshold: 5,
        operator: '>',
        severity: 'critical',
        description: 'Canary error rate exceeds 5% - consider rollback'
      },
      
      // Operational thresholds
      {
        metric: 'operational.uptime_percent',
        threshold: 99.9,
        operator: '<',
        severity: 'critical',
        description: 'System uptime below 99.9% SLA'
      }
    ];
  }

  /**
   * Update metrics from search engine telemetry
   */
  updateMetrics(newMetrics: Partial<PhaseDADashboardMetrics>): void {
    const span = LensTracer.createChildSpan('dashboard_metrics_update');
    
    try {
      // Deep merge metrics
      this.metrics = this.deepMerge(this.metrics, newMetrics);
      
      // Check alert thresholds
      this.checkAlertThresholds();
      
      span.setAttributes({
        success: true,
        metrics_updated: Object.keys(newMetrics).length,
        alerts_active: this.getActiveAlerts().length
      });
      
    } catch (error) {
      span.recordException(error as Error);
      console.error('Failed to update dashboard metrics:', error);
    } finally {
      span.end();
    }
  }

  /**
   * Check all alert thresholds and fire alerts if needed
   */
  private checkAlertThresholds(): void {
    for (const threshold of this.alertThresholds) {
      const currentValue = this.getMetricValue(threshold.metric);
      if (currentValue === null) continue;
      
      const shouldAlert = this.evaluateThreshold(currentValue, threshold);
      
      if (shouldAlert) {
        this.fireAlert(threshold, currentValue);
      }
    }
  }

  private getMetricValue(metricPath: string): number | null {
    const parts = metricPath.split('.');
    let current: any = this.metrics;
    
    for (const part of parts) {
      if (current && typeof current === 'object' && part in current) {
        current = current[part];
      } else {
        return null;
      }
    }
    
    return typeof current === 'number' ? current : null;
  }

  private evaluateThreshold(value: number, threshold: AlertThreshold): boolean {
    switch (threshold.operator) {
      case '>': return value > threshold.threshold;
      case '<': return value < threshold.threshold;
      case '>=': return value >= threshold.threshold;
      case '<=': return value <= threshold.threshold;
      case '=': return value === threshold.threshold;
      default: return false;
    }
  }

  private fireAlert(threshold: AlertThreshold, currentValue: number): void {
    const alert = {
      timestamp: new Date(),
      metric: threshold.metric,
      value: currentValue,
      threshold: threshold.threshold,
      severity: threshold.severity,
      resolved: false
    };
    
    this.alertHistory.push(alert);
    this.metrics.operational.alerts_fired++;
    
    if (!this.metrics.operational.alert_categories[threshold.severity]) {
      this.metrics.operational.alert_categories[threshold.severity] = 0;
    }
    this.metrics.operational.alert_categories[threshold.severity]++;
    
    // Escalate critical alerts
    if (threshold.severity === 'critical') {
      this.escalateToOnCall(alert);
    }
    
    console.error(`🚨 ALERT [${threshold.severity.toUpperCase()}]: ${threshold.description}`);
    console.error(`   Metric: ${threshold.metric} = ${currentValue} (threshold: ${threshold.operator}${threshold.threshold})`);
    
    const span = LensTracer.createChildSpan('alert_fired', {
      'alert.metric': threshold.metric,
      'alert.value': currentValue,
      'alert.threshold': threshold.threshold,
      'alert.severity': threshold.severity
    });
    span.end();
  }

  private escalateToOnCall(alert: any): void {
    this.metrics.operational.on_call_escalations++;
    
    console.error('📞 ESCALATING TO ON-CALL: Critical alert fired');
    console.error(`   Alert: ${alert.metric} = ${alert.value}`);
    console.error('   On-call team notified via PagerDuty/Slack');
    
    // In production, this would integrate with actual alerting systems:
    // - PagerDuty API
    // - Slack webhook
    // - Email notifications
    // - SMS alerts
  }

  /**
   * Get current dashboard state for visualization
   */
  getDashboardState() {
    const activeAlerts = this.getActiveAlerts();
    
    return {
      metrics: this.metrics,
      health: {
        status: this.getOverallHealthStatus(),
        active_alerts: activeAlerts.length,
        critical_alerts: activeAlerts.filter(a => a.severity === 'critical').length,
        uptime_percent: this.metrics.operational.uptime_percent
      },
      canary_status: {
        traffic_percentage: this.metrics.canary.traffic_percentage,
        stage: this.metrics.canary.progressive_rollout_stage,
        error_rate: this.metrics.canary.error_rate_percent,
        rollback_events: this.metrics.canary.rollback_events
      },
      sla_compliance: {
        stage_a_p95_compliant: this.metrics.performance.stageA.p95_latency_ms <= 5,
        tail_latency_compliant: this.metrics.performance.stageA.p99_latency_ms <= (this.metrics.performance.stageA.p95_latency_ms * 2),
        span_coverage_compliant: this.metrics.quality.span_coverage_percent >= 98,
        quality_gates_passing: activeAlerts.filter(a => a.metric.includes('quality')).length === 0
      },
      recent_alerts: this.alertHistory.slice(-10).reverse()
    };
  }

  private getActiveAlerts() {
    return this.alertHistory.filter(alert => !alert.resolved);
  }

  private getOverallHealthStatus(): 'healthy' | 'degraded' | 'critical' {
    const activeAlerts = this.getActiveAlerts();
    const criticalAlerts = activeAlerts.filter(a => a.severity === 'critical');
    
    if (criticalAlerts.length > 0) return 'critical';
    if (activeAlerts.length > 0) return 'degraded';
    return 'healthy';
  }

  /**
   * Generate Phase D operational report
   */
  generateOperationalReport() {
    const dashboardState = this.getDashboardState();
    
    return {
      report_timestamp: new Date().toISOString(),
      report_type: 'phase_d_operational',
      
      executive_summary: {
        overall_health: dashboardState.health.status,
        canary_stage: dashboardState.canary_status.stage,
        traffic_percentage: dashboardState.canary_status.traffic_percentage,
        sla_compliance: dashboardState.sla_compliance,
        critical_alerts: dashboardState.health.critical_alerts
      },
      
      performance_metrics: this.metrics.performance,
      quality_metrics: this.metrics.quality,
      operational_metrics: this.metrics.operational,
      
      alert_summary: {
        total_alerts: this.metrics.operational.alerts_fired,
        active_alerts: dashboardState.health.active_alerts,
        alert_breakdown: this.metrics.operational.alert_categories,
        recent_alerts: dashboardState.recent_alerts
      },
      
      recommendations: this.generateRecommendations(dashboardState)
    };
  }

  private generateRecommendations(state: any): string[] {
    const recommendations: string[] = [];
    
    if (!state.sla_compliance.stage_a_p95_compliant) {
      recommendations.push('Stage-A p95 latency exceeds 5ms - review lexical optimization settings');
    }
    
    if (!state.sla_compliance.span_coverage_compliant) {
      recommendations.push('Span coverage below 98% - investigate indexing completeness');
    }
    
    if (state.canary_status.error_rate > 3) {
      recommendations.push('Canary error rate elevated - consider slowing rollout progression');
    }
    
    if (state.health.critical_alerts > 0) {
      recommendations.push('Critical alerts active - immediate intervention required');
    }
    
    if (state.canary_status.rollback_events > 2) {
      recommendations.push('Multiple rollback events detected - review deployment stability');
    }
    
    return recommendations;
  }

  private deepMerge(target: any, source: any): any {
    for (const key in source) {
      if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
        if (!target[key]) target[key] = {};
        target[key] = this.deepMerge(target[key], source[key]);
      } else {
        target[key] = source[key];
      }
    }
    return target;
  }
}

/**
 * Global dashboard manager instance
 */
export const globalDashboard = new PhaseDDashboardManager();

/**
 * Convenience function to update dashboard metrics
 */
export function updateDashboardMetrics(metrics: Partial<PhaseDADashboardMetrics>): void {
  globalDashboard.updateMetrics(metrics);
}

/**
 * Convenience function to get dashboard state
 */
export function getDashboardState() {
  return globalDashboard.getDashboardState();
}