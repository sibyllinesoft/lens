/**
 * Tail-Latency Monitoring System - Phase D implementation
 * Monitors P99 ≤ 2× P95 validation and alerting for lens release quality gates
 */

import { EventEmitter } from 'events';

export interface LatencyMeasurement {
  timestamp: string;
  operation: string;
  slice: string;
  latency_ms: number;
  metadata: {
    repo_type: string;
    language: string;
    size_category: string;
    query_type: string;
    result_count: number;
  };
}

export interface LatencyMetrics {
  timestamp: string;
  slice: string;
  window_duration_ms: number;
  sample_count: number;
  percentiles: {
    p50: number;
    p90: number;
    p95: number;
    p99: number;
    p99_9: number;
  };
  tail_latency_ratio: number; // P99 / P95
  violation_threshold: number; // Max allowed P99/P95 ratio
  is_violation: boolean;
}

export interface TailLatencyAlert {
  id: string;
  timestamp: string;
  severity: 'warning' | 'error' | 'critical';
  slice: string;
  metrics: LatencyMetrics;
  violation_details: {
    current_ratio: number;
    threshold: number;
    violation_magnitude: number;
    consecutive_violations: number;
  };
  recommended_actions: string[];
}

export interface MonitoringConfig {
  measurement_window_ms: number;
  evaluation_frequency_ms: number;
  p99_p95_threshold: number; // Max allowed P99/P95 ratio (default: 2.0)
  consecutive_violation_threshold: number; // Violations before alert escalation
  slice_definitions: Array<{
    name: string;
    repo_type: string;
    language: string;
    size_category: string;
  }>;
  alert_channels: {
    console: boolean;
    webhook?: string;
    slack?: string;
    email?: string[];
  };
}

/**
 * Real-time tail latency monitoring with P99 ≤ 2× P95 validation
 */
export class TailLatencyMonitor extends EventEmitter {
  private config: MonitoringConfig;
  private measurements: Map<string, LatencyMeasurement[]> = new Map();
  private metrics_history: Map<string, LatencyMetrics[]> = new Map();
  private active_violations: Map<string, TailLatencyAlert> = new Map();
  private evaluation_timer: NodeJS.Timeout | null = null;
  private violation_counters: Map<string, number> = new Map();

  constructor(config: MonitoringConfig) {
    super();
    this.config = config;
    
    // Initialize measurement buckets for each slice
    for (const slice of config.slice_definitions) {
      const sliceKey = this.getSliceKey(slice);
      this.measurements.set(sliceKey, []);
      this.metrics_history.set(sliceKey, []);
      this.violation_counters.set(sliceKey, 0);
    }
  }

  /**
   * Start continuous monitoring
   */
  start(): void {
    console.log('🔍 Starting tail-latency monitoring...');
    console.log(`Window: ${this.config.measurement_window_ms}ms, Threshold: P99 ≤ ${this.config.p99_p95_threshold}× P95`);
    
    this.evaluation_timer = setInterval(() => {
      this.evaluateAllSlices();
    }, this.config.evaluation_frequency_ms);
    
    this.emit('started');
  }

  /**
   * Stop monitoring
   */
  stop(): void {
    if (this.evaluation_timer) {
      clearInterval(this.evaluation_timer);
      this.evaluation_timer = null;
    }
    
    console.log('🛑 Tail-latency monitoring stopped');
    this.emit('stopped');
  }

  /**
   * Record a latency measurement
   */
  recordLatency(measurement: LatencyMeasurement): void {
    const sliceKey = this.getSliceKey(measurement.metadata);
    const measurements = this.measurements.get(sliceKey);
    
    if (!measurements) {
      console.warn(`Unknown slice: ${sliceKey}`);
      return;
    }

    // Add measurement with timestamp
    measurements.push({
      ...measurement,
      timestamp: measurement.timestamp || new Date().toISOString()
    });

    // Trim old measurements outside window
    const cutoffTime = Date.now() - this.config.measurement_window_ms;
    const filtered = measurements.filter(m => 
      new Date(m.timestamp).getTime() > cutoffTime
    );
    
    this.measurements.set(sliceKey, filtered);
    
    this.emit('measurement', measurement);
  }

  /**
   * Get current metrics for a slice
   */
  getSliceMetrics(slice: string): LatencyMetrics | null {
    const measurements = this.measurements.get(slice);
    if (!measurements || measurements.length === 0) {
      return null;
    }

    return this.calculateMetrics(slice, measurements);
  }

  /**
   * Get all current metrics
   */
  getAllMetrics(): Record<string, LatencyMetrics> {
    const result: Record<string, LatencyMetrics> = {};
    
    for (const [slice, measurements] of this.measurements) {
      if (measurements.length > 0) {
        result[slice] = this.calculateMetrics(slice, measurements);
      }
    }
    
    return result;
  }

  /**
   * Get active violations
   */
  getActiveViolations(): TailLatencyAlert[] {
    return Array.from(this.active_violations.values());
  }

  /**
   * Get metrics history for trend analysis
   */
  getMetricsHistory(slice: string, hours: number = 24): LatencyMetrics[] {
    const history = this.metrics_history.get(slice) || [];
    const cutoffTime = Date.now() - (hours * 60 * 60 * 1000);
    
    return history.filter(m => 
      new Date(m.timestamp).getTime() > cutoffTime
    );
  }

  /**
   * Check if system is healthy across all slices
   */
  isSystemHealthy(): {
    healthy: boolean;
    total_slices: number;
    healthy_slices: number;
    violations: TailLatencyAlert[];
    summary: string;
  } {
    const allMetrics = this.getAllMetrics();
    const activeViolations = this.getActiveViolations();
    
    const totalSlices = Object.keys(allMetrics).length;
    const healthySlices = Object.values(allMetrics).filter(m => !m.is_violation).length;
    const healthy = activeViolations.length === 0;
    
    return {
      healthy,
      total_slices: totalSlices,
      healthy_slices: healthySlices,
      violations: activeViolations,
      summary: healthy ? 
        `All ${totalSlices} slices within P99 ≤ 2× P95 limits` :
        `${activeViolations.length} active tail-latency violations`
    };
  }

  /**
   * Generate comprehensive monitoring report
   */
  generateReport(): {
    timestamp: string;
    monitoring_config: MonitoringConfig;
    system_health: ReturnType<typeof TailLatencyMonitor.prototype.isSystemHealthy>;
    slice_metrics: Record<string, LatencyMetrics>;
    trend_analysis: Record<string, {
      slice: string;
      trend: 'improving' | 'stable' | 'degrading';
      trend_strength: number;
      recommendation: string;
    }>;
    alerts_summary: {
      total_alerts: number;
      by_severity: Record<string, number>;
      recent_escalations: TailLatencyAlert[];
    };
  } {
    const systemHealth = this.isSystemHealthy();
    const sliceMetrics = this.getAllMetrics();
    const trendAnalysis = this.analyzeTrends();
    
    const activeViolations = this.getActiveViolations();
    const severityCounts = activeViolations.reduce((acc, alert) => {
      acc[alert.severity] = (acc[alert.severity] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return {
      timestamp: new Date().toISOString(),
      monitoring_config: this.config,
      system_health: systemHealth,
      slice_metrics: sliceMetrics,
      trend_analysis: trendAnalysis,
      alerts_summary: {
        total_alerts: activeViolations.length,
        by_severity: severityCounts,
        recent_escalations: activeViolations.filter(a => a.severity === 'critical')
      }
    };
  }

  // Private implementation methods

  private evaluateAllSlices(): void {
    for (const [slice, measurements] of this.measurements) {
      if (measurements.length > 0) {
        const metrics = this.calculateMetrics(slice, measurements);
        this.evaluateViolations(slice, metrics);
        
        // Store metrics history
        const history = this.metrics_history.get(slice) || [];
        history.push(metrics);
        
        // Keep only last 24 hours of history
        const cutoffTime = Date.now() - (24 * 60 * 60 * 1000);
        const filtered = history.filter(m => 
          new Date(m.timestamp).getTime() > cutoffTime
        );
        this.metrics_history.set(slice, filtered);
      }
    }
  }

  private calculateMetrics(slice: string, measurements: LatencyMeasurement[]): LatencyMetrics {
    const latencies = measurements.map(m => m.latency_ms).sort((a, b) => a - b);
    const count = latencies.length;
    
    const percentiles = {
      p50: this.percentile(latencies, 0.50),
      p90: this.percentile(latencies, 0.90),
      p95: this.percentile(latencies, 0.95),
      p99: this.percentile(latencies, 0.99),
      p99_9: this.percentile(latencies, 0.999)
    };
    
    const tailLatencyRatio = percentiles.p99 / percentiles.p95;
    const isViolation = tailLatencyRatio > this.config.p99_p95_threshold;
    
    return {
      timestamp: new Date().toISOString(),
      slice,
      window_duration_ms: this.config.measurement_window_ms,
      sample_count: count,
      percentiles,
      tail_latency_ratio: tailLatencyRatio,
      violation_threshold: this.config.p99_p95_threshold,
      is_violation: isViolation
    };
  }

  private evaluateViolations(slice: string, metrics: LatencyMetrics): void {
    if (metrics.is_violation) {
      // Increment violation counter
      const currentCount = this.violation_counters.get(slice) || 0;
      this.violation_counters.set(slice, currentCount + 1);
      
      // Create or update alert
      this.createOrUpdateAlert(slice, metrics);
    } else {
      // Reset violation counter and clear alert if resolved
      this.violation_counters.set(slice, 0);
      
      if (this.active_violations.has(slice)) {
        const resolvedAlert = this.active_violations.get(slice)!;
        this.active_violations.delete(slice);
        
        console.log(`✅ Tail-latency violation resolved for ${slice}`);
        this.emit('violation_resolved', resolvedAlert);
      }
    }
  }

  private createOrUpdateAlert(slice: string, metrics: LatencyMetrics): void {
    const consecutiveViolations = this.violation_counters.get(slice) || 0;
    const violationMagnitude = metrics.tail_latency_ratio - metrics.violation_threshold;
    
    // Determine severity based on magnitude and persistence
    let severity: 'warning' | 'error' | 'critical' = 'warning';
    if (consecutiveViolations >= this.config.consecutive_violation_threshold) {
      severity = violationMagnitude > 1.0 ? 'critical' : 'error';
    }
    
    const alert: TailLatencyAlert = {
      id: `tail-latency-${slice}-${Date.now()}`,
      timestamp: new Date().toISOString(),
      severity,
      slice,
      metrics,
      violation_details: {
        current_ratio: metrics.tail_latency_ratio,
        threshold: metrics.violation_threshold,
        violation_magnitude: violationMagnitude,
        consecutive_violations: consecutiveViolations
      },
      recommended_actions: this.generateRecommendations(slice, metrics, consecutiveViolations)
    };
    
    const existingAlert = this.active_violations.get(slice);
    const isNewAlert = !existingAlert;
    const severityEscalated = !!(existingAlert && alert.severity !== existingAlert.severity);
    
    this.active_violations.set(slice, alert);
    
    if (isNewAlert) {
      console.warn(`⚠️  New tail-latency violation detected: ${slice}`);
      console.warn(`   P99/P95 ratio: ${metrics.tail_latency_ratio.toFixed(2)} (threshold: ${metrics.violation_threshold})`);
      this.emit('violation_detected', alert);
    } else if (severityEscalated) {
      console.error(`🚨 Tail-latency violation escalated to ${alert.severity}: ${slice}`);
      this.emit('violation_escalated', alert);
    }
    
    this.sendAlert(alert, isNewAlert || severityEscalated);
  }

  private generateRecommendations(slice: string, metrics: LatencyMetrics, consecutiveViolations: number): string[] {
    const recommendations = [];
    
    if (metrics.tail_latency_ratio > 3.0) {
      recommendations.push('CRITICAL: Investigate immediate performance bottlenecks');
      recommendations.push('Consider rolling back recent changes');
    } else if (metrics.tail_latency_ratio > 2.5) {
      recommendations.push('Review recent deployments for performance regressions');
      recommendations.push('Check system resource utilization');
    } else {
      recommendations.push('Monitor for trending issues');
      recommendations.push('Review query patterns for outliers');
    }
    
    if (consecutiveViolations > 5) {
      recommendations.push('Consider scaling resources or optimizing hot paths');
    }
    
    return recommendations;
  }

  private sendAlert(alert: TailLatencyAlert, shouldNotify: boolean): void {
    if (!shouldNotify) return;
    
    if (this.config.alert_channels.console) {
      this.logAlert(alert);
    }
    
    // Additional alert channels would be implemented here
    // webhook, slack, email, etc.
    
    this.emit('alert_sent', alert);
  }

  private logAlert(alert: TailLatencyAlert): void {
    const icon = alert.severity === 'critical' ? '🚨' : 
                 alert.severity === 'error' ? '❌' : '⚠️';
    
    console.log(`${icon} TAIL-LATENCY ALERT [${alert.severity.toUpperCase()}]`);
    console.log(`   Slice: ${alert.slice}`);
    console.log(`   P99/P95 Ratio: ${alert.violation_details.current_ratio.toFixed(2)} (threshold: ${alert.violation_details.threshold})`);
    console.log(`   Consecutive Violations: ${alert.violation_details.consecutive_violations}`);
    console.log(`   Sample Count: ${alert.metrics.sample_count}`);
    console.log(`   Recommendations:`);
    alert.recommended_actions.forEach(action => console.log(`     - ${action}`));
  }

  private analyzeTrends(): Record<string, {
    slice: string;
    trend: 'improving' | 'stable' | 'degrading';
    trend_strength: number;
    recommendation: string;
  }> {
    const trends: Record<string, any> = {};
    
    for (const slice of this.measurements.keys()) {
      const history = this.getMetricsHistory(slice, 4); // Last 4 hours
      
      if (history.length < 3) {
        trends[slice] = {
          slice,
          trend: 'stable',
          trend_strength: 0,
          recommendation: 'Insufficient data for trend analysis'
        };
        continue;
      }
      
      // Simple trend analysis using tail latency ratios
      const ratios = history.map(h => h.tail_latency_ratio);
      const recentAvg = ratios.slice(-3).reduce((a, b) => a + b, 0) / 3;
      const olderAvg = ratios.slice(0, -3).reduce((a, b) => a + b, 0) / (ratios.length - 3);
      
      const trendStrength = Math.abs(recentAvg - olderAvg);
      let trend: 'improving' | 'stable' | 'degrading' = 'stable';
      let recommendation = 'Performance is stable';
      
      if (recentAvg > olderAvg + 0.1) {
        trend = 'degrading';
        recommendation = 'Performance degrading - investigate recent changes';
      } else if (recentAvg < olderAvg - 0.1) {
        trend = 'improving';
        recommendation = 'Performance improving - monitor for stability';
      }
      
      trends[slice] = {
        slice,
        trend,
        trend_strength: trendStrength,
        recommendation
      };
    }
    
    return trends;
  }

  private percentile(sortedArray: number[], percentile: number): number {
    if (sortedArray.length === 0) return 0;
    
    const index = percentile * (sortedArray.length - 1);
    
    if (Number.isInteger(index)) {
      return sortedArray[index];
    } else {
      const lower = Math.floor(index);
      const upper = Math.ceil(index);
      const weight = index - lower;
      return sortedArray[lower] * (1 - weight) + sortedArray[upper] * weight;
    }
  }

  public getSliceKey(metadata: { repo_type: string; language: string; size_category: string }): string {
    return `${metadata.repo_type}/${metadata.language}/${metadata.size_category}`;
  }
}

/**
 * Integration with benchmark system for automatic latency monitoring
 */
export class BenchmarkLatencyIntegration {
  private monitor: TailLatencyMonitor;

  constructor(monitor: TailLatencyMonitor) {
    this.monitor = monitor;
  }

  /**
   * Hook into benchmark execution to record latencies
   */
  recordBenchmarkLatency(
    operation: string,
    latency_ms: number,
    metadata: {
      repo_type: string;
      language: string;
      size_category: string;
      query_type: string;
      result_count: number;
    }
  ): void {
    this.monitor.recordLatency({
      timestamp: new Date().toISOString(),
      operation,
      slice: this.monitor.getSliceKey ? 
        this.monitor.getSliceKey(metadata) : 
        `${metadata.repo_type}/${metadata.language}/${metadata.size_category}`,
      latency_ms,
      metadata
    });
  }

  /**
   * Check if current benchmarks would pass tail-latency gates
   */
  validateBenchmarkResults(): {
    passed: boolean;
    failing_slices: string[];
    violations: TailLatencyAlert[];
    recommendations: string[];
  } {
    const health = this.monitor.isSystemHealthy();
    const violations = health.violations;
    
    const failingSlices = violations.map(v => v.slice);
    const recommendations = violations.length > 0 ? [
      'Review performance optimization opportunities',
      'Consider increasing resource allocation',
      'Investigate query patterns causing outliers'
    ] : ['All tail-latency gates are passing'];

    return {
      passed: health.healthy,
      failing_slices: failingSlices,
      violations: violations,
      recommendations
    };
  }
}

/**
 * Factory for creating monitoring configurations
 */
export class MonitoringConfigFactory {
  static createPhaseDBenchmarkConfig(): MonitoringConfig {
    return {
      measurement_window_ms: 5 * 60 * 1000, // 5 minutes
      evaluation_frequency_ms: 30 * 1000, // 30 seconds
      p99_p95_threshold: 2.0, // P99 ≤ 2× P95 as per Phase D requirements
      consecutive_violation_threshold: 3,
      slice_definitions: [
        // Backend repositories
        { name: 'backend/typescript/small', repo_type: 'backend', language: 'typescript', size_category: 'small' },
        { name: 'backend/typescript/medium', repo_type: 'backend', language: 'typescript', size_category: 'medium' },
        { name: 'backend/typescript/large', repo_type: 'backend', language: 'typescript', size_category: 'large' },
        { name: 'backend/python/small', repo_type: 'backend', language: 'python', size_category: 'small' },
        { name: 'backend/python/medium', repo_type: 'backend', language: 'python', size_category: 'medium' },
        { name: 'backend/python/large', repo_type: 'backend', language: 'python', size_category: 'large' },
        
        // Frontend repositories
        { name: 'frontend/typescript/small', repo_type: 'frontend', language: 'typescript', size_category: 'small' },
        { name: 'frontend/typescript/medium', repo_type: 'frontend', language: 'typescript', size_category: 'medium' },
        { name: 'frontend/typescript/large', repo_type: 'frontend', language: 'typescript', size_category: 'large' },
        { name: 'frontend/javascript/medium', repo_type: 'frontend', language: 'javascript', size_category: 'medium' },
        
        // Monorepos
        { name: 'monorepo/typescript/large', repo_type: 'monorepo', language: 'typescript', size_category: 'large' },
        { name: 'monorepo/mixed/large', repo_type: 'monorepo', language: 'mixed', size_category: 'large' }
      ],
      alert_channels: {
        console: true,
        // webhook: process.env.SLACK_WEBHOOK_URL,
        // email: process.env.ALERT_EMAIL_ADDRESSES?.split(',')
      }
    };
  }

  static createProductionConfig(): MonitoringConfig {
    return {
      ...this.createPhaseDBenchmarkConfig(),
      measurement_window_ms: 15 * 60 * 1000, // 15 minutes for production
      evaluation_frequency_ms: 60 * 1000, // 1 minute evaluation
      consecutive_violation_threshold: 2, // Stricter for production
    };
  }
}