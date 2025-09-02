/**
 * Operational Monitoring & Alerting Framework
 * 
 * Provides production-ready monitoring, alerting, and observability
 * for the Lens search engine based on Phase 4 robustness findings.
 */

import { promises as fs } from 'fs';
import path from 'path';

export interface MonitoringMetrics {
  timestamp: string;
  stage: string;
  latency_p50: number;
  latency_p95: number;
  latency_p99: number;
  latency_p99_9: number;
  latency_max: number;
  throughput_qps: number;
  error_rate: number;
  availability: number;
  cache_hit_rate: number;
  index_size_mb: number;
  memory_usage_mb: number;
  cpu_utilization_percent: number;
}

export interface AlertRule {
  id: string;
  name: string;
  condition: string;
  threshold: number;
  severity: 'info' | 'warning' | 'critical';
  description: string;
  runbook_url?: string;
}

export interface Alert {
  id: string;
  rule_id: string;
  severity: 'info' | 'warning' | 'critical';
  message: string;
  timestamp: string;
  value: number;
  threshold: number;
  resolved: boolean;
  resolved_at?: string;
  metadata: Record<string, any>;
}

export class OperationalMonitoringSystem {
  private readonly alertRules: AlertRule[];
  private readonly activeAlerts: Map<string, Alert> = new Map();
  private readonly metricsHistory: MonitoringMetrics[] = [];
  private readonly maxHistorySize = 10000; // Keep last 10k metrics

  constructor(
    private readonly outputDir: string,
    customAlertRules?: AlertRule[]
  ) {
    this.alertRules = customAlertRules || this.getDefaultAlertRules();
  }

  /**
   * Record metrics and evaluate alerts
   */
  async recordMetrics(metrics: MonitoringMetrics): Promise<Alert[]> {
    // Store metrics
    this.metricsHistory.push(metrics);
    
    // Maintain history size
    if (this.metricsHistory.length > this.maxHistorySize) {
      this.metricsHistory.shift();
    }

    // Evaluate alert rules
    const newAlerts = await this.evaluateAlerts(metrics);

    // Persist metrics and alerts
    await this.persistMetrics(metrics);
    
    if (newAlerts.length > 0) {
      await this.persistAlerts(newAlerts);
      await this.sendAlertNotifications(newAlerts);
    }

    return newAlerts;
  }

  /**
   * Get current system status
   */
  getCurrentStatus(): {
    status: 'healthy' | 'degraded' | 'critical';
    activeAlerts: Alert[];
    recentMetrics: MonitoringMetrics;
    summary: {
      total_alerts: number;
      critical_alerts: number;
      warning_alerts: number;
      last_update: string;
    };
  } {
    const activeAlerts = Array.from(this.activeAlerts.values()).filter(alert => !alert.resolved);
    const criticalAlerts = activeAlerts.filter(alert => alert.severity === 'critical');
    const warningAlerts = activeAlerts.filter(alert => alert.severity === 'warning');

    let status: 'healthy' | 'degraded' | 'critical' = 'healthy';
    if (criticalAlerts.length > 0) {
      status = 'critical';
    } else if (warningAlerts.length > 0) {
      status = 'degraded';
    }

    const recentMetrics = this.metricsHistory[this.metricsHistory.length - 1] || {} as MonitoringMetrics;

    return {
      status,
      activeAlerts,
      recentMetrics,
      summary: {
        total_alerts: activeAlerts.length,
        critical_alerts: criticalAlerts.length,
        warning_alerts: warningAlerts.length,
        last_update: recentMetrics.timestamp || new Date().toISOString()
      }
    };
  }

  /**
   * Generate monitoring dashboard data
   */
  async generateDashboardData(): Promise<{
    real_time_metrics: MonitoringMetrics;
    trend_data: {
      latency_trend: Array<{ timestamp: string; p95: number; p99: number }>;
      throughput_trend: Array<{ timestamp: string; qps: number }>;
      error_rate_trend: Array<{ timestamp: string; error_rate: number }>;
      availability_trend: Array<{ timestamp: string; availability: number }>;
    };
    alert_summary: {
      active_alerts: Alert[];
      resolved_alerts_24h: Alert[];
      alert_trend: Array<{ hour: string; alert_count: number }>;
    };
    performance_summary: {
      avg_latency_p95: number;
      avg_latency_p99: number;
      avg_throughput: number;
      avg_availability: number;
      slo_compliance: {
        latency_p95_slo: { target: number; current: number; met: boolean };
        availability_slo: { target: number; current: number; met: boolean };
        error_rate_slo: { target: number; current: number; met: boolean };
      };
    };
  }> {
    const recentMetrics = this.metricsHistory.slice(-100); // Last 100 metrics
    const last24h = new Date(Date.now() - 24 * 60 * 60 * 1000);

    // Real-time metrics (latest)
    const realTimeMetrics = this.metricsHistory[this.metricsHistory.length - 1] || {} as MonitoringMetrics;

    // Trend data
    const latencyTrend = recentMetrics.map(m => ({
      timestamp: m.timestamp,
      p95: m.latency_p95,
      p99: m.latency_p99
    }));

    const throughputTrend = recentMetrics.map(m => ({
      timestamp: m.timestamp,
      qps: m.throughput_qps
    }));

    const errorRateTrend = recentMetrics.map(m => ({
      timestamp: m.timestamp,
      error_rate: m.error_rate
    }));

    const availabilityTrend = recentMetrics.map(m => ({
      timestamp: m.timestamp,
      availability: m.availability
    }));

    // Alert summary
    const activeAlerts = Array.from(this.activeAlerts.values()).filter(alert => !alert.resolved);
    const resolvedAlerts24h = Array.from(this.activeAlerts.values())
      .filter(alert => alert.resolved && new Date(alert.resolved_at || '') > last24h);

    // Performance summary
    const avgLatencyP95 = recentMetrics.length > 0 
      ? recentMetrics.reduce((sum, m) => sum + m.latency_p95, 0) / recentMetrics.length
      : 0;
    
    const avgLatencyP99 = recentMetrics.length > 0
      ? recentMetrics.reduce((sum, m) => sum + m.latency_p99, 0) / recentMetrics.length
      : 0;

    const avgThroughput = recentMetrics.length > 0
      ? recentMetrics.reduce((sum, m) => sum + m.throughput_qps, 0) / recentMetrics.length
      : 0;

    const avgAvailability = recentMetrics.length > 0
      ? recentMetrics.reduce((sum, m) => sum + m.availability, 0) / recentMetrics.length
      : 1.0;

    const avgErrorRate = recentMetrics.length > 0
      ? recentMetrics.reduce((sum, m) => sum + m.error_rate, 0) / recentMetrics.length
      : 0;

    // SLO targets (based on Phase 4 requirements)
    const sloTargets = {
      latency_p95: 30, // ms
      availability: 0.995, // 99.5%
      error_rate: 0.001 // 0.1%
    };

    return {
      real_time_metrics: realTimeMetrics,
      trend_data: {
        latency_trend: latencyTrend,
        throughput_trend: throughputTrend,
        error_rate_trend: errorRateTrend,
        availability_trend: availabilityTrend
      },
      alert_summary: {
        active_alerts: activeAlerts,
        resolved_alerts_24h: resolvedAlerts24h,
        alert_trend: [] // Could be implemented to show alert frequency over time
      },
      performance_summary: {
        avg_latency_p95: avgLatencyP95,
        avg_latency_p99: avgLatencyP99,
        avg_throughput: avgThroughput,
        avg_availability: avgAvailability,
        slo_compliance: {
          latency_p95_slo: {
            target: sloTargets.latency_p95,
            current: avgLatencyP95,
            met: avgLatencyP95 <= sloTargets.latency_p95
          },
          availability_slo: {
            target: sloTargets.availability,
            current: avgAvailability,
            met: avgAvailability >= sloTargets.availability
          },
          error_rate_slo: {
            target: sloTargets.error_rate,
            current: avgErrorRate,
            met: avgErrorRate <= sloTargets.error_rate
          }
        }
      }
    };
  }

  /**
   * Generate operational runbook
   */
  async generateRunbook(): Promise<string> {
    const runbook = `# Lens Search Engine - Operational Runbook

## System Overview

The Lens search engine is a multi-stage search system with the following components:
- Stage A: Lexical search (target: <20ms p95)
- Stage B: Structural search (target: <15ms p95)  
- Stage C: Semantic search (target: <25ms p95)
- Overall end-to-end search (target: <50ms p95)

## Service Level Objectives (SLOs)

Based on Phase 4 robustness testing:

- **Latency**: P95 < 30ms, P99 < 50ms
- **Availability**: > 99.5%
- **Error Rate**: < 0.1%
- **Throughput**: Support 100+ QPS sustained

## Alert Response Procedures

### Critical Alerts

#### High Latency (P99 > 2x P95)
**Symptoms**: Search responses are slow, user experience degraded
**Investigation**:
1. Check system resources (CPU, memory, disk I/O)
2. Verify index health and compaction status
3. Check for concurrent operations (indexing, compaction)
4. Review recent changes or deployments

**Mitigation**:
1. Scale up compute resources if needed
2. Defer non-critical operations (compaction, indexing)
3. Enable circuit breakers if available
4. Consider failing over to backup systems

#### Service Unavailable (Availability < 99%)
**Symptoms**: High error rates, failed requests
**Investigation**:
1. Check service health endpoints
2. Verify database/storage connectivity
3. Check for resource exhaustion
4. Review logs for error patterns

**Mitigation**:
1. Restart unhealthy service instances
2. Scale up capacity
3. Activate backup/failover procedures
4. Implement graceful degradation

#### High Error Rate (> 1%)
**Symptoms**: Increased failed requests
**Investigation**:
1. Check error logs for patterns
2. Verify input validation and data quality
3. Check external dependencies
4. Review recent configuration changes

**Mitigation**:
1. Fix identified root cause
2. Implement input validation
3. Rollback problematic changes
4. Enable graceful error handling

### Warning Alerts

#### Tail Latency Increase
**Investigation**: Review query complexity, cache performance
**Action**: Monitor and investigate if trend continues

#### Cache Hit Rate Degradation
**Investigation**: Check cache configuration, memory pressure
**Action**: Optimize cache policies, increase cache size

## Maintenance Procedures

### Planned Compaction
1. Schedule during low-traffic periods
2. Monitor latency increase (should be <50%)
3. Verify service availability >99% during compaction
4. Check for data corruption post-compaction

### Index Rebuilds
1. Use incremental rebuilds when possible
2. Monitor affected shard percentage (<10%)
3. Verify search quality maintained
4. Check rebuild throughput targets

### Deployment Procedures
1. Run Phase 4 robustness tests in staging
2. Verify all quality gates pass
3. Deploy during maintenance windows
4. Monitor tail latencies post-deployment

## Performance Baselines

Based on Phase 4 testing results:

| Metric | Baseline | Warning | Critical |
|--------|----------|---------|----------|
| Stage A P95 | 12-15ms | >20ms | >30ms |
| Stage B P95 | 8-12ms | >15ms | >25ms |
| Stage C P95 | 15-20ms | >25ms | >40ms |
| E2E P95 | 35-45ms | >50ms | >70ms |
| Throughput | 50-100 QPS | <25 QPS | <10 QPS |

## Emergency Contacts

- On-call Engineer: [Contact Info]
- Platform Team: [Contact Info]
- Infrastructure Team: [Contact Info]

## External Dependencies

- Storage System: [Details]
- Monitoring System: [Details]
- Load Balancer: [Details]

## Troubleshooting Commands

### Health Checks
\`\`\`bash
# Check service health
curl http://localhost:3001/health

# Check metrics endpoint
curl http://localhost:3001/metrics

# Check recent performance
curl http://localhost:3001/api/search?q=test&benchmark=true
\`\`\`

### Log Analysis
\`\`\`bash
# Check error logs
tail -f logs/error.log | grep ERROR

# Analyze latency patterns
grep "latency" logs/app.log | tail -100

# Check memory usage
ps aux | grep lens-search
\`\`\`

## Recovery Procedures

### Full System Recovery
1. Stop all services
2. Verify data integrity
3. Clear caches if needed
4. Restart services in order
5. Run smoke tests
6. Gradually increase traffic

### Partial Recovery
1. Identify affected components
2. Restart only affected services
3. Verify functionality
4. Monitor for stability
`;

    const runbookPath = path.join(this.outputDir, 'operational-runbook.md');
    await fs.writeFile(runbookPath, runbook);
    
    console.log(`📖 Operational runbook generated: ${runbookPath}`);
    return runbook;
  }

  // Private methods

  private getDefaultAlertRules(): AlertRule[] {
    return [
      {
        id: 'high_p99_latency',
        name: 'High P99 Latency',
        condition: 'latency_p99 > latency_p95 * 2',
        threshold: 2.0,
        severity: 'critical',
        description: 'P99 latency is more than 2x P95, indicating tail latency issues'
      },
      {
        id: 'high_p95_latency',
        name: 'High P95 Latency',
        condition: 'latency_p95 > 30',
        threshold: 30,
        severity: 'warning',
        description: 'P95 latency exceeds 30ms threshold'
      },
      {
        id: 'low_availability',
        name: 'Low Availability',
        condition: 'availability < 0.99',
        threshold: 0.99,
        severity: 'critical',
        description: 'Service availability below 99%'
      },
      {
        id: 'high_error_rate',
        name: 'High Error Rate',
        condition: 'error_rate > 0.01',
        threshold: 0.01,
        severity: 'critical',
        description: 'Error rate above 1%'
      },
      {
        id: 'low_throughput',
        name: 'Low Throughput',
        condition: 'throughput_qps < 10',
        threshold: 10,
        severity: 'warning',
        description: 'Throughput below expected minimum'
      },
      {
        id: 'low_cache_hit_rate',
        name: 'Low Cache Hit Rate',
        condition: 'cache_hit_rate < 0.8',
        threshold: 0.8,
        severity: 'warning',
        description: 'Cache hit rate below 80%'
      },
      {
        id: 'high_memory_usage',
        name: 'High Memory Usage',
        condition: 'memory_usage_mb > 2048',
        threshold: 2048,
        severity: 'warning',
        description: 'Memory usage above 2GB'
      },
      {
        id: 'high_cpu_usage',
        name: 'High CPU Usage',
        condition: 'cpu_utilization_percent > 80',
        threshold: 80,
        severity: 'warning',
        description: 'CPU utilization above 80%'
      }
    ];
  }

  private async evaluateAlerts(metrics: MonitoringMetrics): Promise<Alert[]> {
    const newAlerts: Alert[] = [];

    for (const rule of this.alertRules) {
      const alertTriggered = this.evaluateAlertCondition(rule, metrics);
      const alertId = `${rule.id}_${metrics.timestamp}`;

      if (alertTriggered) {
        // Check if this alert is already active
        const existingAlert = Array.from(this.activeAlerts.values())
          .find(alert => alert.rule_id === rule.id && !alert.resolved);

        if (!existingAlert) {
          const alert: Alert = {
            id: alertId,
            rule_id: rule.id,
            severity: rule.severity,
            message: `${rule.name}: ${rule.description}`,
            timestamp: metrics.timestamp,
            value: this.getMetricValue(rule, metrics),
            threshold: rule.threshold,
            resolved: false,
            metadata: {
              stage: metrics.stage,
              full_metrics: metrics
            }
          };

          this.activeAlerts.set(alertId, alert);
          newAlerts.push(alert);
        }
      } else {
        // Check if we should resolve any active alerts for this rule
        const activeAlert = Array.from(this.activeAlerts.values())
          .find(alert => alert.rule_id === rule.id && !alert.resolved);

        if (activeAlert) {
          activeAlert.resolved = true;
          activeAlert.resolved_at = metrics.timestamp;
        }
      }
    }

    return newAlerts;
  }

  private evaluateAlertCondition(rule: AlertRule, metrics: MonitoringMetrics): boolean {
    // Simple condition evaluation - in production, use a proper expression evaluator
    switch (rule.id) {
      case 'high_p99_latency':
        return metrics.latency_p99 > metrics.latency_p95 * rule.threshold;
      case 'high_p95_latency':
        return metrics.latency_p95 > rule.threshold;
      case 'low_availability':
        return metrics.availability < rule.threshold;
      case 'high_error_rate':
        return metrics.error_rate > rule.threshold;
      case 'low_throughput':
        return metrics.throughput_qps < rule.threshold;
      case 'low_cache_hit_rate':
        return metrics.cache_hit_rate < rule.threshold;
      case 'high_memory_usage':
        return metrics.memory_usage_mb > rule.threshold;
      case 'high_cpu_usage':
        return metrics.cpu_utilization_percent > rule.threshold;
      default:
        return false;
    }
  }

  private getMetricValue(rule: AlertRule, metrics: MonitoringMetrics): number {
    switch (rule.id) {
      case 'high_p99_latency':
        return metrics.latency_p99 / metrics.latency_p95;
      case 'high_p95_latency':
        return metrics.latency_p95;
      case 'low_availability':
        return metrics.availability;
      case 'high_error_rate':
        return metrics.error_rate;
      case 'low_throughput':
        return metrics.throughput_qps;
      case 'low_cache_hit_rate':
        return metrics.cache_hit_rate;
      case 'high_memory_usage':
        return metrics.memory_usage_mb;
      case 'high_cpu_usage':
        return metrics.cpu_utilization_percent;
      default:
        return 0;
    }
  }

  private async persistMetrics(metrics: MonitoringMetrics): Promise<void> {
    const metricsPath = path.join(this.outputDir, 'monitoring-metrics.ndjson');
    const line = JSON.stringify(metrics) + '\n';
    await fs.appendFile(metricsPath, line);
  }

  private async persistAlerts(alerts: Alert[]): Promise<void> {
    const alertsPath = path.join(this.outputDir, 'monitoring-alerts.ndjson');
    
    for (const alert of alerts) {
      const line = JSON.stringify(alert) + '\n';
      await fs.appendFile(alertsPath, line);
    }
  }

  private async sendAlertNotifications(alerts: Alert[]): Promise<void> {
    // In production, integrate with notification systems (Slack, PagerDuty, etc.)
    const criticalAlerts = alerts.filter(alert => alert.severity === 'critical');
    
    if (criticalAlerts.length > 0) {
      console.log(`🚨 CRITICAL ALERTS (${criticalAlerts.length}):`);
      criticalAlerts.forEach(alert => {
        console.log(`   ${alert.message} (value: ${alert.value}, threshold: ${alert.threshold})`);
      });
    }

    const warningAlerts = alerts.filter(alert => alert.severity === 'warning');
    if (warningAlerts.length > 0) {
      console.log(`⚠️  WARNING ALERTS (${warningAlerts.length}):`);
      warningAlerts.forEach(alert => {
        console.log(`   ${alert.message} (value: ${alert.value}, threshold: ${alert.threshold})`);
      });
    }
  }
}