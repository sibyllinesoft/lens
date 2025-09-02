/**
 * Production Monitoring System with CUSUM Alarms
 * 
 * Implements comprehensive drift detection and alerting:
 * - CUSUM detection for Anchor P@1, Recall@50 drift
 * - Ladder positives-in-candidates monitoring 
 * - LSIF/tree-sitter coverage tracking
 * - Smart alerting with sustained deviation detection
 * - Dashboard KPIs and real-time health monitoring
 * - Integration with canary rollout abort conditions
 */

import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';
import { EventEmitter } from 'events';

interface CUSUMDetector {
  metric_name: string;
  target_mean: number;
  target_std: number;
  threshold: number; // CUSUM threshold (typically 3-5)
  
  // CUSUM state
  positive_sum: number;
  negative_sum: number;
  last_reset: string;
  
  // Detection state
  alarm_active: boolean;
  alarm_start_time?: string;
  consecutive_violations: number;
  max_consecutive_violations: number;
}

interface MetricSnapshot {
  timestamp: string;
  
  // Core search metrics
  anchor_p_at_1: number;
  recall_at_50: number;
  ndcg_at_10: number;
  
  // Ladder metrics
  ladder_positives_in_candidates: number;
  ladder_total_candidates: number;
  ladder_positive_rate: number;
  
  // Coverage metrics
  lsif_coverage: number;
  tree_sitter_coverage: number;
  total_spans: number;
  covered_spans: number;
  
  // Performance metrics
  p95_latency_ms: number;
  p99_latency_ms: number;
  qps: number;
  error_rate: number;
  
  // Results metrics
  results_per_query_mean: number;
  results_per_query_p95: number;
  zero_result_rate: number;
  
  // System health
  memory_usage_gb: number;
  cpu_utilization: number;
  disk_usage_gb: number;
}

interface AlertCondition {
  name: string;
  metric_path: string;
  condition: 'above' | 'below' | 'outside_range' | 'cusum_alarm';
  threshold?: number;
  threshold_range?: [number, number];
  
  // Alert properties
  severity: 'low' | 'medium' | 'high' | 'critical';
  sustained_minutes: number; // Must be violated for this long
  cooldown_minutes: number; // Minimum time between alerts
  
  // Actions
  actions: AlertAction[];
  
  // State
  current_violation_start?: string;
  last_alert_sent?: string;
  violations_count: number;
}

interface AlertAction {
  type: 'webhook' | 'email' | 'pagerduty' | 'kill_switch' | 'rollback';
  config: Record<string, any>;
}

interface MonitoringState {
  cusum_detectors: Record<string, CUSUMDetector>;
  alert_conditions: AlertCondition[];
  metric_history: MetricSnapshot[];
  dashboard_config: DashboardConfig;
  health_status: HealthStatus;
}

interface DashboardConfig {
  refresh_interval_seconds: number;
  metric_retention_hours: number;
  chart_configs: ChartConfig[];
}

interface ChartConfig {
  title: string;
  metrics: string[];
  chart_type: 'line' | 'gauge' | 'histogram' | 'status';
  time_range_hours: number;
  alert_thresholds?: Record<string, number>;
}

interface HealthStatus {
  overall_status: 'healthy' | 'degraded' | 'unhealthy' | 'critical';
  active_alarms: string[];
  recent_violations: Array<{
    metric: string;
    severity: string;
    timestamp: string;
    value: number;
  }>;
  system_uptime_hours: number;
  last_health_check: string;
}

export class ProductionMonitoringSystem extends EventEmitter {
  private readonly monitoringDir: string;
  private monitoringState: MonitoringState;
  private metricsInterval?: NodeJS.Timeout;
  private alertsInterval?: NodeJS.Timeout;
  private isRunning: boolean = false;
  
  constructor(monitoringDir: string = './deployment-artifacts/monitoring') {
    super();
    this.monitoringDir = monitoringDir;
    
    if (!existsSync(this.monitoringDir)) {
      mkdirSync(this.monitoringDir, { recursive: true });
    }
    
    this.monitoringState = this.initializeMonitoringState();
  }
  
  /**
   * Start production monitoring system
   */
  public async startMonitoring(): Promise<void> {
    if (this.isRunning) {
      console.log('📊 Monitoring system already running');
      return;
    }
    
    console.log('🚀 Starting production monitoring system...');
    
    // Initialize CUSUM detectors with baseline values
    await this.initializeCUSUMBaselines();
    
    // Start metrics collection
    this.metricsInterval = setInterval(async () => {
      await this.collectMetrics();
    }, 30000); // 30 seconds
    
    // Start alert evaluation
    this.alertsInterval = setInterval(async () => {
      await this.evaluateAlerts();
    }, 60000); // 1 minute
    
    this.isRunning = true;
    
    console.log('✅ Production monitoring system started');
    this.emit('monitoring_started');
  }
  
  /**
   * Stop monitoring system
   */
  public stopMonitoring(): void {
    if (this.metricsInterval) {
      clearInterval(this.metricsInterval);
      this.metricsInterval = undefined;
    }
    
    if (this.alertsInterval) {
      clearInterval(this.alertsInterval);
      this.alertsInterval = undefined;
    }
    
    this.isRunning = false;
    console.log('🛑 Production monitoring system stopped');
    this.emit('monitoring_stopped');
  }
  
  /**
   * Initialize CUSUM baselines from version config
   */
  private async initializeCUSUMBaselines(): Promise<void> {
    // Load baseline metrics from version manager
    try {
      const { versionManager } = await import('./version-manager.js');
      const config = versionManager.loadVersionConfig();
      const baseline = config.baseline_metrics;
      
      // Initialize CUSUM detectors with baseline values
      this.monitoringState.cusum_detectors = {
        anchor_p_at_1: {
          metric_name: 'anchor_p_at_1',
          target_mean: baseline.p_at_1,
          target_std: baseline.p_at_1 * 0.1, // 10% relative std
          threshold: 3.0,
          positive_sum: 0,
          negative_sum: 0,
          last_reset: new Date().toISOString(),
          alarm_active: false,
          consecutive_violations: 0,
          max_consecutive_violations: 5
        },
        
        recall_at_50: {
          metric_name: 'recall_at_50',
          target_mean: baseline.recall_at_50,
          target_std: baseline.recall_at_50 * 0.05, // 5% relative std
          threshold: 3.0,
          positive_sum: 0,
          negative_sum: 0,
          last_reset: new Date().toISOString(),
          alarm_active: false,
          consecutive_violations: 0,
          max_consecutive_violations: 5
        },
        
        ladder_positive_rate: {
          metric_name: 'ladder_positive_rate',
          target_mean: 0.75, // Expected positive rate in candidates
          target_std: 0.1,
          threshold: 3.0,
          positive_sum: 0,
          negative_sum: 0,
          last_reset: new Date().toISOString(),
          alarm_active: false,
          consecutive_violations: 0,
          max_consecutive_violations: 3
        },
        
        lsif_coverage: {
          metric_name: 'lsif_coverage',
          target_mean: 0.95, // 95% LSIF coverage expected
          target_std: 0.02,
          threshold: 5.0, // Higher threshold for coverage metrics
          positive_sum: 0,
          negative_sum: 0,
          last_reset: new Date().toISOString(),
          alarm_active: false,
          consecutive_violations: 0,
          max_consecutive_violations: 3
        },
        
        tree_sitter_coverage: {
          metric_name: 'tree_sitter_coverage',
          target_mean: 0.98, // 98% tree-sitter coverage expected
          target_std: 0.01,
          threshold: 5.0,
          positive_sum: 0,
          negative_sum: 0,
          last_reset: new Date().toISOString(),
          alarm_active: false,
          consecutive_violations: 0,
          max_consecutive_violations: 3
        }
      };
      
      console.log('📈 CUSUM baselines initialized from version config');
      
    } catch (error) {
      console.warn('⚠️  Failed to load version baselines, using defaults:', error);
      this.initializeDefaultBaselines();
    }
  }
  
  /**
   * Initialize default CUSUM baselines
   */
  private initializeDefaultBaselines(): void {
    this.monitoringState.cusum_detectors = {
      anchor_p_at_1: {
        metric_name: 'anchor_p_at_1',
        target_mean: 0.75,
        target_std: 0.075,
        threshold: 3.0,
        positive_sum: 0,
        negative_sum: 0,
        last_reset: new Date().toISOString(),
        alarm_active: false,
        consecutive_violations: 0,
        max_consecutive_violations: 5
      },
      
      recall_at_50: {
        metric_name: 'recall_at_50',
        target_mean: 0.85,
        target_std: 0.043,
        threshold: 3.0,
        positive_sum: 0,
        negative_sum: 0,
        last_reset: new Date().toISOString(),
        alarm_active: false,
        consecutive_violations: 0,
        max_consecutive_violations: 5
      }
    };
  }
  
  /**
   * Collect current metrics snapshot
   */
  private async collectMetrics(): Promise<void> {
    try {
      // Mock metrics collection - in production would query actual APIs
      const snapshot: MetricSnapshot = {
        timestamp: new Date().toISOString(),
        
        // Core metrics with realistic variation
        anchor_p_at_1: 0.75 + (Math.random() - 0.5) * 0.1,
        recall_at_50: 0.85 + (Math.random() - 0.5) * 0.06,
        ndcg_at_10: 0.78 + (Math.random() - 0.5) * 0.08,
        
        // Ladder metrics
        ladder_positives_in_candidates: Math.floor(180 + Math.random() * 20),
        ladder_total_candidates: Math.floor(240 + Math.random() * 20),
        ladder_positive_rate: 0.75 + (Math.random() - 0.5) * 0.15,
        
        // Coverage metrics
        lsif_coverage: 0.95 + (Math.random() - 0.5) * 0.03,
        tree_sitter_coverage: 0.98 + (Math.random() - 0.5) * 0.02,
        total_spans: Math.floor(50000 + Math.random() * 5000),
        covered_spans: Math.floor(49000 + Math.random() * 1000),
        
        // Performance metrics
        p95_latency_ms: 150 + Math.random() * 50,
        p99_latency_ms: 280 + Math.random() * 70,
        qps: 50 + Math.random() * 20,
        error_rate: Math.random() * 0.005, // 0-0.5% error rate
        
        // Results metrics  
        results_per_query_mean: 5.2 + (Math.random() - 0.5) * 1.5,
        results_per_query_p95: 8.5 + Math.random() * 2,
        zero_result_rate: Math.random() * 0.02, // 0-2% zero results
        
        // System health
        memory_usage_gb: 4.5 + Math.random() * 1.5,
        cpu_utilization: 45 + Math.random() * 25,
        disk_usage_gb: 120 + Math.random() * 10
      };
      
      // Add to history
      this.monitoringState.metric_history.push(snapshot);
      
      // Keep only recent history (configurable retention)
      const retentionHours = this.monitoringState.dashboard_config.metric_retention_hours;
      const cutoffTime = Date.now() - retentionHours * 60 * 60 * 1000;
      this.monitoringState.metric_history = this.monitoringState.metric_history.filter(
        m => new Date(m.timestamp).getTime() > cutoffTime
      );
      
      // Update CUSUM detectors
      await this.updateCUSUMDetectors(snapshot);
      
      // Save current state
      this.saveMonitoringState();
      
      // Emit metrics for real-time dashboard
      this.emit('metrics_collected', snapshot);
      
    } catch (error) {
      console.error('❌ Failed to collect metrics:', error);
      this.emit('metrics_error', error);
    }
  }
  
  /**
   * Update CUSUM detectors with new metrics
   */
  private async updateCUSUMDetectors(metrics: MetricSnapshot): Promise<void> {
    for (const [name, detector] of Object.entries(this.monitoringState.cusum_detectors)) {
      const metricValue = this.getMetricValue(metrics, detector.metric_name);
      if (metricValue === undefined) continue;
      
      // Calculate standardized deviation
      const deviation = (metricValue - detector.target_mean) / detector.target_std;
      
      // Update CUSUM sums
      detector.positive_sum = Math.max(0, detector.positive_sum + deviation - 0.5);
      detector.negative_sum = Math.max(0, detector.negative_sum - deviation - 0.5);
      
      // Check for alarm condition
      const wasActive = detector.alarm_active;
      const shouldAlarm = detector.positive_sum > detector.threshold || detector.negative_sum > detector.threshold;
      
      if (shouldAlarm && !wasActive) {
        // Alarm triggered
        detector.alarm_active = true;
        detector.alarm_start_time = new Date().toISOString();
        detector.consecutive_violations = 1;
        
        console.log(`🚨 CUSUM alarm triggered: ${detector.metric_name}`);
        console.log(`  Current value: ${metricValue.toFixed(3)}, Target: ${detector.target_mean.toFixed(3)}`);
        console.log(`  Positive sum: ${detector.positive_sum.toFixed(2)}, Negative sum: ${detector.negative_sum.toFixed(2)}`);
        
        this.emit('cusum_alarm_triggered', {
          metric: detector.metric_name,
          value: metricValue,
          target: detector.target_mean,
          positive_sum: detector.positive_sum,
          negative_sum: detector.negative_sum,
          timestamp: detector.alarm_start_time
        });
        
      } else if (shouldAlarm && wasActive) {
        // Alarm continues
        detector.consecutive_violations++;
        
        if (detector.consecutive_violations >= detector.max_consecutive_violations) {
          console.log(`🚨🚨 SUSTAINED CUSUM VIOLATION: ${detector.metric_name} (${detector.consecutive_violations} violations)`);
          
          this.emit('sustained_cusum_violation', {
            metric: detector.metric_name,
            consecutive_violations: detector.consecutive_violations,
            alarm_duration_minutes: this.getAlarmDurationMinutes(detector.alarm_start_time!),
            current_value: metricValue
          });
        }
        
      } else if (!shouldAlarm && wasActive) {
        // Alarm cleared
        const alarmDuration = this.getAlarmDurationMinutes(detector.alarm_start_time!);
        
        detector.alarm_active = false;
        detector.alarm_start_time = undefined;
        detector.consecutive_violations = 0;
        detector.last_reset = new Date().toISOString();
        detector.positive_sum = 0;
        detector.negative_sum = 0;
        
        console.log(`✅ CUSUM alarm cleared: ${detector.metric_name} (duration: ${alarmDuration.toFixed(1)}min)`);
        
        this.emit('cusum_alarm_cleared', {
          metric: detector.metric_name,
          duration_minutes: alarmDuration,
          timestamp: new Date().toISOString()
        });
      }
    }
  }
  
  /**
   * Evaluate alert conditions
   */
  private async evaluateAlerts(): Promise<void> {
    if (this.monitoringState.metric_history.length === 0) return;
    
    const latestMetrics = this.monitoringState.metric_history[this.monitoringState.metric_history.length - 1];
    const currentTime = new Date();
    
    for (const condition of this.monitoringState.alert_conditions) {
      try {
        const isViolated = this.evaluateAlertCondition(condition, latestMetrics);
        const wasViolated = condition.current_violation_start !== undefined;
        
        if (isViolated && !wasViolated) {
          // Start new violation
          condition.current_violation_start = currentTime.toISOString();
          condition.violations_count++;
          
        } else if (isViolated && wasViolated) {
          // Continuing violation - check if sustained long enough
          const violationDurationMinutes = this.getMinutesSince(condition.current_violation_start!);
          
          if (violationDurationMinutes >= condition.sustained_minutes) {
            const shouldAlert = this.shouldSendAlert(condition, currentTime);
            
            if (shouldAlert) {
              await this.sendAlert(condition, latestMetrics, violationDurationMinutes);
            }
          }
          
        } else if (!isViolated && wasViolated) {
          // Violation cleared
          console.log(`✅ Alert condition cleared: ${condition.name}`);
          condition.current_violation_start = undefined;
          
          this.emit('alert_cleared', {
            condition: condition.name,
            timestamp: currentTime.toISOString()
          });
        }
        
      } catch (error) {
        console.error(`❌ Failed to evaluate alert condition ${condition.name}:`, error);
      }
    }
    
    // Update overall health status
    this.updateHealthStatus(latestMetrics);
  }
  
  /**
   * Evaluate single alert condition
   */
  private evaluateAlertCondition(condition: AlertCondition, metrics: MetricSnapshot): boolean {
    if (condition.condition === 'cusum_alarm') {
      // Check if any CUSUM detector is active for this metric
      const detector = this.monitoringState.cusum_detectors[condition.metric_path];
      return detector?.alarm_active || false;
    }
    
    const value = this.getMetricValue(metrics, condition.metric_path);
    if (value === undefined) return false;
    
    switch (condition.condition) {
      case 'above':
        return condition.threshold !== undefined && value > condition.threshold;
        
      case 'below':
        return condition.threshold !== undefined && value < condition.threshold;
        
      case 'outside_range':
        if (!condition.threshold_range) return false;
        const [min, max] = condition.threshold_range;
        return value < min || value > max;
        
      default:
        return false;
    }
  }
  
  /**
   * Check if alert should be sent (respecting cooldown)
   */
  private shouldSendAlert(condition: AlertCondition, currentTime: Date): boolean {
    if (!condition.last_alert_sent) return true;
    
    const minutesSinceLastAlert = this.getMinutesSince(condition.last_alert_sent);
    return minutesSinceLastAlert >= condition.cooldown_minutes;
  }
  
  /**
   * Send alert for condition
   */
  private async sendAlert(condition: AlertCondition, metrics: MetricSnapshot, violationDurationMinutes: number): Promise<void> {
    const alertTime = new Date().toISOString();
    condition.last_alert_sent = alertTime;
    
    const metricValue = this.getMetricValue(metrics, condition.metric_path);
    
    console.log(`🚨 ALERT: ${condition.name} (${condition.severity})`);
    console.log(`  Metric: ${condition.metric_path} = ${metricValue}`);
    console.log(`  Violation duration: ${violationDurationMinutes.toFixed(1)} minutes`);
    
    const alertData = {
      condition: condition.name,
      severity: condition.severity,
      metric_path: condition.metric_path,
      metric_value: metricValue,
      violation_duration_minutes: violationDurationMinutes,
      timestamp: alertTime,
      actions_taken: []
    };
    
    // Execute alert actions
    for (const action of condition.actions) {
      try {
        await this.executeAlertAction(action, alertData);
        alertData.actions_taken.push(action.type);
      } catch (error) {
        console.error(`❌ Failed to execute alert action ${action.type}:`, error);
      }
    }
    
    this.emit('alert_sent', alertData);
  }
  
  /**
   * Execute alert action
   */
  private async executeAlertAction(action: AlertAction, alertData: any): Promise<void> {
    switch (action.type) {
      case 'webhook':
        await this.sendWebhookAlert(action.config.url, alertData);
        break;
        
      case 'email':
        await this.sendEmailAlert(action.config, alertData);
        break;
        
      case 'pagerduty':
        await this.sendPagerDutyAlert(action.config, alertData);
        break;
        
      case 'kill_switch':
        await this.activateKillSwitch(action.config, alertData);
        break;
        
      case 'rollback':
        await this.triggerRollback(action.config, alertData);
        break;
        
      default:
        console.warn(`⚠️  Unknown alert action type: ${action.type}`);
    }
  }
  
  /**
   * Send webhook alert
   */
  private async sendWebhookAlert(url: string, alertData: any): Promise<void> {
    // Mock webhook - in production would use actual HTTP client
    console.log(`📡 Webhook alert sent to ${url}:`, JSON.stringify(alertData, null, 2));
  }
  
  /**
   * Send email alert
   */
  private async sendEmailAlert(config: any, alertData: any): Promise<void> {
    // Mock email - in production would use email service
    console.log(`📧 Email alert sent to ${config.recipients?.join(', ')}:`, alertData.condition);
  }
  
  /**
   * Send PagerDuty alert
   */
  private async sendPagerDutyAlert(config: any, alertData: any): Promise<void> {
    // Mock PagerDuty - in production would use PagerDuty API
    console.log(`📟 PagerDuty alert sent (service: ${config.service_key}):`, alertData.condition);
  }
  
  /**
   * Activate kill switch
   */
  private async activateKillSwitch(config: any, alertData: any): Promise<void> {
    console.log(`🚨🚨 KILL SWITCH ACTIVATED: ${alertData.condition}`);
    console.log(`🔧 Kill switch config:`, config);
    
    // Emit kill switch activation
    this.emit('kill_switch_activated', {
      trigger: alertData.condition,
      config,
      timestamp: new Date().toISOString()
    });
    
    // Could integrate with traffic routing or feature flags
    if (config.disable_feature) {
      console.log(`🚫 Feature disabled: ${config.disable_feature}`);
    }
    
    if (config.route_traffic_to_fallback) {
      console.log(`🔄 Traffic routed to fallback: ${config.route_traffic_to_fallback}`);
    }
  }
  
  /**
   * Trigger rollback
   */
  private async triggerRollback(config: any, alertData: any): Promise<void> {
    console.log(`🔄 ROLLBACK TRIGGERED: ${alertData.condition}`);
    
    this.emit('rollback_triggered', {
      trigger: alertData.condition,
      target_version: config.rollback_version,
      timestamp: new Date().toISOString()
    });
    
    // Could integrate with canary rollout system
    console.log(`📦 Rolling back to version: ${config.rollback_version || 'previous'}`);
  }
  
  /**
   * Update overall health status
   */
  private updateHealthStatus(metrics: MetricSnapshot): void {
    const activeAlarms = Object.entries(this.monitoringState.cusum_detectors)
      .filter(([_, detector]) => detector.alarm_active)
      .map(([name, _]) => name);
    
    const recentViolations = this.monitoringState.alert_conditions
      .filter(c => c.current_violation_start)
      .map(c => ({
        metric: c.metric_path,
        severity: c.severity,
        timestamp: c.current_violation_start!,
        value: this.getMetricValue(metrics, c.metric_path) || 0
      }));
    
    // Determine overall status
    let overallStatus: 'healthy' | 'degraded' | 'unhealthy' | 'critical' = 'healthy';
    
    if (recentViolations.some(v => v.severity === 'critical')) {
      overallStatus = 'critical';
    } else if (recentViolations.some(v => v.severity === 'high') || activeAlarms.length > 2) {
      overallStatus = 'unhealthy';
    } else if (recentViolations.length > 0 || activeAlarms.length > 0) {
      overallStatus = 'degraded';
    }
    
    this.monitoringState.health_status = {
      overall_status: overallStatus,
      active_alarms: activeAlarms,
      recent_violations: recentViolations,
      system_uptime_hours: this.getSystemUptimeHours(),
      last_health_check: new Date().toISOString()
    };
    
    // Emit health status updates
    this.emit('health_status_updated', this.monitoringState.health_status);
  }
  
  // Helper methods
  
  private getMetricValue(metrics: MetricSnapshot, metricPath: string): number | undefined {
    const path = metricPath.split('.');
    let value: any = metrics;
    
    for (const key of path) {
      value = value?.[key];
    }
    
    return typeof value === 'number' ? value : undefined;
  }
  
  private getAlarmDurationMinutes(startTime: string): number {
    return (Date.now() - new Date(startTime).getTime()) / (60 * 1000);
  }
  
  private getMinutesSince(timestamp: string): number {
    return (Date.now() - new Date(timestamp).getTime()) / (60 * 1000);
  }
  
  private getSystemUptimeHours(): number {
    // Mock uptime - in production would get actual system uptime
    return Math.random() * 720; // 0-30 days
  }
  
  /**
   * Initialize monitoring state with default configuration
   */
  private initializeMonitoringState(): MonitoringState {
    return {
      cusum_detectors: {},
      alert_conditions: [
        {
          name: 'High Error Rate',
          metric_path: 'error_rate',
          condition: 'above',
          threshold: 0.01, // 1%
          severity: 'high',
          sustained_minutes: 5,
          cooldown_minutes: 30,
          actions: [
            { type: 'webhook', config: { url: 'https://alerts.example.com/webhook' } },
            { type: 'pagerduty', config: { service_key: 'error_rate_alerts' } }
          ],
          violations_count: 0
        },
        
        {
          name: 'High Zero Result Rate',
          metric_path: 'zero_result_rate',
          condition: 'above',
          threshold: 0.05, // 5%
          severity: 'medium',
          sustained_minutes: 10,
          cooldown_minutes: 60,
          actions: [
            { type: 'webhook', config: { url: 'https://alerts.example.com/webhook' } }
          ],
          violations_count: 0
        },
        
        {
          name: 'Low LSIF Coverage',
          metric_path: 'lsif_coverage',
          condition: 'below',
          threshold: 0.9, // 90%
          severity: 'medium',
          sustained_minutes: 15,
          cooldown_minutes: 120,
          actions: [
            { type: 'email', config: { recipients: ['eng-alerts@example.com'] } }
          ],
          violations_count: 0
        },
        
        {
          name: 'P99 Latency Critical',
          metric_path: 'p99_latency_ms',
          condition: 'above',
          threshold: 5000, // 5 seconds
          severity: 'critical',
          sustained_minutes: 2,
          cooldown_minutes: 15,
          actions: [
            { type: 'pagerduty', config: { service_key: 'latency_critical' } },
            { type: 'kill_switch', config: { disable_feature: 'advanced_ranking' } }
          ],
          violations_count: 0
        },
        
        {
          name: 'Anchor P@1 CUSUM Alarm',
          metric_path: 'anchor_p_at_1',
          condition: 'cusum_alarm',
          severity: 'high',
          sustained_minutes: 60, // 1 hour of sustained drift
          cooldown_minutes: 240, // 4 hours between alerts
          actions: [
            { type: 'webhook', config: { url: 'https://alerts.example.com/drift' } },
            { type: 'email', config: { recipients: ['ml-team@example.com'] } }
          ],
          violations_count: 0
        },
        
        {
          name: 'Recall@50 CUSUM Alarm',
          metric_path: 'recall_at_50',
          condition: 'cusum_alarm',
          severity: 'high',
          sustained_minutes: 60,
          cooldown_minutes: 240,
          actions: [
            { type: 'webhook', config: { url: 'https://alerts.example.com/drift' } },
            { type: 'email', config: { recipients: ['ml-team@example.com'] } }
          ],
          violations_count: 0
        }
      ],
      metric_history: [],
      dashboard_config: {
        refresh_interval_seconds: 30,
        metric_retention_hours: 48,
        chart_configs: [
          {
            title: 'Core Search Metrics',
            metrics: ['anchor_p_at_1', 'recall_at_50', 'ndcg_at_10'],
            chart_type: 'line',
            time_range_hours: 24,
            alert_thresholds: { anchor_p_at_1: 0.7, recall_at_50: 0.8 }
          },
          {
            title: 'Coverage Metrics',
            metrics: ['lsif_coverage', 'tree_sitter_coverage'],
            chart_type: 'gauge',
            time_range_hours: 24,
            alert_thresholds: { lsif_coverage: 0.9, tree_sitter_coverage: 0.95 }
          },
          {
            title: 'Performance Metrics',
            metrics: ['p95_latency_ms', 'p99_latency_ms', 'qps'],
            chart_type: 'line',
            time_range_hours: 6
          },
          {
            title: 'System Health',
            metrics: ['error_rate', 'zero_result_rate', 'memory_usage_gb', 'cpu_utilization'],
            chart_type: 'line',
            time_range_hours: 12
          }
        ]
      },
      health_status: {
        overall_status: 'healthy',
        active_alarms: [],
        recent_violations: [],
        system_uptime_hours: 0,
        last_health_check: new Date().toISOString()
      }
    };
  }
  
  private saveMonitoringState(): void {
    const statePath = join(this.monitoringDir, 'monitoring_state.json');
    writeFileSync(statePath, JSON.stringify(this.monitoringState, null, 2));
  }
  
  /**
   * Get current health status
   */
  public getHealthStatus(): HealthStatus {
    return { ...this.monitoringState.health_status };
  }
  
  /**
   * Get dashboard data
   */
  public getDashboardData(): any {
    const recentMetrics = this.monitoringState.metric_history.slice(-100); // Last 100 points
    
    return {
      timestamp: new Date().toISOString(),
      health_status: this.monitoringState.health_status,
      active_cusum_alarms: Object.entries(this.monitoringState.cusum_detectors)
        .filter(([_, d]) => d.alarm_active)
        .map(([name, d]) => ({ name, start_time: d.alarm_start_time, violations: d.consecutive_violations })),
      recent_metrics: recentMetrics,
      chart_configs: this.monitoringState.dashboard_config.chart_configs
    };
  }
  
  /**
   * Manual CUSUM reset (emergency use)
   */
  public resetCUSUMDetector(metricName: string): void {
    const detector = this.monitoringState.cusum_detectors[metricName];
    if (detector) {
      detector.positive_sum = 0;
      detector.negative_sum = 0;
      detector.alarm_active = false;
      detector.alarm_start_time = undefined;
      detector.consecutive_violations = 0;
      detector.last_reset = new Date().toISOString();
      
      console.log(`🔧 CUSUM detector reset manually: ${metricName}`);
      this.emit('cusum_manual_reset', { metric: metricName, timestamp: detector.last_reset });
    }
  }
  
  /**
   * Get CUSUM detector status
   */
  public getCUSUMStatus(): Record<string, CUSUMDetector> {
    return { ...this.monitoringState.cusum_detectors };
  }
}

export const productionMonitoringSystem = new ProductionMonitoringSystem();