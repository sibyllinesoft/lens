/**
 * Comprehensive Production Monitoring Dashboard 
 * 
 * Integrates all monitoring components:
 * - Core production monitoring (CUSUM, alerts)
 * - Comprehensive drift monitoring (LSIF, RAPTOR, pressure, K-S drift)
 * - Week-one post-GA monitoring (stability validation)
 * - 3-stage breach response coordination
 * - Real-time dashboard with full system visibility
 */

import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';
import { EventEmitter } from 'events';
import { ProductionMonitoringSystem } from './production-monitoring-system.js';
import { ComprehensiveDriftMonitoringSystem } from './comprehensive-drift-monitoring-system.js';
import { WeekOnePostGAMonitoring } from './week-one-post-ga-monitoring.js';

interface DashboardMetrics {
  timestamp: string;
  system_status: 'healthy' | 'degraded' | 'unhealthy' | 'critical' | 'breach_response_active';
  
  // Core metrics
  core_metrics: {
    anchor_p_at_1: number;
    anchor_recall_at_50: number;
    ladder_positive_rate: number;
    ndcg_at_10: number;
    error_rate: number;
    p95_latency_ms: number;
    p99_latency_ms: number;
  };
  
  // Coverage & infrastructure
  infrastructure: {
    lsif_coverage: number;
    tree_sitter_coverage: number;
    total_spans: number;
    covered_spans: number;
    memory_usage_gb: number;
    cpu_utilization: number;
  };
  
  // RAPTOR & processing
  raptor_health: {
    staleness_p95_hours: number;
    stale_clusters_count: number;
    total_clusters: number;
    last_recluster_hours_ago: number;
    recluster_success_rate: number;
  };
  
  // Pressure & queues
  processing_pressure: {
    indexing_queue_depth: number;
    embedding_queue_depth: number;
    clustering_queue_depth: number;
    oldest_pending_hours: number;
    sla_breaches_last_hour: number;
    overall_pressure_ratio: number;
  };
  
  // Drift detection
  drift_status: {
    feature_ks_min_pvalue: number;
    feature_drift_magnitude_max: number;
    ks_breaches_count: number;
    drift_breach_active: boolean;
  };
  
  // CUSUM alarms
  cusum_alarms: {
    p_at_1_alarm: boolean;
    recall_at_50_alarm: boolean;
    lsif_coverage_alarm: boolean;
    tree_sitter_coverage_alarm: boolean;
    active_alarms_count: number;
  };
  
  // Breach response
  breach_response: {
    active: boolean;
    stage?: 'freeze_ltr' | 'disable_prior_boost' | 'disable_raptor_features';
    duration_minutes?: number;
    triggered_by?: string[];
    actions_active: {
      ltr_frozen: boolean;
      prior_boost_disabled: boolean;
      raptor_features_disabled: boolean;
    };
  };
  
  // Week-one monitoring (when active)
  week_one_status?: {
    monitoring_active: boolean;
    hours_monitored: number;
    hours_remaining: number;
    overall_stability: 'stable' | 'trending' | 'volatile' | 'unstable';
    raptor_rollout_ready: boolean;
    tentative_rollout_date?: string;
  };
}

interface DashboardAlert {
  id: string;
  timestamp: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  source: 'core' | 'drift' | 'week_one' | 'breach_response';
  title: string;
  message: string;
  acknowledged: boolean;
  auto_resolve: boolean;
  resolution_timestamp?: string;
}

interface DashboardHistoricalData {
  timestamp: string;
  core_metrics: {
    anchor_p_at_1: number;
    recall_at_50: number;
    ndcg_at_10: number;
    error_rate: number;
    p99_latency_ms: number;
  };
  coverage_metrics: {
    lsif_coverage: number;
    tree_sitter_coverage: number;
  };
  pressure_metrics: {
    total_queue_depth: number;
    oldest_pending_hours: number;
  };
  drift_metrics: {
    min_ks_pvalue: number;
    max_drift_magnitude: number;
  };
  system_status: string;
}

interface DashboardConfiguration {
  refresh_interval_seconds: number;
  history_retention_hours: number;
  alert_retention_days: number;
  
  // Chart configurations
  charts: Array<{
    id: string;
    title: string;
    type: 'line' | 'gauge' | 'status' | 'histogram' | 'heatmap';
    metrics: string[];
    time_range_hours: number;
    update_interval_seconds: number;
    alert_thresholds?: Record<string, { warning: number; critical: number }>;
  }>;
  
  // Alert routing
  alert_routing: {
    critical_webhook?: string;
    high_severity_email?: string[];
    breach_response_pagerduty?: string;
    drift_monitoring_slack?: string;
  };
}

export class ComprehensiveProductionDashboard extends EventEmitter {
  private readonly dashboardDir: string;
  private readonly productionMonitoring: ProductionMonitoringSystem;
  private readonly driftMonitoring: ComprehensiveDriftMonitoringSystem;
  private readonly weekOneMonitoring: WeekOnePostGAMonitoring;
  
  private dashboardConfig: DashboardConfiguration;
  private currentMetrics: DashboardMetrics;
  private activeAlerts: Map<string, DashboardAlert> = new Map();
  private historicalData: DashboardHistoricalData[] = [];
  
  private updateInterval?: NodeJS.Timeout;
  private historyInterval?: NodeJS.Timeout;
  private isRunning: boolean = false;
  
  constructor(
    dashboardDir: string = './deployment-artifacts/production-dashboard',
    productionMonitoring: ProductionMonitoringSystem,
    driftMonitoring: ComprehensiveDriftMonitoringSystem,
    weekOneMonitoring: WeekOnePostGAMonitoring
  ) {
    super();
    this.dashboardDir = dashboardDir;
    this.productionMonitoring = productionMonitoring;
    this.driftMonitoring = driftMonitoring;
    this.weekOneMonitoring = weekOneMonitoring;
    
    if (!existsSync(this.dashboardDir)) {
      mkdirSync(this.dashboardDir, { recursive: true });
    }
    
    this.dashboardConfig = this.initializeDashboardConfig();
    this.currentMetrics = this.initializeEmptyMetrics();
    
    // Set up event listeners for all monitoring systems
    this.setupEventListeners();
  }
  
  /**
   * Start comprehensive production dashboard
   */
  public async startDashboard(): Promise<void> {
    if (this.isRunning) {
      console.log('📊 Production dashboard already running');
      return;
    }
    
    console.log('🚀 Starting comprehensive production dashboard...');
    
    // Load existing data
    await this.loadHistoricalData();
    await this.loadActiveAlerts();
    
    // Start monitoring systems if not already running
    if (!this.productionMonitoring.getIsRunning()) {
      await this.productionMonitoring.startMonitoring();
    }
    
    if (!this.driftMonitoring.getIsRunning()) {
      await this.driftMonitoring.startDriftMonitoring();
    }
    
    // Start dashboard update loops
    this.updateInterval = setInterval(async () => {
      await this.updateDashboardMetrics();
    }, this.dashboardConfig.refresh_interval_seconds * 1000);
    
    this.historyInterval = setInterval(async () => {
      await this.captureHistoricalSnapshot();
    }, 60 * 1000); // Every minute
    
    // Initial metrics update
    await this.updateDashboardMetrics();
    
    this.isRunning = true;
    
    console.log('✅ Comprehensive production dashboard started');
    console.log(`📊 Dashboard URL: http://localhost:3000/dashboard`);
    console.log(`🔄 Refresh interval: ${this.dashboardConfig.refresh_interval_seconds}s`);
    console.log(`📈 Monitoring: Core metrics, drift detection, RAPTOR health, pressure backlog`);
    
    this.emit('dashboard_started');
  }
  
  /**
   * Stop production dashboard
   */
  public stopDashboard(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = undefined;
    }
    
    if (this.historyInterval) {
      clearInterval(this.historyInterval);
      this.historyInterval = undefined;
    }
    
    this.isRunning = false;
    console.log('🛑 Comprehensive production dashboard stopped');
    this.emit('dashboard_stopped');
  }
  
  /**
   * Set up event listeners for all monitoring systems
   */
  private setupEventListeners(): void {
    // Production monitoring events
    this.productionMonitoring.on('cusum_alarm_triggered', (data) => {
      this.createAlert({
        id: `cusum_${data.metric}_${Date.now()}`,
        severity: 'high',
        source: 'core',
        title: `CUSUM Alarm: ${data.metric}`,
        message: `Drift detected in ${data.metric}: current=${data.value.toFixed(3)}, target=${data.target.toFixed(3)}`,
        auto_resolve: true
      });
    });
    
    this.productionMonitoring.on('alert_sent', (alertData) => {
      this.createAlert({
        id: `alert_${alertData.condition}_${Date.now()}`,
        severity: alertData.severity as any,
        source: 'core',
        title: `Production Alert: ${alertData.condition}`,
        message: `${alertData.metric_path} = ${alertData.metric_value} (violation for ${alertData.violation_duration_minutes.toFixed(1)}min)`,
        auto_resolve: false
      });
    });
    
    // Drift monitoring events
    this.driftMonitoring.on('breach_response_triggered', (response) => {
      this.createAlert({
        id: `breach_${response.stage}_${Date.now()}`,
        severity: 'critical',
        source: 'breach_response',
        title: `BREACH RESPONSE ACTIVATED: ${response.stage.toUpperCase()}`,
        message: `Triggered by: ${response.triggered_by.join(', ')}. Actions: ${this.formatBreachActions(response)}`,
        auto_resolve: false
      });
    });
    
    this.driftMonitoring.on('coverage_drop_detected', (data) => {
      this.createAlert({
        id: `coverage_drop_${Date.now()}`,
        severity: 'medium',
        source: 'drift',
        title: 'Coverage Drop Detected',
        message: `LSIF: ${(data.lsif_coverage * 100).toFixed(1)}%, Tree-sitter: ${(data.tree_sitter_coverage * 100).toFixed(1)}%`,
        auto_resolve: true
      });
    });
    
    this.driftMonitoring.on('feature_ks_drift_breach', (data) => {
      this.createAlert({
        id: `ks_drift_${Date.now()}`,
        severity: 'high',
        source: 'drift',
        title: 'Feature K-S Drift Breach',
        message: `Drift detected in: ${data.breaches.join(', ')}`,
        auto_resolve: true
      });
    });
    
    // Week-one monitoring events
    this.weekOneMonitoring.on('raptor_rollout_ready', (data) => {
      this.createAlert({
        id: `raptor_ready_${Date.now()}`,
        severity: 'low',
        source: 'week_one',
        title: 'RAPTOR Rollout Ready',
        message: `Stability achieved after ${data.hours_to_readiness.toFixed(1)}h. Tentative start: ${data.tentative_start}`,
        auto_resolve: false
      });
    });
    
    this.weekOneMonitoring.on('cusum_alarm_triggered', (data) => {
      this.createAlert({
        id: `week_one_cusum_${data.metric}_${Date.now()}`,
        severity: 'high',
        source: 'week_one',
        title: `Week-One CUSUM Alarm: ${data.metric}`,
        message: `Post-GA drift detected in ${data.metric}: ${data.value.toFixed(3)}`,
        auto_resolve: true
      });
    });
  }
  
  /**
   * Update dashboard metrics from all monitoring systems
   */
  private async updateDashboardMetrics(): Promise<void> {
    try {
      // Get data from all monitoring systems
      const coreHealth = this.productionMonitoring.getHealthStatus();
      const coreDashboard = this.productionMonitoring.getDashboardData();
      const driftStatus = this.driftMonitoring.getDriftStatus();
      const driftDashboard = this.driftMonitoring.getDriftDashboardData();
      
      // Week-one monitoring (if active)
      let weekOneStatus = undefined;
      try {
        const weekOneState = this.weekOneMonitoring.getMonitoringStatus();
        if (weekOneState.monitoring_active) {
          const weekOneDash = this.weekOneMonitoring.getWeekOneDashboardData();
          weekOneStatus = {
            monitoring_active: true,
            hours_monitored: weekOneDash.progress.hours_monitored,
            hours_remaining: weekOneDash.progress.hours_remaining,
            overall_stability: weekOneDash.stability.overall_status,
            raptor_rollout_ready: weekOneDash.raptor_rollout.ready,
            tentative_rollout_date: weekOneDash.raptor_rollout.tentative_start
          };
        }
      } catch (error) {
        // Week-one monitoring not active
      }
      
      // Get latest metrics from historical data
      const latestCore = coreDashboard.recent_metrics[coreDashboard.recent_metrics.length - 1];
      
      // Determine overall system status
      let systemStatus: DashboardMetrics['system_status'] = 'healthy';
      
      if (driftStatus.active_breach_response) {
        systemStatus = 'breach_response_active';
      } else if (coreHealth.overall_status === 'critical') {
        systemStatus = 'critical';
      } else if (coreHealth.overall_status === 'unhealthy') {
        systemStatus = 'unhealthy';
      } else if (coreHealth.overall_status === 'degraded') {
        systemStatus = 'degraded';
      }
      
      // Build comprehensive metrics
      this.currentMetrics = {
        timestamp: new Date().toISOString(),
        system_status: systemStatus,
        
        core_metrics: {
          anchor_p_at_1: latestCore?.anchor_p_at_1 || 0.75,
          anchor_recall_at_50: latestCore?.recall_at_50 || 0.85,
          ladder_positive_rate: latestCore?.ladder_positive_rate || 0.75,
          ndcg_at_10: latestCore?.ndcg_at_10 || 0.78,
          error_rate: latestCore?.error_rate || 0.001,
          p95_latency_ms: latestCore?.p95_latency_ms || 150,
          p99_latency_ms: latestCore?.p99_latency_ms || 280
        },
        
        infrastructure: {
          lsif_coverage: driftStatus.lsif_tree_sitter_coverage.lsif_coverage,
          tree_sitter_coverage: driftStatus.lsif_tree_sitter_coverage.tree_sitter_coverage,
          total_spans: driftStatus.lsif_tree_sitter_coverage.total_spans,
          covered_spans: driftStatus.lsif_tree_sitter_coverage.covered_spans,
          memory_usage_gb: latestCore?.memory_usage_gb || 5.0,
          cpu_utilization: latestCore?.cpu_utilization || 50
        },
        
        raptor_health: {
          staleness_p95_hours: driftStatus.raptor_staleness.p95_staleness_hours,
          stale_clusters_count: driftStatus.raptor_staleness.stale_clusters,
          total_clusters: driftStatus.raptor_staleness.total_clusters,
          last_recluster_hours_ago: (Date.now() - new Date(driftStatus.raptor_staleness.last_recluster_timestamp).getTime()) / (60 * 60 * 1000),
          recluster_success_rate: driftStatus.raptor_staleness.recluster_success_rate
        },
        
        processing_pressure: {
          indexing_queue_depth: driftStatus.pressure_backlog.indexing_queue_depth,
          embedding_queue_depth: driftStatus.pressure_backlog.embedding_queue_depth,
          clustering_queue_depth: driftStatus.pressure_backlog.clustering_queue_depth,
          oldest_pending_hours: driftStatus.pressure_backlog.oldest_pending_item_hours,
          sla_breaches_last_hour: driftStatus.pressure_backlog.sla_breaches_last_hour,
          overall_pressure_ratio: Math.max(
            driftStatus.pressure_backlog.memory_pressure_ratio,
            driftStatus.pressure_backlog.cpu_pressure_ratio,
            driftStatus.pressure_backlog.io_pressure_ratio
          )
        },
        
        drift_status: {
          feature_ks_min_pvalue: Math.min(
            driftStatus.feature_ks_drift.ltr_feature_ks_pvalue,
            driftStatus.feature_ks_drift.raptor_prior_ks_pvalue,
            driftStatus.feature_ks_drift.semantic_similarity_ks_pvalue,
            driftStatus.feature_ks_drift.lexical_score_ks_pvalue
          ),
          feature_drift_magnitude_max: Math.max(
            driftStatus.feature_ks_drift.ltr_feature_drift_magnitude,
            driftStatus.feature_ks_drift.raptor_drift_magnitude,
            driftStatus.feature_ks_drift.semantic_drift_magnitude,
            driftStatus.feature_ks_drift.lexical_drift_magnitude
          ),
          ks_breaches_count: [
            driftStatus.feature_ks_drift.ltr_feature_ks_pvalue,
            driftStatus.feature_ks_drift.raptor_prior_ks_pvalue,
            driftStatus.feature_ks_drift.semantic_similarity_ks_pvalue,
            driftStatus.feature_ks_drift.lexical_score_ks_pvalue
          ].filter(p => p < driftStatus.breach_thresholds.ks_drift_pvalue_threshold).length,
          drift_breach_active: driftStatus.active_breach_response !== undefined
        },
        
        cusum_alarms: {
          p_at_1_alarm: false, // TODO: Get from production monitoring CUSUM status
          recall_at_50_alarm: false,
          lsif_coverage_alarm: false,
          tree_sitter_coverage_alarm: false,
          active_alarms_count: coreHealth.active_alarms.length
        },
        
        breach_response: {
          active: driftStatus.active_breach_response !== undefined,
          stage: driftStatus.active_breach_response?.stage,
          duration_minutes: driftStatus.active_breach_response ? 
            (Date.now() - new Date(driftStatus.active_breach_response.timestamp).getTime()) / (60 * 1000) : undefined,
          triggered_by: driftStatus.active_breach_response?.triggered_by,
          actions_active: {
            ltr_frozen: driftStatus.active_breach_response?.ltr_frozen || false,
            prior_boost_disabled: driftStatus.active_breach_response?.prior_boost_disabled || false,
            raptor_features_disabled: driftStatus.active_breach_response?.raptor_features_disabled || false
          }
        },
        
        week_one_status: weekOneStatus
      };
      
      // Emit metrics update event
      this.emit('metrics_updated', this.currentMetrics);
      
    } catch (error) {
      console.error('❌ Failed to update dashboard metrics:', error);
      this.emit('dashboard_update_error', error);
    }
  }
  
  /**
   * Capture historical snapshot
   */
  private async captureHistoricalSnapshot(): Promise<void> {
    const snapshot: DashboardHistoricalData = {
      timestamp: new Date().toISOString(),
      core_metrics: {
        anchor_p_at_1: this.currentMetrics.core_metrics.anchor_p_at_1,
        recall_at_50: this.currentMetrics.core_metrics.anchor_recall_at_50,
        ndcg_at_10: this.currentMetrics.core_metrics.ndcg_at_10,
        error_rate: this.currentMetrics.core_metrics.error_rate,
        p99_latency_ms: this.currentMetrics.core_metrics.p99_latency_ms
      },
      coverage_metrics: {
        lsif_coverage: this.currentMetrics.infrastructure.lsif_coverage,
        tree_sitter_coverage: this.currentMetrics.infrastructure.tree_sitter_coverage
      },
      pressure_metrics: {
        total_queue_depth: this.currentMetrics.processing_pressure.indexing_queue_depth + 
                          this.currentMetrics.processing_pressure.embedding_queue_depth + 
                          this.currentMetrics.processing_pressure.clustering_queue_depth,
        oldest_pending_hours: this.currentMetrics.processing_pressure.oldest_pending_hours
      },
      drift_metrics: {
        min_ks_pvalue: this.currentMetrics.drift_status.feature_ks_min_pvalue,
        max_drift_magnitude: this.currentMetrics.drift_status.feature_drift_magnitude_max
      },
      system_status: this.currentMetrics.system_status
    };
    
    this.historicalData.push(snapshot);
    
    // Keep only configured retention period
    const cutoff = Date.now() - this.dashboardConfig.history_retention_hours * 60 * 60 * 1000;
    this.historicalData = this.historicalData.filter(s => new Date(s.timestamp).getTime() > cutoff);
    
    // Save historical data
    this.saveHistoricalData();
  }
  
  /**
   * Create new alert
   */
  private createAlert(alert: Partial<DashboardAlert>): void {
    const fullAlert: DashboardAlert = {
      id: alert.id || `alert_${Date.now()}`,
      timestamp: new Date().toISOString(),
      severity: alert.severity || 'medium',
      source: alert.source || 'core',
      title: alert.title || 'System Alert',
      message: alert.message || 'Alert condition detected',
      acknowledged: false,
      auto_resolve: alert.auto_resolve || false,
      ...alert
    };
    
    this.activeAlerts.set(fullAlert.id, fullAlert);
    
    console.log(`🚨 DASHBOARD ALERT [${fullAlert.severity.toUpperCase()}]: ${fullAlert.title}`);
    console.log(`   ${fullAlert.message}`);
    
    // Save alerts
    this.saveActiveAlerts();
    
    // Emit alert event
    this.emit('alert_created', fullAlert);
  }
  
  /**
   * Format breach response actions
   */
  private formatBreachActions(response: any): string {
    const actions = [];
    if (response.ltr_frozen) actions.push('LTR frozen');
    if (response.prior_boost_disabled) actions.push('Prior boost disabled');
    if (response.raptor_features_disabled) actions.push('RAPTOR disabled');
    return actions.join(', ') || 'None';
  }
  
  /**
   * Get current dashboard data (API endpoint)
   */
  public getDashboardData(): any {
    return {
      timestamp: new Date().toISOString(),
      status: this.currentMetrics.system_status,
      
      // Current metrics
      metrics: this.currentMetrics,
      
      // Active alerts
      alerts: {
        active_count: this.activeAlerts.size,
        by_severity: {
          critical: Array.from(this.activeAlerts.values()).filter(a => a.severity === 'critical').length,
          high: Array.from(this.activeAlerts.values()).filter(a => a.severity === 'high').length,
          medium: Array.from(this.activeAlerts.values()).filter(a => a.severity === 'medium').length,
          low: Array.from(this.activeAlerts.values()).filter(a => a.severity === 'low').length
        },
        recent_alerts: Array.from(this.activeAlerts.values())
          .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
          .slice(0, 10) // Last 10 alerts
      },
      
      // Historical data for charts
      historical: {
        last_24h: this.historicalData.filter(s => {
          const age = Date.now() - new Date(s.timestamp).getTime();
          return age <= 24 * 60 * 60 * 1000; // Last 24 hours
        }),
        last_1h: this.historicalData.filter(s => {
          const age = Date.now() - new Date(s.timestamp).getTime();
          return age <= 60 * 60 * 1000; // Last 1 hour
        })
      },
      
      // System health summary
      health_summary: {
        overall_status: this.currentMetrics.system_status,
        core_systems_healthy: this.currentMetrics.core_metrics.error_rate < 0.01,
        infrastructure_healthy: this.currentMetrics.infrastructure.lsif_coverage > 0.9 && 
                               this.currentMetrics.infrastructure.tree_sitter_coverage > 0.95,
        raptor_healthy: this.currentMetrics.raptor_health.staleness_p95_hours < 18,
        pressure_normal: this.currentMetrics.processing_pressure.overall_pressure_ratio < 0.7,
        drift_controlled: this.currentMetrics.drift_status.ks_breaches_count === 0,
        breach_response_active: this.currentMetrics.breach_response.active
      },
      
      // Configuration
      config: this.dashboardConfig
    };
  }
  
  /**
   * Acknowledge alert
   */
  public acknowledgeAlert(alertId: string, acknowledgedBy: string = 'system'): boolean {
    const alert = this.activeAlerts.get(alertId);
    if (!alert) return false;
    
    alert.acknowledged = true;
    console.log(`✅ Alert acknowledged: ${alert.title} (by ${acknowledgedBy})`);
    
    this.saveActiveAlerts();
    this.emit('alert_acknowledged', { alert, acknowledged_by: acknowledgedBy });
    
    return true;
  }
  
  /**
   * Resolve alert
   */
  public resolveAlert(alertId: string, resolvedBy: string = 'system'): boolean {
    const alert = this.activeAlerts.get(alertId);
    if (!alert) return false;
    
    alert.resolution_timestamp = new Date().toISOString();
    this.activeAlerts.delete(alertId);
    
    console.log(`✅ Alert resolved: ${alert.title} (by ${resolvedBy})`);
    
    this.saveActiveAlerts();
    this.emit('alert_resolved', { alert, resolved_by: resolvedBy });
    
    return true;
  }
  
  private initializeDashboardConfig(): DashboardConfiguration {
    return {
      refresh_interval_seconds: 10,
      history_retention_hours: 168, // 7 days
      alert_retention_days: 30,
      
      charts: [
        {
          id: 'core_metrics',
          title: 'Core Search Metrics',
          type: 'line',
          metrics: ['anchor_p_at_1', 'anchor_recall_at_50', 'ndcg_at_10'],
          time_range_hours: 24,
          update_interval_seconds: 30,
          alert_thresholds: {
            anchor_p_at_1: { warning: 0.70, critical: 0.65 },
            anchor_recall_at_50: { warning: 0.80, critical: 0.75 }
          }
        },
        {
          id: 'system_health',
          title: 'System Health',
          type: 'gauge',
          metrics: ['error_rate', 'p99_latency_ms', 'cpu_utilization'],
          time_range_hours: 6,
          update_interval_seconds: 10,
          alert_thresholds: {
            error_rate: { warning: 0.01, critical: 0.05 },
            p99_latency_ms: { warning: 1000, critical: 5000 }
          }
        },
        {
          id: 'coverage_status',
          title: 'Coverage & Infrastructure',
          type: 'gauge',
          metrics: ['lsif_coverage', 'tree_sitter_coverage'],
          time_range_hours: 12,
          update_interval_seconds: 60,
          alert_thresholds: {
            lsif_coverage: { warning: 0.90, critical: 0.85 },
            tree_sitter_coverage: { warning: 0.95, critical: 0.90 }
          }
        },
        {
          id: 'raptor_health',
          title: 'RAPTOR Health',
          type: 'line',
          metrics: ['staleness_p95_hours', 'stale_clusters_pct', 'recluster_success_rate'],
          time_range_hours: 24,
          update_interval_seconds: 300 // 5 minutes
        },
        {
          id: 'pressure_monitoring',
          title: 'Processing Pressure',
          type: 'line',
          metrics: ['total_queue_depth', 'oldest_pending_hours', 'sla_breaches_last_hour'],
          time_range_hours: 12,
          update_interval_seconds: 60
        },
        {
          id: 'drift_detection',
          title: 'Drift Detection',
          type: 'line',
          metrics: ['min_ks_pvalue', 'max_drift_magnitude', 'ks_breaches_count'],
          time_range_hours: 48,
          update_interval_seconds: 300
        }
      ],
      
      alert_routing: {
        critical_webhook: 'https://alerts.company.com/critical',
        high_severity_email: ['oncall@company.com', 'ml-team@company.com'],
        breach_response_pagerduty: 'breach-response-service-key',
        drift_monitoring_slack: '#ml-alerts'
      }
    };
  }
  
  private initializeEmptyMetrics(): DashboardMetrics {
    return {
      timestamp: new Date().toISOString(),
      system_status: 'healthy',
      core_metrics: {
        anchor_p_at_1: 0,
        anchor_recall_at_50: 0,
        ladder_positive_rate: 0,
        ndcg_at_10: 0,
        error_rate: 0,
        p95_latency_ms: 0,
        p99_latency_ms: 0
      },
      infrastructure: {
        lsif_coverage: 0,
        tree_sitter_coverage: 0,
        total_spans: 0,
        covered_spans: 0,
        memory_usage_gb: 0,
        cpu_utilization: 0
      },
      raptor_health: {
        staleness_p95_hours: 0,
        stale_clusters_count: 0,
        total_clusters: 0,
        last_recluster_hours_ago: 0,
        recluster_success_rate: 0
      },
      processing_pressure: {
        indexing_queue_depth: 0,
        embedding_queue_depth: 0,
        clustering_queue_depth: 0,
        oldest_pending_hours: 0,
        sla_breaches_last_hour: 0,
        overall_pressure_ratio: 0
      },
      drift_status: {
        feature_ks_min_pvalue: 1,
        feature_drift_magnitude_max: 0,
        ks_breaches_count: 0,
        drift_breach_active: false
      },
      cusum_alarms: {
        p_at_1_alarm: false,
        recall_at_50_alarm: false,
        lsif_coverage_alarm: false,
        tree_sitter_coverage_alarm: false,
        active_alarms_count: 0
      },
      breach_response: {
        active: false,
        actions_active: {
          ltr_frozen: false,
          prior_boost_disabled: false,
          raptor_features_disabled: false
        }
      }
    };
  }
  
  private async loadHistoricalData(): Promise<void> {
    const historyPath = join(this.dashboardDir, 'historical_data.json');
    if (existsSync(historyPath)) {
      try {
        const data = JSON.parse(readFileSync(historyPath, 'utf-8'));
        this.historicalData = data;
        console.log(`📈 Loaded ${this.historicalData.length} historical data points`);
      } catch (error) {
        console.warn('⚠️  Failed to load historical data:', error);
      }
    }
  }
  
  private saveHistoricalData(): void {
    const historyPath = join(this.dashboardDir, 'historical_data.json');
    writeFileSync(historyPath, JSON.stringify(this.historicalData, null, 2));
  }
  
  private async loadActiveAlerts(): Promise<void> {
    const alertsPath = join(this.dashboardDir, 'active_alerts.json');
    if (existsSync(alertsPath)) {
      try {
        const data = JSON.parse(readFileSync(alertsPath, 'utf-8'));
        this.activeAlerts = new Map(Object.entries(data));
        console.log(`🚨 Loaded ${this.activeAlerts.size} active alerts`);
      } catch (error) {
        console.warn('⚠️  Failed to load active alerts:', error);
      }
    }
  }
  
  private saveActiveAlerts(): void {
    const alertsPath = join(this.dashboardDir, 'active_alerts.json');
    const alertsObj = Object.fromEntries(this.activeAlerts);
    writeFileSync(alertsPath, JSON.stringify(alertsObj, null, 2));
  }
}

// Factory function for easy instantiation
export async function createComprehensiveProductionDashboard(dashboardDir?: string): Promise<ComprehensiveProductionDashboard> {
  const { productionMonitoringSystem } = await import('./production-monitoring-system.js');
  const { comprehensiveDriftMonitoringSystem } = await import('./comprehensive-drift-monitoring-system.js');
  const { weekOnePostGAMonitoring } = await import('./week-one-post-ga-monitoring.js');
  
  return new ComprehensiveProductionDashboard(
    dashboardDir,
    productionMonitoringSystem,
    comprehensiveDriftMonitoringSystem,
    weekOnePostGAMonitoring
  );
}