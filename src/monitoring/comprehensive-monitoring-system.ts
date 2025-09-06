import { EventEmitter } from 'events';
import { promises as fs } from 'fs';
import { join } from 'path';

interface MonitoringConfig {
  why_mix: {
    semantic_share_jump_max: number;
    quality_lift_required: boolean;
    baseline_semantic_share: number;
  };
  ece_calibration: {
    max_threshold: number;
    slope_tolerance: number;
    intercept_tolerance: number;
    refit_trigger: boolean;
  };
  lsp_coverage: {
    drop_threshold: number;
    refresh_trigger: boolean;
    baseline_coverage: number;
  };
  drift_detection: {
    p95_drift_alarm: number;
    coverage_drop_alarm: number;
    monitoring_window_hours: number;
  };
  auto_remediation: {
    enabled: boolean;
    max_retries: number;
    cooldown_minutes: number;
  };
}

interface SystemMetrics {
  timestamp: number;
  
  // Core quality metrics
  ndcg_delta: number;
  sla_recall_50: number;
  p_at_1_symbol: number;
  p_at_1_nl: number;
  
  // Performance metrics
  p95_latency: number;
  p99_latency: number;
  qps_150ms: number;
  error_rate: number;
  
  // Why-mix composition
  why_mix: {
    semantic_share: number;
    lexical_share: number;
    structural_share: number;
    quality_lift_correlation: number;
  };
  
  // Calibration metrics
  calibration: {
    ece: number; // Expected Calibration Error
    slope: number;
    intercept: number;
    confidence_bins: number[];
    accuracy_bins: number[];
  };
  
  // LSP and coverage metrics
  lsp_coverage: number;
  span_coverage: number;
  sentinel_nzc: number;
  
  // Router metrics
  router: {
    upshift_rate: number;
    paired_ndcg_delta: number;
    confidence_score: number;
  };
  
  // ANN performance
  ann_performance: {
    recall_50_sla: number;
    ef_search_efficiency: number;
    quantization_error: number;
  };
}

interface AlertEvent {
  timestamp: number;
  severity: 'info' | 'warning' | 'critical';
  category: 'quality' | 'performance' | 'calibration' | 'coverage' | 'drift';
  message: string;
  metrics: Partial<SystemMetrics>;
  auto_remediation_triggered: boolean;
}

interface RemediationAction {
  action_type: 'isotonic_refit' | 'lsp_refresh' | 'cache_clear' | 'traffic_reduce';
  timestamp: number;
  trigger_reason: string;
  success: boolean;
  error_message?: string;
  metrics_before: Partial<SystemMetrics>;
  metrics_after?: Partial<SystemMetrics>;
}

export class ComprehensiveMonitoringSystem extends EventEmitter {
  private config: MonitoringConfig;
  private metricsHistory: SystemMetrics[] = [];
  private alertHistory: AlertEvent[] = [];
  private remediationHistory: RemediationAction[] = [];
  private monitoringTimer: NodeJS.Timeout | null = null;
  private remediationCooldowns: Map<string, number> = new Map();

  constructor(config: MonitoringConfig) {
    super();
    this.config = config;
  }

  async startMonitoring(): Promise<void> {
    this.emit('monitoring_start', {
      timestamp: Date.now(),
      config: this.config
    });

    // Start continuous monitoring at 1-minute intervals
    this.monitoringTimer = setInterval(async () => {
      try {
        await this.collectAndAnalyzeMetrics();
      } catch (error) {
        this.emit('monitoring_error', {
          timestamp: Date.now(),
          error: error.message
        });
      }
    }, 60000); // 1 minute

    this.emit('log', {
      level: 'info',
      message: 'Comprehensive monitoring system started',
      timestamp: Date.now()
    });
  }

  async stopMonitoring(): Promise<void> {
    if (this.monitoringTimer) {
      clearInterval(this.monitoringTimer);
      this.monitoringTimer = null;
    }

    this.emit('monitoring_stop', {
      timestamp: Date.now(),
      total_samples: this.metricsHistory.length,
      total_alerts: this.alertHistory.length
    });
  }

  private async collectAndAnalyzeMetrics(): Promise<void> {
    // Collect current system metrics
    const metrics = await this.collectSystemMetrics();
    this.metricsHistory.push(metrics);

    // Keep last 24 hours of data (1440 samples at 1-minute intervals)
    if (this.metricsHistory.length > 1440) {
      this.metricsHistory = this.metricsHistory.slice(-1440);
    }

    // Analyze each monitoring category
    await Promise.all([
      this.analyzeWhyMix(metrics),
      this.analyzeCalibration(metrics),
      this.analyzeLSPCoverage(metrics),
      this.analyzeDriftDetection(metrics),
      this.analyzeRouterPerformance(metrics),
      this.analyzeANNPerformance(metrics)
    ]);

    this.emit('metrics_collected', {
      timestamp: metrics.timestamp,
      metrics,
      alerts_count: this.alertHistory.length
    });
  }

  private async collectSystemMetrics(): Promise<SystemMetrics> {
    // In real implementation, this would query actual system monitoring APIs
    // For now, simulate realistic metrics based on Gemma-256 performance
    return {
      timestamp: Date.now(),
      
      // Core quality (maintaining canary performance)
      ndcg_delta: 3.8 + (Math.random() - 0.5) * 0.4, // Around +3.8pp
      sla_recall_50: 0.87 + (Math.random() - 0.5) * 0.02,
      p_at_1_symbol: 0.76 + (Math.random() - 0.5) * 0.03,
      p_at_1_nl: 0.74 + (Math.random() - 0.5) * 0.03,
      
      // Performance (maintaining SLA)
      p95_latency: 20 + Math.random() * 4, // Under 25ms SLA
      p99_latency: 38 + Math.random() * 8, // Good p99/p95 ratio
      qps_150ms: 1.35 + (Math.random() - 0.5) * 0.05, // Maintaining improvement
      error_rate: 0.005 + Math.random() * 0.003, // Under 1% SLA
      
      // Why-mix analysis
      why_mix: {
        semantic_share: 65 + Math.random() * 10, // Baseline ~60%
        lexical_share: 25 + Math.random() * 5,
        structural_share: 10 + Math.random() * 3,
        quality_lift_correlation: 0.7 + Math.random() * 0.2
      },
      
      // Calibration monitoring
      calibration: {
        ece: 0.03 + Math.random() * 0.015, // Target ≤0.05
        slope: 0.95 + (Math.random() - 0.5) * 0.1,
        intercept: 0.02 + (Math.random() - 0.5) * 0.02,
        confidence_bins: this.generateCalibrationBins(10),
        accuracy_bins: this.generateCalibrationBins(10)
      },
      
      // Coverage metrics
      lsp_coverage: 98.5 + Math.random() * 1.0, // Baseline ~99%
      span_coverage: 100.0, // Should always be 100%
      sentinel_nzc: 99.2 + Math.random() * 0.5,
      
      // Router performance
      router: {
        upshift_rate: 0.05 + (Math.random() - 0.5) * 0.015, // Target 5% ±2pp
        paired_ndcg_delta: 0.03 + Math.random() * 0.01, // Target ≥3pp
        confidence_score: 0.85 + Math.random() * 0.1
      },
      
      // ANN performance
      ann_performance: {
        recall_50_sla: 0.856 + Math.random() * 0.02, // Baseline maintenance
        ef_search_efficiency: 0.95 + Math.random() * 0.03,
        quantization_error: 0.02 + Math.random() * 0.01
      }
    };
  }

  private generateCalibrationBins(count: number): number[] {
    return Array.from({length: count}, () => Math.random());
  }

  private async analyzeWhyMix(metrics: SystemMetrics): Promise<void> {
    const semanticJump = Math.abs(metrics.why_mix.semantic_share - this.config.why_mix.baseline_semantic_share);
    
    if (semanticJump > this.config.why_mix.semantic_share_jump_max) {
      const qualityLiftExists = metrics.why_mix.quality_lift_correlation > 0.5;
      
      if (!qualityLiftExists && this.config.why_mix.quality_lift_required) {
        await this.triggerAlert({
          timestamp: metrics.timestamp,
          severity: 'warning',
          category: 'quality',
          message: `Why-mix semantic share jumped ${semanticJump.toFixed(1)}pp without quality lift (correlation: ${metrics.why_mix.quality_lift_correlation.toFixed(2)})`,
          metrics: { why_mix: metrics.why_mix },
          auto_remediation_triggered: false
        });
      }
    }
  }

  private async analyzeCalibration(metrics: SystemMetrics): Promise<void> {
    const { ece, slope, intercept } = metrics.calibration;
    
    // ECE threshold check
    if (ece > this.config.ece_calibration.max_threshold) {
      await this.triggerAlert({
        timestamp: metrics.timestamp,
        severity: 'critical',
        category: 'calibration',
        message: `ECE calibration exceeded threshold: ${ece.toFixed(3)} > ${this.config.ece_calibration.max_threshold}`,
        metrics: { calibration: metrics.calibration },
        auto_remediation_triggered: await this.triggerRemediation('isotonic_refit', 'ECE calibration drift', metrics)
      });
    }
    
    // Slope/intercept drift check
    const slopeDrift = Math.abs(slope - 1.0);
    const interceptDrift = Math.abs(intercept);
    
    if (slopeDrift > this.config.ece_calibration.slope_tolerance || 
        interceptDrift > this.config.ece_calibration.intercept_tolerance) {
      await this.triggerAlert({
        timestamp: metrics.timestamp,
        severity: 'warning',
        category: 'calibration',
        message: `Calibration drift detected - Slope: ${slope.toFixed(3)} (drift: ${slopeDrift.toFixed(3)}), Intercept: ${intercept.toFixed(3)} (drift: ${interceptDrift.toFixed(3)})`,
        metrics: { calibration: metrics.calibration },
        auto_remediation_triggered: await this.triggerRemediation('isotonic_refit', 'Calibration slope/intercept drift', metrics)
      });
    }
  }

  private async analyzeLSPCoverage(metrics: SystemMetrics): Promise<void> {
    const coverageDrop = this.config.lsp_coverage.baseline_coverage - metrics.lsp_coverage;
    
    if (coverageDrop > this.config.lsp_coverage.drop_threshold) {
      await this.triggerAlert({
        timestamp: metrics.timestamp,
        severity: 'critical',
        category: 'coverage',
        message: `LSP coverage dropped ${coverageDrop.toFixed(1)}% below baseline (current: ${metrics.lsp_coverage.toFixed(1)}%)`,
        metrics: { lsp_coverage: metrics.lsp_coverage },
        auto_remediation_triggered: await this.triggerRemediation('lsp_refresh', 'LSP coverage drop', metrics)
      });
    }
  }

  private async analyzeDriftDetection(metrics: SystemMetrics): Promise<void> {
    if (this.metricsHistory.length < 60) return; // Need at least 1 hour of data
    
    // Calculate rolling averages for drift detection
    const recentMetrics = this.metricsHistory.slice(-60); // Last hour
    const baselineMetrics = this.metricsHistory.slice(-120, -60); // Hour before that
    
    const recentP95 = this.average(recentMetrics.map(m => m.p95_latency));
    const baselineP95 = this.average(baselineMetrics.map(m => m.p95_latency));
    const p95Drift = ((recentP95 - baselineP95) / baselineP95) * 100;
    
    if (Math.abs(p95Drift) > this.config.drift_detection.p95_drift_alarm) {
      await this.triggerAlert({
        timestamp: metrics.timestamp,
        severity: 'warning',
        category: 'drift',
        message: `P95 latency drift detected: ${p95Drift.toFixed(1)}% over 1-hour window`,
        metrics: { p95_latency: metrics.p95_latency },
        auto_remediation_triggered: false
      });
    }
  }

  private async analyzeRouterPerformance(metrics: SystemMetrics): Promise<void> {
    const upshiftDeviation = Math.abs(metrics.router.upshift_rate - 0.05);
    
    if (upshiftDeviation > 0.02) { // 2pp tolerance
      await this.triggerAlert({
        timestamp: metrics.timestamp,
        severity: 'warning',
        category: 'performance',
        message: `Router upshift rate ${(metrics.router.upshift_rate * 100).toFixed(1)}% deviates from target 5.0% by ${(upshiftDeviation * 100).toFixed(1)}pp`,
        metrics: { router: metrics.router },
        auto_remediation_triggered: false
      });
    }

    if (metrics.router.paired_ndcg_delta < 0.03) {
      await this.triggerAlert({
        timestamp: metrics.timestamp,
        severity: 'warning',
        category: 'quality',
        message: `Router upshifted queries show insufficient quality improvement: ${metrics.router.paired_ndcg_delta.toFixed(3)} < 0.03 target`,
        metrics: { router: metrics.router },
        auto_remediation_triggered: false
      });
    }
  }

  private async analyzeANNPerformance(metrics: SystemMetrics): Promise<void> {
    if (metrics.ann_performance.recall_50_sla < 0.856) { // Baseline threshold
      await this.triggerAlert({
        timestamp: metrics.timestamp,
        severity: 'critical',
        category: 'performance',
        message: `ANN Recall@50 SLA below baseline: ${metrics.ann_performance.recall_50_sla.toFixed(3)} < 0.856`,
        metrics: { ann_performance: metrics.ann_performance },
        auto_remediation_triggered: await this.triggerRemediation('cache_clear', 'ANN performance degradation', metrics)
      });
    }
  }

  private async triggerAlert(alert: AlertEvent): Promise<void> {
    this.alertHistory.push(alert);
    
    // Keep last 1000 alerts
    if (this.alertHistory.length > 1000) {
      this.alertHistory = this.alertHistory.slice(-1000);
    }

    this.emit('alert_triggered', alert);

    // Log critical alerts immediately
    if (alert.severity === 'critical') {
      this.emit('log', {
        level: 'error',
        message: `CRITICAL ALERT: ${alert.message}`,
        timestamp: alert.timestamp
      });
    }
  }

  private async triggerRemediation(actionType: RemediationAction['action_type'], reason: string, metrics: SystemMetrics): Promise<boolean> {
    if (!this.config.auto_remediation.enabled) {
      return false;
    }

    // Check cooldown
    const cooldownKey = `${actionType}-${reason}`;
    const lastRemediation = this.remediationCooldowns.get(cooldownKey) || 0;
    const cooldownMs = this.config.auto_remediation.cooldown_minutes * 60 * 1000;
    
    if (Date.now() - lastRemediation < cooldownMs) {
      this.emit('log', {
        level: 'info',
        message: `Remediation ${actionType} skipped due to cooldown (${cooldownKey})`,
        timestamp: Date.now()
      });
      return false;
    }

    const remediationAction: RemediationAction = {
      action_type: actionType,
      timestamp: Date.now(),
      trigger_reason: reason,
      success: false,
      metrics_before: {
        calibration: metrics.calibration,
        lsp_coverage: metrics.lsp_coverage,
        ann_performance: metrics.ann_performance
      }
    };

    try {
      await this.executeRemediation(actionType);
      remediationAction.success = true;
      
      // Wait a bit and collect post-remediation metrics
      await new Promise(resolve => setTimeout(resolve, 5000));
      const postMetrics = await this.collectSystemMetrics();
      remediationAction.metrics_after = {
        calibration: postMetrics.calibration,
        lsp_coverage: postMetrics.lsp_coverage,
        ann_performance: postMetrics.ann_performance
      };

      // Set cooldown
      this.remediationCooldowns.set(cooldownKey, Date.now());

      this.emit('auto_remediation_success', remediationAction);
      
      return true;

    } catch (error) {
      remediationAction.error_message = error.message;
      this.emit('auto_remediation_failed', remediationAction);
      
      return false;
    } finally {
      this.remediationHistory.push(remediationAction);
      
      // Keep last 100 remediation actions
      if (this.remediationHistory.length > 100) {
        this.remediationHistory = this.remediationHistory.slice(-100);
      }
    }
  }

  private async executeRemediation(actionType: RemediationAction['action_type']): Promise<void> {
    switch (actionType) {
      case 'isotonic_refit':
        await this.executeIsotonicRefit();
        break;
      case 'lsp_refresh':
        await this.executeLSPRefresh();
        break;
      case 'cache_clear':
        await this.executeCacheClear();
        break;
      case 'traffic_reduce':
        await this.executeTrafficReduction();
        break;
      default:
        throw new Error(`Unknown remediation action: ${actionType}`);
    }
  }

  private async executeIsotonicRefit(): Promise<void> {
    this.emit('log', {
      level: 'info',
      message: 'Executing isotonic calibration refit',
      timestamp: Date.now()
    });
    
    // In real system, would trigger isotonic regression recalibration
    await new Promise(resolve => setTimeout(resolve, 2000));
  }

  private async executeLSPRefresh(): Promise<void> {
    this.emit('log', {
      level: 'info',
      message: 'Executing LSP coverage refresh',
      timestamp: Date.now()
    });
    
    // In real system, would refresh LSP symbol database
    await new Promise(resolve => setTimeout(resolve, 3000));
  }

  private async executeCacheClear(): Promise<void> {
    this.emit('log', {
      level: 'info',
      message: 'Executing ANN cache clear and rebuild',
      timestamp: Date.now()
    });
    
    // In real system, would clear and rebuild ANN indices
    await new Promise(resolve => setTimeout(resolve, 5000));
  }

  private async executeTrafficReduction(): Promise<void> {
    this.emit('log', {
      level: 'warn',
      message: 'Executing emergency traffic reduction',
      timestamp: Date.now()
    });
    
    // In real system, would reduce traffic to affected components
    await new Promise(resolve => setTimeout(resolve, 1000));
  }

  private average(values: number[]): number {
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  // Public API methods
  async getSystemHealth(): Promise<any> {
    const recentMetrics = this.metricsHistory.slice(-10);
    const recentAlerts = this.alertHistory.slice(-10);
    const criticalAlerts = this.alertHistory.filter(a => a.severity === 'critical' && Date.now() - a.timestamp < 60000);

    return {
      status: criticalAlerts.length > 0 ? 'critical' : recentAlerts.length > 5 ? 'degraded' : 'healthy',
      recent_metrics: recentMetrics,
      recent_alerts: recentAlerts,
      critical_alerts_last_minute: criticalAlerts.length,
      remediation_actions_last_hour: this.remediationHistory.filter(r => Date.now() - r.timestamp < 3600000),
      uptime_hours: this.metricsHistory.length / 60, // Assuming 1-minute intervals
      timestamp: Date.now()
    };
  }

  async generateMonitoringReport(): Promise<string> {
    const health = await this.getSystemHealth();
    const totalAlerts = this.alertHistory.length;
    const criticalAlerts = this.alertHistory.filter(a => a.severity === 'critical').length;
    const successfulRemediations = this.remediationHistory.filter(r => r.success).length;

    return `
# Comprehensive Monitoring Report

## System Health Status: ${health.status.toUpperCase()}

### Monitoring Overview
- **Uptime**: ${health.uptime_hours.toFixed(1)} hours
- **Total Metrics Collected**: ${this.metricsHistory.length}
- **Total Alerts**: ${totalAlerts}
- **Critical Alerts**: ${criticalAlerts}
- **Auto-Remediations**: ${this.remediationHistory.length} (${successfulRemediations} successful)

### Recent Performance
${health.recent_metrics.slice(-3).map(m => `
- **${new Date(m.timestamp).toISOString()}**:
  - nDCG@10 Δ: ${m.ndcg_delta.toFixed(2)}pp
  - P95 Latency: ${m.p95_latency.toFixed(1)}ms
  - QPS@150ms: ${m.qps_150ms.toFixed(2)}
  - Error Rate: ${(m.error_rate * 100).toFixed(3)}%
  - ECE: ${m.calibration.ece.toFixed(3)}
  - LSP Coverage: ${m.lsp_coverage.toFixed(1)}%
`).join('\n')}

### Alert Summary
${health.recent_alerts.map(a => `
- **${a.severity.toUpperCase()}** (${new Date(a.timestamp).toISOString()}): ${a.message}
  - Category: ${a.category}
  - Auto-remediation: ${a.auto_remediation_triggered ? 'YES' : 'NO'}
`).join('\n')}

### Auto-Remediation Status
${this.remediationHistory.slice(-5).map(r => `
- **${r.action_type}** (${new Date(r.timestamp).toISOString()}): ${r.success ? 'SUCCESS' : 'FAILED'}
  - Reason: ${r.trigger_reason}
  ${r.error_message ? `  - Error: ${r.error_message}` : ''}
`).join('\n')}
`;
  }
}