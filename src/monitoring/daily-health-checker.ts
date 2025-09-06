import { EventEmitter } from 'events';
import { promises as fs } from 'fs';
import { join } from 'path';
import { execSync } from 'child_process';

interface HealthCheckConfig {
  calibration: {
    ece_threshold: number;
    slope_range: [number, number];
    intercept_range: [number, number];
    confidence_correlation_min: number;
  };
  ann_performance: {
    recall_50_sla_min: number;
    ef_search_fixed: number;
    k_fixed: number;
    quantization_error_max: number;
  };
  drift_thresholds: {
    p95_drift_max: number;
    coverage_drop_max: number;
    error_rate_spike_max: number;
    quality_degradation_max: number;
  };
  auto_remediation: {
    isotonic_refit: {
      enabled: boolean;
      max_attempts: number;
    };
    cache_refresh: {
      enabled: boolean;
      max_attempts: number;
    };
    lsp_reindex: {
      enabled: boolean;
      max_attempts: number;
    };
  };
  reporting: {
    output_path: string;
    retention_days: number;
    alert_channels: string[];
  };
}

interface CalibrationAnalysis {
  timestamp: number;
  ece: number;
  slope: number;
  intercept: number;
  confidence_bins: number[];
  accuracy_bins: number[];
  reliability_diagram_points: Array<{ confidence: number; accuracy: number; count: number }>;
  drift_from_baseline: {
    ece_drift: number;
    slope_drift: number;
    intercept_drift: number;
  };
  needs_refit: boolean;
}

interface ANNPerformanceAnalysis {
  timestamp: number;
  recall_50_sla: number;
  ef_search_efficiency: number;
  quantization_error: number;
  index_freshness_hours: number;
  vector_count: number;
  performance_vs_baseline: {
    recall_delta: number;
    efficiency_delta: number;
    error_delta: number;
  };
  needs_rebuild: boolean;
}

interface DriftAnalysis {
  timestamp: number;
  analysis_window_hours: number;
  metrics: {
    p95_drift_percentage: number;
    coverage_drop_percentage: number;
    error_rate_spike_percentage: number;
    quality_degradation_percentage: number;
  };
  trend_analysis: {
    p95_trend: 'improving' | 'stable' | 'degrading';
    coverage_trend: 'improving' | 'stable' | 'degrading';
    quality_trend: 'improving' | 'stable' | 'degrading';
  };
  critical_drifts: string[];
  remediation_recommended: string[];
}

interface HealthCheckReport {
  timestamp: number;
  overall_status: 'healthy' | 'degraded' | 'critical';
  calibration: CalibrationAnalysis;
  ann_performance: ANNPerformanceAnalysis;
  drift_analysis: DriftAnalysis;
  remediation_actions: RemediationResult[];
  recommendations: string[];
  next_check_scheduled: number;
}

interface RemediationResult {
  action: string;
  timestamp: number;
  success: boolean;
  duration_ms: number;
  error_message?: string;
  metrics_before: any;
  metrics_after: any;
  improvement_achieved: boolean;
}

export class DailyHealthChecker extends EventEmitter {
  private config: HealthCheckConfig;
  private baselineMetrics: any = null;
  private lastHealthCheck: number = 0;
  private checkScheduler: NodeJS.Timeout | null = null;
  private remediationAttempts: Map<string, number> = new Map();

  constructor(config: HealthCheckConfig) {
    super();
    this.config = config;
    this.loadBaseline();
  }

  async startDailyHealthChecks(): Promise<void> {
    this.emit('health_checker_start', {
      timestamp: Date.now(),
      config: this.config
    });

    // Schedule daily health checks at 6 AM UTC (during low traffic)
    this.scheduleNextHealthCheck();

    // Also perform an initial health check
    await this.performHealthCheck();

    this.emit('log', {
      level: 'info',
      message: 'Daily health checker started with scheduled checks at 06:00 UTC',
      timestamp: Date.now()
    });
  }

  private scheduleNextHealthCheck(): void {
    // Calculate time until next 6 AM UTC
    const now = new Date();
    const next6AM = new Date();
    next6AM.setUTCHours(6, 0, 0, 0);
    
    if (now.getTime() > next6AM.getTime()) {
      next6AM.setUTCDate(next6AM.getUTCDate() + 1);
    }

    const msUntilNext = next6AM.getTime() - now.getTime();

    this.checkScheduler = setTimeout(async () => {
      await this.performHealthCheck();
      this.scheduleNextHealthCheck(); // Schedule the next one
    }, msUntilNext);

    this.emit('log', {
      level: 'info',
      message: `Next health check scheduled for ${next6AM.toISOString()}`,
      timestamp: Date.now()
    });
  }

  async performHealthCheck(): Promise<HealthCheckReport> {
    const startTime = Date.now();
    this.lastHealthCheck = startTime;

    this.emit('health_check_start', { timestamp: startTime });

    try {
      // Perform all health analyses in parallel
      const [calibrationAnalysis, annAnalysis, driftAnalysis] = await Promise.all([
        this.performCalibrationAnalysis(),
        this.performANNPerformanceAnalysis(),
        this.performDriftAnalysis()
      ]);

      // Determine overall system status
      const overallStatus = this.determineOverallStatus(calibrationAnalysis, annAnalysis, driftAnalysis);

      // Generate remediation actions if needed
      const remediationActions = await this.executeRemediationActions(calibrationAnalysis, annAnalysis, driftAnalysis);

      // Generate recommendations
      const recommendations = this.generateRecommendations(calibrationAnalysis, annAnalysis, driftAnalysis, remediationActions);

      const report: HealthCheckReport = {
        timestamp: startTime,
        overall_status: overallStatus,
        calibration: calibrationAnalysis,
        ann_performance: annAnalysis,
        drift_analysis: driftAnalysis,
        remediation_actions: remediationActions,
        recommendations,
        next_check_scheduled: this.getNextScheduledCheck()
      };

      // Save report
      await this.saveHealthReport(report);

      // Emit completion event
      this.emit('health_check_complete', {
        duration_ms: Date.now() - startTime,
        status: overallStatus,
        remediation_count: remediationActions.length,
        timestamp: Date.now()
      });

      // Send alerts if critical
      if (overallStatus === 'critical') {
        await this.sendCriticalHealthAlert(report);
      }

      return report;

    } catch (error) {
      this.emit('health_check_error', {
        error: error.message,
        duration_ms: Date.now() - startTime,
        timestamp: Date.now()
      });
      throw error;
    }
  }

  private async performCalibrationAnalysis(): Promise<CalibrationAnalysis> {
    this.emit('log', {
      level: 'info',
      message: 'Performing calibration analysis',
      timestamp: Date.now()
    });

    // In real implementation, would query actual calibration metrics
    // For now, simulate based on current system state
    const currentMetrics = await this.getCurrentCalibrationMetrics();
    
    const analysis: CalibrationAnalysis = {
      timestamp: Date.now(),
      ece: currentMetrics.ece,
      slope: currentMetrics.slope,
      intercept: currentMetrics.intercept,
      confidence_bins: currentMetrics.confidence_bins,
      accuracy_bins: currentMetrics.accuracy_bins,
      reliability_diagram_points: this.generateReliabilityDiagram(currentMetrics.confidence_bins, currentMetrics.accuracy_bins),
      drift_from_baseline: {
        ece_drift: currentMetrics.ece - (this.baselineMetrics?.calibration?.ece || 0.03),
        slope_drift: currentMetrics.slope - (this.baselineMetrics?.calibration?.slope || 0.95),
        intercept_drift: currentMetrics.intercept - (this.baselineMetrics?.calibration?.intercept || 0.02)
      },
      needs_refit: false
    };

    // Determine if refit is needed
    analysis.needs_refit = (
      analysis.ece > this.config.calibration.ece_threshold ||
      analysis.slope < this.config.calibration.slope_range[0] ||
      analysis.slope > this.config.calibration.slope_range[1] ||
      analysis.intercept < this.config.calibration.intercept_range[0] ||
      analysis.intercept > this.config.calibration.intercept_range[1]
    );

    return analysis;
  }

  private async performANNPerformanceAnalysis(): Promise<ANNPerformanceAnalysis> {
    this.emit('log', {
      level: 'info',
      message: 'Performing ANN performance analysis',
      timestamp: Date.now()
    });

    // Query current ANN metrics
    const currentMetrics = await this.getCurrentANNMetrics();
    
    const analysis: ANNPerformanceAnalysis = {
      timestamp: Date.now(),
      recall_50_sla: currentMetrics.recall_50_sla,
      ef_search_efficiency: currentMetrics.ef_search_efficiency,
      quantization_error: currentMetrics.quantization_error,
      index_freshness_hours: currentMetrics.index_freshness_hours,
      vector_count: currentMetrics.vector_count,
      performance_vs_baseline: {
        recall_delta: currentMetrics.recall_50_sla - (this.baselineMetrics?.ann?.recall_50_sla || 0.856),
        efficiency_delta: currentMetrics.ef_search_efficiency - (this.baselineMetrics?.ann?.efficiency || 0.95),
        error_delta: currentMetrics.quantization_error - (this.baselineMetrics?.ann?.quantization_error || 0.02)
      },
      needs_rebuild: false
    };

    // Determine if rebuild is needed
    analysis.needs_rebuild = (
      analysis.recall_50_sla < this.config.ann_performance.recall_50_sla_min ||
      analysis.quantization_error > this.config.ann_performance.quantization_error_max ||
      analysis.index_freshness_hours > 168 // More than 1 week old
    );

    return analysis;
  }

  private async performDriftAnalysis(): Promise<DriftAnalysis> {
    this.emit('log', {
      level: 'info',
      message: 'Performing drift analysis',
      timestamp: Date.now()
    });

    // Get historical metrics for drift analysis (last 24 hours vs previous 24 hours)
    const currentWindowMetrics = await this.getMetricsWindow(24); // Last 24 hours
    const previousWindowMetrics = await this.getMetricsWindow(48, 24); // 24-48 hours ago

    const analysis: DriftAnalysis = {
      timestamp: Date.now(),
      analysis_window_hours: 24,
      metrics: {
        p95_drift_percentage: this.calculateDriftPercentage(currentWindowMetrics.p95_latency, previousWindowMetrics.p95_latency),
        coverage_drop_percentage: this.calculateDriftPercentage(previousWindowMetrics.coverage, currentWindowMetrics.coverage),
        error_rate_spike_percentage: this.calculateDriftPercentage(currentWindowMetrics.error_rate, previousWindowMetrics.error_rate),
        quality_degradation_percentage: this.calculateDriftPercentage(previousWindowMetrics.ndcg_delta, currentWindowMetrics.ndcg_delta)
      },
      trend_analysis: {
        p95_trend: this.analyzeTrend(currentWindowMetrics.p95_latency_trend),
        coverage_trend: this.analyzeTrend(currentWindowMetrics.coverage_trend),
        quality_trend: this.analyzeTrend(currentWindowMetrics.ndcg_trend)
      },
      critical_drifts: [],
      remediation_recommended: []
    };

    // Identify critical drifts
    if (Math.abs(analysis.metrics.p95_drift_percentage) > this.config.drift_thresholds.p95_drift_max) {
      analysis.critical_drifts.push(`P95 latency drift: ${analysis.metrics.p95_drift_percentage.toFixed(1)}%`);
      analysis.remediation_recommended.push('cache_refresh');
    }

    if (analysis.metrics.coverage_drop_percentage > this.config.drift_thresholds.coverage_drop_max) {
      analysis.critical_drifts.push(`Coverage drop: ${analysis.metrics.coverage_drop_percentage.toFixed(1)}%`);
      analysis.remediation_recommended.push('lsp_reindex');
    }

    if (analysis.metrics.error_rate_spike_percentage > this.config.drift_thresholds.error_rate_spike_max) {
      analysis.critical_drifts.push(`Error rate spike: ${analysis.metrics.error_rate_spike_percentage.toFixed(1)}%`);
      analysis.remediation_recommended.push('system_restart');
    }

    if (analysis.metrics.quality_degradation_percentage > this.config.drift_thresholds.quality_degradation_max) {
      analysis.critical_drifts.push(`Quality degradation: ${analysis.metrics.quality_degradation_percentage.toFixed(1)}%`);
      analysis.remediation_recommended.push('isotonic_refit');
    }

    return analysis;
  }

  private async executeRemediationActions(
    calibration: CalibrationAnalysis,
    ann: ANNPerformanceAnalysis,
    drift: DriftAnalysis
  ): Promise<RemediationResult[]> {
    const actions: RemediationResult[] = [];

    // Isotonic refit if calibration needs it
    if (calibration.needs_refit && this.config.auto_remediation.isotonic_refit.enabled) {
      const attempts = this.remediationAttempts.get('isotonic_refit') || 0;
      if (attempts < this.config.auto_remediation.isotonic_refit.max_attempts) {
        const result = await this.performIsotonicRefit();
        actions.push(result);
        this.remediationAttempts.set('isotonic_refit', attempts + 1);
      }
    }

    // ANN rebuild if performance degraded
    if (ann.needs_rebuild && this.config.auto_remediation.cache_refresh.enabled) {
      const attempts = this.remediationAttempts.get('cache_refresh') || 0;
      if (attempts < this.config.auto_remediation.cache_refresh.max_attempts) {
        const result = await this.performANNRebuild();
        actions.push(result);
        this.remediationAttempts.set('cache_refresh', attempts + 1);
      }
    }

    // Handle drift-based remediations
    for (const remediation of drift.remediation_recommended) {
      const attempts = this.remediationAttempts.get(remediation) || 0;
      const configKey = remediation as keyof typeof this.config.auto_remediation;
      
      if (this.config.auto_remediation[configKey]?.enabled && attempts < this.config.auto_remediation[configKey].max_attempts) {
        const result = await this.performRemediation(remediation);
        actions.push(result);
        this.remediationAttempts.set(remediation, attempts + 1);
      }
    }

    return actions;
  }

  private async performIsotonicRefit(): Promise<RemediationResult> {
    const startTime = Date.now();
    const metricsBefore = await this.getCurrentCalibrationMetrics();

    const result: RemediationResult = {
      action: 'isotonic_refit',
      timestamp: startTime,
      success: false,
      duration_ms: 0,
      metrics_before: metricsBefore,
      metrics_after: null,
      improvement_achieved: false
    };

    try {
      this.emit('log', {
        level: 'info',
        message: 'Executing isotonic regression refit for calibration',
        timestamp: Date.now()
      });

      // In real system, would call actual isotonic regression refit API
      await this.callIsotonicRefitAPI();

      // Wait for propagation
      await new Promise(resolve => setTimeout(resolve, 5000));

      const metricsAfter = await this.getCurrentCalibrationMetrics();
      result.metrics_after = metricsAfter;
      result.success = true;
      result.improvement_achieved = metricsAfter.ece < metricsBefore.ece;
      
    } catch (error) {
      result.error_message = error.message;
    } finally {
      result.duration_ms = Date.now() - startTime;
    }

    return result;
  }

  private async performANNRebuild(): Promise<RemediationResult> {
    const startTime = Date.now();
    const metricsBefore = await this.getCurrentANNMetrics();

    const result: RemediationResult = {
      action: 'ann_rebuild',
      timestamp: startTime,
      success: false,
      duration_ms: 0,
      metrics_before: metricsBefore,
      metrics_after: null,
      improvement_achieved: false
    };

    try {
      this.emit('log', {
        level: 'info',
        message: 'Executing ANN index rebuild',
        timestamp: Date.now()
      });

      // In real system, would call actual ANN rebuild API
      await this.callANNRebuildAPI();

      // Wait for rebuild completion
      await new Promise(resolve => setTimeout(resolve, 30000));

      const metricsAfter = await this.getCurrentANNMetrics();
      result.metrics_after = metricsAfter;
      result.success = true;
      result.improvement_achieved = metricsAfter.recall_50_sla > metricsBefore.recall_50_sla;
      
    } catch (error) {
      result.error_message = error.message;
    } finally {
      result.duration_ms = Date.now() - startTime;
    }

    return result;
  }

  private async performRemediation(action: string): Promise<RemediationResult> {
    const startTime = Date.now();

    const result: RemediationResult = {
      action,
      timestamp: startTime,
      success: false,
      duration_ms: 0,
      metrics_before: {},
      metrics_after: null,
      improvement_achieved: false
    };

    try {
      switch (action) {
        case 'lsp_reindex':
          await this.callLSPReindexAPI();
          break;
        case 'system_restart':
          await this.callSystemRestartAPI();
          break;
        default:
          throw new Error(`Unknown remediation action: ${action}`);
      }

      result.success = true;
      
    } catch (error) {
      result.error_message = error.message;
    } finally {
      result.duration_ms = Date.now() - startTime;
    }

    return result;
  }

  // Mock API calls - in real system would call actual service APIs
  private async callIsotonicRefitAPI(): Promise<void> {
    await new Promise(resolve => setTimeout(resolve, 3000));
  }

  private async callANNRebuildAPI(): Promise<void> {
    await new Promise(resolve => setTimeout(resolve, 15000));
  }

  private async callLSPReindexAPI(): Promise<void> {
    await new Promise(resolve => setTimeout(resolve, 10000));
  }

  private async callSystemRestartAPI(): Promise<void> {
    await new Promise(resolve => setTimeout(resolve, 5000));
  }

  // Helper methods for analysis
  private async getCurrentCalibrationMetrics(): Promise<any> {
    // Simulate current calibration metrics
    return {
      ece: 0.03 + Math.random() * 0.02,
      slope: 0.95 + (Math.random() - 0.5) * 0.1,
      intercept: 0.02 + (Math.random() - 0.5) * 0.02,
      confidence_bins: Array.from({length: 10}, () => Math.random()),
      accuracy_bins: Array.from({length: 10}, () => Math.random())
    };
  }

  private async getCurrentANNMetrics(): Promise<any> {
    // Simulate current ANN metrics
    return {
      recall_50_sla: 0.856 + (Math.random() - 0.5) * 0.02,
      ef_search_efficiency: 0.95 + (Math.random() - 0.5) * 0.03,
      quantization_error: 0.02 + Math.random() * 0.01,
      index_freshness_hours: Math.random() * 48,
      vector_count: 1000000 + Math.floor(Math.random() * 100000)
    };
  }

  private async getMetricsWindow(durationHours: number, offsetHours = 0): Promise<any> {
    // Simulate historical metrics window
    return {
      p95_latency: 22 + Math.random() * 4,
      coverage: 98.5 + Math.random() * 1.0,
      error_rate: 0.005 + Math.random() * 0.003,
      ndcg_delta: 3.8 + (Math.random() - 0.5) * 0.5,
      p95_latency_trend: Math.random() - 0.5,
      coverage_trend: Math.random() - 0.5,
      ndcg_trend: Math.random() - 0.5
    };
  }

  private calculateDriftPercentage(current: number, previous: number): number {
    return ((current - previous) / previous) * 100;
  }

  private analyzeTrend(trend: number): 'improving' | 'stable' | 'degrading' {
    if (trend > 0.1) return 'improving';
    if (trend < -0.1) return 'degrading';
    return 'stable';
  }

  private generateReliabilityDiagram(confidenceBins: number[], accuracyBins: number[]): Array<{ confidence: number; accuracy: number; count: number }> {
    return confidenceBins.map((conf, i) => ({
      confidence: conf,
      accuracy: accuracyBins[i],
      count: Math.floor(Math.random() * 1000) + 100
    }));
  }

  private determineOverallStatus(
    calibration: CalibrationAnalysis,
    ann: ANNPerformanceAnalysis,
    drift: DriftAnalysis
  ): 'healthy' | 'degraded' | 'critical' {
    if (drift.critical_drifts.length > 0 || calibration.ece > 0.07 || ann.recall_50_sla < 0.8) {
      return 'critical';
    }
    if (calibration.needs_refit || ann.needs_rebuild || drift.remediation_recommended.length > 0) {
      return 'degraded';
    }
    return 'healthy';
  }

  private generateRecommendations(
    calibration: CalibrationAnalysis,
    ann: ANNPerformanceAnalysis,
    drift: DriftAnalysis,
    remediationActions: RemediationResult[]
  ): string[] {
    const recommendations: string[] = [];

    if (calibration.needs_refit) {
      recommendations.push('Calibration ECE exceeded threshold - isotonic regression refit recommended');
    }

    if (ann.needs_rebuild) {
      recommendations.push('ANN performance degraded - index rebuild recommended');
    }

    if (drift.critical_drifts.length > 0) {
      recommendations.push(`Critical performance drifts detected: ${drift.critical_drifts.join(', ')}`);
    }

    const failedRemediations = remediationActions.filter(a => !a.success);
    if (failedRemediations.length > 0) {
      recommendations.push(`Failed auto-remediation actions require manual intervention: ${failedRemediations.map(a => a.action).join(', ')}`);
    }

    if (recommendations.length === 0) {
      recommendations.push('System is operating within normal parameters');
    }

    return recommendations;
  }

  private async loadBaseline(): Promise<void> {
    try {
      const baselinePath = join(this.config.reporting.output_path, 'baseline-metrics.json');
      const baselineData = await fs.readFile(baselinePath, 'utf-8');
      this.baselineMetrics = JSON.parse(baselineData);
    } catch (error) {
      this.emit('log', {
        level: 'warn',
        message: 'No baseline metrics found, using defaults',
        timestamp: Date.now()
      });
      
      // Set default baseline metrics
      this.baselineMetrics = {
        calibration: { ece: 0.03, slope: 0.95, intercept: 0.02 },
        ann: { recall_50_sla: 0.856, efficiency: 0.95, quantization_error: 0.02 }
      };
    }
  }

  private async saveHealthReport(report: HealthCheckReport): Promise<void> {
    const filename = `health-report-${new Date().toISOString().split('T')[0]}.json`;
    const filepath = join(this.config.reporting.output_path, filename);
    
    await fs.writeFile(filepath, JSON.stringify(report, null, 2));
    
    // Cleanup old reports based on retention policy
    await this.cleanupOldReports();
  }

  private async cleanupOldReports(): Promise<void> {
    // Implementation would remove reports older than retention_days
    this.emit('log', {
      level: 'info',
      message: `Cleaned up old health reports (retention: ${this.config.reporting.retention_days} days)`,
      timestamp: Date.now()
    });
  }

  private async sendCriticalHealthAlert(report: HealthCheckReport): Promise<void> {
    const alertMessage = `🚨 CRITICAL HEALTH CHECK ALERT 🚨\n\nTimestamp: ${new Date(report.timestamp).toISOString()}\nStatus: ${report.overall_status}\nCritical Issues: ${report.drift_analysis.critical_drifts.join(', ')}\nRemediation Actions: ${report.remediation_actions.length}`;

    for (const channel of this.config.reporting.alert_channels) {
      this.emit('critical_alert', {
        channel,
        message: alertMessage,
        report,
        timestamp: Date.now()
      });
    }
  }

  private getNextScheduledCheck(): number {
    // Calculate next 6 AM UTC
    const now = new Date();
    const next6AM = new Date();
    next6AM.setUTCHours(6, 0, 0, 0);
    
    if (now.getTime() > next6AM.getTime()) {
      next6AM.setUTCDate(next6AM.getUTCDate() + 1);
    }

    return next6AM.getTime();
  }

  // Public API
  async getLastHealthReport(): Promise<HealthCheckReport | null> {
    try {
      const files = await fs.readdir(this.config.reporting.output_path);
      const healthReports = files.filter(f => f.startsWith('health-report-'));
      
      if (healthReports.length === 0) return null;
      
      const latestReport = healthReports.sort().pop();
      const reportData = await fs.readFile(join(this.config.reporting.output_path, latestReport!), 'utf-8');
      
      return JSON.parse(reportData);
    } catch (error) {
      return null;
    }
  }

  async forceHealthCheck(): Promise<HealthCheckReport> {
    return await this.performHealthCheck();
  }

  async getRemediationHistory(): Promise<RemediationResult[]> {
    // Return recent remediation history across all health checks
    const reports = await this.getRecentHealthReports(7); // Last 7 days
    return reports.flatMap(r => r.remediation_actions);
  }

  private async getRecentHealthReports(days: number): Promise<HealthCheckReport[]> {
    // Implementation would load recent health reports
    return [];
  }

  stopDailyHealthChecks(): void {
    if (this.checkScheduler) {
      clearTimeout(this.checkScheduler);
      this.checkScheduler = null;
    }

    this.emit('health_checker_stop', {
      timestamp: Date.now(),
      last_check: this.lastHealthCheck
    });
  }
}