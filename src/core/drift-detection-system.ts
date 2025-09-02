/**
 * Drift Detection and Alerting System
 * 
 * Implements comprehensive drift monitoring for search quality:
 * - CUSUM detection for Anchor P@1 and Recall@50 degradation
 * - Ladder positives-in-candidates monitoring with trend analysis
 * - LSIF/tree-sitter coverage tracking for corpus changes
 * - Real-time alerting with actionable recommendations
 * 
 * Maintains promotion gate validation while providing early warning
 * on quality drift before it affects user experience.
 */

import { EventEmitter } from 'events';
import { LensTracer } from '../telemetry/tracer.js';
import type { SearchHit, SearchContext } from '../types/core.js';
import type { ValidationResult } from '../types/api.js';

export interface CUSUMConfig {
  reference_value: number;      // Expected baseline value
  drift_threshold: number;      // Threshold for drift detection (sigma units)
  decision_interval: number;    // Decision interval (h parameter)
  reset_threshold: number;      // Threshold for resetting CUSUM
  min_samples: number;          // Minimum samples before alerting
}

export interface DriftMetrics {
  timestamp: string;
  anchor_p_at_1: number;        // Anchor precision@1
  anchor_recall_at_50: number;  // Anchor recall@50
  ladder_positives_ratio: number; // Positives-in-candidates ratio
  lsif_coverage_pct: number;    // LSIF coverage percentage
  tree_sitter_coverage_pct: number; // Tree-sitter coverage percentage
  sample_count: number;         // Number of queries in this measurement
  query_complexity_distribution: {
    simple: number;             // Percentage of simple queries
    medium: number;             // Percentage of medium complexity
    complex: number;            // Percentage of complex queries
  };
}

export interface DriftAlert {
  id: string;
  alert_type: 'anchor_p1_drift' | 'anchor_recall_drift' | 'ladder_drift' | 'coverage_drift';
  severity: 'warning' | 'error' | 'critical';
  metric_name: string;
  current_value: number;
  reference_value: number;
  drift_magnitude: number;      // How far from reference (sigma units)
  cusum_statistic: number;      // Current CUSUM value
  consecutive_violations: number;
  sample_count: number;
  timestamp: string;
  recommended_actions: string[];
  context: {
    trend_direction: 'increasing' | 'decreasing' | 'stable';
    confidence_interval: [number, number];
    historical_baseline: number;
    recent_samples: number[];
  };
}

export interface DriftDetectionConfig {
  anchor_p1_cusum: CUSUMConfig;
  anchor_recall_cusum: CUSUMConfig;
  ladder_monitoring: {
    enabled: boolean;
    baseline_ratio: number;      // Expected positives ratio
    degradation_threshold: number; // Threshold for alerting
    trend_window_size: number;   // Samples for trend analysis
  };
  coverage_monitoring: {
    enabled: boolean;
    lsif_baseline_pct: number;
    tree_sitter_baseline_pct: number;
    degradation_threshold_pct: number;
    measurement_interval_hours: number;
  };
  alerting: {
    consolidation_window_minutes: number;
    max_alerts_per_hour: number;
    escalation_thresholds: {
      warning_consecutive: number;
      error_consecutive: number;
      critical_consecutive: number;
    };
  };
}

/**
 * CUSUM (Cumulative Sum) drift detector for quality metrics
 */
class CUSUMDetector {
  private samples: number[] = [];
  private cusumPositive = 0;     // CUSUM for detecting positive drift (degradation)
  private cusumNegative = 0;     // CUSUM for detecting negative drift (improvement)
  private consecutiveViolations = 0;
  
  constructor(private config: CUSUMConfig) {}

  /**
   * Add new sample and check for drift
   */
  addSample(value: number): {
    drift_detected: boolean;
    drift_direction: 'positive' | 'negative' | 'none';
    cusum_statistic: number;
    consecutive_violations: number;
  } {
    this.samples.push(value);
    
    // Keep only recent samples for memory efficiency
    if (this.samples.length > 1000) {
      this.samples = this.samples.slice(-800);
    }

    // Calculate normalized deviation from reference
    const deviation = value - this.config.reference_value;
    
    // Update CUSUM statistics
    this.cusumPositive = Math.max(0, this.cusumPositive + deviation - this.config.drift_threshold);
    this.cusumNegative = Math.max(0, this.cusumNegative - deviation - this.config.drift_threshold);

    // Check for drift detection
    let driftDetected = false;
    let driftDirection: 'positive' | 'negative' | 'none' = 'none';
    
    if (this.cusumPositive > this.config.decision_interval) {
      driftDetected = true;
      driftDirection = 'positive';
      this.consecutiveViolations++;
    } else if (this.cusumNegative > this.config.decision_interval) {
      driftDetected = true;
      driftDirection = 'negative';
      this.consecutiveViolations++;
    } else {
      this.consecutiveViolations = 0;
    }

    // Reset CUSUM if values return to normal
    if (Math.abs(deviation) < this.config.reset_threshold) {
      this.cusumPositive *= 0.9; // Gradual decay
      this.cusumNegative *= 0.9;
    }

    return {
      drift_detected: driftDetected && this.samples.length >= this.config.min_samples,
      drift_direction: driftDirection,
      cusum_statistic: Math.max(this.cusumPositive, this.cusumNegative),
      consecutive_violations: this.consecutiveViolations
    };
  }

  /**
   * Get current statistics
   */
  getStats() {
    const recentSamples = this.samples.slice(-20);
    const mean = recentSamples.reduce((a, b) => a + b, 0) / recentSamples.length;
    const variance = recentSamples.reduce((a, b) => a + (b - mean) ** 2, 0) / recentSamples.length;
    
    return {
      sample_count: this.samples.length,
      current_mean: mean || 0,
      current_variance: variance || 0,
      cusum_positive: this.cusumPositive,
      cusum_negative: this.cusumNegative,
      consecutive_violations: this.consecutiveViolations,
      recent_samples: recentSamples
    };
  }

  reset(): void {
    this.cusumPositive = 0;
    this.cusumNegative = 0;
    this.consecutiveViolations = 0;
  }
}

/**
 * Comprehensive drift detection and alerting system
 */
export class DriftDetectionSystem extends EventEmitter {
  private anchorP1Detector: CUSUMDetector;
  private anchorRecallDetector: CUSUMDetector;
  private metricsHistory: DriftMetrics[] = [];
  private activeAlerts = new Map<string, DriftAlert>();
  private alertCount = new Map<string, number>(); // Hourly alert counts
  private lastCoverageCheck = Date.now();
  
  constructor(private config: DriftDetectionConfig) {
    super();
    
    this.anchorP1Detector = new CUSUMDetector(config.anchor_p1_cusum);
    this.anchorRecallDetector = new CUSUMDetector(config.anchor_recall_cusum);
    
    console.log('🔍 DriftDetectionSystem initialized');
    console.log(`   Anchor P@1 CUSUM: ref=${config.anchor_p1_cusum.reference_value}, h=${config.anchor_p1_cusum.decision_interval}`);
    console.log(`   Anchor Recall CUSUM: ref=${config.anchor_recall_cusum.reference_value}, h=${config.anchor_recall_cusum.decision_interval}`);
    console.log(`   Ladder monitoring: ${config.ladder_monitoring.enabled ? 'enabled' : 'disabled'}`);
    console.log(`   Coverage monitoring: ${config.coverage_monitoring.enabled ? 'enabled' : 'disabled'}`);
  }

  /**
   * Record new drift metrics and check for alerts
   */
  async recordMetrics(metrics: DriftMetrics): Promise<void> {
    const span = LensTracer.createChildSpan('drift_metrics_recording');
    
    try {
      this.metricsHistory.push(metrics);
      
      // Keep only recent history for memory efficiency
      if (this.metricsHistory.length > 10000) {
        this.metricsHistory = this.metricsHistory.slice(-8000);
      }

      // Check anchor P@1 drift
      const p1Result = this.anchorP1Detector.addSample(metrics.anchor_p_at_1);
      if (p1Result.drift_detected && p1Result.drift_direction === 'positive') {
        await this.createAlert('anchor_p1_drift', 'anchor_p_at_1', metrics, p1Result);
      }

      // Check anchor recall drift
      const recallResult = this.anchorRecallDetector.addSample(metrics.anchor_recall_at_50);
      if (recallResult.drift_detected && recallResult.drift_direction === 'positive') {
        await this.createAlert('anchor_recall_drift', 'anchor_recall_at_50', metrics, recallResult);
      }

      // Check ladder positives-in-candidates drift
      if (this.config.ladder_monitoring.enabled) {
        await this.checkLadderDrift(metrics);
      }

      // Check coverage drift (less frequently)
      if (this.config.coverage_monitoring.enabled && 
          Date.now() - this.lastCoverageCheck > this.config.coverage_monitoring.measurement_interval_hours * 3600000) {
        await this.checkCoverageDrift(metrics);
        this.lastCoverageCheck = Date.now();
      }

      span.setAttributes({
        anchor_p1: metrics.anchor_p_at_1,
        anchor_recall: metrics.anchor_recall_at_50,
        ladder_ratio: metrics.ladder_positives_ratio,
        lsif_coverage: metrics.lsif_coverage_pct,
        active_alerts: this.activeAlerts.size
      });

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
   * Check for ladder positives-in-candidates drift using trend analysis
   */
  private async checkLadderDrift(metrics: DriftMetrics): Promise<void> {
    const config = this.config.ladder_monitoring;
    const recentMetrics = this.metricsHistory.slice(-config.trend_window_size);
    
    if (recentMetrics.length < config.trend_window_size) {
      return; // Not enough data for trend analysis
    }

    const recentRatios = recentMetrics.map(m => m.ladder_positives_ratio);
    const meanRatio = recentRatios.reduce((a, b) => a + b, 0) / recentRatios.length;
    
    // Check for significant degradation
    const degradation = config.baseline_ratio - meanRatio;
    
    if (degradation > config.degradation_threshold) {
      // Calculate trend direction
      const firstHalf = recentRatios.slice(0, Math.floor(config.trend_window_size / 2));
      const secondHalf = recentRatios.slice(Math.floor(config.trend_window_size / 2));
      const firstHalfMean = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
      const secondHalfMean = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;
      
      const trendDirection = secondHalfMean < firstHalfMean ? 'decreasing' : 
                            secondHalfMean > firstHalfMean ? 'increasing' : 'stable';

      // Create drift alert
      const alertId = `ladder_drift_${Date.now()}`;
      const alert: DriftAlert = {
        id: alertId,
        alert_type: 'ladder_drift',
        severity: degradation > config.degradation_threshold * 2 ? 'error' : 'warning',
        metric_name: 'ladder_positives_ratio',
        current_value: meanRatio,
        reference_value: config.baseline_ratio,
        drift_magnitude: degradation / config.degradation_threshold,
        cusum_statistic: 0, // Not using CUSUM for ladder monitoring
        consecutive_violations: 1,
        sample_count: recentMetrics.length,
        timestamp: new Date().toISOString(),
        recommended_actions: this.getLadderDriftActions(degradation),
        context: {
          trend_direction: trendDirection,
          confidence_interval: [meanRatio - 0.05, meanRatio + 0.05], // Simplified CI
          historical_baseline: config.baseline_ratio,
          recent_samples: recentRatios.slice(-10)
        }
      };

      await this.processAlert(alert);
    }
  }

  /**
   * Check for LSIF/tree-sitter coverage drift
   */
  private async checkCoverageDrift(metrics: DriftMetrics): Promise<void> {
    const config = this.config.coverage_monitoring;
    
    // Check LSIF coverage degradation
    const lsifDegradation = config.lsif_baseline_pct - metrics.lsif_coverage_pct;
    if (lsifDegradation > config.degradation_threshold_pct) {
      const alert: DriftAlert = {
        id: `lsif_coverage_drift_${Date.now()}`,
        alert_type: 'coverage_drift',
        severity: lsifDegradation > config.degradation_threshold_pct * 2 ? 'error' : 'warning',
        metric_name: 'lsif_coverage_pct',
        current_value: metrics.lsif_coverage_pct,
        reference_value: config.lsif_baseline_pct,
        drift_magnitude: lsifDegradation / config.degradation_threshold_pct,
        cusum_statistic: 0,
        consecutive_violations: 1,
        sample_count: 1,
        timestamp: new Date().toISOString(),
        recommended_actions: this.getCoverageDriftActions('lsif', lsifDegradation),
        context: {
          trend_direction: 'decreasing',
          confidence_interval: [metrics.lsif_coverage_pct - 2, metrics.lsif_coverage_pct + 2],
          historical_baseline: config.lsif_baseline_pct,
          recent_samples: [metrics.lsif_coverage_pct]
        }
      };

      await this.processAlert(alert);
    }

    // Check Tree-sitter coverage degradation
    const treesSitterDegradation = config.tree_sitter_baseline_pct - metrics.tree_sitter_coverage_pct;
    if (treesSitterDegradation > config.degradation_threshold_pct) {
      const alert: DriftAlert = {
        id: `tree_sitter_coverage_drift_${Date.now()}`,
        alert_type: 'coverage_drift',
        severity: treesSitterDegradation > config.degradation_threshold_pct * 2 ? 'error' : 'warning',
        metric_name: 'tree_sitter_coverage_pct',
        current_value: metrics.tree_sitter_coverage_pct,
        reference_value: config.tree_sitter_baseline_pct,
        drift_magnitude: treesSitterDegradation / config.degradation_threshold_pct,
        cusum_statistic: 0,
        consecutive_violations: 1,
        sample_count: 1,
        timestamp: new Date().toISOString(),
        recommended_actions: this.getCoverageDriftActions('tree-sitter', treesSitterDegradation),
        context: {
          trend_direction: 'decreasing',
          confidence_interval: [metrics.tree_sitter_coverage_pct - 2, metrics.tree_sitter_coverage_pct + 2],
          historical_baseline: config.tree_sitter_baseline_pct,
          recent_samples: [metrics.tree_sitter_coverage_pct]
        }
      };

      await this.processAlert(alert);
    }
  }

  /**
   * Create CUSUM-based drift alert
   */
  private async createAlert(
    alertType: DriftAlert['alert_type'],
    metricName: string,
    metrics: DriftMetrics,
    cusumResult: any
  ): Promise<void> {
    const severity = this.calculateSeverity(cusumResult.consecutive_violations);
    const alertId = `${alertType}_${Date.now()}`;
    
    const alert: DriftAlert = {
      id: alertId,
      alert_type: alertType,
      severity,
      metric_name: metricName,
      current_value: metrics[metricName as keyof DriftMetrics] as number,
      reference_value: alertType === 'anchor_p1_drift' 
        ? this.config.anchor_p1_cusum.reference_value 
        : this.config.anchor_recall_cusum.reference_value,
      drift_magnitude: cusumResult.cusum_statistic,
      cusum_statistic: cusumResult.cusum_statistic,
      consecutive_violations: cusumResult.consecutive_violations,
      sample_count: metrics.sample_count,
      timestamp: new Date().toISOString(),
      recommended_actions: this.getRecommendedActions(alertType, cusumResult.consecutive_violations),
      context: {
        trend_direction: cusumResult.drift_direction === 'positive' ? 'decreasing' : 'increasing',
        confidence_interval: [0, 0], // Would be calculated with proper statistics
        historical_baseline: alertType === 'anchor_p1_drift' 
          ? this.config.anchor_p1_cusum.reference_value 
          : this.config.anchor_recall_cusum.reference_value,
        recent_samples: []
      }
    };

    await this.processAlert(alert);
  }

  /**
   * Process and emit drift alert
   */
  private async processAlert(alert: DriftAlert): Promise<void> {
    // Check alert rate limiting
    const hourKey = new Date().getHours().toString();
    const currentHourCount = this.alertCount.get(hourKey) || 0;
    
    if (currentHourCount >= this.config.alerting.max_alerts_per_hour) {
      console.log(`⚠️ Alert rate limit exceeded for hour ${hourKey}, dropping alert: ${alert.id}`);
      return;
    }

    // Store alert and increment count
    this.activeAlerts.set(alert.id, alert);
    this.alertCount.set(hourKey, currentHourCount + 1);

    // Emit alert event
    this.emit('drift_alert', alert);

    // Log alert
    const icon = alert.severity === 'critical' ? '🚨' : 
                alert.severity === 'error' ? '❌' : '⚠️';
    
    console.log(`${icon} DRIFT ALERT [${alert.severity.toUpperCase()}]: ${alert.metric_name}`);
    console.log(`   Current value: ${alert.current_value.toFixed(4)} (baseline: ${alert.reference_value.toFixed(4)})`);
    console.log(`   Drift magnitude: ${alert.drift_magnitude.toFixed(2)}σ`);
    console.log(`   Consecutive violations: ${alert.consecutive_violations}`);
    console.log(`   Recommended actions:`);
    alert.recommended_actions.forEach(action => console.log(`     - ${action}`));

    // Auto-resolve old alerts of same type
    await this.consolidateAlerts(alert);
  }

  /**
   * Calculate alert severity based on consecutive violations
   */
  private calculateSeverity(consecutiveViolations: number): DriftAlert['severity'] {
    const thresholds = this.config.alerting.escalation_thresholds;
    
    if (consecutiveViolations >= thresholds.critical_consecutive) {
      return 'critical';
    } else if (consecutiveViolations >= thresholds.error_consecutive) {
      return 'error';
    } else {
      return 'warning';
    }
  }

  /**
   * Get recommended actions for different alert types
   */
  private getRecommendedActions(alertType: DriftAlert['alert_type'], violations: number): string[] {
    const baseActions: Record<DriftAlert['alert_type'], string[]> = {
      'anchor_p1_drift': [
        'Check for recent model changes or configuration updates',
        'Review anchor dataset for new failure patterns',
        'Analyze query complexity distribution changes',
        'Consider rollback if degradation is severe (>10%)'
      ],
      'anchor_recall_drift': [
        'Investigate index coverage and completeness',
        'Check for corpus changes affecting recall',
        'Review semantic model performance on recent queries',
        'Analyze candidate generation pipeline health'
      ],
      'ladder_drift': [
        'Review hard-negative generation quality',
        'Check for ranking model calibration drift',
        'Analyze positives-in-candidates ratio trends',
        'Consider re-training ranking components'
      ],
      'coverage_drift': [
        'Check indexing pipeline health and completeness',
        'Review recent repository changes for new patterns',
        'Validate language parser configurations',
        'Consider full re-index if degradation is severe'
      ]
    };

    const actions = [...baseActions[alertType]];

    // Add escalation actions for severe cases
    if (violations >= 5) {
      actions.unshift('URGENT: Consider immediate rollback or mitigation');
      actions.push('Escalate to on-call team for immediate investigation');
    }

    return actions;
  }

  /**
   * Get recommended actions for ladder drift
   */
  private getLadderDriftActions(degradation: number): string[] {
    const actions = [
      'Review hard-negative generation pipeline health',
      'Check ranking model calibration and confidence scores',
      'Analyze recent query patterns for distribution changes',
      'Validate positives-in-candidates ratio computation'
    ];

    if (degradation > 0.1) { // Severe degradation
      actions.unshift('Consider disabling experimental ranking features');
      actions.push('Schedule emergency ranking model re-training');
    }

    return actions;
  }

  /**
   * Get recommended actions for coverage drift
   */
  private getCoverageDriftActions(coverageType: 'lsif' | 'tree-sitter', degradation: number): string[] {
    const actions = coverageType === 'lsif' 
      ? [
          'Check LSIF indexing pipeline health and recent failures',
          'Review repository changes for new language patterns',
          'Validate LSIF parser configurations and language support',
          'Check disk space and indexing resource availability'
        ]
      : [
          'Check Tree-sitter parser health and grammar updates',
          'Review recent changes to supported language configurations',
          'Validate Tree-sitter grammar files and parsing logic',
          'Check for parser crashes or timeout issues'
        ];

    if (degradation > 10) { // Severe degradation
      actions.unshift(`Consider emergency ${coverageType} re-indexing`);
      actions.push('Escalate to indexing team for immediate investigation');
    }

    return actions;
  }

  /**
   * Consolidate similar alerts within time window
   */
  private async consolidateAlerts(newAlert: DriftAlert): Promise<void> {
    const consolidationWindow = this.config.alerting.consolidation_window_minutes * 60000;
    const cutoffTime = Date.now() - consolidationWindow;

    // Find and resolve similar alerts within consolidation window
    for (const [alertId, existingAlert] of this.activeAlerts) {
      if (existingAlert.alert_type === newAlert.alert_type &&
          existingAlert.metric_name === newAlert.metric_name &&
          new Date(existingAlert.timestamp).getTime() > cutoffTime &&
          alertId !== newAlert.id) {
        
        console.log(`🔄 Consolidating alert ${alertId} with new alert ${newAlert.id}`);
        this.activeAlerts.delete(alertId);
        this.emit('alert_resolved', { ...existingAlert, resolved_at: new Date().toISOString() });
      }
    }
  }

  /**
   * Get comprehensive drift detection report
   */
  getDriftReport(): {
    system_health: 'healthy' | 'degraded' | 'critical';
    active_alerts: DriftAlert[];
    metrics_summary: {
      recent_anchor_p1: number;
      recent_anchor_recall: number;
      recent_ladder_ratio: number;
      recent_lsif_coverage: number;
      recent_tree_sitter_coverage: number;
    };
    detector_stats: {
      anchor_p1: any;
      anchor_recall: any;
    };
    recommendations: string[];
  } {
    const activeAlerts = Array.from(this.activeAlerts.values());
    const criticalAlerts = activeAlerts.filter(a => a.severity === 'critical');
    const errorAlerts = activeAlerts.filter(a => a.severity === 'error');
    
    const systemHealth = criticalAlerts.length > 0 ? 'critical' :
                        errorAlerts.length > 0 ? 'degraded' : 'healthy';

    const recentMetrics = this.metricsHistory.slice(-10);
    const metricsAverage = recentMetrics.length > 0 ? {
      recent_anchor_p1: recentMetrics.reduce((sum, m) => sum + m.anchor_p_at_1, 0) / recentMetrics.length,
      recent_anchor_recall: recentMetrics.reduce((sum, m) => sum + m.anchor_recall_at_50, 0) / recentMetrics.length,
      recent_ladder_ratio: recentMetrics.reduce((sum, m) => sum + m.ladder_positives_ratio, 0) / recentMetrics.length,
      recent_lsif_coverage: recentMetrics.reduce((sum, m) => sum + m.lsif_coverage_pct, 0) / recentMetrics.length,
      recent_tree_sitter_coverage: recentMetrics.reduce((sum, m) => sum + m.tree_sitter_coverage_pct, 0) / recentMetrics.length
    } : {
      recent_anchor_p1: 0,
      recent_anchor_recall: 0,
      recent_ladder_ratio: 0,
      recent_lsif_coverage: 0,
      recent_tree_sitter_coverage: 0
    };

    const recommendations = this.generateSystemRecommendations(activeAlerts, systemHealth);

    return {
      system_health: systemHealth,
      active_alerts: activeAlerts,
      metrics_summary: metricsAverage,
      detector_stats: {
        anchor_p1: this.anchorP1Detector.getStats(),
        anchor_recall: this.anchorRecallDetector.getStats()
      },
      recommendations
    };
  }

  /**
   * Generate system-level recommendations
   */
  private generateSystemRecommendations(alerts: DriftAlert[], health: string): string[] {
    const recommendations: string[] = [];

    if (health === 'critical') {
      recommendations.push('🚨 CRITICAL: Immediate intervention required');
      recommendations.push('Consider emergency rollback of recent changes');
      recommendations.push('Activate incident response procedures');
    } else if (health === 'degraded') {
      recommendations.push('⚠️ System degraded - enhanced monitoring recommended');
      recommendations.push('Review recent changes and deployment patterns');
    } else {
      recommendations.push('✅ System healthy - continue normal monitoring');
    }

    // Alert-specific recommendations
    const alertTypes = new Set(alerts.map(a => a.alert_type));
    
    if (alertTypes.has('anchor_p1_drift') || alertTypes.has('anchor_recall_drift')) {
      recommendations.push('Review anchor dataset and baseline metrics');
    }
    
    if (alertTypes.has('ladder_drift')) {
      recommendations.push('Investigate ranking model performance and calibration');
    }
    
    if (alertTypes.has('coverage_drift')) {
      recommendations.push('Check indexing pipeline health and completeness');
    }

    return recommendations;
  }

  /**
   * Reset all drift detectors (use with caution)
   */
  resetAllDetectors(): void {
    this.anchorP1Detector.reset();
    this.anchorRecallDetector.reset();
    this.activeAlerts.clear();
    this.alertCount.clear();
    
    console.log('🔄 All drift detectors reset');
    this.emit('detectors_reset');
  }

  /**
   * Get current system statistics
   */
  getSystemStats() {
    return {
      metrics_history_size: this.metricsHistory.length,
      active_alerts_count: this.activeAlerts.size,
      detector_stats: {
        anchor_p1: this.anchorP1Detector.getStats(),
        anchor_recall: this.anchorRecallDetector.getStats()
      },
      config: this.config
    };
  }
}

/**
 * Default configuration for production deployment
 */
export const defaultDriftDetectionConfig: DriftDetectionConfig = {
  anchor_p1_cusum: {
    reference_value: 0.85,     // Expected Anchor P@1
    drift_threshold: 0.02,     // 2% degradation threshold
    decision_interval: 5.0,    // CUSUM decision interval
    reset_threshold: 0.01,     // Reset threshold
    min_samples: 20            // Minimum samples before alerting
  },
  anchor_recall_cusum: {
    reference_value: 0.92,     // Expected Anchor Recall@50
    drift_threshold: 0.03,     // 3% degradation threshold
    decision_interval: 4.0,    // CUSUM decision interval
    reset_threshold: 0.015,    // Reset threshold
    min_samples: 20            // Minimum samples before alerting
  },
  ladder_monitoring: {
    enabled: true,
    baseline_ratio: 0.78,      // Expected positives-in-candidates ratio
    degradation_threshold: 0.05, // 5% degradation threshold
    trend_window_size: 50      // Samples for trend analysis
  },
  coverage_monitoring: {
    enabled: true,
    lsif_baseline_pct: 85.0,   // Expected LSIF coverage
    tree_sitter_baseline_pct: 92.0, // Expected Tree-sitter coverage
    degradation_threshold_pct: 5.0,  // 5% degradation threshold
    measurement_interval_hours: 6    // Check every 6 hours
  },
  alerting: {
    consolidation_window_minutes: 30, // Consolidate similar alerts
    max_alerts_per_hour: 10,         // Rate limiting
    escalation_thresholds: {
      warning_consecutive: 2,         // 2 consecutive for warning
      error_consecutive: 4,           // 4 consecutive for error
      critical_consecutive: 8         // 8 consecutive for critical
    }
  }
};

// Global drift detection instance
export const globalDriftDetectionSystem = new DriftDetectionSystem(defaultDriftDetectionConfig);