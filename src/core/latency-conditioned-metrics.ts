/**
 * Latency-Conditioned Metrics with CUSUM Alarms
 * 
 * Implements SLA-Core@10, SLA-Diversity@10 metrics alongside SLA-Recall@50,
 * with CUSUM alarms for drift detection and why-mix monitoring.
 */

import type { SearchContext, SearchHit } from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';

export interface LatencyConditionedMetric {
  name: string;
  value: number;
  sla_target: number;
  sla_met: boolean;
  p95_latency_ms: number;
  measurement_timestamp: number;
}

export interface SLAMetrics {
  sla_recall_50: LatencyConditionedMetric;
  sla_core_10: LatencyConditionedMetric;
  sla_diversity_10: LatencyConditionedMetric;
  why_mix_diversity: LatencyConditionedMetric;
  topic_normalized_core_10: LatencyConditionedMetric;
}

export interface CUSUMAlarm {
  name: string;
  current_sum: number;
  threshold: number;
  alarm_state: 'normal' | 'warning' | 'critical';
  consecutive_violations: number;
  last_reset_time: number;
  drift_detected: boolean;
  sensitivity: number;
}

export interface DriftDetectionConfig {
  cusum_threshold: number;
  sensitivity: number;
  reset_threshold: number;
  max_consecutive_violations: number;
  measurement_window_minutes: number;
}

/**
 * CUSUM (Cumulative Sum) drift detector
 */
class CUSUMDetector {
  private alarms: Map<string, CUSUMAlarm> = new Map();
  private measurementHistory: Map<string, Array<{
    value: number;
    timestamp: number;
  }>> = new Map();
  
  private config: DriftDetectionConfig;
  
  constructor(config?: Partial<DriftDetectionConfig>) {
    this.config = {
      cusum_threshold: 3.0,        // Standard CUSUM threshold
      sensitivity: 0.5,            // Sensitivity to changes
      reset_threshold: -2.0,       // Reset negative CUSUM at this level
      max_consecutive_violations: 5,
      measurement_window_minutes: 60,
      ...config
    };
  }
  
  /**
   * Initialize alarm for a metric
   */
  initializeAlarm(metricName: string, targetMean: number): void {
    this.alarms.set(metricName, {
      name: metricName,
      current_sum: 0,
      threshold: this.config.cusum_threshold,
      alarm_state: 'normal',
      consecutive_violations: 0,
      last_reset_time: Date.now(),
      drift_detected: false,
      sensitivity: this.config.sensitivity
    });
    
    this.measurementHistory.set(metricName, []);
    console.log(`🚨 CUSUM alarm initialized for ${metricName} (target: ${targetMean})`);
  }
  
  /**
   * Add measurement and update CUSUM
   */
  addMeasurement(metricName: string, value: number, targetMean: number): CUSUMAlarm {
    const alarm = this.alarms.get(metricName);
    if (!alarm) {
      throw new Error(`CUSUM alarm not initialized for ${metricName}`);
    }
    
    const history = this.measurementHistory.get(metricName)!;
    const now = Date.now();
    
    // Add to measurement history
    history.push({ value, timestamp: now });
    
    // Clean old measurements
    const cutoff = now - this.config.measurement_window_minutes * 60 * 1000;
    const recentHistory = history.filter(m => m.timestamp > cutoff);
    this.measurementHistory.set(metricName, recentHistory);
    
    // Calculate deviation from target
    const deviation = value - targetMean;
    
    // Update CUSUM (positive sum for upward drift detection)
    alarm.current_sum = Math.max(0, alarm.current_sum + deviation - alarm.sensitivity);
    
    // Reset negative drift if needed
    if (alarm.current_sum < this.config.reset_threshold) {
      alarm.current_sum = 0;
      alarm.last_reset_time = now;
    }
    
    // Check for alarm conditions
    const previousState = alarm.alarm_state;
    
    if (alarm.current_sum > alarm.threshold) {
      alarm.consecutive_violations++;
      
      if (alarm.consecutive_violations >= this.config.max_consecutive_violations) {
        alarm.alarm_state = 'critical';
        alarm.drift_detected = true;
      } else if (alarm.consecutive_violations >= this.config.max_consecutive_violations / 2) {
        alarm.alarm_state = 'warning';
      }
    } else {
      alarm.consecutive_violations = Math.max(0, alarm.consecutive_violations - 1);
      
      if (alarm.consecutive_violations === 0) {
        alarm.alarm_state = 'normal';
        alarm.drift_detected = false;
      }
    }
    
    // Log state changes
    if (previousState !== alarm.alarm_state) {
      console.log(`🚨 CUSUM ${metricName}: ${previousState} -> ${alarm.alarm_state} (sum: ${alarm.current_sum.toFixed(3)}, violations: ${alarm.consecutive_violations})`);
    }
    
    return { ...alarm };
  }
  
  /**
   * Get all current alarms
   */
  getAllAlarms(): Map<string, CUSUMAlarm> {
    return new Map(this.alarms);
  }
  
  /**
   * Get alarm status for metric
   */
  getAlarmStatus(metricName: string): CUSUMAlarm | null {
    return this.alarms.get(metricName) || null;
  }
  
  /**
   * Reset alarm for metric
   */
  resetAlarm(metricName: string): void {
    const alarm = this.alarms.get(metricName);
    if (alarm) {
      alarm.current_sum = 0;
      alarm.consecutive_violations = 0;
      alarm.alarm_state = 'normal';
      alarm.drift_detected = false;
      alarm.last_reset_time = Date.now();
      console.log(`🔄 CUSUM alarm reset for ${metricName}`);
    }
  }
}

/**
 * Metric calculator for latency-conditioned SLAs
 */
class MetricCalculator {
  /**
   * Calculate SLA-Recall@50: Recall of top-50 results under latency constraints
   */
  calculateSLARecall50(
    hits: SearchHit[],
    groundTruth: Set<string>,
    p95LatencyMs: number,
    slaTargetMs = 20
  ): LatencyConditionedMetric {
    // Only consider results if latency SLA is met
    const slaCompliant = p95LatencyMs <= slaTargetMs;
    
    if (!slaCompliant) {
      return {
        name: 'SLA-Recall@50',
        value: 0,
        sla_target: 0.8, // 80% recall target
        sla_met: false,
        p95_latency_ms: p95LatencyMs,
        measurement_timestamp: Date.now()
      };
    }
    
    // Calculate recall on top-50 results
    const top50 = hits.slice(0, 50);
    const relevant = top50.filter(hit => groundTruth.has(hit.file)).length;
    const recall = groundTruth.size > 0 ? relevant / groundTruth.size : 0;
    
    return {
      name: 'SLA-Recall@50',
      value: recall,
      sla_target: 0.8,
      sla_met: recall >= 0.8,
      p95_latency_ms: p95LatencyMs,
      measurement_timestamp: Date.now()
    };
  }
  
  /**
   * Calculate SLA-Core@10: Quality of core results under latency constraints
   */
  calculateSLACore10(
    hits: SearchHit[],
    groundTruth: Set<string>,
    p95LatencyMs: number,
    slaTargetMs = 20
  ): LatencyConditionedMetric {
    const slaCompliant = p95LatencyMs <= slaTargetMs;
    
    if (!slaCompliant) {
      return {
        name: 'SLA-Core@10',
        value: 0,
        sla_target: 0.9,
        sla_met: false,
        p95_latency_ms: p95LatencyMs,
        measurement_timestamp: Date.now()
      };
    }
    
    // Calculate precision of top-10 core results
    const top10 = hits.slice(0, 10);
    const relevant = top10.filter(hit => groundTruth.has(hit.file)).length;
    const precision = top10.length > 0 ? relevant / top10.length : 0;
    
    return {
      name: 'SLA-Core@10',
      value: precision,
      sla_target: 0.9,
      sla_met: precision >= 0.9,
      p95_latency_ms: p95LatencyMs,
      measurement_timestamp: Date.now()
    };
  }
  
  /**
   * Calculate SLA-Diversity@10: Diversity of results under latency constraints
   */
  calculateSLADiversity10(
    hits: SearchHit[],
    p95LatencyMs: number,
    slaTargetMs = 20
  ): LatencyConditionedMetric {
    const slaCompliant = p95LatencyMs <= slaTargetMs;
    
    if (!slaCompliant) {
      return {
        name: 'SLA-Diversity@10',
        value: 0,
        sla_target: 0.7,
        sla_met: false,
        p95_latency_ms: p95LatencyMs,
        measurement_timestamp: Date.now()
      };
    }
    
    // Calculate file diversity in top-10
    const top10 = hits.slice(0, 10);
    const uniqueFiles = new Set(top10.map(hit => hit.file)).size;
    const diversity = top10.length > 0 ? uniqueFiles / top10.length : 0;
    
    // Also consider language diversity
    const uniqueLanguages = new Set(top10.map(hit => hit.lang)).size;
    const langDiversity = top10.length > 0 ? uniqueLanguages / Math.min(5, top10.length) : 0;
    
    // Combined diversity score
    const combinedDiversity = (diversity + langDiversity) / 2;
    
    return {
      name: 'SLA-Diversity@10',
      value: combinedDiversity,
      sla_target: 0.7,
      sla_met: combinedDiversity >= 0.7,
      p95_latency_ms: p95LatencyMs,
      measurement_timestamp: Date.now()
    };
  }
  
  /**
   * Calculate why-mix diversity (diversity of match reasons)
   */
  calculateWhyMixDiversity(
    hits: SearchHit[],
    p95LatencyMs: number,
    slaTargetMs = 20
  ): LatencyConditionedMetric {
    const slaCompliant = p95LatencyMs <= slaTargetMs;
    
    if (!slaCompliant) {
      return {
        name: 'WhyMix-Diversity',
        value: 0,
        sla_target: 0.6,
        sla_met: false,
        p95_latency_ms: p95LatencyMs,
        measurement_timestamp: Date.now()
      };
    }
    
    // Calculate diversity of match reasons across results
    const top20 = hits.slice(0, 20);
    const allReasons = top20.flatMap(hit => hit.why || []);
    const uniqueReasons = new Set(allReasons).size;
    const totalReasons = allReasons.length;
    
    const whyDiversity = totalReasons > 0 ? uniqueReasons / Math.min(8, totalReasons) : 0;
    
    return {
      name: 'WhyMix-Diversity', 
      value: whyDiversity,
      sla_target: 0.6,
      sla_met: whyDiversity >= 0.6,
      p95_latency_ms: p95LatencyMs,
      measurement_timestamp: Date.now()
    };
  }
  
  /**
   * Calculate topic-normalized Core@10
   */
  calculateTopicNormalizedCore10(
    hits: SearchHit[],
    queryTopicEntropy: number,
    groundTruth: Set<string>,
    p95LatencyMs: number,
    slaTargetMs = 20
  ): LatencyConditionedMetric {
    const slaCompliant = p95LatencyMs <= slaTargetMs;
    
    if (!slaCompliant) {
      return {
        name: 'Topic-Normalized-Core@10',
        value: 0,
        sla_target: 0.85,
        sla_met: false,
        p95_latency_ms: p95LatencyMs,
        measurement_timestamp: Date.now()
      };
    }
    
    const top10 = hits.slice(0, 10);
    const relevant = top10.filter(hit => groundTruth.has(hit.file)).length;
    const precision = top10.length > 0 ? relevant / top10.length : 0;
    
    // Normalize by topic entropy (harder topics get more lenient targets)
    const entropyAdjustment = Math.max(0.8, 1 - queryTopicEntropy * 0.1);
    const normalizedPrecision = precision / entropyAdjustment;
    
    return {
      name: 'Topic-Normalized-Core@10',
      value: normalizedPrecision,
      sla_target: 0.85,
      sla_met: normalizedPrecision >= 0.85,
      p95_latency_ms: p95LatencyMs,
      measurement_timestamp: Date.now()
    };
  }
}

/**
 * Main latency-conditioned metrics engine
 */
export class LatencyConditionedMetrics {
  private cusumDetector: CUSUMDetector;
  private metricCalculator: MetricCalculator;
  private enabled = true;
  
  // Metric targets (can be configured)
  private slaTargets = {
    p95_latency_ms: 20,
    sla_recall_50: 0.8,
    sla_core_10: 0.9,
    sla_diversity_10: 0.7,
    why_mix_diversity: 0.6,
    topic_normalized_core_10: 0.85
  };
  
  // Measurement history
  private measurements: Array<{
    timestamp: number;
    metrics: SLAMetrics;
    context: SearchContext;
  }> = [];
  
  constructor(driftConfig?: Partial<DriftDetectionConfig>) {
    this.cusumDetector = new CUSUMDetector(driftConfig);
    this.metricCalculator = new MetricCalculator();
    
    // Initialize CUSUM alarms for each metric
    this.initializeCUSUMAlarms();
  }
  
  /**
   * Calculate all latency-conditioned metrics for a search result
   */
  async calculateMetrics(
    hits: SearchHit[],
    ctx: SearchContext,
    p95LatencyMs: number,
    groundTruth?: Set<string>,
    queryTopicEntropy = 2.0
  ): Promise<SLAMetrics> {
    if (!this.enabled) {
      return this.getEmptyMetrics(p95LatencyMs);
    }
    
    const span = LensTracer.createChildSpan('latency_conditioned_metrics');
    
    try {
      // Use mock ground truth if not provided
      const gt = groundTruth || this.generateMockGroundTruth(hits, ctx);
      
      // Calculate all SLA metrics
      const metrics: SLAMetrics = {
        sla_recall_50: this.metricCalculator.calculateSLARecall50(
          hits, gt, p95LatencyMs, this.slaTargets.p95_latency_ms
        ),
        sla_core_10: this.metricCalculator.calculateSLACore10(
          hits, gt, p95LatencyMs, this.slaTargets.p95_latency_ms
        ),
        sla_diversity_10: this.metricCalculator.calculateSLADiversity10(
          hits, p95LatencyMs, this.slaTargets.p95_latency_ms
        ),
        why_mix_diversity: this.metricCalculator.calculateWhyMixDiversity(
          hits, p95LatencyMs, this.slaTargets.p95_latency_ms
        ),
        topic_normalized_core_10: this.metricCalculator.calculateTopicNormalizedCore10(
          hits, queryTopicEntropy, gt, p95LatencyMs, this.slaTargets.p95_latency_ms
        )
      };
      
      // Update CUSUM detectors
      this.updateCUSUMDetectors(metrics);
      
      // Store measurement
      this.measurements.push({
        timestamp: Date.now(),
        metrics,
        context: ctx
      });
      
      // Keep only recent measurements
      const cutoff = Date.now() - 24 * 60 * 60 * 1000; // 24 hours
      this.measurements = this.measurements.filter(m => m.timestamp > cutoff);
      
      console.log(`📊 SLA Metrics: Recall@50=${metrics.sla_recall_50.value.toFixed(3)} Core@10=${metrics.sla_core_10.value.toFixed(3)} Diversity@10=${metrics.sla_diversity_10.value.toFixed(3)} (p95=${p95LatencyMs.toFixed(1)}ms)`);
      
      span.setAttributes({
        success: true,
        p95_latency_ms: p95LatencyMs,
        sla_recall_50: metrics.sla_recall_50.value,
        sla_core_10: metrics.sla_core_10.value,
        sla_diversity_10: metrics.sla_diversity_10.value,
        latency_sla_met: p95LatencyMs <= this.slaTargets.p95_latency_ms
      });
      
      return metrics;
      
    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      console.error('Latency-conditioned metrics error:', error);
      return this.getEmptyMetrics(p95LatencyMs);
    } finally {
      span.end();
    }
  }
  
  /**
   * Initialize CUSUM alarms for all metrics
   */
  private initializeCUSUMAlarms(): void {
    this.cusumDetector.initializeAlarm('sla_recall_50', this.slaTargets.sla_recall_50);
    this.cusumDetector.initializeAlarm('sla_core_10', this.slaTargets.sla_core_10);  
    this.cusumDetector.initializeAlarm('sla_diversity_10', this.slaTargets.sla_diversity_10);
    this.cusumDetector.initializeAlarm('why_mix_diversity', this.slaTargets.why_mix_diversity);
    this.cusumDetector.initializeAlarm('topic_normalized_core_10', this.slaTargets.topic_normalized_core_10);
    this.cusumDetector.initializeAlarm('p95_latency_ms', this.slaTargets.p95_latency_ms);
  }
  
  /**
   * Update CUSUM detectors with new measurements
   */
  private updateCUSUMDetectors(metrics: SLAMetrics): void {
    this.cusumDetector.addMeasurement('sla_recall_50', metrics.sla_recall_50.value, this.slaTargets.sla_recall_50);
    this.cusumDetector.addMeasurement('sla_core_10', metrics.sla_core_10.value, this.slaTargets.sla_core_10);
    this.cusumDetector.addMeasurement('sla_diversity_10', metrics.sla_diversity_10.value, this.slaTargets.sla_diversity_10);
    this.cusumDetector.addMeasurement('why_mix_diversity', metrics.why_mix_diversity.value, this.slaTargets.why_mix_diversity);
    this.cusumDetector.addMeasurement('topic_normalized_core_10', metrics.topic_normalized_core_10.value, this.slaTargets.topic_normalized_core_10);
    this.cusumDetector.addMeasurement('p95_latency_ms', metrics.sla_recall_50.p95_latency_ms, this.slaTargets.p95_latency_ms);
  }
  
  /**
   * Generate mock ground truth for testing
   */
  private generateMockGroundTruth(hits: SearchHit[], ctx: SearchContext): Set<string> {
    // Simple heuristic: top scoring hits are likely relevant
    const topHits = hits.slice(0, Math.min(10, Math.ceil(hits.length * 0.3)));
    return new Set(topHits.map(hit => hit.file));
  }
  
  /**
   * Get empty metrics (for disabled state)
   */
  private getEmptyMetrics(p95LatencyMs: number): SLAMetrics {
    const timestamp = Date.now();
    return {
      sla_recall_50: {
        name: 'SLA-Recall@50',
        value: 0,
        sla_target: this.slaTargets.sla_recall_50,
        sla_met: false,
        p95_latency_ms: p95LatencyMs,
        measurement_timestamp: timestamp
      },
      sla_core_10: {
        name: 'SLA-Core@10',
        value: 0,
        sla_target: this.slaTargets.sla_core_10,
        sla_met: false,
        p95_latency_ms: p95LatencyMs,
        measurement_timestamp: timestamp
      },
      sla_diversity_10: {
        name: 'SLA-Diversity@10',
        value: 0,
        sla_target: this.slaTargets.sla_diversity_10,
        sla_met: false,
        p95_latency_ms: p95LatencyMs,
        measurement_timestamp: timestamp
      },
      why_mix_diversity: {
        name: 'WhyMix-Diversity',
        value: 0,
        sla_target: this.slaTargets.why_mix_diversity,
        sla_met: false,
        p95_latency_ms: p95LatencyMs,
        measurement_timestamp: timestamp
      },
      topic_normalized_core_10: {
        name: 'Topic-Normalized-Core@10',
        value: 0,
        sla_target: this.slaTargets.topic_normalized_core_10,
        sla_met: false,
        p95_latency_ms: p95LatencyMs,
        measurement_timestamp: timestamp
      }
    };
  }
  
  /**
   * Get current CUSUM alarm states
   */
  getCUSUMAlarms(): Map<string, CUSUMAlarm> {
    return this.cusumDetector.getAllAlarms();
  }
  
  /**
   * Get drift detection status
   */
  getDriftStatus(): {
    any_critical_alarms: boolean;
    any_warning_alarms: boolean;
    critical_metrics: string[];
    warning_metrics: string[];
    total_alarms: number;
  } {
    const alarms = this.cusumDetector.getAllAlarms();
    const criticalMetrics: string[] = [];
    const warningMetrics: string[] = [];
    
    for (const [name, alarm] of alarms) {
      if (alarm.alarm_state === 'critical') {
        criticalMetrics.push(name);
      } else if (alarm.alarm_state === 'warning') {
        warningMetrics.push(name);
      }
    }
    
    return {
      any_critical_alarms: criticalMetrics.length > 0,
      any_warning_alarms: warningMetrics.length > 0,
      critical_metrics: criticalMetrics,
      warning_metrics: warningMetrics,
      total_alarms: alarms.size
    };
  }
  
  /**
   * Reset CUSUM alarm for specific metric
   */
  resetAlarm(metricName: string): void {
    this.cusumDetector.resetAlarm(metricName);
  }
  
  /**
   * Get recent measurement history
   */
  getRecentMeasurements(count = 50): Array<{
    timestamp: number;
    metrics: SLAMetrics;
    context: SearchContext;
  }> {
    return this.measurements.slice(-count);
  }
  
  /**
   * Get aggregate statistics
   */
  getAggregateStats(hours = 24): {
    measurement_count: number;
    avg_sla_recall_50: number;
    avg_sla_core_10: number;
    avg_sla_diversity_10: number;
    avg_p95_latency_ms: number;
    sla_compliance_rate: number;
  } {
    const cutoff = Date.now() - hours * 60 * 60 * 1000;
    const recentMeasurements = this.measurements.filter(m => m.timestamp > cutoff);
    
    if (recentMeasurements.length === 0) {
      return {
        measurement_count: 0,
        avg_sla_recall_50: 0,
        avg_sla_core_10: 0,
        avg_sla_diversity_10: 0,
        avg_p95_latency_ms: 0,
        sla_compliance_rate: 0
      };
    }
    
    const sums = recentMeasurements.reduce((acc, m) => {
      acc.sla_recall_50 += m.metrics.sla_recall_50.value;
      acc.sla_core_10 += m.metrics.sla_core_10.value;
      acc.sla_diversity_10 += m.metrics.sla_diversity_10.value;
      acc.p95_latency_ms += m.metrics.sla_recall_50.p95_latency_ms;
      acc.sla_compliant += (m.metrics.sla_recall_50.p95_latency_ms <= this.slaTargets.p95_latency_ms) ? 1 : 0;
      return acc;
    }, {
      sla_recall_50: 0,
      sla_core_10: 0,
      sla_diversity_10: 0,
      p95_latency_ms: 0,
      sla_compliant: 0
    });
    
    const count = recentMeasurements.length;
    
    return {
      measurement_count: count,
      avg_sla_recall_50: sums.sla_recall_50 / count,
      avg_sla_core_10: sums.sla_core_10 / count,
      avg_sla_diversity_10: sums.sla_diversity_10 / count,
      avg_p95_latency_ms: sums.p95_latency_ms / count,
      sla_compliance_rate: (sums.sla_compliant / count) * 100
    };
  }
  
  /**
   * Enable/disable metrics collection
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    console.log(`📊 Latency-conditioned metrics ${enabled ? 'ENABLED' : 'DISABLED'}`);
  }
  
  /**
   * Update SLA targets
   */
  updateSLATargets(targets: Partial<typeof this.slaTargets>): void {
    this.slaTargets = { ...this.slaTargets, ...targets };
    console.log('🔧 SLA targets updated:', targets);
    
    // Reinitialize CUSUM alarms with new targets
    this.initializeCUSUMAlarms();
  }
}

// Global instance
export const globalLatencyConditionedMetrics = new LatencyConditionedMetrics();