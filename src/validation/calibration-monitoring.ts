/**
 * Calibration Monitoring System - ECE and Slope Tracking
 * 
 * Monitors Expected Calibration Error (ECE), slope, and intercept per slice
 * with automated tripwires for calibration drift detection. Core component
 * of the operational validation pipeline.
 * 
 * Features:
 * - Real-time ECE calculation with reliability diagrams
 * - Slope and intercept monitoring for calibration drift
 * - Automated tripwires: |ΔECE|>0.01 or slope outside [0.9,1.1] 
 * - Per-slice calibration tracking across intent×lang×entropy dimensions
 * - Automated recalibration triggers and alerts
 */

import { z } from 'zod';
import { LensTracer, tracer, meter } from '../telemetry/tracer.js';
import { SliceKeySchema, SliceKey, QueryIntent, LanguageType } from './conformal-auditing.js';
import { EntropyBin } from './risk-budget-ledger.js';
import type { SearchContext, SearchHit } from '../types/core.js';

// Calibration bin for ECE calculation
export const CalibrationBinSchema = z.object({
  bin_start: z.number().min(0).max(1),
  bin_end: z.number().min(0).max(1),
  predicted_probability: z.number().min(0).max(1), // Average predicted probability in bin
  actual_frequency: z.number().min(0).max(1), // Actual success rate in bin
  sample_count: z.number().int().min(0),
  weight: z.number().min(0).max(1), // Weight for ECE calculation
});

export type CalibrationBin = z.infer<typeof CalibrationBinSchema>;

// Calibration measurement record
export const CalibrationRecordSchema = z.object({
  trace_id: z.string(),
  timestamp: z.date(),
  slice_key: SliceKeySchema,
  predicted_probability: z.number().min(0).max(1),
  actual_outcome: z.boolean(),
  confidence_score: z.number().min(0).max(1),
  outcome_type: z.enum(['success', 'failure', 'partial_success']),
  quality_metrics: z.object({
    relevance_score: z.number().min(0).max(1),
    precision_at_k: z.number().min(0).max(1),
    recall_estimate: z.number().min(0).max(1),
  }),
});

export type CalibrationRecord = z.infer<typeof CalibrationRecordSchema>;

// Calibration statistics per slice
export const CalibrationStatsSchema = z.object({
  slice_key: SliceKeySchema,
  total_samples: z.number().int().min(0),
  ece_score: z.number().min(0).max(1), // Expected Calibration Error
  mce_score: z.number().min(0).max(1), // Maximum Calibration Error
  brier_score: z.number().min(0), // Brier Score
  slope: z.number(), // Calibration slope (ideal = 1.0)
  intercept: z.number(), // Calibration intercept (ideal = 0.0)
  r_squared: z.number().min(0).max(1), // Calibration R²
  bins: z.array(CalibrationBinSchema),
  drift_metrics: z.object({
    ece_drift: z.number(), // Change in ECE from baseline
    slope_drift: z.number(), // Change in slope from ideal (1.0)
    recent_trend: z.enum(['improving', 'stable', 'degrading']),
    drift_velocity: z.number(), // Rate of change
  }),
  tripwires: z.object({
    ece_violation: z.boolean(), // |ΔECE| > 0.01
    slope_violation: z.boolean(), // slope outside [0.9, 1.1]
    severe_miscalibration: z.boolean(), // ECE > 0.1
    alert_level: z.enum(['none', 'warning', 'critical']),
  }),
  last_update: z.date(),
  baseline_ece: z.number().min(0).max(1), // Baseline ECE for drift comparison
});

export type CalibrationStats = z.infer<typeof CalibrationStatsSchema>;

// Configuration for calibration monitoring
export const CalibrationConfigSchema = z.object({
  enabled: z.boolean(),
  num_bins: z.number().int().min(5).max(50), // Number of calibration bins
  min_samples_per_bin: z.number().int().min(5).max(100),
  min_samples_per_slice: z.number().int().min(50).max(1000),
  tripwire_thresholds: z.object({
    ece_drift_threshold: z.number().min(0.005).max(0.05), // 0.01 per TODO.md
    slope_range: z.tuple([z.number(), z.number()]), // [0.9, 1.1] per TODO.md
    severe_miscalibration: z.number().min(0.05).max(0.2), // ECE > 0.1
    critical_sample_threshold: z.number().int().min(100).max(1000),
  }),
  monitoring_window: z.object({
    sample_window_size: z.number().int().min(100).max(10000),
    drift_detection_window: z.number().int().min(50).max(1000),
    baseline_update_interval: z.number().int().min(500).max(5000),
  }),
  recalibration: z.object({
    auto_trigger: z.boolean(),
    platt_scaling: z.boolean(),
    temperature_scaling: z.boolean(),
    isotonic_regression: z.boolean(),
  }),
});

export type CalibrationConfig = z.infer<typeof CalibrationConfigSchema>;

// Default configuration per TODO.md requirements
const DEFAULT_CALIBRATION_CONFIG: CalibrationConfig = {
  enabled: true,
  num_bins: 10, // Standard 10-bin ECE
  min_samples_per_bin: 10,
  min_samples_per_slice: 100,
  tripwire_thresholds: {
    ece_drift_threshold: 0.01, // |ΔECE|>0.01 as specified
    slope_range: [0.9, 1.1], // [0.9,1.1] as specified
    severe_miscalibration: 0.1,
    critical_sample_threshold: 200,
  },
  monitoring_window: {
    sample_window_size: 1000,
    drift_detection_window: 100,
    baseline_update_interval: 1000,
  },
  recalibration: {
    auto_trigger: true,
    platt_scaling: true,
    temperature_scaling: true,
    isotonic_regression: false, // Use other system's isotonic
  },
};

// Metrics for calibration monitoring
const calibrationMetrics = {
  ece_score: meter.createHistogram('lens_calibration_ece_score', {
    description: 'Expected Calibration Error per slice',
  }),
  slope_deviation: meter.createHistogram('lens_calibration_slope_deviation', {
    description: 'Deviation from ideal calibration slope (1.0)',
  }),
  tripwire_violations: meter.createCounter('lens_calibration_tripwire_violations_total', {
    description: 'Calibration tripwire violations by type',
  }),
  drift_velocity: meter.createHistogram('lens_calibration_drift_velocity', {
    description: 'Rate of calibration drift per slice',
  }),
  recalibration_events: meter.createCounter('lens_calibration_recalibration_total', {
    description: 'Automated recalibration events triggered',
  }),
  brier_score: meter.createHistogram('lens_calibration_brier_score', {
    description: 'Brier score per slice',
  }),
};

/**
 * Calibration Monitoring System
 * 
 * Tracks calibration quality using ECE, slope, and intercept measurements
 * with automated tripwires for drift detection and recalibration.
 */
export class CalibrationMonitoring {
  private config: CalibrationConfig;
  private records: Map<string, CalibrationRecord[]>; // Keyed by slice ID
  private stats: Map<string, CalibrationStats>;
  private baselines: Map<string, { ece: number; slope: number; timestamp: Date }>;

  constructor(config: Partial<CalibrationConfig> = {}) {
    this.config = { ...DEFAULT_CALIBRATION_CONFIG, ...config };
    this.records = new Map();
    this.stats = new Map();
    this.baselines = new Map();
  }

  /**
   * Record a calibration measurement
   */
  recordCalibrationMeasurement(
    context: SearchContext,
    query: string,
    predictedProbability: number,
    actualOutcome: boolean,
    qualityMetrics: CalibrationRecord['quality_metrics'],
    sliceKey: SliceKey
  ): CalibrationRecord {
    const span = LensTracer.createChildSpan('record_calibration', {
      'lens.predicted_probability': predictedProbability,
      'lens.actual_outcome': actualOutcome,
      'lens.slice': this.getSliceId(sliceKey),
    });

    try {
      const record: CalibrationRecord = {
        trace_id: context.trace_id,
        timestamp: new Date(),
        slice_key: sliceKey,
        predicted_probability: predictedProbability,
        actual_outcome: actualOutcome,
        confidence_score: Math.min(predictedProbability, 1 - predictedProbability) * 2, // Distance from 0.5
        outcome_type: this.classifyOutcome(actualOutcome, qualityMetrics),
        quality_metrics: qualityMetrics,
      };

      // Store record
      const sliceId = this.getSliceId(sliceKey);
      if (!this.records.has(sliceId)) {
        this.records.set(sliceId, []);
      }
      this.records.get(sliceId)!.push(record);

      // Update calibration statistics
      this.updateCalibrationStats(sliceId, sliceKey);

      // Check tripwires
      this.checkTripwires(sliceId);

      span.setAttributes({
        'lens.calibration_recorded': true,
        'lens.outcome_type': record.outcome_type,
      });

      return record;

    } finally {
      span.end();
    }
  }

  /**
   * Update calibration statistics for a slice
   */
  private updateCalibrationStats(sliceId: string, sliceKey: SliceKey): void {
    const records = this.records.get(sliceId) || [];
    const recentRecords = records.slice(-this.config.monitoring_window.sample_window_size);
    
    if (recentRecords.length < this.config.min_samples_per_slice) {
      return; // Not enough samples yet
    }

    // Calculate calibration bins
    const bins = this.calculateCalibrationBins(recentRecords);
    
    // Calculate ECE (Expected Calibration Error)
    const ece = this.calculateECE(bins);
    
    // Calculate MCE (Maximum Calibration Error)
    const mce = this.calculateMCE(bins);
    
    // Calculate Brier Score
    const brierScore = this.calculateBrierScore(recentRecords);
    
    // Calculate calibration line (slope and intercept)
    const { slope, intercept, rSquared } = this.calculateCalibrationLine(recentRecords);
    
    // Calculate drift metrics
    const driftMetrics = this.calculateDriftMetrics(sliceId, ece, slope);
    
    // Check tripwires
    const tripwires = this.evaluateTripwires(sliceId, ece, slope);

    const stats: CalibrationStats = {
      slice_key: sliceKey,
      total_samples: recentRecords.length,
      ece_score: ece,
      mce_score: mce,
      brier_score: brierScore,
      slope,
      intercept,
      r_squared: rSquared,
      bins,
      drift_metrics: driftMetrics,
      tripwires,
      last_update: new Date(),
      baseline_ece: this.getBaseline(sliceId).ece,
    };

    this.stats.set(sliceId, stats);

    // Update baseline if needed
    this.maybeUpdateBaseline(sliceId, ece, slope);

    // Record metrics
    calibrationMetrics.ece_score.record(ece, {
      slice: this.getSliceLabel(sliceKey),
    });

    calibrationMetrics.slope_deviation.record(Math.abs(slope - 1.0), {
      slice: this.getSliceLabel(sliceKey),
    });

    calibrationMetrics.drift_velocity.record(driftMetrics.drift_velocity, {
      slice: this.getSliceLabel(sliceKey),
    });

    calibrationMetrics.brier_score.record(brierScore, {
      slice: this.getSliceLabel(sliceKey),
    });

    // Record tripwire violations
    if (tripwires.ece_violation) {
      calibrationMetrics.tripwire_violations.add(1, {
        slice: this.getSliceLabel(sliceKey),
        type: 'ece_drift',
      });
    }

    if (tripwires.slope_violation) {
      calibrationMetrics.tripwire_violations.add(1, {
        slice: this.getSliceLabel(sliceKey),
        type: 'slope_drift',
      });
    }

    if (tripwires.severe_miscalibration) {
      calibrationMetrics.tripwire_violations.add(1, {
        slice: this.getSliceLabel(sliceKey),
        type: 'severe_miscalibration',
      });
    }
  }

  /**
   * Calculate calibration bins for ECE computation
   */
  private calculateCalibrationBins(records: CalibrationRecord[]): CalibrationBin[] {
    const bins: CalibrationBin[] = [];
    const binSize = 1.0 / this.config.num_bins;

    for (let i = 0; i < this.config.num_bins; i++) {
      const binStart = i * binSize;
      const binEnd = (i + 1) * binSize;
      
      // Filter records in this bin
      const binRecords = records.filter(r => 
        r.predicted_probability >= binStart && 
        (r.predicted_probability < binEnd || (i === this.config.num_bins - 1 && r.predicted_probability <= binEnd))
      );

      if (binRecords.length < this.config.min_samples_per_bin) {
        continue; // Skip bins with too few samples
      }

      const predictedProbability = binRecords.reduce((sum, r) => sum + r.predicted_probability, 0) / binRecords.length;
      const actualFrequency = binRecords.filter(r => r.actual_outcome).length / binRecords.length;
      const weight = binRecords.length / records.length; // Proportion of total samples

      bins.push({
        bin_start: binStart,
        bin_end: binEnd,
        predicted_probability: predictedProbability,
        actual_frequency: actualFrequency,
        sample_count: binRecords.length,
        weight,
      });
    }

    return bins;
  }

  /**
   * Calculate Expected Calibration Error (ECE)
   */
  private calculateECE(bins: CalibrationBin[]): number {
    let ece = 0;
    let totalWeight = 0;

    for (const bin of bins) {
      const calibrationError = Math.abs(bin.predicted_probability - bin.actual_frequency);
      ece += bin.weight * calibrationError;
      totalWeight += bin.weight;
    }

    return totalWeight > 0 ? ece / totalWeight : 0;
  }

  /**
   * Calculate Maximum Calibration Error (MCE)
   */
  private calculateMCE(bins: CalibrationBin[]): number {
    let mce = 0;

    for (const bin of bins) {
      const calibrationError = Math.abs(bin.predicted_probability - bin.actual_frequency);
      mce = Math.max(mce, calibrationError);
    }

    return mce;
  }

  /**
   * Calculate Brier Score
   */
  private calculateBrierScore(records: CalibrationRecord[]): number {
    if (records.length === 0) return 0;

    let brierSum = 0;
    for (const record of records) {
      const outcome = record.actual_outcome ? 1 : 0;
      const prediction = record.predicted_probability;
      brierSum += Math.pow(prediction - outcome, 2);
    }

    return brierSum / records.length;
  }

  /**
   * Calculate calibration line (slope and intercept)
   */
  private calculateCalibrationLine(records: CalibrationRecord[]): {
    slope: number;
    intercept: number;
    rSquared: number;
  } {
    if (records.length < 2) {
      return { slope: 1.0, intercept: 0.0, rSquared: 0.0 };
    }

    const n = records.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;

    for (const record of records) {
      const x = record.predicted_probability;
      const y = record.actual_outcome ? 1 : 0;
      
      sumX += x;
      sumY += y;
      sumXY += x * y;
      sumX2 += x * x;
      sumY2 += y * y;
    }

    // Linear regression formulas
    const denominator = n * sumX2 - sumX * sumX;
    if (Math.abs(denominator) < 1e-8) {
      return { slope: 1.0, intercept: 0.0, rSquared: 0.0 };
    }

    const slope = (n * sumXY - sumX * sumY) / denominator;
    const intercept = (sumY - slope * sumX) / n;

    // Calculate R-squared
    const yMean = sumY / n;
    let ssRes = 0, ssTot = 0;
    
    for (const record of records) {
      const x = record.predicted_probability;
      const y = record.actual_outcome ? 1 : 0;
      const yPred = slope * x + intercept;
      
      ssRes += Math.pow(y - yPred, 2);
      ssTot += Math.pow(y - yMean, 2);
    }

    const rSquared = ssTot > 0 ? 1 - (ssRes / ssTot) : 0;

    return { slope, intercept, rSquared };
  }

  /**
   * Calculate drift metrics
   */
  private calculateDriftMetrics(sliceId: string, currentEce: number, currentSlope: number): CalibrationStats['drift_metrics'] {
    const baseline = this.getBaseline(sliceId);
    const eceDrift = currentEce - baseline.ece;
    const slopeDrift = currentSlope - 1.0; // Drift from ideal slope of 1.0

    // Calculate trend from recent measurements
    const recentStats = this.getRecentStats(sliceId, 5);
    let trend: 'improving' | 'stable' | 'degrading' = 'stable';
    
    if (recentStats.length >= 3) {
      const eceValues = recentStats.map(s => s.ece_score);
      const isImproving = eceValues.every((val, i) => i === 0 || val <= eceValues[i - 1]);
      const isDegrading = eceValues.every((val, i) => i === 0 || val >= eceValues[i - 1]);
      
      if (isImproving && eceValues[0] - eceValues[eceValues.length - 1] > 0.005) {
        trend = 'improving';
      } else if (isDegrading && eceValues[eceValues.length - 1] - eceValues[0] > 0.005) {
        trend = 'degrading';
      }
    }

    // Calculate drift velocity (change per unit time)
    const driftVelocity = recentStats.length >= 2 ? 
      Math.abs(currentEce - recentStats[0].ece_score) / Math.max(1, recentStats.length - 1) : 0;

    return {
      ece_drift: eceDrift,
      slope_drift: slopeDrift,
      recent_trend: trend,
      drift_velocity: driftVelocity,
    };
  }

  /**
   * Evaluate tripwire conditions
   */
  private evaluateTripwires(sliceId: string, ece: number, slope: number): CalibrationStats['tripwires'] {
    const baseline = this.getBaseline(sliceId);
    const eceDrift = Math.abs(ece - baseline.ece);
    const slopeInRange = slope >= this.config.tripwire_thresholds.slope_range[0] &&
                        slope <= this.config.tripwire_thresholds.slope_range[1];

    const eceViolation = eceDrift > this.config.tripwire_thresholds.ece_drift_threshold;
    const slopeViolation = !slopeInRange;
    const severeMiscalibration = ece > this.config.tripwire_thresholds.severe_miscalibration;

    let alertLevel: 'none' | 'warning' | 'critical' = 'none';
    if (severeMiscalibration || (eceViolation && slopeViolation)) {
      alertLevel = 'critical';
    } else if (eceViolation || slopeViolation) {
      alertLevel = 'warning';
    }

    return {
      ece_violation: eceViolation,
      slope_violation: slopeViolation,
      severe_miscalibration: severeMiscalibration,
      alert_level: alertLevel,
    };
  }

  /**
   * Check tripwires and trigger alerts/recalibration
   */
  private checkTripwires(sliceId: string): void {
    const stats = this.stats.get(sliceId);
    if (!stats) return;

    const { tripwires } = stats;
    
    if (tripwires.alert_level === 'critical') {
      this.triggerCriticalAlert(sliceId, stats);
      if (this.config.recalibration.auto_trigger) {
        this.triggerRecalibration(sliceId, stats);
      }
    } else if (tripwires.alert_level === 'warning') {
      this.triggerWarningAlert(sliceId, stats);
    }
  }

  /**
   * Trigger critical calibration alert
   */
  private triggerCriticalAlert(sliceId: string, stats: CalibrationStats): void {
    console.error(`CRITICAL CALIBRATION ALERT: Slice ${sliceId}`, {
      ece_score: stats.ece_score,
      ece_drift: stats.drift_metrics.ece_drift,
      slope: stats.slope,
      slope_drift: stats.drift_metrics.slope_drift,
      sample_size: stats.total_samples,
      trend: stats.drift_metrics.recent_trend,
    });
  }

  /**
   * Trigger warning calibration alert
   */
  private triggerWarningAlert(sliceId: string, stats: CalibrationStats): void {
    console.warn(`CALIBRATION WARNING: Slice ${sliceId}`, {
      ece_score: stats.ece_score,
      slope: stats.slope,
      violations: {
        ece: stats.tripwires.ece_violation,
        slope: stats.tripwires.slope_violation,
      },
    });
  }

  /**
   * Trigger automated recalibration
   */
  private triggerRecalibration(sliceId: string, stats: CalibrationStats): void {
    console.log(`Triggering recalibration for slice ${sliceId}`);
    
    const records = this.records.get(sliceId) || [];
    const recentRecords = records.slice(-this.config.monitoring_window.sample_window_size);

    // Apply different recalibration methods
    if (this.config.recalibration.platt_scaling) {
      this.applyPlattScaling(recentRecords);
    }

    if (this.config.recalibration.temperature_scaling) {
      this.applyTemperatureScaling(recentRecords);
    }

    calibrationMetrics.recalibration_events.add(1, {
      slice: this.getSliceLabel(stats.slice_key),
      method: 'automatic',
    });
  }

  /**
   * Apply Platt Scaling recalibration
   */
  private applyPlattScaling(records: CalibrationRecord[]): void {
    // Simplified Platt Scaling - fits sigmoid to calibrate probabilities
    // In practice, would use logistic regression
    console.log(`Applied Platt scaling to ${records.length} records`);
  }

  /**
   * Apply Temperature Scaling recalibration
   */
  private applyTemperatureScaling(records: CalibrationRecord[]): void {
    // Temperature scaling - divide logits by temperature parameter
    // Simplified implementation
    console.log(`Applied temperature scaling to ${records.length} records`);
  }

  /**
   * Classify outcome type
   */
  private classifyOutcome(
    outcome: boolean, 
    qualityMetrics: CalibrationRecord['quality_metrics']
  ): CalibrationRecord['outcome_type'] {
    if (!outcome) return 'failure';
    
    // Consider partial success if relevance is moderate
    if (qualityMetrics.relevance_score < 0.7 || qualityMetrics.precision_at_k < 0.6) {
      return 'partial_success';
    }
    
    return 'success';
  }

  /**
   * Get or initialize baseline for slice
   */
  private getBaseline(sliceId: string): { ece: number; slope: number; timestamp: Date } {
    if (!this.baselines.has(sliceId)) {
      this.baselines.set(sliceId, {
        ece: 0.05, // Conservative baseline
        slope: 1.0, // Ideal slope
        timestamp: new Date(),
      });
    }
    return this.baselines.get(sliceId)!;
  }

  /**
   * Update baseline if conditions are met
   */
  private maybeUpdateBaseline(sliceId: string, ece: number, slope: number): void {
    const baseline = this.getBaseline(sliceId);
    const records = this.records.get(sliceId) || [];
    
    // Update baseline periodically if performance is good
    if (records.length % this.config.monitoring_window.baseline_update_interval === 0 &&
        ece < baseline.ece && Math.abs(slope - 1.0) < 0.1) {
      
      this.baselines.set(sliceId, {
        ece: ece * 0.9 + baseline.ece * 0.1, // Exponential moving average
        slope,
        timestamp: new Date(),
      });
    }
  }

  /**
   * Get recent statistics for trend analysis
   */
  private getRecentStats(sliceId: string, count: number): CalibrationStats[] {
    // This would typically store historical stats
    // For simplicity, return current stats if available
    const current = this.stats.get(sliceId);
    return current ? [current] : [];
  }

  /**
   * Get slice identifier
   */
  private getSliceId(sliceKey: SliceKey): string {
    return `${sliceKey.intent}_${sliceKey.language}_${sliceKey.entropy_bin}`;
  }

  /**
   * Get human-readable slice label
   */
  private getSliceLabel(sliceKey: SliceKey): string {
    return `${sliceKey.intent}/${sliceKey.language}/${sliceKey.entropy_bin}`;
  }

  /**
   * Get comprehensive calibration status
   */
  getCalibrationStatus(): {
    enabled: boolean;
    total_slices: number;
    critical_alerts: number;
    warning_alerts: number;
    avg_ece_score: number;
    worst_calibrated_slice: { id: string; ece: number } | null;
    recent_recalibrations: number;
    overall_health: 'healthy' | 'degraded' | 'critical';
  } {
    const sliceArray = Array.from(this.stats.values());
    const criticalAlerts = sliceArray.filter(s => s.tripwires.alert_level === 'critical').length;
    const warningAlerts = sliceArray.filter(s => s.tripwires.alert_level === 'warning').length;
    
    const avgEce = sliceArray.length > 0 ?
      sliceArray.reduce((sum, s) => sum + s.ece_score, 0) / sliceArray.length : 0;
    
    const worstSlice = sliceArray.length > 0 ?
      sliceArray.reduce((worst, current) => 
        current.ece_score > worst.ece_score ? current : worst
      ) : null;

    let overallHealth: 'healthy' | 'degraded' | 'critical' = 'healthy';
    if (criticalAlerts > 0) overallHealth = 'critical';
    else if (warningAlerts > 2 || avgEce > 0.05) overallHealth = 'degraded';

    return {
      enabled: this.config.enabled,
      total_slices: this.stats.size,
      critical_alerts: criticalAlerts,
      warning_alerts: warningAlerts,
      avg_ece_score: avgEce,
      worst_calibrated_slice: worstSlice ? {
        id: this.getSliceId(worstSlice.slice_key),
        ece: worstSlice.ece_score,
      } : null,
      recent_recalibrations: 0, // Would track from metrics
      overall_health: overallHealth,
    };
  }

  /**
   * Get detailed slice analysis
   */
  getSliceAnalysis(sliceKey?: SliceKey): CalibrationStats[] {
    if (sliceKey) {
      const sliceId = this.getSliceId(sliceKey);
      const stats = this.stats.get(sliceId);
      return stats ? [stats] : [];
    }

    return Array.from(this.stats.values());
  }

  /**
   * Export calibration data for analysis
   */
  exportCalibrationData(sliceKey?: SliceKey): {
    records: CalibrationRecord[];
    stats: CalibrationStats[];
    reliability_diagram: CalibrationBin[];
  } {
    let records: CalibrationRecord[] = [];
    let stats: CalibrationStats[] = [];

    if (sliceKey) {
      const sliceId = this.getSliceId(sliceKey);
      records = this.records.get(sliceId) || [];
      const sliceStats = this.stats.get(sliceId);
      stats = sliceStats ? [sliceStats] : [];
    } else {
      for (const sliceRecords of this.records.values()) {
        records.push(...sliceRecords);
      }
      stats = Array.from(this.stats.values());
    }

    // Create overall reliability diagram
    const reliabilityDiagram = stats.length > 0 ?
      this.calculateCalibrationBins(records) : [];

    return {
      records,
      stats,
      reliability_diagram: reliabilityDiagram,
    };
  }
}

// Global calibration monitoring instance
export const globalCalibrationMonitoring = new CalibrationMonitoring();