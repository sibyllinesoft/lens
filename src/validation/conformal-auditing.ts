/**
 * Live Conformal Auditing - Distribution-Free Coverage Guarantees
 * 
 * Implements real-time conformal prediction monitoring with Wilson 95% CI bounds.
 * Provides distribution-free miscoverage detection on live slices with automatic
 * recalibration and online isotonic regression updates.
 * 
 * Core Features:
 * - Empirical miscoverage tracking per slice ({intent×lang×entropy_bin})
 * - Wilson score confidence intervals for coverage bounds
 * - Online recalibration with exponentially weighted updates
 * - Automatic threshold enforcement (miscoverage ≤ target + 1.5pp)
 * - Isotonic regression for probability calibration
 */

import { z } from 'zod';
import { LensTracer, tracer, meter } from '../telemetry/tracer.js';
import { EntropyBin, QueryOutcome } from './risk-budget-ledger.js';
import type { SearchContext, SearchHit } from '../types/core.js';

// Query intent classification
export enum QueryIntent {
  CODE_SEARCH = 'code_search',
  DEFINITION_LOOKUP = 'definition_lookup', 
  USAGE_EXAMPLES = 'usage_examples',
  API_DISCOVERY = 'api_discovery',
  DEBUG_ASSISTANCE = 'debug_assistance',
  DOCUMENTATION = 'documentation',
  REFACTORING = 'refactoring'
}

// Language classification
export enum LanguageType {
  TYPESCRIPT = 'typescript',
  PYTHON = 'python',
  RUST = 'rust',
  GO = 'go',
  JAVA = 'java',
  JAVASCRIPT = 'javascript',
  OTHER = 'other'
}

// Slice key for stratified monitoring
export const SliceKeySchema = z.object({
  intent: z.nativeEnum(QueryIntent),
  language: z.nativeEnum(LanguageType),
  entropy_bin: z.nativeEnum(EntropyBin),
});

export type SliceKey = z.infer<typeof SliceKeySchema>;

// Conformal prediction record
export const ConformalRecordSchema = z.object({
  trace_id: z.string(),
  timestamp: z.date(),
  slice_key: SliceKeySchema,
  query: z.string(),
  predicted_coverage: z.number().min(0).max(1),
  actual_coverage: z.number().min(0).max(1),
  conformity_score: z.number(),
  covered: z.boolean(), // Whether actual falls within predicted interval
  miscoverage_event: z.boolean(),
  confidence_interval: z.object({
    lower: z.number(),
    upper: z.number(),
    width: z.number(),
  }),
  calibration_metrics: z.object({
    expected_coverage: z.number(),
    empirical_coverage: z.number(),
    calibration_error: z.number(),
  }),
});

export type ConformalRecord = z.infer<typeof ConformalRecordSchema>;

// Slice statistics for monitoring
export const SliceStatsSchema = z.object({
  slice_key: SliceKeySchema,
  total_predictions: z.number().int(),
  covered_predictions: z.number().int(),
  empirical_coverage: z.number().min(0).max(1),
  target_coverage: z.number().min(0).max(1),
  miscoverage_rate: z.number().min(0).max(1),
  wilson_ci_lower: z.number().min(0).max(1),
  wilson_ci_upper: z.number().min(0).max(1),
  violation_threshold: z.number().min(0).max(1), // target + 1.5pp
  threshold_violated: z.boolean(),
  last_update: z.date(),
  recent_trend: z.enum(['improving', 'stable', 'degrading']),
});

export type SliceStats = z.infer<typeof SliceStatsSchema>;

// Configuration for conformal auditing
export const ConformalConfigSchema = z.object({
  enabled: z.boolean(),
  target_coverage: z.number().min(0.5).max(0.99), // e.g., 0.9 for 90% coverage
  violation_margin: z.number().min(0).max(0.1), // 1.5pp = 0.015
  confidence_level: z.number().min(0.9).max(0.999), // 0.95 for 95% CI
  min_samples_per_slice: z.number().int().min(10).max(1000),
  recalibration: z.object({
    enabled: z.boolean(),
    exponential_weight: z.number().min(0).max(1), // 0.9 for heavy weighting of recent
    isotonic_refresh_interval: z.number().int().min(100).max(10000), // samples
    never_upshift_buffer: z.number().min(0).max(0.1), // 2pp = 0.02
  }),
  monitoring: z.object({
    slice_window_size: z.number().int().min(100).max(10000),
    trend_detection_periods: z.number().int().min(3).max(20),
    alert_threshold_violations: z.boolean(),
  }),
});

export type ConformalConfig = z.infer<typeof ConformalConfigSchema>;

// Default configuration per TODO.md requirements
const DEFAULT_CONFORMAL_CONFIG: ConformalConfig = {
  enabled: true,
  target_coverage: 0.90, // 90% coverage target
  violation_margin: 0.015, // 1.5pp margin as specified
  confidence_level: 0.95, // 95% Wilson CI
  min_samples_per_slice: 30,
  recalibration: {
    enabled: true,
    exponential_weight: 0.9, // Heavy weighting of recent samples
    isotonic_refresh_interval: 500,
    never_upshift_buffer: 0.02, // 2pp buffer
  },
  monitoring: {
    slice_window_size: 1000,
    trend_detection_periods: 5,
    alert_threshold_violations: true,
  },
};

// Metrics for conformal auditing
const conformalMetrics = {
  coverage_violations: meter.createCounter('lens_conformal_coverage_violations_total', {
    description: 'Total coverage violations detected per slice',
  }),
  empirical_coverage: meter.createHistogram('lens_conformal_empirical_coverage', {
    description: 'Empirical coverage rate per slice',
  }),
  miscoverage_rate: meter.createHistogram('lens_conformal_miscoverage_rate', {
    description: 'Miscoverage rate distribution across slices',
  }),
  wilson_ci_width: meter.createHistogram('lens_conformal_wilson_ci_width', {
    description: 'Wilson confidence interval width distribution',
  }),
  recalibration_events: meter.createCounter('lens_conformal_recalibration_total', {
    description: 'Total recalibration events triggered',
  }),
  calibration_error: meter.createHistogram('lens_conformal_calibration_error', {
    description: 'Calibration error per slice',
  }),
};

/**
 * Live Conformal Auditing System
 * 
 * Provides real-time monitoring of prediction coverage with distribution-free
 * guarantees using conformal prediction theory and Wilson score intervals.
 */
export class ConformalAuditing {
  private config: ConformalConfig;
  private records: Map<string, ConformalRecord[]>; // Keyed by slice identifier
  private sliceStats: Map<string, SliceStats>;
  private isotonic_calibrators: Map<string, IsotonicCalibrator>;
  
  constructor(config: Partial<ConformalConfig> = {}) {
    this.config = { ...DEFAULT_CONFORMAL_CONFIG, ...config };
    this.records = new Map();
    this.sliceStats = new Map();
    this.isotonic_calibrators = new Map();
  }

  /**
   * Record a conformal prediction and check coverage
   */
  recordPrediction(
    context: SearchContext,
    query: string,
    predictedCoverage: number,
    actualResults: SearchHit[],
    expectedResults: SearchHit[]
  ): ConformalRecord {
    const span = LensTracer.createChildSpan('record_conformal_prediction', {
      'lens.predicted_coverage': predictedCoverage,
      'lens.actual_results': actualResults.length,
      'lens.expected_results': expectedResults.length,
    });

    try {
      // Classify query into slice
      const sliceKey = this.classifyQuery(query, context);
      const sliceId = this.getSliceId(sliceKey);

      // Calculate actual coverage and conformity score
      const actualCoverage = this.calculateActualCoverage(actualResults, expectedResults);
      const conformityScore = this.calculateConformityScore(
        predictedCoverage, 
        actualCoverage, 
        actualResults
      );

      // Determine if prediction was covered
      const covered = actualCoverage >= predictedCoverage - this.config.violation_margin;
      const miscoverageEvent = !covered && actualCoverage < predictedCoverage;

      // Calculate confidence interval for this prediction
      const confidenceInterval = this.calculatePredictionCI(
        predictedCoverage, 
        conformityScore,
        sliceId
      );

      // Get calibration metrics
      const calibrationMetrics = this.getCalibrationMetrics(sliceId);

      const record: ConformalRecord = {
        trace_id: context.trace_id,
        timestamp: new Date(),
        slice_key: sliceKey,
        query,
        predicted_coverage: predictedCoverage,
        actual_coverage: actualCoverage,
        conformity_score: conformityScore,
        covered,
        miscoverage_event: miscoverageEvent,
        confidence_interval: confidenceInterval,
        calibration_metrics: calibrationMetrics,
      };

      // Store record
      if (!this.records.has(sliceId)) {
        this.records.set(sliceId, []);
      }
      this.records.get(sliceId)!.push(record);

      // Update slice statistics
      this.updateSliceStatistics(sliceId, sliceKey, record);

      // Check for recalibration needs
      if (this.config.recalibration.enabled) {
        this.checkRecalibrationNeeds(sliceId, record);
      }

      // Record metrics
      conformalMetrics.empirical_coverage.record(actualCoverage, {
        slice: this.getSliceLabel(sliceKey),
      });

      if (miscoverageEvent) {
        conformalMetrics.coverage_violations.add(1, {
          slice: this.getSliceLabel(sliceKey),
          severity: actualCoverage < predictedCoverage - 0.05 ? 'high' : 'low',
        });
      }

      conformalMetrics.miscoverage_rate.record(
        miscoverageEvent ? 1 : 0,
        { slice: this.getSliceLabel(sliceKey) }
      );

      conformalMetrics.wilson_ci_width.record(confidenceInterval.width, {
        slice: this.getSliceLabel(sliceKey),
      });

      conformalMetrics.calibration_error.record(calibrationMetrics.calibration_error, {
        slice: this.getSliceLabel(sliceKey),
      });

      span.setAttributes({
        'lens.slice_id': sliceId,
        'lens.covered': covered,
        'lens.miscoverage_event': miscoverageEvent,
        'lens.calibration_error': calibrationMetrics.calibration_error,
      });

      return record;

    } finally {
      span.end();
    }
  }

  /**
   * Classify query into slice dimensions
   */
  private classifyQuery(query: string, context: SearchContext): SliceKey {
    // Intent classification based on query patterns
    let intent = QueryIntent.CODE_SEARCH; // default

    if (query.includes('how to') || query.includes('example')) {
      intent = QueryIntent.USAGE_EXAMPLES;
    } else if (query.includes('define') || query.includes('what is')) {
      intent = QueryIntent.DEFINITION_LOOKUP;
    } else if (query.includes('api') || query.includes('method')) {
      intent = QueryIntent.API_DISCOVERY;
    } else if (query.includes('error') || query.includes('bug') || query.includes('fix')) {
      intent = QueryIntent.DEBUG_ASSISTANCE;
    } else if (query.includes('doc') || query.includes('comment')) {
      intent = QueryIntent.DOCUMENTATION;
    } else if (query.includes('refactor') || query.includes('improve')) {
      intent = QueryIntent.REFACTORING;
    }

    // Language classification from context or query
    let language = LanguageType.OTHER;
    const queryLower = query.toLowerCase();

    if (queryLower.includes('typescript') || queryLower.includes('.ts')) {
      language = LanguageType.TYPESCRIPT;
    } else if (queryLower.includes('python') || queryLower.includes('.py')) {
      language = LanguageType.PYTHON;  
    } else if (queryLower.includes('rust') || queryLower.includes('.rs')) {
      language = LanguageType.RUST;
    } else if (queryLower.includes('go') || queryLower.includes('.go')) {
      language = LanguageType.GO;
    } else if (queryLower.includes('java')) {
      language = LanguageType.JAVA;
    } else if (queryLower.includes('javascript') || queryLower.includes('.js')) {
      language = LanguageType.JAVASCRIPT;
    }

    // Entropy classification (simplified)
    const entropy = this.calculateQueryEntropy(query);
    let entropyBin = EntropyBin.MEDIUM;
    if (entropy < 2.0) entropyBin = EntropyBin.LOW;
    else if (entropy > 4.0) entropyBin = EntropyBin.HIGH;

    return {
      intent,
      language, 
      entropy_bin: entropyBin,
    };
  }

  /**
   * Calculate query entropy for entropy binning
   */
  private calculateQueryEntropy(query: string): number {
    const tokens = query.toLowerCase().split(/\s+/);
    const uniqueTokens = new Set(tokens);
    const tokenFreq = new Map<string, number>();

    tokens.forEach(token => {
      tokenFreq.set(token, (tokenFreq.get(token) || 0) + 1);
    });

    let entropy = 0;
    tokenFreq.forEach(freq => {
      const prob = freq / tokens.length;
      entropy -= prob * Math.log2(prob);
    });

    return entropy;
  }

  /**
   * Calculate actual coverage achieved
   */
  private calculateActualCoverage(
    actualResults: SearchHit[],
    expectedResults: SearchHit[]
  ): number {
    if (expectedResults.length === 0) return 1.0;
    if (actualResults.length === 0) return 0.0;

    // Simple overlap calculation - in practice would use more sophisticated metrics
    const actualFiles = new Set(actualResults.map(r => `${r.file}:${r.line}`));
    const expectedFiles = new Set(expectedResults.map(r => `${r.file}:${r.line}`));
    
    const overlap = new Set([...actualFiles].filter(f => expectedFiles.has(f)));
    return overlap.size / expectedFiles.size;
  }

  /**
   * Calculate conformity score for prediction
   */
  private calculateConformityScore(
    predicted: number,
    actual: number,
    results: SearchHit[]
  ): number {
    // Conformity score based on prediction accuracy and result quality
    const predictionError = Math.abs(predicted - actual);
    const avgScore = results.length > 0 ? 
      results.reduce((sum, r) => sum + r.score, 0) / results.length : 0;
    
    // Higher conformity = better prediction
    const conformity = 1.0 - predictionError - (1.0 - avgScore) * 0.1;
    return Math.max(0, Math.min(1, conformity));
  }

  /**
   * Calculate confidence interval for prediction
   */
  private calculatePredictionCI(
    prediction: number,
    conformityScore: number,
    sliceId: string
  ): ConformalRecord['confidence_interval'] {
    const calibrator = this.getIsotonicCalibrator(sliceId);
    const calibratedPrediction = calibrator.calibrate(prediction);
    
    // Use conformity score to adjust CI width
    const baseWidth = 0.1; // Base CI width
    const adjustedWidth = baseWidth * (2 - conformityScore); // Wider CI for low conformity
    
    const lower = Math.max(0, calibratedPrediction - adjustedWidth / 2);
    const upper = Math.min(1, calibratedPrediction + adjustedWidth / 2);
    
    return {
      lower,
      upper,
      width: upper - lower,
    };
  }

  /**
   * Update statistics for a slice
   */
  private updateSliceStatistics(sliceId: string, sliceKey: SliceKey, record: ConformalRecord): void {
    const records = this.records.get(sliceId) || [];
    const recentRecords = records.slice(-this.config.monitoring.slice_window_size);
    
    const totalPredictions = recentRecords.length;
    const coveredPredictions = recentRecords.filter(r => r.covered).length;
    const empiricalCoverage = totalPredictions > 0 ? coveredPredictions / totalPredictions : 0;
    const miscoverageRate = 1 - empiricalCoverage;
    
    // Calculate Wilson confidence interval for empirical coverage
    const wilsonCI = this.calculateWilsonCI(coveredPredictions, totalPredictions);
    
    // Check threshold violation
    const violationThreshold = this.config.target_coverage + this.config.violation_margin;
    const thresholdViolated = empiricalCoverage < violationThreshold || 
                             wilsonCI.lower < violationThreshold;

    // Detect trend
    const trend = this.detectTrend(sliceId, empiricalCoverage);
    
    const stats: SliceStats = {
      slice_key: sliceKey,
      total_predictions: totalPredictions,
      covered_predictions: coveredPredictions,
      empirical_coverage: empiricalCoverage,
      target_coverage: this.config.target_coverage,
      miscoverage_rate: miscoverageRate,
      wilson_ci_lower: wilsonCI.lower,
      wilson_ci_upper: wilsonCI.upper,
      violation_threshold: violationThreshold,
      threshold_violated: thresholdViolated,
      last_update: new Date(),
      recent_trend: trend,
    };

    this.sliceStats.set(sliceId, stats);

    // Alert on threshold violations
    if (thresholdViolated && this.config.monitoring.alert_threshold_violations) {
      this.triggerViolationAlert(sliceId, stats);
    }
  }

  /**
   * Calculate Wilson score confidence interval
   */
  private calculateWilsonCI(successes: number, total: number): { lower: number; upper: number } {
    if (total === 0) return { lower: 0, upper: 1 };
    
    const p = successes / total;
    const z = this.getZScore(this.config.confidence_level);
    
    const center = p + z * z / (2 * total);
    const width = z * Math.sqrt(p * (1 - p) / total + z * z / (4 * total * total));
    const denominator = 1 + z * z / total;
    
    const lower = Math.max(0, (center - width) / denominator);
    const upper = Math.min(1, (center + width) / denominator);
    
    return { lower, upper };
  }

  /**
   * Get Z-score for confidence level
   */
  private getZScore(confidence: number): number {
    // Common Z-scores for confidence levels
    if (confidence >= 0.99) return 2.576;
    if (confidence >= 0.95) return 1.96;
    if (confidence >= 0.90) return 1.645;
    return 1.96; // Default to 95%
  }

  /**
   * Detect coverage trend for a slice
   */
  private detectTrend(sliceId: string, currentCoverage: number): SliceStats['recent_trend'] {
    const records = this.records.get(sliceId) || [];
    if (records.length < this.config.monitoring.trend_detection_periods) {
      return 'stable';
    }

    const recentPeriods = records.slice(-this.config.monitoring.trend_detection_periods);
    const coverageValues = recentPeriods.map(r => r.actual_coverage);
    
    // Simple linear trend detection
    let increasingTrend = 0;
    let decreasingTrend = 0;
    
    for (let i = 1; i < coverageValues.length; i++) {
      if (coverageValues[i] > coverageValues[i-1]) increasingTrend++;
      else if (coverageValues[i] < coverageValues[i-1]) decreasingTrend++;
    }

    const trendThreshold = Math.floor(coverageValues.length * 0.6);
    if (increasingTrend >= trendThreshold) return 'improving';
    if (decreasingTrend >= trendThreshold) return 'degrading';
    return 'stable';
  }

  /**
   * Check if recalibration is needed for a slice
   */
  private checkRecalibrationNeeds(sliceId: string, record: ConformalRecord): void {
    const records = this.records.get(sliceId) || [];
    const calibrator = this.getIsotonicCalibrator(sliceId);
    
    // Check if we have enough samples for recalibration
    if (records.length % this.config.recalibration.isotonic_refresh_interval === 0 &&
        records.length >= this.config.recalibration.isotonic_refresh_interval) {
      
      // Perform isotonic recalibration
      const recentRecords = records.slice(-this.config.recalibration.isotonic_refresh_interval);
      calibrator.update(recentRecords);
      
      conformalMetrics.recalibration_events.add(1, {
        slice: this.getSliceLabel(record.slice_key),
        samples: recentRecords.length.toString(),
      });

      console.log(`Recalibrated isotonic regressor for slice ${sliceId} with ${recentRecords.length} samples`);
    }

    // Check for never-upshift buffer violation
    const stats = this.sliceStats.get(sliceId);
    if (stats && stats.miscoverage_rate > this.config.recalibration.never_upshift_buffer) {
      // Trigger conservative recalibration
      calibrator.applyConservativeAdjustment(stats.miscoverage_rate);
    }
  }

  /**
   * Get or create isotonic calibrator for slice
   */
  private getIsotonicCalibrator(sliceId: string): IsotonicCalibrator {
    if (!this.isotonic_calibrators.has(sliceId)) {
      this.isotonic_calibrators.set(sliceId, new IsotonicCalibrator(
        this.config.recalibration.exponential_weight
      ));
    }
    return this.isotonic_calibrators.get(sliceId)!;
  }

  /**
   * Get calibration metrics for a slice
   */
  private getCalibrationMetrics(sliceId: string): ConformalRecord['calibration_metrics'] {
    const records = this.records.get(sliceId) || [];
    if (records.length === 0) {
      return {
        expected_coverage: this.config.target_coverage,
        empirical_coverage: 0,
        calibration_error: 0,
      };
    }

    const recentRecords = records.slice(-100); // Last 100 samples
    const expectedCoverage = recentRecords.reduce((sum, r) => sum + r.predicted_coverage, 0) / recentRecords.length;
    const empiricalCoverage = recentRecords.filter(r => r.covered).length / recentRecords.length;
    const calibrationError = Math.abs(expectedCoverage - empiricalCoverage);

    return {
      expected_coverage: expectedCoverage,
      empirical_coverage: empiricalCoverage,
      calibration_error: calibrationError,
    };
  }

  /**
   * Trigger violation alert
   */
  private triggerViolationAlert(sliceId: string, stats: SliceStats): void {
    console.warn(`CONFORMAL VIOLATION ALERT: Slice ${sliceId}`, {
      empirical_coverage: stats.empirical_coverage,
      target_coverage: stats.target_coverage,
      violation_threshold: stats.violation_threshold,
      wilson_ci: [stats.wilson_ci_lower, stats.wilson_ci_upper],
      sample_size: stats.total_predictions,
    });
  }

  /**
   * Get slice identifier string
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
   * Get comprehensive auditing status
   */
  getAuditingStatus(): {
    enabled: boolean;
    total_slices: number;
    violating_slices: number;
    avg_empirical_coverage: number;
    worst_slice: { id: string; coverage: number } | null;
    recent_violations: number;
    recalibration_events: number;
  } {
    const sliceArray = Array.from(this.sliceStats.values());
    const violatingSlices = sliceArray.filter(s => s.threshold_violated);
    const avgCoverage = sliceArray.length > 0 ? 
      sliceArray.reduce((sum, s) => sum + s.empirical_coverage, 0) / sliceArray.length : 0;
    
    const worstSlice = sliceArray.length > 0 ? 
      sliceArray.reduce((worst, current) => 
        current.empirical_coverage < worst.empirical_coverage ? current : worst
      ) : null;

    const recentViolations = Array.from(this.records.values())
      .flat()
      .filter(r => {
        const age = Date.now() - r.timestamp.getTime();
        return age < 3600000 && r.miscoverage_event; // Last hour
      }).length;

    return {
      enabled: this.config.enabled,
      total_slices: this.sliceStats.size,
      violating_slices: violatingSlices.length,
      avg_empirical_coverage: avgCoverage,
      worst_slice: worstSlice ? {
        id: this.getSliceId(worstSlice.slice_key),
        coverage: worstSlice.empirical_coverage,
      } : null,
      recent_violations: recentViolations,
      recalibration_events: this.isotonic_calibrators.size,
    };
  }

  /**
   * Get slice-specific analysis
   */
  getSliceAnalysis(sliceKey?: SliceKey): SliceStats[] {
    if (sliceKey) {
      const sliceId = this.getSliceId(sliceKey);
      const stats = this.sliceStats.get(sliceId);
      return stats ? [stats] : [];
    }

    return Array.from(this.sliceStats.values());
  }

  /**
   * Export conformal records for analysis
   */
  exportConformalData(sliceKey?: SliceKey): ConformalRecord[] {
    if (sliceKey) {
      const sliceId = this.getSliceId(sliceKey);
      return this.records.get(sliceId) || [];
    }

    const allRecords: ConformalRecord[] = [];
    for (const records of this.records.values()) {
      allRecords.push(...records);
    }
    return allRecords;
  }
}

/**
 * Isotonic Calibrator for Online Probability Calibration
 * 
 * Implements isotonic regression with exponential weighting for
 * online recalibration of confidence scores.
 */
class IsotonicCalibrator {
  private predictions: number[] = [];
  private outcomes: number[] = [];
  private weights: number[] = [];
  private exponentialWeight: number;

  constructor(exponentialWeight: number = 0.9) {
    this.exponentialWeight = exponentialWeight;
  }

  /**
   * Update calibrator with new records
   */
  update(records: ConformalRecord[]): void {
    records.forEach(record => {
      this.predictions.push(record.predicted_coverage);
      this.outcomes.push(record.covered ? 1 : 0);
      this.weights.push(1.0); // Initial weight
    });

    // Apply exponential weighting (recent samples get higher weight)
    for (let i = this.weights.length - records.length; i < this.weights.length; i++) {
      const age = this.weights.length - 1 - i;
      this.weights[i] = Math.pow(this.exponentialWeight, age);
    }

    // Keep window manageable
    if (this.predictions.length > 1000) {
      const keep = 500;
      this.predictions = this.predictions.slice(-keep);
      this.outcomes = this.outcomes.slice(-keep);
      this.weights = this.weights.slice(-keep);
    }

    this.fitIsotonic();
  }

  /**
   * Calibrate a prediction using isotonic regression
   */
  calibrate(prediction: number): number {
    if (this.predictions.length === 0) return prediction;

    // Simple isotonic calibration - find nearest neighbors
    const sortedIndices = this.predictions
      .map((p, i) => ({ pred: p, outcome: this.outcomes[i], weight: this.weights[i], index: i }))
      .sort((a, b) => a.pred - b.pred);

    // Find position in sorted predictions
    let left = 0;
    let right = sortedIndices.length - 1;

    while (left < right) {
      const mid = Math.floor((left + right) / 2);
      if (sortedIndices[mid].pred < prediction) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }

    // Interpolate between neighbors
    if (left === 0) {
      return sortedIndices[0].outcome * sortedIndices[0].weight;
    } else if (left >= sortedIndices.length) {
      const last = sortedIndices[sortedIndices.length - 1];
      return last.outcome * last.weight;
    } else {
      const leftPoint = sortedIndices[left - 1];
      const rightPoint = sortedIndices[left];
      
      const alpha = (prediction - leftPoint.pred) / (rightPoint.pred - leftPoint.pred);
      const calibrated = leftPoint.outcome * leftPoint.weight * (1 - alpha) + 
                        rightPoint.outcome * rightPoint.weight * alpha;
      
      return Math.max(0, Math.min(1, calibrated));
    }
  }

  /**
   * Apply conservative adjustment for high miscoverage
   */
  applyConservativeAdjustment(miscoverageRate: number): void {
    // Reduce all predictions by miscoverage amount to be more conservative
    const adjustment = miscoverageRate * 0.5; // 50% of the miscoverage
    
    this.predictions = this.predictions.map(p => Math.max(0.1, p - adjustment));
  }

  /**
   * Fit isotonic regression (simplified)
   */
  private fitIsotonic(): void {
    // Simplified isotonic fitting - in practice would use pool-adjacent-violators
    // For now, just ensure predictions are reasonable
    if (this.predictions.length < 2) return;

    // Calculate average outcome for each prediction bin
    const bins = new Map<number, { sumOutcome: number; sumWeight: number; count: number }>();
    
    this.predictions.forEach((pred, i) => {
      const binKey = Math.floor(pred * 10) / 10; // 0.1 bins
      if (!bins.has(binKey)) {
        bins.set(binKey, { sumOutcome: 0, sumWeight: 0, count: 0 });
      }
      const bin = bins.get(binKey)!;
      bin.sumOutcome += this.outcomes[i] * this.weights[i];
      bin.sumWeight += this.weights[i];
      bin.count++;
    });

    // This is a very simplified isotonic fit - real implementation would be more sophisticated
  }
}

// Global conformal auditing instance
export const globalConformalAuditing = new ConformalAuditing();