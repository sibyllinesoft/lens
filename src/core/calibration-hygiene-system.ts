/**
 * Calibration Hygiene System
 * 
 * Implements sophisticated calibration management with:
 * - Isotonic EW (Exponentially Weighted) updates with slope clamps per slice (target [0.9,1.1])
 * - Auto-deweight priors when ΔECE > 0.01 until next calibration window
 * - Prevents calibration drift while maintaining adaptation capability
 * - Slice-aware calibration with proper statistical validation
 * - Temperature scaling and Platt scaling support for different model types
 */

import type { SearchContext } from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';

export interface CalibrationUpdate {
  timestamp: number;
  slice_id: string;
  predicted_score: number;
  actual_outcome: number;
  confidence_level: number;
  temperature_before: number;
  temperature_after: number;
  ece_before: number;
  ece_after: number;
  slope_clamp_applied: boolean;
}

export interface CalibrationMetrics {
  slice_id: string;
  expected_calibration_error: number; // ECE
  maximum_calibration_error: number;  // MCE
  reliability_score: number;
  sharpness_score: number;
  brier_score: number;
  log_loss: number;
  sample_count: number;
  last_updated: Date;
  calibration_temperature: number;
  isotonic_mapping: Array<{ bin_start: number; bin_end: number; calibrated_probability: number }>;
  slope_factor: number;
  drift_detected: boolean;
  auto_deweight_active: boolean;
}

export interface CalibrationWindowConfig {
  window_size_hours: number;
  min_samples_per_update: number;
  ece_drift_threshold: number; // 0.01 as specified
  slope_clamp_range: [number, number]; // [0.9, 1.1] as specified
  temperature_update_rate: number; // EW learning rate
  isotonic_bins: number;
  deweight_duration_hours: number; // How long to deweight after drift
}

export interface SliceCalibrationState {
  slice_id: string;
  current_temperature: number;
  isotonic_regressor: IsotonicRegressor;
  recent_predictions: Array<{ predicted: number; actual: number; timestamp: number }>;
  slope_factor: number;
  last_ece: number;
  drift_start_time: number | null;
  deweight_factor: number; // 0-1, how much to deweight priors
  update_count: number;
  calibration_window: CalibrationWindowConfig;
}

/**
 * Isotonic regression for monotonic calibration mapping
 */
class IsotonicRegressor {
  private binEdges: number[] = [];
  private calibratedProbs: number[] = [];
  private isCalibrated = false;

  /**
   * Fit isotonic regression on prediction/outcome pairs
   */
  fit(predictions: number[], outcomes: number[], bins: number = 10): void {
    if (predictions.length !== outcomes.length || predictions.length === 0) {
      console.warn('Invalid isotonic regression input');
      return;
    }

    // Create bins and calculate empirical probabilities
    const sortedData = predictions
      .map((pred, i) => ({ pred, outcome: outcomes[i] }))
      .sort((a, b) => a.pred - b.pred);

    this.binEdges = [];
    this.calibratedProbs = [];

    const binSize = Math.max(1, Math.floor(sortedData.length / bins));
    
    for (let i = 0; i < bins; i++) {
      const start = i * binSize;
      const end = i === bins - 1 ? sortedData.length : (i + 1) * binSize;
      
      if (start >= sortedData.length) break;
      
      const binData = sortedData.slice(start, end);
      const avgPrediction = binData.reduce((sum, d) => sum + d.pred, 0) / binData.length;
      const avgOutcome = binData.reduce((sum, d) => sum + d.outcome, 0) / binData.length;
      
      this.binEdges.push(avgPrediction);
      this.calibratedProbs.push(avgOutcome);
    }

    // Ensure monotonicity (simple isotonic regression)
    this.enforceMonotonicity();
    this.isCalibrated = true;
  }

  /**
   * Transform predictions using fitted isotonic mapping
   */
  transform(predictions: number[]): number[] {
    if (!this.isCalibrated || this.binEdges.length === 0) {
      return predictions; // Return original if not calibrated
    }

    return predictions.map(pred => this.transformSingle(pred));
  }

  private transformSingle(prediction: number): number {
    if (this.binEdges.length === 0) return prediction;

    // Find appropriate bin
    for (let i = 0; i < this.binEdges.length - 1; i++) {
      if (prediction >= this.binEdges[i] && prediction < this.binEdges[i + 1]) {
        // Linear interpolation between bins
        const weight = (prediction - this.binEdges[i]) / (this.binEdges[i + 1] - this.binEdges[i]);
        return this.calibratedProbs[i] * (1 - weight) + this.calibratedProbs[i + 1] * weight;
      }
    }

    // Handle edge cases
    if (prediction < this.binEdges[0]) {
      return this.calibratedProbs[0];
    } else {
      return this.calibratedProbs[this.calibratedProbs.length - 1];
    }
  }

  private enforceMonotonicity(): void {
    // Simple isotonic regression: ensure calibrated probabilities are non-decreasing
    for (let i = 1; i < this.calibratedProbs.length; i++) {
      if (this.calibratedProbs[i] < this.calibratedProbs[i - 1]) {
        // Pool adjacent violators
        let j = i - 1;
        let sum = this.calibratedProbs[j] + this.calibratedProbs[i];
        let count = 2;
        
        // Look backward for more violations
        while (j > 0 && this.calibratedProbs[j - 1] > sum / count) {
          j--;
          sum += this.calibratedProbs[j];
          count++;
        }
        
        // Set all pooled values to average
        const pooledValue = sum / count;
        for (let k = j; k <= i; k++) {
          this.calibratedProbs[k] = pooledValue;
        }
      }
    }
  }

  getMapping(): Array<{ bin_start: number; bin_end: number; calibrated_probability: number }> {
    const mapping: Array<{ bin_start: number; bin_end: number; calibrated_probability: number }> = [];
    
    for (let i = 0; i < this.binEdges.length; i++) {
      const binStart = i === 0 ? 0 : this.binEdges[i - 1];
      const binEnd = i === this.binEdges.length - 1 ? 1 : this.binEdges[i + 1];
      
      mapping.push({
        bin_start: binStart,
        bin_end: binEnd,
        calibrated_probability: this.calibratedProbs[i] || 0
      });
    }
    
    return mapping;
  }
}

/**
 * Temperature scaling calibration method
 */
class TemperatureScaler {
  private temperature: number = 1.0;

  /**
   * Fit temperature parameter using maximum likelihood
   */
  fit(predictions: number[], outcomes: number[]): void {
    if (predictions.length !== outcomes.length || predictions.length === 0) {
      console.warn('Invalid temperature scaling input');
      return;
    }

    // Use gradient descent to find optimal temperature
    let bestTemp = 1.0;
    let bestLoss = this.calculateLogLoss(predictions, outcomes, bestTemp);

    // Grid search for temperature
    for (let temp = 0.1; temp <= 5.0; temp += 0.1) {
      const loss = this.calculateLogLoss(predictions, outcomes, temp);
      if (loss < bestLoss) {
        bestLoss = loss;
        bestTemp = temp;
      }
    }

    this.temperature = bestTemp;
  }

  /**
   * Apply temperature scaling to predictions
   */
  transform(predictions: number[]): number[] {
    return predictions.map(pred => {
      // Apply temperature scaling: p_calibrated = sigmoid(logit(p) / T)
      const logit = Math.log(Math.max(pred, 1e-10) / Math.max(1 - pred, 1e-10));
      const scaledLogit = logit / this.temperature;
      return 1 / (1 + Math.exp(-scaledLogit));
    });
  }

  private calculateLogLoss(predictions: number[], outcomes: number[], temperature: number): number {
    let totalLoss = 0;
    
    for (let i = 0; i < predictions.length; i++) {
      const pred = Math.max(1e-10, Math.min(1 - 1e-10, predictions[i]));
      const logit = Math.log(pred / (1 - pred));
      const scaledLogit = logit / temperature;
      const calibratedPred = 1 / (1 + Math.exp(-scaledLogit));
      
      const clampedPred = Math.max(1e-10, Math.min(1 - 1e-10, calibratedPred));
      totalLoss += outcomes[i] * Math.log(clampedPred) + (1 - outcomes[i]) * Math.log(1 - clampedPred);
    }
    
    return -totalLoss / predictions.length;
  }

  getTemperature(): number {
    return this.temperature;
  }
}

/**
 * Calibration metrics calculator
 */
class CalibrationMetricsCalculator {
  /**
   * Calculate Expected Calibration Error (ECE)
   */
  calculateECE(predictions: number[], outcomes: number[], bins: number = 10): number {
    if (predictions.length !== outcomes.length || predictions.length === 0) {
      return 0;
    }

    let ece = 0;
    const binSize = 1.0 / bins;

    for (let i = 0; i < bins; i++) {
      const binStart = i * binSize;
      const binEnd = (i + 1) * binSize;
      
      const binIndices = predictions
        .map((pred, idx) => ({ pred, idx }))
        .filter(({ pred }) => pred >= binStart && pred < binEnd)
        .map(({ idx }) => idx);

      if (binIndices.length === 0) continue;

      const binPredictions = binIndices.map(idx => predictions[idx]);
      const binOutcomes = binIndices.map(idx => outcomes[idx]);

      const avgPrediction = binPredictions.reduce((sum, p) => sum + p, 0) / binPredictions.length;
      const avgOutcome = binOutcomes.reduce((sum, o) => sum + o, 0) / binOutcomes.length;

      const binWeight = binIndices.length / predictions.length;
      ece += binWeight * Math.abs(avgPrediction - avgOutcome);
    }

    return ece;
  }

  /**
   * Calculate Maximum Calibration Error (MCE)
   */
  calculateMCE(predictions: number[], outcomes: number[], bins: number = 10): number {
    if (predictions.length !== outcomes.length || predictions.length === 0) {
      return 0;
    }

    let mce = 0;
    const binSize = 1.0 / bins;

    for (let i = 0; i < bins; i++) {
      const binStart = i * binSize;
      const binEnd = (i + 1) * binSize;
      
      const binIndices = predictions
        .map((pred, idx) => ({ pred, idx }))
        .filter(({ pred }) => pred >= binStart && pred < binEnd)
        .map(({ idx }) => idx);

      if (binIndices.length === 0) continue;

      const binPredictions = binIndices.map(idx => predictions[idx]);
      const binOutcomes = binIndices.map(idx => outcomes[idx]);

      const avgPrediction = binPredictions.reduce((sum, p) => sum + p, 0) / binPredictions.length;
      const avgOutcome = binOutcomes.reduce((sum, o) => sum + o, 0) / binOutcomes.length;

      mce = Math.max(mce, Math.abs(avgPrediction - avgOutcome));
    }

    return mce;
  }

  /**
   * Calculate Brier Score
   */
  calculateBrierScore(predictions: number[], outcomes: number[]): number {
    if (predictions.length !== outcomes.length || predictions.length === 0) {
      return 1; // Worst possible score
    }

    let score = 0;
    for (let i = 0; i < predictions.length; i++) {
      score += Math.pow(predictions[i] - outcomes[i], 2);
    }

    return score / predictions.length;
  }

  /**
   * Calculate reliability (how well-calibrated the predictions are)
   */
  calculateReliability(predictions: number[], outcomes: number[], bins: number = 10): number {
    const ece = this.calculateECE(predictions, outcomes, bins);
    return Math.max(0, 1 - ece); // Higher is better
  }

  /**
   * Calculate sharpness (how confident the predictions are)
   */
  calculateSharpness(predictions: number[]): number {
    if (predictions.length === 0) return 0;

    // Sharpness is variance of predictions (higher variance = more confident/sharp)
    const mean = predictions.reduce((sum, p) => sum + p, 0) / predictions.length;
    const variance = predictions.reduce((sum, p) => sum + Math.pow(p - mean, 2), 0) / predictions.length;
    
    return variance;
  }
}

/**
 * Main calibration hygiene system
 */
export class CalibrationHygieneSystem {
  private sliceStates: Map<string, SliceCalibrationState> = new Map();
  private metricsCalculator: CalibrationMetricsCalculator;
  private enabled = true;

  private readonly defaultConfig: CalibrationWindowConfig = {
    window_size_hours: 24,
    min_samples_per_update: 50,
    ece_drift_threshold: 0.01, // As specified in TODO
    slope_clamp_range: [0.9, 1.1], // As specified in TODO
    temperature_update_rate: 0.1, // EW learning rate
    isotonic_bins: 10,
    deweight_duration_hours: 168 // 1 week
  };

  constructor() {
    this.metricsCalculator = new CalibrationMetricsCalculator();
  }

  /**
   * Record prediction and outcome for calibration tracking
   */
  recordPrediction(
    sliceId: string,
    predictedScore: number,
    actualOutcome: number,
    confidence: number = 1.0
  ): void {
    if (!this.enabled) return;

    let state = this.sliceStates.get(sliceId);
    if (!state) {
      state = this.initializeSliceState(sliceId);
      this.sliceStates.set(sliceId, state);
    }

    // Add to recent predictions
    state.recent_predictions.push({
      predicted: predictedScore,
      actual: actualOutcome,
      timestamp: Date.now()
    });

    // Keep only recent window
    const windowMs = state.calibration_window.window_size_hours * 60 * 60 * 1000;
    const cutoff = Date.now() - windowMs;
    state.recent_predictions = state.recent_predictions.filter(p => p.timestamp > cutoff);

    // Check if we should update calibration
    if (state.recent_predictions.length >= state.calibration_window.min_samples_per_update) {
      this.updateCalibration(sliceId);
    }
  }

  /**
   * Apply calibration to prediction scores
   */
  calibratePredictions(sliceId: string, predictions: number[]): number[] {
    if (!this.enabled) return predictions;

    const state = this.sliceStates.get(sliceId);
    if (!state) {
      return predictions; // No calibration data available
    }

    // Apply deweighting if drift detected
    if (state.deweight_factor < 1.0) {
      const deweightedPredictions = predictions.map(pred => {
        // Blend towards uniform probability (0.5) when deweighting
        return pred * state.deweight_factor + 0.5 * (1 - state.deweight_factor);
      });
      
      console.log(`⚠️ Applying deweighting (${(state.deweight_factor * 100).toFixed(1)}%) to slice ${sliceId} due to calibration drift`);
      return this.applyCalibratedMapping(deweightedPredictions, state);
    }

    return this.applyCalibratedMapping(predictions, state);
  }

  /**
   * Get current calibration metrics for a slice
   */
  getCalibrationMetrics(sliceId: string): CalibrationMetrics | null {
    const state = this.sliceStates.get(sliceId);
    if (!state || state.recent_predictions.length === 0) {
      return null;
    }

    const predictions = state.recent_predictions.map(p => p.predicted);
    const outcomes = state.recent_predictions.map(p => p.actual);

    // Apply current calibration before calculating metrics
    const calibratedPredictions = this.applyCalibratedMapping(predictions, state);

    const ece = this.metricsCalculator.calculateECE(calibratedPredictions, outcomes);
    const mce = this.metricsCalculator.calculateMCE(calibratedPredictions, outcomes);
    const reliability = this.metricsCalculator.calculateReliability(calibratedPredictions, outcomes);
    const sharpness = this.metricsCalculator.calculateSharpness(calibratedPredictions);
    const brierScore = this.metricsCalculator.calculateBrierScore(calibratedPredictions, outcomes);

    // Calculate log loss
    let logLoss = 0;
    for (let i = 0; i < calibratedPredictions.length; i++) {
      const pred = Math.max(1e-10, Math.min(1 - 1e-10, calibratedPredictions[i]));
      logLoss += outcomes[i] * Math.log(pred) + (1 - outcomes[i]) * Math.log(1 - pred);
    }
    logLoss = -logLoss / calibratedPredictions.length;

    return {
      slice_id: sliceId,
      expected_calibration_error: ece,
      maximum_calibration_error: mce,
      reliability_score: reliability,
      sharpness_score: sharpness,
      brier_score: brierScore,
      log_loss: logLoss,
      sample_count: state.recent_predictions.length,
      last_updated: new Date(),
      calibration_temperature: state.current_temperature,
      isotonic_mapping: state.isotonic_regressor.getMapping(),
      slope_factor: state.slope_factor,
      drift_detected: state.drift_start_time !== null,
      auto_deweight_active: state.deweight_factor < 1.0
    };
  }

  /**
   * Force calibration update for a slice
   */
  updateCalibration(sliceId: string): CalibrationUpdate | null {
    const state = this.sliceStates.get(sliceId);
    if (!state || state.recent_predictions.length < state.calibration_window.min_samples_per_update) {
      return null;
    }

    const predictions = state.recent_predictions.map(p => p.predicted);
    const outcomes = state.recent_predictions.map(p => p.actual);

    // Calculate current ECE before update
    const eceBefore = this.metricsCalculator.calculateECE(predictions, outcomes);
    const temperatureBefore = state.current_temperature;

    // Update temperature scaling with exponential weighting
    const tempScaler = new TemperatureScaler();
    tempScaler.fit(predictions, outcomes);
    const newTemperature = tempScaler.getTemperature();
    
    // Apply exponential weighting for temperature updates
    const learningRate = state.calibration_window.temperature_update_rate;
    const updatedTemperature = (1 - learningRate) * state.current_temperature + learningRate * newTemperature;
    
    // Apply slope clamps
    const [minSlope, maxSlope] = state.calibration_window.slope_clamp_range;
    const slopeFactor = 1.0 / updatedTemperature; // Inverse of temperature is the slope factor
    let clampedSlopeFactor = Math.max(minSlope, Math.min(maxSlope, slopeFactor));
    let slopeClampApplied = slopeFactor !== clampedSlopeFactor;
    
    state.current_temperature = 1.0 / clampedSlopeFactor;
    state.slope_factor = clampedSlopeFactor;

    // Update isotonic regression
    state.isotonic_regressor.fit(predictions, outcomes, state.calibration_window.isotonic_bins);

    // Calculate ECE after update
    const calibratedPredictions = this.applyCalibratedMapping(predictions, state);
    const eceAfter = this.metricsCalculator.calculateECE(calibratedPredictions, outcomes);

    // Check for calibration drift
    const eceDelta = Math.abs(eceAfter - state.last_ece);
    const isDriftDetected = eceDelta > state.calibration_window.ece_drift_threshold;

    if (isDriftDetected) {
      if (state.drift_start_time === null) {
        state.drift_start_time = Date.now();
        console.log(`🔄 Calibration drift detected for slice ${sliceId}: ΔECE=${eceDelta.toFixed(4)} > ${state.calibration_window.ece_drift_threshold}`);
      }
      
      // Activate deweighting
      state.deweight_factor = Math.max(0.1, 1.0 - eceDelta * 10); // More drift = more deweighting
    } else {
      // Check if we can restore weighting
      if (state.drift_start_time !== null) {
        const driftDurationMs = Date.now() - state.drift_start_time;
        const maxDriftMs = state.calibration_window.deweight_duration_hours * 60 * 60 * 1000;
        
        if (driftDurationMs > maxDriftMs || eceDelta < state.calibration_window.ece_drift_threshold / 2) {
          // Gradually restore weighting
          state.deweight_factor = Math.min(1.0, state.deweight_factor + 0.1);
          
          if (state.deweight_factor >= 1.0) {
            state.drift_start_time = null;
            console.log(`✅ Calibration drift resolved for slice ${sliceId}, full weighting restored`);
          }
        }
      }
    }

    state.last_ece = eceAfter;
    state.update_count++;

    console.log(`🎯 Calibration updated for slice ${sliceId}: T=${state.current_temperature.toFixed(3)}, slope=${clampedSlopeFactor.toFixed(3)}, ECE=${eceAfter.toFixed(4)}, deweight=${state.deweight_factor.toFixed(2)}`);

    return {
      timestamp: Date.now(),
      slice_id: sliceId,
      predicted_score: predictions.reduce((sum, p) => sum + p, 0) / predictions.length,
      actual_outcome: outcomes.reduce((sum, o) => sum + o, 0) / outcomes.length,
      confidence_level: 0.95,
      temperature_before: temperatureBefore,
      temperature_after: state.current_temperature,
      ece_before: eceBefore,
      ece_after: eceAfter,
      slope_clamp_applied: slopeClampApplied
    };
  }

  /**
   * Get all active slice states
   */
  getAllSliceMetrics(): Map<string, CalibrationMetrics> {
    const metrics = new Map<string, CalibrationMetrics>();
    
    for (const sliceId of this.sliceStates.keys()) {
      const sliceMetrics = this.getCalibrationMetrics(sliceId);
      if (sliceMetrics) {
        metrics.set(sliceId, sliceMetrics);
      }
    }
    
    return metrics;
  }

  /**
   * Run calibration health check across all slices
   */
  runHealthCheck(): {
    healthy_slices: number;
    drifted_slices: number;
    deweighted_slices: number;
    total_slices: number;
    avg_ece: number;
    worst_ece_slice: string | null;
    recommendations: string[];
  } {
    const allMetrics = this.getAllSliceMetrics();
    let healthySlices = 0;
    let driftedSlices = 0;
    let deweightedSlices = 0;
    let totalEce = 0;
    let worstEce = 0;
    let worstEceSlice: string | null = null;
    const recommendations: string[] = [];

    for (const [sliceId, metrics] of allMetrics) {
      totalEce += metrics.expected_calibration_error;
      
      if (metrics.expected_calibration_error > worstEce) {
        worstEce = metrics.expected_calibration_error;
        worstEceSlice = sliceId;
      }

      if (metrics.drift_detected) {
        driftedSlices++;
      } else {
        healthySlices++;
      }

      if (metrics.auto_deweight_active) {
        deweightedSlices++;
      }

      // Generate recommendations
      if (metrics.expected_calibration_error > 0.05) {
        recommendations.push(`High ECE (${(metrics.expected_calibration_error * 100).toFixed(1)}%) detected for slice ${sliceId}`);
      }

      if (metrics.reliability_score < 0.7) {
        recommendations.push(`Poor reliability (${(metrics.reliability_score * 100).toFixed(1)}%) for slice ${sliceId}`);
      }

      if (metrics.sample_count < 100) {
        recommendations.push(`Low sample count (${metrics.sample_count}) for slice ${sliceId} may affect calibration quality`);
      }
    }

    const avgEce = allMetrics.size > 0 ? totalEce / allMetrics.size : 0;

    if (driftedSlices / Math.max(allMetrics.size, 1) > 0.2) {
      recommendations.push('High proportion of slices experiencing drift - consider model retraining');
    }

    if (avgEce > 0.03) {
      recommendations.push('Average ECE above acceptable threshold - review calibration methodology');
    }

    console.log(`🏥 Calibration health check: ${healthySlices}/${allMetrics.size} healthy, ${driftedSlices} drifted, ${deweightedSlices} deweighted, avg_ECE=${(avgEce * 100).toFixed(1)}%`);

    return {
      healthy_slices: healthySlices,
      drifted_slices: driftedSlices,
      deweighted_slices: deweightedSlices,
      total_slices: allMetrics.size,
      avg_ece: avgEce,
      worst_ece_slice: worstEceSlice,
      recommendations
    };
  }

  private initializeSliceState(sliceId: string): SliceCalibrationState {
    return {
      slice_id: sliceId,
      current_temperature: 1.0,
      isotonic_regressor: new IsotonicRegressor(),
      recent_predictions: [],
      slope_factor: 1.0,
      last_ece: 0.0,
      drift_start_time: null,
      deweight_factor: 1.0,
      update_count: 0,
      calibration_window: { ...this.defaultConfig }
    };
  }

  private applyCalibratedMapping(predictions: number[], state: SliceCalibrationState): number[] {
    // Apply temperature scaling first
    const tempScaled = predictions.map(pred => {
      const logit = Math.log(Math.max(pred, 1e-10) / Math.max(1 - pred, 1e-10));
      const scaledLogit = logit / state.current_temperature;
      return 1 / (1 + Math.exp(-scaledLogit));
    });

    // Apply isotonic mapping
    const isotopicMapped = state.isotonic_regressor.transform(tempScaled);

    return isotopicMapped;
  }

  /**
   * Update calibration window configuration
   */
  updateConfiguration(sliceId: string, config: Partial<CalibrationWindowConfig>): void {
    const state = this.sliceStates.get(sliceId);
    if (state) {
      Object.assign(state.calibration_window, config);
      console.log(`🔧 Updated calibration config for slice ${sliceId}:`, config);
    }
  }

  /**
   * Reset calibration for a slice (useful for model updates)
   */
  resetCalibration(sliceId: string): void {
    if (this.sliceStates.has(sliceId)) {
      this.sliceStates.set(sliceId, this.initializeSliceState(sliceId));
      console.log(`🔄 Reset calibration for slice ${sliceId}`);
    }
  }

  /**
   * Enable/disable calibration hygiene
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    console.log(`🎯 Calibration hygiene system ${enabled ? 'ENABLED' : 'DISABLED'}`);
  }

  /**
   * Cleanup old slice states
   */
  cleanup(): void {
    const cutoff = Date.now() - 7 * 24 * 60 * 60 * 1000; // 7 days
    
    for (const [sliceId, state] of this.sliceStates.entries()) {
      if (state.recent_predictions.length === 0 || 
          Math.max(...state.recent_predictions.map(p => p.timestamp)) < cutoff) {
        this.sliceStates.delete(sliceId);
        console.log(`🗑️ Cleaned up stale calibration state for slice ${sliceId}`);
      }
    }
  }
}

// Global instance
export const globalCalibrationHygiene = new CalibrationHygieneSystem();