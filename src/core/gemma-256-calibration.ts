/**
 * Gemma-256 Production Calibration System
 * 
 * Per-dimension L2 normalization and isotonic regression calibration
 * for both 256d and 768d vectors. Ensures ECE ≤ 0.05 and prevents
 * calibration reuse between dimensions.
 */

import type { SearchHit } from './span_resolver/types.js';
import { LensTracer } from '../telemetry/tracer.js';

export interface VectorDimensions {
  '256': Float32Array;
  '768': Float32Array;
}

export interface CalibrationConfig {
  enabled: boolean;
  minCalibrationData: number;    // Minimum points needed per dimension
  confidenceCutoff: number;      // Skip calibration below this confidence
  maxLatencyMs: number;          // Emergency cutoff for calibration
  updateFreq: number;            // How often to refit calibration
  eceThreshold: number;          // Maximum allowed ECE (0.05 = 5%)
  l2Normalize: boolean;          // Force L2 normalization (mandatory)
}

export interface CalibrationPoint {
  predicted_score: number;
  actual_relevance: number;
  vector_dimension: '256' | '768';
  l2_normalized: boolean;
}

export interface CalibrationMetrics {
  ece: number;                   // Expected Calibration Error
  slope: number;                 // Isotonic regression slope
  intercept: number;             // Isotonic regression intercept
  data_points: number;           // Calibration data size
  dimension: '256' | '768';      // Vector dimension
  last_updated: Date;
}

/**
 * L2 Vector Normalizer - ensures all vectors are properly normalized
 * per dimension without assuming defaults
 */
export class L2VectorNormalizer {
  private normalizationCache: Map<string, Float32Array> = new Map();
  private cacheHits = 0;
  private cacheMisses = 0;

  /**
   * L2 normalize vector ensuring unit length
   */
  normalize(vector: Float32Array, cacheKey?: string): Float32Array {
    if (cacheKey && this.normalizationCache.has(cacheKey)) {
      this.cacheHits++;
      return this.normalizationCache.get(cacheKey)!;
    }

    this.cacheMisses++;

    // Calculate L2 norm
    let norm = 0;
    for (let i = 0; i < vector.length; i++) {
      norm += vector[i]! * vector[i]!;
    }
    norm = Math.sqrt(norm);

    // Avoid division by zero
    if (norm === 0 || !isFinite(norm)) {
      console.warn(`⚠️ Zero or invalid norm detected: ${norm}, returning original vector`);
      return vector;
    }

    // Normalize to unit length
    const normalized = new Float32Array(vector.length);
    for (let i = 0; i < vector.length; i++) {
      normalized[i] = vector[i]! / norm;
    }

    // Cache if key provided
    if (cacheKey && this.normalizationCache.size < 10000) {
      this.normalizationCache.set(cacheKey, normalized);
    }

    return normalized;
  }

  /**
   * Verify vector is L2 normalized (unit length)
   */
  isNormalized(vector: Float32Array, tolerance = 1e-6): boolean {
    let norm = 0;
    for (let i = 0; i < vector.length; i++) {
      norm += vector[i]! * vector[i]!;
    }
    return Math.abs(Math.sqrt(norm) - 1.0) < tolerance;
  }

  getStats() {
    return {
      cache_size: this.normalizationCache.size,
      cache_hits: this.cacheHits,
      cache_misses: this.cacheMisses,
      hit_rate: this.cacheHits / (this.cacheHits + this.cacheMisses)
    };
  }

  clearCache() {
    this.normalizationCache.clear();
    this.cacheHits = 0;
    this.cacheMisses = 0;
  }
}

/**
 * Per-dimension Isotonic Calibrator using Pool-Adjacent-Violators Algorithm (PAVA)
 * Maintains separate calibration for 256d and 768d vectors
 */
export class PerDimensionIsotonicCalibrator {
  private calibrationData256: CalibrationPoint[] = [];
  private calibrationData768: CalibrationPoint[] = [];
  private calibrationMap256: Map<number, number> = new Map();
  private calibrationMap768: Map<number, number> = new Map();
  private isFitted256 = false;
  private isFitted768 = false;
  private l2Normalizer = new L2VectorNormalizer();

  constructor(private config: CalibrationConfig) {
    if (!config.l2Normalize) {
      throw new Error('L2 normalization is mandatory for production deployment');
    }
  }

  /**
   * Add calibration point for specific dimension
   */
  addCalibrationPoint(
    predictedScore: number, 
    actualRelevance: number, 
    dimension: '256' | '768',
    isL2Normalized = false
  ): void {
    const point: CalibrationPoint = {
      predicted_score: predictedScore,
      actual_relevance: actualRelevance,
      vector_dimension: dimension,
      l2_normalized: isL2Normalized
    };

    if (dimension === '256') {
      this.calibrationData256.push(point);
      // Keep bounded to prevent memory growth
      if (this.calibrationData256.length > 5000) {
        this.calibrationData256 = this.calibrationData256.slice(-3000);
      }
    } else {
      this.calibrationData768.push(point);
      if (this.calibrationData768.length > 5000) {
        this.calibrationData768 = this.calibrationData768.slice(-3000);
      }
    }
  }

  /**
   * Fit isotonic regression for specific dimension
   */
  fitCalibration(dimension: '256' | '768'): CalibrationMetrics | null {
    const span = LensTracer.createChildSpan('gemma_256_calibration_fit', {
      dimension,
      'config.ece_threshold': this.config.eceThreshold,
      'config.l2_normalize': this.config.l2Normalize
    });

    try {
      const data = dimension === '256' ? this.calibrationData256 : this.calibrationData768;
      
      if (data.length < this.config.minCalibrationData) {
        span.setAttributes({ 
          skipped: true, 
          reason: 'insufficient_data',
          data_points: data.length,
          min_required: this.config.minCalibrationData
        });
        return null;
      }

      // Verify L2 normalization requirement
      const nonNormalizedCount = data.filter(p => !p.l2_normalized).length;
      if (nonNormalizedCount > 0) {
        console.warn(`⚠️ Found ${nonNormalizedCount} non-L2-normalized points for ${dimension}d`);
      }

      // Sort by predicted score for isotonic regression
      const sortedData = [...data].sort((a, b) => a.predicted_score - b.predicted_score);

      // Apply PAVA
      const calibrated = this.applyPAVA(sortedData);
      
      // Build interpolation map
      const calibrationMap = dimension === '256' ? this.calibrationMap256 : this.calibrationMap768;
      calibrationMap.clear();
      for (const point of calibrated) {
        calibrationMap.set(point.predicted_score, point.actual_relevance);
      }

      // Calculate calibration metrics
      const metrics = this.calculateCalibrationMetrics(calibrated, dimension);

      // Validate ECE threshold
      if (metrics.ece > this.config.eceThreshold) {
        span.setAttributes({ 
          failed: true, 
          reason: 'ece_threshold_exceeded',
          ece: metrics.ece,
          threshold: this.config.eceThreshold
        });
        console.error(`🚨 ECE threshold exceeded for ${dimension}d: ${metrics.ece.toFixed(4)} > ${this.config.eceThreshold}`);
        return null;
      }

      // Mark as fitted
      if (dimension === '256') {
        this.isFitted256 = true;
      } else {
        this.isFitted768 = true;
      }

      span.setAttributes({
        success: true,
        calibration_points: calibrated.length,
        ece: metrics.ece,
        slope: metrics.slope,
        intercept: metrics.intercept
      });

      console.log(`🎯 ${dimension}d calibration fitted: ECE=${metrics.ece.toFixed(4)}, slope=${metrics.slope.toFixed(3)}`);

      return metrics;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      console.error(`❌ Calibration fitting failed for ${dimension}d:`, error);
      return null;
    } finally {
      span.end();
    }
  }

  /**
   * Calibrate score for specific dimension using fitted isotonic regression
   */
  calibrateScore(predictedScore: number, dimension: '256' | '768'): number {
    const isFitted = dimension === '256' ? this.isFitted256 : this.isFitted768;
    const calibrationMap = dimension === '256' ? this.calibrationMap256 : this.calibrationMap768;

    if (!isFitted || calibrationMap.size === 0) {
      return predictedScore; // No calibration available
    }

    return this.interpolateScore(predictedScore, calibrationMap);
  }

  /**
   * L2 normalize vector and ensure proper tracking
   */
  normalizeVector(vector: Float32Array, docId?: string): Float32Array {
    if (!this.config.l2Normalize) {
      throw new Error('L2 normalization is mandatory');
    }

    const cacheKey = docId ? `${docId}_${vector.length}d` : undefined;
    return this.l2Normalizer.normalize(vector, cacheKey);
  }

  /**
   * Pool-Adjacent-Violators Algorithm implementation
   */
  private applyPAVA(sortedData: CalibrationPoint[]): CalibrationPoint[] {
    if (sortedData.length === 0) return [];

    // Group adjacent points with same predicted score
    const groups: Array<{
      predicted_score: number;
      actual_relevance: number;
      weight: number;
    }> = [];

    let currentGroup = {
      predicted_score: sortedData[0]!.predicted_score,
      actual_relevance: sortedData[0]!.actual_relevance,
      weight: 1
    };

    for (let i = 1; i < sortedData.length; i++) {
      const point = sortedData[i]!;
      if (Math.abs(point.predicted_score - currentGroup.predicted_score) < 1e-6) {
        // Same predicted score, update weighted average
        currentGroup.actual_relevance = 
          (currentGroup.actual_relevance * currentGroup.weight + point.actual_relevance) / 
          (currentGroup.weight + 1);
        currentGroup.weight++;
      } else {
        groups.push(currentGroup);
        currentGroup = {
          predicted_score: point.predicted_score,
          actual_relevance: point.actual_relevance,
          weight: 1
        };
      }
    }
    groups.push(currentGroup);

    // Apply PAVA to ensure monotonicity
    const result = [...groups];
    let changed = true;

    while (changed) {
      changed = false;
      for (let i = 0; i < result.length - 1; i++) {
        const curr = result[i]!;
        const next = result[i + 1]!;
        
        if (curr.actual_relevance > next.actual_relevance) {
          // Violation found, merge adjacent groups
          const totalWeight = curr.weight + next.weight;
          const mergedRelevance = 
            (curr.actual_relevance * curr.weight + next.actual_relevance * next.weight) / totalWeight;
          
          result[i] = {
            predicted_score: curr.predicted_score,
            actual_relevance: mergedRelevance,
            weight: totalWeight
          };
          
          result.splice(i + 1, 1);
          changed = true;
          break;
        }
      }
    }

    return result.map(group => ({
      predicted_score: group.predicted_score,
      actual_relevance: group.actual_relevance,
      vector_dimension: sortedData[0]!.vector_dimension,
      l2_normalized: true
    }));
  }

  /**
   * Linear interpolation between calibration points
   */
  private interpolateScore(predictedScore: number, calibrationMap: Map<number, number>): number {
    const sortedScores = Array.from(calibrationMap.keys()).sort((a, b) => a - b);
    
    // Handle edge cases
    if (predictedScore <= sortedScores[0]!) {
      return calibrationMap.get(sortedScores[0]!) || predictedScore;
    }
    if (predictedScore >= sortedScores[sortedScores.length - 1]!) {
      return calibrationMap.get(sortedScores[sortedScores.length - 1]!) || predictedScore;
    }

    // Find interpolation bounds
    let lowerIdx = 0;
    for (let i = 0; i < sortedScores.length - 1; i++) {
      if (sortedScores[i]! <= predictedScore && predictedScore < sortedScores[i + 1]!) {
        lowerIdx = i;
        break;
      }
    }

    const x0 = sortedScores[lowerIdx]!;
    const x1 = sortedScores[lowerIdx + 1]!;
    const y0 = calibrationMap.get(x0)!;
    const y1 = calibrationMap.get(x1)!;

    // Linear interpolation
    const alpha = (predictedScore - x0) / (x1 - x0);
    return y0 + alpha * (y1 - y0);
  }

  /**
   * Calculate calibration metrics including ECE, slope, intercept
   */
  private calculateCalibrationMetrics(
    calibratedData: CalibrationPoint[], 
    dimension: '256' | '768'
  ): CalibrationMetrics {
    if (calibratedData.length === 0) {
      return {
        ece: 1.0,
        slope: 0,
        intercept: 0,
        data_points: 0,
        dimension,
        last_updated: new Date()
      };
    }

    // Calculate Expected Calibration Error (ECE)
    // ECE = Σ |predicted_probability - actual_probability| * bin_weight
    const numBins = 10;
    const bins = Array(numBins).fill(0).map(() => ({ 
      predicted: 0, 
      actual: 0, 
      count: 0 
    }));

    for (const point of calibratedData) {
      const binIdx = Math.min(Math.floor(point.predicted_score * numBins), numBins - 1);
      bins[binIdx]!.predicted += point.predicted_score;
      bins[binIdx]!.actual += point.actual_relevance;
      bins[binIdx]!.count++;
    }

    let ece = 0;
    for (const bin of bins) {
      if (bin.count > 0) {
        const avgPredicted = bin.predicted / bin.count;
        const avgActual = bin.actual / bin.count;
        const weight = bin.count / calibratedData.length;
        ece += Math.abs(avgPredicted - avgActual) * weight;
      }
    }

    // Calculate slope and intercept using least squares
    const n = calibratedData.length;
    const sumX = calibratedData.reduce((sum, p) => sum + p.predicted_score, 0);
    const sumY = calibratedData.reduce((sum, p) => sum + p.actual_relevance, 0);
    const sumXY = calibratedData.reduce((sum, p) => sum + p.predicted_score * p.actual_relevance, 0);
    const sumX2 = calibratedData.reduce((sum, p) => sum + p.predicted_score * p.predicted_score, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    return {
      ece,
      slope: isFinite(slope) ? slope : 1.0,
      intercept: isFinite(intercept) ? intercept : 0.0,
      data_points: calibratedData.length,
      dimension,
      last_updated: new Date()
    };
  }

  /**
   * Check if calibration needs updating for specific dimension
   */
  needsUpdate(dimension: '256' | '768'): boolean {
    const isFitted = dimension === '256' ? this.isFitted256 : this.isFitted768;
    const data = dimension === '256' ? this.calibrationData256 : this.calibrationData768;
    
    return !isFitted || (data.length % this.config.updateFreq === 0);
  }

  /**
   * Get comprehensive statistics for monitoring
   */
  getStats() {
    const metrics256 = this.isFitted256 ? 
      this.calculateCalibrationMetrics(this.calibrationData256, '256') : null;
    const metrics768 = this.isFitted768 ? 
      this.calculateCalibrationMetrics(this.calibrationData768, '768') : null;

    return {
      config: this.config,
      calibration_256d: {
        fitted: this.isFitted256,
        data_points: this.calibrationData256.length,
        map_size: this.calibrationMap256.size,
        metrics: metrics256
      },
      calibration_768d: {
        fitted: this.isFitted768,
        data_points: this.calibrationData768.length,
        map_size: this.calibrationMap768.size,
        metrics: metrics768
      },
      l2_normalizer: this.l2Normalizer.getStats()
    };
  }

  /**
   * Force refit of calibration for specific dimension
   */
  async refitCalibration(dimension: '256' | '768'): Promise<boolean> {
    if (dimension === '256') {
      this.isFitted256 = false;
    } else {
      this.isFitted768 = false;
    }
    
    const metrics = this.fitCalibration(dimension);
    return metrics !== null && metrics.ece <= this.config.eceThreshold;
  }

  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<CalibrationConfig>): void {
    if (newConfig.l2Normalize === false) {
      throw new Error('Cannot disable L2 normalization in production');
    }
    
    this.config = { ...this.config, ...newConfig };
    console.log(`🎯 Calibration config updated: ${JSON.stringify(this.config)}`);
  }
}

/**
 * Production Gemma-256 Calibration Manager
 * Orchestrates L2 normalization and per-dimension calibration
 */
export class Gemma256CalibrationManager {
  private calibrator: PerDimensionIsotonicCalibrator;
  private isProduction: boolean;

  constructor(config: Partial<CalibrationConfig> = {}, isProduction = true) {
    this.isProduction = isProduction;
    
    // Production defaults with strict requirements
    const productionConfig: CalibrationConfig = {
      enabled: true,
      minCalibrationData: 100,        // Minimum for reliable calibration
      confidenceCutoff: 0.05,         // Very low cutoff for production
      maxLatencyMs: 2,                // Tight latency budget
      updateFreq: 50,                 // Frequent updates for drift detection
      eceThreshold: 0.05,             // 5% ECE threshold per requirement
      l2Normalize: true,              // Mandatory L2 normalization
      ...config
    };

    if (!productionConfig.l2Normalize) {
      throw new Error('L2 normalization is mandatory for production deployment');
    }

    this.calibrator = new PerDimensionIsotonicCalibrator(productionConfig);
    
    console.log(`🚀 Gemma-256 Calibration Manager initialized (production=${isProduction})`);
    console.log(`   ECE threshold: ${productionConfig.eceThreshold}`);
    console.log(`   L2 normalization: ${productionConfig.l2Normalize}`);
  }

  /**
   * Process and calibrate vectors with proper L2 normalization
   */
  processVectors(vectors: VectorDimensions, docId?: string): VectorDimensions {
    const normalized256 = this.calibrator.normalizeVector(vectors['256'], docId);
    const normalized768 = this.calibrator.normalizeVector(vectors['768'], docId);

    // Verify normalization
    const normalizer = this.calibrator['l2Normalizer'];
    if (!normalizer.isNormalized(normalized256)) {
      console.warn(`⚠️ 256d vector not properly normalized for ${docId}`);
    }
    if (!normalizer.isNormalized(normalized768)) {
      console.warn(`⚠️ 768d vector not properly normalized for ${docId}`);
    }

    return {
      '256': normalized256,
      '768': normalized768
    };
  }

  /**
   * Calibrate scores with dimension-specific calibration
   */
  calibrateScores(scores256: number, scores768: number): { 
    calibrated256: number, 
    calibrated768: number 
  } {
    return {
      calibrated256: this.calibrator.calibrateScore(scores256, '256'),
      calibrated768: this.calibrator.calibrateScore(scores768, '768')
    };
  }

  /**
   * Record training examples for calibration improvement
   */
  recordTrainingExamples(
    predicted256: number,
    predicted768: number,
    actualRelevance: number
  ): void {
    this.calibrator.addCalibrationPoint(predicted256, actualRelevance, '256', true);
    this.calibrator.addCalibrationPoint(predicted768, actualRelevance, '768', true);
  }

  /**
   * Validate calibration meets production requirements
   */
  async validateCalibration(): Promise<{ 
    valid: boolean, 
    metrics256?: CalibrationMetrics, 
    metrics768?: CalibrationMetrics 
  }> {
    const metrics256 = this.calibrator.fitCalibration('256');
    const metrics768 = this.calibrator.fitCalibration('768');

    const valid = (metrics256?.ece || 1) <= 0.05 && (metrics768?.ece || 1) <= 0.05;

    if (!valid) {
      console.error(`🚨 Calibration validation failed:`);
      if (metrics256) console.error(`   256d ECE: ${metrics256.ece.toFixed(4)} > 0.05`);
      if (metrics768) console.error(`   768d ECE: ${metrics768.ece.toFixed(4)} > 0.05`);
    } else {
      console.log(`✅ Calibration validation passed:`);
      if (metrics256) console.log(`   256d ECE: ${metrics256.ece.toFixed(4)} ≤ 0.05`);
      if (metrics768) console.log(`   768d ECE: ${metrics768.ece.toFixed(4)} ≤ 0.05`);
    }

    return { valid, metrics256: metrics256 || undefined, metrics768: metrics768 || undefined };
  }

  /**
   * Get comprehensive statistics
   */
  getStats() {
    return {
      production_mode: this.isProduction,
      ...this.calibrator.getStats()
    };
  }

  /**
   * Update configuration with production safety checks
   */
  updateConfig(newConfig: Partial<CalibrationConfig>): void {
    if (this.isProduction) {
      if (newConfig.l2Normalize === false) {
        throw new Error('Cannot disable L2 normalization in production');
      }
      if (newConfig.eceThreshold && newConfig.eceThreshold > 0.05) {
        throw new Error('Cannot exceed ECE threshold of 0.05 in production');
      }
    }
    
    this.calibrator.updateConfig(newConfig);
  }
}