/**
 * Isotonic Calibrated Reranker - Phase B3 Enhancement
 * Applies isotonic regression to calibrate the existing logistic reranker scores
 * Target: 12ms → 6-8ms (~40% improvement) with maintained quality
 */

import type { SearchHit } from './span_resolver/types.js';
import type { SearchContext } from '../types/core.js';
import { LearnedReranker } from './learned-reranker.js';
import { LensTracer } from '../telemetry/tracer.js';

export interface CalibrationPoint {
  predicted_score: number;
  actual_relevance: number;
}

export interface IsotonicConfig {
  enabled: boolean;
  minCalibrationData: number;    // Minimum points needed for calibration
  confidenceCutoff: number;      // Skip reranking below this confidence
  maxLatencyMs: number;          // Emergency cutoff for Stage-C budget
  calibrationUpdateFreq: number; // How often to refit calibration
}

/**
 * Isotonic regression calibrator for improving score reliability
 * Uses Pool-Adjacent-Violators Algorithm (PAVA) for monotonic calibration
 */
export class IsotonicCalibrator {
  private calibrationData: CalibrationPoint[] = [];
  private calibrationMap: Map<number, number> = new Map(); // predicted -> calibrated
  private isFitted = false;
  private lastUpdateCount = 0;

  constructor(private config: IsotonicConfig) {}

  /**
   * Add training data point for calibration
   */
  addCalibrationPoint(predictedScore: number, actualRelevance: number): void {
    this.calibrationData.push({
      predicted_score: predictedScore,
      actual_relevance: actualRelevance
    });

    // Keep calibration data bounded to prevent memory growth
    if (this.calibrationData.length > 5000) {
      this.calibrationData = this.calibrationData.slice(-3000); // Keep recent 3000
    }
  }

  /**
   * Fit isotonic regression using Pool-Adjacent-Violators Algorithm (PAVA)
   * This ensures monotonicity: higher predicted scores → higher calibrated scores
   */
  fitCalibration(): boolean {
    const span = LensTracer.createChildSpan('isotonic_calibration_fit', {
      'calibration.data_points': this.calibrationData.length,
      'calibration.min_required': this.config.minCalibrationData
    });

    try {
      if (this.calibrationData.length < this.config.minCalibrationData) {
        span.setAttributes({ skipped: true, reason: 'insufficient_data' });
        return false;
      }

      // Sort by predicted score for isotonic regression
      const sortedData = [...this.calibrationData].sort((a, b) => a.predicted_score - b.predicted_score);

      // Apply Pool-Adjacent-Violators Algorithm
      const calibrated = this.applyPAVA(sortedData);
      
      // Build interpolation map
      this.calibrationMap.clear();
      for (const point of calibrated) {
        this.calibrationMap.set(point.predicted_score, point.actual_relevance);
      }

      this.isFitted = true;
      this.lastUpdateCount = this.calibrationData.length;

      span.setAttributes({
        success: true,
        calibration_points: calibrated.length,
        score_range_min: Math.min(...calibrated.map(p => p.predicted_score)),
        score_range_max: Math.max(...calibrated.map(p => p.predicted_score))
      });

      return true;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      return false;
    } finally {
      span.end();
    }
  }

  /**
   * Pool-Adjacent-Violators Algorithm implementation
   * Ensures isotonic (monotonically increasing) calibration
   */
  private applyPAVA(sortedData: CalibrationPoint[]): CalibrationPoint[] {
    if (sortedData.length === 0) return [];

    // Initialize with grouped data points
    const groups: Array<{
      predicted_score: number;
      actual_relevance: number;
      weight: number;
    }> = [];

    // Group adjacent points with same predicted score
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
      actual_relevance: group.actual_relevance
    }));
  }

  /**
   * Calibrate a predicted score using isotonic regression
   */
  calibrateScore(predictedScore: number): number {
    if (!this.isFitted || this.calibrationMap.size === 0) {
      return predictedScore; // No calibration available
    }

    // Linear interpolation between calibration points
    const sortedScores = Array.from(this.calibrationMap.keys()).sort((a, b) => a - b);
    
    // Handle edge cases
    if (predictedScore <= sortedScores[0]!) {
      return this.calibrationMap.get(sortedScores[0]!) || predictedScore;
    }
    if (predictedScore >= sortedScores[sortedScores.length - 1]!) {
      return this.calibrationMap.get(sortedScores[sortedScores.length - 1]!) || predictedScore;
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
    const y0 = this.calibrationMap.get(x0)!;
    const y1 = this.calibrationMap.get(x1)!;

    // Linear interpolation
    const alpha = (predictedScore - x0) / (x1 - x0);
    return y0 + alpha * (y1 - y0);
  }

  /**
   * Check if calibration needs updating
   */
  needsUpdate(): boolean {
    return (
      !this.isFitted ||
      (this.calibrationData.length - this.lastUpdateCount) >= this.config.calibrationUpdateFreq
    );
  }

  /**
   * Get calibration statistics for monitoring
   */
  getStats() {
    return {
      fitted: this.isFitted,
      calibration_points: this.calibrationData.length,
      map_size: this.calibrationMap.size,
      last_update_count: this.lastUpdateCount
    };
  }
}

/**
 * Enhanced reranker with isotonic calibration and confidence-aware processing
 * Builds on the existing LearnedReranker with improved score reliability
 */
export class IsotonicCalibratedReranker {
  private baseReranker: LearnedReranker;
  private calibrator: IsotonicCalibrator;
  private config: IsotonicConfig;

  constructor(config: Partial<IsotonicConfig> = {}) {
    this.config = {
      enabled: config.enabled ?? true,
      minCalibrationData: config.minCalibrationData ?? 50,
      confidenceCutoff: config.confidenceCutoff ?? 0.12,
      maxLatencyMs: config.maxLatencyMs ?? 8, // Target: 6-8ms for Stage-C
      calibrationUpdateFreq: config.calibrationUpdateFreq ?? 100,
      ...config
    };

    this.baseReranker = new LearnedReranker({
      enabled: true,
      nlThreshold: 0.5,
      minCandidates: 10,
      maxLatencyMs: this.config.maxLatencyMs
    });

    this.calibrator = new IsotonicCalibrator(this.config);

    console.log(`🎯 IsotonicCalibratedReranker initialized: enabled=${this.config.enabled}, confidenceCutoff=${this.config.confidenceCutoff}`);
  }

  /**
   * Rerank search hits with isotonic calibration and confidence gating
   */
  async rerank(
    hits: SearchHit[],
    context: SearchContext
  ): Promise<SearchHit[]> {
    const span = LensTracer.createChildSpan('isotonic_calibrated_rerank', {
      'candidates': hits.length,
      'query': context.query,
      'enabled': this.config.enabled
    });

    const startTime = Date.now();

    try {
      // Early exit if disabled
      if (!this.config.enabled) {
        span.setAttributes({ skipped: true, reason: 'disabled' });
        return hits;
      }

      // Emergency latency cutoff
      const checkLatency = () => {
        const elapsed = Date.now() - startTime;
        if (elapsed > this.config.maxLatencyMs) {
          throw new Error(`Latency budget exceeded: ${elapsed}ms > ${this.config.maxLatencyMs}ms`);
        }
        return elapsed;
      };

      // Update calibration if needed (async, don't block)
      if (this.calibrator.needsUpdate()) {
        setImmediate(() => this.calibrator.fitCalibration());
      }

      checkLatency();

      // Get base reranker scores
      const baseReranked = await this.baseReranker.rerank(hits, context);

      checkLatency();

      // Apply confidence-aware processing
      const confidenceAwareResults = this.applyConfidenceGating(baseReranked, context);

      checkLatency();

      // Apply isotonic calibration to scores
      const calibratedResults = this.applyCalibratedScoring(confidenceAwareResults);

      const latency = Date.now() - startTime;

      span.setAttributes({
        success: true,
        latency_ms: latency,
        candidates_processed: hits.length,
        candidates_gated: hits.length - confidenceAwareResults.length,
        calibration_applied: this.calibrator.getStats().fitted
      });

      console.log(`🎯 Isotonic reranking: ${hits.length}→${calibratedResults.length} in ${latency}ms`);

      return calibratedResults;

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });

      // Fallback to base reranker or original hits
      console.warn(`Isotonic reranker failed: ${errorMsg}, falling back`);
      
      try {
        return await this.baseReranker.rerank(hits, context);
      } catch (fallbackError) {
        return hits; // Final fallback
      }

    } finally {
      span.end();
    }
  }

  /**
   * Apply confidence gating - skip reranking for low-confidence candidates
   */
  private applyConfidenceGating(hits: SearchHit[], context: SearchContext): SearchHit[] {
    if (this.config.confidenceCutoff <= 0) {
      return hits; // No gating
    }

    // Simple confidence estimation based on query characteristics and hit features
    const confidence = this.estimateQueryConfidence(context);
    
    if (confidence < this.config.confidenceCutoff) {
      // Low confidence query, return original hits without reranking
      return hits;
    }

    // Filter out very low scoring hits to reduce processing time
    const scoreThreshold = Math.max(0.1, this.config.confidenceCutoff);
    return hits.filter(hit => hit.score >= scoreThreshold);
  }

  /**
   * Estimate confidence for query-based gating decisions
   */
  private estimateQueryConfidence(context: SearchContext): number {
    const query = context.query.toLowerCase();
    
    let confidence = 0.5; // Base confidence

    // Natural language queries tend to benefit more from semantic reranking
    if (/\b(how|what|where|when|why|find|search|get|show)\b/.test(query)) {
      confidence += 0.3;
    }

    // Multi-word queries benefit from semantic understanding
    if (query.split(' ').length > 2) {
      confidence += 0.2;
    }

    // Very short queries may not have enough context
    if (query.length < 5) {
      confidence -= 0.2;
    }

    // Code-like queries (single identifiers) may not need semantic reranking
    if (/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(query)) {
      confidence -= 0.3;
    }

    return Math.max(0, Math.min(1, confidence));
  }

  /**
   * Apply isotonic calibration to improve score reliability
   */
  private applyCalibratedScoring(hits: SearchHit[]): SearchHit[] {
    if (!this.calibrator.getStats().fitted) {
      return hits; // No calibration available
    }

    return hits.map(hit => ({
      ...hit,
      score: this.calibrator.calibrateScore(hit.score)
    }));
  }

  /**
   * Record training example for calibration improvement
   */
  recordCalibrationExample(
    hit: SearchHit,
    predictedScore: number,
    actualRelevance: number
  ): void {
    this.calibrator.addCalibrationPoint(predictedScore, actualRelevance);
    
    // Also pass through to base reranker if it supports training
    if ('recordTrainingExample' in this.baseReranker) {
      // Note: We'd need the full context to call this properly
      // For now, just accumulate in calibrator
    }
  }

  /**
   * Get comprehensive statistics
   */
  getStats() {
    return {
      config: this.config,
      base_reranker: this.baseReranker.getStats(),
      calibrator: this.calibrator.getStats()
    };
  }

  /**
   * Update configuration for A/B testing and tuning
   */
  updateConfig(newConfig: Partial<IsotonicConfig>): void {
    this.config = { ...this.config, ...newConfig };
    
    // Update base reranker config if needed
    this.baseReranker.updateConfig({
      maxLatencyMs: this.config.maxLatencyMs
    });
    
    console.log(`🎯 Isotonic reranker config updated: ${JSON.stringify(this.config)}`);
  }
}