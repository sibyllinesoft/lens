/**
 * Advanced Calibration System for Gemma Embeddings
 * Implements isotonic calibration with ECE validation and statistical testing
 */

import { z } from 'zod';
import * as fs from 'fs';

const CalibrationConfigSchema = z.object({
  targetECE: z.number().default(0.05),
  minBinSize: z.number().default(50),
  validationSplit: z.number().default(0.2),
  bootstrapSamples: z.number().default(1000),
  confidenceLevel: z.number().default(0.95)
});

export type CalibrationConfig = z.infer<typeof CalibrationConfigSchema>;

interface CalibrationData {
  scores: number[];
  labels: boolean[]; // true for relevant, false for not relevant
}

interface CalibrationResult {
  calibratedScores: number[];
  ece: number;
  reliability: number;
  resolution: number;
  brier: number;
  slope: number;
  intercept: number;
  bins: CalibrationBin[];
  hash: string;
}

interface CalibrationBin {
  binIndex: number;
  confidence: number;
  accuracy: number;
  count: number;
  avgScore: number;
}

/**
 * Isotonic regression implementation for calibration
 */
class IsotonicRegression {
  private x: number[] = [];
  private y: number[] = [];

  fit(scores: number[], labels: boolean[], weights?: number[]): void {
    if (scores.length !== labels.length) {
      throw new Error('Scores and labels must have the same length');
    }

    // Sort by scores
    const indices = scores.map((_, i) => i).sort((a, b) => scores[a] - scores[b]);
    
    this.x = indices.map(i => scores[i]);
    this.y = indices.map(i => labels[i] ? 1.0 : 0.0);
    
    if (weights) {
      const sortedWeights = indices.map(i => weights[i]);
      this.poolAdjacentViolators(sortedWeights);
    } else {
      this.poolAdjacentViolators();
    }
  }

  predict(scores: number[]): number[] {
    return scores.map(score => {
      // Find the appropriate segment using binary search
      let left = 0;
      let right = this.x.length - 1;
      
      if (score <= this.x[left]) return this.y[left];
      if (score >= this.x[right]) return this.y[right];
      
      while (left < right - 1) {
        const mid = Math.floor((left + right) / 2);
        if (this.x[mid] <= score) {
          left = mid;
        } else {
          right = mid;
        }
      }
      
      // Linear interpolation
      const x1 = this.x[left], y1 = this.y[left];
      const x2 = this.x[right], y2 = this.y[right];
      
      if (x2 === x1) return y1;
      return y1 + (y2 - y1) * (score - x1) / (x2 - x1);
    });
  }

  private poolAdjacentViolators(weights?: number[]): void {
    const n = this.x.length;
    if (n <= 1) return;

    const w = weights || new Array(n).fill(1);
    const blocks: Array<{ start: number; end: number; value: number }> = [];
    
    // Initialize blocks
    for (let i = 0; i < n; i++) {
      blocks.push({ start: i, end: i, value: this.y[i] });
    }

    // Pool adjacent violators
    let changed = true;
    while (changed) {
      changed = false;
      
      for (let i = 0; i < blocks.length - 1; i++) {
        if (blocks[i].value > blocks[i + 1].value) {
          // Merge blocks
          const totalWeight = this.sumWeights(w, blocks[i].start, blocks[i].end) +
                             this.sumWeights(w, blocks[i + 1].start, blocks[i + 1].end);
          
          const weightedSum = this.weightedSum(this.y, w, blocks[i].start, blocks[i].end) +
                             this.weightedSum(this.y, w, blocks[i + 1].start, blocks[i + 1].end);
          
          blocks[i] = {
            start: blocks[i].start,
            end: blocks[i + 1].end,
            value: weightedSum / totalWeight
          };
          
          blocks.splice(i + 1, 1);
          changed = true;
          break;
        }
      }
    }

    // Apply pooled values
    for (const block of blocks) {
      for (let i = block.start; i <= block.end; i++) {
        this.y[i] = block.value;
      }
    }
  }

  private sumWeights(weights: number[], start: number, end: number): number {
    let sum = 0;
    for (let i = start; i <= end; i++) {
      sum += weights[i];
    }
    return sum;
  }

  private weightedSum(values: number[], weights: number[], start: number, end: number): number {
    let sum = 0;
    for (let i = start; i <= end; i++) {
      sum += values[i] * weights[i];
    }
    return sum;
  }
}

/**
 * Comprehensive calibration system for Gemma embeddings
 */
export class CalibrationSystem {
  private config: CalibrationConfig;
  private isotonic: IsotonicRegression;

  constructor(config: CalibrationConfig = {}) {
    this.config = CalibrationConfigSchema.parse(config);
    this.isotonic = new IsotonicRegression();
  }

  /**
   * Fit isotonic calibration model on Gemma scores
   */
  fit(data: CalibrationData): CalibrationResult {
    const { scores, labels } = data;
    
    if (scores.length !== labels.length) {
      throw new Error('Scores and labels must have the same length');
    }

    // Split data for validation
    const splitIndex = Math.floor(scores.length * (1 - this.config.validationSplit));
    const trainScores = scores.slice(0, splitIndex);
    const trainLabels = labels.slice(0, splitIndex);
    const valScores = scores.slice(splitIndex);
    const valLabels = labels.slice(splitIndex);

    // Fit isotonic regression
    this.isotonic.fit(trainScores, trainLabels);

    // Calibrate validation set
    const calibratedScores = this.isotonic.predict(valScores);

    // Calculate calibration metrics
    const metrics = this.calculateCalibrationMetrics(valScores, calibratedScores, valLabels);
    
    // Validate ECE threshold
    if (metrics.ece > this.config.targetECE) {
      console.warn(`ECE (${metrics.ece.toFixed(4)}) exceeds target (${this.config.targetECE})`);
    }

    // Generate calibration hash
    const hash = this.generateCalibrationHash(trainScores, trainLabels);

    return {
      calibratedScores,
      ece: metrics.ece,
      reliability: metrics.reliability,
      resolution: metrics.resolution,
      brier: metrics.brier,
      slope: metrics.slope,
      intercept: metrics.intercept,
      bins: metrics.bins,
      hash
    };
  }

  /**
   * Apply calibration to new scores
   */
  predict(scores: number[]): number[] {
    return this.isotonic.predict(scores);
  }

  /**
   * Calculate comprehensive calibration metrics
   */
  private calculateCalibrationMetrics(
    originalScores: number[],
    calibratedScores: number[],
    labels: boolean[]
  ): {
    ece: number;
    reliability: number;
    resolution: number;
    brier: number;
    slope: number;
    intercept: number;
    bins: CalibrationBin[];
  } {
    const n = labels.length;
    const numBins = Math.max(10, Math.floor(n / this.config.minBinSize));
    
    // Create bins based on calibrated scores
    const bins = this.createCalibrationBins(calibratedScores, labels, numBins);
    
    // Expected Calibration Error (ECE)
    const ece = bins.reduce((sum, bin) => {
      return sum + (bin.count / n) * Math.abs(bin.confidence - bin.accuracy);
    }, 0);

    // Reliability and Resolution (Brier decomposition)
    const overallAccuracy = labels.filter(l => l).length / n;
    
    let reliability = 0;
    let resolution = 0;
    
    for (const bin of bins) {
      const binProportion = bin.count / n;
      reliability += binProportion * Math.pow(bin.confidence - bin.accuracy, 2);
      resolution += binProportion * Math.pow(bin.accuracy - overallAccuracy, 2);
    }

    // Brier Score
    const brier = calibratedScores.reduce((sum, score, i) => {
      const label = labels[i] ? 1 : 0;
      return sum + Math.pow(score - label, 2);
    }, 0) / n;

    // Linear regression for slope/intercept
    const { slope, intercept } = this.computeLinearFit(calibratedScores, labels);

    return {
      ece,
      reliability,
      resolution,
      brier,
      slope,
      intercept,
      bins
    };
  }

  /**
   * Create calibration bins for reliability analysis
   */
  private createCalibrationBins(
    scores: number[],
    labels: boolean[],
    numBins: number
  ): CalibrationBin[] {
    const bins: CalibrationBin[] = [];
    const sortedIndices = scores.map((_, i) => i).sort((a, b) => scores[a] - scores[b]);
    
    const binSize = Math.floor(scores.length / numBins);
    
    for (let i = 0; i < numBins; i++) {
      const start = i * binSize;
      const end = i === numBins - 1 ? sortedIndices.length : (i + 1) * binSize;
      
      const binIndices = sortedIndices.slice(start, end);
      const binScores = binIndices.map(idx => scores[idx]);
      const binLabels = binIndices.map(idx => labels[idx]);
      
      const avgScore = binScores.reduce((sum, score) => sum + score, 0) / binScores.length;
      const confidence = avgScore;
      const accuracy = binLabels.filter(l => l).length / binLabels.length;
      
      bins.push({
        binIndex: i,
        confidence,
        accuracy,
        count: binIndices.length,
        avgScore
      });
    }
    
    return bins;
  }

  /**
   * Compute linear regression fit for slope/intercept analysis
   */
  private computeLinearFit(scores: number[], labels: boolean[]): { slope: number; intercept: number } {
    const n = scores.length;
    const x = scores;
    const y = labels.map(l => l ? 1 : 0);
    
    const sumX = x.reduce((sum, xi) => sum + xi, 0);
    const sumY = y.reduce((sum, yi) => sum + yi, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    return { slope, intercept };
  }

  /**
   * Generate hash for calibration configuration
   */
  private generateCalibrationHash(scores: number[], labels: boolean[]): string {
    const configData = {
      scoreHash: this.hashArray(scores),
      labelHash: this.hashArray(labels.map(l => l ? 1 : 0)),
      config: this.config,
      timestamp: Math.floor(Date.now() / 1000) // Unix timestamp
    };
    
    const hashInput = JSON.stringify(configData);
    return require('crypto').createHash('sha256').update(hashInput).digest('hex').substring(0, 16);
  }

  private hashArray(arr: number[]): string {
    const sample = arr.length > 1000 ? 
      arr.filter((_, i) => i % Math.floor(arr.length / 1000) === 0) : 
      arr;
    return require('crypto').createHash('md5').update(JSON.stringify(sample)).digest('hex');
  }

  /**
   * Cross-validation for calibration robustness
   */
  async crossValidateCalibration(
    data: CalibrationData,
    folds: number = 5
  ): Promise<{
    meanECE: number;
    stdECE: number;
    eces: number[];
    confidenceInterval: [number, number];
  }> {
    const { scores, labels } = data;
    const n = scores.length;
    const foldSize = Math.floor(n / folds);
    const eces: number[] = [];
    
    for (let fold = 0; fold < folds; fold++) {
      const testStart = fold * foldSize;
      const testEnd = fold === folds - 1 ? n : (fold + 1) * foldSize;
      
      // Create train/test splits
      const trainScores = [...scores.slice(0, testStart), ...scores.slice(testEnd)];
      const trainLabels = [...labels.slice(0, testStart), ...labels.slice(testEnd)];
      const testScores = scores.slice(testStart, testEnd);
      const testLabels = labels.slice(testStart, testEnd);
      
      // Fit on train, evaluate on test
      const foldCalibrator = new IsotonicRegression();
      foldCalibrator.fit(trainScores, trainLabels);
      const calibratedTestScores = foldCalibrator.predict(testScores);
      
      // Calculate ECE for this fold
      const foldMetrics = this.calculateCalibrationMetrics(testScores, calibratedTestScores, testLabels);
      eces.push(foldMetrics.ece);
    }
    
    const meanECE = eces.reduce((sum, ece) => sum + ece, 0) / eces.length;
    const variance = eces.reduce((sum, ece) => sum + Math.pow(ece - meanECE, 2), 0) / eces.length;
    const stdECE = Math.sqrt(variance);
    
    // Bootstrap confidence interval
    const alpha = 1 - this.config.confidenceLevel;
    const tValue = 1.96; // Approximate for 95% confidence
    const marginOfError = tValue * (stdECE / Math.sqrt(folds));
    
    return {
      meanECE,
      stdECE,
      eces,
      confidenceInterval: [meanECE - marginOfError, meanECE + marginOfError]
    };
  }

  /**
   * Save calibration model for production use
   */
  async saveCalibrationModel(result: CalibrationResult, outputPath: string): Promise<void> {
    const modelData = {
      version: '3.0.0', // isotonic_v3
      type: 'isotonic_regression',
      hash: result.hash,
      metrics: {
        ece: result.ece,
        reliability: result.reliability,
        resolution: result.resolution,
        brier: result.brier,
        slope: result.slope,
        intercept: result.intercept
      },
      bins: result.bins,
      config: this.config,
      timestamp: new Date().toISOString(),
      // Serialized isotonic model parameters would go here
      // In practice, you'd save the fitted x,y points for reconstruction
      modelParams: {
        x: [], // Would contain this.isotonic.x
        y: []  // Would contain this.isotonic.y
      }
    };
    
    await fs.promises.writeFile(
      outputPath,
      JSON.stringify(modelData, null, 2),
      'utf8'
    );
  }

  /**
   * Load calibration model from saved state
   */
  async loadCalibrationModel(inputPath: string): Promise<void> {
    const modelData = JSON.parse(await fs.promises.readFile(inputPath, 'utf8'));
    
    if (modelData.version !== '3.0.0') {
      throw new Error(`Unsupported calibration model version: ${modelData.version}`);
    }
    
    // Reconstruct isotonic regression from saved parameters
    // In practice, you'd restore this.isotonic.x and this.isotonic.y
    // from modelData.modelParams
  }
}

/**
 * Batch calibration processor for different embedding dimensions
 */
export class BatchCalibrationProcessor {
  private calibrators: Map<string, CalibrationSystem>;

  constructor() {
    this.calibrators = new Map();
  }

  /**
   * Process calibration for multiple embedding dimensions
   */
  async processBatch(datasets: Map<string, CalibrationData>): Promise<Map<string, CalibrationResult>> {
    const results = new Map<string, CalibrationResult>();
    
    for (const [modelId, data] of datasets) {
      console.log(`Processing calibration for ${modelId}...`);
      
      const calibrator = new CalibrationSystem();
      this.calibrators.set(modelId, calibrator);
      
      const result = calibrator.fit(data);
      results.set(modelId, result);
      
      console.log(`${modelId} - ECE: ${result.ece.toFixed(4)}, Brier: ${result.brier.toFixed(4)}`);
    }
    
    return results;
  }

  /**
   * Compare calibration quality across models
   */
  compareCalibrations(results: Map<string, CalibrationResult>): {
    bestModel: string;
    comparison: Array<{
      model: string;
      ece: number;
      brier: number;
      reliability: number;
      rank: number;
    }>;
  } {
    const comparison = Array.from(results.entries()).map(([model, result]) => ({
      model,
      ece: result.ece,
      brier: result.brier,
      reliability: result.reliability,
      rank: 0 // Will be filled below
    }));
    
    // Rank by ECE (lower is better)
    comparison.sort((a, b) => a.ece - b.ece);
    comparison.forEach((item, index) => {
      item.rank = index + 1;
    });
    
    const bestModel = comparison[0].model;
    
    return { bestModel, comparison };
  }
}