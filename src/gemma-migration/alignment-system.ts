/**
 * Gemma Vector Alignment System
 * Ensures proper L2 normalization, cosine similarity, and score alignment
 */

import { z } from 'zod';
import * as fs from 'fs';
import * as path from 'path';

const AlignmentConfigSchema = z.object({
  enforceL2Normalization: z.boolean().default(true),
  useCosineSimilarity: z.boolean().default(true),
  affineRescaleParams: z.object({
    slope: z.number(),
    intercept: z.number()
  }).optional(),
  calibrationHash: z.string().optional()
});

export type AlignmentConfig = z.infer<typeof AlignmentConfigSchema>;

/**
 * Vector alignment utilities for Gemma embeddings
 */
export class VectorAlignment {
  private config: AlignmentConfig;

  constructor(config: AlignmentConfig) {
    this.config = AlignmentConfigSchema.parse(config);
  }

  /**
   * Enforce L2 normalization on vectors (don't assume TEI defaults)
   */
  normalizeL2(vector: Float32Array): Float32Array {
    if (!this.config.enforceL2Normalization) {
      return vector;
    }

    const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    if (magnitude === 0) {
      throw new Error('Cannot normalize zero vector');
    }

    return new Float32Array(vector.map(val => val / magnitude));
  }

  /**
   * Compute pure cosine similarity between normalized vectors
   */
  cosineSimilarity(vectorA: Float32Array, vectorB: Float32Array): number {
    if (!this.config.useCosineSimilarity) {
      throw new Error('Cosine similarity not enabled in config');
    }

    if (vectorA.length !== vectorB.length) {
      throw new Error('Vector dimensions must match');
    }

    // Vectors should already be L2 normalized
    let dotProduct = 0;
    for (let i = 0; i < vectorA.length; i++) {
      dotProduct += vectorA[i] * vectorB[i];
    }

    // Clamp to [-1, 1] to handle floating point precision issues
    return Math.max(-1, Math.min(1, dotProduct));
  }

  /**
   * Apply single affine rescale before isotonic calibration
   * Aligns score ranges across different embedding dimensions (768d/256d)
   */
  affineRescale(score: number): number {
    if (!this.config.affineRescaleParams) {
      return score;
    }

    const { slope, intercept } = this.config.affineRescaleParams;
    return slope * score + intercept;
  }

  /**
   * Batch process vectors for alignment
   */
  alignVectorBatch(vectors: Float32Array[]): Float32Array[] {
    return vectors.map(vector => this.normalizeL2(vector));
  }

  /**
   * Validate vector alignment properties
   */
  validateAlignment(vectors: Float32Array[]): {
    allNormalized: boolean;
    averageL2Norm: number;
    normVariance: number;
  } {
    const norms = vectors.map(vector => {
      const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
      return magnitude;
    });

    const averageNorm = norms.reduce((sum, norm) => sum + norm, 0) / norms.length;
    const variance = norms.reduce((sum, norm) => sum + Math.pow(norm - averageNorm, 2), 0) / norms.length;

    return {
      allNormalized: norms.every(norm => Math.abs(norm - 1.0) < 1e-6),
      averageL2Norm: averageNorm,
      normVariance: variance
    };
  }

  /**
   * Generate configuration fingerprint for tracking
   */
  generateConfigHash(): string {
    const configStr = JSON.stringify({
      enforceL2: this.config.enforceL2Normalization,
      cosine: this.config.useCosineSimilarity,
      affine: this.config.affineRescaleParams
    });
    
    return require('crypto').createHash('sha256').update(configStr).digest('hex').substring(0, 16);
  }
}

/**
 * Score alignment manager for consistent similarity scoring
 */
export class ScoreAlignment {
  private alignment: VectorAlignment;
  private scoreRangeStats: Map<string, { min: number; max: number; mean: number }>;

  constructor(alignment: VectorAlignment) {
    this.alignment = alignment;
    this.scoreRangeStats = new Map();
  }

  /**
   * Analyze score distribution for a given embedding model
   */
  analyzeScoreDistribution(scores: number[], modelId: string): {
    min: number;
    max: number;
    mean: number;
    std: number;
    percentiles: Record<string, number>;
  } {
    const sortedScores = [...scores].sort((a, b) => a - b);
    const n = scores.length;
    
    const min = sortedScores[0];
    const max = sortedScores[n - 1];
    const mean = scores.reduce((sum, score) => sum + score, 0) / n;
    const std = Math.sqrt(scores.reduce((sum, score) => sum + Math.pow(score - mean, 2), 0) / n);
    
    const percentiles = {
      p5: sortedScores[Math.floor(n * 0.05)],
      p25: sortedScores[Math.floor(n * 0.25)],
      p50: sortedScores[Math.floor(n * 0.50)],
      p75: sortedScores[Math.floor(n * 0.75)],
      p95: sortedScores[Math.floor(n * 0.95)]
    };

    // Cache for alignment calculations
    this.scoreRangeStats.set(modelId, { min, max, mean });

    return { min, max, mean, std, percentiles };
  }

  /**
   * Compute affine transform to align score ranges between models
   */
  computeAffineAlignment(sourceModelId: string, targetModelId: string): { slope: number; intercept: number } {
    const sourceStats = this.scoreRangeStats.get(sourceModelId);
    const targetStats = this.scoreRangeStats.get(targetModelId);

    if (!sourceStats || !targetStats) {
      throw new Error(`Missing score statistics for alignment: ${sourceModelId} -> ${targetModelId}`);
    }

    // Linear transformation to map source range to target range
    const sourceRange = sourceStats.max - sourceStats.min;
    const targetRange = targetStats.max - targetStats.min;
    
    const slope = sourceRange === 0 ? 1 : targetRange / sourceRange;
    const intercept = targetStats.min - slope * sourceStats.min;

    return { slope, intercept };
  }

  /**
   * Validate score alignment quality
   */
  validateScoreAlignment(
    originalScores: number[], 
    alignedScores: number[], 
    targetStats: { min: number; max: number; mean: number }
  ): {
    rangeMatch: boolean;
    meanDrift: number;
    alignmentQuality: number;
  } {
    const alignedMin = Math.min(...alignedScores);
    const alignedMax = Math.max(...alignedScores);
    const alignedMean = alignedScores.reduce((sum, score) => sum + score, 0) / alignedScores.length;

    const rangeMatch = Math.abs(alignedMin - targetStats.min) < 0.01 && 
                      Math.abs(alignedMax - targetStats.max) < 0.01;
    
    const meanDrift = Math.abs(alignedMean - targetStats.mean);
    
    // Quality metric: correlation between original and aligned score rankings
    const originalRanks = this.computeRanks(originalScores);
    const alignedRanks = this.computeRanks(alignedScores);
    const alignmentQuality = this.computeSpearmanCorrelation(originalRanks, alignedRanks);

    return { rangeMatch, meanDrift, alignmentQuality };
  }

  private computeRanks(scores: number[]): number[] {
    const indexed = scores.map((score, index) => ({ score, index }));
    indexed.sort((a, b) => b.score - a.score); // Descending order
    
    const ranks = new Array(scores.length);
    indexed.forEach((item, rank) => {
      ranks[item.index] = rank + 1;
    });
    
    return ranks;
  }

  private computeSpearmanCorrelation(ranksA: number[], ranksB: number[]): number {
    const n = ranksA.length;
    const meanA = ranksA.reduce((sum, rank) => sum + rank, 0) / n;
    const meanB = ranksB.reduce((sum, rank) => sum + rank, 0) / n;

    let numerator = 0;
    let denomA = 0;
    let denomB = 0;

    for (let i = 0; i < n; i++) {
      const devA = ranksA[i] - meanA;
      const devB = ranksB[i] - meanB;
      
      numerator += devA * devB;
      denomA += devA * devA;
      denomB += devB * devB;
    }

    return numerator / Math.sqrt(denomA * denomB);
  }
}

/**
 * Comprehensive alignment validation and reporting
 */
export class AlignmentValidator {
  private alignment: VectorAlignment;
  private scoreAlignment: ScoreAlignment;

  constructor(alignment: VectorAlignment, scoreAlignment: ScoreAlignment) {
    this.alignment = alignment;
    this.scoreAlignment = scoreAlignment;
  }

  /**
   * Run comprehensive alignment validation
   */
  async validateComprehensive(
    testVectors: Float32Array[],
    testScores: number[],
    modelId: string
  ): Promise<{
    vectorAlignment: ReturnType<VectorAlignment['validateAlignment']>;
    scoreDistribution: ReturnType<ScoreAlignment['analyzeScoreDistribution']>;
    configHash: string;
    timestamp: string;
  }> {
    // Validate vector alignment
    const alignedVectors = this.alignment.alignVectorBatch(testVectors);
    const vectorAlignment = this.alignment.validateAlignment(alignedVectors);

    // Analyze score distribution
    const scoreDistribution = this.scoreAlignment.analyzeScoreDistribution(testScores, modelId);

    // Generate tracking hash
    const configHash = this.alignment.generateConfigHash();

    return {
      vectorAlignment,
      scoreDistribution,
      configHash,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Save alignment report for audit trail
   */
  async saveAlignmentReport(
    report: Awaited<ReturnType<AlignmentValidator['validateComprehensive']>>,
    outputPath: string
  ): Promise<void> {
    const reportWithMetadata = {
      ...report,
      version: '1.0.0',
      generator: 'Gemma Alignment System',
      specifications: {
        enforceL2Normalization: true,
        useCosineSimilarity: true,
        targetECE: 0.05,
        alignmentThreshold: 0.95
      }
    };

    await fs.promises.writeFile(
      outputPath,
      JSON.stringify(reportWithMetadata, null, 2),
      'utf8'
    );
  }
}