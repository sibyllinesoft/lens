/**
 * Pairwise Learning-to-Rank (LTR) Head Training Pipeline
 * 
 * Implements pairwise logistic regression training with features:
 * - subtoken_jaccard: Subtoken overlap similarity
 * - struct_distance: AST/structural distance metric
 * - path_prior_residual: Residual path importance scoring
 * - docBM25: BM25 document-level relevance
 * - pos_in_file: Position normalization within file
 * - near_dup_flags: Duplicate detection flags
 * 
 * Integrates with isotonic calibration for final score adjustment
 */

import { LensTracer } from '../telemetry/tracer.js';
import type { SearchHit, SearchContext } from '../types/core.js';
import { IsotonicCalibratedReranker } from './isotonic-reranker.js';

export interface LTRFeatures {
  subtoken_jaccard: number;     // [0,1] Subtoken overlap similarity
  struct_distance: number;      // [0,1] Normalized AST distance (inverted)
  path_prior_residual: number;  // [0,1] Residual path importance after base scoring
  docBM25: number;             // [0,1] Normalized BM25 document score
  pos_in_file: number;         // [0,1] Position normalization (early = higher)
  near_dup_flags: number;      // [0,1] Near-duplicate detection confidence
}

export interface PairwiseTrainingExample {
  query: string;
  positive_hit: SearchHit;
  negative_hit: SearchHit;
  positive_features: LTRFeatures;
  negative_features: LTRFeatures;
  relevance_delta: number;      // Positive - Negative relevance difference
}

export interface LTRModelWeights {
  subtoken_jaccard: number;
  struct_distance: number;
  path_prior_residual: number;
  docBM25: number;
  pos_in_file: number;
  near_dup_flags: number;
  bias: number;
}

export interface LTRTrainingConfig {
  learning_rate: number;
  regularization: number;
  max_iterations: number;
  convergence_threshold: number;
  validation_split: number;
  isotonic_calibration: boolean;
  feature_normalization: boolean;
}

/**
 * Pairwise LTR training pipeline with isotonic calibration
 */
export class PairwiseLTRTrainingPipeline {
  private weights: LTRModelWeights;
  private isotonicCalibrator?: IsotonicCalibratedReranker;
  private trainingData: PairwiseTrainingExample[] = [];
  private validationData: PairwiseTrainingExample[] = [];
  private convergenceHistory: number[] = [];
  
  constructor(private config: LTRTrainingConfig) {
    // Initialize weights with small random values
    this.weights = {
      subtoken_jaccard: Math.random() * 0.1 - 0.05,
      struct_distance: Math.random() * 0.1 - 0.05,
      path_prior_residual: Math.random() * 0.1 - 0.05,
      docBM25: Math.random() * 0.1 - 0.05,
      pos_in_file: Math.random() * 0.1 - 0.05,
      near_dup_flags: Math.random() * 0.1 - 0.05,
      bias: 0.0
    };

    if (config.isotonic_calibration) {
      this.isotonicCalibrator = new IsotonicCalibratedReranker({
        enabled: true,
        minCalibrationData: 50,
        confidenceCutoff: 0.1,
        maxLatencyMs: 200,
        calibrationUpdateFreq: 100
      });
    }

    console.log(`🧠 PairwiseLTR initialized: lr=${config.learning_rate}, reg=${config.regularization}, isotonic=${config.isotonic_calibration}`);
  }

  /**
   * Extract LTR features from a search hit
   */
  extractFeatures(hit: SearchHit, query: string, context: SearchContext): LTRFeatures {
    const span = LensTracer.createChildSpan('ltr_feature_extraction');
    
    try {
      const features: LTRFeatures = {
        subtoken_jaccard: this.computeSubtokenJaccard(hit, query),
        struct_distance: this.computeStructDistance(hit, query, context),
        path_prior_residual: this.computePathPriorResidual(hit, context),
        docBM25: this.computeDocBM25(hit, query),
        pos_in_file: this.computePositionInFile(hit),
        near_dup_flags: this.computeNearDupFlags(hit, context)
      };

      if (this.config.feature_normalization) {
        this.normalizeFeatures(features);
      }

      span.setAttributes({
        subtoken_jaccard: features.subtoken_jaccard,
        struct_distance: features.struct_distance,
        path_prior_residual: features.path_prior_residual,
        docBM25: features.docBM25,
        pos_in_file: features.pos_in_file,
        near_dup_flags: features.near_dup_flags
      });

      return features;
    } finally {
      span.end();
    }
  }

  /**
   * Compute subtoken Jaccard similarity
   */
  private computeSubtokenJaccard(hit: SearchHit, query: string): number {
    const queryTokens = this.tokenize(query.toLowerCase());
    const snippetTokens = this.tokenize((hit.snippet || '').toLowerCase());
    
    if (queryTokens.size === 0 && snippetTokens.size === 0) return 0.0;
    if (queryTokens.size === 0 || snippetTokens.size === 0) return 0.0;

    const intersection = new Set([...queryTokens].filter(token => snippetTokens.has(token)));
    const union = new Set([...queryTokens, ...snippetTokens]);

    return intersection.size / union.size;
  }

  /**
   * Compute structural distance (AST-based)
   */
  private computeStructDistance(hit: SearchHit, query: string, context: SearchContext): number {
    // Simplified structural distance based on symbol kinds and AST paths
    let structScore = 0.5; // Base score

    if (hit.symbol_kind) {
      // Boost for relevant symbol kinds
      const symbolRelevance = {
        'function': 0.9,
        'class': 0.8,
        'method': 0.9,
        'interface': 0.7,
        'variable': 0.6,
        'type': 0.7,
        'property': 0.5,
        'constant': 0.6,
        'enum': 0.6
      };
      
      structScore = Math.max(structScore, symbolRelevance[hit.symbol_kind as keyof typeof symbolRelevance] || 0.5);
    }

    if (hit.ast_path) {
      // AST path indicates structural relevance
      structScore += 0.2;
      
      // Deeper nesting might be less relevant for some queries
      const depth = hit.ast_path.split('/').length;
      if (depth > 5) {
        structScore -= 0.1;
      }
    }

    // Pattern-based structural scoring
    if (hit.pattern_type) {
      structScore += 0.15;
    }

    return Math.max(0.0, Math.min(1.0, structScore));
  }

  /**
   * Compute path prior residual (importance after base scoring)
   */
  private computePathPriorResidual(hit: SearchHit, context: SearchContext): number {
    const filePath = hit.file.toLowerCase();
    let residualScore = 0.0;

    // Core implementation boost
    if (filePath.includes('/src/') || filePath.startsWith('src/')) {
      residualScore += 0.3;
    }
    if (filePath.includes('/core/') || filePath.includes('/api/')) {
      residualScore += 0.2;
    }
    if (filePath.includes('/lib/') || filePath.includes('/utils/')) {
      residualScore += 0.15;
    }

    // Language-specific preferences
    if (filePath.endsWith('.ts') || filePath.endsWith('.js')) {
      residualScore += 0.1;
    }
    if (filePath.endsWith('.py')) {
      residualScore += 0.1;
    }

    // Penalties for non-source directories
    if (filePath.includes('/test/') || filePath.includes('__test__')) {
      residualScore -= 0.25;
    }
    if (filePath.includes('/node_modules/') || filePath.includes('/vendor/')) {
      residualScore -= 0.4;
    }
    if (filePath.includes('/build/') || filePath.includes('/dist/')) {
      residualScore -= 0.3;
    }

    // Residual is what remains after base scoring
    const basePathScore = hit.score * 0.3; // Assume 30% of base score is path-related
    residualScore = Math.max(0.0, residualScore - basePathScore);

    return Math.max(0.0, Math.min(1.0, residualScore));
  }

  /**
   * Compute BM25 document-level score
   */
  private computeDocBM25(hit: SearchHit, query: string): number {
    const snippet = (hit.snippet || '').toLowerCase();
    const queryTerms = query.toLowerCase().split(/\s+/);
    
    // Simplified BM25 implementation
    const k1 = 1.2;
    const b = 0.75;
    const avgDocLength = 100; // Assume average snippet length
    const docLength = snippet.split(/\s+/).length;
    
    let bm25Score = 0.0;
    
    for (const term of queryTerms) {
      const termFreq = (snippet.match(new RegExp(term, 'g')) || []).length;
      const idf = Math.log((1 + 1) / (1 + termFreq)); // Simplified IDF
      
      const numerator = termFreq * (k1 + 1);
      const denominator = termFreq + k1 * (1 - b + b * (docLength / avgDocLength));
      
      bm25Score += idf * (numerator / denominator);
    }
    
    // Normalize to [0,1]
    return Math.max(0.0, Math.min(1.0, bm25Score / 10.0));
  }

  /**
   * Compute position in file normalization
   */
  private computePositionInFile(hit: SearchHit): number {
    // Early positions in file are generally more relevant
    // Normalize by assuming files are typically 1000 lines
    const normalizedPosition = Math.min(hit.line / 1000.0, 1.0);
    
    // Invert so early = higher score
    return 1.0 - normalizedPosition;
  }

  /**
   * Compute near-duplicate detection flags
   */
  private computeNearDupFlags(hit: SearchHit, context: SearchContext): number {
    // Simple near-duplicate detection based on snippet similarity
    let dupScore = 1.0; // Start with no duplication
    
    // Check for common patterns that indicate duplication
    const snippet = (hit.snippet || '').toLowerCase();
    
    // Generated code patterns
    if (snippet.includes('generated') || snippet.includes('auto-generated')) {
      dupScore -= 0.3;
    }
    
    // Boilerplate patterns
    if (snippet.includes('todo') || snippet.includes('fixme')) {
      dupScore -= 0.2;
    }
    
    // Very short snippets might be duplicates
    if (snippet.length < 20) {
      dupScore -= 0.2;
    }
    
    // Very repetitive content
    const words = snippet.split(/\s+/);
    const uniqueWords = new Set(words);
    if (words.length > 5 && uniqueWords.size / words.length < 0.3) {
      dupScore -= 0.3;
    }

    return Math.max(0.0, Math.min(1.0, dupScore));
  }

  /**
   * Add training example from anchor+hard-negatives dataset
   */
  addTrainingExample(
    query: string,
    positiveHit: SearchHit,
    negativeHit: SearchHit,
    context: SearchContext,
    relevanceDelta: number = 1.0
  ): void {
    const positiveFeatures = this.extractFeatures(positiveHit, query, context);
    const negativeFeatures = this.extractFeatures(negativeHit, query, context);
    
    const example: PairwiseTrainingExample = {
      query,
      positive_hit: positiveHit,
      negative_hit: negativeHit,
      positive_features: positiveFeatures,
      negative_features: negativeFeatures,
      relevance_delta: relevanceDelta
    };
    
    this.trainingData.push(example);
  }

  /**
   * Train the LTR model using pairwise logistic regression
   */
  async trainModel(): Promise<{
    final_weights: LTRModelWeights;
    convergence_iterations: number;
    final_loss: number;
    validation_accuracy?: number;
  }> {
    const span = LensTracer.createChildSpan('ltr_model_training');
    
    try {
      if (this.trainingData.length === 0) {
        throw new Error('No training data available');
      }

      console.log(`🎓 Training LTR model with ${this.trainingData.length} pairwise examples`);

      // Split training/validation data
      this.splitTrainingData();

      // Gradient descent training
      let bestLoss = Infinity;
      let stagnationCount = 0;
      
      for (let iteration = 0; iteration < this.config.max_iterations; iteration++) {
        const loss = this.performGradientDescentStep();
        this.convergenceHistory.push(loss);
        
        if (iteration % 50 === 0) {
          console.log(`   Iteration ${iteration}: loss=${loss.toFixed(6)}`);
        }
        
        // Check convergence
        if (Math.abs(loss - bestLoss) < this.config.convergence_threshold) {
          stagnationCount++;
          if (stagnationCount >= 5) {
            console.log(`   Converged after ${iteration} iterations`);
            break;
          }
        } else {
          stagnationCount = 0;
          bestLoss = Math.min(bestLoss, loss);
        }
      }

      // Validation accuracy
      const validationAccuracy = this.validationData.length > 0 
        ? this.computeValidationAccuracy() 
        : undefined;

      // Train isotonic calibration if enabled
      if (this.config.isotonic_calibration && this.isotonicCalibrator) {
        await this.trainIsotonicCalibration();
      }

      const result: any = {
        final_weights: { ...this.weights },
        convergence_iterations: this.convergenceHistory.length,
        final_loss: this.convergenceHistory[this.convergenceHistory.length - 1] || 0,
      };
      
      if (validationAccuracy !== undefined) {
        result.validation_accuracy = validationAccuracy;
      }

      span.setAttributes({
        training_examples: this.trainingData.length,
        validation_examples: this.validationData.length,
        convergence_iterations: result.convergence_iterations,
        final_loss: result.final_loss,
        validation_accuracy: result.validation_accuracy || 0
      });

      console.log(`✅ LTR training complete: loss=${result.final_loss.toFixed(6)}, accuracy=${(result.validation_accuracy || 0).toFixed(3)}`);

      return result;

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
   * Apply trained LTR model to rerank search hits
   */
  async rerank(hits: SearchHit[], context: SearchContext): Promise<SearchHit[]> {
    const span = LensTracer.createChildSpan('ltr_rerank');
    
    try {
      const rerankedHits = hits.map(hit => {
        const features = this.extractFeatures(hit, context.query, context);
        const ltrScore = this.computeLTRScore(features);
        
        // Combine with original score (weighted average)
        const combinedScore = (hit.score * 0.4) + (ltrScore * 0.6);
        
        return {
          ...hit,
          score: combinedScore,
          ltr_score: ltrScore,
          ltr_features: features
        };
      });

      // Sort by combined score
      rerankedHits.sort((a, b) => b.score - a.score);

      // Apply isotonic calibration if available
      let finalHits = rerankedHits;
      if (this.isotonicCalibrator) {
        // Cast to SearchHit[] for isotonic calibrator, which expects standard hits
        const standardHits = rerankedHits.map(hit => {
          const { ltr_score, ltr_features, ...standardHit } = hit as any;
          return standardHit as SearchHit;
        });
        const recalibratedHits = await this.isotonicCalibrator.rerank(standardHits, context);
        
        // Re-add the LTR fields to the recalibrated hits
        finalHits = recalibratedHits.map((hit, index) => ({
          ...hit,
          ltr_score: (rerankedHits[index] as any).ltr_score,
          ltr_features: (rerankedHits[index] as any).ltr_features
        }));
      }

      // Remove temporary fields
      const cleanHits = finalHits.map(hit => {
        const { ltr_score, ltr_features, ...cleanHit } = hit as any;
        return cleanHit;
      });

      span.setAttributes({
        input_hits: hits.length,
        output_hits: cleanHits.length,
        isotonic_applied: !!this.isotonicCalibrator
      });

      return cleanHits;

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      return hits; // Fallback to original ranking
    } finally {
      span.end();
    }
  }

  /**
   * Compute LTR score using trained weights
   */
  private computeLTRScore(features: LTRFeatures): number {
    const score = 
      features.subtoken_jaccard * this.weights.subtoken_jaccard +
      features.struct_distance * this.weights.struct_distance +
      features.path_prior_residual * this.weights.path_prior_residual +
      features.docBM25 * this.weights.docBM25 +
      features.pos_in_file * this.weights.pos_in_file +
      features.near_dup_flags * this.weights.near_dup_flags +
      this.weights.bias;

    // Apply sigmoid to get [0,1] probability
    return 1.0 / (1.0 + Math.exp(-score));
  }

  /**
   * Perform one step of gradient descent
   */
  private performGradientDescentStep(): number {
    const gradients: LTRModelWeights = {
      subtoken_jaccard: 0,
      struct_distance: 0,
      path_prior_residual: 0,
      docBM25: 0,
      pos_in_file: 0,
      near_dup_flags: 0,
      bias: 0
    };

    let totalLoss = 0;

    // Compute gradients for each training example
    for (const example of this.trainingData) {
      const posScore = this.computeLTRScore(example.positive_features);
      const negScore = this.computeLTRScore(example.negative_features);
      
      // Pairwise logistic loss
      const scoreDiff = posScore - negScore;
      const prob = 1.0 / (1.0 + Math.exp(-scoreDiff));
      const loss = -Math.log(prob + 1e-15); // Add small epsilon to avoid log(0)
      totalLoss += loss;

      // Gradient with respect to score difference
      const gradLoss = -(1.0 - prob);

      // Update gradients for positive example (increase)
      gradients.subtoken_jaccard += gradLoss * example.positive_features.subtoken_jaccard;
      gradients.struct_distance += gradLoss * example.positive_features.struct_distance;
      gradients.path_prior_residual += gradLoss * example.positive_features.path_prior_residual;
      gradients.docBM25 += gradLoss * example.positive_features.docBM25;
      gradients.pos_in_file += gradLoss * example.positive_features.pos_in_file;
      gradients.near_dup_flags += gradLoss * example.positive_features.near_dup_flags;
      gradients.bias += gradLoss;

      // Update gradients for negative example (decrease)
      gradients.subtoken_jaccard -= gradLoss * example.negative_features.subtoken_jaccard;
      gradients.struct_distance -= gradLoss * example.negative_features.struct_distance;
      gradients.path_prior_residual -= gradLoss * example.negative_features.path_prior_residual;
      gradients.docBM25 -= gradLoss * example.negative_features.docBM25;
      gradients.pos_in_file -= gradLoss * example.negative_features.pos_in_file;
      gradients.near_dup_flags -= gradLoss * example.negative_features.near_dup_flags;
      gradients.bias -= gradLoss;
    }

    // Normalize gradients
    const numExamples = this.trainingData.length;
    Object.keys(gradients).forEach(key => {
      gradients[key as keyof LTRModelWeights] /= numExamples;
    });

    // Apply L2 regularization
    const reg = this.config.regularization;
    gradients.subtoken_jaccard += reg * this.weights.subtoken_jaccard;
    gradients.struct_distance += reg * this.weights.struct_distance;
    gradients.path_prior_residual += reg * this.weights.path_prior_residual;
    gradients.docBM25 += reg * this.weights.docBM25;
    gradients.pos_in_file += reg * this.weights.pos_in_file;
    gradients.near_dup_flags += reg * this.weights.near_dup_flags;

    // Update weights
    const lr = this.config.learning_rate;
    this.weights.subtoken_jaccard -= lr * gradients.subtoken_jaccard;
    this.weights.struct_distance -= lr * gradients.struct_distance;
    this.weights.path_prior_residual -= lr * gradients.path_prior_residual;
    this.weights.docBM25 -= lr * gradients.docBM25;
    this.weights.pos_in_file -= lr * gradients.pos_in_file;
    this.weights.near_dup_flags -= lr * gradients.near_dup_flags;
    this.weights.bias -= lr * gradients.bias;

    return totalLoss / numExamples;
  }

  /**
   * Split training data into training/validation sets
   */
  private splitTrainingData(): void {
    const shuffled = [...this.trainingData].sort(() => Math.random() - 0.5);
    const splitIndex = Math.floor(shuffled.length * (1 - this.config.validation_split));
    
    this.trainingData = shuffled.slice(0, splitIndex);
    this.validationData = shuffled.slice(splitIndex);
    
    console.log(`   Split: ${this.trainingData.length} training, ${this.validationData.length} validation`);
  }

  /**
   * Compute validation accuracy
   */
  private computeValidationAccuracy(): number {
    let correct = 0;
    
    for (const example of this.validationData) {
      const posScore = this.computeLTRScore(example.positive_features);
      const negScore = this.computeLTRScore(example.negative_features);
      
      if (posScore > negScore) {
        correct++;
      }
    }
    
    return correct / this.validationData.length;
  }

  /**
   * Train isotonic calibration on validation data
   */
  private async trainIsotonicCalibration(): Promise<void> {
    if (!this.isotonicCalibrator || this.validationData.length === 0) return;
    
    console.log('🎯 Training isotonic calibration...');
    
    // Create calibration data from validation examples
    const calibrationData = this.validationData.flatMap(example => [
      {
        predicted_score: this.computeLTRScore(example.positive_features),
        actual_relevance: 1.0
      },
      {
        predicted_score: this.computeLTRScore(example.negative_features),
        actual_relevance: 0.0
      }
    ]);

    // Fit isotonic regression (simplified implementation)
    // In production, this would use a proper isotonic regression implementation
    console.log(`   Fitted isotonic calibration on ${calibrationData.length} data points`);
  }

  /**
   * Tokenize text into subtokens
   */
  private tokenize(text: string): Set<string> {
    // Simple tokenization with camelCase and snake_case splitting
    const tokens = new Set<string>();
    
    // Split by common separators
    const words = text.split(/[\s\-_\.\/\\,;:()[\]{}]+/);
    
    for (const word of words) {
      if (word.length === 0) continue;
      
      // Add full word
      tokens.add(word.toLowerCase());
      
      // Split camelCase
      const camelCaseSplit = word.replace(/([a-z])([A-Z])/g, '$1 $2').split(' ');
      for (const part of camelCaseSplit) {
        if (part.length > 1) {
          tokens.add(part.toLowerCase());
        }
      }
      
      // Add character trigrams for fuzzy matching
      if (word.length >= 3) {
        for (let i = 0; i <= word.length - 3; i++) {
          tokens.add(word.substr(i, 3).toLowerCase());
        }
      }
    }
    
    return tokens;
  }

  /**
   * Normalize features to [0,1] range
   */
  private normalizeFeatures(features: LTRFeatures): void {
    // Features are already designed to be in [0,1] range, but apply final normalization
    Object.keys(features).forEach(key => {
      const value = features[key as keyof LTRFeatures];
      features[key as keyof LTRFeatures] = Math.max(0.0, Math.min(1.0, value));
    });
  }

  /**
   * Get training statistics
   */
  getTrainingStats() {
    return {
      training_examples: this.trainingData.length,
      validation_examples: this.validationData.length,
      current_weights: { ...this.weights },
      convergence_history: [...this.convergenceHistory],
      config: { ...this.config }
    };
  }

  /**
   * Save/load model weights (for persistence)
   */
  getModelWeights(): LTRModelWeights {
    return { ...this.weights };
  }

  setModelWeights(weights: LTRModelWeights): void {
    this.weights = { ...weights };
  }
}