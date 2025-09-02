/**
 * Phase 2 Path Prior System - Gentler de-boosts for low-priority paths
 * Implements logistic regression with feature engineering for path scoring
 */

import { LensTracer } from '../telemetry/tracer.js';
import { promises as fs } from 'fs';
import path from 'path';

interface PathFeatures {
  is_test_dir: number;
  is_vendor: number;
  depth: number;
  recently_touched: number;
  file_ext_score: number;
  path_unigram_lm: number;
}

interface PathTrainingExample {
  path: string;
  features: PathFeatures;
  relevance_score: number; // Ground truth relevance
}

interface PathPriorModel {
  version: string;
  generated_at: string;
  parameters: {
    l2_regularization: number;
    debias_low_priority_paths: boolean;
    max_deboost: number;
  };
  coefficients: {
    is_test_dir: number;
    is_vendor: number;
    depth: number;
    recently_touched: number;
    file_ext_score: number;
    path_unigram_lm: number;
    intercept: number;
  };
  performance: {
    training_accuracy: number;
    cross_validation_score: number;
    auc_roc: number;
  };
}

export class Phase2PathPrior {
  private fileExtensionScores = new Map<string, number>([
    ['.ts', 1.0], ['.js', 0.95], ['.py', 0.9], ['.go', 0.85], ['.rs', 0.85],
    ['.java', 0.8], ['.cpp', 0.8], ['.c', 0.75], ['.h', 0.75], ['.hpp', 0.75],
    ['.json', 0.4], ['.yaml', 0.4], ['.yml', 0.4], ['.xml', 0.3],
    ['.md', 0.2], ['.txt', 0.1], ['.log', 0.05]
  ]);

  private unigramLanguageModel = new Map<string, number>();
  private currentModel: PathPriorModel | null = null;

  constructor(
    private readonly indexRoot: string,
    private readonly outputDir: string = './path-priors'
  ) {}

  /**
   * Refit path prior model with gentler de-boosts
   */
  async refitPathPrior(params: {
    l2_regularization?: number;
    debias_low_priority_paths?: boolean;
    max_deboost?: number;
  } = {}): Promise<PathPriorModel> {
    const span = LensTracer.createChildSpan('refit_path_prior');
    const { 
      l2_regularization = 1.0, 
      debias_low_priority_paths = true, 
      max_deboost = 0.6 
    } = params;

    try {
      console.log('🔍 Starting path prior refitting with gentler de-boosts...');
      
      // Step 1: Build unigram language model from paths
      await this.buildUnigramLanguageModel();
      
      // Step 2: Extract training examples
      const trainingData = await this.extractTrainingExamples();
      
      // Step 3: Train logistic regression model
      const model = await this.trainLogisticRegression(
        trainingData, 
        { l2_regularization, debias_low_priority_paths, max_deboost }
      );
      
      // Step 4: Evaluate model performance
      const performance = await this.evaluateModel(model, trainingData);
      model.performance = performance;
      
      // Step 5: Save model
      await this.savePathPriorModel(model);
      
      this.currentModel = model;

      console.log(`✅ Path prior model refitted with ${trainingData.length} examples`);
      console.log(`📊 Model performance: AUC-ROC ${performance.auc_roc.toFixed(3)}, Accuracy ${performance.training_accuracy.toFixed(3)}`);
      
      span.setAttributes({
        success: true,
        training_examples: trainingData.length,
        auc_roc: performance.auc_roc,
        training_accuracy: performance.training_accuracy,
        l2_regularization,
        max_deboost,
      });

      return model;

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw new Error(`Path prior refitting failed: ${errorMsg}`);
    } finally {
      span.end();
    }
  }

  /**
   * Build unigram language model from indexed paths
   */
  private async buildUnigramLanguageModel(): Promise<void> {
    const span = LensTracer.createChildSpan('build_unigram_lm');

    try {
      const indexedDir = path.join(this.indexRoot, 'indexed-content');
      const files = await fs.readdir(indexedDir);
      
      const tokenCounts = new Map<string, number>();
      let totalTokens = 0;
      
      for (const file of files) {
        const filePath = file.replace(/[_]/g, '/'); // Convert back from flattened format
        const pathTokens = this.tokenizePath(filePath);
        
        for (const token of pathTokens) {
          tokenCounts.set(token, (tokenCounts.get(token) || 0) + 1);
          totalTokens++;
        }
      }
      
      // Convert counts to probabilities with Laplace smoothing
      for (const [token, count] of tokenCounts) {
        this.unigramLanguageModel.set(token, (count + 1) / (totalTokens + tokenCounts.size));
      }

      console.log(`📊 Built unigram language model with ${this.unigramLanguageModel.size} tokens`);

      span.setAttributes({
        success: true,
        unique_tokens: this.unigramLanguageModel.size,
        total_tokens: totalTokens,
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
   * Tokenize a file path into meaningful components
   */
  private tokenizePath(filePath: string): string[] {
    return filePath
      .toLowerCase()
      .split(/[/\\._-]/)
      .filter(token => token.length > 0 && /[a-zA-Z]/.test(token));
  }

  /**
   * Calculate path unigram language model score
   */
  private calculatePathUnigramScore(filePath: string): number {
    const tokens = this.tokenizePath(filePath);
    let logProb = 0;
    
    for (const token of tokens) {
      const prob = this.unigramLanguageModel.get(token) || 1e-6; // Minimum probability
      logProb += Math.log(prob);
    }
    
    // Normalize by path length to avoid bias against long paths
    return logProb / Math.max(tokens.length, 1);
  }

  /**
   * Extract features for a given file path
   */
  private extractPathFeatures(filePath: string): PathFeatures {
    const normalizedPath = filePath.toLowerCase();
    const pathParts = filePath.split('/');
    const fileName = path.basename(filePath);
    const fileExt = path.extname(fileName);
    
    return {
      is_test_dir: this.isTestDirectory(normalizedPath) ? 1 : 0,
      is_vendor: this.isVendorDirectory(normalizedPath) ? 1 : 0,
      depth: pathParts.length - 1, // Exclude filename from depth
      recently_touched: this.estimateRecentActivity(filePath), // Heuristic based on path patterns
      file_ext_score: this.fileExtensionScores.get(fileExt) || 0.1,
      path_unigram_lm: this.calculatePathUnigramScore(filePath),
    };
  }

  /**
   * Check if path is in a test directory
   */
  private isTestDirectory(path: string): boolean {
    const testPatterns = [
      '/test/', '/tests/', '/__tests__/', '/spec/', '/specs/',
      '.test.', '.spec.', '_test.', '_spec.'
    ];
    return testPatterns.some(pattern => path.includes(pattern));
  }

  /**
   * Check if path is in a vendor/external directory
   */
  private isVendorDirectory(path: string): boolean {
    const vendorPatterns = [
      '/vendor/', '/third_party/', '/node_modules/', '/external/',
      '/lib/', '/libs/', '/dependencies/', '/.git/', '/dist/', '/build/'
    ];
    return vendorPatterns.some(pattern => path.includes(pattern));
  }

  /**
   * Estimate recent activity based on path patterns (heuristic)
   */
  private estimateRecentActivity(filePath: string): number {
    // Heuristics: files in src/, newer extensions, shorter paths tend to be more active
    const srcScore = filePath.includes('/src/') ? 0.8 : 0.4;
    const modernExtScore = ['.ts', '.tsx', '.jsx', '.vue', '.svelte'].includes(path.extname(filePath)) ? 0.9 : 0.5;
    const lengthScore = Math.max(0.1, 1.0 - (filePath.length / 100)); // Penalize very long paths
    
    return (srcScore + modernExtScore + lengthScore) / 3;
  }

  /**
   * Extract training examples from indexed content
   */
  private async extractTrainingExamples(): Promise<PathTrainingExample[]> {
    const span = LensTracer.createChildSpan('extract_training_examples');

    try {
      const indexedDir = path.join(this.indexRoot, 'indexed-content');
      const files = await fs.readdir(indexedDir);
      
      const examples: PathTrainingExample[] = [];
      
      for (const file of files) {
        const filePath = file.replace(/[_]/g, '/'); // Convert from flattened format
        const features = this.extractPathFeatures(filePath);
        
        // Heuristic relevance scoring based on file characteristics
        // In a real system, this would come from user interaction data
        const relevanceScore = this.calculateHeuristicRelevance(filePath, features);
        
        examples.push({
          path: filePath,
          features,
          relevance_score: relevanceScore,
        });
      }

      console.log(`📊 Extracted ${examples.length} training examples`);

      span.setAttributes({
        success: true,
        training_examples: examples.length,
      });

      return examples;

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
   * Calculate heuristic relevance score for training
   */
  private calculateHeuristicRelevance(filePath: string, features: PathFeatures): number {
    let score = 0.5; // Base score
    
    // Boost main source files
    if (features.file_ext_score > 0.8) score += 0.3;
    
    // Penalize test and vendor files
    if (features.is_test_dir) score -= 0.2;
    if (features.is_vendor) score -= 0.3;
    
    // Penalize deeply nested files
    if (features.depth > 5) score -= 0.1 * (features.depth - 5);
    
    // Boost recently touched files
    score += features.recently_touched * 0.2;
    
    // Boost files with good language model score
    score += Math.max(0, features.path_unigram_lm + 5) * 0.1; // Adjust for log scale
    
    return Math.max(0.0, Math.min(1.0, score));
  }

  /**
   * Train logistic regression model
   */
  private async trainLogisticRegression(
    trainingData: PathTrainingExample[],
    params: {
      l2_regularization: number;
      debias_low_priority_paths: boolean;
      max_deboost: number;
    }
  ): Promise<PathPriorModel> {
    const span = LensTracer.createChildSpan('train_logistic_regression');

    try {
      // Simple gradient descent implementation
      // In production, this would use a proper ML library
      
      const features = trainingData.map(example => [
        example.features.is_test_dir,
        example.features.is_vendor,
        example.features.depth,
        example.features.recently_touched,
        example.features.file_ext_score,
        example.features.path_unigram_lm,
        1.0 // intercept
      ]);
      
      const targets = trainingData.map(example => example.relevance_score > 0.5 ? 1 : 0);
      
      // Initialize weights
      let weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // 6 features + intercept
      
      const learningRate = 0.01;
      const epochs = 1000;
      
      // Training loop
      for (let epoch = 0; epoch < epochs; epoch++) {
        const gradients = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let totalLoss = 0;
        
        for (let i = 0; i < features.length; i++) {
          const x = features[i];
          const y = targets[i];
          
          // Forward pass
          const logits = x.reduce((sum, xi, j) => sum + xi * weights[j], 0);
          const prediction = 1 / (1 + Math.exp(-logits));
          
          // Loss
          const loss = -y * Math.log(Math.max(prediction, 1e-15)) - (1 - y) * Math.log(Math.max(1 - prediction, 1e-15));
          totalLoss += loss;
          
          // Backward pass
          const error = prediction - y;
          for (let j = 0; j < weights.length; j++) {
            gradients[j] += error * x[j];
          }
        }
        
        // Update weights with L2 regularization
        for (let j = 0; j < weights.length; j++) {
          const regularization = j < weights.length - 1 ? params.l2_regularization * weights[j] : 0; // No regularization on intercept
          weights[j] -= learningRate * (gradients[j] / features.length + regularization);
        }
        
        if (epoch % 100 === 0) {
          console.log(`Epoch ${epoch}, Loss: ${(totalLoss / features.length).toFixed(4)}`);
        }
      }

      // Apply gentler de-boost constraints
      if (params.debias_low_priority_paths) {
        // Clamp negative weights to prevent excessive penalties
        weights[0] = Math.max(weights[0], -Math.log(1 / params.max_deboost - 1)); // is_test_dir
        weights[1] = Math.max(weights[1], -Math.log(1 / params.max_deboost - 1)); // is_vendor
      }

      const model: PathPriorModel = {
        version: 'path_prior_v2_gentler',
        generated_at: new Date().toISOString(),
        parameters: params,
        coefficients: {
          is_test_dir: weights[0],
          is_vendor: weights[1],
          depth: weights[2],
          recently_touched: weights[3],
          file_ext_score: weights[4],
          path_unigram_lm: weights[5],
          intercept: weights[6],
        },
        performance: {
          training_accuracy: 0, // Will be filled by evaluateModel
          cross_validation_score: 0,
          auc_roc: 0,
        }
      };

      span.setAttributes({
        success: true,
        training_examples: trainingData.length,
        epochs,
      });

      return model;

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
   * Evaluate model performance
   */
  private async evaluateModel(
    model: PathPriorModel,
    trainingData: PathTrainingExample[]
  ): Promise<{ training_accuracy: number; cross_validation_score: number; auc_roc: number }> {
    let correctPredictions = 0;
    const predictions: number[] = [];
    const targets: number[] = [];
    
    for (const example of trainingData) {
      const score = this.scorePathWithModel(example.path, model);
      const prediction = score > 0.5 ? 1 : 0;
      const target = example.relevance_score > 0.5 ? 1 : 0;
      
      predictions.push(score);
      targets.push(target);
      
      if (prediction === target) {
        correctPredictions++;
      }
    }
    
    const accuracy = correctPredictions / trainingData.length;
    
    // Calculate AUC-ROC (simplified implementation)
    const aucRoc = this.calculateAUC(targets, predictions);
    
    return {
      training_accuracy: accuracy,
      cross_validation_score: accuracy, // Simplified - should use proper cross-validation
      auc_roc: aucRoc,
    };
  }

  /**
   * Calculate AUC-ROC (Area Under Curve - Receiver Operating Characteristic)
   */
  private calculateAUC(targets: number[], predictions: number[]): number {
    // Sort by prediction scores
    const sorted = targets
      .map((target, i) => ({ target, prediction: predictions[i] }))
      .sort((a, b) => b.prediction - a.prediction);
    
    let auc = 0;
    let positives = 0;
    let negatives = 0;
    
    // Count positives and negatives
    for (const item of sorted) {
      if (item.target === 1) positives++;
      else negatives++;
    }
    
    if (positives === 0 || negatives === 0) return 0.5; // Random performance
    
    let truePositives = 0;
    let falsePositives = 0;
    
    for (const item of sorted) {
      if (item.target === 1) {
        truePositives++;
      } else {
        falsePositives++;
        // Add area under curve
        auc += truePositives;
      }
    }
    
    return auc / (positives * negatives);
  }

  /**
   * Score a path using the trained model
   */
  scorePathWithModel(filePath: string, model?: PathPriorModel): number {
    const currentModel = model || this.currentModel;
    if (!currentModel) {
      return 0.5; // Default neutral score
    }
    
    const features = this.extractPathFeatures(filePath);
    const logits = 
      features.is_test_dir * currentModel.coefficients.is_test_dir +
      features.is_vendor * currentModel.coefficients.is_vendor +
      features.depth * currentModel.coefficients.depth +
      features.recently_touched * currentModel.coefficients.recently_touched +
      features.file_ext_score * currentModel.coefficients.file_ext_score +
      features.path_unigram_lm * currentModel.coefficients.path_unigram_lm +
      currentModel.coefficients.intercept;
    
    return 1 / (1 + Math.exp(-logits));
  }

  /**
   * Save path prior model to disk
   */
  private async savePathPriorModel(model: PathPriorModel): Promise<void> {
    const span = LensTracer.createChildSpan('save_path_prior_model');

    try {
      await fs.mkdir(this.outputDir, { recursive: true });
      
      const modelPath = path.join(this.outputDir, 'path_prior_model_v2.json');
      await fs.writeFile(modelPath, JSON.stringify(model, null, 2));

      console.log(`💾 Saved path prior model to ${modelPath}`);

      span.setAttributes({
        success: true,
        model_path: modelPath,
        version: model.version,
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
   * Load existing path prior model
   */
  async loadPathPriorModel(version: string = 'path_prior_v2_gentler'): Promise<PathPriorModel | null> {
    const span = LensTracer.createChildSpan('load_path_prior_model');

    try {
      const modelPath = path.join(this.outputDir, 'path_prior_model_v2.json');
      const content = await fs.readFile(modelPath, 'utf-8');
      const model: PathPriorModel = JSON.parse(content);
      
      if (model.version === version) {
        this.currentModel = model;
        span.setAttributes({
          success: true,
          loaded_version: model.version,
          auc_roc: model.performance.auc_roc,
        });
        return model;
      }
      
      return null;

    } catch (error) {
      span.setAttributes({ success: false, error: 'File not found or invalid' });
      return null;
    } finally {
      span.end();
    }
  }
}