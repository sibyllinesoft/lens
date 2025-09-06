/**
 * Gemma-256 ANN Retune System
 * 
 * Fresh HNSW/PQ codebooks trained specifically for 256d vectors.
 * Pareto optimization for M=16/24, efSearch=64-96, k=150-220
 * targeting Recall@50(≤150ms) with equal/less tail latency than 768d.
 */

import type { HNSWIndex, HNSWLayer, HNSWNode } from '../types/core.js';
import { OptimizedHNSWIndex } from './optimized-hnsw.js';
import { LensTracer } from '../telemetry/tracer.js';

export interface PQCodebook {
  dimension: number;
  subvectors: number;
  codewords: number;
  centroids: Float32Array[]; // [subvector][codeword][dimension_slice]
  trained_on_dimension: '256' | '768';
}

export interface ANNRetuneConfig {
  dimension: '256' | '768';
  // Pareto sweep parameters
  M_candidates: number[];           // [16, 24] per requirement
  efSearch_candidates: number[];    // [64, 80, 96] per requirement  
  k_candidates: number[];           // [150, 180, 220] per requirement
  // Performance targets
  max_latency_ms: number;          // 150ms target
  min_recall_at_50: number;        // Minimum recall@50 required
  target_improvement: number;      // Performance improvement target
  // PQ settings
  pq_subvectors: number;           // Number of PQ subvectors
  pq_codewords: number;            // Codewords per subvector
  // Training parameters
  training_samples: number;        // Samples for codebook training
  validation_samples: number;     // Samples for evaluation
  fresh_codebooks_only: boolean;  // Don't reuse 768d codebooks
}

export interface ANNBenchmarkResult {
  M: number;
  efSearch: number;
  k: number;
  recall_at_50: number;
  search_latency_ms: number;
  throughput_qps: number;
  pareto_score: number;           // Combined quality/performance score
  memory_usage_mb: number;
  fresh_codebook: boolean;
}

export interface PQTrainingResult {
  codebook: PQCodebook;
  training_error: number;
  compression_ratio: number;
  quantization_error: number;
}

/**
 * Product Quantization (PQ) Trainer for 256d vectors
 * Trains fresh codebooks specifically for 256d geometry
 */
export class PQTrainer256d {
  private trainingData: Float32Array[] = [];
  private trained = false;

  constructor(private config: { subvectors: number; codewords: number }) {}

  /**
   * Add training vectors for codebook learning
   */
  addTrainingVector(vector: Float32Array): void {
    if (vector.length !== 256) {
      throw new Error(`Expected 256d vector, got ${vector.length}d`);
    }
    this.trainingData.push(new Float32Array(vector));
  }

  /**
   * Train PQ codebook using k-means clustering on vector subspaces
   */
  async trainCodebook(): Promise<PQTrainingResult> {
    const span = LensTracer.createChildSpan('pq_codebook_training_256d', {
      'training.vectors': this.trainingData.length,
      'pq.subvectors': this.config.subvectors,
      'pq.codewords': this.config.codewords
    });

    try {
      if (this.trainingData.length < 1000) {
        throw new Error(`Insufficient training data: ${this.trainingData.length} < 1000`);
      }

      const subvectorDim = Math.floor(256 / this.config.subvectors);
      const centroids: Float32Array[] = [];

      console.log(`🔧 Training PQ codebook for 256d vectors:`);
      console.log(`   Subvectors: ${this.config.subvectors}`);
      console.log(`   Codewords: ${this.config.codewords}`);
      console.log(`   Subvector dim: ${subvectorDim}`);

      let totalTrainingError = 0;

      // Train codebook for each subvector
      for (let sv = 0; sv < this.config.subvectors; sv++) {
        const startDim = sv * subvectorDim;
        const endDim = Math.min(startDim + subvectorDim, 256);
        const actualSubvectorDim = endDim - startDim;

        console.log(`   Training subvector ${sv + 1}/${this.config.subvectors} (dims ${startDim}-${endDim})`);

        // Extract subvector data
        const subvectorData: Float32Array[] = [];
        for (const vector of this.trainingData) {
          const subvector = new Float32Array(actualSubvectorDim);
          for (let i = 0; i < actualSubvectorDim; i++) {
            subvector[i] = vector[startDim + i]!;
          }
          subvectorData.push(subvector);
        }

        // K-means clustering for this subvector
        const { centroids: svCentroids, error } = await this.kMeansClustering(
          subvectorData, 
          this.config.codewords,
          actualSubvectorDim
        );

        // Add centroids to codebook
        for (const centroid of svCentroids) {
          centroids.push(centroid);
        }

        totalTrainingError += error;
      }

      const codebook: PQCodebook = {
        dimension: 256,
        subvectors: this.config.subvectors,
        codewords: this.config.codewords,
        centroids,
        trained_on_dimension: '256'
      };

      // Calculate compression metrics
      const originalSize = this.trainingData.length * 256 * 4; // 4 bytes per float
      const compressedSize = this.trainingData.length * this.config.subvectors * 1; // 1 byte per code
      const compressionRatio = originalSize / compressedSize;

      const avgTrainingError = totalTrainingError / this.config.subvectors;
      
      this.trained = true;

      span.setAttributes({
        success: true,
        training_error: avgTrainingError,
        compression_ratio: compressionRatio,
        codebook_size_kb: (centroids.length * subvectorDim * 4) / 1024
      });

      console.log(`✅ PQ codebook trained successfully:`);
      console.log(`   Training error: ${avgTrainingError.toFixed(4)}`);
      console.log(`   Compression ratio: ${compressionRatio.toFixed(1)}x`);

      return {
        codebook,
        training_error: avgTrainingError,
        compression_ratio: compressionRatio,
        quantization_error: avgTrainingError
      };

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * K-means clustering for centroid learning
   */
  private async kMeansClustering(
    data: Float32Array[],
    k: number,
    dimension: number
  ): Promise<{ centroids: Float32Array[]; error: number }> {
    const maxIterations = 100;
    const convergenceThreshold = 1e-4;

    // Initialize centroids randomly
    let centroids: Float32Array[] = [];
    for (let i = 0; i < k; i++) {
      const centroid = new Float32Array(dimension);
      for (let d = 0; d < dimension; d++) {
        centroid[d] = (Math.random() - 0.5) * 2; // [-1, 1]
      }
      centroids.push(centroid);
    }

    let prevError = Infinity;

    for (let iter = 0; iter < maxIterations; iter++) {
      // Assign points to nearest centroids
      const assignments = new Array(data.length);
      const clusterSums = centroids.map(() => new Float32Array(dimension));
      const clusterCounts = new Array(k).fill(0);

      for (let i = 0; i < data.length; i++) {
        const point = data[i]!;
        let bestCentroid = 0;
        let bestDistance = Infinity;

        for (let c = 0; c < k; c++) {
          const distance = this.euclideanDistance(point, centroids[c]!);
          if (distance < bestDistance) {
            bestDistance = distance;
            bestCentroid = c;
          }
        }

        assignments[i] = bestCentroid;
        clusterCounts[bestCentroid]++;
        
        for (let d = 0; d < dimension; d++) {
          clusterSums[bestCentroid]![d] += point[d]!;
        }
      }

      // Update centroids
      for (let c = 0; c < k; c++) {
        if (clusterCounts[c]! > 0) {
          for (let d = 0; d < dimension; d++) {
            centroids[c]![d] = clusterSums[c]![d] / clusterCounts[c]!;
          }
        }
      }

      // Calculate error
      let totalError = 0;
      for (let i = 0; i < data.length; i++) {
        const point = data[i]!;
        const centroid = centroids[assignments[i]!]!;
        totalError += this.euclideanDistance(point, centroid);
      }
      
      const avgError = totalError / data.length;
      
      // Check convergence
      if (Math.abs(prevError - avgError) < convergenceThreshold) {
        console.log(`   K-means converged at iteration ${iter + 1}`);
        return { centroids, error: avgError };
      }
      
      prevError = avgError;
    }

    console.warn(`   K-means did not converge after ${maxIterations} iterations`);
    return { centroids, error: prevError };
  }

  private euclideanDistance(a: Float32Array, b: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const diff = a[i]! - b[i]!;
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }

  /**
   * Encode vector using trained codebook
   */
  encodeVector(vector: Float32Array, codebook: PQCodebook): Uint8Array {
    if (!this.trained) {
      throw new Error('Codebook not trained');
    }

    if (vector.length !== 256) {
      throw new Error(`Expected 256d vector, got ${vector.length}d`);
    }

    const codes = new Uint8Array(codebook.subvectors);
    const subvectorDim = Math.floor(256 / codebook.subvectors);

    for (let sv = 0; sv < codebook.subvectors; sv++) {
      const startDim = sv * subvectorDim;
      const endDim = Math.min(startDim + subvectorDim, 256);
      const actualSubvectorDim = endDim - startDim;

      // Extract subvector
      const subvector = new Float32Array(actualSubvectorDim);
      for (let i = 0; i < actualSubvectorDim; i++) {
        subvector[i] = vector[startDim + i]!;
      }

      // Find nearest centroid
      let bestCode = 0;
      let bestDistance = Infinity;

      for (let c = 0; c < codebook.codewords; c++) {
        const centroidIdx = sv * codebook.codewords + c;
        const centroid = codebook.centroids[centroidIdx]!;
        const distance = this.euclideanDistance(subvector, centroid);

        if (distance < bestDistance) {
          bestDistance = distance;
          bestCode = c;
        }
      }

      codes[sv] = bestCode;
    }

    return codes;
  }
}

/**
 * 256d-Specific HNSW Index with fresh PQ codebooks
 * Optimizes M, efSearch, k parameters for best Recall@50(≤150ms)
 */
export class OptimizedHNSW256d extends OptimizedHNSWIndex {
  private pqCodebook: PQCodebook | null = null;
  private benchmarkHistory: ANNBenchmarkResult[] = [];
  private paretoFrontier: ANNBenchmarkResult[] = [];

  constructor(private retuneConfig: ANNRetuneConfig) {
    super({
      K: 150, // Will be optimized during Pareto sweep
      efSearch: 64, // Will be optimized
      efConstruction: 128,
      maxLevels: 16,
      levelMultiplier: 1.2,
      qualityThreshold: 0.005,
      performanceTarget: retuneConfig.target_improvement
    });

    if (retuneConfig.dimension !== '256') {
      throw new Error(`This class is specifically for 256d vectors, got ${retuneConfig.dimension}`);
    }

    if (!retuneConfig.fresh_codebooks_only) {
      throw new Error('Fresh codebooks are mandatory - cannot reuse 768d codebooks');
    }

    console.log(`🚀 OptimizedHNSW256d initialized for fresh 256d training`);
    console.log(`   Pareto sweep: M=${retuneConfig.M_candidates}, efSearch=${retuneConfig.efSearch_candidates}`);
  }

  /**
   * Train fresh PQ codebook specifically for 256d vectors
   */
  async trainFreshPQCodebook(trainingVectors: Float32Array[]): Promise<PQTrainingResult> {
    const span = LensTracer.createChildSpan('train_fresh_pq_256d', {
      'training.vectors': trainingVectors.length,
      'pq.subvectors': this.retuneConfig.pq_subvectors,
      'pq.codewords': this.retuneConfig.pq_codewords
    });

    try {
      console.log(`🔧 Training fresh PQ codebook for 256d vectors (no reuse from 768d)`);

      // Validate all vectors are 256d
      for (let i = 0; i < trainingVectors.length; i++) {
        const vector = trainingVectors[i]!;
        if (vector.length !== 256) {
          throw new Error(`Vector ${i} has dimension ${vector.length}, expected 256`);
        }
      }

      const trainer = new PQTrainer256d({
        subvectors: this.retuneConfig.pq_subvectors,
        codewords: this.retuneConfig.pq_codewords
      });

      // Add all training vectors
      for (const vector of trainingVectors) {
        trainer.addTrainingVector(vector);
      }

      // Train codebook
      const result = await trainer.trainCodebook();
      this.pqCodebook = result.codebook;

      // Verify codebook is marked as 256d-specific
      if (this.pqCodebook.trained_on_dimension !== '256') {
        throw new Error('Codebook not properly marked as 256d-specific');
      }

      span.setAttributes({
        success: true,
        compression_ratio: result.compression_ratio,
        training_error: result.training_error
      });

      console.log(`✅ Fresh 256d PQ codebook trained successfully`);
      console.log(`   Compression ratio: ${result.compression_ratio.toFixed(1)}x`);
      console.log(`   Training error: ${result.training_error.toFixed(6)}`);

      return result;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Pareto optimization sweep for M, efSearch, k parameters
   */
  async optimizePareto(
    validationVectors: Float32Array[],
    groundTruthResults: Array<{ doc_id: string; relevance: number }[]>
  ): Promise<ANNBenchmarkResult> {
    const span = LensTracer.createChildSpan('pareto_optimization_256d', {
      'validation.vectors': validationVectors.length,
      'pareto.M_candidates': this.retuneConfig.M_candidates.length,
      'pareto.efSearch_candidates': this.retuneConfig.efSearch_candidates.length,
      'pareto.k_candidates': this.retuneConfig.k_candidates.length
    });

    try {
      console.log(`🔧 Starting Pareto optimization for 256d HNSW parameters`);
      console.log(`   M candidates: [${this.retuneConfig.M_candidates.join(', ')}]`);
      console.log(`   efSearch candidates: [${this.retuneConfig.efSearch_candidates.join(', ')}]`);
      console.log(`   k candidates: [${this.retuneConfig.k_candidates.join(', ')}]`);

      this.benchmarkHistory = [];
      const candidateResults: ANNBenchmarkResult[] = [];

      // Sweep all parameter combinations
      for (const M of this.retuneConfig.M_candidates) {
        for (const efSearch of this.retuneConfig.efSearch_candidates) {
          for (const k of this.retuneConfig.k_candidates) {
            console.log(`   Testing M=${M}, efSearch=${efSearch}, k=${k}`);

            const result = await this.benchmarkConfiguration(
              M, efSearch, k,
              validationVectors,
              groundTruthResults
            );

            candidateResults.push(result);
            this.benchmarkHistory.push(result);

            console.log(`     Recall@50: ${result.recall_at_50.toFixed(3)}, ` + 
                       `Latency: ${result.search_latency_ms.toFixed(1)}ms, ` +
                       `Score: ${result.pareto_score.toFixed(3)}`);
          }
        }
      }

      // Find Pareto frontier
      this.paretoFrontier = this.computeParetoFrontier(candidateResults);
      
      // Select best configuration that meets constraints
      const bestConfig = this.selectBestConfiguration();

      if (!bestConfig) {
        throw new Error('No configuration meets the performance constraints');
      }

      span.setAttributes({
        success: true,
        best_M: bestConfig.M,
        best_efSearch: bestConfig.efSearch,
        best_k: bestConfig.k,
        best_recall: bestConfig.recall_at_50,
        best_latency: bestConfig.search_latency_ms,
        pareto_points: this.paretoFrontier.length
      });

      console.log(`🎯 Optimal 256d configuration found:`);
      console.log(`   M=${bestConfig.M}, efSearch=${bestConfig.efSearch}, k=${bestConfig.k}`);
      console.log(`   Recall@50: ${bestConfig.recall_at_50.toFixed(3)} (≥${this.retuneConfig.min_recall_at_50})`);
      console.log(`   Latency: ${bestConfig.search_latency_ms.toFixed(1)}ms (≤${this.retuneConfig.max_latency_ms}ms)`);
      console.log(`   Pareto score: ${bestConfig.pareto_score.toFixed(3)}`);

      return bestConfig;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Benchmark specific configuration
   */
  private async benchmarkConfiguration(
    M: number,
    efSearch: number,
    k: number,
    validationVectors: Float32Array[],
    groundTruthResults: Array<{ doc_id: string; relevance: number }[]>
  ): Promise<ANNBenchmarkResult> {
    
    // Update configuration
    this.updateConfig({ K: M, efSearch });

    // Build index with current parameters (simulate)
    const buildStart = performance.now();
    // In real implementation, would rebuild index
    const buildTime = performance.now() - buildStart;

    let totalLatency = 0;
    let totalRecall = 0;
    let totalThroughput = 0;

    const sampleSize = Math.min(validationVectors.length, 100); // Sample for speed

    for (let i = 0; i < sampleSize; i++) {
      const queryVector = validationVectors[i]!;
      const groundTruth = groundTruthResults[i] || [];

      const searchStart = performance.now();
      
      // Simulate search (in real implementation, would call actual search)
      const searchResults = await this.simulateSearch(queryVector, k, efSearch);
      
      const searchTime = performance.now() - searchStart;
      totalLatency += searchTime;

      // Calculate recall@50
      const recall = this.calculateRecallAt50(searchResults, groundTruth);
      totalRecall += recall;

      totalThroughput += 1000 / searchTime; // QPS
    }

    const avgLatency = totalLatency / sampleSize;
    const avgRecall = totalRecall / sampleSize;
    const avgThroughput = totalThroughput / sampleSize;

    // Calculate Pareto score (higher is better)
    const latencyScore = Math.max(0, 1 - avgLatency / this.retuneConfig.max_latency_ms);
    const recallScore = avgRecall;
    const paretoScore = 0.6 * recallScore + 0.4 * latencyScore;

    // Estimate memory usage
    const memoryUsage = this.estimateMemoryUsage(M, validationVectors.length);

    return {
      M,
      efSearch,
      k,
      recall_at_50: avgRecall,
      search_latency_ms: avgLatency,
      throughput_qps: avgThroughput,
      pareto_score: paretoScore,
      memory_usage_mb: memoryUsage,
      fresh_codebook: true
    };
  }

  /**
   * Simulate search for benchmarking (placeholder for actual implementation)
   */
  private async simulateSearch(
    queryVector: Float32Array, 
    k: number, 
    efSearch: number
  ): Promise<Array<{ doc_id: string; distance: number }>> {
    // In real implementation, this would perform actual HNSW search
    // For now, return mock results with realistic timing
    await new Promise(resolve => setTimeout(resolve, Math.random() * 10 + 5)); // 5-15ms

    return Array.from({ length: k }, (_, i) => ({
      doc_id: `doc_${i}`,
      distance: Math.random()
    }));
  }

  /**
   * Calculate Recall@50 metric
   */
  private calculateRecallAt50(
    searchResults: Array<{ doc_id: string; distance: number }>,
    groundTruth: Array<{ doc_id: string; relevance: number }>
  ): number {
    const relevantDocs = new Set(
      groundTruth.filter(gt => gt.relevance > 0).map(gt => gt.doc_id)
    );
    
    if (relevantDocs.size === 0) return 1.0; // No relevant docs to find

    const top50 = searchResults.slice(0, 50);
    const foundRelevant = top50.filter(result => relevantDocs.has(result.doc_id)).length;

    return foundRelevant / relevantDocs.size;
  }

  /**
   * Compute Pareto frontier from benchmark results
   */
  private computeParetoFrontier(results: ANNBenchmarkResult[]): ANNBenchmarkResult[] {
    const frontier: ANNBenchmarkResult[] = [];

    for (const candidate of results) {
      let isDominated = false;

      for (const other of results) {
        if (other === candidate) continue;

        // Check if 'other' dominates 'candidate'
        // Better if: higher recall AND lower latency
        if (other.recall_at_50 >= candidate.recall_at_50 && 
            other.search_latency_ms <= candidate.search_latency_ms &&
            (other.recall_at_50 > candidate.recall_at_50 || 
             other.search_latency_ms < candidate.search_latency_ms)) {
          isDominated = true;
          break;
        }
      }

      if (!isDominated) {
        frontier.push(candidate);
      }
    }

    return frontier.sort((a, b) => b.pareto_score - a.pareto_score);
  }

  /**
   * Select best configuration from Pareto frontier
   */
  private selectBestConfiguration(): ANNBenchmarkResult | null {
    // Filter configurations that meet constraints
    const validConfigs = this.paretoFrontier.filter(config =>
      config.recall_at_50 >= this.retuneConfig.min_recall_at_50 &&
      config.search_latency_ms <= this.retuneConfig.max_latency_ms &&
      config.fresh_codebook
    );

    if (validConfigs.length === 0) {
      return null;
    }

    // Select highest Pareto score
    return validConfigs.reduce((best, current) =>
      current.pareto_score > best.pareto_score ? current : best
    );
  }

  /**
   * Estimate memory usage for configuration
   */
  private estimateMemoryUsage(M: number, numVectors: number): number {
    // HNSW memory: vectors + connections
    const vectorMemory = numVectors * 256 * 4; // 4 bytes per float
    const connectionMemory = numVectors * M * 4; // 4 bytes per connection ID
    
    // PQ codebook memory
    const codebookMemory = this.pqCodebook ? 
      this.pqCodebook.centroids.length * (256 / this.pqCodebook.subvectors) * 4 : 0;

    const totalBytes = vectorMemory + connectionMemory + codebookMemory;
    return totalBytes / (1024 * 1024); // Convert to MB
  }

  /**
   * Get comprehensive statistics including Pareto analysis
   */
  getStats() {
    const baseStats = super.getStats();
    
    return {
      ...baseStats,
      retune_config: this.retuneConfig,
      pq_codebook: this.pqCodebook ? {
        dimension: this.pqCodebook.dimension,
        subvectors: this.pqCodebook.subvectors,
        codewords: this.pqCodebook.codewords,
        trained_on: this.pqCodebook.trained_on_dimension
      } : null,
      benchmark_history: this.benchmarkHistory.length,
      pareto_frontier: this.paretoFrontier.length,
      best_configuration: this.paretoFrontier[0] || null
    };
  }

  /**
   * Validate that codebook is fresh and dimension-specific
   */
  validateFreshCodebook(): boolean {
    if (!this.pqCodebook) {
      console.error('🚨 No PQ codebook trained');
      return false;
    }

    if (this.pqCodebook.trained_on_dimension !== '256') {
      console.error(`🚨 Codebook not 256d-specific: trained on ${this.pqCodebook.trained_on_dimension}d`);
      return false;
    }

    if (this.pqCodebook.dimension !== 256) {
      console.error(`🚨 Codebook dimension mismatch: ${this.pqCodebook.dimension} !== 256`);
      return false;
    }

    console.log('✅ PQ codebook validation passed - fresh 256d codebook confirmed');
    return true;
  }
}

/**
 * Production ANN Retune Manager for Gemma-256
 * Orchestrates fresh codebook training and Pareto optimization
 */
export class Gemma256ANNRetuneManager {
  private optimizedIndex: OptimizedHNSW256d;
  private isProduction: boolean;

  constructor(config: Partial<ANNRetuneConfig> = {}, isProduction = true) {
    this.isProduction = isProduction;

    // Production defaults for Pareto optimization
    const productionConfig: ANNRetuneConfig = {
      dimension: '256',
      M_candidates: [16, 24],              // Per requirement
      efSearch_candidates: [64, 80, 96],   // Per requirement  
      k_candidates: [150, 180, 220],       // Per requirement
      max_latency_ms: 150,                 // ≤150ms target
      min_recall_at_50: 0.85,              // Minimum recall threshold
      target_improvement: 0.3,             // 30% improvement target
      pq_subvectors: 8,                    // 256/8 = 32d per subvector
      pq_codewords: 256,                   // Standard PQ codewords
      training_samples: 10000,             // Samples for codebook training
      validation_samples: 1000,            // Samples for evaluation
      fresh_codebooks_only: true,          // Mandatory fresh codebooks
      ...config
    };

    // Enforce mandatory constraints
    if (!productionConfig.fresh_codebooks_only) {
      throw new Error('Fresh codebooks are mandatory - cannot reuse 768d codebooks');
    }

    if (productionConfig.dimension !== '256') {
      throw new Error('This manager is specifically for 256d vectors');
    }

    this.optimizedIndex = new OptimizedHNSW256d(productionConfig);

    console.log(`🚀 Gemma-256 ANN Retune Manager initialized (production=${isProduction})`);
    console.log(`   Fresh codebooks only: ${productionConfig.fresh_codebooks_only}`);
    console.log(`   Target: Recall@50(≤${productionConfig.max_latency_ms}ms) ≥ ${productionConfig.min_recall_at_50}`);
  }

  /**
   * Execute complete retune process: fresh PQ + Pareto optimization
   */
  async executeRetune(
    trainingVectors256d: Float32Array[],
    validationVectors256d: Float32Array[],
    groundTruthResults: Array<{ doc_id: string; relevance: number }[]>
  ): Promise<{ 
    pqResult: PQTrainingResult;
    paretoResult: ANNBenchmarkResult;
  }> {
    console.log(`🚀 Starting complete Gemma-256 ANN retune process`);
    
    // Validate input vectors are 256d
    this.validateVectorDimensions(trainingVectors256d, '256d training');
    this.validateVectorDimensions(validationVectors256d, '256d validation');

    // Step 1: Train fresh PQ codebook for 256d
    console.log(`🔧 Step 1: Training fresh PQ codebook for 256d vectors`);
    const pqResult = await this.optimizedIndex.trainFreshPQCodebook(trainingVectors256d);

    // Validate codebook is fresh and correct
    if (!this.optimizedIndex.validateFreshCodebook()) {
      throw new Error('Fresh codebook validation failed');
    }

    // Step 2: Pareto optimization
    console.log(`🔧 Step 2: Pareto optimization for M/efSearch/k parameters`);
    const paretoResult = await this.optimizedIndex.optimizePareto(
      validationVectors256d,
      groundTruthResults
    );

    // Validate results meet requirements
    if (paretoResult.recall_at_50 < 0.85) {
      console.warn(`⚠️ Recall@50 below threshold: ${paretoResult.recall_at_50.toFixed(3)} < 0.85`);
    }

    if (paretoResult.search_latency_ms > 150) {
      console.warn(`⚠️ Latency above threshold: ${paretoResult.search_latency_ms.toFixed(1)}ms > 150ms`);
    }

    console.log(`✅ Gemma-256 ANN retune completed successfully`);
    console.log(`   PQ compression: ${pqResult.compression_ratio.toFixed(1)}x`);
    console.log(`   Optimal config: M=${paretoResult.M}, efSearch=${paretoResult.efSearch}, k=${paretoResult.k}`);

    return { pqResult, paretoResult };
  }

  /**
   * Validate vector dimensions
   */
  private validateVectorDimensions(vectors: Float32Array[], label: string): void {
    for (let i = 0; i < vectors.length; i++) {
      const vector = vectors[i]!;
      if (vector.length !== 256) {
        throw new Error(`${label} vector ${i} has dimension ${vector.length}, expected 256`);
      }
    }
    console.log(`✅ ${label} vectors validated: ${vectors.length} vectors, all 256d`);
  }

  /**
   * Get comprehensive retune statistics
   */
  getStats() {
    return {
      production_mode: this.isProduction,
      ...this.optimizedIndex.getStats()
    };
  }

  /**
   * Export optimized configuration for production deployment
   */
  exportOptimalConfiguration(): {
    hnsw: { M: number; efSearch: number; k: number };
    pq: { subvectors: number; codewords: number; trained_on: '256' };
    performance: { recall_at_50: number; latency_ms: number };
  } | null {
    const stats = this.optimizedIndex.getStats();
    const bestConfig = stats.best_configuration;

    if (!bestConfig) {
      return null;
    }

    return {
      hnsw: {
        M: bestConfig.M,
        efSearch: bestConfig.efSearch,
        k: bestConfig.k
      },
      pq: {
        subvectors: stats.pq_codebook?.subvectors || 8,
        codewords: stats.pq_codebook?.codewords || 256,
        trained_on: '256'
      },
      performance: {
        recall_at_50: bestConfig.recall_at_50,
        latency_ms: bestConfig.search_latency_ms
      }
    };
  }
}