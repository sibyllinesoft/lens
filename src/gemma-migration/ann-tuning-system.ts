/**
 * ANN (Approximate Nearest Neighbor) Retuning System for Gemma
 * Sweeps HNSW parameters and PQ/INT8 quantization for optimal performance
 */

import { z } from 'zod';
import * as fs from 'fs';

const ANNConfigSchema = z.object({
  buildParams: z.object({
    M: z.array(z.number()).default([16, 24]),
    efConstruction: z.array(z.number()).default([200, 400])
  }),
  queryParams: z.object({
    efSearch: z.array(z.number()).default([48, 64, 96]),
    k: z.array(z.number()).default([150, 220])
  }),
  quantization: z.object({
    pqEnabled: z.boolean().default(true),
    int8Enabled: z.boolean().default(true),
    pqSubvectors: z.number().default(64),
    pqBits: z.number().default(8)
  }),
  slaTargets: z.object({
    recallAt50: z.number().default(0.95),
    p95LatencyMs: z.number().default(150),
    qpsTarget: z.number().default(1000)
  })
});

export type ANNConfig = z.infer<typeof ANNConfigSchema>;

interface HNSWParameters {
  M: number;
  efConstruction: number;
  efSearch: number;
  k: number;
}

interface QuantizationConfig {
  type: 'none' | 'pq' | 'int8' | 'pq+int8';
  pqSubvectors?: number;
  pqBits?: number;
}

interface ANNBenchmarkResult {
  parameters: HNSWParameters;
  quantization: QuantizationConfig;
  metrics: {
    recallAt50: number;
    recallAt50SLA: number; // Recall@50 within SLA latency
    p50LatencyMs: number;
    p95LatencyMs: number;
    p99LatencyMs: number;
    qps: number;
    memoryMB: number;
    indexSizeMB: number;
    buildTimeMs: number;
  };
  paretoOptimal: boolean;
  hash: string;
}

interface ParetoPoint {
  result: ANNBenchmarkResult;
  reason: string;
  tradeoffs: string;
}

/**
 * HNSW parameter sweep and optimization
 */
export class HNSWTuner {
  private config: ANNConfig;
  private benchmarkResults: ANNBenchmarkResult[] = [];

  constructor(config: ANNConfig) {
    this.config = ANNConfigSchema.parse(config);
  }

  /**
   * Generate all parameter combinations for sweeping
   */
  generateParameterSpace(): HNSWParameters[] {
    const combinations: HNSWParameters[] = [];
    
    for (const M of this.config.buildParams.M) {
      for (const efConstruction of this.config.buildParams.efConstruction) {
        for (const efSearch of this.config.queryParams.efSearch) {
          for (const k of this.config.queryParams.k) {
            combinations.push({ M, efConstruction, efSearch, k });
          }
        }
      }
    }
    
    console.log(`Generated ${combinations.length} parameter combinations to test`);
    return combinations;
  }

  /**
   * Benchmark single parameter configuration
   * In production, this would interface with actual HNSW implementation
   */
  async benchmarkParameters(
    params: HNSWParameters,
    quantConfig: QuantizationConfig,
    vectors: Float32Array[],
    queries: Float32Array[],
    groundTruth: number[][]
  ): Promise<ANNBenchmarkResult> {
    console.log(`Benchmarking M=${params.M}, ef_c=${params.efConstruction}, ef_s=${params.efSearch}, k=${params.k}`);
    
    // Simulate HNSW index building
    const buildStartTime = Date.now();
    await this.simulateIndexBuild(vectors, params, quantConfig);
    const buildTimeMs = Date.now() - buildStartTime;
    
    // Simulate query performance
    const queryMetrics = await this.simulateQueryPerformance(
      queries,
      params,
      quantConfig,
      groundTruth
    );
    
    // Calculate memory usage based on parameters
    const memoryEstimate = this.estimateMemoryUsage(vectors.length, params, quantConfig);
    
    const result: ANNBenchmarkResult = {
      parameters: params,
      quantization: quantConfig,
      metrics: {
        ...queryMetrics,
        buildTimeMs,
        memoryMB: memoryEstimate.memory,
        indexSizeMB: memoryEstimate.indexSize
      },
      paretoOptimal: false, // Will be determined later
      hash: this.generateParameterHash(params, quantConfig)
    };
    
    this.benchmarkResults.push(result);
    return result;
  }

  /**
   * Simulate HNSW index building (in production, use actual implementation)
   */
  private async simulateIndexBuild(
    vectors: Float32Array[],
    params: HNSWParameters,
    quantConfig: QuantizationConfig
  ): Promise<void> {
    // Simulate build time based on parameters
    const baseTime = vectors.length * 0.1; // 0.1ms per vector base
    const complexityFactor = params.M * Math.log2(params.efConstruction);
    const quantizationOverhead = quantConfig.type === 'none' ? 1.0 : 1.2;
    
    const buildTimeMs = baseTime * complexityFactor * quantizationOverhead;
    
    // Simulate actual build delay
    await new Promise(resolve => setTimeout(resolve, Math.min(buildTimeMs / 1000, 100)));
  }

  /**
   * Simulate query performance metrics
   */
  private async simulateQueryPerformance(
    queries: Float32Array[],
    params: HNSWParameters,
    quantConfig: QuantizationConfig,
    groundTruth: number[][]
  ): Promise<{
    recallAt50: number;
    recallAt50SLA: number;
    p50LatencyMs: number;
    p95LatencyMs: number;
    p99LatencyMs: number;
    qps: number;
  }> {
    const latencies: number[] = [];
    let totalRecall = 0;
    let slaCompliantQueries = 0;
    let slaRecallSum = 0;
    
    for (let i = 0; i < queries.length; i++) {
      // Simulate query latency based on parameters
      const baseLatency = this.simulateQueryLatency(params, quantConfig);
      latencies.push(baseLatency);
      
      // Simulate recall calculation
      const recall = this.simulateRecallCalculation(params, quantConfig, i, groundTruth[i]);
      totalRecall += recall;
      
      // Check SLA compliance
      if (baseLatency <= this.config.slaTargets.p95LatencyMs) {
        slaCompliantQueries++;
        slaRecallSum += recall;
      }
    }
    
    latencies.sort((a, b) => a - b);
    const n = latencies.length;
    
    return {
      recallAt50: totalRecall / queries.length,
      recallAt50SLA: slaCompliantQueries > 0 ? slaRecallSum / slaCompliantQueries : 0,
      p50LatencyMs: latencies[Math.floor(n * 0.5)],
      p95LatencyMs: latencies[Math.floor(n * 0.95)],
      p99LatencyMs: latencies[Math.floor(n * 0.99)],
      qps: slaCompliantQueries > 0 ? 1000 / (slaRecallSum / slaCompliantQueries) : 0
    };
  }

  /**
   * Simulate single query latency
   */
  private simulateQueryLatency(params: HNSWParameters, quantConfig: QuantizationConfig): number {
    // Base latency factors
    const baseLatency = 5; // 5ms base
    const efSearchFactor = Math.log2(params.efSearch) * 2;
    const quantizationSpeedup = quantConfig.type === 'int8' ? 0.7 : 
                               quantConfig.type === 'pq' ? 0.6 :
                               quantConfig.type === 'pq+int8' ? 0.5 : 1.0;
    
    const latency = (baseLatency + efSearchFactor) * quantizationSpeedup;
    
    // Add some realistic variance
    const variance = latency * (0.1 + Math.random() * 0.2);
    return Math.max(1, latency + variance);
  }

  /**
   * Simulate recall calculation
   */
  private simulateRecallCalculation(
    params: HNSWParameters,
    quantConfig: QuantizationConfig,
    queryIdx: number,
    groundTruth: number[]
  ): number {
    // Base recall depends on efSearch and k
    const baseRecall = Math.min(0.98, 0.7 + (params.efSearch / 200) * 0.25);
    
    // Quantization impact on recall
    const quantizationPenalty = quantConfig.type === 'none' ? 0 :
                               quantConfig.type === 'int8' ? 0.01 :
                               quantConfig.type === 'pq' ? 0.02 :
                               quantConfig.type === 'pq+int8' ? 0.03 : 0;
    
    // k parameter impact
    const kBonus = params.k > 150 ? (params.k - 150) * 0.0002 : 0;
    
    const recall = Math.min(1.0, Math.max(0.5, baseRecall - quantizationPenalty + kBonus));
    
    // Add some realistic variance per query
    const variance = (Math.random() - 0.5) * 0.05;
    return Math.min(1.0, Math.max(0.0, recall + variance));
  }

  /**
   * Estimate memory usage for parameter configuration
   */
  private estimateMemoryUsage(
    numVectors: number,
    params: HNSWParameters,
    quantConfig: QuantizationConfig
  ): { memory: number; indexSize: number } {
    // Vector storage
    const vectorDimension = 768; // Assume 768d for Gemma
    let vectorStorage = numVectors * vectorDimension * 4; // 4 bytes per float32
    
    // Apply quantization compression
    if (quantConfig.type === 'int8' || quantConfig.type === 'pq+int8') {
      vectorStorage *= 0.25; // INT8 is 1/4 size
    }
    if (quantConfig.type === 'pq' || quantConfig.type === 'pq+int8') {
      vectorStorage *= 0.125; // PQ typically ~1/8 size
    }
    
    // HNSW graph overhead
    const avgConnections = params.M * 1.5; // Average connections per node
    const graphOverhead = numVectors * avgConnections * 4; // 4 bytes per connection ID
    
    // Additional metadata
    const metadataOverhead = numVectors * 32; // 32 bytes per vector for metadata
    
    const totalMemory = vectorStorage + graphOverhead + metadataOverhead;
    const indexSize = totalMemory * 1.2; // 20% overhead for index structures
    
    return {
      memory: totalMemory / (1024 * 1024), // Convert to MB
      indexSize: indexSize / (1024 * 1024)
    };
  }

  /**
   * Generate hash for parameter configuration
   */
  private generateParameterHash(params: HNSWParameters, quantConfig: QuantizationConfig): string {
    const hashInput = JSON.stringify({ params, quantConfig });
    return require('crypto').createHash('md5').update(hashInput).digest('hex').substring(0, 12);
  }

  /**
   * Identify Pareto optimal points
   */
  identifyParetoOptimal(): ParetoPoint[] {
    // Mark Pareto optimal points
    for (let i = 0; i < this.benchmarkResults.length; i++) {
      const current = this.benchmarkResults[i];
      let isPareto = true;
      
      for (let j = 0; j < this.benchmarkResults.length; j++) {
        if (i === j) continue;
        
        const other = this.benchmarkResults[j];
        
        // Check if 'other' dominates 'current'
        // Better recall, better (lower) latency, better (lower) memory
        if (other.metrics.recallAt50 >= current.metrics.recallAt50 &&
            other.metrics.p95LatencyMs <= current.metrics.p95LatencyMs &&
            other.metrics.memoryMB <= current.metrics.memoryMB &&
            (other.metrics.recallAt50 > current.metrics.recallAt50 ||
             other.metrics.p95LatencyMs < current.metrics.p95LatencyMs ||
             other.metrics.memoryMB < current.metrics.memoryMB)) {
          isPareto = false;
          break;
        }
      }
      
      current.paretoOptimal = isPareto;
    }
    
    // Extract Pareto points with analysis
    const paretoPoints = this.benchmarkResults
      .filter(result => result.paretoOptimal)
      .sort((a, b) => b.metrics.recallAt50 - a.metrics.recallAt50) // Sort by recall desc
      .map(result => this.analyzeParetoPoint(result));
    
    return paretoPoints;
  }

  /**
   * Analyze why a point is Pareto optimal
   */
  private analyzeParetoPoint(result: ANNBenchmarkResult): ParetoPoint {
    const { metrics, parameters } = result;
    
    let reason = '';
    let tradeoffs = '';
    
    if (metrics.recallAt50 >= 0.95 && metrics.p95LatencyMs <= 100) {
      reason = 'High recall with excellent latency';
    } else if (metrics.recallAt50 >= 0.98) {
      reason = 'Maximum recall configuration';
      tradeoffs = 'Higher latency and memory usage';
    } else if (metrics.p95LatencyMs <= 50) {
      reason = 'Ultra-low latency optimized';
      tradeoffs = 'Moderate recall reduction';
    } else if (metrics.memoryMB <= 1000) {
      reason = 'Memory-efficient configuration';
      tradeoffs = 'Balanced recall and latency';
    } else {
      reason = 'Balanced performance point';
    }
    
    return { result, reason, tradeoffs };
  }

  /**
   * Get recommendations based on use case
   */
  getRecommendations(): {
    production: ANNBenchmarkResult;
    lowLatency: ANNBenchmarkResult;
    highRecall: ANNBenchmarkResult;
  } {
    const paretoPoints = this.identifyParetoOptimal();
    
    if (paretoPoints.length === 0) {
      throw new Error('No Pareto optimal configurations found');
    }
    
    // Production: Best balance meeting SLA requirements
    const production = paretoPoints.find(p => 
      p.result.metrics.recallAt50SLA >= this.config.slaTargets.recallAt50 &&
      p.result.metrics.p95LatencyMs <= this.config.slaTargets.p95LatencyMs
    )?.result || paretoPoints[0].result;
    
    // Low latency: Minimum p95 latency
    const lowLatency = paretoPoints.reduce((min, p) => 
      p.result.metrics.p95LatencyMs < min.result.metrics.p95LatencyMs ? p : min
    ).result;
    
    // High recall: Maximum recall
    const highRecall = paretoPoints.reduce((max, p) => 
      p.result.metrics.recallAt50 > max.result.metrics.recallAt50 ? p : max
    ).result;
    
    return { production, lowLatency, highRecall };
  }

  /**
   * Save tuning results
   */
  async saveTuningResults(outputPath: string): Promise<void> {
    const paretoPoints = this.identifyParetoOptimal();
    const recommendations = this.getRecommendations();
    
    const report = {
      version: '1.0.0',
      timestamp: new Date().toISOString(),
      config: this.config,
      totalConfigurations: this.benchmarkResults.length,
      paretoOptimalCount: paretoPoints.length,
      recommendations: {
        production: {
          config: recommendations.production.parameters,
          quantization: recommendations.production.quantization,
          metrics: recommendations.production.metrics,
          hash: recommendations.production.hash
        },
        lowLatency: {
          config: recommendations.lowLatency.parameters,
          quantization: recommendations.lowLatency.quantization,
          metrics: recommendations.lowLatency.metrics,
          hash: recommendations.lowLatency.hash
        },
        highRecall: {
          config: recommendations.highRecall.parameters,
          quantization: recommendations.highRecall.quantization,
          metrics: recommendations.highRecall.metrics,
          hash: recommendations.highRecall.hash
        }
      },
      paretoPoints: paretoPoints.map(p => ({
        parameters: p.result.parameters,
        quantization: p.result.quantization,
        metrics: p.result.metrics,
        reason: p.reason,
        tradeoffs: p.tradeoffs,
        hash: p.result.hash
      })),
      allResults: this.benchmarkResults
    };
    
    await fs.promises.writeFile(
      outputPath,
      JSON.stringify(report, null, 2),
      'utf8'
    );
  }
}

/**
 * PQ (Product Quantization) and INT8 optimization
 */
export class QuantizationOptimizer {
  private config: ANNConfig;

  constructor(config: ANNConfig) {
    this.config = config;
  }

  /**
   * Train fresh PQ codebooks on Gemma vectors
   * NEVER reuse ada-002 codebooks due to different distributions
   */
  async trainPQCodebooks(
    vectors: Float32Array[],
    subvectors: number = 64,
    bits: number = 8
  ): Promise<{
    codebooks: Float32Array[][];
    reconstructionError: number;
    compressionRatio: number;
  }> {
    console.log(`Training PQ codebooks: ${subvectors} subvectors, ${bits} bits`);
    
    if (vectors.length === 0) {
      throw new Error('No vectors provided for PQ training');
    }
    
    const dimension = vectors[0].length;
    const subvectorDim = Math.floor(dimension / subvectors);
    
    if (subvectorDim === 0) {
      throw new Error(`Too many subvectors (${subvectors}) for dimension ${dimension}`);
    }
    
    const codebooks: Float32Array[][] = [];
    const numCentroids = Math.pow(2, bits);
    let totalReconstructionError = 0;
    
    // Train codebook for each subvector
    for (let s = 0; s < subvectors; s++) {
      const start = s * subvectorDim;
      const end = Math.min(start + subvectorDim, dimension);
      
      // Extract subvectors
      const subvectorData = vectors.map(vector => 
        vector.slice(start, end)
      );
      
      // Run k-means clustering for this subvector
      const { centroids, error } = await this.kMeansClustering(subvectorData, numCentroids);
      codebooks.push(centroids);
      totalReconstructionError += error;
      
      if (s % 10 === 0) {
        console.log(`Trained codebook ${s + 1}/${subvectors}`);
      }
    }
    
    const avgReconstructionError = totalReconstructionError / subvectors;
    const compressionRatio = (dimension * 4) / (subvectors * bits / 8); // float32 to PQ bits
    
    console.log(`PQ training complete. Compression: ${compressionRatio.toFixed(1)}x, Error: ${avgReconstructionError.toFixed(6)}`);
    
    return {
      codebooks,
      reconstructionError: avgReconstructionError,
      compressionRatio
    };
  }

  /**
   * K-means clustering for PQ codebook generation
   */
  private async kMeansClustering(
    data: Float32Array[],
    k: number,
    maxIterations: number = 100
  ): Promise<{ centroids: Float32Array[]; error: number }> {
    const dimension = data[0].length;
    const centroids: Float32Array[] = [];
    
    // Initialize centroids randomly
    for (let i = 0; i < k; i++) {
      const randomIndex = Math.floor(Math.random() * data.length);
      centroids.push(new Float32Array(data[randomIndex]));
    }
    
    let converged = false;
    let iteration = 0;
    let assignments: number[] = [];
    
    while (!converged && iteration < maxIterations) {
      assignments = [];
      const clusterSums: Float32Array[] = centroids.map(() => new Float32Array(dimension));
      const clusterCounts: number[] = new Array(k).fill(0);
      
      // Assign points to closest centroids
      for (let i = 0; i < data.length; i++) {
        let bestDistance = Infinity;
        let bestCentroid = 0;
        
        for (let c = 0; c < k; c++) {
          const distance = this.euclideanDistance(data[i], centroids[c]);
          if (distance < bestDistance) {
            bestDistance = distance;
            bestCentroid = c;
          }
        }
        
        assignments.push(bestCentroid);
        clusterCounts[bestCentroid]++;
        
        // Add to cluster sum
        for (let d = 0; d < dimension; d++) {
          clusterSums[bestCentroid][d] += data[i][d];
        }
      }
      
      // Update centroids
      let totalMovement = 0;
      for (let c = 0; c < k; c++) {
        if (clusterCounts[c] > 0) {
          const oldCentroid = new Float32Array(centroids[c]);
          
          for (let d = 0; d < dimension; d++) {
            centroids[c][d] = clusterSums[c][d] / clusterCounts[c];
          }
          
          totalMovement += this.euclideanDistance(oldCentroid, centroids[c]);
        }
      }
      
      converged = totalMovement < 1e-6;
      iteration++;
    }
    
    // Calculate final error
    let totalError = 0;
    for (let i = 0; i < data.length; i++) {
      const centroidIndex = assignments[i];
      const error = this.euclideanDistance(data[i], centroids[centroidIndex]);
      totalError += error * error;
    }
    
    return {
      centroids,
      error: totalError / data.length
    };
  }

  private euclideanDistance(a: Float32Array, b: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const diff = a[i] - b[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }

  /**
   * Validate PQ quantization impact on quality
   */
  async validateQuantizationQuality(
    originalVectors: Float32Array[],
    quantizedVectors: Float32Array[],
    queries: Float32Array[],
    groundTruth: number[][]
  ): Promise<{
    nDCGDelta: number;
    recallDelta: number;
    p95LatencyImprovement: number;
    compressionAchieved: number;
    qualityAcceptable: boolean;
  }> {
    // Simulate nDCG and recall calculations
    // In production, this would run actual ranking evaluation
    
    const originalNDCG = 0.85; // Baseline
    const quantizedNDCG = 0.844; // Simulated drop
    const nDCGDelta = quantizedNDCG - originalNDCG;
    
    const originalRecall = 0.92;
    const quantizedRecall = 0.918;
    const recallDelta = quantizedRecall - originalRecall;
    
    // Latency improvement from quantization
    const p95LatencyImprovement = 0.25; // 25% improvement
    
    const compressionAchieved = 8.0; // 8x compression
    
    // Quality gate: nDCG drop must be ≤ 0.5 percentage points
    const qualityAcceptable = Math.abs(nDCGDelta) <= 0.005;
    
    return {
      nDCGDelta,
      recallDelta,
      p95LatencyImprovement,
      compressionAchieved,
      qualityAcceptable
    };
  }
}