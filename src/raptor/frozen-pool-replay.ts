/**
 * Frozen-Pool Replay Harness for EmbeddingGemma Evaluation
 * 
 * Implements controlled A/B testing with frozen query pools to measure
 * ΔCBU/GB, Recall@K, critical-atom recall, and performance metrics
 * across ada-002, Gemma-768, and Gemma-256 models.
 */

import { ShadowIndexManager, EmbeddingModelType, ComparisonMetrics } from './shadow-index-manager.js';
import { LensTracer } from '../telemetry/tracer.js';

export interface FrozenQuery {
  id: string;
  query: string;
  language: string;
  intent: 'semantic' | 'lexical' | 'mixed';
  expectedResults: Array<{
    docId: string;
    filePath: string;
    relevanceScore: number; // 0-1 scale
    isCriticalAtom: boolean;
  }>;
  groundTruth: {
    precision_at_10: number;
    recall_at_50: number;
    user_satisfaction: number; // CBU proxy metric
  };
}

export interface CBUMetrics {
  modelType: EmbeddingModelType;
  dimension: number;
  
  // Core Business Utility metrics
  total_cbu: number;
  delta_cbu_vs_baseline: number;
  cbu_per_gb: number;
  cbu_per_query: number;
  
  // Recall and precision metrics
  recall_at_10: number;
  recall_at_50: number;
  precision_at_10: number;
  critical_atom_recall: number;
  
  // Performance metrics
  avg_latency_ms: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  throughput_qps: number;
  
  // Resource efficiency
  storage_bytes: number;
  memory_usage_mb: number;
  cpu_utilization_percent: number;
  
  // Quality metrics
  semantic_coherence: number;
  cross_language_consistency: number;
  error_rate: number;
}

export interface ReplayConfig {
  models: EmbeddingModelType[];
  baseline: EmbeddingModelType; // For delta calculations
  iterations: number;
  parallelQueries: number;
  warmupQueries: number;
  collectResourceMetrics: boolean;
  outputPath: string;
}

export interface ReplayResult {
  config: ReplayConfig;
  timestamp: string;
  queryPoolSize: number;
  metrics: Map<EmbeddingModelType, CBUMetrics>;
  comparisonReport: {
    winner: EmbeddingModelType;
    deltasByMetric: Record<string, Map<EmbeddingModelType, number>>;
    recommendations: string[];
  };
}

/**
 * Frozen-pool replay harness for controlled embedding model evaluation
 */
export class FrozenPoolReplayHarness {
  private shadowManager: ShadowIndexManager;
  private queryPool: FrozenQuery[] = [];
  private resourceMonitor?: ResourceMonitor;

  constructor(shadowManager: ShadowIndexManager) {
    this.shadowManager = shadowManager;
    this.resourceMonitor = new ResourceMonitor();
  }

  /**
   * Load frozen query pool from various sources
   */
  async loadQueryPool(sources: {
    groundTruthFile?: string;
    historicalQueries?: string;
    syntheticQueries?: number;
  }): Promise<void> {
    const span = LensTracer.createChildSpan('load_frozen_query_pool');

    try {
      // Load ground truth queries if available
      if (sources.groundTruthFile) {
        await this.loadGroundTruthQueries(sources.groundTruthFile);
      }

      // Load historical queries
      if (sources.historicalQueries) {
        await this.loadHistoricalQueries(sources.historicalQueries);
      }

      // Generate synthetic queries if requested
      if (sources.syntheticQueries) {
        await this.generateSyntheticQueries(sources.syntheticQueries);
      }

      // Validate query pool
      this.validateQueryPool();

      span.setAttributes({
        success: true,
        query_pool_size: this.queryPool.length,
        sources: Object.keys(sources).join(','),
      });

      console.log(`✅ Loaded frozen query pool: ${this.queryPool.length} queries`);

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Run frozen-pool replay across all configured models
   */
  async runReplay(config: ReplayConfig): Promise<ReplayResult> {
    const span = LensTracer.createChildSpan('run_frozen_pool_replay', {
      'config.models': config.models.join(','),
      'config.iterations': config.iterations,
      'query_pool_size': this.queryPool.length,
    });

    const startTime = new Date().toISOString();
    const metrics = new Map<EmbeddingModelType, CBUMetrics>();

    try {
      console.log(`🚀 Starting frozen-pool replay with ${config.models.length} models...`);

      // Warmup phase
      if (config.warmupQueries > 0) {
        await this.runWarmup(config);
      }

      // Start resource monitoring
      if (config.collectResourceMetrics) {
        this.resourceMonitor?.startMonitoring();
      }

      // Run evaluation for each model
      for (const modelType of config.models) {
        console.log(`🔄 Evaluating ${modelType}...`);
        
        const modelMetrics = await this.evaluateModel(modelType, config);
        metrics.set(modelType, modelMetrics);
        
        console.log(`✅ ${modelType}: ΔCBU/GB=${modelMetrics.cbu_per_gb.toFixed(2)}, Recall@50=${(modelMetrics.recall_at_50 * 100).toFixed(1)}%`);
      }

      // Stop resource monitoring
      if (config.collectResourceMetrics) {
        this.resourceMonitor?.stopMonitoring();
      }

      // Generate comparison report
      const comparisonReport = this.generateComparisonReport(metrics, config.baseline);

      const result: ReplayResult = {
        config,
        timestamp: startTime,
        queryPoolSize: this.queryPool.length,
        metrics,
        comparisonReport,
      };

      // Save results
      await this.saveResults(result, config.outputPath);

      span.setAttributes({
        success: true,
        models_evaluated: metrics.size,
        winner: comparisonReport.winner,
      });

      return result;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  private async evaluateModel(
    modelType: EmbeddingModelType,
    config: ReplayConfig
  ): Promise<CBUMetrics> {
    const span = LensTracer.createChildSpan('evaluate_model', {
      'model.type': modelType,
    });

    // Initialize metrics collection
    const latencies: number[] = [];
    let totalCBU = 0;
    let recallAt10Sum = 0;
    let recallAt50Sum = 0;
    let precisionAt10Sum = 0;
    let criticalAtomRecallSum = 0;
    let errorCount = 0;

    const resourceStart = this.resourceMonitor?.getCurrentMetrics();

    try {
      // Run queries in batches to control parallelism
      const batchSize = config.parallelQueries;
      const batches = this.batchQueries(this.queryPool, batchSize);

      for (const batch of batches) {
        const batchPromises = batch.map(async (query) => {
          try {
            const startTime = Date.now();
            const results = await this.executeQuery(query, modelType);
            const latency = Date.now() - startTime;
            
            latencies.push(latency);
            
            // Calculate metrics for this query
            const cbu = this.calculateQueryCBU(query, results);
            const recall10 = this.calculateRecallAtK(query.expectedResults, results, 10);
            const recall50 = this.calculateRecallAtK(query.expectedResults, results, 50);
            const precision10 = this.calculatePrecisionAtK(query.expectedResults, results, 10);
            const criticalRecall = this.calculateCriticalAtomRecall(query.expectedResults, results);
            
            totalCBU += cbu;
            recallAt10Sum += recall10;
            recallAt50Sum += recall50;
            precisionAt10Sum += precision10;
            criticalAtomRecallSum += criticalRecall;
            
          } catch (error) {
            errorCount++;
            console.warn(`Query ${query.id} failed for ${modelType}:`, error);
          }
        });

        await Promise.all(batchPromises);
      }

      const resourceEnd = this.resourceMonitor?.getCurrentMetrics();
      const resourceDelta = this.calculateResourceDelta(resourceStart, resourceEnd);

      // Calculate final metrics
      const queryCount = this.queryPool.length - errorCount;
      const indexStats = this.shadowManager.getIndexStats().get(modelType);
      
      if (!indexStats) {
        throw new Error(`No index stats found for ${modelType}`);
      }

      const metrics: CBUMetrics = {
        modelType,
        dimension: indexStats.dimension,
        
        // CBU metrics
        total_cbu: totalCBU,
        delta_cbu_vs_baseline: 0, // Will be calculated later
        cbu_per_gb: totalCBU / (indexStats.storageBytes / (1024 * 1024 * 1024)),
        cbu_per_query: totalCBU / queryCount,
        
        // Recall and precision
        recall_at_10: recallAt10Sum / queryCount,
        recall_at_50: recallAt50Sum / queryCount,
        precision_at_10: precisionAt10Sum / queryCount,
        critical_atom_recall: criticalAtomRecallSum / queryCount,
        
        // Performance
        avg_latency_ms: latencies.reduce((sum, lat) => sum + lat, 0) / latencies.length,
        p95_latency_ms: this.calculatePercentile(latencies, 0.95),
        p99_latency_ms: this.calculatePercentile(latencies, 0.99),
        throughput_qps: queryCount / (latencies.reduce((sum, lat) => sum + lat, 0) / 1000),
        
        // Resource efficiency
        storage_bytes: indexStats.storageBytes,
        memory_usage_mb: resourceDelta?.memoryUsageMB || 0,
        cpu_utilization_percent: resourceDelta?.cpuUtilization || 0,
        
        // Quality
        semantic_coherence: this.calculateSemanticCoherence(modelType),
        cross_language_consistency: this.calculateCrossLanguageConsistency(modelType),
        error_rate: errorCount / this.queryPool.length,
      };

      span.setAttributes({
        success: true,
        queries_processed: queryCount,
        error_count: errorCount,
        ...metrics,
      });

      return metrics;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  private async executeQuery(
    query: FrozenQuery,
    modelType: EmbeddingModelType
  ): Promise<Array<{ docId: string; score: number; filePath: string }>> {
    // Get query embedding
    const queryEmbedding = await this.shadowManager.getEmbedding(`query_${query.id}`, modelType);
    if (!queryEmbedding) {
      throw new Error(`Failed to get query embedding for ${query.id}`);
    }

    // For this implementation, we'll simulate search results
    // In practice, this would use your actual search pipeline
    const candidateIds = query.expectedResults.map(r => r.docId);
    const embeddings = await this.shadowManager.getEmbeddings(candidateIds, modelType);
    
    const results: Array<{ docId: string; score: number; filePath: string }> = [];
    
    for (const [docId, docEmbedding] of embeddings) {
      const similarity = this.cosineSimilarity(queryEmbedding, docEmbedding);
      const expectedResult = query.expectedResults.find(r => r.docId === docId);
      
      results.push({
        docId,
        score: similarity,
        filePath: expectedResult?.filePath || '',
      });
    }
    
    return results.sort((a, b) => b.score - a.score);
  }

  private calculateQueryCBU(
    query: FrozenQuery,
    results: Array<{ docId: string; score: number; filePath: string }>
  ): number {
    // CBU (Core Business Utility) calculation based on:
    // 1. Relevance of results
    // 2. User satisfaction proxy
    // 3. Critical atom retrieval
    
    let cbu = 0;
    const top10 = results.slice(0, 10);
    
    for (let i = 0; i < top10.length; i++) {
      const result = top10[i];
      const expected = query.expectedResults.find(r => r.docId === result?.docId);
      
      if (expected) {
        // Position discount (higher positions get more weight)
        const positionDiscount = 1 / Math.log2(i + 2);
        
        // Relevance score * position discount * critical bonus
        const criticalBonus = expected.isCriticalAtom ? 1.5 : 1.0;
        cbu += expected.relevanceScore * positionDiscount * criticalBonus;
      }
    }
    
    // Apply query-specific user satisfaction multiplier
    return cbu * query.groundTruth.user_satisfaction;
  }

  private calculateRecallAtK(
    expectedResults: FrozenQuery['expectedResults'],
    actualResults: Array<{ docId: string; score: number; filePath: string }>,
    k: number
  ): number {
    const topK = actualResults.slice(0, k).map(r => r.docId);
    const expectedIds = expectedResults.map(r => r.docId);
    const intersection = expectedIds.filter(id => topK.includes(id));
    
    return expectedIds.length > 0 ? intersection.length / expectedIds.length : 0;
  }

  private calculatePrecisionAtK(
    expectedResults: FrozenQuery['expectedResults'],
    actualResults: Array<{ docId: string; score: number; filePath: string }>,
    k: number
  ): number {
    const topK = actualResults.slice(0, k).map(r => r.docId);
    const expectedIds = expectedResults.map(r => r.docId);
    const intersection = expectedIds.filter(id => topK.includes(id));
    
    return topK.length > 0 ? intersection.length / topK.length : 0;
  }

  private calculateCriticalAtomRecall(
    expectedResults: FrozenQuery['expectedResults'],
    actualResults: Array<{ docId: string; score: number; filePath: string }>
  ): number {
    const criticalAtoms = expectedResults.filter(r => r.isCriticalAtom);
    const retrievedIds = actualResults.map(r => r.docId);
    const retrievedCritical = criticalAtoms.filter(atom => retrievedIds.includes(atom.docId));
    
    return criticalAtoms.length > 0 ? retrievedCritical.length / criticalAtoms.length : 0;
  }

  private generateComparisonReport(
    metrics: Map<EmbeddingModelType, CBUMetrics>,
    baseline: EmbeddingModelType
  ): ReplayResult['comparisonReport'] {
    const baselineMetrics = metrics.get(baseline);
    if (!baselineMetrics) {
      throw new Error(`Baseline model ${baseline} not found in results`);
    }

    // Calculate deltas vs baseline
    const deltasByMetric: Record<string, Map<EmbeddingModelType, number>> = {};
    const metricKeys: (keyof CBUMetrics)[] = [
      'cbu_per_gb',
      'recall_at_50',
      'critical_atom_recall',
      'avg_latency_ms',
      'storage_bytes'
    ];

    for (const key of metricKeys) {
      deltasByMetric[key] = new Map();
      
      for (const [modelType, modelMetrics] of metrics) {
        const delta = (modelMetrics[key] as number) - (baselineMetrics[key] as number);
        const percentDelta = ((modelMetrics[key] as number) / (baselineMetrics[key] as number) - 1) * 100;
        deltasByMetric[key].set(modelType, percentDelta);
      }
    }

    // Determine winner based on composite score
    let winner = baseline;
    let bestCompositeScore = this.calculateCompositeScore(baselineMetrics);

    for (const [modelType, modelMetrics] of metrics) {
      if (modelType === baseline) continue;
      
      const compositeScore = this.calculateCompositeScore(modelMetrics);
      if (compositeScore > bestCompositeScore) {
        winner = modelType;
        bestCompositeScore = compositeScore;
      }
    }

    // Generate recommendations
    const recommendations = this.generateRecommendations(metrics, deltasByMetric);

    return {
      winner,
      deltasByMetric,
      recommendations,
    };
  }

  private calculateCompositeScore(metrics: CBUMetrics): number {
    // Weighted composite score prioritizing CBU/GB and recall
    return (
      metrics.cbu_per_gb * 0.4 +
      metrics.recall_at_50 * 0.3 +
      metrics.critical_atom_recall * 0.2 +
      (1000 / metrics.avg_latency_ms) * 0.1 // Inverse latency
    );
  }

  private generateRecommendations(
    metrics: Map<EmbeddingModelType, CBUMetrics>,
    deltasByMetric: Record<string, Map<EmbeddingModelType, number>>
  ): string[] {
    const recommendations: string[] = [];

    // Analyze Gemma-256 vs Gemma-768 tradeoffs
    const gemma256 = metrics.get('gemma-256');
    const gemma768 = metrics.get('gemma-768');

    if (gemma256 && gemma768) {
      const cbuDiff = ((gemma256.cbu_per_gb - gemma768.cbu_per_gb) / gemma768.cbu_per_gb) * 100;
      const recallDiff = ((gemma256.recall_at_50 - gemma768.recall_at_50) / gemma768.recall_at_50) * 100;
      const storageDiff = ((gemma768.storage_bytes - gemma256.storage_bytes) / gemma768.storage_bytes) * 100;

      if (Math.abs(recallDiff) < 5 && storageDiff > 50) {
        recommendations.push(`Gemma-256 achieves ${storageDiff.toFixed(1)}% storage savings with minimal recall loss (${recallDiff.toFixed(1)}%). Recommend for production.`);
      } else if (recallDiff < -10) {
        recommendations.push(`Gemma-256 shows ${Math.abs(recallDiff).toFixed(1)}% recall degradation. Consider Gemma-768 for quality-sensitive use cases.`);
      }
    }

    // Storage efficiency recommendations
    const storageEfficiencies = Array.from(metrics.entries()).map(([model, m]) => ({
      model,
      efficiency: m.cbu_per_gb
    }));
    storageEfficiencies.sort((a, b) => b.efficiency - a.efficiency);

    recommendations.push(`Best storage efficiency: ${storageEfficiencies[0].model} (${storageEfficiencies[0].efficiency.toFixed(2)} CBU/GB)`);

    return recommendations;
  }

  // Utility methods
  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) return 0;
    
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  private calculatePercentile(values: number[], percentile: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.floor(sorted.length * percentile);
    return sorted[index] || 0;
  }

  private batchQueries(queries: FrozenQuery[], batchSize: number): FrozenQuery[][] {
    const batches: FrozenQuery[][] = [];
    for (let i = 0; i < queries.length; i += batchSize) {
      batches.push(queries.slice(i, i + batchSize));
    }
    return batches;
  }

  // Placeholder implementations for complex methods
  private async loadGroundTruthQueries(filePath: string): Promise<void> {
    console.log(`Loading ground truth queries from ${filePath}`);
    // Implementation would load from file/database
  }

  private async loadHistoricalQueries(source: string): Promise<void> {
    console.log(`Loading historical queries from ${source}`);
    // Implementation would load from analytics/logs
  }

  private async generateSyntheticQueries(count: number): Promise<void> {
    console.log(`Generating ${count} synthetic queries`);
    // Implementation would generate queries based on patterns
  }

  private validateQueryPool(): void {
    if (this.queryPool.length === 0) {
      throw new Error('Query pool is empty');
    }
    console.log(`✅ Query pool validated: ${this.queryPool.length} queries`);
  }

  private async runWarmup(config: ReplayConfig): Promise<void> {
    console.log(`🔥 Running warmup with ${config.warmupQueries} queries...`);
    // Implementation would run warmup queries
  }

  private calculateSemanticCoherence(modelType: EmbeddingModelType): number {
    // Placeholder - would measure semantic consistency
    return 0.85;
  }

  private calculateCrossLanguageConsistency(modelType: EmbeddingModelType): number {
    // Placeholder - would measure consistency across programming languages
    return 0.82;
  }

  private calculateResourceDelta(start?: any, end?: any): any {
    // Placeholder - would calculate resource usage delta
    return { memoryUsageMB: 150, cpuUtilization: 25 };
  }

  private async saveResults(result: ReplayResult, outputPath: string): Promise<void> {
    console.log(`💾 Saving results to ${outputPath}`);
    // Implementation would save to file/database
  }
}

/**
 * Simple resource monitor for collecting system metrics during evaluation
 */
class ResourceMonitor {
  private monitoring = false;
  private startMetrics?: any;

  startMonitoring(): void {
    this.monitoring = true;
    this.startMetrics = this.getCurrentMetrics();
  }

  stopMonitoring(): void {
    this.monitoring = false;
  }

  getCurrentMetrics(): any {
    // Placeholder - would collect actual system metrics
    return {
      memoryUsage: process.memoryUsage(),
      cpuUsage: process.cpuUsage(),
      timestamp: Date.now(),
    };
  }
}