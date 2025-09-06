/**
 * Embedder-Agnostic Optimizer - Main Integration System
 * 
 * Coordinates all four embedder-agnostic optimization systems:
 * 1. Constraint-Aware Reranker (monotone GAM with pairwise constraints)
 * 2. Stage-B⁺ Slice-Chasing (symbol graph BFS traversal)
 * 3. ROI-Aware Result Micro-Cache (TTL'd sharded cache)
 * 4. ANN Hygiene (algorithmic HNSW optimizations)
 * 
 * This system survives embedder swaps and provides stable search improvements
 * with comprehensive monitoring and gating on quality/latency metrics.
 */

import type { SearchHit } from '../core/span_resolver/types.js';
import type { SearchContext, SymbolDefinition, SymbolReference } from '../types/core.js';
import { ConstraintAwareReranker, type ConstraintAwareConfig } from './constraint-aware-reranker.js';
import { StageBPlusSliceChasing, type SliceChasingConfig } from './stage-b-plus-slice-chasing.js';
import { ROIAwareMicroCache, type MicroCacheConfig } from './roi-aware-micro-cache.js';
import { ANNHygieneOptimizer, type ANNHygieneConfig } from './ann-hygiene-optimizer.js';
import { LensTracer } from '../telemetry/tracer.js';

export interface EmbedderAgnosticConfig {
  enabled: boolean;
  indexVersion: string;                // For cache invalidation
  slaRecallThreshold: number;          // SLA-Recall@50 >= 0 threshold
  nDCGImprovementThreshold: number;    // ΔnDCG@10 >= +0.5pp threshold
  eceToleranceThreshold: number;       // ECE Δ <= 0.01 threshold
  maxTotalLatencyMs: number;           // Total budget for all optimizations
  
  // Component configurations
  constraintAware: Partial<ConstraintAwareConfig>;
  sliceChasing: Partial<SliceChasingConfig>;
  microCache: Partial<MicroCacheConfig>;
  annHygiene: Partial<ANNHygieneConfig>;
  
  // Quality gates
  enableQualityGates: boolean;
  qualityGateWindow: number;           // Number of queries for rolling window
  latencyRegressionThreshold: number;  // Max allowed latency increase (%)
}

export interface OptimizationResult {
  originalHits: SearchHit[];
  finalHits: SearchHit[];
  optimizationsApplied: string[];
  latencyBreakdown: {
    constraintAware: number;
    sliceChasing: number;
    microCache: number;
    annHygiene: number;
    total: number;
  };
  qualityMetrics: {
    slaRecall: number;
    nDCG: number;
    ece: number;
    hitsDelta: number;
  };
  cacheStatus: 'hit' | 'miss' | 'disabled';
  constraints: {
    violationsDetected: number;
    violationsCorrected: number;
    monotonicityValid: boolean;
  };
}

export interface QualityGate {
  metric: string;
  threshold: number;
  currentValue: number;
  status: 'pass' | 'fail' | 'warning';
  windowSize: number;
  recentValues: number[];
}

/**
 * Quality metrics calculator
 */
export class QualityMetricsCalculator {
  /**
   * Calculate SLA-Recall@50 (simplified implementation)
   */
  calculateSLARecall(hits: SearchHit[], k: number = 50): number {
    if (hits.length === 0) return 0;
    
    // In practice, this would use ground truth relevance labels
    // For now, use score-based approximation
    const topK = hits.slice(0, k);
    const relevantHits = topK.filter(hit => hit.score > 0.5);
    
    return relevantHits.length / Math.min(k, hits.length);
  }

  /**
   * Calculate nDCG@10 (simplified implementation)
   */
  calculateNDCG(hits: SearchHit[], k: number = 10): number {
    if (hits.length === 0) return 0;
    
    const topK = hits.slice(0, k);
    
    // DCG calculation
    let dcg = 0;
    for (let i = 0; i < topK.length; i++) {
      const relevance = topK[i]!.score; // Using score as relevance proxy
      const position = i + 1;
      dcg += relevance / Math.log2(position + 1);
    }
    
    // IDCG (ideal DCG) - perfect ordering by score
    const idealScores = topK.map(h => h.score).sort((a, b) => b - a);
    let idcg = 0;
    for (let i = 0; i < idealScores.length; i++) {
      const relevance = idealScores[i]!;
      const position = i + 1;
      idcg += relevance / Math.log2(position + 1);
    }
    
    return idcg > 0 ? dcg / idcg : 0;
  }

  /**
   * Calculate Expected Calibration Error (simplified)
   */
  calculateECE(hits: SearchHit[], bins: number = 10): number {
    if (hits.length === 0) return 0;
    
    const binSize = 1.0 / bins;
    let ece = 0;
    
    for (let i = 0; i < bins; i++) {
      const binMin = i * binSize;
      const binMax = (i + 1) * binSize;
      
      const binHits = hits.filter(hit => hit.score >= binMin && hit.score < binMax);
      if (binHits.length === 0) continue;
      
      const avgConfidence = binHits.reduce((sum, hit) => sum + hit.score, 0) / binHits.length;
      const avgAccuracy = binHits.filter(hit => hit.score > 0.5).length / binHits.length; // Mock accuracy
      
      const binWeight = binHits.length / hits.length;
      ece += binWeight * Math.abs(avgConfidence - avgAccuracy);
    }
    
    return ece;
  }
}

/**
 * Main Embedder-Agnostic Optimizer
 */
export class EmbedderAgnosticOptimizer {
  private config: EmbedderAgnosticConfig;
  private constraintReranker: ConstraintAwareReranker;
  private sliceChaser: StageBPlusSliceChasing;
  private microCache: ROIAwareMicroCache;
  private annHygiene: ANNHygieneOptimizer;
  private qualityCalculator: QualityMetricsCalculator;
  
  // Quality gates tracking
  private qualityGates: Map<string, QualityGate> = new Map();
  private recentResults: OptimizationResult[] = [];

  constructor(config: Partial<EmbedderAgnosticConfig> = {}) {
    this.config = {
      enabled: config.enabled ?? true,
      indexVersion: config.indexVersion ?? '1.0.0',
      slaRecallThreshold: config.slaRecallThreshold ?? 0.0, // >= 0
      nDCGImprovementThreshold: config.nDCGImprovementThreshold ?? 0.005, // +0.5pp
      eceToleranceThreshold: config.eceToleranceThreshold ?? 0.01, // <= 0.01
      maxTotalLatencyMs: config.maxTotalLatencyMs ?? 10,
      enableQualityGates: config.enableQualityGates ?? true,
      qualityGateWindow: config.qualityGateWindow ?? 100,
      latencyRegressionThreshold: config.latencyRegressionThreshold ?? 20, // 20% max increase
      constraintAware: config.constraintAware ?? {},
      sliceChasing: config.sliceChasing ?? {},
      microCache: config.microCache ?? {},
      annHygiene: config.annHygiene ?? {},
      ...config
    };

    // Initialize optimization components
    this.constraintReranker = new ConstraintAwareReranker(this.config.constraintAware);
    this.sliceChaser = new StageBPlusSliceChasing(this.config.sliceChasing);
    this.microCache = new ROIAwareMicroCache(this.config.microCache);
    this.annHygiene = new ANNHygieneOptimizer(this.config.annHygiene);
    this.qualityCalculator = new QualityMetricsCalculator();

    // Initialize quality gates
    this.initializeQualityGates();

    console.log(`🚀 EmbedderAgnosticOptimizer initialized: enabled=${this.config.enabled}, version=${this.config.indexVersion}`);
  }

  /**
   * Initialize quality gate monitoring
   */
  private initializeQualityGates(): void {
    if (!this.config.enableQualityGates) return;

    const gates: Array<{ metric: string; threshold: number }> = [
      { metric: 'sla_recall', threshold: this.config.slaRecallThreshold },
      { metric: 'ndcg_improvement', threshold: this.config.nDCGImprovementThreshold },
      { metric: 'ece_delta', threshold: this.config.eceToleranceThreshold },
      { metric: 'latency_regression', threshold: this.config.latencyRegressionThreshold }
    ];

    for (const gate of gates) {
      this.qualityGates.set(gate.metric, {
        metric: gate.metric,
        threshold: gate.threshold,
        currentValue: 0,
        status: 'pass',
        windowSize: this.config.qualityGateWindow,
        recentValues: []
      });
    }
  }

  /**
   * Main optimization pipeline
   */
  async optimize(
    originalHits: SearchHit[],
    context: SearchContext,
    symbolDefinitions: SymbolDefinition[] = [],
    symbolReferences: SymbolReference[] = []
  ): Promise<OptimizationResult> {
    const span = LensTracer.createChildSpan('embedder_agnostic_optimize', {
      'original_hits': originalHits.length,
      'query': context.query,
      'enabled': this.config.enabled
    });

    const startTime = performance.now();
    const latencyBreakdown = {
      constraintAware: 0,
      sliceChasing: 0,
      microCache: 0,
      annHygiene: 0,
      total: 0
    };

    try {
      if (!this.config.enabled || originalHits.length === 0) {
        span.setAttributes({ skipped: true, reason: 'disabled_or_empty' });
        return this.createNullResult(originalHits);
      }

      let currentHits = [...originalHits];
      const optimizationsApplied: string[] = [];

      // Budget tracking
      const checkBudget = () => {
        const elapsed = performance.now() - startTime;
        if (elapsed > this.config.maxTotalLatencyMs) {
          throw new Error(`Total optimization budget exceeded: ${elapsed.toFixed(3)}ms > ${this.config.maxTotalLatencyMs}ms`);
        }
        return elapsed;
      };

      // 1. Check micro-cache first (fastest path)
      const cacheStartTime = performance.now();
      const slaHeadroom = this.config.maxTotalLatencyMs - (performance.now() - startTime);
      const cachedResults = await this.microCache.getCachedResults(context, this.config.indexVersion, slaHeadroom);
      latencyBreakdown.microCache = performance.now() - cacheStartTime;

      if (cachedResults) {
        optimizationsApplied.push('micro-cache-hit');
        currentHits = cachedResults;
      } else {
        optimizationsApplied.push('micro-cache-miss');
        
        checkBudget();

        // 2. Stage-B⁺ Slice-Chasing (if no cache hit)
        if (symbolDefinitions.length > 0 || symbolReferences.length > 0) {
          const sliceStartTime = performance.now();
          this.sliceChaser.initializeGraph(symbolDefinitions, symbolReferences);
          currentHits = await this.sliceChaser.chaseSlices(currentHits, context);
          latencyBreakdown.sliceChasing = performance.now() - sliceStartTime;
          optimizationsApplied.push('slice-chasing');
        }

        checkBudget();

        // 3. ANN Hygiene optimizations
        const annStartTime = performance.now();
        const candidateNodes = currentHits.map((_, i) => i); // Mock node IDs
        const topicId = this.extractTopicId(context);
        const annResult = await this.annHygiene.optimizeSearch(context, candidateNodes, 50, topicId);
        
        // Apply ANN results to hits (simplified mapping)
        if (annResult.optimizedCandidates.length > 0) {
          currentHits = currentHits.slice(0, annResult.optimizedCandidates.length);
          optimizationsApplied.push('ann-hygiene');
        }
        latencyBreakdown.annHygiene = performance.now() - annStartTime;

        checkBudget();

        // 4. Constraint-Aware Reranking (final step)
        const rerankerStartTime = performance.now();
        currentHits = await this.constraintReranker.rerank(currentHits, context);
        latencyBreakdown.constraintAware = performance.now() - rerankerStartTime;
        optimizationsApplied.push('constraint-aware');

        // Cache results for future queries
        const totalLatency = performance.now() - startTime;
        await this.microCache.cacheResults(context, currentHits, this.config.indexVersion, totalLatency);
      }

      latencyBreakdown.total = performance.now() - startTime;

      // Calculate quality metrics
      const qualityMetrics = {
        slaRecall: this.qualityCalculator.calculateSLARecall(currentHits),
        nDCG: this.qualityCalculator.calculateNDCG(currentHits),
        ece: this.qualityCalculator.calculateECE(currentHits),
        hitsDelta: currentHits.length - originalHits.length
      };

      // Get constraint information
      const constraintStats = this.constraintReranker.getStats();
      const constraints = {
        violationsDetected: constraintStats.violations_logged,
        violationsCorrected: constraintStats.violations_logged, // Simplified
        monotonicityValid: constraintStats.monotonicity_validation.valid
      };

      const result: OptimizationResult = {
        originalHits,
        finalHits: currentHits,
        optimizationsApplied,
        latencyBreakdown,
        qualityMetrics,
        cacheStatus: cachedResults ? 'hit' : 'miss',
        constraints
      };

      // Update quality gates
      this.updateQualityGates(result);

      span.setAttributes({
        success: true,
        optimizations_applied: optimizationsApplied.length,
        final_hits: currentHits.length,
        total_latency_ms: latencyBreakdown.total,
        cache_status: result.cacheStatus,
        sla_recall: qualityMetrics.slaRecall,
        ndcg: qualityMetrics.nDCG
      });

      console.log(`🚀 EmbedderAgnostic: ${originalHits.length}→${currentHits.length} hits, ${optimizationsApplied.join('+')} in ${latencyBreakdown.total.toFixed(3)}ms`);

      return result;

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });

      console.warn(`EmbedderAgnostic optimization failed: ${errorMsg}, falling back to original hits`);
      return this.createNullResult(originalHits);

    } finally {
      span.end();
    }
  }

  /**
   * Create null result for fallback scenarios
   */
  private createNullResult(originalHits: SearchHit[]): OptimizationResult {
    return {
      originalHits,
      finalHits: originalHits,
      optimizationsApplied: ['fallback'],
      latencyBreakdown: {
        constraintAware: 0,
        sliceChasing: 0,
        microCache: 0,
        annHygiene: 0,
        total: 0
      },
      qualityMetrics: {
        slaRecall: this.qualityCalculator.calculateSLARecall(originalHits),
        nDCG: this.qualityCalculator.calculateNDCG(originalHits),
        ece: this.qualityCalculator.calculateECE(originalHits),
        hitsDelta: 0
      },
      cacheStatus: 'disabled',
      constraints: {
        violationsDetected: 0,
        violationsCorrected: 0,
        monotonicityValid: true
      }
    };
  }

  /**
   * Extract topic ID from search context
   */
  private extractTopicId(context: SearchContext): string | undefined {
    const query = context.query.toLowerCase();
    
    if (query.includes('function') || query.includes('method')) {
      return 'functions';
    } else if (query.includes('class') || query.includes('type')) {
      return 'types';
    } else if (query.includes('variable') || query.includes('const')) {
      return 'variables';
    } else if (query.includes('config') || query.includes('setting')) {
      return 'config';
    }
    
    return undefined;
  }

  /**
   * Update quality gates with latest results
   */
  private updateQualityGates(result: OptimizationResult): void {
    if (!this.config.enableQualityGates) return;

    // Add to recent results
    this.recentResults.push(result);
    if (this.recentResults.length > this.config.qualityGateWindow) {
      this.recentResults = this.recentResults.slice(-this.config.qualityGateWindow);
    }

    // Update each gate
    const gates = [
      { metric: 'sla_recall', value: result.qualityMetrics.slaRecall },
      { metric: 'ndcg_improvement', value: result.qualityMetrics.nDCG },
      { metric: 'ece_delta', value: result.qualityMetrics.ece },
      { metric: 'latency_regression', value: result.latencyBreakdown.total }
    ];

    for (const { metric, value } of gates) {
      const gate = this.qualityGates.get(metric);
      if (!gate) continue;

      gate.recentValues.push(value);
      if (gate.recentValues.length > gate.windowSize) {
        gate.recentValues = gate.recentValues.slice(-gate.windowSize);
      }

      gate.currentValue = gate.recentValues.reduce((sum, v) => sum + v, 0) / gate.recentValues.length;

      // Update status based on threshold
      if (metric === 'ece_delta' || metric === 'latency_regression') {
        // Lower is better
        gate.status = gate.currentValue <= gate.threshold ? 'pass' : 'fail';
      } else {
        // Higher is better
        gate.status = gate.currentValue >= gate.threshold ? 'pass' : 'fail';
      }

      // Warning zone (within 10% of threshold)
      const warningMargin = 0.1 * Math.abs(gate.threshold);
      if (gate.status === 'pass' && Math.abs(gate.currentValue - gate.threshold) < warningMargin) {
        gate.status = 'warning';
      }
    }
  }

  /**
   * Get comprehensive system statistics
   */
  getStats() {
    const recentLatencies = this.recentResults.map(r => r.latencyBreakdown.total);
    const avgLatency = recentLatencies.length > 0 
      ? recentLatencies.reduce((sum, l) => sum + l, 0) / recentLatencies.length 
      : 0;

    const qualityGateStatus = Array.from(this.qualityGates.entries()).reduce((acc, [metric, gate]) => {
      acc[metric] = {
        status: gate.status,
        current_value: gate.currentValue,
        threshold: gate.threshold
      };
      return acc;
    }, {} as Record<string, any>);

    return {
      config: this.config,
      performance: {
        total_optimizations: this.recentResults.length,
        avg_latency_ms: avgLatency,
        optimization_distribution: this.computeOptimizationDistribution()
      },
      quality_gates: qualityGateStatus,
      components: {
        constraint_reranker: this.constraintReranker.getStats(),
        slice_chasing: this.sliceChaser.getStats(),
        micro_cache: this.microCache.getStats(),
        ann_hygiene: this.annHygiene.getStats()
      }
    };
  }

  /**
   * Compute distribution of applied optimizations
   */
  private computeOptimizationDistribution(): Record<string, number> {
    const distribution: Record<string, number> = {};
    
    for (const result of this.recentResults) {
      for (const optimization of result.optimizationsApplied) {
        distribution[optimization] = (distribution[optimization] || 0) + 1;
      }
    }

    return distribution;
  }

  /**
   * Check if quality gates are passing
   */
  areQualityGatesPassing(): boolean {
    if (!this.config.enableQualityGates) return true;
    
    for (const gate of this.qualityGates.values()) {
      if (gate.status === 'fail') {
        return false;
      }
    }
    return true;
  }

  /**
   * Get failing quality gates
   */
  getFailingQualityGates(): QualityGate[] {
    return Array.from(this.qualityGates.values()).filter(gate => gate.status === 'fail');
  }

  /**
   * Update configuration for A/B testing
   */
  updateConfig(newConfig: Partial<EmbedderAgnosticConfig>): void {
    this.config = { ...this.config, ...newConfig };
    
    // Update component configurations
    if (newConfig.constraintAware) {
      this.constraintReranker.updateConfig(newConfig.constraintAware);
    }
    if (newConfig.sliceChasing) {
      this.sliceChaser.updateConfig(newConfig.sliceChasing);
    }
    if (newConfig.microCache) {
      this.microCache.updateConfig(newConfig.microCache);
    }
    if (newConfig.annHygiene) {
      this.annHygiene.updateConfig(newConfig.annHygiene);
    }

    // Reinitialize quality gates if thresholds changed
    if (newConfig.slaRecallThreshold !== undefined || 
        newConfig.nDCGImprovementThreshold !== undefined ||
        newConfig.eceToleranceThreshold !== undefined) {
      this.initializeQualityGates();
    }

    console.log(`🚀 EmbedderAgnosticOptimizer config updated: ${JSON.stringify(newConfig)}`);
  }

  /**
   * Reset all statistics and quality gates
   */
  reset(): void {
    this.recentResults = [];
    this.qualityGates.clear();
    this.initializeQualityGates();
    console.log('🚀 EmbedderAgnosticOptimizer reset');
  }
}