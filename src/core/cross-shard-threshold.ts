/**
 * Cross-Shard Thresholded Top-K Implementation
 * Uses global Threshold Algorithm (TA/NRA) over shard-local upper bounds
 * Reduces candidate movement and tail I/O by stopping when τ ≤ score_k
 */

import { LensTracer } from '../telemetry/tracer.js';
import type { SearchContext, Candidate } from '../types/core.js';

export interface ShardUpperBound {
  shardId: string;
  termUpperBounds: Map<string, number>; // term -> max_score
  nextCandidateScore?: number;
}

export interface ThresholdState {
  tau: number; // Global threshold τ = sum_shards UB_next
  currentK: Candidate[];
  thresholdHistory: Array<{ query_id: string; tau_curve: number[]; stopped_early: boolean }>;
}

export interface CrossShardConfig {
  enabled: boolean;
  trafficPercent: number; // Start at 25%
  maxShards: number;
  upperBoundCacheMs: number; // Cache UB oracles
  logTauCurves: boolean;
}

export class CrossShardThresholdOptimizer {
  private config: CrossShardConfig;
  private shardBounds: Map<string, ShardUpperBound> = new Map();
  private thresholdStates: Map<string, ThresholdState> = new Map();
  private queryCounter = 0;

  constructor(config: Partial<CrossShardConfig> = {}) {
    this.config = {
      enabled: false, // Start disabled for gradual rollout
      trafficPercent: 25, // 25% traffic initially
      maxShards: 16,
      upperBoundCacheMs: 5000, // 5s cache for UB oracles
      logTauCurves: true,
      ...config,
    };
  }

  /**
   * Initialize shard upper bounds from impact bucket heads
   */
  async updateShardBounds(shardId: string, termBounds: Map<string, number>): Promise<void> {
    const span = LensTracer.createChildSpan('update_shard_bounds');
    
    try {
      this.shardBounds.set(shardId, {
        shardId,
        termUpperBounds: new Map(termBounds),
        nextCandidateScore: Math.max(...termBounds.values()),
      });

      span.setAttributes({
        success: true,
        shard_id: shardId,
        terms_count: termBounds.size,
        max_bound: Math.max(...termBounds.values()),
      });
    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: (error as Error).message });
    } finally {
      span.end();
    }
  }

  /**
   * Execute threshold algorithm for cross-shard query optimization
   */
  async executeThresholdAlgorithm(
    ctx: SearchContext,
    queryTerms: string[],
    targetK: number
  ): Promise<{
    shouldContinue: boolean;
    threshold: number;
    candidateEstimate: number;
    stoppedEarly: boolean;
  }> {
    const span = LensTracer.createChildSpan('execute_threshold_algorithm');
    const queryId = `${ctx.repo_sha}_${Date.now()}_${++this.queryCounter}`;
    
    try {
      // Check if we should apply optimization based on traffic percentage
      if (!this.shouldApplyOptimization()) {
        return { shouldContinue: true, threshold: 0, candidateEstimate: 0, stoppedEarly: false };
      }

      // Calculate global threshold τ = sum_shards UB_next
      const tau = this.calculateGlobalThreshold(queryTerms);
      
      // Get current top-K candidates estimate
      const currentState = this.thresholdStates.get(queryId) || {
        tau,
        currentK: [],
        thresholdHistory: [],
      };

      // Update threshold state
      currentState.tau = tau;
      this.thresholdStates.set(queryId, currentState);

      // Log tau curve for performance analysis
      if (this.config.logTauCurves) {
        this.logTauCurve(queryId, tau);
      }

      // Calculate score_k (k-th highest score estimate)
      const scoreK = this.estimateScoreK(targetK);

      // Decision: Stop when τ ≤ score_k
      const shouldStop = tau <= scoreK && scoreK > 0;
      
      span.setAttributes({
        success: true,
        query_id: queryId,
        tau_threshold: tau,
        score_k: scoreK,
        target_k: targetK,
        should_stop: shouldStop,
        terms_count: queryTerms.length,
        active_shards: this.shardBounds.size,
      });

      return {
        shouldContinue: !shouldStop,
        threshold: tau,
        candidateEstimate: scoreK,
        stoppedEarly: shouldStop,
      };

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: (error as Error).message });
      
      // Fallback: continue with normal execution
      return { shouldContinue: true, threshold: 0, candidateEstimate: 0, stoppedEarly: false };
    } finally {
      span.end();
    }
  }

  /**
   * Update configuration for A/B testing
   */
  updateConfig(newConfig: Partial<CrossShardConfig>): void {
    this.config = { ...this.config, ...newConfig };
    console.log(`🎯 Cross-Shard Threshold Optimizer config updated:`, this.config);
  }

  /**
   * Get performance metrics for monitoring
   */
  getMetrics(): {
    enabled: boolean;
    traffic_percent: number;
    active_shards: number;
    avg_tau: number;
    early_stop_rate: number;
    cache_hit_rate: number;
  } {
    const recentStates = Array.from(this.thresholdStates.values()).slice(-100);
    const avgTau = recentStates.length > 0 
      ? recentStates.reduce((sum, state) => sum + state.tau, 0) / recentStates.length 
      : 0;

    const earlyStops = recentStates.filter(state => 
      state.thresholdHistory.some(h => h.stopped_early)
    ).length;
    const earlyStopRate = recentStates.length > 0 ? earlyStops / recentStates.length : 0;

    return {
      enabled: this.config.enabled,
      traffic_percent: this.config.trafficPercent,
      active_shards: this.shardBounds.size,
      avg_tau: avgTau,
      early_stop_rate: earlyStopRate,
      cache_hit_rate: this.calculateCacheHitRate(),
    };
  }

  /**
   * Clean up old threshold states to prevent memory leaks
   */
  cleanup(): void {
    const cutoff = Date.now() - (10 * 60 * 1000); // 10 minutes
    
    for (const [queryId, state] of this.thresholdStates.entries()) {
      const isOld = state.thresholdHistory.length === 0 || 
        state.thresholdHistory.every(h => {
          const queryTime = parseInt(queryId.split('_')[1] || '0', 10);
          return queryTime < cutoff;
        });
      
      if (isOld) {
        this.thresholdStates.delete(queryId);
      }
    }
  }

  private shouldApplyOptimization(): boolean {
    if (!this.config.enabled) return false;
    
    // Apply to percentage of traffic
    const rand = Math.random() * 100;
    return rand < this.config.trafficPercent;
  }

  private calculateGlobalThreshold(queryTerms: string[]): number {
    let tau = 0;
    
    for (const [shardId, bound] of this.shardBounds.entries()) {
      let shardContribution = 0;
      
      // For each term, get the upper bound from this shard
      for (const term of queryTerms) {
        const termBound = bound.termUpperBounds.get(term) || 0;
        shardContribution += termBound;
      }
      
      // Use next candidate score if available
      if (bound.nextCandidateScore !== undefined) {
        shardContribution = Math.max(shardContribution, bound.nextCandidateScore);
      }
      
      tau += shardContribution;
    }
    
    return tau;
  }

  private estimateScoreK(targetK: number): number {
    // Simple estimation based on recent query history
    // In production, this would use more sophisticated modeling
    const recentStates = Array.from(this.thresholdStates.values()).slice(-50);
    
    if (recentStates.length === 0) return 0;
    
    // Calculate average k-th percentile score
    const scores = recentStates
      .flatMap(state => state.currentK.map(c => c.score))
      .sort((a, b) => b - a);
    
    if (scores.length < targetK) return 0;
    
    return scores[targetK - 1] || 0;
  }

  private logTauCurve(queryId: string, tau: number): void {
    const state = this.thresholdStates.get(queryId);
    if (!state) return;
    
    if (state.thresholdHistory.length === 0) {
      state.thresholdHistory.push({
        query_id: queryId,
        tau_curve: [tau],
        stopped_early: false,
      });
    } else {
      const last = state.thresholdHistory[state.thresholdHistory.length - 1];
      last.tau_curve.push(tau);
    }
  }

  private calculateCacheHitRate(): number {
    // In production, track actual cache hits/misses
    // For now, return estimated rate based on bound update frequency
    const now = Date.now();
    const recentUpdates = Array.from(this.shardBounds.values()).filter(bound => {
      // Assume bounds were updated recently if they have data
      return bound.termUpperBounds.size > 0;
    }).length;
    
    const totalShards = this.shardBounds.size;
    return totalShards > 0 ? recentUpdates / totalShards : 0;
  }
}

/**
 * Factory for creating cross-shard threshold optimizer
 */
export function createCrossShardOptimizer(config?: Partial<CrossShardConfig>): CrossShardThresholdOptimizer {
  return new CrossShardThresholdOptimizer(config);
}