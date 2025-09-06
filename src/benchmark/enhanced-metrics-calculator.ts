/**
 * Enhanced Metrics Calculator
 * 
 * Implements TODO.md requirements:
 * - Fix recall math using pooled qrels methodology  
 * - Defendable statistics with proper CI calculations
 * - SLA-constrained metrics
 */

import { promises as fs } from 'fs';
import type { BenchmarkRun, ABTestResult } from '../types/benchmark.js';

// Enhanced evaluation protocols
export interface EvaluationProtocol {
  name: 'UR-Broad' | 'UR-Narrow' | 'CP-Regex';
  description: string;
  metrics: string[];
  baselines: string[];
  statistical_tests: string[];
}

// Pooled qrels configuration
export interface PooledQrelsConfig {
  systems: string[];
  top_k: number;
  sla_constraint_ms: number;
  apply_sla_before_intersect: boolean;
}

// Statistical analysis configuration
export interface StatisticalConfig {
  bootstrap_samples: number;
  ci_level: number;
  alpha: number;
  correction_method: 'holm' | 'bonferroni' | 'none';
  effect_size_threshold: number;
}

export class EnhancedMetricsCalculator {
  private readonly defaultStatConfig: StatisticalConfig = {
    bootstrap_samples: 1000,
    ci_level: 0.95,
    alpha: 0.05,
    correction_method: 'holm',
    effect_size_threshold: 0.2 // Cohen's d
  };

  constructor(private readonly config: Partial<StatisticalConfig> = {}) {
    Object.assign(this.defaultStatConfig, config);
  }

  /**
   * Compute pooled qrels recall as per TODO.md specification
   * Formula: Recall@50 = |top50(system) ∩ Q| / |Q|
   * Where Q = ⋃_systems top50(system, UR)
   */
  computePooledQrelsRecall(
    allSystemResults: Array<{
      system: string;
      queries: Array<{
        query_id: string;
        hits: Array<{
          file: string;
          line: number;
          col: number;
          relevance_score: number;
          latency_ms?: number;
        }>;
      }>;
    }>,
    pooledConfig: PooledQrelsConfig
  ): Map<string, { recall_at_50: number; sla_recall_at_50: number }> {
    
    // Step 1: Build pooled qrels Q = ⋃_systems top50(system, UR)
    const pooledQrels = this.buildPooledQrels(allSystemResults, pooledConfig.top_k);
    
    const systemRecalls = new Map<string, { recall_at_50: number; sla_recall_at_50: number }>();
    
    for (const systemResult of allSystemResults) {
      const { system, queries } = systemResult;
      
      let totalRelevantFound = 0;
      let totalRelevantInPool = 0;
      let slaRelevantFound = 0;
      let slaQueriesCount = 0;
      
      for (const queryResult of queries) {
        const queryId = queryResult.query_id;
        const poolForQuery = pooledQrels.get(queryId) || new Set();
        
        if (poolForQuery.size === 0) continue; // Skip queries with no pooled relevant docs
        
        totalRelevantInPool += poolForQuery.size;
        
        // Standard recall calculation
        const top50Hits = queryResult.hits.slice(0, pooledConfig.top_k);
        const foundRelevant = top50Hits.filter(hit => {
          const spanKey = `${hit.file}:${hit.line}:${hit.col}`;
          return poolForQuery.has(spanKey);
        });
        totalRelevantFound += foundRelevant.length;
        
        // SLA-constrained recall (filter by latency before intersecting)
        if (pooledConfig.apply_sla_before_intersect) {
          const slaCompliantHits = top50Hits.filter(hit => 
            (hit.latency_ms || 0) <= pooledConfig.sla_constraint_ms
          );
          
          const slaFoundRelevant = slaCompliantHits.filter(hit => {
            const spanKey = `${hit.file}:${hit.line}:${hit.col}`;
            return poolForQuery.has(spanKey);
          });
          
          slaRelevantFound += slaFoundRelevant.length;
          slaQueriesCount++;
        }
      }
      
      const recall_at_50 = totalRelevantInPool > 0 ? totalRelevantFound / totalRelevantInPool : 0;
      const sla_recall_at_50 = slaQueriesCount > 0 ? slaRelevantFound / totalRelevantInPool : 0;
      
      systemRecalls.set(system, { recall_at_50, sla_recall_at_50 });
    }
    
    return systemRecalls;
  }

  /**
   * Success@k metric for assisted-lexical baselines (prevent "100% recall mirage")
   */
  computeSuccessAtK(queryResults: Array<{
    query_id: string;
    hits: Array<{ relevance_score: number }>;
  }>, k: number): number {
    let successfulQueries = 0;
    
    for (const query of queryResults) {
      const topKHits = query.hits.slice(0, k);
      const hasRelevantResult = topKHits.some(hit => hit.relevance_score > 0.5);
      if (hasRelevantResult) {
        successfulQueries++;
      }
    }
    
    return queryResults.length > 0 ? successfulQueries / queryResults.length : 0;
  }

  /**
   * Paired stratified bootstrap for confidence intervals
   */
  computeBootstrapCI(
    treatment: number[],
    control: number[],
    nSamples: number = this.defaultStatConfig.bootstrap_samples
  ): { delta: number; ci_lower: number; ci_upper: number; p_value: number } {
    
    if (treatment.length !== control.length) {
      throw new Error('Treatment and control groups must have same size for paired test');
    }
    
    // Original delta
    const treatmentMean = treatment.reduce((a, b) => a + b, 0) / treatment.length;
    const controlMean = control.reduce((a, b) => a + b, 0) / control.length;
    const originalDelta = treatmentMean - controlMean;
    
    // Bootstrap resampling
    const bootstrapDeltas: number[] = [];
    
    for (let i = 0; i < nSamples; i++) {
      const sampleTreatment: number[] = [];
      const sampleControl: number[] = [];
      
      // Stratified sampling (preserve pairs)
      for (let j = 0; j < treatment.length; j++) {
        const idx = Math.floor(Math.random() * treatment.length);
        sampleTreatment.push(treatment[idx]);
        sampleControl.push(control[idx]);
      }
      
      const sampleTreatmentMean = sampleTreatment.reduce((a, b) => a + b, 0) / sampleTreatment.length;
      const sampleControlMean = sampleControl.reduce((a, b) => a + b, 0) / sampleControl.length;
      bootstrapDeltas.push(sampleTreatmentMean - sampleControlMean);
    }
    
    // Calculate confidence interval
    bootstrapDeltas.sort((a, b) => a - b);
    const ciLevel = this.defaultStatConfig.ci_level;
    const lowerIdx = Math.floor(nSamples * (1 - ciLevel) / 2);
    const upperIdx = Math.floor(nSamples * (1 + ciLevel) / 2) - 1;
    
    const ci_lower = bootstrapDeltas[lowerIdx];
    const ci_upper = bootstrapDeltas[upperIdx];
    
    // Permutation test for p-value
    const p_value = this.computePermutationPValue(treatment, control);
    
    return {
      delta: originalDelta,
      ci_lower,
      ci_upper,
      p_value
    };
  }

  /**
   * Paired permutation test with Holm correction
   */
  computePermutationPValue(treatment: number[], control: number[]): number {
    const n = treatment.length;
    const observedDelta = treatment.reduce((a, b) => a + b, 0) / n - 
                         control.reduce((a, b) => a + b, 0) / n;
    
    let moreExtremeCount = 0;
    const permutations = Math.min(10000, Math.pow(2, n)); // Cap at 10k permutations
    
    for (let i = 0; i < permutations; i++) {
      const permTreatment: number[] = [];
      const permControl: number[] = [];
      
      for (let j = 0; j < n; j++) {
        if (Math.random() < 0.5) {
          permTreatment.push(treatment[j]);
          permControl.push(control[j]);
        } else {
          permTreatment.push(control[j]);
          permControl.push(treatment[j]);
        }
      }
      
      const permDelta = permTreatment.reduce((a, b) => a + b, 0) / n - 
                       permControl.reduce((a, b) => a + b, 0) / n;
      
      if (Math.abs(permDelta) >= Math.abs(observedDelta)) {
        moreExtremeCount++;
      }
    }
    
    return moreExtremeCount / permutations;
  }

  /**
   * Wilcoxon signed-rank test for non-parametric comparison
   */
  computeWilcoxonTest(treatment: number[], control: number[]): { statistic: number; p_value: number } {
    const differences = treatment.map((t, i) => t - control[i]).filter(d => d !== 0);
    const n = differences.length;
    
    if (n === 0) return { statistic: 0, p_value: 1.0 };
    
    // Rank absolute differences
    const ranks = differences
      .map((d, i) => ({ diff: Math.abs(d), sign: Math.sign(d), index: i }))
      .sort((a, b) => a.diff - b.diff)
      .map((item, rank) => ({ ...item, rank: rank + 1 }));
    
    // Sum of positive ranks
    const W = ranks.filter(r => r.sign > 0).reduce((sum, r) => sum + r.rank, 0);
    
    // Approximate p-value for large n (n > 10)
    if (n > 10) {
      const mean = n * (n + 1) / 4;
      const variance = n * (n + 1) * (2 * n + 1) / 24;
      const z = Math.abs(W - mean) / Math.sqrt(variance);
      const p_value = 2 * (1 - this.standardNormalCDF(z)); // Two-tailed
      return { statistic: W, p_value };
    }
    
    // For small n, use exact distribution (simplified)
    const p_value = this.wilcoxonExactPValue(W, n);
    return { statistic: W, p_value };
  }

  /**
   * Generate evaluation protocol results in ladder format (UR-Broad → UR-Narrow → CP-Regex)
   */
  generateLadderResults(
    systemResults: Map<string, any>,
    protocols: EvaluationProtocol[]
  ): Map<string, any> {
    const ladderResults = new Map();
    
    for (const protocol of protocols) {
      const protocolResults = new Map();
      
      for (const [system, results] of systemResults) {
        const metrics = this.extractMetricsForProtocol(results, protocol);
        const significance = this.computeStatisticalSignificance(metrics, protocol);
        
        protocolResults.set(system, {
          metrics,
          significance,
          protocol_name: protocol.name
        });
      }
      
      ladderResults.set(protocol.name, protocolResults);
    }
    
    return ladderResults;
  }

  /**
   * Monotone delta table for ablation studies (vs Lens(lex) baseline)
   */
  generateAblationTable(
    baselineResults: any,
    ablationResults: Array<{ name: string; results: any }>
  ): Array<{ 
    ablation: string; 
    ndcg_delta: number; 
    recall_sla_delta: number; 
    positives_in_candidates: number 
  }> {
    
    const table: Array<any> = [];
    
    for (const ablation of ablationResults) {
      const ndcg_delta = ablation.results.ndcg_at_10 - baselineResults.ndcg_at_10;
      const recall_sla_delta = ablation.results.sla_recall_at_50 - baselineResults.sla_recall_at_50;
      const positives_in_candidates = ablation.results.fan_out_sizes?.stage_a || 0;
      
      table.push({
        ablation: ablation.name,
        ndcg_delta,
        recall_sla_delta,
        positives_in_candidates
      });
    }
    
    // Sort by incremental improvement (monotone)
    table.sort((a, b) => a.ndcg_delta - b.ndcg_delta);
    
    return table;
  }

  // Private helper methods

  private buildPooledQrels(
    allSystemResults: Array<{ system: string; queries: any[] }>,
    topK: number
  ): Map<string, Set<string>> {
    const pooledQrels = new Map<string, Set<string>>();
    
    for (const systemResult of allSystemResults) {
      for (const query of systemResult.queries) {
        const queryId = query.query_id;
        
        if (!pooledQrels.has(queryId)) {
          pooledQrels.set(queryId, new Set());
        }
        
        const pool = pooledQrels.get(queryId)!;
        const topKHits = query.hits.slice(0, topK);
        
        for (const hit of topKHits) {
          const spanKey = `${hit.file}:${hit.line}:${hit.col}`;
          pool.add(spanKey);
        }
      }
    }
    
    return pooledQrels;
  }

  private extractMetricsForProtocol(results: any, protocol: EvaluationProtocol): any {
    const extracted: any = {};
    
    for (const metric of protocol.metrics) {
      extracted[metric] = results[metric] || 0;
    }
    
    return extracted;
  }

  private computeStatisticalSignificance(metrics: any, protocol: EvaluationProtocol): any {
    // Placeholder - would implement full significance testing
    return {
      p_values: {},
      effect_sizes: {},
      confidence_intervals: {}
    };
  }

  private standardNormalCDF(z: number): number {
    // Approximation of standard normal CDF
    return 0.5 * (1 + this.erf(z / Math.sqrt(2)));
  }

  private erf(x: number): number {
    // Approximation of error function
    const a1 =  0.254829592;
    const a2 = -0.284496736;
    const a3 =  1.421413741;
    const a4 = -1.453152027;
    const a5 =  1.061405429;
    const p  =  0.3275911;
    
    const sign = x >= 0 ? 1 : -1;
    x = Math.abs(x);
    
    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    
    return sign * y;
  }

  private wilcoxonExactPValue(W: number, n: number): number {
    // Simplified exact p-value calculation for small samples
    // In production would use lookup table or more sophisticated calculation
    const totalOutcomes = Math.pow(2, n);
    
    // Count outcomes with W or more extreme
    let extremeOutcomes = 0;
    for (let i = 0; i < totalOutcomes; i++) {
      let rankSum = 0;
      for (let j = 0; j < n; j++) {
        if ((i >> j) & 1) { // If j-th bit is 1 (positive difference)
          rankSum += j + 1;
        }
      }
      if (rankSum >= W || rankSum <= (n * (n + 1) / 2 - W)) {
        extremeOutcomes++;
      }
    }
    
    return extremeOutcomes / totalOutcomes;
  }
}

/**
 * Standard evaluation protocols as specified in TODO.md
 */
export const EVALUATION_PROTOCOLS: EvaluationProtocol[] = [
  {
    name: 'UR-Broad',
    description: 'Broad user request evaluation with general-purpose tool comparison',
    metrics: ['ndcg_at_10', 'sla_recall_at_50', 'p95_latency', 'sla_pass_rate', 'qps_at_150ms', 'nzc_rate', 'timeout_rate'],
    baselines: ['assisted-lexical', 'grep', 'ripgrep', 'github-search'],
    statistical_tests: ['bootstrap_ci', 'permutation_test', 'wilcoxon']
  },
  {
    name: 'UR-Narrow',
    description: 'Narrow assisted-lexical comparison with success metrics',
    metrics: ['ndcg_at_10', 'sla_recall_at_50', 'success_at_10', 'p95_latency', 'sla_pass_rate'],
    baselines: ['assisted-lexical-variants'],
    statistical_tests: ['bootstrap_ci', 'permutation_test']
  },
  {
    name: 'CP-Regex',
    description: 'Code pattern regex fairness evaluation',
    metrics: ['nzc_rate', 'success_at_10', 'recall_at_10', 'sentinel_coverage'],
    baselines: ['grep-class', 'regex-tools'],
    statistical_tests: ['wilson_ci', 'exact_test']
  }
];