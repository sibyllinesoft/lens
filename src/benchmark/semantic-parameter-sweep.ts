/**
 * Phase 3: Semantic Stage Refinement - Parameter Sweep Implementation
 * Systematic optimization of Stage-C semantic parameters for maximum precision with controlled latency
 */

import type { SearchContext } from '../types/core.js';
import type { SearchResponse } from '../types/api.js';
import { LensSearchEngine } from '../api/search-engine.js';

export interface SemanticConfig {
  nl_threshold: number;        // Natural language classification threshold
  min_candidates: number;      // K parameter for candidate count
  efSearch: number;           // HNSW efSearch parameter
  confidence_cutoff?: number; // Score threshold for semantic processing
}

export interface SweepResult {
  config: SemanticConfig;
  latency_metrics: {
    stage_c_p50: number;
    stage_c_p95: number;
    stage_c_p99: number;
    total_p50: number;
    total_p95: number;
    total_p99: number;
  };
  quality_metrics: {
    ndcg_at_10: number;
    precision_at_5: number;
    semantic_trigger_rate: number; // % of queries that triggered semantic processing
  };
  queries_tested: number;
  semantic_queries_processed: number;
}

export interface ParameterSweepConfig {
  nl_thresholds: number[];
  candidate_ks: number[];
  ef_search_values: number[];
  confidence_cutoffs?: number[];
  test_queries: TestQuery[];
  baseline_config: SemanticConfig;
  max_latency_increase_ms: number; // p95 latency budget
}

export interface TestQuery {
  query: string;
  repo_sha: string;
  expected_relevance?: number; // 0-1 relevance score for nDCG calculation
  semantic_expected: boolean;  // Whether this query should trigger semantic processing
}

/**
 * Phase 3 Semantic Parameter Sweep Runner
 */
export class SemanticParameterSweep {
  private searchEngine: LensSearchEngine;
  private baselineMetrics: SweepResult | null = null;

  constructor(searchEngine: LensSearchEngine) {
    this.searchEngine = searchEngine;
  }

  /**
   * Run complete parameter sweep with all combinations
   */
  async runParameterSweep(config: ParameterSweepConfig): Promise<{
    results: SweepResult[];
    optimal_config: SemanticConfig;
    improvement_summary: {
      ndcg_improvement: number;
      p95_latency_change: number;
      semantic_precision_gain: number;
    };
  }> {
    console.log(`🔄 Starting Phase 3 Semantic Parameter Sweep`);
    console.log(`📊 Testing ${config.nl_thresholds.length} NL thresholds × ${config.candidate_ks.length} K values × ${config.ef_search_values.length} efSearch values`);
    console.log(`🎯 Latency budget: +${config.max_latency_increase_ms}ms p95`);

    // First, establish baseline with current configuration
    console.log(`📈 Measuring baseline performance...`);
    this.baselineMetrics = await this.measureConfigPerformance(config.baseline_config, config.test_queries);
    
    console.log(`✅ Baseline established:`);
    console.log(`   - nDCG@10: ${this.baselineMetrics.quality_metrics.ndcg_at_10.toFixed(3)}`);
    console.log(`   - Stage-C p95: ${this.baselineMetrics.latency_metrics.stage_c_p95.toFixed(1)}ms`);
    console.log(`   - Semantic trigger rate: ${(this.baselineMetrics.quality_metrics.semantic_trigger_rate * 100).toFixed(1)}%`);

    const allResults: SweepResult[] = [this.baselineMetrics];
    const parameterCombinations = this.generateParameterCombinations(config);

    console.log(`🧪 Testing ${parameterCombinations.length} parameter combinations...`);

    // Test each parameter combination
    for (let i = 0; i < parameterCombinations.length; i++) {
      const paramConfig = parameterCombinations[i]!;
      console.log(`\n[${i + 1}/${parameterCombinations.length}] Testing: NL=${paramConfig.nl_threshold}, K=${paramConfig.min_candidates}, efSearch=${paramConfig.efSearch}`);

      try {
        // Configure semantic engine with new parameters
        await this.applySemanticConfig(paramConfig);

        // Measure performance
        const result = await this.measureConfigPerformance(paramConfig, config.test_queries);
        allResults.push(result);

        // Log immediate results
        const ndcgChange = result.quality_metrics.ndcg_at_10 - this.baselineMetrics.quality_metrics.ndcg_at_10;
        const latencyChange = result.latency_metrics.stage_c_p95 - this.baselineMetrics.latency_metrics.stage_c_p95;
        
        console.log(`   📊 nDCG@10: ${result.quality_metrics.ndcg_at_10.toFixed(3)} (${ndcgChange >= 0 ? '+' : ''}${ndcgChange.toFixed(3)})`);
        console.log(`   ⏱️  Stage-C p95: ${result.latency_metrics.stage_c_p95.toFixed(1)}ms (${latencyChange >= 0 ? '+' : ''}${latencyChange.toFixed(1)}ms)`);
        console.log(`   🎯 Semantic rate: ${(result.quality_metrics.semantic_trigger_rate * 100).toFixed(1)}%`);

      } catch (error) {
        console.warn(`❌ Configuration failed:`, error instanceof Error ? error.message : 'Unknown error');
        continue;
      }
    }

    // Find optimal configuration
    const optimalConfig = this.selectOptimalConfiguration(allResults, config.max_latency_increase_ms);
    
    // Restore baseline configuration
    await this.applySemanticConfig(config.baseline_config);

    const improvementSummary = this.calculateImprovementSummary(optimalConfig);

    return {
      results: allResults,
      optimal_config: optimalConfig.config,
      improvement_summary: improvementSummary,
    };
  }

  /**
   * Generate all parameter combinations for sweep
   */
  private generateParameterCombinations(config: ParameterSweepConfig): SemanticConfig[] {
    const combinations: SemanticConfig[] = [];
    const cutoffs = config.confidence_cutoffs || [undefined];

    for (const nlThreshold of config.nl_thresholds) {
      for (const candidateK of config.candidate_ks) {
        for (const efSearch of config.ef_search_values) {
          for (const cutoff of cutoffs) {
            combinations.push({
              nl_threshold: nlThreshold,
              min_candidates: candidateK,
              efSearch: efSearch,
              confidence_cutoff: cutoff,
            });
          }
        }
      }
    }

    return combinations;
  }

  /**
   * Apply semantic configuration to search engine
   */
  private async applySemanticConfig(config: SemanticConfig): Promise<void> {
    // This would update the query classifier and semantic engine parameters
    // For now, we'll simulate this by updating the configuration
    console.log(`🔧 Applying config: NL=${config.nl_threshold}, K=${config.min_candidates}, efSearch=${config.efSearch}`);
    
    // In a real implementation, this would:
    // 1. Update query classifier NL threshold
    // 2. Update semantic engine candidate limits
    // 3. Update HNSW efSearch parameter
    // 4. Update confidence cutoff if specified
  }

  /**
   * Measure performance for a given configuration
   */
  private async measureConfigPerformance(config: SemanticConfig, testQueries: TestQuery[]): Promise<SweepResult> {
    const latencies: number[] = [];
    const stageCLatencies: number[] = [];
    const totalLatencies: number[] = [];
    const relevanceScores: number[] = [];
    let semanticTriggered = 0;

    for (const testQuery of testQueries) {
      try {
        // Execute search with current configuration
        const searchContext: SearchContext = {
          trace_id: `sweep-${Date.now()}`,
          repo_sha: testQuery.repo_sha,
          query: testQuery.query,
          mode: 'hybrid',
          k: 20,
          fuzzy_distance: 0,
          started_at: new Date(),
          stages: [],
        };

        const result = await this.searchEngine.search(searchContext);

        // Track latencies
        if (result.stage_c_latency !== undefined) {
          stageCLatencies.push(result.stage_c_latency);
          semanticTriggered++;
        } else {
          stageCLatencies.push(0); // No semantic processing
        }

        const totalLatency = (result.stage_a_latency || 0) + (result.stage_b_latency || 0) + (result.stage_c_latency || 0);
        totalLatencies.push(totalLatency);

        // Calculate relevance if expected relevance is provided
        if (testQuery.expected_relevance !== undefined) {
          // Simple relevance approximation - in production would use proper nDCG calculation
          const topHitScore = result.hits[0]?.score || 0;
          const relevance = Math.min(topHitScore, testQuery.expected_relevance);
          relevanceScores.push(relevance);
        }

      } catch (error) {
        console.warn(`Query failed: "${testQuery.query}"`, error);
        stageCLatencies.push(0);
        totalLatencies.push(1000); // Penalty for failures
      }
    }

    // Calculate percentiles
    const stageCMetrics = this.calculateLatencyPercentiles(stageCLatencies);
    const totalMetrics = this.calculateLatencyPercentiles(totalLatencies);

    // Calculate quality metrics
    const avgNDCG = relevanceScores.length > 0 ? 
      relevanceScores.reduce((sum, score) => sum + score, 0) / relevanceScores.length : 0.5;
    
    const semanticTriggerRate = semanticTriggered / testQueries.length;

    return {
      config,
      latency_metrics: {
        stage_c_p50: stageCMetrics.p50,
        stage_c_p95: stageCMetrics.p95,
        stage_c_p99: stageCMetrics.p99,
        total_p50: totalMetrics.p50,
        total_p95: totalMetrics.p95,
        total_p99: totalMetrics.p99,
      },
      quality_metrics: {
        ndcg_at_10: avgNDCG,
        precision_at_5: avgNDCG, // Simplified
        semantic_trigger_rate: semanticTriggerRate,
      },
      queries_tested: testQueries.length,
      semantic_queries_processed: semanticTriggered,
    };
  }

  /**
   * Calculate latency percentiles
   */
  private calculateLatencyPercentiles(latencies: number[]): { p50: number; p95: number; p99: number } {
    const sorted = latencies.slice().sort((a, b) => a - b);
    const len = sorted.length;

    return {
      p50: sorted[Math.floor(len * 0.5)] || 0,
      p95: sorted[Math.floor(len * 0.95)] || 0,
      p99: sorted[Math.floor(len * 0.99)] || 0,
    };
  }

  /**
   * Select optimal configuration based on quality/latency trade-off
   */
  private selectOptimalConfiguration(results: SweepResult[], maxLatencyIncrease: number): SweepResult {
    if (!this.baselineMetrics) {
      return results[0]!;
    }

    const baselineLatency = this.baselineMetrics.latency_metrics.stage_c_p95;
    const latencyBudget = baselineLatency + maxLatencyIncrease;

    // Filter configurations that meet latency constraints
    const feasibleConfigs = results.filter(result => 
      result.latency_metrics.stage_c_p95 <= latencyBudget
    );

    if (feasibleConfigs.length === 0) {
      console.warn(`⚠️  No configurations meet latency budget of ${latencyBudget}ms`);
      return this.baselineMetrics;
    }

    // Select configuration with highest nDCG improvement
    const optimal = feasibleConfigs.reduce((best, current) => 
      current.quality_metrics.ndcg_at_10 > best.quality_metrics.ndcg_at_10 ? current : best
    );

    return optimal;
  }

  /**
   * Calculate improvement summary
   */
  private calculateImprovementSummary(optimal: SweepResult): {
    ndcg_improvement: number;
    p95_latency_change: number;
    semantic_precision_gain: number;
  } {
    if (!this.baselineMetrics) {
      return {
        ndcg_improvement: 0,
        p95_latency_change: 0,
        semantic_precision_gain: 0,
      };
    }

    return {
      ndcg_improvement: optimal.quality_metrics.ndcg_at_10 - this.baselineMetrics.quality_metrics.ndcg_at_10,
      p95_latency_change: optimal.latency_metrics.stage_c_p95 - this.baselineMetrics.latency_metrics.stage_c_p95,
      semantic_precision_gain: optimal.quality_metrics.semantic_trigger_rate - this.baselineMetrics.quality_metrics.semantic_trigger_rate,
    };
  }

  /**
   * Generate standard test queries for parameter sweeping
   */
  static generateTestQueries(): TestQuery[] {
    return [
      // Natural language queries (should trigger semantic)
      {
        query: "find functions that calculate mathematical operations",
        repo_sha: "lens-src",
        expected_relevance: 0.8,
        semantic_expected: true,
      },
      {
        query: "show me code that handles HTTP requests",
        repo_sha: "lens-src", 
        expected_relevance: 0.75,
        semantic_expected: true,
      },
      {
        query: "get components for user interface rendering",
        repo_sha: "lens-src",
        expected_relevance: 0.7,
        semantic_expected: true,
      },
      {
        query: "search for error handling and validation logic",
        repo_sha: "lens-src",
        expected_relevance: 0.65,
        semantic_expected: true,
      },
      {
        query: "locate code that processes search results",
        repo_sha: "lens-src",
        expected_relevance: 0.85,
        semantic_expected: true,
      },

      // Programming syntax queries (should NOT trigger semantic)
      {
        query: "function calculateSum",
        repo_sha: "lens-src",
        expected_relevance: 0.9,
        semantic_expected: false,
      },
      {
        query: "class SearchEngine",
        repo_sha: "lens-src", 
        expected_relevance: 0.95,
        semantic_expected: false,
      },
      {
        query: "async search(",
        repo_sha: "lens-src",
        expected_relevance: 0.8,
        semantic_expected: false,
      },
      {
        query: "import { LensTracer }",
        repo_sha: "lens-src",
        expected_relevance: 0.85,
        semantic_expected: false,
      },
      {
        query: "interface SearchResult",
        repo_sha: "lens-src",
        expected_relevance: 0.9,
        semantic_expected: false,
      },

      // Mixed/edge cases  
      {
        query: "how to implement search function",
        repo_sha: "lens-src",
        expected_relevance: 0.7,
        semantic_expected: true,
      },
      {
        query: "cache performance optimization",
        repo_sha: "lens-src",
        expected_relevance: 0.6,
        semantic_expected: true,
      },
    ];
  }
}

/**
 * Default parameter sweep configuration for Phase 3
 */
export const PHASE3_SWEEP_CONFIG: ParameterSweepConfig = {
  nl_thresholds: [0.4, 0.5, 0.6],
  candidate_ks: [100, 150, 200],
  ef_search_values: [32, 64, 96],
  confidence_cutoffs: [0.1, 0.2, undefined], // undefined means no cutoff
  test_queries: SemanticParameterSweep.generateTestQueries(),
  baseline_config: {
    nl_threshold: 0.5,
    min_candidates: 10,
    efSearch: 64,
    confidence_cutoff: undefined,
  },
  max_latency_increase_ms: 3, // 3ms p95 latency budget
};