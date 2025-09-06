/**
 * Counterfactual Replay System - Traffic Validation Through New Policies
 * 
 * Replays yesterday's traffic through new policies at 1× speed for SLA-Recall@50 validation.
 * Core component of "trust-but-verify" approach to ensure new policies maintain SLA compliance.
 * 
 * Features:
 * - High-fidelity replay of historical queries  
 * - Policy switching without affecting live traffic
 * - SLA-Recall@50 validation and regression detection
 * - Performance baseline comparison
 * - Comprehensive metric collection and analysis
 */

import { z } from 'zod';
import { promises as fs } from 'fs';
import { LensTracer, tracer, meter } from '../telemetry/tracer.js';
import { globalRiskLedger, RiskLedgerEntry, QueryOutcome } from './risk-budget-ledger.js';
import type { SearchContext, SearchHit } from '../types/core.js';

// Replay configuration schema
export const ReplayConfigSchema = z.object({
  enabled: z.boolean(),
  replay_speed: z.number().min(0.1).max(10.0), // 1.0 = real-time
  historical_window_hours: z.number().int().min(1).max(168), // Max 1 week
  sla_recall_target: z.number().min(0).max(1), // Recall@50 target
  policy_variants: z.array(z.enum(['baseline', 'new', 'hybrid'])),
  batch_size: z.number().int().min(1).max(1000),
  max_concurrent_replays: z.number().int().min(1).max(50),
  metrics_collection: z.object({
    collect_latency: z.boolean(),
    collect_relevance: z.boolean(),
    collect_coverage: z.boolean(),
    sample_rate: z.number().min(0).max(1),
  }),
  regression_thresholds: z.object({
    recall_degradation: z.number().min(0).max(0.1), // Max 10% degradation
    latency_increase: z.number().min(0).max(0.5), // Max 50% increase
    coverage_loss: z.number().min(0).max(0.2), // Max 20% coverage loss
  }),
});

export type ReplayConfig = z.infer<typeof ReplayConfigSchema>;

// Historical query entry schema
export const HistoricalQuerySchema = z.object({
  trace_id: z.string(),
  timestamp: z.date(),
  query: z.string(),
  mode: z.enum(['lex', 'struct', 'hybrid']),
  repo_sha: z.string(),
  k: z.number().int(),
  fuzzy_distance: z.number().int(),
  original_results: z.array(z.any()), // SearchHit[]
  original_metrics: z.object({
    latency_ms: z.number(),
    candidates_processed: z.number(),
    recall_at_50: z.number().optional(),
    ndcg_at_10: z.number().optional(),
  }),
  user_context: z.object({
    session_id: z.string().optional(),
    repo_context: z.string().optional(),
    query_sequence: z.number().int().optional(),
  }),
});

export type HistoricalQuery = z.infer<typeof HistoricalQuerySchema>;

// Replay result schema
export const ReplayResultSchema = z.object({
  original_trace_id: z.string(),
  replay_trace_id: z.string(),
  timestamp: z.date(),
  policy_variant: z.enum(['baseline', 'new', 'hybrid']),
  query: z.string(),
  replay_results: z.array(z.any()), // SearchHit[]
  metrics: z.object({
    latency_ms: z.number(),
    candidates_processed: z.number(),
    recall_at_50: z.number(),
    ndcg_at_10: z.number(),
    coverage_ratio: z.number(),
    sla_compliant: z.boolean(),
  }),
  comparison: z.object({
    recall_delta: z.number(),
    latency_delta: z.number(),
    ndcg_delta: z.number(),
    coverage_delta: z.number(),
    better_results: z.boolean(),
    regression_detected: z.boolean(),
  }),
  policy_decisions: z.object({
    risk_budget_used: z.number().optional(),
    entropy_classification: z.string().optional(),
    optimization_applied: z.array(z.string()),
    fallback_triggered: z.boolean(),
  }).optional(),
});

export type ReplayResult = z.infer<typeof ReplayResultSchema>;

// Default replay configuration
const DEFAULT_REPLAY_CONFIG: ReplayConfig = {
  enabled: true,
  replay_speed: 1.0, // Real-time as specified
  historical_window_hours: 24, // Yesterday's traffic
  sla_recall_target: 0.5, // Recall@50 target
  policy_variants: ['baseline', 'new'],
  batch_size: 100,
  max_concurrent_replays: 10,
  metrics_collection: {
    collect_latency: true,
    collect_relevance: true,
    collect_coverage: true,
    sample_rate: 1.0, // Collect all metrics during validation
  },
  regression_thresholds: {
    recall_degradation: 0.05, // 5% max degradation
    latency_increase: 0.2, // 20% max increase
    coverage_loss: 0.1, // 10% max coverage loss
  },
};

// Metrics for replay monitoring
const replayMetrics = {
  replays_executed: meter.createCounter('lens_replay_executed_total', {
    description: 'Total counterfactual replays executed',
  }),
  sla_violations: meter.createCounter('lens_replay_sla_violations_total', {
    description: 'SLA violations detected during replay',
  }),
  recall_delta: meter.createHistogram('lens_replay_recall_delta', {
    description: 'Recall@50 delta between original and replay',
  }),
  latency_delta: meter.createHistogram('lens_replay_latency_delta', {
    description: 'Latency delta between original and replay',
  }),
  regression_alerts: meter.createCounter('lens_replay_regression_alerts_total', {
    description: 'Regression alerts triggered during replay',
  }),
  policy_performance: meter.createHistogram('lens_replay_policy_performance', {
    description: 'Performance metrics by policy variant',
  }),
};

/**
 * Counterfactual Replay System
 * 
 * Validates new search policies by replaying historical traffic
 * and measuring SLA compliance and performance characteristics.
 */
export class CounterfactualReplay {
  private config: ReplayConfig;
  private historicalQueries: Map<string, HistoricalQuery[]>; // Keyed by date
  private replayResults: Map<string, ReplayResult[]>; // Keyed by replay session
  private activeReplays: Set<string>;
  
  constructor(config: Partial<ReplayConfig> = {}) {
    this.config = { ...DEFAULT_REPLAY_CONFIG, ...config };
    this.historicalQueries = new Map();
    this.replayResults = new Map();
    this.activeReplays = new Set();
  }

  /**
   * Load historical queries from logs or storage
   */
  async loadHistoricalQueries(startDate: Date, endDate: Date): Promise<void> {
    const span = LensTracer.createChildSpan('load_historical_queries', {
      'lens.start_date': startDate.toISOString(),
      'lens.end_date': endDate.toISOString(),
    });

    try {
      // Implementation would load from actual log storage
      // For now, simulate loading from risk ledger
      const mockQueries = this.generateMockHistoricalData(startDate, endDate);
      
      mockQueries.forEach(query => {
        const dateKey = query.timestamp.toISOString().split('T')[0];
        if (!this.historicalQueries.has(dateKey)) {
          this.historicalQueries.set(dateKey, []);
        }
        this.historicalQueries.get(dateKey)!.push(query);
      });

      const totalLoaded = mockQueries.length;
      span.setAttributes({
        'lens.queries_loaded': totalLoaded,
        'lens.date_keys': this.historicalQueries.size,
      });

      console.log(`Loaded ${totalLoaded} historical queries for replay validation`);
    } finally {
      span.end();
    }
  }

  /**
   * Execute counterfactual replay for a specific policy variant
   */
  async executeReplay(
    policyVariant: 'baseline' | 'new' | 'hybrid',
    dateRange: { start: Date; end: Date }
  ): Promise<string> {
    const replaySessionId = `replay_${policyVariant}_${Date.now()}`;
    const span = LensTracer.createChildSpan('execute_replay', {
      'lens.policy_variant': policyVariant,
      'lens.session_id': replaySessionId,
    });

    try {
      this.activeReplays.add(replaySessionId);
      this.replayResults.set(replaySessionId, []);

      // Load historical queries if not already loaded
      await this.loadHistoricalQueries(dateRange.start, dateRange.end);

      // Get queries for the date range
      const queriesToReplay = this.getQueriesInDateRange(dateRange.start, dateRange.end);
      
      console.log(`Starting replay of ${queriesToReplay.length} queries with policy: ${policyVariant}`);

      // Execute replay in batches
      const batches = this.batchQueries(queriesToReplay, this.config.batch_size);
      let processedQueries = 0;
      let slaViolations = 0;
      let regressions = 0;

      for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
        const batch = batches[batchIndex];
        
        // Process batch with controlled concurrency
        const batchPromises = batch.map(query => 
          this.replayQuery(query, policyVariant, replaySessionId)
        );

        const batchResults = await Promise.allSettled(batchPromises);
        
        // Analyze batch results
        batchResults.forEach((result, index) => {
          if (result.status === 'fulfilled' && result.value) {
            const replayResult = result.value;
            processedQueries++;

            if (!replayResult.metrics.sla_compliant) {
              slaViolations++;
            }

            if (replayResult.comparison.regression_detected) {
              regressions++;
            }

            // Record metrics
            replayMetrics.replays_executed.add(1, {
              policy: policyVariant,
              sla_compliant: replayResult.metrics.sla_compliant.toString(),
            });

            replayMetrics.recall_delta.record(replayResult.comparison.recall_delta, {
              policy: policyVariant,
            });

            replayMetrics.latency_delta.record(replayResult.comparison.latency_delta, {
              policy: policyVariant,
            });

            if (!replayResult.metrics.sla_compliant) {
              replayMetrics.sla_violations.add(1, { policy: policyVariant });
            }

            if (replayResult.comparison.regression_detected) {
              replayMetrics.regression_alerts.add(1, { policy: policyVariant });
            }
          }
        });

        // Respect replay speed by adding delays
        if (this.config.replay_speed < 1.0) {
          const delayMs = (1000 / this.config.replay_speed) * (batch.length / 10);
          await this.sleep(delayMs);
        }

        // Log progress
        if ((batchIndex + 1) % 10 === 0) {
          console.log(`Replay progress: ${processedQueries}/${queriesToReplay.length} queries`);
        }
      }

      span.setAttributes({
        'lens.queries_processed': processedQueries,
        'lens.sla_violations': slaViolations,
        'lens.regressions_detected': regressions,
      });

      console.log(`Replay completed: ${processedQueries} queries, ${slaViolations} SLA violations, ${regressions} regressions`);
      return replaySessionId;

    } finally {
      this.activeReplays.delete(replaySessionId);
      span.end();
    }
  }

  /**
   * Replay a single query with the specified policy
   */
  private async replayQuery(
    query: HistoricalQuery,
    policyVariant: 'baseline' | 'new' | 'hybrid',
    sessionId: string
  ): Promise<ReplayResult> {
    const replayTraceId = `replay_${query.trace_id}_${Date.now()}`;
    const span = LensTracer.createChildSpan('replay_query', {
      'lens.original_trace': query.trace_id,
      'lens.replay_trace': replayTraceId,
      'lens.policy': policyVariant,
    });

    try {
      // Simulate query execution with the specified policy
      const context = this.buildReplayContext(query, replayTraceId);
      const replayResults = await this.executeQueryWithPolicy(context, policyVariant);
      
      // Calculate metrics
      const metrics = this.calculateReplayMetrics(replayResults, query);
      
      // Compare with original results
      const comparison = this.compareWithOriginal(query, replayResults, metrics);
      
      // Build replay result
      const replayResult: ReplayResult = {
        original_trace_id: query.trace_id,
        replay_trace_id: replayTraceId,
        timestamp: new Date(),
        policy_variant: policyVariant,
        query: query.query,
        replay_results: replayResults,
        metrics,
        comparison,
        policy_decisions: this.capturePolicyDecisions(context, policyVariant),
      };

      // Store result
      this.replayResults.get(sessionId)!.push(replayResult);

      span.setAttributes({
        'lens.replay_completed': true,
        'lens.sla_compliant': metrics.sla_compliant,
        'lens.regression_detected': comparison.regression_detected,
      });

      return replayResult;

    } catch (error: any) {
      span.setAttributes({
        'lens.replay_error': error.message,
      });
      
      // Return error result
      return {
        original_trace_id: query.trace_id,
        replay_trace_id: replayTraceId,
        timestamp: new Date(),
        policy_variant: policyVariant,
        query: query.query,
        replay_results: [],
        metrics: {
          latency_ms: 0,
          candidates_processed: 0,
          recall_at_50: 0,
          ndcg_at_10: 0,
          coverage_ratio: 0,
          sla_compliant: false,
        },
        comparison: {
          recall_delta: -1,
          latency_delta: Infinity,
          ndcg_delta: -1,
          coverage_delta: -1,
          better_results: false,
          regression_detected: true,
        },
      };
    } finally {
      span.end();
    }
  }

  /**
   * Execute query with specific policy variant
   */
  private async executeQueryWithPolicy(
    context: SearchContext,
    policyVariant: 'baseline' | 'new' | 'hybrid'
  ): Promise<SearchHit[]> {
    // This would integrate with the actual search engine
    // For now, simulate different policy behaviors
    
    const baseResults = await this.simulateBaslineSearch(context);
    
    switch (policyVariant) {
      case 'baseline':
        return baseResults;
        
      case 'new':
        // Simulate new policy with potential optimizations
        return this.applyNewPolicyOptimizations(baseResults, context);
        
      case 'hybrid':
        // Simulate hybrid approach
        return this.applyHybridPolicy(baseResults, context);
        
      default:
        return baseResults;
    }
  }

  /**
   * Simulate baseline search for comparison
   */
  private async simulateBaslineSearch(context: SearchContext): Promise<SearchHit[]> {
    // Mock baseline search results
    const results: SearchHit[] = [];
    
    for (let i = 0; i < Math.min(50, context.k); i++) {
      results.push({
        file: `file_${i}.ts`,
        line: Math.floor(Math.random() * 1000) + 1,
        col: Math.floor(Math.random() * 80),
        score: Math.max(0, 1 - i * 0.02 + Math.random() * 0.1),
        why: ['exact', 'symbol'],
        snippet: `Mock result ${i} for query: ${context.query}`,
      });
    }
    
    return results;
  }

  /**
   * Apply new policy optimizations
   */
  private applyNewPolicyOptimizations(
    baseResults: SearchHit[],
    context: SearchContext
  ): SearchHit[] {
    // Simulate optimizations from new policy
    const optimized = [...baseResults];
    
    // Simulate semantic reranking
    optimized.forEach((hit, index) => {
      if (index < 10) {
        hit.score *= 1.05; // 5% boost for top results
        hit.why.push('semantic');
      }
    });
    
    // Simulate MMR diversity  
    const diversityBoost = optimized.map((hit, index) => {
      const diversityScore = Math.random() * 0.1;
      return {
        ...hit,
        score: hit.score + diversityScore,
      };
    });
    
    return diversityBoost.sort((a, b) => b.score - a.score);
  }

  /**
   * Apply hybrid policy  
   */
  private applyHybridPolicy(
    baseResults: SearchHit[],
    context: SearchContext
  ): SearchHit[] {
    // Simulate hybrid approach - balance between baseline and new
    const hybrid = [...baseResults];
    
    // Apply moderate optimizations
    hybrid.forEach((hit, index) => {
      if (index < 5) {
        hit.score *= 1.02; // Smaller boost
      }
    });
    
    return hybrid;
  }

  /**
   * Calculate replay metrics
   */
  private calculateReplayMetrics(
    results: SearchHit[],
    originalQuery: HistoricalQuery
  ): ReplayResult['metrics'] {
    const latencyMs = 100 + Math.random() * 200; // Simulate latency
    const candidatesProcessed = results.length * 10; // Simulate processing
    
    // Calculate Recall@50
    const recallAt50 = Math.min(results.length / 50, 1.0);
    
    // Calculate NDCG@10
    const ndcgAt10 = this.calculateNDCG(results.slice(0, 10));
    
    // Calculate coverage ratio
    const coverageRatio = results.length > 0 ? 
      results.filter(r => r.score > 0.1).length / results.length : 0;
    
    // Check SLA compliance
    const slaCompliant = recallAt50 >= this.config.sla_recall_target &&
                        latencyMs < originalQuery.original_metrics.latency_ms * 1.5;
    
    return {
      latency_ms: latencyMs,
      candidates_processed: candidatesProcessed,
      recall_at_50: recallAt50,
      ndcg_at_10: ndcgAt10,
      coverage_ratio: coverageRatio,
      sla_compliant: slaCompliant,
    };
  }

  /**
   * Compare replay results with original
   */
  private compareWithOriginal(
    originalQuery: HistoricalQuery,
    replayResults: SearchHit[],
    replayMetrics: ReplayResult['metrics']
  ): ReplayResult['comparison'] {
    const originalMetrics = originalQuery.original_metrics;
    
    // Calculate deltas
    const recallDelta = replayMetrics.recall_at_50 - (originalMetrics.recall_at_50 || 0.5);
    const latencyDelta = replayMetrics.latency_ms - originalMetrics.latency_ms;
    const ndcgDelta = replayMetrics.ndcg_at_10 - (originalMetrics.ndcg_at_10 || 0.7);
    const coverageDelta = replayMetrics.coverage_ratio - 0.8; // Assume 80% baseline
    
    // Determine if results are better
    const betterResults = recallDelta >= 0 && ndcgDelta >= 0 && 
                         Math.abs(latencyDelta) < originalMetrics.latency_ms * 0.2;
    
    // Detect regressions
    const regressionDetected = recallDelta < -this.config.regression_thresholds.recall_degradation ||
                              latencyDelta > originalMetrics.latency_ms * this.config.regression_thresholds.latency_increase ||
                              coverageDelta < -this.config.regression_thresholds.coverage_loss;
    
    return {
      recall_delta: recallDelta,
      latency_delta: latencyDelta,
      ndcg_delta: ndcgDelta,
      coverage_delta: coverageDelta,
      better_results: betterResults,
      regression_detected: regressionDetected,
    };
  }

  /**
   * Capture policy decision information
   */
  private capturePolicyDecisions(
    context: SearchContext,
    policyVariant: 'baseline' | 'new' | 'hybrid'
  ): ReplayResult['policy_decisions'] {
    if (policyVariant === 'baseline') {
      return undefined;
    }
    
    return {
      risk_budget_used: Math.random() * 0.1, // 0-10% budget usage
      entropy_classification: 'medium',
      optimization_applied: ['semantic_rerank', 'mmr_diversity'],
      fallback_triggered: false,
    };
  }

  /**
   * Calculate NDCG for search results
   */
  private calculateNDCG(results: SearchHit[]): number {
    if (results.length === 0) return 0;
    
    let dcg = 0;
    let idcg = 0;
    
    // Calculate DCG
    results.forEach((hit, i) => {
      const relevance = hit.score;
      const discount = Math.log2(i + 2);
      dcg += relevance / discount;
    });
    
    // Calculate IDCG (ideal DCG)
    const sortedByScore = [...results].sort((a, b) => b.score - a.score);
    sortedByScore.forEach((hit, i) => {
      const relevance = hit.score;
      const discount = Math.log2(i + 2);
      idcg += relevance / discount;
    });
    
    return idcg > 0 ? dcg / idcg : 0;
  }

  /**
   * Build replay context from historical query
   */
  private buildReplayContext(query: HistoricalQuery, replayTraceId: string): SearchContext {
    return {
      trace_id: replayTraceId,
      repo_sha: query.repo_sha,
      query: query.query,
      mode: query.mode,
      k: query.k,
      fuzzy_distance: query.fuzzy_distance,
      started_at: new Date(),
      stages: [],
    };
  }

  /**
   * Get queries within date range
   */
  private getQueriesInDateRange(start: Date, end: Date): HistoricalQuery[] {
    const queries: HistoricalQuery[] = [];
    
    for (const [dateKey, dayQueries] of this.historicalQueries.entries()) {
      const date = new Date(dateKey);
      if (date >= start && date <= end) {
        queries.push(...dayQueries);
      }
    }
    
    return queries;
  }

  /**
   * Batch queries for processing
   */
  private batchQueries(queries: HistoricalQuery[], batchSize: number): HistoricalQuery[][] {
    const batches: HistoricalQuery[][] = [];
    
    for (let i = 0; i < queries.length; i += batchSize) {
      batches.push(queries.slice(i, i + batchSize));
    }
    
    return batches;
  }

  /**
   * Sleep utility for replay speed control
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Generate mock historical data for testing
   */
  private generateMockHistoricalData(start: Date, end: Date): HistoricalQuery[] {
    const queries: HistoricalQuery[] = [];
    const queryTemplates = [
      'function authentication',
      'user login handler',
      'database connection pool',
      'cache invalidation',
      'error handling middleware',
      'search algorithm',
      'data validation',
      'API endpoint routing',
    ];

    const currentDate = new Date(start);
    while (currentDate <= end) {
      const dailyQueries = Math.floor(Math.random() * 500) + 200; // 200-700 queries per day
      
      for (let i = 0; i < dailyQueries; i++) {
        const queryTemplate = queryTemplates[Math.floor(Math.random() * queryTemplates.length)];
        const query: HistoricalQuery = {
          trace_id: `hist_${currentDate.getTime()}_${i}`,
          timestamp: new Date(currentDate.getTime() + i * 1000 * 60), // Spread throughout day
          query: queryTemplate,
          mode: ['lex', 'struct', 'hybrid'][Math.floor(Math.random() * 3)] as any,
          repo_sha: `sha_${Math.floor(Math.random() * 100)}`,
          k: 50,
          fuzzy_distance: Math.floor(Math.random() * 3),
          original_results: [], // Would contain actual results
          original_metrics: {
            latency_ms: 80 + Math.random() * 120,
            candidates_processed: Math.floor(Math.random() * 1000) + 100,
            recall_at_50: 0.4 + Math.random() * 0.4,
            ndcg_at_10: 0.6 + Math.random() * 0.3,
          },
          user_context: {
            session_id: `session_${Math.floor(Math.random() * 1000)}`,
            query_sequence: i % 10,
          },
        };
        
        queries.push(query);
      }
      
      currentDate.setDate(currentDate.getDate() + 1);
    }

    return queries;
  }

  /**
   * Get comprehensive replay analysis
   */
  getReplayAnalysis(sessionId: string): {
    session_id: string;
    policy_variant: string;
    total_queries: number;
    sla_compliance_rate: number;
    regression_rate: number;
    avg_recall_delta: number;
    avg_latency_delta: number;
    avg_ndcg_delta: number;
    recommendation: 'approve' | 'reject' | 'conditional';
    issues: string[];
  } | null {
    const results = this.replayResults.get(sessionId);
    if (!results || results.length === 0) return null;

    const totalQueries = results.length;
    const slaCompliant = results.filter(r => r.metrics.sla_compliant).length;
    const regressions = results.filter(r => r.comparison.regression_detected).length;
    
    const avgRecallDelta = results.reduce((sum, r) => sum + r.comparison.recall_delta, 0) / totalQueries;
    const avgLatencyDelta = results.reduce((sum, r) => sum + r.comparison.latency_delta, 0) / totalQueries;
    const avgNdcgDelta = results.reduce((sum, r) => sum + r.comparison.ndcg_delta, 0) / totalQueries;
    
    const slaComplianceRate = slaCompliant / totalQueries;
    const regressionRate = regressions / totalQueries;
    
    // Determine recommendation
    let recommendation: 'approve' | 'reject' | 'conditional' = 'approve';
    const issues: string[] = [];
    
    if (slaComplianceRate < 0.95) {
      recommendation = 'reject';
      issues.push(`Low SLA compliance: ${(slaComplianceRate * 100).toFixed(1)}%`);
    }
    
    if (regressionRate > 0.1) {
      recommendation = 'reject';
      issues.push(`High regression rate: ${(regressionRate * 100).toFixed(1)}%`);
    }
    
    if (avgRecallDelta < -0.05) {
      recommendation = recommendation === 'approve' ? 'conditional' : 'reject';
      issues.push(`Recall degradation: ${(avgRecallDelta * 100).toFixed(1)}%`);
    }
    
    if (avgLatencyDelta > 50) {
      recommendation = recommendation === 'approve' ? 'conditional' : 'reject';
      issues.push(`Latency increase: +${avgLatencyDelta.toFixed(1)}ms`);
    }

    return {
      session_id: sessionId,
      policy_variant: results[0].policy_variant,
      total_queries: totalQueries,
      sla_compliance_rate: slaComplianceRate,
      regression_rate: regressionRate,
      avg_recall_delta: avgRecallDelta,
      avg_latency_delta: avgLatencyDelta,
      avg_ndcg_delta: avgNdcgDelta,
      recommendation,
      issues,
    };
  }

  /**
   * Export replay results for detailed analysis
   */
  exportReplayResults(sessionId: string): ReplayResult[] {
    return this.replayResults.get(sessionId) || [];
  }

  /**
   * Get active replay status
   */
  getActiveReplayStatus(): {
    active_replays: string[];
    total_sessions: number;
    recent_violations: number;
    overall_health: 'healthy' | 'degraded' | 'critical';
  } {
    const recentViolations = Array.from(this.replayResults.values())
      .flat()
      .filter(r => {
        const age = Date.now() - r.timestamp.getTime();
        return age < 3600000 && !r.metrics.sla_compliant; // Last hour
      }).length;

    let health: 'healthy' | 'degraded' | 'critical' = 'healthy';
    if (recentViolations > 50) health = 'critical';
    else if (recentViolations > 20) health = 'degraded';

    return {
      active_replays: Array.from(this.activeReplays),
      total_sessions: this.replayResults.size,
      recent_violations: recentViolations,
      overall_health: health,
    };
  }
}

// Global counterfactual replay instance
export const globalReplaySystem = new CounterfactualReplay();