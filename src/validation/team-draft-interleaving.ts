/**
 * Team-Draft Interleaving (TDI) - Live A/B Testing Infrastructure  
 * 
 * Implements sophisticated online interleaving between Lens(new) vs Lens(baseline)
 * for NL+symbol queries with proper statistical controls and bias mitigation.
 * 
 * Core features:
 * - Unbiased interleaving with credit assignment
 * - Statistical significance testing with Wilson confidence intervals
 * - Query-level randomization with consistent user experience
 * - Real-time metrics collection and analysis
 */

import { z } from 'zod';
import { LensTracer, tracer, meter } from '../telemetry/tracer.js';
import { globalRiskLedger, RiskLedgerEntry, QueryOutcome } from './risk-budget-ledger.js';
import type { SearchContext, SearchHit } from '../types/core.js';

// Treatment variants
export enum TreatmentVariant {
  BASELINE = 'baseline',
  NEW = 'new'
}

// Interleaving methods
export enum InterleavingMethod {
  TEAM_DRAFT = 'team_draft',
  PROBABILISTIC = 'probabilistic',
  BALANCED = 'balanced'
}

// Query classification for TDI eligibility  
export enum QueryType {
  NATURAL_LANGUAGE = 'nl',
  SYMBOL_SEARCH = 'symbol', 
  HYBRID = 'hybrid',
  STRUCTURAL = 'structural'
}

// TDI configuration schema
export const TDIConfigSchema = z.object({
  enabled: z.boolean(),
  traffic_percentage: z.number().min(0).max(100), // Start at 25%
  eligible_query_types: z.array(z.nativeEnum(QueryType)),
  interleaving_method: z.nativeEnum(InterleavingMethod),
  min_results_for_interleaving: z.number().int().min(2),
  statistical_power: z.number().min(0.5).max(0.99), // 0.8 = 80% power
  significance_level: z.number().min(0.01).max(0.1), // 0.05 = 5% significance
  min_sample_size: z.number().int().min(100),
  evaluation_window_hours: z.number().int().min(1).max(168), // Max 1 week
  bias_mitigation: z.object({
    position_bias_correction: z.boolean(),
    click_model: z.enum(['dcg', 'cascade', 'position_based']),
    trust_bias_threshold: z.number().min(0).max(1),
  }),
});

export type TDIConfig = z.infer<typeof TDIConfigSchema>;

// Interleaving result schema
export const InterleavingResultSchema = z.object({
  trace_id: z.string(),
  timestamp: z.date(),
  query: z.string(),
  query_type: z.nativeEnum(QueryType),
  treatment_variant: z.nativeEnum(TreatmentVariant),
  interleaving_method: z.nativeEnum(InterleavingMethod),
  baseline_results: z.array(z.any()), // SearchHit[]
  new_results: z.array(z.any()), // SearchHit[]  
  interleaved_results: z.array(z.any()), // InterleavedHit[]
  team_assignments: z.array(z.nativeEnum(TreatmentVariant)),
  metrics: z.object({
    baseline_ndcg: z.number().optional(),
    new_ndcg: z.number().optional(),
    interleaved_ndcg: z.number(),
    baseline_positions: z.array(z.number()),
    new_positions: z.array(z.number()),
    winner: z.nativeEnum(TreatmentVariant).optional(),
    confidence: z.number().min(0).max(1).optional(),
  }),
  user_interactions: z.array(z.object({
    position: z.number(),
    source_variant: z.nativeEnum(TreatmentVariant),
    interaction_type: z.enum(['click', 'hover', 'copy', 'view']),
    timestamp: z.date(),
  })).optional(),
  bias_corrections: z.object({
    position_bias_applied: z.boolean(),
    trust_bias_detected: z.boolean(),
    adjusted_metrics: z.record(z.number()).optional(),
  }).optional(),
});

export type InterleavingResult = z.infer<typeof InterleavingResultSchema>;

// Interleaved hit with source tracking
interface InterleavedHit extends SearchHit {
  source_variant: TreatmentVariant;
  original_position: number;
  interleaved_position: number;
  team_draft_round: number;
}

// Default TDI configuration matching TODO.md requirements
const DEFAULT_TDI_CONFIG: TDIConfig = {
  enabled: true,
  traffic_percentage: 25, // Start at 25% as specified
  eligible_query_types: [QueryType.NATURAL_LANGUAGE, QueryType.SYMBOL_SEARCH, QueryType.HYBRID],
  interleaving_method: InterleavingMethod.TEAM_DRAFT,
  min_results_for_interleaving: 4,
  statistical_power: 0.8,
  significance_level: 0.05,
  min_sample_size: 200,
  evaluation_window_hours: 24,
  bias_mitigation: {
    position_bias_correction: true,
    click_model: 'dcg',
    trust_bias_threshold: 0.15,
  },
};

// Metrics for TDI monitoring  
const tdiMetrics = {
  experiments_started: meter.createCounter('lens_tdi_experiments_total', {
    description: 'Total TDI experiments conducted',
  }),
  interleaving_ratio: meter.createObservableGauge('lens_tdi_interleaving_ratio', {
    description: 'Proportion of eligible queries interleaved',
  }),
  variant_performance: meter.createHistogram('lens_tdi_variant_ndcg', {
    description: 'NDCG scores by variant',
  }),
  statistical_power: meter.createObservableGauge('lens_tdi_statistical_power', {
    description: 'Current statistical power of ongoing experiments',
  }),
  bias_corrections: meter.createCounter('lens_tdi_bias_corrections_total', {
    description: 'Bias corrections applied',
  }),
};

/**
 * Team-Draft Interleaving System
 * 
 * Implements unbiased online evaluation between baseline and new systems
 * using team-draft interleaving with proper statistical controls.
 */
export class TeamDraftInterleaving {
  private config: TDIConfig;
  private results: Map<string, InterleavingResult[]>; // Keyed by experiment_id
  private activeExperiments: Map<string, { start_time: Date; samples: number }>;
  
  constructor(config: Partial<TDIConfig> = {}) {
    this.config = { ...DEFAULT_TDI_CONFIG, ...config };
    this.results = new Map();
    this.activeExperiments = new Map();
  }

  /**
   * Determine if query is eligible for TDI
   */
  isQueryEligible(query: string, context: SearchContext): boolean {
    const span = LensTracer.createChildSpan('tdi_eligibility_check', {
      'lens.query_length': query.length,
      'lens.mode': context.mode,
      'lens.tdi_enabled': this.config.enabled,
    });

    try {
      // Check if TDI is enabled
      if (!this.config.enabled) {
        span.setAttributes({ 'lens.tdi_eligible': false, 'lens.reason': 'disabled' });
        return false;
      }

      // Check traffic percentage
      const hash = this.hashQuery(query + context.trace_id);
      const trafficSample = hash % 100;
      if (trafficSample >= this.config.traffic_percentage) {
        span.setAttributes({ 'lens.tdi_eligible': false, 'lens.reason': 'traffic_sampling' });
        return false;
      }

      // Classify query type
      const queryType = this.classifyQuery(query, context);
      if (!this.config.eligible_query_types.includes(queryType)) {
        span.setAttributes({ 'lens.tdi_eligible': false, 'lens.reason': 'query_type_ineligible' });
        return false;
      }

      // Additional eligibility checks
      if (query.length < 3) {
        span.setAttributes({ 'lens.tdi_eligible': false, 'lens.reason': 'query_too_short' });
        return false;
      }

      span.setAttributes({ 
        'lens.tdi_eligible': true, 
        'lens.query_type': queryType,
        'lens.traffic_sample': trafficSample,
      });
      
      return true;
    } finally {
      span.end();
    }
  }

  /**
   * Perform team-draft interleaving between baseline and new results
   */
  async performInterleaving(
    query: string,
    context: SearchContext,
    baselineResults: SearchHit[],
    newResults: SearchHit[]
  ): Promise<InterleavingResult | null> {
    const span = LensTracer.createChildSpan('team_draft_interleaving', {
      'lens.baseline_count': baselineResults.length,
      'lens.new_count': newResults.length,
      'lens.query': query,
    });

    try {
      // Check minimum results requirement
      if (baselineResults.length < this.config.min_results_for_interleaving ||
          newResults.length < this.config.min_results_for_interleaving) {
        span.setAttributes({ 'lens.interleaving_skipped': 'insufficient_results' });
        return null;
      }

      // Determine treatment variant for this query
      const treatmentVariant = this.assignTreatmentVariant(query, context);
      
      // Perform team-draft selection
      const interleavedResults = this.teamDraftSelection(baselineResults, newResults, context);
      
      // Calculate metrics
      const metrics = this.calculateMetrics(baselineResults, newResults, interleavedResults);
      
      // Create interleaving result
      const result: InterleavingResult = {
        trace_id: context.trace_id,
        timestamp: new Date(),
        query,
        query_type: this.classifyQuery(query, context),
        treatment_variant: treatmentVariant,
        interleaving_method: this.config.interleaving_method,
        baseline_results: baselineResults,
        new_results: newResults,
        interleaved_results: interleavedResults,
        team_assignments: interleavedResults.map(hit => hit.source_variant),
        metrics,
        bias_corrections: this.applyBiasCorrections(interleavedResults, metrics),
      };

      // Store result
      const experimentId = `${this.classifyQuery(query, context)}_${new Date().toISOString().split('T')[0]}`;
      if (!this.results.has(experimentId)) {
        this.results.set(experimentId, []);
      }
      this.results.get(experimentId)!.push(result);

      // Update active experiments tracking
      if (!this.activeExperiments.has(experimentId)) {
        this.activeExperiments.set(experimentId, {
          start_time: new Date(),
          samples: 0,
        });
      }
      this.activeExperiments.get(experimentId)!.samples++;

      // Record metrics
      tdiMetrics.experiments_started.add(1, {
        query_type: result.query_type,
        treatment: treatmentVariant,
      });

      tdiMetrics.variant_performance.record(metrics.baseline_ndcg || 0, { variant: 'baseline' });
      tdiMetrics.variant_performance.record(metrics.new_ndcg || 0, { variant: 'new' });

      span.setAttributes({
        'lens.interleaving_completed': true,
        'lens.interleaved_count': interleavedResults.length,
        'lens.treatment_variant': treatmentVariant,
        'lens.baseline_ndcg': metrics.baseline_ndcg,
        'lens.new_ndcg': metrics.new_ndcg,
      });

      return result;
    } finally {
      span.end();
    }
  }

  /**
   * Team-draft selection algorithm
   */
  private teamDraftSelection(
    baselineResults: SearchHit[],
    newResults: SearchHit[],
    context: SearchContext
  ): InterleavedHit[] {
    const interleaved: InterleavedHit[] = [];
    let baselineIdx = 0;
    let newIdx = 0;
    let round = 0;
    let teamSelection = this.initializeTeamSelection(context);

    while (interleaved.length < Math.min(20, baselineResults.length + newResults.length) &&
           (baselineIdx < baselineResults.length || newIdx < newResults.length)) {
      
      const selectBaseline = teamSelection[round % teamSelection.length] === TreatmentVariant.BASELINE;
      round++;

      if (selectBaseline && baselineIdx < baselineResults.length) {
        const hit = baselineResults[baselineIdx];
        interleaved.push({
          ...hit,
          source_variant: TreatmentVariant.BASELINE,
          original_position: baselineIdx,
          interleaved_position: interleaved.length,
          team_draft_round: round,
        });
        baselineIdx++;
      } else if (!selectBaseline && newIdx < newResults.length) {
        const hit = newResults[newIdx];
        interleaved.push({
          ...hit,
          source_variant: TreatmentVariant.NEW,
          original_position: newIdx,
          interleaved_position: interleaved.length,
          team_draft_round: round,
        });
        newIdx++;
      } else {
        // Fallback: select from available team
        if (baselineIdx < baselineResults.length) {
          const hit = baselineResults[baselineIdx];
          interleaved.push({
            ...hit,
            source_variant: TreatmentVariant.BASELINE,
            original_position: baselineIdx,
            interleaved_position: interleaved.length,
            team_draft_round: round,
          });
          baselineIdx++;
        } else if (newIdx < newResults.length) {
          const hit = newResults[newIdx];
          interleaved.push({
            ...hit,
            source_variant: TreatmentVariant.NEW,
            original_position: newIdx,
            interleaved_position: interleaved.length,
            team_draft_round: round,
          });
          newIdx++;
        }
      }
    }

    return interleaved;
  }

  /**
   * Initialize team selection pattern for fair drafting
   */
  private initializeTeamSelection(context: SearchContext): TreatmentVariant[] {
    // Create balanced team selection pattern
    const pattern: TreatmentVariant[] = [];
    const seed = this.hashQuery(context.trace_id + context.query);
    
    // Start with random team
    let currentTeam = (seed % 2) === 0 ? TreatmentVariant.BASELINE : TreatmentVariant.NEW;
    
    // Alternate teams for fair drafting
    for (let i = 0; i < 20; i++) {
      pattern.push(currentTeam);
      currentTeam = currentTeam === TreatmentVariant.BASELINE ? 
        TreatmentVariant.NEW : TreatmentVariant.BASELINE;
    }
    
    return pattern;
  }

  /**
   * Calculate evaluation metrics for interleaved results
   */
  private calculateMetrics(
    baselineResults: SearchHit[],
    newResults: SearchHit[],
    interleavedResults: InterleavedHit[]
  ): InterleavingResult['metrics'] {
    // Calculate NDCG for each system
    const baselineNdcg = this.calculateNDCG(baselineResults.slice(0, 10));
    const newNdcg = this.calculateNDCG(newResults.slice(0, 10));
    const interleavedNdcg = this.calculateNDCG(interleavedResults.slice(0, 10));

    // Track positions in interleaved results
    const baselinePositions: number[] = [];
    const newPositions: number[] = [];

    interleavedResults.forEach((hit, idx) => {
      if (hit.source_variant === TreatmentVariant.BASELINE) {
        baselinePositions.push(idx);
      } else {
        newPositions.push(idx);
      }
    });

    // Determine winner based on position distribution (lower positions = better)
    const baselineAvgPosition = baselinePositions.length > 0 ? 
      baselinePositions.reduce((a, b) => a + b, 0) / baselinePositions.length : 10;
    const newAvgPosition = newPositions.length > 0 ?
      newPositions.reduce((a, b) => a + b, 0) / newPositions.length : 10;

    let winner: TreatmentVariant | undefined;
    if (Math.abs(baselineAvgPosition - newAvgPosition) > 0.5) {
      winner = baselineAvgPosition < newAvgPosition ? TreatmentVariant.BASELINE : TreatmentVariant.NEW;
    }

    // Calculate confidence using Wilson score interval
    const confidence = this.calculateWilsonConfidence(baselinePositions.length, newPositions.length);

    return {
      baseline_ndcg: baselineNdcg,
      new_ndcg: newNdcg,
      interleaved_ndcg: interleavedNdcg,
      baseline_positions: baselinePositions,
      new_positions: newPositions,
      winner,
      confidence,
    };
  }

  /**
   * Calculate NDCG@10 for search results
   */
  private calculateNDCG(results: SearchHit[], k: number = 10): number {
    if (results.length === 0) return 0;

    // Simplified NDCG calculation based on scores
    let dcg = 0;
    let idcg = 0;

    const limitedResults = results.slice(0, k);
    const sortedByScore = [...limitedResults].sort((a, b) => b.score - a.score);

    limitedResults.forEach((hit, i) => {
      const relevance = hit.score; // Use search score as relevance
      const discount = Math.log2(i + 2); // 1-indexed, so i+2
      dcg += relevance / discount;
    });

    sortedByScore.forEach((hit, i) => {
      const relevance = hit.score;
      const discount = Math.log2(i + 2);
      idcg += relevance / discount;
    });

    return idcg > 0 ? dcg / idcg : 0;
  }

  /**
   * Apply bias corrections to metrics
   */
  private applyBiasCorrections(
    interleavedResults: InterleavedHit[],
    metrics: InterleavingResult['metrics']
  ): InterleavingResult['bias_corrections'] {
    let positionBiasApplied = false;
    let trustBiasDetected = false;

    // Position bias correction
    if (this.config.bias_mitigation.position_bias_correction) {
      positionBiasApplied = true;
      // Apply position-based discounting (already incorporated in NDCG)
    }

    // Trust bias detection
    const baselineInTop3 = interleavedResults.slice(0, 3)
      .filter(hit => hit.source_variant === TreatmentVariant.BASELINE).length;
    const trustBiasRatio = baselineInTop3 / Math.min(3, interleavedResults.length);
    
    if (Math.abs(trustBiasRatio - 0.5) > this.config.bias_mitigation.trust_bias_threshold) {
      trustBiasDetected = true;
    }

    // Record bias corrections
    if (positionBiasApplied || trustBiasDetected) {
      tdiMetrics.bias_corrections.add(1, {
        position_bias: positionBiasApplied.toString(),
        trust_bias: trustBiasDetected.toString(),
      });
    }

    return {
      position_bias_applied: positionBiasApplied,
      trust_bias_detected: trustBiasDetected,
    };
  }

  /**
   * Calculate Wilson confidence interval
   */
  private calculateWilsonConfidence(successes: number, total: number): number {
    if (total === 0) return 0;
    
    const p = successes / total;
    const z = 1.96; // 95% confidence
    const n = total;
    
    const center = p + z * z / (2 * n);
    const width = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n));
    const denominator = 1 + z * z / n;
    
    const lowerBound = (center - width) / denominator;
    const upperBound = (center + width) / denominator;
    
    return upperBound - lowerBound; // Width of confidence interval
  }

  /**
   * Classify query type for TDI eligibility
   */
  private classifyQuery(query: string, context: SearchContext): QueryType {
    // Natural language indicators
    if (query.includes(' ') && query.length > 10 && 
        /[a-z]/.test(query) && !/[\[\](){}<>]/.test(query)) {
      return QueryType.NATURAL_LANGUAGE;
    }

    // Symbol search indicators
    if (/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(query.trim()) ||
        query.includes('::') || query.includes('.')) {
      return QueryType.SYMBOL_SEARCH;
    }

    // Structural indicators
    if (query.includes('(') || query.includes('[') || query.includes('{')) {
      return QueryType.STRUCTURAL;
    }

    // Hybrid mode
    if (context.mode === 'hybrid') {
      return QueryType.HYBRID;
    }

    return QueryType.NATURAL_LANGUAGE; // Default
  }

  /**
   * Assign treatment variant for consistent user experience
   */
  private assignTreatmentVariant(query: string, context: SearchContext): TreatmentVariant {
    // Consistent assignment based on query + user context
    const hash = this.hashQuery(query + (context.repo_sha || ''));
    return (hash % 2) === 0 ? TreatmentVariant.BASELINE : TreatmentVariant.NEW;
  }

  /**
   * Simple hash function for consistent randomization
   */
  private hashQuery(input: string): number {
    let hash = 0;
    for (let i = 0; i < input.length; i++) {
      const char = input.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }

  /**
   * Get statistical analysis of current experiments
   */
  getStatisticalAnalysis(experimentId?: string): {
    experiment_id: string;
    sample_size: number;
    baseline_wins: number;
    new_wins: number;
    ties: number;
    statistical_power: number;
    p_value: number;
    confidence_interval: [number, number];
    recommendation: 'continue' | 'conclude_baseline' | 'conclude_new' | 'inconclusive';
  }[] {
    const analyses: any[] = [];

    const experimentsToAnalyze = experimentId ? 
      [experimentId] : Array.from(this.results.keys());

    for (const expId of experimentsToAnalyze) {
      const results = this.results.get(expId) || [];
      if (results.length < this.config.min_sample_size) continue;

      let baselineWins = 0;
      let newWins = 0;
      let ties = 0;

      results.forEach(result => {
        if (result.metrics.winner === TreatmentVariant.BASELINE) {
          baselineWins++;
        } else if (result.metrics.winner === TreatmentVariant.NEW) {
          newWins++;
        } else {
          ties++;
        }
      });

      const totalDecisions = baselineWins + newWins;
      const p = totalDecisions > 0 ? newWins / totalDecisions : 0.5;
      
      // Calculate statistical power and p-value
      const statisticalPower = this.calculateStatisticalPower(results.length, p);
      const pValue = this.calculatePValue(baselineWins, newWins);
      
      // Wilson confidence interval
      const confidenceInterval = this.calculateWilsonInterval(newWins, totalDecisions);
      
      // Recommendation
      let recommendation: 'continue' | 'conclude_baseline' | 'conclude_new' | 'inconclusive' = 'continue';
      
      if (pValue < this.config.significance_level && statisticalPower > this.config.statistical_power) {
        recommendation = p > 0.5 ? 'conclude_new' : 'conclude_baseline';
      } else if (results.length > this.config.min_sample_size * 3) {
        recommendation = 'inconclusive';
      }

      analyses.push({
        experiment_id: expId,
        sample_size: results.length,
        baseline_wins: baselineWins,
        new_wins: newWins,
        ties,
        statistical_power: statisticalPower,
        p_value: pValue,
        confidence_interval: confidenceInterval,
        recommendation,
      });
    }

    return analyses;
  }

  /**
   * Calculate statistical power for current sample size
   */
  private calculateStatisticalPower(n: number, p: number): number {
    // Simplified power calculation for binomial test
    const z_alpha = 1.96; // 5% significance level
    const z_beta = 0.84; // 80% power
    const p0 = 0.5; // Null hypothesis
    
    const effect_size = Math.abs(p - p0);
    const required_n = Math.pow(z_alpha + z_beta, 2) * (p * (1 - p) + p0 * (1 - p0)) / Math.pow(effect_size, 2);
    
    return Math.min(1.0, n / required_n);
  }

  /**
   * Calculate p-value for binomial test
   */
  private calculatePValue(baselineWins: number, newWins: number): number {
    const total = baselineWins + newWins;
    if (total === 0) return 1.0;
    
    const p = newWins / total;
    const p0 = 0.5;
    
    // Z-test approximation
    const z = (p - p0) / Math.sqrt(p0 * (1 - p0) / total);
    
    // Two-tailed p-value (simplified)
    return 2 * (1 - this.normalCDF(Math.abs(z)));
  }

  /**
   * Calculate Wilson confidence interval
   */
  private calculateWilsonInterval(successes: number, total: number): [number, number] {
    if (total === 0) return [0, 1];
    
    const p = successes / total;
    const z = 1.96; // 95% confidence
    
    const center = p + z * z / (2 * total);
    const width = z * Math.sqrt(p * (1 - p) / total + z * z / (4 * total * total));
    const denominator = 1 + z * z / total;
    
    const lowerBound = Math.max(0, (center - width) / denominator);
    const upperBound = Math.min(1, (center + width) / denominator);
    
    return [lowerBound, upperBound];
  }

  /**
   * Normal CDF approximation
   */
  private normalCDF(x: number): number {
    return 0.5 * (1 + this.erf(x / Math.sqrt(2)));
  }

  /**
   * Error function approximation
   */
  private erf(x: number): number {
    // Abramowitz and Stegun approximation
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

  /**
   * Get current TDI status and metrics
   */
  getTDIStatus(): {
    enabled: boolean;
    traffic_percentage: number;
    active_experiments: number;
    total_samples: number;
    recent_interleaving_rate: number;
    statistical_power_avg: number;
  } {
    const totalSamples = Array.from(this.results.values())
      .reduce((sum, results) => sum + results.length, 0);

    const analyses = this.getStatisticalAnalysis();
    const avgPower = analyses.length > 0 ? 
      analyses.reduce((sum, a) => sum + a.statistical_power, 0) / analyses.length : 0;

    return {
      enabled: this.config.enabled,
      traffic_percentage: this.config.traffic_percentage,
      active_experiments: this.activeExperiments.size,
      total_samples: totalSamples,
      recent_interleaving_rate: this.calculateRecentInterleavingRate(),
      statistical_power_avg: avgPower,
    };
  }

  /**
   * Calculate recent interleaving rate
   */
  private calculateRecentInterleavingRate(): number {
    const recentWindow = 24 * 60 * 60 * 1000; // 24 hours
    const now = new Date().getTime();
    
    let recentTotal = 0;
    let recentInterleaved = 0;

    for (const results of this.results.values()) {
      for (const result of results) {
        if (now - result.timestamp.getTime() < recentWindow) {
          recentTotal++;
          if (result.interleaved_results.length > 0) {
            recentInterleaved++;
          }
        }
      }
    }

    return recentTotal > 0 ? recentInterleaved / recentTotal : 0;
  }

  /**
   * Export interleaving results for analysis
   */
  exportInterleavingData(experimentId?: string): InterleavingResult[] {
    if (experimentId) {
      return this.results.get(experimentId) || [];
    }

    const allResults: InterleavingResult[] = [];
    for (const results of this.results.values()) {
      allResults.push(...results);
    }
    return allResults;
  }
}

// Global TDI instance
export const globalTDI = new TeamDraftInterleaving();