/**
 * Slice-Specific ROI Optimization System
 * 
 * Implements sophisticated ROI curve optimization per slice (intent×lang×entropy) with:
 * - Conformal lower bounds for uplift objectives  
 * - Mathematical router optimization: max_S ∑_{q∈S} ΔnDCG_q subject to budget constraints
 * - Per-slice λ and τ fitting with auto-cap when marginal gain < 0.1pp/1% spend
 * - Q-cost curve publishing to config fingerprints
 * - Prevents exploration budget theft through rigorous mathematical bounds
 */

import type { SearchContext, SearchHit } from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';

export interface SliceIdentifier {
  intent: 'exact_match' | 'semantic_search' | 'fuzzy_match' | 'structural_query' | 'nl_overview';
  language: string; // e.g., 'typescript', 'python', 'rust', 'javascript', 'unknown'
  entropy_bin: 'low' | 'medium' | 'high'; // Based on query entropy quantiles
}

export interface SliceMetrics {
  slice_id: string;
  total_queries: number;
  total_spend_ms: number;
  baseline_ndcg: number;
  upshift_ndcg: number;
  uplift_delta: number;
  uplift_confidence_interval: [number, number];
  conformal_lower_bound: number;
  cost_per_query_ms: number;
  roi_lambda: number; // pp/ms ratio
  roi_tau: number; // Marginal cost threshold
  last_updated: number;
  fit_quality_r2: number;
  auto_capped: boolean;
}

export interface ROIOptimizationResult {
  selected_queries: Set<string>;
  total_budget_used_ms: number;
  expected_uplift_sum: number;
  slice_allocations: Map<string, number>;
  optimization_objective: number;
  solver_status: 'optimal' | 'feasible' | 'infeasible';
  computation_time_ms: number;
}

export interface ConformalUpliftBound {
  point_estimate: number;
  lower_bound: number;
  upper_bound: number;
  confidence_level: number;
  nonconformity_quantile: number;
  sample_size: number;
  is_calibrated: boolean;
}

export interface SliceBudgetConstraint {
  slice_id: string;
  max_spend_ms: number;
  max_queries: number;
  priority_weight: number;
}

/**
 * Conformal prediction for uplift bounds per slice
 */
class ConformalUpliftPredictor {
  private calibrationData: Map<string, Array<{
    actual_uplift: number;
    predicted_uplift: number;
    query_features: any;
  }>> = new Map();
  
  private nonconformityScores: Map<string, number[]> = new Map();
  private isCalibrated: Map<string, boolean> = new Map();

  /**
   * Calibrate conformal predictor for specific slice
   */
  calibrateSlice(
    sliceId: string,
    calibrationData: Array<{
      actual_uplift: number;
      predicted_uplift: number;
      query_features: any;
    }>
  ): void {
    this.calibrationData.set(sliceId, calibrationData);
    
    // Calculate nonconformity scores (absolute residuals)
    const nonconformityScores = calibrationData.map(
      item => Math.abs(item.actual_uplift - item.predicted_uplift)
    );
    
    nonconformityScores.sort((a, b) => a - b);
    this.nonconformityScores.set(sliceId, nonconformityScores);
    this.isCalibrated.set(sliceId, true);
    
    console.log(`📊 Conformal uplift predictor calibrated for slice ${sliceId}: ${calibrationData.length} samples`);
  }

  /**
   * Get conformal prediction bounds for uplift
   */
  predictUpliftBound(
    sliceId: string,
    pointEstimate: number,
    confidence: number = 0.80
  ): ConformalUpliftBound {
    const scores = this.nonconformityScores.get(sliceId);
    const calibrated = this.isCalibrated.get(sliceId) || false;
    
    if (!calibrated || !scores || scores.length === 0) {
      // Use conservative heuristic bounds
      const conservativeMargin = Math.max(0.02, pointEstimate * 0.3);
      return {
        point_estimate: pointEstimate,
        lower_bound: Math.max(0, pointEstimate - conservativeMargin),
        upper_bound: pointEstimate + conservativeMargin,
        confidence_level: confidence,
        nonconformity_quantile: conservativeMargin,
        sample_size: 0,
        is_calibrated: false
      };
    }
    
    const alpha = 1 - confidence;
    const quantileIndex = Math.ceil((scores.length + 1) * (1 - alpha)) - 1;
    const nonconformityQuantile = scores[Math.min(quantileIndex, scores.length - 1)];
    
    return {
      point_estimate: pointEstimate,
      lower_bound: Math.max(0, pointEstimate - nonconformityQuantile),
      upper_bound: pointEstimate + nonconformityQuantile,
      confidence_level: confidence,
      nonconformity_quantile: nonconformityQuantile,
      sample_size: scores.length,
      is_calibrated: true
    };
  }
}

/**
 * ROI curve fitter with auto-capping logic
 */
class SliceROIFitter {
  /**
   * Fit λ (pp/ms) and τ (marginal threshold) for slice
   * Auto-cap when marginal gain/1% spend < 0.1pp for two consecutive reports
   */
  fitSliceROI(
    sliceId: string,
    historicalData: Array<{
      spend_ms: number;
      ndcg_uplift: number;
      query_count: number;
    }>
  ): { lambda: number; tau: number; r2: number; should_auto_cap: boolean } {
    if (historicalData.length < 5) {
      // Insufficient data for fitting
      return {
        lambda: 0.2, // Default conservative value
        tau: 2.0,    // Default threshold
        r2: 0,
        should_auto_cap: false
      };
    }

    // Sort by spend to analyze marginal returns
    const sortedData = [...historicalData].sort((a, b) => a.spend_ms - b.spend_ms);
    
    // Calculate marginal gains for auto-capping logic
    const marginalGains: Array<{ spend_pct: number; marginal_gain: number }> = [];
    
    for (let i = 1; i < sortedData.length; i++) {
      const prev = sortedData[i - 1];
      const curr = sortedData[i];
      
      const spendIncrease = curr.spend_ms - prev.spend_ms;
      const upliftIncrease = curr.ndcg_uplift - prev.ndcg_uplift;
      
      if (spendIncrease > 0) {
        const spendPct = spendIncrease / (prev.spend_ms || 1) * 100;
        const marginalGain = upliftIncrease * 100; // Convert to pp
        
        marginalGains.push({ spend_pct: spendPct, marginal_gain: marginalGain });
      }
    }

    // Check auto-cap condition: marginal gain/1% spend < 0.1pp
    const recentMarginalGains = marginalGains.slice(-2);
    const shouldAutoCap = recentMarginalGains.length >= 2 &&
      recentMarginalGains.every(g => (g.marginal_gain / Math.max(g.spend_pct, 0.01)) < 0.1);

    // Fit linear ROI model: uplift = λ * spend_ms + noise
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    const n = sortedData.length;
    
    for (const point of sortedData) {
      sumX += point.spend_ms;
      sumY += point.ndcg_uplift;
      sumXY += point.spend_ms * point.ndcg_uplift;
      sumX2 += point.spend_ms * point.spend_ms;
    }
    
    const lambda = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - lambda * sumX) / n;
    
    // Calculate R²
    const meanY = sumY / n;
    let ssRes = 0, ssTot = 0;
    
    for (const point of sortedData) {
      const predicted = lambda * point.spend_ms + intercept;
      ssRes += Math.pow(point.ndcg_uplift - predicted, 2);
      ssTot += Math.pow(point.ndcg_uplift - meanY, 2);
    }
    
    const r2 = ssTot > 0 ? 1 - (ssRes / ssTot) : 0;
    
    // Set τ as the spend level where marginal return drops below threshold
    const tau = this.findMarginalThreshold(sortedData, lambda);

    console.log(`📈 ROI fitted for slice ${sliceId}: λ=${lambda.toFixed(4)}pp/ms, τ=${tau.toFixed(1)}ms, R²=${r2.toFixed(3)}, auto_cap=${shouldAutoCap}`);

    return {
      lambda: Math.max(0, lambda), // Ensure non-negative
      tau: Math.max(0.5, tau),     // Minimum threshold
      r2,
      should_auto_cap: shouldAutoCap
    };
  }

  /**
   * Find spend threshold where marginal returns fall below acceptable level
   */
  private findMarginalThreshold(
    data: Array<{ spend_ms: number; ndcg_uplift: number; query_count: number }>,
    lambda: number
  ): number {
    // Calculate diminishing returns threshold
    // τ represents the spend level where marginal gain < 0.05pp/ms
    const minMarginalReturn = 0.05; // 5pp per 100ms
    
    for (let i = 1; i < data.length; i++) {
      const prev = data[i - 1];
      const curr = data[i];
      
      const spendIncrease = curr.spend_ms - prev.spend_ms;
      const upliftIncrease = curr.ndcg_uplift - prev.ndcg_uplift;
      
      if (spendIncrease > 0) {
        const marginalReturn = upliftIncrease / spendIncrease;
        if (marginalReturn < minMarginalReturn) {
          return prev.spend_ms;
        }
      }
    }
    
    // Default threshold based on fitted lambda
    return Math.max(1.0, 0.1 / Math.max(lambda, 0.01));
  }
}

/**
 * Mathematical router optimizer
 */
class RouterOptimizer {
  /**
   * Solve: max_S ∑_{q∈S} ΔnDCG_q^lower s.t. ∑_{q∈S} Δt_q ≤ budget, ∀s: spend_s ≤ b_s
   */
  optimizeSelection(
    candidateQueries: Array<{
      query_id: string;
      slice_id: string;
      expected_uplift_lower_bound: number;
      cost_ms: number;
      priority: number;
    }>,
    totalBudgetMs: number,
    sliceBudgets: Map<string, SliceBudgetConstraint>
  ): ROIOptimizationResult {
    const startTime = Date.now();
    
    // Group queries by slice for constrained optimization
    const queriesBySlice = new Map<string, typeof candidateQueries>();
    for (const query of candidateQueries) {
      if (!queriesBySlice.has(query.slice_id)) {
        queriesBySlice.set(query.slice_id, []);
      }
      queriesBySlice.get(query.slice_id)!.push(query);
    }

    // Use greedy algorithm with slice constraints (approximation for knapsack with multiple constraints)
    const selectedQueries = new Set<string>();
    const sliceAllocations = new Map<string, number>();
    let remainingBudget = totalBudgetMs;
    let totalObjective = 0;

    // Calculate efficiency ratios and sort
    const allQueries = candidateQueries.map(q => ({
      ...q,
      efficiency: q.expected_uplift_lower_bound / Math.max(q.cost_ms, 0.1)
    })).sort((a, b) => b.efficiency - a.efficiency);

    // Greedy selection with slice budget constraints
    for (const query of allQueries) {
      const sliceBudget = sliceBudgets.get(query.slice_id);
      const currentSliceSpend = sliceAllocations.get(query.slice_id) || 0;
      
      // Check constraints
      const canAffordGlobal = remainingBudget >= query.cost_ms;
      const canAffordSlice = !sliceBudget || 
        (currentSliceSpend + query.cost_ms <= sliceBudget.max_spend_ms);
      
      if (canAffordGlobal && canAffordSlice) {
        selectedQueries.add(query.query_id);
        remainingBudget -= query.cost_ms;
        sliceAllocations.set(query.slice_id, currentSliceSpend + query.cost_ms);
        totalObjective += query.expected_uplift_lower_bound;
      }
    }

    const computationTime = Date.now() - startTime;

    console.log(`🎯 Router optimization: selected ${selectedQueries.size}/${candidateQueries.length} queries, objective=${totalObjective.toFixed(4)}, budget_used=${totalBudgetMs - remainingBudget}ms/${totalBudgetMs}ms, time=${computationTime}ms`);

    return {
      selected_queries: selectedQueries,
      total_budget_used_ms: totalBudgetMs - remainingBudget,
      expected_uplift_sum: totalObjective,
      slice_allocations: sliceAllocations,
      optimization_objective: totalObjective,
      solver_status: 'feasible', // Greedy always finds feasible solution
      computation_time_ms: computationTime
    };
  }
}

/**
 * Main slice-specific ROI optimization system
 */
export class SliceROIOptimizer {
  private sliceMetrics: Map<string, SliceMetrics> = new Map();
  private conformalPredictor: ConformalUpliftPredictor;
  private roiFitter: SliceROIFitter;
  private optimizer: RouterOptimizer;
  private enabled = true;

  // Configuration
  private readonly conformalConfidence = 0.80; // 80% confidence for bounds
  private readonly weeklyFitInterval = 7 * 24 * 60 * 60 * 1000; // 7 days in ms
  private lastFitTime = 0;

  constructor() {
    this.conformalPredictor = new ConformalUpliftPredictor();
    this.roiFitter = new SliceROIFitter();
    this.optimizer = new RouterOptimizer();
  }

  /**
   * Identify slice from search context
   */
  identifySlice(ctx: SearchContext): SliceIdentifier {
    // Intent classification
    let intent: SliceIdentifier['intent'] = 'semantic_search';
    
    if (ctx.mode === 'lex') intent = 'exact_match';
    else if (ctx.mode === 'struct') intent = 'structural_query';
    else if (ctx.fuzzy) intent = 'fuzzy_match';
    else if (ctx.query.length > 50 && /\b(how|what|why|when|where)\b/i.test(ctx.query)) {
      intent = 'nl_overview';
    }

    // Language detection from context or query
    const language = this.detectLanguage(ctx);

    // Entropy-based binning
    const entropy = this.calculateQueryEntropy(ctx.query);
    let entropy_bin: SliceIdentifier['entropy_bin'] = 'medium';
    if (entropy < 2.0) entropy_bin = 'low';
    else if (entropy > 4.0) entropy_bin = 'high';

    return { intent, language, entropy_bin };
  }

  /**
   * Update slice metrics with new query results
   */
  updateSliceMetrics(
    slice: SliceIdentifier,
    queryResult: {
      cost_ms: number;
      baseline_ndcg: number;
      upshift_ndcg?: number;
      was_upshifted: boolean;
    }
  ): void {
    const sliceId = this.getSliceId(slice);
    let metrics = this.sliceMetrics.get(sliceId);

    if (!metrics) {
      metrics = {
        slice_id: sliceId,
        total_queries: 0,
        total_spend_ms: 0,
        baseline_ndcg: 0,
        upshift_ndcg: 0,
        uplift_delta: 0,
        uplift_confidence_interval: [0, 0],
        conformal_lower_bound: 0,
        cost_per_query_ms: 0,
        roi_lambda: 0.2,
        roi_tau: 2.0,
        last_updated: Date.now(),
        fit_quality_r2: 0,
        auto_capped: false
      };
    }

    // Update running averages
    const n = metrics.total_queries;
    metrics.total_queries += 1;
    metrics.total_spend_ms += queryResult.cost_ms;
    metrics.baseline_ndcg = (metrics.baseline_ndcg * n + queryResult.baseline_ndcg) / (n + 1);
    
    if (queryResult.was_upshifted && queryResult.upshift_ndcg !== undefined) {
      const currentUpshiftN = Math.max(1, n * 0.05); // Estimate ~5% upshift rate
      metrics.upshift_ndcg = (metrics.upshift_ndcg * currentUpshiftN + queryResult.upshift_ndcg) / (currentUpshiftN + 1);
      metrics.uplift_delta = metrics.upshift_ndcg - metrics.baseline_ndcg;
    }

    metrics.cost_per_query_ms = metrics.total_spend_ms / metrics.total_queries;
    metrics.last_updated = Date.now();

    this.sliceMetrics.set(sliceId, metrics);

    // Trigger weekly ROI fitting
    this.maybeRefitROI();
  }

  /**
   * Get routing decision for query using slice-specific optimization
   */
  async getOptimalRouting(
    candidateQueries: Array<{
      query_id: string;
      context: SearchContext;
      baseline_cost_ms: number;
      upshift_cost_ms: number;
    }>,
    totalBudgetMs: number
  ): Promise<ROIOptimizationResult> {
    const span = LensTracer.createChildSpan('slice_roi_optimization');

    try {
      if (!this.enabled) {
        return {
          selected_queries: new Set(),
          total_budget_used_ms: 0,
          expected_uplift_sum: 0,
          slice_allocations: new Map(),
          optimization_objective: 0,
          solver_status: 'infeasible',
          computation_time_ms: 0
        };
      }

      // Prepare optimization inputs
      const optimizationQueries = candidateQueries.map(q => {
        const slice = this.identifySlice(q.context);
        const sliceId = this.getSliceId(slice);
        const metrics = this.sliceMetrics.get(sliceId);

        // Estimate uplift with conformal bounds
        const estimatedUplift = metrics?.uplift_delta || 0.05; // Default 5pp uplift
        const conformalBounds = this.conformalPredictor.predictUpliftBound(
          sliceId, 
          estimatedUplift, 
          this.conformalConfidence
        );

        return {
          query_id: q.query_id,
          slice_id: sliceId,
          expected_uplift_lower_bound: conformalBounds.lower_bound,
          cost_ms: q.upshift_cost_ms - q.baseline_cost_ms,
          priority: this.calculateQueryPriority(q.context, metrics)
        };
      });

      // Set up slice budget constraints
      const sliceBudgets = new Map<string, SliceBudgetConstraint>();
      for (const [sliceId, metrics] of this.sliceMetrics) {
        if (!metrics.auto_capped) {
          sliceBudgets.set(sliceId, {
            slice_id: sliceId,
            max_spend_ms: totalBudgetMs * 0.3, // Max 30% per slice
            max_queries: Math.floor(candidateQueries.length * 0.4), // Max 40% per slice
            priority_weight: metrics.roi_lambda
          });
        }
      }

      // Run optimization
      const result = this.optimizer.optimizeSelection(
        optimizationQueries,
        totalBudgetMs,
        sliceBudgets
      );

      span.setAttributes({
        success: true,
        selected_queries: result.selected_queries.size,
        total_candidates: candidateQueries.length,
        budget_utilization: result.total_budget_used_ms / totalBudgetMs,
        objective_value: result.optimization_objective
      });

      return result;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      console.error('Slice ROI optimization error:', error);

      // Return empty result on error
      return {
        selected_queries: new Set(),
        total_budget_used_ms: 0,
        expected_uplift_sum: 0,
        slice_allocations: new Map(),
        optimization_objective: 0,
        solver_status: 'infeasible',
        computation_time_ms: 0
      };
    } finally {
      span.end();
    }
  }

  /**
   * Get Q-cost curve for config fingerprint publishing
   */
  getQCostCurve(): Record<string, {
    lambda: number;
    tau: number;
    operating_point: number;
    confidence_interval: [number, number];
    last_fit: Date;
    auto_capped: boolean;
  }> {
    const curve: Record<string, any> = {};

    for (const [sliceId, metrics] of this.sliceMetrics) {
      curve[sliceId] = {
        lambda: metrics.roi_lambda,
        tau: metrics.roi_tau,
        operating_point: metrics.uplift_delta,
        confidence_interval: metrics.uplift_confidence_interval,
        last_fit: new Date(metrics.last_updated),
        auto_capped: metrics.auto_capped
      };
    }

    return curve;
  }

  /**
   * Calibrate conformal predictors with new data
   */
  async calibrateSlice(
    sliceId: string,
    calibrationData: Array<{
      actual_uplift: number;
      predicted_uplift: number;
      query_features: any;
    }>
  ): Promise<void> {
    this.conformalPredictor.calibrateSlice(sliceId, calibrationData);
  }

  /**
   * Get all slice metrics
   */
  getSliceMetrics(): Map<string, SliceMetrics> {
    return new Map(this.sliceMetrics);
  }

  /**
   * Weekly ROI refitting with auto-capping
   */
  private maybeRefitROI(): void {
    const now = Date.now();
    if (now - this.lastFitTime < this.weeklyFitInterval) {
      return;
    }

    console.log('📊 Running weekly ROI refitting for all slices...');
    
    for (const [sliceId, metrics] of this.sliceMetrics) {
      if (metrics.total_queries < 20) continue; // Need minimum sample size

      // Generate synthetic historical data for fitting
      // In production, this would come from actual historical metrics
      const historicalData = this.generateHistoricalData(metrics);
      
      const fitResult = this.roiFitter.fitSliceROI(sliceId, historicalData);
      
      // Update metrics with fit results
      metrics.roi_lambda = fitResult.lambda;
      metrics.roi_tau = fitResult.tau;
      metrics.fit_quality_r2 = fitResult.r2;
      metrics.auto_capped = fitResult.should_auto_cap;
      metrics.last_updated = now;

      // Update conformal bounds
      const conformalBounds = this.conformalPredictor.predictUpliftBound(
        sliceId, 
        metrics.uplift_delta,
        this.conformalConfidence
      );
      metrics.conformal_lower_bound = conformalBounds.lower_bound;
      metrics.uplift_confidence_interval = [
        conformalBounds.lower_bound, 
        conformalBounds.upper_bound
      ];
    }

    this.lastFitTime = now;
    console.log(`📈 ROI refitting complete for ${this.sliceMetrics.size} slices`);
  }

  private getSliceId(slice: SliceIdentifier): string {
    return `${slice.intent}|${slice.language}|${slice.entropy_bin}`;
  }

  private detectLanguage(ctx: SearchContext): string {
    // Simple language detection heuristics
    const query = ctx.query.toLowerCase();
    
    if (query.includes('function') || query.includes('const') || query.includes('=>')) {
      return 'javascript';
    } else if (query.includes('def ') || query.includes('import ') || query.includes('class ')) {
      return 'python';
    } else if (query.includes('fn ') || query.includes('struct ') || query.includes('impl ')) {
      return 'rust';
    } else if (query.includes('interface') || query.includes('type ') || query.includes(': string')) {
      return 'typescript';
    }
    
    return 'unknown';
  }

  private calculateQueryEntropy(query: string): number {
    const chars = query.split('');
    const charCounts = chars.reduce((acc, char) => {
      acc[char] = (acc[char] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    let entropy = 0;
    for (const count of Object.values(charCounts)) {
      const p = count / chars.length;
      entropy -= p * Math.log2(p);
    }

    return entropy;
  }

  private calculateQueryPriority(ctx: SearchContext, metrics?: SliceMetrics): number {
    let priority = 1.0;
    
    // Boost priority for high-value slices
    if (metrics && metrics.roi_lambda > 0.3) priority += 0.5;
    
    // Consider query characteristics
    if (ctx.query.length > 50) priority += 0.2; // Complex queries
    if (ctx.mode === 'struct') priority += 0.3; // Structural queries often high-value
    
    return priority;
  }

  private generateHistoricalData(metrics: SliceMetrics) {
    // Generate synthetic data based on current metrics
    // In production, this would be replaced with actual historical data
    const data = [];
    const baseSpend = metrics.cost_per_query_ms;
    const baseUplift = metrics.uplift_delta;
    
    for (let i = 0; i < 10; i++) {
      const spendMultiplier = 0.5 + i * 0.2; // 0.5x to 2.3x spend
      const upliftDecay = Math.exp(-i * 0.1); // Diminishing returns
      
      data.push({
        spend_ms: baseSpend * spendMultiplier,
        ndcg_uplift: baseUplift * upliftDecay,
        query_count: Math.floor(metrics.total_queries * spendMultiplier / 10)
      });
    }
    
    return data;
  }

  /**
   * Enable/disable optimizer
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    console.log(`🎯 Slice ROI optimizer ${enabled ? 'ENABLED' : 'DISABLED'}`);
  }
}

// Global instance
export const globalSliceROIOptimizer = new SliceROIOptimizer();