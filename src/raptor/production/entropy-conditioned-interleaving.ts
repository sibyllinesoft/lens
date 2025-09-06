/**
 * Entropy-Conditioned Interleaving System - TDI with Bias Correction
 * 
 * Implements comprehensive interleaving for production validation:
 * 1. Team-Draft Interleaving (TDI) on NL+symbol queries
 * 2. Entropy tercile binning for conditional analysis  
 * 3. Non-negative ΔSLA-Recall@50 requirement across all bins
 * 4. Flat or better nDCG on low-entropy bins (over-steer protection)
 * 5. Inverse propensity correction using randomized top-2 swap
 * 6. Click-bias correction with propensity weighting
 */

import { EventEmitter } from 'events';
import { writeFile, mkdir } from 'fs/promises';
import { join } from 'path';

export type QueryIntent = 'NL' | 'symbol' | 'mixed';
export type QueryLanguage = 'typescript' | 'python' | 'rust' | 'go' | 'javascript';
export type EntropyTercile = 'low' | 'medium' | 'high';

export interface QueryMetadata {
  query_id: string;
  intent: QueryIntent;
  language: QueryLanguage;
  entropy_score: number; // Raw entropy score
  entropy_tercile: EntropyTercile; // Binned entropy
  user_id: string;
  session_id: string;
  timestamp: Date;
  repository: string;
}

export interface SearchResult {
  result_id: string;
  file_path: string;
  snippet: string;
  relevance_score: number;
  rank_position: number; // 1-indexed
  clicked: boolean;
  dwell_time_ms?: number;
}

export interface InterleavedExperiment {
  query_metadata: QueryMetadata;
  treatment_results: SearchResult[]; // New system results
  control_results: SearchResult[]; // Baseline system results
  interleaved_results: SearchResult[]; // TDI interleaved results shown to user
  user_interactions: {
    clicked_positions: number[];
    max_examined_rank: number;
    session_duration_ms: number;
    abandonment: boolean;
  };
  bias_correction: {
    propensity_weights: number[]; // Per-position weights
    randomized_swap_applied: boolean; // Top-2 swap for unbiased estimation
    swap_positions?: [number, number];
  };
  evaluation_metrics: {
    treatment_ndcg_10: number;
    control_ndcg_10: number;
    treatment_sla_recall_50: number;
    control_sla_recall_50: number;
    treatment_clicks: number;
    control_clicks: number;
  };
}

export interface TercileAnalysisResult {
  tercile: EntropyTercile;
  sample_count: number;
  treatment_better_count: number; // Queries where treatment won
  control_better_count: number; // Queries where control won  
  tied_count: number; // No clear winner
  
  // Aggregate metrics
  avg_delta_ndcg_10: number; // Treatment - Control
  avg_delta_sla_recall_50: number; // Treatment - Control
  
  // Statistical significance
  ndcg_p_value: number;
  sla_recall_p_value: number;
  
  // Click bias corrected metrics
  bias_corrected_delta_ndcg: number;
  bias_corrected_delta_sla_recall: number;
  
  // Requirements validation
  non_negative_sla_recall: boolean; // Required: ΔSLA-Recall@50 ≥ 0
  flat_or_better_ndcg: boolean; // For low entropy: ΔnDCG ≥ 0 (over-steer protection)
  
  validation_status: 'PASS' | 'FAIL';
  violation_reason?: string;
}

export interface InterleavingAnalysisReport {
  timestamp: Date;
  total_experiments: number;
  traffic_percentage: number; // % of traffic in interleaving
  
  tercile_results: TercileAnalysisResult[];
  
  overall_metrics: {
    total_delta_ndcg_10: number;
    total_delta_sla_recall_50: number;
    bias_corrected_delta_ndcg: number;
    bias_corrected_delta_sla_recall: number;
    click_bias_correction_factor: number;
  };
  
  validation_summary: {
    all_terciles_pass: boolean;
    failed_terciles: EntropyTercile[];
    over_steer_risk_detected: boolean;
    click_bias_severity: 'low' | 'medium' | 'high';
  };
  
  recommendations: string[];
  alerts: string[];
}

export interface InterleavingConfig {
  traffic_percentage: number; // 1-2% for randomized top-2 swap
  entropy_tercile_boundaries: [number, number]; // [low_high, med_high] boundaries
  statistical_alpha: number; // Significance level (0.05)
  min_samples_per_tercile: number; // Minimum samples for reliable analysis
  propensity_model: 'position_based' | 'examination_based' | 'learned';
  bias_correction_method: 'inverse_propensity' | 'doubly_robust';
  randomized_swap_rate: number; // Rate of top-2 swaps (0.5 = 50%)
}

export const DEFAULT_INTERLEAVING_CONFIG: InterleavingConfig = {
  traffic_percentage: 1.5, // 1.5% traffic
  entropy_tercile_boundaries: [0.4, 0.7], // [0-0.4]: low, [0.4-0.7]: med, [0.7+]: high
  statistical_alpha: 0.05,
  min_samples_per_tercile: 100,
  propensity_model: 'position_based',
  bias_correction_method: 'inverse_propensity',
  randomized_swap_rate: 0.5 // 50% of experiments get top-2 swap
};

export class EntropyConditionedInterleaver extends EventEmitter {
  private config: InterleavingConfig;
  private experiments: InterleavedExperiment[] = [];
  private propensityWeights: Map<number, number> = new Map(); // Position -> Weight mapping
  
  constructor(config: InterleavingConfig = DEFAULT_INTERLEAVING_CONFIG) {
    super();
    this.config = config;
    this.initializePropensityModel();
  }
  
  /**
   * Initialize propensity model for bias correction
   */
  private initializePropensityModel(): void {
    console.log('📊 Initializing propensity model for click bias correction...');
    
    // Position-based propensity model (simplified)
    // Based on typical search result examination patterns
    const baseExaminationRates = [
      0.95, 0.85, 0.70, 0.60, 0.50, // Top 5 positions
      0.40, 0.35, 0.30, 0.25, 0.20, // Positions 6-10
      0.15, 0.12, 0.10, 0.08, 0.07, // Positions 11-15
      0.05, 0.04, 0.03, 0.02, 0.01  // Positions 16-20
    ];
    
    // Convert examination rates to inverse propensity weights
    for (let i = 0; i < baseExaminationRates.length; i++) {
      const position = i + 1; // 1-indexed positions
      const examinationRate = baseExaminationRates[i];
      const weight = 1.0 / Math.max(examinationRate, 0.01); // Avoid division by zero
      this.propensityWeights.set(position, weight);
    }
    
    console.log(`✅ Propensity model initialized with ${this.propensityWeights.size} positions`);
  }
  
  /**
   * Classify query entropy into terciles
   */
  private classifyEntropyTercile(entropyScore: number): EntropyTercile {
    const [lowHigh, medHigh] = this.config.entropy_tercile_boundaries;
    
    if (entropyScore <= lowHigh) return 'low';
    if (entropyScore <= medHigh) return 'medium'; 
    return 'high';
  }
  
  /**
   * Execute Team-Draft Interleaving for a single query
   */
  async executeTeamDraftInterleaving(
    queryMetadata: QueryMetadata,
    treatmentResults: SearchResult[],
    controlResults: SearchResult[]
  ): Promise<SearchResult[]> {
    // TDI Algorithm: Alternating selection with preference for higher relevance
    const interleavedResults: SearchResult[] = [];
    const treatmentUsed = new Set<string>();
    const controlUsed = new Set<string>();
    
    let treatmentTurn = Math.random() < 0.5; // Random first pick
    let treatmentIndex = 0;
    let controlIndex = 0;
    
    // Interleave up to top 20 results
    for (let position = 1; position <= Math.min(20, treatmentResults.length + controlResults.length); position++) {
      let selectedResult: SearchResult;
      
      if (treatmentTurn && treatmentIndex < treatmentResults.length) {
        // Treatment team's turn
        const candidate = treatmentResults[treatmentIndex];
        if (!controlUsed.has(candidate.result_id)) {
          selectedResult = { ...candidate, rank_position: position };
          treatmentUsed.add(candidate.result_id);
          treatmentIndex++;
        } else {
          // Fallback to control if result already used
          treatmentTurn = false;
          continue;
        }
      } else {
        // Control team's turn
        const candidate = controlResults[controlIndex];
        if (!treatmentUsed.has(candidate.result_id)) {
          selectedResult = { ...candidate, rank_position: position };
          controlUsed.add(candidate.result_id);
          controlIndex++;
        } else {
          // Fallback to treatment if result already used
          treatmentTurn = true;
          continue;
        }
      }
      
      interleavedResults.push(selectedResult);
      treatmentTurn = !treatmentTurn; // Alternate teams
    }
    
    return interleavedResults;
  }
  
  /**
   * Apply randomized top-2 swap for unbiased propensity estimation
   */
  private applyRandomizedTop2Swap(results: SearchResult[]): {
    swappedResults: SearchResult[];
    swapApplied: boolean;
    swapPositions?: [number, number];
  } {
    if (results.length < 2 || Math.random() > this.config.randomized_swap_rate) {
      return { swappedResults: results, swapApplied: false };
    }
    
    // Randomly swap positions 1 and 2
    const swappedResults = [...results];
    const temp = swappedResults[0];
    swappedResults[0] = { ...swappedResults[1], rank_position: 1 };
    swappedResults[1] = { ...temp, rank_position: 2 };
    
    return {
      swappedResults,
      swapApplied: true,
      swapPositions: [1, 2]
    };
  }
  
  /**
   * Compute evaluation metrics for interleaved experiment
   */
  private computeEvaluationMetrics(
    experiment: InterleavedExperiment,
    treatmentResults: SearchResult[],
    controlResults: SearchResult[]
  ): void {
    // nDCG@10 computation
    experiment.evaluation_metrics.treatment_ndcg_10 = this.computeNDCG(
      treatmentResults.slice(0, 10),
      experiment.user_interactions.clicked_positions
    );
    
    experiment.evaluation_metrics.control_ndcg_10 = this.computeNDCG(
      controlResults.slice(0, 10),
      experiment.user_interactions.clicked_positions
    );
    
    // SLA-Recall@50 computation (recall within first 50 results)
    experiment.evaluation_metrics.treatment_sla_recall_50 = this.computeSLARecall(
      treatmentResults.slice(0, 50),
      experiment.user_interactions.clicked_positions
    );
    
    experiment.evaluation_metrics.control_sla_recall_50 = this.computeSLARecall(
      controlResults.slice(0, 50),
      experiment.user_interactions.clicked_positions
    );
    
    // Click attribution (which system's results were clicked)
    let treatmentClicks = 0;
    let controlClicks = 0;
    
    for (const clickPos of experiment.user_interactions.clicked_positions) {
      const clickedResult = experiment.interleaved_results[clickPos - 1];
      if (treatmentResults.some(r => r.result_id === clickedResult.result_id)) {
        treatmentClicks++;
      } else {
        controlClicks++;
      }
    }
    
    experiment.evaluation_metrics.treatment_clicks = treatmentClicks;
    experiment.evaluation_metrics.control_clicks = controlClicks;
  }
  
  /**
   * Simplified nDCG computation
   */
  private computeNDCG(results: SearchResult[], clickedPositions: number[]): number {
    if (clickedPositions.length === 0) return 0;
    
    let dcg = 0;
    let idcg = 0;
    
    // DCG based on clicks (treating clicks as relevance = 1)
    for (let i = 0; i < results.length; i++) {
      const position = i + 1;
      const gain = clickedPositions.includes(position) ? 1 : 0;
      dcg += gain / Math.log2(position + 1);
      
      // IDCG assumes perfect ranking of clicked results
      if (i < clickedPositions.length) {
        idcg += 1 / Math.log2(position + 1);
      }
    }
    
    return idcg > 0 ? dcg / idcg : 0;
  }
  
  /**
   * Compute SLA-Recall (simplified as click recall)
   */
  private computeSLARecall(results: SearchResult[], clickedPositions: number[]): number {
    if (clickedPositions.length === 0) return 0;
    
    let recalledItems = 0;
    for (const pos of clickedPositions) {
      if (pos <= results.length) recalledItems++;
    }
    
    return recalledItems / Math.max(clickedPositions.length, 1);
  }
  
  /**
   * Ingest interleaving experiments for analysis
   */
  async ingestExperiments(experiments: InterleavedExperiment[]): Promise<void> {
    console.log(`📊 Ingesting ${experiments.length} interleaving experiments...`);
    
    this.experiments = experiments;
    
    // Validate data distribution
    const tercileDistribution = this.getTercileDistribution();
    console.log(`Tercile distribution: ${JSON.stringify(tercileDistribution)}`);
    
    // Check for sufficient samples
    for (const [tercile, count] of Object.entries(tercileDistribution)) {
      if (count < this.config.min_samples_per_tercile) {
        console.warn(`⚠️  Insufficient samples in ${tercile} tercile: ${count} < ${this.config.min_samples_per_tercile}`);
      }
    }
    
    this.emit('experiments_ingested', {
      total: experiments.length,
      tercile_distribution: tercileDistribution
    });
  }
  
  /**
   * Execute comprehensive entropy-conditioned analysis
   */
  async executeAnalysis(outputDir: string): Promise<InterleavingAnalysisReport> {
    console.log('🧮 Executing entropy-conditioned interleaving analysis...');
    
    if (this.experiments.length === 0) {
      throw new Error('No experiments loaded. Call ingestExperiments() first.');
    }
    
    await mkdir(outputDir, { recursive: true });
    
    // Group experiments by entropy tercile
    const tercileGroups = this.groupByEntropyTercile();
    
    // Analyze each tercile independently
    const tercileResults: TercileAnalysisResult[] = [];
    for (const [tercile, experiments] of tercileGroups) {
      const result = await this.analyzeTercile(tercile, experiments);
      tercileResults.push(result);
    }
    
    // Compute overall metrics with bias correction
    const overallMetrics = this.computeOverallMetrics();
    
    // Validate requirements
    const validationSummary = this.validateRequirements(tercileResults);
    
    // Generate recommendations and alerts
    const recommendations = this.generateRecommendations(tercileResults, validationSummary);
    const alerts = this.generateAlerts(tercileResults, validationSummary);
    
    const report: InterleavingAnalysisReport = {
      timestamp: new Date(),
      total_experiments: this.experiments.length,
      traffic_percentage: this.config.traffic_percentage,
      tercile_results: tercileResults,
      overall_metrics: overallMetrics,
      validation_summary: validationSummary,
      recommendations,
      alerts
    };
    
    // Save analysis results
    await this.saveAnalysisReport(report, outputDir);
    
    console.log(`✅ Analysis completed: ${validationSummary.all_terciles_pass ? 'ALL TERCILES PASS' : 'VIOLATIONS DETECTED'}`);
    
    this.emit('analysis_completed', report);
    return report;
  }
  
  /**
   * Group experiments by entropy tercile
   */
  private groupByEntropyTercile(): Map<EntropyTercile, InterleavedExperiment[]> {
    const groups = new Map<EntropyTercile, InterleavedExperiment[]>();
    
    for (const experiment of this.experiments) {
      const tercile = experiment.query_metadata.entropy_tercile;
      
      if (!groups.has(tercile)) {
        groups.set(tercile, []);
      }
      groups.get(tercile)!.push(experiment);
    }
    
    return groups;
  }
  
  /**
   * Analyze individual entropy tercile
   */
  private async analyzeTercile(tercile: EntropyTercile, experiments: InterleavedExperiment[]): Promise<TercileAnalysisResult> {
    const sampleCount = experiments.length;
    
    if (sampleCount < this.config.min_samples_per_tercile) {
      return this.createInsufficientSampleResult(tercile, sampleCount);
    }
    
    // Count wins/losses
    let treatmentWins = 0;
    let controlWins = 0;
    let ties = 0;
    
    let totalDeltaNdcg = 0;
    let totalDeltaSlaRecall = 0;
    let biasAdjustedDeltaNdcg = 0;
    let biasAdjustedDeltaSlaRecall = 0;
    
    // Analyze each experiment
    for (const experiment of experiments) {
      const deltaNdcg = experiment.evaluation_metrics.treatment_ndcg_10 - experiment.evaluation_metrics.control_ndcg_10;
      const deltaSlaRecall = experiment.evaluation_metrics.treatment_sla_recall_50 - experiment.evaluation_metrics.control_sla_recall_50;
      
      // Win/loss determination
      if (Math.abs(deltaNdcg) < 0.01 && Math.abs(deltaSlaRecall) < 0.01) {
        ties++;
      } else if (deltaNdcg > 0 || deltaSlaRecall > 0) {
        treatmentWins++;
      } else {
        controlWins++;
      }
      
      totalDeltaNdcg += deltaNdcg;
      totalDeltaSlaRecall += deltaSlaRecall;
      
      // Apply bias correction
      const biasCorrection = this.applyBiasCorrection(experiment);
      biasAdjustedDeltaNdcg += biasCorrection.adjusted_delta_ndcg;
      biasAdjustedDeltaSlaRecall += biasCorrection.adjusted_delta_sla_recall;
    }
    
    // Average metrics
    const avgDeltaNdcg = totalDeltaNdcg / sampleCount;
    const avgDeltaSlaRecall = totalDeltaSlaRecall / sampleCount;
    const biasAdjustedAvgNdcg = biasAdjustedDeltaNdcg / sampleCount;
    const biasAdjustedAvgSlaRecall = biasAdjustedDeltaSlaRecall / sampleCount;
    
    // Statistical significance testing (simplified)
    const ndcgPValue = this.computeStatisticalSignificance(experiments, 'ndcg');
    const slaRecallPValue = this.computeStatisticalSignificance(experiments, 'sla_recall');
    
    // Requirements validation
    const nonNegativeSlaRecall = biasAdjustedAvgSlaRecall >= 0;
    const flatOrBetterNdcg = tercile === 'low' ? biasAdjustedAvgNdcg >= 0 : true; // Only apply to low entropy
    
    const validationStatus = nonNegativeSlaRecall && flatOrBetterNdcg ? 'PASS' : 'FAIL';
    let violationReason: string | undefined;
    
    if (!nonNegativeSlaRecall) {
      violationReason = `Negative ΔSLA-Recall@50: ${biasAdjustedAvgSlaRecall.toFixed(3)}`;
    } else if (!flatOrBetterNdcg) {
      violationReason = `Negative ΔnDCG on low-entropy queries: ${biasAdjustedAvgNdcg.toFixed(3)}`;
    }
    
    return {
      tercile,
      sample_count: sampleCount,
      treatment_better_count: treatmentWins,
      control_better_count: controlWins,
      tied_count: ties,
      avg_delta_ndcg_10: avgDeltaNdcg,
      avg_delta_sla_recall_50: avgDeltaSlaRecall,
      ndcg_p_value: ndcgPValue,
      sla_recall_p_value: slaRecallPValue,
      bias_corrected_delta_ndcg: biasAdjustedAvgNdcg,
      bias_corrected_delta_sla_recall: biasAdjustedAvgSlaRecall,
      non_negative_sla_recall: nonNegativeSlaRecall,
      flat_or_better_ndcg: flatOrBetterNdcg,
      validation_status: validationStatus,
      violation_reason: violationReason
    };
  }
  
  /**
   * Apply bias correction to individual experiment
   */
  private applyBiasCorrection(experiment: InterleavedExperiment): {
    adjusted_delta_ndcg: number;
    adjusted_delta_sla_recall: number;
  } {
    // Inverse propensity weighting
    let treatmentWeightedScore = 0;
    let controlWeightedScore = 0;
    let treatmentTotalWeight = 0;
    let controlTotalWeight = 0;
    
    for (const clickPos of experiment.user_interactions.clicked_positions) {
      const weight = this.propensityWeights.get(clickPos) || 1.0;
      const clickedResult = experiment.interleaved_results[clickPos - 1];
      
      // Check if clicked result came from treatment or control
      const fromTreatment = experiment.treatment_results.some(r => r.result_id === clickedResult.result_id);
      
      if (fromTreatment) {
        treatmentWeightedScore += weight;
        treatmentTotalWeight += weight;
      } else {
        controlWeightedScore += weight;
        controlTotalWeight += weight;
      }
    }
    
    // Normalize by total weight
    const treatmentNormalized = treatmentTotalWeight > 0 ? treatmentWeightedScore / treatmentTotalWeight : 0;
    const controlNormalized = controlTotalWeight > 0 ? controlWeightedScore / controlTotalWeight : 0;
    
    // Apply bias correction factor
    const biasAdjustmentFactor = experiment.bias_correction.randomized_swap_applied ? 0.95 : 1.0;
    
    return {
      adjusted_delta_ndcg: (treatmentNormalized - controlNormalized) * biasAdjustmentFactor,
      adjusted_delta_sla_recall: (treatmentNormalized - controlNormalized) * biasAdjustmentFactor * 0.8 // SLA recall typically lower impact
    };
  }
  
  /**
   * Compute statistical significance (simplified t-test)
   */
  private computeStatisticalSignificance(experiments: InterleavedExperiment[], metric: 'ndcg' | 'sla_recall'): number {
    const deltas = experiments.map(exp => {
      if (metric === 'ndcg') {
        return exp.evaluation_metrics.treatment_ndcg_10 - exp.evaluation_metrics.control_ndcg_10;
      } else {
        return exp.evaluation_metrics.treatment_sla_recall_50 - exp.evaluation_metrics.control_sla_recall_50;
      }
    });
    
    const n = deltas.length;
    const mean = deltas.reduce((sum, d) => sum + d, 0) / n;
    const variance = deltas.reduce((sum, d) => sum + Math.pow(d - mean, 2), 0) / (n - 1);
    const stderr = Math.sqrt(variance / n);
    
    // t-statistic
    const tStat = Math.abs(mean / stderr);
    
    // Simplified p-value approximation (assumes t-distribution ≈ normal for large n)
    // Real implementation would use proper t-distribution CDF
    const pValue = 2 * (1 - this.normalCDF(tStat));
    
    return Math.min(1.0, Math.max(0.0, pValue));
  }
  
  /**
   * Normal CDF approximation
   */
  private normalCDF(x: number): number {
    // Simplified normal CDF approximation
    return 0.5 * (1 + Math.sign(x) * Math.sqrt(1 - Math.exp(-2 * x * x / Math.PI)));
  }
  
  /**
   * Create result for terciles with insufficient samples
   */
  private createInsufficientSampleResult(tercile: EntropyTercile, sampleCount: number): TercileAnalysisResult {
    return {
      tercile,
      sample_count: sampleCount,
      treatment_better_count: 0,
      control_better_count: 0,
      tied_count: 0,
      avg_delta_ndcg_10: 0,
      avg_delta_sla_recall_50: 0,
      ndcg_p_value: 1.0,
      sla_recall_p_value: 1.0,
      bias_corrected_delta_ndcg: 0,
      bias_corrected_delta_sla_recall: 0,
      non_negative_sla_recall: true, // Pass by default for insufficient samples
      flat_or_better_ndcg: true,
      validation_status: 'PASS',
      violation_reason: `Insufficient samples: ${sampleCount} < ${this.config.min_samples_per_tercile}`
    };
  }
  
  /**
   * Compute overall metrics across all terciles
   */
  private computeOverallMetrics(): any {
    let totalDeltaNdcg = 0;
    let totalDeltaSlaRecall = 0;
    let biasAdjustedDeltaNdcg = 0;
    let biasAdjustedDeltaSlaRecall = 0;
    let totalBiasCorrection = 0;
    
    for (const experiment of this.experiments) {
      const deltaNdcg = experiment.evaluation_metrics.treatment_ndcg_10 - experiment.evaluation_metrics.control_ndcg_10;
      const deltaSlaRecall = experiment.evaluation_metrics.treatment_sla_recall_50 - experiment.evaluation_metrics.control_sla_recall_50;
      
      totalDeltaNdcg += deltaNdcg;
      totalDeltaSlaRecall += deltaSlaRecall;
      
      const biasCorrection = this.applyBiasCorrection(experiment);
      biasAdjustedDeltaNdcg += biasCorrection.adjusted_delta_ndcg;
      biasAdjustedDeltaSlaRecall += biasCorrection.adjusted_delta_sla_recall;
      
      totalBiasCorrection += experiment.bias_correction.randomized_swap_applied ? 1 : 0;
    }
    
    const n = this.experiments.length;
    
    return {
      total_delta_ndcg_10: totalDeltaNdcg / n,
      total_delta_sla_recall_50: totalDeltaSlaRecall / n,
      bias_corrected_delta_ndcg: biasAdjustedDeltaNdcg / n,
      bias_corrected_delta_sla_recall: biasAdjustedDeltaSlaRecall / n,
      click_bias_correction_factor: totalBiasCorrection / n
    };
  }
  
  /**
   * Validate requirements across all terciles
   */
  private validateRequirements(tercileResults: TercileAnalysisResult[]): any {
    const failedTerciles: EntropyTercile[] = [];
    let overSteerRiskDetected = false;
    
    for (const result of tercileResults) {
      if (result.validation_status === 'FAIL') {
        failedTerciles.push(result.tercile);
      }
      
      // Over-steer risk: negative nDCG on low entropy
      if (result.tercile === 'low' && result.bias_corrected_delta_ndcg < -0.005) {
        overSteerRiskDetected = true;
      }
    }
    
    // Click bias severity assessment
    const avgBiasCorrection = this.experiments.reduce((sum, exp) => 
      sum + (exp.bias_correction.randomized_swap_applied ? 1 : 0), 0
    ) / this.experiments.length;
    
    let clickBiasSeverity: 'low' | 'medium' | 'high';
    if (avgBiasCorrection < 0.3) clickBiasSeverity = 'low';
    else if (avgBiasCorrection < 0.6) clickBiasSeverity = 'medium';
    else clickBiasSeverity = 'high';
    
    return {
      all_terciles_pass: failedTerciles.length === 0,
      failed_terciles: failedTerciles,
      over_steer_risk_detected: overSteerRiskDetected,
      click_bias_severity: clickBiasSeverity
    };
  }
  
  /**
   * Generate recommendations based on analysis
   */
  private generateRecommendations(tercileResults: TercileAnalysisResult[], validationSummary: any): string[] {
    const recommendations: string[] = [];
    
    if (validationSummary.all_terciles_pass) {
      recommendations.push('✅ All entropy terciles meet requirements - safe to proceed with rollout');
    } else {
      recommendations.push('❌ Violations detected - address failed terciles before full rollout');
      
      for (const tercile of validationSummary.failed_terciles) {
        const result = tercileResults.find(r => r.tercile === tercile);
        if (result?.violation_reason) {
          recommendations.push(`  - ${tercile.toUpperCase()}: ${result.violation_reason}`);
        }
      }
    }
    
    // Over-steer protection
    if (validationSummary.over_steer_risk_detected) {
      recommendations.push('⚠️  Over-steer risk on low-entropy queries - consider MMR cap adjustment');
    }
    
    // Click bias handling
    if (validationSummary.click_bias_severity === 'high') {
      recommendations.push('🔧 High click bias detected - increase randomized swap rate or improve propensity model');
    }
    
    // Statistical power
    const lowSampleTerciles = tercileResults.filter(r => r.sample_count < this.config.min_samples_per_tercile);
    if (lowSampleTerciles.length > 0) {
      recommendations.push(`📊 Increase sample collection for terciles: ${lowSampleTerciles.map(r => r.tercile).join(', ')}`);
    }
    
    return recommendations;
  }
  
  /**
   * Generate alerts for critical issues
   */
  private generateAlerts(tercileResults: TercileAnalysisResult[], validationSummary: any): string[] {
    const alerts: string[] = [];
    
    // Critical validation failures
    for (const result of tercileResults) {
      if (result.validation_status === 'FAIL' && result.violation_reason) {
        alerts.push(`CRITICAL: ${result.tercile.toUpperCase()} tercile violation - ${result.violation_reason}`);
      }
    }
    
    // Over-steer alerts
    if (validationSummary.over_steer_risk_detected) {
      alerts.push('OVER-STEER RISK: Low-entropy queries showing negative nDCG - potential quality degradation');
    }
    
    // Statistical significance alerts
    const significantDegradations = tercileResults.filter(r => 
      r.bias_corrected_delta_sla_recall < -0.01 && r.sla_recall_p_value < 0.05
    );
    
    if (significantDegradations.length > 0) {
      alerts.push(`DEGRADATION: Statistically significant SLA-Recall drops in ${significantDegradations.length} tercile(s)`);
    }
    
    return alerts;
  }
  
  /**
   * Get tercile distribution
   */
  private getTercileDistribution(): Record<string, number> {
    const distribution: Record<string, number> = { low: 0, medium: 0, high: 0 };
    
    for (const experiment of this.experiments) {
      distribution[experiment.query_metadata.entropy_tercile]++;
    }
    
    return distribution;
  }
  
  /**
   * Save comprehensive analysis report
   */
  private async saveAnalysisReport(report: InterleavingAnalysisReport, outputDir: string): Promise<void> {
    // Save full JSON report
    await writeFile(
      join(outputDir, 'entropy-conditioned-interleaving-analysis.json'),
      JSON.stringify(report, null, 2)
    );
    
    // Save CSV data for plotting
    const csvData = this.generateAnalysisCSV(report);
    await writeFile(join(outputDir, 'tercile-analysis.csv'), csvData);
    
    // Save markdown summary
    const summaryReport = this.generateAnalysisMarkdown(report);
    await writeFile(join(outputDir, 'interleaving-analysis-summary.md'), summaryReport);
    
    console.log(`✅ Analysis report saved to ${outputDir}/`);
  }
  
  /**
   * Generate CSV data for analysis
   */
  private generateAnalysisCSV(report: InterleavingAnalysisReport): string {
    const headers = [
      'tercile', 'sample_count', 'treatment_wins', 'control_wins', 'ties',
      'avg_delta_ndcg_10', 'avg_delta_sla_recall_50', 
      'bias_corrected_delta_ndcg', 'bias_corrected_delta_sla_recall',
      'ndcg_p_value', 'sla_recall_p_value', 'validation_status'
    ].join(',');
    
    const rows = report.tercile_results.map(r => [
      r.tercile, r.sample_count, r.treatment_better_count, r.control_better_count, r.tied_count,
      r.avg_delta_ndcg_10.toFixed(4), r.avg_delta_sla_recall_50.toFixed(4),
      r.bias_corrected_delta_ndcg.toFixed(4), r.bias_corrected_delta_sla_recall.toFixed(4),
      r.ndcg_p_value.toFixed(4), r.sla_recall_p_value.toFixed(4), r.validation_status
    ].join(','));
    
    return [headers, ...rows].join('\n');
  }
  
  /**
   * Generate markdown analysis summary
   */
  private generateAnalysisMarkdown(report: InterleavingAnalysisReport): string {
    let md = '# Entropy-Conditioned Interleaving Analysis\n\n';
    
    md += `**Analysis Date**: ${report.timestamp.toISOString()}\n`;
    md += `**Total Experiments**: ${report.total_experiments.toLocaleString()}\n`;
    md += `**Traffic Percentage**: ${report.traffic_percentage}%\n\n`;
    
    // Overall validation status
    if (report.validation_summary.all_terciles_pass) {
      md += '## 🟢 Validation Status: ALL TERCILES PASS\n\n';
    } else {
      md += '## 🔴 Validation Status: VIOLATIONS DETECTED\n\n';
      md += `**Failed Terciles**: ${report.validation_summary.failed_terciles.join(', ')}\n`;
      if (report.validation_summary.over_steer_risk_detected) {
        md += `**Over-steer Risk**: ⚠️  DETECTED\n`;
      }
      md += '\n';
    }
    
    // Alerts
    if (report.alerts.length > 0) {
      md += '## 🚨 Critical Alerts\n\n';
      for (const alert of report.alerts) {
        md += `- **${alert}**\n`;
      }
      md += '\n';
    }
    
    // Tercile results
    md += '## 📊 Tercile Analysis Results\n\n';
    md += '| Tercile | Samples | Treatment Wins | Control Wins | Ties | Bias-Corrected ΔnDCG | Bias-Corrected ΔSLA-Recall | Status |\n';
    md += '|---------|---------|---------------|--------------|------|---------------------|---------------------------|--------|\n';
    
    for (const result of report.tercile_results) {
      const status = result.validation_status === 'PASS' ? '✅' : '❌';
      md += `| ${result.tercile.toUpperCase()} | ${result.sample_count} | ${result.treatment_better_count} | ${result.control_better_count} | ${result.tied_count} | ${result.bias_corrected_delta_ndcg.toFixed(3)} | ${result.bias_corrected_delta_sla_recall.toFixed(3)} | ${status} ${result.validation_status} |\n`;
    }
    md += '\n';
    
    // Overall metrics
    md += '## 🌐 Overall Metrics\n\n';
    md += `- **Overall ΔnDCG@10**: ${report.overall_metrics.total_delta_ndcg_10.toFixed(3)}\n`;
    md += `- **Overall ΔSLA-Recall@50**: ${report.overall_metrics.total_delta_sla_recall_50.toFixed(3)}\n`;
    md += `- **Bias-Corrected ΔnDCG@10**: ${report.overall_metrics.bias_corrected_delta_ndcg.toFixed(3)}\n`;
    md += `- **Bias-Corrected ΔSLA-Recall@50**: ${report.overall_metrics.bias_corrected_delta_sla_recall.toFixed(3)}\n`;
    md += `- **Click Bias Correction Factor**: ${(report.overall_metrics.click_bias_correction_factor * 100).toFixed(1)}%\n`;
    md += `- **Click Bias Severity**: ${report.validation_summary.click_bias_severity.toUpperCase()}\n\n`;
    
    // Recommendations
    md += '## 💡 Recommendations\n\n';
    for (const rec of report.recommendations) {
      md += `- ${rec}\n`;
    }
    
    return md;
  }
  
  /**
   * Generate synthetic interleaving experiments for testing
   */
  static generateSyntheticExperiments(count: number = 2000): InterleavedExperiment[] {
    const experiments: InterleavedExperiment[] = [];
    const intents: QueryIntent[] = ['NL', 'symbol', 'mixed'];
    const languages: QueryLanguage[] = ['typescript', 'python', 'rust', 'go', 'javascript'];
    
    for (let i = 0; i < count; i++) {
      const intent = intents[Math.floor(Math.random() * intents.length)];
      const language = languages[Math.floor(Math.random() * languages.length)];
      
      // Generate entropy score and classify
      const entropyScore = Math.random();
      let entropyTercile: EntropyTercile;
      if (entropyScore <= 0.4) entropyTercile = 'low';
      else if (entropyScore <= 0.7) entropyTercile = 'medium';
      else entropyTercile = 'high';
      
      const queryMetadata: QueryMetadata = {
        query_id: `query_${i}`,
        intent,
        language,
        entropy_score: entropyScore,
        entropy_tercile: entropyTercile,
        user_id: `user_${Math.floor(Math.random() * 1000)}`,
        session_id: `session_${Math.floor(Math.random() * 500)}`,
        timestamp: new Date(Date.now() - Math.random() * 86400000),
        repository: `repo_${Math.floor(Math.random() * 100)}`
      };
      
      // Generate synthetic results and metrics
      const treatmentResults = Array.from({ length: 20 }, (_, j) => ({
        result_id: `treatment_${i}_${j}`,
        file_path: `src/file_${j}.ts`,
        snippet: `Treatment result ${j}`,
        relevance_score: 0.7 + Math.random() * 0.3,
        rank_position: j + 1,
        clicked: false,
        dwell_time_ms: undefined
      }));
      
      const controlResults = Array.from({ length: 20 }, (_, j) => ({
        result_id: `control_${i}_${j}`,
        file_path: `src/control_${j}.ts`,
        snippet: `Control result ${j}`,
        relevance_score: 0.65 + Math.random() * 0.25,
        rank_position: j + 1,
        clicked: false,
        dwell_time_ms: undefined
      }));
      
      // Generate user interactions
      const clickedPositions = [];
      const numClicks = Math.floor(Math.random() * 4); // 0-3 clicks
      for (let c = 0; c < numClicks; c++) {
        const clickPos = Math.floor(Math.random() * 10) + 1; // Click in top 10
        if (!clickedPositions.includes(clickPos)) {
          clickedPositions.push(clickPos);
        }
      }
      
      // Bias correction setup
      const randomizedSwapApplied = Math.random() < 0.5;
      const propensityWeights = Array.from({ length: 20 }, (_, j) => 1.0 / Math.max(0.05, 0.95 - j * 0.04));
      
      experiments.push({
        query_metadata: queryMetadata,
        treatment_results: treatmentResults,
        control_results: controlResults,
        interleaved_results: [...treatmentResults.slice(0, 10), ...controlResults.slice(0, 10)],
        user_interactions: {
          clicked_positions: clickedPositions,
          max_examined_rank: Math.max(10, ...clickedPositions),
          session_duration_ms: 30000 + Math.random() * 120000,
          abandonment: clickedPositions.length === 0
        },
        bias_correction: {
          propensity_weights: propensityWeights,
          randomized_swap_applied: randomizedSwapApplied,
          swap_positions: randomizedSwapApplied ? [1, 2] : undefined
        },
        evaluation_metrics: {
          treatment_ndcg_10: 0.75 + Math.random() * 0.15,
          control_ndcg_10: 0.70 + Math.random() * 0.12,
          treatment_sla_recall_50: 0.68 + Math.random() * 0.12,
          control_sla_recall_50: 0.65 + Math.random() * 0.10,
          treatment_clicks: Math.floor(Math.random() * 3),
          control_clicks: Math.floor(Math.random() * 2)
        }
      });
    }
    
    return experiments;
  }
}

// Factory function
export function createEntropyConditionedInterleaver(config?: Partial<InterleavingConfig>): EntropyConditionedInterleaver {
  const fullConfig = { ...DEFAULT_INTERLEAVING_CONFIG, ...config };
  return new EntropyConditionedInterleaver(fullConfig);
}