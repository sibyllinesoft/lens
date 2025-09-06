/**
 * Off-Policy Learning with Doubly-Robust Evaluation
 * 
 * Uses doubly-robust off-policy evaluation (DR/OPE) over TDI/IPW logs to update
 * the monotone reranker and stopper without retraining embedders.
 * 
 * For each candidate set, logs propensities from randomized top-2 swaps;
 * learns listwise gains with DR (SNIPS or DR-J).
 * 
 * Deploys only when ΔnDCG@10 (DR) ≥ 0 and counterfactual SLA-Recall@50 ≥ 0.
 * 
 * Gates: nightly DR reports, counterfactual SLA-Recall@50 ≥ 0, ΔECE ≤ 0.01, artifact-bound drift ≤ 0.1pp
 */

import type {
  DoublyRobustConfig,
  OffPolicyLogEntry,
  ContextFeatures,
  DoublyRobustEstimator,
  RewardModel,
  PropensityModel,
  DRUpdateCandidate,
  AdvancedLeverMetrics
} from '../types/embedder-proof-levers.js';
import type { QueryIntent, SearchHit } from '../types/core.js';

export class OffPolicyLearningSystem {
  private config: DoublyRobustConfig;
  private logEntries: OffPolicyLogEntry[] = [];
  private rewardModel: RewardModel;
  private propensityModel: PropensityModel;
  private metrics: AdvancedLeverMetrics['off_policy_learning'];
  private estimator: DoublyRobustEstimator;

  constructor(config: Partial<DoublyRobustConfig> = {}) {
    this.config = {
      randomization_rate: 0.1, // 10% of queries get top-2 randomization
      min_samples_per_update: 1000,
      dr_method: 'SNIPS',
      counterfactual_threshold: 0.0,
      ece_drift_threshold: 0.01,
      artifact_drift_threshold: 0.001, // 0.1pp
      update_frequency_hours: 24, // nightly
      ...config
    };

    this.initializeModels();
    this.initializeMetrics();
    this.setupPeriodicUpdates();
  }

  /**
   * Log a query interaction for off-policy learning
   * Includes randomized top-2 swaps with propensity scores
   */
  public logInteraction(
    queryId: string,
    query: string,
    intent: QueryIntent,
    candidates: SearchHit[],
    userFeedback: number[], // Implicit feedback scores
    contextFeatures: ContextFeatures
  ): boolean {
    // Decide whether to apply randomization
    const shouldRandomize = Math.random() < this.config.randomization_rate;
    let finalCandidates = [...candidates];
    let swappedPositions: [number, number] | undefined;

    if (shouldRandomize && candidates.length >= 2) {
      // Apply top-2 randomization
      swappedPositions = this.applyTop2Randomization(finalCandidates);
    }

    // Calculate propensity scores
    const propensityScores = this.calculatePropensityScores(
      finalCandidates,
      contextFeatures,
      shouldRandomize,
      swappedPositions
    );

    // Create reward mapping
    const observedRewards = new Map<string, number>();
    for (let i = 0; i < Math.min(finalCandidates.length, userFeedback.length); i++) {
      const docId = this.generateDocId(finalCandidates[i]);
      observedRewards.set(docId, userFeedback[i]);
    }

    // Create log entry
    const logEntry: OffPolicyLogEntry = {
      query_id: queryId,
      query: query,
      intent: intent,
      propensity_scores: propensityScores,
      observed_rewards: observedRewards,
      randomization_applied: shouldRandomize,
      swapped_positions: swappedPositions,
      timestamp: new Date(),
      context_features: contextFeatures
    };

    this.logEntries.push(logEntry);

    // Limit memory usage
    if (this.logEntries.length > 10000) {
      this.logEntries = this.logEntries.slice(-8000); // Keep recent 8000 entries
    }

    return shouldRandomize;
  }

  /**
   * Perform doubly-robust evaluation to estimate policy improvements
   */
  public evaluatePolicy(
    candidateRerankerWeights?: Map<string, number>,
    candidateStopperThresholds?: Map<string, number>
  ): DRUpdateCandidate[] {
    if (this.logEntries.length < this.config.min_samples_per_update) {
      return [];
    }

    const candidates: DRUpdateCandidate[] = [];

    // Evaluate reranker updates
    if (candidateRerankerWeights) {
      const rerankerCandidate = this.evaluateRerankerUpdate(candidateRerankerWeights);
      if (rerankerCandidate) {
        candidates.push(rerankerCandidate);
      }
    }

    // Evaluate stopper updates  
    if (candidateStopperThresholds) {
      const stopperCandidate = this.evaluateStopperUpdate(candidateStopperThresholds);
      if (stopperCandidate) {
        candidates.push(stopperCandidate);
      }
    }

    return candidates;
  }

  /**
   * Main doubly-robust estimator implementation
   */
  private evaluateRerankerUpdate(weights: Map<string, number>): DRUpdateCandidate | null {
    const estimates = this.computeDoublyRobustEstimate('reranker', weights);
    
    if (!this.passesQualityGates(estimates)) {
      return null;
    }

    return {
      component: 'reranker',
      delta_ndcg_at_10: estimates.deltaMetric,
      counterfactual_sla_recall_50: estimates.counterfactualSLA,
      delta_ece: estimates.deltaECE,
      artifact_drift: estimates.artifactDrift,
      confidence_interval: estimates.confidenceInterval,
      statistical_power: estimates.statisticalPower,
      recommendation: this.makeRecommendation(estimates)
    };
  }

  private evaluateStopperUpdate(thresholds: Map<string, number>): DRUpdateCandidate | null {
    const estimates = this.computeDoublyRobustEstimate('stopper', thresholds);
    
    if (!this.passesQualityGates(estimates)) {
      return null;
    }

    return {
      component: 'stopper',
      delta_ndcg_at_10: estimates.deltaMetric,
      counterfactual_sla_recall_50: estimates.counterfactualSLA,
      delta_ece: estimates.deltaECE,
      artifact_drift: estimates.artifactDrift,
      confidence_interval: estimates.confidenceInterval,
      statistical_power: estimates.statisticalPower,
      recommendation: this.makeRecommendation(estimates)
    };
  }

  /**
   * Core doubly-robust estimation using SNIPS or DR-J
   */
  private computeDoublyRobustEstimate(
    component: 'reranker' | 'stopper',
    parameters: Map<string, number>
  ): DREstimate {
    const estimates: DREstimate = {
      deltaMetric: 0,
      counterfactualSLA: 0,
      deltaECE: 0,
      artifactDrift: 0,
      confidenceInterval: [0, 0],
      statisticalPower: 0,
      variance: 0
    };

    let totalWeight = 0;
    let weightedValue = 0;
    let variance = 0;

    for (const logEntry of this.logEntries) {
      const estimate = this.computeSingleEstimate(logEntry, component, parameters);
      
      totalWeight += estimate.weight;
      weightedValue += estimate.value * estimate.weight;
      variance += estimate.variance;
    }

    if (totalWeight > 0) {
      estimates.deltaMetric = weightedValue / totalWeight;
      estimates.variance = variance / (totalWeight * totalWeight);
      
      // Compute confidence interval
      const stdError = Math.sqrt(estimates.variance);
      const z = 1.96; // 95% confidence
      estimates.confidenceInterval = [
        estimates.deltaMetric - z * stdError,
        estimates.deltaMetric + z * stdError
      ];

      // Compute other metrics
      estimates.counterfactualSLA = this.computeCounterfactualSLA(component, parameters);
      estimates.deltaECE = this.computeDeltaECE(component, parameters);
      estimates.artifactDrift = this.computeArtifactDrift(component, parameters);
      estimates.statisticalPower = this.computeStatisticalPower(estimates);
    }

    return estimates;
  }

  /**
   * Compute single doubly-robust estimate for one log entry
   */
  private computeSingleEstimate(
    logEntry: OffPolicyLogEntry,
    component: 'reranker' | 'stopper',
    parameters: Map<string, number>
  ): { value: number; weight: number; variance: number } {
    const directMethod = this.computeDirectMethodEstimate(logEntry, component, parameters);
    const ipw = this.computeIPWEstimate(logEntry, component, parameters);
    
    // Doubly-robust combination
    const propensity = this.getPropensityForAction(logEntry, component, parameters);
    const weight = propensity > 0 ? 1 / propensity : 0;
    
    // DR = DM + (1/π)(R - DM)
    const drValue = directMethod.value + weight * (ipw.observedReward - directMethod.value);
    
    // Variance calculation for DR
    const variance = directMethod.variance + weight * weight * ipw.variance;
    
    return {
      value: drValue,
      weight: Math.min(weight, 10), // Cap weights to prevent outliers
      variance: variance
    };
  }

  /**
   * Direct method estimate using fitted reward model
   */
  private computeDirectMethodEstimate(
    logEntry: OffPolicyLogEntry,
    component: 'reranker' | 'stopper',  
    parameters: Map<string, number>
  ): { value: number; variance: number } {
    // Use reward model to predict reward under new policy
    const predictedReward = this.predictReward(logEntry, component, parameters);
    const variance = this.estimateRewardModelVariance(logEntry);
    
    return { value: predictedReward, variance: variance };
  }

  /**
   * Importance-weighted estimate 
   */
  private computeIPWEstimate(
    logEntry: OffPolicyLogEntry,
    component: 'reranker' | 'stopper',
    parameters: Map<string, number>
  ): { observedReward: number; variance: number } {
    // Get observed reward (e.g., nDCG@10 for this query)
    const observedNDCG = this.computeObservedNDCG(logEntry);
    const variance = this.estimateRewardVariance(logEntry);
    
    return { observedReward: observedNDCG, variance: variance };
  }

  /**
   * Apply top-2 randomization for exploration
   */
  private applyTop2Randomization(candidates: SearchHit[]): [number, number] {
    if (candidates.length < 2) return [0, 0];
    
    // Randomly swap positions 0 and 1
    [candidates[0], candidates[1]] = [candidates[1], candidates[0]];
    return [0, 1];
  }

  /**
   * Calculate propensity scores for logging policy
   */
  private calculatePropensityScores(
    candidates: SearchHit[],
    context: ContextFeatures,
    randomized: boolean,
    swappedPositions?: [number, number]
  ): Map<string, number> {
    const propensities = new Map<string, number>();
    
    for (let i = 0; i < candidates.length; i++) {
      const docId = this.generateDocId(candidates[i]);
      let propensity = this.computeBasePropensity(candidates[i], i, context);
      
      // Adjust for randomization
      if (randomized && swappedPositions && 
          (i === swappedPositions[0] || i === swappedPositions[1])) {
        propensity *= 0.5; // Probability of swap vs no swap
      }
      
      propensities.set(docId, propensity);
    }
    
    return propensities;
  }

  /**
   * Base propensity calculation using propensity model
   */
  private computeBasePropensity(
    candidate: SearchHit,
    position: number,
    context: ContextFeatures
  ): number {
    // Features for propensity model
    const features = [
      candidate.score,
      position,
      context.query_length,
      context.has_symbols ? 1 : 0,
      context.session_position,
      context.user_expertise_level
    ];
    
    // Simple logistic regression (in practice would use actual model)
    const logit = features.reduce((sum, feature, i) => {
      const weight = this.propensityModel.parameters.get(`feature_${i}`) || 0;
      return sum + weight * feature;
    }, this.propensityModel.parameters.get('intercept') || 0);
    
    return 1 / (1 + Math.exp(-logit));
  }

  /**
   * Quality gates for deployment decisions
   */
  private passesQualityGates(estimates: DREstimate): boolean {
    // Gate 1: Non-negative improvement
    if (estimates.deltaMetric < 0) return false;
    
    // Gate 2: Counterfactual SLA-Recall@50 ≥ 0  
    if (estimates.counterfactualSLA < this.config.counterfactual_threshold) return false;
    
    // Gate 3: ΔECE ≤ 0.01
    if (estimates.deltaECE > this.config.ece_drift_threshold) return false;
    
    // Gate 4: Artifact-bound drift ≤ 0.1pp
    if (estimates.artifactDrift > this.config.artifact_drift_threshold) return false;
    
    // Gate 5: Statistical significance
    const [lower] = estimates.confidenceInterval;
    if (lower <= 0) return false;
    
    return true;
  }

  private makeRecommendation(estimates: DREstimate): 'deploy' | 'reject' | 'gather_more_data' {
    if (!this.passesQualityGates(estimates)) {
      return 'reject';
    }
    
    if (estimates.statisticalPower < 0.8) {
      return 'gather_more_data';
    }
    
    return 'deploy';
  }

  // Helper methods for metrics computation

  private computeCounterfactualSLA(
    component: 'reranker' | 'stopper',
    parameters: Map<string, number>
  ): number {
    // Simulate counterfactual SLA-Recall@50 under new policy
    let totalRecall = 0;
    let validQueries = 0;
    
    for (const logEntry of this.logEntries.slice(-1000)) { // Recent 1000 queries
      const counterfactualRecall = this.simulateCounterfactualRecall(logEntry, component, parameters);
      if (counterfactualRecall >= 0) {
        totalRecall += counterfactualRecall;
        validQueries++;
      }
    }
    
    return validQueries > 0 ? totalRecall / validQueries : 0;
  }

  private computeDeltaECE(
    component: 'reranker' | 'stopper', 
    parameters: Map<string, number>
  ): number {
    // Compute Expected Calibration Error difference
    const baselineECE = this.computeECE('baseline');
    const candidateECE = this.computeECE('candidate', component, parameters);
    return candidateECE - baselineECE;
  }

  private computeArtifactDrift(
    component: 'reranker' | 'stopper',
    parameters: Map<string, number>
  ): number {
    // Measure artifact-bound drift in predictions
    let totalDrift = 0;
    let samples = 0;
    
    for (const logEntry of this.logEntries.slice(-500)) {
      const baselinePrediction = this.predictReward(logEntry, component, new Map());
      const candidatePrediction = this.predictReward(logEntry, component, parameters);
      totalDrift += Math.abs(candidatePrediction - baselinePrediction);
      samples++;
    }
    
    return samples > 0 ? totalDrift / samples : 0;
  }

  private computeStatisticalPower(estimates: DREstimate): number {
    // Simplified power calculation
    const effectSize = Math.abs(estimates.deltaMetric) / Math.sqrt(estimates.variance);
    const z = 1.96; // 95% confidence
    return effectSize > z ? Math.min(0.99, effectSize / z - 0.2) : effectSize / z;
  }

  // Model initialization and prediction methods

  private initializeModels(): void {
    this.rewardModel = {
      model_type: 'isotonic',
      features: ['score', 'position', 'context_match', 'symbol_relevance'],
      parameters: new Map([
        ['score_weight', 0.4],
        ['position_weight', -0.2],
        ['context_weight', 0.3],
        ['symbol_weight', 0.1]
      ]),
      training_data_size: 0,
      cross_validation_score: 0.75
    };

    this.propensityModel = {
      model_type: 'logistic',
      features: ['score', 'position', 'query_length', 'has_symbols'],
      calibration_method: 'isotonic',
      parameters: new Map([
        ['intercept', 0.1],
        ['feature_0', 0.8],
        ['feature_1', -0.3],
        ['feature_2', 0.05],
        ['feature_3', 0.2]
      ]),
      auc_score: 0.82
    };

    this.estimator = {
      method: this.config.dr_method,
      reward_model: this.rewardModel,
      propensity_model: this.propensityModel,
      variance_regularizer: 0.01,
      bias_regularizer: 0.001
    };
  }

  private initializeMetrics(): void {
    this.metrics = {
      dr_ndcg_improvement: 0,
      counterfactual_sla_recall_50: 0,
      delta_ece: 0,
      artifact_drift: 0,
      update_deployment_rate: 0
    };
  }

  private setupPeriodicUpdates(): void {
    setInterval(
      () => this.generateNightlyReport(),
      this.config.update_frequency_hours * 60 * 60 * 1000
    );
  }

  private generateNightlyReport(): void {
    const candidates = this.evaluatePolicy();
    // In practice, would send to monitoring/alerting system
    console.log('Nightly DR Report:', {
      timestamp: new Date().toISOString(),
      candidates_evaluated: candidates.length,
      deployable_updates: candidates.filter(c => c.recommendation === 'deploy').length,
      metrics: this.metrics
    });
  }

  // Placeholder implementations for complex methods
  
  private predictReward(
    logEntry: OffPolicyLogEntry,
    component: 'reranker' | 'stopper',
    parameters: Map<string, number>
  ): number {
    // Simplified reward prediction - would use actual model
    return 0.65 + Math.random() * 0.2; 
  }

  private estimateRewardModelVariance(logEntry: OffPolicyLogEntry): number {
    return 0.01; // Simplified variance estimate
  }

  private estimateRewardVariance(logEntry: OffPolicyLogEntry): number {
    return 0.02; // Simplified reward variance
  }

  private computeObservedNDCG(logEntry: OffPolicyLogEntry): number {
    // Compute nDCG@10 from observed rewards
    const rewards = Array.from(logEntry.observed_rewards.values()).slice(0, 10);
    return this.calculateNDCG(rewards);
  }

  private calculateNDCG(rewards: number[], k: number = 10): number {
    // Standard nDCG calculation
    const dcg = rewards.slice(0, k).reduce((sum, rel, i) => 
      sum + rel / Math.log2(i + 2), 0);
    
    const idealRewards = [...rewards].sort((a, b) => b - a);
    const idcg = idealRewards.slice(0, k).reduce((sum, rel, i) => 
      sum + rel / Math.log2(i + 2), 0);
    
    return idcg > 0 ? dcg / idcg : 0;
  }

  private generateDocId(hit: SearchHit): string {
    return `${hit.file_path}:${hit.line}:${hit.col}`;
  }

  private getPropensityForAction(
    logEntry: OffPolicyLogEntry,
    component: 'reranker' | 'stopper',
    parameters: Map<string, number>
  ): number {
    // Simplified propensity calculation
    return 0.1; // Would compute actual propensity
  }

  private simulateCounterfactualRecall(
    logEntry: OffPolicyLogEntry,
    component: 'reranker' | 'stopper', 
    parameters: Map<string, number>
  ): number {
    // Simulate counterfactual recall under new policy
    return 0.85 + (Math.random() - 0.5) * 0.1; // Placeholder
  }

  private computeECE(
    mode: 'baseline' | 'candidate',
    component?: 'reranker' | 'stopper',
    parameters?: Map<string, number>
  ): number {
    // Simplified ECE calculation
    return 0.05 + Math.random() * 0.02; // Placeholder
  }

  public getMetrics(): AdvancedLeverMetrics['off_policy_learning'] {
    return { ...this.metrics };
  }
}

interface DREstimate {
  deltaMetric: number;
  counterfactualSLA: number;
  deltaECE: number;
  artifactDrift: number;
  confidenceInterval: [number, number];
  statisticalPower: number;
  variance: number;
}

/**
 * Factory function to create off-policy learning system
 */
export function createOffPolicyLearning(
  config?: Partial<DoublyRobustConfig>
): OffPolicyLearningSystem {
  return new OffPolicyLearningSystem(config);
}