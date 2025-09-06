/**
 * Learning-to-Stop Optimization System
 * 
 * Implements machine learning-based early termination for WAND/BMW scanning
 * and dynamic ANN efSearch adjustment to maximize ΔnDCG per additional millisecond.
 * 
 * Key Features:
 * - Lightweight ML model trained on live features
 * - Features: impact_prefix_gain, remaining_budget_ms, topic_entropy, pos_in_cands, λ_ann(ms/ΔRecall)
 * - Objective: maximize ΔnDCG per additional millisecond
 * - Never-stop floor when positives_in_candidates < m
 * - Performance gate: p95 -0.8 to -1.5ms with SLA-Recall@50≥0 and upshift ∈[3%,7%]
 */

import type { SearchContext } from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';

// Configuration constants per TODO.md requirements
const LAMBDA_UTILITY_THRESHOLD = 0.1; // Marginal utility threshold
const NEVER_STOP_FLOOR = 5; // m = minimum positives before stopping allowed
const P95_IMPROVEMENT_TARGET_MIN = 0.8; // -0.8ms minimum improvement
const P95_IMPROVEMENT_TARGET_MAX = 1.5; // -1.5ms maximum improvement
const UPSHIFT_TARGET_MIN = 0.03; // 3% minimum upshift
const UPSHIFT_TARGET_MAX = 0.07; // 7% maximum upshift
const BASE_EF_SEARCH = 64; // Default efSearch value
const FEATURE_UPDATE_INTERVAL_MS = 1000; // Update live features every second

export interface StoppingFeatures {
  impact_prefix_gain: number; // Estimated gain from processing next block
  remaining_budget_ms: number; // Time remaining in query budget
  topic_entropy: number; // Entropy of current result set topics
  pos_in_cands: number; // Position in candidate list
  lambda_ann_ms_per_recall: number; // ANN λ(ms/ΔRecall) ratio
  positives_in_candidates: number; // Count of positive results so far
  query_complexity: number; // Estimated query complexity
  current_ndcg: number; // Current nDCG score
}

export interface StoppingPrediction {
  should_stop: boolean;
  confidence: number;
  estimated_gain: number;
  recommended_ef: number;
}

export interface ScannerState {
  blocks_processed: number;
  candidates_found: number;
  time_spent_ms: number;
  last_gain: number;
  marginal_utility: number;
}

export interface ANNState {
  current_ef: number;
  recall_achieved: number;
  time_spent_ms: number;
  risk_level: number;
}

export class LearningToStopSystem {
  private modelWeights = {
    // Lightweight linear model weights (would be trained from data)
    impact_prefix_gain: 0.4,
    remaining_budget_ms: 0.2,
    topic_entropy: -0.15,
    pos_in_cands: -0.1,
    lambda_ann: 0.25,
    positives_count: 0.3,
    query_complexity: -0.05,
    bias: 0.1,
  };
  
  private performanceMetrics = {
    stopping_decisions: [] as { decision: boolean; actual_gain: number; predicted_gain: number }[],
    scanner_improvements: [] as number[],
    ann_improvements: [] as number[],
    sla_violations: 0,
    recall_preservation: [] as number[],
  };
  
  private liveFeatureCache = new Map<string, StoppingFeatures>();
  private lastFeatureUpdate = 0;
  
  /**
   * Initialize the learning-to-stop system
   */
  async initialize(): Promise<void> {
    const span = LensTracer.createChildSpan('learning_stop_init');
    
    try {
      console.log('🛑 Initializing Learning-to-Stop system...');
      
      // Initialize model weights (in production, load from trained model)
      this.loadModelWeights();
      
      // Initialize performance tracking
      this.performanceMetrics = {
        stopping_decisions: [],
        scanner_improvements: [],
        ann_improvements: [],
        sla_violations: 0,
        recall_preservation: [],
      };
      
      span.setAttributes({ success: true });
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw error;
    } finally {
      span.end();
    }
  }
  
  /**
   * Make stopping decision for WAND/BMW scanner
   */
  shouldStopScanning(
    scannerState: ScannerState,
    ctx: SearchContext,
    queryStartTime: number
  ): StoppingPrediction {
    const span = LensTracer.createChildSpan('should_stop_scanning', {
      blocks_processed: scannerState.blocks_processed,
      candidates_found: scannerState.candidates_found,
      time_spent: scannerState.time_spent_ms
    });
    
    try {
      // Extract features for stopping decision
      const features = this.extractStoppingFeatures(scannerState, ctx, queryStartTime);
      
      // Apply never-stop floor constraint
      if (features.positives_in_candidates < NEVER_STOP_FLOOR) {
        span.setAttributes({
          decision: 'continue',
          reason: 'never_stop_floor',
          positives_count: features.positives_in_candidates
        });
        
        return {
          should_stop: false,
          confidence: 1.0,
          estimated_gain: 0.5, // High estimated gain to continue
          recommended_ef: BASE_EF_SEARCH,
        };
      }
      
      // Compute marginal utility using lightweight model
      const estimatedGain = this.predictMarginalGain(features);
      
      // Apply stopping threshold: stop when marginal utility < λ
      const shouldStop = estimatedGain < LAMBDA_UTILITY_THRESHOLD;
      
      // Record decision for model training/evaluation
      this.performanceMetrics.stopping_decisions.push({
        decision: shouldStop,
        actual_gain: 0, // Would be filled in after observing actual results
        predicted_gain: estimatedGain,
      });
      
      span.setAttributes({
        decision: shouldStop ? 'stop' : 'continue',
        estimated_gain: estimatedGain,
        confidence: Math.abs(estimatedGain - LAMBDA_UTILITY_THRESHOLD),
        features: JSON.stringify(features)
      });
      
      return {
        should_stop: shouldStop,
        confidence: Math.abs(estimatedGain - LAMBDA_UTILITY_THRESHOLD),
        estimated_gain: estimatedGain,
        recommended_ef: BASE_EF_SEARCH, // Not used for scanning
      };
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      
      // Default to continue on error
      return {
        should_stop: false,
        confidence: 0.0,
        estimated_gain: 0.5,
        recommended_ef: BASE_EF_SEARCH,
      };
      
    } finally {
      span.end();
    }
  }
  
  /**
   * Dynamically adjust ANN efSearch parameter
   */
  optimizeANNSearch(
    annState: ANNState,
    ctx: SearchContext,
    queryStartTime: number
  ): number {
    const span = LensTracer.createChildSpan('optimize_ann_search', {
      current_ef: annState.current_ef,
      recall_achieved: annState.recall_achieved,
      time_spent: annState.time_spent_ms
    });
    
    try {
      // Extract features for ANN optimization
      const features = this.extractStoppingFeatures(
        {
          blocks_processed: 0,
          candidates_found: Math.round(annState.recall_achieved * 100),
          time_spent_ms: annState.time_spent_ms,
          last_gain: 0,
          marginal_utility: 0,
        },
        ctx,
        queryStartTime
      );
      
      // Calculate risk and headroom
      const remainingTime = Math.max(0, features.remaining_budget_ms);
      const riskLevel = annState.risk_level;
      
      // Compute efSearch adjustment: ef = base_ef + g(risk, headroom_ms)
      const adjustmentFunction = this.computeEfAdjustment(riskLevel, remainingTime);
      
      let recommendedEf = BASE_EF_SEARCH + adjustmentFunction;
      
      // Apply constraint: if g < 0, keep base_ef
      if (adjustmentFunction < 0) {
        recommendedEf = BASE_EF_SEARCH;
      }
      
      // Clamp to reasonable bounds
      recommendedEf = Math.max(16, Math.min(512, recommendedEf));
      
      // Record performance impact
      const improvement = (BASE_EF_SEARCH - recommendedEf) / BASE_EF_SEARCH;
      this.performanceMetrics.ann_improvements.push(improvement);
      
      span.setAttributes({
        base_ef: BASE_EF_SEARCH,
        recommended_ef: recommendedEf,
        adjustment: adjustmentFunction,
        risk_level: riskLevel,
        remaining_time_ms: remainingTime
      });
      
      return recommendedEf;
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      
      // Return base ef on error
      return BASE_EF_SEARCH;
      
    } finally {
      span.end();
    }
  }
  
  /**
   * Extract stopping features from current search state
   */
  private extractStoppingFeatures(
    scannerState: ScannerState,
    ctx: SearchContext,
    queryStartTime: number
  ): StoppingFeatures {
    const now = Date.now();
    const elapsedTime = now - queryStartTime;
    const queryBudget = 20; // 20ms overall p95 target from config
    
    // Update live features cache if needed
    if (now - this.lastFeatureUpdate > FEATURE_UPDATE_INTERVAL_MS) {
      this.updateLiveFeatures();
    }
    
    return {
      impact_prefix_gain: this.estimateImpactPrefixGain(scannerState),
      remaining_budget_ms: Math.max(0, queryBudget - elapsedTime),
      topic_entropy: this.computeTopicEntropy(ctx.query),
      pos_in_cands: scannerState.candidates_found,
      lambda_ann_ms_per_recall: this.computeLambdaANN(elapsedTime, scannerState.candidates_found),
      positives_in_candidates: Math.max(0, scannerState.candidates_found),
      query_complexity: this.estimateQueryComplexity(ctx.query),
      current_ndcg: this.estimateCurrentNDCG(scannerState),
    };
  }
  
  /**
   * Predict marginal gain using lightweight linear model
   */
  private predictMarginalGain(features: StoppingFeatures): number {
    // Linear model: y = w1*x1 + w2*x2 + ... + bias
    let prediction = this.modelWeights.bias;
    prediction += this.modelWeights.impact_prefix_gain * features.impact_prefix_gain;
    prediction += this.modelWeights.remaining_budget_ms * (features.remaining_budget_ms / 20); // Normalize to [0,1]
    prediction += this.modelWeights.topic_entropy * features.topic_entropy;
    prediction += this.modelWeights.pos_in_cands * (features.pos_in_cands / 100); // Normalize
    prediction += this.modelWeights.lambda_ann * features.lambda_ann_ms_per_recall;
    prediction += this.modelWeights.positives_count * (features.positives_in_candidates / 50); // Normalize
    prediction += this.modelWeights.query_complexity * features.query_complexity;
    
    // Apply sigmoid to get probability-like output
    return 1 / (1 + Math.exp(-prediction));
  }
  
  /**
   * Estimate impact of processing next prefix block
   */
  private estimateImpactPrefixGain(scannerState: ScannerState): number {
    if (scannerState.blocks_processed === 0) {
      return 0.8; // High gain expected for first block
    }
    
    // Diminishing returns model: gain = initial_gain * exp(-decay * blocks)
    const decayRate = 0.3;
    const initialGain = 0.8;
    
    return initialGain * Math.exp(-decayRate * scannerState.blocks_processed);
  }
  
  /**
   * Compute topic entropy of current results
   */
  private computeTopicEntropy(query: string): number {
    // Simplified entropy calculation based on query characteristics
    const tokens = query.toLowerCase().split(/\s+/);
    const uniqueTokens = new Set(tokens);
    
    if (uniqueTokens.size <= 1) {
      return 0; // Low entropy for single-term queries
    }
    
    // Shannon entropy approximation
    const tokenCounts = new Map<string, number>();
    for (const token of tokens) {
      tokenCounts.set(token, (tokenCounts.get(token) || 0) + 1);
    }
    
    let entropy = 0;
    for (const count of tokenCounts.values()) {
      const probability = count / tokens.length;
      entropy -= probability * Math.log2(probability);
    }
    
    // Normalize to [0, 1]
    return Math.min(1.0, entropy / Math.log2(tokens.length));
  }
  
  /**
   * Compute λ_ann(ms/ΔRecall) ratio
   */
  private computeLambdaANN(elapsedTime: number, candidatesFound: number): number {
    if (candidatesFound === 0) {
      return Infinity; // High cost, no recall
    }
    
    // Estimate recall as candidates found / expected total
    const estimatedRecall = Math.min(1.0, candidatesFound / 100);
    const deltaRecall = Math.max(0.01, estimatedRecall); // Avoid division by zero
    
    return elapsedTime / deltaRecall;
  }
  
  /**
   * Estimate query complexity based on query string
   */
  private estimateQueryComplexity(query: string): number {
    // Heuristic complexity based on query features
    let complexity = 0.5; // Base complexity
    
    const tokens = query.split(/\s+/);
    
    // More tokens = higher complexity
    complexity += Math.min(0.3, tokens.length * 0.05);
    
    // Special characters increase complexity
    if (/[.*+?^${}()|[\]\\]/.test(query)) {
      complexity += 0.2;
    }
    
    // Very short queries are simpler
    if (query.length < 3) {
      complexity -= 0.2;
    }
    
    // Very long queries are more complex
    if (query.length > 50) {
      complexity += 0.2;
    }
    
    return Math.max(0, Math.min(1, complexity));
  }
  
  /**
   * Estimate current nDCG score
   */
  private estimateCurrentNDCG(scannerState: ScannerState): number {
    // Simplified nDCG estimation based on candidates found
    if (scannerState.candidates_found === 0) {
      return 0;
    }
    
    // Assume diminishing returns: first candidates are more relevant
    let dcg = 0;
    for (let i = 0; i < Math.min(scannerState.candidates_found, 10); i++) {
      const relevance = Math.max(0, 1 - (i * 0.1)); // Decreasing relevance
      dcg += relevance / Math.log2(i + 2);
    }
    
    // Normalize against ideal DCG for k=10
    let idcg = 0;
    for (let i = 0; i < 10; i++) {
      idcg += 1 / Math.log2(i + 2);
    }
    
    return dcg / idcg;
  }
  
  /**
   * Compute efSearch adjustment function g(risk, headroom_ms)
   */
  private computeEfAdjustment(riskLevel: number, headroomMs: number): number {
    // Risk-based adjustment: higher risk = lower ef to save time
    const riskPenalty = -riskLevel * 32;
    
    // Headroom bonus: more time available = higher ef allowed
    const headroomBonus = Math.min(32, headroomMs * 2);
    
    return riskPenalty + headroomBonus;
  }
  
  /**
   * Update live feature cache
   */
  private updateLiveFeatures(): void {
    this.lastFeatureUpdate = Date.now();
    
    // In production, this would update global feature cache from
    // recent queries, system metrics, etc.
    // For now, just record the update
  }
  
  /**
   * Load model weights (placeholder for production model loading)
   */
  private loadModelWeights(): void {
    // In production, load weights from trained model file
    // For now, use hand-tuned heuristic weights
    console.log('📊 Loading Learning-to-Stop model weights...');
  }
  
  /**
   * Update model with feedback (online learning)
   */
  updateModelWithFeedback(
    features: StoppingFeatures,
    actualGain: number,
    decision: boolean
  ): void {
    const span = LensTracer.createChildSpan('update_model_feedback', {
      actual_gain: actualGain,
      decision
    });
    
    try {
      // Simple online learning: adjust weights based on prediction error
      const predictedGain = this.predictMarginalGain(features);
      const error = actualGain - predictedGain;
      const learningRate = 0.01; // Small learning rate for stability
      
      // Update weights proportional to features and error
      this.modelWeights.impact_prefix_gain += learningRate * error * features.impact_prefix_gain;
      this.modelWeights.remaining_budget_ms += learningRate * error * (features.remaining_budget_ms / 20);
      this.modelWeights.topic_entropy += learningRate * error * features.topic_entropy;
      this.modelWeights.pos_in_cands += learningRate * error * (features.pos_in_cands / 100);
      
      // Update performance metrics
      const decisionIndex = this.performanceMetrics.stopping_decisions.findIndex(
        d => Math.abs(d.predicted_gain - predictedGain) < 0.001
      );
      
      if (decisionIndex >= 0) {
        this.performanceMetrics.stopping_decisions[decisionIndex].actual_gain = actualGain;
      }
      
      span.setAttributes({
        success: true,
        prediction_error: error,
        learning_rate: learningRate
      });
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
    } finally {
      span.end();
    }
  }
  
  /**
   * Get performance metrics and check SLA compliance
   */
  getPerformanceMetrics() {
    const scannerImprovements = this.performanceMetrics.scanner_improvements;
    const annImprovements = this.performanceMetrics.ann_improvements;
    
    // Calculate p95 improvement
    const allImprovements = [...scannerImprovements, ...annImprovements];
    const p95Improvement = allImprovements.length > 0
      ? allImprovements.sort((a, b) => a - b)[Math.floor(allImprovements.length * 0.95)]
      : 0;
    
    // Check SLA compliance: p95 -0.8 to -1.5ms
    const slaCompliant = p95Improvement >= P95_IMPROVEMENT_TARGET_MIN && 
                        p95Improvement <= P95_IMPROVEMENT_TARGET_MAX;
    
    // Calculate recall preservation
    const avgRecallPreservation = this.performanceMetrics.recall_preservation.length > 0
      ? this.performanceMetrics.recall_preservation.reduce((a, b) => a + b) / this.performanceMetrics.recall_preservation.length
      : 0;
    
    return {
      stopping_decisions_count: this.performanceMetrics.stopping_decisions.length,
      p95_improvement_ms: p95Improvement,
      sla_compliant: slaCompliant,
      sla_violations: this.performanceMetrics.sla_violations,
      average_recall_preservation: avgRecallPreservation,
      model_accuracy: this.calculateModelAccuracy(),
    };
  }
  
  /**
   * Calculate model accuracy from recorded decisions
   */
  private calculateModelAccuracy(): number {
    const decisions = this.performanceMetrics.stopping_decisions;
    if (decisions.length === 0) {
      return 0;
    }
    
    let correctPredictions = 0;
    for (const decision of decisions) {
      // Consider prediction correct if predicted gain and actual gain
      // are on the same side of the threshold
      const predictedStop = decision.predicted_gain < LAMBDA_UTILITY_THRESHOLD;
      const actualStop = decision.actual_gain < LAMBDA_UTILITY_THRESHOLD;
      
      if (predictedStop === actualStop) {
        correctPredictions++;
      }
    }
    
    return correctPredictions / decisions.length;
  }
  
  /**
   * Cleanup and shutdown
   */
  async shutdown(): Promise<void> {
    const span = LensTracer.createChildSpan('learning_stop_shutdown');
    
    try {
      console.log('🛑 Shutting down Learning-to-Stop system...');
      
      // Save model state (in production)
      // Clear caches
      this.liveFeatureCache.clear();
      
      span.setAttributes({ success: true });
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw error;
    } finally {
      span.end();
    }
  }
}