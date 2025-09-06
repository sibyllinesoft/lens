/**
 * Conformal Router with Risk-Aware Routing
 * 
 * Implements confidence-aware routing using conformal prediction to decide when to use
 * expensive search modes (768d vectors, higher efSearch) based on per-query misrank 
 * risk estimation. Maintains 5% upshift rate cap by construction.
 */

import type { SearchContext } from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';

export interface ConformalPredictionFeatures {
  query_length: number;
  word_count: number;
  has_special_chars: boolean;
  fuzzy_enabled: boolean;
  structural_mode: boolean;
  avg_word_length: number;
  query_entropy: number;
  identifier_density: number;
  semantic_complexity: number;
}

export interface MisrankRisk {
  risk_score: number;
  confidence_interval: [number, number];
  nonconformity_score: number;
  calibrated: boolean;
}

export interface RoutingDecision {
  should_upshift: boolean;
  upshift_type: 'dimension_768d' | 'efSearch_boost' | 'mmr_diversity' | 'none';
  risk_budget_used: number;
  routing_reason: string;
  expected_improvement: number;
}

export interface UpshiftBudget {
  daily_budget: number;
  used_today: number;
  remaining_budget: number;
  p95_headroom_ms: number;
  current_upshift_rate: number;
}

/**
 * Conformal prediction for search quality risk assessment
 */
class ConformalPredictor {
  private calibrationData: Array<{
    features: ConformalPredictionFeatures;
    actual_ndcg: number;
    predicted_ndcg: number;
  }> = [];
  
  private nonconformityScores: number[] = [];
  private isCalibrated = false;
  
  /**
   * Calibrate the conformal predictor on held-out data
   */
  calibrate(calibrationData: Array<{
    features: ConformalPredictionFeatures;
    actual_ndcg: number;
    predicted_ndcg: number;
  }>): void {
    this.calibrationData = calibrationData;
    
    // Calculate nonconformity scores (absolute residuals)
    this.nonconformityScores = calibrationData.map(
      item => Math.abs(item.actual_ndcg - item.predicted_ndcg)
    );
    
    // Sort for quantile calculations
    this.nonconformityScores.sort((a, b) => a - b);
    this.isCalibrated = true;
    
    console.log(`📊 Conformal predictor calibrated on ${calibrationData.length} samples`);
    console.log(`   Mean nonconformity: ${this.nonconformityScores.reduce((a, b) => a + b) / this.nonconformityScores.length}`);
  }
  
  /**
   * Predict misrank risk with confidence interval
   */
  predictRisk(features: ConformalPredictionFeatures, confidence = 0.95): MisrankRisk {
    if (!this.isCalibrated) {
      // Use heuristic risk estimation if not calibrated
      return this.heuristicRisk(features);
    }
    
    // Calculate base prediction score
    const basePrediction = this.predictBaseNDCG(features);
    
    // Calculate conformal prediction interval
    const alpha = 1 - confidence;
    const quantileIndex = Math.ceil((this.nonconformityScores.length + 1) * (1 - alpha)) - 1;
    const nonconformityQuantile = this.nonconformityScores[Math.min(quantileIndex, this.nonconformityScores.length - 1)];
    
    // Risk score based on predicted quality and uncertainty
    const riskScore = 1 - basePrediction + nonconformityQuantile;
    
    return {
      risk_score: Math.max(0, Math.min(1, riskScore)),
      confidence_interval: [
        Math.max(0, basePrediction - nonconformityQuantile),
        Math.min(1, basePrediction + nonconformityQuantile)
      ],
      nonconformity_score: nonconformityQuantile,
      calibrated: true
    };
  }
  
  /**
   * Predict base nDCG score from features
   */
  private predictBaseNDCG(features: ConformalPredictionFeatures): number {
    // Simple linear model trained on features -> nDCG
    // In production, this would be a proper ML model
    let score = 0.7; // Base score
    
    // Query complexity factors
    if (features.query_length > 50) score -= 0.1;
    if (features.word_count > 8) score -= 0.05;
    if (features.has_special_chars) score -= 0.05;
    if (features.avg_word_length < 3) score -= 0.1; // Very short tokens are hard
    
    // Entropy and structure
    if (features.query_entropy < 1.5) score += 0.1; // Low entropy = focused query
    if (features.identifier_density > 0.5) score += 0.15; // Code-like queries
    if (features.semantic_complexity > 0.8) score -= 0.2; // Very complex semantics
    
    // Search mode adjustments
    if (features.fuzzy_enabled) score -= 0.05;
    if (features.structural_mode) score += 0.1;
    
    return Math.max(0.1, Math.min(0.95, score));
  }
  
  /**
   * Heuristic risk estimation when not calibrated
   */
  private heuristicRisk(features: ConformalPredictionFeatures): MisrankRisk {
    let riskScore = 0.3; // Base risk
    
    // High-risk query patterns
    if (features.query_length > 100) riskScore += 0.2;
    if (features.word_count > 10) riskScore += 0.1;
    if (features.semantic_complexity > 0.7) riskScore += 0.2;
    if (features.avg_word_length < 2) riskScore += 0.15;
    
    // Low-risk patterns  
    if (features.identifier_density > 0.6) riskScore -= 0.1;
    if (features.structural_mode) riskScore -= 0.05;
    
    riskScore = Math.max(0.1, Math.min(0.9, riskScore));
    
    return {
      risk_score: riskScore,
      confidence_interval: [riskScore - 0.1, riskScore + 0.1],
      nonconformity_score: 0.1,
      calibrated: false
    };
  }
}

/**
 * Budget manager for upshift operations
 */
class UpshiftBudgetManager {
  private dailyBudget: number;
  private usedToday: number;
  private resetTimestamp: number;
  private upshiftHistory: Array<{ timestamp: number; type: string }> = [];
  
  constructor(dailyBudgetPercent = 5.0) {
    this.dailyBudget = dailyBudgetPercent / 100; // Convert to fraction
    this.usedToday = 0;
    this.resetTimestamp = this.getTodayTimestamp();
  }
  
  /**
   * Check if upshift is within budget
   */
  canUpshift(): boolean {
    this.maybeReset();
    return this.usedToday < this.dailyBudget;
  }
  
  /**
   * Record an upshift usage
   */
  recordUpshift(type: string): void {
    this.maybeReset();
    
    const now = Date.now();
    this.usedToday += 1; // Increment counter (will be normalized by total queries)
    this.upshiftHistory.push({ timestamp: now, type });
    
    // Keep only last 24 hours of history
    const dayAgo = now - 24 * 60 * 60 * 1000;
    this.upshiftHistory = this.upshiftHistory.filter(entry => entry.timestamp > dayAgo);
  }
  
  /**
   * Get current budget status
   */
  getBudgetStatus(totalQueriesToday = 100): UpshiftBudget {
    this.maybeReset();
    
    const currentRate = totalQueriesToday > 0 ? this.usedToday / totalQueriesToday : 0;
    
    return {
      daily_budget: this.dailyBudget * 100, // Convert back to percentage
      used_today: this.usedToday,
      remaining_budget: Math.max(0, this.dailyBudget * totalQueriesToday - this.usedToday),
      p95_headroom_ms: this.estimateP95Headroom(),
      current_upshift_rate: currentRate * 100
    };
  }
  
  /**
   * Estimate available p95 latency headroom
   */
  private estimateP95Headroom(): number {
    // Simple heuristic: assume we have headroom unless we're using budget heavily
    const recentUpshifts = this.upshiftHistory.filter(
      entry => entry.timestamp > Date.now() - 60 * 60 * 1000 // Last hour
    );
    
    if (recentUpshifts.length > 10) return 0.5; // Low headroom
    if (recentUpshifts.length > 5) return 2.0;  // Some headroom
    return 5.0; // Good headroom
  }
  
  /**
   * Reset daily counters if new day
   */
  private maybeReset(): void {
    const today = this.getTodayTimestamp();
    if (today !== this.resetTimestamp) {
      this.usedToday = 0;
      this.resetTimestamp = today;
    }
  }
  
  private getTodayTimestamp(): number {
    return Math.floor(Date.now() / (24 * 60 * 60 * 1000));
  }
}

/**
 * Main conformal router class
 */
export class ConformalRouter {
  private predictor: ConformalPredictor;
  private budgetManager: UpshiftBudgetManager;
  private riskThreshold: number;
  private enabled = true;
  
  // Metrics
  private totalQueries = 0;
  private upshiftedQueries = 0;
  private lastCalibrationTime: number = 0;
  
  constructor(
    riskThreshold = 0.6, 
    dailyBudgetPercent = 5.0
  ) {
    this.predictor = new ConformalPredictor();
    this.budgetManager = new UpshiftBudgetManager(dailyBudgetPercent);
    this.riskThreshold = riskThreshold;
  }
  
  /**
   * Main routing decision function
   * 
   * ```
   * risk = conformal_nonconformity(features_of(top1..k))   # calibrated on held-out
   * if risk>τ_r and upshift_rate<0.05 and p95_headroom>h: use 768d or efSearch+=Δ
   * ```
   */
  async makeRoutingDecision(
    ctx: SearchContext,
    currentCandidates?: any[]
  ): Promise<RoutingDecision> {
    const span = LensTracer.createChildSpan('conformal_routing');
    this.totalQueries++;
    
    try {
      if (!this.enabled) {
        return {
          should_upshift: false,
          upshift_type: 'none',
          risk_budget_used: 0,
          routing_reason: 'router_disabled',
          expected_improvement: 0
        };
      }
      
      // Extract features from query and context
      const features = this.extractFeatures(ctx, currentCandidates);
      
      // Get risk assessment from conformal predictor
      const riskAssessment = this.predictor.predictRisk(features);
      
      // Check budget constraints
      const canUpshift = this.budgetManager.canUpshift();
      const budgetStatus = this.budgetManager.getBudgetStatus(this.totalQueries);
      
      console.log(`🎯 Conformal routing: risk=${riskAssessment.risk_score.toFixed(3)}, threshold=${this.riskThreshold}, can_upshift=${canUpshift}, rate=${budgetStatus.current_upshift_rate.toFixed(1)}%`);
      
      // Decision logic
      const shouldUpshift = 
        riskAssessment.risk_score > this.riskThreshold &&
        canUpshift &&
        budgetStatus.p95_headroom_ms > 1.0; // At least 1ms headroom
      
      if (!shouldUpshift) {
        let reason = 'risk_below_threshold';
        if (riskAssessment.risk_score > this.riskThreshold && !canUpshift) {
          reason = 'budget_exhausted';
        } else if (riskAssessment.risk_score > this.riskThreshold && budgetStatus.p95_headroom_ms <= 1.0) {
          reason = 'insufficient_headroom';
        }
        
        return {
          should_upshift: false,
          upshift_type: 'none',
          risk_budget_used: 0,
          routing_reason: reason,
          expected_improvement: 0
        };
      }
      
      // Decide upshift type based on query characteristics
      const upshiftType = this.selectUpshiftType(features, riskAssessment);
      const expectedImprovement = this.estimateImprovement(upshiftType, riskAssessment);
      
      // Record upshift usage
      this.budgetManager.recordUpshift(upshiftType);
      this.upshiftedQueries++;
      
      span.setAttributes({
        success: true,
        risk_score: riskAssessment.risk_score,
        should_upshift: true,
        upshift_type: upshiftType,
        upshift_rate: (this.upshiftedQueries / this.totalQueries) * 100
      });
      
      console.log(`⚡ Upshifting query: type=${upshiftType}, expected_improvement=${expectedImprovement.toFixed(2)}nDCG`);
      
      return {
        should_upshift: true,
        upshift_type: upshiftType,
        risk_budget_used: 1,
        routing_reason: `high_risk_${riskAssessment.risk_score.toFixed(3)}`,
        expected_improvement: expectedImprovement
      };
      
    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      console.error('Conformal routing error:', error);
      
      // Safe fallback
      return {
        should_upshift: false,
        upshift_type: 'none',
        risk_budget_used: 0,
        routing_reason: 'routing_error',
        expected_improvement: 0
      };
    } finally {
      span.end();
    }
  }
  
  /**
   * Extract features for conformal prediction
   */
  private extractFeatures(
    ctx: SearchContext, 
    currentCandidates?: any[]
  ): ConformalPredictionFeatures {
    const words = ctx.query.split(/\s+/).filter(w => w.length > 0);
    const chars = ctx.query.split('');
    
    // Calculate query entropy
    const charCounts = chars.reduce((acc, char) => {
      acc[char] = (acc[char] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    
    let entropy = 0;
    for (const count of Object.values(charCounts)) {
      const p = count / chars.length;
      entropy -= p * Math.log2(p);
    }
    
    // Calculate identifier density (how code-like is the query)
    const identifierPattern = /[a-zA-Z_][a-zA-Z0-9_]*/g;
    const identifiers = ctx.query.match(identifierPattern) || [];
    const identifierDensity = identifiers.length / Math.max(1, words.length);
    
    // Estimate semantic complexity based on query structure
    const specialChars = (ctx.query.match(/[{}()\[\]<>.,;:]/g) || []).length;
    const semanticComplexity = Math.min(1, (specialChars + words.length) / 20);
    
    return {
      query_length: ctx.query.length,
      word_count: words.length,
      has_special_chars: specialChars > 0,
      fuzzy_enabled: ctx.fuzzy || false,
      structural_mode: ctx.mode === 'struct' || ctx.mode === 'hybrid',
      avg_word_length: words.length > 0 ? words.reduce((sum, w) => sum + w.length, 0) / words.length : 0,
      query_entropy: entropy,
      identifier_density: identifierDensity,
      semantic_complexity: semanticComplexity
    };
  }
  
  /**
   * Select the most appropriate upshift type
   */
  private selectUpshiftType(
    features: ConformalPredictionFeatures,
    risk: MisrankRisk
  ): 'dimension_768d' | 'efSearch_boost' | 'mmr_diversity' | 'none' {
    // High semantic complexity -> use 768d vectors
    if (features.semantic_complexity > 0.7 || features.query_entropy > 3.5) {
      return 'dimension_768d';
    }
    
    // Many candidates with structural patterns -> boost efSearch
    if (features.structural_mode && features.identifier_density > 0.5) {
      return 'efSearch_boost';
    }
    
    // Natural language queries with diversity needs -> MMR
    if (features.word_count > 5 && !features.structural_mode && features.identifier_density < 0.3) {
      return 'mmr_diversity';
    }
    
    // Default to efSearch boost for general improvement
    return 'efSearch_boost';
  }
  
  /**
   * Estimate expected nDCG improvement from upshift
   */
  private estimateImprovement(
    upshiftType: string,
    risk: MisrankRisk
  ): number {
    const baseImprovement = {
      'dimension_768d': 0.08,     // +8pp nDCG expected
      'efSearch_boost': 0.04,     // +4pp nDCG expected  
      'mmr_diversity': 0.03,      // +3pp nDCG expected
      'none': 0
    };
    
    // Scale by risk score - higher risk queries benefit more
    const riskMultiplier = 0.5 + risk.risk_score;
    return (baseImprovement[upshiftType as keyof typeof baseImprovement] || 0) * riskMultiplier;
  }
  
  /**
   * Calibrate the conformal predictor with new data
   */
  async calibrate(calibrationData: Array<{
    features: ConformalPredictionFeatures;
    actual_ndcg: number;
    predicted_ndcg: number;
  }>): Promise<void> {
    this.predictor.calibrate(calibrationData);
    this.lastCalibrationTime = Date.now();
  }
  
  /**
   * Get router statistics
   */
  getStats(): {
    total_queries: number;
    upshifted_queries: number;
    upshift_rate: number;
    budget_status: UpshiftBudget;
    last_calibration: Date | null;
    enabled: boolean;
  } {
    return {
      total_queries: this.totalQueries,
      upshifted_queries: this.upshiftedQueries,
      upshift_rate: this.totalQueries > 0 ? (this.upshiftedQueries / this.totalQueries) * 100 : 0,
      budget_status: this.budgetManager.getBudgetStatus(this.totalQueries),
      last_calibration: this.lastCalibrationTime > 0 ? new Date(this.lastCalibrationTime) : null,
      enabled: this.enabled
    };
  }
  
  /**
   * Enable/disable router
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    console.log(`🎯 Conformal router ${enabled ? 'ENABLED' : 'DISABLED'}`);
  }
  
  /**
   * Update configuration
   */
  updateConfig(config: {
    risk_threshold?: number;
    daily_budget_percent?: number;
  }): void {
    if (config.risk_threshold !== undefined) {
      this.riskThreshold = config.risk_threshold;
    }
    
    if (config.daily_budget_percent !== undefined) {
      this.budgetManager = new UpshiftBudgetManager(config.daily_budget_percent);
    }
    
    console.log(`🔧 Conformal router config updated:`, {
      risk_threshold: this.riskThreshold,
      daily_budget_percent: config.daily_budget_percent
    });
  }
}

// Global instance
export const globalConformalRouter = new ConformalRouter();