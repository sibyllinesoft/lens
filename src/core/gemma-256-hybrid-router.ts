/**
 * Gemma-256 Hybrid Routing System
 * 
 * Agent-ready calibrated routing with explicit pseudocode logic:
 * Route to 768d only when 256d result looks risky under SLA.
 * Uses calibrated signals with audit cap and budget checks.
 */

import type { SearchHit, SearchContext } from './span_resolver/types.js';
import type { QueryIntent, IntentClassification } from '../types/core.js';
import { classifyQuery } from './query-classifier.js';
import { Gemma256CalibrationManager } from './gemma-256-calibration.js';
import { LensTracer } from '../telemetry/tracer.js';

export interface HybridRoutingConfig {
  enabled: boolean;
  // Routing thresholds (from TODO pseudocode)
  confidence_threshold: number;      // p < 0.55 threshold
  margin_threshold: number;          // margin < 0.05 threshold  
  entropy_threshold: number;         // ent > 0.6 threshold
  pic_threshold: number;             // pic < 10 threshold
  upshift_cap_fraction: number;      // routed_768_fraction_last_10m < 0.05
  // Budget controls
  gpu_budget_check: boolean;         // Check GPU/CPU budget before upshift
  max_p95_cost_ms: number;          // Maximum p95 latency cost
  audit_logging: boolean;            // Log each upshift with reasons
  // Performance monitoring
  upshift_window_minutes: number;    // Window for upshift rate calculation
  max_upshift_rate: number;          // Maximum upshift rate (5%)
}

export interface RoutingDecision {
  use_768d: boolean;
  reason: string;
  confidence_256d: number;
  margin_256d: number;
  entropy: number;
  positives_in_candidates: number;
  intent: QueryIntent;
  upshift_rate_last_10m: number;
  budget_check_passed: boolean;
  predicted_p95_cost_ms: number;
}

export interface UpshiftEvent {
  timestamp: Date;
  query: string;
  intent: QueryIntent;
  confidence_256d: number;
  margin_256d: number;
  entropy: number;
  positives_in_candidates: number;
  reason: string;
  predicted_cost_ms: number;
  actual_latency_ms?: number;
}

export interface TopicEntropyCalculator {
  calculateEntropy(query: string): number;
}

export interface RAPTOREntropyCalculator extends TopicEntropyCalculator {
  // From RAPTOR hierarchy for topic entropy calculation
  topicDistribution: Map<string, number>;
}

export class RAPTOREntropyCalculatorImpl implements RAPTOREntropyCalculator {
  topicDistribution: Map<string, number>;
  
  constructor() {
    this.topicDistribution = new Map();
  }
  
  calculateEntropy(query: string): number {
    // Simplified RAPTOR topic entropy calculation
    const queryTokens = query.toLowerCase().split(/\s+/);
    const topicScores = new Map<string, number>();
    
    // Calculate topic scores based on query tokens
    for (const token of queryTokens) {
      for (const [topic, weight] of this.topicDistribution) {
        if (topic.includes(token) || token.includes(topic)) {
          topicScores.set(topic, (topicScores.get(topic) || 0) + weight);
        }
      }
    }
    
    if (topicScores.size === 0) {
      return 0.8; // High entropy for unknown queries
    }
    
    // Normalize and calculate entropy
    const totalScore = Array.from(topicScores.values()).reduce((sum, score) => sum + score, 0);
    let entropy = 0;
    
    for (const score of topicScores.values()) {
      const prob = score / totalScore;
      if (prob > 0) {
        entropy -= prob * Math.log2(prob);
      }
    }
    
    return Math.min(1.0, entropy / Math.log2(topicScores.size || 1));
  }
}

/**
 * Intent Classifier with expanded QueryIntent support
 */
export class IntentClassifier {
  
  /**
   * Classify query intent as per pseudocode: {NL,symbol,struct,lexical}
   */
  classifyIntent(query: string): IntentClassification {
    const normalized = query.toLowerCase().trim();
    const words = normalized.split(/\s+/);
    
    let confidence = 0.5;
    let intent: QueryIntent = 'lexical'; // Default
    
    const features = {
      has_definition_pattern: false,
      has_reference_pattern: false,
      has_symbol_prefix: false,
      has_structural_chars: false,
      is_natural_language: false
    };
    
    // Check for definition patterns
    if (/\b(def|define|definition|function|class|interface)\b/.test(normalized)) {
      features.has_definition_pattern = true;
      intent = 'def';
      confidence += 0.3;
    }
    
    // Check for reference patterns  
    if (/\b(refs|references|usages|calls|uses)\b/.test(normalized)) {
      features.has_reference_pattern = true;
      intent = 'refs';
      confidence += 0.3;
    }
    
    // Check for symbol patterns
    if (/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(normalized) || 
        /\b(symbol|identifier|name)\b/.test(normalized)) {
      features.has_symbol_prefix = true;
      intent = 'symbol';
      confidence += 0.25;
    }
    
    // Check for structural patterns
    if (/[(){}\[\]<>]/.test(query) || /\b(struct|structure|ast|tree)\b/.test(normalized)) {
      features.has_structural_chars = true;
      intent = 'struct';
      confidence += 0.2;
    }
    
    // Check for natural language (high priority)
    const nlClassification = classifyQuery(query);
    if (nlClassification.isNaturalLanguage) {
      features.is_natural_language = true;
      intent = 'NL';
      confidence = nlClassification.confidence;
    }
    
    return {
      intent,
      confidence: Math.max(0, Math.min(1, confidence)),
      features
    };
  }
}

/**
 * Hybrid Router implementing the exact pseudocode from TODO
 */
export class Gemma256HybridRouter {
  private calibrationManager: Gemma256CalibrationManager;
  private intentClassifier: IntentClassifier;
  private entropyCalculator: TopicEntropyCalculator;
  private upshiftHistory: UpshiftEvent[] = [];
  private routingStats = {
    total_queries: 0,
    upshifted_queries: 0,
    budget_rejections: 0,
    cap_rejections: 0
  };

  constructor(
    private config: HybridRoutingConfig,
    calibrationManager: Gemma256CalibrationManager,
    entropyCalculator?: TopicEntropyCalculator
  ) {
    this.calibrationManager = calibrationManager;
    this.intentClassifier = new IntentClassifier();
    
    // Default RAPTOR entropy calculator if none provided
    this.entropyCalculator = entropyCalculator || new RAPTOREntropyCalculator();
    
    console.log(`🚀 Gemma-256 Hybrid Router initialized`);
    console.log(`   Upshift cap: ${config.upshift_cap_fraction * 100}%`);
    console.log(`   Budget check: ${config.gpu_budget_check}`);
    console.log(`   Audit logging: ${config.audit_logging}`);
  }

  /**
   * Main routing decision implementing exact TODO pseudocode
   */
  async makeRoutingDecision(
    query: string,
    candidates256: SearchHit[],
    context: SearchContext
  ): Promise<RoutingDecision> {
    const span = LensTracer.createChildSpan('hybrid_routing_decision', {
      'query': query,
      'candidates.count': candidates256.length,
      'routing.enabled': this.config.enabled
    });

    const startTime = Date.now();
    this.routingStats.total_queries++;

    try {
      if (!this.config.enabled) {
        return {
          use_768d: false,
          reason: 'hybrid_routing_disabled',
          confidence_256d: 1.0,
          margin_256d: 1.0,
          entropy: 0.0,
          positives_in_candidates: candidates256.length,
          intent: 'lexical',
          upshift_rate_last_10m: 0.0,
          budget_check_passed: true,
          predicted_p95_cost_ms: 0
        };
      }

      // EXACT PSEUDOCODE IMPLEMENTATION:
      
      // intent = classify(query)            # {NL,symbol,struct,lexical}
      const intentClassification = this.intentClassifier.classifyIntent(query);
      const intent = intentClassification.intent;
      
      // p = calibrated_top1_prob_256        # from isotonic_256
      const p = this.calculateCalibratedTop1Prob(candidates256);
      
      // margin = score1_256 - score2_256
      const margin = this.calculateScoreMargin(candidates256);
      
      // ent = topic_entropy(query)          # from RAPTOR
      const ent = this.entropyCalculator.calculateEntropy(query);
      
      // pic = positives_in_candidates_256   # Stage-A/B
      const pic = this.countPositivesInCandidates(candidates256);
      
      // hard = (intent in {NL,symbol}) and (p < 0.55 or margin < 0.05 or ent > 0.6 or pic < 10)
      const isHardQuery = (intent === 'NL' || intent === 'symbol') && (
        p < this.config.confidence_threshold ||
        margin < this.config.margin_threshold ||
        ent > this.config.entropy_threshold ||
        pic < this.config.pic_threshold
      );
      
      // routed_768_fraction_last_10m < 0.05
      const upshiftRate = this.calculateUpshiftRate();
      const withinUpshiftCap = upshiftRate < this.config.upshift_cap_fraction;
      
      // GPU/CPU budget check: only upshift if predicted p95 cost fits the SLA bucket
      const budgetCheck = await this.checkBudgetConstraints(query, context);
      
      // if hard and routed_768_fraction_last_10m < 0.05: use 768d else use 256d
      const shouldUpshift = isHardQuery && withinUpshiftCap && budgetCheck.passed;

      const decision: RoutingDecision = {
        use_768d: shouldUpshift,
        reason: this.generateReason(isHardQuery, withinUpshiftCap, budgetCheck.passed, intent, p, margin, ent, pic),
        confidence_256d: p,
        margin_256d: margin,
        entropy: ent,
        positives_in_candidates: pic,
        intent,
        upshift_rate_last_10m: upshiftRate,
        budget_check_passed: budgetCheck.passed,
        predicted_p95_cost_ms: budgetCheck.predicted_cost_ms
      };

      // Log upshift event if applicable
      if (shouldUpshift) {
        await this.logUpshiftEvent(query, decision, budgetCheck.predicted_cost_ms);
        this.routingStats.upshifted_queries++;
      } else if (isHardQuery && !withinUpshiftCap) {
        this.routingStats.cap_rejections++;
      } else if (isHardQuery && !budgetCheck.passed) {
        this.routingStats.budget_rejections++;
      }

      const latency = Date.now() - startTime;
      
      span.setAttributes({
        success: true,
        decision_use_768d: shouldUpshift,
        intent,
        confidence_256d: p,
        margin_256d: margin,
        entropy: ent,
        positives_in_candidates: pic,
        upshift_rate: upshiftRate,
        budget_passed: budgetCheck.passed,
        latency_ms: latency
      });

      return decision;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      
      // Fallback to 256d on error
      return {
        use_768d: false,
        reason: `routing_error: ${error instanceof Error ? error.message : 'unknown'}`,
        confidence_256d: 0.5,
        margin_256d: 0.5,
        entropy: 0.5,
        positives_in_candidates: candidates256.length,
        intent: 'lexical',
        upshift_rate_last_10m: this.calculateUpshiftRate(),
        budget_check_passed: false,
        predicted_p95_cost_ms: 0
      };
    } finally {
      span.end();
    }
  }

  /**
   * Calculate calibrated top-1 probability from isotonic_256
   */
  private calculateCalibratedTop1Prob(candidates: SearchHit[]): number {
    if (candidates.length === 0) {
      return 0.0;
    }

    // Get top candidate score
    const topScore = Math.max(...candidates.map(c => c.score));
    
    // Apply 256d calibration
    const calibratedScore = this.calibrationManager.calibrateScores(topScore, topScore);
    
    // Convert to probability (assume scores are already similarity-like)
    return Math.max(0, Math.min(1, calibratedScore.calibrated256));
  }

  /**
   * Calculate margin between top 2 candidates (score1_256 - score2_256)
   */
  private calculateScoreMargin(candidates: SearchHit[]): number {
    if (candidates.length < 2) {
      return 1.0; // High margin if insufficient candidates
    }

    // Sort by score descending
    const sortedScores = candidates.map(c => c.score).sort((a, b) => b - a);
    
    return sortedScores[0]! - sortedScores[1]!;
  }

  /**
   * Count positives in candidates (Stage-A/B results)
   * For now, assume candidates with score > threshold are "positive"
   */
  private countPositivesInCandidates(candidates: SearchHit[]): number {
    const positiveThreshold = 0.1; // Configurable threshold for "positive" candidates
    return candidates.filter(c => c.score > positiveThreshold).length;
  }

  /**
   * Calculate upshift rate over the last 10 minutes
   */
  private calculateUpshiftRate(): number {
    const now = new Date();
    const windowStart = new Date(now.getTime() - this.config.upshift_window_minutes * 60 * 1000);
    
    const recentEvents = this.upshiftHistory.filter(event => 
      event.timestamp >= windowStart && event.timestamp <= now
    );
    
    const totalQueries = this.routingStats.total_queries;
    if (totalQueries === 0) {
      return 0.0;
    }
    
    return recentEvents.length / totalQueries;
  }

  /**
   * GPU/CPU budget check: only upshift if predicted p95 cost fits the SLA bucket
   */
  private async checkBudgetConstraints(
    query: string, 
    context: SearchContext
  ): Promise<{ passed: boolean; predicted_cost_ms: number }> {
    if (!this.config.gpu_budget_check) {
      return { passed: true, predicted_cost_ms: 0 };
    }

    // Simplified budget estimation (in production, would use actual resource monitoring)
    const baseLatency = 8; // ms for 256d
    const upshiftCost = 15; // additional ms for 768d
    const queryComplexity = Math.min(2.0, query.length / 50); // Complexity factor
    
    const predictedCost = baseLatency + (upshiftCost * queryComplexity);
    const passed = predictedCost <= this.config.max_p95_cost_ms;
    
    return { passed, predicted_cost_ms: predictedCost };
  }

  /**
   * Generate human-readable reason for routing decision
   */
  private generateReason(
    isHard: boolean,
    withinCap: boolean,
    budgetPassed: boolean,
    intent: QueryIntent,
    confidence: number,
    margin: number,
    entropy: number,
    pic: number
  ): string {
    if (!isHard) {
      return `easy_query_256d: intent=${intent}, conf=${confidence.toFixed(3)}, margin=${margin.toFixed(3)}`;
    }
    
    if (!withinCap) {
      return `hard_query_capped: upshift_rate_exceeded`;
    }
    
    if (!budgetPassed) {
      return `hard_query_budget_rejected: p95_cost_too_high`;
    }
    
    const hardReasons = [];
    if (confidence < this.config.confidence_threshold) {
      hardReasons.push(`conf=${confidence.toFixed(3)}<${this.config.confidence_threshold}`);
    }
    if (margin < this.config.margin_threshold) {
      hardReasons.push(`margin=${margin.toFixed(3)}<${this.config.margin_threshold}`);
    }
    if (entropy > this.config.entropy_threshold) {
      hardReasons.push(`ent=${entropy.toFixed(3)}>${this.config.entropy_threshold}`);
    }
    if (pic < this.config.pic_threshold) {
      hardReasons.push(`pic=${pic}<${this.config.pic_threshold}`);
    }
    
    return `hard_query_768d: ${hardReasons.join(', ')}`;
  }

  /**
   * Log upshift event with reasons for audit
   */
  private async logUpshiftEvent(
    query: string,
    decision: RoutingDecision,
    predictedCost: number
  ): Promise<void> {
    const event: UpshiftEvent = {
      timestamp: new Date(),
      query,
      intent: decision.intent,
      confidence_256d: decision.confidence_256d,
      margin_256d: decision.margin_256d,
      entropy: decision.entropy,
      positives_in_candidates: decision.positives_in_candidates,
      reason: decision.reason,
      predicted_cost_ms: predictedCost
    };

    this.upshiftHistory.push(event);

    // Keep history bounded
    if (this.upshiftHistory.length > 1000) {
      this.upshiftHistory = this.upshiftHistory.slice(-500);
    }

    // Audit logging if enabled
    if (this.config.audit_logging) {
      console.log(`🔀 UPSHIFT: ${query} | ${decision.intent} | ${decision.reason} | cost=${predictedCost.toFixed(1)}ms`);
    }
  }

  /**
   * Update actual latency for completed upshift (for learning)
   */
  updateActualLatency(query: string, actualLatency: number): void {
    // Find recent upshift event for this query and update actual latency
    const recentEvent = this.upshiftHistory
      .slice(-10) // Check last 10 events
      .find(event => event.query === query && !event.actual_latency_ms);
    
    if (recentEvent) {
      recentEvent.actual_latency_ms = actualLatency;
    }
  }

  /**
   * Get comprehensive routing statistics
   */
  getStats() {
    const upshiftRate = this.calculateUpshiftRate();
    
    return {
      config: this.config,
      stats: {
        ...this.routingStats,
        upshift_rate_current: upshiftRate,
        upshift_history_size: this.upshiftHistory.length
      },
      recent_upshifts: this.upshiftHistory.slice(-10), // Last 10 events
      performance: {
        avg_predicted_cost: this.upshiftHistory.length > 0 ?
          this.upshiftHistory.reduce((sum, e) => sum + e.predicted_cost_ms, 0) / this.upshiftHistory.length : 0,
        avg_actual_cost: this.upshiftHistory.filter(e => e.actual_latency_ms).length > 0 ?
          this.upshiftHistory
            .filter(e => e.actual_latency_ms)
            .reduce((sum, e) => sum + e.actual_latency_ms!, 0) / 
          this.upshiftHistory.filter(e => e.actual_latency_ms).length : 0
      }
    };
  }

  /**
   * Update configuration for tuning
   */
  updateConfig(newConfig: Partial<HybridRoutingConfig>): void {
    this.config = { ...this.config, ...newConfig };
    console.log(`🔀 Hybrid router config updated: ${JSON.stringify(this.config)}`);
  }

  /**
   * Validate routing configuration meets production requirements
   */
  validateConfiguration(): { valid: boolean; issues: string[] } {
    const issues: string[] = [];

    if (this.config.upshift_cap_fraction > 0.05) {
      issues.push(`Upshift cap ${this.config.upshift_cap_fraction} exceeds 5% requirement`);
    }

    if (!this.config.gpu_budget_check) {
      issues.push('GPU budget check is disabled in production');
    }

    if (!this.config.audit_logging) {
      issues.push('Audit logging is disabled in production');
    }

    if (this.config.confidence_threshold !== 0.55) {
      issues.push(`Confidence threshold ${this.config.confidence_threshold} != 0.55 from pseudocode`);
    }

    if (this.config.margin_threshold !== 0.05) {
      issues.push(`Margin threshold ${this.config.margin_threshold} != 0.05 from pseudocode`);
    }

    if (this.config.entropy_threshold !== 0.6) {
      issues.push(`Entropy threshold ${this.config.entropy_threshold} != 0.6 from pseudocode`);
    }

    if (this.config.pic_threshold !== 10) {
      issues.push(`PIC threshold ${this.config.pic_threshold} != 10 from pseudocode`);
    }

    return {
      valid: issues.length === 0,
      issues
    };
  }
}

/**
 * Production Hybrid Routing Manager
 * Orchestrates calibrated routing with strict production requirements
 */
export class Gemma256HybridRoutingManager {
  private router: Gemma256HybridRouter;
  private isProduction: boolean;

  constructor(
    calibrationManager: Gemma256CalibrationManager,
    entropyCalculator?: TopicEntropyCalculator,
    config: Partial<HybridRoutingConfig> = {},
    isProduction = true
  ) {
    this.isProduction = isProduction;

    // Production defaults matching exact TODO pseudocode
    const productionConfig: HybridRoutingConfig = {
      enabled: true,
      // Exact thresholds from pseudocode
      confidence_threshold: 0.55,         // p < 0.55
      margin_threshold: 0.05,             // margin < 0.05  
      entropy_threshold: 0.6,             // ent > 0.6
      pic_threshold: 10,                  // pic < 10
      upshift_cap_fraction: 0.05,         // < 5% upshift rate
      // Production safety requirements
      gpu_budget_check: true,             // Mandatory budget checking
      max_p95_cost_ms: 25,                // SLA compliance
      audit_logging: true,                // Mandatory audit trail
      upshift_window_minutes: 10,         // 10-minute window
      max_upshift_rate: 0.05,             // 5% max rate
      ...config
    };

    this.router = new Gemma256HybridRouter(
      productionConfig,
      calibrationManager,
      entropyCalculator
    );

    // Validate production configuration
    if (isProduction) {
      const validation = this.router.validateConfiguration();
      if (!validation.valid) {
        throw new Error(`Production router validation failed: ${validation.issues.join(', ')}`);
      }
    }

    console.log(`🚀 Gemma-256 Hybrid Routing Manager initialized (production=${isProduction})`);
    console.log(`   Exact pseudocode implementation with 5% upshift cap`);
    console.log(`   Budget checking: ${productionConfig.gpu_budget_check}`);
  }

  /**
   * Make production routing decision with full audit trail
   */
  async routeQuery(
    query: string,
    candidates256: SearchHit[],
    context: SearchContext
  ): Promise<RoutingDecision> {
    if (this.isProduction) {
      // Additional production validation
      if (!query || query.length === 0) {
        throw new Error('Empty query not allowed in production');
      }
      
      if (candidates256.length === 0) {
        console.warn(`⚠️ No 256d candidates for query: ${query}`);
      }
    }

    return await this.router.makeRoutingDecision(query, candidates256, context);
  }

  /**
   * Get comprehensive statistics for monitoring
   */
  getStats() {
    return {
      production_mode: this.isProduction,
      ...this.router.getStats()
    };
  }

  /**
   * Update routing configuration with production safety checks
   */
  updateConfig(newConfig: Partial<HybridRoutingConfig>): void {
    if (this.isProduction) {
      // Enforce production constraints
      if (newConfig.upshift_cap_fraction && newConfig.upshift_cap_fraction > 0.05) {
        throw new Error('Cannot exceed 5% upshift cap in production');
      }
      
      if (newConfig.gpu_budget_check === false) {
        throw new Error('Cannot disable budget checking in production');
      }
      
      if (newConfig.audit_logging === false) {
        throw new Error('Cannot disable audit logging in production');
      }
    }

    this.router.updateConfig(newConfig);
  }

  /**
   * Emergency circuit breaker - disable routing temporarily
   */
  emergencyDisable(reason: string): void {
    console.warn(`🚨 Emergency disabling hybrid routing: ${reason}`);
    this.router.updateConfig({ enabled: false });
  }

  /**
   * Re-enable routing after emergency disable
   */
  emergencyRenable(): void {
    console.log(`✅ Re-enabling hybrid routing after emergency disable`);
    this.router.updateConfig({ enabled: true });
  }
}