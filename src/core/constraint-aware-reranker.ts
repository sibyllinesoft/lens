/**
 * Constraint-Aware Reranker - Embedder-Agnostic Optimization #1
 * 
 * Replaces "soft features + caps" with monotone, floorable scoring using a GAM/monotone-isotonic 
 * layer where exact token, same-file symbol-def, and AST pattern hits are non-decreasing contributors.
 * 
 * Mathematical constraint: score(x) = Σᵢ fᵢ(xᵢ) with f_exact, f_struct monotone and Δscore_exact ≥ α
 * Pairwise constraints: if two candidates equal on all features except exact_symbol_match, 
 * the match candidate can't rank lower.
 * 
 * Target: SLA-Recall@50≥0 and ΔnDCG@10 ≥ +0.5pp on symbol/NL with ECE Δ≤0.01
 */

import type { SearchHit, MatchReason } from '../core/span_resolver/types.js';
import type { SearchContext } from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';

export interface ConstraintAwareConfig {
  enabled: boolean;
  alpha: number;                    // Floor for exact matches (log-odds), ~0.5
  maxLatencyMs: number;            // Budget for constraint evaluation
  auditFloorWins: boolean;         // Log floor-win events for audit
  monotonicityCheckEnabled: boolean; // Runtime monotonicity validation
  exactMatchMinScore: number;      // Minimum score boost for exact matches
  sameFileSymbolBoost: number;     // Additional boost for same-file symbols
  structuralPatternBoost: number;  // Boost for structural/AST pattern matches
}

export interface FeatureVector {
  exact_token_match: number;       // [0, 1] - exact token match strength
  same_file_symbol_def: number;    // [0, 1] - symbol definition in same file
  ast_pattern_match: number;       // [0, 1] - AST/structural pattern match
  semantic_score: number;          // [0, 1] - base semantic similarity score
  lexical_score: number;          // [0, 1] - lexical/fuzzy match score
  symbol_relevance: number;        // [0, 1] - symbol-specific relevance
}

export interface ConstraintViolation {
  candidateA: string;              // Identifier for first candidate
  candidateB: string;              // Identifier for second candidate
  violationType: 'exact_match' | 'same_file' | 'structural';
  expectedOrder: 'A>B' | 'B>A';   // Expected ordering based on constraints
  actualOrder: 'A>B' | 'B>A';     // Actual ordering after scoring
  scoreA: number;
  scoreB: number;
}

/**
 * Monotonic feature function that ensures non-decreasing contributions
 */
export class MonotonicFeatureFunction {
  private coefficients: number[] = [];
  private breakpoints: number[] = [];
  private values: number[] = [];
  
  constructor(
    private featureName: string,
    private alpha: number = 0.5 // Minimum contribution for positive features
  ) {}

  /**
   * Fit isotonic regression to ensure monotonicity
   */
  fit(xValues: number[], yValues: number[]): void {
    if (xValues.length !== yValues.length) {
      throw new Error(`Feature ${this.featureName}: x and y values must have same length`);
    }

    // Sort by x values for isotonic regression
    const paired = xValues.map((x, i) => ({ x, y: yValues[i]! }))
                           .sort((a, b) => a.x - b.x);

    // Pool-Adjacent-Violators Algorithm (PAVA) for isotonic regression
    const pairedResult = this.applyPAVA(paired);
    
    this.breakpoints = pairedResult.map(p => p.x);
    this.values = pairedResult.map(p => p.y);
  }

  /**
   * Pool-Adjacent-Violators Algorithm implementation
   */
  private applyPAVA(pairs: Array<{x: number, y: number}>): Array<{x: number, y: number}> {
    if (pairs.length === 0) return [];

    const result = [...pairs];
    let changed = true;

    while (changed) {
      changed = false;
      for (let i = 0; i < result.length - 1; i++) {
        if (result[i]!.y > result[i + 1]!.y) {
          // Violation found - merge adjacent points
          const merged = {
            x: result[i]!.x, // Keep first x value
            y: (result[i]!.y + result[i + 1]!.y) / 2 // Average y values
          };
          
          result[i] = merged;
          result.splice(i + 1, 1);
          changed = true;
          break;
        }
      }
    }

    return result;
  }

  /**
   * Apply monotonic function to feature value with floor constraint
   */
  apply(x: number): number {
    if (this.breakpoints.length === 0) {
      return Math.max(this.alpha * x, 0); // Fallback linear with floor
    }

    // Linear interpolation between breakpoints
    if (x <= this.breakpoints[0]!) {
      return this.values[0]!;
    }
    
    if (x >= this.breakpoints[this.breakpoints.length - 1]!) {
      const lastValue = this.values[this.values.length - 1]!;
      return Math.max(lastValue, this.alpha * x); // Apply floor constraint
    }

    // Find interpolation bounds
    for (let i = 0; i < this.breakpoints.length - 1; i++) {
      if (x >= this.breakpoints[i]! && x <= this.breakpoints[i + 1]!) {
        const x0 = this.breakpoints[i]!;
        const x1 = this.breakpoints[i + 1]!;
        const y0 = this.values[i]!;
        const y1 = this.values[i + 1]!;
        
        const alpha = (x - x0) / (x1 - x0);
        const interpolatedValue = y0 + alpha * (y1 - y0);
        
        // Apply floor constraint
        return Math.max(interpolatedValue, this.alpha * x);
      }
    }

    return Math.max(this.alpha * x, 0);
  }

  /**
   * Validate that the function maintains monotonicity
   */
  validateMonotonicity(testPoints: number[] = []): boolean {
    if (testPoints.length === 0) {
      testPoints = Array.from({length: 100}, (_, i) => i / 99);
    }

    for (let i = 0; i < testPoints.length - 1; i++) {
      const x1 = testPoints[i]!;
      const x2 = testPoints[i + 1]!;
      const y1 = this.apply(x1);
      const y2 = this.apply(x2);
      
      if (x1 < x2 && y1 > y2) {
        console.warn(`Monotonicity violation in ${this.featureName}: f(${x1})=${y1} > f(${x2})=${y2}`);
        return false;
      }
    }
    
    return true;
  }
}

/**
 * Constraint-Aware Reranker implementing monotonic GAM with pairwise constraints
 */
export class ConstraintAwareReranker {
  private config: ConstraintAwareConfig;
  private exactMatchFunction: MonotonicFeatureFunction;
  private sameFileFunction: MonotonicFeatureFunction;
  private structuralFunction: MonotonicFeatureFunction;
  private violationLog: ConstraintViolation[] = [];

  constructor(config: Partial<ConstraintAwareConfig> = {}) {
    this.config = {
      enabled: config.enabled ?? true,
      alpha: config.alpha ?? 0.5,
      maxLatencyMs: config.maxLatencyMs ?? 2, // Very tight budget
      auditFloorWins: config.auditFloorWins ?? true,
      monotonicityCheckEnabled: config.monotonicityCheckEnabled ?? true,
      exactMatchMinScore: config.exactMatchMinScore ?? 0.3,
      sameFileSymbolBoost: config.sameFileSymbolBoost ?? 0.2,
      structuralPatternBoost: config.structuralPatternBoost ?? 0.15,
      ...config
    };

    // Initialize monotonic feature functions
    this.exactMatchFunction = new MonotonicFeatureFunction('exact_token_match', this.config.alpha);
    this.sameFileFunction = new MonotonicFeatureFunction('same_file_symbol_def', this.config.alpha);
    this.structuralFunction = new MonotonicFeatureFunction('ast_pattern_match', this.config.alpha);

    console.log(`🔧 ConstraintAwareReranker initialized: alpha=${this.config.alpha}, enabled=${this.config.enabled}`);
  }

  /**
   * Main reranking function with constraint enforcement
   */
  async rerank(hits: SearchHit[], context: SearchContext): Promise<SearchHit[]> {
    const span = LensTracer.createChildSpan('constraint_aware_rerank', {
      'candidates': hits.length,
      'query': context.query,
      'enabled': this.config.enabled
    });

    const startTime = performance.now();

    try {
      if (!this.config.enabled || hits.length === 0) {
        span.setAttributes({ skipped: true, reason: 'disabled_or_empty' });
        return hits;
      }

      // Budget check
      const checkBudget = () => {
        const elapsed = performance.now() - startTime;
        if (elapsed > this.config.maxLatencyMs) {
          throw new Error(`Constraint reranker budget exceeded: ${elapsed.toFixed(2)}ms > ${this.config.maxLatencyMs}ms`);
        }
      };

      // Extract features for all candidates
      const candidatesWithFeatures = hits.map(hit => ({
        hit,
        features: this.extractFeatures(hit, context),
        originalIndex: hits.indexOf(hit)
      }));

      checkBudget();

      // Apply GAM scoring with monotonic constraints
      const candidatesWithScores = candidatesWithFeatures.map(candidate => ({
        ...candidate,
        constraintAwareScore: this.computeGAMScore(candidate.features),
        originalScore: candidate.hit.score
      }));

      checkBudget();

      // Validate pairwise constraints
      const violations = this.validatePairwiseConstraints(candidatesWithScores);
      
      if (this.config.auditFloorWins && violations.length > 0) {
        console.log(`⚠️  Constraint violations detected: ${violations.length} pairs`);
        this.violationLog.push(...violations);
      }

      checkBudget();

      // Apply constraint corrections
      const correctedCandidates = this.applyConstraintCorrections(candidatesWithScores, violations);

      // Sort by constraint-aware score
      correctedCandidates.sort((a, b) => b.constraintAwareScore - a.constraintAwareScore);

      // Update scores in original hits
      const rerankedHits = correctedCandidates.map(candidate => ({
        ...candidate.hit,
        score: candidate.constraintAwareScore
      }));

      const latency = performance.now() - startTime;

      span.setAttributes({
        success: true,
        latency_ms: latency,
        candidates_processed: hits.length,
        violations_detected: violations.length,
        violations_corrected: violations.filter(v => this.wasViolationCorrected(v, correctedCandidates)).length
      });

      if (this.config.auditFloorWins && violations.length > 0) {
        console.log(`🎯 ConstraintAware rerank: ${hits.length} candidates, ${violations.length} violations in ${latency.toFixed(2)}ms`);
      }

      return rerankedHits;

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });

      console.warn(`ConstraintAware reranker failed: ${errorMsg}, falling back to original order`);
      return hits;

    } finally {
      span.end();
    }
  }

  /**
   * Extract feature vector from search hit
   */
  private extractFeatures(hit: SearchHit, context: SearchContext): FeatureVector {
    const features: FeatureVector = {
      exact_token_match: 0,
      same_file_symbol_def: 0,
      ast_pattern_match: 0,
      semantic_score: Math.min(hit.score, 1.0), // Normalize to [0,1]
      lexical_score: 0,
      symbol_relevance: 0
    };

    // Exact token match detection
    const queryTokens = context.query.toLowerCase().split(/\s+/);
    const snippetTokens = (hit.snippet || '').toLowerCase().split(/\s+/);
    const exactMatches = queryTokens.filter(token => snippetTokens.includes(token));
    features.exact_token_match = exactMatches.length / Math.max(queryTokens.length, 1);

    // Same file symbol definition detection
    if (hit.why.includes('exact') || hit.symbol_kind) {
      features.same_file_symbol_def = hit.why.includes('symbol') ? 1.0 : 0.5;
    }

    // AST/structural pattern match
    if (hit.why.includes('struct') || hit.why.includes('structural') || hit.pattern_type) {
      features.ast_pattern_match = 1.0;
    }

    // Lexical scoring based on match reasons
    if (hit.why.includes('exact')) {
      features.lexical_score = 1.0;
    } else if (hit.why.includes('fuzzy')) {
      features.lexical_score = 0.7;
    }

    // Symbol relevance
    if (hit.symbol_kind && hit.symbol_name) {
      const symbolNameMatch = queryTokens.some(token => 
        hit.symbol_name!.toLowerCase().includes(token) || 
        token.includes(hit.symbol_name!.toLowerCase())
      );
      features.symbol_relevance = symbolNameMatch ? 1.0 : 0.5;
    }

    return features;
  }

  /**
   * Compute GAM score with monotonic feature functions
   */
  private computeGAMScore(features: FeatureVector): number {
    // Apply monotonic feature functions
    const exactContribution = this.exactMatchFunction.apply(features.exact_token_match);
    const sameFileContribution = this.sameFileFunction.apply(features.same_file_symbol_def);
    const structuralContribution = this.structuralFunction.apply(features.ast_pattern_match);

    // Base semantic contribution
    const semanticContribution = features.semantic_score * 0.4; // Reduced weight to allow constraint dominance

    // Additional feature contributions
    const lexicalContribution = features.lexical_score * 0.2;
    const symbolContribution = features.symbol_relevance * 0.15;

    // Additive GAM: score(x) = Σᵢ fᵢ(xᵢ)
    const gamScore = 
      exactContribution +
      sameFileContribution +
      structuralContribution +
      semanticContribution +
      lexicalContribution +
      symbolContribution;

    // Apply floor constraints
    let finalScore = gamScore;

    // Floor for exact matches: Δscore_exact ≥ α
    if (features.exact_token_match > 0.5) {
      finalScore = Math.max(finalScore, this.config.exactMatchMinScore);
    }

    // Additional boosts based on configuration
    if (features.same_file_symbol_def > 0.5) {
      finalScore += this.config.sameFileSymbolBoost;
    }

    if (features.ast_pattern_match > 0.5) {
      finalScore += this.config.structuralPatternBoost;
    }

    return Math.max(0, Math.min(1, finalScore)); // Clamp to [0,1]
  }

  /**
   * Validate pairwise constraints between candidates
   */
  private validatePairwiseConstraints(
    candidates: Array<{hit: SearchHit, features: FeatureVector, constraintAwareScore: number, originalIndex: number}>
  ): ConstraintViolation[] {
    const violations: ConstraintViolation[] = [];

    for (let i = 0; i < candidates.length; i++) {
      for (let j = i + 1; j < candidates.length; j++) {
        const candA = candidates[i]!;
        const candB = candidates[j]!;

        // Check exact match constraint
        const exactDiff = candA.features.exact_token_match - candB.features.exact_token_match;
        if (Math.abs(exactDiff) < 1e-6) { // Features equal
          continue; // No constraint applies
        }

        const expectedOrder = exactDiff > 0 ? 'A>B' as const : 'B>A' as const;
        const actualOrder = candA.constraintAwareScore > candB.constraintAwareScore ? 'A>B' as const : 'B>A' as const;

        if (expectedOrder !== actualOrder) {
          violations.push({
            candidateA: `${candA.hit.file}:${candA.hit.line}`,
            candidateB: `${candB.hit.file}:${candB.hit.line}`,
            violationType: 'exact_match',
            expectedOrder,
            actualOrder,
            scoreA: candA.constraintAwareScore,
            scoreB: candB.constraintAwareScore
          });
        }

        // Check same file symbol constraint
        const sameFileDiff = candA.features.same_file_symbol_def - candB.features.same_file_symbol_def;
        if (Math.abs(sameFileDiff) > 1e-6) {
          const expectedOrderSameFile = sameFileDiff > 0 ? 'A>B' as const : 'B>A' as const;
          if (expectedOrderSameFile !== actualOrder) {
            violations.push({
              candidateA: `${candA.hit.file}:${candA.hit.line}`,
              candidateB: `${candB.hit.file}:${candB.hit.line}`,
              violationType: 'same_file',
              expectedOrder: expectedOrderSameFile,
              actualOrder,
              scoreA: candA.constraintAwareScore,
              scoreB: candB.constraintAwareScore
            });
          }
        }

        // Check structural constraint
        const structuralDiff = candA.features.ast_pattern_match - candB.features.ast_pattern_match;
        if (Math.abs(structuralDiff) > 1e-6) {
          const expectedOrderStructural = structuralDiff > 0 ? 'A>B' as const : 'B>A' as const;
          if (expectedOrderStructural !== actualOrder) {
            violations.push({
              candidateA: `${candA.hit.file}:${candA.hit.line}`,
              candidateB: `${candB.hit.file}:${candB.hit.line}`,
              violationType: 'structural',
              expectedOrder: expectedOrderStructural,
              actualOrder,
              scoreA: candA.constraintAwareScore,
              scoreB: candB.constraintAwareScore
            });
          }
        }
      }
    }

    return violations;
  }

  /**
   * Apply corrections to enforce pairwise constraints
   */
  private applyConstraintCorrections(
    candidates: Array<{hit: SearchHit, features: FeatureVector, constraintAwareScore: number, originalIndex: number}>,
    violations: ConstraintViolation[]
  ): Array<{hit: SearchHit, features: FeatureVector, constraintAwareScore: number, originalIndex: number}> {
    const correctedCandidates = [...candidates];

    // Group violations by candidate pairs and apply corrections
    for (const violation of violations) {
      const candAIndex = correctedCandidates.findIndex(c => 
        `${c.hit.file}:${c.hit.line}` === violation.candidateA
      );
      const candBIndex = correctedCandidates.findIndex(c => 
        `${c.hit.file}:${c.hit.line}` === violation.candidateB
      );

      if (candAIndex === -1 || candBIndex === -1) continue;

      const candA = correctedCandidates[candAIndex]!;
      const candB = correctedCandidates[candBIndex]!;

      // Apply constraint-based score adjustment
      if (violation.expectedOrder === 'A>B' && violation.actualOrder === 'B>A') {
        // Boost A's score to be at least α higher than B's score
        correctedCandidates[candAIndex] = {
          ...candA,
          constraintAwareScore: Math.max(candA.constraintAwareScore, candB.constraintAwareScore + this.config.alpha)
        };
      } else if (violation.expectedOrder === 'B>A' && violation.actualOrder === 'A>B') {
        // Boost B's score to be at least α higher than A's score
        correctedCandidates[candBIndex] = {
          ...candB,
          constraintAwareScore: Math.max(candB.constraintAwareScore, candA.constraintAwareScore + this.config.alpha)
        };
      }
    }

    return correctedCandidates;
  }

  /**
   * Check if a violation was corrected in the final candidate set
   */
  private wasViolationCorrected(
    violation: ConstraintViolation,
    candidates: Array<{hit: SearchHit, constraintAwareScore: number}>
  ): boolean {
    const candA = candidates.find(c => `${c.hit.file}:${c.hit.line}` === violation.candidateA);
    const candB = candidates.find(c => `${c.hit.file}:${c.hit.line}` === violation.candidateB);

    if (!candA || !candB) return false;

    const actualOrder = candA.constraintAwareScore > candB.constraintAwareScore ? 'A>B' : 'B>A';
    return actualOrder === violation.expectedOrder;
  }

  /**
   * Run validation of monotonic functions with runtime checks
   */
  validateMonotonicity(): { valid: boolean; errors: string[] } {
    if (!this.config.monotonicityCheckEnabled) {
      return { valid: true, errors: [] };
    }

    const errors: string[] = [];

    if (!this.exactMatchFunction.validateMonotonicity()) {
      errors.push('Exact match function violates monotonicity');
    }

    if (!this.sameFileFunction.validateMonotonicity()) {
      errors.push('Same file function violates monotonicity');
    }

    if (!this.structuralFunction.validateMonotonicity()) {
      errors.push('Structural function violates monotonicity');
    }

    return {
      valid: errors.length === 0,
      errors
    };
  }

  /**
   * Get comprehensive statistics for monitoring
   */
  getStats() {
    return {
      config: this.config,
      violations_logged: this.violationLog.length,
      recent_violations: this.violationLog.slice(-10),
      monotonicity_validation: this.validateMonotonicity()
    };
  }

  /**
   * Update configuration for A/B testing
   */
  updateConfig(newConfig: Partial<ConstraintAwareConfig>): void {
    this.config = { ...this.config, ...newConfig };
    
    // Reinitialize feature functions with new alpha if changed
    if (newConfig.alpha !== undefined) {
      this.exactMatchFunction = new MonotonicFeatureFunction('exact_token_match', this.config.alpha);
      this.sameFileFunction = new MonotonicFeatureFunction('same_file_symbol_def', this.config.alpha);
      this.structuralFunction = new MonotonicFeatureFunction('ast_pattern_match', this.config.alpha);
    }
    
    console.log(`🔧 ConstraintAwareReranker config updated: ${JSON.stringify(newConfig)}`);
  }
}