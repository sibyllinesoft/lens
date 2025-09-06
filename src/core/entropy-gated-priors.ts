/**
 * Entropy-Gated Priors System
 * 
 * Implements entropy-based gating for centrality and topic priors that adapt per query,
 * preventing over-steering for thin recall slices. Uses slice-specific isotonic 
 * calibration and positives-in-candidates monitoring.
 */

import type { SearchContext, SearchHit } from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';

export interface QueryEntropy {
  lexical_entropy: number;
  semantic_entropy: number;
  topic_entropy: number;
  combined_entropy: number;
}

export interface PriorWeights {
  centrality: number;
  topic_relevance: number;
  mmr_diversity: number;
  path_role_boost: number;
  gated_by_entropy: boolean;
  positives_in_candidates: number;
}

export interface EntropyGatingConfig {
  centrality_entropy_threshold: number;
  topic_entropy_threshold: number;
  min_positives_threshold: number;
  max_centrality_weight: number;
  max_topic_weight: number;
  isotonic_calibration_enabled: boolean;
  vendor_third_party_veto: boolean;
}

export interface SliceContext {
  intent: 'NL_overview' | 'code_search' | 'symbol_lookup' | 'definition_jump' | 'unknown';
  language: string;
  entropy_bin: 'low' | 'medium' | 'high';
  topic_category?: string;
}

/**
 * Isotonic regression calibrator for slice-specific weights
 */
class IsotonicCalibrator {
  private calibrationPoints: Array<{
    input: number;
    output: number;
    slice: string;
  }> = [];
  
  private calibratedFunctions: Map<string, (x: number) => number> = new Map();
  
  /**
   * Add calibration data point
   */
  addCalibrationPoint(slice: string, input: number, output: number): void {
    this.calibrationPoints.push({
      input,
      output,
      slice
    });
  }
  
  /**
   * Calibrate isotonic functions for each slice
   */
  calibrate(): void {
    // Group by slice
    const sliceGroups = new Map<string, Array<{input: number, output: number}>>();
    
    for (const point of this.calibrationPoints) {
      if (!sliceGroups.has(point.slice)) {
        sliceGroups.set(point.slice, []);
      }
      sliceGroups.get(point.slice)!.push({
        input: point.input,
        output: point.output
      });
    }
    
    // Build isotonic function for each slice
    for (const [slice, points] of sliceGroups) {
      if (points.length < 3) {
        // Not enough points, use linear interpolation
        this.calibratedFunctions.set(slice, (x: number) => Math.min(0.4, Math.max(-0.4, x)));
        continue;
      }
      
      // Sort by input
      points.sort((a, b) => a.input - b.input);
      
      // Simple isotonic regression (pool adjacent violators algorithm)
      const isotonic = this.poolAdjacentViolators(points);
      
      // Create interpolation function
      this.calibratedFunctions.set(slice, (x: number) => {
        return this.interpolateIsotonic(isotonic, x);
      });
    }
    
    console.log(`📊 Isotonic calibrator trained on ${this.calibrationPoints.length} points across ${sliceGroups.size} slices`);
  }
  
  /**
   * Get calibrated weight for slice
   */
  getCalibratedWeight(slice: string, rawWeight: number): number {
    const calibratedFn = this.calibratedFunctions.get(slice);
    if (calibratedFn) {
      return calibratedFn(rawWeight);
    }
    
    // Fallback to clamping
    return Math.min(0.4, Math.max(-0.4, rawWeight));
  }
  
  /**
   * Pool Adjacent Violators Algorithm for isotonic regression
   */
  private poolAdjacentViolators(points: Array<{input: number, output: number}>): Array<{input: number, output: number}> {
    const result = [...points];
    
    let i = 0;
    while (i < result.length - 1) {
      if (result[i].output > result[i + 1].output) {
        // Violation found, pool adjacent points
        const pooledOutput = (result[i].output + result[i + 1].output) / 2;
        result[i].output = pooledOutput;
        result[i + 1].output = pooledOutput;
        
        // Check backwards for more violations
        let j = i;
        while (j > 0 && result[j - 1].output > result[j].output) {
          const backPooledOutput = (result[j - 1].output + result[j].output) / 2;
          result[j - 1].output = backPooledOutput;
          result[j].output = backPooledOutput;
          j--;
        }
      }
      i++;
    }
    
    return result;
  }
  
  /**
   * Interpolate isotonic function
   */
  private interpolateIsotonic(points: Array<{input: number, output: number}>, x: number): number {
    if (points.length === 0) return 0;
    if (x <= points[0].input) return points[0].output;
    if (x >= points[points.length - 1].input) return points[points.length - 1].output;
    
    // Linear interpolation between adjacent points
    for (let i = 0; i < points.length - 1; i++) {
      if (x >= points[i].input && x <= points[i + 1].input) {
        const t = (x - points[i].input) / (points[i + 1].input - points[i].input);
        return points[i].output + t * (points[i + 1].output - points[i].output);
      }
    }
    
    return points[points.length - 1].output;
  }
}

/**
 * Centrality and topic prior manager
 */
class PriorManager {
  private centralityScores: Map<string, number> = new Map();
  private topicRelevanceScores: Map<string, number> = new Map();
  private pathRoleBoosts: Map<string, number> = new Map();
  
  /**
   * Load centrality scores for files
   */
  loadCentralityScores(scores: Map<string, number>): void {
    this.centralityScores = scores;
    console.log(`📊 Loaded centrality scores for ${scores.size} files`);
  }
  
  /**
   * Load topic relevance scores
   */
  loadTopicRelevanceScores(scores: Map<string, number>): void {
    this.topicRelevanceScores = scores;
    console.log(`🏷️ Loaded topic relevance scores for ${scores.size} files`);
  }
  
  /**
   * Calculate centrality prior for file
   */
  getCentralityPrior(filePath: string): number {
    return this.centralityScores.get(filePath) || 0.0;
  }
  
  /**
   * Calculate topic relevance prior for file
   */  
  getTopicPrior(filePath: string, queryTopic?: string): number {
    const key = queryTopic ? `${filePath}:${queryTopic}` : filePath;
    return this.topicRelevanceScores.get(key) || this.topicRelevanceScores.get(filePath) || 0.0;
  }
  
  /**
   * Calculate path role boost (with vendor/third-party veto)
   */
  getPathRoleBoost(filePath: string, enableVeto = true): number {
    if (enableVeto) {
      // Veto boost for vendor/third-party paths
      if (filePath.includes('/vendor/') || 
          filePath.includes('/third_party/') ||
          filePath.includes('/node_modules/') ||
          filePath.includes('/.git/')) {
        return -0.2; // Penalty for vendor/third-party
      }
    }
    
    // Boost for important paths
    if (filePath.includes('/src/') || filePath.includes('/lib/')) return 0.1;
    if (filePath.includes('/test/') || filePath.includes('/__tests__/')) return -0.05;
    if (filePath.endsWith('.md') || filePath.endsWith('.txt')) return -0.1;
    if (filePath.includes('/docs/')) return -0.05;
    
    return 0.0;
  }
}

/**
 * Entropy calculator for queries and results
 */
class EntropyCalculator {
  /**
   * Calculate query entropy across multiple dimensions
   */
  calculateQueryEntropy(query: string, ctx: SearchContext): QueryEntropy {
    // Lexical entropy (character-level)
    const chars = query.split('');
    const charCounts = this.getFrequencies(chars);
    const lexicalEntropy = this.calculateShannonEntropy(charCounts);
    
    // Token-level entropy
    const tokens = query.toLowerCase().split(/\s+/).filter(t => t.length > 0);
    const tokenCounts = this.getFrequencies(tokens);
    const tokenEntropy = this.calculateShannonEntropy(tokenCounts);
    
    // Semantic entropy (based on query structure)
    const semanticComplexity = this.estimateSemanticComplexity(query, ctx);
    
    // Topic entropy (how focused vs broad the query is)
    const topicEntropy = this.estimateTopicEntropy(query, ctx);
    
    // Combined entropy score
    const combinedEntropy = (lexicalEntropy + tokenEntropy + semanticComplexity + topicEntropy) / 4;
    
    return {
      lexical_entropy: lexicalEntropy,
      semantic_entropy: semanticComplexity,
      topic_entropy: topicEntropy,
      combined_entropy: combinedEntropy
    };
  }
  
  /**
   * Calculate result set entropy (diversity measure)
   */
  calculateResultEntropy(hits: SearchHit[]): number {
    if (hits.length === 0) return 0;
    
    // Calculate file path diversity
    const files = hits.map(h => h.file);
    const fileCounts = this.getFrequencies(files);
    const fileEntropy = this.calculateShannonEntropy(fileCounts);
    
    // Calculate language diversity
    const languages = hits.map(h => h.lang);
    const langCounts = this.getFrequencies(languages);
    const langEntropy = this.calculateShannonEntropy(langCounts);
    
    // Calculate match reason diversity
    const reasons = hits.flatMap(h => h.why || []);
    const reasonCounts = this.getFrequencies(reasons);
    const reasonEntropy = this.calculateShannonEntropy(reasonCounts);
    
    // Combined result entropy
    return (fileEntropy + langEntropy + reasonEntropy) / 3;
  }
  
  private getFrequencies(items: string[]): Map<string, number> {
    const counts = new Map<string, number>();
    for (const item of items) {
      counts.set(item, (counts.get(item) || 0) + 1);
    }
    return counts;
  }
  
  private calculateShannonEntropy(frequencies: Map<string, number>): number {
    const total = Array.from(frequencies.values()).reduce((sum, count) => sum + count, 0);
    if (total === 0) return 0;
    
    let entropy = 0;
    for (const count of frequencies.values()) {
      if (count > 0) {
        const p = count / total;
        entropy -= p * Math.log2(p);
      }
    }
    
    return entropy;
  }
  
  private estimateSemanticComplexity(query: string, ctx: SearchContext): number {
    let complexity = 0;
    
    // Special characters suggest complex queries
    const specialChars = (query.match(/[{}()\[\]<>.,;:]/g) || []).length;
    complexity += Math.min(2, specialChars / 5);
    
    // Multiple words suggest complex queries
    const words = query.split(/\s+/).filter(w => w.length > 0);
    complexity += Math.min(2, words.length / 10);
    
    // Search mode complexity
    if (ctx.mode === 'hybrid') complexity += 0.5;
    if (ctx.mode === 'struct') complexity += 0.3;
    if (ctx.fuzzy) complexity += 0.2;
    
    return Math.min(4, complexity);
  }
  
  private estimateTopicEntropy(query: string, ctx: SearchContext): number {
    // Estimate how focused vs broad the topic is
    const words = query.toLowerCase().split(/\s+/).filter(w => w.length > 0);
    
    // Very specific technical terms suggest low topic entropy (focused)
    const technicalTerms = words.filter(w => 
      w.includes('function') || w.includes('class') || w.includes('interface') ||
      w.includes('method') || w.includes('variable') || w.includes('component')
    ).length;
    
    if (technicalTerms > 0) return Math.max(1, 3 - technicalTerms);
    
    // Generic terms suggest high topic entropy (broad)
    const genericTerms = words.filter(w =>
      ['get', 'set', 'add', 'remove', 'create', 'delete', 'update', 'find', 'search'].includes(w)
    ).length;
    
    if (genericTerms > 0) return Math.min(4, 2 + genericTerms);
    
    // Default moderate entropy
    return 2.5;
  }
}

/**
 * Main entropy-gated priors engine
 */
export class EntropyGatedPriors {
  private config: EntropyGatingConfig;
  private isotonicCalibrator: IsotonicCalibrator;
  private priorManager: PriorManager;
  private entropyCalculator: EntropyCalculator;
  private enabled = true;
  
  // Metrics
  private totalQueries = 0;
  private priorAppliedQueries = 0;
  private entropyGatedQueries = 0;
  
  constructor(config?: Partial<EntropyGatingConfig>) {
    this.config = {
      centrality_entropy_threshold: 2.0,
      topic_entropy_threshold: 2.5,
      min_positives_threshold: 3,
      max_centrality_weight: 0.4,
      max_topic_weight: 0.3,
      isotonic_calibration_enabled: true,
      vendor_third_party_veto: true,
      ...config
    };
    
    this.isotonicCalibrator = new IsotonicCalibrator();
    this.priorManager = new PriorManager();
    this.entropyCalculator = new EntropyCalculator();
  }
  
  /**
   * Apply entropy-gated priors to search results
   * 
   * w_centrality = clip(s_iso[intent,lang,entropy_bin](percentile(C)), ±0.4)
   * enable only when positives_in_candidates ≥ m
   */
  async applyPriors(
    hits: SearchHit[],
    ctx: SearchContext
  ): Promise<SearchHit[]> {
    if (!this.enabled || hits.length === 0) {
      return hits;
    }
    
    const span = LensTracer.createChildSpan('entropy_gated_priors');
    this.totalQueries++;
    
    try {
      // Calculate query entropy
      const queryEntropy = this.entropyCalculator.calculateQueryEntropy(ctx.query, ctx);
      const resultEntropy = this.entropyCalculator.calculateResultEntropy(hits);
      
      // Determine slice context
      const sliceContext = this.determineSliceContext(ctx, queryEntropy);
      
      // Count positives in candidates (hits with good relevance)
      const positivesInCandidates = hits.filter(hit => hit.score > 0.5).length;
      
      console.log(`🌊 Entropy-gated priors: query_entropy=${queryEntropy.combined_entropy.toFixed(2)}, result_entropy=${resultEntropy.toFixed(2)}, positives=${positivesInCandidates}/${hits.length}`);
      
      // Check gating conditions
      if (!this.shouldApplyPriors(queryEntropy, positivesInCandidates)) {
        console.log(`⛔ Entropy gating: priors disabled for this query`);
        return hits;
      }
      
      this.priorAppliedQueries++;
      this.entropyGatedQueries++;
      
      // Apply priors to each hit
      const scoredHits = hits.map(hit => {
        const priorWeights = this.calculatePriorWeights(hit, ctx, sliceContext, queryEntropy);
        const adjustedScore = this.applyPriorWeights(hit, priorWeights);
        
        return {
          ...hit,
          score: adjustedScore,
          prior_weights: priorWeights,
          why: [
            ...(hit.why || []),
            ...(priorWeights.gated_by_entropy ? ['entropy_gated_priors'] : []),
            ...(priorWeights.centrality > 0.1 ? ['centrality_boost'] : []),
            ...(priorWeights.topic_relevance > 0.1 ? ['topic_boost'] : [])
          ]
        };
      });
      
      // Re-sort by adjusted scores
      scoredHits.sort((a, b) => b.score - a.score);
      
      span.setAttributes({
        success: true,
        query_entropy: queryEntropy.combined_entropy,
        result_entropy: resultEntropy,
        positives_in_candidates: positivesInCandidates,
        priors_applied: true
      });
      
      return scoredHits;
      
    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      console.error('Entropy-gated priors error:', error);
      return hits;
    } finally {
      span.end();
    }
  }
  
  /**
   * Check if priors should be applied based on entropy and positives
   */
  private shouldApplyPriors(
    queryEntropy: QueryEntropy,
    positivesInCandidates: number
  ): boolean {
    // Don't apply priors if query entropy is too high (too scattered)
    if (queryEntropy.combined_entropy > this.config.centrality_entropy_threshold) {
      return false;
    }
    
    // Don't apply priors if not enough positive candidates
    if (positivesInCandidates < this.config.min_positives_threshold) {
      return false;
    }
    
    // Don't apply topic priors if topic entropy is too high (too broad)
    if (queryEntropy.topic_entropy > this.config.topic_entropy_threshold) {
      return false;
    }
    
    return true;
  }
  
  /**
   * Determine slice context for isotonic calibration
   */
  private determineSliceContext(
    ctx: SearchContext,
    queryEntropy: QueryEntropy
  ): SliceContext {
    // Classify intent based on query characteristics
    let intent: SliceContext['intent'] = 'unknown';
    
    const query = ctx.query.toLowerCase();
    if (query.includes('overview') || query.includes('summary') || query.length > 50) {
      intent = 'NL_overview';
    } else if (ctx.mode === 'struct' || query.includes('function') || query.includes('class')) {
      intent = 'symbol_lookup';
    } else if (query.includes('definition') || query.includes('declaration')) {
      intent = 'definition_jump';
    } else if (query.split(/\s+/).length > 3) {
      intent = 'NL_overview';
    } else {
      intent = 'code_search';
    }
    
    // Determine entropy bin
    let entropyBin: SliceContext['entropy_bin'] = 'medium';
    if (queryEntropy.combined_entropy < 1.5) {
      entropyBin = 'low';
    } else if (queryEntropy.combined_entropy > 3.0) {
      entropyBin = 'high';
    }
    
    return {
      intent,
      language: this.detectLanguage(ctx) || 'unknown',
      entropy_bin: entropyBin
    };
  }
  
  /**
   * Calculate prior weights for a search hit
   */
  private calculatePriorWeights(
    hit: SearchHit,
    ctx: SearchContext,
    sliceContext: SliceContext,
    queryEntropy: QueryEntropy
  ): PriorWeights {
    const filePath = hit.file;
    
    // Get raw prior scores
    const rawCentrality = this.priorManager.getCentralityPrior(filePath);
    const rawTopicRelevance = this.priorManager.getTopicPrior(filePath, sliceContext.topic_category);
    const pathRoleBoost = this.priorManager.getPathRoleBoost(filePath, this.config.vendor_third_party_veto);
    
    // Apply isotonic calibration if enabled
    let centralityWeight = rawCentrality;
    let topicWeight = rawTopicRelevance;
    
    if (this.config.isotonic_calibration_enabled) {
      const sliceKey = `${sliceContext.intent}_${sliceContext.language}_${sliceContext.entropy_bin}`;
      centralityWeight = this.isotonicCalibrator.getCalibratedWeight(`centrality_${sliceKey}`, rawCentrality);
      topicWeight = this.isotonicCalibrator.getCalibratedWeight(`topic_${sliceKey}`, rawTopicRelevance);
    }
    
    // Clip to maximum weights
    centralityWeight = Math.min(this.config.max_centrality_weight, Math.max(-this.config.max_centrality_weight, centralityWeight));
    topicWeight = Math.min(this.config.max_topic_weight, Math.max(-this.config.max_topic_weight, topicWeight));
    
    // MMR diversity weight for broad queries
    let mmrWeight = 0;
    if (sliceContext.intent === 'NL_overview' && queryEntropy.topic_entropy > 2.5) {
      mmrWeight = 0.15; // Enable MMR for diverse results
    }
    
    return {
      centrality: centralityWeight,
      topic_relevance: topicWeight,
      mmr_diversity: mmrWeight,
      path_role_boost: pathRoleBoost,
      gated_by_entropy: queryEntropy.combined_entropy <= this.config.centrality_entropy_threshold,
      positives_in_candidates: 0 // Will be set by caller
    };
  }
  
  /**
   * Apply prior weights to adjust hit score
   */
  private applyPriorWeights(hit: SearchHit, weights: PriorWeights): number {
    let adjustedScore = hit.score;
    
    // Apply additive boosts (risk-budgeted approach)
    adjustedScore += weights.centrality;
    adjustedScore += weights.topic_relevance;  
    adjustedScore += weights.path_role_boost;
    
    // MMR diversity adjustment (reduces score for similar results)
    // This would be implemented with actual MMR logic in production
    if (weights.mmr_diversity > 0) {
      adjustedScore *= (1 - weights.mmr_diversity * 0.1); // Small diversity penalty
    }
    
    // Keep score in reasonable bounds
    return Math.max(0.01, Math.min(1.0, adjustedScore));
  }
  
  /**
   * Detect primary language from context
   */
  private detectLanguage(ctx: SearchContext): string | null {
    // This would integrate with language detection logic
    // For now, return a simple heuristic
    const query = ctx.query.toLowerCase();
    
    if (query.includes('function') || query.includes('const') || query.includes('let')) {
      return 'typescript';
    }
    if (query.includes('def ') || query.includes('class ') || query.includes('import ')) {
      return 'python';
    }
    if (query.includes('fn ') || query.includes('struct ') || query.includes('impl ')) {
      return 'rust';
    }
    
    return null;
  }
  
  /**
   * Load centrality and topic data
   */
  async loadPriorData(
    centralityScores: Map<string, number>,
    topicScores: Map<string, number>
  ): Promise<void> {
    this.priorManager.loadCentralityScores(centralityScores);
    this.priorManager.loadTopicRelevanceScores(topicScores);
  }
  
  /**
   * Add calibration data for isotonic regression
   */
  addCalibrationData(
    sliceContext: SliceContext,
    rawScore: number,
    actualRelevance: number
  ): void {
    const sliceKey = `${sliceContext.intent}_${sliceContext.language}_${sliceContext.entropy_bin}`;
    this.isotonicCalibrator.addCalibrationPoint(`centrality_${sliceKey}`, rawScore, actualRelevance);
  }
  
  /**
   * Calibrate isotonic functions
   */
  async calibrate(): Promise<void> {
    this.isotonicCalibrator.calibrate();
  }
  
  /**
   * Get statistics
   */
  getStats(): {
    total_queries: number;
    prior_applied_queries: number;
    entropy_gated_queries: number;
    prior_application_rate: number;
    entropy_gating_rate: number;
    enabled: boolean;
    config: EntropyGatingConfig;
  } {
    return {
      total_queries: this.totalQueries,
      prior_applied_queries: this.priorAppliedQueries,
      entropy_gated_queries: this.entropyGatedQueries,
      prior_application_rate: this.totalQueries > 0 ? (this.priorAppliedQueries / this.totalQueries) * 100 : 0,
      entropy_gating_rate: this.totalQueries > 0 ? (this.entropyGatedQueries / this.totalQueries) * 100 : 0,
      enabled: this.enabled,
      config: this.config
    };
  }
  
  /**
   * Enable/disable entropy-gated priors
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    console.log(`🌊 Entropy-gated priors ${enabled ? 'ENABLED' : 'DISABLED'}`);
  }
  
  /**
   * Update configuration
   */
  updateConfig(config: Partial<EntropyGatingConfig>): void {
    this.config = { ...this.config, ...config };
    console.log('🔧 Entropy-gated priors config updated:', config);
  }
}

// Global instance
export const globalEntropyGatedPriors = new EntropyGatedPriors();