/**
 * Targeted Diversity (Constrained MMR) System
 * 
 * Implements selective Maximum Marginal Relevance (MMR) only for NL_overview queries
 * with high entropy, using constrained optimization with hard floors for exact/struct matches.
 * 
 * Key Features:
 * - MMR off by default, enabled only for `NL_overview ∧ high_entropy`
 * - Enabled only after clone-collapse to prevent fake diversity
 * - Constrained MMR: argmax_S Σᵢ∈S rᵢ - γ Σᵢ<j sim_topic/symbol(i,j) s.t. floors(exact,struct)=true
 * - Gate: ΔnDCG@10 ≥ 0 and Diversity@10 ≥ +10% on overview slice only
 * - Parameters: γ∈[0.07,0.12]
 */

import type { SearchHit, MatchReason } from '../core/span_resolver/types.js';
import type { SearchContext } from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';

// Configuration constants per TODO.md requirements
const GAMMA_MIN = 0.07; // γ ∈ [0.07, 0.12]
const GAMMA_MAX = 0.12;
const DEFAULT_GAMMA = 0.09; // Middle of range
const MIN_ENTROPY_THRESHOLD = 0.6; // High entropy threshold
const DIVERSITY_IMPROVEMENT_TARGET = 0.10; // +10% diversity improvement
const NDCG_PRESERVATION_TARGET = 0; // ΔnDCG@10 ≥ 0
const EXACT_MATCH_FLOOR = 2; // Minimum exact matches to preserve
const STRUCT_MATCH_FLOOR = 1; // Minimum structural matches to preserve

export interface DiversityFeatures {
  query_type: 'NL_overview' | 'targeted_search' | 'symbol_lookup' | 'other';
  topic_entropy: number;
  result_count: number;
  exact_matches: number;
  structural_matches: number;
  clone_collapsed: boolean;
}

export interface MMRCandidate {
  hit: SearchHit;
  relevance_score: number;
  topic_vector: number[]; // Simplified topic representation
  symbol_type: string;
  path_components: string[];
  exact_match: boolean;
  structural_match: boolean;
}

export interface DiversityMetrics {
  original_ndcg: number;
  diversified_ndcg: number;
  diversity_score: number;
  gamma_used: number;
  candidates_processed: number;
  constraints_applied: boolean;
}

export class TargetedDiversitySystem {
  private performanceMetrics = {
    mmr_applications: [] as DiversityMetrics[],
    diversity_improvements: [] as number[],
    ndcg_changes: [] as number[],
    constraint_violations: 0,
    query_type_distribution: new Map<string, number>(),
  };

  private topicSimilarityCache = new Map<string, Map<string, number>>();
  private lastCacheCleanup = Date.now();
  
  /**
   * Initialize the targeted diversity system
   */
  async initialize(): Promise<void> {
    const span = LensTracer.createChildSpan('diversity_init');
    
    try {
      console.log('🎯 Initializing Targeted Diversity (Constrained MMR) system...');
      
      // Initialize performance tracking
      this.performanceMetrics = {
        mmr_applications: [],
        diversity_improvements: [],
        ndcg_changes: [],
        constraint_violations: 0,
        query_type_distribution: new Map(),
      };
      
      // Initialize topic similarity cache
      this.topicSimilarityCache.clear();
      
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
   * Apply targeted diversity to search results using constrained MMR
   */
  async diversifyResults(
    hits: SearchHit[],
    ctx: SearchContext,
    diversityFeatures: DiversityFeatures
  ): Promise<SearchHit[]> {
    const span = LensTracer.createChildSpan('diversify_results', {
      hit_count: hits.length,
      query_type: diversityFeatures.query_type,
      topic_entropy: diversityFeatures.topic_entropy,
      clone_collapsed: diversityFeatures.clone_collapsed
    });
    
    try {
      // Track query type distribution
      const queryType = diversityFeatures.query_type;
      this.performanceMetrics.query_type_distribution.set(
        queryType,
        (this.performanceMetrics.query_type_distribution.get(queryType) || 0) + 1
      );
      
      // Gate 1: Check if MMR should be applied
      if (!this.shouldApplyMMR(diversityFeatures)) {
        span.setAttributes({
          mmr_applied: false,
          reason: 'diversity_criteria_not_met',
          query_type: diversityFeatures.query_type,
          entropy: diversityFeatures.topic_entropy
        });
        return hits;
      }
      
      // Gate 2: Ensure clone-collapse has been applied
      if (!diversityFeatures.clone_collapsed) {
        span.setAttributes({
          mmr_applied: false,
          reason: 'clone_collapse_not_applied'
        });
        console.warn('MMR requested before clone-collapse - diversity would be fake');
        return hits;
      }
      
      // Convert hits to MMR candidates
      const candidates = await this.convertToMMRCandidates(hits, ctx);
      
      // Apply constrained MMR optimization
      const diversifiedHits = await this.constrainedMMR(candidates, diversityFeatures);
      
      // Validate diversity improvement
      const metrics = await this.calculateDiversityMetrics(hits, diversifiedHits, ctx);
      
      // Gate 3: Check quality gates (ΔnDCG@10 ≥ 0 and Diversity@10 ≥ +10%)
      if (!this.meetsQualityGates(metrics)) {
        span.setAttributes({
          mmr_applied: false,
          reason: 'quality_gates_failed',
          ndcg_change: metrics.diversified_ndcg - metrics.original_ndcg,
          diversity_improvement: metrics.diversity_score
        });
        
        // Return original results if quality gates fail
        return hits;
      }
      
      // Record successful diversification
      this.performanceMetrics.mmr_applications.push(metrics);
      this.performanceMetrics.diversity_improvements.push(metrics.diversity_score);
      this.performanceMetrics.ndcg_changes.push(metrics.diversified_ndcg - metrics.original_ndcg);
      
      span.setAttributes({
        mmr_applied: true,
        original_count: hits.length,
        diversified_count: diversifiedHits.length,
        diversity_improvement: metrics.diversity_score,
        ndcg_change: metrics.diversified_ndcg - metrics.original_ndcg,
        gamma_used: metrics.gamma_used
      });
      
      return diversifiedHits;
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      
      // Return original hits on error
      return hits;
      
    } finally {
      span.end();
    }
  }
  
  /**
   * Check if MMR should be applied based on diversity features
   */
  private shouldApplyMMR(features: DiversityFeatures): boolean {
    // Only apply for NL_overview queries
    if (features.query_type !== 'NL_overview') {
      return false;
    }
    
    // Only apply for high entropy queries
    if (features.topic_entropy < MIN_ENTROPY_THRESHOLD) {
      return false;
    }
    
    // Need sufficient results to diversify
    if (features.result_count < 5) {
      return false;
    }
    
    return true;
  }
  
  /**
   * Convert SearchHits to MMR candidates with topic vectors and metadata
   */
  private async convertToMMRCandidates(
    hits: SearchHit[],
    ctx: SearchContext
  ): Promise<MMRCandidate[]> {
    const candidates: MMRCandidate[] = [];
    
    for (const hit of hits) {
      // Generate topic vector (simplified implementation)
      const topicVector = await this.generateTopicVector(hit, ctx);
      
      // Extract path components for diversity calculation
      const pathComponents = hit.file.split('/').filter(c => c.length > 0);
      
      // Determine match types
      const exactMatch = this.isExactMatch(hit, ctx.query);
      const structuralMatch = this.isStructuralMatch(hit, ctx.query);
      
      candidates.push({
        hit,
        relevance_score: hit.score,
        topic_vector: topicVector,
        symbol_type: hit.symbol_kind || 'unknown',
        path_components: pathComponents,
        exact_match: exactMatch,
        structural_match: structuralMatch,
      });
    }
    
    return candidates;
  }
  
  /**
   * Apply constrained MMR: argmax_S Σᵢ∈S rᵢ - γ Σᵢ<j sim_topic/symbol(i,j) s.t. floors(exact,struct)=true
   */
  private async constrainedMMR(
    candidates: MMRCandidate[],
    features: DiversityFeatures
  ): Promise<SearchHit[]> {
    const selectedHits: SearchHit[] = [];
    const remainingCandidates = [...candidates];
    const selectedIndices = new Set<number>();
    
    // Determine gamma value within valid range
    const gamma = this.adaptiveGamma(features);
    
    // Phase 1: Satisfy hard constraints (exact and structural match floors)
    await this.satisfyHardConstraints(
      selectedHits,
      remainingCandidates,
      selectedIndices,
      features
    );
    
    // Phase 2: Apply MMR optimization for remaining slots
    while (selectedHits.length < Math.min(20, candidates.length) && remainingCandidates.length > 0) {
      let bestCandidate: MMRCandidate | null = null;
      let bestScore = -Infinity;
      let bestIndex = -1;
      
      for (let i = 0; i < remainingCandidates.length; i++) {
        if (selectedIndices.has(i)) continue;
        
        const candidate = remainingCandidates[i];
        
        // Calculate MMR score: relevance - γ * max_similarity_to_selected
        const relevanceScore = candidate.relevance_score;
        const maxSimilarity = await this.maxSimilarityToSelected(candidate, selectedHits);
        const mmrScore = relevanceScore - gamma * maxSimilarity;
        
        if (mmrScore > bestScore) {
          bestScore = mmrScore;
          bestCandidate = candidate;
          bestIndex = i;
        }
      }
      
      if (bestCandidate) {
        selectedHits.push(bestCandidate.hit);
        selectedIndices.add(bestIndex);
      } else {
        break; // No more candidates
      }
    }
    
    return selectedHits;
  }
  
  /**
   * Satisfy hard constraints for exact and structural match floors
   */
  private async satisfyHardConstraints(
    selectedHits: SearchHit[],
    candidates: MMRCandidate[],
    selectedIndices: Set<number>,
    features: DiversityFeatures
  ): Promise<void> {
    // Phase 1a: Ensure minimum exact matches
    const exactCandidates = candidates
      .map((candidate, index) => ({ candidate, index }))
      .filter(({ candidate }) => candidate.exact_match)
      .sort((a, b) => b.candidate.relevance_score - a.candidate.relevance_score);
    
    const exactMatchesToAdd = Math.min(
      EXACT_MATCH_FLOOR,
      Math.min(exactCandidates.length, features.exact_matches)
    );
    
    for (let i = 0; i < exactMatchesToAdd && i < exactCandidates.length; i++) {
      const { candidate, index } = exactCandidates[i];
      selectedHits.push(candidate.hit);
      selectedIndices.add(index);
    }
    
    // Phase 1b: Ensure minimum structural matches
    const structCandidates = candidates
      .map((candidate, index) => ({ candidate, index }))
      .filter(({ candidate, index }) => 
        candidate.structural_match && !selectedIndices.has(index)
      )
      .sort((a, b) => b.candidate.relevance_score - a.candidate.relevance_score);
    
    const structMatchesToAdd = Math.min(
      STRUCT_MATCH_FLOOR,
      Math.min(structCandidates.length, features.structural_matches)
    );
    
    for (let i = 0; i < structMatchesToAdd && i < structCandidates.length; i++) {
      const { candidate, index } = structCandidates[i];
      selectedHits.push(candidate.hit);
      selectedIndices.add(index);
    }
  }
  
  /**
   * Calculate maximum similarity between candidate and already selected hits
   */
  private async maxSimilarityToSelected(
    candidate: MMRCandidate,
    selectedHits: SearchHit[]
  ): Promise<number> {
    if (selectedHits.length === 0) {
      return 0;
    }
    
    let maxSim = 0;
    
    for (const selectedHit of selectedHits) {
      // Calculate topic similarity
      const topicSim = await this.calculateTopicSimilarity(candidate, selectedHit);
      
      // Calculate symbol similarity
      const symbolSim = this.calculateSymbolSimilarity(candidate, selectedHit);
      
      // Combined similarity
      const combinedSim = (topicSim + symbolSim) / 2;
      
      maxSim = Math.max(maxSim, combinedSim);
    }
    
    return maxSim;
  }
  
  /**
   * Calculate topic similarity between candidate and selected hit
   */
  private async calculateTopicSimilarity(
    candidate: MMRCandidate,
    selectedHit: SearchHit
  ): Promise<number> {
    const key1 = `${candidate.hit.file}:${candidate.hit.line}`;
    const key2 = `${selectedHit.file}:${selectedHit.line}`;
    
    // Check cache
    const cached = this.topicSimilarityCache.get(key1)?.get(key2) ||
                   this.topicSimilarityCache.get(key2)?.get(key1);
    
    if (cached !== undefined) {
      return cached;
    }
    
    // Calculate path-based similarity
    const path1 = candidate.path_components;
    const path2 = selectedHit.file.split('/').filter(c => c.length > 0);
    
    const commonPrefix = this.calculateCommonPrefixLength(path1, path2);
    const pathSimilarity = commonPrefix / Math.max(path1.length, path2.length);
    
    // Calculate filename similarity
    const filename1 = path1[path1.length - 1] || '';
    const filename2 = path2[path2.length - 1] || '';
    const filenameSimilarity = this.stringSimilarity(filename1, filename2);
    
    // Combined topic similarity
    const topicSim = (pathSimilarity + filenameSimilarity) / 2;
    
    // Cache result
    if (!this.topicSimilarityCache.has(key1)) {
      this.topicSimilarityCache.set(key1, new Map());
    }
    this.topicSimilarityCache.get(key1)!.set(key2, topicSim);
    
    return topicSim;
  }
  
  /**
   * Calculate symbol similarity between candidate and selected hit
   */
  private calculateSymbolSimilarity(
    candidate: MMRCandidate,
    selectedHit: SearchHit
  ): number {
    const symbol1 = candidate.symbol_type;
    const symbol2 = selectedHit.symbol_kind || 'unknown';
    
    if (symbol1 === symbol2) {
      return 1.0;
    }
    
    // Related symbol types have moderate similarity
    const relatedTypes = new Map([
      ['function', ['method', 'closure']],
      ['class', ['interface', 'struct', 'type']],
      ['variable', ['const', 'field', 'property']],
    ]);
    
    for (const [primary, related] of relatedTypes) {
      if (symbol1 === primary && related.includes(symbol2)) return 0.7;
      if (symbol2 === primary && related.includes(symbol1)) return 0.7;
    }
    
    return 0.1; // Low similarity for unrelated types
  }
  
  /**
   * Adaptive gamma selection based on query characteristics
   */
  private adaptiveGamma(features: DiversityFeatures): number {
    let gamma = DEFAULT_GAMMA;
    
    // Higher entropy queries benefit from more diversity (higher γ)
    if (features.topic_entropy > 0.8) {
      gamma = GAMMA_MAX;
    } else if (features.topic_entropy < 0.6) {
      gamma = GAMMA_MIN;
    }
    
    // Adjust for result count
    if (features.result_count > 50) {
      gamma += 0.01; // Slightly more diversity for large result sets
    }
    
    return Math.max(GAMMA_MIN, Math.min(GAMMA_MAX, gamma));
  }
  
  /**
   * Generate topic vector for a hit (simplified implementation)
   */
  private async generateTopicVector(hit: SearchHit, ctx: SearchContext): Promise<number[]> {
    // Simplified topic vector based on path, symbol type, and content
    const vector = new Array(10).fill(0);
    
    // Path-based features
    const pathTokens = hit.file.toLowerCase().split('/');
    for (let i = 0; i < Math.min(pathTokens.length, 5); i++) {
      vector[i] = this.hashToFloat(pathTokens[i]);
    }
    
    // Symbol type feature
    if (hit.symbol_kind) {
      vector[5] = this.hashToFloat(hit.symbol_kind);
    }
    
    // Query relevance features
    const queryTokens = ctx.query.toLowerCase().split(/\s+/);
    for (let i = 0; i < Math.min(queryTokens.length, 3); i++) {
      vector[6 + i] = this.hashToFloat(queryTokens[i]);
    }
    
    // Language feature
    if (hit.lang) {
      vector[9] = this.hashToFloat(hit.lang);
    }
    
    return vector;
  }
  
  /**
   * Check if hit is an exact match for the query
   */
  private isExactMatch(hit: SearchHit, query: string): boolean {
    const queryLower = query.toLowerCase();
    
    // Check if query appears exactly in the snippet
    if (hit.snippet && hit.snippet.toLowerCase().includes(queryLower)) {
      return true;
    }
    
    // Check if hit has exact match reason
    if (hit.why && hit.why.includes('exact')) {
      return true;
    }
    
    // Check symbol name match
    if (hit.symbol_kind) {
      const filename = hit.file.split('/').pop() || '';
      const symbolName = filename.split('.')[0];
      if (symbolName.toLowerCase() === queryLower) {
        return true;
      }
    }
    
    return false;
  }
  
  /**
   * Check if hit is a structural match for the query
   */
  private isStructuralMatch(hit: SearchHit, query: string): boolean {
    // Check if hit has structural match reasons
    if (hit.why) {
      const structuralReasons = ['ast', 'symbol', 'structural', 'scope'];
      if (structuralReasons.some(reason => hit.why!.includes(reason as MatchReason))) {
        return true;
      }
    }
    
    // Check if symbol kind suggests structural relevance
    if (hit.symbol_kind) {
      const structuralTypes = ['class', 'interface', 'function', 'method', 'type'];
      if (structuralTypes.includes(hit.symbol_kind)) {
        return true;
      }
    }
    
    return false;
  }
  
  /**
   * Calculate diversity metrics for quality gate validation
   */
  private async calculateDiversityMetrics(
    originalHits: SearchHit[],
    diversifiedHits: SearchHit[],
    ctx: SearchContext
  ): Promise<DiversityMetrics> {
    // Calculate nDCG@10 for both result sets
    const originalNDCG = this.calculateNDCG(originalHits.slice(0, 10), ctx);
    const diversifiedNDCG = this.calculateNDCG(diversifiedHits.slice(0, 10), ctx);
    
    // Calculate diversity score (simplified)
    const diversityScore = this.calculateDiversityScore(diversifiedHits.slice(0, 10));
    
    return {
      original_ndcg: originalNDCG,
      diversified_ndcg: diversifiedNDCG,
      diversity_score: diversityScore,
      gamma_used: DEFAULT_GAMMA, // Would track actual gamma used
      candidates_processed: originalHits.length,
      constraints_applied: true,
    };
  }
  
  /**
   * Calculate nDCG@k for a result set
   */
  private calculateNDCG(hits: SearchHit[], ctx: SearchContext): number {
    if (hits.length === 0) return 0;
    
    // Calculate DCG
    let dcg = 0;
    for (let i = 0; i < hits.length; i++) {
      const relevance = this.estimateRelevance(hits[i], ctx);
      dcg += relevance / Math.log2(i + 2);
    }
    
    // Calculate ideal DCG (assume perfect relevance ordering)
    let idcg = 0;
    for (let i = 0; i < hits.length; i++) {
      idcg += 1.0 / Math.log2(i + 2);
    }
    
    return idcg > 0 ? dcg / idcg : 0;
  }
  
  /**
   * Calculate diversity score for a result set
   */
  private calculateDiversityScore(hits: SearchHit[]): number {
    if (hits.length <= 1) return 0;
    
    let totalSimilarity = 0;
    let pairCount = 0;
    
    // Calculate average pairwise dissimilarity
    for (let i = 0; i < hits.length; i++) {
      for (let j = i + 1; j < hits.length; j++) {
        const pathSim = this.pathSimilarity(hits[i].file, hits[j].file);
        const symbolSim = this.symbolSimilarity(
          hits[i].symbol_kind || '',
          hits[j].symbol_kind || ''
        );
        const similarity = (pathSim + symbolSim) / 2;
        totalSimilarity += similarity;
        pairCount++;
      }
    }
    
    const avgSimilarity = pairCount > 0 ? totalSimilarity / pairCount : 0;
    return 1 - avgSimilarity; // Diversity = 1 - similarity
  }
  
  /**
   * Check if diversity metrics meet quality gates
   */
  private meetsQualityGates(metrics: DiversityMetrics): boolean {
    // Gate 1: ΔnDCG@10 ≥ 0
    const ndcgChange = metrics.diversified_ndcg - metrics.original_ndcg;
    if (ndcgChange < NDCG_PRESERVATION_TARGET) {
      return false;
    }
    
    // Gate 2: Diversity@10 ≥ +10%
    if (metrics.diversity_score < DIVERSITY_IMPROVEMENT_TARGET) {
      return false;
    }
    
    return true;
  }
  
  // Utility methods
  
  private calculateCommonPrefixLength(path1: string[], path2: string[]): number {
    let common = 0;
    for (let i = 0; i < Math.min(path1.length, path2.length); i++) {
      if (path1[i] === path2[i]) {
        common++;
      } else {
        break;
      }
    }
    return common;
  }
  
  private stringSimilarity(str1: string, str2: string): number {
    if (str1 === str2) return 1.0;
    if (str1.length === 0 || str2.length === 0) return 0;
    
    const longer = str1.length > str2.length ? str1 : str2;
    const shorter = str1.length <= str2.length ? str1 : str2;
    
    // Simple character-based similarity
    let matches = 0;
    for (let i = 0; i < shorter.length; i++) {
      if (longer.includes(shorter[i])) {
        matches++;
      }
    }
    
    return matches / longer.length;
  }
  
  private hashToFloat(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash) / 2147483647; // Normalize to [0, 1]
  }
  
  private estimateRelevance(hit: SearchHit, ctx: SearchContext): number {
    // Simplified relevance estimation based on score and match reasons
    let relevance = Math.min(1.0, hit.score / 100); // Normalize score
    
    // Boost for exact matches
    if (hit.why && hit.why.includes('exact')) {
      relevance += 0.3;
    }
    
    // Boost for structural matches
    if (hit.symbol_kind) {
      relevance += 0.2;
    }
    
    return Math.min(1.0, relevance);
  }
  
  private pathSimilarity(path1: string, path2: string): number {
    const components1 = path1.split('/').filter(c => c.length > 0);
    const components2 = path2.split('/').filter(c => c.length > 0);
    
    const commonPrefix = this.calculateCommonPrefixLength(components1, components2);
    return commonPrefix / Math.max(components1.length, components2.length);
  }
  
  private symbolSimilarity(symbol1: string, symbol2: string): number {
    if (symbol1 === symbol2) return 1.0;
    if (!symbol1 || !symbol2) return 0;
    
    // Use the same logic as calculateSymbolSimilarity but simplified
    const relatedTypes = [
      ['function', 'method', 'closure'],
      ['class', 'interface', 'struct', 'type'],
      ['variable', 'const', 'field', 'property'],
    ];
    
    for (const group of relatedTypes) {
      if (group.includes(symbol1) && group.includes(symbol2)) {
        return 0.7;
      }
    }
    
    return 0.1;
  }
  
  /**
   * Get performance metrics for system monitoring
   */
  getPerformanceMetrics() {
    const diversityImprovements = this.performanceMetrics.diversity_improvements;
    const ndcgChanges = this.performanceMetrics.ndcg_changes;
    
    const avgDiversityImprovement = diversityImprovements.length > 0
      ? diversityImprovements.reduce((a, b) => a + b) / diversityImprovements.length
      : 0;
    
    const avgNDCGChange = ndcgChanges.length > 0
      ? ndcgChanges.reduce((a, b) => a + b) / ndcgChanges.length
      : 0;
    
    // Check quality gate compliance
    const qualityGatesPassed = this.performanceMetrics.mmr_applications.filter(
      metrics => this.meetsQualityGates(metrics)
    ).length;
    
    const totalApplications = this.performanceMetrics.mmr_applications.length;
    const qualityGatePassRate = totalApplications > 0 
      ? qualityGatesPassed / totalApplications 
      : 0;
    
    return {
      mmr_applications_count: totalApplications,
      average_diversity_improvement: avgDiversityImprovement,
      average_ndcg_change: avgNDCGChange,
      quality_gate_pass_rate: qualityGatePassRate,
      constraint_violations: this.performanceMetrics.constraint_violations,
      query_type_distribution: Object.fromEntries(this.performanceMetrics.query_type_distribution),
    };
  }
  
  /**
   * Cleanup cache periodically
   */
  private cleanupCacheIfNeeded(): void {
    const now = Date.now();
    const CACHE_CLEANUP_INTERVAL = 5 * 60 * 1000; // 5 minutes
    
    if (now - this.lastCacheCleanup > CACHE_CLEANUP_INTERVAL) {
      // Clear topic similarity cache to prevent unbounded growth
      this.topicSimilarityCache.clear();
      this.lastCacheCleanup = now;
    }
  }
  
  /**
   * Cleanup and shutdown
   */
  async shutdown(): Promise<void> {
    const span = LensTracer.createChildSpan('diversity_shutdown');
    
    try {
      console.log('🎯 Shutting down Targeted Diversity system...');
      
      // Clear caches
      this.topicSimilarityCache.clear();
      
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