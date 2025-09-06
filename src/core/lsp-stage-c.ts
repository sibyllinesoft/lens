/**
 * Stage-C LSP Features with Bounded Contribution
 * Adds bounded features: lsp_def_hit∈{0,1}, lsp_ref_count, type_match, alias_resolved
 * Caps contribution (≤0.4 log-odds) so LSP can't swamp other signals
 */

import type { 
  LSPHint,
  LSPFeatures, 
  Candidate, 
  SearchContext 
} from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';

interface LSPStageCResult {
  enhanced_candidates: Candidate[];
  feature_stats: {
    def_hits: number;
    avg_ref_count: number;
    type_matches: number;
    alias_resolved: number;
  };
  score_distribution: {
    lsp_contribution_avg: number;
    lsp_contribution_max: number;
    bounded_adjustments: number;
  };
  processing_time_ms: number;
}

export class LSPStageCEnhancer {
  private static readonly MAX_LSP_LOG_ODDS = 0.4; // Bounded contribution limit
  private static readonly REF_COUNT_SCALING = 0.01; // Scale factor for reference counts
  private static readonly TYPE_MATCH_BONUS = 0.2;
  private static readonly ALIAS_BONUS = 0.15;
  private static readonly DEF_HIT_BONUS = 0.25;

  private hintsLookup = new Map<string, LSPHint>(); // symbol_id -> hint

  constructor() {}

  /**
   * Load LSP hints for Stage-C processing
   */
  loadHints(hints: LSPHint[]): void {
    const span = LensTracer.createChildSpan('lsp_stage_c_load_hints', {
      'hints.count': hints.length,
    });

    try {
      this.hintsLookup.clear();
      
      for (const hint of hints) {
        this.hintsLookup.set(hint.symbol_id, hint);
        
        // Also index by file:line:col for location-based lookups
        const locationKey = `${hint.file_path}:${hint.line}:${hint.col}`;
        this.hintsLookup.set(locationKey, hint);
      }

      span.setAttributes({
        success: true,
        lookup_size: this.hintsLookup.size,
      });

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Enhance Stage-C candidates with bounded LSP features
   */
  enhanceStageC(
    candidates: Candidate[],
    query: string,
    context: SearchContext
  ): LSPStageCResult {
    const span = LensTracer.createChildSpan('lsp_stage_c_enhance', {
      'candidates.count': candidates.length,
      'search.query': query,
    });

    const startTime = Date.now();
    
    try {
      const enhancedCandidates = [...candidates];
      const stats = {
        def_hits: 0,
        avg_ref_count: 0,
        type_matches: 0,
        alias_resolved: 0,
      };
      const scoreDistribution = {
        lsp_contribution_avg: 0,
        lsp_contribution_max: 0,
        bounded_adjustments: 0,
      };

      let totalRefCount = 0;
      let totalLspContribution = 0;

      // Enhance each candidate with LSP features
      for (const candidate of enhancedCandidates) {
        const lspFeatures = this.extractLSPFeatures(candidate, query);
        const lspContribution = this.calculateLSPContribution(lspFeatures);
        
        // Apply bounded enhancement
        const boundedContribution = Math.min(lspContribution, LSPStageCEnhancer.MAX_LSP_LOG_ODDS);
        if (boundedContribution < lspContribution) {
          scoreDistribution.bounded_adjustments++;
        }

        // Update candidate score
        const originalScore = candidate.score;
        candidate.score = this.combineScores(originalScore, boundedContribution);
        
        // Add LSP features to candidate metadata
        this.addLSPMetadata(candidate, lspFeatures);
        
        // Update statistics
        if (lspFeatures.lsp_def_hit) stats.def_hits++;
        if (lspFeatures.alias_resolved) stats.alias_resolved++;
        if (lspFeatures.type_match > 0) stats.type_matches++;
        
        totalRefCount += lspFeatures.lsp_ref_count;
        totalLspContribution += boundedContribution;
        scoreDistribution.lsp_contribution_max = Math.max(
          scoreDistribution.lsp_contribution_max,
          boundedContribution
        );
      }

      // Calculate final statistics
      if (enhancedCandidates.length > 0) {
        stats.avg_ref_count = totalRefCount / enhancedCandidates.length;
        scoreDistribution.lsp_contribution_avg = totalLspContribution / enhancedCandidates.length;
      }

      const processingTime = Date.now() - startTime;

      const result: LSPStageCResult = {
        enhanced_candidates: enhancedCandidates,
        feature_stats: stats,
        score_distribution: scoreDistribution,
        processing_time_ms: processingTime,
      };

      span.setAttributes({
        success: true,
        def_hits: stats.def_hits,
        type_matches: stats.type_matches,
        alias_resolved: stats.alias_resolved,
        bounded_adjustments: scoreDistribution.bounded_adjustments,
        processing_time_ms: processingTime,
      });

      return result;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Extract LSP features for a candidate
   */
  private extractLSPFeatures(candidate: Candidate, query: string): LSPFeatures {
    // Try to find matching LSP hint
    let hint = this.findHintForCandidate(candidate);
    
    const features: LSPFeatures = {
      lsp_def_hit: 0,
      lsp_ref_count: 0,
      type_match: 0,
      alias_resolved: 0,
    };

    if (!hint) return features;

    // Definition hit: Does LSP confirm this as a definition?
    if (this.isDefinitionHit(hint, candidate)) {
      features.lsp_def_hit = 1;
    }

    // Reference count from LSP
    features.lsp_ref_count = hint.references_count;

    // Type matching score
    features.type_match = this.calculateTypeMatch(hint, query);

    // Alias resolution: Was query resolved through an alias?
    if (this.isAliasResolved(hint, query)) {
      features.alias_resolved = 1;
    }

    return features;
  }

  /**
   * Find LSP hint for a candidate
   */
  private findHintForCandidate(candidate: Candidate): LSPHint | undefined {
    // First try exact location match
    const locationKey = `${candidate.file_path}:${candidate.line}:${candidate.col}`;
    let hint = this.hintsLookup.get(locationKey);
    
    if (hint) return hint;

    // Try symbol_id if available in doc_id
    hint = this.hintsLookup.get(candidate.doc_id);
    if (hint) return hint;

    // Try nearby locations (within 3 lines)
    for (let delta = 1; delta <= 3; delta++) {
      const nearbyKey1 = `${candidate.file_path}:${candidate.line + delta}:${candidate.col}`;
      const nearbyKey2 = `${candidate.file_path}:${candidate.line - delta}:${candidate.col}`;
      
      hint = this.hintsLookup.get(nearbyKey1) || this.hintsLookup.get(nearbyKey2);
      if (hint) return hint;
    }

    // Try matching by name in the same file
    for (const [key, h] of this.hintsLookup) {
      if (h.file_path === candidate.file_path && 
          candidate.context && 
          candidate.context.includes(h.name)) {
        return h;
      }
    }

    return undefined;
  }

  /**
   * Check if this is a definition hit
   */
  private isDefinitionHit(hint: LSPHint, candidate: Candidate): boolean {
    // LSP confirms this as a definition if it has definition URI
    if (hint.definition_uri) {
      return true;
    }

    // Check if candidate position matches hint definition position
    return hint.line === candidate.line && 
           hint.col === candidate.col &&
           hint.file_path === candidate.file_path;
  }

  /**
   * Calculate type matching score
   */
  private calculateTypeMatch(hint: LSPHint, query: string): number {
    if (!hint.type_info) return 0;

    const queryLower = query.toLowerCase();
    const typeInfoLower = hint.type_info.toLowerCase();

    // Exact type match
    if (typeInfoLower === queryLower) return 1.0;

    // Partial type match
    if (typeInfoLower.includes(queryLower)) {
      return queryLower.length / typeInfoLower.length;
    }

    // Generic type match (e.g., "Array<string>" matches "Array")
    const genericMatch = typeInfoLower.match(/^([^<]+)<.*>$/);
    if (genericMatch && genericMatch[1] === queryLower) {
      return 0.8;
    }

    // Union type match
    if (typeInfoLower.includes(' | ') && typeInfoLower.split(' | ').some(t => t.trim() === queryLower)) {
      return 0.7;
    }

    return 0;
  }

  /**
   * Check if query was resolved through an alias
   */
  private isAliasResolved(hint: LSPHint, query: string): boolean {
    const queryLower = query.toLowerCase();
    
    // Check if query matches any of the hint's aliases
    return hint.aliases.some(alias => alias.toLowerCase() === queryLower);
  }

  /**
   * Calculate LSP contribution in log-odds
   */
  private calculateLSPContribution(features: LSPFeatures): number {
    let contribution = 0;

    // Definition hit bonus
    if (features.lsp_def_hit) {
      contribution += LSPStageCEnhancer.DEF_HIT_BONUS;
    }

    // Reference count contribution (logarithmic scaling)
    if (features.lsp_ref_count > 0) {
      contribution += Math.log(1 + features.lsp_ref_count) * LSPStageCEnhancer.REF_COUNT_SCALING;
    }

    // Type match contribution
    contribution += features.type_match * LSPStageCEnhancer.TYPE_MATCH_BONUS;

    // Alias resolution bonus
    if (features.alias_resolved) {
      contribution += LSPStageCEnhancer.ALIAS_BONUS;
    }

    return contribution;
  }

  /**
   * Combine original score with LSP contribution
   */
  private combineScores(originalScore: number, lspContribution: number): number {
    // Convert score to log-odds, add LSP contribution, convert back
    const originalLogOdds = Math.log(originalScore / (1 - Math.min(originalScore, 0.99)));
    const combinedLogOdds = originalLogOdds + lspContribution;
    
    // Convert back to probability
    const combinedScore = Math.exp(combinedLogOdds) / (1 + Math.exp(combinedLogOdds));
    
    // Ensure score stays in valid range [0, 1]
    return Math.max(0, Math.min(1, combinedScore));
  }

  /**
   * Add LSP metadata to candidate
   */
  private addLSPMetadata(candidate: Candidate, features: LSPFeatures): void {
    // Add LSP features to candidate for debugging/analysis
    (candidate as any).lsp_features = features;
    
    // Update match reasons if LSP contributed significantly
    const totalFeatureScore = features.lsp_def_hit * LSPStageCEnhancer.DEF_HIT_BONUS +
                            features.type_match * LSPStageCEnhancer.TYPE_MATCH_BONUS +
                            features.alias_resolved * LSPStageCEnhancer.ALIAS_BONUS;
    
    if (totalFeatureScore > 0.1 && !candidate.match_reasons.includes('lsp_hint')) {
      candidate.match_reasons.push('lsp_hint');
    }
  }

  /**
   * Analyze LSP feature importance across candidates
   */
  analyzeLSPFeatureImportance(candidates: Candidate[]): {
    def_hit_importance: number;
    ref_count_importance: number;
    type_match_importance: number;
    alias_importance: number;
  } {
    const withLSP = candidates.filter(c => (c as any).lsp_features);
    if (withLSP.length === 0) {
      return {
        def_hit_importance: 0,
        ref_count_importance: 0,
        type_match_importance: 0,
        alias_importance: 0,
      };
    }

    // Calculate correlation between features and final scores
    const features = withLSP.map(c => (c as any).lsp_features as LSPFeatures);
    const scores = withLSP.map(c => c.score);

    return {
      def_hit_importance: this.calculateCorrelation(features.map(f => f.lsp_def_hit), scores),
      ref_count_importance: this.calculateCorrelation(features.map(f => Math.log(1 + f.lsp_ref_count)), scores),
      type_match_importance: this.calculateCorrelation(features.map(f => f.type_match), scores),
      alias_importance: this.calculateCorrelation(features.map(f => f.alias_resolved), scores),
    };
  }

  /**
   * Calculate Pearson correlation coefficient
   */
  private calculateCorrelation(x: number[], y: number[]): number {
    if (x.length !== y.length || x.length === 0) return 0;

    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);
    const sumYY = y.reduce((sum, yi) => sum + yi * yi, 0);

    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));

    return denominator === 0 ? 0 : numerator / denominator;
  }

  /**
   * Validate LSP contribution bounds
   */
  validateBounds(result: LSPStageCResult): {
    violations: number;
    max_violation: number;
    avg_contribution: number;
  } {
    let violations = 0;
    let maxViolation = 0;
    const contributions: number[] = [];

    for (const candidate of result.enhanced_candidates) {
      const lspFeatures = (candidate as any).lsp_features as LSPFeatures;
      if (!lspFeatures) continue;

      const contribution = this.calculateLSPContribution(lspFeatures);
      contributions.push(contribution);

      if (contribution > LSPStageCEnhancer.MAX_LSP_LOG_ODDS) {
        violations++;
        maxViolation = Math.max(maxViolation, contribution - LSPStageCEnhancer.MAX_LSP_LOG_ODDS);
      }
    }

    return {
      violations,
      max_violation: maxViolation,
      avg_contribution: contributions.length > 0 ? 
        contributions.reduce((a, b) => a + b, 0) / contributions.length : 0,
    };
  }

  /**
   * Get LSP Stage-C statistics
   */
  getStats(): {
    hints_loaded: number;
    max_log_odds_limit: number;
    feature_weights: {
      def_hit_bonus: number;
      ref_count_scaling: number;
      type_match_bonus: number;
      alias_bonus: number;
    };
  } {
    return {
      hints_loaded: this.hintsLookup.size,
      max_log_odds_limit: LSPStageCEnhancer.MAX_LSP_LOG_ODDS,
      feature_weights: {
        def_hit_bonus: LSPStageCEnhancer.DEF_HIT_BONUS,
        ref_count_scaling: LSPStageCEnhancer.REF_COUNT_SCALING,
        type_match_bonus: LSPStageCEnhancer.TYPE_MATCH_BONUS,
        alias_bonus: LSPStageCEnhancer.ALIAS_BONUS,
      },
    };
  }

  /**
   * Clear loaded hints
   */
  clear(): void {
    this.hintsLookup.clear();
  }
}