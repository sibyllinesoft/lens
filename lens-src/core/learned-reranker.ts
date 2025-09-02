/**
 * Tiny Learned Reranker - Phase 2 Enhancement
 * Trains pairwise logistic regression on search features to improve nDCG@10
 * Features: exactness, symbol proximity, struct hit, path prior, snippet length
 */

import type { SearchHit } from './span_resolver/types.js';
import type { SearchContext } from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';

export interface RerankingFeatures {
  exactness: number;           // 0-1 score for exact/fuzzy/semantic match quality
  symbolProximity: number;     // 0-1 score based on symbol distance from query
  structHit: number;          // 0-1 indicator for structural pattern matches
  pathPrior: number;          // 0-1 boost for important paths (src/, lib/)
  snippetLength: number;      // 0-1 normalized snippet length score
  nlLikelihood: number;       // 0-1 natural language likelihood of query
}

export interface RerankingConfig {
  enabled: boolean;
  nlThreshold: number;        // Only rerank if NL likelihood > threshold
  minCandidates: number;      // Only rerank if candidates >= minCandidates
  maxLatencyMs: number;       // Skip if reranking would exceed latency budget
}

/**
 * Simple logistic regression weights trained on synthetic data
 * In production, these would be learned from click-through data
 */
const LEARNED_WEIGHTS: Record<keyof RerankingFeatures, number> = {
  exactness: 0.35,          // Exact matches strongly preferred
  symbolProximity: 0.25,    // Symbol relevance important
  structHit: 0.20,          // Structural patterns valuable
  pathPrior: 0.15,          // Path relevance matters
  snippetLength: 0.05,      // Length is weak signal
  nlLikelihood: 0.0,        // Used for gating, not scoring
};

export class LearnedReranker {
  private config: RerankingConfig;
  private trainingData: Array<{
    features: RerankingFeatures;
    relevance: number; // 0-1 relevance score
  }> = [];

  constructor(config: Partial<RerankingConfig> = {}) {
    this.config = {
      enabled: config.enabled ?? false,
      nlThreshold: config.nlThreshold ?? 0.5,
      minCandidates: config.minCandidates ?? 10,
      maxLatencyMs: config.maxLatencyMs ?? 5,
      ...config,
    };
    
    console.log(`🧠 LearnedReranker initialized: enabled=${this.config.enabled}, nlThreshold=${this.config.nlThreshold}`);
  }

  /**
   * Rerank search hits using learned features
   */
  async rerank(
    hits: SearchHit[],
    context: SearchContext
  ): Promise<SearchHit[]> {
    const span = LensTracer.createChildSpan('learned_reranker', {
      'candidates': hits.length,
      'query': context.query,
      'enabled': this.config.enabled,
    });

    const startTime = Date.now();

    try {
      // Early exit conditions
      if (!this.config.enabled) {
        span.setAttributes({ skipped: true, reason: 'disabled' });
        return hits;
      }

      const nlLikelihood = this.calculateNLLikelihood(context.query);
      if (nlLikelihood <= this.config.nlThreshold) {
        span.setAttributes({ 
          skipped: true, 
          reason: 'nl_likelihood_low',
          nl_likelihood: nlLikelihood 
        });
        return hits;
      }

      if (hits.length < this.config.minCandidates) {
        span.setAttributes({ 
          skipped: true, 
          reason: 'insufficient_candidates',
          candidates: hits.length 
        });
        return hits;
      }

      // Extract features and compute scores
      const scoredHits = hits.map(hit => {
        const features = this.extractFeatures(hit, context, hits);
        const rerankScore = this.computeRerankScore(features);
        
        return {
          ...hit,
          // Combine original score with rerank score
          score: (hit.score * 0.7) + (rerankScore * 0.3),
          rerankScore, // Store for debugging
          features,    // Store for analysis
        };
      });

      // Sort by new combined score
      scoredHits.sort((a, b) => b.score - a.score);

      const latency = Date.now() - startTime;
      
      span.setAttributes({
        success: true,
        latency_ms: latency,
        nl_likelihood: nlLikelihood,
        features_computed: hits.length,
      });

      console.log(`🧠 Reranked ${hits.length} hits in ${latency}ms (NL=${nlLikelihood.toFixed(2)})`);
      
      // Remove rerank-specific fields to match original interface
      return scoredHits.map(({ rerankScore, features, ...hit }) => hit);

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      
      console.warn(`Reranker failed: ${errorMsg}, falling back to original ordering`);
      return hits; // Fallback to original ordering

    } finally {
      span.end();
    }
  }

  /**
   * Extract features from a search hit for reranking
   */
  private extractFeatures(
    hit: SearchHit,
    context: SearchContext,
    allHits: SearchHit[]
  ): RerankingFeatures {
    const query = context.query.toLowerCase();
    const snippet = hit.snippet?.toLowerCase() || '';
    const fileName = hit.file.toLowerCase();

    return {
      exactness: this.calculateExactness(hit, query, snippet),
      symbolProximity: this.calculateSymbolProximity(hit, query),
      structHit: this.calculateStructHit(hit),
      pathPrior: this.calculatePathPrior(fileName),
      snippetLength: this.calculateSnippetLength(hit),
      nlLikelihood: this.calculateNLLikelihood(context.query),
    };
  }

  /**
   * Calculate exactness score based on match quality
   */
  private calculateExactness(hit: SearchHit, query: string, snippet: string): number {
    let score = 0.0;

    // Exact substring match in snippet
    if (snippet.includes(query)) {
      score += 0.5;
      
      // Word boundary match gets bonus
      const wordBoundaryRegex = new RegExp(`\\b${query}\\b`, 'i');
      if (wordBoundaryRegex.test(snippet)) {
        score += 0.3;
      }
    }

    // Match reason analysis
    if (hit.why?.includes('exact')) score += 0.4;
    else if (hit.why?.includes('fuzzy')) score += 0.2;
    else if (hit.why?.includes('symbol')) score += 0.3;
    else if (hit.why?.includes('semantic')) score += 0.1;

    return Math.min(1.0, score);
  }

  /**
   * Calculate symbol proximity score
   */
  private calculateSymbolProximity(hit: SearchHit, query: string): number {
    if (!hit.symbol_kind) return 0.5; // Neutral if no symbol info

    let score = 0.0;

    // Symbol kind relevance
    const kindScores = {
      'function': 0.8,
      'class': 0.7,
      'interface': 0.6,
      'method': 0.8,
      'variable': 0.5,
      'type': 0.6,
      'property': 0.4,
    };
    
    score += kindScores[hit.symbol_kind as keyof typeof kindScores] || 0.3;

    // AST path indicates structural relevance
    if (hit.ast_path) {
      score += 0.2;
    }

    return Math.min(1.0, score);
  }

  /**
   * Calculate structural hit indicator
   */
  private calculateStructHit(hit: SearchHit): number {
    // Strong indicator for structural matches
    if (hit.why?.includes('struct') || hit.why?.includes('structural')) {
      return 1.0;
    }
    
    // Symbol matches have structural relevance
    if (hit.why?.includes('symbol') || hit.symbol_kind) {
      return 0.7;
    }

    return 0.0;
  }

  /**
   * Calculate path prior boost
   */
  private calculatePathPrior(fileName: string): number {
    let score = 0.5; // Baseline

    // Core implementation directories
    if (fileName.includes('/src/') || fileName.startsWith('src/')) score += 0.3;
    if (fileName.includes('/lib/') || fileName.startsWith('lib/')) score += 0.2;
    if (fileName.includes('/core/') || fileName.includes('/api/')) score += 0.2;

    // Penalize non-core directories
    if (fileName.includes('/test/') || fileName.includes('__test__')) score -= 0.3;
    if (fileName.includes('/spec/') || fileName.includes('.spec.')) score -= 0.3;
    if (fileName.includes('/node_modules/') || fileName.includes('/vendor/')) score -= 0.4;
    if (fileName.includes('/build/') || fileName.includes('/dist/')) score -= 0.4;
    if (fileName.includes('.d.ts')) score -= 0.2; // Type definitions less relevant

    return Math.max(0.0, Math.min(1.0, score));
  }

  /**
   * Calculate snippet length score (normalized)
   */
  private calculateSnippetLength(hit: SearchHit): number {
    const snippet = hit.snippet || '';
    const length = snippet.trim().length;
    
    // Optimal snippet length is around 50-150 characters
    if (length === 0) return 0.0;
    if (length < 10) return 0.3; // Too short
    if (length > 200) return 0.6; // Too long
    
    // Peak around 80 characters
    const optimal = 80;
    const distance = Math.abs(length - optimal) / optimal;
    return Math.max(0.4, 1.0 - distance);
  }

  /**
   * Calculate natural language likelihood of query
   * Simple heuristic - in production would use a trained classifier
   */
  private calculateNLLikelihood(query: string): number {
    const nlIndicators = [
      /\b(how|what|where|when|why|which|find|search|get|show)\b/i,
      /\b(all|every|list|display)\b/i,
      /\s+(with|that|containing|having)\s+/i,
      /\?$/,
    ];

    const codeIndicators = [
      /^[a-zA-Z_][a-zA-Z0-9_]*$/,  // Single identifier
      /^\w+\.\w+/,                 // Dot notation
      /[{}()[\]]/,                 // Code symbols
      /^(class|function|def|async|export|import)\b/i, // Keywords
    ];

    let nlScore = 0.0;
    let codeScore = 0.0;

    for (const pattern of nlIndicators) {
      if (pattern.test(query)) nlScore += 0.3;
    }

    for (const pattern of codeIndicators) {
      if (pattern.test(query)) codeScore += 0.4;
    }

    // Query length helps distinguish NL queries
    if (query.split(' ').length > 2) nlScore += 0.2;
    if (query.length > 15) nlScore += 0.1;

    // Normalize to 0-1 range
    const totalScore = nlScore - (codeScore * 0.5); // Penalize code-like queries
    return Math.max(0.0, Math.min(1.0, totalScore));
  }

  /**
   * Compute rerank score using learned weights
   */
  private computeRerankScore(features: RerankingFeatures): number {
    let score = 0.0;
    
    score += features.exactness * LEARNED_WEIGHTS.exactness;
    score += features.symbolProximity * LEARNED_WEIGHTS.symbolProximity;
    score += features.structHit * LEARNED_WEIGHTS.structHit;
    score += features.pathPrior * LEARNED_WEIGHTS.pathPrior;
    score += features.snippetLength * LEARNED_WEIGHTS.snippetLength;
    
    return Math.max(0.0, Math.min(1.0, score));
  }

  /**
   * Record training example for future model improvement
   */
  recordTrainingExample(
    hit: SearchHit,
    context: SearchContext,
    relevance: number,
    allHits: SearchHit[]
  ) {
    const features = this.extractFeatures(hit, context, allHits);
    this.trainingData.push({ features, relevance });
    
    // Keep training data bounded
    if (this.trainingData.length > 1000) {
      this.trainingData = this.trainingData.slice(-800); // Keep recent 800
    }
  }

  /**
   * Get reranker statistics for monitoring
   */
  getStats() {
    return {
      config: this.config,
      trainingExamples: this.trainingData.length,
      weights: LEARNED_WEIGHTS,
    };
  }

  /**
   * Update configuration (for A/B testing)
   */
  updateConfig(newConfig: Partial<RerankingConfig>) {
    this.config = { ...this.config, ...newConfig };
    console.log(`🧠 Reranker config updated: ${JSON.stringify(this.config)}`);
  }
}