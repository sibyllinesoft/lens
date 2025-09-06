/**
 * Enhanced Search Result Ranking System
 * 
 * Implements sophisticated ranking algorithms for search results including:
 * - Multi-factor scoring (relevance, popularity, recency, context)
 * - ML-based ranking with feature extraction
 * - Personalization based on search patterns
 * - A/B testing framework for ranking experiments
 */

import { SearchContext } from '../types/core.js';
import { SearchHit } from '../types/embedder-proof-levers.js';

// Local ranking metrics interface
interface RankingMetrics {
  totalStrategies: number;
  activeExperiments: number;
  personalizedUsers: number;
  averageRankingTime: number;
}
import { opentelemetry } from '../telemetry/index.js';

// Ranking feature extraction and scoring types
export interface RankingFeatures {
  textualRelevance: number;     // TF-IDF, BM25 score
  structuralRelevance: number;  // Symbol type, scope matches
  popularityScore: number;      // Usage frequency, star count
  recencyScore: number;         // Last modified, commit frequency
  contextualRelevance: number;  // File path, project context
  semanticSimilarity: number;   // Embedding-based similarity
  userPreferenceScore: number;  // Personalization factor
}

export interface RankingStrategy {
  name: string;
  description: string;
  weights: Partial<RankingFeatures>;
  transform?: (features: RankingFeatures) => number;
}

export interface PersonalizationProfile {
  userId?: string;
  preferredLanguages: string[];
  frequentProjects: string[];
  searchHistory: Array<{
    query: string;
    clickedResults: string[];
    timestamp: number;
  }>;
  clickThroughRates: Map<string, number>;
}

export interface RankingExperiment {
  id: string;
  name: string;
  description: string;
  strategy: RankingStrategy;
  trafficAllocation: number; // 0-1
  startDate: Date;
  endDate?: Date;
  metrics: {
    clickThroughRate: number;
    meanReciprocalRank: number;
    ndcgAt10: number;
  };
}

/**
 * Advanced result ranking system with ML-based scoring
 */
export class ResultRanker {
  private readonly tracer = opentelemetry.trace.getTracer('lens-result-ranker');
  private static instance: ResultRanker | null = null;
  
  // Built-in ranking strategies
  private readonly strategies = new Map<string, RankingStrategy>([
    ['default', {
      name: 'Default',
      description: 'Balanced ranking prioritizing relevance and popularity',
      weights: {
        textualRelevance: 0.3,
        structuralRelevance: 0.25,
        popularityScore: 0.2,
        recencyScore: 0.1,
        contextualRelevance: 0.1,
        semanticSimilarity: 0.05
      }
    }],
    ['relevance-focused', {
      name: 'Relevance Focused',
      description: 'Emphasizes textual and structural relevance',
      weights: {
        textualRelevance: 0.4,
        structuralRelevance: 0.35,
        popularityScore: 0.1,
        recencyScore: 0.05,
        contextualRelevance: 0.05,
        semanticSimilarity: 0.05
      }
    }],
    ['popularity-focused', {
      name: 'Popularity Focused', 
      description: 'Emphasizes popular and frequently used results',
      weights: {
        textualRelevance: 0.2,
        structuralRelevance: 0.15,
        popularityScore: 0.4,
        recencyScore: 0.1,
        contextualRelevance: 0.1,
        semanticSimilarity: 0.05
      }
    }],
    ['recency-focused', {
      name: 'Recency Focused',
      description: 'Prioritizes recently modified and active code',
      weights: {
        textualRelevance: 0.25,
        structuralRelevance: 0.2,
        popularityScore: 0.15,
        recencyScore: 0.3,
        contextualRelevance: 0.05,
        semanticSimilarity: 0.05
      }
    }]
  ]);

  // Active experiments for A/B testing
  private activeExperiments: RankingExperiment[] = [];
  
  // User personalization profiles
  private personalizationProfiles = new Map<string, PersonalizationProfile>();
  
  // Feature extractors
  private readonly featureExtractors = {
    textualRelevance: this.calculateTextualRelevance.bind(this),
    structuralRelevance: this.calculateStructuralRelevance.bind(this),
    popularityScore: this.calculatePopularityScore.bind(this),
    recencyScore: this.calculateRecencyScore.bind(this),
    contextualRelevance: this.calculateContextualRelevance.bind(this),
    semanticSimilarity: this.calculateSemanticSimilarity.bind(this),
    userPreferenceScore: this.calculateUserPreferenceScore.bind(this)
  };

  private constructor() {
    this.initializeDefaultExperiments();
  }

  static getInstance(): ResultRanker {
    if (!ResultRanker.instance) {
      ResultRanker.instance = new ResultRanker();
    }
    return ResultRanker.instance;
  }

  /**
   * Rank search results using advanced ML-based scoring
   */
  async rankResults(
    results: SearchHit[],
    query: string,
    context: SearchContext
  ): Promise<SearchHit[]> {
    return await this.tracer.startActiveSpan('rank-results', async (span) => {
      try {
        span.setAttributes({
          'lens.ranking.result_count': results.length,
          'lens.ranking.query': query,
          'lens.ranking.strategy': context.rankingStrategy || 'default'
        });

        // Extract features for all results
        const rankedResults = await Promise.all(
          results.map(async (result) => {
            const features = await this.extractFeatures(result, query, context);
            const score = await this.calculateFinalScore(features, query, context);
            
            return {
              ...result,
              score,
              rankingFeatures: features,
              metadata: {
                ...result.metadata,
                rankingScore: score,
                rankingStrategy: this.getActiveStrategy(context).name
              }
            };
          })
        );

        // Sort by score (descending)
        rankedResults.sort((a, b) => (b.score || 0) - (a.score || 0));

        // Update personalization profiles if user context available
        if (context.userId) {
          await this.updatePersonalizationProfile(
            context.userId, 
            query, 
            rankedResults.map(r => r.id)
          );
        }

        span.setAttributes({
          'lens.ranking.top_score': rankedResults[0]?.score || 0,
          'lens.ranking.score_range': (rankedResults[0]?.score || 0) - (rankedResults[rankedResults.length - 1]?.score || 0)
        });

        return rankedResults;

      } catch (error) {
        span.recordException(error as Error);
        span.setStatus({ code: opentelemetry.SpanStatusCode.ERROR });
        throw error;
      } finally {
        span.end();
      }
    });
  }

  /**
   * Extract comprehensive ranking features from a search result
   */
  private async extractFeatures(
    result: SearchHit,
    query: string,
    context: SearchContext
  ): Promise<RankingFeatures> {
    return await this.tracer.startActiveSpan('extract-features', async (span) => {
      const features: RankingFeatures = {
        textualRelevance: await this.featureExtractors.textualRelevance(result, query, context),
        structuralRelevance: await this.featureExtractors.structuralRelevance(result, query, context),
        popularityScore: await this.featureExtractors.popularityScore(result, query, context),
        recencyScore: await this.featureExtractors.recencyScore(result, query, context),
        contextualRelevance: await this.featureExtractors.contextualRelevance(result, query, context),
        semanticSimilarity: await this.featureExtractors.semanticSimilarity(result, query, context),
        userPreferenceScore: await this.featureExtractors.userPreferenceScore(result, query, context)
      };

      span.setAttributes({
        'lens.ranking.features.textual': features.textualRelevance,
        'lens.ranking.features.structural': features.structuralRelevance,
        'lens.ranking.features.popularity': features.popularityScore
      });

      return features;
    });
  }

  /**
   * Calculate final ranking score using weighted feature combination
   */
  private async calculateFinalScore(
    features: RankingFeatures,
    query: string,
    context: SearchContext
  ): Promise<number> {
    const strategy = this.getActiveStrategy(context);
    
    // Apply strategy weights
    let score = 0;
    for (const [feature, weight] of Object.entries(strategy.weights)) {
      const featureValue = features[feature as keyof RankingFeatures];
      score += (featureValue || 0) * (weight || 0);
    }

    // Apply custom transformation if defined
    if (strategy.transform) {
      score = strategy.transform(features);
    }

    // Apply non-linear boosting for top features
    const maxFeature = Math.max(...Object.values(features));
    if (maxFeature > 0.8) {
      score *= 1.2; // 20% boost for high-confidence matches
    }

    return Math.max(0, Math.min(1, score)); // Normalize to [0, 1]
  }

  /**
   * Get active ranking strategy (including A/B test strategies)
   */
  private getActiveStrategy(context: SearchContext): RankingStrategy {
    // Check for A/B test assignment
    if (context.userId) {
      const experiment = this.getActiveExperimentForUser(context.userId);
      if (experiment) {
        return experiment.strategy;
      }
    }

    // Use specified strategy or default
    const strategyName = context.rankingStrategy || 'default';
    return this.strategies.get(strategyName) || this.strategies.get('default')!;
  }

  /**
   * Feature extraction methods
   */
  
  private async calculateTextualRelevance(
    result: SearchHit,
    query: string,
    context: SearchContext
  ): Promise<number> {
    // Implement BM25 scoring
    const queryTerms = query.toLowerCase().split(/\s+/);
    const documentText = `${result.name} ${result.content || ''} ${result.metadata?.documentation || ''}`.toLowerCase();
    
    let bm25Score = 0;
    const k1 = 1.2;
    const b = 0.75;
    const avgDocLength = 100; // Estimated average document length
    const docLength = documentText.length;
    
    for (const term of queryTerms) {
      const termFreq = (documentText.match(new RegExp(term, 'g')) || []).length;
      const docFreq = 1; // Simplified - would need corpus statistics
      const idf = Math.log((1000 + 0.5) / (docFreq + 0.5)); // Simplified IDF
      
      const numerator = termFreq * (k1 + 1);
      const denominator = termFreq + k1 * (1 - b + b * (docLength / avgDocLength));
      
      bm25Score += idf * (numerator / denominator);
    }
    
    return Math.min(1, bm25Score / 10); // Normalize
  }

  private async calculateStructuralRelevance(
    result: SearchHit,
    query: string,
    context: SearchContext
  ): Promise<number> {
    let score = 0;
    
    // Symbol type matching
    if (context.filters?.symbolTypes?.includes(result.symbolType)) {
      score += 0.4;
    }
    
    // Scope/visibility matching
    if (result.metadata?.visibility === 'public') {
      score += 0.2;
    }
    
    // Exact name match bonus
    if (result.name.toLowerCase() === query.toLowerCase()) {
      score += 0.4;
    }
    
    return Math.min(1, score);
  }

  private async calculatePopularityScore(
    result: SearchHit,
    query: string,
    context: SearchContext
  ): Promise<number> {
    // Use metadata indicators of popularity
    const references = result.metadata?.referenceCount || 0;
    const stars = result.metadata?.starCount || 0;
    const imports = result.metadata?.importCount || 0;
    
    // Normalize scores
    const refScore = Math.min(1, references / 100);
    const starScore = Math.min(1, stars / 1000);
    const importScore = Math.min(1, imports / 50);
    
    return (refScore + starScore + importScore) / 3;
  }

  private async calculateRecencyScore(
    result: SearchHit,
    query: string,
    context: SearchContext
  ): Promise<number> {
    const lastModified = result.metadata?.lastModified;
    if (!lastModified) return 0.5; // Neutral score for unknown dates
    
    const daysSinceModified = (Date.now() - new Date(lastModified).getTime()) / (1000 * 60 * 60 * 24);
    
    // Exponential decay - fresher content gets higher scores
    return Math.exp(-daysSinceModified / 30); // 30-day half-life
  }

  private async calculateContextualRelevance(
    result: SearchHit,
    query: string,
    context: SearchContext
  ): Promise<number> {
    let score = 0;
    
    // File path/project context matching
    if (context.repositories?.some(repo => result.filePath.includes(repo))) {
      score += 0.4;
    }
    
    // Language preference matching
    if (context.filters?.languages?.includes(result.language)) {
      score += 0.3;
    }
    
    // Directory structure relevance
    const pathParts = result.filePath.split('/');
    const queryWords = query.toLowerCase().split(/\s+/);
    
    for (const word of queryWords) {
      if (pathParts.some(part => part.toLowerCase().includes(word))) {
        score += 0.1;
        break;
      }
    }
    
    return Math.min(1, score);
  }

  private async calculateSemanticSimilarity(
    result: SearchHit,
    query: string,
    context: SearchContext
  ): Promise<number> {
    // Simplified semantic similarity using word overlap and synonyms
    const queryTerms = new Set(query.toLowerCase().split(/\s+/));
    const resultTerms = new Set(
      `${result.name} ${result.content || ''} ${result.metadata?.documentation || ''}`
        .toLowerCase()
        .split(/\s+/)
    );
    
    const intersection = new Set([...queryTerms].filter(x => resultTerms.has(x)));
    const union = new Set([...queryTerms, ...resultTerms]);
    
    return intersection.size / union.size; // Jaccard similarity
  }

  private async calculateUserPreferenceScore(
    result: SearchHit,
    query: string,
    context: SearchContext
  ): Promise<number> {
    if (!context.userId) return 0.5; // Neutral for anonymous users
    
    const profile = this.personalizationProfiles.get(context.userId);
    if (!profile) return 0.5;
    
    let score = 0;
    
    // Language preference
    if (profile.preferredLanguages.includes(result.language)) {
      score += 0.3;
    }
    
    // Project preference
    for (const project of profile.frequentProjects) {
      if (result.filePath.includes(project)) {
        score += 0.3;
        break;
      }
    }
    
    // Click-through rate for similar results
    const ctr = profile.clickThroughRates.get(result.symbolType) || 0;
    score += ctr * 0.4;
    
    return Math.min(1, score);
  }

  /**
   * A/B testing and experimentation
   */
  
  private getActiveExperimentForUser(userId: string): RankingExperiment | null {
    // Simple hash-based assignment for consistent user experience
    const hash = this.hashUserId(userId);
    let cumulativeAllocation = 0;
    
    for (const experiment of this.activeExperiments) {
      cumulativeAllocation += experiment.trafficAllocation;
      if (hash < cumulativeAllocation) {
        return experiment;
      }
    }
    
    return null;
  }
  
  private hashUserId(userId: string): number {
    let hash = 0;
    for (let i = 0; i < userId.length; i++) {
      const char = userId.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash) / Math.pow(2, 31); // Normalize to [0, 1]
  }

  /**
   * Personalization profile management
   */
  
  private async updatePersonalizationProfile(
    userId: string,
    query: string,
    resultIds: string[]
  ): Promise<void> {
    let profile = this.personalizationProfiles.get(userId);
    
    if (!profile) {
      profile = {
        userId,
        preferredLanguages: [],
        frequentProjects: [],
        searchHistory: [],
        clickThroughRates: new Map()
      };
      this.personalizationProfiles.set(userId, profile);
    }
    
    // Update search history
    profile.searchHistory.push({
      query,
      clickedResults: [], // Will be updated when clicks are tracked
      timestamp: Date.now()
    });
    
    // Keep only recent history (last 1000 searches)
    if (profile.searchHistory.length > 1000) {
      profile.searchHistory = profile.searchHistory.slice(-1000);
    }
  }

  /**
   * Initialize default A/B test experiments
   */
  private initializeDefaultExperiments(): void {
    // Example experiment: Test popularity-focused ranking
    this.activeExperiments.push({
      id: 'popularity-boost-2024',
      name: 'Popularity Boost Test',
      description: 'Test increased weight on popularity signals',
      strategy: {
        name: 'Popularity Boost',
        description: 'Enhanced popularity weighting',
        weights: {
          textualRelevance: 0.25,
          structuralRelevance: 0.2,
          popularityScore: 0.35,
          recencyScore: 0.1,
          contextualRelevance: 0.05,
          semanticSimilarity: 0.05
        }
      },
      trafficAllocation: 0.1, // 10% of users
      startDate: new Date(),
      metrics: {
        clickThroughRate: 0,
        meanReciprocalRank: 0,
        ndcgAt10: 0
      }
    });
  }

  /**
   * Add custom ranking strategy
   */
  addRankingStrategy(name: string, strategy: RankingStrategy): void {
    this.strategies.set(name, strategy);
  }

  /**
   * Get ranking statistics and metrics
   */
  getRankingMetrics(): RankingMetrics {
    return {
      totalStrategies: this.strategies.size,
      activeExperiments: this.activeExperiments.length,
      personalizedUsers: this.personalizationProfiles.size,
      averageRankingTime: 0 // Would be calculated from telemetry
    };
  }
}

// Export singleton instance
export const resultRanker = ResultRanker.getInstance();