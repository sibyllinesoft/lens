/**
 * Stage-A Topic-Aware Planner for RAPTOR system
 * 
 * Enables topic-aware fan-out for NL and symbol intents only.
 * Uses the formula: k_candidates = base_k * (1 + α*topic_sim + β*topic_entropy)
 * with caps: k≤320, per_file_span_cap≤5, and conditional span cap increases.
 */

import { TopicTree, TopicSearchResult } from './topic-tree.js';
import { CardStore, EnhancedSemanticCard } from './card-store.js';
import { QueryIntent, IntentClassification } from '../types/core.js';

export interface TopicAwarePlannerConfig {
  // Formula coefficients
  alpha: number; // topic similarity weight
  beta: number;  // topic entropy weight
  
  // Caps and limits
  max_k_candidates: number;
  base_k_default: number;
  per_file_span_cap_base: number;
  per_file_span_cap_boosted: number;
  
  // Thresholds
  topic_similarity_threshold: number; // τ - when to increase span cap
  min_topic_entropy: number; // minimum entropy for boost
  intent_filter: QueryIntent[]; // which intents get topic awareness
  
  // Performance controls
  topic_search_timeout_ms: number;
  max_topics_considered: number;
}

export interface TopicPlanningRequest {
  query: string;
  query_embedding?: Float32Array;
  intent_classification: IntentClassification;
  base_k: number;
  candidate_files?: string[];
}

export interface TopicPlanningResult {
  k_candidates: number;
  per_file_span_cap: number;
  topic_boost_applied: boolean;
  topic_matches: TopicMatch[];
  reasoning: PlanningReasoning;
  planning_time_ms: number;
}

export interface TopicMatch {
  topic_id: string;
  similarity_score: number;
  keyword_overlap: number;
  coverage_relevance: number;
  associated_files: string[];
  boost_contribution: number;
}

export interface PlanningReasoning {
  intent_eligible: boolean;
  topic_similarity: number;
  topic_entropy: number;
  base_k: number;
  similarity_boost: number;
  entropy_boost: number;
  final_k: number;
  span_cap_reason: string;
  performance_notes: string[];
}

/**
 * Topic-aware planner for Stage-A candidate selection
 */
export class TopicAwarePlanner {
  private topicTree?: TopicTree;
  private cardStore?: CardStore;
  private config: TopicAwarePlannerConfig;

  constructor(config?: Partial<TopicAwarePlannerConfig>) {
    this.config = {
      // Formula parameters (tuned for semantic search)
      alpha: 0.3, // topic similarity coefficient
      beta: 0.2,  // topic entropy coefficient
      
      // Hard limits
      max_k_candidates: 320,
      base_k_default: 50,
      per_file_span_cap_base: 5,
      per_file_span_cap_boosted: 8,
      
      // Activation thresholds
      topic_similarity_threshold: 0.6, // τ for span cap increase
      min_topic_entropy: 0.3,
      intent_filter: ['NL', 'symbol'], // Only NL and symbol intents get topic awareness
      
      // Performance limits
      topic_search_timeout_ms: 100,
      max_topics_considered: 20,
      
      ...config
    };
  }

  /**
   * Initialize with RAPTOR components
   */
  initialize(topicTree: TopicTree, cardStore: CardStore): void {
    this.topicTree = topicTree;
    this.cardStore = cardStore;
  }

  /**
   * Generate topic-aware planning for Stage-A candidate selection
   */
  async planStageA(request: TopicPlanningRequest): Promise<TopicPlanningResult> {
    const startTime = Date.now();
    
    // Check if intent is eligible for topic awareness
    const intentEligible = this.config.intent_filter.includes(request.intent_classification.intent);
    
    if (!intentEligible) {
      return this.createBasicPlan(request, startTime, 'Intent not eligible for topic awareness');
    }

    if (!this.topicTree || !this.cardStore) {
      return this.createBasicPlan(request, startTime, 'Topic components not initialized');
    }

    try {
      // Search for relevant topics
      const topicMatches = await this.findRelevantTopics(request);
      
      // Compute topic similarity and entropy
      const topicMetrics = this.computeTopicMetrics(topicMatches);
      
      // Apply the planning formula
      const planningResult = this.applyPlanningFormula(
        request,
        topicMetrics,
        topicMatches,
        startTime
      );
      
      return planningResult;
      
    } catch (error) {
      return this.createBasicPlan(
        request, 
        startTime, 
        `Error in topic planning: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  private async findRelevantTopics(request: TopicPlanningRequest): Promise<TopicMatch[]> {
    if (!this.topicTree) {
      return [];
    }

    // Prepare topic search query
    const topicQuery = {
      query_embedding: request.query_embedding,
      keywords: this.extractKeywords(request.query),
      max_results: this.config.max_topics_considered,
      min_similarity: 0.3
    };

    // Search with timeout
    const searchPromise = this.topicTree.searchTopics(topicQuery);
    const timeoutPromise = new Promise<TopicSearchResult[]>((_, reject) => 
      setTimeout(() => reject(new Error('Topic search timeout')), this.config.topic_search_timeout_ms)
    );

    try {
      const topicResults = await Promise.race([searchPromise, timeoutPromise]);
      
      // Convert to TopicMatch format
      const topicMatches: TopicMatch[] = [];
      
      for (const result of topicResults) {
        const topic = this.topicTree!.getTopic(result.topic_id);
        if (!topic) continue;
        
        const associatedFiles = await this.getTopicFiles(result.topic_id);
        
        topicMatches.push({
          topic_id: result.topic_id,
          similarity_score: result.similarity_score,
          keyword_overlap: result.keyword_overlap,
          coverage_relevance: result.coverage_relevance,
          associated_files: associatedFiles,
          boost_contribution: 0 // Will be computed later
        });
      }
      
      return topicMatches;
      
    } catch (error) {
      // Return empty matches on timeout or error
      return [];
    }
  }

  private extractKeywords(query: string): string[] {
    // Simple keyword extraction - in real implementation would use better NLP
    return query.toLowerCase()
      .split(/\s+/)
      .filter(word => word.length > 2)
      .filter(word => !/^(the|and|or|but|in|on|at|to|for|of|with|by)$/.test(word));
  }

  private async getTopicFiles(topicId: string): Promise<string[]> {
    if (!this.topicTree) {
      return [];
    }
    
    const topic = this.topicTree.getTopic(topicId);
    if (!topic) {
      return [];
    }
    
    // Get files from associated card IDs
    const files: string[] = [];
    
    for (const cardId of topic.card_ids) {
      const card = this.cardStore?.getCard(cardId);
      if (card) {
        files.push(card.file_path);
      }
    }
    
    return [...new Set(files)]; // Remove duplicates
  }

  private computeTopicMetrics(topicMatches: TopicMatch[]): {
    topic_similarity: number;
    topic_entropy: number;
  } {
    if (topicMatches.length === 0) {
      return { topic_similarity: 0, topic_entropy: 0 };
    }
    
    // Topic similarity = max similarity among matched topics
    const topic_similarity = Math.max(...topicMatches.map(m => m.similarity_score));
    
    // Topic entropy = measure of how diverse the topic matches are
    // Higher entropy means query spans multiple distinct topics
    const similarities = topicMatches.map(m => m.similarity_score);
    const topic_entropy = this.computeEntropy(similarities);
    
    return { topic_similarity, topic_entropy };
  }

  private computeEntropy(values: number[]): number {
    if (values.length === 0) return 0;
    
    // Normalize values to probabilities
    const sum = values.reduce((acc, val) => acc + val, 0);
    if (sum === 0) return 0;
    
    const probabilities = values.map(val => val / sum);
    
    // Compute Shannon entropy
    let entropy = 0;
    for (const p of probabilities) {
      if (p > 0) {
        entropy -= p * Math.log2(p);
      }
    }
    
    // Normalize by max possible entropy
    const maxEntropy = Math.log2(values.length);
    return maxEntropy > 0 ? entropy / maxEntropy : 0;
  }

  private applyPlanningFormula(
    request: TopicPlanningRequest,
    metrics: { topic_similarity: number; topic_entropy: number },
    topicMatches: TopicMatch[],
    startTime: number
  ): TopicPlanningResult {
    const { alpha, beta } = this.config;
    const base_k = request.base_k || this.config.base_k_default;
    
    // Apply the formula: k_candidates = base_k * (1 + α*topic_sim + β*topic_entropy)
    const similarity_boost = alpha * metrics.topic_similarity;
    const entropy_boost = beta * metrics.topic_entropy;
    const boost_factor = 1 + similarity_boost + entropy_boost;
    
    let k_candidates = Math.round(base_k * boost_factor);
    
    // Apply hard cap
    k_candidates = Math.min(k_candidates, this.config.max_k_candidates);
    
    // Determine per-file span cap
    let per_file_span_cap = this.config.per_file_span_cap_base;
    let span_cap_reason = 'Base span cap applied';
    
    // Only increase span cap when topic_sim > τ
    if (metrics.topic_similarity > this.config.topic_similarity_threshold) {
      per_file_span_cap = this.config.per_file_span_cap_boosted;
      span_cap_reason = `Increased span cap due to high topic similarity (${metrics.topic_similarity.toFixed(3)} > ${this.config.topic_similarity_threshold})`;
    }
    
    // Update topic match boost contributions
    for (const match of topicMatches) {
      match.boost_contribution = (match.similarity_score * similarity_boost) + 
                                 (this.computeMatchEntropy(match, topicMatches) * entropy_boost);
    }
    
    const topic_boost_applied = boost_factor > 1.01; // Meaningful boost threshold
    
    // Performance notes
    const performance_notes = [];
    const planning_time = Date.now() - startTime;
    
    if (planning_time > 50) {
      performance_notes.push(`Topic search took ${planning_time}ms`);
    }
    
    if (topicMatches.length > 10) {
      performance_notes.push(`High topic match count: ${topicMatches.length}`);
    }
    
    return {
      k_candidates,
      per_file_span_cap,
      topic_boost_applied,
      topic_matches: topicMatches,
      reasoning: {
        intent_eligible: true,
        topic_similarity: metrics.topic_similarity,
        topic_entropy: metrics.topic_entropy,
        base_k,
        similarity_boost,
        entropy_boost,
        final_k: k_candidates,
        span_cap_reason,
        performance_notes
      },
      planning_time_ms: planning_time
    };
  }

  private computeMatchEntropy(match: TopicMatch, allMatches: TopicMatch[]): number {
    // Contribution of this match to overall entropy
    const similarities = allMatches.map(m => m.similarity_score);
    const totalSim = similarities.reduce((sum, sim) => sum + sim, 0);
    
    if (totalSim === 0) return 0;
    
    const p = match.similarity_score / totalSim;
    return p > 0 ? -p * Math.log2(p) : 0;
  }

  private createBasicPlan(
    request: TopicPlanningRequest, 
    startTime: number, 
    reason: string
  ): TopicPlanningResult {
    const base_k = request.base_k || this.config.base_k_default;
    
    return {
      k_candidates: base_k,
      per_file_span_cap: this.config.per_file_span_cap_base,
      topic_boost_applied: false,
      topic_matches: [],
      reasoning: {
        intent_eligible: this.config.intent_filter.includes(request.intent_classification.intent),
        topic_similarity: 0,
        topic_entropy: 0,
        base_k,
        similarity_boost: 0,
        entropy_boost: 0,
        final_k: base_k,
        span_cap_reason: 'Basic plan: ' + reason,
        performance_notes: []
      },
      planning_time_ms: Date.now() - startTime
    };
  }

  /**
   * Get file-level recommendations based on topic matches
   */
  getTopicFileRecommendations(planningResult: TopicPlanningResult): {
    boosted_files: string[];
    topic_coverage: Map<string, number>;
  } {
    const boosted_files = new Set<string>();
    const topic_coverage = new Map<string, number>();
    
    for (const match of planningResult.topic_matches) {
      if (match.boost_contribution > 0.1) { // Meaningful contribution threshold
        for (const file of match.associated_files) {
          boosted_files.add(file);
          
          // Track cumulative topic coverage per file
          const currentCoverage = topic_coverage.get(file) || 0;
          topic_coverage.set(file, currentCoverage + match.boost_contribution);
        }
      }
    }
    
    return {
      boosted_files: Array.from(boosted_files),
      topic_coverage
    };
  }

  /**
   * Validate planning result quality
   */
  validatePlanningQuality(result: TopicPlanningResult): {
    quality_score: number;
    issues: string[];
    recommendations: string[];
  } {
    const issues: string[] = [];
    const recommendations: string[] = [];
    let quality_score = 1.0;

    // Check for runaway k values
    if (result.k_candidates > this.config.max_k_candidates * 0.8) {
      issues.push('Very high k_candidates value may impact performance');
      quality_score *= 0.8;
    }

    // Check topic match quality
    if (result.topic_boost_applied) {
      const avgSimilarity = result.topic_matches.reduce((sum, m) => sum + m.similarity_score, 0) / 
                           Math.max(1, result.topic_matches.length);
      
      if (avgSimilarity < 0.4) {
        issues.push('Low average topic similarity despite boost application');
        quality_score *= 0.7;
        recommendations.push('Consider adjusting topic similarity thresholds');
      }
      
      // Check for topic diversity
      if (result.reasoning.topic_entropy < this.config.min_topic_entropy) {
        recommendations.push('Low topic entropy - query might be too narrow');
      }
    }

    // Check planning performance
    if (result.planning_time_ms > this.config.topic_search_timeout_ms * 0.8) {
      issues.push('Topic planning time approaching timeout threshold');
      recommendations.push('Consider reducing max_topics_considered or optimizing topic search');
    }

    // Check span cap logic
    if (result.per_file_span_cap > this.config.per_file_span_cap_base && 
        result.reasoning.topic_similarity <= this.config.topic_similarity_threshold) {
      issues.push('Span cap increased without meeting similarity threshold');
      quality_score *= 0.6;
    }

    return {
      quality_score,
      issues,
      recommendations
    };
  }

  /**
   * Update configuration dynamically
   */
  updateConfig(newConfig: Partial<TopicAwarePlannerConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  /**
   * Get current configuration
   */
  getConfig(): TopicAwarePlannerConfig {
    return { ...this.config };
  }

  /**
   * Export planning statistics for tuning
   */
  exportPlanningStats(results: TopicPlanningResult[]): {
    avg_k_boost: number;
    avg_span_cap_increase: number;
    topic_boost_rate: number;
    avg_planning_time_ms: number;
    intent_distribution: Record<string, number>;
  } {
    if (results.length === 0) {
      return {
        avg_k_boost: 0,
        avg_span_cap_increase: 0,
        topic_boost_rate: 0,
        avg_planning_time_ms: 0,
        intent_distribution: {}
      };
    }

    const k_boosts = results.map(r => 
      r.reasoning.final_k / Math.max(1, r.reasoning.base_k) - 1
    );
    
    const span_cap_increases = results.map(r => 
      r.per_file_span_cap > this.config.per_file_span_cap_base ? 1 : 0
    );
    
    const topic_boosts = results.filter(r => r.topic_boost_applied).length;
    
    return {
      avg_k_boost: k_boosts.reduce((sum, boost) => sum + boost, 0) / k_boosts.length,
      avg_span_cap_increase: span_cap_increases.reduce((sum, inc) => sum + inc, 0) / span_cap_increases.length,
      topic_boost_rate: topic_boosts / results.length,
      avg_planning_time_ms: results.reduce((sum, r) => sum + r.planning_time_ms, 0) / results.length,
      intent_distribution: {} // Would need intent data to compute
    };
  }
}