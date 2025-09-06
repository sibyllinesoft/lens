/**
 * Stage-C RAPTOR Features for semantic reranking
 * 
 * Adds bounded RAPTOR features with capped influence:
 * - raptor_max_sim: Maximum similarity across topic matches
 * - topic_overlap: Overlap between query topics and candidate file topics  
 * - depth_of_best: Best topic match depth in hierarchy
 * - B: Businessness score from semantic cards
 * 
 * Cross-features: type_match ∧ topic_match, alias_resolved ∧ B↑
 * Capped influence: |wi| ≤ 0.4 with isotonic calibration
 * Formula: Δlogit = w1*raptor_max_sim + w2*topic_overlap + w3*B + w4*(type_match∧topic_match)
 */

import { TopicTree, TopicNode } from './topic-tree.js';
import { CardStore, EnhancedSemanticCard } from './card-store.js';
import { SymbolGraph, SymbolNode } from './symbol-graph.js';
import { Candidate, SymbolKind } from '../types/core.js';

export interface RaptorFeatureWeights {
  raptor_max_sim: number;
  topic_overlap: number;
  businessness: number;
  depth_of_best: number;
  // Cross-features
  type_match_and_topic_match: number;
  alias_resolved_and_high_business: number;
}

export interface RaptorFeatureConfig {
  weights: RaptorFeatureWeights;
  weight_cap: number; // |wi| ≤ weight_cap
  businessness_high_threshold: number;
  topic_similarity_threshold: number;
  max_topic_depth: number;
  enable_isotonic_calibration: boolean;
  calibration_points: Array<{feature_value: number; calibrated_score: number}>;
}

export interface RaptorFeatures {
  // Core RAPTOR features
  raptor_max_sim: number;
  topic_overlap: number;
  depth_of_best: number;
  businessness: number;
  
  // Cross-features
  type_match_and_topic_match: number;
  alias_resolved_and_high_business: number;
  
  // Context features for explainability
  topic_matches: TopicFeatureMatch[];
  businessness_components: {
    domain_term_pmi: number;
    resource_counts: number;
    abstraction_level: number;
    user_facing_score: number;
  };
  
  // Metadata
  feature_computation_time_ms: number;
  max_feature_influence: number;
}

export interface TopicFeatureMatch {
  topic_id: string;
  topic_path: string[];
  similarity_score: number;
  depth: number;
  coverage_relevance: number;
}

export interface RaptorScoringResult {
  original_score: number;
  raptor_delta: number;
  final_score: number;
  features: RaptorFeatures;
  feature_contributions: Record<string, number>;
  capped_weights_applied: boolean;
}

export interface QueryContext {
  query: string;
  query_embedding?: Float32Array;
  query_topics: TopicFeatureMatch[];
  nl_symbol_overlap?: string[]; // From NL→Symbol bridge
}

/**
 * Stage-C RAPTOR feature extractor and scorer
 */
export class StageCRaptorFeatures {
  private topicTree?: TopicTree;
  private cardStore?: CardStore;
  private symbolGraph?: SymbolGraph;
  private config: RaptorFeatureConfig;
  
  // Feature computation caches
  private candidateCardCache = new Map<string, EnhancedSemanticCard>();
  private candidateTopicsCache = new Map<string, TopicFeatureMatch[]>();
  private queryContextCache?: QueryContext;

  constructor(config?: Partial<RaptorFeatureConfig>) {
    this.config = {
      weights: {
        raptor_max_sim: 0.35,
        topic_overlap: 0.30,
        businessness: 0.25,
        depth_of_best: -0.15, // Negative: shallower is better
        type_match_and_topic_match: 0.40,
        alias_resolved_and_high_business: 0.30
      },
      weight_cap: 0.4, // |wi| ≤ 0.4
      businessness_high_threshold: 1.0,
      topic_similarity_threshold: 0.5,
      max_topic_depth: 3,
      enable_isotonic_calibration: true,
      calibration_points: [
        { feature_value: 0.0, calibrated_score: 0.0 },
        { feature_value: 0.3, calibrated_score: 0.2 },
        { feature_value: 0.6, calibrated_score: 0.5 },
        { feature_value: 0.8, calibrated_score: 0.8 },
        { feature_value: 1.0, calibrated_score: 1.0 }
      ],
      ...config
    };
    
    // Apply weight capping
    this.applyWeightCapping();
  }

  private applyWeightCapping(): void {
    const weights = this.config.weights;
    for (const [key, value] of Object.entries(weights)) {
      if (Math.abs(value) > this.config.weight_cap) {
        weights[key as keyof RaptorFeatureWeights] = Math.sign(value) * this.config.weight_cap;
      }
    }
  }

  /**
   * Initialize with RAPTOR components
   */
  initialize(topicTree: TopicTree, cardStore: CardStore, symbolGraph: SymbolGraph): void {
    this.topicTree = topicTree;
    this.cardStore = cardStore;
    this.symbolGraph = symbolGraph;
  }

  /**
   * Compute RAPTOR features for a candidate given query context
   */
  async computeRaptorFeatures(
    candidate: Candidate,
    queryContext: QueryContext,
    lspFeatures?: {
      type_match: number;
      alias_resolved: number;
    }
  ): Promise<RaptorFeatures> {
    const startTime = Date.now();
    
    if (!this.topicTree || !this.cardStore || !this.symbolGraph) {
      throw new Error('RAPTOR components not initialized');
    }

    // Cache query context
    this.queryContextCache = queryContext;
    
    // Get enhanced semantic card for candidate
    const card = await this.getCandidateCard(candidate);
    
    // Get topic matches for candidate
    const candidateTopics = await this.getCandidateTopics(candidate, card);
    
    // Compute core RAPTOR features
    const raptor_max_sim = this.computeRaptorMaxSim(candidateTopics, queryContext.query_topics);
    const topic_overlap = this.computeTopicOverlap(candidateTopics, queryContext.query_topics);
    const depth_of_best = this.computeDepthOfBest(candidateTopics);
    const businessness = card?.businessness.B || 0;
    
    // Compute cross-features
    const type_match_and_topic_match = this.computeTypeMatchAndTopicMatch(
      candidate,
      candidateTopics,
      queryContext,
      lspFeatures
    );
    
    const alias_resolved_and_high_business = this.computeAliasResolvedAndHighBusiness(
      candidate,
      businessness,
      lspFeatures
    );
    
    const features: RaptorFeatures = {
      raptor_max_sim,
      topic_overlap,
      depth_of_best,
      businessness,
      type_match_and_topic_match,
      alias_resolved_and_high_business,
      topic_matches: candidateTopics,
      businessness_components: card ? {
        domain_term_pmi: card.businessness.components.domain_term_pmi,
        resource_counts: card.businessness.components.resource_counts,
        abstraction_level: card.businessness.components.abstraction_level,
        user_facing_score: card.businessness.components.user_facing_score
      } : {
        domain_term_pmi: 0,
        resource_counts: 0,
        abstraction_level: 0,
        user_facing_score: 0
      },
      feature_computation_time_ms: Date.now() - startTime,
      max_feature_influence: Math.max(
        Math.abs(raptor_max_sim * this.config.weights.raptor_max_sim),
        Math.abs(topic_overlap * this.config.weights.topic_overlap),
        Math.abs(businessness * this.config.weights.businessness),
        Math.abs(depth_of_best * this.config.weights.depth_of_best)
      )
    };
    
    return features;
  }

  /**
   * Score candidate using RAPTOR features
   */
  scoreCandidate(
    candidate: Candidate,
    features: RaptorFeatures,
    baseScore: number
  ): RaptorScoringResult {
    const weights = this.config.weights;
    
    // Compute individual feature contributions
    const contributions = {
      raptor_max_sim: features.raptor_max_sim * weights.raptor_max_sim,
      topic_overlap: features.topic_overlap * weights.topic_overlap,
      businessness: features.businessness * weights.businessness,
      depth_of_best: features.depth_of_best * weights.depth_of_best,
      type_match_and_topic_match: features.type_match_and_topic_match * weights.type_match_and_topic_match,
      alias_resolved_and_high_business: features.alias_resolved_and_high_business * weights.alias_resolved_and_high_business
    };
    
    // Apply isotonic calibration if enabled
    if (this.config.enable_isotonic_calibration) {
      for (const [key, value] of Object.entries(contributions)) {
        contributions[key as keyof typeof contributions] = this.applyIsotonicCalibration(value);
      }
    }
    
    // Compute total RAPTOR delta
    const raptor_delta = Object.values(contributions).reduce((sum, contrib) => sum + contrib, 0);
    
    // Apply final score
    const final_score = baseScore + raptor_delta;
    
    // Check if weight capping was applied
    const capped_weights_applied = Object.values(weights).some(w => Math.abs(w) === this.config.weight_cap);
    
    return {
      original_score: baseScore,
      raptor_delta,
      final_score,
      features,
      feature_contributions: contributions,
      capped_weights_applied
    };
  }

  private async getCandidateCard(candidate: Candidate): Promise<EnhancedSemanticCard | undefined> {
    const cacheKey = candidate.file_path;
    
    if (this.candidateCardCache.has(cacheKey)) {
      return this.candidateCardCache.get(cacheKey);
    }
    
    // Try to get card by file path or create a basic card
    const card = this.cardStore!.getCard(cacheKey);
    
    if (card) {
      this.candidateCardCache.set(cacheKey, card);
      return card;
    }
    
    // Card not found - candidate may not be in CardStore
    return undefined;
  }

  private async getCandidateTopics(
    candidate: Candidate,
    card?: EnhancedSemanticCard
  ): Promise<TopicFeatureMatch[]> {
    const cacheKey = candidate.file_path;
    
    if (this.candidateTopicsCache.has(cacheKey)) {
      return this.candidateTopicsCache.get(cacheKey)!;
    }
    
    const topics: TopicFeatureMatch[] = [];
    
    if (!card || !card.topic_associations) {
      this.candidateTopicsCache.set(cacheKey, topics);
      return topics;
    }
    
    // Get primary topic
    if (card.topic_associations.primary_topic_id) {
      const primaryTopic = this.topicTree!.getTopic(card.topic_associations.primary_topic_id);
      if (primaryTopic) {
        topics.push({
          topic_id: primaryTopic.id,
          topic_path: this.getTopicPath(primaryTopic),
          similarity_score: 1.0, // Primary topic has max similarity
          depth: primaryTopic.level,
          coverage_relevance: this.computeTopicCoverageRelevance(primaryTopic)
        });
      }
    }
    
    // Get secondary topics
    for (const topicId of card.topic_associations.secondary_topic_ids) {
      const topic = this.topicTree!.getTopic(topicId);
      if (topic) {
        const similarity = card.topic_associations.topic_similarity_scores[topicId] || 0;
        
        topics.push({
          topic_id: topic.id,
          topic_path: this.getTopicPath(topic),
          similarity_score: similarity,
          depth: topic.level,
          coverage_relevance: this.computeTopicCoverageRelevance(topic)
        });
      }
    }
    
    this.candidateTopicsCache.set(cacheKey, topics);
    return topics;
  }

  private getTopicPath(topic: TopicNode): string[] {
    const path: string[] = [topic.id];
    
    let currentId = topic.parent_id;
    while (currentId && currentId !== 'root') {
      const parent = this.topicTree!.getTopic(currentId);
      if (parent) {
        path.unshift(parent.id);
        currentId = parent.parent_id;
      } else {
        break;
      }
    }
    
    return path;
  }

  private computeTopicCoverageRelevance(topic: TopicNode): number {
    // Heuristic: larger, more diverse topics have higher coverage relevance
    const sizeScore = Math.min(topic.coverage.total_cards / 20, 1); // Normalize to 20 cards
    const diversityScore = Object.keys(topic.coverage.card_types).length / 5; // Normalize to 5 types
    
    return (sizeScore + diversityScore) / 2;
  }

  private computeRaptorMaxSim(
    candidateTopics: TopicFeatureMatch[],
    queryTopics: TopicFeatureMatch[]
  ): number {
    if (candidateTopics.length === 0 || queryTopics.length === 0) {
      return 0;
    }
    
    let maxSimilarity = 0;
    
    for (const candidateTopic of candidateTopics) {
      for (const queryTopic of queryTopics) {
        // Check for exact topic match
        if (candidateTopic.topic_id === queryTopic.topic_id) {
          maxSimilarity = Math.max(maxSimilarity, 1.0);
          continue;
        }
        
        // Check for hierarchical relationship
        const hierarchySim = this.computeHierarchicalSimilarity(
          candidateTopic.topic_path,
          queryTopic.topic_path
        );
        
        // Use the minimum of the two similarity scores
        const combinedSim = Math.min(candidateTopic.similarity_score, queryTopic.similarity_score) * hierarchySim;
        maxSimilarity = Math.max(maxSimilarity, combinedSim);
      }
    }
    
    return Math.min(maxSimilarity, 1.0);
  }

  private computeTopicOverlap(
    candidateTopics: TopicFeatureMatch[],
    queryTopics: TopicFeatureMatch[]
  ): number {
    if (candidateTopics.length === 0 || queryTopics.length === 0) {
      return 0;
    }
    
    const candidateTopicIds = new Set(candidateTopics.map(t => t.topic_id));
    const queryTopicIds = new Set(queryTopics.map(t => t.topic_id));
    
    // Direct overlap
    const directOverlap = new Set([...candidateTopicIds].filter(id => queryTopicIds.has(id)));
    
    // Hierarchical overlap (parent-child relationships)
    let hierarchicalOverlap = 0;
    for (const candidateTopic of candidateTopics) {
      for (const queryTopic of queryTopics) {
        if (this.isHierarchicallyRelated(candidateTopic.topic_path, queryTopic.topic_path)) {
          hierarchicalOverlap += 0.5; // Partial credit for hierarchical relationship
        }
      }
    }
    
    const totalOverlap = directOverlap.size + hierarchicalOverlap;
    const maxPossibleOverlap = Math.min(candidateTopics.length, queryTopics.length);
    
    return maxPossibleOverlap > 0 ? totalOverlap / maxPossibleOverlap : 0;
  }

  private computeHierarchicalSimilarity(path1: string[], path2: string[]): number {
    if (path1.length === 0 || path2.length === 0) {
      return 0;
    }
    
    // Find common prefix length
    let commonPrefixLength = 0;
    const minLength = Math.min(path1.length, path2.length);
    
    for (let i = 0; i < minLength; i++) {
      if (path1[i] === path2[i]) {
        commonPrefixLength++;
      } else {
        break;
      }
    }
    
    // Similarity based on common prefix vs path lengths
    const maxLength = Math.max(path1.length, path2.length);
    return commonPrefixLength / maxLength;
  }

  private isHierarchicallyRelated(path1: string[], path2: string[]): boolean {
    // Check if one path is a prefix of the other
    const minLength = Math.min(path1.length, path2.length);
    
    for (let i = 0; i < minLength; i++) {
      if (path1[i] !== path2[i]) {
        return false;
      }
    }
    
    return true; // One is a prefix of the other
  }

  private computeDepthOfBest(candidateTopics: TopicFeatureMatch[]): number {
    if (candidateTopics.length === 0) {
      return this.config.max_topic_depth; // Penalty for no topics
    }
    
    // Find the best topic match by similarity score
    const bestTopic = candidateTopics.reduce((best, current) => 
      current.similarity_score > best.similarity_score ? current : best
    );
    
    // Normalize depth (lower depth is better, so we use negative values)
    return bestTopic.depth / this.config.max_topic_depth;
  }

  private computeTypeMatchAndTopicMatch(
    candidate: Candidate,
    candidateTopics: TopicFeatureMatch[],
    queryContext: QueryContext,
    lspFeatures?: { type_match: number; alias_resolved: number }
  ): number {
    const type_match = lspFeatures?.type_match || 0;
    const topic_match = candidateTopics.length > 0 ? 1 : 0;
    
    // Conjunctive feature: both conditions must be true
    return type_match * topic_match;
  }

  private computeAliasResolvedAndHighBusiness(
    candidate: Candidate,
    businessness: number,
    lspFeatures?: { type_match: number; alias_resolved: number }
  ): number {
    const alias_resolved = lspFeatures?.alias_resolved || 0;
    const high_business = businessness > this.config.businessness_high_threshold ? 1 : 0;
    
    // Conjunctive feature: both conditions must be true
    return alias_resolved * high_business;
  }

  private applyIsotonicCalibration(featureValue: number): number {
    if (!this.config.enable_isotonic_calibration) {
      return featureValue;
    }
    
    const points = this.config.calibration_points;
    
    // Handle boundary cases
    if (featureValue <= points[0].feature_value) {
      return points[0].calibrated_score;
    }
    
    if (featureValue >= points[points.length - 1].feature_value) {
      return points[points.length - 1].calibrated_score;
    }
    
    // Find interpolation points
    for (let i = 0; i < points.length - 1; i++) {
      const p1 = points[i];
      const p2 = points[i + 1];
      
      if (featureValue >= p1.feature_value && featureValue <= p2.feature_value) {
        // Linear interpolation
        const ratio = (featureValue - p1.feature_value) / (p2.feature_value - p1.feature_value);
        return p1.calibrated_score + ratio * (p2.calibrated_score - p1.calibrated_score);
      }
    }
    
    return featureValue; // Fallback
  }

  /**
   * Batch compute features for multiple candidates
   */
  async computeBatchRaptorFeatures(
    candidates: Candidate[],
    queryContext: QueryContext,
    lspFeaturesMap?: Map<string, { type_match: number; alias_resolved: number }>
  ): Promise<Map<string, RaptorFeatures>> {
    const results = new Map<string, RaptorFeatures>();
    
    // Process candidates in parallel batches for efficiency
    const batchSize = 20;
    
    for (let i = 0; i < candidates.length; i += batchSize) {
      const batch = candidates.slice(i, i + batchSize);
      
      const batchPromises = batch.map(async candidate => {
        const candidateKey = `${candidate.file_path}:${candidate.line}:${candidate.col}`;
        const lspFeatures = lspFeaturesMap?.get(candidateKey);
        
        try {
          const features = await this.computeRaptorFeatures(candidate, queryContext, lspFeatures);
          return { candidateKey, features };
        } catch (error) {
          // Return default features on error
          return {
            candidateKey,
            features: this.getDefaultFeatures()
          };
        }
      });
      
      const batchResults = await Promise.all(batchPromises);
      
      for (const { candidateKey, features } of batchResults) {
        results.set(candidateKey, features);
      }
    }
    
    return results;
  }

  private getDefaultFeatures(): RaptorFeatures {
    return {
      raptor_max_sim: 0,
      topic_overlap: 0,
      depth_of_best: this.config.max_topic_depth,
      businessness: 0,
      type_match_and_topic_match: 0,
      alias_resolved_and_high_business: 0,
      topic_matches: [],
      businessness_components: {
        domain_term_pmi: 0,
        resource_counts: 0,
        abstraction_level: 0,
        user_facing_score: 0
      },
      feature_computation_time_ms: 0,
      max_feature_influence: 0
    };
  }

  /**
   * Clear feature computation caches
   */
  clearCaches(): void {
    this.candidateCardCache.clear();
    this.candidateTopicsCache.clear();
    this.queryContextCache = undefined;
  }

  /**
   * Update feature weights with automatic capping
   */
  updateWeights(newWeights: Partial<RaptorFeatureWeights>): void {
    this.config.weights = { ...this.config.weights, ...newWeights };
    this.applyWeightCapping();
  }

  /**
   * Get current configuration
   */
  getConfig(): RaptorFeatureConfig {
    return { ...this.config };
  }

  /**
   * Export feature statistics for analysis
   */
  exportFeatureStats(results: RaptorScoringResult[]): {
    feature_distributions: Record<string, { mean: number; std: number; min: number; max: number }>;
    weight_utilization: Record<string, number>;
    avg_computation_time_ms: number;
    capped_weight_usage_rate: number;
  } {
    if (results.length === 0) {
      return {
        feature_distributions: {},
        weight_utilization: {},
        avg_computation_time_ms: 0,
        capped_weight_usage_rate: 0
      };
    }

    const featureNames = [
      'raptor_max_sim', 'topic_overlap', 'depth_of_best', 'businessness',
      'type_match_and_topic_match', 'alias_resolved_and_high_business'
    ];
    
    const feature_distributions: Record<string, { mean: number; std: number; min: number; max: number }> = {};
    const weight_utilization: Record<string, number> = {};
    
    for (const featureName of featureNames) {
      const values = results.map(r => (r.features as any)[featureName] || 0);
      const contributions = results.map(r => Math.abs(r.feature_contributions[featureName] || 0));
      
      const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
      const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
      const std = Math.sqrt(variance);
      
      feature_distributions[featureName] = {
        mean,
        std,
        min: Math.min(...values),
        max: Math.max(...values)
      };
      
      weight_utilization[featureName] = contributions.reduce((sum, contrib) => sum + contrib, 0) / contributions.length;
    }
    
    const avg_computation_time_ms = results.reduce((sum, r) => 
      sum + r.features.feature_computation_time_ms, 0
    ) / results.length;
    
    const capped_weight_usage_rate = results.filter(r => r.capped_weights_applied).length / results.length;
    
    return {
      feature_distributions,
      weight_utilization,
      avg_computation_time_ms,
      capped_weight_usage_rate
    };
  }
}