/**
 * TopicTree - 2-3 level RAPTOR tree structure with summary embeddings
 * 
 * Implements hierarchical topic clustering for semantic search enhancement.
 * Topics are organized in a tree structure where each node contains:
 * - Summary embeddings of child content
 * - Topic facets (bulletized summaries)
 * - Coverage metrics and age information
 */

import { promises as fs } from 'fs';
import path from 'path';
import { EnhancedSemanticCard } from './card-store.js';

export interface TopicNode {
  id: string;
  level: number; // 0 = root, 1 = topic, 2 = subtopic
  parent_id?: string;
  children_ids: string[];
  
  // Topic content
  summary: string;
  facets: string[]; // Bulletized facets
  keywords: string[]; // Key terms representing this topic
  
  // Vector representations
  e_node: Float32Array; // Summary embedding (384-dim)
  e_keywords: Float32Array; // Keyword-based embedding (256-dim)
  
  // Coverage and freshness
  coverage: TopicCoverage;
  age: TopicAge;
  
  // Associated cards
  card_ids: string[]; // Direct members
  symbol_ids: string[]; // Associated symbols
  
  // Topic metadata
  metadata: {
    created_at: number;
    last_updated: number;
    cluster_method: 'hnsw' | 'kmeans' | 'hierarchical';
    stability_score: number; // 0-1, how stable this clustering is
    split_threshold: number; // When to split this topic
    merge_threshold: number; // When to merge with siblings
  };
}

export interface TopicCoverage {
  total_cards: number;
  card_types: Record<string, number>; // file_type -> count
  symbol_kinds: Record<string, number>; // SymbolKind -> count
  businessness_distribution: {
    high_business: number; // B > 1.0
    medium_business: number; // 0 < B <= 1.0  
    low_business: number; // B <= 0
  };
  quality_distribution: {
    high_quality: number;
    medium_quality: number; 
    low_quality: number;
  };
}

export interface TopicAge {
  oldest_card_age_days: number;
  newest_card_age_days: number;
  avg_card_age_days: number;
  staleness_score: number; // 0-1, higher = more stale
  edit_velocity: number; // Recent edit frequency
}

export interface TopicTree {
  repo_sha: string;
  version: string;
  timestamp: number;
  root_id: string;
  
  // Tree structure
  nodes: Map<string, TopicNode>;
  level_index: Map<number, string[]>; // level -> node_ids
  
  // Global topic statistics
  stats: TopicTreeStats;
  
  // Build configuration
  build_config: TopicTreeBuildConfig;
}

export interface TopicTreeStats {
  total_nodes: number;
  nodes_by_level: Record<number, number>;
  avg_children_per_node: number;
  max_tree_depth: number;
  topic_stability_avg: number;
  coverage_efficiency: number; // How well topics cover the card space
  
  // Quality metrics
  topic_coherence_scores: number[]; // Per-topic coherence
  topic_diversity_score: number; // How diverse topics are
  redundancy_score: number; // How much overlap between topics
}

export interface TopicTreeBuildConfig {
  max_levels: number;
  min_cards_per_topic: number;
  max_cards_per_topic: number;
  similarity_threshold: number;
  stability_threshold: number;
  
  // Clustering parameters
  clustering_method: 'hnsw' | 'kmeans' | 'hierarchical';
  embedding_weights: {
    semantic: number;
    syntax: number;
    context: number;
    businessness: number;
  };
  
  // Quality filters
  min_topic_coherence: number;
  max_topic_redundancy: number;
}

export interface TopicQuery {
  query_embedding?: Float32Array;
  keywords?: string[];
  level_filter?: number[];
  businessness_range?: { min: number; max: number };
  max_results?: number;
  min_similarity?: number;
}

export interface TopicSearchResult {
  topic_id: string;
  similarity_score: number;
  keyword_overlap: number;
  coverage_relevance: number;
  combined_score: number;
  path_to_root: string[]; // Topic hierarchy path
}

/**
 * TopicTree manages hierarchical topic organization for semantic search
 */
export class TopicTree {
  private tree?: TopicTree;
  private storagePath: string;
  private buildConfig: TopicTreeBuildConfig;
  
  // Search indices for efficient querying
  private embeddingIndex?: Float32Array[]; // All topic embeddings
  private topicIdIndex?: string[]; // Corresponding topic IDs
  private keywordIndex?: Map<string, Set<string>>; // keyword -> topic_ids

  constructor(storagePath: string, config?: Partial<TopicTreeBuildConfig>) {
    this.storagePath = storagePath;
    this.buildConfig = {
      max_levels: 3,
      min_cards_per_topic: 5,
      max_cards_per_topic: 50,
      similarity_threshold: 0.7,
      stability_threshold: 0.6,
      clustering_method: 'hnsw',
      embedding_weights: {
        semantic: 0.5,
        syntax: 0.2,
        context: 0.2,
        businessness: 0.1
      },
      min_topic_coherence: 0.5,
      max_topic_redundancy: 0.3,
      ...config
    };
  }

  /**
   * Build topic tree from enhanced semantic cards
   */
  async buildFromCards(
    repoSha: string,
    cards: EnhancedSemanticCard[],
    progressCallback?: (progress: number) => void
  ): Promise<TopicTree> {
    const startTime = Date.now();
    
    // Phase 1: Initial clustering at level 1
    if (progressCallback) progressCallback(0.1);
    const level1Topics = await this.clusterCards(cards, 1);
    
    // Phase 2: Hierarchical refinement to levels 2-3
    if (progressCallback) progressCallback(0.4);
    const allTopics = await this.buildHierarchy(level1Topics, cards);
    
    // Phase 3: Generate topic summaries and embeddings
    if (progressCallback) progressCallback(0.7);
    await this.generateTopicContent(allTopics, cards);
    
    // Phase 4: Build tree structure
    if (progressCallback) progressCallback(0.9);
    const tree = this.assembleTree(repoSha, allTopics);
    
    // Phase 5: Compute statistics and validate
    tree.stats = await this.computeTreeStats(tree);
    
    // Build search indices
    await this.buildSearchIndices(tree);
    
    // Save tree
    await this.saveTree(tree);
    this.tree = tree;
    
    if (progressCallback) progressCallback(1.0);
    
    return tree;
  }

  private async clusterCards(
    cards: EnhancedSemanticCard[],
    level: number
  ): Promise<Map<string, EnhancedSemanticCard[]>> {
    // Use weighted combination of embeddings for clustering
    const features = cards.map(card => this.computeClusteringFeature(card));
    
    // Simple k-means clustering (in real implementation would use HNSW or better)
    const numClusters = Math.min(
      Math.max(2, Math.floor(cards.length / this.buildConfig.max_cards_per_topic)),
      Math.floor(cards.length / this.buildConfig.min_cards_per_topic)
    );
    
    const clusters = new Map<string, EnhancedSemanticCard[]>();
    
    // Initialize cluster centers randomly
    const centers: Float32Array[] = [];
    for (let i = 0; i < numClusters; i++) {
      const randomCard = cards[Math.floor(Math.random() * cards.length)];
      centers.push(new Float32Array(this.computeClusteringFeature(randomCard)));
    }
    
    // K-means iterations
    for (let iter = 0; iter < 10; iter++) {
      // Clear clusters
      for (let i = 0; i < numClusters; i++) {
        clusters.set(`level${level}_topic${i}`, []);
      }
      
      // Assign cards to nearest cluster
      for (let cardIdx = 0; cardIdx < cards.length; cardIdx++) {
        const feature = features[cardIdx];
        let bestCluster = 0;
        let bestDistance = Infinity;
        
        for (let clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
          const distance = this.euclideanDistance(feature, centers[clusterIdx]);
          if (distance < bestDistance) {
            bestDistance = distance;
            bestCluster = clusterIdx;
          }
        }
        
        clusters.get(`level${level}_topic${bestCluster}`)!.push(cards[cardIdx]);
      }
      
      // Update cluster centers
      for (let clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        const clusterCards = clusters.get(`level${level}_topic${clusterIdx}`)!;
        if (clusterCards.length > 0) {
          const newCenter = new Float32Array(features[0].length);
          for (const card of clusterCards) {
            const feature = this.computeClusteringFeature(card);
            for (let i = 0; i < feature.length; i++) {
              newCenter[i] += feature[i] / clusterCards.length;
            }
          }
          centers[clusterIdx] = newCenter;
        }
      }
    }
    
    // Filter out empty clusters
    const filteredClusters = new Map<string, EnhancedSemanticCard[]>();
    for (const [topicId, clusterCards] of clusters) {
      if (clusterCards.length >= this.buildConfig.min_cards_per_topic) {
        filteredClusters.set(topicId, clusterCards);
      }
    }
    
    return filteredClusters;
  }

  private computeClusteringFeature(card: EnhancedSemanticCard): Float32Array {
    const w = this.buildConfig.embedding_weights;
    const feature = new Float32Array(
      card.e_sem.length + card.e_syntax.length + card.e_context.length + 1
    );
    
    let offset = 0;
    
    // Semantic embedding (weighted)
    for (let i = 0; i < card.e_sem.length; i++) {
      feature[offset + i] = card.e_sem[i] * w.semantic;
    }
    offset += card.e_sem.length;
    
    // Syntax embedding (weighted)
    for (let i = 0; i < card.e_syntax.length; i++) {
      feature[offset + i] = card.e_syntax[i] * w.syntax;
    }
    offset += card.e_syntax.length;
    
    // Context embedding (weighted)
    for (let i = 0; i < card.e_context.length; i++) {
      feature[offset + i] = card.e_context[i] * w.context;
    }
    offset += card.e_context.length;
    
    // Businessness score (weighted)
    feature[offset] = card.businessness.B * w.businessness;
    
    return feature;
  }

  private euclideanDistance(a: Float32Array, b: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const diff = a[i] - b[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }

  private async buildHierarchy(
    level1Topics: Map<string, EnhancedSemanticCard[]>,
    allCards: EnhancedSemanticCard[]
  ): Promise<Map<string, EnhancedSemanticCard[]>> {
    const allTopics = new Map(level1Topics);
    
    // Build level 2 by subclustering large level 1 topics
    for (const [topicId, cards] of level1Topics) {
      if (cards.length > this.buildConfig.max_cards_per_topic) {
        const subtopics = await this.clusterCards(cards, 2);
        
        // Add subtopics with hierarchical naming
        for (const [subTopicId, subCards] of subtopics) {
          const hierarchicalId = `${topicId}_${subTopicId}`;
          allTopics.set(hierarchicalId, subCards);
        }
        
        // Remove the large parent topic
        allTopics.delete(topicId);
      }
    }
    
    // Build level 3 for very large level 2 topics
    const level2Topics = new Map<string, EnhancedSemanticCard[]>();
    for (const [topicId, cards] of allTopics) {
      if (topicId.includes('level2')) {
        level2Topics.set(topicId, cards);
      }
    }
    
    for (const [topicId, cards] of level2Topics) {
      if (cards.length > this.buildConfig.max_cards_per_topic) {
        const subtopics = await this.clusterCards(cards, 3);
        
        for (const [subTopicId, subCards] of subtopics) {
          const hierarchicalId = `${topicId}_${subTopicId}`;
          allTopics.set(hierarchicalId, subCards);
        }
        
        allTopics.delete(topicId);
      }
    }
    
    return allTopics;
  }

  private async generateTopicContent(
    topics: Map<string, EnhancedSemanticCard[]>,
    allCards: EnhancedSemanticCard[]
  ): Promise<void> {
    for (const [topicId, cards] of topics) {
      // Generate summary from card content
      const summary = this.generateTopicSummary(cards);
      
      // Extract key facets
      const facets = this.generateTopicFacets(cards);
      
      // Extract keywords
      const keywords = this.generateTopicKeywords(cards);
      
      // Store in temporary map for tree assembly
      (topics as any)[topicId] = {
        cards,
        summary,
        facets,
        keywords
      };
    }
  }

  private generateTopicSummary(cards: EnhancedSemanticCard[]): string {
    // Aggregate the most common themes
    const allRoles = cards.flatMap(card => card.roles);
    const allResources = cards.flatMap(card => [
      ...card.resources.routes,
      ...card.resources.sql,
      ...card.resources.topics,
      ...card.resources.buckets,
      ...card.resources.featureFlags
    ]);
    const allDomainTokens = cards.flatMap(card => card.domainTokens);
    
    const roleFreq = this.getTopFrequent(allRoles, 3);
    const resourceFreq = this.getTopFrequent(allResources, 3);
    const domainFreq = this.getTopFrequent(allDomainTokens, 5);
    
    return `Topic covering ${cards.length} files focusing on: ${roleFreq.join(', ')}. ` +
           `Common resources: ${resourceFreq.join(', ')}. ` +
           `Key domains: ${domainFreq.join(', ')}.`;
  }

  private generateTopicFacets(cards: EnhancedSemanticCard[]): string[] {
    const facets: string[] = [];
    
    // Facet 1: Primary functionality
    const topRoles = this.getTopFrequent(cards.flatMap(c => c.roles), 3);
    if (topRoles.length > 0) {
      facets.push(`• Primary functions: ${topRoles.join(', ')}`);
    }
    
    // Facet 2: Resource usage
    const topResources = this.getTopFrequent(cards.flatMap(c => [
      ...c.resources.routes,
      ...c.resources.sql,
      ...c.resources.topics,
      ...c.resources.buckets,
      ...c.resources.featureFlags
    ]), 3);
    if (topResources.length > 0) {
      facets.push(`• Key resources: ${topResources.join(', ')}`);
    }
    
    // Facet 3: Technical characteristics
    const topShapes = this.getTopFrequent(cards.flatMap(c => [...c.shapes.typeNames, ...c.shapes.jsonKeys]), 3);
    if (topShapes.length > 0) {
      facets.push(`• Technical patterns: ${topShapes.join(', ')}`);
    }
    
    // Facet 4: Business context
    const businessCards = cards.filter(c => c.businessness.B > 0.5);
    if (businessCards.length > 0) {
      facets.push(`• Business relevance: ${businessCards.length}/${cards.length} files with high business logic`);
    }
    
    return facets;
  }

  private generateTopicKeywords(cards: EnhancedSemanticCard[]): string[] {
    const allTerms = [
      ...cards.flatMap(c => c.roles),
      ...cards.flatMap(c => [
        ...c.resources.routes,
        ...c.resources.sql,
        ...c.resources.topics,
        ...c.resources.buckets,
        ...c.resources.featureFlags
      ]),
      ...cards.flatMap(c => c.domainTokens)
    ];
    
    return this.getTopFrequent(allTerms, 10);
  }

  private getTopFrequent(items: string[], topN: number): string[] {
    const freq = new Map<string, number>();
    
    for (const item of items) {
      if (item && item.length > 2) { // Filter short terms
        freq.set(item, (freq.get(item) || 0) + 1);
      }
    }
    
    return Array.from(freq.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, topN)
      .map(([term]) => term);
  }

  private assembleTree(repoSha: string, topics: Map<string, any>): any {
    const nodes = new Map<string, TopicNode>();
    const levelIndex = new Map<number, string[]>();
    
    // Create root node
    const rootId = 'root';
    const rootNode: TopicNode = {
      id: rootId,
      level: 0,
      children_ids: [],
      summary: 'Root topic containing all code structure',
      facets: ['• Complete codebase organization'],
      keywords: ['code', 'structure', 'organization'],
      e_node: new Float32Array(384), // Will be computed as average
      e_keywords: new Float32Array(256),
      coverage: this.computeRootCoverage(topics),
      age: this.computeRootAge(topics),
      card_ids: [],
      symbol_ids: [],
      metadata: {
        created_at: Date.now(),
        last_updated: Date.now(),
        cluster_method: this.buildConfig.clustering_method,
        stability_score: 1.0,
        split_threshold: Infinity,
        merge_threshold: 0
      }
    };
    
    nodes.set(rootId, rootNode);
    levelIndex.set(0, [rootId]);
    
    // Process each topic cluster
    for (const [topicId, topicData] of topics) {
      const level = this.getTopicLevel(topicId);
      const parentId = this.getParentTopicId(topicId);
      
      const node: TopicNode = {
        id: topicId,
        level,
        parent_id: parentId || rootId,
        children_ids: [],
        summary: topicData.summary,
        facets: topicData.facets,
        keywords: topicData.keywords,
        e_node: this.computeTopicEmbedding(topicData.cards),
        e_keywords: this.computeKeywordEmbedding(topicData.keywords),
        coverage: this.computeTopicCoverage(topicData.cards),
        age: this.computeTopicAge(topicData.cards),
        card_ids: topicData.cards.map((card: EnhancedSemanticCard) => card.file_id),
        symbol_ids: topicData.cards.flatMap((card: EnhancedSemanticCard) => 
          card.symbols.map(s => s.id)
        ),
        metadata: {
          created_at: Date.now(),
          last_updated: Date.now(),
          cluster_method: this.buildConfig.clustering_method,
          stability_score: this.computeStability(topicData.cards),
          split_threshold: this.buildConfig.max_cards_per_topic,
          merge_threshold: this.buildConfig.min_cards_per_topic
        }
      };
      
      nodes.set(topicId, node);
      
      // Update level index
      if (!levelIndex.has(level)) {
        levelIndex.set(level, []);
      }
      levelIndex.get(level)!.push(topicId);
      
      // Link to parent
      const parent = nodes.get(node.parent_id!);
      if (parent) {
        parent.children_ids.push(topicId);
      }
    }
    
    return {
      repo_sha: repoSha,
      version: '1.0.0',
      timestamp: Date.now(),
      root_id: rootId,
      nodes,
      level_index: levelIndex,
      stats: {} as TopicTreeStats, // Will be computed later
      build_config: this.buildConfig
    };
  }

  private getTopicLevel(topicId: string): number {
    const matches = topicId.match(/level(\d+)/g);
    if (!matches) return 1;
    
    // Count the number of level indicators
    return matches.length;
  }

  private getParentTopicId(topicId: string): string | undefined {
    const parts = topicId.split('_');
    if (parts.length <= 1) return undefined;
    
    // Remove the last part to get parent
    return parts.slice(0, -1).join('_');
  }

  private computeTopicEmbedding(cards: EnhancedSemanticCard[]): Float32Array {
    if (cards.length === 0) {
      return new Float32Array(384);
    }
    
    // Average semantic embeddings
    const avgEmbedding = new Float32Array(384);
    for (const card of cards) {
      for (let i = 0; i < 384; i++) {
        avgEmbedding[i] += card.e_sem[i] / cards.length;
      }
    }
    
    // Normalize
    const norm = Math.sqrt(avgEmbedding.reduce((sum, val) => sum + val * val, 0));
    if (norm > 0) {
      for (let i = 0; i < avgEmbedding.length; i++) {
        avgEmbedding[i] /= norm;
      }
    }
    
    return avgEmbedding;
  }

  private computeKeywordEmbedding(keywords: string[]): Float32Array {
    // Simple hash-based embedding for keywords
    const embedding = new Float32Array(256);
    
    for (const keyword of keywords) {
      const hash = this.simpleHash(keyword);
      embedding[hash % embedding.length] += 1.0;
    }
    
    // Normalize
    const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    if (norm > 0) {
      for (let i = 0; i < embedding.length; i++) {
        embedding[i] /= norm;
      }
    }
    
    return embedding;
  }

  private computeTopicCoverage(cards: EnhancedSemanticCard[]): TopicCoverage {
    const cardTypes: Record<string, number> = {};
    const symbolKinds: Record<string, number> = {};
    
    let highBusiness = 0, mediumBusiness = 0, lowBusiness = 0;
    let highQuality = 0, mediumQuality = 0, lowQuality = 0;
    
    for (const card of cards) {
      // File types
      const fileType = path.extname(card.file_path).slice(1) || 'unknown';
      cardTypes[fileType] = (cardTypes[fileType] || 0) + 1;
      
      // Symbol kinds
      for (const symbol of card.symbols) {
        symbolKinds[symbol.kind] = (symbolKinds[symbol.kind] || 0) + 1;
      }
      
      // Businessness distribution
      if (card.businessness.B > 1.0) highBusiness++;
      else if (card.businessness.B > 0) mediumBusiness++;
      else lowBusiness++;
      
      // Quality distribution
      const avgQuality = (
        card.extraction_metadata.confidence_scores.semantic_quality +
        card.extraction_metadata.confidence_scores.businessness_confidence +
        card.extraction_metadata.confidence_scores.embedding_quality
      ) / 3;
      
      if (avgQuality > 0.8) highQuality++;
      else if (avgQuality > 0.5) mediumQuality++;
      else lowQuality++;
    }
    
    return {
      total_cards: cards.length,
      card_types: cardTypes,
      symbol_kinds: symbolKinds,
      businessness_distribution: {
        high_business: highBusiness,
        medium_business: mediumBusiness,
        low_business: lowBusiness
      },
      quality_distribution: {
        high_quality: highQuality,
        medium_quality: mediumQuality,
        low_quality: lowQuality
      }
    };
  }

  private computeTopicAge(cards: EnhancedSemanticCard[]): TopicAge {
    if (cards.length === 0) {
      return {
        oldest_card_age_days: 0,
        newest_card_age_days: 0,
        avg_card_age_days: 0,
        staleness_score: 0,
        edit_velocity: 0
      };
    }
    
    const now = Date.now();
    const ages = cards.map(card => 
      (now - card.extraction_metadata.extracted_at) / (1000 * 60 * 60 * 24)
    );
    
    ages.sort((a, b) => a - b);
    
    return {
      oldest_card_age_days: ages[ages.length - 1],
      newest_card_age_days: ages[0],
      avg_card_age_days: ages.reduce((sum, age) => sum + age, 0) / ages.length,
      staleness_score: Math.min(ages[ages.length - 1] / 30, 1), // 30 days = fully stale
      edit_velocity: Math.max(0, 1 - ages[0] / 7) // Recent edits boost velocity
    };
  }

  private computeRootCoverage(topics: Map<string, any>): TopicCoverage {
    let totalCards = 0;
    const cardTypes: Record<string, number> = {};
    const symbolKinds: Record<string, number> = {};
    
    for (const topicData of topics.values()) {
      const coverage = this.computeTopicCoverage(topicData.cards);
      totalCards += coverage.total_cards;
      
      for (const [type, count] of Object.entries(coverage.card_types)) {
        cardTypes[type] = (cardTypes[type] || 0) + count;
      }
      
      for (const [kind, count] of Object.entries(coverage.symbol_kinds)) {
        symbolKinds[kind] = (symbolKinds[kind] || 0) + count;
      }
    }
    
    return {
      total_cards: totalCards,
      card_types: cardTypes,
      symbol_kinds: symbolKinds,
      businessness_distribution: { high_business: 0, medium_business: 0, low_business: 0 },
      quality_distribution: { high_quality: 0, medium_quality: 0, low_quality: 0 }
    };
  }

  private computeRootAge(topics: Map<string, any>): TopicAge {
    const allAges = Array.from(topics.values()).map(topicData => 
      this.computeTopicAge(topicData.cards)
    );
    
    if (allAges.length === 0) {
      return { oldest_card_age_days: 0, newest_card_age_days: 0, avg_card_age_days: 0, staleness_score: 0, edit_velocity: 0 };
    }
    
    return {
      oldest_card_age_days: Math.max(...allAges.map(a => a.oldest_card_age_days)),
      newest_card_age_days: Math.min(...allAges.map(a => a.newest_card_age_days)),
      avg_card_age_days: allAges.reduce((sum, a) => sum + a.avg_card_age_days, 0) / allAges.length,
      staleness_score: allAges.reduce((sum, a) => sum + a.staleness_score, 0) / allAges.length,
      edit_velocity: allAges.reduce((sum, a) => sum + a.edit_velocity, 0) / allAges.length
    };
  }

  private computeStability(cards: EnhancedSemanticCard[]): number {
    // Simple stability heuristic based on card consistency
    if (cards.length < 2) return 1.0;
    
    const avgBusinessness = cards.reduce((sum, card) => sum + card.businessness.B, 0) / cards.length;
    const businessnessVariance = cards.reduce((sum, card) => 
      sum + Math.pow(card.businessness.B - avgBusinessness, 2), 0
    ) / cards.length;
    
    // Lower variance = higher stability
    return Math.max(0, 1 - businessnessVariance);
  }

  private async computeTreeStats(tree: TopicTree): Promise<TopicTreeStats> {
    const nodes = Array.from(tree.nodes.values());
    const levelCounts: Record<number, number> = {};
    
    for (const node of nodes) {
      levelCounts[node.level] = (levelCounts[node.level] || 0) + 1;
    }
    
    const childrenCounts = nodes
      .filter(n => n.level > 0)
      .map(n => n.children_ids.length);
    
    const avgChildren = childrenCounts.length > 0 
      ? childrenCounts.reduce((sum, count) => sum + count, 0) / childrenCounts.length
      : 0;
    
    const coherenceScores = nodes.map(n => n.metadata.stability_score);
    const avgCoherence = coherenceScores.reduce((sum, score) => sum + score, 0) / coherenceScores.length;
    
    return {
      total_nodes: nodes.length,
      nodes_by_level: levelCounts,
      avg_children_per_node: avgChildren,
      max_tree_depth: Math.max(...nodes.map(n => n.level)),
      topic_stability_avg: avgCoherence,
      coverage_efficiency: this.computeCoverageEfficiency(tree),
      topic_coherence_scores: coherenceScores,
      topic_diversity_score: this.computeTopicDiversity(tree),
      redundancy_score: this.computeRedundancy(tree)
    };
  }

  private computeCoverageEfficiency(tree: TopicTree): number {
    // Measure how well topics partition the space without overlap
    const leafNodes = Array.from(tree.nodes.values()).filter(n => n.children_ids.length === 0);
    const totalCards = leafNodes.reduce((sum, node) => sum + node.coverage.total_cards, 0);
    const avgCardsPerTopic = totalCards / Math.max(1, leafNodes.length);
    
    // Efficiency is higher when topics have similar sizes
    const variance = leafNodes.reduce((sum, node) => 
      sum + Math.pow(node.coverage.total_cards - avgCardsPerTopic, 2), 0
    ) / Math.max(1, leafNodes.length);
    
    return Math.max(0, 1 - variance / (avgCardsPerTopic * avgCardsPerTopic));
  }

  private computeTopicDiversity(tree: TopicTree): number {
    // Measure how different topic embeddings are from each other
    const embeddings = Array.from(tree.nodes.values())
      .filter(n => n.level > 0)
      .map(n => n.e_node);
    
    if (embeddings.length < 2) return 1.0;
    
    let totalSimilarity = 0;
    let comparisons = 0;
    
    for (let i = 0; i < embeddings.length; i++) {
      for (let j = i + 1; j < embeddings.length; j++) {
        totalSimilarity += this.cosineSimilarity(embeddings[i], embeddings[j]);
        comparisons++;
      }
    }
    
    const avgSimilarity = totalSimilarity / comparisons;
    return Math.max(0, 1 - avgSimilarity); // Lower similarity = higher diversity
  }

  private computeRedundancy(tree: TopicTree): number {
    // Measure keyword overlap between sibling topics
    const levelNodes = tree.level_index.get(1) || [];
    
    if (levelNodes.length < 2) return 0;
    
    let totalOverlap = 0;
    let comparisons = 0;
    
    for (let i = 0; i < levelNodes.length; i++) {
      for (let j = i + 1; j < levelNodes.length; j++) {
        const node1 = tree.nodes.get(levelNodes[i])!;
        const node2 = tree.nodes.get(levelNodes[j])!;
        
        const keywords1 = new Set(node1.keywords);
        const keywords2 = new Set(node2.keywords);
        const intersection = new Set([...keywords1].filter(k => keywords2.has(k)));
        
        const overlap = intersection.size / Math.max(keywords1.size, keywords2.size);
        totalOverlap += overlap;
        comparisons++;
      }
    }
    
    return totalOverlap / Math.max(1, comparisons);
  }

  private async buildSearchIndices(tree: TopicTree): Promise<void> {
    const nodes = Array.from(tree.nodes.values()).filter(n => n.level > 0);
    
    this.embeddingIndex = nodes.map(n => n.e_node);
    this.topicIdIndex = nodes.map(n => n.id);
    
    this.keywordIndex = new Map();
    for (const node of nodes) {
      for (const keyword of node.keywords) {
        if (!this.keywordIndex.has(keyword)) {
          this.keywordIndex.set(keyword, new Set());
        }
        this.keywordIndex.get(keyword)!.add(node.id);
      }
    }
  }

  /**
   * Search topics based on embedding similarity and keyword matching
   */
  async searchTopics(query: TopicQuery): Promise<TopicSearchResult[]> {
    if (!this.tree || !this.embeddingIndex || !this.topicIdIndex) {
      return [];
    }

    const results: TopicSearchResult[] = [];
    const maxResults = query.max_results || 10;
    const minSimilarity = query.min_similarity || 0.3;

    for (let i = 0; i < this.topicIdIndex.length; i++) {
      const topicId = this.topicIdIndex[i];
      const node = this.tree.nodes.get(topicId)!;
      
      // Level filter
      if (query.level_filter && !query.level_filter.includes(node.level)) {
        continue;
      }
      
      // Businessness filter
      if (query.businessness_range) {
        const avgBusinessness = this.computeAvgBusinessness(node);
        if (avgBusinessness < query.businessness_range.min || 
            avgBusinessness > query.businessness_range.max) {
          continue;
        }
      }

      let similarityScore = 0;
      let keywordOverlap = 0;
      
      // Embedding similarity
      if (query.query_embedding) {
        similarityScore = this.cosineSimilarity(query.query_embedding, this.embeddingIndex[i]);
        if (similarityScore < minSimilarity) continue;
      }
      
      // Keyword overlap
      if (query.keywords) {
        const queryKeywords = new Set(query.keywords);
        const nodeKeywords = new Set(node.keywords);
        const intersection = new Set([...queryKeywords].filter(k => nodeKeywords.has(k)));
        keywordOverlap = intersection.size / Math.max(1, queryKeywords.size);
      }
      
      // Coverage relevance (how well this topic covers the query space)
      const coverageRelevance = this.computeCoverageRelevance(node, query);
      
      // Combined score
      const combinedScore = 0.5 * similarityScore + 0.3 * keywordOverlap + 0.2 * coverageRelevance;
      
      results.push({
        topic_id: topicId,
        similarity_score: similarityScore,
        keyword_overlap: keywordOverlap,
        coverage_relevance: coverageRelevance,
        combined_score: combinedScore,
        path_to_root: this.getPathToRoot(topicId)
      });
    }
    
    // Sort by combined score
    results.sort((a, b) => b.combined_score - a.combined_score);
    
    return results.slice(0, maxResults);
  }

  private computeAvgBusinessness(node: TopicNode): number {
    const dist = node.coverage.businessness_distribution;
    const total = dist.high_business + dist.medium_business + dist.low_business;
    
    if (total === 0) return 0;
    
    // Weighted average: high=2, medium=1, low=0
    return (dist.high_business * 2 + dist.medium_business * 1) / total;
  }

  private computeCoverageRelevance(node: TopicNode, query: TopicQuery): number {
    // Simple heuristic based on topic size and quality
    const sizeScore = Math.min(node.coverage.total_cards / 20, 1); // Normalize to 20 cards
    const qualityScore = node.coverage.quality_distribution.high_quality / 
                        Math.max(1, node.coverage.total_cards);
    
    return 0.7 * sizeScore + 0.3 * qualityScore;
  }

  private getPathToRoot(topicId: string): string[] {
    if (!this.tree) return [];
    
    const path: string[] = [];
    let currentId: string | undefined = topicId;
    
    while (currentId && currentId !== this.tree.root_id) {
      path.unshift(currentId);
      const node = this.tree.nodes.get(currentId);
      currentId = node?.parent_id;
    }
    
    if (currentId === this.tree.root_id) {
      path.unshift(this.tree.root_id);
    }
    
    return path;
  }

  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  private simpleHash(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash);
  }

  // Storage operations
  private async saveTree(tree: TopicTree): Promise<void> {
    const filePath = path.join(this.storagePath, `TopicTree-${tree.repo_sha}.json`);
    
    // Convert to serializable format
    const serializable = {
      ...tree,
      nodes: Object.fromEntries(
        Array.from(tree.nodes.entries()).map(([k, v]) => [k, {
          ...v,
          e_node: Array.from(v.e_node),
          e_keywords: Array.from(v.e_keywords)
        }])
      ),
      level_index: Object.fromEntries(tree.level_index)
    };
    
    await fs.writeFile(filePath, JSON.stringify(serializable, null, 2));
  }

  async loadTree(repoSha: string): Promise<TopicTree> {
    const filePath = path.join(this.storagePath, `TopicTree-${repoSha}.json`);
    
    try {
      const content = await fs.readFile(filePath, 'utf-8');
      const data = JSON.parse(content);
      
      // Convert back to proper types
      const tree: TopicTree = {
        ...data,
        nodes: new Map(
          Object.entries(data.nodes).map(([k, v]: [string, any]) => [k, {
            ...v,
            e_node: new Float32Array(v.e_node),
            e_keywords: new Float32Array(v.e_keywords)
          }])
        ),
        level_index: new Map(Object.entries(data.level_index))
      };
      
      this.tree = tree;
      await this.buildSearchIndices(tree);
      
      return tree;
      
    } catch (error) {
      throw new Error(`Failed to load TopicTree: ${error}`);
    }
  }

  // Public access methods
  getTree(): TopicTree | undefined {
    return this.tree;
  }

  getTopic(topicId: string): TopicNode | undefined {
    return this.tree?.nodes.get(topicId);
  }

  getTopicsAtLevel(level: number): TopicNode[] {
    if (!this.tree) return [];
    const topicIds = this.tree.level_index.get(level) || [];
    return topicIds.map(id => this.tree!.nodes.get(id)!).filter(Boolean);
  }

  getStats(): TopicTreeStats | undefined {
    return this.tree?.stats;
  }

  // Topic staleness and pressure computation for metrics
  computeTopicStaleness(): Record<string, number> {
    if (!this.tree) return {};
    
    const staleness: Record<string, number> = {};
    
    for (const [topicId, node] of this.tree.nodes) {
      staleness[topicId] = node.age.staleness_score;
    }
    
    return staleness;
  }

  computeTopicPressure(): Record<string, number> {
    if (!this.tree) return {};
    
    const pressure: Record<string, number> = {};
    
    for (const [topicId, node] of this.tree.nodes) {
      // Pressure increases with:
      // 1. High staleness
      // 2. Low stability 
      // 3. Size approaching split threshold
      
      const sizePressure = node.coverage.total_cards / node.metadata.split_threshold;
      const stalePressure = node.age.staleness_score;
      const stabilityPressure = 1 - node.metadata.stability_score;
      
      pressure[topicId] = 0.4 * sizePressure + 0.3 * stalePressure + 0.3 * stabilityPressure;
    }
    
    return pressure;
  }
}