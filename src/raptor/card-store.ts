/**
 * CardStore - Enhanced semantic cards with embeddings and businessness scoring
 * 
 * Extends the basic SemanticCard to include vector embeddings, businessness metrics,
 * and efficient querying capabilities for the RAPTOR system.
 */

// Using JSON as fallback for msgpack-lite
const msgpack = {
  encode: (data: any): Buffer => Buffer.from(JSON.stringify(data), 'utf-8'),
  decode: (buffer: Buffer): any => JSON.parse(buffer.toString('utf-8'))
};
import { promises as fs } from 'fs';
import path from 'path';
import { SemanticCard } from './semantic-card.js';
import { SymbolKind } from '../types/core.js';

export interface EnhancedSemanticCard extends SemanticCard {
  // Vector embeddings
  e_sem: Float32Array; // Semantic embedding (384-dim)
  e_syntax: Float32Array; // Syntax embedding (128-dim)  
  e_context: Float32Array; // Context embedding (256-dim)
  
  // Businessness scoring components
  businessness: {
    B: number; // Final businessness score (z-scored)
    components: {
      domain_term_pmi: number; // PMI with domain terms
      resource_counts: number; // APIs, DB calls, external services
      abstraction_level: number; // High-level vs utility code
      user_facing_score: number; // UI/UX related components
      complexity_score: number; // Cyclomatic complexity normalized
    };
    raw_features: {
      api_call_count: number;
      db_interaction_count: number;
      ui_component_count: number;
      business_logic_keywords: string[];
      import_diversity: number;
      method_complexity: number[];
    };
  };
  
  // Enhanced metadata
  extraction_metadata: {
    extracted_at: number;
    extraction_version: string;
    confidence_scores: {
      semantic_quality: number; // 0-1
      businessness_confidence: number; // 0-1
      embedding_quality: number; // 0-1
    };
    processing_stats: {
      tokens_processed: number;
      embedding_time_ms: number;
      businessness_time_ms: number;
    };
  };
  
  // Topic association
  topic_associations: {
    primary_topic_id?: string;
    secondary_topic_ids: string[];
    topic_similarity_scores: Record<string, number>;
    topic_coverage: number; // 0-1, how much of symbol is covered by topics
  };
}

export interface CardStoreSnapshot {
  repo_sha: string;
  version: string;
  timestamp: number;
  cards: Map<string, EnhancedSemanticCard>; // file_id -> card
  symbol_cards: Map<string, EnhancedSemanticCard>; // symbol_id -> card
  businessness_stats: BusinessnessStats;
  embedding_stats: EmbeddingStats;
  coverage_metrics: CoverageMetrics;
  metadata: {
    total_cards: number;
    avg_businessness: number;
    embedding_dimensions: {
      semantic: number;
      syntax: number;
      context: number;
    };
    build_duration_ms: number;
  };
}

export interface BusinessnessStats {
  global_stats: {
    mean_B: number;
    std_B: number;
    min_B: number;
    max_B: number;
    percentiles: Record<string, number>; // P25, P50, P75, P90, P95
  };
  component_distributions: {
    domain_term_pmi: Distribution;
    resource_counts: Distribution;
    abstraction_level: Distribution;
    user_facing_score: Distribution;
    complexity_score: Distribution;
  };
  domain_terms: Map<string, number>; // term -> global frequency
  business_logic_patterns: Map<string, number>; // pattern -> count
}

export interface Distribution {
  mean: number;
  std: number;
  min: number;
  max: number;
  histogram: Array<{bin: number, count: number}>;
}

export interface EmbeddingStats {
  semantic_cluster_info: {
    num_clusters: number;
    cluster_centers: Float32Array[];
    cluster_assignments: Map<string, number>; // card_id -> cluster_id
    intra_cluster_similarity: number[];
  };
  embedding_quality: {
    avg_norm: number;
    dimensionality_scores: number[];
    outlier_detection: {
      outlier_cards: string[];
      outlier_threshold: number;
    };
  };
}

export interface CoverageMetrics {
  file_coverage: {
    files_with_cards: number;
    total_files: number;
    coverage_ratio: number;
  };
  symbol_coverage: {
    symbols_with_cards: number;
    total_symbols: number;
    coverage_by_kind: Record<SymbolKind, number>;
  };
  quality_distribution: {
    high_quality_cards: number; // confidence > 0.8
    medium_quality_cards: number; // 0.5 < confidence <= 0.8
    low_quality_cards: number; // confidence <= 0.5
  };
}

export interface CardStoreQuery {
  semantic_similarity?: {
    query_embedding: Float32Array;
    min_similarity: number;
    max_results: number;
  };
  businessness_range?: {
    min_B: number;
    max_B: number;
  };
  file_filters?: {
    include_patterns: string[];
    exclude_patterns: string[];
  };
  symbol_kind_filter?: SymbolKind[];
  topic_filter?: {
    topic_ids: string[];
    min_topic_similarity: number;
  };
}

export interface QueryResult {
  cards: EnhancedSemanticCard[];
  scores: number[];
  total_matches: number;
  query_time_ms: number;
}

/**
 * CardStore manages enhanced semantic cards with vector embeddings
 * and provides efficient similarity search and businessness analysis
 */
export class CardStore {
  private snapshot?: CardStoreSnapshot;
  private storagePath: string;
  private embeddingDimensions: { semantic: number; syntax: number; context: number };
  
  // In-memory indices for fast querying
  private semanticIndex?: Float32Array[]; // All semantic embeddings
  private cardIdIndex?: string[]; // Corresponding card IDs
  private businessnessIndex?: Map<string, number>; // card_id -> B score

  constructor(storagePath: string) {
    this.storagePath = storagePath;
    this.embeddingDimensions = {
      semantic: 384,  // sentence-transformer default
      syntax: 128,    // Syntax-specific embedding
      context: 256    // Context/usage embedding
    };
  }

  /**
   * Build card store from semantic cards and symbol information
   */
  async buildFromCards(
    repoSha: string,
    baseCards: SemanticCard[],
    symbolEmbeddings: Map<string, Float32Array>,
    progressCallback?: (progress: number) => void
  ): Promise<CardStoreSnapshot> {
    const startTime = Date.now();
    
    const cards = new Map<string, EnhancedSemanticCard>();
    const symbolCards = new Map<string, EnhancedSemanticCard>();
    
    // Phase 1: Enhance base cards with embeddings and businessness
    for (let i = 0; i < baseCards.length; i++) {
      const baseCard = baseCards[i];
      const enhancedCard = await this.enhanceCard(baseCard, symbolEmbeddings);
      
      cards.set(enhancedCard.file_id, enhancedCard);
      
      // Create symbol-specific cards
      for (const symbol of enhancedCard.symbols) {
        const symbolCard = this.createSymbolCard(enhancedCard, symbol);
        symbolCards.set(symbol.id, symbolCard);
      }
      
      if (progressCallback) {
        progressCallback((i / baseCards.length) * 0.6);
      }
    }

    // Phase 2: Compute global businessness statistics
    const businessnessStats = this.computeBusinessnessStats(cards);
    
    // Phase 3: Normalize businessness scores using global stats
    this.normalizeBusinessnessScores(cards, businessnessStats);
    this.normalizeBusinessnessScores(symbolCards, businessnessStats);
    
    // Phase 4: Compute embedding statistics and clustering
    const embeddingStats = await this.computeEmbeddingStats(cards);
    
    // Phase 5: Compute coverage metrics
    const coverageMetrics = this.computeCoverageMetrics(cards, symbolCards);

    const snapshot: CardStoreSnapshot = {
      repo_sha: repoSha,
      version: '1.0.0',
      timestamp: Date.now(),
      cards,
      symbol_cards: symbolCards,
      businessness_stats: businessnessStats,
      embedding_stats: embeddingStats,
      coverage_metrics: coverageMetrics,
      metadata: {
        total_cards: cards.size + symbolCards.size,
        avg_businessness: businessnessStats.global_stats.mean_B,
        embedding_dimensions: this.embeddingDimensions,
        build_duration_ms: Date.now() - startTime
      }
    };

    // Build in-memory indices
    await this.buildIndices(snapshot);
    
    // Save snapshot
    await this.saveSnapshot(snapshot);
    this.snapshot = snapshot;
    
    if (progressCallback) {
      progressCallback(1.0);
    }

    return snapshot;
  }

  private async enhanceCard(
    baseCard: SemanticCard,
    symbolEmbeddings: Map<string, Float32Array>
  ): Promise<EnhancedSemanticCard> {
    const startEnhancement = Date.now();
    
    // Generate semantic embedding from card content
    const e_sem = await this.generateSemanticEmbedding(baseCard);
    const e_syntax = await this.generateSyntaxEmbedding(baseCard);
    const e_context = await this.generateContextEmbedding(baseCard);
    
    const embeddingTime = Date.now() - startEnhancement;
    
    // Compute businessness score
    const businessnessStart = Date.now();
    const businessness = this.computeBusinessness(baseCard);
    const businessnessTime = Date.now() - businessnessStart;
    
    // Compute confidence scores
    const confidenceScores = this.computeConfidenceScores(baseCard, e_sem, businessness);

    return {
      ...baseCard,
      e_sem,
      e_syntax,
      e_context,
      businessness,
      extraction_metadata: {
        extracted_at: Date.now(),
        extraction_version: '2.0.0',
        confidence_scores: confidenceScores,
        processing_stats: {
          tokens_processed: this.countTokens(baseCard),
          embedding_time_ms: embeddingTime,
          businessness_time_ms: businessnessTime
        }
      },
      topic_associations: {
        secondary_topic_ids: [],
        topic_similarity_scores: {},
        topic_coverage: 0
      }
    };
  }

  private async generateSemanticEmbedding(card: SemanticCard): Promise<Float32Array> {
    // Combine all semantic content
    const content = [
      ...card.roles,
      ...card.resources,
      ...card.shapes,
      ...card.domain_tokens,
      card.summary || ''
    ].join(' ');
    
    // Mock embedding - in real implementation would call embedding service
    const embedding = new Float32Array(this.embeddingDimensions.semantic);
    for (let i = 0; i < embedding.length; i++) {
      embedding[i] = Math.random() * 2 - 1; // Random -1 to 1
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

  private async generateSyntaxEmbedding(card: SemanticCard): Promise<Float32Array> {
    // Focus on syntax patterns, AST structure, language constructs
    const syntaxFeatures = card.shapes.concat(
      card.symbols.map(s => `${s.kind}:${s.name}`)
    );
    
    const embedding = new Float32Array(this.embeddingDimensions.syntax);
    // Simple hash-based embedding for syntax
    for (const feature of syntaxFeatures) {
      const hash = this.simpleHash(feature);
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

  private async generateContextEmbedding(card: SemanticCard): Promise<Float32Array> {
    // Focus on usage context, imports, relationships
    const contextFeatures = [
      ...card.imports.map(imp => `import:${imp.name}`),
      ...card.exports.map(exp => `export:${exp.name}`),
      `file_type:${this.getFileType(card.file_path)}`
    ];
    
    const embedding = new Float32Array(this.embeddingDimensions.context);
    for (const feature of contextFeatures) {
      const hash = this.simpleHash(feature);
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

  private computeBusinessness(card: SemanticCard): EnhancedSemanticCard['businessness'] {
    // Extract raw features
    const rawFeatures = {
      api_call_count: this.countApiCalls(card),
      db_interaction_count: this.countDbInteractions(card),
      ui_component_count: this.countUiComponents(card),
      business_logic_keywords: this.extractBusinessLogicKeywords(card),
      import_diversity: new Set(card.imports.map(i => i.name)).size,
      method_complexity: card.symbols.map(s => this.estimateComplexity(s))
    };

    // Compute businessness components
    const components = {
      domain_term_pmi: this.computeDomainTermPMI(card.domain_tokens),
      resource_counts: Math.log1p(rawFeatures.api_call_count + rawFeatures.db_interaction_count),
      abstraction_level: this.computeAbstractionLevel(card),
      user_facing_score: this.computeUserFacingScore(card),
      complexity_score: rawFeatures.method_complexity.length > 0 
        ? rawFeatures.method_complexity.reduce((a, b) => a + b, 0) / rawFeatures.method_complexity.length
        : 0
    };

    // Linear combination (weights will be tuned)
    const B_raw = 0.3 * components.domain_term_pmi +
                 0.25 * components.resource_counts +
                 0.2 * components.abstraction_level +
                 0.15 * components.user_facing_score +
                 0.1 * components.complexity_score;

    return {
      B: B_raw, // Will be z-scored later
      components,
      raw_features: rawFeatures
    };
  }

  private countApiCalls(card: SemanticCard): number {
    const apiPatterns = ['fetch', 'axios', 'http', 'api', 'request', 'client'];
    return card.domain_tokens.filter(token => 
      apiPatterns.some(pattern => token.toLowerCase().includes(pattern))
    ).length;
  }

  private countDbInteractions(card: SemanticCard): number {
    const dbPatterns = ['query', 'select', 'insert', 'update', 'delete', 'database', 'db', 'sql'];
    return card.domain_tokens.filter(token =>
      dbPatterns.some(pattern => token.toLowerCase().includes(pattern))
    ).length;
  }

  private countUiComponents(card: SemanticCard): number {
    const uiPatterns = ['component', 'render', 'jsx', 'html', 'css', 'style', 'ui', 'button', 'input'];
    return card.shapes.filter(shape =>
      uiPatterns.some(pattern => shape.toLowerCase().includes(pattern))
    ).length;
  }

  private extractBusinessLogicKeywords(card: SemanticCard): string[] {
    const businessKeywords = [
      'validate', 'calculate', 'process', 'transform', 'business', 'rule', 'policy',
      'account', 'user', 'order', 'payment', 'invoice', 'customer', 'product'
    ];
    
    return card.domain_tokens.filter(token =>
      businessKeywords.some(keyword => token.toLowerCase().includes(keyword))
    );
  }

  private computeDomainTermPMI(domainTokens: string[]): number {
    // Pointwise Mutual Information with business domain terms
    // Simplified calculation - in real implementation would use large corpus
    const businessTerms = new Set(['business', 'user', 'customer', 'order', 'product', 'service']);
    const businessCount = domainTokens.filter(token => 
      businessTerms.has(token.toLowerCase())
    ).length;
    
    return businessCount > 0 ? Math.log(businessCount / domainTokens.length) : -5;
  }

  private computeAbstractionLevel(card: SemanticCard): number {
    // Higher abstraction = more business logic
    const abstractionIndicators = [
      'interface', 'abstract', 'service', 'manager', 'controller', 'handler'
    ];
    
    let score = 0;
    for (const symbol of card.symbols) {
      if (abstractionIndicators.some(indicator => 
        symbol.name.toLowerCase().includes(indicator)
      )) {
        score += 1;
      }
    }
    
    return Math.min(score / Math.max(1, card.symbols.length), 1);
  }

  private computeUserFacingScore(card: SemanticCard): number {
    const userFacingTerms = ['ui', 'component', 'page', 'view', 'screen', 'form'];
    const score = card.shapes.filter(shape =>
      userFacingTerms.some(term => shape.toLowerCase().includes(term))
    ).length;
    
    return Math.min(score / Math.max(1, card.shapes.length), 1);
  }

  private estimateComplexity(symbol: any): number {
    // Rough complexity estimate based on symbol characteristics
    let complexity = 1;
    
    if (symbol.kind === 'function' || symbol.kind === 'method') {
      complexity += 2;
    }
    
    if (symbol.name.length > 20) {
      complexity += 1; // Long names often indicate complex operations
    }
    
    return complexity;
  }

  private computeConfidenceScores(
    card: SemanticCard, 
    embedding: Float32Array, 
    businessness: EnhancedSemanticCard['businessness']
  ) {
    // Semantic quality based on content richness
    const semantic_quality = Math.min(
      (card.domain_tokens.length + card.roles.length + card.resources.length) / 20,
      1.0
    );
    
    // Businessness confidence based on feature consistency
    const businessness_confidence = businessness.raw_features.business_logic_keywords.length > 0 ? 0.8 : 0.4;
    
    // Embedding quality based on norm and distribution
    const embedding_norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    const embedding_quality = Math.min(embedding_norm, 1.0);
    
    return {
      semantic_quality,
      businessness_confidence,
      embedding_quality
    };
  }

  private createSymbolCard(fileCard: EnhancedSemanticCard, symbol: any): EnhancedSemanticCard {
    // Create a symbol-specific card by focusing embedding on the symbol
    const symbolCard: EnhancedSemanticCard = {
      ...fileCard,
      file_id: symbol.id,
      symbols: [symbol],
      // Reduce other content to focus on this symbol
      roles: fileCard.roles.filter(role => role.includes(symbol.name)),
      resources: fileCard.resources.filter(res => res.includes(symbol.name)),
      shapes: fileCard.shapes.filter(shape => shape.includes(symbol.name))
    };

    // Adjust businessness for symbol-specific context
    symbolCard.businessness.B *= 0.8; // Symbol cards generally less "business" than full files
    
    return symbolCard;
  }

  private computeBusinessnessStats(cards: Map<string, EnhancedSemanticCard>): BusinessnessStats {
    const bScores = Array.from(cards.values()).map(card => card.businessness.B);
    bScores.sort((a, b) => a - b);
    
    const mean_B = bScores.reduce((sum, b) => sum + b, 0) / bScores.length;
    const variance = bScores.reduce((sum, b) => sum + Math.pow(b - mean_B, 2), 0) / bScores.length;
    const std_B = Math.sqrt(variance);
    
    const global_stats = {
      mean_B,
      std_B,
      min_B: bScores[0],
      max_B: bScores[bScores.length - 1],
      percentiles: {
        P25: bScores[Math.floor(bScores.length * 0.25)],
        P50: bScores[Math.floor(bScores.length * 0.5)],
        P75: bScores[Math.floor(bScores.length * 0.75)],
        P90: bScores[Math.floor(bScores.length * 0.9)],
        P95: bScores[Math.floor(bScores.length * 0.95)]
      }
    };

    // Compute component distributions
    const component_distributions = this.computeComponentDistributions(cards);
    
    // Build domain terms frequency map
    const domain_terms = new Map<string, number>();
    const business_logic_patterns = new Map<string, number>();
    
    for (const card of cards.values()) {
      for (const token of card.domain_tokens) {
        domain_terms.set(token, (domain_terms.get(token) || 0) + 1);
      }
      
      for (const keyword of card.businessness.raw_features.business_logic_keywords) {
        business_logic_patterns.set(keyword, (business_logic_patterns.get(keyword) || 0) + 1);
      }
    }

    return {
      global_stats,
      component_distributions,
      domain_terms,
      business_logic_patterns
    };
  }

  private computeComponentDistributions(
    cards: Map<string, EnhancedSemanticCard>
  ): BusinessnessStats['component_distributions'] {
    const components = {
      domain_term_pmi: [] as number[],
      resource_counts: [] as number[],
      abstraction_level: [] as number[],
      user_facing_score: [] as number[],
      complexity_score: [] as number[]
    };

    for (const card of cards.values()) {
      const comp = card.businessness.components;
      components.domain_term_pmi.push(comp.domain_term_pmi);
      components.resource_counts.push(comp.resource_counts);
      components.abstraction_level.push(comp.abstraction_level);
      components.user_facing_score.push(comp.user_facing_score);
      components.complexity_score.push(comp.complexity_score);
    }

    const result: BusinessnessStats['component_distributions'] = {} as any;
    for (const [key, values] of Object.entries(components)) {
      result[key as keyof BusinessnessStats['component_distributions']] = this.computeDistribution(values);
    }

    return result;
  }

  private computeDistribution(values: number[]): Distribution {
    values.sort((a, b) => a - b);
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    const std = Math.sqrt(variance);

    // Create histogram with 10 bins
    const histogram = [];
    const binSize = (values[values.length - 1] - values[0]) / 10;
    
    for (let i = 0; i < 10; i++) {
      const binStart = values[0] + i * binSize;
      const binEnd = binStart + binSize;
      const count = values.filter(v => v >= binStart && v < binEnd).length;
      histogram.push({ bin: i, count });
    }

    return {
      mean,
      std,
      min: values[0],
      max: values[values.length - 1],
      histogram
    };
  }

  private normalizeBusinessnessScores(
    cards: Map<string, EnhancedSemanticCard>, 
    stats: BusinessnessStats
  ): void {
    const { mean_B, std_B } = stats.global_stats;
    
    for (const card of cards.values()) {
      // Z-score normalization
      card.businessness.B = std_B > 0 ? (card.businessness.B - mean_B) / std_B : 0;
    }
  }

  private async computeEmbeddingStats(cards: Map<string, EnhancedSemanticCard>): Promise<EmbeddingStats> {
    // Simple k-means clustering on semantic embeddings
    const embeddings = Array.from(cards.values()).map(card => card.e_sem);
    const cardIds = Array.from(cards.keys());
    
    // Mock clustering - in real implementation would use proper k-means
    const numClusters = Math.min(10, Math.floor(embeddings.length / 5));
    const cluster_centers = [];
    const cluster_assignments = new Map<string, number>();
    
    for (let i = 0; i < numClusters; i++) {
      cluster_centers.push(new Float32Array(this.embeddingDimensions.semantic));
    }
    
    // Assign cards to random clusters for now
    cardIds.forEach((cardId, index) => {
      cluster_assignments.set(cardId, index % numClusters);
    });

    // Compute embedding quality metrics
    const norms = embeddings.map(emb => 
      Math.sqrt(emb.reduce((sum, val) => sum + val * val, 0))
    );
    const avg_norm = norms.reduce((sum, norm) => sum + norm, 0) / norms.length;

    return {
      semantic_cluster_info: {
        num_clusters: numClusters,
        cluster_centers,
        cluster_assignments,
        intra_cluster_similarity: new Array(numClusters).fill(0.8) // Mock values
      },
      embedding_quality: {
        avg_norm,
        dimensionality_scores: new Array(this.embeddingDimensions.semantic).fill(0.5),
        outlier_detection: {
          outlier_cards: cardIds.slice(0, Math.min(3, cardIds.length)),
          outlier_threshold: 2.0
        }
      }
    };
  }

  private computeCoverageMetrics(
    cards: Map<string, EnhancedSemanticCard>,
    symbolCards: Map<string, EnhancedSemanticCard>
  ): CoverageMetrics {
    const fileSet = new Set(Array.from(cards.values()).map(card => card.file_path));
    const symbolKinds = new Map<SymbolKind, number>();
    
    let highQualityCards = 0;
    let mediumQualityCards = 0;
    let lowQualityCards = 0;

    for (const card of cards.values()) {
      const avgConfidence = (
        card.extraction_metadata.confidence_scores.semantic_quality +
        card.extraction_metadata.confidence_scores.businessness_confidence +
        card.extraction_metadata.confidence_scores.embedding_quality
      ) / 3;

      if (avgConfidence > 0.8) highQualityCards++;
      else if (avgConfidence > 0.5) mediumQualityCards++;
      else lowQualityCards++;

      for (const symbol of card.symbols) {
        symbolKinds.set(symbol.kind, (symbolKinds.get(symbol.kind) || 0) + 1);
      }
    }

    const coverage_by_kind: Record<SymbolKind, number> = {} as any;
    for (const kind of symbolKinds.keys()) {
      coverage_by_kind[kind] = symbolKinds.get(kind) || 0;
    }

    return {
      file_coverage: {
        files_with_cards: cards.size,
        total_files: Math.max(cards.size, fileSet.size),
        coverage_ratio: cards.size / Math.max(cards.size, fileSet.size)
      },
      symbol_coverage: {
        symbols_with_cards: symbolCards.size,
        total_symbols: Array.from(cards.values()).reduce((sum, card) => sum + card.symbols.length, 0),
        coverage_by_kind
      },
      quality_distribution: {
        high_quality_cards: highQualityCards,
        medium_quality_cards: mediumQualityCards,
        low_quality_cards: lowQualityCards
      }
    };
  }

  private async buildIndices(snapshot: CardStoreSnapshot): Promise<void> {
    // Build semantic similarity index
    this.semanticIndex = Array.from(snapshot.cards.values()).map(card => card.e_sem);
    this.cardIdIndex = Array.from(snapshot.cards.keys());
    
    // Build businessness index
    this.businessnessIndex = new Map();
    for (const [id, card] of snapshot.cards) {
      this.businessnessIndex.set(id, card.businessness.B);
    }
  }

  /**
   * Query cards with various filters and similarity search
   */
  async query(query: CardStoreQuery): Promise<QueryResult> {
    const startTime = Date.now();
    
    if (!this.snapshot) {
      return { cards: [], scores: [], total_matches: 0, query_time_ms: 0 };
    }

    let candidates = Array.from(this.snapshot.cards.values());
    let scores = new Array(candidates.length).fill(1.0);

    // Apply semantic similarity filter
    if (query.semantic_similarity && this.semanticIndex && this.cardIdIndex) {
      const similarities = this.computeSimilarities(query.semantic_similarity.query_embedding);
      
      const similarResults: Array<{card: EnhancedSemanticCard, score: number}> = [];
      for (let i = 0; i < similarities.length; i++) {
        if (similarities[i] >= query.semantic_similarity.min_similarity) {
          const cardId = this.cardIdIndex[i];
          const card = this.snapshot.cards.get(cardId);
          if (card) {
            similarResults.push({ card, score: similarities[i] });
          }
        }
      }
      
      similarResults.sort((a, b) => b.score - a.score);
      candidates = similarResults.slice(0, query.semantic_similarity.max_results).map(r => r.card);
      scores = similarResults.slice(0, query.semantic_similarity.max_results).map(r => r.score);
    }

    // Apply businessness range filter
    if (query.businessness_range) {
      const filtered: Array<{card: EnhancedSemanticCard, score: number}> = [];
      for (let i = 0; i < candidates.length; i++) {
        const card = candidates[i];
        if (card.businessness.B >= query.businessness_range.min_B && 
            card.businessness.B <= query.businessness_range.max_B) {
          filtered.push({ card, score: scores[i] });
        }
      }
      candidates = filtered.map(f => f.card);
      scores = filtered.map(f => f.score);
    }

    // Apply file pattern filters
    if (query.file_filters) {
      candidates = candidates.filter(card => {
        const path = card.file_path;
        
        if (query.file_filters!.include_patterns.length > 0) {
          const included = query.file_filters!.include_patterns.some(pattern => 
            path.includes(pattern) // Simple pattern matching
          );
          if (!included) return false;
        }
        
        if (query.file_filters!.exclude_patterns.length > 0) {
          const excluded = query.file_filters!.exclude_patterns.some(pattern => 
            path.includes(pattern)
          );
          if (excluded) return false;
        }
        
        return true;
      });
    }

    // Apply symbol kind filter
    if (query.symbol_kind_filter) {
      candidates = candidates.filter(card => 
        card.symbols.some(symbol => 
          query.symbol_kind_filter!.includes(symbol.kind)
        )
      );
    }

    return {
      cards: candidates,
      scores,
      total_matches: candidates.length,
      query_time_ms: Date.now() - startTime
    };
  }

  private computeSimilarities(queryEmbedding: Float32Array): number[] {
    if (!this.semanticIndex) return [];
    
    return this.semanticIndex.map(cardEmbedding => {
      return this.cosineSimilarity(queryEmbedding, cardEmbedding);
    });
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

  // Utility functions
  private countTokens(card: SemanticCard): number {
    return [
      ...card.roles,
      ...card.resources,
      ...card.shapes,
      ...card.domain_tokens
    ].join(' ').split(' ').length;
  }

  private getFileType(filePath: string): string {
    const ext = path.extname(filePath);
    return ext.slice(1) || 'unknown';
  }

  private simpleHash(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash);
  }

  // Storage operations
  private async saveSnapshot(snapshot: CardStoreSnapshot): Promise<void> {
    const filePath = path.join(this.storagePath, `CardStore-${snapshot.repo_sha}.msgpack`);
    
    // Convert to serializable format
    const serializable = {
      ...snapshot,
      cards: Object.fromEntries(
        Array.from(snapshot.cards.entries()).map(([k, v]) => [k, {
          ...v,
          e_sem: Array.from(v.e_sem),
          e_syntax: Array.from(v.e_syntax),
          e_context: Array.from(v.e_context)
        }])
      ),
      symbol_cards: Object.fromEntries(
        Array.from(snapshot.symbol_cards.entries()).map(([k, v]) => [k, {
          ...v,
          e_sem: Array.from(v.e_sem),
          e_syntax: Array.from(v.e_syntax),
          e_context: Array.from(v.e_context)
        }])
      ),
      businessness_stats: {
        ...snapshot.businessness_stats,
        domain_terms: Object.fromEntries(snapshot.businessness_stats.domain_terms),
        business_logic_patterns: Object.fromEntries(snapshot.businessness_stats.business_logic_patterns)
      },
      embedding_stats: {
        ...snapshot.embedding_stats,
        semantic_cluster_info: {
          ...snapshot.embedding_stats.semantic_cluster_info,
          cluster_centers: snapshot.embedding_stats.semantic_cluster_info.cluster_centers.map(c => Array.from(c)),
          cluster_assignments: Object.fromEntries(snapshot.embedding_stats.semantic_cluster_info.cluster_assignments)
        }
      }
    };
    
    const buffer = msgpack.encode(serializable);
    await fs.writeFile(filePath, buffer);
  }

  async loadSnapshot(repoSha: string): Promise<CardStoreSnapshot> {
    const filePath = path.join(this.storagePath, `CardStore-${repoSha}.msgpack`);
    
    try {
      const buffer = await fs.readFile(filePath);
      const data = msgpack.decode(buffer);
      
      // Convert back to proper types
      const snapshot: CardStoreSnapshot = {
        ...data,
        cards: new Map(
          Object.entries(data.cards).map(([k, v]: [string, any]) => [k, {
            ...v,
            e_sem: new Float32Array(v.e_sem),
            e_syntax: new Float32Array(v.e_syntax),
            e_context: new Float32Array(v.e_context)
          }])
        ),
        symbol_cards: new Map(
          Object.entries(data.symbol_cards).map(([k, v]: [string, any]) => [k, {
            ...v,
            e_sem: new Float32Array(v.e_sem),
            e_syntax: new Float32Array(v.e_syntax),
            e_context: new Float32Array(v.e_context)
          }])
        ),
        businessness_stats: {
          ...data.businessness_stats,
          domain_terms: new Map(Object.entries(data.businessness_stats.domain_terms)),
          business_logic_patterns: new Map(Object.entries(data.businessness_stats.business_logic_patterns))
        },
        embedding_stats: {
          ...data.embedding_stats,
          semantic_cluster_info: {
            ...data.embedding_stats.semantic_cluster_info,
            cluster_centers: data.embedding_stats.semantic_cluster_info.cluster_centers.map((c: number[]) => new Float32Array(c)),
            cluster_assignments: new Map(Object.entries(data.embedding_stats.semantic_cluster_info.cluster_assignments))
          }
        }
      };
      
      this.snapshot = snapshot;
      await this.buildIndices(snapshot);
      
      return snapshot;
      
    } catch (error) {
      throw new Error(`Failed to load CardStore snapshot: ${error}`);
    }
  }

  // Public access methods
  getSnapshot(): CardStoreSnapshot | undefined {
    return this.snapshot;
  }

  getCard(cardId: string): EnhancedSemanticCard | undefined {
    return this.snapshot?.cards.get(cardId);
  }

  getSymbolCard(symbolId: string): EnhancedSemanticCard | undefined {
    return this.snapshot?.symbol_cards.get(symbolId);
  }

  getBusinessnessStats(): BusinessnessStats | undefined {
    return this.snapshot?.businessness_stats;
  }

  getEmbeddingStats(): EmbeddingStats | undefined {
    return this.snapshot?.embedding_stats;
  }

  getCoverageMetrics(): CoverageMetrics | undefined {
    return this.snapshot?.coverage_metrics;
  }

  // High-level convenience methods
  async findSimilarCards(
    queryEmbedding: Float32Array, 
    topK: number = 10,
    minSimilarity: number = 0.5
  ): Promise<Array<{card: EnhancedSemanticCard, similarity: number}>> {
    const result = await this.query({
      semantic_similarity: {
        query_embedding: queryEmbedding,
        min_similarity: minSimilarity,
        max_results: topK
      }
    });

    return result.cards.map((card, i) => ({
      card,
      similarity: result.scores[i]
    }));
  }

  async findBusinessCards(minBusinessness: number = 1.0): Promise<EnhancedSemanticCard[]> {
    const result = await this.query({
      businessness_range: {
        min_B: minBusinessness,
        max_B: Infinity
      }
    });
    
    return result.cards;
  }
}