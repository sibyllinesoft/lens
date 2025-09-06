/**
 * Embedding and Businessness Scoring System
 * 
 * Handles semantic embeddings for RAPTOR cards and nodes, plus computation
 * of businessness scores based on domain-specific features.
 */

import { SemanticCard, Resources, Shapes, Role } from './semantic-card.js';

export interface EmbeddingProvider {
  embed(texts: string[]): Promise<number[][]>;
  getDimension(): number;
}

export interface BusinessnessStats {
  pmiMean: number;
  pmiStd: number;
  resourceMean: number;
  resourceStd: number;
  shapeMean: number;
  shapeStd: number;
  utilMean: number;
  utilStd: number;
}

export interface EmbeddingConfig {
  provider: string;
  model: string;
  dimension: number;
  batchSize: number;
  maxTokens: number;
}

/**
 * Mock embedding provider for development and testing
 */
export class MockEmbeddingProvider implements EmbeddingProvider {
  private dimension: number;

  constructor(dimension: number = 384) {
    this.dimension = dimension;
  }

  async embed(texts: string[]): Promise<number[][]> {
    // Generate deterministic mock embeddings based on text content
    return texts.map(text => this.generateMockEmbedding(text));
  }

  getDimension(): number {
    return this.dimension;
  }

  private generateMockEmbedding(text: string): number[] {
    const embedding = new Array(this.dimension);
    let seed = this.hashString(text);
    
    for (let i = 0; i < this.dimension; i++) {
      // Simple PRNG to generate reproducible embeddings
      seed = (seed * 9301 + 49297) % 233280;
      embedding[i] = (seed / 233280) * 2 - 1; // Normalize to [-1, 1]
    }
    
    // L2 normalize
    const norm = Math.sqrt(embedding.reduce((sum, x) => sum + x * x, 0));
    return embedding.map(x => x / norm);
  }

  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }
}

/**
 * Sentence transformer embedding provider (for production)
 */
export class SentenceTransformerProvider implements EmbeddingProvider {
  private dimension: number;
  private apiEndpoint: string;
  private model: string;

  constructor(
    apiEndpoint: string = "http://localhost:8000/embed",
    model: string = "all-MiniLM-L6-v2",
    dimension: number = 384
  ) {
    this.apiEndpoint = apiEndpoint;
    this.model = model;
    this.dimension = dimension;
  }

  async embed(texts: string[]): Promise<number[][]> {
    try {
      const response = await fetch(this.apiEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          texts,
          model: this.model
        })
      });

      if (!response.ok) {
        throw new Error(`Embedding API error: ${response.statusText}`);
      }

      const result = await response.json() as { embeddings: number[][] };
      return result.embeddings;
    } catch (error) {
      console.warn('Falling back to mock embeddings due to error:', error);
      const mockProvider = new MockEmbeddingProvider(this.dimension);
      return mockProvider.embed(texts);
    }
  }

  getDimension(): number {
    return this.dimension;
  }
}

/**
 * Main embedding service for RAPTOR
 */
export class RaptorEmbeddingService {
  private provider: EmbeddingProvider;
  private cache: Map<string, number[]>;

  constructor(provider: EmbeddingProvider) {
    this.provider = provider;
    this.cache = new Map();
  }

  static createDefaultService(): RaptorEmbeddingService {
    return new RaptorEmbeddingService(new MockEmbeddingProvider());
  }

  async embedSemanticCard(card: SemanticCard): Promise<number[]> {
    const facetText = this.buildCardFacetText(card);
    const cacheKey = this.hashText(facetText);
    
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!;
    }

    const embeddings = await this.provider.embed([facetText]);
    const embedding = embeddings[0];
    
    this.cache.set(cacheKey, embedding);
    return embedding;
  }

  async embedSummary(summary: string): Promise<number[]> {
    const cacheKey = this.hashText(summary);
    
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!;
    }

    const embeddings = await this.provider.embed([summary]);
    const embedding = embeddings[0];
    
    this.cache.set(cacheKey, embedding);
    return embedding;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const uncachedTexts: string[] = [];
    const uncachedIndices: number[] = [];
    const results: number[][] = new Array(texts.length);

    // Check cache for each text
    texts.forEach((text, i) => {
      const cacheKey = this.hashText(text);
      if (this.cache.has(cacheKey)) {
        results[i] = this.cache.get(cacheKey)!;
      } else {
        uncachedTexts.push(text);
        uncachedIndices.push(i);
      }
    });

    // Embed uncached texts
    if (uncachedTexts.length > 0) {
      const embeddings = await this.provider.embed(uncachedTexts);
      
      embeddings.forEach((embedding, i) => {
        const originalIndex = uncachedIndices[i];
        const text = texts[originalIndex];
        const cacheKey = this.hashText(text);
        
        this.cache.set(cacheKey, embedding);
        results[originalIndex] = embedding;
      });
    }

    return results;
  }

  private buildCardFacetText(card: SemanticCard): string {
    const parts: string[] = [];

    // Roles and resources
    if (card.roles.length > 0) {
      parts.push(`Roles: ${card.roles.join(', ')}`);
    }

    if (card.resources.routes.length > 0) {
      parts.push(`Routes: ${card.resources.routes.join(', ')}`);
    }

    if (card.resources.sql.length > 0) {
      parts.push(`Tables: ${card.resources.sql.join(', ')}`);
    }

    if (card.resources.topics.length > 0) {
      parts.push(`Topics: ${card.resources.topics.join(', ')}`);
    }

    // Shapes
    if (card.shapes.typeNames.length > 0) {
      parts.push(`Types: ${card.shapes.typeNames.join(', ')}`);
    }

    if (card.shapes.jsonKeys.length > 0) {
      parts.push(`Keys: ${card.shapes.jsonKeys.slice(0, 10).join(', ')}`);
    }

    // Domain tokens (top 10)
    if (card.domainTokens.length > 0) {
      parts.push(`Domain: ${card.domainTokens.slice(0, 10).join(', ')}`);
    }

    // Effects
    if (card.effects.length > 0) {
      parts.push(`Effects: ${card.effects.join(', ')}`);
    }

    return parts.join('. ');
  }

  computeCentroid(embeddings: number[][]): number[] {
    if (embeddings.length === 0) {
      return new Array(this.provider.getDimension()).fill(0);
    }

    const dimension = embeddings[0].length;
    const centroid = new Array(dimension).fill(0);

    // Sum all embeddings
    for (const embedding of embeddings) {
      for (let i = 0; i < dimension; i++) {
        centroid[i] += embedding[i];
      }
    }

    // Average and normalize
    const count = embeddings.length;
    for (let i = 0; i < dimension; i++) {
      centroid[i] /= count;
    }

    // L2 normalize
    const norm = Math.sqrt(centroid.reduce((sum, x) => sum + x * x, 0));
    if (norm > 0) {
      for (let i = 0; i < dimension; i++) {
        centroid[i] /= norm;
      }
    }

    return centroid;
  }

  cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error('Vectors must have same dimension');
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    normA = Math.sqrt(normA);
    normB = Math.sqrt(normB);

    if (normA === 0 || normB === 0) {
      return 0;
    }

    return dotProduct / (normA * normB);
  }

  private hashText(text: string): string {
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      const char = text.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return hash.toString(16);
  }

  clearCache(): void {
    this.cache.clear();
  }

  getCacheSize(): number {
    return this.cache.size;
  }
}

/**
 * Businessness scoring system
 */
export class BusinessnessScorer {
  private stats: BusinessnessStats;

  constructor(stats?: BusinessnessStats) {
    this.stats = stats || this.getDefaultStats();
  }

  private getDefaultStats(): BusinessnessStats {
    return {
      pmiMean: 1.0,
      pmiStd: 0.5,
      resourceMean: 2.0,
      resourceStd: 1.0,
      shapeMean: 3.0,
      shapeStd: 2.0,
      utilMean: 0.3,
      utilStd: 0.2
    };
  }

  computeBusinessnessScore(
    pmiDomain: number,
    resourceCount: number,
    shapeSpecificity: number,
    roles: Role[],
    utilAffinity: number
  ): number {
    // Z-score normalization
    const zPmi = this.zScore(pmiDomain, this.stats.pmiMean, this.stats.pmiStd);
    const zResources = this.zScore(resourceCount, this.stats.resourceMean, this.stats.resourceStd);
    const zShapes = this.zScore(shapeSpecificity, this.stats.shapeMean, this.stats.shapeStd);
    const zUtil = this.zScore(utilAffinity, this.stats.utilMean, this.stats.utilStd);

    // Business role bonus
    const hasBusinessRole = roles.some(role => ["handler", "service", "repo"].includes(role));
    const roleBonus = hasBusinessRole ? 1 : 0;

    // Utility affinity penalty
    const utilPenalty = -0.5 * zUtil;

    const score = zPmi + zResources + zShapes + roleBonus + utilPenalty;
    
    return score;
  }

  computeStatsFromCards(cards: SemanticCard[]): BusinessnessStats {
    if (cards.length === 0) {
      return this.getDefaultStats();
    }

    const pmiValues = cards.map(card => this.computePMIDomain(card.domainTokens));
    const resourceCounts = cards.map(card => this.countResources(card.resources));
    const shapeSpecificities = cards.map(card => this.computeShapeSpecificity(card.shapes));
    const utilAffinities = cards.map(card => card.utilAffinity);

    return {
      pmiMean: this.mean(pmiValues),
      pmiStd: this.std(pmiValues),
      resourceMean: this.mean(resourceCounts),
      resourceStd: this.std(resourceCounts),
      shapeMean: this.mean(shapeSpecificities),
      shapeStd: this.std(shapeSpecificities),
      utilMean: this.mean(utilAffinities),
      utilStd: this.std(utilAffinities)
    };
  }

  private computePMIDomain(domainTokens: string[]): number {
    // Simplified PMI computation - in production, use actual corpus statistics
    return domainTokens.length > 0 ? domainTokens.length * 0.5 : 0;
  }

  private countResources(resources: Resources): number {
    return resources.routes.length + 
           resources.sql.length + 
           resources.topics.length + 
           resources.buckets.length + 
           resources.featureFlags.length;
  }

  private computeShapeSpecificity(shapes: Shapes): number {
    // More specific type names and JSON keys indicate business logic
    const typeSpecificity = shapes.typeNames.reduce((sum, name) => {
      return sum + Math.min(name.length / 10, 2);
    }, 0);

    const keySpecificity = shapes.jsonKeys.reduce((sum, key) => {
      const isGeneric = ["id", "name", "type", "data", "value", "config"].includes(key.toLowerCase());
      return sum + (isGeneric ? 0.1 : 1);
    }, 0);

    return typeSpecificity + keySpecificity;
  }

  private zScore(value: number, mean: number, std: number): number {
    return std > 0 ? (value - mean) / std : 0;
  }

  private mean(values: number[]): number {
    return values.length > 0 ? values.reduce((sum, x) => sum + x, 0) / values.length : 0;
  }

  private std(values: number[]): number {
    if (values.length === 0) return 0;
    
    const meanVal = this.mean(values);
    const squaredDiffs = values.map(x => Math.pow(x - meanVal, 2));
    const variance = this.mean(squaredDiffs);
    
    return Math.sqrt(variance);
  }

  updateStats(newStats: BusinessnessStats): void {
    this.stats = newStats;
  }

  getStats(): BusinessnessStats {
    return { ...this.stats };
  }
}

export default RaptorEmbeddingService;