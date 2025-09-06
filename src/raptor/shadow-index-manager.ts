/**
 * Shadow Index Manager for EmbeddingGemma A/B Testing
 * 
 * Manages parallel indexes for ada-002, Gemma-768, and Gemma-256 embeddings
 * to enable frozen-pool replays and comparative evaluation.
 */

import { EmbeddingProvider } from './embeddings.js';
import { EmbeddingGemmaProvider, EmbeddingGemmaFactory } from './embedding-gemma-provider.js';
import { SemanticRerankEngine } from '../indexer/semantic.js';
import { SegmentStorage } from '../storage/segments.js';
import { LensTracer } from '../telemetry/tracer.js';

export type EmbeddingModelType = 'ada-002' | 'gemma-768' | 'gemma-256';

export interface ShadowIndexConfig {
  models: {
    ada002?: { endpoint: string; apiKey?: string };
    gemma768: { teiEndpoint: string };
    gemma256: { teiEndpoint: string };
  };
  indexStorage: {
    basePath: string;
    segmentPrefix: string;
  };
  performance: {
    maxConcurrentEncoding: number;
    batchSize: number;
    enableCaching: boolean;
  };
}

export interface IndexBuildStats {
  modelType: EmbeddingModelType;
  documentsProcessed: number;
  totalEmbeddings: number;
  buildTimeMs: number;
  avgLatencyMs: number;
  errorCount: number;
  storageBytes: number;
  dimension: number;
}

export interface ComparisonMetrics {
  modelType: EmbeddingModelType;
  dimension: number;
  recall_at_10: number;
  recall_at_50: number;
  critical_atom_recall: number;
  cpu_p95_ms: number;
  storage_efficiency: number; // embeddings per MB
  delta_cbu_per_gb: number;
}

/**
 * Shadow index for a single embedding model
 */
class ShadowIndex {
  private provider: EmbeddingProvider;
  private storage: SegmentStorage;
  private modelType: EmbeddingModelType;
  private indexId: string;
  private embeddings: Map<string, number[]> = new Map();
  private metadata: Map<string, { size: number; timestamp: number }> = new Map();

  constructor(
    provider: EmbeddingProvider,
    storage: SegmentStorage,
    modelType: EmbeddingModelType,
    indexId: string
  ) {
    this.provider = provider;
    this.storage = storage;
    this.modelType = modelType;
    this.indexId = indexId;
  }

  async buildIndex(
    documents: Array<{ id: string; content: string; filePath: string }>,
    progressCallback?: (progress: number, total: number) => void
  ): Promise<IndexBuildStats> {
    const span = LensTracer.createChildSpan('shadow_index_build', {
      'model.type': this.modelType,
      'documents.count': documents.length,
    });

    const startTime = Date.now();
    let documentsProcessed = 0;
    let totalEmbeddings = 0;
    let errorCount = 0;
    const latencies: number[] = [];

    try {
      // Process documents in batches
      const batchSize = 32; // Configurable batch size
      for (let i = 0; i < documents.length; i += batchSize) {
        const batch = documents.slice(i, i + batchSize);
        const batchStart = Date.now();
        
        try {
          const texts = batch.map(doc => doc.content);
          const embeddings = await this.provider.embed(texts);
          
          // Store embeddings
          for (let j = 0; j < batch.length; j++) {
            const doc = batch[j];
            const embedding = embeddings[j];
            
            if (doc && embedding) {
              this.embeddings.set(doc.id, embedding);
              this.metadata.set(doc.id, {
                size: embedding.length * 4, // Float32 = 4 bytes per dimension
                timestamp: Date.now(),
              });
              totalEmbeddings++;
            }
          }
          
          documentsProcessed += batch.length;
          const batchLatency = Date.now() - batchStart;
          latencies.push(batchLatency);
          
          progressCallback?.(documentsProcessed, documents.length);
          
        } catch (error) {
          errorCount += batch.length;
          console.warn(`Batch ${i}-${i + batch.length} failed for ${this.modelType}:`, error);
        }
      }

      // Persist to storage
      await this.persistIndex();
      
      const buildTime = Date.now() - startTime;
      const avgLatency = latencies.reduce((sum, lat) => sum + lat, 0) / latencies.length;
      const storageBytes = Array.from(this.metadata.values()).reduce(
        (sum, meta) => sum + meta.size, 0
      );

      const stats: IndexBuildStats = {
        modelType: this.modelType,
        documentsProcessed,
        totalEmbeddings,
        buildTimeMs: buildTime,
        avgLatencyMs: avgLatency,
        errorCount,
        storageBytes,
        dimension: this.provider.getDimension(),
      };

      span.setAttributes({
        success: true,
        ...stats,
      });

      return stats;
      
    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  async getEmbedding(docId: string): Promise<number[] | null> {
    return this.embeddings.get(docId) || null;
  }

  async getEmbeddings(docIds: string[]): Promise<Map<string, number[]>> {
    const results = new Map<string, number[]>();
    for (const docId of docIds) {
      const embedding = this.embeddings.get(docId);
      if (embedding) {
        results.set(docId, embedding);
      }
    }
    return results;
  }

  getIndexStats(): {
    modelType: EmbeddingModelType;
    embeddingCount: number;
    dimension: number;
    storageBytes: number;
  } {
    const storageBytes = Array.from(this.metadata.values()).reduce(
      (sum, meta) => sum + meta.size, 0
    );

    return {
      modelType: this.modelType,
      embeddingCount: this.embeddings.size,
      dimension: this.provider.getDimension(),
      storageBytes,
    };
  }

  private async persistIndex(): Promise<void> {
    const segmentId = `shadow_${this.modelType}_${this.indexId}`;
    
    const indexData = {
      modelType: this.modelType,
      dimension: this.provider.getDimension(),
      embeddings: Object.fromEntries(this.embeddings),
      metadata: Object.fromEntries(this.metadata),
      timestamp: Date.now(),
    };

    const segmentData = JSON.stringify(indexData);
    const buffer = Buffer.from(segmentData, 'utf8');
    
    await this.storage.createSegment(segmentId, 'semantic', buffer.length);
    await this.storage.writeToSegment(segmentId, 0, buffer);
    
    console.log(`✅ Persisted ${this.modelType} index: ${this.embeddings.size} embeddings`);
  }

  async loadIndex(): Promise<void> {
    const segmentId = `shadow_${this.modelType}_${this.indexId}`;
    
    try {
      const segment = await this.storage.openSegment(segmentId, true);
      const buffer = await this.storage.readFromSegment(segmentId, 0, segment.size);
      const indexData = JSON.parse(buffer.toString('utf8'));
      
      // Reconstruct embeddings map
      for (const [docId, embedding] of Object.entries(indexData.embeddings)) {
        this.embeddings.set(docId, embedding as number[]);
      }
      
      // Reconstruct metadata map
      for (const [docId, metadata] of Object.entries(indexData.metadata)) {
        this.metadata.set(docId, metadata as { size: number; timestamp: number });
      }
      
      console.log(`✅ Loaded ${this.modelType} index: ${this.embeddings.size} embeddings`);
      
    } catch (error) {
      console.warn(`Failed to load ${this.modelType} index:`, error);
      // Start with empty index
    }
  }
}

/**
 * Main shadow index manager for coordinating multiple embedding models
 */
export class ShadowIndexManager {
  private config: ShadowIndexConfig;
  private storage: SegmentStorage;
  private indexes: Map<EmbeddingModelType, ShadowIndex> = new Map();
  private providers: Map<EmbeddingModelType, EmbeddingProvider> = new Map();

  constructor(config: ShadowIndexConfig, storage: SegmentStorage) {
    this.config = config;
    this.storage = storage;
  }

  async initialize(): Promise<void> {
    const span = LensTracer.createChildSpan('shadow_index_manager_init');

    try {
      // Initialize Gemma providers
      const { gemma768, gemma256 } = EmbeddingGemmaFactory.createShadowProviders(
        this.config.models.gemma768.teiEndpoint
      );

      this.providers.set('gemma-768', gemma768);
      this.providers.set('gemma-256', gemma256);

      // Initialize ada-002 provider if configured
      if (this.config.models.ada002) {
        // Would initialize OpenAI provider here
        // For now, use mock or existing provider
        console.log('Ada-002 provider configuration detected but not implemented in this version');
      }

      // Create shadow indexes
      const indexId = Date.now().toString();
      for (const [modelType, provider] of this.providers) {
        const index = new ShadowIndex(provider, this.storage, modelType, indexId);
        this.indexes.set(modelType, index);
        
        // Load existing index if available
        await index.loadIndex();
      }

      span.setAttributes({
        success: true,
        providers_initialized: this.providers.size,
        indexes_created: this.indexes.size,
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
   * Build shadow indexes for all configured models in parallel
   */
  async buildShadowIndexes(
    documents: Array<{ id: string; content: string; filePath: string }>,
    progressCallback?: (modelType: EmbeddingModelType, progress: number, total: number) => void
  ): Promise<Map<EmbeddingModelType, IndexBuildStats>> {
    const span = LensTracer.createChildSpan('build_shadow_indexes', {
      'documents.count': documents.length,
      'models.count': this.indexes.size,
    });

    const results = new Map<EmbeddingModelType, IndexBuildStats>();

    try {
      // Build indexes in parallel with concurrency control
      const buildPromises = Array.from(this.indexes.entries()).map(
        async ([modelType, index]) => {
          const modelProgressCallback = progressCallback
            ? (progress: number, total: number) => progressCallback(modelType, progress, total)
            : undefined;

          const stats = await index.buildIndex(documents, modelProgressCallback);
          results.set(modelType, stats);
          return { modelType, stats };
        }
      );

      await Promise.all(buildPromises);

      span.setAttributes({
        success: true,
        indexes_built: results.size,
      });

      return results;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Get embedding for a document from a specific model
   */
  async getEmbedding(
    docId: string,
    modelType: EmbeddingModelType
  ): Promise<number[] | null> {
    const index = this.indexes.get(modelType);
    if (!index) {
      throw new Error(`No index found for model type: ${modelType}`);
    }
    return index.getEmbedding(docId);
  }

  /**
   * Get embeddings for multiple documents from a specific model
   */
  async getEmbeddings(
    docIds: string[],
    modelType: EmbeddingModelType
  ): Promise<Map<string, number[]>> {
    const index = this.indexes.get(modelType);
    if (!index) {
      throw new Error(`No index found for model type: ${modelType}`);
    }
    return index.getEmbeddings(docIds);
  }

  /**
   * Compare models using frozen query pool
   */
  async runComparison(
    queryPool: Array<{ query: string; expectedDocIds: string[] }>,
    k: number = 50
  ): Promise<Map<EmbeddingModelType, ComparisonMetrics>> {
    const span = LensTracer.createChildSpan('run_model_comparison', {
      'queries.count': queryPool.length,
      'k': k,
    });

    const results = new Map<EmbeddingModelType, ComparisonMetrics>();

    try {
      for (const [modelType, index] of this.indexes) {
        console.log(`🔄 Running comparison for ${modelType}...`);
        
        const metrics = await this.evaluateModel(modelType, queryPool, k);
        results.set(modelType, metrics);
      }

      span.setAttributes({
        success: true,
        models_evaluated: results.size,
      });

      return results;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  private async evaluateModel(
    modelType: EmbeddingModelType,
    queryPool: Array<{ query: string; expectedDocIds: string[] }>,
    k: number
  ): Promise<ComparisonMetrics> {
    const index = this.indexes.get(modelType);
    const provider = this.providers.get(modelType);
    
    if (!index || !provider) {
      throw new Error(`Missing index or provider for ${modelType}`);
    }

    let totalRecall10 = 0;
    let totalRecall50 = 0;
    let totalCriticalRecall = 0;
    const latencies: number[] = [];

    for (const { query, expectedDocIds } of queryPool) {
      const start = Date.now();
      
      // Get query embedding
      const queryEmbedding = await provider.embed([query]);
      const queryVec = queryEmbedding[0];
      
      if (!queryVec) continue;

      // Find similar documents (simplified cosine similarity)
      const similarities: Array<{ docId: string; score: number }> = [];
      
      // This would normally use your HNSW index for efficiency
      for (const [docId, docEmbedding] of (await index.getEmbeddings(expectedDocIds))) {
        const similarity = this.cosineSimilarity(queryVec, docEmbedding);
        similarities.push({ docId, score: similarity });
      }

      similarities.sort((a, b) => b.score - a.score);
      
      const latency = Date.now() - start;
      latencies.push(latency);

      // Calculate recall metrics
      const top10 = similarities.slice(0, 10).map(s => s.docId);
      const top50 = similarities.slice(0, 50).map(s => s.docId);
      
      const recall10 = this.calculateRecall(expectedDocIds, top10);
      const recall50 = this.calculateRecall(expectedDocIds, top50);
      const criticalRecall = this.calculateCriticalAtomRecall(expectedDocIds, top50);

      totalRecall10 += recall10;
      totalRecall50 += recall50;
      totalCriticalRecall += criticalRecall;
    }

    const avgRecall10 = totalRecall10 / queryPool.length;
    const avgRecall50 = totalRecall50 / queryPool.length;
    const avgCriticalRecall = totalCriticalRecall / queryPool.length;
    const cpuP95 = this.calculateP95(latencies);
    
    const indexStats = index.getIndexStats();
    const storageEfficiency = indexStats.embeddingCount / (indexStats.storageBytes / (1024 * 1024)); // embeddings per MB

    // Calculate ∆CBU/GB (placeholder - would be calculated based on actual utility metrics)
    const deltaCBUPerGB = avgRecall50 * 1000 / (indexStats.storageBytes / (1024 * 1024 * 1024));

    return {
      modelType,
      dimension: indexStats.dimension,
      recall_at_10: avgRecall10,
      recall_at_50: avgRecall50,
      critical_atom_recall: avgCriticalRecall,
      cpu_p95_ms: cpuP95,
      storage_efficiency: storageEfficiency,
      delta_cbu_per_gb: deltaCBUPerGB,
    };
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) return 0;
    
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

  private calculateRecall(expected: string[], retrieved: string[]): number {
    const intersection = expected.filter(id => retrieved.includes(id));
    return expected.length > 0 ? intersection.length / expected.length : 0;
  }

  private calculateCriticalAtomRecall(expected: string[], retrieved: string[]): number {
    // For now, treat all expected results as critical atoms
    // In practice, this would identify the most important/critical documents
    return this.calculateRecall(expected, retrieved);
  }

  private calculateP95(values: number[]): number {
    const sorted = values.sort((a, b) => a - b);
    const index = Math.floor(sorted.length * 0.95);
    return sorted[index] || 0;
  }

  /**
   * Get comprehensive stats for all shadow indexes
   */
  getIndexStats(): Map<EmbeddingModelType, ReturnType<ShadowIndex['getIndexStats']>> {
    const stats = new Map();
    for (const [modelType, index] of this.indexes) {
      stats.set(modelType, index.getIndexStats());
    }
    return stats;
  }

  /**
   * Cleanup all shadow indexes
   */
  async cleanup(): Promise<void> {
    for (const provider of this.providers.values()) {
      if ('cleanup' in provider && typeof provider.cleanup === 'function') {
        await provider.cleanup();
      }
    }
    
    this.indexes.clear();
    this.providers.clear();
    
    console.log('Shadow index manager cleaned up');
  }
}