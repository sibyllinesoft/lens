/**
 * Layer 3: Semantic Rerank Implementation
 * ColBERT-v2/SPLADE semantic reranking for high-precision results
 * Target: 5-15ms (Stage-C) - Neural reranking with vectorized similarity
 */

import type { 
  SemanticIndex, 
  HNSWIndex,
  HNSWLayer,
  HNSWNode,
  Candidate,
  SearchContext 
} from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';
import { SegmentStorage } from '../storage/segments.js';

interface EmbeddingModel {
  encode(text: string): Promise<Float32Array>;
  similarity(a: Float32Array, b: Float32Array): number;
}

interface SemanticContext {
  query_embedding: Float32Array;
  candidates: Candidate[];
  reranked_candidates: Candidate[];
}

/**
 * Simplified semantic reranking engine
 * In production, this would use actual ColBERT/SPLADE models
 */
export class SemanticRerankEngine {
  private semanticIndex: Map<string, Float32Array> = new Map(); // doc_id -> vector
  private hnswIndex: HNSWIndex | null = null;
  private segmentStorage: SegmentStorage;
  private embeddingModel: EmbeddingModel;
  
  // Simplified embedding dimensions for demo
  private readonly EMBEDDING_DIM = 128;
  private readonly MAX_CONNECTIONS = 16;
  private readonly LEVEL_MULTIPLIER = 1.2;

  constructor(segmentStorage: SegmentStorage) {
    this.segmentStorage = segmentStorage;
    this.embeddingModel = new SimpleEmbeddingModel(this.EMBEDDING_DIM);
  }

  /**
   * Initialize the semantic rerank engine
   */
  async initialize(): Promise<void> {
    const span = LensTracer.createChildSpan('semantic_engine_init');
    
    try {
      // Load existing semantic segments
      const segments = this.segmentStorage.listSegments();
      const semanticSegments = segments.filter(id => id.includes('semantic'));
      
      for (const segmentId of semanticSegments) {
        await this.loadSemanticSegment(segmentId);
      }
      
      // Initialize HNSW index if we have vectors
      if (this.semanticIndex.size > 0) {
        await this.buildHNSWIndex();
      }
      
      span.setAttributes({ 
        success: true, 
        segments_loaded: semanticSegments.length,
        vectors_loaded: this.semanticIndex.size,
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
   * Rerank candidates using semantic similarity
   */
  async rerankCandidates(
    candidates: Candidate[], 
    context: SearchContext,
    maxResults: number = 100
  ): Promise<Candidate[]> {
    const span = LensTracer.createChildSpan('semantic_rerank', {
      'candidates.input': candidates.length,
      'search.query': context.query,
      'search.max_results': maxResults,
    });

    try {
      // Skip reranking for very few candidates (only single candidate)
      if (candidates.length <= 1) {
        span.setAttributes({ 
          success: true, 
          candidates_output: candidates.length,
          skipped: true,
          reason: 'too_few_candidates'
        });
        return candidates;
      }

      // Generate query embedding
      const queryEmbedding = await this.embeddingModel.encode(context.query);
      
      // Calculate semantic similarities
      const rerankingStart = Date.now();
      const rerankedCandidates: (Candidate & { semantic_score: number })[] = [];
      
      for (const candidate of candidates) {
        // Get or generate document embedding
        let docEmbedding = this.semanticIndex.get(candidate.doc_id);
        
        // Try base document ID if full doc_id not found (format: base_id:line:col)
        if (!docEmbedding) {
          const baseDocId = candidate.doc_id.split(':')[0];
          if (baseDocId) {
            docEmbedding = this.semanticIndex.get(baseDocId);
          }
        }
        
        if (!docEmbedding) {
          // Fallback: generate embedding from context
          const contextText = candidate.context || candidate.file_path;
          docEmbedding = await this.embeddingModel.encode(contextText);
          // Cache using full doc_id for future lookups
          this.semanticIndex.set(candidate.doc_id, docEmbedding);
        }
        
        // Calculate semantic similarity
        const semanticScore = this.embeddingModel.similarity(queryEmbedding, docEmbedding);
        
        rerankedCandidates.push({
          ...candidate,
          semantic_score: semanticScore,
        });
      }
      
      const rerankingLatency = Date.now() - rerankingStart;
      
      // Combine lexical and semantic scores
      const finalCandidates = rerankedCandidates.map(candidate => {
        // Weighted combination: 70% original score, 30% semantic score
        const combinedScore = (candidate.score * 0.7) + (candidate.semantic_score * 0.3);
        
        // Boost candidates with semantic match
        const boost = candidate.semantic_score > 0.5 ? 0.1 : 0;
        
        return {
          ...candidate,
          score: Math.min(1.0, combinedScore + boost),
          match_reasons: candidate.semantic_score > 0.6 ? 
            [...candidate.match_reasons, 'semantic'] : 
            candidate.match_reasons,
        };
      });
      
      // Sort by final combined score
      finalCandidates.sort((a, b) => b.score - a.score);
      
      const results = finalCandidates.slice(0, maxResults).map(({semantic_score, ...candidate}) => candidate as Candidate);
      
      span.setAttributes({
        success: true,
        candidates_output: results.length,
        reranking_latency_ms: rerankingLatency,
        avg_semantic_score: rerankedCandidates.reduce((sum, c) => sum + c.semantic_score, 0) / rerankedCandidates.length,
      });
      
      return results;
      
    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      
      // Fallback: return original candidates on error
      console.warn(`Semantic reranking failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      return candidates.slice(0, maxResults);
      
    } finally {
      span.end();
    }
  }

  /**
   * Index a document for semantic search
   */
  async indexDocument(
    docId: string, 
    content: string, 
    filePath: string
  ): Promise<void> {
    const span = LensTracer.createChildSpan('index_document_semantic', {
      'doc.id': docId,
      'doc.size': content.length,
      'file.path': filePath,
    });

    try {
      // Generate embedding for document content
      const embedding = await this.embeddingModel.encode(content);
      this.semanticIndex.set(docId, embedding);
      
      // Update HNSW index if it exists
      if (this.hnswIndex) {
        await this.addToHNSWIndex(docId, embedding);
      } else if (this.semanticIndex.size > 0) {
        // Build HNSW index if it doesn't exist and we have vectors
        await this.buildHNSWIndex();
      }
      
      span.setAttributes({ 
        success: true,
        embedding_dim: embedding.length,
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
   * Find similar documents using HNSW
   */
  async findSimilarDocuments(
    queryEmbedding: Float32Array, 
    k: number = 50
  ): Promise<Array<{ doc_id: string; score: number }>> {
    const span = LensTracer.createChildSpan('find_similar_docs', {
      'query.embedding_dim': queryEmbedding.length,
      'search.k': k,
    });

    try {
      const similarities: Array<{ doc_id: string; score: number }> = [];
      
      // Simple brute-force search (HNSW would be more efficient)
      for (const [docId, docEmbedding] of this.semanticIndex) {
        const similarity = this.embeddingModel.similarity(queryEmbedding, docEmbedding);
        similarities.push({ doc_id: docId, score: similarity });
      }
      
      // Sort by similarity and return top-k
      similarities.sort((a, b) => b.score - a.score);
      const results = similarities.slice(0, k);
      
      span.setAttributes({
        success: true,
        candidates_found: similarities.length,
        results_returned: results.length,
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
   * Build HNSW index for efficient similarity search
   */
  private async buildHNSWIndex(): Promise<void> {
    const span = LensTracer.createChildSpan('build_hnsw_index');

    try {
      const vectors = Array.from(this.semanticIndex.entries());
      const numVectors = vectors.length;
      
      if (numVectors === 0) {
        span.setAttributes({ success: true, vectors: 0, skipped: true });
        return;
      }

      // Initialize HNSW structure
      this.hnswIndex = {
        layers: [],
        entry_point: 0,
        max_connections: this.MAX_CONNECTIONS,
        level_multiplier: this.LEVEL_MULTIPLIER,
      };

      // Simplified HNSW construction (production would use proper algorithm)
      const layer0: HNSWLayer = {
        level: 0,
        nodes: new Map(),
      };

      // Add all vectors to layer 0
      for (let i = 0; i < vectors.length; i++) {
        const vectorEntry = vectors[i];
        if (!vectorEntry) continue;
        const [docId, embedding] = vectorEntry;
        
        const node: HNSWNode = {
          id: i,
          vector: embedding,
          connections: new Set(),
        };
        
        // Connect to nearest neighbors (simplified)
        const connections = this.findNearestNeighbors(
          embedding, 
          vectors.slice(0, i), 
          Math.min(this.MAX_CONNECTIONS, i)
        );
        
        connections.forEach(connIdx => node.connections.add(connIdx));
        
        layer0.nodes.set(i, node);
      }

      this.hnswIndex.layers.push(layer0);
      
      span.setAttributes({
        success: true,
        vectors: numVectors,
        layers: 1,
        avg_connections: Array.from(layer0.nodes.values()).reduce(
          (sum, node) => sum + node.connections.size, 0
        ) / numVectors,
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
   * Find nearest neighbors for HNSW construction
   */
  private findNearestNeighbors(
    queryVector: Float32Array,
    candidates: Array<[string, Float32Array]>,
    k: number
  ): number[] {
    const similarities = candidates.map(([docId, vector], index) => ({
      index,
      similarity: this.embeddingModel.similarity(queryVector, vector),
    }));

    similarities.sort((a, b) => b.similarity - a.similarity);
    
    return similarities.slice(0, k).map(item => item.index);
  }

  /**
   * Add vector to existing HNSW index
   */
  private async addToHNSWIndex(docId: string, embedding: Float32Array): Promise<void> {
    // Simplified - in production would properly update HNSW structure
    if (!this.hnswIndex || this.hnswIndex.layers.length === 0) {
      return;
    }

    const layer0 = this.hnswIndex.layers[0];
    if (!layer0) {
      return;
    }
    
    const nodeId = layer0.nodes.size;
    
    const node: HNSWNode = {
      id: nodeId,
      vector: embedding,
      connections: new Set(),
    };
    
    // Connect to some existing nodes for basic functionality
    const existingNodes = Array.from(layer0.nodes.values());
    const maxConnections = Math.min(this.MAX_CONNECTIONS, existingNodes.length);
    
    for (let i = 0; i < maxConnections; i++) {
      const existingNode = existingNodes[i];
      if (!existingNode) continue;
      const similarity = this.embeddingModel.similarity(embedding, existingNode.vector);
      if (similarity > 0.1) { // Only connect if reasonably similar
        node.connections.add(existingNode.id);
        existingNode.connections.add(nodeId);
      }
    }
    
    layer0.nodes.set(nodeId, node);
  }

  /**
   * Load semantic vectors from segment
   */
  private async loadSemanticSegment(segmentId: string): Promise<void> {
    const span = LensTracer.createChildSpan('load_semantic_segment', {
      'segment.id': segmentId,
    });

    try {
      const segment = await this.segmentStorage.openSegment(segmentId, true);
      const data = await this.segmentStorage.readFromSegment(segmentId, 0, segment.size);
      
      // Parse semantic data (simplified - would be binary format in production)
      const dataString = data.toString('utf8').trim();
      
      // Skip empty or invalid segments
      if (!dataString || dataString.length < 2) {
        span.setAttributes({ success: true, skipped: true, reason: 'empty_segment' });
        return;
      }
      
      let semanticData;
      try {
        semanticData = JSON.parse(dataString);
      } catch (parseError) {
        // Skip segments with invalid JSON data
        span.setAttributes({ success: true, skipped: true, reason: 'invalid_json' });
        return;
      }
      
      if (semanticData && semanticData.vectors) {
        for (const [docId, vectorArray] of Object.entries(semanticData.vectors)) {
          if (Array.isArray(vectorArray)) {
            const vector = new Float32Array(vectorArray as number[]);
            this.semanticIndex.set(docId, vector);
          }
        }
      }
      
      span.setAttributes({ success: true });
      
    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Get semantic search statistics
   */
  getStats(): { vectors: number; hnsw_layers: number; avg_dim: number } {
    const vectors = this.semanticIndex.size;
    const hnswLayers = this.hnswIndex?.layers.length || 0;
    
    let totalDim = 0;
    for (const vector of this.semanticIndex.values()) {
      totalDim += vector.length;
    }
    const avgDim = vectors > 0 ? totalDim / vectors : 0;
    
    return {
      vectors,
      hnsw_layers: hnswLayers,
      avg_dim: Math.round(avgDim),
    };
  }

  /**
   * Cleanup resources
   */
  async shutdown(): Promise<void> {
    this.semanticIndex.clear();
    this.hnswIndex = null;
    console.log('Semantic rerank engine shut down');
  }
}

/**
 * Simple embedding model for demonstration
 * In production, use actual ColBERT/SPLADE models
 */
class SimpleEmbeddingModel implements EmbeddingModel {
  private readonly dimension: number;
  private readonly vocab: Map<string, number> = new Map();
  
  constructor(dimension: number = 128) {
    this.dimension = dimension;
    this.initializeVocab();
  }

  /**
   * Generate embedding for text
   */
  async encode(text: string): Promise<Float32Array> {
    const tokens = this.tokenize(text);
    const embedding = new Float32Array(this.dimension);
    
    // Simple TF-IDF-like embedding
    const tokenCounts = new Map<string, number>();
    for (const token of tokens) {
      tokenCounts.set(token, (tokenCounts.get(token) || 0) + 1);
    }
    
    for (const [token, count] of tokenCounts) {
      const tokenId = this.vocab.get(token);
      if (tokenId !== undefined) {
        const index = tokenId % this.dimension;
        embedding[index]! += count * Math.log(1 + count);
      }
    }
    
    // Normalize vector
    const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    if (norm > 0) {
      for (let i = 0; i < embedding.length; i++) {
        embedding[i]! /= norm;
      }
    }
    
    return embedding;
  }

  /**
   * Calculate cosine similarity between vectors (assumes vectors are already normalized)
   */
  similarity(a: Float32Array, b: Float32Array): number {
    if (a.length !== b.length) {
      return 0;
    }
    
    let dotProduct = 0;
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i]! * b[i]!;
    }
    
    // Since vectors are already normalized in encode(), dot product = cosine similarity
    return dotProduct;
  }

  /**
   * Initialize vocabulary with common programming terms
   */
  private initializeVocab(): void {
    const commonTerms = [
      'function', 'class', 'interface', 'type', 'variable', 'const', 'let', 'var',
      'import', 'export', 'return', 'if', 'else', 'for', 'while', 'try', 'catch',
      'async', 'await', 'promise', 'callback', 'event', 'handler', 'component',
      'service', 'api', 'endpoint', 'request', 'response', 'data', 'model',
      'controller', 'view', 'template', 'config', 'settings', 'utils', 'helpers',
      'test', 'spec', 'mock', 'stub', 'assert', 'expect', 'describe', 'it',
      // Math and arithmetic terms
      'add', 'subtract', 'multiply', 'divide', 'sum', 'product', 'calculate', 'math',
      'operation', 'compute',
      'number', 'numbers', 'value', 'values', 'result', 'results',
      // String operations  
      'string', 'text', 'concat', 'split', 'join', 'replace', 'substring', 'length',
      'char', 'character', 'trim', 'lowercase', 'uppercase', 'search', 'match',
      // Array operations
      'array', 'list', 'push', 'pop', 'shift', 'unshift', 'slice', 'splice', 'filter',
      'map', 'reduce', 'find', 'includes', 'indexOf', 'sort', 'reverse', 'forEach',
      // HTTP operations
      'http', 'https', 'get', 'post', 'put', 'delete', 'fetch', 'request', 'url',
      'header', 'body', 'json', 'xml', 'rest', 'graphql', 'endpoint', 'client', 'server'
    ];
    
    commonTerms.forEach((term, index) => {
      this.vocab.set(term, index);
    });
  }

  /**
   * Simple tokenization with term normalization
   */
  private tokenize(text: string): string[] {
    // First split camelCase and handle punctuation
    const camelCaseSplit = text
      .replace(/([a-z])([A-Z])/g, '$1 $2')  // Split camelCase
      .replace(/[^a-zA-Z0-9\s]/g, ' ')      // Replace non-alphanumeric with spaces
      .toLowerCase();
      
    const tokens = camelCaseSplit
      .split(/\s+/)
      .filter(token => token.length > 1);
      
    // Normalize related terms to common forms
    return tokens.map(token => this.normalizeToken(token));
  }
  
  /**
   * Normalize tokens to canonical forms for better semantic matching
   */
  private normalizeToken(token: string): string {
    const normalizations: Record<string, string> = {
      // Math operations
      'addition': 'add',
      'subtraction': 'subtract',
      'multiplication': 'multiply',
      'division': 'divide',
      'arithmetic': 'math',
      'operations': 'operation',
      'calculation': 'calculate',
      'computing': 'compute',
      // String operations
      'concatenation': 'concat',
      'concatenate': 'concat',
      'splitting': 'split',
      'joining': 'join',
      // Array operations
      'filtering': 'filter',
      'mapping': 'map',
      'reducing': 'reduce',
      'finding': 'find',
      'searching': 'search',
      // HTTP operations
      'getting': 'get',
      'posting': 'post',
      'putting': 'put',
      'deleting': 'delete',
      'requesting': 'request',
      'responding': 'response',
    };
    
    return normalizations[token] || token;
  }
}