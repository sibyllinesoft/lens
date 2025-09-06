/**
 * Enhanced Semantic Rerank Engine - Phase B3 Integration
 * Integrates isotonic calibration, confidence-aware reranking, and optimized HNSW
 * Target: 12ms → 6-8ms (~40% improvement) for Stage-C with quality preservation
 */

import type { 
  SemanticIndex, 
  HNSWIndex,
  Candidate,
  SearchContext 
} from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';
import { SegmentStorage } from '../storage/segments.js';
import { shouldApplySemanticReranking, explainSemanticDecision } from '../core/query-classifier.js';
import { IsotonicCalibratedReranker } from '../core/isotonic-reranker.js';
import { OptimizedHNSWIndex } from '../core/optimized-hnsw.js';

interface EmbeddingModel {
  encode(text: string): Promise<Float32Array>;
  similarity(a: Float32Array, b: Float32Array): number;
}

interface EnhancedSemanticConfig {
  // Stage-C optimization flags
  enableIsotonicCalibration: boolean;
  enableConfidenceGating: boolean;
  enableOptimizedHNSW: boolean;
  
  // Performance targets
  maxLatencyMs: number;         // 6-8ms target for Stage-C
  qualityThreshold: number;     // Maximum nDCG degradation (0.5%)
  
  // Isotonic calibration settings
  calibrationConfig: {
    enabled: boolean;
    minCalibrationData: number;
    confidenceCutoff: number;
    updateFreq: number;
  };
  
  // HNSW optimization settings
  hnswConfig: {
    K: number;                  // Fixed at 150
    efSearch: number;           // Tunable for performance
    autoTune: boolean;          // Auto-tune efSearch
  };
  
  // Feature flags for gradual rollout
  featureFlags: {
    stageCOptimizations: boolean;
    advancedCalibration: boolean;
    experimentalHNSW: boolean;
  };
}

/**
 * Enhanced semantic reranking engine with Phase B3 optimizations
 * Integrates isotonic calibration, confidence-aware processing, and optimized HNSW
 */
export class EnhancedSemanticRerankEngine {
  private semanticIndex: Map<string, Float32Array> = new Map();
  private optimizedHNSW: OptimizedHNSWIndex;
  private isotonicReranker: IsotonicCalibratedReranker;
  private segmentStorage: SegmentStorage;
  private embeddingModel: EmbeddingModel;
  private config: EnhancedSemanticConfig;
  
  // Performance tracking
  private performanceMetrics: {
    avgLatencyMs: number;
    qualityScore: number;
    throughputQPS: number;
    calibrationAccuracy: number;
    hnswEfficiency: number;
  } = {
    avgLatencyMs: 0,
    qualityScore: 0,
    throughputQPS: 0,
    calibrationAccuracy: 0,
    hnswEfficiency: 0
  };

  private queryEmbeddingCache: Map<string, Float32Array> = new Map();
  private readonly MAX_QUERY_CACHE_SIZE = 1000;
  private readonly EMBEDDING_DIM = 128;

  constructor(
    segmentStorage: SegmentStorage,
    config: Partial<EnhancedSemanticConfig> = {}
  ) {
    this.segmentStorage = segmentStorage;
    
    this.config = {
      enableIsotonicCalibration: config.enableIsotonicCalibration ?? true,
      enableConfidenceGating: config.enableConfidenceGating ?? true,
      enableOptimizedHNSW: config.enableOptimizedHNSW ?? true,
      maxLatencyMs: config.maxLatencyMs ?? 8, // 8ms target for Stage-C
      qualityThreshold: config.qualityThreshold ?? 0.005, // 0.5% max degradation
      
      calibrationConfig: {
        enabled: config.calibrationConfig?.enabled ?? true,
        minCalibrationData: config.calibrationConfig?.minCalibrationData ?? 50,
        confidenceCutoff: config.calibrationConfig?.confidenceCutoff ?? 0.12,
        updateFreq: config.calibrationConfig?.updateFreq ?? 100,
        ...config.calibrationConfig
      },
      
      hnswConfig: {
        K: 150, // Fixed per B3 requirements
        efSearch: config.hnswConfig?.efSearch ?? 64,
        autoTune: config.hnswConfig?.autoTune ?? true,
        ...config.hnswConfig
      },
      
      featureFlags: {
        stageCOptimizations: config.featureFlags?.stageCOptimizations ?? true,
        advancedCalibration: config.featureFlags?.advancedCalibration ?? true,
        experimentalHNSW: config.featureFlags?.experimentalHNSW ?? false,
        ...config.featureFlags
      }
    };

    // Initialize components
    this.embeddingModel = new SimpleEmbeddingModel(this.EMBEDDING_DIM);
    
    this.optimizedHNSW = new OptimizedHNSWIndex({
      K: this.config.hnswConfig.K,
      efSearch: this.config.hnswConfig.efSearch,
      qualityThreshold: this.config.qualityThreshold,
      performanceTarget: 0.4 // 40% improvement target
    });

    this.isotonicReranker = new IsotonicCalibratedReranker({
      enabled: this.config.enableIsotonicCalibration,
      minCalibrationData: this.config.calibrationConfig.minCalibrationData,
      confidenceCutoff: this.config.calibrationConfig.confidenceCutoff,
      maxLatencyMs: this.config.maxLatencyMs,
      calibrationUpdateFreq: this.config.calibrationConfig.updateFreq
    });

    console.log(`🚀 EnhancedSemanticRerankEngine initialized with B3 optimizations`);
    console.log(`  - Isotonic calibration: ${this.config.enableIsotonicCalibration}`);
    console.log(`  - Confidence gating: ${this.config.enableConfidenceGating}`);
    console.log(`  - Optimized HNSW: ${this.config.enableOptimizedHNSW}`);
    console.log(`  - Target latency: ${this.config.maxLatencyMs}ms`);
  }

  /**
   * Initialize the enhanced semantic engine
   */
  async initialize(): Promise<void> {
    const span = LensTracer.createChildSpan('enhanced_semantic_init');
    
    try {
      // Load existing semantic segments
      const segments = this.segmentStorage.listSegments();
      const semanticSegments = segments.filter(id => id.includes('semantic'));
      
      for (const segmentId of semanticSegments) {
        await this.loadSemanticSegment(segmentId);
      }
      
      // Build optimized HNSW index if we have vectors
      if (this.semanticIndex.size > 0 && this.config.enableOptimizedHNSW) {
        console.log(`🚀 Building optimized HNSW index for ${this.semanticIndex.size} vectors...`);
        await this.optimizedHNSW.buildIndex(this.semanticIndex);
        
        // Auto-tune efSearch if enabled
        if (this.config.hnswConfig.autoTune && this.semanticIndex.size > 100) {
          await this.autoTuneHNSWParameters();
        }
      }
      
      span.setAttributes({ 
        success: true, 
        segments_loaded: semanticSegments.length,
        vectors_loaded: this.semanticIndex.size,
        hnsw_enabled: this.config.enableOptimizedHNSW,
        isotonic_enabled: this.config.enableIsotonicCalibration
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
   * Enhanced rerank candidates with B3 optimizations
   */
  async rerankCandidates(
    candidates: Candidate[], 
    context: SearchContext,
    maxResults: number = 100
  ): Promise<Candidate[]> {
    const span = LensTracer.createChildSpan('enhanced_semantic_rerank', {
      'candidates.input': candidates.length,
      'search.query': context.query,
      'search.max_results': maxResults,
      'optimizations.enabled': this.config.featureFlags.stageCOptimizations
    });

    const startTime = Date.now();

    try {
      // Emergency latency cutoff
      const checkLatency = () => {
        const elapsed = Date.now() - startTime;
        if (elapsed > this.config.maxLatencyMs) {
          throw new Error(`Stage-C latency budget exceeded: ${elapsed}ms > ${this.config.maxLatencyMs}ms`);
        }
        return elapsed;
      };

      // Check if semantic reranking should be applied
      const semanticConfig = {
        nlThreshold: this.config.calibrationConfig.confidenceCutoff,
        minCandidates: 10,
        maxCandidates: Math.min(200, candidates.length),
        confidenceCutoff: this.config.calibrationConfig.confidenceCutoff,
      };

      if (!shouldApplySemanticReranking(context.query, candidates.length, context.mode, semanticConfig)) {
        const reason = explainSemanticDecision(context.query, candidates.length, context.mode);
        span.setAttributes({ 
          success: true, 
          candidates_output: candidates.length,
          skipped: true,
          reason: reason
        });
        return candidates.slice(0, maxResults);
      }

      checkLatency();

      // Phase B3 Enhancement Path
      if (this.config.featureFlags.stageCOptimizations) {
        return await this.applyB3Optimizations(candidates, context, maxResults, checkLatency);
      }
      
      // Fallback to basic semantic reranking
      return await this.applyBasicSemanticReranking(candidates, context, maxResults, checkLatency);

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      
      console.warn(`Enhanced semantic reranking failed: ${errorMsg}, falling back to original ordering`);
      return candidates.slice(0, maxResults);
      
    } finally {
      span.end();
    }
  }

  /**
   * Apply Phase B3 optimizations: isotonic calibration + confidence gating + optimized HNSW
   */
  private async applyB3Optimizations(
    candidates: Candidate[],
    context: SearchContext,
    maxResults: number,
    checkLatency: () => number
  ): Promise<Candidate[]> {
    const span = LensTracer.createChildSpan('apply_b3_optimizations');

    try {
      // Step 1: Isotonic calibrated reranking with confidence gating
      let processedCandidates = candidates;
      
      if (this.config.enableIsotonicCalibration) {
        // Convert candidates to SearchHits for reranker
        const searchHits = this.convertCandidatesToSearchHits(candidates);
        const rerankedHits = await this.isotonicReranker.rerank(searchHits, context);
        processedCandidates = this.convertSearchHitsToCandidates(rerankedHits);
        
        checkLatency();
      }

      // Step 2: Enhanced semantic similarity with optimized HNSW
      if (this.config.enableOptimizedHNSW && this.semanticIndex.size > 0) {
        processedCandidates = await this.applyOptimizedSemanticSimilarity(
          processedCandidates,
          context,
          checkLatency
        );
      }

      // Step 3: Final ranking combination and quality preservation
      const finalResults = this.combineRankingSignals(processedCandidates, context);

      // Record performance metrics
      const latency = checkLatency();
      this.updatePerformanceMetrics(latency, finalResults.length, context);

      span.setAttributes({
        success: true,
        candidates_processed: finalResults.length,
        isotonic_applied: this.config.enableIsotonicCalibration,
        hnsw_applied: this.config.enableOptimizedHNSW,
        final_latency_ms: latency
      });

      return finalResults.slice(0, maxResults);

    } catch (error) {
      span.recordException(error as Error);
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Apply optimized semantic similarity using enhanced HNSW
   */
  private async applyOptimizedSemanticSimilarity(
    candidates: Candidate[],
    context: SearchContext,
    checkLatency: () => number
  ): Promise<Candidate[]> {
    // Generate or retrieve cached query embedding
    const queryEmbedding = await this.getOrCacheQueryEmbedding(context.query);
    checkLatency();

    // Use optimized HNSW for similarity search
    const hnswResults = await this.optimizedHNSW.search(
      queryEmbedding,
      Math.min(150, candidates.length * 2) // Search more than we need for quality
    );
    checkLatency();

    // Create similarity lookup
    const similarityMap = new Map<string, number>();
    for (const result of hnswResults) {
      similarityMap.set(result.doc_id, result.score);
    }

    // Enhance candidates with optimized semantic scores
    const enhancedCandidates = candidates.map(candidate => {
      const baseDocId = candidate.doc_id.split(':')[0]!;
      const semanticScore = similarityMap.get(candidate.doc_id) || 
                           similarityMap.get(baseDocId) || 
                           this.calculateFallbackSimilarity(candidate, queryEmbedding);

      // Combine original score with semantic score (optimized weighting)
      const combinedScore = this.combineScoresOptimized(candidate.score, semanticScore, context);

      return {
        ...candidate,
        score: combinedScore,
        match_reasons: semanticScore > 0.6 ? 
          [...candidate.match_reasons, 'semantic' as const] : 
          candidate.match_reasons,
      };
    });

    // Sort by enhanced scores
    enhancedCandidates.sort((a, b) => b.score - a.score);
    
    return enhancedCandidates;
  }

  /**
   * Optimized score combination with learned weights
   */
  private combineScoresOptimized(
    originalScore: number,
    semanticScore: number,
    context: SearchContext
  ): number {
    // Context-aware weighting based on query characteristics
    const queryLength = context.query.split(' ').length;
    const hasNaturalLanguage = /\b(how|what|where|when|why|find|show)\b/i.test(context.query);
    
    // Adaptive weighting: semantic matters more for NL queries
    const semanticWeight = hasNaturalLanguage ? 0.4 : 0.25;
    const originalWeight = 1 - semanticWeight;
    
    // Length-based boost for longer queries
    const lengthBoost = Math.min(0.1, (queryLength - 2) * 0.02);
    
    const combinedScore = (originalScore * originalWeight) + (semanticScore * semanticWeight);
    const boostedScore = Math.min(1.0, combinedScore + (semanticScore > 0.7 ? lengthBoost : 0));
    
    return boostedScore;
  }

  /**
   * Combine multiple ranking signals for final results
   */
  private combineRankingSignals(candidates: Candidate[], context: SearchContext): Candidate[] {
    // Apply final quality-preserving adjustments
    return candidates.map(candidate => ({
      ...candidate,
      score: this.applyQualityPreservation(candidate.score, candidate, context)
    }));
  }

  /**
   * Apply quality preservation logic to maintain nDCG standards
   */
  private applyQualityPreservation(score: number, candidate: Candidate, context: SearchContext): number {
    // Boost high-confidence matches to preserve quality
    if (candidate.match_reasons.includes('exact')) {
      return Math.min(1.0, score + 0.05);
    }
    
    if (candidate.match_reasons.includes('symbol') && candidate.symbol_kind) {
      return Math.min(1.0, score + 0.03);
    }
    
    return score;
  }

  /**
   * Fallback to basic semantic reranking when optimizations are disabled
   */
  private async applyBasicSemanticReranking(
    candidates: Candidate[],
    context: SearchContext,
    maxResults: number,
    checkLatency: () => number
  ): Promise<Candidate[]> {
    // Implementation similar to original SemanticRerankEngine
    const queryEmbedding = await this.getOrCacheQueryEmbedding(context.query);
    checkLatency();

    const rerankedCandidates = [];
    
    for (const candidate of candidates) {
      let docEmbedding = this.semanticIndex.get(candidate.doc_id);
        
      if (!docEmbedding) {
        const baseDocId = candidate.doc_id.split(':')[0];
        if (baseDocId) {
          docEmbedding = this.semanticIndex.get(baseDocId);
        }
      }
      
      if (!docEmbedding) {
        const contextText = candidate.context || candidate.file_path;
        docEmbedding = await this.embeddingModel.encode(contextText);
        this.semanticIndex.set(candidate.doc_id, docEmbedding);
      }
      
      const semanticScore = this.embeddingModel.similarity(queryEmbedding, docEmbedding);
      const combinedScore = (candidate.score * 0.7) + (semanticScore * 0.3);
      const boost = semanticScore > 0.5 ? 0.1 : 0;
      
      rerankedCandidates.push({
        ...candidate,
        score: Math.min(1.0, combinedScore + boost),
        match_reasons: semanticScore > 0.6 ? 
          [...candidate.match_reasons, 'semantic' as const] : 
          candidate.match_reasons,
      });
      
      checkLatency();
    }
    
    rerankedCandidates.sort((a, b) => b.score - a.score);
    return rerankedCandidates.slice(0, maxResults);
  }

  /**
   * Auto-tune HNSW parameters for optimal performance
   */
  private async autoTuneHNSWParameters(): Promise<void> {
    console.log('🔧 Auto-tuning HNSW efSearch parameter...');
    
    try {
      // Generate test queries from existing documents
      const testQueries = await this.generateTestQueries();
      
      if (testQueries.length > 0) {
        const optimalEfSearch = await this.optimizedHNSW.tuneEfSearch(testQueries, []);
        console.log(`🎯 Auto-tuning complete: efSearch=${optimalEfSearch}`);
      }
    } catch (error) {
      console.warn(`Auto-tuning failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Generate test queries for HNSW parameter tuning
   */
  private async generateTestQueries(): Promise<Float32Array[]> {
    const queries: Float32Array[] = [];
    const sampleTexts = [
      'function calculate sum',
      'class user service',
      'http request handler',
      'string manipulation utility',
      'array sorting algorithm',
      'database connection pool',
      'authentication middleware',
      'error handling logic',
      'configuration parser',
      'logging framework'
    ];

    for (const text of sampleTexts) {
      const embedding = await this.embeddingModel.encode(text);
      queries.push(embedding);
    }

    return queries;
  }

  /**
   * Utility methods
   */
  private async getOrCacheQueryEmbedding(query: string): Promise<Float32Array> {
    const cached = this.queryEmbeddingCache.get(query);
    if (cached) {
      return cached;
    }
    
    const embedding = await this.embeddingModel.encode(query);
    
    if (this.queryEmbeddingCache.size >= this.MAX_QUERY_CACHE_SIZE) {
      const firstKey = this.queryEmbeddingCache.keys().next().value;
      if (firstKey) {
        this.queryEmbeddingCache.delete(firstKey);
      }
    }
    
    this.queryEmbeddingCache.set(query, embedding);
    return embedding;
  }

  private calculateFallbackSimilarity(candidate: Candidate, queryEmbedding: Float32Array): number {
    // Simple fallback when HNSW doesn't have the document
    const contextText = candidate.context || candidate.file_path || '';
    // This would be async in reality, but for fallback we use a simple heuristic
    const textLength = contextText.length;
    const queryLength = queryEmbedding.length;
    
    // Simple similarity estimation based on text characteristics
    return Math.min(0.5, textLength / Math.max(100, queryLength * 10));
  }

  private convertCandidatesToSearchHits(candidates: Candidate[]): any[] {
    return candidates.map(candidate => ({
      doc_id: candidate.doc_id,
      file: candidate.file_path,
      line: candidate.line,
      col: candidate.col,
      score: candidate.score,
      snippet: candidate.context || candidate.snippet,
      why: candidate.match_reasons.join(','),
      symbol_kind: candidate.symbol_kind,
      ast_path: candidate.ast_path
    }));
  }

  private convertSearchHitsToCandidates(hits: any[]): Candidate[] {
    return hits.map(hit => ({
      doc_id: hit.doc_id || hit.id,
      file_path: hit.file || hit.file_path,
      line: hit.line || 1,
      col: hit.col || 1,
      score: hit.score || 0,
      match_reasons: (hit.why || '').split(',').filter(Boolean) as any[],
      context: hit.snippet || hit.context,
      symbol_kind: hit.symbol_kind,
      ast_path: hit.ast_path
    }));
  }

  private updatePerformanceMetrics(latency: number, resultCount: number, context: SearchContext): void {
    // Simple exponential moving average
    const alpha = 0.1;
    this.performanceMetrics.avgLatencyMs = (1 - alpha) * this.performanceMetrics.avgLatencyMs + alpha * latency;
    this.performanceMetrics.throughputQPS = 1000 / this.performanceMetrics.avgLatencyMs;
    
    // Estimate quality based on result characteristics
    const estimatedQuality = Math.min(1.0, resultCount / Math.max(1, context.k || 10));
    this.performanceMetrics.qualityScore = (1 - alpha) * this.performanceMetrics.qualityScore + alpha * estimatedQuality;
  }

  private async loadSemanticSegment(segmentId: string): Promise<void> {
    // Implementation similar to original, but optimized for performance
    const span = LensTracer.createChildSpan('load_semantic_segment_enhanced', {
      'segment.id': segmentId,
    });

    try {
      const segment = await this.segmentStorage.openSegment(segmentId, true);
      const data = await this.segmentStorage.readFromSegment(segmentId, 0, segment.size);
      
      const dataString = data.toString('utf8').trim();
      
      if (!dataString || dataString.length < 2) {
        span.setAttributes({ success: true, skipped: true, reason: 'empty_segment' });
        return;
      }
      
      let semanticData;
      try {
        semanticData = JSON.parse(dataString);
      } catch (parseError) {
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
   * Index document with enhanced processing
   */
  async indexDocument(docId: string, content: string, filePath: string): Promise<void> {
    const span = LensTracer.createChildSpan('index_document_enhanced', {
      'doc.id': docId,
      'doc.size': content.length,
      'file.path': filePath,
    });

    try {
      // Generate embedding
      const embedding = await this.embeddingModel.encode(content);
      this.semanticIndex.set(docId, embedding);
      
      // Add to optimized HNSW if enabled
      if (this.config.enableOptimizedHNSW) {
        // Rebuild HNSW periodically for optimal performance
        const shouldRebuild = this.semanticIndex.size % 100 === 0 && this.semanticIndex.size > 0;
        if (shouldRebuild) {
          await this.optimizedHNSW.buildIndex(this.semanticIndex);
        }
      }
      
      span.setAttributes({ 
        success: true,
        embedding_dim: embedding.length,
        total_vectors: this.semanticIndex.size
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
   * Get comprehensive statistics
   */
  getStats() {
    return {
      config: this.config,
      vectors: this.semanticIndex.size,
      performance: this.performanceMetrics,
      isotonic_reranker: this.isotonicReranker.getStats(),
      optimized_hnsw: this.optimizedHNSW.getStats(),
      query_cache_size: this.queryEmbeddingCache.size
    };
  }

  /**
   * Update configuration for A/B testing and optimization
   */
  async updateConfig(newConfig: Partial<EnhancedSemanticConfig>): Promise<void> {
    const span = LensTracer.createChildSpan('update_enhanced_semantic_config');

    try {
      this.config = { ...this.config, ...newConfig };

      // Update components
      if (newConfig.calibrationConfig) {
        this.isotonicReranker.updateConfig(newConfig.calibrationConfig);
      }

      if (newConfig.hnswConfig) {
        this.optimizedHNSW.updateConfig(newConfig.hnswConfig);
      }

      span.setAttributes({ success: true });
      console.log(`🚀 Enhanced semantic config updated`);

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Cleanup resources
   */
  async shutdown(): Promise<void> {
    this.semanticIndex.clear();
    this.queryEmbeddingCache.clear();
    console.log('Enhanced semantic rerank engine shut down');
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

  async encode(text: string): Promise<Float32Array> {
    const tokens = this.tokenize(text);
    const embedding = new Float32Array(this.dimension);
    
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

  similarity(a: Float32Array, b: Float32Array): number {
    if (a.length !== b.length) return 0;
    
    let dotProduct = 0;
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i]! * b[i]!;
    }
    
    return dotProduct;
  }

  private initializeVocab(): void {
    const commonTerms = [
      'function', 'class', 'interface', 'type', 'variable', 'const', 'let', 'var',
      'import', 'export', 'return', 'if', 'else', 'for', 'while', 'try', 'catch',
      'async', 'await', 'promise', 'callback', 'event', 'handler', 'component',
      'service', 'api', 'endpoint', 'request', 'response', 'data', 'model',
      'controller', 'view', 'template', 'config', 'settings', 'utils', 'helpers',
      'test', 'spec', 'mock', 'stub', 'assert', 'expect', 'describe', 'it',
      'add', 'subtract', 'multiply', 'divide', 'sum', 'product', 'calculate', 'math',
      'operation', 'compute', 'number', 'numbers', 'value', 'values', 'result', 'results',
      'string', 'text', 'concat', 'split', 'join', 'replace', 'substring', 'length',
      'char', 'character', 'trim', 'lowercase', 'uppercase', 'search', 'match',
      'array', 'list', 'push', 'pop', 'shift', 'unshift', 'slice', 'splice', 'filter',
      'map', 'reduce', 'find', 'includes', 'indexOf', 'sort', 'reverse', 'forEach',
      'http', 'https', 'get', 'post', 'put', 'delete', 'fetch', 'request', 'url',
      'header', 'body', 'json', 'xml', 'rest', 'graphql', 'endpoint', 'client', 'server'
    ];
    
    commonTerms.forEach((term, index) => {
      this.vocab.set(term, index);
    });
  }

  private tokenize(text: string): string[] {
    const camelCaseSplit = text
      .replace(/([a-z])([A-Z])/g, '$1 $2')
      .replace(/[^a-zA-Z0-9\s]/g, ' ')
      .toLowerCase();
      
    const tokens = camelCaseSplit
      .split(/\s+/)
      .filter(token => token.length > 1);
      
    return tokens.map(token => this.normalizeToken(token));
  }
  
  private normalizeToken(token: string): string {
    const normalizations: Record<string, string> = {
      'addition': 'add', 'subtraction': 'subtract', 'multiplication': 'multiply',
      'division': 'divide', 'arithmetic': 'math', 'operations': 'operation',
      'calculation': 'calculate', 'computing': 'compute', 'concatenation': 'concat',
      'concatenate': 'concat', 'splitting': 'split', 'joining': 'join',
      'filtering': 'filter', 'mapping': 'map', 'reducing': 'reduce',
      'finding': 'find', 'searching': 'search', 'getting': 'get',
      'posting': 'post', 'putting': 'put', 'deleting': 'delete',
      'requesting': 'request', 'responding': 'response',
    };
    
    return normalizations[token] || token;
  }
}