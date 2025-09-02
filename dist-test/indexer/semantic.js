"use strict";
/**
 * Layer 3: Semantic Rerank Implementation
 * ColBERT-v2/SPLADE semantic reranking for high-precision results
 * Target: 5-15ms (Stage-C) - Neural reranking with vectorized similarity
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.SemanticRerankEngine = void 0;
const tracer_js_1 = require("../telemetry/tracer.js");
/**
 * Simplified semantic reranking engine
 * In production, this would use actual ColBERT/SPLADE models
 */
class SemanticRerankEngine {
    semanticIndex = new Map(); // doc_id -> vector
    hnswIndex = null;
    segmentStorage;
    embeddingModel;
    // Simplified embedding dimensions for demo
    EMBEDDING_DIM = 128;
    MAX_CONNECTIONS = 16;
    LEVEL_MULTIPLIER = 1.2;
    constructor(segmentStorage) {
        this.segmentStorage = segmentStorage;
        this.embeddingModel = new SimpleEmbeddingModel(this.EMBEDDING_DIM);
    }
    /**
     * Initialize the semantic rerank engine
     */
    async initialize() {
        const span = tracer_js_1.LensTracer.createChildSpan('semantic_engine_init');
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
        }
        catch (error) {
            span.recordException(error);
            span.setAttributes({ success: false });
            throw error;
        }
        finally {
            span.end();
        }
    }
    /**
     * Rerank candidates using semantic similarity
     */
    async rerankCandidates(candidates, context, maxResults = 100) {
        const span = tracer_js_1.LensTracer.createChildSpan('semantic_rerank', {
            'candidates.input': candidates.length,
            'search.query': context.query,
            'search.max_results': maxResults,
        });
        try {
            // Skip reranking for very few candidates
            if (candidates.length <= 5) {
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
            const rerankedCandidates = [];
            for (const candidate of candidates) {
                // Get or generate document embedding
                let docEmbedding = this.semanticIndex.get(candidate.doc_id);
                if (!docEmbedding) {
                    // Fallback: generate embedding from context
                    const contextText = candidate.context || candidate.file_path;
                    docEmbedding = await this.embeddingModel.encode(contextText);
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
                const boost = candidate.semantic_score > 0.7 ? 0.1 : 0;
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
            const results = finalCandidates.slice(0, maxResults).map(({ semantic_score, ...candidate }) => candidate);
            span.setAttributes({
                success: true,
                candidates_output: results.length,
                reranking_latency_ms: rerankingLatency,
                avg_semantic_score: rerankedCandidates.reduce((sum, c) => sum + c.semantic_score, 0) / rerankedCandidates.length,
            });
            return results;
        }
        catch (error) {
            span.recordException(error);
            span.setAttributes({ success: false });
            // Fallback: return original candidates on error
            console.warn(`Semantic reranking failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
            return candidates.slice(0, maxResults);
        }
        finally {
            span.end();
        }
    }
    /**
     * Index a document for semantic search
     */
    async indexDocument(docId, content, filePath) {
        const span = tracer_js_1.LensTracer.createChildSpan('index_document_semantic', {
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
            }
            span.setAttributes({
                success: true,
                embedding_dim: embedding.length,
            });
        }
        catch (error) {
            span.recordException(error);
            span.setAttributes({ success: false });
            throw error;
        }
        finally {
            span.end();
        }
    }
    /**
     * Find similar documents using HNSW
     */
    async findSimilarDocuments(queryEmbedding, k = 50) {
        const span = tracer_js_1.LensTracer.createChildSpan('find_similar_docs', {
            'query.embedding_dim': queryEmbedding.length,
            'search.k': k,
        });
        try {
            const similarities = [];
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
        }
        catch (error) {
            span.recordException(error);
            span.setAttributes({ success: false });
            throw error;
        }
        finally {
            span.end();
        }
    }
    /**
     * Build HNSW index for efficient similarity search
     */
    async buildHNSWIndex() {
        const span = tracer_js_1.LensTracer.createChildSpan('build_hnsw_index');
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
            const layer0 = {
                level: 0,
                nodes: new Map(),
            };
            // Add all vectors to layer 0
            for (let i = 0; i < vectors.length; i++) {
                const [docId, embedding] = vectors[i];
                const node = {
                    id: i,
                    vector: embedding,
                    connections: new Set(),
                };
                // Connect to nearest neighbors (simplified)
                const connections = this.findNearestNeighbors(embedding, vectors.slice(0, i), Math.min(this.MAX_CONNECTIONS, i));
                connections.forEach(connIdx => node.connections.add(connIdx));
                layer0.nodes.set(i, node);
            }
            this.hnswIndex.layers.push(layer0);
            span.setAttributes({
                success: true,
                vectors: numVectors,
                layers: 1,
                avg_connections: Array.from(layer0.nodes.values()).reduce((sum, node) => sum + node.connections.size, 0) / numVectors,
            });
        }
        catch (error) {
            span.recordException(error);
            span.setAttributes({ success: false });
            throw error;
        }
        finally {
            span.end();
        }
    }
    /**
     * Find nearest neighbors for HNSW construction
     */
    findNearestNeighbors(queryVector, candidates, k) {
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
    async addToHNSWIndex(docId, embedding) {
        // Simplified - in production would properly update HNSW structure
        if (!this.hnswIndex || this.hnswIndex.layers.length === 0) {
            return;
        }
        const layer0 = this.hnswIndex.layers[0];
        const nodeId = layer0.nodes.size;
        const node = {
            id: nodeId,
            vector: embedding,
            connections: new Set(),
        };
        layer0.nodes.set(nodeId, node);
    }
    /**
     * Load semantic vectors from segment
     */
    async loadSemanticSegment(segmentId) {
        const span = tracer_js_1.LensTracer.createChildSpan('load_semantic_segment', {
            'segment.id': segmentId,
        });
        try {
            const segment = await this.segmentStorage.openSegment(segmentId, true);
            const data = await this.segmentStorage.readFromSegment(segmentId, 0, segment.size);
            // Parse semantic data (simplified - would be binary format in production)
            const semanticData = JSON.parse(data.toString('utf8'));
            if (semanticData.vectors) {
                for (const [docId, vectorArray] of Object.entries(semanticData.vectors)) {
                    const vector = new Float32Array(vectorArray);
                    this.semanticIndex.set(docId, vector);
                }
            }
            span.setAttributes({ success: true });
        }
        catch (error) {
            span.recordException(error);
            span.setAttributes({ success: false });
            throw error;
        }
        finally {
            span.end();
        }
    }
    /**
     * Get semantic search statistics
     */
    getStats() {
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
    async shutdown() {
        this.semanticIndex.clear();
        this.hnswIndex = null;
        console.log('Semantic rerank engine shut down');
    }
}
exports.SemanticRerankEngine = SemanticRerankEngine;
/**
 * Simple embedding model for demonstration
 * In production, use actual ColBERT/SPLADE models
 */
class SimpleEmbeddingModel {
    dimension;
    vocab = new Map();
    constructor(dimension = 128) {
        this.dimension = dimension;
        this.initializeVocab();
    }
    /**
     * Generate embedding for text
     */
    async encode(text) {
        const tokens = this.tokenize(text.toLowerCase());
        const embedding = new Float32Array(this.dimension);
        // Simple TF-IDF-like embedding
        const tokenCounts = new Map();
        for (const token of tokens) {
            tokenCounts.set(token, (tokenCounts.get(token) || 0) + 1);
        }
        for (const [token, count] of tokenCounts) {
            const tokenId = this.vocab.get(token);
            if (tokenId !== undefined) {
                const index = tokenId % this.dimension;
                embedding[index] += count * Math.log(1 + count);
            }
        }
        // Normalize vector
        const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
        if (norm > 0) {
            for (let i = 0; i < embedding.length; i++) {
                embedding[i] /= norm;
            }
        }
        return embedding;
    }
    /**
     * Calculate cosine similarity between vectors
     */
    similarity(a, b) {
        if (a.length !== b.length) {
            return 0;
        }
        let dotProduct = 0;
        let normA = 0;
        let normB = 0;
        for (let i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        const norm = Math.sqrt(normA) * Math.sqrt(normB);
        return norm > 0 ? dotProduct / norm : 0;
    }
    /**
     * Initialize vocabulary with common programming terms
     */
    initializeVocab() {
        const commonTerms = [
            'function', 'class', 'interface', 'type', 'variable', 'const', 'let', 'var',
            'import', 'export', 'return', 'if', 'else', 'for', 'while', 'try', 'catch',
            'async', 'await', 'promise', 'callback', 'event', 'handler', 'component',
            'service', 'api', 'endpoint', 'request', 'response', 'data', 'model',
            'controller', 'view', 'template', 'config', 'settings', 'utils', 'helpers',
            'test', 'spec', 'mock', 'stub', 'assert', 'expect', 'describe', 'it',
        ];
        commonTerms.forEach((term, index) => {
            this.vocab.set(term, index);
        });
    }
    /**
     * Simple tokenization
     */
    tokenize(text) {
        return text
            .replace(/[^a-zA-Z0-9\s]/g, ' ')
            .split(/\s+/)
            .filter(token => token.length > 1);
    }
}
