"use strict";
/**
 * Main Lens Search Engine
 * Orchestrates three-layer processing pipeline: Lexical+Fuzzy → Symbol/AST → Semantic Rerank
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.LensSearchEngine = void 0;
const tracer_js_1 = require("../telemetry/tracer.js");
const segments_js_1 = require("../storage/segments.js");
const lexical_js_1 = require("../indexer/lexical.js");
const symbols_js_1 = require("../indexer/symbols.js");
const semantic_js_1 = require("../indexer/semantic.js");
const messaging_js_1 = require("../core/messaging.js");
const config_js_1 = require("../types/config.js");
class LensSearchEngine {
    segmentStorage;
    lexicalEngine;
    symbolEngine;
    semanticEngine;
    messaging;
    isInitialized = false;
    // System health tracking
    activeQueries = 0;
    startTime = Date.now();
    constructor() {
        this.segmentStorage = new segments_js_1.SegmentStorage('./data/segments');
        this.lexicalEngine = new lexical_js_1.LexicalSearchEngine(this.segmentStorage);
        this.symbolEngine = new symbols_js_1.SymbolSearchEngine(this.segmentStorage);
        this.semanticEngine = new semantic_js_1.SemanticRerankEngine(this.segmentStorage);
        this.messaging = new messaging_js_1.MessagingSystem();
    }
    /**
     * Initialize the search engine
     */
    async initialize() {
        const span = tracer_js_1.LensTracer.createChildSpan('search_engine_init');
        try {
            // Initialize components in parallel
            await Promise.all([
                this.messaging.initialize(),
                this.symbolEngine.initialize(),
                this.semanticEngine.initialize(),
                this.loadExistingIndexes(),
            ]);
            this.isInitialized = true;
            console.log('🔍 Lens Search Engine initialized');
            span.setAttributes({ success: true });
        }
        catch (error) {
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            span.recordException(error);
            span.setAttributes({ success: false, error: errorMsg });
            throw new Error(`Failed to initialize search engine: ${errorMsg}`);
        }
        finally {
            span.end();
        }
    }
    /**
     * Main search method implementing three-layer pipeline
     */
    async search(ctx) {
        if (!this.isInitialized) {
            throw new Error('Search engine not initialized');
        }
        const overallSpan = tracer_js_1.LensTracer.startSearchSpan(ctx);
        this.activeQueries++;
        try {
            let candidates = [];
            let stageALatency = 0;
            let stageBLatency = 0;
            let stageCLatency;
            // Stage A: Lexical + Fuzzy Search (2-8ms target)
            const stageASpan = tracer_js_1.LensTracer.startStageSpan(ctx, 'stage_a', 'lexical+fuzzy', 0);
            const stageAStart = Date.now();
            try {
                candidates = await this.lexicalEngine.search(ctx, ctx.query, ctx.fuzzy_distance);
                stageALatency = Date.now() - stageAStart;
                // Check SLA compliance
                if (stageALatency > config_js_1.PRODUCTION_CONFIG.performance.stage_a_target_ms) {
                    console.warn(`Stage A SLA breach: ${stageALatency}ms > ${config_js_1.PRODUCTION_CONFIG.performance.stage_a_target_ms}ms`);
                }
                tracer_js_1.LensTracer.endStageSpan(stageASpan, ctx, 'stage_a', 'lexical+fuzzy', 0, candidates.length, stageALatency);
            }
            catch (error) {
                const errorMsg = error instanceof Error ? error.message : 'Unknown error';
                tracer_js_1.LensTracer.endStageSpan(stageASpan, ctx, 'stage_a', 'lexical+fuzzy', 0, 0, Date.now() - stageAStart, errorMsg);
                throw error;
            }
            // Stage B: Symbol/AST Search (3-10ms target)
            if (ctx.mode === 'struct' || ctx.mode === 'hybrid') {
                const stageBSpan = tracer_js_1.LensTracer.startStageSpan(ctx, 'stage_b', 'symbol+ast', candidates.length);
                const stageBStart = Date.now();
                try {
                    // Search symbols and merge with lexical results
                    const symbolCandidates = await this.symbolEngine.searchSymbols(ctx.query, ctx, Math.min(100, ctx.k * 2) // Get more candidates for merging
                    );
                    // Merge symbol results with lexical results
                    candidates = this.mergeCandidates(candidates, symbolCandidates, ctx.k);
                    stageBLatency = Date.now() - stageBStart;
                    // Check SLA compliance
                    if (stageBLatency > config_js_1.PRODUCTION_CONFIG.performance.stage_b_target_ms) {
                        console.warn(`Stage B SLA breach: ${stageBLatency}ms > ${config_js_1.PRODUCTION_CONFIG.performance.stage_b_target_ms}ms`);
                    }
                    tracer_js_1.LensTracer.endStageSpan(stageBSpan, ctx, 'stage_b', 'symbol+ast', candidates.length + symbolCandidates.length, candidates.length, stageBLatency);
                }
                catch (error) {
                    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
                    tracer_js_1.LensTracer.endStageSpan(stageBSpan, ctx, 'stage_b', 'symbol+ast', candidates.length, 0, Date.now() - stageBStart, errorMsg);
                    throw error;
                }
            }
            // Stage C: Semantic Rerank (5-15ms target) - Optional, only if many candidates
            if (candidates.length > 10 && candidates.length <= config_js_1.PRODUCTION_CONFIG.performance.max_candidates) {
                const stageCSpan = tracer_js_1.LensTracer.startStageSpan(ctx, 'stage_c', 'semantic_rerank', candidates.length);
                const stageCStart = Date.now();
                try {
                    // Apply semantic reranking
                    candidates = await this.semanticEngine.rerankCandidates(candidates, ctx, Math.min(ctx.k, config_js_1.PRODUCTION_CONFIG.performance.max_candidates));
                    stageCLatency = Date.now() - stageCStart;
                    // Check SLA compliance
                    if (stageCLatency > config_js_1.PRODUCTION_CONFIG.performance.stage_c_target_ms) {
                        console.warn(`Stage C SLA breach: ${stageCLatency}ms > ${config_js_1.PRODUCTION_CONFIG.performance.stage_c_target_ms}ms`);
                    }
                    tracer_js_1.LensTracer.endStageSpan(stageCSpan, ctx, 'stage_c', 'semantic_rerank', candidates.length, candidates.length, stageCLatency);
                }
                catch (error) {
                    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
                    tracer_js_1.LensTracer.endStageSpan(stageCSpan, ctx, 'stage_c', 'semantic_rerank', candidates.length, candidates.length, Date.now() - stageCStart, errorMsg);
                    // Don't throw for semantic rerank errors, just log
                    console.warn(`Semantic rerank failed: ${errorMsg}`);
                }
            }
            // Limit results to requested k
            candidates = candidates.slice(0, ctx.k);
            tracer_js_1.LensTracer.endSearchSpan(overallSpan, ctx, candidates.length);
            return {
                candidates,
                stage_a_latency: stageALatency,
                stage_b_latency: stageBLatency,
                stage_c_latency: stageCLatency,
            };
        }
        catch (error) {
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            tracer_js_1.LensTracer.endSearchSpan(overallSpan, ctx, 0, errorMsg);
            throw error;
        }
        finally {
            this.activeQueries--;
        }
    }
    /**
     * Structural search implementation (placeholder)
     */
    async structuralSearch(ctx, language) {
        // For now, delegate to regular search
        // TODO: Implement AST-based structural search
        return this.search(ctx);
    }
    /**
     * Find symbols near a location
     */
    async findSymbolsNear(filePath, line, radius = 25) {
        const span = tracer_js_1.LensTracer.createChildSpan('find_symbols_near', {
            'file_path': filePath,
            'line': line,
            'radius': radius,
        });
        try {
            const candidates = await this.symbolEngine.findSymbolsNear(filePath, line, radius);
            span.setAttributes({
                success: true,
                candidates_found: candidates.length,
            });
            return candidates;
        }
        catch (error) {
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            span.recordException(error);
            span.setAttributes({ success: false, error: errorMsg });
            throw new Error(`Failed to find symbols near location: ${errorMsg}`);
        }
        finally {
            span.end();
        }
    }
    /**
     * Index a repository
     */
    async indexRepository(repoPath, repoSha) {
        const span = tracer_js_1.LensTracer.createChildSpan('index_repository', {
            'repo.path': repoPath,
            'repo.sha': repoSha,
        });
        try {
            // TODO: Implement full repository indexing
            // This would involve:
            // 1. Walking the file tree
            // 2. Creating shards based on path hashing
            // 3. Publishing work units to NATS
            // 4. Processing files through all three layers
            console.log(`Indexing repository: ${repoPath} (${repoSha})`);
            span.setAttributes({ success: true });
        }
        catch (error) {
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            span.recordException(error);
            span.setAttributes({ success: false, error: errorMsg });
            throw new Error(`Failed to index repository: ${errorMsg}`);
        }
        finally {
            span.end();
        }
    }
    /**
     * Get system health status
     */
    async getHealthStatus() {
        const span = tracer_js_1.LensTracer.createChildSpan('get_health_status');
        try {
            // Get basic system stats
            const uptime = Date.now() - this.startTime;
            const memUsage = process.memoryUsage();
            const memUsageGB = memUsage.heapUsed / (1024 * 1024 * 1024);
            // Get worker status from messaging system
            const workerStatus = await this.messaging.getWorkerStatus();
            // Determine overall health
            let status = 'ok';
            if (memUsageGB > config_js_1.PRODUCTION_CONFIG.resources.memory_limit_gb * 0.9) {
                status = 'degraded';
            }
            if (this.activeQueries > config_js_1.PRODUCTION_CONFIG.resources.max_concurrent_queries) {
                status = 'degraded';
            }
            if (!this.isInitialized) {
                status = 'down';
            }
            const health = {
                status,
                shards_healthy: 0, // TODO: Get actual shard count
                shards_total: 0, // TODO: Get actual shard count
                memory_usage_gb: memUsageGB,
                active_queries: this.activeQueries,
                worker_pool_status: workerStatus,
                last_compaction: new Date(this.startTime), // TODO: Track actual compaction
            };
            span.setAttributes({
                success: true,
                status: health.status,
                memory_usage_gb: health.memory_usage_gb,
                active_queries: health.active_queries,
            });
            return health;
        }
        catch (error) {
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            span.recordException(error);
            span.setAttributes({ success: false, error: errorMsg });
            // Return degraded status on error
            return {
                status: 'degraded',
                shards_healthy: 0,
                shards_total: 0,
                memory_usage_gb: 0,
                active_queries: this.activeQueries,
                worker_pool_status: {
                    ingest_active: 0,
                    query_active: 0,
                    maintenance_active: 0,
                },
                last_compaction: new Date(this.startTime),
            };
        }
        finally {
            span.end();
        }
    }
    /**
     * Merge candidates from different search stages
     */
    mergeCandidates(lexicalCandidates, symbolCandidates, maxResults) {
        const merged = [];
        const seen = new Set();
        // Combine all candidates
        const allCandidates = [...lexicalCandidates, ...symbolCandidates];
        // Deduplicate by doc_id and merge match_reasons
        for (const candidate of allCandidates) {
            const existing = merged.find(c => c.doc_id === candidate.doc_id);
            if (existing) {
                // Merge match reasons and take higher score
                existing.match_reasons = Array.from(new Set([
                    ...existing.match_reasons,
                    ...candidate.match_reasons
                ]));
                existing.score = Math.max(existing.score, candidate.score);
                // Prefer symbol information if available
                if (candidate.symbol_kind && !existing.symbol_kind) {
                    existing.symbol_kind = candidate.symbol_kind;
                }
                if (candidate.ast_path && !existing.ast_path) {
                    existing.ast_path = candidate.ast_path;
                }
            }
            else {
                merged.push({ ...candidate });
            }
        }
        // Sort by relevance score and limit results
        merged.sort((a, b) => {
            // Boost candidates with multiple match reasons
            const aBoost = a.match_reasons.length * 0.1;
            const bBoost = b.match_reasons.length * 0.1;
            return (b.score + bBoost) - (a.score + aBoost);
        });
        return merged.slice(0, maxResults);
    }
    /**
     * Load existing indexes on startup
     */
    async loadExistingIndexes() {
        const span = tracer_js_1.LensTracer.createChildSpan('load_existing_indexes');
        try {
            // TODO: Implement index loading from segments
            // For now, just log that we're starting fresh
            console.log('Loading existing indexes...');
            const segments = this.segmentStorage.listSegments();
            console.log(`Found ${segments.length} existing segments`);
            span.setAttributes({
                success: true,
                segments_found: segments.length,
            });
        }
        catch (error) {
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            span.recordException(error);
            span.setAttributes({ success: false, error: errorMsg });
            throw error;
        }
        finally {
            span.end();
        }
    }
    /**
     * Shutdown the search engine
     */
    async shutdown() {
        const span = tracer_js_1.LensTracer.createChildSpan('search_engine_shutdown');
        try {
            console.log('Shutting down Lens Search Engine...');
            // Wait for active queries to complete (with timeout)
            const timeout = 5000; // 5 seconds
            const start = Date.now();
            while (this.activeQueries > 0 && (Date.now() - start) < timeout) {
                await new Promise(resolve => setTimeout(resolve, 100));
            }
            if (this.activeQueries > 0) {
                console.warn(`Forcibly shutting down with ${this.activeQueries} active queries`);
            }
            // Shutdown components
            await Promise.all([
                this.messaging.shutdown(),
                this.symbolEngine.shutdown(),
                this.semanticEngine.shutdown(),
                this.segmentStorage.shutdown(),
            ]);
            this.isInitialized = false;
            console.log('Lens Search Engine shut down successfully');
            span.setAttributes({ success: true });
        }
        catch (error) {
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            span.recordException(error);
            span.setAttributes({ success: false, error: errorMsg });
            throw new Error(`Failed to shutdown search engine: ${errorMsg}`);
        }
        finally {
            span.end();
        }
    }
}
exports.LensSearchEngine = LensSearchEngine;
