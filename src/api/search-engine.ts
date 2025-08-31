/**
 * Main Lens Search Engine
 * Orchestrates three-layer processing pipeline: Lexical+Fuzzy → Symbol/AST → Semantic Rerank
 */

import type { 
  SearchContext, 
  Candidate, 
  SystemHealth,
  HealthStatus
} from '../types/core.js';
import type { SupportedLanguage } from '../types/api.js';
import { LensTracer } from '../telemetry/tracer.js';
import { SegmentStorage } from '../storage/segments.js';
import { LexicalSearchEngine } from '../indexer/lexical.js';
import { SymbolSearchEngine } from '../indexer/symbols.js';
import { SemanticRerankEngine } from '../indexer/semantic.js';
import { MessagingSystem } from '../core/messaging.js';
import { PRODUCTION_CONFIG } from '../types/config.js';

interface SearchResult {
  candidates: Candidate[];
  stage_a_latency?: number;
  stage_b_latency?: number;
  stage_c_latency?: number;
}

export class LensSearchEngine {
  private segmentStorage: SegmentStorage;
  private lexicalEngine: LexicalSearchEngine;
  private symbolEngine: SymbolSearchEngine;
  private semanticEngine: SemanticRerankEngine;
  private messaging: MessagingSystem;
  private isInitialized = false;
  
  // System health tracking
  private activeQueries = 0;
  private startTime = Date.now();

  constructor() {
    this.segmentStorage = new SegmentStorage('./data/segments');
    this.lexicalEngine = new LexicalSearchEngine(this.segmentStorage);
    this.symbolEngine = new SymbolSearchEngine(this.segmentStorage);
    this.semanticEngine = new SemanticRerankEngine(this.segmentStorage);
    this.messaging = new MessagingSystem();
  }

  /**
   * Initialize the search engine
   */
  async initialize(): Promise<void> {
    const span = LensTracer.createChildSpan('search_engine_init');

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

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw new Error(`Failed to initialize search engine: ${errorMsg}`);
    } finally {
      span.end();
    }
  }

  /**
   * Main search method implementing three-layer pipeline
   */
  async search(ctx: SearchContext): Promise<SearchResult> {
    if (!this.isInitialized) {
      throw new Error('Search engine not initialized');
    }

    const overallSpan = LensTracer.startSearchSpan(ctx);
    this.activeQueries++;

    try {
      let candidates: Candidate[] = [];
      let stageALatency = 0;
      let stageBLatency = 0;
      let stageCLatency: number | undefined;

      // Stage A: Lexical + Fuzzy Search (2-8ms target)
      const stageASpan = LensTracer.startStageSpan(ctx, 'stage_a', 'lexical+fuzzy', 0);
      const stageAStart = Date.now();
      
      try {
        candidates = await this.lexicalEngine.search(ctx, ctx.query, ctx.fuzzy_distance);
        stageALatency = Date.now() - stageAStart;

        // Check SLA compliance
        if (stageALatency > PRODUCTION_CONFIG.performance.stage_a_target_ms) {
          console.warn(`Stage A SLA breach: ${stageALatency}ms > ${PRODUCTION_CONFIG.performance.stage_a_target_ms}ms`);
        }

        LensTracer.endStageSpan(
          stageASpan,
          ctx,
          'stage_a',
          'lexical+fuzzy',
          0,
          candidates.length,
          stageALatency
        );

      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : 'Unknown error';
        LensTracer.endStageSpan(
          stageASpan,
          ctx,
          'stage_a',
          'lexical+fuzzy',
          0,
          0,
          Date.now() - stageAStart,
          errorMsg
        );
        throw error;
      }

      // Stage B: Symbol/AST Search (3-10ms target)
      if (ctx.mode === 'struct' || ctx.mode === 'hybrid') {
        const stageBSpan = LensTracer.startStageSpan(ctx, 'stage_b', 'symbol+ast', candidates.length);
        const stageBStart = Date.now();

        try {
          // Search symbols and merge with lexical results
          const symbolCandidates = await this.symbolEngine.searchSymbols(
            ctx.query, 
            ctx, 
            Math.min(100, ctx.k * 2) // Get more candidates for merging
          );

          // Merge symbol results with lexical results
          candidates = this.mergeCandidates(candidates, symbolCandidates, ctx.k);
          
          stageBLatency = Date.now() - stageBStart;

          // Check SLA compliance
          if (stageBLatency > PRODUCTION_CONFIG.performance.stage_b_target_ms) {
            console.warn(`Stage B SLA breach: ${stageBLatency}ms > ${PRODUCTION_CONFIG.performance.stage_b_target_ms}ms`);
          }

          LensTracer.endStageSpan(
            stageBSpan,
            ctx,
            'stage_b',
            'symbol+ast',
            candidates.length + symbolCandidates.length,
            candidates.length,
            stageBLatency
          );

        } catch (error) {
          const errorMsg = error instanceof Error ? error.message : 'Unknown error';
          LensTracer.endStageSpan(
            stageBSpan,
            ctx,
            'stage_b',
            'symbol+ast',
            candidates.length,
            0,
            Date.now() - stageBStart,
            errorMsg
          );
          throw error;
        }
      }

      // Stage C: Semantic Rerank (5-15ms target) - Optional, only if many candidates
      if (candidates.length > 10 && candidates.length <= PRODUCTION_CONFIG.performance.max_candidates) {
        const stageCSpan = LensTracer.startStageSpan(ctx, 'stage_c', 'semantic_rerank', candidates.length);
        const stageCStart = Date.now();

        try {
          // Apply semantic reranking
          candidates = await this.semanticEngine.rerankCandidates(
            candidates, 
            ctx, 
            Math.min(ctx.k, PRODUCTION_CONFIG.performance.max_candidates)
          );
          
          stageCLatency = Date.now() - stageCStart;

          // Check SLA compliance
          if (stageCLatency > PRODUCTION_CONFIG.performance.stage_c_target_ms) {
            console.warn(`Stage C SLA breach: ${stageCLatency}ms > ${PRODUCTION_CONFIG.performance.stage_c_target_ms}ms`);
          }

          LensTracer.endStageSpan(
            stageCSpan,
            ctx,
            'stage_c',
            'semantic_rerank',
            candidates.length,
            candidates.length,
            stageCLatency
          );

        } catch (error) {
          const errorMsg = error instanceof Error ? error.message : 'Unknown error';
          LensTracer.endStageSpan(
            stageCSpan,
            ctx,
            'stage_c',
            'semantic_rerank',
            candidates.length,
            candidates.length,
            Date.now() - stageCStart,
            errorMsg
          );
          // Don't throw for semantic rerank errors, just log
          console.warn(`Semantic rerank failed: ${errorMsg}`);
        }
      }

      // Limit results to requested k
      candidates = candidates.slice(0, ctx.k);

      LensTracer.endSearchSpan(overallSpan, ctx, candidates.length);

      const result: SearchResult = {
        candidates,
        stage_a_latency: stageALatency,
        stage_b_latency: stageBLatency,
      };
      if (stageCLatency !== undefined) {
        result.stage_c_latency = stageCLatency;
      }
      return result;

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      LensTracer.endSearchSpan(overallSpan, ctx, 0, errorMsg);
      throw error;

    } finally {
      this.activeQueries--;
    }
  }

  /**
   * Structural search implementation (placeholder)
   */
  async structuralSearch(ctx: SearchContext, language: SupportedLanguage): Promise<SearchResult> {
    // For now, delegate to regular search
    // TODO: Implement AST-based structural search
    return this.search(ctx);
  }

  /**
   * Find symbols near a location
   */
  async findSymbolsNear(
    filePath: string,
    line: number,
    radius: number = 25
  ): Promise<Candidate[]> {
    const span = LensTracer.createChildSpan('find_symbols_near', {
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

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw new Error(`Failed to find symbols near location: ${errorMsg}`);
    } finally {
      span.end();
    }
  }

  /**
   * Index a repository
   */
  async indexRepository(repoPath: string, repoSha: string): Promise<void> {
    const span = LensTracer.createChildSpan('index_repository', {
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

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw new Error(`Failed to index repository: ${errorMsg}`);
    } finally {
      span.end();
    }
  }

  /**
   * Get system health status
   */
  async getHealthStatus(): Promise<SystemHealth> {
    const span = LensTracer.createChildSpan('get_health_status');

    try {
      // Get basic system stats
      const uptime = Date.now() - this.startTime;
      const memUsage = process.memoryUsage();
      const memUsageGB = memUsage.heapUsed / (1024 * 1024 * 1024);

      // Get worker status from messaging system
      const workerStatus = await this.messaging.getWorkerStatus();

      // Determine overall health
      let status: HealthStatus = 'ok';
      if (memUsageGB > PRODUCTION_CONFIG.resources.memory_limit_gb * 0.9) {
        status = 'degraded';
      }
      if (this.activeQueries > PRODUCTION_CONFIG.resources.max_concurrent_queries) {
        status = 'degraded';
      }
      if (!this.isInitialized) {
        status = 'down';
      }

      const health: SystemHealth = {
        status,
        shards_healthy: 0, // TODO: Get actual shard count
        shards_total: 0,   // TODO: Get actual shard count
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

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
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

    } finally {
      span.end();
    }
  }

  /**
   * Merge candidates from different search stages
   */
  private mergeCandidates(
    lexicalCandidates: Candidate[], 
    symbolCandidates: Candidate[], 
    maxResults: number
  ): Candidate[] {
    const merged: Candidate[] = [];
    const seen = new Set<string>();

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
      } else {
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
  private async loadExistingIndexes(): Promise<void> {
    const span = LensTracer.createChildSpan('load_existing_indexes');

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

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Shutdown the search engine
   */
  async shutdown(): Promise<void> {
    const span = LensTracer.createChildSpan('search_engine_shutdown');

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

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw new Error(`Failed to shutdown search engine: ${errorMsg}`);
    } finally {
      span.end();
    }
  }
}