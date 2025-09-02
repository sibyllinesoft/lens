/**
 * Main Lens Search Engine
 * Orchestrates four-stage processing pipeline: Lexical+Fuzzy → Symbol/AST → Semantic Rerank → Learned Rerank
 * Phase 2 Enhanced with TypeScript patterns, AST caching, and learned reranking
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
import { IndexRegistry, type IndexReader } from '../core/index-registry.js';
import { ASTCache, type CachedAST } from '../core/ast-cache.js';
import { LearnedReranker, type RerankingConfig } from '../core/learned-reranker.js';
import { 
  SearchHit, 
  resolveLexicalMatches, 
  resolveSymbolMatches, 
  resolveSemanticMatches,
  prepareSemanticCandidates,
  LexicalCandidate,
  SymbolCandidate,
  SemanticCandidate
} from '../core/span_resolver/index.js';

interface SearchResult {
  hits: SearchHit[];
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
  private indexRegistry: IndexRegistry;
  private astCache: ASTCache;
  private learnedReranker: LearnedReranker;
  private isInitialized = false;
  
  // System health tracking
  private activeQueries = 0;
  private startTime = Date.now();

  constructor(indexRoot: string = './indexed-content', rerankConfig?: Partial<RerankingConfig>) {
    this.segmentStorage = new SegmentStorage('./data/segments');
    this.lexicalEngine = new LexicalSearchEngine(this.segmentStorage);
    this.symbolEngine = new SymbolSearchEngine(this.segmentStorage);
    this.semanticEngine = new SemanticRerankEngine(this.segmentStorage);
    this.messaging = new MessagingSystem();
    this.indexRegistry = new IndexRegistry(indexRoot);
    this.astCache = new ASTCache(50); // Cache top 50 hot files
    this.learnedReranker = new LearnedReranker({
      enabled: false, // Start disabled for A/B testing
      nlThreshold: 0.5,
      minCandidates: 10,
      maxLatencyMs: 5,
      ...rerankConfig,
    });
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
        this.indexRegistry.refresh(),
        this.loadExistingIndexes(),
      ]);

      // Verify at least one repository is available
      const stats = this.indexRegistry.stats();
      if (stats.totalRepos === 0) {
        throw new Error('No repositories found in index - cannot start search engine');
      }

      this.isInitialized = true;
      console.log(`🔍 Lens Search Engine initialized with ${stats.totalRepos} repositories`);
      
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
   * Main search method implementing four-stage pipeline
   */
  async search(ctx: SearchContext): Promise<SearchResult> {
    if (!this.isInitialized) {
      throw new Error('Search engine not initialized');
    }

    const overallSpan = LensTracer.startSearchSpan(ctx);
    this.activeQueries++;

    try {
      let hits: SearchHit[] = [];
      let stageALatency = 0;
      let stageBLatency = 0;
      let stageCLatency: number | undefined;

      // Stage A: Lexical + Fuzzy Search (2-8ms target)
      const stageASpan = LensTracer.startStageSpan(ctx, 'stage_a', 'lexical+fuzzy', 0);
      const stageAStart = Date.now();
      
      try {
        // Get IndexReader for this repository
        if (!this.indexRegistry.hasRepo(ctx.repo_sha)) {
          throw new Error(`INDEX_MISSING: Repository not found in index: ${ctx.repo_sha}`);
        }
        
        const reader = this.indexRegistry.getReader(ctx.repo_sha);
        
        // Use IndexReader for lexical search
        const lexicalResults = await reader.searchLexical({
          q: ctx.query,
          fuzzy: Math.min(2, Math.max(0, Math.round((ctx.fuzzy_distance || 0) * 2))),
          subtokens: true,
          k: Math.min(200, ctx.k * 4),
        });
        
        // Convert IndexReader results to SearchHit format
        hits = lexicalResults.map(result => ({
          file: result.file,
          line: result.line,
          col: result.col,
          lang: result.lang,
          snippet: result.snippet,
          score: result.score,
          why: result.why as any,
          byte_offset: result.byte_offset,
          span_len: result.span_len,
        }));
        
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
          hits.length,
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

      // Stage B: Symbol/AST Search (3-10ms target) - Enhanced with TypeScript patterns
      if (ctx.mode === 'struct' || ctx.mode === 'hybrid') {
        const stageBSpan = LensTracer.startStageSpan(ctx, 'stage_b', 'structural+ast', hits.length);
        const stageBStart = Date.now();

        try {
          // Use IndexReader for structural search with new TypeScript patterns
          const reader = this.indexRegistry.getReader(ctx.repo_sha);
          
          const structuralResults = await reader.searchStructural({
            q: ctx.query,
            k: Math.min(100, ctx.k * 2),
          });
          
          // Convert structural results to SearchHit format
          const structuralHits: SearchHit[] = structuralResults.map(result => ({
            file: result.file,
            line: result.line,
            col: result.col,
            lang: result.lang,
            snippet: result.snippet,
            score: result.score,
            why: result.why as any,
            byte_offset: result.byte_offset,
            span_len: result.span_len,
            pattern_type: result.pattern_type,
            symbol_name: result.symbol_name,
            signature: result.signature,
          }));
          
          // Merge structural results with lexical results
          hits = this.mergeSearchHits(hits, structuralHits, ctx.k);
          
          stageBLatency = Date.now() - stageBStart;

          LensTracer.endStageSpan(
            stageBSpan,
            ctx,
            'stage_b',
            'structural+ast',
            structuralResults.length + hits.length,
            hits.length,
            stageBLatency
          );

        } catch (error) {
          const errorMsg = error instanceof Error ? error.message : 'Unknown error';
          LensTracer.endStageSpan(
            stageBSpan,
            ctx,
            'stage_b',
            'structural+ast',
            hits.length,
            0,
            Date.now() - stageBStart,
            errorMsg
          );
          throw error;
        }
      }

      // Stage C: Semantic Rerank (5-15ms target)
      if (hits.length > 10 && hits.length <= PRODUCTION_CONFIG.performance.max_candidates) {
        const stageCSpan = LensTracer.startStageSpan(ctx, 'stage_c', 'semantic_rerank', hits.length);
        const stageCStart = Date.now();

        try {
          const candidatesForRerank = this.convertHitsToCandidates(hits);
          
          const rerankedCandidates = await this.semanticEngine.rerankCandidates(
            candidatesForRerank, 
            ctx, 
            Math.min(ctx.k, PRODUCTION_CONFIG.performance.max_candidates)
          );
          
          const rerankedScores: number[] = [];
          for (let i = 0; i < hits.length; i++) {
            if (i < rerankedCandidates.length && rerankedCandidates[i]) {
              rerankedScores.push(rerankedCandidates[i]!.score);
            } else {
              rerankedScores.push(hits[i]?.score ?? 0.1);
            }
          }
          
          const semanticCandidates = prepareSemanticCandidates(hits, rerankedScores);
          hits = await resolveSemanticMatches(semanticCandidates);
          
          stageCLatency = Date.now() - stageCStart;

          LensTracer.endStageSpan(
            stageCSpan,
            ctx,
            'stage_c',
            'semantic_rerank',
            hits.length,
            hits.length,
            stageCLatency
          );

        } catch (error) {
          const errorMsg = error instanceof Error ? error.message : 'Unknown error';
          LensTracer.endStageSpan(
            stageCSpan,
            ctx,
            'stage_c',
            'semantic_rerank',
            hits.length,
            hits.length,
            Date.now() - stageCStart,
            errorMsg
          );
          console.warn(`Semantic rerank failed: ${errorMsg}`);
        }
      }

      // Stage D: Learned Reranker (Phase 2 Enhancement) - Feature flagged
      const stageDSpan = LensTracer.createChildSpan('stage_d_learned_rerank');
      const stageDStart = Date.now();
      
      try {
        hits = await this.learnedReranker.rerank(hits, ctx);
        
        const stageDLatency = Date.now() - stageDStart;
        
        stageDSpan.setAttributes({
          success: true,
          latency_ms: stageDLatency,
          hits_reranked: hits.length,
        });

      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : 'Unknown error';
        stageDSpan.recordException(error as Error);
        stageDSpan.setAttributes({ success: false, error: errorMsg });
        console.warn(`Learned rerank failed: ${errorMsg}`);
      } finally {
        stageDSpan.end();
      }

      // Limit results to requested k
      hits = hits.slice(0, ctx.k);

      LensTracer.endSearchSpan(overallSpan, ctx, hits.length);

      const result: SearchResult = {
        hits,
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
   * Enable/disable learned reranker for A/B testing
   */
  setRerankingEnabled(enabled: boolean) {
    this.learnedReranker.updateConfig({ enabled });
    console.log(`🧠 Learned reranking ${enabled ? 'ENABLED' : 'DISABLED'}`);
  }

  /**
   * Get AST cache coverage statistics
   */
  getASTCoverageStats(): { coverage: any; stats: any } {
    const stats = this.astCache.getStats();
    
    // Count TypeScript files in the index
    const manifests = this.indexRegistry.getManifests();
    let totalTSFiles = 0;
    
    for (const manifest of manifests) {
      totalTSFiles += manifest.shard_paths?.filter(path => 
        path.endsWith('.ts') && !path.endsWith('.d.ts')
      ).length || 0;
    }
    
    const coverage = this.astCache.getCoverageStats(totalTSFiles);
    
    return { coverage, stats };
  }

  /**
   * Get manifest mapping repo_ref to repo_sha
   */
  async getManifest(): Promise<{ [repo_ref: string]: string }> {
    const span = LensTracer.createChildSpan('get_manifest');

    try {
      const manifests = this.indexRegistry.getManifests();
      const mapping: { [repo_ref: string]: string } = {};
      
      for (const manifest of manifests) {
        mapping[manifest.repo_ref] = manifest.repo_sha;
      }

      span.setAttributes({
        success: true,
        manifests_count: manifests.length,
      });

      return mapping;

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw new Error(`Failed to get manifest: ${errorMsg}`);
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
      const uptime = Date.now() - this.startTime;
      const memUsage = process.memoryUsage();
      const memUsageGB = memUsage.heapUsed / (1024 * 1024 * 1024);

      const workerStatus = await this.messaging.getWorkerStatus();
      const registryStats = this.indexRegistry.stats();
      const stageAReady = registryStats.totalRepos > 0;

      let status: HealthStatus = 'ok';
      if (memUsageGB > PRODUCTION_CONFIG.resources.memory_limit_gb * 0.9) {
        status = 'degraded';
      }
      if (this.activeQueries > PRODUCTION_CONFIG.resources.max_concurrent_queries) {
        status = 'degraded';
      }
      if (!this.isInitialized || !stageAReady) {
        status = 'down';
      }

      const health: SystemHealth = {
        status,
        shards_healthy: registryStats.loadedRepos,
        shards_total: registryStats.totalRepos,
        memory_usage_gb: memUsageGB,
        active_queries: this.activeQueries,
        worker_pool_status: workerStatus,
        last_compaction: new Date(this.startTime),
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
   * Shutdown the search engine
   */
  async shutdown(): Promise<void> {
    const span = LensTracer.createChildSpan('search_engine_shutdown');

    try {
      console.log('Shutting down Lens Search Engine...');

      const timeout = 5000;
      const start = Date.now();
      
      while (this.activeQueries > 0 && (Date.now() - start) < timeout) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      if (this.activeQueries > 0) {
        console.warn(`Forcibly shutting down with ${this.activeQueries} active queries`);
      }

      await Promise.all([
        this.messaging.shutdown(),
        this.symbolEngine.shutdown(),
        this.semanticEngine.shutdown(),
        this.segmentStorage.shutdown(),
        this.indexRegistry.shutdown(),
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

  /**
   * Load existing indexes on startup
   */
  private async loadExistingIndexes(): Promise<void> {
    const span = LensTracer.createChildSpan('load_existing_indexes');

    try {
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
   * Merge SearchHits from different search stages
   */
  private mergeSearchHits(
    lexicalHits: SearchHit[], 
    symbolHits: SearchHit[], 
    maxResults: number
  ): SearchHit[] {
    const merged: SearchHit[] = [];

    // Combine all hits
    const allHits = [...lexicalHits, ...symbolHits];

    // Deduplicate by file path and merge match_reasons
    for (const hit of allHits) {
      const key = `${hit.file}:${hit.line}:${hit.col}`;
      const existing = merged.find(h => `${h.file}:${h.line}:${h.col}` === key);
      
      if (existing) {
        // Merge match reasons and take higher score
        existing.why = Array.from(new Set([
          ...existing.why, 
          ...hit.why
        ])) as any;
        existing.score = Math.max(existing.score, hit.score);
        
        // Prefer symbol information if available
        if (hit.symbol_kind && !existing.symbol_kind) {
          existing.symbol_kind = hit.symbol_kind;
        }
        
        if (hit.snippet && !existing.snippet) {
          existing.snippet = hit.snippet;
        }
      } else {
        merged.push({ ...hit });
      }
    }

    // Sort by relevance score and limit results
    merged.sort((a, b) => {
      const aBoost = a.why.length * 0.1;
      const bBoost = b.why.length * 0.1;
      
      return (b.score + bBoost) - (a.score + aBoost);
    });

    return merged.slice(0, maxResults);
  }

  /**
   * Convert SearchHits to Candidates for compatibility with existing semantic engine
   */
  private convertHitsToCandidates(hits: SearchHit[]): Candidate[] {
    return hits.map((hit, index) => ({
      doc_id: `hit_${index}`,
      file_path: hit.file,
      line: hit.line,
      col: hit.col,
      score: hit.score,
      match_reasons: hit.why as any,
      ast_path: hit.ast_path ?? undefined,
      symbol_kind: hit.symbol_kind as any,
      snippet: hit.snippet ?? undefined,
      byte_offset: hit.byte_offset ?? undefined,
      span_len: hit.span_len ?? undefined,
      context_before: hit.context_before ?? undefined,
      context_after: hit.context_after ?? undefined,
      context: hit.context_before || hit.context_after || undefined,
    }));
  }
}