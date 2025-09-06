/**
 * Main Lens Search Engine
 * Orchestrates four-stage processing pipeline: Lexical+Fuzzy → Symbol/AST → Semantic Rerank → Learned Rerank
 * Phase 2 Enhanced with TypeScript patterns, AST caching, and learned reranking
 */

import type { 
  SearchContext, 
  Candidate, 
  SystemHealth,
  HealthStatus,
  LSPHint
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
import { PhaseBComprehensiveOptimizer, type PhaseBConfig } from '../benchmark/phase-b-comprehensive.js';
import { 
  SearchHit, 
  resolveLexicalMatches, 
  resolveSymbolMatches, 
  resolveSemanticMatches,
  prepareSemanticCandidates,
  LexicalCandidate,
  SymbolCandidate,
  SemanticCandidate,
  MatchReason
} from '../core/span_resolver/index.js';
import { globalAdaptiveFanout } from '../core/adaptive-fanout.js';
import { globalWorkConservingANN } from '../core/work-conserving-ann.js';
import { globalPrecisionEngine } from '../core/precision-optimization.js';
import { IntentRouter } from '../core/intent-router.js';
import { LSPStageBEnhancer } from '../core/lsp-stage-b.js';
import { LSPStageCEnhancer } from '../core/lsp-stage-c.js';

interface LensSearchResult {
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
  private phaseBOptimizer: PhaseBComprehensiveOptimizer;
  private intentRouter?: IntentRouter;
  private lspStageBEnhancer?: LSPStageBEnhancer;
  private lspStageCEnhancer?: LSPStageCEnhancer;
  private lspEnabled = false;
  private isInitialized = false;
  
  // System health tracking
  private activeQueries = 0;
  private startTime = Date.now();
  
  // Phase B optimization configuration
  private phaseBEnabled = false;

  constructor(
    indexRoot: string = './indexed-content', 
    rerankConfig?: Partial<RerankingConfig>, 
    phaseBConfig?: Partial<PhaseBConfig>,
    enableLSP: boolean = true
  ) {
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
    
    // Initialize Phase B optimizer
    this.phaseBOptimizer = new PhaseBComprehensiveOptimizer(phaseBConfig);

    // Initialize LSP components if enabled
    this.lspEnabled = enableLSP;
    if (this.lspEnabled) {
      this.lspStageBEnhancer = new LSPStageBEnhancer();
      this.lspStageCEnhancer = new LSPStageCEnhancer();
      this.intentRouter = new IntentRouter(this.lspStageBEnhancer);
    }
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

      // *** STEP 1: LSP SIDECAR STARTUP & ACTIVATION ***
      // Initialize and activate LSP components if enabled
      if (this.lspEnabled && this.lspStageBEnhancer && this.lspStageCEnhancer) {
        console.log('🚀 Activating LSP integration...');
        await this.initializeLSPSidecars();
        console.log('✅ LSP integration activated successfully');
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
  async search(ctx: SearchContext): Promise<LensSearchResult> {
    if (!this.isInitialized) {
      throw new Error('Search engine not initialized');
    }

    // Use Phase B optimized search if enabled
    if (this.phaseBEnabled) {
      return this.searchWithPhaseBOptimizations(ctx);
    }

    // Use LSP intent routing if enabled
    if (this.lspEnabled && this.intentRouter) {
      return this.searchWithLSPIntentRouting(ctx);
    }

    // Declare hardnessScore at function scope for use across stages
    let hardnessScore = 0;

    const overallSpan = LensTracer.startSearchSpan(ctx);
    this.activeQueries++;

    try {
      let hits: SearchHit[] = [];
      let stageALatency = 0;
      let stageBLatency = 0;
      let stageCLatency: number | undefined;

      // Stage A: Lexical + Fuzzy Search (2-8ms target) - Enhanced with Adaptive Fan-out
      const stageASpan = LensTracer.startStageSpan(ctx, 'stage_a', 'lexical+fuzzy', 0);
      const stageAStart = Date.now();
      
      try {
        // Get IndexReader for this repository
        if (!this.indexRegistry.hasRepo(ctx.repo_sha)) {
          throw new Error(`INDEX_MISSING: Repository not found in index: ${ctx.repo_sha}`);
        }
        
        const reader = this.indexRegistry.getReader(ctx.repo_sha);
        
        // Calculate hardness and adaptive parameters
        let adaptiveK = ctx.k * 4;
        
        if (globalAdaptiveFanout.isEnabled()) {
          const features = globalAdaptiveFanout.extractFeatures(ctx.query, ctx);
          hardnessScore = globalAdaptiveFanout.calculateHardness(features);
          adaptiveK = globalAdaptiveFanout.getAdaptiveKCandidates(hardnessScore);
          
          console.log(`🎯 Adaptive fan-out: hardness=${hardnessScore.toFixed(3)}, k_candidates=${adaptiveK}`, {
            query: ctx.query,
            features: features,
            adaptive_k: adaptiveK
          });
        }
        
        // Use IndexReader for lexical search with adaptive k
        const lexicalResults = await reader.searchLexical({
          q: ctx.query,
          fuzzy: Math.min(2, Math.max(0, Math.round((ctx.fuzzy_distance || 0) * 2))),
          subtokens: true,
          k: Math.min(500, adaptiveK), // Cap at 500 per safety requirements
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

      // Stage C: Semantic Rerank (5-15ms target) - Enhanced with Adaptive Gates
      let stageCGateThreshold = 10; // Default min_candidates
      let nlThreshold = 0.35; // Default from existing logic
      
      if (globalAdaptiveFanout.isEnabled()) {
        const adaptiveParams = globalAdaptiveFanout.getAdaptiveParameters(hardnessScore);
        stageCGateThreshold = adaptiveParams.min_candidates;
        nlThreshold = adaptiveParams.nl_threshold;
        
        console.log(`🧠 Adaptive Stage-C gates: min_candidates=${stageCGateThreshold}, nl_threshold=${nlThreshold.toFixed(3)}, hardness=${hardnessScore.toFixed(3)}`);
      }
      
      if (hits.length > stageCGateThreshold && hits.length <= PRODUCTION_CONFIG.performance.max_candidates) {
        const stageCSpan = LensTracer.startStageSpan(ctx, 'stage_c', 'semantic_rerank', hits.length);
        const stageCStart = Date.now();

        try {
          const candidatesForRerank = this.convertHitsToCandidates(hits);
          
          // Apply work-conserving ANN if enabled
          let finalCandidates: any[] = candidatesForRerank;
          
          if (globalWorkConservingANN.isEnabled()) {
            // Convert hits to format expected by work-conserving ANN
            const annCandidates = hits.map(hit => ({
              score: hit.score,
              file: hit.file,
              line: hit.line,
              snippet: hit.snippet ?? '',
              why: hit.why.join(',') // Convert array to string
            }));
            
            const annResults = await globalWorkConservingANN.search(annCandidates, ctx.query);
            
            // Convert back to SearchHit format
            hits = annResults.map(result => ({
              file: result.file,
              line: result.line,
              col: 0, // Default
              lang: 'unknown', // Default
              snippet: result.snippet,
              score: result.score,
              why: [result.why as any], // Convert string back to array
              byte_offset: 0, // Default
              span_len: result.snippet.length
            }));
            
            console.log(`🧠 Work-conserving ANN: ${annCandidates.length} → ${hits.length} candidates`);
          } else {
            // Traditional semantic reranking
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
          }
          
          stageCLatency = Date.now() - stageCStart;

          // Apply precision optimizations after semantic rerank
          hits = await this.applyPrecisionOptimizations(hits, ctx);

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

      const result: LensSearchResult = {
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
   * Enable/disable Phase B optimizations
   */
  setPhaseBOptimizationsEnabled(enabled: boolean): void {
    this.phaseBEnabled = enabled;
    console.log(`🚀 Phase B optimizations ${enabled ? 'ENABLED' : 'DISABLED'}`);
  }

  /**
   * Phase B optimized search pipeline
   */
  private async searchWithPhaseBOptimizations(ctx: SearchContext): Promise<LensSearchResult> {
    const overallSpan = LensTracer.startSearchSpan(ctx);
    this.activeQueries++;

    try {
      const optimizedResult = await this.phaseBOptimizer.executeOptimizedSearch(ctx);

      LensTracer.endSearchSpan(overallSpan, ctx, (optimizedResult as any).hits?.length ?? 0);

      return {
        hits: (optimizedResult as any).hits ?? [],
        stage_a_latency: (optimizedResult as any).stage_a_latency,
        stage_b_latency: (optimizedResult as any).stage_b_latency,
        stage_c_latency: (optimizedResult as any).stage_c_latency,
      };

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      LensTracer.endSearchSpan(overallSpan, ctx, 0, errorMsg);
      throw error;

    } finally {
      this.activeQueries--;
    }
  }

  /**
   * Run Phase B benchmark suite
   */
  async runPhaseBBenchmark(): Promise<any> {
    const span = LensTracer.createChildSpan('phase_b_benchmark');

    try {
      console.log('🎯 Starting Phase B benchmark suite...');
      const benchmarkResult = await this.phaseBOptimizer.runComprehensiveBenchmark();
      
      console.log('📊 Phase B Benchmark completed:', {
        status: benchmarkResult.overall_status,
        stage_a_p95: `${benchmarkResult.stage_a_p95_ms}ms`,
        meets_targets: benchmarkResult.meets_performance_targets && benchmarkResult.meets_quality_targets,
      });

      span.setAttributes({
        success: true,
        benchmark_status: benchmarkResult.overall_status,
        stage_a_p95_ms: benchmarkResult.stage_a_p95_ms,
        meets_targets: benchmarkResult.meets_performance_targets && benchmarkResult.meets_quality_targets,
      });

      return benchmarkResult;

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw new Error(`Phase B benchmark failed: ${errorMsg}`);
    } finally {
      span.end();
    }
  }

  /**
   * Generate calibration plot data for Phase B reporting
   */
  async generateCalibrationPlot(testData?: Array<{ predicted_score: number; actual_relevance: number }>): Promise<any> {
    const span = LensTracer.createChildSpan('generate_calibration_plot');

    try {
      // Use mock test data if none provided
      const mockTestData = testData || [
        { predicted_score: 0.9, actual_relevance: 0.85 },
        { predicted_score: 0.8, actual_relevance: 0.82 },
        { predicted_score: 0.7, actual_relevance: 0.68 },
        { predicted_score: 0.6, actual_relevance: 0.63 },
        { predicted_score: 0.5, actual_relevance: 0.52 },
        { predicted_score: 0.4, actual_relevance: 0.38 },
        { predicted_score: 0.3, actual_relevance: 0.31 },
        { predicted_score: 0.2, actual_relevance: 0.22 },
        { predicted_score: 0.1, actual_relevance: 0.09 },
      ];

      const calibrationData = await this.phaseBOptimizer.generateCalibrationPlotData(mockTestData);

      span.setAttributes({
        success: true,
        calibration_error: calibrationData.calibration_error,
        reliability_score: calibrationData.reliability_score,
        bins_count: calibrationData.bins.length,
      });

      return calibrationData;

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw new Error(`Calibration plot generation failed: ${errorMsg}`);
    } finally {
      span.end();
    }
  }

  /**
   * Update Stage-A configuration for Phase B optimizations
   */
  async updateStageAConfig(config: {
    rare_term_fuzzy?: boolean;
    synonyms_when_identifier_density_below?: number;
    prefilter_enabled?: boolean;
    prefilter_type?: string;
    wand_enabled?: boolean;
    wand_block_max?: boolean;
    per_file_span_cap?: number;
    native_scanner?: 'on' | 'off' | 'auto';
    k_candidates?: string | number;
    fanout_features?: string;
    adaptive_enabled?: boolean;
  }): Promise<void> {
    const span = LensTracer.createChildSpan('update_stage_a_config');

    try {
      // Update lexical engine configuration
      if (this.lexicalEngine) {
        const updateParams: any = {};
        if (config.rare_term_fuzzy !== undefined) updateParams.rareTermFuzzy = config.rare_term_fuzzy;
        if (config.synonyms_when_identifier_density_below !== undefined) updateParams.synonymsWhenIdentifierDensityBelow = config.synonyms_when_identifier_density_below;
        if (config.prefilter_enabled !== undefined) updateParams.prefilterEnabled = config.prefilter_enabled;
        if (config.prefilter_type !== undefined) updateParams.prefilterType = config.prefilter_type;
        if (config.wand_enabled !== undefined) updateParams.wandEnabled = config.wand_enabled;
        if (config.wand_block_max !== undefined) updateParams.wandBlockMax = config.wand_block_max;
        if (config.per_file_span_cap !== undefined) updateParams.perFileSpanCap = config.per_file_span_cap;
        if (config.native_scanner !== undefined) updateParams.nativeScanner = config.native_scanner;
        
        await this.lexicalEngine.updateConfig(updateParams);
      }

      console.log('🔧 Stage-A configuration updated:', {
        rare_term_fuzzy: config.rare_term_fuzzy,
        synonyms_when_identifier_density_below: config.synonyms_when_identifier_density_below,
        prefilter_enabled: config.prefilter_enabled,
        prefilter_type: config.prefilter_type,
        wand_enabled: config.wand_enabled,
        wand_block_max: config.wand_block_max,
        per_file_span_cap: config.per_file_span_cap,
        native_scanner: config.native_scanner,
      });

      span.setAttributes({
        success: true,
        rare_term_fuzzy: config.rare_term_fuzzy || false,
        synonyms_when_identifier_density_below: config.synonyms_when_identifier_density_below || 0,
        prefilter_enabled: config.prefilter_enabled || false,
        wand_enabled: config.wand_enabled || false,
        per_file_span_cap: config.per_file_span_cap || 0,
        native_scanner: config.native_scanner || 'off',
      });

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw new Error(`Failed to update Stage-A config: ${errorMsg}`);
    } finally {
      span.end();
    }
  }

  /**
   * Update semantic stage configuration for parameter tuning (Phase 3)
   */
  async updateSemanticConfig(config: {
    nl_threshold?: number | string;
    min_candidates?: number | string;
    efSearch?: number | string;
    confidence_cutoff?: number;
    ann_k?: number;
    early_exit?: {
      after_probes?: number;
      margin_tau?: number;
      guards?: {
        require_symbol_or_struct?: boolean;
        min_top1_top5_margin?: number;
      };
    };
    adaptive_gates_enabled?: boolean;
  }): Promise<void> {
    const span = LensTracer.createChildSpan('update_semantic_config');

    try {
      // Update semantic engine configuration
      if (this.semanticEngine) {
        const semanticParams: any = {};
        if (config.nl_threshold !== undefined) {
          semanticParams.nlThreshold = typeof config.nl_threshold === 'string' ? parseFloat(config.nl_threshold) : config.nl_threshold;
        }
        if (config.min_candidates !== undefined) {
          semanticParams.minCandidates = typeof config.min_candidates === 'string' ? parseInt(config.min_candidates) : config.min_candidates;
        }
        if (config.efSearch !== undefined) {
          semanticParams.efSearch = typeof config.efSearch === 'string' ? parseInt(config.efSearch) : config.efSearch;
        }
        if (config.confidence_cutoff !== undefined) {
          semanticParams.confidenceCutoff = config.confidence_cutoff;
        }
        
        await this.semanticEngine.updateConfig(semanticParams);
      }

      // Configure work-conserving ANN if provided
      let annEnabled = false;
      if (config.ann_k !== undefined || config.early_exit !== undefined) {
        const annConfig: any = {};
        
        if (config.ann_k !== undefined) {
          annConfig.k = config.ann_k;
        }
        
        if (config.early_exit !== undefined) {
          annConfig.early_exit = config.early_exit;
        }
        
        globalWorkConservingANN.updateConfig(annConfig);
        annEnabled = true;
      }
      
      // Enable/disable work-conserving ANN based on configuration
      globalWorkConservingANN.setEnabled(annEnabled);

      console.log('🔧 Semantic configuration updated:', {
        nl_threshold: config.nl_threshold,
        min_candidates: config.min_candidates,
        efSearch: config.efSearch,
        confidence_cutoff: config.confidence_cutoff,
        ann_k: config.ann_k,
        early_exit_enabled: config.early_exit !== undefined,
        work_conserving_ann_enabled: annEnabled,
        adaptive_gates_enabled: config.adaptive_gates_enabled,
      });

      span.setAttributes({
        success: true,
        nl_threshold: config.nl_threshold || 0,
        min_candidates: config.min_candidates || 0,
        efSearch: config.efSearch || 0,
        confidence_cutoff: config.confidence_cutoff || 0,
      });

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw new Error(`Failed to update semantic config: ${errorMsg}`);
    } finally {
      span.end();
    }
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
   * Get manifest mapping repo_ref to repo_sha with version information
   */
  async getManifest(): Promise<{ [repo_ref: string]: { 
    repo_sha: string; 
    api_version: string; 
    index_version: string; 
    policy_version: string; 
  } }> {
    const span = LensTracer.createChildSpan('get_manifest');

    try {
      const manifests = this.indexRegistry.getManifests();
      const mapping: { [repo_ref: string]: { 
        repo_sha: string; 
        api_version: string; 
        index_version: string; 
        policy_version: string; 
      } } = {};
      
      for (const manifest of manifests) {
        mapping[manifest.repo_ref] = {
          repo_sha: manifest.repo_sha,
          api_version: manifest.api_version,
          index_version: manifest.index_version,
          policy_version: manifest.policy_version,
        };
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
   * Get LSP activation status and statistics
   * *** STEP 5: LSP VALIDATION & TESTING ***
   */
  getLSPActivationStatus(): {
    lsp_enabled: boolean;
    stage_b_ready: boolean;
    stage_c_ready: boolean;
    intent_router_ready: boolean;
    stage_b_stats?: any;
    stage_c_stats?: any;
    intent_router_stats?: any;
  } {
    const status = {
      lsp_enabled: this.lspEnabled,
      stage_b_ready: false,
      stage_c_ready: false,
      intent_router_ready: false,
    };

    if (this.lspStageBEnhancer) {
      status.stage_b_ready = true;
      (status as any).stage_b_stats = this.lspStageBEnhancer.getStats();
    }

    if (this.lspStageCEnhancer) {
      status.stage_c_ready = true;
      (status as any).stage_c_stats = this.lspStageCEnhancer.getStats();
    }

    if (this.intentRouter) {
      status.intent_router_ready = true;
      (status as any).intent_router_stats = this.intentRouter.getStats();
    }

    return status;
  }

  /**
   * Initialize LSP sidecars for all repositories
   */
  private async initializeLSPSidecars(): Promise<void> {
    const span = LensTracer.createChildSpan('initialize_lsp_sidecars');
    
    try {
      console.log('🚀 Initializing LSP sidecars...');
      
      const manifests = this.indexRegistry.getManifests();
      const lspInitPromises: Promise<void>[] = [];
      
      // Start LSP servers for each repository
      for (const manifest of manifests) {
        const initPromise = this.initializeLSPForRepo(manifest.repo_sha, manifest.repo_ref);
        lspInitPromises.push(initPromise);
      }
      
      // Wait for all LSP servers to initialize
      await Promise.all(lspInitPromises);
      
      console.log(`✅ LSP sidecars initialized for ${manifests.length} repositories`);
      
      span.setAttributes({
        success: true,
        repositories_count: manifests.length
      });
      
    } catch (error) {
      console.error('❌ Failed to initialize LSP sidecars:', error);
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      // Don't throw - LSP is optional, continue without it
      this.lspEnabled = false;
      console.warn('⚠️ Disabling LSP integration due to initialization failure');
    } finally {
      span.end();
    }
  }

  /**
   * Initialize LSP for a specific repository
   */
  private async initializeLSPForRepo(repoSha: string, repoRef: string): Promise<void> {
    const span = LensTracer.createChildSpan('initialize_lsp_for_repo', {
      'repo.sha': repoSha,
      'repo.ref': repoRef
    });
    
    try {
      const reader = this.indexRegistry.getReader(repoSha);
      const indexedFiles = await reader.getFileList();
      
      // Detect primary language for the repository
      const language = this.detectPrimaryLanguage(indexedFiles);
      
      if (!language) {
        console.log(`⏭️ Skipping LSP for ${repoRef}: no supported language detected`);
        return;
      }
      
      // Start LSP sidecar and harvest hints
      const success = await this.harvestLSPHintsForRepo(repoSha, repoRef, language, indexedFiles);
      
      if (success) {
        // Load hints into Stage B and C enhancers
        await this.loadLSPHintsForRepo(repoSha, language);
        console.log(`✅ LSP activated for ${repoRef} (${language})`);
      } else {
        console.warn(`⚠️ LSP harvest failed for ${repoRef}`);
      }
      
      span.setAttributes({
        success: success,
        language: language || 'unknown',
        files_count: indexedFiles.length
      });
      
    } catch (error) {
      console.error(`❌ LSP initialization failed for ${repoRef}:`, error);
      span.recordException(error as Error);
      span.setAttributes({ success: false });
    } finally {
      span.end();
    }
  }

  /**
   * Detect primary language for a repository
   */
  private detectPrimaryLanguage(filePaths: string[]): string | null {
    const langCounts: { [key: string]: number } = {};
    
    for (const filePath of filePaths) {
      const ext = filePath.split('.').pop()?.toLowerCase();
      
      switch (ext) {
        case 'ts':
        case 'tsx':
        case 'js':
        case 'jsx':
          langCounts.typescript = (langCounts.typescript || 0) + 1;
          break;
        case 'py':
          langCounts.python = (langCounts.python || 0) + 1;
          break;
        case 'rs':
          langCounts.rust = (langCounts.rust || 0) + 1;
          break;
        case 'go':
          langCounts.go = (langCounts.go || 0) + 1;
          break;
        case 'java':
          langCounts.java = (langCounts.java || 0) + 1;
          break;
        case 'sh':
        case 'bash':
          langCounts.bash = (langCounts.bash || 0) + 1;
          break;
      }
    }
    
    // Return language with highest count
    let maxLang = null;
    let maxCount = 0;
    
    for (const [lang, count] of Object.entries(langCounts)) {
      if (count > maxCount) {
        maxLang = lang;
        maxCount = count;
      }
    }
    
    return maxLang;
  }

  /**
   * *** STEP 2: HINT HARVESTING IMPLEMENTATION ***
   * Harvest LSP hints for a repository
   */
  private async harvestLSPHintsForRepo(
    repoSha: string, 
    repoRef: string, 
    language: string, 
    filePaths: string[]
  ): Promise<boolean> {
    const span = LensTracer.createChildSpan('harvest_lsp_hints', {
      'repo.sha': repoSha,
      'repo.ref': repoRef,
      'language': language,
      'files.count': filePaths.length
    });
    
    try {
      // Import LSPSidecar dynamically to avoid circular dependencies
      const { LSPSidecar } = await import('../core/lsp-sidecar.js');
      
      // Create sidecar configuration
      const sidecarConfig = {
        language: language as any,
        lsp_server: this.getLSPServerPath(language),
        harvest_ttl_hours: 24,
        pressure_threshold: 512, // MB
        workspace_config: {
          include_patterns: this.getIncludePatternsForLanguage(language),
          exclude_patterns: ['node_modules/**', '.git/**', 'dist/**', 'build/**'],
        },
        capabilities: {} as any // Will be populated during initialization
      };
      
      // Determine workspace root (use indexed content path)
      const workspaceRoot = `./indexed-content/${repoRef}`;
      
      // Create and initialize LSP sidecar
      const sidecar = new LSPSidecar(sidecarConfig, repoSha, workspaceRoot);
      
      console.log(`🔧 Starting LSP server for ${repoRef} (${language})...`);
      
      // Initialize LSP server with timeout
      const initTimeout = new Promise((_, reject) => 
        setTimeout(() => reject(new Error('LSP initialization timeout')), 30000)
      );
      
      await Promise.race([
        sidecar.initialize(),
        initTimeout
      ]);
      
      console.log(`📡 Harvesting LSP hints for ${repoRef}...`);
      
      // Harvest hints with timeout
      const harvestTimeout = new Promise((_, reject) => 
        setTimeout(() => reject(new Error('LSP harvest timeout')), 60000)
      );
      
      const hints = await Promise.race([
        sidecar.harvestHints(filePaths),
        harvestTimeout
      ]) as LSPHint[] | never; // Type assertion since we know it's either LSPHint[] or timeout
      
      // Check if we got actual hints or timeout
      if (!Array.isArray(hints)) {
        throw new Error('LSP harvest timeout');
      }
      
      console.log(`✅ Harvested ${hints.length} LSP hints for ${repoRef}`);
      
      // Shutdown sidecar to free resources
      await sidecar.shutdown();
      
      span.setAttributes({
        success: true,
        hints_harvested: hints.length
      });
      
      return hints.length > 0;
      
    } catch (error) {
      console.error(`❌ LSP harvest failed for ${repoRef}:`, error);
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      return false;
    } finally {
      span.end();
    }
  }

  /**
   * *** STEP 3: BENCHMARK INTEGRATION ***
   * Load LSP hints into enhancers
   */
  private async loadLSPHintsForRepo(repoSha: string, language: string): Promise<void> {
    const span = LensTracer.createChildSpan('load_lsp_hints', {
      'repo.sha': repoSha,
      'language': language
    });
    
    try {
      // Import LSPSidecar to load hints from disk
      const { LSPSidecar } = await import('../core/lsp-sidecar.js');
      
      const sidecarConfig = {
        language: language as any,
        lsp_server: '',
        harvest_ttl_hours: 24,
        pressure_threshold: 512,
        workspace_config: { include_patterns: [], exclude_patterns: [] },
        capabilities: {} as any
      };
      
      const workspaceRoot = `./indexed-content`;
      const sidecar = new LSPSidecar(sidecarConfig, repoSha, workspaceRoot);
      
      // Load hints from disk
      const hints = await sidecar.loadHintsFromShard();
      
      if (hints.length > 0) {
        // Load hints into Stage B enhancer
        this.lspStageBEnhancer?.loadHints(hints);
        
        // Load hints into Stage C enhancer
        this.lspStageCEnhancer?.loadHints(hints);
        
        console.log(`📚 Loaded ${hints.length} LSP hints into enhancers for ${repoSha}`);
      } else {
        console.log(`⚠️ No LSP hints found for ${repoSha}`);
      }
      
      span.setAttributes({
        success: true,
        hints_loaded: hints.length
      });
      
    } catch (error) {
      console.error(`❌ Failed to load LSP hints for ${repoSha}:`, error);
      span.recordException(error as Error);
      span.setAttributes({ success: false });
    } finally {
      span.end();
    }
  }

  /**
   * Get LSP server path for a language
   */
  private getLSPServerPath(language: string): string {
    const servers = {
      typescript: 'typescript-language-server',
      python: 'pyright-langserver',
      rust: 'rust-analyzer',
      go: 'gopls',
      java: 'jdtls',
      bash: 'bash-language-server'
    };
    
    return servers[language as keyof typeof servers] || 'typescript-language-server';
  }

  /**
   * Get include patterns for a language
   */
  private getIncludePatternsForLanguage(language: string): string[] {
    const patterns = {
      typescript: ['**/*.ts', '**/*.tsx', '**/*.js', '**/*.jsx'],
      python: ['**/*.py'],
      rust: ['**/*.rs'],
      go: ['**/*.go'],
      java: ['**/*.java'],
      bash: ['**/*.sh', '**/*.bash']
    };
    
    return patterns[language as keyof typeof patterns] || ['**/*'];
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
      snippet: hit.snippet ?? '',
      byte_offset: hit.byte_offset ?? undefined,
      span_len: hit.span_len ?? undefined,
      context_before: hit.context_before ?? undefined,
      context_after: hit.context_after ?? undefined,
      context: hit.context_before || hit.context_after || undefined,
    }));
  }

  /**
   * Apply precision optimizations (Block A, B, C) to search hits
   */
  private async applyPrecisionOptimizations(hits: SearchHit[], ctx: SearchContext): Promise<SearchHit[]> {
    const span = LensTracer.createChildSpan('precision_optimizations');
    
    try {
      let optimizedHits = hits;

      // Apply Block A: Early-exit optimization
      optimizedHits = await globalPrecisionEngine.applyBlockA(optimizedHits, ctx);

      // Apply Block B: Calibrated dynamic_topn  
      optimizedHits = await globalPrecisionEngine.applyBlockB(optimizedHits, ctx);

      // Apply Block C: Gentle deduplication
      optimizedHits = await globalPrecisionEngine.applyBlockC(optimizedHits, ctx);

      span.setAttributes({
        success: true,
        hits_in: hits.length,
        hits_out: optimizedHits.length
      });

      return optimizedHits;

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      console.warn(`Precision optimization failed: ${errorMsg}`);
      return hits; // Return original hits on error
    } finally {
      span.end();
    }
  }

  /**
   * Enable/disable precision optimization blocks
   */
  setPrecisionOptimizationEnabled(block: 'A' | 'B' | 'C', enabled: boolean): void {
    globalPrecisionEngine.setBlockEnabled(block, enabled);
  }

  /**
   * Get precision optimization status
   */
  getPrecisionOptimizationStatus(): any {
    return globalPrecisionEngine.getOptimizationStatus();
  }

  /**
   * LSP Intent-Routed search pipeline
   * *** STEP 4: ACTIVE LSP INTENT ROUTING ***
   */
  private async searchWithLSPIntentRouting(ctx: SearchContext): Promise<LensSearchResult> {
    const overallSpan = LensTracer.startSearchSpan(ctx);
    this.activeQueries++;

    try {
      if (!this.intentRouter) {
        throw new Error('Intent router not initialized');
      }

      console.log(`🎯 LSP Intent routing query: "${ctx.query}" (mode: ${ctx.mode})`);

      // Route query through LSP intent system
      const routingResult = await this.intentRouter.routeQuery(
        ctx.query,
        ctx,
        // Symbols near handler (LSP Stage-B enhancer with fallback to structural search)
        async (filePath: string, line: number) => {
          console.log(`🔍 LSP symbols near: ${filePath}:${line}`);
          
          // First try LSP Stage-B enhancer for nearby symbols
          if (this.lspStageBEnhancer) {
            try {
              const lspCandidates = await this.lspStageBEnhancer.findLSPSymbolsNear(filePath, line, 10);
              if (lspCandidates.length > 0) {
                console.log(`📍 LSP found ${lspCandidates.length} nearby symbols`);
                return lspCandidates; // These are already Candidate objects
              }
            } catch (error) {
              console.warn('LSP symbols near failed, falling back to structural search:', error);
            }
          }
          
          // Fallback to structural search
          const reader = this.indexRegistry.getReader(ctx.repo_sha);
          const structuralResults = await reader.searchStructural({
            q: ctx.query,
            k: Math.min(50, ctx.k),
          });
          
          return structuralResults.map(result => ({
            doc_id: `${result.file}:${result.line}:${result.col}`,
            file_path: result.file,
            file: result.file, // Alternative field name used in some modules
            line: result.line,
            col: result.col,
            lang: result.lang,
            snippet: result.snippet,
            score: result.score,
            // Convert ValidMatchReason[] to MatchReason[] - filter to valid values
            match_reasons: (result.match_reasons || ['struct']).filter(reason => 
              ['exact', 'fuzzy', 'symbol', 'struct', 'semantic', 'lsp_hint'].includes(reason)
            ) as MatchReason[],
            // Why field should be same as match_reasons for consistency
            why: (result.match_reasons || ['struct']).filter(reason => 
              ['exact', 'fuzzy', 'symbol', 'struct', 'semantic', 'lsp_hint'].includes(reason)
            ) as MatchReason[],
            // Copy other optional properties from StructuralResult to Candidate
            byte_offset: result.byte_offset,
            span_len: result.span_len,
            symbol_kind: result.pattern_type === 'function_def' ? 'function' :
                        result.pattern_type === 'class_def' ? 'class' : 
                        undefined,
          }));
        },
        // Full search handler (fallback to vanilla four-stage pipeline)
        async (query: string, context: SearchContext) => {
          console.log(`🔄 LSP fallback to full search for: "${query}"`);
          const vanillaResult = await this.searchVanillaFourStage(context);
          return vanillaResult.hits.map(hit => ({
            doc_id: `${hit.file}:${hit.line}:${hit.col}`,
            file_path: hit.file,
            file: hit.file, // Alternative field name
            line: hit.line,
            col: hit.col,
            lang: hit.lang,
            snippet: hit.snippet,
            score: hit.score,
            // Use valid MatchReason values
            why: hit.why || ['semantic'],
            match_reasons: hit.match_reasons || ['semantic'],
            // Copy other optional properties if available
            ast_path: hit.ast_path,
            symbol_kind: hit.symbol_kind,
            byte_offset: hit.byte_offset,
            span_len: hit.span_len,
            context_before: hit.context_before,
            context_after: hit.context_after,
          }));
        }
      );

      console.log(`🧭 LSP routing result: ${routingResult.classification.intent} (confidence: ${routingResult.classification.confidence.toFixed(2)}), ${routingResult.primary_candidates.length} candidates`);

      // Convert candidates to SearchHit format and add LSP routing info
      const hits: SearchHit[] = routingResult.primary_candidates.map(candidate => ({
        file: candidate.file_path,
        line: candidate.line,
        col: candidate.col,
        lang: candidate.lang || 'unknown',
        snippet: candidate.snippet || candidate.context || '',
        score: candidate.score,
        why: (candidate.match_reasons || ['lsp_hint']).filter((reason): reason is MatchReason => 
          ['exact', 'fuzzy', 'symbol', 'struct', 'semantic', 'lsp_hint'].includes(reason)
        ),
        match_reasons: candidate.match_reasons || ['lsp_hint'],
      }));

      console.log(`✅ LSP Intent routing complete: ${hits.length} hits with LSP markers`);

      LensTracer.endSearchSpan(overallSpan, ctx, hits.length);

      return {
        hits,
        stage_a_latency: 0, // LSP routing bypasses traditional stages
        stage_b_latency: 0,
        stage_c_latency: 0,
      };

    } catch (error) {
      console.error('❌ LSP intent routing failed:', error);
      LensTracer.endSearchSpan(overallSpan, ctx, 0, (error as Error).message);
      
      // Fall back to vanilla search on LSP errors
      console.warn('⚠️ Falling back to vanilla search due to LSP error');
      return this.searchVanillaFourStage(ctx);
      
    } finally {
      this.activeQueries--;
    }
  }

  /**
   * Vanilla four-stage search (original implementation without LSP)
   */
  private async searchVanillaFourStage(ctx: SearchContext): Promise<LensSearchResult> {
    // This would contain the original four-stage search logic
    // For now, implement a minimal version that calls existing stages
    const overallSpan = LensTracer.startSearchSpan(ctx);
    this.activeQueries++;

    try {
      let hits: SearchHit[] = [];
      let hardnessScore = 0;

      // Stage A: Lexical + Fuzzy
      const stageASpan = LensTracer.startStageSpan(ctx, 'stage_a', 'lexical+fuzzy', 0);
      const stageAStart = Date.now();
      
      try {
        // Get lexical candidates
        const lexicalCandidates: LexicalCandidate[] = await this.lexicalEngine.search(
          ctx,
          ctx.query,
          ctx.fuzzy ? 2 : 0 // Convert boolean fuzzy to fuzzy distance
        );

        // Resolve lexical matches to hits
        hits = await resolveLexicalMatches(
          lexicalCandidates, 
          ctx.query,
          ctx.fuzzy ? 2 : 0, // fuzzyDistance
          3 // maxCandidatesPerFile
        );

        const stageALatency = Date.now() - stageAStart;
        LensTracer.endStageSpan(stageASpan, ctx, 'stage_a', 'lexical+fuzzy', 0, hits.length, stageALatency);

      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : 'Unknown error';
        LensTracer.endStageSpan(stageASpan, ctx, 'stage_a', 'lexical+fuzzy', 0, 0, Date.now() - stageAStart, errorMsg);
        throw error;
      }

      LensTracer.endSearchSpan(overallSpan, ctx, hits.length);
      return { hits };

    } catch (error) {
      LensTracer.endSearchSpan(overallSpan, ctx, 0, (error as Error).message);
      throw error;
    } finally {
      this.activeQueries--;
    }
  }
}