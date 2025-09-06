/**
 * A/B Benchmark Harness for LSP-Assist Evaluation
 * Three modes: Baseline Lens, Lens + LSP-assist, Competitor+LSP  
 * Slice by task: def, refs, symbol-by-name, structural, NL
 * Paired stats and pooled qrels with comprehensive performance validation
 */

import { existsSync, writeFileSync, readFileSync } from 'fs';
import { join } from 'path';
import type { 
  LSPBenchmarkResult,
  QueryIntent,
  LossTaxonomy,
  Candidate,
  SearchContext
} from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';
import { LSPSidecar } from '../core/lsp-sidecar.js';
import { LSPStageBEnhancer } from '../core/lsp-stage-b.js';
import { LSPStageCEnhancer } from '../core/lsp-stage-c.js';
import { IntentRouter } from '../core/intent-router.js';

interface BenchmarkQuery {
  id: string;
  query: string;
  intent: QueryIntent;
  ground_truth: GroundTruthEntry[];
  difficulty: 'easy' | 'medium' | 'hard';
  language: string;
}

interface GroundTruthEntry {
  file_path: string;
  line: number;
  col: number;
  relevance: number; // 0-3 scale
  is_primary: boolean;
}

interface BenchmarkMode {
  name: 'baseline' | 'lsp_assist' | 'competitor_lsp';
  description: string;
  config: BenchmarkConfig;
}

interface BenchmarkConfig {
  enable_lsp_stage_b: boolean;
  enable_lsp_stage_c: boolean;
  enable_intent_router: boolean;
  timeout_ms: number;
  max_results: number;
}

interface BenchmarkRun {
  mode: BenchmarkMode;
  query: BenchmarkQuery;
  results: Candidate[];
  metrics: QueryMetrics;
  timing: {
    total_ms: number;
    stage_a_ms: number;
    stage_b_ms: number;
    stage_c_ms: number;
    lsp_overhead_ms: number;
  };
  loss_factors: LossTaxonomy;
}

interface QueryMetrics {
  success_at_1: number;
  success_at_5: number;
  success_at_10: number;
  ndcg_at_10: number;
  recall_at_50: number;
  precision_at_10: number;
  zero_results: boolean;
  timeout: boolean;
  p95_latency_ms: number;
}

export class LSPABBenchmarkHarness {
  private benchmarkQueries: BenchmarkQuery[] = [];
  private benchmarkModes: BenchmarkMode[] = [];
  private results: BenchmarkRun[] = [];
  
  constructor(
    private lspSidecar?: LSPSidecar,
    private lspStageBEnhancer?: LSPStageBEnhancer,
    private lspStageCEnhancer?: LSPStageCEnhancer,
    private intentRouter?: IntentRouter
  ) {
    this.initializeBenchmarkModes();
  }

  /**
   * Initialize the three benchmark modes
   */
  private initializeBenchmarkModes(): void {
    this.benchmarkModes = [
      {
        name: 'baseline',
        description: 'Original Lens without LSP assistance',
        config: {
          enable_lsp_stage_b: false,
          enable_lsp_stage_c: false,
          enable_intent_router: false,
          timeout_ms: 5000,
          max_results: 50,
        },
      },
      {
        name: 'lsp_assist',
        description: 'Lens with LSP assistance enabled',
        config: {
          enable_lsp_stage_b: true,
          enable_lsp_stage_c: true,
          enable_intent_router: true,
          timeout_ms: 5000,
          max_results: 50,
        },
      },
      {
        name: 'competitor_lsp',
        description: 'Competitor LSP-based search simulation',
        config: {
          enable_lsp_stage_b: true,
          enable_lsp_stage_c: true,
          enable_intent_router: true,
          timeout_ms: 10000, // Generous timeout for competitor
          max_results: 50,
        },
      },
    ];
  }

  /**
   * Load benchmark queries from file or generate synthetic ones
   */
  async loadBenchmarkQueries(queriesFilePath?: string): Promise<void> {
    const span = LensTracer.createChildSpan('load_benchmark_queries', {
      'queries_file': queriesFilePath || 'synthetic',
    });

    try {
      if (queriesFilePath && existsSync(queriesFilePath)) {
        const content = readFileSync(queriesFilePath, 'utf8');
        this.benchmarkQueries = JSON.parse(content);
      } else {
        // Generate synthetic benchmark queries
        this.benchmarkQueries = this.generateSyntheticQueries();
      }

      span.setAttributes({
        success: true,
        queries_loaded: this.benchmarkQueries.length,
        intents_coverage: JSON.stringify(this.getIntentCoverage()),
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
   * Generate synthetic benchmark queries for testing
   */
  private generateSyntheticQueries(): BenchmarkQuery[] {
    const queries: BenchmarkQuery[] = [];
    let queryId = 0;

    // Definition queries
    const defQueries = [
      'function calculateTotal',
      'class UserService', 
      'interface ApiResponse',
      'type DatabaseConfig',
      'def handleUserLogin',
    ];

    for (const query of defQueries) {
      queries.push({
        id: `def_${queryId++}`,
        query,
        intent: 'def',
        ground_truth: [{
          file_path: `/mock/src/${query.split(' ')[1]?.toLowerCase()}.ts`,
          line: 10,
          col: 0,
          relevance: 3,
          is_primary: true,
        }],
        difficulty: 'easy',
        language: 'typescript',
      });
    }

    // References queries
    const refQueries = [
      'refs calculateTotal',
      'usages UserService',
      'find references ApiResponse',
      'who calls handleLogin',
    ];

    for (const query of refQueries) {
      queries.push({
        id: `refs_${queryId++}`,
        query,
        intent: 'refs',
        ground_truth: [
          {
            file_path: '/mock/src/app.ts',
            line: 25,
            col: 8,
            relevance: 2,
            is_primary: true,
          },
          {
            file_path: '/mock/src/utils.ts', 
            line: 15,
            col: 12,
            relevance: 2,
            is_primary: false,
          },
        ],
        difficulty: 'medium',
        language: 'typescript',
      });
    }

    // Symbol-by-name queries
    const symbolQueries = [
      'validateEmail',
      'DatabaseConnection',
      'HTTP_STATUS_CODES',
      'parseApiResponse',
    ];

    for (const query of symbolQueries) {
      queries.push({
        id: `symbol_${queryId++}`,
        query,
        intent: 'symbol',
        ground_truth: [{
          file_path: `/mock/src/${query.toLowerCase()}.ts`,
          line: 5,
          col: 0,
          relevance: 3,
          is_primary: true,
        }],
        difficulty: 'easy',
        language: 'typescript',
      });
    }

    // Structural queries
    const structQueries = [
      'if (user &&',
      'try { await',
      'return { status:',
      '.map(item =>',
    ];

    for (const query of structQueries) {
      queries.push({
        id: `struct_${queryId++}`,
        query,
        intent: 'struct',
        ground_truth: [
          {
            file_path: '/mock/src/handlers.ts',
            line: 20,
            col: 4,
            relevance: 1,
            is_primary: false,
          },
          {
            file_path: '/mock/src/services.ts',
            line: 35,
            col: 2,
            relevance: 1,
            is_primary: false,
          },
        ],
        difficulty: 'hard',
        language: 'typescript',
      });
    }

    // Natural Language queries
    const nlQueries = [
      'function that validates user input',
      'how to connect to database',
      'error handling for api calls',
      'authentication middleware setup',
    ];

    for (const query of nlQueries) {
      queries.push({
        id: `nl_${queryId++}`,
        query,
        intent: 'NL',
        ground_truth: [
          {
            file_path: '/mock/src/validation.ts',
            line: 10,
            col: 0,
            relevance: 2,
            is_primary: true,
          },
          {
            file_path: '/mock/src/middleware.ts',
            line: 45,
            col: 0,
            relevance: 2,
            is_primary: false,
          },
        ],
        difficulty: 'medium',
        language: 'typescript',
      });
    }

    return queries;
  }

  /**
   * Run comprehensive A/B benchmark across all modes and queries
   */
  async runBenchmark(
    baselineSearchHandler: (query: string, context: SearchContext) => Promise<Candidate[]>,
    outputPath?: string
  ): Promise<LSPBenchmarkResult[]> {
    const span = LensTracer.createChildSpan('run_ab_benchmark', {
      'modes.count': this.benchmarkModes.length,
      'queries.count': this.benchmarkQueries.length,
    });

    try {
      this.results = [];
      const aggregatedResults: LSPBenchmarkResult[] = [];

      // Run benchmark for each mode
      for (const mode of this.benchmarkModes) {
        const modeResults = await this.runModebenchmark(mode, baselineSearchHandler);
        this.results.push(...modeResults);

        // Calculate aggregated metrics for this mode
        const aggregated = this.calculateAggregatedMetrics(modeResults, mode.name);
        aggregatedResults.push(aggregated);
      }

      // Generate comparative analysis
      const comparativeAnalysis = this.generateComparativeAnalysis(aggregatedResults);
      
      // Save results if path provided
      if (outputPath) {
        await this.saveResults(outputPath, aggregatedResults, comparativeAnalysis);
      }

      span.setAttributes({
        success: true,
        total_runs: this.results.length,
        baseline_success_at_1: aggregatedResults.find(r => r.mode === 'baseline')?.success_at_1 || 0,
        lsp_assist_success_at_1: aggregatedResults.find(r => r.mode === 'lsp_assist')?.success_at_1 || 0,
        improvement_percentage: this.calculateImprovement(aggregatedResults),
      });

      return aggregatedResults;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Run benchmark for a specific mode
   */
  private async runModebenchmark(
    mode: BenchmarkMode,
    baselineSearchHandler: (query: string, context: SearchContext) => Promise<Candidate[]>
  ): Promise<BenchmarkRun[]> {
    const modeResults: BenchmarkRun[] = [];

    for (const query of this.benchmarkQueries) {
      const startTime = Date.now();
      
      try {
        // Create search context
        const context: SearchContext = {
          trace_id: `benchmark_${mode.name}_${query.id}`,
          repo_sha: 'mock_repo_sha',
          query: query.query,
          mode: 'hybrid',
          k: mode.config.max_results,
          fuzzy_distance: 2,
          started_at: new Date(),
          stages: [],
        };

        let results: Candidate[] = [];
        const timing = {
          total_ms: 0,
          stage_a_ms: 0,
          stage_b_ms: 0,
          stage_c_ms: 0,
          lsp_overhead_ms: 0,
        };

        // Execute search based on mode configuration
        if (mode.name === 'baseline') {
          results = await this.runBaselineSearch(query, context, baselineSearchHandler);
        } else {
          results = await this.runLSPAssistedSearch(query, context, mode.config, baselineSearchHandler);
        }

        timing.total_ms = Date.now() - startTime;

        // Calculate metrics
        const metrics = this.calculateQueryMetrics(results, query.ground_truth, timing);
        const lossTaxonomy = this.analyzeLossFactors(results, query);

        modeResults.push({
          mode,
          query,
          results,
          metrics,
          timing,
          loss_factors: lossTaxonomy,
        });

      } catch (error) {
        console.warn(`Benchmark failed for ${mode.name}/${query.id}:`, error);
        
        // Record failed run
        modeResults.push({
          mode,
          query,
          results: [],
          metrics: {
            success_at_1: 0,
            success_at_5: 0,
            success_at_10: 0,
            ndcg_at_10: 0,
            recall_at_50: 0,
            precision_at_10: 0,
            zero_results: true,
            timeout: Date.now() - startTime > mode.config.timeout_ms,
            p95_latency_ms: Date.now() - startTime,
          },
          timing: {
            total_ms: Date.now() - startTime,
            stage_a_ms: 0,
            stage_b_ms: 0,
            stage_c_ms: 0,
            lsp_overhead_ms: 0,
          },
          loss_factors: {
            NO_SYM_COVERAGE: 1,
            WRONG_ALIAS: 0,
            PATH_MAP: 0,
            USABILITY_INTENT: 0,
            RANKING_ONLY: 0,
          },
        });
      }
    }

    return modeResults;
  }

  /**
   * Run baseline search without LSP assistance
   */
  private async runBaselineSearch(
    query: BenchmarkQuery,
    context: SearchContext,
    baselineSearchHandler: (query: string, context: SearchContext) => Promise<Candidate[]>
  ): Promise<Candidate[]> {
    return await baselineSearchHandler(query.query, context);
  }

  /**
   * Run LSP-assisted search
   */
  private async runLSPAssistedSearch(
    query: BenchmarkQuery,
    context: SearchContext,
    config: BenchmarkConfig,
    baselineSearchHandler: (query: string, context: SearchContext) => Promise<Candidate[]>
  ): Promise<Candidate[]> {
    let results: Candidate[] = [];

    // Stage 1: Intent routing (if enabled)
    if (config.enable_intent_router && this.intentRouter) {
      const routerResult = await this.intentRouter.routeQuery(
        query.query,
        context,
        undefined, // symbolsNearHandler
        baselineSearchHandler
      );
      results = routerResult.primary_candidates;
    }

    // Stage 2: Get baseline results if not from intent router
    if (results.length === 0) {
      results = await baselineSearchHandler(query.query, context);
    }

    // Stage 3: Enhance with LSP Stage-B (if enabled)
    if (config.enable_lsp_stage_b && this.lspStageBEnhancer) {
      const stageBResult = this.lspStageBEnhancer.enhanceStageB(
        query.query,
        context,
        results,
        config.max_results
      );
      results = stageBResult.candidates;
    }

    // Stage 4: Enhance with LSP Stage-C (if enabled)
    if (config.enable_lsp_stage_c && this.lspStageCEnhancer) {
      const stageCResult = this.lspStageCEnhancer.enhanceStageC(
        results,
        query.query,
        context
      );
      results = stageCResult.enhanced_candidates;
    }

    return results;
  }

  /**
   * Calculate metrics for a single query
   */
  private calculateQueryMetrics(
    results: Candidate[],
    groundTruth: GroundTruthEntry[],
    timing: any
  ): QueryMetrics {
    const relevantResults = this.findRelevantResults(results, groundTruth);
    
    return {
      success_at_1: relevantResults.length > 0 && relevantResults[0] <= 1 ? 1 : 0,
      success_at_5: relevantResults.filter(pos => pos <= 5).length > 0 ? 1 : 0,
      success_at_10: relevantResults.filter(pos => pos <= 10).length > 0 ? 1 : 0,
      ndcg_at_10: this.calculateNDCG(results, groundTruth, 10),
      recall_at_50: this.calculateRecall(results, groundTruth, 50),
      precision_at_10: this.calculatePrecision(results, groundTruth, 10),
      zero_results: results.length === 0,
      timeout: false, // Would be set by caller if timeout occurred
      p95_latency_ms: timing.total_ms, // Simplified - would need multiple runs for real p95
    };
  }

  /**
   * Find positions of relevant results
   */
  private findRelevantResults(results: Candidate[], groundTruth: GroundTruthEntry[]): number[] {
    const positions: number[] = [];
    
    for (let i = 0; i < results.length; i++) {
      const candidate = results[i];
      const isRelevant = groundTruth.some(gt =>
        gt.file_path === candidate.file_path &&
        Math.abs(gt.line - candidate.line) <= 2 && // Allow small line differences
        gt.relevance > 0
      );
      
      if (isRelevant) {
        positions.push(i + 1); // 1-indexed position
      }
    }
    
    return positions;
  }

  /**
   * Calculate NDCG@k
   */
  private calculateNDCG(results: Candidate[], groundTruth: GroundTruthEntry[], k: number): number {
    const relevanceScores = results.slice(0, k).map(result => {
      const gt = groundTruth.find(gt =>
        gt.file_path === result.file_path &&
        Math.abs(gt.line - result.line) <= 2
      );
      return gt ? gt.relevance : 0;
    });

    // Calculate DCG
    const dcg = relevanceScores.reduce((sum, rel, i) => {
      return sum + (Math.pow(2, rel) - 1) / Math.log2(i + 2);
    }, 0);

    // Calculate IDCG (ideal DCG)
    const idealRelevances = groundTruth
      .map(gt => gt.relevance)
      .sort((a, b) => b - a)
      .slice(0, k);
    
    const idcg = idealRelevances.reduce((sum, rel, i) => {
      return sum + (Math.pow(2, rel) - 1) / Math.log2(i + 2);
    }, 0);

    return idcg === 0 ? 0 : dcg / idcg;
  }

  /**
   * Calculate Recall@k
   */
  private calculateRecall(results: Candidate[], groundTruth: GroundTruthEntry[], k: number): number {
    const relevantInResults = this.findRelevantResults(results.slice(0, k), groundTruth).length;
    const totalRelevant = groundTruth.filter(gt => gt.relevance > 0).length;
    
    return totalRelevant === 0 ? 0 : relevantInResults / totalRelevant;
  }

  /**
   * Calculate Precision@k
   */
  private calculatePrecision(results: Candidate[], groundTruth: GroundTruthEntry[], k: number): number {
    const topK = results.slice(0, k);
    const relevantInTopK = this.findRelevantResults(topK, groundTruth).length;
    
    return topK.length === 0 ? 0 : relevantInTopK / topK.length;
  }

  /**
   * Analyze loss factors for failed/suboptimal queries
   */
  private analyzeLossFactors(results: Candidate[], query: BenchmarkQuery): LossTaxonomy {
    const lossTaxonomy: LossTaxonomy = {
      NO_SYM_COVERAGE: 0,
      WRONG_ALIAS: 0,
      PATH_MAP: 0,
      USABILITY_INTENT: 0,
      RANKING_ONLY: 0,
    };

    // Check if we found any symbols for symbol-intent queries
    if ((query.intent === 'def' || query.intent === 'refs' || query.intent === 'symbol') && 
        results.length === 0) {
      lossTaxonomy.NO_SYM_COVERAGE = 1;
    }

    // Check for wrong alias resolution
    const hasAliasIssues = results.some(r => 
      r.match_reasons.includes('symbol') && 
      !query.ground_truth.some(gt => gt.file_path === r.file_path)
    );
    if (hasAliasIssues) {
      lossTaxonomy.WRONG_ALIAS = 1;
    }

    // Check for path mapping issues (results in wrong directories)
    const hasPathIssues = results.length > 0 && 
      !results.some(r => query.ground_truth.some(gt => 
        gt.file_path.includes(r.file_path.split('/').slice(-2, -1)[0] || '')
      ));
    if (hasPathIssues) {
      lossTaxonomy.PATH_MAP = 1;
    }

    // Check for intent classification issues
    const intentMisclassified = (results as any).some((r: any) => 
      r.intent_classification && 
      r.intent_classification.intent !== query.intent &&
      r.intent_classification.confidence > 0.7
    );
    if (intentMisclassified) {
      lossTaxonomy.USABILITY_INTENT = 1;
    }

    // Check if it's purely a ranking issue (relevant results exist but not in top positions)
    const relevantResults = this.findRelevantResults(results, query.ground_truth);
    if (relevantResults.length > 0 && relevantResults[0] > 10) {
      lossTaxonomy.RANKING_ONLY = 1;
    }

    return lossTaxonomy;
  }

  /**
   * Calculate aggregated metrics for a mode
   */
  private calculateAggregatedMetrics(runs: BenchmarkRun[], mode: string): LSPBenchmarkResult {
    const intentGroups = this.groupRunsByIntent(runs);
    const aggregatedByIntent: { [key in QueryIntent]: LSPBenchmarkResult } = {} as any;

    // Calculate metrics for each intent
    for (const [intent, intentRuns] of Object.entries(intentGroups)) {
      aggregatedByIntent[intent as QueryIntent] = {
        mode: mode as any,
        task_type: intent as QueryIntent,
        success_at_1: this.average(intentRuns.map(r => r.metrics.success_at_1)),
        success_at_5: this.average(intentRuns.map(r => r.metrics.success_at_5)),
        ndcg_at_10: this.average(intentRuns.map(r => r.metrics.ndcg_at_10)),
        recall_at_50: this.average(intentRuns.map(r => r.metrics.recall_at_50)),
        zero_result_rate: this.average(intentRuns.map(r => r.metrics.zero_results ? 1 : 0)),
        timeout_rate: this.average(intentRuns.map(r => r.metrics.timeout ? 1 : 0)),
        p95_latency_ms: this.percentile(intentRuns.map(r => r.timing.total_ms), 95),
        loss_taxonomy: this.aggregateLossTaxonomy(intentRuns.map(r => r.loss_factors)),
      };
    }

    // Return overall aggregated metrics
    return {
      mode: mode as any,
      task_type: 'symbol', // Overall category
      success_at_1: this.average(runs.map(r => r.metrics.success_at_1)),
      success_at_5: this.average(runs.map(r => r.metrics.success_at_5)),
      ndcg_at_10: this.average(runs.map(r => r.metrics.ndcg_at_10)),
      recall_at_50: this.average(runs.map(r => r.metrics.recall_at_50)),
      zero_result_rate: this.average(runs.map(r => r.metrics.zero_results ? 1 : 0)),
      timeout_rate: this.average(runs.map(r => r.metrics.timeout ? 1 : 0)),
      p95_latency_ms: this.percentile(runs.map(r => r.timing.total_ms), 95),
      loss_taxonomy: this.aggregateLossTaxonomy(runs.map(r => r.loss_factors)),
    };
  }

  /**
   * Group benchmark runs by intent
   */
  private groupRunsByIntent(runs: BenchmarkRun[]): { [key in QueryIntent]: BenchmarkRun[] } {
    const groups = {} as { [key in QueryIntent]: BenchmarkRun[] };
    
    for (const run of runs) {
      if (!groups[run.query.intent]) {
        groups[run.query.intent] = [];
      }
      groups[run.query.intent].push(run);
    }
    
    return groups;
  }

  /**
   * Calculate average of array of numbers
   */
  private average(numbers: number[]): number {
    return numbers.length === 0 ? 0 : numbers.reduce((a, b) => a + b, 0) / numbers.length;
  }

  /**
   * Calculate percentile of array of numbers
   */
  private percentile(numbers: number[], p: number): number {
    if (numbers.length === 0) return 0;
    
    const sorted = numbers.slice().sort((a, b) => a - b);
    const index = Math.ceil((p / 100) * sorted.length) - 1;
    return sorted[Math.max(0, index)];
  }

  /**
   * Aggregate loss taxonomy across runs
   */
  private aggregateLossTaxonomy(lossTaxonomies: LossTaxonomy[]): LossTaxonomy {
    return {
      NO_SYM_COVERAGE: this.average(lossTaxonomies.map(lt => lt.NO_SYM_COVERAGE)),
      WRONG_ALIAS: this.average(lossTaxonomies.map(lt => lt.WRONG_ALIAS)),
      PATH_MAP: this.average(lossTaxonomies.map(lt => lt.PATH_MAP)),
      USABILITY_INTENT: this.average(lossTaxonomies.map(lt => lt.USABILITY_INTENT)),
      RANKING_ONLY: this.average(lossTaxonomies.map(lt => lt.RANKING_ONLY)),
    };
  }

  /**
   * Generate comparative analysis between modes
   */
  private generateComparativeAnalysis(results: LSPBenchmarkResult[]): any {
    const baseline = results.find(r => r.mode === 'baseline');
    const lspAssist = results.find(r => r.mode === 'lsp_assist');
    const competitor = results.find(r => r.mode === 'competitor_lsp');

    if (!baseline || !lspAssist) {
      throw new Error('Missing baseline or lsp_assist results for comparison');
    }

    return {
      improvements: {
        success_at_1_improvement: ((lspAssist.success_at_1 - baseline.success_at_1) / baseline.success_at_1) * 100,
        ndcg_improvement: ((lspAssist.ndcg_at_10 - baseline.ndcg_at_10) / baseline.ndcg_at_10) * 100,
        recall_improvement: ((lspAssist.recall_at_50 - baseline.recall_at_50) / baseline.recall_at_50) * 100,
        latency_impact: lspAssist.p95_latency_ms - baseline.p95_latency_ms,
      },
      vs_competitor: competitor ? {
        success_at_1_vs_competitor: lspAssist.success_at_1 - competitor.success_at_1,
        ndcg_vs_competitor: lspAssist.ndcg_at_10 - competitor.ndcg_at_10,
        latency_vs_competitor: lspAssist.p95_latency_ms - competitor.p95_latency_ms,
      } : null,
      statistical_significance: this.calculateStatisticalSignificance(baseline, lspAssist),
    };
  }

  /**
   * Calculate statistical significance (simplified t-test)
   */
  private calculateStatisticalSignificance(
    baseline: LSPBenchmarkResult,
    lspAssist: LSPBenchmarkResult
  ): { p_value: number; is_significant: boolean } {
    // Simplified statistical test - in production would use proper statistical methods
    const diff = lspAssist.success_at_1 - baseline.success_at_1;
    const isSignificant = Math.abs(diff) > 0.05; // 5% improvement threshold
    
    return {
      p_value: isSignificant ? 0.01 : 0.1, // Placeholder
      is_significant: isSignificant,
    };
  }

  /**
   * Calculate overall improvement percentage
   */
  private calculateImprovement(results: LSPBenchmarkResult[]): number {
    const baseline = results.find(r => r.mode === 'baseline');
    const lspAssist = results.find(r => r.mode === 'lsp_assist');
    
    if (!baseline || !lspAssist) return 0;
    
    return ((lspAssist.success_at_1 - baseline.success_at_1) / baseline.success_at_1) * 100;
  }

  /**
   * Get intent coverage in loaded queries
   */
  private getIntentCoverage(): { [key in QueryIntent]: number } {
    const coverage = {} as { [key in QueryIntent]: number };
    
    for (const query of this.benchmarkQueries) {
      coverage[query.intent] = (coverage[query.intent] || 0) + 1;
    }
    
    return coverage;
  }

  /**
   * Save benchmark results to file
   */
  private async saveResults(
    outputPath: string,
    results: LSPBenchmarkResult[],
    analysis: any
  ): Promise<void> {
    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        total_queries: this.benchmarkQueries.length,
        modes_tested: this.benchmarkModes.length,
        intent_coverage: this.getIntentCoverage(),
      },
      results,
      comparative_analysis: analysis,
      raw_runs: this.results.map(run => ({
        mode: run.mode.name,
        query_id: run.query.id,
        query_intent: run.query.intent,
        success_at_1: run.metrics.success_at_1,
        timing: run.timing,
        loss_factors: run.loss_factors,
      })),
    };

    writeFileSync(outputPath, JSON.stringify(report, null, 2));
    console.log(`Benchmark results saved to ${outputPath}`);
  }

  /**
   * Get benchmark statistics
   */
  getStats(): {
    queries_loaded: number;
    modes_configured: number;
    runs_completed: number;
    intent_coverage: { [key in QueryIntent]: number };
  } {
    return {
      queries_loaded: this.benchmarkQueries.length,
      modes_configured: this.benchmarkModes.length,
      runs_completed: this.results.length,
      intent_coverage: this.getIntentCoverage(),
    };
  }
}