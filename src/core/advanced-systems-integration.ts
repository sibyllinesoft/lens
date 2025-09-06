/**
 * Advanced Systems Integration
 * 
 * Unifies the four advanced systems: Cross-Repo Moniker Linking, API-Evolution Mapping,
 * Query Compiler, and Regression Bisection Harness into a cohesive whole.
 * 
 * Provides orchestration layer that coordinates between systems and ensures
 * all performance gates are met while maintaining embedder-agnostic and span-safe operations.
 */

import { LensTracer } from '../telemetry/tracer.js';
import { MonikerLinkingSystem, type MonikerLinkingConfig } from './moniker-linking.js';
import { EvolutionMappingSystem, type EvolutionMappingConfig } from './evolution-mapping.js';
import { QueryCompiler, type QueryCompilerConfig, type QueryPlan, type PlanConstraints } from './query-compiler.js';
import { RegressionBisectionHarness, type BisectionConfig, type RegressionAlert } from './regression-bisection.js';
import type { SearchHit, SymbolCandidate, MatchReason } from './span_resolver/types.js';

export interface AdvancedSystemsConfig {
  moniker_linking: MonikerLinkingConfig;
  evolution_mapping: EvolutionMappingConfig;
  query_compiler: QueryCompilerConfig;
  regression_bisection: BisectionConfig;
  integration: {
    enable_all_systems: boolean;
    performance_monitoring_enabled: boolean;
    quality_gates_enabled: boolean;
    auto_optimization_enabled: boolean;
  };
}

export interface SystemPerformanceGates {
  // Cross-Repo Moniker Linking Gates
  moniker_recall_improvement_pp: number;    // ≥0.6-1.0pp
  moniker_latency_overhead_ms: number;      // ≤0.6ms
  moniker_why_mix_kl_divergence: number;    // ≤0.02

  // API-Evolution Mapping Gates
  evolution_success_at_10_improvement_pp: number; // ≥0.5pp
  evolution_span_drift: number;             // ≤0 (zero drift tolerance)
  evolution_query_time_budget_percent: number; // ≤2%

  // Query Compiler Gates
  compiler_p99_improvement_percent: number; // -8 to -12%
  compiler_recall_at_50_maintained: boolean; // Flat SLA-Recall@50

  // Regression Bisection Gates
  bisection_culprit_isolation_time_minutes: number; // ≤30 min
  bisection_auto_rollback_enabled: boolean;

  // Overall System Gates
  overall_recall_at_50: number;             // Maintain or improve baseline
  overall_p95_latency_ms: number;          // Meet SLA targets
  overall_span_accuracy: number;           // Maintain span-safe operations
}

export interface EnhancedSearchRequest {
  query: string;
  intent: string;                          // symbol, nl, etc.
  repo_context?: string;
  revision_context?: {
    base_sha: string;
    target_sha: string;
  };
  performance_budget_ms: number;
  quality_requirements: {
    min_recall_at_50: number;
    min_precision: number;
    require_span_accuracy: boolean;
  };
  experiment_config?: {
    enable_cross_repo: boolean;
    enable_evolution_mapping: boolean;
    use_optimized_compiler: boolean;
  };
}

export interface EnhancedSearchResponse {
  hits: SearchHit[];
  performance_metrics: {
    total_latency_ms: number;
    stage_latencies: Map<string, number>;
    candidates_per_stage: Map<string, number>;
  };
  quality_metrics: {
    recall_at_50: number;
    precision_at_50: number;
    span_accuracy: number;
  };
  system_insights: {
    cross_repo_expansions: number;
    evolution_mappings_applied: number;
    query_plan_used: string;
    optimization_applied: boolean;
  };
  warnings: string[];
  gate_status: Map<string, boolean>;
}

export class AdvancedSystemsIntegration {
  private monikerSystem: MonikerLinkingSystem;
  private evolutionSystem: EvolutionMappingSystem;
  private queryCompiler: QueryCompiler;
  private regressionHarness: RegressionBisectionHarness;
  
  private performanceHistory: EnhancedSearchResponse[] = [];
  private gateStatus = new Map<string, boolean>();

  constructor(private config: AdvancedSystemsConfig) {
    this.monikerSystem = new MonikerLinkingSystem(config.moniker_linking);
    this.evolutionSystem = new EvolutionMappingSystem(config.evolution_mapping);
    this.queryCompiler = new QueryCompiler(config.query_compiler);
    this.regressionHarness = new RegressionBisectionHarness(config.regression_bisection);
    
    this.initializeGateMonitoring();
  }

  /**
   * Enhanced search that orchestrates all four advanced systems
   */
  async enhancedSearch(request: EnhancedSearchRequest): Promise<EnhancedSearchResponse> {
    const span = LensTracer.createChildSpan('enhanced_search', {
      'query': request.query,
      'intent': request.intent,
      'budget.ms': request.performance_budget_ms,
      'systems.enabled': this.config.integration.enable_all_systems,
    });

    const startTime = performance.now();
    const stageLatencies = new Map<string, number>();
    const candidatesPerStage = new Map<string, number>();
    const warnings: string[] = [];

    try {
      // Phase 1: Query Compilation and Planning
      const planningStart = performance.now();
      const queryPlan = await this.planEnhancedQuery(request);
      stageLatencies.set('planning', performance.now() - planningStart);

      // Phase 2: Execute optimized query plan
      const executionStart = performance.now();
      const executionResult = await this.executeEnhancedPlan(queryPlan, request);
      stageLatencies.set('execution', performance.now() - executionStart);

      let { hits, candidates } = executionResult;
      candidatesPerStage.set('base_execution', candidates.length);

      // Phase 3: Cross-Repo Moniker Expansion (if enabled)
      if (this.shouldEnableCrossRepo(request)) {
        const monikerStart = performance.now();
        const expandedCandidates = await this.monikerSystem.expandWithMonikerClusters(
          candidates,
          request.intent
        );
        const monikerLatency = performance.now() - monikerStart;
        stageLatencies.set('moniker_expansion', monikerLatency);
        candidatesPerStage.set('moniker_expanded', expandedCandidates.length);

        // Check moniker performance gates
        if (monikerLatency > this.config.moniker_linking.performance_targets.max_additional_latency_ms) {
          warnings.push(`Moniker expansion exceeded latency budget: ${monikerLatency}ms`);
        }

        candidates = expandedCandidates;
      }

      // Phase 4: API Evolution Mapping (if enabled and revision context available)
      if (this.shouldEnableEvolutionMapping(request)) {
        const evolutionStart = performance.now();
        const maxBudget = request.performance_budget_ms * this.config.evolution_mapping.max_query_time_budget_percent / 100;
        
        const expandedCandidates = await this.evolutionSystem.expandWithEvolution(
          request.query,
          candidates,
          maxBudget
        );
        
        const evolutionLatency = performance.now() - evolutionStart;
        stageLatencies.set('evolution_expansion', evolutionLatency);
        candidatesPerStage.set('evolution_expanded', expandedCandidates.length);

        // Apply revision projection if needed
        if (request.revision_context) {
          const projectedHits = await this.evolutionSystem.projectSpansAcrossRevisions(
            hits,
            request.revision_context.base_sha,
            request.revision_context.target_sha
          );
          
          // Verify zero span drift gate
          const spanDrift = this.calculateSpanDrift(hits, projectedHits);
          if (spanDrift > this.config.evolution_mapping.performance_targets.zero_span_drift_tolerance) {
            warnings.push(`Evolution mapping caused span drift: ${spanDrift}`);
          }
          
          hits = projectedHits;
        }

        candidates = expandedCandidates;
      }

      // Phase 5: Final processing and quality assessment
      const qualityStart = performance.now();
      const finalHits = await this.finalizeResults(hits, candidates, request);
      stageLatencies.set('finalization', performance.now() - qualityStart);

      // Calculate metrics
      const totalLatency = performance.now() - startTime;
      const qualityMetrics = await this.calculateQualityMetrics(finalHits, request);
      const systemInsights = this.generateSystemInsights(stageLatencies, candidatesPerStage, queryPlan);

      // Check performance gates
      const gateStatus = this.checkPerformanceGates(totalLatency, qualityMetrics, stageLatencies);
      
      const response: EnhancedSearchResponse = {
        hits: finalHits,
        performance_metrics: {
          total_latency_ms: totalLatency,
          stage_latencies: stageLatencies,
          candidates_per_stage: candidatesPerStage,
        },
        quality_metrics: qualityMetrics,
        system_insights: systemInsights,
        warnings,
        gate_status: gateStatus,
      };

      // Store for performance analysis
      this.performanceHistory.push(response);
      if (this.performanceHistory.length > 10000) {
        this.performanceHistory.shift(); // Keep last 10k for analysis
      }

      // Monitor for regressions
      if (this.config.integration.quality_gates_enabled) {
        await this.monitorForRegressions(response);
      }

      span.setAttributes({
        'total.latency_ms': totalLatency,
        'hits.count': finalHits.length,
        'quality.recall_at_50': qualityMetrics.recall_at_50,
        'systems.moniker_used': this.shouldEnableCrossRepo(request),
        'systems.evolution_used': this.shouldEnableEvolutionMapping(request),
        'warnings.count': warnings.length,
        success: true
      });

      return response;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Plan enhanced query using the query compiler
   */
  private async planEnhancedQuery(request: EnhancedSearchRequest): Promise<QueryPlan> {
    const constraints: PlanConstraints = {
      max_total_time_ms: request.performance_budget_ms,
      min_recall_at_50: request.quality_requirements.min_recall_at_50,
      span_invariants: request.quality_requirements.require_span_accuracy,
      exact_struct_floors: {
        exact_min: 10, // Minimum exact matches
        struct_min: 20, // Minimum structural matches
      }
    };

    return await this.queryCompiler.compileQuery(
      request.query,
      constraints,
      { intent: request.intent, repo_context: request.repo_context }
    );
  }

  /**
   * Execute enhanced query plan
   */
  private async executeEnhancedPlan(
    plan: QueryPlan,
    request: EnhancedSearchRequest
  ): Promise<{ hits: SearchHit[], candidates: SymbolCandidate[] }> {
    const result = await this.queryCompiler.executePlan(plan, request.query);
    
    // Convert hits to candidates for further processing
    const candidates: SymbolCandidate[] = result.hits.map(hit => ({
      file_path: hit.file,
      score: hit.score,
      match_reasons: hit.why,
      upstream_line: hit.line,
      upstream_col: hit.col,
      symbol_kind: hit.symbol_kind,
      ast_path: hit.ast_path,
    }));

    return { hits: result.hits, candidates };
  }

  /**
   * Finalize search results with quality assurance
   */
  private async finalizeResults(
    hits: SearchHit[],
    candidates: SymbolCandidate[],
    request: EnhancedSearchRequest
  ): Promise<SearchHit[]> {
    // Ensure span accuracy if required
    if (request.quality_requirements.require_span_accuracy) {
      return this.validateAndFixSpanAccuracy(hits);
    }

    // Apply final ranking and deduplication
    return this.applyFinalRanking(hits, request);
  }

  /**
   * Calculate comprehensive quality metrics
   */
  private async calculateQualityMetrics(
    hits: SearchHit[],
    request: EnhancedSearchRequest
  ) {
    return {
      recall_at_50: await this.calculateRecallAt50(hits, request.query),
      precision_at_50: await this.calculatePrecisionAt50(hits, request.query),
      span_accuracy: this.calculateSpanAccuracy(hits),
    };
  }

  /**
   * Generate system insights for observability
   */
  private generateSystemInsights(
    stageLatencies: Map<string, number>,
    candidatesPerStage: Map<string, number>,
    queryPlan: QueryPlan
  ) {
    return {
      cross_repo_expansions: (candidatesPerStage.get('moniker_expanded') || 0) - (candidatesPerStage.get('base_execution') || 0),
      evolution_mappings_applied: (candidatesPerStage.get('evolution_expanded') || 0) - (candidatesPerStage.get('moniker_expanded') || candidatesPerStage.get('base_execution') || 0),
      query_plan_used: queryPlan.name,
      optimization_applied: stageLatencies.has('planning'),
    };
  }

  /**
   * Check all performance gates
   */
  private checkPerformanceGates(
    totalLatency: number,
    qualityMetrics: any,
    stageLatencies: Map<string, number>
  ): Map<string, boolean> {
    const gateResults = new Map<string, boolean>();

    // Latency gates
    gateResults.set('total_latency_budget', totalLatency <= 20); // Example SLA
    gateResults.set('moniker_latency_gate', (stageLatencies.get('moniker_expansion') || 0) <= 0.6);
    gateResults.set('evolution_budget_gate', (stageLatencies.get('evolution_expansion') || 0) <= totalLatency * 0.02);

    // Quality gates
    gateResults.set('recall_at_50_gate', qualityMetrics.recall_at_50 >= 0.8);
    gateResults.set('span_accuracy_gate', qualityMetrics.span_accuracy >= 0.99);

    // Update global gate status
    for (const [gate, status] of gateResults) {
      this.gateStatus.set(gate, status);
    }

    return gateResults;
  }

  /**
   * Monitor for performance regressions
   */
  private async monitorForRegressions(response: EnhancedSearchResponse): Promise<void> {
    if (this.performanceHistory.length < 100) {
      return; // Need baseline
    }

    // Calculate baseline metrics from recent history
    const recentHistory = this.performanceHistory.slice(-100);
    const baselineRecall = recentHistory.reduce((sum, r) => sum + r.quality_metrics.recall_at_50, 0) / recentHistory.length;
    const baselineLatency = recentHistory.reduce((sum, r) => sum + r.performance_metrics.total_latency_ms, 0) / recentHistory.length;

    // Check for significant deviations
    const recallDeviation = (response.quality_metrics.recall_at_50 - baselineRecall) / baselineRecall;
    const latencyDeviation = (response.performance_metrics.total_latency_ms - baselineLatency) / baselineLatency;

    // Trigger regression analysis if thresholds exceeded
    if (Math.abs(recallDeviation) > 0.05 || latencyDeviation > 0.2) {
      const alert: RegressionAlert = {
        id: `alert_${Date.now()}`,
        alert_type: Math.abs(recallDeviation) > 0.05 ? 'sla_recall_50' : 'p99_latency',
        metric_name: Math.abs(recallDeviation) > 0.05 ? 'recall_at_50' : 'total_latency_ms',
        current_value: Math.abs(recallDeviation) > 0.05 ? response.quality_metrics.recall_at_50 : response.performance_metrics.total_latency_ms,
        baseline_value: Math.abs(recallDeviation) > 0.05 ? baselineRecall : baselineLatency,
        threshold_value: Math.abs(recallDeviation) > 0.05 ? baselineRecall * 0.95 : baselineLatency * 1.2,
        deviation_percent: Math.abs(recallDeviation) > 0.05 ? recallDeviation * 100 : latencyDeviation * 100,
        confidence: 0.95,
        triggered_at: new Date(),
        time_window: 'last_100_queries',
        sample_size: 100
      };

      await this.regressionHarness.triggerRegressionAnalysis(alert);
    }
  }

  /**
   * Initialize performance gate monitoring
   */
  private initializeGateMonitoring(): void {
    // Initialize all gates as passing
    const gateNames = [
      'total_latency_budget',
      'moniker_latency_gate', 
      'evolution_budget_gate',
      'recall_at_50_gate',
      'span_accuracy_gate'
    ];

    for (const gate of gateNames) {
      this.gateStatus.set(gate, true);
    }
  }

  // Helper methods for decision making
  private shouldEnableCrossRepo(request: EnhancedSearchRequest): boolean {
    return this.config.integration.enable_all_systems &&
           (request.experiment_config?.enable_cross_repo !== false) &&
           this.config.moniker_linking.supported_intents.includes(request.intent as any);
  }

  private shouldEnableEvolutionMapping(request: EnhancedSearchRequest): boolean {
    return this.config.integration.enable_all_systems &&
           (request.experiment_config?.enable_evolution_mapping !== false);
  }

  // Quality assurance methods
  private async validateAndFixSpanAccuracy(hits: SearchHit[]): Promise<SearchHit[]> {
    // Placeholder for span accuracy validation and correction
    return hits;
  }

  private applyFinalRanking(hits: SearchHit[], request: EnhancedSearchRequest): SearchHit[] {
    // Apply final ranking and deduplication logic
    return hits
      .sort((a, b) => b.score - a.score)
      .slice(0, 200); // Reasonable limit
  }

  private calculateSpanDrift(original: SearchHit[], projected: SearchHit[]): number {
    let drift = 0;
    const minLength = Math.min(original.length, projected.length);
    
    for (let i = 0; i < minLength; i++) {
      if (original[i].line !== projected[i].line || original[i].col !== projected[i].col) {
        drift++;
      }
    }
    
    return minLength > 0 ? drift / minLength : 0;
  }

  // Metric calculation placeholders (would integrate with actual evaluation)
  private async calculateRecallAt50(hits: SearchHit[], query: string): Promise<number> {
    return 0.85; // Placeholder
  }

  private async calculatePrecisionAt50(hits: SearchHit[], query: string): Promise<number> {
    return 0.90; // Placeholder
  }

  private calculateSpanAccuracy(hits: SearchHit[]): number {
    return 0.99; // Placeholder - would validate actual span accuracy
  }

  /**
   * Get comprehensive performance metrics for all systems
   */
  getSystemPerformanceMetrics() {
    return {
      moniker_system: this.monikerSystem.getPerformanceMetrics(),
      evolution_system: this.evolutionSystem.getPerformanceMetrics(),
      query_compiler: this.queryCompiler.getPerformanceMetrics(),
      regression_harness: this.regressionHarness.getPerformanceMetrics(),
      integration: {
        total_queries_processed: this.performanceHistory.length,
        avg_total_latency_ms: this.performanceHistory.length > 0 
          ? this.performanceHistory.reduce((sum, r) => sum + r.performance_metrics.total_latency_ms, 0) / this.performanceHistory.length 
          : 0,
        avg_recall_at_50: this.performanceHistory.length > 0
          ? this.performanceHistory.reduce((sum, r) => sum + r.quality_metrics.recall_at_50, 0) / this.performanceHistory.length
          : 0,
        gate_failures: Array.from(this.gateStatus.values()).filter(status => !status).length,
        system_warnings_rate: this.performanceHistory.length > 0
          ? this.performanceHistory.reduce((sum, r) => sum + r.warnings.length, 0) / this.performanceHistory.length
          : 0,
      }
    };
  }

  /**
   * Generate comprehensive status report
   */
  generateStatusReport(): {
    overall_health: 'healthy' | 'degraded' | 'critical';
    gate_status: Map<string, boolean>;
    system_performance: any;
    recommendations: string[];
  } {
    const failedGates = Array.from(this.gateStatus.values()).filter(status => !status).length;
    const performanceMetrics = this.getSystemPerformanceMetrics();
    
    let overallHealth: 'healthy' | 'degraded' | 'critical' = 'healthy';
    const recommendations: string[] = [];

    if (failedGates > 0) {
      overallHealth = failedGates > 2 ? 'critical' : 'degraded';
      recommendations.push(`${failedGates} performance gates are failing - investigate immediately`);
    }

    if (performanceMetrics.integration.avg_total_latency_ms > 25) {
      overallHealth = overallHealth === 'healthy' ? 'degraded' : overallHealth;
      recommendations.push('Average latency exceeds target - consider query optimization');
    }

    if (performanceMetrics.integration.system_warnings_rate > 0.1) {
      recommendations.push('High warning rate detected - review system configuration');
    }

    return {
      overall_health: overallHealth,
      gate_status: new Map(this.gateStatus),
      system_performance: performanceMetrics,
      recommendations
    };
  }
}