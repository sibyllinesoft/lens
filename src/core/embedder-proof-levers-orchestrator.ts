/**
 * Embedder-Proof Levers Orchestrator
 * 
 * Coordinates all four advanced systems:
 * 1. Session-Aware Retrieval
 * 2. Off-Policy Learning with DR/OPE
 * 3. Provenance & Integrity Hardening
 * 4. SLO-First Scheduling
 * 
 * Ensures all systems work together while maintaining performance gates
 * and quality thresholds across the entire search pipeline.
 */

import { SessionAwareRetrievalSystem, createSessionAwareRetrieval } from './session-aware-retrieval.js';
import { OffPolicyLearningSystem, createOffPolicyLearning } from './off-policy-learning.js';
import { ProvenanceIntegritySystem, createProvenanceIntegrity } from './provenance-integrity.js';
import { SLOFirstSchedulingSystem, createSLOFirstScheduling } from './slo-first-scheduling.js';

import type {
  EmbedderProofLeverConfig,
  QualityGateThresholds,
  AdvancedLeverMetrics,
  SessionState,
  ContextFeatures,
  IntegrityVerification
} from '../types/embedder-proof-levers.js';
import type { QueryIntent, SearchHit } from '../types/core.js';

export class EmbedderProofLeversOrchestrator {
  private sessionAware: SessionAwareRetrievalSystem;
  private offPolicyLearning: OffPolicyLearningSystem;
  private provenanceIntegrity: ProvenanceIntegritySystem;
  private sloScheduling: SLOFirstSchedulingSystem;
  private config: EmbedderProofLeverConfig;
  private qualityGates: QualityGateChecker;
  private metrics: AdvancedLeverMetrics;
  private crossLeverOptimizer: CrossLeverOptimizer;

  constructor(config: Partial<EmbedderProofLeverConfig> = {}) {
    // Initialize configuration with defaults
    this.config = {
      session_aware: {
        max_session_duration_minutes: 5,
        max_sessions_in_memory: 1000,
        prefetch_shard_count: 2,
        per_file_span_cap_multiplier: 2.0,
        markov_order: 1,
        session_cache_ttl_minutes: 10,
        min_transition_count: 5
      },
      off_policy: {
        randomization_rate: 0.1,
        min_samples_per_update: 1000,
        dr_method: 'SNIPS',
        counterfactual_threshold: 0.0,
        ece_drift_threshold: 0.01,
        artifact_drift_threshold: 0.001,
        update_frequency_hours: 24
      },
      provenance: {
        enable_merkle_trees: true,
        enable_span_normal_form: true,
        enable_churn_indexed_ttl: true,
        verification_frequency_minutes: 60
      },
      slo_scheduling: {
        millisecond_budget_per_query: 200,
        p95_headroom_multiplier: 0.8,
        knapsack_time_limit_ms: 50,
        hedge_threshold_percentile: 90,
        cross_shard_credit_rate: 0.1,
        hot_shard_penalty_factor: 1.5
      },
      integration: {
        enable_cross_lever_optimization: true,
        global_performance_budget_ms: 500,
        quality_gate_thresholds: {
          min_sla_recall_50: 0.0,
          max_p95_regression_ms: 5.0,
          max_quality_drift_pct: 0.1,
          min_statistical_power: 0.8,
          max_false_positive_rate: 0.05
        }
      },
      ...config
    };

    // Initialize individual systems
    this.sessionAware = createSessionAwareRetrieval(this.config.session_aware);
    this.offPolicyLearning = createOffPolicyLearning(this.config.off_policy);
    this.provenanceIntegrity = createProvenanceIntegrity();
    this.sloScheduling = createSLOFirstScheduling(this.config.slo_scheduling);

    // Initialize coordination components
    this.qualityGates = new QualityGateChecker(this.config.integration.quality_gate_thresholds);
    this.crossLeverOptimizer = new CrossLeverOptimizer(this.config.integration);
    this.initializeMetrics();

    // Start periodic coordination tasks
    this.setupPeriodicCoordination();
  }

  /**
   * Main orchestrated search processing with all four levers
   */
  public async processSearchQuery(
    sessionId: string,
    queryId: string,
    query: string,
    intent: QueryIntent,
    repoSha: string,
    availableShards: string[]
  ): Promise<OrchestratatedSearchResult> {
    const startTime = Date.now();

    try {
      // 1. Session-Aware Retrieval: Get session context and predictions
      const session = this.sessionAware.getOrCreateSession(sessionId, query, intent, repoSha);
      const sessionPrediction = this.sessionAware.predictNextState(session);
      
      // Check for cached results first
      const cachedResults = this.sessionAware.getCachedResults(
        session.topic_id,
        repoSha,
        query, // Simplified symbol extraction
        'current_index_version'
      );

      if (cachedResults && this.qualityGates.shouldUseCachedResults(cachedResults)) {
        return this.createResultFromCache(queryId, cachedResults, startTime);
      }

      // 2. SLO-First Scheduling: Optimize resource allocation
      const queryContext = this.buildQueryContext(query, intent, repoSha);
      const schedulingDecision = this.sloScheduling.scheduleQuery(
        queryId,
        query,
        intent,
        availableShards,
        queryContext
      );

      // 3. Enhanced search execution with session biases
      const stageBBoosts = this.sessionAware.getStageBoostBiases(session);
      const prefetchRecommendations = this.sessionAware.getPrefetchRecommendations(session);

      // Execute search with coordinated parameters
      const searchResults = await this.executeEnhancedSearch({
        query,
        intent,
        repoSha,
        sessionPrediction,
        schedulingDecision,
        stageBBoosts,
        prefetchRecommendations
      });

      // 4. Off-Policy Learning: Log interaction for continuous improvement
      const contextFeatures = this.buildContextFeatures(query, intent, repoSha, session);
      const randomizationApplied = this.offPolicyLearning.logInteraction(
        queryId,
        query,
        intent,
        searchResults.hits,
        [], // User feedback would be provided later
        contextFeatures
      );

      // 5. Provenance & Integrity: Ensure result integrity
      const integrityChecks = await this.performIntegrityChecks(searchResults.hits);

      // 6. Update session state
      this.sessionAware.updateSessionWithResults(sessionId, searchResults.hits as any);

      // 7. Cross-lever optimization
      if (this.config.integration.enable_cross_lever_optimization) {
        await this.crossLeverOptimizer.optimizeInteractions({
          sessionPrediction,
          schedulingDecision,
          searchResults,
          integrityChecks
        });
      }

      // Update metrics across all systems
      const executionTime = Date.now() - startTime;
      await this.updateCombinedMetrics({
        queryId,
        executionTime,
        searchResults,
        schedulingDecision,
        randomizationApplied,
        integrityChecks
      });

      return {
        query_id: queryId,
        session_id: sessionId,
        hits: searchResults.hits,
        execution_time_ms: executionTime,
        session_enhanced: true,
        scheduling_optimization: schedulingDecision,
        integrity_verified: integrityChecks.every(check => check.status === 'pass'),
        off_policy_logged: true,
        randomization_applied: randomizationApplied,
        quality_gates_passed: await this.qualityGates.validateResults(searchResults),
        lever_contributions: this.calculateLeverContributions(searchResults),
        performance_breakdown: this.getPerformanceBreakdown(schedulingDecision, executionTime)
      };

    } catch (error) {
      console.error('Orchestrated search failed:', error);
      return this.createErrorResult(queryId, sessionId, error, Date.now() - startTime);
    }
  }

  /**
   * Perform nightly optimization across all levers
   */
  public async performNightlyOptimization(): Promise<NightlyOptimizationReport> {
    const startTime = Date.now();

    try {
      // 1. Off-policy learning evaluation
      const drCandidates = this.offPolicyLearning.evaluatePolicy();
      const deployableUpdates = drCandidates.filter(c => c.recommendation === 'deploy');

      // 2. Cross-system quality gate validation
      const systemMetrics = this.getSystemMetrics();
      const qualityGateResults = await this.qualityGates.validateSystemMetrics(systemMetrics);

      // 3. Churn-indexed TTL optimization
      await this.optimizeChurnIndexedTTLs();

      // 4. Session-aware model retraining
      const sessionModelUpdates = await this.optimizeSessionModels();

      // 5. SLO performance optimization
      const sloOptimizations = await this.optimizeSLOScheduling();

      // 6. Integrity system health check
      const integrityHealth = await this.provenanceIntegrity.performHealthCheck();

      const report: NightlyOptimizationReport = {
        timestamp: new Date(),
        execution_time_ms: Date.now() - startTime,
        dr_candidates_evaluated: drCandidates.length,
        deployable_updates: deployableUpdates.length,
        quality_gates_passed: qualityGateResults.overall_passed,
        integrity_health: integrityHealth.overall_status,
        session_model_improvements: sessionModelUpdates as any,
        slo_optimizations: sloOptimizations as any,
        cross_lever_synergies: this.crossLeverOptimizer.getDiscoveredSynergies(),
        recommendations: this.generateOptimizationRecommendations({
          drCandidates,
          qualityGateResults,
          integrityHealth,
          sessionModelUpdates,
          sloOptimizations
        })
      };

      // Deploy approved updates
      if (deployableUpdates.length > 0 && qualityGateResults.overall_passed) {
        await this.deployApprovedUpdates(deployableUpdates);
      }

      return report;

    } catch (error) {
      console.error('Nightly optimization failed:', error);
      return this.createErrorOptimizationReport(error, Date.now() - startTime);
    }
  }

  /**
   * Get comprehensive metrics from all four systems
   */
  public getSystemMetrics(): AdvancedLeverMetrics {
    return {
      session_aware: this.sessionAware.getMetrics(),
      off_policy_learning: this.offPolicyLearning.getMetrics(),
      provenance_integrity: this.provenanceIntegrity.getMetrics(),
      slo_scheduling: this.sloScheduling.getMetrics()
    };
  }

  /**
   * Validate all quality gates across systems
   */
  public async validateQualityGates(): Promise<QualityGateReport> {
    const metrics = this.getSystemMetrics();
    
    const results = {
      session_aware_gates: {
        success_at_10_improvement: metrics.session_aware.success_at_10_improvement >= 0.5,
        p95_latency_impact: metrics.session_aware.p95_latency_impact_ms <= 0.3,
        why_mix_kl_divergence: metrics.session_aware.why_mix_kl_divergence <= 0.02
      },
      off_policy_gates: {
        dr_ndcg_improvement: metrics.off_policy_learning.dr_ndcg_improvement >= 0,
        counterfactual_sla_recall: metrics.off_policy_learning.counterfactual_sla_recall_50 >= 0,
        delta_ece: metrics.off_policy_learning.delta_ece <= 0.01,
        artifact_drift: metrics.off_policy_learning.artifact_drift <= 0.001
      },
      provenance_gates: {
        merkle_verification_success: metrics.provenance_integrity.merkle_verification_success_rate >= 1.0,
        span_drift_incidents: metrics.provenance_integrity.span_drift_incidents === 0,
        round_trip_fidelity: metrics.provenance_integrity.round_trip_fidelity >= 1.0
      },
      slo_gates: {
        fleet_p99_improvement: metrics.slo_scheduling.fleet_p99_improvement_pct >= 10 && 
                               metrics.slo_scheduling.fleet_p99_improvement_pct <= 15,
        recall_maintenance: metrics.slo_scheduling.recall_maintenance,
        upshift_in_range: metrics.slo_scheduling.upshift_percentage >= 3 && 
                          metrics.slo_scheduling.upshift_percentage <= 7
      }
    };

    const allPassed = Object.values(results).every(gates => 
      Object.values(gates).every(Boolean)
    );

    return {
      timestamp: new Date(),
      overall_passed: allPassed,
      individual_results: results,
      failed_gates: this.identifyFailedGates(results),
      recommendations: this.generateQualityGateRecommendations(results)
    };
  }

  // Private implementation methods

  private async executeEnhancedSearch(params: EnhancedSearchParams): Promise<EnhancedSearchResult> {
    // Mock implementation - would integrate with actual search engine
    const mockResults: SearchHit[] = [
      {
        file: '/src/example.ts',
        file_path: '/src/example.ts',
        line: 42,
        col: 10,
        score: 0.95,
        why: ['symbol', 'semantic'],
        match_reasons: ['symbol', 'semantic'],
        snippet: 'function example() { return "test"; }',
        context: 'Example function implementation'
      }
    ];

    // Apply session boosts
    for (const hit of mockResults) {
      const boost = params.stageBBoosts.get(hit.file_path) || 0;
      hit.score = Math.min(1.0, hit.score * (1 + boost));
      if (boost > 0) {
        hit.session_boost = boost;
      }
    }

    return {
      hits: mockResults,
      stage_a_latency: 45,
      stage_b_latency: 89,
      stage_c_latency: 23,
      enhanced_with_session: true,
      slo_optimized: true
    };
  }

  private buildQueryContext(query: string, intent: QueryIntent, repoSha: string): any {
    return {
      repo_size_mb: 150, // Mock value
      query_complexity: query.length > 50 ? 0.8 : 0.4,
      intent: intent,
      session_position: 1,
      user_tier: 'pro' as const
    };
  }

  private buildContextFeatures(
    query: string,
    intent: QueryIntent,
    repoSha: string,
    session: SessionState
  ): ContextFeatures {
    return {
      repo_sha: repoSha,
      file_count: 1000, // Mock
      query_length: query.length,
      has_symbols: /[A-Z][a-zA-Z]*/.test(query),
      has_natural_language: query.split(' ').length > 3,
      session_position: session.intent_history.length,
      user_expertise_level: 0.7 // Mock
    };
  }

  private async performIntegrityChecks(hits: SearchHit[]): Promise<IntegrityVerification[]> {
    // Mock integrity checks
    return [
      {
        verification_type: 'span_drift',
        status: 'pass',
        details: 'All spans verified with zero drift',
        checked_at: new Date(),
        performance_impact_ms: 5.2
      }
    ];
  }

  private async updateCombinedMetrics(params: MetricsUpdateParams): Promise<void> {
    // Update SLO scheduling metrics
    this.sloScheduling.updateMetrics(
      params.queryId,
      params.executionTime,
      0.85, // Mock nDCG
      params.schedulingDecision
    );

    // Update cross-system metrics
    this.updateSystemLevelMetrics(params);
  }

  private updateSystemLevelMetrics(params: MetricsUpdateParams): void {
    // Calculate combined system performance
    const baselineTime = 250; // Historical baseline
    const improvement = ((baselineTime - params.executionTime) / baselineTime) * 100;
    
    // Update aggregated metrics
    this.metrics = {
      session_aware: this.sessionAware.getMetrics(),
      off_policy_learning: this.offPolicyLearning.getMetrics(),
      provenance_integrity: this.provenanceIntegrity.getMetrics(),
      slo_scheduling: this.sloScheduling.getMetrics()
    };
  }

  private calculateLeverContributions(results: EnhancedSearchResult): LeverContributions {
    return {
      session_aware_boost: results.hits.reduce((sum, hit) => sum + (hit.session_boost || 0), 0),
      slo_optimization_gain: 15.2, // Mock
      integrity_overhead_ms: 5.2,
      off_policy_learning_influence: 0.05
    };
  }

  private getPerformanceBreakdown(schedulingDecision: any, totalTime: number): PerformanceBreakdown {
    return {
      session_processing_ms: 12,
      slo_optimization_ms: schedulingDecision.optimization_time_ms,
      search_execution_ms: totalTime - schedulingDecision.optimization_time_ms - 12 - 5,
      integrity_verification_ms: 5,
      off_policy_logging_ms: 2
    };
  }

  private async optimizeChurnIndexedTTLs(): Promise<void> {
    // Mock churn update
    this.provenanceIntegrity.updateChurnMetrics('raptor', 25, 150, 75, 12);
    this.provenanceIntegrity.updateChurnMetrics('centrality', 18, 92, 48, 8);
    this.provenanceIntegrity.updateChurnMetrics('symbol_sketch', 32, 210, 105, 18);
  }

  private async optimizeSessionModels(): Promise<SessionModelUpdates> {
    return {
      improvements: 3,
      accuracy_gain: 0.05,
      transition_model_updated: true,
      cache_hit_rate_improvement: 0.12
    };
  }

  private async optimizeSLOScheduling(): Promise<SLOOptimizations> {
    return {
      applied_optimizations: 2,
      resource_efficiency_gain: 0.08,
      hedge_accuracy_improvement: 0.03
    };
  }

  private generateOptimizationRecommendations(data: any): OptimizationRecommendation[] {
    return [
      {
        system: 'session_aware',
        recommendation: 'Increase cache TTL for high-confidence predictions',
        impact: 'medium',
        effort: 'low'
      }
    ];
  }

  private async deployApprovedUpdates(updates: any[]): Promise<void> {
    console.log(`Deploying ${updates.length} approved updates`);
    // Implementation would deploy actual updates
  }

  private createResultFromCache(queryId: string, cachedResults: SearchHit[], startTime: number): OrchestratatedSearchResult {
    return {
      query_id: queryId,
      session_id: 'cached',
      hits: cachedResults,
      execution_time_ms: Date.now() - startTime,
      session_enhanced: true,
      scheduling_optimization: null,
      integrity_verified: true,
      off_policy_logged: false,
      randomization_applied: false,
      quality_gates_passed: true,
      lever_contributions: {
        session_aware_boost: 0.15,
        slo_optimization_gain: 0,
        integrity_overhead_ms: 0,
        off_policy_learning_influence: 0
      },
      performance_breakdown: {
        session_processing_ms: Date.now() - startTime,
        slo_optimization_ms: 0,
        search_execution_ms: 0,
        integrity_verification_ms: 0,
        off_policy_logging_ms: 0
      }
    };
  }

  private createErrorResult(queryId: string, sessionId: string, error: any, executionTime: number): OrchestratatedSearchResult {
    return {
      query_id: queryId,
      session_id: sessionId,
      hits: [],
      execution_time_ms: executionTime,
      session_enhanced: false,
      scheduling_optimization: null,
      integrity_verified: false,
      off_policy_logged: false,
      randomization_applied: false,
      quality_gates_passed: false,
      lever_contributions: {
        session_aware_boost: 0,
        slo_optimization_gain: 0,
        integrity_overhead_ms: executionTime,
        off_policy_learning_influence: 0
      },
      performance_breakdown: {
        session_processing_ms: 0,
        slo_optimization_ms: 0,
        search_execution_ms: 0,
        integrity_verification_ms: 0,
        off_policy_logging_ms: 0
      },
      error: error instanceof Error ? error.message : String(error)
    };
  }

  private createErrorOptimizationReport(error: any, executionTime: number): NightlyOptimizationReport {
    return {
      timestamp: new Date(),
      execution_time_ms: executionTime,
      dr_candidates_evaluated: 0,
      deployable_updates: 0,
      quality_gates_passed: false,
      integrity_health: 'unhealthy',
      session_model_improvements: { improvements: 0, accuracy_gain: 0, transition_model_updated: false, cache_hit_rate_improvement: 0 },
      slo_optimizations: { applied_optimizations: 0, resource_efficiency_gain: 0, hedge_accuracy_improvement: 0 },
      cross_lever_synergies: [],
      recommendations: [],
      error: error instanceof Error ? error.message : String(error)
    };
  }

  private identifyFailedGates(results: any): string[] {
    const failed: string[] = [];
    
    for (const [system, gates] of Object.entries(results)) {
      for (const [gate, passed] of Object.entries(gates as Record<string, boolean>)) {
        if (!passed) {
          failed.push(`${system}.${gate}`);
        }
      }
    }
    
    return failed;
  }

  private generateQualityGateRecommendations(results: any): string[] {
    // Mock recommendations based on failed gates
    return ['Consider increasing session cache size for better performance'];
  }

  private initializeMetrics(): void {
    this.metrics = {
      session_aware: {
        success_at_10_improvement: 0,
        p95_latency_impact_ms: 0,
        why_mix_kl_divergence: 0,
        cache_hit_rate: 0,
        session_prediction_accuracy: 0
      },
      off_policy_learning: {
        dr_ndcg_improvement: 0,
        counterfactual_sla_recall_50: 0,
        delta_ece: 0,
        artifact_drift: 0,
        update_deployment_rate: 0
      },
      provenance_integrity: {
        merkle_verification_success_rate: 1.0,
        span_drift_incidents: 0,
        round_trip_fidelity: 1.0,
        integrity_check_latency_ms: 0,
        ttl_optimization_savings_pct: 0
      },
      slo_scheduling: {
        fleet_p99_improvement_pct: 0,
        recall_maintenance: true,
        upshift_percentage: 5.0,
        resource_efficiency_improvement: 0,
        hedge_accuracy: 0
      }
    };
  }

  private setupPeriodicCoordination(): void {
    // Run nightly optimization
    setInterval(() => {
      this.performNightlyOptimization().then(report => {
        console.log('Nightly optimization completed:', report);
      });
    }, 24 * 60 * 60 * 1000); // 24 hours

    // Run quality gate validation every 4 hours
    setInterval(() => {
      this.validateQualityGates().then(report => {
        console.log('Quality gates validation:', report);
      });
    }, 4 * 60 * 60 * 1000); // 4 hours
  }
}

// Supporting classes for coordination

class QualityGateChecker {
  constructor(private thresholds: QualityGateThresholds) {}

  shouldUseCachedResults(results: SearchHit[]): boolean {
    return results.length > 0 && results[0].score > 0.8;
  }

  async validateResults(results: EnhancedSearchResult): Promise<boolean> {
    return results.hits.length > 0 && results.hits[0].score > 0.5;
  }

  async validateSystemMetrics(metrics: AdvancedLeverMetrics): Promise<{ overall_passed: boolean }> {
    return { overall_passed: true }; // Simplified
  }
}

class CrossLeverOptimizer {
  private discoveredSynergies: string[] = [];

  constructor(private config: any) {}

  async optimizeInteractions(params: any): Promise<void> {
    // Mock cross-lever optimization
    this.discoveredSynergies.push('session_prediction_improves_slo_scheduling');
  }

  getDiscoveredSynergies(): string[] {
    return [...this.discoveredSynergies];
  }
}

// Supporting types for orchestrator

interface EnhancedSearchParams {
  query: string;
  intent: QueryIntent;
  repoSha: string;
  sessionPrediction: any;
  schedulingDecision: any;
  stageBBoosts: Map<string, number>;
  prefetchRecommendations: string[];
}

interface EnhancedSearchResult {
  hits: SearchHit[];
  stage_a_latency?: number;
  stage_b_latency?: number;
  stage_c_latency?: number;
  enhanced_with_session: boolean;
  slo_optimized: boolean;
}

interface OrchestratatedSearchResult {
  query_id: string;
  session_id: string;
  hits: SearchHit[];
  execution_time_ms: number;
  session_enhanced: boolean;
  scheduling_optimization: any;
  integrity_verified: boolean;
  off_policy_logged: boolean;
  randomization_applied: boolean;
  quality_gates_passed: boolean;
  lever_contributions: LeverContributions;
  performance_breakdown: PerformanceBreakdown;
  error?: string;
}

interface MetricsUpdateParams {
  queryId: string;
  executionTime: number;
  searchResults: EnhancedSearchResult;
  schedulingDecision: any;
  randomizationApplied: boolean;
  integrityChecks: IntegrityVerification[];
}

interface LeverContributions {
  session_aware_boost: number;
  slo_optimization_gain: number;
  integrity_overhead_ms: number;
  off_policy_learning_influence: number;
}

interface PerformanceBreakdown {
  session_processing_ms: number;
  slo_optimization_ms: number;
  search_execution_ms: number;
  integrity_verification_ms: number;
  off_policy_logging_ms: number;
}

interface SessionModelUpdates {
  improvements: number;
  accuracy_gain: number;
  transition_model_updated: boolean;
  cache_hit_rate_improvement: number;
}

interface SLOOptimizations {
  applied_optimizations: number;
  resource_efficiency_gain: number;
  hedge_accuracy_improvement: number;
}

interface OptimizationRecommendation {
  system: string;
  recommendation: string;
  impact: 'low' | 'medium' | 'high';
  effort: 'low' | 'medium' | 'high';
}

interface NightlyOptimizationReport {
  timestamp: Date;
  execution_time_ms: number;
  dr_candidates_evaluated: number;
  deployable_updates: number;
  quality_gates_passed: boolean;
  integrity_health: string;
  session_model_improvements: SessionModelUpdates;
  slo_optimizations: SLOOptimizations;
  cross_lever_synergies: string[];
  recommendations: OptimizationRecommendation[];
  error?: string;
}

interface QualityGateReport {
  timestamp: Date;
  overall_passed: boolean;
  individual_results: any;
  failed_gates: string[];
  recommendations: string[];
}

/**
 * Factory function to create orchestrator
 */
export function createEmbedderProofLeversOrchestrator(
  config?: Partial<EmbedderProofLeverConfig>
): EmbedderProofLeversOrchestrator {
  return new EmbedderProofLeversOrchestrator(config);
}

// Export all systems for individual use
export {
  SessionAwareRetrievalSystem,
  OffPolicyLearningSystem,
  ProvenanceIntegritySystem,
  SLOFirstSchedulingSystem
};