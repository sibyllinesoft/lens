/**
 * Evergreen Systems Integration - Coordinator for all four optimization systems
 * 
 * Integrates and coordinates:
 * 1. Program-Slice Recall (Stage-B++)
 * 2. Build/Test-Aware Priors  
 * 3. Speculative Multi-Plan Planner
 * 4. Cache Admission That Learns
 * 
 * Ensures systems compose correctly and don't interfere with each other.
 * Provides unified interface for enabling/disabling systems and monitoring quality gates.
 */

import type { 
  SearchContext, 
  Candidate, 
  SymbolDefinition,
  SymbolReference,
  TestFailure,
  ChangeEvent,
  CodeOwner
} from '../types/core.js';
import type { SearchHit } from './span_resolver/types.js';
import { LensTracer } from '../telemetry/tracer.js';

// Import all four evergreen systems
import { 
  SymbolGraph, 
  PathSensitiveSlicing, 
  type SliceResult 
} from './program-slice-recall.js';
import { 
  BuildTestPriors,
  BazelParser,
  GradleParser,
  CargoParser,
  type BuildTarget
} from './build-test-priors.js';
import { 
  SpeculativeMultiPlanPlanner,
  type QueryPlan 
} from './speculative-multi-plan.js';
import { 
  CacheAdmissionLearner,
  type TinyLFUConfig,
  type CacheStats
} from './cache-admission-learner.js';
import {
  EvergreenQualityMonitor,
  type QualityGateConfig,
  type QualityMonitoringReport,
  type SystemMetrics
} from './evergreen-quality-gates.js';

export interface EvergreenSystemsConfig {
  // Global enable/disable
  enabled: boolean;
  
  // System-specific configs
  program_slice_recall: {
    enabled: boolean;
    rollout_percentage: number;
    max_depth: number;
    max_nodes: number;
  };
  
  build_test_priors: {
    enabled: boolean;
    decay_half_life_hours: number;
    max_log_odds_delta: number;
  };
  
  speculative_multi_plan: {
    enabled: boolean;
    p95_headroom_threshold_ms: number;
    max_planner_budget_percent: number;
  };
  
  cache_admission_learner: {
    enabled: boolean;
    tiny_lfu_config: TinyLFUConfig;
  };
  
  // Quality monitoring
  quality_monitoring: {
    enabled: boolean;
    monitoring_interval_hours: number;
    quality_gate_config: QualityGateConfig;
  };
}

export interface SystemsStatus {
  overall_status: 'healthy' | 'degraded' | 'critical' | 'disabled';
  individual_status: {
    program_slice_recall: 'enabled' | 'disabled' | 'error';
    build_test_priors: 'enabled' | 'disabled' | 'error';
    speculative_multi_plan: 'enabled' | 'disabled' | 'error';
    cache_admission_learner: 'enabled' | 'disabled' | 'error';
  };
  performance_impact: {
    p95_latency_delta_ms: number;
    recall_improvement: number;
    cache_hit_improvement: number;
  };
  rollout_status: {
    slice_recall_percentage: number;
    build_priors_active_files: number;
    multi_plan_utilization: number;
    cache_admission_rate: number;
  };
}

export interface SearchPipelineResult {
  primary_hits: SearchHit[];
  slice_recall_hits: SearchHit[];
  cache_served: boolean;
  plan_used: string;
  priors_applied: number;
  total_latency_ms: number;
  stage_breakdown: {
    cache_check_ms: number;
    slice_recall_ms: number;
    plan_execution_ms: number;
    prior_application_ms: number;
  };
}

/**
 * Main integration coordinator for all evergreen systems
 */
export class EvergreenSystemsIntegrator {
  private config: EvergreenSystemsConfig;
  
  // System instances
  private symbolGraph: SymbolGraph;
  private sliceRecall: PathSensitiveSlicing;
  private buildPriors: BuildTestPriors;
  private multiPlan: SpeculativeMultiPlanPlanner;
  private cacheAdmission: CacheAdmissionLearner;
  private qualityMonitor: EvergreenQualityMonitor;
  
  // State tracking
  private currentP95Latency = 15.0; // ms, updated from metrics
  private systemsInitialized = false;
  
  constructor(config: EvergreenSystemsConfig) {
    this.config = config;
    
    // Initialize systems
    this.symbolGraph = new SymbolGraph();
    this.sliceRecall = new PathSensitiveSlicing(this.symbolGraph);
    this.buildPriors = new BuildTestPriors();
    this.multiPlan = new SpeculativeMultiPlanPlanner();
    this.cacheAdmission = new CacheAdmissionLearner(config.cache_admission_learner.tiny_lfu_config);
    this.qualityMonitor = new EvergreenQualityMonitor(
      config.quality_monitoring.quality_gate_config,
      config.quality_monitoring.monitoring_interval_hours
    );
  }

  /**
   * Initialize all systems
   */
  async initialize(): Promise<void> {
    const span = LensTracer.createChildSpan('evergreen_systems_init');
    
    try {
      if (!this.config.enabled) {
        span.setAttributes({ success: true, skipped: true, reason: 'systems_disabled' });
        return;
      }

      // Initialize systems in dependency order
      if (this.config.program_slice_recall.enabled) {
        this.sliceRecall.enableWithRollout(this.config.program_slice_recall.rollout_percentage);
      }
      
      if (this.config.build_test_priors.enabled) {
        this.buildPriors.enable();
      }
      
      if (this.config.speculative_multi_plan.enabled) {
        this.multiPlan.enableWithConstraints(
          this.config.speculative_multi_plan.p95_headroom_threshold_ms,
          this.config.speculative_multi_plan.max_planner_budget_percent
        );
      }
      
      if (this.config.cache_admission_learner.enabled) {
        this.cacheAdmission.enable();
      }
      
      // Start quality monitoring
      if (this.config.quality_monitoring.enabled) {
        this.qualityMonitor.startMonitoring();
      }
      
      this.systemsInitialized = true;
      
      span.setAttributes({
        success: true,
        'systems.slice_recall': this.config.program_slice_recall.enabled,
        'systems.build_priors': this.config.build_test_priors.enabled,
        'systems.multi_plan': this.config.speculative_multi_plan.enabled,
        'systems.cache_admission': this.config.cache_admission_learner.enabled,
        'quality_monitoring': this.config.quality_monitoring.enabled,
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
   * Execute integrated search pipeline with all evergreen systems
   */
  async executeIntegratedSearch(
    context: SearchContext,
    baseCandidates: Candidate[]
  ): Promise<SearchPipelineResult> {
    const span = LensTracer.createChildSpan('integrated_search_pipeline', {
      'context.query': context.query,
      'context.repo_sha': context.repo_sha,
      'base_candidates.count': baseCandidates.length,
    });

    const stageTimings = {
      cache_check_ms: 0,
      slice_recall_ms: 0,
      plan_execution_ms: 0,
      prior_application_ms: 0,
    };

    const pipelineStart = Date.now();

    try {
      if (!this.systemsInitialized || !this.config.enabled) {
        // Fallback to base results
        return {
          primary_hits: baseCandidates.map(c => this.candidateToHit(c)),
          slice_recall_hits: [],
          cache_served: false,
          plan_used: 'fallback',
          priors_applied: 0,
          total_latency_ms: Date.now() - pipelineStart,
          stage_breakdown: stageTimings,
        };
      }

      // Stage 1: Cache check
      const cacheStart = Date.now();
      let cacheHits: SearchHit[] | undefined;
      if (this.config.cache_admission_learner.enabled) {
        cacheHits = await this.cacheAdmission.get(context);
      }
      stageTimings.cache_check_ms = Date.now() - cacheStart;

      if (cacheHits) {
        // Cache hit - return cached results
        span.setAttributes({ 
          success: true, 
          cache_hit: true, 
          results_count: cacheHits.length 
        });
        
        return {
          primary_hits: cacheHits,
          slice_recall_hits: [],
          cache_served: true,
          plan_used: 'cached',
          priors_applied: 0,
          total_latency_ms: Date.now() - pipelineStart,
          stage_breakdown: stageTimings,
        };
      }

      // Stage 2: Apply build/test priors to base candidates
      const priorsStart = Date.now();
      let enhancedCandidates = [...baseCandidates];
      let priorsApplied = 0;
      
      if (this.config.build_test_priors.enabled) {
        enhancedCandidates = this.buildPriors.applyStageATieBreak(enhancedCandidates);
        enhancedCandidates = this.buildPriors.applyStageCFeature(enhancedCandidates);
        priorsApplied = enhancedCandidates.filter(c => 
          c.match_reasons.includes('build_prior' as any)).length;
      }
      stageTimings.prior_application_ms = Date.now() - priorsStart;

      // Stage 3: Program slice recall for plumbing/glue code
      const sliceStart = Date.now();
      let sliceHits: SearchHit[] = [];
      
      if (this.config.program_slice_recall.enabled) {
        sliceHits = await this.sliceRecall.performSliceRecall(context, enhancedCandidates);
      }
      stageTimings.slice_recall_ms = Date.now() - sliceStart;

      // Stage 4: Speculative multi-plan execution
      const planStart = Date.now();
      let planResults: SearchHit[] = [];
      let planUsed = 'single-plan';
      
      if (this.config.speculative_multi_plan.enabled) {
        const maxBudget = 50; // ms - configurable budget
        const speculativeResult = await this.multiPlan.executeSpeculativeSearch(
          context,
          this.currentP95Latency,
          maxBudget
        );
        planResults = speculativeResult.primary_results;
        planUsed = speculativeResult.execution_stats.primary_plan;
      } else {
        // Convert enhanced candidates to hits
        planResults = enhancedCandidates.map(c => this.candidateToHit(c));
      }
      stageTimings.plan_execution_ms = Date.now() - planStart;

      // Combine all results
      const allHits = [...planResults, ...sliceHits];
      
      // Deduplicate and rank final results
      const finalHits = this.deduplicateAndRank(allHits);

      // Stage 5: Cache admission decision
      if (this.config.cache_admission_learner.enabled && finalHits.length > 0) {
        await this.cacheAdmission.set(context, finalHits);
      }

      const totalLatency = Date.now() - pipelineStart;

      span.setAttributes({
        success: true,
        cache_hit: false,
        'results.primary': planResults.length,
        'results.slice_recall': sliceHits.length,
        'results.final': finalHits.length,
        'plan_used': planUsed,
        'priors_applied': priorsApplied,
        'total_latency_ms': totalLatency,
      });

      return {
        primary_hits: finalHits,
        slice_recall_hits: sliceHits,
        cache_served: false,
        plan_used: planUsed,
        priors_applied: priorsApplied,
        total_latency_ms: totalLatency,
        stage_breakdown: stageTimings,
      };
      
    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      
      // Fallback to base results on error
      return {
        primary_hits: baseCandidates.map(c => this.candidateToHit(c)),
        slice_recall_hits: [],
        cache_served: false,
        plan_used: 'error-fallback',
        priors_applied: 0,
        total_latency_ms: Date.now() - pipelineStart,
        stage_breakdown: stageTimings,
      };
    } finally {
      span.end();
    }
  }

  /**
   * Index symbols for program slice recall
   */
  async indexSymbols(symbols: SymbolDefinition[], references: SymbolReference[]): Promise<void> {
    if (!this.config.program_slice_recall.enabled) return;

    for (const symbol of symbols) {
      this.symbolGraph.addSymbolDefinition(symbol);
    }
    
    for (const reference of references) {
      this.symbolGraph.addSymbolReference(reference);
    }
  }

  /**
   * Ingest build graph for build/test priors
   */
  async ingestBuildGraph(files: { path: string; content: string }[]): Promise<void> {
    if (!this.config.build_test_priors.enabled) return;
    
    await this.buildPriors.ingestBuildGraph(files);
  }

  /**
   * Record test failure for build/test priors
   */
  recordTestFailure(failure: TestFailure): void {
    if (!this.config.build_test_priors.enabled) return;
    
    this.buildPriors.recordTestFailure(failure);
  }

  /**
   * Record change event for build/test priors
   */
  recordChangeEvent(change: ChangeEvent): void {
    if (!this.config.build_test_priors.enabled) return;
    
    this.buildPriors.recordChangeEvent(change);
  }

  /**
   * Update code owners for build/test priors
   */
  updateCodeOwners(owners: CodeOwner[]): void {
    if (!this.config.build_test_priors.enabled) return;
    
    this.buildPriors.updateCodeOwners(owners);
  }

  /**
   * Get comprehensive systems status
   */
  getSystemsStatus(): SystemsStatus {
    const sliceStats = this.sliceRecall ? { enabled: true } : { enabled: false };
    const buildStats = this.buildPriors.getStats();
    const planStats = this.multiPlan.getStats();
    const cacheStats = this.cacheAdmission.getStats();
    
    // Calculate performance impact
    const p95Delta = this.estimateP95Impact();
    const recallImprovement = this.estimateRecallImprovement();
    const cacheImprovement = cacheStats.hit_rate - (cacheStats.lru_baseline_hit_rate || 0);
    
    return {
      overall_status: this.config.enabled ? 'healthy' : 'disabled',
      individual_status: {
        program_slice_recall: this.config.program_slice_recall.enabled ? 'enabled' : 'disabled',
        build_test_priors: this.config.build_test_priors.enabled ? 'enabled' : 'disabled',
        speculative_multi_plan: this.config.speculative_multi_plan.enabled ? 'enabled' : 'disabled',
        cache_admission_learner: this.config.cache_admission_learner.enabled ? 'enabled' : 'disabled',
      },
      performance_impact: {
        p95_latency_delta_ms: p95Delta,
        recall_improvement: recallImprovement,
        cache_hit_improvement: cacheImprovement,
      },
      rollout_status: {
        slice_recall_percentage: this.config.program_slice_recall.rollout_percentage,
        build_priors_active_files: buildStats.priors,
        multi_plan_utilization: planStats.max_budget_percent,
        cache_admission_rate: cacheStats.admissions / Math.max(1, cacheStats.total_requests),
      },
    };
  }

  /**
   * Get latest quality monitoring report
   */
  getQualityReport(): QualityMonitoringReport | undefined {
    return this.qualityMonitor.getLatestReport();
  }

  /**
   * Update current P95 latency for multi-plan decisions
   */
  updateCurrentP95Latency(latencyMs: number): void {
    this.currentP95Latency = latencyMs;
  }

  /**
   * Graceful shutdown of all systems
   */
  async shutdown(): Promise<void> {
    this.qualityMonitor.stopMonitoring();
    this.cacheAdmission.shutdown();
    this.systemsInitialized = false;
  }

  // Private helper methods

  private candidateToHit(candidate: Candidate): SearchHit {
    return {
      file: candidate.file_path,
      line: candidate.line,
      col: candidate.col,
      score: candidate.score,
      why: candidate.match_reasons as any[],
      snippet: candidate.context,
      symbol_kind: candidate.symbol_kind,
      ast_path: candidate.ast_path,
    };
  }

  private deduplicateAndRank(hits: SearchHit[]): SearchHit[] {
    // Simple deduplication by file+line+col
    const seen = new Set<string>();
    const deduplicated: SearchHit[] = [];
    
    for (const hit of hits) {
      const key = `${hit.file}:${hit.line}:${hit.col}`;
      if (!seen.has(key)) {
        seen.add(key);
        deduplicated.push(hit);
      }
    }
    
    // Sort by score descending
    deduplicated.sort((a, b) => b.score - a.score);
    
    return deduplicated.slice(0, 50); // Limit to top 50
  }

  private estimateP95Impact(): number {
    // Rough estimation based on enabled systems
    let impact = 0;
    
    if (this.config.program_slice_recall.enabled) {
      impact += 0.7; // Expected +0.7ms from slice recall
    }
    if (this.config.speculative_multi_plan.enabled) {
      impact += 0.3; // May add some overhead
    }
    if (this.config.cache_admission_learner.enabled) {
      impact -= 0.5; // Should reduce latency via cache hits
    }
    
    return impact;
  }

  private estimateRecallImprovement(): number {
    let improvement = 0;
    
    if (this.config.program_slice_recall.enabled) {
      improvement += 0.008; // +0.8pp expected
    }
    if (this.config.build_test_priors.enabled) {
      improvement += 0.003; // +0.3pp expected from better ranking
    }
    
    return improvement;
  }

  /**
   * Get default configuration for all evergreen systems
   */
  static getDefaultConfig(): EvergreenSystemsConfig {
    return {
      enabled: true,
      
      program_slice_recall: {
        enabled: true,
        rollout_percentage: 25, // Start at 25%
        max_depth: 2,
        max_nodes: 64,
      },
      
      build_test_priors: {
        enabled: true,
        decay_half_life_hours: 36, // 36 hours
        max_log_odds_delta: 0.3,
      },
      
      speculative_multi_plan: {
        enabled: true,
        p95_headroom_threshold_ms: 5.0,
        max_planner_budget_percent: 10,
      },
      
      cache_admission_learner: {
        enabled: true,
        tiny_lfu_config: {
          window_size: 100,
          protected_size: 300,
          probation_size: 100,
          sketch_size: 2048,
          admission_threshold: 0.1,
          aging_period_ms: 60000, // 1 minute
        },
      },
      
      quality_monitoring: {
        enabled: true,
        monitoring_interval_hours: 1,
        quality_gate_config: EvergreenQualityMonitor.getDefaultConfig(),
      },
    };
  }
}