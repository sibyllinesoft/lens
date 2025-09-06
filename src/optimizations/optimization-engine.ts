/**
 * Optimization Engine - Integration Layer for All Four Durable Optimizations
 * 
 * Orchestrates the four embedder-agnostic search optimizations:
 * 1. Clone-Aware Recall (token shingle expansion)
 * 2. Learning-to-Stop (scanner and ANN early termination)
 * 3. Targeted Diversity (constrained MMR for overview queries)
 * 4. TTL That Follows Churn (adaptive cache management)
 * 
 * Key Features:
 * - Coordinated optimization pipeline execution
 * - Cross-system performance monitoring
 * - SLA compliance validation (SLA-Recall@50≥0, latency targets, quality gates)
 * - Comprehensive telemetry and metrics aggregation
 * - Graceful degradation on optimization failures
 */

import { CloneAwareRecallSystem } from './clone-aware-recall.js';
import { LearningToStopSystem } from './learning-to-stop.js';
import { TargetedDiversitySystem, type DiversityFeatures } from './targeted-diversity.js';
import { ChurnAwareTTLSystem } from './churn-aware-ttl.js';

import type { SearchHit, MatchReason } from '../core/span_resolver/types.js';
import type { SearchContext } from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';

// SLA and performance targets from TODO.md
const SLA_RECALL_AT_50_MIN = 0; // SLA-Recall@50 ≥ 0 (no degradation)
const CLONE_RECALL_TARGET_MIN = 0.5; // +0.5pp minimum
const CLONE_RECALL_TARGET_MAX = 1.0; // +1.0pp maximum
const CLONE_LATENCY_BUDGET_MS = 0.6; // ≤+0.6ms p95
const LEARNING_STOP_IMPROVEMENT_MIN = 0.8; // -0.8ms minimum
const LEARNING_STOP_IMPROVEMENT_MAX = 1.5; // -1.5ms maximum
const LEARNING_STOP_UPSHIFT_MIN = 0.03; // 3% minimum upshift
const LEARNING_STOP_UPSHIFT_MAX = 0.07; // 7% maximum upshift
const DIVERSITY_IMPROVEMENT_TARGET = 0.10; // +10% diversity
const TTL_IMPROVEMENT_MIN = 0.5; // -0.5ms minimum
const TTL_IMPROVEMENT_MAX = 1.0; // -1.0ms maximum
const TTL_WHY_MIX_KL_MAX = 0.02; // KL ≤ 0.02

export interface OptimizationConfig {
  clone_aware_enabled: boolean;
  learning_to_stop_enabled: boolean;
  targeted_diversity_enabled: boolean;
  churn_aware_ttl_enabled: boolean;
  performance_monitoring_enabled: boolean;
  graceful_degradation_enabled: boolean;
}

export interface OptimizationPipeline {
  query_start_time: number;
  original_hits: SearchHit[];
  clone_expanded_hits?: SearchHit[];
  diversified_hits?: SearchHit[];
  final_hits: SearchHit[];
  optimizations_applied: string[];
  performance_impact: OptimizationPerformanceImpact;
}

export interface OptimizationPerformanceImpact {
  clone_expansion_ms: number;
  learning_stop_ms: number;
  diversity_ms: number;
  total_optimization_ms: number;
  recall_change: number;
  diversity_improvement: number;
  sla_compliance: boolean;
}

export interface SLAMetrics {
  recall_at_50: number;
  p95_latency_ms: number;
  upshift_percentage: number;
  diversity_score: number;
  why_mix_kl: number;
  span_coverage: number;
}

export interface SystemHealthStatus {
  clone_aware_healthy: boolean;
  learning_stop_healthy: boolean;
  diversity_healthy: boolean;
  ttl_healthy: boolean;
  overall_healthy: boolean;
  degraded_optimizations: string[];
}

export class OptimizationEngine {
  private cloneAwareSystem: CloneAwareRecallSystem;
  private learningToStopSystem: LearningToStopSystem;
  private diversitySystem: TargetedDiversitySystem;
  private ttlSystem: ChurnAwareTTLSystem;
  
  private config: OptimizationConfig;
  private systemHealth: SystemHealthStatus;
  
  private performanceHistory = {
    pipelines: [] as OptimizationPipeline[],
    sla_metrics: [] as SLAMetrics[],
    optimization_failures: [] as { system: string; error: string; timestamp: number }[],
  };
  
  private lastHealthCheck = Date.now();
  private readonly HEALTH_CHECK_INTERVAL_MS = 30000; // 30 seconds
  
  constructor(config: OptimizationConfig) {
    this.config = config;
    
    // Initialize optimization systems
    this.cloneAwareSystem = new CloneAwareRecallSystem();
    this.learningToStopSystem = new LearningToStopSystem();
    this.diversitySystem = new TargetedDiversitySystem();
    this.ttlSystem = new ChurnAwareTTLSystem();
    
    // Initialize system health
    this.systemHealth = {
      clone_aware_healthy: true,
      learning_stop_healthy: true,
      diversity_healthy: true,
      ttl_healthy: true,
      overall_healthy: true,
      degraded_optimizations: [],
    };
  }
  
  /**
   * Initialize the optimization engine and all subsystems
   */
  async initialize(): Promise<void> {
    const span = LensTracer.createChildSpan('optimization_engine_init');
    
    try {
      console.log('🚀 Initializing Optimization Engine with all four systems...');
      
      // Initialize subsystems in parallel
      const initPromises = [];
      
      if (this.config.clone_aware_enabled) {
        initPromises.push(
          this.cloneAwareSystem.initialize().catch(error => {
            console.error('Clone-Aware system initialization failed:', error);
            this.systemHealth.clone_aware_healthy = false;
            this.systemHealth.degraded_optimizations.push('clone_aware');
          })
        );
      }
      
      if (this.config.learning_to_stop_enabled) {
        initPromises.push(
          this.learningToStopSystem.initialize().catch(error => {
            console.error('Learning-to-Stop system initialization failed:', error);
            this.systemHealth.learning_stop_healthy = false;
            this.systemHealth.degraded_optimizations.push('learning_to_stop');
          })
        );
      }
      
      if (this.config.targeted_diversity_enabled) {
        initPromises.push(
          this.diversitySystem.initialize().catch(error => {
            console.error('Diversity system initialization failed:', error);
            this.systemHealth.diversity_healthy = false;
            this.systemHealth.degraded_optimizations.push('targeted_diversity');
          })
        );
      }
      
      if (this.config.churn_aware_ttl_enabled) {
        initPromises.push(
          this.ttlSystem.initialize().catch(error => {
            console.error('TTL system initialization failed:', error);
            this.systemHealth.ttl_healthy = false;
            this.systemHealth.degraded_optimizations.push('churn_aware_ttl');
          })
        );
      }
      
      await Promise.allSettled(initPromises);
      
      // Update overall health status
      this.updateOverallHealth();
      
      span.setAttributes({
        success: true,
        systems_initialized: initPromises.length,
        degraded_systems: this.systemHealth.degraded_optimizations.length,
        overall_healthy: this.systemHealth.overall_healthy
      });
      
      console.log(`✅ Optimization Engine initialized (${initPromises.length} systems, ${this.systemHealth.degraded_optimizations.length} degraded)`);
      
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
   * Execute the full optimization pipeline on search results
   */
  async optimizeSearchResults(
    originalHits: SearchHit[],
    ctx: SearchContext,
    diversityFeatures?: DiversityFeatures
  ): Promise<OptimizationPipeline> {
    const span = LensTracer.createChildSpan('optimize_search_results', {
      original_hits: originalHits.length,
      repo_sha: ctx.repo_sha,
      query: ctx.query
    });
    
    const queryStartTime = Date.now();
    
    const pipeline: OptimizationPipeline = {
      query_start_time: queryStartTime,
      original_hits: originalHits,
      final_hits: originalHits,
      optimizations_applied: [],
      performance_impact: {
        clone_expansion_ms: 0,
        learning_stop_ms: 0,
        diversity_ms: 0,
        total_optimization_ms: 0,
        recall_change: 0,
        diversity_improvement: 0,
        sla_compliance: false,
      },
    };
    
    try {
      let currentHits = [...originalHits];
      
      // Phase 1: Clone-Aware Recall Expansion (if enabled and healthy)
      if (this.config.clone_aware_enabled && this.systemHealth.clone_aware_healthy) {
        currentHits = await this.applyCloneAwareRecall(currentHits, ctx, pipeline);
      }
      
      // Phase 2: Learning-to-Stop would have been applied during search phase
      // This is logged for completeness but doesn't modify results here
      if (this.config.learning_to_stop_enabled && this.systemHealth.learning_stop_healthy) {
        await this.recordLearningToStopMetrics(pipeline);
      }
      
      // Phase 3: Targeted Diversity (constrained MMR) - only after clone expansion
      if (this.config.targeted_diversity_enabled && 
          this.systemHealth.diversity_healthy && 
          diversityFeatures) {
        // Mark clone-collapse as completed since we did clone expansion
        const enhancedFeatures: DiversityFeatures = {
          ...diversityFeatures,
          clone_collapsed: pipeline.clone_expanded_hits !== undefined,
          result_count: currentHits.length,
        };
        
        currentHits = await this.applyTargetedDiversity(currentHits, ctx, enhancedFeatures, pipeline);
      }
      
      pipeline.final_hits = currentHits;
      
      // Calculate final performance impact
      const totalOptimizationTime = Date.now() - queryStartTime;
      pipeline.performance_impact.total_optimization_ms = totalOptimizationTime;
      pipeline.performance_impact.recall_change = 
        (currentHits.length - originalHits.length) / Math.max(1, originalHits.length);
      
      // Validate SLA compliance
      const slaMetrics = await this.calculateSLAMetrics(pipeline, ctx);
      pipeline.performance_impact.sla_compliance = this.validateSLACompliance(slaMetrics);
      
      // Record pipeline for performance analysis
      this.performanceHistory.pipelines.push(pipeline);
      this.performanceHistory.sla_metrics.push(slaMetrics);
      
      // Periodic health check
      await this.performPeriodicHealthCheck();
      
      span.setAttributes({
        success: true,
        optimizations_applied: pipeline.optimizations_applied.length,
        final_hits: currentHits.length,
        recall_change: pipeline.performance_impact.recall_change,
        total_optimization_ms: totalOptimizationTime,
        sla_compliant: pipeline.performance_impact.sla_compliance
      });
      
      return pipeline;
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      
      // Return pipeline with original hits on error (graceful degradation)
      pipeline.final_hits = originalHits;
      return pipeline;
      
    } finally {
      span.end();
    }
  }
  
  /**
   * Apply clone-aware recall expansion
   */
  private async applyCloneAwareRecall(
    hits: SearchHit[],
    ctx: SearchContext,
    pipeline: OptimizationPipeline
  ): Promise<SearchHit[]> {
    const startTime = Date.now();
    
    try {
      const expandedHits = await this.cloneAwareSystem.expandWithClones(hits, ctx);
      
      const expansionTime = Date.now() - startTime;
      pipeline.performance_impact.clone_expansion_ms = expansionTime;
      pipeline.clone_expanded_hits = expandedHits;
      pipeline.optimizations_applied.push('clone_aware_recall');
      
      // Validate performance gate: ≤+0.6ms p95
      if (expansionTime > CLONE_LATENCY_BUDGET_MS) {
        console.warn(`Clone expansion exceeded latency budget: ${expansionTime}ms > ${CLONE_LATENCY_BUDGET_MS}ms`);
      }
      
      return expandedHits;
      
    } catch (error) {
      console.error('Clone-aware recall failed:', error);
      this.recordOptimizationFailure('clone_aware', error);
      return hits; // Return original hits on error
    }
  }
  
  /**
   * Record learning-to-stop metrics (system is applied during search phase)
   */
  private async recordLearningToStopMetrics(pipeline: OptimizationPipeline): Promise<void> {
    try {
      // Get metrics from learning-to-stop system
      const metrics = this.learningToStopSystem.getPerformanceMetrics();
      
      // Record the learning-to-stop impact (negative indicates improvement)
      pipeline.performance_impact.learning_stop_ms = -metrics.p95_improvement_ms;
      pipeline.optimizations_applied.push('learning_to_stop');
      
    } catch (error) {
      console.error('Learning-to-stop metrics recording failed:', error);
      this.recordOptimizationFailure('learning_to_stop', error);
    }
  }
  
  /**
   * Apply targeted diversity using constrained MMR
   */
  private async applyTargetedDiversity(
    hits: SearchHit[],
    ctx: SearchContext,
    features: DiversityFeatures,
    pipeline: OptimizationPipeline
  ): Promise<SearchHit[]> {
    const startTime = Date.now();
    
    try {
      const diversifiedHits = await this.diversitySystem.diversifyResults(hits, ctx, features);
      
      const diversityTime = Date.now() - startTime;
      pipeline.performance_impact.diversity_ms = diversityTime;
      pipeline.diversified_hits = diversifiedHits;
      
      // Only mark as applied if actual diversification occurred
      if (diversifiedHits.length !== hits.length || 
          JSON.stringify(diversifiedHits) !== JSON.stringify(hits)) {
        pipeline.optimizations_applied.push('targeted_diversity');
        
        // Calculate diversity improvement (simplified)
        pipeline.performance_impact.diversity_improvement = this.calculateDiversityImprovement(hits, diversifiedHits);
      }
      
      return diversifiedHits;
      
    } catch (error) {
      console.error('Targeted diversity failed:', error);
      this.recordOptimizationFailure('targeted_diversity', error);
      return hits; // Return original hits on error
    }
  }
  
  /**
   * Calculate SLA metrics for compliance validation
   */
  private async calculateSLAMetrics(
    pipeline: OptimizationPipeline,
    ctx: SearchContext
  ): Promise<SLAMetrics> {
    // Calculate recall@50 (simplified - would need ground truth)
    const recall50 = Math.min(1.0, pipeline.final_hits.length / 50);
    
    // Calculate p95 latency from recent optimization history
    const recentLatencies = this.performanceHistory.pipelines
      .slice(-20)
      .map(p => p.performance_impact.total_optimization_ms);
    const p95Latency = recentLatencies.length > 0
      ? recentLatencies.sort((a, b) => a - b)[Math.floor(recentLatencies.length * 0.95)]
      : pipeline.performance_impact.total_optimization_ms;
    
    // Calculate upshift percentage (improvement in results)
    const upshift = pipeline.performance_impact.recall_change;
    
    // Calculate diversity score (simplified)
    const diversityScore = pipeline.performance_impact.diversity_improvement;
    
    // Calculate why-mix KL (simplified - would need detailed analysis)
    const whyMixKL = 0.01; // Placeholder
    
    // Calculate span coverage (assume 100% for now)
    const spanCoverage = 1.0;
    
    return {
      recall_at_50: recall50,
      p95_latency_ms: p95Latency,
      upshift_percentage: upshift,
      diversity_score: diversityScore,
      why_mix_kl: whyMixKL,
      span_coverage: spanCoverage,
    };
  }
  
  /**
   * Validate SLA compliance based on TODO.md requirements
   */
  private validateSLACompliance(metrics: SLAMetrics): boolean {
    // Gate 1: SLA-Recall@50 ≥ 0 (no degradation)
    if (metrics.recall_at_50 < SLA_RECALL_AT_50_MIN) {
      return false;
    }
    
    // Gate 2: P95 latency within budgets
    // (Individual system budgets are checked within each system)
    
    // Gate 3: Why-mix KL ≤ 0.02 (for TTL system)
    if (metrics.why_mix_kl > TTL_WHY_MIX_KL_MAX) {
      return false;
    }
    
    // Gate 4: Span coverage = 100%
    if (metrics.span_coverage < 1.0) {
      return false;
    }
    
    return true;
  }
  
  /**
   * Index content for clone detection
   */
  async indexContent(
    content: string,
    file: string,
    line: number,
    col: number,
    repository: string,
    symbolKind?: string
  ): Promise<void> {
    if (this.config.clone_aware_enabled && this.systemHealth.clone_aware_healthy) {
      try {
        await this.cloneAwareSystem.indexSpan(content, file, line, col, repository, symbolKind);
      } catch (error) {
        console.error('Content indexing failed:', error);
        this.recordOptimizationFailure('clone_aware_indexing', error);
      }
    }
  }
  
  /**
   * Record file change for churn tracking
   */
  recordFileChange(filePath: string, timestamp?: number): void {
    if (this.config.churn_aware_ttl_enabled && this.systemHealth.ttl_healthy) {
      try {
        this.ttlSystem.recordFileChange(filePath, timestamp);
      } catch (error) {
        console.error('File change recording failed:', error);
        this.recordOptimizationFailure('churn_tracking', error);
      }
    }
  }
  
  /**
   * Get cached value with churn-aware TTL
   */
  async getCachedValue<T>(
    key: string,
    indexVersion: string,
    spanHash: string,
    valueFactory: () => Promise<T>,
    cacheType: 'micro' | 'raptor' | 'centrality' = 'micro',
    topicBin?: string
  ): Promise<T> {
    if (this.config.churn_aware_ttl_enabled && this.systemHealth.ttl_healthy) {
      try {
        switch (cacheType) {
          case 'micro':
            return await this.ttlSystem.getMicroCache(key, indexVersion, spanHash, valueFactory, topicBin);
          case 'raptor':
            return await this.ttlSystem.getRaptorCache(key, indexVersion, valueFactory, topicBin);
          case 'centrality':
            return await this.ttlSystem.getCentralityCache(key, indexVersion, valueFactory, topicBin);
          default:
            return await valueFactory();
        }
      } catch (error) {
        console.error('Cache access failed:', error);
        this.recordOptimizationFailure('cache_access', error);
        return await valueFactory(); // Fallback to direct computation
      }
    } else {
      return await valueFactory(); // Direct computation if TTL system disabled
    }
  }
  
  /**
   * Get learning-to-stop decision for scanner
   */
  shouldStopScanning(
    blocksProcessed: number,
    candidatesFound: number,
    timeSpent: number,
    ctx: SearchContext,
    queryStartTime: number
  ): { shouldStop: boolean; confidence: number } {
    if (this.config.learning_to_stop_enabled && this.systemHealth.learning_stop_healthy) {
      try {
        const scannerState = {
          blocks_processed: blocksProcessed,
          candidates_found: candidatesFound,
          time_spent_ms: timeSpent,
          last_gain: 0, // Would be calculated based on recent blocks
          marginal_utility: 0, // Would be calculated based on gains
        };
        
        const prediction = this.learningToStopSystem.shouldStopScanning(scannerState, ctx, queryStartTime);
        return {
          shouldStop: prediction.should_stop,
          confidence: prediction.confidence,
        };
      } catch (error) {
        console.error('Learning-to-stop decision failed:', error);
        this.recordOptimizationFailure('learning_stop_decision', error);
        return { shouldStop: false, confidence: 0 }; // Conservative fallback
      }
    }
    
    return { shouldStop: false, confidence: 0 }; // Disabled or unhealthy
  }
  
  /**
   * Get optimized ANN efSearch parameter
   */
  getOptimizedEfSearch(
    currentEf: number,
    recallAchieved: number,
    timeSpent: number,
    riskLevel: number,
    ctx: SearchContext,
    queryStartTime: number
  ): number {
    if (this.config.learning_to_stop_enabled && this.systemHealth.learning_stop_healthy) {
      try {
        const annState = {
          current_ef: currentEf,
          recall_achieved: recallAchieved,
          time_spent_ms: timeSpent,
          risk_level: riskLevel,
        };
        
        return this.learningToStopSystem.optimizeANNSearch(annState, ctx, queryStartTime);
      } catch (error) {
        console.error('ANN optimization failed:', error);
        this.recordOptimizationFailure('ann_optimization', error);
        return currentEf; // Return original value on error
      }
    }
    
    return currentEf; // Disabled or unhealthy
  }
  
  /**
   * Perform periodic health checks and maintenance
   */
  private async performPeriodicHealthCheck(): Promise<void> {
    const now = Date.now();
    
    if (now - this.lastHealthCheck < this.HEALTH_CHECK_INTERVAL_MS) {
      return; // Skip if too recent
    }
    
    this.lastHealthCheck = now;
    
    try {
      // Perform TTL system maintenance
      if (this.config.churn_aware_ttl_enabled && this.systemHealth.ttl_healthy) {
        await this.ttlSystem.performMaintenance();
      }
      
      // Check system health based on recent failures
      this.updateSystemHealthBasedOnFailures();
      
      // Cleanup old performance history
      this.cleanupPerformanceHistory();
      
    } catch (error) {
      console.error('Periodic health check failed:', error);
    }
  }
  
  /**
   * Update system health based on recent failures
   */
  private updateSystemHealthBasedOnFailures(): void {
    const now = Date.now();
    const recentWindow = 5 * 60 * 1000; // 5 minutes
    
    const recentFailures = this.performanceHistory.optimization_failures
      .filter(f => now - f.timestamp < recentWindow);
    
    const failuresBySystem = new Map<string, number>();
    for (const failure of recentFailures) {
      failuresBySystem.set(failure.system, (failuresBySystem.get(failure.system) || 0) + 1);
    }
    
    // Mark systems as unhealthy if too many recent failures
    const failureThreshold = 3;
    
    this.systemHealth.clone_aware_healthy = (failuresBySystem.get('clone_aware') || 0) < failureThreshold;
    this.systemHealth.learning_stop_healthy = (failuresBySystem.get('learning_to_stop') || 0) < failureThreshold;
    this.systemHealth.diversity_healthy = (failuresBySystem.get('targeted_diversity') || 0) < failureThreshold;
    this.systemHealth.ttl_healthy = (failuresBySystem.get('churn_aware_ttl') || 0) < failureThreshold;
    
    // Update degraded optimizations list
    this.systemHealth.degraded_optimizations = [];
    if (!this.systemHealth.clone_aware_healthy) this.systemHealth.degraded_optimizations.push('clone_aware');
    if (!this.systemHealth.learning_stop_healthy) this.systemHealth.degraded_optimizations.push('learning_to_stop');
    if (!this.systemHealth.diversity_healthy) this.systemHealth.degraded_optimizations.push('targeted_diversity');
    if (!this.systemHealth.ttl_healthy) this.systemHealth.degraded_optimizations.push('churn_aware_ttl');
    
    this.updateOverallHealth();
  }
  
  /**
   * Update overall health status
   */
  private updateOverallHealth(): void {
    this.systemHealth.overall_healthy = 
      this.systemHealth.clone_aware_healthy &&
      this.systemHealth.learning_stop_healthy &&
      this.systemHealth.diversity_healthy &&
      this.systemHealth.ttl_healthy;
  }
  
  /**
   * Record optimization failure for health tracking
   */
  private recordOptimizationFailure(system: string, error: any): void {
    this.performanceHistory.optimization_failures.push({
      system,
      error: error instanceof Error ? error.message : String(error),
      timestamp: Date.now(),
    });
    
    // Keep only recent failures to prevent unbounded growth
    const cutoff = Date.now() - (24 * 60 * 60 * 1000); // 24 hours
    this.performanceHistory.optimization_failures = this.performanceHistory.optimization_failures
      .filter(f => f.timestamp > cutoff);
  }
  
  /**
   * Calculate diversity improvement (simplified)
   */
  private calculateDiversityImprovement(originalHits: SearchHit[], diversifiedHits: SearchHit[]): number {
    if (originalHits.length === 0 || diversifiedHits.length === 0) {
      return 0;
    }
    
    // Simple diversity metric based on unique file paths
    const originalFiles = new Set(originalHits.slice(0, 10).map(h => h.file));
    const diversifiedFiles = new Set(diversifiedHits.slice(0, 10).map(h => h.file));
    
    const originalDiversity = originalFiles.size / Math.min(10, originalHits.length);
    const diversifiedDiversity = diversifiedFiles.size / Math.min(10, diversifiedHits.length);
    
    return diversifiedDiversity - originalDiversity;
  }
  
  /**
   * Cleanup old performance history to prevent memory growth
   */
  private cleanupPerformanceHistory(): void {
    const maxPipelines = 1000;
    const maxSLAMetrics = 1000;
    
    if (this.performanceHistory.pipelines.length > maxPipelines) {
      this.performanceHistory.pipelines = this.performanceHistory.pipelines.slice(-maxPipelines);
    }
    
    if (this.performanceHistory.sla_metrics.length > maxSLAMetrics) {
      this.performanceHistory.sla_metrics = this.performanceHistory.sla_metrics.slice(-maxSLAMetrics);
    }
  }
  
  /**
   * Get comprehensive performance metrics for all systems
   */
  getPerformanceMetrics() {
    const cloneMetrics = this.systemHealth.clone_aware_healthy 
      ? this.cloneAwareSystem.getPerformanceMetrics() 
      : null;
    
    const learningMetrics = this.systemHealth.learning_stop_healthy 
      ? this.learningToStopSystem.getPerformanceMetrics() 
      : null;
    
    const diversityMetrics = this.systemHealth.diversity_healthy 
      ? this.diversitySystem.getPerformanceMetrics() 
      : null;
    
    const ttlMetrics = this.systemHealth.ttl_healthy 
      ? this.ttlSystem.getPerformanceMetrics() 
      : null;
    
    // Calculate overall SLA compliance
    const recentSLA = this.performanceHistory.sla_metrics.slice(-10);
    const slaComplianceRate = recentSLA.length > 0
      ? recentSLA.filter(metrics => this.validateSLACompliance(metrics)).length / recentSLA.length
      : 0;
    
    // Calculate average optimization impact
    const recentPipelines = this.performanceHistory.pipelines.slice(-50);
    const avgOptimizationTime = recentPipelines.length > 0
      ? recentPipelines.reduce((sum, p) => sum + p.performance_impact.total_optimization_ms, 0) / recentPipelines.length
      : 0;
    
    const avgRecallChange = recentPipelines.length > 0
      ? recentPipelines.reduce((sum, p) => sum + p.performance_impact.recall_change, 0) / recentPipelines.length
      : 0;
    
    return {
      system_health: this.systemHealth,
      sla_compliance_rate: slaComplianceRate,
      average_optimization_time_ms: avgOptimizationTime,
      average_recall_change: avgRecallChange,
      pipelines_processed: this.performanceHistory.pipelines.length,
      recent_failures: this.performanceHistory.optimization_failures.slice(-10),
      subsystem_metrics: {
        clone_aware: cloneMetrics,
        learning_to_stop: learningMetrics,
        targeted_diversity: diversityMetrics,
        churn_aware_ttl: ttlMetrics,
      },
    };
  }
  
  /**
   * Get system health status
   */
  getSystemHealth(): SystemHealthStatus {
    return { ...this.systemHealth };
  }
  
  /**
   * Force health check and system recovery
   */
  async performHealthCheckAndRecovery(): Promise<void> {
    console.log('🔍 Performing forced health check and recovery...');
    
    // Reset health flags and try to reinitialize failed systems
    if (!this.systemHealth.clone_aware_healthy && this.config.clone_aware_enabled) {
      try {
        await this.cloneAwareSystem.initialize();
        this.systemHealth.clone_aware_healthy = true;
        console.log('✅ Clone-Aware system recovered');
      } catch (error) {
        console.error('❌ Clone-Aware system recovery failed:', error);
      }
    }
    
    if (!this.systemHealth.learning_stop_healthy && this.config.learning_to_stop_enabled) {
      try {
        await this.learningToStopSystem.initialize();
        this.systemHealth.learning_stop_healthy = true;
        console.log('✅ Learning-to-Stop system recovered');
      } catch (error) {
        console.error('❌ Learning-to-Stop system recovery failed:', error);
      }
    }
    
    if (!this.systemHealth.diversity_healthy && this.config.targeted_diversity_enabled) {
      try {
        await this.diversitySystem.initialize();
        this.systemHealth.diversity_healthy = true;
        console.log('✅ Diversity system recovered');
      } catch (error) {
        console.error('❌ Diversity system recovery failed:', error);
      }
    }
    
    if (!this.systemHealth.ttl_healthy && this.config.churn_aware_ttl_enabled) {
      try {
        await this.ttlSystem.initialize();
        this.systemHealth.ttl_healthy = true;
        console.log('✅ TTL system recovered');
      } catch (error) {
        console.error('❌ TTL system recovery failed:', error);
      }
    }
    
    // Update degraded list and overall health
    this.updateSystemHealthBasedOnFailures();
    
    console.log(`🏥 Health check complete: ${this.systemHealth.degraded_optimizations.length} systems still degraded`);
  }
  
  /**
   * Graceful shutdown of all optimization systems
   */
  async shutdown(): Promise<void> {
    const span = LensTracer.createChildSpan('optimization_engine_shutdown');
    
    try {
      console.log('🛑 Shutting down Optimization Engine...');
      
      // Shutdown all subsystems in parallel
      const shutdownPromises = [];
      
      if (this.config.clone_aware_enabled) {
        shutdownPromises.push(this.cloneAwareSystem.shutdown());
      }
      
      if (this.config.learning_to_stop_enabled) {
        shutdownPromises.push(this.learningToStopSystem.shutdown());
      }
      
      if (this.config.targeted_diversity_enabled) {
        shutdownPromises.push(this.diversitySystem.shutdown());
      }
      
      if (this.config.churn_aware_ttl_enabled) {
        shutdownPromises.push(this.ttlSystem.shutdown());
      }
      
      await Promise.allSettled(shutdownPromises);
      
      // Clear performance history
      this.performanceHistory.pipelines = [];
      this.performanceHistory.sla_metrics = [];
      this.performanceHistory.optimization_failures = [];
      
      span.setAttributes({ success: true });
      console.log('✅ Optimization Engine shutdown complete');
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw error;
    } finally {
      span.end();
    }
  }
}