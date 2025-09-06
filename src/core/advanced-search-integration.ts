/**
 * Advanced Search Integration
 * 
 * Integrates all advanced search optimizations into the main search pipeline:
 * - Conformal router for risk-aware routing
 * - Entropy-gated priors for query-adaptive boosting
 * - RAPTOR hygiene for hierarchical clustering
 * - Embedding roadmap for model distillation
 * - Unicode NFC normalization for spans
 * - Comprehensive monitoring and drift detection
 */

import type { SearchContext, SearchHit } from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';
import { globalConformalRouter, type RoutingDecision } from './conformal-router.js';
import { globalEntropyGatedPriors } from './entropy-gated-priors.js';
import { globalLatencyConditionedMetrics } from './latency-conditioned-metrics.js';
import { globalRAPTORHygiene } from './raptor-hygiene.js';
import { globalEmbeddingRoadmap } from './embedding-roadmap.js';
import { globalUnicodeNormalizer } from './unicode-nfc-normalizer.js';
import { globalComprehensiveMonitoring } from './comprehensive-monitoring.js';

export interface AdvancedSearchConfig {
  conformal_routing_enabled: boolean;
  entropy_gated_priors_enabled: boolean;
  raptor_hierarchical_enabled: boolean;
  embedding_distillation_enabled: boolean;
  unicode_normalization_enabled: boolean;
  comprehensive_monitoring_enabled: boolean;
  safety_gates_enabled: boolean;
  max_p95_latency_ms: number;
  max_upshift_rate_percent: number;
  min_recall_at_50: number;
}

export interface AdvancedSearchResult {
  hits: SearchHit[];
  routing_decision?: RoutingDecision;
  priors_applied: boolean;
  normalization_applied: boolean;
  raptor_used: boolean;
  safety_gates_passed: boolean;
  latency_breakdown: {
    conformal_routing_ms: number;
    entropy_priors_ms: number;
    raptor_search_ms: number;
    normalization_ms: number;
    monitoring_ms: number;
    total_advanced_ms: number;
  };
  quality_metrics: {
    sla_recall_50?: number;
    sla_core_10?: number;
    sla_diversity_10?: number;
  };
}

/**
 * Safety gate validator
 */
class SafetyGateValidator {
  private config: AdvancedSearchConfig;
  
  constructor(config: AdvancedSearchConfig) {
    this.config = config;
  }
  
  /**
   * Validate safety gates before and after search
   */
  validateSafetyGates(
    result: AdvancedSearchResult,
    p95LatencyMs: number
  ): {
    gates_passed: boolean;
    violations: string[];
    recommendations: string[];
  } {
    const violations: string[] = [];
    const recommendations: string[] = [];
    
    // P95 latency gate
    if (p95LatencyMs > this.config.max_p95_latency_ms) {
      violations.push(`P95 latency ${p95LatencyMs.toFixed(1)}ms exceeds ${this.config.max_p95_latency_ms}ms limit`);
      recommendations.push('Reduce Stage-C upshift rate or optimize semantic models');
    }
    
    // Upshift rate gate
    const routerStats = globalConformalRouter.getStats();
    if (routerStats.upshift_rate > this.config.max_upshift_rate_percent) {
      violations.push(`Upshift rate ${routerStats.upshift_rate.toFixed(1)}% exceeds ${this.config.max_upshift_rate_percent}% limit`);
      recommendations.push('Increase conformal risk threshold or reduce risk budget');
    }
    
    // Recall gate
    if (result.quality_metrics.sla_recall_50 && 
        result.quality_metrics.sla_recall_50 < this.config.min_recall_at_50) {
      violations.push(`SLA-Recall@50 ${(result.quality_metrics.sla_recall_50 * 100).toFixed(1)}% below ${(this.config.min_recall_at_50 * 100).toFixed(1)}% minimum`);
      recommendations.push('Review index quality or adjust prior weights');
    }
    
    return {
      gates_passed: violations.length === 0,
      violations,
      recommendations
    };
  }
}

/**
 * Advanced search pipeline orchestrator
 */
export class AdvancedSearchIntegration {
  private static instance: AdvancedSearchIntegration | null = null;
  
  private config: AdvancedSearchConfig;
  private safetyValidator: SafetyGateValidator;
  private enabled = true;
  
  // Execution statistics
  private totalSearches = 0;
  private routedSearches = 0;
  private priorAppliedSearches = 0;
  private raptorSearches = 0;
  private normalizedSearches = 0;
  
  constructor(config?: Partial<AdvancedSearchConfig>) {
    this.config = {
      conformal_routing_enabled: true,
      entropy_gated_priors_enabled: true,
      raptor_hierarchical_enabled: true,
      embedding_distillation_enabled: true,
      unicode_normalization_enabled: true,
      comprehensive_monitoring_enabled: true,
      safety_gates_enabled: true,
      max_p95_latency_ms: 20,
      max_upshift_rate_percent: 5.0,
      min_recall_at_50: 0.8,
      ...config
    };
    
    this.safetyValidator = new SafetyGateValidator(this.config);
    
    // Initialize all subsystems
    this.initializeSubsystems();
  }
  
  /**
   * Get singleton instance
   */
  static getInstance(): AdvancedSearchIntegration {
    if (!AdvancedSearchIntegration.instance) {
      AdvancedSearchIntegration.instance = new AdvancedSearchIntegration();
    }
    return AdvancedSearchIntegration.instance;
  }
  
  /**
   * Initialize all advanced search subsystems
   */
  private initializeSubsystems(): void {
    globalConformalRouter.setEnabled(this.config.conformal_routing_enabled);
    globalEntropyGatedPriors.setEnabled(this.config.entropy_gated_priors_enabled);
    globalRAPTORHygiene.setEnabled(this.config.raptor_hierarchical_enabled);
    globalEmbeddingRoadmap.setEnabled(this.config.embedding_distillation_enabled);
    globalUnicodeNormalizer.setEnabled(this.config.unicode_normalization_enabled);
    globalComprehensiveMonitoring.setEnabled(this.config.comprehensive_monitoring_enabled);
    
    console.log('🚀 Advanced search systems initialized:', {
      conformal_routing: this.config.conformal_routing_enabled,
      entropy_priors: this.config.entropy_gated_priors_enabled,
      raptor_hierarchical: this.config.raptor_hierarchical_enabled,
      embedding_distillation: this.config.embedding_distillation_enabled,
      unicode_normalization: this.config.unicode_normalization_enabled,
      monitoring: this.config.comprehensive_monitoring_enabled
    });
  }
  
  /**
   * Execute advanced search with all optimizations
   */
  async executeAdvancedSearch(
    baseHits: SearchHit[],
    ctx: SearchContext,
    queryEmbedding?: Float32Array
  ): Promise<AdvancedSearchResult> {
    if (!this.enabled) {
      return this.getBaselineResult(baseHits, ctx);
    }
    
    const overallSpan = LensTracer.createChildSpan('advanced_search_integration');
    const timings = {
      conformal_routing_ms: 0,
      entropy_priors_ms: 0,
      raptor_search_ms: 0,
      normalization_ms: 0,
      monitoring_ms: 0,
      total_advanced_ms: 0
    };
    
    const startTime = performance.now();
    this.totalSearches++;
    
    try {
      let hits = [...baseHits];
      let routingDecision: RoutingDecision | undefined;
      let priorsApplied = false;
      let normalizationApplied = false;
      let raptorUsed = false;
      
      // Step 1: Conformal routing decision
      if (this.config.conformal_routing_enabled) {
        const routingStart = performance.now();
        
        routingDecision = await globalConformalRouter.makeRoutingDecision(ctx, hits);
        
        if (routingDecision.should_upshift) {
          this.routedSearches++;
          console.log(`⚡ Conformal routing: ${routingDecision.upshift_type} (expected improvement: +${routingDecision.expected_improvement.toFixed(2)} nDCG)`);
        }
        
        timings.conformal_routing_ms = performance.now() - routingStart;
      }
      
      // Step 2: Unicode NFC normalization for spans
      if (this.config.unicode_normalization_enabled) {
        const normalizationStart = performance.now();
        
        // Normalize snippets in search hits
        for (const hit of hits) {
          if (hit.snippet) {
            const normalizedSpan = globalUnicodeNormalizer.normalizeSpan(
              hit.snippet,
              0,
              hit.snippet.length,
              hit.snippet
            );
            
            if (normalizedSpan.normalization_applied) {
              hit.snippet = normalizedSpan.normalized_text;
              hit.why = [...(hit.why || []), 'unicode_normalized'];
              normalizationApplied = true;
            }
          }
        }
        
        if (normalizationApplied) {
          this.normalizedSearches++;
        }
        
        timings.normalization_ms = performance.now() - normalizationStart;
      }
      
      // Step 3: RAPTOR hierarchical search (if query embedding available)
      if (this.config.raptor_hierarchical_enabled && queryEmbedding) {
        const raptorStart = performance.now();
        
        try {
          const raptorHits = await globalRAPTORHygiene.hierarchicalSearch(
            queryEmbedding,
            ctx,
            Math.min(50, hits.length)
          );
          
          if (raptorHits.length > 0) {
            // Merge RAPTOR results with base hits
            hits = this.mergeRAPTORResults(hits, raptorHits);
            raptorUsed = true;
            this.raptorSearches++;
          }
        } catch (error) {
          console.warn('RAPTOR hierarchical search failed:', error);
        }
        
        timings.raptor_search_ms = performance.now() - raptorStart;
      }
      
      // Step 4: Entropy-gated priors application
      if (this.config.entropy_gated_priors_enabled) {
        const priorsStart = performance.now();
        
        const hitsWithPriors = await globalEntropyGatedPriors.applyPriors(hits, ctx);
        
        if (hitsWithPriors !== hits) {
          hits = hitsWithPriors;
          priorsApplied = true;
          this.priorAppliedSearches++;
        }
        
        timings.entropy_priors_ms = performance.now() - priorsStart;
      }
      
      // Step 5: Generate training data for embedding distillation
      if (this.config.embedding_distillation_enabled && hits.length > 10) {
        try {
          const trainingTriples = await globalEmbeddingRoadmap.generateTrainingData(hits, ctx);
          
          // Run distillation training asynchronously (don't wait)
          if (trainingTriples.length > 0) {
            globalEmbeddingRoadmap.runDistillationTraining(trainingTriples).catch(error => {
              console.warn('Background distillation training failed:', error);
            });
          }
        } catch (error) {
          console.warn('Embedding training data generation failed:', error);
        }
      }
      
      // Step 6: Calculate latency-conditioned metrics
      const totalLatency = performance.now() - startTime;
      timings.total_advanced_ms = totalLatency;
      
      let qualityMetrics = {};
      if (this.config.comprehensive_monitoring_enabled) {
        const monitoringStart = performance.now();
        
        try {
          const metrics = await globalLatencyConditionedMetrics.calculateMetrics(
            hits,
            ctx,
            totalLatency,
            undefined, // No ground truth provided
            2.0 // Default topic entropy
          );
          
          qualityMetrics = {
            sla_recall_50: metrics.sla_recall_50.value,
            sla_core_10: metrics.sla_core_10.value,
            sla_diversity_10: metrics.sla_diversity_10.value
          };
        } catch (error) {
          console.warn('Quality metrics calculation failed:', error);
        }
        
        timings.monitoring_ms = performance.now() - monitoringStart;
        
        // Record query for monitoring
        globalComprehensiveMonitoring.recordQuery(ctx, hits, totalLatency);
      }
      
      // Step 7: Safety gate validation
      const result: AdvancedSearchResult = {
        hits,
        routing_decision: routingDecision,
        priors_applied: priorsApplied,
        normalization_applied: normalizationApplied,
        raptor_used: raptorUsed,
        safety_gates_passed: true, // Will be updated
        latency_breakdown: timings,
        quality_metrics: qualityMetrics
      };
      
      if (this.config.safety_gates_enabled) {
        const safetyValidation = this.safetyValidator.validateSafetyGates(result, totalLatency);
        result.safety_gates_passed = safetyValidation.gates_passed;
        
        if (!safetyValidation.gates_passed) {
          console.warn('🚨 Safety gate violations:', safetyValidation.violations);
          console.warn('💡 Recommendations:', safetyValidation.recommendations);
        }
      }
      
      console.log(`🔬 Advanced search: ${hits.length} results, ${timings.total_advanced_ms.toFixed(1)}ms total (routing: ${routingDecision?.should_upshift ? 'YES' : 'NO'}, priors: ${priorsApplied ? 'YES' : 'NO'}, RAPTOR: ${raptorUsed ? 'YES' : 'NO'})`);
      
      overallSpan.setAttributes({
        success: true,
        hits_count: hits.length,
        routing_upshift: routingDecision?.should_upshift || false,
        priors_applied: priorsApplied,
        raptor_used: raptorUsed,
        normalization_applied: normalizationApplied,
        total_latency_ms: totalLatency,
        safety_gates_passed: result.safety_gates_passed
      });
      
      return result;
      
    } catch (error) {
      overallSpan.recordException(error as Error);
      overallSpan.setAttributes({ success: false });
      console.error('Advanced search integration error:', error);
      
      // Return baseline result on error
      return this.getBaselineResult(baseHits, ctx);
      
    } finally {
      overallSpan.end();
    }
  }
  
  /**
   * Merge RAPTOR hierarchical results with base hits
   */
  private mergeRAPTORResults(baseHits: SearchHit[], raptorHits: SearchHit[]): SearchHit[] {
    const merged: SearchHit[] = [];
    const seen = new Set<string>();
    
    // Add base hits first (higher quality)
    for (const hit of baseHits) {
      const key = `${hit.file}:${hit.line}:${hit.col}`;
      if (!seen.has(key)) {
        merged.push(hit);
        seen.add(key);
      }
    }
    
    // Add RAPTOR hits for diversity
    for (const hit of raptorHits) {
      const key = `${hit.file}:${hit.line}:${hit.col}`;
      if (!seen.has(key) && merged.length < 100) {
        // Boost RAPTOR hits slightly for diversity
        merged.push({
          ...hit,
          score: hit.score * 0.9, // Slight penalty for RAPTOR results
          why: [...(hit.why || []), 'raptor_diversity']
        });
        seen.add(key);
      }
    }
    
    // Re-sort by score
    merged.sort((a, b) => b.score - a.score);
    
    return merged;
  }
  
  /**
   * Get baseline result (no advanced features)
   */
  private getBaselineResult(hits: SearchHit[], ctx: SearchContext): AdvancedSearchResult {
    return {
      hits,
      priors_applied: false,
      normalization_applied: false,
      raptor_used: false,
      safety_gates_passed: true,
      latency_breakdown: {
        conformal_routing_ms: 0,
        entropy_priors_ms: 0,
        raptor_search_ms: 0,
        normalization_ms: 0,
        monitoring_ms: 0,
        total_advanced_ms: 0
      },
      quality_metrics: {}
    };
  }
  
  /**
   * Get advanced search statistics
   */
  getStats(): {
    total_searches: number;
    routed_searches: number;
    prior_applied_searches: number;
    raptor_searches: number;
    normalized_searches: number;
    routing_rate: number;
    prior_application_rate: number;
    raptor_usage_rate: number;
    normalization_rate: number;
    enabled: boolean;
    config: AdvancedSearchConfig;
  } {
    return {
      total_searches: this.totalSearches,
      routed_searches: this.routedSearches,
      prior_applied_searches: this.priorAppliedSearches,
      raptor_searches: this.raptorSearches,
      normalized_searches: this.normalizedSearches,
      routing_rate: this.totalSearches > 0 ? (this.routedSearches / this.totalSearches) * 100 : 0,
      prior_application_rate: this.totalSearches > 0 ? (this.priorAppliedSearches / this.totalSearches) * 100 : 0,
      raptor_usage_rate: this.totalSearches > 0 ? (this.raptorSearches / this.totalSearches) * 100 : 0,
      normalization_rate: this.totalSearches > 0 ? (this.normalizedSearches / this.totalSearches) * 100 : 0,
      enabled: this.enabled,
      config: this.config
    };
  }
  
  /**
   * Enable/disable advanced search integration
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    console.log(`🔬 Advanced search integration ${enabled ? 'ENABLED' : 'DISABLED'}`);
    
    if (!enabled) {
      // Disable all subsystems
      globalConformalRouter.setEnabled(false);
      globalEntropyGatedPriors.setEnabled(false);
      globalRAPTORHygiene.setEnabled(false);
      globalEmbeddingRoadmap.setEnabled(false);
      globalUnicodeNormalizer.setEnabled(false);
      globalComprehensiveMonitoring.setEnabled(false);
    } else {
      // Re-enable based on config
      this.initializeSubsystems();
    }
  }
  
  /**
   * Enable advanced search integration
   */
  enable(): void {
    this.setEnabled(true);
  }
  
  /**
   * Disable advanced search integration
   */
  disable(): void {
    this.setEnabled(false);
  }
  
  /**
   * Update configuration
   */
  updateConfig(config: Partial<AdvancedSearchConfig>): void {
    this.config = { ...this.config, ...config };
    this.safetyValidator = new SafetyGateValidator(this.config);
    
    // Apply config changes to subsystems
    this.initializeSubsystems();
    
    console.log('🔧 Advanced search config updated:', config);
  }
  
  /**
   * Get monitoring dashboard
   */
  async getMonitoringDashboard() {
    if (this.config.comprehensive_monitoring_enabled) {
      return await globalComprehensiveMonitoring.generateDashboard();
    }
    return null;
  }
}

// Global instance
export const globalAdvancedSearch = new AdvancedSearchIntegration();