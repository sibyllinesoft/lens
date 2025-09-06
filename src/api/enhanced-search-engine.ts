/**
 * Enhanced Search Engine with Engineered Plateau Optimizations
 * Integrates all five advanced optimization systems with the main search pipeline
 */

import { LensSearchEngine } from './search-engine-fixed.js';
import { 
  createEngineeredPlateauOrchestrator, 
  type EngineeredPlateauOrchestrator,
  type PlateauConfig 
} from '../core/engineered-plateau-orchestrator.js';
import { LensTracer } from '../telemetry/tracer.js';
import type { SearchContext } from '../types/core.js';
import type { SearchHit } from '../core/span_resolver/index.js';

interface EnhancedSearchResult {
  hits: SearchHit[];
  stage_a_latency?: number;
  stage_b_latency?: number;
  stage_c_latency?: number;
  // Enhanced metrics
  optimizations_applied: string[];
  performance_gains: {
    p95_improvement_ms: number;
    p99_improvement_ms: number;
    cpu_reduction_percent: number;
  };
  quality_validated: boolean;
  plateau_metrics: any;
}

export class EnhancedLensSearchEngine extends LensSearchEngine {
  private plateauOrchestrator: EngineeredPlateauOrchestrator;
  private repoPath: string;

  constructor(
    indexRoot: string = './indexed-content', 
    repoPath: string = '.',
    plateauConfig?: Partial<PlateauConfig>
  ) {
    super(indexRoot);
    this.repoPath = repoPath;
    this.plateauOrchestrator = createEngineeredPlateauOrchestrator(repoPath, plateauConfig);
  }

  /**
   * Enhanced search with plateau optimizations
   */
  async enhancedSearch(ctx: SearchContext): Promise<EnhancedSearchResult> {
    const span = LensTracer.createChildSpan('enhanced_search_with_plateau');

    try {
      // Apply plateau optimizations to the baseline search
      const optimizedResult = await this.plateauOrchestrator.optimizeSearch(
        ctx,
        async () => {
          // Call the baseline search method from parent class
          return await super.search(ctx);
        }
      );

      const baseResult = optimizedResult.result;
      
      // Construct enhanced result
      const enhancedResult: EnhancedSearchResult = {
        hits: baseResult.hits,
        stage_a_latency: baseResult.stage_a_latency,
        stage_b_latency: baseResult.stage_b_latency,
        stage_c_latency: baseResult.stage_c_latency,
        optimizations_applied: optimizedResult.optimizationsApplied,
        performance_gains: {
          p95_improvement_ms: optimizedResult.performanceGains.p95_latency_improvement_ms,
          p99_improvement_ms: optimizedResult.performanceGains.p99_latency_improvement_ms,
          cpu_reduction_percent: optimizedResult.performanceGains.cpu_per_query_reduction_percent,
        },
        quality_validated: optimizedResult.qualityValidated,
        plateau_metrics: optimizedResult.performanceGains,
      };

      span.setAttributes({
        success: true,
        hits_returned: enhancedResult.hits.length,
        optimizations_count: optimizedResult.optimizationsApplied.length,
        quality_validated: optimizedResult.qualityValidated,
        p95_improvement: optimizedResult.performanceGains.p95_latency_improvement_ms,
      });

      return enhancedResult;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: (error as Error).message });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Time-travel search API endpoint (/spans/at?sha=)
   */
  async searchAtRevision(
    ctx: SearchContext,
    targetSha: string
  ): Promise<{
    hits: SearchHit[];
    revision_sha: string;
    translation_success_rate: number;
    metamorphic_validation?: { passed: boolean; errors: string[] };
  }> {
    const span = LensTracer.createChildSpan('search_at_revision');

    try {
      // First, perform regular search at HEAD
      const baseResult = await this.enhancedSearch(ctx);
      
      // Extract span information for translation
      const baseSpans = baseResult.hits.map(hit => ({
        file: hit.file,
        line: hit.line,
        col: hit.col,
        span_len: hit.span_len,
      }));

      // Translate spans to target revision
      const translatedSpans = await this.plateauOrchestrator.translateSpansToRevision(
        baseSpans,
        targetSha
      );

      // Reconstruct hits with translated spans
      const translatedHits: SearchHit[] = [];
      for (let i = 0; i < baseResult.hits.length; i++) {
        const translatedSpan = translatedSpans[i];
        if (translatedSpan) {
          translatedHits.push({
            ...baseResult.hits[i],
            line: translatedSpan.line,
            col: translatedSpan.col,
            span_len: translatedSpan.span_len,
            // Add revision metadata
            revision_sha: targetSha,
            original_line: baseResult.hits[i].line,
            translation_applied: true,
          });
        }
      }

      // Calculate success rate
      const successCount = translatedSpans.filter(span => span !== null).length;
      const successRate = baseSpans.length > 0 ? successCount / baseSpans.length : 0;

      // Optional: Validate metamorphic property for a sample of files
      let metamorphicValidation;
      if (translatedHits.length > 0) {
        const sampleFile = translatedHits[0].file;
        const sampleLines = translatedHits
          .filter(hit => hit.file === sampleFile)
          .map(hit => hit.line)
          .slice(0, 5); // Test first 5 lines

        metamorphicValidation = await this.plateauOrchestrator.validateMetamorphicProperty(
          sampleFile,
          targetSha,
          sampleLines
        );
      }

      span.setAttributes({
        success: true,
        target_sha: targetSha.substring(0, 8),
        original_hits: baseResult.hits.length,
        translated_hits: translatedHits.length,
        translation_success_rate: successRate,
        metamorphic_valid: metamorphicValidation?.passed,
      });

      return {
        hits: translatedHits,
        revision_sha: targetSha,
        translation_success_rate: successRate,
        metamorphic_validation: metamorphicValidation,
      };

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ 
        success: false, 
        error: (error as Error).message,
        target_sha: targetSha.substring(0, 8),
      });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Configure plateau optimizations for A/B testing
   */
  configurePlateauOptimizations(config: Partial<PlateauConfig>): void {
    this.plateauOrchestrator.updateConfiguration(config);
    console.log(`🎯 Enhanced Search Engine plateau optimizations configured`);
  }

  /**
   * Get comprehensive metrics including plateau performance
   */
  async getEnhancedMetrics(): Promise<{
    baseline_metrics: any;
    plateau_metrics: any;
    performance_gates: {
      all_gates_passed: boolean;
      individual_gates: any;
    };
  }> {
    const span = LensTracer.createChildSpan('get_enhanced_metrics');

    try {
      // Get baseline metrics from parent class
      const baselineHealth = await super.getHealthStatus();
      
      // Get plateau optimization metrics
      const plateauMetrics = this.plateauOrchestrator.getComprehensiveMetrics();
      
      // Combine metrics
      const enhancedMetrics = {
        baseline_metrics: {
          system_health: baselineHealth,
          active_queries: baselineHealth.active_queries,
          memory_usage_gb: baselineHealth.memory_usage_gb,
          shards_healthy: baselineHealth.shards_healthy,
        },
        plateau_metrics: plateauMetrics,
        performance_gates: {
          all_gates_passed: plateauMetrics.performance_gates_passed,
          individual_gates: plateauMetrics.performance_gates,
        },
      };

      span.setAttributes({
        success: true,
        performance_gates_passed: plateauMetrics.performance_gates_passed,
        optimizations_enabled: Object.values(plateauMetrics.individual_optimizer_metrics)
          .filter((metrics: any) => metrics.enabled).length,
        p95_improvement_ms: plateauMetrics.p95_latency_improvement_ms,
        cpu_reduction_percent: plateauMetrics.cpu_per_query_reduction_percent,
      });

      return enhancedMetrics;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: (error as Error).message });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Health check with plateau optimization status
   */
  async getEnhancedHealthStatus(): Promise<{
    status: 'ok' | 'degraded' | 'down';
    baseline_health: any;
    plateau_optimizations: {
      enabled: boolean;
      performance_gates_passed: boolean;
      individual_status: {
        cross_shard_threshold: { enabled: boolean; early_stops: number };
        tail_taming: { enabled: boolean; hedge_wins: number };
        revision_aware_spans: { enabled: boolean; translation_rate: number };
        symbol_sketches: { enabled: boolean; cache_hits: number };
        postings_io: { enabled: boolean; compression_ratio: number };
      };
    };
  }> {
    const span = LensTracer.createChildSpan('enhanced_health_status');

    try {
      const baselineHealth = await super.getHealthStatus();
      const plateauMetrics = this.plateauOrchestrator.getComprehensiveMetrics();
      
      const plateauStatus = {
        enabled: true, // Would check overall config
        performance_gates_passed: plateauMetrics.performance_gates_passed,
        individual_status: {
          cross_shard_threshold: {
            enabled: plateauMetrics.individual_optimizer_metrics.cross_shard.enabled,
            early_stops: plateauMetrics.cross_shard_early_stops,
          },
          tail_taming: {
            enabled: plateauMetrics.individual_optimizer_metrics.tail_taming.enabled,
            hedge_wins: plateauMetrics.tail_taming_hedge_wins,
          },
          revision_aware_spans: {
            enabled: plateauMetrics.individual_optimizer_metrics.revision_spans.enabled,
            translation_rate: plateauMetrics.revision_translation_success_rate,
          },
          symbol_sketches: {
            enabled: plateauMetrics.individual_optimizer_metrics.symbol_sketches.enabled,
            cache_hits: plateauMetrics.symbol_sketch_cache_hits,
          },
          postings_io: {
            enabled: plateauMetrics.individual_optimizer_metrics.postings_io.enabled,
            compression_ratio: plateauMetrics.postings_compression_ratio,
          },
        },
      };

      // Overall status considers both baseline and plateau health
      let status: 'ok' | 'degraded' | 'down' = baselineHealth.status;
      
      // Degrade status if plateau optimizations are failing quality gates
      if (status === 'ok' && !plateauMetrics.performance_gates_passed) {
        status = 'degraded';
      }

      span.setAttributes({
        success: true,
        overall_status: status,
        baseline_status: baselineHealth.status,
        plateau_gates_passed: plateauMetrics.performance_gates_passed,
        optimizations_active: Object.values(plateauStatus.individual_status)
          .filter(opt => opt.enabled).length,
      });

      return {
        status,
        baseline_health: baselineHealth,
        plateau_optimizations: plateauStatus,
      };

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: (error as Error).message });
      
      // Return degraded status on error
      const baselineHealth = await super.getHealthStatus();
      return {
        status: 'degraded' as const,
        baseline_health: baselineHealth,
        plateau_optimizations: {
          enabled: false,
          performance_gates_passed: false,
          individual_status: {
            cross_shard_threshold: { enabled: false, early_stops: 0 },
            tail_taming: { enabled: false, hedge_wins: 0 },
            revision_aware_spans: { enabled: false, translation_rate: 0 },
            symbol_sketches: { enabled: false, cache_hits: 0 },
            postings_io: { enabled: false, compression_ratio: 0 },
          },
        },
      };
    } finally {
      span.end();
    }
  }
}