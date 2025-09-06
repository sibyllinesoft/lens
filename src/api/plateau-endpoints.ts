/**
 * API Endpoints for Engineered Plateau Optimizations
 * Implements /spans/at?sha= endpoint and monitoring interfaces
 */

import type { Request, Response } from 'express';
import { LensTracer } from '../telemetry/tracer.js';
import { EnhancedLensSearchEngine } from './enhanced-search-engine.js';
import type { SearchContext } from '../types/core.js';
import type { PlateauConfig } from '../core/engineered-plateau-orchestrator.js';

export class PlateauEndpoints {
  private searchEngine: EnhancedLensSearchEngine;

  constructor(searchEngine: EnhancedLensSearchEngine) {
    this.searchEngine = searchEngine;
  }

  /**
   * Time-travel search endpoint: GET /spans/at?sha=<commit-sha>&q=<query>&repo=<repo>
   */
  async getSpansAtRevision(req: Request, res: Response): Promise<void> {
    const span = LensTracer.createChildSpan('api_spans_at_revision');

    try {
      const { sha, q, repo, k = '50', mode = 'hybrid' } = req.query;

      // Validate required parameters
      if (!sha || typeof sha !== 'string') {
        res.status(400).json({ 
          error: 'Missing required parameter: sha',
          message: 'Provide commit SHA in query parameter ?sha=<commit-sha>'
        });
        return;
      }

      if (!q || typeof q !== 'string') {
        res.status(400).json({ 
          error: 'Missing required parameter: q',
          message: 'Provide search query in query parameter ?q=<search-terms>'
        });
        return;
      }

      if (!repo || typeof repo !== 'string') {
        res.status(400).json({ 
          error: 'Missing required parameter: repo',
          message: 'Provide repository identifier in query parameter ?repo=<repo-sha>'
        });
        return;
      }

      // Validate SHA format (basic check)
      if (!/^[a-f0-9]{6,40}$/i.test(sha)) {
        res.status(400).json({ 
          error: 'Invalid SHA format',
          message: 'SHA must be 6-40 hexadecimal characters'
        });
        return;
      }

      // Create search context
      const searchContext: SearchContext = {
        trace_id: `plateau-${Date.now()}`,
        query: q,
        repo_sha: repo,
        k: Math.min(parseInt(k as string, 10), 200), // Cap at 200 results
        mode: mode as 'lexical' | 'struct' | 'hybrid',
        fuzzy_distance: 1, // Default fuzzy distance
        started_at: new Date(),
        stages: []
      };

      // Execute time-travel search
      const result = await this.searchEngine.searchAtRevision(searchContext, sha);

      // Return results with metadata
      const response = {
        revision_sha: sha,
        query: q,
        repository: repo,
        results: {
          hits: result.hits.map(hit => ({
            file: hit.file,
            line: hit.line,
            col: hit.col,
            snippet: hit.snippet,
            score: hit.score,
            lang: hit.lang,
            why: hit.why,
            span_len: hit.span_len,
            byte_offset: hit.byte_offset,
            // Revision-specific metadata
            original_line: hit.original_line,
            revision_sha: hit.revision_sha,
            translation_applied: hit.translation_applied,
          })),
          total_found: result.hits.length,
          translation_success_rate: result.translation_success_rate,
        },
        metadata: {
          target_revision: sha,
          search_performed_at: new Date().toISOString(),
          translation_stats: {
            success_rate: result.translation_success_rate,
            metamorphic_validation: result.metamorphic_validation,
          },
        },
      };

      span.setAttributes({
        success: true,
        query: q,
        target_sha: sha.substring(0, 8),
        hits_returned: result.hits.length,
        translation_success_rate: result.translation_success_rate,
        metamorphic_valid: result.metamorphic_validation?.passed,
      });

      res.json(response);

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });

      if (errorMsg.includes('Repository not found')) {
        res.status(404).json({ 
          error: 'Repository not found',
          message: `Repository ${req.query.repo} not found in index`
        });
      } else if (errorMsg.includes('disabled')) {
        res.status(503).json({ 
          error: 'Feature disabled',
          message: 'Revision-aware spans feature is currently disabled'
        });
      } else {
        res.status(500).json({ 
          error: 'Internal server error',
          message: 'Time-travel search failed',
          details: errorMsg,
        });
      }
    } finally {
      span.end();
    }
  }

  /**
   * Enhanced search endpoint with plateau optimizations
   */
  async getEnhancedSearch(req: Request, res: Response): Promise<void> {
    const span = LensTracer.createChildSpan('api_enhanced_search');

    try {
      const { q, repo, k = '50', mode = 'hybrid', fuzzy = '1' } = req.query;

      // Validate required parameters
      if (!q || typeof q !== 'string') {
        res.status(400).json({ 
          error: 'Missing required parameter: q',
          message: 'Provide search query in query parameter ?q=<search-terms>'
        });
        return;
      }

      if (!repo || typeof repo !== 'string') {
        res.status(400).json({ 
          error: 'Missing required parameter: repo',
          message: 'Provide repository identifier in query parameter ?repo=<repo-sha>'
        });
        return;
      }

      // Create search context
      const searchContext: SearchContext = {
        trace_id: `plateau-live-${Date.now()}`,
        query: q,
        repo_sha: repo,
        k: Math.min(parseInt(k as string, 10), 200),
        mode: mode as 'lexical' | 'struct' | 'hybrid',
        fuzzy_distance: Math.min(parseInt(fuzzy as string, 10), 2),
        started_at: new Date(),
        stages: []
      };

      // Execute enhanced search with plateau optimizations
      const result = await this.searchEngine.enhancedSearch(searchContext);

      // Return results with optimization metadata
      const response = {
        query: q,
        repository: repo,
        results: {
          hits: result.hits.map(hit => ({
            file: hit.file,
            line: hit.line,
            col: hit.col,
            snippet: hit.snippet,
            score: hit.score,
            lang: hit.lang,
            why: hit.why,
            span_len: hit.span_len,
            byte_offset: hit.byte_offset,
          })),
          total_found: result.hits.length,
        },
        performance: {
          stage_a_latency_ms: result.stage_a_latency,
          stage_b_latency_ms: result.stage_b_latency,
          stage_c_latency_ms: result.stage_c_latency,
          optimizations_applied: result.optimizations_applied,
          performance_gains: result.performance_gains,
          quality_validated: result.quality_validated,
        },
        metadata: {
          search_performed_at: new Date().toISOString(),
          plateau_optimizations_enabled: result.optimizations_applied.length > 0,
          plateau_metrics: result.plateau_metrics,
        },
      };

      span.setAttributes({
        success: true,
        query: q,
        repo: repo,
        hits_returned: result.hits.length,
        optimizations_applied: result.optimizations_applied.length,
        quality_validated: result.quality_validated,
        p95_improvement: result.performance_gains.p95_improvement_ms,
      });

      res.json(response);

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });

      if (errorMsg.includes('Repository not found')) {
        res.status(404).json({ 
          error: 'Repository not found',
          message: `Repository ${req.query.repo} not found in index`
        });
      } else {
        res.status(500).json({ 
          error: 'Internal server error',
          message: 'Enhanced search failed',
          details: errorMsg,
        });
      }
    } finally {
      span.end();
    }
  }

  /**
   * Plateau optimization metrics endpoint
   */
  async getPlateauMetrics(req: Request, res: Response): Promise<void> {
    const span = LensTracer.createChildSpan('api_plateau_metrics');

    try {
      const metrics = await this.searchEngine.getEnhancedMetrics();

      const response = {
        timestamp: new Date().toISOString(),
        plateau_optimizations: {
          overall_status: metrics.performance_gates.all_gates_passed ? 'healthy' : 'degraded',
          performance_gates: metrics.performance_gates.individual_gates,
          global_metrics: {
            p95_latency_improvement_ms: metrics.plateau_metrics.p95_latency_improvement_ms,
            p99_latency_improvement_ms: metrics.plateau_metrics.p99_latency_improvement_ms,
            cpu_per_query_reduction_percent: metrics.plateau_metrics.cpu_per_query_reduction_percent,
            sla_recall_at_50: metrics.plateau_metrics.sla_recall_at_50,
            why_mix_kl_divergence: metrics.plateau_metrics.why_mix_kl_divergence,
          },
          individual_optimizations: {
            cross_shard_threshold: {
              enabled: metrics.plateau_metrics.individual_optimizer_metrics.cross_shard.enabled,
              traffic_percent: metrics.plateau_metrics.individual_optimizer_metrics.cross_shard.traffic_percent,
              early_stop_rate: metrics.plateau_metrics.individual_optimizer_metrics.cross_shard.early_stop_rate,
              avg_tau: metrics.plateau_metrics.individual_optimizer_metrics.cross_shard.avg_tau,
            },
            tail_taming: {
              enabled: metrics.plateau_metrics.individual_optimizer_metrics.tail_taming.enabled,
              hedge_ratio: metrics.plateau_metrics.individual_optimizer_metrics.tail_taming.hedge_ratio,
              p99_p95_ratio: metrics.plateau_metrics.individual_optimizer_metrics.tail_taming.p99_p95_ratio,
              current_p95_ms: metrics.plateau_metrics.individual_optimizer_metrics.tail_taming.current_p95_ms,
            },
            revision_aware_spans: {
              enabled: metrics.plateau_metrics.individual_optimizer_metrics.revision_spans.enabled,
              cached_line_maps: metrics.plateau_metrics.individual_optimizer_metrics.revision_spans.cached_line_maps,
              cache_hit_rate: metrics.plateau_metrics.individual_optimizer_metrics.revision_spans.cache_hit_rate,
            },
            symbol_sketches: {
              enabled: metrics.plateau_metrics.individual_optimizer_metrics.symbol_sketches.enabled,
              cpu_reduction_estimate: metrics.plateau_metrics.individual_optimizer_metrics.symbol_sketches.cpu_reduction_estimate,
              cache_hit_rate: metrics.plateau_metrics.individual_optimizer_metrics.symbol_sketches.cache_hit_rate,
              avg_sketch_size_bytes: metrics.plateau_metrics.individual_optimizer_metrics.symbol_sketches.avg_sketch_size_bytes,
            },
            postings_io: {
              enabled: metrics.plateau_metrics.individual_optimizer_metrics.postings_io.enabled,
              compression_ratio: metrics.plateau_metrics.individual_optimizer_metrics.postings_io.compression_ratio,
              decode_throughput_mbps: metrics.plateau_metrics.individual_optimizer_metrics.postings_io.decode_throughput_mbps,
              storage_efficiency: metrics.plateau_metrics.individual_optimizer_metrics.postings_io.storage_efficiency,
            },
          },
        },
        baseline_system: {
          status: metrics.baseline_metrics.system_health.status,
          active_queries: metrics.baseline_metrics.active_queries,
          memory_usage_gb: metrics.baseline_metrics.memory_usage_gb,
          shards_healthy: metrics.baseline_metrics.shards_healthy,
        },
      };

      span.setAttributes({
        success: true,
        performance_gates_passed: metrics.performance_gates.all_gates_passed,
        active_optimizations: Object.keys(response.plateau_optimizations.individual_optimizations)
          .filter(key => (response.plateau_optimizations.individual_optimizations as any)[key].enabled)
          .length,
        p95_improvement: metrics.plateau_metrics.p95_latency_improvement_ms,
        cpu_reduction: metrics.plateau_metrics.cpu_per_query_reduction_percent,
      });

      res.json(response);

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });

      res.status(500).json({ 
        error: 'Internal server error',
        message: 'Failed to retrieve plateau metrics',
        details: errorMsg,
      });
    } finally {
      span.end();
    }
  }

  /**
   * Configure plateau optimizations (for A/B testing)
   */
  async configurePlateauOptimizations(req: Request, res: Response): Promise<void> {
    const span = LensTracer.createChildSpan('api_configure_plateau');

    try {
      const config: Partial<PlateauConfig> = req.body;

      // Validate configuration
      if (!config || typeof config !== 'object') {
        res.status(400).json({ 
          error: 'Invalid configuration',
          message: 'Configuration must be a valid JSON object'
        });
        return;
      }

      // Apply configuration
      this.searchEngine.configurePlateauOptimizations(config);

      // Return confirmation
      const response = {
        message: 'Plateau optimizations configured successfully',
        applied_config: config,
        timestamp: new Date().toISOString(),
      };

      span.setAttributes({
        success: true,
        config_keys_updated: Object.keys(config).length,
        cross_shard_enabled: config.crossShardThreshold?.enabled,
        tail_taming_enabled: config.tailTaming?.enabled,
        revision_spans_enabled: config.revisionAwareSpans?.enabled,
      });

      res.json(response);

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });

      if (errorMsg.includes('must be ≤')) {
        res.status(400).json({ 
          error: 'Configuration validation failed',
          message: errorMsg,
        });
      } else {
        res.status(500).json({ 
          error: 'Internal server error',
          message: 'Failed to configure plateau optimizations',
          details: errorMsg,
        });
      }
    } finally {
      span.end();
    }
  }

  /**
   * Enhanced health check with plateau optimization status
   */
  async getEnhancedHealthStatus(req: Request, res: Response): Promise<void> {
    const span = LensTracer.createChildSpan('api_enhanced_health');

    try {
      const health = await this.searchEngine.getEnhancedHealthStatus();

      const response = {
        status: health.status,
        timestamp: new Date().toISOString(),
        baseline_system: {
          status: health.baseline_health.status,
          shards_healthy: health.baseline_health.shards_healthy,
          shards_total: health.baseline_health.shards_total,
          memory_usage_gb: health.baseline_health.memory_usage_gb,
          active_queries: health.baseline_health.active_queries,
        },
        plateau_optimizations: health.plateau_optimizations,
        recommendations: this.generateHealthRecommendations(health),
      };

      span.setAttributes({
        success: true,
        overall_status: health.status,
        baseline_status: health.baseline_health.status,
        plateau_gates_passed: health.plateau_optimizations.performance_gates_passed,
        active_optimizations: Object.values(health.plateau_optimizations.individual_status)
          .filter(opt => opt.enabled).length,
      });

      // Set appropriate HTTP status based on health
      const httpStatus = health.status === 'ok' ? 200 : 
                        health.status === 'degraded' ? 206 : 503;
      res.status(httpStatus).json(response);

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });

      res.status(503).json({
        status: 'down',
        error: 'Health check failed',
        message: errorMsg,
        timestamp: new Date().toISOString(),
      });
    } finally {
      span.end();
    }
  }

  private generateHealthRecommendations(health: any): string[] {
    const recommendations: string[] = [];

    // Check individual optimization health
    if (!health.plateau_optimizations.performance_gates_passed) {
      recommendations.push('Performance gates failing - consider adjusting optimization parameters');
    }

    if (health.baseline_health.status === 'degraded') {
      recommendations.push('Baseline system degraded - check memory usage and shard health');
    }

    // Specific optimization recommendations
    const individual = health.plateau_optimizations.individual_status;
    
    if (individual.cross_shard_threshold.enabled && individual.cross_shard_threshold.early_stops === 0) {
      recommendations.push('Cross-shard threshold optimization enabled but no early stops - check tau calculation');
    }

    if (individual.tail_taming.enabled && individual.tail_taming.hedge_wins === 0) {
      recommendations.push('Tail-taming enabled but no hedge wins - consider adjusting trigger threshold');
    }

    if (individual.revision_aware_spans.enabled && individual.revision_aware_spans.translation_rate < 0.9) {
      recommendations.push('Revision span translation success rate below 90% - check git repository integrity');
    }

    if (recommendations.length === 0) {
      recommendations.push('All systems operating within normal parameters');
    }

    return recommendations;
  }
}

/**
 * Factory function for creating plateau endpoints
 */
export function createPlateauEndpoints(searchEngine: EnhancedLensSearchEngine): PlateauEndpoints {
  return new PlateauEndpoints(searchEngine);
}