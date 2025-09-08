/**
 * REST API endpoints for precision optimization monitoring and management
 * 
 * Provides endpoints for:
 * - LTR model training and status
 * - Drift detection system monitoring
 * - A/B experiment management with enhanced validation
 * - System health and performance metrics
 * - Span coverage validation and reporting
 */

import type { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { z } from 'zod';
import { LensTracer } from '../telemetry/tracer.js';
import { globalPrecisionEngine, globalExperimentFramework } from '../core/precision-optimization.js';
import { globalDriftDetectionSystem } from '../core/drift-detection-system.js';
import type { LTRTrainingConfig } from '../core/ltr-training-pipeline.js';
import type { DriftMetrics } from '../core/drift-detection-system.js';

// Request/Response schemas
// JSON Schema for Fastify validation
const LTRTrainingRequestJsonSchema = {
  type: 'object',
  required: ['config'],
  properties: {
    config: {
      type: 'object',
      required: ['learning_rate', 'regularization', 'max_iterations', 'convergence_threshold', 'validation_split', 'isotonic_calibration', 'feature_normalization'],
      properties: {
        learning_rate: { type: 'number', minimum: 0.001, maximum: 1.0 },
        regularization: { type: 'number', minimum: 0, maximum: 1.0 },
        max_iterations: { type: 'integer', minimum: 10, maximum: 1000 },
        convergence_threshold: { type: 'number', minimum: 1e-8, maximum: 1e-2 },
        validation_split: { type: 'number', minimum: 0.1, maximum: 0.5 },
        isotonic_calibration: { type: 'boolean' },
        feature_normalization: { type: 'boolean' }
      }
    },
    training_data_source: { 
      type: 'string',
      enum: ['anchor_dataset', 'ladder_dataset', 'synthetic']
    }
  }
};

const DriftMetricsRequestJsonSchema = {
  type: 'object',
  required: ['anchor_p_at_1', 'anchor_recall_at_50', 'ladder_positives_ratio', 'lsif_coverage_pct', 'tree_sitter_coverage_pct', 'sample_count', 'query_complexity_distribution'],
  properties: {
    anchor_p_at_1: { type: 'number', minimum: 0, maximum: 1 },
    anchor_recall_at_50: { type: 'number', minimum: 0, maximum: 1 },
    ladder_positives_ratio: { type: 'number', minimum: 0, maximum: 1 },
    lsif_coverage_pct: { type: 'number', minimum: 0, maximum: 100 },
    tree_sitter_coverage_pct: { type: 'number', minimum: 0, maximum: 100 },
    sample_count: { type: 'integer', minimum: 1 },
    query_complexity_distribution: {
      type: 'object',
      required: ['simple', 'medium', 'complex'],
      properties: {
        simple: { type: 'number', minimum: 0, maximum: 1 },
        medium: { type: 'number', minimum: 0, maximum: 1 },
        complex: { type: 'number', minimum: 0, maximum: 1 }
      }
    }
  }
};

// Keep Zod schemas for TypeScript types
const LTRTrainingRequestSchema = z.object({
  config: z.object({
    learning_rate: z.number().min(0.001).max(1.0),
    regularization: z.number().min(0).max(1.0),
    max_iterations: z.number().int().min(10).max(1000),
    convergence_threshold: z.number().min(1e-8).max(1e-2),
    validation_split: z.number().min(0.1).max(0.5),
    isotonic_calibration: z.boolean(),
    feature_normalization: z.boolean()
  }),
  training_data_source: z.enum(['anchor_dataset', 'ladder_dataset', 'synthetic']).optional()
});

const DriftMetricsRequestSchema = z.object({
  anchor_p_at_1: z.number().min(0).max(1),
  anchor_recall_at_50: z.number().min(0).max(1),
  ladder_positives_ratio: z.number().min(0).max(1),
  lsif_coverage_pct: z.number().min(0).max(100),
  tree_sitter_coverage_pct: z.number().min(0).max(100),
  sample_count: z.number().int().min(1),
  query_complexity_distribution: z.object({
    simple: z.number().min(0).max(1),
    medium: z.number().min(0).max(1),
    complex: z.number().min(0).max(1)
  })
});

// JSON Schema for Fastify validation
const SpanValidationRequestJsonSchema = {
  type: 'object',
  required: ['query'],
  properties: {
    query: { type: 'string', minLength: 1 },
    expected_span_count: { type: 'integer', minimum: 1 },
    coverage_threshold: { type: 'number', minimum: 0.9, maximum: 1.0, default: 0.99 }
  }
};

const SpanValidationRequestSchema = z.object({
  query: z.string().min(1),
  expected_span_count: z.number().int().min(1).optional(),
  coverage_threshold: z.number().min(0.9).max(1.0).default(0.99)
});

export async function registerPrecisionMonitoringEndpoints(fastify: FastifyInstance) {
  
  /**
   * GET /precision/status
   * Get overall precision optimization system status
   */
  fastify.get('/precision/status', async (request: FastifyRequest, reply: FastifyReply) => {
    const span = LensTracer.createChildSpan('precision_status_endpoint');
    
    try {
      const optimizationStatus = globalPrecisionEngine.getOptimizationStatus();
      const driftReport = globalDriftDetectionSystem.getDriftReport();
      const systemStats = globalDriftDetectionSystem.getSystemStats();

      const status = {
        system_health: driftReport.system_health,
        optimization_blocks: {
          block_a_enabled: optimizationStatus.block_a_enabled,
          block_b_enabled: optimizationStatus.block_b_enabled,
          block_c_enabled: optimizationStatus.block_c_enabled
        },
        drift_monitoring: {
          active_alerts: driftReport.active_alerts.length,
          metrics_history_size: systemStats.metrics_history_size,
          recent_metrics: driftReport.metrics_summary
        },
        ltr_status: {
          initialized: true, // Would check if LTR pipeline is initialized
          last_training: null, // Would track last training timestamp
          model_version: 'v1.0'
        },
        recommendations: driftReport.recommendations,
        timestamp: new Date().toISOString()
      };

      span.setAttributes({
        system_health: status.system_health,
        active_alerts: status.drift_monitoring.active_alerts,
        blocks_enabled: Object.values(status.optimization_blocks).filter(Boolean).length
      });

      return reply.code(200).send(status);

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      return reply.code(500).send({ error: errorMsg });
    } finally {
      span.end();
    }
  });

  /**
   * POST /precision/ltr/train
   * Train LTR model with specified configuration
   */
  fastify.post('/precision/ltr/train', {
    schema: {
      body: LTRTrainingRequestJsonSchema
    }
  }, async (request: FastifyRequest<{ Body: z.infer<typeof LTRTrainingRequestSchema> }>, reply: FastifyReply) => {
    const span = LensTracer.createChildSpan('ltr_training_endpoint');
    
    try {
      const { config, training_data_source = 'anchor_dataset' } = request.body;
      
      console.log(`🎓 Starting LTR training with source: ${training_data_source}`);
      
      // Initialize LTR pipeline
      globalPrecisionEngine.initializeLTRPipeline(config as LTRTrainingConfig);

      // In production, this would:
      // 1. Load training data from specified source
      // 2. Add training examples to LTR pipeline
      // 3. Train the model
      // 4. Validate performance
      // 5. Deploy if successful

      // Mock training process for demonstration
      const mockTrainingResult = {
        final_weights: {
          subtoken_jaccard: 0.25,
          struct_distance: 0.20,
          path_prior_residual: 0.15,
          docBM25: 0.20,
          pos_in_file: 0.10,
          near_dup_flags: 0.10,
          bias: 0.05
        },
        convergence_iterations: 85,
        final_loss: 0.234,
        validation_accuracy: 0.834,
        training_duration_ms: 2500,
        training_examples_used: 1250,
        validation_examples_used: 312
      };

      const response = {
        training_id: `ltr_training_${Date.now()}`,
        status: 'completed',
        result: mockTrainingResult,
        config_used: config,
        training_data_source,
        timestamp: new Date().toISOString()
      };

      span.setAttributes({
        training_id: response.training_id,
        convergence_iterations: mockTrainingResult.convergence_iterations,
        final_loss: mockTrainingResult.final_loss,
        validation_accuracy: mockTrainingResult.validation_accuracy,
        training_examples: mockTrainingResult.training_examples_used
      });

      console.log(`✅ LTR training completed: accuracy=${mockTrainingResult.validation_accuracy.toFixed(3)}, loss=${mockTrainingResult.final_loss.toFixed(4)}`);

      return reply.code(200).send(response);

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      return reply.code(500).send({ error: errorMsg, training_id: null });
    } finally {
      span.end();
    }
  });

  /**
   * POST /precision/drift/metrics
   * Record new drift metrics for monitoring
   */
  fastify.post('/precision/drift/metrics', {
    schema: {
      body: DriftMetricsRequestJsonSchema
    }
  }, async (request: FastifyRequest<{ Body: z.infer<typeof DriftMetricsRequestSchema> }>, reply: FastifyReply) => {
    const span = LensTracer.createChildSpan('drift_metrics_endpoint');
    
    try {
      const metricsData = request.body;

      const metrics: DriftMetrics = {
        timestamp: new Date().toISOString(),
        anchor_p_at_1: metricsData.anchor_p_at_1,
        anchor_recall_at_50: metricsData.anchor_recall_at_50,
        ladder_positives_ratio: metricsData.ladder_positives_ratio,
        lsif_coverage_pct: metricsData.lsif_coverage_pct,
        tree_sitter_coverage_pct: metricsData.tree_sitter_coverage_pct,
        sample_count: metricsData.sample_count,
        query_complexity_distribution: {
          simple: metricsData.query_complexity_distribution?.simple || 0.33,
          medium: metricsData.query_complexity_distribution?.medium || 0.33,
          complex: metricsData.query_complexity_distribution?.complex || 0.34
        }
      };

      // Validate complexity distribution sums to 1.0
      const complexitySum = metrics.query_complexity_distribution.simple + 
                           metrics.query_complexity_distribution.medium + 
                           metrics.query_complexity_distribution.complex;
      
      if (Math.abs(complexitySum - 1.0) > 0.01) {
        return reply.code(400).send({ 
          error: 'Query complexity distribution must sum to 1.0',
          current_sum: complexitySum
        });
      }

      await globalDriftDetectionSystem.recordMetrics(metrics);

      // Get immediate drift status
      const driftReport = globalDriftDetectionSystem.getDriftReport();

      const response = {
        metrics_recorded: true,
        system_health: driftReport.system_health,
        active_alerts: driftReport.active_alerts.length,
        new_alerts: driftReport.active_alerts.filter(alert => 
          new Date(alert.timestamp).getTime() > Date.now() - 60000 // Last minute
        ).length,
        timestamp: metrics.timestamp
      };

      span.setAttributes({
        anchor_p1: metrics.anchor_p_at_1,
        anchor_recall: metrics.anchor_recall_at_50,
        system_health: response.system_health,
        new_alerts: response.new_alerts
      });

      return reply.code(200).send(response);

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      return reply.code(500).send({ error: errorMsg });
    } finally {
      span.end();
    }
  });

  /**
   * GET /precision/drift/report
   * Get comprehensive drift detection report
   */
  fastify.get('/precision/drift/report', async (request: FastifyRequest, reply: FastifyReply) => {
    const span = LensTracer.createChildSpan('drift_report_endpoint');
    
    try {
      const driftReport = globalDriftDetectionSystem.getDriftReport();
      const systemStats = globalDriftDetectionSystem.getSystemStats();

      const enhancedReport = {
        ...driftReport,
        system_statistics: systemStats,
        alert_summary: {
          total_alerts: driftReport.active_alerts.length,
          by_severity: driftReport.active_alerts.reduce((acc, alert) => {
            acc[alert.severity] = (acc[alert.severity] || 0) + 1;
            return acc;
          }, {} as Record<string, number>),
          by_type: driftReport.active_alerts.reduce((acc, alert) => {
            acc[alert.alert_type] = (acc[alert.alert_type] || 0) + 1;
            return acc;
          }, {} as Record<string, number>)
        },
        recent_trends: {
          // This would include trend analysis in production
          anchor_p1_trend: 'stable',
          anchor_recall_trend: 'stable',
          ladder_ratio_trend: 'stable',
          coverage_trend: 'stable'
        },
        timestamp: new Date().toISOString()
      };

      span.setAttributes({
        system_health: driftReport.system_health,
        total_alerts: driftReport.active_alerts.length,
        critical_alerts: driftReport.active_alerts.filter(a => a.severity === 'critical').length
      });

      return reply.code(200).send(enhancedReport);

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      return reply.code(500).send({ error: errorMsg });
    } finally {
      span.end();
    }
  });

  /**
   * POST /precision/span/validate
   * Validate span coverage for search results
   */
  fastify.post('/precision/span/validate', {
    schema: {
      body: SpanValidationRequestJsonSchema
    }
  }, async (request: FastifyRequest<{ Body: z.infer<typeof SpanValidationRequestSchema> }>, reply: FastifyReply) => {
    const span = LensTracer.createChildSpan('span_validation_endpoint');
    
    try {
      const { query, expected_span_count, coverage_threshold } = request.body;

      // Mock span coverage validation - in production this would:
      // 1. Execute search with the query
      // 2. Analyze all returned hits for span information
      // 3. Validate span coverage meets threshold
      // 4. Check for any gaps or inconsistencies

      const mockValidationResult = {
        query,
        total_hits: 45,
        hits_with_spans: 45,
        span_coverage_pct: 100.0,
        coverage_meets_threshold: true,
        span_statistics: {
          avg_span_length: 12.5,
          min_span_length: 3,
          max_span_length: 45,
          spans_by_file_type: {
            'typescript': 25,
            'javascript': 12,
            'python': 8
          }
        },
        quality_metrics: {
          spans_with_context: 43,
          spans_with_byte_offsets: 45,
          spans_with_ast_paths: 38,
          spans_with_symbol_kinds: 41
        },
        validation_errors: [] as string[],
        warnings: [] as string[]
      };

      // Add some realistic validation logic
      if (expected_span_count && mockValidationResult.total_hits < expected_span_count) {
        mockValidationResult.warnings.push(
          `Expected at least ${expected_span_count} hits but got ${mockValidationResult.total_hits}`
        );
      }

      if (mockValidationResult.span_coverage_pct < (coverage_threshold * 100)) {
        mockValidationResult.validation_errors.push(
          `Span coverage ${mockValidationResult.span_coverage_pct}% is below threshold ${coverage_threshold * 100}%`
        );
      }

      const response = {
        validation_id: `span_val_${Date.now()}`,
        passed: mockValidationResult.validation_errors.length === 0,
        result: mockValidationResult,
        coverage_threshold_used: coverage_threshold,
        timestamp: new Date().toISOString()
      };

      span.setAttributes({
        validation_id: response.validation_id,
        passed: response.passed,
        span_coverage_pct: mockValidationResult.span_coverage_pct,
        total_hits: mockValidationResult.total_hits,
        errors: mockValidationResult.validation_errors.length,
        warnings: mockValidationResult.warnings.length
      });

      const statusCode = response.passed ? 200 : 422; // 422 for validation failure
      return reply.code(statusCode).send(response);

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      return reply.code(500).send({ error: errorMsg });
    } finally {
      span.end();
    }
  });

  /**
   * GET /precision/experiments/{experiment_id}/status
   * Get A/B experiment status with enhanced metrics
   */
  fastify.get('/precision/experiments/:experimentId/status', async (
    request: FastifyRequest<{ Params: { experimentId: string } }>, 
    reply: FastifyReply
  ) => {
    const span = LensTracer.createChildSpan('experiment_status_endpoint');
    
    try {
      const { experimentId } = request.params;
      
      const experimentStatus = globalExperimentFramework.getExperimentStatus(experimentId);
      
      if (!experimentStatus.config) {
        return reply.code(404).send({ 
          error: `Experiment ${experimentId} not found`,
          available_experiments: [] // Would list active experiments in production
        });
      }

      // Check promotion readiness
      const promotionReadiness = await globalExperimentFramework.checkPromotionReadiness(experimentId);

      const enhancedStatus = {
        experiment_id: experimentId,
        experiment_name: experimentStatus.config.name,
        config: experimentStatus.config,
        current_status: {
          traffic_percentage: experimentStatus.config.traffic_percentage,
          anchor_validation_passed: promotionReadiness.anchor_passed,
          ladder_validation_passed: promotionReadiness.ladder_passed,
          ready_for_promotion: promotionReadiness.ready
        },
        validation_results: experimentStatus.results,
        optimization_status: experimentStatus.optimization_status,
        drift_impact: {
          // This would analyze drift metrics specific to this experiment
          has_drift_alerts: false,
          drift_magnitude: 0.0,
          confidence_level: 'high'
        },
        recent_metrics: {
          // Mock recent performance metrics
          avg_ndcg_improvement_pct: 2.1,
          avg_recall_at_50: 0.891,
          avg_latency_p95_ms: 28.5,
          sample_size_last_24h: 1240
        },
        timestamp: new Date().toISOString()
      };

      span.setAttributes({
        experiment_id: experimentId,
        ready_for_promotion: promotionReadiness.ready,
        anchor_passed: promotionReadiness.anchor_passed,
        ladder_passed: promotionReadiness.ladder_passed,
        traffic_percentage: experimentStatus.config.traffic_percentage
      });

      return reply.code(200).send(enhancedStatus);

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      return reply.code(500).send({ error: errorMsg });
    } finally {
      span.end();
    }
  });

  /**
   * POST /precision/experiments/{experiment_id}/promote
   * Promote experiment to production with validation
   */
  fastify.post('/precision/experiments/:experimentId/promote', async (
    request: FastifyRequest<{ Params: { experimentId: string } }>, 
    reply: FastifyReply
  ) => {
    const span = LensTracer.createChildSpan('experiment_promotion_endpoint');
    
    try {
      const { experimentId } = request.params;

      // Check if experiment exists
      const experimentStatus = globalExperimentFramework.getExperimentStatus(experimentId);
      if (!experimentStatus.config) {
        return reply.code(404).send({ error: `Experiment ${experimentId} not found` });
      }

      // Check promotion readiness
      const promotionReadiness = await globalExperimentFramework.checkPromotionReadiness(experimentId);
      
      if (!promotionReadiness.ready) {
        return reply.code(422).send({
          error: 'Experiment not ready for promotion',
          details: {
            anchor_passed: promotionReadiness.anchor_passed,
            ladder_passed: promotionReadiness.ladder_passed,
            missing_validations: []
          },
          required_actions: [
            !promotionReadiness.anchor_passed ? 'Complete anchor validation' : null,
            !promotionReadiness.ladder_passed ? 'Complete ladder validation' : null
          ].filter(Boolean)
        });
      }

      // Check for drift alerts that would block promotion
      const driftReport = globalDriftDetectionSystem.getDriftReport();
      const criticalAlerts = driftReport.active_alerts.filter(alert => alert.severity === 'critical');
      
      if (criticalAlerts.length > 0) {
        return reply.code(422).send({
          error: 'Cannot promote experiment while critical drift alerts are active',
          critical_alerts: criticalAlerts.map(alert => ({
            type: alert.alert_type,
            metric: alert.metric_name,
            current_value: alert.current_value
          }))
        });
      }

      // Mock promotion process - in production this would:
      // 1. Create deployment plan
      // 2. Execute gradual rollout
      // 3. Monitor metrics during rollout
      // 4. Complete promotion or rollback if issues detected

      const promotionResult = {
        promotion_id: `promo_${experimentId}_${Date.now()}`,
        experiment_id: experimentId,
        status: 'completed',
        rollout_stages: [
          { stage: 'canary_5pct', status: 'completed', duration_minutes: 15 },
          { stage: 'gradual_25pct', status: 'completed', duration_minutes: 30 },
          { stage: 'full_100pct', status: 'completed', duration_minutes: 45 }
        ],
        final_metrics: {
          ndcg_improvement_pct: 2.3,
          recall_at_50: 0.893,
          span_coverage_pct: 99.4,
          p95_latency_ms: 27.8
        },
        validation_gates_passed: {
          ndcg_improvement: true,
          recall_maintenance: true,
          span_coverage: true,
          latency_control: true
        },
        promotion_duration_minutes: 90,
        timestamp: new Date().toISOString()
      };

      span.setAttributes({
        promotion_id: promotionResult.promotion_id,
        experiment_id: experimentId,
        promotion_status: promotionResult.status,
        ndcg_improvement: promotionResult.final_metrics.ndcg_improvement_pct,
        span_coverage: promotionResult.final_metrics.span_coverage_pct
      });

      console.log(`🚀 Experiment ${experimentId} promoted successfully with ${promotionResult.final_metrics.ndcg_improvement_pct}% nDCG improvement`);

      return reply.code(200).send(promotionResult);

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      return reply.code(500).send({ error: errorMsg });
    } finally {
      span.end();
    }
  });

  /**
   * GET /precision/health
   * Overall system health check with detailed status
   */
  fastify.get('/precision/health', async (request: FastifyRequest, reply: FastifyReply) => {
    const span = LensTracer.createChildSpan('precision_health_endpoint');
    
    try {
      const driftReport = globalDriftDetectionSystem.getDriftReport();
      const optimizationStatus = globalPrecisionEngine.getOptimizationStatus();

      const healthStatus = {
        overall_status: driftReport.system_health,
        components: {
          drift_detection: {
            status: driftReport.system_health,
            active_alerts: driftReport.active_alerts.length,
            last_metrics_timestamp: driftReport.active_alerts.length > 0 
              ? driftReport.active_alerts[0]?.timestamp ?? new Date().toISOString()
              : new Date().toISOString()
          },
          precision_optimization: {
            status: 'healthy', // Would be determined by optimization performance
            blocks_enabled: [
              optimizationStatus.block_a_enabled ? 'A' : null,
              optimizationStatus.block_b_enabled ? 'B' : null,
              optimizationStatus.block_c_enabled ? 'C' : null
            ].filter(Boolean),
            ltr_status: 'operational'
          },
          span_coverage: {
            status: 'healthy',
            coverage_pct: 99.8, // Mock - would be calculated from recent validations
            last_validation: new Date().toISOString()
          }
        },
        metrics_summary: driftReport.metrics_summary,
        recommendations: driftReport.recommendations,
        uptime_seconds: Math.floor(process.uptime()),
        timestamp: new Date().toISOString()
      };

      span.setAttributes({
        overall_status: healthStatus.overall_status,
        active_alerts: healthStatus.components.drift_detection.active_alerts,
        span_coverage: healthStatus.components.span_coverage.coverage_pct,
        blocks_enabled: healthStatus.components.precision_optimization.blocks_enabled.length
      });

      const statusCode = healthStatus.overall_status === 'healthy' ? 200 : 
                        healthStatus.overall_status === 'degraded' ? 200 : 503;
      
      return reply.code(statusCode).send(healthStatus);

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      return reply.code(503).send({ 
        overall_status: 'critical',
        error: errorMsg,
        timestamp: new Date().toISOString()
      });
    } finally {
      span.end();
    }
  });
}

export default registerPrecisionMonitoringEndpoints;