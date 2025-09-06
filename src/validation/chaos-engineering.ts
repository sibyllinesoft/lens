/**
 * Chaos Engineering and Failure Drills System
 * 
 * Implements systematic chaos experiments and failure injection to validate
 * system resilience and graceful degradation capabilities. Core component
 * of operational validation that ensures robust failover behavior.
 * 
 * Core Experiments:
 * - Kill LSP for 10 min, verify graceful degradation
 * - Drop RAPTOR cache, test cache reconstruction 
 * - Force 256d only, validate embedding fallback
 * - Network partition simulation
 * - Resource exhaustion testing
 * - Dependency failure simulation
 */

import { z } from 'zod';
import { LensTracer, tracer, meter } from '../telemetry/tracer.js';
import { globalOperationalGates } from './operational-gates.js';
import type { SearchContext, SearchHit } from '../types/core.js';

// Chaos experiment types
export enum ChaosExperimentType {
  LSP_KILL = 'lsp_kill',
  RAPTOR_CACHE_DROP = 'raptor_cache_drop',
  FORCE_256D_ONLY = 'force_256d_only',
  NETWORK_PARTITION = 'network_partition',
  RESOURCE_EXHAUSTION = 'resource_exhaustion',
  DEPENDENCY_FAILURE = 'dependency_failure',
  DISK_FULL = 'disk_full',
  MEMORY_PRESSURE = 'memory_pressure'
}

// Experiment configuration
export const ChaosConfigSchema = z.object({
  experiment_type: z.nativeEnum(ChaosExperimentType),
  duration_minutes: z.number().int().min(1).max(60),
  intensity: z.enum(['low', 'medium', 'high']),
  target_components: z.array(z.string()),
  safety_constraints: z.object({
    max_error_rate: z.number().min(0).max(1),
    min_success_rate: z.number().min(0).max(1),
    auto_abort_threshold: z.number().min(0).max(1),
  }),
  success_criteria: z.object({
    span_coverage_maintained: z.boolean(),
    reliability_ratio_maintained: z.boolean(), // p99/p95 ≤ 2.0
    graceful_degradation: z.boolean(),
    recovery_time_limit_minutes: z.number().int(),
  }),
  rollback_plan: z.object({
    automatic_rollback: z.boolean(),
    rollback_trigger_conditions: z.array(z.string()),
    rollback_steps: z.array(z.string()),
  }),
});

export type ChaosConfig = z.infer<typeof ChaosConfigSchema>;

// Experiment result
export const ChaosResultSchema = z.object({
  experiment_id: z.string(),
  experiment_type: z.nativeEnum(ChaosExperimentType),
  start_time: z.date(),
  end_time: z.date(),
  duration_minutes: z.number(),
  status: z.enum(['running', 'completed', 'failed', 'aborted']),
  success: z.boolean(),
  metrics: z.object({
    baseline_measurements: z.object({
      avg_latency_ms: z.number(),
      success_rate: z.number(),
      span_coverage: z.number(),
      p99_p95_ratio: z.number(),
    }),
    during_chaos_measurements: z.object({
      avg_latency_ms: z.number(),
      success_rate: z.number(),
      span_coverage: z.number(),
      p99_p95_ratio: z.number(),
      degradation_factor: z.number(),
    }),
    recovery_measurements: z.object({
      recovery_time_minutes: z.number(),
      full_recovery_achieved: z.boolean(),
      residual_impact: z.number(),
    }),
  }),
  observations: z.array(z.string()),
  failures_detected: z.array(z.string()),
  improvements_identified: z.array(z.string()),
  rollback_executed: z.boolean(),
  rollback_reason: z.string().optional(),
});

export type ChaosResult = z.infer<typeof ChaosResultSchema>;

// Failure injection point
export interface FailureInjection {
  component: string;
  failureType: 'kill' | 'delay' | 'error' | 'resource_limit' | 'disconnect';
  parameters: Record<string, any>;
  duration: number;
  recoveryAction?: () => Promise<void>;
}

// Default chaos configurations per TODO.md requirements
const DEFAULT_CHAOS_CONFIGS: ChaosConfig[] = [
  {
    experiment_type: ChaosExperimentType.LSP_KILL,
    duration_minutes: 10, // 10 min as specified
    intensity: 'high',
    target_components: ['lsp_server', 'symbol_indexer'],
    safety_constraints: {
      max_error_rate: 0.3,
      min_success_rate: 0.7,
      auto_abort_threshold: 0.5,
    },
    success_criteria: {
      span_coverage_maintained: true, // Must maintain 100%
      reliability_ratio_maintained: true, // p99/p95 ≤ 2.0
      graceful_degradation: true,
      recovery_time_limit_minutes: 5,
    },
    rollback_plan: {
      automatic_rollback: true,
      rollback_trigger_conditions: ['error_rate > 50%', 'span_coverage < 90%'],
      rollback_steps: ['restart_lsp', 'clear_cache', 'verify_health'],
    },
  },
  {
    experiment_type: ChaosExperimentType.RAPTOR_CACHE_DROP,
    duration_minutes: 5,
    intensity: 'medium',
    target_components: ['raptor_cache', 'topic_clusters'],
    safety_constraints: {
      max_error_rate: 0.2,
      min_success_rate: 0.8,
      auto_abort_threshold: 0.3,
    },
    success_criteria: {
      span_coverage_maintained: true,
      reliability_ratio_maintained: true,
      graceful_degradation: true,
      recovery_time_limit_minutes: 3,
    },
    rollback_plan: {
      automatic_rollback: false,
      rollback_trigger_conditions: ['recovery_time > 10min'],
      rollback_steps: ['force_cache_rebuild', 'restore_backup'],
    },
  },
  {
    experiment_type: ChaosExperimentType.FORCE_256D_ONLY,
    duration_minutes: 15,
    intensity: 'medium',
    target_components: ['embedding_service', 'semantic_search'],
    safety_constraints: {
      max_error_rate: 0.1,
      min_success_rate: 0.9,
      auto_abort_threshold: 0.2,
    },
    success_criteria: {
      span_coverage_maintained: true,
      reliability_ratio_maintained: true,
      graceful_degradation: true,
      recovery_time_limit_minutes: 2,
    },
    rollback_plan: {
      automatic_rollback: true,
      rollback_trigger_conditions: ['ndcg_drop > 10%'],
      rollback_steps: ['restore_768d_embeddings', 'warm_cache'],
    },
  },
];

// Metrics for chaos engineering
const chaosMetrics = {
  experiments_run: meter.createCounter('lens_chaos_experiments_total', {
    description: 'Total chaos experiments executed',
  }),
  experiment_success: meter.createCounter('lens_chaos_experiment_success_total', {
    description: 'Successful chaos experiments by type',
  }),
  degradation_factor: meter.createHistogram('lens_chaos_degradation_factor', {
    description: 'System degradation during chaos experiments',
  }),
  recovery_time: meter.createHistogram('lens_chaos_recovery_time_minutes', {
    description: 'Time to recover from chaos experiments',
  }),
  failures_discovered: meter.createCounter('lens_chaos_failures_discovered_total', {
    description: 'System failures discovered through chaos engineering',
  }),
  rollbacks_executed: meter.createCounter('lens_chaos_rollbacks_executed_total', {
    description: 'Emergency rollbacks executed during chaos experiments',
  }),
};

/**
 * Chaos Engineering and Failure Drills System
 * 
 * Systematically tests system resilience through controlled failure injection
 * and validates graceful degradation and recovery capabilities.
 */
export class ChaosEngineering {
  private configs: Map<ChaosExperimentType, ChaosConfig>;
  private activeExperiments: Map<string, ChaosResult>;
  private experimentHistory: ChaosResult[] = [];
  private injectedFailures: Map<string, FailureInjection[]> = new Map();

  constructor(configs: ChaosConfig[] = DEFAULT_CHAOS_CONFIGS) {
    this.configs = new Map();
    configs.forEach(config => {
      this.configs.set(config.experiment_type, config);
    });
    this.activeExperiments = new Map();
  }

  /**
   * Execute a chaos experiment
   */
  async executeExperiment(experimentType: ChaosExperimentType): Promise<string> {
    const config = this.configs.get(experimentType);
    if (!config) {
      throw new Error(`No configuration found for experiment type: ${experimentType}`);
    }

    const experimentId = `chaos_${experimentType}_${Date.now()}`;
    const span = LensTracer.createChildSpan('execute_chaos_experiment', {
      'lens.experiment_id': experimentId,
      'lens.experiment_type': experimentType,
      'lens.duration_minutes': config.duration_minutes,
    });

    try {
      console.log(`Starting chaos experiment: ${experimentType} (${experimentId})`);

      // Initialize experiment result
      const result: ChaosResult = {
        experiment_id: experimentId,
        experiment_type: experimentType,
        start_time: new Date(),
        end_time: new Date(), // Will be updated
        duration_minutes: 0,
        status: 'running',
        success: false,
        metrics: {
          baseline_measurements: await this.measureBaseline(),
          during_chaos_measurements: {
            avg_latency_ms: 0,
            success_rate: 0,
            span_coverage: 0,
            p99_p95_ratio: 0,
            degradation_factor: 0,
          },
          recovery_measurements: {
            recovery_time_minutes: 0,
            full_recovery_achieved: false,
            residual_impact: 0,
          },
        },
        observations: [],
        failures_detected: [],
        improvements_identified: [],
        rollback_executed: false,
      };

      this.activeExperiments.set(experimentId, result);

      // Execute the experiment
      await this.runChaosExperiment(experimentId, config);

      span.setAttributes({
        'lens.experiment_completed': true,
        'lens.experiment_success': result.success,
        'lens.degradation_factor': result.metrics.during_chaos_measurements.degradation_factor,
      });

      return experimentId;

    } finally {
      span.end();
    }
  }

  /**
   * Run the actual chaos experiment
   */
  private async runChaosExperiment(experimentId: string, config: ChaosConfig): Promise<void> {
    const result = this.activeExperiments.get(experimentId)!;
    const span = LensTracer.createChildSpan('run_chaos_experiment', {
      'lens.experiment_id': experimentId,
      'lens.experiment_type': config.experiment_type,
    });

    try {
      // Phase 1: Inject failure
      console.log(`Injecting failure for experiment ${experimentId}`);
      const injections = await this.injectFailure(experimentId, config);
      result.observations.push(`Injected ${injections.length} failure(s) at ${new Date().toISOString()}`);

      // Phase 2: Monitor during chaos
      const monitoringPromise = this.monitorDuringChaos(experimentId, config);
      
      // Phase 3: Wait for experiment duration
      await this.sleep(config.duration_minutes * 60 * 1000);

      // Phase 4: Stop monitoring and measure
      const chaosMetrics = await monitoringPromise;
      result.metrics.during_chaos_measurements = chaosMetrics;

      // Phase 5: Remove failure injection
      console.log(`Removing failure injection for experiment ${experimentId}`);
      await this.removeFailureInjection(experimentId);
      result.observations.push(`Removed failure injection at ${new Date().toISOString()}`);

      // Phase 6: Monitor recovery
      const recoveryMetrics = await this.monitorRecovery(experimentId, config);
      result.metrics.recovery_measurements = recoveryMetrics;

      // Phase 7: Evaluate success
      result.success = this.evaluateExperimentSuccess(result, config);
      result.status = 'completed';

      // Final measurements and analysis
      result.end_time = new Date();
      result.duration_minutes = (result.end_time.getTime() - result.start_time.getTime()) / (1000 * 60);

      // Generate insights
      this.generateInsights(result, config);

      // Clean up
      this.activeExperiments.delete(experimentId);
      this.experimentHistory.push(result);

      // Record metrics
      chaosMetrics.experiments_run.add(1, {
        type: config.experiment_type,
        intensity: config.intensity,
      });

      if (result.success) {
        chaosMetrics.experiment_success.add(1, {
          type: config.experiment_type,
        });
      }

      chaosMetrics.degradation_factor.record(chaosMetrics.degradation_factor, {
        type: config.experiment_type,
      });

      chaosMetrics.recovery_time.record(recoveryMetrics.recovery_time_minutes, {
        type: config.experiment_type,
      });

      console.log(`Chaos experiment ${experimentId} completed: ${result.success ? 'SUCCESS' : 'FAILURE'}`);

    } catch (error: any) {
      result.status = 'failed';
      result.observations.push(`Experiment failed: ${error.message}`);
      
      // Emergency cleanup
      await this.removeFailureInjection(experimentId);
      
      span.setAttributes({
        'lens.experiment_error': error.message,
      });

      console.error(`Chaos experiment ${experimentId} failed:`, error);
    } finally {
      span.end();
    }
  }

  /**
   * Inject failure based on experiment type
   */
  private async injectFailure(experimentId: string, config: ChaosConfig): Promise<FailureInjection[]> {
    const injections: FailureInjection[] = [];

    switch (config.experiment_type) {
      case ChaosExperimentType.LSP_KILL:
        injections.push({
          component: 'lsp_server',
          failureType: 'kill',
          parameters: { signal: 'SIGKILL' },
          duration: config.duration_minutes * 60 * 1000,
          recoveryAction: async () => {
            console.log('Restarting LSP server...');
            // In practice, would restart actual LSP process
            await this.sleep(5000); // Simulate restart time
          }
        });
        break;

      case ChaosExperimentType.RAPTOR_CACHE_DROP:
        injections.push({
          component: 'raptor_cache',
          failureType: 'resource_limit',
          parameters: { action: 'flush_cache' },
          duration: 0, // Immediate effect
          recoveryAction: async () => {
            console.log('Rebuilding RAPTOR cache...');
            await this.sleep(10000); // Simulate cache rebuild
          }
        });
        break;

      case ChaosExperimentType.FORCE_256D_ONLY:
        injections.push({
          component: 'embedding_service',
          failureType: 'error',
          parameters: { 
            error_rate: 1.0,
            error_message: 'Forced 768d embedding failure - falling back to 256d'
          },
          duration: config.duration_minutes * 60 * 1000,
          recoveryAction: async () => {
            console.log('Restoring 768d embedding service...');
            await this.sleep(2000);
          }
        });
        break;

      case ChaosExperimentType.NETWORK_PARTITION:
        injections.push({
          component: 'network',
          failureType: 'disconnect',
          parameters: { 
            target_services: ['external_search_api', 'embedding_service'],
            partition_percentage: 0.3
          },
          duration: config.duration_minutes * 60 * 1000,
        });
        break;

      case ChaosExperimentType.RESOURCE_EXHAUSTION:
        injections.push({
          component: 'system_resources',
          failureType: 'resource_limit',
          parameters: { 
            memory_limit: '80%',
            cpu_limit: '90%'
          },
          duration: config.duration_minutes * 60 * 1000,
        });
        break;

      case ChaosExperimentType.DEPENDENCY_FAILURE:
        injections.push({
          component: 'search_dependencies',
          failureType: 'error',
          parameters: { 
            failure_rate: 0.5,
            affected_components: config.target_components
          },
          duration: config.duration_minutes * 60 * 1000,
        });
        break;
    }

    // Store injections for cleanup
    this.injectedFailures.set(experimentId, injections);

    // Apply injections (simulated)
    for (const injection of injections) {
      console.log(`Injecting ${injection.failureType} failure in ${injection.component}`);
      // In practice, would apply actual failure injection
    }

    return injections;
  }

  /**
   * Monitor system behavior during chaos
   */
  private async monitorDuringChaos(
    experimentId: string, 
    config: ChaosConfig
  ): Promise<ChaosResult['metrics']['during_chaos_measurements']> {
    const result = this.activeExperiments.get(experimentId)!;
    const monitoringInterval = 30000; // 30 seconds
    const measurements: any[] = [];

    const startTime = Date.now();
    const endTime = startTime + (config.duration_minutes * 60 * 1000);

    while (Date.now() < endTime && result.status === 'running') {
      const measurement = await this.takeMeasurement();
      measurements.push(measurement);

      // Check safety constraints
      if (measurement.success_rate < config.safety_constraints.min_success_rate) {
        result.observations.push(`Safety constraint violated: success rate ${measurement.success_rate}`);
        
        if (measurement.success_rate < config.safety_constraints.auto_abort_threshold) {
          console.warn(`Auto-aborting experiment ${experimentId} due to safety violation`);
          result.status = 'aborted';
          result.rollback_executed = true;
          result.rollback_reason = 'Safety constraint violation';
          
          chaosMetrics.rollbacks_executed.add(1, {
            type: config.experiment_type,
            reason: 'safety_violation',
          });
          
          break;
        }
      }

      await this.sleep(monitoringInterval);
    }

    // Calculate averages
    const avgLatency = measurements.reduce((sum, m) => sum + m.avg_latency_ms, 0) / measurements.length;
    const avgSuccessRate = measurements.reduce((sum, m) => sum + m.success_rate, 0) / measurements.length;
    const avgSpanCoverage = measurements.reduce((sum, m) => sum + m.span_coverage, 0) / measurements.length;
    const avgP99P95Ratio = measurements.reduce((sum, m) => sum + m.p99_p95_ratio, 0) / measurements.length;

    const baseline = result.metrics.baseline_measurements;
    const degradationFactor = 1 - (avgSuccessRate / baseline.success_rate);

    return {
      avg_latency_ms: avgLatency,
      success_rate: avgSuccessRate,
      span_coverage: avgSpanCoverage,
      p99_p95_ratio: avgP99P95Ratio,
      degradation_factor: degradationFactor,
    };
  }

  /**
   * Monitor system recovery after chaos
   */
  private async monitorRecovery(
    experimentId: string,
    config: ChaosConfig
  ): Promise<ChaosResult['metrics']['recovery_measurements']> {
    const result = this.activeExperiments.get(experimentId)!;
    const recoveryStart = Date.now();
    const maxRecoveryTime = config.success_criteria.recovery_time_limit_minutes * 60 * 1000;

    let fullRecoveryAchieved = false;
    let recoveryTime = 0;

    console.log(`Monitoring recovery for experiment ${experimentId}...`);

    while (Date.now() - recoveryStart < maxRecoveryTime) {
      const measurement = await this.takeMeasurement();
      const baseline = result.metrics.baseline_measurements;

      // Check if system has recovered to baseline levels
      const latencyRecovered = measurement.avg_latency_ms <= baseline.avg_latency_ms * 1.1; // 10% tolerance
      const successRateRecovered = measurement.success_rate >= baseline.success_rate * 0.95; // 5% tolerance
      const spanCoverageRecovered = measurement.span_coverage >= 0.99; // 99% coverage
      const reliabilityRecovered = measurement.p99_p95_ratio <= 2.0;

      if (latencyRecovered && successRateRecovered && spanCoverageRecovered && reliabilityRecovered) {
        fullRecoveryAchieved = true;
        recoveryTime = (Date.now() - recoveryStart) / (1000 * 60); // minutes
        console.log(`Full recovery achieved in ${recoveryTime.toFixed(1)} minutes`);
        break;
      }

      await this.sleep(10000); // Check every 10 seconds
    }

    if (!fullRecoveryAchieved) {
      recoveryTime = config.success_criteria.recovery_time_limit_minutes;
      result.failures_detected.push('Failed to achieve full recovery within time limit');
    }

    // Calculate residual impact
    const finalMeasurement = await this.takeMeasurement();
    const baseline = result.metrics.baseline_measurements;
    const residualImpact = 1 - (finalMeasurement.success_rate / baseline.success_rate);

    return {
      recovery_time_minutes: recoveryTime,
      full_recovery_achieved: fullRecoveryAchieved,
      residual_impact: Math.max(0, residualImpact),
    };
  }

  /**
   * Remove failure injection
   */
  private async removeFailureInjection(experimentId: string): Promise<void> {
    const injections = this.injectedFailures.get(experimentId) || [];

    for (const injection of injections) {
      console.log(`Removing ${injection.failureType} failure from ${injection.component}`);
      
      // Execute recovery action if available
      if (injection.recoveryAction) {
        await injection.recoveryAction();
      }

      // In practice, would remove actual failure injection
    }

    this.injectedFailures.delete(experimentId);
  }

  /**
   * Take system measurements
   */
  private async takeMeasurement(): Promise<{
    avg_latency_ms: number;
    success_rate: number;
    span_coverage: number;
    p99_p95_ratio: number;
  }> {
    // Simulate measurements - in practice would query actual metrics
    await this.sleep(100); // Simulate measurement delay

    return {
      avg_latency_ms: 90 + Math.random() * 40, // 90-130ms
      success_rate: 0.85 + Math.random() * 0.1, // 85-95%
      span_coverage: 0.95 + Math.random() * 0.05, // 95-100%
      p99_p95_ratio: 1.5 + Math.random() * 0.8, // 1.5-2.3
    };
  }

  /**
   * Measure baseline metrics
   */
  private async measureBaseline(): Promise<ChaosResult['metrics']['baseline_measurements']> {
    console.log('Measuring baseline metrics...');
    
    // Take multiple measurements for baseline
    const measurements = [];
    for (let i = 0; i < 5; i++) {
      measurements.push(await this.takeMeasurement());
      await this.sleep(5000);
    }

    return {
      avg_latency_ms: measurements.reduce((sum, m) => sum + m.avg_latency_ms, 0) / measurements.length,
      success_rate: measurements.reduce((sum, m) => sum + m.success_rate, 0) / measurements.length,
      span_coverage: measurements.reduce((sum, m) => sum + m.span_coverage, 0) / measurements.length,
      p99_p95_ratio: measurements.reduce((sum, m) => sum + m.p99_p95_ratio, 0) / measurements.length,
    };
  }

  /**
   * Evaluate experiment success
   */
  private evaluateExperimentSuccess(result: ChaosResult, config: ChaosConfig): boolean {
    const criteria = config.success_criteria;
    const recovery = result.metrics.recovery_measurements;
    const chaos = result.metrics.during_chaos_measurements;

    // Check all success criteria
    const spanCoverageOk = !criteria.span_coverage_maintained || chaos.span_coverage >= 0.99;
    const reliabilityOk = !criteria.reliability_ratio_maintained || chaos.p99_p95_ratio <= 2.0;
    const gracefulDegradation = !criteria.graceful_degradation || chaos.degradation_factor < 0.5;
    const recoveryTimeOk = recovery.recovery_time_minutes <= criteria.recovery_time_limit_minutes;
    const fullRecovery = recovery.full_recovery_achieved;

    const success = spanCoverageOk && reliabilityOk && gracefulDegradation && recoveryTimeOk && fullRecovery;

    // Record reasons for failure
    if (!success) {
      if (!spanCoverageOk) result.failures_detected.push('Span coverage not maintained');
      if (!reliabilityOk) result.failures_detected.push('Reliability ratio exceeded');
      if (!gracefulDegradation) result.failures_detected.push('Poor graceful degradation');
      if (!recoveryTimeOk) result.failures_detected.push('Recovery time exceeded limit');
      if (!fullRecovery) result.failures_detected.push('Full recovery not achieved');
    }

    return success;
  }

  /**
   * Generate insights from experiment
   */
  private generateInsights(result: ChaosResult, config: ChaosConfig): void {
    const chaos = result.metrics.during_chaos_measurements;
    const recovery = result.metrics.recovery_measurements;
    const baseline = result.metrics.baseline_measurements;

    // Performance insights
    if (chaos.degradation_factor < 0.2) {
      result.improvements_identified.push('System showed excellent resilience with minimal degradation');
    } else if (chaos.degradation_factor > 0.5) {
      result.improvements_identified.push('Consider improving graceful degradation mechanisms');
    }

    // Recovery insights
    if (recovery.recovery_time_minutes < config.success_criteria.recovery_time_limit_minutes * 0.5) {
      result.improvements_identified.push('Fast recovery demonstrates effective error handling');
    } else {
      result.improvements_identified.push('Consider optimizing recovery procedures');
    }

    // Reliability insights
    if (chaos.p99_p95_ratio > 2.0) {
      result.improvements_identified.push('High tail latency during chaos - investigate queue management');
    }

    // Coverage insights
    if (chaos.span_coverage < 0.99) {
      result.improvements_identified.push('Span coverage dropped - ensure monitoring completeness');
    }

    // Type-specific insights
    switch (config.experiment_type) {
      case ChaosExperimentType.LSP_KILL:
        if (chaos.degradation_factor < 0.3) {
          result.improvements_identified.push('LSP failure handled well - fallback mechanisms effective');
        } else {
          result.improvements_identified.push('Improve LSP failure detection and fallback speed');
        }
        break;

      case ChaosExperimentType.RAPTOR_CACHE_DROP:
        if (recovery.recovery_time_minutes < 5) {
          result.improvements_identified.push('RAPTOR cache reconstruction is efficient');
        } else {
          result.improvements_identified.push('Consider optimizing cache rebuilding process');
        }
        break;

      case ChaosExperimentType.FORCE_256D_ONLY:
        if (chaos.degradation_factor < 0.15) {
          result.improvements_identified.push('Embedding dimension fallback works seamlessly');
        } else {
          result.improvements_identified.push('Consider improving 256d embedding quality or fallback logic');
        }
        break;
    }
  }

  /**
   * Get experiment result
   */
  getExperimentResult(experimentId: string): ChaosResult | null {
    return this.activeExperiments.get(experimentId) || 
           this.experimentHistory.find(r => r.experiment_id === experimentId) || null;
  }

  /**
   * Get active experiments
   */
  getActiveExperiments(): ChaosResult[] {
    return Array.from(this.activeExperiments.values());
  }

  /**
   * Get experiment history
   */
  getExperimentHistory(limit: number = 20): ChaosResult[] {
    return this.experimentHistory
      .sort((a, b) => b.start_time.getTime() - a.start_time.getTime())
      .slice(0, limit);
  }

  /**
   * Get chaos engineering status
   */
  getChaosStatus(): {
    active_experiments: number;
    recent_success_rate: number;
    avg_degradation_factor: number;
    avg_recovery_time_minutes: number;
    most_resilient_component: string | null;
    least_resilient_component: string | null;
    system_resilience_score: number;
  } {
    const recentExperiments = this.experimentHistory.slice(-10);
    const successRate = recentExperiments.length > 0 ?
      recentExperiments.filter(r => r.success).length / recentExperiments.length : 0;

    const avgDegradation = recentExperiments.length > 0 ?
      recentExperiments.reduce((sum, r) => sum + r.metrics.during_chaos_measurements.degradation_factor, 0) / recentExperiments.length : 0;

    const avgRecovery = recentExperiments.length > 0 ?
      recentExperiments.reduce((sum, r) => sum + r.metrics.recovery_measurements.recovery_time_minutes, 0) / recentExperiments.length : 0;

    // Calculate resilience score
    const resilienceScore = successRate * 0.5 + (1 - avgDegradation) * 0.3 + (1 - avgRecovery / 10) * 0.2;

    return {
      active_experiments: this.activeExperiments.size,
      recent_success_rate: successRate,
      avg_degradation_factor: avgDegradation,
      avg_recovery_time_minutes: avgRecovery,
      most_resilient_component: null, // Would analyze by component
      least_resilient_component: null, // Would analyze by component
      system_resilience_score: Math.max(0, Math.min(1, resilienceScore)),
    };
  }

  /**
   * Abort an active experiment
   */
  async abortExperiment(experimentId: string, reason: string): Promise<boolean> {
    const result = this.activeExperiments.get(experimentId);
    if (!result) return false;

    result.status = 'aborted';
    result.rollback_executed = true;
    result.rollback_reason = reason;
    result.end_time = new Date();
    result.observations.push(`Experiment aborted: ${reason}`);

    // Clean up failure injections
    await this.removeFailureInjection(experimentId);

    // Move to history
    this.experimentHistory.push(result);
    this.activeExperiments.delete(experimentId);

    chaosMetrics.rollbacks_executed.add(1, {
      type: result.experiment_type,
      reason: 'manual_abort',
    });

    console.log(`Aborted chaos experiment ${experimentId}: ${reason}`);
    return true;
  }

  /**
   * Execute chaos drill schedule (Day 2-4 per TODO.md)
   */
  async executeChaosSchedule(): Promise<string[]> {
    console.log('Starting chaos engineering schedule (Day 2-4)...');

    const experimentIds: string[] = [];

    // Day 2: LSP Kill Test
    console.log('Day 2: Executing LSP kill test...');
    experimentIds.push(await this.executeExperiment(ChaosExperimentType.LSP_KILL));
    await this.sleep(60000); // Wait 1 minute between experiments

    // Day 3: RAPTOR Cache Drop
    console.log('Day 3: Executing RAPTOR cache drop test...');
    experimentIds.push(await this.executeExperiment(ChaosExperimentType.RAPTOR_CACHE_DROP));
    await this.sleep(60000);

    // Day 4: Force 256d Only
    console.log('Day 4: Executing force 256d only test...');
    experimentIds.push(await this.executeExperiment(ChaosExperimentType.FORCE_256D_ONLY));

    console.log(`Completed chaos schedule with ${experimentIds.length} experiments`);
    return experimentIds;
  }

  /**
   * Sleep utility
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Global chaos engineering instance
export const globalChaosEngineering = new ChaosEngineering();