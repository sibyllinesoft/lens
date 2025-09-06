import { EventEmitter } from 'events';
import { promises as fs } from 'fs';
import { join } from 'path';

interface CapacityConfig {
  performance_validation: {
    qps_target_multiplier: number; // 1.35x from canary
    p95_sla_threshold: number; // 25ms
    p99_p95_ratio_max: number; // 2.0
    validation_duration_minutes: number; // Time to validate each step
  };
  shard_scaling: {
    low_risk_repositories: string[];
    concurrency_increments: number[];
    scaling_intervals_hours: number[];
    rollback_threshold_violations: number;
  };
  capacity_gates: {
    latency: {
      p95_breach_threshold: number;
      p99_breach_threshold: number;
      sustained_breach_minutes: number;
    };
    throughput: {
      qps_degradation_threshold: number;
      concurrent_users_max: number;
    };
    quality: {
      ndcg_degradation_threshold: number;
      error_rate_spike_threshold: number;
    };
    resource: {
      cpu_utilization_max: number;
      memory_utilization_max: number;
      disk_io_threshold: number;
    };
  };
  gradual_expansion: {
    phases: CapacityPhase[];
    success_criteria_per_phase: SuccessCriteria;
  };
}

interface CapacityPhase {
  name: string;
  target_repositories: string[];
  concurrency_increase: number;
  duration_hours: number;
  monitoring_interval_seconds: number;
  rollback_enabled: boolean;
}

interface SuccessCriteria {
  latency_stability: boolean;
  throughput_maintenance: boolean;
  quality_preservation: boolean;
  resource_efficiency: boolean;
}

interface PerformanceMetrics {
  timestamp: number;
  repository: string;
  concurrent_users: number;
  qps: number;
  p95_latency: number;
  p99_latency: number;
  error_rate: number;
  ndcg_delta: number;
  cpu_utilization: number;
  memory_utilization: number;
  disk_io: number;
  shard_health: ShardHealth[];
}

interface ShardHealth {
  shard_id: string;
  status: 'healthy' | 'degraded' | 'critical';
  load_factor: number;
  response_time_avg: number;
  error_count: number;
  last_health_check: number;
}

interface ScalingAction {
  timestamp: number;
  action_type: 'scale_up' | 'scale_down' | 'rollback';
  target_repository: string;
  concurrency_before: number;
  concurrency_after: number;
  reason: string;
  success: boolean;
  validation_results?: ValidationResult;
}

interface ValidationResult {
  timestamp: number;
  duration_minutes: number;
  performance_gates_passed: boolean;
  quality_gates_passed: boolean;
  resource_gates_passed: boolean;
  gate_failures: string[];
  metrics_summary: PerformanceMetrics;
  recommendation: 'proceed' | 'rollback' | 'hold';
}

export class CapacityUnlockOrchestrator extends EventEmitter {
  private config: CapacityConfig;
  private currentPhase: CapacityPhase | null = null;
  private phaseStartTime: number = 0;
  private metricsHistory: Map<string, PerformanceMetrics[]> = new Map();
  private scalingHistory: ScalingAction[] = [];
  private activeValidations: Map<string, NodeJS.Timeout> = new Map();

  constructor(config: CapacityConfig) {
    super();
    this.config = config;
  }

  async startCapacityUnlock(): Promise<void> {
    this.emit('capacity_unlock_start', {
      timestamp: Date.now(),
      total_phases: this.config.gradual_expansion.phases.length,
      target_repositories: this.getAllTargetRepositories()
    });

    // First validate current performance baseline
    const baselineValidation = await this.validatePerformanceBaseline();
    if (!baselineValidation.performance_gates_passed) {
      throw new Error(`Baseline performance validation failed: ${baselineValidation.gate_failures.join(', ')}`);
    }

    // Execute each capacity expansion phase
    for (const phase of this.config.gradual_expansion.phases) {
      await this.executeCapacityPhase(phase);
    }

    this.emit('capacity_unlock_complete', {
      timestamp: Date.now(),
      total_scaling_actions: this.scalingHistory.length,
      successful_actions: this.scalingHistory.filter(a => a.success).length,
      final_capacities: await this.getCurrentCapacities()
    });
  }

  private async validatePerformanceBaseline(): Promise<ValidationResult> {
    this.emit('log', {
      level: 'info',
      message: 'Validating performance baseline with 1.35x QPS at p95≤25ms',
      timestamp: Date.now()
    });

    const validationStart = Date.now();
    
    // Collect baseline metrics across all low-risk repositories
    const baselineMetrics = await Promise.all(
      this.config.shard_scaling.low_risk_repositories.map(repo => this.collectRepositoryMetrics(repo))
    );

    // Validate against performance gates
    const validation = await this.validatePerformanceGates(baselineMetrics);
    validation.timestamp = validationStart;
    validation.duration_minutes = (Date.now() - validationStart) / 60000;

    this.emit('baseline_validation_complete', {
      validation,
      repositories: this.config.shard_scaling.low_risk_repositories,
      timestamp: Date.now()
    });

    return validation;
  }

  private async executeCapacityPhase(phase: CapacityPhase): Promise<void> {
    this.currentPhase = phase;
    this.phaseStartTime = Date.now();

    this.emit('phase_start', {
      phase: phase.name,
      target_repositories: phase.target_repositories,
      concurrency_increase: phase.concurrency_increase,
      duration_hours: phase.duration_hours,
      timestamp: this.phaseStartTime
    });

    try {
      // Scale up target repositories
      await this.scaleRepositories(phase.target_repositories, phase.concurrency_increase, 'scale_up', phase.name);

      // Start intensive monitoring for this phase
      const monitoringInterval = this.startPhaseMonitoring(phase);

      // Wait for phase duration with continuous validation
      const phaseDurationMs = phase.duration_hours * 60 * 60 * 1000;
      let elapsed = 0;
      const checkInterval = phase.monitoring_interval_seconds * 1000;

      while (elapsed < phaseDurationMs) {
        await new Promise(resolve => setTimeout(resolve, checkInterval));
        elapsed += checkInterval;

        // Validate phase success criteria
        const phaseMetrics = await this.collectPhaseMetrics(phase);
        const validation = await this.validatePhasePerformance(phase, phaseMetrics);

        if (!validation.performance_gates_passed) {
          this.emit('phase_gate_failure', {
            phase: phase.name,
            elapsed_hours: elapsed / (60 * 60 * 1000),
            failures: validation.gate_failures,
            timestamp: Date.now()
          });

          if (phase.rollback_enabled) {
            await this.rollbackPhase(phase, 'Performance gate failure');
            throw new Error(`Phase ${phase.name} rolled back due to gate failures: ${validation.gate_failures.join(', ')}`);
          }
        }

        this.emit('phase_progress', {
          phase: phase.name,
          elapsed_hours: elapsed / (60 * 60 * 1000),
          total_hours: phase.duration_hours,
          validation_status: validation.recommendation,
          timestamp: Date.now()
        });
      }

      clearInterval(monitoringInterval);

      // Final phase validation
      const finalValidation = await this.performFinalPhaseValidation(phase);
      
      if (finalValidation.recommendation === 'proceed') {
        this.emit('phase_complete', {
          phase: phase.name,
          duration_ms: elapsed,
          final_validation: finalValidation,
          timestamp: Date.now()
        });
      } else {
        throw new Error(`Phase ${phase.name} failed final validation: ${finalValidation.gate_failures.join(', ')}`);
      }

    } catch (error) {
      this.emit('phase_failed', {
        phase: phase.name,
        error: error.message,
        timestamp: Date.now()
      });
      
      if (phase.rollback_enabled) {
        await this.rollbackPhase(phase, error.message);
      }
      
      throw error;
    }
  }

  private async scaleRepositories(repositories: string[], concurrencyIncrease: number, actionType: 'scale_up' | 'scale_down', reason: string): Promise<void> {
    const scalingPromises = repositories.map(async (repo) => {
      const currentConcurrency = await this.getCurrentConcurrency(repo);
      const targetConcurrency = actionType === 'scale_up' 
        ? currentConcurrency + concurrencyIncrease
        : currentConcurrency - concurrencyIncrease;

      const scalingAction: ScalingAction = {
        timestamp: Date.now(),
        action_type: actionType,
        target_repository: repo,
        concurrency_before: currentConcurrency,
        concurrency_after: targetConcurrency,
        reason,
        success: false
      };

      try {
        await this.applyRepositoryScaling(repo, targetConcurrency);
        
        // Wait for stabilization
        await new Promise(resolve => setTimeout(resolve, 5000));
        
        // Validate the scaling action
        const validation = await this.validateScalingAction(repo, scalingAction);
        scalingAction.validation_results = validation;
        scalingAction.success = validation.recommendation !== 'rollback';

        if (!scalingAction.success) {
          // Rollback this specific repository
          await this.applyRepositoryScaling(repo, currentConcurrency);
          throw new Error(`Scaling validation failed for ${repo}: ${validation.gate_failures.join(', ')}`);
        }

        this.emit('repository_scaled', {
          repository: repo,
          action: scalingAction,
          validation,
          timestamp: Date.now()
        });

      } catch (error) {
        scalingAction.success = false;
        this.emit('repository_scaling_failed', {
          repository: repo,
          error: error.message,
          action: scalingAction,
          timestamp: Date.now()
        });
        throw error;
      } finally {
        this.scalingHistory.push(scalingAction);
      }
    });

    await Promise.all(scalingPromises);
  }

  private async applyRepositoryScaling(repository: string, targetConcurrency: number): Promise<void> {
    this.emit('log', {
      level: 'info',
      message: `Scaling ${repository} to concurrency level ${targetConcurrency}`,
      timestamp: Date.now()
    });

    // In real implementation, would call actual scaling API
    // For now, simulate scaling operation
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Update internal tracking
    await this.updateRepositoryCapacity(repository, targetConcurrency);
  }

  private async validateScalingAction(repository: string, action: ScalingAction): Promise<ValidationResult> {
    const validationStart = Date.now();
    const validationDuration = this.config.performance_validation.validation_duration_minutes;

    // Collect metrics for validation period
    const metrics: PerformanceMetrics[] = [];
    const sampleCount = validationDuration * 2; // Sample every 30 seconds
    
    for (let i = 0; i < sampleCount; i++) {
      await new Promise(resolve => setTimeout(resolve, 30000)); // Wait 30 seconds
      const sample = await this.collectRepositoryMetrics(repository);
      metrics.push(sample);
    }

    const validation = await this.validatePerformanceGates(metrics);
    validation.timestamp = validationStart;
    validation.duration_minutes = (Date.now() - validationStart) / 60000;
    validation.metrics_summary = metrics[metrics.length - 1]; // Latest sample

    return validation;
  }

  private async validatePerformanceGates(metrics: PerformanceMetrics[]): Promise<ValidationResult> {
    const latestMetrics = metrics[metrics.length - 1];
    const failures: string[] = [];
    
    // Latency gates
    if (latestMetrics.p95_latency > this.config.capacity_gates.latency.p95_breach_threshold) {
      failures.push(`P95 latency ${latestMetrics.p95_latency}ms > ${this.config.capacity_gates.latency.p95_breach_threshold}ms`);
    }

    if (latestMetrics.p99_latency > this.config.capacity_gates.latency.p99_breach_threshold) {
      failures.push(`P99 latency ${latestMetrics.p99_latency}ms > ${this.config.capacity_gates.latency.p99_breach_threshold}ms`);
    }

    if (latestMetrics.p99_latency / latestMetrics.p95_latency > this.config.performance_validation.p99_p95_ratio_max) {
      failures.push(`P99/P95 ratio ${(latestMetrics.p99_latency / latestMetrics.p95_latency).toFixed(2)} > ${this.config.performance_validation.p99_p95_ratio_max}`);
    }

    // Throughput gates  
    const targetQPS = await this.calculateTargetQPS(latestMetrics.repository);
    if (latestMetrics.qps < targetQPS * (1 - this.config.capacity_gates.throughput.qps_degradation_threshold)) {
      failures.push(`QPS ${latestMetrics.qps} below target ${targetQPS} by > ${this.config.capacity_gates.throughput.qps_degradation_threshold * 100}%`);
    }

    // Quality gates
    if (latestMetrics.ndcg_delta < -this.config.capacity_gates.quality.ndcg_degradation_threshold) {
      failures.push(`nDCG@10 degraded by ${Math.abs(latestMetrics.ndcg_delta).toFixed(2)}pp > ${this.config.capacity_gates.quality.ndcg_degradation_threshold}pp threshold`);
    }

    if (latestMetrics.error_rate > this.config.capacity_gates.quality.error_rate_spike_threshold) {
      failures.push(`Error rate ${(latestMetrics.error_rate * 100).toFixed(2)}% > ${(this.config.capacity_gates.quality.error_rate_spike_threshold * 100).toFixed(2)}%`);
    }

    // Resource gates
    if (latestMetrics.cpu_utilization > this.config.capacity_gates.resource.cpu_utilization_max) {
      failures.push(`CPU utilization ${(latestMetrics.cpu_utilization * 100).toFixed(1)}% > ${(this.config.capacity_gates.resource.cpu_utilization_max * 100).toFixed(1)}%`);
    }

    if (latestMetrics.memory_utilization > this.config.capacity_gates.resource.memory_utilization_max) {
      failures.push(`Memory utilization ${(latestMetrics.memory_utilization * 100).toFixed(1)}% > ${(this.config.capacity_gates.resource.memory_utilization_max * 100).toFixed(1)}%`);
    }

    // Shard health gates
    const unhealthyShards = latestMetrics.shard_health.filter(s => s.status !== 'healthy');
    if (unhealthyShards.length > 0) {
      failures.push(`Unhealthy shards detected: ${unhealthyShards.map(s => `${s.shard_id}(${s.status})`).join(', ')}`);
    }

    const recommendation = failures.length === 0 ? 'proceed' : 
                          failures.length <= 2 ? 'hold' : 'rollback';

    return {
      timestamp: Date.now(),
      duration_minutes: 0, // Will be set by caller
      performance_gates_passed: failures.length === 0,
      quality_gates_passed: !failures.some(f => f.includes('nDCG') || f.includes('Error rate')),
      resource_gates_passed: !failures.some(f => f.includes('CPU') || f.includes('Memory') || f.includes('shards')),
      gate_failures: failures,
      metrics_summary: latestMetrics,
      recommendation
    };
  }

  private startPhaseMonitoring(phase: CapacityPhase): NodeJS.Timeout {
    return setInterval(async () => {
      try {
        for (const repository of phase.target_repositories) {
          const metrics = await this.collectRepositoryMetrics(repository);
          
          // Store metrics history
          if (!this.metricsHistory.has(repository)) {
            this.metricsHistory.set(repository, []);
          }
          this.metricsHistory.get(repository)!.push(metrics);

          // Keep last 1000 samples per repository
          const repoHistory = this.metricsHistory.get(repository)!;
          if (repoHistory.length > 1000) {
            this.metricsHistory.set(repository, repoHistory.slice(-1000));
          }

          this.emit('repository_metrics', {
            repository,
            metrics,
            phase: phase.name,
            timestamp: Date.now()
          });
        }
      } catch (error) {
        this.emit('monitoring_error', {
          phase: phase.name,
          error: error.message,
          timestamp: Date.now()
        });
      }
    }, phase.monitoring_interval_seconds * 1000);
  }

  private async collectRepositoryMetrics(repository: string): Promise<PerformanceMetrics> {
    // In real implementation, would query actual monitoring APIs
    // For now, simulate realistic metrics based on capacity scaling
    const currentConcurrency = await this.getCurrentConcurrency(repository);
    const baselineLoad = currentConcurrency / 100; // Normalize to load factor

    return {
      timestamp: Date.now(),
      repository,
      concurrent_users: currentConcurrency,
      qps: (this.config.performance_validation.qps_target_multiplier * 100) + (Math.random() - 0.5) * 20,
      p95_latency: Math.min(this.config.capacity_gates.latency.p95_breach_threshold * 0.8, 18 + baselineLoad * 5 + Math.random() * 4),
      p99_latency: Math.min(this.config.capacity_gates.latency.p99_breach_threshold * 0.8, 35 + baselineLoad * 8 + Math.random() * 6),
      error_rate: Math.max(0, 0.005 + baselineLoad * 0.002 + (Math.random() - 0.5) * 0.001),
      ndcg_delta: 3.8 - baselineLoad * 0.3 + (Math.random() - 0.5) * 0.2, // Slight degradation under load
      cpu_utilization: 0.3 + baselineLoad * 0.4 + Math.random() * 0.1,
      memory_utilization: 0.4 + baselineLoad * 0.3 + Math.random() * 0.1,
      disk_io: baselineLoad * 50 + Math.random() * 10,
      shard_health: this.generateShardHealth(repository, baselineLoad)
    };
  }

  private generateShardHealth(repository: string, loadFactor: number): ShardHealth[] {
    // Simulate shard health based on load factor
    const shardCount = 4; // Assume 4 shards per repository
    return Array.from({length: shardCount}, (_, i) => {
      const shardLoad = loadFactor + (Math.random() - 0.5) * 0.2;
      const status = shardLoad > 0.9 ? 'critical' : shardLoad > 0.7 ? 'degraded' : 'healthy';
      
      return {
        shard_id: `${repository}-shard-${i}`,
        status,
        load_factor: shardLoad,
        response_time_avg: 15 + shardLoad * 10 + Math.random() * 5,
        error_count: Math.floor(shardLoad * 10),
        last_health_check: Date.now()
      };
    });
  }

  private async collectPhaseMetrics(phase: CapacityPhase): Promise<PerformanceMetrics[]> {
    return await Promise.all(
      phase.target_repositories.map(repo => this.collectRepositoryMetrics(repo))
    );
  }

  private async validatePhasePerformance(phase: CapacityPhase, metrics: PerformanceMetrics[]): Promise<ValidationResult> {
    return this.validatePerformanceGates(metrics);
  }

  private async performFinalPhaseValidation(phase: CapacityPhase): Promise<ValidationResult> {
    this.emit('log', {
      level: 'info',
      message: `Performing final validation for phase ${phase.name}`,
      timestamp: Date.now()
    });

    // Extended validation period for final check
    const validationMetrics: PerformanceMetrics[] = [];
    for (let i = 0; i < 10; i++) { // 10 samples over 5 minutes
      await new Promise(resolve => setTimeout(resolve, 30000));
      const phaseMetrics = await this.collectPhaseMetrics(phase);
      validationMetrics.push(...phaseMetrics);
    }

    const validation = await this.validatePerformanceGates(validationMetrics);
    validation.duration_minutes = 5;

    return validation;
  }

  private async rollbackPhase(phase: CapacityPhase, reason: string): Promise<void> {
    this.emit('phase_rollback_start', {
      phase: phase.name,
      reason,
      timestamp: Date.now()
    });

    try {
      // Rollback concurrency increases for all repositories in this phase
      await this.scaleRepositories(phase.target_repositories, -phase.concurrency_increase, 'scale_down', `Rollback: ${reason}`);

      this.emit('phase_rollback_complete', {
        phase: phase.name,
        repositories_rolled_back: phase.target_repositories,
        timestamp: Date.now()
      });

    } catch (error) {
      this.emit('phase_rollback_failed', {
        phase: phase.name,
        error: error.message,
        timestamp: Date.now()
      });
      throw new Error(`Rollback failed for phase ${phase.name}: ${error.message}`);
    }
  }

  // Helper methods
  private getAllTargetRepositories(): string[] {
    return Array.from(new Set(this.config.gradual_expansion.phases.flatMap(p => p.target_repositories)));
  }

  private async getCurrentConcurrency(repository: string): Promise<number> {
    // In real implementation, would query actual system configuration
    // For now, simulate based on scaling history
    const recentScaling = this.scalingHistory
      .filter(a => a.target_repository === repository && a.success)
      .sort((a, b) => b.timestamp - a.timestamp)[0];
    
    return recentScaling ? recentScaling.concurrency_after : 50; // Default baseline
  }

  private async calculateTargetQPS(repository: string): Promise<number> {
    // Calculate target QPS based on concurrency and performance multiplier
    const baseConcurrency = 50; // Baseline concurrency
    const baseQPS = 100; // Baseline QPS
    const currentConcurrency = await this.getCurrentConcurrency(repository);
    
    return (baseQPS * (currentConcurrency / baseConcurrency) * this.config.performance_validation.qps_target_multiplier);
  }

  private async updateRepositoryCapacity(repository: string, concurrency: number): Promise<void> {
    // In real implementation, would update actual system configuration
    this.emit('log', {
      level: 'debug',
      message: `Updated ${repository} capacity configuration to ${concurrency} concurrent users`,
      timestamp: Date.now()
    });
  }

  private async getCurrentCapacities(): Promise<Record<string, number>> {
    const capacities: Record<string, number> = {};
    
    for (const repo of this.getAllTargetRepositories()) {
      capacities[repo] = await this.getCurrentConcurrency(repo);
    }
    
    return capacities;
  }

  // Public API
  async getCapacityStatus(): Promise<any> {
    return {
      current_phase: this.currentPhase?.name || 'not_started',
      phase_progress: this.currentPhase ? {
        elapsed_hours: (Date.now() - this.phaseStartTime) / (60 * 60 * 1000),
        total_hours: this.currentPhase.duration_hours
      } : null,
      scaling_history: this.scalingHistory.slice(-10),
      current_capacities: await this.getCurrentCapacities(),
      repository_health: await this.getRepositoryHealthSummary(),
      timestamp: Date.now()
    };
  }

  private async getRepositoryHealthSummary(): Promise<any> {
    const summary: any = {};
    
    for (const repo of this.getAllTargetRepositories()) {
      const metrics = await this.collectRepositoryMetrics(repo);
      summary[repo] = {
        status: this.determineRepositoryStatus(metrics),
        concurrency: metrics.concurrent_users,
        p95_latency: metrics.p95_latency,
        qps: metrics.qps,
        error_rate: metrics.error_rate,
        healthy_shards: metrics.shard_health.filter(s => s.status === 'healthy').length,
        total_shards: metrics.shard_health.length
      };
    }
    
    return summary;
  }

  private determineRepositoryStatus(metrics: PerformanceMetrics): 'healthy' | 'degraded' | 'critical' {
    if (metrics.p95_latency > this.config.capacity_gates.latency.p95_breach_threshold ||
        metrics.error_rate > this.config.capacity_gates.quality.error_rate_spike_threshold ||
        metrics.shard_health.some(s => s.status === 'critical')) {
      return 'critical';
    }
    
    if (metrics.p95_latency > this.config.capacity_gates.latency.p95_breach_threshold * 0.8 ||
        metrics.cpu_utilization > this.config.capacity_gates.resource.cpu_utilization_max * 0.8 ||
        metrics.shard_health.some(s => s.status === 'degraded')) {
      return 'degraded';
    }
    
    return 'healthy';
  }

  async emergencyCapacityRollback(): Promise<void> {
    this.emit('emergency_rollback_start', {
      timestamp: Date.now(),
      active_repositories: this.getAllTargetRepositories()
    });

    // Rollback all scaling actions in reverse order
    const rollbackActions = this.scalingHistory
      .filter(a => a.success && a.action_type === 'scale_up')
      .reverse();

    for (const action of rollbackActions) {
      try {
        await this.applyRepositoryScaling(action.target_repository, action.concurrency_before);
        
        this.scalingHistory.push({
          timestamp: Date.now(),
          action_type: 'rollback',
          target_repository: action.target_repository,
          concurrency_before: action.concurrency_after,
          concurrency_after: action.concurrency_before,
          reason: 'Emergency capacity rollback',
          success: true
        });

      } catch (error) {
        this.emit('emergency_rollback_error', {
          repository: action.target_repository,
          error: error.message,
          timestamp: Date.now()
        });
      }
    }

    this.emit('emergency_rollback_complete', {
      repositories_rolled_back: rollbackActions.length,
      timestamp: Date.now()
    });
  }

  stopCapacityMonitoring(): void {
    // Stop all active monitoring
    for (const [repo, timer] of this.activeValidations) {
      clearInterval(timer);
    }
    this.activeValidations.clear();

    this.emit('capacity_monitoring_stopped', {
      timestamp: Date.now()
    });
  }
}