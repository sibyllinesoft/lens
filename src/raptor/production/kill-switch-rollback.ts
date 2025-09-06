/**
 * Kill-Switch and Auto-Rollback Mechanisms
 * 
 * Implements: Auto-rollback if "p99 > 2×p95, Recall@50_SLA drops, or sentinel NZC < 99%"
 * Addresses: Emergency response system with automated recovery
 */

import { EventEmitter } from 'events';
import { writeFile, readFile, mkdir } from 'fs/promises';
import { join } from 'path';

export interface KillSwitchTrigger {
  id: string;
  name: string;
  condition: string;
  threshold: number;
  operator: '>' | '<' | '>=' | '<=' | '==' | '!=';
  severity: 'critical' | 'major' | 'minor';
  auto_execute: boolean;
  rollback_sequence: string[];
  confirmation_required: boolean;
  cooldown_minutes: number;
}

export interface RollbackStep {
  step_id: string;
  component: string;
  action: 'disable' | 'reduce_weight' | 'fallback' | 'reset_config' | 'emergency_stop';
  parameters: Record<string, any>;
  expected_duration_ms: number;
  validation_check: string;
  rollback_on_failure: boolean;
}

export interface SystemSnapshot {
  timestamp: Date;
  configuration: {
    stage_c_raptor: boolean;
    stage_a_topic_prior: boolean;
    nl_bridge: boolean;
    raptor_weight: number;
    topic_fanout_k: number;
    span_cap: number;
  };
  performance_baseline: {
    p95_latency: number;
    p99_latency: number;
    qps: number;
    error_rate: number;
    recall_50_sla: number;
    sentinel_nzc: number;
  };
  feature_flags: Record<string, boolean>;
  circuit_breaker_states: Record<string, 'closed' | 'open' | 'half_open'>;
}

export interface RollbackExecution {
  trigger_id: string;
  start_time: Date;
  end_time?: Date;
  steps_executed: Array<{
    step: RollbackStep;
    start_time: Date;
    end_time?: Date;
    success: boolean;
    error_message?: string;
    validation_result?: boolean;
  }>;
  success: boolean;
  pre_rollback_snapshot: SystemSnapshot;
  post_rollback_snapshot?: SystemSnapshot;
  recovery_time_ms: number;
}

export class KillSwitchRollbackManager extends EventEmitter {
  private triggers: Map<string, KillSwitchTrigger>;
  private rollbackSteps: Map<string, RollbackStep>;
  private activeTriggers: Set<string>;
  private rollbackHistory: RollbackExecution[];
  private currentSnapshot: SystemSnapshot | null = null;
  private monitoringInterval?: NodeJS.Timeout;
  private emergencyMode: boolean = false;

  constructor() {
    super();
    this.triggers = this.defineKillSwitchTriggers();
    this.rollbackSteps = this.defineRollbackSteps();
    this.activeTriggers = new Set();
    this.rollbackHistory = [];
    this.startContinuousMonitoring();
  }

  /**
   * Define kill-switch triggers based on TODO.md requirements
   */
  private defineKillSwitchTriggers(): Map<string, KillSwitchTrigger> {
    const triggers = new Map<string, KillSwitchTrigger>();

    // CRITICAL: p99 > 2×p95 latency spike
    triggers.set('p99_spike_critical', {
      id: 'p99_spike_critical',
      name: 'p99 Latency Spike Critical',
      condition: 'p99_latency > (2 * p95_latency)',
      threshold: 2.0, // 2x ratio
      operator: '>',
      severity: 'critical',
      auto_execute: true,
      rollback_sequence: ['disable_stage_c', 'disable_topic_fanout', 'emergency_baseline'],
      confirmation_required: false,
      cooldown_minutes: 5
    });

    // CRITICAL: Recall@50_SLA drops significantly
    triggers.set('recall_sla_drop', {
      id: 'recall_sla_drop',
      name: 'Recall@50 SLA Drop',
      condition: 'recall_50_sla < baseline - 0.05',
      threshold: -0.05, // 5pp drop
      operator: '<',
      severity: 'critical',
      auto_execute: true,
      rollback_sequence: ['disable_raptor_features', 'restore_baseline_ranking'],
      confirmation_required: false,
      cooldown_minutes: 10
    });

    // CRITICAL: Sentinel NZC < 99%
    triggers.set('sentinel_nzc_failure', {
      id: 'sentinel_nzc_failure',
      name: 'Sentinel NZC Failure',
      condition: 'sentinel_nzc < 0.99',
      threshold: 0.99, // 99%
      operator: '<',
      severity: 'critical',
      auto_execute: true,
      rollback_sequence: ['disable_nl_bridge', 'disable_topic_prior', 'basic_search_fallback'],
      confirmation_required: false,
      cooldown_minutes: 15
    });

    // MAJOR: Error rate spike
    triggers.set('error_rate_spike', {
      id: 'error_rate_spike',
      name: 'Error Rate Spike',
      condition: 'error_rate > 0.05',
      threshold: 0.05, // 5%
      operator: '>',
      severity: 'major',
      auto_execute: true,
      rollback_sequence: ['reduce_raptor_weight', 'circuit_breaker_open'],
      confirmation_required: false,
      cooldown_minutes: 5
    });

    // MAJOR: QPS degradation
    triggers.set('qps_degradation', {
      id: 'qps_degradation',
      name: 'QPS Degradation',
      condition: 'qps < baseline * 0.7',
      threshold: 0.7, // 70% of baseline
      operator: '<',
      severity: 'major',
      auto_execute: true,
      rollback_sequence: ['reduce_topic_fanout', 'optimize_caching'],
      confirmation_required: false,
      cooldown_minutes: 10
    });

    // MINOR: Success rate degradation
    triggers.set('success_rate_drop', {
      id: 'success_rate_drop',
      name: 'Success Rate Drop',
      condition: 'success_rate < 0.90',
      threshold: 0.90, // 90%
      operator: '<',
      severity: 'minor',
      auto_execute: false,
      rollback_sequence: ['alert_team', 'prepare_rollback'],
      confirmation_required: true,
      cooldown_minutes: 30
    });

    return triggers;
  }

  /**
   * Define rollback steps for different components
   */
  private defineRollbackSteps(): Map<string, RollbackStep> {
    const steps = new Map<string, RollbackStep>();

    // Stage-C RAPTOR features
    steps.set('disable_stage_c', {
      step_id: 'disable_stage_c',
      component: 'stage_c_raptor',
      action: 'disable',
      parameters: { feature_flag: 'stage_c_raptor_enabled', fallback_to: 'basic_ranking' },
      expected_duration_ms: 5000,
      validation_check: 'stage_c_disabled',
      rollback_on_failure: false
    });

    // Topic fanout features
    steps.set('disable_topic_fanout', {
      step_id: 'disable_topic_fanout',
      component: 'stage_a_topic_prior',
      action: 'disable',
      parameters: { feature_flag: 'topic_fanout_enabled', reset_k: 0 },
      expected_duration_ms: 3000,
      validation_check: 'topic_fanout_disabled',
      rollback_on_failure: false
    });

    // NL bridge
    steps.set('disable_nl_bridge', {
      step_id: 'disable_nl_bridge',
      component: 'nl_bridge',
      action: 'disable',
      parameters: { feature_flag: 'nl_bridge_enabled', fallback_to: 'direct_lexical' },
      expected_duration_ms: 2000,
      validation_check: 'nl_bridge_disabled',
      rollback_on_failure: false
    });

    // RAPTOR features
    steps.set('disable_raptor_features', {
      step_id: 'disable_raptor_features',
      component: 'raptor_clustering',
      action: 'disable',
      parameters: { disable_clustering: true, disable_type_tiebreaking: true },
      expected_duration_ms: 8000,
      validation_check: 'raptor_disabled',
      rollback_on_failure: false
    });

    // Weight reduction
    steps.set('reduce_raptor_weight', {
      step_id: 'reduce_raptor_weight',
      component: 'raptor_ranking',
      action: 'reduce_weight',
      parameters: { weight_factor: 0.5, min_weight: 0.1 },
      expected_duration_ms: 1000,
      validation_check: 'weight_reduced',
      rollback_on_failure: true
    });

    // Emergency baseline
    steps.set('emergency_baseline', {
      step_id: 'emergency_baseline',
      component: 'search_engine',
      action: 'reset_config',
      parameters: { config: 'emergency_baseline', disable_all_enhancements: true },
      expected_duration_ms: 15000,
      validation_check: 'baseline_restored',
      rollback_on_failure: false
    });

    // Circuit breaker
    steps.set('circuit_breaker_open', {
      step_id: 'circuit_breaker_open',
      component: 'circuit_breaker',
      action: 'emergency_stop',
      parameters: { component: 'enhanced_features', timeout_ms: 300000 },
      expected_duration_ms: 500,
      validation_check: 'circuit_open',
      rollback_on_failure: false
    });

    return steps;
  }

  /**
   * Take system snapshot for rollback baseline
   */
  async takeSystemSnapshot(): Promise<SystemSnapshot> {
    const snapshot: SystemSnapshot = {
      timestamp: new Date(),
      configuration: {
        stage_c_raptor: await this.getFeatureFlag('stage_c_raptor_enabled'),
        stage_a_topic_prior: await this.getFeatureFlag('topic_fanout_enabled'),
        nl_bridge: await this.getFeatureFlag('nl_bridge_enabled'),
        raptor_weight: await this.getConfigValue('raptor_weight'),
        topic_fanout_k: await this.getConfigValue('topic_fanout_k'),
        span_cap: await this.getConfigValue('span_cap')
      },
      performance_baseline: {
        p95_latency: await this.getMetricValue('p95_latency'),
        p99_latency: await this.getMetricValue('p99_latency'),
        qps: await this.getMetricValue('qps'),
        error_rate: await this.getMetricValue('error_rate'),
        recall_50_sla: await this.getMetricValue('recall_50_sla'),
        sentinel_nzc: await this.getMetricValue('sentinel_nzc')
      },
      feature_flags: await this.getAllFeatureFlags(),
      circuit_breaker_states: await this.getCircuitBreakerStates()
    };

    this.currentSnapshot = snapshot;
    
    this.emit('snapshot_taken', snapshot);
    return snapshot;
  }

  /**
   * Evaluate kill-switch triggers against current metrics
   */
  async evaluateKillSwitches(metrics: Record<string, number>): Promise<void> {
    if (this.emergencyMode) {
      console.log('⚠️ System in emergency mode - skipping additional triggers');
      return;
    }

    for (const [triggerId, trigger] of this.triggers) {
      // Check cooldown
      if (this.activeTriggers.has(triggerId)) {
        continue;
      }

      const shouldTrigger = await this.evaluateTriggerCondition(trigger, metrics);
      
      if (shouldTrigger) {
        console.log(`🚨 KILL-SWITCH TRIGGERED: ${trigger.name}`);
        
        if (trigger.auto_execute && !trigger.confirmation_required) {
          await this.executeRollback(triggerId, metrics);
        } else {
          console.log(`⚠️ Manual confirmation required for: ${trigger.name}`);
          this.emit('rollback_confirmation_required', {
            trigger_id: triggerId,
            trigger_name: trigger.name,
            metrics
          });
        }
      }
    }
  }

  /**
   * Execute rollback sequence
   */
  async executeRollback(triggerId: string, currentMetrics: Record<string, number>): Promise<RollbackExecution> {
    const trigger = this.triggers.get(triggerId);
    if (!trigger) {
      throw new Error(`Trigger not found: ${triggerId}`);
    }

    console.log(`🔄 Starting rollback sequence: ${trigger.name}`);
    
    // Take pre-rollback snapshot
    const preSnapshot = await this.takeSystemSnapshot();
    
    const execution: RollbackExecution = {
      trigger_id: triggerId,
      start_time: new Date(),
      steps_executed: [],
      success: false,
      pre_rollback_snapshot: preSnapshot,
      recovery_time_ms: 0
    };

    // Mark trigger as active
    this.activeTriggers.add(triggerId);
    
    // Enter emergency mode for critical triggers
    if (trigger.severity === 'critical') {
      this.emergencyMode = true;
      console.log('🚨 Entering emergency mode');
    }

    try {
      // Execute rollback steps in sequence
      for (const stepId of trigger.rollback_sequence) {
        const step = this.rollbackSteps.get(stepId);
        if (!step) {
          console.error(`❌ Rollback step not found: ${stepId}`);
          continue;
        }

        console.log(`  🔧 Executing step: ${step.component} - ${step.action}`);
        
        const stepExecution = {
          step,
          start_time: new Date(),
          end_time: undefined as Date | undefined,
          success: false,
          validation_result: false,
          error_message: undefined as string | undefined
        };

        try {
          // Execute the step
          await this.executeRollbackStep(step);
          
          // Wait for step completion
          await new Promise(resolve => setTimeout(resolve, step.expected_duration_ms));
          
          // Validate step success
          const validationResult = await this.validateRollbackStep(step);
          
          stepExecution.end_time = new Date();
          stepExecution.success = true;
          stepExecution.validation_result = validationResult;
          
          console.log(`    ✅ Step completed: ${step.component}`);
          
          if (!validationResult) {
            console.warn(`    ⚠️ Validation failed for: ${step.component}`);
          }

        } catch (error) {
          stepExecution.end_time = new Date();
          stepExecution.success = false;
          stepExecution.error_message = (error as Error).message;
          
          console.error(`    ❌ Step failed: ${step.component} - ${error}`);
          
          // Rollback this step if configured
          if (step.rollback_on_failure) {
            console.log(`    🔄 Rolling back failed step: ${step.component}`);
            await this.rollbackFailedStep(step);
          }
        }

        execution.steps_executed.push(stepExecution);
      }

      // Take post-rollback snapshot
      execution.post_rollback_snapshot = await this.takeSystemSnapshot();
      execution.end_time = new Date();
      execution.recovery_time_ms = execution.end_time.getTime() - execution.start_time.getTime();
      
      // Check overall success
      const successfulSteps = execution.steps_executed.filter(s => s.success).length;
      execution.success = successfulSteps >= Math.ceil(execution.steps_executed.length * 0.8); // 80% success threshold

      if (execution.success) {
        console.log(`✅ Rollback completed successfully in ${execution.recovery_time_ms}ms`);
        
        // Schedule cooldown
        setTimeout(() => {
          this.activeTriggers.delete(triggerId);
          console.log(`🔄 Trigger cooldown expired: ${triggerId}`);
        }, trigger.cooldown_minutes * 60 * 1000);
        
      } else {
        console.error(`❌ Rollback failed: ${triggerId}`);
        // Escalate to emergency baseline
        await this.executeEmergencyBaseline();
      }

    } catch (error) {
      execution.end_time = new Date();
      execution.success = false;
      execution.recovery_time_ms = execution.end_time.getTime() - execution.start_time.getTime();
      
      console.error(`❌ Rollback execution failed: ${error}`);
      
      // Emergency baseline as last resort
      await this.executeEmergencyBaseline();
    }

    // Exit emergency mode if successful
    if (execution.success && trigger.severity === 'critical') {
      this.emergencyMode = false;
      console.log('✅ Exiting emergency mode');
    }

    // Record execution
    this.rollbackHistory.push(execution);
    
    // Emit completion event
    this.emit('rollback_completed', execution);
    
    return execution;
  }

  /**
   * Manual rollback initiation
   */
  async initiateManualRollback(
    triggerId: string, 
    reason: string = 'Manual initiation'
  ): Promise<RollbackExecution> {
    console.log(`🔧 Manual rollback initiated: ${triggerId} - ${reason}`);
    
    const dummyMetrics = {
      p95_latency: 0,
      p99_latency: 0,
      error_rate: 0,
      qps: 0,
      recall_50_sla: 0,
      sentinel_nzc: 0
    };
    
    return await this.executeRollback(triggerId, dummyMetrics);
  }

  /**
   * Emergency baseline restoration
   */
  private async executeEmergencyBaseline(): Promise<void> {
    console.log('🚨 EMERGENCY BASELINE RESTORATION');
    
    this.emergencyMode = true;
    
    const emergencySteps = [
      'disable_stage_c',
      'disable_topic_fanout', 
      'disable_nl_bridge',
      'disable_raptor_features',
      'emergency_baseline'
    ];

    for (const stepId of emergencySteps) {
      const step = this.rollbackSteps.get(stepId);
      if (step) {
        try {
          await this.executeRollbackStep(step);
          console.log(`  ✅ Emergency step: ${step.component}`);
        } catch (error) {
          console.error(`  ❌ Emergency step failed: ${step.component} - ${error}`);
        }
      }
    }

    console.log('🚨 Emergency baseline restoration complete');
  }

  /**
   * Start continuous monitoring
   */
  private startContinuousMonitoring(): void {
    this.monitoringInterval = setInterval(async () => {
      try {
        const metrics = await this.collectCurrentMetrics();
        await this.evaluateKillSwitches(metrics);
      } catch (error) {
        console.error('Kill-switch monitoring error:', error);
      }
    }, 30000); // Check every 30 seconds
  }

  /**
   * Stop monitoring
   */
  stop(): void {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
    }
  }

  // Helper methods for system interaction (would be implemented for actual system)
  private async getFeatureFlag(flag: string): Promise<boolean> {
    // Mock implementation
    return true;
  }

  private async getConfigValue(key: string): Promise<number> {
    // Mock implementation
    const defaults: Record<string, number> = {
      raptor_weight: 0.4,
      topic_fanout_k: 320,
      span_cap: 8
    };
    return defaults[key] || 0;
  }

  private async getMetricValue(metric: string): Promise<number> {
    // Mock implementation - would fetch from monitoring system
    const mockValues: Record<string, number> = {
      p95_latency: 145,
      p99_latency: 220,
      qps: 850,
      error_rate: 0.015,
      recall_50_sla: 0.68,
      sentinel_nzc: 0.995
    };
    return mockValues[metric] || 0;
  }

  private async getAllFeatureFlags(): Promise<Record<string, boolean>> {
    return {
      stage_c_raptor_enabled: true,
      topic_fanout_enabled: true,
      nl_bridge_enabled: true
    };
  }

  private async getCircuitBreakerStates(): Promise<Record<string, 'closed' | 'open' | 'half_open'>> {
    return {
      enhanced_features: 'closed',
      raptor_clustering: 'closed',
      topic_fanout: 'closed'
    };
  }

  private async evaluateTriggerCondition(trigger: KillSwitchTrigger, metrics: Record<string, number>): Promise<boolean> {
    // Simplified condition evaluation - would be more sophisticated in production
    switch (trigger.id) {
      case 'p99_spike_critical':
        return metrics.p99_latency > (2 * metrics.p95_latency);
      case 'recall_sla_drop':
        return metrics.recall_50_sla < (0.68 - 0.05); // Baseline - 5pp
      case 'sentinel_nzc_failure':
        return metrics.sentinel_nzc < 0.99;
      case 'error_rate_spike':
        return metrics.error_rate > 0.05;
      case 'qps_degradation':
        return metrics.qps < (850 * 0.7); // 70% of baseline
      case 'success_rate_drop':
        return (metrics.success_rate || 0.95) < 0.90;
      default:
        return false;
    }
  }

  private async executeRollbackStep(step: RollbackStep): Promise<void> {
    console.log(`    🔧 Executing ${step.action} on ${step.component}`);
    
    // Mock implementation - would interface with actual system
    switch (step.action) {
      case 'disable':
        console.log(`      Disabling ${step.component}`);
        break;
      case 'reduce_weight':
        console.log(`      Reducing weight for ${step.component}`);
        break;
      case 'reset_config':
        console.log(`      Resetting config for ${step.component}`);
        break;
      case 'emergency_stop':
        console.log(`      Emergency stop for ${step.component}`);
        break;
    }
  }

  private async validateRollbackStep(step: RollbackStep): Promise<boolean> {
    // Mock validation - would check actual system state
    return Math.random() > 0.1; // 90% success rate
  }

  private async rollbackFailedStep(step: RollbackStep): Promise<void> {
    console.log(`    🔄 Rolling back failed step: ${step.component}`);
  }

  private async collectCurrentMetrics(): Promise<Record<string, number>> {
    // Mock metrics collection - would integrate with telemetry system
    return {
      p95_latency: 145 + Math.random() * 20,
      p99_latency: 220 + Math.random() * 50,
      qps: 850 + Math.random() * 100,
      error_rate: 0.015 + Math.random() * 0.01,
      recall_50_sla: 0.68 + Math.random() * 0.05,
      sentinel_nzc: 0.995 + Math.random() * 0.005,
      success_rate: 0.95 + Math.random() * 0.04
    };
  }

  /**
   * Generate rollback status report
   */
  generateStatusReport(): string {
    let report = '# Kill-Switch & Rollback Status Report\n\n';
    
    report += `**Emergency Mode**: ${this.emergencyMode ? '🚨 ACTIVE' : '✅ NORMAL'}\n`;
    report += `**Active Triggers**: ${this.activeTriggers.size}\n`;
    report += `**Total Rollbacks**: ${this.rollbackHistory.length}\n\n`;

    if (this.activeTriggers.size > 0) {
      report += '## 🚨 Active Triggers\n\n';
      for (const triggerId of this.activeTriggers) {
        const trigger = this.triggers.get(triggerId);
        if (trigger) {
          report += `- **${trigger.name}** (${trigger.severity})\n`;
        }
      }
      report += '\n';
    }

    if (this.rollbackHistory.length > 0) {
      report += '## 📈 Recent Rollbacks\n\n';
      const recent = this.rollbackHistory.slice(-5);
      
      for (const execution of recent) {
        const trigger = this.triggers.get(execution.trigger_id);
        const status = execution.success ? '✅ SUCCESS' : '❌ FAILED';
        
        report += `### ${trigger?.name || execution.trigger_id}\n`;
        report += `- **Status**: ${status}\n`;
        report += `- **Recovery Time**: ${execution.recovery_time_ms}ms\n`;
        report += `- **Steps Executed**: ${execution.steps_executed.length}\n`;
        report += `- **Success Rate**: ${(execution.steps_executed.filter(s => s.success).length / execution.steps_executed.length * 100).toFixed(1)}%\n\n`;
      }
    }

    if (this.currentSnapshot) {
      report += '## 📊 Current System State\n\n';
      report += `- **Stage-C RAPTOR**: ${this.currentSnapshot.configuration.stage_c_raptor ? 'Enabled' : 'Disabled'}\n`;
      report += `- **Topic Fanout**: ${this.currentSnapshot.configuration.stage_a_topic_prior ? 'Enabled' : 'Disabled'}\n`;
      report += `- **NL Bridge**: ${this.currentSnapshot.configuration.nl_bridge ? 'Enabled' : 'Disabled'}\n`;
      report += `- **p95 Latency**: ${this.currentSnapshot.performance_baseline.p95_latency}ms\n`;
      report += `- **Error Rate**: ${(this.currentSnapshot.performance_baseline.error_rate * 100).toFixed(2)}%\n`;
    }

    return report;
  }
}

// Factory function
export function createKillSwitchManager(): KillSwitchRollbackManager {
  return new KillSwitchRollbackManager();
}

// CLI demo
if (import.meta.main) {
  console.log('🚨 Kill-Switch & Auto-Rollback Demo\n');
  
  const killSwitch = createKillSwitchManager();
  
  // Event listeners
  killSwitch.on('rollback_completed', (execution: RollbackExecution) => {
    console.log(`📊 Rollback completed: ${execution.trigger_id} (${execution.success ? 'SUCCESS' : 'FAILED'})`);
  });

  killSwitch.on('rollback_confirmation_required', (data) => {
    console.log(`⚠️ Manual confirmation required: ${data.trigger_name}`);
  });

  // Demo scenario: p99 latency spike
  const demoScenario = async () => {
    console.log('📊 Taking system snapshot...\n');
    await killSwitch.takeSystemSnapshot();

    console.log('🚨 Simulating p99 latency spike...\n');
    const criticalMetrics = {
      p95_latency: 150,
      p99_latency: 350, // 2.3x p95 - triggers kill-switch
      qps: 800,
      error_rate: 0.02,
      recall_50_sla: 0.68,
      sentinel_nzc: 0.995
    };

    await killSwitch.evaluateKillSwitches(criticalMetrics);

    setTimeout(() => {
      console.log('\n📊 Status Report:');
      console.log(killSwitch.generateStatusReport());
      killSwitch.stop();
    }, 3000);
  };

  demoScenario().catch(console.error);
}