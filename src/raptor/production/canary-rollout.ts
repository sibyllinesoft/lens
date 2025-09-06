/**
 * Canary Rollout Infrastructure
 * 
 * Implements: "Canary 5%→25%→100% with kill order: stageC.raptor → stageA.topic_prior → NL_bridge"
 * Addresses: Progressive rollout with automated promotion/rollback based on gates
 */

import { EventEmitter } from 'events';
import { writeFile, readFile, mkdir, stat } from 'fs/promises';
import { join } from 'path';

export interface CanaryStage {
  name: string;
  traffic_percentage: number;
  duration_minutes: number;
  promotion_gates: PromotionGate[];
  rollback_triggers: RollbackTrigger[];
  feature_config: FeatureConfig;
}

export interface PromotionGate {
  metric: string;
  target: number;
  operator: '>=' | '<=' | '>' | '<';
  confidence_required: number;
  sample_size_min: number;
}

export interface RollbackTrigger {
  metric: string;
  threshold: number;
  operator: '>' | '<';
  immediate: boolean;
  description: string;
}

export interface FeatureConfig {
  stage_c_raptor: boolean;
  stage_a_topic_prior: boolean;
  nl_bridge: boolean;
  raptor_weight: number;
  topic_fanout_k: number;
  span_cap: number;
}

export interface CanaryMetrics {
  timestamp: Date;
  stage: string;
  traffic_percentage: number;
  success_rate: number;
  p95_latency: number;
  p99_latency: number;
  error_rate: number;
  timeout_rate: number;
  nzc_rate: number;
  ndcg_10: number;
  p_at_1: number;
  recall_50_sla: number;
}

export interface RolloutStatus {
  current_stage: string;
  stage_start_time: Date;
  traffic_percentage: number;
  gates_status: Map<string, boolean>;
  metrics_health: 'healthy' | 'degraded' | 'critical';
  next_action: 'wait' | 'promote' | 'rollback' | 'complete';
  time_until_next_action: number;
  rollout_progress: number;
}

export const DEFAULT_CANARY_CONFIG: CanaryStage[] = [
  {
    name: 'canary_5',
    traffic_percentage: 5,
    duration_minutes: 30,
    promotion_gates: [
      { metric: 'error_rate', target: 0.01, operator: '<=', confidence_required: 0.95, sample_size_min: 100 },
      { metric: 'p95_latency', target: 160, operator: '<=', confidence_required: 0.90, sample_size_min: 100 },
      { metric: 'nzc_rate', target: 0.99, operator: '>=', confidence_required: 0.95, sample_size_min: 50 }
    ],
    rollback_triggers: [
      { metric: 'error_rate', threshold: 0.05, operator: '>', immediate: true, description: 'High error rate' },
      { metric: 'p99_latency', threshold: 500, operator: '>', immediate: true, description: 'Tail latency spike' }
    ],
    feature_config: {
      stage_c_raptor: true,
      stage_a_topic_prior: false,
      nl_bridge: false,
      raptor_weight: 0.2,
      topic_fanout_k: 0,
      span_cap: 1
    }
  },
  {
    name: 'canary_25',
    traffic_percentage: 25,
    duration_minutes: 60,
    promotion_gates: [
      { metric: 'ndcg_10', target: 3.0, operator: '>=', confidence_required: 0.90, sample_size_min: 200 },
      { metric: 'p_at_1', target: 2.0, operator: '>=', confidence_required: 0.85, sample_size_min: 200 },
      { metric: 'p95_latency', target: 155, operator: '<=', confidence_required: 0.90, sample_size_min: 500 },
      { metric: 'recall_50_sla', target: 0.0, operator: '>=', confidence_required: 0.85, sample_size_min: 300 }
    ],
    rollback_triggers: [
      { metric: 'error_rate', threshold: 0.03, operator: '>', immediate: true, description: 'Error rate degradation' },
      { metric: 'success_rate', threshold: 0.90, operator: '<', immediate: false, description: 'Success rate drop' }
    ],
    feature_config: {
      stage_c_raptor: true,
      stage_a_topic_prior: true,
      nl_bridge: false,
      raptor_weight: 0.35,
      topic_fanout_k: 160,
      span_cap: 4
    }
  },
  {
    name: 'full_rollout',
    traffic_percentage: 100,
    duration_minutes: 0, // Permanent
    promotion_gates: [
      { metric: 'ndcg_10', target: 3.5, operator: '>=', confidence_required: 0.95, sample_size_min: 1000 },
      { metric: 'p_at_1', target: 5.0, operator: '>=', confidence_required: 0.95, sample_size_min: 1000 },
      { metric: 'qps_150ms', target: 1.2, operator: '>=', confidence_required: 0.90, sample_size_min: 500 }
    ],
    rollback_triggers: [
      { metric: 'error_rate', threshold: 0.02, operator: '>', immediate: false, description: 'Sustained error increase' },
      { metric: 'p95_latency', threshold: 180, operator: '>', immediate: false, description: 'SLA violation' }
    ],
    feature_config: {
      stage_c_raptor: true,
      stage_a_topic_prior: true,
      nl_bridge: true,
      raptor_weight: 0.4,
      topic_fanout_k: 320,
      span_cap: 8
    }
  }
];

export class CanaryRolloutManager extends EventEmitter {
  private stages: CanaryStage[];
  private currentStageIndex: number;
  private stageStartTime: Date;
  private metricsHistory: CanaryMetrics[];
  private rolloutActive: boolean;
  private monitoringInterval?: NodeJS.Timeout;

  constructor(stages: CanaryStage[] = DEFAULT_CANARY_CONFIG) {
    super();
    this.stages = stages;
    this.currentStageIndex = -1; // Not started
    this.stageStartTime = new Date();
    this.metricsHistory = [];
    this.rolloutActive = false;
  }

  /**
   * Start canary rollout
   */
  async startRollout(): Promise<void> {
    if (this.rolloutActive) {
      throw new Error('Rollout already active');
    }

    console.log('🚀 Starting canary rollout...');
    
    this.rolloutActive = true;
    this.currentStageIndex = 0;
    this.stageStartTime = new Date();
    
    // Apply initial canary configuration
    await this.applyStageConfiguration(this.stages[0]);
    
    // Start monitoring
    this.startMonitoring();
    
    // Log rollout start
    const stage = this.stages[0];
    console.log(`✓ Rollout started - Stage: ${stage.name} (${stage.traffic_percentage}%)`);
    console.log(`  Duration: ${stage.duration_minutes} minutes`);
    console.log(`  Gates: ${stage.promotion_gates.length} promotion gates`);
    
    this.emit('rollout_started', {
      stage: stage.name,
      traffic_percentage: stage.traffic_percentage
    });
  }

  /**
   * Stop and rollback rollout
   */
  async stopRollout(reason: string = 'Manual stop'): Promise<void> {
    console.log(`🛑 Stopping rollout: ${reason}`);
    
    this.rolloutActive = false;
    this.stopMonitoring();
    
    // Execute kill order rollback
    await this.executeKillOrderRollback();
    
    console.log('✓ Rollout stopped and rolled back');
    
    this.emit('rollout_stopped', {
      reason,
      stage: this.getCurrentStage()?.name,
      traffic_percentage: this.getCurrentStage()?.traffic_percentage || 0
    });
  }

  /**
   * Execute kill order rollback: stageC.raptor → stageA.topic_prior → NL_bridge
   */
  private async executeKillOrderRollback(): Promise<void> {
    console.log('🔄 Executing kill order rollback...');
    
    const killOrder = [
      { feature: 'stage_c_raptor', description: 'Stage-C RAPTOR features' },
      { feature: 'stage_a_topic_prior', description: 'Stage-A topic prior' },
      { feature: 'nl_bridge', description: 'NL→symbol bridge' }
    ];

    for (const step of killOrder) {
      console.log(`  ⏪ Rolling back: ${step.description}`);
      await this.disableFeature(step.feature);
      
      // Wait briefly between rollback steps
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
    
    console.log('✓ Kill order rollback complete');
  }

  /**
   * Ingest metrics for rollout monitoring
   */
  ingestMetrics(metrics: CanaryMetrics): void {
    this.metricsHistory.push(metrics);
    
    // Keep only last 1000 metrics
    if (this.metricsHistory.length > 1000) {
      this.metricsHistory.shift();
    }

    this.emit('metrics_received', metrics);
    
    // Check if we should take action
    if (this.rolloutActive) {
      setTimeout(() => this.evaluateRolloutStatus(), 1000);
    }
  }

  /**
   * Start monitoring loop
   */
  private startMonitoring(): void {
    this.monitoringInterval = setInterval(async () => {
      if (this.rolloutActive) {
        await this.evaluateRolloutStatus();
      }
    }, 60000); // Check every minute
  }

  /**
   * Stop monitoring
   */
  private stopMonitoring(): void {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = undefined;
    }
  }

  /**
   * Evaluate current rollout status and take action
   */
  private async evaluateRolloutStatus(): Promise<void> {
    const currentStage = this.getCurrentStage();
    if (!currentStage) return;

    const status = this.getRolloutStatus();
    const recentMetrics = this.getRecentMetrics(300); // Last 5 minutes

    // Check rollback triggers first
    const rollbackNeeded = this.checkRollbackTriggers(currentStage, recentMetrics);
    if (rollbackNeeded.triggered) {
      await this.handleRollback(rollbackNeeded.reason);
      return;
    }

    // Check if stage duration has elapsed
    const stageElapsed = Date.now() - this.stageStartTime.getTime();
    const stageDurationMs = currentStage.duration_minutes * 60 * 1000;

    if (stageElapsed >= stageDurationMs && currentStage.duration_minutes > 0) {
      // Check promotion gates
      const promotionReady = this.checkPromotionGates(currentStage, recentMetrics);
      
      if (promotionReady.ready) {
        await this.promoteToNextStage();
      } else {
        console.log(`⏳ Stage ${currentStage.name} duration elapsed but gates not met`);
        console.log(`   Failed gates: ${promotionReady.failedGates.join(', ')}`);
        
        // Could implement wait or rollback logic here
        if (promotionReady.failedGates.length > 2) {
          await this.handleRollback('Multiple promotion gates failed');
        }
      }
    }
  }

  /**
   * Check rollback triggers
   */
  private checkRollbackTriggers(
    stage: CanaryStage, 
    metrics: CanaryMetrics[]
  ): { triggered: boolean; reason?: string } {
    if (metrics.length === 0) return { triggered: false };

    const latestMetric = metrics[metrics.length - 1];

    for (const trigger of stage.rollback_triggers) {
      const value = latestMetric[trigger.metric as keyof CanaryMetrics] as number;
      
      let triggered = false;
      if (trigger.operator === '>' && value > trigger.threshold) {
        triggered = true;
      } else if (trigger.operator === '<' && value < trigger.threshold) {
        triggered = true;
      }

      if (triggered) {
        return {
          triggered: true,
          reason: `${trigger.description}: ${trigger.metric} = ${value} ${trigger.operator} ${trigger.threshold}`
        };
      }
    }

    return { triggered: false };
  }

  /**
   * Check promotion gates
   */
  private checkPromotionGates(
    stage: CanaryStage,
    metrics: CanaryMetrics[]
  ): { ready: boolean; failedGates: string[] } {
    const failedGates: string[] = [];

    if (metrics.length < 10) {
      return { ready: false, failedGates: ['insufficient_data'] };
    }

    for (const gate of stage.promotion_gates) {
      // Get values for this metric
      const values = metrics.map(m => m[gate.metric as keyof CanaryMetrics] as number)
        .filter(v => v !== undefined && !isNaN(v));

      if (values.length < gate.sample_size_min) {
        failedGates.push(`${gate.metric}_sample_size`);
        continue;
      }

      const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
      
      let gatePassed = false;
      switch (gate.operator) {
        case '>=':
          gatePassed = mean >= gate.target;
          break;
        case '<=':
          gatePassed = mean <= gate.target;
          break;
        case '>':
          gatePassed = mean > gate.target;
          break;
        case '<':
          gatePassed = mean < gate.target;
          break;
      }

      if (!gatePassed) {
        failedGates.push(`${gate.metric}_target`);
      }
    }

    return {
      ready: failedGates.length === 0,
      failedGates
    };
  }

  /**
   * Promote to next stage
   */
  private async promoteToNextStage(): Promise<void> {
    if (this.currentStageIndex >= this.stages.length - 1) {
      // Rollout complete
      await this.completeRollout();
      return;
    }

    const nextStageIndex = this.currentStageIndex + 1;
    const nextStage = this.stages[nextStageIndex];

    console.log(`📈 Promoting to ${nextStage.name} (${nextStage.traffic_percentage}%)`);

    this.currentStageIndex = nextStageIndex;
    this.stageStartTime = new Date();

    // Apply new stage configuration
    await this.applyStageConfiguration(nextStage);

    console.log(`✓ Promotion complete - now at ${nextStage.traffic_percentage}% traffic`);

    this.emit('stage_promoted', {
      stage: nextStage.name,
      traffic_percentage: nextStage.traffic_percentage
    });
  }

  /**
   * Complete rollout
   */
  private async completeRollout(): Promise<void> {
    console.log('🎉 Canary rollout completed successfully!');
    
    this.rolloutActive = false;
    this.stopMonitoring();

    const finalStage = this.getCurrentStage()!;
    
    console.log(`✓ Final configuration applied: ${finalStage.traffic_percentage}% traffic`);
    console.log('✓ All production gates passed');
    console.log('✓ RAPTOR system fully deployed');

    this.emit('rollout_completed', {
      final_stage: finalStage.name,
      traffic_percentage: finalStage.traffic_percentage
    });
  }

  /**
   * Handle rollback
   */
  private async handleRollback(reason: string): Promise<void> {
    console.log(`🔄 Rollback triggered: ${reason}`);
    
    await this.stopRollout(reason);
    
    this.emit('rollback_triggered', {
      reason,
      stage: this.getCurrentStage()?.name,
      traffic_percentage: this.getCurrentStage()?.traffic_percentage || 0
    });
  }

  /**
   * Apply stage configuration
   */
  private async applyStageConfiguration(stage: CanaryStage): Promise<void> {
    console.log(`🔧 Applying configuration for ${stage.name}`);
    
    const config = stage.feature_config;
    
    // Apply feature flags
    await this.setFeatureFlag('stage_c_raptor', config.stage_c_raptor);
    await this.setFeatureFlag('stage_a_topic_prior', config.stage_a_topic_prior);
    await this.setFeatureFlag('nl_bridge', config.nl_bridge);
    
    // Apply weights and limits
    await this.setRaptorWeight(config.raptor_weight);
    await this.setTopicFanoutK(config.topic_fanout_k);
    await this.setSpanCap(config.span_cap);
    
    // Apply traffic percentage
    await this.setTrafficPercentage(stage.traffic_percentage);
    
    console.log(`✓ Configuration applied:`);
    console.log(`   Stage-C RAPTOR: ${config.stage_c_raptor}`);
    console.log(`   Stage-A Topic Prior: ${config.stage_a_topic_prior}`);
    console.log(`   NL Bridge: ${config.nl_bridge}`);
    console.log(`   RAPTOR Weight: ${config.raptor_weight}`);
    console.log(`   Topic Fanout K: ${config.topic_fanout_k}`);
    console.log(`   Traffic: ${stage.traffic_percentage}%`);
  }

  // Configuration application methods (would interface with actual system)
  private async setFeatureFlag(feature: string, enabled: boolean): Promise<void> {
    // Implementation would set actual feature flags
    console.log(`    ${feature}: ${enabled ? 'enabled' : 'disabled'}`);
  }

  private async setRaptorWeight(weight: number): Promise<void> {
    console.log(`    raptor_weight: ${weight}`);
  }

  private async setTopicFanoutK(k: number): Promise<void> {
    console.log(`    topic_fanout_k: ${k}`);
  }

  private async setSpanCap(cap: number): Promise<void> {
    console.log(`    span_cap: ${cap}`);
  }

  private async setTrafficPercentage(percentage: number): Promise<void> {
    console.log(`    traffic: ${percentage}%`);
  }

  private async disableFeature(feature: string): Promise<void> {
    console.log(`    disabling: ${feature}`);
  }

  /**
   * Get current stage
   */
  private getCurrentStage(): CanaryStage | undefined {
    return this.currentStageIndex >= 0 ? this.stages[this.currentStageIndex] : undefined;
  }

  /**
   * Get recent metrics
   */
  private getRecentMetrics(seconds: number): CanaryMetrics[] {
    const cutoff = new Date();
    cutoff.setSeconds(cutoff.getSeconds() - seconds);
    
    return this.metricsHistory.filter(m => m.timestamp >= cutoff);
  }

  /**
   * Get rollout status
   */
  getRolloutStatus(): RolloutStatus {
    const currentStage = this.getCurrentStage();
    
    if (!currentStage || !this.rolloutActive) {
      return {
        current_stage: 'not_active',
        stage_start_time: new Date(),
        traffic_percentage: 0,
        gates_status: new Map(),
        metrics_health: 'healthy',
        next_action: 'wait',
        time_until_next_action: 0,
        rollout_progress: 0
      };
    }

    const stageElapsed = Date.now() - this.stageStartTime.getTime();
    const stageDurationMs = currentStage.duration_minutes * 60 * 1000;
    const timeUntilNext = Math.max(0, stageDurationMs - stageElapsed);

    const recentMetrics = this.getRecentMetrics(300);
    const promotionStatus = this.checkPromotionGates(currentStage, recentMetrics);
    const rollbackStatus = this.checkRollbackTriggers(currentStage, recentMetrics);

    // Calculate gates status
    const gatesStatus = new Map<string, boolean>();
    for (const gate of currentStage.promotion_gates) {
      const values = recentMetrics.map(m => m[gate.metric as keyof CanaryMetrics] as number)
        .filter(v => v !== undefined && !isNaN(v));
      
      if (values.length > 0) {
        const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
        let passed = false;
        switch (gate.operator) {
          case '>=': passed = mean >= gate.target; break;
          case '<=': passed = mean <= gate.target; break;
          case '>': passed = mean > gate.target; break;
          case '<': passed = mean < gate.target; break;
        }
        gatesStatus.set(gate.metric, passed);
      }
    }

    // Determine next action
    let nextAction: 'wait' | 'promote' | 'rollback' | 'complete' = 'wait';
    if (rollbackStatus.triggered) {
      nextAction = 'rollback';
    } else if (this.currentStageIndex >= this.stages.length - 1) {
      nextAction = 'complete';
    } else if (timeUntilNext <= 0 && promotionStatus.ready) {
      nextAction = 'promote';
    }

    // Calculate progress
    const progress = Math.min(1.0, (this.currentStageIndex + 1) / this.stages.length);

    return {
      current_stage: currentStage.name,
      stage_start_time: this.stageStartTime,
      traffic_percentage: currentStage.traffic_percentage,
      gates_status: gatesStatus,
      metrics_health: this.assessMetricsHealth(recentMetrics),
      next_action: nextAction,
      time_until_next_action: Math.ceil(timeUntilNext / 1000 / 60), // minutes
      rollout_progress: progress
    };
  }

  private assessMetricsHealth(metrics: CanaryMetrics[]): 'healthy' | 'degraded' | 'critical' {
    if (metrics.length === 0) return 'healthy';
    
    const latest = metrics[metrics.length - 1];
    
    if (latest.error_rate > 0.05 || latest.p99_latency > 500) {
      return 'critical';
    } else if (latest.error_rate > 0.02 || latest.p95_latency > 180) {
      return 'degraded';
    }
    
    return 'healthy';
  }

  /**
   * Generate rollout report
   */
  generateRolloutReport(): string {
    const status = this.getRolloutStatus();
    
    let report = '# Canary Rollout Status Report\n\n';
    report += `**Current Stage**: ${status.current_stage}\n`;
    report += `**Traffic Percentage**: ${status.traffic_percentage}%\n`;
    report += `**Rollout Progress**: ${(status.rollout_progress * 100).toFixed(1)}%\n`;
    report += `**Metrics Health**: ${status.metrics_health.toUpperCase()}\n`;
    report += `**Next Action**: ${status.next_action.toUpperCase()}\n`;
    
    if (status.time_until_next_action > 0) {
      report += `**Time Until Next Action**: ${status.time_until_next_action} minutes\n`;
    }
    
    report += '\n## Promotion Gates Status\n\n';
    if (status.gates_status.size > 0) {
      report += '| Gate | Status |\n';
      report += '|------|--------|\n';
      for (const [gate, passed] of status.gates_status) {
        const status_symbol = passed ? '✅ PASS' : '❌ FAIL';
        report += `| ${gate} | ${status_symbol} |\n`;
      }
    } else {
      report += 'No active gates to evaluate.\n';
    }

    report += '\n## Recent Metrics\n\n';
    const recentMetrics = this.getRecentMetrics(300);
    if (recentMetrics.length > 0) {
      const latest = recentMetrics[recentMetrics.length - 1];
      report += `- Success Rate: ${(latest.success_rate * 100).toFixed(1)}%\n`;
      report += `- Error Rate: ${(latest.error_rate * 100).toFixed(2)}%\n`;
      report += `- p95 Latency: ${latest.p95_latency.toFixed(0)}ms\n`;
      report += `- p99 Latency: ${latest.p99_latency.toFixed(0)}ms\n`;
      report += `- nDCG@10: ${latest.ndcg_10.toFixed(3)}\n`;
      report += `- P@1: ${latest.p_at_1.toFixed(3)}\n`;
    } else {
      report += 'No recent metrics available.\n';
    }

    return report;
  }
}

// Factory function
export function createCanaryRolloutManager(stages?: CanaryStage[]): CanaryRolloutManager {
  return new CanaryRolloutManager(stages);
}

// CLI demo
if (import.meta.main) {
  console.log('🚀 Canary Rollout Infrastructure Demo\n');
  
  const rolloutManager = createCanaryRolloutManager();
  
  // Set up event listeners
  rolloutManager.on('rollout_started', (data) => {
    console.log(`📢 Rollout Started: ${data.stage} (${data.traffic_percentage}%)`);
  });
  
  rolloutManager.on('stage_promoted', (data) => {
    console.log(`📈 Stage Promoted: ${data.stage} (${data.traffic_percentage}%)`);
  });
  
  rolloutManager.on('rollback_triggered', (data) => {
    console.log(`🔄 Rollback Triggered: ${data.reason}`);
  });
  
  rolloutManager.on('rollout_completed', (data) => {
    console.log(`🎉 Rollout Completed: ${data.final_stage}`);
  });

  // Start demo rollout
  const demoRollout = async () => {
    await rolloutManager.startRollout();
    
    // Simulate metrics over time
    const simulateMetrics = (stage: number, time: number) => {
      const baseMetrics = {
        timestamp: new Date(),
        stage: `canary_${[5, 25, 100][stage] || 5}`,
        traffic_percentage: [5, 25, 100][stage] || 5,
        success_rate: 0.95 + Math.random() * 0.04,
        p95_latency: 150 + Math.random() * 20,
        p99_latency: 200 + Math.random() * 50,
        error_rate: 0.005 + Math.random() * 0.01,
        timeout_rate: 0.01 + Math.random() * 0.005,
        nzc_rate: 0.99 + Math.random() * 0.01,
        ndcg_10: 3.2 + Math.random() * 0.5,
        p_at_1: 4.8 + Math.random() * 0.8,
        recall_50_sla: 0.65 + Math.random() * 0.05
      };
      
      rolloutManager.ingestMetrics(baseMetrics);
    };

    // Simulate 5 minutes of metrics
    for (let i = 0; i < 20; i++) {
      simulateMetrics(0, i);
      await new Promise(resolve => setTimeout(resolve, 500));
    }

    console.log('\n📊 Demo Status Report:');
    console.log(rolloutManager.generateRolloutReport());
  };

  demoRollout().catch(console.error);
}