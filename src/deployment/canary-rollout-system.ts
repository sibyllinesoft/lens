/**
 * Canary Rollout System for 3-Block Deployment
 * 
 * Implements graduated rollout with automatic promotion/rollback:
 * - Block A: Early-exit optimization (5%→25%→100%)
 * - Block B: Dynamic topn with tau calibration
 * - Block C: Gentle deduplication with simhash
 * - Comprehensive gate validation and automatic abort
 * - Real-time KPI monitoring and CUSUM alarm integration
 */

import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';
import { EventEmitter } from 'events';
import { versionManager, type ConfigFingerprint } from './version-manager.js';

interface CanaryConfig {
  block_id: 'A' | 'B' | 'C';
  block_name: string;
  feature_description: string;
  traffic_stages: number[]; // [5, 25, 100]
  stage_duration_hours: number;
  promotion_gates: BlockPromotionGates;
  rollback_triggers: RollbackTrigger[];
}

interface BlockPromotionGates {
  min_ndcg_delta: number;
  min_recall_delta: number;
  max_latency_p95_increase: number;
  max_latency_p99_ratio: number;
  required_span_coverage: number;
  max_hard_negative_leakage: number;
  max_results_per_query_drift: number;
  cusum_alarm_quiet_hours: number;
}

interface RollbackTrigger {
  metric: string;
  condition: 'greater_than' | 'less_than' | 'absolute_change';
  threshold: number;
  window_minutes: number;
  priority: 'high' | 'medium' | 'low';
}

interface CanaryStatus {
  deployment_id: string;
  block_id: 'A' | 'B' | 'C';
  current_stage: number;
  traffic_percentage: number;
  stage_start_time: string;
  next_promotion_time: string;
  status: 'running' | 'promoting' | 'rolling_back' | 'completed' | 'failed';
  metrics_snapshot: MetricsSnapshot;
  gate_results: Record<string, boolean>;
  rollback_triggers_fired: string[];
}

interface MetricsSnapshot {
  timestamp: string;
  ndcg_at_10: number;
  recall_at_50: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  p99_p95_ratio: number; // p99/p95 ratio for auto-rollback trigger
  span_coverage: number;
  hard_negative_leakage: number;
  results_per_query_mean: number;
  cusum_alarms_active: string[];
  sentinel_probes_passing: boolean;
  sentinel_nzc_ratio: number; // Sentinel non-zero coverage ratio
}

interface DeploymentPlan {
  version: string;
  blocks: CanaryConfig[];
  global_abort_conditions: AbortCondition[];
  monitoring_config: MonitoringConfig;
}

interface AbortCondition {
  name: string;
  metric: string;
  threshold: number;
  applies_to: ('A' | 'B' | 'C')[];
  action: 'rollback_block' | 'rollback_all' | 'pause';
}

interface MonitoringConfig {
  metrics_collection_interval_seconds: number;
  gate_evaluation_interval_minutes: number;
  dashboard_update_interval_seconds: number;
  alert_webhooks: string[];
}

export class CanaryRolloutSystem extends EventEmitter {
  private readonly deploymentDir: string;
  private currentDeployments: Map<string, CanaryStatus> = new Map();
  private monitoringActive: boolean = false;
  private monitoringInterval?: NodeJS.Timeout;
  
  constructor(deploymentDir: string = './deployment-artifacts/canary') {
    super();
    this.deploymentDir = deploymentDir;
    
    if (!existsSync(this.deploymentDir)) {
      mkdirSync(this.deploymentDir, { recursive: true });
    }
    
    this.loadActiveDeployments();
  }
  
  /**
   * Start complete 3-block canary rollout
   */
  public async startCanaryRollout(version: string): Promise<string> {
    const config = versionManager.loadVersionConfig(version);
    const deploymentId = `canary_${version}_${Date.now()}`;
    
    console.log(`🚀 Starting 3-block canary rollout for version ${version}`);
    
    // Create deployment plan
    const deploymentPlan = this.createDeploymentPlan(version, config);
    
    // Start with Block A
    const blockAStatus = await this.startBlockDeployment(deploymentId, deploymentPlan.blocks[0], config);
    this.currentDeployments.set(deploymentId, blockAStatus);
    
    // Start monitoring
    if (!this.monitoringActive) {
      this.startMonitoring();
    }
    
    // Save deployment state
    this.saveDeploymentState(deploymentId, deploymentPlan);
    
    this.emit('canary_started', { deploymentId, version, block: 'A' });
    
    console.log(`✅ Canary rollout started: ${deploymentId}`);
    console.log(`📊 Block A deployment in progress (${blockAStatus.traffic_percentage}% traffic)`);
    
    return deploymentId;
  }
  
  /**
   * Create complete deployment plan
   */
  private createDeploymentPlan(version: string, config: ConfigFingerprint): DeploymentPlan {
    return {
      version,
      blocks: [
        {
          block_id: 'A',
          block_name: 'Early Exit Optimization',
          feature_description: `Early-exit with margin=${config.early_exit_config.margin}, min_probes=${config.early_exit_config.min_probes}`,
          traffic_stages: [5, 25, 100],
          stage_duration_hours: 24,
          promotion_gates: {
            min_ndcg_delta: 0,
            min_recall_delta: 0,
            max_latency_p95_increase: 0.10,
            max_latency_p99_ratio: 2.0,
            required_span_coverage: 1.0,
            max_hard_negative_leakage: 0.01,
            max_results_per_query_drift: 1.0,
            cusum_alarm_quiet_hours: 24
          },
          rollback_triggers: [
            { metric: 'hard_negative_leakage', condition: 'greater_than', threshold: 0.01, window_minutes: 60, priority: 'high' },
            { metric: 'results_per_query_drift', condition: 'absolute_change', threshold: 1.0, window_minutes: 30, priority: 'high' },
            { metric: 'span_coverage', condition: 'less_than', threshold: 1.0, window_minutes: 15, priority: 'high' }
          ]
        },
        {
          block_id: 'B',
          block_name: 'Dynamic TopN Calibration',
          feature_description: `Dynamic topn with tau=${config.tau_value}, reliability curve calibration`,
          traffic_stages: [5, 25, 100],
          stage_duration_hours: 24,
          promotion_gates: {
            min_ndcg_delta: 0,
            min_recall_delta: 0,
            max_latency_p95_increase: 0.10,
            max_latency_p99_ratio: 2.0,
            required_span_coverage: 1.0,
            max_hard_negative_leakage: 0.01,
            max_results_per_query_drift: 1.0,
            cusum_alarm_quiet_hours: 24
          },
          rollback_triggers: [
            { metric: 'ndcg_at_10', condition: 'less_than', threshold: config.baseline_metrics.ndcg_at_10 * 0.98, window_minutes: 60, priority: 'high' },
            { metric: 'results_per_query_mean', condition: 'absolute_change', threshold: 2.0, window_minutes: 30, priority: 'medium' }
          ]
        },
        {
          block_id: 'C',
          block_name: 'Gentle Deduplication',
          feature_description: `Simhash dedup: k=${config.dedup_params.k}, hamming_max=${config.dedup_params.hamming_max}, keep=${config.dedup_params.keep}`,
          traffic_stages: [5, 25, 100],
          stage_duration_hours: 24,
          promotion_gates: {
            min_ndcg_delta: 0,
            min_recall_delta: 0,
            max_latency_p95_increase: 0.10,
            max_latency_p99_ratio: 2.0,
            required_span_coverage: 1.0,
            max_hard_negative_leakage: 0.01,
            max_results_per_query_drift: 1.0,
            cusum_alarm_quiet_hours: 24
          },
          rollback_triggers: [
            { metric: 'recall_at_50', condition: 'less_than', threshold: config.baseline_metrics.recall_at_50 * 0.98, window_minutes: 60, priority: 'high' },
            { metric: 'results_per_query_mean', condition: 'less_than', threshold: 3.0, window_minutes: 30, priority: 'medium' }
          ]
        }
      ],
      global_abort_conditions: [
        { name: 'catastrophic_latency', metric: 'p99_latency_ms', threshold: 5000, applies_to: ['A', 'B', 'C'], action: 'rollback_all' },
        { name: 'p99_p95_ratio_violation', metric: 'p99_p95_ratio', threshold: 2.0, applies_to: ['A', 'B', 'C'], action: 'rollback_all' },
        { name: 'zero_results', metric: 'results_per_query_mean', threshold: 0.1, applies_to: ['A', 'B', 'C'], action: 'rollback_all' },
        { name: 'span_collapse', metric: 'span_coverage', threshold: 0.9, applies_to: ['A', 'B', 'C'], action: 'rollback_all' },
        { name: 'sentinel_nzc_degradation', metric: 'sentinel_nzc_ratio', threshold: 0.99, applies_to: ['A', 'B', 'C'], action: 'rollback_all' }
      ],
      monitoring_config: {
        metrics_collection_interval_seconds: 30,
        gate_evaluation_interval_minutes: 5,
        dashboard_update_interval_seconds: 10,
        alert_webhooks: []
      }
    };
  }
  
  /**
   * Start individual block deployment
   */
  private async startBlockDeployment(
    deploymentId: string,
    blockConfig: CanaryConfig,
    config: ConfigFingerprint
  ): Promise<CanaryStatus> {
    const startTime = new Date();
    const nextPromotionTime = new Date(startTime.getTime() + blockConfig.stage_duration_hours * 60 * 60 * 1000);
    
    // Initialize block with lowest traffic
    const initialTraffic = blockConfig.traffic_stages[0];
    await this.updateTrafficSplit(blockConfig.block_id, initialTraffic);
    
    // Get initial metrics
    const metricsSnapshot = await this.collectMetrics();
    
    const status: CanaryStatus = {
      deployment_id: deploymentId,
      block_id: blockConfig.block_id,
      current_stage: 0,
      traffic_percentage: initialTraffic,
      stage_start_time: startTime.toISOString(),
      next_promotion_time: nextPromotionTime.toISOString(),
      status: 'running',
      metrics_snapshot: metricsSnapshot,
      gate_results: {},
      rollback_triggers_fired: []
    };
    
    console.log(`🎯 Block ${blockConfig.block_id} started with ${initialTraffic}% traffic`);
    console.log(`⏰ Next evaluation: ${nextPromotionTime.toISOString()}`);
    
    return status;
  }
  
  /**
   * Process canary monitoring cycle
   */
  private async processMonitoringCycle(): Promise<void> {
    for (const [deploymentId, status] of this.currentDeployments) {
      if (status.status !== 'running') continue;
      
      try {
        // Collect current metrics
        const currentMetrics = await this.collectMetrics();
        
        // Check for immediate rollback triggers
        const triggersFired = await this.checkRollbackTriggers(status, currentMetrics);
        if (triggersFired.length > 0) {
          await this.executeRollback(deploymentId, status, triggersFired);
          continue;
        }
        
        // Check for stage promotion
        const shouldPromote = await this.shouldPromoteStage(status, currentMetrics);
        if (shouldPromote) {
          await this.promoteStage(deploymentId, status);
        }
        
        // Update metrics
        status.metrics_snapshot = currentMetrics;
        this.saveDeploymentStatus(deploymentId, status);
        
      } catch (error) {
        console.error(`❌ Monitoring error for ${deploymentId}:`, error);
        await this.executeEmergencyRollback(deploymentId, status, `Monitoring error: ${error}`);
      }
    }
  }
  
  /**
   * Check if stage should be promoted
   */
  private async shouldPromoteStage(status: CanaryStatus, metrics: MetricsSnapshot): Promise<boolean> {
    const stageStartTime = new Date(status.stage_start_time);
    const stageRunTime = Date.now() - stageStartTime.getTime();
    const stageRunTimeHours = stageRunTime / (60 * 60 * 1000);
    
    // Must run for minimum duration
    if (stageRunTimeHours < 24) {
      return false;
    }
    
    // Evaluate promotion gates
    const deploymentPlan = this.loadDeploymentPlan(status.deployment_id);
    const blockConfig = deploymentPlan.blocks.find(b => b.block_id === status.block_id)!;
    const gateResults = await this.evaluatePromotionGates(blockConfig.promotion_gates, metrics);
    
    status.gate_results = gateResults;
    
    // All gates must pass
    const allGatesPassed = Object.values(gateResults).every(Boolean);
    
    if (allGatesPassed) {
      console.log(`✅ Block ${status.block_id} stage ${status.current_stage} gates passed`);
      return true;
    } else {
      console.log(`⏳ Block ${status.block_id} stage ${status.current_stage} gates not yet met`);
      this.logFailedGates(gateResults);
      return false;
    }
  }
  
  /**
   * Promote to next stage or next block
   */
  private async promoteStage(deploymentId: string, status: CanaryStatus): Promise<void> {
    const deploymentPlan = this.loadDeploymentPlan(deploymentId);
    const blockConfig = deploymentPlan.blocks.find(b => b.block_id === status.block_id)!;
    
    if (status.current_stage < blockConfig.traffic_stages.length - 1) {
      // Promote to next traffic stage
      status.current_stage++;
      status.traffic_percentage = blockConfig.traffic_stages[status.current_stage];
      status.stage_start_time = new Date().toISOString();
      status.next_promotion_time = new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString();
      
      await this.updateTrafficSplit(status.block_id, status.traffic_percentage);
      
      console.log(`📈 Block ${status.block_id} promoted to ${status.traffic_percentage}% traffic`);
      this.emit('stage_promoted', { deploymentId, block: status.block_id, traffic: status.traffic_percentage });
      
    } else {
      // Block complete, move to next block
      await this.completeBlock(deploymentId, status);
    }
    
    this.saveDeploymentStatus(deploymentId, status);
  }
  
  /**
   * Complete current block and start next
   */
  private async completeBlock(deploymentId: string, status: CanaryStatus): Promise<void> {
    console.log(`🎉 Block ${status.block_id} completed successfully at 100% traffic`);
    
    const deploymentPlan = this.loadDeploymentPlan(deploymentId);
    const currentBlockIndex = deploymentPlan.blocks.findIndex(b => b.block_id === status.block_id);
    
    if (currentBlockIndex < deploymentPlan.blocks.length - 1) {
      // Start next block
      const nextBlock = deploymentPlan.blocks[currentBlockIndex + 1];
      const config = versionManager.loadVersionConfig(deploymentPlan.version);
      
      const nextBlockStatus = await this.startBlockDeployment(deploymentId, nextBlock, config);
      this.currentDeployments.set(deploymentId, nextBlockStatus);
      
      console.log(`🚀 Starting Block ${nextBlock.block_id}: ${nextBlock.block_name}`);
      this.emit('block_started', { deploymentId, block: nextBlock.block_id });
      
    } else {
      // All blocks complete
      status.status = 'completed';
      console.log(`🎊 Canary rollout completed successfully for ${deploymentId}`);
      this.emit('canary_completed', { deploymentId });
      
      // Set up production drift monitoring
      await this.setupProductionMonitoring(deploymentPlan.version);
    }
  }
  
  /**
   * Check for rollback triggers
   */
  private async checkRollbackTriggers(status: CanaryStatus, metrics: MetricsSnapshot): Promise<string[]> {
    const deploymentPlan = this.loadDeploymentPlan(status.deployment_id);
    const blockConfig = deploymentPlan.blocks.find(b => b.block_id === status.block_id)!;
    const triggeredConditions: string[] = [];
    
    // Check block-specific triggers
    for (const trigger of blockConfig.rollback_triggers) {
      if (this.evaluateTrigger(trigger, metrics)) {
        triggeredConditions.push(`${trigger.metric}: ${trigger.condition} ${trigger.threshold}`);
      }
    }
    
    // Check global abort conditions
    for (const condition of deploymentPlan.global_abort_conditions) {
      if (condition.applies_to.includes(status.block_id)) {
        const metricValue = (metrics as any)[condition.metric];
        if (metricValue !== undefined) {
          if (condition.metric === 'p99_latency_ms' && metricValue > condition.threshold) {
            triggeredConditions.push(`GLOBAL: ${condition.name}`);
          } else if (condition.metric === 'p99_p95_ratio' && metricValue > condition.threshold) {
            triggeredConditions.push(`GLOBAL: ${condition.name} (p99/p95=${metricValue.toFixed(2)})`);
          } else if (condition.metric === 'results_per_query_mean' && metricValue < condition.threshold) {
            triggeredConditions.push(`GLOBAL: ${condition.name}`);
          } else if (condition.metric === 'span_coverage' && metricValue < condition.threshold) {
            triggeredConditions.push(`GLOBAL: ${condition.name}`);
          } else if (condition.metric === 'sentinel_nzc_ratio' && metricValue < condition.threshold) {
            triggeredConditions.push(`GLOBAL: ${condition.name} (NZC=${(metricValue*100).toFixed(1)}%)`);
          }
        }
      }
    }
    
    // Check CUSUM alarms
    if (metrics.cusum_alarms_active.length > 0) {
      triggeredConditions.push(`CUSUM alarms: ${metrics.cusum_alarms_active.join(', ')}`);
    }
    
    // Check sentinel probes
    if (!metrics.sentinel_probes_passing) {
      triggeredConditions.push('Sentinel probes failing');
    }
    
    return triggeredConditions;
  }
  
  /**
   * Execute rollback for triggered conditions
   */
  private async executeRollback(deploymentId: string, status: CanaryStatus, triggers: string[]): Promise<void> {
    console.log(`🚨 Rollback triggered for Block ${status.block_id}:`);
    triggers.forEach(trigger => console.log(`  - ${trigger}`));
    
    status.status = 'rolling_back';
    status.rollback_triggers_fired = triggers;
    
    // Rollback traffic to 0%
    await this.updateTrafficSplit(status.block_id, 0);
    
    // Wait for traffic to drain
    await new Promise(resolve => setTimeout(resolve, 30000)); // 30 seconds
    
    status.status = 'failed';
    
    this.emit('block_rolled_back', {
      deploymentId,
      block: status.block_id,
      triggers,
      timestamp: new Date().toISOString()
    });
    
    console.log(`✅ Block ${status.block_id} rolled back successfully`);
  }
  
  /**
   * Execute emergency rollback for all blocks
   */
  private async executeEmergencyRollback(deploymentId: string, status: CanaryStatus, reason: string): Promise<void> {
    console.log(`🚨🚨 EMERGENCY ROLLBACK for ${deploymentId}: ${reason}`);
    
    // Immediate traffic cutoff
    await this.updateTrafficSplit('A', 0);
    await this.updateTrafficSplit('B', 0);
    await this.updateTrafficSplit('C', 0);
    
    status.status = 'failed';
    status.rollback_triggers_fired = [`EMERGENCY: ${reason}`];
    
    this.emit('emergency_rollback', {
      deploymentId,
      reason,
      timestamp: new Date().toISOString()
    });
  }
  
  /**
   * Update traffic split for block
   */
  private async updateTrafficSplit(blockId: 'A' | 'B' | 'C', percentage: number): Promise<void> {
    // Mock implementation - in production this would update load balancer/feature flags
    console.log(`🔄 Updating Block ${blockId} traffic to ${percentage}%`);
    
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    console.log(`✅ Block ${blockId} traffic updated to ${percentage}%`);
  }
  
  /**
   * Collect current metrics snapshot
   */
  private async collectMetrics(): Promise<MetricsSnapshot> {
    // Mock implementation - in production would query actual metrics APIs
    const p95 = 150 + Math.random() * 30;
    const p99 = 280 + Math.random() * 50;
    const sentinelNzc = 0.99 + Math.random() * 0.005; // Simulate high NZC with small variance
    
    return {
      timestamp: new Date().toISOString(),
      ndcg_at_10: 0.78 + Math.random() * 0.05,
      recall_at_50: 0.85 + Math.random() * 0.03,
      p95_latency_ms: p95,
      p99_latency_ms: p99,
      p99_p95_ratio: p99 / p95, // Calculate ratio for rollback trigger
      span_coverage: 0.99 + Math.random() * 0.01,
      hard_negative_leakage: Math.random() * 0.005,
      results_per_query_mean: 5.2 + Math.random() * 0.8,
      cusum_alarms_active: Math.random() > 0.9 ? ['p95_latency_degradation'] : [],
      sentinel_probes_passing: Math.random() > 0.02, // 2% failure rate
      sentinel_nzc_ratio: sentinelNzc
    };
  }
  
  /**
   * Evaluate promotion gates
   */
  private async evaluatePromotionGates(gates: BlockPromotionGates, metrics: MetricsSnapshot): Promise<Record<string, boolean>> {
    // Load baseline for comparison
    const config = versionManager.loadVersionConfig();
    const baseline = config.baseline_metrics;
    
    return {
      ndcg_gate: (metrics.ndcg_at_10 - baseline.ndcg_at_10) >= gates.min_ndcg_delta,
      recall_gate: (metrics.recall_at_50 - baseline.recall_at_50) >= gates.min_recall_delta,
      p95_latency_gate: ((metrics.p95_latency_ms - baseline.p95_latency_ms) / baseline.p95_latency_ms) <= gates.max_latency_p95_increase,
      p99_ratio_gate: (metrics.p99_latency_ms / metrics.p95_latency_ms) <= gates.max_latency_p99_ratio,
      span_coverage_gate: metrics.span_coverage >= gates.required_span_coverage,
      hard_negative_gate: metrics.hard_negative_leakage <= gates.max_hard_negative_leakage,
      results_drift_gate: Math.abs(metrics.results_per_query_mean - baseline.results_per_query_mean) <= gates.max_results_per_query_drift,
      cusum_quiet_gate: metrics.cusum_alarms_active.length === 0,
      sentinel_gate: metrics.sentinel_probes_passing
    };
  }
  
  /**
   * Evaluate individual rollback trigger
   */
  private evaluateTrigger(trigger: RollbackTrigger, metrics: MetricsSnapshot): boolean {
    const metricValue = (metrics as any)[trigger.metric];
    if (metricValue === undefined) return false;
    
    switch (trigger.condition) {
      case 'greater_than':
        return metricValue > trigger.threshold;
      case 'less_than':
        return metricValue < trigger.threshold;
      case 'absolute_change':
        // Would need baseline comparison - simplified for now
        return false;
      default:
        return false;
    }
  }
  
  /**
   * Setup production monitoring after successful rollout
   */
  private async setupProductionMonitoring(version: string): Promise<void> {
    console.log(`📊 Setting up production drift monitoring for version ${version}`);
    
    // Configure drift alarms (would integrate with actual monitoring system)
    const alarms = [
      'anchor_p_at_1_drift',
      'recall_at_50_drift', 
      'ladder_positives_in_candidates_drift',
      'lsif_coverage_drift',
      'tree_sitter_coverage_drift'
    ];
    
    console.log(`⚠️  Configured drift alarms: ${alarms.join(', ')}`);
    console.log(`🎯 Production monitoring active for version ${version}`);
  }
  
  /**
   * Start monitoring loop
   */
  private startMonitoring(): void {
    if (this.monitoringActive) return;
    
    this.monitoringActive = true;
    this.monitoringInterval = setInterval(async () => {
      await this.processMonitoringCycle();
    }, 30000); // 30 second intervals
    
    console.log('📊 Canary monitoring started');
  }
  
  /**
   * Stop monitoring loop
   */
  private stopMonitoring(): void {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = undefined;
    }
    this.monitoringActive = false;
    
    console.log('📊 Canary monitoring stopped');
  }
  
  private logFailedGates(gateResults: Record<string, boolean>): void {
    const failed = Object.entries(gateResults).filter(([_, passed]) => !passed);
    if (failed.length > 0) {
      console.log(`❌ Failed gates: ${failed.map(([gate, _]) => gate).join(', ')}`);
    }
  }
  
  private loadActiveDeployments(): void {
    // Load any persisted deployment state (simplified)
    this.currentDeployments = new Map();
  }
  
  private saveDeploymentState(deploymentId: string, plan: DeploymentPlan): void {
    const planPath = join(this.deploymentDir, `${deploymentId}_plan.json`);
    writeFileSync(planPath, JSON.stringify(plan, null, 2));
  }
  
  private saveDeploymentStatus(deploymentId: string, status: CanaryStatus): void {
    const statusPath = join(this.deploymentDir, `${deploymentId}_status.json`);
    writeFileSync(statusPath, JSON.stringify(status, null, 2));
  }
  
  private loadDeploymentPlan(deploymentId: string): DeploymentPlan {
    const planPath = join(this.deploymentDir, `${deploymentId}_plan.json`);
    return JSON.parse(readFileSync(planPath, 'utf-8'));
  }
  
  /**
   * Get current deployment status
   */
  public getDeploymentStatus(deploymentId: string): CanaryStatus | undefined {
    return this.currentDeployments.get(deploymentId);
  }
  
  /**
   * List active deployments
   */
  public getActiveDeployments(): string[] {
    return Array.from(this.currentDeployments.keys());
  }
  
  /**
   * Manual rollback of deployment
   */
  public async manualRollback(deploymentId: string, reason: string): Promise<void> {
    const status = this.currentDeployments.get(deploymentId);
    if (!status) {
      throw new Error(`Deployment ${deploymentId} not found`);
    }
    
    await this.executeRollback(deploymentId, status, [`Manual rollback: ${reason}`]);
  }
  
  /**
   * Get deployment dashboard data
   */
  public async getDashboardData(): Promise<any> {
    const deployments = Array.from(this.currentDeployments.entries()).map(([id, status]) => ({
      deploymentId: id,
      block: status.block_id,
      stage: status.current_stage,
      traffic: status.traffic_percentage,
      status: status.status,
      metrics: status.metrics_snapshot,
      gates: status.gate_results,
      nextPromotion: status.next_promotion_time
    }));
    
    return {
      timestamp: new Date().toISOString(),
      activeDeployments: deployments,
      monitoringActive: this.monitoringActive
    };
  }
}

export const canaryRolloutSystem = new CanaryRolloutSystem();