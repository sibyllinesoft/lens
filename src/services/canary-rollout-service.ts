/**
 * Canary rollout service for tail-taming features
 * Implements Section 2 of TODO.md: 5%→25%→50%→100% traffic with auto-revert
 */

import crypto from 'crypto';
import { TailTamingConfig } from '../config/tail-taming-config';

export interface CanaryStageMetrics {
  stage: number;
  traffic_percentage: number;
  start_time: number;
  end_time?: number;
  gate_checks: GateCheckResult[];
  promotion_decision: 'continue' | 'revert' | 'pending';
  promotion_reason?: string;
}

export interface GateCheckResult {
  timestamp: number;
  gates: {
    p99_latency_improvement: number;
    p99_p95_ratio: number;
    sla_recall_at_50_delta: number;
    qps_at_150ms_improvement: number;
    cost_increase: number;
  };
  all_gates_pass: boolean;
  failing_gates: string[];
  consecutive_failures: number;
}

export interface RepositoryBucket {
  repo_id: string;
  bucket: number;
  assigned_treatment: 'control' | 'test';
}

export class CanaryRolloutService {
  private readonly config: TailTamingConfig;
  private currentStage: number = 0;
  private stageStartTime: number = 0;
  private consecutiveFailures: number = 0;
  private rolloutHistory: CanaryStageMetrics[] = [];
  private repositoryBuckets = new Map<string, RepositoryBucket>();

  constructor(config: TailTamingConfig) {
    this.config = config;
  }

  /**
   * Start canary rollout from the beginning
   */
  startRollout(): void {
    console.log('Starting canary rollout for tail-taming features');
    this.currentStage = 0;
    this.stageStartTime = Date.now();
    this.consecutiveFailures = 0;
    this.rolloutHistory = [];
    
    this.logStageStart();
  }

  /**
   * Determine if a repository should receive the tail-taming treatment
   */
  shouldApplyTreatment(repositoryId: string): boolean {
    if (this.currentStage >= this.config.canary.stages.length) {
      return true; // Full rollout complete
    }

    const bucket = this.getRepositoryBucket(repositoryId);
    const currentTrafficPercent = this.config.canary.stages[this.currentStage];
    
    // Assign treatment based on bucket and current stage
    const treatmentThreshold = currentTrafficPercent / 100;
    const bucketRatio = bucket.bucket / 100; // Convert to 0-1 range
    
    const shouldTreat = bucketRatio < treatmentThreshold;
    
    // Update bucket assignment
    bucket.assigned_treatment = shouldTreat ? 'test' : 'control';
    this.repositoryBuckets.set(repositoryId, bucket);
    
    return shouldTreat;
  }

  /**
   * Perform gate check and determine rollout progression
   */
  async performGateCheck(metrics: {
    control_metrics: PerformanceMetrics;
    test_metrics: PerformanceMetrics;
  }): Promise<{
    decision: 'continue' | 'revert' | 'promote';
    reason: string;
    gate_results: GateCheckResult;
  }> {
    const gateResult = this.evaluateGates(metrics.control_metrics, metrics.test_metrics);
    
    // Update current stage history
    const currentStageMetrics = this.getCurrentStageMetrics();
    currentStageMetrics.gate_checks.push(gateResult);
    
    if (!gateResult.all_gates_pass) {
      this.consecutiveFailures = gateResult.consecutive_failures;
      
      if (this.consecutiveFailures >= this.config.canary.consecutive_failures_to_revert) {
        return this.triggerRevert(gateResult);
      }
      
      return {
        decision: 'continue',
        reason: `Gate failures: ${gateResult.failing_gates.join(', ')} (${this.consecutiveFailures}/${this.config.canary.consecutive_failures_to_revert})`,
        gate_results: gateResult
      };
    }

    // Reset consecutive failures on success
    this.consecutiveFailures = 0;
    
    // Check if stage duration is complete
    const stageElapsed = Date.now() - this.stageStartTime;
    const stageDuration = this.config.canary.stage_duration_minutes * 60 * 1000;
    
    if (stageElapsed >= stageDuration) {
      return this.promoteToNextStage(gateResult);
    }
    
    return {
      decision: 'continue',
      reason: `Stage ${this.currentStage} continuing - gates pass, time remaining: ${Math.ceil((stageDuration - stageElapsed) / (60 * 1000))} minutes`,
      gate_results: gateResult
    };
  }

  private evaluateGates(controlMetrics: PerformanceMetrics, testMetrics: PerformanceMetrics): GateCheckResult {
    const gates = this.config.gates;
    const gateResults = {
      p99_latency_improvement: this.calculateImprovement(controlMetrics.p99_latency_ms, testMetrics.p99_latency_ms),
      p99_p95_ratio: testMetrics.p99_latency_ms / testMetrics.p95_latency_ms,
      sla_recall_at_50_delta: testMetrics.sla_recall_at_50 - controlMetrics.sla_recall_at_50,
      qps_at_150ms_improvement: this.calculateImprovement(controlMetrics.qps_at_150ms, testMetrics.qps_at_150ms),
      cost_increase: this.calculateImprovement(controlMetrics.cost_per_query, testMetrics.cost_per_query)
    };

    const failingGates: string[] = [];
    
    // Check each gate
    if (gateResults.p99_latency_improvement < gates.p99_latency_improvement_max || 
        gateResults.p99_latency_improvement > gates.p99_latency_improvement_min) {
      failingGates.push('p99_latency_improvement');
    }
    
    if (gateResults.p99_p95_ratio > gates.p99_p95_ratio_max) {
      failingGates.push('p99_p95_ratio');
    }
    
    if (gateResults.sla_recall_at_50_delta < gates.sla_recall_at_50_delta_min) {
      failingGates.push('sla_recall_at_50');
    }
    
    if (gateResults.qps_at_150ms_improvement < gates.qps_at_150ms_improvement_min || 
        gateResults.qps_at_150ms_improvement > gates.qps_at_150ms_improvement_max) {
      failingGates.push('qps_at_150ms_improvement');
    }
    
    if (gateResults.cost_increase > gates.cost_increase_max) {
      failingGates.push('cost_increase');
    }

    const allGatesPass = failingGates.length === 0;
    const consecutiveFailures = allGatesPass ? 0 : this.consecutiveFailures + 1;

    return {
      timestamp: Date.now(),
      gates: gateResults,
      all_gates_pass: allGatesPass,
      failing_gates: failingGates,
      consecutive_failures: consecutiveFailures
    };
  }

  private promoteToNextStage(gateResult: GateCheckResult): {
    decision: 'continue' | 'promote';
    reason: string;
    gate_results: GateCheckResult;
  } {
    // Complete current stage
    const currentStageMetrics = this.getCurrentStageMetrics();
    currentStageMetrics.end_time = Date.now();
    currentStageMetrics.promotion_decision = 'continue';
    currentStageMetrics.promotion_reason = 'Gates passed, promoting to next stage';

    // Check if this was the final stage
    if (this.currentStage >= this.config.canary.stages.length - 1) {
      console.log('Canary rollout complete - 100% traffic achieved with green gates');
      return {
        decision: 'promote',
        reason: 'Canary rollout complete - all stages passed',
        gate_results: gateResult
      };
    }

    // Move to next stage
    this.currentStage++;
    this.stageStartTime = Date.now();
    
    this.logStageStart();

    return {
      decision: 'promote',
      reason: `Promoted to stage ${this.currentStage} (${this.config.canary.stages[this.currentStage]}% traffic)`,
      gate_results: gateResult
    };
  }

  private triggerRevert(gateResult: GateCheckResult): {
    decision: 'revert';
    reason: string;
    gate_results: GateCheckResult;
  } {
    console.error(`Canary rollout reverting due to gate failures: ${gateResult.failing_gates.join(', ')}`);
    
    // Mark current stage as failed
    const currentStageMetrics = this.getCurrentStageMetrics();
    currentStageMetrics.end_time = Date.now();
    currentStageMetrics.promotion_decision = 'revert';
    currentStageMetrics.promotion_reason = `Auto-revert triggered: ${gateResult.failing_gates.join(', ')}`;

    // Reset to 0% traffic (control only)
    this.currentStage = -1; // Special state indicating revert
    
    return {
      decision: 'revert',
      reason: `Auto-revert triggered after ${this.consecutiveFailures} consecutive gate failures: ${gateResult.failing_gates.join(', ')}`,
      gate_results: gateResult
    };
  }

  private getRepositoryBucket(repositoryId: string): RepositoryBucket {
    if (this.repositoryBuckets.has(repositoryId)) {
      return this.repositoryBuckets.get(repositoryId)!;
    }

    // Create deterministic bucket based on repository ID
    let bucket: number;
    
    if (this.config.canary.repo_bucket_strategy === 'hash') {
      const hash = crypto.createHash('sha256').update(repositoryId).digest('hex');
      const hashInt = parseInt(hash.substring(0, 8), 16);
      bucket = hashInt % 100; // 0-99 bucket
    } else {
      // Simple mod strategy
      bucket = repositoryId.length % 100;
    }

    const repoBucket: RepositoryBucket = {
      repo_id: repositoryId,
      bucket,
      assigned_treatment: 'control' // Default to control
    };

    this.repositoryBuckets.set(repositoryId, repoBucket);
    return repoBucket;
  }

  private getCurrentStageMetrics(): CanaryStageMetrics {
    if (this.rolloutHistory.length === 0 || 
        this.rolloutHistory[this.rolloutHistory.length - 1].end_time !== undefined) {
      // Create new stage metrics
      const stageMetrics: CanaryStageMetrics = {
        stage: this.currentStage,
        traffic_percentage: this.currentStage >= 0 ? this.config.canary.stages[this.currentStage] : 0,
        start_time: this.stageStartTime,
        gate_checks: [],
        promotion_decision: 'pending'
      };
      
      this.rolloutHistory.push(stageMetrics);
    }
    
    return this.rolloutHistory[this.rolloutHistory.length - 1];
  }

  private calculateImprovement(baseline: number, treatment: number): number {
    if (baseline === 0) return 0;
    return (treatment - baseline) / baseline;
  }

  private logStageStart(): void {
    const trafficPercent = this.currentStage >= 0 ? this.config.canary.stages[this.currentStage] : 0;
    console.log(`Canary stage ${this.currentStage} started: ${trafficPercent}% traffic to tail-taming treatment`);
  }

  /**
   * Get current rollout status for monitoring
   */
  getRolloutStatus(): {
    current_stage: number;
    current_traffic_percentage: number;
    stage_elapsed_minutes: number;
    consecutive_failures: number;
    is_reverted: boolean;
    history: CanaryStageMetrics[];
  } {
    const stageElapsed = (Date.now() - this.stageStartTime) / (60 * 1000);
    
    return {
      current_stage: this.currentStage,
      current_traffic_percentage: this.currentStage >= 0 ? this.config.canary.stages[this.currentStage] : 0,
      stage_elapsed_minutes: stageElapsed,
      consecutive_failures: this.consecutiveFailures,
      is_reverted: this.currentStage === -1,
      history: this.rolloutHistory
    };
  }
}

// Supporting interfaces
interface PerformanceMetrics {
  p99_latency_ms: number;
  p95_latency_ms: number;
  sla_recall_at_50: number;
  qps_at_150ms: number;
  cost_per_query: number;
}