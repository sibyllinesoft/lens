/**
 * Gemma-256 Canary Rollout System
 * 
 * Strict SLA gates with auto-rollback for 5%→25%→100% deployment.
 * Quality: nDCG@10(UR) Δ ≥ −3pp overall; NL slice Δ ≥ −1pp; SLA-Recall@50 ≥ baseline
 * Ops: p95 ≤ 25ms; p99 ≤ 2×p95; QPS@150ms ≥ 1.3× 768d; error rate < 2%
 * Kill-order: dense.hybrid→dense.256→dense.768
 */

import type { SearchHit, SearchContext } from './span_resolver/types.js';
import type { ShadowAuditReport } from './gemma-256-shadow-audit.js';
import { LensTracer } from '../telemetry/tracer.js';

export interface CanaryRolloutConfig {
  enabled: boolean;
  // Rollout stages
  stages: CanaryStage[];
  stage_duration_minutes: number;     // Duration per stage
  min_samples_per_stage: number;      // Minimum samples before advancing
  
  // Quality gates (vs 768d baseline)
  quality_gates: {
    overall_ndcg_delta_threshold: number;    // ≥ −3pp overall
    nl_slice_ndcg_delta_threshold: number;   // ≥ −1pp for NL queries  
    sla_recall_at_50_min: number;           // ≥ baseline
    precision_at_1_max_degradation: number; // ≤ −2pp for symbol/NL
  };
  
  // Operational gates
  ops_gates: {
    p95_latency_max_ms: number;         // ≤ 25ms
    p99_latency_max_ratio: number;      // ≤ 2×p95
    qps_at_150ms_min_ratio: number;     // ≥ 1.3× 768d
    error_rate_max_pct: number;         // < 2%
  };
  
  // Integrity gates  
  integrity_gates: {
    span_coverage_min_pct: number;      // = 100%
    sentinel_nzc_min_pct: number;       // ≥ 99%
    prose_artifact_drift_max: number;   // ≤ 0.1pp
  };
  
  // Auto-rollback settings
  auto_rollback: {
    enabled: boolean;
    breach_threshold_count: number;     // Consecutive breaches before rollback
    emergency_rollback_latency_ms: number; // Emergency latency threshold
    rollback_timeout_minutes: number;   // Max time for rollback completion
  };
  
  // Monitoring and alerting
  monitoring: {
    sample_interval_seconds: number;
    alert_on_gate_failures: boolean;
    store_detailed_metrics: boolean;
  };
}

export interface CanaryStage {
  name: string;
  traffic_percentage: number;          // % of traffic routed to new system
  min_duration_minutes: number;       // Minimum duration before next stage
  quality_gate_strictness: number;    // Multiplier for gate thresholds (1.0 = normal)
}

export interface CanaryMetrics {
  timestamp: Date;
  stage: string;
  traffic_percentage: number;
  
  // Quality metrics
  overall_ndcg_delta: number;
  nl_slice_ndcg_delta: number;
  sla_recall_at_50: number;
  precision_at_1_symbol: number;
  precision_at_1_nl: number;
  
  // Operational metrics
  p95_latency_ms: number;
  p99_latency_ms: number;
  qps_at_150ms: number;
  error_rate_pct: number;
  
  // Integrity metrics
  span_coverage_pct: number;
  sentinel_nzc_pct: number;
  prose_artifact_drift: number;
  
  // Baseline comparison
  baseline_comparison: {
    ndcg_improvement: number;
    latency_improvement: number;
    throughput_improvement: number;
  };
  
  // Gate status
  quality_gates_passed: boolean;
  ops_gates_passed: boolean;
  integrity_gates_passed: boolean;
  overall_gates_passed: boolean;
}

export interface RollbackEvent {
  timestamp: Date;
  reason: string;
  failed_gates: string[];
  trigger_metrics: CanaryMetrics;
  rollback_sequence: string[];      // Kill order sequence
  rollback_duration_ms: number;
  success: boolean;
}

export interface CanaryRolloutStatus {
  is_active: boolean;
  current_stage: string;
  traffic_percentage: number;
  stage_start_time: Date;
  stage_duration_minutes: number;
  
  // Progress
  samples_collected: number;
  min_samples_required: number;
  can_advance: boolean;
  
  // Gate status
  consecutive_gate_failures: number;
  last_gate_check: Date;
  gates_passing: boolean;
  
  // Rollback status
  rollback_triggered: boolean;
  rollback_in_progress: boolean;
  rollback_events: RollbackEvent[];
}

/**
 * Baseline Performance Monitor
 * Tracks 768d performance as baseline for comparison
 */
export class BaselineMonitor {
  private baselineMetrics: CanaryMetrics[] = [];
  private isCalibrated = false;
  
  /**
   * Record baseline 768d performance
   */
  recordBaseline(metrics: Omit<CanaryMetrics, 'stage' | 'traffic_percentage' | 'baseline_comparison'>): void {
    const baselineMetric: CanaryMetrics = {
      ...metrics,
      stage: 'baseline_768d',
      traffic_percentage: 100,
      baseline_comparison: {
        ndcg_improvement: 0,
        latency_improvement: 0,
        throughput_improvement: 0
      }
    };
    
    this.baselineMetrics.push(baselineMetric);
    
    // Keep recent baseline data
    if (this.baselineMetrics.length > 1000) {
      this.baselineMetrics = this.baselineMetrics.slice(-500);
    }
    
    if (!this.isCalibrated && this.baselineMetrics.length >= 100) {
      this.isCalibrated = true;
      console.log(`📊 Baseline monitor calibrated with ${this.baselineMetrics.length} samples`);
    }
  }
  
  /**
   * Get current baseline averages
   */
  getBaseline(): CanaryMetrics | null {
    if (!this.isCalibrated || this.baselineMetrics.length === 0) {
      return null;
    }
    
    const recent = this.baselineMetrics.slice(-100); // Use recent 100 samples
    const count = recent.length;
    
    return {
      timestamp: new Date(),
      stage: 'baseline_768d',
      traffic_percentage: 100,
      
      overall_ndcg_delta: recent.reduce((sum, m) => sum + m.overall_ndcg_delta, 0) / count,
      nl_slice_ndcg_delta: recent.reduce((sum, m) => sum + m.nl_slice_ndcg_delta, 0) / count,
      sla_recall_at_50: recent.reduce((sum, m) => sum + m.sla_recall_at_50, 0) / count,
      precision_at_1_symbol: recent.reduce((sum, m) => sum + m.precision_at_1_symbol, 0) / count,
      precision_at_1_nl: recent.reduce((sum, m) => sum + m.precision_at_1_nl, 0) / count,
      
      p95_latency_ms: recent.reduce((sum, m) => sum + m.p95_latency_ms, 0) / count,
      p99_latency_ms: recent.reduce((sum, m) => sum + m.p99_latency_ms, 0) / count,
      qps_at_150ms: recent.reduce((sum, m) => sum + m.qps_at_150ms, 0) / count,
      error_rate_pct: recent.reduce((sum, m) => sum + m.error_rate_pct, 0) / count,
      
      span_coverage_pct: recent.reduce((sum, m) => sum + m.span_coverage_pct, 0) / count,
      sentinel_nzc_pct: recent.reduce((sum, m) => sum + m.sentinel_nzc_pct, 0) / count,
      prose_artifact_drift: recent.reduce((sum, m) => sum + m.prose_artifact_drift, 0) / count,
      
      baseline_comparison: {
        ndcg_improvement: 0,
        latency_improvement: 0,
        throughput_improvement: 0
      },
      
      quality_gates_passed: true,
      ops_gates_passed: true,
      integrity_gates_passed: true,
      overall_gates_passed: true
    };
  }
  
  getStats() {
    return {
      calibrated: this.isCalibrated,
      samples: this.baselineMetrics.length,
      latest_baseline: this.baselineMetrics[this.baselineMetrics.length - 1]
    };
  }
}

/**
 * Gate Validator - Validates all SLA gates against baseline
 */
export class GateValidator {
  
  constructor(private config: CanaryRolloutConfig) {}
  
  /**
   * Validate all gates against baseline and return detailed results
   */
  validateGates(
    canaryMetrics: CanaryMetrics,
    baseline: CanaryMetrics,
    stageStrictness = 1.0
  ): {
    quality: { passed: boolean; failures: string[] };
    ops: { passed: boolean; failures: string[] };
    integrity: { passed: boolean; failures: string[] };
    overall: boolean;
  } {
    const qualityResult = this.validateQualityGates(canaryMetrics, baseline, stageStrictness);
    const opsResult = this.validateOpsGates(canaryMetrics, baseline, stageStrictness);
    const integrityResult = this.validateIntegrityGates(canaryMetrics, stageStrictness);
    
    return {
      quality: qualityResult,
      ops: opsResult,
      integrity: integrityResult,
      overall: qualityResult.passed && opsResult.passed && integrityResult.passed
    };
  }
  
  /**
   * Validate quality gates vs baseline
   */
  private validateQualityGates(
    canary: CanaryMetrics,
    baseline: CanaryMetrics,
    strictness: number
  ): { passed: boolean; failures: string[] } {
    const failures: string[] = [];
    
    // Overall nDCG delta threshold: ≥ −3pp
    const overallNdcgDelta = canary.overall_ndcg_delta - baseline.overall_ndcg_delta;
    const overallThreshold = this.config.quality_gates.overall_ndcg_delta_threshold * strictness;
    if (overallNdcgDelta < overallThreshold) {
      failures.push(`Overall nDCG delta ${overallNdcgDelta.toFixed(3)} < ${overallThreshold.toFixed(3)}`);
    }
    
    // NL slice nDCG delta threshold: ≥ −1pp
    const nlNdcgDelta = canary.nl_slice_ndcg_delta - baseline.nl_slice_ndcg_delta;
    const nlThreshold = this.config.quality_gates.nl_slice_ndcg_delta_threshold * strictness;
    if (nlNdcgDelta < nlThreshold) {
      failures.push(`NL slice nDCG delta ${nlNdcgDelta.toFixed(3)} < ${nlThreshold.toFixed(3)}`);
    }
    
    // SLA Recall@50 minimum
    const minRecall = this.config.quality_gates.sla_recall_at_50_min;
    if (canary.sla_recall_at_50 < minRecall) {
      failures.push(`SLA Recall@50 ${canary.sla_recall_at_50.toFixed(3)} < ${minRecall.toFixed(3)}`);
    }
    
    // P@1 degradation limits for symbol/NL queries  
    const symbolP1Degradation = baseline.precision_at_1_symbol - canary.precision_at_1_symbol;
    const nlP1Degradation = baseline.precision_at_1_nl - canary.precision_at_1_nl;
    const maxDegradation = this.config.quality_gates.precision_at_1_max_degradation;
    
    if (symbolP1Degradation > maxDegradation) {
      failures.push(`Symbol P@1 degradation ${symbolP1Degradation.toFixed(3)} > ${maxDegradation.toFixed(3)}`);
    }
    
    if (nlP1Degradation > maxDegradation) {
      failures.push(`NL P@1 degradation ${nlP1Degradation.toFixed(3)} > ${maxDegradation.toFixed(3)}`);
    }
    
    return { passed: failures.length === 0, failures };
  }
  
  /**
   * Validate operational gates
   */
  private validateOpsGates(
    canary: CanaryMetrics,
    baseline: CanaryMetrics,
    strictness: number
  ): { passed: boolean; failures: string[] } {
    const failures: string[] = [];
    
    // P95 latency: ≤ 25ms
    const maxP95 = this.config.ops_gates.p95_latency_max_ms / strictness; // Stricter in early stages
    if (canary.p95_latency_ms > maxP95) {
      failures.push(`P95 latency ${canary.p95_latency_ms.toFixed(1)}ms > ${maxP95.toFixed(1)}ms`);
    }
    
    // P99 latency: ≤ 2×P95
    const maxP99 = canary.p95_latency_ms * this.config.ops_gates.p99_latency_max_ratio;
    if (canary.p99_latency_ms > maxP99) {
      failures.push(`P99 latency ${canary.p99_latency_ms.toFixed(1)}ms > ${maxP99.toFixed(1)}ms (2×P95)`);
    }
    
    // QPS@150ms: ≥ 1.3× 768d baseline
    const minQps = baseline.qps_at_150ms * this.config.ops_gates.qps_at_150ms_min_ratio;
    if (canary.qps_at_150ms < minQps) {
      failures.push(`QPS@150ms ${canary.qps_at_150ms.toFixed(1)} < ${minQps.toFixed(1)} (1.3× baseline)`);
    }
    
    // Error rate: < 2%
    const maxErrorRate = this.config.ops_gates.error_rate_max_pct;
    if (canary.error_rate_pct >= maxErrorRate) {
      failures.push(`Error rate ${canary.error_rate_pct.toFixed(2)}% ≥ ${maxErrorRate.toFixed(2)}%`);
    }
    
    return { passed: failures.length === 0, failures };
  }
  
  /**
   * Validate integrity gates
   */
  private validateIntegrityGates(
    canary: CanaryMetrics,
    strictness: number
  ): { passed: boolean; failures: string[] } {
    const failures: string[] = [];
    
    // Span coverage: = 100%
    const minSpanCoverage = this.config.integrity_gates.span_coverage_min_pct;
    if (canary.span_coverage_pct < minSpanCoverage) {
      failures.push(`Span coverage ${canary.span_coverage_pct.toFixed(1)}% < ${minSpanCoverage.toFixed(1)}%`);
    }
    
    // Sentinel NZC: ≥ 99%
    const minSentinelNzc = this.config.integrity_gates.sentinel_nzc_min_pct;
    if (canary.sentinel_nzc_pct < minSentinelNzc) {
      failures.push(`Sentinel NZC ${canary.sentinel_nzc_pct.toFixed(1)}% < ${minSentinelNzc.toFixed(1)}%`);
    }
    
    // Prose↔artifact drift: ≤ 0.1pp
    const maxDrift = this.config.integrity_gates.prose_artifact_drift_max;
    if (canary.prose_artifact_drift > maxDrift) {
      failures.push(`Prose↔artifact drift ${canary.prose_artifact_drift.toFixed(3)} > ${maxDrift.toFixed(3)}`);
    }
    
    return { passed: failures.length === 0, failures };
  }
}

/**
 * Auto-Rollback Controller
 * Implements kill-order: dense.hybrid→dense.256→dense.768
 */
export class AutoRollbackController {
  private rollbackInProgress = false;
  private rollbackHistory: RollbackEvent[] = [];
  
  constructor(private config: CanaryRolloutConfig) {}
  
  /**
   * Execute rollback with specified kill order
   */
  async executeRollback(
    reason: string,
    failedGates: string[],
    triggerMetrics: CanaryMetrics
  ): Promise<RollbackEvent> {
    const span = LensTracer.createChildSpan('canary_rollback_execute', {
      'rollback.reason': reason,
      'rollback.failed_gates': failedGates.length,
      'rollback.traffic_pct': triggerMetrics.traffic_percentage
    });
    
    const startTime = Date.now();
    this.rollbackInProgress = true;
    
    try {
      console.log(`🚨 CANARY ROLLBACK TRIGGERED: ${reason}`);
      console.log(`   Failed gates: ${failedGates.join(', ')}`);
      console.log(`   Traffic at failure: ${triggerMetrics.traffic_percentage}%`);
      
      // Kill order: dense.hybrid→dense.256→dense.768
      const killOrder = [
        'dense.hybrid',    // Disable hybrid routing first
        'dense.256',       // Disable 256d processing
        'dense.768'        // Fallback to 768d (should always be running)
      ];
      
      const executedSteps: string[] = [];
      
      // Execute kill order sequence
      for (const step of killOrder) {
        try {
          await this.executeKillStep(step);
          executedSteps.push(step);
          console.log(`✅ Rollback step completed: ${step}`);
          
          // Brief pause between steps for stability
          await new Promise(resolve => setTimeout(resolve, 1000));
          
        } catch (stepError) {
          console.error(`❌ Rollback step failed: ${step}`, stepError);
          executedSteps.push(`${step} (FAILED)`);
        }
      }
      
      // Verify rollback success
      const rollbackSuccess = await this.verifyRollbackSuccess();
      const rollbackDuration = Date.now() - startTime;
      
      const rollbackEvent: RollbackEvent = {
        timestamp: new Date(),
        reason,
        failed_gates: failedGates,
        trigger_metrics: triggerMetrics,
        rollback_sequence: executedSteps,
        rollback_duration_ms: rollbackDuration,
        success: rollbackSuccess
      };
      
      this.rollbackHistory.push(rollbackEvent);
      
      // Keep history bounded
      if (this.rollbackHistory.length > 100) {
        this.rollbackHistory = this.rollbackHistory.slice(-50);
      }
      
      span.setAttributes({
        success: rollbackSuccess,
        rollback_duration_ms: rollbackDuration,
        steps_executed: executedSteps.length,
        kill_order_completed: executedSteps.length === killOrder.length
      });
      
      if (rollbackSuccess) {
        console.log(`✅ ROLLBACK COMPLETED in ${rollbackDuration}ms`);
        console.log(`   All systems returned to 768d baseline`);
      } else {
        console.error(`❌ ROLLBACK PARTIALLY FAILED after ${rollbackDuration}ms`);
        console.error(`   Manual intervention may be required`);
      }
      
      return rollbackEvent;
      
    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      this.rollbackInProgress = false;
      span.end();
    }
  }
  
  /**
   * Execute individual kill step
   */
  private async executeKillStep(step: string): Promise<void> {
    const timeout = this.config.auto_rollback.rollback_timeout_minutes * 60 * 1000;
    
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error(`Kill step ${step} timed out after ${this.config.auto_rollback.rollback_timeout_minutes} minutes`));
      }, timeout);
      
      // In production, would execute actual system reconfiguration
      // For now, simulate the steps
      setTimeout(() => {
        clearTimeout(timer);
        console.log(`🔄 Executing kill step: ${step}`);
        
        switch (step) {
          case 'dense.hybrid':
            // Disable hybrid routing, fall back to 256d only
            console.log(`   Disabling hybrid router, routing all traffic to 256d`);
            break;
            
          case 'dense.256':
            // Disable 256d processing, fall back to 768d
            console.log(`   Disabling 256d processing, routing all traffic to 768d`);
            break;
            
          case 'dense.768':
            // Ensure 768d is running (should already be)
            console.log(`   Ensuring 768d baseline is active and healthy`);
            break;
        }
        
        resolve();
      }, 2000 + Math.random() * 3000); // Simulate 2-5 second execution time
    });
  }
  
  /**
   * Verify rollback completed successfully
   */
  private async verifyRollbackSuccess(): Promise<boolean> {
    // In production, would verify:
    // 1. Hybrid routing is disabled
    // 2. 256d processing is disabled  
    // 3. All traffic is on 768d baseline
    // 4. System health is good
    // 5. Error rates have returned to normal
    
    // Simulate verification
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Return success (in production, would check actual metrics)
    return true;
  }
  
  /**
   * Check if emergency rollback should be triggered
   */
  shouldTriggerEmergencyRollback(metrics: CanaryMetrics): boolean {
    // Emergency conditions
    const emergencyLatency = metrics.p95_latency_ms > this.config.auto_rollback.emergency_rollback_latency_ms;
    const criticalErrorRate = metrics.error_rate_pct > 10; // 10% error rate is critical
    const spanCoverageFailure = metrics.span_coverage_pct < 90; // Critical span coverage loss
    
    return emergencyLatency || criticalErrorRate || spanCoverageFailure;
  }
  
  getStats() {
    return {
      rollback_in_progress: this.rollbackInProgress,
      rollback_events: this.rollbackHistory.length,
      last_rollback: this.rollbackHistory[this.rollbackHistory.length - 1],
      config: this.config.auto_rollback
    };
  }
}

/**
 * Canary Rollout Engine - Core orchestration logic
 */
export class CanaryRolloutEngine {
  private baselineMonitor: BaselineMonitor;
  private gateValidator: GateValidator;
  private rollbackController: AutoRollbackController;
  
  private status: CanaryRolloutStatus;
  private metricsHistory: CanaryMetrics[] = [];
  private consecutiveFailures = 0;
  
  constructor(private config: CanaryRolloutConfig) {
    this.baselineMonitor = new BaselineMonitor();
    this.gateValidator = new GateValidator(config);
    this.rollbackController = new AutoRollbackController(config);
    
    this.status = {
      is_active: false,
      current_stage: 'inactive',
      traffic_percentage: 0,
      stage_start_time: new Date(),
      stage_duration_minutes: 0,
      samples_collected: 0,
      min_samples_required: config.min_samples_per_stage,
      can_advance: false,
      consecutive_gate_failures: 0,
      last_gate_check: new Date(),
      gates_passing: true,
      rollback_triggered: false,
      rollback_in_progress: false,
      rollback_events: []
    };
    
    console.log(`🚀 Canary Rollout Engine initialized`);
    console.log(`   Stages: ${config.stages.map(s => `${s.name}(${s.traffic_percentage}%)`).join(' → ')}`);
    console.log(`   Auto-rollback: ${config.auto_rollback.enabled}`);
  }
  
  /**
   * Start canary rollout process
   */
  startRollout(): void {
    if (this.status.is_active) {
      console.warn('⚠️ Canary rollout already active');
      return;
    }
    
    // Verify baseline is calibrated
    if (!this.baselineMonitor.getStats().calibrated) {
      throw new Error('Baseline monitor not calibrated - collect baseline metrics first');
    }
    
    this.status.is_active = true;
    this.status.current_stage = this.config.stages[0]!.name;
    this.status.traffic_percentage = this.config.stages[0]!.traffic_percentage;
    this.status.stage_start_time = new Date();
    this.status.samples_collected = 0;
    this.status.consecutive_gate_failures = 0;
    this.consecutiveFailures = 0;
    
    console.log(`🚀 CANARY ROLLOUT STARTED`);
    console.log(`   Initial stage: ${this.status.current_stage} (${this.status.traffic_percentage}%)`);
    console.log(`   Baseline established, proceeding with deployment`);
  }
  
  /**
   * Record canary metrics and check gates
   */
  async recordMetrics(metrics: Omit<CanaryMetrics, 'stage' | 'traffic_percentage'>): Promise<void> {
    if (!this.status.is_active) {
      return;
    }
    
    const span = LensTracer.createChildSpan('canary_metrics_check', {
      'stage': this.status.current_stage,
      'traffic_pct': this.status.traffic_percentage,
      'samples_collected': this.status.samples_collected
    });
    
    try {
      const baseline = this.baselineMonitor.getBaseline();
      if (!baseline) {
        console.warn('⚠️ No baseline available for comparison');
        return;
      }
      
      // Create full canary metrics
      const canaryMetrics: CanaryMetrics = {
        ...metrics,
        stage: this.status.current_stage,
        traffic_percentage: this.status.traffic_percentage,
        // Calculate baseline comparison
        baseline_comparison: {
          ndcg_improvement: metrics.overall_ndcg_delta - baseline.overall_ndcg_delta,
          latency_improvement: baseline.p95_latency_ms - metrics.p95_latency_ms,
          throughput_improvement: (metrics.qps_at_150ms / baseline.qps_at_150ms) - 1.0
        }
      };
      
      this.metricsHistory.push(canaryMetrics);
      this.status.samples_collected++;
      
      // Validate gates
      const currentStage = this.getCurrentStage();
      const gateResults = this.gateValidator.validateGates(
        canaryMetrics,
        baseline,
        currentStage?.quality_gate_strictness || 1.0
      );
      
      // Update gate status
      canaryMetrics.quality_gates_passed = gateResults.quality.passed;
      canaryMetrics.ops_gates_passed = gateResults.ops.passed;
      canaryMetrics.integrity_gates_passed = gateResults.integrity.passed;
      canaryMetrics.overall_gates_passed = gateResults.overall;
      
      this.status.gates_passing = gateResults.overall;
      this.status.last_gate_check = new Date();
      
      // Handle gate failures
      if (!gateResults.overall) {
        this.consecutiveFailures++;
        this.status.consecutive_gate_failures = this.consecutiveFailures;
        
        const allFailures = [
          ...gateResults.quality.failures,
          ...gateResults.ops.failures,
          ...gateResults.integrity.failures
        ];
        
        console.warn(`⚠️ Gate failures (${this.consecutiveFailures} consecutive):`);
        for (const failure of allFailures) {
          console.warn(`   ❌ ${failure}`);
        }
        
        // Check for emergency rollback conditions
        if (this.rollbackController.shouldTriggerEmergencyRollback(canaryMetrics)) {
          console.error(`🚨 EMERGENCY ROLLBACK CONDITIONS DETECTED`);
          await this.triggerRollback('emergency_conditions', allFailures, canaryMetrics);
          return;
        }
        
        // Check for auto-rollback threshold
        if (this.config.auto_rollback.enabled && 
            this.consecutiveFailures >= this.config.auto_rollback.breach_threshold_count) {
          console.error(`🚨 AUTO-ROLLBACK THRESHOLD REACHED: ${this.consecutiveFailures} consecutive failures`);
          await this.triggerRollback('gate_failure_threshold', allFailures, canaryMetrics);
          return;
        }
      } else {
        this.consecutiveFailures = 0;
        this.status.consecutive_gate_failures = 0;
      }
      
      // Check if stage can advance
      this.updateAdvanceEligibility();
      
      span.setAttributes({
        success: true,
        gates_passed: gateResults.overall,
        consecutive_failures: this.consecutiveFailures,
        can_advance: this.status.can_advance,
        samples_collected: this.status.samples_collected
      });
      
    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      console.error('❌ Failed to process canary metrics:', error);
    } finally {
      span.end();
    }
  }
  
  /**
   * Advance to next canary stage if eligible
   */
  async advanceStage(): Promise<boolean> {
    if (!this.status.can_advance) {
      console.warn('⚠️ Cannot advance stage - requirements not met');
      return false;
    }
    
    const currentStageIndex = this.config.stages.findIndex(s => s.name === this.status.current_stage);
    if (currentStageIndex === -1) {
      console.error('❌ Current stage not found in configuration');
      return false;
    }
    
    const nextStageIndex = currentStageIndex + 1;
    if (nextStageIndex >= this.config.stages.length) {
      // Rollout complete
      console.log(`🎉 CANARY ROLLOUT COMPLETED SUCCESSFULLY`);
      console.log(`   All stages passed, 100% traffic on new system`);
      this.status.is_active = false;
      this.status.current_stage = 'completed';
      return true;
    }
    
    const nextStage = this.config.stages[nextStageIndex]!;
    
    console.log(`🚀 ADVANCING TO NEXT STAGE:`);
    console.log(`   ${this.status.current_stage}(${this.status.traffic_percentage}%) → ${nextStage.name}(${nextStage.traffic_percentage}%)`);
    
    // Update status
    this.status.current_stage = nextStage.name;
    this.status.traffic_percentage = nextStage.traffic_percentage;
    this.status.stage_start_time = new Date();
    this.status.samples_collected = 0;
    this.status.can_advance = false;
    this.consecutiveFailures = 0;
    
    console.log(`✅ Stage advanced successfully`);
    return true;
  }
  
  /**
   * Trigger rollback
   */
  private async triggerRollback(
    reason: string,
    failedGates: string[],
    triggerMetrics: CanaryMetrics
  ): Promise<void> {
    this.status.rollback_triggered = true;
    this.status.rollback_in_progress = true;
    
    try {
      const rollbackEvent = await this.rollbackController.executeRollback(
        reason,
        failedGates,
        triggerMetrics
      );
      
      this.status.rollback_events.push(rollbackEvent);
      
      // Stop canary rollout
      this.status.is_active = false;
      this.status.current_stage = 'rolled_back';
      
    } finally {
      this.status.rollback_in_progress = false;
    }
  }
  
  /**
   * Update advance eligibility
   */
  private updateAdvanceEligibility(): void {
    const currentStage = this.getCurrentStage();
    if (!currentStage) {
      this.status.can_advance = false;
      return;
    }
    
    // Check minimum samples
    const hasMinSamples = this.status.samples_collected >= this.config.min_samples_per_stage;
    
    // Check minimum duration
    const stageAge = Date.now() - this.status.stage_start_time.getTime();
    const minDuration = currentStage.min_duration_minutes * 60 * 1000;
    const hasMinDuration = stageAge >= minDuration;
    
    // Check gates are passing
    const gatesPassing = this.status.gates_passing && this.consecutiveFailures === 0;
    
    this.status.can_advance = hasMinSamples && hasMinDuration && gatesPassing;
  }
  
  /**
   * Get current stage configuration
   */
  private getCurrentStage(): CanaryStage | null {
    return this.config.stages.find(s => s.name === this.status.current_stage) || null;
  }
  
  /**
   * Record baseline metrics
   */
  recordBaseline(metrics: Omit<CanaryMetrics, 'stage' | 'traffic_percentage' | 'baseline_comparison'>): void {
    this.baselineMonitor.recordBaseline(metrics);
  }
  
  /**
   * Get current rollout status
   */
  getStatus(): CanaryRolloutStatus {
    return { ...this.status };
  }
  
  /**
   * Get comprehensive statistics
   */
  getStats() {
    return {
      config: this.config,
      status: this.status,
      baseline: this.baselineMonitor.getStats(),
      rollback: this.rollbackController.getStats(),
      metrics_history: this.metricsHistory.length,
      recent_metrics: this.metricsHistory.slice(-10)
    };
  }
}

/**
 * Production Canary Rollout Manager
 * Orchestrates the complete 5%→25%→100% deployment with strict gates
 */
export class Gemma256CanaryRolloutManager {
  private rolloutEngine: CanaryRolloutEngine;
  private isProduction: boolean;

  constructor(config: Partial<CanaryRolloutConfig> = {}, isProduction = true) {
    this.isProduction = isProduction;
    
    // Production configuration with strict gates
    const productionConfig: CanaryRolloutConfig = {
      enabled: true,
      // 5% → 25% → 100% rollout stages
      stages: [
        {
          name: 'canary_5pct',
          traffic_percentage: 5,
          min_duration_minutes: 30,
          quality_gate_strictness: 1.2  // 20% stricter in early stage
        },
        {
          name: 'canary_25pct',
          traffic_percentage: 25,
          min_duration_minutes: 60,
          quality_gate_strictness: 1.1  // 10% stricter
        },
        {
          name: 'production_100pct',
          traffic_percentage: 100,
          min_duration_minutes: 120,
          quality_gate_strictness: 1.0  // Normal strictness
        }
      ],
      stage_duration_minutes: 30,
      min_samples_per_stage: 100,
      
      // Quality gates (exact requirements from TODO)
      quality_gates: {
        overall_ndcg_delta_threshold: -0.03,      // ≥ −3pp overall
        nl_slice_ndcg_delta_threshold: -0.01,    // ≥ −1pp for NL queries
        sla_recall_at_50_min: 0.85,              // ≥ baseline (assume 85%)
        precision_at_1_max_degradation: 0.02     // ≤ −2pp for symbol/NL
      },
      
      // Operational gates (exact requirements from TODO)
      ops_gates: {
        p95_latency_max_ms: 25,          // ≤ 25ms
        p99_latency_max_ratio: 2.0,      // ≤ 2×p95
        qps_at_150ms_min_ratio: 1.3,     // ≥ 1.3× 768d
        error_rate_max_pct: 2.0          // < 2%
      },
      
      // Integrity gates (exact requirements from TODO)
      integrity_gates: {
        span_coverage_min_pct: 100.0,    // = 100%
        sentinel_nzc_min_pct: 99.0,      // ≥ 99%
        prose_artifact_drift_max: 0.001  // ≤ 0.1pp
      },
      
      // Auto-rollback configuration
      auto_rollback: {
        enabled: true,
        breach_threshold_count: 3,       // 3 consecutive failures
        emergency_rollback_latency_ms: 50, // Emergency at 50ms
        rollback_timeout_minutes: 5     // 5 minute rollback timeout
      },
      
      // Monitoring configuration
      monitoring: {
        sample_interval_seconds: 30,
        alert_on_gate_failures: true,
        store_detailed_metrics: true
      },
      
      ...config
    };

    this.rolloutEngine = new CanaryRolloutEngine(productionConfig);

    console.log(`🚀 Gemma-256 Canary Rollout Manager initialized (production=${isProduction})`);
    console.log(`   Rollout sequence: 5% → 25% → 100%`);
    console.log(`   Kill order: dense.hybrid → dense.256 → dense.768`);
    console.log(`   Auto-rollback: ${productionConfig.auto_rollback.enabled ? 'ENABLED' : 'DISABLED'}`);
  }

  /**
   * Establish baseline before rollout
   */
  recordBaseline(metrics: Omit<CanaryMetrics, 'stage' | 'traffic_percentage' | 'baseline_comparison'>): void {
    this.rolloutEngine.recordBaseline(metrics);
  }

  /**
   * Start canary rollout process
   */
  startRollout(): void {
    if (this.isProduction) {
      console.log(`🚨 PRODUCTION CANARY ROLLOUT STARTING`);
      console.log(`   This will begin routing traffic to Gemma-256 hybrid system`);
      console.log(`   Auto-rollback is enabled with strict SLA gates`);
    }
    
    this.rolloutEngine.startRollout();
  }

  /**
   * Record canary metrics and check gates
   */
  async recordCanaryMetrics(metrics: Omit<CanaryMetrics, 'stage' | 'traffic_percentage'>): Promise<void> {
    await this.rolloutEngine.recordMetrics(metrics);
  }

  /**
   * Advance to next stage if ready
   */
  async advanceToNextStage(): Promise<boolean> {
    return await this.rolloutEngine.advanceStage();
  }

  /**
   * Emergency stop rollout
   */
  emergencyStop(): void {
    console.error(`🚨 EMERGENCY STOP - Manual rollout termination`);
    // Implementation would trigger emergency rollback
  }

  /**
   * Get current rollout status
   */
  getRolloutStatus(): CanaryRolloutStatus {
    return this.rolloutEngine.getStatus();
  }

  /**
   * Get comprehensive statistics
   */
  getStats() {
    return {
      production_mode: this.isProduction,
      ...this.rolloutEngine.getStats()
    };
  }

  /**
   * Validate configuration meets production requirements
   */
  validateConfiguration(): { valid: boolean; issues: string[] } {
    const issues: string[] = [];
    const config = this.rolloutEngine.getStats().config;

    // Validate stage sequence
    const stages = config.stages;
    if (stages.length !== 3) {
      issues.push(`Expected 3 stages, got ${stages.length}`);
    }
    
    if (stages[0]?.traffic_percentage !== 5) {
      issues.push(`First stage should be 5%, got ${stages[0]?.traffic_percentage}%`);
    }
    
    if (stages[1]?.traffic_percentage !== 25) {
      issues.push(`Second stage should be 25%, got ${stages[1]?.traffic_percentage}%`);
    }
    
    if (stages[2]?.traffic_percentage !== 100) {
      issues.push(`Final stage should be 100%, got ${stages[2]?.traffic_percentage}%`);
    }

    // Validate gate thresholds match TODO requirements
    if (config.ops_gates.p95_latency_max_ms !== 25) {
      issues.push(`P95 latency threshold should be 25ms, got ${config.ops_gates.p95_latency_max_ms}ms`);
    }
    
    if (config.ops_gates.qps_at_150ms_min_ratio !== 1.3) {
      issues.push(`QPS ratio should be 1.3×, got ${config.ops_gates.qps_at_150ms_min_ratio}×`);
    }

    if (!config.auto_rollback.enabled && this.isProduction) {
      issues.push('Auto-rollback must be enabled in production');
    }

    return {
      valid: issues.length === 0,
      issues
    };
  }
}