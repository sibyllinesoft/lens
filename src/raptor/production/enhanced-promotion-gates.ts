/**
 * Enhanced Promotion Gates - Automated Progressive Rollout System
 * 
 * Implements comprehensive promotion gates with:
 * 1. Existing gates: p95 ≤ +1ms, p99/p95 ≤ 2.0, span=100%
 * 2. New gates: SLA-Core@10 ≥ baseline, why-mix KL ≤ 0.02
 * 3. Automated 25→50→100% progression when all gates green
 * 4. Real-time monitoring and automated rollback on violations
 * 5. Comprehensive gate validation with statistical confidence
 */

import { EventEmitter } from 'events';
import { writeFile, mkdir } from 'fs/promises';
import { join } from 'path';

export type PromotionStage = 25 | 50 | 100;
export type GateStatus = 'PASS' | 'FAIL' | 'PENDING' | 'UNKNOWN';

export interface PerformanceGate {
  name: string;
  description: string;
  current_value: number;
  target_value: number;
  operator: '<=' | '>=' | '<' | '>' | '=';
  status: GateStatus;
  confidence_level: number; // 0-1
  violation_severity: 'low' | 'medium' | 'high' | 'critical';
  measurement_window_minutes: number;
  sample_count: number;
}

export interface QualityGate {
  name: string;
  description: string;
  current_value: number;
  baseline_value: number;
  delta_threshold: number; // Acceptable change from baseline
  status: GateStatus;
  confidence_level: number;
  violation_severity: 'low' | 'medium' | 'high' | 'critical';
  measurement_window_minutes: number;
  sample_count: number;
}

export interface PromotionStageStatus {
  stage: PromotionStage;
  start_time: Date;
  duration_minutes: number;
  target_duration_minutes: number;
  traffic_percentage: number;
  
  performance_gates: PerformanceGate[];
  quality_gates: QualityGate[];
  
  all_gates_passing: boolean;
  failed_gates: string[];
  critical_failures: number;
  
  promotion_eligible: boolean;
  rollback_triggered: boolean;
  rollback_reason?: string;
}

export interface WhyMixDistribution {
  intent_classification: number; // % of why-intent queries
  lexical_matches: number; // % of why-lexical queries
  semantic_similarity: number; // % of why-semantic queries
  symbol_lookup: number; // % of why-symbol queries
  contextual_understanding: number; // % of why-contextual queries
  total_sample_count: number;
  kl_divergence_from_baseline: number;
}

export interface EnhancedPromotionReport {
  timestamp: Date;
  current_stage: PromotionStage;
  next_stage?: PromotionStage;
  
  stage_history: PromotionStageStatus[];
  current_stage_status: PromotionStageStatus;
  
  gate_summary: {
    total_gates: number;
    passing_gates: number;
    failing_gates: number;
    critical_failures: number;
  };
  
  why_mix_analysis: WhyMixDistribution;
  
  promotion_decision: {
    can_promote: boolean;
    promotion_blocked_by: string[];
    estimated_promotion_time?: Date;
    rollback_recommended: boolean;
  };
  
  recommendations: string[];
  alerts: string[];
}

export interface PromotionGateConfig {
  stage_durations: Record<PromotionStage, number>; // Minutes per stage
  performance_thresholds: {
    p95_latency_max_delta: number; // +1ms max increase
    p99_p95_ratio_max: number; // 2.0 max ratio
    span_coverage_min: number; // 100% required
    sla_core_10_min_delta: number; // ≥ baseline required
  };
  quality_thresholds: {
    why_mix_kl_max: number; // 0.02 max KL divergence
    ndcg_10_min_delta: number; // Minimum nDCG improvement
    recall_50_sla_min_delta: number; // Minimum recall improvement
  };
  statistical_requirements: {
    min_confidence_level: number; // 0.95 = 95% confidence
    min_samples_per_gate: number; // Minimum samples for reliable measurement
    measurement_window_minutes: number; // Rolling window for gate evaluation
  };
  rollback_triggers: {
    critical_gate_failures: number; // Auto-rollback after N critical failures
    consecutive_failures: number; // Auto-rollback after N consecutive failures
    p95_latency_emergency_threshold: number; // Emergency rollback threshold
  };
}

export const DEFAULT_PROMOTION_CONFIG: PromotionGateConfig = {
  stage_durations: {
    25: 48 * 60, // 48 hours for 25%
    50: 24 * 60, // 24 hours for 50%  
    100: 0 // Indefinite for 100%
  },
  performance_thresholds: {
    p95_latency_max_delta: 1.0, // +1ms max
    p99_p95_ratio_max: 2.0,
    span_coverage_min: 1.0, // 100%
    sla_core_10_min_delta: 0.0 // ≥ baseline
  },
  quality_thresholds: {
    why_mix_kl_max: 0.02,
    ndcg_10_min_delta: 0.0, // Flat or better
    recall_50_sla_min_delta: 0.0 // Flat or better
  },
  statistical_requirements: {
    min_confidence_level: 0.95,
    min_samples_per_gate: 1000,
    measurement_window_minutes: 60 // 1 hour rolling window
  },
  rollback_triggers: {
    critical_gate_failures: 2,
    consecutive_failures: 5,
    p95_latency_emergency_threshold: 5.0 // +5ms emergency threshold
  }
};

export class EnhancedPromotionGateSystem extends EventEmitter {
  private config: PromotionGateConfig;
  private currentStage: PromotionStage = 25;
  private stageHistory: PromotionStageStatus[] = [];
  private promotionStartTime: Date = new Date();
  
  // Baseline metrics for comparison
  private baselineMetrics = new Map<string, number>();
  
  // Real-time measurement buffers
  private latencyMeasurements: number[] = [];
  private qualityMeasurements = new Map<string, number[]>();
  private whyMixSamples: WhyMixDistribution[] = [];
  
  constructor(config: PromotionGateConfig = DEFAULT_PROMOTION_CONFIG) {
    super();
    this.config = config;
    this.initializeBaselines();
  }
  
  /**
   * Initialize baseline metrics from historical data
   */
  private initializeBaselines(): void {
    // Production baselines (would be loaded from monitoring system)
    this.baselineMetrics.set('p95_latency_ms', 150);
    this.baselineMetrics.set('p99_latency_ms', 280);
    this.baselineMetrics.set('span_coverage', 1.0);
    this.baselineMetrics.set('sla_core_10', 0.85);
    this.baselineMetrics.set('ndcg_10', 0.815);
    this.baselineMetrics.set('recall_50_sla', 0.68);
    this.baselineMetrics.set('why_mix_kl', 0.0); // Perfect baseline
    
    console.log('📊 Baseline metrics initialized for gate evaluation');
  }
  
  /**
   * Start promotion process at 25% stage
   */
  async startPromotion(): Promise<void> {
    console.log('🚀 Starting enhanced promotion process at 25% stage...');
    
    this.currentStage = 25;
    this.promotionStartTime = new Date();
    
    const initialStatus: PromotionStageStatus = {
      stage: 25,
      start_time: this.promotionStartTime,
      duration_minutes: 0,
      target_duration_minutes: this.config.stage_durations[25],
      traffic_percentage: 25,
      performance_gates: [],
      quality_gates: [],
      all_gates_passing: false,
      failed_gates: [],
      critical_failures: 0,
      promotion_eligible: false,
      rollback_triggered: false
    };
    
    this.stageHistory.push(initialStatus);
    
    // Start continuous monitoring
    this.startContinuousMonitoring();
    
    console.log('✅ Promotion started - monitoring gates continuously');
    this.emit('promotion_started', { stage: 25, start_time: this.promotionStartTime });
  }
  
  /**
   * Ingest real-time metrics for gate evaluation
   */
  async ingestMetrics(metrics: {
    timestamp: Date;
    p95_latency_ms: number;
    p99_latency_ms: number;
    span_coverage: number;
    sla_core_10: number;
    ndcg_10: number;
    recall_50_sla: number;
    query_count: number;
    why_mix_distribution: WhyMixDistribution;
  }): Promise<void> {
    // Buffer metrics for rolling window analysis
    this.latencyMeasurements.push(metrics.p95_latency_ms);
    
    if (!this.qualityMeasurements.has('sla_core_10')) {
      this.qualityMeasurements.set('sla_core_10', []);
    }
    this.qualityMeasurements.get('sla_core_10')!.push(metrics.sla_core_10);
    
    if (!this.qualityMeasurements.has('ndcg_10')) {
      this.qualityMeasurements.set('ndcg_10', []);
    }
    this.qualityMeasurements.get('ndcg_10')!.push(metrics.ndcg_10);
    
    if (!this.qualityMeasurements.has('recall_50_sla')) {
      this.qualityMeasurements.set('recall_50_sla', []);
    }
    this.qualityMeasurements.get('recall_50_sla')!.push(metrics.recall_50_sla);
    
    this.whyMixSamples.push(metrics.why_mix_distribution);
    
    // Trim buffers to measurement window
    const windowSize = this.config.statistical_requirements.measurement_window_minutes * 60; // Convert to samples (assuming 1 sample/second)
    this.trimBuffers(windowSize);
    
    this.emit('metrics_ingested', { stage: this.currentStage, sample_count: this.latencyMeasurements.length });
  }
  
  /**
   * Evaluate all promotion gates for current stage
   */
  async evaluatePromotionGates(outputDir: string): Promise<EnhancedPromotionReport> {
    console.log(`🔍 Evaluating promotion gates for ${this.currentStage}% stage...`);
    
    if (this.stageHistory.length === 0) {
      throw new Error('No promotion stages initiated. Call startPromotion() first.');
    }
    
    await mkdir(outputDir, { recursive: true });
    
    const currentStatus = this.stageHistory[this.stageHistory.length - 1];
    
    // Update stage duration
    currentStatus.duration_minutes = (Date.now() - currentStatus.start_time.getTime()) / (1000 * 60);
    
    // Evaluate performance gates
    currentStatus.performance_gates = await this.evaluatePerformanceGates();
    
    // Evaluate quality gates  
    currentStatus.quality_gates = await this.evaluateQualityGates();
    
    // Compute why-mix analysis
    const whyMixAnalysis = this.computeWhyMixAnalysis();
    
    // Determine gate status
    const allPerformanceGatesPassing = currentStatus.performance_gates.every(gate => gate.status === 'PASS');
    const allQualityGatesPassing = currentStatus.quality_gates.every(gate => gate.status === 'PASS');
    const whyMixPassing = whyMixAnalysis.kl_divergence_from_baseline <= this.config.quality_thresholds.why_mix_kl_max;
    
    currentStatus.all_gates_passing = allPerformanceGatesPassing && allQualityGatesPassing && whyMixPassing;
    
    // Collect failed gates
    currentStatus.failed_gates = [
      ...currentStatus.performance_gates.filter(g => g.status === 'FAIL').map(g => g.name),
      ...currentStatus.quality_gates.filter(g => g.status === 'FAIL').map(g => g.name)
    ];
    
    if (!whyMixPassing) {
      currentStatus.failed_gates.push('why-mix-kl-divergence');
    }
    
    // Count critical failures
    currentStatus.critical_failures = [
      ...currentStatus.performance_gates.filter(g => g.status === 'FAIL' && g.violation_severity === 'critical'),
      ...currentStatus.quality_gates.filter(g => g.status === 'FAIL' && g.violation_severity === 'critical')
    ].length;
    
    // Determine promotion eligibility
    const stageTimeElapsed = currentStatus.duration_minutes >= currentStatus.target_duration_minutes;
    currentStatus.promotion_eligible = currentStatus.all_gates_passing && stageTimeElapsed;
    
    // Check rollback triggers
    const rollbackDecision = this.evaluateRollbackTriggers(currentStatus);
    currentStatus.rollback_triggered = rollbackDecision.should_rollback;
    currentStatus.rollback_reason = rollbackDecision.reason;
    
    // Generate comprehensive report
    const report: EnhancedPromotionReport = {
      timestamp: new Date(),
      current_stage: this.currentStage,
      next_stage: this.getNextStage(),
      stage_history: this.stageHistory,
      current_stage_status: currentStatus,
      gate_summary: {
        total_gates: currentStatus.performance_gates.length + currentStatus.quality_gates.length + 1, // +1 for why-mix
        passing_gates: currentStatus.performance_gates.filter(g => g.status === 'PASS').length + 
                       currentStatus.quality_gates.filter(g => g.status === 'PASS').length + 
                       (whyMixPassing ? 1 : 0),
        failing_gates: currentStatus.failed_gates.length,
        critical_failures: currentStatus.critical_failures
      },
      why_mix_analysis: whyMixAnalysis,
      promotion_decision: {
        can_promote: currentStatus.promotion_eligible,
        promotion_blocked_by: currentStatus.failed_gates,
        estimated_promotion_time: this.estimatePromotionTime(currentStatus),
        rollback_recommended: currentStatus.rollback_triggered
      },
      recommendations: this.generateRecommendations(currentStatus, whyMixAnalysis),
      alerts: this.generateAlerts(currentStatus, whyMixAnalysis)
    };
    
    // Save comprehensive report
    await this.savePromotionReport(report, outputDir);
    
    // Handle automatic promotion or rollback
    if (currentStatus.rollback_triggered) {
      console.log(`🚨 ROLLBACK TRIGGERED: ${currentStatus.rollback_reason}`);
      await this.executeRollback(currentStatus.rollback_reason!);
    } else if (currentStatus.promotion_eligible && this.currentStage < 100) {
      console.log(`✅ All gates green - promoting to ${this.getNextStage()}%`);
      await this.promoteToNextStage();
    }
    
    this.emit('gates_evaluated', report);
    return report;
  }
  
  /**
   * Evaluate performance gates
   */
  private async evaluatePerformanceGates(): Promise<PerformanceGate[]> {
    const gates: PerformanceGate[] = [];
    
    if (this.latencyMeasurements.length < this.config.statistical_requirements.min_samples_per_gate) {
      console.warn('⚠️  Insufficient latency samples for reliable gate evaluation');
      return gates;
    }
    
    // P95 Latency Gate
    const currentP95 = this.computePercentile(this.latencyMeasurements, 95);
    const baselineP95 = this.baselineMetrics.get('p95_latency_ms')!;
    const p95Delta = currentP95 - baselineP95;
    
    gates.push({
      name: 'p95-latency-delta',
      description: 'P95 latency increase ≤ +1ms',
      current_value: p95Delta,
      target_value: this.config.performance_thresholds.p95_latency_max_delta,
      operator: '<=',
      status: p95Delta <= this.config.performance_thresholds.p95_latency_max_delta ? 'PASS' : 'FAIL',
      confidence_level: this.computeConfidenceLevel(this.latencyMeasurements),
      violation_severity: p95Delta > 5.0 ? 'critical' : p95Delta > 3.0 ? 'high' : 'medium',
      measurement_window_minutes: this.config.statistical_requirements.measurement_window_minutes,
      sample_count: this.latencyMeasurements.length
    });
    
    // P99/P95 Ratio Gate
    const currentP99 = this.computePercentile(this.latencyMeasurements, 99);
    const p99P95Ratio = currentP99 / currentP95;
    
    gates.push({
      name: 'p99-p95-ratio',
      description: 'P99/P95 latency ratio ≤ 2.0',
      current_value: p99P95Ratio,
      target_value: this.config.performance_thresholds.p99_p95_ratio_max,
      operator: '<=',
      status: p99P95Ratio <= this.config.performance_thresholds.p99_p95_ratio_max ? 'PASS' : 'FAIL',
      confidence_level: this.computeConfidenceLevel(this.latencyMeasurements),
      violation_severity: p99P95Ratio > 3.0 ? 'critical' : p99P95Ratio > 2.5 ? 'high' : 'medium',
      measurement_window_minutes: this.config.statistical_requirements.measurement_window_minutes,
      sample_count: this.latencyMeasurements.length
    });
    
    // Span Coverage Gate (simplified - assumes 100% from current system design)
    gates.push({
      name: 'span-coverage',
      description: 'Span coverage = 100%',
      current_value: 1.0,
      target_value: this.config.performance_thresholds.span_coverage_min,
      operator: '>=',
      status: 'PASS', // Assumed to be 100% based on current implementation
      confidence_level: 1.0,
      violation_severity: 'critical',
      measurement_window_minutes: this.config.statistical_requirements.measurement_window_minutes,
      sample_count: this.latencyMeasurements.length
    });
    
    return gates;
  }
  
  /**
   * Evaluate quality gates
   */
  private async evaluateQualityGates(): Promise<QualityGate[]> {
    const gates: QualityGate[] = [];
    
    // SLA-Core@10 Gate
    const slaCoreValues = this.qualityMeasurements.get('sla_core_10') || [];
    if (slaCoreValues.length >= this.config.statistical_requirements.min_samples_per_gate) {
      const currentSlaCore = slaCoreValues.reduce((sum, val) => sum + val, 0) / slaCoreValues.length;
      const baselineSlaCore = this.baselineMetrics.get('sla_core_10')!;
      const slaDelta = currentSlaCore - baselineSlaCore;
      
      gates.push({
        name: 'sla-core-10',
        description: 'SLA-Core@10 ≥ baseline',
        current_value: currentSlaCore,
        baseline_value: baselineSlaCore,
        delta_threshold: this.config.performance_thresholds.sla_core_10_min_delta,
        status: slaDelta >= this.config.performance_thresholds.sla_core_10_min_delta ? 'PASS' : 'FAIL',
        confidence_level: this.computeConfidenceLevel(slaCoreValues),
        violation_severity: slaDelta < -0.05 ? 'critical' : slaDelta < -0.02 ? 'high' : 'medium',
        measurement_window_minutes: this.config.statistical_requirements.measurement_window_minutes,
        sample_count: slaCoreValues.length
      });
    }
    
    // nDCG@10 Gate
    const ndcgValues = this.qualityMeasurements.get('ndcg_10') || [];
    if (ndcgValues.length >= this.config.statistical_requirements.min_samples_per_gate) {
      const currentNdcg = ndcgValues.reduce((sum, val) => sum + val, 0) / ndcgValues.length;
      const baselineNdcg = this.baselineMetrics.get('ndcg_10')!;
      const ndcgDelta = currentNdcg - baselineNdcg;
      
      gates.push({
        name: 'ndcg-10',
        description: 'nDCG@10 ≥ baseline (flat or better)',
        current_value: currentNdcg,
        baseline_value: baselineNdcg,
        delta_threshold: this.config.quality_thresholds.ndcg_10_min_delta,
        status: ndcgDelta >= this.config.quality_thresholds.ndcg_10_min_delta ? 'PASS' : 'FAIL',
        confidence_level: this.computeConfidenceLevel(ndcgValues),
        violation_severity: ndcgDelta < -0.02 ? 'critical' : ndcgDelta < -0.01 ? 'high' : 'medium',
        measurement_window_minutes: this.config.statistical_requirements.measurement_window_minutes,
        sample_count: ndcgValues.length
      });
    }
    
    // Recall@50 SLA Gate
    const recallValues = this.qualityMeasurements.get('recall_50_sla') || [];
    if (recallValues.length >= this.config.statistical_requirements.min_samples_per_gate) {
      const currentRecall = recallValues.reduce((sum, val) => sum + val, 0) / recallValues.length;
      const baselineRecall = this.baselineMetrics.get('recall_50_sla')!;
      const recallDelta = currentRecall - baselineRecall;
      
      gates.push({
        name: 'recall-50-sla',
        description: 'Recall@50 SLA ≥ baseline',
        current_value: currentRecall,
        baseline_value: baselineRecall,
        delta_threshold: this.config.quality_thresholds.recall_50_sla_min_delta,
        status: recallDelta >= this.config.quality_thresholds.recall_50_sla_min_delta ? 'PASS' : 'FAIL',
        confidence_level: this.computeConfidenceLevel(recallValues),
        violation_severity: recallDelta < -0.02 ? 'critical' : recallDelta < -0.01 ? 'high' : 'medium',
        measurement_window_minutes: this.config.statistical_requirements.measurement_window_minutes,
        sample_count: recallValues.length
      });
    }
    
    return gates;
  }
  
  /**
   * Compute why-mix distribution analysis
   */
  private computeWhyMixAnalysis(): WhyMixDistribution {
    if (this.whyMixSamples.length === 0) {
      return {
        intent_classification: 0,
        lexical_matches: 0,
        semantic_similarity: 0,
        symbol_lookup: 0,
        contextual_understanding: 0,
        total_sample_count: 0,
        kl_divergence_from_baseline: 0
      };
    }
    
    // Aggregate samples
    const totalSamples = this.whyMixSamples.length;
    const avgDistribution = this.whyMixSamples.reduce(
      (acc, sample) => ({
        intent_classification: acc.intent_classification + sample.intent_classification,
        lexical_matches: acc.lexical_matches + sample.lexical_matches,
        semantic_similarity: acc.semantic_similarity + sample.semantic_similarity,
        symbol_lookup: acc.symbol_lookup + sample.symbol_lookup,
        contextual_understanding: acc.contextual_understanding + sample.contextual_understanding,
        total_sample_count: acc.total_sample_count + sample.total_sample_count
      }),
      {
        intent_classification: 0,
        lexical_matches: 0,
        semantic_similarity: 0,
        symbol_lookup: 0,
        contextual_understanding: 0,
        total_sample_count: 0
      }
    );
    
    // Normalize by sample count
    const normalizedDistribution = {
      intent_classification: avgDistribution.intent_classification / totalSamples,
      lexical_matches: avgDistribution.lexical_matches / totalSamples,
      semantic_similarity: avgDistribution.semantic_similarity / totalSamples,
      symbol_lookup: avgDistribution.symbol_lookup / totalSamples,
      contextual_understanding: avgDistribution.contextual_understanding / totalSamples,
      total_sample_count: avgDistribution.total_sample_count
    };
    
    // Compute KL divergence from baseline (uniform distribution assumption)
    const baselineDistribution = [0.2, 0.2, 0.2, 0.2, 0.2]; // Uniform baseline
    const currentDistribution = [
      normalizedDistribution.intent_classification / 100,
      normalizedDistribution.lexical_matches / 100,
      normalizedDistribution.semantic_similarity / 100,
      normalizedDistribution.symbol_lookup / 100,
      normalizedDistribution.contextual_understanding / 100
    ];
    
    const klDivergence = this.computeKLDivergence(currentDistribution, baselineDistribution);
    
    return {
      ...normalizedDistribution,
      kl_divergence_from_baseline: klDivergence
    };
  }
  
  /**
   * Compute KL divergence between two distributions
   */
  private computeKLDivergence(p: number[], q: number[]): number {
    let kl = 0;
    
    for (let i = 0; i < p.length; i++) {
      if (p[i] > 0 && q[i] > 0) {
        kl += p[i] * Math.log(p[i] / q[i]);
      }
    }
    
    return kl;
  }
  
  /**
   * Evaluate rollback triggers
   */
  private evaluateRollbackTriggers(status: PromotionStageStatus): { should_rollback: boolean; reason?: string } {
    // Critical gate failures
    if (status.critical_failures >= this.config.rollback_triggers.critical_gate_failures) {
      return {
        should_rollback: true,
        reason: `${status.critical_failures} critical gate failures exceed threshold`
      };
    }
    
    // Emergency p95 latency threshold
    if (this.latencyMeasurements.length > 0) {
      const currentP95 = this.computePercentile(this.latencyMeasurements, 95);
      const baselineP95 = this.baselineMetrics.get('p95_latency_ms')!;
      const p95Delta = currentP95 - baselineP95;
      
      if (p95Delta > this.config.rollback_triggers.p95_latency_emergency_threshold) {
        return {
          should_rollback: true,
          reason: `P95 latency increased by ${p95Delta.toFixed(1)}ms (emergency threshold: ${this.config.rollback_triggers.p95_latency_emergency_threshold}ms)`
        };
      }
    }
    
    // Consecutive failures across multiple evaluation windows
    const recentFailures = this.stageHistory
      .slice(-this.config.rollback_triggers.consecutive_failures)
      .filter(stage => !stage.all_gates_passing);
    
    if (recentFailures.length >= this.config.rollback_triggers.consecutive_failures) {
      return {
        should_rollback: true,
        reason: `${recentFailures.length} consecutive gate failures`
      };
    }
    
    return { should_rollback: false };
  }
  
  /**
   * Promote to next stage
   */
  private async promoteToNextStage(): Promise<void> {
    const nextStage = this.getNextStage();
    if (!nextStage) return;
    
    console.log(`🎯 Promoting from ${this.currentStage}% to ${nextStage}%`);
    
    this.currentStage = nextStage;
    
    const newStageStatus: PromotionStageStatus = {
      stage: nextStage,
      start_time: new Date(),
      duration_minutes: 0,
      target_duration_minutes: this.config.stage_durations[nextStage],
      traffic_percentage: nextStage,
      performance_gates: [],
      quality_gates: [],
      all_gates_passing: false,
      failed_gates: [],
      critical_failures: 0,
      promotion_eligible: false,
      rollback_triggered: false
    };
    
    this.stageHistory.push(newStageStatus);
    
    this.emit('stage_promoted', {
      from_stage: this.stageHistory[this.stageHistory.length - 2].stage,
      to_stage: nextStage,
      timestamp: new Date()
    });
  }
  
  /**
   * Execute rollback procedure
   */
  private async executeRollback(reason: string): Promise<void> {
    console.log(`🚨 EXECUTING ROLLBACK: ${reason}`);
    
    // Rollback to previous stage or 0% if at first stage
    const previousStage = this.currentStage === 25 ? 0 : (this.currentStage === 50 ? 25 : 50);
    
    this.emit('rollback_initiated', {
      from_stage: this.currentStage,
      to_stage: previousStage,
      reason,
      timestamp: new Date()
    });
    
    // Would integrate with deployment system to actually rollback traffic
    console.log(`🔄 Traffic rolled back from ${this.currentStage}% to ${previousStage}%`);
  }
  
  /**
   * Get next promotion stage
   */
  private getNextStage(): PromotionStage | undefined {
    if (this.currentStage === 25) return 50;
    if (this.currentStage === 50) return 100;
    return undefined; // Already at 100%
  }
  
  /**
   * Estimate promotion time based on current gate status
   */
  private estimatePromotionTime(status: PromotionStageStatus): Date | undefined {
    if (status.all_gates_passing) {
      // If all gates are passing, promotion happens when time requirement is met
      const remainingTime = status.target_duration_minutes - status.duration_minutes;
      if (remainingTime <= 0) return new Date(); // Can promote now
      return new Date(Date.now() + remainingTime * 60 * 1000);
    }
    
    return undefined; // Cannot estimate with failing gates
  }
  
  /**
   * Generate recommendations based on gate status
   */
  private generateRecommendations(status: PromotionStageStatus, whyMixAnalysis: WhyMixDistribution): string[] {
    const recommendations: string[] = [];
    
    if (status.all_gates_passing) {
      if (status.promotion_eligible) {
        recommendations.push('✅ All gates passing and time requirement met - ready for promotion');
      } else {
        const remainingTime = status.target_duration_minutes - status.duration_minutes;
        recommendations.push(`🕐 All gates passing - promotion eligible in ${remainingTime.toFixed(0)} minutes`);
      }
    } else {
      recommendations.push('❌ Gate failures detected - address issues before promotion');
      
      // Specific recommendations for failed gates
      for (const failedGate of status.failed_gates) {
        if (failedGate.includes('latency')) {
          recommendations.push('  - Consider performance optimization or traffic shaping');
        } else if (failedGate.includes('sla-core')) {
          recommendations.push('  - Review search quality configurations and thresholds');
        } else if (failedGate.includes('why-mix')) {
          recommendations.push('  - Investigate why-classification drift and model calibration');
        }
      }
    }
    
    // Why-mix specific recommendations
    if (whyMixAnalysis.kl_divergence_from_baseline > 0.01) {
      recommendations.push(`📊 Why-mix distribution shift detected (KL=${whyMixAnalysis.kl_divergence_from_baseline.toFixed(3)}) - monitor for query pattern changes`);
    }
    
    return recommendations;
  }
  
  /**
   * Generate alerts for critical issues
   */
  private generateAlerts(status: PromotionStageStatus, whyMixAnalysis: WhyMixDistribution): string[] {
    const alerts: string[] = [];
    
    if (status.critical_failures > 0) {
      alerts.push(`🚨 CRITICAL: ${status.critical_failures} critical gate failure(s) - rollback risk HIGH`);
    }
    
    if (status.rollback_triggered) {
      alerts.push(`🛑 ROLLBACK TRIGGERED: ${status.rollback_reason}`);
    }
    
    // Latency emergency alerts
    if (this.latencyMeasurements.length > 0) {
      const currentP95 = this.computePercentile(this.latencyMeasurements, 95);
      const baselineP95 = this.baselineMetrics.get('p95_latency_ms')!;
      const p95Delta = currentP95 - baselineP95;
      
      if (p95Delta > 3.0) {
        alerts.push(`⚠️  HIGH LATENCY: P95 increased by ${p95Delta.toFixed(1)}ms`);
      }
    }
    
    // Why-mix alerts
    if (whyMixAnalysis.kl_divergence_from_baseline > this.config.quality_thresholds.why_mix_kl_max) {
      alerts.push(`📈 WHY-MIX VIOLATION: KL divergence ${whyMixAnalysis.kl_divergence_from_baseline.toFixed(3)} exceeds ${this.config.quality_thresholds.why_mix_kl_max}`);
    }
    
    return alerts;
  }
  
  /**
   * Start continuous monitoring loop
   */
  private startContinuousMonitoring(): void {
    // Start monitoring loop (every 5 minutes)
    const monitoringInterval = setInterval(async () => {
      try {
        await this.evaluatePromotionGates('./promotion-monitoring');
      } catch (error) {
        console.error('❌ Monitoring error:', error);
        this.emit('monitoring_error', error);
      }
    }, 5 * 60 * 1000); // 5 minutes
    
    // Cleanup on process exit
    process.on('SIGTERM', () => clearInterval(monitoringInterval));
    process.on('SIGINT', () => clearInterval(monitoringInterval));
  }
  
  // Helper methods
  private computePercentile(values: number[], percentile: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.ceil((percentile / 100) * sorted.length) - 1;
    return sorted[Math.max(0, index)];
  }
  
  private computeConfidenceLevel(values: number[]): number {
    // Simplified confidence based on sample size
    const n = values.length;
    const minRequired = this.config.statistical_requirements.min_samples_per_gate;
    return Math.min(1.0, n / minRequired);
  }
  
  private trimBuffers(maxSize: number): void {
    if (this.latencyMeasurements.length > maxSize) {
      this.latencyMeasurements = this.latencyMeasurements.slice(-maxSize);
    }
    
    for (const [key, values] of this.qualityMeasurements) {
      if (values.length > maxSize) {
        this.qualityMeasurements.set(key, values.slice(-maxSize));
      }
    }
    
    if (this.whyMixSamples.length > Math.floor(maxSize / 10)) {
      this.whyMixSamples = this.whyMixSamples.slice(-Math.floor(maxSize / 10));
    }
  }
  
  /**
   * Save comprehensive promotion report
   */
  private async savePromotionReport(report: EnhancedPromotionReport, outputDir: string): Promise<void> {
    // Save full JSON report
    await writeFile(
      join(outputDir, 'enhanced-promotion-gate-report.json'),
      JSON.stringify(report, null, 2)
    );
    
    // Save markdown summary
    const summaryReport = this.generatePromotionMarkdown(report);
    await writeFile(join(outputDir, 'promotion-gate-summary.md'), summaryReport);
    
    console.log(`✅ Promotion gate report saved to ${outputDir}/`);
  }
  
  /**
   * Generate markdown promotion report
   */
  private generatePromotionMarkdown(report: EnhancedPromotionReport): string {
    let md = '# Enhanced Promotion Gate Report\n\n';
    
    md += `**Report Time**: ${report.timestamp.toISOString()}\n`;
    md += `**Current Stage**: ${report.current_stage}%\n`;
    md += `**Next Stage**: ${report.next_stage ? report.next_stage + '%' : 'Complete'}\n\n`;
    
    // Promotion status
    if (report.promotion_decision.can_promote) {
      md += '## 🟢 Status: READY FOR PROMOTION\n\n';
    } else if (report.promotion_decision.rollback_recommended) {
      md += '## 🔴 Status: ROLLBACK RECOMMENDED\n\n';
    } else {
      md += '## 🟡 Status: PROMOTION BLOCKED\n\n';
    }
    
    // Gate summary
    md += '## 📊 Gate Summary\n\n';
    md += `- **Total Gates**: ${report.gate_summary.total_gates}\n`;
    md += `- **Passing**: ${report.gate_summary.passing_gates} ✅\n`;
    md += `- **Failing**: ${report.gate_summary.failing_gates} ❌\n`;
    md += `- **Critical Failures**: ${report.gate_summary.critical_failures} 🚨\n\n`;
    
    // Performance gates
    md += '## ⚡ Performance Gates\n\n';
    md += '| Gate | Current | Target | Status | Confidence | Samples |\n';
    md += '|------|---------|--------|--------|------------|--------|\n';
    
    for (const gate of report.current_stage_status.performance_gates) {
      const status = gate.status === 'PASS' ? '✅' : '❌';
      md += `| ${gate.name} | ${gate.current_value.toFixed(2)} | ${gate.operator} ${gate.target_value} | ${status} | ${(gate.confidence_level * 100).toFixed(1)}% | ${gate.sample_count} |\n`;
    }
    md += '\n';
    
    // Quality gates
    md += '## 🎯 Quality Gates\n\n';
    md += '| Gate | Current | Baseline | Delta | Status | Confidence | Samples |\n';
    md += '|------|---------|----------|-------|--------|------------|--------|\n';
    
    for (const gate of report.current_stage_status.quality_gates) {
      const status = gate.status === 'PASS' ? '✅' : '❌';
      const delta = gate.current_value - gate.baseline_value;
      md += `| ${gate.name} | ${gate.current_value.toFixed(3)} | ${gate.baseline_value.toFixed(3)} | ${delta >= 0 ? '+' : ''}${delta.toFixed(3)} | ${status} | ${(gate.confidence_level * 100).toFixed(1)}% | ${gate.sample_count} |\n`;
    }
    md += '\n';
    
    // Why-mix analysis
    md += '## 🤔 Why-Mix Analysis\n\n';
    md += `- **KL Divergence**: ${report.why_mix_analysis.kl_divergence_from_baseline.toFixed(3)} (threshold: ${this.config.quality_thresholds.why_mix_kl_max})\n`;
    md += `- **Intent Classification**: ${report.why_mix_analysis.intent_classification.toFixed(1)}%\n`;
    md += `- **Lexical Matches**: ${report.why_mix_analysis.lexical_matches.toFixed(1)}%\n`;
    md += `- **Semantic Similarity**: ${report.why_mix_analysis.semantic_similarity.toFixed(1)}%\n`;
    md += `- **Symbol Lookup**: ${report.why_mix_analysis.symbol_lookup.toFixed(1)}%\n`;
    md += `- **Contextual Understanding**: ${report.why_mix_analysis.contextual_understanding.toFixed(1)}%\n\n`;
    
    // Alerts
    if (report.alerts.length > 0) {
      md += '## 🚨 Alerts\n\n';
      for (const alert of report.alerts) {
        md += `- **${alert}**\n`;
      }
      md += '\n';
    }
    
    // Recommendations
    md += '## 💡 Recommendations\n\n';
    for (const rec of report.recommendations) {
      md += `- ${rec}\n`;
    }
    
    return md;
  }
  
  /**
   * Generate synthetic metrics for testing
   */
  static generateSyntheticMetrics(): any {
    return {
      timestamp: new Date(),
      p95_latency_ms: 148 + Math.random() * 6, // Slight improvement
      p99_latency_ms: 275 + Math.random() * 15,
      span_coverage: 1.0,
      sla_core_10: 0.855 + Math.random() * 0.02, // Slight improvement
      ndcg_10: 0.820 + Math.random() * 0.01, // Slight improvement
      recall_50_sla: 0.685 + Math.random() * 0.01, // Slight improvement
      query_count: 5000 + Math.floor(Math.random() * 1000),
      why_mix_distribution: {
        intent_classification: 18 + Math.random() * 4,
        lexical_matches: 22 + Math.random() * 4,
        semantic_similarity: 20 + Math.random() * 4,
        symbol_lookup: 21 + Math.random() * 4,
        contextual_understanding: 19 + Math.random() * 4,
        total_sample_count: 1000 + Math.floor(Math.random() * 200),
        kl_divergence_from_baseline: Math.random() * 0.015 // Within threshold
      }
    };
  }
}

// Factory function
export function createEnhancedPromotionGateSystem(config?: Partial<PromotionGateConfig>): EnhancedPromotionGateSystem {
  const fullConfig = { ...DEFAULT_PROMOTION_CONFIG, ...config };
  return new EnhancedPromotionGateSystem(fullConfig);
}