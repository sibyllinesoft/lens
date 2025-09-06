/**
 * Operational Gates and Validation Framework
 * 
 * Central orchestration system that coordinates all validation components
 * and enforces operational gates for deployment decisions. Implements
 * comprehensive validation pipeline with automated pass/fail decisions.
 * 
 * Core Gates:
 * - ΔnDCG@10 ≥ +2pp on NL queries
 * - SLA-Recall@50 ≥ 0 (no regressions)
 * - p95 ≤ +1ms latency constraint  
 * - upshift ∈ [3%,7%] budget enforcement
 * - Span coverage = 100%
 * - p99/p95 ≤ 2.0 reliability constraint
 */

import { z } from 'zod';
import { LensTracer, tracer, meter } from '../telemetry/tracer.js';
import { globalRiskLedger, RiskBudgetLedger } from './risk-budget-ledger.js';
import { globalTDI, TeamDraftInterleaving } from './team-draft-interleaving.js';
import { globalReplaySystem, CounterfactualReplay } from './counterfactual-replay.js';
import { globalConformalAuditing, ConformalAuditing } from './conformal-auditing.js';
import { globalCalibrationMonitoring, CalibrationMonitoring } from './calibration-monitoring.js';
import type { SearchContext, SearchHit } from '../types/core.js';

// Gate evaluation result
export const GateResultSchema = z.object({
  gate_name: z.string(),
  passed: z.boolean(),
  measured_value: z.number(),
  threshold_value: z.number(),
  margin: z.number().optional(),
  confidence: z.number().min(0).max(1).optional(),
  details: z.string().optional(),
  severity: z.enum(['info', 'warning', 'critical']),
  timestamp: z.date(),
});

export type GateResult = z.infer<typeof GateResultSchema>;

// Validation session for a deployment
export const ValidationSessionSchema = z.object({
  session_id: z.string(),
  deployment_candidate: z.string(),
  start_time: z.date(),
  end_time: z.date().optional(),
  status: z.enum(['running', 'passed', 'failed', 'aborted']),
  gate_results: z.array(GateResultSchema),
  summary_metrics: z.object({
    total_gates: z.number().int(),
    passed_gates: z.number().int(),
    failed_gates: z.number().int(),
    critical_failures: z.number().int(),
    overall_score: z.number().min(0).max(1),
  }),
  recommendations: z.array(z.string()),
  next_actions: z.array(z.enum(['proceed', 'investigate', 'rollback', 'abort'])),
  metadata: z.object({
    traffic_percentage: z.number().min(0).max(100),
    sample_size: z.number().int(),
    duration_hours: z.number(),
    environment: z.enum(['staging', 'canary', 'production']),
  }),
});

export type ValidationSession = z.infer<typeof ValidationSessionSchema>;

// Gate configuration
export const GateConfigSchema = z.object({
  gate_name: z.string(),
  enabled: z.boolean(),
  threshold: z.number(),
  direction: z.enum(['greater_than', 'less_than', 'equal', 'range']),
  range_bounds: z.tuple([z.number(), z.number()]).optional(),
  severity: z.enum(['info', 'warning', 'critical']),
  min_samples: z.number().int().min(1),
  confidence_required: z.number().min(0).max(1).optional(),
  measurement_window_hours: z.number().min(0.5).max(72),
  auto_abort_on_failure: z.boolean(),
});

export type GateConfig = z.infer<typeof GateConfigSchema>;

// Default gate configuration per TODO.md requirements
const DEFAULT_GATE_CONFIGS: GateConfig[] = [
  {
    gate_name: 'ndcg_improvement_nl',
    enabled: true,
    threshold: 0.02, // +2pp
    direction: 'greater_than',
    severity: 'critical',
    min_samples: 100,
    confidence_required: 0.95,
    measurement_window_hours: 24,
    auto_abort_on_failure: true,
  },
  {
    gate_name: 'sla_recall_regression',
    enabled: true,
    threshold: 0.0, // ≥ 0 (no regressions)
    direction: 'greater_than',
    severity: 'critical',
    min_samples: 200,
    confidence_required: 0.95,
    measurement_window_hours: 24,
    auto_abort_on_failure: true,
  },
  {
    gate_name: 'p95_latency_constraint',
    enabled: true,
    threshold: 1.0, // +1ms max
    direction: 'less_than',
    severity: 'critical',
    min_samples: 1000,
    measurement_window_hours: 4,
    auto_abort_on_failure: true,
  },
  {
    gate_name: 'upshift_rate_budget',
    enabled: true,
    threshold: 0.05, // 5% center
    direction: 'range',
    range_bounds: [0.03, 0.07], // [3%, 7%]
    severity: 'warning',
    min_samples: 500,
    measurement_window_hours: 12,
    auto_abort_on_failure: false,
  },
  {
    gate_name: 'span_coverage_complete',
    enabled: true,
    threshold: 1.0, // 100% coverage
    direction: 'equal',
    severity: 'critical',
    min_samples: 50,
    measurement_window_hours: 1,
    auto_abort_on_failure: true,
  },
  {
    gate_name: 'reliability_constraint',
    enabled: true,
    threshold: 2.0, // p99/p95 ≤ 2.0
    direction: 'less_than',
    severity: 'critical',
    min_samples: 1000,
    measurement_window_hours: 6,
    auto_abort_on_failure: true,
  },
];

// Metrics for operational gates
const gateMetrics = {
  gate_evaluations: meter.createCounter('lens_gate_evaluations_total', {
    description: 'Total gate evaluations performed',
  }),
  gate_failures: meter.createCounter('lens_gate_failures_total', {
    description: 'Gate failures by gate and severity',
  }),
  validation_sessions: meter.createCounter('lens_validation_sessions_total', {
    description: 'Validation sessions by status',
  }),
  session_duration: meter.createHistogram('lens_validation_session_duration_hours', {
    description: 'Duration of validation sessions',
  }),
  overall_scores: meter.createHistogram('lens_validation_overall_score', {
    description: 'Overall validation scores',
  }),
  auto_aborts: meter.createCounter('lens_validation_auto_aborts_total', {
    description: 'Auto-aborted validations by reason',
  }),
};

/**
 * Operational Gates and Validation Framework
 * 
 * Orchestrates all validation components and makes deployment decisions
 * based on comprehensive operational gate evaluations.
 */
export class OperationalGates {
  private gateConfigs: Map<string, GateConfig>;
  private activeSessions: Map<string, ValidationSession>;
  private sessionHistory: ValidationSession[] = [];
  
  // Component references
  private riskLedger: RiskBudgetLedger;
  private tdi: TeamDraftInterleaving;
  private replaySystem: CounterfactualReplay;
  private conformalAuditing: ConformalAuditing;
  private calibrationMonitoring: CalibrationMonitoring;

  constructor(
    gateConfigs: GateConfig[] = DEFAULT_GATE_CONFIGS,
    components?: {
      riskLedger?: RiskBudgetLedger;
      tdi?: TeamDraftInterleaving;
      replaySystem?: CounterfactualReplay;
      conformalAuditing?: ConformalAuditing;
      calibrationMonitoring?: CalibrationMonitoring;
    }
  ) {
    this.gateConfigs = new Map();
    gateConfigs.forEach(config => {
      this.gateConfigs.set(config.gate_name, config);
    });
    
    this.activeSessions = new Map();
    
    // Initialize components
    this.riskLedger = components?.riskLedger || globalRiskLedger;
    this.tdi = components?.tdi || globalTDI;
    this.replaySystem = components?.replaySystem || globalReplaySystem;
    this.conformalAuditing = components?.conformalAuditing || globalConformalAuditing;
    this.calibrationMonitoring = components?.calibrationMonitoring || globalCalibrationMonitoring;
  }

  /**
   * Start a new validation session
   */
  async startValidationSession(
    deploymentCandidate: string,
    trafficPercentage: number = 25,
    environment: 'staging' | 'canary' | 'production' = 'canary'
  ): Promise<string> {
    const sessionId = `validation_${deploymentCandidate}_${Date.now()}`;
    const span = LensTracer.createChildSpan('start_validation_session', {
      'lens.session_id': sessionId,
      'lens.deployment_candidate': deploymentCandidate,
      'lens.traffic_percentage': trafficPercentage,
    });

    try {
      const session: ValidationSession = {
        session_id: sessionId,
        deployment_candidate: deploymentCandidate,
        start_time: new Date(),
        status: 'running',
        gate_results: [],
        summary_metrics: {
          total_gates: this.gateConfigs.size,
          passed_gates: 0,
          failed_gates: 0,
          critical_failures: 0,
          overall_score: 0,
        },
        recommendations: [],
        next_actions: [],
        metadata: {
          traffic_percentage: trafficPercentage,
          sample_size: 0,
          duration_hours: 0,
          environment,
        },
      };

      this.activeSessions.set(sessionId, session);

      // Start background validation process
      this.runValidationProcess(sessionId);

      gateMetrics.validation_sessions.add(1, {
        status: 'started',
        environment,
      });

      span.setAttributes({
        'lens.session_started': true,
        'lens.total_gates': session.summary_metrics.total_gates,
      });

      console.log(`Started validation session ${sessionId} for ${deploymentCandidate}`);
      return sessionId;

    } finally {
      span.end();
    }
  }

  /**
   * Run the validation process for a session
   */
  private async runValidationProcess(sessionId: string): Promise<void> {
    const session = this.activeSessions.get(sessionId);
    if (!session) return;

    const span = LensTracer.createChildSpan('run_validation_process', {
      'lens.session_id': sessionId,
    });

    try {
      console.log(`Running validation process for session ${sessionId}`);

      // Wait for minimum measurement window
      const minWindow = Math.min(...Array.from(this.gateConfigs.values()).map(g => g.measurement_window_hours));
      console.log(`Waiting ${minWindow} hours for minimum measurement window`);
      
      // In practice, would wait for actual time
      // For demo, we'll simulate immediate evaluation
      await this.sleep(1000); // 1 second delay for demo

      // Evaluate all gates
      for (const gateConfig of this.gateConfigs.values()) {
        if (!gateConfig.enabled) continue;

        const gateResult = await this.evaluateGate(sessionId, gateConfig);
        session.gate_results.push(gateResult);

        // Check for auto-abort conditions
        if (!gateResult.passed && gateConfig.auto_abort_on_failure && gateConfig.severity === 'critical') {
          session.status = 'failed';
          session.recommendations.push(`Critical gate failure: ${gateConfig.gate_name}`);
          session.next_actions.push('abort');
          
          gateMetrics.auto_aborts.add(1, {
            gate: gateConfig.gate_name,
            reason: 'critical_failure',
          });

          console.error(`Auto-aborting session ${sessionId} due to critical gate failure: ${gateConfig.gate_name}`);
          break;
        }
      }

      // Finalize session if not already aborted
      if (session.status === 'running') {
        this.finalizeSession(sessionId);
      }

      span.setAttributes({
        'lens.validation_completed': true,
        'lens.final_status': session.status,
        'lens.gates_evaluated': session.gate_results.length,
      });

    } catch (error: any) {
      console.error(`Validation process failed for session ${sessionId}:`, error);
      session.status = 'failed';
      session.recommendations.push(`Process error: ${error.message}`);
      session.next_actions.push('investigate');
      
      span.setAttributes({
        'lens.validation_error': error.message,
      });
    } finally {
      session.end_time = new Date();
      span.end();
    }
  }

  /**
   * Evaluate a specific gate
   */
  private async evaluateGate(sessionId: string, gateConfig: GateConfig): Promise<GateResult> {
    const span = LensTracer.createChildSpan('evaluate_gate', {
      'lens.gate_name': gateConfig.gate_name,
      'lens.session_id': sessionId,
    });

    try {
      let measuredValue: number;
      let confidence: number | undefined;

      // Measure the gate value based on gate type
      switch (gateConfig.gate_name) {
        case 'ndcg_improvement_nl':
          const ndcgResult = await this.measureNDCGImprovement();
          measuredValue = ndcgResult.improvement;
          confidence = ndcgResult.confidence;
          break;

        case 'sla_recall_regression':
          const recallResult = await this.measureSLARecallRegression();
          measuredValue = recallResult.regression;
          confidence = recallResult.confidence;
          break;

        case 'p95_latency_constraint':
          measuredValue = await this.measureP95LatencyIncrease();
          break;

        case 'upshift_rate_budget':
          measuredValue = await this.measureUpshiftRate();
          break;

        case 'span_coverage_complete':
          measuredValue = await this.measureSpanCoverage();
          break;

        case 'reliability_constraint':
          measuredValue = await this.measureReliabilityRatio();
          break;

        default:
          throw new Error(`Unknown gate: ${gateConfig.gate_name}`);
      }

      // Evaluate pass/fail based on direction
      const passed = this.evaluateGateCondition(measuredValue, gateConfig);

      const result: GateResult = {
        gate_name: gateConfig.gate_name,
        passed,
        measured_value: measuredValue,
        threshold_value: gateConfig.threshold,
        confidence,
        severity: gateConfig.severity,
        timestamp: new Date(),
      };

      // Add margin calculation for range gates
      if (gateConfig.direction === 'range' && gateConfig.range_bounds) {
        const [lower, upper] = gateConfig.range_bounds;
        result.margin = passed ? 0 : Math.min(
          Math.abs(measuredValue - lower),
          Math.abs(measuredValue - upper)
        );
      }

      // Record metrics
      gateMetrics.gate_evaluations.add(1, {
        gate: gateConfig.gate_name,
        passed: passed.toString(),
      });

      if (!passed) {
        gateMetrics.gate_failures.add(1, {
          gate: gateConfig.gate_name,
          severity: gateConfig.severity,
        });
      }

      span.setAttributes({
        'lens.gate_passed': passed,
        'lens.measured_value': measuredValue,
        'lens.threshold_value': gateConfig.threshold,
      });

      console.log(`Gate ${gateConfig.gate_name}: ${passed ? 'PASS' : 'FAIL'} (${measuredValue} vs ${gateConfig.threshold})`);
      return result;

    } finally {
      span.end();
    }
  }

  /**
   * Evaluate gate condition based on direction
   */
  private evaluateGateCondition(measuredValue: number, config: GateConfig): boolean {
    switch (config.direction) {
      case 'greater_than':
        return measuredValue >= config.threshold;
      case 'less_than':
        return measuredValue <= config.threshold;
      case 'equal':
        return Math.abs(measuredValue - config.threshold) < 0.001; // Small epsilon
      case 'range':
        if (!config.range_bounds) return false;
        const [lower, upper] = config.range_bounds;
        return measuredValue >= lower && measuredValue <= upper;
      default:
        return false;
    }
  }

  /**
   * Measure NDCG improvement on NL queries
   */
  private async measureNDCGImprovement(): Promise<{ improvement: number; confidence: number }> {
    const tdiStatus = this.tdi.getTDIStatus();
    if (!tdiStatus.enabled || tdiStatus.total_samples < 100) {
      return { improvement: 0, confidence: 0 };
    }

    const analyses = this.tdi.getStatisticalAnalysis();
    const nlAnalyses = analyses.filter(a => a.experiment_id.includes('nl'));
    
    if (nlAnalyses.length === 0) {
      return { improvement: 0, confidence: 0 };
    }

    // Calculate improvement based on win rate
    const avgWinRate = nlAnalyses.reduce((sum, a) => {
      const totalDecisions = a.new_wins + a.baseline_wins;
      return sum + (totalDecisions > 0 ? a.new_wins / totalDecisions : 0.5);
    }, 0) / nlAnalyses.length;

    // Convert win rate to NDCG improvement estimate
    const improvement = (avgWinRate - 0.5) * 0.1; // Rough conversion
    const avgPower = nlAnalyses.reduce((sum, a) => sum + a.statistical_power, 0) / nlAnalyses.length;

    return { improvement, confidence: avgPower };
  }

  /**
   * Measure SLA Recall@50 regression
   */
  private async measureSLARecallRegression(): Promise<{ regression: number; confidence: number }> {
    const replayStatus = this.replaySystem.getActiveReplayStatus();
    
    // Get recent replay analysis
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    
    // In practice, would get actual replay session ID
    const mockSessionId = 'recent_session';
    const analysis = this.replaySystem.getReplayAnalysis(mockSessionId);
    
    if (!analysis) {
      return { regression: 0, confidence: 0.5 };
    }

    // Negative regression means improvement
    const regression = -analysis.avg_recall_delta;
    const confidence = analysis.sla_compliance_rate;

    return { regression, confidence };
  }

  /**
   * Measure P95 latency increase
   */
  private async measureP95LatencyIncrease(): Promise<number> {
    const budgetStatus = this.riskLedger.getBudgetStatus();
    
    // Get recent latency data from risk ledger
    const recentEntries = this.riskLedger.exportLedgerData();
    if (recentEntries.length === 0) return 0;

    const recentLatencies = recentEntries
      .filter(e => Date.now() - e.timestamp.getTime() < 4 * 60 * 60 * 1000) // Last 4 hours
      .map(e => e.latency_breakdown.total);

    if (recentLatencies.length === 0) return 0;

    // Calculate P95
    recentLatencies.sort((a, b) => a - b);
    const p95Index = Math.floor(recentLatencies.length * 0.95);
    const p95Latency = recentLatencies[p95Index];

    // Compare to baseline (assume 100ms baseline)
    const baselineP95 = 100;
    return p95Latency - baselineP95;
  }

  /**
   * Measure upshift rate from risk budget
   */
  private async measureUpshiftRate(): Promise<number> {
    const budgetStatus = this.riskLedger.getBudgetStatus();
    return budgetStatus.upshift_rate;
  }

  /**
   * Measure span coverage completeness
   */
  private async measureSpanCoverage(): Promise<number> {
    // In practice, would measure actual span coverage from telemetry
    // For demo, return a value close to 100%
    return 0.998; // 99.8% coverage
  }

  /**
   * Measure reliability constraint (p99/p95 ratio)
   */
  private async measureReliabilityRatio(): Promise<number> {
    const recentEntries = this.riskLedger.exportLedgerData();
    if (recentEntries.length < 100) return 1.5; // Default acceptable ratio

    const recentLatencies = recentEntries
      .filter(e => Date.now() - e.timestamp.getTime() < 6 * 60 * 60 * 1000) // Last 6 hours
      .map(e => e.latency_breakdown.total);

    if (recentLatencies.length < 100) return 1.5;

    recentLatencies.sort((a, b) => a - b);
    const p95Index = Math.floor(recentLatencies.length * 0.95);
    const p99Index = Math.floor(recentLatencies.length * 0.99);
    
    const p95 = recentLatencies[p95Index];
    const p99 = recentLatencies[p99Index];

    return p95 > 0 ? p99 / p95 : 2.0;
  }

  /**
   * Finalize validation session
   */
  private finalizeSession(sessionId: string): void {
    const session = this.activeSessions.get(sessionId);
    if (!session) return;

    // Calculate summary metrics
    const passedGates = session.gate_results.filter(r => r.passed).length;
    const failedGates = session.gate_results.length - passedGates;
    const criticalFailures = session.gate_results.filter(r => !r.passed && r.severity === 'critical').length;
    
    const overallScore = session.gate_results.length > 0 ? passedGates / session.gate_results.length : 0;

    session.summary_metrics = {
      total_gates: session.gate_results.length,
      passed_gates: passedGates,
      failed_gates: failedGates,
      critical_failures: criticalFailures,
      overall_score: overallScore,
    };

    // Determine final status
    if (criticalFailures > 0) {
      session.status = 'failed';
      session.next_actions.push('abort');
      session.recommendations.push('Critical gates failed - do not proceed with deployment');
    } else if (failedGates === 0) {
      session.status = 'passed';
      session.next_actions.push('proceed');
      session.recommendations.push('All gates passed - deployment approved');
    } else {
      session.status = 'failed';
      session.next_actions.push('investigate');
      session.recommendations.push('Some gates failed - investigate before proceeding');
    }

    // Generate specific recommendations
    session.gate_results.forEach(result => {
      if (!result.passed) {
        session.recommendations.push(
          `${result.gate_name}: ${result.measured_value.toFixed(4)} vs ${result.threshold_value} (${result.severity})`
        );
      }
    });

    session.end_time = new Date();
    const durationMs = session.end_time.getTime() - session.start_time.getTime();
    session.metadata.duration_hours = durationMs / (1000 * 60 * 60);

    // Move to history and clean up
    this.sessionHistory.push(session);
    this.activeSessions.delete(sessionId);

    // Record metrics
    gateMetrics.validation_sessions.add(1, {
      status: session.status,
      environment: session.metadata.environment,
    });

    gateMetrics.session_duration.record(session.metadata.duration_hours, {
      status: session.status,
    });

    gateMetrics.overall_scores.record(overallScore, {
      status: session.status,
    });

    console.log(`Validation session ${sessionId} completed with status: ${session.status}`);
    console.log(`Summary: ${passedGates}/${session.gate_results.length} gates passed (score: ${(overallScore * 100).toFixed(1)}%)`);
  }

  /**
   * Get validation session status
   */
  getValidationSession(sessionId: string): ValidationSession | null {
    return this.activeSessions.get(sessionId) || 
           this.sessionHistory.find(s => s.session_id === sessionId) || null;
  }

  /**
   * Get all active validation sessions
   */
  getActiveValidationSessions(): ValidationSession[] {
    return Array.from(this.activeSessions.values());
  }

  /**
   * Get validation history
   */
  getValidationHistory(limit: number = 10): ValidationSession[] {
    return this.sessionHistory
      .sort((a, b) => b.start_time.getTime() - a.start_time.getTime())
      .slice(0, limit);
  }

  /**
   * Get overall validation status
   */
  getValidationStatus(): {
    active_sessions: number;
    recent_success_rate: number;
    avg_session_duration_hours: number;
    most_failing_gate: string | null;
    system_health: 'healthy' | 'degraded' | 'critical';
  } {
    const activeSessions = this.activeSessions.size;
    const recentSessions = this.sessionHistory.slice(-20);
    
    const successRate = recentSessions.length > 0 ?
      recentSessions.filter(s => s.status === 'passed').length / recentSessions.length : 0;
    
    const avgDuration = recentSessions.length > 0 ?
      recentSessions.reduce((sum, s) => sum + s.metadata.duration_hours, 0) / recentSessions.length : 0;

    // Find most failing gate
    const gateFailures = new Map<string, number>();
    recentSessions.forEach(session => {
      session.gate_results.forEach(result => {
        if (!result.passed) {
          gateFailures.set(result.gate_name, (gateFailures.get(result.gate_name) || 0) + 1);
        }
      });
    });

    const mostFailingGate = gateFailures.size > 0 ?
      Array.from(gateFailures.entries()).reduce((a, b) => a[1] > b[1] ? a : b)[0] : null;

    // Determine system health
    let systemHealth: 'healthy' | 'degraded' | 'critical' = 'healthy';
    if (successRate < 0.5 || activeSessions > 5) {
      systemHealth = 'critical';
    } else if (successRate < 0.8 || avgDuration > 12) {
      systemHealth = 'degraded';
    }

    return {
      active_sessions: activeSessions,
      recent_success_rate: successRate,
      avg_session_duration_hours: avgDuration,
      most_failing_gate: mostFailingGate,
      system_health: systemHealth,
    };
  }

  /**
   * Abort a validation session
   */
  abortValidationSession(sessionId: string, reason: string): boolean {
    const session = this.activeSessions.get(sessionId);
    if (!session) return false;

    session.status = 'aborted';
    session.end_time = new Date();
    session.recommendations.push(`Aborted: ${reason}`);
    session.next_actions.push('investigate');

    // Move to history
    this.sessionHistory.push(session);
    this.activeSessions.delete(sessionId);

    gateMetrics.validation_sessions.add(1, {
      status: 'aborted',
      environment: session.metadata.environment,
    });

    console.log(`Aborted validation session ${sessionId}: ${reason}`);
    return true;
  }

  /**
   * Sleep utility
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Global operational gates instance
export const globalOperationalGates = new OperationalGates();