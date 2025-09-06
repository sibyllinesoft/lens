/**
 * RAPTOR Semantic-Card Rollout Scheduler
 * 
 * Manages the phased rollout of RAPTOR semantic-card features:
 * - NL (Natural Language) strata rollout
 * - Struct (Structured) strata rollout  
 * - Mixed query handling
 * - Success criteria validation
 * - Automatic abort conditions
 * - Rollback capabilities
 */

import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';
import { EventEmitter } from 'events';

interface RolloutPhase {
  id: string;
  name: string;
  description: string;
  
  // Target strata
  target_strata: ('natural_language' | 'structured' | 'mixed')[];
  
  // Traffic configuration
  rollout_percentage: number; // 0-100
  canary_percentage?: number; // Optional canary within rollout
  
  // Timing
  duration_hours: number;
  ramp_up_hours?: number; // Gradual rollout period
  
  // Success criteria (must meet ALL)
  success_criteria: {
    min_p_at_1: number;
    min_recall_at_50: number;
    max_error_rate: number;
    max_p99_latency_ms: number;
    min_ndcg_improvement_pct?: number; // vs baseline
  };
  
  // Abort conditions (ANY triggers abort)
  abort_conditions: {
    max_p_at_1_drop_pct: number;
    max_recall_at_50_drop_pct: number;
    max_error_rate: number;
    max_p99_latency_ms: number;
    min_stability_score: number;
  };
  
  // Prerequisites
  prerequisites: string[];
  blocking_conditions: string[];
}

interface RolloutSchedule {
  // Schedule metadata
  schedule_id: string;
  created_timestamp: string;
  approved_by?: string;
  approval_timestamp?: string;
  
  // Phases
  phases: RolloutPhase[];
  current_phase_index: number;
  
  // Status
  status: 'scheduled' | 'approval_pending' | 'in_progress' | 'paused' | 'completed' | 'aborted' | 'failed';
  
  // Timing
  planned_start: string;
  actual_start?: string;
  estimated_completion: string;
  actual_completion?: string;
  
  // Success tracking
  overall_success_criteria: {
    all_phases_completed: boolean;
    stability_maintained: boolean;
    performance_maintained: boolean;
    no_critical_incidents: boolean;
  };
  
  // Rollback plan
  rollback_plan: {
    automatic_rollback_enabled: boolean;
    rollback_conditions: string[];
    rollback_duration_minutes: number;
    recovery_validation_criteria: string[];
  };
}

interface PhaseExecution {
  phase_id: string;
  
  // Execution status
  status: 'pending' | 'starting' | 'ramping_up' | 'steady_state' | 'evaluating' | 'completed' | 'aborted' | 'failed';
  
  // Timing
  start_time?: string;
  ramp_complete_time?: string;
  evaluation_start_time?: string;
  end_time?: string;
  
  // Traffic management
  current_rollout_percentage: number;
  target_percentage: number;
  traffic_split: Record<string, number>; // strata -> percentage
  
  // Metrics tracking
  baseline_metrics: {
    p_at_1: number;
    recall_at_50: number;
    ndcg_at_10: number;
    error_rate: number;
    p99_latency_ms: number;
  };
  
  current_metrics: {
    p_at_1: number;
    recall_at_50: number;
    ndcg_at_10: number;
    error_rate: number;
    p99_latency_ms: number;
    sample_size: number;
    last_updated: string;
  };
  
  // Success/abort evaluation
  success_criteria_met: boolean;
  abort_conditions_triggered: string[];
  evaluation_history: Array<{
    timestamp: string;
    criteria_met: boolean;
    abort_triggered: boolean;
    metrics_snapshot: any;
  }>;
  
  // Issues and notes
  issues_encountered: Array<{
    timestamp: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    description: string;
    resolution?: string;
  }>;
  
  operator_notes: string[];
}

interface RolloutState {
  // Current schedule
  active_schedule?: RolloutSchedule;
  
  // Phase executions
  phase_executions: Map<string, PhaseExecution>;
  
  // Global rollout control
  rollout_enabled: boolean;
  emergency_stop_active: boolean;
  maintenance_mode: boolean;
  
  // Historical schedules
  completed_schedules: RolloutSchedule[];
  
  // Configuration
  rollout_config: {
    evaluation_interval_minutes: number;
    metrics_stabilization_minutes: number;
    success_confirmation_minutes: number;
    abort_confirmation_minutes: number;
    
    // Safety limits
    max_concurrent_rollouts: number;
    min_rollback_window_hours: number;
    require_manual_approval: boolean;
  };
}

export class RAPTORSemanticRolloutScheduler extends EventEmitter {
  private readonly rolloutDir: string;
  private rolloutState: RolloutState;
  private evaluationInterval?: NodeJS.Timeout;
  private isRunning: boolean = false;
  
  constructor(rolloutDir: string = './deployment-artifacts/raptor-rollout') {
    super();
    this.rolloutDir = rolloutDir;
    
    if (!existsSync(this.rolloutDir)) {
      mkdirSync(this.rolloutDir, { recursive: true });
    }
    
    this.rolloutState = this.initializeRolloutState();
  }
  
  /**
   * Start RAPTOR rollout scheduler
   */
  public async startRolloutScheduler(): Promise<void> {
    if (this.isRunning) {
      console.log('📅 RAPTOR rollout scheduler already running');
      return;
    }
    
    console.log('🚀 Starting RAPTOR semantic-card rollout scheduler...');
    
    // Load existing state
    await this.loadRolloutState();
    
    // Start evaluation loop
    this.evaluationInterval = setInterval(async () => {
      await this.evaluateActiveRollout();
    }, this.rolloutState.rollout_config.evaluation_interval_minutes * 60 * 1000);
    
    this.isRunning = true;
    
    console.log('✅ RAPTOR rollout scheduler started');
    console.log(`🔄 Evaluation interval: ${this.rolloutState.rollout_config.evaluation_interval_minutes}min`);
    console.log('📋 Ready to schedule NL/Struct strata rollouts');
    
    this.emit('rollout_scheduler_started');
  }
  
  /**
   * Stop rollout scheduler
   */
  public stopRolloutScheduler(): void {
    if (this.evaluationInterval) {
      clearInterval(this.evaluationInterval);
      this.evaluationInterval = undefined;
    }
    
    this.isRunning = false;
    console.log('🛑 RAPTOR rollout scheduler stopped');
    this.emit('rollout_scheduler_stopped');
  }
  
  /**
   * Create rollout schedule for RAPTOR semantic-card
   */
  public createRolloutSchedule(plannedStart: string, approvalRequired: boolean = true): RolloutSchedule {
    const scheduleId = `raptor_semantic_${Date.now()}`;
    
    const schedule: RolloutSchedule = {
      schedule_id: scheduleId,
      created_timestamp: new Date().toISOString(),
      
      phases: [
        // Phase 1: NL Strata Pilot (25% traffic)
        {
          id: 'nl_pilot',
          name: 'Natural Language Strata Pilot',
          description: 'RAPTOR semantic-card for NL queries at 25% traffic',
          
          target_strata: ['natural_language'],
          rollout_percentage: 25,
          canary_percentage: 5, // Start with 5% canary
          
          duration_hours: 48,
          ramp_up_hours: 4, // Gradual ramp to 25% over 4h
          
          success_criteria: {
            min_p_at_1: 0.75,
            min_recall_at_50: 0.85,
            max_error_rate: 0.01,
            max_p99_latency_ms: 500,
            min_ndcg_improvement_pct: 2.0 // 2% improvement required
          },
          
          abort_conditions: {
            max_p_at_1_drop_pct: 5, // >5% drop triggers abort
            max_recall_at_50_drop_pct: 3, // >3% drop triggers abort
            max_error_rate: 0.02,
            max_p99_latency_ms: 1000,
            min_stability_score: 0.8
          },
          
          prerequisites: [
            'week_one_monitoring_stable',
            'no_active_breach_response',
            'raptor_clusters_healthy'
          ],
          
          blocking_conditions: [
            'cusum_alarms_active',
            'infrastructure_degraded',
            'high_pressure_backlog'
          ]
        },
        
        // Phase 2: Struct Strata Pilot (25% traffic)
        {
          id: 'struct_pilot',
          name: 'Structured Strata Pilot',
          description: 'RAPTOR semantic-card for structured queries at 25% traffic',
          
          target_strata: ['structured'],
          rollout_percentage: 25,
          canary_percentage: 5,
          
          duration_hours: 48,
          ramp_up_hours: 4,
          
          success_criteria: {
            min_p_at_1: 0.75,
            min_recall_at_50: 0.85,
            max_error_rate: 0.01,
            max_p99_latency_ms: 500,
            min_ndcg_improvement_pct: 1.5 // Structured queries may have smaller improvement
          },
          
          abort_conditions: {
            max_p_at_1_drop_pct: 5,
            max_recall_at_50_drop_pct: 3,
            max_error_rate: 0.02,
            max_p99_latency_ms: 1000,
            min_stability_score: 0.8
          },
          
          prerequisites: [
            'nl_pilot_completed_successfully',
            'no_active_incidents',
            'system_stability_maintained'
          ],
          
          blocking_conditions: [
            'nl_pilot_metrics_degraded',
            'cusum_alarms_active',
            'raptor_staleness_high'
          ]
        },
        
        // Phase 3: Full Rollout (100% traffic, all strata)
        {
          id: 'full_rollout',
          name: 'Full RAPTOR Semantic Rollout',
          description: 'RAPTOR semantic-card for all query types at 100% traffic',
          
          target_strata: ['natural_language', 'structured', 'mixed'],
          rollout_percentage: 100,
          
          duration_hours: 168, // 7 days
          ramp_up_hours: 12, // Gradual ramp to 100% over 12h
          
          success_criteria: {
            min_p_at_1: 0.75,
            min_recall_at_50: 0.85,
            max_error_rate: 0.005, // Stricter for full rollout
            max_p99_latency_ms: 400, // Stricter latency
            min_ndcg_improvement_pct: 1.0 // Overall improvement
          },
          
          abort_conditions: {
            max_p_at_1_drop_pct: 3, // Stricter for full rollout
            max_recall_at_50_drop_pct: 2,
            max_error_rate: 0.015,
            max_p99_latency_ms: 800,
            min_stability_score: 0.85
          },
          
          prerequisites: [
            'nl_pilot_completed_successfully',
            'struct_pilot_completed_successfully',
            'overall_system_health_good',
            'no_recent_incidents'
          ],
          
          blocking_conditions: [
            'any_pilot_metrics_degraded',
            'infrastructure_issues',
            'drift_detection_active'
          ]
        }
      ],
      
      current_phase_index: 0,
      status: approvalRequired ? 'approval_pending' : 'scheduled',
      
      planned_start: plannedStart,
      estimated_completion: this.calculateEstimatedCompletion(plannedStart),
      
      overall_success_criteria: {
        all_phases_completed: false,
        stability_maintained: false,
        performance_maintained: false,
        no_critical_incidents: false
      },
      
      rollback_plan: {
        automatic_rollback_enabled: true,
        rollback_conditions: [
          'abort_conditions_triggered',
          'critical_incident_detected',
          'manual_emergency_stop'
        ],
        rollback_duration_minutes: 15,
        recovery_validation_criteria: [
          'error_rate_below_baseline',
          'latency_within_normal_range',
          'no_active_alarms'
        ]
      }
    };
    
    console.log(`📋 RAPTOR semantic-card rollout schedule created: ${scheduleId}`);
    console.log(`📅 Planned start: ${plannedStart}`);
    console.log(`⏱️  Estimated completion: ${schedule.estimated_completion}`);
    console.log(`🚀 Phases: ${schedule.phases.length} (NL pilot → Struct pilot → Full rollout)`);
    
    return schedule;
  }
  
  /**
   * Schedule RAPTOR rollout (from week-one monitoring readiness)
   */
  public async scheduleRAPTORRollout(
    plannedStart?: string,
    approvalRequired: boolean = true
  ): Promise<{ schedule: RolloutSchedule; requires_approval: boolean }> {
    
    // Default to 3 days from now if not specified
    const startTime = plannedStart || new Date(Date.now() + 3 * 24 * 60 * 60 * 1000).toISOString();
    
    // Check prerequisites
    const prerequisiteCheck = await this.checkGlobalPrerequisites();
    if (!prerequisiteCheck.all_met) {
      throw new Error(`Prerequisites not met: ${prerequisiteCheck.missing.join(', ')}`);
    }
    
    // Create schedule
    const schedule = this.createRolloutSchedule(startTime, approvalRequired);
    
    // Set as active schedule
    this.rolloutState.active_schedule = schedule;
    
    // Initialize phase executions
    this.initializePhaseExecutions(schedule);
    
    // Save state
    this.saveRolloutState();
    
    console.log('✅ RAPTOR semantic-card rollout scheduled');
    
    if (approvalRequired) {
      console.log('⏳ Awaiting approval to begin rollout');
      this.emit('rollout_scheduled_pending_approval', { schedule });
    } else {
      console.log('🚀 Rollout will begin automatically at planned start time');
      this.emit('rollout_scheduled', { schedule });
    }
    
    return {
      schedule,
      requires_approval: approvalRequired
    };
  }
  
  /**
   * Approve pending rollout schedule
   */
  public approveRolloutSchedule(scheduleId: string, approvedBy: string): boolean {
    if (!this.rolloutState.active_schedule || this.rolloutState.active_schedule.schedule_id !== scheduleId) {
      console.log(`❌ Schedule ${scheduleId} not found or not active`);
      return false;
    }
    
    if (this.rolloutState.active_schedule.status !== 'approval_pending') {
      console.log(`❌ Schedule ${scheduleId} is not pending approval (status: ${this.rolloutState.active_schedule.status})`);
      return false;
    }
    
    // Approve the schedule
    this.rolloutState.active_schedule.status = 'scheduled';
    this.rolloutState.active_schedule.approved_by = approvedBy;
    this.rolloutState.active_schedule.approval_timestamp = new Date().toISOString();
    
    this.saveRolloutState();
    
    console.log(`✅ RAPTOR rollout schedule approved by ${approvedBy}`);
    console.log(`🚀 Rollout will begin at: ${this.rolloutState.active_schedule.planned_start}`);
    
    this.emit('rollout_approved', {
      schedule: this.rolloutState.active_schedule,
      approved_by: approvedBy
    });
    
    return true;
  }
  
  /**
   * Evaluate active rollout progress
   */
  private async evaluateActiveRollout(): Promise<void> {
    if (!this.rolloutState.active_schedule || this.rolloutState.active_schedule.status !== 'in_progress') {
      // Check if scheduled rollout should start
      await this.checkScheduledRolloutStart();
      return;
    }
    
    const currentPhase = this.rolloutState.active_schedule.phases[this.rolloutState.active_schedule.current_phase_index];
    const phaseExecution = this.rolloutState.phase_executions.get(currentPhase.id);
    
    if (!phaseExecution) {
      console.error(`❌ No execution found for phase ${currentPhase.id}`);
      return;
    }
    
    try {
      console.log(`🔍 Evaluating phase: ${currentPhase.name} (${phaseExecution.status})`);
      
      switch (phaseExecution.status) {
        case 'pending':
          await this.startPhase(currentPhase, phaseExecution);
          break;
          
        case 'starting':
          await this.managePhaseRampUp(currentPhase, phaseExecution);
          break;
          
        case 'ramping_up':
          await this.managePhaseRampUp(currentPhase, phaseExecution);
          break;
          
        case 'steady_state':
          await this.evaluatePhaseMetrics(currentPhase, phaseExecution);
          break;
          
        case 'evaluating':
          await this.finalizePhaseEvaluation(currentPhase, phaseExecution);
          break;
          
        case 'completed':
          await this.advanceToNextPhase();
          break;
          
        case 'aborted':
        case 'failed':
          await this.handlePhaseFailure(currentPhase, phaseExecution);
          break;
      }
      
    } catch (error) {
      console.error(`❌ Error evaluating rollout phase:`, error);
      await this.handleEvaluationError(currentPhase, phaseExecution, error);
    }
  }
  
  /**
   * Check if scheduled rollout should start
   */
  private async checkScheduledRolloutStart(): Promise<void> {
    if (!this.rolloutState.active_schedule || this.rolloutState.active_schedule.status !== 'scheduled') {
      return;
    }
    
    const now = new Date();
    const plannedStart = new Date(this.rolloutState.active_schedule.planned_start);
    
    // Check if it's time to start (within 5 minutes of planned start)
    if (now.getTime() >= plannedStart.getTime() - 5 * 60 * 1000) {
      console.log('🚀 Starting RAPTOR semantic-card rollout...');
      
      // Final prerequisite check
      const prerequisiteCheck = await this.checkGlobalPrerequisites();
      if (!prerequisiteCheck.all_met) {
        console.log(`❌ Prerequisites not met at start time: ${prerequisiteCheck.missing.join(', ')}`);
        console.log('⏳ Delaying rollout start by 30 minutes');
        
        // Delay start by 30 minutes
        this.rolloutState.active_schedule.planned_start = new Date(now.getTime() + 30 * 60 * 1000).toISOString();
        this.saveRolloutState();
        
        this.emit('rollout_start_delayed', {
          reason: 'prerequisites_not_met',
          missing: prerequisiteCheck.missing,
          new_start_time: this.rolloutState.active_schedule.planned_start
        });
        
        return;
      }
      
      // Begin rollout
      this.rolloutState.active_schedule.status = 'in_progress';
      this.rolloutState.active_schedule.actual_start = now.toISOString();
      
      // Start first phase
      const firstPhase = this.rolloutState.active_schedule.phases[0];
      const firstExecution = this.rolloutState.phase_executions.get(firstPhase.id);
      if (firstExecution) {
        firstExecution.status = 'pending';
      }
      
      this.saveRolloutState();
      
      console.log(`✅ RAPTOR semantic-card rollout started: ${firstPhase.name}`);
      
      this.emit('rollout_started', {
        schedule: this.rolloutState.active_schedule,
        first_phase: firstPhase
      });
    }
  }
  
  /**
   * Start rollout phase
   */
  private async startPhase(phase: RolloutPhase, execution: PhaseExecution): Promise<void> {
    console.log(`🚀 Starting phase: ${phase.name}`);
    
    // Check phase prerequisites
    const prereqCheck = await this.checkPhasePrerequisites(phase);
    if (!prereqCheck.all_met) {
      console.log(`⏳ Phase prerequisites not met: ${prereqCheck.missing.join(', ')}`);
      
      // Add to issues
      execution.issues_encountered.push({
        timestamp: new Date().toISOString(),
        severity: 'medium',
        description: `Prerequisites not met: ${prereqCheck.missing.join(', ')}`,
      });
      
      return; // Wait for next evaluation cycle
    }
    
    // Initialize phase execution
    execution.status = 'starting';
    execution.start_time = new Date().toISOString();
    execution.current_rollout_percentage = phase.canary_percentage || 0;
    execution.target_percentage = phase.rollout_percentage;
    
    // Capture baseline metrics
    execution.baseline_metrics = await this.captureBaselineMetrics(phase.target_strata);
    
    // Configure traffic split (mock implementation)
    execution.traffic_split = this.calculateTrafficSplit(phase.target_strata, execution.current_rollout_percentage);
    
    console.log(`📊 Baseline metrics captured for ${phase.name}`);
    console.log(`🚦 Traffic split configured: ${JSON.stringify(execution.traffic_split)}`);
    
    this.emit('phase_started', {
      phase,
      execution,
      baseline_metrics: execution.baseline_metrics
    });
  }
  
  /**
   * Manage phase ramp-up
   */
  private async managePhaseRampUp(phase: RolloutPhase, execution: PhaseExecution): Promise<void> {
    if (!execution.start_time) return;
    
    const elapsedMinutes = (Date.now() - new Date(execution.start_time).getTime()) / (60 * 1000);
    const rampUpMinutes = (phase.ramp_up_hours || 1) * 60;
    
    if (elapsedMinutes >= rampUpMinutes) {
      // Ramp-up complete
      execution.status = 'steady_state';
      execution.ramp_complete_time = new Date().toISOString();
      execution.current_rollout_percentage = execution.target_percentage;
      
      console.log(`✅ Phase ramp-up complete: ${phase.name} at ${execution.target_percentage}%`);
      
      this.emit('phase_ramp_complete', { phase, execution });
      
    } else {
      // Continue ramping up
      const rampProgress = elapsedMinutes / rampUpMinutes;
      const startPercentage = phase.canary_percentage || 0;
      execution.current_rollout_percentage = startPercentage + (execution.target_percentage - startPercentage) * rampProgress;
      
      // Update traffic split
      execution.traffic_split = this.calculateTrafficSplit(phase.target_strata, execution.current_rollout_percentage);
      
      console.log(`📈 Ramping up ${phase.name}: ${execution.current_rollout_percentage.toFixed(1)}% (${rampProgress.toFixed(1)} complete)`);
    }
  }
  
  /**
   * Evaluate phase metrics during steady state
   */
  private async evaluatePhaseMetrics(phase: RolloutPhase, execution: PhaseExecution): Promise<void> {
    // Update current metrics
    execution.current_metrics = await this.captureCurrentMetrics(phase.target_strata);
    execution.current_metrics.last_updated = new Date().toISOString();
    
    // Check abort conditions first
    const abortCheck = this.checkAbortConditions(phase, execution);
    if (abortCheck.triggered) {
      console.log(`🚨 ABORT CONDITIONS TRIGGERED for ${phase.name}: ${abortCheck.reasons.join(', ')}`);
      
      execution.status = 'aborted';
      execution.abort_conditions_triggered = abortCheck.reasons;
      execution.end_time = new Date().toISOString();
      
      this.emit('phase_aborted', {
        phase,
        execution,
        abort_reasons: abortCheck.reasons
      });
      
      // Trigger rollback
      await this.triggerRollback('abort_conditions_triggered');
      return;
    }
    
    // Check success criteria
    const successCheck = this.checkSuccessCriteria(phase, execution);
    execution.success_criteria_met = successCheck.met;
    
    // Add to evaluation history
    execution.evaluation_history.push({
      timestamp: new Date().toISOString(),
      criteria_met: successCheck.met,
      abort_triggered: false,
      metrics_snapshot: { ...execution.current_metrics }
    });
    
    // Check if phase duration completed
    const phaseDurationMs = phase.duration_hours * 60 * 60 * 1000;
    const phaseElapsedMs = execution.start_time ? Date.now() - new Date(execution.start_time).getTime() : 0;
    
    if (phaseElapsedMs >= phaseDurationMs) {
      // Phase duration complete, start final evaluation
      execution.status = 'evaluating';
      execution.evaluation_start_time = new Date().toISOString();
      
      console.log(`⏱️  Phase duration complete for ${phase.name}, starting final evaluation`);
      
      this.emit('phase_evaluation_started', { phase, execution });
    }
    
    // Log progress
    console.log(`📊 Phase ${phase.name} metrics: P@1=${execution.current_metrics.p_at_1.toFixed(3)}, R@50=${execution.current_metrics.recall_at_50.toFixed(3)}, Err=${(execution.current_metrics.error_rate * 100).toFixed(2)}%`);
  }
  
  /**
   * Finalize phase evaluation
   */
  private async finalizePhaseEvaluation(phase: RolloutPhase, execution: PhaseExecution): Promise<void> {
    const evaluationMinutes = this.rolloutState.rollout_config.success_confirmation_minutes;
    const elapsedEvaluationMs = execution.evaluation_start_time ? 
      Date.now() - new Date(execution.evaluation_start_time).getTime() : 0;
    
    if (elapsedEvaluationMs >= evaluationMinutes * 60 * 1000) {
      // Evaluation period complete
      if (execution.success_criteria_met) {
        // Phase successful
        execution.status = 'completed';
        execution.end_time = new Date().toISOString();
        
        console.log(`✅ Phase completed successfully: ${phase.name}`);
        
        this.emit('phase_completed', { phase, execution });
        
      } else {
        // Phase failed success criteria
        execution.status = 'failed';
        execution.end_time = new Date().toISOString();
        
        console.log(`❌ Phase failed success criteria: ${phase.name}`);
        
        this.emit('phase_failed', { phase, execution });
        
        // Trigger rollback
        await this.triggerRollback('success_criteria_not_met');
      }
    } else {
      console.log(`⏳ Final evaluation continuing for ${phase.name} (${(elapsedEvaluationMs / (60 * 1000)).toFixed(1)}/${evaluationMinutes}min)`);
    }
  }
  
  /**
   * Advance to next phase
   */
  private async advanceToNextPhase(): Promise<void> {
    if (!this.rolloutState.active_schedule) return;
    
    const nextPhaseIndex = this.rolloutState.active_schedule.current_phase_index + 1;
    
    if (nextPhaseIndex >= this.rolloutState.active_schedule.phases.length) {
      // All phases complete - rollout successful!
      this.rolloutState.active_schedule.status = 'completed';
      this.rolloutState.active_schedule.actual_completion = new Date().toISOString();
      
      // Update overall success criteria
      this.rolloutState.active_schedule.overall_success_criteria = {
        all_phases_completed: true,
        stability_maintained: true, // TODO: Check actual stability
        performance_maintained: true, // TODO: Check actual performance
        no_critical_incidents: true // TODO: Check incident log
      };
      
      // Move to completed schedules
      this.rolloutState.completed_schedules.push(this.rolloutState.active_schedule);
      this.rolloutState.active_schedule = undefined;
      
      console.log('🎉 RAPTOR semantic-card rollout COMPLETED SUCCESSFULLY!');
      console.log('📊 All phases completed with success criteria met');
      
      this.emit('rollout_completed', {
        schedule: this.rolloutState.completed_schedules[this.rolloutState.completed_schedules.length - 1]
      });
      
    } else {
      // Advance to next phase
      this.rolloutState.active_schedule.current_phase_index = nextPhaseIndex;
      const nextPhase = this.rolloutState.active_schedule.phases[nextPhaseIndex];
      
      console.log(`➡️  Advancing to next phase: ${nextPhase.name}`);
      
      this.emit('phase_advanced', {
        completed_phase: this.rolloutState.active_schedule.phases[nextPhaseIndex - 1],
        next_phase: nextPhase
      });
    }
    
    this.saveRolloutState();
  }
  
  /**
   * Trigger rollback
   */
  private async triggerRollback(reason: string): Promise<void> {
    if (!this.rolloutState.active_schedule) return;
    
    console.log(`🚨 TRIGGERING ROLLBACK: ${reason}`);
    
    // Mark schedule as aborted
    this.rolloutState.active_schedule.status = 'aborted';
    this.rolloutState.active_schedule.actual_completion = new Date().toISOString();
    
    // Execute rollback (mock implementation)
    const rollbackPlan = this.rolloutState.active_schedule.rollback_plan;
    
    console.log(`🔄 Executing rollback plan (${rollbackPlan.rollback_duration_minutes}min)`);
    console.log(`🔒 Disabling RAPTOR semantic-card features for all strata`);
    
    // Simulate rollback actions
    await new Promise(resolve => setTimeout(resolve, 1000)); // Mock rollback delay
    
    console.log(`✅ Rollback completed - system reverted to baseline`);
    
    // Move to completed schedules (as aborted)
    this.rolloutState.completed_schedules.push(this.rolloutState.active_schedule);
    this.rolloutState.active_schedule = undefined;
    
    this.emit('rollback_completed', {
      reason,
      rollback_duration_minutes: rollbackPlan.rollback_duration_minutes
    });
  }
  
  // Helper methods
  
  private async checkGlobalPrerequisites(): Promise<{ all_met: boolean; missing: string[] }> {
    const missing: string[] = [];
    
    // Mock prerequisite checks - in production would check actual systems
    const prerequisites = [
      'week_one_monitoring_stable',
      'no_active_breach_response', 
      'raptor_clusters_healthy',
      'system_stability_maintained'
    ];
    
    // Simulate some prerequisites being met
    const meetRate = 0.8 + Math.random() * 0.2; // 80-100% met
    const metCount = Math.floor(prerequisites.length * meetRate);
    
    for (let i = metCount; i < prerequisites.length; i++) {
      missing.push(prerequisites[i]);
    }
    
    return {
      all_met: missing.length === 0,
      missing
    };
  }
  
  private async checkPhasePrerequisites(phase: RolloutPhase): Promise<{ all_met: boolean; missing: string[] }> {
    const missing: string[] = [];
    
    // Check each prerequisite
    for (const prereq of phase.prerequisites) {
      // Mock prerequisite check
      if (Math.random() > 0.9) { // 10% chance of prerequisite not met
        missing.push(prereq);
      }
    }
    
    return {
      all_met: missing.length === 0,
      missing
    };
  }
  
  private async captureBaselineMetrics(strata: string[]): Promise<PhaseExecution['baseline_metrics']> {
    // Mock baseline metrics
    return {
      p_at_1: 0.75,
      recall_at_50: 0.85,
      ndcg_at_10: 0.78,
      error_rate: 0.005,
      p99_latency_ms: 350
    };
  }
  
  private async captureCurrentMetrics(strata: string[]): Promise<PhaseExecution['current_metrics']> {
    // Mock current metrics with some improvement
    const baselineImprovement = 0.02 + Math.random() * 0.03; // 2-5% improvement
    const noiseLevel = 0.01; // ±1% noise
    
    return {
      p_at_1: 0.75 + baselineImprovement + (Math.random() - 0.5) * noiseLevel,
      recall_at_50: 0.85 + baselineImprovement * 0.8 + (Math.random() - 0.5) * noiseLevel,
      ndcg_at_10: 0.78 + baselineImprovement * 1.2 + (Math.random() - 0.5) * noiseLevel,
      error_rate: Math.max(0.001, 0.005 + (Math.random() - 0.7) * 0.003), // Slight error rate improvement
      p99_latency_ms: Math.max(200, 350 + (Math.random() - 0.5) * 50), // Latency variation
      sample_size: Math.floor(8000 + Math.random() * 4000),
      last_updated: new Date().toISOString()
    };
  }
  
  private calculateTrafficSplit(strata: string[], percentage: number): Record<string, number> {
    const split: Record<string, number> = {};
    
    for (const stratum of strata) {
      split[stratum] = percentage;
    }
    
    return split;
  }
  
  private checkAbortConditions(phase: RolloutPhase, execution: PhaseExecution): { triggered: boolean; reasons: string[] } {
    const reasons: string[] = [];
    const current = execution.current_metrics;
    const baseline = execution.baseline_metrics;
    const conditions = phase.abort_conditions;
    
    // Check each abort condition
    if (current.p_at_1 < baseline.p_at_1 * (1 - conditions.max_p_at_1_drop_pct / 100)) {
      reasons.push(`P@1 drop: ${((baseline.p_at_1 - current.p_at_1) / baseline.p_at_1 * 100).toFixed(1)}%`);
    }
    
    if (current.recall_at_50 < baseline.recall_at_50 * (1 - conditions.max_recall_at_50_drop_pct / 100)) {
      reasons.push(`Recall@50 drop: ${((baseline.recall_at_50 - current.recall_at_50) / baseline.recall_at_50 * 100).toFixed(1)}%`);
    }
    
    if (current.error_rate > conditions.max_error_rate) {
      reasons.push(`Error rate: ${(current.error_rate * 100).toFixed(2)}%`);
    }
    
    if (current.p99_latency_ms > conditions.max_p99_latency_ms) {
      reasons.push(`P99 latency: ${current.p99_latency_ms.toFixed(0)}ms`);
    }
    
    return {
      triggered: reasons.length > 0,
      reasons
    };
  }
  
  private checkSuccessCriteria(phase: RolloutPhase, execution: PhaseExecution): { met: boolean; details: string[] } {
    const details: string[] = [];
    const current = execution.current_metrics;
    const baseline = execution.baseline_metrics;
    const criteria = phase.success_criteria;
    
    let allMet = true;
    
    if (current.p_at_1 >= criteria.min_p_at_1) {
      details.push(`✅ P@1: ${current.p_at_1.toFixed(3)} ≥ ${criteria.min_p_at_1.toFixed(3)}`);
    } else {
      details.push(`❌ P@1: ${current.p_at_1.toFixed(3)} < ${criteria.min_p_at_1.toFixed(3)}`);
      allMet = false;
    }
    
    if (current.recall_at_50 >= criteria.min_recall_at_50) {
      details.push(`✅ Recall@50: ${current.recall_at_50.toFixed(3)} ≥ ${criteria.min_recall_at_50.toFixed(3)}`);
    } else {
      details.push(`❌ Recall@50: ${current.recall_at_50.toFixed(3)} < ${criteria.min_recall_at_50.toFixed(3)}`);
      allMet = false;
    }
    
    if (current.error_rate <= criteria.max_error_rate) {
      details.push(`✅ Error rate: ${(current.error_rate * 100).toFixed(2)}% ≤ ${(criteria.max_error_rate * 100).toFixed(2)}%`);
    } else {
      details.push(`❌ Error rate: ${(current.error_rate * 100).toFixed(2)}% > ${(criteria.max_error_rate * 100).toFixed(2)}%`);
      allMet = false;
    }
    
    // Check improvement if specified
    if (criteria.min_ndcg_improvement_pct) {
      const improvementPct = ((current.ndcg_at_10 - baseline.ndcg_at_10) / baseline.ndcg_at_10) * 100;
      if (improvementPct >= criteria.min_ndcg_improvement_pct) {
        details.push(`✅ nDCG improvement: ${improvementPct.toFixed(1)}% ≥ ${criteria.min_ndcg_improvement_pct.toFixed(1)}%`);
      } else {
        details.push(`❌ nDCG improvement: ${improvementPct.toFixed(1)}% < ${criteria.min_ndcg_improvement_pct.toFixed(1)}%`);
        allMet = false;
      }
    }
    
    return { met: allMet, details };
  }
  
  private calculateEstimatedCompletion(startTime: string): string {
    // Calculate total duration across all phases
    const totalHours = [48, 48, 168].reduce((sum, hours) => sum + hours, 0); // NL + Struct + Full
    const estimatedMs = new Date(startTime).getTime() + totalHours * 60 * 60 * 1000;
    return new Date(estimatedMs).toISOString();
  }
  
  private initializePhaseExecutions(schedule: RolloutSchedule): void {
    for (const phase of schedule.phases) {
      const execution: PhaseExecution = {
        phase_id: phase.id,
        status: 'pending',
        current_rollout_percentage: 0,
        target_percentage: phase.rollout_percentage,
        traffic_split: {},
        baseline_metrics: {
          p_at_1: 0,
          recall_at_50: 0,
          ndcg_at_10: 0,
          error_rate: 0,
          p99_latency_ms: 0
        },
        current_metrics: {
          p_at_1: 0,
          recall_at_50: 0,
          ndcg_at_10: 0,
          error_rate: 0,
          p99_latency_ms: 0,
          sample_size: 0,
          last_updated: ''
        },
        success_criteria_met: false,
        abort_conditions_triggered: [],
        evaluation_history: [],
        issues_encountered: [],
        operator_notes: []
      };
      
      this.rolloutState.phase_executions.set(phase.id, execution);
    }
  }
  
  private async handlePhaseFailure(phase: RolloutPhase, execution: PhaseExecution): Promise<void> {
    console.log(`❌ Handling phase failure: ${phase.name} (${execution.status})`);
    
    // Log failure details
    execution.operator_notes.push(`Phase ${execution.status}: ${execution.abort_conditions_triggered.join(', ')}`);
    
    // Already handled by abort/failure logic
  }
  
  private async handleEvaluationError(phase: RolloutPhase, execution: PhaseExecution, error: any): Promise<void> {
    console.error(`❌ Evaluation error for ${phase.name}:`, error);
    
    execution.issues_encountered.push({
      timestamp: new Date().toISOString(),
      severity: 'high',
      description: `Evaluation error: ${error.message}`,
    });
    
    this.emit('evaluation_error', { phase, execution, error });
  }
  
  public getRolloutStatus(): RolloutState {
    return { ...this.rolloutState };
  }
  
  public getRolloutDashboardData(): any {
    return {
      timestamp: new Date().toISOString(),
      
      // Active rollout status
      active_rollout: this.rolloutState.active_schedule ? {
        schedule_id: this.rolloutState.active_schedule.schedule_id,
        status: this.rolloutState.active_schedule.status,
        current_phase: this.rolloutState.active_schedule.phases[this.rolloutState.active_schedule.current_phase_index],
        progress_pct: ((this.rolloutState.active_schedule.current_phase_index) / this.rolloutState.active_schedule.phases.length) * 100,
        planned_start: this.rolloutState.active_schedule.planned_start,
        actual_start: this.rolloutState.active_schedule.actual_start,
        estimated_completion: this.rolloutState.active_schedule.estimated_completion
      } : null,
      
      // Phase execution status
      phase_executions: Array.from(this.rolloutState.phase_executions.values()).map(exec => ({
        phase_id: exec.phase_id,
        status: exec.status,
        current_percentage: exec.current_rollout_percentage,
        success_criteria_met: exec.success_criteria_met,
        issues_count: exec.issues_encountered.length,
        start_time: exec.start_time,
        end_time: exec.end_time
      })),
      
      // Historical rollouts
      completed_rollouts: this.rolloutState.completed_schedules.length,
      
      // System status
      rollout_enabled: this.rolloutState.rollout_enabled,
      emergency_stop_active: this.rolloutState.emergency_stop_active,
      maintenance_mode: this.rolloutState.maintenance_mode
    };
  }
  
  private initializeRolloutState(): RolloutState {
    return {
      phase_executions: new Map(),
      rollout_enabled: true,
      emergency_stop_active: false,
      maintenance_mode: false,
      completed_schedules: [],
      
      rollout_config: {
        evaluation_interval_minutes: 5,
        metrics_stabilization_minutes: 30,
        success_confirmation_minutes: 60,
        abort_confirmation_minutes: 10,
        max_concurrent_rollouts: 1,
        min_rollback_window_hours: 2,
        require_manual_approval: true
      }
    };
  }
  
  private async loadRolloutState(): Promise<void> {
    const statePath = join(this.rolloutDir, 'rollout_state.json');
    if (existsSync(statePath)) {
      try {
        const data = JSON.parse(readFileSync(statePath, 'utf-8'));
        this.rolloutState = { ...this.rolloutState, ...data };
        
        // Reconstruct phase executions Map
        if (data.phase_executions) {
          this.rolloutState.phase_executions = new Map(Object.entries(data.phase_executions));
        }
        
        console.log('📋 Rollout state loaded from disk');
      } catch (error) {
        console.warn('⚠️  Failed to load rollout state:', error);
      }
    }
  }
  
  private saveRolloutState(): void {
    const statePath = join(this.rolloutDir, 'rollout_state.json');
    
    // Convert Map to object for JSON serialization
    const serializable = {
      ...this.rolloutState,
      phase_executions: Object.fromEntries(this.rolloutState.phase_executions)
    };
    
    writeFileSync(statePath, JSON.stringify(serializable, null, 2));
  }
}

export const raptorSemanticRolloutScheduler = new RAPTORSemanticRolloutScheduler();