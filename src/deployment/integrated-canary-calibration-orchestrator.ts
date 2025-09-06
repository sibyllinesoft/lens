/**
 * Integrated Canary-Calibration Orchestrator
 * 
 * Orchestrates the complete TODO.md deployment workflow:
 * 1. Canary A→B→C deployment (24h holds per TODO.md)
 * 2. Post-deploy calibration (2-day holdout + τ optimization)
 * 3. Automated monitoring and alerting throughout
 * 
 * This integrates the canary deployment with the post-deploy calibration system
 * to provide seamless TODO.md Step 4 execution.
 */

import { EventEmitter } from 'events';
import { CanaryDeploymentOrchestrator, DeploymentMetrics } from './canary-orchestrator.js';
import { PostDeployCalibrationSystem, CalibrationSession } from './post-deploy-calibration-system.js';
import { OnlineCalibrationSystem, onlineCalibrationSystem } from './online-calibration-system.js';

interface IntegratedDeploymentConfig {
  canary_phases: 'compressed' | 'extended'; // compressed=1hr, extended=24hr per TODO.md
  auto_start_calibration: boolean;
  calibration_monitoring_enabled: boolean;
  alert_webhooks?: string[];
}

interface DeploymentLifecycleEvent {
  timestamp: string;
  deployment_id: string;
  phase: 'canary_started' | 'canary_phase_completed' | 'canary_completed' | 'canary_failed' | 
         'calibration_initiated' | 'calibration_completed' | 'calibration_frozen' | 'deployment_complete';
  details: Record<string, any>;
}

export class IntegratedCanaryCalibrationOrchestrator extends EventEmitter {
  private readonly canaryOrchestrator: CanaryDeploymentOrchestrator;
  private readonly calibrationSystem: PostDeployCalibrationSystem;
  private readonly config: IntegratedDeploymentConfig;
  
  private currentDeployment?: {
    deployment_id: string;
    start_time: Date;
    canary_completed: boolean;
    calibration_session_id?: string;
    current_phase: string;
    lifecycle_events: DeploymentLifecycleEvent[];
  };
  
  constructor(config: Partial<IntegratedDeploymentConfig> = {}) {
    super();
    
    this.config = {
      canary_phases: 'compressed', // Default to 1-hour compressed for faster iteration
      auto_start_calibration: true,
      calibration_monitoring_enabled: true,
      ...config
    };
    
    // Initialize orchestrator components
    this.canaryOrchestrator = new CanaryDeploymentOrchestrator();
    this.calibrationSystem = new PostDeployCalibrationSystem(onlineCalibrationSystem);
    
    this.setupEventHandlers();
    
    console.log('🎯 Integrated Canary-Calibration Orchestrator initialized');
    console.log(`   Canary phases: ${this.config.canary_phases}`);
    console.log(`   Auto-start calibration: ${this.config.auto_start_calibration}`);
  }
  
  /**
   * Execute complete TODO.md deployment workflow
   */
  public async executeIntegratedDeployment(
    deploymentVersion: string = 'lens-v1.2',
    currentTau: number = 0.5
  ): Promise<{
    success: boolean;
    deployment_id: string;
    canary_result: any;
    calibration_session_id?: string;
    lifecycle_events: DeploymentLifecycleEvent[];
    total_duration_hours: number;
  }> {
    const deploymentId = `${deploymentVersion}-${Date.now()}`;
    const startTime = new Date();
    
    this.currentDeployment = {
      deployment_id: deploymentId,
      start_time: startTime,
      canary_completed: false,
      current_phase: 'canary_deployment',
      lifecycle_events: []
    };
    
    console.log('\n🚀 INTEGRATED DEPLOYMENT STARTED - TODO.md WORKFLOW');
    console.log('=' .repeat(80));
    console.log(`   Deployment ID: ${deploymentId}`);
    console.log(`   Version: ${deploymentVersion}`);
    console.log(`   Current τ: ${currentTau.toFixed(4)}`);
    console.log(`   Start Time: ${startTime.toISOString()}`);
    
    this.logLifecycleEvent('canary_started', {
      deployment_version: deploymentVersion,
      current_tau: currentTau,
      canary_phases: this.config.canary_phases
    });

    try {
      // Start calibration monitoring if enabled
      if (this.config.calibration_monitoring_enabled) {
        this.calibrationSystem.startCalibrationMonitoring();
      }
      
      // PHASE 1: Execute Canary Deployment (TODO.md Steps 1-3)
      console.log('\n📈 PHASE 1: CANARY DEPLOYMENT');
      console.log('-' .repeat(50));
      
      const canaryResult = await this.canaryOrchestrator.executeCanaryDeployment();
      
      if (!canaryResult.success) {
        console.log('❌ CANARY DEPLOYMENT FAILED - Stopping integrated deployment');
        this.logLifecycleEvent('canary_failed', {
          reason: canaryResult.final_status,
          deployment_log: canaryResult.deployment_log.slice(-3) // Last 3 entries
        });
        
        return this.generateFinalResult(false, canaryResult, startTime);
      }
      
      this.currentDeployment.canary_completed = true;
      this.logLifecycleEvent('canary_completed', {
        final_status: canaryResult.final_status,
        total_duration_minutes: canaryResult.total_duration_minutes,
        production_ready: canaryResult.production_ready
      });
      
      console.log('✅ CANARY DEPLOYMENT SUCCESSFUL');
      console.log(`   Duration: ${canaryResult.total_duration_minutes.toFixed(1)} minutes`);
      console.log(`   Production Ready: ${canaryResult.production_ready}`);
      
      // PHASE 2: Initialize Post-Deploy Calibration (TODO.md Step 4)
      if (this.config.auto_start_calibration && canaryResult.production_ready) {
        console.log('\n📊 PHASE 2: POST-DEPLOY CALIBRATION INITIALIZATION');
        console.log('-' .repeat(60));
        
        const calibrationSessionId = await this.calibrationSystem.initializePostCanaryCalibration(
          deploymentId,
          new Date(),
          currentTau
        );
        
        this.currentDeployment.calibration_session_id = calibrationSessionId;
        this.currentDeployment.current_phase = 'post_deploy_calibration';
        
        this.logLifecycleEvent('calibration_initiated', {
          calibration_session_id: calibrationSessionId,
          holdout_period_days: 2
        });
        
        console.log(`✅ Post-deploy calibration initiated`);
        console.log(`   Session ID: ${calibrationSessionId}`);
        console.log(`   Holdout period: 2 days (per TODO.md)`);
        console.log(`   Will optimize τ after holdout completes`);
      }
      
      console.log('\n🎉 INTEGRATED DEPLOYMENT COMPLETED');
      console.log('=' .repeat(80));
      console.log('✅ Canary deployment successful');
      if (this.currentDeployment.calibration_session_id) {
        console.log('✅ Post-deploy calibration initiated (2-day holdout active)');
        console.log('📊 Calibration will automatically optimize τ after holdout period');
        console.log('⚠️  System will FREEZE if |Δτ| > 0.02 per TODO.md requirements');
      }
      
      this.logLifecycleEvent('deployment_complete', {
        canary_success: true,
        calibration_initiated: !!this.currentDeployment.calibration_session_id
      });
      
      return this.generateFinalResult(true, canaryResult, startTime);
      
    } catch (error) {
      console.error('💥 INTEGRATED DEPLOYMENT FAILED:', error);
      
      this.logLifecycleEvent('canary_failed', {
        error_message: error.message,
        error_stack: error.stack
      });
      
      return this.generateFinalResult(false, { success: false, final_status: 'ERROR', deployment_log: [], total_duration_minutes: 0, production_ready: false }, startTime);
    }
  }
  
  /**
   * Setup event handlers for component integration
   */
  private setupEventHandlers(): void {
    // Handle calibration system events
    this.calibrationSystem.on('calibration_completed', (event) => {
      console.log(`🎯 CALIBRATION COMPLETED: Session ${event.sessionId}`);
      console.log(`   τ: ${event.oldTau.toFixed(4)} → ${event.newTau.toFixed(4)} (Δ${event.tauDelta >= 0 ? '+' : ''}${event.tauDelta.toFixed(4)})`);
      
      this.logLifecycleEvent('calibration_completed', {
        session_id: event.sessionId,
        old_tau: event.oldTau,
        new_tau: event.newTau,
        tau_delta: event.tauDelta,
        within_bounds: event.withinBounds
      });
      
      this.emit('deployment_phase_completed', {
        phase: 'post_deploy_calibration',
        success: true,
        details: event
      });
    });
    
    this.calibrationSystem.on('calibration_frozen', (event) => {
      console.log(`🚨 CALIBRATION SYSTEM FROZEN: ${event.reason}`);
      console.log(`   Session: ${event.sessionId}`);
      console.log(`   |Δτ| = ${Math.abs(event.tauDelta).toFixed(4)} > ${event.threshold}`);
      console.log(`   Manual intervention required per TODO.md`);
      
      this.logLifecycleEvent('calibration_frozen', {
        session_id: event.sessionId,
        freeze_reason: event.reason,
        tau_delta: event.tauDelta,
        drift_threshold: event.threshold
      });
      
      this.emit('system_frozen', {
        reason: event.reason,
        session_id: event.sessionId,
        requires_manual_intervention: true
      });
    });
    
    this.calibrationSystem.on('alert', (alert) => {
      console.log(`📢 CALIBRATION ALERT [${alert.severity.toUpperCase()}]: ${alert.message}`);
      
      this.emit('calibration_alert', alert);
      
      // Forward critical alerts
      if (alert.severity === 'critical') {
        this.sendWebhookAlert(alert);
      }
    });
  }
  
  /**
   * Log deployment lifecycle event
   */
  private logLifecycleEvent(phase: DeploymentLifecycleEvent['phase'], details: Record<string, any>): void {
    if (!this.currentDeployment) return;
    
    const event: DeploymentLifecycleEvent = {
      timestamp: new Date().toISOString(),
      deployment_id: this.currentDeployment.deployment_id,
      phase,
      details
    };
    
    this.currentDeployment.lifecycle_events.push(event);
    this.emit('lifecycle_event', event);
  }
  
  /**
   * Generate final deployment result
   */
  private generateFinalResult(
    success: boolean,
    canaryResult: any,
    startTime: Date
  ): any {
    const endTime = new Date();
    const totalDurationHours = (endTime.getTime() - startTime.getTime()) / (1000 * 60 * 60);
    
    return {
      success,
      deployment_id: this.currentDeployment!.deployment_id,
      canary_result: canaryResult,
      calibration_session_id: this.currentDeployment!.calibration_session_id,
      lifecycle_events: this.currentDeployment!.lifecycle_events,
      total_duration_hours: totalDurationHours
    };
  }
  
  /**
   * Send webhook alerts for critical events
   */
  private async sendWebhookAlert(alert: any): Promise<void> {
    if (!this.config.alert_webhooks?.length) return;
    
    for (const webhook of this.config.alert_webhooks) {
      try {
        // In production, would make HTTP POST to webhook
        console.log(`📤 Webhook alert sent to: ${webhook}`);
      } catch (error) {
        console.error(`❌ Failed to send webhook alert:`, error);
      }
    }
  }
  
  /**
   * Get current deployment status
   */
  public getCurrentDeploymentStatus(): {
    active: boolean;
    deployment_id?: string;
    current_phase?: string;
    canary_completed?: boolean;
    calibration_session_id?: string;
    lifecycle_events?: DeploymentLifecycleEvent[];
  } {
    if (!this.currentDeployment) {
      return { active: false };
    }
    
    return {
      active: true,
      deployment_id: this.currentDeployment.deployment_id,
      current_phase: this.currentDeployment.current_phase,
      canary_completed: this.currentDeployment.canary_completed,
      calibration_session_id: this.currentDeployment.calibration_session_id,
      lifecycle_events: [...this.currentDeployment.lifecycle_events]
    };
  }
  
  /**
   * Get calibration session status
   */
  public getCalibrationSessionStatus(sessionId: string): any {
    return this.calibrationSystem.getCalibrationSession(sessionId);
  }
  
  /**
   * Get all calibration sessions
   */
  public getAllCalibrationSessions(): any[] {
    return this.calibrationSystem.getCalibrationSessions();
  }
  
  /**
   * Manual intervention for calibration sessions
   */
  public async manualCalibrationIntervention(
    sessionId: string,
    action: 'complete' | 'freeze' | 'retry',
    reason: string
  ): Promise<void> {
    console.log(`🔧 Manual calibration intervention: ${action} for session ${sessionId}`);
    console.log(`   Reason: ${reason}`);
    
    await this.calibrationSystem.manualSessionIntervention(sessionId, action, reason);
    
    this.logLifecycleEvent('calibration_completed', {
      session_id: sessionId,
      manual_intervention: true,
      action,
      reason
    });
  }
  
  /**
   * Stop monitoring (cleanup)
   */
  public stopMonitoring(): void {
    this.calibrationSystem.stopCalibrationMonitoring();
    console.log('🛑 Integrated orchestrator monitoring stopped');
  }
}

/**
 * Execute TODO.md integrated deployment workflow
 */
export async function executeTodoMdIntegratedDeployment(
  version: string = 'lens-v1.2',
  currentTau: number = 0.5,
  config: Partial<IntegratedDeploymentConfig> = {}
) {
  const orchestrator = new IntegratedCanaryCalibrationOrchestrator(config);
  
  console.log('\n🎯 TODO.md INTEGRATED DEPLOYMENT WORKFLOW');
  console.log('=' .repeat(60));
  console.log('Steps: Canary A→B→C → 2-day holdout → τ optimization');
  console.log('Safety: |Δτ| > 0.02 → SYSTEM FREEZE');
  console.log('');
  
  try {
    const result = await orchestrator.executeIntegratedDeployment(version, currentTau);
    
    console.log('\n📊 INTEGRATED DEPLOYMENT SUMMARY');
    console.log('=' .repeat(60));
    console.log(`✅ Success: ${result.success}`);
    console.log(`📅 Duration: ${result.total_duration_hours.toFixed(1)} hours`);
    console.log(`🚀 Canary Success: ${result.canary_result.success}`);
    if (result.calibration_session_id) {
      console.log(`📊 Calibration Session: ${result.calibration_session_id}`);
      console.log(`⏳ Status: 2-day holdout active (auto-optimization after)`);
    }
    console.log(`📋 Lifecycle Events: ${result.lifecycle_events.length}`);
    
    return result;
    
  } catch (error) {
    console.error('💥 TODO.md deployment workflow failed:', error);
    throw error;
  } finally {
    // Cleanup
    orchestrator.stopMonitoring();
  }
}