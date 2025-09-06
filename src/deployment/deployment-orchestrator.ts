/**
 * Integrated Deployment Orchestrator
 * 
 * Master orchestration system implementing the complete TODO.md deployment plan:
 * 1. Tag + freeze: Version artifacts with config fingerprints
 * 2. Final bench: AnchorSmoke + LadderFull validation on pinned dataset
 * 3. Start canary: 3-block deployment with automatic promotion/rollback
 * 4. Online calibration: Daily reliability curve updates with safeguards
 * 5. Production monitoring: CUSUM alarms and drift detection
 * 6. Sentinel probes: Kill switches and emergency controls
 */

import { EventEmitter } from 'events';
import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';
import { versionManager } from './version-manager.js';
import { finalBenchSystem } from './final-bench-system.js';
import { canaryRolloutSystem } from './canary-rollout-system.js';
import { onlineCalibrationSystem } from './online-calibration-system.js';
import { productionMonitoringSystem } from './production-monitoring-system.js';
import { sentinelKillSwitchSystem } from './sentinel-kill-switch-system.js';

interface DeploymentPipeline {
  version: string;
  started_at: string;
  current_phase: DeploymentPhase;
  phase_history: PhaseExecution[];
  overall_status: 'running' | 'completed' | 'failed' | 'aborted';
  
  // Phase results
  tag_freeze_result?: TagFreezeResult;
  final_bench_result?: any; // BenchmarkSuite from final-bench-system
  canary_deployment_id?: string;
  calibration_active?: boolean;
  monitoring_active?: boolean;
  sentinel_active?: boolean;
}

interface PhaseExecution {
  phase: DeploymentPhase;
  started_at: string;
  completed_at?: string;
  status: 'running' | 'completed' | 'failed' | 'skipped';
  result?: any;
  error_message?: string;
  duration_ms?: number;
}

interface TagFreezeResult {
  version: string;
  config_fingerprint: string;
  artifacts_created: string[];
  baseline_metrics: any;
}

type DeploymentPhase = 
  | 'tag_freeze'
  | 'final_bench' 
  | 'canary_block_a'
  | 'canary_block_b'
  | 'canary_block_c'
  | 'online_calibration'
  | 'production_monitoring'
  | 'sentinel_activation'
  | 'ga_complete';

interface DeploymentConfig {
  // Version and baseline
  target_version?: string;
  force_new_version?: boolean;
  baseline_override?: any;
  
  // Final bench configuration
  skip_final_bench?: boolean;
  bench_timeout_minutes?: number;
  required_gate_success?: boolean;
  
  // Canary configuration
  canary_config?: {
    skip_canary?: boolean;
    accelerated_rollout?: boolean;
    stage_duration_hours?: number;
    manual_promotion?: boolean;
  };
  
  // Monitoring configuration
  monitoring_config?: {
    enable_cusum_alarms?: boolean;
    enable_drift_detection?: boolean;
    alert_webhooks?: string[];
  };
  
  // Sentinel configuration
  sentinel_config?: {
    probe_frequency_minutes?: number;
    kill_switch_enabled?: boolean;
  };
}

export class DeploymentOrchestrator extends EventEmitter {
  private readonly orchestratorDir: string;
  private currentPipeline?: DeploymentPipeline;
  private readonly defaultConfig: DeploymentConfig;
  
  constructor(orchestratorDir: string = './deployment-artifacts/orchestrator') {
    super();
    this.orchestratorDir = orchestratorDir;
    
    if (!existsSync(this.orchestratorDir)) {
      mkdirSync(this.orchestratorDir, { recursive: true });
    }
    
    this.defaultConfig = {
      required_gate_success: true,
      canary_config: {
        accelerated_rollout: false,
        stage_duration_hours: 24,
        manual_promotion: false
      },
      monitoring_config: {
        enable_cusum_alarms: true,
        enable_drift_detection: true
      },
      sentinel_config: {
        probe_frequency_minutes: 60,
        kill_switch_enabled: true
      }
    };
    
    this.setupEventListeners();
  }
  
  /**
   * Execute complete deployment pipeline per TODO.md
   */
  public async executeDeploymentPipeline(config: DeploymentConfig = {}): Promise<string> {
    const mergedConfig = { ...this.defaultConfig, ...config };
    const version = mergedConfig.target_version || this.generateNewVersion();
    
    console.log(`🚀 Starting complete deployment pipeline for version ${version}`);
    console.log('📋 Phases: Tag+Freeze → FinalBench → Canary(A→B→C) → Calibration → Monitoring → Sentinel');
    
    // Initialize pipeline
    this.currentPipeline = {
      version,
      started_at: new Date().toISOString(),
      current_phase: 'tag_freeze',
      phase_history: [],
      overall_status: 'running'
    };
    
    this.savePipelineState();
    this.emit('pipeline_started', { version, config: mergedConfig });
    
    try {
      // Phase 1: Tag + Freeze
      await this.executePhase('tag_freeze', async () => {
        return await this.executeTagFreeze(version, mergedConfig);
      });
      
      // Phase 2: Final Bench (pinned dataset validation)
      if (!mergedConfig.skip_final_bench) {
        await this.executePhase('final_bench', async () => {
          return await this.executeFinalBench(version, mergedConfig);
        });
      }
      
      // Phase 3-5: Canary rollout (Block A → B → C)
      if (!mergedConfig.canary_config?.skip_canary) {
        await this.executePhase('canary_block_a', async () => {
          return await this.executeCanaryRollout(version, mergedConfig);
        });
        
        // Wait for canary completion (blocks B and C handled by canary system)
        await this.waitForCanaryCompletion();
      }
      
      // Phase 6: Online Calibration
      await this.executePhase('online_calibration', async () => {
        return await this.activateOnlineCalibration(version, mergedConfig);
      });
      
      // Phase 7: Production Monitoring
      await this.executePhase('production_monitoring', async () => {
        return await this.activateProductionMonitoring(version, mergedConfig);
      });
      
      // Phase 8: Sentinel Activation
      await this.executePhase('sentinel_activation', async () => {
        return await this.activateSentinelSystem(version, mergedConfig);
      });
      
      // Phase 9: GA Complete
      await this.executePhase('ga_complete', async () => {
        return await this.completeGADeployment(version);
      });
      
      this.currentPipeline.overall_status = 'completed';
      this.currentPipeline.current_phase = 'ga_complete';
      
      console.log(`🎉 Deployment pipeline completed successfully for version ${version}`);
      console.log('✅ All systems active: Canary → Calibration → Monitoring → Sentinel');
      
      this.emit('pipeline_completed', { 
        version, 
        duration_ms: Date.now() - new Date(this.currentPipeline.started_at).getTime() 
      });
      
    } catch (error) {
      console.error('❌ Deployment pipeline failed:', error);
      
      this.currentPipeline.overall_status = 'failed';
      await this.handlePipelineFailure(error);
      
      this.emit('pipeline_failed', { version, error: error instanceof Error ? error.message : String(error) });
      throw error;
    } finally {
      this.savePipelineState();
    }
    
    return version;
  }
  
  /**
   * Execute individual pipeline phase with error handling
   */
  private async executePhase(phase: DeploymentPhase, executor: () => Promise<any>): Promise<void> {
    console.log(`📍 Starting phase: ${phase}`);
    
    const execution: PhaseExecution = {
      phase,
      started_at: new Date().toISOString(),
      status: 'running'
    };
    
    this.currentPipeline!.current_phase = phase;
    this.currentPipeline!.phase_history.push(execution);
    
    this.emit('phase_started', { phase, version: this.currentPipeline!.version });
    
    try {
      const startTime = Date.now();
      const result = await executor();
      const endTime = Date.now();
      
      execution.completed_at = new Date().toISOString();
      execution.status = 'completed';
      execution.result = result;
      execution.duration_ms = endTime - startTime;
      
      console.log(`✅ Phase completed: ${phase} (${execution.duration_ms}ms)`);
      
      this.emit('phase_completed', { 
        phase, 
        version: this.currentPipeline!.version,
        duration_ms: execution.duration_ms,
        result
      });
      
    } catch (error) {
      execution.completed_at = new Date().toISOString();
      execution.status = 'failed';
      execution.error_message = error instanceof Error ? error.message : String(error);
      execution.duration_ms = Date.now() - new Date(execution.started_at).getTime();
      
      console.error(`❌ Phase failed: ${phase} - ${execution.error_message}`);
      
      this.emit('phase_failed', { 
        phase, 
        version: this.currentPipeline!.version,
        error: execution.error_message
      });
      
      throw error;
    } finally {
      this.savePipelineState();
    }
  }
  
  /**
   * Phase 1: Tag + Freeze implementation
   */
  private async executeTagFreeze(version: string, config: DeploymentConfig): Promise<TagFreezeResult> {
    console.log(`🏷️  Executing Tag + Freeze for version ${version}...`);
    
    // Generate baseline metrics (mock implementation)
    const baselineMetrics = config.baseline_override || {
      p_at_1: 0.75,
      ndcg_at_10: 0.78,
      recall_at_50: 0.85,
      p95_latency_ms: 150,
      p99_latency_ms: 280,
      span_coverage: 1.0,
      results_per_query_mean: 5.2,
      results_per_query_std: 2.1
    };
    
    // Mock reliability curve
    const reliabilityCurve = [
      { predicted_score: 0.1, actual_precision: 0.12, sample_size: 100, confidence_interval: [0.08, 0.16] as [number, number] },
      { predicted_score: 0.5, actual_precision: 0.52, sample_size: 200, confidence_interval: [0.48, 0.56] as [number, number] },
      { predicted_score: 0.9, actual_precision: 0.88, sample_size: 150, confidence_interval: [0.84, 0.92] as [number, number] }
    ];
    
    // Create version with fingerprint
    const createdVersion = await versionManager.createVersion(
      0.5, // tau value
      'mock_ltr_model_hash_' + Date.now().toString(16),
      baselineMetrics,
      reliabilityCurve,
      await this.getCurrentGitCommit()
    );
    
    const configFingerprint = versionManager.calculateConfigHash(
      versionManager.loadVersionConfig(createdVersion)
    );
    
    const result: TagFreezeResult = {
      version: createdVersion,
      config_fingerprint: configFingerprint,
      artifacts_created: [
        `config_fingerprint_${createdVersion}.json`,
        `version_artifacts_${createdVersion}/`
      ],
      baseline_metrics: baselineMetrics
    };
    
    console.log(`✅ Tag + Freeze completed: version ${createdVersion}, fingerprint ${configFingerprint}`);
    
    return result;
  }
  
  /**
   * Phase 2: Final Bench implementation
   */
  private async executeFinalBench(version: string, config: DeploymentConfig): Promise<any> {
    console.log(`📊 Executing Final Bench validation for version ${version}...`);
    
    const benchResult = await finalBenchSystem.runFinalValidation(version);
    
    // Check gate requirements
    if (config.required_gate_success && !benchResult.validation_status.passed) {
      throw new Error(`Final bench gates failed: ${benchResult.validation_status.issues.join(', ')}`);
    }
    
    console.log(`✅ Final Bench completed: ${benchResult.validation_status.passed ? 'PASSED' : 'PASSED WITH WARNINGS'}`);
    console.log(`📈 Core metrics: P@1=${(benchResult.aggregate_metrics.mean_p_at_1 * 100).toFixed(1)}%, nDCG@10=${(benchResult.aggregate_metrics.mean_ndcg_at_10 * 100).toFixed(1)}%`);
    
    return benchResult;
  }
  
  /**
   * Phase 3: Canary rollout implementation
   */
  private async executeCanaryRollout(version: string, config: DeploymentConfig): Promise<string> {
    console.log(`🕯️  Starting Canary rollout for version ${version}...`);
    
    const deploymentId = await canaryRolloutSystem.startCanaryRollout(version);
    
    this.currentPipeline!.canary_deployment_id = deploymentId;
    
    console.log(`✅ Canary rollout initiated: ${deploymentId}`);
    console.log(`⏳ Canary system will handle Block A → B → C progression automatically`);
    
    return deploymentId;
  }
  
  /**
   * Wait for canary rollout completion
   */
  private async waitForCanaryCompletion(): Promise<void> {
    if (!this.currentPipeline?.canary_deployment_id) return;
    
    console.log('⏳ Waiting for canary rollout completion...');
    
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Canary rollout timeout (48 hours)'));
      }, 48 * 60 * 60 * 1000); // 48 hour timeout
      
      const checkCompletion = () => {
        const status = canaryRolloutSystem.getDeploymentStatus(this.currentPipeline!.canary_deployment_id!);
        
        if (status?.status === 'completed') {
          clearTimeout(timeout);
          console.log('✅ Canary rollout completed successfully');
          resolve();
        } else if (status?.status === 'failed') {
          clearTimeout(timeout);
          reject(new Error(`Canary rollout failed: ${status.rollback_triggers_fired.join(', ')}`));
        } else {
          // Still running, check again in 5 minutes
          setTimeout(checkCompletion, 5 * 60 * 1000);
        }
      };
      
      // Start checking
      setTimeout(checkCompletion, 60000); // Check after 1 minute
    });
  }
  
  /**
   * Phase 6: Online calibration activation
   */
  private async activateOnlineCalibration(version: string, config: DeploymentConfig): Promise<boolean> {
    console.log(`📊 Activating online calibration for version ${version}...`);
    
    await onlineCalibrationSystem.startOnlineCalibration();
    
    this.currentPipeline!.calibration_active = true;
    
    console.log('✅ Online calibration system activated');
    console.log('🔄 Daily reliability curve updates will maintain 5±2 results/query target');
    
    return true;
  }
  
  /**
   * Phase 7: Production monitoring activation
   */
  private async activateProductionMonitoring(version: string, config: DeploymentConfig): Promise<boolean> {
    console.log(`📈 Activating production monitoring for version ${version}...`);
    
    await productionMonitoringSystem.startMonitoring();
    
    this.currentPipeline!.monitoring_active = true;
    
    console.log('✅ Production monitoring system activated');
    console.log('⚠️  CUSUM alarms configured for Anchor P@1, Recall@50, and coverage metrics');
    
    return true;
  }
  
  /**
   * Phase 8: Sentinel system activation
   */
  private async activateSentinelSystem(version: string, config: DeploymentConfig): Promise<boolean> {
    console.log(`🕵️ Activating sentinel probe system for version ${version}...`);
    
    await sentinelKillSwitchSystem.startSentinelSystem();
    
    this.currentPipeline!.sentinel_active = true;
    
    console.log('✅ Sentinel system activated');
    console.log('🔍 Hourly probes: "class", "def" queries will trigger kill switches on failure');
    
    return true;
  }
  
  /**
   * Phase 9: GA deployment completion
   */
  private async completeGADeployment(version: string): Promise<any> {
    console.log(`🎊 Completing GA deployment for version ${version}...`);
    
    // Generate deployment summary
    const summary = {
      version,
      deployment_completed_at: new Date().toISOString(),
      total_duration_hours: this.getTotalDeploymentDurationHours(),
      phases_completed: this.currentPipeline!.phase_history.filter(p => p.status === 'completed').length,
      systems_active: {
        canary_rollout: true,
        online_calibration: this.currentPipeline!.calibration_active || false,
        production_monitoring: this.currentPipeline!.monitoring_active || false,
        sentinel_probes: this.currentPipeline!.sentinel_active || false
      },
      next_actions: [
        'Monitor drift alarms for sustained deviation',
        'Validate 5±2 results/query target maintenance',
        'Confirm sentinel probes passing hourly',
        'Review canary metrics for optimization opportunities'
      ]
    };
    
    // Save GA completion record
    const summaryPath = join(this.orchestratorDir, `ga_completion_${version}.json`);
    writeFileSync(summaryPath, JSON.stringify(summary, null, 2));
    
    console.log('🏁 GA deployment complete - all systems operational');
    console.log(`📊 Total deployment duration: ${summary.total_duration_hours.toFixed(1)} hours`);
    
    return summary;
  }
  
  /**
   * Handle pipeline failure with cleanup
   */
  private async handlePipelineFailure(error: any): Promise<void> {
    console.error('🚨 Handling deployment pipeline failure...');
    
    const currentPhase = this.currentPipeline?.current_phase;
    
    // Phase-specific cleanup
    switch (currentPhase) {
      case 'canary_block_a':
      case 'canary_block_b':
      case 'canary_block_c':
        // Rollback canary if it was started
        if (this.currentPipeline?.canary_deployment_id) {
          try {
            await canaryRolloutSystem.manualRollback(
              this.currentPipeline.canary_deployment_id,
              'Pipeline failure cleanup'
            );
          } catch (rollbackError) {
            console.error('❌ Failed to rollback canary:', rollbackError);
          }
        }
        break;
        
      case 'online_calibration':
        // Stop calibration if it was started
        onlineCalibrationSystem.stopOnlineCalibration();
        break;
        
      case 'production_monitoring':
        // Stop monitoring if it was started
        productionMonitoringSystem.stopMonitoring();
        break;
        
      case 'sentinel_activation':
        // Stop sentinel if it was started
        sentinelKillSwitchSystem.stopSentinelSystem();
        break;
    }
    
    // Generate failure report
    const failureReport = {
      version: this.currentPipeline?.version,
      failed_at: new Date().toISOString(),
      failed_phase: currentPhase,
      error_message: error instanceof Error ? error.message : String(error),
      phases_completed: this.currentPipeline?.phase_history.filter(p => p.status === 'completed').length || 0,
      cleanup_actions_performed: true
    };
    
    const failurePath = join(this.orchestratorDir, `failure_report_${Date.now()}.json`);
    writeFileSync(failurePath, JSON.stringify(failureReport, null, 2));
    
    console.log(`💾 Failure report saved: ${failurePath}`);
  }
  
  /**
   * Setup event listeners for subsystems
   */
  private setupEventListeners(): void {
    // Canary system events
    canaryRolloutSystem.on('block_started', (data) => {
      console.log(`🕯️  Canary block started: ${data.block}`);
      this.emit('canary_block_started', data);
    });
    
    canaryRolloutSystem.on('block_rolled_back', (data) => {
      console.log(`🚨 Canary block rolled back: ${data.block}`);
      this.emit('canary_rollback', data);
    });
    
    canaryRolloutSystem.on('canary_completed', (data) => {
      console.log(`🎉 Canary rollout completed: ${data.deploymentId}`);
      this.emit('canary_completed', data);
    });
    
    // Monitoring system events
    productionMonitoringSystem.on('cusum_alarm_triggered', (data) => {
      console.log(`⚠️  CUSUM alarm: ${data.metric}`);
      this.emit('drift_alarm', data);
    });
    
    productionMonitoringSystem.on('kill_switch_activated', (data) => {
      console.log(`🚨 Kill switch activated: ${data.trigger}`);
      this.emit('emergency_activated', data);
    });
    
    // Sentinel system events
    sentinelKillSwitchSystem.on('probe_executed', (data) => {
      if (!data.success) {
        console.log(`❌ Sentinel probe failed: ${data.probe_name}`);
      }
      this.emit('sentinel_probe_result', data);
    });
    
    sentinelKillSwitchSystem.on('kill_switch_activated', (data) => {
      console.log(`🚨 Sentinel kill switch: ${data.switch_name}`);
      this.emit('sentinel_emergency', data);
    });
    
    // Calibration system events
    onlineCalibrationSystem.on('calibration_updated', (data) => {
      console.log(`📊 Calibration updated: tau=${data.tau.toFixed(3)}`);
      this.emit('calibration_update', data);
    });
    
    onlineCalibrationSystem.on('ltr_fallback_activated', (data) => {
      console.log(`🚨 LTR fallback activated: ${data.reason}`);
      this.emit('calibration_fallback', data);
    });
  }
  
  // Helper methods
  
  private generateNewVersion(): string {
    const currentVersion = versionManager.getCurrentVersion();
    const parts = currentVersion.split('.').map(Number);
    parts[2]++; // Increment patch version
    return parts.join('.');
  }
  
  private async getCurrentGitCommit(): Promise<string> {
    try {
      const { execSync } = require('child_process');
      return execSync('git rev-parse HEAD', { encoding: 'utf-8' }).trim();
    } catch {
      return 'unknown';
    }
  }
  
  private getTotalDeploymentDurationHours(): number {
    if (!this.currentPipeline) return 0;
    return (Date.now() - new Date(this.currentPipeline.started_at).getTime()) / (60 * 60 * 1000);
  }
  
  private savePipelineState(): void {
    if (!this.currentPipeline) return;
    
    const statePath = join(this.orchestratorDir, `pipeline_${this.currentPipeline.version}.json`);
    writeFileSync(statePath, JSON.stringify(this.currentPipeline, null, 2));
    
    // Also save as current pipeline
    const currentPath = join(this.orchestratorDir, 'current_pipeline.json');
    writeFileSync(currentPath, JSON.stringify(this.currentPipeline, null, 2));
  }
  
  /**
   * Get current pipeline status
   */
  public getCurrentPipelineStatus(): DeploymentPipeline | null {
    return this.currentPipeline ? { ...this.currentPipeline } : null;
  }
  
  /**
   * Load pipeline from saved state
   */
  public loadPipeline(version: string): DeploymentPipeline | null {
    const statePath = join(this.orchestratorDir, `pipeline_${version}.json`);
    if (!existsSync(statePath)) return null;
    
    try {
      return JSON.parse(readFileSync(statePath, 'utf-8'));
    } catch {
      return null;
    }
  }
  
  /**
   * Emergency abort current pipeline
   */
  public async emergencyAbortPipeline(reason: string): Promise<void> {
    if (!this.currentPipeline || this.currentPipeline.overall_status !== 'running') {
      console.log('⚠️  No active pipeline to abort');
      return;
    }
    
    console.log(`🚨 EMERGENCY PIPELINE ABORT: ${reason}`);
    
    this.currentPipeline.overall_status = 'aborted';
    
    try {
      await this.handlePipelineFailure(new Error(`Emergency abort: ${reason}`));
      
      this.emit('pipeline_aborted', {
        version: this.currentPipeline.version,
        reason,
        aborted_at: new Date().toISOString()
      });
      
    } catch (error) {
      console.error('❌ Error during emergency abort:', error);
    } finally {
      this.savePipelineState();
    }
  }
  
  /**
   * Get deployment dashboard data
   */
  public getDashboardData(): any {
    const pipeline = this.currentPipeline;
    
    return {
      timestamp: new Date().toISOString(),
      current_pipeline: pipeline ? {
        version: pipeline.version,
        status: pipeline.overall_status,
        current_phase: pipeline.current_phase,
        started_at: pipeline.started_at,
        duration_hours: pipeline ? this.getTotalDeploymentDurationHours() : 0,
        phases_completed: pipeline.phase_history.filter(p => p.status === 'completed').length,
        phases_total: 9 // Total phases in pipeline
      } : null,
      
      system_status: {
        canary_active: pipeline?.canary_deployment_id ? true : false,
        calibration_active: pipeline?.calibration_active || false,
        monitoring_active: pipeline?.monitoring_active || false,
        sentinel_active: pipeline?.sentinel_active || false
      },
      
      recent_events: this.getRecentEvents(),
      next_actions: this.getNextActions()
    };
  }
  
  private getRecentEvents(): any[] {
    // Mock recent events - in production would maintain event log
    return [
      { timestamp: new Date().toISOString(), type: 'info', message: 'Sentinel probes passing' },
      { timestamp: new Date(Date.now() - 5 * 60 * 1000).toISOString(), type: 'success', message: 'Calibration updated: tau=0.523' },
      { timestamp: new Date(Date.now() - 15 * 60 * 1000).toISOString(), type: 'info', message: 'CUSUM alarms quiet' }
    ];
  }
  
  private getNextActions(): string[] {
    if (!this.currentPipeline) {
      return ['Ready to start new deployment pipeline'];
    }
    
    switch (this.currentPipeline.current_phase) {
      case 'tag_freeze':
        return ['Completing version tagging and config freeze'];
      case 'final_bench':
        return ['Running AnchorSmoke + LadderFull validation'];
      case 'canary_block_a':
        return ['Monitoring Block A canary (5% → 25% → 100%)'];
      case 'canary_block_b':
        return ['Monitoring Block B canary (dynamic topn)'];
      case 'canary_block_c':
        return ['Monitoring Block C canary (deduplication)'];
      case 'online_calibration':
        return ['Activating daily reliability curve updates'];
      case 'production_monitoring':
        return ['Setting up CUSUM drift detection'];
      case 'sentinel_activation':
        return ['Configuring hourly sentinel probes'];
      case 'ga_complete':
        return ['Monitoring production stability'];
      default:
        return ['Unknown phase'];
    }
  }
}

export const deploymentOrchestrator = new DeploymentOrchestrator();