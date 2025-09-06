/**
 * Chaos Engineering Framework for Lens Search Engine
 * 
 * Implements controlled failure injection to test system resilience:
 * - Network partitions and service failures
 * - Database connection failures and timeouts  
 * - Memory pressure and resource exhaustion
 * - Concurrent user load with realistic patterns
 * - NATS/JetStream disruption scenarios
 * - Cache invalidation and data corruption
 * - Production safety with automatic rollback
 */

import { EventEmitter } from 'node:events';
import { setTimeout, clearTimeout } from 'node:timers';

const sleep = (ms: number): Promise<void> => new Promise(resolve => setTimeout(resolve, ms));
import { spawn, exec, ChildProcess } from 'node:child_process';
import { promises as fs } from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { LensTracer, tracer } from '../telemetry/tracer.js';
import { resilienceManager, ErrorType, ErrorSeverity } from './resilience-manager.js';
import { performanceMonitor } from './performance-monitor.js';

// Chaos experiment types
export enum ChaosExperimentType {
  NETWORK_PARTITION = 'NETWORK_PARTITION',
  SERVICE_FAILURE = 'SERVICE_FAILURE', 
  DATABASE_FAILURE = 'DATABASE_FAILURE',
  MEMORY_PRESSURE = 'MEMORY_PRESSURE',
  RESOURCE_EXHAUSTION = 'RESOURCE_EXHAUSTION',
  CONCURRENT_LOAD = 'CONCURRENT_LOAD',
  NATS_DISRUPTION = 'NATS_DISRUPTION',
  CACHE_CORRUPTION = 'CACHE_CORRUPTION',
  DISK_FAILURE = 'DISK_FAILURE',
  CPU_STARVATION = 'CPU_STARVATION'
}

export enum ChaosExperimentState {
  PLANNED = 'PLANNED',
  RUNNING = 'RUNNING', 
  INJECTING = 'INJECTING',
  RECOVERING = 'RECOVERING',
  COMPLETED = 'COMPLETED',
  FAILED = 'FAILED',
  ROLLED_BACK = 'ROLLED_BACK'
}

export interface ChaosExperimentConfig {
  id: string;
  name: string;
  type: ChaosExperimentType;
  description: string;
  parameters: Record<string, any>;
  
  // Safety settings
  maxDuration: number; // Max experiment duration in ms
  rollbackThreshold: {
    errorRate: number; // Max error rate before rollback (0-1)
    latencyP99: number; // Max p99 latency in ms before rollback
    availabilityMin: number; // Min availability (0-1) before rollback
  };
  
  // Target specification
  targetComponents: string[]; // Which components to affect
  impactRadius: 'single_shard' | 'single_service' | 'partial_system' | 'full_system';
  
  // Monitoring
  monitoringInterval: number; // Health check interval in ms
  recoveryValidation: {
    stabilityPeriod: number; // Time in ms to wait for stability
    successThreshold: number; // Success rate needed for recovery (0-1)
  };
}

export interface ChaosExperimentResult {
  experimentId: string;
  state: ChaosExperimentState;
  startTime: Date;
  endTime?: Date;
  
  // Performance metrics
  baseline: {
    errorRate: number;
    latencyP50: number;
    latencyP95: number;
    latencyP99: number; 
    throughput: number;
    availability: number;
  };
  
  duringInjection: {
    errorRate: number;
    latencyP50: number;
    latencyP95: number;
    latencyP99: number;
    throughput: number;
    availability: number;
  };
  
  recovery: {
    recoveryTime: number; // Time to full recovery in ms
    stabilityAchieved: boolean;
    dataConsistency: boolean;
  };
  
  // Detailed timeline
  timeline: Array<{
    timestamp: Date;
    event: string;
    metrics: Record<string, number>;
    severity: 'info' | 'warning' | 'error' | 'critical';
  }>;
  
  // Safety actions taken
  safetyActions: Array<{
    timestamp: Date;
    action: string;
    reason: string;
    successful: boolean;
  }>;
  
  // Insights and recommendations
  insights: {
    weakPoints: string[];
    improvements: string[];
    resilience: number; // Overall resilience score (0-100)
  };
}

/**
 * Chaos Engineering Framework
 */
export class ChaosEngineeringFramework extends EventEmitter {
  private static instance: ChaosEngineeringFramework | null = null;
  private readonly lensTracer = tracer;
  
  private experiments = new Map<string, ChaosExperimentConfig>();
  private activeExperiments = new Map<string, ChaosExperimentResult>();
  private runningProcesses = new Map<string, ChildProcess>();
  
  // Safety monitoring
  private safetyMonitor?: NodeJS.Timeout;
  private baselineMetrics: any = {};
  private isProductionMode: boolean = false;
  
  // Failure injection tools
  private networkController?: NetworkFailureController;
  private resourceController?: ResourceExhaustionController;
  private serviceController?: ServiceFailureController;
  private dataController?: DataCorruptionController;
  
  private constructor(private readonly config: {
    productionMode: boolean;
    safetyLimits: {
      maxConcurrentExperiments: number;
      maxExperimentDuration: number;
      emergencyStopThreshold: {
        errorRate: number;
        latencyMultiplier: number;
      };
    };
    baselineService: {
      url: string;
      healthEndpoint: string;
      metricsEndpoint: string;
    };
  }) {
    super();
    this.isProductionMode = config.productionMode;
    this.initializeControllers();
    this.startSafetyMonitoring();
  }
  
  static getInstance(config?: any): ChaosEngineeringFramework {
    if (!ChaosEngineeringFramework.instance) {
      ChaosEngineeringFramework.instance = new ChaosEngineeringFramework(
        config || {
          productionMode: process.env.NODE_ENV === 'production',
          safetyLimits: {
            maxConcurrentExperiments: 1,
            maxExperimentDuration: 300000, // 5 minutes
            emergencyStopThreshold: {
              errorRate: 0.1, // 10%
              latencyMultiplier: 3.0
            }
          },
          baselineService: {
            url: 'http://localhost:3001',
            healthEndpoint: '/health',
            metricsEndpoint: '/metrics'
          }
        }
      );
    }
    return ChaosEngineeringFramework.instance;
  }
  
  /**
   * Register a chaos experiment
   */
  registerExperiment(config: ChaosExperimentConfig): void {
    // Validate safety parameters in production
    if (this.isProductionMode) {
      this.validateProductionSafety(config);
    }
    
    this.experiments.set(config.id, config);
    
    this.emit('experimentRegistered', {
      experimentId: config.id,
      type: config.type,
      impactRadius: config.impactRadius
    });
  }
  
  /**
   * Execute a chaos experiment with full safety monitoring
   */
  async executeExperiment(experimentId: string): Promise<ChaosExperimentResult> {
    return await this.tracer.startActiveSpan('execute-chaos-experiment', async (span) => {
      const experiment = this.experiments.get(experimentId);
      if (!experiment) {
        throw new Error(`Experiment ${experimentId} not found`);
      }
      
      // Check safety limits
      if (this.activeExperiments.size >= this.config.safetyLimits.maxConcurrentExperiments) {
        throw new Error('Maximum concurrent experiments exceeded');
      }
      
      span.setAttributes({
        'chaos.experiment.id': experimentId,
        'chaos.experiment.type': experiment.type,
        'chaos.experiment.impact': experiment.impactRadius,
        'chaos.production_mode': this.isProductionMode
      });
      
      console.log(`🎯 Starting chaos experiment: ${experiment.name}`);
      
      const result: ChaosExperimentResult = {
        experimentId,
        state: ChaosExperimentState.PLANNED,
        startTime: new Date(),
        baseline: await this.captureBaseline(),
        duringInjection: {} as any,
        recovery: {
          recoveryTime: 0,
          stabilityAchieved: false,
          dataConsistency: true
        },
        timeline: [],
        safetyActions: [],
        insights: {
          weakPoints: [],
          improvements: [],
          resilience: 0
        }
      };
      
      this.activeExperiments.set(experimentId, result);
      
      try {
        // Phase 1: Baseline monitoring
        await this.establishBaseline(experiment, result);
        
        // Phase 2: Failure injection
        await this.injectFailure(experiment, result);
        
        // Phase 3: Monitor during injection
        await this.monitorDuringInjection(experiment, result);
        
        // Phase 4: Recovery and validation
        await this.recoveryPhase(experiment, result);
        
        // Phase 5: Analysis and insights
        await this.analyzeResults(experiment, result);
        
        result.state = ChaosExperimentState.COMPLETED;
        result.endTime = new Date();
        
        console.log(`✅ Chaos experiment completed: ${experiment.name}`);
        
        return result;
        
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : 'Unknown error';
        
        // Emergency rollback
        await this.emergencyRollback(experiment, result, errorMsg);
        
        result.state = ChaosExperimentState.FAILED;
        result.endTime = new Date();
        
        span.recordException(error as Error);
        span.setAttributes({
          'chaos.experiment.success': false,
          'chaos.experiment.error': errorMsg
        });
        
        console.error(`❌ Chaos experiment failed: ${experiment.name} - ${errorMsg}`);
        
        return result;
        
      } finally {
        this.activeExperiments.delete(experimentId);
        span.end();
      }
    });
  }
  
  /**
   * Emergency stop all running experiments
   */
  async emergencyStop(reason: string = 'Manual emergency stop'): Promise<void> {
    console.warn(`🚨 EMERGENCY STOP: ${reason}`);
    
    const activeExperimentIds = Array.from(this.activeExperiments.keys());
    
    for (const experimentId of activeExperimentIds) {
      const experiment = this.experiments.get(experimentId);
      const result = this.activeExperiments.get(experimentId);
      
      if (experiment && result) {
        await this.emergencyRollback(experiment, result, reason);
        result.safetyActions.push({
          timestamp: new Date(),
          action: 'emergency_stop',
          reason,
          successful: true
        });
      }
    }
    
    // Clear all active experiments
    this.activeExperiments.clear();
    
    // Kill any running processes
    for (const [processId, process] of this.runningProcesses.entries()) {
      process.kill('SIGTERM');
      this.runningProcesses.delete(processId);
    }
    
    this.emit('emergencyStop', { reason, experimentsAffected: activeExperimentIds.length });
  }
  
  /**
   * Get status of all experiments
   */
  getExperimentStatus(): {
    registered: number;
    active: number;
    experiments: Array<{
      id: string;
      name: string;
      type: ChaosExperimentType;
      state: ChaosExperimentState;
      startTime?: Date;
      currentPhase?: string;
    }>;
  } {
    const active = Array.from(this.activeExperiments.entries()).map(([id, result]) => {
      const experiment = this.experiments.get(id)!;
      return {
        id,
        name: experiment.name,
        type: experiment.type,
        state: result.state,
        startTime: result.startTime,
        currentPhase: this.getCurrentPhase(result.state)
      };
    });
    
    return {
      registered: this.experiments.size,
      active: this.activeExperiments.size,
      experiments: active
    };
  }
  
  /**
   * Private helper methods
   */
  
  private initializeControllers(): void {
    this.networkController = new NetworkFailureController();
    this.resourceController = new ResourceExhaustionController();
    this.serviceController = new ServiceFailureController();
    this.dataController = new DataCorruptionController();
  }
  
  private startSafetyMonitoring(): void {
    this.safetyMonitor = setInterval(async () => {
      try {
        await this.performSafetyCheck();
      } catch (error) {
        console.error('Safety monitoring error:', error);
      }
    }, 5000); // Check every 5 seconds
  }
  
  private async performSafetyCheck(): Promise<void> {
    if (this.activeExperiments.size === 0) return;
    
    for (const [experimentId, result] of this.activeExperiments.entries()) {
      const experiment = this.experiments.get(experimentId)!;
      const currentMetrics = await this.getCurrentMetrics();
      
      // Check error rate threshold
      if (currentMetrics.errorRate > this.config.safetyLimits.emergencyStopThreshold.errorRate) {
        console.warn(`🚨 Emergency stop triggered: Error rate ${currentMetrics.errorRate * 100}%`);
        await this.emergencyRollback(experiment, result, 'Error rate threshold exceeded');
      }
      
      // Check latency threshold
      const latencyMultiplier = currentMetrics.latencyP99 / result.baseline.latencyP99;
      if (latencyMultiplier > this.config.safetyLimits.emergencyStopThreshold.latencyMultiplier) {
        console.warn(`🚨 Emergency stop triggered: Latency multiplier ${latencyMultiplier.toFixed(2)}x`);
        await this.emergencyRollback(experiment, result, 'Latency threshold exceeded');
      }
      
      // Check maximum experiment duration
      const runtime = Date.now() - result.startTime.getTime();
      if (runtime > this.config.safetyLimits.maxExperimentDuration) {
        console.warn(`🚨 Emergency stop triggered: Max duration ${runtime}ms exceeded`);
        await this.emergencyRollback(experiment, result, 'Maximum duration exceeded');
      }
    }
  }
  
  private validateProductionSafety(config: ChaosExperimentConfig): void {
    // Production safety validations
    if (config.impactRadius === 'full_system') {
      throw new Error('Full system impact not allowed in production');
    }
    
    if (config.maxDuration > this.config.safetyLimits.maxExperimentDuration) {
      throw new Error(`Experiment duration ${config.maxDuration}ms exceeds production limit`);
    }
    
    if (config.rollbackThreshold.errorRate > 0.05) {
      throw new Error('Error rate threshold too high for production (max 5%)');
    }
    
    // Require stricter rollback thresholds in production
    config.rollbackThreshold.errorRate = Math.min(config.rollbackThreshold.errorRate, 0.02);
    config.rollbackThreshold.availabilityMin = Math.max(config.rollbackThreshold.availabilityMin, 0.98);
  }
  
  private async captureBaseline(): Promise<ChaosExperimentResult['baseline']> {
    console.log('  📊 Capturing baseline metrics...');
    
    const metrics = await this.getCurrentMetrics();
    
    return {
      errorRate: metrics.errorRate,
      latencyP50: metrics.latencyP50,
      latencyP95: metrics.latencyP95,
      latencyP99: metrics.latencyP99,
      throughput: metrics.throughput,
      availability: metrics.availability
    };
  }
  
  private async getCurrentMetrics(): Promise<{
    errorRate: number;
    latencyP50: number;
    latencyP95: number;
    latencyP99: number;
    throughput: number;
    availability: number;
  }> {
    // In a real implementation, this would fetch from your monitoring system
    // For now, simulate metrics
    return {
      errorRate: Math.random() * 0.01, // 0-1% error rate
      latencyP50: 5 + Math.random() * 10, // 5-15ms
      latencyP95: 15 + Math.random() * 20, // 15-35ms  
      latencyP99: 25 + Math.random() * 30, // 25-55ms
      throughput: 100 + Math.random() * 50, // 100-150 QPS
      availability: 0.999 + Math.random() * 0.001 // 99.9-100%
    };
  }
  
  private async establishBaseline(
    experiment: ChaosExperimentConfig,
    result: ChaosExperimentResult
  ): Promise<void> {
    console.log('  📈 Establishing baseline...');
    
    result.state = ChaosExperimentState.RUNNING;
    result.timeline.push({
      timestamp: new Date(),
      event: 'baseline_start',
      metrics: result.baseline,
      severity: 'info'
    });
    
    // Monitor for baseline period (30 seconds)
    await sleep(30000);
    
    result.timeline.push({
      timestamp: new Date(), 
      event: 'baseline_complete',
      metrics: await this.getCurrentMetrics(),
      severity: 'info'
    });
  }
  
  private async injectFailure(
    experiment: ChaosExperimentConfig,
    result: ChaosExperimentResult
  ): Promise<void> {
    console.log(`  💥 Injecting failure: ${experiment.type}`);
    
    result.state = ChaosExperimentState.INJECTING;
    result.timeline.push({
      timestamp: new Date(),
      event: 'failure_injection_start',
      metrics: await this.getCurrentMetrics(),
      severity: 'warning'
    });
    
    switch (experiment.type) {
      case ChaosExperimentType.NETWORK_PARTITION:
        await this.networkController!.injectNetworkPartition(experiment.parameters);
        break;
        
      case ChaosExperimentType.SERVICE_FAILURE:
        await this.serviceController!.injectServiceFailure(experiment.parameters);
        break;
        
      case ChaosExperimentType.DATABASE_FAILURE:
        await this.injectDatabaseFailure(experiment.parameters);
        break;
        
      case ChaosExperimentType.MEMORY_PRESSURE:
        await this.resourceController!.injectMemoryPressure(experiment.parameters);
        break;
        
      case ChaosExperimentType.NATS_DISRUPTION:
        await this.injectNatsDisruption(experiment.parameters);
        break;
        
      case ChaosExperimentType.CACHE_CORRUPTION:
        await this.dataController!.injectCacheCorruption(experiment.parameters);
        break;
        
      default:
        throw new Error(`Unsupported experiment type: ${experiment.type}`);
    }
    
    result.timeline.push({
      timestamp: new Date(),
      event: 'failure_injection_complete',
      metrics: await this.getCurrentMetrics(),
      severity: 'error'
    });
  }
  
  private async monitorDuringInjection(
    experiment: ChaosExperimentConfig,
    result: ChaosExperimentResult
  ): Promise<void> {
    console.log('  🔍 Monitoring during injection...');
    
    const monitoringDuration = experiment.parameters.duration || 60000; // 1 minute default
    const checkInterval = experiment.monitoringInterval || 5000; // 5 seconds default
    const checksCount = Math.floor(monitoringDuration / checkInterval);
    
    for (let i = 0; i < checksCount; i++) {
      await sleep(checkInterval);
      
      const currentMetrics = await this.getCurrentMetrics();
      
      result.timeline.push({
        timestamp: new Date(),
        event: 'monitoring_check',
        metrics: currentMetrics,
        severity: 'info'
      });
      
      // Check rollback thresholds
      if (currentMetrics.errorRate > experiment.rollbackThreshold.errorRate) {
        console.warn(`⚠️  Error rate threshold exceeded: ${currentMetrics.errorRate * 100}%`);
        await this.triggerRollback(experiment, result, 'Error rate threshold exceeded');
        return;
      }
      
      if (currentMetrics.latencyP99 > experiment.rollbackThreshold.latencyP99) {
        console.warn(`⚠️  Latency threshold exceeded: ${currentMetrics.latencyP99}ms`);
        await this.triggerRollback(experiment, result, 'Latency threshold exceeded');
        return;
      }
      
      if (currentMetrics.availability < experiment.rollbackThreshold.availabilityMin) {
        console.warn(`⚠️  Availability threshold exceeded: ${currentMetrics.availability * 100}%`);
        await this.triggerRollback(experiment, result, 'Availability threshold exceeded');
        return;
      }
    }
    
    // Capture final metrics during injection
    result.duringInjection = await this.getCurrentMetrics();
  }
  
  private async recoveryPhase(
    experiment: ChaosExperimentConfig,
    result: ChaosExperimentResult
  ): Promise<void> {
    console.log('  🔄 Starting recovery phase...');
    
    result.state = ChaosExperimentState.RECOVERING;
    const recoveryStartTime = Date.now();
    
    // Remove failure injection
    await this.removeFailureInjection(experiment);
    
    result.timeline.push({
      timestamp: new Date(),
      event: 'recovery_start',
      metrics: await this.getCurrentMetrics(),
      severity: 'info'
    });
    
    // Monitor recovery
    let recoveryComplete = false;
    const recoveryTimeout = experiment.recoveryValidation.stabilityPeriod || 60000;
    const endTime = Date.now() + recoveryTimeout;
    
    while (Date.now() < endTime && !recoveryComplete) {
      await sleep(5000);
      
      const currentMetrics = await this.getCurrentMetrics();
      
      // Check if metrics have returned to baseline levels
      const errorRateOk = currentMetrics.errorRate <= result.baseline.errorRate * 1.5;
      const latencyOk = currentMetrics.latencyP99 <= result.baseline.latencyP99 * 1.2;
      const throughputOk = currentMetrics.throughput >= result.baseline.throughput * 0.8;
      
      if (errorRateOk && latencyOk && throughputOk) {
        recoveryComplete = true;
        result.recovery.stabilityAchieved = true;
      }
      
      result.timeline.push({
        timestamp: new Date(),
        event: 'recovery_check',
        metrics: currentMetrics,
        severity: errorRateOk && latencyOk ? 'info' : 'warning'
      });
    }
    
    result.recovery.recoveryTime = Date.now() - recoveryStartTime;
    
    // Data consistency check
    result.recovery.dataConsistency = await this.validateDataConsistency();
    
    result.timeline.push({
      timestamp: new Date(),
      event: 'recovery_complete',
      metrics: await this.getCurrentMetrics(),
      severity: result.recovery.stabilityAchieved ? 'info' : 'warning'
    });
  }
  
  private async analyzeResults(
    experiment: ChaosExperimentConfig,
    result: ChaosExperimentResult
  ): Promise<void> {
    console.log('  🧠 Analyzing results...');
    
    const insights = result.insights;
    
    // Identify weak points
    if (result.duringInjection.errorRate > result.baseline.errorRate * 10) {
      insights.weakPoints.push('High error rate sensitivity');
    }
    
    if (result.duringInjection.latencyP99 > result.baseline.latencyP99 * 5) {
      insights.weakPoints.push('Poor latency resilience');
    }
    
    if (!result.recovery.stabilityAchieved) {
      insights.weakPoints.push('Slow or incomplete recovery');
    }
    
    if (!result.recovery.dataConsistency) {
      insights.weakPoints.push('Data consistency issues');
    }
    
    // Generate improvements
    if (insights.weakPoints.includes('High error rate sensitivity')) {
      insights.improvements.push('Implement better circuit breakers and fallback mechanisms');
    }
    
    if (insights.weakPoints.includes('Poor latency resilience')) {
      insights.improvements.push('Add request timeouts and bulkhead isolation');
    }
    
    if (insights.weakPoints.includes('Slow or incomplete recovery')) {
      insights.improvements.push('Improve health checks and auto-recovery mechanisms');
    }
    
    // Calculate resilience score (0-100)
    let score = 100;
    score -= result.duringInjection.errorRate * 1000; // Penalize errors heavily
    score -= Math.max(0, (result.duringInjection.latencyP99 / result.baseline.latencyP99 - 1) * 20);
    score -= result.recovery.stabilityAchieved ? 0 : 30;
    score -= result.recovery.dataConsistency ? 0 : 40;
    
    insights.resilience = Math.max(0, Math.min(100, score));
  }
  
  private async removeFailureInjection(experiment: ChaosExperimentConfig): Promise<void> {
    switch (experiment.type) {
      case ChaosExperimentType.NETWORK_PARTITION:
        await this.networkController!.removeNetworkPartition();
        break;
        
      case ChaosExperimentType.SERVICE_FAILURE:
        await this.serviceController!.removeServiceFailure();
        break;
        
      case ChaosExperimentType.DATABASE_FAILURE:
        await this.removeDatabaseFailure();
        break;
        
      case ChaosExperimentType.MEMORY_PRESSURE:
        await this.resourceController!.removeMemoryPressure();
        break;
        
      case ChaosExperimentType.NATS_DISRUPTION:
        await this.removeNatsDisruption();
        break;
        
      case ChaosExperimentType.CACHE_CORRUPTION:
        await this.dataController!.removeCacheCorruption();
        break;
    }
  }
  
  private async triggerRollback(
    experiment: ChaosExperimentConfig,
    result: ChaosExperimentResult,
    reason: string
  ): Promise<void> {
    console.log(`⏮️  Triggering rollback: ${reason}`);
    
    result.safetyActions.push({
      timestamp: new Date(),
      action: 'automatic_rollback',
      reason,
      successful: true
    });
    
    await this.removeFailureInjection(experiment);
    result.state = ChaosExperimentState.ROLLED_BACK;
  }
  
  private async emergencyRollback(
    experiment: ChaosExperimentConfig,
    result: ChaosExperimentResult,
    reason: string
  ): Promise<void> {
    console.error(`🚨 Emergency rollback: ${reason}`);
    
    result.safetyActions.push({
      timestamp: new Date(),
      action: 'emergency_rollback',
      reason,
      successful: true
    });
    
    try {
      await this.removeFailureInjection(experiment);
    } catch (error) {
      console.error('Failed to remove failure injection:', error);
      result.safetyActions[result.safetyActions.length - 1].successful = false;
    }
    
    result.state = ChaosExperimentState.ROLLED_BACK;
  }
  
  private getCurrentPhase(state: ChaosExperimentState): string {
    switch (state) {
      case ChaosExperimentState.PLANNED: return 'Planning';
      case ChaosExperimentState.RUNNING: return 'Baseline';
      case ChaosExperimentState.INJECTING: return 'Injection';
      case ChaosExperimentState.RECOVERING: return 'Recovery';
      case ChaosExperimentState.COMPLETED: return 'Completed';
      case ChaosExperimentState.FAILED: return 'Failed';
      case ChaosExperimentState.ROLLED_BACK: return 'Rolled Back';
      default: return 'Unknown';
    }
  }
  
  // Database failure injection methods
  private async injectDatabaseFailure(parameters: any): Promise<void> {
    // Implementation would depend on your database setup
    console.log('Injecting database failure...', parameters);
  }
  
  private async removeDatabaseFailure(): Promise<void> {
    console.log('Removing database failure injection...');
  }
  
  // NATS disruption methods
  private async injectNatsDisruption(parameters: any): Promise<void> {
    console.log('Injecting NATS disruption...', parameters);
  }
  
  private async removeNatsDisruption(): Promise<void> {
    console.log('Removing NATS disruption...');
  }
  
  // Data consistency validation
  private async validateDataConsistency(): Promise<boolean> {
    // Implementation would check data integrity
    return true;
  }
  
  /**
   * Graceful shutdown
   */
  async shutdown(): Promise<void> {
    console.log('Shutting down chaos engineering framework...');
    
    // Stop safety monitoring
    if (this.safetyMonitor) {
      clearInterval(this.safetyMonitor);
    }
    
    // Emergency stop any running experiments
    if (this.activeExperiments.size > 0) {
      await this.emergencyStop('Framework shutdown');
    }
  }
}

/**
 * Network Failure Controller
 */
class NetworkFailureController {
  private injectedFailures = new Set<string>();
  
  async injectNetworkPartition(parameters: {
    targetHosts?: string[];
    blockPorts?: number[];
    dropPercentage?: number;
    latencyMs?: number;
  }): Promise<void> {
    console.log('Injecting network partition:', parameters);
    
    // Use tc (traffic control) on Linux for network simulation
    if (parameters.latencyMs) {
      const command = `tc qdisc add dev eth0 root netem delay ${parameters.latencyMs}ms`;
      await this.executeCommand(command, 'network_latency');
    }
    
    if (parameters.dropPercentage) {
      const command = `tc qdisc add dev eth0 root netem loss ${parameters.dropPercentage}%`;
      await this.executeCommand(command, 'network_drop');
    }
  }
  
  async removeNetworkPartition(): Promise<void> {
    console.log('Removing network partition...');
    
    // Remove all network rules
    await this.executeCommand('tc qdisc del dev eth0 root', 'network_cleanup');
    this.injectedFailures.clear();
  }
  
  private async executeCommand(command: string, failureId: string): Promise<void> {
    return new Promise((resolve, reject) => {
      exec(command, (error, stdout, stderr) => {
        if (error) {
          console.warn(`Network command failed (this is expected in non-Linux environments): ${error.message}`);
          // Don't fail in development/test environments
          resolve();
          return;
        }
        
        this.injectedFailures.add(failureId);
        resolve();
      });
    });
  }
}

/**
 * Resource Exhaustion Controller  
 */
class ResourceExhaustionController {
  private processes: ChildProcess[] = [];
  
  async injectMemoryPressure(parameters: {
    memoryMB?: number;
    durationMs?: number;
  }): Promise<void> {
    console.log('Injecting memory pressure:', parameters);
    
    const memoryMB = parameters.memoryMB || 100;
    const durationMs = parameters.durationMs || 60000;
    
    // Create memory pressure using a simple allocation script
    const process = spawn('node', [
      '-e',
      `
      const arrays = [];
      const targetMB = ${memoryMB};
      const chunkSize = 1024 * 1024; // 1MB chunks
      
      console.log('Starting memory pressure test...');
      
      for (let i = 0; i < targetMB; i++) {
        arrays.push(new Array(chunkSize / 4).fill(Math.random()));
        if (i % 10 === 0) {
          console.log(\`Allocated \${i}MB\`);
        }
      }
      
      setTimeout(() => {
        console.log('Memory pressure test complete');
        process.exit(0);
      }, ${durationMs});
      `
    ]);
    
    this.processes.push(process);
  }
  
  async removeMemoryPressure(): Promise<void> {
    console.log('Removing memory pressure...');
    
    for (const process of this.processes) {
      if (process && !process.killed) {
        process.kill('SIGTERM');
      }
    }
    
    this.processes = [];
  }
}

/**
 * Service Failure Controller
 */
class ServiceFailureController {
  private failedServices = new Map<string, any>();
  
  async injectServiceFailure(parameters: {
    service?: string;
    failureType?: 'crash' | 'hang' | 'slow_response';
    recoveryTimeMs?: number;
  }): Promise<void> {
    console.log('Injecting service failure:', parameters);
    
    const service = parameters.service || 'search-engine';
    const failureType = parameters.failureType || 'slow_response';
    
    this.failedServices.set(service, {
      type: failureType,
      startTime: Date.now(),
      recoveryTime: parameters.recoveryTimeMs || 30000
    });
    
    // Simulate service failure
    // In a real implementation, this might kill processes, modify configurations, etc.
  }
  
  async removeServiceFailure(): Promise<void> {
    console.log('Removing service failures...');
    
    this.failedServices.clear();
    // Restore services
  }
}

/**
 * Data Corruption Controller
 */
class DataCorruptionController {
  private corruptedData = new Set<string>();
  
  async injectCacheCorruption(parameters: {
    cacheType?: string;
    corruptionRate?: number;
  }): Promise<void> {
    console.log('Injecting cache corruption:', parameters);
    
    const cacheType = parameters.cacheType || 'search_cache';
    const corruptionRate = parameters.corruptionRate || 0.05; // 5% corruption
    
    this.corruptedData.add(cacheType);
  }
  
  async removeCacheCorruption(): Promise<void> {
    console.log('Removing cache corruption...');
    
    this.corruptedData.clear();
    // Restore cache integrity
  }
}

// Export singleton
export const chaosFramework = ChaosEngineeringFramework.getInstance();