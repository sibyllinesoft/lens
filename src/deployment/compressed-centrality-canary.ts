/**
 * Compressed 2-Hour Centrality Canary Deployment
 * 
 * Executes rapid canary deployment with intensive monitoring, hardening measures,
 * and comprehensive validation gates for centrality system promotion.
 */

import { EventEmitter } from 'events';
import { CentralityConfig } from '../core/centrality-config.js';
import { CanaryGatesValidator } from './canary-gates-validator.js';
import { RealTimeMonitor } from '../monitoring/real-time-monitor.js';
import { AAShadowTester } from './aa-shadow-tester.js';
import { RouterInterplayMonitor } from './router-interplay-monitor.js';
import { MMRPilot } from './mmr-pilot.js';
import { AutoRollback } from './auto-rollback.js';

interface CanaryPhase {
  name: string;
  trafficPercentage: number;
  durationMinutes: number;
  validationIntervalMinutes: number;
  gates: CanaryGate[];
}

interface CanaryGate {
  metric: string;
  operator: 'gte' | 'lte' | 'eq';
  threshold: number;
  pValue?: number;
  description: string;
}

interface CentralityHardeningConfig {
  stageACentralityPrior: boolean;
  centralityLogOddsCap: number;
  mmrEnabled: boolean;
  perFileSpanCap: number;
  onlyNLAndSymbol: boolean;
}

export class CompressedCentralityCanary extends EventEmitter {
  private config: CentralityHardeningConfig;
  private gatesValidator: CanaryGatesValidator;
  private realTimeMonitor: RealTimeMonitor;
  private aaShadowTester: AAShadowTester;
  private routerMonitor: RouterInterplayMonitor;
  private mmrPilot: MMRPilot;
  private autoRollback: AutoRollback;
  private currentPhase = 0;
  private startTime: Date | null = null;
  private isRollbackInProgress = false;

  private readonly phases: CanaryPhase[] = [
    {
      name: 'Phase 1: Intensive Monitoring',
      trafficPercentage: 5,
      durationMinutes: 60,
      validationIntervalMinutes: 10,
      gates: [
        { metric: 'nDCG@10_delta', operator: 'gte', threshold: 1.0, pValue: 0.05, description: 'nDCG@10 improvement ≥+1.0pt (p<0.05)' },
        { metric: 'recall@50', operator: 'gte', threshold: 88.9, description: 'Maintain baseline recall@50 ≥88.9%' },
        { metric: 'diversity@10_delta', operator: 'gte', threshold: 10, description: 'Diversity@10 improvement ≥+10%' },
        { metric: 'stageA_p95_delta', operator: 'lte', threshold: 1.0, description: 'Stage-A p95 latency ≤+1ms' },
        { metric: 'stageC_p95_delta_pct', operator: 'lte', threshold: 5.0, description: 'Stage-C p95 latency ≤+5%' },
      ]
    },
    {
      name: 'Phase 2: Full Validation',
      trafficPercentage: 25,
      durationMinutes: 60,
      validationIntervalMinutes: 10,
      gates: [
        { metric: 'core@10_delta', operator: 'gte', threshold: 10, description: 'Core@10 improvement ≥+10pp' },
        { metric: 'p99_p95_ratio', operator: 'lte', threshold: 2.0, description: 'p99/p95 ratio ≤2.0 for SLA compliance' },
        { metric: 'span_coverage', operator: 'gte', threshold: 100, description: 'Complete span coverage' },
        { metric: 'semantic_share_delta', operator: 'lte', threshold: 15, description: 'Semantic share increase ≤+15pp unless nDCG rises' },
        { metric: 'router_upshift_rate', operator: 'lte', threshold: 7, description: 'Router upshift rate ≤5%+2pp' },
      ]
    }
  ];

  constructor() {
    super();
    
    // Initialize hardening configuration
    this.config = {
      stageACentralityPrior: true, // Only for NL and symbol queries
      centralityLogOddsCap: 0.4,   // Cap net centrality log-odds ≤ 0.4
      mmrEnabled: false,           // Leave MMR off unless query classifier says "overview"
      perFileSpanCap: 5,           // Preserve span cap ≤ 5 (→8 only at high topic-sim)
      onlyNLAndSymbol: true        // Enable centrality only for NL and symbol queries
    };

    this.initializeComponents();
    this.setupEventHandlers();
  }

  private initializeComponents(): void {
    this.gatesValidator = new CanaryGatesValidator();
    this.realTimeMonitor = new RealTimeMonitor({
      validationWindowMinutes: 10,
      statisticalSignificanceThreshold: 0.05
    });
    this.aaShadowTester = new AAShadowTester({
      trafficPercentage: 2,
      shuffleCentralityWeights: true
    });
    this.routerMonitor = new RouterInterplayMonitor({
      targetUpshiftRate: 5,
      tolerancePercentagePoints: 2
    });
    this.mmrPilot = new MMRPilot({
      gamma: 0.10,
      delta: 0.05,
      targetSlice: 'NL-overview',
      trafficPercentage: 10
    });
    this.autoRollback = new AutoRollback({
      rollbackOrder: ['rerank.mmr', 'stageA.centrality_prior', 'stageC.centrality']
    });
  }

  private setupEventHandlers(): void {
    this.gatesValidator.on('gateFailure', this.handleGateFailure.bind(this));
    this.realTimeMonitor.on('anomalyDetected', this.handleAnomalyDetected.bind(this));
    this.aaShadowTester.on('spuriousLiftDetected', this.handleSpuriousLift.bind(this));
    this.routerMonitor.on('upshiftRateViolation', this.handleRouterViolation.bind(this));
    this.autoRollback.on('rollbackComplete', this.handleRollbackComplete.bind(this));
  }

  public async startCanary(): Promise<void> {
    console.log('🚀 Starting Compressed 2-Hour Centrality Canary Deployment');
    console.log(`📊 Target metrics: +1.8 nDCG pts, +23% diversity, +22pp Core@10`);
    
    this.startTime = new Date();
    this.emit('canaryStarted', { startTime: this.startTime, config: this.config });

    try {
      // Apply hardening configuration
      await this.applyHardeningMeasures();
      
      // Start monitoring components
      await this.startMonitoring();
      
      // Execute phases sequentially
      for (let i = 0; i < this.phases.length; i++) {
        this.currentPhase = i;
        await this.executePhase(this.phases[i]);
        
        if (this.isRollbackInProgress) {
          throw new Error('Rollback initiated, aborting canary');
        }
      }
      
      // Final validation and promotion decision
      await this.makeFinalPromotionDecision();
      
    } catch (error) {
      console.error('❌ Canary deployment failed:', error);
      if (!this.isRollbackInProgress) {
        await this.initiateRollback('Deployment failure');
      }
      throw error;
    }
  }

  private async applyHardeningMeasures(): void {
    console.log('🔒 Applying hardening measures...');
    
    const centralityConfig = new CentralityConfig();
    
    // Configure Stage-A centrality prior
    await centralityConfig.updateStageAConfig({
      centralityPriorEnabled: this.config.stageACentralityPrior,
      queryTypes: this.config.onlyNLAndSymbol ? ['NL', 'symbol'] : ['NL', 'symbol', 'path'],
      centralityLogOddsCap: this.config.centralityLogOddsCap
    });
    
    // Configure span caps with topic-similarity-based adjustment
    await centralityConfig.updateSpanConfig({
      basePerFileSpanCap: this.config.perFileSpanCap,
      highTopicSimCap: 8,
      topicSimilarityThreshold: 0.7
    });
    
    // Keep MMR disabled initially
    await centralityConfig.updateMMRConfig({
      enabled: this.config.mmrEnabled,
      overviewQueryOverride: true // Enable only for overview queries
    });
    
    console.log('✅ Hardening measures applied');
    this.emit('hardeningApplied', this.config);
  }

  private async startMonitoring(): void {
    console.log('📈 Starting monitoring components...');
    
    // Start real-time statistical validation
    await this.realTimeMonitor.start();
    
    // Start A/A shadow testing
    await this.aaShadowTester.start();
    
    // Start router interplay monitoring
    await this.routerMonitor.start();
    
    // Initialize MMR pilot (but don't activate yet)
    await this.mmrPilot.initialize();
    
    console.log('✅ All monitoring components active');
    this.emit('monitoringStarted');
  }

  private async executePhase(phase: CanaryPhase): Promise<void> {
    console.log(`🎯 Starting ${phase.name} (${phase.trafficPercentage}% traffic, ${phase.durationMinutes}min)`);
    
    // Ramp traffic to target percentage
    await this.rampTraffic(phase.trafficPercentage);
    
    const phaseStartTime = Date.now();
    const phaseDurationMs = phase.durationMinutes * 60 * 1000;
    const validationIntervalMs = phase.validationIntervalMinutes * 60 * 1000;
    
    // Run validation cycles throughout the phase
    while (Date.now() - phaseStartTime < phaseDurationMs) {
      if (this.isRollbackInProgress) {
        throw new Error('Rollback in progress, terminating phase');
      }
      
      // Wait for validation interval
      await new Promise(resolve => setTimeout(resolve, validationIntervalMs));
      
      // Run comprehensive validation
      const validationResults = await this.runPhaseValidation(phase);
      
      if (!validationResults.allGatesPassed) {
        console.error(`❌ Gates failed in ${phase.name}:`, validationResults.failedGates);
        await this.initiateRollback(`Phase ${this.currentPhase + 1} gate failures`);
        throw new Error('Phase validation failed');
      }
      
      console.log(`✅ Validation passed for ${phase.name}`);
      this.emit('phaseValidationPassed', { phase: phase.name, results: validationResults });
    }
    
    console.log(`✅ ${phase.name} completed successfully`);
    this.emit('phaseCompleted', { phase: phase.name });
  }

  private async rampTraffic(targetPercentage: number): Promise<void> {
    console.log(`📈 Ramping traffic to ${targetPercentage}%...`);
    
    // Implement gradual traffic ramp to avoid shock
    const rampSteps = Math.min(5, targetPercentage);
    const stepSize = targetPercentage / rampSteps;
    
    for (let i = 1; i <= rampSteps; i++) {
      const currentPercentage = Math.round(stepSize * i);
      await this.setTrafficPercentage(currentPercentage);
      
      // Wait 2 minutes between ramp steps for metrics stabilization
      if (i < rampSteps) {
        await new Promise(resolve => setTimeout(resolve, 2 * 60 * 1000));
      }
    }
    
    console.log(`✅ Traffic ramped to ${targetPercentage}%`);
    this.emit('trafficRamped', { percentage: targetPercentage });
  }

  private async setTrafficPercentage(percentage: number): Promise<void> {
    // Implementation would interact with feature flag system
    console.log(`Setting centrality canary traffic to ${percentage}%`);
    // await featureFlagService.setTrafficPercentage('centrality-canary', percentage);
  }

  private async runPhaseValidation(phase: CanaryPhase): Promise<{
    allGatesPassed: boolean;
    failedGates: string[];
    metrics: Record<string, number>;
  }> {
    console.log(`🔍 Running validation for ${phase.name}...`);
    
    // Collect real-time metrics
    const metrics = await this.realTimeMonitor.collectMetrics();
    
    // Validate against phase gates
    const gateResults = await this.gatesValidator.validateGates(phase.gates, metrics);
    
    // Check for drift and anomalies
    const driftResults = await this.checkForDrift(metrics);
    const anomalyResults = await this.checkForAnomalies();
    
    const allGatesPassed = gateResults.passed && driftResults.passed && anomalyResults.passed;
    const failedGates = [
      ...gateResults.failedGates,
      ...driftResults.failedChecks,
      ...anomalyResults.anomalies
    ];
    
    return {
      allGatesPassed,
      failedGates,
      metrics
    };
  }

  private async checkForDrift(metrics: Record<string, number>): Promise<{
    passed: boolean;
    failedChecks: string[];
  }> {
    const failedChecks: string[] = [];
    
    // Check semantic share drift
    const semanticShareDelta = metrics['semantic_share_delta'] || 0;
    const nDCGDelta = metrics['nDCG@10_delta'] || 0;
    
    if (semanticShareDelta > 15 && nDCGDelta < 1.0) {
      failedChecks.push('Semantic share increased >15pp without corresponding nDCG improvement');
    }
    
    // Check utility flag hits
    const utilFlagHits = metrics['util_flag_hits_delta'] || 0;
    if (utilFlagHits > 5) {
      failedChecks.push('Utility flag hits increased >5pp over baseline');
    }
    
    // Check topic-normalized centrality drift
    const topicNormalizedCore = metrics['topic_normalized_core@10'] || 0;
    const core10Delta = metrics['core@10_delta'] || 0;
    
    if (topicNormalizedCore > core10Delta * 1.2) {
      failedChecks.push('Topic-normalized centrality drift detected - Core@10 rising without nDCG plateau');
    }
    
    return {
      passed: failedChecks.length === 0,
      failedChecks
    };
  }

  private async checkForAnomalies(): Promise<{
    passed: boolean;
    anomalies: string[];
  }> {
    const anomalies: string[] = [];
    
    // Check A/A shadow test results
    const spuriousLift = await this.aaShadowTester.checkForSpuriousLift();
    if (spuriousLift.detected) {
      anomalies.push(`Spurious lift detected in A/A shadow test: ${spuriousLift.description}`);
    }
    
    // Check router interplay
    const routerViolation = await this.routerMonitor.checkUpshiftRate();
    if (routerViolation.violated) {
      anomalies.push(`Router upshift rate violation: ${routerViolation.currentRate}% (target: 5%±2pp)`);
    }
    
    return {
      passed: anomalies.length === 0,
      anomalies
    };
  }

  private async makeFinalPromotionDecision(): Promise<void> {
    console.log('🎯 Making final promotion decision...');
    
    // Collect comprehensive metrics for final evaluation
    const finalMetrics = await this.realTimeMonitor.collectComprehensiveMetrics();
    
    // Validate against all quality gates
    const qualityGates: CanaryGate[] = [
      { metric: 'nDCG@10_delta', operator: 'gte', threshold: 1.0, pValue: 0.05, description: 'Final nDCG@10 validation' },
      { metric: 'diversity@10_delta', operator: 'gte', threshold: 10, description: 'Final diversity validation' },
      { metric: 'core@10_delta', operator: 'gte', threshold: 10, description: 'Final Core@10 validation' },
      { metric: 'recall@50', operator: 'gte', threshold: 88.9, description: 'Final recall validation' }
    ];
    
    const finalValidation = await this.gatesValidator.validateGates(qualityGates, finalMetrics);
    
    if (finalValidation.passed) {
      console.log('🎉 All gates passed! Recommending promotion to 100%');
      await this.promoteToFullTraffic();
      this.emit('promotionRecommended', { metrics: finalMetrics });
    } else {
      console.log('❌ Final validation failed, maintaining current traffic level');
      this.emit('promotionDenied', { failedGates: finalValidation.failedGates });
    }
  }

  private async promoteToFullTraffic(): Promise<void> {
    console.log('🚀 Promoting centrality system to 100% traffic...');
    await this.setTrafficPercentage(100);
    
    // Continue monitoring for post-deployment validation
    setTimeout(async () => {
      await this.runPostDeploymentValidation();
    }, 10 * 60 * 1000); // 10-minute post-deployment check
    
    this.emit('promotionCompleted');
  }

  private async runPostDeploymentValidation(): Promise<void> {
    console.log('🔍 Running post-deployment validation...');
    
    const postMetrics = await this.realTimeMonitor.collectMetrics();
    const validation = await this.gatesValidator.validateGates(this.phases[1].gates, postMetrics);
    
    if (!validation.passed) {
      console.warn('⚠️ Post-deployment validation issues detected');
      this.emit('postDeploymentIssues', { failedGates: validation.failedGates });
    } else {
      console.log('✅ Post-deployment validation passed');
      this.emit('postDeploymentSuccess');
    }
  }

  private async handleGateFailure(failure: any): void {
    console.error(`❌ Gate failure detected: ${failure.gate} - ${failure.reason}`);
    await this.initiateRollback(`Gate failure: ${failure.gate}`);
  }

  private async handleAnomalyDetected(anomaly: any): void {
    console.warn(`⚠️ Anomaly detected: ${anomaly.type} - ${anomaly.description}`);
    
    if (anomaly.severity === 'critical') {
      await this.initiateRollback(`Critical anomaly: ${anomaly.type}`);
    } else {
      this.emit('anomalyWarning', anomaly);
    }
  }

  private async handleSpuriousLift(lift: any): void {
    console.error(`❌ Spurious lift detected in A/A shadow test: ${lift.description}`);
    await this.initiateRollback('Spurious lift detection');
  }

  private async handleRouterViolation(violation: any): void {
    console.warn(`⚠️ Router upshift rate violation: ${violation.currentRate}%`);
    
    // Attempt automatic correction first
    if (violation.currentRate > 7) { // 5% + 2pp tolerance
      await this.routerMonitor.adjustThresholds();
      
      // If still violating after adjustment, rollback
      setTimeout(async () => {
        const recheck = await this.routerMonitor.checkUpshiftRate();
        if (recheck.violated) {
          await this.initiateRollback('Router upshift rate still violated after adjustment');
        }
      }, 5 * 60 * 1000); // Check again in 5 minutes
    }
  }

  private async initiateRollback(reason: string): Promise<void> {
    if (this.isRollbackInProgress) {
      console.log('Rollback already in progress, skipping duplicate initiation');
      return;
    }
    
    console.error(`🚨 Initiating rollback - Reason: ${reason}`);
    this.isRollbackInProgress = true;
    
    this.emit('rollbackInitiated', { reason, timestamp: new Date() });
    
    try {
      await this.autoRollback.execute();
      console.log('✅ Rollback completed successfully');
    } catch (error) {
      console.error('❌ Rollback failed:', error);
      this.emit('rollbackFailed', error);
    }
  }

  private async handleRollbackComplete(): void {
    console.log('✅ Automatic rollback completed');
    this.emit('rollbackCompleted');
  }

  public async runMMRPilot(): Promise<void> {
    console.log('🧪 Starting MMR pilot during canary...');
    
    try {
      const results = await this.mmrPilot.run();
      
      if (results.diversityImprovement >= 15 && results.latencyImpact <= 1) {
        console.log('✅ MMR pilot successful - diversity +15% with ≤1ms tail');
        this.emit('mmrPilotSuccess', results);
      } else {
        console.log('❌ MMR pilot failed criteria - keeping MMR off');
        this.emit('mmrPilotFailed', results);
      }
    } catch (error) {
      console.error('MMR pilot error:', error);
      this.emit('mmrPilotError', error);
    }
  }

  public getCanaryStatus(): {
    phase: string;
    trafficPercentage: number;
    elapsedMinutes: number;
    isRollbackInProgress: boolean;
  } {
    const currentPhaseInfo = this.phases[this.currentPhase];
    const elapsedMinutes = this.startTime ? 
      Math.floor((Date.now() - this.startTime.getTime()) / (1000 * 60)) : 0;
    
    return {
      phase: currentPhaseInfo?.name || 'Not started',
      trafficPercentage: currentPhaseInfo?.trafficPercentage || 0,
      elapsedMinutes,
      isRollbackInProgress: this.isRollbackInProgress
    };
  }
}