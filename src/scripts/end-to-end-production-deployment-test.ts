#!/usr/bin/env bun

/**
 * End-to-End Production Deployment Test
 * 
 * Executes a complete TODO.md deployment workflow to validate production readiness:
 * 
 * 1. Final pinned benchmark (with span coverage 100%)
 * 2. Tag + freeze configuration (version increment)
 * 3. Canary A→B→C deployment (24h holds with abort conditions)
 * 4. Post-deploy calibration (2-day holdout + τ optimization)
 * 5. Drift monitoring setup (comprehensive breach response)
 * 6. Week-one monitoring + RAPTOR rollout scheduling
 * 
 * This is the final validation before production deployment.
 */

import { TODOCompleteDeploymentOrchestrator } from '../deployment/todo-complete-deployment-orchestrator.js';
import { PinnedGroundTruthLoader } from '../benchmark/pinned-ground-truth-loader.js';

interface E2ETestResult {
  success: boolean;
  deployment_id: string;
  total_duration_hours: number;
  steps_completed: string[];
  steps_failed: string[];
  promotion_gates_passed: boolean;
  canary_completed: boolean;
  calibration_initiated: boolean;
  monitoring_active: boolean;
  final_metrics: Record<string, number>;
  production_ready: boolean;
}

class EndToEndProductionDeploymentTest {
  private orchestrator: TODOCompleteDeploymentOrchestrator;
  private pinnedLoader: PinnedGroundTruthLoader;
  private testStartTime: Date;
  private testResults: Partial<E2ETestResult>;
  
  constructor() {
    this.orchestrator = new TODOCompleteDeploymentOrchestrator();
    this.pinnedLoader = new PinnedGroundTruthLoader();
    this.testStartTime = new Date();
    this.testResults = {
      steps_completed: [],
      steps_failed: [],
      final_metrics: {}
    };
    
    this.setupEventHandlers();
  }
  
  /**
   * Execute complete end-to-end production deployment test
   */
  public async executeE2ETest(): Promise<E2ETestResult> {
    console.log('\n🎯 END-TO-END PRODUCTION DEPLOYMENT TEST');
    console.log('=' .repeat(80));
    console.log('Executing complete TODO.md workflow with validation at each step');
    console.log(`Test Start: ${this.testStartTime.toISOString()}`);
    console.log('');
    
    const deploymentId = `e2e-test-${Date.now()}`;
    this.testResults.deployment_id = deploymentId;
    
    try {
      // PHASE 1: Pre-deployment validation
      console.log('🔍 PHASE 1: PRE-DEPLOYMENT VALIDATION');
      console.log('-' .repeat(50));
      await this.validatePreDeploymentState();
      
      // PHASE 2: Execute complete TODO.md workflow
      console.log('\n🚀 PHASE 2: COMPLETE TODO.md WORKFLOW EXECUTION');
      console.log('-' .repeat(50));
      const deploymentResult = await (this.orchestrator as any).executeComplete6StepWorkflow(
        deploymentId,
        'lens-v1.2-prod',
        0.85 // Current τ value
      );
      
      // PHASE 3: Post-deployment validation
      console.log('\n✅ PHASE 3: POST-DEPLOYMENT VALIDATION');
      console.log('-' .repeat(50));
      await this.validatePostDeploymentState(deploymentResult);
      
      // PHASE 4: Generate final results
      console.log('\n📊 PHASE 4: FINAL RESULTS COMPILATION');
      console.log('-' .repeat(50));
      const finalResult = this.compileFinalResults(deploymentResult, true);
      
      console.log('\n🎉 END-TO-END TEST COMPLETED SUCCESSFULLY');
      console.log('=' .repeat(80));
      this.displayFinalResults(finalResult);
      
      return finalResult;
      
    } catch (error) {
      console.error('\n💥 END-TO-END TEST FAILED:', error);
      
      const finalResult = this.compileFinalResults(null, false, error);
      this.displayFinalResults(finalResult);
      
      return finalResult;
    }
  }
  
  /**
   * Validate system state before deployment
   */
  private async validatePreDeploymentState(): Promise<void> {
    console.log('📋 Validating pinned dataset consistency...');
    await this.pinnedLoader.loadPinnedDataset();
    const { passed, report } = await this.pinnedLoader.validatePinnedDatasetConsistency();
    
    if (!passed) {
      throw new Error(`Pinned dataset validation failed: ${report.inconsistent_results} inconsistencies`);
    }
    
    console.log(`✅ Pinned dataset validated: ${report.total_items} items, 100% consistent`);
    this.testResults.steps_completed?.push('pinned_dataset_validation');
    
    // Validate system configuration
    console.log('⚙️ Validating system configuration...');
    const configValidation = await (this.orchestrator as any).validateSystemConfiguration();
    if (!configValidation.valid) {
      throw new Error(`System configuration validation failed: ${configValidation.errors.join(', ')}`);
    }
    
    console.log('✅ System configuration validated');
    this.testResults.steps_completed?.push('system_config_validation');
    
    // Check deployment readiness
    console.log('🔧 Checking deployment readiness...');
    const readinessCheck = await (this.orchestrator as any).checkDeploymentReadiness();
    if (!readinessCheck.ready) {
      throw new Error(`Deployment readiness check failed: ${readinessCheck.blocking_issues.join(', ')}`);
    }
    
    console.log('✅ Deployment readiness confirmed');
    this.testResults.steps_completed?.push('deployment_readiness_check');
  }
  
  /**
   * Validate system state after deployment
   */
  private async validatePostDeploymentState(deploymentResult: any): Promise<void> {
    console.log('🔍 Validating deployment results...');
    
    // Check that all 6 steps completed
    const expectedSteps = [
      'final_pinned_benchmark',
      'tag_and_freeze',
      'canary_deployment',
      'post_deploy_calibration',
      'drift_monitoring_setup',
      'week_one_monitoring_setup'
    ];
    
    for (const step of expectedSteps) {
      const stepResult = deploymentResult.step_results[step];
      if (!stepResult || !stepResult.success) {
        throw new Error(`Step ${step} failed or incomplete`);
      }
      this.testResults.steps_completed?.push(step);
    }
    
    console.log('✅ All 6 TODO.md steps completed successfully');
    
    // Validate promotion gates
    console.log('🚪 Validating promotion gates...');
    const gateValidation = this.validatePromotionGates(deploymentResult.step_results.final_pinned_benchmark);
    this.testResults.promotion_gates_passed = gateValidation.passed;
    
    if (!gateValidation.passed) {
      console.warn('⚠️ Some promotion gates not met:');
      gateValidation.failed_gates.forEach(gate => {
        console.warn(`  - ${gate.name}: ${gate.actual} vs ${gate.expected}`);
      });
    } else {
      console.log('✅ All promotion gates passed');
    }
    
    // Check canary completion
    console.log('🕊️ Validating canary deployment...');
    const canaryResult = deploymentResult.step_results.canary_deployment;
    this.testResults.canary_completed = canaryResult.success && canaryResult.production_ready;
    
    if (this.testResults.canary_completed) {
      console.log('✅ Canary deployment completed successfully');
    } else {
      console.warn('⚠️ Canary deployment issues detected');
    }
    
    // Check calibration initiation
    console.log('📊 Validating calibration system...');
    const calibrationResult = deploymentResult.step_results.post_deploy_calibration;
    this.testResults.calibration_initiated = !!calibrationResult.session_id;
    
    if (this.testResults.calibration_initiated) {
      console.log(`✅ Calibration session initiated: ${calibrationResult.session_id}`);
      console.log(`   Holdout period: 2 days (per TODO.md)`);
      console.log(`   Current τ: ${calibrationResult.current_tau.toFixed(4)}`);
    }
    
    // Check monitoring systems
    console.log('📡 Validating monitoring systems...');
    const driftMonitoring = deploymentResult.step_results.drift_monitoring_setup;
    const weekOneMonitoring = deploymentResult.step_results.week_one_monitoring_setup;
    
    this.testResults.monitoring_active = driftMonitoring.active && weekOneMonitoring.active;
    
    if (this.testResults.monitoring_active) {
      console.log('✅ All monitoring systems active');
      console.log(`   Drift monitoring: ${driftMonitoring.monitors_configured} monitors configured`);
      console.log(`   Week-one monitoring: ${weekOneMonitoring.metrics_tracked} metrics tracked`);
    }
  }
  
  /**
   * Validate promotion gates from benchmark results
   */
  private validatePromotionGates(benchmarkResult: any): { passed: boolean; failed_gates: any[] } {
    const gates = [
      {
        name: 'Span Coverage',
        actual: benchmarkResult.span_coverage,
        expected: 100.0,
        operator: '=='
      },
      {
        name: 'Recall@50 Delta',
        actual: benchmarkResult.recall_at_50_delta,
        expected: 3.0,
        operator: '>='
      },
      {
        name: 'nDCG@10 Delta',
        actual: benchmarkResult.ndcg_at_10_delta,
        expected: 0.0,
        operator: '>='
      },
      {
        name: 'E2E P95 Latency Factor',
        actual: benchmarkResult.e2e_p95_latency_factor,
        expected: 1.1,
        operator: '<='
      },
      {
        name: 'E2E P99/P95 Ratio',
        actual: benchmarkResult.e2e_p99_p95_ratio,
        expected: 2.0,
        operator: '<='
      }
    ];
    
    const failedGates = gates.filter(gate => {
      switch (gate.operator) {
        case '==': return Math.abs(gate.actual - gate.expected) > 0.01;
        case '>=': return gate.actual < gate.expected;
        case '<=': return gate.actual > gate.expected;
        default: return false;
      }
    });
    
    return {
      passed: failedGates.length === 0,
      failed_gates: failedGates
    };
  }
  
  /**
   * Compile final test results
   */
  private compileFinalResults(
    deploymentResult: any,
    success: boolean,
    error?: any
  ): E2ETestResult {
    const endTime = new Date();
    const durationHours = (endTime.getTime() - this.testStartTime.getTime()) / (1000 * 60 * 60);
    
    // Extract final metrics
    const finalMetrics: Record<string, number> = {};
    
    if (deploymentResult?.step_results?.final_pinned_benchmark) {
      const benchmark = deploymentResult.step_results.final_pinned_benchmark;
      finalMetrics.span_coverage = benchmark.span_coverage || 0;
      finalMetrics.recall_at_50_delta = benchmark.recall_at_50_delta || 0;
      finalMetrics.ndcg_at_10_delta = benchmark.ndcg_at_10_delta || 0;
      finalMetrics.p95_latency_ms = benchmark.p95_latency_ms || 0;
      finalMetrics.p99_latency_ms = benchmark.p99_latency_ms || 0;
    }
    
    if (deploymentResult?.step_results?.post_deploy_calibration) {
      finalMetrics.current_tau = deploymentResult.step_results.post_deploy_calibration.current_tau || 0;
    }
    
    // Determine production readiness
    const productionReady = success && 
      this.testResults.promotion_gates_passed &&
      this.testResults.canary_completed &&
      this.testResults.calibration_initiated &&
      this.testResults.monitoring_active;
    
    if (error) {
      this.testResults.steps_failed?.push(`error: ${error.message}`);
    }
    
    return {
      success,
      deployment_id: this.testResults.deployment_id!,
      total_duration_hours: durationHours,
      steps_completed: this.testResults.steps_completed || [],
      steps_failed: this.testResults.steps_failed || [],
      promotion_gates_passed: this.testResults.promotion_gates_passed || false,
      canary_completed: this.testResults.canary_completed || false,
      calibration_initiated: this.testResults.calibration_initiated || false,
      monitoring_active: this.testResults.monitoring_active || false,
      final_metrics: finalMetrics,
      production_ready: productionReady
    };
  }
  
  /**
   * Display comprehensive final results
   */
  private displayFinalResults(result: E2ETestResult): void {
    const statusIcon = result.success ? '✅' : '❌';
    const readinessIcon = result.production_ready ? '🟢' : '🔴';
    
    console.log('\n📊 END-TO-END TEST RESULTS SUMMARY');
    console.log('=' .repeat(80));
    console.log(`${statusIcon} Overall Success: ${result.success}`);
    console.log(`${readinessIcon} Production Ready: ${result.production_ready}`);
    console.log(`🎯 Deployment ID: ${result.deployment_id}`);
    console.log(`⏱️ Total Duration: ${result.total_duration_hours.toFixed(2)} hours`);
    console.log('');
    
    console.log('📋 STEP COMPLETION STATUS:');
    console.log(`✅ Steps Completed: ${result.steps_completed.length}`);
    result.steps_completed.forEach(step => console.log(`   • ${step}`));
    
    if (result.steps_failed.length > 0) {
      console.log(`❌ Steps Failed: ${result.steps_failed.length}`);
      result.steps_failed.forEach(step => console.log(`   • ${step}`));
    }
    console.log('');
    
    console.log('🚪 PROMOTION GATES:');
    console.log(`Status: ${result.promotion_gates_passed ? '✅ PASSED' : '❌ FAILED'}`);
    console.log('');
    
    console.log('🕊️ CANARY DEPLOYMENT:');
    console.log(`Status: ${result.canary_completed ? '✅ COMPLETED' : '❌ INCOMPLETE'}`);
    console.log('');
    
    console.log('📊 CALIBRATION SYSTEM:');
    console.log(`Status: ${result.calibration_initiated ? '✅ ACTIVE' : '❌ INACTIVE'}`);
    if (result.final_metrics.current_tau) {
      console.log(`Current τ: ${result.final_metrics.current_tau.toFixed(4)}`);
    }
    console.log('');
    
    console.log('📡 MONITORING SYSTEMS:');
    console.log(`Status: ${result.monitoring_active ? '✅ ACTIVE' : '❌ INACTIVE'}`);
    console.log('');
    
    console.log('🎯 KEY METRICS:');
    Object.entries(result.final_metrics).forEach(([metric, value]) => {
      console.log(`   ${metric}: ${typeof value === 'number' ? value.toFixed(3) : value}`);
    });
    console.log('');
    
    if (result.production_ready) {
      console.log('🚀 PRODUCTION DEPLOYMENT APPROVED');
      console.log('   All systems validated and ready for production deployment');
      console.log('   Proceed with confidence to production rollout');
    } else {
      console.log('🛑 PRODUCTION DEPLOYMENT BLOCKED');
      console.log('   Critical issues must be resolved before production deployment');
      console.log('   Review failed steps and address issues before retry');
    }
    
    console.log('\n' + '=' .repeat(80));
  }
  
  /**
   * Setup event handlers for test monitoring
   */
  private setupEventHandlers(): void {
    this.orchestrator.on('step_completed', (event) => {
      console.log(`   ✅ Step completed: ${event.step_name} (${event.duration_minutes.toFixed(1)}m)`);
      this.testResults.steps_completed?.push(event.step_name);
    });
    
    this.orchestrator.on('step_failed', (event) => {
      console.log(`   ❌ Step failed: ${event.step_name} - ${event.error_message}`);
      this.testResults.steps_failed?.push(`${event.step_name}: ${event.error_message}`);
    });
    
    this.orchestrator.on('promotion_gate_result', (event) => {
      const icon = event.passed ? '✅' : '❌';
      console.log(`   ${icon} Promotion gate: ${event.gate_name} (${event.actual} vs ${event.expected})`);
    });
    
    this.orchestrator.on('system_alert', (alert) => {
      const icon = alert.severity === 'critical' ? '🚨' : 
                   alert.severity === 'warning' ? '⚠️' : 'ℹ️';
      console.log(`   ${icon} System alert: ${alert.message}`);
    });
  }
}

// Execute test if called directly
if (import.meta.main) {
  const test = new EndToEndProductionDeploymentTest();
  
  try {
    const result = await test.executeE2ETest();
    
    // Exit with appropriate code
    process.exit(result.production_ready ? 0 : 1);
    
  } catch (error) {
    console.error('💥 E2E test execution failed:', error);
    process.exit(1);
  }
}

export { EndToEndProductionDeploymentTest };