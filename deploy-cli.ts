#!/usr/bin/env node
/**
 * Deployment CLI - Command Line Interface
 * 
 * Provides command-line access to the complete deployment system:
 * - `deploy start` - Execute full deployment pipeline
 * - `deploy status` - Check deployment status and system health
 * - `deploy abort` - Emergency abort current deployment
 * - `deploy canary` - Manage canary rollouts
 * - `deploy bench` - Run benchmarks and validation
 * - `deploy monitor` - Monitor system health and alarms
 * - `deploy sentinel` - Manage sentinel probes and kill switches
 */

import { program } from 'commander';
import { deploymentOrchestrator } from './src/deployment/deployment-orchestrator.js';
import { versionManager } from './src/deployment/version-manager.js';
import { finalBenchSystem } from './src/deployment/final-bench-system.js';
import { canaryRolloutSystem } from './src/deployment/canary-rollout-system.js';
import { onlineCalibrationSystem } from './src/deployment/online-calibration-system.js';
import { productionMonitoringSystem } from './src/deployment/production-monitoring-system.js';
import { sentinelKillSwitchSystem } from './src/deployment/sentinel-kill-switch-system.js';

/**
 * Main deployment command - execute full pipeline
 */
program
  .command('start')
  .description('Start complete deployment pipeline (Tag+Freeze → FinalBench → Canary → Calibration → Monitoring → Sentinel)')
  .option('-v, --version <version>', 'Target version (auto-generated if not specified)')
  .option('--skip-bench', 'Skip final benchmark validation')
  .option('--skip-canary', 'Skip canary rollout (immediate deployment)')
  .option('--accelerated', 'Use accelerated canary rollout (shorter stage durations)')
  .option('--force', 'Force deployment even if gates fail')
  .action(async (options) => {
    try {
      console.log('🚀 Starting complete deployment pipeline...');
      console.log('📋 Pipeline: Tag+Freeze → FinalBench → Canary(A→B→C) → Calibration → Monitoring → Sentinel');
      console.log();
      
      const config = {
        target_version: options.version,
        skip_final_bench: options.skipBench,
        required_gate_success: !options.force,
        canary_config: {
          skip_canary: options.skipCanary,
          accelerated_rollout: options.accelerated,
          stage_duration_hours: options.accelerated ? 2 : 24
        }
      };
      
      if (options.force) {
        console.log('⚠️  WARNING: Force mode enabled - gates failures will be ignored');
      }
      
      const version = await deploymentOrchestrator.executeDeploymentPipeline(config);
      
      console.log();
      console.log('🎉 Deployment pipeline completed successfully!');
      console.log(`✅ Version ${version} is now GA with full monitoring active`);
      console.log();
      console.log('📊 Active systems:');
      console.log('  • Canary rollout: 100% traffic');
      console.log('  • Online calibration: Daily reliability curve updates');
      console.log('  • Production monitoring: CUSUM drift alarms');
      console.log('  • Sentinel probes: Hourly health validation');
      console.log();
      console.log('💡 Next steps:');
      console.log('  • Monitor: npm run deploy status');
      console.log('  • Dashboard: npm run deploy monitor --dashboard');
      console.log('  • Health check: npm run deploy sentinel --status');
      
    } catch (error) {
      console.error('❌ Deployment failed:', error.message);
      console.log();
      console.log('🔧 Recovery options:');
      console.log('  • Check status: npm run deploy status');
      console.log('  • Emergency abort: npm run deploy abort "reason"');
      console.log('  • Review logs: ./deployment-artifacts/orchestrator/');
      process.exit(1);
    }
  });

/**
 * Status command - check deployment and system health
 */
program
  .command('status')
  .description('Check current deployment status and system health')
  .option('--detailed', 'Show detailed phase information')
  .option('--json', 'Output as JSON')
  .action(async (options) => {
    try {
      const pipelineStatus = deploymentOrchestrator.getCurrentPipelineStatus();
      const dashboardData = deploymentOrchestrator.getDashboardData();
      const healthStatus = productionMonitoringSystem.getHealthStatus();
      const sentinelStatus = sentinelKillSwitchSystem.getSystemStatus();
      
      if (options.json) {
        console.log(JSON.stringify({
          pipeline: pipelineStatus,
          dashboard: dashboardData,
          health: healthStatus,
          sentinel: sentinelStatus
        }, null, 2));
        return;
      }
      
      console.log('📊 Deployment Status Report');
      console.log('=' .repeat(50));
      console.log();
      
      if (pipelineStatus) {
        console.log(`🚀 Pipeline: Version ${pipelineStatus.version}`);
        console.log(`📍 Status: ${pipelineStatus.overall_status.toUpperCase()}`);
        console.log(`📋 Current Phase: ${pipelineStatus.current_phase}`);
        console.log(`⏱️  Duration: ${dashboardData.current_pipeline?.duration_hours?.toFixed(1)} hours`);
        console.log(`✅ Phases Complete: ${pipelineStatus.phase_history.filter(p => p.status === 'completed').length}/9`);
        console.log();
        
        if (options.detailed) {
          console.log('📋 Phase History:');
          pipelineStatus.phase_history.forEach(phase => {
            const status = phase.status === 'completed' ? '✅' : 
                          phase.status === 'failed' ? '❌' : 
                          phase.status === 'running' ? '🔄' : '⏸️';
            const duration = phase.duration_ms ? `(${(phase.duration_ms / 1000).toFixed(1)}s)` : '';
            console.log(`  ${status} ${phase.phase} ${duration}`);
            if (phase.error_message) {
              console.log(`     Error: ${phase.error_message}`);
            }
          });
          console.log();
        }
      } else {
        console.log('📋 No active deployment pipeline');
        console.log();
      }
      
      console.log('🏥 System Health:');
      console.log(`  Overall: ${healthStatus.overall_status.toUpperCase()}`);
      console.log(`  Sentinel Success Rate: ${(sentinelStatus.health.sentinel_passing_rate * 100).toFixed(1)}%`);
      console.log(`  Active Alarms: ${healthStatus.active_alarms.length}`);
      console.log(`  Kill Switches Active: ${sentinelStatus.health.active_kill_switches.length}`);
      
      if (healthStatus.active_alarms.length > 0) {
        console.log('⚠️  Active Alarms:');
        healthStatus.active_alarms.forEach(alarm => console.log(`    • ${alarm}`));
      }
      
      console.log();
      console.log('🎯 System Components:');
      console.log(`  Canary: ${dashboardData.system_status.canary_active ? '✅ Active' : '⏸️  Inactive'}`);
      console.log(`  Calibration: ${dashboardData.system_status.calibration_active ? '✅ Active' : '⏸️  Inactive'}`);
      console.log(`  Monitoring: ${dashboardData.system_status.monitoring_active ? '✅ Active' : '⏸️  Inactive'}`);
      console.log(`  Sentinel: ${dashboardData.system_status.sentinel_active ? '✅ Active' : '⏸️  Inactive'}`);
      
    } catch (error) {
      console.error('❌ Failed to get status:', error.message);
      process.exit(1);
    }
  });

/**
 * Abort command - emergency deployment abort
 */
program
  .command('abort <reason>')
  .description('Emergency abort current deployment pipeline')
  .action(async (reason) => {
    try {
      console.log('🚨 EMERGENCY DEPLOYMENT ABORT');
      console.log(`🔧 Reason: ${reason}`);
      console.log();
      
      await deploymentOrchestrator.emergencyAbortPipeline(reason);
      
      console.log('✅ Deployment pipeline aborted successfully');
      console.log('🧹 Cleanup actions performed');
      console.log();
      console.log('💡 Next steps:');
      console.log('  • Review failure report: ./deployment-artifacts/orchestrator/');
      console.log('  • Check system health: npm run deploy status');
      console.log('  • Manual recovery may be required');
      
    } catch (error) {
      console.error('❌ Failed to abort deployment:', error.message);
      process.exit(1);
    }
  });

/**
 * Canary command - manage canary rollouts
 */
const canaryCmd = program
  .command('canary')
  .description('Manage canary rollouts');

canaryCmd
  .command('status')
  .description('Show canary rollout status')
  .option('--deployment-id <id>', 'Specific deployment ID')
  .action(async (options) => {
    try {
      if (options.deploymentId) {
        const status = canaryRolloutSystem.getDeploymentStatus(options.deploymentId);
        if (!status) {
          console.log(`❌ Deployment not found: ${options.deploymentId}`);
          return;
        }
        
        console.log(`🕯️  Canary Status: ${options.deploymentId}`);
        console.log(`📍 Block: ${status.block_id}`);
        console.log(`📊 Traffic: ${status.traffic_percentage}%`);
        console.log(`⏱️  Status: ${status.status}`);
        console.log(`🎯 Stage: ${status.current_stage + 1}`);
        
        if (status.metrics_snapshot) {
          console.log();
          console.log('📈 Current Metrics:');
          console.log(`  nDCG@10: ${(status.metrics_snapshot.ndcg_at_10 * 100).toFixed(1)}%`);
          console.log(`  Recall@50: ${(status.metrics_snapshot.recall_at_50 * 100).toFixed(1)}%`);
          console.log(`  P95 Latency: ${status.metrics_snapshot.p95_latency_ms.toFixed(0)}ms`);
        }
        
      } else {
        const activeDeployments = canaryRolloutSystem.getActiveDeployments();
        const dashboardData = await canaryRolloutSystem.getDashboardData();
        
        console.log('🕯️  Canary Rollout Status');
        console.log(`📊 Active Deployments: ${activeDeployments.length}`);
        
        if (dashboardData.activeDeployments.length > 0) {
          console.log();
          dashboardData.activeDeployments.forEach((deployment: any) => {
            console.log(`  ${deployment.deploymentId}:`);
            console.log(`    Block: ${deployment.block} (${deployment.traffic}% traffic)`);
            console.log(`    Status: ${deployment.status}`);
          });
        }
      }
      
    } catch (error) {
      console.error('❌ Failed to get canary status:', error.message);
      process.exit(1);
    }
  });

canaryCmd
  .command('rollback <deployment-id> <reason>')
  .description('Manual rollback of canary deployment')
  .action(async (deploymentId, reason) => {
    try {
      console.log(`🔄 Rolling back canary deployment: ${deploymentId}`);
      console.log(`🔧 Reason: ${reason}`);
      
      await canaryRolloutSystem.manualRollback(deploymentId, reason);
      
      console.log('✅ Canary rollback completed');
      
    } catch (error) {
      console.error('❌ Failed to rollback canary:', error.message);
      process.exit(1);
    }
  });

/**
 * Bench command - run benchmarks
 */
const benchCmd = program
  .command('bench')
  .description('Run benchmarks and validation');

benchCmd
  .command('run')
  .description('Run final benchmark validation')
  .option('-v, --version <version>', 'Version to benchmark')
  .action(async (options) => {
    try {
      console.log('📊 Running final benchmark validation...');
      
      const result = await finalBenchSystem.runFinalValidation(options.version);
      
      console.log(`✅ Benchmark completed: ${result.validation_status.passed ? 'PASSED' : 'FAILED'}`);
      console.log();
      console.log('📈 Results:');
      console.log(`  P@1: ${(result.aggregate_metrics.mean_p_at_1 * 100).toFixed(1)}%`);
      console.log(`  nDCG@10: ${(result.aggregate_metrics.mean_ndcg_at_10 * 100).toFixed(1)}%`);
      console.log(`  Recall@50: ${(result.aggregate_metrics.mean_recall_at_50 * 100).toFixed(1)}%`);
      console.log(`  Span Coverage: ${(result.aggregate_metrics.span_coverage_rate * 100).toFixed(1)}%`);
      
      if (!result.validation_status.passed) {
        console.log();
        console.log('❌ Gate Failures:');
        result.validation_status.issues.forEach((issue: string) => console.log(`  • ${issue}`));
        process.exit(1);
      }
      
    } catch (error) {
      console.error('❌ Benchmark failed:', error.message);
      process.exit(1);
    }
  });

/**
 * Monitor command - system monitoring
 */
const monitorCmd = program
  .command('monitor')
  .description('Monitor system health and alarms');

monitorCmd
  .command('status')
  .description('Show monitoring status')
  .action(async () => {
    try {
      const healthStatus = productionMonitoringSystem.getHealthStatus();
      const cusumStatus = productionMonitoringSystem.getCUSUMStatus();
      const dashboardData = productionMonitoringSystem.getDashboardData();
      
      console.log('📊 Production Monitoring Status');
      console.log('=' .repeat(40));
      console.log();
      
      console.log(`🏥 Overall Health: ${healthStatus.overall_status.toUpperCase()}`);
      console.log(`⏰ Last Check: ${new Date(healthStatus.last_health_check).toLocaleString()}`);
      console.log(`⏱️  System Uptime: ${healthStatus.system_uptime_hours.toFixed(1)} hours`);
      console.log();
      
      console.log('📈 CUSUM Detectors:');
      Object.entries(cusumStatus).forEach(([name, detector]) => {
        const status = detector.alarm_active ? '🚨 ALARM' : '✅ OK';
        console.log(`  ${name}: ${status} (failures: ${detector.consecutive_violations})`);
      });
      
      if (healthStatus.active_alarms.length > 0) {
        console.log();
        console.log('⚠️  Active Alarms:');
        healthStatus.active_alarms.forEach(alarm => console.log(`  • ${alarm}`));
      }
      
    } catch (error) {
      console.error('❌ Failed to get monitoring status:', error.message);
      process.exit(1);
    }
  });

monitorCmd
  .command('reset <metric>')
  .description('Reset CUSUM detector for metric')
  .action(async (metric) => {
    try {
      console.log(`🔧 Resetting CUSUM detector: ${metric}`);
      
      productionMonitoringSystem.resetCUSUMDetector(metric);
      
      console.log('✅ CUSUM detector reset');
      
    } catch (error) {
      console.error('❌ Failed to reset CUSUM detector:', error.message);
      process.exit(1);
    }
  });

/**
 * Sentinel command - manage probes and kill switches
 */
const sentinelCmd = program
  .command('sentinel')
  .description('Manage sentinel probes and kill switches');

sentinelCmd
  .command('status')
  .description('Show sentinel system status')
  .action(async () => {
    try {
      const status = sentinelKillSwitchSystem.getSystemStatus();
      
      console.log('🕵️ Sentinel System Status');
      console.log('=' .repeat(40));
      console.log();
      
      console.log(`🏥 System Health: ${status.health.overall_status.toUpperCase()}`);
      console.log(`📊 Probe Success Rate: ${(status.health.sentinel_passing_rate * 100).toFixed(1)}%`);
      console.log(`🚨 Emergency Mode: ${status.health.emergency_mode_active ? 'ACTIVE' : 'Inactive'}`);
      console.log();
      
      console.log('🔍 Sentinel Probes:');
      Object.entries(status.probes).forEach(([id, probe]) => {
        const status = probe.consecutive_failures > 0 ? 
          `❌ ${probe.consecutive_failures} failures` : 
          '✅ Passing';
        console.log(`  ${probe.name}: ${status} (${(probe.success_rate * 100).toFixed(1)}%)`);
      });
      console.log();
      
      console.log('🔒 Kill Switches:');
      Object.entries(status.kill_switches).forEach(([id, ks]) => {
        const status = ks.is_active ? '🚨 ACTIVE' : '⏸️  Inactive';
        console.log(`  ${ks.name}: ${status}`);
        if (ks.is_active && ks.activated_at) {
          console.log(`    Activated: ${new Date(ks.activated_at).toLocaleString()}`);
        }
      });
      
    } catch (error) {
      console.error('❌ Failed to get sentinel status:', error.message);
      process.exit(1);
    }
  });

sentinelCmd
  .command('probe <probe-id>')
  .description('Execute specific probe manually')
  .action(async (probeId) => {
    try {
      console.log(`🕵️ Executing probe: ${probeId}`);
      
      const result = await sentinelKillSwitchSystem.manualProbeExecution(probeId);
      
      console.log(`${result.success ? '✅' : '❌'} Probe ${result.success ? 'passed' : 'failed'}`);
      console.log(`⏱️  Execution time: ${result.execution_time_ms}ms`);
      console.log(`📊 Results: ${result.results_count}`);
      
      if (!result.success && result.error_message) {
        console.log(`❌ Error: ${result.error_message}`);
      }
      
    } catch (error) {
      console.error('❌ Failed to execute probe:', error.message);
      process.exit(1);
    }
  });

sentinelCmd
  .command('killswitch <action> <switch-id>')
  .description('Manage kill switches (activate/deactivate)')
  .option('-r, --reason <reason>', 'Reason for action')
  .action(async (action, switchId, options) => {
    try {
      const reason = options.reason || 'Manual CLI action';
      
      if (action === 'activate') {
        console.log(`🚨 Activating kill switch: ${switchId}`);
        console.log(`🔧 Reason: ${reason}`);
        
        await sentinelKillSwitchSystem.activateKillSwitch(switchId, 'manual', reason);
        
        console.log('✅ Kill switch activated');
        
      } else if (action === 'deactivate') {
        console.log(`🔄 Deactivating kill switch: ${switchId}`);
        console.log(`🔧 Reason: ${reason}`);
        
        await sentinelKillSwitchSystem.deactivateKillSwitch(switchId, reason);
        
        console.log('✅ Kill switch deactivated');
        
      } else {
        console.error('❌ Invalid action. Use: activate or deactivate');
        process.exit(1);
      }
      
    } catch (error) {
      console.error('❌ Failed to manage kill switch:', error.message);
      process.exit(1);
    }
  });

/**
 * Version command - manage versions
 */
const versionCmd = program
  .command('version')
  .description('Manage deployment versions');

versionCmd
  .command('list')
  .description('List available versions')
  .action(() => {
    try {
      const versions = versionManager.getAvailableVersions();
      const current = versionManager.getCurrentVersion();
      
      console.log('📦 Available Versions:');
      versions.forEach(version => {
        const marker = version === current ? ' (current)' : '';
        console.log(`  • ${version}${marker}`);
      });
      
    } catch (error) {
      console.error('❌ Failed to list versions:', error.message);
      process.exit(1);
    }
  });

versionCmd
  .command('show [version]')
  .description('Show version configuration')
  .action((version) => {
    try {
      const config = versionManager.loadVersionConfig(version);
      
      console.log(`📦 Version Configuration: ${config.version}`);
      console.log(`🏷️  Created: ${config.timestamp}`);
      console.log(`🔧 Git Commit: ${config.git_commit}`);
      console.log();
      console.log('⚙️  Parameters:');
      console.log(`  Tau: ${config.tau_value}`);
      console.log(`  LTR Model: ${config.ltr_model_hash.substring(0, 16)}...`);
      console.log(`  Early Exit: margin=${config.early_exit_config.margin}, min_probes=${config.early_exit_config.min_probes}`);
      console.log(`  Deduplication: k=${config.dedup_params.k}, hamming_max=${config.dedup_params.hamming_max}`);
      console.log();
      console.log('📊 Baseline Metrics:');
      console.log(`  P@1: ${(config.baseline_metrics.p_at_1 * 100).toFixed(1)}%`);
      console.log(`  nDCG@10: ${(config.baseline_metrics.ndcg_at_10 * 100).toFixed(1)}%`);
      console.log(`  Recall@50: ${(config.baseline_metrics.recall_at_50 * 100).toFixed(1)}%`);
      
    } catch (error) {
      console.error('❌ Failed to show version:', error.message);
      process.exit(1);
    }
  });

/**
 * Main program configuration
 */
program
  .name('deploy')
  .description('Lens Deployment Pipeline CLI')
  .version('1.0.0')
  .hook('preAction', () => {
    // Add timestamp to all output
    const originalLog = console.log;
    console.log = (...args) => {
      originalLog(`[${new Date().toISOString()}]`, ...args);
    };
  });

// Global error handling
process.on('unhandledRejection', (reason, promise) => {
  console.error('❌ Unhandled Promise Rejection:', reason);
  process.exit(1);
});

process.on('uncaughtException', (error) => {
  console.error('❌ Uncaught Exception:', error.message);
  process.exit(1);
});

// Parse arguments
program.parse();

// If no command provided, show help
if (!process.argv.slice(2).length) {
  program.outputHelp();
}