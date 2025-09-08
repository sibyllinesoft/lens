#!/usr/bin/env tsx

/**
 * Phase 4 Robustness Tests Demo
 * 
 * Demonstrates the Phase 4 robustness testing suite execution
 * Shows what the production validation would look like.
 */

import { promises as fs } from 'fs';
import path from 'path';
import { Phase4RobustnessTestSuite, Phase4TestConfig } from './src/benchmark/phase4-robustness-suite.js';
import { OperationalMonitoringSystem, MonitoringMetrics } from './src/benchmark/operational-monitoring.js';
import type { BenchmarkConfig } from './src/types/benchmark.js';

async function main() {
  console.log('🎯 Lens Search Engine - Phase 4 Robustness Testing Demo');
  console.log('=====================================================');
  
  // Configuration for Phase 4 testing
  const phase4Config: Phase4TestConfig = {
    repositories: [
      {
        name: 'lens-main',
        path: '/media/nathan/Seagate Hub/Projects/lens',
        language: 'typescript'
      },
      {
        name: 'sample-typescript',
        path: '/media/nathan/Seagate Hub/Projects/lens/sample-code',
        language: 'typescript'
      },
      {
        name: 'sample-storyviz',
        path: '/media/nathan/Seagate Hub/Projects/lens/sample-storyviz',
        language: 'typescript'
      }
    ],
    churnConfig: {
      modificationPercentage: 2, // 2% of files
      targetThroughputMaintenance: 5.0 // 5 files/second
    },
    qpsConfig: {
      backgroundLoad: 10, // 10 QPS background load
      compactionBounds: {
        latencyIncreaseMax: 1.5, // Max 50% increase
        availabilityMin: 0.99 // 99% availability
      }
    },
    latencyConfig: {
      p99AlertThreshold: 2.0, // Alert if p99 > 2x p95
      monitoringDurationMs: 30000 // 30 seconds
    }
  };

  const benchmarkConfig: BenchmarkConfig = {
    trace_id: crypto.randomUUID(),
    suite: ['codesearch', 'structural'],
    systems: ['lex+symbols+semantic'],
    slices: 'SMOKE_DEFAULT',
    seeds: 1,
    cache_mode: 'warm',
    robustness: true,
    metamorphic: false,
    k_candidates: 200,
    top_n: 50,
    fuzzy: 2,
    subtokens: true,
    semantic_gating: {
      nl_likelihood_threshold: 0.5,
      min_candidates: 10
    },
    latency_budgets: {
      stage_a_ms: 200,
      stage_b_ms: 300,
      stage_c_ms: 300
    }
  };

  try {
    // Create output directory
    const outputDir = path.join(process.cwd(), 'phase4-demo-results');
    await fs.mkdir(outputDir, { recursive: true });
    console.log(`📁 Results will be saved to: ${outputDir}\n`);

    // Initialize test suite and monitoring
    const testSuite = new Phase4RobustnessTestSuite(outputDir, phase4Config);
    const monitoring = new OperationalMonitoringSystem(outputDir);

    console.log('🏃 Running Phase 4 Robustness Tests...\n');

    // Run the complete test suite
    const results = await testSuite.runPhase4Tests(benchmarkConfig);

    // Display comprehensive results
    console.log('\n' + '='.repeat(70));
    console.log('📊 PHASE 4 ROBUSTNESS TESTING RESULTS');
    console.log('='.repeat(70));

    // Overall status
    const statusIcon = results.overallStatus === 'PASS' ? '✅' : '❌';
    console.log(`\n🎯 Overall Status: ${statusIcon} ${results.overallStatus}`);

    // Multi-repository testing results
    console.log(`\n🔍 Multi-Repository Testing:`);
    console.log(`   Repositories tested: ${results.multiRepoResults.length}`);
    
    let passedRepos = 0;
    results.multiRepoResults.forEach(repo => {
      const status = repo.qualityGatesPassed ? '✅' : '❌';
      if (repo.qualityGatesPassed) passedRepos++;
      
      console.log(`   ${status} ${repo.repository} (${repo.language}):`);
      console.log(`      Indexing time: ${repo.metrics.indexingTimeMs}ms`);
      console.log(`      Search P95: ${repo.metrics.searchLatencyP95.toFixed(1)}ms`);
      console.log(`      Search P99: ${repo.metrics.searchLatencyP99.toFixed(1)}ms`);
      console.log(`      Recall@50: ${(repo.metrics.recallAt50 * 100).toFixed(1)}%`);
      console.log(`      NDCG@10: ${(repo.metrics.ndcgAt10 * 100).toFixed(1)}%`);
      
      if (repo.errors.length > 0) {
        console.log(`      Errors: ${repo.errors.join(', ')}`);
      }
    });

    console.log(`   Quality Gate Pass Rate: ${(passedRepos / results.multiRepoResults.length * 100).toFixed(1)}%`);

    // Churn testing results
    console.log(`\n🔄 Churn Testing (File Modification Resilience):`);
    console.log(`   Files modified: ${results.churnTestResult.filesModified} (${results.churnTestResult.filesModifiedPercentage}%)`);
    console.log(`   Rebuild time: ${results.churnTestResult.rebuildTimeMs}ms`);
    console.log(`   Rebuild throughput: ${results.churnTestResult.rebuildThroughput.toFixed(2)} files/sec`);
    console.log(`   Quality maintained: ${results.churnTestResult.qualityMaintained ? '✅' : '❌'}`);
    console.log(`     Pre-churn recall: ${(results.churnTestResult.metrics.preChurnRecall * 100).toFixed(1)}%`);
    console.log(`     Post-churn recall: ${(results.churnTestResult.metrics.postChurnRecall * 100).toFixed(1)}%`);
    console.log(`     Recall delta: ${(results.churnTestResult.metrics.recallDelta * 100).toFixed(2)}%`);
    console.log(`   Incremental rebuild: ${results.churnTestResult.incrementalRebuildWorked ? '✅' : '❌'}`);
    console.log(`     Shards affected: ${results.churnTestResult.metrics.affectedShards}/${results.churnTestResult.metrics.totalShards} (${results.churnTestResult.metrics.shardsAffectedPercentage.toFixed(1)}%)`);

    // Compaction under load results
    console.log(`\n🗜️  Compaction Under Load Testing:`);
    results.compactionResults.forEach((result, index) => {
      console.log(`   Test ${index + 1} (${result.backgroundQPS} QPS background load):`);
      console.log(`     Compaction duration: ${result.compactionDurationMs}ms`);
      console.log(`     Service availability: ${(result.serviceAvailabilityDuringCompaction * 100).toFixed(2)}%`);
      console.log(`     Latency impact:`);
      console.log(`       Pre-compaction P95: ${result.latencyMetrics.preCompactionP95.toFixed(1)}ms`);
      console.log(`       During compaction P95: ${result.latencyMetrics.duringCompactionP95.toFixed(1)}ms`);
      console.log(`       Post-compaction P95: ${result.latencyMetrics.postCompactionP95.toFixed(1)}ms`);
      console.log(`       Latency increase: ${result.latencyMetrics.latencyBumpRatio.toFixed(2)}x`);
      console.log(`     Partial service continued: ${result.partialServiceContinued ? '✅' : '❌'}`);
      console.log(`     Data corruption detected: ${result.dataCorruption ? '❌ YES' : '✅ NO'}`);
    });

    // Tail latency analysis
    console.log(`\n📊 Tail Latency Analysis:`);
    results.tailLatencyAnalysis.forEach(analysis => {
      const p99ToP95Ratio = analysis.metrics.p99 / analysis.metrics.p95;
      const alertIcon = analysis.alertsTriggered.length > 0 ? '⚠️' : '✅';
      
      console.log(`   ${analysis.stage.toUpperCase()}:`);
      console.log(`     P50: ${analysis.metrics.p50.toFixed(1)}ms`);
      console.log(`     P95: ${analysis.metrics.p95.toFixed(1)}ms`);
      console.log(`     P99: ${analysis.metrics.p99.toFixed(1)}ms`);
      console.log(`     P99.9: ${analysis.metrics.p99_9.toFixed(1)}ms`);
      console.log(`     Max: ${analysis.metrics.max.toFixed(1)}ms`);
      console.log(`     P99/P95 Ratio: ${p99ToP95Ratio.toFixed(2)}x ${alertIcon}`);
      
      if (analysis.alertsTriggered.length > 0) {
        console.log(`     Alerts triggered: ${analysis.alertsTriggered.length}`);
        analysis.alertsTriggered.forEach(alert => {
          console.log(`       ${alert.severity.toUpperCase()}: ${alert.condition} (${alert.actualValue.toFixed(2)} > ${alert.threshold})`);
        });
      }

      if (analysis.worstCaseScenarios.length > 0) {
        console.log(`     Worst-case scenarios: ${analysis.worstCaseScenarios.length}`);
        analysis.worstCaseScenarios.forEach(scenario => {
          console.log(`       ${scenario.scenario}: ${scenario.latencyMs.toFixed(1)}ms (${(scenario.frequency * 100).toFixed(2)}% frequency)`);
          console.log(`         Root cause: ${scenario.rootCause}`);
        });
      }
    });

    // Production recommendations
    console.log(`\n💡 Production Recommendations:`);
    results.recommendationsForProduction.forEach(rec => {
      const icon = rec.startsWith('✅') ? '' : '  •';
      console.log(`${icon} ${rec}`);
    });

    // Demonstrate monitoring system
    console.log(`\n📈 Operational Monitoring Demo:`);
    
    // Simulate some monitoring metrics
    const mockMetrics: MonitoringMetrics = {
      timestamp: new Date().toISOString(),
      stage: 'e2e',
      latency_p50: 18.5,
      latency_p95: 25.2,
      latency_p99: 45.1,
      latency_p99_9: 87.3,
      latency_max: 120.5,
      throughput_qps: 25.8,
      error_rate: 0.005,
      availability: 0.998,
      cache_hit_rate: 0.85,
      index_size_mb: 1250,
      memory_usage_mb: 512,
      cpu_utilization_percent: 35
    };

    const alerts = await monitoring.recordMetrics(mockMetrics);
    const status = monitoring.getCurrentStatus();
    const runbook = await monitoring.generateRunbook();

    console.log(`   System Status: ${status.status.toUpperCase()}`);
    console.log(`   Active Alerts: ${status.activeAlerts.length}`);
    console.log(`   Latest Metrics:`);
    console.log(`     Latency P95: ${status.recentMetrics.latency_p95}ms`);
    console.log(`     Latency P99: ${status.recentMetrics.latency_p99}ms`);
    console.log(`     Throughput: ${status.recentMetrics.throughput_qps} QPS`);
    console.log(`     Availability: ${(status.recentMetrics.availability * 100).toFixed(2)}%`);

    if (alerts.length > 0) {
      console.log(`   New Alerts: ${alerts.length}`);
      alerts.forEach(alert => {
        console.log(`     ${alert.severity.toUpperCase()}: ${alert.message}`);
      });
    }

    // Test performance summary
    console.log(`\n⚡ Test Suite Performance:`);
    console.log(`   Test scenarios executed: 4`);
    console.log(`   Repositories validated: ${phase4Config.repositories.length}`);
    console.log(`   Output files generated: Multiple reports in ${outputDir}`);

    // Final verdict
    console.log(`\n🏁 Final Assessment:`);
    if (results.overallStatus === 'PASS') {
      console.log(`   ✅ System demonstrates production readiness`);
      console.log(`   ✅ All robustness tests passed`);
      console.log(`   ✅ Quality maintained across multiple repositories`);
      console.log(`   ✅ Incremental operations work correctly`);
      console.log(`   ✅ System handles operational stress appropriately`);
      console.log(`   ✅ Monitoring and alerting framework operational`);
      console.log(`\n🚀 RECOMMENDATION: System is READY for production deployment!`);
    } else {
      console.log(`   ❌ System needs improvements before production`);
      console.log(`   ❌ Address the recommendations above`);
      console.log(`   ❌ Re-run Phase 4 tests after fixes`);
      console.log(`\n⚠️  RECOMMENDATION: DO NOT deploy to production yet.`);
    }

    console.log('\n' + '='.repeat(70));
    console.log(`📊 Detailed reports saved in: ${outputDir}`);
    console.log(`📖 Operational runbook available: ${path.join(outputDir, 'operational-runbook.md')}`);

  } catch (error) {
    console.error('\n💥 Phase 4 testing failed:', error);
    console.error('This indicates critical issues that must be resolved before production.');
    process.exit(1);
  }
}

// Run the demo
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}

export { main as runPhase4Demo };