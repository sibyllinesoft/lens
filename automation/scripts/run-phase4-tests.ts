#!/usr/bin/env tsx

/**
 * Phase 4 Robustness & Ops Test Runner
 * 
 * Comprehensive production readiness validation suite.
 * Executes all Phase 4 robustness tests to prove system stability under operational conditions.
 * 
 * Usage:
 *   npm run phase4-tests
 *   npx tsx run-phase4-tests.ts
 */

import { promises as fs } from 'fs';
import path from 'path';
import { Phase4RobustnessTestSuite, Phase4TestConfig } from './src/benchmark/phase4-robustness-suite.js';
import type { BenchmarkConfig } from './src/types/benchmark.js';

async function main() {
  console.log('🚀 Lens Search Engine - Phase 4: Robustness & Ops Testing');
  console.log('========================================================');
  
  const startTime = Date.now();
  
  // Configuration for Phase 4 testing
  const phase4Config: Phase4TestConfig = {
    repositories: [
      {
        name: 'lens',
        path: '/media/nathan/Seagate Hub/Projects/lens',
        language: 'typescript'
      },
      {
        name: 'sample-typescript',
        path: '/media/nathan/Seagate Hub/Projects/lens/sample-code',
        language: 'typescript'
      },
      {
        name: 'storyviz-content',
        path: '/media/nathan/Seagate Hub/Projects/lens/sample-storyviz',
        language: 'typescript'
      }
    ],
    churnConfig: {
      modificationPercentage: 2, // Modify 2% of files
      targetThroughputMaintenance: 5.0 // files/second rebuild throughput
    },
    qpsConfig: {
      backgroundLoad: 10, // 10 QPS during compaction
      compactionBounds: {
        latencyIncreaseMax: 1.5, // Max 50% latency increase
        availabilityMin: 0.99 // Min 99% availability
      }
    },
    latencyConfig: {
      p99AlertThreshold: 2.0, // Alert if p99 > 2x p95
      monitoringDurationMs: 30000 // 30 seconds of monitoring
    }
  };

  // Benchmark configuration for testing
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
    const outputDir = path.join(process.cwd(), 'phase4-results');
    await fs.mkdir(outputDir, { recursive: true });
    console.log(`📁 Output directory: ${outputDir}`);

    // Initialize test suite
    const testSuite = new Phase4RobustnessTestSuite(outputDir, phase4Config);

    // Run complete Phase 4 test suite
    console.log('\n🔍 Executing Phase 4 robustness test suite...\n');
    const results = await testSuite.runPhase4Tests(benchmarkConfig);

    // Display results summary
    console.log('\n' + '='.repeat(60));
    console.log('📊 PHASE 4 RESULTS SUMMARY');
    console.log('='.repeat(60));
    
    console.log(`\n🎯 Overall Status: ${results.overallStatus === 'PASS' ? '✅ PASS' : '❌ FAIL'}`);
    
    // Multi-repo results
    const multiRepoPassRate = results.multiRepoResults.filter(r => r.qualityGatesPassed).length / results.multiRepoResults.length;
    console.log(`\n🔍 Multi-Repository Testing:`);
    console.log(`   Repositories tested: ${results.multiRepoResults.length}`);
    console.log(`   Quality gates passed: ${(multiRepoPassRate * 100).toFixed(1)}% (${results.multiRepoResults.filter(r => r.qualityGatesPassed).length}/${results.multiRepoResults.length})`);
    
    results.multiRepoResults.forEach(repo => {
      const status = repo.qualityGatesPassed ? '✅' : '❌';
      console.log(`     ${status} ${repo.repository} (${repo.language}): P95=${repo.metrics.searchLatencyP95.toFixed(1)}ms, Recall@50=${(repo.metrics.recallAt50 * 100).toFixed(1)}%`);
    });

    // Churn test results
    console.log(`\n🔄 Churn Testing:`);
    console.log(`   Files modified: ${results.churnTestResult.filesModified} (${results.churnTestResult.filesModifiedPercentage}%)`);
    console.log(`   Rebuild throughput: ${results.churnTestResult.rebuildThroughput.toFixed(2)} files/sec`);
    console.log(`   Quality maintained: ${results.churnTestResult.qualityMaintained ? '✅' : '❌'} (Δ${(results.churnTestResult.metrics.recallDelta * 100).toFixed(2)}%)`);
    console.log(`   Incremental rebuild: ${results.churnTestResult.incrementalRebuildWorked ? '✅' : '❌'} (${results.churnTestResult.metrics.shardsAffectedPercentage.toFixed(1)}% shards affected)`);

    // Compaction results
    console.log(`\n🗜️  Compaction Under Load:`);
    results.compactionResults.forEach((result, index) => {
      console.log(`   Test ${index + 1}:`);
      console.log(`     Background QPS: ${result.backgroundQPS}`);
      console.log(`     Service availability: ${(result.serviceAvailabilityDuringCompaction * 100).toFixed(2)}%`);
      console.log(`     Latency increase: ${result.latencyMetrics.latencyBumpRatio.toFixed(2)}x (${result.latencyMetrics.preCompactionP95.toFixed(1)}ms → ${result.latencyMetrics.duringCompactionP95.toFixed(1)}ms)`);
      console.log(`     Partial service continued: ${result.partialServiceContinued ? '✅' : '❌'}`);
      console.log(`     Data corruption: ${result.dataCorruption ? '❌' : '✅'}`);
    });

    // Tail latency analysis
    console.log(`\n📊 Tail Latency Analysis:`);
    results.tailLatencyAnalysis.forEach(analysis => {
      const p99ToP95Ratio = analysis.metrics.p99 / analysis.metrics.p95;
      const alertStatus = analysis.alertsTriggered.length > 0 ? `⚠️  ${analysis.alertsTriggered.length} alerts` : '✅';
      console.log(`   ${analysis.stage}: P50=${analysis.metrics.p50.toFixed(1)}ms, P95=${analysis.metrics.p95.toFixed(1)}ms, P99=${analysis.metrics.p99.toFixed(1)}ms (${p99ToP95Ratio.toFixed(2)}x) ${alertStatus}`);
    });

    // Production recommendations
    console.log(`\n💡 Production Recommendations:`);
    results.recommendationsForProduction.forEach(rec => {
      console.log(`   ${rec}`);
    });

    // Test performance metrics
    const totalDuration = Date.now() - startTime;
    console.log(`\n⏱️  Test Suite Performance:`);
    console.log(`   Total duration: ${(totalDuration / 1000).toFixed(2)}s`);
    console.log(`   Repositories tested: ${phase4Config.repositories.length}`);
    console.log(`   Test scenarios executed: 4 (multi-repo, churn, compaction, tail latency)`);

    // Exit with appropriate code
    const exitCode = results.overallStatus === 'PASS' ? 0 : 1;
    
    if (exitCode === 0) {
      console.log('\n🎉 Phase 4 testing completed successfully! System ready for production deployment.');
    } else {
      console.log('\n❌ Phase 4 testing failed. Address the issues above before production deployment.');
    }

    console.log('='.repeat(60));
    process.exit(exitCode);

  } catch (error) {
    console.error('\n💥 Phase 4 testing failed with error:', error);
    console.error('\nSystem is not ready for production deployment.');
    process.exit(1);
  }
}

// Handle unhandled rejections
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  process.exit(1);
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  process.exit(1);
});

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(error => {
    console.error('Failed to run Phase 4 tests:', error);
    process.exit(1);
  });
}

export { main as runPhase4Tests, type Phase4TestConfig };