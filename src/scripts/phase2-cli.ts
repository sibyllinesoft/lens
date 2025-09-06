#!/usr/bin/env bun
/**
 * Phase 2 Recall Pack CLI
 * Command-line interface for executing Phase 2 workflow
 */

import { parseArgs } from 'util';
import { Phase2RecallPack, type Phase2Results } from '../core/phase2-recall-pack.js';
import { promises as fs } from 'fs';
import path from 'path';

interface CLIOptions {
  indexRoot: string;
  outputDir: string;
  apiUrl: string;
  help: boolean;
  synonymsOnly: boolean;
  pathPriorOnly: boolean;
  benchmarkOnly: boolean;
  verbose: boolean;
  dryRun: boolean;
}

const DEFAULT_OPTIONS: Partial<CLIOptions> = {
  indexRoot: './indexed-content',
  outputDir: './phase2-results',
  apiUrl: 'http://localhost:3001',
  help: false,
  synonymsOnly: false,
  pathPriorOnly: false,
  benchmarkOnly: false,
  verbose: false,
  dryRun: false,
};

/**
 * Display help message
 */
function showHelp() {
  console.log(`
🎯 Phase 2 Recall Pack CLI - Lens Search Enhancement

USAGE:
  bun run src/scripts/phase2-cli.ts [OPTIONS]

OPTIONS:
  --index-root <path>      Path to indexed content (default: ./indexed-content)
  --output-dir <path>      Output directory for results (default: ./phase2-results)
  --api-url <url>          API base URL (default: http://localhost:3001)
  --synonyms-only          Only run synonym mining
  --pathprior-only         Only run path prior refitting
  --benchmark-only         Only run benchmarks (requires existing synonyms/path prior)
  --dry-run                Show what would be executed without running
  --verbose                Enable verbose logging
  --help                   Show this help message

EXAMPLES:
  # Run complete Phase 2 workflow
  bun run src/scripts/phase2-cli.ts

  # Run only synonym mining
  bun run src/scripts/phase2-cli.ts --synonyms-only

  # Run with custom paths
  bun run src/scripts/phase2-cli.ts --index-root ./my-index --output-dir ./my-results

  # Dry run to see what would be executed
  bun run src/scripts/phase2-cli.ts --dry-run

PHASE 2 WORKFLOW:
  1. Capture baseline metrics (Recall@50, nDCG@10)
  2. Mine PMI-based synonyms (τ_pmi=3.0, min_freq≥20, K=8)
  3. Refit path priors with gentler de-boosts (max_deboost=0.6)
  4. Apply policy deltas to stageA configuration
  5. Run smoke benchmark (warm, single seed)
  6. Run full benchmark (cold+warm, 3 seeds)
  7. Check acceptance gates and tripwires
  8. Prepare promotion or rollback

ACCEPTANCE GATES:
  • Recall@50 ≥ +5% improvement (target: ≥0.899)
  • nDCG@10 ≥ 0 change (target: ≥0.743)
  • Span coverage ≥98%
  • E2E p95 ≤ +25% increase (target: ≤97.5ms)

TRIPWIRES:
  • Recall@50 ≈ Recall@10 gap check
  • LSIF coverage ≥85%
  • Sentinel query regression check
`);
}

/**
 * Parse command line arguments
 */
function parseArguments(): CLIOptions {
  try {
    const { values } = parseArgs({
      args: process.argv.slice(2),
      options: {
        'index-root': { type: 'string' },
        'output-dir': { type: 'string' },
        'api-url': { type: 'string' },
        'synonyms-only': { type: 'boolean' },
        'pathprior-only': { type: 'boolean' },
        'benchmark-only': { type: 'boolean' },
        'dry-run': { type: 'boolean' },
        'verbose': { type: 'boolean' },
        'help': { type: 'boolean' },
      },
      allowPositionals: false,
    });

    return {
      indexRoot: values['index-root'] || DEFAULT_OPTIONS.indexRoot!,
      outputDir: values['output-dir'] || DEFAULT_OPTIONS.outputDir!,
      apiUrl: values['api-url'] || DEFAULT_OPTIONS.apiUrl!,
      synonymsOnly: values['synonyms-only'] || false,
      pathPriorOnly: values['pathprior-only'] || false,
      benchmarkOnly: values['benchmark-only'] || false,
      dryRun: values['dry-run'] || false,
      verbose: values['verbose'] || false,
      help: values['help'] || false,
    };
  } catch (error) {
    console.error('❌ Error parsing arguments:', error);
    process.exit(1);
  }
}

/**
 * Execute synonym mining only
 */
async function executeSynonymsOnly(options: CLIOptions): Promise<void> {
  console.log('🔍 Executing synonym mining only...');
  
  if (options.dryRun) {
    console.log(`[DRY RUN] Would mine synonyms with:`);
    console.log(`  - Index root: ${options.indexRoot}`);
    console.log(`  - Output dir: ${options.outputDir}/synonyms`);
    console.log(`  - Parameters: τ_pmi=3.0, min_freq=20, K=8`);
    return;
  }

  const { Phase2SynonymMiner } = await import('../core/phase2-synonym-miner.js');
  const synonymMiner = new Phase2SynonymMiner(
    options.indexRoot,
    path.join(options.outputDir, 'synonyms')
  );

  const synonymTable = await synonymMiner.mineSynonyms({
    tau_pmi: 3.0,
    min_freq: 20,
    k_synonyms: 8,
  });

  console.log(`✅ Synonym mining completed: ${synonymTable.entries.length} entries generated`);
  console.log(`💾 Results saved to: ${path.join(options.outputDir, 'synonyms')}`);
}

/**
 * Execute path prior refitting only
 */
async function executePathPriorOnly(options: CLIOptions): Promise<void> {
  console.log('🔍 Executing path prior refitting only...');
  
  if (options.dryRun) {
    console.log(`[DRY RUN] Would refit path priors with:`);
    console.log(`  - Index root: ${options.indexRoot}`);
    console.log(`  - Output dir: ${options.outputDir}/path-priors`);
    console.log(`  - Parameters: L2=1.0, debias=true, max_deboost=0.6`);
    return;
  }

  const { Phase2PathPrior } = await import('../core/phase2-path-prior.js');
  const pathPrior = new Phase2PathPrior(
    options.indexRoot,
    path.join(options.outputDir, 'path-priors')
  );

  const model = await pathPrior.refitPathPrior({
    l2_regularization: 1.0,
    debias_low_priority_paths: true,
    max_deboost: 0.6,
  });

  console.log(`✅ Path prior refitting completed: AUC-ROC ${model.performance.auc_roc.toFixed(3)}`);
  console.log(`💾 Model saved to: ${path.join(options.outputDir, 'path-priors')}`);
}

/**
 * Execute benchmarks only
 */
async function executeBenchmarkOnly(options: CLIOptions): Promise<void> {
  console.log('🔍 Executing benchmarks only...');
  
  if (options.dryRun) {
    console.log(`[DRY RUN] Would run benchmarks:`);
    console.log(`  - API URL: ${options.apiUrl}`);
    console.log(`  - Smoke test: single seed, warm only`);
    console.log(`  - Full test: 3 seeds, cold+warm`);
    return;
  }

  // This would require the Phase2RecallPack orchestrator
  // but only run the benchmarking portions
  console.log('⚠️  Benchmark-only mode not yet fully implemented');
  console.log('💡 Use full Phase 2 execution for complete benchmarking workflow');
}

/**
 * Execute complete Phase 2 workflow
 */
async function executeComplete(options: CLIOptions): Promise<Phase2Results> {
  console.log('🎯 Executing complete Phase 2 Recall Pack workflow...');
  
  if (options.dryRun) {
    console.log(`[DRY RUN] Would execute complete Phase 2 workflow:`);
    console.log(`  - Index root: ${options.indexRoot}`);
    console.log(`  - Output dir: ${options.outputDir}`);
    console.log(`  - API URL: ${options.apiUrl}`);
    console.log(`  - 1. Capture baseline metrics`);
    console.log(`  - 2. Mine PMI-based synonyms`);
    console.log(`  - 3. Refit path priors with gentler de-boosts`);
    console.log(`  - 4. Apply policy deltas`);
    console.log(`  - 5. Run smoke benchmark`);
    console.log(`  - 6. Run full benchmark`);
    console.log(`  - 7. Check acceptance gates`);
    console.log(`  - 8. Check tripwires`);
    console.log(`  - 9. Prepare promotion or rollback`);
    
    // Return mock results for dry run
    return {
      baseline_recall_50: 0.856,
      baseline_ndcg_10: 0.743,
      new_recall_50: 0.899,
      new_ndcg_10: 0.748,
      recall_improvement_pct: 5.0,
      ndcg_change: 0.005,
      span_coverage_pct: 98.2,
      e2e_p95_ms: 85.0,
      acceptance_gates_passed: true,
      tripwires_status: 'green',
      promotion_ready: true,
    };
  }

  const phase2 = new Phase2RecallPack(
    options.indexRoot,
    options.outputDir,
    options.apiUrl
  );

  return await phase2.executePhase2();
}

/**
 * Display results summary
 */
function displayResults(results: Phase2Results, options: CLIOptions): void {
  console.log('\n📊 Phase 2 Recall Pack Results Summary');
  console.log('=' .repeat(50));
  
  console.log('\n🎯 Key Metrics:');
  console.log(`  Recall@50: ${results.baseline_recall_50.toFixed(3)} → ${results.new_recall_50.toFixed(3)} (${results.recall_improvement_pct >= 0 ? '+' : ''}${results.recall_improvement_pct.toFixed(1)}%)`);
  console.log(`  nDCG@10:   ${results.baseline_ndcg_10.toFixed(3)} → ${results.new_ndcg_10.toFixed(3)} (${results.ndcg_change >= 0 ? '+' : ''}${results.ndcg_change.toFixed(3)})`);
  console.log(`  Span Coverage: ${results.span_coverage_pct.toFixed(1)}%`);
  console.log(`  E2E p95 Latency: ${results.e2e_p95_ms.toFixed(1)}ms`);
  
  console.log('\n🚦 Status:');
  console.log(`  Acceptance Gates: ${results.acceptance_gates_passed ? '✅ PASSED' : '❌ FAILED'}`);
  console.log(`  Tripwires: ${results.tripwires_status === 'green' ? '🟢 GREEN' : results.tripwires_status === 'yellow' ? '🟡 YELLOW' : '🔴 RED'}`);
  console.log(`  Promotion Ready: ${results.promotion_ready ? '✅ YES' : '❌ NO'}`);
  
  if (results.promotion_ready) {
    console.log('\n🚀 Next Steps:');
    console.log('  1. Review detailed results in output directory');
    console.log('  2. Execute promotion: policy_version++ and tag v1.1-recall-pack');
    console.log('  3. Deploy to production');
    console.log(`  📁 Results location: ${options.outputDir}`);
  } else {
    console.log('\n⏪ Rollback Information:');
    console.log('  Phase 2 failed acceptance criteria');
    console.log('  Policy has been automatically reverted to baseline');
    console.log('  Review logs and metrics for improvement opportunities');
    console.log(`  📁 Rollback details: ${options.outputDir}/rollback-info.json`);
  }
  
  console.log(`\n💾 Full results available at: ${options.outputDir}`);
}

/**
 * Validate environment and prerequisites
 */
async function validateEnvironment(options: CLIOptions): Promise<boolean> {
  const checks = [
    {
      name: 'Index root exists',
      check: async () => {
        try {
          const stat = await fs.stat(options.indexRoot);
          return stat.isDirectory();
        } catch {
          return false;
        }
      }
    },
    {
      name: 'API server accessibility',
      check: async () => {
        try {
          const response = await fetch(`${options.apiUrl}/health`, { 
            method: 'GET',
            signal: AbortSignal.timeout(5000) 
          });
          return response.ok;
        } catch {
          return false;
        }
      }
    },
    {
      name: 'Output directory writable',
      check: async () => {
        try {
          await fs.mkdir(options.outputDir, { recursive: true });
          return true;
        } catch {
          return false;
        }
      }
    }
  ];

  console.log('🔍 Validating environment...');
  
  let allPassed = true;
  for (const check of checks) {
    const passed = await check.check();
    const status = passed ? '✅' : '❌';
    console.log(`  ${status} ${check.name}`);
    
    if (!passed) {
      allPassed = false;
    }
  }

  return allPassed;
}

/**
 * Main CLI function
 */
async function main(): Promise<void> {
  console.log('🎯 Phase 2 Recall Pack CLI');
  console.log('Target: +5-10% Recall@50 with spans intact\n');

  const options = parseArguments();

  if (options.help) {
    showHelp();
    return;
  }

  if (options.verbose) {
    console.log('🔧 Configuration:');
    console.log(`  Index root: ${options.indexRoot}`);
    console.log(`  Output dir: ${options.outputDir}`);
    console.log(`  API URL: ${options.apiUrl}`);
    console.log(`  Dry run: ${options.dryRun}`);
    console.log('');
  }

  // Validate environment unless dry run
  if (!options.dryRun) {
    const valid = await validateEnvironment(options);
    if (!valid) {
      console.error('❌ Environment validation failed');
      process.exit(1);
    }
    console.log('✅ Environment validation passed\n');
  }

  try {
    const startTime = Date.now();
    let results: Phase2Results | null = null;

    // Route to appropriate execution mode
    if (options.synonymsOnly) {
      await executeSynonymsOnly(options);
    } else if (options.pathPriorOnly) {
      await executePathPriorOnly(options);
    } else if (options.benchmarkOnly) {
      await executeBenchmarkOnly(options);
    } else {
      results = await executeComplete(options);
    }

    const duration = Date.now() - startTime;
    console.log(`\n⏱️  Total execution time: ${duration}ms`);

    // Display results summary if available
    if (results) {
      displayResults(results, options);
      
      // Exit with appropriate code
      process.exit(results.promotion_ready ? 0 : 1);
    }

  } catch (error) {
    console.error('\n💥 Phase 2 execution failed:', error);
    console.error('\n🔧 Troubleshooting:');
    console.error('  1. Ensure the Lens server is running');
    console.error('  2. Verify index content exists');
    console.error('  3. Check API connectivity');
    console.error('  4. Review logs for detailed error information');
    process.exit(1);
  }
}

// Execute CLI if run directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error('💥 Fatal error:', error);
    process.exit(1);
  });
}

export { main as runPhase2CLI };