#!/usr/bin/env bun
/**
 * Phase C CLI Tool - Comprehensive Benchmarking and Quality Gates
 * Implements the exact API shapes from TODO.md with full statistical rigor
 */

import { Command } from 'commander';
import { BenchmarkSuiteRunner } from '../../benchmarks/src/suite-runner.js';
import { GroundTruthBuilder } from '../../benchmarks/src/ground-truth-builder.js';
import { promises as fs } from 'fs';
import path from 'path';
import chalk from 'chalk';
import ora from 'ora';
import type { BenchmarkConfig } from '../types/benchmark.js';

const program = new Command();

program
  .name('phase-c')
  .description('Phase C - Benchmark & Quality Gates CLI')
  .version('1.0.0');

// Global options
program
  .option('-o, --output <dir>', 'Output directory for results', '../../benchmarks/src-results')
  .option('--nats-url <url>', 'NATS server URL for telemetry', 'nats://localhost:4222')
  .option('--verbose', 'Enable verbose logging')
  .option('--trace-id <id>', 'Custom trace ID for the run');

/**
 * Smoke benchmark command - implements exact TODO.md API shape
 */
program
  .command('smoke')
  .description('Run smoke benchmark suite (PR gate) - ~10 minutes')
  .option('--systems <systems...>', 'Systems to test', ['lex', '+symbols', '+symbols+semantic'])
  .option('--cache-mode <mode>', 'Cache mode', 'warm')
  .option('--seeds <count>', 'Number of seeds', '1')
  .option('--k-candidates <count>', 'Number of candidates', '200')
  .option('--top-n <count>', 'Top N results', '50')
  .option('--fuzzy <distance>', 'Fuzzy search distance', '2')
  .option('--hard-negatives', 'Enable hard negative testing')
  .option('--skip-hardening', 'Skip Phase C hardening analysis')
  .action(async (options) => {
    const spinner = ora('Initializing smoke benchmark suite...').start();
    
    try {
      const { runner, outputDir } = await initializeBenchmarkRunner(program.opts());
      
      const config: BenchmarkConfig = {
        trace_id: program.opts().traceId || `smoke-${Date.now()}`,
        suite: ['codesearch', 'structural'],
        systems: options.systems,
        slices: 'SMOKE_DEFAULT',
        seeds: parseInt(options.seeds),
        cache_mode: options.cacheMode as 'warm' | 'cold',
        robustness: false,
        metamorphic: false,
        k_candidates: parseInt(options.kCandidates),
        top_n: parseInt(options.topN),
        fuzzy: parseInt(options.fuzzy),
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

      spinner.text = 'Running smoke benchmark...';
      const result = await runner.runSmokeSuite(config);
      
      if (!options.skipHardening) {
        spinner.text = 'Running Phase C hardening analysis...';
        const { hardeningReport, pdfReport } = await runner.runPhaseCHardening({
          ...config,
          hard_negatives: {
            enabled: options.hardNegatives || false,
            per_query_count: 5,
            shared_subtoken_min: 2
          },
          tripwires: {
            min_span_coverage: 0.98,
            recall_convergence_threshold: 0.005,
            lsif_coverage_drop_threshold: 0.05,
            p99_p95_ratio_threshold: 2.0
          },
          per_slice_gates: {
            enabled: true,
            min_recall_at_10: 0.7,
            min_ndcg_at_10: 0.6,
            max_p95_latency_ms: 500
          },
          plots: {
            enabled: true,
            output_dir: path.join(outputDir, 'plots'),
            formats: ['png', 'svg']
          }
        }, [result]);
        
        (result as any).hardening_report = hardeningReport;
        (result as any).pdf_report_path = pdfReport;
      }

      spinner.succeed('Smoke benchmark completed');
      
      // Display results
      displayBenchmarkResults(result, options.skipHardening);
      
      // Save results
      await saveResults(outputDir, 'smoke', result, config);
      
    } catch (error) {
      spinner.fail(`Smoke benchmark failed: ${error instanceof Error ? error.message : error}`);
      process.exit(1);
    }
  });

/**
 * Full benchmark command - nightly comprehensive suite
 */
program
  .command('full')
  .description('Run full benchmark suite (nightly) - multi-repo + metamorphic + robustness')
  .option('--systems <systems...>', 'Systems to test', ['lex', '+symbols', '+symbols+semantic'])
  .option('--cache-mode <modes...>', 'Cache modes', ['warm', 'cold'])
  .option('--seeds <count>', 'Number of seeds', '3')
  .option('--robustness', 'Enable robustness testing')
  .option('--metamorphic', 'Enable metamorphic testing')
  .option('--slices <slices>', 'Test slices', 'ALL')
  .action(async (options) => {
    const spinner = ora('Initializing full benchmark suite...').start();
    
    try {
      const { runner, outputDir } = await initializeBenchmarkRunner(program.opts());
      
      const config: BenchmarkConfig = {
        trace_id: program.opts().traceId || `full-${Date.now()}`,
        suite: ['codesearch', 'structural', 'docs'],
        systems: options.systems,
        slices: options.slices,
        seeds: parseInt(options.seeds),
        cache_mode: Array.isArray(options.cacheMode) ? options.cacheMode : [options.cacheMode],
        robustness: options.robustness,
        metamorphic: options.metamorphic,
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

      spinner.text = 'Running full benchmark suite...';
      const result = await runner.runFullSuiteWithHardening(config);
      
      spinner.succeed('Full benchmark completed');
      
      // Display results
      displayBenchmarkResults(result, false);
      
      // Save results
      await saveResults(outputDir, 'full', result, config);
      
    } catch (error) {
      spinner.fail(`Full benchmark failed: ${error instanceof Error ? error.message : error}`);
      process.exit(1);
    }
  });

/**
 * Hard negatives testing command
 */
program
  .command('hard-negatives')
  .description('Run hard negative injection testing - stress test ranking robustness')
  .option('--per-query <count>', 'Hard negatives per query', '5')
  .option('--shared-subtokens <count>', 'Minimum shared subtokens', '2')
  .option('--strategies <strategies...>', 'Generation strategies', 
    ['shared_class', 'shared_method', 'shared_variable', 'shared_imports'])
  .action(async (options) => {
    const spinner = ora('Running hard negative testing...').start();
    
    try {
      const { runner, outputDir } = await initializeBenchmarkRunner(program.opts());
      
      const { hardeningReport } = await runner.runPhaseCHardening({
        hard_negatives: {
          enabled: true,
          per_query_count: parseInt(options.perQuery),
          shared_subtoken_min: parseInt(options.sharedSubtokens)
        },
        tripwires: {
          min_span_coverage: 0.95,
          recall_convergence_threshold: 0.01,
          lsif_coverage_drop_threshold: 0.1,
          p99_p95_ratio_threshold: 3.0
        },
        per_slice_gates: { enabled: false, min_recall_at_10: 0.8, min_ndcg_at_10: 0.75, max_p95_latency_ms: 1000 },
        plots: { enabled: false, output_dir: './plots', formats: ['png'] }
      });
      
      spinner.succeed('Hard negative testing completed');
      
      // Display hard negative results
      console.log(chalk.bold('\n🎯 Hard Negative Testing Results'));
      console.log(chalk.blue('─'.repeat(50)));
      console.log(`Generated: ${hardeningReport.hard_negatives.total_generated} hard negatives`);
      console.log(`Baseline Recall@10: ${hardeningReport.hard_negatives.impact_on_metrics.baseline_recall_at_10.toFixed(3)}`);
      console.log(`With Negatives: ${hardeningReport.hard_negatives.impact_on_metrics.with_negatives_recall_at_10.toFixed(3)}`);
      console.log(`Degradation: ${hardeningReport.hard_negatives.impact_on_metrics.degradation_percent.toFixed(2)}%`);
      
      const robustnessLevel = hardeningReport.hard_negatives.impact_on_metrics.degradation_percent < 10 ? 
        chalk.green('ROBUST') : chalk.yellow('SENSITIVE');
      console.log(`Assessment: ${robustnessLevel}`);
      
      // Save results
      const reportPath = path.join(outputDir, `hard-negatives-report-${Date.now()}.json`);
      await fs.writeFile(reportPath, JSON.stringify(hardeningReport, null, 2));
      console.log(`\n📄 Report saved: ${reportPath}`);
      
    } catch (error) {
      spinner.fail(`Hard negative testing failed: ${error instanceof Error ? error.message : error}`);
      process.exit(1);
    }
  });

/**
 * Tripwire validation command
 */
program
  .command('tripwires')
  .description('Run tripwire validation - hard failure quality gates')
  .option('--span-coverage <threshold>', 'Minimum span coverage', '0.98')
  .option('--recall-convergence <threshold>', 'Recall convergence threshold', '0.005')
  .option('--lsif-drop <threshold>', 'LSIF coverage drop threshold', '0.05')
  .option('--p99-p95-ratio <threshold>', 'P99/P95 ratio threshold', '2.0')
  .action(async (options) => {
    const spinner = ora('Running tripwire validation...').start();
    
    try {
      const { runner, outputDir } = await initializeBenchmarkRunner(program.opts());
      
      // Run base benchmark first
      const baseResult = await runner.runSmokeSuite({
        trace_id: `tripwire-base-${Date.now()}`,
        k_candidates: 100
      });
      
      const { hardeningReport } = await runner.runPhaseCHardening({
        tripwires: {
          min_span_coverage: parseFloat(options.spanCoverage),
          recall_convergence_threshold: parseFloat(options.recallConvergence),
          lsif_coverage_drop_threshold: parseFloat(options.lsifDrop),
          p99_p95_ratio_threshold: parseFloat(options.p99P95Ratio)
        },
        hard_negatives: { enabled: false, per_query_count: 5, shared_subtoken_min: 2 },
        per_slice_gates: { enabled: false, min_recall_at_10: 0.8, min_ndcg_at_10: 0.75, max_p95_latency_ms: 1000 },
        plots: { enabled: false, output_dir: './plots', formats: ['png'] }
      }, [baseResult]);
      
      spinner.succeed('Tripwire validation completed');
      
      // Display tripwire results
      console.log(chalk.bold('\n⚡ Tripwire Validation Results'));
      console.log(chalk.blue('─'.repeat(60)));
      
      for (const tripwire of hardeningReport.tripwire_results) {
        const status = tripwire.status === 'pass' ? 
          chalk.green('✅ PASS') : chalk.red('❌ FAIL');
        const value = formatTripwireValue(tripwire.name, tripwire.actual_value);
        const threshold = formatTripwireValue(tripwire.name, tripwire.threshold);
        
        console.log(`${status} ${tripwire.name}: ${value} (threshold: ${threshold})`);
      }
      
      const overallStatus = hardeningReport.tripwire_summary.overall_status === 'pass' ?
        chalk.green('PASS') : chalk.red('FAIL');
      console.log(`\nOverall Status: ${overallStatus}`);
      console.log(`Passed: ${hardeningReport.tripwire_summary.passed_tripwires}/${hardeningReport.tripwire_summary.total_tripwires}`);
      
      // Save results
      const reportPath = path.join(outputDir, `tripwires-report-${Date.now()}.json`);
      await fs.writeFile(reportPath, JSON.stringify(hardeningReport, null, 2));
      console.log(`\n📄 Report saved: ${reportPath}`);
      
    } catch (error) {
      spinner.fail(`Tripwire validation failed: ${error instanceof Error ? error.message : error}`);
      process.exit(1);
    }
  });

/**
 * CI gates command - strict validation for CI/CD
 */
program
  .command('ci-gates')
  .description('Run CI hardening gates - strict validation for CI/CD pipeline')
  .action(async (options) => {
    const spinner = ora('Running CI hardening gates...').start();
    
    try {
      const { runner, outputDir } = await initializeBenchmarkRunner(program.opts());
      
      const passed = await runner.runPhaseCCIGates({
        tripwires: {
          min_span_coverage: 0.99, // 99% for CI
          recall_convergence_threshold: 0.003, // 0.3% for CI
          lsif_coverage_drop_threshold: 0.03, // 3% for CI
          p99_p95_ratio_threshold: 1.8, // 1.8× for CI
        },
        per_slice_gates: {
          enabled: true,
          min_recall_at_10: 0.75, // Higher bar for CI
          min_ndcg_at_10: 0.65, // Higher bar for CI
          max_p95_latency_ms: 450, // Tighter latency for CI
        }
      });
      
      if (passed) {
        spinner.succeed('CI gates PASSED');
        console.log(chalk.green('\n✅ All CI hardening gates passed - ready for deployment'));
      } else {
        spinner.fail('CI gates FAILED');
        console.log(chalk.red('\n❌ CI hardening gates failed - deployment blocked'));
        console.log(chalk.yellow('Check logs for specific failure details'));
        process.exit(1);
      }
      
    } catch (error) {
      spinner.fail(`CI gates execution failed: ${error instanceof Error ? error.message : error}`);
      process.exit(1);
    }
  });

/**
 * Plots generation command
 */
program
  .command('plots')
  .description('Generate Phase C visualization plots')
  .option('--formats <formats...>', 'Output formats', ['png', 'svg'])
  .option('--plot-dir <dir>', 'Plots output directory', './plots')
  .action(async (options) => {
    const spinner = ora('Generating Phase C visualization plots...').start();
    
    try {
      const { runner, outputDir } = await initializeBenchmarkRunner(program.opts());
      
      const { hardeningReport } = await runner.runPhaseCHardening({
        plots: {
          enabled: true,
          output_dir: path.resolve(options.plotDir),
          formats: options.formats
        },
        hard_negatives: { enabled: false, per_query_count: 5, shared_subtoken_min: 2 },
        tripwires: {
          min_span_coverage: 0.95,
          recall_convergence_threshold: 0.01,
          lsif_coverage_drop_threshold: 0.1,
          p99_p95_ratio_threshold: 3.0
        },
        per_slice_gates: { enabled: false, min_recall_at_10: 0.8, min_ndcg_at_10: 0.75, max_p95_latency_ms: 1000 }
      });
      
      spinner.succeed('Plots generated successfully');
      
      console.log(chalk.bold('\n📊 Generated Visualization Plots'));
      console.log(chalk.blue('─'.repeat(50)));
      
      for (const [plotName, plotPath] of Object.entries(hardeningReport.plots_generated)) {
        const formattedName = plotName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        console.log(`${formattedName}: ${plotPath}`);
      }
      
      console.log(`\n📁 Output directory: ${path.resolve(options.plotDir)}`);
      
    } catch (error) {
      spinner.fail(`Plot generation failed: ${error instanceof Error ? error.message : error}`);
      process.exit(1);
    }
  });

/**
 * Status command - check benchmark system status
 */
program
  .command('status')
  .description('Check benchmark system status and recent runs')
  .action(async (options) => {
    const spinner = ora('Checking benchmark system status...').start();
    
    try {
      const { runner, outputDir } = await initializeBenchmarkRunner(program.opts());
      
      // Check output directory for recent runs
      const files = await fs.readdir(outputDir).catch(() => []);
      const recentFiles = files.filter(f => f.includes('benchmark-') || f.includes('report-'))
        .sort().reverse().slice(0, 5);
      
      spinner.succeed('Benchmark system status retrieved');
      
      console.log(chalk.bold('\n📊 Benchmark System Status'));
      console.log(chalk.blue('─'.repeat(50)));
      console.log(`Output Directory: ${outputDir}`);
      console.log(`Recent Files: ${files.length} total`);
      
      if (recentFiles.length > 0) {
        console.log(chalk.bold('\nRecent Benchmark Runs:'));
        for (const file of recentFiles) {
          console.log(`  ${file}`);
        }
      } else {
        console.log('\nNo recent benchmark runs found');
      }
      
      // System health checks
      console.log(chalk.bold('\nSystem Health:'));
      console.log('✅ Benchmark runner initialized');
      console.log('✅ Ground truth builder ready');
      console.log('✅ Output directory accessible');
      
    } catch (error) {
      spinner.fail(`Status check failed: ${error instanceof Error ? error.message : error}`);
      process.exit(1);
    }
  });

/**
 * Initialize benchmark runner with common configuration
 */
async function initializeBenchmarkRunner(globalOpts: any) {
  const outputDir = path.resolve(globalOpts.output);
  await fs.mkdir(outputDir, { recursive: true });
  
  const groundTruthBuilder = new GroundTruthBuilder('./', outputDir);
  
  const runner = new BenchmarkSuiteRunner(
    groundTruthBuilder,
    outputDir,
    globalOpts.natsUrl
  );
  
  if (globalOpts.verbose) {
    console.log(`Output directory: ${outputDir}`);
    console.log(`NATS URL: ${globalOpts.natsUrl}`);
  }
  
  return { runner, outputDir };
}

/**
 * Display benchmark results in a formatted way
 */
function displayBenchmarkResults(result: any, skipHardening: boolean) {
  console.log(chalk.bold('\n🏆 Benchmark Results'));
  console.log(chalk.blue('═'.repeat(60)));
  
  // Basic metrics
  console.log(chalk.bold('Quality Metrics:'));
  console.log(`  Recall@10: ${result.metrics.recall_at_10.toFixed(3)}`);
  console.log(`  Recall@50: ${result.metrics.recall_at_50.toFixed(3)}`);
  console.log(`  nDCG@10:   ${result.metrics.ndcg_at_10.toFixed(3)}`);
  console.log(`  MRR:       ${result.metrics.mrr.toFixed(3)}`);
  
  // Latency metrics
  console.log(chalk.bold('\nLatency Metrics:'));
  console.log(`  Stage A P95: ${result.metrics.stage_latencies.stage_a_p95.toFixed(1)}ms`);
  console.log(`  Stage B P95: ${result.metrics.stage_latencies.stage_b_p95.toFixed(1)}ms`);
  if (result.metrics.stage_latencies.stage_c_p95) {
    console.log(`  Stage C P95: ${result.metrics.stage_latencies.stage_c_p95.toFixed(1)}ms`);
  }
  console.log(`  E2E P95:     ${result.metrics.stage_latencies.e2e_p95.toFixed(1)}ms`);
  
  // Execution stats
  console.log(chalk.bold('\nExecution Stats:'));
  console.log(`  Status: ${result.status === 'completed' ? chalk.green('COMPLETED') : chalk.red('FAILED')}`);
  console.log(`  Queries: ${result.completed_queries}/${result.total_queries}`);
  console.log(`  Failures: ${result.failed_queries}`);
  
  // Hardening results if available
  if (!skipHardening && result.hardening_report) {
    console.log(chalk.bold('\n🔒 Hardening Analysis:'));
    const hardeningStatus = result.hardening_report.hardening_status === 'pass' ?
      chalk.green('PASS') : chalk.red('FAIL');
    console.log(`  Overall Status: ${hardeningStatus}`);
    console.log(`  Tripwires: ${result.hardening_report.tripwire_summary.passed_tripwires}/${result.hardening_report.tripwire_summary.total_tripwires} passed`);
    
    if (result.hardening_report.slice_gate_summary) {
      console.log(`  Slices: ${result.hardening_report.slice_gate_summary.passed_slices}/${result.hardening_report.slice_gate_summary.total_slices} passed`);
    }
    
    if (result.hardening_report.hard_negatives.total_generated > 0) {
      console.log(`  Hard Negatives: ${result.hardening_report.hard_negatives.total_generated} generated`);
      console.log(`  Impact: ${result.hardening_report.hard_negatives.impact_on_metrics.degradation_percent.toFixed(2)}% degradation`);
    }
  }
  
  console.log(chalk.blue('═'.repeat(60)));
}

/**
 * Save benchmark results and artifacts
 */
async function saveResults(outputDir: string, benchmarkType: string, result: any, config: BenchmarkConfig) {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const baseFilename = `${benchmarkType}-${timestamp}`;
  
  // Save main result
  const resultPath = path.join(outputDir, `${baseFilename}-result.json`);
  await fs.writeFile(resultPath, JSON.stringify(result, null, 2));
  
  // Save config
  const configPath = path.join(outputDir, `${baseFilename}-config.json`);
  await fs.writeFile(configPath, JSON.stringify(config, null, 2));
  
  console.log(`\n📄 Results saved:`);
  console.log(`  ${resultPath}`);
  console.log(`  ${configPath}`);
  
  if (result.hardening_report) {
    const hardeningPath = path.join(outputDir, `${baseFilename}-hardening.json`);
    await fs.writeFile(hardeningPath, JSON.stringify(result.hardening_report, null, 2));
    console.log(`  ${hardeningPath}`);
  }
}

/**
 * Format tripwire values for display
 */
function formatTripwireValue(tripwireName: string, value: number): string {
  switch (tripwireName) {
    case 'span_coverage':
    case 'lsif_coverage_drop':
    case 'recall_convergence':
      return `${(value * 100).toFixed(2)}%`;
    case 'p99_p95_ratio':
      return `${value.toFixed(2)}×`;
    default:
      return value.toFixed(3);
  }
}

// Parse command line arguments
program.parse();

export { program };