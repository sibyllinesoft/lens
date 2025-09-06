/**
 * CLI Command for Phase C Hardening
 * Provides command-line interface for running hardening tests
 */

import { Command } from 'commander';
import path from 'path';
import { CIHardeningOrchestrator, createDefaultCIConfig, type CIHardeningConfig } from './ci-hardening.js';
import { BenchmarkSuiteRunner } from './suite-runner.js';
import { GroundTruthBuilder } from './ground-truth-builder.js';
import { createDefaultHardeningConfig } from './phase-c-hardening.js';

export function createPhaseCCommand(): Command {
  const cmd = new Command('phase-c');
  
  cmd
    .description('Run Phase C Benchmark Hardening - "Keep the crank honest"')
    .option('-m, --mode <mode>', 'CI mode: pr, nightly, release', 'nightly')
    .option('-o, --output <dir>', 'Output directory for reports and artifacts', './benchmark-results')
    .option('--nats-url <url>', 'NATS server URL for telemetry', 'nats://localhost:4222')
    .option('--ci', 'Run in CI mode with strict gates')
    .option('--fail-fast', 'Exit immediately on first failure')
    .option('--no-plots', 'Disable plot generation')
    .option('--no-hard-negatives', 'Disable hard negative testing')
    .option('--slice-gates', 'Enable per-slice performance gates')
    .option('--min-score <score>', 'Minimum hardening score (0-100)', parseFloat, 75)
    .option('--max-degradation <percent>', 'Maximum acceptable degradation %', parseFloat, 15)
    .option('--timeout <minutes>', 'Maximum execution time in minutes', parseFloat, 30)
    .option('--slack-webhook <url>', 'Slack webhook URL for notifications')
    .option('--verbose', 'Enable verbose logging')
    .action(async (options) => {
      await runPhaseCHardening(options);
    });

  return cmd;
}

async function runPhaseCHardening(options: any) {
  const startTime = Date.now();
  
  try {
    console.log('🔒 Phase C - Benchmark Hardening');
    console.log(`  Mode: ${options.mode}`);
    console.log(`  Output: ${options.output}`);
    console.log(`  CI Mode: ${options.ci ? 'YES' : 'NO'}`);
    console.log('');

    // Initialize components
    const outputDir = path.resolve(options.output);
    const groundTruthBuilder = new GroundTruthBuilder(process.cwd(), outputDir);
    const suiteRunner = new BenchmarkSuiteRunner(groundTruthBuilder, outputDir, options.natsUrl);

    if (options.ci) {
      // CI Mode - Use CIHardeningOrchestrator
      const ciOrchestrator = new CIHardeningOrchestrator(outputDir, options.natsUrl);
      
      const ciConfig = createCIConfig(options);
      const result = await ciOrchestrator.executeInCI(ciConfig);
      
      // CI Exit handling
      const exitCode = result.success ? 0 : 1;
      
      console.log('');
      console.log(`🎯 Phase C CI completed: ${result.success ? 'SUCCESS' : 'FAILED'}`);
      console.log(`  Hardening Score: ${result.hardening_score}/100`);
      console.log(`  Execution Time: ${(result.execution_time_ms / 1000).toFixed(1)}s`);
      
      if (!result.success && result.failure_summary) {
        console.log(`  Key Issues: ${result.failure_summary.recommendations.length}`);
      }
      
      process.exit(exitCode);
      
    } else {
      // Standard Mode - Use BenchmarkSuiteRunner directly
      const hardeningConfig = createHardeningConfig(options);
      const { hardeningReport, pdfReport } = await suiteRunner.runPhaseCHardening(hardeningConfig);
      
      const executionTime = Date.now() - startTime;
      
      console.log('');
      console.log(`🎯 Phase C Hardening completed: ${hardeningReport.hardening_status.toUpperCase()}`);
      console.log(`  Execution Time: ${(executionTime / 1000).toFixed(1)}s`);
      console.log(`  Tripwires: ${hardeningReport.tripwire_summary.passed_tripwires}/${hardeningReport.tripwire_summary.total_tripwires} passed`);
      
      if (hardeningReport.slice_gate_summary) {
        console.log(`  Slice Gates: ${hardeningReport.slice_gate_summary.passed_slices}/${hardeningReport.slice_gate_summary.total_slices} passed`);
      }
      
      if (hardeningReport.hard_negatives.total_generated > 0) {
        console.log(`  Hard Negatives: ${hardeningReport.hard_negatives.total_generated} generated`);
        console.log(`  Ranking Degradation: ${hardeningReport.hard_negatives.impact_on_metrics.degradation_percent.toFixed(2)}%`);
      }
      
      console.log(`  Report: ${pdfReport}`);
      console.log(`  Raw Data: ${path.join(outputDir, 'phase-c-hardening-report.json')}`);
      
      if (hardeningReport.plots_generated) {
        console.log(`  Plots: ${Object.keys(hardeningReport.plots_generated).length} visualization files`);
      }
      
      // Print recommendations if any
      if (hardeningReport.recommendations.length > 0) {
        console.log('');
        console.log('📋 Recommendations:');
        hardeningReport.recommendations.forEach((rec, index) => {
          console.log(`  ${index + 1}. ${rec}`);
        });
      }
      
      // Exit with appropriate code
      const exitCode = hardeningReport.hardening_status === 'pass' ? 0 : 1;
      process.exit(exitCode);
    }
    
  } catch (error) {
    console.error('');
    console.error('💥 Phase C Hardening failed:');
    console.error(error instanceof Error ? error.message : String(error));
    
    if (options.verbose && error instanceof Error && error.stack) {
      console.error('');
      console.error('Stack trace:');
      console.error(error.stack);
    }
    
    process.exit(1);
  }
}

function createCIConfig(options: any): CIHardeningConfig {
  const baseConfig = createDefaultCIConfig(options.mode);
  
  return {
    ...baseConfig,
    fail_fast: options.failFast ?? baseConfig.fail_fast,
    max_execution_time_minutes: options.timeout ?? baseConfig.max_execution_time_minutes,
    slack_webhook_url: options.slackWebhook,
    quality_gates: {
      ...baseConfig.quality_gates,
      min_hardening_score: options.minScore ?? baseConfig.quality_gates.min_hardening_score,
      max_degradation_percent: options.maxDegradation ?? baseConfig.quality_gates.max_degradation_percent,
      enforce_slice_gates: options.sliceGates ?? baseConfig.quality_gates.enforce_slice_gates
    }
  };
}

function createHardeningConfig(options: any) {
  const baseConfig = createDefaultHardeningConfig({
    trace_id: `cli-${Date.now()}`,
    suite: ['codesearch', 'structural'],
    systems: ['lex', '+symbols', '+symbols+semantic'],
    slices: 'SMOKE_DEFAULT',
    seeds: 1,
    cache_mode: 'warm',
    robustness: false,
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
  });

  // Apply CLI overrides
  return {
    ...baseConfig,
    plots: {
      ...baseConfig.plots,
      enabled: !options.noPlots
    },
    hard_negatives: {
      ...baseConfig.hard_negatives,
      enabled: !options.noHardNegatives
    },
    per_slice_gates: {
      ...baseConfig.per_slice_gates,
      enabled: options.sliceGates ?? baseConfig.per_slice_gates.enabled
    }
  };
}

/**
 * Standalone CLI runner for Phase C
 */
export async function runStandalonePhaseCCLI(): Promise<void> {
  const program = new Command();
  
  program
    .name('lens-phase-c')
    .description('Lens Phase C Benchmark Hardening CLI')
    .version('1.0.0');
  
  program.addCommand(createPhaseCCommand());
  
  // Add additional utility commands
  
  const reportCmd = new Command('report');
  reportCmd
    .description('Generate report from existing hardening data')
    .requiredOption('-i, --input <file>', 'Input hardening report JSON file')
    .option('-o, --output <dir>', 'Output directory', './reports')
    .option('-f, --format <format>', 'Output format: pdf, html, markdown', 'markdown')
    .action(async (options) => {
      // Implementation would load existing data and regenerate report
      console.log(`📄 Generating report from ${options.input}...`);
      console.log(`  Format: ${options.format}`);
      console.log(`  Output: ${options.output}`);
      console.log('✅ Report generated (placeholder implementation)');
    });
  
  const validateCmd = new Command('validate');
  validateCmd
    .description('Validate Phase C configuration')
    .option('-c, --config <file>', 'Configuration file to validate')
    .action(async (options) => {
      console.log('🔍 Validating Phase C configuration...');
      console.log(`  Config: ${options.config || 'default'}`);
      console.log('✅ Configuration valid (placeholder implementation)');
    });
  
  const plotsCmd = new Command('plots');
  plotsCmd
    .description('Generate plots from hardening data')
    .requiredOption('-i, --input <file>', 'Input hardening report JSON file')
    .option('-o, --output <dir>', 'Output directory for plots', './plots')
    .option('-f, --format <format>', 'Plot format: png, svg, pdf', 'png')
    .action(async (options) => {
      console.log(`📊 Generating plots from ${options.input}...`);
      console.log(`  Format: ${options.format}`);
      console.log(`  Output: ${options.output}`);
      console.log('✅ Plots generated (placeholder implementation)');
    });
  
  program.addCommand(reportCmd);
  program.addCommand(validateCmd);
  program.addCommand(plotsCmd);
  
  await program.parseAsync(process.argv);
}

// Run standalone if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runStandalonePhaseCCLI().catch(error => {
    console.error('💥 CLI execution failed:', error);
    process.exit(1);
  });
}