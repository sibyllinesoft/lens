#!/usr/bin/env bun
/**
 * Phase B Validation CLI Script
 * Provides command-line interface for running Phase B performance validation
 * Usage: bun scripts/run-phase-b-validation.ts [options]
 */

import { parseArgs } from 'util';
import { promises as fs } from 'fs';
import path from 'path';
import {
  PhaseBOrchestrator,
  type ValidationConfig,
  OptimizationStage,
  StressTestingBenchmark,
  RegressionPreventionSystem
} from '../src/benchmark/index.js';

// CLI argument parsing
const { values: args, positionals } = parseArgs({
  args: Bun.argv.slice(2),
  options: {
    mode: {
      type: 'string',
      short: 'm',
      default: 'comprehensive'
    },
    commit: {
      type: 'string',
      short: 'c'
    },
    output: {
      type: 'string',
      short: 'o',
      default: './benchmark-results'
    },
    history: {
      type: 'string',
      default: './benchmark-history'
    },
    'skip-stress': {
      type: 'boolean',
      default: false
    },
    'skip-regression': {
      type: 'boolean', 
      default: false
    },
    'performance-only': {
      type: 'boolean',
      default: false
    },
    'ci-mode': {
      type: 'boolean',
      default: false
    },
    'stage-a-target': {
      type: 'string',
      default: '5'
    },
    'e2e-target': {
      type: 'string',
      default: '10'
    },
    help: {
      type: 'boolean',
      short: 'h',
      default: false
    }
  },
  allowPositionals: true
});

// Show help
if (args.help) {
  console.log(`
Phase B Performance Validation Tool

Usage: bun scripts/run-phase-b-validation.ts [options]

Modes:
  comprehensive    Run full validation suite (default)
  ci              Fast CI validation mode
  performance     Performance benchmarks only
  stress          Stress testing only  
  regression      Regression detection only

Options:
  -m, --mode <mode>           Validation mode (comprehensive|ci|performance|stress|regression)
  -c, --commit <sha>          Git commit SHA for regression analysis
  -o, --output <dir>          Output directory for reports (default: ./benchmark-results)
  --history <dir>             History directory for regression data (default: ./benchmark-history)
  --skip-stress               Skip stress testing
  --skip-regression           Skip regression detection
  --performance-only          Run only performance benchmarks
  --ci-mode                   Enable CI-optimized mode (faster, less comprehensive)
  --stage-a-target <ms>       Stage-A p95 target in ms (default: 5)
  --e2e-target <pct>          E2E p95 increase limit % (default: 10)
  -h, --help                  Show this help message

Examples:
  # Run comprehensive validation
  bun scripts/run-phase-b-validation.ts

  # CI validation for specific commit
  bun scripts/run-phase-b-validation.ts --ci-mode --commit abc1234

  # Performance benchmarks only
  bun scripts/run-phase-b-validation.ts --mode performance

  # Stress testing with custom targets
  bun scripts/run-phase-b-validation.ts --mode stress --stage-a-target 3 --e2e-target 5

  # Regression detection
  bun scripts/run-phase-b-validation.ts --mode regression --commit def5678
`);
  process.exit(0);
}

async function main() {
  const mode = args.mode as string;
  const outputDir = path.resolve(args.output as string);
  const historyDir = path.resolve(args.history as string);
  const commitSha = args.commit as string | undefined;

  console.log('🚀 Phase B Performance Validation');
  console.log(`📁 Output: ${outputDir}`);
  console.log(`📊 History: ${historyDir}`);
  console.log(`🎯 Mode: ${mode.toUpperCase()}`);
  
  if (commitSha) {
    console.log(`📍 Commit: ${commitSha.substring(0, 8)}`);
  }

  // Ensure directories exist
  await fs.mkdir(outputDir, { recursive: true });
  await fs.mkdir(historyDir, { recursive: true });

  // Initialize orchestrator
  const orchestrator = new PhaseBOrchestrator(outputDir, historyDir);

  try {
    let result;

    switch (mode) {
      case 'comprehensive':
        result = await runComprehensiveValidation(orchestrator, commitSha);
        break;
      
      case 'ci':
        result = await runCIValidation(orchestrator, commitSha);
        break;
      
      case 'performance':
        result = await runPerformanceOnly(orchestrator);
        break;
      
      case 'stress':
        result = await runStressOnly(orchestrator);
        break;
      
      case 'regression':
        result = await runRegressionOnly(orchestrator, commitSha);
        break;
      
      default:
        console.error(`❌ Unknown mode: ${mode}`);
        console.log('Run with --help for available modes');
        process.exit(1);
    }

    // Print results summary
    printResultsSummary(result, mode);

    // Exit with appropriate code
    const exitCode = getExitCode(result, mode);
    console.log(`🎯 Validation ${exitCode === 0 ? 'PASSED' : 'FAILED'}`);
    process.exit(exitCode);

  } catch (error) {
    console.error('❌ Validation failed:', error);
    process.exit(1);
  }
}

async function runComprehensiveValidation(orchestrator: PhaseBOrchestrator, commitSha?: string) {
  const config: ValidationConfig = {
    validation_scope: {
      performance_validation: true,
      stress_testing: !args['skip-stress'],
      regression_detection: !args['skip-regression'] && !!commitSha,
      quality_validation: true,
      resource_monitoring: true
    },
    phase_b_config: {
      optimizations: {
        roaring_bitmap: { enabled: true, prefilter_candidate_files: true, roaring_compression: true },
        ast_cache: { enabled: true, max_files: 200, ttl_minutes: 60, batch_processing: true, stale_while_revalidate: true },
        isotonic_calibration: { enabled: true, confidence_cutoff: 0.12, ann_k: 150, ann_ef_search: 64 }
      },
      performance_targets: {
        stage_a_p95_ms: parseFloat(args['stage-a-target'] as string),
        stage_b_improvement_pct: 40,
        stage_c_improvement_pct: 40,
        e2e_p95_increase_max_pct: parseFloat(args['e2e-target'] as string),
        quality_ndcg_improvement_min: 0.02,
        quality_recall_maintain: true,
        span_coverage_min: 0.98
      }
    },
    stress_config: {
      load_tests: { concurrent_queries: [1, 5, 10, 20, 50], duration_minutes: 5, ramp_up_seconds: 30, query_rate_qps: [1, 5, 10, 20] },
      resource_pressure: { memory_pressure_enabled: true, memory_limit_mb: 1024, cpu_throttling_enabled: true, cpu_limit_percent: 80, io_pressure_enabled: true },
      endurance_testing: { long_running_hours: 2, query_burst_enabled: true, burst_multiplier: 5, burst_duration_seconds: 60, burst_interval_minutes: 15 },
      degradation_thresholds: { max_latency_degradation_pct: 50, max_throughput_degradation_pct: 30, max_error_rate_pct: 5, max_memory_growth_mb: 512, recovery_time_max_seconds: 60 }
    },
    regression_config: {
      baseline_config: { lookback_days: 7, min_samples: 5, percentile: 95 },
      thresholds: { stage_a_p95_regression_pct: 20, stage_b_p95_regression_pct: 15, stage_c_p95_regression_pct: 15, e2e_p95_regression_pct: 10, ndcg_regression_pct: 2, recall_regression_pct: 5, span_coverage_regression_pct: 2, memory_increase_pct: 25, cpu_increase_pct: 30, error_rate_increase_pct: 2, timeout_rate_increase_pct: 1 },
      monitoring: { early_warning_enabled: true, early_warning_threshold_pct: 75, trend_analysis_enabled: true, trend_window_measurements: 10, anomaly_detection_enabled: true, anomaly_sigma_threshold: 2.5 },
      ci_integration: { fail_build_on_regression: true, allow_override_flag: '--ignore-regression', notification_enabled: true, notification_channels: ['github'], auto_bisect_enabled: false, max_bisect_commits: 20 }
    },
    reporting: {
      comprehensive_report_enabled: true,
      executive_summary_enabled: true,
      detailed_analysis_enabled: true,
      performance_charts_enabled: true,
      export_formats: ['json', 'markdown'],
      notification_on_completion: false
    }
  };

  return await orchestrator.runComprehensiveValidation(config, commitSha);
}

async function runCIValidation(orchestrator: PhaseBOrchestrator, commitSha?: string) {
  if (!commitSha) {
    console.error('❌ CI mode requires --commit option');
    process.exit(1);
  }

  return await orchestrator.runCIValidation(commitSha);
}

async function runPerformanceOnly(orchestrator: PhaseBOrchestrator) {
  const config = {
    optimizations: {
      roaring_bitmap: { enabled: true },
      ast_cache: { enabled: true },
      isotonic_calibration: { enabled: true }
    },
    performance_targets: {
      stage_a_p95_ms: parseFloat(args['stage-a-target'] as string),
      stage_b_improvement_pct: 40,
      stage_c_improvement_pct: 40,
      e2e_p95_increase_max_pct: parseFloat(args['e2e-target'] as string),
      quality_ndcg_improvement_min: 0.02,
      quality_recall_maintain: true,
      span_coverage_min: 0.98
    }
  };

  return await orchestrator.runPerformanceValidationOnly(config);
}

async function runStressOnly(orchestrator: PhaseBOrchestrator) {
  const stages = [OptimizationStage.BASELINE, OptimizationStage.INTEGRATED];
  const config = {
    load_tests: { concurrent_queries: [1, 5, 10, 20], duration_minutes: 3, ramp_up_seconds: 15, query_rate_qps: [1, 5, 10] },
    resource_pressure: { memory_pressure_enabled: true, memory_limit_mb: 1024, cpu_throttling_enabled: true, cpu_limit_percent: 80, io_pressure_enabled: false },
    endurance_testing: { long_running_hours: 1, query_burst_enabled: true, burst_multiplier: 3, burst_duration_seconds: 30, burst_interval_minutes: 10 },
    degradation_thresholds: { max_latency_degradation_pct: 50, max_throughput_degradation_pct: 30, max_error_rate_pct: 5, max_memory_growth_mb: 256, recovery_time_max_seconds: 60 }
  };

  return await orchestrator.runStressTestingOnly(stages, config);
}

async function runRegressionOnly(orchestrator: PhaseBOrchestrator, commitSha?: string) {
  if (!commitSha) {
    console.error('❌ Regression mode requires --commit option');
    process.exit(1);
  }

  // Create regression system directly
  const regressionSystem = new RegressionPreventionSystem(
    {} as any, // BenchmarkSuiteRunner - would be properly initialized
    args.output as string,
    args.history as string
  );

  const config = {
    baseline_config: { lookback_days: 7, min_samples: 5, percentile: 95 },
    thresholds: { stage_a_p95_regression_pct: 20, stage_b_p95_regression_pct: 15, stage_c_p95_regression_pct: 15, e2e_p95_regression_pct: 10, ndcg_regression_pct: 2, recall_regression_pct: 5, span_coverage_regression_pct: 2, memory_increase_pct: 25, cpu_increase_pct: 30, error_rate_increase_pct: 2, timeout_rate_increase_pct: 1 },
    monitoring: { early_warning_enabled: true, early_warning_threshold_pct: 75, trend_analysis_enabled: true, trend_window_measurements: 10, anomaly_detection_enabled: true, anomaly_sigma_threshold: 2.5 },
    ci_integration: { fail_build_on_regression: true, allow_override_flag: '--ignore-regression', notification_enabled: false, notification_channels: ['github'], auto_bisect_enabled: false, max_bisect_commits: 20 }
  };

  return await regressionSystem.detectRegressions(commitSha, config);
}

function printResultsSummary(result: any, mode: string) {
  console.log('\n📊 RESULTS SUMMARY');
  console.log('==================');

  switch (mode) {
    case 'comprehensive':
      console.log(`Status: ${result.overall_assessment.validation_passed ? '✅ PASSED' : '❌ FAILED'}`);
      console.log(`Gate: ${result.overall_assessment.promotion_gate_status.toUpperCase()}`);
      console.log(`Duration: ${(result.execution_metadata.total_duration_ms / 1000 / 60).toFixed(1)} min`);
      
      if (result.overall_assessment.critical_issues.length > 0) {
        console.log('\n❌ Critical Issues:');
        result.overall_assessment.critical_issues.forEach((issue: string) => {
          console.log(`   • ${issue}`);
        });
      }
      
      if (result.overall_assessment.warnings.length > 0) {
        console.log('\n⚠️  Warnings:');
        result.overall_assessment.warnings.forEach((warning: string) => {
          console.log(`   • ${warning}`);
        });
      }

      console.log('\n💡 Recommendations:');
      result.overall_assessment.recommendations.forEach((rec: string) => {
        console.log(`   • ${rec}`);
      });
      break;

    case 'ci':
      console.log(`Status: ${result.passed ? '✅ PASSED' : '❌ FAILED'}`);
      console.log(`Summary: ${result.summary}`);
      break;

    case 'performance':
      console.log(`Gate: ${result.promotion_gate.passed ? '✅ PASSED' : '❌ FAILED'}`);
      console.log(`Summary: ${result.promotion_gate.summary}`);
      
      if (result.comparisons && result.comparisons.length > 0) {
        const integrated = result.comparisons.find((c: any) => c.treatment_stage === 'B_integrated');
        if (integrated) {
          console.log('\n🎯 Performance Improvements:');
          console.log(`   Stage-A: ${integrated.performance_improvement.stage_a_improvement_pct.toFixed(1)}%`);
          console.log(`   Stage-B: ${integrated.performance_improvement.stage_b_improvement_pct.toFixed(1)}%`);
          console.log(`   E2E: ${integrated.performance_improvement.e2e_improvement_pct.toFixed(1)}%`);
        }
      }
      break;

    case 'stress':
      Object.entries(result).forEach(([stage, stressResult]: [string, any]) => {
        console.log(`${stage}: ${stressResult.stress_gate_evaluation.passed ? '✅ PASSED' : '❌ FAILED'} (${stressResult.stress_gate_evaluation.stability_rating})`);
      });
      break;

    case 'regression':
      console.log(`Gate: ${result.gate_evaluation.passed ? '✅ PASSED' : '❌ FAILED'}`);
      console.log(`Recommendation: ${result.gate_evaluation.recommendation.toUpperCase()}`);
      
      if (result.gate_evaluation.blocking_regressions.length > 0) {
        console.log('\n❌ Blocking Regressions:');
        result.gate_evaluation.blocking_regressions.forEach((reg: string) => {
          console.log(`   • ${reg}`);
        });
      }
      break;
  }
}

function getExitCode(result: any, mode: string): number {
  switch (mode) {
    case 'comprehensive':
      return result.overall_assessment.validation_passed ? 0 : 1;
    
    case 'ci':
      return result.passed ? 0 : 1;
    
    case 'performance':
      return result.promotion_gate.passed ? 0 : 1;
    
    case 'stress':
      const allPassed = Object.values(result).every((r: any) => r.stress_gate_evaluation.passed);
      return allPassed ? 0 : 1;
    
    case 'regression':
      return result.gate_evaluation.passed ? 0 : 1;
    
    default:
      return 1;
  }
}

// Run main function
main().catch((error) => {
  console.error('❌ Script failed:', error);
  process.exit(1);
});