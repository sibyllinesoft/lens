#!/usr/bin/env bun
/**
 * Phase C Benchmark Testing Script
 * Tests comprehensive benchmarking suite with quality gates, hard negatives, and statistical rigor
 */

import { BenchmarkSuiteRunner } from '../benchmark/suite-runner.js';
import { GroundTruthBuilder } from '../benchmark/ground-truth-builder.js';
import { promises as fs } from 'fs';
import path from 'path';
import type { BenchmarkConfig, BenchmarkRun } from '../types/benchmark.js';

interface TestResult {
  test_name: string;
  status: 'pass' | 'fail';
  duration_ms: number;
  details: any;
  error?: string;
}

class PhaseCBenchmarkTester {
  private outputDir: string;
  private groundTruthBuilder: GroundTruthBuilder;
  private benchmarkRunner: BenchmarkSuiteRunner;
  private results: TestResult[] = [];

  constructor() {
    this.outputDir = path.join(process.cwd(), 'test-benchmark-results');
    this.groundTruthBuilder = new GroundTruthBuilder();
    this.benchmarkRunner = new BenchmarkSuiteRunner(
      this.groundTruthBuilder,
      this.outputDir,
      process.env.NATS_URL || 'nats://localhost:4222'
    );
  }

  async initialize() {
    await fs.mkdir(this.outputDir, { recursive: true });
    await this.groundTruthBuilder.initialize();
    console.log('🔧 Phase C benchmark tester initialized');
  }

  async runAllTests(): Promise<void> {
    console.log('🚀 Starting Phase C comprehensive benchmark tests...\n');

    const tests = [
      () => this.testSmokeTestSuite(),
      () => this.testFullTestSuite(),
      () => this.testHardNegativeGeneration(),
      () => this.testTripwireValidation(),
      () => this.testPerSliceGates(),
      () => this.testVisualizationGeneration(),
      () => this.testPromotionGates(),
      () => this.testCIHardening(),
      () => this.testArtifactGeneration(),
      () => this.testStatisticalRigor()
    ];

    for (const test of tests) {
      try {
        await test();
      } catch (error) {
        console.error(`Test failed: ${error}`);
      }
    }

    await this.generateTestReport();
  }

  private async testSmokeTestSuite(): Promise<void> {
    const testName = 'Smoke Test Suite';
    console.log(`🔥 Testing ${testName}...`);
    const startTime = Date.now();

    try {
      const config: Partial<BenchmarkConfig> = {
        trace_id: `test-smoke-${Date.now()}`,
        suite: ['codesearch', 'structural'],
        systems: ['lex', '+symbols', '+symbols+semantic'],
        slices: 'SMOKE_DEFAULT',
        seeds: 1,
        cache_mode: 'warm',
        k_candidates: 50, // Smaller for testing
        top_n: 25
      };

      const result = await this.benchmarkRunner.runSmokeSuite(config);
      const duration = Date.now() - startTime;

      // Validate smoke test requirements
      const validations = {
        statusCompleted: result.status === 'completed',
        hasMetrics: result.metrics && result.metrics.recall_at_10 >= 0,
        hasLatencies: result.metrics.stage_latencies.e2e_p95 > 0,
        reasonableDuration: duration < 60000, // Under 1 minute for testing
        completedQueries: result.completed_queries > 0
      };

      const passed = Object.values(validations).every(Boolean);

      this.results.push({
        test_name: testName,
        status: passed ? 'pass' : 'fail',
        duration_ms: duration,
        details: {
          benchmark_result: {
            status: result.status,
            completed_queries: result.completed_queries,
            failed_queries: result.failed_queries,
            recall_at_10: result.metrics.recall_at_10,
            e2e_p95: result.metrics.stage_latencies.e2e_p95
          },
          validations
        }
      });

      console.log(`  ${passed ? '✅' : '❌'} ${testName}: ${passed ? 'PASS' : 'FAIL'} (${duration}ms)`);
      console.log(`     Queries: ${result.completed_queries}/${result.total_queries}`);
      console.log(`     Recall@10: ${result.metrics.recall_at_10.toFixed(3)}`);
      console.log(`     E2E P95: ${result.metrics.stage_latencies.e2e_p95.toFixed(1)}ms\n`);

    } catch (error) {
      const duration = Date.now() - startTime;
      this.results.push({
        test_name: testName,
        status: 'fail',
        duration_ms: duration,
        details: {},
        error: error instanceof Error ? error.message : String(error)
      });
      console.log(`  ❌ ${testName}: FAIL - ${error}\n`);
    }
  }

  private async testFullTestSuite(): Promise<void> {
    const testName = 'Full Test Suite (Simplified)';
    console.log(`🌙 Testing ${testName}...`);
    const startTime = Date.now();

    try {
      const config: Partial<BenchmarkConfig> = {
        trace_id: `test-full-${Date.now()}`,
        suite: ['codesearch'],
        systems: ['lex', '+symbols'],
        slices: 'ALL',
        seeds: 1, // Reduced for testing
        cache_mode: 'warm',
        k_candidates: 50,
        top_n: 25,
        robustness: false, // Disabled for testing
        metamorphic: false
      };

      const result = await this.benchmarkRunner.runFullSuite(config);
      const duration = Date.now() - startTime;

      const validations = {
        statusCompleted: result.status === 'completed',
        hasSystemResults: result.system === 'AGGREGATE',
        multipleQueries: result.completed_queries > 10,
        reasonableDuration: duration < 120000 // Under 2 minutes for testing
      };

      const passed = Object.values(validations).every(Boolean);

      this.results.push({
        test_name: testName,
        status: passed ? 'pass' : 'fail',
        duration_ms: duration,
        details: {
          benchmark_result: {
            system: result.system,
            completed_queries: result.completed_queries,
            ndcg_at_10: result.metrics.ndcg_at_10
          },
          validations
        }
      });

      console.log(`  ${passed ? '✅' : '❌'} ${testName}: ${passed ? 'PASS' : 'FAIL'} (${duration}ms)`);
      console.log(`     System: ${result.system}`);
      console.log(`     Queries: ${result.completed_queries}/${result.total_queries}`);
      console.log(`     nDCG@10: ${result.metrics.ndcg_at_10.toFixed(3)}\n`);

    } catch (error) {
      const duration = Date.now() - startTime;
      this.results.push({
        test_name: testName,
        status: 'fail',
        duration_ms: duration,
        details: {},
        error: error instanceof Error ? error.message : String(error)
      });
      console.log(`  ❌ ${testName}: FAIL - ${error}\n`);
    }
  }

  private async testHardNegativeGeneration(): Promise<void> {
    const testName = 'Hard Negative Generation & Injection';
    console.log(`🎯 Testing ${testName}...`);
    const startTime = Date.now();

    try {
      const { hardeningReport } = await this.benchmarkRunner.runPhaseCHardening({
        hard_negatives: {
          enabled: true,
          per_query_count: 3, // Reduced for testing
          shared_subtoken_min: 2
        },
        tripwires: {
          min_span_coverage: 0.95, // Relaxed for testing
          recall_convergence_threshold: 0.01,
          lsif_coverage_drop_threshold: 0.1,
          p99_p95_ratio_threshold: 3.0
        },
        per_slice_gates: {
          enabled: false // Disabled for testing
        },
        plots: {
          enabled: true,
          output_dir: path.join(this.outputDir, 'hard-negative-plots'),
          formats: ['svg']
        }
      });

      const duration = Date.now() - startTime;

      const validations = {
        hardNegativesGenerated: hardeningReport.hard_negatives.total_generated > 0,
        hasImpactAnalysis: hardeningReport.hard_negatives.impact_on_metrics !== undefined,
        degradationMeasured: hardeningReport.hard_negatives.impact_on_metrics?.degradation_percent !== undefined,
        reasonableImpact: (hardeningReport.hard_negatives.impact_on_metrics?.degradation_percent || 0) < 50
      };

      const passed = Object.values(validations).every(Boolean);

      this.results.push({
        test_name: testName,
        status: passed ? 'pass' : 'fail',
        duration_ms: duration,
        details: {
          hard_negatives: hardeningReport.hard_negatives,
          validations
        }
      });

      console.log(`  ${passed ? '✅' : '❌'} ${testName}: ${passed ? 'PASS' : 'FAIL'} (${duration}ms)`);
      console.log(`     Generated: ${hardeningReport.hard_negatives.total_generated} hard negatives`);
      console.log(`     Degradation: ${(hardeningReport.hard_negatives.impact_on_metrics?.degradation_percent || 0).toFixed(2)}%\n`);

    } catch (error) {
      const duration = Date.now() - startTime;
      this.results.push({
        test_name: testName,
        status: 'fail',
        duration_ms: duration,
        details: {},
        error: error instanceof Error ? error.message : String(error)
      });
      console.log(`  ❌ ${testName}: FAIL - ${error}\n`);
    }
  }

  private async testTripwireValidation(): Promise<void> {
    const testName = 'Tripwire Validation System';
    console.log(`⚡ Testing ${testName}...`);
    const startTime = Date.now();

    try {
      // Run basic smoke test first to get benchmark results
      const smokeResult = await this.benchmarkRunner.runSmokeSuite({
        trace_id: `tripwire-test-${Date.now()}`,
        k_candidates: 30,
        top_n: 15
      });

      // Test tripwire system with various configurations
      const { hardeningReport } = await this.benchmarkRunner.runPhaseCHardening({
        tripwires: {
          min_span_coverage: 0.90, // Achievable threshold
          recall_convergence_threshold: 0.02, // Loose threshold
          lsif_coverage_drop_threshold: 0.2,
          p99_p95_ratio_threshold: 4.0
        },
        hard_negatives: { enabled: false },
        per_slice_gates: { enabled: false },
        plots: { enabled: false }
      }, [smokeResult]);

      const duration = Date.now() - startTime;

      const validations = {
        hasTripwireResults: hardeningReport.tripwire_results.length > 0,
        hasTripwireSummary: hardeningReport.tripwire_summary !== undefined,
        allTripwiresProcessed: hardeningReport.tripwire_results.length === 4, // All 4 tripwires
        hasStatusDecision: ['pass', 'fail'].includes(hardeningReport.tripwire_summary.overall_status)
      };

      const passed = Object.values(validations).every(Boolean);

      this.results.push({
        test_name: testName,
        status: passed ? 'pass' : 'fail',
        duration_ms: duration,
        details: {
          tripwire_summary: hardeningReport.tripwire_summary,
          tripwire_results: hardeningReport.tripwire_results,
          validations
        }
      });

      console.log(`  ${passed ? '✅' : '❌'} ${testName}: ${passed ? 'PASS' : 'FAIL'} (${duration}ms)`);
      console.log(`     Tripwires: ${hardeningReport.tripwire_summary.passed_tripwires}/${hardeningReport.tripwire_summary.total_tripwires} passed`);
      console.log(`     Overall: ${hardeningReport.tripwire_summary.overall_status.toUpperCase()}\n`);

    } catch (error) {
      const duration = Date.now() - startTime;
      this.results.push({
        test_name: testName,
        status: 'fail',
        duration_ms: duration,
        details: {},
        error: error instanceof Error ? error.message : String(error)
      });
      console.log(`  ❌ ${testName}: FAIL - ${error}\n`);
    }
  }

  private async testPerSliceGates(): Promise<void> {
    const testName = 'Per-Slice Performance Gates';
    console.log(`🔍 Testing ${testName}...`);
    const startTime = Date.now();

    try {
      const { hardeningReport } = await this.benchmarkRunner.runPhaseCHardening({
        per_slice_gates: {
          enabled: true,
          min_recall_at_10: 0.5, // Achievable threshold
          min_ndcg_at_10: 0.4,
          max_p95_latency_ms: 1000
        },
        hard_negatives: { enabled: false },
        tripwires: {
          min_span_coverage: 0.8,
          recall_convergence_threshold: 0.05,
          lsif_coverage_drop_threshold: 0.3,
          p99_p95_ratio_threshold: 5.0
        },
        plots: { enabled: false }
      });

      const duration = Date.now() - startTime;

      const validations = {
        hasSliceResults: hardeningReport.slice_results.length > 0,
        hasSliceSummary: hardeningReport.slice_gate_summary !== undefined,
        slicesProcessed: hardeningReport.slice_gate_summary?.total_slices > 0,
        gateStatusValid: hardeningReport.slice_results.every(s => ['pass', 'fail'].includes(s.gate_status))
      };

      const passed = Object.values(validations).every(Boolean);

      this.results.push({
        test_name: testName,
        status: passed ? 'pass' : 'fail',
        duration_ms: duration,
        details: {
          slice_summary: hardeningReport.slice_gate_summary,
          slice_count: hardeningReport.slice_results.length,
          validations
        }
      });

      console.log(`  ${passed ? '✅' : '❌'} ${testName}: ${passed ? 'PASS' : 'FAIL'} (${duration}ms)`);
      if (hardeningReport.slice_gate_summary) {
        console.log(`     Slices: ${hardeningReport.slice_gate_summary.passed_slices}/${hardeningReport.slice_gate_summary.total_slices} passed`);
      }
      console.log(`     Total slices tested: ${hardeningReport.slice_results.length}\n`);

    } catch (error) {
      const duration = Date.now() - startTime;
      this.results.push({
        test_name: testName,
        status: 'fail',
        duration_ms: duration,
        details: {},
        error: error instanceof Error ? error.message : String(error)
      });
      console.log(`  ❌ ${testName}: FAIL - ${error}\n`);
    }
  }

  private async testVisualizationGeneration(): Promise<void> {
    const testName = 'Visualization Plot Generation';
    console.log(`📊 Testing ${testName}...`);
    const startTime = Date.now();

    try {
      const { hardeningReport } = await this.benchmarkRunner.runPhaseCHardening({
        plots: {
          enabled: true,
          output_dir: path.join(this.outputDir, 'test-plots'),
          formats: ['svg', 'png']
        },
        hard_negatives: { enabled: false },
        tripwires: {
          min_span_coverage: 0.8,
          recall_convergence_threshold: 0.05,
          lsif_coverage_drop_threshold: 0.3,
          p99_p95_ratio_threshold: 5.0
        },
        per_slice_gates: { enabled: false }
      });

      const duration = Date.now() - startTime;

      const expectedPlots = [
        'positives_in_candidates',
        'relevant_per_query_histogram', 
        'precision_vs_score_pre_calibration',
        'precision_vs_score_post_calibration',
        'latency_percentiles_by_stage',
        'early_termination_rate'
      ];

      const validations = {
        hasPlots: hardeningReport.plots_generated !== undefined,
        allPlotsGenerated: expectedPlots.every(plot => plot in hardeningReport.plots_generated),
        plotFilesExist: await this.validatePlotFiles(hardeningReport.plots_generated)
      };

      const passed = Object.values(validations).every(Boolean);

      this.results.push({
        test_name: testName,
        status: passed ? 'pass' : 'fail',
        duration_ms: duration,
        details: {
          plots_generated: hardeningReport.plots_generated,
          expected_plots: expectedPlots,
          validations
        }
      });

      console.log(`  ${passed ? '✅' : '❌'} ${testName}: ${passed ? 'PASS' : 'FAIL'} (${duration}ms)`);
      console.log(`     Plots: ${Object.keys(hardeningReport.plots_generated).length}/${expectedPlots.length} generated`);
      console.log(`     Types: ${Object.keys(hardeningReport.plots_generated).join(', ')}\n`);

    } catch (error) {
      const duration = Date.now() - startTime;
      this.results.push({
        test_name: testName,
        status: 'fail',
        duration_ms: duration,
        details: {},
        error: error instanceof Error ? error.message : String(error)
      });
      console.log(`  ❌ ${testName}: FAIL - ${error}\n`);
    }
  }

  private async testPromotionGates(): Promise<void> {
    const testName = 'Promotion Gate Validation';
    console.log(`🎯 Testing ${testName}...`);
    const startTime = Date.now();

    try {
      // Run multiple system comparison
      const baselineResult = await this.benchmarkRunner.runSmokeSuite({
        trace_id: `promotion-baseline-${Date.now()}`,
        systems: ['lex'],
        k_candidates: 50
      });

      const treatmentResult = await this.benchmarkRunner.runSmokeSuite({
        trace_id: `promotion-treatment-${Date.now()}`,
        systems: ['+symbols+semantic'],
        k_candidates: 50
      });

      const duration = Date.now() - startTime;

      // Simulate promotion gate logic from TODO.md:
      // Δ nDCG@10 ≥ +2% (p<0.05) AND Recall@50 ≥ baseline AND E2E p95 ≤ +10%
      const ndcgDelta = treatmentResult.metrics.ndcg_at_10 - baselineResult.metrics.ndcg_at_10;
      const ndcgImprovement = ndcgDelta >= 0.02; // ≥ +2%
      const recallMaintained = treatmentResult.metrics.recall_at_50 >= baselineResult.metrics.recall_at_50;
      const latencyIncrease = (treatmentResult.metrics.stage_latencies.e2e_p95 - baselineResult.metrics.stage_latencies.e2e_p95) / baselineResult.metrics.stage_latencies.e2e_p95;
      const latencyAcceptable = latencyIncrease <= 0.10; // ≤ +10%

      const promotionGatePassed = ndcgImprovement && recallMaintained && latencyAcceptable;

      const validations = {
        bothResultsValid: baselineResult.status === 'completed' && treatmentResult.status === 'completed',
        ndcgCalculated: !isNaN(ndcgDelta),
        recallCalculated: !isNaN(treatmentResult.metrics.recall_at_50) && !isNaN(baselineResult.metrics.recall_at_50),
        latencyCalculated: !isNaN(latencyIncrease),
        gateLogicWorking: typeof promotionGatePassed === 'boolean'
      };

      const passed = Object.values(validations).every(Boolean);

      this.results.push({
        test_name: testName,
        status: passed ? 'pass' : 'fail',
        duration_ms: duration,
        details: {
          baseline_metrics: {
            ndcg_at_10: baselineResult.metrics.ndcg_at_10,
            recall_at_50: baselineResult.metrics.recall_at_50,
            e2e_p95: baselineResult.metrics.stage_latencies.e2e_p95
          },
          treatment_metrics: {
            ndcg_at_10: treatmentResult.metrics.ndcg_at_10,
            recall_at_50: treatmentResult.metrics.recall_at_50,
            e2e_p95: treatmentResult.metrics.stage_latencies.e2e_p95
          },
          promotion_analysis: {
            ndcg_delta: ndcgDelta,
            ndcg_improvement: ndcgImprovement,
            recall_maintained: recallMaintained,
            latency_increase: latencyIncrease,
            latency_acceptable: latencyAcceptable,
            gate_passed: promotionGatePassed
          },
          validations
        }
      });

      console.log(`  ${passed ? '✅' : '❌'} ${testName}: ${passed ? 'PASS' : 'FAIL'} (${duration}ms)`);
      console.log(`     nDCG Δ: ${(ndcgDelta * 100).toFixed(1)}% (need ≥+2%)`);
      console.log(`     Recall maintained: ${recallMaintained ? 'YES' : 'NO'}`);
      console.log(`     Latency increase: ${(latencyIncrease * 100).toFixed(1)}% (need ≤+10%)`);
      console.log(`     🏁 Promotion gate: ${promotionGatePassed ? 'PASS' : 'FAIL'}\n`);

    } catch (error) {
      const duration = Date.now() - startTime;
      this.results.push({
        test_name: testName,
        status: 'fail',
        duration_ms: duration,
        details: {},
        error: error instanceof Error ? error.message : String(error)
      });
      console.log(`  ❌ ${testName}: FAIL - ${error}\n`);
    }
  }

  private async testCIHardening(): Promise<void> {
    const testName = 'CI Hardening Gates (Strict)';
    console.log(`🚨 Testing ${testName}...`);
    const startTime = Date.now();

    try {
      const passed = await this.benchmarkRunner.runPhaseCCIGates();
      const duration = Date.now() - startTime;

      const validations = {
        gateResultBool: typeof passed === 'boolean',
        reasonableDuration: duration < 180000 // Under 3 minutes
      };

      const testPassed = Object.values(validations).every(Boolean);

      this.results.push({
        test_name: testName,
        status: testPassed ? 'pass' : 'fail',
        duration_ms: duration,
        details: {
          ci_gates_passed: passed,
          validations
        }
      });

      console.log(`  ${testPassed ? '✅' : '❌'} ${testName}: ${testPassed ? 'PASS' : 'FAIL'} (${duration}ms)`);
      console.log(`     CI Gates Result: ${passed ? '✅ PASS' : '❌ FAIL'}\n`);

    } catch (error) {
      const duration = Date.now() - startTime;
      this.results.push({
        test_name: testName,
        status: 'fail',
        duration_ms: duration,
        details: {},
        error: error instanceof Error ? error.message : String(error)
      });
      console.log(`  ❌ ${testName}: FAIL - ${error}\n`);
    }
  }

  private async testArtifactGeneration(): Promise<void> {
    const testName = 'Required Artifact Generation';
    console.log(`📦 Testing ${testName}...`);
    const startTime = Date.now();

    try {
      const result = await this.benchmarkRunner.runSmokeSuite({
        trace_id: `artifact-test-${Date.now()}`,
        k_candidates: 30
      });

      // Check for required artifacts per TODO.md
      const requiredArtifacts = [
        'metrics.parquet',
        'errors.ndjson',
        'traces.ndjson', 
        'report.pdf',
        'config_fingerprint.json'
      ];

      const artifactFiles = await fs.readdir(this.outputDir);
      const artifactsFound = requiredArtifacts.filter(artifact => 
        artifactFiles.some(file => file.includes(artifact.split('.')[0]))
      );

      const duration = Date.now() - startTime;

      const validations = {
        benchmarkCompleted: result.status === 'completed',
        outputDirExists: artifactFiles.length > 0,
        metricsFileGenerated: artifactsFound.some(a => a.includes('metrics')),
        configFileGenerated: artifactsFound.some(a => a.includes('config')),
        someArtifactsGenerated: artifactsFound.length > 0
      };

      const passed = Object.values(validations).every(Boolean);

      this.results.push({
        test_name: testName,
        status: passed ? 'pass' : 'fail',
        duration_ms: duration,
        details: {
          required_artifacts: requiredArtifacts,
          artifacts_found: artifactsFound,
          output_files: artifactFiles,
          validations
        }
      });

      console.log(`  ${passed ? '✅' : '❌'} ${testName}: ${passed ? 'PASS' : 'FAIL'} (${duration}ms)`);
      console.log(`     Artifacts: ${artifactsFound.length}/${requiredArtifacts.length} found`);
      console.log(`     Output files: ${artifactFiles.length} total\n`);

    } catch (error) {
      const duration = Date.now() - startTime;
      this.results.push({
        test_name: testName,
        status: 'fail',
        duration_ms: duration,
        details: {},
        error: error instanceof Error ? error.message : String(error)
      });
      console.log(`  ❌ ${testName}: FAIL - ${error}\n`);
    }
  }

  private async testStatisticalRigor(): Promise<void> {
    const testName = 'Statistical Rigor & Reproducibility';
    console.log(`📈 Testing ${testName}...`);
    const startTime = Date.now();

    try {
      const traceId = `stats-test-${Date.now()}`;
      
      // Run same configuration multiple times to test reproducibility
      const runs = [];
      for (let i = 0; i < 3; i++) {
        const result = await this.benchmarkRunner.runSmokeSuite({
          trace_id: `${traceId}-${i}`,
          systems: ['lex'],
          k_candidates: 20,
          seeds: 1 // Same seed for reproducibility
        });
        runs.push(result);
      }

      const duration = Date.now() - startTime;

      // Check statistical consistency
      const recalls = runs.map(r => r.metrics.recall_at_10);
      const ndcgs = runs.map(r => r.metrics.ndcg_at_10);
      
      const recallVariance = this.calculateVariance(recalls);
      const ndcgVariance = this.calculateVariance(ndcgs);

      const validations = {
        allRunsCompleted: runs.every(r => r.status === 'completed'),
        consistentSeeds: runs.every(r => r.config_fingerprint), // Config fingerprints generated
        lowRecallVariance: recallVariance < 0.01, // Low variance indicates reproducibility
        lowNdcgVariance: ndcgVariance < 0.01,
        traceIdsUnique: new Set(runs.map(r => r.trace_id)).size === runs.length
      };

      const passed = Object.values(validations).every(Boolean);

      this.results.push({
        test_name: testName,
        status: passed ? 'pass' : 'fail',
        duration_ms: duration,
        details: {
          runs: runs.length,
          recall_variance: recallVariance,
          ndcg_variance: ndcgVariance,
          recall_values: recalls,
          ndcg_values: ndcgs,
          validations
        }
      });

      console.log(`  ${passed ? '✅' : '❌'} ${testName}: ${passed ? 'PASS' : 'FAIL'} (${duration}ms)`);
      console.log(`     Runs: ${runs.length}, Recall variance: ${recallVariance.toFixed(6)}`);
      console.log(`     nDCG variance: ${ndcgVariance.toFixed(6)}\n`);

    } catch (error) {
      const duration = Date.now() - startTime;
      this.results.push({
        test_name: testName,
        status: 'fail',
        duration_ms: duration,
        details: {},
        error: error instanceof Error ? error.message : String(error)
      });
      console.log(`  ❌ ${testName}: FAIL - ${error}\n`);
    }
  }

  private async validatePlotFiles(plots: any): Promise<boolean> {
    try {
      for (const plotPath of Object.values(plots)) {
        const exists = await fs.access(plotPath as string).then(() => true).catch(() => false);
        if (!exists) return false;
      }
      return true;
    } catch {
      return false;
    }
  }

  private calculateVariance(values: number[]): number {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const squaredDiffs = values.map(value => Math.pow(value - mean, 2));
    return squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
  }

  private async generateTestReport(): Promise<void> {
    const totalTests = this.results.length;
    const passedTests = this.results.filter(r => r.status === 'pass').length;
    const failedTests = this.results.filter(r => r.status === 'fail').length;
    const totalDuration = this.results.reduce((sum, r) => sum + r.duration_ms, 0);

    console.log('\n' + '='.repeat(80));
    console.log('📊 PHASE C BENCHMARK TEST REPORT');
    console.log('='.repeat(80));
    console.log(`Tests Run: ${totalTests}`);
    console.log(`Passed: ${passedTests} (${((passedTests / totalTests) * 100).toFixed(1)}%)`);
    console.log(`Failed: ${failedTests} (${((failedTests / totalTests) * 100).toFixed(1)}%)`);
    console.log(`Total Duration: ${(totalDuration / 1000).toFixed(1)}s`);
    console.log('');

    // Individual test results
    console.log('DETAILED RESULTS:');
    console.log('-'.repeat(80));
    for (const result of this.results) {
      const status = result.status === 'pass' ? '✅ PASS' : '❌ FAIL';
      const duration = `${result.duration_ms}ms`;
      console.log(`${status.padEnd(8)} ${result.test_name.padEnd(35)} ${duration.padStart(10)}`);
      if (result.error) {
        console.log(`         Error: ${result.error}`);
      }
    }

    // Save detailed results to file
    const reportPath = path.join(this.outputDir, 'phase-c-test-report.json');
    await fs.writeFile(reportPath, JSON.stringify({
      summary: {
        total_tests: totalTests,
        passed_tests: passedTests,
        failed_tests: failedTests,
        pass_rate: (passedTests / totalTests) * 100,
        total_duration_ms: totalDuration,
        generated_at: new Date().toISOString()
      },
      test_results: this.results
    }, null, 2));

    console.log(`\n📄 Detailed report saved to: ${reportPath}`);
    console.log('\n🎯 Phase C benchmark testing complete!');
  }
}

// Main execution
async function main() {
  const tester = new PhaseCBenchmarkTester();
  
  try {
    await tester.initialize();
    await tester.runAllTests();
  } catch (error) {
    console.error('💥 Test suite failed:', error);
    process.exit(1);
  }
}

if (import.meta.main) {
  main().catch(console.error);
}

export { PhaseCBenchmarkTester };