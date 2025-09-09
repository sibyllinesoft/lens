#!/usr/bin/env node

/**
 * Governance Framework Demo Script
 * Demonstrates the complete governance and statistical rigor system
 * Usage: npm run governance:demo [--full] [--output-dir <dir>] [--audit-bundle]
 */

import path from 'path';
import { promises as fs } from 'fs';
import { Command } from 'commander';
import {
  createBenchmarkOrchestrator,
  BenchmarkGovernanceSystem,
  AuditBundleGenerator,
  RedTeamValidationSuite
} from '../../benchmarks/src/index.js';
import type { BenchmarkConfig } from '../types/benchmark.js';

const program = new Command();

program
  .name('governance-demo')
  .description('Demonstrate Lens governance and statistical rigor framework')
  .option('-f, --full', 'Run full benchmark suite with governance validation', false)
  .option('-o, --output-dir <dir>', 'Output directory for results', './governance-demo-output')
  .option('-a, --audit-bundle', 'Generate audit bundle for reproducibility', false)
  .option('-r, --redteam', 'Run red-team validation suite', false)
  .option('-s, --statistical-power', 'Run statistical power analysis demo', false)
  .option('-c, --calibration', 'Run calibration monitoring demo', false)
  .option('-m, --multiple-testing', 'Run multiple testing correction demo', false)
  .option('-v, --verbose', 'Enable verbose output', false)
  .parse();

const options = program.opts();

async function main() {
  console.log('🏛️ Lens Governance Framework Demonstration');
  console.log('==========================================');
  console.log();
  
  // Ensure output directory exists
  await fs.mkdir(options.outputDir, { recursive: true });
  
  // Initialize governance system
  const governanceSystem = new BenchmarkGovernanceSystem(options.outputDir);
  
  if (options.full) {
    await runFullGovernanceDemo(options.outputDir);
  }
  
  if (options.auditBundle) {
    await runAuditBundleDemo(options.outputDir);
  }
  
  if (options.redteam) {
    await runRedTeamDemo(options.outputDir);
  }
  
  if (options.statisticalPower) {
    await runStatisticalPowerDemo(governanceSystem);
  }
  
  if (options.calibration) {
    await runCalibrationDemo(governanceSystem);
  }
  
  if (options.multipleTesting) {
    await runMultipleTestingDemo(governanceSystem);
  }
  
  // Default: run versioned fingerprint demo
  if (!options.full && !options.auditBundle && !options.redteam && 
      !options.statisticalPower && !options.calibration && !options.multipleTesting) {
    await runVersionedFingerprintDemo(governanceSystem);
  }
  
  console.log();
  console.log(`✅ Governance demo complete. Results in: ${options.outputDir}`);
}

/**
 * Run complete governance-enabled benchmark
 */
async function runFullGovernanceDemo(outputDir: string) {
  console.log('🚀 Running Full Governance-Enabled Benchmark');
  console.log('=============================================');
  
  // Create benchmark orchestrator with governance enabled
  const orchestrator = createBenchmarkOrchestrator({
    workingDir: process.cwd(),
    outputDir,
    governanceEnabled: true,
    repositories: [
      { name: 'lens-demo', path: process.cwd() }
    ],
    auditBundleConfig: {
      outputDir,
      includeSourceCode: true,
      includeDatasets: true,
      includeModels: true,
      includeDependencies: true
    },
    redteamConfig: {
      outputDir,
      leakSentinelEnabled: true,
      verbosityDopingEnabled: true,
      tamperDetectionEnabled: true,
      ngramOverlapThreshold: 0.1,
      weeklySchedule: false // Disable for demo
    }
  });
  
  // Mock benchmark configuration
  const benchmarkConfig: Partial<BenchmarkConfig> = {
    trace_id: 'governance-demo-' + Date.now(),
    suite: ['codesearch'],
    systems: ['lex', '+symbols', '+symbols+semantic'],
    slices: 'SMOKE_DEFAULT',
    seeds: 1
  };
  
  console.log('📊 Executing benchmark with governance validation...');
  const result = await orchestrator.runCompleteBenchmark(benchmarkConfig, 'full');
  
  console.log('📋 Governance Results:');
  if (result.governance_validation) {
    console.log(`   🔒 Fingerprint version: CBU v${result.governance_validation.fingerprint.cbu_coeff_v}, Contract v${result.governance_validation.fingerprint.contract_v}`);
    console.log(`   🔬 Validation status: ${result.governance_validation.validation_results.overallPassed ? '✅ PASSED' : '❌ FAILED'}`);
    console.log(`   🔍 Red-team status: ${result.governance_validation.redteam_results.overallPassed ? '✅ PASSED' : '❌ FAILED'}`);
    console.log(`   📦 Audit bundle: ${result.governance_validation.audit_bundle_path}`);
  }
  
  await orchestrator.cleanup();
}

/**
 * Demonstrate audit bundle generation
 */
async function runAuditBundleDemo(outputDir: string) {
  console.log('📦 Audit Bundle Generation Demo');
  console.log('===============================');
  
  const auditGenerator = new AuditBundleGenerator({
    outputDir,
    includeSourceCode: true,
    includeDatasets: true,
    includeModels: false,
    includeDependencies: true,
    compressionLevel: 6
  });
  
  const governanceSystem = new BenchmarkGovernanceSystem(outputDir);
  
  // Generate sample fingerprint
  const fingerprint = await governanceSystem.generateVersionedFingerprint(
    { trace_id: 'audit-demo-' + Date.now() },
    [42, 123, 456],
    { gamma: 1.0, delta: 0.5, beta: 0.3 }
  );
  
  // Mock benchmark results
  const mockResults = [
    {
      trace_id: fingerprint.bench_schema,
      metrics: {
        ndcg_at_10: 0.75,
        recall_at_50: 0.82,
        cbu_score: 0.78,
        ece_score: 0.04
      }
    }
  ];
  
  // Mock ground truth data
  const mockGroundTruth = [
    {
      id: 'sample-1',
      query: 'find TypeScript interfaces',
      expected_results: [
        { file: 'src/types.ts', line: 10, col: 0, relevance_score: 0.9 }
      ]
    }
  ];
  
  console.log('🔧 Generating comprehensive audit bundle...');
  const bundle = await auditGenerator.generateAuditBundle(
    fingerprint,
    mockResults,
    mockGroundTruth
  );
  
  console.log(`📦 Audit bundle created: ${bundle.bundlePath}`);
  console.log(`🔐 Verification hash: ${bundle.verificationHash}`);
  console.log(`📄 Manifest contains ${bundle.manifest.files.length} files`);
  
  // Validate the bundle
  console.log('🔍 Validating audit bundle integrity...');
  const validation = await auditGenerator.validateAuditBundle(bundle.bundlePath);
  console.log(`✅ Bundle validation: ${validation.isValid ? 'PASSED' : 'FAILED'}`);
  
  if (!validation.isValid) {
    console.log('❌ Validation errors:');
    validation.errors.forEach(error => console.log(`   - ${error}`));
  }
}

/**
 * Demonstrate red-team validation suite
 */
async function runRedTeamDemo(outputDir: string) {
  console.log('🔍 Red-Team Validation Suite Demo');
  console.log('=================================');
  
  const redteamSuite = new RedTeamValidationSuite({
    outputDir,
    leakSentinelEnabled: true,
    verbosityDopingEnabled: true,
    tamperDetectionEnabled: true,
    ngramOverlapThreshold: 0.1,
    weeklySchedule: false
  });
  
  const governanceSystem = new BenchmarkGovernanceSystem(outputDir);
  const fingerprint = await governanceSystem.generateVersionedFingerprint(
    { trace_id: 'redteam-demo-' + Date.now() },
    [42, 123],
    { gamma: 1.0, delta: 0.5, beta: 0.3 }
  );
  
  // Mock candidate pool with potential leaks
  const candidatePool = [
    {
      id: 'candidate-1',
      text: 'function calculateSum(a, b) { return a + b; }',
      teacherRationale: 'This function performs basic arithmetic addition'
    },
    {
      id: 'candidate-2',
      text: 'class DataProcessor { process(data) { return data.filter(item => item.valid); } }',
      teacherRationale: 'This class filters valid data items from input'
    },
    {
      id: 'candidate-3',
      text: 'function calculateSum(a, b) { return a + b; }', // Potential leak - same as candidate-1
      teacherRationale: 'This function performs basic arithmetic addition'
    }
  ];
  
  // Mock test queries for verbosity doping
  const testQueries = [
    { id: 'query-1', query: 'find sum calculation functions', expectedCBU: 0.85 },
    { id: 'query-2', query: 'data processing classes', expectedCBU: 0.78 },
    { id: 'query-3', query: 'filter operations', expectedCBU: 0.82 }
  ];
  
  console.log('🔍 Running comprehensive red-team validation...');
  const results = await redteamSuite.runCompleteValidation(
    fingerprint,
    candidatePool,
    testQueries
  );
  
  console.log('📋 Red-Team Results:');
  console.log(`   Overall Status: ${results.overallPassed ? '✅ PASSED' : '❌ FAILED'}`);
  console.log(`   Total Tests: ${results.summary.totalTests}`);
  console.log(`   Passed: ${results.summary.passedTests}`);
  console.log(`   Failed: ${results.summary.failedTests}`);
  
  if (results.summary.criticalFailures.length > 0) {
    console.log('🚨 Critical Failures:');
    results.summary.criticalFailures.forEach(failure => {
      console.log(`   - ${failure}`);
    });
  }
  
  // Print detailed test results
  for (const testResult of results.testResults) {
    console.log(`   🧪 ${testResult.testName}: ${testResult.passed ? '✅ PASSED' : '❌ FAILED'}`);
    if (!testResult.passed && testResult.violations.length > 0) {
      console.log(`      Violations: ${testResult.violations.length}`);
    }
  }
}

/**
 * Demonstrate statistical power analysis
 */
async function runStatisticalPowerDemo(governanceSystem: BenchmarkGovernanceSystem) {
  console.log('📊 Statistical Power Analysis Demo');
  console.log('=================================');
  
  const { StatisticalPowerAnalyzer } = await import('../../benchmarks/src/governance-system.js');
  const powerAnalyzer = new StatisticalPowerAnalyzer();
  
  // Demo: Calculate required sample sizes for different scenarios
  const scenarios = [
    { name: 'Small Effect', baseline: 0.65, mde: 0.02, alpha: 0.05, power: 0.8 },
    { name: 'Medium Effect', baseline: 0.65, mde: 0.05, alpha: 0.05, power: 0.8 },
    { name: 'Large Effect', baseline: 0.65, mde: 0.10, alpha: 0.05, power: 0.8 },
    { name: 'High Power', baseline: 0.65, mde: 0.03, alpha: 0.05, power: 0.9 }
  ];
  
  console.log('📈 Sample Size Requirements:');
  console.log('   Scenario          | Baseline | MDE   | α     | Power | Required N');
  console.log('   ------------------|----------|-------|-------|-------|----------');
  
  for (const scenario of scenarios) {
    const requiredN = powerAnalyzer.calculateSampleSizeForProportion(
      scenario.baseline,
      scenario.mde,
      scenario.alpha,
      scenario.power
    );
    
    console.log(`   ${scenario.name.padEnd(17)} | ${scenario.baseline.toFixed(2).padStart(8)} | ${scenario.mde.toFixed(2).padStart(5)} | ${scenario.alpha.toFixed(2).padStart(5)} | ${scenario.power.toFixed(1).padStart(5)} | ${requiredN.toString().padStart(9)}`);
  }
  
  // Demo: Validate power for mock slice results
  console.log();
  console.log('🔍 Power Validation for Mock Slices:');
  
  const mockSlices = [
    { name: 'TypeScript', actualN: 150, requiredN: 120, baseline: 0.72, mde: 0.03 },
    { name: 'Python', actualN: 95, requiredN: 120, baseline: 0.68, mde: 0.03 },
    { name: 'JavaScript', actualN: 180, requiredN: 140, baseline: 0.75, mde: 0.025 }
  ];
  
  for (const slice of mockSlices) {
    const validation = powerAnalyzer.validatePowerRequirements(
      slice.actualN,
      slice.requiredN,
      slice.name
    );
    
    console.log(`   ${slice.name}: ${validation.isPowered ? '✅ Powered' : '❌ Under-powered'} (${slice.actualN}/${slice.requiredN})`);
    if (!validation.isPowered) {
      console.log(`      ${validation.recommendation}`);
    }
  }
}

/**
 * Demonstrate calibration monitoring
 */
async function runCalibrationDemo(governanceSystem: BenchmarkGovernanceSystem) {
  console.log('🎯 Calibration Monitoring Demo');
  console.log('==============================');
  
  const { CalibrationMonitor } = await import('../../benchmarks/src/governance-system.js');
  const calibrationMonitor = new CalibrationMonitor();
  
  // Generate mock prediction data
  const mockPredictions = [];
  for (let i = 0; i < 100; i++) {
    const confidence = Math.random();
    // Simulate well-calibrated predictions with some noise
    const isCorrect = Math.random() < (confidence * 0.8 + 0.1); // Slightly under-confident
    mockPredictions.push({ confidence, isCorrect });
  }
  
  console.log('📊 Calculating calibration metrics...');
  
  // Calculate ECE
  const eceResult = calibrationMonitor.calculateECE(mockPredictions);
  console.log(`   ECE: ${eceResult.ece.toFixed(4)} (target: ≤ 0.05)`);
  console.log(`   MCE: ${eceResult.mce.toFixed(4)}`);
  
  // Calculate Brier score
  const brierPredictions = mockPredictions.map(p => ({
    probability: p.confidence,
    outcome: p.isCorrect
  }));
  
  const brierResult = calibrationMonitor.calculateBrierScore(brierPredictions);
  console.log(`   Brier Score: ${brierResult.brierScore.toFixed(4)} (lower is better)`);
  console.log(`   Reliability: ${brierResult.reliability.toFixed(4)}`);
  console.log(`   Uncertainty: ${brierResult.uncertainty.toFixed(4)}`);
  
  // Mock calibration slope calculation
  const logOddsPredictions = mockPredictions.map(p => ({
    logOdds: Math.log(p.confidence / (1 - p.confidence + 1e-8)),
    isCorrect: p.isCorrect
  }));
  
  const slopeResult = calibrationMonitor.calculateCalibrationSlope(logOddsPredictions);
  console.log(`   Slope: ${slopeResult.slope.toFixed(4)} (target: ∈ [0.9, 1.1])`);
  console.log(`   Intercept: ${slopeResult.intercept.toFixed(4)}`);
  console.log(`   R²: ${slopeResult.goodnessOfFit.toFixed(4)}`);
  
  // Validate calibration gates
  const fingerprint = await governanceSystem.generateVersionedFingerprint(
    { trace_id: 'calibration-demo' },
    [42],
    { gamma: 1.0, delta: 0.5, beta: 0.3 }
  );
  
  const gateValidation = calibrationMonitor.validateCalibrationGates(
    eceResult.ece,
    slopeResult.slope,
    Math.abs(slopeResult.intercept - 0), // Delta from expected intercept of 0
    fingerprint.calibration_gates
  );
  
  console.log();
  console.log('🚪 Calibration Gate Validation:');
  console.log(`   Overall: ${gateValidation.overallPass ? '✅ PASSED' : '❌ FAILED'}`);
  console.log(`   ECE: ${gateValidation.ecePass ? '✅' : '❌'} (${eceResult.ece.toFixed(4)} ≤ ${fingerprint.calibration_gates.ece_max})`);
  console.log(`   Slope: ${gateValidation.slopePass ? '✅' : '❌'} (${slopeResult.slope.toFixed(4)} ∈ [${fingerprint.calibration_gates.slope_range[0]}, ${fingerprint.calibration_gates.slope_range[1]}])`);
  console.log(`   Intercept: ${gateValidation.interceptPass ? '✅' : '❌'} (|${slopeResult.intercept.toFixed(4)}| ≤ ${fingerprint.calibration_gates.intercept_delta_max})`);
  
  if (gateValidation.violations.length > 0) {
    console.log('   Violations:');
    gateValidation.violations.forEach(violation => {
      console.log(`     - ${violation}`);
    });
  }
}

/**
 * Demonstrate multiple testing correction
 */
async function runMultipleTestingDemo(governanceSystem: BenchmarkGovernanceSystem) {
  console.log('🔢 Multiple Testing Correction Demo');
  console.log('==================================');
  
  const { MultipleTestingCorrector } = await import('../../benchmarks/src/governance-system.js');
  const multipleTestingCorrector = new MultipleTestingCorrector();
  
  // Generate mock slice results with some significant regressions
  const mockSliceResults = [
    { sliceName: 'TypeScript', baselineMetric: 0.75, treatmentMetric: 0.77, pValue: 0.02 },
    { sliceName: 'Python', baselineMetric: 0.72, treatmentMetric: 0.69, pValue: 0.01 }, // Regression
    { sliceName: 'JavaScript', baselineMetric: 0.68, treatmentMetric: 0.70, pValue: 0.08 },
    { sliceName: 'Java', baselineMetric: 0.71, treatmentMetric: 0.68, pValue: 0.03 }, // Regression
    { sliceName: 'Go', baselineMetric: 0.69, treatmentMetric: 0.71, pValue: 0.12 },
    { sliceName: 'Rust', baselineMetric: 0.73, treatmentMetric: 0.70, pValue: 0.04 } // Minor regression
  ];
  
  console.log('📊 Original p-values:');
  mockSliceResults.forEach(result => {
    const regression = (result.baselineMetric - result.treatmentMetric) * 100;
    console.log(`   ${result.sliceName}: p=${result.pValue.toFixed(3)}, Δ=${regression.toFixed(1)}pp`);
  });
  
  // Apply Holm correction
  console.log();
  console.log('🧮 Holm Correction Results:');
  const pValues = mockSliceResults.map(r => r.pValue);
  const holmResults = multipleTestingCorrector.holmCorrection(pValues, 0.05);
  
  console.log('   Slice         | Original p | Adjusted p | Critical α | Significant');
  console.log('   --------------|------------|------------|------------|------------');
  
  mockSliceResults.forEach((result, i) => {
    const originalP = result.pValue;
    const adjustedP = holmResults.adjustedPValues[i];
    const criticalAlpha = holmResults.criticalValues[i];
    const significant = holmResults.rejectedHypotheses[i];
    
    console.log(`   ${result.sliceName.padEnd(13)} | ${originalP.toFixed(3).padStart(10)} | ${adjustedP.toFixed(3).padStart(10)} | ${criticalAlpha.toFixed(3).padStart(10)} | ${significant ? '    ✅' : '    ❌'}`);
  });
  
  // Validate slice regressions with correction
  console.log();
  console.log('🎯 Slice Regression Validation:');
  
  const regressionValidation = multipleTestingCorrector.validateSliceRegressions(
    mockSliceResults,
    2.0, // Max 2pp regression allowed
    'holm',
    0.05
  );
  
  console.log(`   Overall: ${regressionValidation.overallPass ? '✅ PASSED' : '❌ FAILED'}`);
  console.log(`   Family-wise error controlled: ${regressionValidation.correctionSummary.familyWiseErrorControlled ? '✅' : '❌'}`);
  console.log(`   Significant regressions: ${regressionValidation.correctionSummary.significantRegressions}/${regressionValidation.correctionSummary.totalTests}`);
  
  if (!regressionValidation.overallPass) {
    console.log('   Significant violations:');
    regressionValidation.sliceViolations
      .filter(v => v.isSignificantRegression)
      .forEach(violation => {
        console.log(`     - ${violation.sliceName}: ${violation.regressionPP.toFixed(1)}pp regression (p_adj=${violation.adjustedPValue.toFixed(3)})`);
      });
  }
}

/**
 * Demonstrate versioned fingerprint generation
 */
async function runVersionedFingerprintDemo(governanceSystem: BenchmarkGovernanceSystem) {
  console.log('🔐 Versioned Fingerprint Demo');
  console.log('=============================');
  
  const mockConfig = {
    trace_id: 'fingerprint-demo-' + Date.now(),
    suite: ['codesearch'],
    systems: ['lex', '+symbols', '+symbols+semantic'],
    slices: 'SMOKE_DEFAULT'
  };
  
  console.log('🔧 Generating versioned fingerprint...');
  const fingerprint = await governanceSystem.generateVersionedFingerprint(
    mockConfig,
    [42, 123, 456],
    { gamma: 1.0, delta: 0.5, beta: 0.3 }
  );
  
  console.log('📋 Fingerprint Details:');
  console.log(`   CBU Coefficient Version: v${fingerprint.cbu_coeff_v}`);
  console.log(`   Contract Version: v${fingerprint.contract_v}`);
  console.log(`   Pool Version: ${fingerprint.pool_v.substring(0, 8)}...`);
  console.log(`   Oracle Version: ${fingerprint.oracle_v.substring(0, 8)}...`);
  console.log();
  
  console.log('⚙️ Statistical Power Configuration:');
  console.log(`   Alpha (Type I error): ${fingerprint.mde_config.alpha}`);
  console.log(`   Power (1-β): ${fingerprint.mde_config.power}`);
  console.log(`   MDE Threshold: ${fingerprint.mde_config.mde_threshold}`);
  console.log(`   Cluster Unit: ${fingerprint.mde_config.cluster_unit}`);
  console.log();
  
  console.log('🎯 Calibration Gates:');
  console.log(`   ECE Max: ${fingerprint.calibration_gates.ece_max}`);
  console.log(`   Slope Range: [${fingerprint.calibration_gates.slope_range[0]}, ${fingerprint.calibration_gates.slope_range[1]}]`);
  console.log(`   Intercept Delta Max: ${fingerprint.calibration_gates.intercept_delta_max}`);
  console.log(`   Brier Tracking: ${fingerprint.calibration_gates.brier_tracking ? '✅' : '❌'}`);
  console.log();
  
  console.log('🔄 Bootstrap Configuration:');
  console.log(`   Method: ${fingerprint.bootstrap_config.method}`);
  console.log(`   Cluster By: ${fingerprint.bootstrap_config.cluster_by}`);
  console.log(`   Default Samples: ${fingerprint.bootstrap_config.b_default}`);
  console.log(`   Threshold Samples: ${fingerprint.bootstrap_config.b_threshold}`);
  console.log();
  
  console.log('🔢 Multiple Testing:');
  console.log(`   Method: ${fingerprint.multiple_testing.method}`);
  console.log(`   Max Slice Regression: ${fingerprint.multiple_testing.slice_regression_max_pp}pp`);
  console.log(`   Family-wise Error Rate: ${fingerprint.multiple_testing.family_wise_error_rate}`);
  console.log();
  
  console.log('🔍 Red-team Configuration:');
  console.log(`   Leak Sentinel: ${fingerprint.redteam_config.leak_sentinel_enabled ? '✅' : '❌'}`);
  console.log(`   Verbosity Doping: ${fingerprint.redteam_config.verbosity_doping_enabled ? '✅' : '❌'}`);
  console.log(`   Tamper Detection: ${fingerprint.redteam_config.tamper_detection_enabled ? '✅' : '❌'}`);
  console.log(`   N-gram Overlap Threshold: ${fingerprint.redteam_config.ngram_overlap_threshold}`);
  
  // Save fingerprint for reference
  const fingerprintPath = path.join(options.outputDir, 'demo-fingerprint.json');
  await fs.writeFile(fingerprintPath, JSON.stringify(fingerprint, null, 2));
  console.log();
  console.log(`💾 Fingerprint saved to: ${fingerprintPath}`);
}

// Error handling
process.on('unhandledRejection', (reason, promise) => {
  console.error('❌ Unhandled Rejection at:', promise, 'reason:', reason);
  process.exit(1);
});

process.on('uncaughtException', (error) => {
  console.error('❌ Uncaught Exception:', error);
  process.exit(1);
});

// Run main function
main().catch((error) => {
  console.error('❌ Demo failed:', error);
  process.exit(1);
});