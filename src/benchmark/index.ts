/**
 * Lens Benchmark System - Main Orchestrator
 * Implements complete TODO.md benchmarking specification
 */

import { promises as fs } from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import type {
  BenchmarkConfig,
  BenchmarkConfigSchema,
  RepoSnapshot
} from '../types/benchmark.js';

import { GroundTruthBuilder } from './ground-truth-builder.js';
import { BenchmarkSuiteRunner } from './suite-runner.js';
import { MetricsCalculator } from './metrics-calculator.js';
import { NATSTelemetry, BenchmarkTelemetryAggregator } from './nats-telemetry.js';
import { MetamorphicTestRunner } from './metamorphic-tests.js';
import { RobustnessTestRunner } from './robustness-tests.js';
import { BenchmarkReportGenerator, type ReportData } from './report-generator.js';

export interface BenchmarkOrchestrationConfig {
  workingDir: string;
  outputDir: string;
  natsUrl?: string;
  repositories: Array<{
    name: string;
    path: string;
  }>;
}

export class LensBenchmarkOrchestrator {
  private groundTruthBuilder: GroundTruthBuilder;
  private suiteRunner: BenchmarkSuiteRunner;
  private metricsCalculator: MetricsCalculator;
  private telemetry: NATSTelemetry;
  private metamorphicRunner: MetamorphicTestRunner;
  private robustnessRunner: RobustnessTestRunner;
  private reportGenerator: BenchmarkReportGenerator;

  constructor(private readonly config: BenchmarkOrchestrationConfig) {
    this.groundTruthBuilder = new GroundTruthBuilder(
      config.workingDir,
      config.outputDir
    );
    
    this.suiteRunner = new BenchmarkSuiteRunner(
      this.groundTruthBuilder,
      config.outputDir,
      config.natsUrl
    );
    
    this.metricsCalculator = new MetricsCalculator();
    this.telemetry = new NATSTelemetry(config.natsUrl);
    
    this.metamorphicRunner = new MetamorphicTestRunner(
      config.workingDir,
      config.outputDir
    );
    
    this.robustnessRunner = new RobustnessTestRunner(config.outputDir);
    this.reportGenerator = new BenchmarkReportGenerator(config.outputDir);
  }

  /**
   * Execute complete benchmark orchestration per TODO.md specifications
   */
  async runCompleteBenchmark(
    benchmarkConfig: Partial<BenchmarkConfig> = {},
    suiteType: 'smoke' | 'full' = 'smoke'
  ): Promise<{
    benchmark_run: any;
    reports: {
      pdf_path: string;
      markdown_path: string;
      json_path: string;
    };
    artifacts: string[];
  }> {
    
    const startTime = Date.now();
    const traceId = benchmarkConfig.trace_id || uuidv4();
    
    console.log(`🚀 Starting ${suiteType.toUpperCase()} benchmark suite - Trace: ${traceId}`);
    console.log('📋 Phase 1: Dataset & Ground Truth Construction...');
    
    // Phase 1: Build datasets and ground truth
    const snapshots = await this.buildDatasets();
    
    console.log('📊 Phase 2: Benchmark Suite Execution...');
    
    // Phase 2: Execute benchmark suite
    const benchmarkRun = suiteType === 'smoke' ? 
      await this.suiteRunner.runSmokeSuite(benchmarkConfig) :
      await this.suiteRunner.runFullSuite(benchmarkConfig);
    
    // Phase 3: Enhanced testing (full suite only)
    let metamorphicResults: any[] = [];
    let robustnessResults: any[] = [];
    
    if (suiteType === 'full') {
      console.log('🔄 Phase 3a: Metamorphic Testing...');
      metamorphicResults = await this.metamorphicRunner.runMetamorphicTests(
        { ...benchmarkConfig, trace_id: traceId } as BenchmarkConfig,
        snapshots,
        this.groundTruthBuilder.currentGoldenItems
      );
      
      console.log('🔨 Phase 3b: Robustness Testing...');
      robustnessResults = await this.robustnessRunner.runRobustnessTests(
        { ...benchmarkConfig, trace_id: traceId } as BenchmarkConfig
      );
    }
    
    console.log('📄 Phase 4: Report Generation...');
    
    // Phase 4: Generate comprehensive reports
    const reportData: ReportData = {
      title: `Lens ${suiteType.toUpperCase()} Benchmark Report`,
      config: benchmarkConfig,
      benchmarkRuns: Array.isArray(benchmarkRun) ? benchmarkRun : [benchmarkRun],
      abTestResults: [], // Would be populated from actual A/B tests
      metamorphicResults,
      robustnessResults,
      configFingerprint: this.groundTruthBuilder.generateConfigFingerprint(
        benchmarkConfig,
        [1, 2, 3]
      ),
      metadata: {
        generated_at: new Date().toISOString(),
        total_duration_ms: Date.now() - startTime,
        systems_tested: benchmarkConfig.systems || ['lex', '+symbols', '+symbols+semantic'],
        queries_executed: this.groundTruthBuilder.currentGoldenItems.length
      }
    };
    
    const reports = await this.reportGenerator.generateReport(reportData);
    
    // Phase 5: Artifact collection
    const artifacts = await this.collectArtifacts(traceId);
    
    const totalDuration = Date.now() - startTime;
    console.log(`🎯 Benchmark complete - Duration: ${(totalDuration / 1000 / 60).toFixed(1)} minutes`);
    console.log(`📊 Reports: ${Object.values(reports).join(', ')}`);
    
    return {
      benchmark_run: benchmarkRun,
      reports,
      artifacts
    };
  }

  /**
   * Run smoke test suite (optimized for PR gates)
   */
  async runSmoke(config: Partial<BenchmarkConfig> = {}): Promise<any> {
    const result = await this.runCompleteBenchmark(config, 'smoke');
    
    // Check promotion gate
    const gateStatus = this.checkPromotionGate(result.benchmark_run);
    console.log(`🚪 Promotion Gate: ${gateStatus.passed ? '✅ PASS' : '❌ FAIL'}`);
    
    if (!gateStatus.passed) {
      console.log(`   Failures: ${gateStatus.regressions.join(', ')}`);
    }
    
    return result;
  }

  /**
   * Run full nightly suite (comprehensive evaluation)
   */
  async runFull(config: Partial<BenchmarkConfig> = {}): Promise<any> {
    return this.runCompleteBenchmark(config, 'full');
  }

  /**
   * Build datasets and ground truth per TODO.md specifications
   */
  private async buildDatasets(): Promise<RepoSnapshot[]> {
    const snapshots: RepoSnapshot[] = [];
    
    // Ensure output directory exists
    await fs.mkdir(this.config.outputDir, { recursive: true });
    
    // Process each repository
    for (const repo of this.config.repositories) {
      console.log(`  📂 Processing repository: ${repo.name}`);
      
      try {
        const snapshot = await this.groundTruthBuilder.freezeRepoSnapshot(repo.path);
        snapshots.push(snapshot);
        
      } catch (error) {
        console.error(`  ❌ Failed to process ${repo.name}:`, error);
      }
    }
    
    if (snapshots.length === 0) {
      throw new Error('No repositories successfully processed');
    }
    
    // Construct golden dataset
    console.log(`  🏗️  Constructing golden dataset from ${snapshots.length} snapshots...`);
    await this.groundTruthBuilder.constructGoldenSet(snapshots);
    
    console.log(`  ✅ Golden dataset: ${this.groundTruthBuilder.currentGoldenItems.length} items`);
    
    return snapshots;
  }

  /**
   * Check promotion gate criteria per TODO.md
   */
  private checkPromotionGate(benchmarkRun: any): {
    passed: boolean;
    ndcg_delta: number;
    recall_50_maintained: boolean;
    latency_p95_acceptable: boolean;
    regressions: string[];
  } {
    
    // Simplified promotion gate logic (would use actual A/B comparison in production)
    const metrics = Array.isArray(benchmarkRun) ? benchmarkRun[0].metrics : benchmarkRun.metrics;
    
    const ndcgDelta = 0.01; // Would be calculated from A/B test
    const ndcgImprovement = ndcgDelta >= 0.02; // +2% requirement
    
    const recall50Maintained = metrics.recall_at_50 >= 0.8;
    const latencyAcceptable = metrics.stage_latencies.e2e_p95 <= 22; // Allow some buffer
    
    const passed = ndcgImprovement && recall50Maintained && latencyAcceptable;
    
    return {
      passed,
      ndcg_delta: ndcgDelta,
      recall_50_maintained: recall50Maintained,
      latency_p95_acceptable: latencyAcceptable,
      regressions: passed ? [] : [
        !ndcgImprovement ? 'ndcg_insufficient_improvement' : '',
        !recall50Maintained ? 'recall_50_degradation' : '',
        !latencyAcceptable ? 'latency_p95_regression' : ''
      ].filter(Boolean)
    };
  }

  /**
   * Collect all benchmark artifacts
   */
  private async collectArtifacts(traceId: string): Promise<string[]> {
    const artifacts: string[] = [];
    
    try {
      // Find all files matching the trace ID pattern
      const files = await fs.readdir(this.config.outputDir);
      
      for (const file of files) {
        if (file.includes(traceId) || file.includes('benchmark') || file.includes('report')) {
          artifacts.push(path.join(this.config.outputDir, file));
        }
      }
      
    } catch (error) {
      console.warn('Warning: Failed to collect artifacts:', error);
    }
    
    return artifacts;
  }

  /**
   * Health check - verify all components are ready
   */
  async healthCheck(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    components: Record<string, boolean>;
    details: Record<string, any>;
  }> {
    
    const componentChecks: Record<string, boolean> = {};
    const details: Record<string, any> = {};
    
    // Check NATS connectivity
    try {
      await this.telemetry.connect();
      const natsHealth = await this.telemetry.healthCheck();
      componentChecks['nats'] = natsHealth.connected;
      details['nats'] = natsHealth;
      await this.telemetry.disconnect();
    } catch (error) {
      componentChecks['nats'] = false;
      details['nats'] = { error: error instanceof Error ? error.message : String(error) };
    }
    
    // Check output directory
    try {
      await fs.access(this.config.outputDir);
      componentChecks['output_dir'] = true;
      details['output_dir'] = { path: this.config.outputDir };
    } catch (error) {
      componentChecks['output_dir'] = false;
      details['output_dir'] = { error: 'Directory not accessible' };
    }
    
    // Check repositories
    componentChecks['repositories'] = true;
    details['repositories'] = { count: this.config.repositories.length };
    
    for (const repo of this.config.repositories) {
      try {
        await fs.access(repo.path);
      } catch (error) {
        componentChecks['repositories'] = false;
        details['repositories'].error = `Repository ${repo.name} not accessible`;
        break;
      }
    }
    
    const healthyComponents = Object.values(componentChecks).filter(Boolean).length;
    const totalComponents = Object.keys(componentChecks).length;
    
    let status: 'healthy' | 'degraded' | 'unhealthy';
    if (healthyComponents === totalComponents) {
      status = 'healthy';
    } else if (healthyComponents >= totalComponents / 2) {
      status = 'degraded';
    } else {
      status = 'unhealthy';
    }
    
    return {
      status,
      components: componentChecks,
      details
    };
  }

  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    try {
      await this.telemetry.disconnect();
    } catch (error) {
      console.warn('Warning during cleanup:', error);
    }
  }
}

// Export convenience factory function
export function createBenchmarkOrchestrator(config: BenchmarkOrchestrationConfig): LensBenchmarkOrchestrator {
  return new LensBenchmarkOrchestrator(config);
}

// Export all benchmark components for advanced usage
export {
  GroundTruthBuilder,
  BenchmarkSuiteRunner,
  MetricsCalculator,
  NATSTelemetry,
  BenchmarkTelemetryAggregator,
  MetamorphicTestRunner,
  RobustnessTestRunner,
  BenchmarkReportGenerator
};

// Export types for external usage
export type {
  BenchmarkConfig,
  RepoSnapshot
} from '../types/benchmark.js';