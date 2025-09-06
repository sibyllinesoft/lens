/**
 * CI Integration for Phase C Hardening
 * Integrates hardening checks into CI/CD pipeline with automated failure handling
 */

import { promises as fs } from 'fs';
import path from 'path';
import { BenchmarkSuiteRunner } from './suite-runner.js';
import { GroundTruthBuilder } from './ground-truth-builder.js';
import { createDefaultHardeningConfig, type HardeningConfig } from './phase-c-hardening.js';
import type { BenchmarkConfig } from '../types/benchmark.js';

export interface CIHardeningConfig {
  // CI Environment
  ci_mode: 'pr' | 'nightly' | 'release';
  fail_fast: boolean;
  max_execution_time_minutes: number;
  
  // Reporting
  generate_artifacts: boolean;
  upload_to_s3: boolean;
  slack_webhook_url?: string;
  
  // Quality Gates
  quality_gates: {
    enforce_tripwires: boolean;
    enforce_slice_gates: boolean;
    min_hardening_score: number; // 0-100
    max_degradation_percent: number;
  };
  
  // Retry Policy
  retry_policy: {
    enabled: boolean;
    max_retries: number;
    backoff_multiplier: number;
  };
}

export interface CIHardeningResult {
  success: boolean;
  execution_time_ms: number;
  hardening_score: number;
  quality_gate_results: {
    tripwires_passed: number;
    tripwires_total: number;
    slice_gates_passed: number;
    slice_gates_total: number;
  };
  artifacts: {
    report_path: string;
    plots_directory: string;
    raw_data_path: string;
  };
  failure_summary?: {
    failing_tripwires: string[];
    failing_slices: string[];
    degradation_percent: number;
    recommendations: string[];
  };
}

export class CIHardeningOrchestrator {
  private suiteRunner: BenchmarkSuiteRunner;
  
  constructor(
    private readonly outputDir: string,
    private readonly natsUrl: string = 'nats://localhost:4222'
  ) {
    const groundTruthBuilder = new GroundTruthBuilder(process.cwd(), outputDir);
    this.suiteRunner = new BenchmarkSuiteRunner(groundTruthBuilder, outputDir, natsUrl);
  }

  /**
   * Execute Phase C hardening in CI environment
   */
  async executeInCI(
    ciConfig: CIHardeningConfig,
    benchmarkConfig: Partial<BenchmarkConfig> = {}
  ): Promise<CIHardeningResult> {
    
    const startTime = Date.now();
    console.log(`🔄 Starting CI Phase C Hardening (${ciConfig.ci_mode} mode)`);
    
    try {
      // Set timeout for CI execution
      const timeoutPromise = this.createTimeoutPromise(ciConfig.max_execution_time_minutes);
      const hardeningPromise = this.executeHardeningWithRetry(ciConfig, benchmarkConfig);
      
      const result = await Promise.race([hardeningPromise, timeoutPromise]);
      
      if (result.timedOut) {
        throw new Error(`CI hardening timed out after ${ciConfig.max_execution_time_minutes} minutes`);
      }
      
      const executionTime = Date.now() - startTime;
      const ciResult = await this.processHardeningResults(result, executionTime, ciConfig);
      
      // Handle CI notifications
      await this.sendCINotifications(ciResult, ciConfig);
      
      // Handle CI artifacts
      if (ciConfig.generate_artifacts) {
        await this.handleCIArtifacts(ciResult, ciConfig);
      }
      
      // Final CI decision
      if (!ciResult.success && ciConfig.fail_fast) {
        console.error('❌ CI Phase C Hardening FAILED - Failing fast');
        process.exit(1);
      }
      
      console.log(`🎯 CI Phase C Hardening completed: ${ciResult.success ? 'SUCCESS' : 'FAILED'} (${(executionTime/1000).toFixed(1)}s)`);
      
      return ciResult;
      
    } catch (error) {
      const executionTime = Date.now() - startTime;
      
      console.error('💥 CI Phase C Hardening encountered critical error:', error);
      
      const failureResult: CIHardeningResult = {
        success: false,
        execution_time_ms: executionTime,
        hardening_score: 0,
        quality_gate_results: {
          tripwires_passed: 0,
          tripwires_total: 0,
          slice_gates_passed: 0,
          slice_gates_total: 0
        },
        artifacts: {
          report_path: '',
          plots_directory: '',
          raw_data_path: ''
        },
        failure_summary: {
          failing_tripwires: [],
          failing_slices: [],
          degradation_percent: 100,
          recommendations: [`Critical CI error: ${error instanceof Error ? error.message : String(error)}`]
        }
      };
      
      await this.sendCINotifications(failureResult, ciConfig);
      
      if (ciConfig.fail_fast) {
        process.exit(1);
      }
      
      return failureResult;
    }
  }

  /**
   * Execute hardening with retry policy
   */
  private async executeHardeningWithRetry(
    ciConfig: CIHardeningConfig,
    benchmarkConfig: Partial<BenchmarkConfig>
  ): Promise<any> {
    
    let lastError: Error | null = null;
    let attempts = 0;
    const maxAttempts = ciConfig.retry_policy.enabled ? ciConfig.retry_policy.max_retries + 1 : 1;
    
    while (attempts < maxAttempts) {
      try {
        console.log(`  Attempt ${attempts + 1}/${maxAttempts}`);
        
        // Create hardening configuration based on CI mode
        const hardeningConfig = this.createCIModeHardeningConfig(ciConfig, benchmarkConfig);
        
        // Execute hardening
        const result = await this.suiteRunner.runPhaseCHardening(hardeningConfig);
        
        // Success - return immediately
        return result;
        
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        attempts++;
        
        if (attempts < maxAttempts) {
          const backoffMs = Math.pow(ciConfig.retry_policy.backoff_multiplier, attempts) * 1000;
          console.warn(`⚠️ Attempt ${attempts} failed, retrying in ${backoffMs}ms:`, lastError.message);
          await this.sleep(backoffMs);
        }
      }
    }
    
    throw lastError || new Error('Unknown hardening failure');
  }

  /**
   * Create hardening configuration based on CI mode
   */
  private createCIModeHardeningConfig(
    ciConfig: CIHardeningConfig,
    benchmarkConfig: Partial<BenchmarkConfig>
  ): HardeningConfig {
    
    // Create a complete BenchmarkConfig with defaults
    const fullBenchmarkConfig: BenchmarkConfig = {
      trace_id: benchmarkConfig.trace_id ?? crypto.randomUUID(),
      suite: benchmarkConfig.suite ?? ['codesearch'],
      systems: benchmarkConfig.systems ?? ['lex'],
      slices: benchmarkConfig.slices ?? 'SMOKE_DEFAULT',
      seeds: benchmarkConfig.seeds ?? 1,
      cache_mode: benchmarkConfig.cache_mode ?? 'warm',
      robustness: benchmarkConfig.robustness ?? false,
      metamorphic: benchmarkConfig.metamorphic ?? false,
      k_candidates: benchmarkConfig.k_candidates ?? 200,
      top_n: benchmarkConfig.top_n ?? 50,
      fuzzy: benchmarkConfig.fuzzy ?? 2,
      subtokens: benchmarkConfig.subtokens ?? true,
      semantic_gating: benchmarkConfig.semantic_gating ?? {
        nl_likelihood_threshold: 0.5,
        min_candidates: 10
      },
      latency_budgets: benchmarkConfig.latency_budgets ?? {
        stage_a_ms: 50,
        stage_b_ms: 200,
        stage_c_ms: 500
      }
    };
    
    const baseConfig = createDefaultHardeningConfig(fullBenchmarkConfig);
    
    // CI mode specific configurations
    const ciModeConfigs = {
      'pr': {
        // Lighter configuration for PR checks
        tripwires: {
          min_span_coverage: 0.96,
          recall_convergence_threshold: 0.01,
          lsif_coverage_drop_threshold: 0.1,
          p99_p95_ratio_threshold: 2.5
        },
        per_slice_gates: {
          enabled: true,
          min_recall_at_10: 0.65,
          min_ndcg_at_10: 0.55,
          max_p95_latency_ms: 600
        },
        hard_negatives: {
          enabled: true,
          per_query_count: 3, // Reduced for faster PR checks
          shared_subtoken_min: 2
        }
      },
      'nightly': {
        // Full configuration for nightly builds  
        tripwires: {
          min_span_coverage: 0.98,
          recall_convergence_threshold: 0.005,
          lsif_coverage_drop_threshold: 0.05,
          p99_p95_ratio_threshold: 2.0
        },
        per_slice_gates: {
          enabled: true,
          min_recall_at_10: 0.70,
          min_ndcg_at_10: 0.60,
          max_p95_latency_ms: 500
        },
        hard_negatives: {
          enabled: true,
          per_query_count: 5,
          shared_subtoken_min: 2
        }
      },
      'release': {
        // Strictest configuration for releases
        tripwires: {
          min_span_coverage: 0.99,
          recall_convergence_threshold: 0.003,
          lsif_coverage_drop_threshold: 0.03,
          p99_p95_ratio_threshold: 1.8
        },
        per_slice_gates: {
          enabled: true,
          min_recall_at_10: 0.75,
          min_ndcg_at_10: 0.65,
          max_p95_latency_ms: 450
        },
        hard_negatives: {
          enabled: true,
          per_query_count: 7, // More adversarial testing for releases
          shared_subtoken_min: 1
        }
      }
    };
    
    const ciModeConfig = ciModeConfigs[ciConfig.ci_mode];
    
    return {
      ...baseConfig,
      ...ciModeConfig,
      plots: {
        enabled: ciConfig.generate_artifacts,
        output_dir: path.join(this.outputDir, 'ci-plots'),
        formats: ['png', 'svg']
      }
    };
  }

  /**
   * Process hardening results for CI
   */
  private async processHardeningResults(
    hardeningResult: any,
    executionTime: number,
    ciConfig: CIHardeningConfig
  ): Promise<CIHardeningResult> {
    
    const { hardeningReport } = hardeningResult;
    
    // Calculate hardening score (0-100)
    const hardeningScore = this.calculateHardeningScore(hardeningReport);
    
    // Check quality gates
    const qualityGateResults = {
      tripwires_passed: hardeningReport.tripwire_summary.passed_tripwires,
      tripwires_total: hardeningReport.tripwire_summary.total_tripwires,
      slice_gates_passed: hardeningReport.slice_gate_summary?.passed_slices || 0,
      slice_gates_total: hardeningReport.slice_gate_summary?.total_slices || 0
    };
    
    // Determine overall success
    const success = this.evaluateCISuccess(hardeningReport, hardeningScore, ciConfig);
    
    const ciResult: CIHardeningResult = {
      success,
      execution_time_ms: executionTime,
      hardening_score: hardeningScore,
      quality_gate_results: qualityGateResults,
      artifacts: {
        report_path: hardeningResult.pdfReport || '',
        plots_directory: path.join(this.outputDir, 'hardening-plots'),
        raw_data_path: path.join(this.outputDir, 'phase-c-hardening-report.json')
      }
    };
    
    // Add failure summary if needed
    if (!success) {
      ciResult.failure_summary = {
        failing_tripwires: hardeningReport.tripwire_results
          .filter((t: any) => t.status === 'fail')
          .map((t: any) => t.name),
        failing_slices: hardeningReport.slice_results
          .filter((s: any) => s.gate_status === 'fail')
          .map((s: any) => s.slice_id),
        degradation_percent: hardeningReport.hard_negatives?.impact_on_metrics?.degradation_percent || 0,
        recommendations: hardeningReport.recommendations || []
      };
    }
    
    return ciResult;
  }

  /**
   * Calculate hardening score (0-100)
   */
  private calculateHardeningScore(hardeningReport: any): number {
    const tripwireScore = (hardeningReport.tripwire_summary.passed_tripwires / hardeningReport.tripwire_summary.total_tripwires) * 40;
    
    const sliceScore = hardeningReport.slice_gate_summary ? 
      (hardeningReport.slice_gate_summary.passed_slices / hardeningReport.slice_gate_summary.total_slices) * 30 : 30;
    
    const robustnessScore = Math.max(0, 30 - (hardeningReport.hard_negatives?.impact_on_metrics?.degradation_percent || 0));
    
    return Math.round(tripwireScore + sliceScore + robustnessScore);
  }

  /**
   * Evaluate CI success based on configuration
   */
  private evaluateCISuccess(hardeningReport: any, hardeningScore: number, ciConfig: CIHardeningConfig): boolean {
    
    // Check minimum hardening score
    if (hardeningScore < ciConfig.quality_gates.min_hardening_score) {
      return false;
    }
    
    // Check tripwires if enforced
    if (ciConfig.quality_gates.enforce_tripwires && hardeningReport.tripwire_summary.failed_tripwires > 0) {
      return false;
    }
    
    // Check slice gates if enforced
    if (ciConfig.quality_gates.enforce_slice_gates && 
        hardeningReport.slice_gate_summary?.failed_slices > 0) {
      return false;
    }
    
    // Check degradation limit
    const degradation = hardeningReport.hard_negatives?.impact_on_metrics?.degradation_percent || 0;
    if (degradation > ciConfig.quality_gates.max_degradation_percent) {
      return false;
    }
    
    return true;
  }

  /**
   * Send CI notifications
   */
  private async sendCINotifications(result: CIHardeningResult, ciConfig: CIHardeningConfig): Promise<void> {
    
    // Slack notification
    if (ciConfig.slack_webhook_url) {
      try {
        await this.sendSlackNotification(result, ciConfig);
      } catch (error) {
        console.warn('⚠️ Failed to send Slack notification:', error);
      }
    }
    
    // GitHub Actions output
    if (process.env['GITHUB_ACTIONS']) {
      await this.setGitHubActionsOutput(result, ciConfig);
    }
    
    // Standard output for CI systems
    await this.generateCIOutput(result, ciConfig);
  }

  /**
   * Send Slack notification
   */
  private async sendSlackNotification(result: CIHardeningResult, ciConfig: CIHardeningConfig): Promise<void> {
    if (!ciConfig.slack_webhook_url) return;
    
    const emoji = result.success ? '✅' : '❌';
    const color = result.success ? 'good' : 'danger';
    
    const message = {
      text: `${emoji} Lens Phase C Hardening (${ciConfig.ci_mode})`,
      attachments: [{
        color,
        fields: [
          {
            title: 'Status',
            value: result.success ? 'PASSED' : 'FAILED',
            short: true
          },
          {
            title: 'Hardening Score',
            value: `${result.hardening_score}/100`,
            short: true
          },
          {
            title: 'Tripwires',
            value: `${result.quality_gate_results.tripwires_passed}/${result.quality_gate_results.tripwires_total}`,
            short: true
          },
          {
            title: 'Execution Time',
            value: `${(result.execution_time_ms / 1000).toFixed(1)}s`,
            short: true
          }
        ]
      }]
    };
    
    if (!result.success && result.failure_summary) {
      message.attachments[0]?.fields.push({
        title: 'Key Issues',
        value: result.failure_summary.recommendations.slice(0, 3).join('\n'),
        short: false
      });
    }
    
    await fetch(ciConfig.slack_webhook_url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(message)
    });
  }

  /**
   * Set GitHub Actions output
   */
  private async setGitHubActionsOutput(result: CIHardeningResult, ciConfig: CIHardeningConfig): Promise<void> {
    
    const outputs = [
      `hardening-success=${result.success}`,
      `hardening-score=${result.hardening_score}`,
      `execution-time=${result.execution_time_ms}`,
      `tripwires-passed=${result.quality_gate_results.tripwires_passed}`,
      `tripwires-total=${result.quality_gate_results.tripwires_total}`,
      `report-path=${result.artifacts.report_path}`
    ];
    
    for (const output of outputs) {
      console.log(`::set-output name=${output}`);
    }
    
    // Set job summary
    const summaryPath = process.env['GITHUB_STEP_SUMMARY'];
    if (summaryPath) {
      const summary = this.generateGitHubSummary(result, ciConfig);
      await fs.writeFile(summaryPath, summary, { flag: 'a' });
    }
  }

  /**
   * Generate GitHub Actions job summary
   */
  private generateGitHubSummary(result: CIHardeningResult, ciConfig: CIHardeningConfig): string {
    const emoji = result.success ? '✅' : '❌';
    
    let summary = `
## ${emoji} Lens Phase C Hardening (${ciConfig.ci_mode})

**Status**: ${result.success ? 'PASSED' : 'FAILED'}  
**Hardening Score**: ${result.hardening_score}/100  
**Execution Time**: ${(result.execution_time_ms / 1000).toFixed(1)}s

### Quality Gate Results

| Gate | Status | Results |
|------|--------|---------|
| Tripwires | ${result.quality_gate_results.tripwires_passed === result.quality_gate_results.tripwires_total ? '✅' : '❌'} | ${result.quality_gate_results.tripwires_passed}/${result.quality_gate_results.tripwires_total} |
| Slice Gates | ${result.quality_gate_results.slice_gates_passed === result.quality_gate_results.slice_gates_total ? '✅' : '❌'} | ${result.quality_gate_results.slice_gates_passed}/${result.quality_gate_results.slice_gates_total} |

`;

    if (!result.success && result.failure_summary) {
      summary += `
### ❌ Failure Summary

**Failing Tripwires**: ${result.failure_summary.failing_tripwires.join(', ') || 'None'}  
**Failing Slices**: ${result.failure_summary.failing_slices.length} slices  
**Performance Degradation**: ${result.failure_summary.degradation_percent.toFixed(1)}%

**Key Recommendations**:
${result.failure_summary.recommendations.slice(0, 5).map(r => `- ${r}`).join('\n')}
`;
    }
    
    summary += `
### Artifacts

- **Report**: \`${result.artifacts.report_path}\`
- **Plots**: \`${result.artifacts.plots_directory}\`
- **Raw Data**: \`${result.artifacts.raw_data_path}\`
`;
    
    return summary;
  }

  /**
   * Generate standard CI output
   */
  private async generateCIOutput(result: CIHardeningResult, ciConfig: CIHardeningConfig): Promise<void> {
    
    const outputPath = path.join(this.outputDir, 'ci-hardening-result.json');
    await fs.writeFile(outputPath, JSON.stringify(result, null, 2));
    
    console.log('📊 CI Hardening Results:');
    console.log(`  Success: ${result.success ? '✅' : '❌'}`);
    console.log(`  Hardening Score: ${result.hardening_score}/100`);
    console.log(`  Execution Time: ${(result.execution_time_ms / 1000).toFixed(1)}s`);
    console.log(`  Tripwires: ${result.quality_gate_results.tripwires_passed}/${result.quality_gate_results.tripwires_total}`);
    console.log(`  Slice Gates: ${result.quality_gate_results.slice_gates_passed}/${result.quality_gate_results.slice_gates_total}`);
    
    if (!result.success && result.failure_summary) {
      console.log('❌ Failure Details:');
      result.failure_summary.recommendations.forEach((rec, i) => {
        console.log(`  ${i + 1}. ${rec}`);
      });
    }
    
    console.log(`📄 Full results written to: ${outputPath}`);
  }

  /**
   * Handle CI artifacts (upload, archive, etc.)
   */
  private async handleCIArtifacts(result: CIHardeningResult, ciConfig: CIHardeningConfig): Promise<void> {
    
    // Create artifacts archive
    const archivePath = path.join(this.outputDir, 'ci-hardening-artifacts.tar.gz');
    
    try {
      // In a real implementation, this would create a proper archive
      const artifactsList = {
        report: result.artifacts.report_path,
        plots: result.artifacts.plots_directory,
        raw_data: result.artifacts.raw_data_path,
        ci_result: path.join(this.outputDir, 'ci-hardening-result.json')
      };
      
      await fs.writeFile(
        path.join(this.outputDir, 'artifacts-manifest.json'),
        JSON.stringify(artifactsList, null, 2)
      );
      
      console.log(`📦 CI artifacts manifest created: artifacts-manifest.json`);
      
      // S3 upload placeholder
      if (ciConfig.upload_to_s3) {
        console.log('☁️ S3 upload would happen here in real implementation');
      }
      
    } catch (error) {
      console.warn('⚠️ Failed to handle CI artifacts:', error);
    }
  }

  /**
   * Utility methods
   */
  private createTimeoutPromise(minutes: number): Promise<{ timedOut: boolean }> {
    return new Promise(resolve => {
      setTimeout(() => resolve({ timedOut: true }), minutes * 60 * 1000);
    });
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

/**
 * Default CI configuration factory
 */
export function createDefaultCIConfig(ciMode: 'pr' | 'nightly' | 'release'): CIHardeningConfig {
  const baseConfig: CIHardeningConfig = {
    ci_mode: ciMode,
    fail_fast: true,
    max_execution_time_minutes: 30,
    generate_artifacts: true,
    upload_to_s3: false,
    quality_gates: {
      enforce_tripwires: true,
      enforce_slice_gates: true,
      min_hardening_score: 75,
      max_degradation_percent: 15
    },
    retry_policy: {
      enabled: true,
      max_retries: 2,
      backoff_multiplier: 2
    }
  };

  // CI mode specific overrides
  const overrides = {
    'pr': {
      max_execution_time_minutes: 15,
      quality_gates: {
        ...baseConfig.quality_gates,
        min_hardening_score: 65,
        max_degradation_percent: 20
      }
    },
    'nightly': {
      max_execution_time_minutes: 45,
      upload_to_s3: true,
      quality_gates: {
        ...baseConfig.quality_gates,
        min_hardening_score: 80,
        max_degradation_percent: 10
      }
    },
    'release': {
      max_execution_time_minutes: 60,
      upload_to_s3: true,
      fail_fast: true,
      quality_gates: {
        ...baseConfig.quality_gates,
        min_hardening_score: 90,
        max_degradation_percent: 5
      }
    }
  };

  return { ...baseConfig, ...overrides[ciMode] };
}