/**
 * CLI Integration for Lens Quality Gates
 * Provides command-line interface for running tests and managing CI integration
 */

import { Command } from 'commander';
import { promises as fs } from 'fs';
import path from 'path';
import { TestOrchestrator, TestExecutionConfig } from './test-orchestrator.js';
import { GroundTruthBuilder } from './ground-truth-builder.js';

const program = new Command();

// Default configuration
const DEFAULT_CONFIG = {
  outputDir: './benchmark-results',
  natsUrl: 'nats://localhost:4222',
  maxDurationMinutes: 10
};

export class CLIIntegration {
  private orchestrator?: TestOrchestrator;
  
  async initialize(outputDir: string = DEFAULT_CONFIG.outputDir): Promise<void> {
    // Ensure output directory exists
    await fs.mkdir(outputDir, { recursive: true });
    
    // Initialize ground truth builder
    const groundTruthBuilder = new GroundTruthBuilder();
    await groundTruthBuilder.loadGoldenDataset();
    
    // Initialize test orchestrator
    this.orchestrator = new TestOrchestrator(
      groundTruthBuilder,
      outputDir,
      DEFAULT_CONFIG.natsUrl
    );
  }

  async runSmokeTests(options: any): Promise<void> {
    if (!this.orchestrator) {
      throw new Error('CLI not initialized - call initialize() first');
    }

    console.log('🔥 Running smoke tests for PR gate...');
    
    const config: TestExecutionConfig = {
      test_type: 'smoke_pr',
      trigger: 'manual',
      baseline_comparison: options.baseline || false,
      baseline_trace_id: options.baselineId,
      max_duration_minutes: options.timeout || 10,
      parallel_execution: true,
      generate_dashboard_data: true,
      pr_context: options.pr ? {
        pr_number: options.pr,
        branch_name: process.env.GITHUB_HEAD_REF || 'unknown',
        commit_sha: process.env.GITHUB_SHA || 'unknown',
        base_branch: process.env.GITHUB_BASE_REF || 'main'
      } : undefined
    };

    try {
      const result = await this.orchestrator.executeSmokeTests(config);
      
      console.log('\n📊 SMOKE TEST RESULTS:');
      console.log(`   Status: ${result.passed ? '✅ PASS' : '❌ FAIL'}`);
      console.log(`   Duration: ${(result.duration_ms / 1000).toFixed(1)}s`);
      console.log(`   Quality Score: ${(result.quality_score * 100).toFixed(1)}%`);
      console.log(`   Merge Status: ${result.blocking_merge ? '🚫 BLOCKED' : '✅ ALLOWED'}`);
      
      if (result.artifacts) {
        console.log('\n📄 Generated Artifacts:');
        Object.entries(result.artifacts).forEach(([key, path]) => {
          console.log(`   ${key}: ${path}`);
        });
      }

      // Exit with appropriate code
      process.exit(result.passed ? 0 : 1);
      
    } catch (error) {
      console.error('❌ Smoke test execution failed:', error);
      process.exit(1);
    }
  }

  async runNightlyTests(options: any): Promise<void> {
    if (!this.orchestrator) {
      throw new Error('CLI not initialized - call initialize() first');
    }

    console.log('🌙 Running full nightly tests...');
    
    const config: TestExecutionConfig = {
      test_type: 'full_nightly',
      trigger: 'manual',
      baseline_comparison: options.baseline || false,
      baseline_trace_id: options.baselineId,
      max_duration_minutes: options.timeout || 120,
      parallel_execution: true,
      generate_dashboard_data: true
    };

    try {
      const result = await this.orchestrator.executeFullNightlyTests(config);
      
      console.log('\n📊 NIGHTLY TEST RESULTS:');
      console.log(`   Status: ${result.passed ? '✅ PASS' : '⚠️ DEGRADED'}`);
      console.log(`   Duration: ${(result.duration_ms / (1000 * 60)).toFixed(1)} minutes`);
      console.log(`   Quality Score: ${(result.quality_score * 100).toFixed(1)}%`);
      console.log(`   Stability Score: ${(result.stability_score * 100).toFixed(1)}%`);
      console.log(`   Performance Score: ${(result.performance_score * 100).toFixed(1)}%`);
      
      if (result.artifacts) {
        console.log('\n📄 Generated Artifacts:');
        Object.entries(result.artifacts).forEach(([key, path]) => {
          console.log(`   ${key}: ${path}`);
        });
      }

      // Nightly tests don't fail the process - just report
      process.exit(0);
      
    } catch (error) {
      console.error('❌ Nightly test execution failed:', error);
      process.exit(1);
    }
  }

  async generateReport(options: any): Promise<void> {
    console.log('📋 Generating comprehensive test report...');
    
    const outputDir = options.output || DEFAULT_CONFIG.outputDir;
    
    try {
      // Find recent test results
      const files = await fs.readdir(outputDir);
      const summaryFiles = files.filter(f => f.includes('_summary.json'));
      
      if (summaryFiles.length === 0) {
        console.log('No test results found to generate report from');
        return;
      }
      
      // Load the most recent results
      const recentResults = [];
      for (const file of summaryFiles.slice(-5)) { // Last 5 results
        const filePath = path.join(outputDir, file);
        const content = await fs.readFile(filePath, 'utf8');
        recentResults.push(JSON.parse(content));
      }
      
      // Generate consolidated report
      const report = await this.generateConsolidatedReport(recentResults);
      
      const reportPath = path.join(outputDir, `consolidated_report_${new Date().toISOString().replace(/[:.]/g, '-')}.json`);
      await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
      
      console.log(`📊 Consolidated report generated: ${reportPath}`);
      
      // Print summary to console
      this.printReportSummary(report);
      
    } catch (error) {
      console.error('❌ Report generation failed:', error);
      process.exit(1);
    }
  }

  async validateCI(options: any): Promise<void> {
    console.log('🔍 Validating CI configuration...');
    
    const checks = [
      { name: 'GitHub Actions workflow', path: '.github/workflows/lens-quality-gates.yml' },
      { name: 'Package.json scripts', path: 'package.json' },
      { name: 'Golden dataset', path: 'sample-storyviz/golden-dataset.jsonl' },
      { name: 'Indexed content', path: 'indexed-content' }
    ];
    
    let allValid = true;
    
    for (const check of checks) {
      try {
        const exists = await fs.access(check.path).then(() => true).catch(() => false);
        if (exists) {
          console.log(`   ✅ ${check.name}: Found`);
        } else {
          console.log(`   ❌ ${check.name}: Missing (${check.path})`);
          allValid = false;
        }
      } catch (error) {
        console.log(`   ⚠️ ${check.name}: Error checking (${error})`);
        allValid = false;
      }
    }
    
    // Check environment variables
    const envVars = ['GITHUB_TOKEN', 'NATS_URL'];
    console.log('\n🌍 Environment Variables:');
    for (const envVar of envVars) {
      const value = process.env[envVar];
      if (value) {
        console.log(`   ✅ ${envVar}: Set`);
      } else {
        console.log(`   ⚠️ ${envVar}: Not set (optional for local testing)`);
      }
    }
    
    console.log(`\n🎯 CI Validation: ${allValid ? '✅ PASS' : '❌ FAIL'}`);
    
    if (!allValid && !options.force) {
      process.exit(1);
    }
  }

  private async generateConsolidatedReport(results: any[]): Promise<any> {
    const report = {
      metadata: {
        generated_at: new Date().toISOString(),
        total_test_runs: results.length,
        date_range: {
          oldest: results[0]?.timestamp || 'unknown',
          newest: results[results.length - 1]?.timestamp || 'unknown'
        }
      },
      
      summary: {
        smoke_tests: results.filter(r => r.test_type === 'smoke_pr').length,
        nightly_tests: results.filter(r => r.test_type === 'full_nightly').length,
        average_quality_score: results.reduce((sum, r) => sum + (r.quality_score || 0), 0) / results.length,
        average_stability_score: results.reduce((sum, r) => sum + (r.stability_score || 0), 0) / results.length,
        pass_rate: results.filter(r => r.passed).length / results.length
      },
      
      trends: {
        quality_trend: this.calculateTrend(results.map(r => r.quality_score || 0)),
        stability_trend: this.calculateTrend(results.map(r => r.stability_score || 0)),
        performance_trend: this.calculateTrend(results.map(r => r.performance_score || 0))
      },
      
      recent_failures: results.filter(r => !r.passed).slice(-3),
      
      test_results: results
    };
    
    return report;
  }

  private calculateTrend(values: number[]): { direction: 'up' | 'down' | 'stable'; change: number } {
    if (values.length < 2) return { direction: 'stable', change: 0 };
    
    const recent = values.slice(-3).reduce((a, b) => a + b, 0) / Math.min(3, values.length);
    const historical = values.slice(0, -3).reduce((a, b) => a + b, 0) / Math.max(1, values.length - 3);
    
    const change = recent - historical;
    const direction = Math.abs(change) < 0.01 ? 'stable' : (change > 0 ? 'up' : 'down');
    
    return { direction, change };
  }

  private printReportSummary(report: any): void {
    console.log('\n📈 QUALITY TRENDS:');
    console.log(`   Quality Score: ${(report.summary.average_quality_score * 100).toFixed(1)}% (${report.trends.quality_trend.direction})`);
    console.log(`   Stability Score: ${(report.summary.average_stability_score * 100).toFixed(1)}% (${report.trends.stability_trend.direction})`);
    console.log(`   Pass Rate: ${(report.summary.pass_rate * 100).toFixed(1)}%`);
    console.log(`   Total Test Runs: ${report.metadata.total_test_runs}`);
    
    if (report.recent_failures.length > 0) {
      console.log('\n⚠️ RECENT FAILURES:');
      report.recent_failures.forEach((failure: any, index: number) => {
        console.log(`   ${index + 1}. ${failure.test_type} - ${failure.timestamp}`);
      });
    }
  }
}

// CLI Command definitions
program
  .name('lens-gates')
  .description('Lens Quality Gates CLI')
  .version('1.0.0');

program
  .command('smoke')
  .description('Run smoke tests for PR gate')
  .option('-b, --baseline', 'Enable baseline comparison')
  .option('--baseline-id <id>', 'Baseline trace ID for comparison')
  .option('-t, --timeout <minutes>', 'Test timeout in minutes', '10')
  .option('--pr <number>', 'PR number for context')
  .option('-o, --output <dir>', 'Output directory', DEFAULT_CONFIG.outputDir)
  .action(async (options) => {
    const cli = new CLIIntegration();
    await cli.initialize(options.output);
    await cli.runSmokeTests(options);
  });

program
  .command('nightly')
  .description('Run full nightly tests')
  .option('-b, --baseline', 'Enable baseline comparison')
  .option('--baseline-id <id>', 'Baseline trace ID for comparison')
  .option('-t, --timeout <minutes>', 'Test timeout in minutes', '120')
  .option('-o, --output <dir>', 'Output directory', DEFAULT_CONFIG.outputDir)
  .action(async (options) => {
    const cli = new CLIIntegration();
    await cli.initialize(options.output);
    await cli.runNightlyTests(options);
  });

program
  .command('report')
  .description('Generate consolidated test report')
  .option('-o, --output <dir>', 'Output directory', DEFAULT_CONFIG.outputDir)
  .action(async (options) => {
    const cli = new CLIIntegration();
    await cli.generateReport(options);
  });

program
  .command('validate-ci')
  .description('Validate CI configuration')
  .option('--force', 'Continue even if validation fails')
  .action(async (options) => {
    const cli = new CLIIntegration();
    await cli.validateCI(options);
  });

// Export for programmatic use
export { program };

// CLI execution when run directly
if (import.meta.url === `file://${process.argv[1]}`) {
  program.parse();
}