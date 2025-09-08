#!/usr/bin/env tsx

/**
 * Master Execution Script for Benchmark Protocol v1.0 
 * 
 * Executes the complete 6-step benchmarking pipeline:
 * 1. Build pooled qrels (CoIR, SWE-bench Verified, CSN, CoSQA)
 * 2. Warmup competitors (Lens, BM25, BM25+proximity, Hybrid)  
 * 3. Execute benchmark runs (150ms SLA enforcement)
 * 4. Score results (bootstrap B≥2000, permutation + Holm)
 * 5. Mine performance gaps (intent×language slice analysis)
 * 6. Generate publication plots (hero charts + confidence intervals)
 */

import { exec } from 'child_process';
import { promises as fs } from 'fs';
import * as path from 'path';
import { promisify } from 'util';

const execAsync = promisify(exec);

interface BenchmarkConfig {
  suites: string[];
  systems: string[];
  sla_ms: number;
  bootstrap_iterations: number;
  output_base: string;
  corpus_path: string;
}

const CONFIG: BenchmarkConfig = {
  suites: ['coir', 'swe_verified', 'csn', 'cosqa'],
  systems: ['lens', 'bm25', 'bm25_prox', 'hybrid'],
  sla_ms: 150,
  bootstrap_iterations: 2000,
  output_base: './benchmark-protocol-results',
  corpus_path: './benchmark-corpus'
};

class BenchmarkExecutor {
  private config: BenchmarkConfig;
  private startTime: Date;

  constructor(config: BenchmarkConfig) {
    this.config = config;
    this.startTime = new Date();
  }

  async execute(): Promise<void> {
    console.log('🚀 Starting Benchmark Protocol v1.0 Execution');
    console.log(`📅 Started: ${this.startTime.toISOString()}`);
    console.log(`⚙️  Configuration:`);
    console.log(`   • Suites: ${this.config.suites.join(', ')}`);
    console.log(`   • Systems: ${this.config.systems.join(', ')}`);
    console.log(`   • SLA: ${this.config.sla_ms}ms`);
    console.log(`   • Bootstrap: ${this.config.bootstrap_iterations} iterations`);
    console.log();

    try {
      // Step 0: Setup and validation
      await this.setupEnvironment();

      // Step 1: Build pooled qrels
      await this.step1_BuildPooledQrels();

      // Step 2: Warmup competitors
      await this.step2_WarmupCompetitors();

      // Step 3: Execute benchmark runs
      await this.step3_ExecuteBenchmarkRuns();

      // Step 4: Score results with statistical rigor
      await this.step4_ScoreResults();

      // Step 5: Mine performance gaps
      await this.step5_MinePerformanceGaps();

      // Step 6: Generate publication plots
      await this.step6_GeneratePublicationPlots();

      // Final summary
      await this.generateFinalSummary();

    } catch (error) {
      console.error('❌ Benchmark execution failed:', error);
      process.exit(1);
    }
  }

  private async setupEnvironment(): Promise<void> {
    console.log('🏗️  Step 0: Setting up environment...');

    // Create output directories
    const dirs = [
      this.config.output_base,
      path.join(this.config.output_base, 'pool'),
      path.join(this.config.output_base, 'runs'),
      path.join(this.config.output_base, 'scored'),
      path.join(this.config.output_base, 'gaps'),
      path.join(this.config.output_base, 'plots'),
      path.join(this.config.output_base, 'reports')
    ];

    for (const dir of dirs) {
      await fs.mkdir(dir, { recursive: true });
    }

    // Ensure TypeScript is compiled
    console.log('   Compiling TypeScript...');
    await execAsync('npm run build');

    // Validate corpus exists
    try {
      await fs.access(this.config.corpus_path);
      const files = await fs.readdir(this.config.corpus_path);
      console.log(`   ✅ Corpus validated: ${files.length} files`);
    } catch (error) {
      throw new Error(`Corpus not found at ${this.config.corpus_path}`);
    }

    console.log('   ✅ Environment setup complete\n');
  }

  private async step1_BuildPooledQrels(): Promise<void> {
    console.log('📋 Step 1: Building pooled qrels...');

    const outputDir = path.join(this.config.output_base, 'pool');
    
    const cmd = [
      'node dist/bench/cli.js build-pool',
      `--suites ${this.config.suites.join(',')}`,
      `--systems ${this.config.systems.join(',')}`,
      `--sla ${this.config.sla_ms}`,
      `--top-k 50`,
      `--min-agreement 2`,
      `--out ${outputDir}`
    ].join(' ');

    console.log(`   Running: ${cmd}`);
    
    try {
      const { stdout, stderr } = await execAsync(cmd);
      if (stdout) console.log('   ' + stdout.split('\n').join('\n   '));
      if (stderr) console.warn('   ' + stderr.split('\n').join('\n   '));

      // Validate pooled qrels were created
      for (const suite of this.config.suites) {
        const poolFile = path.join(outputDir, `${suite}_pooled_qrels.json`);
        try {
          await fs.access(poolFile);
          const poolData = JSON.parse(await fs.readFile(poolFile, 'utf8'));
          console.log(`   ✅ ${suite}: ${Object.keys(poolData).length} pooled queries`);
        } catch (error) {
          console.warn(`   ⚠️  ${suite}: Pool file not found or invalid`);
        }
      }

    } catch (error) {
      console.error('   ❌ Pooled qrels building failed:', error);
      throw error;
    }

    console.log('   ✅ Pooled qrels complete\n');
  }

  private async step2_WarmupCompetitors(): Promise<void> {
    console.log('🔥 Step 2: Warming up competitors...');

    const attestationFile = path.join(this.config.output_base, 'attestation.json');
    
    const cmd = [
      'node dist/bench/cli.js warmup',
      `--systems ${this.config.systems.join(',')}`,
      `--warmup-queries 10`,
      `--hardware-check strict`,
      `--attest ${attestationFile}`
    ].join(' ');

    console.log(`   Running: ${cmd}`);
    
    try {
      const { stdout, stderr } = await execAsync(cmd);
      if (stdout) console.log('   ' + stdout.split('\n').join('\n   '));
      if (stderr) console.warn('   ' + stderr.split('\n').join('\n   '));

      // Validate attestation file
      const attestation = JSON.parse(await fs.readFile(attestationFile, 'utf8'));
      console.log(`   ✅ Hardware attestation: ${attestation.systems.length} systems ready`);
      console.log(`   📊 Hardware fingerprint: ${attestation.hardware_fingerprint.substring(0, 16)}...`);

    } catch (error) {
      console.error('   ❌ Competitor warmup failed:', error);
      throw error;
    }

    console.log('   ✅ Competitor warmup complete\n');
  }

  private async step3_ExecuteBenchmarkRuns(): Promise<void> {
    console.log('🚀 Step 3: Executing benchmark runs...');

    const runsDir = path.join(this.config.output_base, 'runs');

    for (const suite of this.config.suites) {
      console.log(`   🔄 Running suite: ${suite}`);
      
      const cmd = [
        'node dist/bench/cli.js run',
        `--suite ${suite}`,
        `--systems ${this.config.systems.join(',')}`,
        `--sla ${this.config.sla_ms}`,
        `--queries-per-system 1000`,
        `--parallel-workers 4`,
        `--out ${runsDir}`
      ].join(' ');

      try {
        const { stdout, stderr } = await execAsync(cmd, { timeout: 30 * 60 * 1000 }); // 30min timeout
        if (stdout) console.log('     ' + stdout.split('\n').join('\n     '));
        if (stderr) console.warn('     ' + stderr.split('\n').join('\n     '));

        // Validate run results
        const resultsFile = path.join(runsDir, `${suite}_results.json`);
        const statsFile = path.join(runsDir, `${suite}_stats.json`);
        
        const results = JSON.parse(await fs.readFile(resultsFile, 'utf8'));
        const stats = JSON.parse(await fs.readFile(statsFile, 'utf8'));
        
        console.log(`     ✅ ${suite}: ${results.length} queries executed`);
        console.log(`     📊 SLA compliance: ${(stats.sla_compliance_rate * 100).toFixed(1)}%`);
        console.log(`     📈 Error rate: ${(stats.error_rate * 100).toFixed(1)}%`);

      } catch (error) {
        console.error(`     ❌ ${suite} execution failed:`, error);
        throw error;
      }
    }

    console.log('   ✅ Benchmark runs complete\n');
  }

  private async step4_ScoreResults(): Promise<void> {
    console.log('🧮 Step 4: Scoring results with statistical testing...');

    const runsDir = path.join(this.config.output_base, 'runs');
    const poolDir = path.join(this.config.output_base, 'pool');
    const scoredDir = path.join(this.config.output_base, 'scored');

    const cmd = [
      'node dist/bench/cli.js score',
      `--runs ${runsDir}`,
      `--pool ${poolDir}`,
      `--bootstrap ${this.config.bootstrap_iterations}`,
      `--permute`,
      `--holm`,
      `--out ${scoredDir}`
    ].join(' ');

    console.log(`   Running: ${cmd}`);
    
    try {
      const { stdout, stderr } = await execAsync(cmd, { timeout: 60 * 60 * 1000 }); // 1 hour timeout
      if (stdout) console.log('   ' + stdout.split('\n').join('\n   '));
      if (stderr) console.warn('   ' + stderr.split('\n').join('\n   '));

      // Validate scored results
      const aggregateFile = path.join(scoredDir, 'benchmark_agg.parquet');
      const detailFile = path.join(scoredDir, 'benchmark_detail.parquet');
      const csvFile = path.join(scoredDir, 'benchmark_results.csv');

      try {
        await fs.access(aggregateFile);
        await fs.access(detailFile);  
        await fs.access(csvFile);
        console.log('   ✅ Scored results exported to Parquet and CSV');
      } catch (error) {
        console.warn('   ⚠️  Some output files missing, continuing...');
      }

    } catch (error) {
      console.error('   ❌ Result scoring failed:', error);
      throw error;
    }

    console.log('   ✅ Result scoring complete\n');
  }

  private async step5_MinePerformanceGaps(): Promise<void> {
    console.log('⛏️  Step 5: Mining performance gaps...');

    const scoredDir = path.join(this.config.output_base, 'scored');
    const gapsDir = path.join(this.config.output_base, 'gaps');
    const gapsFile = path.join(gapsDir, 'performance_gaps.csv');

    const cmd = [
      'node dist/bench/cli.js mine',
      `--in ${scoredDir}`,
      `--slice-analysis intent,language`,
      `--witness-attribution`,
      `--timeout-analysis`,
      `--calibration-flags`,
      `--out ${gapsFile}`
    ].join(' ');

    console.log(`   Running: ${cmd}`);
    
    try {
      const { stdout, stderr } = await execAsync(cmd);
      if (stdout) console.log('   ' + stdout.split('\n').join('\n   '));
      if (stderr) console.warn('   ' + stderr.split('\n').join('\n   '));

      // Validate gaps analysis
      const gapsContent = await fs.readFile(gapsFile, 'utf8');
      const gapsLines = gapsContent.trim().split('\n');
      console.log(`   ✅ Gap analysis: ${gapsLines.length - 1} performance gaps identified`);

      // Create summary
      const gapsSummary = {
        timestamp: new Date().toISOString(),
        gaps_identified: gapsLines.length - 1,
        analysis_dimensions: ['intent', 'language'],
        output_file: gapsFile
      };

      await fs.writeFile(
        path.join(gapsDir, 'gaps_summary.json'), 
        JSON.stringify(gapsSummary, null, 2)
      );

    } catch (error) {
      console.error('   ❌ Gap mining failed:', error);
      throw error;
    }

    console.log('   ✅ Performance gap mining complete\n');
  }

  private async step6_GeneratePublicationPlots(): Promise<void> {
    console.log('📊 Step 6: Generating publication plots...');

    const scoredDir = path.join(this.config.output_base, 'scored');
    const plotsDir = path.join(this.config.output_base, 'plots');

    const cmd = [
      'node dist/bench/cli.js plot',
      `--in ${scoredDir}`,
      `--figures hero,latency,calibration,gaps,witness,utility`,
      `--format png`,
      `--dpi 300`,
      `--out ${plotsDir}`
    ].join(' ');

    console.log(`   Running: ${cmd}`);
    
    try {
      const { stdout, stderr } = await execAsync(cmd);
      if (stdout) console.log('   ' + stdout.split('\n').join('\n   '));
      if (stderr) console.warn('   ' + stderr.split('\n').join('\n   '));

      // Validate plots
      const manifestFile = path.join(plotsDir, 'plot_manifest.json');
      const manifest = JSON.parse(await fs.readFile(manifestFile, 'utf8'));
      console.log(`   ✅ Generated ${manifest.figures.length} publication figures`);

    } catch (error) {
      console.error('   ❌ Plot generation failed:', error);
      throw error;
    }

    console.log('   ✅ Publication plots complete\n');
  }

  private async generateFinalSummary(): Promise<void> {
    console.log('📝 Generating final summary...');

    const endTime = new Date();
    const duration = endTime.getTime() - this.startTime.getTime();
    
    const summary = {
      execution_info: {
        protocol_version: '1.0',
        started: this.startTime.toISOString(),
        completed: endTime.toISOString(),
        duration_minutes: Math.round(duration / 60000),
        git_hash: await this.getCurrentGitHash()
      },
      configuration: this.config,
      results: {
        pooled_qrels: `${this.config.output_base}/pool/`,
        benchmark_runs: `${this.config.output_base}/runs/`,
        scored_results: `${this.config.output_base}/scored/`,
        performance_gaps: `${this.config.output_base}/gaps/`,
        publication_plots: `${this.config.output_base}/plots/`
      },
      reproducibility: {
        attestation_file: `${this.config.output_base}/attestation.json`,
        configuration_fingerprint: this.generateConfigFingerprint(),
        system_info: await this.getSystemInfo()
      }
    };

    const summaryFile = path.join(this.config.output_base, 'execution_summary.json');
    await fs.writeFile(summaryFile, JSON.stringify(summary, null, 2));

    console.log('🎉 Benchmark Protocol v1.0 Execution Complete!');
    console.log(`📅 Duration: ${Math.round(duration / 60000)} minutes`);
    console.log(`📁 Results: ${this.config.output_base}/`);
    console.log(`📋 Summary: ${summaryFile}`);
    console.log();
    console.log('📊 Key Outputs:');
    console.log(`   • Pooled qrels: ${this.config.suites.length} suites`);
    console.log(`   • Benchmark runs: ${this.config.systems.length} systems × ${this.config.suites.length} suites`);
    console.log(`   • Statistical analysis: ${this.config.bootstrap_iterations} bootstrap iterations`);
    console.log(`   • Performance gaps: Intent×Language slice analysis`);
    console.log(`   • Publication plots: Hero charts with confidence intervals`);
    console.log();
    console.log('✅ Ready for competitive evaluation and publication!');
  }

  private async getCurrentGitHash(): Promise<string> {
    try {
      const { stdout } = await execAsync('git rev-parse HEAD');
      return stdout.trim().substring(0, 8);
    } catch {
      return 'unknown';
    }
  }

  private generateConfigFingerprint(): string {
    const configStr = JSON.stringify(this.config, Object.keys(this.config).sort());
    const crypto = require('crypto');
    return crypto.createHash('sha256').update(configStr).digest('hex').substring(0, 16);
  }

  private async getSystemInfo(): Promise<any> {
    try {
      const { stdout: nodeVersion } = await execAsync('node --version');
      const { stdout: cpuInfo } = await execAsync('lscpu | grep "Model name" | cut -d: -f2 | xargs');
      const { stdout: memInfo } = await execAsync('free -h | grep "Mem:" | awk \'{print $2}\'');
      
      return {
        node_version: nodeVersion.trim(),
        cpu_model: cpuInfo.trim(),
        total_memory: memInfo.trim(),
        platform: process.platform,
        arch: process.arch
      };
    } catch {
      return {
        node_version: process.version,
        platform: process.platform,
        arch: process.arch
      };
    }
  }
}

// Execute if run directly
if (require.main === module) {
  const executor = new BenchmarkExecutor(CONFIG);
  executor.execute();
}

export default BenchmarkExecutor;