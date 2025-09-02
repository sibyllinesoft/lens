#!/usr/bin/env node

/**
 * Nightly Benchmark Automation Script
 * 
 * This script orchestrates the automated nightly benchmark execution for the Lens code search system.
 * It handles corpus validation, benchmark execution, and result archiving with comprehensive error handling.
 */

import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Import our benchmark infrastructure
const projectRoot = path.resolve(__dirname, '..');
process.chdir(projectRoot);

// Dynamic imports to handle ES modules properly
let BenchmarkSuiteRunner, GroundTruthBuilder;

async function loadBenchmarkModules() {
  try {
    // Build the project first if needed
    const { execSync } = await import('child_process');
    
    try {
      await fs.access('./dist/benchmark/suite-runner.js');
    } catch {
      console.log('📦 Building project for benchmark execution...');
      execSync('npm run build', { stdio: 'inherit' });
    }
    
    const suiteRunnerModule = await import('../dist/benchmark/suite-runner.js');
    const groundTruthModule = await import('../dist/benchmark/ground-truth-builder.js');
    
    BenchmarkSuiteRunner = suiteRunnerModule.BenchmarkSuiteRunner;
    GroundTruthBuilder = groundTruthModule.GroundTruthBuilder;
    
    console.log('✅ Benchmark modules loaded successfully');
  } catch (error) {
    console.error('❌ Failed to load benchmark modules:', error.message);
    process.exit(1);
  }
}

/**
 * Configuration for nightly benchmark automation
 */
const AUTOMATION_CONFIG = {
  // Timeouts and retries
  CORPUS_VALIDATION_TIMEOUT_MS: 5 * 60 * 1000, // 5 minutes
  BENCHMARK_TIMEOUT_MS: 3 * 60 * 60 * 1000,    // 3 hours
  SERVER_HEALTH_CHECK_RETRIES: 10,
  
  // Performance thresholds for alerts
  REGRESSION_THRESHOLDS: {
    recall_at_10: 0.10,    // 10% drop triggers alert
    latency_p95: 0.20,     // 20% increase triggers alert
    error_rate: 0.05       // 5% error rate triggers alert
  },
  
  // NATS telemetry configuration
  NATS_CONFIG: {
    url: process.env.NATS_URL || 'nats://localhost:4222',
    timeout: 30000,
    reconnectAttempts: 5
  },
  
  // Search server configuration
  SEARCH_SERVER: {
    url: 'http://localhost:4000',
    health_endpoint: '/health',
    search_endpoint: '/search'
  }
};

/**
 * Main automation orchestrator class
 */
class NightlyBenchmarkAutomation {
  constructor(options = {}) {
    this.runId = options.runId || `nightly-${new Date().toISOString().replace(/[:.]/g, '-')}`;
    this.outputDir = options.outputDir || path.join('benchmark-results', this.runId);
    this.suiteType = options.suiteType || 'full';
    this.verbose = options.verbose || false;
    
    this.groundTruthBuilder = null;
    this.suiteRunner = null;
    
    console.log(`🎯 Initializing nightly benchmark automation`);
    console.log(`   Run ID: ${this.runId}`);
    console.log(`   Suite Type: ${this.suiteType}`);
    console.log(`   Output Dir: ${this.outputDir}`);
  }
  
  async initialize() {
    // Ensure output directory exists
    await fs.mkdir(this.outputDir, { recursive: true });
    
    // Initialize benchmark infrastructure
    console.log('🔧 Initializing benchmark infrastructure...');
    
    this.groundTruthBuilder = new GroundTruthBuilder();
    await this.groundTruthBuilder.initialize();
    
    this.suiteRunner = new BenchmarkSuiteRunner(
      this.groundTruthBuilder,
      this.outputDir,
      AUTOMATION_CONFIG.NATS_CONFIG.url
    );
    
    console.log('✅ Benchmark infrastructure initialized');
  }
  
  /**
   * Validate corpus-golden consistency before running benchmarks
   */
  async validateConsistency() {
    console.log('🔍 Running corpus-golden consistency validation...');
    
    const startTime = Date.now();
    
    try {
      const result = await Promise.race([
        this.suiteRunner.validateCorpusGoldenConsistency(),
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Consistency validation timeout')), 
                     AUTOMATION_CONFIG.CORPUS_VALIDATION_TIMEOUT_MS)
        )
      ]);
      
      const duration = Date.now() - startTime;
      
      // Write validation report
      const reportPath = path.join(this.outputDir, 'consistency-validation.json');
      await fs.writeFile(reportPath, JSON.stringify({
        timestamp: new Date().toISOString(),
        duration_ms: duration,
        ...result
      }, null, 2));
      
      if (!result.passed) {
        const errorMsg = `Consistency validation failed: ${result.report.inconsistent_results} inconsistencies found`;
        console.error(`❌ ${errorMsg}`);
        throw new Error(errorMsg);
      }
      
      console.log(`✅ Consistency validation passed in ${duration}ms`);
      console.log(`   Valid results: ${result.report.valid_results}/${result.report.total_expected_results}`);
      console.log(`   Pass rate: ${(result.report.pass_rate * 100).toFixed(1)}%`);
      
      return result;
    } catch (error) {
      console.error('❌ Consistency validation failed:', error.message);
      throw error;
    }
  }
  
  /**
   * Check if search server is healthy and ready
   */
  async checkServerHealth() {
    console.log('🏥 Checking search server health...');
    
    const healthUrl = `${AUTOMATION_CONFIG.SEARCH_SERVER.url}${AUTOMATION_CONFIG.SEARCH_SERVER.health_endpoint}`;
    
    for (let attempt = 1; attempt <= AUTOMATION_CONFIG.SERVER_HEALTH_CHECK_RETRIES; attempt++) {
      try {
        const response = await fetch(healthUrl, { 
          signal: AbortSignal.timeout(5000) 
        });
        
        if (response.ok) {
          console.log(`✅ Search server healthy (attempt ${attempt})`);
          return true;
        }
        
        console.warn(`⚠️ Search server returned ${response.status} (attempt ${attempt})`);
      } catch (error) {
        console.warn(`⚠️ Health check failed (attempt ${attempt}): ${error.message}`);
      }
      
      if (attempt < AUTOMATION_CONFIG.SERVER_HEALTH_CHECK_RETRIES) {
        const delay = Math.min(1000 * Math.pow(2, attempt), 10000); // Exponential backoff, max 10s
        console.log(`⏳ Waiting ${delay}ms before retry...`);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
    
    throw new Error(`Search server health check failed after ${AUTOMATION_CONFIG.SERVER_HEALTH_CHECK_RETRIES} attempts`);
  }
  
  /**
   * Execute the benchmark suite
   */
  async runBenchmarkSuite() {
    console.log(`🚀 Starting ${this.suiteType} benchmark suite execution...`);
    
    const startTime = Date.now();
    const metadata = {
      run_id: this.runId,
      suite_type: this.suiteType,
      start_time: new Date().toISOString(),
      environment: {
        node_version: process.version,
        platform: process.platform,
        arch: process.arch,
        memory_limit_mb: Math.round(process.memoryUsage().heapTotal / 1024 / 1024)
      }
    };
    
    try {
      let benchmarkResult;
      
      if (this.suiteType === 'smoke') {
        benchmarkResult = await Promise.race([
          this.suiteRunner.runSmokeSuite(),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Smoke suite timeout')), 30 * 60 * 1000) // 30 min
          )
        ]);
      } else {
        benchmarkResult = await Promise.race([
          this.suiteRunner.runFullSuite(),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Full suite timeout')), 
                       AUTOMATION_CONFIG.BENCHMARK_TIMEOUT_MS)
          )
        ]);
      }
      
      const duration = Date.now() - startTime;
      
      // Enrich metadata with results
      metadata.end_time = new Date().toISOString();
      metadata.duration_ms = duration;
      metadata.status = 'completed';
      metadata.total_queries = benchmarkResult.total_queries;
      metadata.completed_queries = benchmarkResult.completed_queries;
      metadata.failed_queries = benchmarkResult.failed_queries;
      
      // Write execution metadata
      const metadataPath = path.join(this.outputDir, 'execution-metadata.json');
      await fs.writeFile(metadataPath, JSON.stringify(metadata, null, 2));
      
      // Write benchmark results
      const resultsPath = path.join(this.outputDir, 'benchmark-results.json');
      await fs.writeFile(resultsPath, JSON.stringify(benchmarkResult, null, 2));
      
      console.log(`✅ Benchmark suite completed in ${Math.round(duration / 1000)}s`);
      console.log(`   Completed queries: ${benchmarkResult.completed_queries}`);
      console.log(`   Failed queries: ${benchmarkResult.failed_queries}`);
      console.log(`   Key metrics:`);
      console.log(`     Recall@10: ${(benchmarkResult.metrics.recall_at_10 * 100).toFixed(1)}%`);
      console.log(`     NDCG@10: ${(benchmarkResult.metrics.ndcg_at_10 * 100).toFixed(1)}%`);
      console.log(`     P95 Latency: ${benchmarkResult.metrics.stage_latencies.e2e_p95.toFixed(0)}ms`);
      
      return benchmarkResult;
    } catch (error) {
      const duration = Date.now() - startTime;
      
      metadata.end_time = new Date().toISOString();
      metadata.duration_ms = duration;
      metadata.status = 'failed';
      metadata.error = error.message;
      
      const metadataPath = path.join(this.outputDir, 'execution-metadata.json');
      await fs.writeFile(metadataPath, JSON.stringify(metadata, null, 2));
      
      console.error(`❌ Benchmark suite failed after ${Math.round(duration / 1000)}s: ${error.message}`);
      throw error;
    }
  }
  
  /**
   * Generate summary for quick status checks
   */
  async generateSummary(benchmarkResult) {
    console.log('📊 Generating benchmark summary...');
    
    const summary = {
      run_id: this.runId,
      timestamp: new Date().toISOString(),
      suite_type: this.suiteType,
      status: benchmarkResult.status,
      
      // Key performance metrics
      performance: {
        recall_at_10: benchmarkResult.metrics.recall_at_10,
        recall_at_50: benchmarkResult.metrics.recall_at_50,
        ndcg_at_10: benchmarkResult.metrics.ndcg_at_10,
        mrr: benchmarkResult.metrics.mrr
      },
      
      // Latency metrics
      latency: {
        stage_a_p95: benchmarkResult.metrics.stage_latencies.stage_a_p95,
        stage_b_p95: benchmarkResult.metrics.stage_latencies.stage_b_p95,
        stage_c_p95: benchmarkResult.metrics.stage_latencies.stage_c_p95,
        e2e_p95: benchmarkResult.metrics.stage_latencies.e2e_p95
      },
      
      // Quality metrics
      quality: {
        total_queries: benchmarkResult.total_queries,
        completed_queries: benchmarkResult.completed_queries,
        failed_queries: benchmarkResult.failed_queries,
        success_rate: benchmarkResult.completed_queries / benchmarkResult.total_queries,
        error_rate: benchmarkResult.failed_queries / benchmarkResult.total_queries
      }
    };
    
    // Write summary
    const summaryPath = path.join(this.outputDir, 'summary.json');
    await fs.writeFile(summaryPath, JSON.stringify(summary, null, 2));
    
    console.log(`✅ Summary written to ${summaryPath}`);
    return summary;
  }
}

/**
 * Command-line interface
 */
async function main() {
  const args = process.argv.slice(2);
  const command = args[0];
  
  // Parse command line arguments
  function getArg(name, defaultValue = null) {
    const index = args.findIndex(arg => arg === `--${name}`);
    return index !== -1 && index + 1 < args.length ? args[index + 1] : defaultValue;
  }
  
  function hasFlag(name) {
    return args.includes(`--${name}`);
  }
  
  try {
    await loadBenchmarkModules();
    
    switch (command) {
      case 'validate-consistency': {
        const automation = new NightlyBenchmarkAutomation({
          runId: getArg('run-id'),
          outputDir: getArg('output-dir'),
          verbose: hasFlag('verbose')
        });
        
        await automation.initialize();
        await automation.validateConsistency();
        
        console.log('✅ Corpus-golden consistency validation completed');
        break;
      }
      
      case 'run-suite': {
        const automation = new NightlyBenchmarkAutomation({
          runId: getArg('run-id'),
          outputDir: getArg('output-dir'),
          suiteType: getArg('suite-type', 'full'),
          verbose: hasFlag('verbose')
        });
        
        await automation.initialize();
        
        // Health check first
        await automation.checkServerHealth();
        
        // Run consistency validation
        await automation.validateConsistency();
        
        // Execute benchmark suite
        const result = await automation.runBenchmarkSuite();
        
        // Generate summary
        await automation.generateSummary(result);
        
        console.log(`🎯 Benchmark automation completed successfully (Run ID: ${automation.runId})`);
        break;
      }
      
      case 'health-check': {
        const automation = new NightlyBenchmarkAutomation();
        await automation.checkServerHealth();
        console.log('✅ Server health check passed');
        break;
      }
      
      default:
        console.error('❌ Unknown command. Available commands:');
        console.error('  validate-consistency   Run corpus-golden consistency check');
        console.error('  run-suite             Execute benchmark suite');
        console.error('  health-check          Check search server health');
        console.error('');
        console.error('Options:');
        console.error('  --run-id <id>         Unique run identifier');
        console.error('  --output-dir <dir>    Output directory for results');
        console.error('  --suite-type <type>   Suite type (full|smoke)');
        console.error('  --verbose             Enable verbose logging');
        process.exit(1);
    }
  } catch (error) {
    console.error(`❌ Command failed: ${error.message}`);
    if (hasFlag('verbose')) {
      console.error('Stack trace:', error.stack);
    }
    process.exit(1);
  }
}

// Execute main function if this script is run directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(error => {
    console.error('❌ Fatal error:', error);
    process.exit(1);
  });
}

export { NightlyBenchmarkAutomation, AUTOMATION_CONFIG };
