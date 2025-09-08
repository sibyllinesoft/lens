#!/usr/bin/env node

/**
 * Baseline Benchmark Runner with Pinned Golden Dataset
 * 
 * This script runs benchmarks using the pinned golden dataset to establish
 * stable baselines for performance comparison.
 */

import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { PinnedGroundTruthLoader } from './src/benchmark/pinned-ground-truth-loader.js';
import { BenchmarkSuiteRunner } from './src/benchmark/suite-runner.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class BaselineBenchmarkRunner {
  constructor() {
    this.workingDir = process.cwd();
    this.outputDir = path.join(this.workingDir, 'baseline-results');
    this.pinnedLoader = new PinnedGroundTruthLoader(this.workingDir);
  }

  async runBaselineBenchmarks() {
    console.log('🎯 Running Baseline Benchmarks with Pinned Golden Dataset');
    console.log('=========================================================\n');

    // Ensure output directory exists
    await fs.mkdir(this.outputDir, { recursive: true });

    // Load the pinned dataset
    console.log('📌 Loading pinned golden dataset...');
    const pinnedDataset = await this.pinnedLoader.loadPinnedDataset();
    
    console.log(`✅ Loaded pinned dataset: ${pinnedDataset.version}`);
    console.log(`   Items: ${pinnedDataset.total_items}`);
    console.log(`   Languages: ${Object.keys(pinnedDataset.language_distribution).join(', ')}`);
    console.log(`   Query classes: ${Object.keys(pinnedDataset.query_class_distribution).join(', ')}\n`);

    // Validate consistency with current corpus
    console.log('🔍 Validating pinned dataset consistency...');
    const consistencyResult = await this.pinnedLoader.validatePinnedDatasetConsistency();
    
    if (!consistencyResult.passed) {
      console.warn(`⚠️ Consistency check failed: ${consistencyResult.report.inconsistent_results} inconsistencies`);
      console.warn(`   Pass rate: ${(consistencyResult.report.pass_rate * 100).toFixed(1)}%`);
      console.warn(`   This may indicate corpus changes since pinning.`);
      
      // Write consistency report
      const consistencyReportPath = path.join(this.outputDir, 'consistency-report.json');
      await fs.writeFile(consistencyReportPath, JSON.stringify(consistencyResult.report, null, 2));
      console.warn(`   Detailed report: ${consistencyReportPath}\n`);
    } else {
      console.log(`✅ Consistency check passed: ${consistencyResult.report.pass_rate * 100}% aligned\n`);
    }

    // Create a custom GroundTruthBuilder that uses the pinned data
    const mockGroundTruthBuilder = this.createMockGroundTruthBuilder();

    // Initialize the benchmark suite runner
    const suiteRunner = new BenchmarkSuiteRunner(
      mockGroundTruthBuilder,
      this.outputDir,
      'nats://localhost:4222'
    );

    // Run SMOKE benchmark with pinned data
    console.log('🔥 Running SMOKE baseline benchmark...');
    const baselineConfig = {
      trace_id: `baseline-${Date.now()}`,
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
      subtokens: true
    };

    const smokeResult = await suiteRunner.runSmokeSuite(baselineConfig);
    
    console.log('\n✅ SMOKE baseline benchmark completed!');
    console.log(`   Status: ${smokeResult.status}`);
    console.log(`   Total queries: ${smokeResult.total_queries}`);
    console.log(`   Completed queries: ${smokeResult.completed_queries}`);
    console.log(`   Failed queries: ${smokeResult.failed_queries}`);

    // Generate baseline metrics report
    const baselineReport = await this.generateBaselineReport(pinnedDataset, smokeResult, consistencyResult);
    const reportPath = path.join(this.outputDir, `baseline-report-${pinnedDataset.version}.md`);
    await fs.writeFile(reportPath, baselineReport);

    console.log('\n📄 Generated baseline report:', reportPath);

    // Save baseline results for future comparison
    const baselineResultsPath = path.join(this.outputDir, `baseline-${pinnedDataset.version}.json`);
    const baselineData = {
      pinned_dataset_version: pinnedDataset.version,
      benchmark_results: smokeResult,
      consistency_check: consistencyResult,
      generated_at: new Date().toISOString()
    };
    await fs.writeFile(baselineResultsPath, JSON.stringify(baselineData, null, 2));

    console.log('💾 Saved baseline data:', baselineResultsPath);
    
    console.log('\n🎯 Baseline Establishment Complete!');
    console.log('====================================');
    console.log('This baseline can now be used for:');
    console.log('1. Performance regression detection');
    console.log('2. Comparative benchmarking');
    console.log('3. TODO.md validation against stable metrics');

    return baselineData;
  }

  createMockGroundTruthBuilder() {
    const pinnedLoader = this.pinnedLoader;
    
    return {
      get currentGoldenItems() {
        return pinnedLoader.getCurrentGoldenItems();
      },
      
      get currentSnapshots() {
        return pinnedLoader.getCurrentSnapshots();
      },
      
      generateConfigFingerprint(config, seedSet) {
        return pinnedLoader.generateConfigFingerprint(config, seedSet);
      },
      
      filterGoldenItemsBySlice(sliceTags) {
        return pinnedLoader.filterGoldenItemsBySlice(sliceTags);
      }
    };
  }

  async generateBaselineReport(pinnedDataset, benchmarkResult, consistencyResult) {
    const metrics = benchmarkResult.metrics;
    
    return `# Baseline Benchmark Report

## Pinned Dataset Information

- **Dataset Version**: ${pinnedDataset.version}
- **Git SHA**: ${pinnedDataset.git_sha}  
- **Total Items**: ${pinnedDataset.total_items}
- **Pinned At**: ${pinnedDataset.pinned_at}

## Consistency Check Results

- **Pass Rate**: ${(consistencyResult.report.pass_rate * 100).toFixed(1)}%
- **Valid Results**: ${consistencyResult.report.valid_results}/${consistencyResult.report.total_expected_results}
- **Status**: ${consistencyResult.passed ? '✅ PASSED' : '❌ FAILED'}
- **Corpus Files**: ${consistencyResult.report.corpus_file_count}

## Baseline Benchmark Results

### Summary
- **Status**: ${benchmarkResult.status}
- **System**: ${benchmarkResult.system}
- **Total Queries**: ${benchmarkResult.total_queries}
- **Completed Queries**: ${benchmarkResult.completed_queries}
- **Failed Queries**: ${benchmarkResult.failed_queries}

### Metrics
- **Recall@10**: ${metrics.recall_at_10.toFixed(4)}
- **Recall@50**: ${metrics.recall_at_50.toFixed(4)}
- **NDCG@10**: ${metrics.ndcg_at_10.toFixed(4)}
- **MRR**: ${metrics.mrr.toFixed(4)}
- **First Relevant Tokens**: ${metrics.first_relevant_tokens}

### Latencies
- **E2E P50**: ${metrics.stage_latencies.e2e_p50.toFixed(1)}ms
- **E2E P95**: ${metrics.stage_latencies.e2e_p95.toFixed(1)}ms
- **Stage A P95**: ${metrics.stage_latencies.stage_a_p95.toFixed(1)}ms
- **Stage B P95**: ${metrics.stage_latencies.stage_b_p95.toFixed(1)}ms
- **Stage C P95**: ${metrics.stage_latencies.stage_c_p95.toFixed(1)}ms

### Fan-out Sizes
- **Stage A**: ${metrics.fan_out_sizes.stage_a}
- **Stage B**: ${metrics.fan_out_sizes.stage_b}
- **Stage C**: ${metrics.fan_out_sizes.stage_c}

## Usage as Baseline

This baseline establishes the reference metrics for the pinned golden dataset ${pinnedDataset.version}.

**Key Metrics for Comparison:**
- **Target Recall@10**: ≥ ${metrics.recall_at_10.toFixed(4)}
- **Target NDCG@10**: ≥ ${metrics.ndcg_at_10.toFixed(4)}
- **Target E2E P95**: ≤ ${metrics.stage_latencies.e2e_p95.toFixed(1)}ms

**Regression Detection:**
- Use these metrics as thresholds for detecting performance regressions
- Compare future runs against this stable baseline
- Investigate any significant deviations from these values

## Next Steps

1. **Validate TODO.md Items**: Check if current performance meets TODO.md specifications
2. **Set CI Gates**: Use baseline metrics to configure CI performance gates
3. **Monitor Trends**: Track performance over time against this stable reference
4. **Update Baselines**: Re-pin dataset when corpus significantly changes

---

**Generated**: ${new Date().toISOString()}  
**Purpose**: Establish stable performance baseline using pinned golden dataset  
**Dataset**: ${pinnedDataset.version} (${pinnedDataset.total_items} items)
`;
  }

  async listBaselines() {
    try {
      const files = await fs.readdir(this.outputDir);
      const baselineFiles = files.filter(f => f.startsWith('baseline-') && f.endsWith('.json'));
      
      console.log('\n📊 Available Baselines:');
      console.log('=======================');
      
      for (const file of baselineFiles.sort()) {
        const filePath = path.join(this.outputDir, file);
        const stat = await fs.stat(filePath);
        
        try {
          const data = JSON.parse(await fs.readFile(filePath, 'utf-8'));
          console.log(`   ${file}`);
          console.log(`     Dataset Version: ${data.pinned_dataset_version}`);
          console.log(`     Generated: ${data.generated_at}`);
          console.log(`     Status: ${data.benchmark_results.status}`);
          console.log(`     Queries: ${data.benchmark_results.completed_queries}/${data.benchmark_results.total_queries}`);
          console.log('');
        } catch (error) {
          console.log(`   ${file} (parse error)`);
        }
      }
    } catch (error) {
      console.log('📁 No baseline results directory found');
    }
  }
}

// Main execution
async function main() {
  const runner = new BaselineBenchmarkRunner();
  
  const command = process.argv[2];
  
  try {
    if (command === 'list') {
      await runner.listBaselines();
    } else {
      await runner.runBaselineBenchmarks();
    }
  } catch (error) {
    console.error('❌ Error:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

if (import.meta.url.startsWith('file:') && process.argv[1] && import.meta.url.endsWith(path.basename(process.argv[1]))) {
  main();
}

export default BaselineBenchmarkRunner;