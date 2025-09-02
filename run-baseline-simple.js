#!/usr/bin/env node

/**
 * Simple Baseline Benchmark with Pinned Golden Dataset
 * 
 * This script establishes baseline metrics using the pinned golden dataset
 * without depending on the complex TypeScript benchmark infrastructure.
 */

import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { PinnedGroundTruthLoader } from './src/benchmark/pinned-ground-truth-loader.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class SimpleBaselineRunner {
  constructor() {
    this.workingDir = process.cwd();
    this.outputDir = path.join(this.workingDir, 'baseline-results');
    this.pinnedLoader = new PinnedGroundTruthLoader(this.workingDir);
  }

  async runSimpleBaseline() {
    console.log('🎯 Establishing Baseline with Pinned Golden Dataset');
    console.log('==================================================\n');

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
      console.log(`✅ Consistency check passed: ${(consistencyResult.report.pass_rate * 100).toFixed(1)}% aligned\n`);
    }

    // Get the golden items for analysis
    const goldenItems = this.pinnedLoader.getCurrentGoldenItems();
    const smokeItems = this.pinnedLoader.getSmokeDataset();
    const stats = this.pinnedLoader.getDatasetStats();

    console.log('📊 Dataset Analysis:');
    console.log(`   Total items: ${goldenItems.length}`);
    console.log(`   SMOKE items: ${smokeItems.length}`);
    console.log(`   Available slices: ${Object.keys(stats.slices).join(', ')}`);
    console.log(`   Languages: ${Object.keys(stats.languages).join(', ')}`);
    console.log(`   Query classes: ${Object.keys(stats.query_classes).join(', ')}\n`);

    // Analyze query distribution
    const queryAnalysis = this.analyzeQueries(goldenItems);
    
    console.log('🔍 Query Analysis:');
    console.log(`   Average query length: ${queryAnalysis.avgQueryLength.toFixed(1)} chars`);
    console.log(`   Unique queries: ${queryAnalysis.uniqueQueries}`);
    console.log(`   Average expected results per query: ${queryAnalysis.avgExpectedResults.toFixed(1)}`);
    console.log(`   Total expected results: ${queryAnalysis.totalExpectedResults}\n`);

    // Generate the baseline data package
    const baselineData = {
      pinned_dataset_version: pinnedDataset.version,
      git_sha: pinnedDataset.git_sha,
      established_at: new Date().toISOString(),
      consistency_check: consistencyResult,
      dataset_stats: stats,
      query_analysis: queryAnalysis,
      corpus_stats: pinnedDataset.corpus_stats,
      golden_items_count: goldenItems.length,
      smoke_items_count: smokeItems.length
    };

    // Save baseline data
    const baselineResultsPath = path.join(this.outputDir, `baseline-${pinnedDataset.version}.json`);
    await fs.writeFile(baselineResultsPath, JSON.stringify(baselineData, null, 2));

    // Generate baseline report
    const baselineReport = await this.generateBaselineReport(baselineData);
    const reportPath = path.join(this.outputDir, `baseline-report-${pinnedDataset.version}.md`);
    await fs.writeFile(reportPath, baselineReport);

    console.log('✅ Baseline Established Successfully!');
    console.log('====================================');
    console.log(`📊 Baseline data: ${baselineResultsPath}`);
    console.log(`📄 Baseline report: ${reportPath}`);
    
    if (consistencyResult.passed) {
      console.log('\n🎯 Ready for Benchmarking!');
      console.log('This pinned dataset can now be used for:');
      console.log('1. Consistent benchmark runs');
      console.log('2. Performance regression detection');
      console.log('3. TODO.md validation');
    } else {
      console.log('\n⚠️ Corpus Alignment Issues Detected');
      console.log('Consider:');
      console.log('1. Re-indexing the corpus');
      console.log('2. Re-pinning the golden dataset');
      console.log('3. Investigating path mapping issues');
    }

    return baselineData;
  }

  analyzeQueries(goldenItems) {
    const queries = goldenItems.map(item => item.query);
    const uniqueQueries = new Set(queries).size;
    const avgQueryLength = queries.reduce((sum, q) => sum + q.length, 0) / queries.length;
    
    const expectedResults = goldenItems.map(item => item.expected_results.length);
    const avgExpectedResults = expectedResults.reduce((sum, count) => sum + count, 0) / expectedResults.length;
    const totalExpectedResults = expectedResults.reduce((sum, count) => sum + count, 0);

    return {
      totalQueries: goldenItems.length,
      uniqueQueries,
      avgQueryLength,
      avgExpectedResults,
      totalExpectedResults
    };
  }

  async generateBaselineReport(baselineData) {
    const { pinned_dataset_version, git_sha, established_at, consistency_check, dataset_stats, query_analysis, corpus_stats } = baselineData;
    
    return `# Pinned Dataset Baseline Report

## Overview

This report establishes the baseline metrics for the pinned golden dataset \`${pinned_dataset_version}\`. This dataset is now pinned and will be used consistently across all benchmark runs to ensure reproducible results.

## Pinned Dataset Information

- **Version**: ${pinned_dataset_version}
- **Git SHA**: ${git_sha}  
- **Established**: ${established_at}
- **Total Items**: ${baselineData.golden_items_count}
- **SMOKE Items**: ${baselineData.smoke_items_count}

## Corpus Consistency Check

- **Status**: ${consistency_check.passed ? '✅ PASSED' : '❌ FAILED'}
- **Pass Rate**: ${(consistency_check.report.pass_rate * 100).toFixed(1)}%
- **Valid Results**: ${consistency_check.report.valid_results}/${consistency_check.report.total_expected_results}
- **Corpus Files**: ${consistency_check.report.corpus_file_count}

${!consistency_check.passed ? `
### ⚠️ Consistency Issues

The pinned dataset has ${consistency_check.report.inconsistent_results} inconsistencies with the current corpus. This may indicate:

1. **Corpus changes** since the dataset was pinned
2. **Path mapping issues** between expected results and corpus files
3. **Missing files** that were expected in the golden dataset

**Recommendation**: ${consistency_check.report.pass_rate < 0.8 ? 'Re-pin the golden dataset' : 'Investigate path mapping'}
` : ''}

## Dataset Distribution

### Language Distribution
${Object.entries(dataset_stats.languages)
  .map(([lang, count]) => `- **${lang}**: ${count} items (${((count / baselineData.golden_items_count) * 100).toFixed(1)}%)`)
  .join('\n')}

### Query Class Distribution  
${Object.entries(dataset_stats.query_classes)
  .map(([queryClass, count]) => `- **${queryClass}**: ${count} items (${((count / baselineData.golden_items_count) * 100).toFixed(1)}%)`)
  .join('\n')}

### Slice Distribution
${Object.entries(dataset_stats.slices)
  .map(([slice, count]) => `- **${slice}**: ${count} items`)
  .join('\n')}

## Query Analysis

- **Total Queries**: ${query_analysis.totalQueries}
- **Unique Queries**: ${query_analysis.uniqueQueries} (${((query_analysis.uniqueQueries / query_analysis.totalQueries) * 100).toFixed(1)}% unique)
- **Average Query Length**: ${query_analysis.avgQueryLength.toFixed(1)} characters
- **Average Expected Results**: ${query_analysis.avgExpectedResults.toFixed(1)} per query
- **Total Expected Results**: ${query_analysis.totalExpectedResults}

## Corpus Statistics

- **Total Files**: ${corpus_stats.total_files}
- **File Types**: ${JSON.stringify(corpus_stats.file_types, null, 2)}

## Baseline Establishment

### What This Baseline Provides

1. **Stable Reference**: All future benchmarks will use this exact dataset
2. **Reproducible Results**: Eliminates dataset drift between benchmark runs  
3. **Comparative Analysis**: Changes can be measured against this stable baseline
4. **Regression Detection**: Consistent baseline enables reliable performance monitoring

### Usage Instructions

To use this pinned dataset in benchmarks:

\`\`\`javascript
import { PinnedGroundTruthLoader } from './src/benchmark/pinned-ground-truth-loader.js';

const loader = new PinnedGroundTruthLoader();
await loader.loadPinnedDataset(); // Loads current pinned version
const goldenItems = loader.getCurrentGoldenItems();
const smokeItems = loader.getSmokeDataset(); // For SMOKE tests
\`\`\`

### Next Steps

1. **Run Benchmarks**: Use this pinned dataset for all benchmark runs
2. **Validate TODO.md**: Check if performance meets TODO.md specifications  
3. **Set CI Gates**: Configure CI using baseline metrics (when available)
4. **Monitor Performance**: Track changes against this stable reference

${!consistency_check.passed ? `
### ⚠️ Action Required

The consistency check failed with ${(consistency_check.report.pass_rate * 100).toFixed(1)}% pass rate. Before using this baseline for critical benchmarking:

1. **Review inconsistencies**: Check \`consistency-report.json\` for details
2. **Fix path mappings**: Update corpus or golden dataset paths
3. **Consider re-pinning**: If corpus has changed significantly

` : ''}

## Quality Assurance

- ✅ Dataset pinned with version control
- ✅ Consistency validation performed  
- ✅ Distribution analysis completed
- ✅ Query analysis documented
- ✅ Usage instructions provided

---

**Generated**: ${new Date().toISOString()}  
**Purpose**: Establish stable benchmark baseline using pinned golden dataset  
**Dataset**: ${pinned_dataset_version} (${baselineData.golden_items_count} items)
**Status**: ${consistency_check.passed ? 'Ready for production use' : 'Needs attention before production use'}
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
        
        try {
          const data = JSON.parse(await fs.readFile(filePath, 'utf-8'));
          console.log(`   ${file}`);
          console.log(`     Dataset Version: ${data.pinned_dataset_version}`);
          console.log(`     Established: ${data.established_at}`);
          console.log(`     Items: ${data.golden_items_count} (${data.smoke_items_count} SMOKE)`);
          console.log(`     Consistency: ${data.consistency_check.passed ? '✅ PASSED' : '❌ FAILED'} (${(data.consistency_check.report.pass_rate * 100).toFixed(1)}%)`);
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
  const runner = new SimpleBaselineRunner();
  
  const command = process.argv[2];
  
  try {
    if (command === 'list') {
      await runner.listBaselines();
    } else {
      await runner.runSimpleBaseline();
    }
  } catch (error) {
    console.error('❌ Error:', error.message);
    if (process.env.DEBUG) {
      console.error(error.stack);
    }
    process.exit(1);
  }
}

if (import.meta.url.startsWith('file:') && process.argv[1] && import.meta.url.endsWith(path.basename(process.argv[1]))) {
  main();
}

export default SimpleBaselineRunner;