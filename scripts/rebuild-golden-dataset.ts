#!/usr/bin/env node

/**
 * Rebuild Golden Dataset for Python Corpus
 * Regenerates aligned golden dataset from current indexed Python files
 */

import path from 'path';
import { GroundTruthBuilder } from '../src/benchmark/ground-truth-builder.js';
import { BenchmarkSuiteRunner } from '../src/benchmark/suite-runner.js';

async function main() {
  console.log('🔄 Rebuilding golden dataset for Python corpus...');
  
  const workingDir = process.cwd();
  const outputDir = path.join(workingDir, 'benchmark-results');
  
  // Initialize components
  const groundTruthBuilder = new GroundTruthBuilder(workingDir, outputDir);
  const suiteRunner = new BenchmarkSuiteRunner(groundTruthBuilder, outputDir);
  
  try {
    // Step 1: Generate aligned Python dataset
    await groundTruthBuilder.generateAlignedPythonDataset();
    
    // Step 2: Validate corpus-golden consistency
    const consistencyResult = await suiteRunner.validateCorpusGoldenConsistency();
    
    if (consistencyResult.passed) {
      console.log('✅ Corpus-golden consistency validation PASSED');
      console.log(`📊 Dataset stats:`, {
        total_golden_items: consistencyResult.report.total_golden_items,
        valid_results: consistencyResult.report.valid_results,
        corpus_file_count: consistencyResult.report.corpus_file_count,
        pass_rate: (consistencyResult.report.pass_rate * 100).toFixed(1) + '%'
      });
    } else {
      console.log('❌ Corpus-golden consistency validation FAILED');
      console.log(`📊 Issues found:`, {
        inconsistent_results: consistencyResult.report.inconsistent_results,
        missing_patterns: consistencyResult.report.missing_patterns,
        pass_rate: (consistencyResult.report.pass_rate * 100).toFixed(1) + '%'
      });
      
      // Still continue - this helps us understand what was fixed
      console.log('ℹ️  Continuing anyway to show improvement...');
    }
    
    // Step 3: Show preview of generated data
    const goldenItems = groundTruthBuilder.currentGoldenItems;
    console.log(`\n📋 Generated dataset preview:`);
    console.log(`   Total items: ${goldenItems.length}`);
    
    // Group by query class
    const byQueryClass: Record<string, number> = {};
    const bySource: Record<string, number> = {};
    
    for (const item of goldenItems) {
      byQueryClass[item.query_class] = (byQueryClass[item.query_class] || 0) + 1;
      bySource[item.source] = (bySource[item.source] || 0) + 1;
    }
    
    console.log(`   By query class:`, byQueryClass);
    console.log(`   By source:`, bySource);
    
    // Show sample queries
    console.log(`\n🔍 Sample queries:`);
    for (let i = 0; i < Math.min(5, goldenItems.length); i++) {
      const item = goldenItems[i];
      console.log(`   ${i+1}. "${item.query}" (${item.query_class}) -> ${item.expected_results[0].file}:${item.expected_results[0].line}`);
    }
    
    console.log('\n✅ Golden dataset rebuild complete!');
    console.log('💡 You can now run benchmarks with aligned data that should show non-zero recall metrics.');
    
  } catch (error) {
    console.error('❌ Failed to rebuild golden dataset:', error);
    process.exit(1);
  }
}

if (require.main === module) {
  main().catch(console.error);
}