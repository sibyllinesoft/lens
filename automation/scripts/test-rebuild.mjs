/**
 * Quick test of the golden dataset rebuild functionality
 */

import path from 'path';
import { GroundTruthBuilder } from './dist/benchmark/ground-truth-builder.js';
import { BenchmarkSuiteRunner } from './dist/benchmark/suite-runner.js';

async function main() {
  console.log('🔄 Testing golden dataset rebuild for Python corpus...');
  
  const workingDir = process.cwd();
  const outputDir = path.join(workingDir, 'benchmark-results');
  
  try {
    // Initialize components
    const groundTruthBuilder = new GroundTruthBuilder(workingDir, outputDir);
    const suiteRunner = new BenchmarkSuiteRunner(groundTruthBuilder, outputDir);
    
    // Step 1: Generate aligned Python dataset
    await groundTruthBuilder.generateAlignedPythonDataset();
    
    // Step 2: Validate corpus-golden consistency
    const consistencyResult = await suiteRunner.validateCorpusGoldenConsistency();
    
    console.log('✅ Consistency check result:', {
      passed: consistencyResult.passed,
      total_items: consistencyResult.report.total_golden_items,
      valid_results: consistencyResult.report.valid_results,
      pass_rate: (consistencyResult.report.pass_rate * 100).toFixed(1) + '%'
    });
    
    if (!consistencyResult.passed) {
      console.log('📊 Missing patterns:', consistencyResult.report.missing_patterns);
    }
    
    // Step 3: Show preview
    const goldenItems = groundTruthBuilder.currentGoldenItems;
    console.log(`\n📋 Generated ${goldenItems.length} golden items`);
    
    const sampleQueries = goldenItems.slice(0, 3);
    for (let i = 0; i < sampleQueries.length; i++) {
      const item = sampleQueries[i];
      console.log(`   ${i+1}. "${item.query}" (${item.query_class}) -> ${item.expected_results[0].file}:${item.expected_results[0].line}`);
    }
    
  } catch (error) {
    console.error('❌ Error:', error);
    process.exit(1);
  }
}

main().catch(console.error);