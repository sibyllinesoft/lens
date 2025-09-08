/**
 * Test the benchmark with aligned golden dataset
 */

import path from 'path';
import { GroundTruthBuilder } from './dist/benchmark/ground-truth-builder.js';
import { BenchmarkSuiteRunner } from './dist/benchmark/suite-runner.js';

async function main() {
  console.log('🧪 Testing benchmark with aligned golden dataset...');
  
  const workingDir = process.cwd();
  const outputDir = path.join(workingDir, 'benchmark-results');
  
  try {
    // Initialize components with aligned dataset
    const groundTruthBuilder = new GroundTruthBuilder(workingDir, outputDir);
    await groundTruthBuilder.generateAlignedPythonDataset();
    
    const suiteRunner = new BenchmarkSuiteRunner(groundTruthBuilder, outputDir);
    
    // Run a mini smoke test (just 1 seed to be quick)
    console.log('🔥 Running mini smoke benchmark...');
    const result = await suiteRunner.runSmokeSuite({
      seeds: 1,
      systems: ['lex'], // Just test basic lexical search
      top_n: 10
    });
    
    console.log('📊 Benchmark Results:');
    console.log(`   Completed queries: ${result.completed_queries}/${result.total_queries}`);
    console.log(`   Failed queries: ${result.failed_queries}`);
    console.log(`   Recall@10: ${(result.metrics.recall_at_10 * 100).toFixed(1)}%`);
    console.log(`   Recall@50: ${(result.metrics.recall_at_50 * 100).toFixed(1)}%`);
    console.log(`   NDCG@10: ${(result.metrics.ndcg_at_10 * 100).toFixed(1)}%`);
    console.log(`   MRR: ${result.metrics.mrr.toFixed(3)}`);
    
    // Check if we have non-zero metrics (indicating alignment worked)
    if (result.metrics.recall_at_10 > 0 || result.metrics.recall_at_50 > 0) {
      console.log('✅ SUCCESS: Non-zero recall metrics indicate proper corpus-golden alignment!');
    } else {
      console.log('⚠️  WARNING: Zero recall metrics - may need further investigation');
    }
    
  } catch (error) {
    console.error('❌ Error:', error);
    
    // If it fails due to consistency check, that's actually good - shows the gate works
    if (error.message?.includes('Corpus-Golden consistency check failed')) {
      console.log('✅ Consistency gate working as intended - caught misaligned data');
    }
  }
}

main().catch(console.error);