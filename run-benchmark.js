import { BenchmarkSuiteRunner } from './dist/benchmark/suite-runner.js';
import { GroundTruthBuilder } from './dist/benchmark/ground-truth-builder.js';
import { promises as fs } from 'fs';

async function runBenchmark() {
  try {
    console.log('🔥 Initializing benchmark suite...');
    const groundTruthBuilder = new GroundTruthBuilder('./benchmark-results', 'lens');
    
    // Load the golden dataset we created manually
    const goldenDataPath = './benchmark-results/golden-dataset.json';
    console.log('📚 Loading golden dataset...');
    const goldenData = JSON.parse(await fs.readFile(goldenDataPath, 'utf-8'));
    
    // Set the golden items directly (since there's no loadGoldenDataset method)
    groundTruthBuilder.goldenItems = Array.isArray(goldenData) ? goldenData : [];
    console.log(`✅ Loaded ${groundTruthBuilder.goldenItems.length} golden items`);
    
    const runner = new BenchmarkSuiteRunner(groundTruthBuilder, './benchmark-results');
    
    console.log('🎯 Running smoke test with adaptive system...');
    // Skip corpus validation and run directly with a subset of queries for testing
    const testQueries = groundTruthBuilder.currentGoldenItems.slice(0, 10); // Just test with 10 queries
    console.log(`📊 Using ${testQueries.length} test queries for benchmark`);
    
    const result = await runner.executeBenchmarkRun({
      trace_id: 'test-adaptive-' + Date.now(),
      systems: ['lex', '+symbols+adaptive', '+symbols+semantic+adaptive'],
      k_candidates: 200,
      fuzzy: 2
    }, '+symbols+adaptive', testQueries);
    
    console.log('✅ Benchmark completed successfully!');
    console.log('📊 Results:', {
      trace_id: result.trace_id,
      system: result.system,
      completed_queries: result.completed_queries,
      failed_queries: result.failed_queries,
      status: result.status
    });
    
  } catch (error) {
    console.error('❌ Benchmark failed:', error.message);
    console.error('Stack:', error.stack);
  }
}

runBenchmark();