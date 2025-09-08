#!/usr/bin/env node

import { PinnedGroundTruthLoader } from './src/benchmark/pinned-ground-truth-loader.js';

console.log('🎯 Running SMOKE Benchmark with Pinned Golden Dataset');
console.log('='.repeat(55));

async function runPinnedSmokeBenchmark() {
  try {
    // Load pinned dataset
    console.log('\n📌 Loading pinned golden dataset...');
    const loader = new PinnedGroundTruthLoader();
    await loader.loadPinnedDataset();
    
    const currentInfo = loader.getCurrentDatasetInfo();
    console.log(`✅ Loaded pinned dataset: ${currentInfo.version}`);
    console.log(`   Items: ${currentInfo.total_items}`);
    console.log(`   Pinned at: ${currentInfo.created_at}`);
    
    // Validate consistency
    console.log('\n🔍 Validating dataset consistency...');
    const consistencyResult = await loader.validateConsistency();
    
    if (!consistencyResult.passed) {
      console.error('❌ Consistency check failed:', consistencyResult.report.inconsistent_results, 'issues');
      return;
    }
    
    console.log(`✅ Consistency: ${(consistencyResult.report.pass_rate * 100).toFixed(1)}% (${consistencyResult.report.valid_results}/${consistencyResult.report.total_expected_results})`);
    
    // Run SMOKE benchmark
    console.log('\n🚀 Running SMOKE benchmark...');
    
    const traceId = `pinned-smoke-${Date.now()}`;
    const payload = {
      suite: ['codesearch', 'structural'],
      systems: ['lex', '+symbols', '+symbols+semantic'], 
      slices: 'SMOKE_DEFAULT',
      seeds: 1,
      cache_mode: 'warm',
      trace_id: traceId,
      use_pinned_dataset: true
    };
    
    console.log(`📊 Trace ID: ${traceId}`);
    console.log('📋 Configuration:');
    console.log(`   Suite: ${payload.suite.join(', ')}`);
    console.log(`   Systems: ${payload.systems.join(', ')}`);
    console.log(`   Slices: ${payload.slices}`);
    console.log(`   Seeds: ${payload.seeds}`);
    console.log(`   Cache: ${payload.cache_mode}`);
    
    const response = await fetch('http://localhost:3003/bench/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    console.log(`\n📊 Response Status: ${response.status}`);
    
    if (response.status === 200) {
      const result = await response.json();
      console.log('\n🎉 SMOKE BENCHMARK COMPLETED!');
      console.log('\n📈 Results:');
      console.log(`   Total Queries: ${result.total_queries || 'N/A'}`);
      console.log(`   Duration (ms): ${result.duration_ms || 'N/A'}`);
      console.log(`   Systems: ${result.systems_tested?.join(', ') || 'N/A'}`);
      console.log(`   Trace ID: ${result.trace_id || traceId}`);
      
      console.log('\n🚪 TODO.md Pass Gate Check:');
      if (result.pass_gates) {
        console.log('   Quality Gates:', result.pass_gates.quality ? '✅ PASSED' : '❌ FAILED');
        console.log('   Safety Gates:', result.pass_gates.safety ? '✅ PASSED' : '❌ FAILED');
        
        if (result.metrics) {
          console.log('\n📊 Key Metrics:');
          console.log(`   Recall@50 Δ: ${result.metrics.recall_delta || 'N/A'}%`);
          console.log(`   nDCG@10 Δ: ${result.metrics.ndcg_delta || 'N/A'}%`);
          console.log(`   Spans: ${result.metrics.spans || 'N/A'}%`);
          console.log(`   Latency: ${result.metrics.latency_delta || 'N/A'}%`);
        }
        
        if (result.pass_gates.quality && result.pass_gates.safety) {
          console.log('\n🎉 PROMOTION GATES: ✅ PASSED');
          console.log('   Ready to promote to v1.3-adaptive');
        } else {
          console.log('\n❌ PROMOTION GATES: FAILED');
          console.log('   Execute TODO.md rollback if needed');
        }
      }
      
      if (result.report_path) {
        console.log(`\n📄 Full Report: ${result.report_path}`);
      }
      
    } else {
      const error = await response.text();
      console.log(`\n❌ Benchmark Failed (Status ${response.status}):`);
      console.log(error);
      
      // Try to parse error for more details
      try {
        const errorObj = JSON.parse(error);
        if (errorObj.message) {
          console.log(`   Error: ${errorObj.message}`);
        }
      } catch (e) {
        // Error text not JSON, already printed above
      }
    }
    
  } catch (error) {
    console.error('\n❌ Error running pinned SMOKE benchmark:', error.message);
  }
}

// Handle server not ready
const serverReady = await fetch('http://localhost:3003/health').then(() => true).catch(() => false);
if (!serverReady) {
  console.error('❌ Server not ready on port 3003');
  console.log('   Start server with: PORT=3003 node dist/server.js');
  process.exit(1);
}

runPinnedSmokeBenchmark();