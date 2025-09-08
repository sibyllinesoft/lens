#!/usr/bin/env node

/**
 * Run SMOKE benchmark suite as specified in TODO.md
 * 
 * POST /bench/run {
 *   suite:["codesearch","structural"],
 *   systems:["lex","+symbols","+symbols+semantic"],
 *   slices:"SMOKE_DEFAULT",
 *   seeds:1, cache_mode:"warm", trace_id
 * }
 */

const API_BASE = process.env.API_BASE || 'http://localhost:3000';

function logTimestamp() {
  return new Date().toISOString();
}

function handleError(error, context) {
  console.error(`❌ ${context}: ${error.message}`);
  process.exit(1);
}

async function runSmokeBenchmark() {
  console.log(`\n🧪 [${logTimestamp()}] Running SMOKE benchmark suite`);
  console.log(`   API Base: ${API_BASE}`);
  
  const traceId = `smoke-adaptive-${Date.now()}`;
  
  const benchmarkRequest = {
    suite: ["codesearch", "structural"],
    systems: ["lex", "+symbols", "+symbols+semantic"],
    slices: "SMOKE_DEFAULT", 
    seeds: 1,
    cache_mode: "warm",
    trace_id: traceId
  };
  
  console.log(`   📋 Benchmark config:`, JSON.stringify(benchmarkRequest, null, 2));
  console.log(`   🔍 Trace ID: ${traceId}`);
  
  console.log(`\n📡 [${logTimestamp()}] POST /bench/run`);
  
  try {
    const response = await fetch(`${API_BASE}/bench/run`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(benchmarkRequest)
    });

    if (!response.ok) {
      console.error(`❌ Benchmark failed: ${response.status} ${response.statusText}`);
      const errorText = await response.text();
      console.error(`   Response: ${errorText}`);
      process.exit(1);
    }

    const result = await response.json();
    console.log(`\n✅ [${logTimestamp()}] SMOKE benchmark completed successfully`);
    
    // Display results summary
    if (result.results) {
      console.log('\n📊 Results Summary:');
      result.results.forEach((systemResult, idx) => {
        console.log(`   ${idx + 1}. ${systemResult.system}:`);
        if (systemResult.metrics) {
          console.log(`      Recall@50: ${systemResult.metrics.recall_at_50 || 'N/A'}`);
          console.log(`      nDCG@10: ${systemResult.metrics.ndcg_at_10 || 'N/A'}`);
          console.log(`      P95 Latency: ${systemResult.metrics.p95_latency_ms || 'N/A'}ms`);
          console.log(`      Success Rate: ${systemResult.metrics.success_rate || 'N/A'}`);
        }
      });
    }
    
    // Check pass gates as specified in TODO.md
    console.log('\n🚦 Pass Gates Check:');
    console.log('   Quality Gates (must hit ONE):');
    console.log('   • ΔRecall@50 ≥ +3%'); 
    console.log('   • ΔnDCG@10 ≥ +1.5% (p<0.05)');
    console.log('\n   Safety Gates (must hit ALL):');
    console.log('   • spans ≥ 98%');
    console.log('   • hard-negative leakage to top-5 ≤ +1.5% abs');
    console.log('   • p95 ≤ +15% vs v1.2 and p99 ≤ 2× p95');
    
    // Save results
    const resultsFile = `smoke-benchmark-${traceId}.json`;
    console.log(`\n💾 Saving results to: ${resultsFile}`);
    
    const fs = require('fs');
    fs.writeFileSync(resultsFile, JSON.stringify(result, null, 2));
    
    return result;
    
  } catch (error) {
    console.error(`❌ Benchmark request failed: ${error.message}`);
    process.exit(1);
  }
}

async function main() {
  try {
    console.log('🧪 Starting SMOKE benchmark suite for adaptive system validation');
    console.log(`   Started: ${logTimestamp()}`);
    
    const results = await runSmokeBenchmark();
    
    console.log(`\n🎯 [${logTimestamp()}] SMOKE benchmark completed!`);
    console.log('');
    console.log('📋 Next Steps:');
    console.log('   1. Analyze results against pass gates');
    console.log('   2. If gates pass → run Full benchmark (seeds=3, cold+warm)'); 
    console.log('   3. If gates fail → execute rollback');
    console.log('');
    console.log('🔍 Pass Gate Analysis:');
    console.log('   - Compare Recall@50 and nDCG@10 vs baseline');
    console.log('   - Verify spans coverage ≥ 98%'); 
    console.log('   - Check p95 latency increase ≤ +15%');
    console.log('   - Ensure p99 ≤ 2× p95');

  } catch (error) {
    console.error(`❌ Fatal error: ${error.message}`);
    process.exit(1);
  }
}

main();
