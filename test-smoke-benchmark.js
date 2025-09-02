#!/usr/bin/env node

/**
 * Test the SMOKE benchmark functionality by making a direct API call
 * This bypasses TypeScript compilation issues to verify the core functionality
 */

import fetch from 'node-fetch';

async function testSmokeBenchmark() {
  console.log('🧪 Testing SMOKE benchmark functionality...\n');
  
  // Check if Lens server is running
  console.log('🔍 Checking if Lens server is running...');
  
  try {
    const healthResponse = await fetch('http://localhost:3035/health', {
      method: 'GET',
      timeout: 5000
    });
    
    if (!healthResponse.ok) {
      throw new Error(`Health check failed: ${healthResponse.status}`);
    }
    
    console.log('✅ Lens server is running');
    
  } catch (error) {
    console.log('❌ Lens server is not running or not accessible');
    console.log('💡 Please start the server first with: npm start');
    return { success: false, error: 'Server not running' };
  }
  
  // Test SMOKE benchmark endpoint
  console.log('\n🎯 Testing SMOKE benchmark endpoint...');
  
  try {
    const benchmarkPayload = {
      suite: ['codesearch'],
      systems: ['lex', '+symbols'],
      slices: 'SMOKE_DEFAULT',
      seeds: 1,
      cache_mode: 'warm',
      trace_id: `test-${Date.now()}`,
      robustness: false,
      metamorphic: false
    };
    
    console.log('📤 Sending benchmark request...');
    
    const benchmarkResponse = await fetch('http://localhost:3035/bench/run', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(benchmarkPayload),
      timeout: 30000 // 30 second timeout
    });
    
    const result = await benchmarkResponse.json();
    
    if (!benchmarkResponse.ok) {
      throw new Error(`Benchmark failed: ${result.error || result.message || benchmarkResponse.status}`);
    }
    
    console.log('✅ Benchmark request completed');
    console.log(`🔍 Trace ID: ${result.trace_id}`);
    
    // Check if we got results
    if (result.benchmark_results) {
      const totalQueries = result.benchmark_results.total_queries || 0;
      console.log(`📊 Total queries executed: ${totalQueries}`);
      
      if (totalQueries === 0) {
        console.log('⚠️ No queries were executed - golden dataset may not be loaded');
        return { 
          success: false, 
          error: 'No queries executed', 
          totalQueries: 0 
        };
      }
      
      // Show key metrics if available
      if (result.benchmark_results.metrics) {
        const metrics = result.benchmark_results.metrics;
        console.log('🎯 Key metrics:');
        console.log(`   - Recall@10: ${(metrics.recall_at_10 * 100).toFixed(1)}%`);
        console.log(`   - Recall@50: ${(metrics.recall_at_50 * 100).toFixed(1)}%`);
        console.log(`   - nDCG@10: ${(metrics.ndcg_at_10 * 100).toFixed(1)}%`);
        
        if (metrics.stage_latencies) {
          console.log('⚡ Latencies:');
          console.log(`   - Stage A P95: ${metrics.stage_latencies.stage_a_p95}ms`);
          console.log(`   - Stage B P95: ${metrics.stage_latencies.stage_b_p95}ms`);
          console.log(`   - E2E P95: ${metrics.stage_latencies.e2e_p95}ms`);
        }
      }
      
      return {
        success: true,
        totalQueries: totalQueries,
        traceId: result.trace_id,
        metrics: result.benchmark_results.metrics
      };
      
    } else {
      console.log('⚠️ No benchmark results in response');
      console.log('Response:', JSON.stringify(result, null, 2));
      return { 
        success: false, 
        error: 'No benchmark results', 
        response: result 
      };
    }
    
  } catch (error) {
    console.error('❌ Benchmark test failed:', error.message);
    return { 
      success: false, 
      error: error.message 
    };
  }
}

// Run test if called directly
testSmokeBenchmark().then(result => {
  if (result.success) {
    console.log('\n🎉 SMOKE benchmark test passed!');
    console.log(`📊 Executed ${result.totalQueries} queries successfully`);
    console.log('✅ Golden dataset is properly loaded and functional');
    process.exit(0);
  } else {
    console.log('\n💥 SMOKE benchmark test failed');
    console.log(`❌ Error: ${result.error}`);
    if (result.totalQueries === 0) {
      console.log('\n💡 This suggests the golden dataset is not being loaded correctly');
      console.log('   Check the server logs for loading errors');
    }
    process.exit(1);
  }
}).catch(error => {
  console.error('💥 Unexpected error:', error);
  process.exit(1);
});

export default testSmokeBenchmark;