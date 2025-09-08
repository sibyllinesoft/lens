#!/usr/bin/env bun
/**
 * Phase 2 Implementation Test Script
 * Validates the complete Phase 2 Recall Pack implementation
 */

const API_BASE_URL = 'http://localhost:3001';

/**
 * Test Phase 2 policy configuration
 */
async function testPolicyConfiguration() {
  console.log('\n🔧 Testing Phase 2 policy configuration...');
  
  try {
    // Apply Phase 2 policy deltas
    const policyResponse = await fetch(`${API_BASE_URL}/policy/stageA`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        rare_term_fuzzy: 'backoff',
        synonyms_when_identifier_density_below: 0.65,
        per_file_span_cap: 5,
        wand: {
          enabled: true,
          block_max: true,
          prune_aggressiveness: 'low',
          bound_type: 'max'
        }
      })
    });

    if (!policyResponse.ok) {
      throw new Error(`Policy update failed: ${policyResponse.statusText}`);
    }

    const policyResult = await policyResponse.json();
    console.log('✅ Policy configuration applied successfully');
    console.log(`   Applied config: ${JSON.stringify(policyResult.applied_config, null, 2)}`);
    
    return { success: true, result: policyResult };
    
  } catch (error) {
    console.log('❌ Policy configuration failed:', error.message);
    return { success: false, error: error.message };
  }
}

/**
 * Test synonym mining
 */
async function testSynonymMining() {
  console.log('\n🔍 Testing PMI-based synonym mining...');
  
  try {
    const synonymResponse = await fetch(`${API_BASE_URL}/phase2/synonyms/mine`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        tau_pmi: 3.0,
        min_freq: 20,
        k_synonyms: 8,
        index_root: './indexed-content',
        output_dir: './test-synonyms'
      })
    });

    if (!synonymResponse.ok) {
      throw new Error(`Synonym mining failed: ${synonymResponse.statusText}`);
    }

    const synonymResult = await synonymResponse.json();
    console.log('✅ Synonym mining completed successfully');
    console.log(`   Generated ${synonymResult.synonym_table.entries.length} synonym entries`);
    console.log(`   Duration: ${synonymResult.duration_ms}ms`);
    console.log(`   Version: ${synonymResult.synonym_table.version}`);
    
    return { success: true, result: synonymResult };
    
  } catch (error) {
    console.log('❌ Synonym mining failed:', error.message);
    return { success: false, error: error.message };
  }
}

/**
 * Test path prior refitting
 */
async function testPathPriorRefitting() {
  console.log('\n🎯 Testing path prior refitting...');
  
  try {
    const pathPriorResponse = await fetch(`${API_BASE_URL}/phase2/pathprior/refit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        l2_regularization: 1.0,
        debias_low_priority_paths: true,
        max_deboost: 0.6,
        index_root: './indexed-content',
        output_dir: './test-path-priors'
      })
    });

    if (!pathPriorResponse.ok) {
      throw new Error(`Path prior refitting failed: ${pathPriorResponse.statusText}`);
    }

    const pathPriorResult = await pathPriorResponse.json();
    console.log('✅ Path prior refitting completed successfully');
    console.log(`   Model version: ${pathPriorResult.model.version}`);
    console.log(`   AUC-ROC: ${pathPriorResult.model.performance.auc_roc.toFixed(3)}`);
    console.log(`   Training accuracy: ${pathPriorResult.model.performance.training_accuracy.toFixed(3)}`);
    console.log(`   Duration: ${pathPriorResult.duration_ms}ms`);
    
    return { success: true, result: pathPriorResult };
    
  } catch (error) {
    console.log('❌ Path prior refitting failed:', error.message);
    return { success: false, error: error.message };
  }
}

/**
 * Test complete Phase 2 execution
 */
async function testPhase2Execution() {
  console.log('\n🎯 Testing complete Phase 2 execution...');
  
  try {
    console.log('⏳ This may take several minutes...');
    
    const phase2Response = await fetch(`${API_BASE_URL}/phase2/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        index_root: './indexed-content',
        output_dir: './test-phase2-results',
        api_base_url: API_BASE_URL
      })
    });

    if (!phase2Response.ok) {
      throw new Error(`Phase 2 execution failed: ${phase2Response.statusText}`);
    }

    const phase2Result = await phase2Response.json();
    console.log('✅ Phase 2 execution completed successfully');
    
    const results = phase2Result.results;
    console.log(`   Recall@50: ${results.baseline_recall_50.toFixed(3)} → ${results.new_recall_50.toFixed(3)} (${results.recall_improvement_pct >= 0 ? '+' : ''}${results.recall_improvement_pct.toFixed(1)}%)`);
    console.log(`   nDCG@10: ${results.baseline_ndcg_10.toFixed(3)} → ${results.new_ndcg_10.toFixed(3)} (${results.ndcg_change >= 0 ? '+' : ''}${results.ndcg_change.toFixed(3)})`);
    console.log(`   Span coverage: ${results.span_coverage_pct.toFixed(1)}%`);
    console.log(`   E2E p95: ${results.e2e_p95_ms.toFixed(1)}ms`);
    console.log(`   Acceptance gates: ${results.acceptance_gates_passed ? '✅ PASSED' : '❌ FAILED'}`);
    console.log(`   Tripwires: ${results.tripwires_status.toUpperCase()}`);
    console.log(`   Promotion ready: ${results.promotion_ready ? '✅ YES' : '❌ NO'}`);
    console.log(`   Duration: ${phase2Result.duration_ms}ms`);
    
    return { success: true, result: phase2Result };
    
  } catch (error) {
    console.log('❌ Phase 2 execution failed:', error.message);
    return { success: false, error: error.message };
  }
}

/**
 * Test search functionality with Phase 2 enhancements
 */
async function testEnhancedSearch() {
  console.log('\n🔍 Testing enhanced search functionality...');
  
  const testQueries = [
    'async function',
    'class Component', 
    'interface Config',
    'type SearchResult',
    'import React'
  ];
  
  const results = [];
  
  for (const query of testQueries) {
    try {
      console.log(`  Testing query: "${query}"`);
      
      const searchResponse = await fetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          q: query,
          repo_sha: 'lens-src',
          k: 10,
          mode: 'hybrid',
          fuzzy_distance: 2
        })
      });

      if (!searchResponse.ok) {
        throw new Error(`Search failed: ${searchResponse.statusText}`);
      }

      const searchResult = await searchResponse.json();
      results.push({
        query,
        hits: searchResult.hits.length,
        stage_a_latency: searchResult.stage_a_latency,
        stage_b_latency: searchResult.stage_b_latency,
        stage_c_latency: searchResult.stage_c_latency
      });
      
      console.log(`    ✅ ${searchResult.hits.length} hits, ${searchResult.stage_a_latency + (searchResult.stage_b_latency || 0)}ms total`);
      
    } catch (error) {
      console.log(`    ❌ Query "${query}" failed: ${error.message}`);
      results.push({ query, error: error.message });
    }
  }
  
  const successfulQueries = results.filter(r => !r.error);
  const avgLatency = successfulQueries.length > 0 
    ? successfulQueries.reduce((sum, r) => sum + (r.stage_a_latency || 0) + (r.stage_b_latency || 0), 0) / successfulQueries.length
    : 0;
    
  console.log(`✅ Enhanced search testing completed: ${successfulQueries.length}/${testQueries.length} queries successful`);
  console.log(`   Average latency: ${avgLatency.toFixed(1)}ms`);
  
  return { success: successfulQueries.length > 0, results, avgLatency };
}

/**
 * Test API health and connectivity
 */
async function testAPIHealth() {
  console.log('\n🏥 Testing API health...');
  
  try {
    const healthResponse = await fetch(`${API_BASE_URL}/health`);
    
    if (!healthResponse.ok) {
      throw new Error(`Health check failed: ${healthResponse.statusText}`);
    }
    
    const health = await healthResponse.json();
    console.log('✅ API is healthy');
    console.log(`   Status: ${health.status}`);
    console.log(`   Shards: ${health.shards_healthy}/${health.shards_total}`);
    console.log(`   Memory usage: ${health.memory_usage_gb.toFixed(2)}GB`);
    console.log(`   Active queries: ${health.active_queries}`);
    
    return { success: true, health };
    
  } catch (error) {
    console.log('❌ API health check failed:', error.message);
    return { success: false, error: error.message };
  }
}

/**
 * Main test execution function
 */
async function runPhase2Tests() {
  console.log('🎯 Phase 2 Recall Pack Implementation Test Suite');
  console.log('================================================\n');
  
  const startTime = Date.now();
  const testResults = [];
  
  // Test 1: API Health
  const healthResult = await testAPIHealth();
  testResults.push({ name: 'API Health', ...healthResult });
  
  if (!healthResult.success) {
    console.log('\n💥 API is not available. Please start the Lens server first:');
    console.log('   bun run src/server.ts');
    process.exit(1);
  }
  
  // Test 2: Policy Configuration
  const policyResult = await testPolicyConfiguration();
  testResults.push({ name: 'Policy Configuration', ...policyResult });
  
  // Test 3: Enhanced Search
  const searchResult = await testEnhancedSearch();
  testResults.push({ name: 'Enhanced Search', ...searchResult });
  
  // Test 4: Synonym Mining
  const synonymResult = await testSynonymMining();
  testResults.push({ name: 'Synonym Mining', ...synonymResult });
  
  // Test 5: Path Prior Refitting
  const pathPriorResult = await testPathPriorRefitting();
  testResults.push({ name: 'Path Prior Refitting', ...pathPriorResult });
  
  // Test 6: Complete Phase 2 Execution (optional, takes longer)
  const runFullTest = process.argv.includes('--full');
  if (runFullTest) {
    const phase2Result = await testPhase2Execution();
    testResults.push({ name: 'Complete Phase 2 Execution', ...phase2Result });
  } else {
    console.log('\n⚠️  Skipping complete Phase 2 execution test (use --full to include)');
  }
  
  // Summary
  const duration = Date.now() - startTime;
  const successCount = testResults.filter(r => r.success).length;
  const totalCount = testResults.length;
  
  console.log('\n📊 Test Results Summary');
  console.log('=======================');
  
  testResults.forEach(result => {
    const status = result.success ? '✅' : '❌';
    const errorText = result.error ? ` (${result.error})` : '';
    console.log(`  ${status} ${result.name}${errorText}`);
  });
  
  console.log(`\n🎯 Overall: ${successCount}/${totalCount} tests passed`);
  console.log(`⏱️  Total duration: ${duration}ms`);
  
  if (successCount === totalCount) {
    console.log('\n🎉 Phase 2 implementation is working correctly!');
    console.log('Ready for production deployment.');
    process.exit(0);
  } else {
    console.log('\n💥 Some tests failed. Please review the errors above.');
    console.log('🔧 Troubleshooting:');
    console.log('  1. Ensure Lens server is running on port 3001');
    console.log('  2. Verify indexed content exists in ./indexed-content');
    console.log('  3. Check server logs for detailed error information');
    console.log('  4. Ensure all dependencies are installed');
    process.exit(1);
  }
}

// CLI usage information
if (process.argv.includes('--help') || process.argv.includes('-h')) {
  console.log(`
🎯 Phase 2 Implementation Test Suite

USAGE:
  bun test-phase2-implementation.js [OPTIONS]

OPTIONS:
  --full      Include complete Phase 2 execution test (takes longer)
  --help, -h  Show this help message

PREREQUISITES:
  1. Start the Lens server: bun run src/server.ts
  2. Ensure indexed content exists in ./indexed-content
  3. Install dependencies: bun install

EXAMPLES:
  # Run basic tests
  bun test-phase2-implementation.js
  
  # Run all tests including full Phase 2 execution
  bun test-phase2-implementation.js --full
  
This script validates the Phase 2 Recall Pack implementation including:
- API connectivity and health
- Policy configuration updates
- Enhanced search functionality
- PMI-based synonym mining
- Path prior refitting with gentler de-boosts
- Complete Phase 2 workflow execution (optional)
`);
  process.exit(0);
}

// Run tests if executed directly
if (import.meta.main) {
  runPhase2Tests().catch(error => {
    console.error('💥 Test execution failed:', error);
    process.exit(1);
  });
}