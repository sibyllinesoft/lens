#!/usr/bin/env node
/**
 * Test script for adaptive fan-out and work-conserving ANN system
 * Verifies both Patch A and Patch B implementations
 */

import fetch from 'node-fetch';

const API_BASE = 'http://localhost:3000';

// Error handling wrapper
const handleError = (error, context) => {
  console.error(`❌ Error in ${context}:`, error.message);
  if (error.code === 'ECONNREFUSED') {
    console.error('💡 Hint: Make sure the server is running on port 8080');
    console.error('   Try: npm run dev or start your server first');
  }
  process.exit(1);
};

// Success logger
const logSuccess = (message, data) => {
  console.log(`✅ ${message}`);
  if (data) console.log('   Response:', JSON.stringify(data, null, 2));
};

async function testAdaptiveFanout() {
  console.log('🎯 Testing Patch A: Adaptive Fan-out & Gates...\n');

  // Test stageA adaptive configuration
  console.log('1. Configuring adaptive fan-out for Stage-A...');
  const stageAResponse = await fetch(`${API_BASE}/policy/stageA`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      k_candidates: "adaptive(180,380)",
      fanout_features: "+rare_terms,+fuzzy_edits,+id_entropy,+path_var,+cand_slope",
      rare_term_fuzzy: true,
      per_file_span_cap: 3
    })
  }).catch(error => handleError(error, 'Stage-A configuration'));

  const stageAResult = await stageAResponse.json().catch(error => handleError(error, 'Stage-A response parsing'));
  console.log('Stage-A response:', stageAResult);
  console.log();

  // Test stageC adaptive gates
  console.log('2. Configuring adaptive gates for Stage-C...');
  const stageCResponse = await fetch(`${API_BASE}/policy/stageC`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      gate: {
        nl_threshold: "adaptive(0.55→0.30)", 
        min_candidates: "adaptive(8→14)"
      }
    })
  });

  const stageCResult = await stageCResponse.json();
  console.log('Stage-C response:', stageCResult);
  console.log();

  return stageAResult.success && stageCResult.success;
}

async function testWorkConservingANN() {
  console.log('🧠 Testing Patch B: Work-conserving ANN with Early Exit...\n');

  console.log('1. Configuring work-conserving ANN...');
  const response = await fetch(`${API_BASE}/policy/stageC`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ann: {
        k: 220,
        efSearch: "dynamic(48 + 24*log2(1 + |candidates|/150))",
        early_exit: {
          after_probes: 64,
          margin_tau: 0.07,
          guards: {
            require_symbol_or_struct: true,
            min_top1_top5_margin: 0.14
          }
        }
      }
    })
  });

  const result = await response.json();
  console.log('Work-conserving ANN response:', result);
  console.log();

  return result.success;
}

async function runSampleQueries() {
  console.log('🔍 Testing adaptive system with sample queries...\n');

  const testQueries = [
    {
      query: "function getUserData",
      description: "Simple query (low hardness)"
    },
    {
      query: "complex_async_function_with_multiple_parameters",
      description: "Complex query (high hardness)"
    },
    {
      query: "rare_identifier_entropy_calculation_method_with_fuzzy_matching",
      description: "Very complex query (maximum hardness)"
    }
  ];

  for (const testQuery of testQueries) {
    console.log(`Query: "${testQuery.query}" (${testQuery.description})`);
    
    try {
      const searchResponse = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          q: testQuery.query,
          mode: 'hybrid',
          k: 20,
          repo_sha: 'test-repo'
        })
      });

      const searchResult = await searchResponse.json();
      
      if (searchResult.error) {
        console.log(`  ❌ Search failed: ${searchResult.error}`);
      } else {
        console.log(`  ✅ Results: ${searchResult.hits?.length || 0} hits`);
        console.log(`  ⏱️  Stage-A: ${searchResult.stage_a_latency}ms`);
        if (searchResult.stage_c_latency) {
          console.log(`  ⏱️  Stage-C: ${searchResult.stage_c_latency}ms`);
        }
      }
    } catch (error) {
      console.log(`  ❌ Query failed: ${error.message}`);
    }
    
    console.log();
  }
}

async function main() {
  console.log('🚀 Adaptive System Test Suite\n');
  console.log('=' .repeat(50));
  console.log('Debug: Script started, main function called');

  try {
    // Test Patch A
    const patchASuccess = await testAdaptiveFanout();
    console.log(`✅ Patch A (Adaptive Fan-out): ${patchASuccess ? 'CONFIGURED' : 'FAILED'}\n`);

    // Test Patch B  
    const patchBSuccess = await testWorkConservingANN();
    console.log(`✅ Patch B (Work-conserving ANN): ${patchBSuccess ? 'CONFIGURED' : 'FAILED'}\n`);

    // Run sample queries to test the system
    await runSampleQueries();

    console.log('=' .repeat(50));
    console.log('🎯 Adaptive system test completed!');
    
    if (patchASuccess && patchBSuccess) {
      console.log('✅ Both patches configured successfully');
      console.log('Ready for benchmark testing (Patch C)');
    } else {
      console.log('❌ Some patches failed to configure');
      process.exit(1);
    }

  } catch (error) {
    console.error('❌ Test suite failed:', error.message);
    process.exit(1);
  }
}

// Use pathToFileURL to handle path encoding correctly
import { pathToFileURL } from 'url';

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}