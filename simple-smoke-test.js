#!/usr/bin/env node
/**
 * Simple smoke test for adaptive system policies
 * Tests if our Patch A and Patch B endpoint changes work
 */

import fetch from 'node-fetch';
import { pathToFileURL } from 'url';

const API_BASE = 'http://localhost:3000';

async function testPolicyEndpoints() {
  console.log('🎯 Smoke Test: Policy Configuration Endpoints\n');

  // Test 1: Stage-A adaptive policy
  console.log('1. Testing Stage-A adaptive policy configuration...');
  try {
    const response = await fetch(`${API_BASE}/policy/stageA`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        k_candidates: "adaptive(180,380)",
        fanout_features: "+rare_terms,+fuzzy_edits,+id_entropy,+path_var,+cand_slope",
        rare_term_fuzzy: true,
        per_file_span_cap: 3
      })
    });

    if (!response.ok) {
      console.log(`   ❌ Stage-A policy PATCH failed: ${response.status} ${response.statusText}`);
      return false;
    }

    const result = await response.json();
    console.log('   ✅ Stage-A adaptive policy configured successfully');
    console.log(`   📝 Response: ${JSON.stringify(result, null, 2)}`);
  } catch (error) {
    console.log(`   ❌ Stage-A policy error: ${error.message}`);
    return false;
  }

  console.log();

  // Test 2: Stage-C adaptive gates
  console.log('2. Testing Stage-C adaptive gates configuration...');
  try {
    const response = await fetch(`${API_BASE}/policy/stageC`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        nl_threshold: "adaptive(0.55,0.30)",
        min_candidates: "adaptive(8,14)",
        efSearch: "adaptive(48,200)",
        early_exit_enabled: true,
        margin_tau: 0.07
      })
    });

    if (!response.ok) {
      console.log(`   ❌ Stage-C policy PATCH failed: ${response.status} ${response.statusText}`);
      return false;
    }

    const result = await response.json();
    console.log('   ✅ Stage-C adaptive gates configured successfully');
    console.log(`   📝 Response: ${JSON.stringify(result, null, 2)}`);
  } catch (error) {
    console.log(`   ❌ Stage-C policy error: ${error.message}`);
    return false;
  }

  return true;
}

async function testHealthCheck() {
  console.log('🏥 Testing server health...');
  
  const endpoints = [
    '/health',
    '/api/health',
    '/status',
    '/',
  ];

  for (const endpoint of endpoints) {
    try {
      const response = await fetch(`${API_BASE}${endpoint}`);
      console.log(`   ${endpoint}: ${response.status} ${response.statusText}`);
      
      if (response.ok) {
        console.log('   ✅ Server is responding');
        return true;
      }
    } catch (error) {
      console.log(`   ${endpoint}: ${error.message}`);
    }
  }
  
  console.log('   ❌ Server not accessible');
  return false;
}

async function main() {
  console.log('🚀 Simple Smoke Test for Lens Adaptive System\n');
  console.log('=' .repeat(60));

  // Check server health first
  const serverHealthy = await testHealthCheck();
  console.log();

  if (!serverHealthy) {
    console.log('❌ SMOKE TEST FAILED: Server not accessible');
    console.log('💡 Please start the Lens server first:');
    console.log('   npm run dev  # or');
    console.log('   node dist/server.js');
    process.exit(1);
  }

  // Test our adaptive policy endpoints
  const policiesConfigured = await testPolicyEndpoints();
  console.log();

  console.log('=' .repeat(60));
  if (policiesConfigured) {
    console.log('✅ SMOKE TEST PASSED');
    console.log('🎯 Adaptive policies are configured and ready');
    console.log('🚀 Ready for full benchmark testing!');
  } else {
    console.log('❌ SMOKE TEST FAILED');
    console.log('💡 Check server logs for configuration errors');
    process.exit(1);
  }
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}