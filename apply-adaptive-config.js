#!/usr/bin/env node
/**
 * Apply Adaptive Configuration Patches
 * 
 * This script applies the TODO.md configuration patches to enable:
 * - Adaptive fan-out system with hardness scoring
 * - Work-conserving ANN reranker with early exit
 */

import { promises as fs } from 'fs';
import fetch from 'node-fetch';

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function waitForServer(apiUrl, maxAttempts = 30) {
  for (let i = 0; i < maxAttempts; i++) {
    try {
      const response = await fetch(`${apiUrl}/health`);
      if (response.ok) {
        console.log('✅ Server is ready');
        return true;
      }
    } catch (error) {
      // Server not ready yet
    }
    console.log(`⏳ Waiting for server... (${i + 1}/${maxAttempts})`);
    await sleep(2000);
  }
  return false;
}

async function applyAdaptiveConfiguration() {
  try {
    console.log('🎯 Applying Adaptive Configuration Patches...');
    
    // Try common ports for the API server
    const possiblePorts = [3000, 3001, 4000, 4001, 8000];
    let apiUrl = null;
    
    for (const port of possiblePorts) {
      try {
        const testUrl = `http://localhost:${port}`;
        const response = await fetch(`${testUrl}/health`, { timeout: 1000 });
        if (response.ok) {
          apiUrl = testUrl;
          break;
        }
      } catch (error) {
        // Try next port
      }
    }
    
    if (!apiUrl) {
      throw new Error('Could not find running server on common ports. Please start server first.');
    }
    
    console.log(`📡 API URL: ${apiUrl}`);
    
    // Wait for server to be ready
    const serverReady = await waitForServer(apiUrl);
    if (!serverReady) {
      throw new Error('Server is not responding after 60 seconds');
    }
    
    // Patch A — Adaptive fan-out & gates
    console.log('🔄 Applying Patch A: Adaptive fan-out & gates...');
    
    const stageAResponse = await fetch(`${apiUrl}/policy/stageA`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        k_candidates: "adaptive(180,380)",
        fanout_features: "+rare_terms,+fuzzy_edits,+id_entropy,+path_var,+cand_slope",
        adaptive_enabled: true
      })
    });
    
    if (stageAResponse.ok) {
      const stageAResult = await stageAResponse.json();
      console.log('✅ Stage A configuration applied:', stageAResult);
    } else {
      console.error('❌ Stage A configuration failed:', await stageAResponse.text());
    }
    
    const stageCGatesResponse = await fetch(`${apiUrl}/policy/stageC`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        gate: { 
          nl_threshold: "adaptive(0.55→0.30)", 
          min_candidates: "adaptive(8→14)" 
        },
        adaptive_gates_enabled: true
      })
    });
    
    if (stageCGatesResponse.ok) {
      const stageCGatesResult = await stageCGatesResponse.json();
      console.log('✅ Stage C gates configuration applied:', stageCGatesResult);
    } else {
      console.error('❌ Stage C gates configuration failed:', await stageCGatesResponse.text());
    }
    
    // Patch B — Work-conserving ANN with guarded early exit
    console.log('🔄 Applying Patch B: Work-conserving ANN...');
    
    const stageCResponse = await fetch(`${apiUrl}/policy/stageC`, {
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
    
    if (stageCResponse.ok) {
      const stageCResult = await stageCResponse.json();
      console.log('✅ Work-conserving ANN configuration applied:', stageCResult);
    } else {
      console.error('❌ Work-conserving ANN configuration failed:', await stageCResponse.text());
    }
    
    // Verify configuration
    console.log('🔍 Verifying adaptive system status...');
    
    const statusResponse = await fetch(`${apiUrl}/health`);
    if (statusResponse.ok) {
      const status = await statusResponse.json();
      console.log('📊 System Status:', status);
    }
    
    console.log('🎉 Adaptive system configuration completed!');
    console.log('');
    console.log('Next steps:');
    console.log('- Run smoke tests: node run-smoke-benchmark.js');
    console.log('- Monitor performance: check latency and quality metrics');
    console.log('- If issues occur, use rollback commands from TODO.md');
    
  } catch (error) {
    console.error('❌ Configuration failed:', error.message);
    console.error('Stack:', error.stack);
    
    console.log('');
    console.log('💡 Troubleshooting:');
    console.log('1. Ensure the server is running (npm run dev)');
    console.log('2. Check if ports are available');
    console.log('3. Verify the golden dataset exists');
    process.exit(1);
  }
}

// Run the configuration
applyAdaptiveConfiguration();