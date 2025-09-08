#!/usr/bin/env node

/**
 * Run SMOKE benchmark suite as specified in TODO.md
 */

import fs from 'fs';

const API_BASE = process.env.API_BASE || 'http://localhost:3000';

function logTimestamp() {
  return new Date().toISOString();
}

async function runSmokeBenchmark() {
  console.log(`\n🧪 [${logTimestamp()}] Running SMOKE benchmark suite`);
  
  const traceId = `smoke-adaptive-${Date.now()}`;
  const benchmarkRequest = {
    suite: ["codesearch", "structural"],
    systems: ["lex", "+symbols", "+symbols+semantic"],
    slices: "SMOKE_DEFAULT", 
    seeds: 1,
    cache_mode: "warm",
    trace_id: traceId
  };
  
  console.log(`   📋 Config:`, JSON.stringify(benchmarkRequest, null, 2));
  
  const response = await fetch(`${API_BASE}/bench/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(benchmarkRequest)
  });

  if (!response.ok) {
    throw new Error(`Benchmark failed: ${response.status} ${response.statusText}`);
  }

  const result = await response.json();
  console.log(`\n✅ SMOKE benchmark completed`);
  
  // Save and return results
  const resultsFile = `smoke-benchmark-${traceId}.json`;
  fs.writeFileSync(resultsFile, JSON.stringify(result, null, 2));
  console.log(`💾 Results saved: ${resultsFile}`);
  
  return result;
}

try {
  console.log('🧪 Starting SMOKE benchmark for adaptive system');
  const results = await runSmokeBenchmark();
  console.log(`\n🎯 SMOKE benchmark complete!`);
} catch (error) {
  console.error(`❌ Error: ${error.message}`);
  process.exit(1);
}
