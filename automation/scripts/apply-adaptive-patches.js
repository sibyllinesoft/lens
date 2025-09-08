#!/usr/bin/env node

/**
 * Apply adaptive patches from TODO.md exactly as specified
 * 
 * Patch A - Adaptive fan-out & gates
 * Patch B - Work-conserving ANN with guarded early exit
 */

const API_BASE = process.env.API_BASE || 'http://localhost:3001';

function logTimestamp() {
  return new Date().toISOString();
}

function handleError(error, context) {
  console.error(`❌ ${context}: ${error.message}`);
  process.exit(1);
}

async function applyPatchA() {
  console.log(`\n🔧 [${logTimestamp()}] Applying Patch A - Adaptive fan-out & gates`);
  
  // PATCH /policy/stageA
  console.log('   📡 PATCH /policy/stageA');
  const stageAResponse = await fetch(`${API_BASE}/policy/stageA`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      k_candidates: "adaptive(180,380)",
      fanout_features: "+rare_terms,+fuzzy_edits,+id_entropy,+path_var,+cand_slope"
    })
  }).catch(error => handleError(error, 'Stage-A patch'));

  if (!stageAResponse.ok) {
    console.error(`❌ Stage-A patch failed: ${stageAResponse.status} ${stageAResponse.statusText}`);
    const errorText = await stageAResponse.text();
    console.error(`   Response: ${errorText}`);
    process.exit(1);
  }

  const stageAResult = await stageAResponse.json().catch(error => handleError(error, 'Stage-A response parsing'));
  console.log('   ✅ Stage-A adaptive fan-out configured');
  console.log(`   📊 Applied: k_candidates="adaptive(180,380)", fanout_features enabled`);

  // PATCH /policy/stageC gates
  console.log('   📡 PATCH /policy/stageC (adaptive gates)');
  const stageCGatesResponse = await fetch(`${API_BASE}/policy/stageC`, {
    method: 'PATCH', 
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      gate: {
        nl_threshold: "adaptive(0.55→0.30)",
        min_candidates: "adaptive(8→14)"
      }
    })
  }).catch(error => handleError(error, 'Stage-C gates patch'));

  if (!stageCGatesResponse.ok) {
    console.error(`❌ Stage-C gates patch failed: ${stageCGatesResponse.status} ${stageCGatesResponse.statusText}`);
    const errorText = await stageCGatesResponse.text(); 
    console.error(`   Response: ${errorText}`);
    process.exit(1);
  }

  const stageCGatesResult = await stageCGatesResponse.json().catch(error => handleError(error, 'Stage-C gates response parsing'));
  console.log('   ✅ Stage-C adaptive gates configured');
  console.log(`   📊 Applied: nl_threshold="adaptive(0.55→0.30)", min_candidates="adaptive(8→14)"`);

  return { stageA: stageAResult, stageCGates: stageCGatesResult };
}

async function applyPatchB() {
  console.log(`\n🧠 [${logTimestamp()}] Applying Patch B - Work-conserving ANN with guarded early exit`);
  
  // PATCH /policy/stageC ANN config
  console.log('   📡 PATCH /policy/stageC (work-conserving ANN)');
  const stageCResponse = await fetch(`${API_BASE}/policy/stageC`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json'
    },
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
  }).catch(error => handleError(error, 'Stage-C ANN patch'));

  if (!stageCResponse.ok) {
    console.error(`❌ Stage-C ANN patch failed: ${stageCResponse.status} ${stageCResponse.statusText}`);
    const errorText = await stageCResponse.text();
    console.error(`   Response: ${errorText}`);
    process.exit(1);
  }

  const stageCResult = await stageCResponse.json().catch(error => handleError(error, 'Stage-C ANN response parsing'));
  console.log('   ✅ Work-conserving ANN with guarded early exit configured');
  console.log(`   📊 Applied: k=220, dynamic efSearch, early_exit enabled`);

  return stageCResult;
}

async function verifyConfiguration() {
  console.log(`\n🔍 [${logTimestamp()}] Verifying adaptive configuration`);
  
  try {
    const response = await fetch(`${API_BASE}/policy/dump`);
    if (!response.ok) {
      console.warn(`⚠️ Policy dump failed: ${response.status} ${response.statusText}`);
      return false;
    }

    const config = await response.json();
    console.log('   ✅ Configuration dump retrieved successfully');
    
    // Log key adaptive settings
    if (config.stage_configurations) {
      console.log(`   📊 Policy version: ${config.policy_version}`);
      console.log(`   📊 Config fingerprint: ${config.config_fingerprint}`);
    }
    
    return true;
  } catch (error) {
    console.warn(`⚠️ Configuration verification failed: ${error.message}`);
    return false;
  }
}

async function main() {
  try {
    console.log('🚀 Applying adaptive patches from TODO.md');
    console.log(`   API Base: ${API_BASE}`);
    console.log(`   Started: ${logTimestamp()}`);

    // Apply patches in sequence
    const patchAResults = await applyPatchA();
    const patchBResults = await applyPatchB();
    
    // Verify configuration
    await verifyConfiguration();

    console.log(`\n✅ [${logTimestamp()}] Adaptive patches applied successfully!`);
    console.log('');
    console.log('📋 Next steps:');
    console.log('   1. Run SMOKE benchmark suite');
    console.log('   2. Verify pass gates (quality + safety metrics)'); 
    console.log('   3. If SMOKE passes → run Full benchmark');
    console.log('   4. Promote to v1.3-adaptive or execute rollback');
    console.log('');
    console.log('🔄 Rollback commands (if needed):');
    console.log('   PATCH /policy/stageA { k_candidates:320, fanout_features:"off" }');
    console.log('   PATCH /policy/stageC { gate:{ nl_threshold:0.35, min_candidates:8 }, ann:{ k:220, efSearch:96, early_exit:{ enabled:false } } }');

  } catch (error) {
    console.error(`❌ Fatal error: ${error.message}`);
    process.exit(1);
  }
}

main();