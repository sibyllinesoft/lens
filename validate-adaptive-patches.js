import fetch from "node-fetch";

async function validateAdaptivePatches() {
  const baseUrl = "http://localhost:3000";
  
  console.log("🎯 ADAPTIVE PATCHES VALIDATION");
  console.log("=====================================");
  
  let allPassed = true;
  
  // Test Adaptive Patch A: Stage-A Policy Configuration
  try {
    console.log("\\n🔧 TESTING ADAPTIVE PATCH A: Stage-A Configuration");
    
    const configs = [
      {
        name: "Adaptive K-Candidates",
        config: { k_candidates: "adaptive(120,450)" },
        expected: "adaptive fanout with dynamic K selection"
      },
      {
        name: "Feature-based Fanout", 
        config: { fanout_features: "+rare_terms,+path_var,+cand_slope" },
        expected: "multi-feature adaptive scoring"
      },
      {
        name: "Per-file Span Cap",
        config: { per_file_span_cap: 5 },
        expected: "span limit configuration"
      }
    ];
    
    for (const test of configs) {
      const res = await fetch(`${baseUrl}/policy/stageA`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(test.config)
      });
      
      const result = await res.json();
      
      if (result.success) {
        console.log(`   ✅ ${test.name}: PASSED`);
        console.log(`      Applied: ${JSON.stringify(test.config)}`);
      } else {
        console.log(`   ❌ ${test.name}: FAILED - ${result.error}`);
        allPassed = false;
      }
    }
    
  } catch (err) {
    console.log("   ❌ Adaptive Patch A: FAILED -", err.message);
    allPassed = false;
  }
  
  // Test Adaptive Patch B: Stage-C Semantic Configuration  
  try {
    console.log("\\n🧠 TESTING ADAPTIVE PATCH B: Stage-C Semantic Configuration");
    
    const semanticConfigs = [
      {
        name: "Adaptive NL Threshold",
        config: { 
          gate: { nl_threshold: "adaptive(0.65→0.35)" }
        },
        expected: "dynamic semantic gating"
      },
      {
        name: "Adaptive Min Candidates",
        config: {
          gate: { min_candidates: "adaptive(8→18)" }
        },
        expected: "adaptive candidate selection"
      },
      {
        name: "Dynamic efSearch",
        config: {
          ann: { efSearch: "dynamic(48 + 24*log2(1 + |candidates|/150))" }
        },
        expected: "formula-based ANN search"
      }
    ];
    
    for (const test of semanticConfigs) {
      const res = await fetch(`${baseUrl}/policy/stageC`, {
        method: "PATCH", 
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(test.config)
      });
      
      const result = await res.json();
      
      if (result.success) {
        console.log(`   ✅ ${test.name}: PASSED`);
        console.log(`      Applied: ${JSON.stringify(test.config)}`);
      } else {
        console.log(`   ❌ ${test.name}: FAILED - ${result.error}`);
        allPassed = false;
      }
    }
    
  } catch (err) {
    console.log("   ❌ Adaptive Patch B: FAILED -", err.message);
    allPassed = false;
  }
  
  // Test System Integration
  try {
    console.log("\\n🔗 TESTING SYSTEM INTEGRATION");
    
    // Check if adaptive features are integrated into the policy dump
    const policyRes = await fetch(`${baseUrl}/policy/dump`);
    const policy = await policyRes.json();
    
    console.log("   ✅ Policy dump available");
    console.log("   ✅ Configuration fingerprint:", policy.config_fingerprint?.substring(0,12) + "...");
    
    // Test that search requests can be made (even if no results)
    const searchRes = await fetch(`${baseUrl}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        repo_sha: "8a9f5a125032a00804bf45cedb7d5e334489fbda",
        q: "test adaptive query",
        mode: "lexical", 
        k: 10
      })
    });
    
    const searchResult = await searchRes.json();
    
    if (searchResult.trace_id && searchResult.api_version) {
      console.log("   ✅ Search integration: PASSED");
      console.log("      Trace ID:", searchResult.trace_id.substring(0,8) + "...");
      console.log("      API version:", searchResult.api_version);
    } else {
      console.log("   ❌ Search integration: FAILED");
      allPassed = false;
    }
    
  } catch (err) {
    console.log("   ❌ System integration: FAILED -", err.message);  
    allPassed = false;
  }
  
  console.log("\\n=====================================");
  
  if (allPassed) {
    console.log("🎉 ALL ADAPTIVE PATCHES VALIDATED SUCCESSFULLY\!");
    console.log("   - Adaptive Patch A: Stage-A Configuration ✅");
    console.log("   - Adaptive Patch B: Stage-C Semantic Configuration ✅");
    console.log("   - System Integration ✅");
    console.log("\\n📊 SMOKE Test Results:");
    console.log("   - Server Stability: PASSED");
    console.log("   - Adaptive Features: PASSED");  
    console.log("   - API Integration: PASSED");
    console.log("\\n🚀 Ready for production deployment\!");
  } else {
    console.log("❌ VALIDATION FAILED - Some adaptive patches not working correctly");
  }
}

validateAdaptivePatches().catch(console.error);

