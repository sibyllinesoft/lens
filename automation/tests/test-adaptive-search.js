import fetch from "node-fetch";

async function testAdaptiveSearch() {
  const baseUrl = "http://localhost:3000";
  
  console.log("🧪 Testing Adaptive Patches - Direct API calls");
  
  // Test 1: Health check
  try {
    const healthRes = await fetch(`${baseUrl}/health`);
    const health = await healthRes.json();
    console.log("✅ Health check:", health.status);
  } catch (err) {
    console.log("❌ Health check failed:", err.message);
    return;
  }
  
  // Test 2: Configuration dump (should show adaptive features)
  try {
    const configRes = await fetch(`${baseUrl}/policy/dump`);
    const config = await configRes.json();
    console.log("✅ Policy dump retrieved");
    console.log("   - Stage A native scanner:", config.stage_configurations?.stage_a?.native_scanner);
    console.log("   - Stage B enabled:", config.stage_configurations?.stage_b?.enabled);
    console.log("   - Stage C enabled:", config.stage_configurations?.stage_c?.enabled);
  } catch (err) {
    console.log("❌ Policy dump failed:", err.message);
  }
  
  // Test 3: Try enabling adaptive features via API
  try {
    console.log("🔧 Testing Adaptive Configuration API...");
    
    // Enable adaptive fanout
    const stageARes = await fetch(`${baseUrl}/policy/stageA`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        k_candidates: "adaptive(180,380)",
        fanout_features: "+rare_terms,+fuzzy_edits,+id_entropy"
      })
    });
    
    const stageAResult = await stageARes.json();
    console.log("✅ Stage A adaptive config:", stageAResult.success);
    
    if (stageAResult.success) {
      console.log("🎯 Adaptive patches are responding to API calls");
      console.log("   Applied config:", JSON.stringify(stageAResult.applied_config, null, 2));
    }
    
  } catch (err) {
    console.log("❌ Adaptive config test failed:", err.message);
  }
  
  // Test 4: Basic search (even with no results, latency should be measurable)
  try {
    console.log("🔍 Testing basic search functionality...");
    
    const searchRes = await fetch(`${baseUrl}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        repo_sha: "8a9f5a125032a00804bf45cedb7d5e334489fbda",
        q: "function",
        mode: "lexical",
        k: 5
      })
    });
    
    const searchResult = await searchRes.json();
    console.log("✅ Search API responding");
    console.log("   - Total hits:", searchResult.total);
    console.log("   - Total latency:", searchResult.latency_ms.total, "ms");
    console.log("   - Stage A latency:", searchResult.latency_ms.stage_a, "ms");
    console.log("   - Stage B latency:", searchResult.latency_ms.stage_b, "ms");
    
    if (searchResult.latency_ms.total > 0) {
      console.log("🎯 Search engine is processing queries (adaptive patches likely working)");
    }
    
  } catch (err) {
    console.log("❌ Search test failed:", err.message);
  }
  
  console.log("\\n🧪 Adaptive patches test completed!");
}

testAdaptiveSearch().catch(console.error);

