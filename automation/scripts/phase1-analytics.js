#!/usr/bin/env node

/**
 * Phase 1 Analytics - Manual Baseline Performance Collection
 * Bypasses the broken benchmark system to establish core metrics
 */

const REPO_SHA = "8a9f5a125032a00804bf45cedb7d5e334489fbda";
const BASE_URL = "http://localhost:3000";

// Test queries for different search patterns
const TEST_QUERIES = [
  // Lexical matches
  { query: "function", mode: "lex", expected_type: "common_keyword" },
  { query: "SearchEngine", mode: "lex", expected_type: "identifier" },
  { query: "async", mode: "lex", expected_type: "common_keyword" },
  { query: "benchmark", mode: "lex", expected_type: "domain_term" },
  { query: "interface", mode: "lex", expected_type: "typescript_keyword" },
  
  // Structural patterns
  { query: "class definition", mode: "struct", expected_type: "structural" },
  { query: "function implementation", mode: "struct", expected_type: "structural" },
  { query: "type definition", mode: "struct", expected_type: "structural" },
  { query: "import statement", mode: "struct", expected_type: "structural" },
  { query: "export function", mode: "struct", expected_type: "structural" },
  
  // Hybrid queries
  { query: "search implementation", mode: "hybrid", expected_type: "semantic" },
  { query: "error handling", mode: "hybrid", expected_type: "semantic" },
  { query: "api endpoint", mode: "hybrid", expected_type: "semantic" },
  { query: "benchmark results", mode: "hybrid", expected_type: "semantic" },
  { query: "validate schema", mode: "hybrid", expected_type: "semantic" }
];

async function runSearch(query, mode, k = 50) {
  const requestData = {
    repo_sha: REPO_SHA,
    q: query,
    mode: mode,
    fuzzy: 2,
    k: k,
    timeout_ms: 1000
  };
  
  const response = await fetch(`${BASE_URL}/search`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestData)
  });
  
  if (!response.ok) {
    throw new Error(`Search failed: ${response.status} ${response.statusText}`);
  }
  
  return await response.json();
}

function analyzeResults(queryInfo, results) {
  const analysis = {
    query: queryInfo.query,
    mode: queryInfo.mode,
    expected_type: queryInfo.expected_type,
    total_hits: results.total,
    returned_hits: results.hits.length,
    latency_ms: results.latency_ms,
    has_results: results.hits.length > 0,
    
    // Why attribution analysis
    why_counts: {},
    score_distribution: [],
    stage_performance: {
      stage_a_ms: results.latency_ms.stage_a,
      stage_b_ms: results.latency_ms.stage_b, 
      stage_c_ms: results.latency_ms.stage_c || 0,
      total_ms: results.latency_ms.total
    }
  };
  
  // Analyze why attributions and scores
  for (const hit of results.hits) {
    analysis.score_distribution.push(hit.score);
    
    for (const why of hit.why) {
      analysis.why_counts[why] = (analysis.why_counts[why] || 0) + 1;
    }
  }
  
  return analysis;
}

function calculateSystemMetrics(analyses) {
  const metrics = {
    total_queries: analyses.length,
    successful_queries: analyses.filter(a => a.has_results).length,
    zero_result_queries: analyses.filter(a => !a.has_results).length,
    
    avg_latency: {
      stage_a: 0,
      stage_b: 0, 
      stage_c: 0,
      total: 0
    },
    
    latency_percentiles: {
      p50: { stage_a: 0, total: 0 },
      p95: { stage_a: 0, total: 0 },
      p99: { stage_a: 0, total: 0 }
    },
    
    why_attribution_summary: {},
    
    recall_proxy_metrics: {
      // These are proxies since we don't have golden truth
      lexical_hit_rate: 0,
      structural_hit_rate: 0,
      hybrid_hit_rate: 0,
      avg_results_per_query: 0
    }
  };
  
  // Calculate averages
  const totalLatencies = {
    stage_a: analyses.map(a => a.stage_performance.stage_a_ms),
    stage_b: analyses.map(a => a.stage_performance.stage_b_ms),
    stage_c: analyses.map(a => a.stage_performance.stage_c_ms),
    total: analyses.map(a => a.stage_performance.total_ms)
  };
  
  metrics.avg_latency.stage_a = totalLatencies.stage_a.reduce((a,b) => a+b, 0) / analyses.length;
  metrics.avg_latency.stage_b = totalLatencies.stage_b.reduce((a,b) => a+b, 0) / analyses.length;
  metrics.avg_latency.stage_c = totalLatencies.stage_c.reduce((a,b) => a+b, 0) / analyses.length;
  metrics.avg_latency.total = totalLatencies.total.reduce((a,b) => a+b, 0) / analyses.length;
  
  // Calculate percentiles
  const sortedStageA = [...totalLatencies.stage_a].sort((a,b) => a-b);
  const sortedTotal = [...totalLatencies.total].sort((a,b) => a-b);
  
  const p50Index = Math.floor(0.5 * analyses.length);
  const p95Index = Math.floor(0.95 * analyses.length);  
  const p99Index = Math.floor(0.99 * analyses.length);
  
  metrics.latency_percentiles.p50.stage_a = sortedStageA[p50Index];
  metrics.latency_percentiles.p50.total = sortedTotal[p50Index];
  metrics.latency_percentiles.p95.stage_a = sortedStageA[p95Index];
  metrics.latency_percentiles.p95.total = sortedTotal[p95Index];
  metrics.latency_percentiles.p99.stage_a = sortedStageA[p99Index];
  metrics.latency_percentiles.p99.total = sortedTotal[p99Index];
  
  // Aggregate why attributions
  for (const analysis of analyses) {
    for (const [why, count] of Object.entries(analysis.why_counts)) {
      metrics.why_attribution_summary[why] = (metrics.why_attribution_summary[why] || 0) + count;
    }
  }
  
  // Calculate recall proxy metrics
  const byMode = {
    lex: analyses.filter(a => a.mode === 'lex'),
    struct: analyses.filter(a => a.mode === 'struct'), 
    hybrid: analyses.filter(a => a.mode === 'hybrid')
  };
  
  metrics.recall_proxy_metrics.lexical_hit_rate = byMode.lex.filter(a => a.has_results).length / byMode.lex.length;
  metrics.recall_proxy_metrics.structural_hit_rate = byMode.struct.filter(a => a.has_results).length / byMode.struct.length;
  metrics.recall_proxy_metrics.hybrid_hit_rate = byMode.hybrid.filter(a => a.has_results).length / byMode.hybrid.length;
  metrics.recall_proxy_metrics.avg_results_per_query = analyses.reduce((sum, a) => sum + a.returned_hits, 0) / analyses.length;
  
  return metrics;
}

function generateDecisionAnalysis(metrics) {
  const decision = {
    focus: null,
    reasoning: [],
    confidence: 'high',
    key_findings: []
  };
  
  // Analyze hit rates as proxy for recall
  const lowHitRateThreshold = 0.7; // If <70% of queries return results, possible recall issue
  const avgHitRate = metrics.successful_queries / metrics.total_queries;
  
  decision.key_findings.push(`Overall hit rate: ${(avgHitRate * 100).toFixed(1)}%`);
  decision.key_findings.push(`Average results per query: ${metrics.recall_proxy_metrics.avg_results_per_query.toFixed(1)}`);
  decision.key_findings.push(`Stage A latency p95: ${metrics.latency_percentiles.p95.stage_a}ms`);
  
  // Decision logic based on available metrics
  if (avgHitRate < lowHitRateThreshold) {
    decision.focus = "recall";
    decision.reasoning.push(`Low hit rate (${(avgHitRate * 100).toFixed(1)}%) suggests recall limitations`);
  } else if (metrics.latency_percentiles.p95.total > 200) {
    decision.focus = "precision";  
    decision.reasoning.push(`High latency (p95: ${metrics.latency_percentiles.p95.total}ms) suggests too many candidates, focus on precision`);
  } else if (metrics.recall_proxy_metrics.avg_results_per_query > 30) {
    decision.focus = "precision";
    decision.reasoning.push(`High average result count (${metrics.recall_proxy_metrics.avg_results_per_query.toFixed(1)}) suggests precision issues`);
  } else {
    decision.focus = "precision";
    decision.reasoning.push("Reasonable hit rates and latency suggest system is recall-adequate, focus on ranking/precision");
    decision.confidence = 'medium';
  }
  
  return decision;
}

async function main() {
  console.log("🔄 Starting Phase 1 Analytics - Baseline Performance Collection");
  console.log(`📊 Testing ${TEST_QUERIES.length} queries across different modes\n`);
  
  const analyses = [];
  
  for (let i = 0; i < TEST_QUERIES.length; i++) {
    const queryInfo = TEST_QUERIES[i];
    console.log(`[${i+1}/${TEST_QUERIES.length}] Testing: "${queryInfo.query}" (${queryInfo.mode})`);
    
    try {
      const results = await runSearch(queryInfo.query, queryInfo.mode);
      const analysis = analyzeResults(queryInfo, results);
      analyses.push(analysis);
      
      console.log(`  ✅ ${analysis.returned_hits} hits in ${analysis.latency_ms.total}ms`);
      
      // Brief pause to avoid overwhelming the server
      await new Promise(resolve => setTimeout(resolve, 100));
      
    } catch (error) {
      console.log(`  ❌ Failed: ${error.message}`);
    }
  }
  
  console.log("\n📈 Calculating system metrics...");
  const systemMetrics = calculateSystemMetrics(analyses);
  
  console.log("\n🎯 Generating decision analysis...");
  const decision = generateDecisionAnalysis(systemMetrics);
  
  // Generate comprehensive report
  const report = {
    timestamp: new Date().toISOString(),
    trace_id: "phase1-analytics-baseline",
    phase: "Phase 1 - Analytics Pass",
    
    summary: {
      total_queries: systemMetrics.total_queries,
      successful_queries: systemMetrics.successful_queries,
      hit_rate: systemMetrics.successful_queries / systemMetrics.total_queries,
      avg_latency_ms: Math.round(systemMetrics.avg_latency.total),
      decision_focus: decision.focus
    },
    
    performance_metrics: systemMetrics,
    decision_analysis: decision,
    individual_query_results: analyses.slice(0, 5), // First 5 for brevity
    
    next_steps: decision.focus === "recall" ? [
      "Focus on Stage A recall improvements",
      "Analyze lexical matching coverage", 
      "Consider expanding fuzzy matching parameters",
      "Review corpus indexing completeness"
    ] : [
      "Focus on Stage B/C precision improvements",
      "Analyze ranking algorithm effectiveness", 
      "Consider semantic scoring improvements",
      "Review result filtering and ranking"
    ]
  };
  
  console.log("\n" + "=".repeat(60));
  console.log("🏁 PHASE 1 ANALYTICS RESULTS");
  console.log("=".repeat(60));
  console.log(`📊 Hit Rate: ${(report.summary.hit_rate * 100).toFixed(1)}%`);
  console.log(`⚡ Avg Latency: ${report.summary.avg_latency_ms}ms`);
  console.log(`🎯 Decision Focus: ${report.summary.decision_focus.toUpperCase()}`);
  console.log(`🔍 Key Findings:`);
  for (const finding of decision.key_findings) {
    console.log(`   • ${finding}`);
  }
  console.log(`💡 Reasoning:`);
  for (const reason of decision.reasoning) {
    console.log(`   • ${reason}`);
  }
  
  // Save detailed report
  const reportPath = `/media/nathan/Seagate Hub/Projects/lens/phase1-baseline-analytics-${Date.now()}.json`;
  await require('fs').promises.writeFile(reportPath, JSON.stringify(report, null, 2));
  console.log(`\n📄 Detailed report saved to: ${reportPath}`);
  
  return report;
}

// Run if called directly
if (require.main === module) {
  main().catch(console.error);
}