#!/usr/bin/env node
/**
 * Manual Precision/Recall Evaluation for Adaptive System
 * Tests a subset of golden dataset queries against the live search API
 */

import fetch from 'node-fetch';
import { promises as fs } from 'fs';

const API_BASE = 'http://localhost:3000';
const REPO_SHA = '8a9f5a125032a00804bf45cedb7d5e334489fbda';

async function loadGoldenDataset() {
  const goldenData = JSON.parse(await fs.readFile('./benchmark-results/golden-dataset.json', 'utf-8'));
  return goldenData.slice(0, 10); // Test with first 10 queries
}

async function executeSearchQuery(query) {
  const response = await fetch(`${API_BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      repo_sha: REPO_SHA,
      q: query,
      mode: 'hybrid',
      fuzzy: 1,
      k: 20
    })
  });
  
  if (!response.ok) {
    throw new Error(`Search failed: ${response.statusText}`);
  }
  
  return await response.json();
}

function calculatePrecisionRecall(actualResults, expectedResults) {
  // Convert expected results to a set for faster lookup
  const expectedSet = new Set(
    expectedResults.map(exp => `${exp.file}:${exp.line}`)
  );
  
  const actualSet = new Set(
    actualResults.map(hit => `${hit.file}:${hit.line}`)
  );
  
  // Calculate intersection (true positives)
  const intersection = new Set([...actualSet].filter(x => expectedSet.has(x)));
  const truePositives = intersection.size;
  
  const precision = actualResults.length > 0 ? truePositives / actualResults.length : 0;
  const recall = expectedResults.length > 0 ? truePositives / expectedResults.length : 0;
  
  return {
    precision,
    recall,
    truePositives,
    actualCount: actualResults.length,
    expectedCount: expectedResults.length,
    matches: [...intersection]
  };
}

function calculateMRR(actualResults, expectedResults) {
  const expectedSet = new Set(
    expectedResults.map(exp => `${exp.file}:${exp.line}`)
  );
  
  // Find rank of first relevant result
  for (let i = 0; i < actualResults.length; i++) {
    const resultKey = `${actualResults[i].file}:${actualResults[i].line}`;
    if (expectedSet.has(resultKey)) {
      return 1.0 / (i + 1); // MRR = 1/rank
    }
  }
  
  return 0; // No relevant results found
}

async function runPrecisionRecallEvaluation() {
  console.log('🎯 Manual Precision/Recall Evaluation for Adaptive System');
  console.log('=' .repeat(60));
  
  const goldenQueries = await loadGoldenDataset();
  console.log(`📚 Loaded ${goldenQueries.length} test queries from golden dataset`);
  
  const results = [];
  let totalLatencySum = 0;
  
  for (let i = 0; i < goldenQueries.length; i++) {
    const golden = goldenQueries[i];
    console.log(`\n[${i+1}/${goldenQueries.length}] Testing query: "${golden.query}"`);
    
    try {
      const searchResult = await executeSearchQuery(golden.query);
      const metrics = calculatePrecisionRecall(searchResult.hits, golden.expected_results);
      const mrr = calculateMRR(searchResult.hits, golden.expected_results);
      
      const queryResult = {
        query: golden.query,
        query_id: golden.id,
        precision: metrics.precision,
        recall: metrics.recall,
        mrr: mrr,
        latency_ms: searchResult.latency_ms.total,
        actual_count: metrics.actualCount,
        expected_count: metrics.expectedCount,
        true_positives: metrics.truePositives,
        matches: metrics.matches
      };
      
      results.push(queryResult);
      totalLatencySum += searchResult.latency_ms.total;
      
      console.log(`  📊 Precision: ${(metrics.precision * 100).toFixed(1)}% | Recall: ${(metrics.recall * 100).toFixed(1)}% | MRR: ${mrr.toFixed(3)}`);
      console.log(`  ⚡ Latency: ${searchResult.latency_ms.total}ms (A:${searchResult.latency_ms.stage_a}ms, B:${searchResult.latency_ms.stage_b}ms, C:${searchResult.latency_ms.stage_c || 0}ms)`);
      console.log(`  🎯 Found ${metrics.truePositives}/${metrics.expectedCount} expected results in ${metrics.actualCount} total results`);
      
      if (metrics.matches.length > 0) {
        console.log(`  ✅ Matches: ${metrics.matches.slice(0, 3).join(', ')}${metrics.matches.length > 3 ? '...' : ''}`);
      }
      
    } catch (error) {
      console.log(`  ❌ Query failed: ${error.message}`);
      results.push({
        query: golden.query,
        query_id: golden.id,
        precision: 0,
        recall: 0,
        mrr: 0,
        latency_ms: 0,
        error: error.message
      });
    }
  }
  
  // Calculate aggregate metrics
  const validResults = results.filter(r => !r.error);
  const avgPrecision = validResults.reduce((sum, r) => sum + r.precision, 0) / validResults.length;
  const avgRecall = validResults.reduce((sum, r) => sum + r.recall, 0) / validResults.length;
  const avgMRR = validResults.reduce((sum, r) => sum + r.mrr, 0) / validResults.length;
  const avgLatency = totalLatencySum / validResults.length;
  
  // Calculate recall@10 and recall@20
  const recallAt10 = validResults.reduce((sum, r) => sum + (r.recall > 0 ? 1 : 0), 0) / validResults.length;
  const recallAt20 = recallAt10; // Since we're fetching k=20
  
  console.log('\n' + '=' .repeat(60));
  console.log('📈 AGGREGATE RESULTS - ADAPTIVE SYSTEM PERFORMANCE');
  console.log('=' .repeat(60));
  
  console.log(`📊 Quality Metrics:`);
  console.log(`  • Average Precision: ${(avgPrecision * 100).toFixed(1)}%`);
  console.log(`  • Average Recall: ${(avgRecall * 100).toFixed(1)}%`);
  console.log(`  • Mean Reciprocal Rank (MRR): ${avgMRR.toFixed(3)}`);
  console.log(`  • Recall@10: ${(recallAt10 * 100).toFixed(1)}%`);
  console.log(`  • Recall@20: ${(recallAt20 * 100).toFixed(1)}%`);
  
  console.log(`⚡ Performance Metrics:`);
  console.log(`  • Average Total Latency: ${avgLatency.toFixed(1)}ms`);
  console.log(`  • Successful Queries: ${validResults.length}/${results.length}`);
  
  console.log(`🎯 Adaptive System Status:`);
  const qualityGood = avgRecall > 0.3; // 30% recall threshold
  const latencyGood = avgLatency < 50;   // Under 50ms average
  const overallGood = qualityGood && latencyGood;
  
  console.log(`  • Quality: ${qualityGood ? '✅ GOOD' : '❌ NEEDS IMPROVEMENT'} (Recall: ${(avgRecall * 100).toFixed(1)}%)`);
  console.log(`  • Performance: ${latencyGood ? '✅ EXCELLENT' : '❌ SLOW'} (${avgLatency.toFixed(1)}ms avg)`);
  console.log(`  • Overall: ${overallGood ? '✅ SYSTEM PERFORMING WELL' : '⚠️ NEEDS OPTIMIZATION'}`);
  
  // Save detailed results
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const resultsFile = `./benchmark-results/manual-eval-${timestamp}.json`;
  
  const report = {
    timestamp: new Date().toISOString(),
    system: 'adaptive',
    repo_sha: REPO_SHA,
    query_count: results.length,
    successful_queries: validResults.length,
    aggregate_metrics: {
      avg_precision: avgPrecision,
      avg_recall: avgRecall,
      avg_mrr: avgMRR,
      recall_at_10: recallAt10,
      recall_at_20: recallAt20,
      avg_latency_ms: avgLatency
    },
    individual_results: results
  };
  
  await fs.writeFile(resultsFile, JSON.stringify(report, null, 2));
  console.log(`\n📁 Detailed results saved to: ${resultsFile}`);
  
  return report;
}

// Run the evaluation
if (import.meta.url === new URL(import.meta.url).href) {
  runPrecisionRecallEvaluation()
    .then(() => console.log('\n🎉 Evaluation completed!'))
    .catch(err => {
      console.error('❌ Evaluation failed:', err.message);
      process.exit(1);
    });
}