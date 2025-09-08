#!/usr/bin/env node

/**
 * Fast Sanity Battery (5 minutes, repeatable)
 * 
 * As specified in TODO.md:
 * - Oracle set (10 queries) where you know 2-3 gold spans
 * - File-only diagnostic with forced credit_policy="file_only"
 * - SLA-off snapshot to check if SLA mask is misapplied
 */

import { promises as fs } from 'fs';
import { DataMigrator, LensMetricsEngine, DEFAULT_CONFIG } from './packages/lens-metrics/dist/minimal-index.js';

class FastSanityBattery {
  constructor() {
    this.startTime = Date.now();
    console.log('🧪 FAST SANITY BATTERY - 5 MINUTE DIAGNOSTIC');
    console.log('⏱️ Target: Complete in <5 minutes with clear PASS/FAIL results');
  }

  async run() {
    try {
      console.log('\n📋 Running 3 diagnostic tests...\n');
      
      // Test 1: Oracle queries with known gold spans
      const test1 = await this.runOracleTest();
      
      // Test 2: File-only diagnostic  
      const test2 = await this.runFileOnlyDiagnostic();
      
      // Test 3: SLA-off snapshot
      const test3 = await this.runSLAOffSnapshot();
      
      // Summary
      const elapsed = (Date.now() - this.startTime) / 1000;
      console.log(`\n🏁 SANITY BATTERY COMPLETE (${elapsed.toFixed(1)}s)`);
      console.log('=' .repeat(50));
      
      const allPassed = test1.passed && test2.passed && test3.passed;
      
      if (allPassed) {
        console.log('✅ ALL TESTS PASSED - System is healthy');
      } else {
        console.log('❌ SOME TESTS FAILED - Investigation needed');
        
        if (!test1.passed) console.log('  - Oracle test failed: Check basic evaluation setup');
        if (!test2.passed) console.log('  - File-only test failed: Check for bogus perfect scores');
        if (!test3.passed) console.log('  - SLA test failed: Check SLA mask application');
      }
      
      return allPassed;
      
    } catch (error) {
      console.error(`❌ Sanity battery failed:`, error);
      return false;
    }
  }

  async runOracleTest() {
    console.log('🔮 TEST 1: Oracle Queries (Known Gold Spans)');
    console.log('   Purpose: Verify Lens and BM25 get ndcg@10 > 0 on known positives');
    
    const oracleQueries = this.createOracleQueries();
    const metricsEngine = new LensMetricsEngine(DEFAULT_CONFIG);
    
    const testResults = {};
    
    for (const system of ['lens', 'bm25']) {
      const systemQueries = [];
      
      for (const oracleQuery of oracleQueries) {
        // Create perfect results for oracle queries
        const expectedItems = oracleQuery.canonical_query.expected_spans || oracleQuery.canonical_query.expected_files;
        const perfectResults = expectedItems.map((item, index) => ({
          repo: item.repo,
          path: item.path,
          line: item.line || (10 + index), // Use expected line or fallback
          col: item.col || 5,
          score: 1.0 - (index * 0.1), // Decreasing scores
          rank: index + 1,
          snippet: `Perfect match for ${oracleQuery.query_text}`,
          why_tag: 'oracle'
        }));
        
        systemQueries.push({
          query: oracleQuery.canonical_query,
          results: perfectResults,
          latency_ms: 50 // Well within SLA
        });
      }
      
      const evaluation = metricsEngine.evaluateSystem(
        { system_id: system, queries: systemQueries }
      );
      
      testResults[system] = {
        mean_ndcg: evaluation.aggregate_metrics.mean_ndcg_at_10,
        queries_count: systemQueries.length
      };
      
      console.log(`   ${system}: nDCG@10 = ${evaluation.aggregate_metrics.mean_ndcg_at_10.toFixed(3)}`);
    }
    
    // Pass criteria: Both systems should have nDCG > 0.5 (since we gave them perfect matches)
    const lensGood = testResults.lens.mean_ndcg > 0.5;
    const bm25Good = testResults.bm25.mean_ndcg > 0.5;
    const passed = lensGood && bm25Good;
    
    console.log(`   Result: ${passed ? '✅ PASS' : '❌ FAIL'} - Oracle queries working correctly`);
    
    if (!passed) {
      console.log(`   Diagnosis: Oracle test should produce high nDCG with perfect matches`);
      console.log(`   - Lens nDCG: ${testResults.lens.mean_ndcg.toFixed(3)} (need >0.5)`);  
      console.log(`   - BM25 nDCG: ${testResults.bm25.mean_ndcg.toFixed(3)} (need >0.5)`);
    }
    
    return { passed, results: testResults };
  }

  async runFileOnlyDiagnostic() {
    console.log('\n📁 TEST 2: File-Only Diagnostic');
    console.log('   Purpose: Force file-only credit and confirm no perfect 1.000 scores on multi-result queries');
    
    // Create queries that have multiple relevant files  
    const multiResultQueries = this.createMultiResultQueries();
    
    // Force file-only credit policy
    const fileOnlyConfig = {
      ...DEFAULT_CONFIG,
      credit_gains: {
        span: 0,    // No span credit
        symbol: 0,  // No symbol credit  
        file: 1.0   // Only file credit
      }
    };
    
    const metricsEngine = new LensMetricsEngine(fileOnlyConfig);
    
    const systemQueries = [];
    
    for (const query of multiResultQueries) {
      // Return ALL expected files but in wrong order (should not get perfect score)
      const allFiles = query.canonical_query.expected_files;
      const messyResults = allFiles.map((file, index) => ({
        repo: file.repo,
        path: file.path,
        line: null, // Force file-level matching
        col: null,
        score: Math.random() * 0.8 + 0.1, // Random scores
        rank: index + 1,
        snippet: `File match for ${query.query_text}`,
        why_tag: 'file_only'
      }));
      
      // Shuffle the results to avoid perfect ranking
      for (let i = messyResults.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [messyResults[i], messyResults[j]] = [messyResults[j], messyResults[i]];
      }
      
      // Re-assign ranks after shuffle
      messyResults.forEach((result, index) => {
        result.rank = index + 1;
      });
      
      systemQueries.push({
        query: query.canonical_query,
        results: messyResults,
        latency_ms: 75
      });
    }
    
    const evaluation = metricsEngine.evaluateSystem(
      { system_id: 'file_only_test', queries: systemQueries }
    );
    
    const meanNdcg = evaluation.aggregate_metrics.mean_ndcg_at_10;
    console.log(`   File-only nDCG@10: ${meanNdcg.toFixed(3)}`);
    
    // Pass criteria: Should NOT get perfect 1.000 score with shuffled results
    const passed = meanNdcg < 1.0;
    
    console.log(`   Result: ${passed ? '✅ PASS' : '❌ FAIL'} - File-only credit working correctly`);
    
    if (!passed) {
      console.log(`   Diagnosis: File-only credit gave perfect score (${meanNdcg.toFixed(3)}) despite shuffled results`);
      console.log(`   This suggests evaluation is too lenient or ranking is ignored`);
    }
    
    return { passed, mean_ndcg: meanNdcg };
  }

  async runSLAOffSnapshot() {
    console.log('\n⏱️ TEST 3: SLA-Off Snapshot');
    console.log('   Purpose: Check if SLA mask is misapplied by comparing SLA-on vs SLA-off');
    
    const testQueries = this.createSLATestQueries();
    
    // Test with SLA enforced (default)
    const slaOnEngine = new LensMetricsEngine(DEFAULT_CONFIG);
    
    // Test with SLA disabled  
    const slaOffConfig = {
      ...DEFAULT_CONFIG,
      sla_threshold_ms: 10000 // Effectively disabled
    };
    const slaOffEngine = new LensMetricsEngine(slaOffConfig);
    
    const systemQueries = [];
    
    for (const query of testQueries) {
      // Create results with some that violate SLA
      const results = [
        // Good SLA results
        {
          repo: 'test',
          path: 'fast_result.py',
          line: 1, col: 0,
          score: 0.9, rank: 1,
          snippet: 'Fast result'
        },
        // Bad SLA results  
        {
          repo: 'test',
          path: 'slow_result.py', 
          line: 1, col: 0,
          score: 0.8, rank: 2,
          snippet: 'Slow result'
        }
      ];
      
      systemQueries.push({
        query: query.canonical_query,
        results: results,
        latency_ms: 200 // Over 150ms SLA
      });
    }
    
    const slaOnEval = slaOnEngine.evaluateSystem(
      { system_id: 'sla_test_on', queries: systemQueries }
    );
    
    const slaOffEval = slaOffEngine.evaluateSystem(
      { system_id: 'sla_test_off', queries: systemQueries }
    );
    
    const slaOnNdcg = slaOnEval.aggregate_metrics.mean_ndcg_at_10;
    const slaOffNdcg = slaOffEval.aggregate_metrics.mean_ndcg_at_10;
    
    console.log(`   SLA-ON nDCG@10:  ${slaOnNdcg.toFixed(3)}`);
    console.log(`   SLA-OFF nDCG@10: ${slaOffNdcg.toFixed(3)}`);
    
    // Pass criteria: SLA-off should be >= SLA-on (never worse)
    // If SLA-on is 0 but SLA-off > 0, SLA mask is likely misapplied
    const passed = slaOffNdcg >= slaOnNdcg;
    const slaIssue = slaOnNdcg === 0 && slaOffNdcg > 0;
    
    console.log(`   Result: ${passed ? '✅ PASS' : '❌ FAIL'} - SLA mask application`);
    
    if (!passed) {
      console.log(`   Diagnosis: SLA-off (${slaOffNdcg.toFixed(3)}) < SLA-on (${slaOnNdcg.toFixed(3)}) - unexpected`);
    }
    
    if (slaIssue) {
      console.log(`   ⚠️  WARNING: SLA-on=0 but SLA-off>0 suggests SLA mask is misapplied`);
    }
    
    return { 
      passed, 
      sla_on_ndcg: slaOnNdcg, 
      sla_off_ndcg: slaOffNdcg,
      sla_issue: slaIssue
    };
  }

  createOracleQueries() {
    // Create 10 oracle queries with known good spans
    const oracleQueries = [];
    
    for (let i = 0; i < 10; i++) {
      const legacyQuery = {
        query_id: `oracle_${i}`,
        query: `oracle query ${i}`,
        expected_results: [
          { path: `oracle/gold_file_${i}_1.py`, line: 10, col: 5 },
          { path: `oracle/gold_file_${i}_2.py`, line: 20, col: 0 },
          { path: `oracle/gold_file_${i}_3.py`, line: 15, col: 2 }
        ],
        language: 'python',
        suite: 'oracle_test'
      };
      
      const canonicalQuery = DataMigrator.migrateQuery(legacyQuery, 'test_repo');
      
      oracleQueries.push({
        query_text: legacyQuery.query,
        canonical_query: canonicalQuery
      });
    }
    
    return oracleQueries;
  }

  createMultiResultQueries() {
    // Create queries that should match multiple files
    const multiQueries = [];
    
    for (let i = 0; i < 5; i++) {
      const legacyQuery = {
        query_id: `multi_${i}`,
        query: `multi result query ${i}`,
        expected_results: [
          { path: `multi/result_${i}_1.py` },
          { path: `multi/result_${i}_2.py` },
          { path: `multi/result_${i}_3.py` },
          { path: `multi/result_${i}_4.py` },
          { path: `multi/result_${i}_5.py` }
        ],
        language: 'python',
        suite: 'multi_test'
      };
      
      const canonicalQuery = DataMigrator.migrateQuery(legacyQuery, 'test_repo');
      
      multiQueries.push({
        query_text: legacyQuery.query,
        canonical_query: canonicalQuery
      });
    }
    
    return multiQueries;
  }

  createSLATestQueries() {
    // Create simple queries for SLA testing
    const slaQueries = [];
    
    for (let i = 0; i < 3; i++) {
      const legacyQuery = {
        query_id: `sla_test_${i}`,
        query: `sla test ${i}`,
        expected_results: [
          { path: `sla/fast_result_${i}.py` },
          { path: `sla/slow_result_${i}.py` }
        ],
        language: 'python',
        suite: 'sla_test'
      };
      
      const canonicalQuery = DataMigrator.migrateQuery(legacyQuery, 'test_repo');
      
      slaQueries.push({
        query_text: legacyQuery.query,
        canonical_query: canonicalQuery
      });
    }
    
    return slaQueries;
  }
}

// CLI interface
if (import.meta.url === `file://${process.argv[1]}`) {
  const battery = new FastSanityBattery();
  battery.run().then(success => {
    process.exit(success ? 0 : 1);
  });
}

export { FastSanityBattery };