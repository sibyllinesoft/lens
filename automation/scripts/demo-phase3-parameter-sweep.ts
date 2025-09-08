#!/usr/bin/env bun
/**
 * Phase 3 Semantic Parameter Sweep Demo
 * Demonstrates systematic optimization of Stage-C semantic parameters
 */

import { LensSearchEngine } from './src/api/search-engine.js';
import { SemanticParameterSweep, PHASE3_SWEEP_CONFIG, type TestQuery } from './src/benchmark/semantic-parameter-sweep.js';

async function runPhase3Demo() {
  console.log('🚀 Phase 3: Semantic Stage Refinement Demo');
  console.log('🎯 Goal: Optimize Stage-C parameters for maximum precision with controlled latency\n');

  const searchEngine = new LensSearchEngine('./indexed-content');
  
  try {
    // Initialize search engine
    console.log('🔧 Initializing search engine...');
    await searchEngine.initialize();
    console.log('✅ Search engine ready\n');

    // Create parameter sweep instance
    const parameterSweep = new SemanticParameterSweep(searchEngine);

    // Run a quick demo sweep with focused parameters
    const demoSweepConfig = {
      ...PHASE3_SWEEP_CONFIG,
      nl_thresholds: [0.4, 0.5, 0.6],    // Focused range
      candidate_ks: [100, 150],           // Reduced combinations for demo
      ef_search_values: [32, 64],         // Focused ANN parameters
      confidence_cutoffs: [0.1, undefined], // Include no-cutoff option
      max_latency_increase_ms: 3,         // 3ms budget
      test_queries: generateDemoQueries(),
    };

    console.log('📊 Demo Configuration:');
    console.log(`   - NL thresholds: ${demoSweepConfig.nl_thresholds}`);
    console.log(`   - Candidate Ks: ${demoSweepConfig.candidate_ks}`);
    console.log(`   - efSearch values: ${demoSweepConfig.ef_search_values}`);
    console.log(`   - Confidence cutoffs: ${demoSweepConfig.confidence_cutoffs.map(c => c || 'none')}`);
    console.log(`   - Max latency increase: +${demoSweepConfig.max_latency_increase_ms}ms`);
    console.log(`   - Test queries: ${demoSweepConfig.test_queries.length}\n`);

    console.log('🧪 Starting parameter sweep...');
    const results = await parameterSweep.runParameterSweep(demoSweepConfig);

    console.log('\n📈 Parameter Sweep Results:');
    console.log('==========================');
    
    console.log(`🔍 Configurations tested: ${results.results.length}`);
    console.log(`🏆 Optimal configuration found:`);
    console.log(`   - NL threshold: ${results.optimal_config.nl_threshold}`);
    console.log(`   - Min candidates: ${results.optimal_config.min_candidates}`);
    console.log(`   - efSearch: ${results.optimal_config.efSearch}`);
    console.log(`   - Confidence cutoff: ${results.optimal_config.confidence_cutoff || 'none'}`);

    console.log(`\n📊 Performance improvements:`);
    console.log(`   - nDCG@10 improvement: ${results.improvement_summary.ndcg_improvement >= 0 ? '+' : ''}${results.improvement_summary.ndcg_improvement.toFixed(3)}`);
    console.log(`   - Stage-C p95 latency change: ${results.improvement_summary.p95_latency_change >= 0 ? '+' : ''}${results.improvement_summary.p95_latency_change.toFixed(1)}ms`);
    console.log(`   - Semantic trigger rate change: ${results.improvement_summary.semantic_precision_gain >= 0 ? '+' : ''}${(results.improvement_summary.semantic_precision_gain * 100).toFixed(1)}%`);

    // Display top 3 configurations
    console.log('\n🏅 Top 3 Configurations:');
    const sorted = results.results.sort((a, b) => b.quality_metrics.ndcg_at_10 - a.quality_metrics.ndcg_at_10).slice(0, 3);
    
    sorted.forEach((result, index) => {
      console.log(`\n${index + 1}. nDCG: ${result.quality_metrics.ndcg_at_10.toFixed(3)}, ` +
                  `Stage-C p95: ${result.latency_metrics.stage_c_p95.toFixed(1)}ms`);
      console.log(`   Config: NL=${result.config.nl_threshold}, K=${result.config.min_candidates}, ` +
                  `efSearch=${result.config.efSearch}, cutoff=${result.config.confidence_cutoff || 'none'}`);
      console.log(`   Semantic trigger rate: ${(result.quality_metrics.semantic_trigger_rate * 100).toFixed(1)}%`);
    });

    // Demonstrate API configuration
    console.log('\n🔧 Applying optimal configuration via API...');
    await searchEngine.updateSemanticConfig({
      nl_threshold: results.optimal_config.nl_threshold,
      min_candidates: results.optimal_config.min_candidates,
      efSearch: results.optimal_config.efSearch,
      confidence_cutoff: results.optimal_config.confidence_cutoff,
    });

    // Test a few queries with optimal configuration
    console.log('\n🔍 Testing queries with optimal configuration:');
    const testQueries = [
      'find functions that calculate mathematical operations',
      'function calculateSum',
      'how to implement search function'
    ];

    for (const query of testQueries) {
      const start = Date.now();
      const result = await searchEngine.search({
        trace_id: `demo-${Date.now()}`,
        repo_sha: 'lens-src',
        query,
        mode: 'hybrid',
        k: 10,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      });
      const latency = Date.now() - start;

      console.log(`\n   Query: "${query}"`);
      console.log(`   Results: ${result.hits.length}, Total latency: ${latency}ms`);
      console.log(`   Stage-C: ${result.stage_c_latency ? `${result.stage_c_latency}ms` : 'skipped'}`);
      console.log(`   Top hit: ${result.hits[0]?.file || 'none'} (score: ${result.hits[0]?.score.toFixed(2) || 'N/A'})`);
    }

    console.log('\n✅ Phase 3 parameter sweep demo completed!');
    console.log('\n📝 Summary:');
    console.log(`   - Successfully tested ${results.results.length} parameter combinations`);
    console.log(`   - Found optimal configuration balancing precision and latency`);
    console.log(`   - Applied configuration to search engine`);
    console.log(`   - Demonstrated real query performance with optimized parameters`);

    console.log('\n💡 Next steps:');
    console.log('   - Use PATCH /policy/stageC API to apply configuration in production');
    console.log('   - Monitor nDCG@10 and Stage-C latency metrics');
    console.log('   - Run periodic parameter sweeps as corpus evolves');
    console.log('   - Consider A/B testing with different configurations');

  } catch (error) {
    console.error('❌ Demo failed:', error);
    process.exit(1);
  } finally {
    await searchEngine.shutdown();
  }
}

function generateDemoQueries(): TestQuery[] {
  return [
    // Natural language queries (should trigger semantic)
    {
      query: "find functions that calculate mathematical operations",
      repo_sha: "lens-src",
      expected_relevance: 0.8,
      semantic_expected: true,
    },
    {
      query: "show me code that handles HTTP requests",
      repo_sha: "lens-src", 
      expected_relevance: 0.75,
      semantic_expected: true,
    },
    {
      query: "get components for user interface",
      repo_sha: "lens-src",
      expected_relevance: 0.7,
      semantic_expected: true,
    },
    
    // Programming syntax queries (should NOT trigger semantic)
    {
      query: "function search",
      repo_sha: "lens-src",
      expected_relevance: 0.9,
      semantic_expected: false,
    },
    {
      query: "class LensSearchEngine",
      repo_sha: "lens-src", 
      expected_relevance: 0.95,
      semantic_expected: false,
    },
    
    // Mixed cases
    {
      query: "how to implement search",
      repo_sha: "lens-src",
      expected_relevance: 0.7,
      semantic_expected: true,
    },
  ];
}

// Run the demo
runPhase3Demo().catch(console.error);