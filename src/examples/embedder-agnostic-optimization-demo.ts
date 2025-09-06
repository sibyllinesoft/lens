/**
 * Embedder-Agnostic Optimization Demo
 * 
 * Demonstrates how to integrate the four embedder-agnostic optimization systems
 * into a production search pipeline. Shows configuration, usage patterns, and
 * monitoring integration.
 */

import { EmbedderAgnosticOptimizer } from '../core/embedder-agnostic-optimizer.js';
import type { SearchHit } from '../core/span_resolver/types.js';
import type { SearchContext, SymbolDefinition, SymbolReference } from '../types/core.js';

// Mock data for demonstration
const createDemoSearchHits = (): SearchHit[] => [
  {
    file: 'src/utils/stringHelpers.ts',
    line: 15,
    col: 0,
    score: 0.85,
    why: ['exact'],
    snippet: 'export function formatString(input: string): string {',
    symbol_kind: 'function',
    symbol_name: 'formatString'
  },
  {
    file: 'src/components/Header.tsx',
    line: 23,
    col: 4,
    score: 0.72,
    why: ['semantic'],
    snippet: 'const headerText = formatString(title);',
    symbol_name: 'formatString'
  },
  {
    file: 'tests/stringHelpers.test.ts',
    line: 8,
    col: 0,
    score: 0.68,
    why: ['fuzzy'],
    snippet: 'describe("formatString", () => {',
    symbol_name: 'formatString'
  },
  {
    file: 'src/utils/arrayHelpers.ts',
    line: 42,
    col: 0,
    score: 0.55,
    why: ['structural'],
    snippet: 'export function formatArray<T>(items: T[]): string[] {',
    symbol_kind: 'function',
    symbol_name: 'formatArray'
  },
  {
    file: 'docs/api.md',
    line: 120,
    col: 0,
    score: 0.45,
    why: ['semantic'],
    snippet: '## String Formatting Functions',
  }
];

const createDemoSymbolDefinitions = (): SymbolDefinition[] => [
  {
    name: 'formatString',
    kind: 'function',
    file_path: 'src/utils/stringHelpers.ts',
    line: 15,
    col: 0,
    scope: 'global',
    signature: '(input: string): string'
  },
  {
    name: 'formatArray',
    kind: 'function',
    file_path: 'src/utils/arrayHelpers.ts',
    line: 42,
    col: 0,
    scope: 'global',
    signature: '<T>(items: T[]): string[]'
  },
  {
    name: 'Header',
    kind: 'class',
    file_path: 'src/components/Header.tsx',
    line: 10,
    col: 0,
    scope: 'global',
    signature: 'React.Component'
  }
];

const createDemoSymbolReferences = (): SymbolReference[] => [
  {
    symbol_name: 'formatString',
    file_path: 'src/components/Header.tsx',
    line: 23,
    col: 4,
    context: 'const headerText = formatString(title);'
  },
  {
    symbol_name: 'formatString',
    file_path: 'tests/stringHelpers.test.ts',
    line: 8,
    col: 0,
    context: 'describe("formatString", () => {'
  },
  {
    symbol_name: 'formatArray',
    file_path: 'src/components/List.tsx',
    line: 15,
    col: 8,
    context: 'const formatted = formatArray(items);'
  }
];

/**
 * Production-ready configuration for embedder-agnostic optimizations
 */
function createProductionConfig() {
  return {
    enabled: true,
    indexVersion: '1.2.3', // Should be updated when index changes
    slaRecallThreshold: 0.0, // SLA-Recall@50 >= 0
    nDCGImprovementThreshold: 0.005, // +0.5pp improvement
    eceToleranceThreshold: 0.01, // <= 0.01 calibration error
    maxTotalLatencyMs: 8, // Total budget for all optimizations
    
    // Quality gates configuration
    enableQualityGates: true,
    qualityGateWindow: 1000, // Rolling window for quality metrics
    latencyRegressionThreshold: 15, // Max 15% latency increase
    
    // Component-specific configurations
    constraintAware: {
      enabled: true,
      alpha: 0.5, // Floor for exact matches (log-odds)
      maxLatencyMs: 2,
      auditFloorWins: true,
      exactMatchMinScore: 0.3,
      sameFileSymbolBoost: 0.2,
      structuralPatternBoost: 0.15
    },
    
    sliceChasing: {
      enabled: true,
      maxDepth: 2,
      maxNodes: 64, // K ≤ 64
      budgetMs: 1.2, // 0.3-1.2ms budget
      topicSimilarityThreshold: 0.3,
      enableVendorVeto: true,
      enableRaptorTopicLeash: true,
      maxEdgesPerNode: 10,
      rolloutPercentage: 25 // 25% rollout for NL+symbol queries
    },
    
    microCache: {
      enabled: true,
      shardCount: 16,
      ttlSeconds: 2, // 1-3s TTL
      maxEntriesPerShard: 100,
      slaHeadroomThresholdMs: 2, // Only use cache when tight on budget
      enableKMerge: true,
      kMergeRatio: 0.7,
      canonicalizationEnabled: true,
      spanInvariantCheck: true,
      maxCachedResults: 50
    },
    
    annHygiene: {
      enabled: true,
      enableVisitedSetReuse: true,
      enableHardwarePrefetch: true,
      enableBatchedTopK: true,
      enableHotTopicPrewarm: true,
      visitedSetPoolSize: 256,
      prefetchDistance: 2,
      batchSize: 32,
      hotTopicCount: 20,
      efSearchMultiplier: 1.2,
      maxLatencyBudgetMs: 2, // -1 to -2ms target
      upshiftTargetMin: 3, // 3% minimum upshift
      upshiftTargetMax: 7 // 7% maximum upshift
    }
  };
}

/**
 * Demonstration of embedder-agnostic optimization in a search pipeline
 */
async function demonstrateOptimization() {
  console.log('🚀 Embedder-Agnostic Search Optimization Demo');
  console.log('============================================\n');

  // Initialize optimizer with production configuration
  const optimizer = new EmbedderAgnosticOptimizer(createProductionConfig());

  // Simulate various search queries
  const queries = [
    { query: 'format string function', type: 'NL query' },
    { query: 'formatString', type: 'symbol lookup' },
    { query: 'how to format arrays', type: 'NL query' },
    { query: 'Header component', type: 'type search' },
    { query: 'format', type: 'general search' }
  ];

  const symbolDefinitions = createDemoSymbolDefinitions();
  const symbolReferences = createDemoSymbolReferences();

  console.log(`📊 Processing ${queries.length} demo queries with ${symbolDefinitions.length} symbols...`);

  // Process each query through the optimization pipeline
  for (let i = 0; i < queries.length; i++) {
    const { query, type } = queries[i]!;
    console.log(`\n${i + 1}. Query: "${query}" (${type})`);
    
    const context: SearchContext = {
      trace_id: `demo-${Date.now()}-${i}`,
      repo_sha: 'demo-sha',
      query,
      mode: 'hybrid' as any,
      k: 10,
      fuzzy_distance: 2,
      started_at: new Date(),
      stages: []
    };
    
    const originalHits = createDemoSearchHits();
    
    try {
      const result = await optimizer.optimize(
        originalHits,
        context,
        symbolDefinitions,
        symbolReferences
      );
      
      // Display optimization results
      console.log(`   Original hits: ${result.originalHits.length}`);
      console.log(`   Final hits: ${result.finalHits.length}`);
      console.log(`   Optimizations applied: ${result.optimizationsApplied.join(', ')}`);
      console.log(`   Cache status: ${result.cacheStatus}`);
      console.log(`   Total latency: ${result.latencyBreakdown.total.toFixed(3)}ms`);
      console.log(`   Latency breakdown:`);
      console.log(`     - Constraint-aware: ${result.latencyBreakdown.constraintAware.toFixed(3)}ms`);
      console.log(`     - Slice-chasing: ${result.latencyBreakdown.sliceChasing.toFixed(3)}ms`);
      console.log(`     - Micro-cache: ${result.latencyBreakdown.microCache.toFixed(3)}ms`);
      console.log(`     - ANN hygiene: ${result.latencyBreakdown.annHygiene.toFixed(3)}ms`);
      console.log(`   Quality metrics:`);
      console.log(`     - SLA-Recall: ${result.qualityMetrics.slaRecall.toFixed(3)}`);
      console.log(`     - nDCG: ${result.qualityMetrics.nDCG.toFixed(3)}`);
      console.log(`     - ECE: ${result.qualityMetrics.ece.toFixed(3)}`);
      
      if (result.constraints.violationsDetected > 0) {
        console.log(`   Constraints:`);
        console.log(`     - Violations detected: ${result.constraints.violationsDetected}`);
        console.log(`     - Violations corrected: ${result.constraints.violationsCorrected}`);
        console.log(`     - Monotonicity valid: ${result.constraints.monotonicityValid}`);
      }
      
      // Show top 3 results
      console.log(`   Top results:`);
      result.finalHits.slice(0, 3).forEach((hit, idx) => {
        console.log(`     ${idx + 1}. ${hit.file}:${hit.line} (score: ${hit.score.toFixed(3)}, why: ${hit.why.join(', ')})`);
      });
      
    } catch (error) {
      console.error(`   ❌ Optimization failed: ${error}`);
    }
    
    // Add small delay to demonstrate cache behavior
    if (i < queries.length - 1) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }
  
  // Show overall system statistics
  console.log('\n📈 System Statistics');
  console.log('===================');
  
  const stats = optimizer.getStats();
  console.log(`Total optimizations processed: ${stats.performance.total_optimizations}`);
  console.log(`Average latency: ${stats.performance.avg_latency_ms.toFixed(3)}ms`);
  
  console.log('\nOptimization distribution:');
  for (const [optimization, count] of Object.entries(stats.performance.optimization_distribution)) {
    console.log(`  - ${optimization}: ${count} times`);
  }
  
  console.log('\nQuality gates status:');
  for (const [metric, gate] of Object.entries(stats.quality_gates)) {
    const status = gate.status === 'pass' ? '✅' : gate.status === 'warning' ? '⚠️' : '❌';
    console.log(`  ${status} ${metric}: ${gate.current_value.toFixed(3)} (threshold: ${gate.threshold})`);
  }
  
  console.log('\nComponent statistics:');
  console.log(`  Constraint reranker violations: ${stats.components.constraint_reranker.violations_logged}`);
  console.log(`  Slice chasing graph nodes: ${stats.components.slice_chasing.graph_nodes}`);
  console.log(`  Micro-cache hit rate: ${(stats.components.micro_cache.hitRate * 100).toFixed(1)}%`);
  console.log(`  ANN hygiene searches: ${stats.components.ann_hygiene.performance.total_searches}`);
  
  const qualityGatesPass = optimizer.areQualityGatesPassing();
  console.log(`\n${qualityGatesPass ? '✅' : '❌'} Quality gates: ${qualityGatesPass ? 'PASSING' : 'FAILING'}`);
  
  if (!qualityGatesPass) {
    const failingGates = optimizer.getFailingQualityGates();
    console.log('Failing gates:');
    failingGates.forEach(gate => {
      console.log(`  - ${gate.metric}: ${gate.currentValue.toFixed(3)} vs ${gate.threshold}`);
    });
  }
  
  console.log('\n✨ Demo completed successfully!');
  console.log('\nThe embedder-agnostic optimization system provides:');
  console.log('• Monotone constraint-aware reranking for stable exact/structural match prioritization');
  console.log('• Symbol graph traversal for improved recall without new embeddings');
  console.log('• Intelligent result caching with semantic canonicalization');
  console.log('• HNSW algorithmic optimizations for faster ANN search');
  console.log('• Comprehensive quality monitoring with automatic gating');
  console.log('• Complete embedder independence - survives model swaps');
}

/**
 * A/B testing configuration example
 */
function demonstrateABTesting() {
  console.log('\n🧪 A/B Testing Configuration Demo');
  console.log('=================================');
  
  const optimizer = new EmbedderAgnosticOptimizer(createProductionConfig());
  
  // Experiment 1: More aggressive constraint floors
  const experimentConfig1 = {
    constraintAware: {
      alpha: 0.8, // Higher floor for exact matches
      exactMatchMinScore: 0.4,
      sameFileSymbolBoost: 0.3
    }
  };
  
  // Experiment 2: Reduced slice-chasing budget
  const experimentConfig2 = {
    sliceChasing: {
      budgetMs: 0.5, // Tighter budget
      maxNodes: 32, // Fewer nodes
      rolloutPercentage: 50 // Broader rollout
    }
  };
  
  // Experiment 3: Longer cache TTL
  const experimentConfig3 = {
    microCache: {
      ttlSeconds: 5, // Longer TTL
      slaHeadroomThresholdMs: 1 // Use cache more aggressively
    }
  };
  
  console.log('Available experiment configurations:');
  console.log('1. Aggressive constraint floors (alpha: 0.8, higher boosts)');
  console.log('2. Reduced slice-chasing budget (0.5ms, 32 nodes, 50% rollout)');
  console.log('3. Longer cache TTL (5s TTL, more aggressive usage)');
  
  // Apply configuration dynamically
  optimizer.updateConfig(experimentConfig1);
  console.log('✅ Applied experiment configuration 1');
  
  // In production, you would:
  // 1. Route traffic based on user ID % 100
  // 2. Track metrics separately for each configuration
  // 3. Use statistical testing to compare performance
  // 4. Gradually roll out winning configurations
}

/**
 * Monitoring and alerting integration example
 */
function demonstrateMonitoring() {
  console.log('\n📊 Production Monitoring Integration');
  console.log('==================================');
  
  const optimizer = new EmbedderAgnosticOptimizer({
    ...createProductionConfig(),
    enableQualityGates: true
  });
  
  // Example monitoring loop (would be called periodically in production)
  function monitorOptimizer() {
    const stats = optimizer.getStats();
    
    // Check quality gates
    if (!optimizer.areQualityGatesPassing()) {
      const failingGates = optimizer.getFailingQualityGates();
      console.log('🚨 ALERT: Quality gates failing!');
      
      for (const gate of failingGates) {
        console.log(`   ${gate.metric}: ${gate.currentValue.toFixed(3)} vs ${gate.threshold}`);
        
        // In production, you would send alerts to monitoring systems:
        // - PagerDuty for critical violations
        // - Slack for warnings
        // - Metrics to DataDog/Grafana
        // - Log structured events for analysis
      }
    }
    
    // Performance metrics
    const perfMetrics = stats.performance;
    console.log(`Performance: ${perfMetrics.total_optimizations} queries, avg ${perfMetrics.avg_latency_ms.toFixed(3)}ms`);
    
    // Component health checks
    const components = stats.components;
    console.log(`Cache hit rate: ${(components.micro_cache.hitRate * 100).toFixed(1)}%`);
    console.log(`Constraint violations: ${components.constraint_reranker.violations_logged}`);
    console.log(`Graph nodes: ${components.slice_chasing.graph_nodes}`);
    console.log(`ANN searches: ${components.ann_hygiene.performance.total_searches}`);
    
    // Suggested production integrations:
    console.log('\n📋 Production Integration Checklist:');
    console.log('• Set up automated quality gate monitoring');
    console.log('• Configure alerts for performance degradation');
    console.log('• Track component-specific metrics in dashboards');
    console.log('• Implement automated rollback on quality failures');
    console.log('• Set up A/B testing infrastructure');
    console.log('• Monitor cache hit rates and eviction patterns');
    console.log('• Track constraint violation patterns for tuning');
  }
  
  monitorOptimizer();
}

// Run the demonstration
if (import.meta.url === `file://${process.argv[1]}`) {
  (async () => {
    await demonstrateOptimization();
    demonstrateABTesting();
    demonstrateMonitoring();
  })().catch(console.error);
}

export {
  demonstrateOptimization,
  demonstrateABTesting,
  demonstrateMonitoring,
  createProductionConfig
};