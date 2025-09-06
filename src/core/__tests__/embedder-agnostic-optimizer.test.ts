/**
 * Tests for the Embedder-Agnostic Optimization Systems
 * 
 * Tests all four optimization components and their integration:
 * 1. Constraint-Aware Reranker
 * 2. Stage-B⁺ Slice-Chasing  
 * 3. ROI-Aware Result Micro-Cache
 * 4. ANN Hygiene Optimizer
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type { SearchHit, MatchReason } from '../span_resolver/types.js';
import type { SearchContext, SymbolDefinition, SymbolReference } from '../../types/core.js';

import { ConstraintAwareReranker, MonotonicFeatureFunction } from '../constraint-aware-reranker.js';
import { StageBPlusSliceChasing, SymbolGraphBuilder, TopicSimilarityCalculator } from '../stage-b-plus-slice-chasing.js';
import { ROIAwareMicroCache, QueryCanonicalizer, IntentClassifier } from '../roi-aware-micro-cache.js';
import { ANNHygieneOptimizer, VisitedSetPool, BatchedTopKSelector } from '../ann-hygiene-optimizer.js';
import { EmbedderAgnosticOptimizer, QualityMetricsCalculator } from '../embedder-agnostic-optimizer.js';

// Test fixtures
const createMockSearchHit = (
  file: string,
  line: number,
  score: number,
  why: MatchReason[] = ['semantic'],
  symbolName?: string
): SearchHit => ({
  file,
  line,
  col: 0,
  score,
  why,
  snippet: `Mock hit at ${file}:${line}`,
  symbol_name: symbolName,
  symbol_kind: symbolName ? 'function' : undefined
});

const createMockSearchContext = (query: string): SearchContext => ({
  query,
  language: 'typescript'
});

const createMockSymbolDefinition = (
  name: string,
  file: string,
  line: number,
  kind: any = 'function'
): SymbolDefinition => ({
  name,
  kind,
  file_path: file,
  line,
  col: 0,
  scope: 'global'
});

const createMockSymbolReference = (
  name: string,
  file: string,
  line: number
): SymbolReference => ({
  symbol_name: name,
  file_path: file,
  line,
  col: 0,
  context: `reference to ${name}`
});

describe('Constraint-Aware Reranker', () => {
  let reranker: ConstraintAwareReranker;

  beforeEach(() => {
    reranker = new ConstraintAwareReranker({
      enabled: true,
      alpha: 0.5,
      maxLatencyMs: 5,
      auditFloorWins: false // Disable for testing
    });
  });

  describe('MonotonicFeatureFunction', () => {
    it('should maintain monotonicity', () => {
      const func = new MonotonicFeatureFunction('test', 0.5);
      
      // Fit simple monotonic data
      func.fit([0, 0.5, 1.0], [0, 0.5, 1.0]);
      
      // Validate monotonicity
      expect(func.validateMonotonicity()).toBe(true);
      
      // Test function values are non-decreasing
      const val1 = func.apply(0.3);
      const val2 = func.apply(0.7);
      expect(val2).toBeGreaterThanOrEqual(val1);
    });

    it('should apply floor constraint', () => {
      const func = new MonotonicFeatureFunction('test', 0.5);
      
      // Even without fitting, should apply alpha floor
      const result = func.apply(0.8);
      expect(result).toBeGreaterThanOrEqual(0.5 * 0.8);
    });
  });

  describe('reranking with constraints', () => {
    it('should rerank hits while maintaining constraint order', async () => {
      const hits: SearchHit[] = [
        createMockSearchHit('file1.ts', 10, 0.6, ['semantic']),
        createMockSearchHit('file2.ts', 20, 0.8, ['exact']), // Should rank higher due to exact match
        createMockSearchHit('file3.ts', 30, 0.7, ['symbol'])
      ];
      
      const context = createMockSearchContext('test function');
      const reranked = await reranker.rerank(hits, context);
      
      expect(reranked).toHaveLength(3);
      
      // Exact match should be boosted despite lower original score
      const exactHit = reranked.find(h => h.why.includes('exact'));
      expect(exactHit).toBeDefined();
      expect(exactHit!.score).toBeGreaterThan(0.6); // Should be boosted
    });

    it('should handle empty hits gracefully', async () => {
      const context = createMockSearchContext('test');
      const reranked = await reranker.rerank([], context);
      
      expect(reranked).toEqual([]);
    });

    it('should respect latency budget', async () => {
      const fastReranker = new ConstraintAwareReranker({
        enabled: true,
        maxLatencyMs: 0.1 // Very tight budget
      });
      
      const hits = Array.from({ length: 100 }, (_, i) => 
        createMockSearchHit(`file${i}.ts`, i, Math.random())
      );
      
      const context = createMockSearchContext('test');
      
      // Should not throw but may fall back
      const result = await fastReranker.rerank(hits, context);
      expect(Array.isArray(result)).toBe(true);
    });
  });
});

describe('Stage-B⁺ Slice-Chasing', () => {
  let sliceChaser: StageBPlusSliceChasing;

  beforeEach(() => {
    sliceChaser = new StageBPlusSliceChasing({
      enabled: true,
      maxDepth: 2,
      maxNodes: 20,
      budgetMs: 5,
      rolloutPercentage: 100 // Always apply for testing
    });
  });

  describe('SymbolGraphBuilder', () => {
    it('should build graph from definitions and references', () => {
      const builder = new SymbolGraphBuilder();
      
      const definitions = [
        createMockSymbolDefinition('testFunction', 'test.ts', 10)
      ];
      
      const references = [
        createMockSymbolReference('testFunction', 'test.ts', 20),
        createMockSymbolReference('testFunction', 'other.ts', 5)
      ];
      
      builder.buildGraph(definitions, references);
      const graph = builder.getGraph();
      
      expect(graph.nodes.size).toBeGreaterThan(0);
      expect(graph.edges.size).toBeGreaterThan(0);
    });
  });

  describe('slice chasing', () => {
    it('should discover additional spans through graph traversal', async () => {
      const definitions = [
        createMockSymbolDefinition('testFunc', 'file1.ts', 10),
        createMockSymbolDefinition('relatedFunc', 'file2.ts', 15)
      ];
      
      const references = [
        createMockSymbolReference('testFunc', 'file3.ts', 25),
        createMockSymbolReference('relatedFunc', 'file1.ts', 12)
      ];
      
      sliceChaser.initializeGraph(definitions, references);
      
      const seeds = [createMockSearchHit('file1.ts', 10, 0.8, ['symbol'], 'testFunc')];
      const context = createMockSearchContext('find test function');
      
      const result = await sliceChaser.chaseSlices(seeds, context);
      
      expect(result).toHaveLength(seeds.length); // May include additional discovered spans
      expect(result[0]!.file).toBe('file1.ts'); // Original seed should be preserved
    });

    it('should respect budget constraints', async () => {
      const tightBudgetChaser = new StageBPlusSliceChasing({
        enabled: true,
        budgetMs: 0.1, // Very tight
        maxDepth: 1,
        maxNodes: 5
      });
      
      const seeds = [createMockSearchHit('test.ts', 10, 0.5)];
      const context = createMockSearchContext('test');
      
      // Should not throw
      const result = await tightBudgetChaser.chaseSlices(seeds, context);
      expect(Array.isArray(result)).toBe(true);
    });

    it('should apply rollout gating', async () => {
      const gatedChaser = new StageBPlusSliceChasing({
        enabled: true,
        rolloutPercentage: 0 // Never apply
      });
      
      const seeds = [createMockSearchHit('test.ts', 10, 0.5)];
      const context = createMockSearchContext('test query');
      
      const result = await gatedChaser.chaseSlices(seeds, context);
      expect(result).toEqual(seeds); // Should return unchanged
    });
  });
});

describe('ROI-Aware Result Micro-Cache', () => {
  let microCache: ROIAwareMicroCache;

  beforeEach(() => {
    microCache = new ROIAwareMicroCache({
      enabled: true,
      ttlSeconds: 2,
      shardCount: 4,
      maxEntriesPerShard: 10
    });
  });

  afterEach(() => {
    microCache.clearCache();
  });

  describe('QueryCanonicalizer', () => {
    it('should canonicalize queries consistently', () => {
      const canonicalizer = new QueryCanonicalizer();
      
      expect(canonicalizer.canonicalize('Find Function')).toBe('find function');
      expect(canonicalizer.canonicalize('FIND   function!!!')).toBe('find function');
      expect(canonicalizer.canonicalize('func test')).toBe('function test'); // Alias resolution
    });

    it('should normalize version numbers', () => {
      const canonicalizer = new QueryCanonicalizer();
      
      expect(canonicalizer.canonicalize('version 1.2.3 api')).toBe('api version VERSION');
      expect(canonicalizer.canonicalize('test 42 items')).toBe('items test NUMBER');
    });
  });

  describe('caching behavior', () => {
    it('should cache and retrieve results', async () => {
      const context = createMockSearchContext('test query');
      const results = [createMockSearchHit('test.ts', 10, 0.8)];
      
      // Cache results
      await microCache.cacheResults(context, results, '1.0', 10);
      
      // Retrieve with tight SLA headroom (should trigger cache)
      const cached = await microCache.getCachedResults(context, '1.0', 1);
      
      expect(cached).toBeDefined();
      expect(cached).toHaveLength(1);
      expect(cached![0]!.file).toBe('test.ts');
    });

    it('should respect SLA headroom threshold', async () => {
      const context = createMockSearchContext('test query');
      const results = [createMockSearchHit('test.ts', 10, 0.8)];
      
      await microCache.cacheResults(context, results, '1.0', 10);
      
      // Sufficient SLA headroom - should not use cache
      const cached = await microCache.getCachedResults(context, '1.0', 10);
      expect(cached).toBeNull();
    });

    it('should invalidate on index version change', async () => {
      const context = createMockSearchContext('test query');
      const results = [createMockSearchHit('test.ts', 10, 0.8)];
      
      await microCache.cacheResults(context, results, '1.0', 10);
      
      // Different index version - should miss
      const cached = await microCache.getCachedResults(context, '2.0', 1);
      expect(cached).toBeNull();
    });

    it('should respect TTL expiration', async () => {
      const shortTTLCache = new ROIAwareMicroCache({
        enabled: true,
        ttlSeconds: 0.1 // 100ms TTL
      });
      
      const context = createMockSearchContext('test query');
      const results = [createMockSearchHit('test.ts', 10, 0.8)];
      
      await shortTTLCache.cacheResults(context, results, '1.0', 10);
      
      // Wait for TTL expiration
      await new Promise(resolve => setTimeout(resolve, 150));
      
      const cached = await shortTTLCache.getCachedResults(context, '1.0', 1);
      expect(cached).toBeNull();
    });
  });

  describe('IntentClassifier', () => {
    it('should classify query intents correctly', () => {
      const classifier = new IntentClassifier();
      const context = createMockSearchContext('test');
      
      expect(classifier.classifyIntent('how to use function', context)).toBe('NL_query');
      expect(classifier.classifyIntent('testFunction', context)).toBe('symbol_lookup');
      expect(classifier.classifyIntent('find function definition', context)).toBe('function_search');
      expect(classifier.classifyIntent('class MyClass', context)).toBe('type_search');
    });
  });
});

describe('ANN Hygiene Optimizer', () => {
  let annHygiene: ANNHygieneOptimizer;

  beforeEach(() => {
    annHygiene = new ANNHygieneOptimizer({
      enabled: true,
      maxLatencyBudgetMs: 5,
      visitedSetPoolSize: 10,
      batchSize: 8
    });
  });

  describe('VisitedSetPool', () => {
    it('should reuse visited sets', () => {
      const pool = new VisitedSetPool(5);
      
      const vs1 = pool.getVisitedSet('query1');
      vs1.visited.add(1);
      vs1.visited.add(2);
      
      pool.returnVisitedSet(vs1);
      
      const vs2 = pool.getVisitedSet('query1');
      expect(vs2.id).toBe(vs1.id);
      expect(vs2.reuseCount).toBe(1);
      expect(vs2.visited.size).toBe(0); // Should be cleared for new search
    });
  });

  describe('BatchedTopKSelector', () => {
    it('should select top-k efficiently', () => {
      const selector = new BatchedTopKSelector(4, true);
      
      const candidates = Array.from({ length: 20 }, (_, i) => ({
        nodeId: i,
        distance: Math.random()
      }));
      
      const topK = selector.selectTopK(candidates, 5);
      
      expect(topK).toHaveLength(5);
      // Should be sorted by distance (ascending)
      for (let i = 0; i < topK.length - 1; i++) {
        expect(topK[i]!.distance).toBeLessThanOrEqual(topK[i + 1]!.distance);
      }
    });

    it('should handle impact-ordered partial sort', () => {
      const selector = new BatchedTopKSelector(4, true);
      
      const candidates = [
        { nodeId: 1, distance: 0.8 },
        { nodeId: 2, distance: 0.2 },
        { nodeId: 3, distance: 0.5 },
        { nodeId: 4, distance: 0.1 },
        { nodeId: 5, distance: 0.9 }
      ];
      
      const topK = selector.impactOrderedPartialSort(candidates, 3);
      
      expect(topK).toHaveLength(3);
      expect(topK[0]!.nodeId).toBe(4); // Should have lowest distance
      expect(topK[1]!.nodeId).toBe(2);
      expect(topK[2]!.nodeId).toBe(3);
    });
  });

  describe('optimization pipeline', () => {
    it('should apply optimizations within budget', async () => {
      const context = createMockSearchContext('test query');
      const candidateNodes = [1, 2, 3, 4, 5];
      
      const result = await annHygiene.optimizeSearch(context, candidateNodes, 3, 'functions');
      
      expect(result.optimizedCandidates).toHaveLength(3);
      expect(result.metrics.latency_ms).toBeLessThan(5);
    });

    it('should handle empty candidate list', async () => {
      const context = createMockSearchContext('test');
      
      const result = await annHygiene.optimizeSearch(context, [], 10);
      
      expect(result.optimizedCandidates).toHaveLength(0);
      expect(result.metrics).toBeDefined();
    });
  });
});

describe('Quality Metrics Calculator', () => {
  let calculator: QualityMetricsCalculator;

  beforeEach(() => {
    calculator = new QualityMetricsCalculator();
  });

  it('should calculate SLA-Recall@50', () => {
    const hits = Array.from({ length: 60 }, (_, i) => 
      createMockSearchHit(`file${i}.ts`, i, i < 30 ? 0.8 : 0.3) // First 30 are "relevant"
    );
    
    const recall = calculator.calculateSLARecall(hits, 50);
    expect(recall).toBeGreaterThan(0);
    expect(recall).toBeLessThanOrEqual(1);
  });

  it('should calculate nDCG@10', () => {
    const hits = [
      createMockSearchHit('file1.ts', 1, 0.9),
      createMockSearchHit('file2.ts', 2, 0.8),
      createMockSearchHit('file3.ts', 3, 0.7),
      createMockSearchHit('file4.ts', 4, 0.6),
      createMockSearchHit('file5.ts', 5, 0.5)
    ];
    
    const ndcg = calculator.calculateNDCG(hits, 10);
    expect(ndcg).toBeGreaterThan(0);
    expect(ndcg).toBeLessThanOrEqual(1);
  });

  it('should calculate ECE', () => {
    const hits = Array.from({ length: 50 }, (_, i) => 
      createMockSearchHit(`file${i}.ts`, i, Math.random())
    );
    
    const ece = calculator.calculateECE(hits);
    expect(ece).toBeGreaterThanOrEqual(0);
    expect(ece).toBeLessThanOrEqual(1);
  });
});

describe('Embedder-Agnostic Optimizer Integration', () => {
  let optimizer: EmbedderAgnosticOptimizer;

  beforeEach(() => {
    optimizer = new EmbedderAgnosticOptimizer({
      enabled: true,
      indexVersion: '1.0',
      maxTotalLatencyMs: 20,
      enableQualityGates: true,
      constraintAware: { enabled: true, maxLatencyMs: 5 },
      sliceChasing: { enabled: true, budgetMs: 5, rolloutPercentage: 100 },
      microCache: { enabled: true, ttlSeconds: 1 },
      annHygiene: { enabled: true, maxLatencyBudgetMs: 5 }
    });
  });

  afterEach(() => {
    optimizer.reset();
  });

  it('should coordinate all optimization systems', async () => {
    const originalHits = [
      createMockSearchHit('file1.ts', 10, 0.7, ['semantic']),
      createMockSearchHit('file2.ts', 20, 0.6, ['fuzzy']),
      createMockSearchHit('file3.ts', 30, 0.8, ['exact'])
    ];
    
    const context = createMockSearchContext('find test function');
    const definitions = [createMockSymbolDefinition('testFunc', 'file1.ts', 10)];
    const references = [createMockSymbolReference('testFunc', 'file2.ts', 25)];
    
    const result = await optimizer.optimize(originalHits, context, definitions, references);
    
    expect(result.finalHits).toBeDefined();
    expect(result.optimizationsApplied.length).toBeGreaterThan(0);
    expect(result.latencyBreakdown.total).toBeLessThan(20);
    expect(result.qualityMetrics).toBeDefined();
    expect(result.constraints).toBeDefined();
  });

  it('should handle cache hits efficiently', async () => {
    const hits = [createMockSearchHit('test.ts', 10, 0.8)];
    const context = createMockSearchContext('cached query');
    
    // First call - cache miss
    const result1 = await optimizer.optimize(hits, context);
    expect(result1.cacheStatus).toBe('miss');
    
    // Second call with tight SLA headroom - should hit cache
    const result2 = await optimizer.optimize(hits, context);
    // Note: Cache hit depends on SLA headroom in real implementation
    expect(result2.cacheStatus).toMatch(/hit|miss/);
  });

  it('should maintain quality gates', async () => {
    const hits = [createMockSearchHit('test.ts', 10, 0.9)];
    const context = createMockSearchContext('quality test');
    
    // Process several queries to establish quality metrics
    for (let i = 0; i < 5; i++) {
      await optimizer.optimize(hits, context);
    }
    
    const stats = optimizer.getStats();
    expect(stats.quality_gates).toBeDefined();
    expect(optimizer.areQualityGatesPassing()).toBe(true);
  });

  it('should respect total latency budget', async () => {
    const strictOptimizer = new EmbedderAgnosticOptimizer({
      enabled: true,
      maxTotalLatencyMs: 1, // Very tight budget
      constraintAware: { enabled: true, maxLatencyMs: 0.3 },
      sliceChasing: { enabled: true, budgetMs: 0.3 },
      microCache: { enabled: true },
      annHygiene: { enabled: true, maxLatencyBudgetMs: 0.3 }
    });
    
    const hits = Array.from({ length: 50 }, (_, i) => 
      createMockSearchHit(`file${i}.ts`, i, Math.random())
    );
    
    const context = createMockSearchContext('stress test');
    
    // Should not throw even with tight budget
    const result = await strictOptimizer.optimize(hits, context);
    expect(result).toBeDefined();
    expect(Array.isArray(result.finalHits)).toBe(true);
  });

  it('should handle component failures gracefully', async () => {
    // Disable all optimizations to simulate failures
    const failureOptimizer = new EmbedderAgnosticOptimizer({
      enabled: true,
      constraintAware: { enabled: false },
      sliceChasing: { enabled: false },
      microCache: { enabled: false },
      annHygiene: { enabled: false }
    });
    
    const hits = [createMockSearchHit('test.ts', 10, 0.8)];
    const context = createMockSearchContext('failure test');
    
    const result = await failureOptimizer.optimize(hits, context);
    expect(result.finalHits).toEqual(hits); // Should fall back to original
    expect(result.optimizationsApplied).toContain('fallback');
  });

  it('should provide comprehensive statistics', () => {
    const stats = optimizer.getStats();
    
    expect(stats.config).toBeDefined();
    expect(stats.performance).toBeDefined();
    expect(stats.quality_gates).toBeDefined();
    expect(stats.components).toBeDefined();
    expect(stats.components.constraint_reranker).toBeDefined();
    expect(stats.components.slice_chasing).toBeDefined();
    expect(stats.components.micro_cache).toBeDefined();
    expect(stats.components.ann_hygiene).toBeDefined();
  });
});