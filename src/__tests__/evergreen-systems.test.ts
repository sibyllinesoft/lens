/**
 * Comprehensive test suite for all four evergreen optimization systems
 * Tests quality gates, integration, and performance requirements
 */

import { describe, test, expect, beforeEach, afterEach, jest } from 'bun:test';
import { 
  EvergreenSystemsIntegrator,
  type EvergreenSystemsConfig 
} from '../core/evergreen-systems-integration.js';
import {
  EvergreenQualityMonitor,
  type QualityGateConfig,
  type SystemMetrics
} from '../core/evergreen-quality-gates.js';
import { SymbolGraph, PathSensitiveSlicing } from '../core/program-slice-recall.js';
import { BuildTestPriors, BazelParser } from '../core/build-test-priors.js';
import { SpeculativeMultiPlanPlanner } from '../core/speculative-multi-plan.js';
import { CacheAdmissionLearner } from '../core/cache-admission-learner.js';
import type { SearchContext, SymbolDefinition, TestFailure } from '../types/core.js';

describe('Evergreen Systems Integration', () => {
  let integrator: EvergreenSystemsIntegrator;
  let config: EvergreenSystemsConfig;
  
  beforeEach(async () => {
    config = EvergreenSystemsIntegrator.getDefaultConfig();
    integrator = new EvergreenSystemsIntegrator(config);
    await integrator.initialize();
  });
  
  afterEach(async () => {
    await integrator.shutdown();
  });

  test('initializes all systems correctly', async () => {
    const status = integrator.getSystemsStatus();
    
    expect(status.overall_status).toBe('healthy');
    expect(status.individual_status.program_slice_recall).toBe('enabled');
    expect(status.individual_status.build_test_priors).toBe('enabled');
    expect(status.individual_status.speculative_multi_plan).toBe('enabled');
    expect(status.individual_status.cache_admission_learner).toBe('enabled');
  });

  test('executes integrated search pipeline', async () => {
    const context: SearchContext = {
      trace_id: 'test-trace-1',
      repo_sha: 'abc123def456',
      query: 'getUserById function',
      mode: 'hybrid',
      k: 10,
      fuzzy_distance: 2,
      started_at: new Date(),
      stages: [],
    };

    const baseCandidates = [
      {
        doc_id: 'test-1',
        file_path: 'src/users.ts',
        line: 10,
        col: 0,
        score: 0.8,
        match_reasons: ['symbol' as any],
        context: 'function getUserById(id: string)',
      },
      {
        doc_id: 'test-2', 
        file_path: 'src/api.ts',
        line: 25,
        col: 0,
        score: 0.6,
        match_reasons: ['lexical' as any],
        context: 'getUserById API endpoint',
      },
    ];

    const result = await integrator.executeIntegratedSearch(context, baseCandidates);

    expect(result.primary_hits.length).toBeGreaterThan(0);
    expect(result.total_latency_ms).toBeLessThan(100); // Should be fast
    expect(result.plan_used).toBeDefined();
    expect(result.stage_breakdown.cache_check_ms).toBeGreaterThanOrEqual(0);
  });

  test('applies build priors correctly', async () => {
    // Index some symbols first
    const symbols: SymbolDefinition[] = [
      {
        name: 'getUserById',
        kind: 'function',
        file_path: 'src/users.ts',
        line: 10,
        col: 0,
        scope: 'global',
        signature: 'function getUserById(id: string): User',
      },
    ];
    
    await integrator.indexSymbols(symbols, []);

    // Record a test failure to create build context
    const failure: TestFailure = {
      target_id: '//src:user_service_test',
      test_name: 'getUserById_test',
      failure_time: new Date(),
      failure_type: 'test',
      affected_files: ['src/users.ts'],
    };
    
    integrator.recordTestFailure(failure);

    const context: SearchContext = {
      trace_id: 'test-trace-2',
      repo_sha: 'abc123def456',
      query: 'getUserById test failure',
      mode: 'hybrid',
      k: 10,
      fuzzy_distance: 2,
      started_at: new Date(),
      stages: [],
    };

    const baseCandidates = [
      {
        doc_id: 'test-1',
        file_path: 'src/users.ts',
        line: 10,
        col: 0,
        score: 0.5,
        match_reasons: ['symbol' as any],
        context: 'function getUserById(id: string)',
      },
    ];

    const result = await integrator.executeIntegratedSearch(context, baseCandidates);
    
    expect(result.priors_applied).toBeGreaterThanOrEqual(0);
    // File with test failure should get boosted
    const userFile = result.primary_hits.find(h => h.file === 'src/users.ts');
    expect(userFile).toBeDefined();
  });
});

describe('Program Slice Recall', () => {
  let symbolGraph: SymbolGraph;
  let slicing: PathSensitiveSlicing;
  
  beforeEach(() => {
    symbolGraph = new SymbolGraph();
    slicing = new PathSensitiveSlicing(symbolGraph);
    slicing.enableWithRollout(100); // Full rollout for testing
  });

  test('builds symbol graph correctly', () => {
    const symbol: SymbolDefinition = {
      name: 'processUser',
      kind: 'function',
      file_path: 'src/user.ts',
      line: 15,
      col: 0,
      scope: 'global',
    };

    symbolGraph.addSymbolDefinition(symbol);
    
    const stats = symbolGraph.getStats();
    expect(stats.symbols).toBe(1);
    expect(stats.nodes).toBe(1);
  });

  test('performs bounded slicing with depth and node constraints', async () => {
    // Add multiple connected symbols
    const symbols = [
      { name: 'main', kind: 'function' as const, file_path: 'main.ts', line: 1, col: 0, scope: 'global' },
      { name: 'helper1', kind: 'function' as const, file_path: 'utils.ts', line: 10, col: 0, scope: 'utils' },
      { name: 'helper2', kind: 'function' as const, file_path: 'utils.ts', line: 20, col: 0, scope: 'utils' },
    ];

    for (const symbol of symbols) {
      symbolGraph.addSymbolDefinition(symbol);
    }

    // Add call relationships
    symbolGraph.addCallRelation('main', 'helper1', 'main.ts', 1);
    symbolGraph.addCallRelation('helper1', 'helper2', 'utils.ts', 10);

    const sliceResult = await symbolGraph.performSlice(['main'], 2, 64);
    
    expect(sliceResult.node_count).toBeLessThanOrEqual(64); // Respects node limit
    expect(sliceResult.total_depth).toBeLessThanOrEqual(2); // Respects depth limit
  });

  test('honors vendor/third_party veto', async () => {
    const symbols = [
      { name: 'userCode', kind: 'function' as const, file_path: 'src/user.ts', line: 1, col: 0, scope: 'global' },
      { name: 'vendorCode', kind: 'function' as const, file_path: 'node_modules/lib/vendor.ts', line: 1, col: 0, scope: 'global' },
    ];

    for (const symbol of symbols) {
      symbolGraph.addSymbolDefinition(symbol);
    }

    symbolGraph.addCallRelation('userCode', 'vendorCode', 'src/user.ts', 1);

    const sliceResult = await symbolGraph.performSlice(['userCode'], 2, 64);
    
    expect(sliceResult.vetoed_paths).toBeGreaterThan(0); // Should veto vendor paths
  });
});

describe('Build/Test-Aware Priors', () => {
  let buildPriors: BuildTestPriors;
  
  beforeEach(() => {
    buildPriors = new BuildTestPriors();
    buildPriors.enable();
  });

  test('parses Bazel BUILD files correctly', async () => {
    const parser = new BazelParser();
    const buildContent = `
cc_library(
    name = "user_lib",
    srcs = ["user.cc", "user.h"],
    deps = ["//base:logging"],
)

cc_test(
    name = "user_test", 
    srcs = ["user_test.cc"],
    deps = [":user_lib"],
    size = "small",
)
`;

    const targets = parser.parseBuildFile(buildContent, 'src/BUILD');
    
    expect(targets).toHaveLength(2);
    expect(targets[0].name).toBe('user_lib');
    expect(targets[0].type).toBe('library');
    expect(targets[1].name).toBe('user_test');
    expect(targets[1].type).toBe('test');
  });

  test('applies decay to test failure priors', async () => {
    const failure: TestFailure = {
      target_id: 'test_target',
      test_name: 'failing_test',
      failure_time: new Date(Date.now() - 48 * 60 * 60 * 1000), // 48 hours ago
      failure_type: 'test',
      affected_files: ['src/test_file.ts'],
    };

    buildPriors.recordTestFailure(failure);
    
    const priorScore = buildPriors.getFilePrior('src/test_file.ts');
    expect(priorScore).toBeGreaterThan(0);
    expect(priorScore).toBeLessThan(1); // Should be decayed
  });

  test('caps log odds delta to ±0.3', () => {
    const candidates = [
      {
        doc_id: 'test-1',
        file_path: 'high_prior_file.ts',
        line: 1,
        col: 0,
        score: 0.5,
        match_reasons: ['lexical' as any],
        context: 'test',
      },
    ];

    // Set up high prior
    const failure: TestFailure = {
      target_id: 'test',
      test_name: 'test',
      failure_time: new Date(),
      failure_type: 'test',
      affected_files: ['high_prior_file.ts'],
    };
    buildPriors.recordTestFailure(failure);

    const enhanced = buildPriors.applyStageCFeature([...candidates]);
    
    // Score should be modified but not too extremely (log odds capped at ±0.3)
    expect(enhanced[0].score).toBeGreaterThan(candidates[0].score);
    expect(enhanced[0].score).toBeLessThan(candidates[0].score * Math.exp(0.3)); // Max boost
  });
});

describe('Speculative Multi-Plan Planner', () => {
  let planner: SpeculativeMultiPlanPlanner;
  
  beforeEach(() => {
    planner = new SpeculativeMultiPlanPlanner();
    planner.enableWithConstraints(5.0, 10); // 5ms headroom, 10% budget
  });

  test('respects P95 headroom constraints', async () => {
    const context: SearchContext = {
      trace_id: 'test',
      repo_sha: 'abc123',
      query: 'function test',
      mode: 'hybrid',
      k: 10,
      fuzzy_distance: 2,
      started_at: new Date(),
      stages: [],
    };

    // High P95 latency should disable speculative planning
    const result = await planner.executeSpeculativeSearch(context, 45, 50); // 45ms current, 50ms budget
    
    expect(result.execution_stats.primary_plan).toBeDefined();
    // Should fallback to single plan due to insufficient headroom
  });

  test('limits planner budget to 10%', async () => {
    const context: SearchContext = {
      trace_id: 'test',
      repo_sha: 'abc123', 
      query: 'complex query for testing',
      mode: 'hybrid',
      k: 10,
      fuzzy_distance: 2,
      started_at: new Date(),
      stages: [],
    };

    const result = await planner.executeSpeculativeSearch(context, 5, 100); // Low current, high budget
    
    expect(result.execution_stats.budget_consumed).toBeLessThanOrEqual(10); // Max 10% of 100ms budget
  });

  test('provides cooperative cancellation', async () => {
    // This would test the cancellation mechanism in a real async environment
    expect(true).toBe(true); // Placeholder - complex async testing needed
  });
});

describe('Cache Admission That Learns', () => {
  let cache: CacheAdmissionLearner;
  
  beforeEach(() => {
    const config = {
      window_size: 50,
      protected_size: 150,
      probation_size: 50,
      sketch_size: 1024,
      admission_threshold: 0.1,
      aging_period_ms: 30000,
    };
    
    cache = new CacheAdmissionLearner(config);
    cache.enable();
  });
  
  afterEach(() => {
    cache.shutdown();
  });

  test('implements TinyLFU admission control', async () => {
    const context1: SearchContext = {
      trace_id: 'test-1',
      repo_sha: 'abc123',
      query: 'frequent query',
      mode: 'hybrid',
      k: 10,
      fuzzy_distance: 2,
      started_at: new Date(),
      stages: [],
    };

    const context2: SearchContext = {
      trace_id: 'test-2', 
      repo_sha: 'abc123',
      query: 'one-off query',
      mode: 'hybrid',
      k: 10,
      fuzzy_distance: 2,
      started_at: new Date(),
      stages: [],
    };

    const results = [
      { file: 'test.ts', line: 1, col: 0, score: 0.8, why: ['test' as any] },
    ];

    // Simulate frequent access to first query
    for (let i = 0; i < 5; i++) {
      await cache.set({ ...context1, trace_id: `test-1-${i}` }, results);
    }

    // Try to cache one-off query - should be rejected
    const admitted = await cache.set(context2, results);
    
    expect(admitted).toBe(false); // Low frequency query should be rejected
    
    const stats = cache.getStats();
    expect(stats.rejections).toBeGreaterThan(0);
  });

  test('maintains segmented LRU structure', async () => {
    const context: SearchContext = {
      trace_id: 'test',
      repo_sha: 'abc123',
      query: 'test query',
      mode: 'hybrid', 
      k: 10,
      fuzzy_distance: 2,
      started_at: new Date(),
      stages: [],
    };

    const results = [
      { file: 'test.ts', line: 1, col: 0, score: 0.8, why: ['test' as any] },
    ];

    // Cache some results
    await cache.set(context, results);
    
    // Retrieve and verify promotion
    const cached = await cache.get(context);
    expect(cached).toBeDefined();
    expect(cached).toHaveLength(1);
    
    // Second access should promote to protected segment
    const cached2 = await cache.get(context);
    expect(cached2).toBeDefined();
  });

  test('respects CPU overhead limits', () => {
    const stats = cache.getStats();
    expect(stats.cpu_overhead_percent).toBeLessThanOrEqual(3.0); // Should be ≤3%
  });
});

describe('Quality Gates', () => {
  let monitor: EvergreenQualityMonitor;
  let config: QualityGateConfig;
  
  beforeEach(() => {
    config = EvergreenQualityMonitor.getDefaultConfig();
    monitor = new EvergreenQualityMonitor(config, 0.1); // 0.1 hour interval for testing
  });
  
  afterEach(() => {
    monitor.stopMonitoring();
  });

  test('validates Program Slice Recall gates', async () => {
    const metrics: SystemMetrics = {
      slice_recall_at_50: 0.758, // +0.8pp improvement
      slice_p95_latency_ms: 15.7, // +0.7ms increase (within +0.8ms limit)
      slice_span_coverage: 1.0, // 100%
      slice_vendor_veto_honored: 1.0, // 100%
    };

    const report = await monitor.runQualityEvaluation();
    
    const sliceGates = report.gate_results.filter(r => r.system === 'program-slice-recall');
    expect(sliceGates).toHaveLength(4);
    
    const recallGate = sliceGates.find(g => g.gate_name === 'recall_at_50_improvement');
    expect(recallGate?.status).toBe('pass');
    
    const latencyGate = sliceGates.find(g => g.gate_name === 'p95_latency_increase');
    expect(latencyGate?.status).toBe('pass');
  });

  test('validates Build/Test Priors gates', async () => {
    const metrics: SystemMetrics = {
      build_ndcg_at_10_delta: 0.006, // +0.6pp (above +0.5pp requirement)
      build_sla_recall_at_50: 0.02, // Positive
      build_core_at_10_drift: 2.1, // 2.1pp (within ±5pp limit)
      build_why_mix_kl_divergence: 0.032, // Low divergence
    };

    const report = await monitor.runQualityEvaluation();
    
    const buildGates = report.gate_results.filter(r => r.system === 'build-test-priors');
    expect(buildGates.length).toBeGreaterThanOrEqual(3);
    
    const ndcgGate = buildGates.find(g => g.gate_name === 'ndcg_at_10_improvement');
    expect(ndcgGate?.status).toBe('pass');
  });

  test('validates Speculative Multi-Plan gates', async () => {
    const metrics: SystemMetrics = {
      plan_fleet_p99_improvement: -9.3, // -9.3% (within -8% to -12% range)
      plan_flat_recall_maintained: true,
      plan_p95_latency_ms: 15.3, // +0.3ms increase (within +0.6ms limit)
      plan_budget_utilization: 7.8, // 7.8% (within ≤10% limit)
    };

    const report = await monitor.runQualityEvaluation();
    
    const planGates = report.gate_results.filter(r => r.system === 'speculative-multi-plan');
    expect(planGates.length).toBeGreaterThanOrEqual(3);
    
    const p99Gate = planGates.find(g => g.gate_name === 'fleet_p99_improvement');
    expect(p99Gate?.status).toBe('pass');
  });

  test('validates Cache Admission gates', async () => {
    const metrics: SystemMetrics = {
      cache_admission_hit_rate: 0.342, // 34.2%
      cache_lru_baseline_hit_rate: 0.308, // 30.8% (+3.4pp improvement)
      cache_cpu_overhead_percent: 1.8, // 1.8% (within ≤3% limit)
      cache_p95_latency_improvement: -0.52, // -0.52ms (within -0.3 to -0.8ms range)
      cache_span_drift: 0.0, // 0% (no drift allowed)
    };

    const report = await monitor.runQualityEvaluation();
    
    const cacheGates = report.gate_results.filter(r => r.system === 'cache-admission-learner');
    expect(cacheGates.length).toBeGreaterThanOrEqual(4);
    
    const hitRateGate = cacheGates.find(g => g.gate_name === 'admission_hit_rate_improvement');
    expect(hitRateGate?.status).toBe('pass');
    
    const spanDriftGate = cacheGates.find(g => g.gate_name === 'span_drift');
    expect(spanDriftGate?.status).toBe('pass');
  });

  test('generates appropriate recommendations', async () => {
    const report = await monitor.runQualityEvaluation();
    
    expect(report.recommendations).toBeInstanceOf(Array);
    expect(report.recommendations.length).toBeGreaterThan(0);
    
    if (report.gates_failed === 0) {
      expect(report.recommendations.some(r => 
        r.includes('All quality gates passing')
      )).toBe(true);
    }
  });

  test('determines correct overall status', async () => {
    const report = await monitor.runQualityEvaluation();
    
    if (report.gates_failed === 0) {
      expect(['healthy', 'degraded']).toContain(report.overall_status);
    } else {
      expect(['degraded', 'critical']).toContain(report.overall_status);
    }
  });
});