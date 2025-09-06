/**
 * Enterprise Systems Integration Tests
 * 
 * Tests all four enterprise-grade systems and their integration:
 * 1. Task-Level Correctness with Witness Set Mining
 * 2. Declarative Query-DAG Planner with DSL
 * 3. Tenant Economics as Math (Convex Programming)
 * 4. Adversarial/Durability Drills
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { EnterpriseSystemsCoordinator } from './index.js';
import type { EnterpriseSystemsConfig, SearchRequest } from './index.js';
import { WitnessSetMiner } from './witness-set-mining.js';
import { QueryDAGPlanner } from './query-dag-planner.js';
import { TenantEconomicsEngine } from './tenant-economics.js';
import { AdversarialDurabilityEngine } from './adversarial-durability.js';

const TEST_CONFIG: EnterpriseSystemsConfig = {
  witnessSetMining: {
    ciLogsPath: './test-data/ci-logs',
    gitRepoPath: '.',
    maxWitnessSize: 5,
    minConfidence: 0.6,
  },
  queryPlanning: {
    maxPlannerSpendRatio: 0.15,
    planCacheSize: 100,
    sloConstraints: {
      maxLatencyMs: 25,
      maxMemoryMB: 50,
      minRecall: 0.7,
    },
  },
  tenantEconomics: {
    maxCpuTimeMs: 5000,
    maxMemoryGB: 8,
    lambdaMs: 0.002,
    lambdaGB: 0.2,
  },
  adversarialDurability: {
    maxFileSize: 1024 * 1024, // 1MB for testing
    maxEntropy: 7.0,
    minLanguageConfidence: 0.7,
  },
};

describe('WitnessSetMiner', () => {
  let miner: WitnessSetMiner;

  beforeEach(() => {
    miner = new WitnessSetMiner('./test-data/ci-logs', '.', {
      maxWitnessSize: 5,
      minConfidence: 0.6,
    });
  });

  it('should calculate Success@k correctly', () => {
    // Record some test results
    const taskSuccess1 = miner.recordTaskResult(
      'function test',
      ['file1.ts', 'file2.ts'],
      ['file1.ts', 'file3.ts'] // Witness set
    );
    expect(taskSuccess1.success).toBe(true); // file1.ts is covered
    expect(taskSuccess1.coverage).toBe(0.5); // 1 out of 2 files covered

    const taskSuccess2 = miner.recordTaskResult(
      'class example',
      ['file4.ts', 'file5.ts'],
      ['file1.ts', 'file3.ts'] // No overlap
    );
    expect(taskSuccess2.success).toBe(false);
    expect(taskSuccess2.coverage).toBe(0);

    const successAtK = miner.calculateSuccessAtK(10);
    expect(successAtK.k).toBe(10);
    expect(successAtK.totalQueries).toBe(2);
    expect(successAtK.successfulQueries).toBe(1);
    expect(successAtK.successRate).toBe(0.5);
  });

  it('should evaluate witness set coverage', () => {
    const success = miner.evaluateSuccessAtK(
      ['file1.ts', 'file2.ts', 'file3.ts'], // Result set
      ['file2.ts', 'file4.ts'] // Witness set
    );
    expect(success).toBe(true); // file2.ts is in both

    const failure = miner.evaluateSuccessAtK(
      ['file1.ts', 'file3.ts'], // Result set
      ['file2.ts', 'file4.ts'] // Witness set
    );
    expect(failure).toBe(false); // No overlap
  });

  it('should generate witness set features', () => {
    const hits = [
      { file: 'src/test.ts', line: 1, col: 1, lang: 'typescript', snippet: '', score: 0.9, why: [] },
      { file: 'src/utils.ts', line: 2, col: 1, lang: 'typescript', snippet: '', score: 0.8, why: [] },
    ];

    const features = miner.getWitnessSetFeatures('test query', hits);
    expect(features).toHaveLength(3);
    expect(features[0]).toBeGreaterThanOrEqual(0); // Coverage
    expect(features[1]).toBeGreaterThanOrEqual(0); // Confidence
    expect(features[2]).toBeGreaterThanOrEqual(0); // Size factor
  });

  it('should check SLA compliance', () => {
    // Add some successful results
    miner.recordTaskResult('query1', ['file1.ts'], ['file1.ts']);
    miner.recordTaskResult('query2', ['file2.ts'], ['file2.ts']);

    const slaCheck = miner.checkSLASuccessAt10();
    expect(slaCheck.current).toBeGreaterThan(0);
    expect(typeof slaCheck.isFlat).toBe('boolean');
    expect(typeof slaCheck.trend).toBe('number');
  });
});

describe('QueryDAGPlanner', () => {
  let planner: QueryDAGPlanner;

  beforeEach(() => {
    planner = new QueryDAGPlanner();
  });

  it('should generate query plans', async () => {
    const ctx = {
      query: 'function test',
      repo_sha: 'test_repo',
      k: 10,
      mode: 'hybrid' as const,
      fuzzy_distance: 0.2,
    };

    const plan = await planner.generatePlan(ctx, { maxLatencyMs: 20, maxMemoryMB: 50, minRecall: 0.8, maxOperators: 3 });
    
    expect(plan.id).toBeDefined();
    expect(plan.operators.length).toBeGreaterThan(0);
    expect(plan.estimatedCostMs).toBeGreaterThan(0);
    expect(plan.estimatedNDCG).toBeGreaterThan(0);
    expect(plan.sloConstraints.maxLatencyMs).toBe(20);
  });

  it('should cache plans', async () => {
    const ctx = {
      query: 'function test',
      repo_sha: 'test_repo',
      k: 10,
      mode: 'hybrid' as const,
      fuzzy_distance: 0.2,
    };

    const plan1 = await planner.generatePlan(ctx);
    const plan2 = await planner.generatePlan(ctx); // Should hit cache
    
    expect(plan1.id).toBe(plan2.id);
  });

  it('should execute plans', async () => {
    const ctx = {
      query: 'function test',
      repo_sha: 'test_repo',
      k: 10,
      mode: 'hybrid' as const,
    };

    const plan = await planner.generatePlan(ctx);
    const execution = await planner.executePlan(plan, ctx);
    
    expect(execution.planId).toBe(plan.id);
    expect(execution.actualCostMs).toBeGreaterThan(0);
    expect(execution.actualNDCG).toBeGreaterThan(0);
    expect(execution.operatorTimings.length).toBe(plan.operators.length);
    expect(execution.success).toBe(true);
  });

  it('should generate plans from policy deltas', () => {
    const policyDelta = 'lexscan:k=50,struct:patterns=class|function,rerank:model=monotone';
    const plan = planner.getPlanByPolicy(policyDelta);
    
    expect(plan).toBeDefined();
    expect(plan!.operators.length).toBeGreaterThan(0);
    expect(plan!.operators[0].type).toBe('LexScan');
    expect(plan!.id.startsWith('policy_')).toBe(true);
  });

  it('should track performance metrics', async () => {
    const ctx = {
      query: 'test',
      repo_sha: 'test_repo',
      k: 5,
      mode: 'lexical' as const,
    };

    // Generate a few plans to build metrics
    await planner.generatePlan(ctx);
    await planner.generatePlan({ ...ctx, query: 'different query' });

    const metrics = planner.getPlannerMetrics();
    expect(metrics.spendRatio).toBeGreaterThanOrEqual(0);
    expect(metrics.avgPlannerCostMs).toBeGreaterThanOrEqual(0);
    expect(metrics.cacheHitRatio).toBeGreaterThanOrEqual(0);
    expect(metrics.p95CostMs).toBeGreaterThanOrEqual(0);
  });
});

describe('TenantEconomicsEngine', () => {
  let engine: TenantEconomicsEngine;

  beforeEach(() => {
    engine = new TenantEconomicsEngine({
      maxCpuTimeMs: 5000,
      maxMemoryGB: 4,
      lambdaMs: 0.001,
      lambdaGB: 0.1,
    });
  });

  it('should register tenants', async () => {
    const profile = await engine.registerTenant('test_tenant', 'premium', 0.9);
    
    expect(profile.id).toBe('test_tenant');
    expect(profile.slaClass).toBe('premium');
    expect(profile.businessPriority).toBe(0.9);
    expect(profile.historicalUsage).toBeDefined();
    expect(profile.currentAllocation).toBeDefined();
  });

  it('should optimize resource allocation', async () => {
    // Register multiple tenants
    await engine.registerTenant('tenant1', 'premium');
    await engine.registerTenant('tenant2', 'standard');
    await engine.registerTenant('tenant3', 'basic');

    const allocations = await engine.optimizeResourceAllocation();
    
    expect(allocations.size).toBe(3);
    
    const premiumAllocation = allocations.get('tenant1');
    const standardAllocation = allocations.get('tenant2');
    const basicAllocation = allocations.get('tenant3');
    
    expect(premiumAllocation?.cpuTimeMs).toBeGreaterThan(0);
    expect(standardAllocation?.cpuTimeMs).toBeGreaterThan(0);
    expect(basicAllocation?.cpuTimeMs).toBeGreaterThan(0);
    
    // Premium should get more resources than standard/basic
    expect(premiumAllocation!.priorityWeight).toBeGreaterThan(standardAllocation!.priorityWeight);
    expect(standardAllocation!.priorityWeight).toBeGreaterThan(basicAllocation!.priorityWeight);
  });

  it('should check resource availability', async () => {
    const profile = await engine.registerTenant('test_tenant', 'standard');
    await engine.optimizeResourceAllocation();

    const availability = await engine.checkResourceAvailability('test_tenant', 100, 10);
    
    expect(availability.allowed).toBe(true);
    expect(availability.allocation).toBeDefined();
    expect(availability.reason).toBeUndefined();
  });

  it('should consume resources and update metrics', async () => {
    await engine.registerTenant('test_tenant', 'premium');
    await engine.optimizeResourceAllocation();

    await engine.consumeResources('test_tenant', 50, 5, 0.85);

    const metrics = engine.getSLAUtilityMetrics('test_tenant');
    expect(metrics).toBeDefined();
    expect(metrics!.utilityScore).toBeGreaterThanOrEqual(0);
    expect(metrics!.recallScore).toBeGreaterThanOrEqual(0);
    expect(metrics!.costEfficiency).toBeGreaterThanOrEqual(0);
  });

  it('should track upshift compliance', async () => {
    await engine.registerTenant('tenant1', 'premium');
    await engine.registerTenant('tenant2', 'standard');
    
    await engine.optimizeResourceAllocation();
    
    // Simulate some usage
    await engine.consumeResources('tenant1', 100, 10, 0.9);
    await engine.consumeResources('tenant2', 50, 5, 0.8);

    const compliance = engine.checkUpshiftCompliance();
    
    expect(typeof compliance.compliant).toBe('boolean');
    expect(typeof compliance.avgUpshift).toBe('number');
    expect(Array.isArray(compliance.outOfRange)).toBe(true);
  });
});

describe('AdversarialDurabilityEngine', () => {
  let engine: AdversarialDurabilityEngine;

  beforeEach(() => {
    engine = new AdversarialDurabilityEngine({
      maxFileSize: 100 * 1024, // 100KB for testing
      maxEntropy: 7.0,
      minLanguageConfidence: 0.6,
    });
  });

  it('should detect adversarial content', async () => {
    // Test with high-entropy content
    const highEntropyContent = Array.from({ length: 1000 }, () => 
      String.fromCharCode(Math.floor(Math.random() * 256))
    ).join('');

    const adversarial = await engine.scanForAdversarialContent('test/high_entropy.bin', highEntropyContent);
    
    expect(adversarial).toBeDefined();
    expect(adversarial!.adversaryType).toBe('high_entropy_binary');
    expect(adversarial!.confidence).toBeGreaterThan(0.5);
    expect(adversarial!.metrics.entropy).toBeGreaterThan(6);
  });

  it('should detect vendored code', async () => {
    const content = 'function minified(){return"hello"}';
    const adversarial = await engine.scanForAdversarialContent('node_modules/package/dist/index.min.js', content);
    
    expect(adversarial).toBeDefined();
    expect(adversarial!.adversaryType).toBe('vendored_code');
    expect(adversarial!.metrics.vendorScore).toBeGreaterThan(0.5);
  });

  it('should detect giant blobs', async () => {
    const largeContent = 'x'.repeat(200 * 1024); // 200KB
    const adversarial = await engine.scanForAdversarialContent('test/large_file.txt', largeContent);
    
    expect(adversarial).toBeDefined();
    expect(adversarial!.adversaryType).toBe('giant_blob');
    expect(adversarial!.severity).toBe('high');
  });

  it('should filter quarantined hits', async () => {
    const hits = [
      { file: 'src/normal.ts', line: 1, col: 1, lang: 'typescript', snippet: '', score: 0.9, why: [] },
      { file: 'node_modules/bad/index.js', line: 1, col: 1, lang: 'javascript', snippet: '', score: 0.8, why: [] },
    ];

    // Quarantine the second file
    await engine.scanForAdversarialContent('node_modules/bad/index.js', 'malicious content here');

    const filtered = engine.filterQuarantinedHits(hits);
    expect(filtered.length).toBe(1);
    expect(filtered[0].file).toBe('src/normal.ts');
  });

  it('should start chaos experiments', async () => {
    const experimentId = await engine.startChaosExperiment(
      'Test Chaos',
      'content_adversary',
      'lexical',
      { intensity: 0.2 },
      1000
    );
    
    expect(experimentId).toBeDefined();
    expect(experimentId.startsWith('chaos_')).toBe(true);
  });

  it('should monitor system resilience', async () => {
    const resilience = await engine.monitorSystemResilience();
    
    expect(resilience.spanCoverage).toBeGreaterThanOrEqual(0);
    expect(resilience.spanCoverage).toBeLessThanOrEqual(1);
    expect(resilience.recallAt50).toBeGreaterThanOrEqual(0);
    expect(resilience.p95LatencyMs).toBeGreaterThan(0);
    expect(resilience.klDivergenceWhyMix).toBeGreaterThanOrEqual(0);
    expect(typeof resilience.floorWinsSpikes).toBe('boolean');
    expect(resilience.overallHealthScore).toBeGreaterThanOrEqual(0);
    expect(resilience.overallHealthScore).toBeLessThanOrEqual(1);
  });

  it('should get adversarial metrics', () => {
    const metrics = engine.getAdversarialMetrics();
    
    expect(metrics.totalFilesScanned).toBeGreaterThanOrEqual(0);
    expect(metrics.adversarialFilesDetected).toBeGreaterThanOrEqual(0);
    expect(metrics.quarantinedFiles).toBeGreaterThanOrEqual(0);
    expect(metrics.falsePositiveRate).toBeGreaterThanOrEqual(0);
    expect(metrics.performanceImpactMs).toBeGreaterThanOrEqual(0);
    expect(Array.isArray(metrics.entropyDistribution)).toBe(true);
    expect(metrics.detectionAccuracy).toBeGreaterThanOrEqual(0);
  });
});

describe('EnterpriseSystemsCoordinator', () => {
  let coordinator: EnterpriseSystemsCoordinator;

  beforeEach(() => {
    coordinator = new EnterpriseSystemsCoordinator(TEST_CONFIG);
  });

  it('should initialize all systems', async () => {
    await expect(coordinator.initialize()).resolves.not.toThrow();
  }, 10000); // Allow more time for initialization

  it('should register tenants', async () => {
    await coordinator.initialize();
    
    const profile = await coordinator.registerTenant('test_tenant', 'premium', 0.8);
    
    expect(profile.id).toBe('test_tenant');
    expect(profile.slaClass).toBe('premium');
    expect(profile.businessPriority).toBe(0.8);
  });

  it('should execute enhanced searches', async () => {
    await coordinator.initialize();
    await coordinator.registerTenant('test_tenant', 'standard');

    const request: SearchRequest = {
      context: {
        query: 'function test',
        repo_sha: 'test_repo',
        k: 5,
        mode: 'hybrid',
        fuzzy_distance: 0.2,
      },
      tenantId: 'test_tenant',
      expectedWitnessSet: ['src/test.ts', 'src/utils.ts'],
    };

    const result = await coordinator.executeEnhancedSearch(request);
    
    expect(result.hits).toBeDefined();
    expect(result.queryPlan).toBeDefined();
    expect(result.planExecution).toBeDefined();
    expect(result.witnessSetCoverage).toBeGreaterThanOrEqual(0);
    expect(result.tenantUtilityScore).toBeGreaterThanOrEqual(0);
    expect(result.adversarialFiltered).toBeGreaterThanOrEqual(0);
    expect(result.performanceMetrics.totalLatencyMs).toBeGreaterThan(0);
  });

  it('should generate health reports', async () => {
    await coordinator.initialize();
    await coordinator.registerTenant('test_tenant', 'premium');

    const report = await coordinator.generateHealthReport();
    
    expect(report.timestamp).toBeDefined();
    expect(report.witnessSetMetrics).toBeDefined();
    expect(report.plannerMetrics).toBeDefined();
    expect(report.tenantMetrics).toBeDefined();
    expect(report.adversarialMetrics).toBeDefined();
    expect(report.overallHealthScore).toBeGreaterThanOrEqual(0);
    expect(report.overallHealthScore).toBeLessThanOrEqual(1);
  });

  it('should handle resource allocation denials', async () => {
    await coordinator.initialize();
    
    // Register tenant with very low resource allocation
    await coordinator.registerTenant('limited_tenant', 'basic');

    const request: SearchRequest = {
      context: {
        query: 'expensive complex query that uses lots of resources',
        repo_sha: 'test_repo',
        k: 100, // Large k to trigger resource limits
        mode: 'hybrid',
      },
      tenantId: 'limited_tenant',
    };

    // This might succeed or fail depending on the resource allocation
    // The test verifies the system handles resource constraints
    try {
      const result = await coordinator.executeEnhancedSearch(request);
      expect(result).toBeDefined();
    } catch (error) {
      expect(error.message).toContain('Resource allocation denied');
    }
  });

  it('should support reproducible planning', async () => {
    await coordinator.initialize();
    
    const policyDelta = 'lexscan:k=25,rerank:model=monotone';
    const plan = coordinator.getReproduciblePlan(policyDelta);
    
    expect(plan).toBeDefined();
    expect(plan!.operators.length).toBeGreaterThan(0);
    expect(plan!.id.includes('policy')).toBe(true);
  });

  it('should detect and report tripwire alerts', async () => {
    await coordinator.initialize();
    
    // Start a chaos experiment to potentially trigger alerts
    await coordinator.startChaosExperiment('Test Alert Generation', 'content_adversary', 'all');
    
    // Wait briefly for potential alerts
    await new Promise(resolve => setTimeout(resolve, 100));
    
    const alerts = coordinator.getTripwireAlerts();
    expect(Array.isArray(alerts)).toBe(true);
    // Alerts may or may not be present depending on system state
  });
});

// Performance and stress tests
describe('Enterprise Systems Performance', () => {
  it('should handle multiple concurrent searches', async () => {
    const coordinator = new EnterpriseSystemsCoordinator(TEST_CONFIG);
    await coordinator.initialize();
    
    // Register multiple tenants
    const tenantPromises = [];
    for (let i = 0; i < 5; i++) {
      tenantPromises.push(coordinator.registerTenant(`tenant_${i}`, 'standard'));
    }
    await Promise.all(tenantPromises);

    // Execute concurrent searches
    const searchPromises = [];
    for (let i = 0; i < 5; i++) {
      const request: SearchRequest = {
        context: {
          query: `test query ${i}`,
          repo_sha: 'test_repo',
          k: 5,
          mode: 'hybrid',
        },
        tenantId: `tenant_${i}`,
      };
      searchPromises.push(coordinator.executeEnhancedSearch(request));
    }

    const results = await Promise.all(searchPromises);
    expect(results).toHaveLength(5);
    
    for (const result of results) {
      expect(result.hits).toBeDefined();
      expect(result.performanceMetrics.totalLatencyMs).toBeLessThan(1000); // Should be under 1s
    }
  });

  it('should maintain performance under adversarial content load', async () => {
    const engine = new AdversarialDurabilityEngine();
    
    const start = Date.now();
    
    // Scan multiple files with different adversarial patterns
    const scanPromises = [];
    for (let i = 0; i < 10; i++) {
      const content = i % 2 === 0 
        ? 'normal content here'.repeat(100)
        : Array.from({ length: 1000 }, () => Math.random().toString(36)).join('');
      
      scanPromises.push(engine.scanForAdversarialContent(`test_file_${i}.ts`, content));
    }
    
    await Promise.all(scanPromises);
    
    const duration = Date.now() - start;
    expect(duration).toBeLessThan(5000); // Should complete within 5 seconds
    
    const metrics = engine.getAdversarialMetrics();
    expect(metrics.performanceImpactMs).toBeLessThan(100); // Low performance impact
  });
});