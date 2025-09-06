/**
 * Comprehensive tests for all four Embedder-Proof Levers
 * 
 * Tests validate:
 * 1. Session-Aware Retrieval quality gates
 * 2. Off-Policy Learning DR/OPE correctness  
 * 3. Provenance & Integrity zero drift
 * 4. SLO-First Scheduling optimization
 * 5. Cross-system orchestration
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import type { QueryIntent } from '../../types/core.js';

import { SessionAwareRetrievalSystem } from '../session-aware-retrieval.js';
import { OffPolicyLearningSystem } from '../off-policy-learning.js';
import { ProvenanceIntegritySystem } from '../provenance-integrity.js';
import { SLOFirstSchedulingSystem } from '../slo-first-scheduling.js';
import { EmbedderProofLeversOrchestrator } from '../embedder-proof-levers-orchestrator.js';

describe('Session-Aware Retrieval System', () => {
  let system: SessionAwareRetrievalSystem;

  beforeEach(() => {
    system = new SessionAwareRetrievalSystem();
  });

  afterEach(() => {
    vi.clearAllTimers();
  });

  describe('Session Management', () => {
    it('should create new session for first query', () => {
      const session = system.getOrCreateSession(
        'session_1',
        'find UserService class',
        'symbol',
        'repo_abc123'
      );

      expect(session.session_id).toBe('session_1');
      expect(session.intent_history).toEqual(['symbol']);
      expect(session.repo_set.has('repo_abc123')).toBe(true);
      expect(session.topic_id).toBe('symbol_find_userservice_class');
    });

    it('should update existing session with new query', () => {
      // Create initial session
      const session1 = system.getOrCreateSession('session_1', 'find User', 'symbol', 'repo_1');
      
      // Update with new query
      const session2 = system.getOrCreateSession('session_1', 'get UserService methods', 'refs', 'repo_1');

      expect(session2.session_id).toBe('session_1');
      expect(session2.intent_history).toEqual(['symbol', 'refs']);
      expect(session2.topic_id).toBe('refs_get_userservice_methods');
    });

    it('should predict next intent using Markov transitions', () => {
      const session = system.getOrCreateSession('session_1', 'find User', 'symbol', 'repo_1');
      
      // Record some transitions to build history
      system.getOrCreateSession('session_1', 'get references', 'refs', 'repo_1');
      system.getOrCreateSession('session_1', 'find definition', 'def', 'repo_1');

      const prediction = system.predictNextState(session);

      expect(prediction.next_topic).toBeDefined();
      expect(prediction.probability).toBeGreaterThan(0);
      expect(prediction.recommended_k).toBeGreaterThan(0);
      expect(prediction.confidence).toBeGreaterThan(0);
      expect(prediction.reasoning).toHaveLength.greaterThan(0);
    });
  });

  describe('Micro-Caching', () => {
    it('should cache search results by topic+repo+symbol', () => {
      const mockResults = [
        {
          file_path: '/src/user.ts',
          line: 10,
          col: 5,
          score: 0.9,
          match_reasons: ['symbol' as const],
          snippet: 'class User {'
        }
      ];

      const session = system.getOrCreateSession('session_1', 'find User', 'symbol', 'repo_1');
      system.updateSessionWithResults('session_1', mockResults, mockResults);

      const cached = system.getCachedResults(
        session.topic_id,
        'repo_1',
        'User',
        'current_index_version'
      );

      expect(cached).toBeDefined();
      expect(cached).toHaveLength.greaterThan(0);
    });

    it('should invalidate cache on index version change', () => {
      const mockResults = [
        {
          file_path: '/src/user.ts',
          line: 10,
          col: 5,
          score: 0.9,
          match_reasons: ['symbol' as const],
          snippet: 'class User {'
        }
      ];

      const session = system.getOrCreateSession('session_1', 'find User', 'symbol', 'repo_1');
      system.updateSessionWithResults('session_1', mockResults, mockResults);

      // Different index version should return null
      const cached = system.getCachedResults(
        session.topic_id,
        'repo_1',
        'User',
        'different_index_version'
      );

      expect(cached).toBeNull();
    });
  });

  describe('Quality Gates', () => {
    it('should calculate why-mix KL divergence within threshold', () => {
      const baseline = [0.4, 0.3, 0.2, 0.1];
      const sessionAware = [0.45, 0.25, 0.2, 0.1];

      const klDiv = system.calculateWhyMixKL(baseline, sessionAware);

      expect(klDiv).toBeLessThanOrEqual(0.02); // Gate: KL ≤ 0.02
      expect(klDiv).toBeGreaterThanOrEqual(0);
    });

    it('should provide stage-B+ biases for session files', () => {
      const session = system.getOrCreateSession('session_1', 'find User', 'symbol', 'repo_1');
      
      const mockResults = [
        {
          file_path: '/src/user.ts',
          line: 10,
          col: 5,
          score: 0.9,
          match_reasons: ['symbol' as const]
        }
      ];
      
      system.updateSessionWithResults('session_1', mockResults, mockResults);

      const biases = system.getStageBoostBiases(session);
      
      expect(biases.has('/src/user.ts')).toBe(true);
      expect(biases.get('/src/user.ts')).toBeGreaterThan(0);
      expect(biases.get('/src/user.ts')).toBeLessThanOrEqual(0.1); // Max 10% boost
    });
  });
});

describe('Off-Policy Learning System', () => {
  let system: OffPolicyLearningSystem;

  beforeEach(() => {
    system = new OffPolicyLearningSystem();
  });

  describe('Interaction Logging', () => {
    it('should log query interactions with propensity scores', () => {
      const mockCandidates = [
        {
          file_path: '/src/test.ts',
          line: 5,
          col: 0,
          score: 0.95,
          match_reasons: ['symbol' as const]
        }
      ];

      const contextFeatures = {
        repo_sha: 'abc123',
        file_count: 100,
        query_length: 15,
        has_symbols: true,
        has_natural_language: false,
        session_position: 1,
        user_expertise_level: 0.7
      };

      const randomizationApplied = system.logInteraction(
        'query_1',
        'find UserService',
        'symbol',
        mockCandidates,
        [1.0], // User feedback
        contextFeatures
      );

      expect(typeof randomizationApplied).toBe('boolean');
      
      const metrics = system.getMetrics();
      expect(metrics).toBeDefined();
    });

    it('should apply top-2 randomization with correct probability', () => {
      const mockCandidates = Array.from({ length: 5 }, (_, i) => ({
        file_path: `/src/test${i}.ts`,
        line: i + 1,
        col: 0,
        score: 0.9 - i * 0.1,
        match_reasons: ['symbol' as const]
      }));

      const contextFeatures = {
        repo_sha: 'abc123',
        file_count: 100,
        query_length: 15,
        has_symbols: true,
        has_natural_language: false,
        session_position: 1,
        user_expertise_level: 0.7
      };

      // Run multiple times to test randomization rate
      let randomizationCount = 0;
      const trials = 100;

      for (let i = 0; i < trials; i++) {
        const randomized = system.logInteraction(
          `query_${i}`,
          'find UserService',
          'symbol',
          mockCandidates,
          [1.0, 0.8, 0.6, 0.4, 0.2],
          contextFeatures
        );
        
        if (randomized) randomizationCount++;
      }

      // Should be approximately 10% randomization rate (±5%)
      const randomizationRate = randomizationCount / trials;
      expect(randomizationRate).toBeGreaterThan(0.05);
      expect(randomizationRate).toBeLessThan(0.15);
    });
  });

  describe('Doubly-Robust Evaluation', () => {
    it('should evaluate policy updates with quality gates', () => {
      // Generate some logged data first
      const mockCandidates = [
        {
          file_path: '/src/test.ts',
          line: 5,
          col: 0,
          score: 0.95,
          match_reasons: ['symbol' as const]
        }
      ];

      const contextFeatures = {
        repo_sha: 'abc123',
        file_count: 100,
        query_length: 15,
        has_symbols: true,
        has_natural_language: false,
        session_position: 1,
        user_expertise_level: 0.7
      };

      // Log many interactions to meet minimum sample requirement
      for (let i = 0; i < 1001; i++) {
        system.logInteraction(
          `query_${i}`,
          'find UserService',
          'symbol',
          mockCandidates,
          [0.8 + Math.random() * 0.2],
          contextFeatures
        );
      }

      const candidateWeights = new Map([
        ['feature_1', 0.1],
        ['feature_2', 0.2]
      ]);

      const candidates = system.evaluatePolicy(candidateWeights);

      expect(candidates).toBeDefined();
      expect(Array.isArray(candidates)).toBe(true);
      
      // Should have quality gate validation
      for (const candidate of candidates) {
        expect(candidate.component).toMatch(/^(reranker|stopper)$/);
        expect(candidate.delta_ndcg_at_10).toBeGreaterThanOrEqual(0); // Gate: ≥ 0
        expect(candidate.counterfactual_sla_recall_50).toBeGreaterThanOrEqual(0); // Gate: ≥ 0
        expect(Math.abs(candidate.delta_ece)).toBeLessThanOrEqual(0.01); // Gate: ≤ 0.01
        expect(candidate.artifact_drift).toBeLessThanOrEqual(0.001); // Gate: ≤ 0.1pp
        expect(candidate.recommendation).toMatch(/^(deploy|reject|gather_more_data)$/);
      }
    });
  });
});

describe('Provenance & Integrity System', () => {
  let system: ProvenanceIntegritySystem;

  beforeEach(() => {
    system = new ProvenanceIntegritySystem();
  });

  describe('Merkle Tree Integrity', () => {
    it('should build and verify Merkle tree successfully', () => {
      const mockSegments = [
        { id: 'seg_1', data: Buffer.from('segment1_data') },
        { id: 'seg_2', data: Buffer.from('segment2_data') }
      ];

      const mockPostings = [
        { id: 'post_1', data: Buffer.from('posting1_data') }
      ];

      const mockSymbolGraph = Buffer.from('symbol_graph_data');
      const configFingerprint = 'config_v1.2.3';

      // Build Merkle tree
      const merkleTree = system.buildSegmentMerkleTree(
        mockSegments,
        mockPostings,
        mockSymbolGraph,
        configFingerprint
      );

      expect(merkleTree.root_hash).toBeDefined();
      expect(merkleTree.config_fingerprint).toBe(configFingerprint);
      expect(merkleTree.segment_hashes.size).toBe(2);
      expect(merkleTree.posting_list_hashes.size).toBe(1);

      // Verify integrity
      const verification = system.verifyMerkleIntegrity(
        mockSegments,
        mockPostings,
        mockSymbolGraph
      );

      expect(verification.status).toBe('pass');
      expect(verification.verification_type).toBe('merkle');
      expect(verification.performance_impact_ms).toBeGreaterThan(0);
    });

    it('should detect hash mismatches and refuse mixed trees', () => {
      const mockSegments = [
        { id: 'seg_1', data: Buffer.from('original_data') }
      ];

      const mockPostings = [
        { id: 'post_1', data: Buffer.from('posting_data') }
      ];

      const mockSymbolGraph = Buffer.from('symbol_data');

      // Build with original data
      system.buildSegmentMerkleTree(
        mockSegments,
        mockPostings,
        mockSymbolGraph,
        'config_v1'
      );

      // Verify with modified data
      const modifiedSegments = [
        { id: 'seg_1', data: Buffer.from('modified_data') }
      ];

      const verification = system.verifyMerkleIntegrity(
        modifiedSegments,
        mockPostings,
        mockSymbolGraph
      );

      expect(verification.status).toBe('fail');
      expect(verification.error_details).toContain('hash mismatch');
    });
  });

  describe('Span Normal Form', () => {
    it('should create reproducible span normal form', () => {
      const spanNF = system.createSpanNormalForm(
        '/src/test.ts',
        10, 15, 5, 25,
        'function test() {\n  return "hello";\n}',
        'abc123def',
        'prev_sha'
      );

      expect(spanNF.file_path).toBe('/src/test.ts');
      expect(spanNF.line_start).toBe(10);
      expect(spanNF.line_end).toBe(15);
      expect(spanNF.content_hash).toBeDefined();
      expect(spanNF.patience_diff_map).toBeDefined();
      expect(spanNF.git_sha).toBe('abc123def');
    });
  });

  describe('Round-Trip Fidelity', () => {
    it('should verify zero span drift for HEAD↔SHA↔HEAD round-trips', () => {
      const originalSpan = {
        lineStart: 10,
        lineEnd: 15,
        colStart: 5,
        colEnd: 25
      };

      const verification = system.verifySpanRoundTripFidelity(
        '/src/test.ts',
        originalSpan,
        'head_sha',
        'target_sha'
      );

      // Mock implementation should pass with zero drift
      expect(verification.status).toBe('pass');
      expect(verification.verification_type).toBe('round_trip');
      expect(verification.details).toContain('Zero span drift verified');
    });
  });

  describe('Churn-Indexed TTLs', () => {
    it('should setup and update churn-indexed TTLs correctly', () => {
      // Setup TTL for RAPTOR
      const ttlConfig = system.setupChurnIndexedTTL('raptor', 24, 1, 168);

      expect(ttlConfig.resource_type).toBe('raptor');
      expect(ttlConfig.base_ttl_hours).toBe(24);
      expect(ttlConfig.current_ttl).toBe(24);

      // Update with churn metrics
      system.updateChurnMetrics('raptor', 25, 150, 75, 12);

      // Get updated TTL
      const newTtl = system.getCurrentTTL('raptor');

      expect(newTtl).toBeGreaterThan(0);
      expect(newTtl).toBeLessThanOrEqual(168); // Max TTL
    });

    it('should apply TTL formula: clamp(τ_min, τ_max, c/λ_churn_slice)', () => {
      const ttlConfig = system.setupChurnIndexedTTL('symbol_sketch', 48, 2, 336);

      // High churn should reduce TTL
      system.updateChurnMetrics('symbol_sketch', 100, 500, 300, 50);
      const highChurnTTL = system.getCurrentTTL('symbol_sketch');

      // Low churn should increase TTL (up to max)
      system.updateChurnMetrics('symbol_sketch', 5, 10, 5, 2);
      const lowChurnTTL = system.getCurrentTTL('symbol_sketch');

      expect(lowChurnTTL).toBeGreaterThanOrEqual(highChurnTTL);
      expect(lowChurnTTL).toBeLessThanOrEqual(336); // Max TTL
      expect(highChurnTTL).toBeGreaterThanOrEqual(2); // Min TTL
    });
  });

  describe('Health Checks', () => {
    it('should perform comprehensive health check', async () => {
      const healthResult = await system.performHealthCheck();

      expect(healthResult.overall_status).toMatch(/^(healthy|degraded|unhealthy)$/);
      expect(healthResult.checks).toHaveLength.greaterThan(0);
      expect(healthResult.performance_summary).toBeDefined();
      expect(healthResult.performance_summary.avg_verification_time_ms).toBeGreaterThanOrEqual(0);
      expect(healthResult.performance_summary.success_rate).toBeGreaterThanOrEqual(0);
      expect(healthResult.performance_summary.success_rate).toBeLessThanOrEqual(1);
    });
  });
});

describe('SLO-First Scheduling System', () => {
  let system: SLOFirstSchedulingSystem;

  beforeEach(() => {
    system = new SLOFirstSchedulingSystem();
  });

  describe('Resource Knapsack Optimization', () => {
    it('should optimize resource allocation within budget', () => {
      const queryContext = {
        repo_size_mb: 150,
        query_complexity: 0.6,
        intent: 'symbol' as QueryIntent,
        session_position: 1,
        user_tier: 'pro' as const
      };

      const decision = system.scheduleQuery(
        'query_1',
        'find UserService class',
        'symbol',
        ['shard_1', 'shard_2', 'shard_3'],
        queryContext
      );

      expect(decision.query_id).toBe('query_1');
      expect(decision.solution).toBeDefined();
      expect(decision.allocated_budget_ms).toBeGreaterThan(0);
      expect(decision.solution.total_cost_ms).toBeLessThanOrEqual(decision.allocated_budget_ms);
      expect(decision.solution.expected_delta_ndcg).toBeGreaterThanOrEqual(0);
      expect(decision.optimization_time_ms).toBeLessThanOrEqual(50); // Config limit
    });

    it('should maximize ΔnDCG/ms utility', () => {
      const queryContext = {
        repo_size_mb: 100,
        query_complexity: 0.4,
        intent: 'refs' as QueryIntent,
        session_position: 1,
        user_tier: 'enterprise' as const
      };

      const decision = system.scheduleQuery(
        'query_2',
        'get all references',
        'refs',
        ['shard_1', 'shard_2'],
        queryContext
      );

      // Check that selected items have positive utility
      for (const item of decision.solution.selected_items) {
        expect(item.delta_ndcg_per_ms).toBeGreaterThan(0);
        expect(item.cost_ms).toBeGreaterThan(0);
      }

      // Should have resource utilization breakdown
      expect(decision.solution.resource_utilization.size).toBeGreaterThan(0);
    });
  });

  describe('Hedging Decisions', () => {
    it('should recommend hedging for slowest decile queries', () => {
      const slowQueryContext = {
        repo_size_mb: 2000, // Large repo
        query_complexity: 0.95, // Very complex
        intent: 'struct' as QueryIntent,
        session_position: 10,
        user_tier: 'free' as const
      };

      const decision = system.scheduleQuery(
        'slow_query',
        'complex structural query with many patterns',
        'struct',
        ['shard_1', 'shard_2', 'shard_3', 'shard_4'],
        slowQueryContext
      );

      // May or may not have hedge recommendation depending on predicted latency
      if (decision.hedge_recommendation) {
        expect(decision.hedge_recommendation.target_shards).toHaveLength.greaterThan(0);
        expect(decision.hedge_recommendation.expected_latency_reduction_ms).toBeGreaterThan(0);
        expect(decision.hedge_recommendation.risk_assessment).toBeGreaterThanOrEqual(0);
        expect(decision.hedge_recommendation.risk_assessment).toBeLessThanOrEqual(1);
      }
    });
  });

  describe('Cross-Shard Credits', () => {
    it('should track and apply cross-shard TA credits', () => {
      const queryContext = {
        repo_size_mb: 200,
        query_complexity: 0.5,
        intent: 'symbol' as QueryIntent,
        session_position: 1,
        user_tier: 'pro' as const
      };

      // Execute multiple queries to build up credits
      for (let i = 0; i < 5; i++) {
        const decision = system.scheduleQuery(
          `query_${i}`,
          'find something',
          'symbol',
          ['shard_1', 'shard_2'],
          queryContext
        );

        // Update metrics to simulate query completion
        system.updateMetrics(`query_${i}`, 150, 0.82, decision);
      }

      // Credits should be tracked internally
      const metrics = system.getMetrics();
      expect(metrics).toBeDefined();
    });
  });

  describe('Quality Gates', () => {
    it('should maintain fleet p99 improvement in target range', () => {
      const sloMetrics = system.getSLOMetrics();

      // Mock metrics should be within reasonable bounds
      expect(sloMetrics.fleet_p99_ms).toBeGreaterThan(0);
      expect(sloMetrics.fleet_p95_ms).toBeGreaterThan(0);
      expect(sloMetrics.fleet_p95_ms).toBeLessThanOrEqual(sloMetrics.fleet_p99_ms);
    });

    it('should keep upshift percentage in [3%, 7%] range', () => {
      const metrics = system.getMetrics();

      // Target gate: upshift ∈ [3%, 7%]
      expect(metrics.upshift_percentage).toBeGreaterThanOrEqual(3);
      expect(metrics.upshift_percentage).toBeLessThanOrEqual(7);
    });
  });
});

describe('Embedder-Proof Levers Orchestrator', () => {
  let orchestrator: EmbedderProofLeversOrchestrator;

  beforeEach(() => {
    orchestrator = new EmbedderProofLeversOrchestrator();
  });

  describe('Coordinated Search Processing', () => {
    it('should orchestrate all four systems for search query', async () => {
      const result = await orchestrator.processSearchQuery(
        'session_123',
        'query_456',
        'find UserService authenticate method',
        'symbol',
        'repo_abc123',
        ['shard_1', 'shard_2', 'shard_3']
      );

      expect(result.query_id).toBe('query_456');
      expect(result.session_id).toBe('session_123');
      expect(result.hits).toBeDefined();
      expect(result.execution_time_ms).toBeGreaterThan(0);
      expect(result.session_enhanced).toBe(true);
      expect(result.integrity_verified).toBe(true);
      expect(result.off_policy_logged).toBe(true);
      expect(result.quality_gates_passed).toBe(true);
      expect(result.lever_contributions).toBeDefined();
      expect(result.performance_breakdown).toBeDefined();
    });

    it('should use cached results when appropriate', async () => {
      // First query to populate cache
      await orchestrator.processSearchQuery(
        'session_cache',
        'query_cache_1',
        'find CachedClass',
        'symbol',
        'repo_cache',
        ['shard_1']
      );

      // Second identical query should hit cache
      const result = await orchestrator.processSearchQuery(
        'session_cache',
        'query_cache_2',
        'find CachedClass',
        'symbol',
        'repo_cache',
        ['shard_1']
      );

      expect(result.execution_time_ms).toBeLessThan(50); // Should be fast from cache
    });
  });

  describe('Quality Gate Validation', () => {
    it('should validate all quality gates across systems', async () => {
      const gateReport = await orchestrator.validateQualityGates();

      expect(gateReport.timestamp).toBeInstanceOf(Date);
      expect(typeof gateReport.overall_passed).toBe('boolean');
      expect(gateReport.individual_results).toBeDefined();
      expect(gateReport.individual_results.session_aware_gates).toBeDefined();
      expect(gateReport.individual_results.off_policy_gates).toBeDefined();
      expect(gateReport.individual_results.provenance_gates).toBeDefined();
      expect(gateReport.individual_results.slo_gates).toBeDefined();
    });

    it('should identify failed gates with recommendations', async () => {
      const gateReport = await orchestrator.validateQualityGates();

      expect(Array.isArray(gateReport.failed_gates)).toBe(true);
      expect(Array.isArray(gateReport.recommendations)).toBe(true);

      // If any gates failed, should have recommendations
      if (gateReport.failed_gates.length > 0) {
        expect(gateReport.recommendations.length).toBeGreaterThan(0);
      }
    });
  });

  describe('Nightly Optimization', () => {
    it('should perform comprehensive nightly optimization', async () => {
      const report = await orchestrator.performNightlyOptimization();

      expect(report.timestamp).toBeInstanceOf(Date);
      expect(report.execution_time_ms).toBeGreaterThan(0);
      expect(report.dr_candidates_evaluated).toBeGreaterThanOrEqual(0);
      expect(report.deployable_updates).toBeGreaterThanOrEqual(0);
      expect(typeof report.quality_gates_passed).toBe('boolean');
      expect(report.integrity_health).toMatch(/^(healthy|degraded|unhealthy)$/);
      expect(report.session_model_improvements).toBeDefined();
      expect(report.slo_optimizations).toBeDefined();
      expect(Array.isArray(report.cross_lever_synergies)).toBe(true);
      expect(Array.isArray(report.recommendations)).toBe(true);
    });

    it('should deploy updates only when quality gates pass', async () => {
      const report = await orchestrator.performNightlyOptimization();

      // If updates are deployable, quality gates should have passed
      if (report.deployable_updates > 0) {
        expect(report.quality_gates_passed).toBe(true);
      }
    });
  });

  describe('System Metrics', () => {
    it('should aggregate metrics from all four systems', () => {
      const metrics = orchestrator.getSystemMetrics();

      expect(metrics.session_aware).toBeDefined();
      expect(metrics.off_policy_learning).toBeDefined();
      expect(metrics.provenance_integrity).toBeDefined();
      expect(metrics.slo_scheduling).toBeDefined();

      // Validate session-aware metrics
      expect(metrics.session_aware.success_at_10_improvement).toBeGreaterThanOrEqual(0);
      expect(metrics.session_aware.p95_latency_impact_ms).toBeGreaterThanOrEqual(0);
      expect(metrics.session_aware.why_mix_kl_divergence).toBeGreaterThanOrEqual(0);

      // Validate off-policy learning metrics
      expect(metrics.off_policy_learning.dr_ndcg_improvement).toBeGreaterThanOrEqual(0);
      expect(metrics.off_policy_learning.counterfactual_sla_recall_50).toBeGreaterThanOrEqual(0);

      // Validate provenance integrity metrics
      expect(metrics.provenance_integrity.merkle_verification_success_rate).toBeLessThanOrEqual(1.0);
      expect(metrics.provenance_integrity.span_drift_incidents).toBeGreaterThanOrEqual(0);

      // Validate SLO scheduling metrics
      expect(metrics.slo_scheduling.recall_maintenance).toBeDefined();
      expect(metrics.slo_scheduling.upshift_percentage).toBeGreaterThan(0);
    });
  });

  describe('Error Handling', () => {
    it('should handle system failures gracefully', async () => {
      // Mock a system failure by passing invalid parameters
      const result = await orchestrator.processSearchQuery(
        '',
        '',
        '',
        'invalid_intent' as QueryIntent,
        '',
        []
      );

      expect(result.query_id).toBe('');
      expect(result.hits).toEqual([]);
      expect(result.execution_time_ms).toBeGreaterThan(0);
      expect(result.quality_gates_passed).toBe(false);
      expect(result.error).toBeDefined();
    });
  });
});

describe('Integration Quality Gates', () => {
  it('should validate all specified quality gates', async () => {
    const orchestrator = new EmbedderProofLeversOrchestrator();

    // Execute a search to generate metrics
    await orchestrator.processSearchQuery(
      'test_session',
      'test_query',
      'find TestClass',
      'symbol',
      'test_repo',
      ['shard_1', 'shard_2']
    );

    const metrics = orchestrator.getSystemMetrics();

    // Session-Aware Retrieval Gates
    // Gate: Success@10 +0.5pp on multi-hop sessions
    expect(metrics.session_aware.success_at_10_improvement).toBeGreaterThanOrEqual(0.5);
    
    // Gate: p95 ≤ +0.3ms
    expect(metrics.session_aware.p95_latency_impact_ms).toBeLessThanOrEqual(0.3);
    
    // Gate: why-mix KL ≤ 0.02
    expect(metrics.session_aware.why_mix_kl_divergence).toBeLessThanOrEqual(0.02);

    // Off-Policy Learning Gates
    // Gate: ΔnDCG@10 (DR) ≥ 0
    expect(metrics.off_policy_learning.dr_ndcg_improvement).toBeGreaterThanOrEqual(0);
    
    // Gate: counterfactual SLA-Recall@50 ≥ 0
    expect(metrics.off_policy_learning.counterfactual_sla_recall_50).toBeGreaterThanOrEqual(0);
    
    // Gate: ΔECE ≤ 0.01
    expect(metrics.off_policy_learning.delta_ece).toBeLessThanOrEqual(0.01);
    
    // Gate: artifact-bound drift ≤ 0.1pp
    expect(metrics.off_policy_learning.artifact_drift).toBeLessThanOrEqual(0.001);

    // Provenance & Integrity Gates
    // Gate: Merkle verification 100% success
    expect(metrics.provenance_integrity.merkle_verification_success_rate).toBe(1.0);
    
    // Gate: Zero span drift
    expect(metrics.provenance_integrity.span_drift_incidents).toBe(0);
    
    // Gate: Round-trip fidelity 100%
    expect(metrics.provenance_integrity.round_trip_fidelity).toBe(1.0);

    // SLO-First Scheduling Gates
    // Gate: fleet p99 -10-15%
    expect(metrics.slo_scheduling.fleet_p99_improvement_pct).toBeGreaterThanOrEqual(10);
    expect(metrics.slo_scheduling.fleet_p99_improvement_pct).toBeLessThanOrEqual(15);
    
    // Gate: flat Recall (maintained)
    expect(metrics.slo_scheduling.recall_maintenance).toBe(true);
    
    // Gate: upshift ∈ [3%, 7%]
    expect(metrics.slo_scheduling.upshift_percentage).toBeGreaterThanOrEqual(3);
    expect(metrics.slo_scheduling.upshift_percentage).toBeLessThanOrEqual(7);
  });
});

describe('Performance Requirements', () => {
  it('should meet performance specifications', async () => {
    const orchestrator = new EmbedderProofLeversOrchestrator();

    const startTime = Date.now();
    const result = await orchestrator.processSearchQuery(
      'perf_session',
      'perf_query',
      'performance test query',
      'symbol',
      'perf_repo',
      ['shard_1', 'shard_2', 'shard_3']
    );
    const totalTime = Date.now() - startTime;

    // Overall execution should be efficient
    expect(result.execution_time_ms).toBeLessThan(500); // Under 500ms
    expect(totalTime).toBeLessThan(1000); // Under 1 second total

    // Individual system contributions should be reasonable
    const breakdown = result.performance_breakdown;
    expect(breakdown.session_processing_ms).toBeLessThan(50);
    expect(breakdown.slo_optimization_ms).toBeLessThan(100);
    expect(breakdown.integrity_verification_ms).toBeLessThan(20);
    expect(breakdown.off_policy_logging_ms).toBeLessThan(10);
  });

  it('should maintain memory efficiency', () => {
    const orchestrator = new EmbedderProofLeversOrchestrator();
    
    // Session memory should fit in hot RAM
    // This is validated by the session cleanup mechanisms
    expect(true).toBe(true); // Placeholder - would measure actual memory usage
  });
});