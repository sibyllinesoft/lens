/**
 * Unit Tests for Embedder-Proof Levers Types
 * Tests four production-ready systems for advanced search capabilities
 */

import { describe, it, expect } from 'bun:test';
import {
  type SessionState,
  type SpanReference,
  type MarkovTransition,
  type SessionAwareConfig,
  type SessionMicroCache,
  type SessionPrediction,
  type DoublyRobustConfig,
  type OffPolicyLogEntry,
  type ContextFeatures,
  type DoublyRobustEstimator,
  type RewardModel,
  type PropensityModel,
  type DRUpdateCandidate,
  type SegmentMerkleTree,
  type SpanNormalForm,
  type PatienceDiffMap,
  type DiffHunk,
  type NormalizationRule,
  type ChurnIndexedTTL,
  type ChurnMetric,
  type IntegrityVerification,
  type SLOSchedulingConfig,
  type ResourceKnapsackItem,
  type ResourceConfiguration,
  type CachePolicy,
  type KnapsackSolution,
  type HedgeRecommendation,
  type SLOMetrics,
  type AdvancedLeverMetrics,
  type EmbedderProofLeverConfig,
  type QualityGateThresholds,
  type SearchHit,
} from '../embedder-proof-levers.js';

describe('Embedder-Proof Levers Types - Session-Aware Retrieval', () => {
  describe('SessionState Interface', () => {
    const validSessionState: SessionState = {
      session_id: 'session-123',
      topic_id: 'auth-flow',
      intent_history: ['def', 'refs', 'symbol'],
      last_spans: [
        {
          file_path: 'src/auth.ts',
          line: 42,
          col: 10,
          span_len: 15,
          symbol_kind: 'function',
          access_count: 3,
          last_access: new Date('2024-01-15T10:30:00Z')
        }
      ],
      repo_set: new Set(['repo1', 'repo2']),
      created_at: new Date('2024-01-15T10:00:00Z'),
      last_accessed: new Date('2024-01-15T10:30:00Z'),
      ttl_minutes: 5
    };

    it('should define complete session state', () => {
      expect(validSessionState.session_id).toBe('session-123');
      expect(validSessionState.intent_history).toHaveLength(3);
      expect(validSessionState.repo_set.has('repo1')).toBe(true);
      expect(validSessionState.ttl_minutes).toBe(5);
    });

    it('should handle empty session state', () => {
      const emptySession: SessionState = {
        session_id: 'empty-session',
        topic_id: 'unknown',
        intent_history: [],
        last_spans: [],
        repo_set: new Set(),
        created_at: new Date(),
        last_accessed: new Date(),
        ttl_minutes: 5
      };

      expect(emptySession.intent_history).toHaveLength(0);
      expect(emptySession.repo_set.size).toBe(0);
    });
  });

  describe('MarkovTransition Interface', () => {
    const validTransition: MarkovTransition = {
      from_topic: 'auth-flow',
      to_topic: 'data-validation',
      probability: 0.65,
      intent_context: ['def', 'refs'],
      transition_count: 150,
      confidence_interval: [0.58, 0.72]
    };

    it('should define transition probability', () => {
      expect(validTransition.probability).toBe(0.65);
      expect(validTransition.confidence_interval).toHaveLength(2);
      expect(validTransition.confidence_interval[0]).toBeLessThan(validTransition.confidence_interval[1]);
    });

    it('should validate probability range', () => {
      expect(validTransition.probability).toBeGreaterThan(0);
      expect(validTransition.probability).toBeLessThan(1);
    });
  });

  describe('SessionAwareConfig Interface', () => {
    const validConfig: SessionAwareConfig = {
      max_session_duration_minutes: 60,
      max_sessions_in_memory: 1000,
      prefetch_shard_count: 3,
      per_file_span_cap_multiplier: 2.5,
      markov_order: 1,
      session_cache_ttl_minutes: 30,
      min_transition_count: 10
    };

    it('should define session configuration', () => {
      expect(validConfig.max_session_duration_minutes).toBe(60);
      expect(validConfig.per_file_span_cap_multiplier).toBe(2.5);
      expect(validConfig.markov_order).toBe(1);
    });

    it('should have reasonable default values', () => {
      expect(validConfig.max_sessions_in_memory).toBeGreaterThan(0);
      expect(validConfig.prefetch_shard_count).toBeGreaterThan(0);
      expect(validConfig.min_transition_count).toBeGreaterThan(0);
    });
  });

  describe('SessionPrediction Interface', () => {
    const validPrediction: SessionPrediction = {
      next_topic: 'error-handling',
      probability: 0.78,
      recommended_k: 50,
      per_file_cap: 10,
      ann_ef_search: 200,
      confidence: 0.85,
      reasoning: ['Historical pattern match', 'User expertise level']
    };

    it('should define prediction with confidence', () => {
      expect(validPrediction.probability).toBe(0.78);
      expect(validPrediction.confidence).toBe(0.85);
      expect(validPrediction.reasoning).toHaveLength(2);
    });
  });
});

describe('Embedder-Proof Levers Types - Off-Policy Learning', () => {
  describe('DoublyRobustConfig Interface', () => {
    const validDRConfig: DoublyRobustConfig = {
      randomization_rate: 0.05,
      min_samples_per_update: 1000,
      dr_method: 'SNIPS',
      counterfactual_threshold: 0,
      ece_drift_threshold: 0.01,
      artifact_drift_threshold: 0.1,
      update_frequency_hours: 24
    };

    it('should validate DR method options', () => {
      const validMethods = ['SNIPS', 'DR-J', 'IPS'];
      expect(validMethods).toContain(validDRConfig.dr_method);
    });

    it('should have reasonable thresholds', () => {
      expect(validDRConfig.randomization_rate).toBeGreaterThan(0);
      expect(validDRConfig.randomization_rate).toBeLessThan(1);
      expect(validDRConfig.ece_drift_threshold).toBeGreaterThan(0);
      expect(validDRConfig.artifact_drift_threshold).toBeGreaterThan(0);
    });
  });

  describe('OffPolicyLogEntry Interface', () => {
    const validLogEntry: OffPolicyLogEntry = {
      query_id: 'query-456',
      query: 'authentication middleware',
      intent: 'symbol',
      propensity_scores: new Map([
        ['doc1', 0.25],
        ['doc2', 0.15]
      ]),
      observed_rewards: new Map([
        ['doc1', 1.0],
        ['doc2', 0.8]
      ]),
      randomization_applied: true,
      swapped_positions: [0, 1],
      timestamp: new Date('2024-01-15T10:30:00Z'),
      context_features: {
        repo_sha: 'abc123',
        file_count: 150,
        query_length: 20,
        has_symbols: true,
        has_natural_language: false,
        session_position: 3,
        user_expertise_level: 0.7
      }
    };

    it('should define complete log entry', () => {
      expect(validLogEntry.query_id).toBe('query-456');
      expect(validLogEntry.propensity_scores.get('doc1')).toBe(0.25);
      expect(validLogEntry.observed_rewards.get('doc1')).toBe(1.0);
      expect(validLogEntry.randomization_applied).toBe(true);
    });

    it('should validate swapped positions', () => {
      expect(validLogEntry.swapped_positions).toHaveLength(2);
      expect(validLogEntry.swapped_positions?.[0]).toBe(0);
      expect(validLogEntry.swapped_positions?.[1]).toBe(1);
    });

    it('should validate context features', () => {
      const features = validLogEntry.context_features;
      expect(features.has_symbols).toBe(true);
      expect(features.has_natural_language).toBe(false);
      expect(features.user_expertise_level).toBe(0.7);
    });
  });

  describe('DoublyRobustEstimator Interface', () => {
    const validEstimator: DoublyRobustEstimator = {
      method: 'DR-J',
      reward_model: {
        model_type: 'xgboost',
        features: ['query_length', 'has_symbols'],
        parameters: new Map([['learning_rate', 0.1], ['max_depth', 6]]),
        training_data_size: 10000,
        cross_validation_score: 0.85
      },
      propensity_model: {
        model_type: 'logistic',
        features: ['user_expertise_level', 'session_position'],
        calibration_method: 'isotonic',
        parameters: new Map([['C', 1.0], ['penalty', 0.01]]),
        auc_score: 0.78
      },
      variance_regularizer: 0.01,
      bias_regularizer: 0.005
    };

    it('should validate model types', () => {
      const validRewardTypes = ['isotonic', 'linear', 'xgboost'];
      const validPropensityTypes = ['logistic', 'gbm'];
      
      expect(validRewardTypes).toContain(validEstimator.reward_model.model_type);
      expect(validPropensityTypes).toContain(validEstimator.propensity_model.model_type);
    });

    it('should have reasonable model performance', () => {
      expect(validEstimator.reward_model.cross_validation_score).toBeGreaterThan(0.5);
      expect(validEstimator.propensity_model.auc_score).toBeGreaterThan(0.5);
    });
  });

  describe('DRUpdateCandidate Interface', () => {
    const validCandidate: DRUpdateCandidate = {
      component: 'reranker',
      delta_ndcg_at_10: 0.03,
      counterfactual_sla_recall_50: 0.82,
      delta_ece: 0.008,
      artifact_drift: 0.05,
      confidence_interval: [0.01, 0.05],
      statistical_power: 0.8,
      recommendation: 'deploy'
    };

    it('should validate component types', () => {
      const validComponents = ['reranker', 'stopper'];
      expect(validComponents).toContain(validCandidate.component);
    });

    it('should validate recommendations', () => {
      const validRecommendations = ['deploy', 'reject', 'gather_more_data'];
      expect(validRecommendations).toContain(validCandidate.recommendation);
    });
  });
});

describe('Embedder-Proof Levers Types - Provenance & Integrity', () => {
  describe('SegmentMerkleTree Interface', () => {
    const validMerkleTree: SegmentMerkleTree = {
      root_hash: 'sha256:abc123def456',
      config_fingerprint: 'config-fp-789',
      segment_hashes: new Map([
        ['segment1', 'hash1'],
        ['segment2', 'hash2']
      ]),
      posting_list_hashes: new Map([
        ['list1', 'ph1'],
        ['list2', 'ph2']
      ]),
      symbol_graph_hash: 'symbol-hash-xyz',
      created_at: new Date('2024-01-15T10:30:00Z'),
      verification_depth: 3
    };

    it('should define complete merkle tree structure', () => {
      expect(validMerkleTree.root_hash).toMatch(/^sha256:/);
      expect(validMerkleTree.segment_hashes.size).toBe(2);
      expect(validMerkleTree.verification_depth).toBe(3);
    });
  });

  describe('SpanNormalForm Interface', () => {
    const validSpanNormal: SpanNormalForm = {
      file_path: 'src/auth.ts',
      line_start: 10,
      line_end: 15,
      col_start: 5,
      col_end: 25,
      content_hash: 'sha256:content123',
      patience_diff_map: {
        original_sha: 'orig123',
        target_sha: 'target456',
        line_mappings: new Map([[10, 12], [11, 13]]),
        diff_hunks: [
          {
            original_start: 10,
            original_count: 5,
            target_start: 12,
            target_count: 5,
            operation: 'context',
            confidence: 1.0
          }
        ],
        patience_algorithm_version: 'v1.2'
      },
      git_sha: 'git-abc123',
      normalization_rules: [
        {
          rule_type: 'whitespace',
          pattern: '\\s+',
          replacement: ' ',
          preserve_semantics: true,
          language_specific: false
        }
      ]
    };

    it('should define span normal form with diff mapping', () => {
      expect(validSpanNormal.line_start).toBeLessThan(validSpanNormal.line_end);
      expect(validSpanNormal.col_start).toBeLessThan(validSpanNormal.col_end);
      expect(validSpanNormal.patience_diff_map.line_mappings.get(10)).toBe(12);
    });

    it('should validate diff hunk operations', () => {
      const validOperations = ['insert', 'delete', 'modify', 'context'];
      const hunk = validSpanNormal.patience_diff_map.diff_hunks[0];
      expect(validOperations).toContain(hunk.operation);
    });

    it('should validate normalization rules', () => {
      const rule = validSpanNormal.normalization_rules[0];
      const validRuleTypes = ['whitespace', 'comments', 'imports', 'formatting'];
      expect(validRuleTypes).toContain(rule.rule_type);
    });
  });

  describe('ChurnIndexedTTL Interface', () => {
    const validChurnTTL: ChurnIndexedTTL = {
      resource_type: 'raptor',
      base_ttl_hours: 24,
      churn_lambda: 0.1,
      ttl_min_hours: 6,
      ttl_max_hours: 72,
      current_ttl: 18,
      last_updated: new Date('2024-01-15T10:30:00Z'),
      churn_history: [
        {
          timestamp: new Date('2024-01-15T09:30:00Z'),
          files_changed: 5,
          lines_added: 100,
          lines_deleted: 50,
          symbols_affected: 15,
          churn_rate: 0.15
        }
      ]
    };

    it('should validate resource types', () => {
      const validTypes = ['raptor', 'centrality', 'symbol_sketch'];
      expect(validTypes).toContain(validChurnTTL.resource_type);
    });

    it('should validate TTL bounds', () => {
      expect(validChurnTTL.ttl_min_hours).toBeLessThan(validChurnTTL.ttl_max_hours);
      expect(validChurnTTL.current_ttl).toBeGreaterThanOrEqual(validChurnTTL.ttl_min_hours);
      expect(validChurnTTL.current_ttl).toBeLessThanOrEqual(validChurnTTL.ttl_max_hours);
    });
  });

  describe('IntegrityVerification Interface', () => {
    const validVerification: IntegrityVerification = {
      verification_type: 'merkle',
      status: 'pass',
      details: 'All hashes verified successfully',
      checked_at: new Date('2024-01-15T10:30:00Z'),
      performance_impact_ms: 2.5
    };

    it('should validate verification types', () => {
      const validTypes = ['merkle', 'span_drift', 'round_trip'];
      expect(validTypes).toContain(validVerification.verification_type);
    });

    it('should validate status options', () => {
      const validStatuses = ['pass', 'fail', 'warning'];
      expect(validStatuses).toContain(validVerification.status);
    });
  });
});

describe('Embedder-Proof Levers Types - SLO-First Scheduling', () => {
  describe('SLOSchedulingConfig Interface', () => {
    const validSLOConfig: SLOSchedulingConfig = {
      millisecond_budget_per_query: 200,
      p95_headroom_multiplier: 0.8,
      knapsack_time_limit_ms: 10,
      hedge_threshold_percentile: 0.9,
      cross_shard_credit_rate: 0.1,
      hot_shard_penalty_factor: 1.5
    };

    it('should define SLO scheduling parameters', () => {
      expect(validSLOConfig.millisecond_budget_per_query).toBe(200);
      expect(validSLOConfig.p95_headroom_multiplier).toBe(0.8);
      expect(validSLOConfig.hedge_threshold_percentile).toBe(0.9);
    });

    it('should validate reasonable values', () => {
      expect(validSLOConfig.p95_headroom_multiplier).toBeGreaterThan(0);
      expect(validSLOConfig.p95_headroom_multiplier).toBeLessThan(1);
      expect(validSLOConfig.hot_shard_penalty_factor).toBeGreaterThan(1);
    });
  });

  describe('ResourceKnapsackItem Interface', () => {
    const validKnapsackItem: ResourceKnapsackItem = {
      resource_type: 'ann_ef',
      cost_ms: 50,
      delta_ndcg_per_ms: 0.002,
      configuration: {
        ann_ef_search: 200,
        cache_policy: {
          policy_type: 'lru',
          max_entries: 1000,
          ttl_seconds: 300,
          eviction_threshold: 0.9,
          warming_strategy: 'predictive'
        }
      },
      probability_improvement: 0.75,
      variance_estimate: 0.01
    };

    it('should validate resource types', () => {
      const validTypes = ['ann_ef', 'stage_b_depth', 'cache_policy', 'shard_fanout'];
      expect(validTypes).toContain(validKnapsackItem.resource_type);
    });

    it('should validate cache policy types', () => {
      const cachePolicy = validKnapsackItem.configuration.cache_policy;
      const validPolicies = ['lru', 'lfu', 'adaptive', 'session_aware'];
      expect(validPolicies).toContain(cachePolicy?.policy_type);
    });

    it('should validate warming strategies', () => {
      const cachePolicy = validKnapsackItem.configuration.cache_policy;
      const validStrategies = ['eager', 'lazy', 'predictive'];
      expect(validStrategies).toContain(cachePolicy?.warming_strategy);
    });
  });

  describe('KnapsackSolution Interface', () => {
    const validSolution: KnapsackSolution = {
      selected_items: [],
      total_cost_ms: 150,
      expected_delta_ndcg: 0.05,
      confidence_interval: [0.03, 0.07],
      resource_utilization: new Map([
        ['cpu', 0.75],
        ['memory', 0.6]
      ]),
      hedge_recommendation: {
        hedge_type: 'cross_shard',
        target_shards: ['shard1', 'shard2'],
        expected_latency_reduction_ms: 20,
        risk_assessment: 0.2,
        cost_in_credits: 5
      }
    };

    it('should validate hedge types', () => {
      const hedge = validSolution.hedge_recommendation;
      const validTypes = ['cross_shard', 'alternative_algorithm', 'cache_warmup'];
      expect(validTypes).toContain(hedge?.hedge_type);
    });

    it('should validate cost-benefit analysis', () => {
      expect(validSolution.total_cost_ms).toBeGreaterThan(0);
      expect(validSolution.expected_delta_ndcg).toBeGreaterThan(0);
      expect(validSolution.confidence_interval[0]).toBeLessThan(validSolution.confidence_interval[1]);
    });
  });

  describe('SLOMetrics Interface', () => {
    const validSLOMetrics: SLOMetrics = {
      fleet_p99_ms: 250,
      fleet_p95_ms: 180,
      recall_sla_compliance: 95.5,
      upshift_percentage: 5.2,
      resource_efficiency: 0.003,
      hot_shard_starvation_events: 2,
      hedge_success_rate: 0.82
    };

    it('should validate percentile ordering', () => {
      expect(validSLOMetrics.fleet_p95_ms).toBeLessThan(validSLOMetrics.fleet_p99_ms);
    });

    it('should validate upshift percentage in target range', () => {
      expect(validSLOMetrics.upshift_percentage).toBeGreaterThanOrEqual(3);
      expect(validSLOMetrics.upshift_percentage).toBeLessThanOrEqual(7);
    });
  });
});

describe('Embedder-Proof Levers Types - Integration & Metrics', () => {
  describe('AdvancedLeverMetrics Interface', () => {
    const validAdvancedMetrics: AdvancedLeverMetrics = {
      session_aware: {
        success_at_10_improvement: 0.6,
        p95_latency_impact_ms: 0.2,
        why_mix_kl_divergence: 0.015,
        cache_hit_rate: 0.85,
        session_prediction_accuracy: 0.78
      },
      off_policy_learning: {
        dr_ndcg_improvement: 0.02,
        counterfactual_sla_recall_50: 0.81,
        delta_ece: 0.008,
        artifact_drift: 0.08,
        update_deployment_rate: 0.25
      },
      provenance_integrity: {
        merkle_verification_success_rate: 1.0,
        span_drift_incidents: 0,
        round_trip_fidelity: 1.0,
        integrity_check_latency_ms: 1.2,
        ttl_optimization_savings_pct: 15.5
      },
      slo_scheduling: {
        fleet_p99_improvement_pct: -12.5,
        recall_maintenance: true,
        upshift_percentage: 4.8,
        resource_efficiency_improvement: 0.25,
        hedge_accuracy: 0.88
      }
    };

    it('should validate session-aware metrics targets', () => {
      const sessionMetrics = validAdvancedMetrics.session_aware;
      expect(sessionMetrics.success_at_10_improvement).toBeGreaterThanOrEqual(0.5); // Target: +0.5pp
      expect(sessionMetrics.p95_latency_impact_ms).toBeLessThanOrEqual(0.3); // Target: ≤ +0.3ms
      expect(sessionMetrics.why_mix_kl_divergence).toBeLessThanOrEqual(0.02); // Target: ≤ 0.02
    });

    it('should validate off-policy learning targets', () => {
      const oplMetrics = validAdvancedMetrics.off_policy_learning;
      expect(oplMetrics.dr_ndcg_improvement).toBeGreaterThanOrEqual(0); // Target: ≥ 0
      expect(oplMetrics.counterfactual_sla_recall_50).toBeGreaterThanOrEqual(0); // Target: ≥ 0
      expect(oplMetrics.delta_ece).toBeLessThanOrEqual(0.01); // Target: ≤ 0.01
      expect(oplMetrics.artifact_drift).toBeLessThanOrEqual(0.1); // Target: ≤ 0.1pp
    });

    it('should validate provenance integrity targets', () => {
      const provMetrics = validAdvancedMetrics.provenance_integrity;
      expect(provMetrics.merkle_verification_success_rate).toBe(1.0); // Target: 100%
      expect(provMetrics.span_drift_incidents).toBe(0); // Target: 0
      expect(provMetrics.round_trip_fidelity).toBe(1.0); // Target: 100%
    });

    it('should validate SLO scheduling targets', () => {
      const sloMetrics = validAdvancedMetrics.slo_scheduling;
      expect(sloMetrics.fleet_p99_improvement_pct).toBeGreaterThanOrEqual(-15); // Target: -10% to -15%
      expect(sloMetrics.fleet_p99_improvement_pct).toBeLessThanOrEqual(-10);
      expect(sloMetrics.recall_maintenance).toBe(true); // Target: true
      expect(sloMetrics.upshift_percentage).toBeGreaterThanOrEqual(3); // Target: [3%, 7%]
      expect(sloMetrics.upshift_percentage).toBeLessThanOrEqual(7);
    });
  });

  describe('EmbedderProofLeverConfig Interface', () => {
    const validLeverConfig: EmbedderProofLeverConfig = {
      session_aware: {
        max_session_duration_minutes: 60,
        max_sessions_in_memory: 1000,
        prefetch_shard_count: 3,
        per_file_span_cap_multiplier: 2.0,
        markov_order: 1,
        session_cache_ttl_minutes: 30,
        min_transition_count: 10
      },
      off_policy: {
        randomization_rate: 0.05,
        min_samples_per_update: 1000,
        dr_method: 'SNIPS',
        counterfactual_threshold: 0,
        ece_drift_threshold: 0.01,
        artifact_drift_threshold: 0.1,
        update_frequency_hours: 24
      },
      provenance: {
        enable_merkle_trees: true,
        enable_span_normal_form: true,
        enable_churn_indexed_ttl: false,
        verification_frequency_minutes: 60
      },
      slo_scheduling: {
        millisecond_budget_per_query: 200,
        p95_headroom_multiplier: 0.8,
        knapsack_time_limit_ms: 10,
        hedge_threshold_percentile: 0.9,
        cross_shard_credit_rate: 0.1,
        hot_shard_penalty_factor: 1.5
      },
      integration: {
        enable_cross_lever_optimization: true,
        global_performance_budget_ms: 500,
        quality_gate_thresholds: {
          min_sla_recall_50: 0.8,
          max_p95_regression_ms: 50,
          max_quality_drift_pct: 5.0,
          min_statistical_power: 0.8,
          max_false_positive_rate: 0.05
        }
      }
    };

    it('should define complete configuration', () => {
      expect(validLeverConfig.session_aware.markov_order).toBe(1);
      expect(validLeverConfig.off_policy.dr_method).toBe('SNIPS');
      expect(validLeverConfig.provenance.enable_merkle_trees).toBe(true);
      expect(validLeverConfig.slo_scheduling.p95_headroom_multiplier).toBe(0.8);
    });

    it('should validate quality gate thresholds', () => {
      const gates = validLeverConfig.integration.quality_gate_thresholds;
      expect(gates.min_sla_recall_50).toBeGreaterThan(0);
      expect(gates.max_p95_regression_ms).toBeGreaterThan(0);
      expect(gates.min_statistical_power).toBeGreaterThan(0.5);
      expect(gates.max_false_positive_rate).toBeLessThan(0.1);
    });
  });

  describe('SearchHit Interface (Enhanced)', () => {
    const validSearchHit: SearchHit = {
      file_path: 'src/auth.ts',
      file: 'src/auth.ts',
      line: 42,
      col: 10,
      score: 0.85,
      match_reasons: ['exact', 'symbol'],
      why: ['exact', 'symbol'],
      snippet: 'function authenticate(user: User)',
      context: 'Authentication module',
      symbol_kind: 'function',
      session_boost: 0.15,
      lang: 'typescript',
      language: 'typescript',
      ast_path: 'Program.FunctionDeclaration',
      byte_offset: 1024,
      span_len: 25,
      context_before: 'export class AuthService {',
      context_after: '  return validateUser(user);'
    };

    it('should extend base SearchHit with session capabilities', () => {
      expect(validSearchHit.file_path).toBe('src/auth.ts');
      expect(validSearchHit.file).toBe('src/auth.ts'); // Alias
      expect(validSearchHit.match_reasons).toEqual(validSearchHit.why); // Alias
      expect(validSearchHit.session_boost).toBe(0.15);
    });

    it('should support compatibility fields', () => {
      expect(validSearchHit.language).toBe(validSearchHit.lang);
      expect(validSearchHit.byte_offset).toBe(1024);
      expect(validSearchHit.context_before).toBeDefined();
      expect(validSearchHit.context_after).toBeDefined();
    });

    it('should handle optional metadata fields', () => {
      const minimalHit: SearchHit = {
        file_path: 'minimal.ts',
        file: 'minimal.ts',
        line: 1,
        col: 0,
        score: 0.5,
        match_reasons: ['fuzzy'],
        why: ['fuzzy']
      };

      expect(minimalHit.session_boost).toBeUndefined();
      expect(minimalHit.snippet).toBeUndefined();
      expect(minimalHit.symbol_kind).toBeUndefined();
    });
  });
});

describe('Embedder-Proof Levers Types - Type Compatibility', () => {
  it('should ensure proper typing for complex nested structures', () => {
    // Test that Map types work correctly
    const propensityScores = new Map<string, number>();
    propensityScores.set('doc1', 0.25);
    propensityScores.set('doc2', 0.15);
    
    expect(propensityScores.get('doc1')).toBe(0.25);
    expect(propensityScores.size).toBe(2);
  });

  it('should validate enum-like type constraints', () => {
    const validIntents = ['def', 'refs', 'symbol', 'struct', 'lexical', 'NL'];
    const validModes = ['lex', 'lexical', 'struct', 'hybrid'];
    const validReasons = ['exact', 'fuzzy', 'symbol', 'struct', 'semantic'];

    // These should compile without errors
    expect(validIntents).toContain('symbol');
    expect(validModes).toContain('hybrid');
    expect(validReasons).toContain('semantic');
  });

  it('should handle confidence interval tuples', () => {
    const confidenceInterval: [number, number] = [0.58, 0.72];
    expect(confidenceInterval).toHaveLength(2);
    expect(confidenceInterval[0]).toBeLessThan(confidenceInterval[1]);
  });

  it('should validate Date types in interfaces', () => {
    const timestamp = new Date('2024-01-15T10:30:00Z');
    const sessionState: Partial<SessionState> = {
      created_at: timestamp,
      last_accessed: timestamp
    };
    
    expect(sessionState.created_at).toBeInstanceOf(Date);
    expect(sessionState.last_accessed).toBeInstanceOf(Date);
  });
});