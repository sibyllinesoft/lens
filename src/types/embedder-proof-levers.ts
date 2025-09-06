/**
 * Advanced Embedder-Proof Levers Types
 * Four production-ready systems that survive future embedder swaps
 * and compound with existing search infrastructure.
 */

import type { QueryIntent, SymbolKind, SearchMode, MatchReason } from './core.js';

// =============================================================================
// 1. SESSION-AWARE RETRIEVAL TYPES
// =============================================================================

export interface SessionState {
  session_id: string;
  topic_id: string;
  intent_history: QueryIntent[];
  last_spans: SpanReference[];
  repo_set: Set<string>;
  created_at: Date;
  last_accessed: Date;
  ttl_minutes: number; // Default 5 minutes
}

export interface SpanReference {
  file_path: string;
  line: number;
  col: number;
  span_len: number;
  symbol_kind?: SymbolKind;
  access_count: number;
  last_access: Date;
}

export interface MarkovTransition {
  from_topic: string;
  to_topic: string;
  probability: number;
  intent_context: QueryIntent[];
  transition_count: number;
  confidence_interval: [number, number];
}

export interface SessionAwareConfig {
  max_session_duration_minutes: number;
  max_sessions_in_memory: number;
  prefetch_shard_count: number;
  per_file_span_cap_multiplier: number; // e.g., 2.0x for in-session files
  markov_order: number; // First-order: 1
  session_cache_ttl_minutes: number;
  min_transition_count: number; // For statistical significance
}

export interface SessionMicroCache {
  key: string; // (topic_id, repo, symbol)
  results: SearchHit[];
  index_version: string;
  span_hash: string;
  created_at: Date;
  hit_count: number;
  cache_score: number; // Quality metric
}

export interface SessionPrediction {
  next_topic: string;
  probability: number;
  recommended_k: number;
  per_file_cap: number;
  ann_ef_search: number;
  confidence: number;
  reasoning: string[];
}

// =============================================================================
// 2. OFF-POLICY LEARNING WITH DR/OPE TYPES
// =============================================================================

export interface DoublyRobustConfig {
  randomization_rate: number; // For top-2 swaps
  min_samples_per_update: number;
  dr_method: 'SNIPS' | 'DR-J' | 'IPS';
  counterfactual_threshold: number; // SLA-Recall@50 ≥ 0
  ece_drift_threshold: number; // ΔECE ≤ 0.01
  artifact_drift_threshold: number; // ≤ 0.1pp
  update_frequency_hours: number; // nightly
}

export interface OffPolicyLogEntry {
  query_id: string;
  query: string;
  intent: QueryIntent;
  propensity_scores: Map<string, number>; // doc_id -> propensity
  observed_rewards: Map<string, number>; // doc_id -> reward
  randomization_applied: boolean;
  swapped_positions?: [number, number];
  timestamp: Date;
  context_features: ContextFeatures;
}

export interface ContextFeatures {
  repo_sha: string;
  file_count: number;
  query_length: number;
  has_symbols: boolean;
  has_natural_language: boolean;
  session_position: number;
  user_expertise_level: number;
}

export interface DoublyRobustEstimator {
  method: 'SNIPS' | 'DR-J' | 'IPS';
  reward_model: RewardModel;
  propensity_model: PropensityModel;
  variance_regularizer: number;
  bias_regularizer: number;
}

export interface RewardModel {
  model_type: 'isotonic' | 'linear' | 'xgboost';
  features: string[];
  parameters: Map<string, number>;
  training_data_size: number;
  cross_validation_score: number;
}

export interface PropensityModel {
  model_type: 'logistic' | 'gbm';
  features: string[];
  calibration_method: 'platt' | 'isotonic';
  parameters: Map<string, number>;
  auc_score: number;
}

export interface DRUpdateCandidate {
  component: 'reranker' | 'stopper';
  delta_ndcg_at_10: number;
  counterfactual_sla_recall_50: number;
  delta_ece: number;
  artifact_drift: number;
  confidence_interval: [number, number];
  statistical_power: number;
  recommendation: 'deploy' | 'reject' | 'gather_more_data';
}

// =============================================================================
// 3. PROVENANCE & INTEGRITY HARDENING TYPES
// =============================================================================

export interface SegmentMerkleTree {
  root_hash: string;
  config_fingerprint: string;
  segment_hashes: Map<string, string>; // segment_id -> hash
  posting_list_hashes: Map<string, string>;
  symbol_graph_hash: string;
  created_at: Date;
  verification_depth: number;
}

export interface SpanNormalForm {
  file_path: string;
  line_start: number;
  line_end: number;
  col_start: number;
  col_end: number;
  content_hash: string; // SHA-256 of normalized content
  patience_diff_map: PatienceDiffMap;
  git_sha: string;
  normalization_rules: NormalizationRule[];
}

export interface PatienceDiffMap {
  original_sha: string;
  target_sha: string;
  line_mappings: Map<number, number>; // original_line -> target_line
  diff_hunks: DiffHunk[];
  patience_algorithm_version: string;
}

export interface DiffHunk {
  original_start: number;
  original_count: number;
  target_start: number;
  target_count: number;
  operation: 'insert' | 'delete' | 'modify' | 'context';
  confidence: number;
}

export interface NormalizationRule {
  rule_type: 'whitespace' | 'comments' | 'imports' | 'formatting';
  pattern: string;
  replacement: string;
  preserve_semantics: boolean;
  language_specific: boolean;
}

export interface ChurnIndexedTTL {
  resource_type: 'raptor' | 'centrality' | 'symbol_sketch';
  base_ttl_hours: number;
  churn_lambda: number; // λ_churn_slice
  ttl_min_hours: number; // τ_min
  ttl_max_hours: number; // τ_max
  current_ttl: number; // Computed: clamp(τ_min, τ_max, c/λ_churn_slice)
  last_updated: Date;
  churn_history: ChurnMetric[];
}

export interface ChurnMetric {
  timestamp: Date;
  files_changed: number;
  lines_added: number;
  lines_deleted: number;
  symbols_affected: number;
  churn_rate: number; // Normalized churn intensity
}

export interface IntegrityVerification {
  verification_type: 'merkle' | 'span_drift' | 'round_trip';
  status: 'pass' | 'fail' | 'warning';
  details: string;
  checked_at: Date;
  performance_impact_ms: number;
  error_details?: string;
}

// =============================================================================
// 4. SLO-FIRST SCHEDULING TYPES
// =============================================================================

export interface SLOSchedulingConfig {
  millisecond_budget_per_query: number;
  p95_headroom_multiplier: number; // e.g., 0.8 for 20% headroom
  knapsack_time_limit_ms: number; // Optimization time limit
  hedge_threshold_percentile: number; // Only hedge slowest 10%
  cross_shard_credit_rate: number;
  hot_shard_penalty_factor: number;
}

export interface ResourceKnapsackItem {
  resource_type: 'ann_ef' | 'stage_b_depth' | 'cache_policy' | 'shard_fanout';
  cost_ms: number;
  delta_ndcg_per_ms: number; // Utility per cost
  configuration: ResourceConfiguration;
  probability_improvement: number;
  variance_estimate: number;
}

export interface ResourceConfiguration {
  ann_ef_search?: number;
  stage_b_max_depth?: number;
  cache_policy?: CachePolicy;
  shard_fanout?: number;
  prefetch_enabled?: boolean;
  parallel_execution?: boolean;
}

export interface CachePolicy {
  policy_type: 'lru' | 'lfu' | 'adaptive' | 'session_aware';
  max_entries: number;
  ttl_seconds: number;
  eviction_threshold: number;
  warming_strategy: 'eager' | 'lazy' | 'predictive';
}

export interface KnapsackSolution {
  selected_items: ResourceKnapsackItem[];
  total_cost_ms: number;
  expected_delta_ndcg: number;
  confidence_interval: [number, number];
  resource_utilization: Map<string, number>;
  hedge_recommendation: HedgeRecommendation | null;
}

export interface HedgeRecommendation {
  hedge_type: 'cross_shard' | 'alternative_algorithm' | 'cache_warmup';
  target_shards: string[];
  expected_latency_reduction_ms: number;
  risk_assessment: number; // 0-1 scale
  cost_in_credits: number;
}

export interface SLOMetrics {
  fleet_p99_ms: number;
  fleet_p95_ms: number;
  recall_sla_compliance: number; // Percentage
  upshift_percentage: number; // Should be in [3%, 7%]
  resource_efficiency: number; // nDCG gained per ms spent
  hot_shard_starvation_events: number;
  hedge_success_rate: number;
}

// =============================================================================
// SHARED TYPES FOR INTEGRATION
// =============================================================================

export interface AdvancedLeverMetrics {
  session_aware: {
    success_at_10_improvement: number; // Target: +0.5pp
    p95_latency_impact_ms: number; // Target: ≤ +0.3ms
    why_mix_kl_divergence: number; // Target: ≤ 0.02
    cache_hit_rate: number;
    session_prediction_accuracy: number;
  };
  off_policy_learning: {
    dr_ndcg_improvement: number; // Target: ≥ 0
    counterfactual_sla_recall_50: number; // Target: ≥ 0
    delta_ece: number; // Target: ≤ 0.01
    artifact_drift: number; // Target: ≤ 0.1pp
    update_deployment_rate: number;
  };
  provenance_integrity: {
    merkle_verification_success_rate: number; // Target: 100%
    span_drift_incidents: number; // Target: 0
    round_trip_fidelity: number; // Target: 100%
    integrity_check_latency_ms: number;
    ttl_optimization_savings_pct: number;
  };
  slo_scheduling: {
    fleet_p99_improvement_pct: number; // Target: -10% to -15%
    recall_maintenance: boolean; // Target: true (flat recall)
    upshift_percentage: number; // Target: [3%, 7%]
    resource_efficiency_improvement: number;
    hedge_accuracy: number;
  };
}

export interface EmbedderProofLeverConfig {
  session_aware: SessionAwareConfig;
  off_policy: DoublyRobustConfig;
  provenance: {
    enable_merkle_trees: boolean;
    enable_span_normal_form: boolean;
    enable_churn_indexed_ttl: boolean;
    verification_frequency_minutes: number;
  };
  slo_scheduling: SLOSchedulingConfig;
  integration: {
    enable_cross_lever_optimization: boolean;
    global_performance_budget_ms: number;
    quality_gate_thresholds: QualityGateThresholds;
  };
}

export interface QualityGateThresholds {
  min_sla_recall_50: number;
  max_p95_regression_ms: number;
  max_quality_drift_pct: number;
  min_statistical_power: number;
  max_false_positive_rate: number;
}

// Search hit interface for session-aware retrieval
export interface SearchHit {
  file_path: string;
  file: string; // Alias for compatibility with span_resolver
  line: number;
  col: number;
  score: number;
  match_reasons: MatchReason[];
  why: MatchReason[]; // Alias for compatibility with span_resolver  
  snippet?: string;
  context?: string;
  symbol_kind?: SymbolKind;
  session_boost?: number;
  // Additional optional fields for span_resolver compatibility
  lang?: string;
  ast_path?: string;
  byte_offset?: number;
  span_len?: number;
  context_before?: string;
}