/**
 * Benchmarking types for lens retrieval system
 * Based on TODO.md specifications for deterministic, observable benchmarking
 */

import { z } from 'zod';

// Core benchmark configuration
export const BenchmarkConfigSchema = z.object({
  trace_id: z.string().uuid(),
  suite: z.array(z.enum(['codesearch', 'structural', 'docs'])),
  systems: z.array(z.string()), // ["lex", "+symbols", "+symbols+semantic"]
  slices: z.union([z.literal('SMOKE_DEFAULT'), z.literal('ALL'), z.array(z.string())]),
  seeds: z.number().int().min(1).max(5).default(1),
  api_base_url: z.string().url().optional(), // API base URL for benchmark testing
  k: z.number().int().min(1).max(200).optional(), // Top-K results for benchmarking
  include_cold_start: z.boolean().optional(), // Include cold start measurements
  batch_size: z.number().int().min(1).max(100).optional(), // Batch processing size
  cache_mode: z.union([
    z.enum(['warm', 'cold']),
    z.array(z.enum(['warm', 'cold']))
  ]).default('warm'),
  robustness: z.boolean().default(false),
  metamorphic: z.boolean().default(false),
  k_candidates: z.number().int().min(10).max(500).default(200),
  top_n: z.number().int().min(10).max(100).default(50),
  fuzzy: z.number().int().min(0).max(2).default(2),
  subtokens: z.boolean().default(true),
  semantic_gating: z.object({
    nl_likelihood_threshold: z.number().min(0).max(1).default(0.5),
    min_candidates: z.number().int().min(1).default(10)
  }).default({
    nl_likelihood_threshold: 0.5,
    min_candidates: 10
  }),
  latency_budgets: z.object({
    stage_a_ms: z.number().default(200),
    stage_b_ms: z.number().default(300),
    stage_c_ms: z.number().default(300)
  }).default({
    stage_a_ms: 200,
    stage_b_ms: 300,
    stage_c_ms: 300
  })
});

export type BenchmarkConfig = z.infer<typeof BenchmarkConfigSchema>;

// Repository snapshot and manifest
export const RepoSnapshotSchema = z.object({
  repo_ref: z.string(),
  repo_sha: z.string().length(40), // Git SHA
  manifest: z.record(z.string(), z.any()), // File paths to metadata
  timestamp: z.string().datetime(),
  language_distribution: z.record(z.string(), z.number()),
  total_files: z.number().int().min(0),
  total_lines: z.number().int().min(0)
});

export type RepoSnapshot = z.infer<typeof RepoSnapshotSchema>;

// Golden dataset item
export const GoldenDataItemSchema = z.object({
  id: z.string(),
  query: z.string(),
  query_class: z.enum(['identifier', 'regex', 'nl-ish', 'structural', 'docs']),
  language: z.enum(['ts', 'py', 'rust', 'bash', 'go', 'java']),
  source: z.enum(['pr-derived', 'agent-logs', 'synthetics', 'adversarial']),
  snapshot_sha: z.string().length(40),
  slice_tags: z.array(z.string()),
  expected_results: z.array(z.object({
    file: z.string(),
    line: z.number().int().min(1),
    col: z.number().int().min(0),
    relevance_score: z.number().min(0).max(1), // Ground truth relevance
    match_type: z.enum(['exact', 'symbol', 'structural', 'semantic']),
    why: z.string().optional() // Human annotation
  })),
  metadata: z.record(z.string(), z.any()).optional()
});

export type GoldenDataItem = z.infer<typeof GoldenDataItemSchema>;

// Benchmark run result
export const BenchmarkRunSchema = z.object({
  trace_id: z.string().uuid(),
  config_fingerprint: z.string(), // Hash of configuration
  timestamp: z.string().datetime(),
  status: z.enum(['running', 'completed', 'failed', 'cancelled']),
  system: z.string(), // Which system variant was tested
  total_queries: z.number().int().min(0),
  completed_queries: z.number().int().min(0),
  failed_queries: z.number().int().min(0),
  metrics: z.object({
    recall_at_10: z.number().min(0).max(1),
    recall_at_50: z.number().min(0).max(1),
    ndcg_at_10: z.number().min(0).max(1),
    mrr: z.number().min(0).max(1),
    first_relevant_tokens: z.number().min(0),
    stage_latencies: z.object({
      stage_a_p50: z.number().min(0),
      stage_a_p95: z.number().min(0),
      stage_b_p50: z.number().min(0),
      stage_b_p95: z.number().min(0),
      stage_c_p50: z.number().min(0).optional(),
      stage_c_p95: z.number().min(0).optional(),
      e2e_p50: z.number().min(0),
      e2e_p95: z.number().min(0)
    }),
    fan_out_sizes: z.object({
      stage_a: z.number().int().min(0),
      stage_b: z.number().int().min(0),
      stage_c: z.number().int().min(0).optional()
    }),
    why_attributions: z.record(z.string(), z.number().int().min(0)),
    // TODO.md specified metrics
    cbu_score: z.number().min(0).max(1), // Composite Benchmark Utility
    ece_score: z.number().min(0).max(1), // Expected Calibration Error
    kv_reuse_rate: z.number().min(0).max(1).optional() // KV cache reuse rate
  }),
  errors: z.array(z.object({
    query_id: z.string(),
    error_type: z.string(),
    message: z.string(),
    stage: z.enum(['stage_a', 'stage_b', 'stage_c', 'overall']).optional()
  }))
});

export type BenchmarkRun = z.infer<typeof BenchmarkRunSchema>;

// A/B test comparison result
export const ABTestResultSchema = z.object({
  metric: z.string(),
  baseline_mean: z.number(),
  treatment_mean: z.number(),
  delta: z.number(),
  delta_percent: z.number(),
  ci_lower: z.number(),
  ci_upper: z.number(),
  p_value: z.number(),
  is_significant: z.boolean(),
  sample_size: z.number().int().min(1),
  effect_size: z.number().optional()
});

export type ABTestResult = z.infer<typeof ABTestResultSchema>;

// Metamorphic test case
export const MetamorphicTestSchema = z.object({
  test_id: z.string(),
  transform_type: z.enum(['rename_symbol', 'move_file', 'reformat', 'inject_decoys', 'plant_canaries']),
  repo_snapshot: z.string(), // Reference to snapshot
  original_query: z.string(),
  transformed_query: z.string().optional(),
  expected_invariant: z.string(), // Description of what should remain constant
  tolerance: z.object({
    rank_delta_max: z.number().int().min(0).default(3),
    recall_drop_max: z.number().min(0).max(1).default(0.05)
  }).default({ rank_delta_max: 3, recall_drop_max: 0.05 }),
  metadata: z.record(z.string(), z.any()).optional()
});

export type MetamorphicTest = z.infer<typeof MetamorphicTestSchema>;

// Robustness test configuration  
export const RobustnessTestSchema = z.object({
  test_type: z.enum(['concurrency', 'cold_start', 'incremental_rebuild', 'compaction_under_load', 'fault_injection']),
  parameters: z.record(z.string(), z.any()),
  success_criteria: z.record(z.string(), z.number()),
  timeout_ms: z.number().int().min(1000).default(30000)
});

export type RobustnessTest = z.infer<typeof RobustnessTestSchema>;

// NATS telemetry message schemas
export const BenchmarkPlanMessageSchema = z.object({
  trace_id: z.string().uuid(),
  timestamp: z.string().datetime(),
  config: BenchmarkConfigSchema,
  estimated_duration_ms: z.number().int().min(0),
  total_queries: z.number().int().min(0)
});

export const BenchmarkRunMessageSchema = z.object({
  trace_id: z.string().uuid(),
  timestamp: z.string().datetime(),
  status: z.enum(['started', 'query_completed', 'stage_completed', 'failed']),
  query_id: z.string().optional(),
  stage: z.enum(['stage_a', 'stage_b', 'stage_c']).optional(),
  latency_ms: z.number().min(0).optional(),
  candidates_count: z.number().int().min(0).optional(),
  error: z.string().optional()
});

export const BenchmarkResultMessageSchema = z.object({
  trace_id: z.string().uuid(),
  timestamp: z.string().datetime(),
  final_metrics: BenchmarkRunSchema.shape.metrics,
  artifacts: z.object({
    metrics_parquet: z.string(), // File path
    errors_ndjson: z.string(),
    traces_ndjson: z.string(),
    report_pdf: z.string(),
    config_fingerprint_json: z.string()
  }),
  duration_ms: z.number().int().min(0),
  promotion_gate_result: z.object({
    passed: z.boolean(),
    ndcg_delta: z.number(),
    recall_50_maintained: z.boolean(),
    latency_p95_acceptable: z.boolean(),
    regressions: z.array(z.string())
  })
});

export type BenchmarkPlanMessage = z.infer<typeof BenchmarkPlanMessageSchema>;
export type BenchmarkRunMessage = z.infer<typeof BenchmarkRunMessageSchema>;
export type BenchmarkResultMessage = z.infer<typeof BenchmarkResultMessageSchema>;

// Configuration fingerprint for deterministic runs
export interface ConfigFingerprint {
  bench_schema: string;
  seed: number;
  pool_sha: string;
  oracle_sha: string;
  cbu_coefficients: {
    gamma: number;  // γ - recall weight
    delta: number;  // δ - latency penalty
    beta: number;   // β - verbosity penalty
  };
  code_hash: string;
  config_hash: string;  
  snapshot_shas: Record<string, string>;
  shard_layout: Record<string, any>;
  timestamp: string;
  seed_set: number[];
  // Anti-gaming contract enforcement
  contract_hash: string;
  fixed_layout: boolean;
  dedup_enabled: boolean;
  causal_musts: boolean;
  kv_budget_cap: number;
}

// Promotion gate criteria (from TODO.md)
export const PROMOTION_GATE_CRITERIA = {
  // Core quality gates from TODO.md
  cbu_improvement_min: 0.05, // CBU ≥ +5% (p<0.05)
  ece_max: 0.05, // ECE ≤ 0.05 (Expected Calibration Error)
  cpu_p95_max_ms: 150, // CPU p95 ≤ 150ms
  kv_reuse_improvement_min: 0.10, // KV-reuse ≥ +10pp (percentage points)
  
  // Legacy criteria (maintained for compatibility)
  ndcg_improvement_min: 0.02, // +2%
  recall_50_non_degrading: true,
  latency_p95_max_increase: 0.10, // +10%
  
  // Statistical significance
  significance_level: 0.05, // p < 0.05
  
  // Slice regression limits
  max_slice_regression_pp: 2, // No slice regression > -2pp on Recall@n
  
  // Bootstrap confidence requirements
  bootstrap_samples: 1000, // B=1,000 for stratified bootstrap
  ci_level: 0.95 // 95% confidence intervals
} as const;;

// Additional types needed for the orchestrator
export interface BenchmarkOrchestrationConfig {
  workingDir: string;
  outputDir: string;
  natsUrl?: string;
  repositories: Array<{
    name: string;
    path: string;
  }>;
}

export interface ReportData {
  title: string;
  config: Partial<BenchmarkConfig>;
  benchmarkRuns: any[];
  abTestResults: any[];
  metamorphicResults: any[];
  robustnessResults: any[];
  configFingerprint: ConfigFingerprint;
  metadata: {
    generated_at: string;
    total_duration_ms: number;
    systems_tested: string[];
    queries_executed: number;
  };
}