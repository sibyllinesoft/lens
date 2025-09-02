/**
 * API types derived from architecture.cue
 * Enforces request/response contracts at compile-time
 */

import { z } from 'zod';

// Version schemas for API, index, and policy compatibility
export const ApiVersionSchema = z.literal('v1');
export const IndexVersionSchema = z.literal('v1');
export const PolicyVersionSchema = z.literal('v1');

export type ApiVersion = z.infer<typeof ApiVersionSchema>;
export type IndexVersion = z.infer<typeof IndexVersionSchema>;
export type PolicyVersion = z.infer<typeof PolicyVersionSchema>;

// Search request schema matching CUE specification
export const SearchRequestSchema = z.object({
  repo_sha: z.string().min(1).max(64),                     // Repository SHA identifier
  q: z.string().min(1).max(1000),                          // Query not empty, reasonable length
  mode: z.enum(['lex', 'struct', 'hybrid']),               // Valid search modes only
  fuzzy: z.number().int().min(0).max(2),                   // Edit distance 0-2
  k: z.number().int().min(1).max(200),                     // Top-K between 1-200
  timeout_ms: z.number().int().min(100).max(5000).optional(), // Optional timeout
});

export type SearchRequest = z.infer<typeof SearchRequestSchema>;

// Search hit schema
export const SearchHitSchema = z.object({
  file: z.string().min(1),                                 // File path not empty
  line: z.number().int().min(1),                           // Line number >= 1
  col: z.number().int().min(0),                            // Column >= 0
  lang: z.string().optional(),                             // Language identifier
  snippet: z.string().optional(),                          // Code snippet
  score: z.number().min(0).max(1),                         // Normalized score
  why: z.array(z.enum(['exact', 'fuzzy', 'symbol', 'struct', 'structural', 'semantic', 'subtoken'])), // Match reasons
  // Optional metadata
  ast_path: z.string().optional(),                         // Optional AST path
  symbol_kind: z.enum(['function', 'class', 'variable', 'type', 'interface', 'constant', 'enum', 'method', 'property']).optional(),
  byte_offset: z.number().int().min(0).optional(),         // Byte offset in file
  span_len: z.number().int().min(0).optional(),            // Length of match span
  context_before: z.string().optional(),                   // Context before match
  context_after: z.string().optional(),                    // Context after match
  // Structural search metadata
  pattern_type: z.enum(['function_def', 'class_def', 'import', 'async_def', 'decorator', 'try_except', 'for_loop', 'if_statement']).optional(),
  symbol_name: z.string().optional(),                       // Symbol name from structural match
  signature: z.string().optional(),                         // Function/class signature
});

export type SearchHit = z.infer<typeof SearchHitSchema>;

// Search response schema
export const SearchResponseSchema = z.object({
  hits: z.array(SearchHitSchema),
  total: z.number().int().min(0),
  latency_ms: z.object({
    stage_a: z.number().int().min(0).max(500),              // Lexical stage timing
    stage_b: z.number().int().min(0).max(500),              // Symbol stage timing  
    stage_c: z.number().int().min(0).max(1000).optional(),  // Semantic stage (optional)
    total: z.number().int().min(0).max(2000),               // Total latency
  }),
  trace_id: z.string().min(1).max(100),                   // Flexible trace ID format
  api_version: ApiVersionSchema,                          // API version for compatibility
  index_version: IndexVersionSchema,                      // Index version for compatibility
  policy_version: PolicyVersionSchema,                    // Policy version for compatibility
  error: z.string().optional(),                           // Optional error field
  message: z.string().optional(),                         // Optional error message
});

export type SearchResponse = z.infer<typeof SearchResponseSchema>;

// Structural search request
export const StructRequestSchema = z.object({
  repo_sha: z.string().min(1).max(64),                     // Repository SHA identifier
  pattern: z.string().min(1).max(500),
  lang: z.enum(['typescript', 'python', 'rust', 'bash', 'go', 'java']),
  max_results: z.number().int().min(1).max(100).optional(),
});

export type StructRequest = z.infer<typeof StructRequestSchema>;

// Symbols near request
export const SymbolsNearRequestSchema = z.object({
  file: z.string().min(1),
  line: z.number().int().min(1),
  radius: z.number().int().min(1).max(50).optional(),      // Lines around target
});

export type SymbolsNearRequest = z.infer<typeof SymbolsNearRequestSchema>;

// Health response
export const HealthResponseSchema = z.object({
  status: z.enum(['ok', 'degraded', 'down']),
  timestamp: z.string(),
  shards_healthy: z.number().int().min(0),
});

export type HealthResponse = z.infer<typeof HealthResponseSchema>;

// Compatibility check request and response
export const CompatibilityCheckRequestSchema = z.object({
  api_version: ApiVersionSchema,
  index_version: IndexVersionSchema,
  policy_version: PolicyVersionSchema.optional(),         // Policy version for compatibility
  allow_compat: z.boolean().optional().default(false),    // Flag to allow mismatched versions
});

export const CompatibilityCheckResponseSchema = z.object({
  compatible: z.boolean(),
  api_version: ApiVersionSchema,
  index_version: IndexVersionSchema,
  policy_version: PolicyVersionSchema.optional(),
  server_api_version: ApiVersionSchema,
  server_index_version: IndexVersionSchema,
  server_policy_version: PolicyVersionSchema,
  warnings: z.array(z.string()).optional(),
  errors: z.array(z.string()).optional(),
});

export type CompatibilityCheckRequest = z.infer<typeof CompatibilityCheckRequestSchema>;
export type CompatibilityCheckResponse = z.infer<typeof CompatibilityCheckResponseSchema>;

// API endpoint definitions with SLA targets
export interface APIEndpoints {
  '/search': {
    method: 'POST';
    request: SearchRequest;
    response: SearchResponse;
    sla_ms: 20;  // p95 target
  };
  '/struct': {
    method: 'POST';
    request: StructRequest;
    response: SearchResponse;
    sla_ms: 30;
  };
  '/symbols/near': {
    method: 'POST';
    request: SymbolsNearRequest;
    response: SearchResponse;
    sla_ms: 15;
  };
  '/health': {
    method: 'GET';
    response: HealthResponse;
    sla_ms: 5;
  };
  '/compat/check': {
    method: 'GET';
    request: CompatibilityCheckRequest;
    response: CompatibilityCheckResponse;
    sla_ms: 10;
  };
}

// Search modes
export type SearchMode = 'lex' | 'struct' | 'hybrid';
export type HealthStatus = 'ok' | 'degraded' | 'down';
export type SupportedLanguage = 'typescript' | 'python' | 'rust' | 'bash' | 'go' | 'java';

// Precision optimization types
export const PrecisionOptimizationConfigSchema = z.object({
  block_a_early_exit: z.object({
    enabled: z.boolean(),
    margin: z.number().min(0).max(1),
    min_probes: z.number().int().min(16).max(512)
  }).optional(),
  block_a_ann: z.object({
    k: z.number().int().min(100).max(500),
    efSearch: z.number().int().min(16).max(512)
  }).optional(),
  block_a_gate: z.object({
    nl_threshold: z.number().min(0).max(1),
    min_candidates: z.number().int().min(1).max(100),
    confidence_cutoff: z.number().min(0).max(1)
  }).optional(),
  block_b_dynamic_topn: z.object({
    enabled: z.boolean(),
    score_threshold: z.number().min(0).max(1),
    hard_cap: z.number().int().min(5).max(50)
  }).optional(),
  block_c_dedup: z.object({
    in_file: z.object({
      simhash: z.object({
        k: z.number().int().min(3).max(10),
        hamming_max: z.number().int().min(1).max(5)
      }),
      keep: z.number().int().min(1).max(10)
    }),
    cross_file: z.object({
      vendor_deboost: z.number().min(0).max(1)
    })
  }).optional()
});

export type PrecisionOptimizationConfig = z.infer<typeof PrecisionOptimizationConfigSchema>;

// A/B experiment schemas
export const ExperimentConfigSchema = z.object({
  experiment_id: z.string().min(1),
  name: z.string().min(1),
  description: z.string().optional(),
  traffic_percentage: z.number().min(0).max(100),
  control_config: z.record(z.any()).optional(),
  treatment_config: z.record(z.any()),
  promotion_gates: z.object({
    min_ndcg_improvement_pct: z.number().min(0),
    min_recall_at_50: z.number().min(0).max(1),
    min_span_coverage_pct: z.number().min(0).max(100),
    max_latency_multiplier: z.number().min(1)
  }),
  anchor_validation_required: z.boolean().default(true),
  ladder_validation_required: z.boolean().default(true)
});

export type ExperimentConfig = z.infer<typeof ExperimentConfigSchema>;

// Validation results schemas  
export const ValidationResultSchema = z.object({
  validation_type: z.enum(['anchor', 'ladder']),
  passed: z.boolean(),
  metrics: z.object({
    ndcg_at_10_delta_pct: z.number(),
    recall_at_50: z.number().min(0).max(1),
    span_coverage_pct: z.number().min(0).max(100),
    p99_latency_ms: z.number().min(0),
    p95_latency_ms: z.number().min(0)
  }),
  gate_results: z.record(z.boolean()),
  timestamp: z.string()
});

export type ValidationResult = z.infer<typeof ValidationResultSchema>;

// Re-export from core for convenience
export type { SymbolKind, MatchReason } from './core.js';