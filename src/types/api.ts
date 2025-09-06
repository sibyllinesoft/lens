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

// Search SPI schemas (Phase 0)
export const SpiSearchRequestSchema = SearchRequestSchema.extend({
  scopes: z.array(z.string()).optional(),                    // Repository scopes/path globs
  constraints: z.object({
    lang: z.string().optional(),                             // Language filter
    symbol_kind: z.enum(['function', 'class', 'variable', 'type', 'interface', 'constant', 'enum', 'method', 'property']).optional(),
  }).optional(),
  budget_ms: z.number().int().min(50).max(5000).optional(),  // Budget in milliseconds
  token_budget: z.number().int().min(100).max(10000).optional().default(10000), // Token budget for results (default 10k)
  page: z.number().int().min(0).optional().default(0),       // Page number for pagination (0-indexed)
  return: z.object({
    ast_path: z.boolean().optional(),                        // Return AST path
    neighbors: z.boolean().optional(),                       // Return neighboring symbols
    why: z.boolean().optional(),                             // Return match reasoning
  }).optional(),
});

export type SpiSearchRequest = z.infer<typeof SpiSearchRequestSchema>;

// Extended search hit with SPI features (Phase 1)
export const SpiSearchHitSchema = SearchHitSchema.extend({
  ref: z.string().optional(),                                // Stable reference: lens://{repo_sha}/{file}@{source_hash}#L{start}:{end}|B{byte_start}:{byte_end}|AST:{path}
  source_hash: z.string().optional(),                        // File content hash for integrity
  byte_start: z.number().int().min(0).optional(),            // Byte start position
  byte_end: z.number().int().min(0).optional(),              // Byte end position
});

export type SpiSearchHit = z.infer<typeof SpiSearchHitSchema>;

export const SpiSearchResponseSchema = SearchResponseSchema.extend({
  hits: z.array(SpiSearchHitSchema),
  timed_out: z.boolean().optional(),                         // Budget timeout occurred
  token_usage: z.object({
    used_tokens: z.number().int().min(0),                    // Tokens used in this response
    budget_tokens: z.number().int().min(0),                  // Total token budget
    budget_exceeded: z.boolean(),                            // True if budget was hit and results truncated
    estimated_total_tokens: z.number().int().min(0).optional(), // Estimated tokens if all results were included
  }),
  pagination: z.object({
    page: z.number().int().min(0),                           // Current page (0-indexed)
    results_in_page: z.number().int().min(0),                // Results returned in this page
    total_results: z.number().int().min(0),                  // Total available results (if known)
    has_next_page: z.boolean(),                              // True if more pages available
    next_page: z.number().int().min(0).optional(),           // Next page number if available
    budget_per_page: z.number().int().min(0),                // Token budget used for page sizing
  }),
});

export type SpiSearchResponse = z.infer<typeof SpiSearchResponseSchema>;

// SPI Health response with extended SLA fields
export const SpiHealthResponseSchema = HealthResponseSchema.extend({
  sla: z.object({
    p95_latency_ms: z.number().min(0),
    p99_latency_ms: z.number().min(0),
    availability_pct: z.number().min(0).max(100),
    error_rate_pct: z.number().min(0).max(100),
  }).optional(),
  version_info: z.object({
    api_version: ApiVersionSchema,
    index_version: IndexVersionSchema,
    policy_version: PolicyVersionSchema,
  }).optional(),
});

export type SpiHealthResponse = z.infer<typeof SpiHealthResponseSchema>;

// Resolve endpoint schemas (Phase 1)
export const ResolveRequestSchema = z.object({
  ref: z.string().min(1),                                    // Lens reference to resolve
});

export const ResolveResponseSchema = z.object({
  ref: z.string(),
  file_path: z.string(),
  content: z.string(),                                       // Exact slice content
  source_hash: z.string(),                                   // File hash for integrity
  line_start: z.number().int().min(1),
  line_end: z.number().int().min(1),
  byte_start: z.number().int().min(0),
  byte_end: z.number().int().min(0),
  ast_path: z.string().optional(),
  surrounding_lines: z.object({
    before: z.array(z.string()),
    after: z.array(z.string()),
  }).optional(),
  metadata: z.object({
    lang: z.string().optional(),
    symbol_kind: z.enum(['function', 'class', 'variable', 'type', 'interface', 'constant', 'enum', 'method', 'property']).optional(),
    symbol_name: z.string().optional(),
  }).optional(),
});

export type ResolveRequest = z.infer<typeof ResolveRequestSchema>;
export type ResolveResponse = z.infer<typeof ResolveResponseSchema>;

// Context endpoint schemas (Phase 2)
export const ContextRequestSchema = z.object({
  refs: z.array(z.string()).min(1),                          // List of refs to resolve
  token_budget: z.number().int().min(100).max(10000).optional().default(10000), // Token budget (default 10k)
  page: z.number().int().min(0).optional().default(0),       // Page number for pagination (0-indexed)
  dedupe: z.boolean().optional().default(true),              // Deduplicate overlapping content
});

export const ContextResponseSchema = z.object({
  contexts: z.array(z.object({
    ref: z.string(),
    content: z.string(),
    token_count: z.number().int().min(0),
    truncated: z.boolean(),
  })),
  total_tokens: z.number().int().min(0),
  deduped_count: z.number().int().min(0).optional(),
});

export type ContextRequest = z.infer<typeof ContextRequestSchema>;
export type ContextResponse = z.infer<typeof ContextResponseSchema>;

// Cross-reference endpoint schemas (Phase 2) 
export const XrefRequestSchema = z.object({
  ref: z.string().optional(),                                // Ref to find xrefs for
  symbol_id: z.string().optional(),                          // Or symbol ID
  types: z.array(z.enum(['definitions', 'references', 'implementations'])).optional().default(['definitions', 'references']),
});

export const XrefResponseSchema = z.object({
  symbol_id: z.string().optional(),
  symbol_name: z.string().optional(),
  definitions: z.array(SpiSearchHitSchema),
  references: z.array(SpiSearchHitSchema),
  implementations: z.array(SpiSearchHitSchema).optional(),
  total: z.number().int().min(0),
});

export type XrefRequest = z.infer<typeof XrefRequestSchema>;
export type XrefResponse = z.infer<typeof XrefResponseSchema>;

// Symbols listing endpoint (Phase 2)
export const SymbolsListRequestSchema = z.object({
  repo_sha: z.string().min(1).max(64),
  page: z.number().int().min(0).optional().default(0),
  page_size: z.number().int().min(10).max(1000).optional().default(100),
  kind_filter: z.enum(['function', 'class', 'variable', 'type', 'interface', 'constant', 'enum', 'method', 'property']).optional(),
  lang_filter: z.string().optional(),
});

export const SymbolsListResponseSchema = z.object({
  symbols: z.array(z.object({
    symbol_id: z.string(),
    name: z.string(),
    kind: z.enum(['function', 'class', 'variable', 'type', 'interface', 'constant', 'enum', 'method', 'property']),
    file_path: z.string(),
    line: z.number().int().min(1),
    ref: z.string(),
    lang: z.string().optional(),
  })),
  total: z.number().int().min(0),
  page: z.number().int().min(0),
  page_size: z.number().int().min(10).max(1000),
  has_next: z.boolean(),
});

export type SymbolsListRequest = z.infer<typeof SymbolsListRequestSchema>;
export type SymbolsListResponse = z.infer<typeof SymbolsListResponseSchema>;

/*** LSP SPI SCHEMAS (NEW) ***/

// LSP Capabilities Schema
export const LSPCapabilitiesResponseSchema = z.object({
  languages: z.array(z.object({
    lang: z.string(),
    features: z.array(z.enum([
      'diagnostics', 'format', 'selectionRanges', 'foldingRanges', 
      'prepareRename', 'rename', 'codeActions', 'callHierarchy', 'typeHierarchy'
    ]))
  }))
});

export type LSPCapabilitiesResponse = z.infer<typeof LSPCapabilitiesResponseSchema>;

// Base LSP request with budget enforcement
export const LSPBaseRequestSchema = z.object({
  budget_ms: z.number().int().min(100).max(30000).optional(),
});

// LSP Diagnostics Schema
export const LSPDiagnosticsRequestSchema = LSPBaseRequestSchema.extend({
  files: z.array(z.object({
    path: z.string().min(1),
    source_hash: z.string().min(1)
  }))
});

export const LSPDiagnosticsResponseSchema = z.object({
  diags: z.array(z.object({
    path: z.string().min(1),
    source_hash: z.string().min(1),
    items: z.array(z.object({
      range: z.object({
        b0: z.number().int().min(0),
        b1: z.number().int().min(0)
      }),
      severity: z.enum(['hint', 'info', 'warning', 'error']),
      code: z.string().optional(),
      message: z.string()
    }))
  })),
  duration_ms: z.number().int().min(0),
  timed_out: z.boolean().optional()
});

// LSP Format Schema
export const LSPFormatRequestSchema = LSPBaseRequestSchema.extend({
  ref: z.string().optional(), // lens:// ref format
  path: z.string().optional(),
  range: z.object({
    b0: z.number().int().min(0),
    b1: z.number().int().min(0)
  }).optional(),
  options: z.record(z.any()).optional() // Formatting options
});

export const LSPFormatResponseSchema = z.object({
  edits: z.array(z.object({
    path: z.string().min(1),
    range: z.object({
      b0: z.number().int().min(0),
      b1: z.number().int().min(0)
    }),
    new_text: z.string()
  })),
  idempotent: z.boolean(),
  duration_ms: z.number().int().min(0)
});

// LSP Selection Ranges Schema
export const LSPSelectionRangesRequestSchema = LSPBaseRequestSchema.extend({
  refs: z.array(z.string()) // Array of lens:// refs
});

export const LSPSelectionRangesResponseSchema = z.object({
  chains: z.array(z.array(z.object({
    range: z.object({
      b0: z.number().int().min(0),
      b1: z.number().int().min(0)
    }),
    parent_ix: z.number().int().min(0).optional()
  }))),
  duration_ms: z.number().int().min(0)
});

// LSP Folding Ranges Schema
export const LSPFoldingRangesRequestSchema = LSPBaseRequestSchema.extend({
  files: z.array(z.object({
    path: z.string().min(1),
    source_hash: z.string().min(1)
  }))
});

export const LSPFoldingRangesResponseSchema = z.object({
  folds: z.array(z.object({
    path: z.string().min(1),
    ranges: z.array(z.object({
      b0: z.number().int().min(0),
      b1: z.number().int().min(0),
      kind: z.enum(['comment', 'imports', 'region']).optional()
    }))
  })),
  duration_ms: z.number().int().min(0)
});

// LSP Prepare Rename Schema
export const LSPPrepareRenameRequestSchema = LSPBaseRequestSchema.extend({
  ref: z.string() // lens:// ref
});

export const LSPPrepareRenameResponseSchema = z.object({
  allowed: z.boolean(),
  placeholder: z.string().optional(),
  range: z.object({
    b0: z.number().int().min(0),
    b1: z.number().int().min(0)
  }).optional(),
  reason: z.string().optional(),
  duration_ms: z.number().int().min(0)
});

// LSP Rename Schema
export const LSPRenameRequestSchema = LSPBaseRequestSchema.extend({
  ref: z.string(), // lens:// ref
  new_name: z.string().min(1)
});

export const LSPRenameResponseSchema = z.object({
  workspaceEdit: z.object({
    changes: z.array(z.object({
      path: z.string().min(1),
      source_hash: z.string().min(1),
      edits: z.array(z.object({
        b0: z.number().int().min(0),
        b1: z.number().int().min(0),
        new_text: z.string()
      }))
    }))
  }),
  duration_ms: z.number().int().min(0)
});

// LSP Code Actions Schema
export const LSPCodeActionsRequestSchema = LSPBaseRequestSchema.extend({
  ref: z.string(), // lens:// ref
  kinds: z.array(z.string()).optional(), // ["quickfix", "refactor", "source.organizeImports", ...]
  diagnostics: z.array(z.any()).optional() // Related diagnostics
});

export const LSPCodeActionsResponseSchema = z.object({
  actions: z.array(z.object({
    title: z.string(),
    kind: z.string(),
    workspaceEdit: z.object({
      changes: z.array(z.object({
        path: z.string().min(1),
        source_hash: z.string().min(1),
        edits: z.array(z.object({
          b0: z.number().int().min(0),
          b1: z.number().int().min(0),
          new_text: z.string()
        }))
      }))
    }).optional(),
    data: z.any().optional()
  })),
  duration_ms: z.number().int().min(0)
});

// LSP Hierarchy Schema
export const LSPHierarchyRequestSchema = LSPBaseRequestSchema.extend({
  ref: z.string(), // lens:// ref
  kind: z.enum(['call', 'type']),
  dir: z.enum(['incoming', 'outgoing']),
  depth: z.number().int().min(1).max(5).optional(),
  fanout_cap: z.number().int().min(10).max(1000).optional()
});

export const LSPHierarchyResponseSchema = z.object({
  nodes: z.array(z.object({
    symbol_id: z.string(),
    name: z.string(),
    kind: z.string(),
    def_ref: z.string().optional() // lens:// ref to definition
  })),
  edges: z.array(z.object({
    src: z.string(), // symbol_id
    dst: z.string(), // symbol_id
    role: z.string()
  })),
  truncated: z.boolean(),
  duration_ms: z.number().int().min(0)
});

// LensSymbol Schema
export const LensSymbolSchema = z.object({
  symbol_id: z.string(),
  lang: z.string(),
  name: z.string(),
  kind: z.string(),
  def_ref: z.string().optional(), // lens:// ref to definition
  container: z.array(z.string()), // Container hierarchy
  moniker: z.string().optional() // Stable cross-repo identifier
});

export type LSPDiagnosticsRequest = z.infer<typeof LSPDiagnosticsRequestSchema>;
export type LSPDiagnosticsResponse = z.infer<typeof LSPDiagnosticsResponseSchema>;
export type LSPFormatRequest = z.infer<typeof LSPFormatRequestSchema>;
export type LSPFormatResponse = z.infer<typeof LSPFormatResponseSchema>;
export type LSPSelectionRangesRequest = z.infer<typeof LSPSelectionRangesRequestSchema>;
export type LSPSelectionRangesResponse = z.infer<typeof LSPSelectionRangesResponseSchema>;
export type LSPFoldingRangesRequest = z.infer<typeof LSPFoldingRangesRequestSchema>;
export type LSPFoldingRangesResponse = z.infer<typeof LSPFoldingRangesResponseSchema>;
export type LSPPrepareRenameRequest = z.infer<typeof LSPPrepareRenameRequestSchema>;
export type LSPPrepareRenameResponse = z.infer<typeof LSPPrepareRenameResponseSchema>;
export type LSPRenameRequest = z.infer<typeof LSPRenameRequestSchema>;
export type LSPRenameResponse = z.infer<typeof LSPRenameResponseSchema>;
export type LSPCodeActionsRequest = z.infer<typeof LSPCodeActionsRequestSchema>;
export type LSPCodeActionsResponse = z.infer<typeof LSPCodeActionsResponseSchema>;
export type LSPHierarchyRequest = z.infer<typeof LSPHierarchyRequestSchema>;
export type LSPHierarchyResponse = z.infer<typeof LSPHierarchyResponseSchema>;
export type LensSymbol = z.infer<typeof LensSymbolSchema>;

// Re-export from core for convenience
export type { SymbolKind, MatchReason } from './core.js';