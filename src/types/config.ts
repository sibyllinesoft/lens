/**
 * Core configuration types derived from architecture.cue
 * Enforces production constraints at compile-time
 */

export interface LensConfig {
  performance: {
    stage_a_target_ms: number;    // 2-8ms: Lexical+fuzzy (trigrams+FST)
    stage_b_target_ms: number;    // 3-10ms: Symbol/AST (ctags+LSIF+tree-sitter)
    stage_c_target_ms: number;    // 5-15ms: Semantic rerank (ColBERT-v2/SPLADE)
    overall_p95_ms: number;       // <=20ms: End-to-end p95 target
    max_candidates: number;       // 50-200: Top-K for rerank stage
  };
  
  resources: {
    memory_limit_gb: number;      // 4-64GB: Reasonable memory bounds
    max_concurrent_queries: number; // 10-1000: Query parallelism
    shard_size_limit_mb: number;  // 100-2048MB: Shard size limits
    worker_pools: {
      ingest: number;             // 2-16: NATS/JetStream workers
      query: number;              // 4-32: Query processing parallelism
      maintenance: number;        // 1-4: Compaction background work
    };
  };
  
  sharding: {
    strategy: 'path_hash';        // Consistent with design
    replication_factor: number;   // 1-3: Local or replicated
    compaction_threshold_mb: number; // 100-1024MB
    segments_per_shard: number;   // 3-5: lexical+symbols+ast+(semantic)
  };
  
  api_limits: {
    max_query_length: number;     // 100-2000: Query length limits
    max_fuzzy_distance: number;   // 0-2: ≤2-edit distance spec
    max_results_per_request: number; // 10-500: Results per request
    rate_limit_per_sec: number;   // 10-1000: Rate limiting
  };
  
  tech_stack: {
    languages: ['typescript', 'python', 'rust', 'bash'];
    messaging: 'nats_jetstream';
    storage: 'memory_mapped_segments';
    observability: 'opentelemetry';
    semantic_models: ['colbert_v2', 'splade_v2'];
  };
}

// Production configuration from architecture.cue
export const PRODUCTION_CONFIG: LensConfig = {
  performance: {
    stage_a_target_ms: 5,    // Trigram+FST target
    stage_b_target_ms: 7,    // ctags+LSIF+tree-sitter target
    stage_c_target_ms: 12,   // ColBERT-v2 rerank target
    overall_p95_ms: 20,      // Total p95 SLA
    max_candidates: 200,     // Rerank top-200
  },
  resources: {
    memory_limit_gb: 16,     // 16GB for local NVMe box
    max_concurrent_queries: 100,
    shard_size_limit_mb: 512,
    worker_pools: {
      ingest: 4,             // Balanced for NATS throughput
      query: 8,              // Query parallelism
      maintenance: 2,        // Background compaction
    },
  },
  sharding: {
    strategy: 'path_hash',
    replication_factor: 1,   // Single local node
    compaction_threshold_mb: 256,
    segments_per_shard: 4,   // All layers present
  },
  api_limits: {
    max_query_length: 1000,
    max_fuzzy_distance: 2,   // Exact specification
    max_results_per_request: 200,
    rate_limit_per_sec: 50,  // Reasonable for local use
  },
  tech_stack: {
    languages: ['typescript', 'python', 'rust', 'bash'],
    messaging: 'nats_jetstream',
    storage: 'memory_mapped_segments',
    observability: 'opentelemetry',
    semantic_models: ['colbert_v2', 'splade_v2'],
  },
} as const;