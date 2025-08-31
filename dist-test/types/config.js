"use strict";
/**
 * Core configuration types derived from architecture.cue
 * Enforces production constraints at compile-time
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.PRODUCTION_CONFIG = void 0;
// Production configuration from architecture.cue
exports.PRODUCTION_CONFIG = {
    performance: {
        stage_a_target_ms: 5, // Trigram+FST target
        stage_b_target_ms: 7, // ctags+LSIF+tree-sitter target
        stage_c_target_ms: 12, // ColBERT-v2 rerank target
        overall_p95_ms: 20, // Total p95 SLA
        max_candidates: 200, // Rerank top-200
    },
    resources: {
        memory_limit_gb: 16, // 16GB for local NVMe box
        max_concurrent_queries: 100,
        shard_size_limit_mb: 512,
        worker_pools: {
            ingest: 4, // Balanced for NATS throughput
            query: 8, // Query parallelism
            maintenance: 2, // Background compaction
        },
    },
    sharding: {
        strategy: 'path_hash',
        replication_factor: 1, // Single local node
        compaction_threshold_mb: 256,
        segments_per_shard: 4, // All layers present
    },
    api_limits: {
        max_query_length: 1000,
        max_fuzzy_distance: 2, // Exact specification
        max_results_per_request: 200,
        rate_limit_per_sec: 50, // Reasonable for local use
    },
    tech_stack: {
        languages: ['typescript', 'python', 'rust', 'bash'],
        messaging: 'nats_jetstream',
        storage: 'memory_mapped_segments',
        observability: 'opentelemetry',
        semantic_models: ['colbert_v2', 'splade_v2'],
    },
};
