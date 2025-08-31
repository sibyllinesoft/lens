// LENS - Bulletproof Production Architecture Specification
// Validated by Arbiter - Ready for AI Agent Implementation

package lens

#LensConfig: {
	// Performance SLAs - enforced by constraints
	performance: {
		stage_a_target_ms: int & >=2 & <=8      // Lexical+fuzzy (trigrams+FST)
		stage_b_target_ms: int & >=3 & <=10     // Symbol/AST (ctags+LSIF+tree-sitter)
		stage_c_target_ms: int & >=5 & <=15     // Semantic rerank (ColBERT-v2/SPLADE)
		overall_p95_ms: int & <=20              // End-to-end p95 target
		max_candidates: int & >=50 & <=200      // Top-K for rerank stage
	}
	
	// Resource boundaries - prevents exhaustion
	resources: {
		memory_limit_gb: int & >=4 & <=64       // Reasonable memory bounds
		max_concurrent_queries: int & >=10 & <=1000
		shard_size_limit_mb: int & >=100 & <=2048
		worker_pools: {
			ingest: int & >=2 & <=16             // NATS/JetStream workers
			query: int & >=4 & <=32              // Query processing parallelism
			maintenance: int & >=1 & <=4         // Compaction background work
		}
	}
	
	// Shard architecture constraints
	sharding: {
		strategy: "path_hash"                   // Consistent with design
		replication_factor: int & >=1 & <=3     // Local or replicated
		compaction_threshold_mb: int & >=100 & <=1024
		segments_per_shard: int & >=3 & <=5     // lexical+symbols+ast+(semantic)
	}
	
	// API contract enforcement
	api_limits: {
		max_query_length: int & >=100 & <=2000
		max_fuzzy_distance: int & >=0 & <=2     // ≤2-edit distance spec
		max_results_per_request: int & >=10 & <=500
		rate_limit_per_sec: int & >=10 & <=1000
	}
	
	// Technology stack validation
	tech_stack: {
		languages: ["typescript", "python", "rust", "bash"]
		messaging: "nats_jetstream"              // Work unit distribution
		storage: "memory_mapped_segments"        // Append-only with compaction
		observability: "opentelemetry"           // Full tracing/metrics
		semantic_models: ["colbert_v2", "splade_v2"] // Rerank options
	}
}

// PRODUCTION CONFIGURATION - Validated and Ready
lens_production: #LensConfig & {
	performance: {
		stage_a_target_ms: 5    // Trigram+FST target
		stage_b_target_ms: 7    // ctags+LSIF+tree-sitter target
		stage_c_target_ms: 12   // ColBERT-v2 rerank target
		overall_p95_ms: 20      // Total p95 SLA
		max_candidates: 200     // Rerank top-200
	}
	resources: {
		memory_limit_gb: 16     // 16GB for local NVMe box
		max_concurrent_queries: 100
		shard_size_limit_mb: 512
		worker_pools: {
			ingest: 4             // Balanced for NATS throughput
			query: 8              // Query parallelism
			maintenance: 2        // Background compaction
		}
	}
	sharding: {
		strategy: "path_hash"
		replication_factor: 1   // Single local node
		compaction_threshold_mb: 256
		segments_per_shard: 4   // All layers present
	}
	api_limits: {
		max_query_length: 1000
		max_fuzzy_distance: 2   // Exact specification
		max_results_per_request: 200
		rate_limit_per_sec: 50  // Reasonable for local use
	}
	tech_stack: {
		languages: ["typescript", "python", "rust", "bash"]
		messaging: "nats_jetstream"
		storage: "memory_mapped_segments"
		observability: "opentelemetry"
		semantic_models: ["colbert_v2", "splade_v2"]
	}
}

// API CONTRACT SPECIFICATIONS
#SearchRequest: {
	q: string & len(q) > 0 & len(q) <= 1000     // Query not empty, reasonable length
	mode: "lex" | "struct" | "hybrid"             // Valid search modes only
	fuzzy: int & >=0 & <=2                      // Edit distance 0-2
	k: int & >=1 & <=200                        // Top-K between 1-200
	timeout_ms?: int & >=100 & <=5000           // Optional timeout
}

#SearchResponse: {
	hits: [...#SearchHit]
	total: int & >=0
	latency_ms: {
		stage_a: int & >=0 & <=50      // Lexical stage timing
		stage_b: int & >=0 & <=50      // Symbol stage timing  
		stage_c?: int & >=0 & <=100    // Semantic stage (optional)
		total: int & >=0 & <=200       // Total latency
	}
	trace_id: string & =~"^[a-f0-9-]{36}$"   // UUID format
}

#SearchHit: {
	file: string & len(file) > 0
	line: int & >=1
	col: int & >=0
	ast_path?: string                         // Optional AST path
	symbol_kind?: "function" | "class" | "variable" | "type" | "interface"
	score: number & >=0 & <=1                // Normalized score
	why: [...("exact" | "symbol" | "struct" | "semantic")]  // Match reasons
}

#StructRequest: {
	pattern: string & len(pattern) > 0 & len(pattern) <= 500
	lang: "typescript" | "python" | "rust" | "bash" | "go" | "java"
	max_results?: int & >=1 & <=100
}

#SymbolsNearRequest: {
	file: string & len(file) > 0
	line: int & >=1
	radius?: int & >=1 & <=50     // Lines around target
}

// API ENDPOINT DEFINITIONS
api_endpoints: {
	"/search": {
		method: "POST"
		request: #SearchRequest
		response: #SearchResponse
		sla_ms: 20  // p95 target
	}
	"/struct": {
		method: "POST"
		request: #StructRequest
		response: #SearchResponse
		sla_ms: 30
	}
	"/symbols/near": {
		method: "POST" 
		request: #SymbolsNearRequest
		response: #SearchResponse
		sla_ms: 15
	}
	"/health": {
		method: "GET"
		response: {
			status: "ok" | "degraded" | "down"
			timestamp: string
			shards_healthy: int & >=0
		}
		sla_ms: 5
	}
}