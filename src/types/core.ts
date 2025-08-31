/**
 * Core domain types for the Lens search system
 * Three-layer processing pipeline: Lexical+Fuzzy → Symbol/AST → Semantic Rerank
 */

// Shard and segment types
export interface Shard {
  id: string;
  path_hash: string;
  segments: Segment[];
  status: ShardStatus;
  last_compacted: Date;
  size_mb: number;
}

export interface Segment {
  id: string;
  type: SegmentType;
  file_path: string;
  size_bytes: number;
  memory_mapped: boolean;
  last_accessed: Date;
}

export type SegmentType = 'lexical' | 'symbols' | 'ast' | 'semantic';
export type ShardStatus = 'active' | 'compacting' | 'inactive' | 'error';

// Index structures for Layer 1 (Lexical+Fuzzy)
export interface TrigramIndex {
  trigrams: Map<string, Set<DocumentPosition>>;
  fst: FST; // Finite State Transducer for fuzzy search
}

export interface FST {
  states: FSTState[];
  transitions: Map<string, FSTTransition[]>;
}

export interface FSTState {
  id: number;
  is_final: boolean;
  edit_distance: number;
}

export interface FSTTransition {
  from_state: number;
  to_state: number;
  input_char?: string;
  output_char?: string;
  cost: number;
}

export interface DocumentPosition {
  doc_id: string;
  file_path: string;
  line: number;
  col: number;
  length: number;
}

// Symbol and AST types for Layer 2
export interface SymbolIndex {
  definitions: Map<string, SymbolDefinition[]>;
  references: Map<string, SymbolReference[]>;
  ast_nodes: Map<string, ASTNode[]>;
}

export interface SymbolDefinition {
  name: string;
  kind: SymbolKind;
  file_path: string;
  line: number;
  col: number;
  scope: string;
  signature?: string;
}

export interface SymbolReference {
  symbol_name: string;
  file_path: string;
  line: number;
  col: number;
  context: string;
}

export interface ASTNode {
  id: string;
  type: string;
  file_path: string;
  start_line: number;
  start_col: number;
  end_line: number;
  end_col: number;
  parent_id?: string;
  children_ids: string[];
  text: string;
}

export type SymbolKind = 
  | 'function' 
  | 'class' 
  | 'variable' 
  | 'type' 
  | 'interface'
  | 'constant'
  | 'enum'
  | 'method'
  | 'property';

// Semantic types for Layer 3 (Rerank)
export interface SemanticIndex {
  vectors: Map<string, Float32Array>; // ColBERT or SPLADE vectors
  hnsw_index?: HNSWIndex;
}

export interface HNSWIndex {
  layers: HNSWLayer[];
  entry_point: number;
  max_connections: number;
  level_multiplier: number;
}

export interface HNSWLayer {
  level: number;
  nodes: Map<number, HNSWNode>;
}

export interface HNSWNode {
  id: number;
  vector: Float32Array;
  connections: Set<number>;
}

// Search processing types
export interface SearchContext {
  trace_id: string;
  query: string;
  mode: SearchMode;
  k: number;
  fuzzy_distance: number;
  started_at: Date;
  stages: StageResult[];
}

export interface StageResult {
  stage: 'stage_a' | 'stage_b' | 'stage_c';
  latency_ms: number;
  candidates_in: number;
  candidates_out: number;
  method: string;
  error?: string;
}

export interface Candidate {
  doc_id: string;
  file_path: string;
  line: number;
  col: number;
  score: number;
  match_reasons: MatchReason[];
  ast_path?: string;
  symbol_kind?: SymbolKind;
  context?: string;
}

export type SearchMode = 'lex' | 'struct' | 'hybrid';
export type MatchReason = 'exact' | 'symbol' | 'struct' | 'semantic';

// Work units for NATS/JetStream
export interface WorkUnit {
  id: string;
  type: WorkType;
  repo_sha: string;
  shard_id: string;
  payload: unknown;
  priority: number;
  created_at: Date;
  assigned_to?: string;
}

export type WorkType = 
  | 'index_shard'
  | 'compact_shard'
  | 'build_symbols'
  | 'build_ast'
  | 'build_semantic'
  | 'health_check';

// File-based segment interface (simulated memory mapping for extFAT compatibility)
export interface MMapSegment {
  file_path: string;
  fd: number;
  size: number;
  buffer: Buffer; // In-memory buffer simulating mmap
  readonly: boolean;
}

// Telemetry and monitoring
export interface SearchMetrics {
  total_queries: number;
  queries_by_mode: Map<SearchMode, number>;
  stage_latencies: {
    stage_a_p50: number;
    stage_a_p95: number;
    stage_b_p50: number;
    stage_b_p95: number;
    stage_c_p50: number;
    stage_c_p95: number;
    total_p50: number;
    total_p95: number;
  };
  cache_hit_rates: {
    trigram_cache: number;
    symbol_cache: number;
    semantic_cache: number;
  };
  error_rates: Map<string, number>;
}

export interface SystemHealth {
  status: HealthStatus;
  shards_healthy: number;
  shards_total: number;
  memory_usage_gb: number;
  active_queries: number;
  worker_pool_status: {
    ingest_active: number;
    query_active: number;
    maintenance_active: number;
  };
  last_compaction: Date;
}

export type HealthStatus = 'ok' | 'degraded' | 'down';