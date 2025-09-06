/**
 * Core domain types for the Lens search system
 * Three-layer processing pipeline: Lexical+Fuzzy → Symbol/AST → Semantic Rerank
 */

import type { SearchHit } from '../core/span_resolver/index.js';
import type { SupportedLanguage } from './api.js';

// Re-export SearchHit for use by other modules
export type { SearchHit };

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
  repo_sha: string;
  query: string;
  mode: SearchMode;
  k: number;
  fuzzy_distance: number;
  fuzzy?: boolean | undefined; // Alternative fuzzy flag used in some modules
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
  file?: string; // Alternative field name used in some modules
  line: number;
  col: number;
  score: number;
  match_reasons: MatchReason[];
  why?: string[] | undefined; // Human-readable match reasons  
  lang?: string | undefined;
  ast_path?: string | undefined;
  symbol_kind?: SymbolKind | undefined;
  // Extended fields for span-accurate resolution
  snippet?: string | undefined;
  byte_offset?: number | undefined;
  span_len?: number | undefined;
  context_before?: string | undefined;
  context_after?: string | undefined;
  context?: string | undefined; // Keep for backward compatibility
}

export type SearchMode = 'lex' | 'lexical' | 'struct' | 'hybrid';
export type MatchReason = 'exact' | 'fuzzy' | 'symbol' | 'struct' | 'semantic' | 'lsp_hint' | 'unicode_normalized' | 'raptor_diversity' | 'structural' | 'exact_name' | 'semantic_type';

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
  | 'health_check'
  | 'lsp_harvest'
  | 'lsp_sync';

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

// LSP-assist system types
export interface LSPHint {
  symbol_id: string;
  name: string;
  kind: SymbolKind;
  file_path: string;
  line: number;
  col: number;
  definition_uri?: string;
  signature?: string;
  type_info?: string;
  aliases: string[];
  resolved_imports: string[];
  references_count: number;
}

export interface LSPSidecarConfig {
  language: SupportedLanguage;
  lsp_server: string;
  capabilities: LSPCapabilities;
  workspace_config: WorkspaceConfig;
  harvest_ttl_hours: number;
  pressure_threshold: number;
}

export interface LSPCapabilities {
  definition: boolean;
  references: boolean;
  hover: boolean;
  completion: boolean;
  rename: boolean;
  workspace_symbols: boolean;
}

export interface WorkspaceConfig {
  root_path?: string | undefined; // Make optional to allow partial configs
  include_patterns: string[];
  exclude_patterns: string[];
  path_mappings?: Map<string, string> | undefined; // Make optional
  config_files?: string[] | undefined; // Make optional
}

export interface LSPFeatures {
  lsp_def_hit: 0 | 1;
  lsp_ref_count: number;
  type_match: number;
  alias_resolved: 0 | 1;
}

export type QueryIntent = 'def' | 'refs' | 'symbol' | 'struct' | 'lexical' | 'NL';

export interface IntentClassification {
  intent: QueryIntent;
  confidence: number;
  features: {
    has_definition_pattern: boolean;
    has_reference_pattern: boolean;
    has_symbol_prefix: boolean;
    has_structural_chars: boolean;
    is_natural_language: boolean;
  };
}

export interface LSPBenchmarkResult {
  mode: 'baseline' | 'lsp_assist' | 'competitor_lsp';
  task_type: QueryIntent;
  success_at_1: number;
  success_at_5: number;
  ndcg_at_10: number;
  recall_at_50: number;
  zero_result_rate: number;
  timeout_rate: number;
  p95_latency_ms: number;
  loss_taxonomy: LossTaxonomy;
}

export interface LossTaxonomy {
  NO_SYM_COVERAGE: number;
  WRONG_ALIAS: number;
  PATH_MAP: number;
  USABILITY_INTENT: number;
  RANKING_ONLY: number;
}

// Search result interface
export interface SearchResult {
  hits: SearchHit[];
  stage_a_latency?: number;
  stage_b_latency?: number;
  stage_c_latency?: number;
  stage_a_skipped?: boolean;
  stage_b_skipped?: boolean;
  stage_c_skipped?: boolean;
}

// Additional interfaces needed by various modules
export interface TestFailure {
  test_name: string;
  error_message: string;
  file_path: string;
  line_number: number;
  timestamp: Date;
  stack_trace?: string;
}

export interface ChangeEvent {
  event_type: 'file_added' | 'file_modified' | 'file_deleted' | 'file_renamed';
  file_path: string;
  old_file_path?: string; // For rename events
  timestamp: Date;
  change_id: string;
  metadata?: any;
}

export interface CodeOwner {
  email: string;
  username: string;
  file_patterns: string[];
  team?: string;
  role?: string;
  last_updated: Date;
}

// SupportedLanguage export (referenced by lsp-sidecar)
export type SupportedLanguage = 'typescript' | 'javascript' | 'python' | 'rust' | 'go' | 'java' | 'cpp' | 'c' | 'csharp' | 'php' | 'ruby' | 'scala' | 'kotlin' | 'swift' | 'dart' | 'lua' | 'r' | 'shell' | 'yaml' | 'json' | 'markdown' | 'html' | 'css' | 'sql';