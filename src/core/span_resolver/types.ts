/**
 * Types for span-accurate hit resolution
 * Ensures all search stages return proper {file,line,col} coordinates
 */

// Re-export SearchContext from core types
export type { SearchContext } from '../../types/core.js';

export interface SearchHit {
  file: string;
  line: number;        // 1-based line number
  col: number;         // 0-based column (Unicode code points)
  lang?: string | undefined;
  snippet?: string | undefined;
  score: number;
  why: MatchReason[];
  match_reasons?: MatchReason[] | undefined; // Alternative field name used in some modules
  
  // Optional metadata
  ast_path?: string | undefined;
  symbol_kind?: SymbolKind | undefined;
  byte_offset?: number | undefined;
  span_len?: number | undefined;
  context_before?: string | undefined;
  context_after?: string | undefined;
  
  // Additional fields referenced in codebase
  original_line?: number | undefined;
  revision_sha?: string | undefined;
  translation_applied?: boolean | undefined;
  
  // Structural search metadata
  pattern_type?: 'function_def' | 'class_def' | 'import' | 'async_def' | 'decorator' | 'try_except' | 'for_loop' | 'if_statement' | undefined;
  symbol_name?: string | undefined;
  signature?: string | undefined;
  
  // Extended properties for compatibility with various modules
  file_path?: string | undefined;     // Alternative to 'file' for some modules
  language?: string | undefined;      // Alternative to 'lang' for some modules
  session_boost?: number | undefined; // Session-aware retrieval boost
  document_path?: string | undefined; // Document path reference
  metadata?: any;                     // Generic metadata container
  repository?: string | undefined;    // Repository reference
  name?: string | undefined;          // Symbol or file name
  content?: string | undefined;       // Content or context alternative
  context?: string | undefined;       // Context content
  symbolType?: string | undefined;    // Symbol type classification
  filePath?: string | undefined;      // CamelCase file path alternative
  id?: string | undefined;            // Unique identifier
}

export type MatchReason = 'exact' | 'fuzzy' | 'symbol' | 'struct' | 'semantic' | 'lsp_hint' | 'unicode_normalized' | 'raptor_diversity' | 'structural' | 'exact_name' | 'semantic_type' | 'nl_bridge' | 'raptor_hierarchical' | string;

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

// Internal types for span resolution
export interface SpanLocation {
  line: number;        // 1-based
  col: number;         // 0-based (code points)
  byte_offset?: number;
  span_len?: number;
}

export interface ContextLines {
  context_before?: string | undefined;
  context_after?: string | undefined;
}

// Stage-specific input formats
export interface LexicalCandidate {
  file_path: string;
  score: number;
  match_reasons: MatchReason[];
}

export interface SymbolCandidate {
  file_path: string;
  score: number;
  match_reasons: MatchReason[];
  symbol_kind?: SymbolKind | undefined;
  symbol_name?: string | undefined;
  ast_path?: string | undefined;
  // Upstream coordinates from Stage A (if available)
  upstream_line?: number | undefined;
  upstream_col?: number | undefined;
}

export interface SemanticCandidate {
  file_path: string;
  score: number;
  match_reasons: MatchReason[];
  symbol_kind?: SymbolKind | undefined;
  ast_path?: string | undefined;
  // Upstream coordinates from Stage A or B (required for pass-through)
  upstream_line: number;
  upstream_col: number;
  upstream_snippet?: string | undefined;
  upstream_context_before?: string | undefined;
  upstream_context_after?: string | undefined;
}