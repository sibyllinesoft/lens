/**
 * Types for span-accurate hit resolution
 * Ensures all search stages return proper {file,line,col} coordinates
 */

export interface SearchHit {
  file: string;
  line: number;        // 1-based line number
  col: number;         // 0-based column (Unicode code points)
  lang?: string | undefined;
  snippet?: string | undefined;
  score: number;
  why: MatchReason[];
  
  // Optional metadata
  ast_path?: string | undefined;
  symbol_kind?: SymbolKind | undefined;
  byte_offset?: number | undefined;
  span_len?: number | undefined;
  context_before?: string | undefined;
  context_after?: string | undefined;
  
  // Structural search metadata
  pattern_type?: 'function_def' | 'class_def' | 'import' | 'async_def' | 'decorator' | 'try_except' | 'for_loop' | 'if_statement' | undefined;
  symbol_name?: string | undefined;
  signature?: string | undefined;
}

export type MatchReason = 'exact' | 'fuzzy' | 'symbol' | 'struct' | 'structural' | 'semantic';

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