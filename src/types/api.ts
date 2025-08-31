/**
 * API types derived from architecture.cue
 * Enforces request/response contracts at compile-time
 */

import { z } from 'zod';

// Search request schema matching CUE specification
export const SearchRequestSchema = z.object({
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
  ast_path: z.string().optional(),                         // Optional AST path
  symbol_kind: z.enum(['function', 'class', 'variable', 'type', 'interface', 'constant', 'enum', 'method', 'property']).optional(),
  score: z.number().min(0).max(1),                         // Normalized score
  why: z.array(z.enum(['exact', 'symbol', 'struct', 'semantic'])), // Match reasons
});

export type SearchHit = z.infer<typeof SearchHitSchema>;

// Search response schema
export const SearchResponseSchema = z.object({
  hits: z.array(SearchHitSchema),
  total: z.number().int().min(0),
  latency_ms: z.object({
    stage_a: z.number().int().min(0).max(50),               // Lexical stage timing
    stage_b: z.number().int().min(0).max(50),               // Symbol stage timing  
    stage_c: z.number().int().min(0).max(100).optional(),   // Semantic stage (optional)
    total: z.number().int().min(0).max(200),                // Total latency
  }),
  trace_id: z.string().min(1).max(100),                   // Flexible trace ID format
});

export type SearchResponse = z.infer<typeof SearchResponseSchema>;

// Structural search request
export const StructRequestSchema = z.object({
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
}

// Search modes
export type SearchMode = 'lex' | 'struct' | 'hybrid';
export type HealthStatus = 'ok' | 'degraded' | 'down';
export type SupportedLanguage = 'typescript' | 'python' | 'rust' | 'bash' | 'go' | 'java';

// Re-export from core for convenience
export type { SymbolKind, MatchReason } from './core.js';