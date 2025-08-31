"use strict";
/**
 * API types derived from architecture.cue
 * Enforces request/response contracts at compile-time
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.HealthResponseSchema = exports.SymbolsNearRequestSchema = exports.StructRequestSchema = exports.SearchResponseSchema = exports.SearchHitSchema = exports.SearchRequestSchema = void 0;
const zod_1 = require("zod");
// Search request schema matching CUE specification
exports.SearchRequestSchema = zod_1.z.object({
    q: zod_1.z.string().min(1).max(1000), // Query not empty, reasonable length
    mode: zod_1.z.enum(['lex', 'struct', 'hybrid']), // Valid search modes only
    fuzzy: zod_1.z.number().int().min(0).max(2), // Edit distance 0-2
    k: zod_1.z.number().int().min(1).max(200), // Top-K between 1-200
    timeout_ms: zod_1.z.number().int().min(100).max(5000).optional(), // Optional timeout
});
// Search hit schema
exports.SearchHitSchema = zod_1.z.object({
    file: zod_1.z.string().min(1), // File path not empty
    line: zod_1.z.number().int().min(1), // Line number >= 1
    col: zod_1.z.number().int().min(0), // Column >= 0
    ast_path: zod_1.z.string().optional(), // Optional AST path
    symbol_kind: zod_1.z.enum(['function', 'class', 'variable', 'type', 'interface']).optional(),
    score: zod_1.z.number().min(0).max(1), // Normalized score
    why: zod_1.z.array(zod_1.z.enum(['exact', 'symbol', 'struct', 'semantic'])), // Match reasons
});
// Search response schema
exports.SearchResponseSchema = zod_1.z.object({
    hits: zod_1.z.array(exports.SearchHitSchema),
    total: zod_1.z.number().int().min(0),
    latency_ms: zod_1.z.object({
        stage_a: zod_1.z.number().int().min(0).max(50), // Lexical stage timing
        stage_b: zod_1.z.number().int().min(0).max(50), // Symbol stage timing  
        stage_c: zod_1.z.number().int().min(0).max(100).optional(), // Semantic stage (optional)
        total: zod_1.z.number().int().min(0).max(200), // Total latency
    }),
    trace_id: zod_1.z.string().min(1).max(100), // Flexible trace ID format
});
// Structural search request
exports.StructRequestSchema = zod_1.z.object({
    pattern: zod_1.z.string().min(1).max(500),
    lang: zod_1.z.enum(['typescript', 'python', 'rust', 'bash', 'go', 'java']),
    max_results: zod_1.z.number().int().min(1).max(100).optional(),
});
// Symbols near request
exports.SymbolsNearRequestSchema = zod_1.z.object({
    file: zod_1.z.string().min(1),
    line: zod_1.z.number().int().min(1),
    radius: zod_1.z.number().int().min(1).max(50).optional(), // Lines around target
});
// Health response
exports.HealthResponseSchema = zod_1.z.object({
    status: zod_1.z.enum(['ok', 'degraded', 'down']),
    timestamp: zod_1.z.string(),
    shards_healthy: zod_1.z.number().int().min(0),
});
