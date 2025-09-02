/**
 * Span Resolver - Central module for span-accurate hit resolution
 * Ensures all search stages return proper {file,line,col} coordinates
 */

export * from './types.js';
export * from './normalize.js';
export { resolveLexicalMatches } from './lexical.js';
export { resolveSymbolMatches } from './symbols.js';
export { resolveSemanticMatches, prepareSemanticCandidates } from './semantic.js';

// Re-export for convenience
export type { SearchHit, MatchReason, SymbolKind } from './types.js';