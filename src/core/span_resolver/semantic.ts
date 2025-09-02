/**
 * Stage C: Semantic Reranking Span Resolution  
 * NEVER invents spans - only passes through upstream coordinates with updated scores
 */

import { SearchHit, SemanticCandidate } from './types.js';

/**
 * Resolve semantic reranking results by passing through upstream spans
 * This stage MUST NOT create new spans - only updates scores and match reasons
 */
export async function resolveSemanticMatches(
  candidates: SemanticCandidate[]
): Promise<SearchHit[]> {
  const results: SearchHit[] = [];
  
  for (const candidate of candidates) {
    try {
      const hit = resolveSemanticCandidate(candidate);
      if (hit) {
        results.push(hit);
      }
    } catch (error) {
      console.warn(`Failed to resolve semantic candidate ${candidate.file_path}:`, error);
      
      // Fallback: create hit with upstream coordinates
      results.push({
        file: candidate.file_path,
        line: candidate.upstream_line,
        col: candidate.upstream_col,
        snippet: candidate.upstream_snippet,
        score: candidate.score,
        why: candidate.match_reasons as any,
        symbol_kind: candidate.symbol_kind as any,
        ast_path: candidate.ast_path,
        context_before: candidate.upstream_context_before,
        context_after: candidate.upstream_context_after,
      });
    }
  }
  
  return results;
}

/**
 * Resolve a single semantic candidate by passing through upstream span
 */
function resolveSemanticCandidate(candidate: SemanticCandidate): SearchHit {
  // Semantic reranking NEVER creates new spans
  // It only updates scores and adds 'semantic' to match reasons
  
  // Ensure 'semantic' is in the match reasons
  const matchReasons = [...candidate.match_reasons];
  if (!matchReasons.includes('semantic')) {
    matchReasons.push('semantic');
  }
  
  return {
    file: candidate.file_path,
    line: candidate.upstream_line,
    col: candidate.upstream_col,
    snippet: candidate.upstream_snippet,
    score: candidate.score, // This is the re-ranked score
    why: matchReasons as any,
    symbol_kind: candidate.symbol_kind as any,
    ast_path: candidate.ast_path,
    context_before: candidate.upstream_context_before,
    context_after: candidate.upstream_context_after,
  };
}

/**
 * Prepare semantic candidates from upstream SearchHits
 * Converts SearchHits to SemanticCandidates for reranking
 */
export function prepareSemanticCandidates(
  upstreamHits: SearchHit[],
  newScores: number[]
): SemanticCandidate[] {
  if (upstreamHits.length !== newScores.length) {
    console.warn(
      `⚠️  Hit count mismatch in semantic reranking: ${upstreamHits.length} hits vs ${newScores.length} scores. Adjusting...`
    );
    
    // Gracefully handle mismatch by padding or truncating scores
    const adjustedScores = [...newScores];
    while (adjustedScores.length < upstreamHits.length) {
      adjustedScores.push(upstreamHits[adjustedScores.length]?.score || 0.1);
    }
    adjustedScores.length = upstreamHits.length; // Truncate if too many scores
    
    return prepareSemanticCandidates(upstreamHits, adjustedScores);
  }
  
  return upstreamHits.map((hit, index) => ({
    file_path: hit.file,
    score: newScores[index] ?? hit.score, // New semantic score, fallback to original
    match_reasons: hit.why,
    symbol_kind: hit.symbol_kind,
    ast_path: hit.ast_path,
    upstream_line: hit.line,
    upstream_col: hit.col,
    upstream_snippet: hit.snippet,
    upstream_context_before: hit.context_before,
    upstream_context_after: hit.context_after,
  }));
}