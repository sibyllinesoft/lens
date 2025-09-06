/**
 * Clone-Aware Recall System
 * 
 * Implements token-shingle MinHash/SimHash indexing to build clone sets for 
 * expanding search results across code clones, forks, and backports.
 * 
 * Key Features:
 * - Token-shingle indexing with w=5-7 subtokens
 * - Clone set expansion with strict budget |C(s)| ≤ k_clone
 * - Same-repo, same-symbol-kind veto for expansion
 * - Jaccard token similarity bonus β·Jaccard_tokens(·) bounded in log-odds
 * - Performance gate: +0.5-1.0pp Recall@50 at ≤+0.6ms p95
 */

import { createHash } from 'crypto';
import type { SearchHit, MatchReason } from '../core/span_resolver/types.js';
import type { SearchContext } from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';

// Configuration constants per TODO.md requirements
const CLONE_BUDGET_MAX = 3; // k_clone ≤ 3
const JACCARD_BONUS_MAX = 0.2; // β ≤ 0.2 log-odds
const SHINGLE_WIDTH_MIN = 5; // w = 5-7 subtokens
const SHINGLE_WIDTH_MAX = 7;
const MIN_TOPIC_SIMILARITY = 0.75; // τ threshold for topic similarity
const PERFORMANCE_BUDGET_MS = 0.6; // ≤ +0.6ms p95 latency

export interface TokenShingle {
  tokens: string[];
  hash: number;
  file: string;
  line: number;
  col: number;
}

export interface CloneCandidate {
  file: string;
  line: number;
  col: number;
  jaccard_score: number;
  topic_similarity: number;
  symbol_kind?: string;
  repository: string;
}

export interface CloneSet {
  representative_hash: number;
  members: CloneCandidate[];
  token_jaccard: number;
  last_updated: Date;
}

export class CloneAwareRecallSystem {
  private shingleIndex = new Map<number, CloneSet>();
  private fileToShingles = new Map<string, Set<number>>();
  private repoSymbolKindIndex = new Map<string, Set<string>>(); // repo:symbol_kind mapping
  private performanceMetrics = {
    expansionLatency: [] as number[],
    cacheHitRate: 0,
    cloneSetSize: [] as number[],
  };
  
  /**
   * Initialize the clone-aware recall system
   */
  async initialize(): Promise<void> {
    const span = LensTracer.createChildSpan('clone_recall_init');
    
    try {
      console.log('🔗 Initializing Clone-Aware Recall system...');
      
      // Initialize empty indexes
      this.shingleIndex.clear();
      this.fileToShingles.clear();
      this.repoSymbolKindIndex.clear();
      
      span.setAttributes({ success: true });
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw error;
    } finally {
      span.end();
    }
  }
  
  /**
   * Index a code span for clone detection using token shingles
   */
  async indexSpan(
    content: string,
    file: string,
    line: number,
    col: number,
    repository: string,
    symbolKind?: string
  ): Promise<void> {
    const span = LensTracer.createChildSpan('index_span_for_clones', {
      file, line, col, repository, symbol_kind: symbolKind
    });
    
    try {
      // Tokenize content into subtokens
      const tokens = this.tokenizeToSubtokens(content);
      
      if (tokens.length < SHINGLE_WIDTH_MIN) {
        // Skip indexing for very short spans
        span.setAttributes({ skipped: true, reason: 'too_short' });
        return;
      }
      
      // Generate token shingles with variable width
      const shingles = this.generateTokenShingles(tokens, file, line, col);
      
      // Update repository symbol kind index
      if (symbolKind) {
        const repoKey = `${repository}:${symbolKind}`;
        if (!this.repoSymbolKindIndex.has(repoKey)) {
          this.repoSymbolKindIndex.set(repoKey, new Set());
        }
        this.repoSymbolKindIndex.get(repoKey)!.add(`${file}:${line}:${col}`);
      }
      
      // Index each shingle
      const fileShingles = new Set<number>();
      for (const shingle of shingles) {
        const hash = this.computeMinHash(shingle.tokens);
        fileShingles.add(hash);
        
        if (!this.shingleIndex.has(hash)) {
          this.shingleIndex.set(hash, {
            representative_hash: hash,
            members: [],
            token_jaccard: 1.0,
            last_updated: new Date(),
          });
        }
        
        const cloneSet = this.shingleIndex.get(hash)!;
        cloneSet.members.push({
          file,
          line,
          col,
          jaccard_score: 1.0, // Will be computed during expansion
          topic_similarity: 1.0, // Will be computed during expansion
          symbol_kind: symbolKind,
          repository,
        });
        
        cloneSet.last_updated = new Date();
      }
      
      // Update file-to-shingles mapping
      this.fileToShingles.set(file, fileShingles);
      
      span.setAttributes({ 
        success: true,
        shingles_generated: shingles.length,
        tokens_count: tokens.length
      });
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw error;
    } finally {
      span.end();
    }
  }
  
  /**
   * Expand search results using clone-aware recall
   */
  async expandWithClones(
    originalHits: SearchHit[],
    ctx: SearchContext
  ): Promise<SearchHit[]> {
    const span = LensTracer.createChildSpan('expand_with_clones', {
      original_hits: originalHits.length,
      repo_sha: ctx.repo_sha,
      query: ctx.query
    });
    
    const expandStart = Date.now();
    
    try {
      const expandedHits: SearchHit[] = [...originalHits];
      const seenHits = new Set<string>();
      
      // Track original hits to avoid duplicates
      for (const hit of originalHits) {
        seenHits.add(`${hit.file}:${hit.line}:${hit.col}`);
      }
      
      // Process each original hit for clone expansion
      for (const hit of originalHits) {
        if (expandedHits.length >= ctx.k * 2) {
          // Limit total expansion to prevent explosion
          break;
        }
        
        const cloneMatches = await this.findCloneMatches(
          hit, 
          ctx.repo_sha,
          hit.symbol_kind
        );
        
        // Apply clone budget constraint |C(s)| ≤ k_clone
        const budgetedClones = cloneMatches
          .slice(0, CLONE_BUDGET_MAX)
          .filter(clone => {
            const key = `${clone.file}:${clone.line}:${clone.col}`;
            return !seenHits.has(key);
          });
        
        // Convert clone candidates to SearchHits with Jaccard bonus
        for (const clone of budgetedClones) {
          if (this.shouldExpandClone(clone, hit, ctx)) {
            const expandedHit = this.createExpandedHit(hit, clone);
            expandedHits.push(expandedHit);
            seenHits.add(`${clone.file}:${clone.line}:${clone.col}`);
          }
        }
      }
      
      // Record performance metrics
      const expansionLatency = Date.now() - expandStart;
      this.performanceMetrics.expansionLatency.push(expansionLatency);
      
      // Check performance gate: ≤ +0.6ms p95
      if (expansionLatency > PERFORMANCE_BUDGET_MS) {
        console.warn(`Clone expansion SLA breach: ${expansionLatency}ms > ${PERFORMANCE_BUDGET_MS}ms`);
      }
      
      span.setAttributes({
        success: true,
        original_hits: originalHits.length,
        expanded_hits: expandedHits.length,
        expansion_latency_ms: expansionLatency,
        clones_added: expandedHits.length - originalHits.length
      });
      
      return expandedHits;
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      
      // Return original hits on error to maintain system stability
      return originalHits;
      
    } finally {
      span.end();
    }
  }
  
  /**
   * Find clone matches for a given search hit
   */
  private async findCloneMatches(
    hit: SearchHit,
    currentRepo: string,
    symbolKind?: string
  ): Promise<CloneCandidate[]> {
    const fileShingles = this.fileToShingles.get(hit.file);
    if (!fileShingles) {
      return [];
    }
    
    const cloneMatches: CloneCandidate[] = [];
    
    // Find all clone sets that contain shingles from this file
    for (const shingleHash of fileShingles) {
      const cloneSet = this.shingleIndex.get(shingleHash);
      if (!cloneSet) continue;
      
      for (const member of cloneSet.members) {
        // Apply same-repo, same-symbol-kind veto
        if (this.shouldVetoClone(member, hit, currentRepo, symbolKind)) {
          continue;
        }
        
        // Calculate Jaccard similarity for token overlap
        const jaccardScore = await this.computeTokenJaccard(hit, member);
        
        // Calculate topic similarity (simplified implementation)
        const topicSimilarity = await this.computeTopicSimilarity(hit, member);
        
        // Apply topic similarity threshold τ
        if (topicSimilarity < MIN_TOPIC_SIMILARITY) {
          continue;
        }
        
        cloneMatches.push({
          ...member,
          jaccard_score: jaccardScore,
          topic_similarity: topicSimilarity,
        });
      }
    }
    
    // Sort by combined score (Jaccard + topic similarity)
    cloneMatches.sort((a, b) => 
      (b.jaccard_score + b.topic_similarity) - (a.jaccard_score + a.topic_similarity)
    );
    
    return cloneMatches;
  }
  
  /**
   * Apply same-repo, same-symbol-kind veto
   */
  private shouldVetoClone(
    candidate: CloneCandidate,
    original: SearchHit,
    currentRepo: string,
    symbolKind?: string
  ): boolean {
    // Veto if same repository and same symbol kind
    if (candidate.repository === currentRepo && 
        candidate.symbol_kind === symbolKind &&
        symbolKind !== undefined) {
      return true;
    }
    
    // Veto if exact same location
    if (candidate.file === original.file && 
        candidate.line === original.line &&
        candidate.col === original.col) {
      return true;
    }
    
    return false;
  }
  
  /**
   * Check if clone should be expanded based on quality gates
   */
  private shouldExpandClone(
    clone: CloneCandidate,
    original: SearchHit,
    ctx: SearchContext
  ): boolean {
    // Check topic similarity threshold
    if (clone.topic_similarity < MIN_TOPIC_SIMILARITY) {
      return false;
    }
    
    // Check minimum Jaccard score
    if (clone.jaccard_score < 0.3) {
      return false;
    }
    
    // Check for vendor/third-party path filtering
    if (this.isVendorPath(clone.file)) {
      return false;
    }
    
    return true;
  }
  
  /**
   * Create expanded SearchHit from clone candidate
   */
  private createExpandedHit(original: SearchHit, clone: CloneCandidate): SearchHit {
    // Apply bounded Jaccard bonus: β·Jaccard_tokens(·) bounded in log-odds
    const jaccardBonus = Math.min(JACCARD_BONUS_MAX, clone.jaccard_score * 0.1);
    const adjustedScore = original.score + jaccardBonus;
    
    return {
      file: clone.file,
      line: clone.line,
      col: clone.col,
      lang: original.lang, // Assume same language for clones
      snippet: original.snippet, // Use original snippet as placeholder
      score: adjustedScore,
      why: [...(original.why || []), 'clone_expansion'] as MatchReason[],
      byte_offset: undefined, // Would need to be resolved separately
      span_len: original.span_len,
      symbol_kind: clone.symbol_kind as any,
      context_before: original.context_before,
      context_after: original.context_after,
    };
  }
  
  /**
   * Tokenize content into subtokens for shingle generation
   */
  private tokenizeToSubtokens(content: string): string[] {
    // Simple tokenization - split on non-alphanumeric characters
    // In production, would use language-specific tokenizers
    return content
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter(token => token.length > 0)
      .flatMap(token => {
        // Split camelCase and snake_case
        return token.split(/(?=[A-Z])|_/).filter(subtoken => subtoken.length > 0);
      });
  }
  
  /**
   * Generate token shingles with variable width w=5-7
   */
  private generateTokenShingles(
    tokens: string[], 
    file: string, 
    line: number, 
    col: number
  ): TokenShingle[] {
    const shingles: TokenShingle[] = [];
    
    for (let width = SHINGLE_WIDTH_MIN; width <= SHINGLE_WIDTH_MAX; width++) {
      for (let i = 0; i <= tokens.length - width; i++) {
        const shingleTokens = tokens.slice(i, i + width);
        const hash = this.computeMinHash(shingleTokens);
        
        shingles.push({
          tokens: shingleTokens,
          hash,
          file,
          line,
          col,
        });
      }
    }
    
    return shingles;
  }
  
  /**
   * Compute MinHash for token shingle
   */
  private computeMinHash(tokens: string[]): number {
    const combined = tokens.join('|');
    const hash = createHash('md5').update(combined).digest('hex');
    
    // Convert hex to number (simplified MinHash)
    return parseInt(hash.substring(0, 8), 16);
  }
  
  /**
   * Compute Jaccard similarity between two spans using tokens
   */
  private async computeTokenJaccard(hit1: SearchHit, candidate: CloneCandidate): Promise<number> {
    // Simplified implementation - in production would extract actual content
    // and compute proper Jaccard similarity
    
    // For now, use a heuristic based on file path similarity
    const path1 = hit1.file.split('/').pop() || '';
    const path2 = candidate.file.split('/').pop() || '';
    
    if (path1 === path2) {
      return 0.9; // High similarity for same filename
    }
    
    // Basic string similarity
    const longer = path1.length > path2.length ? path1 : path2;
    const shorter = path1.length <= path2.length ? path1 : path2;
    
    if (longer.length === 0) return 0;
    
    const editDistance = this.levenshteinDistance(longer, shorter);
    return (longer.length - editDistance) / longer.length;
  }
  
  /**
   * Compute topic similarity between spans
   */
  private async computeTopicSimilarity(hit: SearchHit, candidate: CloneCandidate): Promise<number> {
    // Simplified topic similarity based on path structure and symbol kind
    const hit1Path = hit.file.split('/');
    const hit2Path = candidate.file.split('/');
    
    // Find common path prefix length
    let commonPrefix = 0;
    for (let i = 0; i < Math.min(hit1Path.length, hit2Path.length); i++) {
      if (hit1Path[i] === hit2Path[i]) {
        commonPrefix++;
      } else {
        break;
      }
    }
    
    // Path similarity component
    const pathSimilarity = commonPrefix / Math.max(hit1Path.length, hit2Path.length);
    
    // Symbol kind similarity component
    let symbolSimilarity = 0.5; // Default neutral
    if (hit.symbol_kind && candidate.symbol_kind) {
      symbolSimilarity = hit.symbol_kind === candidate.symbol_kind ? 1.0 : 0.3;
    }
    
    // Combined topic similarity
    return (pathSimilarity + symbolSimilarity) / 2;
  }
  
  /**
   * Check if path is vendor/third-party code
   */
  private isVendorPath(filePath: string): boolean {
    const vendorPaths = [
      'node_modules/',
      'vendor/',
      'third_party/',
      '.git/',
      'build/',
      'dist/',
      '__pycache__/',
    ];
    
    return vendorPaths.some(vendor => filePath.includes(vendor));
  }
  
  /**
   * Compute Levenshtein distance for string similarity
   */
  private levenshteinDistance(str1: string, str2: string): number {
    const matrix = Array(str2.length + 1).fill(null).map(() => Array(str1.length + 1).fill(null));
    
    for (let i = 0; i <= str1.length; i += 1) {
      matrix[0][i] = i;
    }
    
    for (let j = 0; j <= str2.length; j += 1) {
      matrix[j][0] = j;
    }
    
    for (let j = 1; j <= str2.length; j += 1) {
      for (let i = 1; i <= str1.length; i += 1) {
        const indicator = str1[i - 1] === str2[j - 1] ? 0 : 1;
        matrix[j][i] = Math.min(
          matrix[j][i - 1] + 1, // deletion
          matrix[j - 1][i] + 1, // insertion
          matrix[j - 1][i - 1] + indicator, // substitution
        );
      }
    }
    
    return matrix[str2.length][str1.length];
  }
  
  /**
   * Get performance metrics
   */
  getPerformanceMetrics() {
    const expansionLatencies = this.performanceMetrics.expansionLatency;
    const p95Latency = expansionLatencies.length > 0 
      ? expansionLatencies.sort((a, b) => a - b)[Math.floor(expansionLatencies.length * 0.95)]
      : 0;
    
    return {
      clone_sets_count: this.shingleIndex.size,
      indexed_files: this.fileToShingles.size,
      expansion_p95_latency_ms: p95Latency,
      cache_hit_rate: this.performanceMetrics.cacheHitRate,
      performance_gate_breaches: expansionLatencies.filter(l => l > PERFORMANCE_BUDGET_MS).length,
    };
  }
  
  /**
   * Cleanup and shutdown
   */
  async shutdown(): Promise<void> {
    const span = LensTracer.createChildSpan('clone_recall_shutdown');
    
    try {
      console.log('🔗 Shutting down Clone-Aware Recall system...');
      
      // Clear all indexes
      this.shingleIndex.clear();
      this.fileToShingles.clear();
      this.repoSymbolKindIndex.clear();
      
      span.setAttributes({ success: true });
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw error;
    } finally {
      span.end();
    }
  }
}