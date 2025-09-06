/**
 * Stage-B LSP Integration
 * Treats LSP hints as another symbol source: why+=["lsp_hint"]
 * Raises Stage-B coverage by mapping query subtokens to hinted symbol IDs
 */

import type { 
  LSPHint, 
  Candidate, 
  SearchContext,
  SymbolKind 
} from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';

interface LSPStageResult {
  candidates: Candidate[];
  hint_matches: number;
  subtoken_coverage: number;
  alias_resolutions: number;
  processing_time_ms: number;
}

export class LSPStageBEnhancer {
  private hintsIndex = new Map<string, LSPHint[]>(); // symbol_name -> hints
  private aliasIndex = new Map<string, LSPHint[]>(); // alias -> hints
  private typeIndex = new Map<string, LSPHint[]>(); // type -> hints
  private pathIndex = new Map<string, LSPHint[]>(); // file_path -> hints

  constructor() {}

  /**
   * Load LSP hints into indices for fast lookups
   */
  loadHints(hints: LSPHint[]): void {
    const span = LensTracer.createChildSpan('lsp_stage_b_load_hints', {
      'hints.count': hints.length,
    });

    try {
      // Clear existing indices
      this.hintsIndex.clear();
      this.aliasIndex.clear();
      this.typeIndex.clear();
      this.pathIndex.clear();

      for (const hint of hints) {
        // Index by symbol name
        const nameKey = hint.name.toLowerCase();
        if (!this.hintsIndex.has(nameKey)) {
          this.hintsIndex.set(nameKey, []);
        }
        this.hintsIndex.get(nameKey)!.push(hint);

        // Index by aliases
        for (const alias of hint.aliases) {
          const aliasKey = alias.toLowerCase();
          if (!this.aliasIndex.has(aliasKey)) {
            this.aliasIndex.set(aliasKey, []);
          }
          this.aliasIndex.get(aliasKey)!.push(hint);
        }

        // Index by type information
        if (hint.type_info) {
          const typeKey = hint.type_info.toLowerCase();
          if (!this.typeIndex.has(typeKey)) {
            this.typeIndex.set(typeKey, []);
          }
          this.typeIndex.get(typeKey)!.push(hint);
        }

        // Index by file path
        const pathKey = hint.file_path.toLowerCase();
        if (!this.pathIndex.has(pathKey)) {
          this.pathIndex.set(pathKey, []);
        }
        this.pathIndex.get(pathKey)!.push(hint);
      }

      span.setAttributes({
        success: true,
        name_index_size: this.hintsIndex.size,
        alias_index_size: this.aliasIndex.size,
        type_index_size: this.typeIndex.size,
        path_index_size: this.pathIndex.size,
      });

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Enhance Stage-B results with LSP hints
   */
  enhanceStageB(
    query: string,
    context: SearchContext,
    baseCandidates: Candidate[],
    maxResults: number = 50
  ): LSPStageResult {
    const span = LensTracer.createChildSpan('lsp_stage_b_enhance', {
      'search.query': query,
      'base_candidates': baseCandidates.length,
      'max_results': maxResults,
    });

    const startTime = Date.now();
    
    try {
      // Tokenize query for subtoken matching
      const queryTokens = this.tokenizeQuery(query);
      const lspCandidates: Candidate[] = [];
      
      let hintMatches = 0;
      let subtokenCoverage = 0;
      let aliasResolutions = 0;

      // Search by exact symbol name matches
      for (const token of queryTokens) {
        const tokenLower = token.toLowerCase();
        
        // Direct symbol name matches
        const directHints = this.hintsIndex.get(tokenLower) || [];
        for (const hint of directHints) {
          lspCandidates.push(this.convertHintToCandidate(hint, 1.0, 'exact'));
          hintMatches++;
        }

        // Alias matches
        const aliasHints = this.aliasIndex.get(tokenLower) || [];
        for (const hint of aliasHints) {
          lspCandidates.push(this.convertHintToCandidate(hint, 0.9, 'alias'));
          aliasResolutions++;
        }

        // Partial matches for longer symbols
        for (const [symbolName, hints] of this.hintsIndex) {
          if (symbolName.includes(tokenLower) && symbolName !== tokenLower) {
            const score = this.calculatePartialMatchScore(tokenLower, symbolName);
            if (score > 0.5) {
              for (const hint of hints) {
                lspCandidates.push(this.convertHintToCandidate(hint, score, 'partial'));
                hintMatches++;
              }
            }
          }
        }
      }

      // Calculate subtoken coverage
      const coveredTokens = new Set<string>();
      for (const candidate of lspCandidates) {
        for (const token of queryTokens) {
          if (candidate.context?.toLowerCase().includes(token.toLowerCase())) {
            coveredTokens.add(token);
          }
        }
      }
      subtokenCoverage = queryTokens.length > 0 ? coveredTokens.size / queryTokens.length : 0;

      // Enhance with structural context
      this.enhanceWithStructuralContext(lspCandidates, query);

      // Merge with base candidates and deduplicate
      const allCandidates = this.mergeCandidates(baseCandidates, lspCandidates);
      
      // Sort by relevance and limit results
      allCandidates.sort((a, b) => b.score - a.score);
      const finalCandidates = allCandidates.slice(0, maxResults);

      const processingTime = Date.now() - startTime;

      const result: LSPStageResult = {
        candidates: finalCandidates,
        hint_matches: hintMatches,
        subtoken_coverage: subtokenCoverage,
        alias_resolutions: aliasResolutions,
        processing_time_ms: processingTime,
      };

      span.setAttributes({
        success: true,
        hint_matches: hintMatches,
        subtoken_coverage: subtokenCoverage,
        alias_resolutions: aliasResolutions,
        final_candidates: finalCandidates.length,
        processing_time_ms: processingTime,
      });

      return result;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Tokenize query into meaningful parts
   */
  private tokenizeQuery(query: string): string[] {
    const tokens: string[] = [];
    
    // Split on common delimiters
    const basicTokens = query.split(/[\s._\-:]+/).filter(t => t.length > 0);
    
    for (const token of basicTokens) {
      tokens.push(token);
      
      // Handle camelCase and PascalCase
      const camelCaseTokens = token.split(/(?=[A-Z])/).filter(t => t.length > 0);
      if (camelCaseTokens.length > 1) {
        tokens.push(...camelCaseTokens.map(t => t.toLowerCase()));
      }
      
      // Handle snake_case (already split above)
      
      // Handle kebab-case (already split above)
    }

    // Remove duplicates and short tokens
    return Array.from(new Set(tokens)).filter(t => t.length >= 2);
  }

  /**
   * Calculate partial match score for symbol names
   */
  private calculatePartialMatchScore(token: string, symbolName: string): number {
    const tokenLower = token.toLowerCase();
    const symbolLower = symbolName.toLowerCase();
    
    // Exact match
    if (tokenLower === symbolLower) return 1.0;
    
    // Prefix match
    if (symbolLower.startsWith(tokenLower)) {
      return 0.9 * (tokenLower.length / symbolLower.length);
    }
    
    // Contains match
    if (symbolLower.includes(tokenLower)) {
      return 0.7 * (tokenLower.length / symbolLower.length);
    }
    
    // Fuzzy match using edit distance
    const editDistance = this.calculateEditDistance(tokenLower, symbolLower);
    const maxLength = Math.max(tokenLower.length, symbolLower.length);
    const similarity = (maxLength - editDistance) / maxLength;
    
    return similarity > 0.6 ? similarity * 0.6 : 0;
  }

  /**
   * Calculate edit distance between two strings
   */
  private calculateEditDistance(str1: string, str2: string): number {
    const matrix: number[][] = [];
    
    for (let i = 0; i <= str2.length; i++) {
      matrix[i] = [i];
    }
    
    for (let j = 0; j <= str1.length; j++) {
      matrix[0][j] = j;
    }
    
    for (let i = 1; i <= str2.length; i++) {
      for (let j = 1; j <= str1.length; j++) {
        if (str2[i - 1] === str1[j - 1]) {
          matrix[i][j] = matrix[i - 1][j - 1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i - 1][j - 1] + 1,
            matrix[i][j - 1] + 1,
            matrix[i - 1][j] + 1
          );
        }
      }
    }
    
    return matrix[str2.length][str1.length];
  }

  /**
   * Convert LSP hint to Lens candidate
   */
  private convertHintToCandidate(
    hint: LSPHint,
    baseScore: number,
    matchType: string
  ): Candidate {
    // Apply LSP-specific scoring bonuses
    let score = baseScore;
    
    // Reference count bonus (more references = more important)
    const refBonus = Math.min(hint.references_count * 0.01, 0.2);
    score += refBonus;
    
    // Symbol kind bonus
    const kindBonus = this.getSymbolKindBonus(hint.kind);
    score += kindBonus;
    
    // Type information bonus
    if (hint.type_info) {
      score += 0.1;
    }
    
    // Signature bonus
    if (hint.signature) {
      score += 0.05;
    }

    return {
      doc_id: hint.symbol_id,
      file_path: hint.file_path,
      line: hint.line,
      col: hint.col,
      score,
      match_reasons: ['lsp_hint'],
      symbol_kind: hint.kind,
      context: hint.signature || `${hint.kind} ${hint.name}`,
      snippet: hint.signature,
      // Additional LSP-specific metadata
      ast_path: hint.definition_uri,
    };
  }

  /**
   * Get scoring bonus for symbol kinds
   */
  private getSymbolKindBonus(kind: SymbolKind): number {
    const bonuses = {
      function: 0.3,
      class: 0.25,
      interface: 0.2,
      type: 0.2,
      method: 0.15,
      variable: 0.1,
      property: 0.1,
      constant: 0.05,
      enum: 0.05,
    };
    
    return bonuses[kind] || 0;
  }

  /**
   * Enhance candidates with structural context from LSP
   */
  private enhanceWithStructuralContext(candidates: Candidate[], query: string): void {
    for (const candidate of candidates) {
      // Find related symbols in the same file
      const fileHints = this.pathIndex.get(candidate.file_path.toLowerCase()) || [];
      
      // Count related symbols for context scoring
      const relatedSymbols = fileHints.filter(hint => 
        Math.abs(hint.line - candidate.line) <= 10
      );
      
      if (relatedSymbols.length > 1) {
        candidate.score += 0.05; // Boost for symbols with nearby context
      }

      // Check for query matches in related symbols
      const contextMatches = relatedSymbols.filter(hint =>
        hint.name.toLowerCase().includes(query.toLowerCase()) ||
        (hint.signature && hint.signature.toLowerCase().includes(query.toLowerCase()))
      );
      
      if (contextMatches.length > 0) {
        candidate.score += 0.1; // Boost for contextual relevance
      }
    }
  }

  /**
   * Merge base candidates with LSP candidates, avoiding duplicates
   */
  private mergeCandidates(baseCandidates: Candidate[], lspCandidates: Candidate[]): Candidate[] {
    const merged = [...baseCandidates];
    const existingKeys = new Set(
      baseCandidates.map(c => `${c.file_path}:${c.line}:${c.col}`)
    );

    for (const lspCandidate of lspCandidates) {
      const key = `${lspCandidate.file_path}:${lspCandidate.line}:${lspCandidate.col}`;
      
      if (!existingKeys.has(key)) {
        merged.push(lspCandidate);
        existingKeys.add(key);
      } else {
        // Enhance existing candidate with LSP information
        const existing = merged.find(c => 
          c.file_path === lspCandidate.file_path &&
          c.line === lspCandidate.line &&
          c.col === lspCandidate.col
        );
        
        if (existing) {
          // Boost score for LSP confirmation
          existing.score += 0.2;
          
          // Add LSP hint to match reasons
          if (!existing.match_reasons.includes('lsp_hint')) {
            existing.match_reasons.push('lsp_hint');
          }
          
          // Enhance context with LSP information
          if (lspCandidate.context && (!existing.context || existing.context.length < lspCandidate.context.length)) {
            existing.context = lspCandidate.context;
          }
          
          if (lspCandidate.snippet && !existing.snippet) {
            existing.snippet = lspCandidate.snippet;
          }
        }
      }
    }

    return merged;
  }

  /**
   * Find symbols near a specific location using LSP hints
   */
  async findLSPSymbolsNear(
    filePath: string,
    line: number,
    radius: number = 10
  ): Promise<Candidate[]> {
    const span = LensTracer.createChildSpan('lsp_symbols_near', {
      'file.path': filePath,
      'location.line': line,
      'search.radius': radius,
    });

    try {
      const candidates: Candidate[] = [];
      const fileHints = this.pathIndex.get(filePath.toLowerCase()) || [];
      
      const nearbyHints = fileHints.filter(hint =>
        Math.abs(hint.line - line) <= radius
      );

      for (const hint of nearbyHints) {
        const distance = Math.abs(hint.line - line);
        const proximityScore = 1.0 - (distance / radius);
        
        candidates.push(this.convertHintToCandidate(hint, proximityScore, 'proximity'));
      }

      candidates.sort((a, b) => b.score - a.score);

      span.setAttributes({
        success: true,
        nearby_hints: nearbyHints.length,
        candidates_returned: candidates.length,
      });

      return candidates;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Get hints statistics
   */
  getStats(): {
    total_hints: number;
    name_index_size: number;
    alias_index_size: number;
    type_index_size: number;
    path_index_size: number;
  } {
    const totalHints = Array.from(this.hintsIndex.values())
      .reduce((sum, hints) => sum + hints.length, 0);

    return {
      total_hints: totalHints,
      name_index_size: this.hintsIndex.size,
      alias_index_size: this.aliasIndex.size,
      type_index_size: this.typeIndex.size,
      path_index_size: this.pathIndex.size,
    };
  }

  /**
   * Clear all indices
   */
  clear(): void {
    this.hintsIndex.clear();
    this.aliasIndex.clear();
    this.typeIndex.clear();
    this.pathIndex.clear();
  }
}