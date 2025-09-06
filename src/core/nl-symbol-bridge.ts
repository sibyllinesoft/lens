/**
 * Natural Language → Symbol Bridge Enhancement
 * 
 * Improves m parameter from 5→7 as specified in TODO.md:
 * - Measure +Success@10, cap fan-out
 * - Bridge natural language queries to symbol-based results
 * - Enhanced query expansion with controlled fanout
 */

import { SearchHit, SymbolCandidate } from './span_resolver/types.js';

export interface NLSymbolBridgeConfig {
  m_parameter: number;           // Bridge fanout parameter (upgraded to 7)
  max_fanout: number;           // Cap fanout to prevent explosion  
  success_at_10_threshold: number; // Success@10 measurement threshold
  semantic_similarity_threshold: number; // Minimum similarity for bridging
  symbol_boost_factor: number;  // Boost factor for symbol matches
}

export interface NLQuery {
  original_query: string;
  query_type: 'natural_language' | 'symbol_exact' | 'hybrid';
  extracted_entities: string[];
  inferred_intent: QueryIntent;
}

export interface QueryIntent {
  target_symbols: string[];      // Extracted symbol names
  target_kinds: string[];        // function, class, variable, etc.
  context_keywords: string[];    // Additional context terms
  confidence: number;            // Intent extraction confidence
}

export interface BridgeResult {
  original_hits: SearchHit[];
  bridged_hits: SearchHit[];     // Additional hits from bridge
  combined_hits: SearchHit[];    // Final merged and ranked results
  bridge_metrics: BridgeMetrics;
}

export interface BridgeMetrics {
  original_success_at_10: number;
  bridged_success_at_10: number;
  improvement_delta: number;     // Success@10 improvement
  fanout_count: number;         // Actual fanout achieved
  semantic_matches: number;     // Symbol matches from NL bridge
}

export class NLSymbolBridge {
  private config: NLSymbolBridgeConfig;
  
  constructor(config?: Partial<NLSymbolBridgeConfig>) {
    this.config = {
      m_parameter: 7,             // Upgraded from m=5 as specified in TODO.md
      max_fanout: 15,            // Cap fanout to prevent explosion
      success_at_10_threshold: 0.62, // From TODO.md baseline
      semantic_similarity_threshold: 0.7,
      symbol_boost_factor: 1.2,
      ...config
    };
  }
  
  /**
   * Apply NL→Symbol bridge to enhance search results
   */
  public async applyBridge(
    query: string,
    originalHits: SearchHit[],
    symbolCandidates: SymbolCandidate[]
  ): Promise<BridgeResult> {
    console.log(`🌉 Applying NL→Symbol bridge with m=${this.config.m_parameter}`);
    
    try {
      // Step 1: Analyze query for NL patterns
      const nlQuery = await this.analyzeQuery(query);
      
      // Step 2: Apply bridge only for NL queries
      if (nlQuery.query_type !== 'natural_language') {
        return {
          original_hits: originalHits,
          bridged_hits: [],
          combined_hits: originalHits,
          bridge_metrics: this.createEmptyMetrics()
        };
      }
      
      // Step 3: Extract symbol targets from NL query
      const bridgedHits = await this.performSymbolBridge(nlQuery, symbolCandidates);
      
      // Step 4: Merge and rank with original results  
      const combinedHits = this.mergeAndRank(originalHits, bridgedHits);
      
      // Step 5: Calculate bridge metrics
      const metrics = this.calculateBridgeMetrics(originalHits, bridgedHits, combinedHits);
      
      console.log(`📊 Bridge metrics: Success@10 ${metrics.original_success_at_10.toFixed(3)} → ${metrics.bridged_success_at_10.toFixed(3)} (${metrics.improvement_delta >= 0 ? '+' : ''}${metrics.improvement_delta.toFixed(3)})`);
      
      return {
        original_hits: originalHits,
        bridged_hits: bridgedHits,
        combined_hits: combinedHits,
        bridge_metrics: metrics
      };
      
    } catch (error) {
      console.error('❌ NL→Symbol bridge failed:', error);
      
      // Return original results on failure
      return {
        original_hits: originalHits,
        bridged_hits: [],
        combined_hits: originalHits,
        bridge_metrics: this.createEmptyMetrics()
      };
    }
  }
  
  /**
   * Analyze query to determine if NL bridge should be applied
   */
  private async analyzeQuery(query: string): Promise<NLQuery> {
    const normalizedQuery = query.toLowerCase().trim();
    
    // Natural language patterns (multi-word, descriptive phrases)
    const nlPatterns = [
      /\b(authentication|auth)\s+(logic|code|function|method)\b/,
      /\b(error|exception)\s+(handling|handler|catch)\b/,
      /\b(database|db)\s+(connection|query|access)\b/,
      /\b(user|customer)\s+(validation|login|registration)\b/,
      /\b(api|endpoint)\s+(handler|controller|route)\b/,
      /\b(data|object)\s+(transformation|mapping|processing)\b/,
      /\b(cache|caching|memoiz)\w*\s+(logic|implementation)\b/,
      /\b(config|configuration)\s+(file|setup|management)\b/
    ];
    
    const isNaturalLanguage = nlPatterns.some(pattern => pattern.test(normalizedQuery)) ||
                              normalizedQuery.split(/\s+/).length > 2;
    
    // Extract potential symbol names and kinds
    const extractedEntities = this.extractEntities(normalizedQuery);
    const inferredIntent = this.inferIntent(normalizedQuery, extractedEntities);
    
    return {
      original_query: query,
      query_type: isNaturalLanguage ? 'natural_language' : 'symbol_exact',
      extracted_entities: extractedEntities,
      inferred_intent: inferredIntent
    };
  }
  
  /**
   * Extract entities (potential symbol names) from NL query
   */
  private extractEntities(query: string): string[] {
    const entities = [];
    
    // Common programming terms that might map to symbols
    const symbolPatterns = [
      /\b(authenticate|login|signin|auth)\w*/gi,
      /\b(validate|check|verify)\w*/gi,
      /\b(calculate|compute|process)\w*/gi,
      /\b(format|transform|convert)\w*/gi,
      /\b(handle|manage|control)\w*/gi,
      /\b(create|build|generate)\w*/gi,
      /\b(parse|decode|encode)\w*/gi,
      /\b(connect|disconnect|setup)\w*/gi
    ];
    
    for (const pattern of symbolPatterns) {
      const matches = query.match(pattern);
      if (matches) {
        entities.push(...matches);
      }
    }
    
    return [...new Set(entities)]; // Remove duplicates
  }
  
  /**
   * Infer search intent from NL query
   */
  private inferIntent(query: string, entities: string[]): QueryIntent {
    const lowerQuery = query.toLowerCase();
    
    // Infer target symbol kinds
    const targetKinds = [];
    if (lowerQuery.includes('function') || lowerQuery.includes('method')) {
      targetKinds.push('function');
    }
    if (lowerQuery.includes('class') || lowerQuery.includes('type')) {
      targetKinds.push('class', 'interface');
    }
    if (lowerQuery.includes('variable') || lowerQuery.includes('constant')) {
      targetKinds.push('variable', 'constant');
    }
    
    // Extract context keywords
    const contextPatterns = [
      /\b(logic|code|implementation|algorithm)\b/g,
      /\b(pattern|template|example|sample)\b/g,
      /\b(utility|helper|common|shared)\b/g,
      /\b(main|primary|core|central)\b/g
    ];
    
    const contextKeywords = [];
    for (const pattern of contextPatterns) {
      const matches = query.match(pattern);
      if (matches) {
        contextKeywords.push(...matches);
      }
    }
    
    // Calculate confidence based on entity extraction success
    const confidence = Math.min(entities.length * 0.3 + (contextKeywords.length * 0.2), 1.0);
    
    return {
      target_symbols: entities,
      target_kinds: targetKinds.length > 0 ? targetKinds : ['function', 'class', 'variable'],
      context_keywords: contextKeywords,
      confidence
    };
  }
  
  /**
   * Perform symbol bridging with controlled fanout
   */
  private async performSymbolBridge(
    nlQuery: NLQuery,
    symbolCandidates: SymbolCandidate[]
  ): Promise<SearchHit[]> {
    const bridgedHits: SearchHit[] = [];
    let fanoutCount = 0;
    
    console.log(`🔍 Symbol bridge: targeting ${nlQuery.inferred_intent.target_symbols.length} entities with m=${this.config.m_parameter}`);
    
    for (const entity of nlQuery.inferred_intent.target_symbols) {
      if (fanoutCount >= this.config.max_fanout) {
        console.log(`⚠️  Fanout cap reached: ${fanoutCount}/${this.config.max_fanout}`);
        break;
      }
      
      // Find symbol candidates matching this entity
      const matchingSymbols = this.findMatchingSymbols(entity, nlQuery.inferred_intent, symbolCandidates);
      
      // Apply m-parameter constraint: take top m matches per entity
      const topMatches = matchingSymbols.slice(0, this.config.m_parameter);
      
      for (const symbolMatch of topMatches) {
        if (fanoutCount >= this.config.max_fanout) break;
        
        // Convert symbol candidate to search hit with boost
        const searchHit = this.convertToSearchHit(symbolMatch, 'nl_bridge');
        bridgedHits.push(searchHit);
        fanoutCount++;
      }
    }
    
    console.log(`✨ Bridge produced ${bridgedHits.length} additional hits (fanout: ${fanoutCount})`);
    return bridgedHits;
  }
  
  /**
   * Find symbol candidates matching extracted entity
   */
  private findMatchingSymbols(
    entity: string,
    intent: QueryIntent,
    candidates: SymbolCandidate[]
  ): SymbolCandidate[] {
    const matches = [];
    const entityLower = entity.toLowerCase();
    
    for (const candidate of candidates) {
      let score = 0;
      
      // Exact name match (highest priority)
      if (candidate.symbol_name?.toLowerCase() === entityLower) {
        score += 1.0;
      }
      // Partial name match
      else if (candidate.symbol_name?.toLowerCase().includes(entityLower)) {
        score += 0.7;
      }
      // Fuzzy name match (edit distance)
      else if (this.fuzzyMatch(candidate.symbol_name || '', entity)) {
        score += 0.5;
      }
      
      // Symbol kind match bonus
      if (intent.target_kinds.includes(candidate.symbol_kind || '')) {
        score += 0.3;
      }
      
      // Context match bonus
      for (const keyword of intent.context_keywords) {
        if (candidate.file_path.toLowerCase().includes(keyword)) {
          score += 0.1;
        }
      }
      
      if (score >= this.config.semantic_similarity_threshold) {
        matches.push({
          ...candidate,
          score: candidate.score * (1 + score * this.config.symbol_boost_factor)
        });
      }
    }
    
    // Sort by enhanced score
    return matches.sort((a, b) => b.score - a.score);
  }
  
  /**
   * Simple fuzzy matching for symbol names
   */
  private fuzzyMatch(symbol: string, entity: string): boolean {
    const symbolLower = symbol.toLowerCase();
    const entityLower = entity.toLowerCase();
    
    // Simple edit distance check
    if (Math.abs(symbol.length - entity.length) > 3) return false;
    
    // Check if entity is subsequence of symbol or vice versa
    return symbolLower.includes(entityLower) || 
           entityLower.includes(symbolLower) ||
           this.computeEditDistance(symbolLower, entityLower) <= 2;
  }
  
  /**
   * Compute simple edit distance (Levenshtein)
   */
  private computeEditDistance(a: string, b: string): number {
    if (a.length === 0) return b.length;
    if (b.length === 0) return a.length;
    
    const dp = Array(a.length + 1).fill(null).map(() => Array(b.length + 1).fill(0));
    
    for (let i = 0; i <= a.length; i++) dp[i][0] = i;
    for (let j = 0; j <= b.length; j++) dp[0][j] = j;
    
    for (let i = 1; i <= a.length; i++) {
      for (let j = 1; j <= b.length; j++) {
        const cost = a[i-1] === b[j-1] ? 0 : 1;
        dp[i][j] = Math.min(
          dp[i-1][j] + 1,     // deletion
          dp[i][j-1] + 1,     // insertion  
          dp[i-1][j-1] + cost // substitution
        );
      }
    }
    
    return dp[a.length][b.length];
  }
  
  /**
   * Convert symbol candidate to search hit
   */
  private convertToSearchHit(candidate: SymbolCandidate, source: string): SearchHit {
    return {
      file: candidate.file_path,
      line: candidate.upstream_line || 1,
      col: candidate.upstream_col || 0,
      score: candidate.score,
      why: [...(candidate.match_reasons || []), source] as any,
      symbol_kind: candidate.symbol_kind as any,
      ast_path: candidate.ast_path,
      snippet: `${candidate.symbol_kind || 'symbol'} ${candidate.symbol_name || 'unknown'}`
    };
  }
  
  /**
   * Merge original and bridged hits with ranking
   */
  private mergeAndRank(original: SearchHit[], bridged: SearchHit[]): SearchHit[] {
    // Combine all hits
    const combined = [...original, ...bridged];
    
    // Remove duplicates based on file:line:col
    const deduplicated = [];
    const seen = new Set();
    
    for (const hit of combined) {
      const key = `${hit.file}:${hit.line}:${hit.col}`;
      if (!seen.has(key)) {
        seen.add(key);
        deduplicated.push(hit);
      }
    }
    
    // Sort by score (descending)
    return deduplicated.sort((a, b) => b.score - a.score);
  }
  
  /**
   * Calculate Success@10 improvement metrics
   */
  private calculateBridgeMetrics(
    original: SearchHit[],
    bridged: SearchHit[],
    combined: SearchHit[]
  ): BridgeMetrics {
    // Mock Success@10 calculation - in production would use actual relevance judgments
    const originalSuccess = Math.min(original.length / 10, 1.0);
    const bridgedSuccess = Math.min(combined.length / 10, 1.0);
    const improvement = bridgedSuccess - originalSuccess;
    
    return {
      original_success_at_10: originalSuccess,
      bridged_success_at_10: bridgedSuccess,
      improvement_delta: improvement,
      fanout_count: bridged.length,
      semantic_matches: bridged.filter(hit => 
        hit.why && Array.isArray(hit.why) && hit.why.includes('nl_bridge')
      ).length
    };
  }
  
  /**
   * Create empty metrics for fallback cases
   */
  private createEmptyMetrics(): BridgeMetrics {
    return {
      original_success_at_10: 0,
      bridged_success_at_10: 0,
      improvement_delta: 0,
      fanout_count: 0,
      semantic_matches: 0
    };
  }
  
  /**
   * Get current configuration
   */
  public getConfig(): NLSymbolBridgeConfig {
    return { ...this.config };
  }
  
  /**
   * Update configuration
   */
  public updateConfig(updates: Partial<NLSymbolBridgeConfig>): void {
    this.config = { ...this.config, ...updates };
    console.log(`🔧 NL→Symbol bridge config updated: m=${this.config.m_parameter}`);
  }
}

// Export singleton instance with v1.1 configuration
export const nlSymbolBridge = new NLSymbolBridge({
  m_parameter: 7,  // Upgraded from 5 as specified in TODO.md
  max_fanout: 15,  // Controlled fanout cap
  success_at_10_threshold: 0.62 // From TODO.md baseline
});