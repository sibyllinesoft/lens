/**
 * NL→Symbol Bridge for RAPTOR system
 * 
 * Provides natural language to symbol name mapping using:
 * - BM25 search over topic summaries and facets
 * - Subtoken synonym expansion for code terminology
 * - Symbol name extraction from top-m=5 relevant topics
 * - Exposes `nl_symbol_overlap` as rerank feature
 */

import { TopicTree, TopicNode } from './topic-tree.js';
import { SymbolGraph, SymbolNode } from './symbol-graph.js';
import { CardStore, EnhancedSemanticCard } from './card-store.js';

export interface NLSymbolBridgeConfig {
  // BM25 parameters
  k1: number; // term frequency saturation parameter
  b: number;  // length normalization parameter
  
  // Subtoken expansion
  enable_subtoken_expansion: boolean;
  max_subtokens_per_term: number;
  subtoken_min_length: number;
  
  // Symbol extraction
  top_m_topics: number; // Extract symbols from top-m topics
  min_topic_score: number;
  max_symbols_per_topic: number;
  
  // Overlap feature computation
  symbol_match_weight: number;
  partial_match_weight: number;
  subtoken_match_weight: number;
  
  // Performance controls
  search_timeout_ms: number;
  max_documents: number; // Max topic documents to consider
}

export interface NLQueryAnalysis {
  original_query: string;
  normalized_query: string;
  query_terms: string[];
  expanded_terms: SubtokenExpansion[];
  domain_keywords: string[];
  intent_indicators: string[];
}

export interface SubtokenExpansion {
  original_term: string;
  subtokens: string[];
  synonyms: string[];
  code_patterns: string[];
}

export interface BM25Document {
  id: string;
  content: string;
  tokens: string[];
  term_frequencies: Map<string, number>;
  doc_length: number;
}

export interface BM25SearchResult {
  document_id: string;
  score: number;
  matched_terms: string[];
  term_contributions: Map<string, number>;
}

export interface TopicSymbolExtraction {
  topic_id: string;
  topic_score: number;
  extracted_symbols: ExtractedSymbol[];
  symbol_sources: string[]; // Where symbols were found (summary, facets, keywords)
}

export interface ExtractedSymbol {
  name: string;
  kind?: string;
  confidence: number;
  extraction_reason: string;
  file_locations: string[];
  definition_contexts: string[];
}

export interface NLSymbolBridgeResult {
  query_analysis: NLQueryAnalysis;
  topic_matches: BM25SearchResult[];
  symbol_extractions: TopicSymbolExtraction[];
  final_symbols: ExtractedSymbol[];
  nl_symbol_overlap: string[]; // For rerank feature
  bridge_time_ms: number;
  bridge_quality_score: number;
}

/**
 * BM25 search engine for topic summaries
 */
class TopicBM25Engine {
  private documents: BM25Document[] = [];
  private vocab: Set<string> = new Set();
  private idf: Map<string, number> = new Map();
  private avgDocLength: number = 0;
  private config: NLSymbolBridgeConfig;

  constructor(config: NLSymbolBridgeConfig) {
    this.config = config;
  }

  indexTopics(topics: TopicNode[]): void {
    this.documents = [];
    this.vocab.clear();
    
    let totalLength = 0;
    
    for (const topic of topics) {
      // Combine topic content for BM25 indexing
      const content = [
        topic.summary,
        ...topic.facets,
        ...topic.keywords
      ].join(' ').toLowerCase();
      
      const tokens = this.tokenize(content);
      const termFreqs = new Map<string, number>();
      
      // Count term frequencies
      for (const token of tokens) {
        termFreqs.set(token, (termFreqs.get(token) || 0) + 1);
        this.vocab.add(token);
      }
      
      const document: BM25Document = {
        id: topic.id,
        content,
        tokens,
        term_frequencies: termFreqs,
        doc_length: tokens.length
      };
      
      this.documents.push(document);
      totalLength += tokens.length;
    }
    
    this.avgDocLength = totalLength / Math.max(1, this.documents.length);
    
    // Compute IDF for all terms
    this.computeIDF();
  }

  private tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(token => token.length > 1);
  }

  private computeIDF(): void {
    this.idf.clear();
    const N = this.documents.length;
    
    for (const term of this.vocab) {
      const df = this.documents.filter(doc => doc.term_frequencies.has(term)).length;
      const idf = Math.log((N - df + 0.5) / (df + 0.5));
      this.idf.set(term, Math.max(0, idf)); // Ensure non-negative IDF
    }
  }

  search(query: string[], maxResults: number = 10): BM25SearchResult[] {
    const results: BM25SearchResult[] = [];
    
    for (const document of this.documents) {
      const score = this.computeBM25Score(query, document);
      
      if (score > 0) {
        const matchedTerms = query.filter(term => document.term_frequencies.has(term));
        const termContributions = new Map<string, number>();
        
        for (const term of query) {
          if (document.term_frequencies.has(term)) {
            const termScore = this.computeTermScore(term, document);
            termContributions.set(term, termScore);
          }
        }
        
        results.push({
          document_id: document.id,
          score,
          matched_terms: matchedTerms,
          term_contributions: termContributions
        });
      }
    }
    
    results.sort((a, b) => b.score - a.score);
    return results.slice(0, maxResults);
  }

  private computeBM25Score(query: string[], document: BM25Document): number {
    let score = 0;
    
    for (const term of query) {
      const termScore = this.computeTermScore(term, document);
      score += termScore;
    }
    
    return score;
  }

  private computeTermScore(term: string, document: BM25Document): number {
    const tf = document.term_frequencies.get(term) || 0;
    if (tf === 0) return 0;
    
    const idf = this.idf.get(term) || 0;
    const { k1, b } = this.config;
    
    const numerator = tf * (k1 + 1);
    const denominator = tf + k1 * (1 - b + b * (document.doc_length / this.avgDocLength));
    
    return idf * (numerator / denominator);
  }
}

/**
 * NL→Symbol bridge implementation
 */
export class NLSymbolBridge {
  private topicTree?: TopicTree;
  private symbolGraph?: SymbolGraph;
  private cardStore?: CardStore;
  private config: NLSymbolBridgeConfig;
  private bm25Engine: TopicBM25Engine;
  
  // Caches for performance
  private subtokenCache = new Map<string, SubtokenExpansion>();
  private synonymCache = new Map<string, string[]>();

  constructor(config?: Partial<NLSymbolBridgeConfig>) {
    this.config = {
      // BM25 parameters (tuned for code search)
      k1: 1.2,
      b: 0.75,
      
      // Subtoken expansion
      enable_subtoken_expansion: true,
      max_subtokens_per_term: 5,
      subtoken_min_length: 2,
      
      // Symbol extraction
      top_m_topics: 5,
      min_topic_score: 0.1,
      max_symbols_per_topic: 20,
      
      // Overlap computation
      symbol_match_weight: 1.0,
      partial_match_weight: 0.6,
      subtoken_match_weight: 0.4,
      
      // Performance
      search_timeout_ms: 200,
      max_documents: 100,
      
      ...config
    };
    
    this.bm25Engine = new TopicBM25Engine(this.config);
  }

  /**
   * Initialize with RAPTOR components
   */
  initialize(topicTree: TopicTree, symbolGraph: SymbolGraph, cardStore: CardStore): void {
    this.topicTree = topicTree;
    this.symbolGraph = symbolGraph;
    this.cardStore = cardStore;
    
    // Index topics for BM25 search
    this.indexTopics();
  }

  private indexTopics(): void {
    if (!this.topicTree) {
      throw new Error('TopicTree not initialized');
    }
    
    const tree = this.topicTree.getTree();
    if (!tree) {
      throw new Error('TopicTree not loaded');
    }
    
    const topics = Array.from(tree.nodes.values())
      .filter(node => node.level > 0) // Exclude root
      .slice(0, this.config.max_documents);
    
    this.bm25Engine.indexTopics(topics);
  }

  /**
   * Bridge natural language query to symbol names
   */
  async bridgeNLToSymbols(query: string): Promise<NLSymbolBridgeResult> {
    const startTime = Date.now();
    
    if (!this.topicTree || !this.symbolGraph || !this.cardStore) {
      throw new Error('Components not initialized');
    }

    try {
      // Phase 1: Analyze and expand natural language query
      const queryAnalysis = await this.analyzeNLQuery(query);
      
      // Phase 2: Search topics using BM25
      const topicMatches = await this.searchTopicsWithBM25(queryAnalysis);
      
      // Phase 3: Extract symbols from top-m relevant topics
      const symbolExtractions = await this.extractSymbolsFromTopics(topicMatches);
      
      // Phase 4: Consolidate and rank final symbols
      const finalSymbols = this.consolidateSymbols(symbolExtractions);
      
      // Phase 5: Compute NL-symbol overlap for reranking
      const nl_symbol_overlap = this.computeNLSymbolOverlap(queryAnalysis, finalSymbols);
      
      const bridgeTime = Date.now() - startTime;
      const bridgeQuality = this.assessBridgeQuality(queryAnalysis, topicMatches, finalSymbols);
      
      return {
        query_analysis: queryAnalysis,
        topic_matches: topicMatches,
        symbol_extractions: symbolExtractions,
        final_symbols: finalSymbols,
        nl_symbol_overlap,
        bridge_time_ms: bridgeTime,
        bridge_quality_score: bridgeQuality
      };
      
    } catch (error) {
      // Return minimal result on error
      const basicAnalysis = this.createBasicQueryAnalysis(query);
      
      return {
        query_analysis: basicAnalysis,
        topic_matches: [],
        symbol_extractions: [],
        final_symbols: [],
        nl_symbol_overlap: [],
        bridge_time_ms: Date.now() - startTime,
        bridge_quality_score: 0
      };
    }
  }

  private async analyzeNLQuery(query: string): Promise<NLQueryAnalysis> {
    // Normalize query
    const normalized = query.toLowerCase().trim();
    
    // Basic tokenization
    const queryTerms = normalized
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(term => term.length > 1);
    
    // Expand terms with subtokens if enabled
    const expandedTerms: SubtokenExpansion[] = [];
    
    if (this.config.enable_subtoken_expansion) {
      for (const term of queryTerms) {
        const expansion = await this.expandWithSubtokens(term);
        expandedTerms.push(expansion);
      }
    }
    
    // Extract domain keywords (code-related terms)
    const domainKeywords = this.extractDomainKeywords(queryTerms);
    
    // Extract intent indicators
    const intentIndicators = this.extractIntentIndicators(queryTerms);
    
    return {
      original_query: query,
      normalized_query: normalized,
      query_terms: queryTerms,
      expanded_terms: expandedTerms,
      domain_keywords: domainKeywords,
      intent_indicators: intentIndicators
    };
  }

  private async expandWithSubtokens(term: string): Promise<SubtokenExpansion> {
    if (this.subtokenCache.has(term)) {
      return this.subtokenCache.get(term)!;
    }
    
    // Extract subtokens using camelCase and underscore patterns
    const subtokens = this.extractSubtokens(term);
    
    // Get synonyms for the term
    const synonyms = await this.getSynonyms(term);
    
    // Extract code patterns
    const codePatterns = this.extractCodePatterns(term);
    
    const expansion: SubtokenExpansion = {
      original_term: term,
      subtokens,
      synonyms,
      code_patterns: codePatterns
    };
    
    this.subtokenCache.set(term, expansion);
    return expansion;
  }

  private extractSubtokens(term: string): string[] {
    const subtokens: string[] = [];
    
    // Split camelCase: getUserData -> [get, user, data]
    const camelCaseSplit = term.replace(/([a-z])([A-Z])/g, '$1 $2').toLowerCase();
    subtokens.push(...camelCaseSplit.split(/\s+/));
    
    // Split underscores: user_data -> [user, data]
    const underscoreSplit = term.split('_');
    subtokens.push(...underscoreSplit);
    
    // Split hyphens: user-data -> [user, data]
    const hyphenSplit = term.split('-');
    subtokens.push(...hyphenSplit);
    
    // Filter and deduplicate
    return [...new Set(subtokens.filter(token => 
      token.length >= this.config.subtoken_min_length
    ))].slice(0, this.config.max_subtokens_per_term);
  }

  private async getSynonyms(term: string): Promise<string[]> {
    if (this.synonymCache.has(term)) {
      return this.synonymCache.get(term)!;
    }
    
    // Simple synonym mapping for common programming terms
    const synonymMap: Record<string, string[]> = {
      'get': ['fetch', 'retrieve', 'obtain', 'acquire'],
      'set': ['assign', 'update', 'modify', 'change'],
      'create': ['make', 'build', 'generate', 'construct'],
      'delete': ['remove', 'destroy', 'eliminate', 'drop'],
      'find': ['search', 'locate', 'discover', 'lookup'],
      'show': ['display', 'render', 'present', 'view'],
      'handle': ['process', 'manage', 'deal', 'treat'],
      'validate': ['check', 'verify', 'confirm', 'ensure'],
      'parse': ['analyze', 'process', 'decode', 'interpret'],
      'format': ['style', 'layout', 'structure', 'arrange'],
      'calculate': ['compute', 'evaluate', 'determine', 'derive'],
      'initialize': ['setup', 'prepare', 'configure', 'start'],
      'authenticate': ['login', 'verify', 'authorize', 'validate'],
      'serialize': ['encode', 'convert', 'transform', 'stringify']
    };
    
    const synonyms = synonymMap[term.toLowerCase()] || [];
    this.synonymCache.set(term, synonyms);
    
    return synonyms;
  }

  private extractCodePatterns(term: string): string[] {
    const patterns: string[] = [];
    
    // Common method patterns
    if (term.match(/^(get|set|is|has|can|should|will)/i)) {
      patterns.push('accessor_method');
    }
    
    if (term.match(/^(create|make|build|generate)/i)) {
      patterns.push('factory_method');
    }
    
    if (term.match(/^(validate|check|verify|ensure)/i)) {
      patterns.push('validation_method');
    }
    
    if (term.match(/^(handle|process|manage)/i)) {
      patterns.push('handler_method');
    }
    
    // Common class patterns
    if (term.match(/(manager|handler|service|controller|provider)$/i)) {
      patterns.push('service_class');
    }
    
    if (term.match(/(factory|builder|creator)$/i)) {
      patterns.push('factory_class');
    }
    
    return patterns;
  }

  private extractDomainKeywords(terms: string[]): string[] {
    const domainTerms = [
      'function', 'method', 'class', 'interface', 'type', 'variable',
      'component', 'service', 'api', 'endpoint', 'route', 'handler',
      'database', 'query', 'model', 'schema', 'table', 'field',
      'user', 'authentication', 'authorization', 'permission', 'role',
      'data', 'config', 'setting', 'option', 'parameter', 'argument',
      'response', 'request', 'client', 'server', 'protocol', 'format'
    ];
    
    return terms.filter(term => domainTerms.includes(term.toLowerCase()));
  }

  private extractIntentIndicators(terms: string[]): string[] {
    const intentTerms = [
      'find', 'search', 'get', 'fetch', 'retrieve',
      'show', 'display', 'list', 'view', 'render',
      'create', 'make', 'build', 'generate', 'add',
      'update', 'modify', 'change', 'set', 'edit',
      'delete', 'remove', 'drop', 'destroy', 'clear',
      'validate', 'check', 'verify', 'test', 'ensure'
    ];
    
    return terms.filter(term => intentTerms.includes(term.toLowerCase()));
  }

  private async searchTopicsWithBM25(queryAnalysis: NLQueryAnalysis): Promise<BM25SearchResult[]> {
    // Combine original terms with expanded terms for search
    const searchTerms = new Set<string>();
    
    // Add original query terms
    for (const term of queryAnalysis.query_terms || []) {
      searchTerms.add(term);
    }
    
    // Add subtokens and synonyms
    for (const expansion of queryAnalysis.expanded_terms || []) {
      for (const subtoken of expansion.subtokens || []) {
        searchTerms.add(subtoken);
      }
      for (const synonym of expansion.synonyms || []) {
        searchTerms.add(synonym);
      }
    }
    
    // Add domain keywords with higher weight
    for (const keyword of queryAnalysis.domain_keywords || []) {
      searchTerms.add(keyword);
    }
    
    const searchQuery = Array.from(searchTerms);
    return this.bm25Engine.search(searchQuery, this.config.top_m_topics * 2); // Get extra for filtering
  }

  private async extractSymbolsFromTopics(
    topicMatches: BM25SearchResult[]
  ): Promise<TopicSymbolExtraction[]> {
    const extractions: TopicSymbolExtraction[] = [];
    
    // Take top-m topics that meet minimum score threshold
    const qualifyingTopics = topicMatches
      .filter(match => match.score >= this.config.min_topic_score)
      .slice(0, this.config.top_m_topics);
    
    for (const topicMatch of qualifyingTopics) {
      const topic = this.topicTree!.getTopic(topicMatch.document_id);
      if (!topic) continue;
      
      const extractedSymbols = await this.extractSymbolsFromTopic(topic, topicMatch);
      
      extractions.push({
        topic_id: topic.id,
        topic_score: topicMatch.score,
        extracted_symbols: extractedSymbols,
        symbol_sources: ['summary', 'facets', 'keywords'] // Where symbols were found
      });
    }
    
    return extractions;
  }

  private async extractSymbolsFromTopic(
    topic: TopicNode,
    topicMatch: BM25SearchResult
  ): Promise<ExtractedSymbol[]> {
    const symbols: ExtractedSymbol[] = [];
    
    // Get cards associated with this topic
    const associatedCards: EnhancedSemanticCard[] = [];
    
    for (const cardId of topic.card_ids) {
      const card = this.cardStore!.getCard(cardId);
      if (card) {
        associatedCards.push(card);
      }
    }
    
    // Extract symbols from cards
    const seenSymbols = new Set<string>();
    
    for (const card of associatedCards.slice(0, this.config.max_symbols_per_topic)) {
      for (const symbol of card.symbols) {
        if (seenSymbols.has(symbol.name)) continue;
        seenSymbols.add(symbol.name);
        
        const confidence = this.computeSymbolExtractionConfidence(
          symbol,
          topicMatch,
          card
        );
        
        if (confidence > 0.1) { // Minimum confidence threshold
          symbols.push({
            name: symbol.name,
            kind: symbol.kind,
            confidence,
            extraction_reason: `Found in topic ${topic.id} via ${topicMatch.matched_terms.join(', ')}`,
            file_locations: [card.file_path],
            definition_contexts: [card.summary || '']
          });
        }
      }
    }
    
    // Sort by confidence and take top symbols
    return symbols
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, this.config.max_symbols_per_topic);
  }

  private computeSymbolExtractionConfidence(
    symbol: any,
    topicMatch: BM25SearchResult,
    card: EnhancedSemanticCard
  ): number {
    let confidence = 0;
    
    // Base confidence from topic match score
    confidence += topicMatch.score * 0.3;
    
    // Boost for symbol name appearing in matched terms
    const symbolNameLower = symbol.name.toLowerCase();
    if (topicMatch.matched_terms.some(term => 
      symbolNameLower.includes(term) || term.includes(symbolNameLower)
    )) {
      confidence += 0.4;
    }
    
    // Boost for high businessness (more likely to be relevant)
    if (card.businessness.B > 0.5) {
      confidence += 0.2;
    }
    
    // Boost for certain symbol kinds
    if (['function', 'method', 'class', 'interface'].includes(symbol.kind)) {
      confidence += 0.1;
    }
    
    return Math.min(confidence, 1.0);
  }

  private consolidateSymbols(extractions: TopicSymbolExtraction[]): ExtractedSymbol[] {
    const symbolMap = new Map<string, ExtractedSymbol>();
    
    for (const extraction of extractions) {
      for (const symbol of extraction.extracted_symbols) {
        const existing = symbolMap.get(symbol.name);
        
        if (existing) {
          // Merge information
          existing.confidence = Math.max(existing.confidence, symbol.confidence);
          existing.file_locations.push(...symbol.file_locations);
          existing.definition_contexts.push(...symbol.definition_contexts);
          existing.extraction_reason += ` | ${symbol.extraction_reason}`;
        } else {
          symbolMap.set(symbol.name, { ...symbol });
        }
      }
    }
    
    // Deduplicate arrays and sort by confidence
    const consolidated = Array.from(symbolMap.values());
    
    for (const symbol of consolidated) {
      symbol.file_locations = [...new Set(symbol.file_locations)];
      symbol.definition_contexts = [...new Set(symbol.definition_contexts)];
    }
    
    return consolidated.sort((a, b) => b.confidence - a.confidence);
  }

  private computeNLSymbolOverlap(
    queryAnalysis: NLQueryAnalysis,
    extractedSymbols: ExtractedSymbol[]
  ): string[] {
    const overlap: string[] = [];
    
    if (extractedSymbols.length === 0) {
      return overlap;
    }
    
    // Get all query terms (including expanded)
    const allQueryTerms = new Set<string>();
    for (const term of queryAnalysis.query_terms || []) {
      allQueryTerms.add(term);
    }
    
    for (const expansion of queryAnalysis.expanded_terms || []) {
      for (const subtoken of expansion.subtokens || []) {
        allQueryTerms.add(subtoken);
      }
      for (const synonym of expansion.synonyms || []) {
        allQueryTerms.add(synonym);
      }
    }
    
    // Check for overlaps with symbol names
    for (const symbol of extractedSymbols) {
      const symbolNameLower = symbol.name.toLowerCase();
      
      // Exact matches
      if (allQueryTerms.has(symbolNameLower)) {
        overlap.push(symbol.name);
        continue;
      }
      
      // Partial matches (symbol name contains query term)
      for (const term of allQueryTerms) {
        if (symbolNameLower.includes(term) || term.includes(symbolNameLower)) {
          overlap.push(symbol.name);
          break;
        }
      }
    }
    
    return [...new Set(overlap)]; // Remove duplicates
  }

  private assessBridgeQuality(
    queryAnalysis: NLQueryAnalysis,
    topicMatches: BM25SearchResult[],
    finalSymbols: ExtractedSymbol[]
  ): number {
    let qualityScore = 0;
    
    // Quality based on topic match strength
    if (topicMatches.length > 0) {
      const avgTopicScore = topicMatches.reduce((sum, match) => sum + match.score, 0) / topicMatches.length;
      qualityScore += avgTopicScore * 0.4;
    }
    
    // Quality based on symbol extraction confidence
    if (finalSymbols.length > 0) {
      const avgSymbolConfidence = finalSymbols.reduce((sum, symbol) => sum + symbol.confidence, 0) / finalSymbols.length;
      qualityScore += avgSymbolConfidence * 0.4;
    }
    
    // Quality based on query analysis richness
    const queryRichness = Math.min(
      queryAnalysis.expanded_terms.length / 5 +
      queryAnalysis.domain_keywords.length / 3 +
      queryAnalysis.intent_indicators.length / 2,
      1.0
    );
    qualityScore += queryRichness * 0.2;
    
    return Math.min(qualityScore, 1.0);
  }

  private createBasicQueryAnalysis(query: string): NLQueryAnalysis {
    const normalized = query.toLowerCase().trim();
    const queryTerms = normalized.split(/\s+/).filter(term => term.length > 1);
    
    return {
      original_query: query,
      normalized_query: normalized,
      query_terms: queryTerms,
      expanded_terms: [],
      domain_keywords: [],
      intent_indicators: []
    };
  }

  /**
   * Compute overlap feature for reranking
   */
  computeNLSymbolOverlapFeature(
    candidateSymbols: string[],
    nlSymbolOverlap: string[]
  ): number {
    if (nlSymbolOverlap.length === 0 || candidateSymbols.length === 0) {
      return 0;
    }
    
    let overlapScore = 0;
    
    for (const candidateSymbol of candidateSymbols) {
      const candidateNameLower = candidateSymbol.toLowerCase();
      
      for (const nlSymbol of nlSymbolOverlap) {
        const nlSymbolLower = nlSymbol.toLowerCase();
        
        if (candidateNameLower === nlSymbolLower) {
          overlapScore += this.config.symbol_match_weight;
        } else if (candidateNameLower.includes(nlSymbolLower) || nlSymbolLower.includes(candidateNameLower)) {
          overlapScore += this.config.partial_match_weight;
        } else {
          // Check subtoken matches
          const candidateSubtokens = this.extractSubtokens(candidateSymbol);
          const nlSubtokens = this.extractSubtokens(nlSymbol);
          
          for (const cSubtoken of candidateSubtokens) {
            for (const nSubtoken of nlSubtokens) {
              if (cSubtoken === nSubtoken) {
                overlapScore += this.config.subtoken_match_weight;
              }
            }
          }
        }
      }
    }
    
    // Normalize by candidate symbol count
    return overlapScore / candidateSymbols.length;
  }

  /**
   * Get bridge configuration
   */
  getConfig(): NLSymbolBridgeConfig {
    return { ...this.config };
  }

  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<NLSymbolBridgeConfig>): void {
    this.config = { ...this.config, ...newConfig };
    this.bm25Engine = new TopicBM25Engine(this.config);
    
    // Re-index if topics are available
    if (this.topicTree) {
      this.indexTopics();
    }
  }

  /**
   * Clear caches
   */
  clearCaches(): void {
    this.subtokenCache.clear();
    this.synonymCache.clear();
  }

  /**
   * Export bridge performance statistics
   */
  exportBridgeStats(results: NLSymbolBridgeResult[]): {
    avg_bridge_time_ms: number;
    avg_quality_score: number;
    avg_symbols_extracted: number;
    topic_match_rate: number;
    symbol_overlap_rate: number;
  } {
    if (results.length === 0) {
      return {
        avg_bridge_time_ms: 0,
        avg_quality_score: 0,
        avg_symbols_extracted: 0,
        topic_match_rate: 0,
        symbol_overlap_rate: 0
      };
    }

    const avgBridgeTime = results.reduce((sum, r) => sum + r.bridge_time_ms, 0) / results.length;
    const avgQuality = results.reduce((sum, r) => sum + r.bridge_quality_score, 0) / results.length;
    const avgSymbols = results.reduce((sum, r) => sum + r.final_symbols.length, 0) / results.length;
    
    const topicMatchRate = results.filter(r => r.topic_matches.length > 0).length / results.length;
    const symbolOverlapRate = results.filter(r => r.nl_symbol_overlap.length > 0).length / results.length;
    
    return {
      avg_bridge_time_ms: avgBridgeTime,
      avg_quality_score: avgQuality,
      avg_symbols_extracted: avgSymbols,
      topic_match_rate: topicMatchRate,
      symbol_overlap_rate: symbolOverlapRate
    };
  }
}