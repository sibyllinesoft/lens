/**
 * Intent Router for Query Classification
 * Pre-parser before /search that classifies queries into {def, refs, symbol, struct, lexical, NL}
 * When def|refs, hit /symbols/near first (or LSP hint cache)
 * Fall back to full search if empty, reflect in "why" so benchmark knows intent was honored
 */

import type { 
  QueryIntent, 
  IntentClassification, 
  SearchContext,
  Candidate 
} from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';
import { LSPStageBEnhancer } from './lsp-stage-b.js';

interface IntentRouterResult {
  classification: IntentClassification;
  primary_candidates: Candidate[];
  fallback_triggered: boolean;
  routing_path: string[];
  confidence_threshold_met: boolean;
}

export class IntentRouter {
  private static readonly CONFIDENCE_THRESHOLD = 0.7;
  private static readonly MAX_PRIMARY_RESULTS = 20;

  constructor(
    private lspEnhancer: LSPStageBEnhancer
  ) {}

  /**
   * Classify query intent and route to appropriate search strategy
   */
  async routeQuery(
    query: string,
    context: SearchContext,
    symbolsNearHandler?: (filePath: string, line: number) => Promise<Candidate[]>,
    fullSearchHandler?: (query: string, context: SearchContext) => Promise<Candidate[]>
  ): Promise<IntentRouterResult> {
    const span = LensTracer.createChildSpan('intent_router_route', {
      'search.query': query,
      'context.mode': context.mode,
    });

    try {
      // Step 1: Classify query intent
      const classification = this.classifyQueryIntent(query);
      const routingPath: string[] = [`classified_as_${classification.intent}`];

      let primaryCandidates: Candidate[] = [];
      let fallbackTriggered = false;
      const confidenceThresholdMet = classification.confidence >= IntentRouter.CONFIDENCE_THRESHOLD;

      // Step 2: Route based on intent and confidence
      if (confidenceThresholdMet) {
        switch (classification.intent) {
          case 'def':
            primaryCandidates = await this.handleDefinitionIntent(query, context, symbolsNearHandler);
            routingPath.push('definition_search');
            break;
          
          case 'refs':
            primaryCandidates = await this.handleReferencesIntent(query, context, symbolsNearHandler);
            routingPath.push('references_search');
            break;
          
          case 'symbol':
            primaryCandidates = await this.handleSymbolIntent(query, context);
            routingPath.push('symbol_search');
            break;
          
          case 'struct':
            primaryCandidates = await this.handleStructuralIntent(query, context);
            routingPath.push('structural_search');
            break;
          
          case 'lexical':
            // Skip specialized routing, go directly to full search
            routingPath.push('lexical_direct');
            break;
          
          case 'NL':
            primaryCandidates = await this.handleNaturalLanguageIntent(query, context);
            routingPath.push('nl_search');
            break;
        }

        // Step 3: Check if we need fallback
        if (primaryCandidates.length === 0 && 
            ['def', 'refs', 'symbol'].includes(classification.intent) &&
            fullSearchHandler) {
          
          routingPath.push('fallback_triggered');
          primaryCandidates = await fullSearchHandler(query, context);
          fallbackTriggered = true;
        }
      } else {
        // Low confidence - go directly to full search
        routingPath.push('low_confidence_full_search');
        if (fullSearchHandler) {
          primaryCandidates = await fullSearchHandler(query, context);
        }
      }

      // Step 4: Add intent information to candidates
      this.enrichCandidatesWithIntent(primaryCandidates, classification, fallbackTriggered);

      const result: IntentRouterResult = {
        classification,
        primary_candidates: primaryCandidates,
        fallback_triggered: fallbackTriggered,
        routing_path: routingPath,
        confidence_threshold_met: confidenceThresholdMet,
      };

      span.setAttributes({
        success: true,
        intent: classification.intent,
        confidence: classification.confidence,
        primary_results: primaryCandidates.length,
        fallback_triggered: fallbackTriggered,
        routing_path: routingPath.join(' -> '),
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
   * Classify query intent using pattern matching and heuristics
   */
  classifyQueryIntent(query: string): IntentClassification {
    const features = this.extractQueryFeatures(query);
    let intent: QueryIntent = 'lexical';
    let confidence = 0.5;

    // Definition patterns
    if (features.has_definition_pattern) {
      intent = 'def';
      confidence = 0.9;
    } 
    // References patterns
    else if (features.has_reference_pattern) {
      intent = 'refs';
      confidence = 0.9;
    }
    // Symbol search patterns
    else if (features.has_symbol_prefix) {
      intent = 'symbol';
      confidence = 0.8;
    }
    // Structural patterns
    else if (features.has_structural_chars) {
      intent = 'struct';
      confidence = 0.75;
    }
    // Natural language patterns
    else if (features.is_natural_language) {
      intent = 'NL';
      confidence = 0.7;
    }
    // Default to lexical with moderate confidence
    else {
      intent = 'lexical';
      confidence = 0.6;
    }

    return {
      intent,
      confidence,
      features,
    };
  }

  /**
   * Extract features from query for intent classification
   */
  private extractQueryFeatures(query: string): IntentClassification['features'] {
    const queryLower = query.toLowerCase().trim();
    
    return {
      has_definition_pattern: this.hasDefinitionPattern(queryLower),
      has_reference_pattern: this.hasReferencePattern(queryLower),
      has_symbol_prefix: this.hasSymbolPrefix(queryLower),
      has_structural_chars: this.hasStructuralChars(query),
      is_natural_language: this.isNaturalLanguage(queryLower),
    };
  }

  /**
   * Check for definition-seeking patterns
   */
  private hasDefinitionPattern(query: string): boolean {
    const definitionPatterns = [
      /^(def|define|definition|declare)\s+/,
      /^(what is|where is|find definition)\s+/,
      /^(class|function|interface|type)\s+\w+$/,
      /^go to definition/,
      /^\w+\s+(definition|declaration)$/,
    ];

    return definitionPatterns.some(pattern => pattern.test(query));
  }

  /**
   * Check for reference-seeking patterns
   */
  private hasReferencePattern(query: string): boolean {
    const referencePatterns = [
      /^(refs|references|usages|uses)\s+/,
      /^(find|show|list)\s+(references|usages|uses)/,
      /^(where|who)\s+(uses|calls|references)/,
      /^\w+\s+(references|usages|calls)$/,
    ];

    return referencePatterns.some(pattern => pattern.test(query));
  }

  /**
   * Check for symbol-specific patterns
   */
  private hasSymbolPrefix(query: string): boolean {
    const symbolPrefixes = [
      /^(class|function|method|var|const|let|type|interface|enum)\s+/,
      /^[A-Z][a-zA-Z0-9_]*$/, // PascalCase (likely class/type)
      /^[a-z][a-zA-Z0-9_]*\(\)$/, // function call
      /^\w+\.\w+/, // member access
      /^@\w+/, // decorators/annotations
    ];

    return symbolPrefixes.some(pattern => pattern.test(query));
  }

  /**
   * Check for structural/syntactic characters
   */
  private hasStructuralChars(query: string): boolean {
    const structuralChars = /[{}[\]()<>=!&|+\-*/^%~]/;
    const bracketPairs = /[{}[\]()]/g;
    const operators = /[=!<>&|+\-*/^%~]/g;
    
    // Has structural characters
    if (!structuralChars.test(query)) return false;
    
    // Count brackets and operators
    const brackets = (query.match(bracketPairs) || []).length;
    const operatorMatches = (query.match(operators) || []).length;
    
    // Consider structural if has significant syntax elements
    return brackets >= 2 || operatorMatches >= 1;
  }

  /**
   * Check if query appears to be natural language
   */
  private isNaturalLanguage(query: string): boolean {
    const words = query.split(/\s+/);
    
    // Too short to be NL
    if (words.length < 3) return false;
    
    // Check for natural language indicators
    const nlIndicators = [
      /^(how to|how do|what|where|when|why|who)/,
      /^(find|show|get|list|search for)/,
      /(that|which|with|without|for|in|on|at|by)/,
      /(function|method|class|variable|property|component)/,
    ];

    const hasNlIndicators = nlIndicators.some(pattern => pattern.test(query));
    
    // Check for articles and prepositions
    const articles = ['a', 'an', 'the'];
    const prepositions = ['in', 'on', 'at', 'by', 'for', 'with', 'without', 'to', 'from'];
    const hasArticlesOrPrepositions = words.some(word => 
      articles.includes(word) || prepositions.includes(word)
    );

    return hasNlIndicators && hasArticlesOrPrepositions;
  }

  /**
   * Handle definition intent queries
   */
  private async handleDefinitionIntent(
    query: string,
    context: SearchContext,
    symbolsNearHandler?: (filePath: string, line: number) => Promise<Candidate[]>
  ): Promise<Candidate[]> {
    // Extract symbol name from definition query
    const symbolName = this.extractSymbolFromDefinitionQuery(query);
    if (!symbolName) return [];

    // Try LSP hints first
    const lspCandidates = await this.searchLSPForDefinition(symbolName, context);
    if (lspCandidates.length > 0) {
      return lspCandidates.slice(0, IntentRouter.MAX_PRIMARY_RESULTS);
    }

    // Fallback to symbols near if we have location context
    if (symbolsNearHandler && context.repo_sha) {
      // This would need file/line context - placeholder for now
      return [];
    }

    return [];
  }

  /**
   * Handle references intent queries
   */
  private async handleReferencesIntent(
    query: string,
    context: SearchContext,
    symbolsNearHandler?: (filePath: string, line: number) => Promise<Candidate[]>
  ): Promise<Candidate[]> {
    const symbolName = this.extractSymbolFromReferenceQuery(query);
    if (!symbolName) return [];

    // Search LSP hints for references
    const lspCandidates = await this.searchLSPForReferences(symbolName, context);
    return lspCandidates.slice(0, IntentRouter.MAX_PRIMARY_RESULTS);
  }

  /**
   * Handle symbol intent queries
   */
  private async handleSymbolIntent(
    query: string,
    context: SearchContext
  ): Promise<Candidate[]> {
    // Direct symbol search through LSP
    const results = this.lspEnhancer.enhanceStageB(query, context, [], IntentRouter.MAX_PRIMARY_RESULTS);
    return results.candidates;
  }

  /**
   * Handle structural intent queries
   */
  private async handleStructuralIntent(
    query: string,
    context: SearchContext
  ): Promise<Candidate[]> {
    // For structural queries, we might want to parse the syntax
    // and look for matching code patterns
    // This is a simplified implementation
    return [];
  }

  /**
   * Handle natural language intent queries
   */
  private async handleNaturalLanguageIntent(
    query: string,
    context: SearchContext
  ): Promise<Candidate[]> {
    // For NL queries, we'd want to use semantic search
    // This is a placeholder for semantic/embedding-based search
    return [];
  }

  /**
   * Extract symbol name from definition query
   */
  private extractSymbolFromDefinitionQuery(query: string): string | null {
    const patterns = [
      /^(?:def|define|definition|declare)\s+(\w+)/i,
      /^(?:what is|where is|find definition)\s+(\w+)/i,
      /^(?:class|function|interface|type)\s+(\w+)$/i,
      /^(\w+)\s+(?:definition|declaration)$/i,
    ];

    for (const pattern of patterns) {
      const match = query.match(pattern);
      if (match && match[1]) return match[1];
    }

    // If no pattern matches, assume the whole query is a symbol name
    const cleaned = query.replace(/^(def|define|definition|declare)\s+/i, '').trim();
    return cleaned.match(/^\w+$/) ? cleaned : null;
  }

  /**
   * Extract symbol name from references query
   */
  private extractSymbolFromReferenceQuery(query: string): string | null {
    const patterns = [
      /^(?:refs|references|usages|uses)\s+(\w+)/i,
      /^(?:find|show|list)\s+(?:references|usages|uses)\s+(?:of\s+)?(\w+)/i,
      /^(?:where|who)\s+(?:uses|calls|references)\s+(\w+)/i,
      /^(\w+)\s+(?:references|usages|calls)$/i,
    ];

    for (const pattern of patterns) {
      const match = query.match(pattern);
      if (match && match[1]) return match[1];
    }

    // If no pattern matches, assume the whole query is a symbol name
    const cleaned = query.replace(/^(refs|references|usages|uses)\s+/i, '').trim();
    return cleaned.match(/^\w+$/) ? cleaned : null;
  }

  /**
   * Search LSP hints for definitions
   */
  private async searchLSPForDefinition(symbolName: string, context: SearchContext): Promise<Candidate[]> {
    // Use LSP Stage-B enhancer to find definitions
    const results = this.lspEnhancer.enhanceStageB(symbolName, context, []);
    
    // Filter for definition-like results (higher score, specific match reasons)
    return results.candidates.filter(candidate => 
      candidate.match_reasons.includes('lsp_hint') &&
      candidate.score > 0.8
    );
  }

  /**
   * Search LSP hints for references
   */
  private async searchLSPForReferences(symbolName: string, context: SearchContext): Promise<Candidate[]> {
    // This would require extending LSP sidecar to track reference locations
    // For now, return enhanced results filtered by reference patterns
    const results = this.lspEnhancer.enhanceStageB(symbolName, context, []);
    return results.candidates;
  }

  /**
   * Enrich candidates with intent information
   */
  private enrichCandidatesWithIntent(
    candidates: Candidate[],
    classification: IntentClassification,
    fallbackTriggered: boolean
  ): void {
    for (const candidate of candidates) {
      // Add intent metadata
      (candidate as any).intent_classification = classification;
      (candidate as any).intent_honored = !fallbackTriggered;
      
      // Update why array to reflect routing decisions
      if (!(candidate as any).why) {
        (candidate as any).why = [];
      }
      
      (candidate as any).why.push(`intent_${classification.intent}`);
      
      if (fallbackTriggered) {
        (candidate as any).why.push('fallback_triggered');
      }
      
      if (classification.confidence >= IntentRouter.CONFIDENCE_THRESHOLD) {
        (candidate as any).why.push('high_confidence_routing');
      } else {
        (candidate as any).why.push('low_confidence_routing');
      }
    }
  }

  /**
   * Get intent router statistics
   */
  getStats(): {
    confidence_threshold: number;
    max_primary_results: number;
    classification_features: string[];
  } {
    return {
      confidence_threshold: IntentRouter.CONFIDENCE_THRESHOLD,
      max_primary_results: IntentRouter.MAX_PRIMARY_RESULTS,
      classification_features: [
        'has_definition_pattern',
        'has_reference_pattern', 
        'has_symbol_prefix',
        'has_structural_chars',
        'is_natural_language'
      ],
    };
  }

  /**
   * Update intent classification model (placeholder for ML integration)
   */
  updateClassificationModel(trainingData: Array<{
    query: string;
    actual_intent: QueryIntent;
    user_satisfaction: number;
  }>): void {
    // Placeholder for machine learning model updates
    // In production, this could train a more sophisticated classifier
    console.log(`Received ${trainingData.length} training examples for intent classification`);
  }
}