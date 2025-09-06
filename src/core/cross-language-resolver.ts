/**
 * Cross-Language Symbol Resolution System
 * Enables intelligent symbol resolution across different programming languages
 * Implements universal AST patterns, name normalization, and semantic matching
 */

import { performance } from 'perf_hooks';
import { LensTracer } from '../telemetry/tracer.js';
import { globalCacheManager } from './advanced-cache-manager.js';
import { globalParallelProcessor } from './parallel-processor.js';
import type { SearchContext, SearchHit } from '../types/core.js';

interface LanguageConfig {
  name: string;
  extensions: string[];
  commentPatterns: {
    singleLine: string[];
    multiLine: Array<{ start: string; end: string }>;
  };
  symbolPatterns: {
    function: RegExp[];
    class: RegExp[];
    method: RegExp[];
    variable: RegExp[];
    constant: RegExp[];
    interface: RegExp[];
    type: RegExp[];
    import: RegExp[];
    export: RegExp[];
  };
  namingConventions: {
    camelCase: boolean;
    snake_case: boolean;
    PascalCase: boolean;
    kebab_case: boolean;
    SCREAMING_SNAKE_CASE: boolean;
  };
  identifierRules: {
    allowedChars: RegExp;
    reservedWords: string[];
    caseSensitive: boolean;
  };
}

interface UniversalSymbol {
  id: string;
  name: string;
  normalizedName: string;
  type: SymbolType;
  language: string;
  file: string;
  line: number;
  column: number;
  signature?: string;
  returnType?: string;
  parameters?: Parameter[];
  modifiers: string[];
  scope: SymbolScope;
  documentation?: string;
  crossReferences: CrossReference[];
  semanticTags: string[];
}

interface Parameter {
  name: string;
  type: string;
  optional: boolean;
  defaultValue?: string;
}

interface CrossReference {
  id: string;
  type: 'implements' | 'extends' | 'calls' | 'imports' | 'exports' | 'references';
  targetSymbol: string;
  language: string;
  confidence: number;
}

enum SymbolType {
  FUNCTION = 'function',
  METHOD = 'method',
  CLASS = 'class',
  INTERFACE = 'interface',
  TYPE = 'type',
  VARIABLE = 'variable',
  CONSTANT = 'constant',
  PROPERTY = 'property',
  ENUM = 'enum',
  NAMESPACE = 'namespace',
  MODULE = 'module',
  IMPORT = 'import',
  EXPORT = 'export'
}

enum SymbolScope {
  GLOBAL = 'global',
  MODULE = 'module',
  CLASS = 'class',
  FUNCTION = 'function',
  BLOCK = 'block'
}

interface NameNormalizationRule {
  pattern: RegExp;
  replacement: string;
  language?: string;
  symbolType?: SymbolType;
}

interface SemanticMapping {
  sourcePattern: RegExp;
  targetLanguages: string[];
  equivalentPatterns: { [language: string]: RegExp };
  confidence: number;
}

export class CrossLanguageResolver {
  private static instance: CrossLanguageResolver;
  
  // Language configurations
  private languageConfigs: Map<string, LanguageConfig> = new Map();
  private extensionToLanguage: Map<string, string> = new Map();
  
  // Symbol storage and indexing
  private symbolIndex: Map<string, UniversalSymbol> = new Map();
  private nameToSymbols: Map<string, Set<string>> = new Map();
  private normalizedNameIndex: Map<string, Set<string>> = new Map();
  private crossReferenceGraph: Map<string, Set<string>> = new Map();
  
  // Resolution rules and mappings
  private normalizationRules: NameNormalizationRule[] = [];
  private semanticMappings: SemanticMapping[] = [];
  
  // Performance tracking
  private resolutionStats = {
    totalQueries: 0,
    crossLanguageHits: 0,
    cacheHitRate: 0,
    avgResolutionTime: 0
  };
  
  private constructor() {
    this.initializeLanguageConfigs();
    this.initializeNormalizationRules();
    this.initializeSemanticMappings();
  }
  
  public static getInstance(): CrossLanguageResolver {
    if (!CrossLanguageResolver.instance) {
      CrossLanguageResolver.instance = new CrossLanguageResolver();
    }
    return CrossLanguageResolver.instance;
  }
  
  /**
   * Initialize language configurations
   */
  private initializeLanguageConfigs(): void {
    const span = LensTracer.createChildSpan('init_language_configs');
    
    try {
      // TypeScript/JavaScript
      this.registerLanguage({
        name: 'typescript',
        extensions: ['.ts', '.tsx', '.d.ts'],
        commentPatterns: {
          singleLine: ['//'],
          multiLine: [{ start: '/*', end: '*/' }]
        },
        symbolPatterns: {
          function: [
            /function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(/g,
            /(?:export\s+)?(?:async\s+)?function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)/g,
            /const\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(?:async\s+)?\(/g,
            /([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:\s*\([^)]*\)\s*=>/g
          ],
          class: [
            /(?:export\s+)?(?:abstract\s+)?class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)/g
          ],
          method: [
            /(?:public|private|protected|static)?\s*([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(/g,
            /([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\([^)]*\)\s*\{/g
          ],
          variable: [
            /(?:let|var|const)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)/g
          ],
          constant: [
            /const\s+([A-Z_][A-Z0-9_]*)\s*=/g
          ],
          interface: [
            /(?:export\s+)?interface\s+([a-zA-Z_$][a-zA-Z0-9_$]*)/g
          ],
          type: [
            /(?:export\s+)?type\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=/g
          ],
          import: [
            /import\s+(?:\{[^}]*\}|\*\s+as\s+\w+|\w+)\s+from\s+['"']([^'"']+)['"']/g
          ],
          export: [
            /export\s+(?:default\s+)?(?:class|function|interface|type|const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)/g
          ]
        },
        namingConventions: {
          camelCase: true,
          snake_case: false,
          PascalCase: true,
          kebab_case: false,
          SCREAMING_SNAKE_CASE: true
        },
        identifierRules: {
          allowedChars: /^[a-zA-Z_$][a-zA-Z0-9_$]*$/,
          reservedWords: ['class', 'function', 'const', 'let', 'var', 'if', 'else', 'for', 'while'],
          caseSensitive: true
        }
      });
      
      // Python
      this.registerLanguage({
        name: 'python',
        extensions: ['.py', '.pyi', '.pyx'],
        commentPatterns: {
          singleLine: ['#'],
          multiLine: [{ start: '"""', end: '"""' }, { start: "'''", end: "'''" }]
        },
        symbolPatterns: {
          function: [
            /(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/g
          ],
          class: [
            /class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\([^)]*\))?\s*:/g
          ],
          method: [
            /def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(self/g,
            /@\w+\s*\n\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)/g
          ],
          variable: [
            /^([a-zA-Z_][a-zA-Z0-9_]*)\s*=/gm
          ],
          constant: [
            /^([A-Z_][A-Z0-9_]*)\s*=/gm
          ],
          interface: [
            /class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(Protocol\)/g
          ],
          type: [
            /([A-Z][a-zA-Z0-9_]*)\s*=\s*(?:TypeVar|NewType|Union|Optional)/g
          ],
          import: [
            /from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import/g,
            /import\s+([a-zA-Z_][a-zA-Z0-9_.]*)/g
          ],
          export: [
            /__all__\s*=\s*\[([^\]]+)\]/g
          ]
        },
        namingConventions: {
          camelCase: false,
          snake_case: true,
          PascalCase: true,
          kebab_case: false,
          SCREAMING_SNAKE_CASE: true
        },
        identifierRules: {
          allowedChars: /^[a-zA-Z_][a-zA-Z0-9_]*$/,
          reservedWords: ['class', 'def', 'import', 'from', 'if', 'else', 'for', 'while', 'try', 'except'],
          caseSensitive: true
        }
      });
      
      // Rust
      this.registerLanguage({
        name: 'rust',
        extensions: ['.rs'],
        commentPatterns: {
          singleLine: ['//'],
          multiLine: [{ start: '/*', end: '*/' }]
        },
        symbolPatterns: {
          function: [
            /(?:pub\s+)?(?:async\s+)?fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/g
          ],
          class: [
            /(?:pub\s+)?struct\s+([A-Z][a-zA-Z0-9_]*)/g,
            /(?:pub\s+)?enum\s+([A-Z][a-zA-Z0-9_]*)/g
          ],
          method: [
            /(?:pub\s+)?(?:async\s+)?fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(&?self/g
          ],
          variable: [
            /let\s+(?:mut\s+)?([a-zA-Z_][a-zA-Z0-9_]*)/g
          ],
          constant: [
            /const\s+([A-Z_][A-Z0-9_]*)\s*:/g
          ],
          interface: [
            /(?:pub\s+)?trait\s+([A-Z][a-zA-Z0-9_]*)/g
          ],
          type: [
            /type\s+([A-Z][a-zA-Z0-9_]*)\s*=/g
          ],
          import: [
            /use\s+([a-zA-Z_][a-zA-Z0-9_:]*)/g
          ],
          export: [
            /pub\s+(?:fn|struct|enum|trait|type|const|static)\s+([a-zA-Z_][a-zA-Z0-9_]*)/g
          ]
        },
        namingConventions: {
          camelCase: false,
          snake_case: true,
          PascalCase: true,
          kebab_case: true,
          SCREAMING_SNAKE_CASE: true
        },
        identifierRules: {
          allowedChars: /^[a-zA-Z_][a-zA-Z0-9_]*$/,
          reservedWords: ['fn', 'struct', 'enum', 'trait', 'impl', 'use', 'pub', 'let', 'const'],
          caseSensitive: true
        }
      });
      
      // Go
      this.registerLanguage({
        name: 'go',
        extensions: ['.go'],
        commentPatterns: {
          singleLine: ['//'],
          multiLine: [{ start: '/*', end: '*/' }]
        },
        symbolPatterns: {
          function: [
            /func\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/g
          ],
          class: [
            /type\s+([A-Z][a-zA-Z0-9_]*)\s+struct/g
          ],
          method: [
            /func\s+\([^)]+\)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/g
          ],
          variable: [
            /var\s+([a-zA-Z_][a-zA-Z0-9_]*)/g,
            /([a-zA-Z_][a-zA-Z0-9_]*)\s*:=/g
          ],
          constant: [
            /const\s+([A-Z_][A-Z0-9_]*)/g
          ],
          interface: [
            /type\s+([A-Z][a-zA-Z0-9_]*)\s+interface/g
          ],
          type: [
            /type\s+([A-Z][a-zA-Z0-9_]*)\s+/g
          ],
          import: [
            /import\s+"([^"]+)"/g,
            /import\s+\(\s*"([^"]+)"/g
          ],
          export: [] // Go uses capitalization for exports
        },
        namingConventions: {
          camelCase: true,
          snake_case: false,
          PascalCase: true,
          kebab_case: false,
          SCREAMING_SNAKE_CASE: false
        },
        identifierRules: {
          allowedChars: /^[a-zA-Z_][a-zA-Z0-9_]*$/,
          reservedWords: ['func', 'type', 'struct', 'interface', 'import', 'package', 'var', 'const'],
          caseSensitive: true
        }
      });
      
      console.log(`🌐 Initialized ${this.languageConfigs.size} language configurations for cross-language resolution`);
      
      span.setAttributes({
        success: true,
        language_count: this.languageConfigs.size
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
   * Register a language configuration
   */
  private registerLanguage(config: LanguageConfig): void {
    this.languageConfigs.set(config.name, config);
    
    // Build extension to language mapping
    for (const ext of config.extensions) {
      this.extensionToLanguage.set(ext, config.name);
    }
  }
  
  /**
   * Initialize name normalization rules
   */
  private initializeNormalizationRules(): void {
    this.normalizationRules = [
      // Convert camelCase to snake_case
      {
        pattern: /([a-z])([A-Z])/g,
        replacement: '$1_$2'
      },
      // Convert PascalCase to snake_case
      {
        pattern: /([A-Z])([A-Z][a-z])/g,
        replacement: '$1_$2'
      },
      // Normalize kebab-case to snake_case
      {
        pattern: /-/g,
        replacement: '_'
      },
      // Remove leading underscores for comparison
      {
        pattern: /^_+/,
        replacement: ''
      },
      // Normalize multiple underscores
      {
        pattern: /_+/g,
        replacement: '_'
      }
    ];
  }
  
  /**
   * Initialize semantic mappings between languages
   */
  private initializeSemanticMappings(): void {
    this.semanticMappings = [
      // Function/method patterns
      {
        sourcePattern: /function|def|fn|func/,
        targetLanguages: ['typescript', 'python', 'rust', 'go'],
        equivalentPatterns: {
          typescript: /function|=>/,
          python: /def/,
          rust: /fn/,
          go: /func/
        },
        confidence: 0.9
      },
      // Class/struct patterns
      {
        sourcePattern: /class|struct/,
        targetLanguages: ['typescript', 'python', 'rust', 'go'],
        equivalentPatterns: {
          typescript: /class/,
          python: /class/,
          rust: /struct/,
          go: /struct/
        },
        confidence: 0.85
      },
      // Interface/trait patterns
      {
        sourcePattern: /interface|trait/,
        targetLanguages: ['typescript', 'python', 'rust', 'go'],
        equivalentPatterns: {
          typescript: /interface/,
          python: /Protocol/,
          rust: /trait/,
          go: /interface/
        },
        confidence: 0.8
      }
    ];
  }
  
  /**
   * Resolve symbols across languages based on search context
   */
  async resolveCrossLanguage(context: SearchContext): Promise<SearchHit[]> {
    const span = LensTracer.createChildSpan('cross_language_resolve');
    const startTime = performance.now();
    this.resolutionStats.totalQueries++;
    
    try {
      // Check cache first
      const cacheKey = this.generateCacheKey(context);
      const cached = await globalCacheManager.get<SearchHit[]>(cacheKey, context);
      if (cached) {
        this.resolutionStats.cacheHitRate = 
          (this.resolutionStats.cacheHitRate * 0.9) + (1.0 * 0.1);
        return cached;
      }
      
      // Extract symbol information from query
      const querySymbols = this.extractSymbolsFromQuery(context.query);
      
      if (querySymbols.length === 0) {
        return [];
      }
      
      // Find equivalent symbols across languages
      const crossLanguageMatches: SearchHit[] = [];
      
      for (const querySymbol of querySymbols) {
        const matches = await this.findEquivalentSymbols(querySymbol, context);
        crossLanguageMatches.push(...matches);
      }
      
      // Rank and deduplicate results
      const rankedResults = this.rankCrossLanguageResults(crossLanguageMatches, context);
      const finalResults = rankedResults.slice(0, context.k);
      
      // Cache results
      await globalCacheManager.set(cacheKey, finalResults, 300000, context); // 5 minute TTL
      
      // Update statistics
      const duration = performance.now() - startTime;
      this.resolutionStats.avgResolutionTime = 
        (this.resolutionStats.avgResolutionTime * 0.9) + (duration * 0.1);
      
      if (finalResults.some(hit => hit.language && hit.language !== this.getQueryLanguage(context))) {
        this.resolutionStats.crossLanguageHits++;
      }
      
      span.setAttributes({
        success: true,
        query_symbols: querySymbols.length,
        cross_matches: crossLanguageMatches.length,
        final_results: finalResults.length,
        duration_ms: duration
      });
      
      return finalResults;
      
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
   * Extract symbols from search query
   */
  private extractSymbolsFromQuery(query: string): Array<{
    name: string;
    normalizedName: string;
    type?: SymbolType;
    context: string;
  }> {
    const symbols: Array<{
      name: string;
      normalizedName: string;
      type?: SymbolType;
      context: string;
    }> = [];
    
    // Simple heuristics for symbol extraction
    const patterns = [
      { pattern: /function\s+(\w+)/g, type: SymbolType.FUNCTION },
      { pattern: /class\s+(\w+)/g, type: SymbolType.CLASS },
      { pattern: /def\s+(\w+)/g, type: SymbolType.FUNCTION },
      { pattern: /fn\s+(\w+)/g, type: SymbolType.FUNCTION },
      { pattern: /interface\s+(\w+)/g, type: SymbolType.INTERFACE },
      { pattern: /trait\s+(\w+)/g, type: SymbolType.INTERFACE },
      { pattern: /struct\s+(\w+)/g, type: SymbolType.CLASS },
      { pattern: /type\s+(\w+)/g, type: SymbolType.TYPE },
      { pattern: /const\s+(\w+)/g, type: SymbolType.CONSTANT },
      { pattern: /let\s+(\w+)/g, type: SymbolType.VARIABLE },
      { pattern: /var\s+(\w+)/g, type: SymbolType.VARIABLE }
    ];
    
    for (const { pattern, type } of patterns) {
      let match;
      while ((match = pattern.exec(query)) !== null) {
        const name = match[1];
        const normalizedName = this.normalizeSymbolName(name);
        
        symbols.push({
          name,
          normalizedName,
          type,
          context: match[0]
        });
      }
    }
    
    // Also extract standalone identifiers
    const identifierPattern = /\b[a-zA-Z_][a-zA-Z0-9_]*\b/g;
    let match;
    while ((match = identifierPattern.exec(query)) !== null) {
      const name = match[0];
      
      // Skip if already found with context
      if (symbols.some(s => s.name === name)) continue;
      
      // Skip common words and reserved words
      if (this.isCommonWord(name)) continue;
      
      const normalizedName = this.normalizeSymbolName(name);
      symbols.push({
        name,
        normalizedName,
        context: name
      });
    }
    
    return symbols;
  }
  
  /**
   * Find equivalent symbols across languages
   */
  private async findEquivalentSymbols(
    querySymbol: { name: string; normalizedName: string; type?: SymbolType; context: string },
    context: SearchContext
  ): Promise<SearchHit[]> {
    const matches: SearchHit[] = [];
    
    // Direct name matches
    const directMatches = this.nameToSymbols.get(querySymbol.name) || new Set();
    for (const symbolId of directMatches) {
      const symbol = this.symbolIndex.get(symbolId);
      if (symbol) {
        matches.push(this.symbolToSearchHit(symbol, 1.0, 'exact_name'));
      }
    }
    
    // Normalized name matches
    const normalizedMatches = this.normalizedNameIndex.get(querySymbol.normalizedName) || new Set();
    for (const symbolId of normalizedMatches) {
      const symbol = this.symbolIndex.get(symbolId);
      if (symbol && !directMatches.has(symbolId)) {
        matches.push(this.symbolToSearchHit(symbol, 0.8, 'normalized_name'));
      }
    }
    
    // Semantic pattern matches
    if (querySymbol.type) {
      const semanticMatches = await this.findSemanticMatches(querySymbol, context);
      matches.push(...semanticMatches);
    }
    
    // Fuzzy name matches
    const fuzzyMatches = await this.findFuzzyNameMatches(querySymbol.name, context);
    matches.push(...fuzzyMatches);
    
    return matches;
  }
  
  /**
   * Find semantic matches based on symbol patterns
   */
  private async findSemanticMatches(
    querySymbol: { name: string; normalizedName: string; type?: SymbolType },
    context: SearchContext
  ): Promise<SearchHit[]> {
    const matches: SearchHit[] = [];
    
    if (!querySymbol.type) return matches;
    
    // Find symbols of the same type across languages
    for (const [symbolId, symbol] of this.symbolIndex.entries()) {
      if (symbol.type === querySymbol.type) {
        const similarity = this.calculateNameSimilarity(querySymbol.name, symbol.name);
        
        if (similarity > 0.6) {
          const hit = this.symbolToSearchHit(symbol, similarity * 0.7, 'semantic_type');
          matches.push(hit);
        }
      }
    }
    
    return matches;
  }
  
  /**
   * Find fuzzy name matches using parallel processing
   */
  private async findFuzzyNameMatches(queryName: string, context: SearchContext): Promise<SearchHit[]> {
    try {
      const allSymbolNames = Array.from(this.nameToSymbols.keys());
      
      if (allSymbolNames.length === 0) return [];
      
      // Use parallel processor for fuzzy matching
      const fuzzyResults = await globalParallelProcessor.submitTask(
        'fuzzy_matching' as any,
        {
          query: queryName,
          candidates: allSymbolNames,
          maxDistance: 2
        },
        1, // HIGH priority
        context,
        5000 // 5 second timeout
      );
      
      const matches: SearchHit[] = [];
      
      for (const match of fuzzyResults) {
        if (match.score > 0.7) {
          const symbolIds = this.nameToSymbols.get(match.text) || new Set();
          
          for (const symbolId of symbolIds) {
            const symbol = this.symbolIndex.get(symbolId);
            if (symbol) {
              const hit = this.symbolToSearchHit(symbol, match.score * 0.6, 'fuzzy_name');
              matches.push(hit);
            }
          }
        }
      }
      
      return matches;
      
    } catch (error) {
      console.warn('Fuzzy matching failed:', error);
      return [];
    }
  }
  
  /**
   * Rank cross-language results
   */
  private rankCrossLanguageResults(hits: SearchHit[], context: SearchContext): SearchHit[] {
    const queryLanguage = this.getQueryLanguage(context);
    
    return hits.sort((a, b) => {
      let scoreA = a.score || 0;
      let scoreB = b.score || 0;
      
      // Boost same-language results slightly
      if (a.language === queryLanguage) scoreA *= 1.1;
      if (b.language === queryLanguage) scoreB *= 1.1;
      
      // Boost exact matches
      if (a.why?.includes('exact_name')) scoreA *= 1.3;
      if (b.why?.includes('exact_name')) scoreB *= 1.3;
      
      // Boost semantic matches
      if (a.why?.includes('semantic_type')) scoreA *= 1.2;
      if (b.why?.includes('semantic_type')) scoreB *= 1.2;
      
      return scoreB - scoreA;
    });
  }
  
  /**
   * Convert universal symbol to search hit
   */
  private symbolToSearchHit(symbol: UniversalSymbol, score: number, reason: string): SearchHit {
    return {
      file: symbol.file,
      line: symbol.line,
      col: symbol.column,
      lang: symbol.language as any,
      snippet: this.generateSymbolSnippet(symbol),
      score,
      why: [reason as any],
      byte_offset: 0, // Would need to calculate
      span_len: symbol.name.length,
      symbol_kind: symbol.type as any,
      symbol_name: symbol.name,
      signature: symbol.signature,
      language: symbol.language,
      cross_language_match: true
    };
  }
  
  /**
   * Generate snippet for symbol
   */
  private generateSymbolSnippet(symbol: UniversalSymbol): string {
    let snippet = '';
    
    switch (symbol.type) {
      case SymbolType.FUNCTION:
      case SymbolType.METHOD:
        snippet = `${symbol.signature || `${symbol.name}(...)`}`;
        if (symbol.returnType) {
          snippet += ` -> ${symbol.returnType}`;
        }
        break;
        
      case SymbolType.CLASS:
        snippet = `class ${symbol.name}`;
        break;
        
      case SymbolType.INTERFACE:
        snippet = `interface ${symbol.name}`;
        break;
        
      case SymbolType.TYPE:
        snippet = `type ${symbol.name}`;
        break;
        
      case SymbolType.VARIABLE:
      case SymbolType.CONSTANT:
        snippet = symbol.name;
        break;
        
      default:
        snippet = symbol.name;
    }
    
    // Add language context
    snippet += ` (${symbol.language})`;
    
    return snippet;
  }
  
  /**
   * Normalize symbol name for cross-language comparison
   */
  private normalizeSymbolName(name: string): string {
    let normalized = name.toLowerCase();
    
    // Apply normalization rules
    for (const rule of this.normalizationRules) {
      if (!rule.language || rule.language === 'all') {
        normalized = normalized.replace(rule.pattern, rule.replacement);
      }
    }
    
    // Trim and clean up
    normalized = normalized.trim().replace(/^_+|_+$/g, '');
    
    return normalized;
  }
  
  /**
   * Calculate name similarity between two strings
   */
  private calculateNameSimilarity(name1: string, name2: string): number {
    if (name1 === name2) return 1.0;
    
    const normalized1 = this.normalizeSymbolName(name1);
    const normalized2 = this.normalizeSymbolName(name2);
    
    if (normalized1 === normalized2) return 0.9;
    
    // Simple edit distance-based similarity
    const distance = this.levenshteinDistance(normalized1, normalized2);
    const maxLength = Math.max(normalized1.length, normalized2.length);
    
    return maxLength > 0 ? 1.0 - (distance / maxLength) : 0;
  }
  
  /**
   * Calculate Levenshtein distance
   */
  private levenshteinDistance(str1: string, str2: string): number {
    const matrix: number[][] = [];
    
    for (let i = 0; i <= str2.length; i++) {
      matrix[i] = [i];
    }
    
    for (let j = 0; j <= str1.length; j++) {
      matrix[0][j] = j;
    }
    
    for (let i = 1; i <= str2.length; i++) {
      for (let j = 1; j <= str1.length; j++) {
        if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
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
   * Check if a name is a common word to skip
   */
  private isCommonWord(name: string): boolean {
    const commonWords = new Set([
      'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
      'with', 'by', 'from', 'up', 'about', 'into', 'over', 'after',
      'a', 'an', 'as', 'be', 'is', 'was', 'are', 'been', 'have', 'has',
      'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
      'if', 'then', 'else', 'when', 'where', 'why', 'how', 'what',
      'this', 'that', 'these', 'those', 'get', 'set', 'new', 'old',
      'first', 'last', 'next', 'prev', 'true', 'false', 'null', 'undefined'
    ]);
    
    return commonWords.has(name.toLowerCase()) || name.length < 2;
  }
  
  /**
   * Get query language from context
   */
  private getQueryLanguage(context: SearchContext): string {
    // Simple heuristic - could be enhanced with more sophisticated detection
    const query = context.query.toLowerCase();
    
    if (query.includes('function') || query.includes('const') || query.includes('let')) {
      return 'typescript';
    }
    if (query.includes('def') || query.includes('class') && query.includes(':')) {
      return 'python';
    }
    if (query.includes('fn') || query.includes('struct') || query.includes('impl')) {
      return 'rust';
    }
    if (query.includes('func') || query.includes('type') && query.includes('struct')) {
      return 'go';
    }
    
    return 'unknown';
  }
  
  /**
   * Generate cache key for cross-language resolution
   */
  private generateCacheKey(context: SearchContext): string {
    const key = JSON.stringify({
      query: context.query,
      mode: context.mode,
      k: context.k
    });
    
    return `cross_lang_${Buffer.from(key).toString('base64').substring(0, 32)}`;
  }
  
  /**
   * Index symbols from a file
   */
  async indexFileSymbols(filePath: string, content: string): Promise<void> {
    const span = LensTracer.createChildSpan('index_file_symbols');
    
    try {
      const language = this.getLanguageFromFile(filePath);
      if (!language) return;
      
      const config = this.languageConfigs.get(language);
      if (!config) return;
      
      const symbols = await this.extractSymbolsFromContent(content, filePath, language, config);
      
      for (const symbol of symbols) {
        this.indexSymbol(symbol);
      }
      
      span.setAttributes({
        success: true,
        file_path: filePath,
        language,
        symbols_count: symbols.length
      });
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
    } finally {
      span.end();
    }
  }
  
  /**
   * Extract symbols from file content
   */
  private async extractSymbolsFromContent(
    content: string, 
    filePath: string, 
    language: string,
    config: LanguageConfig
  ): Promise<UniversalSymbol[]> {
    const symbols: UniversalSymbol[] = [];
    const lines = content.split('\n');
    
    // Extract symbols using language-specific patterns
    for (const [symbolType, patterns] of Object.entries(config.symbolPatterns)) {
      for (const pattern of patterns) {
        let match;
        
        while ((match = pattern.exec(content)) !== null) {
          const name = match[1];
          if (!name) continue;
          
          // Find line and column
          const matchIndex = match.index || 0;
          const beforeMatch = content.substring(0, matchIndex);
          const line = beforeMatch.split('\n').length;
          const column = beforeMatch.length - beforeMatch.lastIndexOf('\n') - 1;
          
          const symbol: UniversalSymbol = {
            id: this.generateSymbolId(filePath, name, line),
            name,
            normalizedName: this.normalizeSymbolName(name),
            type: symbolType as SymbolType,
            language,
            file: filePath,
            line,
            column,
            signature: match[0],
            modifiers: [],
            scope: SymbolScope.MODULE, // Simplified
            crossReferences: [],
            semanticTags: []
          };
          
          symbols.push(symbol);
        }
      }
    }
    
    return symbols;
  }
  
  /**
   * Index a symbol
   */
  private indexSymbol(symbol: UniversalSymbol): void {
    // Store in main index
    this.symbolIndex.set(symbol.id, symbol);
    
    // Index by name
    if (!this.nameToSymbols.has(symbol.name)) {
      this.nameToSymbols.set(symbol.name, new Set());
    }
    this.nameToSymbols.get(symbol.name)!.add(symbol.id);
    
    // Index by normalized name
    if (!this.normalizedNameIndex.has(symbol.normalizedName)) {
      this.normalizedNameIndex.set(symbol.normalizedName, new Set());
    }
    this.normalizedNameIndex.get(symbol.normalizedName)!.add(symbol.id);
    
    // Initialize cross-reference graph
    if (!this.crossReferenceGraph.has(symbol.id)) {
      this.crossReferenceGraph.set(symbol.id, new Set());
    }
  }
  
  /**
   * Get language from file extension
   */
  private getLanguageFromFile(filePath: string): string | null {
    const extension = filePath.substring(filePath.lastIndexOf('.'));
    return this.extensionToLanguage.get(extension) || null;
  }
  
  /**
   * Generate unique symbol ID
   */
  private generateSymbolId(filePath: string, name: string, line: number): string {
    return `${filePath}:${name}:${line}`;
  }
  
  /**
   * Get cross-language resolution statistics
   */
  getStats(): {
    indexedSymbols: number;
    languagesSupported: number;
    resolutionStats: typeof this.resolutionStats;
  } {
    return {
      indexedSymbols: this.symbolIndex.size,
      languagesSupported: this.languageConfigs.size,
      resolutionStats: this.resolutionStats
    };
  }
  
  /**
   * Clear all indexed symbols
   */
  clearIndex(): void {
    this.symbolIndex.clear();
    this.nameToSymbols.clear();
    this.normalizedNameIndex.clear();
    this.crossReferenceGraph.clear();
    
    console.log('🧹 Cross-language symbol index cleared');
  }
  
  /**
   * Shutdown cross-language resolver
   */
  shutdown(): void {
    this.clearIndex();
    console.log('🛑 Cross-Language Resolver shutdown complete');
  }
}

// Global instance
export const globalCrossLanguageResolver = CrossLanguageResolver.getInstance();