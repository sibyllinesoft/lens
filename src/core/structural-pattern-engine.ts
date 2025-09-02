/**
 * Structural Pattern Engine for Precompiled Query Patterns
 * Phase B2 Enhancement: Cache compiled structural query patterns for faster matching
 * Supports complex structural queries with optimized pattern compilation
 */

import { LensTracer } from '../telemetry/tracer.js';
import type { SymbolDefinition, ASTNode, SymbolKind } from '../types/core.js';
import type { SupportedLanguage } from '../types/api.js';

export interface StructuralPattern {
  id: string;
  name: string;
  description: string;
  language: SupportedLanguage | 'all';
  pattern: RegExp;
  symbolKind?: SymbolKind;
  compiledAt: number;
  usageCount: number;
  averageMatchTime: number;
}

export interface PatternMatchResult {
  patternId: string;
  matches: PatternMatch[];
  matchTimeMs: number;
  symbolsFound: number;
}

export interface PatternMatch {
  text: string;
  startIndex: number;
  endIndex: number;
  line: number;
  col: number;
  captureGroups: string[];
  symbolInfo?: {
    name: string;
    kind: SymbolKind;
    signature?: string;
    scope?: string;
  };
}

export interface PatternCompileOptions {
  caseSensitive: boolean;
  multiline: boolean;
  global: boolean;
  dotAll: boolean;
  unicode: boolean;
}

export interface PatternEngineConfig {
  maxPatterns: number;
  enableStatistics: boolean;
  cacheCompilationResults: boolean;
  optimizeCommonPatterns: boolean;
  autoCleanupThreshold: number;
}

/**
 * High-performance structural pattern engine with compilation caching
 */
export class StructuralPatternEngine {
  private patterns = new Map<string, StructuralPattern>();
  private compilationCache = new Map<string, RegExp>();
  private usageStats = new Map<string, { count: number; totalTime: number }>();
  
  private config: PatternEngineConfig;

  constructor(config: Partial<PatternEngineConfig> = {}) {
    this.config = {
      maxPatterns: 1000,
      enableStatistics: true,
      cacheCompilationResults: true,
      optimizeCommonPatterns: true,
      autoCleanupThreshold: 10000,
      ...config
    };

    this.initializeBuiltinPatterns();
  }

  /**
   * Register a new structural pattern
   */
  registerPattern(
    id: string,
    name: string,
    patternSource: string,
    language: SupportedLanguage | 'all' = 'all',
    options: Partial<PatternCompileOptions> = {},
    symbolKind?: SymbolKind
  ): void {
    const span = LensTracer.createChildSpan('pattern_register', {
      'pattern.id': id,
      'pattern.language': language
    });

    try {
      if (this.patterns.has(id)) {
        throw new Error(`Pattern ${id} already exists`);
      }

      if (this.patterns.size >= this.config.maxPatterns) {
        this.performCleanup();
      }

      // Compile pattern with optimizations
      const compiledPattern = this.compilePattern(patternSource, options);
      
      const pattern: StructuralPattern = {
        id,
        name,
        description: `${name} pattern for ${language}`,
        language,
        pattern: compiledPattern,
        symbolKind,
        compiledAt: Date.now(),
        usageCount: 0,
        averageMatchTime: 0
      };

      this.patterns.set(id, pattern);
      
      if (this.config.cacheCompilationResults) {
        this.compilationCache.set(id, compiledPattern);
      }

      span.setAttributes({ 
        success: true, 
        'pattern.compiled': true,
        'pattern.cached': this.config.cacheCompilationResults
      });

      console.log(`📐 Registered structural pattern: ${name} (${id}) for ${language}`);

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Execute pattern matching against content
   */
  async executePattern(
    patternId: string,
    content: string,
    language: SupportedLanguage
  ): Promise<PatternMatchResult> {
    const span = LensTracer.createChildSpan('pattern_execute', {
      'pattern.id': patternId,
      'content.language': language,
      'content.length': content.length
    });

    try {
      const pattern = this.patterns.get(patternId);
      if (!pattern) {
        throw new Error(`Pattern ${patternId} not found`);
      }

      // Check language compatibility
      if (pattern.language !== 'all' && pattern.language !== language) {
        span.setAttributes({ 
          success: false, 
          reason: 'language_mismatch',
          'pattern.language': pattern.language,
          'content.language': language
        });
        return {
          patternId,
          matches: [],
          matchTimeMs: 0,
          symbolsFound: 0
        };
      }

      const matchStart = Date.now();
      const matches = this.findMatches(pattern.pattern, content);
      const matchTime = Date.now() - matchStart;

      // Update usage statistics
      if (this.config.enableStatistics) {
        this.updateUsageStats(patternId, matchTime);
        pattern.usageCount++;
        pattern.averageMatchTime = 
          (pattern.averageMatchTime * (pattern.usageCount - 1) + matchTime) / pattern.usageCount;
      }

      const result: PatternMatchResult = {
        patternId,
        matches,
        matchTimeMs: matchTime,
        symbolsFound: matches.length
      };

      span.setAttributes({
        success: true,
        'matches.count': matches.length,
        'match.time_ms': matchTime,
        'pattern.usage_count': pattern.usageCount
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
   * Execute multiple patterns in parallel
   */
  async executePatterns(
    patternIds: string[],
    content: string,
    language: SupportedLanguage
  ): Promise<PatternMatchResult[]> {
    const span = LensTracer.createChildSpan('pattern_execute_batch', {
      'patterns.count': patternIds.length,
      'content.language': language
    });

    try {
      const promises = patternIds.map(id => 
        this.executePattern(id, content, language)
      );

      const results = await Promise.all(promises);
      const totalMatches = results.reduce((sum, r) => sum + r.symbolsFound, 0);

      span.setAttributes({
        success: true,
        'results.total_matches': totalMatches,
        'patterns.executed': results.length
      });

      return results;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Find symbols using structural patterns
   */
  async findSymbols(
    content: string,
    language: SupportedLanguage,
    targetKinds?: SymbolKind[]
  ): Promise<SymbolDefinition[]> {
    const span = LensTracer.createChildSpan('pattern_find_symbols', {
      'content.language': language,
      'target.kinds': targetKinds?.join(',') || 'all'
    });

    try {
      const symbols: SymbolDefinition[] = [];
      
      // Filter patterns by language and symbol kind
      const applicablePatterns = Array.from(this.patterns.values()).filter(pattern => {
        if (pattern.language !== 'all' && pattern.language !== language) return false;
        if (targetKinds && pattern.symbolKind && !targetKinds.includes(pattern.symbolKind)) return false;
        return true;
      });

      // Execute patterns in parallel
      const patternIds = applicablePatterns.map(p => p.id);
      const results = await this.executePatterns(patternIds, content, language);

      // Convert matches to symbol definitions
      for (const result of results) {
        const pattern = this.patterns.get(result.patternId);
        if (!pattern) continue;

        for (const match of result.matches) {
          if (match.symbolInfo) {
            symbols.push({
              name: match.symbolInfo.name,
              kind: match.symbolInfo.kind,
              file_path: '', // Will be set by caller
              line: match.line,
              col: match.col,
              scope: match.symbolInfo.scope || 'unknown',
              signature: match.symbolInfo.signature
            });
          }
        }
      }

      span.setAttributes({
        success: true,
        'symbols.found': symbols.length,
        'patterns.used': applicablePatterns.length
      });

      return symbols;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Get pattern performance statistics
   */
  getPatternStats(): Array<{
    id: string;
    name: string;
    language: string;
    usageCount: number;
    averageMatchTime: number;
    totalTime: number;
    efficiency: number;
  }> {
    return Array.from(this.patterns.values()).map(pattern => {
      const stats = this.usageStats.get(pattern.id) || { count: 0, totalTime: 0 };
      return {
        id: pattern.id,
        name: pattern.name,
        language: pattern.language,
        usageCount: pattern.usageCount,
        averageMatchTime: pattern.averageMatchTime,
        totalTime: stats.totalTime,
        efficiency: stats.totalTime > 0 ? stats.count / stats.totalTime : 0
      };
    }).sort((a, b) => b.usageCount - a.usageCount);
  }

  /**
   * Optimize frequently used patterns
   */
  async optimizePatterns(): Promise<void> {
    if (!this.config.optimizeCommonPatterns) return;

    const span = LensTracer.createChildSpan('pattern_optimization');

    try {
      const stats = this.getPatternStats();
      const highUsagePatterns = stats.filter(s => s.usageCount > 100);

      for (const stat of highUsagePatterns) {
        const pattern = this.patterns.get(stat.id);
        if (!pattern) continue;

        // Recompile with aggressive optimizations for high-usage patterns
        const optimizedPattern = this.compilePattern(
          pattern.pattern.source,
          {
            caseSensitive: true,
            global: true,
            multiline: true,
            dotAll: false,
            unicode: false
          }
        );

        pattern.pattern = optimizedPattern;
        this.compilationCache.set(stat.id, optimizedPattern);
      }

      span.setAttributes({
        success: true,
        'patterns.optimized': highUsagePatterns.length
      });

      console.log(`🔧 Optimized ${highUsagePatterns.length} high-usage patterns`);

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
    } finally {
      span.end();
    }
  }

  /**
   * Clean up unused patterns
   */
  performCleanup(): void {
    const now = Date.now();
    const oldThreshold = now - (24 * 60 * 60 * 1000); // 24 hours
    
    let removedCount = 0;
    for (const [id, pattern] of this.patterns) {
      if (pattern.usageCount === 0 && pattern.compiledAt < oldThreshold) {
        this.patterns.delete(id);
        this.compilationCache.delete(id);
        this.usageStats.delete(id);
        removedCount++;
      }
    }

    if (removedCount > 0) {
      console.log(`🧹 Cleaned up ${removedCount} unused patterns`);
    }
  }

  /**
   * Get total number of registered patterns
   */
  getPatternCount(): number {
    return this.patterns.size;
  }

  /**
   * Clear all patterns
   */
  clear(): void {
    this.patterns.clear();
    this.compilationCache.clear();
    this.usageStats.clear();
    this.initializeBuiltinPatterns();
  }

  // Private methods

  private compilePattern(source: string, options: Partial<PatternCompileOptions>): RegExp {
    const flags = this.buildRegexFlags(options);
    try {
      return new RegExp(source, flags);
    } catch (error) {
      throw new Error(`Failed to compile pattern: ${source}. Error: ${error}`);
    }
  }

  private buildRegexFlags(options: Partial<PatternCompileOptions>): string {
    let flags = '';
    if (options.global !== false) flags += 'g';
    if (!options.caseSensitive) flags += 'i';
    if (options.multiline) flags += 'm';
    if (options.dotAll) flags += 's';
    if (options.unicode) flags += 'u';
    return flags;
  }

  private findMatches(pattern: RegExp, content: string): PatternMatch[] {
    const matches: PatternMatch[] = [];
    let match: RegExpExecArray | null;

    // Reset pattern to ensure we get all matches
    pattern.lastIndex = 0;

    while ((match = pattern.exec(content)) !== null) {
      const line = this.getLineNumber(content, match.index);
      const col = match.index - content.lastIndexOf('\n', match.index) - 1;
      
      const patternMatch: PatternMatch = {
        text: match[0],
        startIndex: match.index,
        endIndex: match.index + match[0].length,
        line,
        col,
        captureGroups: match.slice(1),
        symbolInfo: this.extractSymbolInfo(match)
      };

      matches.push(patternMatch);

      // Prevent infinite loops with zero-length matches
      if (match[0].length === 0) {
        pattern.lastIndex++;
      }
    }

    return matches;
  }

  private extractSymbolInfo(match: RegExpExecArray): PatternMatch['symbolInfo'] {
    // Extract symbol information from regex capture groups
    if (match.length < 2) return undefined;

    const name = match[1]?.trim();
    if (!name) return undefined;

    // Determine symbol kind from context
    let kind: SymbolKind = 'variable';
    const fullMatch = match[0].toLowerCase();
    
    if (fullMatch.includes('function') || fullMatch.includes('def ')) {
      kind = 'function';
    } else if (fullMatch.includes('class')) {
      kind = 'class';
    } else if (fullMatch.includes('interface')) {
      kind = 'interface';
    } else if (fullMatch.includes('type ')) {
      kind = 'type';
    } else if (fullMatch.includes('const ') || fullMatch.includes('let ') || fullMatch.includes('var ')) {
      kind = 'variable';
    }

    return {
      name,
      kind,
      signature: match[0].trim(),
      scope: 'detected'
    };
  }

  private getLineNumber(content: string, index: number): number {
    return content.substring(0, index).split('\n').length;
  }

  private updateUsageStats(patternId: string, matchTime: number): void {
    const existing = this.usageStats.get(patternId) || { count: 0, totalTime: 0 };
    existing.count++;
    existing.totalTime += matchTime;
    this.usageStats.set(patternId, existing);
  }

  private initializeBuiltinPatterns(): void {
    // TypeScript/JavaScript patterns
    this.registerBuiltinPattern(
      'ts-function-declarations',
      'TypeScript Function Declarations',
      /(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*(?:<[^>]*>)?\s*\([^)]*\)(?:\s*:\s*[^{]+)?\s*\{/g,
      'typescript',
      'function'
    );

    this.registerBuiltinPattern(
      'ts-arrow-functions',
      'TypeScript Arrow Functions',
      /(?:const|let|var)\s+(\w+)\s*=\s*(?:\([^)]*\)\s*=>\s*\{|\([^)]*\)\s*=>)/g,
      'typescript',
      'function'
    );

    this.registerBuiltinPattern(
      'ts-class-declarations',
      'TypeScript Class Declarations',
      /(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s*<[^>]*>)?(?:\s+extends\s+[\w<>,\s]+)?(?:\s+implements\s+[\w<>,\s]+)?\s*\{/g,
      'typescript',
      'class'
    );

    this.registerBuiltinPattern(
      'ts-interface-declarations',
      'TypeScript Interface Declarations',
      /(?:export\s+)?interface\s+(\w+)(?:\s*<[^>]*>)?(?:\s+extends\s+[\w<>,\s]+)?\s*\{/g,
      'typescript',
      'interface'
    );

    this.registerBuiltinPattern(
      'ts-type-aliases',
      'TypeScript Type Aliases',
      /(?:export\s+)?type\s+(\w+)(?:\s*<[^>]*>)?\s*=\s*[^;]+;?/g,
      'typescript',
      'type'
    );

    // Python patterns
    this.registerBuiltinPattern(
      'py-function-definitions',
      'Python Function Definitions',
      /(?:async\s+)?def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:/g,
      'python',
      'function'
    );

    this.registerBuiltinPattern(
      'py-class-definitions',
      'Python Class Definitions',
      /class\s+(\w+)(?:\([^)]*\))?\s*:/g,
      'python',
      'class'
    );

    // Generic patterns for all languages
    this.registerBuiltinPattern(
      'generic-imports',
      'Generic Import Statements',
      /(?:import|from|#include|use)\s+([^\s;]+)/g,
      'all',
      'constant'
    );

    this.registerBuiltinPattern(
      'generic-comments',
      'Generic Comments',
      /(?:\/\/|#|\/\*|\*\/|\*|<!--).*$/gm,
      'all'
    );
  }

  private registerBuiltinPattern(
    id: string,
    name: string,
    pattern: RegExp,
    language: SupportedLanguage | 'all',
    symbolKind?: SymbolKind
  ): void {
    try {
      const structuralPattern: StructuralPattern = {
        id,
        name,
        description: `Built-in ${name} pattern`,
        language,
        pattern,
        symbolKind,
        compiledAt: Date.now(),
        usageCount: 0,
        averageMatchTime: 0
      };

      this.patterns.set(id, structuralPattern);
      
      if (this.config.cacheCompilationResults) {
        this.compilationCache.set(id, pattern);
      }
    } catch (error) {
      console.warn(`Failed to register builtin pattern ${id}:`, error);
    }
  }
}

// Export pattern presets for common use cases
export const PATTERN_PRESETS = {
  // TypeScript comprehensive symbol extraction
  typescript_symbols: [
    'ts-function-declarations',
    'ts-arrow-functions', 
    'ts-class-declarations',
    'ts-interface-declarations',
    'ts-type-aliases'
  ],
  
  // Python symbol extraction
  python_symbols: [
    'py-function-definitions',
    'py-class-definitions'
  ],
  
  // Cross-language patterns
  generic_patterns: [
    'generic-imports',
    'generic-comments'
  ]
} as const;