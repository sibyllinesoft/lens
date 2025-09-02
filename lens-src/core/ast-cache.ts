/**
 * Tree-sitter AST Cache for Top-N Hot Files
 * Implements incremental parsing with LRU eviction
 * Phase 2 Enhancement: Cache tree-sitter AST for frequently accessed TypeScript files
 */

import { LRUCache } from 'lru-cache';
import * as crypto from 'crypto';

export interface CachedAST {
  fileHash: string;
  parseTime: number;
  lastAccessed: number;
  language: 'typescript' | 'javascript' | 'python';
  symbolCount: number;
  // In production, this would be a tree-sitter Tree object
  mockAST: {
    functions: Array<{ name: string; line: number; col: number; signature?: string }>;
    classes: Array<{ name: string; line: number; col: number; extends?: string; implements?: string[] }>;
    interfaces: Array<{ name: string; line: number; col: number; extends?: string[] }>;
    types: Array<{ name: string; line: number; col: number; definition: string }>;
    imports: Array<{ module: string; line: number; col: number; imports: string[] }>;
  };
}

export class ASTCache {
  private cache: LRUCache<string, CachedAST>;
  private fileHashCache = new Map<string, string>();
  private hitCount = 0;
  private missCount = 0;
  
  constructor(maxFiles: number = 50) {
    this.cache = new LRUCache<string, CachedAST>({
      max: maxFiles,
      ttl: 1000 * 60 * 30, // 30 minutes
    });
  }

  /**
   * Get AST for a file, parsing if not cached or if file changed
   */
  async getAST(filePath: string, content: string, language: CachedAST['language']): Promise<CachedAST> {
    const fileHash = this.calculateContentHash(content);
    const cached = this.cache.get(filePath);

    // Return cached AST if content hasn't changed
    if (cached && cached.fileHash === fileHash) {
      cached.lastAccessed = Date.now();
      this.hitCount++;
      console.log(`📋 AST cache HIT for ${filePath} (${this.getStats().hitRate}% hit rate)`);
      return cached;
    }

    // Parse file and cache result
    this.missCount++;
    console.log(`📋 AST cache MISS for ${filePath} - parsing...`);
    
    const parseStart = Date.now();
    const ast = await this.parseFile(content, language, filePath);
    const parseTime = Date.now() - parseStart;

    const cachedAST: CachedAST = {
      fileHash,
      parseTime,
      lastAccessed: Date.now(),
      language,
      symbolCount: ast.functions.length + ast.classes.length + ast.interfaces.length + ast.types.length,
      mockAST: ast,
    };

    this.cache.set(filePath, cachedAST);
    console.log(`⚡ Parsed ${filePath} in ${parseTime}ms - found ${cachedAST.symbolCount} symbols`);
    
    return cachedAST;
  }

  /**
   * Get cache statistics for coverage reporting
   */
  getStats() {
    const total = this.hitCount + this.missCount;
    return {
      cacheSize: this.cache.size,
      hitCount: this.hitCount,
      missCount: this.missCount,
      hitRate: total > 0 ? Math.round((this.hitCount / total) * 100) : 0,
      totalRequests: total,
    };
  }

  /**
   * Calculate coverage percentage for TypeScript files
   */
  getCoverageStats(totalTSFiles: number) {
    const cachedTSFiles = Array.from(this.cache.values()).filter(ast => ast.language === 'typescript').length;
    const coverage = totalTSFiles > 0 ? Math.round((cachedTSFiles / totalTSFiles) * 100) : 0;
    
    return {
      totalTSFiles,
      cachedTSFiles,
      coveragePercentage: coverage,
      symbolsCached: Array.from(this.cache.values()).reduce((sum, ast) => sum + ast.symbolCount, 0),
    };
  }

  /**
   * Force refresh AST for a file (useful when file is known to have changed)
   */
  async refreshAST(filePath: string, content: string, language: CachedAST['language']): Promise<CachedAST> {
    this.cache.delete(filePath);
    return this.getAST(filePath, content, language);
  }

  /**
   * Clear cache
   */
  clear() {
    this.cache.clear();
    this.fileHashCache.clear();
    this.hitCount = 0;
    this.missCount = 0;
  }

  /**
   * Calculate hash of file content for change detection
   */
  private calculateContentHash(content: string): string {
    return crypto.createHash('sha256').update(content).digest('hex').substring(0, 16);
  }

  /**
   * Parse file content into structured AST representation
   * In production, this would use actual tree-sitter parsing
   */
  private async parseFile(
    content: string, 
    language: CachedAST['language'], 
    filePath: string
  ): Promise<CachedAST['mockAST']> {
    const lines = content.split('\n');
    const ast: CachedAST['mockAST'] = {
      functions: [],
      classes: [],
      interfaces: [],
      types: [],
      imports: [],
    };

    // TypeScript/JavaScript parsing
    if (language === 'typescript' || language === 'javascript') {
      await this.parseTypeScriptFile(content, lines, ast);
    }
    // Python parsing
    else if (language === 'python') {
      await this.parsePythonFile(content, lines, ast);
    }

    return ast;
  }

  /**
   * Parse TypeScript/JavaScript file using enhanced patterns
   */
  private async parseTypeScriptFile(content: string, lines: string[], ast: CachedAST['mockAST']) {
    // Enhanced TypeScript patterns for better symbol extraction
    
    // 1. Function declarations (including exported and async)
    const functionPattern = /^(\s*)(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*(<[^>]*>)?\s*\([^)]*\)(?:\s*:\s*[^{]+)?\s*\{/gm;
    let match: RegExpExecArray | null;
    
    while ((match = functionPattern.exec(content)) !== null) {
      const lineNum = this.getLineNumber(content, match.index || 0);
      const colNum = (match.index || 0) - content.lastIndexOf('\n', match.index || 0) - 1;
      
      ast.functions.push({
        name: match[2] || '',
        line: lineNum,
        col: colNum,
        signature: match[0]?.trim() || '',
      });
    }

    // 2. Class declarations with inheritance
    const classPattern = /^(\s*)(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s*<[^>]*>)?(?:\s+(extends|implements)\s+([\w<>,\s]+))?\s*\{/gm;
    classPattern.lastIndex = 0;
    
    while ((match = classPattern.exec(content)) !== null) {
      const lineNum = this.getLineNumber(content, match.index || 0);
      const colNum = (match.index || 0) - content.lastIndexOf('\n', match.index || 0) - 1;
      
      const classData: {
        name: string;
        line: number;
        col: number;
        extends?: string;
        implements?: string[];
      } = {
        name: match[2] || '',
        line: lineNum,
        col: colNum,
      };
      
      if (match[3] === 'extends' && match[4]) {
        classData.extends = match[4].trim();
      } else if (match[3] === 'implements' && match[4]) {
        classData.implements = match[4].split(',').map(s => s.trim());
      }
      
      ast.classes.push(classData);
    }

    // 3. Interface declarations with extends
    const interfacePattern = /^(\s*)(?:export\s+)?interface\s+(\w+)(?:\s*<[^>]*>)?(?:\s+extends\s+([\w<>,\s]+))?\s*\{/gm;
    interfacePattern.lastIndex = 0;
    
    while ((match = interfacePattern.exec(content)) !== null) {
      const lineNum = this.getLineNumber(content, match.index || 0);
      const colNum = (match.index || 0) - content.lastIndexOf('\n', match.index || 0) - 1;
      
      const interfaceData: {
        name: string;
        line: number;
        col: number;
        extends?: string[];
      } = {
        name: match[2] || '',
        line: lineNum,
        col: colNum,
      };
      
      if (match[3]) {
        interfaceData.extends = match[3].split(',').map(s => s.trim());
      }
      
      ast.interfaces.push(interfaceData);
    }

    // 4. Type definitions
    const typePattern = /^(\s*)(?:export\s+)?type\s+(\w+)(?:\s*<[^>]*>)?\s*=\s*(.+);?$/gm;
    typePattern.lastIndex = 0;
    
    while ((match = typePattern.exec(content)) !== null) {
      const lineNum = this.getLineNumber(content, match.index || 0);
      const colNum = (match.index || 0) - content.lastIndexOf('\n', match.index || 0) - 1;
      
      ast.types.push({
        name: match[2] || '',
        line: lineNum,
        col: colNum,
        definition: match[3]?.trim() || '',
      });
    }

    // 5. Import statements
    const importPattern = /^(\s*)import\s+(?:(\w+)(?:\s*,\s*)?)?(?:\{([^}]+)\})?\s+from\s+['"]([^'"]+)['"]/gm;
    importPattern.lastIndex = 0;
    
    while ((match = importPattern.exec(content)) !== null) {
      const lineNum = this.getLineNumber(content, match.index || 0);
      const colNum = (match.index || 0) - content.lastIndexOf('\n', match.index || 0) - 1;
      
      const imports: string[] = [];
      if (match[2]) imports.push(match[2]); // Default import
      if (match[3]) imports.push(...match[3].split(',').map(s => s.trim())); // Named imports
      
      ast.imports.push({
        module: match[4] || '',
        line: lineNum,
        col: colNum,
        imports,
      });
    }
  }

  /**
   * Parse Python file (existing functionality)
   */
  private async parsePythonFile(content: string, lines: string[], ast: CachedAST['mockAST']) {
    // Function definitions
    const functionPattern = /^(\s*)(?:async\s+)?def\s+(\w+)\s*\([^)]*\)\s*(?:->.*?)?:/gm;
    let match: RegExpExecArray | null;
    
    while ((match = functionPattern.exec(content)) !== null) {
      const lineNum = this.getLineNumber(content, match.index || 0);
      const colNum = (match.index || 0) - content.lastIndexOf('\n', match.index || 0) - 1;
      
      ast.functions.push({
        name: match[2] || '',
        line: lineNum,
        col: colNum,
        signature: match[0]?.trim() || '',
      });
    }

    // Class definitions
    const classPattern = /^(\s*)class\s+(\w+)(?:\([^)]*\))?\s*:/gm;
    classPattern.lastIndex = 0;
    
    while ((match = classPattern.exec(content)) !== null) {
      const lineNum = this.getLineNumber(content, match.index || 0);
      const colNum = (match.index || 0) - content.lastIndexOf('\n', match.index || 0) - 1;
      
      ast.classes.push({
        name: match[2] || '',
        line: lineNum,
        col: colNum,
      });
    }
  }

  /**
   * Get line number from character index in content
   */
  private getLineNumber(content: string, index: number): number {
    return content.substring(0, index).split('\n').length;
  }
}