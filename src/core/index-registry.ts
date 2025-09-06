/**
 * IndexRegistry - Single source of truth for discovering and managing index shards
 * Maps repo_ref/repo_sha to shard paths and provides IndexReader instances
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import { LRUCache } from 'lru-cache';
import { LensTracer } from '../telemetry/tracer.js';

/**
 * Valid enum values for match reasons according to the API schema
 */
export type ValidMatchReason = 'exact' | 'fuzzy' | 'symbol' | 'struct' | 'structural' | 'semantic' | 'subtoken';

/**
 * Map any reason string to a valid enum value
 */
function mapToValidReason(reason: string): ValidMatchReason {
  // Handle fuzzy variants (fuzzy_1, fuzzy_2, etc.)
  if (reason.startsWith('fuzzy')) return 'fuzzy';
  
  // Map prefix/suffix to exact matches
  if (reason === 'prefix' || reason === 'suffix') return 'exact';
  
  // Map word_exact to exact
  if (reason === 'word_exact') return 'exact';
  
  // Map AST pattern types to structural
  if (reason === 'function_def' || reason === 'class_def' || reason === 'async_def' || 
      reason === 'import' || reason === 'decorator' || reason === 'try_except' || 
      reason === 'for_loop' || reason === 'if_statement') {
    return 'structural';
  }
  
  // Map pattern descriptions to structural  
  if (reason.includes('typescript') || reason.includes('python') || reason.includes('function') ||
      reason.includes('class') || reason.includes('interface') || reason.includes('definition') ||
      reason.includes('fallback')) {
    return 'structural';
  }
  
  // Map known valid values
  const validReasons = ['exact', 'fuzzy', 'symbol', 'struct', 'structural', 'semantic', 'subtoken'];
  if (validReasons.includes(reason as ValidMatchReason)) {
    return reason as ValidMatchReason;
  }
  
  // Default fallback for any unmapped values
  return 'structural';
}

export interface IndexReader {
  repoSha: string;
  repoRef: string;
  shardPaths: string[];
  version: string;
  languages: string[];
  
  /**
   * Search lexical index with fuzzy matching and subtokens
   */
  searchLexical(params: {
    q: string;
    fuzzy: number;
    subtokens: boolean;
    k: number;
  }): Promise<LexicalResult[]>;

  /**
   * Search structural patterns using AST-based matching
   */
  searchStructural(params: {
    q: string;
    k: number;
    patterns?: string[];
  }): Promise<StructuralResult[]>;

  /**
   * Get list of files in the index
   */
  getFileList(): Promise<string[]>;

  /**
   * Close reader and free resources
   */
  close(): Promise<void>;
}

export interface StructuralResult {
  file: string;
  line: number;
  col: number;
  lang: string;
  snippet: string;
  score: number;
  why: string[];
  match_reasons?: ValidMatchReason[] | undefined; // Add match_reasons field
  byte_offset?: number;
  span_len?: number;
  pattern_type: 'function_def' | 'class_def' | 'import' | 'async_def' | 'decorator' | 'try_except' | 'for_loop' | 'if_statement';
  symbol_name?: string | undefined;
  signature?: string | undefined;
}

export interface LexicalResult {
  file: string;
  line: number;
  col: number;
  lang: string;
  snippet: string;
  score: number;
  why: string[];
  byte_offset?: number;
  span_len?: number;
}

export interface RegistryStats {
  totalRepos: number;
  loadedRepos: number;
  shardPaths: number;
  memoryUsageMB: number;
  cacheHitRate: number;
}

export interface RepoManifest {
  repo_sha: string;
  repo_ref: string;
  version: string;
  api_version: string;
  index_version: string;
  policy_version: string;
  languages: string[];
  shard_paths: string[];
  created_at: Date;
  file_count: number;
}

export class IndexRegistry {
  private manifest = new Map<string, RepoManifest>();
  private readerCache: LRUCache<string, IndexReader>;
  private indexRoot: string;
  private refreshPromise: Promise<void> | null = null;

  constructor(indexRoot: string = './indexed-content', maxReaders: number = 10) {
    this.indexRoot = path.resolve(indexRoot);
    this.readerCache = new LRUCache<string, IndexReader>({
      max: maxReaders,
      dispose: async (reader: IndexReader) => {
        try {
          await reader.close();
        } catch (error) {
          console.warn('Error closing index reader:', error);
        }
      },
    });
  }

  /**
   * Scan disk or fetch manifest to discover available repositories
   */
  async refresh(): Promise<void> {
    // Prevent concurrent refreshes
    if (this.refreshPromise) {
      return this.refreshPromise;
    }

    const span = LensTracer.createChildSpan('index_registry_refresh');
    
    this.refreshPromise = this.doRefresh(span);
    
    try {
      await this.refreshPromise;
    } finally {
      this.refreshPromise = null;
      span.end();
    }
  }

  private async doRefresh(span: any): Promise<void> {
    try {
      console.log(`🔍 Scanning for index shards in ${this.indexRoot}`);

      // Check if index root exists
      try {
        await fs.access(this.indexRoot);
      } catch (error) {
        throw new Error(`Index root does not exist: ${this.indexRoot}`);
      }

      // Read proper manifest files from the index root
      const manifests = await this.loadManifestFiles();
      
      // Clear and rebuild manifest
      this.manifest.clear();
      for (const manifest of manifests) {
        this.manifest.set(manifest.repo_sha, manifest);
      }

      span.setAttributes({
        success: true,
        repos_discovered: this.manifest.size,
        index_root: this.indexRoot,
      });

      console.log(`📂 Discovered ${this.manifest.size} repositories in index`);

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw new Error(`Failed to refresh index registry: ${errorMsg}`);
    }
  }

  /**
   * Load manifest files from the index root directory
   */
  private async loadManifestFiles(): Promise<RepoManifest[]> {
    const manifests: RepoManifest[] = [];
    
    try {
      const files = await fs.readdir(this.indexRoot);
      const manifestFiles = files.filter(f => f.endsWith('.manifest.json'));
      
      for (const manifestFile of manifestFiles) {
        try {
          const manifestPath = path.join(this.indexRoot, manifestFile);
          const manifestData = await fs.readFile(manifestPath, 'utf-8');
          const manifest = JSON.parse(manifestData) as RepoManifest;
          
          // Validate manifest structure
          if (manifest.repo_sha && manifest.repo_ref && manifest.shard_paths) {
            // Ensure version fields exist, defaulting to v1 for compatibility
            manifest.api_version = manifest.api_version || 'v1';
            manifest.index_version = manifest.index_version || 'v1';
            manifest.policy_version = manifest.policy_version || 'v1';
            manifests.push(manifest);
          } else {
            console.warn(`Invalid manifest file: ${manifestFile}`);
          }
        } catch (error) {
          console.warn(`Failed to load manifest ${manifestFile}:`, error);
        }
      }
      
      if (manifests.length === 0) {
        console.log('No valid manifest files found in index root');
      }
      
    } catch (error) {
      console.warn('Failed to scan for manifest files:', error);
    }
    
    return manifests;
  }

  /**
   * Get IndexReader for a specific repository
   */
  getReader(repoSha: string): IndexReader {
    // Check cache first
    const cached = this.readerCache.get(repoSha);
    if (cached) {
      return cached;
    }

    // Check if we have this repo in our manifest
    const manifest = this.manifest.get(repoSha);
    if (!manifest) {
      throw new Error(`Repository not found in index: ${repoSha}`);
    }

    // Create new reader
    const reader = new FileBasedIndexReader(manifest);
    this.readerCache.set(repoSha, reader);
    
    return reader;
  }

  /**
   * Check if repository exists in the index
   */
  hasRepo(repoSha: string): boolean {
    return this.manifest.has(repoSha);
  }

  /**
   * Get registry statistics
   */
  stats(): RegistryStats {
    return {
      totalRepos: this.manifest.size,
      loadedRepos: this.readerCache.size,
      shardPaths: Array.from(this.manifest.values()).reduce((sum, m) => sum + m.shard_paths.length, 0),
      memoryUsageMB: this.readerCache.size * 0.1, // Rough estimate
      cacheHitRate: 0.9, // Would be calculated from actual metrics
    };
  }

  /**
   * Resolve repo_ref to repo_sha via manifest
   */
  async resolveRef(repoRef: string): Promise<string | null> {
    // Simple lookup - in production this might query a separate service
    for (const manifest of this.manifest.values()) {
      if (manifest.repo_ref === repoRef) {
        return manifest.repo_sha;
      }
    }
    return null;
  }

  /**
   * Get all available repository manifests
   */
  getManifests(): RepoManifest[] {
    return Array.from(this.manifest.values());
  }

  /**
   * Shutdown registry and close all readers
   */
  async shutdown(): Promise<void> {
    const span = LensTracer.createChildSpan('index_registry_shutdown');

    try {
      console.log('Shutting down IndexRegistry...');
      
      // Clear cache (will dispose all readers)
      this.readerCache.clear();
      
      // Clear manifest
      this.manifest.clear();

      span.setAttributes({ success: true });
      console.log('IndexRegistry shut down successfully');

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw new Error(`Failed to shutdown IndexRegistry: ${errorMsg}`);
    } finally {
      span.end();
    }
  }
}

/**
 * File-based IndexReader implementation
 * This reads directly from the indexed files on disk
 */
class FileBasedIndexReader implements IndexReader {
  readonly repoSha: string;
  readonly repoRef: string;
  readonly shardPaths: string[];
  readonly version: string;
  readonly languages: string[];

  private fileCache = new Map<string, string>();
  private synonymMap: Map<string, string[]>;

  constructor(manifest: RepoManifest) {
    this.repoSha = manifest.repo_sha;
    this.repoRef = manifest.repo_ref;
    this.shardPaths = manifest.shard_paths;
    this.version = manifest.version;
    this.languages = manifest.languages;
    
    // Initialize synonym map for common programming terms
    this.synonymMap = new Map([
      ['retry', ['backoff', 'rerun', 'reattempt']],
      ['backoff', ['retry', 'delay', 'wait']],
      ['timeout', ['deadline', 'expire', 'ttl']],
      ['deadline', ['timeout', 'expire', 'cutoff']],
      ['async', ['asynchronous', 'promise', 'await']],
      ['asynchronous', ['async', 'promise', 'await']],
      ['promise', ['async', 'await', 'future']],
      ['await', ['async', 'promise', 'wait']],
      ['config', ['configuration', 'settings', 'options']],
      ['configuration', ['config', 'settings', 'setup']],
      ['settings', ['config', 'configuration', 'options']],
      ['user', ['customer', 'account', 'profile']],
      ['customer', ['user', 'client', 'account']],
      ['error', ['exception', 'fault', 'failure']],
      ['exception', ['error', 'fault', 'throw']],
      ['cache', ['store', 'buffer', 'memory']],
      ['buffer', ['cache', 'storage', 'temp']],
      ['validate', ['check', 'verify', 'confirm']],
      ['verify', ['validate', 'check', 'confirm']],
    ]);
  }

  async searchLexical(params: {
    q: string;
    fuzzy: number;
    subtokens: boolean;
    k: number;
  }): Promise<LexicalResult[]> {
    const span = LensTracer.createChildSpan('file_based_search_lexical', {
      query: params.q,
      fuzzy: params.fuzzy,
      subtokens: params.subtokens,
      k: params.k,
    });

    try {
      const results: LexicalResult[] = [];
      const query = params.q.toLowerCase();
      
      // Generate search terms including synonyms and subtokens
      const searchTerms = this.generateSearchTerms(params.q, params.subtokens);
      
      console.log(`🔍 Searching ${this.shardPaths.length} files for: "${params.q}" + variants`);
      console.log(`📝 Search terms: ${searchTerms.map(t => t.term).join(', ')}`);

      // Performance optimization: limit number of files to process for speed
      const maxFilesToProcess = Math.min(this.shardPaths.length, 100); // Limit to first 100 files
      const filesToProcess = this.shardPaths.slice(0, maxFilesToProcess);
      
      // Early termination: track results and exit early if we have enough good matches
      const targetResults = params.k * 3; // Get 3x more than needed for better ranking

      // Search through each file
      for (let fileIndex = 0; fileIndex < filesToProcess.length; fileIndex++) {
        const filePath = filesToProcess[fileIndex];
        
        // Early exit if we have enough high-quality results
        if (results.length >= targetResults && results.some(r => r.score > 0.8)) {
          console.log(`⚡ Early exit after processing ${fileIndex + 1} files with ${results.length} results`);
          break;
        }
        try {
          const content = await this.getFileContent(filePath!);
          const lines = content.split('\n');
          const pathBoost = this.calculatePathBoost(filePath!);
          
          // Performance optimization: limit lines processed per file
          const maxLinesToProcess = Math.min(lines.length, 500); // Process max 500 lines per file
          
          // Search line by line for all search terms
          for (let lineNum = 0; lineNum < maxLinesToProcess; lineNum++) {
            const line = lines[lineNum];
            if (!line || line.length > 200) continue; // Skip very long lines
            
            // Quick check if line contains any of our search terms (performance optimization)
            const lineContainsQuery = searchTerms.some(term => 
              line.toLowerCase().includes(term.term)
            );
            if (!lineContainsQuery) continue;
            
            const lineWords = this.tokenizeLine(line);
            
            // Check each search term against this line
            for (const searchTerm of searchTerms) {
              const matches = this.findMatches(line, lineWords, searchTerm, params.fuzzy);
              if (matches.length === 0) continue; // Skip if no matches
              
              for (const match of matches) {
                const result: LexicalResult = {
                  file: path.relative(this.shardPaths[0]?.split('/').slice(0, -1).join('/') || '.', filePath!),
                  line: lineNum + 1,
                  col: match.col,
                  lang: this.getLanguageFromPath(filePath!),
                  snippet: line.trim(),
                  score: match.score * pathBoost * searchTerm.boost,
                  why: match.reasons.map(mapToValidReason), // Fix: ensure valid enum values
                  byte_offset: this.calculateByteOffset(lines, lineNum, match.col),
                  span_len: match.span_len,
                };
                
                results.push(result);
              }
            }
          }
        } catch (error) {
          console.warn(`Error searching file ${filePath}:`, error);
          continue;
        }
      }

      // Deduplicate and merge results with same location
      const dedupedResults = this.deduplicateResults(results);
      
      // Sort by relevance score
      dedupedResults.sort((a, b) => b.score - a.score);
      
      const finalResults = dedupedResults.slice(0, params.k);

      span.setAttributes({
        success: true,
        results_found: finalResults.length,
        files_searched: this.shardPaths.length,
        search_terms_count: searchTerms.length,
      });

      console.log(`✅ Found ${finalResults.length} matches for "${params.q}"`);
      return finalResults;

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw new Error(`Lexical search failed: ${errorMsg}`);
    } finally {
      span.end();
    }
  }

  private async getFileContent(filePath: string): Promise<string> {
    // Simple caching to avoid re-reading files
    if (this.fileCache.has(filePath)) {
      return this.fileCache.get(filePath)!;
    }

    const content = await fs.readFile(filePath, 'utf8');
    this.fileCache.set(filePath, content);
    return content;
  }

  private getLanguageFromPath(filePath: string): string {
    const ext = path.extname(filePath);
    switch (ext) {
      case '.py': return 'python';
      case '.js': return 'javascript';
      case '.ts': return 'typescript';
      case '.java': return 'java';
      case '.go': return 'go';
      case '.rs': return 'rust';
      case '.cpp':
      case '.cc':
      case '.cxx': return 'cpp';
      case '.c': return 'c';
      case '.h': return 'c';
      case '.hpp': return 'cpp';
      default: return 'unknown';
    }
  }

  private calculateByteOffset(lines: string[], lineNum: number, colIndex: number): number {
    // Calculate rough byte offset (simplified)
    let offset = 0;
    for (let i = 0; i < lineNum; i++) {
      const line = lines[i];
      if (line !== undefined) {
        offset += Buffer.byteLength(line, 'utf8') + 1; // +1 for newline
      }
    }
    const targetLine = lines[lineNum];
    if (targetLine !== undefined) {
      offset += Buffer.byteLength(targetLine.substring(0, colIndex), 'utf8');
    }
    return offset;
  }

  async searchStructural(params: {
    q: string;
    k: number;
    patterns?: string[];
  }): Promise<StructuralResult[]> {
    const span = LensTracer.createChildSpan('file_based_search_structural', {
      query: params.q,
      k: params.k,
      patterns: params.patterns?.join(',') || 'auto',
    });

    try {
      const results: StructuralResult[] = [];
      const query = params.q.toLowerCase();
      
      // Generate structural patterns based on query
      const patterns = this.generateStructuralPatterns(params.q, params.patterns);
      
      console.log(`🔍 Structural search for: "${params.q}" using ${patterns.length} patterns`);

      // Search through TypeScript, Python, and JavaScript files for structural patterns
      for (const filePath of this.shardPaths) {
        // Support TypeScript (.ts), Python (.py), and JavaScript (.js) files
        const isTypeScript = filePath.endsWith('.ts') && !filePath.endsWith('.d.ts');
        const isPython = filePath.endsWith('.py');
        const isJavaScript = filePath.endsWith('.js');
        
        if (!isTypeScript && !isPython && !isJavaScript) continue;
        
        try {
          const content = await this.getFileContent(filePath);
          const lines = content.split('\n');
          const pathBoost = this.calculatePathBoost(filePath);
          
          // Apply each structural pattern - filter by language compatibility
          for (const pattern of patterns) {
            // Skip Python-specific patterns for TypeScript/JS files and vice versa
            if (isTypeScript || isJavaScript) {
              if (pattern.description.includes('python') || pattern.type === 'async_def') continue;
            }
            if (isPython) {
              if (pattern.description.includes('typescript') || pattern.description.includes('javascript')) continue;
            }
            
            const matches = this.findStructuralMatches(content, lines, pattern, filePath, query, isTypeScript);
            
            for (const match of matches) {
              const result: StructuralResult = {
                file: path.relative(this.shardPaths[0]?.split('/').slice(0, -1).join('/') || '.', filePath),
                line: match.line,
                col: Math.max(match.col, 0), // Fix: ensure col >= 0
                lang: isTypeScript ? 'typescript' : isPython ? 'python' : 'javascript',
                snippet: match.snippet,
                score: Math.min(Math.max(match.score * pathBoost * pattern.boost, 0), 1), // Fix: clamp score to [0,1]
                why: match.reasons.map(mapToValidReason), // Fix: map to valid enum values
                byte_offset: match.byte_offset,
                span_len: match.span_len,
                pattern_type: match.pattern_type,
                symbol_name: match.symbol_name,
                signature: match.signature,
              };
              
              results.push(result);
            }
          }
        } catch (error) {
          console.warn(`Error in structural search for file ${filePath}:`, error);
          continue;
        }
      }

      // Deduplicate and sort results
      const dedupedResults = this.deduplicateStructuralResults(results);
      dedupedResults.sort((a, b) => b.score - a.score);
      
      const finalResults = dedupedResults.slice(0, params.k);

      span.setAttributes({
        success: true,
        results_found: finalResults.length,
        files_searched: this.shardPaths.filter(p => p.endsWith('.py')).length,
        patterns_count: patterns.length,
      });

      console.log(`✅ Found ${finalResults.length} structural matches for "${params.q}"`);
      return finalResults;

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw new Error(`Structural search failed: ${errorMsg}`);
    } finally {
      span.end();
    }
  }

  async close(): Promise<void> {
    // Clear file cache
    this.fileCache.clear();
  }

  /**
   * Generate search terms including original query, subtokens, and synonyms
   */
  private generateSearchTerms(query: string, useSubtokens: boolean): Array<{ term: string; boost: number; source: string }> {
    const terms: Array<{ term: string; boost: number; source: string }> = [];
    const queryLower = query.toLowerCase();

    // Original query gets highest boost
    terms.push({ term: queryLower, boost: 1.0, source: 'exact' });

    // Add subtokens if enabled
    if (useSubtokens) {
      const subtokens = this.extractSubtokens(query);
      for (const subtoken of subtokens) {
        if (subtoken !== queryLower && subtoken.length > 1) {
          terms.push({ term: subtoken, boost: 0.7, source: 'subtoken' });
        }
      }
    }

    // Add synonyms
    const synonyms = this.synonymMap.get(queryLower) || [];
    for (const synonym of synonyms) {
      terms.push({ term: synonym.toLowerCase(), boost: 0.8, source: 'synonym' });
    }

    return terms;
  }

  /**
   * Extract subtokens from camelCase and snake_case identifiers
   */
  private extractSubtokens(query: string): string[] {
    const tokens: string[] = [];
    
    // Handle camelCase (e.g., getUserData -> [get, user, data])
    const camelSplit = query.split(/(?=[A-Z])/).filter(t => t.length > 0);
    if (camelSplit.length > 1) {
      tokens.push(...camelSplit.map(t => t.toLowerCase()));
    }
    
    // Handle snake_case (e.g., user_service -> [user, service])
    const snakeSplit = query.split('_').filter(t => t.length > 0);
    if (snakeSplit.length > 1) {
      tokens.push(...snakeSplit.map(t => t.toLowerCase()));
    }
    
    // Handle kebab-case (e.g., user-service -> [user, service])
    const kebabSplit = query.split('-').filter(t => t.length > 0);
    if (kebabSplit.length > 1) {
      tokens.push(...kebabSplit.map(t => t.toLowerCase()));
    }
    
    // Handle dot notation (e.g., user.service -> [user, service])
    const dotSplit = query.split('.').filter(t => t.length > 0);
    if (dotSplit.length > 1) {
      tokens.push(...dotSplit.map(t => t.toLowerCase()));
    }

    return tokens;
  }

  /**
   * Calculate path-based boost for search results
   */
  private calculatePathBoost(filePath: string): number {
    const pathLower = filePath.toLowerCase();
    
    // Boost core implementation directories
    if (pathLower.includes('/src/') || pathLower.startsWith('src/')) return 1.5;
    if (pathLower.includes('/lib/') || pathLower.startsWith('lib/')) return 1.3;
    if (pathLower.includes('/app/') || pathLower.startsWith('app/')) return 1.2;
    
    // De-boost non-core directories
    if (pathLower.includes('/test/') || pathLower.includes('/__test__/')) return 0.7;
    if (pathLower.includes('/spec/') || pathLower.includes('.spec.')) return 0.7;
    if (pathLower.includes('/vendor/') || pathLower.includes('/node_modules/')) return 0.3;
    if (pathLower.includes('/build/') || pathLower.includes('/dist/')) return 0.3;
    if (pathLower.includes('.d.ts')) return 0.4; // Type definitions are less relevant
    if (pathLower.includes('/docs/') || pathLower.includes('/documentation/')) return 0.5;
    
    return 1.0; // Default boost
  }

  /**
   * Tokenize a line into words for matching
   */
  private tokenizeLine(line: string): Array<{ word: string; col: number }> {
    const words: Array<{ word: string; col: number }> = [];
    const wordRegex = /\b\w+\b/g;
    let match: RegExpExecArray | null;
    
    while ((match = wordRegex.exec(line)) !== null) {
      words.push({
        word: match[0].toLowerCase(),
        col: match.index
      });
    }
    
    return words;
  }

  /**
   * Find matches for a search term in a line with fuzzy matching support
   */
  private findMatches(
    line: string, 
    lineWords: Array<{ word: string; col: number }>, 
    searchTerm: { term: string; boost: number; source: string }, 
    fuzzyDistance: number
  ): Array<{ col: number; score: number; reasons: string[]; span_len: number }> {
    const matches: Array<{ col: number; score: number; reasons: string[]; span_len: number }> = [];
    const query = searchTerm.term;
    const lineLower = line.toLowerCase();
    
    // 1. Exact substring match (highest priority)
    let exactIndex = lineLower.indexOf(query);
    while (exactIndex !== -1) {
      matches.push({
        col: exactIndex,
        score: 1.0,
        reasons: ['exact', searchTerm.source],
        span_len: query.length
      });
      exactIndex = lineLower.indexOf(query, exactIndex + 1);
    }
    
    // 2. Word boundary matches
    for (const word of lineWords) {
      if (word.word === query) {
        // Exact word match (if not already found as substring)
        const alreadyFound = matches.some(m => 
          Math.abs(m.col - word.col) <= 1 && m.reasons.includes('exact')
        );
        if (!alreadyFound) {
          matches.push({
            col: word.col,
            score: 0.9,
            reasons: ['word_exact', searchTerm.source],
            span_len: word.word.length
          });
        }
      } else if (fuzzyDistance > 0) {
        // Fuzzy word match
        const editDistance = this.calculateEditDistance(word.word, query);
        if (editDistance > 0 && editDistance <= fuzzyDistance && editDistance <= Math.max(2, query.length / 3)) {
          const fuzzyScore = Math.max(0.3, 1.0 - (editDistance / Math.max(query.length, word.word.length)));
          matches.push({
            col: word.col,
            score: fuzzyScore,
            reasons: ['fuzzy', searchTerm.source],  // Fixed: use valid enum value
            span_len: word.word.length
          });
        }
        
        // Prefix match
        if (word.word.startsWith(query) && query.length >= 3) {
          matches.push({
            col: word.col,
            score: 0.8,
            reasons: ['exact', searchTerm.source],  // Fixed: map prefix to exact
            span_len: query.length
          });
        }
        
        // Suffix match
        if (word.word.endsWith(query) && query.length >= 3) {
          matches.push({
            col: word.col + word.word.length - query.length,
            score: 0.7,
            reasons: ['exact', searchTerm.source],  // Fixed: map suffix to exact
            span_len: query.length
          });
        }
      }
    }
    
    return matches;
  }

  /**
   * Calculate edit distance between two strings (Levenshtein distance)
   */
  private calculateEditDistance(str1: string, str2: string): number {
    const matrix: number[][] = Array(str2.length + 1).fill(null).map(() => Array(str1.length + 1).fill(0));
    
    // Initialize first row and column
    for (let i = 0; i <= str1.length; i++) matrix[0]![i] = i;
    for (let j = 0; j <= str2.length; j++) matrix[j]![0] = j;
    
    // Fill matrix
    for (let j = 1; j <= str2.length; j++) {
      for (let i = 1; i <= str1.length; i++) {
        const currentRow = matrix[j]!;
        const prevRow = matrix[j - 1]!;
        
        if (str1[i - 1] === str2[j - 1]) {
          currentRow[i] = prevRow[i - 1]!; // No operation needed
        } else {
          currentRow[i] = Math.min(
            prevRow[i]! + 1,      // Deletion
            currentRow[i - 1]! + 1, // Insertion
            prevRow[i - 1]! + 1   // Substitution
          );
        }
      }
    }
    
    return matrix[str2.length]![str1.length]!;
  }

  /**
   * Deduplicate results with the same file/line/col and merge their reasons
   */
  private deduplicateResults(results: LexicalResult[]): LexicalResult[] {
    const grouped = new Map<string, LexicalResult>();
    
    for (const result of results) {
      const key = `${result.file}:${result.line}:${result.col}`;
      const existing = grouped.get(key);
      
      if (existing) {
        // Merge with higher score and combined reasons
        existing.score = Math.max(existing.score, result.score);
        existing.why = Array.from(new Set([...existing.why, ...result.why]));
      } else {
        grouped.set(key, { ...result });
      }
    }
    
    return Array.from(grouped.values());
  }

  /**
   * Deduplicate structural results with the same file/line/col
   */
  private deduplicateStructuralResults(results: StructuralResult[]): StructuralResult[] {
    const grouped = new Map<string, StructuralResult>();
    
    for (const result of results) {
      const key = `${result.file}:${result.line}:${result.col}`;
      const existing = grouped.get(key);
      
      if (existing) {
        // Merge with higher score and combined reasons
        existing.score = Math.max(existing.score, result.score);
        existing.why = Array.from(new Set([...existing.why, ...result.why]));
        
        // Prefer more specific pattern types
        if (result.pattern_type === 'async_def' && existing.pattern_type === 'function_def') {
          existing.pattern_type = result.pattern_type;
        }
      } else {
        grouped.set(key, { ...result });
      }
    }
    
    return Array.from(grouped.values());
  }

  /**
   * Generate structural patterns based on the query
   */
  private generateStructuralPatterns(query: string, explicitPatterns?: string[]): Array<{
    pattern: RegExp;
    type: StructuralResult['pattern_type'];
    boost: number;
    description: string;
  }> {
    const patterns: Array<{
      pattern: RegExp;
      type: StructuralResult['pattern_type'];
      boost: number;
      description: string;
    }> = [];

    const queryLower = query.toLowerCase();

    // If explicit patterns provided, use those
    if (explicitPatterns?.length) {
      for (const patternName of explicitPatterns) {
        const pattern = this.getStructuralPattern(patternName);
        if (pattern) patterns.push(pattern);
      }
      return patterns;
    }

    // Auto-generate patterns based on query content
    
    // === TypeScript-specific patterns (high-yield Phase 2 additions) ===
    
    // 1. Function with export pattern
    if (queryLower.includes('function') || queryLower.includes('export')) {
      patterns.push({
        pattern: /^(\s*)export\s+(?:async\s+)?function\s+(\w+)\s*[<(]/gm,
        type: 'function_def',
        boost: 1.3,
        description: 'typescript exported function'
      });
    }

    // 2. Class with implements/extends pattern  
    if (queryLower.includes('class') || queryLower.includes('implements') || queryLower.includes('extends')) {
      patterns.push({
        pattern: /^(\s*)export\s+(?:abstract\s+)?class\s+(\w+)(?:\s+(?:extends|implements)\s+[\w<>,\s]+)?\s*\{/gm,
        type: 'class_def',
        boost: 1.4,
        description: 'typescript class with inheritance'
      });
    }

    // 3. Interface extends pattern
    if (queryLower.includes('interface') || queryLower.includes('extends')) {
      patterns.push({
        pattern: /^(\s*)export\s+interface\s+(\w+)(?:\s+extends\s+[\w<>,\s]+)?\s*\{/gm,
        type: 'class_def',
        boost: 1.2,
        description: 'typescript interface definition'
      });
    }

    // 4. Type definitions pattern
    if (queryLower.includes('type')) {
      patterns.push({
        pattern: /^(\s*)export\s+type\s+(\w+)(?:<[^>]*>)?\s*=/gm,
        type: 'class_def',
        boost: 1.1,
        description: 'typescript type definition'
      });
    }

    // 5. Async function pattern (TypeScript/JavaScript)
    if (queryLower.includes('async')) {
      patterns.push({
        pattern: /^(\s*)(?:export\s+)?async\s+function\s+(\w+)\s*[<(]/gm,
        type: 'async_def',
        boost: 1.2,
        description: 'typescript async function'
      });
    }
    
    // === Python patterns (existing) ===
    
    // Function definition patterns
    if (queryLower.includes('def') || queryLower.includes('function')) {
      patterns.push({
        pattern: /^(\s*)def\s+(\w+)\s*\([^)]*\)\s*(?:->.*?)?:/gm,
        type: 'function_def',
        boost: 1.0,
        description: 'python function definition'
      });
    }

    // Python async function patterns
    if (queryLower.includes('async') && !queryLower.includes('function')) {
      patterns.push({
        pattern: /^(\s*)async\s+def\s+(\w+)\s*\([^)]*\)\s*(?:->.*?)?:/gm,
        type: 'async_def',
        boost: 1.2,
        description: 'python async function definition'
      });
    }

    // Python class definition patterns
    if (queryLower.includes('class')) {
      patterns.push({
        pattern: /^(\s*)class\s+(\w+)(?:\([^)]*\))?\s*:/gm,
        type: 'class_def',
        boost: 1.0,
        description: 'python class definition'
      });
    }

    // Import patterns
    if (queryLower.includes('import')) {
      patterns.push({
        pattern: /^(\s*)from\s+([\w.]+)\s+import\s+(.+)/gm,
        type: 'import',
        boost: 0.8,
        description: 'from import statement'
      });
      patterns.push({
        pattern: /^(\s*)import\s+([\w.,\s]+)/gm,
        type: 'import',
        boost: 0.8,
        description: 'import statement'
      });
    }

    // Decorator patterns
    if (queryLower.includes('@') || queryLower.includes('decorator')) {
      patterns.push({
        pattern: /^(\s*)@(\w+)(?:\([^)]*\))?/gm,
        type: 'decorator',
        boost: 0.9,
        description: 'decorator'
      });
    }

    // Try/except patterns
    if (queryLower.includes('try') || queryLower.includes('except')) {
      patterns.push({
        pattern: /^(\s*)try\s*:/gm,
        type: 'try_except',
        boost: 0.9,
        description: 'try block'
      });
    }

    // For loop patterns
    if (queryLower.includes('for') && !queryLower.includes('format')) {
      patterns.push({
        pattern: /^(\s*)for\s+(\w+)\s+in\s+(.+):/gm,
        type: 'for_loop',
        boost: 0.8,
        description: 'for loop'
      });
    }

    // If statement patterns
    if (queryLower.includes('if')) {
      patterns.push({
        pattern: /^(\s*)if\s+(.+):/gm,
        type: 'if_statement',
        boost: 0.7,
        description: 'if statement'
      });
    }

    // If no specific patterns match, add common Python patterns
    if (patterns.length === 0) {
      patterns.push(
        {
          pattern: /^(\s*)def\s+(\w+)\s*\([^)]*\)\s*(?:->.*?)?:/gm,
          type: 'function_def',
          boost: 0.6,
          description: 'function definition (fallback)'
        },
        {
          pattern: /^(\s*)class\s+(\w+)(?:\([^)]*\))?\s*:/gm,
          type: 'class_def',
          boost: 0.6,
          description: 'class definition (fallback)'
        }
      );
    }

    return patterns;
  }

  /**
   * Get a specific structural pattern by name
   */
  private getStructuralPattern(patternName: string): {
    pattern: RegExp;
    type: StructuralResult['pattern_type'];
    boost: number;
    description: string;
  } | null {
    const patterns = {
      'function_def': {
        pattern: /^(\s*)def\s+(\w+)\s*\([^)]*\)\s*(?:->.*?)?:/gm,
        type: 'function_def' as const,
        boost: 1.0,
        description: 'function definition'
      },
      'async_def': {
        pattern: /^(\s*)async\s+def\s+(\w+)\s*\([^)]*\)\s*(?:->.*?)?:/gm,
        type: 'async_def' as const,
        boost: 1.2,
        description: 'async function definition'
      },
      'class_def': {
        pattern: /^(\s*)class\s+(\w+)(?:\([^)]*\))?\s*:/gm,
        type: 'class_def' as const,
        boost: 1.0,
        description: 'class definition'
      },
      'import': {
        pattern: /^(\s*)(?:from\s+([\w.]+)\s+)?import\s+(.+)/gm,
        type: 'import' as const,
        boost: 0.8,
        description: 'import statement'
      },
    };

    return patterns[patternName as keyof typeof patterns] || null;
  }

  /**
   * Find structural matches in content using AST-like patterns
   */
  private findStructuralMatches(
    content: string,
    lines: string[],
    pattern: {
      pattern: RegExp;
      type: StructuralResult['pattern_type'];
      boost: number;
      description: string;
    },
    filePath: string,
    query: string,
    isTypeScript: boolean = false
  ): Array<{
    line: number;
    col: number;
    snippet: string;
    score: number;
    reasons: string[];
    byte_offset: number;
    span_len: number;
    pattern_type: StructuralResult['pattern_type'];
    symbol_name?: string | undefined;
    signature?: string | undefined;
  }> {
    const matches: Array<{
      line: number;
      col: number;
      snippet: string;
      score: number;
      reasons: string[];
      byte_offset: number;
      span_len: number;
      pattern_type: StructuralResult['pattern_type'];
      symbol_name?: string | undefined;
      signature?: string | undefined;
    }> = [];

    // Reset regex to start from beginning
    pattern.pattern.lastIndex = 0;
    
    let match: RegExpExecArray | null;
    while ((match = pattern.pattern.exec(content)) !== null) {
      const lineNumber = this.getLineNumber(content, match.index!);
      const lineContent = lines[lineNumber - 1] || '';
      const colNumber = Math.max(match.index! - content.lastIndexOf('\n', match.index!) - 1, 0); // Fix: ensure col >= 0
      
      // Extract symbol name based on pattern type
      let symbolName: string | undefined;
      let signature: string | undefined;
      
      switch (pattern.type) {
        case 'function_def':
        case 'async_def':
          symbolName = match[2]; // Function name is typically in capture group 2
          signature = match[0].trim();
          break;
        case 'class_def':
          symbolName = match[2]; // Class name
          signature = match[0].trim();
          break;
        case 'import':
          if (match[2]) {
            symbolName = match[2]; // Module name for 'from X import'
          } else if (match[3]) {
            symbolName = match[3]; // Import list
          }
          break;
        case 'decorator':
          symbolName = match[2]; // Decorator name
          break;
      }
      
      // Calculate score based on pattern specificity and context
      let score = 0.8; // Base score for structural match
      
      // Boost exact symbol name matches
      const querySymbol = this.extractQuerySymbol(query);
      if (symbolName && symbolName.toLowerCase().includes(querySymbol)) {
        score += 0.2;
      }
      
      matches.push({
        line: lineNumber,
        col: Math.max(colNumber, 0), // Fix: ensure col >= 0
        snippet: lineContent.trim(),
        score: Math.min(Math.max(score, 0), 1), // Fix: clamp score to [0,1]
        reasons: ['structural'], // Fix: use only valid enum values
        byte_offset: this.calculateByteOffset(lines, lineNumber - 1, colNumber),
        span_len: match[0].length,
        pattern_type: pattern.type,
        symbol_name: symbolName,
        signature,
      });
    }

    return matches;
  }

  /**
   * Extract potential symbol name from query
   */
  private extractQuerySymbol(query: string): string {
    // Extract alphanumeric sequences that might be symbol names
    const matches = query.toLowerCase().match(/[a-zA-Z_][a-zA-Z0-9_]*/g);
    return matches?.[0] || query.toLowerCase();
  }

  /**
   * Get line number from character index in content
   */
  private getLineNumber(content: string, index: number): number {
    return content.substring(0, index).split('\n').length;
  }

  /**
   * Get list of files in the index
   */
  async getFileList(): Promise<string[]> {
    return this.shardPaths;
  }
}