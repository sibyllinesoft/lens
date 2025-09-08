/**
 * Path and Data Normalization for Consistent Joins
 */

import crypto from 'crypto';
import { SearchResult, ExpectedSpan, ExpectedSymbol, ExpectedFile, MetricsConfig } from './types.js';

export class DataNormalizer {
  constructor(private config: MetricsConfig) {}

  /**
   * Normalize path to canonical form: repo-relative, NFC, / separators
   */
  normalizePath(repo: string, rawPath: string): string {
    // Remove repo prefix if present
    let path = rawPath;
    if (path.startsWith(repo)) {
      path = path.slice(repo.length);
    }
    
    // Remove leading slash
    if (path.startsWith('/')) {
      path = path.slice(1);
    }
    
    // Normalize to NFC Unicode form
    path = path.normalize('NFC');
    
    // Use consistent path separators
    path = path.replace(/\\/g, this.config.normalization.path_separator);
    
    return path;
  }

  /**
   * Generate snippet hash for robust joining when spans don't match
   */
  generateSnippetHash(snippet: string): string {
    if (!snippet) return '';
    
    // Normalize whitespace and generate consistent hash
    const normalized = snippet.trim().replace(/\s+/g, ' ');
    return crypto.createHash('sha256').update(normalized).digest('hex').slice(0, 16);
  }

  /**
   * Normalize search result for consistent matching
   */
  normalizeSearchResult(result: SearchResult): SearchResult {
    const normalized = { ...result };
    
    // Normalize path
    normalized.path = this.normalizePath(result.repo, result.path);
    
    // Generate snippet hash if snippet provided
    if (result.snippet) {
      normalized.snippet_hash = this.generateSnippetHash(result.snippet);
    }
    
    // Normalize line/col to expected base
    if (normalized.line !== null && normalized.line !== undefined) {
      // Ensure 1-based line numbering
      if (this.config.normalization.line_base === 1 && normalized.line === 0) {
        normalized.line = 1;
      }
    }
    
    if (normalized.col !== null && normalized.col !== undefined) {
      // Ensure 0-based column numbering  
      if (this.config.normalization.col_base === 0 && normalized.col < 0) {
        normalized.col = 0;
      }
    }
    
    return normalized;
  }

  /**
   * Create join keys for hierarchical matching
   */
  createJoinKeys(result: SearchResult): {
    spanKey: string | null;
    symbolKey: string | null; 
    fileKey: string;
    snippetKey: string | null;
  } {
    const normalized = this.normalizeSearchResult(result);
    
    const spanKey = (normalized.line !== null && normalized.line !== undefined && 
                    normalized.col !== null && normalized.col !== undefined) 
      ? `${normalized.repo}:${normalized.path}:${normalized.line}:${normalized.col}`
      : null;
    
    const symbolKey = null; // Would need symbol extraction logic
    
    const fileKey = `${normalized.repo}:${normalized.path}`;
    
    const snippetKey = normalized.snippet_hash 
      ? `${normalized.repo}:${normalized.path}:${normalized.snippet_hash}`
      : null;
    
    return { spanKey, symbolKey, fileKey, snippetKey };
  }

  /**
   * Create expected keys from canonical query
   */
  createExpectedKeys(query: {
    expected_spans?: ExpectedSpan[];
    expected_symbols?: ExpectedSymbol[];
    expected_files: ExpectedFile[];
  }): {
    spanKeys: Set<string>;
    symbolKeys: Set<string>;
    fileKeys: Set<string>;
  } {
    const spanKeys = new Set<string>();
    const symbolKeys = new Set<string>();
    const fileKeys = new Set<string>();
    
    // Add span keys
    if (query.expected_spans) {
      for (const span of query.expected_spans) {
        const normalizedPath = this.normalizePath(span.repo, span.path);
        spanKeys.add(`${span.repo}:${normalizedPath}:${span.line}:${span.col}`);
      }
    }
    
    // Add symbol keys  
    if (query.expected_symbols) {
      for (const symbol of query.expected_symbols) {
        const normalizedPath = this.normalizePath(symbol.repo, symbol.path);
        symbolKeys.add(`${symbol.repo}:${normalizedPath}:${symbol.symbol}`);
      }
    }
    
    // Add file keys
    for (const file of query.expected_files) {
      const normalizedPath = this.normalizePath(file.repo, file.path);
      fileKeys.add(`${file.repo}:${normalizedPath}`);
    }
    
    return { spanKeys, symbolKeys, fileKeys };
  }
}