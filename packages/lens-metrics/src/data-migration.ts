/**
 * Data Migration Utilities
 * Migrate legacy datasets to canonical format
 */

import { CanonicalQuery, ExpectedFile, ExpectedSpan } from './types.js';

export interface LegacyQuery {
  query_id?: string;
  query?: string;
  expected_results?: any[];
  expected_files?: string[];
  language?: string;
  intent?: string;
  suite?: string;
  [key: string]: any;
}

export class DataMigrator {
  /**
   * Migrate legacy query format to canonical schema
   */
  static migrateQuery(legacy: LegacyQuery, defaultRepo: string = ''): CanonicalQuery {
    // Extract basic info
    const queryId = legacy.query_id || legacy.id || `query_${Math.random().toString(36).slice(2)}`;
    const suite = legacy.suite || 'unknown';
    const lang = this.normalizeLang(legacy.language || legacy.lang || 'unknown');
    
    // Migrate expected results
    let expectedFiles: ExpectedFile[] = [];
    let expectedSpans: ExpectedSpan[] = [];
    
    // Handle expected_results array
    if (legacy.expected_results && Array.isArray(legacy.expected_results)) {
      for (const result of legacy.expected_results) {
        if (typeof result === 'string') {
          // Simple file path
          expectedFiles.push({
            repo: defaultRepo,
            path: result
          });
        } else if (result && typeof result === 'object') {
          // Complex result object
          const path = result.path || result.file || result.filename || '';
          if (path) {
            if (result.line !== undefined && result.col !== undefined) {
              // Has span info
              expectedSpans.push({
                repo: result.repo || defaultRepo,
                path,
                line: Number(result.line),
                col: Number(result.col)
              });
            } else {
              // File-only
              expectedFiles.push({
                repo: result.repo || defaultRepo,
                path
              });
            }
          }
        }
      }
    }
    
    // Handle expected_files array
    if (legacy.expected_files && Array.isArray(legacy.expected_files)) {
      for (const file of legacy.expected_files) {
        if (typeof file === 'string') {
          expectedFiles.push({
            repo: defaultRepo,
            path: file
          });
        }
      }
    }
    
    // Determine credit policy
    const hasSpans = expectedSpans.length > 0;
    const creditPolicy = hasSpans ? 'hierarchical' : 'hierarchical'; // Always hierarchical for robustness
    
    return {
      query_id: queryId,
      expected_spans: expectedSpans.length > 0 ? expectedSpans : undefined,
      expected_files: expectedFiles,
      credit_policy: creditPolicy,
      metadata: {
        suite: this.normalizeSuite(suite),
        lang: lang,
        intent: legacy.intent || 'unknown',
        query_type: legacy.query_type || 'lexical'
      }
    };
  }

  /**
   * Batch migrate an array of legacy queries
   */
  static migrateQueries(legacyQueries: LegacyQuery[], defaultRepo: string = ''): CanonicalQuery[] {
    return legacyQueries.map(query => this.migrateQuery(query, defaultRepo));
  }

  /**
   * Normalize language codes to canonical form
   */
  private static normalizeLang(lang: string): 'py' | 'ts' | 'js' | 'go' | 'rust' | 'java' | 'cpp' {
    const normalized = lang.toLowerCase();
    
    switch (normalized) {
      case 'python': return 'py';
      case 'typescript': return 'ts';
      case 'javascript': return 'js';
      case 'golang': return 'go';
      case 'rust': return 'rust';
      case 'java': return 'java';
      case 'c++': case 'cpp': return 'cpp';
      default: return 'py'; // Default fallback
    }
  }

  /**
   * Normalize suite names to canonical form
   */
  private static normalizeSuite(suite: string): 'coir' | 'swe_verified' | 'csn' | 'cosqa' {
    const normalized = suite.toLowerCase();
    
    switch (normalized) {
      case 'coir': return 'coir';
      case 'swe_verified': case 'swe-verified': case 'swebench': return 'swe_verified';
      case 'csn': case 'codesearchnet': return 'csn';
      case 'cosqa': return 'cosqa';
      default: return 'coir'; // Default fallback
    }
  }

  /**
   * Calculate span coverage for migrated dataset
   */
  static calculateSpanCoverage(queries: CanonicalQuery[]): {
    total_queries: number;
    queries_with_spans: number;
    span_coverage_percent: number;
    coverage_by_suite: Record<string, number>;
  } {
    const totalQueries = queries.length;
    const queriesWithSpans = queries.filter(q => q.expected_spans && q.expected_spans.length > 0).length;
    
    // Calculate coverage by suite
    const coverageBySuite: Record<string, { total: number; withSpans: number }> = {};
    
    for (const query of queries) {
      const suite = query.metadata.suite;
      if (!coverageBySuite[suite]) {
        coverageBySuite[suite] = { total: 0, withSpans: 0 };
      }
      coverageBySuite[suite].total++;
      if (query.expected_spans && query.expected_spans.length > 0) {
        coverageBySuite[suite].withSpans++;
      }
    }
    
    const coveragePercents: Record<string, number> = {};
    for (const [suite, counts] of Object.entries(coverageBySuite)) {
      coveragePercents[suite] = counts.total > 0 ? (counts.withSpans / counts.total) * 100 : 0;
    }
    
    return {
      total_queries: totalQueries,
      queries_with_spans: queriesWithSpans,
      span_coverage_percent: totalQueries > 0 ? (queriesWithSpans / totalQueries) * 100 : 0,
      coverage_by_suite: coveragePercents
    };
  }
}