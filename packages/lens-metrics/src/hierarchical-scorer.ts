/**
 * Hierarchical Credit System: span → symbol → file
 * Prevents bogus 1.000 scores by using graded gains
 */

import { 
  CanonicalQuery, 
  SearchResult, 
  ScoredResult, 
  QueryMetrics, 
  MetricsConfig,
  CreditMode 
} from './types.js';
import { DataNormalizer } from './normalizer.js';

export class HierarchicalScorer {
  private normalizer: DataNormalizer;

  constructor(private config: MetricsConfig) {
    this.normalizer = new DataNormalizer(config);
  }

  /**
   * Score a query using hierarchical credit system
   */
  scoreQuery(
    query: CanonicalQuery, 
    searchResults: SearchResult[],
    latencyMs: number
  ): QueryMetrics {
    // Filter for SLA-compliant results
    const slaResults = searchResults.filter(r => latencyMs <= this.config.sla_threshold_ms);
    
    // Get expected keys for matching
    const expectedKeys = this.normalizer.createExpectedKeys(query);
    
    // Score each result with hierarchical credit
    const scoredResults = this.scoreResults(query, slaResults, expectedKeys);
    
    // Calculate metrics
    return this.calculateMetrics(query, scoredResults, searchResults, latencyMs);
  }

  /**
   * Score individual results using hierarchical matching
   */
  private scoreResults(
    query: CanonicalQuery,
    results: SearchResult[],
    expectedKeys: { spanKeys: Set<string>; symbolKeys: Set<string>; fileKeys: Set<string> }
  ): ScoredResult[] {
    return results.map(result => {
      const joinKeys = this.normalizer.createJoinKeys(result);
      let creditMode: CreditMode;
      let relevanceGain: number;

      // Hierarchical matching: span → symbol → file
      if (query.credit_policy === 'span_strict') {
        // Span-only mode: only exact spans get credit
        if (joinKeys.spanKey && expectedKeys.spanKeys.has(joinKeys.spanKey)) {
          creditMode = 'span';
          relevanceGain = this.config.credit_gains.span;
        } else {
          creditMode = 'span';
          relevanceGain = 0;
        }
      } else {
        // Hierarchical mode: try span → symbol → file
        if (joinKeys.spanKey && expectedKeys.spanKeys.has(joinKeys.spanKey)) {
          creditMode = 'span';
          relevanceGain = this.config.credit_gains.span;
        } else if (joinKeys.symbolKey && expectedKeys.symbolKeys.has(joinKeys.symbolKey)) {
          creditMode = 'symbol'; 
          relevanceGain = this.config.credit_gains.symbol;
        } else if (expectedKeys.fileKeys.has(joinKeys.fileKey)) {
          creditMode = 'file';
          relevanceGain = this.config.credit_gains.file;
        } else {
          // Try snippet hash fallback for file-level match
          if (joinKeys.snippetKey) {
            // Check if snippet matches any expected file by path
            const fileMatch = Array.from(expectedKeys.fileKeys).some(fileKey => {
              const [repo, path] = fileKey.split(':', 2);
              return joinKeys.fileKey === `${repo}:${path}`;
            });
            if (fileMatch) {
              creditMode = 'file';
              relevanceGain = this.config.credit_gains.file * 0.8; // Slight penalty for snippet match
            } else {
              creditMode = 'file';
              relevanceGain = 0;
            }
          } else {
            creditMode = 'file';
            relevanceGain = 0;
          }
        }
      }

      return {
        ...result,
        credit_mode_used: creditMode,
        relevance_gain: relevanceGain
      };
    });
  }

  /**
   * Calculate all metrics from scored results
   */
  private calculateMetrics(
    query: CanonicalQuery,
    scoredResults: ScoredResult[],
    allResults: SearchResult[],
    latencyMs: number
  ): QueryMetrics {
    const k10 = Math.min(10, scoredResults.length);
    const k50 = Math.min(50, scoredResults.length);
    
    // Sort by rank for metrics calculation
    const sortedResults = scoredResults.sort((a, b) => a.rank - b.rank);
    
    // Calculate nDCG@10
    const ndcg_at_10 = this.calculateNDCG(sortedResults.slice(0, k10));
    
    // Calculate Success@10 (any relevant result in top 10)
    const success_at_10 = sortedResults.slice(0, k10).some(r => r.relevance_gain > 0) ? 1 : 0;
    
    // Calculate Recall@50
    const totalRelevant = this.getTotalRelevantCount(query);
    const relevantInTop50 = sortedResults.slice(0, k50).filter(r => r.relevance_gain > 0).length;
    const recall_at_50 = totalRelevant > 0 ? relevantInTop50 / totalRelevant : 0;
    
    // Calculate Precision@10
    const relevantInTop10 = sortedResults.slice(0, k10).filter(r => r.relevance_gain > 0).length;
    const precision_at_10 = k10 > 0 ? relevantInTop10 / k10 : 0;
    
    // Calculate MAP (simplified)
    const map = this.calculateMAP(sortedResults);
    
    // Calculate credit breakdown
    const creditHistogram = {
      span: scoredResults.filter(r => r.credit_mode_used === 'span').length,
      symbol: scoredResults.filter(r => r.credit_mode_used === 'symbol').length,
      file: scoredResults.filter(r => r.credit_mode_used === 'file').length
    };
    
    // Calculate span coverage in labels
    const spanCoverage = this.calculateSpanCoverage(query);
    
    return {
      ndcg_at_10,
      success_at_10,
      recall_at_50,
      precision_at_10,
      map,
      credit_mode_used: scoredResults.map(r => r.credit_mode_used),
      span_coverage_in_labels: spanCoverage,
      credit_histogram: creditHistogram,
      hits_in_pool: relevantInTop50,
      total_hits: sortedResults.length,
      sla_compliant_hits: scoredResults.length, // Already filtered
      latency_ms: latencyMs
    };
  }

  /**
   * Calculate nDCG using hierarchical gains
   */
  private calculateNDCG(results: ScoredResult[]): number {
    if (results.length === 0) return 0;
    
    // Calculate DCG
    const dcg = results.reduce((sum, result, index) => {
      const rank = index + 1;
      const gain = result.relevance_gain;
      return sum + (gain / Math.log2(rank + 1));
    }, 0);
    
    // Calculate IDCG (ideal DCG)
    const sortedGains = results
      .map(r => r.relevance_gain)
      .sort((a, b) => b - a);
    
    const idcg = sortedGains.reduce((sum, gain, index) => {
      const rank = index + 1;
      return sum + (gain / Math.log2(rank + 1));
    }, 0);
    
    return idcg > 0 ? dcg / idcg : 0;
  }

  /**
   * Calculate Mean Average Precision
   */
  private calculateMAP(results: ScoredResult[]): number {
    let relevantCount = 0;
    let precisionSum = 0;
    
    results.forEach((result, index) => {
      if (result.relevance_gain > 0) {
        relevantCount++;
        const precision = relevantCount / (index + 1);
        precisionSum += precision;
      }
    });
    
    return relevantCount > 0 ? precisionSum / relevantCount : 0;
  }

  /**
   * Get total number of relevant items for recall calculation
   */
  private getTotalRelevantCount(query: CanonicalQuery): number {
    let total = 0;
    if (query.expected_spans) total += query.expected_spans.length;
    if (query.expected_symbols) total += query.expected_symbols.length;
    total += query.expected_files.length;
    return total;
  }

  /**
   * Calculate span coverage percentage in labels
   */
  private calculateSpanCoverage(query: CanonicalQuery): number {
    const totalItems = this.getTotalRelevantCount(query);
    const spanItems = query.expected_spans?.length || 0;
    return totalItems > 0 ? (spanItems / totalItems) * 100 : 0;
  }
}