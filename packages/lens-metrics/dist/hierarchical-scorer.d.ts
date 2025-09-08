/**
 * Hierarchical Credit System: span → symbol → file
 * Prevents bogus 1.000 scores by using graded gains
 */
import { CanonicalQuery, SearchResult, QueryMetrics, MetricsConfig } from './types.js';
export declare class HierarchicalScorer {
    private config;
    private normalizer;
    constructor(config: MetricsConfig);
    /**
     * Score a query using hierarchical credit system
     */
    scoreQuery(query: CanonicalQuery, searchResults: SearchResult[], latencyMs: number): QueryMetrics;
    /**
     * Score individual results using hierarchical matching
     */
    private scoreResults;
    /**
     * Calculate all metrics from scored results
     */
    private calculateMetrics;
    /**
     * Calculate nDCG using hierarchical gains
     */
    private calculateNDCG;
    /**
     * Calculate Mean Average Precision
     */
    private calculateMAP;
    /**
     * Get total number of relevant items for recall calculation
     */
    private getTotalRelevantCount;
    /**
     * Calculate span coverage percentage in labels
     */
    private calculateSpanCoverage;
}
