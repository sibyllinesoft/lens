/**
 * @lens/metrics - Canonical Metrics Engine
 * Single source of truth for all Lens benchmarking evaluation
 */
export { LensMetricsEngine } from './metrics-engine.js';
export { HierarchicalScorer } from './hierarchical-scorer.js';
export { DataNormalizer } from './normalizer.js';
export { DataMigrator } from './data-migration.js';
export type { CanonicalQuery, SearchResult, ScoredResult, QueryMetrics, MetricsConfig, ExpectedSpan, ExpectedSymbol, ExpectedFile, CreditPolicy, CreditMode, SystemResults, PooledQrels, ValidationGates, AggregateMetrics, ValidationReport, PoolMembershipReport } from './types.js';
export declare const DEFAULT_CONFIG: MetricsConfig;
export declare const DEFAULT_VALIDATION_GATES: ValidationGates;
/**
 * Quick evaluation function for simple use cases
 */
export declare function evaluateQuickly(systemId: string, queries: Array<{
    query: CanonicalQuery;
    results: SearchResult[];
    latency_ms: number;
}>, config?: Partial<MetricsConfig>): any;
/**
 * Build pooled qrels from multiple systems
 */
export declare function buildPooledQrels(allSystemResults: SystemResults[], topK?: number, config?: Partial<MetricsConfig>): PooledQrels[];
//# sourceMappingURL=index.d.ts.map