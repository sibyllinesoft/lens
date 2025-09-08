/**
 * @lens/metrics - Canonical Metrics Engine
 * Single source of truth for all Lens benchmarking evaluation
 */
export { LensMetricsEngine } from './metrics-engine.js';
export { HierarchicalScorer } from './hierarchical-scorer.js';
export { DataNormalizer } from './normalizer.js';
export { DataMigrator } from './data-migration.js';
// Default configuration
export const DEFAULT_CONFIG = {
    sla_threshold_ms: 150,
    k_values: [10, 50],
    credit_gains: {
        span: 1.0,
        symbol: 0.7,
        file: 0.5
    },
    normalization: {
        case_sensitive: true,
        path_separator: '/',
        line_base: 1,
        col_base: 0
    }
};
// Default validation gates
export const DEFAULT_VALIDATION_GATES = {
    sla_threshold_ms: 150,
    min_queries_for_gate: 10,
    min_median_hits_in_pool: 0,
    max_perfect_score_threshold: 0.95
};
/**
 * Quick evaluation function for simple use cases
 */
export function evaluateQuickly(systemId, queries, config) {
    const engine = new LensMetricsEngine(config);
    return engine.evaluateSystem({ system_id: systemId, queries }, undefined, DEFAULT_VALIDATION_GATES);
}
/**
 * Build pooled qrels from multiple systems
 */
export function buildPooledQrels(allSystemResults, topK = 50, config) {
    const engine = new LensMetricsEngine(config);
    return engine.buildPooledQrels(allSystemResults, topK);
}
//# sourceMappingURL=index.js.map