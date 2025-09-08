/**
 * @lens/metrics - Canonical Metrics Engine
 * Single source of truth for all Lens benchmarking evaluation
 */

export { LensMetricsEngine } from './metrics-engine.js';
export { HierarchicalScorer } from './hierarchical-scorer.js';
export { DataNormalizer } from './normalizer.js';
export { DataMigrator } from './data-migration.js';

export type {
  CanonicalQuery,
  SearchResult,
  ScoredResult,
  QueryMetrics,
  MetricsConfig,
  ExpectedSpan,
  ExpectedSymbol,
  ExpectedFile,
  CreditPolicy,
  CreditMode,
  SystemResults,
  PooledQrels,
  ValidationGates,
  AggregateMetrics,
  ValidationReport,
  PoolMembershipReport
} from './types.js';

import type { 
  MetricsConfig, 
  ValidationGates, 
  CanonicalQuery, 
  SearchResult, 
  SystemResults, 
  PooledQrels 
} from './types.js';
import { LensMetricsEngine } from './metrics-engine.js';

// Default configuration
export const DEFAULT_CONFIG: MetricsConfig = {
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
export const DEFAULT_VALIDATION_GATES: ValidationGates = {
  sla_threshold_ms: 150,
  min_queries_for_gate: 10,
  min_median_hits_in_pool: 0,
  max_perfect_score_threshold: 0.95
};

/**
 * Quick evaluation function for simple use cases
 */
export function evaluateQuickly(
  systemId: string,
  queries: Array<{
    query: CanonicalQuery;
    results: SearchResult[];
    latency_ms: number;
  }>,
  config?: Partial<MetricsConfig>
) {
  const engine = new LensMetricsEngine(config);
  
  return engine.evaluateSystem(
    { system_id: systemId, queries },
    undefined,
    DEFAULT_VALIDATION_GATES
  );
}

/**
 * Build pooled qrels from multiple systems
 */
export function buildPooledQrels(
  allSystemResults: SystemResults[],
  topK: number = 50,
  config?: Partial<MetricsConfig>
): PooledQrels[] {
  const engine = new LensMetricsEngine(config);
  return engine.buildPooledQrels(allSystemResults, topK);
}