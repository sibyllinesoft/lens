/**
 * Minimal @lens/metrics export for immediate use
 */

export { LensMetricsEngine } from './metrics-engine.js';
export { DataMigrator } from './data-migration.js';

export type {
  CanonicalQuery,
  SearchResult,
  MetricsConfig,
  SystemResults,
  ValidationGates
} from './types.js';

import type { MetricsConfig, ValidationGates } from './types.js';

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

export const DEFAULT_VALIDATION_GATES: ValidationGates = {
  sla_threshold_ms: 150,
  min_queries_for_gate: 10,
  min_median_hits_in_pool: 0,
  max_perfect_score_threshold: 0.95
};