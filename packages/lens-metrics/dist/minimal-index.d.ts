/**
 * Minimal @lens/metrics export for immediate use
 */
export { LensMetricsEngine } from './metrics-engine.js';
export { DataMigrator } from './data-migration.js';
export type { CanonicalQuery, SearchResult, MetricsConfig, SystemResults, ValidationGates } from './types.js';
import type { MetricsConfig, ValidationGates } from './types.js';
export declare const DEFAULT_CONFIG: MetricsConfig;
export declare const DEFAULT_VALIDATION_GATES: ValidationGates;
