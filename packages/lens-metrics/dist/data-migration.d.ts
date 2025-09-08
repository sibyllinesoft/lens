/**
 * Data Migration Utilities
 * Migrate legacy datasets to canonical format
 */
import { CanonicalQuery } from './types.js';
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
export declare class DataMigrator {
    /**
     * Migrate legacy query format to canonical schema
     */
    static migrateQuery(legacy: LegacyQuery, defaultRepo?: string): CanonicalQuery;
    /**
     * Batch migrate an array of legacy queries
     */
    static migrateQueries(legacyQueries: LegacyQuery[], defaultRepo?: string): CanonicalQuery[];
    /**
     * Normalize language codes to canonical form
     */
    private static normalizeLang;
    /**
     * Normalize suite names to canonical form
     */
    private static normalizeSuite;
    /**
     * Calculate span coverage for migrated dataset
     */
    static calculateSpanCoverage(queries: CanonicalQuery[]): {
        total_queries: number;
        queries_with_spans: number;
        span_coverage_percent: number;
        coverage_by_suite: Record<string, number>;
    };
}
