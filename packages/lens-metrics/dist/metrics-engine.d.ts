/**
 * @lens/metrics - Canonical Metrics Engine
 * Single source of truth for all evaluation
 */
import { QueryMetrics, MetricsConfig, SystemResults, PooledQrels, ValidationGates, AggregateMetrics, ValidationReport, PoolMembershipReport } from './types.js';
export declare class LensMetricsEngine {
    private config;
    private scorer;
    private normalizer;
    constructor(config?: Partial<MetricsConfig>);
    /**
     * Primary evaluation method - single source of truth
     */
    evaluateSystem(systemResults: SystemResults, pooledQrels?: PooledQrels[], validationGates?: ValidationGates): {
        system_id: string;
        query_metrics: QueryMetrics[];
        aggregate_metrics: AggregateMetrics;
        validation_report: ValidationReport;
        pool_membership: PoolMembershipReport;
    };
    /**
     * Build pooled qrels from multiple system results
     */
    buildPooledQrels(allSystemResults: SystemResults[], topK?: number): PooledQrels[];
    /**
     * Hard validation gates to prevent regression
     */
    private runValidationGates;
    private generatePoolMembershipReport;
    private calculateAggregateMetrics;
    private calculateMedian;
    private calculatePercentile;
}
