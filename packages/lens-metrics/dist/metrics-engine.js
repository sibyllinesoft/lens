/**
 * @lens/metrics - Canonical Metrics Engine
 * Single source of truth for all evaluation
 */
import { HierarchicalScorer } from './hierarchical-scorer.js';
import { DataNormalizer } from './normalizer.js';
// Types are imported from ./types.js
export class LensMetricsEngine {
    config;
    scorer;
    normalizer;
    constructor(config) {
        this.config = {
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
            },
            ...config
        };
        this.scorer = new HierarchicalScorer(this.config);
        this.normalizer = new DataNormalizer(this.config);
    }
    /**
     * Primary evaluation method - single source of truth
     */
    evaluateSystem(systemResults, pooledQrels, validationGates) {
        const queryMetrics = [];
        // Score each query
        for (const queryData of systemResults.queries) {
            const metrics = this.scorer.scoreQuery(queryData.query, queryData.results, queryData.latency_ms);
            queryMetrics.push(metrics);
        }
        // Calculate aggregates
        const aggregateMetrics = this.calculateAggregateMetrics(queryMetrics);
        // Run validation gates
        const validationReport = this.runValidationGates(systemResults.system_id, queryMetrics, validationGates);
        // Generate pool membership report
        const poolMembershipReport = this.generatePoolMembershipReport(systemResults, pooledQrels);
        return {
            system_id: systemResults.system_id,
            query_metrics: queryMetrics,
            aggregate_metrics: aggregateMetrics,
            validation_report: validationReport,
            pool_membership: poolMembershipReport
        };
    }
    /**
     * Build pooled qrels from multiple system results
     */
    buildPooledQrels(allSystemResults, topK = 50) {
        const pooledQrels = new Map();
        for (const systemResult of allSystemResults) {
            for (const queryData of systemResult.queries) {
                const queryId = queryData.query.query_id;
                // Filter SLA-compliant results and take top-K
                const slaResults = queryData.results
                    .filter(r => queryData.latency_ms <= this.config.sla_threshold_ms)
                    .slice(0, topK);
                if (!pooledQrels.has(queryId)) {
                    pooledQrels.set(queryId, {
                        query_id: queryId,
                        expected_spans: [],
                        expected_files: [],
                        contributing_systems: [],
                        pool_source: 'union_top_k'
                    });
                }
                const pool = pooledQrels.get(queryId);
                pool.contributing_systems.push(systemResult.system_id);
                // Add unique spans and files to pool
                for (const result of slaResults) {
                    const normalizedResult = this.normalizer.normalizeSearchResult(result);
                    // Add span if available
                    if (normalizedResult.line !== null && normalizedResult.col !== null) {
                        const spanExists = pool.expected_spans.some(span => span.repo === normalizedResult.repo &&
                            span.path === normalizedResult.path &&
                            span.line === normalizedResult.line &&
                            span.col === normalizedResult.col);
                        if (!spanExists) {
                            pool.expected_spans.push({
                                repo: normalizedResult.repo,
                                path: normalizedResult.path,
                                line: normalizedResult.line,
                                col: normalizedResult.col
                            });
                        }
                    }
                    // Add file
                    const fileExists = pool.expected_files.some(file => file.repo === normalizedResult.repo &&
                        file.path === normalizedResult.path);
                    if (!fileExists) {
                        pool.expected_files.push({
                            repo: normalizedResult.repo,
                            path: normalizedResult.path
                        });
                    }
                }
            }
        }
        return Array.from(pooledQrels.values());
    }
    /**
     * Hard validation gates to prevent regression
     */
    runValidationGates(systemId, queryMetrics, gates) {
        const report = {
            system_id: systemId,
            gates_passed: true,
            warnings: [],
            errors: [],
            gate_results: {}
        };
        if (!gates || queryMetrics.length < gates.min_queries_for_gate) {
            return report;
        }
        // Gate 1: median(|hits∩pooled_qrels|) > 0
        const medianHitsInPool = this.calculateMedian(queryMetrics.map(m => m.hits_in_pool));
        report.gate_results.median_hits_in_pool = medianHitsInPool;
        if (medianHitsInPool <= gates.min_median_hits_in_pool) {
            report.gates_passed = false;
            report.errors.push(`GATE FAIL: median(hits_in_pool) = ${medianHitsInPool} <= ${gates.min_median_hits_in_pool}. ` +
                'Likely cause: join mismatch, SLA mask error, or labels with span_coverage=0');
        }
        // Gate 2: mean(ndcg@10) should not be 0 or 1.0 on real suites
        const meanNdcg = queryMetrics.reduce((sum, m) => sum + m.ndcg_at_10, 0) / queryMetrics.length;
        report.gate_results.mean_ndcg_at_10 = meanNdcg;
        if (meanNdcg === 0) {
            report.gates_passed = false;
            report.errors.push(`GATE FAIL: mean(ndcg@10) = 0.0. ` +
                'Likely cause: SLA mask wrong, pool join failure, or evaluation bug');
        }
        if (meanNdcg >= gates.max_perfect_score_threshold) {
            report.warnings.push(`WARN: mean(ndcg@10) = ${meanNdcg.toFixed(3)} >= ${gates.max_perfect_score_threshold}. ` +
                'Check credit_histogram for file-only labels causing inflated scores');
        }
        // Gate 3: Credit distribution sanity check
        const totalCreditModes = queryMetrics.reduce((acc, m) => {
            acc.span += m.credit_histogram.span;
            acc.symbol += m.credit_histogram.symbol;
            acc.file += m.credit_histogram.file;
            return acc;
        }, { span: 0, symbol: 0, file: 0 });
        const totalCredits = totalCreditModes.span + totalCreditModes.symbol + totalCreditModes.file;
        if (totalCredits > 0) {
            const fileRatio = totalCreditModes.file / totalCredits;
            report.gate_results.file_credit_ratio = fileRatio;
            if (fileRatio > 0.8) {
                report.warnings.push(`WARN: ${(fileRatio * 100).toFixed(1)}% of credits from file-level matches. ` +
                    'Consider improving span coverage in labels');
            }
        }
        return report;
    }
    generatePoolMembershipReport(systemResults, pooledQrels) {
        if (!pooledQrels) {
            return {
                system_id: systemResults.system_id,
                queries_in_pool: 0,
                total_queries: systemResults.queries.length,
                pool_coverage: 0
            };
        }
        const systemQueryIds = new Set(systemResults.queries.map(q => q.query.query_id));
        const poolQueryIds = new Set(pooledQrels.map(p => p.query_id));
        const queriesInPool = Array.from(systemQueryIds).filter(id => poolQueryIds.has(id)).length;
        return {
            system_id: systemResults.system_id,
            queries_in_pool: queriesInPool,
            total_queries: systemResults.queries.length,
            pool_coverage: systemResults.queries.length > 0 ? queriesInPool / systemResults.queries.length : 0
        };
    }
    calculateAggregateMetrics(queryMetrics) {
        if (queryMetrics.length === 0) {
            return {
                mean_ndcg_at_10: 0,
                mean_success_at_10: 0,
                mean_recall_at_50: 0,
                mean_precision_at_10: 0,
                mean_map: 0,
                median_latency_ms: 0,
                p95_latency_ms: 0,
                p99_latency_ms: 0,
                sla_compliance_rate: 0,
                total_queries: 0,
                span_coverage_avg: 0,
                credit_distribution: { span: 0, symbol: 0, file: 0 }
            };
        }
        const latencies = queryMetrics.map(m => m.latency_ms).sort((a, b) => a - b);
        return {
            mean_ndcg_at_10: queryMetrics.reduce((sum, m) => sum + m.ndcg_at_10, 0) / queryMetrics.length,
            mean_success_at_10: queryMetrics.reduce((sum, m) => sum + m.success_at_10, 0) / queryMetrics.length,
            mean_recall_at_50: queryMetrics.reduce((sum, m) => sum + m.recall_at_50, 0) / queryMetrics.length,
            mean_precision_at_10: queryMetrics.reduce((sum, m) => sum + m.precision_at_10, 0) / queryMetrics.length,
            mean_map: queryMetrics.reduce((sum, m) => sum + m.map, 0) / queryMetrics.length,
            median_latency_ms: this.calculatePercentile(latencies, 50),
            p95_latency_ms: this.calculatePercentile(latencies, 95),
            p99_latency_ms: this.calculatePercentile(latencies, 99),
            sla_compliance_rate: queryMetrics.filter(m => m.latency_ms <= this.config.sla_threshold_ms).length / queryMetrics.length,
            total_queries: queryMetrics.length,
            span_coverage_avg: queryMetrics.reduce((sum, m) => sum + m.span_coverage_in_labels, 0) / queryMetrics.length,
            credit_distribution: queryMetrics.reduce((acc, m) => {
                acc.span += m.credit_histogram.span;
                acc.symbol += m.credit_histogram.symbol;
                acc.file += m.credit_histogram.file;
                return acc;
            }, { span: 0, symbol: 0, file: 0 })
        };
    }
    calculateMedian(values) {
        const sorted = values.slice().sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        return sorted.length % 2 === 0
            ? (sorted[mid - 1] + sorted[mid]) / 2
            : sorted[mid];
    }
    calculatePercentile(sortedValues, percentile) {
        const index = (percentile / 100) * (sortedValues.length - 1);
        const lower = Math.floor(index);
        const upper = Math.ceil(index);
        if (lower === upper) {
            return sortedValues[lower];
        }
        const weight = index - lower;
        return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
    }
}
// Interfaces exported from types.js
