/**
 * Canonical Label Schema - All queries must use this normalized shape
 */
export interface ExpectedSpan {
    repo: string;
    path: string;
    line: number;
    col: number;
}
export interface ExpectedSymbol {
    repo: string;
    path: string;
    symbol: string;
}
export interface ExpectedFile {
    repo: string;
    path: string;
}
export type CreditPolicy = 'span_strict' | 'hierarchical';
export type CreditMode = 'span' | 'symbol' | 'file';
export interface SystemResults {
    system_id: string;
    queries: Array<{
        query: CanonicalQuery;
        results: SearchResult[];
        latency_ms: number;
    }>;
}
export interface PooledQrels {
    query_id: string;
    expected_spans: Array<{
        repo: string;
        path: string;
        line: number;
        col: number;
    }>;
    expected_files: Array<{
        repo: string;
        path: string;
    }>;
    contributing_systems: string[];
    pool_source: 'union_top_k' | 'manual';
}
export interface ValidationGates {
    sla_threshold_ms: number;
    min_queries_for_gate: number;
    min_median_hits_in_pool: number;
    max_perfect_score_threshold: number;
}
export interface AggregateMetrics {
    mean_ndcg_at_10: number;
    mean_success_at_10: number;
    mean_recall_at_50: number;
    mean_precision_at_10: number;
    mean_map: number;
    median_latency_ms: number;
    p95_latency_ms: number;
    p99_latency_ms: number;
    sla_compliance_rate: number;
    total_queries: number;
    span_coverage_avg: number;
    credit_distribution: {
        span: number;
        symbol: number;
        file: number;
    };
}
export interface ValidationReport {
    system_id: string;
    gates_passed: boolean;
    warnings: string[];
    errors: string[];
    gate_results: {
        median_hits_in_pool?: number;
        mean_ndcg_at_10?: number;
        file_credit_ratio?: number;
    };
}
export interface PoolMembershipReport {
    system_id: string;
    queries_in_pool: number;
    total_queries: number;
    pool_coverage: number;
}
export interface CanonicalQuery {
    query_id: string;
    expected_spans?: ExpectedSpan[];
    expected_symbols?: ExpectedSymbol[];
    expected_files: ExpectedFile[];
    credit_policy: CreditPolicy;
    metadata: {
        suite: 'coir' | 'swe_verified' | 'csn' | 'cosqa';
        lang: 'py' | 'ts' | 'js' | 'go' | 'rust' | 'java' | 'cpp';
        intent?: string;
        query_type?: 'lexical' | 'semantic';
    };
}
export interface SearchResult {
    repo: string;
    path: string;
    line?: number | null;
    col?: number | null;
    score: number;
    snippet?: string;
    snippet_hash?: string;
    rank: number;
    why_tag?: string;
}
export interface ScoredResult extends SearchResult {
    credit_mode_used: CreditMode;
    relevance_gain: number;
}
export interface QueryMetrics {
    ndcg_at_10: number;
    success_at_10: number;
    recall_at_50: number;
    precision_at_10: number;
    map: number;
    credit_mode_used: CreditMode[];
    span_coverage_in_labels: number;
    credit_histogram: {
        span: number;
        symbol: number;
        file: number;
    };
    hits_in_pool: number;
    total_hits: number;
    sla_compliant_hits: number;
    latency_ms: number;
}
export interface MetricsConfig {
    sla_threshold_ms: number;
    k_values: number[];
    credit_gains: {
        span: number;
        symbol: number;
        file: number;
    };
    normalization: {
        case_sensitive: boolean;
        path_separator: string;
        line_base: number;
        col_base: number;
    };
}
