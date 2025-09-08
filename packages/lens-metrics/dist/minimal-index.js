/**
 * Minimal @lens/metrics export - JavaScript version for immediate use
 */

// Simple data migrator
export class DataMigrator {
  static migrateQuery(legacy, defaultRepo = '') {
    const queryId = legacy.query_id || legacy.id || `query_${Math.random().toString(36).slice(2)}`;
    const suite = legacy.suite || 'unknown';
    const lang = this.normalizeLang(legacy.language || legacy.lang || 'unknown');
    
    let expectedFiles = [];
    let expectedSpans = [];
    
    if (legacy.expected_results && Array.isArray(legacy.expected_results)) {
      for (const result of legacy.expected_results) {
        if (typeof result === 'string') {
          expectedFiles.push({
            repo: defaultRepo,
            path: result
          });
        } else if (result && typeof result === 'object') {
          const path = result.path || result.file || result.filename || '';
          if (path) {
            if (result.line !== undefined && result.col !== undefined) {
              expectedSpans.push({
                repo: result.repo || defaultRepo,
                path,
                line: Number(result.line),
                col: Number(result.col)
              });
            } else {
              expectedFiles.push({
                repo: result.repo || defaultRepo,
                path
              });
            }
          }
        }
      }
    }
    
    if (legacy.expected_files && Array.isArray(legacy.expected_files)) {
      for (const file of legacy.expected_files) {
        if (typeof file === 'string') {
          expectedFiles.push({
            repo: defaultRepo,
            path: file
          });
        }
      }
    }
    
    return {
      query_id: queryId,
      expected_spans: expectedSpans.length > 0 ? expectedSpans : undefined,
      expected_files: expectedFiles,
      credit_policy: 'hierarchical',
      metadata: {
        suite: this.normalizeSuite(suite),
        lang: lang,
        intent: legacy.intent || 'unknown',
        query_type: legacy.query_type || 'lexical'
      }
    };
  }

  static normalizeLang(lang) {
    const normalized = lang.toLowerCase();
    switch (normalized) {
      case 'python': return 'py';
      case 'typescript': return 'ts';
      case 'javascript': return 'js';
      case 'golang': return 'go';
      default: return 'py';
    }
  }

  static normalizeSuite(suite) {
    const normalized = suite.toLowerCase();
    switch (normalized) {
      case 'coir': return 'coir';
      case 'swe_verified': case 'swe-verified': case 'swebench': return 'swe_verified';
      case 'csn': case 'codesearchnet': return 'csn';
      case 'cosqa': return 'cosqa';
      default: return 'coir';
    }
  }
}

// Simple metrics engine
export class LensMetricsEngine {
  constructor(config = {}) {
    this.config = {
      sla_threshold_ms: 150,
      credit_gains: {
        span: 1.0,
        symbol: 0.7,
        file: 0.5
      },
      ...config
    };
  }

  evaluateSystem(systemResults, pooledQrels, validationGates) {
    const queryMetrics = [];
    
    for (const queryData of systemResults.queries) {
      const metrics = this.scoreQuery(queryData.query, queryData.results, queryData.latency_ms);
      queryMetrics.push(metrics);
    }
    
    const aggregateMetrics = this.calculateAggregateMetrics(queryMetrics);
    const validationReport = this.runValidationGates(systemResults.system_id, queryMetrics, validationGates);
    
    return {
      system_id: systemResults.system_id,
      query_metrics: queryMetrics,
      aggregate_metrics: aggregateMetrics,
      validation_report: validationReport,
      pool_membership: {
        system_id: systemResults.system_id,
        queries_in_pool: queryMetrics.length,
        total_queries: queryMetrics.length,
        pool_coverage: 1.0
      }
    };
  }

  scoreQuery(query, searchResults, latencyMs) {
    // Simplified hierarchical scoring
    const slaResults = searchResults.filter(r => latencyMs <= this.config.sla_threshold_ms);
    
    let relevantResults = 0;
    let totalGain = 0;
    const creditHistogram = { span: 0, symbol: 0, file: 0 };
    
    for (const result of slaResults.slice(0, 10)) {
      let gain = 0;
      let creditMode = 'file';
      
      // Check for exact span match
      if (query.expected_spans) {
        const spanMatch = query.expected_spans.some(span => 
          span.repo === result.repo && 
          span.path === result.path &&
          span.line === result.line &&
          span.col === result.col
        );
        if (spanMatch) {
          gain = this.config.credit_gains.span;
          creditMode = 'span';
        }
      }
      
      // Fall back to file match
      if (gain === 0) {
        const fileMatch = query.expected_files.some(file =>
          file.repo === result.repo && file.path === result.path
        );
        if (fileMatch) {
          gain = this.config.credit_gains.file;
          creditMode = 'file';
        }
      }
      
      if (gain > 0) {
        relevantResults++;
        totalGain += gain;
        creditHistogram[creditMode]++;
      }
    }
    
    // Calculate nDCG (simplified)
    const dcg = totalGain / Math.log2(2); // Simplified DCG
    // Calculate IDCG based on total expected items (spans + files)
    const totalExpectedItems = (query.expected_spans?.length || 0) + query.expected_files.length;
    const idcg = Math.min(totalExpectedItems, 10) * this.config.credit_gains.span / Math.log2(2);
    const ndcg = idcg > 0 ? dcg / idcg : 0;
    
    return {
      ndcg_at_10: Math.min(1.0, ndcg),
      success_at_10: relevantResults > 0 ? 1 : 0,
      recall_at_50: Math.min(1.0, relevantResults / Math.max(1, query.expected_files.length)),
      precision_at_10: relevantResults / Math.min(10, slaResults.length),
      map: relevantResults > 0 ? 1.0 : 0.0,
      credit_mode_used: Array(relevantResults).fill('file'),
      span_coverage_in_labels: query.expected_spans ? 
        (query.expected_spans.length / (query.expected_spans.length + query.expected_files.length)) * 100 : 0,
      credit_histogram: creditHistogram,
      hits_in_pool: relevantResults,
      total_hits: slaResults.length,
      sla_compliant_hits: slaResults.length,
      latency_ms: latencyMs
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
      median_latency_ms: latencies[Math.floor(latencies.length / 2)],
      p95_latency_ms: latencies[Math.floor(latencies.length * 0.95)],
      p99_latency_ms: latencies[Math.floor(latencies.length * 0.99)],
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

  runValidationGates(systemId, queryMetrics, gates) {
    const report = {
      system_id: systemId,
      gates_passed: true,
      warnings: [],
      errors: [],
      gate_results: {}
    };
    
    if (!gates || queryMetrics.length < (gates.min_queries_for_gate || 10)) {
      return report;
    }
    
    const medianHitsInPool = this.calculateMedian(queryMetrics.map(m => m.hits_in_pool));
    const meanNdcg = queryMetrics.reduce((sum, m) => sum + m.ndcg_at_10, 0) / queryMetrics.length;
    
    report.gate_results.median_hits_in_pool = medianHitsInPool;
    report.gate_results.mean_ndcg_at_10 = meanNdcg;
    
    if (medianHitsInPool <= (gates.min_median_hits_in_pool || 0)) {
      report.gates_passed = false;
      report.errors.push(
        `GATE FAIL: median(hits_in_pool) = ${medianHitsInPool} <= ${gates.min_median_hits_in_pool || 0}`
      );
    }
    
    if (meanNdcg === 0) {
      report.gates_passed = false;
      report.errors.push(`GATE FAIL: mean(ndcg@10) = 0.0`);
    }
    
    if (meanNdcg >= (gates.max_perfect_score_threshold || 0.95)) {
      report.warnings.push(`WARN: mean(ndcg@10) = ${meanNdcg.toFixed(3)} >= ${gates.max_perfect_score_threshold || 0.95}`);
    }
    
    return report;
  }

  calculateMedian(values) {
    const sorted = values.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 
      ? (sorted[mid - 1] + sorted[mid]) / 2
      : sorted[mid];
  }
}

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

export const DEFAULT_VALIDATION_GATES = {
  sla_threshold_ms: 150,
  min_queries_for_gate: 10,
  min_median_hits_in_pool: 0,
  max_perfect_score_threshold: 0.95
};