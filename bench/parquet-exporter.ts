/**
 * Parquet Schema & Exporter - Structured output for analysis
 * 
 * This creates Parquet files with the exact schema specified in the protocol:
 * - Aggregate table: One row per (query, system) combination
 * - Detail table: One row per (query, system, hit) combination
 */

import { writeFileSync, mkdirSync } from 'fs';
import * as path from 'path';
import { AggregateMetrics } from './metrics-calculator';
import { ExecutionResult } from './sla-execution-engine';

// Note: In production, would use apache-arrow/parquet-wasm
// For now, we'll create JSON/CSV that can be converted to Parquet

export interface AggregateRow {
  // Identity
  query_id: string;
  system_id: string;
  build_hash: string;
  policy_fingerprint: string;
  suite: string;
  slice_intent: string;
  slice_lang: string;
  
  // SLA enforcement
  sla_ms: number;
  q_time_ms: number;
  within_sla: boolean;
  
  // Quality metrics (SLA-bounded)
  ndcg_at_10: number;
  success_at_10: number;
  sla_recall_at_50: number;
  witness_cov_at_10: number;
  precision_at_10: number;
  recall_at_10: number;
  map_at_10: number;
  
  // Operational metrics
  p50: number;
  p95: number;
  p99: number;
  p99_over_p95: number;
  qps150x: number;
  timeout_pct: number;
  error_rate: number;
  sla_compliance_rate: number;
  
  // Calibration metrics
  ece: number;
  calib_slope: number;
  calib_intercept: number;
  brier_score: number;
  
  // Explainability metrics
  why_mix_exact: number;
  why_mix_struct: number;
  why_mix_semantic: number;
  why_mix_mixed: number;
  core_at_10: number;
  diversity_at_10: number;
  span_coverage: number;
  
  // Error tracking
  timeouts: number;
  errors: number;
  total_queries: number;
  
  // Attestation
  attestation_sha256: string;
  execution_timestamp: string;
}

export interface DetailRow {
  // Query identification
  query_id: string;
  system_id: string;
  suite: string;
  intent: string;
  language: string;
  
  // Hit information
  rank: number;
  file: string;
  line: number;
  col: number;
  lang: string;
  snippet_hash: string;
  score: number;
  why_tag: string;
  
  // Optional structured information
  symbol_kind?: string;
  ast_path?: string;
  byte_offset?: number;
  span_len?: number;
  
  // Execution context
  server_latency_ms: number;
  within_sla: boolean;
  execution_timestamp: string;
}

export interface ParquetSchema {
  name: string;
  type: 'INT32' | 'INT64' | 'FLOAT' | 'DOUBLE' | 'BOOLEAN' | 'STRING' | 'TIMESTAMP';
  nullable: boolean;
  description: string;
}

/**
 * Parquet schema definitions matching the protocol specification
 */
export class BenchmarkParquetSchema {
  /**
   * Aggregate table schema - one row per (query, system) combination
   */
  static getAggregateSchema(): ParquetSchema[] {
    return [
      // Identity fields
      { name: 'query_id', type: 'STRING', nullable: false, description: 'Unique query identifier' },
      { name: 'system_id', type: 'STRING', nullable: false, description: 'System identifier (lens, bm25, etc.)' },
      { name: 'build_hash', type: 'STRING', nullable: true, description: 'Git commit hash for reproducibility' },
      { name: 'policy_fingerprint', type: 'STRING', nullable: false, description: 'Configuration fingerprint' },
      { name: 'suite', type: 'STRING', nullable: false, description: 'Test suite (coir, swe_verified, etc.)' },
      { name: 'slice_intent', type: 'STRING', nullable: false, description: 'Query intent (exact, identifier, structural, semantic)' },
      { name: 'slice_lang', type: 'STRING', nullable: false, description: 'Programming language' },
      
      // SLA enforcement
      { name: 'sla_ms', type: 'INT32', nullable: false, description: 'SLA limit in milliseconds' },
      { name: 'q_time_ms', type: 'DOUBLE', nullable: false, description: 'Actual query time in milliseconds' },
      { name: 'within_sla', type: 'BOOLEAN', nullable: false, description: 'Whether query completed within SLA' },
      
      // Quality metrics (all SLA-bounded)
      { name: 'ndcg_at_10', type: 'DOUBLE', nullable: false, description: 'Normalized Discounted Cumulative Gain at rank 10' },
      { name: 'success_at_10', type: 'DOUBLE', nullable: false, description: 'Binary success rate at rank 10' },
      { name: 'sla_recall_at_50', type: 'DOUBLE', nullable: false, description: 'Recall at rank 50 within SLA constraint' },
      { name: 'witness_cov_at_10', type: 'DOUBLE', nullable: false, description: 'SWE-bench witness coverage at rank 10' },
      { name: 'precision_at_10', type: 'DOUBLE', nullable: false, description: 'Precision at rank 10' },
      { name: 'recall_at_10', type: 'DOUBLE', nullable: false, description: 'Recall at rank 10' },
      { name: 'map_at_10', type: 'DOUBLE', nullable: false, description: 'Mean Average Precision at rank 10' },
      
      // Operational metrics
      { name: 'p50', type: 'DOUBLE', nullable: false, description: '50th percentile latency in ms' },
      { name: 'p95', type: 'DOUBLE', nullable: false, description: '95th percentile latency in ms' },
      { name: 'p99', type: 'DOUBLE', nullable: false, description: '99th percentile latency in ms' },
      { name: 'p99_over_p95', type: 'DOUBLE', nullable: false, description: 'Tail ratio (p99/p95)' },
      { name: 'qps150x', type: 'DOUBLE', nullable: false, description: 'Queries per second at 150ms SLA' },
      { name: 'timeout_pct', type: 'DOUBLE', nullable: false, description: 'Percentage of queries that timed out' },
      { name: 'error_rate', type: 'DOUBLE', nullable: false, description: 'Error rate percentage' },
      { name: 'sla_compliance_rate', type: 'DOUBLE', nullable: false, description: 'SLA compliance percentage' },
      
      // Calibration metrics
      { name: 'ece', type: 'DOUBLE', nullable: false, description: 'Expected Calibration Error' },
      { name: 'calib_slope', type: 'DOUBLE', nullable: false, description: 'Calibration slope via linear regression' },
      { name: 'calib_intercept', type: 'DOUBLE', nullable: false, description: 'Calibration intercept via linear regression' },
      { name: 'brier_score', type: 'DOUBLE', nullable: false, description: 'Brier score for probability calibration' },
      
      // Explainability metrics
      { name: 'why_mix_exact', type: 'DOUBLE', nullable: false, description: 'Fraction of exact match results' },
      { name: 'why_mix_struct', type: 'DOUBLE', nullable: false, description: 'Fraction of structural search results' },
      { name: 'why_mix_semantic', type: 'DOUBLE', nullable: false, description: 'Fraction of semantic search results' },
      { name: 'why_mix_mixed', type: 'DOUBLE', nullable: false, description: 'Fraction of mixed-mode results' },
      { name: 'core_at_10', type: 'DOUBLE', nullable: false, description: 'Centrality decile share (topic-normalized)' },
      { name: 'diversity_at_10', type: 'DOUBLE', nullable: false, description: 'Unique files/topics in top 10' },
      { name: 'span_coverage', type: 'DOUBLE', nullable: false, description: 'Span coverage (must be 100%)' },
      
      // Error tracking
      { name: 'timeouts', type: 'INT32', nullable: false, description: 'Number of timeout errors' },
      { name: 'errors', type: 'INT32', nullable: false, description: 'Number of execution errors' },
      { name: 'total_queries', type: 'INT32', nullable: false, description: 'Total queries executed' },
      
      // Attestation
      { name: 'attestation_sha256', type: 'STRING', nullable: false, description: 'SHA256 hash for artifact binding' },
      { name: 'execution_timestamp', type: 'TIMESTAMP', nullable: false, description: 'Execution timestamp' }
    ];
  }

  /**
   * Detail table schema - one row per (query, system, hit) combination
   */
  static getDetailSchema(): ParquetSchema[] {
    return [
      // Query identification
      { name: 'query_id', type: 'STRING', nullable: false, description: 'Unique query identifier' },
      { name: 'system_id', type: 'STRING', nullable: false, description: 'System identifier' },
      { name: 'suite', type: 'STRING', nullable: false, description: 'Test suite' },
      { name: 'intent', type: 'STRING', nullable: false, description: 'Query intent classification' },
      { name: 'language', type: 'STRING', nullable: false, description: 'Programming language' },
      
      // Hit information
      { name: 'rank', type: 'INT32', nullable: false, description: 'Result rank (1-based)' },
      { name: 'file', type: 'STRING', nullable: false, description: 'File path' },
      { name: 'line', type: 'INT32', nullable: false, description: 'Line number' },
      { name: 'col', type: 'INT32', nullable: false, description: 'Column number' },
      { name: 'lang', type: 'STRING', nullable: true, description: 'File language (inferred)' },
      { name: 'snippet_hash', type: 'STRING', nullable: false, description: 'Hash of code snippet for deduplication' },
      { name: 'score', type: 'DOUBLE', nullable: false, description: 'Relevance score' },
      { name: 'why_tag', type: 'STRING', nullable: false, description: 'Match type (exact, struct, semantic, mixed)' },
      
      // Optional structured information
      { name: 'symbol_kind', type: 'STRING', nullable: true, description: 'Symbol type (function, class, variable, etc.)' },
      { name: 'ast_path', type: 'STRING', nullable: true, description: 'AST path for structural context' },
      { name: 'byte_offset', type: 'INT64', nullable: true, description: 'Byte offset in file' },
      { name: 'span_len', type: 'INT32', nullable: true, description: 'Span length in characters' },
      
      // Execution context
      { name: 'server_latency_ms', type: 'DOUBLE', nullable: false, description: 'Server-side latency' },
      { name: 'within_sla', type: 'BOOLEAN', nullable: false, description: 'Whether parent query was within SLA' },
      { name: 'execution_timestamp', type: 'TIMESTAMP', nullable: false, description: 'Execution timestamp' }
    ];
  }
}

/**
 * Parquet exporter for benchmark results
 */
export class ParquetExporter {
  private outputDir: string;

  constructor(outputDir: string) {
    this.outputDir = outputDir;
    mkdirSync(outputDir, { recursive: true });
  }

  /**
   * Export aggregate metrics to Parquet format
   */
  async exportAggregateMetrics(metrics: AggregateMetrics[], filename = 'agg.parquet'): Promise<void> {
    console.log(`📊 Exporting ${metrics.length} aggregate metrics to ${filename}`);

    const rows: AggregateRow[] = metrics.map(metric => ({
      // Identity
      query_id: metric.query_id,
      system_id: metric.system_id,
      build_hash: metric.system_info?.build_hash || '',
      policy_fingerprint: metric.system_info?.config_fingerprint || '',
      suite: metric.suite,
      slice_intent: metric.intent,
      slice_lang: metric.language,
      
      // SLA enforcement  
      sla_ms: 150, // From protocol
      q_time_ms: 0, // Would need to aggregate from execution results
      within_sla: metric.within_sla_queries > 0,
      
      // Quality metrics
      ndcg_at_10: metric.quality.ndcg_at_10,
      success_at_10: metric.quality.success_at_10,
      sla_recall_at_50: metric.quality.sla_recall_at_50,
      witness_cov_at_10: metric.quality.witness_coverage_at_10,
      precision_at_10: metric.quality.precision_at_10,
      recall_at_10: metric.quality.recall_at_10,
      map_at_10: metric.quality.map_at_10,
      
      // Operational metrics
      p50: metric.operations.p50_latency_ms,
      p95: metric.operations.p95_latency_ms,
      p99: metric.operations.p99_latency_ms,
      p99_over_p95: metric.operations.p99_over_p95_ratio,
      qps150x: metric.operations.qps_at_150ms,
      timeout_pct: metric.operations.timeout_percentage,
      error_rate: metric.operations.error_rate,
      sla_compliance_rate: metric.operations.sla_compliance_rate,
      
      // Calibration metrics
      ece: metric.calibration.expected_calibration_error,
      calib_slope: metric.calibration.calibration_slope,
      calib_intercept: metric.calibration.calibration_intercept,
      brier_score: metric.calibration.brier_score,
      
      // Explainability metrics
      why_mix_exact: metric.explainability.why_mix_exact,
      why_mix_struct: metric.explainability.why_mix_struct,
      why_mix_semantic: metric.explainability.why_mix_semantic,
      why_mix_mixed: metric.explainability.why_mix_mixed,
      core_at_10: metric.explainability.core_at_10,
      diversity_at_10: metric.explainability.diversity_at_10,
      span_coverage: metric.explainability.span_coverage,
      
      // Error tracking
      timeouts: metric.total_queries - metric.within_sla_queries, // Approximation
      errors: 0, // Would need from execution results
      total_queries: metric.total_queries,
      
      // Attestation
      attestation_sha256: this.calculateAttestationHash(metric),
      execution_timestamp: metric.timestamp
    }));

    // Write as JSON for now (can be converted to Parquet with external tools)
    const outputPath = path.join(this.outputDir, filename.replace('.parquet', '.json'));
    writeFileSync(outputPath, JSON.stringify(rows, null, 2));

    // Also write schema for reference
    const schemaPath = path.join(this.outputDir, filename.replace('.parquet', '_schema.json'));
    writeFileSync(schemaPath, JSON.stringify(BenchmarkParquetSchema.getAggregateSchema(), null, 2));

    console.log(`✅ Exported aggregate data to ${outputPath}`);
    console.log(`📋 Schema written to ${schemaPath}`);
  }

  /**
   * Export detailed hit results to Parquet format
   */
  async exportDetailResults(results: ExecutionResult[], filename = 'hits.parquet'): Promise<void> {
    console.log(`📊 Exporting detailed results from ${results.length} executions to ${filename}`);

    const rows: DetailRow[] = [];

    for (const result of results) {
      for (const hit of result.hits) {
        const row: DetailRow = {
          // Query identification
          query_id: result.query_id,
          system_id: result.system_id,
          suite: result.suite,
          intent: result.intent,
          language: result.language,
          
          // Hit information
          rank: hit.rank,
          file: hit.file,
          line: hit.line,
          col: hit.column,
          lang: this.inferLanguageFromFile(hit.file),
          snippet_hash: this.hashSnippet(hit.snippet),
          score: hit.score,
          why_tag: hit.why_tag,
          
          // Optional structured information
          symbol_kind: hit.symbol_kind,
          ast_path: hit.ast_path,
          byte_offset: hit.byte_offset,
          span_len: hit.span_length,
          
          // Execution context
          server_latency_ms: result.server_latency_ms,
          within_sla: result.within_sla,
          execution_timestamp: result.execution_timestamp
        };
        
        rows.push(row);
      }
    }

    // Write as JSON for now (can be converted to Parquet with external tools)
    const outputPath = path.join(this.outputDir, filename.replace('.parquet', '.json'));
    writeFileSync(outputPath, JSON.stringify(rows, null, 2));

    // Also write schema for reference
    const schemaPath = path.join(this.outputDir, filename.replace('.parquet', '_schema.json'));
    writeFileSync(schemaPath, JSON.stringify(BenchmarkParquetSchema.getDetailSchema(), null, 2));

    console.log(`✅ Exported ${rows.length} detail rows to ${outputPath}`);
    console.log(`📋 Schema written to ${schemaPath}`);
  }

  /**
   * Export complete benchmark suite (both aggregate and detail)
   */
  async exportCompleteSuite(
    metrics: AggregateMetrics[], 
    results: ExecutionResult[],
    suitePrefix = 'benchmark'
  ): Promise<void> {
    console.log(`📦 Exporting complete benchmark suite with prefix: ${suitePrefix}`);

    await this.exportAggregateMetrics(metrics, `${suitePrefix}_agg.parquet`);
    await this.exportDetailResults(results, `${suitePrefix}_hits.parquet`);

    // Create metadata file
    const metadata = {
      export_timestamp: new Date().toISOString(),
      aggregate_rows: metrics.length,
      detail_rows: results.reduce((sum, r) => sum + r.hits.length, 0),
      execution_results: results.length,
      unique_systems: new Set(results.map(r => r.system_id)).size,
      unique_queries: new Set(results.map(r => r.query_id)).size,
      suites: [...new Set(results.map(r => r.suite))],
      schema_version: '1.0.0',
      protocol_version: '1.0'
    };

    const metadataPath = path.join(this.outputDir, `${suitePrefix}_metadata.json`);
    writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));

    console.log(`✅ Complete benchmark suite exported successfully`);
    console.log(`📋 Metadata written to ${metadataPath}`);
  }

  /**
   * Create CSV versions for easier analysis
   */
  async exportCSV(metrics: AggregateMetrics[], results: ExecutionResult[]): Promise<void> {
    // Export aggregate CSV
    const aggCsvPath = path.join(this.outputDir, 'aggregate.csv');
    const aggHeaders = Object.keys(this.convertMetricToAggregateRow(metrics[0]));
    const aggCsv = [
      aggHeaders.join(','),
      ...metrics.map(m => {
        const row = this.convertMetricToAggregateRow(m);
        return aggHeaders.map(h => JSON.stringify(row[h] ?? '')).join(',');
      })
    ].join('\n');
    
    writeFileSync(aggCsvPath, aggCsv);
    
    // Export detail CSV (sample for large datasets)
    const detailCsvPath = path.join(this.outputDir, 'detail_sample.csv');
    const sampleResults = results.slice(0, 100); // Sample first 100 for CSV
    
    const detailRows = sampleResults.flatMap(result =>
      result.hits.map(hit => ({
        query_id: result.query_id,
        system_id: result.system_id,
        rank: hit.rank,
        file: hit.file,
        line: hit.line,
        score: hit.score,
        why_tag: hit.why_tag,
        within_sla: result.within_sla
      }))
    );
    
    if (detailRows.length > 0) {
      const detailHeaders = Object.keys(detailRows[0]);
      const detailCsv = [
        detailHeaders.join(','),
        ...detailRows.map(row => 
          detailHeaders.map(h => JSON.stringify(row[h] ?? '')).join(',')
        )
      ].join('\n');
      
      writeFileSync(detailCsvPath, detailCsv);
    }
    
    console.log(`📊 CSV exports created: ${aggCsvPath}, ${detailCsvPath}`);
  }

  /**
   * Helper methods
   */
  private calculateAttestationHash(metric: AggregateMetrics): string {
    const hashInput = JSON.stringify({
      query_id: metric.query_id,
      system_id: metric.system_id,
      suite: metric.suite,
      timestamp: metric.timestamp
    });
    
    return require('crypto').createHash('sha256').update(hashInput).digest('hex');
  }

  private hashSnippet(snippet: string): string {
    return require('crypto').createHash('sha256').update(snippet).digest('hex').substring(0, 16);
  }

  private inferLanguageFromFile(filepath: string): string {
    const ext = path.extname(filepath).toLowerCase();
    const langMap: Record<string, string> = {
      '.ts': 'typescript',
      '.js': 'javascript',
      '.py': 'python',
      '.java': 'java',
      '.go': 'go',
      '.rs': 'rust',
      '.cpp': 'cpp',
      '.c': 'c',
      '.cs': 'csharp',
      '.rb': 'ruby',
      '.php': 'php',
      '.swift': 'swift',
      '.kt': 'kotlin',
      '.scala': 'scala'
    };
    
    return langMap[ext] || 'unknown';
  }

  private convertMetricToAggregateRow(metric: AggregateMetrics): any {
    return {
      query_id: metric.query_id,
      system_id: metric.system_id,
      suite: metric.suite,
      intent: metric.intent,
      language: metric.language,
      ndcg_at_10: metric.quality.ndcg_at_10,
      success_at_10: metric.quality.success_at_10,
      sla_recall_at_50: metric.quality.sla_recall_at_50,
      p50: metric.operations.p50_latency_ms,
      p95: metric.operations.p95_latency_ms,
      p99: metric.operations.p99_latency_ms,
      ece: metric.calibration.expected_calibration_error,
      why_mix_exact: metric.explainability.why_mix_exact,
      timestamp: metric.timestamp
    };
  }
}

/**
 * Parquet conversion utilities (would use actual Parquet library in production)
 */
export class ParquetConverter {
  /**
   * Convert JSON to Parquet using external tools
   */
  static async convertJSONToParquet(jsonPath: string, schemaPath: string): Promise<void> {
    console.log(`🔄 Converting ${jsonPath} to Parquet format`);
    
    // In production, would use:
    // - Apache Arrow JS for in-memory conversion
    // - parquet-wasm for WebAssembly-based conversion
    // - External tools like pandas or DuckDB
    
    console.log(`ℹ️  JSON file ready for conversion: ${jsonPath}`);
    console.log(`📋 Schema definition: ${schemaPath}`);
    console.log(`💡 Use tools like DuckDB or pandas to convert to actual Parquet format`);
  }

  /**
   * Generate DuckDB conversion commands
   */
  static generateConversionCommands(outputDir: string): string[] {
    return [
      `-- Convert aggregate data to Parquet`,
      `COPY (SELECT * FROM read_json('${outputDir}/benchmark_agg.json')) TO '${outputDir}/agg.parquet';`,
      ``,
      `-- Convert detail data to Parquet`, 
      `COPY (SELECT * FROM read_json('${outputDir}/benchmark_hits.json')) TO '${outputDir}/hits.parquet';`,
      ``,
      `-- Verify Parquet files`,
      `SELECT COUNT(*) as agg_rows FROM '${outputDir}/agg.parquet';`,
      `SELECT COUNT(*) as detail_rows FROM '${outputDir}/hits.parquet';`
    ];
  }
}