/**
 * LSP SPI Metrics Collection
 * Implements observability metrics: lsp_diag_latency_ms, lsp_format_idempotent_rate, lsp_cache_hit_ratio, lsp_timeout_rate
 */

import { LensTracer } from '../telemetry/tracer.js';
import { globalLSPCache } from './cache.js';

interface LSPOperationMetrics {
  operation: string;
  duration_ms: number;
  success: boolean;
  language?: string;
  cache_hit?: boolean;
  timed_out?: boolean;
  idempotent?: boolean; // for format operations
  diagnostics_count?: number; // for diagnostics operations
  edits_count?: number; // for format/rename operations
}

class LSPMetricsCollector {
  private operationCounts = new Map<string, number>();
  private operationDurations = new Map<string, number[]>();
  private timeoutCounts = new Map<string, number>();
  private idempotentCounts = new Map<string, { total: number; idempotent: number }>();

  /**
   * Record metrics for an LSP operation
   */
  recordOperation(metrics: LSPOperationMetrics): void {
    const span = LensTracer.createChildSpan('lsp_metrics_record');
    
    try {
      const operation = metrics.operation;
      
      // Count total operations
      this.operationCounts.set(operation, (this.operationCounts.get(operation) || 0) + 1);
      
      // Track duration
      if (!this.operationDurations.has(operation)) {
        this.operationDurations.set(operation, []);
      }
      this.operationDurations.get(operation)!.push(metrics.duration_ms);
      
      // Track timeouts
      if (metrics.timed_out) {
        this.timeoutCounts.set(operation, (this.timeoutCounts.get(operation) || 0) + 1);
      }
      
      // Track format idempotence
      if (metrics.operation === 'format' && metrics.idempotent !== undefined) {
        if (!this.idempotentCounts.has('format')) {
          this.idempotentCounts.set('format', { total: 0, idempotent: 0 });
        }
        const counts = this.idempotentCounts.get('format')!;
        counts.total++;
        if (metrics.idempotent) {
          counts.idempotent++;
        }
      }
      
      span.setAttributes({
        success: true,
        operation: metrics.operation,
        duration_ms: metrics.duration_ms,
        language: metrics.language || 'unknown',
        cache_hit: metrics.cache_hit || false,
        timed_out: metrics.timed_out || false,
      });
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error);
      span.setAttributes({ success: false, error: errorMsg });
      console.warn('Failed to record LSP metrics:', error);
    } finally {
      span.end();
    }
  }

  /**
   * Get LSP diagnostics latency metrics by language
   */
  getLSPDiagLatencyMs(): Record<string, { 
    count: number; 
    avg_ms: number; 
    p50_ms: number; 
    p95_ms: number; 
    p99_ms: number; 
  }> {
    const durations = this.operationDurations.get('diagnostics') || [];
    if (durations.length === 0) {
      return { all: { count: 0, avg_ms: 0, p50_ms: 0, p95_ms: 0, p99_ms: 0 } };
    }

    const sorted = [...durations].sort((a, b) => a - b);
    const sum = durations.reduce((a, b) => a + b, 0);
    
    return {
      all: {
        count: durations.length,
        avg_ms: Math.round(sum / durations.length),
        p50_ms: Math.round(this.percentile(sorted, 50)),
        p95_ms: Math.round(this.percentile(sorted, 95)),
        p99_ms: Math.round(this.percentile(sorted, 99)),
      }
    };
  }

  /**
   * Get LSP format idempotent rate
   */
  getLSPFormatIdempotentRate(): number {
    const counts = this.idempotentCounts.get('format');
    if (!counts || counts.total === 0) {
      return 0;
    }
    return counts.idempotent / counts.total;
  }

  /**
   * Get LSP cache hit ratio from cache manager
   */
  getLSPCacheHitRatio(): number {
    const stats = globalLSPCache.getStats();
    return stats.hit_ratio;
  }

  /**
   * Get LSP timeout rate by operation
   */
  getLSPTimeoutRate(): Record<string, number> {
    const rates: Record<string, number> = {};
    
    for (const [operation, totalCount] of this.operationCounts) {
      const timeoutCount = this.timeoutCounts.get(operation) || 0;
      rates[operation] = totalCount > 0 ? timeoutCount / totalCount : 0;
    }
    
    return rates;
  }

  /**
   * Get comprehensive LSP metrics summary
   */
  getMetricsSummary(): {
    operations: Record<string, { count: number; avg_duration_ms: number }>;
    diagnostics_latency: Record<string, { count: number; avg_ms: number; p95_ms: number }>;
    format_idempotent_rate: number;
    cache_hit_ratio: number;
    timeout_rates: Record<string, number>;
    cache_stats: any;
  } {
    const operations: Record<string, { count: number; avg_duration_ms: number }> = {};
    
    // Calculate average durations
    for (const [operation, count] of this.operationCounts) {
      const durations = this.operationDurations.get(operation) || [];
      const avgDuration = durations.length > 0 
        ? Math.round(durations.reduce((a, b) => a + b, 0) / durations.length)
        : 0;
      
      operations[operation] = {
        count,
        avg_duration_ms: avgDuration
      };
    }

    return {
      operations,
      diagnostics_latency: this.getLSPDiagLatencyMs(),
      format_idempotent_rate: this.getLSPFormatIdempotentRate(),
      cache_hit_ratio: this.getLSPCacheHitRatio(),
      timeout_rates: this.getLSPTimeoutRate(),
      cache_stats: globalLSPCache.getStats(),
    };
  }

  /**
   * Reset all metrics (useful for testing)
   */
  reset(): void {
    this.operationCounts.clear();
    this.operationDurations.clear();
    this.timeoutCounts.clear();
    this.idempotentCounts.clear();
    globalLSPCache.clear();
  }

  /**
   * Get Prometheus-style metrics for export
   */
  getPrometheusMetrics(): string {
    const lines: string[] = [];
    const timestamp = Date.now();

    // Operation counts
    lines.push('# HELP lsp_operations_total Total number of LSP operations by type');
    lines.push('# TYPE lsp_operations_total counter');
    for (const [operation, count] of this.operationCounts) {
      lines.push(`lsp_operations_total{operation="${operation}"} ${count} ${timestamp}`);
    }

    // Diagnostic latency
    const diagLatency = this.getLSPDiagLatencyMs();
    for (const [lang, stats] of Object.entries(diagLatency)) {
      lines.push('# HELP lsp_diag_latency_ms Diagnostic operation latency in milliseconds');
      lines.push('# TYPE lsp_diag_latency_ms histogram');
      lines.push(`lsp_diag_latency_ms{language="${lang}",quantile="0.5"} ${stats.p50_ms} ${timestamp}`);
      lines.push(`lsp_diag_latency_ms{language="${lang}",quantile="0.95"} ${stats.p95_ms} ${timestamp}`);
      lines.push(`lsp_diag_latency_ms{language="${lang}",quantile="0.99"} ${stats.p99_ms} ${timestamp}`);
    }

    // Format idempotent rate
    lines.push('# HELP lsp_format_idempotent_rate Rate of format operations that are idempotent');
    lines.push('# TYPE lsp_format_idempotent_rate gauge');
    lines.push(`lsp_format_idempotent_rate ${this.getLSPFormatIdempotentRate()} ${timestamp}`);

    // Cache hit ratio
    lines.push('# HELP lsp_cache_hit_ratio LSP cache hit ratio');
    lines.push('# TYPE lsp_cache_hit_ratio gauge');
    lines.push(`lsp_cache_hit_ratio ${this.getLSPCacheHitRatio()} ${timestamp}`);

    // Timeout rates
    lines.push('# HELP lsp_timeout_rate Rate of operations that timeout');
    lines.push('# TYPE lsp_timeout_rate gauge');
    const timeoutRates = this.getLSPTimeoutRate();
    for (const [operation, rate] of Object.entries(timeoutRates)) {
      lines.push(`lsp_timeout_rate{operation="${operation}"} ${rate} ${timestamp}`);
    }

    return lines.join('\n') + '\n';
  }

  private percentile(sorted: number[], p: number): number {
    if (sorted.length === 0) return 0;
    const index = (p / 100) * (sorted.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    const weight = index % 1;
    
    if (upper >= sorted.length) return sorted[sorted.length - 1];
    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
  }
}

// Global metrics collector
export const globalLSPMetrics = new LSPMetricsCollector();

/**
 * Middleware to automatically record metrics for LSP operations
 */
export function withLSPMetrics<T extends (...args: any[]) => Promise<any>>(
  operation: string,
  fn: T
): T {
  return (async (...args: any[]) => {
    const startTime = Date.now();
    let success = true;
    let timed_out = false;
    let additionalMetrics: Partial<LSPOperationMetrics> = {};

    try {
      const result = await fn(...args);
      
      // Extract additional metrics from result
      if (result && typeof result === 'object') {
        if ('timed_out' in result) timed_out = result.timed_out;
        if ('idempotent' in result) additionalMetrics.idempotent = result.idempotent;
        if ('diags' in result && Array.isArray(result.diags)) {
          additionalMetrics.diagnostics_count = result.diags.reduce(
            (sum: number, d: any) => sum + (d.items?.length || 0), 0
          );
        }
        if ('edits' in result && Array.isArray(result.edits)) {
          additionalMetrics.edits_count = result.edits.length;
        }
      }
      
      return result;
    } catch (error) {
      success = false;
      throw error;
    } finally {
      const duration_ms = Date.now() - startTime;
      
      globalLSPMetrics.recordOperation({
        operation,
        duration_ms,
        success,
        timed_out,
        ...additionalMetrics
      });
    }
  }) as T;
}