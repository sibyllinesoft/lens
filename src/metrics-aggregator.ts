import { promises as fs } from 'fs';

export interface BenchmarkMetrics {
  // Query information
  query: string;
  k: number;
  timestamp: string;
  
  // Performance metrics
  latency_ms: {
    stage_a: number;
    stage_b?: number;
    stage_c?: number;
    total: number;
  };
  
  // Result metrics
  total_results: number;
  result_quality: {
    precision?: number;
    recall?: number;
    f1_score?: number;
    mrr?: number; // Mean Reciprocal Rank
  };
  
  // System metrics
  system_info: {
    cpu_usage?: number;
    memory_usage_mb?: number;
    active_queries?: number;
  };
  
  // Trace information
  trace_id?: string;
  
  // Stage-specific details
  stage_details?: {
    stage_a?: {
      candidates_found: number;
      processing_time_ms: number;
    };
    stage_b?: {
      candidates_found: number;
      processing_time_ms: number;
      symbols_matched: number;
    };
    stage_c?: {
      candidates_reranked: number;
      processing_time_ms: number;
      score_improvement: number;
    };
  };
}

export class MetricsAggregator {
  private metrics: BenchmarkMetrics[] = [];
  private outputFile: string;

  constructor(outputFile: string = 'benchmark-metrics.json') {
    this.outputFile = outputFile;
  }

  async recordMetric(metric: BenchmarkMetrics): Promise<void> {
    // Add timestamp if not provided
    if (!metric.timestamp) {
      metric.timestamp = new Date().toISOString();
    }
    
    this.metrics.push(metric);
    
    // Optionally write to file immediately for real-time monitoring
    await this.saveToFile();
  }

  async saveToFile(): Promise<void> {
    try {
      const jsonData = JSON.stringify(this.metrics, null, 2);
      await fs.writeFile(this.outputFile, jsonData, 'utf-8');
    } catch (error) {
      console.error('Failed to save metrics to file:', error);
    }
  }

  async loadFromFile(): Promise<void> {
    try {
      const data = await fs.readFile(this.outputFile, 'utf-8');
      this.metrics = JSON.parse(data);
    } catch (error) {
      // File doesn't exist yet, start with empty metrics
      this.metrics = [];
    }
  }

  getAggregatedMetrics() {
    if (this.metrics.length === 0) {
      return null;
    }

    const totalQueries = this.metrics.length;
    
    // Calculate latency statistics
    const latencies = this.metrics.map(m => m.latency_ms.total);
    const stageALatencies = this.metrics.map(m => m.latency_ms.stage_a);
    const stageBLatencies = this.metrics.filter(m => m.latency_ms.stage_b !== undefined).map(m => m.latency_ms.stage_b!);
    const stageCLatencies = this.metrics.filter(m => m.latency_ms.stage_c !== undefined).map(m => m.latency_ms.stage_c!);

    // Calculate result statistics
    const resultCounts = this.metrics.map(m => m.total_results);
    const f1Scores = this.metrics.filter(m => m.result_quality.f1_score !== undefined).map(m => m.result_quality.f1_score!);
    const precisionScores = this.metrics.filter(m => m.result_quality.precision !== undefined).map(m => m.result_quality.precision!);
    const recallScores = this.metrics.filter(m => m.result_quality.recall !== undefined).map(m => m.result_quality.recall!);

    return {
      summary: {
        total_queries: totalQueries,
        time_range: {
          start: this.metrics[0]?.timestamp || '',
          end: this.metrics[this.metrics.length - 1]?.timestamp || ''
        }
      },
      latency_metrics: {
        total: this.calculateStats(latencies),
        stage_a: this.calculateStats(stageALatencies),
        stage_b: stageBLatencies.length > 0 ? this.calculateStats(stageBLatencies) : null,
        stage_c: stageCLatencies.length > 0 ? this.calculateStats(stageCLatencies) : null
      },
      result_metrics: {
        total_results: this.calculateStats(resultCounts),
        quality_scores: {
          f1: f1Scores.length > 0 ? this.calculateStats(f1Scores) : null,
          precision: precisionScores.length > 0 ? this.calculateStats(precisionScores) : null,
          recall: recallScores.length > 0 ? this.calculateStats(recallScores) : null
        }
      },
      performance_analysis: {
        queries_per_second: this.calculateQPS(),
        sla_compliance: this.calculateSLACompliance(),
        error_rate: this.calculateErrorRate()
      }
    };
  }

  private calculateStats(values: number[]): {
    min: number;
    max: number;
    avg: number;
    p50: number;
    p95: number;
    p99: number;
    count: number;
  } {
    if (values.length === 0) {
      return { min: 0, max: 0, avg: 0, p50: 0, p95: 0, p99: 0, count: 0 };
    }

    const sorted = [...values].sort((a, b) => a - b);
    const sum = values.reduce((a, b) => a + b, 0);

    return {
      min: sorted[0] || 0,
      max: sorted[sorted.length - 1] || 0,
      avg: sum / values.length,
      p50: this.percentile(sorted, 50),
      p95: this.percentile(sorted, 95),
      p99: this.percentile(sorted, 99),
      count: values.length
    };
  }

  private percentile(sortedArray: number[], p: number): number {
    if (sortedArray.length === 0) return 0;
    if (sortedArray.length === 1) return sortedArray[0] || 0;

    const index = (p / 100) * (sortedArray.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);

    if (lower === upper) {
      return sortedArray[lower] || 0;
    }

    const lowerVal = sortedArray[lower] || 0;
    const upperVal = sortedArray[upper] || 0;
    return lowerVal + (upperVal - lowerVal) * (index - lower);
  }

  private calculateQPS(): number {
    if (this.metrics.length < 2) return 0;

    const startTime = new Date(this.metrics[0]?.timestamp || '').getTime();
    const endTime = new Date(this.metrics[this.metrics.length - 1]?.timestamp || '').getTime();
    const durationSeconds = (endTime - startTime) / 1000;

    return durationSeconds > 0 ? this.metrics.length / durationSeconds : 0;
  }

  private calculateSLACompliance(): {
    stage_a_under_8ms: number;
    stage_b_under_10ms: number;
    total_under_20ms: number;
  } {
    const stageACompliance = this.metrics.filter(m => m.latency_ms.stage_a < 8).length / this.metrics.length;
    const stageBMetrics = this.metrics.filter(m => m.latency_ms.stage_b !== undefined);
    const stageBCompliance = stageBMetrics.length > 0 
      ? stageBMetrics.filter(m => (m.latency_ms.stage_b ?? 0) < 10).length / stageBMetrics.length 
      : 1;
    const totalCompliance = this.metrics.filter(m => m.latency_ms.total < 20).length / this.metrics.length;

    return {
      stage_a_under_8ms: stageACompliance,
      stage_b_under_10ms: stageBCompliance,
      total_under_20ms: totalCompliance
    };
  }

  private calculateErrorRate(): number {
    // Count queries that failed (no trace_id or total_results is 0 when expected > 0)
    const errorCount = this.metrics.filter(m => 
      !m.trace_id || (m.total_results === 0 && m.query.trim().length > 0)
    ).length;
    
    return this.metrics.length > 0 ? errorCount / this.metrics.length : 0;
  }

  generateReport(): string {
    const aggregated = this.getAggregatedMetrics();
    if (!aggregated) {
      return 'No metrics data available';
    }

    const lines: string[] = [];
    lines.push('# Lens Search Engine Benchmark Report');
    lines.push('');
    lines.push('## Summary');
    lines.push('- **Total Queries**: ' + aggregated.summary.total_queries);
    lines.push('- **Time Range**: ' + aggregated.summary.time_range.start + ' to ' + aggregated.summary.time_range.end);
    lines.push('- **Queries per Second**: ' + aggregated.performance_analysis.queries_per_second.toFixed(2));
    lines.push('');
    lines.push('## Latency Metrics');
    lines.push('');
    lines.push('### Total Response Time');
    lines.push('- **Average**: ' + aggregated.latency_metrics.total.avg.toFixed(2) + 'ms');
    lines.push('- **p50**: ' + aggregated.latency_metrics.total.p50.toFixed(2) + 'ms');
    lines.push('- **p95**: ' + aggregated.latency_metrics.total.p95.toFixed(2) + 'ms');
    lines.push('- **p99**: ' + aggregated.latency_metrics.total.p99.toFixed(2) + 'ms');
    lines.push('');
    lines.push('### Stage A (Lexical + Fuzzy)');
    lines.push('- **Average**: ' + aggregated.latency_metrics.stage_a.avg.toFixed(2) + 'ms');
    lines.push('- **p95**: ' + aggregated.latency_metrics.stage_a.p95.toFixed(2) + 'ms');
    lines.push('- **SLA Compliance**: ' + (aggregated.performance_analysis.sla_compliance.stage_a_under_8ms * 100).toFixed(1) + '% under 8ms');
    
    if (aggregated.latency_metrics.stage_b) {
      lines.push('');
      lines.push('### Stage B (Symbol + AST)');
      lines.push('- **Average**: ' + aggregated.latency_metrics.stage_b.avg.toFixed(2) + 'ms');
      lines.push('- **p95**: ' + aggregated.latency_metrics.stage_b.p95.toFixed(2) + 'ms');
      lines.push('- **SLA Compliance**: ' + (aggregated.performance_analysis.sla_compliance.stage_b_under_10ms * 100).toFixed(1) + '% under 10ms');
    }
    
    if (aggregated.latency_metrics.stage_c) {
      lines.push('');
      lines.push('### Stage C (Semantic Rerank)');
      lines.push('- **Average**: ' + aggregated.latency_metrics.stage_c.avg.toFixed(2) + 'ms');
      lines.push('- **p95**: ' + aggregated.latency_metrics.stage_c.p95.toFixed(2) + 'ms');
    }
    
    lines.push('');
    lines.push('## Result Quality');
    lines.push('');
    lines.push('### Result Counts');
    lines.push('- **Average Results per Query**: ' + aggregated.result_metrics.total_results.avg.toFixed(1));
    lines.push('- **Max Results**: ' + aggregated.result_metrics.total_results.max);
    
    if (aggregated.result_metrics.quality_scores.f1) {
      lines.push('');
      lines.push('### Quality Scores');
      lines.push('- **F1 Score**: ' + aggregated.result_metrics.quality_scores.f1.avg.toFixed(3) + ' (avg)');
      if (aggregated.result_metrics.quality_scores.precision) {
        lines.push('- **Precision**: ' + aggregated.result_metrics.quality_scores.precision.avg.toFixed(3) + ' (avg)');
      }
      if (aggregated.result_metrics.quality_scores.recall) {
        lines.push('- **Recall**: ' + aggregated.result_metrics.quality_scores.recall.avg.toFixed(3) + ' (avg)');
      }
    }
    
    lines.push('');
    lines.push('## Performance Analysis');
    lines.push('- **Error Rate**: ' + (aggregated.performance_analysis.error_rate * 100).toFixed(2) + '%');
    lines.push('- **Total SLA Compliance**: ' + (aggregated.performance_analysis.sla_compliance.total_under_20ms * 100).toFixed(1) + '% under 20ms');
    
    lines.push('');
    lines.push('## Recommendations');
    lines.push(this.generateRecommendations(aggregated));

    return lines.join('\n');
  }

  private generateRecommendations(aggregated: any): string {
    const recommendations: string[] = [];

    // Latency recommendations
    if (aggregated.performance_analysis.sla_compliance.stage_a_under_8ms < 0.95) {
      recommendations.push('- Consider optimizing Stage A (lexical) search performance');
    }
    if (aggregated.latency_metrics.stage_b && aggregated.performance_analysis.sla_compliance.stage_b_under_10ms < 0.90) {
      recommendations.push('- Stage B (symbol) search may need optimization');
    }
    if (aggregated.performance_analysis.sla_compliance.total_under_20ms < 0.95) {
      recommendations.push('- Overall response time needs improvement for SLA compliance');
    }

    // Result quality recommendations
    if (aggregated.result_metrics.total_results.avg < 1) {
      recommendations.push('- Index content needs to be added - very few results returned');
    }
    if (aggregated.result_metrics.quality_scores.f1 && aggregated.result_metrics.quality_scores.f1.avg < 0.5) {
      recommendations.push('- Result quality (F1 score) could be improved with better ranking');
    }

    // Error rate recommendations
    if (aggregated.performance_analysis.error_rate > 0.05) {
      recommendations.push('- Error rate is high - investigate failed queries');
    }

    if (recommendations.length === 0) {
      recommendations.push('- System is performing well within expected parameters');
    }

    return recommendations.join('\n');
  }

  clear(): void {
    this.metrics = [];
  }

  getMetrics(): BenchmarkMetrics[] {
    return [...this.metrics];
  }
}

// Singleton instance
export const metricsAggregator = new MetricsAggregator();