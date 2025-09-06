/**
 * Telemetry and Observability Layer
 * 
 * Implements: "log topic_hit rate, alias_resolved_depth, type_match impact"
 * Addresses: Comprehensive observability with structured logging and metrics
 */

import { EventEmitter } from 'events';
import { writeFile, mkdir, readFile } from 'fs/promises';
import { join } from 'path';

export interface TelemetryEvent {
  event_type: string;
  timestamp: Date;
  trace_id: string;
  span_id: string;
  query_id?: string;
  user_id?: string;
  session_id?: string;
  data: Record<string, any>;
  metadata: {
    component: string;
    version: string;
    environment: string;
    hostname: string;
  };
}

export interface MetricPoint {
  name: string;
  value: number;
  timestamp: Date;
  tags: Record<string, string>;
  type: 'counter' | 'gauge' | 'histogram' | 'timer';
}

export interface LogEntry {
  level: 'debug' | 'info' | 'warn' | 'error' | 'fatal';
  timestamp: Date;
  component: string;
  message: string;
  trace_id?: string;
  span_id?: string;
  query_id?: string;
  context: Record<string, any>;
}

export interface QueryTrace {
  query_id: string;
  trace_id: string;
  start_time: Date;
  end_time?: Date;
  query_text: string;
  results_count: number;
  stages: QueryStageTrace[];
  metrics: {
    total_latency_ms: number;
    stage_breakdown: Record<string, number>;
    topic_stats: TopicStats;
    alias_stats: AliasStats;
    why_breakdown: WhyBreakdown;
  };
}

export interface QueryStageTrace {
  stage: string;
  start_time: Date;
  end_time: Date;
  duration_ms: number;
  input_size: number;
  output_size: number;
  features_used: string[];
  debug_info: Record<string, any>;
}

export interface TopicStats {
  hit_rate: number;
  clusters_used: number;
  avg_cluster_size: number;
  topic_diversity: number;
  top_topics: Array<{ topic: string; weight: number; usage_count: number }>;
}

export interface AliasStats {
  resolved_depth: number;
  max_depth: number;
  resolution_chains: Array<{ from: string; to: string; depth: number }>;
  re_export_hops: number;
  type_match_impact: number;
}

export interface WhyBreakdown {
  exact_fuzzy: number;
  symbol_struct: number;
  semantic: number;
  total_results: number;
  dominant_mode: 'exact' | 'fuzzy' | 'symbol' | 'struct' | 'semantic';
}

export interface ObservabilityDashboard {
  current_qps: number;
  avg_latency_p95: number;
  error_rate: number;
  topic_hit_rate: number;
  alias_depth_avg: number;
  why_mix_distribution: WhyBreakdown;
  recent_alerts: Array<{ time: Date; severity: string; message: string }>;
  performance_trend: Array<{ time: Date; p95: number; qps: number }>;
}

export class TelemetryObservabilityLayer extends EventEmitter {
  private traces: Map<string, QueryTrace>;
  private metrics: MetricPoint[];
  private logs: LogEntry[];
  private activeSpans: Map<string, any>;
  private outputDirectory: string;
  private flushInterval?: NodeJS.Timeout;

  constructor(outputDirectory: string = './observability-data') {
    super();
    this.traces = new Map();
    this.metrics = [];
    this.logs = [];
    this.activeSpans = new Map();
    this.outputDirectory = outputDirectory;
    this.startBackgroundTasks();
  }

  /**
   * Start a new query trace
   */
  startQueryTrace(queryId: string, queryText: string, traceId?: string): QueryTrace {
    const trace: QueryTrace = {
      query_id: queryId,
      trace_id: traceId || this.generateTraceId(),
      start_time: new Date(),
      query_text: queryText,
      results_count: 0,
      stages: [],
      metrics: {
        total_latency_ms: 0,
        stage_breakdown: {},
        topic_stats: {
          hit_rate: 0,
          clusters_used: 0,
          avg_cluster_size: 0,
          topic_diversity: 0,
          top_topics: []
        },
        alias_stats: {
          resolved_depth: 0,
          max_depth: 0,
          resolution_chains: [],
          re_export_hops: 0,
          type_match_impact: 0
        },
        why_breakdown: {
          exact_fuzzy: 0,
          symbol_struct: 0,
          semantic: 0,
          total_results: 0,
          dominant_mode: 'exact'
        }
      }
    };

    this.traces.set(queryId, trace);
    
    this.logInfo('query_started', `Query started: ${queryText.substring(0, 100)}`, {
      query_id: queryId,
      trace_id: trace.trace_id,
      query_length: queryText.length
    });

    return trace;
  }

  /**
   * Add a stage trace to a query
   */
  startStageTrace(
    queryId: string, 
    stage: string, 
    inputSize: number = 0,
    featuresUsed: string[] = []
  ): string {
    const trace = this.traces.get(queryId);
    if (!trace) {
      this.logError('stage_trace_error', `Query trace not found: ${queryId}`);
      return '';
    }

    const spanId = this.generateSpanId();
    const stageTrace: QueryStageTrace = {
      stage,
      start_time: new Date(),
      end_time: new Date(), // Will be updated on finish
      duration_ms: 0,
      input_size: inputSize,
      output_size: 0,
      features_used: featuresUsed,
      debug_info: {}
    };

    trace.stages.push(stageTrace);
    this.activeSpans.set(spanId, { queryId, stageIndex: trace.stages.length - 1 });

    this.logDebug('stage_started', `Stage ${stage} started`, {
      query_id: queryId,
      trace_id: trace.trace_id,
      span_id: spanId,
      stage,
      input_size: inputSize,
      features_used: featuresUsed.join(',')
    });

    return spanId;
  }

  /**
   * Finish a stage trace
   */
  finishStageTrace(
    spanId: string, 
    outputSize: number = 0, 
    debugInfo: Record<string, any> = {}
  ): void {
    const span = this.activeSpans.get(spanId);
    if (!span) {
      this.logError('span_not_found', `Span not found: ${spanId}`);
      return;
    }

    const trace = this.traces.get(span.queryId);
    if (!trace) return;

    const stageTrace = trace.stages[span.stageIndex];
    stageTrace.end_time = new Date();
    stageTrace.duration_ms = stageTrace.end_time.getTime() - stageTrace.start_time.getTime();
    stageTrace.output_size = outputSize;
    stageTrace.debug_info = debugInfo;

    // Update stage breakdown
    trace.metrics.stage_breakdown[stageTrace.stage] = stageTrace.duration_ms;

    this.activeSpans.delete(spanId);

    this.logDebug('stage_finished', `Stage ${stageTrace.stage} finished`, {
      query_id: span.queryId,
      trace_id: trace.trace_id,
      span_id: spanId,
      stage: stageTrace.stage,
      duration_ms: stageTrace.duration_ms,
      output_size: outputSize
    });

    // Record stage timing metric
    this.recordMetric({
      name: 'stage_duration_ms',
      value: stageTrace.duration_ms,
      timestamp: new Date(),
      tags: {
        stage: stageTrace.stage,
        query_id: span.queryId
      },
      type: 'timer'
    });
  }

  /**
   * Record topic statistics for a query
   */
  recordTopicStats(queryId: string, stats: TopicStats): void {
    const trace = this.traces.get(queryId);
    if (!trace) return;

    trace.metrics.topic_stats = stats;

    this.logInfo('topic_stats', 'Topic clustering statistics recorded', {
      query_id: queryId,
      trace_id: trace.trace_id,
      hit_rate: stats.hit_rate,
      clusters_used: stats.clusters_used,
      topic_diversity: stats.topic_diversity
    });

    // Record metrics
    this.recordMetric({
      name: 'topic_hit_rate',
      value: stats.hit_rate,
      timestamp: new Date(),
      tags: { query_id: queryId },
      type: 'gauge'
    });

    this.recordMetric({
      name: 'topic_clusters_used',
      value: stats.clusters_used,
      timestamp: new Date(),
      tags: { query_id: queryId },
      type: 'gauge'
    });
  }

  /**
   * Record alias resolution statistics
   */
  recordAliasStats(queryId: string, stats: AliasStats): void {
    const trace = this.traces.get(queryId);
    if (!trace) return;

    trace.metrics.alias_stats = stats;

    this.logInfo('alias_stats', 'Alias resolution statistics recorded', {
      query_id: queryId,
      trace_id: trace.trace_id,
      resolved_depth: stats.resolved_depth,
      max_depth: stats.max_depth,
      type_match_impact: stats.type_match_impact
    });

    // Record alias depth metric
    this.recordMetric({
      name: 'alias_resolved_depth',
      value: stats.resolved_depth,
      timestamp: new Date(),
      tags: { query_id: queryId },
      type: 'histogram'
    });

    // Log interesting alias chains
    if (stats.resolution_chains.length > 0) {
      this.logDebug('alias_chains', 'Alias resolution chains', {
        query_id: queryId,
        chains: stats.resolution_chains.slice(0, 5) // Top 5 chains
      });
    }
  }

  /**
   * Record why-mix breakdown
   */
  recordWhyBreakdown(queryId: string, breakdown: WhyBreakdown): void {
    const trace = this.traces.get(queryId);
    if (!trace) return;

    trace.metrics.why_breakdown = breakdown;

    this.logInfo('why_breakdown', 'Why-mix breakdown recorded', {
      query_id: queryId,
      trace_id: trace.trace_id,
      exact_fuzzy: breakdown.exact_fuzzy,
      symbol_struct: breakdown.symbol_struct,
      semantic: breakdown.semantic,
      dominant_mode: breakdown.dominant_mode
    });

    // Record individual why-mix components
    const components = [
      { name: 'exact_fuzzy', value: breakdown.exact_fuzzy },
      { name: 'symbol_struct', value: breakdown.symbol_struct },
      { name: 'semantic', value: breakdown.semantic }
    ];

    for (const component of components) {
      this.recordMetric({
        name: 'why_mix_ratio',
        value: component.value,
        timestamp: new Date(),
        tags: { 
          query_id: queryId, 
          component: component.name 
        },
        type: 'gauge'
      });
    }
  }

  /**
   * Finish a query trace
   */
  finishQueryTrace(queryId: string, resultsCount: number): void {
    const trace = this.traces.get(queryId);
    if (!trace) return;

    trace.end_time = new Date();
    trace.results_count = resultsCount;
    trace.metrics.total_latency_ms = trace.end_time.getTime() - trace.start_time.getTime();

    this.logInfo('query_finished', `Query completed: ${resultsCount} results`, {
      query_id: queryId,
      trace_id: trace.trace_id,
      total_latency_ms: trace.metrics.total_latency_ms,
      results_count: resultsCount,
      stages_count: trace.stages.length
    });

    // Record overall query metrics
    this.recordMetric({
      name: 'query_latency_ms',
      value: trace.metrics.total_latency_ms,
      timestamp: new Date(),
      tags: { query_id: queryId },
      type: 'timer'
    });

    this.recordMetric({
      name: 'query_results_count',
      value: resultsCount,
      timestamp: new Date(),
      tags: { query_id: queryId },
      type: 'counter'
    });

    // Emit completion event
    this.emit('query_completed', trace);
  }

  /**
   * Record custom telemetry event
   */
  recordEvent(eventType: string, data: Record<string, any>, queryId?: string): void {
    const event: TelemetryEvent = {
      event_type: eventType,
      timestamp: new Date(),
      trace_id: this.generateTraceId(),
      span_id: this.generateSpanId(),
      query_id: queryId,
      data,
      metadata: {
        component: 'raptor_search',
        version: '1.0.0-rc.2',
        environment: process.env.NODE_ENV || 'development',
        hostname: process.env.HOSTNAME || 'localhost'
      }
    };

    this.emit('telemetry_event', event);

    this.logDebug('telemetry_event', `Event: ${eventType}`, {
      event_type: eventType,
      trace_id: event.trace_id,
      query_id: queryId,
      data_keys: Object.keys(data).join(',')
    });
  }

  /**
   * Record metric point
   */
  recordMetric(metric: MetricPoint): void {
    this.metrics.push(metric);
    
    // Keep only last 10,000 metrics in memory
    if (this.metrics.length > 10000) {
      this.metrics = this.metrics.slice(-10000);
    }

    this.emit('metric_recorded', metric);
  }

  /**
   * Logging methods
   */
  private logDebug(component: string, message: string, context: Record<string, any> = {}): void {
    this.log('debug', component, message, context);
  }

  private logInfo(component: string, message: string, context: Record<string, any> = {}): void {
    this.log('info', component, message, context);
  }

  private logWarn(component: string, message: string, context: Record<string, any> = {}): void {
    this.log('warn', component, message, context);
  }

  private logError(component: string, message: string, context: Record<string, any> = {}): void {
    this.log('error', component, message, context);
  }

  private log(
    level: 'debug' | 'info' | 'warn' | 'error' | 'fatal',
    component: string,
    message: string,
    context: Record<string, any> = {}
  ): void {
    const entry: LogEntry = {
      level,
      timestamp: new Date(),
      component,
      message,
      trace_id: context.trace_id,
      span_id: context.span_id,
      query_id: context.query_id,
      context
    };

    this.logs.push(entry);

    // Keep only last 1,000 log entries in memory
    if (this.logs.length > 1000) {
      this.logs = this.logs.slice(-1000);
    }

    // Console output for development
    if (level === 'error' || level === 'fatal') {
      console.error(`[${level.toUpperCase()}] ${component}: ${message}`, context);
    } else if (level === 'warn') {
      console.warn(`[WARN] ${component}: ${message}`, context);
    } else if (process.env.DEBUG || level === 'info') {
      console.log(`[${level.toUpperCase()}] ${component}: ${message}`, context);
    }

    this.emit('log_entry', entry);
  }

  /**
   * Generate observability dashboard data
   */
  generateDashboardData(): ObservabilityDashboard {
    const recentMetrics = this.getRecentMetrics(300); // Last 5 minutes
    const recentLogs = this.getRecentLogs(300);

    // Calculate aggregations
    const qpsMetrics = recentMetrics.filter(m => m.name === 'query_latency_ms');
    const latencyMetrics = recentMetrics.filter(m => m.name === 'query_latency_ms');
    const topicHitMetrics = recentMetrics.filter(m => m.name === 'topic_hit_rate');
    const aliasDepthMetrics = recentMetrics.filter(m => m.name === 'alias_resolved_depth');

    const currentQps = qpsMetrics.length > 0 ? qpsMetrics.length / 5 : 0; // Queries per minute / 5
    const avgLatencyP95 = latencyMetrics.length > 0 
      ? this.calculatePercentile(latencyMetrics.map(m => m.value), 95)
      : 0;

    const errorLogs = recentLogs.filter(l => l.level === 'error' || l.level === 'fatal');
    const errorRate = qpsMetrics.length > 0 ? errorLogs.length / qpsMetrics.length : 0;

    const topicHitRate = topicHitMetrics.length > 0 
      ? topicHitMetrics.reduce((sum, m) => sum + m.value, 0) / topicHitMetrics.length
      : 0;

    const aliasDepthAvg = aliasDepthMetrics.length > 0
      ? aliasDepthMetrics.reduce((sum, m) => sum + m.value, 0) / aliasDepthMetrics.length
      : 0;

    // Why-mix distribution
    const whyMixMetrics = recentMetrics.filter(m => m.name === 'why_mix_ratio');
    const whyBreakdown = this.aggregateWhyMix(whyMixMetrics);

    // Recent alerts
    const alerts = recentLogs
      .filter(l => l.level === 'warn' || l.level === 'error' || l.level === 'fatal')
      .slice(-10)
      .map(l => ({
        time: l.timestamp,
        severity: l.level,
        message: l.message
      }));

    // Performance trend (last hour in 10-minute buckets)
    const trendMetrics = this.getRecentMetrics(3600); // Last hour
    const performanceTrend = this.calculatePerformanceTrend(trendMetrics);

    return {
      current_qps: currentQps,
      avg_latency_p95: avgLatencyP95,
      error_rate: errorRate,
      topic_hit_rate: topicHitRate,
      alias_depth_avg: aliasDepthAvg,
      why_mix_distribution: whyBreakdown,
      recent_alerts: alerts,
      performance_trend: performanceTrend
    };
  }

  /**
   * Start background tasks
   */
  private startBackgroundTasks(): void {
    // Flush data to disk every 60 seconds
    this.flushInterval = setInterval(async () => {
      await this.flushToDisk();
    }, 60000);
  }

  /**
   * Stop background tasks
   */
  stop(): void {
    if (this.flushInterval) {
      clearInterval(this.flushInterval);
    }
  }

  /**
   * Flush data to disk
   */
  private async flushToDisk(): Promise<void> {
    try {
      await mkdir(this.outputDirectory, { recursive: true });

      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');

      // Write traces
      if (this.traces.size > 0) {
        const tracesData = Array.from(this.traces.values());
        await writeFile(
          join(this.outputDirectory, `traces-${timestamp}.json`),
          JSON.stringify(tracesData, null, 2)
        );
      }

      // Write metrics
      if (this.metrics.length > 0) {
        await writeFile(
          join(this.outputDirectory, `metrics-${timestamp}.json`),
          JSON.stringify(this.metrics, null, 2)
        );
      }

      // Write logs
      if (this.logs.length > 0) {
        const logsNdjson = this.logs.map(log => JSON.stringify(log)).join('\n');
        await writeFile(
          join(this.outputDirectory, `logs-${timestamp}.ndjson`),
          logsNdjson
        );
      }

    } catch (error) {
      console.error('Failed to flush observability data:', error);
    }
  }

  // Helper methods
  private generateTraceId(): string {
    return `trace_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;
  }

  private generateSpanId(): string {
    return `span_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`;
  }

  private getRecentMetrics(seconds: number): MetricPoint[] {
    const cutoff = new Date();
    cutoff.setSeconds(cutoff.getSeconds() - seconds);
    return this.metrics.filter(m => m.timestamp >= cutoff);
  }

  private getRecentLogs(seconds: number): LogEntry[] {
    const cutoff = new Date();
    cutoff.setSeconds(cutoff.getSeconds() - seconds);
    return this.logs.filter(l => l.timestamp >= cutoff);
  }

  private calculatePercentile(values: number[], percentile: number): number {
    if (values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.ceil((percentile / 100) * sorted.length) - 1;
    return sorted[Math.max(0, index)];
  }

  private aggregateWhyMix(metrics: MetricPoint[]): WhyBreakdown {
    const components = ['exact_fuzzy', 'symbol_struct', 'semantic'];
    const breakdown = { exact_fuzzy: 0, symbol_struct: 0, semantic: 0, total_results: 0, dominant_mode: 'exact' as const };

    for (const component of components) {
      const componentMetrics = metrics.filter(m => m.tags.component === component);
      if (componentMetrics.length > 0) {
        breakdown[component as keyof typeof breakdown] = componentMetrics.reduce((sum, m) => sum + m.value, 0) / componentMetrics.length;
      }
    }

    // Determine dominant mode
    if (breakdown.semantic > breakdown.symbol_struct && breakdown.semantic > breakdown.exact_fuzzy) {
      breakdown.dominant_mode = 'semantic';
    } else if (breakdown.symbol_struct > breakdown.exact_fuzzy) {
      breakdown.dominant_mode = 'symbol';
    }

    return breakdown;
  }

  private calculatePerformanceTrend(metrics: MetricPoint[]): Array<{ time: Date; p95: number; qps: number }> {
    // Group metrics into 10-minute buckets
    const buckets = new Map<string, { latencies: number[]; count: number }>();
    const bucketSize = 10 * 60 * 1000; // 10 minutes in ms

    for (const metric of metrics) {
      if (metric.name === 'query_latency_ms') {
        const bucketKey = Math.floor(metric.timestamp.getTime() / bucketSize) * bucketSize;
        const bucketTime = new Date(bucketKey).toISOString();
        
        if (!buckets.has(bucketTime)) {
          buckets.set(bucketTime, { latencies: [], count: 0 });
        }
        
        const bucket = buckets.get(bucketTime)!;
        bucket.latencies.push(metric.value);
        bucket.count++;
      }
    }

    // Calculate trend points
    return Array.from(buckets.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([timeStr, data]) => ({
        time: new Date(timeStr),
        p95: this.calculatePercentile(data.latencies, 95),
        qps: data.count / 10 // QPS over 10-minute period
      }));
  }
}

// Factory function
export function createTelemetryLayer(outputDirectory?: string): TelemetryObservabilityLayer {
  return new TelemetryObservabilityLayer(outputDirectory);
}

// CLI demo
if (import.meta.main) {
  console.log('📊 Telemetry and Observability Layer Demo\n');
  
  const telemetry = createTelemetryLayer('./demo-observability');
  
  // Set up event listeners
  telemetry.on('query_completed', (trace: QueryTrace) => {
    console.log(`✓ Query completed: ${trace.query_id} (${trace.metrics.total_latency_ms}ms)`);
  });

  // Demo query trace
  const demoQuery = async () => {
    console.log('🔍 Starting demo query trace...\n');
    
    const queryId = 'demo_query_001';
    const queryText = 'class UserService definition';
    
    // Start trace
    const trace = telemetry.startQueryTrace(queryId, queryText);
    
    // Stage 1: Lexical
    const stage1Span = telemetry.startStageTrace(queryId, 'lexical', 1, ['trigram_index', 'fuzzy_match']);
    await new Promise(resolve => setTimeout(resolve, 50));
    telemetry.finishStageTrace(stage1Span, 234, { trigram_matches: 156, fuzzy_matches: 78 });
    
    // Stage 2: Symbol
    const stage2Span = telemetry.startStageTrace(queryId, 'symbol', 234, ['lsp_symbols', 'type_inference']);
    await new Promise(resolve => setTimeout(resolve, 75));
    telemetry.finishStageTrace(stage2Span, 89, { symbol_matches: 45, type_matches: 34 });
    
    // Record topic stats
    telemetry.recordTopicStats(queryId, {
      hit_rate: 0.78,
      clusters_used: 12,
      avg_cluster_size: 45,
      topic_diversity: 0.85,
      top_topics: [
        { topic: 'user_management', weight: 0.45, usage_count: 23 },
        { topic: 'service_layer', weight: 0.32, usage_count: 18 }
      ]
    });
    
    // Record alias stats
    telemetry.recordAliasStats(queryId, {
      resolved_depth: 2.3,
      max_depth: 5,
      resolution_chains: [
        { from: 'User', to: 'UserEntity', depth: 2 },
        { from: 'Service', to: 'BaseService', depth: 1 }
      ],
      re_export_hops: 1,
      type_match_impact: 0.18
    });
    
    // Record why-mix breakdown
    telemetry.recordWhyBreakdown(queryId, {
      exact_fuzzy: 0.42,
      symbol_struct: 0.38,
      semantic: 0.20,
      total_results: 89,
      dominant_mode: 'exact'
    });
    
    // Stage 3: Semantic
    const stage3Span = telemetry.startStageTrace(queryId, 'semantic', 89, ['raptor_clustering', 'embedding_search']);
    await new Promise(resolve => setTimeout(resolve, 100));
    telemetry.finishStageTrace(stage3Span, 67, { semantic_matches: 23, cluster_boosts: 12 });
    
    // Finish trace
    telemetry.finishQueryTrace(queryId, 67);
    
    console.log('\n📈 Dashboard Data:');
    console.log(JSON.stringify(telemetry.generateDashboardData(), null, 2));
  };
  
  demoQuery().then(() => {
    console.log('\n✓ Demo completed');
    telemetry.stop();
  }).catch(console.error);
}