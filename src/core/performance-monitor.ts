/**
 * Enhanced Performance Monitoring System
 * 
 * Comprehensive monitoring solution for lens search engine with:
 * - Real-time performance metrics collection
 * - RED (Rate, Errors, Duration) metrics
 * - USE (Utilization, Saturation, Errors) metrics
 * - Custom business metrics tracking
 * - Anomaly detection and alerting
 * - Performance regression detection
 * - Resource utilization monitoring
 */

import { EventEmitter } from 'node:events';
import { performance, PerformanceEntry } from 'node:perf_hooks';
import { cpus, totalmem, freemem } from 'node:os';
import { opentelemetry } from '../telemetry/index.js';
import { SearchContext } from '../types/core.js';
import { SearchHit } from '../types/embedder-proof-levers.js';

// Core metric types
export interface PerformanceMetric {
  name: string;
  value: number;
  unit: string;
  timestamp: number;
  labels?: Record<string, string>;
}

export interface AggregatedMetric {
  name: string;
  count: number;
  sum: number;
  min: number;
  max: number;
  mean: number;
  p50: number;
  p95: number;
  p99: number;
  stddev: number;
  timestamp: number;
  window: string; // '1m', '5m', '15m', '1h', '24h'
}

// RED Metrics (Rate, Errors, Duration)
export interface REDMetrics {
  rate: {
    requestsPerSecond: number;
    searchesPerSecond: number;
    indexUpdatesPerSecond: number;
  };
  errors: {
    errorRate: number;
    errorCount: number;
    errorsByType: Map<string, number>;
  };
  duration: {
    averageLatency: number;
    p50Latency: number;
    p95Latency: number;
    p99Latency: number;
  };
}

// USE Metrics (Utilization, Saturation, Errors)
export interface USEMetrics {
  utilization: {
    cpuUsage: number;
    memoryUsage: number;
    diskUsage: number;
    networkUsage: number;
  };
  saturation: {
    cpuSaturation: number;
    memorySaturation: number;
    queueDepth: number;
    threadPoolSaturation: number;
  };
  errors: {
    systemErrors: number;
    resourceExhaustion: number;
    timeoutErrors: number;
  };
}

// Business metrics specific to search engine
export interface SearchMetrics {
  queryVolume: {
    totalQueries: number;
    uniqueQueries: number;
    queryComplexity: AggregatedMetric;
  };
  resultQuality: {
    averageResultCount: number;
    zeroResultQueries: number;
    clickThroughRate: number;
    meanReciprocalRank: number;
  };
  indexHealth: {
    indexSize: number;
    indexingRate: number;
    stalenessScore: number;
    fragmentationRatio: number;
  };
}

// Performance alerts and thresholds
export interface PerformanceThreshold {
  metric: string;
  operator: 'gt' | 'lt' | 'eq' | 'gte' | 'lte';
  value: number;
  window: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  description: string;
}

export interface PerformanceAlert {
  id: string;
  threshold: PerformanceThreshold;
  currentValue: number;
  timestamp: number;
  resolved: boolean;
  resolvedAt?: number;
}

// Time series data point
export interface TimeSeriesDataPoint {
  timestamp: number;
  value: number;
  labels?: Record<string, string>;
}

/**
 * Comprehensive performance monitoring system
 */
export class PerformanceMonitor extends EventEmitter {
  private readonly tracer = opentelemetry.trace.getTracer('lens-performance-monitor');
  private static instance: PerformanceMonitor | null = null;
  
  // Metric storage - in production, would use time series database
  private metrics = new Map<string, TimeSeriesDataPoint[]>();
  private aggregatedMetrics = new Map<string, AggregatedMetric[]>();
  
  // Performance thresholds and alerts
  private thresholds: PerformanceThreshold[] = [];
  private activeAlerts = new Map<string, PerformanceAlert>();
  
  // System resource monitoring
  private systemMetrics = {
    startTime: Date.now(),
    cpuUsage: 0,
    memoryUsage: 0,
    gcStats: { collections: 0, pauseTime: 0 }
  };
  
  // Request tracking
  private activeRequests = new Map<string, { startTime: number; context: SearchContext }>();
  private requestHistogram: number[] = [];
  
  // Business metrics
  private searchMetrics: SearchMetrics = {
    queryVolume: {
      totalQueries: 0,
      uniqueQueries: 0,
      queryComplexity: this.createEmptyAggregatedMetric('query_complexity')
    },
    resultQuality: {
      averageResultCount: 0,
      zeroResultQueries: 0,
      clickThroughRate: 0,
      meanReciprocalRank: 0
    },
    indexHealth: {
      indexSize: 0,
      indexingRate: 0,
      stalenessScore: 0,
      fragmentationRatio: 0
    }
  };

  private constructor() {
    super();
    this.initializeDefaultThresholds();
    this.startSystemMonitoring();
    this.setupGCMonitoring();
  }

  static getInstance(): PerformanceMonitor {
    if (!PerformanceMonitor.instance) {
      PerformanceMonitor.instance = new PerformanceMonitor();
    }
    return PerformanceMonitor.instance;
  }

  /**
   * Record a performance metric
   */
  recordMetric(
    name: string, 
    value: number, 
    unit: string = 'ms',
    labels?: Record<string, string>
  ): void {
    const timestamp = Date.now();
    const dataPoint: TimeSeriesDataPoint = { timestamp, value, labels };
    
    if (!this.metrics.has(name)) {
      this.metrics.set(name, []);
    }
    
    const series = this.metrics.get(name)!;
    series.push(dataPoint);
    
    // Keep only last 24 hours of data
    const cutoff = timestamp - (24 * 60 * 60 * 1000);
    this.metrics.set(name, series.filter(point => point.timestamp > cutoff));
    
    // Update aggregated metrics
    this.updateAggregatedMetrics(name);
    
    // Check thresholds
    this.checkThresholds(name, value);
    
    // Emit metric event for real-time monitoring
    this.emit('metric', { name, value, unit, timestamp, labels });
  }

  /**
   * Start timing a request/operation
   */
  startTiming(operationId: string, context?: SearchContext): void {
    this.activeRequests.set(operationId, {
      startTime: performance.now(),
      context: context || {} as SearchContext
    });
  }

  /**
   * End timing and record duration
   */
  endTiming(operationId: string, labels?: Record<string, string>): number {
    const request = this.activeRequests.get(operationId);
    if (!request) return 0;
    
    const duration = performance.now() - request.startTime;
    this.activeRequests.delete(operationId);
    
    // Record duration metric
    this.recordMetric(`${operationId}_duration`, duration, 'ms', labels);
    
    // Update request histogram
    this.requestHistogram.push(duration);
    if (this.requestHistogram.length > 10000) {
      this.requestHistogram = this.requestHistogram.slice(-5000); // Keep last 5000
    }
    
    return duration;
  }

  /**
   * Record search-specific metrics
   */
  recordSearchMetrics(
    query: string,
    results: SearchHit[],
    duration: number,
    context: SearchContext
  ): void {
    return this.tracer.startActiveSpan('record-search-metrics', (span) => {
      try {
        // Update query volume metrics
        this.searchMetrics.queryVolume.totalQueries++;
        
        // Record query complexity (length, special chars, etc.)
        const complexity = this.calculateQueryComplexity(query);
        this.recordMetric('query_complexity', complexity, 'score');
        
        // Record result metrics
        const resultCount = results.length;
        this.recordMetric('result_count', resultCount, 'count');
        
        if (resultCount === 0) {
          this.searchMetrics.resultQuality.zeroResultQueries++;
        }
        
        // Record search duration
        this.recordMetric('search_duration', duration, 'ms', {
          query_type: context.searchType || 'unknown',
          language: context.filters?.languages?.[0] || 'unknown'
        });
        
        // Record stage-specific timings if available
        if (context.stageTimings) {
          for (const [stage, timing] of Object.entries(context.stageTimings)) {
            this.recordMetric(`stage_${stage}_duration`, timing, 'ms');
          }
        }
        
        span.setAttributes({
          'lens.search.query_length': query.length,
          'lens.search.result_count': resultCount,
          'lens.search.duration_ms': duration
        });

      } catch (error) {
        span.recordException(error as Error);
        span.setStatus({ code: opentelemetry.SpanStatusCode.ERROR });
      } finally {
        span.end();
      }
    });
  }

  /**
   * Get RED metrics for the system
   */
  getREDMetrics(window: string = '5m'): REDMetrics {
    const now = Date.now();
    const windowMs = this.parseTimeWindow(window);
    const since = now - windowMs;
    
    // Calculate rate metrics
    const searchDurations = this.getMetricValues('search_duration', since);
    const errorMetrics = this.getMetricValues('errors', since);
    
    return {
      rate: {
        requestsPerSecond: searchDurations.length / (windowMs / 1000),
        searchesPerSecond: searchDurations.length / (windowMs / 1000),
        indexUpdatesPerSecond: this.getMetricValues('index_updates', since).length / (windowMs / 1000)
      },
      errors: {
        errorRate: errorMetrics.length / Math.max(1, searchDurations.length),
        errorCount: errorMetrics.length,
        errorsByType: this.getErrorsByType(since)
      },
      duration: {
        averageLatency: this.calculateMean(searchDurations),
        p50Latency: this.calculatePercentile(searchDurations, 0.5),
        p95Latency: this.calculatePercentile(searchDurations, 0.95),
        p99Latency: this.calculatePercentile(searchDurations, 0.99)
      }
    };
  }

  /**
   * Get USE metrics for the system
   */
  getUSEMetrics(): USEMetrics {
    const cpuUsage = process.cpuUsage();
    const memInfo = process.memoryUsage();
    
    return {
      utilization: {
        cpuUsage: this.systemMetrics.cpuUsage,
        memoryUsage: (memInfo.heapUsed / memInfo.heapTotal) * 100,
        diskUsage: 0, // Would need disk monitoring
        networkUsage: 0 // Would need network monitoring
      },
      saturation: {
        cpuSaturation: Math.max(0, this.systemMetrics.cpuUsage - 80), // Saturation starts at 80%
        memorySaturation: Math.max(0, ((memInfo.heapUsed / memInfo.heapTotal) * 100) - 85),
        queueDepth: this.activeRequests.size,
        threadPoolSaturation: 0 // Would need thread pool monitoring
      },
      errors: {
        systemErrors: this.getMetricValues('system_errors', Date.now() - 300000).length, // Last 5 minutes
        resourceExhaustion: this.getMetricValues('resource_exhaustion', Date.now() - 300000).length,
        timeoutErrors: this.getMetricValues('timeout_errors', Date.now() - 300000).length
      }
    };
  }

  /**
   * Get current search-specific metrics
   */
  getSearchMetrics(): SearchMetrics {
    return { ...this.searchMetrics };
  }

  /**
   * Get aggregated metrics for a specific metric name
   */
  getAggregatedMetrics(name: string, window: string = '1h'): AggregatedMetric | null {
    const aggregated = this.aggregatedMetrics.get(name);
    if (!aggregated) return null;
    
    return aggregated.find(metric => metric.window === window) || null;
  }

  /**
   * Add performance threshold for alerting
   */
  addThreshold(threshold: PerformanceThreshold): void {
    this.thresholds.push(threshold);
  }

  /**
   * Get active alerts
   */
  getActiveAlerts(): PerformanceAlert[] {
    return Array.from(this.activeAlerts.values()).filter(alert => !alert.resolved);
  }

  /**
   * Get performance dashboard data
   */
  getDashboardData(): {
    red: REDMetrics;
    use: USEMetrics;
    search: SearchMetrics;
    activeAlerts: PerformanceAlert[];
    systemInfo: any;
  } {
    return {
      red: this.getREDMetrics(),
      use: this.getUSEMetrics(),
      search: this.getSearchMetrics(),
      activeAlerts: this.getActiveAlerts(),
      systemInfo: {
        uptime: Date.now() - this.systemMetrics.startTime,
        nodeVersion: process.version,
        cpuCount: cpus().length,
        totalMemory: totalmem(),
        freeMemory: freemem()
      }
    };
  }

  /**
   * Private helper methods
   */
  
  private initializeDefaultThresholds(): void {
    const defaultThresholds: PerformanceThreshold[] = [
      {
        metric: 'search_duration',
        operator: 'gt',
        value: 20, // 20ms threshold
        window: '5m',
        severity: 'warning',
        description: 'Search latency exceeding 20ms'
      },
      {
        metric: 'search_duration',
        operator: 'gt',
        value: 50, // 50ms critical threshold
        window: '1m',
        severity: 'critical',
        description: 'Search latency critically high'
      },
      {
        metric: 'error_rate',
        operator: 'gt',
        value: 0.01, // 1% error rate
        window: '5m',
        severity: 'error',
        description: 'Error rate exceeding 1%'
      },
      {
        metric: 'memory_usage',
        operator: 'gt',
        value: 90, // 90% memory usage
        window: '1m',
        severity: 'warning',
        description: 'Memory usage high'
      }
    ];
    
    this.thresholds.push(...defaultThresholds);
  }

  private startSystemMonitoring(): void {
    // Monitor system resources every 5 seconds
    setInterval(() => {
      this.collectSystemMetrics();
    }, 5000);
  }

  private collectSystemMetrics(): void {
    const memInfo = process.memoryUsage();
    
    // Record system metrics
    this.recordMetric('heap_used', memInfo.heapUsed / 1024 / 1024, 'MB');
    this.recordMetric('heap_total', memInfo.heapTotal / 1024 / 1024, 'MB');
    this.recordMetric('external_memory', memInfo.external / 1024 / 1024, 'MB');
    this.recordMetric('process_memory', memInfo.rss / 1024 / 1024, 'MB');
    
    // Update cached values
    this.systemMetrics.memoryUsage = (memInfo.heapUsed / memInfo.heapTotal) * 100;
  }

  private setupGCMonitoring(): void {
    // Monitor garbage collection if available
    if (global.gc && process.env.NODE_ENV === 'development') {
      const originalGC = global.gc;
      global.gc = async () => {
        const start = performance.now();
        originalGC();
        const duration = performance.now() - start;
        
        this.systemMetrics.gcStats.collections++;
        this.systemMetrics.gcStats.pauseTime += duration;
        
        this.recordMetric('gc_duration', duration, 'ms');
      };
    }
  }

  private calculateQueryComplexity(query: string): number {
    let complexity = 0;
    
    // Length factor
    complexity += Math.min(query.length / 10, 5);
    
    // Special characters
    complexity += (query.match(/[.*+?^${}()|[\]\\]/g) || []).length * 0.5;
    
    // Word count
    complexity += query.split(/\s+/).length * 0.2;
    
    return Math.min(complexity, 10); // Cap at 10
  }

  private updateAggregatedMetrics(metricName: string): void {
    const dataPoints = this.metrics.get(metricName) || [];
    if (dataPoints.length === 0) return;
    
    const windows = ['1m', '5m', '15m', '1h', '24h'];
    const now = Date.now();
    
    for (const window of windows) {
      const windowMs = this.parseTimeWindow(window);
      const windowData = dataPoints.filter(point => 
        point.timestamp > now - windowMs
      ).map(point => point.value);
      
      if (windowData.length === 0) continue;
      
      const aggregated: AggregatedMetric = {
        name: metricName,
        count: windowData.length,
        sum: windowData.reduce((a, b) => a + b, 0),
        min: Math.min(...windowData),
        max: Math.max(...windowData),
        mean: this.calculateMean(windowData),
        p50: this.calculatePercentile(windowData, 0.5),
        p95: this.calculatePercentile(windowData, 0.95),
        p99: this.calculatePercentile(windowData, 0.99),
        stddev: this.calculateStdDev(windowData),
        timestamp: now,
        window
      };
      
      if (!this.aggregatedMetrics.has(metricName)) {
        this.aggregatedMetrics.set(metricName, []);
      }
      
      const metrics = this.aggregatedMetrics.get(metricName)!;
      const existing = metrics.findIndex(m => m.window === window);
      
      if (existing >= 0) {
        metrics[existing] = aggregated;
      } else {
        metrics.push(aggregated);
      }
    }
  }

  private checkThresholds(metricName: string, value: number): void {
    const relevantThresholds = this.thresholds.filter(t => t.metric === metricName);
    
    for (const threshold of relevantThresholds) {
      const violated = this.checkThresholdViolation(threshold, value);
      const alertId = `${threshold.metric}_${threshold.operator}_${threshold.value}`;
      
      if (violated && !this.activeAlerts.has(alertId)) {
        // Create new alert
        const alert: PerformanceAlert = {
          id: alertId,
          threshold,
          currentValue: value,
          timestamp: Date.now(),
          resolved: false
        };
        
        this.activeAlerts.set(alertId, alert);
        this.emit('alert', alert);
        
      } else if (!violated && this.activeAlerts.has(alertId)) {
        // Resolve existing alert
        const alert = this.activeAlerts.get(alertId)!;
        alert.resolved = true;
        alert.resolvedAt = Date.now();
        this.emit('alertResolved', alert);
      }
    }
  }

  private checkThresholdViolation(threshold: PerformanceThreshold, value: number): boolean {
    switch (threshold.operator) {
      case 'gt': return value > threshold.value;
      case 'gte': return value >= threshold.value;
      case 'lt': return value < threshold.value;
      case 'lte': return value <= threshold.value;
      case 'eq': return value === threshold.value;
      default: return false;
    }
  }

  private getMetricValues(metricName: string, since: number): number[] {
    const dataPoints = this.metrics.get(metricName) || [];
    return dataPoints
      .filter(point => point.timestamp > since)
      .map(point => point.value);
  }

  private getErrorsByType(since: number): Map<string, number> {
    const errorsByType = new Map<string, number>();
    
    for (const [metricName, dataPoints] of this.metrics.entries()) {
      if (metricName.includes('error')) {
        const count = dataPoints.filter(point => point.timestamp > since).length;
        if (count > 0) {
          errorsByType.set(metricName, count);
        }
      }
    }
    
    return errorsByType;
  }

  private parseTimeWindow(window: string): number {
    const match = window.match(/^(\d+)([smhd])$/);
    if (!match) return 300000; // Default 5 minutes
    
    const [, amount, unit] = match;
    const multipliers = { s: 1000, m: 60000, h: 3600000, d: 86400000 };
    
    return parseInt(amount) * (multipliers[unit as keyof typeof multipliers] || 60000);
  }

  private calculateMean(values: number[]): number {
    return values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : 0;
  }

  private calculatePercentile(values: number[], percentile: number): number {
    if (values.length === 0) return 0;
    
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.ceil(sorted.length * percentile) - 1;
    return sorted[Math.max(0, index)];
  }

  private calculateStdDev(values: number[]): number {
    if (values.length <= 1) return 0;
    
    const mean = this.calculateMean(values);
    const squaredDiffs = values.map(value => Math.pow(value - mean, 2));
    const avgSquaredDiff = this.calculateMean(squaredDiffs);
    
    return Math.sqrt(avgSquaredDiff);
  }

  private createEmptyAggregatedMetric(name: string): AggregatedMetric {
    return {
      name,
      count: 0,
      sum: 0,
      min: 0,
      max: 0,
      mean: 0,
      p50: 0,
      p95: 0,
      p99: 0,
      stddev: 0,
      timestamp: Date.now(),
      window: '1h'
    };
  }
}

// Export singleton instance
export const performanceMonitor = PerformanceMonitor.getInstance();