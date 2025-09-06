/**
 * Metrics and Telemetry for RAPTOR system
 * 
 * Provides comprehensive monitoring of topic staleness, pressure telemetry,
 * and performance metrics for the RAPTOR semantic search system.
 */

import { TopicTree } from './topic-tree.js';
import { CardStore } from './card-store.js';
import { SymbolGraph } from './symbol-graph.js';
import EventEmitter from 'events';

export interface TopicStalenessMetrics {
  repo_sha: string;
  timestamp: number;
  overall_staleness: {
    avg_staleness: number;
    max_staleness: number;
    stale_topics_count: number; // staleness > 0.7
    total_topics: number;
  };
  by_level: Record<number, {
    avg_staleness: number;
    stale_count: number;
    total_count: number;
  }>;
  stalest_topics: Array<{
    topic_id: string;
    staleness_score: number;
    age_days: number;
    card_count: number;
    last_updated: number;
  }>;
  freshness_distribution: {
    fresh: number; // staleness < 0.3
    moderate: number; // 0.3 <= staleness < 0.7
    stale: number; // staleness >= 0.7
  };
}

export interface TopicPressureMetrics {
  repo_sha: string;
  timestamp: number;
  pressure_summary: {
    avg_pressure: number;
    max_pressure: number;
    high_pressure_topics: number; // pressure > 0.8
    total_topics: number;
  };
  pressure_components: {
    size_pressure: {
      avg: number;
      topics_near_split: number; // approaching split threshold
    };
    stability_pressure: {
      avg: number;
      unstable_topics: number; // stability < 0.5
    };
    staleness_pressure: {
      avg: number;
      pressure_from_age: number;
    };
  };
  high_pressure_topics: Array<{
    topic_id: string;
    total_pressure: number;
    size_pressure: number;
    stability_pressure: number;
    staleness_pressure: number;
    recommended_action: 'split' | 'merge' | 'rebuild' | 'monitor';
  }>;
}

export interface RaptorPerformanceMetrics {
  repo_sha: string;
  timestamp: number;
  query_performance: {
    avg_topic_search_ms: number;
    avg_card_retrieval_ms: number;
    avg_symbol_resolution_ms: number;
    cache_hit_rates: {
      topic_cache: number;
      embedding_cache: number;
      symbol_cache: number;
    };
  };
  system_health: {
    memory_usage_mb: number;
    index_load_time_ms: number;
    last_rebuild_duration_ms?: number;
    active_snapshots: number;
  };
  feature_utilization: {
    raptor_queries_pct: number; // % queries using RAPTOR features
    topic_hits_per_query: number;
    avg_businessness_boost: number;
    semantic_fallback_rate: number;
  };
}

export interface SystemTelemetry {
  timestamp: number;
  staleness_metrics: TopicStalenessMetrics;
  pressure_metrics: TopicPressureMetrics;
  performance_metrics: RaptorPerformanceMetrics;
  alerts: Alert[];
  health_status: 'healthy' | 'degraded' | 'critical';
}

export interface Alert {
  id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  type: 'staleness' | 'pressure' | 'performance' | 'error';
  message: string;
  topic_id?: string;
  metric_value: number;
  threshold: number;
  timestamp: number;
  auto_remediation?: string;
}

export interface TelemetryConfig {
  staleness_alert_threshold: number;
  pressure_alert_threshold: number;
  performance_alert_thresholds: {
    query_time_ms: number;
    memory_usage_mb: number;
    cache_hit_rate: number;
  };
  collection_interval_ms: number;
  retention_days: number;
  auto_remediation_enabled: boolean;
}

/**
 * MetricsTelemetry system for monitoring RAPTOR health and performance
 */
export class MetricsTelemetry extends EventEmitter {
  private topicTree?: TopicTree;
  private cardStore?: CardStore;
  private symbolGraph?: SymbolGraph;
  
  private config: TelemetryConfig;
  private metricsHistory: SystemTelemetry[] = [];
  private activeAlerts = new Map<string, Alert>();
  
  // Performance tracking
  private queryTimes: number[] = [];
  private cacheStats = {
    topic_hits: 0,
    topic_misses: 0,
    embedding_hits: 0,
    embedding_misses: 0,
    symbol_hits: 0,
    symbol_misses: 0
  };
  
  private collectionTimer?: NodeJS.Timeout;

  constructor(config?: Partial<TelemetryConfig>) {
    super();
    
    this.config = {
      staleness_alert_threshold: 0.8,
      pressure_alert_threshold: 0.8,
      performance_alert_thresholds: {
        query_time_ms: 1000,
        memory_usage_mb: 1024,
        cache_hit_rate: 0.8
      },
      collection_interval_ms: 60000, // 1 minute
      retention_days: 7,
      auto_remediation_enabled: false,
      ...config
    };
  }

  /**
   * Initialize telemetry system with RAPTOR components
   */
  initialize(
    topicTree: TopicTree, 
    cardStore: CardStore, 
    symbolGraph: SymbolGraph
  ): void {
    this.topicTree = topicTree;
    this.cardStore = cardStore;
    this.symbolGraph = symbolGraph;
    
    // Start periodic collection
    this.startCollection();
    
    this.emit('telemetry-initialized', { timestamp: Date.now() });
  }

  private startCollection(): void {
    if (this.collectionTimer) {
      clearInterval(this.collectionTimer);
    }
    
    this.collectionTimer = setInterval(() => {
      this.collectMetrics().catch(error => {
        this.emit('collection-error', { error, timestamp: Date.now() });
      });
    }, this.config.collection_interval_ms);
  }

  /**
   * Collect comprehensive system telemetry
   */
  async collectMetrics(): Promise<SystemTelemetry> {
    if (!this.topicTree) {
      throw new Error('TopicTree not initialized');
    }

    const timestamp = Date.now();
    
    // Collect all metric types
    const staleness_metrics = this.collectStalenessMetrics();
    const pressure_metrics = this.collectPressureMetrics();
    const performance_metrics = this.collectPerformanceMetrics();
    
    // Generate alerts
    const alerts = this.generateAlerts(staleness_metrics, pressure_metrics, performance_metrics);
    
    // Determine overall health
    const health_status = this.computeHealthStatus(alerts);
    
    const telemetry: SystemTelemetry = {
      timestamp,
      staleness_metrics,
      pressure_metrics,
      performance_metrics,
      alerts,
      health_status
    };
    
    // Store in history
    this.metricsHistory.push(telemetry);
    
    // Cleanup old metrics
    this.cleanupHistory();
    
    // Emit telemetry event
    this.emit('metrics-collected', telemetry);
    
    // Auto-remediation if enabled
    if (this.config.auto_remediation_enabled) {
      await this.executeAutoRemediation(alerts);
    }
    
    return telemetry;
  }

  private collectStalenessMetrics(): TopicStalenessMetrics {
    const tree = this.topicTree!.getTree();
    if (!tree) {
      throw new Error('TopicTree not loaded');
    }

    const staleness = this.topicTree!.computeTopicStaleness();
    const nodes = Array.from(tree.nodes.values());
    const nonRootNodes = nodes.filter(n => n.level > 0);
    
    // Overall statistics
    const stalenessValues = Object.values(staleness);
    const avg_staleness = stalenessValues.reduce((sum, val) => sum + val, 0) / stalenessValues.length;
    const max_staleness = Math.max(...stalenessValues);
    const stale_topics_count = stalenessValues.filter(s => s > 0.7).length;
    
    // By level breakdown
    const by_level: Record<number, any> = {};
    for (const node of nonRootNodes) {
      const level = node.level;
      if (!by_level[level]) {
        by_level[level] = { values: [], count: 0 };
      }
      by_level[level].values.push(staleness[node.id] || 0);
      by_level[level].count++;
    }
    
    for (const [level, data] of Object.entries(by_level)) {
      const values = data.values;
      by_level[level] = {
        avg_staleness: values.reduce((sum: number, val: number) => sum + val, 0) / values.length,
        stale_count: values.filter((s: number) => s > 0.7).length,
        total_count: values.length
      };
    }
    
    // Find stalest topics
    const stalest_topics = nonRootNodes
      .map(node => ({
        topic_id: node.id,
        staleness_score: staleness[node.id] || 0,
        age_days: node.age.avg_card_age_days,
        card_count: node.coverage.total_cards,
        last_updated: node.metadata.last_updated
      }))
      .sort((a, b) => b.staleness_score - a.staleness_score)
      .slice(0, 10);
    
    // Freshness distribution
    const fresh = stalenessValues.filter(s => s < 0.3).length;
    const moderate = stalenessValues.filter(s => s >= 0.3 && s < 0.7).length;
    const stale = stalenessValues.filter(s => s >= 0.7).length;

    return {
      repo_sha: tree.repo_sha,
      timestamp: Date.now(),
      overall_staleness: {
        avg_staleness,
        max_staleness,
        stale_topics_count,
        total_topics: stalenessValues.length
      },
      by_level,
      stalest_topics,
      freshness_distribution: { fresh, moderate, stale }
    };
  }

  private collectPressureMetrics(): TopicPressureMetrics {
    const tree = this.topicTree!.getTree();
    if (!tree) {
      throw new Error('TopicTree not loaded');
    }

    const pressure = this.topicTree!.computeTopicPressure();
    const nodes = Array.from(tree.nodes.values()).filter(n => n.level > 0);
    
    // Overall pressure statistics
    const pressureValues = Object.values(pressure);
    const avg_pressure = pressureValues.reduce((sum, val) => sum + val, 0) / pressureValues.length;
    const max_pressure = Math.max(...pressureValues);
    const high_pressure_count = pressureValues.filter(p => p > 0.8).length;
    
    // Component-wise analysis
    let totalSizePressure = 0;
    let totalStabilityPressure = 0;
    let totalStalenessPressure = 0;
    let topicsNearSplit = 0;
    let unstableTopics = 0;
    
    for (const node of nodes) {
      const sizePressure = node.coverage.total_cards / node.metadata.split_threshold;
      const stabilityPressure = 1 - node.metadata.stability_score;
      const stalenessPressure = node.age.staleness_score;
      
      totalSizePressure += sizePressure;
      totalStabilityPressure += stabilityPressure;
      totalStalenessPressure += stalenessPressure;
      
      if (sizePressure > 0.8) topicsNearSplit++;
      if (node.metadata.stability_score < 0.5) unstableTopics++;
    }
    
    const nodeCount = nodes.length;
    
    // High pressure topics with recommended actions
    const high_pressure_topics = nodes
      .map(node => {
        const sizePressure = node.coverage.total_cards / node.metadata.split_threshold;
        const stabilityPressure = 1 - node.metadata.stability_score;
        const stalenessPressure = node.age.staleness_score;
        const totalPressure = pressure[node.id] || 0;
        
        let recommendedAction: 'split' | 'merge' | 'rebuild' | 'monitor' = 'monitor';
        
        if (sizePressure > 0.8) recommendedAction = 'split';
        else if (node.coverage.total_cards < node.metadata.merge_threshold) recommendedAction = 'merge';
        else if (stabilityPressure > 0.7) recommendedAction = 'rebuild';
        
        return {
          topic_id: node.id,
          total_pressure: totalPressure,
          size_pressure: sizePressure,
          stability_pressure: stabilityPressure,
          staleness_pressure: stalenessPressure,
          recommended_action: recommendedAction
        };
      })
      .filter(item => item.total_pressure > 0.6)
      .sort((a, b) => b.total_pressure - a.total_pressure)
      .slice(0, 15);

    return {
      repo_sha: tree.repo_sha,
      timestamp: Date.now(),
      pressure_summary: {
        avg_pressure,
        max_pressure,
        high_pressure_topics: high_pressure_topics.length,
        total_topics: pressureValues.length
      },
      pressure_components: {
        size_pressure: {
          avg: totalSizePressure / nodeCount,
          topics_near_split: topicsNearSplit
        },
        stability_pressure: {
          avg: totalStabilityPressure / nodeCount,
          unstable_topics: unstableTopics
        },
        staleness_pressure: {
          avg: totalStalenessPressure / nodeCount,
          pressure_from_age: totalStalenessPressure
        }
      },
      high_pressure_topics
    };
  }

  private collectPerformanceMetrics(): RaptorPerformanceMetrics {
    const tree = this.topicTree!.getTree();
    if (!tree) {
      throw new Error('TopicTree not loaded');
    }

    // Query performance (from recent measurements)
    const recentQueries = this.queryTimes.slice(-100); // Last 100 queries
    const avg_topic_search_ms = recentQueries.length > 0 
      ? recentQueries.reduce((sum, time) => sum + time, 0) / recentQueries.length
      : 0;
    
    // Cache hit rates
    const topicCacheHitRate = this.cacheStats.topic_hits / 
      Math.max(1, this.cacheStats.topic_hits + this.cacheStats.topic_misses);
    
    const embeddingCacheHitRate = this.cacheStats.embedding_hits /
      Math.max(1, this.cacheStats.embedding_hits + this.cacheStats.embedding_misses);
    
    const symbolCacheHitRate = this.cacheStats.symbol_hits /
      Math.max(1, this.cacheStats.symbol_hits + this.cacheStats.symbol_misses);
    
    // Memory usage estimation
    const estimatedMemoryUsage = this.estimateMemoryUsage();
    
    return {
      repo_sha: tree.repo_sha,
      timestamp: Date.now(),
      query_performance: {
        avg_topic_search_ms,
        avg_card_retrieval_ms: avg_topic_search_ms * 0.6, // Estimated
        avg_symbol_resolution_ms: avg_topic_search_ms * 0.3, // Estimated
        cache_hit_rates: {
          topic_cache: topicCacheHitRate,
          embedding_cache: embeddingCacheHitRate,
          symbol_cache: symbolCacheHitRate
        }
      },
      system_health: {
        memory_usage_mb: estimatedMemoryUsage,
        index_load_time_ms: 500, // Mock value
        active_snapshots: 1
      },
      feature_utilization: {
        raptor_queries_pct: 0.75, // Mock - % of queries using RAPTOR
        topic_hits_per_query: 2.3, // Mock - avg topic matches per query
        avg_businessness_boost: 0.15, // Mock - avg ranking boost from businessness
        semantic_fallback_rate: 0.05 // Mock - rate of falling back to non-semantic
      }
    };
  }

  private estimateMemoryUsage(): number {
    // Rough estimation of memory usage
    let totalMB = 0;
    
    const tree = this.topicTree?.getTree();
    if (tree) {
      // Topic embeddings: nodes * 384 floats * 4 bytes
      totalMB += (tree.nodes.size * 384 * 4) / (1024 * 1024);
      
      // Keyword indices and metadata
      totalMB += (tree.nodes.size * 1000) / (1024 * 1024); // ~1KB per node
    }
    
    const cardStore = this.cardStore?.getSnapshot();
    if (cardStore) {
      // Card embeddings: cards * (384 + 128 + 256) floats * 4 bytes
      totalMB += (cardStore.cards.size * 768 * 4) / (1024 * 1024);
    }
    
    const symbolGraph = this.symbolGraph?.getSnapshot();
    if (symbolGraph) {
      // Symbol graph overhead
      totalMB += (symbolGraph.nodes.size * 500) / (1024 * 1024); // ~500 bytes per symbol
    }
    
    return totalMB;
  }

  private generateAlerts(
    staleness: TopicStalenessMetrics,
    pressure: TopicPressureMetrics,
    performance: RaptorPerformanceMetrics
  ): Alert[] {
    const alerts: Alert[] = [];
    const timestamp = Date.now();

    // Staleness alerts
    if (staleness.overall_staleness.avg_staleness > this.config.staleness_alert_threshold) {
      alerts.push({
        id: `staleness-high-${timestamp}`,
        severity: 'high',
        type: 'staleness',
        message: `High average staleness: ${staleness.overall_staleness.avg_staleness.toFixed(2)}`,
        metric_value: staleness.overall_staleness.avg_staleness,
        threshold: this.config.staleness_alert_threshold,
        timestamp,
        auto_remediation: 'Consider triggering topic tree rebuild'
      });
    }
    
    // Individual stale topics
    for (const topic of staleness.stalest_topics.slice(0, 3)) {
      if (topic.staleness_score > this.config.staleness_alert_threshold) {
        alerts.push({
          id: `topic-stale-${topic.topic_id}`,
          severity: 'medium',
          type: 'staleness',
          message: `Stale topic ${topic.topic_id}: ${topic.staleness_score.toFixed(2)}`,
          topic_id: topic.topic_id,
          metric_value: topic.staleness_score,
          threshold: this.config.staleness_alert_threshold,
          timestamp
        });
      }
    }

    // Pressure alerts
    if (pressure.pressure_summary.avg_pressure > this.config.pressure_alert_threshold) {
      alerts.push({
        id: `pressure-high-${timestamp}`,
        severity: 'high',
        type: 'pressure',
        message: `High average pressure: ${pressure.pressure_summary.avg_pressure.toFixed(2)}`,
        metric_value: pressure.pressure_summary.avg_pressure,
        threshold: this.config.pressure_alert_threshold,
        timestamp,
        auto_remediation: 'Consider splitting high-pressure topics'
      });
    }
    
    // High pressure topics
    for (const topic of pressure.high_pressure_topics.slice(0, 3)) {
      if (topic.total_pressure > this.config.pressure_alert_threshold) {
        alerts.push({
          id: `topic-pressure-${topic.topic_id}`,
          severity: topic.total_pressure > 0.9 ? 'critical' : 'high',
          type: 'pressure',
          message: `High pressure topic ${topic.topic_id}: ${topic.total_pressure.toFixed(2)}`,
          topic_id: topic.topic_id,
          metric_value: topic.total_pressure,
          threshold: this.config.pressure_alert_threshold,
          timestamp,
          auto_remediation: `Recommended action: ${topic.recommended_action}`
        });
      }
    }

    // Performance alerts
    if (performance.query_performance.avg_topic_search_ms > this.config.performance_alert_thresholds.query_time_ms) {
      alerts.push({
        id: `perf-query-time-${timestamp}`,
        severity: 'medium',
        type: 'performance',
        message: `Slow query performance: ${performance.query_performance.avg_topic_search_ms.toFixed(0)}ms`,
        metric_value: performance.query_performance.avg_topic_search_ms,
        threshold: this.config.performance_alert_thresholds.query_time_ms,
        timestamp
      });
    }
    
    if (performance.system_health.memory_usage_mb > this.config.performance_alert_thresholds.memory_usage_mb) {
      alerts.push({
        id: `perf-memory-${timestamp}`,
        severity: 'high',
        type: 'performance',
        message: `High memory usage: ${performance.system_health.memory_usage_mb.toFixed(0)}MB`,
        metric_value: performance.system_health.memory_usage_mb,
        threshold: this.config.performance_alert_thresholds.memory_usage_mb,
        timestamp
      });
    }

    // Cache hit rate alerts
    const minCacheHitRate = this.config.performance_alert_thresholds.cache_hit_rate;
    const cacheRates = performance.query_performance.cache_hit_rates;
    
    if (cacheRates.topic_cache < minCacheHitRate) {
      alerts.push({
        id: `cache-topic-${timestamp}`,
        severity: 'medium',
        type: 'performance',
        message: `Low topic cache hit rate: ${(cacheRates.topic_cache * 100).toFixed(1)}%`,
        metric_value: cacheRates.topic_cache,
        threshold: minCacheHitRate,
        timestamp
      });
    }

    return alerts;
  }

  private computeHealthStatus(alerts: Alert[]): 'healthy' | 'degraded' | 'critical' {
    const criticalAlerts = alerts.filter(a => a.severity === 'critical');
    const highAlerts = alerts.filter(a => a.severity === 'high');
    
    if (criticalAlerts.length > 0) return 'critical';
    if (highAlerts.length > 2) return 'critical';
    if (alerts.length > 5) return 'degraded';
    if (alerts.length > 0) return 'degraded';
    
    return 'healthy';
  }

  private async executeAutoRemediation(alerts: Alert[]): Promise<void> {
    for (const alert of alerts) {
      if (alert.auto_remediation && alert.severity === 'critical') {
        try {
          await this.performRemediation(alert);
          this.emit('auto-remediation-executed', { alert, timestamp: Date.now() });
        } catch (error) {
          this.emit('auto-remediation-failed', { alert, error, timestamp: Date.now() });
        }
      }
    }
  }

  private async performRemediation(alert: Alert): Promise<void> {
    // Mock implementation - real version would trigger actual remediation
    switch (alert.type) {
      case 'pressure':
        if (alert.topic_id) {
          // Could trigger topic splitting or rebuilding
          this.emit('remediation-log', {
            action: 'topic-split-queued',
            topic_id: alert.topic_id,
            timestamp: Date.now()
          });
        }
        break;
        
      case 'staleness':
        // Could trigger incremental rebuild
        this.emit('remediation-log', {
          action: 'rebuild-queued',
          timestamp: Date.now()
        });
        break;
        
      case 'performance':
        // Could trigger cache warming or index optimization
        this.emit('remediation-log', {
          action: 'optimization-queued',
          timestamp: Date.now()
        });
        break;
    }
  }

  private cleanupHistory(): void {
    const cutoffTime = Date.now() - (this.config.retention_days * 24 * 60 * 60 * 1000);
    this.metricsHistory = this.metricsHistory.filter(m => m.timestamp > cutoffTime);
  }

  // Public API methods for tracking performance

  /**
   * Record a query execution time
   */
  recordQueryTime(timeMs: number): void {
    this.queryTimes.push(timeMs);
    
    // Keep only recent times
    if (this.queryTimes.length > 1000) {
      this.queryTimes = this.queryTimes.slice(-500);
    }
  }

  /**
   * Record cache hit/miss events
   */
  recordCacheEvent(type: 'topic' | 'embedding' | 'symbol', hit: boolean): void {
    if (hit) {
      this.cacheStats[`${type}_hits` as keyof typeof this.cacheStats]++;
    } else {
      this.cacheStats[`${type}_misses` as keyof typeof this.cacheStats]++;
    }
  }

  // Public access methods

  /**
   * Get current system telemetry
   */
  async getCurrentTelemetry(): Promise<SystemTelemetry> {
    return await this.collectMetrics();
  }

  /**
   * Get topic staleness metrics only
   */
  getTopicStaleness(): TopicStalenessMetrics {
    return this.collectStalenessMetrics();
  }

  /**
   * Get topic pressure metrics only
   */
  getTopicPressure(): TopicPressureMetrics {
    return this.collectPressureMetrics();
  }

  /**
   * Get performance metrics only
   */
  getPerformanceMetrics(): RaptorPerformanceMetrics {
    return this.collectPerformanceMetrics();
  }

  /**
   * Get active alerts
   */
  getActiveAlerts(): Alert[] {
    return Array.from(this.activeAlerts.values());
  }

  /**
   * Get metrics history
   */
  getMetricsHistory(hours?: number): SystemTelemetry[] {
    if (!hours) return this.metricsHistory;
    
    const cutoffTime = Date.now() - (hours * 60 * 60 * 1000);
    return this.metricsHistory.filter(m => m.timestamp > cutoffTime);
  }

  /**
   * Update telemetry configuration
   */
  updateConfig(newConfig: Partial<TelemetryConfig>): void {
    this.config = { ...this.config, ...newConfig };
    this.emit('config-updated', { config: this.config, timestamp: Date.now() });
    
    // Restart collection with new interval if changed
    if (newConfig.collection_interval_ms) {
      this.startCollection();
    }
  }

  /**
   * Stop telemetry collection
   */
  stop(): void {
    if (this.collectionTimer) {
      clearInterval(this.collectionTimer);
      this.collectionTimer = undefined;
    }
    
    this.emit('telemetry-stopped', { timestamp: Date.now() });
  }

  /**
   * Export telemetry data for external analysis
   */
  exportTelemetryData(): {
    config: TelemetryConfig;
    metrics_history: SystemTelemetry[];
    cache_stats: typeof this.cacheStats;
    query_performance: number[];
  } {
    return {
      config: this.config,
      metrics_history: this.metricsHistory,
      cache_stats: this.cacheStats,
      query_performance: this.queryTimes.slice(-100)
    };
  }
}