/**
 * Search Engine Telemetry Integration
 * 
 * Integrates RAPTOR telemetry with the existing search pipeline.
 * Automatically captures query events, mix breakdowns, and topic explanations.
 */

import { raptorTelemetry, type QueryEvent, type MixBreakdown } from './raptor-metrics.js';
import { QueryIntent, IntentClassification } from '../types/core.js';
import { TopicPlanningResult } from '../raptor/stage-a-planner.js';

export interface SearchTelemetryConfig {
  enabled: boolean;
  record_topic_explanations: boolean;
  record_mix_breakdown: boolean;
  record_performance_metrics: boolean;
  min_query_length: number;
}

/**
 * Search telemetry integration wrapper
 */
export class SearchTelemetryIntegration {
  private config: SearchTelemetryConfig;
  private activeQueries: Map<string, { startTime: number; query: string }> = new Map();

  constructor(config?: Partial<SearchTelemetryConfig>) {
    this.config = {
      enabled: true,
      record_topic_explanations: true,
      record_mix_breakdown: true,
      record_performance_metrics: true,
      min_query_length: 3,
      ...config
    };
  }

  /**
   * Start tracking a search query
   */
  startQuery(queryId: string, query: string): void {
    if (!this.config.enabled || query.length < this.config.min_query_length) {
      return;
    }

    this.activeQueries.set(queryId, {
      startTime: Date.now(),
      query
    });
  }

  /**
   * Complete tracking and record telemetry
   */
  completeQuery(
    queryId: string,
    intent: IntentClassification,
    results: SearchResult[],
    topicPlanning?: TopicPlanningResult
  ): void {
    if (!this.config.enabled) {
      return;
    }

    const queryInfo = this.activeQueries.get(queryId);
    if (!queryInfo) {
      console.warn(`Query ${queryId} not found in active queries`);
      return;
    }

    const latencyMs = Date.now() - queryInfo.startTime;
    
    // Record the query event
    const queryEvent: QueryEvent = {
      query: queryInfo.query,
      intent: intent.intent,
      results: results.map(result => ({
        file_path: result.file_path,
        score: result.score,
        mix_breakdown: result.mix_breakdown || this.createDefaultMixBreakdown(),
        topic_ids: result.topic_ids
      })),
      latency_ms: latencyMs,
      timestamp: new Date()
    };

    raptorTelemetry.recordQuery(queryEvent);

    // Record topic planning metrics if available
    if (topicPlanning && this.config.record_topic_explanations) {
      this.recordTopicPlanningTelemetry(topicPlanning, queryEvent);
    }

    // Clean up
    this.activeQueries.delete(queryId);
  }

  private recordTopicPlanningTelemetry(
    planning: TopicPlanningResult,
    queryEvent: QueryEvent
  ): void {
    // Record topic hit if topic boost was applied
    if (planning.topic_boost_applied) {
      // This would be handled by the main recordQuery call
    }

    // Record planning performance
    if (planning.planning_time_ms > 50) {
      console.log(`Slow topic planning: ${planning.planning_time_ms}ms for query: ${queryEvent.query}`);
    }
  }

  /**
   * Record symbol resolution event
   */
  recordSymbolResolution(symbol: string, resolvedDepth: number, typeMatched: boolean): void {
    if (!this.config.enabled) {
      return;
    }

    // Record alias resolution depth
    raptorTelemetry.recordAliasResolution(resolvedDepth);

    // Record type match
    raptorTelemetry.recordTypeMatch(typeMatched);
  }

  /**
   * Record system health update
   */
  recordSystemHealth(indexStaleness: number, systemPressure: number): void {
    if (!this.config.enabled) {
      return;
    }

    raptorTelemetry.updateSystemHealth(indexStaleness, systemPressure);
  }

  /**
   * Record topic tree recluster
   */
  recordTopicRecluster(): void {
    if (!this.config.enabled) {
      return;
    }

    raptorTelemetry.recordRecluster();
  }

  private createDefaultMixBreakdown(): MixBreakdown {
    return {
      exact: 0,
      fuzzy: 0,
      symbol: 0,
      struct: 0,
      semantic: 0,
      topic_hit: 0
    };
  }

  /**
   * Enable/disable telemetry
   */
  setEnabled(enabled: boolean): void {
    this.config.enabled = enabled;
    console.log(`Search telemetry ${enabled ? 'enabled' : 'disabled'}`);
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<SearchTelemetryConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get telemetry statistics
   */
  getStats(): {
    active_queries: number;
    config: SearchTelemetryConfig;
    uptime_ms: number;
  } {
    return {
      active_queries: this.activeQueries.size,
      config: { ...this.config },
      uptime_ms: process.uptime() * 1000
    };
  }
}

export interface SearchResult {
  file_path: string;
  score: number;
  mix_breakdown?: MixBreakdown;
  topic_ids?: string[];
  rank?: number;
  line_number?: number;
  snippet?: string;
}

/**
 * Telemetry-aware search result builder
 */
export class TelemetrySearchResultBuilder {
  private results: SearchResult[] = [];

  addResult(
    filePath: string,
    score: number,
    mixBreakdown: Partial<MixBreakdown>,
    topicIds?: string[]
  ): this {
    this.results.push({
      file_path: filePath,
      score,
      mix_breakdown: {
        exact: 0,
        fuzzy: 0,
        symbol: 0,
        struct: 0,
        semantic: 0,
        topic_hit: 0,
        ...mixBreakdown
      },
      topic_ids: topicIds,
      rank: this.results.length + 1
    });

    return this;
  }

  addExactMatch(filePath: string, score: number): this {
    return this.addResult(filePath, score, { exact: 1.0 });
  }

  addFuzzyMatch(filePath: string, score: number): this {
    return this.addResult(filePath, score, { fuzzy: 1.0 });
  }

  addSymbolMatch(filePath: string, score: number, topicIds?: string[]): this {
    return this.addResult(filePath, score, { symbol: 1.0 }, topicIds);
  }

  addSemanticMatch(filePath: string, score: number, topicIds?: string[]): this {
    return this.addResult(filePath, score, { semantic: 1.0 }, topicIds);
  }

  addTopicMatch(filePath: string, score: number, topicIds: string[]): this {
    return this.addResult(filePath, score, { topic_hit: 1.0 }, topicIds);
  }

  addHybridMatch(
    filePath: string,
    score: number,
    mixBreakdown: MixBreakdown,
    topicIds?: string[]
  ): this {
    return this.addResult(filePath, score, mixBreakdown, topicIds);
  }

  build(): SearchResult[] {
    // Sort by score descending
    const sortedResults = this.results.sort((a, b) => b.score - a.score);
    
    // Update ranks
    sortedResults.forEach((result, index) => {
      result.rank = index + 1;
    });

    return sortedResults;
  }

  clear(): this {
    this.results = [];
    return this;
  }

  getCount(): number {
    return this.results.length;
  }
}

// Global search telemetry integration instance
export const searchTelemetry = new SearchTelemetryIntegration();

/**
 * Utility functions for common telemetry patterns
 */
export const TelemetryUtils = {
  /**
   * Generate unique query ID
   */
  generateQueryId(): string {
    return `q_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  },

  /**
   * Create mix breakdown from search components
   */
  createMixBreakdown(components: {
    exact?: number;
    fuzzy?: number;
    symbol?: number;
    struct?: number;
    semantic?: number;
    topic_hit?: number;
  }): MixBreakdown {
    return {
      exact: components.exact || 0,
      fuzzy: components.fuzzy || 0,
      symbol: components.symbol || 0,
      struct: components.struct || 0,
      semantic: components.semantic || 0,
      topic_hit: components.topic_hit || 0
    };
  },

  /**
   * Normalize mix breakdown to sum to 1.0
   */
  normalizeMixBreakdown(mix: MixBreakdown): MixBreakdown {
    const total = mix.exact + mix.fuzzy + mix.symbol + mix.struct + mix.semantic + mix.topic_hit;
    
    if (total === 0) {
      return mix;
    }

    return {
      exact: mix.exact / total,
      fuzzy: mix.fuzzy / total,
      symbol: mix.symbol / total,
      struct: mix.struct / total,
      semantic: mix.semantic / total,
      topic_hit: mix.topic_hit / total
    };
  },

  /**
   * Check if query meets telemetry recording criteria
   */
  shouldRecordQuery(query: string, intent: string): boolean {
    return query.length >= 3 && !query.startsWith('test_');
  },

  /**
   * Extract topic IDs from RAPTOR planning result
   */
  extractTopicIds(planning?: TopicPlanningResult): string[] {
    if (!planning?.topic_matches) {
      return [];
    }

    return planning.topic_matches
      .filter(match => match.boost_contribution > 0.1)
      .map(match => match.topic_id);
  }
};

export default searchTelemetry;