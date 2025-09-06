/**
 * ROI-Aware Result Micro-Cache - Embedder-Agnostic Optimization #3
 * 
 * Sharded, TTL'd result cache keyed by (intent, canonical_query, topic_bin)
 * Value: {topN, why-mix, ECE slice}
 * On cache hit with SLA headroom tight: return cached top-k, skip expensive ANN, or k-merge with small efSearch
 * 
 * Target: p95 -0.5 to -1.0ms with flat nDCG
 */

import type { SearchHit } from '../core/span_resolver/types.js';
import type { SearchContext } from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';

export interface MicroCacheConfig {
  enabled: boolean;
  shardCount: number;                  // Number of cache shards
  ttlSeconds: number;                  // TTL for cache entries (1-3s)
  maxEntriesPerShard: number;          // LRU limit per shard
  slaHeadroomThresholdMs: number;      // SLA headroom threshold for cache hit path
  enableKMerge: boolean;               // Enable k-merge with small efSearch on cache hit
  kMergeRatio: number;                 // Ratio of cached results vs new ANN results
  canonicalizationEnabled: boolean;    // Enable query canonicalization
  spanInvariantCheck: boolean;         // Validate span invariants on cache hit
  maxCachedResults: number;            // Max results to cache per query
}

export interface CacheKey {
  intent: string;                      // Query intent classification
  canonicalQuery: string;              // Canonicalized query
  topicId: string;                     // Topic bin identifier
  language?: string;                   // Programming language filter
  indexVersion: string;                // Index version for invalidation
}

export interface CacheValue {
  topN: SearchHit[];                   // Top-N cached results
  whyMix: Map<string, number>;         // Match reason distribution
  eceSlice: {                          // ECE (Expected Calibration Error) slice
    confidence_bins: number[];
    accuracy_per_bin: number[];
    calibration_score: number;
  };
  timestamp: number;                   // Cache entry timestamp
  hitCount: number;                    // Usage counter
  averageLatencySaved: number;         // Estimated latency savings
}

export interface CacheStats {
  totalHits: number;
  totalMisses: number;
  hitRate: number;
  averageLatencySaved: number;
  totalEntriesCount: number;
  evictionsCount: number;
  spanInvariantViolations: number;
}

/**
 * Query canonicalizer for consistent cache keys
 */
export class QueryCanonicalizer {
  private aliasMap: Map<string, string> = new Map();

  constructor() {
    this.initializeAliases();
  }

  /**
   * Initialize common aliases and synonyms
   */
  private initializeAliases(): void {
    // Common programming synonyms
    this.aliasMap.set('func', 'function');
    this.aliasMap.set('fn', 'function');
    this.aliasMap.set('method', 'function');
    this.aliasMap.set('def', 'definition');
    this.aliasMap.set('var', 'variable');
    this.aliasMap.set('const', 'constant');
    this.aliasMap.set('impl', 'implementation');
    this.aliasMap.set('interface', 'type');
    this.aliasMap.set('struct', 'type');
  }

  /**
   * Canonicalize query string for consistent cache keys
   */
  canonicalize(query: string): string {
    let canonical = query.toLowerCase().trim();

    // Remove extra whitespace
    canonical = canonical.replace(/\s+/g, ' ');

    // Strip common punctuation
    canonical = canonical.replace(/[.,;:!?'"(){}[\]]/g, ' ').trim();

    // Normalize numerics (version numbers, etc)
    canonical = canonical.replace(/\d+\.\d+(\.\d+)?/g, 'VERSION');
    canonical = canonical.replace(/\b\d+\b/g, 'NUMBER');

    // Resolve aliases
    const tokens = canonical.split(' ');
    const resolvedTokens = tokens.map(token => this.aliasMap.get(token) || token);
    canonical = resolvedTokens.join(' ');

    // Sort tokens for consistent ordering (except for quoted phrases)
    if (!canonical.includes('"') && !canonical.includes("'")) {
      const sortedTokens = canonical.split(' ').filter(t => t.length > 0).sort();
      canonical = sortedTokens.join(' ');
    }

    return canonical;
  }

  /**
   * Add new alias mapping
   */
  addAlias(alias: string, canonical: string): void {
    this.aliasMap.set(alias.toLowerCase(), canonical.toLowerCase());
  }
}

/**
 * Intent classifier for cache key generation
 */
export class IntentClassifier {
  /**
   * Classify query intent for cache partitioning
   */
  classifyIntent(query: string, context: SearchContext): string {
    const lowerQuery = query.toLowerCase();

    // Natural language queries
    if (/\b(how|what|where|when|why|find|search|get|show|explain)\b/.test(lowerQuery)) {
      return 'NL_query';
    }

    // Symbol-specific queries
    if (/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(query.trim())) {
      return 'symbol_lookup';
    }

    // Function/method queries
    if (lowerQuery.includes('function') || lowerQuery.includes('method') || lowerQuery.includes('def')) {
      return 'function_search';
    }

    // Type/class queries
    if (lowerQuery.includes('class') || lowerQuery.includes('type') || lowerQuery.includes('interface')) {
      return 'type_search';
    }

    // Pattern/structural queries
    if (lowerQuery.includes('pattern') || lowerQuery.includes('struct') || /\b(for|while|if|try|catch)\b/.test(lowerQuery)) {
      return 'structural_search';
    }

    // Configuration/settings queries
    if (lowerQuery.includes('config') || lowerQuery.includes('setting') || lowerQuery.includes('option')) {
      return 'config_search';
    }

    // Error/debug queries
    if (lowerQuery.includes('error') || lowerQuery.includes('bug') || lowerQuery.includes('debug')) {
      return 'error_search';
    }

    return 'general_search';
  }
}

/**
 * Topic bin generator for cache partitioning
 */
export class TopicBinGenerator {
  private topicBins: Map<string, string> = new Map();

  /**
   * Generate topic bin for cache key
   */
  generateTopicBin(query: string, context: SearchContext): string {
    const cached = this.topicBins.get(query);
    if (cached) {
      return cached;
    }

    const lowerQuery = query.toLowerCase();
    let topicBin = 'general';

    // Language-specific topics
    if (context.query.includes('typescript') || context.query.includes('ts')) {
      topicBin = 'typescript';
    } else if (lowerQuery.includes('python') || lowerQuery.includes('py')) {
      topicBin = 'python';  
    } else if (lowerQuery.includes('javascript') || lowerQuery.includes('js')) {
      topicBin = 'javascript';
    } else if (lowerQuery.includes('rust') || lowerQuery.includes('rs')) {
      topicBin = 'rust';
    } else if (lowerQuery.includes('go') || lowerQuery.includes('golang')) {
      topicBin = 'go';
    }

    // Framework-specific topics
    if (lowerQuery.includes('react') || lowerQuery.includes('jsx')) {
      topicBin = 'react';
    } else if (lowerQuery.includes('node') || lowerQuery.includes('npm')) {
      topicBin = 'nodejs';
    } else if (lowerQuery.includes('api') || lowerQuery.includes('endpoint')) {
      topicBin = 'api';
    }

    // Domain-specific topics
    if (lowerQuery.includes('auth') || lowerQuery.includes('login') || lowerQuery.includes('security')) {
      topicBin = 'security';
    } else if (lowerQuery.includes('db') || lowerQuery.includes('database') || lowerQuery.includes('sql')) {
      topicBin = 'database';
    } else if (lowerQuery.includes('ui') || lowerQuery.includes('component') || lowerQuery.includes('style')) {
      topicBin = 'ui';
    }

    this.topicBins.set(query, topicBin);
    return topicBin;
  }
}

/**
 * LRU cache shard with TTL support
 */
export class CacheShard {
  private cache: Map<string, CacheValue> = new Map();
  private accessOrder: string[] = [];
  private maxEntries: number;
  private ttlMs: number;

  constructor(maxEntries: number, ttlSeconds: number) {
    this.maxEntries = maxEntries;
    this.ttlMs = ttlSeconds * 1000;
  }

  /**
   * Get cached value if valid
   */
  get(key: string): CacheValue | null {
    const value = this.cache.get(key);
    if (!value) {
      return null;
    }

    // Check TTL
    if (Date.now() - value.timestamp > this.ttlMs) {
      this.cache.delete(key);
      this.removeFromAccessOrder(key);
      return null;
    }

    // Update access order
    this.moveToFront(key);
    value.hitCount++;
    
    return value;
  }

  /**
   * Set cached value
   */
  set(key: string, value: CacheValue): void {
    // Remove if already exists
    if (this.cache.has(key)) {
      this.removeFromAccessOrder(key);
    }

    // Add to front
    this.cache.set(key, value);
    this.accessOrder.unshift(key);

    // Evict LRU if over capacity
    while (this.accessOrder.length > this.maxEntries) {
      const lruKey = this.accessOrder.pop()!;
      this.cache.delete(lruKey);
    }
  }

  /**
   * Move key to front of access order
   */
  private moveToFront(key: string): void {
    this.removeFromAccessOrder(key);
    this.accessOrder.unshift(key);
  }

  /**
   * Remove key from access order
   */
  private removeFromAccessOrder(key: string): void {
    const index = this.accessOrder.indexOf(key);
    if (index !== -1) {
      this.accessOrder.splice(index, 1);
    }
  }

  /**
   * Get shard statistics
   */
  getStats(): { entries: number; oldestEntry: number; newestEntry: number } {
    let oldestTimestamp = Date.now();
    let newestTimestamp = 0;

    for (const value of this.cache.values()) {
      if (value.timestamp < oldestTimestamp) oldestTimestamp = value.timestamp;
      if (value.timestamp > newestTimestamp) newestTimestamp = value.timestamp;
    }

    return {
      entries: this.cache.size,
      oldestEntry: Date.now() - oldestTimestamp,
      newestEntry: Date.now() - newestTimestamp
    };
  }

  /**
   * Clear all entries
   */
  clear(): void {
    this.cache.clear();
    this.accessOrder = [];
  }
}

/**
 * ROI-Aware Result Micro-Cache implementation
 */
export class ROIAwareMicroCache {
  private config: MicroCacheConfig;
  private shards: CacheShard[] = [];
  private canonicalizer: QueryCanonicalizer;
  private intentClassifier: IntentClassifier;
  private topicBinGenerator: TopicBinGenerator;
  private stats: CacheStats = {
    totalHits: 0,
    totalMisses: 0,
    hitRate: 0,
    averageLatencySaved: 0,
    totalEntriesCount: 0,
    evictionsCount: 0,
    spanInvariantViolations: 0
  };

  constructor(config: Partial<MicroCacheConfig> = {}) {
    this.config = {
      enabled: config.enabled ?? true,
      shardCount: config.shardCount ?? 16,
      ttlSeconds: config.ttlSeconds ?? 2,
      maxEntriesPerShard: config.maxEntriesPerShard ?? 100,
      slaHeadroomThresholdMs: config.slaHeadroomThresholdMs ?? 2,
      enableKMerge: config.enableKMerge ?? true,
      kMergeRatio: config.kMergeRatio ?? 0.7,
      canonicalizationEnabled: config.canonicalizationEnabled ?? true,
      spanInvariantCheck: config.spanInvariantCheck ?? true,
      maxCachedResults: config.maxCachedResults ?? 50,
      ...config
    };

    this.canonicalizer = new QueryCanonicalizer();
    this.intentClassifier = new IntentClassifier();
    this.topicBinGenerator = new TopicBinGenerator();

    // Initialize shards
    for (let i = 0; i < this.config.shardCount; i++) {
      this.shards.push(new CacheShard(
        this.config.maxEntriesPerShard,
        this.config.ttlSeconds
      ));
    }

    console.log(`💾 ROIAwareMicroCache initialized: ${this.config.shardCount} shards, TTL=${this.config.ttlSeconds}s, enabled=${this.config.enabled}`);
  }

  /**
   * Attempt to get cached results
   */
  async getCachedResults(
    context: SearchContext,
    indexVersion: string,
    slaHeadroomMs: number
  ): Promise<SearchHit[] | null> {
    const span = LensTracer.createChildSpan('micro_cache_get', {
      'query': context.query,
      'enabled': this.config.enabled,
      'sla_headroom_ms': slaHeadroomMs
    });

    try {
      if (!this.config.enabled) {
        span.setAttributes({ skipped: true, reason: 'disabled' });
        return null;
      }

      // Check SLA headroom - only use cache when tight on budget
      if (slaHeadroomMs > this.config.slaHeadroomThresholdMs) {
        span.setAttributes({ skipped: true, reason: 'sufficient_sla_headroom' });
        return null;
      }

      // Generate cache key
      const cacheKey = this.generateCacheKey(context, indexVersion);
      const keyHash = this.hashCacheKey(cacheKey);

      // Get shard and lookup
      const shard = this.getShard(keyHash);
      const cached = shard.get(keyHash);

      if (!cached) {
        this.stats.totalMisses++;
        span.setAttributes({ cache_hit: false });
        return null;
      }

      // Validate span invariants if enabled
      if (this.config.spanInvariantCheck && !this.validateSpanInvariants(cached.topN)) {
        this.stats.spanInvariantViolations++;
        span.setAttributes({ cache_hit: false, reason: 'span_invariant_violation' });
        return null;
      }

      this.stats.totalHits++;
      this.stats.averageLatencySaved = 
        (this.stats.averageLatencySaved * (this.stats.totalHits - 1) + cached.averageLatencySaved) / this.stats.totalHits;

      span.setAttributes({ 
        cache_hit: true,
        cached_results_count: cached.topN.length,
        hit_count: cached.hitCount,
        latency_saved: cached.averageLatencySaved
      });

      console.log(`💾 Cache HIT: ${cached.topN.length} results for "${context.query}" (${cached.hitCount} hits)`);

      return [...cached.topN]; // Return copy to prevent mutations

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      return null;

    } finally {
      span.end();
      this.updateHitRate();
    }
  }

  /**
   * Cache search results
   */
  async cacheResults(
    context: SearchContext,
    results: SearchHit[],
    indexVersion: string,
    latencySaved: number
  ): Promise<void> {
    const span = LensTracer.createChildSpan('micro_cache_set', {
      'query': context.query,
      'results_count': results.length,
      'enabled': this.config.enabled
    });

    try {
      if (!this.config.enabled || results.length === 0) {
        span.setAttributes({ skipped: true, reason: 'disabled_or_empty_results' });
        return;
      }

      // Limit cached results
      const resultsToCache = results.slice(0, this.config.maxCachedResults);

      // Generate cache key and value
      const cacheKey = this.generateCacheKey(context, indexVersion);
      const keyHash = this.hashCacheKey(cacheKey);

      const cacheValue: CacheValue = {
        topN: resultsToCache,
        whyMix: this.computeWhyMix(resultsToCache),
        eceSlice: this.computeECESlice(resultsToCache),
        timestamp: Date.now(),
        hitCount: 0,
        averageLatencySaved: latencySaved
      };

      // Store in appropriate shard
      const shard = this.getShard(keyHash);
      shard.set(keyHash, cacheValue);

      span.setAttributes({
        success: true,
        cached_results_count: resultsToCache.length,
        why_mix_size: cacheValue.whyMix.size,
        ece_score: cacheValue.eceSlice.calibration_score
      });

      console.log(`💾 Cache SET: ${resultsToCache.length} results for "${context.query}"`);

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });

    } finally {
      span.end();
      this.updateTotalEntries();
    }
  }

  /**
   * Generate cache key from context
   */
  private generateCacheKey(context: SearchContext, indexVersion: string): CacheKey {
    const intent = this.intentClassifier.classifyIntent(context.query, context);
    const canonicalQuery = this.config.canonicalizationEnabled 
      ? this.canonicalizer.canonicalize(context.query)
      : context.query.toLowerCase().trim();
    const topicId = this.topicBinGenerator.generateTopicBin(context.query, context);

    return {
      intent,
      canonicalQuery,
      topicId,
      language: context.language,
      indexVersion
    };
  }

  /**
   * Hash cache key to string
   */
  private hashCacheKey(key: CacheKey): string {
    const keyStr = `${key.intent}:${key.canonicalQuery}:${key.topicId}:${key.language || ''}:${key.indexVersion}`;
    return this.hashString(keyStr);
  }

  /**
   * Simple hash function for cache keys
   */
  private hashString(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(36);
  }

  /**
   * Get cache shard for hash
   */
  private getShard(keyHash: string): CacheShard {
    const hashNum = parseInt(keyHash, 36) || 0;
    const shardIndex = hashNum % this.config.shardCount;
    return this.shards[shardIndex]!;
  }

  /**
   * Compute match reason distribution
   */
  private computeWhyMix(results: SearchHit[]): Map<string, number> {
    const whyMix = new Map<string, number>();
    
    for (const result of results) {
      for (const reason of result.why) {
        const count = whyMix.get(reason) || 0;
        whyMix.set(reason, count + 1);
      }
    }

    // Normalize to percentages
    const total = results.length;
    for (const [reason, count] of whyMix) {
      whyMix.set(reason, count / total);
    }

    return whyMix;
  }

  /**
   * Compute Expected Calibration Error slice
   */
  private computeECESlice(results: SearchHit[]): CacheValue['eceSlice'] {
    const bins = 10;
    const confidenceBins = Array(bins).fill(0);
    const accuracyPerBin = Array(bins).fill(0);

    // Simple ECE computation based on scores
    for (const result of results) {
      const binIndex = Math.min(Math.floor(result.score * bins), bins - 1);
      confidenceBins[binIndex]++;
      // For demonstration - would need actual relevance labels in practice
      accuracyPerBin[binIndex] += result.score > 0.5 ? 1 : 0;
    }

    // Normalize accuracy per bin
    for (let i = 0; i < bins; i++) {
      if (confidenceBins[i] > 0) {
        accuracyPerBin[i] /= confidenceBins[i];
      }
    }

    // Compute overall calibration score
    let calibrationScore = 0;
    const total = results.length;
    for (let i = 0; i < bins; i++) {
      const binWeight = confidenceBins[i] / total;
      const binConfidence = (i + 0.5) / bins;
      const binAccuracy = accuracyPerBin[i];
      calibrationScore += binWeight * Math.abs(binConfidence - binAccuracy);
    }

    return {
      confidence_bins: confidenceBins,
      accuracy_per_bin: accuracyPerBin,
      calibration_score: calibrationScore
    };
  }

  /**
   * Validate span invariants for cached results
   */
  private validateSpanInvariants(results: SearchHit[]): boolean {
    for (const result of results) {
      // Check required fields
      if (!result.file || result.line < 1 || result.col < 0) {
        return false;
      }

      // Check score bounds
      if (result.score < 0 || result.score > 1) {
        return false;
      }

      // Check why array is non-empty
      if (!result.why || result.why.length === 0) {
        return false;
      }

      // Check span length consistency if present
      if (result.span_len !== undefined && result.span_len < 0) {
        return false;
      }
    }

    return true;
  }

  /**
   * Update hit rate statistics
   */
  private updateHitRate(): void {
    const total = this.stats.totalHits + this.stats.totalMisses;
    this.stats.hitRate = total > 0 ? this.stats.totalHits / total : 0;
  }

  /**
   * Update total entries count across all shards
   */
  private updateTotalEntries(): void {
    this.stats.totalEntriesCount = this.shards.reduce((sum, shard) => 
      sum + shard.getStats().entries, 0);
  }

  /**
   * Get comprehensive cache statistics
   */
  getStats(): CacheStats & { shardStats: Array<{ entries: number; oldestEntry: number; newestEntry: number }> } {
    this.updateTotalEntries();
    
    const shardStats = this.shards.map(shard => shard.getStats());

    return {
      ...this.stats,
      shardStats
    };
  }

  /**
   * Clear all cache entries
   */
  clearCache(): void {
    for (const shard of this.shards) {
      shard.clear();
    }
    
    this.stats = {
      totalHits: 0,
      totalMisses: 0,
      hitRate: 0,
      averageLatencySaved: 0,
      totalEntriesCount: 0,
      evictionsCount: 0,
      spanInvariantViolations: 0
    };

    console.log('💾 MicroCache cleared');
  }

  /**
   * Update configuration for A/B testing
   */
  updateConfig(newConfig: Partial<MicroCacheConfig>): void {
    this.config = { ...this.config, ...newConfig };
    console.log(`💾 ROIAwareMicroCache config updated: ${JSON.stringify(newConfig)}`);
  }
}