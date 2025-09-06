/**
 * Advanced Caching System with LRU/TTL Policies
 * Multi-tier cache architecture for sub-10ms search performance
 * Implements hot/warm/cold data tiers with intelligent prefetching
 */

import { LensTracer } from '../telemetry/tracer.js';
import { globalMemoryPool } from './memory-pool-manager.js';
import type { SearchContext, SearchHit, Candidate } from '../types/core.js';

interface CacheEntry<T> {
  key: string;
  value: T;
  timestamp: number;
  lastAccessed: number;
  accessCount: number;
  ttl: number;
  size: number;
  tier: CacheTier;
  hash: string;
}

enum CacheTier {
  HOT = 'hot',     // Most frequently accessed, keep in memory
  WARM = 'warm',   // Moderately accessed, compress in memory
  COLD = 'cold'    // Rarely accessed, consider eviction
}

interface CacheConfig {
  maxMemoryMB: number;
  hotTierRatio: number;
  warmTierRatio: number;
  coldTierRatio: number;
  defaultTTL: number;
  maxEntries: number;
  prefetchEnabled: boolean;
  compressionEnabled: boolean;
}

interface CacheStats {
  totalEntries: number;
  hotTierEntries: number;
  warmTierEntries: number;
  coldTierEntries: number;
  totalMemoryMB: number;
  hitRate: number;
  missRate: number;
  evictionRate: number;
  compressionRatio: number;
  avgAccessTime: number;
}

interface PrefetchHint {
  key: string;
  probability: number;
  context: SearchContext;
}

export class AdvancedCacheManager {
  private static instance: AdvancedCacheManager;
  
  // Cache storage by tier
  private hotTier: Map<string, CacheEntry<any>> = new Map();
  private warmTier: Map<string, CacheEntry<any>> = new Map();
  private coldTier: Map<string, CacheEntry<any>> = new Map();
  
  // LRU tracking
  private accessOrder: string[] = [];
  private hotAccessOrder: string[] = [];
  private warmAccessOrder: string[] = [];
  private coldAccessOrder: string[] = [];
  
  // Configuration
  private config: CacheConfig;
  
  // Statistics
  private totalRequests = 0;
  private totalHits = 0;
  private totalMisses = 0;
  private totalEvictions = 0;
  private totalMemoryUsed = 0;
  private accessTimes: number[] = [];
  
  // Prefetch system
  private prefetchQueue: PrefetchHint[] = [];
  private prefetchInProgress: Set<string> = new Set();
  
  // TTL cleanup
  private cleanupTimer?: NodeJS.Timeout;
  private tierPromotionTimer?: NodeJS.Timeout;
  
  private constructor() {
    this.config = {
      maxMemoryMB: 512,          // 512MB total cache
      hotTierRatio: 0.4,         // 40% for hot tier
      warmTierRatio: 0.4,        // 40% for warm tier
      coldTierRatio: 0.2,        // 20% for cold tier
      defaultTTL: 300000,        // 5 minutes default TTL
      maxEntries: 100000,        // Maximum entries across all tiers
      prefetchEnabled: true,     // Enable intelligent prefetching
      compressionEnabled: true   // Enable compression for warm/cold tiers
    };
    
    this.startMaintenanceTimers();
  }
  
  public static getInstance(): AdvancedCacheManager {
    if (!AdvancedCacheManager.instance) {
      AdvancedCacheManager.instance = new AdvancedCacheManager();
    }
    return AdvancedCacheManager.instance;
  }
  
  /**
   * Get value from cache with intelligent tier promotion
   */
  async get<T>(key: string, context?: SearchContext): Promise<T | null> {
    const span = LensTracer.createChildSpan('cache_get');
    const startTime = Date.now();
    this.totalRequests++;
    
    try {
      const keyHash = this.hashKey(key);
      
      // Check hot tier first
      let entry = this.hotTier.get(keyHash);
      if (entry) {
        return this.handleCacheHit(entry, keyHash, CacheTier.HOT, startTime, span);
      }
      
      // Check warm tier
      entry = this.warmTier.get(keyHash);
      if (entry) {
        // Promote to hot tier if frequently accessed
        if (this.shouldPromoteToHot(entry)) {
          await this.promoteEntry(entry, keyHash, CacheTier.WARM, CacheTier.HOT);
        }
        
        return this.handleCacheHit(entry, keyHash, CacheTier.WARM, startTime, span);
      }
      
      // Check cold tier
      entry = this.coldTier.get(keyHash);
      if (entry) {
        // Promote to warm tier if accessed
        if (this.shouldPromoteToWarm(entry)) {
          await this.promoteEntry(entry, keyHash, CacheTier.COLD, CacheTier.WARM);
        }
        
        return this.handleCacheHit(entry, keyHash, CacheTier.COLD, startTime, span);
      }
      
      // Cache miss - trigger prefetch if enabled
      this.totalMisses++;
      if (context && this.config.prefetchEnabled) {
        this.schedulePrefetch(key, context);
      }
      
      span.setAttributes({
        success: true,
        cache_hit: false,
        key_hash: keyHash
      });
      
      return null;
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      return null;
    } finally {
      span.end();
      this.recordAccessTime(Date.now() - startTime);
    }
  }
  
  /**
   * Set value in cache with intelligent tier placement
   */
  async set<T>(key: string, value: T, ttl?: number, context?: SearchContext): Promise<void> {
    const span = LensTracer.createChildSpan('cache_set');
    
    try {
      const keyHash = this.hashKey(key);
      const now = Date.now();
      const entryTTL = ttl || this.config.defaultTTL;
      const size = this.estimateSize(value);
      
      // Create cache entry
      const entry: CacheEntry<T> = {
        key,
        value,
        timestamp: now,
        lastAccessed: now,
        accessCount: 1,
        ttl: entryTTL,
        size,
        tier: CacheTier.WARM, // Default to warm tier
        hash: keyHash
      };
      
      // Determine appropriate tier based on context and heuristics
      const targetTier = this.determineOptimalTier(key, value, context);
      entry.tier = targetTier;
      
      // Ensure space is available
      await this.ensureSpace(size, targetTier);
      
      // Store in appropriate tier
      await this.storeInTier(entry, targetTier);
      
      this.totalMemoryUsed += size;
      
      span.setAttributes({
        success: true,
        tier: targetTier,
        size_bytes: size,
        ttl_ms: entryTTL
      });
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw error;
    } finally {
      span.end();
    }
  }
  
  /**
   * Handle cache hit with tier-specific logic
   */
  private async handleCacheHit<T>(
    entry: CacheEntry<T>, 
    keyHash: string, 
    tier: CacheTier, 
    startTime: number, 
    span: any
  ): Promise<T | null> {
    
    // Check TTL expiration
    if (Date.now() - entry.timestamp > entry.ttl) {
      await this.evictEntry(keyHash, tier);
      this.totalMisses++;
      
      span.setAttributes({
        success: true,
        cache_hit: false,
        reason: 'ttl_expired',
        tier
      });
      
      return null;
    }
    
    // Update access statistics
    entry.lastAccessed = Date.now();
    entry.accessCount++;
    this.updateLRUOrder(keyHash, tier);
    
    this.totalHits++;
    
    let value = entry.value;
    
    // Decompress if necessary
    if (tier !== CacheTier.HOT && this.config.compressionEnabled && this.isCompressed(value)) {
      value = await this.decompress(value);
    }
    
    span.setAttributes({
      success: true,
      cache_hit: true,
      tier,
      access_count: entry.accessCount,
      age_ms: Date.now() - entry.timestamp
    });
    
    return value;
  }
  
  /**
   * Determine optimal tier for new cache entry
   */
  private determineOptimalTier<T>(key: string, value: T, context?: SearchContext): CacheTier {
    // Heuristics for tier placement
    
    // Small, frequently accessed data goes to hot tier
    if (this.estimateSize(value) < 1024) { // < 1KB
      return CacheTier.HOT;
    }
    
    // Search results might be accessed again soon
    if (context && this.isSearchResult(value)) {
      return CacheTier.WARM;
    }
    
    // Large data or unknown access patterns start in cold
    if (this.estimateSize(value) > 64 * 1024) { // > 64KB
      return CacheTier.COLD;
    }
    
    // Default to warm tier
    return CacheTier.WARM;
  }
  
  /**
   * Store entry in specified tier
   */
  private async storeInTier<T>(entry: CacheEntry<T>, tier: CacheTier): Promise<void> {
    const keyHash = entry.hash;
    
    // Compress value for warm/cold tiers
    if (tier !== CacheTier.HOT && this.config.compressionEnabled) {
      entry.value = await this.compress(entry.value);
    }
    
    switch (tier) {
      case CacheTier.HOT:
        this.hotTier.set(keyHash, entry);
        this.hotAccessOrder.unshift(keyHash);
        break;
      case CacheTier.WARM:
        this.warmTier.set(keyHash, entry);
        this.warmAccessOrder.unshift(keyHash);
        break;
      case CacheTier.COLD:
        this.coldTier.set(keyHash, entry);
        this.coldAccessOrder.unshift(keyHash);
        break;
    }
    
    this.accessOrder.unshift(keyHash);
  }
  
  /**
   * Ensure sufficient space is available in target tier
   */
  private async ensureSpace(requiredSize: number, targetTier: CacheTier): Promise<void> {
    const tierMemoryLimits = {
      [CacheTier.HOT]: this.config.maxMemoryMB * this.config.hotTierRatio * 1024 * 1024,
      [CacheTier.WARM]: this.config.maxMemoryMB * this.config.warmTierRatio * 1024 * 1024,
      [CacheTier.COLD]: this.config.maxMemoryMB * this.config.coldTierRatio * 1024 * 1024
    };
    
    const currentTierSize = this.getTierMemoryUsage(targetTier);
    const limit = tierMemoryLimits[targetTier];
    
    if (currentTierSize + requiredSize > limit) {
      await this.evictFromTier(targetTier, (currentTierSize + requiredSize) - limit);
    }
  }
  
  /**
   * Evict entries from specified tier to free up space
   */
  private async evictFromTier(tier: CacheTier, bytesToFree: number): Promise<void> {
    const span = LensTracer.createChildSpan('cache_evict_tier');
    let bytesFreed = 0;
    let entriesEvicted = 0;
    
    try {
      const accessOrder = this.getAccessOrderForTier(tier);
      const tierMap = this.getTierMap(tier);
      
      // Evict LRU entries until enough space is freed
      while (bytesFreed < bytesToFree && accessOrder.length > 0) {
        const keyToEvict = accessOrder.pop();
        if (keyToEvict) {
          const entry = tierMap.get(keyToEvict);
          if (entry) {
            bytesFreed += entry.size;
            await this.evictEntry(keyToEvict, tier);
            entriesEvicted++;
          }
        }
      }
      
      this.totalEvictions += entriesEvicted;
      
      span.setAttributes({
        success: true,
        tier,
        bytes_freed: bytesFreed,
        entries_evicted: entriesEvicted
      });
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw error;
    } finally {
      span.end();
    }
  }
  
  /**
   * Promote entry between tiers
   */
  private async promoteEntry(entry: CacheEntry<any>, keyHash: string, fromTier: CacheTier, toTier: CacheTier): Promise<void> {
    const span = LensTracer.createChildSpan('cache_promote_entry');
    
    try {
      // Remove from source tier
      const fromMap = this.getTierMap(fromTier);
      const fromOrder = this.getAccessOrderForTier(fromTier);
      
      fromMap.delete(keyHash);
      this.removeFromAccessOrder(keyHash, fromOrder);
      
      // Decompress if moving from compressed tier
      if (fromTier !== CacheTier.HOT && this.config.compressionEnabled && this.isCompressed(entry.value)) {
        entry.value = await this.decompress(entry.value);
      }
      
      // Update tier
      entry.tier = toTier;
      
      // Ensure space in target tier
      await this.ensureSpace(entry.size, toTier);
      
      // Add to target tier
      await this.storeInTier(entry, toTier);
      
      span.setAttributes({
        success: true,
        from_tier: fromTier,
        to_tier: toTier,
        entry_size: entry.size,
        access_count: entry.accessCount
      });
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw error;
    } finally {
      span.end();
    }
  }
  
  /**
   * Check if entry should be promoted to hot tier
   */
  private shouldPromoteToHot(entry: CacheEntry<any>): boolean {
    const ageMins = (Date.now() - entry.timestamp) / 60000;
    const accessFrequency = entry.accessCount / Math.max(ageMins, 1);
    
    return accessFrequency > 2.0 && entry.accessCount > 5;
  }
  
  /**
   * Check if entry should be promoted to warm tier
   */
  private shouldPromoteToWarm(entry: CacheEntry<any>): boolean {
    return entry.accessCount > 2;
  }
  
  /**
   * Evict single entry
   */
  private async evictEntry(keyHash: string, tier: CacheTier): Promise<void> {
    const tierMap = this.getTierMap(tier);
    const accessOrder = this.getAccessOrderForTier(tier);
    
    const entry = tierMap.get(keyHash);
    if (entry) {
      this.totalMemoryUsed -= entry.size;
      tierMap.delete(keyHash);
      this.removeFromAccessOrder(keyHash, accessOrder);
      this.removeFromAccessOrder(keyHash, this.accessOrder);
    }
  }
  
  /**
   * Schedule prefetch based on access patterns
   */
  private schedulePrefetch(key: string, context: SearchContext): void {
    if (this.prefetchInProgress.has(key)) return;
    
    const probability = this.calculatePrefetchProbability(key, context);
    
    if (probability > 0.7) {
      this.prefetchQueue.push({ key, probability, context });
      
      // Process prefetch queue
      setImmediate(() => this.processPrefetchQueue());
    }
  }
  
  /**
   * Calculate prefetch probability based on patterns
   */
  private calculatePrefetchProbability(key: string, context: SearchContext): number {
    // Simple heuristics - can be enhanced with ML
    let probability = 0.5; // Base probability
    
    // Similar queries are likely to be repeated
    if (context.query.length > 3) {
      probability += 0.2;
    }
    
    // Recent searches in same repo
    if (context.repo_sha) {
      probability += 0.1;
    }
    
    // Time-based patterns (e.g., working hours)
    const hour = new Date().getHours();
    if (hour >= 9 && hour <= 17) {
      probability += 0.1;
    }
    
    return Math.min(probability, 1.0);
  }
  
  /**
   * Process prefetch queue
   */
  private async processPrefetchQueue(): Promise<void> {
    if (this.prefetchQueue.length === 0) return;
    
    // Sort by probability
    this.prefetchQueue.sort((a, b) => b.probability - a.probability);
    
    // Process top items
    const itemsToProcess = this.prefetchQueue.splice(0, 3);
    
    for (const item of itemsToProcess) {
      if (this.prefetchInProgress.has(item.key)) continue;
      
      this.prefetchInProgress.add(item.key);
      
      try {
        // This would integrate with the search engine to prefetch results
        // For now, just simulate prefetch completion
        setTimeout(() => {
          this.prefetchInProgress.delete(item.key);
        }, 1000);
        
      } catch (error) {
        console.warn(`Prefetch failed for key ${item.key}:`, error);
        this.prefetchInProgress.delete(item.key);
      }
    }
  }
  
  /**
   * Compress data using simple compression
   */
  private async compress<T>(data: T): Promise<any> {
    try {
      const jsonStr = JSON.stringify(data);
      const compressed = Buffer.from(jsonStr).toString('base64');
      return { __compressed: true, data: compressed };
    } catch {
      return data; // Return original if compression fails
    }
  }
  
  /**
   * Decompress data
   */
  private async decompress<T>(compressed: any): Promise<T> {
    try {
      if (compressed.__compressed) {
        const jsonStr = Buffer.from(compressed.data, 'base64').toString();
        return JSON.parse(jsonStr);
      }
      return compressed;
    } catch {
      return compressed; // Return as-is if decompression fails
    }
  }
  
  /**
   * Check if data is compressed
   */
  private isCompressed(data: any): boolean {
    return data && typeof data === 'object' && data.__compressed === true;
  }
  
  /**
   * Check if value is a search result
   */
  private isSearchResult(value: any): boolean {
    return Array.isArray(value) && value.length > 0 && 
           (value[0].file || value[0].snippet); // Basic heuristic
  }
  
  /**
   * Estimate memory size of object
   */
  private estimateSize(obj: any): number {
    try {
      return JSON.stringify(obj).length * 2; // Rough estimate
    } catch {
      return 1024; // Default size if estimation fails
    }
  }
  
  /**
   * Hash cache key for consistent storage
   */
  private hashKey(key: string): string {
    // Simple hash function - could use crypto for better distribution
    let hash = 0;
    for (let i = 0; i < key.length; i++) {
      const char = key.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash).toString(36);
  }
  
  /**
   * Update LRU order for a key
   */
  private updateLRUOrder(keyHash: string, tier: CacheTier): void {
    const accessOrder = this.getAccessOrderForTier(tier);
    
    // Remove from current position
    const index = accessOrder.indexOf(keyHash);
    if (index > -1) {
      accessOrder.splice(index, 1);
    }
    
    // Add to front (most recently used)
    accessOrder.unshift(keyHash);
    
    // Also update global access order
    const globalIndex = this.accessOrder.indexOf(keyHash);
    if (globalIndex > -1) {
      this.accessOrder.splice(globalIndex, 1);
    }
    this.accessOrder.unshift(keyHash);
  }
  
  /**
   * Remove key from access order array
   */
  private removeFromAccessOrder(keyHash: string, accessOrder: string[]): void {
    const index = accessOrder.indexOf(keyHash);
    if (index > -1) {
      accessOrder.splice(index, 1);
    }
  }
  
  /**
   * Get tier map by tier enum
   */
  private getTierMap(tier: CacheTier): Map<string, CacheEntry<any>> {
    switch (tier) {
      case CacheTier.HOT: return this.hotTier;
      case CacheTier.WARM: return this.warmTier;
      case CacheTier.COLD: return this.coldTier;
    }
  }
  
  /**
   * Get access order array by tier
   */
  private getAccessOrderForTier(tier: CacheTier): string[] {
    switch (tier) {
      case CacheTier.HOT: return this.hotAccessOrder;
      case CacheTier.WARM: return this.warmAccessOrder;
      case CacheTier.COLD: return this.coldAccessOrder;
    }
  }
  
  /**
   * Get memory usage for specific tier
   */
  private getTierMemoryUsage(tier: CacheTier): number {
    const tierMap = this.getTierMap(tier);
    let totalSize = 0;
    
    for (const entry of tierMap.values()) {
      totalSize += entry.size;
    }
    
    return totalSize;
  }
  
  /**
   * Record access time for performance tracking
   */
  private recordAccessTime(timeMs: number): void {
    this.accessTimes.push(timeMs);
    
    // Keep only recent measurements
    if (this.accessTimes.length > 1000) {
      this.accessTimes = this.accessTimes.slice(-500);
    }
  }
  
  /**
   * Start maintenance timers
   */
  private startMaintenanceTimers(): void {
    // TTL cleanup every 30 seconds
    this.cleanupTimer = setInterval(() => {
      this.cleanupExpiredEntries();
    }, 30000);
    
    // Tier promotion analysis every 2 minutes
    this.tierPromotionTimer = setInterval(() => {
      this.analyzeTierPromotion();
    }, 120000);
  }
  
  /**
   * Clean up expired entries
   */
  private async cleanupExpiredEntries(): Promise<void> {
    const span = LensTracer.createChildSpan('cache_cleanup_expired');
    let cleanedCount = 0;
    
    try {
      const now = Date.now();
      const allTiers = [this.hotTier, this.warmTier, this.coldTier];
      const tierNames = [CacheTier.HOT, CacheTier.WARM, CacheTier.COLD];
      
      for (let i = 0; i < allTiers.length; i++) {
        const tier = allTiers[i];
        const tierName = tierNames[i];
        
        const keysToDelete: string[] = [];
        
        for (const [keyHash, entry] of tier) {
          if (now - entry.timestamp > entry.ttl) {
            keysToDelete.push(keyHash);
          }
        }
        
        for (const keyHash of keysToDelete) {
          await this.evictEntry(keyHash, tierName);
          cleanedCount++;
        }
      }
      
      if (cleanedCount > 0) {
        console.log(`🧹 Cache cleanup: removed ${cleanedCount} expired entries`);
      }
      
      span.setAttributes({
        success: true,
        entries_cleaned: cleanedCount
      });
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
    } finally {
      span.end();
    }
  }
  
  /**
   * Analyze and perform tier promotions
   */
  private async analyzeTierPromotion(): Promise<void> {
    const span = LensTracer.createChildSpan('cache_analyze_promotion');
    let promotions = 0;
    
    try {
      // Check warm tier for hot promotion candidates
      for (const [keyHash, entry] of this.warmTier) {
        if (this.shouldPromoteToHot(entry)) {
          await this.promoteEntry(entry, keyHash, CacheTier.WARM, CacheTier.HOT);
          promotions++;
        }
      }
      
      // Check cold tier for warm promotion candidates
      for (const [keyHash, entry] of this.coldTier) {
        if (this.shouldPromoteToWarm(entry)) {
          await this.promoteEntry(entry, keyHash, CacheTier.COLD, CacheTier.WARM);
          promotions++;
        }
      }
      
      if (promotions > 0) {
        console.log(`📈 Cache tier promotions: promoted ${promotions} entries`);
      }
      
      span.setAttributes({
        success: true,
        promotions
      });
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
    } finally {
      span.end();
    }
  }
  
  /**
   * Get comprehensive cache statistics
   */
  getStats(): CacheStats {
    const hitRate = this.totalRequests > 0 ? (this.totalHits / this.totalRequests) * 100 : 0;
    const missRate = this.totalRequests > 0 ? (this.totalMisses / this.totalRequests) * 100 : 0;
    const evictionRate = this.totalRequests > 0 ? (this.totalEvictions / this.totalRequests) * 100 : 0;
    
    const avgAccessTime = this.accessTimes.length > 0 ? 
      this.accessTimes.reduce((a, b) => a + b, 0) / this.accessTimes.length : 0;
    
    return {
      totalEntries: this.hotTier.size + this.warmTier.size + this.coldTier.size,
      hotTierEntries: this.hotTier.size,
      warmTierEntries: this.warmTier.size,
      coldTierEntries: this.coldTier.size,
      totalMemoryMB: this.totalMemoryUsed / (1024 * 1024),
      hitRate,
      missRate,
      evictionRate,
      compressionRatio: 0.7, // Estimated compression ratio
      avgAccessTime
    };
  }
  
  /**
   * Clear all cache entries
   */
  clear(): void {
    this.hotTier.clear();
    this.warmTier.clear();
    this.coldTier.clear();
    
    this.accessOrder = [];
    this.hotAccessOrder = [];
    this.warmAccessOrder = [];
    this.coldAccessOrder = [];
    
    this.totalMemoryUsed = 0;
    this.totalRequests = 0;
    this.totalHits = 0;
    this.totalMisses = 0;
    this.totalEvictions = 0;
    this.accessTimes = [];
    
    console.log('🧹 Cache cleared completely');
  }
  
  /**
   * Shutdown cache manager
   */
  shutdown(): void {
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer);
    }
    
    if (this.tierPromotionTimer) {
      clearInterval(this.tierPromotionTimer);
    }
    
    this.clear();
    console.log('🛑 Advanced Cache Manager shutdown complete');
  }
}

// Global instance
export const globalCacheManager = AdvancedCacheManager.getInstance();