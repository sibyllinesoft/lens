/**
 * LSP SPI Cache Manager
 * Implements (repo_sha, path, source_hash) keyed caching with TTL 10-60s
 */

import { LRUCache } from 'lru-cache';
import { createHash } from 'crypto';
import { LensTracer } from '../telemetry/tracer.js';

export interface LSPCacheEntry<T = any> {
  data: T;
  timestamp: number;
  duration_ms: number;
  hits: number;
}

export interface LSPCacheKey {
  repo_sha: string;
  path: string;
  source_hash: string;
  operation: string; // diagnostics, format, etc.
}

export class LSPCacheManager {
  private cache: LRUCache<string, LSPCacheEntry>;
  private hitCounter = 0;
  private missCounter = 0;
  private readonly defaultTTL: number;

  constructor(options: {
    maxSize?: number;
    ttlMs?: number;
  } = {}) {
    this.defaultTTL = options.ttlMs || 30000; // 30 seconds default

    this.cache = new LRUCache({
      max: options.maxSize || 10000,
      ttl: this.defaultTTL,
      updateAgeOnGet: false,
      updateAgeOnHas: false,
    });
  }

  /**
   * Generate cache key from components
   */
  private generateKey(key: LSPCacheKey): string {
    // Create deterministic key
    const keyStr = `${key.repo_sha}:${key.path}:${key.source_hash}:${key.operation}`;
    return createHash('sha256').update(keyStr).digest('hex');
  }

  /**
   * Get cached entry if valid
   */
  async get<T = any>(key: LSPCacheKey): Promise<T | null> {
    const span = LensTracer.createChildSpan('lsp_cache_get');
    
    try {
      const cacheKey = this.generateKey(key);
      const entry = this.cache.get(cacheKey);
      
      if (!entry) {
        this.missCounter++;
        span.setAttributes({ 
          cache_hit: false,
          operation: key.operation,
          repo_sha: key.repo_sha.slice(0, 8),
          path: key.path,
        });
        return null;
      }

      // Check if entry is still valid (additional TTL check)
      const age = Date.now() - entry.timestamp;
      const ttl = this.getTTLForOperation(key.operation);
      
      if (age > ttl) {
        this.cache.delete(cacheKey);
        this.missCounter++;
        span.setAttributes({ 
          cache_hit: false,
          cache_expired: true,
          age_ms: age,
          ttl_ms: ttl,
          operation: key.operation,
        });
        return null;
      }

      // Update hit counter
      entry.hits++;
      this.hitCounter++;
      
      span.setAttributes({
        cache_hit: true,
        age_ms: age,
        ttl_ms: ttl,
        hit_count: entry.hits,
        operation: key.operation,
        duration_ms: entry.duration_ms,
      });
      
      return entry.data;
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error);
      span.setAttributes({ success: false, error: errorMsg });
      this.missCounter++;
      return null;
    } finally {
      span.end();
    }
  }

  /**
   * Store entry in cache
   */
  async set<T = any>(key: LSPCacheKey, data: T, duration_ms: number): Promise<void> {
    const span = LensTracer.createChildSpan('lsp_cache_set');
    
    try {
      const cacheKey = this.generateKey(key);
      const entry: LSPCacheEntry<T> = {
        data,
        timestamp: Date.now(),
        duration_ms,
        hits: 0
      };

      const ttl = this.getTTLForOperation(key.operation);
      this.cache.set(cacheKey, entry, { ttl });
      
      span.setAttributes({
        success: true,
        operation: key.operation,
        repo_sha: key.repo_sha.slice(0, 8),
        path: key.path,
        ttl_ms: ttl,
        duration_ms,
        data_size: JSON.stringify(data).length,
      });
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error);
      span.setAttributes({ success: false, error: errorMsg });
    } finally {
      span.end();
    }
  }

  /**
   * Invalidate cache entries by pattern
   */
  async invalidate(pattern: Partial<LSPCacheKey>): Promise<number> {
    const span = LensTracer.createChildSpan('lsp_cache_invalidate');
    
    try {
      let invalidatedCount = 0;
      
      // Get all keys and filter by pattern
      for (const [cacheKey, entry] of this.cache.entries()) {
        // For a more sophisticated implementation, we'd decode the key
        // For now, use simple string matching on key components
        let matches = true;
        
        if (pattern.repo_sha && !cacheKey.includes(pattern.repo_sha)) {
          matches = false;
        }
        
        if (pattern.path && !cacheKey.includes(pattern.path)) {
          matches = false;
        }
        
        if (pattern.operation && !cacheKey.includes(pattern.operation)) {
          matches = false;
        }
        
        if (matches) {
          this.cache.delete(cacheKey);
          invalidatedCount++;
        }
      }
      
      span.setAttributes({
        success: true,
        invalidated_count: invalidatedCount,
        pattern: JSON.stringify(pattern),
      });
      
      return invalidatedCount;
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error);
      span.setAttributes({ success: false, error: errorMsg });
      return 0;
    } finally {
      span.end();
    }
  }

  /**
   * Get cache statistics
   */
  getStats(): {
    size: number;
    hits: number;
    misses: number;
    hit_ratio: number;
    max_size: number;
  } {
    const size = this.cache.size;
    const hits = this.hitCounter;
    const misses = this.missCounter;
    const total = hits + misses;
    
    return {
      size,
      hits,
      misses,
      hit_ratio: total > 0 ? hits / total : 0,
      max_size: this.cache.max,
    };
  }

  /**
   * Clear all cache entries
   */
  clear(): void {
    this.cache.clear();
    this.hitCounter = 0;
    this.missCounter = 0;
  }

  /**
   * Get TTL for specific operation type
   */
  private getTTLForOperation(operation: string): number {
    // Different operations have different cache lifetimes
    const ttlMap: Record<string, number> = {
      // Fast operations can be cached longer
      'diagnostics': 30000,   // 30s
      'foldingRanges': 60000, // 60s
      'selectionRanges': 45000, // 45s
      
      // Formatting should be cached briefly (files change)
      'format': 10000,        // 10s
      
      // Code actions depend on diagnostics, shorter cache
      'codeActions': 15000,   // 15s
      
      // Rename operations are expensive, cache longer
      'prepareRename': 45000, // 45s
      'rename': 30000,        // 30s
      
      // Hierarchy operations are expensive
      'hierarchy': 60000,     // 60s
    };
    
    return ttlMap[operation] || this.defaultTTL;
  }

  /**
   * Compute source hash for caching
   */
  static computeSourceHash(content: string): string {
    return createHash('sha256').update(content).digest('hex').slice(0, 16);
  }

  /**
   * Generate lens:// ref from components
   */
  static generateLensRef(repo_sha: string, path: string, source_hash: string, start?: number, end?: number): string {
    let ref = `lens://${repo_sha}/${path}@${source_hash}`;
    if (start !== undefined && end !== undefined) {
      ref += `#B${start}:${end}`;
    }
    return ref;
  }

  /**
   * Parse lens:// ref into components
   */
  static parseLensRef(ref: string): {
    repo_sha: string;
    path: string;
    source_hash: string;
    start?: number;
    end?: number;
  } | null {
    const match = ref.match(/^lens:\/\/([^\/]+)\/([^@]+)@([^#]+)(?:#B(\d+):(\d+))?$/);
    if (!match) return null;
    
    const [, repo_sha, path, source_hash, startStr, endStr] = match;
    
    return {
      repo_sha,
      path,
      source_hash,
      start: startStr ? parseInt(startStr, 10) : undefined,
      end: endStr ? parseInt(endStr, 10) : undefined,
    };
  }
}

// Global cache manager instance
export const globalLSPCache = new LSPCacheManager({
  maxSize: 10000,
  ttlMs: 30000,
});