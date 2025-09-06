/**
 * Memory Pool Manager for Reduced GC Pressure
 * Optimizes memory allocation patterns to achieve sub-10ms search performance
 * Based on production requirements for minimal garbage collection overhead
 */

import { LensTracer } from '../telemetry/tracer.js';
import { PRODUCTION_CONFIG } from '../types/config.js';

interface PooledBuffer {
  buffer: Buffer;
  size: number;
  inUse: boolean;
  lastUsed: number;
  hits: number;
}

interface PooledArray<T> {
  array: T[];
  capacity: number;
  inUse: boolean;
  lastUsed: number;
  hits: number;
}

interface PooledObject {
  object: any;
  type: string;
  inUse: boolean;
  lastUsed: number;
  hits: number;
}

interface MemoryPoolStats {
  totalBuffers: number;
  activeBuffers: number;
  totalArrays: number;
  activeArrays: number;
  totalObjects: number;
  activeObjects: number;
  totalMemoryMB: number;
  hitRate: number;
  gcPressureReduction: number;
}

export class MemoryPoolManager {
  private static instance: MemoryPoolManager;
  
  // Buffer pools organized by size tiers
  private bufferPools: Map<number, PooledBuffer[]> = new Map();
  private arrayPools: Map<string, PooledArray<any>[]> = new Map();
  private objectPools: Map<string, PooledObject[]> = new Map();
  
  // Memory tracking
  private totalAllocatedBytes = 0;
  private gcEventsBeforePool = 0;
  private gcEventsAfterPool = 0;
  private totalRequests = 0;
  private totalHits = 0;
  
  // Pool configuration
  private readonly BUFFER_SIZE_TIERS = [
    1024,      // 1KB - small snippets, tokens
    4096,      // 4KB - document chunks
    16384,     // 16KB - file content
    65536,     // 64KB - large files
    262144,    // 256KB - very large files
    1048576    // 1MB - maximum single allocation
  ];
  
  private readonly MAX_POOLS_PER_TIER = 50;
  private readonly ARRAY_CAPACITY_TIERS = [16, 64, 256, 1024, 4096];
  private readonly POOL_CLEANUP_INTERVAL = 30000; // 30 seconds
  private readonly MAX_IDLE_TIME = 60000; // 1 minute
  
  private cleanupTimer?: NodeJS.Timeout;
  
  private constructor() {
    this.initializePools();
    this.startCleanupTimer();
    this.monitorGC();
  }
  
  public static getInstance(): MemoryPoolManager {
    if (!MemoryPoolManager.instance) {
      MemoryPoolManager.instance = new MemoryPoolManager();
    }
    return MemoryPoolManager.instance;
  }
  
  /**
   * Initialize memory pools with pre-allocated buffers and arrays
   */
  private initializePools(): void {
    const span = LensTracer.createChildSpan('memory_pool_init');
    
    try {
      // Initialize buffer pools
      for (const size of this.BUFFER_SIZE_TIERS) {
        const pool: PooledBuffer[] = [];
        
        // Pre-allocate some buffers for immediate use
        const preAllocCount = Math.min(10, this.MAX_POOLS_PER_TIER);
        for (let i = 0; i < preAllocCount; i++) {
          pool.push({
            buffer: Buffer.alloc(size),
            size,
            inUse: false,
            lastUsed: Date.now(),
            hits: 0
          });
          this.totalAllocatedBytes += size;
        }
        
        this.bufferPools.set(size, pool);
      }
      
      // Initialize array pools for common types
      const arrayTypes = ['string', 'number', 'SearchHit', 'Candidate'];
      for (const type of arrayTypes) {
        for (const capacity of this.ARRAY_CAPACITY_TIERS) {
          const poolKey = `${type}-${capacity}`;
          const pool: PooledArray<any>[] = [];
          
          // Pre-allocate arrays
          for (let i = 0; i < 5; i++) {
            pool.push({
              array: new Array(capacity),
              capacity,
              inUse: false,
              lastUsed: Date.now(),
              hits: 0
            });
          }
          
          this.arrayPools.set(poolKey, pool);
        }
      }
      
      console.log(`🧠 Memory Pool Manager initialized with ${this.bufferPools.size} buffer tiers and ${this.arrayPools.size} array pools`);
      
      span.setAttributes({
        success: true,
        buffer_tiers: this.bufferPools.size,
        array_pools: this.arrayPools.size,
        initial_memory_mb: this.totalAllocatedBytes / (1024 * 1024)
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
   * Get a pooled buffer of specified size (or closest larger tier)
   */
  getPooledBuffer(requestedSize: number): Buffer {
    const span = LensTracer.createChildSpan('get_pooled_buffer');
    this.totalRequests++;
    
    try {
      // Find the appropriate size tier
      const tierSize = this.BUFFER_SIZE_TIERS.find(size => size >= requestedSize) || 
                      this.BUFFER_SIZE_TIERS[this.BUFFER_SIZE_TIERS.length - 1];
      
      const pool = this.bufferPools.get(tierSize);
      if (!pool) {
        // Fallback to regular allocation
        return Buffer.alloc(requestedSize);
      }
      
      // Find an available buffer
      const availableBuffer = pool.find(item => !item.inUse);
      
      if (availableBuffer) {
        // Reuse existing buffer
        availableBuffer.inUse = true;
        availableBuffer.lastUsed = Date.now();
        availableBuffer.hits++;
        this.totalHits++;
        
        // Clear the buffer for reuse
        availableBuffer.buffer.fill(0);
        
        span.setAttributes({
          success: true,
          requested_size: requestedSize,
          tier_size: tierSize,
          pool_hit: true,
          reused_buffer: true
        });
        
        return availableBuffer.buffer.subarray(0, requestedSize);
      }
      
      // Create new buffer if pool not full
      if (pool.length < this.MAX_POOLS_PER_TIER) {
        const newBuffer = Buffer.alloc(tierSize);
        const pooledBuffer: PooledBuffer = {
          buffer: newBuffer,
          size: tierSize,
          inUse: true,
          lastUsed: Date.now(),
          hits: 1
        };
        
        pool.push(pooledBuffer);
        this.totalAllocatedBytes += tierSize;
        
        span.setAttributes({
          success: true,
          requested_size: requestedSize,
          tier_size: tierSize,
          pool_hit: true,
          new_buffer: true
        });
        
        return newBuffer.subarray(0, requestedSize);
      }
      
      // Pool full, allocate directly
      span.setAttributes({
        success: true,
        requested_size: requestedSize,
        pool_hit: false,
        direct_allocation: true
      });
      
      return Buffer.alloc(requestedSize);
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      return Buffer.alloc(requestedSize); // Fallback
    } finally {
      span.end();
    }
  }
  
  /**
   * Return a buffer to the pool
   */
  returnPooledBuffer(buffer: Buffer): void {
    const span = LensTracer.createChildSpan('return_pooled_buffer');
    
    try {
      // Find the buffer in our pools
      for (const pool of this.bufferPools.values()) {
        const pooledBuffer = pool.find(item => item.buffer === buffer || 
          (item.inUse && buffer.length <= item.size));
        
        if (pooledBuffer && pooledBuffer.inUse) {
          pooledBuffer.inUse = false;
          pooledBuffer.lastUsed = Date.now();
          
          span.setAttributes({
            success: true,
            returned_to_pool: true
          });
          return;
        }
      }
      
      span.setAttributes({
        success: true,
        returned_to_pool: false,
        reason: 'buffer_not_found_in_pools'
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
   * Get a pooled array of specified type and capacity
   */
  getPooledArray<T>(type: string, requestedCapacity: number): T[] {
    const span = LensTracer.createChildSpan('get_pooled_array');
    this.totalRequests++;
    
    try {
      // Find appropriate capacity tier
      const tierCapacity = this.ARRAY_CAPACITY_TIERS.find(cap => cap >= requestedCapacity) ||
                          this.ARRAY_CAPACITY_TIERS[this.ARRAY_CAPACITY_TIERS.length - 1];
      
      const poolKey = `${type}-${tierCapacity}`;
      const pool = this.arrayPools.get(poolKey);
      
      if (!pool) {
        return new Array<T>(requestedCapacity);
      }
      
      // Find available array
      const availableArray = pool.find(item => !item.inUse);
      
      if (availableArray) {
        availableArray.inUse = true;
        availableArray.lastUsed = Date.now();
        availableArray.hits++;
        this.totalHits++;
        
        // Clear array for reuse
        availableArray.array.length = 0;
        
        span.setAttributes({
          success: true,
          type,
          requested_capacity: requestedCapacity,
          tier_capacity: tierCapacity,
          pool_hit: true
        });
        
        return availableArray.array as T[];
      }
      
      // Create new array if pool not full
      if (pool.length < this.MAX_POOLS_PER_TIER) {
        const newArray = new Array<T>(tierCapacity);
        const pooledArray: PooledArray<T> = {
          array: newArray,
          capacity: tierCapacity,
          inUse: true,
          lastUsed: Date.now(),
          hits: 1
        };
        
        pool.push(pooledArray);
        
        span.setAttributes({
          success: true,
          type,
          requested_capacity: requestedCapacity,
          tier_capacity: tierCapacity,
          new_array: true
        });
        
        return newArray;
      }
      
      // Fallback to direct allocation
      return new Array<T>(requestedCapacity);
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      return new Array<T>(requestedCapacity);
    } finally {
      span.end();
    }
  }
  
  /**
   * Return an array to the pool
   */
  returnPooledArray<T>(array: T[], type: string): void {
    const span = LensTracer.createChildSpan('return_pooled_array');
    
    try {
      // Find the array in appropriate pools
      for (const capacity of this.ARRAY_CAPACITY_TIERS) {
        const poolKey = `${type}-${capacity}`;
        const pool = this.arrayPools.get(poolKey);
        
        if (pool) {
          const pooledArray = pool.find(item => item.array === array && item.inUse);
          
          if (pooledArray) {
            pooledArray.inUse = false;
            pooledArray.lastUsed = Date.now();
            
            span.setAttributes({
              success: true,
              type,
              returned_to_pool: true
            });
            return;
          }
        }
      }
      
      span.setAttributes({
        success: true,
        type,
        returned_to_pool: false
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
   * Get memory pool statistics
   */
  getStats(): MemoryPoolStats {
    let totalBuffers = 0;
    let activeBuffers = 0;
    let totalArrays = 0;
    let activeArrays = 0;
    
    // Count buffers
    for (const pool of this.bufferPools.values()) {
      totalBuffers += pool.length;
      activeBuffers += pool.filter(item => item.inUse).length;
    }
    
    // Count arrays
    for (const pool of this.arrayPools.values()) {
      totalArrays += pool.length;
      activeArrays += pool.filter(item => item.inUse).length;
    }
    
    const hitRate = this.totalRequests > 0 ? (this.totalHits / this.totalRequests) * 100 : 0;
    const gcPressureReduction = this.gcEventsBeforePool > 0 ? 
      ((this.gcEventsBeforePool - this.gcEventsAfterPool) / this.gcEventsBeforePool) * 100 : 0;
    
    return {
      totalBuffers,
      activeBuffers,
      totalArrays,
      activeArrays,
      totalObjects: 0, // Not implemented yet
      activeObjects: 0,
      totalMemoryMB: this.totalAllocatedBytes / (1024 * 1024),
      hitRate,
      gcPressureReduction
    };
  }
  
  /**
   * Clean up unused pooled objects
   */
  private cleanup(): void {
    const span = LensTracer.createChildSpan('memory_pool_cleanup');
    let cleaned = 0;
    
    try {
      const now = Date.now();
      
      // Cleanup buffers
      for (const [tierSize, pool] of this.bufferPools.entries()) {
        const initialLength = pool.length;
        const retained = pool.filter(item => {
          if (!item.inUse && (now - item.lastUsed) > this.MAX_IDLE_TIME) {
            this.totalAllocatedBytes -= item.size;
            cleaned++;
            return false;
          }
          return true;
        });
        
        if (retained.length !== initialLength) {
          this.bufferPools.set(tierSize, retained);
        }
      }
      
      // Cleanup arrays
      for (const [poolKey, pool] of this.arrayPools.entries()) {
        const retained = pool.filter(item => {
          if (!item.inUse && (now - item.lastUsed) > this.MAX_IDLE_TIME) {
            cleaned++;
            return false;
          }
          return true;
        });
        
        if (retained.length !== pool.length) {
          this.arrayPools.set(poolKey, retained);
        }
      }
      
      if (cleaned > 0) {
        console.log(`🧹 Memory pool cleanup: removed ${cleaned} unused objects, saved ${(this.totalAllocatedBytes / (1024 * 1024)).toFixed(2)}MB`);
      }
      
      span.setAttributes({
        success: true,
        objects_cleaned: cleaned,
        total_memory_mb: this.totalAllocatedBytes / (1024 * 1024)
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
   * Start cleanup timer
   */
  private startCleanupTimer(): void {
    this.cleanupTimer = setInterval(() => {
      this.cleanup();
    }, this.POOL_CLEANUP_INTERVAL);
  }
  
  /**
   * Monitor GC events to measure effectiveness
   */
  private monitorGC(): void {
    // Track initial GC baseline
    const initialMemUsage = process.memoryUsage();
    this.gcEventsBeforePool = 0; // Would need actual GC monitoring
    
    // Set up periodic GC monitoring
    setInterval(() => {
      const memUsage = process.memoryUsage();
      
      // Log memory statistics
      console.log(`📊 Memory Pool Stats: ${JSON.stringify({
        heap_used_mb: (memUsage.heapUsed / (1024 * 1024)).toFixed(2),
        pool_memory_mb: (this.totalAllocatedBytes / (1024 * 1024)).toFixed(2),
        hit_rate_percent: this.totalRequests > 0 ? ((this.totalHits / this.totalRequests) * 100).toFixed(1) : '0.0',
        active_buffers: this.getStats().activeBuffers,
        active_arrays: this.getStats().activeArrays
      })}`);
    }, 60000); // Every minute
  }
  
  /**
   * Shutdown the memory pool manager
   */
  shutdown(): void {
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer);
    }
    
    // Clear all pools
    this.bufferPools.clear();
    this.arrayPools.clear();
    this.objectPools.clear();
    
    console.log('🛑 Memory Pool Manager shutdown complete');
  }
}

// Global instance
export const globalMemoryPool = MemoryPoolManager.getInstance();