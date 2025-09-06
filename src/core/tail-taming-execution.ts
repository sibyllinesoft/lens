/**
 * Tail-Taming Execution System
 * Implements hedged Stage-A probes with cooperative cancellation
 * Targets slowest 5-10% queries with bounded result streaming
 */

import { LensTracer } from '../telemetry/tracer.js';
import type { SearchContext, Candidate } from '../types/core.js';

export interface HedgedRequestConfig {
  enabled: boolean;
  slowQueryPercentile: number; // 90-95th percentile
  hedgeTriggerRatio: number; // 0.5 = p95_budget/2
  maxHedgeRate: number; // Cap hedge rate under high load
  cooperativeCancelTimeoutMs: number;
  enableResultStreaming: boolean;
}

export interface ShardRequest {
  shardId: string;
  requestId: string;
  startTime: number;
  isHedged: boolean;
  cancelled: boolean;
  results?: Candidate[];
  error?: Error;
}

export interface HedgeMetrics {
  total_requests: number;
  hedged_requests: number;
  hedge_wins: number; // Hedged request finished first
  cancelled_requests: number;
  avg_hedge_latency_ms: number;
  p95_improvement_ms: number;
  load_based_throttles: number;
}

export class TailTamingExecutor {
  private config: HedgedRequestConfig;
  private performanceHistory: number[] = [];
  private activeRequests: Map<string, ShardRequest[]> = new Map();
  private metrics: HedgeMetrics = {
    total_requests: 0,
    hedged_requests: 0,
    hedge_wins: 0,
    cancelled_requests: 0,
    avg_hedge_latency_ms: 0,
    p95_improvement_ms: 0,
    load_based_throttles: 0,
  };
  private hotFlags: Map<string, boolean> = new Map(); // For cooperative cancellation

  constructor(config: Partial<HedgedRequestConfig> = {}) {
    this.config = {
      enabled: false, // Start disabled for gradual rollout
      slowQueryPercentile: 90, // Target slowest 10% initially
      hedgeTriggerRatio: 0.5, // Fire hedge at p95_budget/2
      maxHedgeRate: 0.1, // Max 10% of requests can be hedged
      cooperativeCancelTimeoutMs: 100, // 100ms for scan blocks to check
      enableResultStreaming: true,
      ...config,
    };
  }

  /**
   * Execute hedged request with cooperative cancellation
   */
  async executeWithHedging<T>(
    ctx: SearchContext,
    shardIds: string[],
    executor: (shardId: string, requestId: string) => Promise<T>,
    combiner: (results: T[]) => T
  ): Promise<{ result: T; wasHedged: boolean; latencyMs: number }> {
    const span = LensTracer.createChildSpan('hedged_execution');
    const requestId = `hedged_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const startTime = Date.now();

    try {
      this.metrics.total_requests++;

      // Check if we should apply hedging
      if (!this.shouldApplyHedging()) {
        span.setAttributes({ hedging_applied: false, reason: 'disabled_or_throttled' });
        
        // Execute normally without hedging
        const results = await Promise.all(
          shardIds.map(shardId => executor(shardId, requestId))
        );
        
        const latency = Date.now() - startTime;
        this.updatePerformanceHistory(latency);
        
        return { result: combiner(results), wasHedged: false, latencyMs: latency };
      }

      // Calculate hedge trigger timeout based on p95 budget
      const p95Budget = this.calculateP95Budget();
      const hedgeTriggerMs = p95Budget * this.config.hedgeTriggerRatio;

      span.setAttributes({ 
        hedging_applied: true, 
        p95_budget_ms: p95Budget,
        hedge_trigger_ms: hedgeTriggerMs,
        shard_count: shardIds.length,
      });

      // Initialize shard requests
      const shardRequests: ShardRequest[] = shardIds.map(shardId => ({
        shardId,
        requestId: `${requestId}_${shardId}`,
        startTime: Date.now(),
        isHedged: false,
        cancelled: false,
        error: undefined,
      }));
      
      this.activeRequests.set(requestId, shardRequests);
      
      // Launch initial requests
      const primaryPromises = shardRequests.map(async (req) => {
        try {
          return await this.executeWithCancellation(
            req.shardId, 
            req.requestId,
            executor
          );
        } catch (error) {
          req.error = error as Error;
          throw error;
        }
      });

      // Set up hedge timeout
      const hedgeTimeout = setTimeout(async () => {
        await this.launchHedgedRequests(requestId, shardRequests, executor);
      }, hedgeTriggerMs);

      // Race between primary requests and hedge timeout
      let results: T[];
      let hedgeWasUsed = false;

      try {
        // Use Promise.allSettled to handle partial results
        if (this.config.enableResultStreaming) {
          results = await this.executeWithStreaming(primaryPromises, combiner);
        } else {
          results = await Promise.all(primaryPromises);
        }
        
        // Cancel hedge timeout if primary requests completed
        clearTimeout(hedgeTimeout);
        
      } catch (error) {
        // If primary requests fail, wait for hedged requests
        clearTimeout(hedgeTimeout);
        
        const hedgedResults = await this.waitForHedgedResults(requestId);
        if (hedgedResults.length > 0) {
          results = hedgedResults as T[];
          hedgeWasUsed = true;
          this.metrics.hedge_wins++;
        } else {
          throw error;
        }
      }

      // Clean up and record metrics
      await this.cleanupRequest(requestId);
      
      const latency = Date.now() - startTime;
      this.updatePerformanceHistory(latency);
      
      if (hedgeWasUsed) {
        this.metrics.hedged_requests++;
        this.metrics.avg_hedge_latency_ms = 
          (this.metrics.avg_hedge_latency_ms + latency) / 2;
      }

      span.setAttributes({
        success: true,
        hedge_used: hedgeWasUsed,
        latency_ms: latency,
        results_count: results.length,
      });

      return { 
        result: combiner(results), 
        wasHedged: hedgeWasUsed, 
        latencyMs: latency 
      };

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: (error as Error).message });
      
      await this.cleanupRequest(requestId);
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Update configuration for performance tuning
   */
  updateConfig(newConfig: Partial<HedgedRequestConfig>): void {
    this.config = { ...this.config, ...newConfig };
    console.log(`🚀 Tail-Taming Executor config updated:`, this.config);
  }

  /**
   * Get performance metrics
   */
  getMetrics(): HedgeMetrics & { 
    current_p95_ms: number; 
    hedge_ratio: number;
    p99_p95_ratio: number;
  } {
    const p95 = this.calculateP95Budget();
    const p99 = this.calculatePercentile(99);
    const hedgeRatio = this.metrics.total_requests > 0 
      ? this.metrics.hedged_requests / this.metrics.total_requests 
      : 0;

    return {
      ...this.metrics,
      current_p95_ms: p95,
      hedge_ratio: hedgeRatio,
      p99_p95_ratio: p95 > 0 ? p99 / p95 : 0,
    };
  }

  /**
   * Check system health for load-based throttling
   */
  shouldThrottleHedging(): boolean {
    const currentLoad = this.activeRequests.size;
    const hedgeRatio = this.metrics.total_requests > 0 
      ? this.metrics.hedged_requests / this.metrics.total_requests 
      : 0;

    // Throttle if hedge rate exceeds maximum or system is under high load
    if (hedgeRatio > this.config.maxHedgeRate) {
      this.metrics.load_based_throttles++;
      return true;
    }

    if (currentLoad > 100) { // Arbitrary high load threshold
      this.metrics.load_based_throttles++;
      return true;
    }

    return false;
  }

  private shouldApplyHedging(): boolean {
    if (!this.config.enabled) return false;
    if (this.shouldThrottleHedging()) return false;
    
    // Only hedge if we have enough performance history
    return this.performanceHistory.length >= 10;
  }

  private calculateP95Budget(): number {
    if (this.performanceHistory.length === 0) return 100; // Default budget
    return this.calculatePercentile(95);
  }

  private calculatePercentile(percentile: number): number {
    if (this.performanceHistory.length === 0) return 0;
    
    const sorted = [...this.performanceHistory].sort((a, b) => a - b);
    const index = Math.floor((percentile / 100) * sorted.length);
    return sorted[index] || sorted[sorted.length - 1];
  }

  private updatePerformanceHistory(latency: number): void {
    this.performanceHistory.push(latency);
    
    // Keep only recent history (last 1000 requests)
    if (this.performanceHistory.length > 1000) {
      this.performanceHistory = this.performanceHistory.slice(-500);
    }
  }

  private async executeWithCancellation<T>(
    shardId: string,
    requestId: string,
    executor: (shardId: string, requestId: string) => Promise<T>
  ): Promise<T> {
    // Set up cancellation mechanism
    this.hotFlags.set(requestId, false);
    
    try {
      // Execute with periodic cancellation checks for long-running operations
      const result = await this.wrapWithCancellationCheck(
        executor(shardId, requestId),
        requestId
      );
      
      return result;
    } finally {
      this.hotFlags.delete(requestId);
    }
  }

  private async wrapWithCancellationCheck<T>(
    promise: Promise<T>,
    requestId: string
  ): Promise<T> {
    // Create a cancellation promise that resolves when cancelled
    const cancellationPromise = new Promise<never>((_, reject) => {
      const checkCancellation = () => {
        if (this.hotFlags.get(requestId) === true) {
          reject(new Error('Request cancelled cooperatively'));
          return;
        }
        
        // Check again after timeout
        setTimeout(checkCancellation, this.config.cooperativeCancelTimeoutMs);
      };
      
      checkCancellation();
    });

    // Race between the actual promise and cancellation
    return Promise.race([promise, cancellationPromise]);
  }

  private async launchHedgedRequests<T>(
    requestId: string,
    shardRequests: ShardRequest[],
    executor: (shardId: string, requestId: string) => Promise<T>
  ): Promise<void> {
    const requests = this.activeRequests.get(requestId);
    if (!requests) return;

    // Launch hedged requests for shards that haven't responded
    const hedgePromises = shardRequests
      .filter(req => !req.results && !req.error && !req.cancelled)
      .map(async (req) => {
        const hedgedRequestId = `${req.requestId}_hedge`;
        const hedgedReq: ShardRequest = {
          shardId: req.shardId,
          requestId: hedgedRequestId,
          startTime: Date.now(),
          isHedged: true,
          cancelled: false,
        };

        try {
          const result = await this.executeWithCancellation(
            req.shardId,
            hedgedRequestId,
            executor
          );
          
          hedgedReq.results = [result as any];
          
          // Cancel the original request
          this.cancelRequest(req.requestId);
          
        } catch (error) {
          hedgedReq.error = error as Error;
        }

        requests.push(hedgedReq);
      });

    await Promise.allSettled(hedgePromises);
  }

  private async executeWithStreaming<T>(
    promises: Promise<T>[],
    combiner: (results: T[]) => T
  ): Promise<T[]> {
    const results: T[] = [];
    const settled = await Promise.allSettled(promises);
    
    for (const result of settled) {
      if (result.status === 'fulfilled') {
        results.push(result.value);
      }
    }

    // For streaming, we can start Stage-C processing as soon as we have partial results
    if (results.length > 0) {
      return results;
    }

    throw new Error('All shard requests failed');
  }

  private async waitForHedgedResults(requestId: string): Promise<any[]> {
    const requests = this.activeRequests.get(requestId);
    if (!requests) return [];

    const hedgedRequests = requests.filter(req => req.isHedged);
    const results = hedgedRequests
      .filter(req => req.results && req.results.length > 0)
      .flatMap(req => req.results!);

    return results;
  }

  private cancelRequest(requestId: string): void {
    this.hotFlags.set(requestId, true);
    this.metrics.cancelled_requests++;
  }

  private async cleanupRequest(requestId: string): Promise<void> {
    const requests = this.activeRequests.get(requestId);
    if (!requests) return;

    // Cancel any remaining active requests
    for (const req of requests) {
      if (!req.cancelled && !req.results && !req.error) {
        this.cancelRequest(req.requestId);
      }
    }

    // Remove from active requests
    this.activeRequests.delete(requestId);
    
    // Clean up hot flags
    for (const req of requests) {
      this.hotFlags.delete(req.requestId);
    }
  }
}

/**
 * Factory for creating tail-taming executor
 */
export function createTailTamingExecutor(config?: Partial<HedgedRequestConfig>): TailTamingExecutor {
  return new TailTamingExecutor(config);
}