/**
 * Hedged probe service for tail-taming
 * Implements Section 2 of TODO.md: secondary probe at t = min(6ms, 0.1·p50_shard); cancel on first success
 */

import { LensClient, LensSearchRequest, LensSearchResponse } from '../clients/lens-client';
import { TailTamingConfig, calculateHedgeDelay } from '../config/tail-taming-config';

export interface HedgedProbeMetrics {
  probe_id: string;
  issued_ts: number;
  first_byte_ts?: number;
  cancel_ts?: number;
  latency_ms: number;
  hedge_delay_ms: number;
  was_hedged: boolean;
  won_race: boolean; // True if this probe returned first
  cancelled: boolean;
}

export interface HedgedSearchResult {
  response?: LensSearchResponse;
  primary_metrics: HedgedProbeMetrics;
  hedge_metrics?: HedgedProbeMetrics;
  total_latency_ms: number;
  hedge_improvement_ms?: number; // How much hedge saved vs primary alone
}

export class HedgedProbeService {
  private readonly config: TailTamingConfig;
  private p50ShardLatency = 100; // Default, should be updated from metrics

  constructor(config: TailTamingConfig) {
    this.config = config;
  }

  /**
   * Update the p50 shard latency for hedge delay calculation
   */
  updateP50ShardLatency(p50: number): void {
    this.p50ShardLatency = p50;
  }

  /**
   * Perform search with hedged probes if enabled
   */
  async search(
    primaryClient: LensClient, 
    hedgeClient: LensClient,
    request: LensSearchRequest
  ): Promise<HedgedSearchResult> {
    const startTime = Date.now();
    const probe_id_primary = `primary_${startTime}_${Math.random().toString(36).substr(2, 9)}`;
    const probe_id_hedge = `hedge_${startTime}_${Math.random().toString(36).substr(2, 9)}`;

    if (!this.config.TAIL_HEDGE) {
      // No hedging - just use primary client
      return await this.performPrimaryOnly(primaryClient, request, probe_id_primary);
    }

    const hedgeDelay = calculateHedgeDelay(this.config, this.p50ShardLatency);
    
    return await this.performHedgedSearch(
      primaryClient,
      hedgeClient, 
      request,
      probe_id_primary,
      probe_id_hedge,
      hedgeDelay
    );
  }

  private async performPrimaryOnly(
    primaryClient: LensClient,
    request: LensSearchRequest,
    probeId: string
  ): Promise<HedgedSearchResult> {
    const issued_ts = Date.now();
    
    const { response, metrics } = await primaryClient.search(request);
    
    const primary_metrics: HedgedProbeMetrics = {
      probe_id: probeId,
      issued_ts,
      first_byte_ts: issued_ts + metrics.lat_ms, // Approximate
      latency_ms: metrics.lat_ms,
      hedge_delay_ms: 0,
      was_hedged: false,
      won_race: true, // Only probe, so it "wins"
      cancelled: false
    };

    return {
      response,
      primary_metrics,
      total_latency_ms: metrics.lat_ms
    };
  }

  private async performHedgedSearch(
    primaryClient: LensClient,
    hedgeClient: LensClient,
    request: LensSearchRequest,
    primaryProbeId: string,
    hedgeProbeId: string,
    hedgeDelay: number
  ): Promise<HedgedSearchResult> {
    const startTime = Date.now();
    let primaryResult: any = null;
    let hedgeResult: any = null;
    let primaryCancelled = false;
    let hedgeCancelled = false;

    const primaryController = new AbortController();
    const hedgeController = new AbortController();

    // Start primary request immediately
    const primaryPromise = this.trackProbeExecution(
      primaryClient, 
      request, 
      primaryProbeId,
      primaryController.signal
    );

    // Start hedge request after delay
    const hedgePromise = new Promise<any>((resolve) => {
      setTimeout(async () => {
        if (!primaryController.signal.aborted) {
          const hedgeResult = await this.trackProbeExecution(
            hedgeClient, 
            request, 
            hedgeProbeId, 
            hedgeController.signal
          );
          resolve(hedgeResult);
        } else {
          resolve(null); // Primary already won
        }
      }, hedgeDelay);
    });

    try {
      // Race between primary and hedge
      const raceResults = await Promise.allSettled([primaryPromise, hedgePromise]);
      
      const primarySettled = raceResults[0];
      const hedgeSettled = raceResults[1];

      // Determine winner and cancel loser if configured
      if (primarySettled.status === 'fulfilled' && primarySettled.value) {
        primaryResult = primarySettled.value;
        
        if (this.config.cancel_on_first_success && !hedgeController.signal.aborted) {
          hedgeController.abort();
          hedgeCancelled = true;
        }
      } else if (hedgeSettled.status === 'fulfilled' && hedgeSettled.value) {
        hedgeResult = hedgeSettled.value;
        
        if (this.config.cancel_on_first_success && !primaryController.signal.aborted) {
          primaryController.abort();
          primaryCancelled = true;
        }
      }

      // If both failed, wait for whichever completes
      if (!primaryResult && !hedgeResult) {
        // Both failed - use primary failure as representative
        if (primarySettled.status === 'fulfilled') {
          primaryResult = primarySettled.value;
        }
      }

    } catch (error) {
      // Handle any race condition errors
      console.warn('Error in hedged search race:', error);
    }

    const totalLatency = Date.now() - startTime;
    
    // Build result metrics
    const winnerResult = primaryResult?.response ? primaryResult : hedgeResult;
    const winner = primaryResult?.response ? 'primary' : 'hedge';

    const primary_metrics: HedgedProbeMetrics = {
      probe_id: primaryProbeId,
      issued_ts: startTime,
      first_byte_ts: primaryResult?.first_byte_ts,
      cancel_ts: primaryCancelled ? Date.now() : undefined,
      latency_ms: primaryResult?.metrics.lat_ms || 0,
      hedge_delay_ms: 0,
      was_hedged: false,
      won_race: winner === 'primary',
      cancelled: primaryCancelled
    };

    let hedge_metrics: HedgedProbeMetrics | undefined;
    if (hedgeResult) {
      hedge_metrics = {
        probe_id: hedgeProbeId,
        issued_ts: startTime + hedgeDelay,
        first_byte_ts: hedgeResult?.first_byte_ts,
        cancel_ts: hedgeCancelled ? Date.now() : undefined,
        latency_ms: hedgeResult?.metrics.lat_ms || 0,
        hedge_delay_ms: hedgeDelay,
        was_hedged: true,
        won_race: winner === 'hedge',
        cancelled: hedgeCancelled
      };
    }

    // Calculate hedge improvement
    let hedge_improvement_ms: number | undefined;
    if (hedge_metrics && primary_metrics.latency_ms > 0) {
      const primaryTotal = primary_metrics.latency_ms;
      const hedgeTotal = hedgeDelay + hedge_metrics.latency_ms;
      hedge_improvement_ms = Math.max(0, primaryTotal - hedgeTotal);
    }

    return {
      response: winnerResult?.response,
      primary_metrics,
      hedge_metrics,
      total_latency_ms: totalLatency,
      hedge_improvement_ms
    };
  }

  private async trackProbeExecution(
    client: LensClient,
    request: LensSearchRequest,
    probeId: string,
    signal: AbortSignal
  ): Promise<{
    response?: LensSearchResponse;
    metrics: any;
    first_byte_ts: number;
    probe_id: string;
  }> {
    const issued_ts = Date.now();
    
    try {
      // Wrap the client search to respect abort signal
      const searchPromise = client.search(request);
      
      // Race between search and abort
      const result = await Promise.race([
        searchPromise,
        new Promise((_, reject) => {
          signal.addEventListener('abort', () => {
            reject(new Error('PROBE_CANCELLED'));
          });
        })
      ]);

      const first_byte_ts = Date.now();
      
      return {
        ...(result as any),
        first_byte_ts,
        probe_id: probeId
      };
      
    } catch (error) {
      if (error instanceof Error && error.message === 'PROBE_CANCELLED') {
        return {
          metrics: {
            lat_ms: Date.now() - issued_ts,
            success: false,
            error_code: 'CANCELLED'
          },
          first_byte_ts: Date.now(),
          probe_id: probeId
        };
      }
      
      throw error;
    }
  }

  /**
   * Analyze hedge effectiveness for metrics collection
   */
  analyzeHedgeEffectiveness(results: HedgedSearchResult[]): {
    hedge_win_rate: number;
    avg_improvement_ms: number;
    hedge_usage_rate: number;
    p95_improvement_ms: number;
  } {
    const hedgedResults = results.filter(r => r.hedge_metrics);
    const hedgeWins = results.filter(r => r.hedge_metrics?.won_race).length;
    
    const improvements = results
      .map(r => r.hedge_improvement_ms)
      .filter((imp): imp is number => imp !== undefined && imp > 0);

    const avgImprovement = improvements.length > 0 
      ? improvements.reduce((sum, imp) => sum + imp, 0) / improvements.length 
      : 0;

    const p95Index = Math.floor(improvements.length * 0.95);
    const p95Improvement = improvements.length > 0 
      ? improvements.sort((a, b) => a - b)[p95Index] || 0 
      : 0;

    return {
      hedge_win_rate: hedgedResults.length > 0 ? hedgeWins / hedgedResults.length : 0,
      avg_improvement_ms: avgImprovement,
      hedge_usage_rate: hedgedResults.length / results.length,
      p95_improvement_ms: p95Improvement
    };
  }
}