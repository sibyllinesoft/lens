/**
 * Simulated data ingestor for testing and fallback
 * Implements fallback behavior when DATA_SOURCE=sim
 */

import crypto from 'crypto';
import { AggregationRecord, HitsRecord } from '../schemas/output-schemas';
import { LensSearchRequest } from '../clients/lens-client';
import { IngestorMetrics } from './prod-ingestor';

export class SimIngestor {
  private readonly configHash: string;

  constructor() {
    this.configHash = this.generateConfigHash();
  }

  async ingestQueries(queries: LensSearchRequest[]): Promise<{
    aggRecords: AggregationRecord[];
    hitsRecords: HitsRecord[];
    metrics: IngestorMetrics;
  }> {
    console.log(`Starting simulated ingestion of ${queries.length} queries...`);
    
    const aggRecords: AggregationRecord[] = [];
    const hitsRecords: HitsRecord[] = [];
    
    let successful_queries = 0;
    let total_latency = 0;
    let within_sla_count = 0;

    for (let i = 0; i < queries.length; i++) {
      const query = queries[i];
      
      // Simulate processing delay
      const simulatedLatency = this.generateRealisticLatency();
      await this.sleep(simulatedLatency);
      
      const result = this.simulateQueryResult(query, simulatedLatency);
      
      aggRecords.push(result.aggRecord);
      hitsRecords.push(...result.hitsRecords);
      
      successful_queries++;
      total_latency += simulatedLatency;
      
      if (result.aggRecord.within_sla) {
        within_sla_count++;
      }
    }

    const metrics: IngestorMetrics = {
      total_queries: queries.length,
      successful_queries,
      failed_queries: queries.length - successful_queries,
      avg_latency_ms: total_latency / queries.length,
      within_sla_percentage: (within_sla_count / queries.length) * 100,
      attestation_sha256: this.generateAttestationHash(aggRecords, hitsRecords)
    };

    console.log(`Simulated ingestion complete. Metrics:`, metrics);

    return { aggRecords, hitsRecords, metrics };
  }

  private simulateQueryResult(query: LensSearchRequest, latency: number): {
    aggRecord: AggregationRecord;
    hitsRecords: HitsRecord[];
  } {
    const queryId = `sim_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const req_ts = Date.now() - latency;
    
    // Simulate realistic hit counts and why distributions
    const totalHits = Math.floor(Math.random() * 20) + 1;
    const lexHits = Math.floor(totalHits * (0.4 + Math.random() * 0.3));
    const structHits = Math.floor(totalHits * (0.2 + Math.random() * 0.3));
    const semHits = totalHits - lexHits - structHits;

    const aggRecord: AggregationRecord = {
      query_id: queryId,
      req_ts,
      cfg_hash: this.configHash,
      shard: `sim_shard_${Math.floor(Math.random() * 4)}`,
      lat_ms: latency,
      within_sla: latency <= 150,
      why_mix_lex: Math.max(0, lexHits),
      why_mix_struct: Math.max(0, structHits), 
      why_mix_sem: Math.max(0, semHits),
      endpoint_url: 'http://simulator:3000',
      success: true,
      total_hits: totalHits
    };

    const hitsRecords: HitsRecord[] = [];
    
    // Generate simulated hits
    for (let i = 0; i < totalHits; i++) {
      const why = i < lexHits ? 'lex' : (i < lexHits + structHits ? 'struct' : 'sem');
      
      hitsRecords.push({
        query_id: queryId,
        req_ts,
        hit_id: `${queryId}_hit_${i}`,
        file_path: `src/simulated/file_${Math.floor(Math.random() * 100)}.ts`,
        line_start: Math.floor(Math.random() * 1000) + 1,
        line_end: Math.floor(Math.random() * 1000) + 1,
        score: Math.random() * 0.9 + 0.1, // Score between 0.1 and 1.0
        why: why as 'lex' | 'struct' | 'sem',
        span_start: Math.floor(Math.random() * 100),
        span_end: Math.floor(Math.random() * 100) + 100,
        content_preview: `Simulated content for query "${query.query}" - hit ${i}`,
        shard_id: aggRecord.shard,
        shard_latency_ms: latency
      });
    }

    return { aggRecord, hitsRecords };
  }

  private generateRealisticLatency(): number {
    // Simulate realistic latency distribution
    // 80% of queries under 100ms, 15% between 100-150ms, 5% over 150ms
    const rand = Math.random();
    
    if (rand < 0.8) {
      // Fast queries: 10-100ms
      return Math.floor(Math.random() * 90) + 10;
    } else if (rand < 0.95) {
      // Medium queries: 100-150ms
      return Math.floor(Math.random() * 50) + 100;
    } else {
      // Slow queries: 150-300ms (SLA violations)
      return Math.floor(Math.random() * 150) + 150;
    }
  }

  private generateConfigHash(): string {
    const configData = {
      DATA_SOURCE: 'sim',
      timestamp: new Date().toISOString().split('T')[0] // Daily rotation
    };
    
    const configString = JSON.stringify(configData);
    return crypto.createHash('sha256').update(configString).digest('hex').substring(0, 16);
  }

  private generateAttestationHash(aggRecords: AggregationRecord[], hitsRecords: HitsRecord[]): string {
    const attestationData = {
      config_hash: this.configHash,
      agg_record_count: aggRecords.length,
      hits_record_count: hitsRecords.length,
      timestamp: Math.floor(Date.now() / 1000),
      simulation_version: '1.0.0'
    };
    
    const attestationString = JSON.stringify(attestationData);
    return crypto.createHash('sha256').update(attestationString).digest('hex');
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  cleanup(): void {
    // No cleanup needed for simulator
  }
}