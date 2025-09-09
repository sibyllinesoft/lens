/**
 * Production data ingestor for lens search results
 * Implements Section 1 of TODO.md: ProdIngestor with real endpoint integration
 */

import crypto from 'crypto';
import { LensClient, LensSearchRequest } from '../clients/lens-client.js';
import { AggregationRecord, HitsRecord, validateAggregationSchema, validateHitsSchema } from '../schemas/output-schemas.js';
import { getDataSourceConfig, createLensEndpointConfigs } from '../config/data-source-config.js';

export interface IngestorMetrics {
  total_queries: number;
  successful_queries: number;
  failed_queries: number;
  avg_latency_ms: number;
  within_sla_percentage: number;
  attestation_sha256: string;
}

export class ProdIngestor {
  private readonly clients: LensClient[];
  private readonly config = getDataSourceConfig();
  private readonly configHash: string;

  constructor() {
    if (this.config.DATA_SOURCE !== 'prod') {
      throw new Error('ProdIngestor can only be used with DATA_SOURCE=prod');
    }

    const endpointConfigs = createLensEndpointConfigs(this.config);
    this.clients = endpointConfigs.map(config => new LensClient(config));
    
    // Generate configuration hash for reproducibility tracking
    this.configHash = this.generateConfigHash();
  }

  async ingestQueries(queries: LensSearchRequest[]): Promise<{
    aggRecords: AggregationRecord[];
    hitsRecords: HitsRecord[];
    metrics: IngestorMetrics;
  }> {
    const aggRecords: AggregationRecord[] = [];
    const hitsRecords: HitsRecord[] = [];
    
    let successful_queries = 0;
    let failed_queries = 0;
    let total_latency = 0;
    let within_sla_count = 0;

    console.log(`Starting production ingestion of ${queries.length} queries...`);

    for (let i = 0; i < queries.length; i++) {
      const query = queries[i];
      console.log(`Processing query ${i + 1}/${queries.length}: ${query.query}`);

      try {
        const result = await this.processQuery(query);
        
        aggRecords.push(result.aggRecord);
        hitsRecords.push(...result.hitsRecords);
        
        if (result.aggRecord.success) {
          successful_queries++;
        } else {
          failed_queries++;
        }
        
        total_latency += result.aggRecord.lat_ms;
        if (result.aggRecord.within_sla) {
          within_sla_count++;
        }
        
      } catch (error) {
        console.error(`Query ${i + 1} failed:`, error);
        failed_queries++;
        
        // Create error record
        const errorRecord = this.createErrorRecord(query, error as Error);
        aggRecords.push(errorRecord);
        total_latency += this.config.SLA_MS; // Assume SLA violation
      }
    }

    // Validate schemas before returning
    const aggValidation = validateAggregationSchema(aggRecords);
    const hitsValidation = validateHitsSchema(hitsRecords);

    if (!aggValidation.valid) {
      throw new Error(`Aggregation schema validation failed: ${JSON.stringify(aggValidation)}`);
    }
    
    if (!hitsValidation.valid) {
      throw new Error(`Hits schema validation failed: ${JSON.stringify(hitsValidation)}`);
    }

    const metrics: IngestorMetrics = {
      total_queries: queries.length,
      successful_queries,
      failed_queries,
      avg_latency_ms: total_latency / queries.length,
      within_sla_percentage: (within_sla_count / queries.length) * 100,
      attestation_sha256: this.generateAttestationHash(aggRecords, hitsRecords)
    };

    console.log(`Production ingestion complete. Metrics:`, metrics);

    return { aggRecords, hitsRecords, metrics };
  }

  private async processQuery(query: LensSearchRequest): Promise<{
    aggRecord: AggregationRecord;
    hitsRecords: HitsRecord[];
  }> {
    // Use the first client for now - could implement load balancing later
    const client = this.clients[0];
    
    const { response, metrics } = await client.search(query);
    
    // Count why types from hits
    const whyCounts = this.countWhyTypes(response?.hits || []);
    
    const aggRecord: AggregationRecord = {
      query_id: response?.query_id || `error_${Date.now()}`,
      req_ts: metrics.req_ts,
      cfg_hash: this.configHash,
      shard: metrics.shard,
      lat_ms: metrics.lat_ms,
      within_sla: metrics.within_sla,
      why_mix_lex: whyCounts.lex,
      why_mix_struct: whyCounts.struct,
      why_mix_sem: whyCounts.sem,
      endpoint_url: metrics.endpoint_url,
      success: metrics.success,
      error_code: metrics.error_code,
      total_hits: response?.total_hits || 0
    };

    const hitsRecords: HitsRecord[] = response?.hits.map(hit => ({
      query_id: response.query_id,
      req_ts: metrics.req_ts,
      hit_id: `${response.query_id}_${hit.file_path}_${hit.line_start}`,
      file_path: hit.file_path,
      line_start: hit.line_start,
      line_end: hit.line_end,
      score: hit.score,
      why: hit.why,
      span_start: hit.span_start,
      span_end: hit.span_end,
      content_preview: hit.content.substring(0, 500), // Truncate for storage
      shard_id: response.shard_id,
      shard_latency_ms: response.latency_ms
    })) || [];

    return { aggRecord, hitsRecords };
  }

  private createErrorRecord(query: LensSearchRequest, error: Error): AggregationRecord {
    return {
      query_id: `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      req_ts: Date.now(),
      cfg_hash: this.configHash,
      shard: 'error',
      lat_ms: this.config.SLA_MS, // Assume SLA violation
      within_sla: false,
      why_mix_lex: 0,
      why_mix_struct: 0,
      why_mix_sem: 0,
      endpoint_url: this.clients[0] ? this.config.LENS_ENDPOINTS[0] : 'unknown',
      success: false,
      error_code: error.message,
      total_hits: 0
    };
  }

  private countWhyTypes(hits: any[]): { lex: number; struct: number; sem: number } {
    const counts = { lex: 0, struct: 0, sem: 0 };
    
    for (const hit of hits) {
      if (hit.why === 'lex') counts.lex++;
      else if (hit.why === 'struct') counts.struct++;
      else if (hit.why === 'sem') counts.sem++;
    }
    
    return counts;
  }

  private generateConfigHash(): string {
    const configForHashing = {
      DATA_SOURCE: this.config.DATA_SOURCE,
      LENS_ENDPOINTS: this.config.LENS_ENDPOINTS.sort(), // Sort for consistency
      SLA_MS: this.config.SLA_MS,
      RETRY_COUNT: this.config.RETRY_COUNT,
      CANCEL_ON_FIRST: this.config.CANCEL_ON_FIRST,
      // Exclude AUTH_TOKEN from hash for security
    };
    
    const configString = JSON.stringify(configForHashing);
    return crypto.createHash('sha256').update(configString).digest('hex').substring(0, 16);
  }

  private generateAttestationHash(aggRecords: AggregationRecord[], hitsRecords: HitsRecord[]): string {
    // Create a deterministic hash of the results for attestation
    const attestationData = {
      config_hash: this.configHash,
      agg_record_count: aggRecords.length,
      hits_record_count: hitsRecords.length,
      timestamp: Math.floor(Date.now() / 1000), // Round to seconds for stability
      total_successful: aggRecords.filter(r => r.success).length,
      avg_latency: aggRecords.reduce((sum, r) => sum + r.lat_ms, 0) / aggRecords.length
    };
    
    const attestationString = JSON.stringify(attestationData);
    return crypto.createHash('sha256').update(attestationString).digest('hex');
  }

  // Graceful shutdown
  cleanup(): void {
    this.clients.forEach(client => client.cancel());
  }
}

// Factory function to create appropriate ingestor based on configuration
export async function createIngestor() {
  const config = getDataSourceConfig();
  
  if (config.DATA_SOURCE === 'prod') {
    return new ProdIngestor();
  } else {
    // Import SimIngestor dynamically
    const { SimIngestor } = await import('./sim-ingestor');
    return new SimIngestor();
  }
}