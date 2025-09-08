/**
 * Schema guard service to refuse writes if required columns missing
 * Implements Section 1 of TODO.md: schema validation and attestation
 */

import crypto from 'crypto';
import { AggregationRecord, HitsRecord, validateAggregationSchema, validateHitsSchema } from '../schemas/output-schemas';

export interface SchemaGuardResult {
  valid: boolean;
  attestation_sha256?: string;
  errors: string[];
  warnings: string[];
}

export class SchemaGuard {
  /**
   * Validate and prepare data for write operations
   * Refuses writes if any required column is missing
   */
  static validateForWrite(
    aggRecords: AggregationRecord[],
    hitsRecords: HitsRecord[]
  ): SchemaGuardResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Validate aggregation records
    const aggValidation = validateAggregationSchema(aggRecords);
    if (!aggValidation.valid) {
      if (aggValidation.missing_columns.length > 0) {
        errors.push(`Missing required aggregation columns: ${aggValidation.missing_columns.join(', ')}`);
      }
      
      if (aggValidation.type_mismatches.length > 0) {
        errors.push(`Aggregation type mismatches: ${JSON.stringify(aggValidation.type_mismatches)}`);
      }
      
      if (aggValidation.extra_columns.length > 0) {
        warnings.push(`Extra aggregation columns (will be ignored): ${aggValidation.extra_columns.join(', ')}`);
      }
    }

    // Validate hits records  
    const hitsValidation = validateHitsSchema(hitsRecords);
    if (!hitsValidation.valid) {
      if (hitsValidation.missing_columns.length > 0) {
        errors.push(`Missing required hits columns: ${hitsValidation.missing_columns.join(', ')}`);
      }
      
      if (hitsValidation.type_mismatches.length > 0) {
        errors.push(`Hits type mismatches: ${JSON.stringify(hitsValidation.type_mismatches)}`);
      }
      
      if (hitsValidation.extra_columns.length > 0) {
        warnings.push(`Extra hits columns (will be ignored): ${hitsValidation.extra_columns.join(', ')}`);
      }
    }

    // Additional business logic validation
    SchemaGuard.validateBusinessLogic(aggRecords, hitsRecords, errors, warnings);

    const valid = errors.length === 0;
    const attestation_sha256 = valid ? SchemaGuard.generateAttestation(aggRecords, hitsRecords) : undefined;

    return {
      valid,
      attestation_sha256,
      errors,
      warnings
    };
  }

  /**
   * Additional business logic validation beyond basic schema checks
   */
  private static validateBusinessLogic(
    aggRecords: AggregationRecord[],
    hitsRecords: HitsRecord[], 
    errors: string[],
    warnings: string[]
  ): void {
    // Validate query_id consistency
    const aggQueryIds = new Set(aggRecords.map(r => r.query_id));
    const hitsQueryIds = new Set(hitsRecords.map(r => r.query_id));
    
    // Check for orphaned hits (hits without corresponding agg record)
    const orphanedHitsQueryIds = [...hitsQueryIds].filter(id => !aggQueryIds.has(id));
    if (orphanedHitsQueryIds.length > 0) {
      errors.push(`Orphaned hits found for query_ids: ${orphanedHitsQueryIds.join(', ')}`);
    }

    // Check for agg records without hits (allowed but worth warning)
    const aggWithoutHits = [...aggQueryIds].filter(id => !hitsQueryIds.has(id));
    if (aggWithoutHits.length > 0) {
      warnings.push(`Aggregation records without hits: ${aggWithoutHits.length} queries`);
    }

    // Validate timestamp consistency (hits should have same or newer timestamp than agg)
    for (const aggRecord of aggRecords) {
      const relatedHits = hitsRecords.filter(h => h.query_id === aggRecord.query_id);
      
      for (const hit of relatedHits) {
        if (hit.req_ts < aggRecord.req_ts) {
          errors.push(`Hit timestamp (${hit.req_ts}) is earlier than agg timestamp (${aggRecord.req_ts}) for query ${aggRecord.query_id}`);
        }
      }
    }

    // Validate hit counts match
    for (const aggRecord of aggRecords) {
      const relatedHits = hitsRecords.filter(h => h.query_id === aggRecord.query_id);
      
      if (aggRecord.success && relatedHits.length !== aggRecord.total_hits) {
        warnings.push(`Hit count mismatch for query ${aggRecord.query_id}: agg.total_hits=${aggRecord.total_hits}, actual hits=${relatedHits.length}`);
      }
    }

    // Validate why_mix counts
    for (const aggRecord of aggRecords) {
      const relatedHits = hitsRecords.filter(h => h.query_id === aggRecord.query_id);
      const actualWhyCounts = {
        lex: relatedHits.filter(h => h.why === 'lex').length,
        struct: relatedHits.filter(h => h.why === 'struct').length,
        sem: relatedHits.filter(h => h.why === 'sem').length
      };

      if (actualWhyCounts.lex !== aggRecord.why_mix_lex ||
          actualWhyCounts.struct !== aggRecord.why_mix_struct ||
          actualWhyCounts.sem !== aggRecord.why_mix_sem) {
        warnings.push(
          `Why count mismatch for query ${aggRecord.query_id}: ` +
          `expected(${aggRecord.why_mix_lex},${aggRecord.why_mix_struct},${aggRecord.why_mix_sem}) ` +
          `actual(${actualWhyCounts.lex},${actualWhyCounts.struct},${actualWhyCounts.sem})`
        );
      }
    }

    // Validate latency reasonableness
    for (const aggRecord of aggRecords) {
      if (aggRecord.lat_ms < 0) {
        errors.push(`Negative latency for query ${aggRecord.query_id}: ${aggRecord.lat_ms}ms`);
      }
      
      if (aggRecord.lat_ms > 60000) { // 60 seconds seems unreasonable
        warnings.push(`Extremely high latency for query ${aggRecord.query_id}: ${aggRecord.lat_ms}ms`);
      }
      
      // Validate within_sla consistency
      const expectedWithinSla = aggRecord.lat_ms <= 150; // Hardcoded SLA for now
      if (aggRecord.within_sla !== expectedWithinSla) {
        errors.push(`Inconsistent within_sla for query ${aggRecord.query_id}: lat_ms=${aggRecord.lat_ms}, within_sla=${aggRecord.within_sla}`);
      }
    }
  }

  /**
   * Generate cryptographic attestation of the data
   * This creates a tamper-evident hash of the entire dataset
   */
  private static generateAttestation(
    aggRecords: AggregationRecord[],
    hitsRecords: HitsRecord[]
  ): string {
    // Sort records by query_id for deterministic hashing
    const sortedAggRecords = [...aggRecords].sort((a, b) => a.query_id.localeCompare(b.query_id));
    const sortedHitsRecords = [...hitsRecords].sort((a, b) => 
      a.query_id.localeCompare(b.query_id) || a.hit_id.localeCompare(b.hit_id)
    );

    const attestationData = {
      version: '1.0',
      timestamp: Math.floor(Date.now() / 1000),
      records: {
        aggregation: {
          count: sortedAggRecords.length,
          hash: SchemaGuard.hashRecords(sortedAggRecords)
        },
        hits: {
          count: sortedHitsRecords.length, 
          hash: SchemaGuard.hashRecords(sortedHitsRecords)
        }
      },
      metadata: {
        successful_queries: sortedAggRecords.filter(r => r.success).length,
        total_hits: sortedHitsRecords.length,
        avg_latency: sortedAggRecords.reduce((sum, r) => sum + r.lat_ms, 0) / sortedAggRecords.length
      }
    };

    const attestationString = JSON.stringify(attestationData);
    return crypto.createHash('sha256').update(attestationString).digest('hex');
  }

  /**
   * Create a deterministic hash of record array
   */
  private static hashRecords(records: any[]): string {
    const recordsString = JSON.stringify(records);
    return crypto.createHash('sha256').update(recordsString).digest('hex');
  }

  /**
   * Verify an existing attestation against data
   */
  static verifyAttestation(
    aggRecords: AggregationRecord[],
    hitsRecords: HitsRecord[],
    expectedAttestation: string
  ): boolean {
    const actualAttestation = SchemaGuard.generateAttestation(aggRecords, hitsRecords);
    return actualAttestation === expectedAttestation;
  }
}