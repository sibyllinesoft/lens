/**
 * Output schemas for parquet files
 * Implements canonical record format from TODO.md Section 1
 */

export interface AggregationRecord {
  // Query identification
  query_id: string;
  req_ts: number; // Request timestamp
  cfg_hash: string; // Configuration hash for reproducibility
  
  // Performance metrics  
  shard: string;
  lat_ms: number; // Latency in milliseconds
  within_sla: boolean; // lat_ms <= SLA_MS (150ms default)
  
  // Why mix counts from hit.why analysis
  why_mix_lex: number;
  why_mix_struct: number; 
  why_mix_sem: number;
  
  // Additional metadata
  endpoint_url: string;
  success: boolean;
  error_code?: string;
  total_hits: number;
}

export interface HitsRecord {
  // Query identification  
  query_id: string;
  req_ts: number;
  
  // Hit details
  hit_id: string;
  file_path: string;
  line_start: number;
  line_end: number;
  score: number;
  why: 'lex' | 'struct' | 'sem';
  
  // Span information
  span_start?: number;
  span_end?: number;
  content_preview: string;
  
  // Shard information
  shard_id: string;
  shard_latency_ms: number;
}

export interface SchemaValidationResult {
  valid: boolean;
  missing_columns: string[];
  extra_columns: string[];
  type_mismatches: Array<{
    column: string;
    expected: string;
    actual: string;
  }>;
}

// Required columns for schema validation
export const REQUIRED_AGG_COLUMNS: (keyof AggregationRecord)[] = [
  'query_id',
  'req_ts', 
  'cfg_hash',
  'shard',
  'lat_ms',
  'within_sla',
  'why_mix_lex',
  'why_mix_struct', 
  'why_mix_sem',
  'endpoint_url',
  'success',
  'total_hits'
];

export const REQUIRED_HITS_COLUMNS: (keyof HitsRecord)[] = [
  'query_id',
  'req_ts',
  'hit_id', 
  'file_path',
  'line_start',
  'line_end',
  'score',
  'why',
  'content_preview',
  'shard_id',
  'shard_latency_ms'
];

// Type guards
export function isAggregationRecord(obj: any): obj is AggregationRecord {
  return (
    typeof obj === 'object' &&
    typeof obj.query_id === 'string' &&
    typeof obj.req_ts === 'number' &&
    typeof obj.cfg_hash === 'string' &&
    typeof obj.shard === 'string' &&
    typeof obj.lat_ms === 'number' &&
    typeof obj.within_sla === 'boolean' &&
    typeof obj.why_mix_lex === 'number' &&
    typeof obj.why_mix_struct === 'number' &&
    typeof obj.why_mix_sem === 'number' &&
    typeof obj.endpoint_url === 'string' &&
    typeof obj.success === 'boolean' &&
    typeof obj.total_hits === 'number'
  );
}

export function isHitsRecord(obj: any): obj is HitsRecord {
  return (
    typeof obj === 'object' &&
    typeof obj.query_id === 'string' &&
    typeof obj.req_ts === 'number' &&
    typeof obj.hit_id === 'string' &&
    typeof obj.file_path === 'string' &&
    typeof obj.line_start === 'number' &&
    typeof obj.line_end === 'number' &&
    typeof obj.score === 'number' &&
    ['lex', 'struct', 'sem'].includes(obj.why) &&
    typeof obj.content_preview === 'string' &&
    typeof obj.shard_id === 'string' &&
    typeof obj.shard_latency_ms === 'number'
  );
}

// Schema validation functions
export function validateAggregationSchema(records: any[]): SchemaValidationResult {
  if (records.length === 0) {
    return { valid: true, missing_columns: [], extra_columns: [], type_mismatches: [] };
  }
  
  const sample = records[0];
  const actualColumns = Object.keys(sample);
  
  const missing_columns = REQUIRED_AGG_COLUMNS.filter(col => !actualColumns.includes(col));
  const extra_columns = actualColumns.filter(col => !REQUIRED_AGG_COLUMNS.includes(col as keyof AggregationRecord));
  
  const type_mismatches: Array<{column: string; expected: string; actual: string}> = [];
  
  // Basic type checking on first record
  if (!isAggregationRecord(sample)) {
    // More detailed type validation would go here
    type_mismatches.push({
      column: 'overall',
      expected: 'AggregationRecord',
      actual: typeof sample
    });
  }
  
  return {
    valid: missing_columns.length === 0 && type_mismatches.length === 0,
    missing_columns,
    extra_columns,
    type_mismatches
  };
}

export function validateHitsSchema(records: any[]): SchemaValidationResult {
  if (records.length === 0) {
    return { valid: true, missing_columns: [], extra_columns: [], type_mismatches: [] };
  }
  
  const sample = records[0];
  const actualColumns = Object.keys(sample);
  
  const missing_columns = REQUIRED_HITS_COLUMNS.filter(col => !actualColumns.includes(col));
  const extra_columns = actualColumns.filter(col => !REQUIRED_HITS_COLUMNS.includes(col as keyof HitsRecord));
  
  const type_mismatches: Array<{column: string; expected: string; actual: string}> = [];
  
  if (!isHitsRecord(sample)) {
    type_mismatches.push({
      column: 'overall', 
      expected: 'HitsRecord',
      actual: typeof sample
    });
  }
  
  return {
    valid: missing_columns.length === 0 && type_mismatches.length === 0,
    missing_columns,
    extra_columns, 
    type_mismatches
  };
}