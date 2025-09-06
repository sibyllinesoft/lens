/**
 * NDJSON formatting utilities for lens
 * Implements Phase A3.2 requirement: stable NDJSON output format
 */

import { SearchHit } from '../types/api.js';

export interface ErrorRecord {
  error: string;
  query?: string;
  repo_sha?: string;
  stage?: string;
  latency_ms?: number;
  timestamp: string;
  details?: Record<string, any>;
}

export interface TraceRecord {
  trace_id: string;
  repo_sha: string;
  query: string;
  stage: string;
  operation: string;
  start_time: string;
  end_time: string;
  duration_ms: number;
  candidates_found?: number;
  metadata?: Record<string, any>;
}

/**
 * Format search results as NDJSON
 * Each result becomes one JSON line
 */
export const formatResultsAsNDJSON = (results: SearchHit[]): string => {
  if (results.length === 0) {
    return '';
  }
  
  return results
    .map(result => JSON.stringify(result))
    .join('\n');
};

/**
 * Format error records as NDJSON
 * Each error becomes one JSON line
 */
export const formatErrorsAsNDJSON = (errors: ErrorRecord[]): string => {
  if (errors.length === 0) {
    return '';
  }
  
  return errors
    .map(error => JSON.stringify(error))
    .join('\n');
};

/**
 * Format trace records as NDJSON
 * Each trace becomes one JSON line
 */
export const formatTracesAsNDJSON = (traces: TraceRecord[]): string => {
  if (traces.length === 0) {
    return '';
  }
  
  return traces
    .map(trace => JSON.stringify(trace))
    .join('\n');
};

/**
 * Generic NDJSON formatter for any array of objects
 * Useful for benchmarking and monitoring data
 */
export const formatAsNDJSON = <T extends Record<string, any>>(items: T[]): string => {
  if (items.length === 0) {
    return '';
  }
  
  return items
    .map(item => JSON.stringify(item))
    .join('\n');
};

/**
 * Parse NDJSON string back into array of objects
 * Useful for reading back NDJSON files
 */
export const parseNDJSON = <T = Record<string, any>>(ndjson: string): T[] => {
  if (!ndjson.trim()) {
    return [];
  }
  
  return ndjson
    .trim()
    .split('\n')
    .map(line => JSON.parse(line) as T);
};

/**
 * Validate NDJSON format
 * Returns true if valid, throws error with details if invalid
 */
export const validateNDJSON = (ndjson: string): boolean => {
  if (!ndjson.trim()) {
    return true; // Empty is valid
  }
  
  const lines = ndjson.trim().split('\n');
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!line || !line.trim()) {
      throw new Error(`NDJSON validation failed: empty line at index ${i}`);
    }
    
    try {
      JSON.parse(line);
    } catch (err) {
      throw new Error(`NDJSON validation failed: invalid JSON at line ${i + 1}: ${err}`);
    }
  }
  
  return true;
};

/**
 * Get NDJSON line count without parsing
 * Efficient for large files
 */
export const getNDJSONLineCount = (ndjson: string): number => {
  if (!ndjson.trim()) {
    return 0;
  }
  
  return ndjson.trim().split('\n').length;
};