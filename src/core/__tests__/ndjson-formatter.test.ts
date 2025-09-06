/**
 * Tests for NDJSON Formatter
 * Covers all NDJSON formatting and parsing utilities
 */

import { describe, it, expect } from 'vitest';
import {
  formatResultsAsNDJSON,
  formatErrorsAsNDJSON,
  formatTracesAsNDJSON,
  formatAsNDJSON,
  parseNDJSON,
  validateNDJSON,
  getNDJSONLineCount,
  type ErrorRecord,
  type TraceRecord,
} from '../ndjson-formatter.js';
import type { SearchHit } from '../../types/api.js';

describe('NDJSON Formatter', () => {
  describe('formatResultsAsNDJSON', () => {
    it('should format empty results as empty string', () => {
      const result = formatResultsAsNDJSON([]);
      expect(result).toBe('');
    });

    it('should format single result correctly', () => {
      const results: SearchHit[] = [
        {
          file: 'test.ts',
          line: 1,
          column: 0,
          score: 0.95,
          reason: 'exact_match',
          query: 'test',
          snippet: 'test code',
        },
      ];

      const result = formatResultsAsNDJSON(results);
      const expected = JSON.stringify(results[0]);
      
      expect(result).toBe(expected);
    });

    it('should format multiple results with newlines', () => {
      const results: SearchHit[] = [
        {
          file: 'test1.ts',
          line: 1,
          column: 0,
          score: 0.95,
          reason: 'exact_match',
          query: 'test',
          snippet: 'test code 1',
        },
        {
          file: 'test2.ts',
          line: 2,
          column: 5,
          score: 0.85,
          reason: 'fuzzy_match',
          query: 'test',
          snippet: 'test code 2',
        },
      ];

      const result = formatResultsAsNDJSON(results);
      const lines = result.split('\n');
      
      expect(lines).toHaveLength(2);
      expect(lines[0]).toBe(JSON.stringify(results[0]));
      expect(lines[1]).toBe(JSON.stringify(results[1]));
    });

    it('should handle results with special characters', () => {
      const results: SearchHit[] = [
        {
          file: 'test with spaces.ts',
          line: 1,
          column: 0,
          score: 0.95,
          reason: 'exact_match',
          query: 'test "quoted" text',
          snippet: 'test with\nnewlines and "quotes"',
        },
      ];

      const result = formatResultsAsNDJSON(results);
      expect(() => JSON.parse(result)).not.toThrow();
    });
  });

  describe('formatErrorsAsNDJSON', () => {
    it('should format empty errors as empty string', () => {
      const result = formatErrorsAsNDJSON([]);
      expect(result).toBe('');
    });

    it('should format single error correctly', () => {
      const errors: ErrorRecord[] = [
        {
          error: 'Test error',
          query: 'test query',
          stage: 'lexical',
          timestamp: '2023-01-01T00:00:00Z',
        },
      ];

      const result = formatErrorsAsNDJSON(errors);
      const expected = JSON.stringify(errors[0]);
      
      expect(result).toBe(expected);
    });

    it('should format multiple errors with newlines', () => {
      const errors: ErrorRecord[] = [
        {
          error: 'First error',
          timestamp: '2023-01-01T00:00:00Z',
        },
        {
          error: 'Second error',
          query: 'failing query',
          stage: 'semantic',
          latency_ms: 150,
          timestamp: '2023-01-01T00:01:00Z',
          details: { code: 500, message: 'Internal error' },
        },
      ];

      const result = formatErrorsAsNDJSON(errors);
      const lines = result.split('\n');
      
      expect(lines).toHaveLength(2);
      expect(lines[0]).toBe(JSON.stringify(errors[0]));
      expect(lines[1]).toBe(JSON.stringify(errors[1]));
    });

    it('should handle error records with optional fields', () => {
      const errors: ErrorRecord[] = [
        {
          error: 'Error with minimal fields',
          timestamp: '2023-01-01T00:00:00Z',
        },
        {
          error: 'Error with all fields',
          query: 'test query',
          repo_sha: 'abc123',
          stage: 'lexical',
          latency_ms: 100,
          timestamp: '2023-01-01T00:01:00Z',
          details: { 
            error_code: 'TIMEOUT',
            retry_count: 3,
            metadata: { user_id: 'test' }
          },
        },
      ];

      const result = formatErrorsAsNDJSON(errors);
      expect(() => parseNDJSON(result)).not.toThrow();
      
      const parsed = parseNDJSON<ErrorRecord>(result);
      expect(parsed).toHaveLength(2);
      expect(parsed[0].error).toBe('Error with minimal fields');
      expect(parsed[1].details?.error_code).toBe('TIMEOUT');
    });
  });

  describe('formatTracesAsNDJSON', () => {
    it('should format empty traces as empty string', () => {
      const result = formatTracesAsNDJSON([]);
      expect(result).toBe('');
    });

    it('should format single trace correctly', () => {
      const traces: TraceRecord[] = [
        {
          trace_id: 'trace-123',
          repo_sha: 'abc123',
          query: 'test query',
          stage: 'lexical',
          operation: 'search',
          start_time: '2023-01-01T00:00:00Z',
          end_time: '2023-01-01T00:00:01Z',
          duration_ms: 1000,
        },
      ];

      const result = formatTracesAsNDJSON(traces);
      const expected = JSON.stringify(traces[0]);
      
      expect(result).toBe(expected);
    });

    it('should format multiple traces with optional fields', () => {
      const traces: TraceRecord[] = [
        {
          trace_id: 'trace-1',
          repo_sha: 'abc123',
          query: 'first query',
          stage: 'lexical',
          operation: 'search',
          start_time: '2023-01-01T00:00:00Z',
          end_time: '2023-01-01T00:00:01Z',
          duration_ms: 1000,
          candidates_found: 10,
        },
        {
          trace_id: 'trace-2',
          repo_sha: 'def456',
          query: 'second query',
          stage: 'semantic',
          operation: 'rerank',
          start_time: '2023-01-01T00:01:00Z',
          end_time: '2023-01-01T00:01:02Z',
          duration_ms: 2000,
          candidates_found: 25,
          metadata: {
            model: 'bert-base',
            confidence_threshold: 0.7,
            batch_size: 16,
          },
        },
      ];

      const result = formatTracesAsNDJSON(traces);
      const lines = result.split('\n');
      
      expect(lines).toHaveLength(2);
      expect(lines[0]).toBe(JSON.stringify(traces[0]));
      expect(lines[1]).toBe(JSON.stringify(traces[1]));
      
      const parsed = parseNDJSON<TraceRecord>(result);
      expect(parsed[1].metadata?.model).toBe('bert-base');
    });
  });

  describe('formatAsNDJSON (Generic)', () => {
    it('should format empty array as empty string', () => {
      const result = formatAsNDJSON([]);
      expect(result).toBe('');
    });

    it('should format array of simple objects', () => {
      const items = [
        { id: 1, name: 'first' },
        { id: 2, name: 'second' },
      ];

      const result = formatAsNDJSON(items);
      const lines = result.split('\n');
      
      expect(lines).toHaveLength(2);
      expect(lines[0]).toBe(JSON.stringify(items[0]));
      expect(lines[1]).toBe(JSON.stringify(items[1]));
    });

    it('should format array of complex objects', () => {
      const items = [
        {
          id: 1,
          data: {
            nested: true,
            values: [1, 2, 3],
            metadata: { created: '2023-01-01' },
          },
        },
        {
          id: 2,
          data: {
            nested: false,
            values: [],
            metadata: null,
          },
        },
      ];

      const result = formatAsNDJSON(items);
      expect(() => parseNDJSON(result)).not.toThrow();
      
      const parsed = parseNDJSON(result);
      expect(parsed).toHaveLength(2);
      expect(parsed[0].data.nested).toBe(true);
      expect(parsed[1].data.metadata).toBe(null);
    });

    it('should handle objects with various data types', () => {
      const items = [
        {
          string: 'text',
          number: 42,
          boolean: true,
          null_value: null,
          undefined_value: undefined, // Will be omitted in JSON
          array: [1, 'two', true],
          object: { nested: 'value' },
          date: new Date('2023-01-01T00:00:00Z').toISOString(),
        },
      ];

      const result = formatAsNDJSON(items);
      const parsed = parseNDJSON(result);
      
      expect(parsed[0].string).toBe('text');
      expect(parsed[0].number).toBe(42);
      expect(parsed[0].boolean).toBe(true);
      expect(parsed[0].null_value).toBe(null);
      expect(parsed[0].undefined_value).toBeUndefined();
      expect(parsed[0].array).toEqual([1, 'two', true]);
      expect(parsed[0].object).toEqual({ nested: 'value' });
    });
  });

  describe('parseNDJSON', () => {
    it('should parse empty string to empty array', () => {
      const result = parseNDJSON('');
      expect(result).toEqual([]);
    });

    it('should parse whitespace-only string to empty array', () => {
      const result = parseNDJSON('   \n\t  ');
      expect(result).toEqual([]);
    });

    it('should parse single JSON line', () => {
      const input = '{"id": 1, "name": "test"}';
      const result = parseNDJSON(input);
      
      expect(result).toHaveLength(1);
      expect(result[0]).toEqual({ id: 1, name: 'test' });
    });

    it('should parse multiple JSON lines', () => {
      const input = '{"id": 1}\n{"id": 2}\n{"id": 3}';
      const result = parseNDJSON(input);
      
      expect(result).toHaveLength(3);
      expect(result[0]).toEqual({ id: 1 });
      expect(result[1]).toEqual({ id: 2 });
      expect(result[2]).toEqual({ id: 3 });
    });

    it('should parse with generic type parameter', () => {
      interface TestItem {
        id: number;
        name: string;
      }
      
      const input = '{"id": 1, "name": "first"}\n{"id": 2, "name": "second"}';
      const result = parseNDJSON<TestItem>(input);
      
      expect(result).toHaveLength(2);
      expect(result[0].id).toBe(1);
      expect(result[0].name).toBe('first');
      expect(result[1].id).toBe(2);
      expect(result[1].name).toBe('second');
    });

    it('should handle complex nested objects', () => {
      const input = '{"data": {"nested": true, "values": [1, 2]}}';
      const result = parseNDJSON(input);
      
      expect(result[0].data.nested).toBe(true);
      expect(result[0].data.values).toEqual([1, 2]);
    });

    it('should throw error for malformed JSON', () => {
      const input = '{"valid": true}\n{invalid json}\n{"also_valid": true}';
      
      expect(() => parseNDJSON(input)).toThrow('JSON');
    });

    it('should handle trailing newlines correctly', () => {
      const input = '{"id": 1}\n{"id": 2}\n';
      const result = parseNDJSON(input);
      
      expect(result).toHaveLength(2);
    });
  });

  describe('validateNDJSON', () => {
    it('should validate empty string as true', () => {
      expect(validateNDJSON('')).toBe(true);
    });

    it('should validate whitespace-only string as true', () => {
      expect(validateNDJSON('   \n\t  ')).toBe(true);
    });

    it('should validate single valid JSON line', () => {
      const input = '{"valid": true}';
      expect(validateNDJSON(input)).toBe(true);
    });

    it('should validate multiple valid JSON lines', () => {
      const input = '{"id": 1}\n{"id": 2}\n{"id": 3}';
      expect(validateNDJSON(input)).toBe(true);
    });

    it('should throw error for invalid JSON line', () => {
      const input = '{"valid": true}\n{invalid json}';
      
      expect(() => validateNDJSON(input)).toThrow('NDJSON validation failed: invalid JSON at line 2');
    });

    it('should throw error for empty line in middle', () => {
      const input = '{"id": 1}\n\n{"id": 2}';
      
      expect(() => validateNDJSON(input)).toThrow('NDJSON validation failed: empty line at index 1');
    });

    it('should throw error for whitespace-only line', () => {
      const input = '{"id": 1}\n   \n{"id": 2}';
      
      expect(() => validateNDJSON(input)).toThrow('NDJSON validation failed: empty line at index 1');
    });

    it('should provide detailed error messages with line numbers', () => {
      const input = '{"valid": true}\n{"also": "valid"}\n{"invalid": syntax}';
      
      expect(() => validateNDJSON(input)).toThrow(/line 3/);
    });

    it('should handle complex JSON validation', () => {
      const input = '{"complex": {"nested": true, "array": [1, 2, 3]}}';
      expect(validateNDJSON(input)).toBe(true);
    });

    it('should validate real-world NDJSON examples', () => {
      const errorRecord = JSON.stringify({
        error: 'Test error',
        query: 'search term',
        timestamp: '2023-01-01T00:00:00Z',
      });
      
      const traceRecord = JSON.stringify({
        trace_id: 'trace-123',
        operation: 'search',
        duration_ms: 150,
      });
      
      const input = `${errorRecord}\n${traceRecord}`;
      expect(validateNDJSON(input)).toBe(true);
    });
  });

  describe('getNDJSONLineCount', () => {
    it('should return 0 for empty string', () => {
      expect(getNDJSONLineCount('')).toBe(0);
    });

    it('should return 0 for whitespace-only string', () => {
      expect(getNDJSONLineCount('   \n\t  ')).toBe(0);
    });

    it('should return 1 for single line', () => {
      expect(getNDJSONLineCount('{"single": "line"}')).toBe(1);
    });

    it('should return correct count for multiple lines', () => {
      const input = '{"line": 1}\n{"line": 2}\n{"line": 3}';
      expect(getNDJSONLineCount(input)).toBe(3);
    });

    it('should handle trailing newlines correctly', () => {
      const input = '{"line": 1}\n{"line": 2}\n';
      expect(getNDJSONLineCount(input)).toBe(2);
    });

    it('should count lines without parsing JSON', () => {
      // This should still count lines even if JSON is invalid
      const input = 'invalid json\nsecond invalid line\nthird line';
      expect(getNDJSONLineCount(input)).toBe(3);
    });

    it('should handle very large line counts efficiently', () => {
      // Generate large NDJSON string without actually parsing
      const lines = Array.from({ length: 1000 }, (_, i) => `{"id": ${i}}`);
      const input = lines.join('\n');
      
      expect(getNDJSONLineCount(input)).toBe(1000);
    });
  });

  describe('Round-trip Compatibility', () => {
    it('should maintain data integrity through format/parse cycle', () => {
      const originalData = [
        { id: 1, name: 'first', active: true },
        { id: 2, name: 'second', active: false },
        { id: 3, name: 'third', metadata: { count: 10, tags: ['a', 'b'] } },
      ];

      const ndjson = formatAsNDJSON(originalData);
      const parsed = parseNDJSON(ndjson);
      
      expect(parsed).toEqual(originalData);
    });

    it('should handle format/validate/parse cycle', () => {
      const results: SearchHit[] = [
        {
          file: 'test.ts',
          line: 1,
          column: 0,
          score: 0.95,
          reason: 'exact_match',
          query: 'test',
          snippet: 'test code',
        },
      ];

      const ndjson = formatResultsAsNDJSON(results);
      expect(validateNDJSON(ndjson)).toBe(true);
      
      const parsed = parseNDJSON<SearchHit>(ndjson);
      expect(parsed).toEqual(results);
    });

    it('should preserve type information through generic formatting', () => {
      interface CustomType {
        id: number;
        timestamp: string;
        data: {
          values: number[];
          flags: boolean[];
        };
      }

      const items: CustomType[] = [
        {
          id: 1,
          timestamp: '2023-01-01T00:00:00Z',
          data: {
            values: [1, 2, 3],
            flags: [true, false, true],
          },
        },
      ];

      const ndjson = formatAsNDJSON(items);
      const parsed = parseNDJSON<CustomType>(ndjson);
      
      expect(parsed[0].id).toBe(1);
      expect(parsed[0].data.values).toEqual([1, 2, 3]);
      expect(parsed[0].data.flags).toEqual([true, false, true]);
    });
  });

  describe('Edge Cases and Error Handling', () => {
    it('should handle objects with circular references', () => {
      const obj: any = { id: 1 };
      obj.self = obj; // Create circular reference
      
      expect(() => formatAsNDJSON([obj])).toThrow('Converting circular structure to JSON');
    });

    it('should handle very large objects', () => {
      const largeObj = {
        id: 1,
        data: Array.from({ length: 1000 }, (_, i) => ({ index: i, value: `item-${i}` })),
      };

      const ndjson = formatAsNDJSON([largeObj]);
      expect(getNDJSONLineCount(ndjson)).toBe(1);
      
      const parsed = parseNDJSON(ndjson);
      expect(parsed[0].data).toHaveLength(1000);
    });

    it('should handle special string values', () => {
      const items = [
        { text: 'normal string' },
        { text: 'string with "quotes"' },
        { text: 'string with\nnewlines' },
        { text: 'string with\ttabs' },
        { text: 'string with unicode: 🚀 ñ 中文' },
        { text: 'string with backslashes: \\n \\t \\\\' },
      ];

      const ndjson = formatAsNDJSON(items);
      expect(validateNDJSON(ndjson)).toBe(true);
      
      const parsed = parseNDJSON(ndjson);
      expect(parsed).toEqual(items);
    });

    it('should handle objects with undefined, null, and empty values', () => {
      const items = [
        { defined: 'value', undefined: undefined, null: null, empty: '' },
        { array: [], object: {} },
        { zero: 0, false: false },
      ];

      const ndjson = formatAsNDJSON(items);
      const parsed = parseNDJSON(ndjson);
      
      // undefined should be omitted, others preserved
      expect(parsed[0]).not.toHaveProperty('undefined');
      expect(parsed[0].null).toBe(null);
      expect(parsed[0].empty).toBe('');
      expect(parsed[1].array).toEqual([]);
      expect(parsed[1].object).toEqual({});
      expect(parsed[2].zero).toBe(0);
      expect(parsed[2].false).toBe(false);
    });
  });
});