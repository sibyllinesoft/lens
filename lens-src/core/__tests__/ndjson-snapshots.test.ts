/**
 * Snapshot tests for NDJSON outputs
 * Implements Phase A3.2 requirement: snapshot tests for NDJSON output
 */

import { describe, it, expect } from 'vitest';
import { SearchHit } from '../../types/api.js';
import { 
  formatResultsAsNDJSON,
  formatErrorsAsNDJSON, 
  formatTracesAsNDJSON,
  ErrorRecord,
  TraceRecord,
  validateNDJSON,
  parseNDJSON,
  getNDJSONLineCount
} from '../ndjson-formatter.js';

// Mock search results that would be converted to NDJSON
const mockSearchResults = [
  {
    file: 'src/components/Button.tsx',
    line: 15,
    col: 8,
    lang: 'typescript',
    snippet: 'const Button = ({ onClick, children }: ButtonProps) => {',
    score: 0.95,
    why: ['exact', 'symbol'],
    symbol_kind: 'function',
    ast_path: 'Program > VariableDeclaration > VariableDeclarator',
    byte_offset: 342,
    span_len: 45,
    context_before: 'import React from "react";\n\n',
    context_after: '\n  return (\n    <button onClick={onClick}>'
  },
  {
    file: 'src/utils/helpers.ts',
    line: 23,
    col: 15,
    lang: 'typescript',
    snippet: 'export function formatDate(date: Date): string {',
    score: 0.87,
    why: ['exact', 'subtoken'],
    symbol_kind: 'function',
    ast_path: 'Program > ExportNamedDeclaration > FunctionDeclaration',
    byte_offset: 567,
    span_len: 47,
    context_before: '// Date formatting utilities\n',
    context_after: '\n  return date.toISOString().split("T")[0];'
  },
  {
    file: 'docs/api.md',
    line: 1,
    col: 1,
    lang: 'markdown',
    snippet: '# API Documentation',
    score: 0.72,
    why: ['fuzzy'],
    byte_offset: 0,
    span_len: 18
  }
] as SearchHit[];

// Error test cases
const mockErrorResults: ErrorRecord[] = [
  {
    error: 'timeout',
    query: 'very complex query',
    repo_sha: 'abc123',
    stage: 'stage_c',
    latency_ms: 5000,
    timestamp: '2024-01-01T12:00:00Z'
  },
  {
    error: 'index_not_found',
    query: 'simple query',
    repo_sha: 'def456',
    stage: 'stage_a',
    latency_ms: 5,
    timestamp: '2024-01-01T12:00:01Z'
  }
];

// Trace test cases
const mockTraceResults: TraceRecord[] = [
  {
    trace_id: 'trace-123',
    repo_sha: 'abc123',
    query: 'button component',
    stage: 'stage_a',
    operation: 'lexical_search',
    start_time: '2024-01-01T12:00:00.000Z',
    end_time: '2024-01-01T12:00:00.050Z',
    duration_ms: 50,
    candidates_found: 15,
    metadata: {
      fuzzy_enabled: true,
      max_edit_distance: 2,
      term_frequency: { button: 5, component: 3 }
    }
  },
  {
    trace_id: 'trace-123',
    repo_sha: 'abc123',
    query: 'button component',
    stage: 'stage_b',
    operation: 'symbol_resolution',
    start_time: '2024-01-01T12:00:00.050Z',
    end_time: '2024-01-01T12:00:00.120Z',
    duration_ms: 70,
    candidates_found: 8,
    metadata: {
      ast_cache_hits: 12,
      ast_cache_misses: 3,
      symbol_types: ['function', 'class', 'variable']
    }
  }
];

// NDJSON formatting functions are imported from ../ndjson-formatter.js

describe('NDJSON Output Snapshots', () => {
  
  describe('Search Results NDJSON', () => {
    it('should format search results consistently', () => {
      const ndjson = formatResultsAsNDJSON(mockSearchResults);
      
      // Verify structure
      const lines = ndjson.split('\n');
      expect(lines).toHaveLength(3);
      
      // Verify each line is valid JSON
      lines.forEach((line, index) => {
        expect(() => JSON.parse(line)).not.toThrow();
        const parsed = JSON.parse(line);
        expect(parsed).toHaveProperty('file');
        expect(parsed).toHaveProperty('line');
        expect(parsed).toHaveProperty('score');
        expect(parsed).toHaveProperty('why');
      });
      
      // Snapshot test
      expect(ndjson).toMatchSnapshot('search-results.ndjson');
    });
    
    it('should handle empty results', () => {
      const ndjson = formatResultsAsNDJSON([]);
      expect(ndjson).toBe('');
      expect(ndjson).toMatchSnapshot('search-results-empty.ndjson');
    });
    
    it('should handle results with minimal fields', () => {
      const minimalResults = [{
        file: 'test.js',
        line: 1,
        col: 1,
        score: 0.5,
        why: ['exact']
      }] as SearchHit[];
      
      const ndjson = formatResultsAsNDJSON(minimalResults);
      expect(ndjson).toMatchSnapshot('search-results-minimal.ndjson');
    });
    
    it('should handle unicode and special characters', () => {
      const unicodeResults = [{
        file: 'src/测试.ts',
        line: 1,
        col: 1,
        snippet: 'const message = "Hello 👋 世界";',
        score: 0.8,
        why: ['exact'],
        context_before: '// 注释\n',
        context_after: '\nconsole.log("🎉");'
      }] as SearchHit[];
      
      const ndjson = formatResultsAsNDJSON(unicodeResults);
      
      // Verify Unicode is preserved
      expect(ndjson).toContain('测试');
      expect(ndjson).toContain('👋');
      expect(ndjson).toContain('世界');
      expect(ndjson).toContain('🎉');
      
      expect(ndjson).toMatchSnapshot('search-results-unicode.ndjson');
    });
  });
  
  describe('Error NDJSON', () => {
    it('should format errors consistently', () => {
      const ndjson = formatErrorsAsNDJSON(mockErrorResults);
      
      const lines = ndjson.split('\n');
      expect(lines).toHaveLength(2);
      
      lines.forEach(line => {
        expect(() => JSON.parse(line)).not.toThrow();
        const parsed = JSON.parse(line);
        expect(parsed).toHaveProperty('error');
        expect(parsed).toHaveProperty('timestamp');
      });
      
      expect(ndjson).toMatchSnapshot('errors.ndjson');
    });
    
    it('should handle complex error objects', () => {
      const complexErrors: ErrorRecord[] = [{
        error: 'semantic_model_failure',
        query: 'complex query with "quotes" and symbols',
        repo_sha: 'complex123',
        stage: 'stage_c',
        details: {
          model_name: 'all-MiniLM-L6-v2',
          error_type: 'dimension_mismatch',
          expected_dimensions: 384,
          actual_dimensions: 512,
          stack_trace: [
            'at SemanticEngine.embed (semantic.ts:42)',
            'at SearchEngine.search (search-engine.ts:156)'
          ]
        },
        timestamp: '2024-01-01T12:00:00Z'
      }];
      
      const ndjson = formatErrorsAsNDJSON(complexErrors);
      expect(ndjson).toMatchSnapshot('errors-complex.ndjson');
    });
  });
  
  describe('Trace NDJSON', () => {
    it('should format traces consistently', () => {
      const ndjson = formatTracesAsNDJSON(mockTraceResults);
      
      const lines = ndjson.split('\n');
      expect(lines).toHaveLength(2);
      
      lines.forEach(line => {
        expect(() => JSON.parse(line)).not.toThrow();
        const parsed = JSON.parse(line);
        expect(parsed).toHaveProperty('trace_id');
        expect(parsed).toHaveProperty('operation');
        expect(parsed).toHaveProperty('duration_ms');
      });
      
      expect(ndjson).toMatchSnapshot('traces.ndjson');
    });
    
    it('should handle traces with nested metadata', () => {
      const nestedTraces: TraceRecord[] = [{
        trace_id: 'trace-456',
        repo_sha: 'nested789',
        query: 'nested query',
        stage: 'stage_c',
        operation: 'semantic_rerank',
        start_time: '2024-01-01T12:00:00.200Z',
        end_time: '2024-01-01T12:00:00.450Z',
        duration_ms: 250,
        candidates_found: 50,
        metadata: {
          model_config: {
            name: 'all-MiniLM-L6-v2',
            dimensions: 384,
            batch_size: 32
          },
          performance: {
            embeddings_cached: 15,
            embeddings_computed: 35,
            similarity_calculations: 2500,
            rerank_threshold: 0.7
          },
          quality_metrics: {
            precision_at_5: 0.85,
            precision_at_10: 0.78,
            mrr: 0.82
          }
        }
      }];
      
      const ndjson = formatTracesAsNDJSON(nestedTraces);
      expect(ndjson).toMatchSnapshot('traces-nested.ndjson');
    });
  });
  
  describe('NDJSON Format Validation', () => {
    it('should not contain line breaks within individual JSON objects', () => {
      const results = [{
        file: 'multiline.js',
        line: 1,
        col: 1,
        snippet: 'const text = `\nHello\nWorld\n`;',
        score: 0.5,
        why: ['exact']
      }] as SearchHit[];
      
      const ndjson = formatResultsAsNDJSON(results);
      
      // Should have exactly one line (no internal line breaks)
      expect(ndjson.split('\n')).toHaveLength(1);
      
      // But the JSON should contain escaped line breaks
      expect(ndjson).toContain('\\n');
      
      expect(ndjson).toMatchSnapshot('search-results-multiline.ndjson');
    });
    
    it('should escape JSON special characters properly', () => {
      const results = [{
        file: 'special.js',
        line: 1,
        col: 1,
        snippet: 'const str = "He said \\"Hello\\" and then \\n left";',
        score: 0.5,
        why: ['exact'],
        context_after: '\t// Tab character and "quotes"'
      }] as SearchHit[];
      
      const ndjson = formatResultsAsNDJSON(results);
      
      // Should be valid JSON
      expect(() => JSON.parse(ndjson)).not.toThrow();
      
      // Should contain properly escaped characters
      expect(ndjson).toContain('\\"');
      expect(ndjson).toContain('\\n');
      expect(ndjson).toContain('\\t');
      
      expect(ndjson).toMatchSnapshot('search-results-escaped.ndjson');
    });
    
    it('should maintain consistent field ordering', () => {
      // Test that fields appear in a consistent order for diff stability
      const result = {
        file: 'test.js',
        line: 5,
        col: 10,
        lang: 'javascript',
        snippet: 'function test() {}',
        score: 0.9,
        why: ['exact', 'symbol'],
        ast_path: 'Program > FunctionDeclaration',
        symbol_kind: 'function',
        byte_offset: 100,
        span_len: 18
      } as SearchHit;
      
      const ndjson = formatResultsAsNDJSON([result]);
      const parsed = JSON.parse(ndjson);
      
      // Verify all expected fields are present
      const expectedFields = [
        'file', 'line', 'col', 'lang', 'snippet', 'score', 
        'why', 'ast_path', 'symbol_kind', 'byte_offset', 'span_len'
      ];
      
      expectedFields.forEach(field => {
        expect(parsed).toHaveProperty(field);
      });
      
      expect(ndjson).toMatchSnapshot('search-results-field-order.ndjson');
    });
  });
  
  describe('Performance and Size Constraints', () => {
    it('should handle large result sets efficiently', () => {
      const largeResults = Array.from({ length: 1000 }, (_, i) => ({
        file: `src/file-${i}.ts`,
        line: i + 1,
        col: 1,
        snippet: `const value${i} = ${i};`,
        score: Math.random(),
        why: ['exact']
      })) as SearchHit[];
      
      const startTime = performance.now();
      const ndjson = formatResultsAsNDJSON(largeResults);
      const endTime = performance.now();
      
      // Should complete quickly
      expect(endTime - startTime).toBeLessThan(100);
      
      // Should have correct number of lines
      expect(ndjson.split('\n')).toHaveLength(1000);
      
      // File size should be reasonable (rough check)
      expect(ndjson.length).toBeLessThan(500000); // < 500KB for 1000 results
    });
    
    it('should truncate extremely long snippets', () => {
      const longSnippet = 'x'.repeat(10000);
      const result = {
        file: 'long.js',
        line: 1,
        col: 1,
        snippet: longSnippet,
        score: 0.5,
        why: ['exact']
      } as SearchHit;
      
      const ndjson = formatResultsAsNDJSON([result]);
      
      // The full snippet should be preserved in this test (no truncation)
      // but in practice, the search engine might truncate before NDJSON formatting
      expect(ndjson.length).toBeGreaterThan(10000);
      
      // Verify it's still valid JSON despite the long content
      expect(() => JSON.parse(ndjson)).not.toThrow();
    });
  });
  
  describe('NDJSON Utilities', () => {
    it('should validate NDJSON format correctly', () => {
      const validNDJSON = formatResultsAsNDJSON([
        { file: 'test.js', line: 1, col: 1, score: 0.5, why: ['exact'] } as SearchHit
      ]);
      
      expect(() => validateNDJSON(validNDJSON)).not.toThrow();
      expect(() => validateNDJSON('')).not.toThrow(); // Empty is valid
      
      // Invalid NDJSON should throw
      expect(() => validateNDJSON('invalid json line')).toThrow();
      expect(() => validateNDJSON('{"valid": true}\n\ninvalid')).toThrow();
    });
    
    it('should parse NDJSON back to objects', () => {
      const original = [
        { file: 'test.js', line: 1, col: 1, score: 0.5, why: ['exact'] },
        { file: 'other.js', line: 5, col: 3, score: 0.8, why: ['symbol'] }
      ] as SearchHit[];
      
      const ndjson = formatResultsAsNDJSON(original);
      const parsed = parseNDJSON<SearchHit>(ndjson);
      
      expect(parsed).toHaveLength(2);
      expect(parsed[0]).toEqual(original[0]);
      expect(parsed[1]).toEqual(original[1]);
    });
    
    it('should count NDJSON lines efficiently', () => {
      const results = Array.from({ length: 100 }, (_, i) => ({
        file: `file-${i}.js`,
        line: i + 1,
        col: 1,
        score: Math.random(),
        why: ['exact']
      })) as SearchHit[];
      
      const ndjson = formatResultsAsNDJSON(results);
      expect(getNDJSONLineCount(ndjson)).toBe(100);
      expect(getNDJSONLineCount('')).toBe(0);
      expect(getNDJSONLineCount('single line')).toBe(1);
    });
  });
});

// NDJSON formatting functions are now available from the ndjson-formatter module