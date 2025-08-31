/**
 * Unit tests for Lexical Search Engine
 * Tests trigram indexing, fuzzy search, and subtoken handling
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { LexicalSearchEngine } from '../../src/indexer/lexical.js';
import { SegmentStorage } from '../../src/storage/segments.js';
import type { SearchContext } from '../../src/types/core.js';

describe('LexicalSearchEngine', () => {
  let engine: LexicalSearchEngine;
  let segmentStorage: SegmentStorage;

  beforeEach(async () => {
    segmentStorage = new SegmentStorage('./test-segments');
    engine = new LexicalSearchEngine(segmentStorage);
  });

  afterEach(async () => {
    engine.clear();
    await segmentStorage.shutdown();
  });

  describe('Document Indexing', () => {
    it('should index a simple document', async () => {
      const content = `
function calculateSum(a, b) {
  return a + b;
}

const myVariable = 42;
class TestClass {
  methodName() {
    return 'test';
  }
}
      `;

      await engine.indexDocument('doc1', '/test/file.js', content);
      
      const stats = engine.getStats();
      expect(stats.document_count).toBe(1);
      expect(stats.trigram_count).toBeGreaterThan(0);
      expect(stats.total_positions).toBeGreaterThan(0);
    });

    it('should handle camelCase tokenization', async () => {
      const content = 'calculateSum myVariable TestClass methodName';
      
      await engine.indexDocument('doc1', '/test/camel.js', content);
      
      // Search for subtokens
      const ctx: SearchContext = {
        trace_id: 'test-trace-1',
        query: 'calculate',
        mode: 'lex',
        k: 10,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      };

      const results = await engine.search(ctx, 'calculate', 0);
      expect(results.length).toBeGreaterThan(0);
    });

    it('should handle snake_case tokenization', async () => {
      const content = 'my_variable test_function snake_case_name';
      
      await engine.indexDocument('doc1', '/test/snake.py', content);
      
      const ctx: SearchContext = {
        trace_id: 'test-trace-2',
        query: 'variable',
        mode: 'lex',
        k: 10,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      };

      const results = await engine.search(ctx, 'variable', 0);
      expect(results.length).toBeGreaterThan(0);
    });
  });

  describe('Exact Search', () => {
    beforeEach(async () => {
      const content = `
function calculateSum(a, b) {
  return a + b;
}

function calculateProduct(x, y) {
  return x * y;
}

const myVariable = 42;
const anotherVar = 'test';
      `;

      await engine.indexDocument('doc1', '/test/math.js', content);
    });

    it('should find exact matches', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-trace-3',
        query: 'function',
        mode: 'lex',
        k: 10,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      };

      const results = await engine.search(ctx, 'function', 0);
      expect(results.length).toBeGreaterThan(0);
      
      // All results should have exact match reason
      results.forEach(result => {
        expect(result.match_reasons).toContain('exact');
        expect(result.score).toBe(1.0);
      });
    });

    it('should handle multi-word queries', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-trace-4',
        query: 'calculateSum',
        mode: 'lex',
        k: 10,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      };

      const results = await engine.search(ctx, 'calculateSum', 0);
      expect(results.length).toBeGreaterThan(0);
    });

    it('should limit results to k parameter', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-trace-5',
        query: 'var',
        mode: 'lex',
        k: 1,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      };

      const results = await engine.search(ctx, 'var', 0);
      expect(results.length).toBeLessThanOrEqual(1);
    });
  });

  describe('Fuzzy Search', () => {
    beforeEach(async () => {
      const content = `
function calculateSum(a, b) {
  const result = a + b;
  return result;
}

const myVariable = 42;
const calculate = true;
      `;

      await engine.indexDocument('doc1', '/test/fuzzy.js', content);
    });

    it('should find fuzzy matches within edit distance', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-trace-6',
        query: 'calcualte', // Typo: requires edit distance 2
        mode: 'lex',
        k: 10,
        fuzzy_distance: 2,
        started_at: new Date(),
        stages: [],
      };

      const results = await engine.search(ctx, 'calcualte', 2);
      
      // Should find matches despite typo
      expect(results.length).toBeGreaterThan(0);
    });

    it('should respect fuzzy distance limits', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-trace-7',
        query: 'xyz', // Very different from 'calculate'
        mode: 'lex',
        k: 10,
        fuzzy_distance: 1,
        started_at: new Date(),
        stages: [],
      };

      const results = await engine.search(ctx, 'xyz', 1);
      
      // Should not find matches for very different strings
      expect(results.length).toBe(0);
    });
  });

  describe('Subtoken Search', () => {
    beforeEach(async () => {
      const content = `
function calculateSum(a, b) {
  return a + b;
}

const myVariableName = 'test';
const snake_case_var = 42;
class TestClass {
  methodName() {}
}
      `;

      await engine.indexDocument('doc1', '/test/subtokens.js', content);
    });

    it('should find camelCase subtokens', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-trace-8',
        query: 'Sum', // Part of 'calculateSum'
        mode: 'lex',
        k: 10,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      };

      const results = await engine.search(ctx, 'Sum', 0);
      expect(results.length).toBeGreaterThan(0);
    });

    it('should find snake_case subtokens', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-trace-9',
        query: 'case', // Part of 'snake_case_var'
        mode: 'lex',
        k: 10,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      };

      const results = await engine.search(ctx, 'case', 0);
      expect(results.length).toBeGreaterThan(0);
    });
  });

  describe('Performance', () => {
    it('should complete search within time limits', async () => {
      // Index a larger document
      const lines = [];
      for (let i = 0; i < 100; i++) {
        lines.push(`function test${i}(param${i}) { return param${i} * 2; }`);
        lines.push(`const var${i} = 'value${i}';`);
        lines.push(`class Class${i} { method${i}() {} }`);
      }
      const content = lines.join('\n');

      await engine.indexDocument('large-doc', '/test/large.js', content);

      const startTime = Date.now();
      
      const ctx: SearchContext = {
        trace_id: 'test-trace-10',
        query: 'function',
        mode: 'lex',
        k: 50,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      };

      const results = await engine.search(ctx, 'function', 0);
      
      const latency = Date.now() - startTime;
      
      expect(results.length).toBeGreaterThan(0);
      expect(latency).toBeLessThan(50); // Should be well under Stage-A target of 8ms
    });
  });

  describe('Error Handling', () => {
    it('should handle empty documents', async () => {
      await engine.indexDocument('empty-doc', '/test/empty.js', '');
      
      const stats = engine.getStats();
      expect(stats.document_count).toBe(1);
    });

    it('should handle empty queries', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-trace-11',
        query: '',
        mode: 'lex',
        k: 10,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      };

      const results = await engine.search(ctx, '', 0);
      expect(results.length).toBe(0);
    });

    it('should handle very short queries', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-trace-12',
        query: 'a',
        mode: 'lex',
        k: 10,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      };

      const results = await engine.search(ctx, 'a', 0);
      
      // Should not throw error
      expect(Array.isArray(results)).toBe(true);
    });
  });

  describe('Index Statistics', () => {
    it('should provide accurate statistics', async () => {
      await engine.indexDocument('doc1', '/test/file1.js', 'function test() {}');
      await engine.indexDocument('doc2', '/test/file2.js', 'const value = 42;');

      const stats = engine.getStats();
      
      expect(stats.document_count).toBe(2);
      expect(stats.trigram_count).toBeGreaterThan(0);
      expect(stats.total_positions).toBeGreaterThan(0);
    });

    it('should clear index correctly', async () => {
      await engine.indexDocument('doc1', '/test/file.js', 'test content');
      
      let stats = engine.getStats();
      expect(stats.document_count).toBe(1);
      
      engine.clear();
      
      stats = engine.getStats();
      expect(stats.document_count).toBe(0);
      expect(stats.trigram_count).toBe(0);
      expect(stats.total_positions).toBe(0);
    });
  });
});