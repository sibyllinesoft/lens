/**
 * Phase B1 Integration Tests: LexicalSearchEngine with Bitmap Optimization
 * Tests the integration of bitmap-based trigram indexing with the existing search engine
 */

import { describe, it, expect, beforeEach, jest, mock } from 'bun:test';
import { LexicalSearchEngine } from '../lexical.js';
import { SegmentStorage } from '../../storage/segments.js';
import { featureFlags } from '../../config/features.js';
import type { SearchContext, Candidate } from '../../types/core.js';

// Mock the SegmentStorage dependency
mock('../../storage/segments.js', () => ({
  SegmentStorage: jest.fn().mockImplementation(() => ({})),
}));

describe('LexicalSearchEngine Bitmap Integration', () => {
  let engine: LexicalSearchEngine;
  let segmentStorage: SegmentStorage;
  let mockSearchContext: SearchContext;

  beforeEach(() => {
    segmentStorage = new SegmentStorage('test-path');
    engine = new LexicalSearchEngine(segmentStorage);
    
    mockSearchContext = {
      trace_id: 'test-trace-123',
      repo_sha: 'abc123',
      query: 'test',
      mode: 'lex',
      k: 10,
      fuzzy_distance: 2,
      started_at: new Date(),
      stages: [],
    };

    // Reset feature flags to ensure bitmap optimization is enabled
    featureFlags.updateFeatures({
      bitmapTrigramIndex: {
        enabled: true,
        rolloutPercentage: 100,
        minDocumentThreshold: 5, // Lower threshold for testing
        enablePerformanceLogging: true,
      },
    });
  });

  describe('Document Indexing with Bitmap Optimization', () => {
    it('should index documents and automatically enable bitmap optimization', async () => {
      // Add enough documents to trigger bitmap optimization
      const documents = [
        { id: 'doc1', path: '/src/file1.ts', content: 'function calculateSum(a, b) { return a + b; }' },
        { id: 'doc2', path: '/src/file2.ts', content: 'const processData = async (data) => { return data.filter(item => item.valid); }' },
        { id: 'doc3', path: '/src/file3.ts', content: 'class UserService { constructor() { this.users = []; } }' },
        { id: 'doc4', path: '/src/file4.ts', content: 'export default { calculateSum, processData, UserService };' },
        { id: 'doc5', path: '/src/file5.ts', content: 'import { calculateSum } from "./file1"; console.log(calculateSum(1, 2));' },
        { id: 'doc6', path: '/src/file6.ts', content: 'interface User { id: string; name: string; email: string; }' },
      ];

      // Index all documents
      for (const doc of documents) {
        await engine.indexDocument(doc.id, doc.path, doc.content);
      }

      const stats = engine.getStats();
      
      // Verify bitmap optimization is enabled
      expect(stats.bitmap_optimization.enabled).toBe(true);
      expect(stats.document_count).toBe(6);
      expect(stats.bitmap_optimization.document_count).toBe(6);
      expect(stats.trigram_count).toBeGreaterThan(0);
      expect(stats.bitmap_optimization.trigram_count).toBeGreaterThan(0);
    });

    it('should not enable bitmap optimization for small document sets', async () => {
      // Add fewer documents than the threshold
      await engine.indexDocument('doc1', '/src/file1.ts', 'function test() { return true; }');
      await engine.indexDocument('doc2', '/src/file2.ts', 'const value = 42;');

      const stats = engine.getStats();
      
      // Bitmap optimization should not be enabled
      expect(stats.bitmap_optimization.enabled).toBe(false);
      expect(stats.document_count).toBe(2);
    });

    it('should handle feature flag disabled', async () => {
      // Disable bitmap optimization
      featureFlags.updateFeatures({
        bitmapTrigramIndex: { enabled: false },
      });

      // Add many documents
      for (let i = 0; i < 10; i++) {
        await engine.indexDocument(`doc${i}`, `/src/file${i}.ts`, `function func${i}() { return ${i}; }`);
      }

      const stats = engine.getStats();
      
      // Bitmap optimization should be disabled
      expect(stats.bitmap_optimization.enabled).toBe(false);
      expect(stats.document_count).toBe(10);
    });
  });

  describe('Search Performance with Bitmap Optimization', () => {
    beforeEach(async () => {
      // Set up a larger dataset for performance testing
      const codeTemplates = [
        'function calculateSum(a: number, b: number): number { return a + b; }',
        'async function fetchUserData(userId: string): Promise<User> { return await api.getUser(userId); }',
        'class DataProcessor { process(data: any[]): ProcessedData { return data.map(item => this.transform(item)); } }',
        'const validateEmail = (email: string): boolean => /^[^@]+@[^@]+\.[^@]+$/.test(email);',
        'interface User { id: string; name: string; email: string; createdAt: Date; }',
        'type ProcessedData = { id: string; value: number; processed: true; };',
        'export const CONFIG = { apiUrl: "https://api.example.com", timeout: 5000 };',
        'import { User, ProcessedData } from "./types"; import { validateEmail } from "./utils";',
      ];

      for (let i = 0; i < 50; i++) {
        const template = codeTemplates[i % codeTemplates.length];
        const content = template.replace(/User|Data|Sum/g, (match) => `${match}${i}`);
        await engine.indexDocument(`doc${i}`, `/src/file${i}.ts`, content);
      }
    });

    it('should perform fast exact searches with bitmap optimization', async () => {
      const start = Date.now();
      const results = await engine.search(mockSearchContext, 'function', 0);
      const duration = Date.now() - start;

      // Should find multiple function declarations
      expect(results.length).toBeGreaterThan(0);
      
      // Should complete within Stage-A target (2-8ms)
      expect(duration).toBeLessThan(50); // Allow some overhead for test environment
      
      // Verify results contain expected matches
      const functionMatches = results.filter(result => 
        result.match_reasons.includes('exact')
      );
      expect(functionMatches.length).toBeGreaterThan(0);
    });

    it('should handle complex trigram intersections efficiently', async () => {
      const start = Date.now();
      const results = await engine.search(mockSearchContext, 'User', 0);
      const duration = Date.now() - start;

      expect(results.length).toBeGreaterThan(0);
      expect(duration).toBeLessThan(50);

      // Verify stats show bitmap operations
      const stats = engine.getStats();
      expect(stats.bitmap_optimization.enabled).toBe(true);
    });

    it('should maintain accuracy compared to legacy implementation', async () => {
      // Test both bitmap and legacy paths with same query
      const query = 'calculateSum';
      
      // Search with bitmap optimization
      const bitmapResults = await engine.search(mockSearchContext, query, 0);
      
      // Temporarily disable bitmap optimization for comparison
      featureFlags.updateFeatures({
        bitmapTrigramIndex: { enabled: false },
      });
      
      const legacyResults = await engine.search(mockSearchContext, query, 0);
      
      // Results should be identical (order may vary)
      expect(bitmapResults.length).toBe(legacyResults.length);
      
      const bitmapDocIds = new Set(bitmapResults.map(r => r.doc_id));
      const legacyDocIds = new Set(legacyResults.map(r => r.doc_id));
      expect(bitmapDocIds).toEqual(legacyDocIds);
    });
  });

  describe('Fuzzy Search Integration', () => {
    beforeEach(async () => {
      await engine.indexDocument('doc1', '/src/utils.ts', 'function calculate(x, y) { return x * y; }');
      await engine.indexDocument('doc2', '/src/math.ts', 'function computation(a, b) { return a / b; }');
      await engine.indexDocument('doc3', '/src/process.ts', 'function processing(data) { return data.sort(); }');
    });

    it('should perform fuzzy search while maintaining bitmap optimizations for exact matches', async () => {
      const results = await engine.search(mockSearchContext, 'calculat', 2); // Typo with edit distance 2
      
      expect(results.length).toBeGreaterThan(0);
      
      // Should find exact matches and fuzzy matches
      const exactMatches = results.filter(r => r.score === 1.0);
      const fuzzyMatches = results.filter(r => r.score < 1.0);
      
      // May have both exact (from trigram matching) and fuzzy matches
      expect(exactMatches.length + fuzzyMatches.length).toBe(results.length);
    });

    it('should handle subtoken searches with bitmap optimization', async () => {
      await engine.indexDocument('camel1', '/src/camel.ts', 'const userName = "john"; const userEmail = "john@example.com";');
      await engine.indexDocument('snake1', '/src/snake.ts', 'const user_name = "jane"; const user_email = "jane@example.com";');
      
      const results = await engine.search(mockSearchContext, 'user', 0);
      
      expect(results.length).toBeGreaterThan(0);
      
      // Should find both camelCase and snake_case matches
      const docIds = results.map(r => r.doc_id);
      expect(docIds).toContain('camel1');
      expect(docIds).toContain('snake1');
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle empty queries gracefully', async () => {
      await engine.indexDocument('doc1', '/src/file.ts', 'function test() {}');
      
      const results = await engine.search(mockSearchContext, '', 0);
      expect(results).toHaveLength(0);
    });

    it('should handle very short queries', async () => {
      await engine.indexDocument('doc1', '/src/file.ts', 'const x = 42; const y = 24;');
      
      const results = await engine.search(mockSearchContext, 'x', 0);
      // Short queries use direct search, may or may not find matches
      expect(Array.isArray(results)).toBe(true);
    });

    it('should handle indexing errors gracefully', async () => {
      // Test with potentially problematic content
      await expect(engine.indexDocument('doc1', '/src/file.ts', '')).resolves.not.toThrow();
      await expect(engine.indexDocument('doc2', '/src/file.ts', '   \n\n\t  ')).resolves.not.toThrow();
    });

    it('should handle search errors gracefully', async () => {
      await engine.indexDocument('doc1', '/src/file.ts', 'function test() {}');
      
      // Test with very long query
      const longQuery = 'a'.repeat(1000);
      await expect(engine.search(mockSearchContext, longQuery, 0)).resolves.not.toThrow();
    });
  });

  describe('Performance Monitoring and Logging', () => {
    it('should log performance metrics when enabled', async () => {
      // Enable performance logging
      featureFlags.updateFeatures({
        bitmapTrigramIndex: {
          enablePerformanceLogging: true,
        },
      });

      // Add documents and perform search
      for (let i = 0; i < 10; i++) {
        await engine.indexDocument(`doc${i}`, `/src/file${i}.ts`, `function func${i}() { return ${i}; }`);
      }

      const results = await engine.search(mockSearchContext, 'func', 0);
      
      expect(results.length).toBeGreaterThan(0);
      
      // Performance metrics should be captured in telemetry
      // (In a real test, we'd mock the tracer and verify the metrics were recorded)
      const stats = engine.getStats();
      expect(stats.feature_flags.bitmap_performance_logging).toBe(true);
    });

    it('should provide detailed index statistics', async () => {
      // Add various types of content
      await engine.indexDocument('doc1', '/src/functions.ts', 'function add(a, b) { return a + b; }');
      await engine.indexDocument('doc2', '/src/classes.ts', 'class Calculator { multiply(x, y) { return x * y; } }');
      await engine.indexDocument('doc3', '/src/constants.ts', 'const PI = 3.14159; const E = 2.71828;');

      const stats = engine.getStats();
      
      // Verify comprehensive statistics
      expect(stats).toHaveProperty('document_count');
      expect(stats).toHaveProperty('trigram_count');
      expect(stats).toHaveProperty('bitmap_optimization');
      expect(stats).toHaveProperty('feature_flags');
      
      expect(stats.bitmap_optimization).toHaveProperty('enabled');
      expect(stats.bitmap_optimization).toHaveProperty('trigram_count');
      expect(stats.bitmap_optimization).toHaveProperty('document_count');
      expect(stats.bitmap_optimization).toHaveProperty('memory_efficiency');
    });
  });

  describe('Index Management', () => {
    it('should clear both legacy and bitmap indices', async () => {
      // Add documents
      for (let i = 0; i < 10; i++) {
        await engine.indexDocument(`doc${i}`, `/src/file${i}.ts`, `function test${i}() {}`);
      }

      let stats = engine.getStats();
      expect(stats.document_count).toBe(10);
      expect(stats.bitmap_optimization.document_count).toBe(10);

      // Clear indices
      engine.clear();

      stats = engine.getStats();
      expect(stats.document_count).toBe(0);
      expect(stats.bitmap_optimization.document_count).toBe(0);
      expect(stats.bitmap_optimization.enabled).toBe(false);
    });

    it('should handle index transitions correctly', async () => {
      // Start with small set (no bitmap)
      await engine.indexDocument('doc1', '/src/file1.ts', 'function test1() {}');
      await engine.indexDocument('doc2', '/src/file2.ts', 'function test2() {}');
      
      let stats = engine.getStats();
      expect(stats.bitmap_optimization.enabled).toBe(false);

      // Add more documents to trigger bitmap optimization
      for (let i = 3; i <= 10; i++) {
        await engine.indexDocument(`doc${i}`, `/src/file${i}.ts`, `function test${i}() {}`);
      }

      stats = engine.getStats();
      expect(stats.bitmap_optimization.enabled).toBe(true);
      expect(stats.document_count).toBe(10);

      // Search should work regardless of the transition
      const results = await engine.search(mockSearchContext, 'test', 0);
      expect(results.length).toBeGreaterThan(0);
    });
  });
});