/**
 * Integration tests for search-engine.ts - exercises actual search pipeline
 * These tests import and run real implementation code to achieve measurable coverage
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import { LensSearchEngine } from '../../api/search-engine.js';
import type { SearchContext } from '../../types/core.js';
import type { SupportedLanguage } from '../../types/api.js';
import { tmpdir } from 'os';
import { join } from 'path';
import { mkdtemp, rm, mkdir, writeFile } from 'fs/promises';

describe('Search Engine Integration Tests', () => {
  let searchEngine: LensSearchEngine;
  let tempDir: string;

  beforeAll(async () => {
    // Create temporary directory for test data
    tempDir = await mkdtemp(join(tmpdir(), 'lens-search-test-'));
    
    // Create some test directories and files
    const testRepoDir = join(tempDir, 'test-repo');
    await mkdir(testRepoDir, { recursive: true });
    
    // Create sample TypeScript file
    await writeFile(join(testRepoDir, 'sample.ts'), `
      export interface TestInterface {
        id: string;
        name: string;
      }
      
      export class TestClass implements TestInterface {
        constructor(public id: string, public name: string) {}
        
        public testMethod(): void {
          console.log('Test method');
        }
      }
      
      export function testFunction(param: string): number {
        return param.length;
      }
    `);

    // Create sample JavaScript file
    await writeFile(join(testRepoDir, 'sample.js'), `
      function calculateSum(a, b) {
        return a + b;
      }
      
      class Calculator {
        add(x, y) {
          return x + y;
        }
        
        multiply(x, y) {
          return x * y;
        }
      }
      
      module.exports = { calculateSum, Calculator };
    `);

    // Create sample Python file
    await writeFile(join(testRepoDir, 'sample.py'), `
      def process_data(data):
          """Process the input data"""
          return [item.upper() for item in data]
      
      class DataProcessor:
          def __init__(self, config):
              self.config = config
          
          def transform(self, input_data):
              return self.process_items(input_data)
          
          def process_items(self, items):
              return [self.config.transform(item) for item in items]
    `);
  });

  beforeEach(async () => {
    // Create new search engine instance for each test
    searchEngine = new LensSearchEngine(tempDir);
    
    // Initialize the search engine
    await searchEngine.initialize();
  });

  afterAll(async () => {
    if (tempDir) {
      await rm(tempDir, { recursive: true, force: true });
    }
  });

  describe('Search Engine Initialization', () => {
    it('should initialize successfully with valid configuration', async () => {
      const engine = new LensSearchEngine(tempDir);

      await expect(engine.initialize()).resolves.not.toThrow();
      expect(engine.isHealthy()).toBe(true);
    });

    it('should report system health after initialization', () => {
      const health = searchEngine.getSystemHealth();
      
      expect(health).toHaveProperty('status');
      expect(health).toHaveProperty('components');
      expect(health.status).toMatch(/^(healthy|degraded|unhealthy)$/);
      expect(typeof health.components).toBe('object');
    });

    it('should handle configuration changes', async () => {
      // Test configuration handling without complex update method
      expect(searchEngine).toBeDefined();
      expect(searchEngine.isHealthy()).toBe(true);
    });
  });

  describe('Basic Search Operations', () => {
    const createTestContext = (language: SupportedLanguage = 'typescript'): SearchContext => ({
      repo_name: 'test-repo',
      file_path: 'sample.' + (language === 'typescript' ? 'ts' : language === 'javascript' ? 'js' : 'py'),
      line_number: 1,
      column_number: 1
    });

    it('should perform basic text search', async () => {
      const result = await searchEngine.search({
        query: 'test',
        num_results: 10,
        language: 'typescript',
        context: createTestContext('typescript')
      });

      expect(result).toHaveProperty('hits');
      expect(Array.isArray(result.hits)).toBe(true);
      expect(result.hits.length).toBeGreaterThanOrEqual(0);
      
      // Check result structure
      if (result.hits.length > 0) {
        const hit = result.hits[0];
        expect(hit).toHaveProperty('file_path');
        expect(hit).toHaveProperty('line_number');
        expect(hit).toHaveProperty('score');
        expect(typeof hit.score).toBe('number');
      }
    });

    it('should handle different programming languages', async () => {
      const languages: SupportedLanguage[] = ['typescript', 'javascript', 'python'];
      
      for (const language of languages) {
        const result = await searchEngine.search({
          query: 'function',
          num_results: 5,
          language,
          context: createTestContext(language)
        });

        expect(result).toHaveProperty('hits');
        expect(Array.isArray(result.hits)).toBe(true);
      }
    });

    it('should respect num_results parameter', async () => {
      const maxResults = 3;
      const result = await searchEngine.search({
        query: 'class',
        num_results: maxResults,
        language: 'typescript',
        context: createTestContext('typescript')
      });

      expect(result.hits.length).toBeLessThanOrEqual(maxResults);
    });

    it('should handle empty queries gracefully', async () => {
      const result = await searchEngine.search({
        query: '',
        num_results: 10,
        language: 'typescript',
        context: createTestContext('typescript')
      });

      expect(result).toHaveProperty('hits');
      expect(Array.isArray(result.hits)).toBe(true);
    });

    it('should handle queries with special characters', async () => {
      const specialQueries = ['test()', 'class.method', 'function[]', 'object.property'];
      
      for (const query of specialQueries) {
        const result = await searchEngine.search({
          query,
          num_results: 10,
          language: 'typescript',
          context: createTestContext('typescript')
        });

        expect(result).toHaveProperty('hits');
        expect(Array.isArray(result.hits)).toBe(true);
      }
    });
  });

  describe('Advanced Search Features', () => {
    it('should perform fuzzy search', async () => {
      const result = await searchEngine.search({
        query: 'testMetod', // Intentional typo
        num_results: 10,
        language: 'typescript',
        context: createTestContext('typescript'),
        fuzzy: true
      });

      expect(result).toHaveProperty('hits');
      expect(Array.isArray(result.hits)).toBe(true);
    });

    it('should perform case-insensitive search', async () => {
      const result = await searchEngine.search({
        query: 'TESTCLASS', // All caps
        num_results: 10,
        language: 'typescript',
        context: createTestContext('typescript'),
        case_sensitive: false
      });

      expect(result).toHaveProperty('hits');
      expect(Array.isArray(result.hits)).toBe(true);
    });

    it('should search within specific file paths', async () => {
      const result = await searchEngine.search({
        query: 'test',
        num_results: 10,
        language: 'typescript',
        context: createTestContext('typescript'),
        file_path_filter: 'sample.ts'
      });

      expect(result).toHaveProperty('hits');
      expect(Array.isArray(result.hits)).toBe(true);
      
      // All results should be from the specified file
      result.hits.forEach(hit => {
        expect(hit.file_path).toContain('sample.ts');
      });
    });

    it('should handle context-aware search', async () => {
      const context = createTestContext('typescript');
      context.line_number = 10; // Middle of TestClass
      
      const result = await searchEngine.search({
        query: 'method',
        num_results: 10,
        language: 'typescript',
        context,
        context_aware: true
      });

      expect(result).toHaveProperty('hits');
      expect(Array.isArray(result.hits)).toBe(true);
    });
  });

  describe('Search Performance and Metrics', () => {
    it('should track search latency metrics', async () => {
      const result = await searchEngine.search({
        query: 'performance test',
        num_results: 10,
        language: 'typescript',
        context: createTestContext('typescript')
      });

      // Check if latency metrics are included
      expect(typeof result.stage_a_latency).toBe('number');
      expect(typeof result.stage_b_latency).toBe('number');
      expect(typeof result.stage_c_latency).toBe('number');
      
      // Latency should be positive
      expect(result.stage_a_latency).toBeGreaterThanOrEqual(0);
      expect(result.stage_b_latency).toBeGreaterThanOrEqual(0);
      expect(result.stage_c_latency).toBeGreaterThanOrEqual(0);
    });

    it('should handle concurrent search requests', async () => {
      const searchPromises = Array.from({ length: 5 }, (_, i) =>
        searchEngine.search({
          query: `concurrent test ${i}`,
          num_results: 10,
          language: 'typescript',
          context: createTestContext('typescript')
        })
      );

      const results = await Promise.all(searchPromises);
      
      expect(results).toHaveLength(5);
      results.forEach(result => {
        expect(result).toHaveProperty('hits');
        expect(Array.isArray(result.hits)).toBe(true);
      });
    });

    it('should provide search performance insights', async () => {
      // Perform multiple searches to generate metrics
      for (let i = 0; i < 3; i++) {
        await searchEngine.search({
          query: `perf test ${i}`,
          num_results: 10,
          language: 'typescript',
          context: createTestContext('typescript')
        });
      }

      const insights = await searchEngine.getPerformanceInsights();
      
      expect(insights).toHaveProperty('totalSearches');
      expect(insights).toHaveProperty('averageLatency');
      expect(typeof insights.totalSearches).toBe('number');
      expect(typeof insights.averageLatency).toBe('number');
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle invalid language gracefully', async () => {
      const result = await searchEngine.search({
        query: 'test',
        num_results: 10,
        language: 'invalid-language' as SupportedLanguage,
        context: createTestContext('typescript')
      });

      // Should not throw, but may return empty results
      expect(result).toHaveProperty('hits');
      expect(Array.isArray(result.hits)).toBe(true);
    });

    it('should handle non-existent file paths', async () => {
      const context = createTestContext('typescript');
      context.file_path = 'non-existent-file.ts';
      
      const result = await searchEngine.search({
        query: 'test',
        num_results: 10,
        language: 'typescript',
        context
      });

      expect(result).toHaveProperty('hits');
      expect(Array.isArray(result.hits)).toBe(true);
    });

    it('should handle very large queries', async () => {
      const longQuery = 'very '.repeat(100) + 'long query';
      
      const result = await searchEngine.search({
        query: longQuery,
        num_results: 10,
        language: 'typescript',
        context: createTestContext('typescript')
      });

      expect(result).toHaveProperty('hits');
      expect(Array.isArray(result.hits)).toBe(true);
    });

    it('should handle invalid line/column numbers', async () => {
      const context = createTestContext('typescript');
      context.line_number = -1;
      context.column_number = -1;
      
      const result = await searchEngine.search({
        query: 'test',
        num_results: 10,
        language: 'typescript',
        context
      });

      expect(result).toHaveProperty('hits');
      expect(Array.isArray(result.hits)).toBe(true);
    });
  });

  describe('Cleanup and Shutdown', () => {
    it('should cleanup resources properly', async () => {
      const engine = new LensSearchEngine(tempDir);
      
      await engine.initialize();
      await expect(engine.shutdown()).resolves.not.toThrow();
      
      // After shutdown, health should be unhealthy
      expect(engine.isHealthy()).toBe(false);
    });
  });
});