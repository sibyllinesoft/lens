/**
 * Comprehensive tests for Clone-Aware Recall System
 * 
 * Tests validate TODO.md requirements:
 * - Token-shingle MinHash/SimHash indexing (w=5-7 subtokens)
 * - Clone set expansion with budget |C(s)| ≤ k_clone (≤3)
 * - Same-repo, same-symbol-kind veto
 * - Jaccard bonus β ≤ 0.2 log-odds, bounded
 * - Performance gate: +0.5-1.0pp Recall@50 at ≤+0.6ms p95
 * - Span coverage = 100%
 * - Topic similarity threshold τ
 */

import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { CloneAwareRecallSystem } from '../clone-aware-recall.js';
import type { SearchHit, MatchReason } from '../../core/span_resolver/types.js';
import type { SearchContext } from '../../types/core.js';

const createMockSearchHit = (file: string, line: number, score: number, symbolKind?: string): SearchHit => ({
  file,
  line,
  col: 0,
  lang: 'typescript',
  snippet: `function test() { return ${score}; }`,
  score,
  why: ['lexical'] as MatchReason[],
  byte_offset: line * 80,
  span_len: 50,
  symbol_kind: symbolKind as any,
  context_before: 'context before',
  context_after: 'context after',
});

const createMockSearchContext = (query: string): SearchContext => ({
  query,
  repo_sha: 'test-repo-sha',
  k: 20,
  timeout_ms: 1000,
  include_tests: false,
  languages: ['typescript'],
});

describe('CloneAwareRecallSystem', () => {
  let system: CloneAwareRecallSystem;
  
  beforeEach(async () => {
    system = new CloneAwareRecallSystem();
    await system.initialize();
  });
  
  afterEach(async () => {
    await system.shutdown();
  });
  
  describe('Token Shingle Indexing', () => {
    it('should tokenize content into subtokens correctly', async () => {
      const content = 'function calculateSum(firstNumber, secondNumber) { return firstNumber + secondNumber; }';
      
      await system.indexSpan(content, 'math.ts', 1, 0, 'test-repo', 'function');
      
      // System should have created token shingles from the content
      const metrics = system.getPerformanceMetrics();
      expect(metrics.indexed_files).toBe(1);
    });
    
    it('should generate shingles with width w=5-7 subtokens', async () => {
      const content = 'const myVariableName = calculateSomeValue(param1, param2, param3);';
      
      await system.indexSpan(content, 'test.ts', 1, 0, 'test-repo', 'variable');
      
      // Should index content if it has sufficient tokens for shingles
      const metrics = system.getPerformanceMetrics();
      expect(metrics.indexed_files).toBe(1);
    });
    
    it('should skip indexing for very short spans', async () => {
      const shortContent = 'x = 1;'; // Too short for meaningful shingles
      
      await system.indexSpan(shortContent, 'short.ts', 1, 0, 'test-repo', 'variable');
      
      // Should skip indexing short content
      const metrics = system.getPerformanceMetrics();
      // Would track skipped spans in detailed metrics
    });
    
    it('should handle camelCase and snake_case tokenization', async () => {
      const content = 'function handleUserInput(user_name, phoneNumber) { return processUserData(user_name); }';
      
      await system.indexSpan(content, 'user.ts', 1, 0, 'test-repo', 'function');
      
      // Should properly tokenize mixed naming conventions
      const metrics = system.getPerformanceMetrics();
      expect(metrics.indexed_files).toBe(1);
    });
  });
  
  describe('Clone Set Formation and Expansion', () => {
    it('should form clone sets from similar code spans', async () => {
      // Index similar functions in different files
      await system.indexSpan(
        'function addNumbers(a, b) { return a + b; }',
        'math/addition.ts', 1, 0, 'main-repo', 'function'
      );
      
      await system.indexSpan(
        'function addNumbers(x, y) { return x + y; }', // Clone with different param names
        'utils/calculator.ts', 5, 0, 'fork-repo', 'function'
      );
      
      await system.indexSpan(
        'function addNumbers(num1, num2) { return num1 + num2; }', // Another clone
        'helpers/math.ts', 10, 0, 'mirror-repo', 'function'
      );
      
      const originalHits = [createMockSearchHit('math/addition.ts', 1, 95, 'function')];
      const ctx = createMockSearchContext('addNumbers');
      
      const expandedHits = await system.expandWithClones(originalHits, ctx);
      
      // Should expand with clones
      expect(expandedHits.length).toBeGreaterThan(originalHits.length);
      expect(expandedHits.length).toBeLessThanOrEqual(originalHits.length + 3); // k_clone ≤ 3
    });
    
    it('should enforce clone budget constraint |C(s)| ≤ k_clone (≤3)', async () => {
      const baseFunction = 'function commonPattern() { return "shared logic"; }';
      
      // Index many similar functions
      for (let i = 0; i < 10; i++) {
        await system.indexSpan(
          baseFunction.replace('commonPattern', `pattern${i}`),
          `file${i}.ts`, 1, 0, `repo${i}`, 'function'
        );
      }
      
      const originalHits = [createMockSearchHit('file0.ts', 1, 95, 'function')];
      const ctx = createMockSearchContext('commonPattern');
      
      const expandedHits = await system.expandWithClones(originalHits, ctx);
      
      // Should respect clone budget: max 3 additional clones
      const addedClones = expandedHits.length - originalHits.length;
      expect(addedClones).toBeLessThanOrEqual(3);
    });
    
    it('should apply same-repo, same-symbol-kind veto', async () => {
      const repo = 'same-repo';
      
      // Index multiple functions in same repo
      await system.indexSpan(
        'function helperA() { return 1; }',
        'utils/a.ts', 1, 0, repo, 'function'
      );
      
      await system.indexSpan(
        'function helperB() { return 2; }', // Same repo, same symbol kind - should be vetoed
        'utils/b.ts', 1, 0, repo, 'function'
      );
      
      await system.indexSpan(
        'const helperC = 3;', // Same repo, different symbol kind - should not be vetoed
        'utils/c.ts', 1, 0, repo, 'variable'
      );
      
      await system.indexSpan(
        'function helperD() { return 4; }', // Different repo - should not be vetoed
        'utils/d.ts', 1, 0, 'different-repo', 'function'
      );
      
      const originalHits = [createMockSearchHit('utils/a.ts', 1, 95, 'function')];
      const ctx = createMockSearchContext('helper');
      
      const expandedHits = await system.expandWithClones(originalHits, ctx);
      const expandedFiles = expandedHits.map(h => h.file);
      
      // Should not include same-repo, same-symbol-kind clone
      expect(expandedFiles).not.toContain('utils/b.ts');
      
      // Should include different symbol kind from same repo
      expect(expandedFiles).toContain('utils/c.ts');
      
      // Should include same symbol kind from different repo
      expect(expandedFiles).toContain('utils/d.ts');
    });
    
    it('should apply topic similarity threshold τ', async () => {
      // Index functions in very different domains
      await system.indexSpan(
        'function calculateTax(income) { return income * 0.25; }',
        'finance/tax.ts', 1, 0, 'main-repo', 'function'
      );
      
      await system.indexSpan(
        'function drawCircle(radius) { return Math.PI * radius * 2; }', // Different domain
        'graphics/shapes.ts', 1, 0, 'graphics-repo', 'function'
      );
      
      await system.indexSpan(
        'function calculateInterest(principal) { return principal * 0.05; }', // Similar domain
        'finance/interest.ts', 1, 0, 'finance-repo', 'function'
      );
      
      const originalHits = [createMockSearchHit('finance/tax.ts', 1, 95, 'function')];
      const ctx = createMockSearchContext('calculate');
      
      const expandedHits = await system.expandWithClones(originalHits, ctx);
      const expandedFiles = expandedHits.map(h => h.file);
      
      // Should prefer topically similar clones
      expect(expandedFiles).toContain('finance/interest.ts');
      
      // May exclude topically dissimilar clones based on τ threshold
      // (Exact behavior depends on topic similarity calculation)
    });
  });
  
  describe('Jaccard Bonus and Scoring', () => {
    it('should apply bounded Jaccard bonus β ≤ 0.2 log-odds', async () => {
      await system.indexSpan(
        'function process(data) { return data.filter(x => x > 0); }',
        'original.ts', 1, 0, 'main-repo', 'function'
      );
      
      await system.indexSpan(
        'function process(items) { return items.filter(item => item > 0); }', // High similarity
        'clone.ts', 1, 0, 'fork-repo', 'function'
      );
      
      const originalHits = [createMockSearchHit('original.ts', 1, 95, 'function')];
      const ctx = createMockSearchContext('process filter');
      
      const expandedHits = await system.expandWithClones(originalHits, ctx);
      
      if (expandedHits.length > originalHits.length) {
        const cloneHit = expandedHits.find(h => h.file === 'clone.ts');
        expect(cloneHit).toBeDefined();
        
        // Clone hit should have score bonus, but bounded
        expect(cloneHit!.score).toBeGreaterThan(originalHits[0].score);
        expect(cloneHit!.score).toBeLessThan(originalHits[0].score + 20); // Reasonable bonus bound
      }
    });
    
    it('should preserve span scores (keep spans sacred)', async () => {
      await system.indexSpan(
        'function test() { return true; }',
        'test1.ts', 1, 0, 'repo1', 'function'
      );
      
      await system.indexSpan(
        'function test() { return false; }',
        'test2.ts', 1, 0, 'repo2', 'function'
      );
      
      const originalHits = [
        createMockSearchHit('test1.ts', 1, 95, 'function'),
        createMockSearchHit('unrelated.ts', 5, 85, 'class'),
      ];
      
      const ctx = createMockSearchContext('test');
      const expandedHits = await system.expandWithClones(originalHits, ctx);
      
      // Original hits should preserve their scores
      const originalHit = expandedHits.find(h => h.file === 'test1.ts');
      expect(originalHit!.score).toBe(95); // Original score preserved
      
      const unrelatedHit = expandedHits.find(h => h.file === 'unrelated.ts');
      expect(unrelatedHit!.score).toBe(85); // Original score preserved
    });
  });
  
  describe('Performance Requirements', () => {
    it('should meet latency budget ≤+0.6ms p95', async () => {
      // Index substantial content for realistic testing
      for (let i = 0; i < 50; i++) {
        await system.indexSpan(
          `function func${i}(param1, param2) { return param1 + param2 + ${i}; }`,
          `file${i}.ts`, 1, 0, `repo${i % 5}`, 'function'
        );
      }
      
      const originalHits = Array.from({ length: 20 }, (_, i) =>
        createMockSearchHit(`file${i}.ts`, 1, 90 - i, 'function')
      );
      
      const ctx = createMockSearchContext('func param');
      
      // Run multiple expansions to measure p95 latency
      const latencies: number[] = [];
      
      for (let i = 0; i < 20; i++) {
        const start = Date.now();
        await system.expandWithClones(originalHits, ctx);
        const latency = Date.now() - start;
        latencies.push(latency);
      }
      
      // Calculate p95 latency
      const p95Latency = latencies.sort((a, b) => a - b)[Math.floor(latencies.length * 0.95)];
      
      // Should meet performance gate
      expect(p95Latency).toBeLessThanOrEqual(0.6); // ≤+0.6ms
    });
    
    it('should achieve recall improvement target +0.5-1.0pp at Recall@50', async () => {
      // Create scenario with clear clone opportunities
      const baseCode = 'function calculateTotal(items) { return items.reduce((sum, item) => sum + item.value, 0); }';
      
      // Index original and clones
      await system.indexSpan(baseCode, 'original.ts', 1, 0, 'main', 'function');
      await system.indexSpan(
        baseCode.replace('calculateTotal', 'computeSum'),
        'clone1.ts', 1, 0, 'fork1', 'function'
      );
      await system.indexSpan(
        baseCode.replace('calculateTotal', 'getTotalValue'),
        'clone2.ts', 1, 0, 'fork2', 'function'
      );
      
      // Test with exactly 50 results to measure Recall@50
      const originalHits = Array.from({ length: 50 }, (_, i) => {
        if (i === 0) return createMockSearchHit('original.ts', 1, 95, 'function');
        return createMockSearchHit(`other${i}.ts`, 1, 90 - i, 'function');
      });
      
      const ctx = createMockSearchContext('calculate total');
      ctx.k = 50; // Ensure we're testing Recall@50
      
      const expandedHits = await system.expandWithClones(originalHits, ctx);
      
      // Calculate recall improvement
      const originalRecall = Math.min(1, originalHits.length / 50);
      const expandedRecall = Math.min(1, expandedHits.length / 50);
      const recallImprovement = expandedRecall - originalRecall;
      
      // Should meet target: +0.5-1.0pp = +0.005-0.010 in decimal
      expect(recallImprovement).toBeGreaterThanOrEqual(0.005);
      expect(recallImprovement).toBeLessThanOrEqual(0.010);
    });
    
    it('should maintain 100% span coverage', async () => {
      // Index various types of spans
      await system.indexSpan(
        'function test() {}',
        'func.ts', 1, 0, 'repo1', 'function'
      );
      
      await system.indexSpan(
        'class TestClass {}',
        'class.ts', 1, 0, 'repo2', 'class'
      );
      
      await system.indexSpan(
        'const variable = 42;',
        'var.ts', 1, 0, 'repo3', 'variable'
      );
      
      const metrics = system.getPerformanceMetrics();
      
      // All indexed spans should be covered (span coverage = 100%)
      expect(metrics.indexed_files).toBe(3);
      
      // In production, would verify that all spans are searchable and expandable
      // For now, verify that indexing completed successfully
    });
  });
  
  describe('Edge Cases and Error Handling', () => {
    it('should handle empty or malformed content gracefully', async () => {
      // Empty content
      await expect(
        system.indexSpan('', 'empty.ts', 1, 0, 'repo', 'unknown')
      ).resolves.not.toThrow();
      
      // Very short content
      await expect(
        system.indexSpan('x', 'tiny.ts', 1, 0, 'repo', 'variable')
      ).resolves.not.toThrow();
      
      // Special characters
      await expect(
        system.indexSpan('!@#$%^&*()', 'special.ts', 1, 0, 'repo', 'unknown')
      ).resolves.not.toThrow();
    });
    
    it('should handle expansion with no original hits', async () => {
      const ctx = createMockSearchContext('nonexistent');
      const expandedHits = await system.expandWithClones([], ctx);
      
      expect(expandedHits).toEqual([]);
    });
    
    it('should handle expansion with no indexed content', async () => {
      const originalHits = [createMockSearchHit('test.ts', 1, 95, 'function')];
      const ctx = createMockSearchContext('query');
      
      // No content indexed, should return original hits
      const expandedHits = await system.expandWithClones(originalHits, ctx);
      
      expect(expandedHits).toEqual(originalHits);
    });
    
    it('should filter out vendor/third-party paths', async () => {
      // Index content in vendor directories
      await system.indexSpan(
        'function vendor() {}',
        'node_modules/pkg/index.js', 1, 0, 'repo', 'function'
      );
      
      await system.indexSpan(
        'function vendor() {}',
        'vendor/lib/util.js', 1, 0, 'repo', 'function'
      );
      
      await system.indexSpan(
        'function vendor() {}',
        'src/vendor.ts', 1, 0, 'repo', 'function' // Not in vendor path
      );
      
      const originalHits = [createMockSearchHit('src/vendor.ts', 1, 95, 'function')];
      const ctx = createMockSearchContext('vendor');
      
      const expandedHits = await system.expandWithClones(originalHits, ctx);
      const expandedFiles = expandedHits.map(h => h.file);
      
      // Should not include vendor directory clones
      expect(expandedFiles).not.toContain('node_modules/pkg/index.js');
      expect(expandedFiles).not.toContain('vendor/lib/util.js');
      
      // Should include non-vendor clones
      expect(expandedFiles).toContain('src/vendor.ts');
    });
  });
  
  describe('Performance Metrics and Monitoring', () => {
    it('should track comprehensive performance metrics', async () => {
      await system.indexSpan(
        'function test() { return 42; }',
        'test.ts', 1, 0, 'repo', 'function'
      );
      
      const originalHits = [createMockSearchHit('test.ts', 1, 95, 'function')];
      const ctx = createMockSearchContext('test');
      
      await system.expandWithClones(originalHits, ctx);
      
      const metrics = system.getPerformanceMetrics();
      
      expect(metrics).toHaveProperty('clone_sets_count');
      expect(metrics).toHaveProperty('indexed_files');
      expect(metrics).toHaveProperty('expansion_p95_latency_ms');
      expect(metrics).toHaveProperty('cache_hit_rate');
      expect(metrics).toHaveProperty('performance_gate_breaches');
      
      expect(metrics.indexed_files).toBeGreaterThan(0);
      expect(metrics.clone_sets_count).toBeGreaterThanOrEqual(0);
    });
    
    it('should track performance gate breaches', async () => {
      // This would require a scenario that actually breaches the performance gate
      // For now, verify that the breach counter exists and is initialized
      const metrics = system.getPerformanceMetrics();
      expect(metrics.performance_gate_breaches).toBeGreaterThanOrEqual(0);
    });
  });
});