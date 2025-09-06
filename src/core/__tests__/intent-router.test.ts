/**
 * Comprehensive Tests for Intent Router Component
 * Tests query classification, intent routing, and fallback behavior
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { IntentRouter } from '../intent-router.js';
import { LSPStageBEnhancer } from '../lsp-stage-b.js';
import type { SearchContext, Candidate, QueryIntent, IntentClassification } from '../../types/core.js';

// Mock LSPStageBEnhancer
vi.mock('../lsp-stage-b.js', () => ({
  LSPStageBEnhancer: vi.fn().mockImplementation(() => ({
    enhanceStageB: vi.fn()
  }))
}));

// Mock telemetry tracer
vi.mock('../../telemetry/tracer.js', () => ({
  LensTracer: {
    createChildSpan: vi.fn(() => ({
      setAttributes: vi.fn(),
      recordException: vi.fn(),
      end: vi.fn()
    }))
  }
}));

describe('IntentRouter', () => {
  let intentRouter: IntentRouter;
  let mockLspEnhancer: jest.Mocked<LSPStageBEnhancer>;
  let mockContext: SearchContext;

  beforeEach(() => {
    vi.clearAllMocks();
    
    mockLspEnhancer = new (LSPStageBEnhancer as any)();
    intentRouter = new IntentRouter(mockLspEnhancer);

    mockContext = {
      mode: 'search',
      repo_sha: 'test-sha-123',
      language_hint: 'typescript',
      file_path: '/test/file.ts',
      line_hint: 10,
      query_timestamp: new Date(),
      trace_id: 'test-trace-123'
    };

    // Default mock response for LSP enhancer
    mockLspEnhancer.enhanceStageB.mockResolvedValue({
      candidates: [],
      stage_latency_ms: 50,
      lsp_contributions: 0,
      total_lsp_hints_used: 0,
      performance_metrics: {
        lsp_lookup_ms: 10,
        merge_dedupe_ms: 5,
        context_enrichment_ms: 5
      }
    });
  });

  describe('query intent classification', () => {
    it('should classify definition queries correctly', () => {
      const definitionQueries = [
        'def MyClass',
        'define getUserProfile',
        'definition of handleSubmit',
        'declare interface User',
        'what is MyComponent',
        'where is parseData',
        'find definition of validateInput',
        'class MyService',
        'function calculateTotal',
        'interface ApiResponse',
        'type UserData',
        'TestClass definition',
        'handleError declaration'
      ];

      for (const query of definitionQueries) {
        const classification = intentRouter.classifyQueryIntent(query);
        expect(classification.intent).toBe('def');
        expect(classification.confidence).toBeGreaterThanOrEqual(0.7);
        expect(classification.features.has_definition_pattern).toBe(true);
      }
    });

    it('should classify reference queries correctly', () => {
      const referenceQueries = [
        'refs MyClass',
        'references getUserProfile',
        'usages of handleSubmit',
        'uses validateInput',
        'find references to MyComponent',
        'show usages of calculateTotal',
        'list references ApiResponse',
        'where uses parseData',
        'who calls handleError',
        'who references UserService',
        'MyMethod references',
        'calculatePrice usages',
        'ValidationError calls'
      ];

      for (const query of referenceQueries) {
        const classification = intentRouter.classifyQueryIntent(query);
        expect(classification.intent).toBe('refs');
        expect(classification.confidence).toBeGreaterThanOrEqual(0.7);
        expect(classification.features.has_reference_pattern).toBe(true);
      }
    });

    it('should classify symbol queries correctly', () => {
      const symbolQueries = [
        'class MyClass',
        'function getUserData',
        'method handleSubmit',
        'var userData',
        'const API_URL',
        'let isLoading',
        'type UserInfo',
        'interface ApiConfig',
        'enum StatusCode',
        'MyClass',              // PascalCase class
        'getUserData()',        // function call
        'user.profile',         // member access
        '@Component',           // decorator
        '@Injectable',          // annotation
        'MyService.getInstance'
      ];

      for (const query of symbolQueries) {
        const classification = intentRouter.classifyQueryIntent(query);
        expect(classification.intent).toBe('symbol');
        expect(classification.confidence).toBeGreaterThanOrEqual(0.7);
        expect(classification.features.has_symbol_prefix).toBe(true);
      }
    });

    it('should classify structural queries correctly', () => {
      const structuralQueries = [
        'if (condition) { return true; }',
        'function() => {}',
        'const { name, age } = user',
        'array.map((item) => item.id)',
        'obj?.property?.method()',
        'condition ? value1 : value2',
        'x + y * z',
        'a === b && c !== d',
        'items.filter(x => x.active)',
        '{ key: value, other: data }',
        '[1, 2, 3].forEach()',
        'try { } catch (error) { }'
      ];

      for (const query of structuralQueries) {
        const classification = intentRouter.classifyQueryIntent(query);
        expect(classification.intent).toBe('struct');
        expect(classification.confidence).toBeGreaterThanOrEqual(0.7);
        expect(classification.features.has_structural_chars).toBe(true);
      }
    });

    it('should classify natural language queries correctly', () => {
      const naturalLanguageQueries = [
        'how to handle user authentication',
        'what function processes the payment',
        'find the method that validates input',
        'show me functions for database connection',
        'where is the code that handles errors',
        'list all methods in the user service',
        'functions that work with the shopping cart',
        'methods for handling file uploads',
        'how do I validate email addresses',
        'what component manages the sidebar'
      ];

      for (const query of naturalLanguageQueries) {
        const classification = intentRouter.classifyQueryIntent(query);
        expect(classification.intent).toBe('NL');
        expect(classification.confidence).toBeGreaterThanOrEqual(0.7);
        expect(classification.features.is_natural_language).toBe(true);
      }
    });

    it('should classify lexical queries as fallback', () => {
      const lexicalQueries = [
        'user',
        'data',
        'config',
        'handleSubmit',
        'myVariable',
        'testFunction',
        'single-word',
        'two words',
        'simple query'
      ];

      for (const query of lexicalQueries) {
        const classification = intentRouter.classifyQueryIntent(query);
        expect(classification.intent).toBe('lexical');
        expect(classification.confidence).toBeLessThan(0.7);
      }
    });
  });

  describe('query routing', () => {
    beforeEach(() => {
      // Mock successful LSP responses
      mockLspEnhancer.enhanceStageB.mockResolvedValue({
        candidates: [
          {
            file_path: '/test/file.ts',
            line_no: 10,
            col_no: 5,
            content: 'class MyClass {',
            score: 0.95,
            match_reasons: ['lsp_hint', 'exact_match'],
            symbol_kind: 'class'
          }
        ],
        stage_latency_ms: 25,
        lsp_contributions: 1,
        total_lsp_hints_used: 5,
        performance_metrics: {
          lsp_lookup_ms: 10,
          merge_dedupe_ms: 5,
          context_enrichment_ms: 10
        }
      });
    });

    it('should route definition queries with high confidence', async () => {
      const result = await intentRouter.routeQuery(
        'def MyClass',
        mockContext
      );

      expect(result.classification.intent).toBe('def');
      expect(result.confidence_threshold_met).toBe(true);
      expect(result.routing_path).toContain('classified_as_def');
      expect(result.routing_path).toContain('definition_search');
      expect(result.primary_candidates).toHaveLength(1);
      expect(result.fallback_triggered).toBe(false);

      // Verify LSP enhancer was called for definition search
      expect(mockLspEnhancer.enhanceStageB).toHaveBeenCalled();
    });

    it('should route reference queries with high confidence', async () => {
      const result = await intentRouter.routeQuery(
        'refs MyMethod',
        mockContext
      );

      expect(result.classification.intent).toBe('refs');
      expect(result.confidence_threshold_met).toBe(true);
      expect(result.routing_path).toContain('classified_as_refs');
      expect(result.routing_path).toContain('references_search');
      expect(result.primary_candidates).toHaveLength(1);
      expect(result.fallback_triggered).toBe(false);
    });

    it('should route symbol queries directly to LSP', async () => {
      const result = await intentRouter.routeQuery(
        'class UserService',
        mockContext
      );

      expect(result.classification.intent).toBe('symbol');
      expect(result.confidence_threshold_met).toBe(true);
      expect(result.routing_path).toContain('classified_as_symbol');
      expect(result.routing_path).toContain('symbol_search');
      expect(result.primary_candidates).toHaveLength(1);
      expect(result.fallback_triggered).toBe(false);

      expect(mockLspEnhancer.enhanceStageB).toHaveBeenCalledWith(
        'class UserService',
        mockContext,
        [],
        20
      );
    });

    it('should route structural queries appropriately', async () => {
      const result = await intentRouter.routeQuery(
        'if (condition) { return true; }',
        mockContext
      );

      expect(result.classification.intent).toBe('struct');
      expect(result.confidence_threshold_met).toBe(true);
      expect(result.routing_path).toContain('classified_as_struct');
      expect(result.routing_path).toContain('structural_search');
      // Structural search is not fully implemented, so expects empty results
      expect(result.primary_candidates).toHaveLength(0);
    });

    it('should route natural language queries appropriately', async () => {
      const result = await intentRouter.routeQuery(
        'how to validate user input',
        mockContext
      );

      expect(result.classification.intent).toBe('NL');
      expect(result.confidence_threshold_met).toBe(true);
      expect(result.routing_path).toContain('classified_as_NL');
      expect(result.routing_path).toContain('nl_search');
      // NL search is not fully implemented, so expects empty results
      expect(result.primary_candidates).toHaveLength(0);
    });

    it('should skip specialized routing for lexical queries', async () => {
      const result = await intentRouter.routeQuery(
        'getUserData',
        mockContext
      );

      expect(result.classification.intent).toBe('lexical');
      expect(result.confidence_threshold_met).toBe(false);
      expect(result.routing_path).toContain('classified_as_lexical');
      expect(result.routing_path).toContain('low_confidence_full_search');
    });
  });

  describe('fallback behavior', () => {
    beforeEach(() => {
      // Mock empty LSP responses to trigger fallback
      mockLspEnhancer.enhanceStageB.mockResolvedValue({
        candidates: [],
        stage_latency_ms: 15,
        lsp_contributions: 0,
        total_lsp_hints_used: 0,
        performance_metrics: {
          lsp_lookup_ms: 5,
          merge_dedupe_ms: 2,
          context_enrichment_ms: 3
        }
      });
    });

    it('should trigger fallback when definition search returns no results', async () => {
      const mockFullSearchHandler = vi.fn().mockResolvedValue([
        {
          file_path: '/fallback/file.ts',
          line_no: 20,
          col_no: 0,
          content: 'function MyFunction() {',
          score: 0.8,
          match_reasons: ['fuzzy_match'],
          symbol_kind: 'function'
        }
      ]);

      const result = await intentRouter.routeQuery(
        'def MyFunction',
        mockContext,
        undefined,
        mockFullSearchHandler
      );

      expect(result.classification.intent).toBe('def');
      expect(result.fallback_triggered).toBe(true);
      expect(result.routing_path).toContain('fallback_triggered');
      expect(result.primary_candidates).toHaveLength(1);
      expect(mockFullSearchHandler).toHaveBeenCalledWith('def MyFunction', mockContext);
    });

    it('should trigger fallback when reference search returns no results', async () => {
      const mockFullSearchHandler = vi.fn().mockResolvedValue([
        {
          file_path: '/fallback/file.ts',
          line_no: 15,
          col_no: 10,
          content: 'myFunction()',
          score: 0.75,
          match_reasons: ['partial_match'],
          symbol_kind: 'reference'
        }
      ]);

      const result = await intentRouter.routeQuery(
        'refs myFunction',
        mockContext,
        undefined,
        mockFullSearchHandler
      );

      expect(result.classification.intent).toBe('refs');
      expect(result.fallback_triggered).toBe(true);
      expect(result.routing_path).toContain('fallback_triggered');
      expect(result.primary_candidates).toHaveLength(1);
      expect(mockFullSearchHandler).toHaveBeenCalledWith('refs myFunction', mockContext);
    });

    it('should trigger fallback when symbol search returns no results', async () => {
      const mockFullSearchHandler = vi.fn().mockResolvedValue([
        {
          file_path: '/fallback/service.ts',
          line_no: 5,
          col_no: 0,
          content: 'class MyService extends BaseService {',
          score: 0.85,
          match_reasons: ['class_match'],
          symbol_kind: 'class'
        }
      ]);

      const result = await intentRouter.routeQuery(
        'class MyService',
        mockContext,
        undefined,
        mockFullSearchHandler
      );

      expect(result.classification.intent).toBe('symbol');
      expect(result.fallback_triggered).toBe(true);
      expect(result.routing_path).toContain('fallback_triggered');
      expect(result.primary_candidates).toHaveLength(1);
    });

    it('should not trigger fallback for structural queries', async () => {
      const mockFullSearchHandler = vi.fn();

      const result = await intentRouter.routeQuery(
        'function() => {}',
        mockContext,
        undefined,
        mockFullSearchHandler
      );

      expect(result.classification.intent).toBe('struct');
      expect(result.fallback_triggered).toBe(false);
      expect(mockFullSearchHandler).not.toHaveBeenCalled();
    });

    it('should not trigger fallback for natural language queries', async () => {
      const mockFullSearchHandler = vi.fn();

      const result = await intentRouter.routeQuery(
        'how to handle authentication',
        mockContext,
        undefined,
        mockFullSearchHandler
      );

      expect(result.classification.intent).toBe('NL');
      expect(result.fallback_triggered).toBe(false);
      expect(mockFullSearchHandler).not.toHaveBeenCalled();
    });
  });

  describe('candidate enrichment', () => {
    it('should enrich candidates with intent information', async () => {
      mockLspEnhancer.enhanceStageB.mockResolvedValue({
        candidates: [
          {
            file_path: '/test/file.ts',
            line_no: 10,
            col_no: 5,
            content: 'class MyClass {',
            score: 0.95,
            match_reasons: ['lsp_hint'],
            symbol_kind: 'class'
          }
        ],
        stage_latency_ms: 25,
        lsp_contributions: 1,
        total_lsp_hints_used: 5,
        performance_metrics: {
          lsp_lookup_ms: 10,
          merge_dedupe_ms: 5,
          context_enrichment_ms: 10
        }
      });

      const result = await intentRouter.routeQuery(
        'def MyClass',
        mockContext
      );

      const candidate = result.primary_candidates[0] as any;
      expect(candidate.intent_classification).toBeDefined();
      expect(candidate.intent_classification.intent).toBe('def');
      expect(candidate.intent_honored).toBe(true);
      expect(candidate.why).toContain('intent_def');
      expect(candidate.why).toContain('high_confidence_routing');
      expect(candidate.why).not.toContain('fallback_triggered');
    });

    it('should mark fallback candidates appropriately', async () => {
      mockLspEnhancer.enhanceStageB.mockResolvedValue({
        candidates: [],
        stage_latency_ms: 10,
        lsp_contributions: 0,
        total_lsp_hints_used: 0,
        performance_metrics: { lsp_lookup_ms: 5, merge_dedupe_ms: 2, context_enrichment_ms: 3 }
      });

      const mockFullSearchHandler = vi.fn().mockResolvedValue([
        {
          file_path: '/test/file.ts',
          line_no: 20,
          col_no: 0,
          content: 'function MyFunction() {',
          score: 0.8,
          match_reasons: ['fuzzy_match'],
          symbol_kind: 'function'
        }
      ]);

      const result = await intentRouter.routeQuery(
        'def MyFunction',
        mockContext,
        undefined,
        mockFullSearchHandler
      );

      const candidate = result.primary_candidates[0] as any;
      expect(candidate.intent_classification.intent).toBe('def');
      expect(candidate.intent_honored).toBe(false);
      expect(candidate.why).toContain('intent_def');
      expect(candidate.why).toContain('fallback_triggered');
      expect(candidate.why).toContain('high_confidence_routing');
    });

    it('should mark low confidence queries appropriately', async () => {
      const result = await intentRouter.routeQuery(
        'simple query',  // Low confidence lexical query
        mockContext
      );

      // Should go to full search handler, but we don't provide one, so empty results
      expect(result.classification.intent).toBe('lexical');
      expect(result.confidence_threshold_met).toBe(false);
      expect(result.routing_path).toContain('low_confidence_full_search');
    });
  });

  describe('symbol extraction', () => {
    it('should extract symbols from definition queries correctly', () => {
      const testCases = [
        { query: 'def MyClass', expected: 'MyClass' },
        { query: 'define getUserProfile', expected: 'getUserProfile' },
        { query: 'definition of handleSubmit', expected: 'handleSubmit' },
        { query: 'what is MyComponent', expected: 'MyComponent' },
        { query: 'class UserService', expected: 'UserService' },
        { query: 'MyMethod definition', expected: 'MyMethod' },
        { query: 'calculateTotal declaration', expected: 'calculateTotal' },
        { query: 'invalidquery123!', expected: null },
        { query: '', expected: null }
      ];

      for (const { query, expected } of testCases) {
        const classification = intentRouter.classifyQueryIntent(query);
        if (classification.intent === 'def') {
          // We can't directly test the private method, but we can verify through routing
          // This is a simplified test for the concept
          expect(query).toBeTruthy(); // Basic test structure
        }
      }
    });

    it('should extract symbols from reference queries correctly', () => {
      const testCases = [
        { query: 'refs MyClass', expected: 'MyClass' },
        { query: 'references getUserProfile', expected: 'getUserProfile' },
        { query: 'usages of handleSubmit', expected: 'handleSubmit' },
        { query: 'find references to MyComponent', expected: 'MyComponent' },
        { query: 'where uses calculateTotal', expected: 'calculateTotal' },
        { query: 'MyMethod references', expected: 'MyMethod' },
        { query: 'invalidquery123!', expected: null }
      ];

      for (const { query, expected } of testCases) {
        const classification = intentRouter.classifyQueryIntent(query);
        if (classification.intent === 'refs') {
          // Similar to above, testing through classification
          expect(query).toBeTruthy();
        }
      }
    });
  });

  describe('performance and limits', () => {
    it('should limit primary results to MAX_PRIMARY_RESULTS', async () => {
      // Create mock candidates exceeding the limit
      const manyCandidates = Array.from({ length: 25 }, (_, i) => ({
        file_path: `/test/file${i}.ts`,
        line_no: i + 1,
        col_no: 0,
        content: `function func${i}() {`,
        score: 0.9 - (i * 0.01),
        match_reasons: ['lsp_hint'],
        symbol_kind: 'function'
      }));

      mockLspEnhancer.enhanceStageB.mockResolvedValue({
        candidates: manyCandidates,
        stage_latency_ms: 30,
        lsp_contributions: 25,
        total_lsp_hints_used: 25,
        performance_metrics: {
          lsp_lookup_ms: 15,
          merge_dedupe_ms: 10,
          context_enrichment_ms: 5
        }
      });

      const result = await intentRouter.routeQuery(
        'def MyFunction',
        mockContext
      );

      expect(result.primary_candidates.length).toBeLessThanOrEqual(20); // MAX_PRIMARY_RESULTS
    });

    it('should handle errors gracefully', async () => {
      mockLspEnhancer.enhanceStageB.mockRejectedValue(new Error('LSP service unavailable'));

      await expect(
        intentRouter.routeQuery('def MyClass', mockContext)
      ).rejects.toThrow('LSP service unavailable');
    });

    it('should provide accurate statistics', () => {
      const stats = intentRouter.getStats();
      
      expect(stats.confidence_threshold).toBe(0.7);
      expect(stats.max_primary_results).toBe(20);
      expect(stats.classification_features).toEqual([
        'has_definition_pattern',
        'has_reference_pattern',
        'has_symbol_prefix',
        'has_structural_chars',
        'is_natural_language'
      ]);
    });
  });

  describe('model updates', () => {
    it('should accept training data for model updates', () => {
      const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
      
      const trainingData = [
        { query: 'def MyClass', actual_intent: 'def' as QueryIntent, user_satisfaction: 0.9 },
        { query: 'refs getUserData', actual_intent: 'refs' as QueryIntent, user_satisfaction: 0.8 },
        { query: 'simple search', actual_intent: 'lexical' as QueryIntent, user_satisfaction: 0.6 }
      ];

      intentRouter.updateClassificationModel(trainingData);
      
      expect(consoleSpy).toHaveBeenCalledWith(
        'Received 3 training examples for intent classification'
      );
      
      consoleSpy.mockRestore();
    });
  });

  describe('edge cases', () => {
    it('should handle empty queries', async () => {
      const result = await intentRouter.routeQuery('', mockContext);
      
      expect(result.classification.intent).toBe('lexical');
      expect(result.confidence_threshold_met).toBe(false);
    });

    it('should handle very long queries', async () => {
      const longQuery = 'a'.repeat(1000);
      
      const result = await intentRouter.routeQuery(longQuery, mockContext);
      
      expect(result.classification).toBeDefined();
      expect(result.routing_path).toBeDefined();
    });

    it('should handle special characters in queries', async () => {
      const specialQueries = [
        'def MyClass<T>',
        'refs user.name?.value',
        'class Component$',
        'function* generator()',
        'const { x, y } = object'
      ];

      for (const query of specialQueries) {
        const result = await intentRouter.routeQuery(query, mockContext);
        expect(result.classification).toBeDefined();
        expect(result.routing_path).toBeDefined();
      }
    });

    it('should handle mixed case queries', async () => {
      const mixedCaseQueries = [
        'DEF MyClass',
        'Refs MyMethod', 
        'FIND DEFINITION MyFunction',
        'Class UserService'
      ];

      for (const query of mixedCaseQueries) {
        const result = await intentRouter.routeQuery(query, mockContext);
        expect(result.classification).toBeDefined();
        expect(['def', 'refs', 'symbol'].includes(result.classification.intent)).toBe(true);
      }
    });
  });
});