/**
 * Unit tests for Query Classification
 * Tests natural language vs keyword query detection for Stage-C
 */

import { describe, it, expect } from 'vitest';
import { 
  classifyQuery, 
  shouldApplySemanticReranking, 
  explainSemanticDecision 
} from '../../src/core/query-classifier.js';

describe('Query Classification', () => {
  describe('classifyQuery', () => {
    it('should identify natural language queries', () => {
      const naturalLanguageQueries = [
        'find authentication logic',
        'show me the login function',
        'where is the user validation code',
        'get all database connection methods',
        'locate the error handling for api calls',
        'what are the utility functions for string processing',
        'how to calculate the sum of numbers',
        'search for functions that handle file uploads'
      ];

      naturalLanguageQueries.forEach(query => {
        const result = classifyQuery(query);
        expect(result.isNaturalLanguage).toBe(true);
        expect(result.confidence).toBeGreaterThan(0.5);
        expect(result.characteristics.length).toBeGreaterThan(0);
      });
    });

    it('should identify keyword/programming queries', () => {
      const keywordQueries = [
        'def login',
        'class User',
        'function calculateSum',
        'const API_KEY',
        'import React',
        'user.save()',
        'if error',
        'for loop',
        'try catch',
        'calculateSum(a, b)',
        '{user: "admin"}',
        'arr.filter(x => x > 0)'
      ];

      keywordQueries.forEach(query => {
        const result = classifyQuery(query);
        expect(result.isNaturalLanguage).toBe(false);
        expect(result.confidence).toBeLessThanOrEqual(0.5);
        expect(result.characteristics).toContain('has_programming_syntax');
      });
    });

    it('should detect specific natural language characteristics', () => {
      const queries = [
        { query: 'find the user login function', expected: ['has_articles', 'has_descriptive_words'] },
        { query: 'search for authentication in the codebase', expected: ['has_prepositions', 'has_descriptive_words'] },
        { query: 'what functions handle user registration', expected: ['has_questions'] },
        { query: 'show me all utility functions for data processing', expected: ['has_descriptive_words', 'has_prepositions', 'has_multiple_words'] }
      ];

      queries.forEach(({ query, expected }) => {
        const result = classifyQuery(query);
        expected.forEach(characteristic => {
          expect(result.characteristics).toContain(characteristic);
        });
      });
    });

    it('should handle edge cases', () => {
      const edgeCases = [
        { query: '', expected: false },
        { query: '   ', expected: false },
        { query: 'a', expected: false },
        { query: 'the', expected: true }, // Single article
        { query: 'find', expected: true }, // Single descriptive word
      ];

      edgeCases.forEach(({ query, expected }) => {
        const result = classifyQuery(query);
        expect(result.isNaturalLanguage).toBe(expected);
      });
    });
  });

  describe('shouldApplySemanticReranking', () => {
    it('should apply for natural language queries with sufficient candidates', () => {
      const shouldApply = shouldApplySemanticReranking(
        'find authentication logic in the user service',
        25,
        'hybrid'
      );
      expect(shouldApply).toBe(true);
    });

    it('should not apply for keyword queries', () => {
      const shouldApply = shouldApplySemanticReranking(
        'def authenticate',
        25,
        'hybrid'
      );
      expect(shouldApply).toBe(false);
    });

    it('should not apply for insufficient candidates', () => {
      const shouldApply = shouldApplySemanticReranking(
        'find authentication logic',
        5,
        'hybrid'
      );
      expect(shouldApply).toBe(false);
    });

    it('should not apply for non-hybrid mode', () => {
      const shouldApply = shouldApplySemanticReranking(
        'find authentication logic',
        25,
        'struct'
      );
      expect(shouldApply).toBe(false);
    });

    it('should not apply for too many candidates (performance limit)', () => {
      const shouldApply = shouldApplySemanticReranking(
        'find authentication logic',
        250,
        'hybrid'
      );
      expect(shouldApply).toBe(false);
    });
  });

  describe('explainSemanticDecision', () => {
    it('should explain why semantic reranking was applied', () => {
      const explanation = explainSemanticDecision(
        'find authentication logic in the user service',
        25,
        'hybrid'
      );
      expect(explanation).toContain('Semantic reranking applied');
      expect(explanation).toContain('natural language query');
    });

    it('should explain why semantic reranking was skipped for keyword queries', () => {
      const explanation = explainSemanticDecision(
        'def authenticate',
        25,
        'hybrid'
      );
      expect(explanation).toContain('Semantic reranking skipped');
      expect(explanation).toContain('keyword query detected');
    });

    it('should explain why semantic reranking was skipped for insufficient candidates', () => {
      const explanation = explainSemanticDecision(
        'find authentication logic',
        5,
        'hybrid'
      );
      expect(explanation).toContain('Semantic reranking skipped');
      expect(explanation).toContain('need ≥10');
    });

    it('should explain why semantic reranking was skipped for wrong mode', () => {
      const explanation = explainSemanticDecision(
        'find authentication logic',
        25,
        'struct'
      );
      expect(explanation).toContain('Semantic reranking skipped');
      expect(explanation).toContain("mode is 'struct'");
    });

    it('should explain why semantic reranking was skipped for too many candidates', () => {
      const explanation = explainSemanticDecision(
        'find authentication logic',
        250,
        'hybrid'
      );
      expect(explanation).toContain('Semantic reranking skipped');
      expect(explanation).toContain('exceed performance limit');
    });
  });

  describe('Performance and Edge Cases', () => {
    it('should handle mixed queries appropriately', () => {
      const mixedQueries = [
        { query: 'find function calculateSum', expected: true }, // Natural + programming
        { query: 'class User for authentication', expected: true }, // Programming + natural
        { query: 'def login() method', expected: false }, // Mostly programming
        { query: 'search for const variables', expected: true }, // Natural + programming term
      ];

      mixedQueries.forEach(({ query, expected }) => {
        const result = classifyQuery(query);
        expect(result.isNaturalLanguage).toBe(expected);
      });
    });

    it('should be case-insensitive', () => {
      const queries = [
        'FIND THE LOGIN FUNCTION',
        'Find The Login Function',
        'find the login function'
      ];

      queries.forEach(query => {
        const result = classifyQuery(query);
        expect(result.isNaturalLanguage).toBe(true);
      });
    });

    it('should handle different query lengths', () => {
      const queries = [
        { query: 'find auth', length: 2, expectNL: true },
        { query: 'find authentication logic', length: 3, expectNL: true },
        { query: 'find authentication logic in user service module', length: 7, expectNL: true },
        { query: 'user', length: 1, expectNL: false },
      ];

      queries.forEach(({ query, expectNL }) => {
        const result = classifyQuery(query);
        expect(result.isNaturalLanguage).toBe(expectNL);
      });
    });
  });
});