/**
 * Integration tests for server components - exercises API schemas, validation, and business logic
 * These tests import and run real implementation code to achieve measurable coverage
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { tmpdir } from 'os';
import { join } from 'path';
import { mkdtemp, rm } from 'fs/promises';

// Import schemas and types directly to test validation logic
import {
  SearchRequestSchema,
  SearchResponseSchema,
  SpiSearchRequestSchema,
  SpiSearchResponseSchema,
  HealthResponseSchema,
  ResolveRequestSchema,
  ResolveResponseSchema,
  ContextRequestSchema,
  ContextResponseSchema
} from '../../types/api.js';

describe('Server Components Integration Tests', () => {
  let tempDir: string;

  beforeAll(async () => {
    // Create temporary directory for test data
    tempDir = await mkdtemp(join(tmpdir(), 'lens-server-test-'));
  });

  afterAll(async () => {
    if (tempDir) {
      await rm(tempDir, { recursive: true, force: true });
    }
  });

  describe('API Schema Validation', () => {
    it('should validate SearchRequestSchema with valid data', () => {
      const validSearchRequest = {
        query: 'test function',
        q: 'test function', // Required field
        num_results: 10,
        language: 'typescript',
        repo_sha: 'abc123def456', // Required field
        mode: 'hybrid', // Required field: 'lex' | 'struct' | 'hybrid'
        fuzzy: 1, // Required field: number
        k: 10, // Required field: number
        context: {
          repo_name: 'test-repo',
          file_path: 'src/test.ts',
          line_number: 1,
          column_number: 1
        }
      };

      const result = SearchRequestSchema.safeParse(validSearchRequest);
      expect(result.success).toBe(true);
      
      if (result.success) {
        expect(result.data.q).toBe('test function');
        expect(result.data.repo_sha).toBe('abc123def456');
        expect(result.data.mode).toBe('hybrid');
        expect(result.data.fuzzy).toBe(1);
        expect(result.data.k).toBe(10);
      }
    });

    it('should reject invalid SearchRequestSchema data', () => {
      const invalidSearchRequest = {
        query: '', // Empty query should fail
        num_results: -1, // Negative results should fail
        language: 'invalid-language'
      };

      const result = SearchRequestSchema.safeParse(invalidSearchRequest);
      expect(result.success).toBe(false);
      
      if (!result.success) {
        expect(result.error.issues.length).toBeGreaterThan(0);
      }
    });

    it('should validate SpiSearchRequestSchema', () => {
      const validSpiRequest = {
        query: 'test',
        max_results: 5,
        repo: 'test-repo',
        language: 'typescript'
      };

      const result = SpiSearchRequestSchema.safeParse(validSpiRequest);
      expect(result.success).toBe(true);
      
      if (result.success) {
        expect(result.data.query).toBe('test');
        expect(result.data.max_results).toBe(5);
        expect(result.data.repo).toBe('test-repo');
        expect(result.data.language).toBe('typescript');
      }
    });

    it('should validate ResolveRequestSchema', () => {
      const validResolveRequest = {
        symbol: 'TestClass',
        context: {
          repo_name: 'test-repo',
          file_path: 'src/test.ts',
          line_number: 10,
          column_number: 5
        },
        language: 'typescript'
      };

      const result = ResolveRequestSchema.safeParse(validResolveRequest);
      expect(result.success).toBe(true);
      
      if (result.success) {
        expect(result.data.symbol).toBe('TestClass');
        expect(result.data.language).toBe('typescript');
        expect(result.data.context.line_number).toBe(10);
      }
    });

    it('should validate ContextRequestSchema', () => {
      const validContextRequest = {
        symbol: 'TestFunction',
        context: {
          repo_name: 'test-repo',
          file_path: 'src/test.ts',
          line_number: 15,
          column_number: 10
        },
        language: 'typescript',
        max_results: 10
      };

      const result = ContextRequestSchema.safeParse(validContextRequest);
      expect(result.success).toBe(true);
      
      if (result.success) {
        expect(result.data.symbol).toBe('TestFunction');
        expect(result.data.max_results).toBe(10);
      }
    });
  });

  describe('Response Schema Validation', () => {
    it('should validate SearchResponseSchema structure', () => {
      const validSearchResponse = {
        hits: [
          {
            file_path: 'src/example.ts',
            line_number: 10,
            column_number: 5,
            score: 0.95,
            snippet: 'export function testFunction() {',
            reason: 'exact',
            context_before: 'import { test } from "./test";',
            context_after: 'return "test";'
          }
        ],
        total_results: 1,
        query: 'testFunction',
        language: 'typescript'
      };

      const result = SearchResponseSchema.safeParse(validSearchResponse);
      expect(result.success).toBe(true);
      
      if (result.success) {
        expect(result.data.hits).toHaveLength(1);
        expect(result.data.hits[0].score).toBe(0.95);
        expect(result.data.hits[0].reason).toBe('exact');
        expect(result.data.total_results).toBe(1);
      }
    });

    it('should validate SpiSearchResponseSchema structure', () => {
      const validSpiResponse = {
        results: [
          {
            file: 'src/utils.ts',
            line: 20,
            column: 8,
            confidence: 0.88,
            preview: 'function utilityHelper()',
            match_type: 'symbol'
          }
        ],
        metadata: {
          query_time_ms: 15,
          total_matches: 1,
          engine_version: '1.0.0'
        }
      };

      const result = SpiSearchResponseSchema.safeParse(validSpiResponse);
      expect(result.success).toBe(true);
      
      if (result.success) {
        expect(result.data.results).toHaveLength(1);
        expect(result.data.results[0].confidence).toBe(0.88);
        expect(result.data.metadata.query_time_ms).toBe(15);
      }
    });

    it('should validate HealthResponseSchema structure', () => {
      const validHealthResponse = {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        uptime: 3600.5,
        version: '1.0.0-rc.2',
        components: {
          search_engine: 'healthy',
          index_registry: 'healthy',
          lsp_service: 'healthy'
        }
      };

      const result = HealthResponseSchema.safeParse(validHealthResponse);
      expect(result.success).toBe(true);
      
      if (result.success) {
        expect(result.data.status).toBe('healthy');
        expect(typeof result.data.uptime).toBe('number');
        expect(result.data.components).toHaveProperty('search_engine');
      }
    });

    it('should validate ResolveResponseSchema structure', () => {
      const validResolveResponse = {
        definition: {
          file_path: 'src/models/user.ts',
          line_number: 5,
          column_number: 13,
          symbol_name: 'User',
          symbol_type: 'interface'
        },
        references: [
          {
            file_path: 'src/services/user.service.ts',
            line_number: 15,
            column_number: 25,
            usage_type: 'type_annotation'
          }
        ]
      };

      const result = ResolveResponseSchema.safeParse(validResolveResponse);
      expect(result.success).toBe(true);
      
      if (result.success) {
        expect(result.data.definition.symbol_name).toBe('User');
        expect(result.data.references).toHaveLength(1);
        expect(result.data.references[0].usage_type).toBe('type_annotation');
      }
    });

    it('should validate ContextResponseSchema structure', () => {
      const validContextResponse = {
        references: [
          {
            file_path: 'src/components/header.tsx',
            line_number: 8,
            column_number: 12,
            context: 'import { Header } from "./header";',
            reference_type: 'import'
          },
          {
            file_path: 'src/pages/home.tsx',
            line_number: 25,
            column_number: 15,
            context: '<Header title="Home" />',
            reference_type: 'usage'
          }
        ],
        total_references: 2
      };

      const result = ContextResponseSchema.safeParse(validContextResponse);
      expect(result.success).toBe(true);
      
      if (result.success) {
        expect(result.data.references).toHaveLength(2);
        expect(result.data.total_references).toBe(2);
        expect(result.data.references[0].reference_type).toBe('import');
        expect(result.data.references[1].reference_type).toBe('usage');
      }
    });
  });

  describe('Schema Edge Cases and Error Handling', () => {
    it('should handle optional fields in SearchRequest', () => {
      const requestWithOptionals = {
        query: 'test',
        num_results: 10,
        language: 'typescript',
        context: {
          repo_name: 'test-repo',
          file_path: 'test.ts',
          line_number: 1,
          column_number: 1
        },
        fuzzy: true,
        case_sensitive: false,
        context_aware: true,
        file_path_filter: '*.ts'
      };

      const result = SearchRequestSchema.safeParse(requestWithOptionals);
      expect(result.success).toBe(true);
      
      if (result.success) {
        expect(result.data.fuzzy).toBe(true);
        expect(result.data.case_sensitive).toBe(false);
        expect(result.data.context_aware).toBe(true);
        expect(result.data.file_path_filter).toBe('*.ts');
      }
    });

    it('should handle missing optional fields gracefully', () => {
      const minimalRequest = {
        query: 'test',
        num_results: 5,
        language: 'javascript',
        context: {
          repo_name: 'minimal-repo',
          file_path: 'index.js',
          line_number: 1,
          column_number: 1
        }
      };

      const result = SearchRequestSchema.safeParse(minimalRequest);
      expect(result.success).toBe(true);
      
      if (result.success) {
        expect(result.data.query).toBe('test');
        expect(result.data.fuzzy).toBeUndefined();
        expect(result.data.case_sensitive).toBeUndefined();
      }
    });

    it('should validate language enum values', () => {
      const supportedLanguages = ['typescript', 'javascript', 'python', 'go', 'rust', 'java'];
      const unsupportedLanguage = 'cobol';

      supportedLanguages.forEach(lang => {
        const request = {
          query: 'test',
          num_results: 5,
          language: lang,
          context: {
            repo_name: 'test',
            file_path: 'test.ts',
            line_number: 1,
            column_number: 1
          }
        };

        const result = SearchRequestSchema.safeParse(request);
        expect(result.success).toBe(true);
      });

      // Test unsupported language
      const invalidRequest = {
        query: 'test',
        num_results: 5,
        language: unsupportedLanguage,
        context: {
          repo_name: 'test',
          file_path: 'test.ts',
          line_number: 1,
          column_number: 1
        }
      };

      const invalidResult = SearchRequestSchema.safeParse(invalidRequest);
      expect(invalidResult.success).toBe(false);
    });

    it('should validate numeric constraints', () => {
      const testCases = [
        { num_results: 0, shouldPass: false }, // Zero results not allowed
        { num_results: 1, shouldPass: true },  // Minimum valid
        { num_results: 100, shouldPass: true }, // Normal range
        { num_results: -5, shouldPass: false }, // Negative not allowed
        { line_number: 0, shouldPass: false },  // Line numbers start at 1
        { line_number: 1, shouldPass: true },   // Minimum valid line
        { column_number: -1, shouldPass: false }, // Negative column not allowed
        { column_number: 0, shouldPass: true }    // Column 0 is valid
      ];

      testCases.forEach(({ num_results, line_number, column_number, shouldPass }) => {
        const request = {
          query: 'test',
          num_results: num_results || 10,
          language: 'typescript',
          context: {
            repo_name: 'test',
            file_path: 'test.ts',
            line_number: line_number || 1,
            column_number: column_number || 1
          }
        };

        const result = SearchRequestSchema.safeParse(request);
        expect(result.success).toBe(shouldPass);
      });
    });
  });

  describe('Complex Nested Structure Validation', () => {
    it('should validate deeply nested context structures', () => {
      const complexRequest = {
        query: 'complex test',
        num_results: 20,
        language: 'typescript',
        context: {
          repo_name: 'complex-repo',
          file_path: 'src/modules/auth/services/user.service.ts',
          line_number: 156,
          column_number: 23
        },
        fuzzy: true,
        case_sensitive: false,
        context_aware: true,
        file_path_filter: 'src/**/*.{ts,tsx}'
      };

      const result = SearchRequestSchema.safeParse(complexRequest);
      expect(result.success).toBe(true);
      
      if (result.success) {
        expect(result.data.context.file_path).toContain('auth/services');
        expect(result.data.context.line_number).toBe(156);
        expect(result.data.file_path_filter).toContain('**/*.{ts,tsx}');
      }
    });

    it('should validate complex search response with multiple hits', () => {
      const complexResponse = {
        hits: Array.from({ length: 15 }, (_, i) => ({
          file_path: `src/file${i}.ts`,
          line_number: 10 + i,
          column_number: 5,
          score: 0.9 - (i * 0.02),
          snippet: `export function test${i}() {`,
          reason: i % 2 === 0 ? 'exact' : 'fuzzy',
          context_before: `// Context before test${i}`,
          context_after: `return "result${i}";`
        })),
        total_results: 15,
        query: 'test functions',
        language: 'typescript'
      };

      const result = SearchResponseSchema.safeParse(complexResponse);
      expect(result.success).toBe(true);
      
      if (result.success) {
        expect(result.data.hits).toHaveLength(15);
        expect(result.data.hits[0].score).toBeCloseTo(0.9);
        expect(result.data.hits[14].score).toBeCloseTo(0.62);
        expect(result.data.total_results).toBe(15);
      }
    });
  });
});