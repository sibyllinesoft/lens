/**
 * Unit Tests for API Types and Schemas
 * Tests Zod validation schemas for API contracts
 */

import { describe, it, expect } from 'bun:test';
import {
  ApiVersionSchema,
  IndexVersionSchema,
  PolicyVersionSchema,
  SearchRequestSchema,
  SearchHitSchema,
  type SearchRequest,
  type SearchHit,
  type ApiVersion,
  type IndexVersion,
  type PolicyVersion,
} from '../api.js';

describe('API Version Schemas', () => {
  describe('ApiVersionSchema', () => {
    it('should accept valid v1 version', () => {
      const result = ApiVersionSchema.parse('v1');
      expect(result).toBe('v1');
    });

    it('should reject invalid versions', () => {
      expect(() => ApiVersionSchema.parse('v2')).toThrow();
      expect(() => ApiVersionSchema.parse('1')).toThrow();
      expect(() => ApiVersionSchema.parse('')).toThrow();
    });
  });

  describe('IndexVersionSchema', () => {
    it('should accept valid v1 version', () => {
      const result = IndexVersionSchema.parse('v1');
      expect(result).toBe('v1');
    });

    it('should reject invalid versions', () => {
      expect(() => IndexVersionSchema.parse('v2')).toThrow();
      expect(() => IndexVersionSchema.parse('invalid')).toThrow();
    });
  });

  describe('PolicyVersionSchema', () => {
    it('should accept valid v1 version', () => {
      const result = PolicyVersionSchema.parse('v1');
      expect(result).toBe('v1');
    });

    it('should reject invalid versions', () => {
      expect(() => PolicyVersionSchema.parse('v0')).toThrow();
      expect(() => PolicyVersionSchema.parse(null)).toThrow();
    });
  });
});

describe('SearchRequestSchema', () => {
  const validSearchRequest: SearchRequest = {
    repo_sha: 'abc123def456',
    q: 'function search',
    mode: 'hybrid',
    fuzzy: 1,
    k: 50,
  };

  describe('Valid Requests', () => {
    it('should accept valid search request with all required fields', () => {
      const result = SearchRequestSchema.parse(validSearchRequest);
      expect(result).toEqual(validSearchRequest);
    });

    it('should accept valid search request with optional timeout', () => {
      const requestWithTimeout = {
        ...validSearchRequest,
        timeout_ms: 2000,
      };
      const result = SearchRequestSchema.parse(requestWithTimeout);
      expect(result).toEqual(requestWithTimeout);
    });

    it('should accept different valid modes', () => {
      expect(() => SearchRequestSchema.parse({ ...validSearchRequest, mode: 'lex' })).not.toThrow();
      expect(() => SearchRequestSchema.parse({ ...validSearchRequest, mode: 'struct' })).not.toThrow();
      expect(() => SearchRequestSchema.parse({ ...validSearchRequest, mode: 'hybrid' })).not.toThrow();
    });

    it('should accept valid fuzzy values', () => {
      expect(() => SearchRequestSchema.parse({ ...validSearchRequest, fuzzy: 0 })).not.toThrow();
      expect(() => SearchRequestSchema.parse({ ...validSearchRequest, fuzzy: 1 })).not.toThrow();
      expect(() => SearchRequestSchema.parse({ ...validSearchRequest, fuzzy: 2 })).not.toThrow();
    });

    it('should accept valid k values', () => {
      expect(() => SearchRequestSchema.parse({ ...validSearchRequest, k: 1 })).not.toThrow();
      expect(() => SearchRequestSchema.parse({ ...validSearchRequest, k: 100 })).not.toThrow();
      expect(() => SearchRequestSchema.parse({ ...validSearchRequest, k: 200 })).not.toThrow();
    });
  });

  describe('Invalid Requests', () => {
    it('should reject missing required fields', () => {
      expect(() => SearchRequestSchema.parse({})).toThrow();
      expect(() => SearchRequestSchema.parse({ repo_sha: 'test' })).toThrow();
      expect(() => SearchRequestSchema.parse({ q: 'test' })).toThrow();
    });

    it('should reject invalid repo_sha values', () => {
      expect(() => SearchRequestSchema.parse({ ...validSearchRequest, repo_sha: '' })).toThrow();
      expect(() => SearchRequestSchema.parse({ ...validSearchRequest, repo_sha: 'a'.repeat(65) })).toThrow();
    });

    it('should reject invalid query values', () => {
      expect(() => SearchRequestSchema.parse({ ...validSearchRequest, q: '' })).toThrow();
      expect(() => SearchRequestSchema.parse({ ...validSearchRequest, q: 'a'.repeat(1001) })).toThrow();
    });

    it('should reject invalid mode values', () => {
      expect(() => SearchRequestSchema.parse({ ...validSearchRequest, mode: 'invalid' as any })).toThrow();
      expect(() => SearchRequestSchema.parse({ ...validSearchRequest, mode: 'semantic' as any })).toThrow();
    });

    it('should reject invalid fuzzy values', () => {
      expect(() => SearchRequestSchema.parse({ ...validSearchRequest, fuzzy: -1 })).toThrow();
      expect(() => SearchRequestSchema.parse({ ...validSearchRequest, fuzzy: 3 })).toThrow();
      expect(() => SearchRequestSchema.parse({ ...validSearchRequest, fuzzy: 1.5 })).toThrow();
    });

    it('should reject invalid k values', () => {
      expect(() => SearchRequestSchema.parse({ ...validSearchRequest, k: 0 })).toThrow();
      expect(() => SearchRequestSchema.parse({ ...validSearchRequest, k: 201 })).toThrow();
      expect(() => SearchRequestSchema.parse({ ...validSearchRequest, k: -10 })).toThrow();
    });

    it('should reject invalid timeout values', () => {
      expect(() => SearchRequestSchema.parse({ ...validSearchRequest, timeout_ms: 99 })).toThrow();
      expect(() => SearchRequestSchema.parse({ ...validSearchRequest, timeout_ms: 5001 })).toThrow();
      expect(() => SearchRequestSchema.parse({ ...validSearchRequest, timeout_ms: -100 })).toThrow();
    });
  });

  describe('Type Inference', () => {
    it('should correctly infer TypeScript types', () => {
      const request: SearchRequest = {
        repo_sha: 'abc123',
        q: 'test query',
        mode: 'lex',
        fuzzy: 2,
        k: 10,
        timeout_ms: 1000,
      };

      expect(request.mode).toBe('lex');
      expect(typeof request.fuzzy).toBe('number');
      expect(typeof request.k).toBe('number');
    });
  });
});

describe('SearchHitSchema', () => {
  const validSearchHit: SearchHit = {
    file: 'src/index.ts',
    line: 42,
    col: 0,
    score: 0.85,
    why: ['exact', 'symbol'],
  };

  describe('Valid Hits', () => {
    it('should accept valid search hit with required fields', () => {
      const result = SearchHitSchema.parse(validSearchHit);
      expect(result).toEqual(validSearchHit);
    });

    it('should accept search hit with all optional fields', () => {
      const fullHit: SearchHit = {
        ...validSearchHit,
        lang: 'typescript',
        snippet: 'function test() { return 42; }',
        ast_path: 'root.function_def[0]',
        symbol_kind: 'function',
        byte_offset: 1024,
        span_len: 15,
        context_before: 'export ',
        context_after: ' { ... }',
        pattern_type: 'function_def',
        symbol_name: 'testFunction',
        signature: 'function testFunction(): number',
      };

      const result = SearchHitSchema.parse(fullHit);
      expect(result).toEqual(fullHit);
    });

    it('should accept different valid symbol kinds', () => {
      const validSymbolKinds = [
        'function', 'class', 'variable', 'type', 'interface',
        'constant', 'enum', 'method', 'property'
      ];

      validSymbolKinds.forEach(kind => {
        expect(() => SearchHitSchema.parse({
          ...validSearchHit,
          symbol_kind: kind as any
        })).not.toThrow();
      });
    });

    it('should accept different valid pattern types', () => {
      const validPatternTypes = [
        'function_def', 'class_def', 'import', 'async_def',
        'decorator', 'try_except', 'for_loop', 'if_statement'
      ];

      validPatternTypes.forEach(pattern => {
        expect(() => SearchHitSchema.parse({
          ...validSearchHit,
          pattern_type: pattern as any
        })).not.toThrow();
      });
    });

    it('should accept different valid match reasons', () => {
      const validWhy = [
        'exact', 'fuzzy', 'symbol', 'struct', 
        'structural', 'semantic', 'subtoken'
      ];

      validWhy.forEach(reason => {
        expect(() => SearchHitSchema.parse({
          ...validSearchHit,
          why: [reason as any]
        })).not.toThrow();
      });
    });

    it('should accept multiple match reasons', () => {
      const result = SearchHitSchema.parse({
        ...validSearchHit,
        why: ['exact', 'fuzzy', 'symbol', 'semantic']
      });
      expect(result.why).toHaveLength(4);
    });
  });

  describe('Invalid Hits', () => {
    it('should reject missing required fields', () => {
      expect(() => SearchHitSchema.parse({})).toThrow();
      expect(() => SearchHitSchema.parse({ file: 'test.ts' })).toThrow();
      expect(() => SearchHitSchema.parse({ file: 'test.ts', line: 1 })).toThrow();
    });

    it('should reject invalid file values', () => {
      expect(() => SearchHitSchema.parse({ ...validSearchHit, file: '' })).toThrow();
    });

    it('should reject invalid line numbers', () => {
      expect(() => SearchHitSchema.parse({ ...validSearchHit, line: 0 })).toThrow();
      expect(() => SearchHitSchema.parse({ ...validSearchHit, line: -1 })).toThrow();
      expect(() => SearchHitSchema.parse({ ...validSearchHit, line: 1.5 })).toThrow();
    });

    it('should reject invalid column numbers', () => {
      expect(() => SearchHitSchema.parse({ ...validSearchHit, col: -1 })).toThrow();
      expect(() => SearchHitSchema.parse({ ...validSearchHit, col: 1.5 })).toThrow();
    });

    it('should reject invalid score values', () => {
      expect(() => SearchHitSchema.parse({ ...validSearchHit, score: -0.1 })).toThrow();
      expect(() => SearchHitSchema.parse({ ...validSearchHit, score: 1.1 })).toThrow();
    });

    it('should reject invalid match reasons', () => {
      expect(() => SearchHitSchema.parse({
        ...validSearchHit,
        why: ['invalid_reason' as any]
      })).toThrow();
    });

    it('should reject invalid symbol kinds', () => {
      expect(() => SearchHitSchema.parse({
        ...validSearchHit,
        symbol_kind: 'invalid_kind' as any
      })).toThrow();
    });

    it('should reject invalid pattern types', () => {
      expect(() => SearchHitSchema.parse({
        ...validSearchHit,
        pattern_type: 'invalid_pattern' as any
      })).toThrow();
    });

    it('should reject invalid byte offset values', () => {
      expect(() => SearchHitSchema.parse({
        ...validSearchHit,
        byte_offset: -1
      })).toThrow();
    });

    it('should reject invalid span length values', () => {
      expect(() => SearchHitSchema.parse({
        ...validSearchHit,
        span_len: -1
      })).toThrow();
    });
  });

  describe('Edge Cases', () => {
    it('should accept minimum valid values', () => {
      const minimalHit = {
        file: 'a',
        line: 1,
        col: 0,
        score: 0,
        why: ['exact'] as const,
      };

      expect(() => SearchHitSchema.parse(minimalHit)).not.toThrow();
    });

    it('should accept maximum valid values', () => {
      const maximalHit = {
        file: 'very-long-file-path-that-is-still-valid.ts',
        line: 999999,
        col: 999999,
        score: 1.0,
        why: ['exact', 'fuzzy', 'symbol', 'struct', 'structural', 'semantic', 'subtoken'] as const,
      };

      expect(() => SearchHitSchema.parse(maximalHit)).not.toThrow();
    });

    it('should handle empty why array', () => {
      expect(() => SearchHitSchema.parse({
        ...validSearchHit,
        why: []
      })).not.toThrow();
    });
  });
});

describe('Type Safety', () => {
  it('should provide compile-time type safety for API versions', () => {
    const apiVersion: ApiVersion = 'v1';
    const indexVersion: IndexVersion = 'v1'; 
    const policyVersion: PolicyVersion = 'v1';

    expect(apiVersion).toBe('v1');
    expect(indexVersion).toBe('v1');
    expect(policyVersion).toBe('v1');
  });

  it('should provide compile-time type safety for search requests', () => {
    const request: SearchRequest = {
      repo_sha: 'test123',
      q: 'search query',
      mode: 'hybrid',
      fuzzy: 1,
      k: 50,
      timeout_ms: 2000,
    };

    expect(request.mode).toBe('hybrid');
    expect(request.fuzzy).toBe(1);
    expect(request.k).toBe(50);
  });

  it('should provide compile-time type safety for search hits', () => {
    const hit: SearchHit = {
      file: 'test.ts',
      line: 10,
      col: 5,
      score: 0.9,
      why: ['exact', 'symbol'],
      symbol_kind: 'function',
      pattern_type: 'function_def',
    };

    expect(hit.symbol_kind).toBe('function');
    expect(hit.pattern_type).toBe('function_def');
    expect(hit.why).toContain('exact');
  });
});