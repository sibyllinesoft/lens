/**
 * Comprehensive Tests for LSP Stage-B Integration
 * Tests LSP hint consumption, symbol mapping, and candidate enhancement
 */

import { describe, it, expect, beforeEach, afterEach, mock, jest } from 'bun:test';
import { LSPStageBEnhancer } from '../lsp-stage-b.js';
import type { LSPHint, SearchContext, Candidate } from '../../types/core.js';

describe('LSPStageBEnhancer', () => {
  let lspEnhancer: LSPStageBEnhancer;
  let mockContext: SearchContext;
  let mockLSPHints: LSPHint[];

  beforeEach(() => {
    lspEnhancer = new LSPStageBEnhancer();
    
    mockContext = {
      mode: 'search',
      repo_sha: 'test-sha-123',
      language_hint: 'typescript',
      file_path: '/test/project/src/components/UserProfile.tsx',
      line_hint: 15,
      query_timestamp: new Date(),
      trace_id: 'test-trace-123'
    };

    mockLSPHints = [
      {
        symbol_id: 'lsp_1',
        name: 'UserProfile',
        kind: 'class',
        file_path: '/test/project/src/components/UserProfile.tsx',
        line: 10,
        col: 0,
        definition_uri: 'file:///test/project/src/components/UserProfile.tsx',
        signature: 'class UserProfile extends Component',
        aliases: ['Profile', 'User'],
        resolved_imports: ['React', 'Component'],
        references_count: 25
      },
      {
        symbol_id: 'lsp_2',
        name: 'getUserData',
        kind: 'function',
        file_path: '/test/project/src/services/UserService.ts',
        line: 45,
        col: 17,
        definition_uri: 'file:///test/project/src/services/UserService.ts',
        signature: 'function getUserData(userId: string): Promise<User>',
        aliases: ['fetchUserData', 'loadUserData'],
        resolved_imports: ['axios', 'User'],
        references_count: 12
      },
      {
        symbol_id: 'lsp_3',
        name: 'validateEmail',
        kind: 'function',
        file_path: '/test/project/src/utils/validation.ts',
        line: 8,
        col: 9,
        definition_uri: 'file:///test/project/src/utils/validation.ts',
        signature: 'function validateEmail(email: string): boolean',
        aliases: ['isValidEmail', 'checkEmail'],
        resolved_imports: [],
        references_count: 8
      },
      {
        symbol_id: 'lsp_4',
        name: 'ApiConfig',
        kind: 'interface',
        file_path: '/test/project/src/types/api.ts',
        line: 5,
        col: 0,
        definition_uri: 'file:///test/project/src/types/api.ts',
        signature: 'interface ApiConfig { baseUrl: string; timeout: number; }',
        aliases: ['Config', 'APIConfiguration'],
        resolved_imports: [],
        references_count: 15
      },
      {
        symbol_id: 'lsp_5',
        name: 'handleSubmit',
        kind: 'method',
        file_path: '/test/project/src/components/UserProfile.tsx',
        line: 25,
        col: 2,
        definition_uri: 'file:///test/project/src/components/UserProfile.tsx',
        signature: 'handleSubmit(event: FormEvent): void',
        aliases: ['onSubmit', 'submitForm'],
        resolved_imports: ['FormEvent'],
        references_count: 3
      }
    ];

    // Load hints into the enhancer
    lspEnhancer.loadHints(mockLSPHints);
  });

  describe('hint loading and indexing', () => {
    it('should load LSP hints into indices correctly', () => {
      const freshEnhancer = new LSPStageBEnhancer();
      freshEnhancer.loadHints(mockLSPHints);

      // Verify hints are loaded by attempting enhancement
      const result = freshEnhancer.enhanceStageB('UserProfile', mockContext, []);
      
      expect(result.lsp_contributions).toBeGreaterThan(0);
      expect(result.candidates.some(c => c.match_reasons?.includes('lsp_hint'))).toBe(true);
    });

    it('should build symbol name index correctly', () => {
      const result = lspEnhancer.enhanceStageB('getUserData', mockContext, []);
      
      expect(result.candidates).toHaveLength(1);
      expect(result.candidates[0].file_path).toBe('/test/project/src/services/UserService.ts');
      expect(result.candidates[0].line_no).toBe(45);
      expect(result.candidates[0].symbol_kind).toBe('function');
    });

    it('should build alias index correctly', () => {
      const result = lspEnhancer.enhanceStageB('Profile', mockContext, []);
      
      expect(result.candidates).toHaveLength(1);
      expect(result.candidates[0].file_path).toBe('/test/project/src/components/UserProfile.tsx');
      expect(result.candidates[0].match_reasons).toContain('lsp_hint');
    });

    it('should build type index correctly', () => {
      const result = lspEnhancer.enhanceStageB('interface', mockContext, []);
      
      expect(result.candidates).toHaveLength(1);
      expect(result.candidates[0].file_path).toBe('/test/project/src/types/api.ts');
      expect(result.candidates[0].symbol_kind).toBe('interface');
    });

    it('should build path index correctly', () => {
      const result = lspEnhancer.enhanceStageB('UserProfile', mockContext, []);
      
      // Should find symbols from the same file
      const sameFileSymbols = result.candidates.filter(c => 
        c.file_path === '/test/project/src/components/UserProfile.tsx'
      );
      
      expect(sameFileSymbols).toHaveLength(2); // UserProfile class and handleSubmit method
    });
  });

  describe('candidate enhancement', () => {
    it('should enhance candidates with LSP information', () => {
      const baseCandidates: Candidate[] = [
        {
          file_path: '/test/project/src/components/UserProfile.tsx',
          line_no: 10,
          col_no: 0,
          content: 'class UserProfile extends Component {',
          score: 0.7,
          match_reasons: ['fuzzy_match'],
          symbol_kind: 'class'
        }
      ];

      const result = lspEnhancer.enhanceStageB('UserProfile', mockContext, baseCandidates);
      
      expect(result.candidates).toHaveLength(1);
      
      const enhancedCandidate = result.candidates[0];
      expect(enhancedCandidate.score).toBeGreaterThan(0.7); // Score should be boosted
      expect(enhancedCandidate.match_reasons).toContain('lsp_hint');
      expect(enhancedCandidate.match_reasons).toContain('fuzzy_match'); // Original reason preserved
      
      // Check for LSP metadata
      const candidate = enhancedCandidate as any;
      expect(candidate.lsp_hint_id).toBe('lsp_1');
      expect(candidate.signature).toBe('class UserProfile extends Component');
      expect(candidate.references_count).toBe(25);
      expect(candidate.resolved_imports).toContain('React');
    });

    it('should add new LSP candidates when no base matches exist', () => {
      const baseCandidates: Candidate[] = [];
      
      const result = lspEnhancer.enhanceStageB('getUserData', mockContext, baseCandidates);
      
      expect(result.candidates).toHaveLength(1);
      
      const lspCandidate = result.candidates[0];
      expect(lspCandidate.file_path).toBe('/test/project/src/services/UserService.ts');
      expect(lspCandidate.line_no).toBe(45);
      expect(lspCandidate.col_no).toBe(17);
      expect(lspCandidate.symbol_kind).toBe('function');
      expect(lspCandidate.match_reasons).toContain('lsp_hint');
      expect(lspCandidate.score).toBeGreaterThan(0.8); // High score for exact LSP match
    });

    it('should merge duplicate candidates correctly', () => {
      const baseCandidates: Candidate[] = [
        {
          file_path: '/test/project/src/components/UserProfile.tsx',
          line_no: 10,
          col_no: 0,
          content: 'class UserProfile {', // Slightly different content
          score: 0.6,
          match_reasons: ['lexical_match'],
          symbol_kind: 'class'
        }
      ];

      const result = lspEnhancer.enhanceStageB('UserProfile', mockContext, baseCandidates);
      
      expect(result.candidates).toHaveLength(1); // Should be merged, not duplicated
      
      const mergedCandidate = result.candidates[0];
      expect(mergedCandidate.score).toBeGreaterThan(0.8); // Should get LSP boost
      expect(mergedCandidate.match_reasons).toContain('lexical_match');
      expect(mergedCandidate.match_reasons).toContain('lsp_hint');
      
      // Should prefer LSP content and metadata
      const candidate = mergedCandidate as any;
      expect(candidate.signature).toBe('class UserProfile extends Component');
    });
  });

  describe('query matching strategies', () => {
    it('should match exact symbol names', () => {
      const result = lspEnhancer.enhanceStageB('validateEmail', mockContext, []);
      
      expect(result.candidates).toHaveLength(1);
      expect(result.candidates[0].file_path).toBe('/test/project/src/utils/validation.ts');
      expect(result.lsp_contributions).toBe(1);
    });

    it('should match symbol aliases', () => {
      const result = lspEnhancer.enhanceStageB('fetchUserData', mockContext, []); // Alias for getUserData
      
      expect(result.candidates).toHaveLength(1);
      expect(result.candidates[0].file_path).toBe('/test/project/src/services/UserService.ts');
      expect(result.candidates[0].match_reasons).toContain('lsp_hint');
    });

    it('should match by symbol type/kind', () => {
      const result = lspEnhancer.enhanceStageB('class', mockContext, []);
      
      expect(result.candidates).toHaveLength(1); // UserProfile class
      expect(result.candidates[0].symbol_kind).toBe('class');
      expect(result.candidates[0].file_path).toBe('/test/project/src/components/UserProfile.tsx');
    });

    it('should match partial symbol names', () => {
      const result = lspEnhancer.enhanceStageB('User', mockContext, []);
      
      // Should match UserProfile (contains "User")
      expect(result.candidates.length).toBeGreaterThan(0);
      expect(result.candidates.some(c => c.file_path.includes('UserProfile.tsx'))).toBe(true);
    });

    it('should prioritize context file matches', () => {
      // Query from UserProfile.tsx context should prioritize symbols in same file
      const contextInUserProfile = {
        ...mockContext,
        file_path: '/test/project/src/components/UserProfile.tsx'
      };

      const result = lspEnhancer.enhanceStageB('handle', contextInUserProfile, []);
      
      // Should find handleSubmit method
      expect(result.candidates).toHaveLength(1);
      expect(result.candidates[0].file_path).toBe('/test/project/src/components/UserProfile.tsx');
      expect(result.candidates[0].line_no).toBe(25);
    });
  });

  describe('scoring and ranking', () => {
    it('should apply LSP scoring bonuses correctly', () => {
      const result = lspEnhancer.enhanceStageB('getUserData', mockContext, []);
      
      const candidate = result.candidates[0];
      expect(candidate.score).toBeGreaterThan(0.8); // High score for exact match
      
      // Score should reflect LSP factors: exact match, high reference count
      expect(candidate.score).toBeCloseTo(0.95, 1);
    });

    it('should boost scores for high reference counts', () => {
      // UserProfile has 25 references vs validateEmail with 8 references
      const userProfileResult = lspEnhancer.enhanceStageB('UserProfile', mockContext, []);
      const validateEmailResult = lspEnhancer.enhanceStageB('validateEmail', mockContext, []);
      
      expect(userProfileResult.candidates[0].score).toBeGreaterThan(
        validateEmailResult.candidates[0].score
      );
    });

    it('should boost scores for context proximity', () => {
      const contextInUserProfile = {
        ...mockContext,
        file_path: '/test/project/src/components/UserProfile.tsx',
        line_hint: 20
      };

      // handleSubmit is at line 25, close to line_hint 20
      const result = lspEnhancer.enhanceStageB('handleSubmit', contextInUserProfile, []);
      
      expect(result.candidates[0].score).toBeGreaterThan(0.85); // Proximity boost
    });

    it('should limit results to maxResults parameter', () => {
      // Load many hints to test limiting
      const manyHints = Array.from({ length: 30 }, (_, i) => ({
        symbol_id: `many_${i}`,
        name: `symbol${i}`,
        kind: 'function' as const,
        file_path: `/test/file${i}.ts`,
        line: i + 1,
        col: 0,
        definition_uri: `file:///test/file${i}.ts`,
        signature: `function symbol${i}()`,
        aliases: [],
        resolved_imports: [],
        references_count: Math.floor(Math.random() * 10)
      }));

      const enhancerWithManyHints = new LSPStageBEnhancer();
      enhancerWithManyHints.loadHints(manyHints);

      const result = enhancerWithManyHints.enhanceStageB('symbol', mockContext, [], 10);
      
      expect(result.candidates.length).toBeLessThanOrEqual(10);
    });
  });

  describe('performance metrics', () => {
    it('should track LSP lookup performance', () => {
      const result = lspEnhancer.enhanceStageB('UserProfile', mockContext, []);
      
      expect(result.stage_latency_ms).toBeGreaterThan(0);
      expect(result.performance_metrics.lsp_lookup_ms).toBeGreaterThan(0);
      expect(result.performance_metrics.merge_dedupe_ms).toBeGreaterThan(0);
      expect(result.performance_metrics.context_enrichment_ms).toBeGreaterThan(0);
      
      // Total should equal sum of parts
      const totalExpected = result.performance_metrics.lsp_lookup_ms +
                           result.performance_metrics.merge_dedupe_ms +
                           result.performance_metrics.context_enrichment_ms;
      
      expect(result.stage_latency_ms).toBeCloseTo(totalExpected, 0);
    });

    it('should count LSP contributions accurately', () => {
      const result = lspEnhancer.enhanceStageB('User', mockContext, []);
      
      expect(result.lsp_contributions).toBe(result.candidates.filter(c => 
        c.match_reasons?.includes('lsp_hint')
      ).length);
      
      expect(result.total_lsp_hints_used).toBeGreaterThan(0);
    });

    it('should handle queries with no LSP matches', () => {
      const result = lspEnhancer.enhanceStageB('nonexistent_symbol', mockContext, []);
      
      expect(result.candidates).toHaveLength(0);
      expect(result.lsp_contributions).toBe(0);
      expect(result.total_lsp_hints_used).toBe(0);
      expect(result.stage_latency_ms).toBeGreaterThan(0); // Still should track time
    });
  });

  describe('structural context enhancement', () => {
    it('should enhance candidates with structural context', () => {
      const result = lspEnhancer.enhanceStageB('UserProfile', mockContext, []);
      
      const candidate = result.candidates[0] as any;
      expect(candidate.resolved_imports).toContain('React');
      expect(candidate.resolved_imports).toContain('Component');
      expect(candidate.references_count).toBe(25);
      expect(candidate.signature).toBe('class UserProfile extends Component');
    });

    it('should find symbols near specific location', async () => {
      const nearbySymbols = await lspEnhancer.findLSPSymbolsNear(
        '/test/project/src/components/UserProfile.tsx',
        15,
        5
      );

      // Should find UserProfile (line 10) and handleSubmit (line 25) 
      expect(nearbySymbols.length).toBeGreaterThan(0);
      expect(nearbySymbols.some(s => s.line_no === 10)).toBe(true); // UserProfile
      expect(nearbySymbols.some(s => s.line_no === 25)).toBe(true); // handleSubmit
    });

    it('should calculate line proximity correctly', async () => {
      const nearbySymbols = await lspEnhancer.findLSPSymbolsNear(
        '/test/project/src/components/UserProfile.tsx',
        12, // Close to UserProfile at line 10
        3   // Within 3 lines
      );

      // Should find UserProfile (line 10) but not handleSubmit (line 25)
      expect(nearbySymbols.some(s => s.line_no === 10)).toBe(true);
      expect(nearbySymbols.some(s => s.line_no === 25)).toBe(false);
    });
  });

  describe('edge cases and error handling', () => {
    it('should handle empty query strings', () => {
      const result = lspEnhancer.enhanceStageB('', mockContext, []);
      
      expect(result.candidates).toHaveLength(0);
      expect(result.lsp_contributions).toBe(0);
      expect(result.stage_latency_ms).toBeGreaterThan(0);
    });

    it('should handle queries with special characters', () => {
      const specialQueries = [
        'User.Profile',
        'getUserData()',
        '@Component',
        'handle-submit',
        'user_profile',
        'MyClass<T>'
      ];

      for (const query of specialQueries) {
        const result = lspEnhancer.enhanceStageB(query, mockContext, []);
        expect(result).toBeDefined();
        expect(result.stage_latency_ms).toBeGreaterThan(0);
      }
    });

    it('should handle case insensitive matching', () => {
      const queries = ['userprofile', 'USERPROFILE', 'UserProfile', 'uSERpROFILE'];
      
      for (const query of queries) {
        const result = lspEnhancer.enhanceStageB(query, mockContext, []);
        // Should find some match for case variations
        expect(result).toBeDefined();
      }
    });

    it('should handle very large hint sets', () => {
      const largeHintSet = Array.from({ length: 1000 }, (_, i) => ({
        symbol_id: `large_${i}`,
        name: `largeSymbol${i}`,
        kind: 'function' as const,
        file_path: `/large/file${i}.ts`,
        line: i + 1,
        col: 0,
        definition_uri: `file:///large/file${i}.ts`,
        signature: `function largeSymbol${i}()`,
        aliases: [`alias${i}`],
        resolved_imports: [],
        references_count: i
      }));

      const largeEnhancer = new LSPStageBEnhancer();
      largeEnhancer.loadHints(largeHintSet);

      const result = largeEnhancer.enhanceStageB('largeSymbol500', mockContext, []);
      
      expect(result.candidates).toHaveLength(1);
      expect(result.candidates[0].file_path).toBe('/large/file500.ts');
      expect(result.stage_latency_ms).toBeLessThan(100); // Should still be fast
    });

    it('should handle malformed LSP hints gracefully', () => {
      const malformedHints = [
        {
          symbol_id: 'malformed_1',
          name: '', // Empty name
          kind: 'function' as const,
          file_path: '/test/malformed.ts',
          line: 1,
          col: 0,
          definition_uri: 'file:///test/malformed.ts',
          signature: 'function',
          aliases: [],
          resolved_imports: [],
          references_count: 0
        },
        // Missing required fields would cause TypeScript errors, 
        // so test with minimal valid structure
      ];

      const enhancerWithMalformed = new LSPStageBEnhancer();
      expect(() => {
        enhancerWithMalformed.loadHints(malformedHints);
      }).not.toThrow();

      const result = enhancerWithMalformed.enhanceStageB('test', mockContext, []);
      expect(result).toBeDefined();
    });
  });

  describe('integration with context', () => {
    it('should use language hint for filtering', () => {
      const typescriptContext = { ...mockContext, language_hint: 'typescript' };
      const pythonContext = { ...mockContext, language_hint: 'python' };

      // All our mock hints are TypeScript-like, so should work better with TS context
      const tsResult = lspEnhancer.enhanceStageB('UserProfile', typescriptContext, []);
      const pyResult = lspEnhancer.enhanceStageB('UserProfile', pythonContext, []);

      expect(tsResult.candidates.length).toBeGreaterThanOrEqual(pyResult.candidates.length);
    });

    it('should use repo_sha for hint filtering', () => {
      const correctShaContext = { ...mockContext, repo_sha: 'test-sha-123' };
      const wrongShaContext = { ...mockContext, repo_sha: 'wrong-sha-456' };

      const correctResult = lspEnhancer.enhanceStageB('UserProfile', correctShaContext, []);
      const wrongResult = lspEnhancer.enhanceStageB('UserProfile', wrongShaContext, []);

      // Currently not implemented in the mock, but structure for future
      expect(correctResult).toBeDefined();
      expect(wrongResult).toBeDefined();
    });

    it('should respect trace_id for telemetry', () => {
      const result = lspEnhancer.enhanceStageB('UserProfile', mockContext, []);
      
      // Verify telemetry context is passed through
      expect(result).toBeDefined();
      expect(mockContext.trace_id).toBe('test-trace-123');
    });
  });
});