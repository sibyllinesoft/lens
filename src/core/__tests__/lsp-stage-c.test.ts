/**
 * Comprehensive Tests for LSP Stage-C Features Component
 * Tests bounded LSP features and log-odds capping (≤0.4)
 */

import { describe, it, expect, beforeEach, afterEach, mock, jest } from 'bun:test';
import { LSPStageCEnhancer } from '../lsp-stage-c.js';
import type { LSPHint, SearchContext, Candidate, LSPFeatures } from '../../types/core.js';

describe('LSPStageCEnhancer', () => {
  let lspEnhancer: LSPStageCEnhancer;
  let mockContext: SearchContext;
  let mockLSPHints: LSPHint[];

  beforeEach(() => {
    lspEnhancer = new LSPStageCEnhancer();
    
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
        symbol_id: 'lsp_def_1',
        name: 'UserProfile',
        kind: 'class',
        file_path: '/test/project/src/components/UserProfile.tsx',
        line: 10,
        col: 0,
        definition_uri: 'file:///test/project/src/components/UserProfile.tsx',
        signature: 'class UserProfile extends React.Component<UserProps, UserState>',
        aliases: ['Profile', 'UserComponent'],
        resolved_imports: ['React', 'Component', 'UserProps', 'UserState'],
        references_count: 25
      },
      {
        symbol_id: 'lsp_ref_1',
        name: 'handleSubmit',
        kind: 'method',
        file_path: '/test/project/src/components/UserProfile.tsx',
        line: 45,
        col: 2,
        definition_uri: 'file:///test/project/src/components/UserProfile.tsx',
        signature: 'handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void>',
        aliases: ['onSubmit', 'submitForm'],
        resolved_imports: ['FormEvent'],
        references_count: 8
      },
      {
        symbol_id: 'lsp_type_1',
        name: 'ApiResponse',
        kind: 'interface',
        file_path: '/test/project/src/types/api.ts',
        line: 5,
        col: 0,
        definition_uri: 'file:///test/project/src/types/api.ts',
        signature: 'interface ApiResponse<T> { data: T; status: number; message?: string; }',
        aliases: ['Response', 'APIResponse'],
        resolved_imports: [],
        references_count: 15
      },
      {
        symbol_id: 'lsp_alias_1',
        name: 'getUserData',
        kind: 'function',
        file_path: '/test/project/src/services/userService.ts',
        line: 20,
        col: 17,
        definition_uri: 'file:///test/project/src/services/userService.ts',
        signature: 'function getUserData(userId: string): Promise<ApiResponse<User>>',
        aliases: ['fetchUser', 'loadUserData', 'retrieveUserInfo'],
        resolved_imports: ['ApiResponse', 'User'],
        references_count: 12
      },
      {
        symbol_id: 'lsp_high_refs',
        name: 'validateInput',
        kind: 'function',
        file_path: '/test/project/src/utils/validation.ts',
        line: 30,
        col: 9,
        definition_uri: 'file:///test/project/src/utils/validation.ts',
        signature: 'function validateInput(value: string, rules: ValidationRule[]): boolean',
        aliases: ['validate', 'checkInput'],
        resolved_imports: ['ValidationRule'],
        references_count: 50 // High reference count for testing bounds
      }
    ];

    lspEnhancer.loadHints(mockLSPHints);
  });

  describe('bounded contribution enforcement', () => {
    it('should enforce maximum log-odds limit of 0.4', () => {
      const candidates: Candidate[] = [
        {
          file_path: '/test/project/src/components/UserProfile.tsx',
          line_no: 10,
          col_no: 0,
          content: 'class UserProfile extends React.Component {',
          score: 0.5, // Base score
          match_reasons: ['exact_match'],
          symbol_kind: 'class'
        }
      ];

      const result = lspEnhancer.enhanceStageC(candidates, 'UserProfile', mockContext);
      
      expect(result.candidates).toHaveLength(1);
      
      // Verify bounds are enforced
      const boundsValidation = lspEnhancer.validateBounds(result);
      expect(boundsValidation.bounds_enforced).toBe(true);
      expect(boundsValidation.max_contribution).toBeLessThanOrEqual(0.4);
      expect(boundsValidation.violations_count).toBe(0);
      
      // Score should be enhanced but not excessive
      const enhancedCandidate = result.candidates[0];
      expect(enhancedCandidate.score).toBeGreaterThan(0.5);
      expect(enhancedCandidate.score).toBeLessThan(1.0); // Reasonable upper bound
    });

    it('should cap contribution when LSP features would exceed limit', () => {
      // Create a candidate that would naturally get very high LSP contribution
      const candidates: Candidate[] = [
        {
          file_path: '/test/project/src/utils/validation.ts',
          line_no: 30,
          col_no: 9,
          content: 'function validateInput(value, rules) {',
          score: 0.9, // High base score
          match_reasons: ['exact_match', 'location_match'],
          symbol_kind: 'function'
        }
      ];

      const result = lspEnhancer.enhanceStageC(candidates, 'validateInput', mockContext);
      
      // Calculate what the contribution would be without bounds
      const enhancedCandidate = result.candidates[0];
      const lspFeatures = (enhancedCandidate as any).lsp_features as LSPFeatures;
      
      expect(lspFeatures).toBeDefined();
      expect(lspFeatures.lsp_ref_count).toBe(50); // High reference count
      
      // But the actual score increase should be bounded
      const scoreIncrease = enhancedCandidate.score - 0.9;
      const logOddsContribution = Math.log(scoreIncrease / (1 - scoreIncrease));
      expect(logOddsContribution).toBeLessThanOrEqual(0.4);
    });

    it('should validate bounds correctly across multiple candidates', () => {
      const candidates: Candidate[] = [
        {
          file_path: '/test/project/src/components/UserProfile.tsx',
          line_no: 10,
          col_no: 0,
          content: 'class UserProfile {',
          score: 0.7,
          match_reasons: ['class_match'],
          symbol_kind: 'class'
        },
        {
          file_path: '/test/project/src/components/UserProfile.tsx', 
          line_no: 45,
          col_no: 2,
          content: 'handleSubmit(event) {',
          score: 0.6,
          match_reasons: ['method_match'],
          symbol_kind: 'method'
        },
        {
          file_path: '/test/project/src/utils/validation.ts',
          line_no: 30,
          col_no: 9,
          content: 'function validateInput {',
          score: 0.8,
          match_reasons: ['function_match'],
          symbol_kind: 'function'
        }
      ];

      const result = lspEnhancer.enhanceStageC(candidates, 'User', mockContext);
      
      const boundsValidation = lspEnhancer.validateBounds(result);
      expect(boundsValidation.bounds_enforced).toBe(true);
      expect(boundsValidation.violations_count).toBe(0);
      expect(boundsValidation.max_contribution).toBeLessThanOrEqual(0.4);
    });
  });

  describe('LSP feature extraction', () => {
    it('should extract definition hit features correctly', () => {
      const candidates: Candidate[] = [
        {
          file_path: '/test/project/src/components/UserProfile.tsx',
          line_no: 10,
          col_no: 0,
          content: 'class UserProfile extends React.Component {',
          score: 0.7,
          match_reasons: ['exact_match'],
          symbol_kind: 'class'
        }
      ];

      const result = lspEnhancer.enhanceStageC(candidates, 'UserProfile', mockContext);
      
      const enhancedCandidate = result.candidates[0] as any;
      const lspFeatures: LSPFeatures = enhancedCandidate.lsp_features;
      
      expect(lspFeatures).toBeDefined();
      expect(lspFeatures.lsp_def_hit).toBe(1); // Should be a definition hit
      expect(lspFeatures.lsp_ref_count).toBe(25); // From the hint
      expect(lspFeatures.type_match).toBeGreaterThan(0); // Should have type similarity
      expect(lspFeatures.alias_resolved).toBe(0); // Not an alias query
    });

    it('should detect alias resolutions correctly', () => {
      const candidates: Candidate[] = [
        {
          file_path: '/test/project/src/services/userService.ts',
          line_no: 20,
          col_no: 17,
          content: 'function getUserData(userId) {',
          score: 0.6,
          match_reasons: ['fuzzy_match'],
          symbol_kind: 'function'
        }
      ];

      const result = lspEnhancer.enhanceStageC(candidates, 'fetchUser', mockContext); // Using alias
      
      const enhancedCandidate = result.candidates[0] as any;
      const lspFeatures: LSPFeatures = enhancedCandidate.lsp_features;
      
      expect(lspFeatures).toBeDefined();
      expect(lspFeatures.alias_resolved).toBe(1); // Should detect alias resolution
      expect(lspFeatures.lsp_def_hit).toBe(1); // Still a definition
      expect(lspFeatures.lsp_ref_count).toBe(12);
    });

    it('should calculate type match scores accurately', () => {
      const candidates: Candidate[] = [
        {
          file_path: '/test/project/src/types/api.ts',
          line_no: 5,
          col_no: 0,
          content: 'interface ApiResponse<T> {',
          score: 0.8,
          match_reasons: ['type_match'],
          symbol_kind: 'interface'
        }
      ];

      const result = lspEnhancer.enhanceStageC(candidates, 'ApiResponse', mockContext);
      
      const enhancedCandidate = result.candidates[0] as any;
      const lspFeatures: LSPFeatures = enhancedCandidate.lsp_features;
      
      expect(lspFeatures).toBeDefined();
      expect(lspFeatures.type_match).toBeGreaterThan(0.8); // High type match for exact match
      expect(lspFeatures.lsp_def_hit).toBe(1);
      expect(lspFeatures.lsp_ref_count).toBe(15);
    });

    it('should handle queries with no matching LSP hints', () => {
      const candidates: Candidate[] = [
        {
          file_path: '/test/project/src/unknown/file.ts',
          line_no: 1,
          col_no: 0,
          content: 'function unknownFunction() {',
          score: 0.5,
          match_reasons: ['lexical_match'],
          symbol_kind: 'function'
        }
      ];

      const result = lspEnhancer.enhanceStageC(candidates, 'unknownFunction', mockContext);
      
      const enhancedCandidate = result.candidates[0] as any;
      const lspFeatures: LSPFeatures = enhancedCandidate.lsp_features;
      
      expect(lspFeatures).toBeDefined();
      expect(lspFeatures.lsp_def_hit).toBe(0);
      expect(lspFeatures.lsp_ref_count).toBe(0);
      expect(lspFeatures.type_match).toBe(0);
      expect(lspFeatures.alias_resolved).toBe(0);
      
      // Score should remain largely unchanged
      expect(enhancedCandidate.score).toBeCloseTo(0.5, 1);
    });
  });

  describe('log-odds score combination', () => {
    it('should combine scores using log-odds correctly', () => {
      const candidates: Candidate[] = [
        {
          file_path: '/test/project/src/components/UserProfile.tsx',
          line_no: 10,
          col_no: 0,
          content: 'class UserProfile extends React.Component {',
          score: 0.7, // Base score
          match_reasons: ['exact_match'],
          symbol_kind: 'class'
        }
      ];

      const result = lspEnhancer.enhanceStageC(candidates, 'UserProfile', mockContext);
      
      const enhancedCandidate = result.candidates[0];
      
      // Score should be higher than base, but bounded
      expect(enhancedCandidate.score).toBeGreaterThan(0.7);
      expect(enhancedCandidate.score).toBeLessThan(1.0);
      
      // The increase should correspond to bounded log-odds contribution
      const originalLogOdds = Math.log(0.7 / (1 - 0.7));
      const newLogOdds = Math.log(enhancedCandidate.score / (1 - enhancedCandidate.score));
      const contribution = newLogOdds - originalLogOdds;
      
      expect(contribution).toBeGreaterThan(0);
      expect(contribution).toBeLessThanOrEqual(0.4);
    });

    it('should handle edge case scores correctly', () => {
      const edgeScores = [0.01, 0.1, 0.5, 0.9, 0.99];
      
      for (const baseScore of edgeScores) {
        const candidates: Candidate[] = [
          {
            file_path: '/test/project/src/components/UserProfile.tsx',
            line_no: 10,
            col_no: 0,
            content: 'class UserProfile {',
            score: baseScore,
            match_reasons: ['match'],
            symbol_kind: 'class'
          }
        ];

        const result = lspEnhancer.enhanceStageC(candidates, 'UserProfile', mockContext);
        const enhancedScore = result.candidates[0].score;
        
        // Score should improve but stay within bounds
        expect(enhancedScore).toBeGreaterThan(baseScore);
        expect(enhancedScore).toBeLessThan(1.0);
        
        // Log-odds contribution should be bounded
        const originalLogOdds = Math.log(baseScore / (1 - baseScore));
        const newLogOdds = Math.log(enhancedScore / (1 - enhancedScore));
        const contribution = newLogOdds - originalLogOdds;
        
        expect(contribution).toBeLessThanOrEqual(0.4);
      }
    });
  });

  describe('feature importance analysis', () => {
    it('should analyze LSP feature importance correctly', () => {
      const candidates: Candidate[] = [
        {
          file_path: '/test/project/src/components/UserProfile.tsx',
          line_no: 10,
          col_no: 0,
          content: 'class UserProfile {',
          score: 0.7,
          match_reasons: ['exact_match'],
          symbol_kind: 'class'
        },
        {
          file_path: '/test/project/src/services/userService.ts',
          line_no: 20,
          col_no: 17,
          content: 'function getUserData {',
          score: 0.6,
          match_reasons: ['fuzzy_match'],
          symbol_kind: 'function'
        }
      ];

      const result = lspEnhancer.enhanceStageC(candidates, 'User', mockContext);
      
      const importance = lspEnhancer.analyzeLSPFeatureImportance(result.candidates);
      
      expect(importance.total_candidates).toBe(2);
      expect(importance.lsp_enhanced_candidates).toBeGreaterThan(0);
      expect(importance.feature_correlations).toBeDefined();
      expect(importance.feature_correlations.def_hit_score_correlation).toBeDefined();
      expect(importance.feature_correlations.ref_count_score_correlation).toBeDefined();
      expect(importance.feature_correlations.type_match_score_correlation).toBeDefined();
      expect(importance.average_contribution).toBeGreaterThan(0);
      expect(importance.average_contribution).toBeLessThanOrEqual(0.4);
    });

    it('should handle cases with no LSP-enhanced candidates', () => {
      const candidates: Candidate[] = [
        {
          file_path: '/test/project/src/unknown/file.ts',
          line_no: 1,
          col_no: 0,
          content: 'function unknown() {',
          score: 0.5,
          match_reasons: ['lexical'],
          symbol_kind: 'function'
        }
      ];

      const result = lspEnhancer.enhanceStageC(candidates, 'unknown', mockContext);
      const importance = lspEnhancer.analyzeLSPFeatureImportance(result.candidates);
      
      expect(importance.total_candidates).toBe(1);
      expect(importance.lsp_enhanced_candidates).toBe(0);
      expect(importance.average_contribution).toBe(0);
    });
  });

  describe('performance metrics and statistics', () => {
    it('should track stage performance correctly', () => {
      const candidates: Candidate[] = [
        {
          file_path: '/test/project/src/components/UserProfile.tsx',
          line_no: 10,
          col_no: 0,
          content: 'class UserProfile {',
          score: 0.7,
          match_reasons: ['match'],
          symbol_kind: 'class'
        }
      ];

      const result = lspEnhancer.enhanceStageC(candidates, 'UserProfile', mockContext);
      
      expect(result.stage_latency_ms).toBeGreaterThan(0);
      expect(result.performance_metrics.feature_extraction_ms).toBeGreaterThan(0);
      expect(result.performance_metrics.score_calculation_ms).toBeGreaterThan(0);
      expect(result.performance_metrics.bounds_validation_ms).toBeGreaterThan(0);
      expect(result.performance_metrics.metadata_enrichment_ms).toBeGreaterThan(0);
      
      // Total should equal sum of parts
      const totalExpected = result.performance_metrics.feature_extraction_ms +
                           result.performance_metrics.score_calculation_ms +
                           result.performance_metrics.bounds_validation_ms +
                           result.performance_metrics.metadata_enrichment_ms;
      
      expect(result.stage_latency_ms).toBeCloseTo(totalExpected, 0);
    });

    it('should provide accurate statistics', () => {
      const stats = lspEnhancer.getStats();
      
      expect(stats.max_log_odds_limit).toBe(0.4);
      expect(stats.feature_weights.def_hit_bonus).toBe(0.3);
      expect(stats.feature_weights.ref_count_scaling).toBe(0.05);
      expect(stats.feature_weights.type_match_bonus).toBe(0.15);
      expect(stats.feature_weights.alias_bonus).toBe(0.1);
      expect(stats.hints_loaded).toBe(5); // From our mock hints
      expect(stats.total_enhancements).toBe(0); // No enhancements run yet
    });

    it('should count enhancements accurately', () => {
      const candidates: Candidate[] = [
        {
          file_path: '/test/project/src/components/UserProfile.tsx',
          line_no: 10,
          col_no: 0,
          content: 'class UserProfile {',
          score: 0.7,
          match_reasons: ['match'],
          symbol_kind: 'class'
        }
      ];

      // Run enhancement
      lspEnhancer.enhanceStageC(candidates, 'UserProfile', mockContext);
      
      const stats = lspEnhancer.getStats();
      expect(stats.total_enhancements).toBe(1);
    });
  });

  describe('bounds validation edge cases', () => {
    it('should handle candidates with very high base scores', () => {
      const candidates: Candidate[] = [
        {
          file_path: '/test/project/src/components/UserProfile.tsx',
          line_no: 10,
          col_no: 0,
          content: 'class UserProfile {',
          score: 0.98, // Very high base score
          match_reasons: ['perfect_match'],
          symbol_kind: 'class'
        }
      ];

      const result = lspEnhancer.enhanceStageC(candidates, 'UserProfile', mockContext);
      
      const boundsValidation = lspEnhancer.validateBounds(result);
      expect(boundsValidation.bounds_enforced).toBe(true);
      expect(boundsValidation.violations_count).toBe(0);
      
      // Even with high base score, enhancement should be bounded
      const enhancedScore = result.candidates[0].score;
      expect(enhancedScore).toBeLessThan(1.0);
      expect(enhancedScore).toBeGreaterThanOrEqual(0.98);
    });

    it('should handle candidates with very low base scores', () => {
      const candidates: Candidate[] = [
        {
          file_path: '/test/project/src/components/UserProfile.tsx',
          line_no: 10,
          col_no: 0,
          content: 'class UserProfile {',
          score: 0.02, // Very low base score
          match_reasons: ['weak_match'],
          symbol_kind: 'class'
        }
      ];

      const result = lspEnhancer.enhanceStageC(candidates, 'UserProfile', mockContext);
      
      const boundsValidation = lspEnhancer.validateBounds(result);
      expect(boundsValidation.bounds_enforced).toBe(true);
      expect(boundsValidation.violations_count).toBe(0);
      
      // Low base score should get significant but bounded boost
      const enhancedScore = result.candidates[0].score;
      expect(enhancedScore).toBeGreaterThan(0.02);
      
      // Calculate log-odds contribution
      const originalLogOdds = Math.log(0.02 / (1 - 0.02));
      const newLogOdds = Math.log(enhancedScore / (1 - enhancedScore));
      const contribution = newLogOdds - originalLogOdds;
      
      expect(contribution).toBeLessThanOrEqual(0.4);
    });

    it('should detect and report bounds violations during development', () => {
      // Temporarily modify the max limit for testing violation detection
      const originalLimit = (LSPStageCEnhancer as any).MAX_LSP_LOG_ODDS;
      (LSPStageCEnhancer as any).MAX_LSP_LOG_ODDS = 0.1; // Very low limit
      
      const candidates: Candidate[] = [
        {
          file_path: '/test/project/src/utils/validation.ts',
          line_no: 30,
          col_no: 9,
          content: 'function validateInput {',
          score: 0.5,
          match_reasons: ['function_match'],
          symbol_kind: 'function'
        }
      ];

      const result = lspEnhancer.enhanceStageC(candidates, 'validateInput', mockContext);
      const boundsValidation = lspEnhancer.validateBounds(result);
      
      // Should still enforce bounds even with features that would naturally exceed
      expect(boundsValidation.bounds_enforced).toBe(true);
      // But might detect the natural tendency to violate
      expect(boundsValidation.max_contribution).toBeLessThanOrEqual(0.1);
      
      // Restore original limit
      (LSPStageCEnhancer as any).MAX_LSP_LOG_ODDS = originalLimit;
    });
  });

  describe('metadata enrichment', () => {
    it('should add LSP metadata to candidates', () => {
      const candidates: Candidate[] = [
        {
          file_path: '/test/project/src/components/UserProfile.tsx',
          line_no: 10,
          col_no: 0,
          content: 'class UserProfile {',
          score: 0.7,
          match_reasons: ['class_match'],
          symbol_kind: 'class'
        }
      ];

      const result = lspEnhancer.enhanceStageC(candidates, 'UserProfile', mockContext);
      
      const enhancedCandidate = result.candidates[0] as any;
      
      expect(enhancedCandidate.lsp_features).toBeDefined();
      expect(enhancedCandidate.match_reasons).toContain('class_match'); // Original preserved
      expect(enhancedCandidate.match_reasons).toContain('lsp_enhanced'); // LSP reason added
    });

    it('should update match reasons based on LSP contribution significance', () => {
      const candidates: Candidate[] = [
        {
          file_path: '/test/project/src/services/userService.ts',
          line_no: 20,
          col_no: 17,
          content: 'function getUserData {',
          score: 0.4, // Low base score to ensure significant LSP contribution
          match_reasons: ['fuzzy_match'],
          symbol_kind: 'function'
        }
      ];

      const result = lspEnhancer.enhanceStageC(candidates, 'fetchUser', mockContext); // Using alias
      
      const enhancedCandidate = result.candidates[0] as any;
      
      expect(enhancedCandidate.match_reasons).toContain('fuzzy_match'); // Original
      expect(enhancedCandidate.match_reasons).toContain('lsp_enhanced'); // Added
      expect(enhancedCandidate.match_reasons).toContain('lsp_alias_resolved'); // Specific LSP reason
    });
  });

  describe('performance constraint validation', () => {
    it('should complete enhancement within reasonable time bounds', () => {
      // Create a larger set of candidates to test performance
      const candidates: Candidate[] = Array.from({ length: 20 }, (_, i) => ({
        file_path: `/test/project/src/file${i}.ts`,
        line_no: i + 1,
        col_no: 0,
        content: `function func${i}() {`,
        score: 0.5 + (i * 0.01),
        match_reasons: ['generated_match'],
        symbol_kind: 'function'
      }));

      const start = Date.now();
      const result = lspEnhancer.enhanceStageC(candidates, 'func', mockContext);
      const duration = Date.now() - start;
      
      expect(result.stage_latency_ms).toBeLessThan(100); // Should be fast
      expect(duration).toBeLessThan(100); // Actual time should also be fast
      expect(result.candidates).toHaveLength(20);
    });

    it('should maintain bounded contributions under performance pressure', () => {
      const candidates: Candidate[] = Array.from({ length: 50 }, (_, i) => ({
        file_path: `/test/project/src/file${i}.ts`,
        line_no: i + 1,
        col_no: 0,
        content: `function func${i}() {`,
        score: 0.3 + (i * 0.01),
        match_reasons: ['generated'],
        symbol_kind: 'function'
      }));

      const result = lspEnhancer.enhanceStageC(candidates, 'validateInput', mockContext);
      
      // Even under pressure, bounds must be enforced
      const boundsValidation = lspEnhancer.validateBounds(result);
      expect(boundsValidation.bounds_enforced).toBe(true);
      expect(boundsValidation.violations_count).toBe(0);
      expect(boundsValidation.max_contribution).toBeLessThanOrEqual(0.4);
    });
  });

  describe('error handling and edge cases', () => {
    it('should handle empty candidates array', () => {
      const result = lspEnhancer.enhanceStageC([], 'UserProfile', mockContext);
      
      expect(result.candidates).toHaveLength(0);
      expect(result.stage_latency_ms).toBeGreaterThan(0);
      expect(result.performance_metrics).toBeDefined();
    });

    it('should handle malformed candidates gracefully', () => {
      const malformedCandidates: any[] = [
        {
          file_path: '/test/file.ts',
          line_no: 1,
          col_no: 0,
          content: 'code',
          score: NaN, // Invalid score
          match_reasons: ['test'],
          symbol_kind: 'function'
        },
        {
          file_path: '/test/file2.ts',
          line_no: 2,
          col_no: 0,
          content: 'more code',
          score: -0.5, // Invalid negative score
          match_reasons: ['test'],
          symbol_kind: 'function'
        }
      ];

      expect(() => {
        lspEnhancer.enhanceStageC(malformedCandidates, 'test', mockContext);
      }).not.toThrow();
    });

    it('should handle missing context fields gracefully', () => {
      const incompleteContext = {
        mode: 'search'
      } as any; // Minimal context

      const candidates: Candidate[] = [
        {
          file_path: '/test/file.ts',
          line_no: 1,
          col_no: 0,
          content: 'function test() {',
          score: 0.5,
          match_reasons: ['match'],
          symbol_kind: 'function'
        }
      ];

      expect(() => {
        lspEnhancer.enhanceStageC(candidates, 'test', incompleteContext);
      }).not.toThrow();
    });

    it('should handle queries with special characters', () => {
      const specialQueries = [
        'User.Profile',
        'getUserData()',
        '@Component',
        'handle-event',
        'user_data',
        'MyClass<T>',
        'function*',
        'async/await'
      ];

      const candidates: Candidate[] = [
        {
          file_path: '/test/file.ts',
          line_no: 1,
          col_no: 0,
          content: 'special content',
          score: 0.5,
          match_reasons: ['match'],
          symbol_kind: 'function'
        }
      ];

      for (const query of specialQueries) {
        expect(() => {
          lspEnhancer.enhanceStageC(candidates, query, mockContext);
        }).not.toThrow();
      }
    });
  });
});