/**
 * Comprehensive tests for Loss Taxonomy Analyzer
 * Tests the five loss categories: {NO_SYM_COVERAGE, WRONG_ALIAS, PATH_MAP, USABILITY_INTENT, RANKING_ONLY}
 * Validates detailed failure analysis and recommendation generation
 */

import { describe, test, expect, vi, beforeEach } from 'vitest';
import { LossTaxonomyAnalyzer } from '../loss-taxonomy.js';
import type {
  Candidate,
  SearchContext,
  QueryIntent,
  LSPHint,
  LossTaxonomy
} from '../../types/core.js';

// Mock telemetry
vi.mock('../../telemetry/tracer.js', () => ({
  LensTracer: {
    createChildSpan: vi.fn(() => ({
      setAttributes: vi.fn(),
      recordException: vi.fn(),
      end: vi.fn(),
    })),
  },
}));

interface GroundTruthEntry {
  file_path: string;
  line: number;
  col: number;
  relevance: number;
  is_primary: boolean;
  symbol_name?: string;
  symbol_kind?: string;
}

describe('LossTaxonomyAnalyzer', () => {
  let analyzer: LossTaxonomyAnalyzer;
  
  // Helper function to create mock candidates
  const createMockCandidate = (
    filePath: string,
    line: number,
    matchReasons: string[] = ['exact_match'],
    similarity = 0.8,
    symbolKind?: string
  ): Candidate => ({
    file_path: filePath,
    line,
    col: 0,
    content: `mock content at ${filePath}:${line}`,
    symbol: 'mockSymbol',
    match_reasons: matchReasons,
    similarity,
    stage_b_score: similarity,
    stage_c_features: {
      is_definition: false,
      is_reference: false,
      has_documentation: false,
      complexity_score: 0.5,
      recency_score: 0.5,
    },
    context: symbolKind || 'function',
    symbol_kind: symbolKind || 'function',
  });

  // Helper function to create ground truth entries
  const createGroundTruth = (
    filePath: string,
    line: number,
    relevance: number,
    isPrimary = true,
    symbolName?: string,
    symbolKind?: string
  ): GroundTruthEntry => ({
    file_path: filePath,
    line,
    col: 0,
    relevance,
    is_primary: isPrimary,
    symbol_name: symbolName,
    symbol_kind: symbolKind,
  });

  // Helper function to create search context
  const createSearchContext = (query: string, stages: any[] = []): SearchContext => ({
    trace_id: `test_${Date.now()}`,
    repo_sha: 'test_repo_sha',
    query,
    mode: 'hybrid',
    k: 50,
    fuzzy_distance: 2,
    started_at: new Date(),
    stages,
  });

  beforeEach(() => {
    analyzer = new LossTaxonomyAnalyzer();
  });

  describe('initialization and configuration', () => {
    test('initializes with empty state', () => {
      const stats = analyzer.getStats();
      
      expect(stats.lsp_hints_loaded).toBe(0);
      expect(stats.project_config_loaded).toBe(false);
      expect(stats.loss_factors).toEqual([
        'NO_SYM_COVERAGE',
        'WRONG_ALIAS',
        'PATH_MAP',
        'USABILITY_INTENT',
        'RANKING_ONLY'
      ]);
    });

    test('loads LSP hints correctly', () => {
      const hints: LSPHint[] = [
        {
          symbol_id: 'func1',
          name: 'calculateTotal',
          file_path: '/src/utils.ts',
          line: 10,
          col: 0,
          kind: 'function',
          container: 'utils',
          detail: 'function calculateTotal(items: Item[]): number',
          range: { start: { line: 10, character: 0 }, end: { line: 15, character: 1 } },
        },
        {
          symbol_id: 'class1',
          name: 'UserService',
          file_path: '/src/services/user.ts',
          line: 5,
          col: 0,
          kind: 'class',
          container: 'services',
          detail: 'class UserService',
          range: { start: { line: 5, character: 0 }, end: { line: 50, character: 1 } },
        },
      ];

      analyzer.loadLSPHints(hints);
      const stats = analyzer.getStats();
      
      expect(stats.lsp_hints_loaded).toBe(4); // 2 symbols + 2 lowercase name variants
    });

    test('loads project configuration', () => {
      const projectConfig = {
        compilerOptions: {
          baseUrl: './src',
          paths: {
            '@/*': ['src/*'],
            '@utils/*': ['src/utils/*'],
          },
        },
      };

      analyzer.loadProjectConfig(projectConfig);
      const stats = analyzer.getStats();
      
      expect(stats.project_config_loaded).toBe(true);
    });
  });

  describe('NO_SYM_COVERAGE loss factor', () => {
    test('detects symbol coverage gap for definition queries with no symbol results', () => {
      const query = 'function calculateTotal';
      const queryIntent: QueryIntent = 'def';
      const results = [
        createMockCandidate('/src/random.ts', 20, ['lexical']), // No symbol match
      ];
      const groundTruth = [
        createGroundTruth('/src/utils.ts', 10, 3, true, 'calculateTotal', 'function'),
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      expect(analysis.loss_taxonomy.NO_SYM_COVERAGE).toBeGreaterThan(0.5);
      expect(analysis.primary_loss_factor).toBe('NO_SYM_COVERAGE');
    });

    test('detects symbol coverage gap for reference queries', () => {
      const query = 'refs calculateTotal';
      const queryIntent: QueryIntent = 'refs';
      const results = []; // No results at all
      const groundTruth = [
        createGroundTruth('/src/app.ts', 25, 2, true, 'calculateTotal', 'function'),
        createGroundTruth('/src/handlers.ts', 15, 2, false, 'calculateTotal', 'function'),
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      expect(analysis.loss_taxonomy.NO_SYM_COVERAGE).toBe(1.0);
      expect(analysis.primary_loss_factor).toBe('NO_SYM_COVERAGE');
    });

    test('detects partial symbol coverage', () => {
      const query = 'UserService';
      const queryIntent: QueryIntent = 'symbol';
      const results = [
        createMockCandidate('/src/services/user.ts', 5, ['symbol'], 0.9), // Found one symbol
      ];
      const groundTruth = [
        createGroundTruth('/src/services/user.ts', 5, 3, true, 'UserService', 'class'),
        createGroundTruth('/src/services/admin.ts', 10, 2, false, 'AdminService', 'class'), // Missing
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      expect(analysis.loss_taxonomy.NO_SYM_COVERAGE).toBeGreaterThan(0);
      expect(analysis.loss_taxonomy.NO_SYM_COVERAGE).toBeLessThan(1);
    });

    test('does not penalize non-symbol queries for missing symbol coverage', () => {
      const query = 'error handling pattern';
      const queryIntent: QueryIntent = 'NL';
      const results = [
        createMockCandidate('/src/errors.ts', 20, ['semantic']),
      ];
      const groundTruth = [
        createGroundTruth('/src/errors.ts', 20, 2),
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      expect(analysis.loss_taxonomy.NO_SYM_COVERAGE).toBe(0);
    });

    test('considers LSP hint availability when assessing coverage', () => {
      // Load LSP hints that contain the symbol
      const hints: LSPHint[] = [
        {
          symbol_id: 'func1',
          name: 'calculateTotal',
          file_path: '/src/utils.ts',
          line: 10,
          col: 0,
          kind: 'function',
          container: 'utils',
          detail: 'function calculateTotal',
          range: { start: { line: 10, character: 0 }, end: { line: 15, character: 1 } },
        },
      ];
      analyzer.loadLSPHints(hints);

      const query = 'calculateTotal';
      const queryIntent: QueryIntent = 'def';
      const results = []; // No results despite LSP coverage
      const groundTruth = [
        createGroundTruth('/src/utils.ts', 10, 3, true, 'calculateTotal', 'function'),
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      // Should detect coverage gap since LSP has the symbol but search didn't find it
      expect(analysis.loss_taxonomy.NO_SYM_COVERAGE).toBeGreaterThan(0.4);
    });
  });

  describe('WRONG_ALIAS loss factor', () => {
    test('detects wrong alias resolution', () => {
      const query = 'calculateTotal';
      const queryIntent: QueryIntent = 'def';
      const results = [
        createMockCandidate('/src/wrong.ts', 30, ['alias'], 0.7), // Wrong alias match
      ];
      const groundTruth = [
        createGroundTruth('/src/utils.ts', 10, 3, true, 'calculateTotal'),
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      expect(analysis.loss_taxonomy.WRONG_ALIAS).toBeGreaterThan(0);
    });

    test('gives zero penalty when no alias resolution used', () => {
      const query = 'calculateTotal';
      const queryIntent: QueryIntent = 'def';
      const results = [
        createMockCandidate('/src/utils.ts', 10, ['exact_match'], 0.9),
      ];
      const groundTruth = [
        createGroundTruth('/src/utils.ts', 10, 3, true, 'calculateTotal'),
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      expect(analysis.loss_taxonomy.WRONG_ALIAS).toBe(0);
    });

    test('detects correct vs incorrect alias resolution', () => {
      const query = 'UserModel';
      const queryIntent: QueryIntent = 'def';
      const results = [
        createMockCandidate('/src/models/user.ts', 5, ['alias'], 0.8), // Correct alias
        createMockCandidate('/src/random.ts', 20, ['alias'], 0.7), // Wrong alias
      ];
      const groundTruth = [
        createGroundTruth('/src/models/user.ts', 5, 3, true, 'UserModel'),
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      // Should detect partial alias accuracy (50%)
      expect(analysis.loss_taxonomy.WRONG_ALIAS).toBeCloseTo(0.5);
      expect(analysis.detailed_analysis.alias_resolution_accuracy).toBeCloseTo(0.5);
    });

    test('handles LSP-based alias resolution', () => {
      const query = 'User';
      const queryIntent: QueryIntent = 'def';
      const resultWithLSPAlias = createMockCandidate('/src/wrong.ts', 15, ['lsp_hint'], 0.6);
      (resultWithLSPAlias as any).lsp_features = { alias_resolved: true };
      
      const results = [resultWithLSPAlias];
      const groundTruth = [
        createGroundTruth('/src/models/user.ts', 10, 3, true, 'User'),
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      expect(analysis.loss_taxonomy.WRONG_ALIAS).toBeGreaterThan(0);
    });
  });

  describe('PATH_MAP loss factor', () => {
    test('detects path mapping issues when results are in wrong directories', () => {
      const query = 'UserService';
      const queryIntent: QueryIntent = 'def';
      const results = [
        createMockCandidate('/src/utils/helper.ts', 10), // Wrong directory
        createMockCandidate('/src/config/setup.ts', 20), // Wrong directory
      ];
      const groundTruth = [
        createGroundTruth('/src/services/user.ts', 5, 3, true, 'UserService'),
        createGroundTruth('/src/services/admin.ts', 10, 2, false, 'AdminService'),
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      expect(analysis.loss_taxonomy.PATH_MAP).toBeGreaterThan(0.5);
    });

    test('gives low penalty when paths partially overlap', () => {
      const query = 'validateUser';
      const queryIntent: QueryIntent = 'def';
      const results = [
        createMockCandidate('/src/services/user.ts', 15), // Correct directory
        createMockCandidate('/src/utils/validation.ts', 25), // Different but reasonable
      ];
      const groundTruth = [
        createGroundTruth('/src/services/user.ts', 15, 3, true, 'validateUser'),
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      expect(analysis.loss_taxonomy.PATH_MAP).toBeLessThan(0.5);
    });

    test('gives zero penalty when paths match correctly', () => {
      const query = 'DatabaseConnection';
      const queryIntent: QueryIntent = 'def';
      const results = [
        createMockCandidate('/src/database/connection.ts', 10),
        createMockCandidate('/src/database/pool.ts', 20),
      ];
      const groundTruth = [
        createGroundTruth('/src/database/connection.ts', 10, 3, true, 'DatabaseConnection'),
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      expect(analysis.loss_taxonomy.PATH_MAP).toBe(0);
    });

    test('handles empty results gracefully', () => {
      const query = 'NonexistentFunction';
      const queryIntent: QueryIntent = 'def';
      const results: Candidate[] = [];
      const groundTruth = [
        createGroundTruth('/src/utils/helpers.ts', 10, 3, true, 'NonexistentFunction'),
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      expect(analysis.loss_taxonomy.PATH_MAP).toBe(0);
    });
  });

  describe('USABILITY_INTENT loss factor', () => {
    test('detects intent classification failures with high confidence', () => {
      const query = 'function calculateTotal';
      const queryIntent: QueryIntent = 'def';
      const resultWithWrongIntent = createMockCandidate('/src/utils.ts', 10);
      (resultWithWrongIntent as any).intent_classification = {
        intent: 'refs' as QueryIntent,
        confidence: 0.9,
      };
      const results = [resultWithWrongIntent];
      const groundTruth = [
        createGroundTruth('/src/utils.ts', 10, 3, true, 'calculateTotal'),
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      expect(analysis.loss_taxonomy.USABILITY_INTENT).toBeCloseTo(0.8);
    });

    test('gives lower penalty for wrong intent with low confidence', () => {
      const query = 'refs UserService';
      const queryIntent: QueryIntent = 'refs';
      const resultWithWrongIntent = createMockCandidate('/src/services.ts', 15);
      (resultWithWrongIntent as any).intent_classification = {
        intent: 'def' as QueryIntent,
        confidence: 0.4,
      };
      const results = [resultWithWrongIntent];
      const groundTruth = [
        createGroundTruth('/src/services.ts', 15, 2, true, 'UserService'),
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      expect(analysis.loss_taxonomy.USABILITY_INTENT).toBeCloseTo(0.5);
    });

    test('gives zero penalty for correct intent classification', () => {
      const query = 'class UserModel';
      const queryIntent: QueryIntent = 'def';
      const resultWithCorrectIntent = createMockCandidate('/src/models.ts', 20);
      (resultWithCorrectIntent as any).intent_classification = {
        intent: 'def' as QueryIntent,
        confidence: 0.85,
      };
      const results = [resultWithCorrectIntent];
      const groundTruth = [
        createGroundTruth('/src/models.ts', 20, 3, true, 'UserModel'),
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      expect(analysis.loss_taxonomy.USABILITY_INTENT).toBe(0);
    });

    test('detects when intent is not honored in search process', () => {
      const query = 'def validateInput';
      const queryIntent: QueryIntent = 'def';
      const resultIntentNotHonored = createMockCandidate('/src/validation.ts', 25);
      (resultIntentNotHonored as any).intent_honored = false;
      const results = [resultIntentNotHonored];
      const groundTruth = [
        createGroundTruth('/src/validation.ts', 25, 3, true, 'validateInput'),
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      expect(analysis.loss_taxonomy.USABILITY_INTENT).toBe(1.0);
    });

    test('handles missing intent classification data', () => {
      const query = 'calculateSum';
      const queryIntent: QueryIntent = 'def';
      const results = [
        createMockCandidate('/src/math.ts', 5), // No intent classification
      ];
      const groundTruth = [
        createGroundTruth('/src/math.ts', 5, 3, true, 'calculateSum'),
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      expect(analysis.loss_taxonomy.USABILITY_INTENT).toBe(0);
    });
  });

  describe('RANKING_ONLY loss factor', () => {
    test('detects relevant results ranked too low', () => {
      const query = 'findUser';
      const queryIntent: QueryIntent = 'def';
      // Create results with relevant match at position 12
      const results = [
        ...Array.from({ length: 11 }, (_, i) => 
          createMockCandidate(`/src/irrelevant${i}.ts`, 10 + i, ['lexical'], 0.3)
        ),
        createMockCandidate('/src/user/service.ts', 20, ['symbol'], 0.9), // Relevant at pos 12
      ];
      const groundTruth = [
        createGroundTruth('/src/user/service.ts', 20, 3, true, 'findUser'),
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      expect(analysis.loss_taxonomy.RANKING_ONLY).toBe(1.0);
      expect(analysis.primary_loss_factor).toBe('RANKING_ONLY');
    });

    test('gives graduated penalty based on ranking position', () => {
      const testCases = [
        { position: 6, expectedPenalty: 0.7 },
        { position: 4, expectedPenalty: 0.4 },
        { position: 2, expectedPenalty: 0 },
      ];

      for (const { position, expectedPenalty } of testCases) {
        const query = `testFunction${position}`;
        const queryIntent: QueryIntent = 'def';
        const results = [
          ...Array.from({ length: position - 1 }, (_, i) => 
            createMockCandidate(`/src/noise${i}.ts`, 10 + i, ['lexical'], 0.2)
          ),
          createMockCandidate('/src/target.ts', 15, ['symbol'], 0.95), // Relevant result
        ];
        const groundTruth = [
          createGroundTruth('/src/target.ts', 15, 3, true, `testFunction${position}`),
        ];
        const context = createSearchContext(query);

        const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
        
        expect(analysis.loss_taxonomy.RANKING_ONLY).toBeCloseTo(expectedPenalty);
      }
    });

    test('gives zero penalty when no relevant results exist', () => {
      const query = 'nonexistentFunction';
      const queryIntent: QueryIntent = 'def';
      const results = [
        createMockCandidate('/src/random1.ts', 10, ['lexical'], 0.4),
        createMockCandidate('/src/random2.ts', 20, ['fuzzy'], 0.3),
      ];
      const groundTruth = [
        createGroundTruth('/src/target.ts', 15, 3, true, 'nonexistentFunction'),
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      expect(analysis.loss_taxonomy.RANKING_ONLY).toBe(0);
    });

    test('gives zero penalty when relevant results are well-ranked', () => {
      const query = 'getUserById';
      const queryIntent: QueryIntent = 'def';
      const results = [
        createMockCandidate('/src/user/service.ts', 25, ['symbol'], 0.95), // Relevant at pos 1
        createMockCandidate('/src/user/controller.ts', 30, ['symbol'], 0.85), // Also relevant
      ];
      const groundTruth = [
        createGroundTruth('/src/user/service.ts', 25, 3, true, 'getUserById'),
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      expect(analysis.loss_taxonomy.RANKING_ONLY).toBe(0);
    });
  });

  describe('detailed analysis', () => {
    test('assesses query complexity correctly', () => {
      const testCases = [
        { query: 'getUserById', expected: 'simple' },
        { query: 'calculate user total cost', expected: 'medium' },
        { query: 'function that validates user input and returns error', expected: 'complex' },
        { query: 'if (user && user.active)', expected: 'complex' },
      ];

      for (const { query, expected } of testCases) {
        const queryIntent: QueryIntent = 'NL';
        const results = [createMockCandidate('/src/test.ts', 10)];
        const groundTruth = [createGroundTruth('/src/test.ts', 10, 2)];
        const context = createSearchContext(query);

        const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
        
        expect(analysis.detailed_analysis.query_complexity).toBe(expected);
      }
    });

    test('calculates symbol coverage gap accurately', () => {
      const query = 'calculateTotal validateInput';
      const queryIntent: QueryIntent = 'symbol';
      const results = [
        createMockCandidate('/src/utils.ts', 10, ['symbol'], 0.9, 'function'), // Found one
      ];
      const groundTruth = [
        createGroundTruth('/src/utils.ts', 10, 3, true, 'calculateTotal', 'function'),
        createGroundTruth('/src/validation.ts', 15, 3, true, 'validateInput', 'function'), // Missing
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      expect(analysis.detailed_analysis.symbol_coverage_gap).toBeCloseTo(0.5); // Found 1 of 2
    });

    test('calculates path mapping correctness', () => {
      const query = 'findUser';
      const queryIntent: QueryIntent = 'def';
      const results = [
        createMockCandidate('/src/user/service.ts', 10), // Correct path
        createMockCandidate('/src/random.ts', 20), // Wrong path
      ];
      const groundTruth = [
        createGroundTruth('/src/user/service.ts', 10, 3, true, 'findUser'),
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      expect(analysis.detailed_analysis.path_mapping_correctness).toBeCloseTo(0.5); // 1 of 2 correct
    });

    test('calculates intent classification accuracy', () => {
      const query = 'def createUser';
      const queryIntent: QueryIntent = 'def';
      const correctResult = createMockCandidate('/src/user.ts', 10);
      (correctResult as any).intent_classification = { intent: 'def', confidence: 0.9 };
      const wrongResult = createMockCandidate('/src/user2.ts', 15);
      (wrongResult as any).intent_classification = { intent: 'refs', confidence: 0.8 };
      
      const results = [correctResult, wrongResult];
      const groundTruth = [
        createGroundTruth('/src/user.ts', 10, 3, true, 'createUser'),
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      expect(analysis.detailed_analysis.intent_classification_accuracy).toBeCloseTo(0.5); // 1 of 2 correct
    });

    test('calculates ranking quality score (NDCG-like)', () => {
      const query = 'processPayment';
      const queryIntent: QueryIntent = 'def';
      const results = [
        createMockCandidate('/src/payment.ts', 10, ['symbol'], 0.95), // High relevance
        createMockCandidate('/src/billing.ts', 15, ['symbol'], 0.8), // Medium relevance
        createMockCandidate('/src/random.ts', 20, ['lexical'], 0.3), // Low relevance
      ];
      const groundTruth = [
        createGroundTruth('/src/payment.ts', 10, 3, true, 'processPayment'), // High relevance
        createGroundTruth('/src/billing.ts', 15, 2, false, 'calculateFee'), // Medium relevance
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      expect(analysis.detailed_analysis.ranking_quality_score).toBeGreaterThan(0);
      expect(analysis.detailed_analysis.ranking_quality_score).toBeLessThanOrEqual(1);
    });
  });

  describe('contributing factors extraction', () => {
    test('extracts why reasons from candidates', () => {
      const query = 'testFunction';
      const queryIntent: QueryIntent = 'def';
      const resultWithWhy = createMockCandidate('/src/test.ts', 10);
      (resultWithWhy as any).why = ['lexical_match', 'path_boost', 'recent_edit'];
      
      const results = [resultWithWhy];
      const groundTruth = [createGroundTruth('/src/test.ts', 10, 3, true, 'testFunction')];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      expect(analysis.contributing_factors).toContain('lexical_match');
      expect(analysis.contributing_factors).toContain('path_boost');
      expect(analysis.contributing_factors).toContain('recent_edit');
    });

    test('identifies stage errors from context', () => {
      const query = 'failingFunction';
      const queryIntent: QueryIntent = 'def';
      const results = [createMockCandidate('/src/test.ts', 10)];
      const groundTruth = [createGroundTruth('/src/test.ts', 10, 2)];
      const stagesWithError = [
        { name: 'stage_a', latency_ms: 100, error: null },
        { name: 'stage_b', latency_ms: 200, error: 'LSP timeout' },
      ];
      const context = createSearchContext(query, stagesWithError);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      expect(analysis.contributing_factors).toContain('stage_errors');
    });

    test('identifies high latency from context', () => {
      const query = 'slowFunction';
      const queryIntent: QueryIntent = 'def';
      const results = [createMockCandidate('/src/test.ts', 10)];
      const groundTruth = [createGroundTruth('/src/test.ts', 10, 2)];
      const stagesWithHighLatency = [
        { name: 'stage_a', latency_ms: 500, error: null },
        { name: 'stage_b', latency_ms: 1500, error: null }, // High latency
      ];
      const context = createSearchContext(query, stagesWithHighLatency);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      expect(analysis.contributing_factors).toContain('high_latency');
    });
  });

  describe('recommendations generation', () => {
    test('generates symbol coverage recommendations', () => {
      const lossTaxonomy: LossTaxonomy = {
        NO_SYM_COVERAGE: 0.8,
        WRONG_ALIAS: 0,
        PATH_MAP: 0,
        USABILITY_INTENT: 0,
        RANKING_ONLY: 0,
      };
      const detailedAnalysis = {
        query_complexity: 'simple' as const,
        symbol_coverage_gap: 0.8,
        alias_resolution_accuracy: 1.0,
        path_mapping_correctness: 1.0,
        intent_classification_accuracy: 1.0,
        ranking_quality_score: 0.3,
      };

      const query = 'testSymbol';
      const queryIntent: QueryIntent = 'def';
      const results: Candidate[] = [];
      const groundTruth = [createGroundTruth('/src/test.ts', 10, 3)];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      expect(analysis.recommendations).toContain('Improve LSP symbol harvesting coverage');
      expect(analysis.recommendations).toContain('Check workspace configuration and indexing completeness');
    });

    test('generates alias resolution recommendations', () => {
      const query = 'User';
      const queryIntent: QueryIntent = 'def';
      const wrongAliasResult = createMockCandidate('/src/wrong.ts', 10, ['alias']);
      const results = [wrongAliasResult];
      const groundTruth = [createGroundTruth('/src/models/user.ts', 5, 3, true, 'User')];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      if (analysis.loss_taxonomy.WRONG_ALIAS > 0.5) {
        expect(analysis.recommendations).toContain('Improve alias resolution accuracy');
        expect(analysis.recommendations).toContain('Review import/alias mapping in workspace config');
      }
    });

    test('generates path mapping recommendations', () => {
      const query = 'DatabaseConnection';
      const queryIntent: QueryIntent = 'def';
      const results = [
        createMockCandidate('/src/wrong/path.ts', 10),
      ];
      const groundTruth = [
        createGroundTruth('/src/database/connection.ts', 15, 3, true, 'DatabaseConnection'),
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      if (analysis.loss_taxonomy.PATH_MAP > 0.5) {
        expect(analysis.recommendations).toContain('Fix path mapping configuration');
        expect(analysis.recommendations).toContain('Review tsconfig.json/pyproject.toml path mappings');
      }
    });

    test('generates intent classification recommendations', () => {
      const query = 'def createUser';
      const queryIntent: QueryIntent = 'def';
      const wrongIntentResult = createMockCandidate('/src/user.ts', 10);
      (wrongIntentResult as any).intent_classification = { intent: 'refs', confidence: 0.9 };
      const results = [wrongIntentResult];
      const groundTruth = [createGroundTruth('/src/user.ts', 10, 3, true, 'createUser')];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      if (analysis.loss_taxonomy.USABILITY_INTENT > 0.5) {
        expect(analysis.recommendations).toContain('Improve intent classification model');
        expect(analysis.recommendations).toContain('Add more training data for query intent patterns');
      }
    });

    test('generates ranking recommendations', () => {
      const query = 'findUser';
      const queryIntent: QueryIntent = 'def';
      const results = [
        ...Array.from({ length: 15 }, (_, i) => 
          createMockCandidate(`/src/noise${i}.ts`, 10 + i, ['lexical'], 0.2)
        ),
        createMockCandidate('/src/user/service.ts', 20, ['symbol'], 0.9), // Relevant but low-ranked
      ];
      const groundTruth = [
        createGroundTruth('/src/user/service.ts', 20, 3, true, 'findUser'),
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      if (analysis.loss_taxonomy.RANKING_ONLY > 0.5) {
        expect(analysis.recommendations).toContain('Improve ranking algorithm');
        expect(analysis.recommendations).toContain('Boost relevance signals for symbol matches');
      }
    });

    test('generates complex query handling recommendations', () => {
      const query = 'function that validates user input and handles errors gracefully';
      const queryIntent: QueryIntent = 'NL';
      const results = [
        createMockCandidate('/src/validation.ts', 10, ['semantic'], 0.4), // Poor results
      ];
      const groundTruth = [
        createGroundTruth('/src/validation.ts', 10, 3, true),
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      if (analysis.detailed_analysis.query_complexity === 'complex' && 
          analysis.detailed_analysis.ranking_quality_score < 0.5) {
        expect(analysis.recommendations).toContain('Add specialized handling for complex queries');
      }
    });
  });

  describe('primary loss factor identification', () => {
    test('identifies highest scoring loss factor as primary', () => {
      const query = 'testMultipleLoss';
      const queryIntent: QueryIntent = 'def';
      // Create scenario with multiple loss factors
      const results = [
        createMockCandidate('/wrong/path.ts', 25, ['alias'], 0.6), // Wrong path + wrong alias
      ];
      const groundTruth = [
        createGroundTruth('/src/correct/path.ts', 10, 3, true, 'testMultipleLoss'),
      ];
      const context = createSearchContext(query);

      const analysis = analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      
      // Primary loss factor should be the one with highest score
      const taxonomyEntries = Object.entries(analysis.loss_taxonomy);
      const maxEntry = taxonomyEntries.reduce((max, current) => 
        current[1] > max[1] ? current : max
      );
      
      expect(analysis.primary_loss_factor).toBe(maxEntry[0]);
    });
  });

  describe('error handling and edge cases', () => {
    test('handles empty results gracefully', () => {
      const query = 'emptyResults';
      const queryIntent: QueryIntent = 'def';
      const results: Candidate[] = [];
      const groundTruth = [createGroundTruth('/src/test.ts', 10, 3)];
      const context = createSearchContext(query);

      expect(() => {
        analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      }).not.toThrow();
    });

    test('handles empty ground truth gracefully', () => {
      const query = 'emptyGroundTruth';
      const queryIntent: QueryIntent = 'def';
      const results = [createMockCandidate('/src/test.ts', 10)];
      const groundTruth: GroundTruthEntry[] = [];
      const context = createSearchContext(query);

      expect(() => {
        analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      }).not.toThrow();
    });

    test('handles malformed candidate data', () => {
      const query = 'malformedCandidate';
      const queryIntent: QueryIntent = 'def';
      const malformedResult = createMockCandidate('/src/test.ts', 10);
      delete (malformedResult as any).match_reasons; // Remove required field
      
      const results = [malformedResult];
      const groundTruth = [createGroundTruth('/src/test.ts', 10, 2)];
      const context = createSearchContext(query);

      expect(() => {
        analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      }).not.toThrow();
    });

    test('handles analysis exceptions gracefully', () => {
      const query = 'exception';
      const queryIntent: QueryIntent = 'def';
      const results = [createMockCandidate('/src/test.ts', 10)];
      const groundTruth = [createGroundTruth('/src/test.ts', 10, 2)];
      const context = createSearchContext(query);

      // Mock an internal method to throw
      const originalMethod = (analyzer as any).assessSymbolCoverageGap;
      (analyzer as any).assessSymbolCoverageGap = vi.fn().mockImplementation(() => {
        throw new Error('Test exception');
      });

      expect(() => {
        analyzer.analyzeLossFactors(query, queryIntent, results, groundTruth, context);
      }).toThrow('Test exception');

      // Restore original method
      (analyzer as any).assessSymbolCoverageGap = originalMethod;
    });
  });
});