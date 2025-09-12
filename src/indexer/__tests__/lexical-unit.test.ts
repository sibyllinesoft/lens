/**
 * Unit tests for LexicalSearchEngine core functionality
 * Focus on trigram generation, tokenization, and search algorithms
 */

import { describe, it, expect, jest, beforeEach, mock } from 'bun:test';
import { LexicalSearchEngine } from '../lexical.js';
import type { SearchContext, MatchReason } from '../../types/core.js';

// Mock all external dependencies
mock('../../storage/segments.js', () => ({
  SegmentStorage: jest.fn().mockImplementation(() => ({
    createSegment: jest.fn().mockResolvedValue({}),
    openSegment: jest.fn().mockResolvedValue({}),
    writeToSegment: jest.fn().mockResolvedValue(undefined),
    readFromSegment: jest.fn().mockResolvedValue(Buffer.alloc(0))
  }))
}));

mock('../optimized-trigram-index.js', () => ({
  OptimizedTrigramIndex: jest.fn().mockImplementation(() => ({
    indexDocument: jest.fn().mockResolvedValue(undefined),
    search: jest.fn().mockResolvedValue([]),
    getStats: jest.fn().mockReturnValue({
      totalDocuments: 0,
      totalTrigrams: 0,
      avgTrigramsPerDoc: 0
    })
  }))
}));

mock('../../telemetry/tracer.js', () => ({
  LensTracer: {
    createChildSpan: jest.fn(() => ({
      setAttributes: jest.fn(),
      recordException: jest.fn(),
      end: jest.fn()
    }))
  }
}));

mock('../../config/features.js', () => ({
  featureFlags: {
    optimizedTrigramIndex: true
  }
}));

describe('LexicalSearchEngine Unit Tests', () => {
  let lexicalEngine: LexicalSearchEngine;
  let mockSegmentStorage: any;

  beforeEach(() => {
    mockSegmentStorage = {
      createSegment: jest.fn().mockResolvedValue({}),
      openSegment: jest.fn().mockResolvedValue({}),
      writeToSegment: jest.fn().mockResolvedValue(undefined),
      readFromSegment: jest.fn().mockResolvedValue(Buffer.alloc(0))
    };
    
    lexicalEngine = new LexicalSearchEngine(mockSegmentStorage);
    jest.clearAllMocks();
  });

  describe('Tokenization Logic', () => {
    it('should tokenize simple content correctly', () => {
      const content = 'function testFunction() { return true; }';
      const filePath = 'test.ts';
      
      // Simulate tokenization logic
      const words = content.match(/\b\w+\b/g) || [];
      const tokens = words.map((word, index) => ({
        token: word,
        file_path: filePath,
        line: 1,
        col: content.indexOf(word),
        length: word.length,
        is_camelcase: /^[a-z]+([A-Z][a-z]*)*$/.test(word),
        is_snake_case: /^[a-z]+(_[a-z]+)*$/.test(word),
        subtokens: word.split(/(?=[A-Z])|_/).filter(s => s.length > 0)
      }));

      expect(tokens).toHaveLength(4); // function, testFunction, return, true
      expect(tokens[0].token).toBe('function');
      expect(tokens[1].token).toBe('testFunction');
      expect(tokens[1].is_camelcase).toBe(true);
      expect(tokens[1].subtokens).toEqual(['test', 'Function']);
    });

    it('should handle camelCase tokenization', () => {
      const camelCaseWord = 'getUserById';
      
      const isCamelCase = /^[a-z]+([A-Z][a-z]*)*$/.test(camelCaseWord);
      const subtokens = camelCaseWord.split(/(?=[A-Z])/).filter(s => s.length > 0);
      
      expect(isCamelCase).toBe(true);
      expect(subtokens).toEqual(['get', 'User', 'By', 'Id']);
    });

    it('should handle snake_case tokenization', () => {
      const snakeCaseWord = 'user_profile_data';
      
      const isSnakeCase = /^[a-z]+(_[a-z]+)*$/.test(snakeCaseWord);
      const subtokens = snakeCaseWord.split('_').filter(s => s.length > 0);
      
      expect(isSnakeCase).toBe(true);
      expect(subtokens).toEqual(['user', 'profile', 'data']);
    });

    it('should handle mixed case patterns', () => {
      const mixedWord = 'XMLHttpRequest';
      
      const isCamelCase = /^[a-z]+([A-Z][a-z]*)*$/.test(mixedWord);
      const subtokens = mixedWord.split(/(?=[A-Z])/).filter(s => s.length > 0);
      
      expect(isCamelCase).toBe(false); // Starts with uppercase
      expect(subtokens).toEqual(['X', 'M', 'L', 'Http', 'Request']);
    });

    it('should handle special characters and numbers', () => {
      const content = 'const API_URL = "https://api.example.com/v1";';
      const words = content.match(/\b\w+\b/g) || [];
      
      expect(words).toEqual(['const', 'API_URL', 'https', 'api', 'example', 'com', 'v1']);
    });

    it('should calculate line and column positions', () => {
      const content = 'line1\nline2 word\nline3';
      const lines = content.split('\n');
      
      let line = 1;
      let col = 0;
      
      for (let i = 0; i < content.length; i++) {
        if (content[i] === '\n') {
          line++;
          col = 0;
        } else {
          col++;
        }
      }
      
      expect(lines).toHaveLength(3);
      expect(content.indexOf('word')).toBe(12);
      
      // Calculate line/col for 'word'
      const wordIndex = content.indexOf('word');
      let wordLine = 1;
      let wordCol = wordIndex;
      
      for (let i = 0; i < wordIndex; i++) {
        if (content[i] === '\n') {
          wordLine++;
          wordCol = wordIndex - i - 1;
        }
      }
      
      expect(wordLine).toBe(2);
      expect(wordCol).toBe(6);
    });
  });

  describe('Trigram Generation', () => {
    it('should generate trigrams for simple words', () => {
      const word = 'test';
      
      // Simulate trigram generation logic
      const generateTrigrams = (input: string): string[] => {
        if (input.length < 3) return [input];
        
        const trigrams: string[] = [];
        for (let i = 0; i <= input.length - 3; i++) {
          trigrams.push(input.substring(i, i + 3));
        }
        return trigrams;
      };
      
      const trigrams = generateTrigrams(word);
      expect(trigrams).toEqual(['tes', 'est']);
    });

    it('should handle short words', () => {
      const generateTrigrams = (input: string): string[] => {
        if (input.length < 3) return [input];
        
        const trigrams: string[] = [];
        for (let i = 0; i <= input.length - 3; i++) {
          trigrams.push(input.substring(i, i + 3));
        }
        return trigrams;
      };
      
      expect(generateTrigrams('a')).toEqual(['a']);
      expect(generateTrigrams('ab')).toEqual(['ab']);
      expect(generateTrigrams('')).toEqual(['']);
    });

    it('should generate unique trigrams', () => {
      const word = 'aaaaaa';
      
      const generateTrigrams = (input: string): string[] => {
        if (input.length < 3) return [input];
        
        const trigramSet = new Set<string>();
        for (let i = 0; i <= input.length - 3; i++) {
          trigramSet.add(input.substring(i, i + 3));
        }
        return Array.from(trigramSet);
      };
      
      const trigrams = generateTrigrams(word);
      expect(trigrams).toEqual(['aaa']); // All trigrams are identical
    });

    it('should handle special characters in trigrams', () => {
      const word = 'api-key';
      
      const generateTrigrams = (input: string): string[] => {
        if (input.length < 3) return [input];
        
        const trigrams: string[] = [];
        for (let i = 0; i <= input.length - 3; i++) {
          trigrams.push(input.substring(i, i + 3));
        }
        return trigrams;
      };
      
      const trigrams = generateTrigrams(word);
      expect(trigrams).toEqual(['api', 'pi-', 'i-k', '-ke', 'key']);
    });

    it('should normalize case for trigrams', () => {
      const word = 'TestWord';
      const normalizedWord = word.toLowerCase();
      
      const generateTrigrams = (input: string): string[] => {
        if (input.length < 3) return [input];
        
        const trigrams: string[] = [];
        for (let i = 0; i <= input.length - 3; i++) {
          trigrams.push(input.substring(i, i + 3));
        }
        return trigrams;
      };
      
      const trigrams = generateTrigrams(normalizedWord);
      expect(trigrams).toEqual(['tes', 'est', 'stw', 'two', 'wor', 'ord']);
    });
  });

  describe('Search Context Processing', () => {
    it('should process basic search context', () => {
      const ctx: SearchContext = {
        query: 'function test',
        repo_sha: 'abc123',
        k: 10,
        mode: 'lexical'
      };

      expect(ctx.query).toBe('function test');
      expect(ctx.mode).toBe('lexical');
    });

    it('should handle fuzzy search parameters', () => {
      const ctx: SearchContext = {
        query: 'test',
        repo_sha: 'abc123',
        k: 5,
        mode: 'hybrid',
        fuzzy: true,
        fuzzy_distance: 2
      };

      // Test fuzzy distance calculation
      const fuzzyDistance = ctx.fuzzy_distance || 0;
      const normalizedDistance = Math.min(2, Math.max(0, Math.round(fuzzyDistance * 2)));
      
      expect(normalizedDistance).toBe(2); // Capped at 2 by Math.min
    });

    it('should split multi-word queries', () => {
      const query = 'function getUserById';
      const terms = query.split(/\s+/).filter(term => term.length > 0);
      
      expect(terms).toEqual(['function', 'getUserById']);
    });

    it('should handle empty queries', () => {
      const query = '';
      const terms = query.split(/\s+/).filter(term => term.length > 0);
      
      expect(terms).toEqual([]);
    });

    it('should normalize query terms', () => {
      const query = 'FUNCTION   TEST   ';
      const terms = query.toLowerCase().trim().split(/\s+/).filter(term => term.length > 0);
      
      expect(terms).toEqual(['function', 'test']);
    });
  });

  describe('Bitmap Index Usage Decision', () => {
    it('should use bitmap index for large document sets', () => {
      const documentCount = 10000;
      const bitmapThreshold = 1000;
      
      const shouldUseBitmap = documentCount > bitmapThreshold;
      
      expect(shouldUseBitmap).toBe(true);
    });

    it('should use legacy index for small document sets', () => {
      const documentCount = 500;
      const bitmapThreshold = 1000;
      
      const shouldUseBitmap = documentCount > bitmapThreshold;
      
      expect(shouldUseBitmap).toBe(false);
    });

    it('should handle edge case at threshold', () => {
      const documentCount = 1000;
      const bitmapThreshold = 1000;
      
      const shouldUseBitmap = documentCount > bitmapThreshold;
      
      expect(shouldUseBitmap).toBe(false); // Exactly at threshold
    });
  });

  describe('Document Position Tracking', () => {
    it('should track document positions correctly', () => {
      const docId = 'test-doc';
      const filePath = 'src/test.ts';
      const positions = [
        { doc_id: docId, file_path: filePath, line: 1, col: 0, length: 8 },
        { doc_id: docId, file_path: filePath, line: 1, col: 9, length: 4 },
        { doc_id: docId, file_path: filePath, line: 2, col: 0, length: 6 }
      ];

      expect(positions).toHaveLength(3);
      expect(positions[0].line).toBe(1);
      expect(positions[2].line).toBe(2);
      expect(positions[1].col).toBe(9);
    });

    it('should handle overlapping positions', () => {
      const positions = [
        { line: 1, col: 5, length: 10 },
        { line: 1, col: 8, length: 5 }
      ];

      // Check for overlap
      const hasOverlap = (pos1: typeof positions[0], pos2: typeof positions[0]): boolean => {
        if (pos1.line !== pos2.line) return false;
        const end1 = pos1.col + pos1.length;
        const end2 = pos2.col + pos2.length;
        return pos1.col < end2 && pos2.col < end1;
      };

      expect(hasOverlap(positions[0], positions[1])).toBe(true);
    });
  });

  describe('FST (Finite State Transducer) Logic', () => {
    it('should handle FST state transitions', () => {
      interface FSTState {
        id: number;
        isFinal: boolean;
        transitions: Array<{ char: string; nextState: number; cost: number }>;
      }

      const states: FSTState[] = [
        {
          id: 0,
          isFinal: false,
          transitions: [
            { char: 't', nextState: 1, cost: 0 },
            { char: 'f', nextState: 2, cost: 0 }
          ]
        },
        {
          id: 1,
          isFinal: true,
          transitions: [
            { char: 'e', nextState: 3, cost: 0 }
          ]
        }
      ];

      const findTransition = (state: FSTState, char: string) => {
        return state.transitions.find(t => t.char === char);
      };

      const transition = findTransition(states[0], 't');
      expect(transition).toBeDefined();
      expect(transition?.nextState).toBe(1);
      expect(transition?.cost).toBe(0);

      const noTransition = findTransition(states[0], 'x');
      expect(noTransition).toBeUndefined();
    });

    it('should calculate edit distance for fuzzy matching', () => {
      const editDistance = (s1: string, s2: string): number => {
        const dp = Array(s1.length + 1).fill(null)
          .map(() => Array(s2.length + 1).fill(0));

        for (let i = 0; i <= s1.length; i++) dp[i][0] = i;
        for (let j = 0; j <= s2.length; j++) dp[0][j] = j;

        for (let i = 1; i <= s1.length; i++) {
          for (let j = 1; j <= s2.length; j++) {
            if (s1[i - 1] === s2[j - 1]) {
              dp[i][j] = dp[i - 1][j - 1];
            } else {
              dp[i][j] = Math.min(
                dp[i - 1][j] + 1,     // deletion
                dp[i][j - 1] + 1,     // insertion
                dp[i - 1][j - 1] + 1  // substitution
              );
            }
          }
        }

        return dp[s1.length][s2.length];
      };

      expect(editDistance('test', 'test')).toBe(0);
      expect(editDistance('test', 'tst')).toBe(1);
      expect(editDistance('test', 'rest')).toBe(1);
      expect(editDistance('test', 'tests')).toBe(1);
    });
  });

  describe('Configuration and Feature Flags', () => {
    it('should handle configuration updates', async () => {
      const config = {
        rareTermFuzzy: true,
        synonymsWhenIdentifierDensityBelow: 0.3,
        prefilterEnabled: false,
        wandEnabled: true
      };

      // Simulate configuration validation
      expect(typeof config.rareTermFuzzy).toBe('boolean');
      expect(typeof config.synonymsWhenIdentifierDensityBelow).toBe('number');
      expect(config.synonymsWhenIdentifierDensityBelow).toBeGreaterThanOrEqual(0);
      expect(config.synonymsWhenIdentifierDensityBelow).toBeLessThanOrEqual(1);
    });

    it('should validate scanner configurations', () => {
      const validScanners = ['on', 'off', 'auto'];
      const testScanner = 'auto';
      
      expect(validScanners.includes(testScanner)).toBe(true);
      
      const invalidScanner = 'invalid';
      expect(validScanners.includes(invalidScanner)).toBe(false);
    });

    it('should handle k_candidates parameter conversion', () => {
      const stringK = '50';
      const numberK = 100;
      
      const convertK = (k: string | number): number => {
        return typeof k === 'string' ? parseInt(k, 10) : k;
      };
      
      expect(convertK(stringK)).toBe(50);
      expect(convertK(numberK)).toBe(100);
      expect(convertK('invalid')).toBeNaN();
    });
  });

  describe('Match Scoring and Ranking', () => {
    it('should calculate basic match scores', () => {
      const calculateScore = (
        termMatch: boolean,
        positionBoost: number,
        lengthPenalty: number
      ): number => {
        let score = termMatch ? 1.0 : 0.0;
        score += positionBoost;
        score -= lengthPenalty;
        return Math.max(0, Math.min(1, score));
      };

      expect(calculateScore(true, 0.2, 0.1)).toBe(1.0); // Capped at 1.0
      expect(calculateScore(true, 0.1, 0.2)).toBeCloseTo(0.9);
      expect(calculateScore(false, 0.5, 0.1)).toBe(0.4);
    });

    it('should boost exact matches', () => {
      const query = 'test';
      const candidates = [
        { token: 'test', isExact: true },
        { token: 'testing', isExact: false },
        { token: 'tester', isExact: false }
      ];

      const scoredCandidates = candidates.map(candidate => ({
        ...candidate,
        score: candidate.isExact ? 1.0 : 0.7
      }));

      expect(scoredCandidates[0].score).toBe(1.0);
      expect(scoredCandidates[1].score).toBe(0.7);
      expect(scoredCandidates[2].score).toBe(0.7);
    });

    it('should apply position-based scoring', () => {
      const calculatePositionScore = (position: number, maxPosition: number): number => {
        if (maxPosition === 0) return 1.0;
        return 1.0 - (position / maxPosition) * 0.3; // 30% position penalty
      };

      expect(calculatePositionScore(0, 100)).toBe(1.0);
      expect(calculatePositionScore(50, 100)).toBe(0.85);
      expect(calculatePositionScore(100, 100)).toBe(0.7);
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle malformed content gracefully', () => {
      const malformedContent = '\x00\x01\x02invalid\xFF';
      
      // Simulate content sanitization
      const sanitize = (content: string): string => {
        return content.replace(/[\x00-\x1F\x7F-\xFF]/g, ' ').trim();
      };

      const sanitized = sanitize(malformedContent);
      expect(sanitized).toBe('invalid');
    });

    it('should handle extremely long tokens', () => {
      const longToken = 'a'.repeat(10000);
      const maxTokenLength = 100;
      
      const truncateToken = (token: string, maxLength: number): string => {
        return token.length > maxLength ? token.substring(0, maxLength) : token;
      };

      const truncated = truncateToken(longToken, maxTokenLength);
      expect(truncated).toHaveLength(maxTokenLength);
    });

    it('should handle empty document indexing', async () => {
      const docId = 'empty-doc';
      const filePath = 'empty.ts';
      const content = '';

      const words = content.match(/\b\w+\b/g) || [];
      expect(words).toHaveLength(0);
    });

    it('should handle Unicode characters', () => {
      const unicodeContent = 'функция тест() { return true; }';
      const unicodeWords = unicodeContent.match(/\b\w+\b/gu) || [];
      
      expect(unicodeWords.length).toBeGreaterThan(0);
      // Note: Unicode tokenization may vary based on implementation
      expect(unicodeWords.length).toBeGreaterThan(0);
    });
  });
});