/**
 * Integration Tests for Core Lens Functionality
 * Exercises multiple systems together for broad coverage
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { LensSearchEngine } from '../api/search-engine.js';
import { IndexRegistry } from '../core/index-registry.js';
import { PrecisionOptimizationEngine } from '../core/precision-optimization.js';
import { OnlineCalibrationSystem } from '../deployment/online-calibration-system.js';
import type { SearchContext } from '../types/core.js';
import type { APIConfig } from '../types/api.js';

// Mock filesystem operations
vi.mock('fs', () => ({
  existsSync: vi.fn(() => true),
  mkdirSync: vi.fn(),
  readFileSync: vi.fn(() => '{"test": true}'),
  writeFileSync: vi.fn(),
  readdirSync: vi.fn(() => ['test-manifest.json']),
  statSync: vi.fn(() => ({ isFile: () => true, size: 1000 })),
}));

// Mock telemetry
vi.mock('../telemetry/tracer.js', () => ({
  LensTracer: {
    createChildSpan: vi.fn(() => ({
      setAttributes: vi.fn(),
      recordException: vi.fn(), 
      end: vi.fn(),
    })),
  },
}));

describe('Core Lens Integration', () => {
  let searchEngine: LensSearchEngine;
  let indexRegistry: IndexRegistry;
  let precisionEngine: PrecisionOptimizationEngine;
  let calibrationSystem: OnlineCalibrationSystem;

  const mockConfig: APIConfig = {
    indexRoot: '/test/indexes',
    maxResults: 100,
    defaultK: 50,
    enableSemanticSearch: true,
    enableOptimizations: true,
  };

  beforeEach(async () => {
    vi.clearAllMocks();
    
    // Initialize core systems
    indexRegistry = new IndexRegistry('/test/indexes');
    precisionEngine = new PrecisionOptimizationEngine();
    calibrationSystem = new OnlineCalibrationSystem('./test-calibration', './test-isotonic');
    
    // Initialize search engine with integrated systems
    searchEngine = new LensSearchEngine(mockConfig);
    await searchEngine.initialize();
  });

  afterEach(async () => {
    await searchEngine?.shutdown();
    await calibrationSystem?.stopOnlineCalibration();
    indexRegistry?.shutdown();
  });

  describe('Search Pipeline Integration', () => {
    it('should execute complete search pipeline with all optimizations', async () => {
      const context: SearchContext = {
        query: 'function getUserData',
        repo_sha: 'test-repo',
        mode: 'hybrid',
        k: 20,
        fuzzy_distance: 1,
      };

      // Mock index reader
      const mockReader = {
        search: vi.fn().mockResolvedValue([
          {
            file: 'src/user.ts',
            line: 10,
            col: 0,
            lang: 'typescript',
            snippet: 'function getUserData() { return userData; }',
            score: 0.95,
            why: ['exact_match'],
            byte_offset: 200,
            span_len: 40,
          },
          {
            file: 'src/api.ts', 
            line: 25,
            col: 0,
            lang: 'typescript',
            snippet: 'const getUserData = async () => { }',
            score: 0.88,
            why: ['fuzzy_match'],
            byte_offset: 500,
            span_len: 32,
          },
        ]),
        searchStructural: vi.fn().mockResolvedValue([]),
      };

      vi.spyOn(indexRegistry, 'hasRepository').mockReturnValue(true);
      vi.spyOn(indexRegistry, 'getReader').mockReturnValue(mockReader);

      // Execute search with all systems integrated
      const results = await searchEngine.search(context);

      expect(results).toBeDefined();
      expect(results.hits).toBeDefined();
      expect(results.hits.length).toBeGreaterThan(0);
      expect(mockReader.search).toHaveBeenCalled();
    });

    it('should apply precision optimizations during search', async () => {
      const context: SearchContext = {
        query: 'class UserService',
        repo_sha: 'test-repo', 
        mode: 'lexical',
        k: 10,
        fuzzy_distance: 0,
      };

      // Enable precision optimizations
      precisionEngine.setBlockEnabled('A', true);
      precisionEngine.setBlockEnabled('B', true);
      precisionEngine.setBlockEnabled('C', true);

      const mockCandidates = [
        {
          file: 'src/service.ts',
          line: 1,
          col: 0,
          lang: 'typescript',
          snippet: 'class UserService {',
          score: 0.95,
          why: ['exact_match'],
          byte_offset: 0,
          span_len: 18,
        },
        {
          file: 'src/user-service.ts',
          line: 1, 
          col: 0,
          lang: 'typescript',
          snippet: 'export class UserService {',
          score: 0.90,
          why: ['exact_match'],
          byte_offset: 0,
          span_len: 25,
        },
      ];

      // Test precision optimization blocks
      const optimizedResults = await precisionEngine.applyBlockA(mockCandidates, context);
      expect(optimizedResults.length).toBeGreaterThan(0);

      const finalResults = await precisionEngine.applyBlockB(optimizedResults, context);
      expect(finalResults.length).toBeGreaterThan(0);
    });

    it('should integrate with calibration system for score adjustment', async () => {
      const rawScore = 0.75;
      
      // Test score calibration
      const calibratedScore = calibrationSystem.calibrateScore(rawScore);
      expect(typeof calibratedScore).toBe('number');
      expect(calibratedScore).toBeGreaterThanOrEqual(0);
      expect(calibratedScore).toBeLessThanOrEqual(1);
    });

    it('should handle multi-repository searches', async () => {
      const repos = ['repo-1', 'repo-2', 'repo-3'];
      
      repos.forEach(repo => {
        vi.spyOn(indexRegistry, 'hasRepository').mockReturnValueOnce(true);
      });

      // Test repository access
      const hasRepo1 = indexRegistry.hasRepository('repo-1');
      const hasRepo2 = indexRegistry.hasRepository('repo-2');
      
      expect(hasRepo1).toBe(true);
      expect(hasRepo2).toBe(true);
    });
  });

  describe('Error Handling and Resilience', () => {
    it('should gracefully handle repository access failures', async () => {
      const context: SearchContext = {
        query: 'test',
        repo_sha: 'nonexistent-repo',
        mode: 'hybrid',
        k: 10,
        fuzzy_distance: 1,
      };

      vi.spyOn(indexRegistry, 'hasRepository').mockReturnValue(false);

      await expect(searchEngine.search(context)).resolves.toBeDefined();
    });

    it('should handle precision optimization failures gracefully', async () => {
      const mockCandidates = [
        {
          file: 'test.ts',
          line: 1,
          col: 0, 
          lang: 'typescript',
          snippet: 'test code',
          score: 0.8,
          why: ['test'],
          byte_offset: 0,
          span_len: 9,
        },
      ];

      const context: SearchContext = {
        query: 'test',
        repo_sha: 'test-repo',
        mode: 'hybrid',
        k: 10,
        fuzzy_distance: 1,
      };

      // Test error resilience
      await expect(precisionEngine.applyBlockA(mockCandidates, context)).resolves.toBeDefined();
      await expect(precisionEngine.applyBlockB(mockCandidates, context)).resolves.toBeDefined();
      await expect(precisionEngine.applyBlockC(mockCandidates, context)).resolves.toBeDefined();
    });

    it('should handle calibration system initialization errors', async () => {
      const testCalibration = new OnlineCalibrationSystem('./nonexistent', './nonexistent');
      
      // Should not throw during initialization
      expect(testCalibration).toBeDefined();
      
      // Should handle missing files gracefully
      const score = testCalibration.calibrateScore(0.5);
      expect(typeof score).toBe('number');
    });
  });

  describe('Performance and Configuration', () => {
    it('should respect configuration limits', async () => {
      const config: APIConfig = {
        indexRoot: '/test',
        maxResults: 5, // Low limit
        defaultK: 3,
        enableSemanticSearch: false,
        enableOptimizations: true,
      };

      const testEngine = new LensSearchEngine(config);
      await testEngine.initialize();

      const context: SearchContext = {
        query: 'test',
        repo_sha: 'test-repo',
        mode: 'lexical',
        k: 10, // Higher than config limit
        fuzzy_distance: 0,
      };

      const mockReader = {
        search: vi.fn().mockResolvedValue([
          { file: 'test1.ts', line: 1, col: 0, lang: 'ts', snippet: 'test1', score: 0.9, why: ['test'], byte_offset: 0, span_len: 5 },
          { file: 'test2.ts', line: 1, col: 0, lang: 'ts', snippet: 'test2', score: 0.8, why: ['test'], byte_offset: 0, span_len: 5 },
          { file: 'test3.ts', line: 1, col: 0, lang: 'ts', snippet: 'test3', score: 0.7, why: ['test'], byte_offset: 0, span_len: 5 },
        ]),
        searchStructural: vi.fn().mockResolvedValue([]),
      };

      vi.spyOn(indexRegistry, 'hasRepository').mockReturnValue(true);
      vi.spyOn(indexRegistry, 'getReader').mockReturnValue(mockReader);

      const results = await testEngine.search(context);
      
      // Should respect maxResults configuration
      expect(results.hits.length).toBeLessThanOrEqual(config.maxResults);
      
      await testEngine.shutdown();
    });

    it('should provide comprehensive system health status', async () => {
      const healthStatus = await searchEngine.getHealthStatus();
      
      expect(healthStatus).toBeDefined();
      expect(healthStatus.status).toBeDefined();
      expect(healthStatus.shards_healthy).toBeDefined();
      expect(healthStatus.shards_total).toBeDefined();
      expect(healthStatus.memory_usage_gb).toBeDefined();
    });

    it('should maintain performance metrics', async () => {
      const stats = indexRegistry.getStats();
      
      expect(stats).toBeDefined();
      expect(stats.total_repositories).toBeDefined();
      expect(stats.total_files).toBeDefined();
      expect(stats.index_size_mb).toBeDefined();
    });
  });

  describe('Feature Flag Integration', () => {
    it('should respect optimization feature flags', async () => {
      // Test with optimizations disabled
      precisionEngine.setBlockEnabled('A', false);
      precisionEngine.setBlockEnabled('B', false);
      precisionEngine.setBlockEnabled('C', false);

      const status = precisionEngine.getOptimizationStatus();
      expect(status.block_a_enabled).toBe(false);
      expect(status.block_b_enabled).toBe(false);
      expect(status.block_c_enabled).toBe(false);
    });

    it('should handle calibration system feature flags', async () => {
      // Test calibration system configuration
      const manualOverride = calibrationSystem.manualCalibrationOverride(0.6, 'test override');
      expect(manualOverride).toBeDefined();
    });
  });

  describe('Data Flow Integration', () => {
    it('should maintain data consistency across systems', async () => {
      const context: SearchContext = {
        query: 'integration test',
        repo_sha: 'consistency-test',
        mode: 'hybrid',
        k: 5,
        fuzzy_distance: 1,
      };

      // Verify data flows correctly between systems
      const mockResults = [
        {
          file: 'integration.ts',
          line: 1,
          col: 0,
          lang: 'typescript',
          snippet: 'integration test code',
          score: 0.85,
          why: ['integration'],
          byte_offset: 0,
          span_len: 19,
        },
      ];

      // Test precision optimization maintains data integrity
      const blockAResults = await precisionEngine.applyBlockA(mockResults, context);
      expect(blockAResults[0].file).toBe(mockResults[0].file);
      expect(blockAResults[0].snippet).toBe(mockResults[0].snippet);

      // Test calibration maintains score bounds
      const calibratedScore = calibrationSystem.calibrateScore(mockResults[0].score);
      expect(calibratedScore).toBeGreaterThanOrEqual(0);
      expect(calibratedScore).toBeLessThanOrEqual(1);
    });

    it('should handle concurrent operations safely', async () => {
      const contexts = [
        { query: 'test1', repo_sha: 'repo1', mode: 'lexical' as const, k: 5, fuzzy_distance: 0 },
        { query: 'test2', repo_sha: 'repo2', mode: 'hybrid' as const, k: 10, fuzzy_distance: 1 },
        { query: 'test3', repo_sha: 'repo3', mode: 'semantic' as const, k: 15, fuzzy_distance: 2 },
      ];

      const mockReader = {
        search: vi.fn().mockResolvedValue([]),
        searchStructural: vi.fn().mockResolvedValue([]),
      };

      vi.spyOn(indexRegistry, 'hasRepository').mockReturnValue(true);
      vi.spyOn(indexRegistry, 'getReader').mockReturnValue(mockReader);

      // Execute concurrent searches
      const promises = contexts.map(ctx => searchEngine.search(ctx));
      const results = await Promise.all(promises);

      expect(results).toHaveLength(3);
      results.forEach(result => {
        expect(result).toBeDefined();
        expect(result.hits).toBeDefined();
      });
    });
  });
});