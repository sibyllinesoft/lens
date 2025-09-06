/**
 * Integration tests for all four optimization systems working together
 * 
 * Validates end-to-end performance requirements from TODO.md:
 * - Combined SLA compliance across all systems
 * - Coordination between clone expansion and diversity
 * - Learning-to-stop integration with search pipeline
 * - Churn-aware caching across all operations
 * - Performance compound effects and trade-offs
 */

import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { OptimizationEngine, type OptimizationConfig } from '../optimization-engine.js';
import type { SearchHit, MatchReason } from '../../core/span_resolver/types.js';
import type { SearchContext } from '../../types/core.js';
import type { DiversityFeatures } from '../targeted-diversity.js';

// Test data generation helpers
const generateCodeContent = (functionName: string, params: string[], body: string): string => {
  return `function ${functionName}(${params.join(', ')}) { ${body} }`;
};

const generateCloneVariants = (baseName: string, paramSets: string[][], bodies: string[]): Array<{content: string, file: string, repo: string}> => {
  return paramSets.map((params, i) => ({
    content: generateCodeContent(`${baseName}${i}`, params, bodies[i] || 'return true;'),
    file: `src/${baseName}_${i}.ts`,
    repo: `repo-${i}`,
  }));
};

const createMockSearchHit = (
  file: string, 
  line: number, 
  score: number, 
  symbolKind?: string, 
  why?: MatchReason[]
): SearchHit => ({
  file,
  line,
  col: 0,
  lang: 'typescript',
  snippet: `// Code at ${file}:${line}`,
  score,
  why: why || ['lexical'],
  byte_offset: line * 80,
  span_len: 50,
  symbol_kind: symbolKind as any,
  context_before: 'context before',
  context_after: 'context after',
});

describe('Optimization Systems Integration', () => {
  let engine: OptimizationEngine;
  let config: OptimizationConfig;
  
  beforeEach(async () => {
    config = {
      clone_aware_enabled: true,
      learning_to_stop_enabled: true,
      targeted_diversity_enabled: true,
      churn_aware_ttl_enabled: true,
      performance_monitoring_enabled: true,
      graceful_degradation_enabled: true,
    };
    
    engine = new OptimizationEngine(config);
    await engine.initialize();
  });
  
  afterEach(async () => {
    await engine.shutdown();
  });
  
  describe('End-to-End Pipeline Integration', () => {
    it('should coordinate clone expansion and targeted diversity without conflicts', async () => {
      // Set up scenario with clone opportunities and diversity needs
      const cloneVariants = generateCloneVariants(
        'processData',
        [['data', 'options'], ['input', 'config'], ['items', 'settings']],
        [
          'return data.map(item => transform(item, options));',
          'return input.filter(item => validate(item, config));',
          'return items.reduce((acc, item) => merge(acc, item, settings), {});',
        ]
      );
      
      // Index clone variants across different repos
      for (const variant of cloneVariants) {
        await engine.indexContent(
          variant.content,
          variant.file,
          1, 0,
          variant.repo,
          'function'
        );
      }
      
      // Create original hits with diversity opportunities
      const originalHits = [
        createMockSearchHit('src/processData_0.ts', 1, 95, 'function', ['exact']),
        createMockSearchHit('src/processData_0.ts', 20, 85, 'function', ['lexical']), // Same file - low diversity
        createMockSearchHit('lib/different.ts', 10, 80, 'class', ['ast']),
      ];
      
      const ctx = { query: 'process data overview', repo_sha: 'main', k: 20, timeout_ms: 1000, include_tests: false, languages: ['typescript'] };
      const diversityFeatures: DiversityFeatures = {
        query_type: 'NL_overview',
        topic_entropy: 0.9,
        result_count: originalHits.length,
        exact_matches: 1,
        structural_matches: 1,
        clone_collapsed: false, // Will be set to true after clone expansion
      };
      
      const pipeline = await engine.optimizeSearchResults(originalHits, ctx, diversityFeatures);
      
      // Verify pipeline coordination
      expect(pipeline.optimizations_applied).toContain('clone_aware_recall');
      expect(pipeline.optimizations_applied).toContain('targeted_diversity');
      
      // Clone expansion should come before diversity
      const cloneIndex = pipeline.optimizations_applied.indexOf('clone_aware_recall');
      const diversityIndex = pipeline.optimizations_applied.indexOf('targeted_diversity');
      expect(cloneIndex).toBeLessThan(diversityIndex);
      
      // Final results should be more diverse than clone-expanded results
      expect(pipeline.diversified_hits).toBeDefined();
      if (pipeline.clone_expanded_hits && pipeline.diversified_hits) {
        // Diversity should operate on clone-expanded results, not originals
        expect(pipeline.diversified_hits.length).toBeGreaterThanOrEqual(pipeline.clone_expanded_hits.length);
      }
    });
    
    it('should maintain SLA compliance across all optimizations', async () => {
      // Create comprehensive test scenario
      const testHits = Array.from({ length: 50 }, (_, i) => {
        const fileTypes = ['utils', 'helpers', 'services', 'components', 'types'];
        const symbolTypes = ['function', 'class', 'interface', 'variable', 'type'];
        
        return createMockSearchHit(
          `src/${fileTypes[i % fileTypes.length]}/file${i}.ts`,
          (i + 1) * 10,
          95 - i,
          symbolTypes[i % symbolTypes.length],
          i < 5 ? ['exact'] : i < 15 ? ['ast'] : ['lexical']
        );
      });
      
      const ctx = { query: 'comprehensive search query', repo_sha: 'main', k: 50, timeout_ms: 2000, include_tests: false, languages: ['typescript'] };
      const features: DiversityFeatures = {
        query_type: 'NL_overview',
        topic_entropy: 0.85,
        result_count: testHits.length,
        exact_matches: 5,
        structural_matches: 10,
        clone_collapsed: true,
      };
      
      const startTime = Date.now();
      const pipeline = await engine.optimizeSearchResults(testHits, ctx, features);
      const totalTime = Date.now() - startTime;
      
      // SLA Validation
      expect(pipeline.performance_impact.sla_compliance).toBe(true);
      
      // Recall should not be degraded (SLA-Recall@50 ≥ 0)
      expect(pipeline.performance_impact.recall_change).toBeGreaterThanOrEqual(0);
      
      // Overall latency should be reasonable
      expect(totalTime).toBeLessThan(100); // 100ms total budget
      
      // Individual system budgets should be respected
      expect(pipeline.performance_impact.clone_expansion_ms).toBeLessThanOrEqual(0.6);
      expect(pipeline.performance_impact.diversity_ms).toBeLessThan(50); // Reasonable diversity budget
    });
    
    it('should handle learning-to-stop decisions during search phase', () => {
      const ctx = { query: 'complex multi-term query', repo_sha: 'main', k: 20, timeout_ms: 1000, include_tests: false, languages: ['typescript'] };
      const queryStart = Date.now();
      
      // Test scanner stopping decision
      const scannerDecision = engine.shouldStopScanning(
        15, // blocks processed
        30, // candidates found
        12, // time spent ms
        ctx,
        queryStart
      );
      
      expect(scannerDecision.shouldStop).toBeDefined();
      expect(scannerDecision.confidence).toBeGreaterThanOrEqual(0);
      
      // Test ANN optimization
      const optimizedEf = engine.getOptimizedEfSearch(
        64, // current ef
        0.75, // recall achieved
        8, // time spent
        0.4, // risk level
        ctx,
        queryStart
      );
      
      expect(optimizedEf).toBeGreaterThan(0);
      expect(optimizedEf).toBeLessThanOrEqual(512);
    });
    
    it('should coordinate caching across all optimizations', async () => {
      const cacheKey = 'integration-test-key';
      const indexVersion = 'v1.0.0';
      const spanHash = 'integration123';
      
      let computeCount = 0;
      const expensiveComputation = async () => {
        computeCount++;
        await new Promise(resolve => setTimeout(resolve, 10)); // Simulate work
        return { result: `computed-${computeCount}` };
      };
      
      // First call should compute and cache
      const result1 = await engine.getCachedValue(
        cacheKey, indexVersion, spanHash, expensiveComputation, 'micro', 'integration'
      );
      expect(result1.result).toBe('computed-1');
      expect(computeCount).toBe(1);
      
      // Second call should hit cache
      const result2 = await engine.getCachedValue(
        cacheKey, indexVersion, spanHash, expensiveComputation, 'micro', 'integration'
      );
      expect(result2.result).toBe('computed-1'); // Same result
      expect(computeCount).toBe(1); // No additional computation
      
      // File change should affect TTL calculation
      engine.recordFileChange('src/integration-test.ts');
      
      // Different cache types should work independently
      const raptorResult = await engine.getCachedValue(
        cacheKey, indexVersion, spanHash, expensiveComputation, 'raptor', 'integration'
      );
      expect(raptorResult.result).toBe('computed-2'); // New computation for different cache
    });
  });
  
  describe('Performance Compound Effects', () => {
    it('should measure compound optimization benefits', async () => {
      // Create baseline with no optimizations
      await engine.shutdown();
      const noOptsConfig = {
        ...config,
        clone_aware_enabled: false,
        learning_to_stop_enabled: false,
        targeted_diversity_enabled: false,
        churn_aware_ttl_enabled: false,
      };
      const baselineEngine = new OptimizationEngine(noOptsConfig);
      await baselineEngine.initialize();
      
      const testHits = Array.from({ length: 20 }, (_, i) =>
        createMockSearchHit(`file${i}.ts`, i * 10, 90 - i, 'function')
      );
      
      const ctx = { query: 'compound test', repo_sha: 'main', k: 20, timeout_ms: 1000, include_tests: false, languages: ['typescript'] };
      
      // Baseline performance
      const baselineStart = Date.now();
      const baselinePipeline = await baselineEngine.optimizeSearchResults(testHits, ctx);
      const baselineTime = Date.now() - baselineStart;
      
      await baselineEngine.shutdown();
      
      // Full optimizations
      await engine.initialize();
      
      const optimizedStart = Date.now();
      const optimizedPipeline = await engine.optimizeSearchResults(testHits, ctx, {
        query_type: 'NL_overview',
        topic_entropy: 0.8,
        result_count: testHits.length,
        exact_matches: 5,
        structural_matches: 3,
        clone_collapsed: true,
      });
      const optimizedTime = Date.now() - optimizedStart;
      
      // Analyze compound effects
      const timeDifference = optimizedTime - baselineTime;
      const recallImprovement = optimizedPipeline.performance_impact.recall_change;
      const diversityImprovement = optimizedPipeline.performance_impact.diversity_improvement;
      
      // Optimizations should provide net positive value
      // Either improved recall/diversity or better performance (or both)
      const hasPositiveValue = recallImprovement > 0 || 
                              diversityImprovement > 0.05 || 
                              timeDifference < 0;
      
      expect(hasPositiveValue).toBe(true);
      
      // Should not significantly degrade performance
      expect(timeDifference).toBeLessThan(50); // 50ms penalty maximum
    });
    
    it('should handle optimization trade-offs gracefully', async () => {
      // Test scenario where optimizations might conflict
      const conflictingHits = [
        // High-scoring exact matches that might conflict with diversity
        createMockSearchHit('same-file.ts', 10, 95, 'function', ['exact']),
        createMockSearchHit('same-file.ts', 20, 94, 'function', ['exact']),
        createMockSearchHit('same-file.ts', 30, 93, 'function', ['lexical']),
        createMockSearchHit('different-file.ts', 10, 70, 'class', ['ast']),
      ];
      
      const ctx = { query: 'conflicting optimization test', repo_sha: 'main', k: 10, timeout_ms: 1000, include_tests: false, languages: ['typescript'] };
      const features: DiversityFeatures = {
        query_type: 'NL_overview',
        topic_entropy: 0.95, // Very high entropy - strong diversity preference
        result_count: conflictingHits.length,
        exact_matches: 2,
        structural_matches: 1,
        clone_collapsed: true,
      };
      
      const pipeline = await engine.optimizeSearchResults(conflictingHits, ctx, features);
      
      // Should maintain quality gates despite conflicts
      expect(pipeline.performance_impact.sla_compliance).toBe(true);
      
      // Should preserve exact matches (hard constraint)
      const finalExactMatches = pipeline.final_hits.filter(h => 
        h.why?.includes('exact')
      ).length;
      expect(finalExactMatches).toBeGreaterThanOrEqual(2);
      
      // Should achieve some diversity improvement despite high-scoring same-file matches
      if (pipeline.optimizations_applied.includes('targeted_diversity')) {
        expect(pipeline.performance_impact.diversity_improvement).toBeGreaterThan(0);
      }
    });
  });
  
  describe('System Health and Recovery', () => {
    it('should maintain system health under load', async () => {
      const initialHealth = engine.getSystemHealth();
      expect(initialHealth.overall_healthy).toBe(true);
      
      // Simulate load with multiple concurrent operations
      const loadPromises = [];
      
      for (let i = 0; i < 10; i++) {
        const hits = Array.from({ length: 5 }, (_, j) =>
          createMockSearchHit(`load-test-${i}-${j}.ts`, j * 10, 80 - j, 'function')
        );
        
        const ctx = { query: `load test ${i}`, repo_sha: 'load', k: 10, timeout_ms: 1000, include_tests: false, languages: ['typescript'] };
        
        loadPromises.push(engine.optimizeSearchResults(hits, ctx));
      }
      
      // Wait for all operations
      const results = await Promise.allSettled(loadPromises);
      
      // All operations should succeed
      const failures = results.filter(r => r.status === 'rejected');
      expect(failures.length).toBe(0);
      
      // System health should remain good
      const postLoadHealth = engine.getSystemHealth();
      expect(postLoadHealth.overall_healthy).toBe(true);
    });
    
    it('should recover from individual system failures', async () => {
      // This test simulates recovery scenarios
      // In a real implementation, we would inject failures
      
      await engine.performHealthCheckAndRecovery();
      
      const health = engine.getSystemHealth();
      expect(health.overall_healthy).toBe(true);
      
      // System should continue operating even with some degraded components
      const testHits = [createMockSearchHit('recovery-test.ts', 1, 95, 'function')];
      const ctx = { query: 'recovery test', repo_sha: 'test', k: 5, timeout_ms: 1000, include_tests: false, languages: ['typescript'] };
      
      const pipeline = await engine.optimizeSearchResults(testHits, ctx);
      
      // Should return valid results even during recovery
      expect(pipeline.final_hits).toBeDefined();
      expect(pipeline.final_hits.length).toBeGreaterThan(0);
    });
  });
  
  describe('Comprehensive Metrics and Monitoring', () => {
    it('should provide detailed performance metrics across all systems', async () => {
      // Generate some activity
      const testHits = [
        createMockSearchHit('metrics-test.ts', 1, 95, 'function'),
        createMockSearchHit('metrics-test.ts', 10, 85, 'class'),
      ];
      
      const ctx = { query: 'metrics test', repo_sha: 'metrics', k: 10, timeout_ms: 1000, include_tests: false, languages: ['typescript'] };
      
      await engine.optimizeSearchResults(testHits, ctx, {
        query_type: 'NL_overview',
        topic_entropy: 0.8,
        result_count: testHits.length,
        exact_matches: 1,
        structural_matches: 1,
        clone_collapsed: true,
      });
      
      const metrics = engine.getPerformanceMetrics();
      
      // Comprehensive metrics structure
      expect(metrics).toHaveProperty('system_health');
      expect(metrics).toHaveProperty('sla_compliance_rate');
      expect(metrics).toHaveProperty('average_optimization_time_ms');
      expect(metrics).toHaveProperty('subsystem_metrics');
      
      // Subsystem metrics
      expect(metrics.subsystem_metrics).toHaveProperty('clone_aware');
      expect(metrics.subsystem_metrics).toHaveProperty('learning_to_stop');
      expect(metrics.subsystem_metrics).toHaveProperty('targeted_diversity');
      expect(metrics.subsystem_metrics).toHaveProperty('churn_aware_ttl');
      
      // Health status
      expect(metrics.system_health).toHaveProperty('overall_healthy');
      expect(metrics.system_health).toHaveProperty('degraded_optimizations');
      
      // Performance tracking
      expect(metrics.sla_compliance_rate).toBeGreaterThanOrEqual(0);
      expect(metrics.sla_compliance_rate).toBeLessThanOrEqual(1);
      expect(metrics.average_optimization_time_ms).toBeGreaterThanOrEqual(0);
    });
    
    it('should track SLA compliance over time', async () => {
      // Run multiple optimization cycles
      for (let i = 0; i < 5; i++) {
        const hits = [createMockSearchHit(`sla-test-${i}.ts`, 1, 90, 'function')];
        const ctx = { query: `sla test ${i}`, repo_sha: 'sla', k: 5, timeout_ms: 1000, include_tests: false, languages: ['typescript'] };
        
        await engine.optimizeSearchResults(hits, ctx);
      }
      
      const metrics = engine.getPerformanceMetrics();
      
      // Should have processed multiple pipelines
      expect(metrics.pipelines_processed).toBeGreaterThan(0);
      
      // SLA compliance rate should be tracked
      expect(metrics.sla_compliance_rate).toBeDefined();
      expect(metrics.sla_compliance_rate).toBeGreaterThanOrEqual(0);
      expect(metrics.sla_compliance_rate).toBeLessThanOrEqual(1);
    });
  });
  
  describe('Real-World Scenarios', () => {
    it('should handle typical code search patterns', async () => {
      // Simulate realistic code search scenarios
      const scenarios = [
        {
          query: 'user authentication function',
          hits: [
            createMockSearchHit('auth/login.ts', 15, 95, 'function', ['exact']),
            createMockSearchHit('auth/verify.ts', 8, 85, 'function', ['lexical']),
            createMockSearchHit('utils/auth-helpers.ts', 22, 75, 'function', ['ast']),
          ],
          features: { query_type: 'targeted_search' as const, topic_entropy: 0.6 },
        },
        {
          query: 'overview of data processing utilities',
          hits: Array.from({ length: 15 }, (_, i) =>
            createMockSearchHit(`data/processor-${i}.ts`, i * 5, 90 - i * 2, 'function')
          ),
          features: { query_type: 'NL_overview' as const, topic_entropy: 0.9 },
        },
        {
          query: 'database connection interface',
          hits: [
            createMockSearchHit('db/connection.ts', 5, 92, 'interface', ['exact']),
            createMockSearchHit('db/pool.ts', 12, 88, 'class', ['ast']),
            createMockSearchHit('types/db.ts', 1, 80, 'type', ['lexical']),
          ],
          features: { query_type: 'targeted_search' as const, topic_entropy: 0.4 },
        },
      ];
      
      for (const scenario of scenarios) {
        const ctx = { 
          query: scenario.query, 
          repo_sha: 'realistic', 
          k: 20, 
          timeout_ms: 1000, 
          include_tests: false, 
          languages: ['typescript'] 
        };
        
        const features: DiversityFeatures = {
          query_type: scenario.features.query_type,
          topic_entropy: scenario.features.topic_entropy,
          result_count: scenario.hits.length,
          exact_matches: scenario.hits.filter(h => h.why?.includes('exact')).length,
          structural_matches: scenario.hits.filter(h => h.why?.includes('ast')).length,
          clone_collapsed: true,
        };
        
        const pipeline = await engine.optimizeSearchResults(scenario.hits, ctx, features);
        
        // Should handle each scenario appropriately
        expect(pipeline.final_hits).toBeDefined();
        expect(pipeline.final_hits.length).toBeGreaterThan(0);
        expect(pipeline.performance_impact.sla_compliance).toBe(true);
        
        // Targeted searches should not apply diversity
        if (scenario.features.query_type === 'targeted_search') {
          expect(pipeline.optimizations_applied).not.toContain('targeted_diversity');
        }
        
        // Overview searches with high entropy should apply diversity
        if (scenario.features.query_type === 'NL_overview' && scenario.features.topic_entropy > 0.6) {
          expect(pipeline.optimizations_applied).toContain('targeted_diversity');
        }
      }
    });
  });
});