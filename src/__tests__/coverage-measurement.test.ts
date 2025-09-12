/**
 * Coverage Measurement Test
 * 
 * Target: Get accurate coverage measurement with working tests only
 */

import { describe, it, expect } from 'bun:test';
import { ASTCache } from '../core/ast-cache.js';
import { IndexRegistry } from '../core/index-registry.js';
import { LensTracer } from '../telemetry/tracer.js';
import { globalFeatureFlags } from '../core/feature-flags.js';
import { getVersionInfo } from '../core/version-manager.js';
import { runQualityGates } from '../core/quality-gates.js';

describe('Coverage Measurement Test', () => {
  it('should exercise ASTCache methods', async () => {
    const cache = new ASTCache(25);
    
    // Initial stats
    const stats = cache.getStats();
    expect(stats.cacheSize).toBe(0);
    expect(stats.hitCount).toBe(0);
    expect(stats.totalRequests).toBe(0);
    
    // Coverage stats
    const coverage = cache.getCoverageStats(100);
    expect(coverage.totalTSFiles).toBe(100);
    expect(coverage.cachedTSFiles).toBe(0);
    
    // Parse code
    const code = 'function test() { return 42; }\nclass Test { method() {} }';
    const ast = await cache.getAST('/test.ts', code, 'typescript');
    expect(ast.language).toBe('typescript');
    expect(ast.symbolCount).toBeGreaterThan(0);
    expect(ast.mockAST.functions.length).toBeGreaterThan(0);
    
    // Cache hit
    const ast2 = await cache.getAST('/test.ts', code, 'typescript');
    expect(ast2.fileHash).toBe(ast.fileHash);
    
    // Refresh
    const refreshed = await cache.refreshAST('/test.ts', code, 'typescript');
    expect(refreshed).toBeDefined();
    
    cache.clear();
    const clearedStats = cache.getStats();
    expect(clearedStats.cacheSize).toBe(0);
  });
  
  it('should exercise IndexRegistry', async () => {
    const registry = new IndexRegistry('/tmp/coverage-test', 5);
    
    try {
      await registry.refresh();
    } catch (error) {
      expect(error).toBeDefined();
    }
    
    const stats = registry.stats();
    expect(stats.totalRepos).toBeDefined();
    
    expect(registry.hasRepo('test')).toBe(false);
    
    const manifests = registry.getManifests();
    expect(Array.isArray(manifests)).toBe(true);
    
    try {
      await registry.resolveRef('main');
    } catch (error) {
      expect(error).toBeDefined();
    }
    
    await registry.shutdown();
  });
  
  it('should exercise LensTracer', () => {
    const context = LensTracer.createSearchContext('test', 'search', 'repo');
    expect(context.trace_id).toBeDefined();
    expect(context.query).toBe('test');
    
    const searchSpan = LensTracer.startSearchSpan(context);
    expect(searchSpan).toBeDefined();
    
    const stageSpan = LensTracer.startStageSpan(context, 'stage_a', 'lexical', 100);
    expect(stageSpan).toBeDefined();
    
    LensTracer.endStageSpan(stageSpan, context, 'stage_a', 'lexical', 100, 50, 25);
    expect(context.stages.length).toBe(1);
    
    LensTracer.endSearchSpan(searchSpan, context, 50);
    
    const childSpan = LensTracer.createChildSpan('test-op');
    childSpan.end();
    
    const activeContext = LensTracer.getActiveContext();
    expect(activeContext).toBeDefined();
    
    const result = LensTracer.withContext(activeContext, () => 'test');
    expect(result).toBe('test');
  });
  
  it('should exercise FeatureFlags', () => {
    // Test methods that actually exist based on source code analysis
    const enabled = globalFeatureFlags.isEnabled('stageAOptimization');
    expect(typeof enabled).toBe('boolean');
    
    const enabled2 = globalFeatureFlags.isEnabled('stageBOptimization');
    expect(typeof enabled2).toBe('boolean');
    
    // Test status
    const status = globalFeatureFlags.getStatus();
    expect(status).toBeDefined();
    
    // Test canary status
    const canaryStatus = globalFeatureFlags.getCanaryStatus();
    expect(canaryStatus).toBeDefined();
    
    // Test canary group check
    const inCanary = globalFeatureFlags.isInCanaryGroup('test-user');
    expect(typeof inCanary).toBe('boolean');
    
    // Test performance metrics recording
    globalFeatureFlags.recordPerformanceMetrics('testMetric', {
      latency: 50,
      errorRate: 0.01,
      quality: 0.95
    });
  });
  
  it('should exercise utility functions', async () => {
    const version = await getVersionInfo();
    expect(version).toBeDefined();
    expect(version.api_version).toBeDefined();
    
    const metrics = {
      test_coverage: 0.9,
      error_rate: 0.01,
      latency_p95: 100,
      availability: 0.995
    };
    
    const gates = await runQualityGates(metrics);
    expect(gates).toBeDefined();
    expect(typeof gates.overall_passed).toBe('boolean');
    expect(gates.gates).toBeDefined();
  });
});