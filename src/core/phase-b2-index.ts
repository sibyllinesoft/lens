/**
 * Phase B2 Enhanced AST Cache System - Main Export
 * Complete integration of all Phase B2 optimizations for ~40% Stage-B performance improvement
 * 
 * Components:
 * - OptimizedASTCache: Enhanced AST caching with batch processing (50→200 files, stale-while-revalidate)
 * - StructuralPatternEngine: Precompiled pattern caching for faster symbol extraction
 * - CoverageTracker: LSIF/ctags coverage monitoring and reporting
 * - EnhancedSymbolSearchEngine: Complete integration with all optimizations
 * - FeatureFlagManager: Safe rollout system with A/B testing capabilities
 * 
 * Performance Targets:
 * - Stage-B Latency: 7ms → 3-4ms (≥40% improvement)
 * - Cache Hit Rate: >90% for hot files
 * - Batch Processing: >20 files/second throughput
 * - Memory Usage: <100MB for 200 cached files
 */

// Core optimized components
export { OptimizedASTCache, PERFORMANCE_PRESETS } from './optimized-ast-cache.js';
import { OptimizedASTCache, PERFORMANCE_PRESETS } from './optimized-ast-cache.js';
export type { 
  OptimizedCacheConfig, 
  BatchParseRequest, 
  BatchParseResult, 
  CacheMetrics 
} from './optimized-ast-cache.js';

export { StructuralPatternEngine, PATTERN_PRESETS } from './structural-pattern-engine.js';
export type { 
  StructuralPattern, 
  PatternMatchResult, 
  PatternMatch, 
  PatternEngineConfig 
} from './structural-pattern-engine.js';

export { CoverageTracker, COVERAGE_THRESHOLDS, COVERAGE_PRIORITIES } from './coverage-tracker.js';
export type { 
  CoverageMetrics, 
  FileIndexingStatus, 
  CoverageGap, 
  CoverageReport 
} from './coverage-tracker.js';

// Enhanced search engine with all optimizations
export { 
  EnhancedSymbolSearchEngine 
} from '../indexer/enhanced-symbols.js';
export type { 
  EnhancedSearchConfig, 
  StagePerformanceMetrics 
} from '../indexer/enhanced-symbols.js';

// Import for local use
import { 
  EnhancedSymbolSearchEngine 
} from '../indexer/enhanced-symbols.js';
import { FeatureFlagManager } from './feature-flags.js';
import { StructuralPatternEngine, PATTERN_PRESETS } from './structural-pattern-engine.js';
import { CoverageTracker, COVERAGE_THRESHOLDS } from './coverage-tracker.js';

// Feature flag system for safe deployment
export { 
  FeatureFlagManager,
  globalFeatureFlags as featureFlags
} from './feature-flags.js';
export type { 
  FeatureFlagConfig, 
  FeatureFlagMetrics,
  FeatureFlagOverride
} from './feature-flags.js';

// Performance verification
export { PhaseB2PerformanceVerifier } from '../scripts/verify-phase-b2-performance.js';

// Backwards compatibility with original AST cache
export { ASTCache } from './ast-cache.js';
export type { CachedAST } from './ast-cache.js';

/**
 * Phase B2 Factory Function
 * Creates a fully configured enhanced symbol search engine with all optimizations
 */
export function createEnhancedSymbolEngine(
  segmentStorage: any,
  options: {
    preset?: 'performance' | 'balanced' | 'memory_efficient';
    enableAllOptimizations?: boolean;
    stageBTargetMs?: number;
    featureFlags?: FeatureFlagManager;
  } = {}
) {
  const {
    preset = 'balanced',
    enableAllOptimizations = true,
    stageBTargetMs = 4,
    featureFlags = new FeatureFlagManager()
  } = options;

  // Create feature flag context
  const flagContext = { 
    userId: 'system', 
    language: 'typescript',
    timestamp: Date.now()
  };

  // Determine which features to enable
  const enabledFeatures = enableAllOptimizations ? {
    enhancedCache: featureFlags.isEnabled('stageCOptimizations', flagContext),
    structuralPatterns: featureFlags.isEnabled('isotonicCalibration', flagContext),
    coverageTracking: featureFlags.isEnabled('performanceMonitoring', flagContext),
    batchProcessing: featureFlags.isEnabled('qualityGating', flagContext),
  } : {
    enhancedCache: false,
    structuralPatterns: false,
    coverageTracking: false,
    batchProcessing: false,
  };

  const config = {
    cacheConfig: PERFORMANCE_PRESETS[preset],
    enableStructuralPatterns: enabledFeatures.structuralPatterns,
    enableCoverageTracking: enabledFeatures.coverageTracking,
    batchProcessingEnabled: enabledFeatures.batchProcessing,
    preloadHotFiles: enabledFeatures.enhancedCache,
    stageBTargetMs,
  };

  return new EnhancedSymbolSearchEngine(segmentStorage, config);
}

/**
 * Phase B2 Health Check
 * Verifies all components are working correctly
 */
export async function performPhaseB2HealthCheck(): Promise<{
  status: 'healthy' | 'degraded' | 'unhealthy';
  components: Record<string, boolean>;
  performance: {
    avgStageBLatency: number;
    cacheHitRate: number;
    coveragePercentage: number;
  };
  recommendations: string[];
}> {
  const results = {
    status: 'healthy' as 'healthy' | 'degraded' | 'unhealthy',
    components: {
      optimizedCache: false,
      patternEngine: false,
      coverageTracker: false,
      featureFlags: false,
    },
    performance: {
      avgStageBLatency: 0,
      cacheHitRate: 0,
      coveragePercentage: 0,
    },
    recommendations: [] as string[],
  };

  try {
    // Test OptimizedASTCache
    const cache = new OptimizedASTCache(PERFORMANCE_PRESETS.balanced);
    const testStart = Date.now();
    await cache.getAST('/health/test.ts', 'export const test = "health";', 'typescript');
    const cacheTime = Date.now() - testStart;
    const cacheMetrics = cache.getMetrics();
    await cache.shutdown();
    
    results.components.optimizedCache = cacheTime < 10; // Should be fast
    results.performance.cacheHitRate = cacheMetrics.hitRate;

    // Test StructuralPatternEngine
    const patternEngine = new StructuralPatternEngine();
    const patternStart = Date.now();
    await patternEngine.executePattern('ts-function-declarations', 'function test() {}', 'typescript');
    const patternTime = Date.now() - patternStart;
    
    results.components.patternEngine = patternTime < 5;

    // Test CoverageTracker
    const coverageTracker = new CoverageTracker();
    coverageTracker.recordFileIndexing('/health/test.ts', 'typescript', [], 1);
    const coverage = coverageTracker.getCurrentMetrics();
    coverageTracker.shutdown();
    
    results.components.coverageTracker = coverage.indexedFiles > 0;
    results.performance.coveragePercentage = coverage.coveragePercentage;

    // Test FeatureFlags
    const flags = new FeatureFlagManager();
    const flagResult = flags.isEnabled('stageCOptimizations', { userId: 'test' });
    results.components.featureFlags = typeof flagResult === 'boolean';

    // Calculate overall performance
    results.performance.avgStageBLatency = (cacheTime + patternTime) / 2;

    // Determine overall status
    const healthyComponents = Object.values(results.components).filter(Boolean).length;
    const totalComponents = Object.keys(results.components).length;
    
    if (healthyComponents === totalComponents && results.performance.avgStageBLatency <= 4) {
      results.status = 'healthy';
    } else if (healthyComponents >= totalComponents * 0.75) {
      results.status = 'degraded';
      results.recommendations.push('Some Phase B2 components are not functioning optimally');
    } else {
      results.status = 'unhealthy';
      results.recommendations.push('Multiple Phase B2 components are failing');
    }

    // Performance recommendations
    if (results.performance.avgStageBLatency > 4) {
      results.recommendations.push(`Stage-B latency ${results.performance.avgStageBLatency.toFixed(1)}ms exceeds 4ms target`);
    }

    if (results.performance.cacheHitRate < 80) {
      results.recommendations.push('Cache hit rate below 80% - consider cache preloading');
    }

  } catch (error) {
    results.status = 'unhealthy';
    results.recommendations.push(`Health check failed: ${error}`);
  }

  return results;
}

/**
 * Phase B2 Configuration Validator
 * Ensures optimal configuration for different environments
 */
export function validatePhaseB2Config(config: any): {
  valid: boolean;
  warnings: string[];
  errors: string[];
  recommendations: string[];
} {
  const result = {
    valid: true,
    warnings: [] as string[],
    errors: [] as string[],
    recommendations: [] as string[],
  };

  // Validate cache configuration
  if (config.cacheConfig) {
    if (config.cacheConfig.maxFiles > 500) {
      result.warnings.push('Cache maxFiles > 500 may use significant memory');
    }
    
    if (config.cacheConfig.maxFiles < 50) {
      result.warnings.push('Cache maxFiles < 50 may have low hit rate');
    }

    if (!config.cacheConfig.precompiledPatterns) {
      result.recommendations.push('Enable precompiled patterns for better performance');
    }
  }

  // Validate Stage-B target
  if (config.stageBTargetMs) {
    if (config.stageBTargetMs < 1) {
      result.errors.push('Stage-B target < 1ms is unrealistic');
      result.valid = false;
    }
    
    if (config.stageBTargetMs > 10) {
      result.warnings.push('Stage-B target > 10ms may not meet user expectations');
    }
  }

  // Validate feature combinations
  if (config.batchProcessingEnabled && !config.enableStructuralPatterns) {
    result.warnings.push('Batch processing without structural patterns may not be optimal');
  }

  return result;
}

// Default export for convenience
export default {
  createEnhancedSymbolEngine,
  performPhaseB2HealthCheck,
  validatePhaseB2Config,
  OptimizedASTCache,
  StructuralPatternEngine,
  CoverageTracker,
  EnhancedSymbolSearchEngine,
  FeatureFlagManager,
  PERFORMANCE_PRESETS,
  PATTERN_PRESETS,
  COVERAGE_THRESHOLDS,
};