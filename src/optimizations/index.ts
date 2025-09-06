/**
 * Lens Search Optimization Systems - Main Export
 * 
 * Four durable, embedder-agnostic search optimizations:
 * 1. Clone-Aware Recall - Token shingle expansion across forks/backports
 * 2. Learning-to-Stop - ML-based early termination for scanners and ANN
 * 3. Targeted Diversity - Constrained MMR for overview queries only
 * 4. TTL That Follows Churn - Adaptive caching with span invalidation
 * 
 * All systems implement SLA compliance validation per TODO.md requirements.
 */

// Core optimization systems
export { CloneAwareRecallSystem } from './clone-aware-recall.js';
export { LearningToStopSystem } from './learning-to-stop.js';
export { TargetedDiversitySystem, type DiversityFeatures } from './targeted-diversity.js';
export { ChurnAwareTTLSystem } from './churn-aware-ttl.js';

// Integration and orchestration
export { 
  OptimizationEngine,
  type OptimizationConfig,
  type OptimizationPipeline,
  type OptimizationPerformanceImpact,
  type SLAMetrics,
  type SystemHealthStatus
} from './optimization-engine.js';

// Performance monitoring
export { 
  PerformanceMonitor,
  type BenchmarkResult,
  type MonitoringConfig
} from './performance-monitor.js';

// Configuration presets
export const OPTIMIZATION_PRESETS = {
  /**
   * Production configuration with all optimizations enabled
   * Recommended for production deployments with full SLA compliance
   */
  PRODUCTION: {
    clone_aware_enabled: true,
    learning_to_stop_enabled: true,
    targeted_diversity_enabled: true,
    churn_aware_ttl_enabled: true,
    performance_monitoring_enabled: true,
    graceful_degradation_enabled: true,
  } as OptimizationConfig,
  
  /**
   * Development configuration with reduced optimizations
   * Suitable for development environments with faster iteration
   */
  DEVELOPMENT: {
    clone_aware_enabled: true,
    learning_to_stop_enabled: false, // Disable ML-based optimization in dev
    targeted_diversity_enabled: true,
    churn_aware_ttl_enabled: false, // Use simpler caching in dev
    performance_monitoring_enabled: false,
    graceful_degradation_enabled: true,
  } as OptimizationConfig,
  
  /**
   * Testing configuration with all optimizations for validation
   * Used in CI/CD and comprehensive testing scenarios
   */
  TESTING: {
    clone_aware_enabled: true,
    learning_to_stop_enabled: true,
    targeted_diversity_enabled: true,
    churn_aware_ttl_enabled: true,
    performance_monitoring_enabled: true,
    graceful_degradation_enabled: true,
  } as OptimizationConfig,
  
  /**
   * Minimal configuration for debugging and baseline comparison
   * All optimizations disabled for measuring baseline performance
   */
  BASELINE: {
    clone_aware_enabled: false,
    learning_to_stop_enabled: false,
    targeted_diversity_enabled: false,
    churn_aware_ttl_enabled: false,
    performance_monitoring_enabled: false,
    graceful_degradation_enabled: true,
  } as OptimizationConfig,
} as const;

// Monitoring configuration presets
export const MONITORING_PRESETS = {
  /**
   * Production monitoring with real-time alerts
   */
  PRODUCTION: {
    benchmark_interval_ms: 60000, // 1 minute
    alert_threshold_violations: 3,
    performance_degradation_threshold: 0.2,
    enable_real_time_monitoring: true,
    enable_alerting: true,
    log_level: 'warn',
  } as MonitoringConfig,
  
  /**
   * Development monitoring with detailed logging
   */
  DEVELOPMENT: {
    benchmark_interval_ms: 300000, // 5 minutes
    alert_threshold_violations: 5,
    performance_degradation_threshold: 0.5,
    enable_real_time_monitoring: false,
    enable_alerting: false,
    log_level: 'debug',
  } as MonitoringConfig,
  
  /**
   * Testing monitoring for CI/CD validation
   */
  TESTING: {
    benchmark_interval_ms: 10000, // 10 seconds
    alert_threshold_violations: 1,
    performance_degradation_threshold: 0.1,
    enable_real_time_monitoring: true,
    enable_alerting: false,
    log_level: 'info',
  } as MonitoringConfig,
} as const;

/**
 * Initialize optimization engine with preset configuration
 */
export async function createOptimizationEngine(
  preset: keyof typeof OPTIMIZATION_PRESETS = 'PRODUCTION'
): Promise<OptimizationEngine> {
  const config = OPTIMIZATION_PRESETS[preset];
  const engine = new OptimizationEngine(config);
  await engine.initialize();
  return engine;
}

/**
 * Initialize performance monitor with preset configuration
 */
export function createPerformanceMonitor(
  engine: OptimizationEngine,
  preset: keyof typeof MONITORING_PRESETS = 'PRODUCTION'
): PerformanceMonitor {
  const config = MONITORING_PRESETS[preset];
  return new PerformanceMonitor(engine, config);
}

/**
 * Validate SLA compliance for a pipeline result
 * Utility function for external SLA monitoring
 */
export function validatePipelineSLA(pipeline: OptimizationPipeline): {
  compliant: boolean;
  violations: string[];
} {
  const violations: string[] = [];
  
  // Check recall degradation
  if (pipeline.performance_impact.recall_change < 0) {
    violations.push('Recall degradation detected');
  }
  
  // Check overall optimization time budget
  if (pipeline.performance_impact.total_optimization_ms > 100) {
    violations.push('Total optimization time exceeds budget');
  }
  
  // Check clone expansion latency
  if (pipeline.performance_impact.clone_expansion_ms > 0.6) {
    violations.push('Clone expansion exceeds latency budget');
  }
  
  // Check diversity improvement requirement (if applied)
  if (pipeline.optimizations_applied.includes('targeted_diversity')) {
    if (pipeline.performance_impact.diversity_improvement < 0.1) {
      violations.push('Diversity improvement below target');
    }
  }
  
  return {
    compliant: violations.length === 0,
    violations,
  };
}

/**
 * System health check utility
 */
export function checkSystemHealth(engine: OptimizationEngine): {
  healthy: boolean;
  issues: string[];
  degraded_systems: string[];
} {
  const health = engine.getSystemHealth();
  const issues: string[] = [];
  
  if (!health.overall_healthy) {
    issues.push('Overall system health degraded');
  }
  
  if (health.degraded_optimizations.length > 0) {
    issues.push(`${health.degraded_optimizations.length} optimization systems degraded`);
  }
  
  return {
    healthy: health.overall_healthy,
    issues,
    degraded_systems: health.degraded_optimizations,
  };
}

/**
 * Quick optimization engine setup for common use cases
 */
export async function setupOptimizedSearch(environment: 'production' | 'development' | 'testing' = 'production') {
  const presetMap = {
    production: 'PRODUCTION',
    development: 'DEVELOPMENT', 
    testing: 'TESTING',
  } as const;
  
  const engine = await createOptimizationEngine(presetMap[environment]);
  
  let monitor: PerformanceMonitor | undefined;
  if (environment !== 'development') {
    monitor = createPerformanceMonitor(engine, presetMap[environment]);
    await monitor.startMonitoring();
  }
  
  return {
    engine,
    monitor,
    async shutdown() {
      if (monitor) {
        monitor.stopMonitoring();
      }
      await engine.shutdown();
    }
  };
}

// Re-export types for convenience
export type {
  SearchHit,
  MatchReason,
} from '../core/span_resolver/types.js';

export type {
  SearchContext,
} from '../types/core.js';