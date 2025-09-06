/**
 * Type definitions for advanced Lens systems
 * Extends core types to support cross-repo linking, evolution mapping, 
 * query compilation, and regression bisection capabilities
 */

export * from '../core/moniker-linking.js';
export * from '../core/evolution-mapping.js';
export * from '../core/query-compiler.js';
export * from '../core/regression-bisection.js';
export * from '../core/advanced-systems-integration.js';

// Re-export with aliases for backwards compatibility
export type {
  MonikerLinkingSystem as CrossRepoLinkingSystem,
  EvolutionMappingSystem as APIEvolutionSystem,
  QueryCompiler as CostBasedOptimizer,
  RegressionBisectionHarness as FactorialDebugger,
  AdvancedSystemsIntegration as EnhancedLensCore,
} from '../core/advanced-systems-integration.js';

// Extended search hit with advanced system metadata
export interface EnhancedSearchHit extends import('./core.js').SearchHit {
  // Moniker linking metadata
  moniker_cluster_id?: string;
  cross_repo_source?: string;
  centrality_score?: number;
  
  // Evolution mapping metadata
  evolution_events?: Array<{
    type: 'rename' | 'move' | 'signature_change';
    from_symbol: string;
    to_symbol: string;
    confidence: number;
  }>;
  revision_projected?: {
    original_line: number;
    original_col: number;
    projection_confidence: number;
  };
  
  // Query compiler metadata
  plan_operator?: string;
  optimization_applied?: boolean;
  cost_contribution?: number;
  
  // Enhanced why reasons
  why_detailed?: Array<{
    reason: string;
    confidence: number;
    system: 'moniker' | 'evolution' | 'compiler' | 'base';
    explanation: string;
  }>;
}

// Performance telemetry for advanced systems
export interface AdvancedSystemsTelemetry {
  timestamp: Date;
  query_id: string;
  
  // System utilization
  systems_enabled: {
    moniker_linking: boolean;
    evolution_mapping: boolean;
    query_compiler: boolean;
    regression_bisection: boolean;
  };
  
  // Performance breakdown
  latency_breakdown: {
    base_search_ms: number;
    moniker_expansion_ms: number;
    evolution_mapping_ms: number;
    query_compilation_ms: number;
    total_overhead_ms: number;
  };
  
  // Quality impact
  quality_impact: {
    base_recall_at_50: number;
    enhanced_recall_at_50: number;
    base_precision_at_50: number;
    enhanced_precision_at_50: number;
    span_accuracy: number;
  };
  
  // Gate compliance
  gate_compliance: {
    moniker_latency_budget: boolean;
    evolution_time_budget: boolean;
    compiler_performance_target: boolean;
    overall_sla_compliance: boolean;
  };
  
  // Resource utilization
  resources: {
    memory_overhead_mb: number;
    cpu_overhead_percent: number;
    cache_hit_rates: Map<string, number>;
    index_utilization: Map<string, number>;
  };
}

// Configuration schema for advanced systems deployment
export interface AdvancedSystemsDeploymentConfig {
  // Rollout strategy
  rollout: {
    strategy: 'blue_green' | 'canary' | 'shadow' | 'feature_flag';
    traffic_percentage: number;
    rollback_threshold: {
      latency_regression_percent: number;
      quality_regression_percent: number;
      error_rate_threshold: number;
    };
    monitoring_duration_hours: number;
  };
  
  // A/B testing configuration
  experimentation: {
    enabled: boolean;
    control_group_size: number;
    treatment_groups: Array<{
      name: string;
      config_overrides: Partial<import('../core/advanced-systems-integration.js').AdvancedSystemsConfig>;
      traffic_allocation: number;
    }>;
  };
  
  // Quality gates for production readiness
  production_gates: {
    min_recall_improvement_pp: number;
    max_latency_regression_percent: number;
    min_span_accuracy: number;
    max_error_rate_increase: number;
    required_test_coverage: number;
    required_benchmark_runs: number;
  };
  
  // Monitoring and alerting
  monitoring: {
    metrics_collection_rate: number;
    alert_thresholds: {
      p95_latency_ms: number;
      p99_latency_ms: number;
      recall_at_50_minimum: number;
      error_rate_maximum: number;
      cache_miss_rate_maximum: number;
    };
    dashboard_refresh_interval: number;
    retention_days: number;
  };
}

// Integration points with existing Lens architecture
export interface LensAdvancedIntegration {
  // Indexer extensions
  indexer_extensions: {
    lsif_moniker_extraction: boolean;
    symbol_lineage_tracking: boolean;
    cross_repo_reference_mapping: boolean;
    structural_diff_analysis: boolean;
  };
  
  // Query processing pipeline hooks
  query_pipeline_hooks: {
    pre_search_optimization: boolean;
    post_search_enhancement: boolean;
    cross_repo_expansion: boolean;
    evolution_projection: boolean;
  };
  
  // Storage layer extensions
  storage_extensions: {
    moniker_cluster_storage: boolean;
    evolution_lineage_storage: boolean;
    cost_model_persistence: boolean;
    experiment_state_storage: boolean;
  };
  
  // Telemetry and observability
  observability_integration: {
    opentelemetry_spans: boolean;
    prometheus_metrics: boolean;
    structured_logging: boolean;
    distributed_tracing: boolean;
  };
}

// Backward compatibility types
export type LegacySearchHit = import('./core.js').SearchHit;
export type LegacySymbolCandidate = import('../core/span_resolver/types.js').SymbolCandidate;

// Type guards for enhanced functionality
export function isEnhancedSearchHit(hit: any): hit is EnhancedSearchHit {
  return hit && typeof hit === 'object' && 
         ('moniker_cluster_id' in hit || 'evolution_events' in hit || 'plan_operator' in hit);
}

export function hasAdvancedMetadata(hit: any): boolean {
  return isEnhancedSearchHit(hit) && (
    hit.moniker_cluster_id !== undefined ||
    hit.evolution_events !== undefined ||
    hit.plan_operator !== undefined
  );
}

// Utility types for configuration management
export type SystemComponent = 'moniker' | 'evolution' | 'compiler' | 'bisection';

export interface ComponentStatus {
  component: SystemComponent;
  enabled: boolean;
  healthy: boolean;
  last_health_check: Date;
  performance_grade: 'A' | 'B' | 'C' | 'D' | 'F';
  issues: string[];
  metrics: Record<string, number>;
}

export interface SystemHealthReport {
  overall_status: 'healthy' | 'degraded' | 'critical';
  component_status: ComponentStatus[];
  gate_compliance: Map<string, boolean>;
  recommendations: string[];
  generated_at: Date;
  next_check_at: Date;
}