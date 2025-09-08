/**
 * Unit Tests for Advanced Systems Types
 * Tests interfaces, type guards, and utility functions for advanced Lens systems
 */

import { describe, it, expect } from 'vitest';
import {
  type EnhancedSearchHit,
  type AdvancedSystemsTelemetry,
  type AdvancedSystemsDeploymentConfig,
  type LensAdvancedIntegration,
  type LegacySearchHit,
  type SystemComponent,
  type ComponentStatus,
  type SystemHealthReport,
  isEnhancedSearchHit,
  hasAdvancedMetadata,
} from '../advanced-systems.js';

describe('Advanced Systems Types - Interfaces', () => {
  describe('EnhancedSearchHit Interface', () => {
    const baseHit = {
      doc_id: 'test-doc',
      file_path: 'src/test.ts',
      line: 10,
      col: 5,
      score: 0.85,
      match_reasons: ['exact'],
      snippet: 'function testFunction() {',
    };

    it('should extend base SearchHit with advanced metadata', () => {
      const enhancedHit: EnhancedSearchHit = {
        ...baseHit,
        // Moniker linking metadata
        moniker_cluster_id: 'cluster-123',
        cross_repo_source: 'upstream-repo',
        centrality_score: 0.92,
        
        // Evolution mapping metadata
        evolution_events: [
          {
            type: 'rename',
            from_symbol: 'oldFunction',
            to_symbol: 'newFunction',
            confidence: 0.95
          }
        ],
        revision_projected: {
          original_line: 8,
          original_col: 3,
          projection_confidence: 0.88
        },
        
        // Query compiler metadata
        plan_operator: 'IndexScan',
        optimization_applied: true,
        cost_contribution: 0.15,
        
        // Enhanced why reasons
        why_detailed: [
          {
            reason: 'Symbol name exact match',
            confidence: 0.95,
            system: 'base',
            explanation: 'Direct symbol name match found'
          },
          {
            reason: 'Cross-repository reference',
            confidence: 0.78,
            system: 'moniker',
            explanation: 'Symbol found through moniker linking'
          }
        ]
      };

      expect(enhancedHit.moniker_cluster_id).toBe('cluster-123');
      expect(enhancedHit.evolution_events).toHaveLength(1);
      expect(enhancedHit.why_detailed).toHaveLength(2);
      expect(enhancedHit.optimization_applied).toBe(true);
    });

    it('should handle optional advanced metadata fields', () => {
      const minimalEnhanced: EnhancedSearchHit = {
        ...baseHit,
        moniker_cluster_id: 'cluster-456'
        // Other advanced fields omitted
      };

      expect(minimalEnhanced.moniker_cluster_id).toBe('cluster-456');
      expect(minimalEnhanced.evolution_events).toBeUndefined();
      expect(minimalEnhanced.plan_operator).toBeUndefined();
    });

    it('should validate evolution event types', () => {
      const hitWithEvolution: EnhancedSearchHit = {
        ...baseHit,
        evolution_events: [
          { type: 'rename', from_symbol: 'a', to_symbol: 'b', confidence: 0.9 },
          { type: 'move', from_symbol: 'c', to_symbol: 'd', confidence: 0.8 },
          { type: 'signature_change', from_symbol: 'e', to_symbol: 'f', confidence: 0.7 }
        ]
      };

      const validTypes = ['rename', 'move', 'signature_change'];
      hitWithEvolution.evolution_events?.forEach(event => {
        expect(validTypes).toContain(event.type);
        expect(event.confidence).toBeGreaterThan(0);
        expect(event.confidence).toBeLessThanOrEqual(1);
      });
    });

    it('should validate why_detailed system types', () => {
      const hitWithDetailedWhy: EnhancedSearchHit = {
        ...baseHit,
        why_detailed: [
          { reason: 'test', confidence: 0.9, system: 'moniker', explanation: 'test' },
          { reason: 'test', confidence: 0.8, system: 'evolution', explanation: 'test' },
          { reason: 'test', confidence: 0.7, system: 'compiler', explanation: 'test' },
          { reason: 'test', confidence: 0.6, system: 'base', explanation: 'test' }
        ]
      };

      const validSystems = ['moniker', 'evolution', 'compiler', 'base'];
      hitWithDetailedWhy.why_detailed?.forEach(why => {
        expect(validSystems).toContain(why.system);
      });
    });
  });

  describe('AdvancedSystemsTelemetry Interface', () => {
    const validTelemetry: AdvancedSystemsTelemetry = {
      timestamp: new Date('2024-01-15T10:30:00Z'),
      query_id: 'query-123',
      
      systems_enabled: {
        moniker_linking: true,
        evolution_mapping: true,
        query_compiler: false,
        regression_bisection: false
      },
      
      latency_breakdown: {
        base_search_ms: 45.2,
        moniker_expansion_ms: 12.8,
        evolution_mapping_ms: 8.5,
        query_compilation_ms: 0, // disabled
        total_overhead_ms: 21.3
      },
      
      quality_impact: {
        base_recall_at_50: 0.75,
        enhanced_recall_at_50: 0.83,
        base_precision_at_50: 0.68,
        enhanced_precision_at_50: 0.72,
        span_accuracy: 0.91
      },
      
      gate_compliance: {
        moniker_latency_budget: true,
        evolution_time_budget: true,
        compiler_performance_target: true,
        overall_sla_compliance: true
      },
      
      resources: {
        memory_overhead_mb: 45.2,
        cpu_overhead_percent: 12.5,
        cache_hit_rates: new Map([['moniker', 0.85], ['evolution', 0.78]]),
        index_utilization: new Map([['primary', 0.92], ['secondary', 0.67]])
      }
    };

    it('should define complete telemetry structure', () => {
      expect(validTelemetry.query_id).toBe('query-123');
      expect(validTelemetry.systems_enabled.moniker_linking).toBe(true);
      expect(validTelemetry.latency_breakdown.base_search_ms).toBe(45.2);
      expect(validTelemetry.quality_impact.enhanced_recall_at_50).toBeGreaterThan(
        validTelemetry.quality_impact.base_recall_at_50
      );
    });

    it('should handle Map types in resources', () => {
      expect(validTelemetry.resources.cache_hit_rates.get('moniker')).toBe(0.85);
      expect(validTelemetry.resources.index_utilization.get('primary')).toBe(0.92);
    });

    it('should validate quality metrics ranges', () => {
      const metrics = validTelemetry.quality_impact;
      
      expect(metrics.base_recall_at_50).toBeGreaterThanOrEqual(0);
      expect(metrics.base_recall_at_50).toBeLessThanOrEqual(1);
      expect(metrics.enhanced_recall_at_50).toBeGreaterThanOrEqual(0);
      expect(metrics.enhanced_recall_at_50).toBeLessThanOrEqual(1);
      expect(metrics.span_accuracy).toBeGreaterThanOrEqual(0);
      expect(metrics.span_accuracy).toBeLessThanOrEqual(1);
    });
  });

  describe('AdvancedSystemsDeploymentConfig Interface', () => {
    const validDeploymentConfig: AdvancedSystemsDeploymentConfig = {
      rollout: {
        strategy: 'canary',
        traffic_percentage: 10,
        rollback_threshold: {
          latency_regression_percent: 15,
          quality_regression_percent: 5,
          error_rate_threshold: 0.02
        },
        monitoring_duration_hours: 24
      },
      
      experimentation: {
        enabled: true,
        control_group_size: 50,
        treatment_groups: [
          {
            name: 'enhanced-moniker',
            config_overrides: {},
            traffic_allocation: 25
          },
          {
            name: 'full-advanced',
            config_overrides: {},
            traffic_allocation: 25
          }
        ]
      },
      
      production_gates: {
        min_recall_improvement_pp: 2,
        max_latency_regression_percent: 10,
        min_span_accuracy: 0.9,
        max_error_rate_increase: 0.01,
        required_test_coverage: 0.9,
        required_benchmark_runs: 100
      },
      
      monitoring: {
        metrics_collection_rate: 0.1,
        alert_thresholds: {
          p95_latency_ms: 500,
          p99_latency_ms: 1000,
          recall_at_50_minimum: 0.8,
          error_rate_maximum: 0.01,
          cache_miss_rate_maximum: 0.3
        },
        dashboard_refresh_interval: 30,
        retention_days: 30
      }
    };

    it('should define complete deployment configuration', () => {
      expect(validDeploymentConfig.rollout.strategy).toBe('canary');
      expect(validDeploymentConfig.experimentation.enabled).toBe(true);
      expect(validDeploymentConfig.production_gates.min_recall_improvement_pp).toBe(2);
    });

    it('should validate rollout strategy options', () => {
      const validStrategies = ['blue_green', 'canary', 'shadow', 'feature_flag'];
      expect(validStrategies).toContain(validDeploymentConfig.rollout.strategy);
    });

    it('should validate traffic percentages', () => {
      const totalTreatmentTraffic = validDeploymentConfig.experimentation.treatment_groups
        .reduce((sum, group) => sum + group.traffic_allocation, 0);
      
      expect(validDeploymentConfig.rollout.traffic_percentage).toBeGreaterThan(0);
      expect(validDeploymentConfig.rollout.traffic_percentage).toBeLessThanOrEqual(100);
      expect(totalTreatmentTraffic).toBeLessThanOrEqual(100);
    });

    it('should have reasonable monitoring thresholds', () => {
      const thresholds = validDeploymentConfig.monitoring.alert_thresholds;
      
      expect(thresholds.p95_latency_ms).toBeLessThan(thresholds.p99_latency_ms);
      expect(thresholds.recall_at_50_minimum).toBeGreaterThan(0);
      expect(thresholds.error_rate_maximum).toBeLessThan(0.1);
    });
  });

  describe('LensAdvancedIntegration Interface', () => {
    const validIntegration: LensAdvancedIntegration = {
      indexer_extensions: {
        lsif_moniker_extraction: true,
        symbol_lineage_tracking: true,
        cross_repo_reference_mapping: false,
        structural_diff_analysis: false
      },
      
      query_pipeline_hooks: {
        pre_search_optimization: true,
        post_search_enhancement: true,
        cross_repo_expansion: false,
        evolution_projection: false
      },
      
      storage_extensions: {
        moniker_cluster_storage: true,
        evolution_lineage_storage: false,
        cost_model_persistence: true,
        experiment_state_storage: true
      },
      
      observability_integration: {
        opentelemetry_spans: true,
        prometheus_metrics: true,
        structured_logging: true,
        distributed_tracing: false
      }
    };

    it('should define all integration points', () => {
      expect(validIntegration.indexer_extensions.lsif_moniker_extraction).toBe(true);
      expect(validIntegration.query_pipeline_hooks.pre_search_optimization).toBe(true);
      expect(validIntegration.storage_extensions.moniker_cluster_storage).toBe(true);
      expect(validIntegration.observability_integration.opentelemetry_spans).toBe(true);
    });

    it('should support partial enablement of features', () => {
      const partiallyEnabled = {
        ...validIntegration,
        indexer_extensions: {
          ...validIntegration.indexer_extensions,
          cross_repo_reference_mapping: false,
          structural_diff_analysis: false
        }
      };

      expect(partiallyEnabled.indexer_extensions.lsif_moniker_extraction).toBe(true);
      expect(partiallyEnabled.indexer_extensions.cross_repo_reference_mapping).toBe(false);
    });
  });

  describe('System Health and Status Types', () => {
    const validComponentStatus: ComponentStatus = {
      component: 'moniker',
      enabled: true,
      healthy: true,
      last_health_check: new Date('2024-01-15T10:30:00Z'),
      performance_grade: 'A',
      issues: [],
      metrics: {
        'latency_p95': 45.2,
        'cache_hit_rate': 0.85,
        'throughput_qps': 150
      }
    };

    const validHealthReport: SystemHealthReport = {
      overall_status: 'healthy',
      component_status: [validComponentStatus],
      gate_compliance: new Map([
        ['latency_sla', true],
        ['quality_gate', true],
        ['resource_utilization', false]
      ]),
      recommendations: [
        'Monitor resource utilization',
        'Consider scaling moniker service'
      ],
      generated_at: new Date('2024-01-15T10:30:00Z'),
      next_check_at: new Date('2024-01-15T11:30:00Z')
    };

    it('should validate SystemComponent type', () => {
      const validComponents: SystemComponent[] = ['moniker', 'evolution', 'compiler', 'bisection'];
      validComponents.forEach(component => {
        expect(['moniker', 'evolution', 'compiler', 'bisection']).toContain(component);
      });
    });

    it('should validate ComponentStatus structure', () => {
      expect(validComponentStatus.component).toBe('moniker');
      expect(validComponentStatus.enabled).toBe(true);
      expect(validComponentStatus.performance_grade).toBe('A');
      expect(validComponentStatus.issues).toEqual([]);
      expect(validComponentStatus.metrics['latency_p95']).toBe(45.2);
    });

    it('should validate performance grade options', () => {
      const validGrades = ['A', 'B', 'C', 'D', 'F'];
      expect(validGrades).toContain(validComponentStatus.performance_grade);
    });

    it('should validate SystemHealthReport structure', () => {
      expect(validHealthReport.overall_status).toBe('healthy');
      expect(validHealthReport.component_status).toHaveLength(1);
      expect(validHealthReport.gate_compliance.get('latency_sla')).toBe(true);
      expect(validHealthReport.recommendations).toHaveLength(2);
    });

    it('should validate overall status options', () => {
      const validStatuses = ['healthy', 'degraded', 'critical'];
      expect(validStatuses).toContain(validHealthReport.overall_status);
    });
  });
});

describe('Advanced Systems Types - Type Guards', () => {
  const baseHit = {
    doc_id: 'test-doc',
    file_path: 'src/test.ts',
    line: 10,
    col: 5,
    score: 0.85,
    match_reasons: ['exact'],
    snippet: 'function testFunction() {',
  };

  describe('isEnhancedSearchHit', () => {
    it('should return true for hits with moniker metadata', () => {
      const hitWithMoniker = {
        ...baseHit,
        moniker_cluster_id: 'cluster-123'
      };
      
      expect(isEnhancedSearchHit(hitWithMoniker)).toBe(true);
    });

    it('should return true for hits with evolution metadata', () => {
      const hitWithEvolution = {
        ...baseHit,
        evolution_events: [
          { type: 'rename', from_symbol: 'old', to_symbol: 'new', confidence: 0.9 }
        ]
      };
      
      expect(isEnhancedSearchHit(hitWithEvolution)).toBe(true);
    });

    it('should return true for hits with compiler metadata', () => {
      const hitWithCompiler = {
        ...baseHit,
        plan_operator: 'IndexScan'
      };
      
      expect(isEnhancedSearchHit(hitWithCompiler)).toBe(true);
    });

    it('should return false for base search hits', () => {
      expect(isEnhancedSearchHit(baseHit)).toBe(false);
    });

    it('should handle null and undefined inputs', () => {
      expect(isEnhancedSearchHit(null)).toBeFalsy();
      expect(isEnhancedSearchHit(undefined)).toBeFalsy();
    });

    it('should handle non-object inputs', () => {
      expect(isEnhancedSearchHit('string')).toBe(false);
      expect(isEnhancedSearchHit(123)).toBe(false);
      expect(isEnhancedSearchHit([])).toBe(false);
    });

    it('should return true for objects with multiple advanced properties', () => {
      const enhancedHit = {
        ...baseHit,
        moniker_cluster_id: 'cluster-123',
        evolution_events: [],
        plan_operator: 'IndexScan'
      };
      
      expect(isEnhancedSearchHit(enhancedHit)).toBe(true);
    });
  });

  describe('hasAdvancedMetadata', () => {
    it('should return true for hits with defined moniker metadata', () => {
      const hitWithMoniker = {
        ...baseHit,
        moniker_cluster_id: 'cluster-123'
      };
      
      expect(hasAdvancedMetadata(hitWithMoniker)).toBe(true);
    });

    it('should return true for hits with defined evolution metadata', () => {
      const hitWithEvolution = {
        ...baseHit,
        evolution_events: []
      };
      
      expect(hasAdvancedMetadata(hitWithEvolution)).toBe(true);
    });

    it('should return true for hits with defined compiler metadata', () => {
      const hitWithCompiler = {
        ...baseHit,
        plan_operator: 'SeqScan'
      };
      
      expect(hasAdvancedMetadata(hitWithCompiler)).toBe(true);
    });

    it('should return false for base search hits', () => {
      expect(hasAdvancedMetadata(baseHit)).toBe(false);
    });

    it('should return false for enhanced hits with undefined advanced metadata', () => {
      const enhancedButEmptyHit = {
        ...baseHit,
        moniker_cluster_id: undefined,
        evolution_events: undefined,
        plan_operator: undefined
      };
      
      expect(hasAdvancedMetadata(enhancedButEmptyHit)).toBe(false);
    });

    it('should return false for non-enhanced hits', () => {
      expect(hasAdvancedMetadata(baseHit)).toBe(false);
      expect(hasAdvancedMetadata(null)).toBeFalsy();
      expect(hasAdvancedMetadata(undefined)).toBeFalsy();
    });

    it('should handle empty arrays and empty strings as defined', () => {
      const hitWithEmptyEvolution = {
        ...baseHit,
        evolution_events: []
      };
      
      const hitWithEmptyCluster = {
        ...baseHit,
        moniker_cluster_id: ''
      };
      
      expect(hasAdvancedMetadata(hitWithEmptyEvolution)).toBe(true);
      expect(hasAdvancedMetadata(hitWithEmptyCluster)).toBe(true);
    });
  });
});

describe('Advanced Systems Types - Legacy Compatibility', () => {
  describe('LegacySearchHit Type', () => {
    it('should maintain compatibility with core SearchHit', () => {
      const legacyHit: import('../advanced-systems.js').LegacySearchHit = {
        doc_id: 'legacy-doc',
        file_path: 'src/legacy.ts',
        line: 20,
        col: 10,
        score: 0.75,
        match_reasons: ['fuzzy'],
        snippet: 'const legacyFunction = () => {};',
      };

      expect(legacyHit.doc_id).toBe('legacy-doc');
      expect(legacyHit.match_reasons).toContain('fuzzy');
    });
  });
});

describe('Advanced Systems Types - Type Compatibility', () => {
  it('should ensure EnhancedSearchHit extends base SearchHit', () => {
    const baseHit = {
      doc_id: 'test-doc',
      file_path: 'src/test.ts',
      line: 10,
      col: 5,
      score: 0.85,
      match_reasons: ['exact'],
      snippet: 'function testFunction() {',
    };

    const enhancedHit: EnhancedSearchHit = {
      ...baseHit,
      moniker_cluster_id: 'cluster-123',
      centrality_score: 0.92
    };

    // Should have all base properties
    expect(enhancedHit.doc_id).toBe(baseHit.doc_id);
    expect(enhancedHit.file_path).toBe(baseHit.file_path);
    expect(enhancedHit.line).toBe(baseHit.line);
    expect(enhancedHit.col).toBe(baseHit.col);
    expect(enhancedHit.score).toBe(baseHit.score);
    
    // Plus enhanced properties
    expect(enhancedHit.moniker_cluster_id).toBe('cluster-123');
    expect(enhancedHit.centrality_score).toBe(0.92);
  });

  it('should handle complex evolution event structures', () => {
    const enhancedHit: EnhancedSearchHit = {
      doc_id: 'test-doc',
      file_path: 'src/test.ts',
      line: 10,
      col: 5,
      score: 0.85,
      match_reasons: ['semantic'],
      evolution_events: [
        {
          type: 'rename',
          from_symbol: 'processUserData',
          to_symbol: 'processUserInput',
          confidence: 0.95
        },
        {
          type: 'signature_change',
          from_symbol: 'function(user: User)',
          to_symbol: 'function(user: UserInput, options: ProcessOptions)',
          confidence: 0.88
        }
      ]
    };

    expect(enhancedHit.evolution_events).toHaveLength(2);
    expect(enhancedHit.evolution_events?.[0].type).toBe('rename');
    expect(enhancedHit.evolution_events?.[1].confidence).toBe(0.88);
  });

  it('should support complex why_detailed explanations', () => {
    const enhancedHit: EnhancedSearchHit = {
      doc_id: 'test-doc',
      file_path: 'src/test.ts',
      line: 10,
      col: 5,
      score: 0.85,
      match_reasons: ['exact', 'semantic'],
      why_detailed: [
        {
          reason: 'Exact symbol name match',
          confidence: 1.0,
          system: 'base',
          explanation: 'Direct match of symbol name in query'
        },
        {
          reason: 'Cross-repository reference found',
          confidence: 0.85,
          system: 'moniker',
          explanation: 'Symbol is referenced from external repository through LSIF moniker'
        },
        {
          reason: 'Historical symbol usage',
          confidence: 0.72,
          system: 'evolution',
          explanation: 'Symbol has evolved from previous versions with similar usage patterns'
        }
      ]
    };

    expect(enhancedHit.why_detailed).toHaveLength(3);
    expect(enhancedHit.why_detailed?.[0].system).toBe('base');
    expect(enhancedHit.why_detailed?.[1].system).toBe('moniker');
    expect(enhancedHit.why_detailed?.[2].system).toBe('evolution');
    
    // All confidence scores should be valid
    enhancedHit.why_detailed?.forEach(why => {
      expect(why.confidence).toBeGreaterThan(0);
      expect(why.confidence).toBeLessThanOrEqual(1);
    });
  });
});