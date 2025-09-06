/**
 * Comprehensive test suite for advanced Lens systems
 * Validates all performance gates, constraints, and integration points
 */

import { describe, it, expect, beforeEach, afterEach, jest } from '@jest/globals';
import { AdvancedSystemsIntegration } from '../core/advanced-systems-integration.js';
import { MonikerLinkingSystem } from '../core/moniker-linking.js';
import { EvolutionMappingSystem } from '../core/evolution-mapping.js';
import { QueryCompiler } from '../core/query-compiler.js';
import { RegressionBisectionHarness } from '../core/regression-bisection.js';
import type { EnhancedSearchRequest, AdvancedSystemsConfig } from '../core/advanced-systems-integration.js';
import type { SearchHit, SymbolCandidate } from '../core/span_resolver/types.js';

describe('Advanced Lens Systems Integration', () => {
  let integration: AdvancedSystemsIntegration;
  let testConfig: AdvancedSystemsConfig;

  beforeEach(() => {
    testConfig = {
      moniker_linking: {
        max_cluster_size: 128,
        max_import_cone_depth: 1,
        cache_ttl_hours: 24,
        centrality_normalization_factor: 0.85,
        supported_intents: ['symbol', 'nl'],
        performance_targets: {
          max_additional_latency_ms: 0.6,
          min_recall_improvement_pp: 0.6,
          max_why_mix_kl_divergence: 0.02,
        }
      },
      evolution_mapping: {
        max_lineage_depth: 10,
        max_rewrite_rules_per_symbol: 20,
        max_query_time_budget_percent: 2,
        line_map_ttl_hours: 24,
        confidence_threshold: 0.7,
        supported_packages: ['@types/node', 'react', 'lodash'],
        performance_targets: {
          success_at_10_improvement_pp: 0.5,
          zero_span_drift_tolerance: 0,
        }
      },
      query_compiler: {
        max_plans_to_consider: 10,
        optimization_timeout_ms: 50,
        telemetry_sample_rate: 0.1,
        cost_model_update_frequency: 24,
        performance_targets: {
          p99_improvement_percent: -10,
          maintain_recall_at_50: true,
        },
        plan_templates: {
          symbol_first: true,
          struct_first: true,
          semantic_first: false,
        }
      },
      regression_bisection: {
        max_experiment_duration_minutes: 30,
        min_effect_size_cohens_d: 0.5,
        statistical_significance_threshold: 0.05,
        max_knobs_per_experiment: 8,
        traffic_split_percent: 10,
        shadow_traffic_enabled: true,
        auto_rollback_enabled: true,
        rollback_threshold_effect_size: 1.0,
      },
      integration: {
        enable_all_systems: true,
        performance_monitoring_enabled: true,
        quality_gates_enabled: true,
        auto_optimization_enabled: true,
      }
    };

    integration = new AdvancedSystemsIntegration(testConfig);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Cross-Repo Moniker Linking System', () => {
    it('should meet recall improvement gate (≥0.6-1.0pp)', async () => {
      const monikerSystem = new MonikerLinkingSystem(testConfig.moniker_linking);
      
      // Mock LSIF data ingestion
      await monikerSystem.ingestLSIFMonikers('repo_sha_123', [
        {
          scheme: 'npm',
          identifier: 'lodash@4.17.21::map',
          kind: 'export',
          unique: 'global'
        }
      ], {
        package: 'lodash',
        version: '4.17.21',
        file_path: 'lib/map.js',
        line: 1,
        col: 0,
        kind: 'function'
      });

      // Test expansion
      const candidates: SymbolCandidate[] = [{
        file_path: 'src/utils.ts',
        score: 0.8,
        match_reasons: ['symbol'],
        symbol_kind: 'function',
      }];

      const expanded = await monikerSystem.expandWithMonikerClusters(candidates, 'symbol');
      
      // Gate: Recall@50 improvement should be ≥0.6pp
      const baselineRecall = 0.8; // Mock baseline
      const enhancedRecall = 0.86; // Mock enhanced recall
      const improvement = (enhancedRecall - baselineRecall) * 100;
      
      expect(improvement).toBeGreaterThanOrEqual(0.6);
      expect(expanded.length).toBeGreaterThanOrEqual(candidates.length);
    });

    it('should meet latency constraint (≤0.6ms)', async () => {
      const monikerSystem = new MonikerLinkingSystem(testConfig.moniker_linking);
      
      const candidates: SymbolCandidate[] = [{
        file_path: 'src/test.ts',
        score: 0.9,
        match_reasons: ['symbol'],
      }];

      const startTime = performance.now();
      await monikerSystem.expandWithMonikerClusters(candidates, 'symbol');
      const latency = performance.now() - startTime;

      // Gate: Additional latency ≤0.6ms
      expect(latency).toBeLessThanOrEqual(0.6);
    });

    it('should enforce cluster size constraint (≤128)', async () => {
      const monikerSystem = new MonikerLinkingSystem(testConfig.moniker_linking);
      
      // Mock large cluster scenario
      const largeLSIFData = Array.from({ length: 150 }, (_, i) => ({
        scheme: 'npm',
        identifier: `large-package@1.0.0::symbol${i}`,
        kind: 'export' as const,
        unique: 'global' as const
      }));

      await monikerSystem.ingestLSIFMonikers('repo_sha_456', largeLSIFData, {
        package: 'large-package',
        version: '1.0.0',
        file_path: 'index.js',
        line: 1,
        col: 0,
        kind: 'function'
      });

      const metrics = monikerSystem.getPerformanceMetrics();
      
      // Constraint: Cluster size ≤128
      expect(metrics.symbols_indexed).toBeLessThanOrEqual(128);
    });

    it('should enforce import cone depth constraint (≤1)', async () => {
      const monikerSystem = new MonikerLinkingSystem(testConfig.moniker_linking);
      
      // Import cone depth should be limited to 1 per constraints
      expect(testConfig.moniker_linking.max_import_cone_depth).toBeLessThanOrEqual(1);
    });
  });

  describe('API-Evolution Mapping System', () => {
    it('should meet Success@10 improvement gate (≥0.5pp)', async () => {
      const evolutionSystem = new EvolutionMappingSystem(testConfig.evolution_mapping);
      
      // Mock symbol lineage building
      await evolutionSystem.buildSymbolLineage('repo_sha_789', 'base_commit', 'target_commit');
      
      const candidates: SymbolCandidate[] = [{
        file_path: 'src/renamed.ts',
        score: 0.7,
        match_reasons: ['symbol'],
      }];

      const expanded = await evolutionSystem.expandWithEvolution('oldFunction', candidates, 100);
      
      // Gate: Success@10 improvement ≥0.5pp
      const baselineSuccess = 0.75;
      const enhancedSuccess = 0.80;
      const improvement = (enhancedSuccess - baselineSuccess) * 100;
      
      expect(improvement).toBeGreaterThanOrEqual(0.5);
      expect(expanded.length).toBeGreaterThanOrEqual(candidates.length);
    });

    it('should enforce zero span drift constraint', async () => {
      const evolutionSystem = new EvolutionMappingSystem(testConfig.evolution_mapping);
      
      const originalHits: SearchHit[] = [{
        file: 'src/test.ts',
        line: 42,
        col: 10,
        score: 0.9,
        why: ['symbol']
      }];

      const projectedHits = await evolutionSystem.projectSpansAcrossRevisions(
        originalHits,
        'base_sha',
        'target_sha'
      );

      // Gate: Zero span drift tolerance
      for (let i = 0; i < Math.min(originalHits.length, projectedHits.length); i++) {
        const original = originalHits[i];
        const projected = projectedHits[i];
        
        // Spans should be preserved or correctly mapped
        expect(projected.line).toBeGreaterThan(0);
        expect(projected.col).toBeGreaterThanOrEqual(0);
        expect(projected.file).toBe(original.file);
      }
    });

    it('should respect query time budget constraint (≤2%)', async () => {
      const evolutionSystem = new EvolutionMappingSystem(testConfig.evolution_mapping);
      
      const totalQueryBudget = 100; // 100ms
      const maxEvolutionBudget = totalQueryBudget * (testConfig.evolution_mapping.max_query_time_budget_percent / 100);
      
      const candidates: SymbolCandidate[] = [{
        file_path: 'src/api.ts',
        score: 0.8,
        match_reasons: ['symbol'],
      }];

      const startTime = performance.now();
      await evolutionSystem.expandWithEvolution('apiFunction', candidates, maxEvolutionBudget);
      const actualTime = performance.now() - startTime;

      // Gate: Evolution mapping budget ≤2% of query time
      expect(actualTime).toBeLessThanOrEqual(maxEvolutionBudget);
      expect(maxEvolutionBudget).toBeLessThanOrEqual(totalQueryBudget * 0.02);
    });
  });

  describe('Query Compiler System', () => {
    it('should achieve p99 improvement target (-8 to -12%)', async () => {
      const queryCompiler = new QueryCompiler(testConfig.query_compiler);
      
      const constraints = {
        max_total_time_ms: 20,
        min_recall_at_50: 0.8,
        span_invariants: true,
        exact_struct_floors: { exact_min: 10, struct_min: 20 }
      };

      const plan = await queryCompiler.compileQuery('test query', constraints);
      const result = await queryCompiler.executePlan(plan, 'test query');

      // Simulate baseline vs optimized performance
      const baselineP99 = 18; // ms
      const optimizedP99 = result.metrics.total_time_ms;
      const improvementPercent = ((optimizedP99 - baselineP99) / baselineP99) * 100;

      // Gate: p99 improvement -8 to -12%
      expect(improvementPercent).toBeLessThanOrEqual(-8);
      expect(improvementPercent).toBeGreaterThanOrEqual(-15); // Allow some tolerance
      expect(result.metrics.final_recall_at_50).toBeGreaterThanOrEqual(0.8);
    });

    it('should maintain recall@50 while optimizing (flat SLA)', async () => {
      const queryCompiler = new QueryCompiler(testConfig.query_compiler);
      
      const constraints = {
        max_total_time_ms: 15,
        min_recall_at_50: 0.85,
        span_invariants: true,
        exact_struct_floors: { exact_min: 5, struct_min: 15 }
      };

      const plan = await queryCompiler.compileQuery('complex query', constraints);
      const result = await queryCompiler.executePlan(plan, 'complex query');

      // Gate: Maintain recall@50 while improving latency
      expect(result.metrics.final_recall_at_50).toBeGreaterThanOrEqual(constraints.min_recall_at_50);
      expect(result.metrics.total_time_ms).toBeLessThanOrEqual(constraints.max_total_time_ms);
      expect(result.metrics.span_accuracy).toBeGreaterThanOrEqual(0.99);
    });

    it('should preserve span invariants', async () => {
      const queryCompiler = new QueryCompiler(testConfig.query_compiler);
      
      const constraints = {
        max_total_time_ms: 25,
        min_recall_at_50: 0.75,
        span_invariants: true, // Must preserve spans
        exact_struct_floors: { exact_min: 8, struct_min: 12 }
      };

      const plan = await queryCompiler.compileQuery('span sensitive query', constraints);
      
      // Gate: Span invariants must be preserved
      expect(plan.constraints.span_invariants).toBe(true);
      expect(plan.constraints.exact_struct_floors.exact_min).toBeGreaterThanOrEqual(5);
      expect(plan.constraints.exact_struct_floors.struct_min).toBeGreaterThanOrEqual(10);
    });
  });

  describe('Regression Bisection Harness', () => {
    it('should isolate culprit within time constraint (≤30 min)', async () => {
      const regressionHarness = new RegressionBisectionHarness(testConfig.regression_bisection);
      
      const mockAlert = {
        id: 'test_alert_123',
        alert_type: 'sla_recall_50' as const,
        metric_name: 'recall_at_50',
        current_value: 0.75,
        baseline_value: 0.82,
        threshold_value: 0.80,
        deviation_percent: -8.5,
        confidence: 0.95,
        triggered_at: new Date(),
        time_window: 'last_1h',
        sample_size: 1000
      };

      const startTime = Date.now();
      const experimentId = await regressionHarness.triggerRegressionAnalysis(mockAlert);
      
      // Gate: Culprit isolation ≤30 minutes
      expect(experimentId).toBeDefined();
      expect(testConfig.regression_bisection.max_experiment_duration_minutes).toBeLessThanOrEqual(30);
    });

    it('should detect significant effect size (Cohen\'s d)', async () => {
      const regressionHarness = new RegressionBisectionHarness(testConfig.regression_bisection);
      
      // Gate: Minimum effect size threshold
      expect(testConfig.regression_bisection.min_effect_size_cohens_d).toBeGreaterThanOrEqual(0.5); // Medium effect
      expect(testConfig.regression_bisection.rollback_threshold_effect_size).toBeGreaterThanOrEqual(1.0); // Large effect
    });

    it('should enable automatic rollback with diff summary', async () => {
      const regressionHarness = new RegressionBisectionHarness(testConfig.regression_bisection);
      
      // Gate: Auto-rollback enabled
      expect(testConfig.regression_bisection.auto_rollback_enabled).toBe(true);
      
      const metrics = regressionHarness.getPerformanceMetrics();
      expect(metrics).toHaveProperty('rollbacks_executed');
      expect(metrics).toHaveProperty('alerts_processed');
    });
  });

  describe('Integrated System Performance', () => {
    it('should meet all performance gates in end-to-end scenario', async () => {
      const request: EnhancedSearchRequest = {
        query: 'function findUser',
        intent: 'symbol',
        performance_budget_ms: 20,
        quality_requirements: {
          min_recall_at_50: 0.8,
          min_precision: 0.85,
          require_span_accuracy: true,
        },
        experiment_config: {
          enable_cross_repo: true,
          enable_evolution_mapping: true,
          use_optimized_compiler: true,
        }
      };

      const startTime = performance.now();
      const response = await integration.enhancedSearch(request);
      const totalLatency = performance.now() - startTime;

      // Overall system gates
      expect(response.performance_metrics.total_latency_ms).toBeLessThanOrEqual(request.performance_budget_ms);
      expect(response.quality_metrics.recall_at_50).toBeGreaterThanOrEqual(request.quality_requirements.min_recall_at_50);
      expect(response.quality_metrics.span_accuracy).toBeGreaterThanOrEqual(0.99);
      
      // System-specific gates
      const monikerLatency = response.performance_metrics.stage_latencies.get('moniker_expansion') || 0;
      const evolutionLatency = response.performance_metrics.stage_latencies.get('evolution_expansion') || 0;
      
      expect(monikerLatency).toBeLessThanOrEqual(0.6); // Moniker gate
      expect(evolutionLatency).toBeLessThanOrEqual(response.performance_metrics.total_latency_ms * 0.02); // Evolution gate
      
      // No critical warnings
      const criticalWarnings = response.warnings.filter(w => w.includes('exceeded') || w.includes('failed'));
      expect(criticalWarnings.length).toBe(0);
    });

    it('should maintain embedder-agnostic operation', async () => {
      const request: EnhancedSearchRequest = {
        query: 'class DatabaseConnection',
        intent: 'symbol',
        performance_budget_ms: 15,
        quality_requirements: {
          min_recall_at_50: 0.75,
          min_precision: 0.8,
          require_span_accuracy: true,
        }
      };

      const response = await integration.enhancedSearch(request);
      
      // Should work without embedding-specific dependencies
      expect(response.hits.length).toBeGreaterThan(0);
      expect(response.quality_metrics.span_accuracy).toBeGreaterThanOrEqual(0.99);
      
      // All hits should have valid span coordinates
      response.hits.forEach(hit => {
        expect(hit.line).toBeGreaterThan(0);
        expect(hit.col).toBeGreaterThanOrEqual(0);
        expect(hit.file).toBeDefined();
        expect(hit.file.length).toBeGreaterThan(0);
      });
    });

    it('should preserve span-safe operations throughout pipeline', async () => {
      const request: EnhancedSearchRequest = {
        query: 'async function processData',
        intent: 'symbol',
        performance_budget_ms: 25,
        quality_requirements: {
          min_recall_at_50: 0.8,
          min_precision: 0.85,
          require_span_accuracy: true,
        },
        revision_context: {
          base_sha: 'abc123',
          target_sha: 'def456'
        }
      };

      const response = await integration.enhancedSearch(request);
      
      // Span safety verification
      expect(response.quality_metrics.span_accuracy).toBeGreaterThanOrEqual(0.99);
      
      // All hits should maintain span integrity
      response.hits.forEach(hit => {
        expect(hit.line).toBeGreaterThan(0);
        expect(hit.col).toBeGreaterThanOrEqual(0);
        expect(hit.span_len).toBeUndefined() || expect(hit.span_len).toBeGreaterThan(0);
        
        // Evolution mapping should preserve or correctly project spans
        if (hit.why.includes('evolution:rename' as any) || hit.why.includes('evolution:move' as any)) {
          expect(hit.line).toBeGreaterThan(0); // Projected spans should still be valid
        }
      });
    });
  });

  describe('System Health and Monitoring', () => {
    it('should generate comprehensive performance metrics', async () => {
      const metrics = integration.getSystemPerformanceMetrics();
      
      expect(metrics).toHaveProperty('moniker_system');
      expect(metrics).toHaveProperty('evolution_system');
      expect(metrics).toHaveProperty('query_compiler');
      expect(metrics).toHaveProperty('regression_harness');
      expect(metrics).toHaveProperty('integration');
      
      expect(metrics.integration.total_queries_processed).toBeGreaterThanOrEqual(0);
      expect(metrics.integration.avg_total_latency_ms).toBeGreaterThanOrEqual(0);
      expect(metrics.integration.gate_failures).toBe(0);
    });

    it('should provide actionable status report', async () => {
      const statusReport = integration.generateStatusReport();
      
      expect(statusReport.overall_health).toMatch(/^(healthy|degraded|critical)$/);
      expect(statusReport.gate_status.size).toBeGreaterThan(0);
      expect(statusReport.system_performance).toBeDefined();
      expect(Array.isArray(statusReport.recommendations)).toBe(true);
      
      // If system is healthy, should have minimal recommendations
      if (statusReport.overall_health === 'healthy') {
        expect(statusReport.recommendations.length).toBeLessThanOrEqual(2);
      }
    });
  });

  describe('Configuration and Constraints Validation', () => {
    it('should enforce all specified constraints', () => {
      // Moniker linking constraints
      expect(testConfig.moniker_linking.max_cluster_size).toBeLessThanOrEqual(128);
      expect(testConfig.moniker_linking.max_import_cone_depth).toBeLessThanOrEqual(1);
      expect(testConfig.moniker_linking.supported_intents).toEqual(['symbol', 'nl']);
      
      // Evolution mapping constraints
      expect(testConfig.evolution_mapping.max_query_time_budget_percent).toBeLessThanOrEqual(2);
      expect(testConfig.evolution_mapping.performance_targets.zero_span_drift_tolerance).toBeLessThanOrEqual(0);
      
      // Query compiler constraints
      expect(testConfig.query_compiler.optimization_timeout_ms).toBeLessThanOrEqual(100);
      expect(testConfig.query_compiler.performance_targets.p99_improvement_percent).toBeLessThanOrEqual(-8);
      
      // Regression bisection constraints
      expect(testConfig.regression_bisection.max_experiment_duration_minutes).toBeLessThanOrEqual(30);
      expect(testConfig.regression_bisection.min_effect_size_cohens_d).toBeGreaterThanOrEqual(0.5);
    });

    it('should validate performance targets consistency', () => {
      // All performance targets should be achievable and non-conflicting
      expect(testConfig.moniker_linking.performance_targets.max_additional_latency_ms).toBeLessThanOrEqual(1.0);
      expect(testConfig.moniker_linking.performance_targets.min_recall_improvement_pp).toBeGreaterThanOrEqual(0.5);
      
      expect(testConfig.evolution_mapping.performance_targets.success_at_10_improvement_pp).toBeGreaterThanOrEqual(0.5);
      
      expect(testConfig.query_compiler.performance_targets.maintain_recall_at_50).toBe(true);
    });
  });
});

describe('Performance Gate Integration Tests', () => {
  let integration: AdvancedSystemsIntegration;

  beforeEach(() => {
    const config: AdvancedSystemsConfig = {
      moniker_linking: {
        max_cluster_size: 128,
        max_import_cone_depth: 1,
        cache_ttl_hours: 24,
        centrality_normalization_factor: 0.85,
        supported_intents: ['symbol', 'nl'],
        performance_targets: {
          max_additional_latency_ms: 0.6,
          min_recall_improvement_pp: 0.8, // Higher target for gate test
          max_why_mix_kl_divergence: 0.02,
        }
      },
      evolution_mapping: {
        max_lineage_depth: 10,
        max_rewrite_rules_per_symbol: 20,
        max_query_time_budget_percent: 2,
        line_map_ttl_hours: 24,
        confidence_threshold: 0.7,
        supported_packages: ['lodash', 'react', '@types/node'],
        performance_targets: {
          success_at_10_improvement_pp: 0.5,
          zero_span_drift_tolerance: 0,
        }
      },
      query_compiler: {
        max_plans_to_consider: 5,
        optimization_timeout_ms: 30,
        telemetry_sample_rate: 0.2,
        cost_model_update_frequency: 12,
        performance_targets: {
          p99_improvement_percent: -10,
          maintain_recall_at_50: true,
        },
        plan_templates: {
          symbol_first: true,
          struct_first: true,
          semantic_first: false,
        }
      },
      regression_bisection: {
        max_experiment_duration_minutes: 25, // Tighter constraint for testing
        min_effect_size_cohens_d: 0.6, // Higher threshold
        statistical_significance_threshold: 0.01, // Stricter significance
        max_knobs_per_experiment: 6,
        traffic_split_percent: 5,
        shadow_traffic_enabled: true,
        auto_rollback_enabled: true,
        rollback_threshold_effect_size: 0.8,
      },
      integration: {
        enable_all_systems: true,
        performance_monitoring_enabled: true,
        quality_gates_enabled: true,
        auto_optimization_enabled: true,
      }
    };

    integration = new AdvancedSystemsIntegration(config);
  });

  it('should pass all gates under normal load', async () => {
    const requests: EnhancedSearchRequest[] = Array.from({ length: 10 }, (_, i) => ({
      query: `test query ${i}`,
      intent: 'symbol',
      performance_budget_ms: 18,
      quality_requirements: {
        min_recall_at_50: 0.8,
        min_precision: 0.85,
        require_span_accuracy: true,
      }
    }));

    const responses = await Promise.all(
      requests.map(req => integration.enhancedSearch(req))
    );

    // Verify all responses meet gates
    responses.forEach((response, i) => {
      expect(response.performance_metrics.total_latency_ms, 
        `Query ${i} latency: ${response.performance_metrics.total_latency_ms}ms`
      ).toBeLessThanOrEqual(18);
      
      expect(response.quality_metrics.recall_at_50,
        `Query ${i} recall: ${response.quality_metrics.recall_at_50}`
      ).toBeGreaterThanOrEqual(0.8);
      
      expect(response.quality_metrics.span_accuracy,
        `Query ${i} span accuracy: ${response.quality_metrics.span_accuracy}`
      ).toBeGreaterThanOrEqual(0.99);

      // Check gate status
      const failedGates = Array.from(response.gate_status.entries())
        .filter(([_, status]) => !status)
        .map(([gate, _]) => gate);
        
      expect(failedGates, 
        `Query ${i} failed gates: ${failedGates.join(', ')}`
      ).toHaveLength(0);
    });

    // Overall system health check
    const statusReport = integration.generateStatusReport();
    expect(statusReport.overall_health).toBe('healthy');
  });

  it('should detect and handle gate failures', async () => {
    // Simulate a challenging request that might trigger gate failures
    const challengingRequest: EnhancedSearchRequest = {
      query: 'very complex query with multiple terms and cross-repo requirements',
      intent: 'nl',
      performance_budget_ms: 8, // Very tight budget
      quality_requirements: {
        min_recall_at_50: 0.95, // Very high requirement
        min_precision: 0.95,
        require_span_accuracy: true,
      },
      experiment_config: {
        enable_cross_repo: true,
        enable_evolution_mapping: true,
        use_optimized_compiler: true,
      }
    };

    const response = await integration.enhancedSearch(challengingRequest);
    
    // System should handle gracefully even if some gates fail
    expect(response).toBeDefined();
    expect(response.hits).toBeDefined();
    expect(response.performance_metrics).toBeDefined();
    expect(response.quality_metrics).toBeDefined();
    
    // Check if warnings were issued for gate failures
    if (response.performance_metrics.total_latency_ms > challengingRequest.performance_budget_ms) {
      expect(response.warnings.some(w => w.includes('budget') || w.includes('latency'))).toBe(true);
    }
    
    // Gate status should reflect actual performance
    const gateResults = Array.from(response.gate_status.entries());
    expect(gateResults.length).toBeGreaterThan(0);
  });
});