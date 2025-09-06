/**
 * Industrial Systems Integration Tests
 * 
 * Comprehensive test suite for all four industrial strength systems:
 * - Ground-Truth Engine
 * - Economics/SLO Controller  
 * - Counterfactual "Why" Tooling
 * - Multi-Tenant Boundaries & Scale Safety
 */

import { describe, test, expect, beforeEach, afterEach, jest } from '@jest/globals';
import { IndustrialStrengthLensSystem, DEFAULT_INDUSTRIAL_CONFIG } from '../industrial-strength-integration.js';

// Mock external dependencies
jest.mock('../ground-truth-engine.js');
jest.mock('../economics-slo-controller.js');
jest.mock('../counterfactual-why-tooling.js');
jest.mock('../multi-tenant-boundaries.js');

describe('Industrial Systems Integration', () => {
  let system: IndustrialStrengthLensSystem;

  beforeEach(async () => {
    system = new IndustrialStrengthLensSystem(DEFAULT_INDUSTRIAL_CONFIG);
  });

  afterEach(async () => {
    await system.shutdown();
  });

  describe('System Initialization', () => {
    test('should initialize all four core systems', async () => {
      await system.initialize();
      
      const health = await system.getSystemHealth();
      expect(health.overall_status).toBe('healthy');
      expect(health.ground_truth_engine.status).toBe('healthy');
      expect(health.economics_controller.status).toBe('healthy');
      expect(health.counterfactual_tooling.status).toBe('healthy');
      expect(health.multi_tenant_boundaries.status).toBe('healthy');
    });

    test('should emit initialization events', async () => {
      const initStartedSpy = jest.fn();
      const initCompletedSpy = jest.fn();
      
      system.on('initialization_started', initStartedSpy);
      system.on('initialization_completed', initCompletedSpy);
      
      await system.initialize();
      
      expect(initStartedSpy).toHaveBeenCalled();
      expect(initCompletedSpy).toHaveBeenCalled();
      expect(initCompletedSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          systems_initialized: 4,
          health_status: expect.any(Object)
        })
      );
    });

    test('should handle initialization failures gracefully', async () => {
      // Mock a system initialization failure
      const mockError = new Error('Mock initialization failure');
      jest.spyOn(system as any, 'healthOrchestrator').mockImplementation({
        start: jest.fn().mockRejectedValue(mockError)
      });

      const initFailedSpy = jest.fn();
      system.on('initialization_failed', initFailedSpy);

      await expect(system.initialize()).rejects.toThrow('Mock initialization failure');
      expect(initFailedSpy).toHaveBeenCalledWith({ error: 'Mock initialization failure' });
    });
  });

  describe('Unified Query Processing', () => {
    beforeEach(async () => {
      await system.initialize();
    });

    test('should process query through all systems with full integration', async () => {
      const unifiedQuery = {
        query: 'find authentication function',
        tenant_id: 'test-tenant',
        user_id: 'test-user',
        context: {
          ground_truth_eligible: true,
          economics_constraints: {
            max_latency_ms: 50,
            max_cost_units: 10,
            quality_sacrifice_allowed: false,
            headroom_priority: 'medium' as const
          },
          multi_tenant_isolation: {
            isolation_level: 'hard' as const,
            privacy_constraints: ['no-cross-tenant-expansion'],
            resource_quotas: { queries_per_minute: 100 },
            cross_shard_credits_available: 1000
          },
          counterfactual_tracking: true
        },
        debug_mode: true
      };

      const result = await system.executeQuery(unifiedQuery);

      // Verify result structure
      expect(result).toMatchObject({
        results: expect.any(Array),
        metadata: {
          execution_time_ms: expect.any(Number),
          processing_mode: expect.any(String),
          quality_score: expect.any(Number),
          cost_units_consumed: expect.any(Number),
          span_coverage: expect.any(Number),
          degradation_applied: expect.any(Boolean)
        }
      });

      // Verify cross-system enrichments
      expect(result.ground_truth_feedback).toBeDefined();
      expect(result.economics_metrics).toBeDefined();
      expect(result.counterfactual_links).toBeDefined();
      expect(result.tenant_usage).toBeDefined();

      // Verify ground truth feedback
      expect(result.ground_truth_feedback).toMatchObject({
        eligible_for_annotation: expect.any(Boolean),
        gap_score: expect.any(Number),
        exploration_bonus: expect.any(Number),
        sampling_probability: expect.any(Number)
      });

      // Verify economics metrics
      expect(result.economics_metrics).toMatchObject({
        utility_score: expect.any(Number),
        lambda_ms_applied: expect.any(Number),
        lambda_gb_applied: expect.any(Number),
        headroom_trade_applied: expect.any(Boolean),
        bandit_arm_selected: expect.any(String)
      });

      // Verify counterfactual links
      expect(result.counterfactual_links).toMatchObject({
        debug_session_url: expect.any(String),
        floor_wins_detected: expect.any(Number),
        why_explanation_available: expect.any(Boolean)
      });

      // Verify tenant usage tracking
      expect(result.tenant_usage).toMatchObject({
        quota_utilization: expect.any(Object),
        violation_risk_score: expect.any(Number),
        credits_remaining: expect.any(Number),
        disaster_mode_eligible: expect.any(Boolean)
      });
    });

    test('should handle tenant access denial', async () => {
      const unifiedQuery = {
        query: 'test query',
        tenant_id: 'non-existent-tenant',
        context: {
          ground_truth_eligible: false,
          economics_constraints: {
            max_latency_ms: 50,
            max_cost_units: 10,
            quality_sacrifice_allowed: false,
            headroom_priority: 'low' as const
          },
          multi_tenant_isolation: {
            isolation_level: 'soft' as const,
            privacy_constraints: [],
            resource_quotas: {},
            cross_shard_credits_available: 0
          },
          counterfactual_tracking: false
        }
      };

      await expect(system.executeQuery(unifiedQuery))
        .rejects.toThrow(/Tenant.*not found/);
    });

    test('should emit query execution events', async () => {
      const queryExecutedSpy = jest.fn();
      system.on('query_executed', queryExecutedSpy);

      const unifiedQuery = {
        query: 'test query',
        tenant_id: 'system',
        context: {
          ground_truth_eligible: false,
          economics_constraints: {
            max_latency_ms: 50,
            max_cost_units: 10,
            quality_sacrifice_allowed: false,
            headroom_priority: 'low' as const
          },
          multi_tenant_isolation: {
            isolation_level: 'soft' as const,
            privacy_constraints: [],
            resource_quotas: {},
            cross_shard_credits_available: 100
          },
          counterfactual_tracking: false
        }
      };

      await system.executeQuery(unifiedQuery);

      expect(queryExecutedSpy).toHaveBeenCalledWith({
        tenant_id: 'system',
        execution_time_ms: expect.any(Number),
        result_metadata: expect.any(Object)
      });
    });
  });

  describe('Cross-System Coordination', () => {
    beforeEach(async () => {
      await system.initialize();
    });

    test('should coordinate between ground truth engine and economics controller', async () => {
      // Mock a quality degradation event from economics controller
      const economicsController = (system as any).economicsController;
      
      const groundTruthSpy = jest.fn();
      (system as any).groundTruthEngine.on('quality_degradation_detected', groundTruthSpy);

      // Simulate economics optimization with quality degradation
      economicsController.emit('optimization_completed', {
        utility: { delta_ndcg_at_10: -0.1 }, // Significant quality degradation
        selected_arm: { id: 'test-arm' }
      });

      expect(groundTruthSpy).toHaveBeenCalled();
    });

    test('should coordinate between multi-tenant system and counterfactual tooling', async () => {
      const multiTenantSystem = (system as any).multiTenantSystem;
      const counterfactualTooling = (system as any).counterfactualTooling;
      
      const floorWinSpikeSpy = jest.fn();
      counterfactualTooling.on('floor_win_spike', floorWinSpikeSpy);

      // Simulate high override rate event
      multiTenantSystem.emit('high_override_rate', {
        rate: 0.4, // 40% override rate
        threshold: 0.3
      });

      expect(floorWinSpikeSpy).toHaveBeenCalled();
    });

    test('should coordinate disaster mode across all systems', async () => {
      const economicsController = (system as any).economicsController;
      const groundTruthEngine = (system as any).groundTruthEngine;
      const counterfactualTooling = (system as any).counterfactualTooling;
      
      const ecoDisasterSpy = jest.fn();
      const gtDisasterSpy = jest.fn();
      const cfDisasterSpy = jest.fn();

      economicsController.on('disaster_mode_active', ecoDisasterSpy);
      groundTruthEngine.on('disaster_mode_active', gtDisasterSpy);
      counterfactualTooling.on('disaster_mode_active', cfDisasterSpy);

      // Simulate disaster mode activation
      const multiTenantSystem = (system as any).multiTenantSystem;
      multiTenantSystem.emit('disaster_mode_activated', {
        mode: { name: 'cache_first' },
        reason: 'Test disaster mode'
      });

      expect(ecoDisasterSpy).toHaveBeenCalled();
      expect(gtDisasterSpy).toHaveBeenCalled();
      expect(cfDisasterSpy).toHaveBeenCalled();
    });
  });

  describe('Health Monitoring', () => {
    beforeEach(async () => {
      await system.initialize();
    });

    test('should provide comprehensive health status', async () => {
      const health = await system.getSystemHealth();

      expect(health).toMatchObject({
        ground_truth_engine: {
          status: expect.stringMatching(/^(healthy|degraded|critical|down)$/),
          metrics: {
            availability_pct: expect.any(Number),
            performance_score: expect.any(Number),
            error_rate: expect.any(Number),
            resource_utilization: expect.any(Number)
          },
          alerts: expect.any(Array),
          dependencies: expect.any(Array),
          last_health_check: expect.any(Date)
        },
        economics_controller: expect.any(Object),
        counterfactual_tooling: expect.any(Object),
        multi_tenant_boundaries: expect.any(Object),
        integration_layer: expect.any(Object),
        overall_status: expect.stringMatching(/^(healthy|degraded|critical|down)$/),
        last_updated: expect.any(Date)
      });
    });

    test('should track system dependencies correctly', async () => {
      const health = await system.getSystemHealth();

      // Ground truth engine depends on multi-tenant boundaries
      expect(health.ground_truth_engine.dependencies).toContain('multi_tenant_boundaries');
      
      // Economics controller depends on multi-tenant boundaries
      expect(health.economics_controller.dependencies).toContain('multi_tenant_boundaries');
      
      // Integration layer depends on all core systems
      expect(health.integration_layer.dependencies).toEqual(
        expect.arrayContaining([
          'ground_truth_engine',
          'economics_controller',
          'counterfactual_tooling',
          'multi_tenant_boundaries'
        ])
      );
    });
  });

  describe('Governance Reporting', () => {
    beforeEach(async () => {
      await system.initialize();
    });

    test('should generate comprehensive governance report', async () => {
      const report = await system.generateGovernanceReport(24);

      expect(report).toMatchObject({
        period_start: expect.any(Date),
        period_end: expect.any(Date),
        ground_truth_pool_health: {
          total_queries_added: expect.any(Number),
          inter_rater_kappa_avg: expect.any(Number),
          slice_coverage_pct: expect.any(Number),
          quality_trend: expect.stringMatching(/^(improving|stable|degrading)$/)
        },
        economics_utility_trend: {
          avg_utility_score: expect.any(Number),
          cost_optimization_pct: expect.any(Number),
          quality_trade_effectiveness: expect.any(Number),
          bandit_exploration_efficiency: expect.any(Number)
        },
        counterfactual_insights: {
          debug_sessions_created: expect.any(Number),
          policy_recommendations_generated: expect.any(Number),
          rollout_simulations_run: expect.any(Number),
          operator_actions_prevented: expect.any(Number)
        },
        tenant_compliance: {
          total_tenants: expect.any(Number),
          quota_violation_rate: expect.any(Number),
          privacy_breach_incidents: expect.any(Number),
          disaster_mode_activations: expect.any(Number),
          sla_compliance_pct: expect.any(Number)
        },
        system_recommendations: expect.arrayContaining([
          expect.objectContaining({
            type: expect.stringMatching(/^(ground_truth|economics|counterfactual|multi_tenant|integration)$/),
            priority: expect.stringMatching(/^(low|medium|high|critical)$/),
            title: expect.any(String),
            description: expect.any(String),
            impact_assessment: expect.any(String),
            implementation_steps: expect.any(Array),
            estimated_effort_hours: expect.any(Number)
          })
        ])
      });
    });

    test('should emit governance report events', async () => {
      const reportGeneratedSpy = jest.fn();
      system.on('governance_report_generated', reportGeneratedSpy);

      const report = await system.generateGovernanceReport();

      expect(reportGeneratedSpy).toHaveBeenCalledWith({ report });
    });
  });

  describe('Emergency Procedures', () => {
    beforeEach(async () => {
      await system.initialize();
    });

    test('should execute disaster mode activation', async () => {
      const emergencyStartedSpy = jest.fn();
      const emergencyCompletedSpy = jest.fn();

      system.on('emergency_procedure_started', emergencyStartedSpy);
      system.on('emergency_procedure_completed', emergencyCompletedSpy);

      await system.executeEmergencyProcedure('activate_disaster_mode', 'high');

      expect(emergencyStartedSpy).toHaveBeenCalledWith({
        procedure: 'activate_disaster_mode',
        severity: 'high'
      });

      expect(emergencyCompletedSpy).toHaveBeenCalledWith({
        procedure: 'activate_disaster_mode',
        severity: 'high'
      });
    });

    test('should execute kill sequence', async () => {
      const emergencyStartedSpy = jest.fn();
      system.on('emergency_procedure_started', emergencyStartedSpy);

      await system.executeEmergencyProcedure('execute_kill_sequence', 'critical');

      expect(emergencyStartedSpy).toHaveBeenCalledWith({
        procedure: 'execute_kill_sequence',
        severity: 'critical'
      });
    });

    test('should handle unknown emergency procedures', async () => {
      await expect(system.executeEmergencyProcedure('unknown_procedure', 'medium'))
        .rejects.toThrow('Unknown emergency procedure: unknown_procedure');
    });

    test('should emit failure events for failed procedures', async () => {
      const emergencyFailedSpy = jest.fn();
      system.on('emergency_procedure_failed', emergencyFailedSpy);

      // Mock a failure in the multi-tenant system
      jest.spyOn((system as any).multiTenantSystem, 'activateDisasterMode')
        .mockRejectedValue(new Error('Mock disaster mode failure'));

      await expect(system.executeEmergencyProcedure('activate_disaster_mode', 'high'))
        .rejects.toThrow('Mock disaster mode failure');

      expect(emergencyFailedSpy).toHaveBeenCalledWith({
        procedure: 'activate_disaster_mode',
        severity: 'high',
        error: 'Mock disaster mode failure'
      });
    });
  });

  describe('System Shutdown', () => {
    beforeEach(async () => {
      await system.initialize();
    });

    test('should shutdown gracefully', async () => {
      const shutdownStartedSpy = jest.fn();
      const shutdownCompletedSpy = jest.fn();

      system.on('shutdown_started', shutdownStartedSpy);
      system.on('shutdown_completed', shutdownCompletedSpy);

      await system.shutdown();

      expect(shutdownStartedSpy).toHaveBeenCalled();
      expect(shutdownCompletedSpy).toHaveBeenCalled();
    });

    test('should handle shutdown failures', async () => {
      const shutdownFailedSpy = jest.fn();
      system.on('shutdown_failed', shutdownFailedSpy);

      // Mock a shutdown failure
      jest.spyOn((system as any).healthOrchestrator, 'stop')
        .mockRejectedValue(new Error('Mock shutdown failure'));

      await expect(system.shutdown()).rejects.toThrow('Mock shutdown failure');

      expect(shutdownFailedSpy).toHaveBeenCalledWith({
        error: 'Mock shutdown failure'
      });
    });
  });

  describe('Configuration Validation', () => {
    test('should use default configuration when none provided', () => {
      const defaultSystem = new IndustrialStrengthLensSystem();
      expect((defaultSystem as any).config).toEqual(DEFAULT_INDUSTRIAL_CONFIG);
    });

    test('should merge partial configurations with defaults', () => {
      const partialConfig = {
        integration: {
          cross_system_coordination: false
        }
      };

      const system = new IndustrialStrengthLensSystem(partialConfig);
      const config = (system as any).config;

      expect(config.integration.cross_system_coordination).toBe(false);
      expect(config.integration.unified_monitoring).toBe(true); // Should keep default
      expect(config.ground_truth).toEqual(DEFAULT_INDUSTRIAL_CONFIG.ground_truth);
    });
  });

  describe('Performance and Load Testing', () => {
    beforeEach(async () => {
      await system.initialize();
    });

    test('should handle concurrent query processing', async () => {
      const queries = Array.from({ length: 10 }, (_, i) => ({
        query: `test query ${i}`,
        tenant_id: 'system',
        context: {
          ground_truth_eligible: false,
          economics_constraints: {
            max_latency_ms: 100,
            max_cost_units: 5,
            quality_sacrifice_allowed: true,
            headroom_priority: 'low' as const
          },
          multi_tenant_isolation: {
            isolation_level: 'soft' as const,
            privacy_constraints: [],
            resource_quotas: {},
            cross_shard_credits_available: 1000
          },
          counterfactual_tracking: false
        }
      }));

      const results = await Promise.all(queries.map(query => system.executeQuery(query)));

      expect(results).toHaveLength(10);
      results.forEach(result => {
        expect(result).toMatchObject({
          results: expect.any(Array),
          metadata: expect.any(Object)
        });
      });
    });

    test('should maintain performance under load', async () => {
      const startTime = Date.now();
      
      const promises = Array.from({ length: 50 }, () => 
        system.executeQuery({
          query: 'performance test query',
          tenant_id: 'system',
          context: {
            ground_truth_eligible: false,
            economics_constraints: {
              max_latency_ms: 50,
              max_cost_units: 3,
              quality_sacrifice_allowed: true,
              headroom_priority: 'low' as const
            },
            multi_tenant_isolation: {
              isolation_level: 'soft' as const,
              privacy_constraints: [],
              resource_quotas: {},
              cross_shard_credits_available: 500
            },
            counterfactual_tracking: false
          }
        })
      );

      await Promise.all(promises);
      
      const totalTime = Date.now() - startTime;
      const avgTimePerQuery = totalTime / promises.length;

      // Should process queries reasonably quickly even under load
      expect(avgTimePerQuery).toBeLessThan(100); // Less than 100ms per query on average
    });
  });

  describe('Error Handling and Resilience', () => {
    beforeEach(async () => {
      await system.initialize();
    });

    test('should handle system component failures gracefully', async () => {
      // Mock a failure in one of the core systems
      jest.spyOn((system as any).economicsController, 'classifyAndRoute')
        .mockRejectedValue(new Error('Economics controller failure'));

      const queryFailedSpy = jest.fn();
      system.on('query_failed', queryFailedSpy);

      const query = {
        query: 'test query',
        tenant_id: 'system',
        context: {
          ground_truth_eligible: false,
          economics_constraints: {
            max_latency_ms: 50,
            max_cost_units: 5,
            quality_sacrifice_allowed: false,
            headroom_priority: 'medium' as const
          },
          multi_tenant_isolation: {
            isolation_level: 'hard' as const,
            privacy_constraints: [],
            resource_quotas: {},
            cross_shard_credits_available: 100
          },
          counterfactual_tracking: false
        }
      };

      await expect(system.executeQuery(query))
        .rejects.toThrow('Economics controller failure');

      expect(queryFailedSpy).toHaveBeenCalledWith({
        tenant_id: 'system',
        query: 'test query',
        error: 'Economics controller failure',
        execution_time_ms: expect.any(Number)
      });
    });
  });
});

describe('Individual System Unit Tests', () => {
  // Additional tests for individual systems would go here
  // These would test each system in isolation with proper mocking
  
  describe('Ground Truth Engine', () => {
    test('should be tested separately');
  });

  describe('Economics/SLO Controller', () => {
    test('should be tested separately');
  });

  describe('Counterfactual Why Tooling', () => {
    test('should be tested separately');
  });

  describe('Multi-Tenant Boundaries', () => {
    test('should be tested separately');
  });
});