/**
 * Comprehensive test suite for Chaos Engineering Framework
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { 
  ChaosEngineeringFramework,
  ChaosExperimentType,
  ChaosExperimentState,
  ChaosExperimentConfig,
  ErrorType,
  ErrorSeverity
} from '../chaos-engineering-framework.js';
import { ChaosExperimentSuite } from '../chaos-scenarios.js';

// Mock dependencies
vi.mock('../performance-monitor.js', () => ({
  performanceMonitor: {
    recordMetric: vi.fn(),
  }
}));

vi.mock('../resilience-manager.js', () => ({
  resilienceManager: {
    getMetrics: vi.fn().mockReturnValue({
      circuitBreakers: [],
      bulkheads: [],
      rateLimiters: [],
      errors: [],
      fallbackCacheSize: 0
    })
  }
}));

vi.mock('../telemetry/tracer.js', () => ({
  LensTracer: {
    createTracer: vi.fn().mockReturnValue({
      startActiveSpan: vi.fn().mockImplementation(async (name, fn) => {
        const span = {
          setAttributes: vi.fn(),
          addEvent: vi.fn(),
          recordException: vi.fn(),
          end: vi.fn()
        };
        return await fn(span);
      })
    })
  }
}));

describe('ChaosEngineeringFramework', () => {
  let chaosFramework: ChaosEngineeringFramework;
  
  beforeEach(() => {
    // Reset singleton instance
    (ChaosEngineeringFramework as any).instance = null;
    
    chaosFramework = ChaosEngineeringFramework.getInstance({
      productionMode: false,
      safetyLimits: {
        maxConcurrentExperiments: 2,
        maxExperimentDuration: 60000,
        emergencyStopThreshold: {
          errorRate: 0.1,
          latencyMultiplier: 3.0
        }
      },
      baselineService: {
        url: 'http://localhost:3001',
        healthEndpoint: '/health',
        metricsEndpoint: '/metrics'
      }
    });
  });
  
  afterEach(async () => {
    await chaosFramework.shutdown();
  });
  
  describe('Initialization', () => {
    it('should create singleton instance', () => {
      const instance1 = ChaosEngineeringFramework.getInstance();
      const instance2 = ChaosEngineeringFramework.getInstance();
      
      expect(instance1).toBe(instance2);
    });
    
    it('should initialize with default config', () => {
      const defaultFramework = ChaosEngineeringFramework.getInstance();
      const status = defaultFramework.getExperimentStatus();
      
      expect(status.registered).toBe(0);
      expect(status.active).toBe(0);
      expect(status.experiments).toEqual([]);
    });
  });
  
  describe('Experiment Registration', () => {
    it('should register experiment successfully', () => {
      const experiment: ChaosExperimentConfig = {
        id: 'test-experiment',
        name: 'Test Experiment',
        type: ChaosExperimentType.NETWORK_PARTITION,
        description: 'Test network partition scenario',
        parameters: { duration: 30000 },
        maxDuration: 60000,
        rollbackThreshold: {
          errorRate: 0.05,
          latencyP99: 1000,
          availabilityMin: 0.95
        },
        targetComponents: ['test-service'],
        impactRadius: 'single_service',
        monitoringInterval: 5000,
        recoveryValidation: {
          stabilityPeriod: 30000,
          successThreshold: 0.95
        }
      };
      
      chaosFramework.registerExperiment(experiment);
      
      const status = chaosFramework.getExperimentStatus();
      expect(status.registered).toBe(1);
    });
    
    it('should validate production safety constraints', () => {
      const productionFramework = ChaosEngineeringFramework.getInstance({
        productionMode: true
      });
      
      const unsafeExperiment: ChaosExperimentConfig = {
        id: 'unsafe-experiment',
        name: 'Unsafe Experiment',
        type: ChaosExperimentType.SERVICE_FAILURE,
        description: 'Unsafe full system failure',
        parameters: { duration: 300000 },
        maxDuration: 600000, // Too long for production
        rollbackThreshold: {
          errorRate: 0.20, // Too high for production
          latencyP99: 5000,
          availabilityMin: 0.80 // Too low for production
        },
        targetComponents: ['all-services'],
        impactRadius: 'full_system', // Not allowed in production
        monitoringInterval: 10000,
        recoveryValidation: {
          stabilityPeriod: 60000,
          successThreshold: 0.90
        }
      };
      
      expect(() => {
        productionFramework.registerExperiment(unsafeExperiment);
      }).toThrow('Full system impact not allowed in production');
    });
  });
  
  describe('Experiment Execution', () => {
    beforeEach(() => {
      const experiment: ChaosExperimentConfig = {
        id: 'network-test',
        name: 'Network Test',
        type: ChaosExperimentType.NETWORK_PARTITION,
        description: 'Network partition test',
        parameters: { 
          duration: 10000,
          latencyMs: 100,
          dropPercentage: 5
        },
        maxDuration: 30000,
        rollbackThreshold: {
          errorRate: 0.10,
          latencyP99: 2000,
          availabilityMin: 0.90
        },
        targetComponents: ['network'],
        impactRadius: 'single_service',
        monitoringInterval: 2000,
        recoveryValidation: {
          stabilityPeriod: 5000,
          successThreshold: 0.95
        }
      };
      
      chaosFramework.registerExperiment(experiment);
    });
    
    it('should execute experiment successfully', async () => {
      const result = await chaosFramework.executeExperiment('network-test');
      
      expect(result.experimentId).toBe('network-test');
      expect(result.state).toBe(ChaosExperimentState.COMPLETED);
      expect(result.startTime).toBeDefined();
      expect(result.endTime).toBeDefined();
      expect(result.timeline.length).toBeGreaterThan(0);
      expect(result.insights).toBeDefined();
      expect(result.insights.resilience).toBeGreaterThanOrEqual(0);
      expect(result.insights.resilience).toBeLessThanOrEqual(100);
    });
    
    it('should handle experiment not found', async () => {
      await expect(chaosFramework.executeExperiment('non-existent')).rejects.toThrow(
        'Experiment non-existent not found'
      );
    });
    
    it('should respect concurrent experiment limits', async () => {
      // Register multiple experiments
      const experiment2: ChaosExperimentConfig = {
        id: 'network-test-2',
        name: 'Network Test 2',
        type: ChaosExperimentType.NETWORK_PARTITION,
        description: 'Second network test',
        parameters: { duration: 10000 },
        maxDuration: 30000,
        rollbackThreshold: {
          errorRate: 0.05,
          latencyP99: 1000,
          availabilityMin: 0.95
        },
        targetComponents: ['network'],
        impactRadius: 'single_service',
        monitoringInterval: 5000,
        recoveryValidation: {
          stabilityPeriod: 30000,
          successThreshold: 0.95
        }
      };
      
      const experiment3: ChaosExperimentConfig = {
        ...experiment2,
        id: 'network-test-3',
        name: 'Network Test 3'
      };
      
      chaosFramework.registerExperiment(experiment2);
      chaosFramework.registerExperiment(experiment3);
      
      // Start two experiments (should reach limit)
      const promise1 = chaosFramework.executeExperiment('network-test');
      const promise2 = chaosFramework.executeExperiment('network-test-2');
      
      // Third experiment should be rejected
      await expect(chaosFramework.executeExperiment('network-test-3')).rejects.toThrow(
        'Maximum concurrent experiments exceeded'
      );
      
      // Wait for first two to complete
      await Promise.all([promise1, promise2]);
    }, 60000); // Increase timeout for this test
  });
  
  describe('Emergency Stop', () => {
    beforeEach(() => {
      const experiment: ChaosExperimentConfig = {
        id: 'long-running-test',
        name: 'Long Running Test',
        type: ChaosExperimentType.MEMORY_PRESSURE,
        description: 'Long running memory pressure test',
        parameters: { duration: 30000, memoryMB: 100 },
        maxDuration: 60000,
        rollbackThreshold: {
          errorRate: 0.05,
          latencyP99: 1000,
          availabilityMin: 0.95
        },
        targetComponents: ['memory'],
        impactRadius: 'single_service',
        monitoringInterval: 5000,
        recoveryValidation: {
          stabilityPeriod: 10000,
          successThreshold: 0.95
        }
      };
      
      chaosFramework.registerExperiment(experiment);
    });
    
    it('should emergency stop all experiments', async () => {
      // Start experiment
      const experimentPromise = chaosFramework.executeExperiment('long-running-test');
      
      // Give it time to start
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Emergency stop
      await chaosFramework.emergencyStop('Test emergency stop');
      
      // Wait for experiment to be stopped
      const result = await experimentPromise;
      
      expect(result.state).toBe(ChaosExperimentState.ROLLED_BACK);
      expect(result.safetyActions).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            action: 'emergency_rollback',
            reason: 'Test emergency stop'
          })
        ])
      );
    });
  });
  
  describe('Safety Monitoring', () => {
    it('should track safety thresholds', async () => {
      // Mock getCurrentMetrics to return high error rate
      const originalGetCurrentMetrics = (chaosFramework as any).getCurrentMetrics;
      (chaosFramework as any).getCurrentMetrics = vi.fn().mockResolvedValue({
        errorRate: 0.15, // Above emergency threshold (0.1)
        latencyP50: 10,
        latencyP95: 50,
        latencyP99: 100,
        throughput: 100,
        availability: 0.95
      });
      
      const experiment: ChaosExperimentConfig = {
        id: 'high-error-test',
        name: 'High Error Test',
        type: ChaosExperimentType.SERVICE_FAILURE,
        description: 'Test that should trigger emergency stop',
        parameters: { duration: 5000 },
        maxDuration: 10000,
        rollbackThreshold: {
          errorRate: 0.05,
          latencyP99: 1000,
          availabilityMin: 0.95
        },
        targetComponents: ['service'],
        impactRadius: 'single_service',
        monitoringInterval: 1000,
        recoveryValidation: {
          stabilityPeriod: 5000,
          successThreshold: 0.95
        }
      };
      
      chaosFramework.registerExperiment(experiment);
      
      const result = await chaosFramework.executeExperiment('high-error-test');
      
      // Should have been rolled back due to high error rate
      expect(result.state).toBe(ChaosExperimentState.ROLLED_BACK);
      expect(result.safetyActions).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            action: expect.stringMatching(/rollback/)
          })
        ])
      );
      
      // Restore original method
      (chaosFramework as any).getCurrentMetrics = originalGetCurrentMetrics;
    });
  });
  
  describe('Experiment Status Tracking', () => {
    it('should track experiment states correctly', async () => {
      const experiment: ChaosExperimentConfig = {
        id: 'status-test',
        name: 'Status Test',
        type: ChaosExperimentType.NETWORK_PARTITION,
        description: 'Test for status tracking',
        parameters: { duration: 5000 },
        maxDuration: 10000,
        rollbackThreshold: {
          errorRate: 0.05,
          latencyP99: 1000,
          availabilityMin: 0.95
        },
        targetComponents: ['network'],
        impactRadius: 'single_service',
        monitoringInterval: 2000,
        recoveryValidation: {
          stabilityPeriod: 3000,
          successThreshold: 0.95
        }
      };
      
      chaosFramework.registerExperiment(experiment);
      
      // Check initial status
      let status = chaosFramework.getExperimentStatus();
      expect(status.registered).toBe(1);
      expect(status.active).toBe(0);
      
      // Start experiment
      const experimentPromise = chaosFramework.executeExperiment('status-test');
      
      // Give it time to start
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Check active status
      status = chaosFramework.getExperimentStatus();
      expect(status.active).toBe(1);
      expect(status.experiments).toHaveLength(1);
      expect(status.experiments[0].id).toBe('status-test');
      expect(status.experiments[0].state).toBeDefined();
      
      // Wait for completion
      await experimentPromise;
      
      // Check final status
      status = chaosFramework.getExperimentStatus();
      expect(status.active).toBe(0);
    });
  });
});

describe('ChaosExperimentSuite', () => {
  describe('Experiment Suite Creation', () => {
    it('should create comprehensive test suite', () => {
      const experiments = ChaosExperimentSuite.createRobustnessTestSuite();
      
      expect(experiments.length).toBeGreaterThan(10);
      
      // Check that all major experiment types are represented
      const types = experiments.map(exp => exp.type);
      expect(types).toContain(ChaosExperimentType.DATABASE_FAILURE);
      expect(types).toContain(ChaosExperimentType.NATS_DISRUPTION);
      expect(types).toContain(ChaosExperimentType.NETWORK_PARTITION);
      expect(types).toContain(ChaosExperimentType.MEMORY_PRESSURE);
      expect(types).toContain(ChaosExperimentType.CONCURRENT_LOAD);
      expect(types).toContain(ChaosExperimentType.SERVICE_FAILURE);
      expect(types).toContain(ChaosExperimentType.CACHE_CORRUPTION);
    });
    
    it('should create production-safe test suite', () => {
      const experiments = ChaosExperimentSuite.createProductionSafeTestSuite();
      
      // All experiments should have safe impact radius
      experiments.forEach(exp => {
        expect(exp.impactRadius).not.toBe('full_system');
        expect(exp.maxDuration).toBeLessThanOrEqual(180000); // Max 3 minutes
        expect(exp.rollbackThreshold.errorRate).toBeLessThanOrEqual(0.02);
        expect(exp.rollbackThreshold.availabilityMin).toBeGreaterThanOrEqual(0.98);
      });
    });
    
    it('should register all experiments', async () => {
      const chaosFramework = ChaosEngineeringFramework.getInstance();
      
      await ChaosExperimentSuite.registerAllExperiments(chaosFramework, false);
      
      const status = chaosFramework.getExperimentStatus();
      expect(status.registered).toBeGreaterThan(10);
    });
    
    it('should register only production-safe experiments in production mode', async () => {
      const chaosFramework = ChaosEngineeringFramework.getInstance();
      
      await ChaosExperimentSuite.registerAllExperiments(chaosFramework, true);
      
      const status = chaosFramework.getExperimentStatus();
      const allExperiments = ChaosExperimentSuite.createRobustnessTestSuite();
      const productionSafeExperiments = ChaosExperimentSuite.createProductionSafeTestSuite();
      
      expect(status.registered).toBe(productionSafeExperiments.length);
      expect(status.registered).toBeLessThan(allExperiments.length);
    });
  });
  
  describe('Individual Scenario Creation', () => {
    it('should create valid database scenarios', () => {
      const scenarios = [
        ChaosExperimentSuite.createRobustnessTestSuite().find(exp => 
          exp.name.includes('Connection Pool')
        ),
        ChaosExperimentSuite.createRobustnessTestSuite().find(exp => 
          exp.name.includes('Query Timeout')
        )
      ];
      
      scenarios.forEach(scenario => {
        expect(scenario).toBeDefined();
        expect(scenario!.type).toBe(ChaosExperimentType.DATABASE_FAILURE);
        expect(scenario!.parameters).toBeDefined();
        expect(scenario!.rollbackThreshold).toBeDefined();
        expect(scenario!.targetComponents).toContain('database');
      });
    });
    
    it('should create valid NATS scenarios', () => {
      const scenarios = [
        ChaosExperimentSuite.createRobustnessTestSuite().find(exp => 
          exp.name.includes('NATS Server')
        ),
        ChaosExperimentSuite.createRobustnessTestSuite().find(exp => 
          exp.name.includes('Message Loss')
        )
      ];
      
      scenarios.forEach(scenario => {
        expect(scenario).toBeDefined();
        expect(scenario!.type).toBe(ChaosExperimentType.NATS_DISRUPTION);
        expect(scenario!.parameters).toBeDefined();
        expect(scenario!.targetComponents.some(comp => 
          comp.includes('messaging') || comp.includes('jetstream')
        )).toBe(true);
      });
    });
    
    it('should create valid load test scenarios', () => {
      const scenarios = [
        ChaosExperimentSuite.createRobustnessTestSuite().find(exp => 
          exp.name.includes('Realistic')
        ),
        ChaosExperimentSuite.createRobustnessTestSuite().find(exp => 
          exp.name.includes('Traffic Spike')
        )
      ];
      
      scenarios.forEach(scenario => {
        expect(scenario).toBeDefined();
        expect(scenario!.type).toBe(ChaosExperimentType.CONCURRENT_LOAD);
        expect(scenario!.parameters).toBeDefined();
        expect(scenario!.parameters.loadPattern).toBeDefined();
      });
    });
  });
});

describe('Safety Validation', () => {
  let framework: ChaosEngineeringFramework;
  
  beforeEach(() => {
    (ChaosEngineeringFramework as any).instance = null;
    framework = ChaosEngineeringFramework.getInstance({
      productionMode: true,
      safetyLimits: {
        maxConcurrentExperiments: 1,
        maxExperimentDuration: 120000,
        emergencyStopThreshold: {
          errorRate: 0.02,
          latencyMultiplier: 2.0
        }
      }
    });
  });
  
  it('should reject unsafe experiments in production', () => {
    const unsafeExperiment: ChaosExperimentConfig = {
      id: 'unsafe-test',
      name: 'Unsafe Test',
      type: ChaosExperimentType.SERVICE_FAILURE,
      description: 'Unsafe test',
      parameters: {},
      maxDuration: 600000, // Too long
      rollbackThreshold: {
        errorRate: 0.10, // Too high
        latencyP99: 1000,
        availabilityMin: 0.80 // Too low
      },
      targetComponents: ['all'],
      impactRadius: 'full_system', // Not allowed
      monitoringInterval: 5000,
      recoveryValidation: {
        stabilityPeriod: 60000,
        successThreshold: 0.95
      }
    };
    
    expect(() => {
      framework.registerExperiment(unsafeExperiment);
    }).toThrow();
  });
  
  it('should accept safe experiments in production', () => {
    const safeExperiment: ChaosExperimentConfig = {
      id: 'safe-test',
      name: 'Safe Test',
      type: ChaosExperimentType.NETWORK_PARTITION,
      description: 'Safe test',
      parameters: { duration: 30000 },
      maxDuration: 60000,
      rollbackThreshold: {
        errorRate: 0.01, // Low
        latencyP99: 500,
        availabilityMin: 0.99 // High
      },
      targetComponents: ['single-service'],
      impactRadius: 'single_service',
      monitoringInterval: 5000,
      recoveryValidation: {
        stabilityPeriod: 60000,
        successThreshold: 0.98
      }
    };
    
    expect(() => {
      framework.registerExperiment(safeExperiment);
    }).not.toThrow();
    
    const status = framework.getExperimentStatus();
    expect(status.registered).toBe(1);
  });
});

describe('Integration Tests', () => {
  it('should integrate with existing resilience manager', () => {
    const framework = ChaosEngineeringFramework.getInstance();
    
    // Framework should initialize without throwing
    expect(framework).toBeDefined();
    
    // Should be able to get status
    const status = framework.getExperimentStatus();
    expect(status).toHaveProperty('registered');
    expect(status).toHaveProperty('active');
    expect(status).toHaveProperty('experiments');
  });
  
  it('should handle telemetry integration', async () => {
    const framework = ChaosEngineeringFramework.getInstance();
    
    const experiment: ChaosExperimentConfig = {
      id: 'telemetry-test',
      name: 'Telemetry Test',
      type: ChaosExperimentType.NETWORK_PARTITION,
      description: 'Test telemetry integration',
      parameters: { duration: 1000 },
      maxDuration: 5000,
      rollbackThreshold: {
        errorRate: 0.05,
        latencyP99: 1000,
        availabilityMin: 0.95
      },
      targetComponents: ['network'],
      impactRadius: 'single_service',
      monitoringInterval: 1000,
      recoveryValidation: {
        stabilityPeriod: 2000,
        successThreshold: 0.95
      }
    };
    
    framework.registerExperiment(experiment);
    const result = await framework.executeExperiment('telemetry-test');
    
    // Should complete without telemetry errors
    expect(result.state).toBe(ChaosExperimentState.COMPLETED);
  });
});