/**
 * Chaos Engineering Scenarios for Lens Search Engine
 * 
 * Pre-defined chaos experiments targeting specific failure modes:
 * - Database connection failures and timeouts
 * - NATS/JetStream disruption scenarios  
 * - Concurrent user load with realistic patterns
 * - Search pipeline degradation testing
 * - Memory-mapped storage corruption
 * - Circuit breaker validation scenarios
 */

import { v4 as uuidv4 } from 'uuid';
import { 
  ChaosExperimentConfig, 
  ChaosExperimentType,
  ChaosEngineeringFramework 
} from './chaos-engineering-framework.js';
import { MessagingSystem } from './messaging.js';
import { NatsConnection, connect } from 'nats';

/**
 * Database Failure Scenarios
 */
export class DatabaseChaosScenarios {
  
  /**
   * Database connection pool exhaustion
   */
  static createConnectionPoolExhaustion(): ChaosExperimentConfig {
    return {
      id: uuidv4(),
      name: 'Database Connection Pool Exhaustion',
      type: ChaosExperimentType.DATABASE_FAILURE,
      description: 'Exhaust database connection pool to test connection handling and recovery',
      
      parameters: {
        failureType: 'connection_exhaustion',
        exhaustionLevel: 0.9, // Use 90% of connection pool
        duration: 120000, // 2 minutes
        targetOperations: ['search', 'index', 'metadata']
      },
      
      maxDuration: 180000, // 3 minutes max
      rollbackThreshold: {
        errorRate: 0.15, // 15% max error rate
        latencyP99: 1000, // 1 second max latency
        availabilityMin: 0.85 // 85% min availability
      },
      
      targetComponents: ['database', 'search_engine'],
      impactRadius: 'partial_system',
      monitoringInterval: 5000,
      
      recoveryValidation: {
        stabilityPeriod: 60000,
        successThreshold: 0.95
      }
    };
  }
  
  /**
   * Database query timeout simulation
   */
  static createQueryTimeouts(): ChaosExperimentConfig {
    return {
      id: uuidv4(),
      name: 'Database Query Timeouts',
      type: ChaosExperimentType.DATABASE_FAILURE,
      description: 'Inject artificial delays in database queries to test timeout handling',
      
      parameters: {
        failureType: 'query_timeout',
        delayRange: { min: 5000, max: 15000 }, // 5-15 second delays
        affectedQueries: ['complex_search', 'symbol_lookup', 'metadata_fetch'],
        injectionRate: 0.3 // 30% of queries affected
      },
      
      maxDuration: 300000, // 5 minutes
      rollbackThreshold: {
        errorRate: 0.20, // 20% max error rate (higher tolerance for timeouts)
        latencyP99: 20000, // 20 seconds max
        availabilityMin: 0.80 // 80% min availability
      },
      
      targetComponents: ['database', 'query_engine'],
      impactRadius: 'partial_system',
      monitoringInterval: 10000,
      
      recoveryValidation: {
        stabilityPeriod: 90000,
        successThreshold: 0.95
      }
    };
  }
  
  /**
   * Database connection intermittent failures
   */
  static createIntermittentConnectionFailures(): ChaosExperimentConfig {
    return {
      id: uuidv4(),
      name: 'Database Intermittent Connection Failures',
      type: ChaosExperimentType.DATABASE_FAILURE,
      description: 'Random connection drops to test retry and recovery mechanisms',
      
      parameters: {
        failureType: 'intermittent_connection',
        failureRate: 0.05, // 5% of connections fail
        reconnectDelay: { min: 1000, max: 5000 }, // 1-5 second reconnect delays
        burstFailures: true, // Simulate burst failures
        burstDuration: 10000 // 10-second bursts
      },
      
      maxDuration: 600000, // 10 minutes
      rollbackThreshold: {
        errorRate: 0.10, // 10% max error rate
        latencyP99: 5000, // 5 seconds max
        availabilityMin: 0.90 // 90% min availability
      },
      
      targetComponents: ['database', 'connection_pool'],
      impactRadius: 'partial_system',
      monitoringInterval: 5000,
      
      recoveryValidation: {
        stabilityPeriod: 120000,
        successThreshold: 0.98
      }
    };
  }
}

/**
 * NATS/JetStream Disruption Scenarios
 */
export class NATSChaosScenarios {
  
  /**
   * NATS server unavailability
   */
  static createNATSServerFailure(): ChaosExperimentConfig {
    return {
      id: uuidv4(),
      name: 'NATS Server Failure',
      type: ChaosExperimentType.NATS_DISRUPTION,
      description: 'Simulate NATS server going down to test message queue resilience',
      
      parameters: {
        failureType: 'server_down',
        duration: 60000, // 1 minute downtime
        reconnectAttempts: 10,
        backoffMultiplier: 2,
        maxReconnectDelay: 30000
      },
      
      maxDuration: 300000, // 5 minutes max
      rollbackThreshold: {
        errorRate: 0.25, // 25% max error rate during messaging failure
        latencyP99: 10000, // 10 seconds max
        availabilityMin: 0.75 // 75% min availability (degraded mode)
      },
      
      targetComponents: ['messaging', 'work_distribution', 'index_pipeline'],
      impactRadius: 'partial_system',
      monitoringInterval: 10000,
      
      recoveryValidation: {
        stabilityPeriod: 180000, // 3 minutes for full recovery
        successThreshold: 0.95
      }
    };
  }
  
  /**
   * JetStream message loss simulation
   */
  static createMessageLoss(): ChaosExperimentConfig {
    return {
      id: uuidv4(),
      name: 'JetStream Message Loss',
      type: ChaosExperimentType.NATS_DISRUPTION,
      description: 'Simulate message loss in JetStream to test duplicate handling and recovery',
      
      parameters: {
        failureType: 'message_loss',
        lossRate: 0.02, // 2% message loss
        affectedStreams: ['LENS_WORK'],
        affectedSubjects: ['lens.work.index_shard', 'lens.work.build_symbols'],
        duration: 120000 // 2 minutes
      },
      
      maxDuration: 240000, // 4 minutes max
      rollbackThreshold: {
        errorRate: 0.05, // 5% max error rate
        latencyP99: 30000, // 30 seconds max (for reprocessing)
        availabilityMin: 0.95 // 95% min availability
      },
      
      targetComponents: ['jetstream', 'work_processing'],
      impactRadius: 'single_service',
      monitoringInterval: 15000,
      
      recoveryValidation: {
        stabilityPeriod: 120000,
        successThreshold: 0.98
      }
    };
  }
  
  /**
   * Consumer processing delays
   */
  static createConsumerBacklog(): ChaosExperimentConfig {
    return {
      id: uuidv4(),
      name: 'NATS Consumer Processing Backlog',
      type: ChaosExperimentType.NATS_DISRUPTION,
      description: 'Create artificial consumer delays to test backlog handling and recovery',
      
      parameters: {
        failureType: 'consumer_delay',
        processingDelay: { min: 2000, max: 10000 }, // 2-10 second delays
        affectedConsumers: ['lens_index_shard_worker', 'lens_build_symbols_worker'],
        backlogThreshold: 1000, // Messages in backlog
        duration: 180000 // 3 minutes
      },
      
      maxDuration: 360000, // 6 minutes max
      rollbackThreshold: {
        errorRate: 0.02, // 2% max error rate
        latencyP99: 45000, // 45 seconds max
        availabilityMin: 0.98 // 98% min availability
      },
      
      targetComponents: ['consumer_workers', 'message_processing'],
      impactRadius: 'partial_system',
      monitoringInterval: 20000,
      
      recoveryValidation: {
        stabilityPeriod: 300000, // 5 minutes for backlog processing
        successThreshold: 0.99
      }
    };
  }
}

/**
 * Concurrent Load Testing Scenarios
 */
export class ConcurrentLoadScenarios {
  
  /**
   * Realistic user traffic pattern simulation
   */
  static createRealisticLoadTest(): ChaosExperimentConfig {
    return {
      id: uuidv4(),
      name: 'Realistic Concurrent User Load',
      type: ChaosExperimentType.CONCURRENT_LOAD,
      description: 'Simulate realistic user patterns with varying query complexity and frequency',
      
      parameters: {
        loadPattern: 'realistic',
        peakConcurrency: 200, // 200 concurrent users at peak
        rampUpDuration: 60000, // 1 minute ramp up
        sustainDuration: 300000, // 5 minutes sustained
        rampDownDuration: 60000, // 1 minute ramp down
        
        queryDistribution: {
          simple_search: 0.6, // 60% simple searches
          complex_search: 0.25, // 25% complex searches  
          symbol_lookup: 0.10, // 10% symbol lookups
          semantic_search: 0.05 // 5% semantic searches
        },
        
        userBehaviorPatterns: {
          think_time_range: { min: 1000, max: 5000 }, // 1-5 second think time
          session_duration_range: { min: 30000, max: 300000 }, // 30s-5min sessions
          abandon_rate: 0.05, // 5% abandon searches mid-flight
          repeat_query_rate: 0.15 // 15% repeat previous queries
        }
      },
      
      maxDuration: 600000, // 10 minutes max
      rollbackThreshold: {
        errorRate: 0.02, // 2% max error rate
        latencyP99: 500, // 500ms max p99 latency
        availabilityMin: 0.98 // 98% min availability
      },
      
      targetComponents: ['search_engine', 'query_processing', 'index_pipeline'],
      impactRadius: 'full_system',
      monitoringInterval: 10000,
      
      recoveryValidation: {
        stabilityPeriod: 120000,
        successThreshold: 0.99
      }
    };
  }
  
  /**
   * Spike traffic simulation
   */
  static createTrafficSpike(): ChaosExperimentConfig {
    return {
      id: uuidv4(),
      name: 'Traffic Spike Simulation',
      type: ChaosExperimentType.CONCURRENT_LOAD,
      description: 'Sudden traffic spike to test auto-scaling and rate limiting',
      
      parameters: {
        loadPattern: 'spike',
        baselineQPS: 50, // 50 queries per second baseline
        spikeQPS: 500, // 500 queries per second spike (10x)
        spikeDuration: 30000, // 30 second spike
        spikeCount: 3, // 3 spikes during test
        intervalBetweenSpikes: 120000 // 2 minutes between spikes
      },
      
      maxDuration: 480000, // 8 minutes max
      rollbackThreshold: {
        errorRate: 0.05, // 5% max error rate during spikes
        latencyP99: 2000, // 2 seconds max p99 latency
        availabilityMin: 0.90 // 90% min availability during spikes
      },
      
      targetComponents: ['rate_limiter', 'load_balancer', 'circuit_breakers'],
      impactRadius: 'full_system',
      monitoringInterval: 5000,
      
      recoveryValidation: {
        stabilityPeriod: 60000,
        successThreshold: 0.98
      }
    };
  }
  
  /**
   * Slow client simulation
   */
  static createSlowClientTest(): ChaosExperimentConfig {
    return {
      id: uuidv4(),
      name: 'Slow Client Connection Test',
      type: ChaosExperimentType.CONCURRENT_LOAD,
      description: 'Simulate slow clients to test connection timeout and resource management',
      
      parameters: {
        loadPattern: 'slow_clients',
        slowClientPercentage: 0.20, // 20% slow clients
        connectionTimeout: { min: 30000, max: 120000 }, // 30s-2min slow connections
        normalConcurrency: 100, // Normal concurrent users
        slowConcurrency: 25, // Slow concurrent users
        duration: 300000 // 5 minutes
      },
      
      maxDuration: 420000, // 7 minutes max
      rollbackThreshold: {
        errorRate: 0.03, // 3% max error rate
        latencyP99: 1000, // 1 second max for normal requests
        availabilityMin: 0.95 // 95% min availability
      },
      
      targetComponents: ['connection_management', 'resource_limits', 'timeouts'],
      impactRadius: 'single_service',
      monitoringInterval: 15000,
      
      recoveryValidation: {
        stabilityPeriod: 90000,
        successThreshold: 0.97
      }
    };
  }
}

/**
 * Search Pipeline Degradation Scenarios
 */
export class SearchPipelineChaosScenarios {
  
  /**
   * Lexical search degradation
   */
  static createLexicalSearchDegradation(): ChaosExperimentConfig {
    return {
      id: uuidv4(),
      name: 'Lexical Search Pipeline Degradation',
      type: ChaosExperimentType.SERVICE_FAILURE,
      description: 'Degrade lexical search performance to test fallback to other search methods',
      
      parameters: {
        failureType: 'performance_degradation',
        targetService: 'lexical_search',
        degradationLevel: 0.8, // 80% slower responses
        affectedOperations: ['trigram_search', 'token_matching'],
        duration: 180000 // 3 minutes
      },
      
      maxDuration: 300000, // 5 minutes max
      rollbackThreshold: {
        errorRate: 0.01, // 1% max error rate
        latencyP99: 1000, // 1 second max
        availabilityMin: 0.99 // 99% min availability (should fallback)
      },
      
      targetComponents: ['lexical_search', 'search_pipeline'],
      impactRadius: 'single_service',
      monitoringInterval: 10000,
      
      recoveryValidation: {
        stabilityPeriod: 60000,
        successThreshold: 0.99
      }
    };
  }
  
  /**
   * Symbol index corruption
   */
  static createSymbolIndexCorruption(): ChaosExperimentConfig {
    return {
      id: uuidv4(),
      name: 'Symbol Index Corruption',
      type: ChaosExperimentType.CACHE_CORRUPTION,
      description: 'Corrupt symbol index data to test error handling and fallback mechanisms',
      
      parameters: {
        corruptionType: 'symbol_index',
        affectedShards: ['shard_0', 'shard_1'], // Corrupt specific shards
        corruptionLevel: 0.1, // 10% of symbol data corrupted
        corruptionPattern: 'random_bytes', // Random byte corruption
        duration: 240000 // 4 minutes
      },
      
      maxDuration: 360000, // 6 minutes max
      rollbackThreshold: {
        errorRate: 0.15, // 15% max error rate (partial results expected)
        latencyP99: 2000, // 2 seconds max
        availabilityMin: 0.85 // 85% min availability (graceful degradation)
      },
      
      targetComponents: ['symbol_index', 'enhanced_symbols'],
      impactRadius: 'partial_system',
      monitoringInterval: 15000,
      
      recoveryValidation: {
        stabilityPeriod: 180000, // 3 minutes for index rebuild
        successThreshold: 0.95
      }
    };
  }
  
  /**
   * Memory-mapped file corruption
   */
  static createMemoryMappedFileCorruption(): ChaosExperimentConfig {
    return {
      id: uuidv4(),
      name: 'Memory-Mapped File Corruption',
      type: ChaosExperimentType.CACHE_CORRUPTION,
      description: 'Corrupt memory-mapped storage files to test data integrity checks',
      
      parameters: {
        corruptionType: 'mmap_file',
        targetFiles: ['bitmap_index', 'roaring_bitmaps'],
        corruptionLocations: 'header_and_random', // Corrupt headers and random locations
        checksumValidation: true, // Test checksum validation
        duration: 300000 // 5 minutes
      },
      
      maxDuration: 450000, // 7.5 minutes max
      rollbackThreshold: {
        errorRate: 0.25, // 25% max error rate (significant data corruption)
        latencyP99: 5000, // 5 seconds max (recovery operations)
        availabilityMin: 0.75 // 75% min availability (major degradation expected)
      },
      
      targetComponents: ['storage', 'memory_mapping', 'bitmap_operations'],
      impactRadius: 'partial_system',
      monitoringInterval: 20000,
      
      recoveryValidation: {
        stabilityPeriod: 300000, // 5 minutes for full recovery
        successThreshold: 0.90
      }
    };
  }
}

/**
 * Circuit Breaker Validation Scenarios
 */
export class CircuitBreakerChaosScenarios {
  
  /**
   * External service failure to test circuit breaker
   */
  static createExternalServiceFailure(): ChaosExperimentConfig {
    return {
      id: uuidv4(),
      name: 'External Service Failure - Circuit Breaker Test',
      type: ChaosExperimentType.SERVICE_FAILURE,
      description: 'Fail external dependencies to validate circuit breaker behavior',
      
      parameters: {
        failureType: 'external_service',
        targetServices: ['embedding_service', 'analytics_service'],
        failurePattern: 'complete_failure', // All requests fail
        circuitBreakerName: 'external-api',
        expectedState: 'OPEN', // Circuit should open
        duration: 90000 // 1.5 minutes
      },
      
      maxDuration: 180000, // 3 minutes max
      rollbackThreshold: {
        errorRate: 0.05, // 5% max error rate (circuit breaker should prevent most errors)
        latencyP99: 100, // 100ms max (fast failures)
        availabilityMin: 0.95 // 95% min availability (fallback mechanisms)
      },
      
      targetComponents: ['circuit_breakers', 'external_apis', 'fallback_mechanisms'],
      impactRadius: 'single_service',
      monitoringInterval: 10000,
      
      recoveryValidation: {
        stabilityPeriod: 120000, // 2 minutes for circuit breaker recovery
        successThreshold: 0.99
      }
    };
  }
  
  /**
   * Gradual service degradation for half-open circuit testing
   */
  static createGradualServiceRecovery(): ChaosExperimentConfig {
    return {
      id: uuidv4(),
      name: 'Gradual Service Recovery - Half-Open Circuit Test',
      type: ChaosExperimentType.SERVICE_FAILURE,
      description: 'Test circuit breaker half-open state with gradual service recovery',
      
      parameters: {
        failureType: 'gradual_recovery',
        targetService: 'search-engine',
        recoveryPattern: 'exponential', // Exponential recovery curve
        initialFailureRate: 1.0, // 100% failures initially
        finalFailureRate: 0.0, // 0% failures at end
        recoveryDuration: 240000, // 4-minute recovery period
        circuitBreakerName: 'search-engine'
      },
      
      maxDuration: 360000, // 6 minutes max
      rollbackThreshold: {
        errorRate: 0.30, // 30% max error rate during recovery
        latencyP99: 2000, // 2 seconds max
        availabilityMin: 0.70 // 70% min availability during recovery
      },
      
      targetComponents: ['circuit_breakers', 'service_health', 'recovery_logic'],
      impactRadius: 'single_service',
      monitoringInterval: 15000,
      
      recoveryValidation: {
        stabilityPeriod: 180000, // 3 minutes for full stability
        successThreshold: 0.95
      }
    };
  }
}

/**
 * Experiment Suite Factory
 */
export class ChaosExperimentSuite {
  
  /**
   * Create a comprehensive robustness test suite
   */
  static createRobustnessTestSuite(): ChaosExperimentConfig[] {
    return [
      // Database scenarios
      DatabaseChaosScenarios.createConnectionPoolExhaustion(),
      DatabaseChaosScenarios.createQueryTimeouts(),
      DatabaseChaosScenarios.createIntermittentConnectionFailures(),
      
      // NATS/JetStream scenarios
      NATSChaosScenarios.createNATSServerFailure(),
      NATSChaosScenarios.createMessageLoss(),
      NATSChaosScenarios.createConsumerBacklog(),
      
      // Load testing scenarios
      ConcurrentLoadScenarios.createRealisticLoadTest(),
      ConcurrentLoadScenarios.createTrafficSpike(),
      ConcurrentLoadScenarios.createSlowClientTest(),
      
      // Search pipeline scenarios
      SearchPipelineChaosScenarios.createLexicalSearchDegradation(),
      SearchPipelineChaosScenarios.createSymbolIndexCorruption(),
      SearchPipelineChaosScenarios.createMemoryMappedFileCorruption(),
      
      // Circuit breaker scenarios
      CircuitBreakerChaosScenarios.createExternalServiceFailure(),
      CircuitBreakerChaosScenarios.createGradualServiceRecovery()
    ];
  }
  
  /**
   * Create production-safe experiment suite
   */
  static createProductionSafeTestSuite(): ChaosExperimentConfig[] {
    const experiments = this.createRobustnessTestSuite();
    
    // Apply production safety constraints
    return experiments.map(experiment => {
      // Reduce impact radius for production
      if (experiment.impactRadius === 'full_system') {
        experiment.impactRadius = 'partial_system';
      }
      
      // Shorter durations for production
      experiment.maxDuration = Math.min(experiment.maxDuration, 180000); // Max 3 minutes
      
      // Stricter rollback thresholds
      experiment.rollbackThreshold.errorRate = Math.min(experiment.rollbackThreshold.errorRate, 0.02);
      experiment.rollbackThreshold.availabilityMin = Math.max(experiment.rollbackThreshold.availabilityMin, 0.98);
      
      return experiment;
    }).filter(experiment => 
      // Only include low-risk experiments in production
      experiment.impactRadius !== 'full_system' && 
      experiment.rollbackThreshold.errorRate <= 0.02
    );
  }
  
  /**
   * Register all experiments with the chaos framework
   */
  static async registerAllExperiments(chaosFramework: ChaosEngineeringFramework, production: boolean = false): Promise<void> {
    const experiments = production ? 
      this.createProductionSafeTestSuite() : 
      this.createRobustnessTestSuite();
    
    console.log(`🧪 Registering ${experiments.length} chaos experiments (production: ${production})`);
    
    for (const experiment of experiments) {
      chaosFramework.registerExperiment(experiment);
    }
    
    console.log('✅ All chaos experiments registered');
  }
}