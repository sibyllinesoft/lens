/**
 * Enterprise-Grade Systems Integration
 * 
 * Coordinates the four enterprise systems:
 * 1. Task-Level Correctness with Witness Set Mining
 * 2. Declarative Query-DAG Planner with DSL
 * 3. Tenant Economics as Math (Convex Programming)
 * 4. Adversarial/Durability Drills
 * 
 * All systems are embedder-agnostic, span-safe, and optimize outcomes-per-millisecond.
 */

import type { SearchContext, SearchHit } from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';
import { WitnessSetMiner, type WitnessSet, type WitnessSetMiningConfig, type TaskSuccess, type SuccessAtK } from './witness-set-mining.js';
import { QueryDAGPlanner, type QueryPlan, type PlanExecution, type LexScanOp } from './query-dag-planner.js';
import { TenantEconomicsEngine, type TenantProfile, type SLAUtilityMetrics } from './tenant-economics.js';
import { AdversarialDurabilityEngine, type AdversarialContent, type SystemResilience } from './adversarial-durability.js';

export interface EnterpriseSystemsConfig {
  readonly witnessSetMining: {
    readonly ciLogsPath: string;
    readonly gitRepoPath: string;
    readonly maxWitnessSize?: number;
    readonly minConfidence?: number;
  };
  readonly queryPlanning: {
    readonly maxPlannerSpendRatio?: number;
    readonly planCacheSize?: number;
    readonly sloConstraints?: {
      readonly maxLatencyMs?: number;
      readonly maxMemoryMB?: number;
      readonly minRecall?: number;
    };
  };
  readonly tenantEconomics: {
    readonly maxCpuTimeMs?: number;
    readonly maxMemoryGB?: number;
    readonly lambdaMs?: number;
    readonly lambdaGB?: number;
  };
  readonly adversarialDurability: {
    readonly maxFileSize?: number;
    readonly maxEntropy?: number;
    readonly minLanguageConfidence?: number;
  };
}

export interface SystemHealthReport {
  readonly timestamp: Date;
  readonly witnessSetMetrics: {
    readonly successAt10: SuccessAtK;
    readonly totalWitnessSets: number;
    readonly slaCompliant: boolean;
  };
  readonly plannerMetrics: {
    readonly spendRatio: number;
    readonly cacheHitRatio: number;
    readonly p99CostMs: number;
    readonly p95CostMs: number;
  };
  readonly tenantMetrics: {
    readonly totalTenants: number;
    readonly avgUtilityScore: number;
    readonly upshiftCompliant: boolean;
    readonly avgUpshift: number;
  };
  readonly adversarialMetrics: {
    readonly quarantinedFiles: number;
    readonly detectionAccuracy: number;
    readonly systemResilience: SystemResilience;
  };
  readonly overallHealthScore: number;
}

export interface SearchRequest {
  readonly context: SearchContext;
  readonly tenantId: string;
  readonly expectedWitnessSet?: readonly string[];
}

export interface EnhancedSearchResult {
  readonly hits: readonly SearchHit[];
  readonly queryPlan: QueryPlan;
  readonly planExecution: PlanExecution;
  readonly witnessSetCoverage: number;
  readonly tenantUtilityScore: number;
  readonly adversarialFiltered: number;
  readonly performanceMetrics: {
    readonly totalLatencyMs: number;
    readonly planningLatencyMs: number;
    readonly executionLatencyMs: number;
    readonly filteringLatencyMs: number;
  };
}

export class EnterpriseSystemsCoordinator {
  private witnessSetMiner: WitnessSetMiner;
  private queryPlanner: QueryDAGPlanner;
  private tenantEconomics: TenantEconomicsEngine;
  private adversarialDurability: AdversarialDurabilityEngine;
  private isInitialized = false;

  constructor(private config: EnterpriseSystemsConfig) {
    this.witnessSetMiner = new WitnessSetMiner(
      config.witnessSetMining.ciLogsPath,
      config.witnessSetMining.gitRepoPath,
      {
        maxWitnessSize: config.witnessSetMining.maxWitnessSize,
        minConfidence: config.witnessSetMining.minConfidence,
      }
    );

    this.queryPlanner = new QueryDAGPlanner();

    this.tenantEconomics = new TenantEconomicsEngine({
      maxCpuTimeMs: config.tenantEconomics.maxCpuTimeMs,
      maxMemoryGB: config.tenantEconomics.maxMemoryGB,
      lambdaMs: config.tenantEconomics.lambdaMs,
      lambdaGB: config.tenantEconomics.lambdaGB,
    });

    this.adversarialDurability = new AdversarialDurabilityEngine({
      maxFileSize: config.adversarialDurability.maxFileSize,
      maxEntropy: config.adversarialDurability.maxEntropy,
      minLanguageConfidence: config.adversarialDurability.minLanguageConfidence,
    });
  }

  /**
   * Initialize all enterprise systems
   */
  async initialize(): Promise<void> {
    const span = LensTracer.createChildSpan('initialize_enterprise_systems');

    try {
      // Initialize systems in parallel where possible
      await Promise.all([
        this.witnessSetMiner.mineWitnessSets(),
        this.tenantEconomics.optimizeResourceAllocation(),
        this.adversarialDurability.addAdversarialCorpusToBenchmark(),
      ]);

      this.isInitialized = true;

      span.setAttributes({ success: true });
      console.log('🏢 Enterprise Systems Coordinator initialized successfully');

    } catch (error) {
      span.recordException(error as Error);
      throw new Error(`Failed to initialize enterprise systems: ${error}`);
    } finally {
      span.end();
    }
  }

  /**
   * Execute enhanced search with all enterprise systems
   */
  async executeEnhancedSearch(request: SearchRequest): Promise<EnhancedSearchResult> {
    if (!this.isInitialized) {
      throw new Error('Enterprise systems not initialized');
    }

    const span = LensTracer.createChildSpan('execute_enhanced_search');
    const searchStart = Date.now();

    try {
      // Phase 1: Resource Allocation Check
      const planningStart = Date.now();
      
      // Generate optimal query plan
      const queryPlan = await this.queryPlanner.generatePlan(
        request.context,
        this.config.queryPlanning.sloConstraints
      );

      // Check tenant resource availability
      const resourceCheck = await this.tenantEconomics.checkResourceAvailability(
        request.tenantId,
        queryPlan.estimatedCostMs,
        queryPlan.estimatedCostMs * 0.1 // Rough memory estimate
      );

      if (!resourceCheck.allowed) {
        throw new Error(`Resource allocation denied: ${resourceCheck.reason}`);
      }

      const planningLatencyMs = Date.now() - planningStart;

      // Phase 2: Query Execution with Adversarial Filtering
      const executionStart = Date.now();
      
      const planExecution = await this.queryPlanner.executePlan(queryPlan, request.context);
      
      // For this demo, we'll simulate search hits
      const rawHits: SearchHit[] = await this.simulateSearchExecution(request.context);
      
      const executionLatencyMs = Date.now() - executionStart;

      // Phase 3: Adversarial Content Filtering
      const filteringStart = Date.now();
      
      const filteredHits = this.adversarialDurability.filterQuarantinedHits(rawHits);
      const adversarialFiltered = rawHits.length - filteredHits.length;
      
      const filteringLatencyMs = Date.now() - filteringStart;

      // Phase 4: Witness Set Evaluation and Resource Consumption
      let witnessSetCoverage = 0;
      if (request.expectedWitnessSet) {
        const hitFiles = filteredHits.map(hit => hit.file);
        const taskSuccess = this.witnessSetMiner.recordTaskResult(
          request.context.query,
          hitFiles,
          request.expectedWitnessSet
        );
        witnessSetCoverage = taskSuccess.coverage;
      } else {
        // Use witness set features for ranking enhancement
        const witnessFeatures = this.witnessSetMiner.getWitnessSetFeatures(
          request.context.query,
          filteredHits
        );
        witnessSetCoverage = witnessFeatures[0]; // Coverage feature
      }

      // Consume tenant resources
      await this.tenantEconomics.consumeResources(
        request.tenantId,
        planExecution.actualCostMs,
        planExecution.memoryUsageMB,
        this.estimateNDCG(filteredHits, witnessSetCoverage)
      );

      // Get tenant utility score
      const tenantMetrics = this.tenantEconomics.getSLAUtilityMetrics(request.tenantId);
      const tenantUtilityScore = tenantMetrics?.utilityScore || 0;

      const totalLatencyMs = Date.now() - searchStart;

      const result: EnhancedSearchResult = {
        hits: filteredHits,
        queryPlan,
        planExecution,
        witnessSetCoverage,
        tenantUtilityScore,
        adversarialFiltered,
        performanceMetrics: {
          totalLatencyMs,
          planningLatencyMs,
          executionLatencyMs,
          filteringLatencyMs,
        },
      };

      span.setAttributes({
        success: true,
        tenant_id: request.tenantId,
        total_hits: filteredHits.length,
        adversarial_filtered: adversarialFiltered,
        witness_coverage: witnessSetCoverage,
        utility_score: tenantUtilityScore,
        total_latency_ms: totalLatencyMs,
      });

      return result;

    } catch (error) {
      span.recordException(error as Error);
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Register a new tenant
   */
  async registerTenant(
    tenantId: string,
    slaClass: 'premium' | 'standard' | 'basic',
    businessPriority?: number
  ): Promise<TenantProfile> {
    return await this.tenantEconomics.registerTenant(tenantId, slaClass, businessPriority);
  }

  /**
   * Generate comprehensive system health report
   */
  async generateHealthReport(): Promise<SystemHealthReport> {
    const span = LensTracer.createChildSpan('generate_health_report');

    try {
      // Gather metrics from all systems
      const [
        successAt10,
        plannerMetrics,
        allSLAMetrics,
        upshiftCompliance,
        adversarialMetrics,
        systemResilience,
      ] = await Promise.all([
        this.witnessSetMiner.calculateSuccessAtK(10),
        this.queryPlanner.getPlannerMetrics(),
        this.tenantEconomics.getAllSLAMetrics(),
        this.tenantEconomics.checkUpshiftCompliance(),
        this.adversarialDurability.getAdversarialMetrics(),
        this.adversarialDurability.monitorSystemResilience(),
      ]);

      // Check SLA compliance for witness sets
      const witnessSetSLA = this.witnessSetMiner.checkSLASuccessAt10();

      const avgUtilityScore = allSLAMetrics.length > 0
        ? allSLAMetrics.reduce((sum, m) => sum + m.utilityScore, 0) / allSLAMetrics.length
        : 0;

      // Calculate overall health score
      const healthComponents = [
        witnessSetSLA.isFlat ? 1.0 : 0.5, // Witness set SLA compliance
        plannerMetrics.spendRatio <= 0.1 ? 1.0 : 0.7, // Planner efficiency
        upshiftCompliance.compliant ? 1.0 : 0.6, // Tenant upshift compliance
        systemResilience.overallHealthScore, // Adversarial resilience
      ];

      const overallHealthScore = healthComponents.reduce((sum, score) => sum + score, 0) / healthComponents.length;

      const report: SystemHealthReport = {
        timestamp: new Date(),
        witnessSetMetrics: {
          successAt10,
          totalWitnessSets: 0, // Would be populated from actual data
          slaCompliant: witnessSetSLA.isFlat,
        },
        plannerMetrics,
        tenantMetrics: {
          totalTenants: allSLAMetrics.length,
          avgUtilityScore,
          upshiftCompliant: upshiftCompliance.compliant,
          avgUpshift: upshiftCompliance.avgUpshift,
        },
        adversarialMetrics: {
          quarantinedFiles: adversarialMetrics.quarantinedFiles,
          detectionAccuracy: adversarialMetrics.detectionAccuracy,
          systemResilience,
        },
        overallHealthScore,
      };

      span.setAttributes({
        success: true,
        overall_health_score: overallHealthScore,
        witness_sla_compliant: witnessSetSLA.isFlat,
        planner_spend_ratio: plannerMetrics.spendRatio,
        upshift_compliant: upshiftCompliance.compliant,
        quarantined_files: adversarialMetrics.quarantinedFiles,
      });

      return report;

    } catch (error) {
      span.recordException(error as Error);
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Start a chaos experiment
   */
  async startChaosExperiment(
    name: string,
    type: 'content_adversary' | 'performance_stress' | 'memory_pressure' | 'disk_corruption',
    targetSystem: 'lexical' | 'structural' | 'semantic' | 'all',
    parameters: Record<string, any> = {},
    durationMs: number = 60000
  ): Promise<string> {
    return await this.adversarialDurability.startChaosExperiment(
      name,
      type,
      targetSystem,
      parameters,
      durationMs
    );
  }

  /**
   * Get query plan by policy for reproducible rankings
   */
  getReproduciblePlan(policyDelta: string): QueryPlan | null {
    return this.queryPlanner.getPlanByPolicy(policyDelta);
  }

  /**
   * Scan file for adversarial content
   */
  async scanFileForAdversarialContent(filePath: string, content?: string): Promise<AdversarialContent | null> {
    return await this.adversarialDurability.scanForAdversarialContent(filePath, content);
  }

  /**
   * Get all active tripwire alerts
   */
  getTripwireAlerts() {
    return this.adversarialDurability.getTripwireAlerts();
  }

  /**
   * Reset daily budgets for all tenants
   */
  async resetDailyBudgets(): Promise<void> {
    await this.tenantEconomics.resetDailyBudgets();
  }

  /**
   * Clear planner cache
   */
  clearPlannerCache(): void {
    this.queryPlanner.clearCache();
  }

  // Private helper methods

  /**
   * Simulate search execution (placeholder)
   */
  private async simulateSearchExecution(context: SearchContext): Promise<SearchHit[]> {
    // This would integrate with the actual search engine
    // For now, return mock results
    return [
      {
        file: 'src/example.ts',
        line: 42,
        col: 10,
        lang: 'typescript',
        snippet: 'function example() { return "hello world"; }',
        score: 0.95,
        why: ['lexical_match', 'symbol_match'],
        byte_offset: 1024,
        span_len: 45,
      },
      {
        file: 'src/utils.ts',
        line: 15,
        col: 5,
        lang: 'typescript',
        snippet: 'export const utils = { example: () => {} };',
        score: 0.82,
        why: ['lexical_match'],
        byte_offset: 512,
        span_len: 38,
      },
    ];
  }

  /**
   * Estimate nDCG based on hits and witness set coverage
   */
  private estimateNDCG(hits: readonly SearchHit[], witnessSetCoverage: number): number {
    if (hits.length === 0) return 0;
    
    // Simple heuristic combining hit quality and witness set coverage
    const avgScore = hits.reduce((sum, hit) => sum + hit.score, 0) / hits.length;
    return (avgScore * 0.7) + (witnessSetCoverage * 0.3);
  }
}

// Export all system types for external use
export type {
  WitnessSet,
  WitnessSetMiningConfig,
  TaskSuccess,
  SuccessAtK,
  QueryPlan,
  PlanExecution,
  LexScanOp,
  TenantProfile,
  SLAUtilityMetrics,
  AdversarialContent,
  SystemResilience,
};

// Export individual system classes for direct access if needed
export {
  WitnessSetMiner,
  QueryDAGPlanner,
  TenantEconomicsEngine,
  AdversarialDurabilityEngine,
};