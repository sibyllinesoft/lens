/**
 * Enterprise System Types
 * 
 * Type definitions for the four enterprise-grade systems:
 * - Task-Level Correctness with Witness Set Mining
 * - Declarative Query-DAG Planner with DSL  
 * - Tenant Economics as Math (Convex Programming)
 * - Adversarial/Durability Drills
 */

// Re-export all types from the systems
export type {
  // Witness Set Mining Types
  WitnessSet,
  WitnessSetMiningConfig,
  TaskSuccess,
  SuccessAtK,
  
  // Query DAG Planner Types
  LexScanOp,
  StructOp,
  SliceOp,
  ANNOp,
  RerankOp,
  PlanOperator,
  QueryPlan,
  SLOConstraints,
  OperatorCostModel,
  PlanExecution,
  PlanCache,
  
  // Tenant Economics Types
  TenantProfile,
  TenantUsageHistory,
  ResourceAllocation,
  SystemResources,
  UtilityFunction,
  SLAUtilityMetrics,
  SpendGovernor,
  ShardCredit,
  
  // Adversarial Durability Types
  AdversarialContent,
  QuarantinePolicy,
  AdversarialMetrics,
  ChaosExperiment,
  SystemResilience,
  TripwireAlert,
  
  // Coordinator Types
  EnterpriseSystemsConfig,
  SystemHealthReport,
  SearchRequest,
  EnhancedSearchResult,
} from '../systems/index.js';

// Additional enterprise-specific interfaces

export interface EnterpriseSearchMetrics {
  readonly witnessSetSuccessRate: number;
  readonly plannerEfficiency: number;
  readonly tenantUtilityScore: number;
  readonly adversarialFilterRate: number;
  readonly overallPerformanceMs: number;
  readonly costPerMillisecond: number;
  readonly qualityScore: number; // Combined nDCG and witness coverage
}

export interface EnterpriseDashboard {
  readonly systemHealth: SystemHealthReport;
  readonly recentMetrics: readonly EnterpriseSearchMetrics[];
  readonly activeAlerts: readonly TripwireAlert[];
  readonly tenantSummary: readonly {
    readonly tenantId: string;
    readonly slaClass: 'premium' | 'standard' | 'basic';
    readonly utilityScore: number;
    readonly resourceUsage: number;
    readonly upshift: number;
  }[];
  readonly plannerInsights: {
    readonly topPlans: readonly QueryPlan[];
    readonly planCacheEfficiency: number;
    readonly costOptimization: number;
  };
}

export interface EnterpriseAPI {
  readonly search: (request: SearchRequest) => Promise<EnhancedSearchResult>;
  readonly registerTenant: (tenantId: string, slaClass: 'premium' | 'standard' | 'basic', priority?: number) => Promise<TenantProfile>;
  readonly getHealthReport: () => Promise<SystemHealthReport>;
  readonly getDashboard: () => Promise<EnterpriseDashboard>;
  readonly startChaosExperiment: (name: string, type: string, params?: Record<string, any>) => Promise<string>;
  readonly getReproduciblePlan: (policyDelta: string) => QueryPlan | null;
  readonly getTripwireAlerts: () => readonly TripwireAlert[];
  readonly resetTenantBudgets: () => Promise<void>;
}