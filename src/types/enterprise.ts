/**
 * Enterprise System Types
 * 
 * Type definitions for the four enterprise-grade systems:
 * - Task-Level Correctness with Witness Set Mining
 * - Declarative Query-DAG Planner with DSL  
 * - Tenant Economics as Math (Convex Programming)
 * - Adversarial/Durability Drills
 */

// Import and re-export all types from the systems
import type {
  // Witness Set Mining Types
  WitnessSet,
  WitnessSetMiningConfig,
  TaskSuccess,
  SuccessAtK,
  
  // Query DAG Planner Types
  LexScanOp,
  QueryPlan,
  PlanExecution,
  
  // Tenant Economics Types
  TenantProfile,
  SLAUtilityMetrics,
  
  // Adversarial Durability Types
  AdversarialContent,
  SystemResilience,
  
  // Coordinator Types
  EnterpriseSystemsConfig,
  SystemHealthReport,
  SearchRequest,
  EnhancedSearchResult,
} from '../systems/index.js';

// Re-export the imported types
export type {
  WitnessSet,
  WitnessSetMiningConfig,
  TaskSuccess,
  SuccessAtK,
  LexScanOp,
  QueryPlan,
  PlanExecution,
  TenantProfile,
  SLAUtilityMetrics,
  AdversarialContent,
  SystemResilience,
  EnterpriseSystemsConfig,
  SystemHealthReport,
  SearchRequest,
  EnhancedSearchResult,
};

// Local type definitions for missing types from systems
export interface TenantUsageHistory {
  readonly tenantId: string;
  readonly date: Date;
  readonly totalCpuMs: number;
  readonly totalMemoryMB: number;
  readonly totalQueries: number;
  readonly avgUtilityScore: number;
}

export interface ResourceAllocation {
  readonly cpuTimeMs: number;
  readonly memoryMB: number;
  readonly priority: number;
}

export interface SystemResources {
  readonly totalCpuMs: number;
  readonly totalMemoryMB: number;
  readonly availableCpuMs: number;
  readonly availableMemoryMB: number;
}

export interface UtilityFunction {
  readonly a: number;
  readonly b: number;
  readonly c: number;
}

export interface SpendGovernor {
  readonly dailyBudgetMs: number;
  readonly currentSpendMs: number;
  readonly budgetUtilization: number;
}

export interface ShardCredit {
  readonly creditMs: number;
  readonly expirationDate: Date;
}

export interface QuarantinePolicy {
  readonly maxFileSize: number;
  readonly maxEntropy: number;
  readonly quarantineDurationMs: number;
}

export interface AdversarialMetrics {
  readonly quarantinedFiles: number;
  readonly detectionAccuracy: number;
  readonly falsePositiveRate: number;
}

export interface ChaosExperiment {
  readonly id: string;
  readonly name: string;
  readonly type: string;
  readonly status: 'running' | 'completed' | 'failed';
  readonly startTime: Date;
  readonly endTime?: Date;
}

export interface TripwireAlert {
  readonly id: string;
  readonly severity: 'low' | 'medium' | 'high' | 'critical';
  readonly message: string;
  readonly timestamp: Date;
  readonly acknowledged: boolean;
}

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