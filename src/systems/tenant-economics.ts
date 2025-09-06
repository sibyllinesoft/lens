/**
 * Tenant Economics as Math (Convex Programming)
 * 
 * Allocates compute with convex program: maximize Σᵢ uᵢ(xᵢ) subject to Σᵢ xᵢᵐˢ ≤ M, Σᵢ xᵢᵐᵉᵐ ≤ G
 * Where uᵢ = αᵢ·ΔnDCG - λₘₛxᵢᵐˢ - λ_GB xᵢᵐᵉᵐ and αᵢ reflects business priority
 */

import type { SearchContext } from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';
import { PRODUCTION_CONFIG } from '../types/config.js';

export interface TenantProfile {
  readonly id: string;
  readonly businessPriority: number; // αᵢ - business priority coefficient
  readonly slaClass: 'premium' | 'standard' | 'basic';
  readonly historicalUsage: TenantUsageHistory;
  readonly currentAllocation: ResourceAllocation;
  readonly createdAt: Date;
}

export interface TenantUsageHistory {
  readonly avgLatencyMs: number;
  readonly avgMemoryMB: number;
  readonly avgNDCG: number;
  readonly queryVolume24h: number;
  readonly utilityScore: number;
  readonly lastUpdated: Date;
}

export interface ResourceAllocation {
  readonly tenantId: string;
  readonly cpuTimeMs: number;     // xᵢᵐˢ
  readonly memoryMB: number;      // xᵢᵐᵉᵐ
  readonly priorityWeight: number; // Priority in resource contention
  readonly estimatedUtility: number; // uᵢ(xᵢ)
  readonly allocatedAt: Date;
  readonly expiresAt: Date;
}

export interface SystemResources {
  readonly maxCpuTimeMs: number;  // M - total CPU budget
  readonly maxMemoryGB: number;   // G - total memory budget
  readonly lambdaMs: number;      // λₘₛ - CPU cost coefficient  
  readonly lambdaGB: number;      // λ_GB - memory cost coefficient
  readonly windowSizeMs: number;  // Allocation window duration
}

export interface UtilityFunction {
  readonly tenantId: string;
  readonly alpha: number;         // Business priority coefficient
  readonly baseNDCG: number;      // Baseline nDCG for tenant
  readonly marginalNDCG: (cpuMs: number, memMB: number) => number;
  readonly lastCalibrated: Date;
}

export interface SLAUtilityMetrics {
  readonly tenantId: string;
  readonly utilityScore: number;   // Current utility achievement
  readonly recallScore: number;    // SLA-Recall metric
  readonly coreScore: number;      // SLA-Core metric  
  readonly diversityScore: number; // SLA-Diversity metric
  readonly costEfficiency: number; // Utility per dollar
  readonly upshift: number;        // Performance improvement %
  readonly timestamp: Date;
}

export interface SpendGovernor {
  readonly tenantId: string;
  readonly dailyBudgetMs: number;
  readonly currentSpentMs: number;
  readonly budgetResetAt: Date;
  readonly isThrottled: boolean;
  readonly warningThreshold: number; // 0.8 = 80% of budget
}

export interface ShardCredit {
  readonly shardId: string;
  readonly tenantId: string;
  readonly creditsAvailable: number;
  readonly creditsUsed: number;
  readonly creditRate: number; // Credits per millisecond
  readonly lastUpdated: Date;
}

const DEFAULT_SYSTEM_RESOURCES: SystemResources = {
  maxCpuTimeMs: 10000, // 10 seconds per window
  maxMemoryGB: PRODUCTION_CONFIG.resources.memory_limit_gb,
  lambdaMs: 0.001,     // $0.001 per CPU millisecond
  lambdaGB: 0.1,       // $0.1 per GB memory
  windowSizeMs: 60000, // 1-minute allocation windows
};

const SLA_CLASSES = {
  premium: { priority: 1.0, guaranteedUpshift: 0.07 }, // 7% guaranteed improvement
  standard: { priority: 0.5, guaranteedUpshift: 0.05 }, // 5% guaranteed improvement  
  basic: { priority: 0.2, guaranteedUpshift: 0.03 },    // 3% guaranteed improvement
} as const;

export class TenantEconomicsEngine {
  private tenantProfiles = new Map<string, TenantProfile>();
  private utilityFunctions = new Map<string, UtilityFunction>();
  private currentAllocations = new Map<string, ResourceAllocation>();
  private slaMetrics = new Map<string, SLAUtilityMetrics>();
  private spendGovernors = new Map<string, SpendGovernor>();
  private shardCredits = new Map<string, ShardCredit>();
  private systemResources: SystemResources;
  private allocationWindow: { start: Date; end: Date } | null = null;

  constructor(systemResources: Partial<SystemResources> = {}) {
    this.systemResources = { ...DEFAULT_SYSTEM_RESOURCES, ...systemResources };
  }

  /**
   * Register a new tenant with business priority
   */
  async registerTenant(
    tenantId: string, 
    slaClass: 'premium' | 'standard' | 'basic',
    businessPriority?: number
  ): Promise<TenantProfile> {
    const span = LensTracer.createChildSpan('register_tenant');
    
    try {
      const alpha = businessPriority || SLA_CLASSES[slaClass].priority;
      
      const profile: TenantProfile = {
        id: tenantId,
        businessPriority: alpha,
        slaClass,
        historicalUsage: {
          avgLatencyMs: 0,
          avgMemoryMB: 0,
          avgNDCG: 0.5, // Start with baseline
          queryVolume24h: 0,
          utilityScore: 0,
          lastUpdated: new Date(),
        },
        currentAllocation: {
          tenantId,
          cpuTimeMs: 0,
          memoryMB: 0,
          priorityWeight: alpha,
          estimatedUtility: 0,
          allocatedAt: new Date(),
          expiresAt: new Date(Date.now() + this.systemResources.windowSizeMs),
        },
        createdAt: new Date(),
      };

      // Initialize utility function
      this.utilityFunctions.set(tenantId, {
        tenantId,
        alpha,
        baseNDCG: 0.5,
        marginalNDCG: (cpuMs, memMB) => this.defaultMarginalNDCG(cpuMs, memMB),
        lastCalibrated: new Date(),
      });

      // Initialize spend governor
      this.spendGovernors.set(tenantId, {
        tenantId,
        dailyBudgetMs: this.calculateDailyBudget(slaClass),
        currentSpentMs: 0,
        budgetResetAt: new Date(Date.now() + 24 * 60 * 60 * 1000),
        isThrottled: false,
        warningThreshold: 0.8,
      });

      this.tenantProfiles.set(tenantId, profile);

      span.setAttributes({
        success: true,
        tenant_id: tenantId,
        sla_class: slaClass,
        business_priority: alpha,
      });

      return profile;

    } catch (error) {
      span.recordException(error as Error);
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Solve convex optimization problem for resource allocation
   */
  async optimizeResourceAllocation(): Promise<Map<string, ResourceAllocation>> {
    const span = LensTracer.createChildSpan('optimize_resource_allocation');
    
    try {
      const tenants = Array.from(this.tenantProfiles.values());
      const M = this.systemResources.maxCpuTimeMs;
      const G = this.systemResources.maxMemoryGB * 1024; // Convert to MB
      
      // Solve convex program using Lagrangian dual decomposition
      const allocations = await this.solveConvexProgram(tenants, M, G);
      
      // Update current allocations
      for (const [tenantId, allocation] of allocations.entries()) {
        this.currentAllocations.set(tenantId, allocation);
      }

      // Start new allocation window
      this.allocationWindow = {
        start: new Date(),
        end: new Date(Date.now() + this.systemResources.windowSizeMs),
      };

      span.setAttributes({
        success: true,
        allocated_tenants: allocations.size,
        total_cpu_allocated: Array.from(allocations.values()).reduce((sum, a) => sum + a.cpuTimeMs, 0),
        total_memory_allocated: Array.from(allocations.values()).reduce((sum, a) => sum + a.memoryMB, 0),
      });

      return allocations;

    } catch (error) {
      span.recordException(error as Error);
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Check if tenant has sufficient resources for query
   */
  async checkResourceAvailability(tenantId: string, estimatedCostMs: number, estimatedMemoryMB: number): Promise<{
    allowed: boolean;
    allocation: ResourceAllocation | null;
    reason?: string;
  }> {
    const allocation = this.currentAllocations.get(tenantId);
    const governor = this.spendGovernors.get(tenantId);
    
    if (!allocation || !governor) {
      return { allowed: false, allocation: null, reason: 'TENANT_NOT_REGISTERED' };
    }

    // Check allocation window validity
    if (this.allocationWindow && new Date() > this.allocationWindow.end) {
      return { allowed: false, allocation: null, reason: 'ALLOCATION_WINDOW_EXPIRED' };
    }

    // Check spend governor
    if (governor.isThrottled || governor.currentSpentMs + estimatedCostMs > governor.dailyBudgetMs) {
      return { allowed: false, allocation: null, reason: 'BUDGET_EXCEEDED' };
    }

    // Check resource allocation
    if (estimatedCostMs > allocation.cpuTimeMs || estimatedMemoryMB > allocation.memoryMB) {
      return { allowed: false, allocation: null, reason: 'INSUFFICIENT_ALLOCATION' };
    }

    return { allowed: true, allocation };
  }

  /**
   * Consume resources for a query
   */
  async consumeResources(
    tenantId: string, 
    actualCostMs: number, 
    actualMemoryMB: number, 
    actualNDCG: number
  ): Promise<void> {
    const span = LensTracer.createChildSpan('consume_resources');
    
    try {
      // Update spend governor
      const governor = this.spendGovernors.get(tenantId);
      if (governor) {
        (governor as any).currentSpentMs += actualCostMs;
        if ((governor as any).currentSpentMs > governor.dailyBudgetMs * governor.warningThreshold) {
          (governor as any).isThrottled = (governor as any).currentSpentMs > governor.dailyBudgetMs;
        }
        this.spendGovernors.set(tenantId, governor);
      }

      // Update allocation
      const allocation = this.currentAllocations.get(tenantId);
      if (allocation) {
        const updatedAllocation: ResourceAllocation = {
          ...allocation,
          cpuTimeMs: Math.max(0, allocation.cpuTimeMs - actualCostMs),
          memoryMB: Math.max(0, allocation.memoryMB - actualMemoryMB),
        };
        this.currentAllocations.set(tenantId, updatedAllocation);
      }

      // Update shard credits
      await this.updateShardCredits(tenantId, actualCostMs);

      // Update SLA utility metrics
      await this.updateSLAMetrics(tenantId, actualCostMs, actualMemoryMB, actualNDCG);

      span.setAttributes({
        success: true,
        tenant_id: tenantId,
        cost_ms: actualCostMs,
        memory_mb: actualMemoryMB,
        ndcg: actualNDCG,
      });

    } catch (error) {
      span.recordException(error as Error);
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Get SLA utility metrics for a tenant
   */
  getSLAUtilityMetrics(tenantId: string): SLAUtilityMetrics | null {
    return this.slaMetrics.get(tenantId) || null;
  }

  /**
   * Get all SLA utility metrics for reporting
   */
  getAllSLAMetrics(): SLAUtilityMetrics[] {
    return Array.from(this.slaMetrics.values());
  }

  /**
   * Check if upshift is within target range [3%, 7%]
   */
  checkUpshiftCompliance(): { compliant: boolean; avgUpshift: number; outOfRange: string[] } {
    const metrics = Array.from(this.slaMetrics.values());
    const validMetrics = metrics.filter(m => !isNaN(m.upshift));
    
    if (validMetrics.length === 0) {
      return { compliant: true, avgUpshift: 0, outOfRange: [] };
    }

    const avgUpshift = validMetrics.reduce((sum, m) => sum + m.upshift, 0) / validMetrics.length;
    const outOfRange = validMetrics
      .filter(m => m.upshift < 0.03 || m.upshift > 0.07)
      .map(m => m.tenantId);

    return {
      compliant: outOfRange.length === 0,
      avgUpshift,
      outOfRange,
    };
  }

  /**
   * Reset daily budgets for all tenants
   */
  async resetDailyBudgets(): Promise<void> {
    const now = new Date();
    const tomorrow = new Date(now.getTime() + 24 * 60 * 60 * 1000);

    for (const [tenantId, governor] of this.spendGovernors.entries()) {
      if (now >= governor.budgetResetAt) {
        this.spendGovernors.set(tenantId, {
          ...governor,
          currentSpentMs: 0,
          budgetResetAt: tomorrow,
          isThrottled: false,
        });
      }
    }
  }

  /**
   * Solve convex optimization problem using dual decomposition
   */
  private async solveConvexProgram(
    tenants: TenantProfile[], 
    maxCpuMs: number, 
    maxMemoryMB: number
  ): Promise<Map<string, ResourceAllocation>> {
    const allocations = new Map<string, ResourceAllocation>();
    
    // Simplified dual decomposition with gradient ascent
    let lambdaCpu = this.systemResources.lambdaMs;
    let lambdaMemory = this.systemResources.lambdaGB / 1024; // Convert to per MB
    
    const maxIterations = 100;
    const learningRate = 0.01;
    
    for (let iter = 0; iter < maxIterations; iter++) {
      let totalCpuDemand = 0;
      let totalMemoryDemand = 0;
      
      // Solve each tenant's individual problem
      for (const tenant of tenants) {
        const utilityFn = this.utilityFunctions.get(tenant.id);
        if (!utilityFn) continue;
        
        // Solve: max uᵢ(xᵢ) = αᵢ·ΔnDCG(xᵢ) - λ_cpu·xᵢᵐˢ - λ_memory·xᵢᵐᵉᵐ
        const allocation = this.solveTenantOptimization(tenant, utilityFn, lambdaCpu, lambdaMemory);
        allocations.set(tenant.id, allocation);
        
        totalCpuDemand += allocation.cpuTimeMs;
        totalMemoryDemand += allocation.memoryMB;
      }
      
      // Update dual variables using gradient ascent
      const cpuViolation = totalCpuDemand - maxCpuMs;
      const memoryViolation = totalMemoryDemand - maxMemoryMB;
      
      lambdaCpu = Math.max(0, lambdaCpu + learningRate * cpuViolation);
      lambdaMemory = Math.max(0, lambdaMemory + learningRate * memoryViolation);
      
      // Check convergence
      if (Math.abs(cpuViolation) < 10 && Math.abs(memoryViolation) < 10) {
        break;
      }
    }
    
    return allocations;
  }

  /**
   * Solve individual tenant optimization problem
   */
  private solveTenantOptimization(
    tenant: TenantProfile,
    utilityFn: UtilityFunction,
    lambdaCpu: number,
    lambdaMemory: number
  ): ResourceAllocation {
    // Simplified analytical solution for quadratic utility function
    // In practice, this would use numerical optimization
    
    const alpha = utilityFn.alpha;
    const governor = this.spendGovernors.get(tenant.id);
    const maxBudget = governor ? Math.max(0, governor.dailyBudgetMs - governor.currentSpentMs) : 1000;
    
    // Optimal allocation considering marginal utility and dual prices
    const optimalCpu = Math.min(
      maxBudget,
      Math.max(0, (alpha * 100 - lambdaCpu) / (2 * lambdaCpu)) // Simplified quadratic
    );
    
    const optimalMemory = Math.min(
      200, // Max memory per tenant
      Math.max(0, (alpha * 50 - lambdaMemory) / (2 * lambdaMemory))
    );
    
    const estimatedUtility = this.calculateUtility(tenant.id, optimalCpu, optimalMemory);
    
    return {
      tenantId: tenant.id,
      cpuTimeMs: optimalCpu,
      memoryMB: optimalMemory,
      priorityWeight: alpha,
      estimatedUtility,
      allocatedAt: new Date(),
      expiresAt: new Date(Date.now() + this.systemResources.windowSizeMs),
    };
  }

  /**
   * Calculate utility for a tenant given resource allocation
   */
  private calculateUtility(tenantId: string, cpuMs: number, memoryMB: number): number {
    const utilityFn = this.utilityFunctions.get(tenantId);
    if (!utilityFn) return 0;
    
    const deltaNDCG = utilityFn.marginalNDCG(cpuMs, memoryMB);
    const cpuCost = this.systemResources.lambdaMs * cpuMs;
    const memoryCost = this.systemResources.lambdaGB * (memoryMB / 1024);
    
    return utilityFn.alpha * deltaNDCG - cpuCost - memoryCost;
  }

  /**
   * Default marginal nDCG function
   */
  private defaultMarginalNDCG(cpuMs: number, memoryMB: number): number {
    // Diminishing returns model: nDCG = base + log(1 + resources) * factor
    const resourceFactor = Math.log(1 + cpuMs / 100 + memoryMB / 50);
    return Math.min(0.8 * resourceFactor, 0.5); // Cap at 50% improvement
  }

  /**
   * Calculate daily budget based on SLA class
   */
  private calculateDailyBudget(slaClass: 'premium' | 'standard' | 'basic'): number {
    const baseBudget = 1000; // 1 second base budget
    const multipliers = { premium: 10, standard: 3, basic: 1 };
    return baseBudget * multipliers[slaClass];
  }

  /**
   * Update shard credits for tenant
   */
  private async updateShardCredits(tenantId: string, costMs: number): Promise<void> {
    // Simplified shard credit accounting
    const creditRate = 1.0; // 1 credit per millisecond
    const creditsUsed = costMs * creditRate;
    
    // Find or create shard credit entry
    const shardId = `shard_${tenantId}_${Date.now()}`;
    const existing = this.shardCredits.get(shardId) || {
      shardId,
      tenantId,
      creditsAvailable: 10000, // Default allocation
      creditsUsed: 0,
      creditRate,
      lastUpdated: new Date(),
    };
    
    this.shardCredits.set(shardId, {
      ...existing,
      creditsUsed: existing.creditsUsed + creditsUsed,
      creditsAvailable: Math.max(0, existing.creditsAvailable - creditsUsed),
      lastUpdated: new Date(),
    });
  }

  /**
   * Update SLA utility metrics
   */
  private async updateSLAMetrics(
    tenantId: string, 
    costMs: number, 
    memoryMB: number, 
    actualNDCG: number
  ): Promise<void> {
    const tenant = this.tenantProfiles.get(tenantId);
    if (!tenant) return;

    const existing = this.slaMetrics.get(tenantId);
    const baseline = tenant.historicalUsage.avgNDCG || 0.5;
    const upshift = baseline > 0 ? (actualNDCG - baseline) / baseline : 0;
    const costEfficiency = actualNDCG > 0 ? actualNDCG / (costMs * this.systemResources.lambdaMs) : 0;

    const metrics: SLAUtilityMetrics = {
      tenantId,
      utilityScore: this.calculateUtility(tenantId, costMs, memoryMB),
      recallScore: Math.min(actualNDCG / 0.8, 1.0), // Normalize to 80% target
      coreScore: actualNDCG, // Simplified
      diversityScore: 0.8, // Placeholder
      costEfficiency,
      upshift,
      timestamp: new Date(),
    };

    this.slaMetrics.set(tenantId, metrics);

    // Update tenant historical usage with exponential moving average
    const alpha = 0.1;
    const usage = tenant.historicalUsage;
    (tenant as any).historicalUsage = {
      avgLatencyMs: usage.avgLatencyMs * (1 - alpha) + costMs * alpha,
      avgMemoryMB: usage.avgMemoryMB * (1 - alpha) + memoryMB * alpha,
      avgNDCG: usage.avgNDCG * (1 - alpha) + actualNDCG * alpha,
      queryVolume24h: usage.queryVolume24h + 1,
      utilityScore: usage.utilityScore * (1 - alpha) + metrics.utilityScore * alpha,
      lastUpdated: new Date(),
    };
  }
}