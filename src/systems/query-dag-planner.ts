/**
 * Declarative Query-DAG Planner with DSL
 * 
 * Generalizes pipeline into query compiler with tiny DSL:
 * PLAN := LexScan(k₁) ▷ Struct(patterns, K₂) ▷ Slice(BFS≤2, K₃) ▷ ANN(risk, ef) ▷ Rerank(monotone)
 * 
 * Optimizes budgets by per-operator cost model learned from telemetry to maximize ΔnDCG/ms under SLO knapsack
 */

import type { SearchContext } from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';
import { PRODUCTION_CONFIG } from '../types/config.js';

// DSL Operators
export interface LexScanOp {
  readonly type: 'LexScan';
  readonly k: number;
  readonly fuzzyDistance: number;
  readonly subtokens: boolean;
}

export interface StructOp {
  readonly type: 'Struct';
  readonly patterns: readonly string[];
  readonly k: number;
  readonly astDepth: number;
}

export interface SliceOp {
  readonly type: 'Slice';
  readonly algorithm: 'BFS' | 'DFS' | 'Radius';
  readonly maxDepth: number;
  readonly k: number;
}

export interface ANNOp {
  readonly type: 'ANN';
  readonly riskThreshold: number;
  readonly ef: number;
  readonly dimensions: number;
}

export interface RerankOp {
  readonly type: 'Rerank';
  readonly model: 'monotone' | 'learned' | 'hybrid';
  readonly features: readonly string[];
  readonly alpha: number;
}

export type PlanOperator = LexScanOp | StructOp | SliceOp | ANNOp | RerankOp;

export interface QueryPlan {
  readonly id: string;
  readonly operators: readonly PlanOperator[];
  readonly estimatedCostMs: number;
  readonly estimatedNDCG: number;
  readonly sloConstraints: SLOConstraints;
  readonly createdAt: Date;
}

export interface SLOConstraints {
  readonly maxLatencyMs: number;
  readonly maxMemoryMB: number;
  readonly minRecall: number;
  readonly maxOperators: number;
}

export interface OperatorCostModel {
  readonly type: string;
  readonly baseCostMs: number;
  readonly scalingFactor: number;
  readonly memoryMB: number;
  readonly qualityGain: number;
  readonly lastUpdated: Date;
}

export interface PlanExecution {
  readonly planId: string;
  readonly actualCostMs: number;
  readonly actualNDCG: number;
  readonly operatorTimings: readonly { type: string; costMs: number }[];
  readonly memoryUsageMB: number;
  readonly success: boolean;
  readonly executedAt: Date;
}

export interface PlanCache {
  readonly key: string;
  readonly plan: QueryPlan;
  readonly hitCount: number;
  readonly lastAccess: Date;
  readonly avgNDCG: number;
}

const DEFAULT_SLO_CONSTRAINTS: SLOConstraints = {
  maxLatencyMs: PRODUCTION_CONFIG.performance.overall_p95_ms,
  maxMemoryMB: 100,
  minRecall: 0.8,
  maxOperators: 5,
};

const DEFAULT_COST_MODELS: OperatorCostModel[] = [
  {
    type: 'LexScan',
    baseCostMs: 2,
    scalingFactor: 0.01,
    memoryMB: 10,
    qualityGain: 0.6,
    lastUpdated: new Date(),
  },
  {
    type: 'Struct',
    baseCostMs: 3,
    scalingFactor: 0.02,
    memoryMB: 15,
    qualityGain: 0.3,
    lastUpdated: new Date(),
  },
  {
    type: 'Slice',
    baseCostMs: 1,
    scalingFactor: 0.005,
    memoryMB: 5,
    qualityGain: 0.1,
    lastUpdated: new Date(),
  },
  {
    type: 'ANN',
    baseCostMs: 5,
    scalingFactor: 0.05,
    memoryMB: 50,
    qualityGain: 0.4,
    lastUpdated: new Date(),
  },
  {
    type: 'Rerank',
    baseCostMs: 4,
    scalingFactor: 0.03,
    memoryMB: 20,
    qualityGain: 0.5,
    lastUpdated: new Date(),
  },
];

export class QueryDAGPlanner {
  private costModels = new Map<string, OperatorCostModel>();
  private planCache = new Map<string, PlanCache>();
  private executionHistory: PlanExecution[] = [];
  private plannerCostMs = 0;
  private plannerSpendRatio = 0;

  constructor() {
    // Initialize cost models
    for (const model of DEFAULT_COST_MODELS) {
      this.costModels.set(model.type, model);
    }
  }

  /**
   * Generate optimal query plan using DSL and cost model
   */
  async generatePlan(
    ctx: SearchContext,
    sloConstraints: Partial<SLOConstraints> = {}
  ): Promise<QueryPlan> {
    const span = LensTracer.createChildSpan('generate_query_plan');
    const planStart = Date.now();

    try {
      const constraints: SLOConstraints = { ...DEFAULT_SLO_CONSTRAINTS, ...sloConstraints };
      
      // Check cache first
      const cacheKey = this.generateCacheKey(ctx, constraints);
      const cachedPlan = this.planCache.get(cacheKey);
      
      if (cachedPlan) {
        (cachedPlan as any).hitCount++;
        (cachedPlan as any).lastAccess = new Date();
        this.planCache.set(cacheKey, cachedPlan);
        
        span.setAttributes({ cached: true, plan_id: cachedPlan.plan.id });
        return cachedPlan.plan;
      }

      // Generate new plan using knapsack optimization
      const plan = await this.optimizePlan(ctx, constraints);
      
      // Cache the plan
      this.planCache.set(cacheKey, {
        key: cacheKey,
        plan,
        hitCount: 1,
        lastAccess: new Date(),
        avgNDCG: plan.estimatedNDCG,
      });

      // Track planner cost
      this.plannerCostMs = Date.now() - planStart;
      this.updatePlannerSpendRatio();

      span.setAttributes({
        cached: false,
        plan_id: plan.id,
        estimated_cost_ms: plan.estimatedCostMs,
        estimated_ndcg: plan.estimatedNDCG,
        planner_cost_ms: this.plannerCostMs,
      });

      return plan;

    } catch (error) {
      span.recordException(error as Error);
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Execute a query plan
   */
  async executePlan(plan: QueryPlan, ctx: SearchContext): Promise<PlanExecution> {
    const span = LensTracer.createChildSpan('execute_query_plan');
    const executionStart = Date.now();
    const operatorTimings: { type: string; costMs: number }[] = [];
    let memoryUsage = 0;
    let success = true;

    try {
      // Execute operators in sequence
      for (const operator of plan.operators) {
        const opStart = Date.now();
        
        try {
          await this.executeOperator(operator, ctx);
          const opCost = Date.now() - opStart;
          operatorTimings.push({ type: operator.type, costMs: opCost });
          
          // Track memory usage
          const opMemory = this.costModels.get(operator.type)?.memoryMB || 0;
          memoryUsage += opMemory;
          
        } catch (error) {
          success = false;
          console.warn(`Operator ${operator.type} failed: ${error}`);
          break;
        }
      }

      const execution: PlanExecution = {
        planId: plan.id,
        actualCostMs: Date.now() - executionStart,
        actualNDCG: success ? this.estimateActualNDCG(operatorTimings) : 0,
        operatorTimings,
        memoryUsageMB: memoryUsage,
        success,
        executedAt: new Date(),
      };

      // Update execution history
      this.executionHistory.push(execution);
      if (this.executionHistory.length > 1000) {
        this.executionHistory = this.executionHistory.slice(-500);
      }

      // Learn from execution to update cost models
      await this.updateCostModels(execution);

      span.setAttributes({
        success,
        actual_cost_ms: execution.actualCostMs,
        actual_ndcg: execution.actualNDCG,
        memory_usage_mb: execution.memoryUsageMB,
        operators_executed: operatorTimings.length,
      });

      return execution;

    } catch (error) {
      span.recordException(error as Error);
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Get plan by policy delta for reproducible rankings
   */
  getPlanByPolicy(policyDelta: string): QueryPlan | null {
    // Parse policy delta format: "lexscan:k=100,struct:patterns=class|function,rerank:model=monotone"
    try {
      const operators = this.parsePolicyDelta(policyDelta);
      
      return {
        id: `policy_${policyDelta.replace(/[^a-zA-Z0-9]/g, '_')}`,
        operators,
        estimatedCostMs: this.estimatePlanCost(operators),
        estimatedNDCG: this.estimatePlanNDCG(operators),
        sloConstraints: DEFAULT_SLO_CONSTRAINTS,
        createdAt: new Date(),
      };
    } catch (error) {
      console.warn(`Failed to parse policy delta: ${policyDelta}`);
      return null;
    }
  }

  /**
   * Get planner performance metrics
   */
  getPlannerMetrics(): {
    spendRatio: number;
    avgPlannerCostMs: number;
    cacheHitRatio: number;
    p99CostMs: number;
    p95CostMs: number;
  } {
    const recentExecutions = this.executionHistory.slice(-100);
    const plannerCosts = recentExecutions.map(e => this.plannerCostMs).filter(c => c > 0);
    
    const totalHits = Array.from(this.planCache.values()).reduce((sum, cache) => sum + cache.hitCount, 0);
    const totalRequests = totalHits + this.planCache.size;
    
    const sortedCosts = plannerCosts.sort((a, b) => a - b);
    const p95Index = Math.floor(sortedCosts.length * 0.95);
    const p99Index = Math.floor(sortedCosts.length * 0.99);

    return {
      spendRatio: this.plannerSpendRatio,
      avgPlannerCostMs: plannerCosts.length > 0 ? plannerCosts.reduce((s, c) => s + c, 0) / plannerCosts.length : 0,
      cacheHitRatio: totalRequests > 0 ? (totalHits - this.planCache.size) / totalRequests : 0,
      p99CostMs: sortedCosts[p99Index] || 0,
      p95CostMs: sortedCosts[p95Index] || 0,
    };
  }

  /**
   * Clear plan cache
   */
  clearCache(): void {
    this.planCache.clear();
  }

  /**
   * Optimize plan using knapsack algorithm
   */
  private async optimizePlan(ctx: SearchContext, constraints: SLOConstraints): Promise<QueryPlan> {
    const candidateOperators = this.generateCandidateOperators(ctx);
    
    // Dynamic programming knapsack for cost/quality optimization
    const dp = new Map<string, { cost: number; quality: number; operators: PlanOperator[] }>();
    dp.set('', { cost: 0, quality: 0, operators: [] });
    
    for (const operator of candidateOperators) {
      const newDp = new Map(dp);
      
      for (const [key, state] of dp.entries()) {
        const newCost = state.cost + this.estimateOperatorCost(operator);
        const newQuality = state.quality + this.estimateOperatorQuality(operator);
        const newOperators = [...state.operators, operator];
        const newKey = `${key}:${operator.type}`;
        
        // Check constraints
        if (newCost <= constraints.maxLatencyMs && 
            newOperators.length <= constraints.maxOperators) {
          
          const existing = newDp.get(newKey);
          if (!existing || newQuality > existing.quality) {
            newDp.set(newKey, {
              cost: newCost,
              quality: newQuality,
              operators: newOperators,
            });
          }
        }
      }
      
      dp.clear();
      for (const [key, value] of newDp.entries()) {
        dp.set(key, value);
      }
    }
    
    // Find optimal solution
    let bestState = { cost: 0, quality: 0, operators: [] as PlanOperator[] };
    for (const state of dp.values()) {
      if (state.quality > bestState.quality) {
        bestState = state;
      }
    }
    
    // Ensure minimum recall constraint
    if (bestState.operators.length === 0) {
      // Fallback plan with minimal operators
      bestState.operators = [
        { type: 'LexScan', k: ctx.k, fuzzyDistance: ctx.fuzzy_distance || 0, subtokens: true },
      ];
      bestState.cost = this.estimateOperatorCost(bestState.operators[0]);
      bestState.quality = this.estimateOperatorQuality(bestState.operators[0]);
    }
    
    return {
      id: `plan_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      operators: bestState.operators,
      estimatedCostMs: bestState.cost,
      estimatedNDCG: bestState.quality,
      sloConstraints: constraints,
      createdAt: new Date(),
    };
  }

  /**
   * Generate candidate operators based on query context
   */
  private generateCandidateOperators(ctx: SearchContext): PlanOperator[] {
    const candidates: PlanOperator[] = [];
    
    // Always include LexScan as base
    candidates.push({
      type: 'LexScan',
      k: Math.min(ctx.k * 4, 200),
      fuzzyDistance: ctx.fuzzy_distance || 0,
      subtokens: true,
    });
    
    // Add Struct for structural queries
    if (ctx.mode === 'struct' || ctx.mode === 'hybrid') {
      candidates.push({
        type: 'Struct',
        patterns: this.extractStructuralPatterns(ctx.query),
        k: Math.min(ctx.k * 2, 100),
        astDepth: 3,
      });
    }
    
    // Add Slice for context expansion
    if (ctx.query.length > 10) {
      candidates.push({
        type: 'Slice',
        algorithm: 'BFS',
        maxDepth: 2,
        k: Math.min(ctx.k * 3, 150),
      });
    }
    
    // Add ANN for semantic search
    if (ctx.k > 10) {
      candidates.push({
        type: 'ANN',
        riskThreshold: 0.1,
        ef: 128,
        dimensions: 768,
      });
    }
    
    // Add Rerank for quality improvement
    candidates.push({
      type: 'Rerank',
      model: 'monotone',
      features: ['lexical', 'structural', 'semantic'],
      alpha: 0.5,
    });
    
    return candidates;
  }

  /**
   * Execute a single operator
   */
  private async executeOperator(operator: PlanOperator, ctx: SearchContext): Promise<void> {
    const span = LensTracer.createChildSpan(`execute_${operator.type.toLowerCase()}`);
    
    try {
      switch (operator.type) {
        case 'LexScan':
          await this.executeLexScan(operator, ctx);
          break;
        case 'Struct':
          await this.executeStruct(operator, ctx);
          break;
        case 'Slice':
          await this.executeSlice(operator, ctx);
          break;
        case 'ANN':
          await this.executeANN(operator, ctx);
          break;
        case 'Rerank':
          await this.executeRerank(operator, ctx);
          break;
        default:
          throw new Error(`Unknown operator type: ${(operator as any).type}`);
      }
      
      span.setAttributes({ success: true });
      
    } catch (error) {
      span.recordException(error as Error);
      throw error;
    } finally {
      span.end();
    }
  }

  // Operator execution methods (simplified implementations)
  private async executeLexScan(operator: LexScanOp, ctx: SearchContext): Promise<void> {
    // Simulate lexical scan execution
    await new Promise(resolve => setTimeout(resolve, 2));
  }

  private async executeStruct(operator: StructOp, ctx: SearchContext): Promise<void> {
    // Simulate structural search execution
    await new Promise(resolve => setTimeout(resolve, 3));
  }

  private async executeSlice(operator: SliceOp, ctx: SearchContext): Promise<void> {
    // Simulate slice operation execution
    await new Promise(resolve => setTimeout(resolve, 1));
  }

  private async executeANN(operator: ANNOp, ctx: SearchContext): Promise<void> {
    // Simulate ANN search execution
    await new Promise(resolve => setTimeout(resolve, 5));
  }

  private async executeRerank(operator: RerankOp, ctx: SearchContext): Promise<void> {
    // Simulate reranking execution
    await new Promise(resolve => setTimeout(resolve, 4));
  }

  /**
   * Estimate cost of an operator
   */
  private estimateOperatorCost(operator: PlanOperator): number {
    const model = this.costModels.get(operator.type);
    if (!model) return 10; // Default fallback
    
    let scalingInput = 1;
    if ('k' in operator && typeof operator.k === 'number') {
      scalingInput = operator.k;
    }
    
    return model.baseCostMs + (model.scalingFactor * scalingInput);
  }

  /**
   * Estimate quality gain of an operator
   */
  private estimateOperatorQuality(operator: PlanOperator): number {
    const model = this.costModels.get(operator.type);
    return model?.qualityGain || 0.1;
  }

  /**
   * Estimate total plan cost
   */
  private estimatePlanCost(operators: readonly PlanOperator[]): number {
    return operators.reduce((sum, op) => sum + this.estimateOperatorCost(op), 0);
  }

  /**
   * Estimate total plan nDCG
   */
  private estimatePlanNDCG(operators: readonly PlanOperator[]): number {
    // Diminishing returns model for quality combination
    let quality = 0;
    for (const operator of operators) {
      const opQuality = this.estimateOperatorQuality(operator);
      quality = quality + opQuality * (1 - quality); // Diminishing returns
    }
    return Math.min(quality, 1.0);
  }

  /**
   * Update cost models based on execution feedback
   */
  private async updateCostModels(execution: PlanExecution): Promise<void> {
    for (const timing of execution.operatorTimings) {
      const existing = this.costModels.get(timing.type);
      if (existing) {
        // Simple exponential moving average update
        const alpha = 0.1;
        const newBaseCost = existing.baseCostMs * (1 - alpha) + timing.costMs * alpha;
        
        this.costModels.set(timing.type, {
          ...existing,
          baseCostMs: newBaseCost,
          lastUpdated: new Date(),
        });
      }
    }
  }

  /**
   * Generate cache key for plan caching
   */
  private generateCacheKey(ctx: SearchContext, constraints: SLOConstraints): string {
    return `${ctx.query}:${ctx.mode}:${ctx.k}:${ctx.fuzzy_distance}:${constraints.maxLatencyMs}:${constraints.minRecall}`;
  }

  /**
   * Update planner spend ratio
   */
  private updatePlannerSpendRatio(): void {
    const recentExecutions = this.executionHistory.slice(-50);
    const totalExecutionTime = recentExecutions.reduce((sum, e) => sum + e.actualCostMs, 0);
    const totalPlannerTime = recentExecutions.length * this.plannerCostMs;
    
    this.plannerSpendRatio = totalExecutionTime > 0 ? totalPlannerTime / (totalExecutionTime + totalPlannerTime) : 0;
  }

  /**
   * Parse policy delta string into operators
   */
  private parsePolicyDelta(policyDelta: string): PlanOperator[] {
    const operators: PlanOperator[] = [];
    const parts = policyDelta.split(',');
    
    for (const part of parts) {
      const [opType, params] = part.split(':');
      const paramMap = new Map<string, string>();
      
      if (params) {
        const paramPairs = params.split(',');
        for (const pair of paramPairs) {
          const [key, value] = pair.split('=');
          if (key && value) {
            paramMap.set(key.trim(), value.trim());
          }
        }
      }
      
      switch (opType.trim().toLowerCase()) {
        case 'lexscan':
          operators.push({
            type: 'LexScan',
            k: parseInt(paramMap.get('k') || '100', 10),
            fuzzyDistance: parseFloat(paramMap.get('fuzzy') || '0'),
            subtokens: paramMap.get('subtokens') === 'true',
          });
          break;
        case 'struct':
          operators.push({
            type: 'Struct',
            patterns: (paramMap.get('patterns') || 'class,function').split('|'),
            k: parseInt(paramMap.get('k') || '50', 10),
            astDepth: parseInt(paramMap.get('depth') || '3', 10),
          });
          break;
        case 'rerank':
          operators.push({
            type: 'Rerank',
            model: (paramMap.get('model') || 'monotone') as 'monotone' | 'learned' | 'hybrid',
            features: (paramMap.get('features') || 'lexical,structural').split('|'),
            alpha: parseFloat(paramMap.get('alpha') || '0.5'),
          });
          break;
      }
    }
    
    return operators;
  }

  /**
   * Extract structural patterns from query
   */
  private extractStructuralPatterns(query: string): string[] {
    const patterns: string[] = [];
    
    // Common code patterns
    if (/class|struct|interface/i.test(query)) patterns.push('class');
    if (/function|method|def/i.test(query)) patterns.push('function');
    if (/import|require|include/i.test(query)) patterns.push('import');
    if (/variable|var|let|const/i.test(query)) patterns.push('variable');
    
    return patterns.length > 0 ? patterns : ['class', 'function'];
  }

  /**
   * Estimate actual nDCG from execution timings
   */
  private estimateActualNDCG(operatorTimings: readonly { type: string; costMs: number }[]): number {
    // Simple heuristic: higher cost usually means more processing and potentially better quality
    const totalCost = operatorTimings.reduce((sum, t) => sum + t.costMs, 0);
    const maxExpectedCost = 50; // ms
    
    return Math.min(totalCost / maxExpectedCost, 1.0) * 0.8 + 0.2; // Base quality of 0.2
  }
}