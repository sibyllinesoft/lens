/**
 * Query Compiler - Cost-Based Optimization System
 * 
 * Treats Lens plans as algebra: PLAN = (Scan ∥ Struct ∥ Symbol ∥ ANN_risk) ▷ Slice ▷ Rerank
 * Learns cost model (ms, candidate gain) per operator on live telemetry.
 * Runs optimizer picking plan and budgets (efSearch, slice K, BFS depth) to maximize ΔnDCG/ms under SLO knapsack.
 * Guardrails: never violate span invariants, floors on exact/struct still apply.
 * Start with two plans (symbol-first vs struct-first) using existing multi-plan scaffolding.
 * 
 * Gate: publish Q-cost curve, require p99 -8–12% at flat SLA-Recall@50
 */

import { LensTracer } from '../telemetry/tracer.js';
import type { SearchHit, SymbolCandidate, MatchReason } from './span_resolver/types.js';

export interface QueryPlan {
  id: string;
  name: string;
  operators: PlanOperator[];
  estimated_cost_ms: number;
  estimated_ndcg_gain: number;
  estimated_recall_at_50: number;
  constraints: PlanConstraints;
}

export interface PlanOperator {
  type: 'scan' | 'struct' | 'symbol' | 'ann_risk' | 'slice' | 'rerank';
  name: string;
  parameters: OperatorParameters;
  estimated_cost_ms: number;
  estimated_candidates: number;
  estimated_precision: number;
  dependencies: string[];        // Operator IDs this depends on
  parallel_eligible: boolean;    // Can run in parallel with siblings
}

export interface OperatorParameters {
  // Scan parameters
  fuzzy_distance?: number;       // 0-2 for fuzzy matching
  trigram_threshold?: number;    // Trigram similarity threshold
  
  // Struct/Symbol parameters
  max_ast_depth?: number;        // BFS depth limit
  symbol_filters?: string[];     // Symbol kind filters
  
  // ANN parameters
  ef_search?: number;            // HNSW search parameter
  num_candidates?: number;       // Vector search candidates
  
  // Slice parameters
  slice_k?: number;              // Top-K to keep
  score_threshold?: number;      // Minimum score threshold
  
  // Rerank parameters
  rerank_model?: string;         // ColBERT-v2, SPLADE-v2, etc.
  rerank_top_k?: number;         // How many to rerank
}

export interface PlanConstraints {
  max_total_time_ms: number;     // SLO constraint
  min_recall_at_50: number;      // Quality floor
  span_invariants: boolean;      // Must preserve span accuracy
  exact_struct_floors: {         // Minimum candidates from exact/struct
    exact_min: number;
    struct_min: number;
  };
}

export interface CostModel {
  operator_costs: Map<string, OperatorCostModel>;
  learned_at: Date;
  confidence: number;            // 0.0-1.0
  sample_count: number;
  last_updated: Date;
}

export interface OperatorCostModel {
  operator_type: string;
  base_cost_ms: number;          // Fixed cost
  cost_per_candidate: number;    // Variable cost per input candidate
  cost_per_parameter: Map<string, number>; // Parameter-specific costs
  candidate_selectivity: number; // Output/Input candidate ratio
  quality_impact: number;        // Impact on nDCG
  confidence_intervals: {
    cost_lower: number;
    cost_upper: number;
    selectivity_lower: number;
    selectivity_upper: number;
  };
}

export interface QueryCompilerConfig {
  max_plans_to_consider: number;
  optimization_timeout_ms: number;
  telemetry_sample_rate: number;     // 0.0-1.0
  cost_model_update_frequency: number; // Updates per hour
  performance_targets: {
    p99_improvement_percent: number;   // -8 to -12%
    maintain_recall_at_50: boolean;    // Flat SLA-Recall@50
  };
  plan_templates: {
    symbol_first: boolean;
    struct_first: boolean;
    semantic_first: boolean;         // Future extension
  };
}

export interface ExecutionMetrics {
  plan_id: string;
  total_time_ms: number;
  operator_times: Map<string, number>;
  candidates_per_stage: Map<string, number>;
  final_recall_at_50: number;
  final_ndcg: number;
  span_accuracy: number;
  timestamp: Date;
}

export class QueryCompiler {
  private costModel: CostModel;
  private planCache = new Map<string, QueryPlan>();
  private executionHistory: ExecutionMetrics[] = [];
  private telemetryBuffer: ExecutionMetrics[] = [];

  constructor(
    private config: QueryCompilerConfig = {
      max_plans_to_consider: 10,
      optimization_timeout_ms: 50,  // Very tight budget for real-time
      telemetry_sample_rate: 0.1,
      cost_model_update_frequency: 24, // Once per hour
      performance_targets: {
        p99_improvement_percent: -10,  // Target 10% improvement
        maintain_recall_at_50: true,
      },
      plan_templates: {
        symbol_first: true,
        struct_first: true,
        semantic_first: false, // Future
      }
    }
  ) {
    this.costModel = this.initializeBaseCostModel();
  }

  /**
   * Compile and optimize query plan for given query and constraints
   */
  async compileQuery(
    query: string,
    constraints: PlanConstraints,
    queryContext?: any
  ): Promise<QueryPlan> {
    const span = LensTracer.createChildSpan('compile_query', {
      'query': query,
      'constraints.max_time_ms': constraints.max_total_time_ms,
      'constraints.min_recall': constraints.min_recall_at_50,
    });

    const startTime = performance.now();

    try {
      // Generate candidate plans
      const candidatePlans = await this.generateCandidatePlans(query, constraints);
      
      // Optimize plan selection
      const optimizedPlan = await this.optimizePlanSelection(
        candidatePlans, 
        constraints,
        this.config.optimization_timeout_ms
      );

      // Validate constraints
      this.validatePlanConstraints(optimizedPlan, constraints);

      const compilationTime = performance.now() - startTime;
      
      span.setAttributes({
        'plan.id': optimizedPlan.id,
        'plan.name': optimizedPlan.name,
        'plan.estimated_cost_ms': optimizedPlan.estimated_cost_ms,
        'plan.estimated_ndcg': optimizedPlan.estimated_ndcg_gain,
        'candidates.considered': candidatePlans.length,
        'compilation.time_ms': compilationTime,
        success: true
      });

      return optimizedPlan;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Execute optimized query plan and collect telemetry
   */
  async executePlan(
    plan: QueryPlan,
    query: string,
    candidates?: SymbolCandidate[]
  ): Promise<{ hits: SearchHit[], metrics: ExecutionMetrics }> {
    const span = LensTracer.createChildSpan('execute_plan', {
      'plan.id': plan.id,
      'plan.operators': plan.operators.length,
      'query': query,
    });

    const startTime = performance.now();
    const operatorTimes = new Map<string, number>();
    const candidatesPerStage = new Map<string, number>();

    try {
      let currentCandidates = candidates || [];
      let finalHits: SearchHit[] = [];

      // Execute operators in plan order
      for (let i = 0; i < plan.operators.length; i++) {
        const operator = plan.operators[i];
        const operatorStart = performance.now();

        // Execute operator based on type
        const result = await this.executeOperator(
          operator, 
          currentCandidates, 
          query
        );

        currentCandidates = result.candidates;
        if (result.hits) {
          finalHits = result.hits;
        }

        const operatorTime = performance.now() - operatorStart;
        operatorTimes.set(operator.name, operatorTime);
        candidatesPerStage.set(operator.name, currentCandidates.length);

        // Check if we're exceeding time budget
        if (performance.now() - startTime > plan.constraints.max_total_time_ms * 1.2) {
          console.warn(`Plan ${plan.id} exceeding time budget, may need reoptimization`);
        }
      }

      const totalTime = performance.now() - startTime;

      // Calculate final metrics
      const metrics: ExecutionMetrics = {
        plan_id: plan.id,
        total_time_ms: totalTime,
        operator_times: operatorTimes,
        candidates_per_stage: candidatesPerStage,
        final_recall_at_50: await this.calculateRecallAt50(finalHits, query),
        final_ndcg: await this.calculateNDCG(finalHits, query),
        span_accuracy: this.calculateSpanAccuracy(finalHits),
        timestamp: new Date()
      };

      // Sample for telemetry
      if (Math.random() < this.config.telemetry_sample_rate) {
        this.telemetryBuffer.push(metrics);
        
        // Trigger cost model update if buffer is full
        if (this.telemetryBuffer.length >= 100) {
          await this.updateCostModel();
        }
      }

      span.setAttributes({
        'execution.time_ms': totalTime,
        'execution.within_budget': totalTime <= plan.estimated_cost_ms * 1.1,
        'hits.count': finalHits.length,
        'recall.at_50': metrics.final_recall_at_50,
        'span.accuracy': metrics.span_accuracy,
        success: true
      });

      return { hits: finalHits, metrics };

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Generate candidate query plans using templates
   */
  private async generateCandidatePlans(
    query: string,
    constraints: PlanConstraints
  ): Promise<QueryPlan[]> {
    const plans: QueryPlan[] = [];

    // Symbol-first plan
    if (this.config.plan_templates.symbol_first) {
      plans.push(await this.generateSymbolFirstPlan(query, constraints));
    }

    // Struct-first plan
    if (this.config.plan_templates.struct_first) {
      plans.push(await this.generateStructFirstPlan(query, constraints));
    }

    // Future: semantic-first plan
    if (this.config.plan_templates.semantic_first) {
      plans.push(await this.generateSemanticFirstPlan(query, constraints));
    }

    return plans.slice(0, this.config.max_plans_to_consider);
  }

  /**
   * Generate symbol-first query plan
   */
  private async generateSymbolFirstPlan(
    query: string,
    constraints: PlanConstraints
  ): Promise<QueryPlan> {
    const operators: PlanOperator[] = [
      {
        type: 'symbol',
        name: 'symbol_scan',
        parameters: {
          max_ast_depth: 5,
          symbol_filters: ['function', 'class', 'variable']
        },
        estimated_cost_ms: this.estimateOperatorCost('symbol', {}),
        estimated_candidates: 200,
        estimated_precision: 0.8,
        dependencies: [],
        parallel_eligible: true
      },
      {
        type: 'struct',
        name: 'structural_filter',
        parameters: {
          max_ast_depth: 3
        },
        estimated_cost_ms: this.estimateOperatorCost('struct', {}),
        estimated_candidates: 150,
        estimated_precision: 0.85,
        dependencies: ['symbol_scan'],
        parallel_eligible: false
      },
      {
        type: 'slice',
        name: 'top_k_slice',
        parameters: {
          slice_k: Math.min(100, constraints.exact_struct_floors.struct_min * 2)
        },
        estimated_cost_ms: this.estimateOperatorCost('slice', {}),
        estimated_candidates: 100,
        estimated_precision: 0.85,
        dependencies: ['structural_filter'],
        parallel_eligible: false
      },
      {
        type: 'rerank',
        name: 'semantic_rerank',
        parameters: {
          rerank_model: 'colbert_v2',
          rerank_top_k: 50
        },
        estimated_cost_ms: this.estimateOperatorCost('rerank', {}),
        estimated_candidates: 50,
        estimated_precision: 0.9,
        dependencies: ['top_k_slice'],
        parallel_eligible: false
      }
    ];

    const totalCost = operators.reduce((sum, op) => sum + op.estimated_cost_ms, 0);

    return {
      id: `symbol_first_${Date.now()}`,
      name: 'Symbol-First Plan',
      operators,
      estimated_cost_ms: totalCost,
      estimated_ndcg_gain: 0.85,
      estimated_recall_at_50: 0.82,
      constraints
    };
  }

  /**
   * Generate struct-first query plan
   */
  private async generateStructFirstPlan(
    query: string,
    constraints: PlanConstraints
  ): Promise<QueryPlan> {
    const operators: PlanOperator[] = [
      {
        type: 'struct',
        name: 'structural_scan',
        parameters: {
          max_ast_depth: 4
        },
        estimated_cost_ms: this.estimateOperatorCost('struct', {}),
        estimated_candidates: 300,
        estimated_precision: 0.75,
        dependencies: [],
        parallel_eligible: true
      },
      {
        type: 'symbol',
        name: 'symbol_refine',
        parameters: {
          max_ast_depth: 3,
          symbol_filters: ['function', 'class']
        },
        estimated_cost_ms: this.estimateOperatorCost('symbol', {}),
        estimated_candidates: 200,
        estimated_precision: 0.8,
        dependencies: ['structural_scan'],
        parallel_eligible: false
      },
      {
        type: 'slice',
        name: 'confidence_slice',
        parameters: {
          slice_k: 80,
          score_threshold: 0.3
        },
        estimated_cost_ms: this.estimateOperatorCost('slice', {}),
        estimated_candidates: 80,
        estimated_precision: 0.85,
        dependencies: ['symbol_refine'],
        parallel_eligible: false
      },
      {
        type: 'rerank',
        name: 'semantic_rerank',
        parameters: {
          rerank_model: 'colbert_v2',
          rerank_top_k: 50
        },
        estimated_cost_ms: this.estimateOperatorCost('rerank', {}),
        estimated_candidates: 50,
        estimated_precision: 0.9,
        dependencies: ['confidence_slice'],
        parallel_eligible: false
      }
    ];

    const totalCost = operators.reduce((sum, op) => sum + op.estimated_cost_ms, 0);

    return {
      id: `struct_first_${Date.now()}`,
      name: 'Struct-First Plan',
      operators,
      estimated_cost_ms: totalCost,
      estimated_ndcg_gain: 0.83,
      estimated_recall_at_50: 0.84,
      constraints
    };
  }

  /**
   * Future: Generate semantic-first query plan
   */
  private async generateSemanticFirstPlan(
    query: string,
    constraints: PlanConstraints
  ): Promise<QueryPlan> {
    // Placeholder for future semantic-first optimization
    return await this.generateSymbolFirstPlan(query, constraints);
  }

  /**
   * Optimize plan selection to maximize ΔnDCG/ms under SLO constraints
   */
  private async optimizePlanSelection(
    plans: QueryPlan[],
    constraints: PlanConstraints,
    timeoutMs: number
  ): Promise<QueryPlan> {
    const startTime = performance.now();

    // Sort plans by estimated efficiency (ΔnDCG/ms ratio)
    const rankedPlans = plans
      .filter(plan => plan.estimated_cost_ms <= constraints.max_total_time_ms)
      .filter(plan => plan.estimated_recall_at_50 >= constraints.min_recall_at_50)
      .sort((a, b) => {
        const efficiencyA = a.estimated_ndcg_gain / a.estimated_cost_ms;
        const efficiencyB = b.estimated_ndcg_gain / b.estimated_cost_ms;
        return efficiencyB - efficiencyA;
      });

    if (rankedPlans.length === 0) {
      throw new Error('No plans satisfy constraints');
    }

    // For now, return the most efficient plan
    // Future: implement more sophisticated knapsack optimization
    const selectedPlan = rankedPlans[0];

    // Fine-tune operator parameters within time budget
    await this.tuneOperatorParameters(selectedPlan, constraints, timeoutMs);

    return selectedPlan;
  }

  /**
   * Fine-tune operator parameters to maximize efficiency
   */
  private async tuneOperatorParameters(
    plan: QueryPlan,
    constraints: PlanConstraints,
    remainingTimeMs: number
  ): Promise<void> {
    // Implement parameter tuning - adjust efSearch, slice K, BFS depth
    // within the remaining optimization budget
    
    for (const operator of plan.operators) {
      if (operator.type === 'ann_risk' && operator.parameters.ef_search) {
        // Tune efSearch parameter
        const currentEf = operator.parameters.ef_search;
        const costModel = this.costModel.operator_costs.get('ann_risk');
        
        if (costModel) {
          // Simple heuristic: increase efSearch if we have budget
          const additionalCost = costModel.cost_per_parameter.get('ef_search') || 0;
          if (plan.estimated_cost_ms + additionalCost < constraints.max_total_time_ms * 0.9) {
            operator.parameters.ef_search = Math.min(currentEf * 1.2, 100);
          }
        }
      }
    }
  }

  /**
   * Execute individual operator
   */
  private async executeOperator(
    operator: PlanOperator,
    inputCandidates: SymbolCandidate[],
    query: string
  ): Promise<{ candidates: SymbolCandidate[], hits?: SearchHit[] }> {
    // Placeholder implementations - would integrate with existing span resolver
    switch (operator.type) {
      case 'symbol':
        return this.executeSymbolOperator(operator, inputCandidates, query);
      case 'struct':
        return this.executeStructOperator(operator, inputCandidates, query);
      case 'slice':
        return this.executeSliceOperator(operator, inputCandidates);
      case 'rerank':
        return this.executeRerankOperator(operator, inputCandidates, query);
      default:
        throw new Error(`Unknown operator type: ${operator.type}`);
    }
  }

  private async executeSymbolOperator(
    operator: PlanOperator,
    inputCandidates: SymbolCandidate[],
    query: string
  ): Promise<{ candidates: SymbolCandidate[] }> {
    // Placeholder for symbol operator execution
    return { candidates: inputCandidates };
  }

  private async executeStructOperator(
    operator: PlanOperator,
    inputCandidates: SymbolCandidate[],
    query: string
  ): Promise<{ candidates: SymbolCandidate[] }> {
    // Placeholder for structural operator execution
    return { candidates: inputCandidates };
  }

  private async executeSliceOperator(
    operator: PlanOperator,
    inputCandidates: SymbolCandidate[]
  ): Promise<{ candidates: SymbolCandidate[] }> {
    const k = operator.parameters.slice_k || 100;
    const threshold = operator.parameters.score_threshold || 0;
    
    const sliced = inputCandidates
      .filter(c => c.score >= threshold)
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
    
    return { candidates: sliced };
  }

  private async executeRerankOperator(
    operator: PlanOperator,
    inputCandidates: SymbolCandidate[],
    query: string
  ): Promise<{ candidates: SymbolCandidate[], hits: SearchHit[] }> {
    // Placeholder for rerank operation - would use actual reranking model
    const hits: SearchHit[] = inputCandidates.map(c => ({
      file: c.file_path,
      line: c.upstream_line || 1,
      col: c.upstream_col || 0,
      score: c.score,
      why: c.match_reasons
    }));
    
    return { candidates: inputCandidates, hits };
  }

  // Cost model and telemetry methods

  private initializeBaseCostModel(): CostModel {
    const operatorCosts = new Map<string, OperatorCostModel>();

    // Initialize with reasonable defaults
    operatorCosts.set('symbol', {
      operator_type: 'symbol',
      base_cost_ms: 2.0,
      cost_per_candidate: 0.01,
      cost_per_parameter: new Map([['max_ast_depth', 0.5]]),
      candidate_selectivity: 0.7,
      quality_impact: 0.8,
      confidence_intervals: {
        cost_lower: 1.5, cost_upper: 3.0,
        selectivity_lower: 0.6, selectivity_upper: 0.8
      }
    });

    operatorCosts.set('struct', {
      operator_type: 'struct',
      base_cost_ms: 3.0,
      cost_per_candidate: 0.015,
      cost_per_parameter: new Map([['max_ast_depth', 0.8]]),
      candidate_selectivity: 0.6,
      quality_impact: 0.75,
      confidence_intervals: {
        cost_lower: 2.0, cost_upper: 4.5,
        selectivity_lower: 0.5, selectivity_upper: 0.7
      }
    });

    return {
      operator_costs: operatorCosts,
      learned_at: new Date(),
      confidence: 0.5,
      sample_count: 0,
      last_updated: new Date()
    };
  }

  private estimateOperatorCost(operatorType: string, parameters: any): number {
    const costModel = this.costModel.operator_costs.get(operatorType);
    if (!costModel) return 5.0; // Default

    let cost = costModel.base_cost_ms;
    
    // Add parameter costs
    for (const [param, value] of Object.entries(parameters)) {
      const paramCost = costModel.cost_per_parameter.get(param) || 0;
      cost += paramCost * (value as number || 1);
    }

    return cost;
  }

  private async updateCostModel(): Promise<void> {
    if (this.telemetryBuffer.length < 10) return;

    // Update cost model based on collected telemetry
    const metrics = this.telemetryBuffer.splice(0); // Clear buffer
    
    for (const metric of metrics) {
      for (const [operatorName, actualTime] of metric.operator_times) {
        const operatorType = this.extractOperatorType(operatorName);
        const costModel = this.costModel.operator_costs.get(operatorType);
        
        if (costModel) {
          // Simple exponential moving average update
          const alpha = 0.1;
          costModel.base_cost_ms = (1 - alpha) * costModel.base_cost_ms + alpha * actualTime;
        }
      }
    }

    this.costModel.last_updated = new Date();
    this.costModel.sample_count += metrics.length;
  }

  private extractOperatorType(operatorName: string): string {
    if (operatorName.includes('symbol')) return 'symbol';
    if (operatorName.includes('struct')) return 'struct';
    if (operatorName.includes('slice')) return 'slice';
    if (operatorName.includes('rerank')) return 'rerank';
    return 'unknown';
  }

  private validatePlanConstraints(plan: QueryPlan, constraints: PlanConstraints): void {
    if (plan.estimated_cost_ms > constraints.max_total_time_ms) {
      throw new Error(`Plan exceeds time constraint: ${plan.estimated_cost_ms}ms > ${constraints.max_total_time_ms}ms`);
    }

    if (plan.estimated_recall_at_50 < constraints.min_recall_at_50) {
      throw new Error(`Plan below recall constraint: ${plan.estimated_recall_at_50} < ${constraints.min_recall_at_50}`);
    }

    if (constraints.span_invariants && !this.validateSpanInvariants(plan)) {
      throw new Error('Plan violates span invariants');
    }
  }

  private validateSpanInvariants(plan: QueryPlan): boolean {
    // Ensure plan preserves span accuracy requirements
    return true; // Placeholder
  }

  // Metric calculation methods
  private async calculateRecallAt50(hits: SearchHit[], query: string): Promise<number> {
    // Placeholder - would calculate actual recall@50
    return 0.8;
  }

  private async calculateNDCG(hits: SearchHit[], query: string): Promise<number> {
    // Placeholder - would calculate actual nDCG
    return 0.85;
  }

  private calculateSpanAccuracy(hits: SearchHit[]): number {
    // Placeholder - would validate span accuracy
    return 0.99;
  }

  /**
   * Generate Q-cost curve for performance analysis
   */
  generateQCostCurve(): Array<{ time_ms: number, recall_at_50: number, ndcg: number }> {
    if (this.executionHistory.length < 100) {
      return [];
    }

    // Group by time buckets and calculate metrics
    const buckets = new Map<number, ExecutionMetrics[]>();
    
    for (const metric of this.executionHistory) {
      const bucket = Math.floor(metric.total_time_ms / 5) * 5; // 5ms buckets
      const existing = buckets.get(bucket) || [];
      existing.push(metric);
      buckets.set(bucket, existing);
    }

    const curve: Array<{ time_ms: number, recall_at_50: number, ndcg: number }> = [];
    
    for (const [timeBucket, metrics] of buckets) {
      if (metrics.length >= 5) { // Require minimum sample size
        const avgRecall = metrics.reduce((sum, m) => sum + m.final_recall_at_50, 0) / metrics.length;
        const avgNDCG = metrics.reduce((sum, m) => sum + m.final_ndcg, 0) / metrics.length;
        
        curve.push({
          time_ms: timeBucket,
          recall_at_50: avgRecall,
          ndcg: avgNDCG
        });
      }
    }

    return curve.sort((a, b) => a.time_ms - b.time_ms);
  }

  /**
   * Get performance metrics for gate monitoring
   */
  getPerformanceMetrics() {
    const recentMetrics = this.executionHistory.slice(-1000); // Last 1000 executions
    
    if (recentMetrics.length === 0) {
      return {
        plans_executed: 0,
        avg_execution_time_ms: 0,
        p99_execution_time_ms: 0,
        avg_recall_at_50: 0,
        cost_model_confidence: this.costModel.confidence,
        cost_model_samples: this.costModel.sample_count,
      };
    }

    const times = recentMetrics.map(m => m.total_time_ms).sort((a, b) => a - b);
    const p99Index = Math.floor(times.length * 0.99);
    const p99Time = times[p99Index];

    return {
      plans_executed: recentMetrics.length,
      avg_execution_time_ms: times.reduce((a, b) => a + b, 0) / times.length,
      p99_execution_time_ms: p99Time,
      avg_recall_at_50: recentMetrics.reduce((sum, m) => sum + m.final_recall_at_50, 0) / recentMetrics.length,
      cost_model_confidence: this.costModel.confidence,
      cost_model_samples: this.costModel.sample_count,
    };
  }
}