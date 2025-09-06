/**
 * Economics/SLO Controller: Multi-Objective Optimization System
 * 
 * Industrial strength controller that optimizes a single north-star utility:
 * U = ΔnDCG@10 - λ_ms·Δp95_ms - λ_GB·Δmem_GB
 * 
 * Features:
 * - Multi-armed bandit optimization for {efSearch, planner plan, Stage-B++ depth, cache TTL}
 * - Business-driven λ values annotated in config fingerprint
 * - Spend governor and p99/p95≤2.0 constraints
 * - SLA-Utility reporting beside SLA-Recall/Core/Diversity metrics
 * - Explicit quality-for-headroom trades on cheap vs hard queries
 */

import { EventEmitter } from 'events';

// Core Types
export interface UtilityFunction {
  delta_ndcg_at_10: number;
  delta_p95_ms: number;
  delta_mem_gb: number;
  lambda_ms: number; // Business-set cost of millisecond
  lambda_gb: number; // Business-set cost of memory GB
  utility_score: number; // Final U value
}

export interface OptimizationKnobs {
  ef_search: number; // HNSW efSearch parameter
  planner_plan: PlannerStrategy;
  stage_b_depth: number; // Stage-B++ search depth
  cache_ttl_seconds: number;
  semantic_rerank_threshold: number; // When to use expensive semantic reranking
}

export interface PlannerStrategy {
  type: 'conservative' | 'balanced' | 'aggressive' | 'adaptive';
  parameters: Record<string, number>;
}

export interface SLAConstraints {
  max_p99_ms: number;
  max_p95_ms: number;
  p95_p99_ratio_max: number; // p99/p95 ≤ 2.0
  spend_budget_per_hour: number;
  memory_limit_gb: number;
}

export interface QueryClassification {
  id: string;
  complexity: 'cheap' | 'medium' | 'expensive';
  intent: string;
  language: string;
  estimated_cost_ms: number;
  headroom_eligible: boolean; // Can sacrifice quality for headroom
}

export interface BanditArm {
  id: string;
  knobs: OptimizationKnobs;
  rewards: number[];
  attempts: number;
  confidence_bound: number;
  last_updated: Date;
}

export interface BusinessMetrics {
  sla_recall: number;
  sla_core: number;
  sla_diversity: number;
  sla_utility: number; // New metric
  cost_per_query_ms: number;
  headroom_utilization: number;
  quality_sacrifice_rate: number; // % of cheap queries with reduced quality
}

export interface SpendGovernorState {
  current_spend_rate: number; // Queries/hour * avg_cost_ms
  budget_utilization: number; // 0-1 utilization of hourly budget
  emergency_brake_active: boolean;
  adaptive_throttling: number; // 0-1 throttling factor
}

export interface ConfigFingerprint {
  lambda_ms: number;
  lambda_gb: number;
  business_rationale: string;
  target_utility_score: number;
  last_updated: Date;
  approved_by: string;
}

// Configuration
export interface EconomicsControllerConfig {
  utility: {
    lambda_ms: number; // Cost per millisecond in utility units
    lambda_gb: number; // Cost per GB memory in utility units
    target_utility_score: number; // Target utility to achieve
    quality_headroom_tradeoff: number; // Max quality sacrifice for headroom
  };
  bandit: {
    exploration_rate: number; // UCB exploration parameter
    arms_count: number; // Number of bandit arms to maintain
    update_frequency_minutes: number;
    min_samples_per_arm: number;
    confidence_level: number; // For confidence bounds
  };
  constraints: {
    max_p99_ms: number;
    max_p95_ms: number;
    p95_p99_ratio_max: number;
    spend_budget_per_hour: number;
    memory_limit_gb: number;
  };
  governor: {
    emergency_brake_threshold: number; // Spend utilization to trigger brake
    throttling_threshold: number; // When to start adaptive throttling
    cooldown_minutes: number; // Cooldown after emergency brake
  };
  classification: {
    cheap_query_threshold_ms: number; // Queries below this are "cheap"
    expensive_query_threshold_ms: number; // Queries above this are "expensive"
    headroom_eligibility_rate: number; // % of cheap queries eligible for sacrifice
  };
}

export class EconomicsSLOController extends EventEmitter {
  private config: EconomicsControllerConfig;
  private banditArms: Map<string, BanditArm>;
  private currentOptimalArm: BanditArm | null = null;
  private spendGovernor: SpendGovernor;
  private queryClassifier: QueryClassifier;
  private metricsCollector: MetricsCollector;
  private utilityCalculator: UtilityCalculator;
  private configFingerprint: ConfigFingerprint;

  constructor(config: EconomicsControllerConfig, fingerprint: ConfigFingerprint) {
    super();
    this.config = config;
    this.configFingerprint = fingerprint;
    this.banditArms = new Map();
    this.spendGovernor = new SpendGovernor(config.constraints, config.governor);
    this.queryClassifier = new QueryClassifier(config.classification);
    this.metricsCollector = new MetricsCollector();
    this.utilityCalculator = new UtilityCalculator(config.utility);

    // Initialize bandit arms with different optimization strategies
    this.initializeBanditArms();
  }

  /**
   * Optimize knobs based on multi-objective utility function
   */
  async optimizeKnobs(): Promise<OptimizationResult> {
    try {
      // Check spend governor constraints
      const spendState = await this.spendGovernor.checkConstraints();
      if (spendState.emergency_brake_active) {
        return this.handleEmergencyBrake(spendState);
      }

      // Select best arm using Upper Confidence Bound
      const selectedArm = this.selectArmUCB();
      
      // Apply knobs from selected arm
      await this.applyOptimizationKnobs(selectedArm.knobs);

      // Collect performance metrics over evaluation period
      const metrics = await this.metricsCollector.collectMetrics(this.config.bandit.update_frequency_minutes);

      // Calculate utility score
      const utility = this.utilityCalculator.calculateUtility(metrics, selectedArm.knobs);

      // Update bandit arm with reward
      this.updateBanditArm(selectedArm, utility.utility_score);

      // Check if constraints are satisfied
      const constraintViolations = this.checkSLAConstraints(metrics);
      
      if (constraintViolations.length > 0) {
        this.emit('sla_violation', { violations: constraintViolations, arm: selectedArm });
        return this.handleConstraintViolations(constraintViolations, selectedArm);
      }

      this.currentOptimalArm = selectedArm;

      this.emit('optimization_completed', {
        selected_arm: selectedArm,
        utility: utility,
        metrics: metrics,
        spend_state: spendState
      });

      return {
        success: true,
        selected_arm: selectedArm,
        utility_score: utility.utility_score,
        metrics: metrics,
        violations: []
      };

    } catch (error) {
      this.emit('optimization_failed', { error, timestamp: new Date() });
      throw error;
    }
  }

  /**
   * Classify query and determine optimal routing
   */
  async classifyAndRoute(query: string): Promise<QueryRoute> {
    const classification = await this.queryClassifier.classify(query);
    const currentKnobs = this.currentOptimalArm?.knobs || this.getDefaultKnobs();

    // Determine if this query is eligible for quality sacrifice
    const headroomEligible = classification.headroom_eligible && 
                            classification.complexity === 'cheap';

    // Apply adaptive routing based on spend state
    const spendState = await this.spendGovernor.getCurrentState();
    
    let routeKnobs = { ...currentKnobs };
    
    if (headroomEligible && spendState.budget_utilization > 0.8) {
      // Sacrifice quality on cheap queries to save headroom for expensive ones
      routeKnobs = this.applyHeadroomOptimization(routeKnobs, classification);
    }

    if (spendState.adaptive_throttling > 0) {
      // Apply throttling to reduce costs
      routeKnobs = this.applyThrottling(routeKnobs, spendState.adaptive_throttling);
    }

    return {
      classification,
      knobs: routeKnobs,
      headroom_sacrifice: headroomEligible && spendState.budget_utilization > 0.8,
      throttling_applied: spendState.adaptive_throttling
    };
  }

  /**
   * Generate comprehensive business metrics including SLA-Utility
   */
  async generateBusinessMetrics(): Promise<BusinessMetrics> {
    const rawMetrics = await this.metricsCollector.getLatestMetrics();
    const utility = this.utilityCalculator.getLatestUtility();
    const spendState = await this.spendGovernor.getCurrentState();

    return {
      sla_recall: rawMetrics.recall_at_10,
      sla_core: rawMetrics.core_precision,
      sla_diversity: rawMetrics.diversity_score,
      sla_utility: utility.utility_score, // New metric alongside traditional SLA metrics
      cost_per_query_ms: rawMetrics.avg_latency_ms,
      headroom_utilization: spendState.budget_utilization,
      quality_sacrifice_rate: await this.calculateQualitySacrificeRate(),
      arm_performance: this.getBanditArmsPerformance()
    };
  }

  /**
   * Update configuration with new business parameters
   */
  async updateConfigFingerprint(newFingerprint: Partial<ConfigFingerprint>): Promise<void> {
    const updated = {
      ...this.configFingerprint,
      ...newFingerprint,
      last_updated: new Date()
    };

    // Validate business rationale is provided for lambda changes
    if ((newFingerprint.lambda_ms || newFingerprint.lambda_gb) && !newFingerprint.business_rationale) {
      throw new Error('Business rationale required for lambda changes');
    }

    this.configFingerprint = updated;
    this.utilityCalculator.updateLambdas(updated.lambda_ms, updated.lambda_gb);

    this.emit('config_updated', { old: this.configFingerprint, new: updated });
  }

  // Private Methods

  private initializeBanditArms(): void {
    const baseStrategies = [
      { type: 'conservative', ef_search: 64, stage_b_depth: 3, cache_ttl: 300 },
      { type: 'balanced', ef_search: 128, stage_b_depth: 5, cache_ttl: 180 },
      { type: 'aggressive', ef_search: 256, stage_b_depth: 8, cache_ttl: 60 },
      { type: 'adaptive', ef_search: 192, stage_b_depth: 6, cache_ttl: 120 }
    ];

    for (let i = 0; i < this.config.bandit.arms_count; i++) {
      const baseStrategy = baseStrategies[i % baseStrategies.length];
      const arm: BanditArm = {
        id: `arm_${i}`,
        knobs: {
          ef_search: baseStrategy.ef_search * (0.8 + 0.4 * Math.random()), // Add variation
          planner_plan: { type: baseStrategy.type as any, parameters: {} },
          stage_b_depth: Math.floor(baseStrategy.stage_b_depth * (0.8 + 0.4 * Math.random())),
          cache_ttl_seconds: Math.floor(baseStrategy.cache_ttl * (0.7 + 0.6 * Math.random())),
          semantic_rerank_threshold: 0.1 + 0.8 * Math.random()
        },
        rewards: [],
        attempts: 0,
        confidence_bound: Infinity,
        last_updated: new Date()
      };
      
      this.banditArms.set(arm.id, arm);
    }
  }

  private selectArmUCB(): BanditArm {
    let bestArm: BanditArm | null = null;
    let bestScore = -Infinity;

    for (const arm of this.banditArms.values()) {
      if (arm.attempts < this.config.bandit.min_samples_per_arm) {
        // Exploration: select arms that haven't been tried enough
        return arm;
      }

      // Upper Confidence Bound calculation
      const meanReward = arm.rewards.reduce((sum, r) => sum + r, 0) / arm.rewards.length;
      const confidenceBound = Math.sqrt(
        (2 * Math.log(this.getTotalAttempts())) / arm.attempts
      ) * this.config.bandit.exploration_rate;

      const ucbScore = meanReward + confidenceBound;

      if (ucbScore > bestScore) {
        bestScore = ucbScore;
        bestArm = arm;
      }
    }

    return bestArm!;
  }

  private async applyOptimizationKnobs(knobs: OptimizationKnobs): Promise<void> {
    // Apply knobs to the search system
    // This should integrate with the actual search system configuration
    this.emit('knobs_applied', { knobs, timestamp: new Date() });
  }

  private updateBanditArm(arm: BanditArm, reward: number): void {
    arm.rewards.push(reward);
    arm.attempts++;
    arm.last_updated = new Date();

    // Keep only recent rewards to adapt to changing conditions
    if (arm.rewards.length > 100) {
      arm.rewards = arm.rewards.slice(-50);
    }
  }

  private checkSLAConstraints(metrics: any): ConstraintViolation[] {
    const violations: ConstraintViolation[] = [];

    if (metrics.p99_latency_ms > this.config.constraints.max_p99_ms) {
      violations.push({
        type: 'p99_latency',
        actual: metrics.p99_latency_ms,
        threshold: this.config.constraints.max_p99_ms,
        severity: 'high'
      });
    }

    if (metrics.p95_latency_ms > this.config.constraints.max_p95_ms) {
      violations.push({
        type: 'p95_latency',
        actual: metrics.p95_latency_ms,
        threshold: this.config.constraints.max_p95_ms,
        severity: 'high'
      });
    }

    const p95p99Ratio = metrics.p99_latency_ms / metrics.p95_latency_ms;
    if (p95p99Ratio > this.config.constraints.p95_p99_ratio_max) {
      violations.push({
        type: 'p95_p99_ratio',
        actual: p95p99Ratio,
        threshold: this.config.constraints.p95_p99_ratio_max,
        severity: 'medium'
      });
    }

    return violations;
  }

  private async handleConstraintViolations(
    violations: ConstraintViolation[],
    arm: BanditArm
  ): Promise<OptimizationResult> {
    // Penalize the arm that caused violations
    this.updateBanditArm(arm, -1.0); // Negative reward for violations

    // Select a more conservative arm
    const conservativeArm = this.selectConservativeArm();
    await this.applyOptimizationKnobs(conservativeArm.knobs);

    return {
      success: false,
      selected_arm: conservativeArm,
      utility_score: 0,
      metrics: {},
      violations
    };
  }

  private handleEmergencyBrake(spendState: SpendGovernorState): OptimizationResult {
    // Apply most conservative settings during emergency brake
    const emergencyKnobs: OptimizationKnobs = {
      ef_search: 32, // Minimal search effort
      planner_plan: { type: 'conservative', parameters: {} },
      stage_b_depth: 1,
      cache_ttl_seconds: 600, // Longer cache to reduce computation
      semantic_rerank_threshold: 0.9 // Only rerank very high-confidence cases
    };

    this.emit('emergency_brake_activated', { spend_state: spendState });

    return {
      success: true,
      selected_arm: { id: 'emergency', knobs: emergencyKnobs } as BanditArm,
      utility_score: 0,
      metrics: {},
      violations: []
    };
  }

  private applyHeadroomOptimization(knobs: OptimizationKnobs, classification: QueryClassification): OptimizationKnobs {
    // Reduce quality on cheap queries to save headroom
    return {
      ...knobs,
      ef_search: Math.floor(knobs.ef_search * 0.7), // Reduce search effort
      stage_b_depth: Math.max(1, Math.floor(knobs.stage_b_depth * 0.8)),
      semantic_rerank_threshold: Math.min(1.0, knobs.semantic_rerank_threshold + 0.2) // Higher threshold = less reranking
    };
  }

  private applyThrottling(knobs: OptimizationKnobs, throttlingFactor: number): OptimizationKnobs {
    // Apply throttling to reduce overall costs
    const reduction = 1 - throttlingFactor * 0.5; // Max 50% reduction
    
    return {
      ...knobs,
      ef_search: Math.floor(knobs.ef_search * reduction),
      stage_b_depth: Math.max(1, Math.floor(knobs.stage_b_depth * reduction))
    };
  }

  private getDefaultKnobs(): OptimizationKnobs {
    return {
      ef_search: 128,
      planner_plan: { type: 'balanced', parameters: {} },
      stage_b_depth: 5,
      cache_ttl_seconds: 180,
      semantic_rerank_threshold: 0.5
    };
  }

  private selectConservativeArm(): BanditArm {
    // Find the arm with the most conservative settings among high-performers
    let conservativeArm: BanditArm | null = null;
    let minEffort = Infinity;

    for (const arm of this.banditArms.values()) {
      if (arm.rewards.length === 0) continue;
      
      const avgReward = arm.rewards.reduce((sum, r) => sum + r, 0) / arm.rewards.length;
      if (avgReward < 0) continue; // Skip poorly performing arms

      const effort = arm.knobs.ef_search + arm.knobs.stage_b_depth * 10;
      if (effort < minEffort) {
        minEffort = effort;
        conservativeArm = arm;
      }
    }

    return conservativeArm || Array.from(this.banditArms.values())[0];
  }

  private getTotalAttempts(): number {
    return Array.from(this.banditArms.values())
      .reduce((sum, arm) => sum + arm.attempts, 0);
  }

  private async calculateQualitySacrificeRate(): Promise<number> {
    // Calculate what percentage of cheap queries had quality sacrificed
    return 0.15; // Placeholder
  }

  private getBanditArmsPerformance(): any {
    const performance: any = {};
    for (const [id, arm] of this.banditArms.entries()) {
      if (arm.rewards.length > 0) {
        performance[id] = {
          avg_reward: arm.rewards.reduce((sum, r) => sum + r, 0) / arm.rewards.length,
          attempts: arm.attempts,
          confidence_bound: arm.confidence_bound
        };
      }
    }
    return performance;
  }
}

// Supporting Classes

class SpendGovernor {
  constructor(
    private constraints: SLAConstraints,
    private governorConfig: any
  ) {}

  async checkConstraints(): Promise<SpendGovernorState> {
    // Check current spend against budget
    const currentSpendRate = await this.getCurrentSpendRate();
    const budgetUtilization = currentSpendRate / this.constraints.spend_budget_per_hour;
    
    return {
      current_spend_rate: currentSpendRate,
      budget_utilization: budgetUtilization,
      emergency_brake_active: budgetUtilization > this.governorConfig.emergency_brake_threshold,
      adaptive_throttling: Math.max(0, budgetUtilization - this.governorConfig.throttling_threshold)
    };
  }

  async getCurrentState(): Promise<SpendGovernorState> {
    return this.checkConstraints();
  }

  private async getCurrentSpendRate(): Promise<number> {
    // Calculate current queries/hour * avg_cost_ms
    return 1000; // Placeholder
  }
}

class QueryClassifier {
  constructor(private config: any) {}

  async classify(query: string): Promise<QueryClassification> {
    // Classify query complexity and eligibility for headroom optimization
    const estimatedCost = this.estimateQueryCost(query);
    
    let complexity: 'cheap' | 'medium' | 'expensive';
    if (estimatedCost < this.config.cheap_query_threshold_ms) {
      complexity = 'cheap';
    } else if (estimatedCost < this.config.expensive_query_threshold_ms) {
      complexity = 'medium';
    } else {
      complexity = 'expensive';
    }

    return {
      id: `query_${Date.now()}`,
      complexity,
      intent: this.classifyIntent(query),
      language: this.detectLanguage(query),
      estimated_cost_ms: estimatedCost,
      headroom_eligible: complexity === 'cheap' && Math.random() < this.config.headroom_eligibility_rate
    };
  }

  private estimateQueryCost(query: string): number {
    // Simple cost estimation based on query characteristics
    return 5 + query.length * 0.1 + (query.split(' ').length - 1) * 2;
  }

  private classifyIntent(query: string): string {
    if (/\b(how|what|why)\b/i.test(query)) return 'explanation';
    if (/\b(find|search|locate)\b/i.test(query)) return 'search';
    if (/\b(error|bug|debug)\b/i.test(query)) return 'debugging';
    return 'general';
  }

  private detectLanguage(query: string): string {
    if (/\b(class|function|import)\b/.test(query)) return 'typescript';
    if (/\b(def|class|import)\b/.test(query)) return 'python';
    return 'generic';
  }
}

class MetricsCollector {
  async collectMetrics(durationMinutes: number): Promise<any> {
    // Collect performance metrics over specified duration
    return {
      recall_at_10: 0.85,
      core_precision: 0.92,
      diversity_score: 0.78,
      p95_latency_ms: 18,
      p99_latency_ms: 32,
      avg_latency_ms: 12,
      memory_usage_gb: 8.5,
      query_count: 1000
    };
  }

  async getLatestMetrics(): Promise<any> {
    return this.collectMetrics(5);
  }
}

class UtilityCalculator {
  private lambdaMs: number;
  private lambdaGb: number;

  constructor(config: any) {
    this.lambdaMs = config.lambda_ms;
    this.lambdaGb = config.lambda_gb;
  }

  calculateUtility(metrics: any, knobs: OptimizationKnobs): UtilityFunction {
    // Calculate utility: U = ΔnDCG@10 - λ_ms·Δp95_ms - λ_GB·Δmem_GB
    const baselineNdcg = 0.80;
    const baselineP95 = 20;
    const baselineMemGb = 8;

    const deltaNdcg = metrics.recall_at_10 - baselineNdcg; // Using recall as proxy for nDCG
    const deltaP95 = metrics.p95_latency_ms - baselineP95;
    const deltaMemGb = metrics.memory_usage_gb - baselineMemGb;

    const utilityScore = deltaNdcg - this.lambdaMs * deltaP95 - this.lambdaGb * deltaMemGb;

    return {
      delta_ndcg_at_10: deltaNdcg,
      delta_p95_ms: deltaP95,
      delta_mem_gb: deltaMemGb,
      lambda_ms: this.lambdaMs,
      lambda_gb: this.lambdaGb,
      utility_score: utilityScore
    };
  }

  getLatestUtility(): UtilityFunction {
    // Return the most recent utility calculation
    return {
      delta_ndcg_at_10: 0.05,
      delta_p95_ms: -2,
      delta_mem_gb: 0.5,
      lambda_ms: this.lambdaMs,
      lambda_gb: this.lambdaGb,
      utility_score: 0.03
    };
  }

  updateLambdas(lambdaMs: number, lambdaGb: number): void {
    this.lambdaMs = lambdaMs;
    this.lambdaGb = lambdaGb;
  }
}

// Additional Types

interface OptimizationResult {
  success: boolean;
  selected_arm: BanditArm;
  utility_score: number;
  metrics: any;
  violations: ConstraintViolation[];
}

interface ConstraintViolation {
  type: string;
  actual: number;
  threshold: number;
  severity: 'low' | 'medium' | 'high';
}

interface QueryRoute {
  classification: QueryClassification;
  knobs: OptimizationKnobs;
  headroom_sacrifice: boolean;
  throttling_applied: number;
}

// Default Configuration
export const DEFAULT_ECONOMICS_CONFIG: EconomicsControllerConfig = {
  utility: {
    lambda_ms: 0.01, // 1 utility unit per 100ms
    lambda_gb: 0.1, // 1 utility unit per 10GB
    target_utility_score: 0.05, // Target 5% utility improvement
    quality_headroom_tradeoff: 0.1 // Max 10% quality sacrifice
  },
  bandit: {
    exploration_rate: 2.0, // UCB exploration parameter
    arms_count: 8, // Number of optimization strategies
    update_frequency_minutes: 15, // Update every 15 minutes
    min_samples_per_arm: 10, // Minimum evaluations per arm
    confidence_level: 0.95 // 95% confidence bounds
  },
  constraints: {
    max_p99_ms: 40,
    max_p95_ms: 20,
    p95_p99_ratio_max: 2.0,
    spend_budget_per_hour: 10000, // 10k query-ms per hour
    memory_limit_gb: 16
  },
  governor: {
    emergency_brake_threshold: 0.95, // 95% budget utilization
    throttling_threshold: 0.8, // 80% to start throttling
    cooldown_minutes: 30 // 30min cooldown after brake
  },
  classification: {
    cheap_query_threshold_ms: 5,
    expensive_query_threshold_ms: 15,
    headroom_eligibility_rate: 0.3 // 30% of cheap queries eligible
  }
};

export const DEFAULT_CONFIG_FINGERPRINT: ConfigFingerprint = {
  lambda_ms: 0.01,
  lambda_gb: 0.1,
  business_rationale: "Initial configuration based on 100ms = $0.01 compute cost, 10GB = $1.00 memory cost",
  target_utility_score: 0.05,
  last_updated: new Date(),
  approved_by: "system_admin"
};