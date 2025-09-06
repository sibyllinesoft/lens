/**
 * Speculative Multi-Plan Planner - Evergreen Optimization System #3
 * 
 * Maintain three plans: {symbol-first, struct-first, lexical-first}
 * Predict distribution over plans from {intent, entropy, tokens, LSP coverage}
 * Execute winner + truncated backup (10-20% budget) with cooperative cancel
 * Promote whichever first reaches "k candidates with pos_in_cands≥m"
 * Start with {symbol-first vs struct-first}
 * 
 * Gate: fleet p99 -8-12% at flat recall, abort if p95 > +0.6ms
 * Hedge only when p95_headroom>h, keep planner spend ≤10% query budget
 */

import type { 
  SearchContext, 
  Candidate, 
  SearchMode,
  QueryIntent,
  IntentClassification 
} from '../types/core.js';
import type { SearchHit } from './span_resolver/types.js';
import { LensTracer } from '../telemetry/tracer.js';

export interface QueryPlan {
  id: string;
  name: string;
  mode: SearchMode;
  execution_order: ('lexical' | 'symbol' | 'struct')[];
  budget_allocation: number; // 0-1
  early_termination_threshold: number; // k candidates needed
  position_threshold: number; // m minimum position requirement
}

export interface PlanPrediction {
  plan_id: string;
  confidence: number; // 0-1
  expected_latency_ms: number;
  expected_recall: number;
  features_used: string[];
}

export interface PlanExecution {
  plan: QueryPlan;
  status: 'pending' | 'running' | 'completed' | 'cancelled' | 'failed';
  start_time: number;
  candidates: Candidate[];
  latency_ms?: number;
  early_terminated: boolean;
  budget_consumed: number;
}

export interface CooperativeCancel {
  should_cancel: boolean;
  reason: 'primary_success' | 'timeout' | 'budget_exceeded' | 'error';
  winning_plan?: string;
}

/**
 * Query feature extractor for plan prediction
 */
export class QueryFeatureExtractor {
  
  /**
   * Extract features from query and context for plan prediction
   */
  extractFeatures(query: string, context: SearchContext): Map<string, number> {
    const features = new Map<string, number>();
    const queryLower = query.toLowerCase();
    
    // Intent-based features
    const intent = this.classifyIntent(query);
    features.set('intent_def', intent.intent === 'def' ? 1 : 0);
    features.set('intent_refs', intent.intent === 'refs' ? 1 : 0);
    features.set('intent_symbol', intent.intent === 'symbol' ? 1 : 0);
    features.set('intent_struct', intent.intent === 'struct' ? 1 : 0);
    features.set('intent_lexical', intent.intent === 'lexical' ? 1 : 0);
    features.set('intent_nl', intent.intent === 'NL' ? 1 : 0);
    features.set('intent_confidence', intent.confidence);
    
    // Entropy-based features
    const entropy = this.calculateEntropy(query);
    features.set('query_entropy', entropy);
    features.set('entropy_high', entropy > 3.0 ? 1 : 0);
    features.set('entropy_low', entropy < 1.0 ? 1 : 0);
    
    // Token-based features
    const tokens = query.split(/\s+/).filter(t => t.length > 0);
    features.set('token_count', tokens.length);
    features.set('avg_token_length', tokens.reduce((sum, t) => sum + t.length, 0) / Math.max(tokens.length, 1));
    features.set('has_camel_case', /[a-z][A-Z]/.test(query) ? 1 : 0);
    features.set('has_snake_case', /_/.test(query) ? 1 : 0);
    features.set('has_special_chars', /[^\w\s]/.test(query) ? 1 : 0);
    features.set('has_quotes', /["']/.test(query) ? 1 : 0);
    
    // Pattern-based features
    features.set('has_function_pattern', /\w+\s*\(/.test(query) ? 1 : 0);
    features.set('has_class_pattern', /class\s+\w+/.test(queryLower) ? 1 : 0);
    features.set('has_import_pattern', /import|from|require/.test(queryLower) ? 1 : 0);
    features.set('has_type_pattern', /interface|type|struct/.test(queryLower) ? 1 : 0);
    
    // LSP coverage (simulated - would integrate with actual LSP data)
    features.set('lsp_coverage', Math.random()); // TODO: Replace with real LSP coverage
    features.set('lsp_symbols_available', Math.random() > 0.3 ? 1 : 0);
    features.set('lsp_definitions_found', Math.random() > 0.5 ? 1 : 0);
    
    // Query complexity
    features.set('query_length', query.length);
    features.set('is_single_word', tokens.length === 1 ? 1 : 0);
    features.set('is_phrase', tokens.length > 1 && tokens.length <= 5 ? 1 : 0);
    features.set('is_sentence', tokens.length > 5 ? 1 : 0);
    
    return features;
  }

  private classifyIntent(query: string): IntentClassification {
    const queryLower = query.toLowerCase();
    
    // Simple intent classification - could be enhanced with ML
    if (/\b(def|define|definition)\b/.test(queryLower)) {
      return { intent: 'def', confidence: 0.9, features: { has_definition_pattern: true } as any };
    }
    if (/\b(ref|reference|usage|used)\b/.test(queryLower)) {
      return { intent: 'refs', confidence: 0.9, features: { has_reference_pattern: true } as any };
    }
    if (/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(query.trim())) {
      return { intent: 'symbol', confidence: 0.8, features: { has_symbol_prefix: true } as any };
    }
    if (/[{}[\]().]/.test(query)) {
      return { intent: 'struct', confidence: 0.7, features: { has_structural_chars: true } as any };
    }
    if (query.split(/\s+/).length > 2) {
      return { intent: 'NL', confidence: 0.6, features: { is_natural_language: true } as any };
    }
    
    return { intent: 'lexical', confidence: 0.5, features: {} as any };
  }

  private calculateEntropy(query: string): number {
    const chars = query.toLowerCase();
    const freq = new Map<string, number>();
    
    for (const char of chars) {
      freq.set(char, (freq.get(char) || 0) + 1);
    }
    
    let entropy = 0;
    const total = chars.length;
    
    for (const count of freq.values()) {
      const p = count / total;
      entropy -= p * Math.log2(p);
    }
    
    return entropy;
  }
}

/**
 * Plan predictor using simple heuristics (could be enhanced with ML)
 */
export class PlanPredictor {
  private featureExtractor = new QueryFeatureExtractor();
  
  // Learned weights for plan prediction (would be trained on historical data)
  private readonly planWeights = {
    'symbol-first': {
      intent_symbol: 0.8,
      intent_def: 0.6,
      has_camel_case: 0.4,
      lsp_symbols_available: 0.5,
      is_single_word: 0.3,
    },
    'struct-first': {
      intent_struct: 0.9,
      has_special_chars: 0.6,
      has_function_pattern: 0.5,
      has_type_pattern: 0.4,
      entropy_high: 0.3,
    },
    'lexical-first': {
      intent_lexical: 0.7,
      intent_nl: 0.8,
      is_sentence: 0.6,
      entropy_low: 0.4,
      has_quotes: 0.3,
    },
  };

  /**
   * Predict best plans for a query
   */
  predictPlans(query: string, context: SearchContext): PlanPrediction[] {
    const features = this.featureExtractor.extractFeatures(query, context);
    const predictions: PlanPrediction[] = [];
    
    for (const [planId, weights] of Object.entries(this.planWeights)) {
      let score = 0.5; // Base score
      const usedFeatures: string[] = [];
      
      for (const [feature, weight] of Object.entries(weights)) {
        const featureValue = features.get(feature) || 0;
        score += weight * featureValue;
        if (featureValue > 0) {
          usedFeatures.push(feature);
        }
      }
      
      // Normalize score to probability
      const confidence = Math.max(0.1, Math.min(0.9, score));
      
      // Estimate latency based on plan type and features
      const expectedLatency = this.estimateLatency(planId, features);
      
      // Estimate recall based on intent match
      const expectedRecall = this.estimateRecall(planId, features);
      
      predictions.push({
        plan_id: planId,
        confidence,
        expected_latency_ms: expectedLatency,
        expected_recall,
        features_used: usedFeatures,
      });
    }
    
    // Sort by confidence
    predictions.sort((a, b) => b.confidence - a.confidence);
    
    return predictions;
  }

  private estimateLatency(planId: string, features: Map<string, number>): number {
    // Base latencies for each plan type
    const baseLat = {
      'symbol-first': 8,  // Symbol search is fast
      'struct-first': 12, // Structural search is medium
      'lexical-first': 5, // Lexical search is fastest
    }[planId] || 10;
    
    // Adjust based on query complexity
    const complexity = features.get('query_length')! / 20 + 
                      features.get('token_count')! * 0.5 +
                      features.get('query_entropy')! * 0.3;
    
    return baseLat + complexity;
  }

  private estimateRecall(planId: string, features: Map<string, number>): number {
    // Base recall for each plan type
    const baseRecall = {
      'symbol-first': 0.75,
      'struct-first': 0.65,
      'lexical-first': 0.85,
    }[planId] || 0.7;
    
    // Adjust based on intent match
    const intentBonus = this.getIntentMatch(planId, features) * 0.15;
    
    return Math.min(0.95, baseRecall + intentBonus);
  }

  private getIntentMatch(planId: string, features: Map<string, number>): number {
    const matches = {
      'symbol-first': (features.get('intent_symbol') || 0) + (features.get('intent_def') || 0),
      'struct-first': features.get('intent_struct') || 0,
      'lexical-first': (features.get('intent_lexical') || 0) + (features.get('intent_nl') || 0),
    };
    
    return matches[planId as keyof typeof matches] || 0;
  }
}

/**
 * Speculative multi-plan query planner
 */
export class SpeculativeMultiPlanPlanner {
  private predictor = new PlanPredictor();
  private enabled = false;
  private p95HeadroomThreshold = 5.0; // ms - only hedge when we have headroom
  private maxPlannerBudgetPercent = 10; // ≤10% query budget
  
  // Available query plans
  private readonly plans: Map<string, QueryPlan> = new Map([
    ['symbol-first', {
      id: 'symbol-first',
      name: 'Symbol-First Search',
      mode: 'struct',
      execution_order: ['symbol', 'struct', 'lexical'],
      budget_allocation: 1.0,
      early_termination_threshold: 10, // k candidates
      position_threshold: 3, // m minimum position
    }],
    ['struct-first', {
      id: 'struct-first', 
      name: 'Structural-First Search',
      mode: 'struct',
      execution_order: ['struct', 'symbol', 'lexical'],
      budget_allocation: 1.0,
      early_termination_threshold: 10,
      position_threshold: 3,
    }],
    ['lexical-first', {
      id: 'lexical-first',
      name: 'Lexical-First Search', 
      mode: 'lex',
      execution_order: ['lexical', 'symbol', 'struct'],
      budget_allocation: 1.0,
      early_termination_threshold: 15, // More candidates needed for lexical
      position_threshold: 5,
    }],
  ]);

  /**
   * Enable speculative planning with performance constraints
   */
  enableWithConstraints(p95HeadroomMs: number, maxBudgetPercent: number): void {
    this.enabled = true;
    this.p95HeadroomThreshold = p95HeadroomMs;
    this.maxPlannerBudgetPercent = Math.min(10, maxBudgetPercent);
  }

  /**
   * Execute speculative multi-plan search
   */
  async executeSpeculativeSearch(
    context: SearchContext,
    currentP95Latency: number,
    maxLatencyBudget: number
  ): Promise<{
    primary_results: SearchHit[];
    backup_results: SearchHit[];
    execution_stats: {
      primary_plan: string;
      backup_plan?: string;
      primary_latency: number;
      backup_latency?: number;
      early_termination: boolean;
      budget_consumed: number;
    };
  }> {
    const span = LensTracer.createChildSpan('speculative_multi_plan', {
      'context.query': context.query,
      'current_p95': currentP95Latency,
      'budget.max_ms': maxLatencyBudget,
    });

    try {
      // Check if we should use speculative planning
      if (!this.shouldUseSpeculativePlanning(currentP95Latency, maxLatencyBudget)) {
        span.setAttributes({ 
          success: true, 
          skipped: true, 
          reason: 'insufficient_headroom_or_disabled' 
        });
        // Fallback to simple single-plan execution
        return this.executeSinglePlan(context, 'symbol-first');
      }

      // Predict best plans
      const predictions = this.predictor.predictPlans(context.query, context);
      const primaryPlan = this.plans.get(predictions[0].plan_id)!;
      const backupPlan = predictions.length > 1 ? this.plans.get(predictions[1].plan_id) : undefined;

      const plannerBudget = Math.min(
        maxLatencyBudget * (this.maxPlannerBudgetPercent / 100),
        maxLatencyBudget * 0.1 // Hard cap at 10%
      );

      // Execute primary and backup plans concurrently
      const executions: PlanExecution[] = [];
      const cancelSignal = { should_cancel: false } as CooperativeCancel;
      
      // Start primary plan with full budget
      const primaryExecution = this.executePlan(
        primaryPlan, 
        context, 
        maxLatencyBudget - plannerBudget,
        cancelSignal
      );
      executions.push(await this.wrapExecution(primaryExecution, 'primary'));

      // Start backup plan with truncated budget (10-20%)
      let backupExecution: PlanExecution | undefined;
      if (backupPlan) {
        const backupBudget = Math.min(
          maxLatencyBudget * 0.2, // Max 20% backup budget
          plannerBudget
        );
        
        const backupPromise = this.executePlan(
          backupPlan,
          context,
          backupBudget, 
          cancelSignal
        );
        backupExecution = await this.wrapExecution(backupPromise, 'backup');
        executions.push(backupExecution);
      }

      // Wait for first success or all completions
      const winner = await this.waitForWinner(executions, cancelSignal);
      
      // Cancel remaining executions
      cancelSignal.should_cancel = true;
      cancelSignal.reason = 'primary_success';
      cancelSignal.winning_plan = winner.plan.id;

      const primaryResults = winner.plan.id === primaryPlan.id ? winner.candidates : [];
      const backupResults = winner.plan.id !== primaryPlan.id ? winner.candidates : 
                           (backupExecution?.candidates || []);

      span.setAttributes({
        success: true,
        'primary_plan': primaryPlan.id,
        'backup_plan': backupPlan?.id || 'none',
        'winner': winner.plan.id,
        'primary_latency': executions[0]?.latency_ms || 0,
        'backup_latency': backupExecution?.latency_ms || 0,
        'early_termination': winner.early_terminated,
        'results_count': winner.candidates.length,
      });

      return {
        primary_results: primaryResults.map(c => this.candidateToHit(c)),
        backup_results: backupResults.map(c => this.candidateToHit(c)),
        execution_stats: {
          primary_plan: primaryPlan.id,
          backup_plan: backupPlan?.id,
          primary_latency: executions[0]?.latency_ms || 0,
          backup_latency: backupExecution?.latency_ms,
          early_termination: winner.early_terminated,
          budget_consumed: executions.reduce((sum, e) => sum + e.budget_consumed, 0),
        },
      };
      
    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      // Fallback to simple execution
      return this.executeSinglePlan(context, 'symbol-first');
    } finally {
      span.end();
    }
  }

  // Private helper methods

  private shouldUseSpeculativePlanning(currentP95: number, maxBudget: number): boolean {
    if (!this.enabled) return false;
    
    // Check if we have sufficient headroom
    const headroom = maxBudget - currentP95;
    return headroom > this.p95HeadroomThreshold;
  }

  private async executePlan(
    plan: QueryPlan,
    context: SearchContext,
    budgetMs: number,
    cancelSignal: CooperativeCancel
  ): Promise<PlanExecution> {
    const execution: PlanExecution = {
      plan,
      status: 'running',
      start_time: Date.now(),
      candidates: [],
      early_terminated: false,
      budget_consumed: 0,
    };

    const startTime = Date.now();
    let totalBudgetUsed = 0;

    try {
      // Execute stages in order specified by plan
      for (const stage of plan.execution_order) {
        if (cancelSignal.should_cancel) {
          execution.status = 'cancelled';
          break;
        }

        const stageBudget = budgetMs / plan.execution_order.length;
        const stageStart = Date.now();
        
        // Simulate stage execution (would integrate with actual search stages)
        const stageCandidates = await this.executeStage(stage, context, stageBudget);
        execution.candidates.push(...stageCandidates);
        
        const stageLatency = Date.now() - stageStart;
        totalBudgetUsed += stageLatency;
        
        // Check for early termination condition
        if (this.shouldTerminateEarly(execution, plan)) {
          execution.early_terminated = true;
          break;
        }

        // Check budget exhaustion
        if (totalBudgetUsed >= budgetMs) {
          break;
        }
      }

      execution.status = 'completed';
      execution.latency_ms = Date.now() - startTime;
      execution.budget_consumed = totalBudgetUsed;
      
    } catch (error) {
      execution.status = 'failed';
      execution.latency_ms = Date.now() - startTime;
      execution.budget_consumed = totalBudgetUsed;
    }

    return execution;
  }

  private async executeStage(
    stage: string,
    context: SearchContext,
    budgetMs: number
  ): Promise<Candidate[]> {
    // Simulate stage execution - would integrate with actual search engines
    const candidates: Candidate[] = [];
    
    const baseCount = {
      'lexical': 25,
      'symbol': 15, 
      'struct': 10,
    }[stage] || 10;
    
    for (let i = 0; i < baseCount; i++) {
      candidates.push({
        doc_id: `${stage}_${i}`,
        file_path: `/mock/file_${i}.ts`,
        line: i + 1,
        col: 0,
        score: Math.random() * 0.5 + 0.5,
        match_reasons: [stage as any],
        context: `Mock ${stage} result ${i}`,
      });
    }
    
    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, Math.random() * budgetMs * 0.1));
    
    return candidates;
  }

  private shouldTerminateEarly(execution: PlanExecution, plan: QueryPlan): boolean {
    if (execution.candidates.length < plan.early_termination_threshold) {
      return false;
    }
    
    // Check if we have k candidates with position ≥ m
    const qualityResults = execution.candidates.filter(c => c.score >= 0.7).length;
    return qualityResults >= plan.position_threshold;
  }

  private async wrapExecution(
    executionPromise: Promise<PlanExecution>,
    type: 'primary' | 'backup'
  ): Promise<PlanExecution> {
    try {
      return await executionPromise;
    } catch (error) {
      return {
        plan: { id: 'failed', name: 'Failed', mode: 'hybrid' } as QueryPlan,
        status: 'failed',
        start_time: Date.now(),
        candidates: [],
        early_terminated: false,
        budget_consumed: 0,
      };
    }
  }

  private async waitForWinner(
    executions: PlanExecution[],
    cancelSignal: CooperativeCancel
  ): Promise<PlanExecution> {
    // Simple implementation - wait for first completion or timeout
    return new Promise((resolve) => {
      const checkInterval = setInterval(() => {
        const completed = executions.find(e => e.status === 'completed');
        if (completed) {
          clearInterval(checkInterval);
          resolve(completed);
        }
        
        const allDone = executions.every(e => 
          e.status === 'completed' || e.status === 'failed' || e.status === 'cancelled'
        );
        if (allDone) {
          clearInterval(checkInterval);
          const best = executions.reduce((best, current) => 
            current.candidates.length > best.candidates.length ? current : best
          );
          resolve(best);
        }
      }, 10);
      
      // Timeout after 1 second
      setTimeout(() => {
        clearInterval(checkInterval);
        cancelSignal.should_cancel = true;
        cancelSignal.reason = 'timeout';
        resolve(executions[0] || executions.find(e => e.candidates.length > 0)!);
      }, 1000);
    });
  }

  private async executeSinglePlan(
    context: SearchContext,
    planId: string
  ): Promise<{
    primary_results: SearchHit[];
    backup_results: SearchHit[];
    execution_stats: any;
  }> {
    // Fallback single plan execution
    const plan = this.plans.get(planId)!;
    const cancelSignal = { should_cancel: false } as CooperativeCancel;
    
    const execution = await this.executePlan(plan, context, 100, cancelSignal);
    
    return {
      primary_results: execution.candidates.map(c => this.candidateToHit(c)),
      backup_results: [],
      execution_stats: {
        primary_plan: planId,
        primary_latency: execution.latency_ms || 0,
        early_termination: execution.early_terminated,
        budget_consumed: execution.budget_consumed,
      },
    };
  }

  private candidateToHit(candidate: Candidate): SearchHit {
    return {
      file: candidate.file_path,
      line: candidate.line,
      col: candidate.col,
      score: candidate.score,
      why: candidate.match_reasons as any[],
      snippet: candidate.context,
      symbol_kind: candidate.symbol_kind,
      ast_path: candidate.ast_path,
    };
  }

  /**
   * Get planner statistics
   */
  getStats(): {
    enabled: boolean;
    plans_available: number;
    p95_headroom_threshold: number;
    max_budget_percent: number;
  } {
    return {
      enabled: this.enabled,
      plans_available: this.plans.size,
      p95_headroom_threshold: this.p95HeadroomThreshold,
      max_budget_percent: this.maxPlannerBudgetPercent,
    };
  }
}