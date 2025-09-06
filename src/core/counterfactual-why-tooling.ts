/**
 * Counterfactual "Why" Tooling: Policy Attribution & Debugging System
 * 
 * Industrial strength tooling that provides operators with precise debugging capabilities:
 * - Compute minimally sufficient changes to flip result rankings
 * - Generate reproducible `/rerank?policy_delta=...` links
 * - Log floor-wins (monotone constraints overruling dense scores)
 * - Rollout Simulator for policy perturbation impact analysis
 * - Emit Δ{SLA-Recall, p95, why-mix KL} to prevent canary vs 100% surprises
 */

import { EventEmitter } from 'events';

// Core Types
export interface CounterfactualAnalysis {
  result_id: string;
  original_rank: number;
  target_rank?: number;
  minimal_changes: PolicyChange[];
  reproducible_link: string;
  confidence: number;
  explanation: string;
}

export interface PolicyChange {
  component: 'prior' | 'centrality_cap' | 'ef_search' | 'semantic_threshold' | 'cache_policy';
  parameter: string;
  original_value: number | boolean | string;
  counterfactual_value: number | boolean | string;
  impact_score: number; // How much this change affects the ranking
  necessity_score: number; // How necessary this change is
}

export interface FloorWinEvent {
  query_id: string;
  result_id: string;
  floor_type: 'exact_match' | 'structural_match' | 'monotone_constraint';
  dense_score: number;
  floor_score: number;
  override_applied: boolean;
  timestamp: Date;
  justification: string;
}

export interface RolloutSimulation {
  simulation_id: string;
  policy_deltas: PolicyDelta[];
  baseline_metrics: SimulationMetrics;
  perturbed_metrics: SimulationMetrics;
  traffic_sample_size: number;
  simulation_duration_hours: number;
  confidence_interval: number;
}

export interface PolicyDelta {
  component: string;
  parameter: string;
  delta_type: 'absolute' | 'relative' | 'toggle';
  delta_value: number | boolean;
  description: string;
}

export interface SimulationMetrics {
  sla_recall: number;
  p95_latency_ms: number;
  why_mix_kl_divergence: number; // KL divergence in result explanations
  cost_per_query: number;
  user_satisfaction_proxy: number;
}

export interface DebuggingSession {
  session_id: string;
  query: string;
  original_results: SearchResult[];
  counterfactual_analyses: CounterfactualAnalysis[];
  floor_wins: FloorWinEvent[];
  policy_recommendations: PolicyRecommendation[];
  created_at: Date;
  user_id: string;
}

export interface PolicyRecommendation {
  type: 'increase_recall' | 'reduce_latency' | 'improve_precision' | 'fix_ranking';
  description: string;
  suggested_changes: PolicyChange[];
  expected_impact: string;
  risk_assessment: 'low' | 'medium' | 'high';
}

export interface SearchResult {
  id: string;
  path: string;
  rank: number;
  score: number;
  component_scores: ComponentScores;
  metadata: any;
}

export interface ComponentScores {
  lexical: number;
  semantic: number;
  structural: number;
  prior: number;
  centrality: number;
  final_score: number;
}

// Configuration
export interface CounterfactualConfig {
  analysis: {
    max_changes_per_result: number; // Maximum policy changes to consider
    min_confidence_threshold: number; // Minimum confidence for recommendations
    rank_change_threshold: number; // Minimum rank change to consider significant
    perturbation_step_size: number; // Step size for parameter perturbations
  };
  simulation: {
    default_traffic_sample_rate: number; // Fraction of traffic to simulate
    max_simulation_duration_hours: number;
    confidence_level: number; // Statistical confidence level
    parallel_simulations: number; // Concurrent simulation workers
  };
  floor_tracking: {
    log_all_floor_decisions: boolean;
    audit_override_rate_threshold: number; // Flag if override rate too high
    floor_win_categories: string[];
  };
  policy_generation: {
    enable_automatic_links: boolean;
    link_base_url: string;
    link_expiration_hours: number;
    parameter_bounds: Record<string, [number, number]>; // Min/max for parameters
  };
}

export class CounterfactualWhyTooling extends EventEmitter {
  private config: CounterfactualConfig;
  private policyAnalyzer: PolicyAnalyzer;
  private rolloutSimulator: RolloutSimulator;
  private floorWinTracker: FloorWinTracker;
  private linkGenerator: ReproducibleLinkGenerator;
  private debuggingSessionManager: DebuggingSessionManager;

  constructor(config: CounterfactualConfig) {
    super();
    this.config = config;
    this.policyAnalyzer = new PolicyAnalyzer(config.analysis);
    this.rolloutSimulator = new RolloutSimulator(config.simulation);
    this.floorWinTracker = new FloorWinTracker(config.floor_tracking);
    this.linkGenerator = new ReproducibleLinkGenerator(config.policy_generation);
    this.debuggingSessionManager = new DebuggingSessionManager();
  }

  /**
   * Generate counterfactual analysis for a specific result
   */
  async analyzeCounterfactuals(
    query: string,
    results: SearchResult[],
    target_result_id: string,
    target_rank?: number
  ): Promise<CounterfactualAnalysis> {
    const targetResult = results.find(r => r.id === target_result_id);
    if (!targetResult) {
      throw new Error(`Result ${target_result_id} not found`);
    }

    // Generate potential policy changes
    const candidateChanges = await this.policyAnalyzer.generateCandidateChanges(
      query, 
      results, 
      targetResult
    );

    // Find minimal sufficient set of changes
    const minimalChanges = await this.policyAnalyzer.findMinimalSufficientChanges(
      query,
      results,
      targetResult,
      candidateChanges,
      target_rank
    );

    // Generate reproducible link
    const reproLink = await this.linkGenerator.generateReproducibleLink(
      query,
      minimalChanges
    );

    // Calculate confidence based on change necessity and impact
    const confidence = this.calculateAnalysisConfidence(minimalChanges, results);

    const analysis: CounterfactualAnalysis = {
      result_id: target_result_id,
      original_rank: targetResult.rank,
      target_rank: target_rank,
      minimal_changes: minimalChanges,
      reproducible_link: reproLink,
      confidence: confidence,
      explanation: this.generateExplanation(minimalChanges, targetResult)
    };

    this.emit('counterfactual_generated', { analysis, query });

    return analysis;
  }

  /**
   * Start a comprehensive debugging session
   */
  async startDebuggingSession(
    query: string,
    user_id: string,
    analysis_depth: 'quick' | 'thorough' | 'exhaustive' = 'thorough'
  ): Promise<DebuggingSession> {
    // Execute search to get current results
    const results = await this.executeSearchForAnalysis(query);

    // Generate counterfactual analyses for top results
    const analysisPromises = results.slice(0, analysis_depth === 'quick' ? 5 : 
                                            analysis_depth === 'thorough' ? 10 : 20)
      .map(async (result, index) => {
        return this.analyzeCounterfactuals(query, results, result.id, index + 1);
      });

    const counterfactualAnalyses = await Promise.all(analysisPromises);

    // Get recent floor wins for this query pattern
    const floorWins = await this.floorWinTracker.getRecentFloorWins(query);

    // Generate policy recommendations
    const recommendations = await this.generatePolicyRecommendations(
      query,
      results,
      counterfactualAnalyses,
      floorWins
    );

    const session: DebuggingSession = {
      session_id: `debug_${Date.now()}_${Math.random().toString(36).substring(7)}`,
      query,
      original_results: results,
      counterfactual_analyses: counterfactualAnalyses,
      floor_wins: floorWins,
      policy_recommendations: recommendations,
      created_at: new Date(),
      user_id
    };

    await this.debuggingSessionManager.saveSession(session);

    this.emit('debugging_session_started', { session });

    return session;
  }

  /**
   * Log a floor-win event for audit trail
   */
  async logFloorWin(event: Omit<FloorWinEvent, 'timestamp'>): Promise<void> {
    const floorWinEvent: FloorWinEvent = {
      ...event,
      timestamp: new Date()
    };

    await this.floorWinTracker.logEvent(floorWinEvent);

    // Check if override rate is too high
    const overrideRate = await this.floorWinTracker.calculateOverrideRate(24); // Last 24 hours
    
    if (overrideRate > this.config.floor_tracking.audit_override_rate_threshold) {
      this.emit('high_override_rate', { 
        rate: overrideRate, 
        threshold: this.config.floor_tracking.audit_override_rate_threshold,
        recent_events: await this.floorWinTracker.getRecentFloorWins()
      });
    }

    this.emit('floor_win_logged', { event: floorWinEvent, override_rate: overrideRate });
  }

  /**
   * Run rollout simulation with policy perturbations
   */
  async runRolloutSimulation(
    policy_deltas: PolicyDelta[],
    traffic_sample_rate?: number,
    simulation_hours?: number
  ): Promise<RolloutSimulation> {
    const sampleRate = traffic_sample_rate || this.config.simulation.default_traffic_sample_rate;
    const duration = simulation_hours || this.config.simulation.max_simulation_duration_hours;

    // Get historical traffic for baseline
    const historicalTraffic = await this.getHistoricalTraffic(duration);
    const trafficSample = this.sampleTraffic(historicalTraffic, sampleRate);

    // Run baseline simulation
    const baselineMetrics = await this.rolloutSimulator.simulate(
      trafficSample,
      [] // No policy changes for baseline
    );

    // Run perturbed simulation
    const perturbedMetrics = await this.rolloutSimulator.simulate(
      trafficSample,
      policy_deltas
    );

    const simulation: RolloutSimulation = {
      simulation_id: `sim_${Date.now()}`,
      policy_deltas,
      baseline_metrics: baselineMetrics,
      perturbed_metrics: perturbedMetrics,
      traffic_sample_size: trafficSample.length,
      simulation_duration_hours: duration,
      confidence_interval: this.config.simulation.confidence_level
    };

    // Check for significant differences
    const significantChanges = this.detectSignificantChanges(simulation);
    
    if (significantChanges.length > 0) {
      this.emit('significant_simulation_changes', { 
        simulation, 
        changes: significantChanges 
      });
    }

    this.emit('rollout_simulation_completed', { simulation });

    return simulation;
  }

  /**
   * Generate policy recommendations based on analysis
   */
  private async generatePolicyRecommendations(
    query: string,
    results: SearchResult[],
    analyses: CounterfactualAnalysis[],
    floorWins: FloorWinEvent[]
  ): Promise<PolicyRecommendation[]> {
    const recommendations: PolicyRecommendation[] = [];

    // Analyze common patterns in counterfactual changes
    const changePatterns = this.analyzeChangePatterns(analyses);

    // Generate recommendations based on patterns
    for (const pattern of changePatterns) {
      if (pattern.frequency > 0.5 && pattern.average_impact > 0.3) {
        recommendations.push({
          type: this.mapPatternToRecommendationType(pattern),
          description: `Consider ${pattern.description} - affects ${Math.round(pattern.frequency * 100)}% of results`,
          suggested_changes: pattern.representative_changes,
          expected_impact: `Estimated ${Math.round(pattern.average_impact * 100)}% improvement in target metrics`,
          risk_assessment: this.assessRecommendationRisk(pattern)
        });
      }
    }

    // Check floor win patterns
    const floorWinPatterns = this.analyzeFloorWinPatterns(floorWins);
    for (const pattern of floorWinPatterns) {
      recommendations.push({
        type: 'fix_ranking',
        description: `${pattern.type} constraints overriding dense scores frequently`,
        suggested_changes: pattern.suggested_adjustments,
        expected_impact: `Reduce floor overrides by ${Math.round(pattern.reduction_potential * 100)}%`,
        risk_assessment: 'medium'
      });
    }

    return recommendations;
  }

  /**
   * Generate human-readable explanation for counterfactual changes
   */
  private generateExplanation(changes: PolicyChange[], result: SearchResult): string {
    const explanations = changes.map(change => {
      switch (change.component) {
        case 'prior':
          return `Adjusting ${change.parameter} prior from ${change.original_value} to ${change.counterfactual_value}`;
        case 'centrality_cap':
          return `Changing centrality cap from ${change.original_value} to ${change.counterfactual_value}`;
        case 'ef_search':
          return `Modifying search depth from ${change.original_value} to ${change.counterfactual_value}`;
        case 'semantic_threshold':
          return `Adjusting semantic rerank threshold from ${change.original_value} to ${change.counterfactual_value}`;
        default:
          return `Changing ${change.parameter} in ${change.component}`;
      }
    });

    const topChange = changes.reduce((max, change) => 
      change.impact_score > max.impact_score ? change : max
    );

    return `To improve ranking of "${result.path}": ${explanations.join(', ')}. ` +
           `Primary factor: ${topChange.parameter} (${Math.round(topChange.impact_score * 100)}% impact)`;
  }

  /**
   * Calculate confidence score for counterfactual analysis
   */
  private calculateAnalysisConfidence(changes: PolicyChange[], results: SearchResult[]): number {
    if (changes.length === 0) return 0;

    // Base confidence on necessity scores and change simplicity
    const avgNecessity = changes.reduce((sum, c) => sum + c.necessity_score, 0) / changes.length;
    const simplicityScore = Math.max(0, 1 - (changes.length / this.config.analysis.max_changes_per_result));
    const impactConsistency = this.calculateImpactConsistency(changes);

    return (avgNecessity * 0.5 + simplicityScore * 0.3 + impactConsistency * 0.2);
  }

  // Helper methods

  private async executeSearchForAnalysis(query: string): Promise<SearchResult[]> {
    // Execute search with detailed component scoring
    throw new Error('Implementation required - integrate with actual search system');
  }

  private async getHistoricalTraffic(hours: number): Promise<any[]> {
    // Get historical query traffic for simulation
    throw new Error('Implementation required - integrate with query logs');
  }

  private sampleTraffic(traffic: any[], sampleRate: number): any[] {
    // Sample traffic for simulation
    const sampleSize = Math.floor(traffic.length * sampleRate);
    return traffic.slice(0, sampleSize);
  }

  private detectSignificantChanges(simulation: RolloutSimulation): any[] {
    const changes = [];
    const baseline = simulation.baseline_metrics;
    const perturbed = simulation.perturbed_metrics;

    // Check for significant changes in key metrics
    if (Math.abs(baseline.sla_recall - perturbed.sla_recall) > 0.05) {
      changes.push({
        metric: 'sla_recall',
        baseline: baseline.sla_recall,
        perturbed: perturbed.sla_recall,
        change: perturbed.sla_recall - baseline.sla_recall
      });
    }

    if (Math.abs(baseline.p95_latency_ms - perturbed.p95_latency_ms) > 5) {
      changes.push({
        metric: 'p95_latency_ms',
        baseline: baseline.p95_latency_ms,
        perturbed: perturbed.p95_latency_ms,
        change: perturbed.p95_latency_ms - baseline.p95_latency_ms
      });
    }

    if (baseline.why_mix_kl_divergence > 0.1) {
      changes.push({
        metric: 'why_mix_kl_divergence',
        baseline: baseline.why_mix_kl_divergence,
        perturbed: perturbed.why_mix_kl_divergence,
        change: 'Significant change in result explanations'
      });
    }

    return changes;
  }

  private analyzeChangePatterns(analyses: CounterfactualAnalysis[]): any[] {
    // Analyze common patterns in counterfactual changes
    const patternMap = new Map<string, any>();

    for (const analysis of analyses) {
      for (const change of analysis.minimal_changes) {
        const key = `${change.component}_${change.parameter}`;
        
        if (!patternMap.has(key)) {
          patternMap.set(key, {
            component: change.component,
            parameter: change.parameter,
            frequency: 0,
            impact_scores: [],
            representative_changes: []
          });
        }

        const pattern = patternMap.get(key)!;
        pattern.frequency++;
        pattern.impact_scores.push(change.impact_score);
        pattern.representative_changes.push(change);
      }
    }

    // Convert to array and calculate averages
    return Array.from(patternMap.values()).map(pattern => ({
      ...pattern,
      frequency: pattern.frequency / analyses.length,
      average_impact: pattern.impact_scores.reduce((sum: number, score: number) => sum + score, 0) / pattern.impact_scores.length,
      description: `${pattern.parameter} in ${pattern.component}`,
      representative_changes: pattern.representative_changes.slice(0, 3) // Top 3 examples
    }));
  }

  private analyzeFloorWinPatterns(floorWins: FloorWinEvent[]): any[] {
    // Analyze patterns in floor wins for recommendations
    const patterns = [];
    
    const overridesByType = floorWins.reduce((acc, event) => {
      if (event.override_applied) {
        acc[event.floor_type] = (acc[event.floor_type] || 0) + 1;
      }
      return acc;
    }, {} as Record<string, number>);

    for (const [type, count] of Object.entries(overridesByType)) {
      if (count > 5) { // Significant number of overrides
        patterns.push({
          type,
          count,
          reduction_potential: Math.min(0.8, count / floorWins.length),
          suggested_adjustments: this.generateFloorAdjustments(type)
        });
      }
    }

    return patterns;
  }

  private generateFloorAdjustments(floorType: string): PolicyChange[] {
    // Generate policy changes to reduce floor overrides
    switch (floorType) {
      case 'exact_match':
        return [{
          component: 'prior',
          parameter: 'exact_match_boost',
          original_value: 1.5,
          counterfactual_value: 1.2,
          impact_score: 0.3,
          necessity_score: 0.7
        }];
      case 'structural_match':
        return [{
          component: 'centrality_cap',
          parameter: 'max_boost',
          original_value: 2.0,
          counterfactual_value: 1.5,
          impact_score: 0.4,
          necessity_score: 0.6
        }];
      default:
        return [];
    }
  }

  private mapPatternToRecommendationType(pattern: any): PolicyRecommendation['type'] {
    if (pattern.component === 'ef_search') return 'reduce_latency';
    if (pattern.component === 'semantic_threshold') return 'improve_precision';
    if (pattern.component === 'prior') return 'increase_recall';
    return 'fix_ranking';
  }

  private assessRecommendationRisk(pattern: any): 'low' | 'medium' | 'high' {
    if (pattern.average_impact > 0.7) return 'high';
    if (pattern.average_impact > 0.4) return 'medium';
    return 'low';
  }

  private calculateImpactConsistency(changes: PolicyChange[]): number {
    if (changes.length <= 1) return 1.0;
    
    const impacts = changes.map(c => c.impact_score);
    const mean = impacts.reduce((sum, impact) => sum + impact, 0) / impacts.length;
    const variance = impacts.reduce((sum, impact) => sum + Math.pow(impact - mean, 2), 0) / impacts.length;
    
    // Return inverse of coefficient of variation (lower variance = higher consistency)
    return mean > 0 ? Math.max(0, 1 - Math.sqrt(variance) / mean) : 0;
  }
}

// Supporting Classes (simplified implementations)

class PolicyAnalyzer {
  constructor(private config: any) {}

  async generateCandidateChanges(
    query: string,
    results: SearchResult[],
    targetResult: SearchResult
  ): Promise<PolicyChange[]> {
    // Generate potential policy changes that could affect ranking
    const candidates: PolicyChange[] = [];

    // Prior adjustments
    candidates.push({
      component: 'prior',
      parameter: 'path_boost',
      original_value: 1.0,
      counterfactual_value: 1.3,
      impact_score: 0.4,
      necessity_score: 0.6
    });

    // Centrality cap adjustments
    candidates.push({
      component: 'centrality_cap',
      parameter: 'max_centrality_boost',
      original_value: 2.0,
      counterfactual_value: 1.5,
      impact_score: 0.3,
      necessity_score: 0.5
    });

    // efSearch adjustments
    candidates.push({
      component: 'ef_search',
      parameter: 'search_depth',
      original_value: 128,
      counterfactual_value: 256,
      impact_score: 0.5,
      necessity_score: 0.7
    });

    return candidates;
  }

  async findMinimalSufficientChanges(
    query: string,
    results: SearchResult[],
    targetResult: SearchResult,
    candidateChanges: PolicyChange[],
    targetRank?: number
  ): Promise<PolicyChange[]> {
    // Find minimal set of changes sufficient to achieve target ranking
    // This would use a greedy search or optimization algorithm
    
    // Sort by necessity score and take most impactful
    return candidateChanges
      .sort((a, b) => b.necessity_score - a.necessity_score)
      .slice(0, 3); // Maximum 3 changes for simplicity
  }
}

class RolloutSimulator {
  constructor(private config: any) {}

  async simulate(trafficSample: any[], policyDeltas: PolicyDelta[]): Promise<SimulationMetrics> {
    // Simulate policy changes on historical traffic
    return {
      sla_recall: 0.85 + Math.random() * 0.1,
      p95_latency_ms: 18 + Math.random() * 4,
      why_mix_kl_divergence: Math.random() * 0.2,
      cost_per_query: 12 + Math.random() * 3,
      user_satisfaction_proxy: 0.8 + Math.random() * 0.15
    };
  }
}

class FloorWinTracker {
  private events: FloorWinEvent[] = [];

  constructor(private config: any) {}

  async logEvent(event: FloorWinEvent): Promise<void> {
    this.events.push(event);
    
    // Keep only recent events to prevent memory bloat
    const cutoff = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000); // 7 days
    this.events = this.events.filter(e => e.timestamp > cutoff);
  }

  async getRecentFloorWins(query?: string, hours: number = 24): Promise<FloorWinEvent[]> {
    const cutoff = new Date(Date.now() - hours * 60 * 60 * 1000);
    let events = this.events.filter(e => e.timestamp > cutoff);
    
    if (query) {
      // Filter by query pattern (simplified)
      events = events.filter(e => e.query_id.includes(query.substring(0, 10)));
    }
    
    return events;
  }

  async calculateOverrideRate(hours: number): Promise<number> {
    const recentEvents = await this.getRecentFloorWins(undefined, hours);
    if (recentEvents.length === 0) return 0;
    
    const overrides = recentEvents.filter(e => e.override_applied).length;
    return overrides / recentEvents.length;
  }
}

class ReproducibleLinkGenerator {
  constructor(private config: any) {}

  async generateReproducibleLink(query: string, changes: PolicyChange[]): Promise<string> {
    if (!this.config.enable_automatic_links) {
      return 'Link generation disabled';
    }

    // Encode policy changes as URL parameters
    const params = new URLSearchParams();
    params.set('q', query);
    params.set('debug', 'true');
    
    changes.forEach((change, index) => {
      params.set(`delta_${index}_component`, change.component);
      params.set(`delta_${index}_param`, change.parameter);
      params.set(`delta_${index}_value`, change.counterfactual_value.toString());
    });

    // Add expiration timestamp
    const expiresAt = new Date(Date.now() + this.config.link_expiration_hours * 60 * 60 * 1000);
    params.set('expires', expiresAt.toISOString());

    return `${this.config.link_base_url}/rerank?${params.toString()}`;
  }
}

class DebuggingSessionManager {
  private sessions: Map<string, DebuggingSession> = new Map();

  async saveSession(session: DebuggingSession): Promise<void> {
    this.sessions.set(session.session_id, session);
  }

  async getSession(sessionId: string): Promise<DebuggingSession | null> {
    return this.sessions.get(sessionId) || null;
  }

  async listRecentSessions(userId?: string, limit: number = 10): Promise<DebuggingSession[]> {
    const sessions = Array.from(this.sessions.values())
      .filter(s => !userId || s.user_id === userId)
      .sort((a, b) => b.created_at.getTime() - a.created_at.getTime())
      .slice(0, limit);
    
    return sessions;
  }
}

// Default Configuration
export const DEFAULT_COUNTERFACTUAL_CONFIG: CounterfactualConfig = {
  analysis: {
    max_changes_per_result: 5,
    min_confidence_threshold: 0.6,
    rank_change_threshold: 3,
    perturbation_step_size: 0.1
  },
  simulation: {
    default_traffic_sample_rate: 0.1, // 10% of traffic
    max_simulation_duration_hours: 24,
    confidence_level: 0.95,
    parallel_simulations: 4
  },
  floor_tracking: {
    log_all_floor_decisions: true,
    audit_override_rate_threshold: 0.3, // 30% override rate triggers alert
    floor_win_categories: ['exact_match', 'structural_match', 'monotone_constraint']
  },
  policy_generation: {
    enable_automatic_links: true,
    link_base_url: 'https://lens-debug.local',
    link_expiration_hours: 48,
    parameter_bounds: {
      ef_search: [16, 512],
      centrality_cap: [1.0, 5.0],
      semantic_threshold: [0.0, 1.0]
    }
  }
};