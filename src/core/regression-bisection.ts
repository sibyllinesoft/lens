/**
 * Regression Bisection Harness - Factorial Debugging System
 * 
 * Auto-run factorial delta-debugger when gates trip (SLA-Recall@50, p99/p95, ECE).
 * Toggle features/params in gray-box tree (router thresholds, priors, caches, ANN knobs).
 * Stop when isolating single culprit knob with ≥X% effect (Cohen's d).
 * Bind culprit to config_fingerprint.json and block promotion until fix reproduces.
 * Integrate with A/A shadow + counterfactual replay.
 * 
 * Gate: culprit isolation ≤30 min wall-clock, automatic rollback with diff summary
 */

import { LensTracer } from '../telemetry/tracer.js';
import type { SearchHit } from './span_resolver/types.js';

export interface RegressionAlert {
  id: string;
  alert_type: 'sla_recall_50' | 'p99_latency' | 'p95_latency' | 'ece_drift' | 'custom';
  metric_name: string;
  current_value: number;
  baseline_value: number;
  threshold_value: number;
  deviation_percent: number;
  confidence: number;            // Statistical significance
  triggered_at: Date;
  time_window: string;          // e.g., "last_1h", "last_24h"
  sample_size: number;
}

export interface ConfigurationKnob {
  id: string;
  name: string;
  category: 'router_threshold' | 'prior' | 'cache' | 'ann_knob' | 'rerank_param';
  current_value: any;
  default_value: any;
  possible_values: any[];       // Discrete values to try
  value_type: 'boolean' | 'integer' | 'float' | 'string' | 'enum';
  description: string;
  impact_scope: string[];       // Which metrics this knob affects
  dependencies: string[];       // Other knobs this depends on
  risk_level: 'low' | 'medium' | 'high'; // Risk of toggling
}

export interface FactorialExperiment {
  experiment_id: string;
  alert_id: string;
  knob_combinations: KnobCombination[];
  baseline_config: Map<string, any>;
  traffic_split: number;        // Percentage of traffic for each combination
  duration_minutes: number;
  status: 'running' | 'completed' | 'failed' | 'timeout';
  results: ExperimentResult[];
  culprit_identified: boolean;
  culprit_knobs: string[];      // Knobs identified as causing regression
  effect_size: number;          // Cohen's d
  started_at: Date;
  completed_at?: Date;
}

export interface KnobCombination {
  combination_id: string;
  knob_settings: Map<string, any>;
  is_baseline: boolean;
  traffic_allocation: number;
  metrics: CombinationMetrics;
}

export interface CombinationMetrics {
  queries_processed: number;
  avg_recall_at_50: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  error_rate: number;
  ece_score: number;            // Expected Calibration Error
  timestamp: Date;
}

export interface ExperimentResult {
  combination_id: string;
  metric_deltas: Map<string, number>; // Difference from baseline
  statistical_significance: number;
  effect_size: number;          // Cohen's d
  confidence_interval: [number, number];
  sample_size: number;
}

export interface BisectionConfig {
  max_experiment_duration_minutes: number; // ≤30 per gate requirement
  min_effect_size_cohens_d: number;        // Minimum effect size to consider
  statistical_significance_threshold: number; // p-value threshold
  max_knobs_per_experiment: number;         // Limit factorial explosion
  traffic_split_percent: number;            // % of traffic for experiments
  shadow_traffic_enabled: boolean;          // Use A/A shadow testing
  auto_rollback_enabled: boolean;
  rollback_threshold_effect_size: number;
}

export interface RollbackAction {
  rollback_id: string;
  triggered_by: string;         // experiment_id or manual
  culprit_knobs: string[];
  rollback_config: Map<string, any>;
  executed_at: Date;
  verification_metrics: Map<string, number>;
  success: boolean;
  diff_summary: string;
}

export class RegressionBisectionHarness {
  private activeExperiments = new Map<string, FactorialExperiment>();
  private configKnobs = new Map<string, ConfigurationKnob>();
  private alertHistory: RegressionAlert[] = [];
  private rollbackHistory: RollbackAction[] = [];
  private shadowTrafficSplitter: ShadowTrafficSplitter;

  constructor(
    private config: BisectionConfig = {
      max_experiment_duration_minutes: 30,
      min_effect_size_cohens_d: 0.5,    // Medium effect size
      statistical_significance_threshold: 0.05, // p < 0.05
      max_knobs_per_experiment: 8,       // Manageable factorial size
      traffic_split_percent: 10,         // 10% for experiments
      shadow_traffic_enabled: true,
      auto_rollback_enabled: true,
      rollback_threshold_effect_size: 1.0, // Large effect triggers rollback
    }
  ) {
    this.shadowTrafficSplitter = new ShadowTrafficSplitter();
    this.initializeConfigKnobs();
  }

  /**
   * Trigger regression analysis when gates trip
   */
  async triggerRegressionAnalysis(alert: RegressionAlert): Promise<string> {
    const span = LensTracer.createChildSpan('trigger_regression_analysis', {
      'alert.id': alert.id,
      'alert.type': alert.alert_type,
      'alert.deviation_percent': alert.deviation_percent,
    });

    try {
      // Record alert
      this.alertHistory.push(alert);

      // Check if we already have an active experiment for this alert type
      const existingExperiment = Array.from(this.activeExperiments.values())
        .find(exp => exp.alert_id === alert.id && exp.status === 'running');

      if (existingExperiment) {
        return existingExperiment.experiment_id;
      }

      // Select relevant configuration knobs based on alert type
      const relevantKnobs = this.selectRelevantKnobs(alert);

      // Generate factorial experiment design
      const experiment = await this.designFactorialExperiment(alert, relevantKnobs);

      // Start experiment
      await this.startExperiment(experiment);

      span.setAttributes({
        'experiment.id': experiment.experiment_id,
        'knobs.count': relevantKnobs.length,
        'combinations.count': experiment.knob_combinations.length,
        success: true
      });

      return experiment.experiment_id;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Design factorial experiment to isolate culprit knobs
   */
  private async designFactorialExperiment(
    alert: RegressionAlert,
    relevantKnobs: ConfigurationKnob[]
  ): Promise<FactorialExperiment> {
    const experimentId = `bisect_${alert.id}_${Date.now()}`;

    // Limit knobs to prevent factorial explosion
    const knobsToTest = relevantKnobs
      .sort((a, b) => {
        // Prioritize by risk level and impact scope
        const aScore = (a.risk_level === 'high' ? 3 : a.risk_level === 'medium' ? 2 : 1) * a.impact_scope.length;
        const bScore = (b.risk_level === 'high' ? 3 : b.risk_level === 'medium' ? 2 : 1) * b.impact_scope.length;
        return bScore - aScore;
      })
      .slice(0, this.config.max_knobs_per_experiment);

    // Generate factorial combinations (2^n for boolean knobs, limited for others)
    const combinations = this.generateKnobCombinations(knobsToTest);

    const experiment: FactorialExperiment = {
      experiment_id: experimentId,
      alert_id: alert.id,
      knob_combinations: combinations,
      baseline_config: this.getCurrentConfig(),
      traffic_split: this.config.traffic_split_percent,
      duration_minutes: this.config.max_experiment_duration_minutes,
      status: 'running',
      results: [],
      culprit_identified: false,
      culprit_knobs: [],
      effect_size: 0,
      started_at: new Date()
    };

    this.activeExperiments.set(experimentId, experiment);
    return experiment;
  }

  /**
   * Generate knob combinations for factorial experiment
   */
  private generateKnobCombinations(knobs: ConfigurationKnob[]): KnobCombination[] {
    const combinations: KnobCombination[] = [];

    // Start with baseline (all current values)
    const baselineSettings = new Map<string, any>();
    for (const knob of knobs) {
      baselineSettings.set(knob.id, knob.current_value);
    }

    combinations.push({
      combination_id: 'baseline',
      knob_settings: baselineSettings,
      is_baseline: true,
      traffic_allocation: 1.0 / (Math.pow(2, knobs.length)), // Equal split
      metrics: this.createEmptyMetrics()
    });

    // Generate factorial combinations
    const numCombinations = Math.min(Math.pow(2, knobs.length), 32); // Cap at 32 combinations
    
    for (let i = 1; i < numCombinations; i++) {
      const settings = new Map<string, any>();
      
      for (let j = 0; j < knobs.length; j++) {
        const knob = knobs[j];
        const useBinary = (i >> j) & 1;
        
        if (useBinary && knob.possible_values.length >= 2) {
          // Use alternative value
          const altValue = knob.possible_values.find(v => v !== knob.current_value) || knob.default_value;
          settings.set(knob.id, altValue);
        } else {
          // Use current value
          settings.set(knob.id, knob.current_value);
        }
      }

      combinations.push({
        combination_id: `combo_${i}`,
        knob_settings: settings,
        is_baseline: false,
        traffic_allocation: 1.0 / numCombinations,
        metrics: this.createEmptyMetrics()
      });
    }

    return combinations;
  }

  /**
   * Start factorial experiment with traffic splitting
   */
  private async startExperiment(experiment: FactorialExperiment): Promise<void> {
    const span = LensTracer.createChildSpan('start_factorial_experiment', {
      'experiment.id': experiment.experiment_id,
      'combinations.count': experiment.knob_combinations.length,
      'duration.minutes': experiment.duration_minutes,
    });

    try {
      // Configure shadow traffic splitting
      if (this.config.shadow_traffic_enabled) {
        await this.shadowTrafficSplitter.configureExperiment(
          experiment.experiment_id,
          experiment.knob_combinations
        );
      }

      // Schedule experiment termination
      setTimeout(async () => {
        await this.completeExperiment(experiment.experiment_id);
      }, experiment.duration_minutes * 60 * 1000);

      // Start monitoring metrics collection
      this.startMetricsCollection(experiment);

      span.setAttributes({
        'shadow.enabled': this.config.shadow_traffic_enabled,
        'auto_termination.scheduled': true,
        success: true
      });

    } catch (error) {
      experiment.status = 'failed';
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Start collecting metrics for all combinations
   */
  private startMetricsCollection(experiment: FactorialExperiment): void {
    const metricsInterval = setInterval(async () => {
      if (experiment.status !== 'running') {
        clearInterval(metricsInterval);
        return;
      }

      // Collect metrics for each combination
      for (const combination of experiment.knob_combinations) {
        const metrics = await this.collectCombinationMetrics(
          experiment.experiment_id,
          combination.combination_id
        );
        
        if (metrics) {
          combination.metrics = metrics;
        }
      }

      // Check for early termination if strong effect detected
      const preliminaryResults = this.analyzePreliminaryResults(experiment);
      if (preliminaryResults.strongEffectDetected) {
        await this.completeExperiment(experiment.experiment_id);
      }

    }, 30 * 1000); // Collect every 30 seconds
  }

  /**
   * Complete experiment and analyze results
   */
  async completeExperiment(experimentId: string): Promise<void> {
    const experiment = this.activeExperiments.get(experimentId);
    if (!experiment || experiment.status !== 'running') {
      return;
    }

    const span = LensTracer.createChildSpan('complete_experiment', {
      'experiment.id': experimentId,
      'duration.actual_minutes': (Date.now() - experiment.started_at.getTime()) / (1000 * 60),
    });

    try {
      experiment.status = 'completed';
      experiment.completed_at = new Date();

      // Analyze factorial results to identify culprit knobs
      const analysis = await this.performFactorialAnalysis(experiment);

      experiment.results = analysis.results;
      experiment.culprit_identified = analysis.culpritIdentified;
      experiment.culprit_knobs = analysis.culpritKnobs;
      experiment.effect_size = analysis.maxEffectSize;

      // Stop traffic splitting
      if (this.config.shadow_traffic_enabled) {
        await this.shadowTrafficSplitter.stopExperiment(experimentId);
      }

      // Trigger rollback if culprit identified and effect size is large
      if (analysis.culpritIdentified && analysis.maxEffectSize >= this.config.rollback_threshold_effect_size) {
        if (this.config.auto_rollback_enabled) {
          await this.triggerAutoRollback(experiment, analysis);
        }
      }

      // Update config fingerprint
      await this.updateConfigFingerprint(experiment, analysis);

      span.setAttributes({
        'culprit.identified': analysis.culpritIdentified,
        'culprit.knobs': analysis.culpritKnobs.join(','),
        'effect.size': analysis.maxEffectSize,
        'rollback.triggered': analysis.maxEffectSize >= this.config.rollback_threshold_effect_size,
        success: true
      });

    } catch (error) {
      experiment.status = 'failed';
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Perform factorial analysis to identify culprit knobs
   */
  private async performFactorialAnalysis(
    experiment: FactorialExperiment
  ): Promise<FactorialAnalysisResult> {
    const baseline = experiment.knob_combinations.find(c => c.is_baseline);
    if (!baseline) {
      throw new Error('No baseline combination found');
    }

    const results: ExperimentResult[] = [];
    let maxEffectSize = 0;
    const culpritKnobs: string[] = [];

    // Analyze each combination against baseline
    for (const combination of experiment.knob_combinations) {
      if (combination.is_baseline) continue;

      const result = this.compareCombinationToBaseline(combination, baseline);
      results.push(result);

      // Check if this combination shows significant regression
      if (result.statistical_significance < this.config.statistical_significance_threshold &&
          Math.abs(result.effect_size) >= this.config.min_effect_size_cohens_d) {
        
        maxEffectSize = Math.max(maxEffectSize, Math.abs(result.effect_size));

        // Identify which knobs are different from baseline
        for (const [knobId, value] of combination.knob_settings) {
          const baselineValue = baseline.knob_settings.get(knobId);
          if (value !== baselineValue && !culpritKnobs.includes(knobId)) {
            culpritKnobs.push(knobId);
          }
        }
      }
    }

    // Refine culprit identification using factorial analysis
    const refinedCulprits = this.refineCulpritIdentification(experiment, results, culpritKnobs);

    return {
      results,
      culpritIdentified: refinedCulprits.length > 0 && maxEffectSize >= this.config.min_effect_size_cohens_d,
      culpritKnobs: refinedCulprits,
      maxEffectSize,
      analysis: {
        total_combinations: experiment.knob_combinations.length,
        significant_effects: results.filter(r => r.statistical_significance < this.config.statistical_significance_threshold).length,
        largest_effect_combination: results.reduce((max, r) => 
          Math.abs(r.effect_size) > Math.abs(max.effect_size) ? r : max, results[0])?.combination_id
      }
    };
  }

  /**
   * Refine culprit identification using factorial analysis techniques
   */
  private refineCulpritIdentification(
    experiment: FactorialExperiment,
    results: ExperimentResult[],
    candidateCulprits: string[]
  ): string[] {
    // Implement more sophisticated factorial analysis
    // For now, return candidates that appear in multiple significant combinations
    const knobEffects = new Map<string, number[]>();

    // Collect effect sizes for each knob
    for (const result of results) {
      if (result.statistical_significance < this.config.statistical_significance_threshold) {
        const combination = experiment.knob_combinations.find(c => c.combination_id === result.combination_id);
        if (combination) {
          for (const [knobId] of combination.knob_settings) {
            if (candidateCulprits.includes(knobId)) {
              const effects = knobEffects.get(knobId) || [];
              effects.push(result.effect_size);
              knobEffects.set(knobId, effects);
            }
          }
        }
      }
    }

    // Identify knobs with consistent large effects
    const refinedCulprits: string[] = [];
    for (const [knobId, effects] of knobEffects) {
      const avgEffect = effects.reduce((sum, e) => sum + Math.abs(e), 0) / effects.length;
      if (avgEffect >= this.config.min_effect_size_cohens_d && effects.length >= 2) {
        refinedCulprits.push(knobId);
      }
    }

    return refinedCulprits;
  }

  /**
   * Trigger automatic rollback based on experiment results
   */
  private async triggerAutoRollback(
    experiment: FactorialExperiment,
    analysis: FactorialAnalysisResult
  ): Promise<void> {
    const rollbackId = `rollback_${experiment.experiment_id}_${Date.now()}`;

    const rollbackConfig = new Map<string, any>();
    for (const knobId of analysis.culpritKnobs) {
      const knob = this.configKnobs.get(knobId);
      if (knob) {
        rollbackConfig.set(knobId, knob.default_value); // Rollback to default
      }
    }

    const rollback: RollbackAction = {
      rollback_id: rollbackId,
      triggered_by: experiment.experiment_id,
      culprit_knobs: analysis.culpritKnobs,
      rollback_config: rollbackConfig,
      executed_at: new Date(),
      verification_metrics: new Map(),
      success: false,
      diff_summary: this.generateDiffSummary(analysis.culpritKnobs, rollbackConfig)
    };

    try {
      // Apply rollback configuration
      await this.applyConfigurationChanges(rollbackConfig);

      // Verify rollback effectiveness
      await this.verifyRollback(rollback);

      rollback.success = true;
      console.log(`Auto-rollback ${rollbackId} completed successfully`);

    } catch (error) {
      console.error(`Auto-rollback ${rollbackId} failed:`, error);
      rollback.success = false;
    }

    this.rollbackHistory.push(rollback);
  }

  /**
   * Update config fingerprint with experiment results
   */
  private async updateConfigFingerprint(
    experiment: FactorialExperiment,
    analysis: FactorialAnalysisResult
  ): Promise<void> {
    const fingerprint = {
      experiment_id: experiment.experiment_id,
      culprit_knobs: analysis.culpritKnobs,
      effect_size: analysis.maxEffectSize,
      timestamp: new Date().toISOString(),
      promotion_blocked: analysis.culpritIdentified && analysis.maxEffectSize >= this.config.rollback_threshold_effect_size
    };

    // Write to config_fingerprint.json (placeholder - would write to actual file)
    console.log('Config fingerprint updated:', fingerprint);
  }

  // Utility and helper methods

  private initializeConfigKnobs(): void {
    // Initialize with comprehensive set of configuration knobs
    const knobs: ConfigurationKnob[] = [
      {
        id: 'fuzzy_distance_threshold',
        name: 'Fuzzy Distance Threshold',
        category: 'router_threshold',
        current_value: 2,
        default_value: 2,
        possible_values: [0, 1, 2],
        value_type: 'integer',
        description: 'Maximum edit distance for fuzzy matching',
        impact_scope: ['recall', 'precision', 'latency'],
        dependencies: [],
        risk_level: 'medium'
      },
      {
        id: 'symbol_centrality_prior',
        name: 'Symbol Centrality Prior Weight',
        category: 'prior',
        current_value: 0.3,
        default_value: 0.3,
        possible_values: [0.1, 0.3, 0.5, 0.7],
        value_type: 'float',
        description: 'Weight for symbol centrality in scoring',
        impact_scope: ['ranking', 'recall'],
        dependencies: [],
        risk_level: 'low'
      },
      {
        id: 'semantic_cache_enabled',
        name: 'Semantic Cache Enabled',
        category: 'cache',
        current_value: true,
        default_value: true,
        possible_values: [true, false],
        value_type: 'boolean',
        description: 'Enable semantic result caching',
        impact_scope: ['latency', 'consistency'],
        dependencies: [],
        risk_level: 'high'
      },
      {
        id: 'hnsw_ef_search',
        name: 'HNSW efSearch Parameter',
        category: 'ann_knob',
        current_value: 50,
        default_value: 50,
        possible_values: [20, 50, 100, 200],
        value_type: 'integer',
        description: 'HNSW search parameter for ANN queries',
        impact_scope: ['recall', 'latency'],
        dependencies: [],
        risk_level: 'medium'
      },
      {
        id: 'rerank_top_k',
        name: 'Rerank Top-K',
        category: 'rerank_param',
        current_value: 100,
        default_value: 100,
        possible_values: [50, 100, 200],
        value_type: 'integer',
        description: 'Number of candidates to rerank',
        impact_scope: ['quality', 'latency'],
        dependencies: [],
        risk_level: 'low'
      }
    ];

    for (const knob of knobs) {
      this.configKnobs.set(knob.id, knob);
    }
  }

  private selectRelevantKnobs(alert: RegressionAlert): ConfigurationKnob[] {
    const relevantKnobs: ConfigurationKnob[] = [];

    for (const [_, knob] of this.configKnobs) {
      // Select knobs based on alert type and impact scope
      if (this.isKnobRelevantToAlert(knob, alert)) {
        relevantKnobs.push(knob);
      }
    }

    return relevantKnobs;
  }

  private isKnobRelevantToAlert(knob: ConfigurationKnob, alert: RegressionAlert): boolean {
    switch (alert.alert_type) {
      case 'sla_recall_50':
        return knob.impact_scope.includes('recall') || knob.impact_scope.includes('ranking');
      case 'p99_latency':
      case 'p95_latency':
        return knob.impact_scope.includes('latency');
      case 'ece_drift':
        return knob.impact_scope.includes('quality') || knob.impact_scope.includes('consistency');
      default:
        return true; // Include all knobs for unknown alert types
    }
  }

  private getCurrentConfig(): Map<string, any> {
    const config = new Map<string, any>();
    for (const [id, knob] of this.configKnobs) {
      config.set(id, knob.current_value);
    }
    return config;
  }

  private createEmptyMetrics(): CombinationMetrics {
    return {
      queries_processed: 0,
      avg_recall_at_50: 0,
      p95_latency_ms: 0,
      p99_latency_ms: 0,
      error_rate: 0,
      ece_score: 0,
      timestamp: new Date()
    };
  }

  private async collectCombinationMetrics(
    experimentId: string,
    combinationId: string
  ): Promise<CombinationMetrics | null> {
    // Placeholder for actual metrics collection
    // Would integrate with telemetry system
    return null;
  }

  private analyzePreliminaryResults(experiment: FactorialExperiment): { strongEffectDetected: boolean } {
    // Placeholder for early termination logic
    return { strongEffectDetected: false };
  }

  private compareCombinationToBaseline(
    combination: KnobCombination,
    baseline: KnobCombination
  ): ExperimentResult {
    // Placeholder for statistical comparison
    const mockResult: ExperimentResult = {
      combination_id: combination.combination_id,
      metric_deltas: new Map([
        ['recall_at_50', combination.metrics.avg_recall_at_50 - baseline.metrics.avg_recall_at_50],
        ['p99_latency', combination.metrics.p99_latency_ms - baseline.metrics.p99_latency_ms]
      ]),
      statistical_significance: 0.03, // Mock p-value
      effect_size: this.calculateCohensD(combination.metrics, baseline.metrics),
      confidence_interval: [-0.2, 0.8],
      sample_size: combination.metrics.queries_processed
    };

    return mockResult;
  }

  private calculateCohensD(metrics1: CombinationMetrics, metrics2: CombinationMetrics): number {
    // Simplified Cohen's d calculation
    const diff = metrics1.avg_recall_at_50 - metrics2.avg_recall_at_50;
    const pooledStd = 0.1; // Placeholder - would calculate actual pooled standard deviation
    return diff / pooledStd;
  }

  private generateDiffSummary(culpritKnobs: string[], rollbackConfig: Map<string, any>): string {
    const changes: string[] = [];
    
    for (const knobId of culpritKnobs) {
      const knob = this.configKnobs.get(knobId);
      const newValue = rollbackConfig.get(knobId);
      
      if (knob) {
        changes.push(`${knob.name}: ${knob.current_value} → ${newValue}`);
      }
    }

    return `Rollback changes:\n${changes.join('\n')}`;
  }

  private async applyConfigurationChanges(config: Map<string, any>): Promise<void> {
    // Placeholder for configuration application
    for (const [knobId, value] of config) {
      const knob = this.configKnobs.get(knobId);
      if (knob) {
        knob.current_value = value;
      }
    }
  }

  private async verifyRollback(rollback: RollbackAction): Promise<void> {
    // Placeholder for rollback verification
    // Would monitor metrics for a period after rollback
    rollback.verification_metrics.set('recall_improvement', 0.05);
    rollback.verification_metrics.set('latency_improvement', -10.0);
  }

  /**
   * Get performance metrics and status
   */
  getPerformanceMetrics() {
    const activeCount = Array.from(this.activeExperiments.values()).filter(e => e.status === 'running').length;
    const completedExperiments = Array.from(this.activeExperiments.values()).filter(e => e.status === 'completed');
    
    const avgDurationMs = completedExperiments.length > 0 
      ? completedExperiments.reduce((sum, exp) => {
          const duration = exp.completed_at ? 
            exp.completed_at.getTime() - exp.started_at.getTime() : 0;
          return sum + duration;
        }, 0) / completedExperiments.length
      : 0;

    return {
      active_experiments: activeCount,
      completed_experiments: completedExperiments.length,
      avg_experiment_duration_ms: avgDurationMs,
      successful_culprit_identifications: completedExperiments.filter(e => e.culprit_identified).length,
      rollbacks_executed: this.rollbackHistory.length,
      alerts_processed: this.alertHistory.length,
      config_knobs_managed: this.configKnobs.size,
    };
  }
}

// Supporting classes and interfaces

class ShadowTrafficSplitter {
  async configureExperiment(experimentId: string, combinations: KnobCombination[]): Promise<void> {
    // Placeholder for shadow traffic configuration
  }

  async stopExperiment(experimentId: string): Promise<void> {
    // Placeholder for stopping traffic split
  }
}

interface FactorialAnalysisResult {
  results: ExperimentResult[];
  culpritIdentified: boolean;
  culpritKnobs: string[];
  maxEffectSize: number;
  analysis: {
    total_combinations: number;
    significant_effects: number;
    largest_effect_combination: string;
  };
}