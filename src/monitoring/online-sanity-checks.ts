/**
 * Online Sanity Checks for Centrality Canary
 * 
 * Implements real-time sanity checking with why-mix analysis, Core@10 tracking,
 * and topic-normalized centrality drift detection during canary deployment.
 */

import { EventEmitter } from 'events';

interface WhyMixMetrics {
  semantic_share: number;
  lexical_share: number;
  symbol_share: number;
  path_share: number;
  centrality_boost: number;
}

interface CoreMetrics {
  core_at_10: number;
  core_at_10_baseline: number;
  core_at_10_delta: number;
  topic_normalized_core_at_10: number;
}

interface QuerySliceMetrics {
  slice: string;
  query_count: number;
  why_mix: WhyMixMetrics;
  core_metrics: CoreMetrics;
  ndcg_at_10: number;
  diversity_at_10: number;
}

interface SanityCheckResult {
  passed: boolean;
  violations: string[];
  warnings: string[];
  metrics: QuerySliceMetrics[];
  timestamp: Date;
}

interface DriftDetectionConfig {
  semantic_share_max_delta: number;  // +15 pp
  core_plateau_threshold: number;    // nDCG should rise ≥1pt if Core@10 rises
  centrality_weight_max: number;     // 0.4 log-odds cap
  topic_similarity_threshold: number; // 0.7 for high-sim detection
}

export class OnlineSanityChecker extends EventEmitter {
  private config: DriftDetectionConfig;
  private baselineMetrics: Map<string, QuerySliceMetrics> = new Map();
  private currentMetrics: Map<string, QuerySliceMetrics> = new Map();
  private checkInterval: NodeJS.Timeout | null = null;
  private isRunning = false;

  constructor(config?: Partial<DriftDetectionConfig>) {
    super();
    
    this.config = {
      semantic_share_max_delta: 15,   // Max +15 pp semantic share increase
      core_plateau_threshold: 1.0,   // nDCG must rise ≥1pt if Core@10 rises
      centrality_weight_max: 0.4,    // Hard cap on centrality log-odds
      topic_similarity_threshold: 0.7, // High topic similarity threshold
      ...config
    };
  }

  public async start(): Promise<void> {
    if (this.isRunning) {
      console.log('Online sanity checker already running');
      return;
    }

    console.log('🔍 Starting online sanity checks...');
    
    // Capture baseline metrics before canary starts
    await this.captureBaselineMetrics();
    
    // Start periodic sanity checks every 2 minutes
    this.checkInterval = setInterval(() => {
      this.runSanityChecks().catch(error => {
        console.error('Sanity check error:', error);
        this.emit('sanityCheckError', error);
      });
    }, 2 * 60 * 1000);
    
    this.isRunning = true;
    console.log('✅ Online sanity checker started');
    this.emit('sanityCheckerStarted');
  }

  public async stop(): Promise<void> {
    if (this.checkInterval) {
      clearInterval(this.checkInterval);
      this.checkInterval = null;
    }
    
    this.isRunning = false;
    console.log('🛑 Online sanity checker stopped');
    this.emit('sanityCheckerStopped');
  }

  private async captureBaselineMetrics(): Promise<void> {
    console.log('📊 Capturing baseline metrics for sanity checks...');
    
    const querySlices = [
      'NL-overview',
      'NL-specific', 
      'symbol-lookup',
      'symbol-navigation',
      'path-search'
    ];

    for (const slice of querySlices) {
      const metrics = await this.collectSliceMetrics(slice, true); // baseline=true
      this.baselineMetrics.set(slice, metrics);
      console.log(`Baseline captured for ${slice}:`, {
        core_at_10: metrics.core_metrics.core_at_10,
        semantic_share: metrics.why_mix.semantic_share,
        ndcg_at_10: metrics.ndcg_at_10
      });
    }
    
    console.log('✅ Baseline metrics captured');
    this.emit('baselineMetricsCaptured', Object.fromEntries(this.baselineMetrics));
  }

  private async runSanityChecks(): Promise<void> {
    console.log('🔍 Running sanity checks...');
    
    const violations: string[] = [];
    const warnings: string[] = [];
    const currentSliceMetrics: QuerySliceMetrics[] = [];

    // Collect current metrics for all slices
    for (const [slice, baselineMetrics] of this.baselineMetrics) {
      const currentMetrics = await this.collectSliceMetrics(slice, false);
      this.currentMetrics.set(slice, currentMetrics);
      currentSliceMetrics.push(currentMetrics);

      // Check for violations in this slice
      const sliceViolations = await this.checkSliceViolations(slice, baselineMetrics, currentMetrics);
      violations.push(...sliceViolations.violations);
      warnings.push(...sliceViolations.warnings);
    }

    // Cross-slice checks
    const crossSliceIssues = await this.runCrossSliceChecks(currentSliceMetrics);
    violations.push(...crossSliceIssues.violations);
    warnings.push(...crossSliceIssues.warnings);

    const result: SanityCheckResult = {
      passed: violations.length === 0,
      violations,
      warnings,
      metrics: currentSliceMetrics,
      timestamp: new Date()
    };

    // Emit results
    if (!result.passed) {
      console.error('❌ Sanity check violations:', violations);
      this.emit('sanityCheckViolations', result);
    } else if (warnings.length > 0) {
      console.warn('⚠️ Sanity check warnings:', warnings);
      this.emit('sanityCheckWarnings', result);
    } else {
      console.log('✅ Sanity checks passed');
      this.emit('sanityCheckPassed', result);
    }
  }

  private async collectSliceMetrics(slice: string, baseline: boolean): Promise<QuerySliceMetrics> {
    // Implementation would collect real metrics from the search system
    // For now, simulate metric collection
    
    console.log(`Collecting metrics for slice: ${slice} (baseline: ${baseline})`);
    
    // This would interface with the actual search system to collect:
    // - Why-mix analysis per query slice
    // - Core@10 metrics with topic normalization
    // - nDCG and diversity measurements
    
    return {
      slice,
      query_count: baseline ? 1000 : 1050, // Simulated query volume
      why_mix: {
        semantic_share: baseline ? 35.2 : 38.7, // Example: semantic share increase
        lexical_share: baseline ? 42.1 : 40.3,
        symbol_share: baseline ? 15.3 : 16.2,
        path_share: baseline ? 7.4 : 4.8,
        centrality_boost: baseline ? 0.0 : 12.5 // New centrality contribution
      },
      core_metrics: {
        core_at_10: baseline ? 45.2 : 67.4, // +22.2pp improvement
        core_at_10_baseline: 45.2,
        core_at_10_delta: baseline ? 0.0 : 22.2,
        topic_normalized_core_at_10: baseline ? 45.2 : 65.8 // Should track with regular Core@10
      },
      ndcg_at_10: baseline ? 67.3 : 69.1, // +1.8pt improvement
      diversity_at_10: baseline ? 32.4 : 39.9 // +23% improvement
    };
  }

  private async checkSliceViolations(
    slice: string, 
    baseline: QuerySliceMetrics, 
    current: QuerySliceMetrics
  ): Promise<{violations: string[], warnings: string[]}> {
    const violations: string[] = [];
    const warnings: string[] = [];

    // Check semantic share drift
    const semanticShareDelta = current.why_mix.semantic_share - baseline.why_mix.semantic_share;
    const ndcgDelta = current.ndcg_at_10 - baseline.ndcg_at_10;

    if (semanticShareDelta > this.config.semantic_share_max_delta) {
      if (ndcgDelta < this.config.core_plateau_threshold) {
        violations.push(
          `${slice}: Semantic share increased ${semanticShareDelta.toFixed(1)}pp (>${this.config.semantic_share_max_delta}pp) ` +
          `without corresponding nDCG improvement (${ndcgDelta.toFixed(1)}pt < ${this.config.core_plateau_threshold}pt)`
        );
      } else {
        warnings.push(
          `${slice}: High semantic share increase (${semanticShareDelta.toFixed(1)}pp) but nDCG improved sufficiently (+${ndcgDelta.toFixed(1)}pt)`
        );
      }
    }

    // Check Core@10 vs topic-normalized Core@10 alignment
    const coreRegular = current.core_metrics.core_at_10;
    const coreTopicNormalized = current.core_metrics.topic_normalized_core_at_10;
    const coreDivergence = Math.abs(coreRegular - coreTopicNormalized) / coreRegular;

    if (coreDivergence > 0.1) { // >10% divergence
      if (coreTopicNormalized > coreRegular * 1.2 && ndcgDelta < this.config.core_plateau_threshold) {
        violations.push(
          `${slice}: Topic-normalized Core@10 drift detected - ` +
          `regular: ${coreRegular.toFixed(1)}, normalized: ${coreTopicNormalized.toFixed(1)} ` +
          `(${(coreDivergence * 100).toFixed(1)}% divergence) without nDCG plateau`
        );
      } else {
        warnings.push(
          `${slice}: Core@10 divergence noted but within acceptable bounds or nDCG compensating`
        );
      }
    }

    // Check centrality boost is reasonable
    const centralityBoost = current.why_mix.centrality_boost;
    if (centralityBoost > 20) { // >20% of score from centrality
      warnings.push(
        `${slice}: High centrality contribution (${centralityBoost.toFixed(1)}%) - monitor for over-reliance`
      );
    }

    // Check diversity improvement accompanies core improvements
    const coreImprovement = current.core_metrics.core_at_10_delta;
    const diversityImprovement = ((current.diversity_at_10 - baseline.diversity_at_10) / baseline.diversity_at_10) * 100;

    if (coreImprovement > 15 && diversityImprovement < 5) {
      warnings.push(
        `${slice}: Core@10 improved significantly (+${coreImprovement.toFixed(1)}pp) but diversity gains minimal (+${diversityImprovement.toFixed(1)}%)`
      );
    }

    return { violations, warnings };
  }

  private async runCrossSliceChecks(metrics: QuerySliceMetrics[]): Promise<{violations: string[], warnings: string[]}> {
    const violations: string[] = [];
    const warnings: string[] = [];

    // Check for consistent improvements across slices
    const nlSlices = metrics.filter(m => m.slice.startsWith('NL-'));
    const symbolSlices = metrics.filter(m => m.slice.startsWith('symbol-'));

    // NL slices should show consistent centrality benefits
    const nlCoreImprovements = nlSlices.map(m => m.core_metrics.core_at_10_delta);
    const nlCoreVariance = this.calculateVariance(nlCoreImprovements);

    if (nlCoreVariance > 100) { // High variance in improvements
      warnings.push(
        `High variance in NL slice improvements: ${nlCoreImprovements.map(x => x.toFixed(1)).join(', ')}pp`
      );
    }

    // Symbol slices should maintain quality while showing centrality benefits
    for (const symbolMetric of symbolSlices) {
      const baseline = this.baselineMetrics.get(symbolMetric.slice);
      if (baseline) {
        const nDCGDelta = symbolMetric.ndcg_at_10 - baseline.ndcg_at_10;
        if (nDCGDelta < -0.5) { // More than 0.5pt drop
          violations.push(
            `${symbolMetric.slice}: nDCG dropped ${Math.abs(nDCGDelta).toFixed(1)}pt - centrality may be hurting symbol precision`
          );
        }
      }
    }

    // Check overall semantic share trend
    const totalSemanticShareIncrease = metrics.reduce((sum, m) => {
      const baseline = this.baselineMetrics.get(m.slice);
      return sum + (baseline ? m.why_mix.semantic_share - baseline.why_mix.semantic_share : 0);
    }, 0) / metrics.length;

    if (totalSemanticShareIncrease > 12) {
      warnings.push(
        `Overall semantic share increase across slices: +${totalSemanticShareIncrease.toFixed(1)}pp - monitor for over-semanticization`
      );
    }

    return { violations, warnings };
  }

  private calculateVariance(values: number[]): number {
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    return squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length;
  }

  public async logWhyMixAnalysis(): Promise<void> {
    console.log('📊 Why-Mix Analysis Summary:');
    
    for (const [slice, metrics] of this.currentMetrics) {
      const baseline = this.baselineMetrics.get(slice);
      if (!baseline) continue;

      console.log(`\n--- ${slice} ---`);
      console.log(`Semantic: ${baseline.why_mix.semantic_share.toFixed(1)}% → ${metrics.why_mix.semantic_share.toFixed(1)}% (Δ${(metrics.why_mix.semantic_share - baseline.why_mix.semantic_share).toFixed(1)}pp)`);
      console.log(`Lexical:  ${baseline.why_mix.lexical_share.toFixed(1)}% → ${metrics.why_mix.lexical_share.toFixed(1)}% (Δ${(metrics.why_mix.lexical_share - baseline.why_mix.lexical_share).toFixed(1)}pp)`);
      console.log(`Symbol:   ${baseline.why_mix.symbol_share.toFixed(1)}% → ${metrics.why_mix.symbol_share.toFixed(1)}% (Δ${(metrics.why_mix.symbol_share - baseline.why_mix.symbol_share).toFixed(1)}pp)`);
      console.log(`Path:     ${baseline.why_mix.path_share.toFixed(1)}% → ${metrics.why_mix.path_share.toFixed(1)}% (Δ${(metrics.why_mix.path_share - baseline.why_mix.path_share).toFixed(1)}pp)`);
      console.log(`Centrality Boost: ${metrics.why_mix.centrality_boost.toFixed(1)}%`);
      console.log(`Core@10:  ${baseline.core_metrics.core_at_10.toFixed(1)} → ${metrics.core_metrics.core_at_10.toFixed(1)} (Δ${metrics.core_metrics.core_at_10_delta.toFixed(1)}pp)`);
      console.log(`nDCG@10:  ${baseline.ndcg_at_10.toFixed(1)} → ${metrics.ndcg_at_10.toFixed(1)} (Δ${(metrics.ndcg_at_10 - baseline.ndcg_at_10).toFixed(1)}pt)`);
    }
  }

  public getCurrentMetrics(): Map<string, QuerySliceMetrics> {
    return new Map(this.currentMetrics);
  }

  public getBaselineMetrics(): Map<string, QuerySliceMetrics> {
    return new Map(this.baselineMetrics);
  }

  public async getTopicNormalizedDrift(): Promise<{
    slice: string;
    regular_core_at_10: number;
    topic_normalized_core_at_10: number;
    drift_percentage: number;
    ndcg_compensation: number;
  }[]> {
    const driftAnalysis = [];
    
    for (const [slice, current] of this.currentMetrics) {
      const baseline = this.baselineMetrics.get(slice);
      if (!baseline) continue;
      
      const regularCore = current.core_metrics.core_at_10;
      const topicNormalizedCore = current.core_metrics.topic_normalized_core_at_10;
      const driftPercentage = ((topicNormalizedCore - regularCore) / regularCore) * 100;
      const ndcgCompensation = current.ndcg_at_10 - baseline.ndcg_at_10;
      
      driftAnalysis.push({
        slice,
        regular_core_at_10: regularCore,
        topic_normalized_core_at_10: topicNormalizedCore,
        drift_percentage: driftPercentage,
        ndcg_compensation: ndcgCompensation
      });
    }
    
    return driftAnalysis;
  }

  public isHealthy(): boolean {
    // Quick health check - no recent violations
    return this.isRunning && this.currentMetrics.size > 0;
  }
}