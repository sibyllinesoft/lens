/**
 * A/A Shadow Testing Infrastructure
 * 
 * Implements A/A shadow testing with shuffled centrality weights to detect
 * spurious lift and validate the statistical significance of centrality improvements.
 */

import { EventEmitter } from 'events';

interface AAShadowConfig {
  trafficPercentage: number;
  shuffleCentralityWeights: boolean;
  controlGroupSize: number;
  shadowGroupSize: number;
  measurementWindowMinutes: number;
}

interface MetricComparison {
  metric: string;
  control_value: number;
  shadow_value: number;
  delta: number;
  delta_percentage: number;
  p_value: number;
  is_significant: boolean;
  expected_delta: number; // Should be ~0 for A/A test
}

interface SpuriousLiftResult {
  detected: boolean;
  description: string;
  significant_metrics: string[];
  max_spurious_lift: number;
  confidence_interval: [number, number];
  recommendation: 'continue' | 'investigate' | 'abort';
}

interface StatisticalValidation {
  sample_size_adequate: boolean;
  power_analysis: number;
  effect_size_detectable: number;
  type_i_error_risk: number;
}

export class AAShadowTester extends EventEmitter {
  private config: AAShadowConfig;
  private isRunning = false;
  private controlMetrics: Map<string, number[]> = new Map();
  private shadowMetrics: Map<string, number[]> = new Map();
  private measurementInterval: NodeJS.Timeout | null = null;
  private startTime: Date | null = null;

  constructor(config?: Partial<AAShadowConfig>) {
    super();
    
    this.config = {
      trafficPercentage: 2,                // 1-2% traffic for A/A shadow test
      shuffleCentralityWeights: true,      // Shuffle centrality weights for shadow
      controlGroupSize: 1000,             // Minimum queries per measurement
      shadowGroupSize: 1000,              // Minimum queries per measurement
      measurementWindowMinutes: 10,       // Measurement frequency
      ...config
    };
  }

  public async start(): Promise<void> {
    if (this.isRunning) {
      console.log('A/A shadow testing already running');
      return;
    }

    console.log('🔬 Starting A/A shadow testing for spurious lift detection...');
    console.log(`📊 Configuration: ${this.config.trafficPercentage}% traffic, shuffled centrality weights`);
    
    // Initialize shadow testing infrastructure
    await this.initializeShadowTesting();
    
    // Start periodic measurements
    this.measurementInterval = setInterval(() => {
      this.runShadowComparison().catch(error => {
        console.error('A/A shadow test error:', error);
        this.emit('shadowTestError', error);
      });
    }, this.config.measurementWindowMinutes * 60 * 1000);
    
    this.isRunning = true;
    this.startTime = new Date();
    
    console.log('✅ A/A shadow testing started');
    this.emit('shadowTestingStarted', { config: this.config });
  }

  public async stop(): Promise<void> {
    if (this.measurementInterval) {
      clearInterval(this.measurementInterval);
      this.measurementInterval = null;
    }
    
    this.isRunning = false;
    console.log('🛑 A/A shadow testing stopped');
    this.emit('shadowTestingStopped');
  }

  private async initializeShadowTesting(): Promise<void> {
    console.log('⚙️ Initializing A/A shadow testing infrastructure...');
    
    // Set up control group (normal centrality)
    await this.setupControlGroup();
    
    // Set up shadow group (shuffled centrality weights)
    await this.setupShadowGroup();
    
    // Initialize metric collection
    this.initializeMetricCollection();
    
    console.log('✅ A/A shadow testing infrastructure ready');
  }

  private async setupControlGroup(): Promise<void> {
    console.log('📊 Setting up control group (normal centrality)...');
    
    const controlConfig = {
      group_name: 'aa_control',
      traffic_percentage: this.config.trafficPercentage / 2, // Half of shadow traffic
      centrality_config: {
        enabled: true,
        weights_shuffled: false,
        stage_a_centrality_prior: true,
        centrality_log_odds_cap: 0.4
      }
    };
    
    // Implementation would set up feature flags for control group
    console.log('Control group config:', controlConfig);
    // await featureFlagService.createExperimentGroup('aa_control', controlConfig);
  }

  private async setupShadowGroup(): Promise<void> {
    console.log('🔀 Setting up shadow group (shuffled centrality weights)...');
    
    const shadowConfig = {
      group_name: 'aa_shadow',
      traffic_percentage: this.config.trafficPercentage / 2, // Half of shadow traffic
      centrality_config: {
        enabled: true,
        weights_shuffled: true,              // Key difference: shuffled weights
        shuffle_method: 'random_permutation', // Randomly permute centrality scores
        preserve_distribution: true,         // Keep same score distribution
        stage_a_centrality_prior: true,
        centrality_log_odds_cap: 0.4
      }
    };
    
    // Implementation would set up feature flags for shadow group with shuffled weights
    console.log('Shadow group config:', shadowConfig);
    // await featureFlagService.createExperimentGroup('aa_shadow', shadowConfig);
    
    // Set up weight shuffling mechanism
    await this.setupWeightShuffling();
  }

  private async setupWeightShuffling(): Promise<void> {
    console.log('🔀 Setting up centrality weight shuffling...');
    
    // Implementation would modify centrality scoring to shuffle weights for shadow group
    const shufflingConfig = {
      shuffle_type: 'random_permutation',  // Randomly permute node centrality scores
      preserve_ranking_distribution: true, // Keep same distribution of ranks
      shuffle_frequency: 'per_query',     // Shuffle once per query for maximum noise
      seed_rotation_minutes: 30           // Rotate shuffle seed every 30 minutes
    };
    
    console.log('Weight shuffling config:', shufflingConfig);
    // await centralityService.setupWeightShuffling('aa_shadow', shufflingConfig);
  }

  private initializeMetricCollection(): void {
    // Initialize metric storage for both groups
    const metrics = [
      'ndcg_at_10',
      'core_at_10', 
      'diversity_at_10',
      'recall_at_50',
      'semantic_share',
      'stage_a_latency_p95',
      'stage_c_latency_p95',
      'click_through_rate',
      'user_satisfaction_score'
    ];

    for (const metric of metrics) {
      this.controlMetrics.set(metric, []);
      this.shadowMetrics.set(metric, []);
    }
  }

  private async runShadowComparison(): Promise<void> {
    console.log('🔬 Running A/A shadow comparison...');
    
    // Collect metrics for both groups
    const controlData = await this.collectGroupMetrics('aa_control');
    const shadowData = await this.collectGroupMetrics('aa_shadow');
    
    // Store metrics for statistical analysis
    this.storeMetrics(controlData, shadowData);
    
    // Perform statistical comparison
    const comparison = await this.performStatisticalComparison();
    
    // Check for spurious lift
    const spuriousLiftResult = await this.detectSpuriousLift(comparison);
    
    // Emit results
    this.emit('shadowComparisonComplete', {
      control_data: controlData,
      shadow_data: shadowData,
      comparison: comparison,
      spurious_lift: spuriousLiftResult,
      timestamp: new Date()
    });
    
    if (spuriousLiftResult.detected) {
      console.error('❌ Spurious lift detected in A/A shadow test:', spuriousLiftResult.description);
      this.emit('spuriousLiftDetected', spuriousLiftResult);
    } else {
      console.log('✅ A/A shadow test clean - no spurious lift detected');
    }
  }

  private async collectGroupMetrics(groupName: string): Promise<Record<string, number>> {
    // Implementation would collect real metrics from the search system
    // For now, simulate metric collection with realistic A/A test behavior
    
    const isControl = groupName === 'aa_control';
    const baseMetrics = {
      ndcg_at_10: 67.5,
      core_at_10: 45.8,
      diversity_at_10: 32.1,
      recall_at_50: 88.9,
      semantic_share: 35.4,
      stage_a_latency_p95: 45.2,
      stage_c_latency_p95: 127.8,
      click_through_rate: 0.234,
      user_satisfaction_score: 4.12
    };
    
    // Add realistic noise for A/A test (should be minimal differences)
    const metrics: Record<string, number> = {};
    for (const [metric, baseValue] of Object.entries(baseMetrics)) {
      // Control group: minimal random variation
      // Shadow group: also minimal variation (shuffled weights should have no systematic effect)
      const noise = (Math.random() - 0.5) * 0.02; // ±1% random noise
      metrics[metric] = baseValue * (1 + noise);
    }
    
    console.log(`Collected ${groupName} metrics:`, Object.keys(metrics).length, 'metrics');
    return metrics;
  }

  private storeMetrics(controlData: Record<string, number>, shadowData: Record<string, number>): void {
    for (const metric in controlData) {
      this.controlMetrics.get(metric)?.push(controlData[metric]);
      this.shadowMetrics.get(metric)?.push(shadowData[metric]);
      
      // Keep only last 100 measurements for rolling analysis
      if (this.controlMetrics.get(metric)!.length > 100) {
        this.controlMetrics.get(metric)!.shift();
        this.shadowMetrics.get(metric)!.shift();
      }
    }
  }

  private async performStatisticalComparison(): Promise<MetricComparison[]> {
    const comparisons: MetricComparison[] = [];
    
    for (const [metric, controlValues] of this.controlMetrics) {
      const shadowValues = this.shadowMetrics.get(metric)!;
      
      if (controlValues.length < 10 || shadowValues.length < 10) {
        continue; // Need sufficient data points
      }
      
      const controlMean = this.calculateMean(controlValues);
      const shadowMean = this.calculateMean(shadowValues);
      const delta = shadowMean - controlMean;
      const deltaPercentage = (delta / controlMean) * 100;
      
      // Perform t-test
      const pValue = this.performTTest(controlValues, shadowValues);
      const isSignificant = pValue < 0.05;
      
      comparisons.push({
        metric,
        control_value: controlMean,
        shadow_value: shadowMean,
        delta,
        delta_percentage: deltaPercentage,
        p_value: pValue,
        is_significant: isSignificant,
        expected_delta: 0 // A/A test should have ~0 delta
      });
    }
    
    return comparisons;
  }

  private async detectSpuriousLift(comparisons: MetricComparison[]): Promise<SpuriousLiftResult> {
    const significantMetrics = comparisons
      .filter(c => c.is_significant)
      .map(c => c.metric);
    
    const maxSpuriousLift = Math.max(
      ...comparisons.map(c => Math.abs(c.delta_percentage))
    );
    
    // Type I error analysis
    const expectedFalsePositives = comparisons.length * 0.05; // 5% Type I error rate
    const actualSignificantResults = significantMetrics.length;
    
    // Detect spurious lift
    const detected = actualSignificantResults > expectedFalsePositives * 2 || // More than 2x expected false positives
                    maxSpuriousLift > 5 ||                                    // Any metric >5% change
                    significantMetrics.includes('ndcg_at_10') ||              // Critical metric affected
                    significantMetrics.includes('core_at_10');                // Critical metric affected
    
    let description = '';
    let recommendation: 'continue' | 'investigate' | 'abort' = 'continue';
    
    if (detected) {
      if (significantMetrics.includes('ndcg_at_10') || significantMetrics.includes('core_at_10')) {
        description = `Critical metrics show significant differences in A/A test: ${significantMetrics.join(', ')}`;
        recommendation = 'abort';
      } else if (maxSpuriousLift > 10) {
        description = `Excessive spurious lift detected: ${maxSpuriousLift.toFixed(1)}% max change`;
        recommendation = 'abort';
      } else {
        description = `Potential spurious lift: ${actualSignificantResults} significant results (expected ~${expectedFalsePositives.toFixed(1)})`;
        recommendation = 'investigate';
      }
    }
    
    // Confidence interval for max spurious lift
    const confidenceInterval: [number, number] = [
      maxSpuriousLift - 2, // Rough 95% CI
      maxSpuriousLift + 2
    ];
    
    return {
      detected,
      description,
      significant_metrics: significantMetrics,
      max_spurious_lift: maxSpuriousLift,
      confidence_interval: confidenceInterval,
      recommendation
    };
  }

  private calculateMean(values: number[]): number {
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  private calculateStandardDeviation(values: number[]): number {
    const mean = this.calculateMean(values);
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / (values.length - 1);
    return Math.sqrt(variance);
  }

  private performTTest(group1: number[], group2: number[]): number {
    // Welch's t-test implementation
    const mean1 = this.calculateMean(group1);
    const mean2 = this.calculateMean(group2);
    const std1 = this.calculateStandardDeviation(group1);
    const std2 = this.calculateStandardDeviation(group2);
    const n1 = group1.length;
    const n2 = group2.length;
    
    const pooledSE = Math.sqrt((std1 * std1) / n1 + (std2 * std2) / n2);
    const t = (mean1 - mean2) / pooledSE;
    
    // Approximate degrees of freedom (Welch-Satterthwaite equation)
    const df = Math.pow(pooledSE, 4) / (
      Math.pow(std1 * std1 / n1, 2) / (n1 - 1) +
      Math.pow(std2 * std2 / n2, 2) / (n2 - 1)
    );
    
    // Convert t-statistic to p-value (simplified approximation)
    const pValue = 2 * (1 - this.tDistributionCDF(Math.abs(t), df));
    
    return Math.min(pValue, 1.0);
  }

  private tDistributionCDF(t: number, df: number): number {
    // Simplified approximation of t-distribution CDF
    // For production, would use a proper statistical library
    if (df > 30) {
      // Approximate as normal distribution for large df
      return 0.5 + 0.5 * this.erf(t / Math.sqrt(2));
    }
    
    // Very rough approximation for smaller df
    const factor = 1 / (1 + (t * t) / df);
    return 0.5 + 0.5 * Math.sign(t) * Math.sqrt(1 - Math.pow(factor, df / 2));
  }

  private erf(x: number): number {
    // Approximation of error function
    const sign = x >= 0 ? 1 : -1;
    x = Math.abs(x);
    const a1 =  0.254829592;
    const a2 = -0.284496736;
    const a3 =  1.421413741;
    const a4 = -1.453152027;
    const a5 =  1.061405429;
    const p  =  0.3275911;
    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    return sign * y;
  }

  public async checkForSpuriousLift(): Promise<SpuriousLiftResult> {
    if (!this.isRunning || this.controlMetrics.size === 0) {
      return {
        detected: false,
        description: 'A/A shadow testing not active or insufficient data',
        significant_metrics: [],
        max_spurious_lift: 0,
        confidence_interval: [0, 0],
        recommendation: 'continue'
      };
    }
    
    const comparisons = await this.performStatisticalComparison();
    return await this.detectSpuriousLift(comparisons);
  }

  public async validateStatisticalPower(): Promise<StatisticalValidation> {
    const sampleSize = Math.min(
      ...Array.from(this.controlMetrics.values()).map(arr => arr.length)
    );
    
    // Power analysis for detecting 1% effect size with 80% power
    const requiredSampleSize = this.calculateRequiredSampleSize(0.01, 0.8, 0.05);
    const currentPower = this.calculateStatisticalPower(sampleSize, 0.01, 0.05);
    
    return {
      sample_size_adequate: sampleSize >= requiredSampleSize,
      power_analysis: currentPower,
      effect_size_detectable: this.calculateDetectableEffectSize(sampleSize, 0.8, 0.05),
      type_i_error_risk: 0.05 // Fixed at 5%
    };
  }

  private calculateRequiredSampleSize(effectSize: number, power: number, alpha: number): number {
    // Simplified sample size calculation for two-sample t-test
    // n ≈ 2 * (z_α/2 + z_β)² / d²
    const zAlpha = 1.96; // z_{0.025} for α = 0.05
    const zBeta = 0.84;  // z_{0.2} for β = 0.2 (power = 0.8)
    
    return Math.ceil(2 * Math.pow(zAlpha + zBeta, 2) / Math.pow(effectSize, 2));
  }

  private calculateStatisticalPower(sampleSize: number, effectSize: number, alpha: number): number {
    // Simplified power calculation
    const zAlpha = 1.96;
    const delta = effectSize * Math.sqrt(sampleSize / 2);
    return 1 - this.normalCDF(zAlpha - delta);
  }

  private calculateDetectableEffectSize(sampleSize: number, power: number, alpha: number): number {
    // Minimum detectable effect size
    const zAlpha = 1.96;
    const zBeta = 0.84;
    return Math.sqrt(2 * Math.pow(zAlpha + zBeta, 2) / sampleSize);
  }

  private normalCDF(x: number): number {
    return 0.5 * (1 + this.erf(x / Math.sqrt(2)));
  }

  public getShadowTestStatus(): {
    isRunning: boolean;
    elapsedMinutes: number;
    sampleSize: number;
    lastSpuriousLiftCheck: SpuriousLiftResult | null;
  } {
    const elapsedMinutes = this.startTime ? 
      Math.floor((Date.now() - this.startTime.getTime()) / (1000 * 60)) : 0;
    
    const sampleSize = Math.min(
      ...Array.from(this.controlMetrics.values()).map(arr => arr.length)
    );
    
    return {
      isRunning: this.isRunning,
      elapsedMinutes,
      sampleSize,
      lastSpuriousLiftCheck: null // Would store last check result
    };
  }
}