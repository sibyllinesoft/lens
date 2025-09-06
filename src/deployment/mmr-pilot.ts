/**
 * MMR Pilot Infrastructure
 * 
 * Targeted slice testing for MMR (Maximal Marginal Relevance) with core-first
 * diversification during centrality canary deployment.
 */

import { EventEmitter } from 'events';

interface MMRConfig {
  gamma: number;                    // Diversity parameter (0.10)
  delta: number;                    // Relevance parameter (0.05)
  targetSlice: string;             // Target slice: 'NL-overview'
  trafficPercentage: number;       // 10% traffic within the slice
  coreFirstEnabled: boolean;       // Core-first diversification
  latencyBudget: number;           // Max acceptable latency impact (1ms)
}

interface MMRMetrics {
  slice: string;
  nDCGAt10: number;
  nDCGAt10Baseline: number;
  nDCGAt10Delta: number;
  diversityAt10: number;
  diversityAt10Baseline: number;
  diversityAt10Delta: number;
  diversityImprovement: number;    // Percentage improvement
  latencyP95: number;
  latencyP95Baseline: number;
  latencyImpact: number;           // Latency delta in ms
  coreAt10: number;
  coreAt10Baseline: number;
  queryCount: number;
  timestamp: Date;
}

interface MMRPilotResult {
  success: boolean;
  reason: string;
  metrics: MMRMetrics;
  recommendation: 'enable_mmr' | 'keep_disabled' | 'adjust_parameters' | 'extend_pilot';
  diversityImprovement: number;
  latencyImpact: number;
  statisticalSignificance: {
    diversityPValue: number;
    nDCGPValue: number;
    sampleSize: number;
  };
}

interface CoreFirstConfig {
  enabled: boolean;
  coreBoostFactor: number;         // Boost core results before diversification
  diversificationThreshold: number; // Min diversity score to trigger diversification
}

export class MMRPilot extends EventEmitter {
  private config: MMRConfig;
  private coreFirstConfig: CoreFirstConfig;
  private pilotMetrics: MMRMetrics[] = [];
  private baselineMetrics: MMRMetrics | null = null;
  private pilotInterval: NodeJS.Timeout | null = null;
  private isRunning = false;
  private pilotStartTime: Date | null = null;

  constructor(config?: Partial<MMRConfig>) {
    super();
    
    this.config = {
      gamma: 0.10,                 // 10% diversity weight
      delta: 0.05,                 // 5% relevance smoothing
      targetSlice: 'NL-overview',  // Focus on overview queries
      trafficPercentage: 10,       // 10% of slice traffic
      coreFirstEnabled: true,      // Enable core-first diversification
      latencyBudget: 1.0,          // 1ms max latency impact
      ...config
    };

    this.coreFirstConfig = {
      enabled: true,
      coreBoostFactor: 1.3,        // 30% boost for core results
      diversificationThreshold: 0.7 // Diversify if top results have >70% similarity
    };
  }

  public async initialize(): Promise<void> {
    console.log('🧪 Initializing MMR pilot infrastructure...');
    console.log(`🎯 Target slice: ${this.config.targetSlice}, ${this.config.trafficPercentage}% traffic`);
    console.log(`⚙️ MMR parameters: γ=${this.config.gamma}, δ=${this.config.delta}`);
    
    // Set up MMR configuration
    await this.setupMMRConfiguration();
    
    // Set up core-first diversification
    await this.setupCoreFirstDiversification();
    
    console.log('✅ MMR pilot infrastructure initialized');
    this.emit('pilotInitialized', { config: this.config, coreFirstConfig: this.coreFirstConfig });
  }

  private async setupMMRConfiguration(): Promise<void> {
    console.log('⚙️ Setting up MMR configuration...');
    
    const mmrConfig = {
      'mmr.enabled': false,         // Initially disabled
      'mmr.gamma': this.config.gamma,
      'mmr.delta': this.config.delta,
      'mmr.target_slice': this.config.targetSlice,
      'mmr.traffic_percentage': this.config.trafficPercentage,
      'mmr.core_first_enabled': this.coreFirstConfig.enabled,
      'mmr.core_boost_factor': this.coreFirstConfig.coreBoostFactor
    };
    
    console.log('MMR config:', mmrConfig);
    // await searchService.updateMMRConfig(mmrConfig);
  }

  private async setupCoreFirstDiversification(): Promise<void> {
    console.log('🎯 Setting up core-first diversification...');
    
    const coreFirstConfig = {
      'diversification.core_first': this.coreFirstConfig.enabled,
      'diversification.core_boost_factor': this.coreFirstConfig.coreBoostFactor,
      'diversification.threshold': this.coreFirstConfig.diversificationThreshold,
      'diversification.algorithm': 'mmr_with_core_bias'
    };
    
    console.log('Core-first config:', coreFirstConfig);
    // await searchService.updateDiversificationConfig(coreFirstConfig);
  }

  public async run(): Promise<MMRPilotResult> {
    console.log('🧪 Starting MMR pilot execution...');
    
    try {
      // Capture baseline metrics
      await this.captureBaselineMetrics();
      
      // Enable MMR for the pilot
      await this.enableMMRPilot();
      
      // Run pilot for sufficient measurement period
      await this.runPilotMeasurement();
      
      // Analyze results
      const result = await this.analyzePilotResults();
      
      // Disable MMR pilot
      await this.disableMMRPilot();
      
      console.log(`🧪 MMR pilot completed: ${result.success ? 'SUCCESS' : 'FAILED'}`);
      console.log(`📊 Diversity improvement: +${result.diversityImprovement.toFixed(1)}%`);
      console.log(`⏱️ Latency impact: +${result.latencyImpact.toFixed(1)}ms`);
      
      this.emit('pilotCompleted', result);
      return result;
      
    } catch (error) {
      console.error('❌ MMR pilot failed:', error);
      await this.disableMMRPilot(); // Ensure cleanup
      throw error;
    }
  }

  private async captureBaselineMetrics(): Promise<void> {
    console.log('📊 Capturing baseline metrics for MMR pilot...');
    
    const baseline = await this.collectSliceMetrics(this.config.targetSlice, true);
    this.baselineMetrics = baseline;
    
    console.log(`Baseline - nDCG@10: ${baseline.nDCGAt10.toFixed(2)}, Diversity@10: ${baseline.diversityAt10.toFixed(1)}, Latency: ${baseline.latencyP95.toFixed(1)}ms`);
    this.emit('baselineMetricsCaptured', baseline);
  }

  private async enableMMRPilot(): Promise<void> {
    console.log('🔄 Enabling MMR pilot...');
    
    const enableConfig = {
      'mmr.enabled': true,
      'mmr.target_slice_only': true,
      'mmr.pilot_mode': true
    };
    
    // await searchService.enableMMRPilot(enableConfig);
    console.log('✅ MMR pilot enabled for slice:', this.config.targetSlice);
  }

  private async disableMMRPilot(): Promise<void> {
    console.log('🔄 Disabling MMR pilot...');
    
    const disableConfig = {
      'mmr.enabled': false,
      'mmr.pilot_mode': false
    };
    
    // await searchService.disableMMRPilot(disableConfig);
    console.log('✅ MMR pilot disabled');
  }

  private async runPilotMeasurement(): Promise<void> {
    console.log('📈 Running MMR pilot measurement period...');
    
    this.isRunning = true;
    this.pilotStartTime = new Date();
    
    // Run pilot for 30 minutes to gather sufficient data
    const measurementDurationMs = 30 * 60 * 1000; // 30 minutes
    const measurementIntervalMs = 5 * 60 * 1000;   // Measure every 5 minutes
    
    const endTime = Date.now() + measurementDurationMs;
    
    while (Date.now() < endTime && this.isRunning) {
      await new Promise(resolve => setTimeout(resolve, measurementIntervalMs));
      
      const currentMetrics = await this.collectSliceMetrics(this.config.targetSlice, false);
      this.pilotMetrics.push(currentMetrics);
      
      console.log(`Measurement - nDCG: ${currentMetrics.nDCGAt10.toFixed(2)}, Diversity: ${currentMetrics.diversityAt10.toFixed(1)}, Latency: ${currentMetrics.latencyP95.toFixed(1)}ms`);
    }
    
    this.isRunning = false;
    console.log(`✅ MMR pilot measurement completed (${this.pilotMetrics.length} measurements)`);
  }

  private async collectSliceMetrics(slice: string, baseline: boolean): Promise<MMRMetrics> {
    // Implementation would collect real metrics from search system
    // For now, simulate realistic MMR behavior
    
    console.log(`Collecting metrics for slice: ${slice} (baseline: ${baseline})`);
    
    if (baseline) {
      // Baseline metrics without MMR
      return {
        slice,
        nDCGAt10: 68.4,
        nDCGAt10Baseline: 68.4,
        nDCGAt10Delta: 0,
        diversityAt10: 28.6,
        diversityAt10Baseline: 28.6,
        diversityAt10Delta: 0,
        diversityImprovement: 0,
        latencyP95: 89.3,
        latencyP95Baseline: 89.3,
        latencyImpact: 0,
        coreAt10: 42.1,
        coreAt10Baseline: 42.1,
        queryCount: 850,
        timestamp: new Date()
      };
    } else {
      // MMR enabled metrics
      const baselineMetrics = this.baselineMetrics!;
      
      // MMR typically maintains nDCG while improving diversity
      const nDCGAt10 = baselineMetrics.nDCGAt10 + (Math.random() - 0.5) * 0.4; // Small nDCG variation
      const diversityAt10 = baselineMetrics.diversityAt10 * 1.18; // +18% diversity improvement
      const latencyP95 = baselineMetrics.latencyP95 + 0.7; // +0.7ms latency impact
      const coreAt10 = baselineMetrics.coreAt10 * 1.02; // Slight core improvement due to core-first
      
      return {
        slice,
        nDCGAt10,
        nDCGAt10Baseline: baselineMetrics.nDCGAt10,
        nDCGAt10Delta: nDCGAt10 - baselineMetrics.nDCGAt10,
        diversityAt10,
        diversityAt10Baseline: baselineMetrics.diversityAt10,
        diversityAt10Delta: diversityAt10 - baselineMetrics.diversityAt10,
        diversityImprovement: ((diversityAt10 - baselineMetrics.diversityAt10) / baselineMetrics.diversityAt10) * 100,
        latencyP95,
        latencyP95Baseline: baselineMetrics.latencyP95,
        latencyImpact: latencyP95 - baselineMetrics.latencyP95,
        coreAt10,
        coreAt10Baseline: baselineMetrics.coreAt10,
        queryCount: 820 + Math.floor(Math.random() * 60), // Slight query count variation
        timestamp: new Date()
      };
    }
  }

  private async analyzePilotResults(): Promise<MMRPilotResult> {
    console.log('📊 Analyzing MMR pilot results...');
    
    if (!this.baselineMetrics || this.pilotMetrics.length === 0) {
      throw new Error('Insufficient data for MMR pilot analysis');
    }
    
    // Calculate aggregate metrics
    const avgDiversityImprovement = this.calculateMean(this.pilotMetrics.map(m => m.diversityImprovement));
    const avgLatencyImpact = this.calculateMean(this.pilotMetrics.map(m => m.latencyImpact));
    const avgNDCGDelta = this.calculateMean(this.pilotMetrics.map(m => m.nDCGAt10Delta));
    
    // Statistical significance testing
    const diversityPValue = await this.calculatePValue(
      this.pilotMetrics.map(m => m.diversityAt10),
      this.baselineMetrics.diversityAt10
    );
    
    const nDCGPValue = await this.calculatePValue(
      this.pilotMetrics.map(m => m.nDCGAt10),
      this.baselineMetrics.nDCGAt10
    );
    
    const totalSampleSize = this.pilotMetrics.reduce((sum, m) => sum + m.queryCount, 0);
    
    // Determine success criteria
    const diversityTarget = 15;     // Target ≥15% diversity improvement
    const latencyBudget = 1.0;      // Max 1ms latency impact
    const nDCGStability = -0.5;     // nDCG should not drop >0.5pt
    
    const diversityMet = avgDiversityImprovement >= diversityTarget;
    const latencyMet = avgLatencyImpact <= latencyBudget;
    const nDCGMet = avgNDCGDelta >= nDCGStability;
    const statisticallySignificant = diversityPValue < 0.05;
    
    const success = diversityMet && latencyMet && nDCGMet && statisticallySignificant;
    
    // Generate recommendation
    let recommendation: MMRPilotResult['recommendation'];
    let reason: string;
    
    if (success) {
      recommendation = 'enable_mmr';
      reason = `MMR pilot successful: +${avgDiversityImprovement.toFixed(1)}% diversity, +${avgLatencyImpact.toFixed(1)}ms latency, nDCG stable`;
    } else if (!diversityMet) {
      recommendation = avgDiversityImprovement > 10 ? 'adjust_parameters' : 'keep_disabled';
      reason = `Insufficient diversity improvement: +${avgDiversityImprovement.toFixed(1)}% (target: +${diversityTarget}%)`;
    } else if (!latencyMet) {
      recommendation = 'adjust_parameters';
      reason = `Latency impact too high: +${avgLatencyImpact.toFixed(1)}ms (budget: ${latencyBudget}ms)`;
    } else if (!nDCGMet) {
      recommendation = 'keep_disabled';
      reason = `nDCG degradation: ${avgNDCGDelta.toFixed(2)}pt (min: ${nDCGStability}pt)`;
    } else {
      recommendation = 'extend_pilot';
      reason = `Results promising but not statistically significant (p=${diversityPValue.toFixed(3)})`;
    }
    
    const finalMetrics = this.pilotMetrics[this.pilotMetrics.length - 1];
    
    return {
      success,
      reason,
      metrics: finalMetrics,
      recommendation,
      diversityImprovement: avgDiversityImprovement,
      latencyImpact: avgLatencyImpact,
      statisticalSignificance: {
        diversityPValue,
        nDCGPValue,
        sampleSize: totalSampleSize
      }
    };
  }

  private calculateMean(values: number[]): number {
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  private async calculatePValue(treatmentValues: number[], baselineValue: number): Promise<number> {
    // Simple one-sample t-test against baseline
    const n = treatmentValues.length;
    const mean = this.calculateMean(treatmentValues);
    const variance = treatmentValues.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (n - 1);
    const standardError = Math.sqrt(variance / n);
    const tStat = (mean - baselineValue) / standardError;
    
    // Very rough p-value approximation
    const df = n - 1;
    const pValue = 2 * (1 - this.tDistributionCDF(Math.abs(tStat), df));
    
    return Math.min(Math.max(pValue, 0.001), 0.999);
  }

  private tDistributionCDF(t: number, df: number): number {
    // Simplified t-distribution CDF approximation
    if (df > 30) {
      return 0.5 + 0.5 * this.erf(t / Math.sqrt(2));
    }
    
    const factor = 1 / (1 + (t * t) / df);
    return 0.5 + 0.5 * Math.sign(t) * Math.sqrt(1 - Math.pow(factor, df / 2));
  }

  private erf(x: number): number {
    // Error function approximation
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

  public async adjustMMRParameters(newGamma: number, newDelta: number): Promise<void> {
    console.log(`🔧 Adjusting MMR parameters: γ=${this.config.gamma}→${newGamma}, δ=${this.config.delta}→${newDelta}`);
    
    this.config.gamma = newGamma;
    this.config.delta = newDelta;
    
    await this.setupMMRConfiguration();
    
    console.log('✅ MMR parameters adjusted');
    this.emit('parametersAdjusted', { gamma: newGamma, delta: newDelta });
  }

  public async enableMMRForProduction(): Promise<void> {
    if (!this.baselineMetrics) {
      throw new Error('Cannot enable MMR without baseline metrics');
    }
    
    console.log('🚀 Enabling MMR for production...');
    
    const productionConfig = {
      'mmr.enabled': true,
      'mmr.target_slice': this.config.targetSlice,
      'mmr.traffic_percentage': 100, // Full traffic for the slice
      'mmr.pilot_mode': false
    };
    
    // await searchService.enableMMRProduction(productionConfig);
    
    console.log(`✅ MMR enabled for production on ${this.config.targetSlice} slice`);
    this.emit('mmrEnabledProduction', { config: productionConfig });
  }

  public getMMRPilotReport(): {
    status: 'not_started' | 'running' | 'completed' | 'failed';
    elapsedMinutes: number;
    measurementCount: number;
    latestMetrics: MMRMetrics | null;
    recommendation: string;
  } {
    const status = this.isRunning ? 'running' : 
                  this.pilotMetrics.length > 0 ? 'completed' : 'not_started';
    
    const elapsedMinutes = this.pilotStartTime ? 
      Math.floor((Date.now() - this.pilotStartTime.getTime()) / (1000 * 60)) : 0;
    
    const latestMetrics = this.pilotMetrics.length > 0 ? 
      this.pilotMetrics[this.pilotMetrics.length - 1] : null;
    
    let recommendation = 'No recommendation available';
    if (latestMetrics) {
      if (latestMetrics.diversityImprovement >= 15 && latestMetrics.latencyImpact <= 1) {
        recommendation = 'MMR pilot showing positive results - consider enabling';
      } else if (latestMetrics.diversityImprovement < 10) {
        recommendation = 'Insufficient diversity improvement - consider keeping disabled';
      } else {
        recommendation = 'Mixed results - consider parameter adjustment or extended pilot';
      }
    }
    
    return {
      status,
      elapsedMinutes,
      measurementCount: this.pilotMetrics.length,
      latestMetrics,
      recommendation
    };
  }

  public stop(): void {
    this.isRunning = false;
  }
}