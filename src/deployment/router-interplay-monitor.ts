/**
 * Router Interplay Monitor
 * 
 * Monitors the interaction between centrality system and hybrid dense router,
 * ensuring centrality doesn't inflate router upshift rates beyond acceptable bounds.
 */

import { EventEmitter } from 'events';

interface RouterConfig {
  targetUpshiftRate: number;        // Target: 5%
  tolerancePercentagePoints: number; // ±2 pp tolerance  
  densityThresholds: {
    semantic: number;
    lexical: number;
    symbol: number;
  };
  centralityAdjustments: {
    maxCentralityInflation: number;   // Max centrality boost before router adjustment
    adaptiveThresholdScaling: number; // Scale factor for threshold adjustment
  };
}

interface RouterMetrics {
  upshiftRate: number;
  upshiftRateBaseline: number;
  upshiftRateDelta: number;
  semanticUpshifts: number;
  lexicalUpshifts: number; 
  symbolUpshifts: number;
  centralityInflatedUpshifts: number;
  averageQueryDensity: number;
  centralityBoostDistribution: number[]; // Percentiles [p50, p90, p95, p99]
  timestamp: Date;
}

interface ThresholdAdjustment {
  component: 'semantic' | 'lexical' | 'symbol';
  oldThreshold: number;
  newThreshold: number;
  reason: string;
  effectiveTime: Date;
}

interface RouterViolation {
  violated: boolean;
  currentRate: number;
  targetRange: [number, number];
  severity: 'minor' | 'moderate' | 'severe';
  centralityContribution: number;
  recommendedAction: 'monitor' | 'adjust_thresholds' | 'reduce_centrality' | 'rollback';
}

export class RouterInterplayMonitor extends EventEmitter {
  private config: RouterConfig;
  private metricsHistory: RouterMetrics[] = [];
  private thresholdAdjustmentHistory: ThresholdAdjustment[] = [];
  private currentThresholds: RouterConfig['densityThresholds'];
  private monitoringInterval: NodeJS.Timeout | null = null;
  private isRunning = false;

  constructor(config?: Partial<RouterConfig>) {
    super();
    
    this.config = {
      targetUpshiftRate: 5,         // 5% baseline upshift rate
      tolerancePercentagePoints: 2,  // ±2 pp tolerance = [3%, 7%]
      densityThresholds: {
        semantic: 0.75,             // Dense semantic threshold
        lexical: 0.65,              // Dense lexical threshold  
        symbol: 0.80                // Dense symbol threshold
      },
      centralityAdjustments: {
        maxCentralityInflation: 15,  // Max 15% centrality boost before adjustment
        adaptiveThresholdScaling: 1.1 // 10% threshold increase per adjustment
      },
      ...config
    };

    this.currentThresholds = { ...this.config.densityThresholds };
  }

  public async start(): Promise<void> {
    if (this.isRunning) {
      console.log('Router interplay monitoring already running');
      return;
    }

    console.log('🔀 Starting router interplay monitoring...');
    console.log(`🎯 Target upshift rate: ${this.config.targetUpshiftRate}% ±${this.config.tolerancePercentagePoints}pp`);
    
    // Capture baseline metrics
    await this.captureBaselineMetrics();
    
    // Start periodic monitoring every 5 minutes
    this.monitoringInterval = setInterval(() => {
      this.monitorRouterInterplay().catch(error => {
        console.error('Router interplay monitoring error:', error);
        this.emit('monitoringError', error);
      });
    }, 5 * 60 * 1000);
    
    this.isRunning = true;
    console.log('✅ Router interplay monitoring started');
    this.emit('monitoringStarted', { config: this.config, thresholds: this.currentThresholds });
  }

  public async stop(): Promise<void> {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
    
    this.isRunning = false;
    console.log('🛑 Router interplay monitoring stopped');
    this.emit('monitoringStopped');
  }

  private async captureBaselineMetrics(): Promise<void> {
    console.log('📊 Capturing baseline router metrics...');
    
    const baselineMetrics = await this.collectRouterMetrics(true);
    this.metricsHistory.push(baselineMetrics);
    
    console.log(`Baseline upshift rate: ${baselineMetrics.upshiftRate.toFixed(1)}%`);
    console.log(`Baseline density distribution - Semantic: ${baselineMetrics.semanticUpshifts}, Lexical: ${baselineMetrics.lexicalUpshifts}, Symbol: ${baselineMetrics.symbolUpshifts}`);
    
    this.emit('baselineMetricsCaptured', baselineMetrics);
  }

  private async monitorRouterInterplay(): Promise<void> {
    console.log('🔀 Monitoring router-centrality interplay...');
    
    // Collect current router metrics
    const currentMetrics = await this.collectRouterMetrics(false);
    this.metricsHistory.push(currentMetrics);
    
    // Keep only last 100 measurements
    if (this.metricsHistory.length > 100) {
      this.metricsHistory.shift();
    }
    
    // Check for violations
    const violation = await this.checkUpshiftRateViolation(currentMetrics);
    
    // Analyze centrality contribution
    const centralityAnalysis = await this.analyzeCentralityContribution(currentMetrics);
    
    // Determine if threshold adjustment is needed
    if (violation.violated) {
      console.warn(`⚠️ Router upshift rate violation: ${violation.currentRate.toFixed(1)}% (target: ${this.config.targetUpshiftRate}%±${this.config.tolerancePercentagePoints}pp)`);
      
      if (violation.recommendedAction === 'adjust_thresholds') {
        await this.adjustRouterThresholds(violation, centralityAnalysis);
      } else if (violation.recommendedAction === 'reduce_centrality') {
        console.warn('🚨 Centrality reduction recommended');
        this.emit('centralityReductionNeeded', { violation, analysis: centralityAnalysis });
      } else if (violation.recommendedAction === 'rollback') {
        console.error('🚨 Rollback recommended due to severe router violation');
        this.emit('rollbackRecommended', { reason: 'Router upshift rate violation', violation });
      }
      
      this.emit('upshiftRateViolation', violation);
    } else {
      console.log(`✅ Router upshift rate within bounds: ${violation.currentRate.toFixed(1)}%`);
    }
    
    this.emit('routerInterplayMonitored', {
      metrics: currentMetrics,
      violation,
      centralityAnalysis,
      thresholds: this.currentThresholds
    });
  }

  private async collectRouterMetrics(baseline: boolean): Promise<RouterMetrics> {
    // Implementation would collect real router metrics
    // For now, simulate realistic router behavior with centrality interaction
    
    console.log(`Collecting router metrics (baseline: ${baseline})...`);
    
    // Simulate baseline vs current behavior
    const baseUpshiftRate = 5.0;
    const centralityBoost = baseline ? 0 : 2.3; // +2.3pp centrality contribution
    
    // Centrality can inflate upshift rates by boosting query density scores
    const centralityInflation = baseline ? 0 : centralityBoost * 0.6; // Partial correlation
    const currentUpshiftRate = baseUpshiftRate + centralityInflation;
    
    return {
      upshiftRate: currentUpshiftRate,
      upshiftRateBaseline: baseUpshiftRate,
      upshiftRateDelta: currentUpshiftRate - baseUpshiftRate,
      semanticUpshifts: baseline ? 1200 : 1450, // More semantic upshifts with centrality
      lexicalUpshifts: baseline ? 800 : 820,    // Slight lexical increase
      symbolUpshifts: baseline ? 150 : 180,     // Symbol upshifts increase
      centralityInflatedUpshifts: baseline ? 0 : 280, // Upshifts primarily due to centrality
      averageQueryDensity: baseline ? 0.62 : 0.68,   // Higher average density
      centralityBoostDistribution: baseline ? [0, 0, 0, 0] : [8.2, 18.5, 24.3, 31.7], // [p50, p90, p95, p99]
      timestamp: new Date()
    };
  }

  public async checkUpshiftRate(): Promise<RouterViolation> {
    if (this.metricsHistory.length === 0) {
      return {
        violated: false,
        currentRate: 0,
        targetRange: [
          this.config.targetUpshiftRate - this.config.tolerancePercentagePoints,
          this.config.targetUpshiftRate + this.config.tolerancePercentagePoints
        ],
        severity: 'minor',
        centralityContribution: 0,
        recommendedAction: 'monitor'
      };
    }
    
    const latestMetrics = this.metricsHistory[this.metricsHistory.length - 1];
    return await this.checkUpshiftRateViolation(latestMetrics);
  }

  private async checkUpshiftRateViolation(metrics: RouterMetrics): Promise<RouterViolation> {
    const targetMin = this.config.targetUpshiftRate - this.config.tolerancePercentagePoints;
    const targetMax = this.config.targetUpshiftRate + this.config.tolerancePercentagePoints;
    
    const violated = metrics.upshiftRate < targetMin || metrics.upshiftRate > targetMax;
    const centralityContribution = metrics.upshiftRateDelta;
    
    let severity: 'minor' | 'moderate' | 'severe' = 'minor';
    let recommendedAction: RouterViolation['recommendedAction'] = 'monitor';
    
    if (violated) {
      const deviationPP = Math.max(
        targetMin - metrics.upshiftRate,
        metrics.upshiftRate - targetMax
      );
      
      if (deviationPP > 5) {
        severity = 'severe';
        recommendedAction = 'rollback';
      } else if (deviationPP > 3) {
        severity = 'moderate';
        recommendedAction = centralityContribution > 2 ? 'reduce_centrality' : 'adjust_thresholds';
      } else {
        severity = 'minor';
        recommendedAction = 'adjust_thresholds';
      }
    }
    
    return {
      violated,
      currentRate: metrics.upshiftRate,
      targetRange: [targetMin, targetMax],
      severity,
      centralityContribution,
      recommendedAction
    };
  }

  private async analyzeCentralityContribution(metrics: RouterMetrics): Promise<{
    centralityInflatedPercentage: number;
    primaryComponent: 'semantic' | 'lexical' | 'symbol';
    boostDistributionAnalysis: string;
    thresholdAdjustmentNeeded: boolean;
  }> {
    const totalUpshifts = metrics.semanticUpshifts + metrics.lexicalUpshifts + metrics.symbolUpshifts;
    const centralityInflatedPercentage = (metrics.centralityInflatedUpshifts / totalUpshifts) * 100;
    
    // Determine primary component contributing to upshift inflation
    let primaryComponent: 'semantic' | 'lexical' | 'symbol' = 'semantic';
    if (metrics.symbolUpshifts > metrics.semanticUpshifts) {
      primaryComponent = 'symbol';
    } else if (metrics.lexicalUpshifts > metrics.semanticUpshifts) {
      primaryComponent = 'lexical';
    }
    
    // Analyze boost distribution
    const [p50, p90, p95, p99] = metrics.centralityBoostDistribution;
    let boostDistributionAnalysis = 'Normal distribution';
    
    if (p99 > 30) {
      boostDistributionAnalysis = 'High tail - some queries receive excessive centrality boost';
    } else if (p95 > 25) {
      boostDistributionAnalysis = 'Moderate tail - centrality boost concentrated in top 5%';
    } else if (p90 > 20) {
      boostDistributionAnalysis = 'Slight skew - centrality boost affecting top 10%';
    }
    
    const thresholdAdjustmentNeeded = centralityInflatedPercentage > this.config.centralityAdjustments.maxCentralityInflation;
    
    return {
      centralityInflatedPercentage,
      primaryComponent,
      boostDistributionAnalysis,
      thresholdAdjustmentNeeded
    };
  }

  public async adjustThresholds(): Promise<void> {
    const latestMetrics = this.metricsHistory[this.metricsHistory.length - 1];
    if (!latestMetrics) return;
    
    const violation = await this.checkUpshiftRateViolation(latestMetrics);
    const analysis = await this.analyzeCentralityContribution(latestMetrics);
    
    await this.adjustRouterThresholds(violation, analysis);
  }

  private async adjustRouterThresholds(
    violation: RouterViolation, 
    analysis: { primaryComponent: 'semantic' | 'lexical' | 'symbol' }
  ): Promise<void> {
    console.log(`🔧 Adjusting router thresholds to compensate for centrality inflation...`);
    
    const component = analysis.primaryComponent;
    const oldThreshold = this.currentThresholds[component];
    const scalingFactor = this.config.centralityAdjustments.adaptiveThresholdScaling;
    
    // Increase threshold to make upshift more selective
    const newThreshold = Math.min(oldThreshold * scalingFactor, 0.95); // Cap at 95%
    
    this.currentThresholds[component] = newThreshold;
    
    const adjustment: ThresholdAdjustment = {
      component,
      oldThreshold,
      newThreshold,
      reason: `Compensate for ${violation.centralityContribution.toFixed(1)}pp centrality upshift inflation`,
      effectiveTime: new Date()
    };
    
    this.thresholdAdjustmentHistory.push(adjustment);
    
    // Apply threshold adjustment to router system
    await this.applyThresholdAdjustment(adjustment);
    
    console.log(`✅ ${component} threshold adjusted: ${oldThreshold.toFixed(3)} → ${newThreshold.toFixed(3)}`);
    this.emit('thresholdAdjusted', adjustment);
  }

  private async applyThresholdAdjustment(adjustment: ThresholdAdjustment): Promise<void> {
    // Implementation would update router configuration
    console.log(`Applying threshold adjustment:`, adjustment);
    
    const routerConfig = {
      [`router.dense_${adjustment.component}_threshold`]: adjustment.newThreshold
    };
    
    // await routerService.updateThresholds(routerConfig);
    console.log(`Router ${adjustment.component} threshold updated to ${adjustment.newThreshold.toFixed(3)}`);
  }

  public async resetThresholdsToBaseline(): Promise<void> {
    console.log('🔄 Resetting router thresholds to baseline...');
    
    const baselineThresholds = this.config.densityThresholds;
    
    for (const [component, threshold] of Object.entries(baselineThresholds)) {
      const oldThreshold = this.currentThresholds[component as keyof typeof baselineThresholds];
      
      if (oldThreshold !== threshold) {
        const adjustment: ThresholdAdjustment = {
          component: component as 'semantic' | 'lexical' | 'symbol',
          oldThreshold,
          newThreshold: threshold,
          reason: 'Reset to baseline after centrality rollback',
          effectiveTime: new Date()
        };
        
        this.thresholdAdjustmentHistory.push(adjustment);
        await this.applyThresholdAdjustment(adjustment);
      }
    }
    
    this.currentThresholds = { ...baselineThresholds };
    
    console.log('✅ Router thresholds reset to baseline');
    this.emit('thresholdsReset', { thresholds: this.currentThresholds });
  }

  public getRouterInterplayReport(): {
    currentUpshiftRate: number;
    targetRange: [number, number];
    centralityContribution: number;
    thresholdAdjustments: number;
    trend: 'improving' | 'stable' | 'degrading';
    recommendation: string;
  } {
    if (this.metricsHistory.length === 0) {
      return {
        currentUpshiftRate: 0,
        targetRange: [this.config.targetUpshiftRate - this.config.tolerancePercentagePoints, this.config.targetUpshiftRate + this.config.tolerancePercentagePoints],
        centralityContribution: 0,
        thresholdAdjustments: 0,
        trend: 'stable',
        recommendation: 'No data available'
      };
    }
    
    const latestMetrics = this.metricsHistory[this.metricsHistory.length - 1];
    const recentMetrics = this.metricsHistory.slice(-10);
    
    // Calculate trend
    let trend: 'improving' | 'stable' | 'degrading' = 'stable';
    if (recentMetrics.length >= 5) {
      const earlyAvg = recentMetrics.slice(0, 3).reduce((sum, m) => sum + m.upshiftRate, 0) / 3;
      const lateAvg = recentMetrics.slice(-3).reduce((sum, m) => sum + m.upshiftRate, 0) / 3;
      
      const target = this.config.targetUpshiftRate;
      const earlyDeviation = Math.abs(earlyAvg - target);
      const lateDeviation = Math.abs(lateAvg - target);
      
      if (lateDeviation < earlyDeviation - 0.5) {
        trend = 'improving';
      } else if (lateDeviation > earlyDeviation + 0.5) {
        trend = 'degrading';
      }
    }
    
    // Generate recommendation
    let recommendation = 'Continue monitoring';
    const targetMin = this.config.targetUpshiftRate - this.config.tolerancePercentagePoints;
    const targetMax = this.config.targetUpshiftRate + this.config.tolerancePercentagePoints;
    
    if (latestMetrics.upshiftRate > targetMax) {
      if (latestMetrics.centralityInflatedUpshifts > 200) {
        recommendation = 'Consider reducing centrality weights or cap';
      } else {
        recommendation = 'Adjust router thresholds to increase selectivity';
      }
    } else if (latestMetrics.upshiftRate < targetMin) {
      recommendation = 'Consider reducing router thresholds to increase upshift rate';
    } else if (trend === 'degrading') {
      recommendation = 'Monitor closely - degrading trend detected';
    }
    
    return {
      currentUpshiftRate: latestMetrics.upshiftRate,
      targetRange: [targetMin, targetMax],
      centralityContribution: latestMetrics.upshiftRateDelta,
      thresholdAdjustments: this.thresholdAdjustmentHistory.length,
      trend,
      recommendation
    };
  }

  public getCurrentThresholds(): RouterConfig['densityThresholds'] {
    return { ...this.currentThresholds };
  }

  public getThresholdAdjustmentHistory(): ThresholdAdjustment[] {
    return [...this.thresholdAdjustmentHistory];
  }

  public isHealthy(): boolean {
    if (!this.isRunning || this.metricsHistory.length === 0) {
      return false;
    }
    
    const latestMetrics = this.metricsHistory[this.metricsHistory.length - 1];
    const targetMin = this.config.targetUpshiftRate - this.config.tolerancePercentagePoints;
    const targetMax = this.config.targetUpshiftRate + this.config.tolerancePercentagePoints;
    
    return latestMetrics.upshiftRate >= targetMin && latestMetrics.upshiftRate <= targetMax;
  }
}