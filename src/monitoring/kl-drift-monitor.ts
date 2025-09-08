/**
 * KL Drift Monitoring System
 * 
 * Implements comprehensive KL divergence drift detection for:
 * - Query distribution drift monitoring
 * - Confidence score distribution changes
 * - Intent classification distribution shifts
 * - Language distribution changes
 * - Why-mix distribution evolution
 * - Router upshift monitoring within policy constraints
 */

import { EventEmitter } from 'events';
import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';

export interface KLDistribution {
  name: string;
  bins: number[];
  probabilities: number[];
  total_samples: number;
  timestamp: string;
}

export interface KLDriftMetrics {
  timestamp: string;
  distribution_type: 'query_intent' | 'query_language' | 'confidence_scores' | 'why_mix' | 'router_upshift';
  current_distribution: KLDistribution;
  baseline_distribution: KLDistribution;
  kl_divergence: number;
  js_divergence: number; // Jensen-Shannon divergence (symmetric)
  wasserstein_distance: number;
  drift_significance: number; // Statistical significance of drift
  sample_size: number;
}

export interface KLDriftAlert {
  alert_id: string;
  distribution_type: string;
  kl_divergence: number;
  threshold_exceeded: number;
  drift_magnitude: 'low' | 'medium' | 'high' | 'critical';
  timestamp: string;
  recommended_actions: string[];
  statistical_confidence: number;
  sample_size: number;
}

export interface WhyMixDistribution {
  exact_match: number;
  structural_match: number;
  semantic_match: number;
  hybrid_match: number;
  fallback: number;
}

export interface RouterUpshiftMetrics {
  dense_route_percentage: number;
  sparse_route_percentage: number;
  fallback_route_percentage: number;
  upshift_rate: number; // Rate of upshifting from sparse to dense
  policy_compliance: boolean;
  total_queries: number;
}

/**
 * Statistical utilities for distribution analysis
 */
class DistributionAnalyzer {
  /**
   * Calculate KL divergence D(P||Q) = sum(P * log(P/Q))
   */
  static calculateKLDivergence(p: number[], q: number[]): number {
    if (p.length !== q.length) {
      throw new Error('Distribution arrays must have equal length');
    }

    let kl = 0;
    for (let i = 0; i < p.length; i++) {
      if (p[i] > 0 && q[i] > 0) {
        kl += p[i] * Math.log(p[i] / q[i]);
      } else if (p[i] > 0 && q[i] === 0) {
        // Handle case where Q has zero probability but P doesn't
        return Infinity;
      }
    }
    return kl;
  }

  /**
   * Calculate Jensen-Shannon divergence (symmetric version of KL)
   */
  static calculateJSDivergence(p: number[], q: number[]): number {
    const m = p.map((pi, i) => (pi + q[i]) / 2);
    const klPM = this.calculateKLDivergence(p, m);
    const klQM = this.calculateKLDivergence(q, m);
    return (klPM + klQM) / 2;
  }

  /**
   * Calculate Wasserstein distance (Earth Mover's Distance)
   */
  static calculateWassersteinDistance(p: number[], q: number[]): number {
    if (p.length !== q.length) {
      throw new Error('Distribution arrays must have equal length');
    }

    let distance = 0;
    let cumulativeDiff = 0;

    for (let i = 0; i < p.length; i++) {
      cumulativeDiff += (p[i] - q[i]);
      distance += Math.abs(cumulativeDiff);
    }

    return distance;
  }

  /**
   * Perform statistical significance test for distribution drift
   */
  static calculateDriftSignificance(
    currentSamples: number, 
    baselineSamples: number, 
    klDivergence: number
  ): number {
    // Simplified chi-square test approximation
    // In production, would use proper bootstrap or permutation tests
    const effectiveN = Math.min(currentSamples, baselineSamples);
    const testStatistic = 2 * effectiveN * klDivergence;
    
    // Approximate p-value using chi-square distribution
    // This is a rough approximation - real implementation would use proper statistical tests
    return Math.exp(-testStatistic / 2);
  }

  /**
   * Create distribution from raw samples
   */
  static createDistribution(samples: any[], binCount: number = 10): KLDistribution {
    const bins = new Array(binCount).fill(0);
    const binSize = 1.0 / binCount;
    
    // Normalize samples to [0,1] range if they're numbers
    let normalizedSamples: number[];
    if (typeof samples[0] === 'number') {
      const min = Math.min(...samples as number[]);
      const max = Math.max(...samples as number[]);
      const range = max - min;
      normalizedSamples = (samples as number[]).map(s => range > 0 ? (s - min) / range : 0);
    } else {
      // For categorical data, create hash-based bins
      const categories = Array.from(new Set(samples));
      normalizedSamples = samples.map(s => 
        categories.indexOf(s) / Math.max(categories.length - 1, 1)
      );
    }

    // Fill bins
    for (const sample of normalizedSamples) {
      const binIndex = Math.min(Math.floor(sample / binSize), binCount - 1);
      bins[binIndex]++;
    }

    // Convert to probabilities
    const probabilities = bins.map(count => count / samples.length);

    return {
      name: `distribution_${Date.now()}`,
      bins,
      probabilities,
      total_samples: samples.length,
      timestamp: new Date().toISOString()
    };
  }
}

/**
 * Main KL Drift Monitoring System
 */
export class KLDriftMonitor extends EventEmitter {
  private baselineDistributions = new Map<string, KLDistribution>();
  private currentDistributions = new Map<string, KLDistribution>();
  private driftHistory = new Map<string, KLDriftMetrics[]>();
  private monitoringDir: string;
  private config: {
    kl_drift_threshold: number;
    js_drift_threshold: number;
    wasserstein_threshold: number;
    significance_threshold: number;
    min_sample_size: number;
    monitoring_interval_minutes: number;
  };

  constructor(monitoringDir: string = './monitoring-data') {
    super();
    
    this.monitoringDir = monitoringDir;
    this.config = {
      kl_drift_threshold: 0.02,
      js_drift_threshold: 0.01,
      wasserstein_threshold: 0.05,
      significance_threshold: 0.05,
      min_sample_size: 100,
      monitoring_interval_minutes: 10
    };

    if (!existsSync(this.monitoringDir)) {
      mkdirSync(this.monitoringDir, { recursive: true });
    }

    this.loadBaselines();
    this.startMonitoring();
  }

  /**
   * Monitor query intent distribution drift
   */
  monitorQueryIntentDrift(intentSamples: string[]): void {
    const distributionType = 'query_intent';
    const currentDist = DistributionAnalyzer.createDistribution(intentSamples);
    
    this.analyzeDistributionDrift(distributionType, currentDist);
  }

  /**
   * Monitor query language distribution drift
   */
  monitorQueryLanguageDrift(languageSamples: string[]): void {
    const distributionType = 'query_language';
    const currentDist = DistributionAnalyzer.createDistribution(languageSamples);
    
    this.analyzeDistributionDrift(distributionType, currentDist);
  }

  /**
   * Monitor confidence score distribution drift
   */
  monitorConfidenceScoreDrift(confidenceScores: number[]): void {
    const distributionType = 'confidence_scores';
    const currentDist = DistributionAnalyzer.createDistribution(confidenceScores, 20);
    
    this.analyzeDistributionDrift(distributionType, currentDist);
  }

  /**
   * Monitor why-mix distribution evolution
   */
  monitorWhyMixDrift(whyMixSamples: WhyMixDistribution[]): void {
    // Convert why-mix to numeric samples for distribution analysis
    const samples: number[] = [];
    
    for (const sample of whyMixSamples) {
      // Weight each type by its proportion
      samples.push(
        0 * sample.exact_match +
        1 * sample.structural_match +
        2 * sample.semantic_match +
        3 * sample.hybrid_match +
        4 * sample.fallback
      );
    }

    const distributionType = 'why_mix';
    const currentDist = DistributionAnalyzer.createDistribution(samples, 15);
    
    this.analyzeDistributionDrift(distributionType, currentDist);
  }

  /**
   * Monitor router upshift distribution within policy
   */
  monitorRouterUpshiftDrift(routerMetrics: RouterUpshiftMetrics[]): void {
    const upshiftRates = routerMetrics.map(m => m.upshift_rate);
    const distributionType = 'router_upshift';
    const currentDist = DistributionAnalyzer.createDistribution(upshiftRates, 12);
    
    // Additional policy compliance check
    const policyViolations = routerMetrics.filter(m => !m.policy_compliance).length;
    const violationRate = policyViolations / routerMetrics.length;
    
    if (violationRate > 0.05) { // 5% violation threshold
      this.emitAlert({
        alert_id: `router_policy_violation_${Date.now()}`,
        distribution_type: 'router_upshift',
        kl_divergence: violationRate,
        threshold_exceeded: 0.05,
        drift_magnitude: 'critical',
        timestamp: new Date().toISOString(),
        recommended_actions: [
          'Review router policy configuration',
          'Check for upshift logic errors',
          'Validate dense routing constraints',
          'Consider temporary policy relaxation'
        ],
        statistical_confidence: 1.0,
        sample_size: routerMetrics.length
      });
    }
    
    this.analyzeDistributionDrift(distributionType, currentDist);
  }

  /**
   * Core drift analysis logic
   */
  private analyzeDistributionDrift(distributionType: string, currentDist: KLDistribution): void {
    // Update current distribution
    this.currentDistributions.set(distributionType, currentDist);
    
    // Get baseline for comparison
    const baseline = this.baselineDistributions.get(distributionType);
    if (!baseline) {
      // Set as new baseline if first measurement
      this.baselineDistributions.set(distributionType, currentDist);
      console.log(`📊 Set new ${distributionType} baseline distribution`);
      return;
    }
    
    // Skip if insufficient sample size
    if (currentDist.total_samples < this.config.min_sample_size) {
      return;
    }

    // Calculate drift metrics
    const klDiv = DistributionAnalyzer.calculateKLDivergence(
      currentDist.probabilities,
      baseline.probabilities
    );
    
    const jsDiv = DistributionAnalyzer.calculateJSDivergence(
      currentDist.probabilities,
      baseline.probabilities
    );
    
    const wassersteinDist = DistributionAnalyzer.calculateWassersteinDistance(
      currentDist.probabilities,
      baseline.probabilities
    );
    
    const significance = DistributionAnalyzer.calculateDriftSignificance(
      currentDist.total_samples,
      baseline.total_samples,
      klDiv
    );

    const metrics: KLDriftMetrics = {
      timestamp: currentDist.timestamp,
      distribution_type: distributionType as any,
      current_distribution: currentDist,
      baseline_distribution: baseline,
      kl_divergence: klDiv,
      js_divergence: jsDiv,
      wasserstein_distance: wassersteinDist,
      drift_significance: significance,
      sample_size: currentDist.total_samples
    };

    // Store metrics history
    if (!this.driftHistory.has(distributionType)) {
      this.driftHistory.set(distributionType, []);
    }
    
    const history = this.driftHistory.get(distributionType)!;
    history.push(metrics);
    
    // Keep only recent history
    if (history.length > 100) {
      history.shift();
    }

    // Check for significant drift
    this.checkDriftThresholds(distributionType, metrics);
  }

  /**
   * Check if drift exceeds thresholds and emit alerts
   */
  private checkDriftThresholds(distributionType: string, metrics: KLDriftMetrics): void {
    const { kl_divergence, js_divergence, wasserstein_distance, drift_significance } = metrics;
    
    // Determine drift magnitude
    let driftMagnitude: 'low' | 'medium' | 'high' | 'critical' = 'low';
    let shouldAlert = false;
    let threshold = 0;
    
    if (kl_divergence > this.config.kl_drift_threshold * 3) {
      driftMagnitude = 'critical';
      shouldAlert = true;
      threshold = this.config.kl_drift_threshold * 3;
    } else if (kl_divergence > this.config.kl_drift_threshold * 2) {
      driftMagnitude = 'high';
      shouldAlert = true;
      threshold = this.config.kl_drift_threshold * 2;
    } else if (kl_divergence > this.config.kl_drift_threshold) {
      driftMagnitude = 'medium';
      shouldAlert = drift_significance < this.config.significance_threshold;
      threshold = this.config.kl_drift_threshold;
    }

    // Also check JS divergence for additional confirmation
    if (js_divergence > this.config.js_drift_threshold) {
      shouldAlert = true;
      driftMagnitude = driftMagnitude === 'low' ? 'medium' : driftMagnitude;
    }

    if (shouldAlert) {
      this.emitAlert({
        alert_id: `kl_drift_${distributionType}_${Date.now()}`,
        distribution_type: distributionType,
        kl_divergence,
        threshold_exceeded: threshold,
        drift_magnitude: driftMagnitude,
        timestamp: metrics.timestamp,
        recommended_actions: this.getRecommendedActions(distributionType, driftMagnitude),
        statistical_confidence: 1 - drift_significance,
        sample_size: metrics.sample_size
      });
    }
  }

  /**
   * Get recommended actions based on distribution type and severity
   */
  private getRecommendedActions(distributionType: string, magnitude: string): string[] {
    const baseActions: Record<string, string[]> = {
      query_intent: [
        'Review recent query patterns for classification changes',
        'Check intent classification model for drift',
        'Validate training data distribution alignment',
        'Consider intent classifier retraining'
      ],
      query_language: [
        'Check for new language support or changes',
        'Review corpus language distribution changes',
        'Validate language detection accuracy',
        'Check for repository language shifts'
      ],
      confidence_scores: [
        'Review confidence calibration in model serving',
        'Check for model serving temperature changes',
        'Validate confidence score normalization',
        'Consider confidence recalibration'
      ],
      why_mix: [
        'Review routing logic for why-mix changes',
        'Check semantic vs structural balance',
        'Validate fallback routing behavior',
        'Monitor exact match performance'
      ],
      router_upshift: [
        'Check router upshift policy compliance',
        'Review dense routing thresholds',
        'Validate upshift triggers and conditions',
        'Monitor resource utilization impact'
      ]
    };

    const actions = [...(baseActions[distributionType] || [])];
    
    if (magnitude === 'critical') {
      actions.unshift('URGENT: Consider immediate rollback');
      actions.push('Escalate to on-call team for investigation');
    } else if (magnitude === 'high') {
      actions.unshift('HIGH PRIORITY: Schedule immediate review');
    }

    return actions;
  }

  /**
   * Emit KL drift alert
   */
  private emitAlert(alert: KLDriftAlert): void {
    console.log(`🚨 KL DRIFT ALERT [${alert.drift_magnitude.toUpperCase()}]: ${alert.distribution_type}`);
    console.log(`   KL Divergence: ${alert.kl_divergence.toFixed(6)} (threshold: ${alert.threshold_exceeded.toFixed(6)})`);
    console.log(`   Statistical confidence: ${(alert.statistical_confidence * 100).toFixed(1)}%`);
    console.log(`   Sample size: ${alert.sample_size}`);
    console.log(`   Recommended actions:`);
    alert.recommended_actions.forEach(action => console.log(`     - ${action}`));

    this.emit('kl_drift_alert', alert);
  }

  /**
   * Start periodic monitoring
   */
  private startMonitoring(): void {
    setInterval(() => {
      this.saveState();
    }, this.config.monitoring_interval_minutes * 60 * 1000);

    console.log(`📊 KL drift monitoring started`);
    console.log(`   KL threshold: ${this.config.kl_drift_threshold}`);
    console.log(`   JS threshold: ${this.config.js_drift_threshold}`);
    console.log(`   Minimum sample size: ${this.config.min_sample_size}`);
  }

  /**
   * Get current drift status report
   */
  getDriftStatusReport(): {
    timestamp: string;
    distributions_monitored: number;
    active_drifts: number;
    distribution_status: Array<{
      type: string;
      current_kl: number;
      baseline_samples: number;
      current_samples: number;
      drift_status: 'stable' | 'warning' | 'critical';
      last_updated: string;
    }>;
    overall_health: 'healthy' | 'degraded' | 'critical';
  } {
    const timestamp = new Date().toISOString();
    const status = [];
    let activeDrifts = 0;

    for (const [type, history] of this.driftHistory.entries()) {
      if (history.length === 0) continue;
      
      const latest = history[history.length - 1];
      let driftStatus: 'stable' | 'warning' | 'critical' = 'stable';
      
      if (latest.kl_divergence > this.config.kl_drift_threshold * 3) {
        driftStatus = 'critical';
        activeDrifts++;
      } else if (latest.kl_divergence > this.config.kl_drift_threshold) {
        driftStatus = 'warning';
        activeDrifts++;
      }

      status.push({
        type,
        current_kl: latest.kl_divergence,
        baseline_samples: latest.baseline_distribution.total_samples,
        current_samples: latest.current_distribution.total_samples,
        drift_status: driftStatus,
        last_updated: latest.timestamp
      });
    }

    const overallHealth = activeDrifts === 0 ? 'healthy' :
                         status.some(s => s.drift_status === 'critical') ? 'critical' : 'degraded';

    return {
      timestamp,
      distributions_monitored: status.length,
      active_drifts: activeDrifts,
      distribution_status: status,
      overall_health: overallHealth
    };
  }

  /**
   * Reset baseline for specific distribution type
   */
  resetBaseline(distributionType: string): boolean {
    const current = this.currentDistributions.get(distributionType);
    if (!current) {
      return false;
    }

    this.baselineDistributions.set(distributionType, { ...current });
    console.log(`🔄 Reset ${distributionType} baseline distribution`);
    
    return true;
  }

  private loadBaselines(): void {
    try {
      const baselinePath = join(this.monitoringDir, 'kl_baselines.json');
      if (existsSync(baselinePath)) {
        const data = JSON.parse(readFileSync(baselinePath, 'utf8'));
        
        for (const [key, value] of Object.entries(data)) {
          this.baselineDistributions.set(key, value as KLDistribution);
        }
        
        console.log(`📈 Loaded ${this.baselineDistributions.size} KL baseline distributions`);
      }
    } catch (error) {
      console.warn('⚠️ Failed to load KL baselines:', error);
    }
  }

  private saveState(): void {
    try {
      // Save baselines
      const baselinePath = join(this.monitoringDir, 'kl_baselines.json');
      const baselineData: Record<string, KLDistribution> = {};
      
      for (const [key, value] of this.baselineDistributions.entries()) {
        baselineData[key] = value;
      }
      
      writeFileSync(baselinePath, JSON.stringify(baselineData, null, 2));

      // Save current metrics
      const metricsPath = join(this.monitoringDir, 'kl_drift_history.json');
      const historyData: Record<string, KLDriftMetrics[]> = {};
      
      for (const [key, value] of this.driftHistory.entries()) {
        historyData[key] = value;
      }
      
      writeFileSync(metricsPath, JSON.stringify(historyData, null, 2));
      
    } catch (error) {
      console.error('❌ Failed to save KL drift state:', error);
    }
  }
}

// Global instance
export const globalKLDriftMonitor = new KLDriftMonitor();