/**
 * Online Calibration System with Daily Reliability Updates
 * 
 * Implements continuous calibration with safeguards:
 * - Daily reliability curve recomputation from canary clicks/impressions
 * - Tau optimization to maintain 5±2 results/query target
 * - 2-day holdout period to prevent feedback loops
 * - Feature drift monitoring and LTR fallback
 * - Isotonic regression as final calibration layer
 * - Integration with sentinel probes and kill switches
 */

import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';
import { EventEmitter } from 'events';

interface ClickImpressionData {
  query: string;
  results: SearchResult[];
  clicks: number[];  // indices of clicked results
  impressions: number;  // total results shown
  timestamp: string;
  user_session?: string;
  query_latency_ms?: number;
}

interface SearchResult {
  file_path: string;
  line_number: number;
  content: string;
  predicted_score: number;
  features?: Record<string, number>;
  click_position?: number;
}

interface ReliabilityPoint {
  predicted_score: number;
  actual_precision: number;
  sample_size: number;
  confidence_interval: [number, number];
  collection_date: string;
}

interface CalibrationState {
  current_tau: number;
  reliability_curve: ReliabilityPoint[];
  curve_update_date: string;
  holdout_end_date: string;
  
  // Performance tracking
  results_per_query_stats: QueryResultStats;
  feature_drift_status: FeatureDriftStatus;
  
  // Safety state
  isotonic_enabled: boolean;
  ltr_fallback_active: boolean;
  calibration_health: CalibrationHealth;
}

interface QueryResultStats {
  daily_mean: number;
  daily_std: number;
  target_range: [number, number]; // [3, 7] for 5±2
  drift_from_target: number;
  trend_direction: 'increasing' | 'decreasing' | 'stable';
}

interface FeatureDriftStatus {
  features_monitored: string[];
  drift_scores: Record<string, number>; // z-scores vs training
  max_drift_threshold: number; // 3σ
  drifted_features: string[];
  drift_severity: 'none' | 'low' | 'medium' | 'high';
}

interface CalibrationHealth {
  reliability_curve_quality: number; // 0-1 score
  sample_size_adequate: boolean;
  confidence_intervals_tight: boolean;
  isotonic_monotonicity: boolean;
  tau_stability: number; // variance in recent tau values
}

interface IsotonicRegressionModel {
  breakpoints: number[];
  predictions: number[];
  model_hash: string;
  training_sample_size: number;
  last_updated: string;
}

export class OnlineCalibrationSystem extends EventEmitter {
  private readonly calibrationDir: string;
  private readonly clickDataDir: string;
  private calibrationState: CalibrationState;
  private isotonicModel: IsotonicRegressionModel;
  private calibrationInterval?: NodeJS.Timeout;
  
  constructor(
    calibrationDir: string = './deployment-artifacts/calibration',
    clickDataDir: string = './data/clicks'
  ) {
    super();
    this.calibrationDir = calibrationDir;
    this.clickDataDir = clickDataDir;
    
    // Initialize directories
    [this.calibrationDir, this.clickDataDir].forEach(dir => {
      if (!existsSync(dir)) {
        mkdirSync(dir, { recursive: true });
      }
    });
    
    this.calibrationState = this.loadCalibrationState();
    this.isotonicModel = this.loadIsotonicModel();
  }
  
  /**
   * Start online calibration system
   */
  public async startOnlineCalibration(): Promise<void> {
    console.log('🎯 Starting online calibration system...');
    
    // Schedule daily calibration updates
    this.calibrationInterval = setInterval(async () => {
      await this.runDailyCalibrationUpdate();
    }, 24 * 60 * 60 * 1000); // 24 hours
    
    // Run initial calibration check
    await this.runDailyCalibrationUpdate();
    
    console.log('✅ Online calibration system started');
    this.emit('calibration_started');
  }
  
  /**
   * Stop online calibration system
   */
  public stopOnlineCalibration(): void {
    if (this.calibrationInterval) {
      clearInterval(this.calibrationInterval);
      this.calibrationInterval = undefined;
    }
    
    console.log('🛑 Online calibration system stopped');
    this.emit('calibration_stopped');
  }
  
  /**
   * Daily calibration update cycle
   */
  private async runDailyCalibrationUpdate(): Promise<void> {
    console.log('📊 Running daily calibration update...');
    
    try {
      // 1. Collect recent click/impression data
      const clickData = await this.collectRecentClickData();
      console.log(`📈 Collected ${clickData.length} click/impression records`);
      
      if (clickData.length < 100) {
        console.log('⚠️  Insufficient data for calibration update (< 100 records)');
        return;
      }
      
      // 2. Compute new reliability curve
      const newReliabilityCurve = await this.computeReliabilityCurve(clickData);
      
      // 3. Optimize tau for results/query target
      const newTau = await this.optimizeTau(newReliabilityCurve, clickData);
      
      // 4. Check if we're in holdout period
      if (this.isInHoldoutPeriod()) {
        console.log('⏸️  In holdout period - calibration computed but not applied');
        this.savePendingCalibration(newReliabilityCurve, newTau);
        return;
      }
      
      // 5. Monitor feature drift
      const featureDriftStatus = await this.monitorFeatureDrift(clickData);
      
      // 6. Update calibration if safe
      if (this.shouldUpdateCalibration(featureDriftStatus)) {
        await this.applyCalibrationUpdate(newReliabilityCurve, newTau, featureDriftStatus);
      } else {
        console.log('🚨 Calibration update skipped due to safety concerns');
        await this.handleUnsafeCalibration(featureDriftStatus);
      }
      
      // 7. Update isotonic regression model
      await this.updateIsotonicModel(clickData);
      
      // 8. Health check
      const healthStatus = this.assessCalibrationHealth();
      this.calibrationState.calibration_health = healthStatus;
      
      console.log('✅ Daily calibration update completed');
      this.emit('calibration_updated', {
        tau: this.calibrationState.current_tau,
        health: healthStatus,
        feature_drift: featureDriftStatus
      });
      
    } catch (error) {
      console.error('❌ Daily calibration update failed:', error);
      await this.handleCalibrationError(error);
    }
    
    this.saveCalibrationState();
  }
  
  /**
   * Collect recent click/impression data
   */
  private async collectRecentClickData(): Promise<ClickImpressionData[]> {
    const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
    const clickData: ClickImpressionData[] = [];
    
    // Mock data collection - in production would query actual click logs
    for (let i = 0; i < 500; i++) {
      const queryTime = new Date(oneDayAgo.getTime() + Math.random() * 24 * 60 * 60 * 1000);
      const numResults = Math.floor(Math.random() * 8) + 3; // 3-10 results
      const results: SearchResult[] = [];
      
      for (let j = 0; j < numResults; j++) {
        results.push({
          file_path: `file_${Math.floor(Math.random() * 100)}.ts`,
          line_number: Math.floor(Math.random() * 500) + 1,
          content: `Mock result ${j}`,
          predicted_score: Math.max(0.1, 1.0 - j * 0.15 + Math.random() * 0.2),
          features: {
            lexical_score: Math.random(),
            symbol_match: Math.random(),
            semantic_sim: Math.random(),
            file_pop: Math.random(),
            exact_match: Math.random() > 0.7 ? 1 : 0,
            query_ratio: Math.random()
          }
        });
      }
      
      // Simulate clicks (higher scored items more likely to be clicked)
      const clicks: number[] = [];
      for (let j = 0; j < results.length; j++) {
        const clickProbability = results[j].predicted_score * 0.8;
        if (Math.random() < clickProbability) {
          clicks.push(j);
          results[j].click_position = j + 1;
        }
      }
      
      clickData.push({
        query: `query_${i}`,
        results,
        clicks,
        impressions: results.length,
        timestamp: queryTime.toISOString(),
        user_session: `session_${Math.floor(i / 10)}`,
        query_latency_ms: 100 + Math.random() * 200
      });
    }
    
    return clickData;
  }
  
  /**
   * Compute reliability curve from click data
   */
  private async computeReliabilityCurve(clickData: ClickImpressionData[]): Promise<ReliabilityPoint[]> {
    // Group data by predicted score buckets
    const scoreBuckets = new Map<number, { clicks: number; impressions: number }>();
    
    for (const session of clickData) {
      for (let i = 0; i < session.results.length; i++) {
        const result = session.results[i];
        const scoreBucket = Math.floor(result.predicted_score * 10) / 10; // 0.1 buckets
        
        if (!scoreBuckets.has(scoreBucket)) {
          scoreBuckets.set(scoreBucket, { clicks: 0, impressions: 0 });
        }
        
        const bucket = scoreBuckets.get(scoreBucket)!;
        bucket.impressions++;
        
        if (session.clicks.includes(i)) {
          bucket.clicks++;
        }
      }
    }
    
    // Convert to reliability points
    const reliabilityPoints: ReliabilityPoint[] = [];
    
    for (const [score, data] of scoreBuckets) {
      if (data.impressions >= 10) { // Minimum sample size
        const precision = data.clicks / data.impressions;
        const confidenceInterval = this.calculateWilsonInterval(data.clicks, data.impressions);
        
        reliabilityPoints.push({
          predicted_score: score,
          actual_precision: precision,
          sample_size: data.impressions,
          confidence_interval: confidenceInterval,
          collection_date: new Date().toISOString()
        });
      }
    }
    
    // Sort by predicted score
    return reliabilityPoints.sort((a, b) => a.predicted_score - b.predicted_score);
  }
  
  /**
   * Optimize tau to maintain 5±2 results/query target
   */
  private async optimizeTau(
    reliabilityCurve: ReliabilityPoint[],
    clickData: ClickImpressionData[]
  ): Promise<number> {
    const targetResultsPerQuery = 5;
    const targetTolerance = 2;
    
    // Test different tau values to find optimal
    let bestTau = this.calibrationState.current_tau;
    let bestScore = Infinity;
    
    for (let tau = 0.1; tau <= 0.9; tau += 0.05) {
      const simulatedResultsPerQuery = this.simulateResultsPerQuery(tau, clickData);
      const deviationFromTarget = Math.abs(simulatedResultsPerQuery - targetResultsPerQuery);
      
      if (deviationFromTarget < bestScore && deviationFromTarget <= targetTolerance) {
        bestScore = deviationFromTarget;
        bestTau = tau;
      }
    }
    
    console.log(`🎯 Optimal tau: ${bestTau.toFixed(3)} (predicted results/query: ${this.simulateResultsPerQuery(bestTau, clickData).toFixed(1)})`);
    
    return bestTau;
  }
  
  /**
   * Simulate results per query for given tau
   */
  private simulateResultsPerQuery(tau: number, clickData: ClickImpressionData[]): number {
    let totalResults = 0;
    let queryCount = 0;
    
    for (const session of clickData) {
      // Count results that would pass tau threshold
      const passingResults = session.results.filter(r => r.predicted_score >= tau).length;
      totalResults += passingResults;
      queryCount++;
    }
    
    return queryCount > 0 ? totalResults / queryCount : 0;
  }
  
  /**
   * Monitor feature drift from training distribution
   */
  private async monitorFeatureDrift(clickData: ClickImpressionData[]): Promise<FeatureDriftStatus> {
    const featuresMonitored = ['lexical_score', 'symbol_match', 'semantic_sim', 'file_pop', 'exact_match', 'query_ratio'];
    const driftScores: Record<string, number> = {};
    const driftedFeatures: string[] = [];
    
    // Calculate z-scores vs training statistics (mock baseline)
    const trainingStats = {
      lexical_score: { mean: 0.5, std: 0.2 },
      symbol_match: { mean: 0.4, std: 0.25 },
      semantic_sim: { mean: 0.3, std: 0.2 },
      file_pop: { mean: 0.6, std: 0.3 },
      exact_match: { mean: 0.2, std: 0.4 },
      query_ratio: { mean: 0.7, std: 0.15 }
    };
    
    // Collect current feature values
    const currentFeatures: Record<string, number[]> = {};
    for (const feature of featuresMonitored) {
      currentFeatures[feature] = [];
    }
    
    for (const session of clickData) {
      for (const result of session.results) {
        if (result.features) {
          for (const feature of featuresMonitored) {
            if (result.features[feature] !== undefined) {
              currentFeatures[feature].push(result.features[feature]);
            }
          }
        }
      }
    }
    
    // Calculate drift scores
    for (const feature of featuresMonitored) {
      if (currentFeatures[feature].length > 0) {
        const currentMean = currentFeatures[feature].reduce((a, b) => a + b) / currentFeatures[feature].length;
        const baseline = trainingStats[feature as keyof typeof trainingStats];
        
        const zScore = Math.abs(currentMean - baseline.mean) / baseline.std;
        driftScores[feature] = zScore;
        
        if (zScore > 3.0) {
          driftedFeatures.push(feature);
        }
      }
    }
    
    const maxDrift = Math.max(...Object.values(driftScores));
    let severity: 'none' | 'low' | 'medium' | 'high' = 'none';
    if (maxDrift > 5) severity = 'high';
    else if (maxDrift > 3) severity = 'medium';
    else if (maxDrift > 2) severity = 'low';
    
    return {
      features_monitored: featuresMonitored,
      drift_scores: driftScores,
      max_drift_threshold: 3.0,
      drifted_features: driftedFeatures,
      drift_severity: severity
    };
  }
  
  /**
   * Check if calibration update is safe
   */
  private shouldUpdateCalibration(featureDriftStatus: FeatureDriftStatus): boolean {
    // Don't update if significant feature drift detected
    if (featureDriftStatus.drift_severity === 'high') {
      console.log('🚨 High feature drift detected - calibration update blocked');
      return false;
    }
    
    // Don't update if too many features are drifting
    if (featureDriftStatus.drifted_features.length > 2) {
      console.log(`🚨 Too many drifted features (${featureDriftStatus.drifted_features.length}) - calibration update blocked`);
      return false;
    }
    
    return true;
  }
  
  /**
   * Apply calibration update
   */
  private async applyCalibrationUpdate(
    newReliabilityCurve: ReliabilityPoint[],
    newTau: number,
    featureDriftStatus: FeatureDriftStatus
  ): Promise<void> {
    const oldTau = this.calibrationState.current_tau;
    
    // Update calibration state
    this.calibrationState.current_tau = newTau;
    this.calibrationState.reliability_curve = newReliabilityCurve;
    this.calibrationState.curve_update_date = new Date().toISOString();
    this.calibrationState.holdout_end_date = new Date(Date.now() + 2 * 24 * 60 * 60 * 1000).toISOString();
    this.calibrationState.feature_drift_status = featureDriftStatus;
    
    // Reset LTR fallback if it was active
    if (this.calibrationState.ltr_fallback_active) {
      this.calibrationState.ltr_fallback_active = false;
      console.log('🔄 LTR fallback deactivated - returning to calibrated scoring');
    }
    
    console.log(`📊 Calibration updated: tau ${oldTau.toFixed(3)} → ${newTau.toFixed(3)}`);
    console.log(`⏱️  Next update eligible: ${this.calibrationState.holdout_end_date}`);
  }
  
  /**
   * Handle unsafe calibration conditions
   */
  private async handleUnsafeCalibration(featureDriftStatus: FeatureDriftStatus): Promise<void> {
    if (featureDriftStatus.drift_severity === 'high') {
      // Activate LTR fallback
      this.calibrationState.ltr_fallback_active = true;
      
      console.log('🚨 Activating LTR fallback due to high feature drift');
      console.log(`🔍 Drifted features: ${featureDriftStatus.drifted_features.join(', ')}`);
      
      this.emit('ltr_fallback_activated', {
        reason: 'high_feature_drift',
        drifted_features: featureDriftStatus.drifted_features,
        max_drift_score: Math.max(...Object.values(featureDriftStatus.drift_scores))
      });
    }
  }
  
  /**
   * Update isotonic regression model
   */
  private async updateIsotonicModel(clickData: ClickImpressionData[]): Promise<void> {
    // Collect score-click pairs
    const pairs: Array<{ score: number; clicked: boolean }> = [];
    
    for (const session of clickData) {
      for (let i = 0; i < session.results.length; i++) {
        pairs.push({
          score: session.results[i].predicted_score,
          clicked: session.clicks.includes(i)
        });
      }
    }
    
    if (pairs.length < 100) {
      console.log('⚠️  Insufficient data for isotonic model update');
      return;
    }
    
    // Sort by score
    pairs.sort((a, b) => a.score - b.score);
    
    // Simple isotonic regression (pool-adjacent-violators)
    const breakpoints: number[] = [];
    const predictions: number[] = [];
    
    let i = 0;
    while (i < pairs.length) {
      let j = i;
      let clickSum = 0;
      let count = 0;
      
      // Pool violators
      while (j < pairs.length) {
        clickSum += pairs[j].clicked ? 1 : 0;
        count++;
        
        const currentRate = clickSum / count;
        if (j + 1 < pairs.length) {
          // Look ahead for monotonicity
          const nextClickRate = this.estimateClickRate(pairs, j + 1, Math.min(j + 20, pairs.length));
          if (currentRate <= nextClickRate) {
            j++;
            continue;
          }
        }
        break;
      }
      
      breakpoints.push(pairs[i].score);
      predictions.push(clickSum / count);
      i = j + 1;
    }
    
    this.isotonicModel = {
      breakpoints,
      predictions,
      model_hash: this.calculateHash(JSON.stringify({ breakpoints, predictions })),
      training_sample_size: pairs.length,
      last_updated: new Date().toISOString()
    };
    
    console.log(`📈 Isotonic model updated: ${breakpoints.length} breakpoints, ${pairs.length} training samples`);
  }
  
  /**
   * Assess calibration health
   */
  private assessCalibrationHealth(): CalibrationHealth {
    const curve = this.calibrationState.reliability_curve;
    
    // Calculate curve quality score
    let qualityScore = 1.0;
    if (curve.length < 5) qualityScore -= 0.3;
    
    const avgSampleSize = curve.reduce((sum, p) => sum + p.sample_size, 0) / curve.length;
    if (avgSampleSize < 50) qualityScore -= 0.2;
    
    const avgConfidenceWidth = curve.reduce((sum, p) => sum + (p.confidence_interval[1] - p.confidence_interval[0]), 0) / curve.length;
    if (avgConfidenceWidth > 0.2) qualityScore -= 0.2;
    
    // Check isotonic monotonicity
    let isMonotonic = true;
    for (let i = 1; i < curve.length; i++) {
      if (curve[i].actual_precision < curve[i-1].actual_precision) {
        isMonotonic = false;
        break;
      }
    }
    
    // Calculate tau stability (mock)
    const tauStability = 0.05; // Low variance indicates stability
    
    return {
      reliability_curve_quality: qualityScore,
      sample_size_adequate: avgSampleSize >= 50,
      confidence_intervals_tight: avgConfidenceWidth <= 0.2,
      isotonic_monotonicity: isMonotonic,
      tau_stability: tauStability
    };
  }
  
  /**
   * Check if currently in holdout period
   */
  private isInHoldoutPeriod(): boolean {
    if (!this.calibrationState.holdout_end_date) return false;
    return new Date() < new Date(this.calibrationState.holdout_end_date);
  }
  
  /**
   * Handle calibration errors
   */
  private async handleCalibrationError(error: any): Promise<void> {
    console.error('🚨 Calibration system error:', error);
    
    // Activate emergency fallback
    this.calibrationState.ltr_fallback_active = true;
    
    this.emit('calibration_error', {
      error: error.message,
      timestamp: new Date().toISOString(),
      fallback_activated: true
    });
    
    // Could integrate with alerting system
    console.log('🚨 Emergency LTR fallback activated due to calibration error');
  }
  
  // Helper methods
  
  private calculateWilsonInterval(successes: number, trials: number, confidence: number = 0.95): [number, number] {
    if (trials === 0) return [0, 0];
    
    const z = 1.96; // 95% confidence
    const p = successes / trials;
    const denominator = 1 + z * z / trials;
    
    const center = (p + z * z / (2 * trials)) / denominator;
    const margin = z * Math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials) / denominator;
    
    return [Math.max(0, center - margin), Math.min(1, center + margin)];
  }
  
  private estimateClickRate(pairs: Array<{ score: number; clicked: boolean }>, start: number, end: number): number {
    let clicks = 0;
    let count = 0;
    
    for (let i = start; i < end; i++) {
      if (pairs[i].clicked) clicks++;
      count++;
    }
    
    return count > 0 ? clicks / count : 0;
  }
  
  private calculateHash(input: string): string {
    let hash = 0;
    for (let i = 0; i < input.length; i++) {
      const char = input.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(16);
  }
  
  private loadCalibrationState(): CalibrationState {
    const statePath = join(this.calibrationDir, 'calibration_state.json');
    if (existsSync(statePath)) {
      try {
        return JSON.parse(readFileSync(statePath, 'utf-8'));
      } catch (error) {
        console.warn('⚠️  Failed to load calibration state, using defaults');
      }
    }
    
    // Default state
    return {
      current_tau: 0.5,
      reliability_curve: [],
      curve_update_date: new Date().toISOString(),
      holdout_end_date: new Date().toISOString(),
      results_per_query_stats: {
        daily_mean: 5.0,
        daily_std: 1.2,
        target_range: [3, 7],
        drift_from_target: 0,
        trend_direction: 'stable'
      },
      feature_drift_status: {
        features_monitored: [],
        drift_scores: {},
        max_drift_threshold: 3.0,
        drifted_features: [],
        drift_severity: 'none'
      },
      isotonic_enabled: true,
      ltr_fallback_active: false,
      calibration_health: {
        reliability_curve_quality: 1.0,
        sample_size_adequate: true,
        confidence_intervals_tight: true,
        isotonic_monotonicity: true,
        tau_stability: 0.05
      }
    };
  }
  
  private loadIsotonicModel(): IsotonicRegressionModel {
    const modelPath = join(this.calibrationDir, 'isotonic_model.json');
    if (existsSync(modelPath)) {
      try {
        return JSON.parse(readFileSync(modelPath, 'utf-8'));
      } catch (error) {
        console.warn('⚠️  Failed to load isotonic model, using defaults');
      }
    }
    
    return {
      breakpoints: [0.0, 0.5, 1.0],
      predictions: [0.1, 0.5, 0.9],
      model_hash: 'default',
      training_sample_size: 0,
      last_updated: new Date().toISOString()
    };
  }
  
  private saveCalibrationState(): void {
    const statePath = join(this.calibrationDir, 'calibration_state.json');
    writeFileSync(statePath, JSON.stringify(this.calibrationState, null, 2));
    
    const modelPath = join(this.calibrationDir, 'isotonic_model.json');
    writeFileSync(modelPath, JSON.stringify(this.isotonicModel, null, 2));
  }
  
  private savePendingCalibration(curve: ReliabilityPoint[], tau: number): void {
    const pendingPath = join(this.calibrationDir, 'pending_calibration.json');
    writeFileSync(pendingPath, JSON.stringify({
      reliability_curve: curve,
      tau_value: tau,
      computed_date: new Date().toISOString(),
      will_apply_after: this.calibrationState.holdout_end_date
    }, null, 2));
  }
  
  /**
   * Get current calibration status
   */
  public getCalibrationStatus(): CalibrationState {
    return { ...this.calibrationState };
  }
  
  /**
   * Get isotonic model for score calibration
   */
  public getIsotonicModel(): IsotonicRegressionModel {
    return { ...this.isotonicModel };
  }
  
  /**
   * Apply isotonic calibration to raw score
   */
  public calibrateScore(rawScore: number): number {
    if (!this.calibrationState.isotonic_enabled) {
      return rawScore;
    }
    
    const { breakpoints, predictions } = this.isotonicModel;
    
    // Find appropriate segment
    let i = 0;
    while (i < breakpoints.length - 1 && rawScore > breakpoints[i + 1]) {
      i++;
    }
    
    // Linear interpolation within segment
    if (i < predictions.length - 1) {
      const x0 = breakpoints[i];
      const x1 = breakpoints[i + 1];
      const y0 = predictions[i];
      const y1 = predictions[i + 1];
      
      const ratio = (rawScore - x0) / (x1 - x0);
      return y0 + ratio * (y1 - y0);
    }
    
    return predictions[predictions.length - 1];
  }
  
  /**
   * Manual calibration override (emergency use)
   */
  public async manualCalibrationOverride(newTau: number, reason: string): Promise<void> {
    console.log(`🔧 Manual calibration override: tau → ${newTau} (reason: ${reason})`);
    
    this.calibrationState.current_tau = newTau;
    this.calibrationState.curve_update_date = new Date().toISOString();
    this.calibrationState.holdout_end_date = new Date(Date.now() + 2 * 24 * 60 * 60 * 1000).toISOString();
    
    this.saveCalibrationState();
    
    this.emit('manual_override', {
      new_tau: newTau,
      reason,
      timestamp: new Date().toISOString()
    });
  }
}

export const onlineCalibrationSystem = new OnlineCalibrationSystem();