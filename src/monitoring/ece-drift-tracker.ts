/**
 * ECE (Expected Calibration Error) Drift Tracking System
 * 
 * Implements live tracking of ECE and miscoverage by intent×language with:
 * - Real-time ECE computation and drift detection
 * - Intent classification (semantic, structural, lexical, hybrid)
 * - Language stratification (TypeScript, Python, JavaScript, etc.)
 * - KL drift monitoring ≤ 0.02 threshold
 * - A/A shadow testing with drift ≤ 0.1 pp tolerance
 * - Integration with production monitoring alerts
 */

import { EventEmitter } from 'events';
import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';

export interface IntentLanguageQuery {
  query_id: string;
  intent: 'semantic' | 'structural' | 'lexical' | 'hybrid';
  language: 'typescript' | 'python' | 'javascript' | 'rust' | 'go' | 'java' | 'other';
  confidence: number;    // Model confidence [0,1]
  actual_relevance: number; // Ground truth relevance [0,1]
  timestamp: string;
  sla_bounded: boolean;  // Whether query completed within SLA
}

export interface ECEMetrics {
  timestamp: string;
  intent: string;
  language: string;
  total_queries: number;
  ece_value: number;      // Expected Calibration Error
  miscoverage_rate: number; // Rate of miscalibrated predictions
  confidence_bins: CalibrationBin[];
  reliability_diagram: ReliabilityPoint[];
  kl_divergence: number;  // KL divergence from uniform
  average_confidence: number;
  average_accuracy: number;
}

export interface CalibrationBin {
  bin_lower: number;
  bin_upper: number;
  count: number;
  avg_confidence: number;
  avg_accuracy: number;
  contribution_to_ece: number;
}

export interface ReliabilityPoint {
  confidence: number;
  accuracy: number;
  count: number;
}

export interface DriftAlert {
  alert_id: string;
  alert_type: 'ece_drift' | 'kl_drift' | 'miscoverage_spike' | 'aa_drift';
  severity: 'warning' | 'critical';
  intent: string;
  language: string;
  current_value: number;
  baseline_value: number;
  drift_magnitude: number;
  threshold_exceeded: number;
  timestamp: string;
  recommended_actions: string[];
}

export interface AATestConfig {
  enabled: boolean;
  traffic_split: number;  // Percentage for A/A testing
  sample_size: number;    // Minimum samples for comparison
  drift_threshold: number; // Maximum allowed drift (pp)
  test_duration_minutes: number;
}

/**
 * ECE computation with reliability binning
 */
export class ECECalculator {
  private readonly numBins: number;

  constructor(numBins: number = 15) {
    this.numBins = numBins;
  }

  /**
   * Calculate ECE and calibration metrics for a set of predictions
   */
  calculateECE(queries: IntentLanguageQuery[]): {
    ece: number;
    miscoverage_rate: number;
    bins: CalibrationBin[];
    reliability_points: ReliabilityPoint[];
    kl_divergence: number;
  } {
    if (queries.length === 0) {
      return {
        ece: 0,
        miscoverage_rate: 0,
        bins: [],
        reliability_points: [],
        kl_divergence: 0
      };
    }

    // Create bins for calibration
    const bins = this.createCalibrationBins(queries);
    
    // Calculate ECE as weighted average of |confidence - accuracy| per bin
    let totalECE = 0;
    const totalQueries = queries.length;
    
    for (const bin of bins) {
      if (bin.count > 0) {
        const binWeight = bin.count / totalQueries;
        const calibrationError = Math.abs(bin.avg_confidence - bin.avg_accuracy);
        bin.contribution_to_ece = binWeight * calibrationError;
        totalECE += bin.contribution_to_ece;
      }
    }

    // Calculate miscoverage rate (queries with |confidence - accuracy| > 0.1)
    const miscalibratedCount = queries.filter(q => 
      Math.abs(q.confidence - q.actual_relevance) > 0.1
    ).length;
    const miscoverageRate = miscalibratedCount / totalQueries;

    // Generate reliability diagram points
    const reliabilityPoints = this.generateReliabilityPoints(bins);

    // Calculate KL divergence from uniform confidence distribution
    const klDivergence = this.calculateKLDivergence(queries);

    return {
      ece: totalECE,
      miscoverage_rate: miscoverageRate,
      bins,
      reliability_points: reliabilityPoints,
      kl_divergence: klDivergence
    };
  }

  private createCalibrationBins(queries: IntentLanguageQuery[]): CalibrationBin[] {
    const bins: CalibrationBin[] = [];
    const binSize = 1.0 / this.numBins;

    // Initialize bins
    for (let i = 0; i < this.numBins; i++) {
      bins.push({
        bin_lower: i * binSize,
        bin_upper: (i + 1) * binSize,
        count: 0,
        avg_confidence: 0,
        avg_accuracy: 0,
        contribution_to_ece: 0
      });
    }

    // Assign queries to bins and calculate statistics
    for (const query of queries) {
      const binIndex = Math.min(
        Math.floor(query.confidence / binSize),
        this.numBins - 1
      );
      
      const bin = bins[binIndex];
      const oldCount = bin.count;
      const newCount = oldCount + 1;
      
      // Update running averages
      bin.avg_confidence = (bin.avg_confidence * oldCount + query.confidence) / newCount;
      bin.avg_accuracy = (bin.avg_accuracy * oldCount + query.actual_relevance) / newCount;
      bin.count = newCount;
    }

    return bins;
  }

  private generateReliabilityPoints(bins: CalibrationBin[]): ReliabilityPoint[] {
    return bins
      .filter(bin => bin.count > 0)
      .map(bin => ({
        confidence: bin.avg_confidence,
        accuracy: bin.avg_accuracy,
        count: bin.count
      }));
  }

  private calculateKLDivergence(queries: IntentLanguageQuery[]): number {
    // Calculate KL divergence from uniform distribution
    const bins = new Array(10).fill(0);
    
    for (const query of queries) {
      const binIndex = Math.min(Math.floor(query.confidence * 10), 9);
      bins[binIndex]++;
    }

    // Convert to probabilities
    const totalQueries = queries.length;
    const probabilities = bins.map(count => count / totalQueries);
    const uniform = 1.0 / bins.length;

    // Calculate KL divergence: sum(p * log(p / q))
    let klDiv = 0;
    for (const p of probabilities) {
      if (p > 0) {
        klDiv += p * Math.log(p / uniform);
      }
    }

    return klDiv;
  }
}

/**
 * A/A Testing Infrastructure for drift detection
 */
export class AATestManager {
  private currentTest: {
    test_id: string;
    start_time: string;
    group_a_queries: IntentLanguageQuery[];
    group_b_queries: IntentLanguageQuery[];
    config: AATestConfig;
  } | null = null;

  private eceCalculator: ECECalculator;

  constructor() {
    this.eceCalculator = new ECECalculator();
  }

  /**
   * Start A/A test for drift detection
   */
  startAATest(config: AATestConfig): string {
    const testId = `aa_test_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    this.currentTest = {
      test_id: testId,
      start_time: new Date().toISOString(),
      group_a_queries: [],
      group_b_queries: [],
      config
    };

    console.log(`🔬 Starting A/A test: ${testId}`);
    console.log(`   Traffic split: ${config.traffic_split}%`);
    console.log(`   Duration: ${config.test_duration_minutes} minutes`);
    console.log(`   Drift threshold: ${config.drift_threshold} pp`);

    return testId;
  }

  /**
   * Add query to A/A test (randomly assigned to group)
   */
  addQueryToAATest(query: IntentLanguageQuery): boolean {
    if (!this.currentTest?.config.enabled) {
      return false;
    }

    // Random assignment to groups
    const assignToGroupA = Math.random() < 0.5;
    
    if (assignToGroupA) {
      this.currentTest.group_a_queries.push(query);
    } else {
      this.currentTest.group_b_queries.push(query);
    }

    return true;
  }

  /**
   * Evaluate A/A test for significant drift
   */
  evaluateAATest(): {
    has_drift: boolean;
    drift_magnitude: number;
    group_a_ece: number;
    group_b_ece: number;
    sample_sizes: [number, number];
    p_value: number;
    confidence_interval: [number, number];
  } | null {
    if (!this.currentTest) {
      return null;
    }

    const { group_a_queries, group_b_queries, config } = this.currentTest;
    
    // Check minimum sample size
    if (group_a_queries.length < config.sample_size || 
        group_b_queries.length < config.sample_size) {
      return null;
    }

    // Calculate ECE for both groups
    const groupAMetrics = this.eceCalculator.calculateECE(group_a_queries);
    const groupBMetrics = this.eceCalculator.calculateECE(group_b_queries);
    
    const driftMagnitude = Math.abs(groupAMetrics.ece - groupBMetrics.ece);
    const hasDrift = driftMagnitude > config.drift_threshold;

    // Simple statistical test (would use proper bootstrap/permutation in production)
    const pooledVariance = this.calculatePooledVariance(group_a_queries, group_b_queries);
    const standardError = Math.sqrt(pooledVariance * (1/group_a_queries.length + 1/group_b_queries.length));
    const tStatistic = driftMagnitude / standardError;
    const pValue = 2 * (1 - this.normalCDF(Math.abs(tStatistic))); // Two-tailed test

    return {
      has_drift: hasDrift,
      drift_magnitude: driftMagnitude,
      group_a_ece: groupAMetrics.ece,
      group_b_ece: groupBMetrics.ece,
      sample_sizes: [group_a_queries.length, group_b_queries.length],
      p_value: pValue,
      confidence_interval: [
        driftMagnitude - 1.96 * standardError,
        driftMagnitude + 1.96 * standardError
      ]
    };
  }

  private calculatePooledVariance(groupA: IntentLanguageQuery[], groupB: IntentLanguageQuery[]): number {
    // Simplified variance calculation for ECE difference
    const allQueries = [...groupA, ...groupB];
    const allECEValues = allQueries.map(q => Math.abs(q.confidence - q.actual_relevance));
    
    const mean = allECEValues.reduce((sum, val) => sum + val, 0) / allECEValues.length;
    const variance = allECEValues.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (allECEValues.length - 1);
    
    return variance;
  }

  private normalCDF(z: number): number {
    // Approximation of normal CDF using error function
    return 0.5 * (1 + this.erf(z / Math.sqrt(2)));
  }

  private erf(x: number): number {
    // Abramowitz and Stegun approximation
    const a1 =  0.254829592;
    const a2 = -0.284496736;
    const a3 =  1.421413741;
    const a4 = -1.453152027;
    const a5 =  1.061405429;
    const p  =  0.3275911;

    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x);

    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

    return sign * y;
  }

  /**
   * Stop current A/A test and return results
   */
  stopAATest(): any {
    if (!this.currentTest) {
      return null;
    }

    const results = this.evaluateAATest();
    const testData = { ...this.currentTest, final_results: results };
    
    this.currentTest = null;
    
    console.log(`🏁 A/A test completed: ${testData.test_id}`);
    if (results) {
      console.log(`   Drift detected: ${results.has_drift ? 'YES' : 'NO'}`);
      console.log(`   Drift magnitude: ${results.drift_magnitude.toFixed(4)}`);
      console.log(`   Sample sizes: A=${results.sample_sizes[0]}, B=${results.sample_sizes[1]}`);
    }

    return testData;
  }
}

/**
 * Main ECE Drift Tracking System
 */
export class ECEDriftTracker extends EventEmitter {
  private eceCalculator: ECECalculator;
  private aaTestManager: AATestManager;
  private queryBuffer: Map<string, IntentLanguageQuery[]>; // intent_language -> queries
  private eceHistory: Map<string, ECEMetrics[]>; // intent_language -> metrics history
  private baselineECE: Map<string, number>; // intent_language -> baseline ECE
  private monitoringDir: string;
  private config: {
    ece_drift_threshold: number;
    kl_drift_threshold: number;
    miscoverage_threshold: number;
    buffer_size: number;
    evaluation_interval_minutes: number;
  };

  constructor(monitoringDir: string = './monitoring-data') {
    super();
    
    this.eceCalculator = new ECECalculator();
    this.aaTestManager = new AATestManager();
    this.queryBuffer = new Map();
    this.eceHistory = new Map();
    this.baselineECE = new Map();
    this.monitoringDir = monitoringDir;
    
    this.config = {
      ece_drift_threshold: 0.02,
      kl_drift_threshold: 0.02,
      miscoverage_threshold: 0.15,
      buffer_size: 1000,
      evaluation_interval_minutes: 5
    };

    if (!existsSync(this.monitoringDir)) {
      mkdirSync(this.monitoringDir, { recursive: true });
    }

    this.loadBaselines();
    this.startMonitoring();
  }

  /**
   * Record new query for ECE tracking
   */
  recordQuery(query: IntentLanguageQuery): void {
    const key = `${query.intent}_${query.language}`;
    
    if (!this.queryBuffer.has(key)) {
      this.queryBuffer.set(key, []);
    }
    
    const buffer = this.queryBuffer.get(key)!;
    buffer.push(query);
    
    // Maintain buffer size
    if (buffer.length > this.config.buffer_size) {
      buffer.shift(); // Remove oldest query
    }
    
    // Add to A/A test if running
    this.aaTestManager.addQueryToAATest(query);
  }

  /**
   * Start ECE monitoring with periodic evaluation
   */
  private startMonitoring(): void {
    setInterval(() => {
      this.evaluateAllMetrics();
    }, this.config.evaluation_interval_minutes * 60 * 1000);

    console.log(`📊 ECE drift tracking started`);
    console.log(`   Monitoring directory: ${this.monitoringDir}`);
    console.log(`   ECE drift threshold: ${this.config.ece_drift_threshold}`);
    console.log(`   KL drift threshold: ${this.config.kl_drift_threshold}`);
  }

  /**
   * Evaluate ECE metrics for all intent×language combinations
   */
  private evaluateAllMetrics(): void {
    const timestamp = new Date().toISOString();
    
    for (const [key, queries] of this.queryBuffer.entries()) {
      if (queries.length < 10) continue; // Minimum sample size
      
      const [intent, language] = key.split('_');
      
      // Calculate current ECE metrics
      const eceResults = this.eceCalculator.calculateECE(queries);
      
      const metrics: ECEMetrics = {
        timestamp,
        intent,
        language,
        total_queries: queries.length,
        ece_value: eceResults.ece,
        miscoverage_rate: eceResults.miscoverage_rate,
        confidence_bins: eceResults.bins,
        reliability_diagram: eceResults.reliability_points,
        kl_divergence: eceResults.kl_divergence,
        average_confidence: queries.reduce((sum, q) => sum + q.confidence, 0) / queries.length,
        average_accuracy: queries.reduce((sum, q) => sum + q.actual_relevance, 0) / queries.length
      };
      
      // Store metrics history
      if (!this.eceHistory.has(key)) {
        this.eceHistory.set(key, []);
      }
      
      const history = this.eceHistory.get(key)!;
      history.push(metrics);
      
      // Keep only recent history
      if (history.length > 100) {
        history.shift();
      }
      
      // Check for drift
      this.checkForDrift(key, metrics);
    }
    
    // Save current state
    this.saveMetricsState();
    
    // Evaluate A/A test
    const aaResults = this.aaTestManager.evaluateAATest();
    if (aaResults?.has_drift) {
      this.emitAlert({
        alert_id: `aa_drift_${Date.now()}`,
        alert_type: 'aa_drift',
        severity: 'critical',
        intent: 'mixed',
        language: 'mixed',
        current_value: aaResults.drift_magnitude,
        baseline_value: 0,
        drift_magnitude: aaResults.drift_magnitude,
        threshold_exceeded: this.config.ece_drift_threshold,
        timestamp,
        recommended_actions: [
          'Review recent system changes for A/A test drift',
          'Check for infrastructure issues affecting query processing',
          'Consider pausing traffic routing changes',
          'Investigate model serving consistency'
        ]
      });
    }
  }

  /**
   * Check for drift in ECE metrics
   */
  private checkForDrift(key: string, metrics: ECEMetrics): void {
    const baseline = this.baselineECE.get(key);
    if (!baseline) {
      // Set as new baseline if first measurement
      this.baselineECE.set(key, metrics.ece_value);
      return;
    }
    
    const eceDrift = Math.abs(metrics.ece_value - baseline);
    const klDrift = metrics.kl_divergence;
    const miscoverageSpike = metrics.miscoverage_rate;
    
    // Check ECE drift
    if (eceDrift > this.config.ece_drift_threshold) {
      this.emitAlert({
        alert_id: `ece_drift_${key}_${Date.now()}`,
        alert_type: 'ece_drift',
        severity: eceDrift > this.config.ece_drift_threshold * 2 ? 'critical' : 'warning',
        intent: metrics.intent,
        language: metrics.language,
        current_value: metrics.ece_value,
        baseline_value: baseline,
        drift_magnitude: eceDrift,
        threshold_exceeded: this.config.ece_drift_threshold,
        timestamp: metrics.timestamp,
        recommended_actions: this.getECEDriftActions(metrics)
      });
    }
    
    // Check KL drift
    if (klDrift > this.config.kl_drift_threshold) {
      this.emitAlert({
        alert_id: `kl_drift_${key}_${Date.now()}`,
        alert_type: 'kl_drift',
        severity: 'warning',
        intent: metrics.intent,
        language: metrics.language,
        current_value: klDrift,
        baseline_value: 0,
        drift_magnitude: klDrift,
        threshold_exceeded: this.config.kl_drift_threshold,
        timestamp: metrics.timestamp,
        recommended_actions: [
          'Review confidence calibration in model serving',
          'Check for changes in query distribution patterns',
          'Validate confidence score normalization',
          'Consider recalibrating confidence thresholds'
        ]
      });
    }
    
    // Check miscoverage spike
    if (miscoverageSpike > this.config.miscoverage_threshold) {
      this.emitAlert({
        alert_id: `miscoverage_${key}_${Date.now()}`,
        alert_type: 'miscoverage_spike',
        severity: 'warning',
        intent: metrics.intent,
        language: metrics.language,
        current_value: miscoverageSpike,
        baseline_value: this.config.miscoverage_threshold,
        drift_magnitude: miscoverageSpike - this.config.miscoverage_threshold,
        threshold_exceeded: this.config.miscoverage_threshold,
        timestamp: metrics.timestamp,
        recommended_actions: [
          'Check calibration on recent query types',
          'Review model uncertainty estimates',
          'Validate ground truth labeling quality',
          'Consider confidence score recalibration'
        ]
      });
    }
  }

  private getECEDriftActions(metrics: ECEMetrics): string[] {
    const actions = [
      'Review recent model or configuration changes',
      'Check calibration temperature in model serving',
      'Validate confidence score distribution changes',
      'Consider recalibrating model on recent data'
    ];
    
    if (metrics.ece_value > 0.1) {
      actions.unshift('URGENT: Consider immediate model rollback');
      actions.push('Escalate to ML engineering team');
    }
    
    return actions;
  }

  /**
   * Emit drift alert
   */
  private emitAlert(alert: DriftAlert): void {
    console.log(`🚨 ECE DRIFT ALERT [${alert.severity.toUpperCase()}]: ${alert.alert_type}`);
    console.log(`   Intent: ${alert.intent}, Language: ${alert.language}`);
    console.log(`   Current: ${alert.current_value.toFixed(4)}, Baseline: ${alert.baseline_value.toFixed(4)}`);
    console.log(`   Drift: ${alert.drift_magnitude.toFixed(4)}, Threshold: ${alert.threshold_exceeded.toFixed(4)}`);
    
    this.emit('drift_alert', alert);
  }

  /**
   * Start A/A shadow testing
   */
  startAATesting(config: AATestConfig): string {
    return this.aaTestManager.startAATest(config);
  }

  /**
   * Get current ECE status report
   */
  getECEStatusReport(): {
    timestamp: string;
    total_intent_language_combinations: number;
    active_alerts: number;
    overall_ece_health: 'healthy' | 'degraded' | 'critical';
    intent_language_breakdown: Array<{
      intent: string;
      language: string;
      current_ece: number;
      baseline_ece: number;
      query_count: number;
      drift_status: 'stable' | 'warning' | 'critical';
    }>;
    aa_test_status: any;
  } {
    const timestamp = new Date().toISOString();
    const breakdown = [];
    let totalAlerts = 0;
    
    for (const [key, history] of this.eceHistory.entries()) {
      if (history.length === 0) continue;
      
      const [intent, language] = key.split('_');
      const latest = history[history.length - 1];
      const baseline = this.baselineECE.get(key) || 0;
      const drift = Math.abs(latest.ece_value - baseline);
      
      let driftStatus: 'stable' | 'warning' | 'critical' = 'stable';
      if (drift > this.config.ece_drift_threshold * 2) {
        driftStatus = 'critical';
        totalAlerts++;
      } else if (drift > this.config.ece_drift_threshold) {
        driftStatus = 'warning';
        totalAlerts++;
      }
      
      breakdown.push({
        intent,
        language,
        current_ece: latest.ece_value,
        baseline_ece: baseline,
        query_count: latest.total_queries,
        drift_status: driftStatus
      });
    }
    
    const overallHealth = totalAlerts === 0 ? 'healthy' :
                         breakdown.some(b => b.drift_status === 'critical') ? 'critical' : 'degraded';
    
    return {
      timestamp,
      total_intent_language_combinations: breakdown.length,
      active_alerts: totalAlerts,
      overall_ece_health: overallHealth,
      intent_language_breakdown: breakdown,
      aa_test_status: this.aaTestManager.evaluateAATest()
    };
  }

  private loadBaselines(): void {
    try {
      const baselinePath = join(this.monitoringDir, 'ece_baselines.json');
      if (existsSync(baselinePath)) {
        const data = JSON.parse(readFileSync(baselinePath, 'utf8'));
        this.baselineECE = new Map(Object.entries(data));
        console.log(`📈 Loaded ${this.baselineECE.size} ECE baselines`);
      }
    } catch (error) {
      console.warn('⚠️ Failed to load ECE baselines:', error);
    }
  }

  private saveMetricsState(): void {
    try {
      // Save current metrics
      const metricsPath = join(this.monitoringDir, 'ece_metrics_current.json');
      const metricsData: Record<string, any> = {};
      
      for (const [key, history] of this.eceHistory.entries()) {
        metricsData[key] = history;
      }
      
      writeFileSync(metricsPath, JSON.stringify(metricsData, null, 2));
      
      // Save baselines
      const baselinePath = join(this.monitoringDir, 'ece_baselines.json');
      const baselineData: Record<string, number> = {};
      
      for (const [key, value] of this.baselineECE.entries()) {
        baselineData[key] = value;
      }
      
      writeFileSync(baselinePath, JSON.stringify(baselineData, null, 2));
      
    } catch (error) {
      console.error('❌ Failed to save ECE metrics state:', error);
    }
  }
}

// Global instance
export const globalECEDriftTracker = new ECEDriftTracker();