/**
 * Week One Post-GA Monitoring System (TODO.md Step 6)
 * 
 * Monitors stability during first week after GA:
 * - Anchor P@1/Recall@50 CUSUM monitoring
 * - Ladder positives-in-candidates tracking
 * - Stability validation for RAPTOR semantic-card rollout scheduling
 */

import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';
import { EventEmitter } from 'events';

interface AnchorCUSUMMetrics {
  // Current period metrics
  current_p_at_1: number;
  current_recall_at_50: number;
  
  // CUSUM tracking
  p_at_1_cusum_positive: number;
  p_at_1_cusum_negative: number;
  recall_at_50_cusum_positive: number;
  recall_at_50_cusum_negative: number;
  
  // Detection status
  p_at_1_alarm_active: boolean;
  recall_at_50_alarm_active: boolean;
  
  // Stability assessment
  consecutive_stable_hours: number;
  total_monitoring_hours: number;
  stability_score: number; // 0.0-1.0
}

interface LadderMetrics {
  // Current metrics
  positives_in_candidates: number;
  total_candidates: number;
  positive_rate: number;
  
  // Baseline comparison
  baseline_positive_rate: number;
  rate_change_pct: number;
  
  // Stability tracking
  rate_variance_last_24h: number;
  trend_direction: 'stable' | 'increasing' | 'decreasing' | 'volatile';
  
  // Quality indicators
  hard_negative_leakage_rate: number;
  false_positive_rate: number;
}

interface StabilityAssessment {
  overall_stability: 'stable' | 'trending' | 'volatile' | 'unstable';
  stability_factors: {
    anchor_metrics_stable: boolean;
    ladder_metrics_stable: boolean;
    no_cusum_alarms: boolean;
    variance_within_bounds: boolean;
  };
  
  // Readiness for next phase
  raptor_rollout_ready: boolean;
  readiness_criteria_met: string[];
  blocking_issues: string[];
  
  // Time tracking
  monitoring_start: string;
  hours_monitored: number;
  target_monitoring_hours: number; // 168 hours = 7 days
}

interface RAPTORRolloutPlan {
  // Rollout phases
  phases: Array<{
    name: string;
    description: string;
    target_strata: string[];
    rollout_percentage: number;
    duration_hours: number;
    success_criteria: Record<string, number>;
    abort_conditions: Record<string, number>;
  }>;
  
  // Current phase
  current_phase?: string;
  phase_start_time?: string;
  phase_progress_pct: number;
  
  // Scheduling
  tentative_start_date?: string;
  readiness_assessment_date: string;
  approval_required: boolean;
}

interface WeekOneMonitoringState {
  // Monitoring metadata
  monitoring_active: boolean;
  monitoring_start: string;
  target_end: string;
  
  // Core metrics
  anchor_cusum: AnchorCUSUMMetrics;
  ladder_metrics: LadderMetrics;
  stability_assessment: StabilityAssessment;
  
  // Future planning
  raptor_rollout_plan: RAPTORRolloutPlan;
  
  // Historical data
  hourly_snapshots: Array<{
    timestamp: string;
    anchor_p_at_1: number;
    anchor_recall_at_50: number;
    ladder_positive_rate: number;
    stability_score: number;
  }>;
  
  // Configuration
  stability_thresholds: {
    cusum_alarm_threshold: number;
    variance_threshold: number;
    trend_threshold: number;
    minimum_stable_hours: number;
  };
}

export class WeekOnePostGAMonitoring extends EventEmitter {
  private readonly monitoringDir: string;
  private monitoringState: WeekOneMonitoringState;
  private monitoringInterval?: NodeJS.Timeout;
  private snapshotInterval?: NodeJS.Timeout;
  private isRunning: boolean = false;
  
  constructor(monitoringDir: string = './deployment-artifacts/week-one-monitoring') {
    super();
    this.monitoringDir = monitoringDir;
    
    if (!existsSync(this.monitoringDir)) {
      mkdirSync(this.monitoringDir, { recursive: true });
    }
    
    this.monitoringState = this.initializeMonitoringState();
  }
  
  /**
   * Start week-one post-GA monitoring
   */
  public async startWeekOneMonitoring(): Promise<void> {
    if (this.isRunning) {
      console.log('📊 Week-one monitoring already running');
      return;
    }
    
    const startTime = new Date().toISOString();
    const endTime = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(); // 7 days from now
    
    console.log('🚀 Starting week-one post-GA monitoring...');
    console.log(`📅 Monitoring period: ${startTime} → ${endTime}`);
    
    // Initialize monitoring period
    this.monitoringState.monitoring_active = true;
    this.monitoringState.monitoring_start = startTime;
    this.monitoringState.target_end = endTime;
    this.monitoringState.stability_assessment.monitoring_start = startTime;
    
    // Load baseline metrics from deployment
    await this.loadBaselineMetrics();
    
    // Start monitoring intervals
    this.monitoringInterval = setInterval(async () => {
      await this.performStabilityCheck();
    }, 5 * 60 * 1000); // Every 5 minutes
    
    this.snapshotInterval = setInterval(async () => {
      await this.captureHourlySnapshot();
    }, 60 * 60 * 1000); // Every hour
    
    this.isRunning = true;
    
    console.log('✅ Week-one post-GA monitoring started');
    console.log('📊 Tracking: Anchor P@1/Recall@50 CUSUM, Ladder positives-in-candidates');
    console.log('🎯 Goal: Validate stability for RAPTOR semantic-card rollout scheduling');
    
    this.emit('week_one_monitoring_started', {
      start_time: startTime,
      end_time: endTime,
      target_hours: 168
    });
  }
  
  /**
   * Stop week-one monitoring
   */
  public stopWeekOneMonitoring(): void {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = undefined;
    }
    
    if (this.snapshotInterval) {
      clearInterval(this.snapshotInterval);
      this.snapshotInterval = undefined;
    }
    
    this.isRunning = false;
    this.monitoringState.monitoring_active = false;
    
    console.log('🛑 Week-one post-GA monitoring stopped');
    this.emit('week_one_monitoring_stopped');
  }
  
  /**
   * Load baseline metrics from version manager
   */
  private async loadBaselineMetrics(): Promise<void> {
    try {
      const { versionManager } = await import('./version-manager.js');
      const config = versionManager.loadVersionConfig();
      const baseline = config.baseline_metrics;
      
      // Initialize CUSUM baselines
      this.monitoringState.anchor_cusum = {
        current_p_at_1: baseline.p_at_1,
        current_recall_at_50: baseline.recall_at_50,
        p_at_1_cusum_positive: 0,
        p_at_1_cusum_negative: 0,
        recall_at_50_cusum_positive: 0,
        recall_at_50_cusum_negative: 0,
        p_at_1_alarm_active: false,
        recall_at_50_alarm_active: false,
        consecutive_stable_hours: 0,
        total_monitoring_hours: 0,
        stability_score: 1.0
      };
      
      // Initialize Ladder baselines
      this.monitoringState.ladder_metrics = {
        positives_in_candidates: baseline.ladder_positives || 180,
        total_candidates: baseline.ladder_total || 240,
        positive_rate: (baseline.ladder_positives || 180) / (baseline.ladder_total || 240),
        baseline_positive_rate: (baseline.ladder_positives || 180) / (baseline.ladder_total || 240),
        rate_change_pct: 0,
        rate_variance_last_24h: 0,
        trend_direction: 'stable',
        hard_negative_leakage_rate: 0.01, // 1% baseline
        false_positive_rate: 0.02 // 2% baseline
      };
      
      console.log('📈 Week-one baselines loaded:');
      console.log(`  Anchor P@1: ${baseline.p_at_1.toFixed(3)}, Recall@50: ${baseline.recall_at_50.toFixed(3)}`);
      console.log(`  Ladder positive rate: ${this.monitoringState.ladder_metrics.baseline_positive_rate.toFixed(3)} (${this.monitoringState.ladder_metrics.positives_in_candidates}/${this.monitoringState.ladder_metrics.total_candidates})`);
      
    } catch (error) {
      console.warn('⚠️  Failed to load baselines, using defaults:', error);
      this.initializeDefaultBaselines();
    }
  }
  
  /**
   * Initialize default baselines if version config unavailable
   */
  private initializeDefaultBaselines(): void {
    this.monitoringState.anchor_cusum = {
      current_p_at_1: 0.75,
      current_recall_at_50: 0.85,
      p_at_1_cusum_positive: 0,
      p_at_1_cusum_negative: 0,
      recall_at_50_cusum_positive: 0,
      recall_at_50_cusum_negative: 0,
      p_at_1_alarm_active: false,
      recall_at_50_alarm_active: false,
      consecutive_stable_hours: 0,
      total_monitoring_hours: 0,
      stability_score: 1.0
    };
    
    this.monitoringState.ladder_metrics = {
      positives_in_candidates: 180,
      total_candidates: 240,
      positive_rate: 0.75,
      baseline_positive_rate: 0.75,
      rate_change_pct: 0,
      rate_variance_last_24h: 0,
      trend_direction: 'stable',
      hard_negative_leakage_rate: 0.01,
      false_positive_rate: 0.02
    };
    
    console.log('📈 Default baselines initialized');
  }
  
  /**
   * Perform stability check
   */
  private async performStabilityCheck(): Promise<void> {
    const checkTime = new Date().toISOString();
    
    try {
      // 1. Update current metrics (mock - in production would query actual systems)
      await this.updateCurrentMetrics();
      
      // 2. Update CUSUM tracking
      this.updateCUSUMTracking();
      
      // 3. Assess stability
      this.assessStability();
      
      // 4. Update RAPTOR rollout readiness
      this.updateRAPTORRolloutReadiness();
      
      // 5. Save state
      this.saveMonitoringState();
      
      // Log stability status every hour
      const hoursMonitored = this.monitoringState.stability_assessment.hours_monitored;
      if (hoursMonitored > 0 && hoursMonitored % 1 === 0) { // On hour boundaries
        this.logStabilityStatus();
      }
      
      this.emit('stability_check_completed', {
        timestamp: checkTime,
        stability: this.monitoringState.stability_assessment.overall_stability,
        hours_monitored: hoursMonitored,
        raptor_ready: this.monitoringState.stability_assessment.raptor_rollout_ready
      });
      
    } catch (error) {
      console.error('❌ Failed stability check:', error);
      this.emit('stability_check_error', { timestamp: checkTime, error: error.message });
    }
  }
  
  /**
   * Update current metrics (mock implementation)
   */
  private async updateCurrentMetrics(): Promise<void> {
    // Mock current metrics with realistic variation around baseline
    const baseP1 = 0.75;
    const baseRecall50 = 0.85;
    const baseLadderRate = 0.75;
    
    // Add some realistic drift over time (small upward trend + noise)
    const hoursElapsed = (Date.now() - new Date(this.monitoringState.monitoring_start).getTime()) / (60 * 60 * 1000);
    const trendEffect = Math.min(0.02, hoursElapsed * 0.0001); // Small positive trend
    const noise = (Math.random() - 0.5) * 0.04; // ±2% noise
    
    // Update anchor metrics
    this.monitoringState.anchor_cusum.current_p_at_1 = Math.max(0.6, Math.min(0.9, baseP1 + trendEffect + noise));
    this.monitoringState.anchor_cusum.current_recall_at_50 = Math.max(0.7, Math.min(0.95, baseRecall50 + trendEffect * 0.8 + noise * 0.7));
    
    // Update ladder metrics
    const totalCandidates = Math.floor(240 + Math.random() * 20 - 10);
    const expectedPositives = Math.floor(totalCandidates * (baseLadderRate + trendEffect + noise * 0.5));
    
    this.monitoringState.ladder_metrics.total_candidates = totalCandidates;
    this.monitoringState.ladder_metrics.positives_in_candidates = Math.max(100, Math.min(totalCandidates, expectedPositives));
    this.monitoringState.ladder_metrics.positive_rate = this.monitoringState.ladder_metrics.positives_in_candidates / totalCandidates;
    
    // Calculate rate change
    this.monitoringState.ladder_metrics.rate_change_pct = 
      ((this.monitoringState.ladder_metrics.positive_rate - this.monitoringState.ladder_metrics.baseline_positive_rate) / this.monitoringState.ladder_metrics.baseline_positive_rate) * 100;
    
    // Update quality indicators
    this.monitoringState.ladder_metrics.hard_negative_leakage_rate = Math.max(0, 0.01 + (Math.random() - 0.5) * 0.008);
    this.monitoringState.ladder_metrics.false_positive_rate = Math.max(0, 0.02 + (Math.random() - 0.5) * 0.01);
  }
  
  /**
   * Update CUSUM tracking for drift detection
   */
  private updateCUSUMTracking(): void {
    // CUSUM parameters
    const cusumThreshold = this.monitoringState.stability_thresholds.cusum_alarm_threshold;
    const drift = 0.5; // Drift to detect (half-sigma)
    
    // P@1 CUSUM
    const p1Target = 0.75; // Target from baseline
    const p1Std = 0.075; // 10% relative std
    const p1Deviation = (this.monitoringState.anchor_cusum.current_p_at_1 - p1Target) / p1Std;
    
    this.monitoringState.anchor_cusum.p_at_1_cusum_positive = Math.max(0, 
      this.monitoringState.anchor_cusum.p_at_1_cusum_positive + p1Deviation - drift);
    this.monitoringState.anchor_cusum.p_at_1_cusum_negative = Math.max(0, 
      this.monitoringState.anchor_cusum.p_at_1_cusum_negative - p1Deviation - drift);
    
    // Recall@50 CUSUM
    const recallTarget = 0.85;
    const recallStd = 0.043; // 5% relative std
    const recallDeviation = (this.monitoringState.anchor_cusum.current_recall_at_50 - recallTarget) / recallStd;
    
    this.monitoringState.anchor_cusum.recall_at_50_cusum_positive = Math.max(0,
      this.monitoringState.anchor_cusum.recall_at_50_cusum_positive + recallDeviation - drift);
    this.monitoringState.anchor_cusum.recall_at_50_cusum_negative = Math.max(0,
      this.monitoringState.anchor_cusum.recall_at_50_cusum_negative - recallDeviation - drift);
    
    // Update alarm status
    const wasP1Alarm = this.monitoringState.anchor_cusum.p_at_1_alarm_active;
    const wasRecallAlarm = this.monitoringState.anchor_cusum.recall_at_50_alarm_active;
    
    this.monitoringState.anchor_cusum.p_at_1_alarm_active = 
      this.monitoringState.anchor_cusum.p_at_1_cusum_positive > cusumThreshold || 
      this.monitoringState.anchor_cusum.p_at_1_cusum_negative > cusumThreshold;
      
    this.monitoringState.anchor_cusum.recall_at_50_alarm_active = 
      this.monitoringState.anchor_cusum.recall_at_50_cusum_positive > cusumThreshold || 
      this.monitoringState.anchor_cusum.recall_at_50_cusum_negative > cusumThreshold;
    
    // Log alarm state changes
    if (this.monitoringState.anchor_cusum.p_at_1_alarm_active && !wasP1Alarm) {
      console.log('🚨 CUSUM ALARM: Anchor P@1 drift detected');
      this.emit('cusum_alarm_triggered', { metric: 'anchor_p_at_1', value: this.monitoringState.anchor_cusum.current_p_at_1 });
    }
    if (this.monitoringState.anchor_cusum.recall_at_50_alarm_active && !wasRecallAlarm) {
      console.log('🚨 CUSUM ALARM: Recall@50 drift detected');
      this.emit('cusum_alarm_triggered', { metric: 'recall_at_50', value: this.monitoringState.anchor_cusum.current_recall_at_50 });
    }
    
    // Clear alarms
    if (!this.monitoringState.anchor_cusum.p_at_1_alarm_active && wasP1Alarm) {
      console.log('✅ CUSUM ALARM CLEARED: Anchor P@1');
      this.emit('cusum_alarm_cleared', { metric: 'anchor_p_at_1' });
    }
    if (!this.monitoringState.anchor_cusum.recall_at_50_alarm_active && wasRecallAlarm) {
      console.log('✅ CUSUM ALARM CLEARED: Recall@50');
      this.emit('cusum_alarm_cleared', { metric: 'recall_at_50' });
    }
  }
  
  /**
   * Assess overall stability
   */
  private assessStability(): void {
    const hoursMonitored = (Date.now() - new Date(this.monitoringState.monitoring_start).getTime()) / (60 * 60 * 1000);
    this.monitoringState.stability_assessment.hours_monitored = hoursMonitored;
    
    // Stability factors
    const factors = {
      anchor_metrics_stable: !this.monitoringState.anchor_cusum.p_at_1_alarm_active && 
                            !this.monitoringState.anchor_cusum.recall_at_50_alarm_active,
      ladder_metrics_stable: Math.abs(this.monitoringState.ladder_metrics.rate_change_pct) < 5, // Within 5%
      no_cusum_alarms: !this.monitoringState.anchor_cusum.p_at_1_alarm_active && 
                       !this.monitoringState.anchor_cusum.recall_at_50_alarm_active,
      variance_within_bounds: this.monitoringState.ladder_metrics.rate_variance_last_24h < this.monitoringState.stability_thresholds.variance_threshold
    };
    
    this.monitoringState.stability_assessment.stability_factors = factors;
    
    // Overall stability assessment
    const stableFactors = Object.values(factors).filter(Boolean).length;
    
    if (stableFactors === 4) {
      this.monitoringState.stability_assessment.overall_stability = 'stable';
    } else if (stableFactors === 3) {
      this.monitoringState.stability_assessment.overall_stability = 'trending';
    } else if (stableFactors === 2) {
      this.monitoringState.stability_assessment.overall_stability = 'volatile';
    } else {
      this.monitoringState.stability_assessment.overall_stability = 'unstable';
    }
    
    // Update consecutive stable hours
    if (this.monitoringState.stability_assessment.overall_stability === 'stable') {
      this.monitoringState.anchor_cusum.consecutive_stable_hours++;
    } else {
      this.monitoringState.anchor_cusum.consecutive_stable_hours = 0;
    }
    
    // Calculate stability score (0.0-1.0)
    const maxScore = 1.0;
    const stabilityPenalty = {
      'stable': 0,
      'trending': 0.1,
      'volatile': 0.3,
      'unstable': 0.6
    };
    
    const alarmPenalty = (this.monitoringState.anchor_cusum.p_at_1_alarm_active ? 0.2 : 0) + 
                        (this.monitoringState.anchor_cusum.recall_at_50_alarm_active ? 0.2 : 0);
    
    this.monitoringState.anchor_cusum.stability_score = Math.max(0, 
      maxScore - stabilityPenalty[this.monitoringState.stability_assessment.overall_stability] - alarmPenalty);
    
    // Update Ladder trend analysis
    this.updateLadderTrend();
  }
  
  /**
   * Update Ladder trend analysis
   */
  private updateLadderTrend(): void {
    // Simple trend analysis based on recent snapshots
    if (this.monitoringState.hourly_snapshots.length < 6) {
      this.monitoringState.ladder_metrics.trend_direction = 'stable';
      return;
    }
    
    const recent = this.monitoringState.hourly_snapshots.slice(-6); // Last 6 hours
    const rates = recent.map(s => s.ladder_positive_rate);
    const trend = this.calculateTrend(rates);
    
    if (Math.abs(trend) < 0.001) {
      this.monitoringState.ladder_metrics.trend_direction = 'stable';
    } else if (trend > 0.002) {
      this.monitoringState.ladder_metrics.trend_direction = 'increasing';
    } else if (trend < -0.002) {
      this.monitoringState.ladder_metrics.trend_direction = 'decreasing';
    } else {
      this.monitoringState.ladder_metrics.trend_direction = 'volatile';
    }
    
    // Calculate variance
    const mean = rates.reduce((a, b) => a + b, 0) / rates.length;
    const variance = rates.reduce((acc, rate) => acc + Math.pow(rate - mean, 2), 0) / rates.length;
    this.monitoringState.ladder_metrics.rate_variance_last_24h = variance;
  }
  
  /**
   * Calculate simple linear trend
   */
  private calculateTrend(values: number[]): number {
    const n = values.length;
    const sumX = (n * (n - 1)) / 2; // 0 + 1 + 2 + ... + (n-1)
    const sumY = values.reduce((a, b) => a + b, 0);
    const sumXY = values.reduce((acc, y, x) => acc + x * y, 0);
    const sumXX = (n * (n - 1) * (2 * n - 1)) / 6; // Sum of squares
    
    return (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  }
  
  /**
   * Update RAPTOR rollout readiness
   */
  private updateRAPTORRolloutReadiness(): void {
    const minStableHours = this.monitoringState.stability_thresholds.minimum_stable_hours;
    const hoursMonitored = this.monitoringState.stability_assessment.hours_monitored;
    
    // Readiness criteria
    const criteria = {
      sufficient_monitoring_time: hoursMonitored >= 24, // At least 24 hours
      stability_maintained: this.monitoringState.anchor_cusum.consecutive_stable_hours >= minStableHours,
      no_active_alarms: !this.monitoringState.anchor_cusum.p_at_1_alarm_active && 
                       !this.monitoringState.anchor_cusum.recall_at_50_alarm_active,
      ladder_metrics_healthy: this.monitoringState.ladder_metrics.trend_direction !== 'volatile' && 
                             Math.abs(this.monitoringState.ladder_metrics.rate_change_pct) < 10,
      stability_score_good: this.monitoringState.anchor_cusum.stability_score >= 0.8
    };
    
    // Update readiness status
    const metCriteria = Object.entries(criteria).filter(([_, met]) => met).map(([name, _]) => name);
    const blockingIssues = Object.entries(criteria).filter(([_, met]) => !met).map(([name, _]) => name);
    
    this.monitoringState.stability_assessment.readiness_criteria_met = metCriteria;
    this.monitoringState.stability_assessment.blocking_issues = blockingIssues;
    this.monitoringState.stability_assessment.raptor_rollout_ready = blockingIssues.length === 0;
    
    // Update RAPTOR rollout plan
    if (this.monitoringState.stability_assessment.raptor_rollout_ready && !this.monitoringState.raptor_rollout_plan.tentative_start_date) {
      // Schedule tentative start date (3 days from readiness)
      const tentativeStart = new Date(Date.now() + 3 * 24 * 60 * 60 * 1000).toISOString();
      this.monitoringState.raptor_rollout_plan.tentative_start_date = tentativeStart;
      
      console.log('🎯 RAPTOR rollout readiness achieved!');
      console.log(`📅 Tentative rollout start: ${tentativeStart}`);
      
      this.emit('raptor_rollout_ready', {
        readiness_achieved: new Date().toISOString(),
        tentative_start: tentativeStart,
        hours_to_readiness: hoursMonitored,
        stability_score: this.monitoringState.anchor_cusum.stability_score
      });
    }
  }
  
  /**
   * Capture hourly snapshot
   */
  private async captureHourlySnapshot(): Promise<void> {
    const snapshot = {
      timestamp: new Date().toISOString(),
      anchor_p_at_1: this.monitoringState.anchor_cusum.current_p_at_1,
      anchor_recall_at_50: this.monitoringState.anchor_cusum.current_recall_at_50,
      ladder_positive_rate: this.monitoringState.ladder_metrics.positive_rate,
      stability_score: this.monitoringState.anchor_cusum.stability_score
    };
    
    this.monitoringState.hourly_snapshots.push(snapshot);
    
    // Keep only last 7 days of snapshots
    const cutoff = Date.now() - 7 * 24 * 60 * 60 * 1000;
    this.monitoringState.hourly_snapshots = this.monitoringState.hourly_snapshots.filter(
      s => new Date(s.timestamp).getTime() > cutoff
    );
    
    console.log(`📊 Hourly snapshot captured - P@1: ${snapshot.anchor_p_at_1.toFixed(3)}, R@50: ${snapshot.anchor_recall_at_50.toFixed(3)}, Ladder: ${snapshot.ladder_positive_rate.toFixed(3)}, Stability: ${snapshot.stability_score.toFixed(3)}`);
    
    this.emit('hourly_snapshot_captured', snapshot);
  }
  
  /**
   * Log stability status (every hour)
   */
  private logStabilityStatus(): void {
    const hours = Math.floor(this.monitoringState.stability_assessment.hours_monitored);
    const stability = this.monitoringState.stability_assessment.overall_stability;
    const stableHours = this.monitoringState.anchor_cusum.consecutive_stable_hours;
    const raptorReady = this.monitoringState.stability_assessment.raptor_rollout_ready;
    
    console.log(`\n📋 === WEEK ONE MONITORING STATUS (${hours}h monitored) ===`);
    console.log(`🎯 Overall Stability: ${stability.toUpperCase()} (${stableHours}h consecutive stable)`);
    console.log(`📊 Anchor Metrics: P@1=${this.monitoringState.anchor_cusum.current_p_at_1.toFixed(3)}, R@50=${this.monitoringState.anchor_cusum.current_recall_at_50.toFixed(3)}`);
    console.log(`📈 Ladder Rate: ${(this.monitoringState.ladder_metrics.positive_rate * 100).toFixed(1)}% (${this.monitoringState.ladder_metrics.rate_change_pct >= 0 ? '+' : ''}${this.monitoringState.ladder_metrics.rate_change_pct.toFixed(1)}% vs baseline)`);
    console.log(`🚨 CUSUM Alarms: P@1=${this.monitoringState.anchor_cusum.p_at_1_alarm_active ? 'ACTIVE' : 'clear'}, R@50=${this.monitoringState.anchor_cusum.recall_at_50_alarm_active ? 'ACTIVE' : 'clear'}`);
    console.log(`🎯 RAPTOR Rollout Ready: ${raptorReady ? 'YES ✅' : 'NO ⏳'}`);
    
    if (raptorReady && this.monitoringState.raptor_rollout_plan.tentative_start_date) {
      console.log(`📅 Tentative RAPTOR Rollout: ${this.monitoringState.raptor_rollout_plan.tentative_start_date}`);
    }
    
    if (this.monitoringState.stability_assessment.blocking_issues.length > 0) {
      console.log(`⚠️  Blocking Issues: ${this.monitoringState.stability_assessment.blocking_issues.join(', ')}`);
    }
    
    console.log('==================================================\n');
  }
  
  /**
   * Get current monitoring status
   */
  public getMonitoringStatus(): WeekOneMonitoringState {
    return { ...this.monitoringState };
  }
  
  /**
   * Get week-one dashboard data
   */
  public getWeekOneDashboardData(): any {
    const hoursRemaining = Math.max(0, 168 - this.monitoringState.stability_assessment.hours_monitored);
    
    return {
      timestamp: new Date().toISOString(),
      monitoring_active: this.monitoringState.monitoring_active,
      
      // Progress tracking
      progress: {
        hours_monitored: this.monitoringState.stability_assessment.hours_monitored,
        hours_remaining: hoursRemaining,
        progress_pct: Math.min(100, (this.monitoringState.stability_assessment.hours_monitored / 168) * 100),
        target_completion: this.monitoringState.target_end
      },
      
      // Current metrics
      current_metrics: {
        anchor_p_at_1: this.monitoringState.anchor_cusum.current_p_at_1,
        anchor_recall_at_50: this.monitoringState.anchor_cusum.current_recall_at_50,
        ladder_positive_rate: this.monitoringState.ladder_metrics.positive_rate,
        ladder_rate_change_pct: this.monitoringState.ladder_metrics.rate_change_pct
      },
      
      // Stability assessment
      stability: {
        overall_status: this.monitoringState.stability_assessment.overall_stability,
        stability_score: this.monitoringState.anchor_cusum.stability_score,
        consecutive_stable_hours: this.monitoringState.anchor_cusum.consecutive_stable_hours,
        stability_factors: this.monitoringState.stability_assessment.stability_factors
      },
      
      // CUSUM status
      cusum_status: {
        p_at_1_alarm: this.monitoringState.anchor_cusum.p_at_1_alarm_active,
        recall_at_50_alarm: this.monitoringState.anchor_cusum.recall_at_50_alarm_active,
        p_at_1_cusum_positive: this.monitoringState.anchor_cusum.p_at_1_cusum_positive,
        p_at_1_cusum_negative: this.monitoringState.anchor_cusum.p_at_1_cusum_negative,
        recall_at_50_cusum_positive: this.monitoringState.anchor_cusum.recall_at_50_cusum_positive,
        recall_at_50_cusum_negative: this.monitoringState.anchor_cusum.recall_at_50_cusum_negative
      },
      
      // RAPTOR rollout readiness
      raptor_rollout: {
        ready: this.monitoringState.stability_assessment.raptor_rollout_ready,
        tentative_start: this.monitoringState.raptor_rollout_plan.tentative_start_date,
        criteria_met: this.monitoringState.stability_assessment.readiness_criteria_met,
        blocking_issues: this.monitoringState.stability_assessment.blocking_issues,
        approval_required: this.monitoringState.raptor_rollout_plan.approval_required
      },
      
      // Historical data
      hourly_trends: this.monitoringState.hourly_snapshots.slice(-48), // Last 48 hours
      
      // Quality indicators
      quality_indicators: {
        hard_negative_leakage: this.monitoringState.ladder_metrics.hard_negative_leakage_rate,
        false_positive_rate: this.monitoringState.ladder_metrics.false_positive_rate,
        trend_direction: this.monitoringState.ladder_metrics.trend_direction,
        variance_last_24h: this.monitoringState.ladder_metrics.rate_variance_last_24h
      }
    };
  }
  
  private initializeMonitoringState(): WeekOneMonitoringState {
    return {
      monitoring_active: false,
      monitoring_start: '',
      target_end: '',
      
      anchor_cusum: {
        current_p_at_1: 0.75,
        current_recall_at_50: 0.85,
        p_at_1_cusum_positive: 0,
        p_at_1_cusum_negative: 0,
        recall_at_50_cusum_positive: 0,
        recall_at_50_cusum_negative: 0,
        p_at_1_alarm_active: false,
        recall_at_50_alarm_active: false,
        consecutive_stable_hours: 0,
        total_monitoring_hours: 0,
        stability_score: 1.0
      },
      
      ladder_metrics: {
        positives_in_candidates: 180,
        total_candidates: 240,
        positive_rate: 0.75,
        baseline_positive_rate: 0.75,
        rate_change_pct: 0,
        rate_variance_last_24h: 0,
        trend_direction: 'stable',
        hard_negative_leakage_rate: 0.01,
        false_positive_rate: 0.02
      },
      
      stability_assessment: {
        overall_stability: 'stable',
        stability_factors: {
          anchor_metrics_stable: true,
          ladder_metrics_stable: true,
          no_cusum_alarms: true,
          variance_within_bounds: true
        },
        raptor_rollout_ready: false,
        readiness_criteria_met: [],
        blocking_issues: ['sufficient_monitoring_time'],
        monitoring_start: '',
        hours_monitored: 0,
        target_monitoring_hours: 168
      },
      
      raptor_rollout_plan: {
        phases: [
          {
            name: 'NL Strata Pilot',
            description: 'Natural language queries, 25% traffic',
            target_strata: ['natural_language'],
            rollout_percentage: 25,
            duration_hours: 48,
            success_criteria: { p_at_1: 0.75, recall_at_50: 0.85 },
            abort_conditions: { p_at_1: 0.70, recall_at_50: 0.80 }
          },
          {
            name: 'Struct Strata Pilot',
            description: 'Structured queries, 25% traffic',
            target_strata: ['structured'],
            rollout_percentage: 25,
            duration_hours: 48,
            success_criteria: { p_at_1: 0.75, recall_at_50: 0.85 },
            abort_conditions: { p_at_1: 0.70, recall_at_50: 0.80 }
          },
          {
            name: 'Full Rollout',
            description: 'All queries, 100% traffic',
            target_strata: ['natural_language', 'structured', 'mixed'],
            rollout_percentage: 100,
            duration_hours: 168,
            success_criteria: { p_at_1: 0.75, recall_at_50: 0.85 },
            abort_conditions: { p_at_1: 0.70, recall_at_50: 0.80 }
          }
        ],
        phase_progress_pct: 0,
        readiness_assessment_date: new Date().toISOString(),
        approval_required: true
      },
      
      hourly_snapshots: [],
      
      stability_thresholds: {
        cusum_alarm_threshold: 3.0,  // CUSUM threshold for drift detection
        variance_threshold: 0.01,    // Max variance for stability (1%)
        trend_threshold: 0.02,       // Max trend magnitude (2%)
        minimum_stable_hours: 12     // Hours of stability required for RAPTOR readiness
      }
    };
  }
  
  private saveMonitoringState(): void {
    const statePath = join(this.monitoringDir, 'week_one_monitoring_state.json');
    writeFileSync(statePath, JSON.stringify(this.monitoringState, null, 2));
  }
}

export const weekOnePostGAMonitoring = new WeekOnePostGAMonitoring();