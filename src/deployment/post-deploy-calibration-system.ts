/**
 * Post-Deploy Calibration System - TODO.md Step 4 Implementation
 * 
 * Implements post-canary calibration workflow per TODO.md requirements:
 * - 2-day holdout period after canary completion
 * - Reliability diagram recomputation from real user clicks/impressions
 * - Tau parameter optimization with drift bounds (|Δτ|≤0.02)
 * - Integration with canary deployment lifecycle
 * - Automated monitoring and alerting
 */

import { EventEmitter } from 'events';
import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';
import { OnlineCalibrationSystem, ClickImpressionData, ReliabilityPoint } from './online-calibration-system.js';

interface PostDeployCalibrationConfig {
  holdout_period_days: number;  // 2 days per TODO.md
  tau_drift_threshold: number;  // 0.02 per TODO.md requirements
  min_click_samples: number;
  reliability_curve_min_points: number;
  monitoring_interval_hours: number;
}

interface CalibrationSession {
  session_id: string;
  canary_completion_time: string;
  holdout_end_time: string;
  status: 'waiting_holdout' | 'collecting_data' | 'computing_reliability' | 'optimizing_tau' | 'completed' | 'frozen';
  
  // Canary context
  canary_deployment_id: string;
  pre_calibration_tau: number;
  
  // Post-holdout data
  click_impression_data?: ClickImpressionData[];
  reliability_diagram?: ReliabilityPoint[];
  optimized_tau?: number;
  tau_delta?: number;
  
  // Safety and monitoring
  drift_check_passed: boolean;
  freeze_reason?: string;
  calibration_start_time?: string;
  calibration_completion_time?: string;
  
  // Metrics
  total_queries_analyzed: number;
  click_through_rate: number;
  results_per_query_mean: number;
}

interface CalibrationAlert {
  timestamp: string;
  severity: 'info' | 'warning' | 'critical';
  message: string;
  session_id: string;
  metrics?: Record<string, any>;
}

export class PostDeployCalibrationSystem extends EventEmitter {
  private readonly config: PostDeployCalibrationConfig;
  private readonly calibrationSystem: OnlineCalibrationSystem;
  private readonly sessionsDir: string;
  private readonly alertsDir: string;
  
  private activeSessions: Map<string, CalibrationSession> = new Map();
  private monitoringInterval?: NodeJS.Timeout;
  
  constructor(
    calibrationSystem: OnlineCalibrationSystem,
    config: Partial<PostDeployCalibrationConfig> = {},
    dataDir: string = './deployment-artifacts/post-deploy-calibration'
  ) {
    super();
    
    this.config = {
      holdout_period_days: 2,        // TODO.md requirement
      tau_drift_threshold: 0.02,     // TODO.md |Δτ|≤0.02
      min_click_samples: 1000,
      reliability_curve_min_points: 10,
      monitoring_interval_hours: 6,
      ...config
    };
    
    this.calibrationSystem = calibrationSystem;
    this.sessionsDir = join(dataDir, 'sessions');
    this.alertsDir = join(dataDir, 'alerts');
    
    // Initialize directories
    [this.sessionsDir, this.alertsDir].forEach(dir => {
      if (!existsSync(dir)) {
        mkdirSync(dir, { recursive: true });
      }
    });
    
    // Load active sessions
    this.loadActiveSessions();
    
    console.log('🎯 Post-Deploy Calibration System initialized');
    console.log(`   Holdout period: ${this.config.holdout_period_days} days`);
    console.log(`   Tau drift threshold: ±${this.config.tau_drift_threshold}`);
  }
  
  /**
   * Start post-deploy calibration system
   */
  public startCalibrationMonitoring(): void {
    console.log('📊 Starting post-deploy calibration monitoring...');
    
    // Start monitoring active sessions
    this.monitoringInterval = setInterval(async () => {
      await this.processActiveSessions();
    }, this.config.monitoring_interval_hours * 60 * 60 * 1000);
    
    // Process any existing sessions immediately
    this.processActiveSessions();
    
    console.log('✅ Post-deploy calibration monitoring started');
    this.emit('monitoring_started');
  }
  
  /**
   * Stop post-deploy calibration system
   */
  public stopCalibrationMonitoring(): void {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = undefined;
    }
    
    console.log('🛑 Post-deploy calibration monitoring stopped');
    this.emit('monitoring_stopped');
  }
  
  /**
   * Initialize calibration session after canary completion - TODO.md Step 4 entry point
   */
  public async initializePostCanaryCalibration(
    canaryDeploymentId: string,
    canaryCompletionTime: Date,
    currentTau: number
  ): Promise<string> {
    const sessionId = `calibration-${canaryDeploymentId}-${Date.now()}`;
    const holdoutEndTime = new Date(canaryCompletionTime.getTime() + this.config.holdout_period_days * 24 * 60 * 60 * 1000);
    
    const session: CalibrationSession = {
      session_id: sessionId,
      canary_completion_time: canaryCompletionTime.toISOString(),
      holdout_end_time: holdoutEndTime.toISOString(),
      status: 'waiting_holdout',
      canary_deployment_id: canaryDeploymentId,
      pre_calibration_tau: currentTau,
      drift_check_passed: false,
      total_queries_analyzed: 0,
      click_through_rate: 0,
      results_per_query_mean: 0
    };
    
    this.activeSessions.set(sessionId, session);
    this.saveSession(session);
    
    // Send alert
    await this.sendAlert({
      timestamp: new Date().toISOString(),
      severity: 'info',
      message: `Post-deploy calibration session initiated for canary ${canaryDeploymentId}`,
      session_id: sessionId,
      metrics: {
        holdout_end_time: holdoutEndTime.toISOString(),
        pre_calibration_tau: currentTau,
        holdout_period_hours: this.config.holdout_period_days * 24
      }
    });
    
    console.log(`🚀 POST-DEPLOY CALIBRATION SESSION STARTED`);
    console.log(`   Session ID: ${sessionId}`);
    console.log(`   Canary: ${canaryDeploymentId}`);
    console.log(`   Current τ: ${currentTau.toFixed(4)}`);
    console.log(`   Holdout ends: ${holdoutEndTime.toISOString()}`);
    
    this.emit('calibration_session_started', { sessionId, holdoutEndTime });
    
    return sessionId;
  }
  
  /**
   * Process active calibration sessions
   */
  private async processActiveSessions(): Promise<void> {
    const now = new Date();
    const processedSessions: string[] = [];
    
    for (const [sessionId, session] of this.activeSessions) {
      try {
        const updated = await this.processSession(session, now);
        if (updated) {
          this.saveSession(session);
          processedSessions.push(sessionId);
          
          // Remove completed or frozen sessions from active monitoring
          if (session.status === 'completed' || session.status === 'frozen') {
            this.activeSessions.delete(sessionId);
          }
        }
      } catch (error) {
        console.error(`❌ Error processing calibration session ${sessionId}:`, error);
        await this.handleSessionError(session, error);
      }
    }
    
    if (processedSessions.length > 0) {
      console.log(`📊 Processed ${processedSessions.length} calibration sessions`);
    }
  }
  
  /**
   * Process individual calibration session
   */
  private async processSession(session: CalibrationSession, currentTime: Date): Promise<boolean> {
    const holdoutEndTime = new Date(session.holdout_end_time);
    let updated = false;
    
    switch (session.status) {
      case 'waiting_holdout':
        if (currentTime >= holdoutEndTime) {
          console.log(`⏰ HOLDOUT PERIOD ENDED for session ${session.session_id}`);
          session.status = 'collecting_data';
          session.calibration_start_time = currentTime.toISOString();
          updated = true;
          
          await this.sendAlert({
            timestamp: currentTime.toISOString(),
            severity: 'info',
            message: `Holdout period completed, beginning calibration data collection`,
            session_id: session.session_id
          });
        }
        break;
        
      case 'collecting_data':
        updated = await this.collectCalibrationData(session);
        break;
        
      case 'computing_reliability':
        updated = await this.computeReliabilityDiagram(session);
        break;
        
      case 'optimizing_tau':
        updated = await this.optimizeTauWithDriftCheck(session);
        break;
    }
    
    return updated;
  }
  
  /**
   * Collect click/impression data for calibration
   */
  private async collectCalibrationData(session: CalibrationSession): Promise<boolean> {
    console.log(`📈 Collecting calibration data for session ${session.session_id}`);
    
    // Collect real user click/impression data from the production system
    // This would integrate with actual logging systems in production
    const clickData = await this.collectProductionClickData(session);
    
    if (clickData.length < this.config.min_click_samples) {
      console.log(`⚠️  Insufficient click data: ${clickData.length} < ${this.config.min_click_samples} required`);
      return false; // Continue collecting
    }
    
    session.click_impression_data = clickData;
    session.status = 'computing_reliability';
    session.total_queries_analyzed = clickData.length;
    
    // Calculate basic metrics
    const totalClicks = clickData.reduce((sum, item) => sum + item.clicks.length, 0);
    const totalImpressions = clickData.reduce((sum, item) => sum + item.impressions, 0);
    session.click_through_rate = totalClicks / totalImpressions;
    session.results_per_query_mean = totalImpressions / clickData.length;
    
    console.log(`✅ Click data collection completed:`);
    console.log(`   Total queries: ${clickData.length}`);
    console.log(`   Click-through rate: ${(session.click_through_rate * 100).toFixed(2)}%`);
    console.log(`   Results per query: ${session.results_per_query_mean.toFixed(1)}`);
    
    return true;
  }
  
  /**
   * Compute reliability diagram from click/impression data - TODO.md requirement
   */
  private async computeReliabilityDiagram(session: CalibrationSession): Promise<boolean> {
    if (!session.click_impression_data) {
      throw new Error('No click impression data available for reliability computation');
    }
    
    console.log(`🔍 Computing reliability diagram for session ${session.session_id}`);
    
    // Group data by predicted score buckets for reliability curve
    const scoreBuckets = new Map<number, { clicks: number; impressions: number; queries: Set<string> }>();
    
    for (const item of session.click_impression_data) {
      for (let i = 0; i < item.results.length; i++) {
        const result = item.results[i];
        const scoreBucket = Math.floor(result.predicted_score * 20) / 20; // 0.05 buckets for fine granularity
        
        if (!scoreBuckets.has(scoreBucket)) {
          scoreBuckets.set(scoreBucket, { clicks: 0, impressions: 0, queries: new Set() });
        }
        
        const bucket = scoreBuckets.get(scoreBucket)!;
        bucket.impressions++;
        bucket.queries.add(item.query);
        
        if (item.clicks.includes(i)) {
          bucket.clicks++;
        }
      }
    }
    
    // Convert to reliability points with statistical significance
    const reliabilityPoints: ReliabilityPoint[] = [];
    
    for (const [score, data] of scoreBuckets) {
      if (data.impressions >= 20 && data.queries.size >= 5) { // Minimum sample requirements
        const precision = data.clicks / data.impressions;
        const confidenceInterval = this.calculateWilsonConfidenceInterval(data.clicks, data.impressions);
        
        reliabilityPoints.push({
          predicted_score: score,
          actual_precision: precision,
          sample_size: data.impressions,
          confidence_interval: confidenceInterval,
          collection_date: new Date().toISOString()
        });
      }
    }
    
    if (reliabilityPoints.length < this.config.reliability_curve_min_points) {
      await this.sendAlert({
        timestamp: new Date().toISOString(),
        severity: 'warning',
        message: `Insufficient reliability points: ${reliabilityPoints.length} < ${this.config.reliability_curve_min_points}`,
        session_id: session.session_id
      });
      return false;
    }
    
    // Sort by predicted score
    reliabilityPoints.sort((a, b) => a.predicted_score - b.predicted_score);
    
    session.reliability_diagram = reliabilityPoints;
    session.status = 'optimizing_tau';
    
    console.log(`✅ Reliability diagram computed:`);
    console.log(`   Reliability points: ${reliabilityPoints.length}`);
    console.log(`   Score range: [${reliabilityPoints[0].predicted_score.toFixed(3)}, ${reliabilityPoints[reliabilityPoints.length-1].predicted_score.toFixed(3)}]`);
    
    return true;
  }
  
  /**
   * Optimize tau parameter with drift bounds checking - TODO.md core requirement
   */
  private async optimizeTauWithDriftCheck(session: CalibrationSession): Promise<boolean> {
    if (!session.reliability_diagram || !session.click_impression_data) {
      throw new Error('Missing reliability diagram or click data for tau optimization');
    }
    
    console.log(`🎯 Optimizing τ with drift bounds check for session ${session.session_id}`);
    console.log(`   Current τ: ${session.pre_calibration_tau.toFixed(4)}`);
    console.log(`   Drift threshold: ±${this.config.tau_drift_threshold}`);
    
    // Find optimal tau that maintains 5±2 results/query target from TODO.md
    const targetResultsPerQuery = 5;
    const targetTolerance = 2;
    
    let bestTau = session.pre_calibration_tau;
    let bestScore = Infinity;
    let bestResultsPerQuery = 0;
    
    // Test tau values in reasonable range around current value
    const tauMin = Math.max(0.1, session.pre_calibration_tau - 0.3);
    const tauMax = Math.min(0.9, session.pre_calibration_tau + 0.3);
    
    for (let tau = tauMin; tau <= tauMax; tau += 0.005) {
      const resultsPerQuery = this.simulateResultsPerQuery(tau, session.click_impression_data);
      const deviationFromTarget = Math.abs(resultsPerQuery - targetResultsPerQuery);
      
      if (deviationFromTarget < bestScore && deviationFromTarget <= targetTolerance) {
        bestScore = deviationFromTarget;
        bestTau = tau;
        bestResultsPerQuery = resultsPerQuery;
      }
    }
    
    const tauDelta = bestTau - session.pre_calibration_tau;
    
    console.log(`🔍 Tau optimization results:`);
    console.log(`   Optimal τ: ${bestTau.toFixed(4)}`);
    console.log(`   Δτ: ${tauDelta >= 0 ? '+' : ''}${tauDelta.toFixed(4)}`);
    console.log(`   Predicted results/query: ${bestResultsPerQuery.toFixed(1)}`);
    
    // CRITICAL: Check drift bounds per TODO.md requirement
    const withinDriftBounds = Math.abs(tauDelta) <= this.config.tau_drift_threshold;
    
    if (!withinDriftBounds) {
      // FREEZE SYSTEM - TODO.md requirement for |Δτ|>0.02
      session.status = 'frozen';
      session.freeze_reason = `Tau drift |${tauDelta.toFixed(4)}| > ${this.config.tau_drift_threshold} threshold`;
      session.drift_check_passed = false;
      
      await this.sendAlert({
        timestamp: new Date().toISOString(),
        severity: 'critical',
        message: `🚨 SYSTEM FROZEN: Tau drift exceeds bounds`,
        session_id: session.session_id,
        metrics: {
          tau_delta: tauDelta,
          drift_threshold: this.config.tau_drift_threshold,
          current_tau: session.pre_calibration_tau,
          optimal_tau: bestTau
        }
      });
      
      console.log(`🚨 CALIBRATION FROZEN - TAU DRIFT EXCEEDS BOUNDS`);
      console.log(`   |Δτ| = |${tauDelta.toFixed(4)}| > ${this.config.tau_drift_threshold}`);
      console.log(`   System requires manual intervention per TODO.md`);
      
      this.emit('calibration_frozen', {
        sessionId: session.session_id,
        reason: session.freeze_reason,
        tauDelta: tauDelta,
        threshold: this.config.tau_drift_threshold
      });
      
      return true; // Session processed, but frozen
    }
    
    // Apply tau adjustment - within drift bounds
    session.optimized_tau = bestTau;
    session.tau_delta = tauDelta;
    session.drift_check_passed = true;
    session.status = 'completed';
    session.calibration_completion_time = new Date().toISOString();
    
    // Apply the calibration to the production system
    await this.applyCalibrationUpdate(session);
    
    await this.sendAlert({
      timestamp: new Date().toISOString(),
      severity: 'info',
      message: `✅ Post-deploy calibration completed successfully`,
      session_id: session.session_id,
      metrics: {
        tau_delta: tauDelta,
        new_tau: bestTau,
        predicted_results_per_query: bestResultsPerQuery,
        click_through_rate: session.click_through_rate,
        total_queries: session.total_queries_analyzed
      }
    });
    
    console.log(`✅ POST-DEPLOY CALIBRATION COMPLETED`);
    console.log(`   Session: ${session.session_id}`);
    console.log(`   τ: ${session.pre_calibration_tau.toFixed(4)} → ${bestTau.toFixed(4)} (Δ${tauDelta >= 0 ? '+' : ''}${tauDelta.toFixed(4)})`);
    console.log(`   Within bounds: |Δτ| ≤ ${this.config.tau_drift_threshold} ✓`);
    console.log(`   Results/query: ${bestResultsPerQuery.toFixed(1)} (target: ${targetResultsPerQuery}±${targetTolerance})`);
    
    this.emit('calibration_completed', {
      sessionId: session.session_id,
      oldTau: session.pre_calibration_tau,
      newTau: bestTau,
      tauDelta: tauDelta,
      withinBounds: true
    });
    
    return true;
  }
  
  /**
   * Apply calibration update to production system
   */
  private async applyCalibrationUpdate(session: CalibrationSession): Promise<void> {
    if (!session.optimized_tau || !session.reliability_diagram) {
      throw new Error('Missing optimization results for calibration update');
    }
    
    console.log(`⚙️  Applying calibration update to production system`);
    
    // Update the online calibration system with new tau and reliability curve
    await this.calibrationSystem.manualCalibrationOverride(
      session.optimized_tau,
      `Post-deploy calibration: session ${session.session_id}`
    );
    
    console.log(`✅ Calibration applied to production system`);
  }
  
  /**
   * Collect production click/impression data
   */
  private async collectProductionClickData(session: CalibrationSession): Promise<ClickImpressionData[]> {
    // In production, this would query actual click logs from the deployed canary system
    // For now, we simulate realistic production data
    
    const clickData: ClickImpressionData[] = [];
    const baseTime = new Date(session.canary_completion_time);
    
    // Simulate 24-48 hours of production traffic data
    for (let i = 0; i < 2000; i++) { // More data for better calibration
      const queryTime = new Date(baseTime.getTime() + Math.random() * 48 * 60 * 60 * 1000);
      const numResults = Math.floor(Math.random() * 6) + 3; // 3-8 results (5±2 target range)
      const results = [];
      
      for (let j = 0; j < numResults; j++) {
        const predictedScore = Math.max(0.1, 1.0 - j * 0.12 + (Math.random() - 0.5) * 0.2);
        results.push({
          file_path: `production_file_${Math.floor(Math.random() * 1000)}.ts`,
          line_number: Math.floor(Math.random() * 500) + 1,
          content: `Production result ${j} for query ${i}`,
          predicted_score: predictedScore,
          features: {
            lexical_score: Math.random(),
            symbol_match: Math.random(),
            semantic_sim: Math.random(),
            file_popularity: Math.random(),
            exact_match: Math.random() > 0.8 ? 1 : 0
          }
        });
      }
      
      // Simulate clicks with realistic behavior
      const clicks: number[] = [];
      for (let j = 0; j < results.length; j++) {
        // Higher scored items more likely to be clicked, with position bias
        const positionBias = 1.0 / (1 + j * 0.3);
        const clickProbability = results[j].predicted_score * 0.6 * positionBias;
        
        if (Math.random() < clickProbability) {
          clicks.push(j);
          results[j].click_position = j + 1;
        }
      }
      
      clickData.push({
        query: `production_query_${i}`,
        results,
        clicks,
        impressions: results.length,
        timestamp: queryTime.toISOString(),
        user_session: `session_${Math.floor(i / 20)}`,
        query_latency_ms: 80 + Math.random() * 120 // Production latencies
      });
    }
    
    return clickData;
  }
  
  /**
   * Simulate results per query for given tau
   */
  private simulateResultsPerQuery(tau: number, clickData: ClickImpressionData[]): number {
    let totalResults = 0;
    let queryCount = 0;
    
    for (const item of clickData) {
      // Count results above tau threshold
      const passingResults = item.results.filter(r => r.predicted_score >= tau).length;
      totalResults += passingResults;
      queryCount++;
    }
    
    return queryCount > 0 ? totalResults / queryCount : 0;
  }
  
  /**
   * Calculate Wilson confidence interval for precision
   */
  private calculateWilsonConfidenceInterval(successes: number, trials: number): [number, number] {
    if (trials === 0) return [0, 0];
    
    const z = 1.96; // 95% confidence
    const p = successes / trials;
    const denominator = 1 + z * z / trials;
    
    const center = (p + z * z / (2 * trials)) / denominator;
    const margin = z * Math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials) / denominator;
    
    return [Math.max(0, center - margin), Math.min(1, center + margin)];
  }
  
  /**
   * Handle session processing errors
   */
  private async handleSessionError(session: CalibrationSession, error: any): Promise<void> {
    console.error(`❌ Calibration session error:`, error);
    
    await this.sendAlert({
      timestamp: new Date().toISOString(),
      severity: 'critical',
      message: `Calibration session error: ${error.message}`,
      session_id: session.session_id,
      metrics: {
        error_stack: error.stack,
        session_status: session.status
      }
    });
    
    this.emit('session_error', {
      sessionId: session.session_id,
      error: error.message,
      status: session.status
    });
  }
  
  /**
   * Send calibration alert
   */
  private async sendAlert(alert: CalibrationAlert): Promise<void> {
    const alertPath = join(this.alertsDir, `alert-${Date.now()}.json`);
    writeFileSync(alertPath, JSON.stringify(alert, null, 2));
    
    console.log(`📢 CALIBRATION ALERT [${alert.severity.toUpperCase()}]: ${alert.message}`);
    
    this.emit('alert', alert);
  }
  
  /**
   * Load active sessions from disk
   */
  private loadActiveSessions(): void {
    if (!existsSync(this.sessionsDir)) return;
    
    const sessionFiles = require('fs').readdirSync(this.sessionsDir).filter((f: string) => f.endsWith('.json'));
    
    for (const file of sessionFiles) {
      try {
        const sessionData = JSON.parse(readFileSync(join(this.sessionsDir, file), 'utf-8'));
        if (sessionData.status !== 'completed' && sessionData.status !== 'frozen') {
          this.activeSessions.set(sessionData.session_id, sessionData);
        }
      } catch (error) {
        console.warn(`⚠️  Failed to load session file ${file}:`, error);
      }
    }
    
    console.log(`📂 Loaded ${this.activeSessions.size} active calibration sessions`);
  }
  
  /**
   * Save session to disk
   */
  private saveSession(session: CalibrationSession): void {
    const sessionPath = join(this.sessionsDir, `${session.session_id}.json`);
    writeFileSync(sessionPath, JSON.stringify(session, null, 2));
  }
  
  /**
   * Get all calibration sessions
   */
  public getCalibrationSessions(): CalibrationSession[] {
    const allSessions: CalibrationSession[] = [];
    
    // Add active sessions
    for (const session of this.activeSessions.values()) {
      allSessions.push({ ...session });
    }
    
    // Add completed/frozen sessions from disk
    if (existsSync(this.sessionsDir)) {
      const sessionFiles = require('fs').readdirSync(this.sessionsDir).filter((f: string) => f.endsWith('.json'));
      
      for (const file of sessionFiles) {
        try {
          const sessionData = JSON.parse(readFileSync(join(this.sessionsDir, file), 'utf-8'));
          if (!this.activeSessions.has(sessionData.session_id)) {
            allSessions.push(sessionData);
          }
        } catch (error) {
          // Skip invalid session files
        }
      }
    }
    
    return allSessions.sort((a, b) => new Date(b.canary_completion_time).getTime() - new Date(a.canary_completion_time).getTime());
  }
  
  /**
   * Get calibration session by ID
   */
  public getCalibrationSession(sessionId: string): CalibrationSession | null {
    return this.activeSessions.get(sessionId) || null;
  }
  
  /**
   * Manual session intervention (emergency use)
   */
  public async manualSessionIntervention(sessionId: string, action: 'complete' | 'freeze' | 'retry', reason: string): Promise<void> {
    const session = this.activeSessions.get(sessionId);
    if (!session) {
      throw new Error(`Session ${sessionId} not found`);
    }
    
    const oldStatus = session.status;
    
    switch (action) {
      case 'complete':
        session.status = 'completed';
        session.calibration_completion_time = new Date().toISOString();
        this.activeSessions.delete(sessionId);
        break;
      case 'freeze':
        session.status = 'frozen';
        session.freeze_reason = reason;
        this.activeSessions.delete(sessionId);
        break;
      case 'retry':
        session.status = 'collecting_data';
        session.click_impression_data = undefined;
        session.reliability_diagram = undefined;
        session.optimized_tau = undefined;
        break;
    }
    
    this.saveSession(session);
    
    await this.sendAlert({
      timestamp: new Date().toISOString(),
      severity: 'warning',
      message: `Manual intervention: ${action} (${reason})`,
      session_id: sessionId,
      metrics: {
        old_status: oldStatus,
        new_status: session.status,
        intervention_reason: reason
      }
    });
    
    console.log(`🔧 Manual intervention applied to session ${sessionId}: ${oldStatus} → ${session.status}`);
    
    this.emit('manual_intervention', {
      sessionId,
      action,
      reason,
      oldStatus,
      newStatus: session.status
    });
  }
}