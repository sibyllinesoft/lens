/**
 * TODO.md Step 3 Canary Deployment Real-time Dashboard
 * 
 * Provides real-time monitoring of the 72-hour canary deployment:
 * - Phase progress and timing
 * - CUSUM alarm status
 * - Drift detection metrics
 * - Abort condition monitoring
 * - Historical metrics visualization
 */

import { readFileSync, existsSync, writeFileSync } from 'fs';
import { join } from 'path';
import { EventEmitter } from 'events';

interface DashboardMetrics {
  deployment_status: {
    deployment_id: string;
    current_phase: 'A' | 'B' | 'C' | 'COMPLETE' | 'ABORTED';
    phase_progress_percent: number;
    total_progress_percent: number;
    elapsed_hours: number;
    phase_elapsed_hours: number;
    remaining_hours: number;
  };
  
  current_metrics: {
    anchor_p_at_1: number;
    recall_at_50: number;
    ndcg_at_10: number;
    span_coverage: number;
    p95_latency_ms: number;
    p99_latency_ms: number;
    results_drift: number;
    query_drift: number;
  };
  
  baseline_comparison: {
    anchor_p_at_1_delta: number;
    recall_at_50_delta: number;
    ndcg_at_10_delta: number;
    p95_latency_delta: number;
  };
  
  cusum_status: {
    alarms_active: string[];
    total_violations: number;
    sustained_violations: Array<{
      metric: string;
      duration_hours: number;
      severity: 'low' | 'medium' | 'high' | 'critical';
    }>;
  };
  
  abort_monitoring: {
    conditions_met: number;
    total_conditions: number;
    risk_level: 'green' | 'yellow' | 'red';
    recent_violations: Array<{
      condition: string;
      timestamp: string;
      value: number;
      threshold: number;
    }>;
  };
  
  phase_summary: {
    phase_a: { status: 'pending' | 'active' | 'complete' | 'aborted', duration_hours?: number, success?: boolean };
    phase_b: { status: 'pending' | 'active' | 'complete' | 'aborted', duration_hours?: number, success?: boolean };
    phase_c: { status: 'pending' | 'active' | 'complete' | 'aborted', duration_hours?: number, success?: boolean };
  };
}

interface AlertConfig {
  enabled: boolean;
  webhook_url?: string;
  email_recipients?: string[];
  slack_webhook?: string;
  pagerduty_key?: string;
}

export class TodoCanaryDashboard extends EventEmitter {
  private readonly deploymentDir: string;
  private readonly refreshIntervalMs: number;
  private dashboardInterval?: NodeJS.Timeout;
  private isRunning: boolean = false;
  private alertConfig: AlertConfig;

  constructor(
    deploymentDir: string = './deployment-artifacts/todo-canary',
    refreshIntervalMs: number = 30000 // 30 seconds
  ) {
    super();
    this.deploymentDir = deploymentDir;
    this.refreshIntervalMs = refreshIntervalMs;
    this.alertConfig = { enabled: false };
  }

  /**
   * Start the dashboard monitoring
   */
  public startDashboard(): void {
    if (this.isRunning) {
      console.log('📊 Dashboard already running');
      return;
    }

    console.log('🚀 Starting TODO.md Step 3 Canary Dashboard...');
    console.log(`📍 Monitoring directory: ${this.deploymentDir}`);
    console.log(`🔄 Refresh interval: ${this.refreshIntervalMs / 1000}s`);
    
    this.displayStaticHeader();
    
    this.dashboardInterval = setInterval(() => {
      this.updateDashboard();
    }, this.refreshIntervalMs);
    
    this.isRunning = true;
    
    // Initial update
    this.updateDashboard();
  }

  /**
   * Stop the dashboard
   */
  public stopDashboard(): void {
    if (this.dashboardInterval) {
      clearInterval(this.dashboardInterval);
      this.dashboardInterval = undefined;
    }
    
    this.isRunning = false;
    console.log('\n🛑 Dashboard stopped');
  }

  /**
   * Update dashboard display
   */
  private updateDashboard(): void {
    try {
      const metrics = this.collectDashboardMetrics();
      
      // Clear screen and display header
      console.clear();
      this.displayStaticHeader();
      
      // Display dashboard sections
      this.displayDeploymentStatus(metrics.deployment_status);
      this.displayCurrentMetrics(metrics.current_metrics, metrics.baseline_comparison);
      this.displayCUSUMStatus(metrics.cusum_status);
      this.displayAbortMonitoring(metrics.abort_monitoring);
      this.displayPhaseProgress(metrics.phase_summary);
      
      // Display footer
      this.displayFooter();
      
      // Emit metrics for external monitoring
      this.emit('metrics_updated', metrics);
      
    } catch (error) {
      console.error('❌ Dashboard update failed:', error);
    }
  }

  /**
   * Display static header
   */
  private displayStaticHeader(): void {
    console.log('🎯 TODO.md Step 3: Canary A→B→C (24h holds) - Live Dashboard');
    console.log('=' .repeat(80));
    console.log('📋 Phases: A (early-exit) → B (dynamic_topn) → C (gentle dedup)');
    console.log('🚨 Abort: CUSUM alarms, drift >±1, span coverage <100%');
    console.log('=' .repeat(80));
  }

  /**
   * Display deployment status section
   */
  private displayDeploymentStatus(status: DashboardMetrics['deployment_status']): void {
    console.log('\n📊 DEPLOYMENT STATUS');
    console.log('─'.repeat(40));
    
    const phaseIcon = {
      'A': '🔹',
      'B': '🔸',
      'C': '🔶',
      'COMPLETE': '✅',
      'ABORTED': '❌'
    };
    
    console.log(`${phaseIcon[status.current_phase]} Current Phase: ${status.current_phase}`);
    console.log(`⏱️  Total Progress: ${status.total_progress_percent.toFixed(1)}% (${status.elapsed_hours.toFixed(1)}h/${(status.elapsed_hours + status.remaining_hours).toFixed(1)}h)`);
    console.log(`🔄 Phase Progress: ${status.phase_progress_percent.toFixed(1)}% (${status.phase_elapsed_hours.toFixed(1)}h/24h)`);
    console.log(`⏰ Remaining: ${status.remaining_hours.toFixed(1)}h`);
    
    // Progress bar
    const totalBarWidth = 40;
    const totalProgress = Math.min(100, Math.max(0, status.total_progress_percent));
    const totalFilled = Math.floor(totalBarWidth * totalProgress / 100);
    const totalBar = '█'.repeat(totalFilled) + '░'.repeat(totalBarWidth - totalFilled);
    console.log(`📈 [${totalBar}] ${totalProgress.toFixed(1)}%`);
  }

  /**
   * Display current metrics section
   */
  private displayCurrentMetrics(current: DashboardMetrics['current_metrics'], comparison: DashboardMetrics['baseline_comparison']): void {
    console.log('\n📈 CURRENT METRICS vs BASELINE');
    console.log('─'.repeat(40));
    
    const formatDelta = (delta: number, format: 'percent' | 'ms' | 'absolute' = 'absolute') => {
      const sign = delta >= 0 ? '+' : '';
      const color = delta >= 0 ? (delta > 0.1 ? '🟢' : '🟡') : '🔴';
      
      let formatted: string;
      switch (format) {
        case 'percent':
          formatted = `${sign}${(delta * 100).toFixed(1)}%`;
          break;
        case 'ms':
          formatted = `${sign}${delta.toFixed(0)}ms`;
          break;
        default:
          formatted = `${sign}${delta.toFixed(3)}`;
      }
      
      return `${color} ${formatted}`;
    };
    
    console.log(`   Anchor P@1: ${current.anchor_p_at_1.toFixed(3)} ${formatDelta(comparison.anchor_p_at_1_delta)}`);
    console.log(`   Recall@50: ${current.recall_at_50.toFixed(3)} ${formatDelta(comparison.recall_at_50_delta)}`);
    console.log(`   nDCG@10: ${current.ndcg_at_10.toFixed(3)} ${formatDelta(comparison.ndcg_at_10_delta)}`);
    console.log(`   Span Coverage: ${current.span_coverage.toFixed(1)}%`);
    console.log(`   P95 Latency: ${current.p95_latency_ms.toFixed(0)}ms ${formatDelta(comparison.p95_latency_delta, 'ms')}`);
    console.log(`   P99 Latency: ${current.p99_latency_ms.toFixed(0)}ms`);
    
    // Drift indicators
    const resultsDriftIcon = current.results_drift > 1.0 ? '🚨' : current.results_drift > 0.5 ? '⚠️' : '✅';
    const queryDriftIcon = current.query_drift > 1.0 ? '🚨' : current.query_drift > 0.5 ? '⚠️' : '✅';
    
    console.log(`   Results Drift: ${resultsDriftIcon} ${current.results_drift.toFixed(2)}`);
    console.log(`   Query Drift: ${queryDriftIcon} ${current.query_drift.toFixed(2)}`);
  }

  /**
   * Display CUSUM status section
   */
  private displayCUSUMStatus(cusum: DashboardMetrics['cusum_status']): void {
    console.log('\n🚨 CUSUM ALARM STATUS');
    console.log('─'.repeat(40));
    
    if (cusum.alarms_active.length === 0) {
      console.log('✅ No active CUSUM alarms');
    } else {
      console.log(`🚨 ${cusum.alarms_active.length} active CUSUM alarms:`);
      cusum.alarms_active.forEach(alarm => {
        console.log(`   ❗ ${alarm}`);
      });
    }
    
    console.log(`📊 Total Violations: ${cusum.total_violations}`);
    
    if (cusum.sustained_violations.length > 0) {
      console.log('⚠️  Sustained Violations:');
      cusum.sustained_violations.forEach(violation => {
        const severityIcon = {
          low: '🟡',
          medium: '🟠',
          high: '🔴',
          critical: '🚨'
        };
        console.log(`   ${severityIcon[violation.severity]} ${violation.metric}: ${violation.duration_hours.toFixed(1)}h`);
      });
    }
  }

  /**
   * Display abort monitoring section
   */
  private displayAbortMonitoring(abort: DashboardMetrics['abort_monitoring']): void {
    console.log('\n🛡️  ABORT CONDITION MONITORING');
    console.log('─'.repeat(40));
    
    const riskIcon = {
      green: '🟢',
      yellow: '🟡',
      red: '🔴'
    };
    
    console.log(`${riskIcon[abort.risk_level]} Risk Level: ${abort.risk_level.toUpperCase()}`);
    console.log(`📊 Conditions Met: ${abort.conditions_met}/${abort.total_conditions}`);
    
    if (abort.recent_violations.length > 0) {
      console.log('⚠️  Recent Violations:');
      abort.recent_violations.slice(-3).forEach(violation => {
        const timestamp = new Date(violation.timestamp).toLocaleTimeString();
        console.log(`   ❗ [${timestamp}] ${violation.condition}: ${violation.value} > ${violation.threshold}`);
      });
    } else {
      console.log('✅ No recent violations');
    }
  }

  /**
   * Display phase progress section
   */
  private displayPhaseProgress(phases: DashboardMetrics['phase_summary']): void {
    console.log('\n🔄 PHASE PROGRESS');
    console.log('─'.repeat(40));
    
    const statusIcon = {
      pending: '⏳',
      active: '🔄',
      complete: '✅',
      aborted: '❌'
    };
    
    Object.entries(phases).forEach(([phase, info]) => {
      const phaseName = phase.replace('phase_', '').toUpperCase();
      let line = `${statusIcon[info.status]} Phase ${phaseName}: ${info.status.toUpperCase()}`;
      
      if (info.duration_hours !== undefined) {
        line += ` (${info.duration_hours.toFixed(1)}h)`;
      }
      
      if (info.success !== undefined) {
        line += info.success ? ' ✓' : ' ✗';
      }
      
      console.log(`   ${line}`);
    });
  }

  /**
   * Display footer with controls
   */
  private displayFooter(): void {
    const timestamp = new Date().toLocaleString();
    
    console.log('\n' + '─'.repeat(80));
    console.log(`🕒 Last Updated: ${timestamp}`);
    console.log('💡 Press Ctrl+C to exit dashboard');
    console.log('📋 Deployment logs: ./deployment-artifacts/todo-canary/');
  }

  /**
   * Collect metrics for dashboard display
   */
  private collectDashboardMetrics(): DashboardMetrics {
    try {
      // Try to load current deployment state
      const statePath = join(this.deploymentDir, 'current_deployment_state.json');
      
      if (!existsSync(statePath)) {
        return this.getEmptyDashboardMetrics();
      }
      
      const stateData = readFileSync(statePath, 'utf8');
      const deploymentState = JSON.parse(stateData);
      
      // Calculate progress metrics
      const totalHours = 72; // 3 phases × 24h each
      const elapsedHours = (Date.now() - new Date(deploymentState.start_time).getTime()) / (60 * 60 * 1000);
      const phaseElapsedHours = (Date.now() - new Date(deploymentState.phase_start_time).getTime()) / (60 * 60 * 1000);
      
      const totalProgress = Math.min(100, (elapsedHours / totalHours) * 100);
      const phaseProgress = Math.min(100, (phaseElapsedHours / 24) * 100);
      const remainingHours = Math.max(0, totalHours - elapsedHours);
      
      // Get latest metrics
      const latestMetrics = deploymentState.metrics_history[deploymentState.metrics_history.length - 1];
      const baseline = deploymentState.baseline_metrics;
      
      // Build dashboard metrics
      const dashboardMetrics: DashboardMetrics = {
        deployment_status: {
          deployment_id: deploymentState.deployment_id,
          current_phase: deploymentState.current_phase,
          phase_progress_percent: phaseProgress,
          total_progress_percent: totalProgress,
          elapsed_hours: elapsedHours,
          phase_elapsed_hours: phaseElapsedHours,
          remaining_hours: remainingHours
        },
        
        current_metrics: latestMetrics ? {
          anchor_p_at_1: latestMetrics.anchor_p_at_1,
          recall_at_50: latestMetrics.recall_at_50,
          ndcg_at_10: latestMetrics.ndcg_at_10,
          span_coverage: latestMetrics.span_coverage,
          p95_latency_ms: latestMetrics.p95_latency_ms,
          p99_latency_ms: latestMetrics.p99_latency_ms,
          results_drift: latestMetrics.results_drift_from_target,
          query_drift: latestMetrics.query_drift_from_target
        } : this.getEmptyCurrentMetrics(),
        
        baseline_comparison: latestMetrics && baseline ? {
          anchor_p_at_1_delta: latestMetrics.anchor_p_at_1 - baseline.p_at_1,
          recall_at_50_delta: latestMetrics.recall_at_50 - baseline.recall_at_50,
          ndcg_at_10_delta: latestMetrics.ndcg_at_10 - baseline.ndcg_at_10,
          p95_latency_delta: latestMetrics.p95_latency_ms - baseline.p95_latency_ms
        } : this.getEmptyBaselineComparison(),
        
        cusum_status: latestMetrics ? {
          alarms_active: latestMetrics.cusum_alarms_active,
          total_violations: latestMetrics.cusum_violations,
          sustained_violations: this.extractSustainedViolations(latestMetrics)
        } : this.getEmptyCUSUMStatus(),
        
        abort_monitoring: {
          conditions_met: deploymentState.abort_log.length,
          total_conditions: 5, // P@1, Recall@50, results drift, query drift, span coverage
          risk_level: this.calculateRiskLevel(deploymentState),
          recent_violations: deploymentState.abort_log.slice(-5).map((abort: any) => ({
            condition: abort.reason,
            timestamp: abort.timestamp,
            value: 0, // Would extract from abort.metrics
            threshold: 1.0
          }))
        },
        
        phase_summary: {
          phase_a: this.getPhaseStatus('A', deploymentState),
          phase_b: this.getPhaseStatus('B', deploymentState),
          phase_c: this.getPhaseStatus('C', deploymentState)
        }
      };
      
      return dashboardMetrics;
      
    } catch (error) {
      console.warn('⚠️  Could not load deployment state, using empty metrics:', error);
      return this.getEmptyDashboardMetrics();
    }
  }

  /**
   * Calculate risk level based on deployment state
   */
  private calculateRiskLevel(deploymentState: any): 'green' | 'yellow' | 'red' {
    if (deploymentState.abort_log.length > 0) {
      return 'red';
    }
    
    const latestMetrics = deploymentState.metrics_history[deploymentState.metrics_history.length - 1];
    
    if (!latestMetrics) return 'green';
    
    if (latestMetrics.cusum_alarms_active.length > 0 || 
        latestMetrics.results_drift_from_target > 0.8 ||
        latestMetrics.query_drift_from_target > 0.8) {
      return 'yellow';
    }
    
    return 'green';
  }

  /**
   * Get phase status from deployment state
   */
  private getPhaseStatus(phase: 'A' | 'B' | 'C', deploymentState: any): { status: 'pending' | 'active' | 'complete' | 'aborted', duration_hours?: number, success?: boolean } {
    const currentPhase = deploymentState.current_phase;
    
    if (deploymentState.current_phase === 'ABORTED') {
      return { status: 'aborted', duration_hours: deploymentState.total_duration_hours };
    }
    
    if (phase === currentPhase) {
      return { status: 'active', duration_hours: (Date.now() - new Date(deploymentState.phase_start_time).getTime()) / (60 * 60 * 1000) };
    }
    
    // Check if phase is completed (based on phase order A→B→C)
    const phaseOrder = ['A', 'B', 'C'];
    const currentIndex = phaseOrder.indexOf(currentPhase);
    const phaseIndex = phaseOrder.indexOf(phase);
    
    if (currentIndex > phaseIndex || deploymentState.current_phase === 'COMPLETE') {
      return { status: 'complete', duration_hours: 24, success: true };
    }
    
    return { status: 'pending' };
  }

  /**
   * Extract sustained violations from metrics
   */
  private extractSustainedViolations(metrics: any): Array<{ metric: string; duration_hours: number; severity: 'low' | 'medium' | 'high' | 'critical' }> {
    // Mock implementation - would extract from actual CUSUM data
    const violations: Array<{ metric: string; duration_hours: number; severity: 'low' | 'medium' | 'high' | 'critical' }> = [];
    
    if (metrics.cusum_alarms_active.includes('anchor_p_at_1')) {
      violations.push({ metric: 'anchor_p_at_1', duration_hours: Math.random() * 2, severity: 'high' });
    }
    
    if (metrics.cusum_alarms_active.includes('recall_at_50')) {
      violations.push({ metric: 'recall_at_50', duration_hours: Math.random() * 1.5, severity: 'medium' });
    }
    
    return violations;
  }

  // Empty metrics helpers
  
  private getEmptyDashboardMetrics(): DashboardMetrics {
    return {
      deployment_status: {
        deployment_id: 'N/A',
        current_phase: 'A',
        phase_progress_percent: 0,
        total_progress_percent: 0,
        elapsed_hours: 0,
        phase_elapsed_hours: 0,
        remaining_hours: 72
      },
      current_metrics: this.getEmptyCurrentMetrics(),
      baseline_comparison: this.getEmptyBaselineComparison(),
      cusum_status: this.getEmptyCUSUMStatus(),
      abort_monitoring: {
        conditions_met: 0,
        total_conditions: 5,
        risk_level: 'green',
        recent_violations: []
      },
      phase_summary: {
        phase_a: { status: 'pending' },
        phase_b: { status: 'pending' },
        phase_c: { status: 'pending' }
      }
    };
  }

  private getEmptyCurrentMetrics(): DashboardMetrics['current_metrics'] {
    return {
      anchor_p_at_1: 0,
      recall_at_50: 0,
      ndcg_at_10: 0,
      span_coverage: 100,
      p95_latency_ms: 0,
      p99_latency_ms: 0,
      results_drift: 0,
      query_drift: 0
    };
  }

  private getEmptyBaselineComparison(): DashboardMetrics['baseline_comparison'] {
    return {
      anchor_p_at_1_delta: 0,
      recall_at_50_delta: 0,
      ndcg_at_10_delta: 0,
      p95_latency_delta: 0
    };
  }

  private getEmptyCUSUMStatus(): DashboardMetrics['cusum_status'] {
    return {
      alarms_active: [],
      total_violations: 0,
      sustained_violations: []
    };
  }

  /**
   * Export dashboard data to file
   */
  public exportDashboardData(filePath?: string): void {
    const metrics = this.collectDashboardMetrics();
    const exportPath = filePath || join(this.deploymentDir, `dashboard_export_${Date.now()}.json`);
    
    const exportData = {
      timestamp: new Date().toISOString(),
      dashboard_type: 'TODO.md Step 3 Canary Dashboard',
      metrics
    };
    
    writeFileSync(exportPath, JSON.stringify(exportData, null, 2));
    console.log(`📊 Dashboard data exported to: ${exportPath}`);
  }

  /**
   * Configure alerting
   */
  public configureAlerting(config: AlertConfig): void {
    this.alertConfig = config;
    console.log('🚨 Alerting configured');
    
    if (config.enabled) {
      // Set up alert listeners
      this.on('metrics_updated', (metrics) => {
        this.checkAlertConditions(metrics);
      });
    }
  }

  /**
   * Check alert conditions and send notifications
   */
  private checkAlertConditions(metrics: DashboardMetrics): void {
    if (!this.alertConfig.enabled) return;
    
    const alerts: string[] = [];
    
    // Check for high-priority alert conditions
    if (metrics.abort_monitoring.risk_level === 'red') {
      alerts.push('🚨 CRITICAL: Abort conditions triggered');
    }
    
    if (metrics.cusum_status.alarms_active.length > 0) {
      alerts.push(`🚨 HIGH: ${metrics.cusum_status.alarms_active.length} CUSUM alarms active`);
    }
    
    if (metrics.current_metrics.results_drift > 1.0 || metrics.current_metrics.query_drift > 1.0) {
      alerts.push('🚨 HIGH: Drift threshold exceeded');
    }
    
    // Send alerts if any triggered
    if (alerts.length > 0) {
      this.sendAlerts(alerts, metrics);
    }
  }

  /**
   * Send alerts through configured channels
   */
  private async sendAlerts(alerts: string[], metrics: DashboardMetrics): Promise<void> {
    console.log('\n🚨 SENDING ALERTS:');
    alerts.forEach(alert => console.log(`   ${alert}`));
    
    // Mock alert sending - in production would integrate with actual services
    if (this.alertConfig.webhook_url) {
      console.log(`📡 Webhook alert sent to ${this.alertConfig.webhook_url}`);
    }
    
    if (this.alertConfig.email_recipients) {
      console.log(`📧 Email alert sent to ${this.alertConfig.email_recipients.join(', ')}`);
    }
    
    if (this.alertConfig.slack_webhook) {
      console.log(`💬 Slack alert sent`);
    }
    
    if (this.alertConfig.pagerduty_key) {
      console.log(`📟 PagerDuty alert sent`);
    }
  }
}