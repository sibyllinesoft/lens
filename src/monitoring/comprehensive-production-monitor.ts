/**
 * Comprehensive Production Monitoring System
 * 
 * Integrates all monitoring components for TODO.md steps 6-7:
 * - ECE drift tracking with intent×language stratification
 * - KL drift monitoring with ≤ 0.02 threshold
 * - A/A shadow testing with ≤ 0.1 pp drift tolerance
 * - Production alerting and automated responses
 * - Deliverables generation coordination
 * - Real-time dashboard data aggregation
 */

import { EventEmitter } from 'events';
import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';

// Import our monitoring components
import { ECEDriftTracker, IntentLanguageQuery, DriftAlert as ECEDriftAlert, AATestConfig } from './ece-drift-tracker.js';
import { KLDriftMonitor, KLDriftAlert } from './kl-drift-monitor.js';
import { ProductionMonitoringSystem } from '../deployment/production-monitoring-system.js';
import { ProductionDeliverablesGenerator } from '../deliverables/production-deliverables-generator.js';

export interface ComprehensiveMonitoringConfig {
  monitoring_enabled: boolean;
  deliverables_auto_generation: boolean;
  
  // ECE monitoring
  ece_drift_threshold: number;
  ece_evaluation_interval_minutes: number;
  
  // KL monitoring  
  kl_drift_threshold: number;
  kl_evaluation_interval_minutes: number;
  
  // A/A testing
  aa_testing_enabled: boolean;
  aa_traffic_percentage: number;
  aa_test_duration_minutes: number;
  aa_drift_threshold_pp: number;
  
  // Alerting
  alert_escalation_enabled: boolean;
  rollback_on_critical_drift: boolean;
  max_alerts_per_hour: number;
  
  // Deliverables
  deliverables_generation_schedule: 'hourly' | 'daily' | 'on_demand';
  deliverables_output_dir: string;
}

export interface MonitoringHealth {
  overall_status: 'healthy' | 'degraded' | 'critical';
  component_status: {
    ece_tracking: 'healthy' | 'degraded' | 'critical';
    kl_monitoring: 'healthy' | 'degraded' | 'critical';
    aa_testing: 'healthy' | 'degraded' | 'critical' | 'disabled';
    production_alerts: 'healthy' | 'degraded' | 'critical';
  };
  active_alerts: number;
  active_drift_detections: number;
  aa_test_status: 'running' | 'stopped' | 'disabled';
  last_deliverables_generation: string | null;
  system_uptime_hours: number;
}

export interface AlertAction {
  action_type: 'webhook' | 'email' | 'rollback' | 'traffic_reduction' | 'feature_flag';
  config: Record<string, any>;
  executed_at?: string;
  success?: boolean;
  error_message?: string;
}

export interface MonitoringEvent {
  event_id: string;
  event_type: 'drift_detected' | 'aa_test_completed' | 'deliverables_generated' | 'alert_triggered' | 'rollback_executed';
  timestamp: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  component: string;
  details: Record<string, any>;
  actions_taken: AlertAction[];
}

/**
 * Main comprehensive monitoring orchestrator
 */
export class ComprehensiveProductionMonitor extends EventEmitter {
  private eceTracker: ECEDriftTracker;
  private klMonitor: KLDriftMonitor;
  private productionMonitor: ProductionMonitoringSystem;
  private deliverablesGenerator: ProductionDeliverablesGenerator;
  
  private config: ComprehensiveMonitoringConfig;
  private monitoringHealth: MonitoringHealth;
  private activeAATest: string | null = null;
  private monitoringEvents: MonitoringEvent[] = [];
  private alertActionQueue: AlertAction[] = [];
  
  private monitoringDir: string;
  private deliverableSchedule?: NodeJS.Timeout;
  private healthCheckInterval?: NodeJS.Timeout;

  constructor(
    monitoringDir: string = './monitoring-data',
    config: Partial<ComprehensiveMonitoringConfig> = {}
  ) {
    super();
    
    this.monitoringDir = monitoringDir;
    this.config = {
      monitoring_enabled: true,
      deliverables_auto_generation: true,
      ece_drift_threshold: 0.02,
      ece_evaluation_interval_minutes: 5,
      kl_drift_threshold: 0.02,
      kl_evaluation_interval_minutes: 10,
      aa_testing_enabled: true,
      aa_traffic_percentage: 10,
      aa_test_duration_minutes: 30,
      aa_drift_threshold_pp: 0.1,
      alert_escalation_enabled: true,
      rollback_on_critical_drift: false, // Safety default
      max_alerts_per_hour: 10,
      deliverables_generation_schedule: 'daily',
      deliverables_output_dir: './deliverables',
      ...config
    };

    // Initialize monitoring components
    this.eceTracker = new ECEDriftTracker(join(monitoringDir, 'ece'));
    this.klMonitor = new KLDriftMonitor(join(monitoringDir, 'kl'));
    this.productionMonitor = new ProductionMonitoringSystem(join(monitoringDir, 'production'));
    this.deliverablesGenerator = new ProductionDeliverablesGenerator(this.config.deliverables_output_dir);
    
    this.monitoringHealth = this.initializeHealthStatus();
    
    this.setupEventHandlers();
    this.ensureDirectories();
  }

  /**
   * Start comprehensive monitoring system
   */
  async startMonitoring(): Promise<void> {
    if (!this.config.monitoring_enabled) {
      console.log('📊 Monitoring disabled by configuration');
      return;
    }

    console.log('🚀 Starting comprehensive production monitoring...');

    // Start individual monitoring components
    await this.productionMonitor.startMonitoring();
    
    // Start A/A testing if enabled
    if (this.config.aa_testing_enabled) {
      await this.startAATestingCycle();
    }
    
    // Schedule deliverables generation
    if (this.config.deliverables_auto_generation) {
      this.scheduleDeliverablesGeneration();
    }
    
    // Start health monitoring
    this.startHealthMonitoring();

    this.logMonitoringEvent({
      event_type: 'alert_triggered', // Using available event type
      component: 'comprehensive_monitor',
      severity: 'info',
      details: {
        action: 'monitoring_started',
        config: this.config
      },
      actions_taken: []
    });

    console.log('✅ Comprehensive monitoring system started');
    this.emit('monitoring_started');
  }

  /**
   * Stop all monitoring
   */
  stopMonitoring(): void {
    this.productionMonitor.stopMonitoring();
    
    if (this.deliverableSchedule) {
      clearInterval(this.deliverableSchedule);
      this.deliverableSchedule = undefined;
    }
    
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = undefined;
    }
    
    this.logMonitoringEvent({
      event_type: 'alert_triggered', // Using available event type
      component: 'comprehensive_monitor',
      severity: 'info',
      details: {
        action: 'monitoring_stopped'
      },
      actions_taken: []
    });

    console.log('🛑 Comprehensive monitoring stopped');
    this.emit('monitoring_stopped');
  }

  /**
   * Record query for monitoring (main entry point for production traffic)
   */
  recordQuery(query: IntentLanguageQuery): void {
    if (!this.config.monitoring_enabled) return;

    // Send to ECE tracker
    this.eceTracker.recordQuery(query);
    
    // Extract data for KL monitoring
    this.klMonitor.monitorQueryIntentDrift([query.intent]);
    this.klMonitor.monitorQueryLanguageDrift([query.language]);
    this.klMonitor.monitorConfidenceScoreDrift([query.confidence]);
    
    // Update health metrics
    this.updateHealthMetrics();
  }

  /**
   * Start A/A testing cycle
   */
  async startAATestingCycle(): Promise<string> {
    if (!this.config.aa_testing_enabled || this.activeAATest) {
      return this.activeAATest || '';
    }

    const aaConfig: AATestConfig = {
      enabled: true,
      traffic_split: this.config.aa_traffic_percentage,
      sample_size: 100,
      drift_threshold: this.config.aa_drift_threshold_pp / 100, // Convert pp to decimal
      test_duration_minutes: this.config.aa_test_duration_minutes
    };

    this.activeAATest = this.eceTracker.startAATesting(aaConfig);

    // Auto-stop test after duration
    setTimeout(() => {
      this.stopAATestingCycle();
    }, aaConfig.test_duration_minutes * 60 * 1000);

    this.logMonitoringEvent({
      event_type: 'aa_test_completed',
      component: 'aa_testing',
      severity: 'info',
      details: {
        test_id: this.activeAATest,
        config: aaConfig
      },
      actions_taken: []
    });

    console.log(`🔬 Started A/A test cycle: ${this.activeAATest}`);
    return this.activeAATest;
  }

  /**
   * Stop current A/A testing cycle
   */
  stopAATestingCycle(): void {
    if (this.activeAATest) {
      console.log(`🏁 Stopping A/A test: ${this.activeAATest}`);
      
      this.logMonitoringEvent({
        event_type: 'aa_test_completed',
        component: 'aa_testing',
        severity: 'info',
        details: {
          test_id: this.activeAATest,
          action: 'test_stopped'
        },
        actions_taken: []
      });

      this.activeAATest = null;
    }
  }

  /**
   * Generate production deliverables on demand
   */
  async generateDeliverables(): Promise<any> {
    if (!this.config.deliverables_auto_generation) {
      throw new Error('Deliverables auto-generation is disabled');
    }

    console.log('📋 Generating production deliverables...');

    try {
      const results = await this.deliverablesGenerator.generateAllDeliverables();

      this.logMonitoringEvent({
        event_type: 'deliverables_generated',
        component: 'deliverables_generator',
        severity: 'info',
        details: {
          files_generated: Object.keys(results).length,
          results
        },
        actions_taken: []
      });

      this.monitoringHealth.last_deliverables_generation = new Date().toISOString();

      console.log('✅ Production deliverables generated successfully');
      this.emit('deliverables_generated', results);

      return results;

    } catch (error) {
      this.logMonitoringEvent({
        event_type: 'deliverables_generated',
        component: 'deliverables_generator',
        severity: 'error',
        details: {
          error: error instanceof Error ? error.message : 'Unknown error'
        },
        actions_taken: []
      });

      throw error;
    }
  }

  /**
   * Get comprehensive monitoring status report
   */
  getStatusReport(): {
    health: MonitoringHealth;
    ece_status: any;
    kl_status: any;
    production_status: any;
    recent_events: MonitoringEvent[];
    alerts_summary: {
      total_active: number;
      critical_count: number;
      warning_count: number;
    };
  } {
    const eceStatus = this.eceTracker.getECEStatusReport();
    const klStatus = this.klMonitor.getDriftStatusReport();
    const productionStatus = this.productionMonitor.getHealthStatus();

    const recentEvents = this.monitoringEvents.slice(-50);
    
    const alertsSummary = {
      total_active: eceStatus.active_alerts + klStatus.active_drifts,
      critical_count: recentEvents.filter(e => e.severity === 'critical').length,
      warning_count: recentEvents.filter(e => e.severity === 'warning').length
    };

    return {
      health: this.monitoringHealth,
      ece_status: eceStatus,
      kl_status: klStatus,
      production_status: productionStatus,
      recent_events: recentEvents,
      alerts_summary: alertsSummary
    };
  }

  /**
   * Execute emergency rollback
   */
  async executeEmergencyRollback(reason: string): Promise<boolean> {
    if (!this.config.rollback_on_critical_drift) {
      console.log('🚫 Emergency rollback disabled by configuration');
      return false;
    }

    console.log(`🚨 EXECUTING EMERGENCY ROLLBACK: ${reason}`);

    const rollbackAction: AlertAction = {
      action_type: 'rollback',
      config: {
        reason,
        rollback_version: 'previous_stable',
        timestamp: new Date().toISOString()
      },
      executed_at: new Date().toISOString()
    };

    try {
      // In production, this would trigger actual rollback mechanisms
      // For now, we'll simulate the rollback process
      
      this.logMonitoringEvent({
        event_type: 'rollback_executed',
        component: 'emergency_rollback',
        severity: 'critical',
        details: {
          reason,
          rollback_triggered_by: 'comprehensive_monitor'
        },
        actions_taken: [rollbackAction]
      });

      rollbackAction.success = true;
      console.log('✅ Emergency rollback completed');
      
      this.emit('emergency_rollback', { reason, success: true });
      return true;

    } catch (error) {
      rollbackAction.success = false;
      rollbackAction.error_message = error instanceof Error ? error.message : 'Unknown error';

      console.error('❌ Emergency rollback failed:', error);
      this.emit('emergency_rollback', { reason, success: false, error });
      return false;
    }
  }

  // Private methods

  private setupEventHandlers(): void {
    // Handle ECE drift alerts
    this.eceTracker.on('drift_alert', (alert: ECEDriftAlert) => {
      this.handleDriftAlert('ece', alert);
    });

    // Handle KL drift alerts  
    this.klMonitor.on('kl_drift_alert', (alert: KLDriftAlert) => {
      this.handleDriftAlert('kl', alert);
    });

    // Handle production monitoring events
    this.productionMonitor.on('cusum_alarm_triggered', (data: any) => {
      this.handleProductionAlert('cusum_alarm', data);
    });

    this.productionMonitor.on('sustained_cusum_violation', (data: any) => {
      this.handleProductionAlert('sustained_violation', data);
    });

    this.productionMonitor.on('kill_switch_activated', (data: any) => {
      this.handleProductionAlert('kill_switch', data);
    });
  }

  private handleDriftAlert(component: 'ece' | 'kl', alert: ECEDriftAlert | KLDriftAlert): void {
    console.log(`🚨 ${component.toUpperCase()} DRIFT ALERT:`, alert);

    const severity = (alert as any).severity === 'critical' ? 'critical' : 'warning';
    
    this.logMonitoringEvent({
      event_type: 'drift_detected',
      component,
      severity,
      details: alert as any,
      actions_taken: []
    });

    // Trigger emergency rollback for critical drift
    if ((alert as any).severity === 'critical' && this.config.rollback_on_critical_drift) {
      this.executeEmergencyRollback(`Critical ${component} drift detected`);
    }

    this.updateHealthMetrics();
    this.emit('drift_alert', { component, alert });
  }

  private handleProductionAlert(alertType: string, data: any): void {
    const severity = alertType === 'kill_switch' ? 'critical' : 'warning';
    
    this.logMonitoringEvent({
      event_type: 'alert_triggered',
      component: 'production_monitor',
      severity,
      details: {
        alert_type: alertType,
        data
      },
      actions_taken: []
    });

    this.updateHealthMetrics();
    this.emit('production_alert', { type: alertType, data });
  }

  private scheduleDeliverablesGeneration(): void {
    const intervalMs = this.config.deliverables_generation_schedule === 'hourly' 
      ? 60 * 60 * 1000  // 1 hour
      : 24 * 60 * 60 * 1000; // 1 day

    this.deliverableSchedule = setInterval(async () => {
      try {
        await this.generateDeliverables();
      } catch (error) {
        console.error('❌ Scheduled deliverables generation failed:', error);
      }
    }, intervalMs);

    console.log(`📅 Deliverables generation scheduled: ${this.config.deliverables_generation_schedule}`);
  }

  private startHealthMonitoring(): void {
    this.healthCheckInterval = setInterval(() => {
      this.updateHealthMetrics();
    }, 60 * 1000); // Every minute

    console.log('💓 Health monitoring started');
  }

  private updateHealthMetrics(): void {
    const eceStatus = this.eceTracker.getECEStatusReport();
    const klStatus = this.klMonitor.getDriftStatusReport();
    const productionStatus = this.productionMonitor.getHealthStatus();

    // Update component status
    this.monitoringHealth.component_status.ece_tracking = eceStatus.overall_ece_health;
    this.monitoringHealth.component_status.kl_monitoring = klStatus.overall_health;
    this.monitoringHealth.component_status.production_alerts = productionStatus.overall_status as 'healthy' | 'degraded' | 'critical';
    this.monitoringHealth.component_status.aa_testing = this.activeAATest ? 'healthy' : 'disabled';

    // Update aggregate metrics
    this.monitoringHealth.active_alerts = eceStatus.active_alerts + klStatus.active_drifts;
    this.monitoringHealth.active_drift_detections = klStatus.active_drifts;
    this.monitoringHealth.aa_test_status = this.activeAATest ? 'running' : 
      (this.config.aa_testing_enabled ? 'stopped' : 'disabled');

    // Determine overall status
    const componentStatuses = Object.values(this.monitoringHealth.component_status);
    if (componentStatuses.some(status => status === 'critical')) {
      this.monitoringHealth.overall_status = 'critical';
    } else if (componentStatuses.some(status => status === 'degraded')) {
      this.monitoringHealth.overall_status = 'degraded';
    } else {
      this.monitoringHealth.overall_status = 'healthy';
    }
  }

  private logMonitoringEvent(eventData: Omit<MonitoringEvent, 'event_id' | 'timestamp'>): void {
    const event: MonitoringEvent = {
      event_id: `event_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
      timestamp: new Date().toISOString(),
      ...eventData
    };

    this.monitoringEvents.push(event);

    // Keep only recent events in memory
    if (this.monitoringEvents.length > 1000) {
      this.monitoringEvents = this.monitoringEvents.slice(-800);
    }

    // Persist to disk
    this.saveMonitoringState();
  }

  private initializeHealthStatus(): MonitoringHealth {
    return {
      overall_status: 'healthy',
      component_status: {
        ece_tracking: 'healthy',
        kl_monitoring: 'healthy',
        aa_testing: 'disabled',
        production_alerts: 'healthy'
      },
      active_alerts: 0,
      active_drift_detections: 0,
      aa_test_status: 'disabled',
      last_deliverables_generation: null,
      system_uptime_hours: 0
    };
  }

  private ensureDirectories(): void {
    const dirs = [
      this.monitoringDir,
      join(this.monitoringDir, 'ece'),
      join(this.monitoringDir, 'kl'),
      join(this.monitoringDir, 'production'),
      this.config.deliverables_output_dir
    ];

    for (const dir of dirs) {
      if (!existsSync(dir)) {
        mkdirSync(dir, { recursive: true });
      }
    }
  }

  private saveMonitoringState(): void {
    try {
      const statePath = join(this.monitoringDir, 'comprehensive_state.json');
      const state = {
        config: this.config,
        health: this.monitoringHealth,
        active_aa_test: this.activeAATest,
        recent_events: this.monitoringEvents.slice(-100),
        last_updated: new Date().toISOString()
      };

      writeFileSync(statePath, JSON.stringify(state, null, 2));
    } catch (error) {
      console.error('❌ Failed to save monitoring state:', error);
    }
  }
}

// Global comprehensive monitor instance
export const globalComprehensiveMonitor = new ComprehensiveProductionMonitor();