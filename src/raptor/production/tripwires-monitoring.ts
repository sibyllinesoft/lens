/**
 * Tripwires and Monitoring System
 * 
 * Implements: "util-heavy topic takeover, Stage-C p95 >+5%, topic staleness > TTL"
 * Addresses: Real-time monitoring with automatic intervention triggers
 */

import { EventEmitter } from 'events';
import { writeFile, readFile, mkdir } from 'fs/promises';
import { join } from 'path';

export interface TripwireCondition {
  id: string;
  name: string;
  category: 'performance' | 'quality' | 'behavior' | 'system';
  threshold: number;
  operator: '>' | '<' | '>=' | '<=' | '==' | '!=';
  unit: string;
  severity: 'critical' | 'warning' | 'info';
  action: TripwireAction;
  cooldown_minutes: number;
  description: string;
  enabled: boolean;
}

export interface TripwireAction {
  type: 'disable_feature' | 'reduce_weight' | 'force_refresh' | 'alert_only' | 'emergency_rollback';
  parameters: Record<string, any>;
  auto_execute: boolean;
  require_confirmation: boolean;
}

export interface MonitoringMetric {
  name: string;
  value: number;
  timestamp: Date;
  source: string;
  tags: Record<string, string>;
}

export interface TripwireEvent {
  tripwire_id: string;
  metric: MonitoringMetric;
  threshold_exceeded: number;
  action_taken: string;
  timestamp: Date;
  auto_resolved: boolean;
}

export interface SystemHealth {
  overall_status: 'healthy' | 'degraded' | 'critical';
  active_tripwires: string[];
  recent_events: TripwireEvent[];
  performance_summary: {
    p95_latency: number;
    p99_latency: number;
    qps: number;
    error_rate: number;
  };
  feature_status: {
    raptor_features: boolean;
    topic_fanout: boolean;
    nl_bridge: boolean;
  };
  why_mix_breakdown: {
    exact_fuzzy: number;
    symbol_struct: number;
    semantic: number;
  };
  topic_stats: {
    hit_rate: number;
    staleness_max: number;
    cluster_health: number;
  };
}

export class TripwireMonitor extends EventEmitter {
  private tripwires: Map<string, TripwireCondition>;
  private metricHistory: Map<string, MonitoringMetric[]>;
  private activeTripwires: Set<string>;
  private eventHistory: TripwireEvent[];
  private cooldowns: Map<string, Date>;
  private monitoringInterval?: NodeJS.Timeout;

  constructor() {
    super();
    this.tripwires = this.defineTripwires();
    this.metricHistory = new Map();
    this.activeTripwires = new Set();
    this.eventHistory = [];
    this.cooldowns = new Map();
  }

  /**
   * Define comprehensive tripwire conditions
   */
  private defineTripwires(): Map<string, TripwireCondition> {
    const tripwires = new Map<string, TripwireCondition>();

    // CRITICAL: Util-heavy topic takeover detection
    tripwires.set('util_semantic_takeover', {
      id: 'util_semantic_takeover',
      name: 'Utility Code Semantic Dominance',
      category: 'behavior',
      threshold: 0.45, // If semantic share >45% (normally ~20%)
      operator: '>',
      unit: 'ratio',
      severity: 'critical',
      action: {
        type: 'reduce_weight',
        parameters: {
          feature: 'raptor_features',
          weight_reduction: 0.5,
          target_semantic_share: 0.25
        },
        auto_execute: true,
        require_confirmation: false
      },
      cooldown_minutes: 10,
      description: 'Semantic matching share exceeds threshold, indicating util code over-boosting',
      enabled: true
    });

    // CRITICAL: Stage-C performance degradation
    tripwires.set('stage_c_latency_spike', {
      id: 'stage_c_latency_spike', 
      name: 'Stage-C p95 Latency Spike',
      category: 'performance',
      threshold: 1.05, // >5% increase
      operator: '>',
      unit: 'ratio_to_baseline',
      severity: 'critical',
      action: {
        type: 'disable_feature',
        parameters: {
          feature: 'stage_c_raptor',
          fallback_to: 'basic_ranking'
        },
        auto_execute: true,
        require_confirmation: false
      },
      cooldown_minutes: 15,
      description: 'Stage-C p95 latency >5% above baseline - disable RAPTOR features',
      enabled: true
    });

    // CRITICAL: Topic staleness beyond TTL
    tripwires.set('topic_staleness_exceeded', {
      id: 'topic_staleness_exceeded',
      name: 'Topic Cluster Staleness',
      category: 'system',
      threshold: 3600, // 1 hour TTL
      operator: '>',
      unit: 'seconds',
      severity: 'critical', 
      action: {
        type: 'force_refresh',
        parameters: {
          component: 'topic_clusters',
          partial_refresh: true,
          max_refresh_time: 300
        },
        auto_execute: true,
        require_confirmation: false
      },
      cooldown_minutes: 30,
      description: 'Topic cluster data exceeds TTL - force partial recluster',
      enabled: true
    });

    // WARNING: Quality degradation detection
    tripwires.set('quality_degradation', {
      id: 'quality_degradation',
      name: 'Quality Metric Degradation',
      category: 'quality',
      threshold: 0.95, // <5% drop from baseline
      operator: '<',
      unit: 'ratio_to_baseline',
      severity: 'warning',
      action: {
        type: 'alert_only',
        parameters: {
          alert_channels: ['slack', 'email'],
          escalation_time: 1800 // 30 minutes
        },
        auto_execute: true,
        require_confirmation: false
      },
      cooldown_minutes: 5,
      description: 'Quality metrics showing degradation trend',
      enabled: true
    });

    // WARNING: High p99 latency
    tripwires.set('p99_latency_spike', {
      id: 'p99_latency_spike',
      name: 'p99 Latency Spike',
      category: 'performance',
      threshold: 500, // >500ms p99
      operator: '>',
      unit: 'milliseconds',
      severity: 'warning',
      action: {
        type: 'reduce_weight',
        parameters: {
          feature: 'topic_fanout',
          weight_reduction: 0.3,
          duration_minutes: 15
        },
        auto_execute: false,
        require_confirmation: true
      },
      cooldown_minutes: 10,
      description: 'p99 latency spike detected - consider reducing topic fanout',
      enabled: true
    });

    // CRITICAL: Error rate spike
    tripwires.set('error_rate_spike', {
      id: 'error_rate_spike',
      name: 'Error Rate Spike',
      category: 'system',
      threshold: 0.05, // >5% error rate
      operator: '>',
      unit: 'ratio',
      severity: 'critical',
      action: {
        type: 'emergency_rollback',
        parameters: {
          rollback_order: ['stage_c_raptor', 'stage_a_topic_prior', 'nl_bridge'],
          max_rollback_time: 60
        },
        auto_execute: false,
        require_confirmation: true
      },
      cooldown_minutes: 5,
      description: 'High error rate - consider emergency rollback',
      enabled: true
    });

    // INFO: Topic hit rate monitoring
    tripwires.set('low_topic_hit_rate', {
      id: 'low_topic_hit_rate',
      name: 'Low Topic Hit Rate',
      category: 'behavior',
      threshold: 0.6, // <60% hit rate
      operator: '<',
      unit: 'ratio',
      severity: 'info',
      action: {
        type: 'alert_only',
        parameters: {
          alert_channels: ['dashboard'],
          include_diagnostics: true
        },
        auto_execute: true,
        require_confirmation: false
      },
      cooldown_minutes: 30,
      description: 'Topic clustering not effectively used',
      enabled: true
    });

    // INFO: Alias resolution depth monitoring
    tripwires.set('high_alias_depth', {
      id: 'high_alias_depth',
      name: 'High Alias Resolution Depth',
      category: 'behavior',
      threshold: 5.0, // >5 average depth
      operator: '>',
      unit: 'average_depth',
      severity: 'info',
      action: {
        type: 'alert_only',
        parameters: {
          alert_channels: ['dashboard'],
          track_symbol_patterns: true
        },
        auto_execute: true,
        require_confirmation: false
      },
      cooldown_minutes: 60,
      description: 'Deep alias resolution may indicate complex symbol chains',
      enabled: true
    });

    return tripwires;
  }

  /**
   * Start continuous monitoring
   */
  startMonitoring(intervalMs: number = 30000): void {
    console.log('🔍 Starting tripwire monitoring...');
    
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
    }

    this.monitoringInterval = setInterval(async () => {
      try {
        await this.checkAllTripwires();
      } catch (error) {
        console.error('❌ Tripwire monitoring error:', error);
      }
    }, intervalMs);

    console.log(`✓ Monitoring started with ${intervalMs}ms interval`);
  }

  /**
   * Stop monitoring
   */
  stopMonitoring(): void {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = undefined;
      console.log('⏹️ Tripwire monitoring stopped');
    }
  }

  /**
   * Ingest new metrics for monitoring
   */
  ingestMetrics(metrics: MonitoringMetric[]): void {
    for (const metric of metrics) {
      // Store metric history
      if (!this.metricHistory.has(metric.name)) {
        this.metricHistory.set(metric.name, []);
      }
      
      const history = this.metricHistory.get(metric.name)!;
      history.push(metric);
      
      // Keep only last 1000 data points
      if (history.length > 1000) {
        history.shift();
      }

      // Emit metric event
      this.emit('metric_received', metric);
    }
  }

  /**
   * Check all enabled tripwires
   */
  private async checkAllTripwires(): Promise<void> {
    const enabledTripwires = Array.from(this.tripwires.values()).filter(t => t.enabled);
    
    for (const tripwire of enabledTripwires) {
      // Check cooldown
      const cooldownUntil = this.cooldowns.get(tripwire.id);
      if (cooldownUntil && new Date() < cooldownUntil) {
        continue;
      }

      // Get relevant metrics
      const relevantMetrics = this.getRelevantMetrics(tripwire);
      
      for (const metric of relevantMetrics) {
        const triggered = this.evaluateTripwire(tripwire, metric);
        
        if (triggered) {
          await this.handleTripwireActivation(tripwire, metric);
        }
      }
    }
  }

  /**
   * Get metrics relevant to a tripwire
   */
  private getRelevantMetrics(tripwire: TripwireCondition): MonitoringMetric[] {
    // Map tripwire IDs to metric names
    const metricMapping: Record<string, string[]> = {
      'util_semantic_takeover': ['why_mix_semantic_ratio'],
      'stage_c_latency_spike': ['stage_c_p95_latency'],
      'topic_staleness_exceeded': ['topic_cluster_staleness'],
      'quality_degradation': ['ndcg_10', 'p_at_1', 'recall_50'],
      'p99_latency_spike': ['p99_latency'],
      'error_rate_spike': ['error_rate'],
      'low_topic_hit_rate': ['topic_hit_rate'],
      'high_alias_depth': ['alias_resolution_depth']
    };

    const metricNames = metricMapping[tripwire.id] || [];
    const metrics: MonitoringMetric[] = [];

    for (const metricName of metricNames) {
      const history = this.metricHistory.get(metricName);
      if (history && history.length > 0) {
        // Get most recent metric
        metrics.push(history[history.length - 1]);
      }
    }

    return metrics;
  }

  /**
   * Evaluate if a tripwire should trigger
   */
  private evaluateTripwire(tripwire: TripwireCondition, metric: MonitoringMetric): boolean {
    const value = metric.value;
    const threshold = tripwire.threshold;

    switch (tripwire.operator) {
      case '>': return value > threshold;
      case '<': return value < threshold;
      case '>=': return value >= threshold;
      case '<=': return value <= threshold;
      case '==': return Math.abs(value - threshold) < 0.001;
      case '!=': return Math.abs(value - threshold) >= 0.001;
      default: return false;
    }
  }

  /**
   * Handle tripwire activation
   */
  private async handleTripwireActivation(
    tripwire: TripwireCondition, 
    metric: MonitoringMetric
  ): Promise<void> {
    console.log(`🚨 TRIPWIRE ACTIVATED: ${tripwire.name}`);
    console.log(`   Metric: ${metric.name} = ${metric.value} ${tripwire.unit}`);
    console.log(`   Threshold: ${tripwire.operator} ${tripwire.threshold} ${tripwire.unit}`);

    // Create tripwire event
    const event: TripwireEvent = {
      tripwire_id: tripwire.id,
      metric,
      threshold_exceeded: Math.abs(metric.value - tripwire.threshold),
      action_taken: tripwire.action.type,
      timestamp: new Date(),
      auto_resolved: false
    };

    // Execute action if auto_execute is true
    if (tripwire.action.auto_execute && !tripwire.action.require_confirmation) {
      await this.executeTripwireAction(tripwire, event);
      event.action_taken = `auto_executed_${tripwire.action.type}`;
    } else if (tripwire.action.require_confirmation) {
      console.log(`⚠️  Action requires confirmation: ${tripwire.action.type}`);
      event.action_taken = 'pending_confirmation';
    }

    // Record event
    this.eventHistory.push(event);
    this.activeTripwires.add(tripwire.id);

    // Set cooldown
    const cooldownUntil = new Date();
    cooldownUntil.setMinutes(cooldownUntil.getMinutes() + tripwire.cooldown_minutes);
    this.cooldowns.set(tripwire.id, cooldownUntil);

    // Emit events
    this.emit('tripwire_activated', event);
    
    if (tripwire.severity === 'critical') {
      this.emit('critical_alert', event);
    }
  }

  /**
   * Execute tripwire action
   */
  private async executeTripwireAction(tripwire: TripwireCondition, event: TripwireEvent): Promise<void> {
    const action = tripwire.action;
    
    console.log(`🛠️ Executing action: ${action.type}`);

    switch (action.type) {
      case 'disable_feature':
        await this.disableFeature(action.parameters);
        break;
        
      case 'reduce_weight':
        await this.reduceWeight(action.parameters);
        break;
        
      case 'force_refresh':
        await this.forceRefresh(action.parameters);
        break;
        
      case 'alert_only':
        await this.sendAlert(action.parameters, event);
        break;
        
      case 'emergency_rollback':
        await this.initiateEmergencyRollback(action.parameters);
        break;
        
      default:
        console.warn(`Unknown action type: ${action.type}`);
    }

    console.log(`✓ Action ${action.type} executed`);
  }

  private async disableFeature(params: any): Promise<void> {
    console.log(`🚫 Disabling feature: ${params.feature}`);
    // Implementation would disable the specified feature
    // This might involve updating configuration flags, routing changes, etc.
  }

  private async reduceWeight(params: any): Promise<void> {
    console.log(`📉 Reducing weight for: ${params.feature} by ${params.weight_reduction}`);
    // Implementation would reduce weights in the ranking algorithm
  }

  private async forceRefresh(params: any): Promise<void> {
    console.log(`🔄 Force refreshing: ${params.component}`);
    // Implementation would trigger refresh of stale components
  }

  private async sendAlert(params: any, event: TripwireEvent): Promise<void> {
    console.log(`📢 Sending alert to: ${params.alert_channels?.join(', ')}`);
    // Implementation would send alerts to specified channels
  }

  private async initiateEmergencyRollback(params: any): Promise<void> {
    console.log(`🚨 Initiating emergency rollback: ${params.rollback_order?.join(' → ')}`);
    // Implementation would execute rollback sequence
  }

  /**
   * Get current system health status
   */
  getSystemHealth(): SystemHealth {
    // Get recent metrics for health calculation
    const recentMetrics = this.getRecentMetrics(300); // Last 5 minutes

    return {
      overall_status: this.calculateOverallStatus(),
      active_tripwires: Array.from(this.activeTripwires),
      recent_events: this.eventHistory.slice(-10),
      performance_summary: {
        p95_latency: this.getLatestMetricValue('p95_latency') || 0,
        p99_latency: this.getLatestMetricValue('p99_latency') || 0,
        qps: this.getLatestMetricValue('qps') || 0,
        error_rate: this.getLatestMetricValue('error_rate') || 0
      },
      feature_status: {
        raptor_features: !this.activeTripwires.has('stage_c_latency_spike'),
        topic_fanout: !this.activeTripwires.has('util_semantic_takeover'),
        nl_bridge: !this.activeTripwires.has('error_rate_spike')
      },
      why_mix_breakdown: {
        exact_fuzzy: this.getLatestMetricValue('why_mix_exact_fuzzy') || 0.45,
        symbol_struct: this.getLatestMetricValue('why_mix_symbol_struct') || 0.35,
        semantic: this.getLatestMetricValue('why_mix_semantic_ratio') || 0.20
      },
      topic_stats: {
        hit_rate: this.getLatestMetricValue('topic_hit_rate') || 0.76,
        staleness_max: this.getLatestMetricValue('topic_cluster_staleness') || 0,
        cluster_health: this.getLatestMetricValue('cluster_health') || 1.0
      }
    };
  }

  private calculateOverallStatus(): 'healthy' | 'degraded' | 'critical' {
    const criticalTripwires = Array.from(this.activeTripwires)
      .filter(id => this.tripwires.get(id)?.severity === 'critical');
    
    if (criticalTripwires.length > 0) return 'critical';
    if (this.activeTripwires.size > 0) return 'degraded';
    return 'healthy';
  }

  private getRecentMetrics(seconds: number): MonitoringMetric[] {
    const cutoff = new Date();
    cutoff.setSeconds(cutoff.getSeconds() - seconds);
    
    const recent: MonitoringMetric[] = [];
    for (const metrics of this.metricHistory.values()) {
      recent.push(...metrics.filter(m => m.timestamp >= cutoff));
    }
    
    return recent;
  }

  private getLatestMetricValue(metricName: string): number | undefined {
    const history = this.metricHistory.get(metricName);
    return history && history.length > 0 ? history[history.length - 1].value : undefined;
  }

  /**
   * Generate monitoring dashboard data
   */
  generateDashboardData(): any {
    const health = this.getSystemHealth();
    
    return {
      timestamp: new Date(),
      health_status: health.overall_status,
      active_alerts: health.active_tripwires.length,
      performance: health.performance_summary,
      features: health.feature_status,
      why_mix: health.why_mix_breakdown,
      topics: health.topic_stats,
      recent_events: health.recent_events.map(e => ({
        time: e.timestamp,
        type: e.tripwire_id,
        action: e.action_taken
      }))
    };
  }
}

// Factory function
export function createTripwireMonitor(): TripwireMonitor {
  return new TripwireMonitor();
}

// CLI demo
if (import.meta.main) {
  console.log('🔍 Tripwire Monitoring System Demo\n');
  
  const monitor = createTripwireMonitor();
  
  // Set up event listeners
  monitor.on('tripwire_activated', (event: TripwireEvent) => {
    console.log(`🚨 Alert: ${event.tripwire_id} activated`);
  });
  
  monitor.on('critical_alert', (event: TripwireEvent) => {
    console.log(`💥 CRITICAL: ${event.tripwire_id} - immediate action required`);
  });

  // Generate demo metrics
  const demoMetrics: MonitoringMetric[] = [
    {
      name: 'why_mix_semantic_ratio',
      value: 0.48, // Will trigger util_semantic_takeover
      timestamp: new Date(),
      source: 'query_analyzer',
      tags: { component: 'ranking' }
    },
    {
      name: 'stage_c_p95_latency',
      value: 1.07, // Will trigger stage_c_latency_spike (7% increase)
      timestamp: new Date(),
      source: 'latency_monitor',
      tags: { component: 'stage_c' }
    },
    {
      name: 'topic_cluster_staleness',
      value: 3800, // Will trigger topic_staleness_exceeded (>1 hour)
      timestamp: new Date(),
      source: 'cluster_monitor', 
      tags: { component: 'topics' }
    }
  ];

  // Ingest metrics and check tripwires
  monitor.ingestMetrics(demoMetrics);
  
  // Start monitoring (for demo, we'll just check once)
  console.log('📊 Demo metrics ingested, checking tripwires...\n');
  
  setTimeout(async () => {
    const health = monitor.getSystemHealth();
    
    console.log('🏥 System Health Summary:');
    console.log(`Status: ${health.overall_status.toUpperCase()}`);
    console.log(`Active Tripwires: ${health.active_tripwires.length}`);
    console.log(`Recent Events: ${health.recent_events.length}`);
    
    if (health.active_tripwires.length > 0) {
      console.log('\n🚨 Active Tripwires:');
      health.active_tripwires.forEach(id => console.log(`  - ${id}`));
    }
    
    console.log('\n📊 Dashboard Data:');
    console.log(JSON.stringify(monitor.generateDashboardData(), null, 2));
  }, 1000);
}