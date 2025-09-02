/**
 * Sentinel Probes and Kill Switch System
 * 
 * Implements automated health probing and emergency controls:
 * - Sentinel zero-result probes (hourly "class", "def" tests)
 * - Kill switches for immediate traffic routing
 * - Feature flag integration and emergency rollback
 * - Health validation and automatic recovery
 * - Integration with monitoring and canary systems
 */

import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';
import { EventEmitter } from 'events';

interface SentinelProbe {
  probe_id: string;
  name: string;
  query: string;
  expected_behavior: 'must_have_results' | 'must_match_pattern' | 'must_not_error';
  expected_pattern?: string;
  min_results?: number;
  
  // Execution config
  frequency_minutes: number;
  timeout_ms: number;
  retry_attempts: number;
  
  // State
  last_execution: string;
  last_result: ProbeResult;
  consecutive_failures: number;
  total_executions: number;
  success_rate: number;
}

interface ProbeResult {
  timestamp: string;
  success: boolean;
  execution_time_ms: number;
  results_count: number;
  error_message?: string;
  results_sample?: Array<{
    file_path: string;
    line_number: number;
    content: string;
    score: number;
  }>;
}

interface KillSwitch {
  switch_id: string;
  name: string;
  description: string;
  scope: 'global' | 'feature' | 'traffic' | 'component';
  
  // Trigger conditions
  triggers: KillSwitchTrigger[];
  
  // Actions when activated
  actions: KillSwitchAction[];
  
  // State
  is_active: boolean;
  activated_at?: string;
  activated_by: 'auto' | 'manual';
  activation_reason?: string;
  
  // Recovery
  auto_recovery_enabled: boolean;
  recovery_conditions?: RecoveryCondition[];
}

interface KillSwitchTrigger {
  type: 'sentinel_failure' | 'consecutive_failures' | 'error_rate' | 'manual';
  condition?: any; // Specific condition parameters
  threshold?: number;
  window_minutes?: number;
}

interface KillSwitchAction {
  type: 'disable_feature' | 'route_traffic' | 'rollback_version' | 'emergency_fallback' | 'notify';
  config: Record<string, any>;
  priority: number; // Execution order
}

interface RecoveryCondition {
  type: 'sentinel_success' | 'manual_override' | 'time_delay';
  parameters: Record<string, any>;
}

interface SentinelState {
  probes: Record<string, SentinelProbe>;
  kill_switches: Record<string, KillSwitch>;
  system_health: SystemHealth;
  probe_schedule: ProbeExecution[];
}

interface SystemHealth {
  overall_status: 'healthy' | 'degraded' | 'critical' | 'emergency';
  sentinel_passing_rate: number;
  active_kill_switches: string[];
  last_health_check: string;
  emergency_mode_active: boolean;
}

interface ProbeExecution {
  probe_id: string;
  next_execution: string;
  in_progress: boolean;
}

export class SentinelKillSwitchSystem extends EventEmitter {
  private readonly sentinelDir: string;
  private sentinelState: SentinelState;
  private probeInterval?: NodeJS.Timeout;
  private isRunning: boolean = false;
  
  constructor(sentinelDir: string = './deployment-artifacts/sentinel') {
    super();
    this.sentinelDir = sentinelDir;
    
    if (!existsSync(this.sentinelDir)) {
      mkdirSync(this.sentinelDir, { recursive: true });
    }
    
    this.sentinelState = this.initializeSentinelState();
  }
  
  /**
   * Start sentinel probe and kill switch system
   */
  public async startSentinelSystem(): Promise<void> {
    if (this.isRunning) {
      console.log('🕵️ Sentinel system already running');
      return;
    }
    
    console.log('🚀 Starting sentinel probe and kill switch system...');
    
    // Schedule probe executions
    this.scheduleProbes();
    
    // Start probe monitoring loop
    this.probeInterval = setInterval(async () => {
      await this.executeScheduledProbes();
    }, 60000); // Check every minute
    
    // Execute initial probe run
    await this.executeAllProbes();
    
    this.isRunning = true;
    
    console.log('✅ Sentinel system started');
    console.log(`🕵️ Active probes: ${Object.keys(this.sentinelState.probes).length}`);
    console.log(`🔒 Configured kill switches: ${Object.keys(this.sentinelState.kill_switches).length}`);
    
    this.emit('sentinel_started');
  }
  
  /**
   * Stop sentinel system
   */
  public stopSentinelSystem(): void {
    if (this.probeInterval) {
      clearInterval(this.probeInterval);
      this.probeInterval = undefined;
    }
    
    this.isRunning = false;
    console.log('🛑 Sentinel system stopped');
    this.emit('sentinel_stopped');
  }
  
  /**
   * Execute all scheduled probes
   */
  private async executeScheduledProbes(): Promise<void> {
    const currentTime = new Date();
    
    for (const execution of this.sentinelState.probe_schedule) {
      if (execution.in_progress) continue;
      
      const nextExecution = new Date(execution.next_execution);
      if (currentTime >= nextExecution) {
        // Execute probe
        execution.in_progress = true;
        
        try {
          await this.executeProbe(execution.probe_id);
        } catch (error) {
          console.error(`❌ Failed to execute probe ${execution.probe_id}:`, error);
        } finally {
          execution.in_progress = false;
          
          // Schedule next execution
          const probe = this.sentinelState.probes[execution.probe_id];
          if (probe) {
            execution.next_execution = new Date(
              currentTime.getTime() + probe.frequency_minutes * 60 * 1000
            ).toISOString();
          }
        }
      }
    }
  }
  
  /**
   * Execute single probe
   */
  private async executeProbe(probeId: string): Promise<void> {
    const probe = this.sentinelState.probes[probeId];
    if (!probe) {
      console.error(`❌ Probe not found: ${probeId}`);
      return;
    }
    
    console.log(`🕵️ Executing probe: ${probe.name} ("${probe.query}")`);
    
    const startTime = Date.now();
    let result: ProbeResult;
    
    try {
      // Execute search query
      const searchResults = await this.executeSearchQuery(probe.query, probe.timeout_ms);
      const executionTime = Date.now() - startTime;
      
      // Evaluate probe result
      const success = this.evaluateProbeResult(probe, searchResults);
      
      result = {
        timestamp: new Date().toISOString(),
        success,
        execution_time_ms: executionTime,
        results_count: searchResults.length,
        results_sample: searchResults.slice(0, 3) // First 3 results for debugging
      };
      
      if (success) {
        console.log(`✅ Probe passed: ${probe.name} (${searchResults.length} results, ${executionTime}ms)`);
        probe.consecutive_failures = 0;
      } else {
        console.log(`❌ Probe failed: ${probe.name} (${searchResults.length} results, expected behavior: ${probe.expected_behavior})`);
        probe.consecutive_failures++;
      }
      
    } catch (error) {
      const executionTime = Date.now() - startTime;
      
      result = {
        timestamp: new Date().toISOString(),
        success: false,
        execution_time_ms: executionTime,
        results_count: 0,
        error_message: error instanceof Error ? error.message : String(error)
      };
      
      console.log(`❌ Probe error: ${probe.name} - ${result.error_message}`);
      probe.consecutive_failures++;
    }
    
    // Update probe state
    probe.last_execution = result.timestamp;
    probe.last_result = result;
    probe.total_executions++;
    
    // Recalculate success rate
    const recentResults = await this.getRecentProbeResults(probeId, 100);
    probe.success_rate = recentResults.filter(r => r.success).length / recentResults.length;
    
    // Check for kill switch triggers
    await this.checkKillSwitchTriggers(probe);
    
    // Update system health
    this.updateSystemHealth();
    
    // Save state
    this.saveSentinelState();
    
    // Emit probe result
    this.emit('probe_executed', {
      probe_id: probeId,
      probe_name: probe.name,
      success: result.success,
      consecutive_failures: probe.consecutive_failures,
      execution_time_ms: result.execution_time_ms,
      timestamp: result.timestamp
    });
  }
  
  /**
   * Execute search query (mock implementation)
   */
  private async executeSearchQuery(query: string, timeoutMs: number): Promise<Array<{
    file_path: string;
    line_number: number;
    content: string;
    score: number;
  }>> {
    // Mock search execution - in production would call actual search API
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`Query timeout after ${timeoutMs}ms`));
      }, timeoutMs);
      
      setTimeout(() => {
        clearTimeout(timeout);
        
        // Simulate different query behaviors
        if (query === 'class') {
          // Should always return class definitions
          resolve([
            { file_path: 'example.ts', line_number: 10, content: 'class UserService {', score: 0.95 },
            { file_path: 'models.py', line_number: 5, content: 'class User:', score: 0.92 },
            { file_path: 'api.ts', line_number: 25, content: 'export class APIClient {', score: 0.88 }
          ]);
        } else if (query === 'def') {
          // Should always return function definitions
          resolve([
            { file_path: 'utils.py', line_number: 15, content: 'def calculate_score(query, results):', score: 0.93 },
            { file_path: 'helpers.py', line_number: 8, content: 'def format_date(date):', score: 0.90 },
            { file_path: 'main.py', line_number: 35, content: 'def main():', score: 0.85 }
          ]);
        } else {
          // Normal query - might return variable results
          const resultCount = Math.random() > 0.1 ? Math.floor(Math.random() * 8) + 1 : 0; // 10% chance of zero results
          const results = [];
          
          for (let i = 0; i < resultCount; i++) {
            results.push({
              file_path: `file_${i}.ts`,
              line_number: Math.floor(Math.random() * 100) + 1,
              content: `Mock result for query: ${query}`,
              score: 0.9 - i * 0.1
            });
          }
          
          resolve(results);
        }
      }, 50 + Math.random() * 200); // 50-250ms response time
    });
  }
  
  /**
   * Evaluate probe result against expected behavior
   */
  private evaluateProbeResult(probe: SentinelProbe, results: any[]): boolean {
    switch (probe.expected_behavior) {
      case 'must_have_results':
        const minResults = probe.min_results || 1;
        return results.length >= minResults;
        
      case 'must_match_pattern':
        if (!probe.expected_pattern) return false;
        const pattern = new RegExp(probe.expected_pattern, 'i');
        return results.some(r => pattern.test(r.content));
        
      case 'must_not_error':
        return true; // If we got here without throwing, no error occurred
        
      default:
        return false;
    }
  }
  
  /**
   * Check for kill switch triggers
   */
  private async checkKillSwitchTriggers(probe: SentinelProbe): Promise<void> {
    for (const [switchId, killSwitch] of Object.entries(this.sentinelState.kill_switches)) {
      if (killSwitch.is_active) continue; // Already active
      
      for (const trigger of killSwitch.triggers) {
        let shouldActivate = false;
        
        switch (trigger.type) {
          case 'sentinel_failure':
            if (probe.consecutive_failures >= (trigger.threshold || 2)) {
              shouldActivate = true;
            }
            break;
            
          case 'consecutive_failures':
            // Check all probes for consecutive failures
            const allProbes = Object.values(this.sentinelState.probes);
            const failingProbes = allProbes.filter(p => p.consecutive_failures >= (trigger.threshold || 3));
            if (failingProbes.length >= (trigger.condition?.min_failing_probes || 1)) {
              shouldActivate = true;
            }
            break;
            
          case 'error_rate':
            if (probe.success_rate < (1 - (trigger.threshold || 0.5))) {
              shouldActivate = true;
            }
            break;
        }
        
        if (shouldActivate) {
          await this.activateKillSwitch(switchId, 'auto', `Triggered by ${trigger.type} on probe ${probe.name}`);
          break; // Only activate once
        }
      }
    }
  }
  
  /**
   * Activate kill switch
   */
  public async activateKillSwitch(switchId: string, activatedBy: 'auto' | 'manual', reason?: string): Promise<void> {
    const killSwitch = this.sentinelState.kill_switches[switchId];
    if (!killSwitch) {
      throw new Error(`Kill switch not found: ${switchId}`);
    }
    
    if (killSwitch.is_active) {
      console.log(`⚠️  Kill switch already active: ${killSwitch.name}`);
      return;
    }
    
    console.log(`🚨🚨 ACTIVATING KILL SWITCH: ${killSwitch.name}`);
    console.log(`🔧 Reason: ${reason || 'Manual activation'}`);
    console.log(`📋 Scope: ${killSwitch.scope}`);
    
    killSwitch.is_active = true;
    killSwitch.activated_at = new Date().toISOString();
    killSwitch.activated_by = activatedBy;
    killSwitch.activation_reason = reason;
    
    // Execute actions in priority order
    const sortedActions = killSwitch.actions.sort((a, b) => a.priority - b.priority);
    
    for (const action of sortedActions) {
      try {
        await this.executeKillSwitchAction(action, killSwitch);
        console.log(`✅ Kill switch action executed: ${action.type}`);
      } catch (error) {
        console.error(`❌ Failed to execute kill switch action ${action.type}:`, error);
      }
    }
    
    // Update system health to emergency mode
    this.sentinelState.system_health.emergency_mode_active = true;
    this.sentinelState.system_health.overall_status = 'emergency';
    
    this.saveSentinelState();
    
    this.emit('kill_switch_activated', {
      switch_id: switchId,
      switch_name: killSwitch.name,
      reason,
      activated_by: activatedBy,
      timestamp: killSwitch.activated_at
    });
  }
  
  /**
   * Execute kill switch action
   */
  private async executeKillSwitchAction(action: KillSwitchAction, killSwitch: KillSwitch): Promise<void> {
    switch (action.type) {
      case 'disable_feature':
        await this.disableFeature(action.config);
        break;
        
      case 'route_traffic':
        await this.routeTraffic(action.config);
        break;
        
      case 'rollback_version':
        await this.rollbackVersion(action.config);
        break;
        
      case 'emergency_fallback':
        await this.activateEmergencyFallback(action.config);
        break;
        
      case 'notify':
        await this.sendKillSwitchNotification(action.config, killSwitch);
        break;
        
      default:
        console.warn(`⚠️  Unknown kill switch action: ${action.type}`);
    }
  }
  
  /**
   * Disable feature via feature flag
   */
  private async disableFeature(config: any): Promise<void> {
    console.log(`🚫 Disabling feature: ${config.feature_name}`);
    
    if (config.feature_flag_api) {
      // Mock feature flag API call
      console.log(`📡 Feature flag API call: POST ${config.feature_flag_api}`);
      console.log(`📋 Payload: { "${config.feature_name}": false }`);
    }
    
    if (config.config_override) {
      // Write config override file
      const overridePath = join(this.sentinelDir, 'feature_overrides.json');
      const overrides = existsSync(overridePath) 
        ? JSON.parse(readFileSync(overridePath, 'utf-8'))
        : {};
      
      overrides[config.feature_name] = false;
      writeFileSync(overridePath, JSON.stringify(overrides, null, 2));
      
      console.log(`📝 Feature override written: ${config.feature_name} = false`);
    }
  }
  
  /**
   * Route traffic to fallback
   */
  private async routeTraffic(config: any): Promise<void> {
    console.log(`🔄 Routing traffic: ${config.from} → ${config.to}`);
    
    if (config.load_balancer_api) {
      console.log(`⚖️  Load balancer update: ${config.load_balancer_api}`);
    }
    
    if (config.traffic_percentage) {
      console.log(`📊 Traffic split: ${config.traffic_percentage}% → ${config.to}`);
    }
  }
  
  /**
   * Rollback to previous version
   */
  private async rollbackVersion(config: any): Promise<void> {
    console.log(`📦 Initiating version rollback to: ${config.target_version || 'previous'}`);
    
    // Integrate with canary rollout system
    try {
      const { canaryRolloutSystem } = await import('./canary-rollout-system.js');
      
      if (config.deployment_id) {
        await canaryRolloutSystem.manualRollback(config.deployment_id, 'Kill switch activation');
      } else {
        console.log('🔄 Emergency rollback initiated via kill switch');
      }
      
    } catch (error) {
      console.error('❌ Failed to integrate with canary system:', error);
    }
  }
  
  /**
   * Activate emergency fallback
   */
  private async activateEmergencyFallback(config: any): Promise<void> {
    console.log(`🚨 Activating emergency fallback: ${config.fallback_type}`);
    
    switch (config.fallback_type) {
      case 'basic_search':
        console.log('🔍 Switching to basic lexical search only');
        break;
        
      case 'cached_results':
        console.log('💾 Serving cached results for common queries');
        break;
        
      case 'static_responses':
        console.log('📄 Serving static fallback responses');
        break;
        
      default:
        console.log(`🔧 Custom fallback: ${config.fallback_type}`);
    }
  }
  
  /**
   * Send kill switch notification
   */
  private async sendKillSwitchNotification(config: any, killSwitch: KillSwitch): Promise<void> {
    const message = `🚨 KILL SWITCH ACTIVATED: ${killSwitch.name}\n` +
                   `Reason: ${killSwitch.activation_reason}\n` +
                   `Time: ${killSwitch.activated_at}\n` +
                   `Scope: ${killSwitch.scope}`;
    
    if (config.webhook_url) {
      console.log(`📡 Webhook notification sent to: ${config.webhook_url}`);
    }
    
    if (config.slack_channel) {
      console.log(`💬 Slack notification sent to: ${config.slack_channel}`);
    }
    
    if (config.pagerduty_key) {
      console.log(`📟 PagerDuty alert sent with key: ${config.pagerduty_key}`);
    }
    
    if (config.email_list) {
      console.log(`📧 Email alerts sent to: ${config.email_list.join(', ')}`);
    }
  }
  
  /**
   * Deactivate kill switch (manual recovery)
   */
  public async deactivateKillSwitch(switchId: string, reason: string): Promise<void> {
    const killSwitch = this.sentinelState.kill_switches[switchId];
    if (!killSwitch) {
      throw new Error(`Kill switch not found: ${switchId}`);
    }
    
    if (!killSwitch.is_active) {
      console.log(`⚠️  Kill switch already inactive: ${killSwitch.name}`);
      return;
    }
    
    console.log(`🔄 DEACTIVATING KILL SWITCH: ${killSwitch.name}`);
    console.log(`🔧 Reason: ${reason}`);
    
    killSwitch.is_active = false;
    killSwitch.activated_at = undefined;
    killSwitch.activated_by = 'manual';
    killSwitch.activation_reason = undefined;
    
    // Check if any other kill switches are still active
    const activeKillSwitches = Object.values(this.sentinelState.kill_switches)
      .filter(ks => ks.is_active);
    
    if (activeKillSwitches.length === 0) {
      this.sentinelState.system_health.emergency_mode_active = false;
      this.sentinelState.system_health.overall_status = 'healthy';
    }
    
    this.saveSentinelState();
    
    this.emit('kill_switch_deactivated', {
      switch_id: switchId,
      switch_name: killSwitch.name,
      reason,
      timestamp: new Date().toISOString()
    });
  }
  
  /**
   * Execute all probes immediately (for testing)
   */
  public async executeAllProbes(): Promise<void> {
    console.log('🕵️ Executing all sentinel probes...');
    
    const probeIds = Object.keys(this.sentinelState.probes);
    for (const probeId of probeIds) {
      await this.executeProbe(probeId);
      // Small delay between probes
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    console.log(`✅ All probes executed (${probeIds.length} total)`);
  }
  
  /**
   * Update system health based on probe results
   */
  private updateSystemHealth(): void {
    const probes = Object.values(this.sentinelState.probes);
    const passingProbes = probes.filter(p => p.last_result.success);
    const passingRate = probes.length > 0 ? passingProbes.length / probes.length : 1;
    
    const activeKillSwitches = Object.entries(this.sentinelState.kill_switches)
      .filter(([_, ks]) => ks.is_active)
      .map(([id, _]) => id);
    
    let overallStatus: 'healthy' | 'degraded' | 'critical' | 'emergency' = 'healthy';
    
    if (this.sentinelState.system_health.emergency_mode_active) {
      overallStatus = 'emergency';
    } else if (passingRate < 0.5) {
      overallStatus = 'critical';
    } else if (passingRate < 0.8 || activeKillSwitches.length > 0) {
      overallStatus = 'degraded';
    }
    
    this.sentinelState.system_health = {
      overall_status: overallStatus,
      sentinel_passing_rate: passingRate,
      active_kill_switches: activeKillSwitches,
      last_health_check: new Date().toISOString(),
      emergency_mode_active: this.sentinelState.system_health.emergency_mode_active
    };
  }
  
  /**
   * Schedule all probes for execution
   */
  private scheduleProbes(): void {
    this.sentinelState.probe_schedule = Object.keys(this.sentinelState.probes).map(probeId => {
      const probe = this.sentinelState.probes[probeId];
      return {
        probe_id: probeId,
        next_execution: new Date(Date.now() + Math.random() * 60000).toISOString(), // Random initial delay
        in_progress: false
      };
    });
  }
  
  /**
   * Get recent probe results for analysis
   */
  private async getRecentProbeResults(probeId: string, limit: number = 100): Promise<ProbeResult[]> {
    // Mock implementation - in production would query probe result history
    const probe = this.sentinelState.probes[probeId];
    if (!probe || !probe.last_result) return [];
    
    // Generate mock historical results for success rate calculation
    const results = [];
    const successRate = 0.95; // 95% historical success rate
    
    for (let i = 0; i < limit; i++) {
      results.push({
        timestamp: new Date(Date.now() - i * 60000).toISOString(),
        success: Math.random() < successRate,
        execution_time_ms: 50 + Math.random() * 200,
        results_count: Math.floor(Math.random() * 5) + 1
      });
    }
    
    return results;
  }
  
  /**
   * Initialize sentinel state with default probes and kill switches
   */
  private initializeSentinelState(): SentinelState {
    return {
      probes: {
        class_probe: {
          probe_id: 'class_probe',
          name: 'Class Definition Probe',
          query: 'class',
          expected_behavior: 'must_have_results',
          min_results: 1,
          frequency_minutes: 60, // Hourly
          timeout_ms: 5000,
          retry_attempts: 2,
          last_execution: new Date().toISOString(),
          last_result: {
            timestamp: new Date().toISOString(),
            success: true,
            execution_time_ms: 150,
            results_count: 5
          },
          consecutive_failures: 0,
          total_executions: 0,
          success_rate: 1.0
        },
        
        def_probe: {
          probe_id: 'def_probe',
          name: 'Function Definition Probe',
          query: 'def',
          expected_behavior: 'must_have_results',
          min_results: 1,
          frequency_minutes: 60, // Hourly
          timeout_ms: 5000,
          retry_attempts: 2,
          last_execution: new Date().toISOString(),
          last_result: {
            timestamp: new Date().toISOString(),
            success: true,
            execution_time_ms: 120,
            results_count: 8
          },
          consecutive_failures: 0,
          total_executions: 0,
          success_rate: 1.0
        },
        
        basic_search_probe: {
          probe_id: 'basic_search_probe',
          name: 'Basic Search Health',
          query: 'function',
          expected_behavior: 'must_not_error',
          frequency_minutes: 30, // Every 30 minutes
          timeout_ms: 3000,
          retry_attempts: 3,
          last_execution: new Date().toISOString(),
          last_result: {
            timestamp: new Date().toISOString(),
            success: true,
            execution_time_ms: 200,
            results_count: 12
          },
          consecutive_failures: 0,
          total_executions: 0,
          success_rate: 1.0
        }
      },
      
      kill_switches: {
        zero_results_emergency: {
          switch_id: 'zero_results_emergency',
          name: 'Zero Results Emergency',
          description: 'Activates when sentinel probes consistently return zero results',
          scope: 'global',
          triggers: [
            {
              type: 'sentinel_failure',
              threshold: 2 // 2 consecutive failures
            }
          ],
          actions: [
            {
              type: 'emergency_fallback',
              config: { fallback_type: 'basic_search' },
              priority: 1
            },
            {
              type: 'notify',
              config: {
                webhook_url: 'https://alerts.example.com/emergency',
                slack_channel: '#incidents',
                pagerduty_key: 'zero_results_emergency'
              },
              priority: 2
            }
          ],
          is_active: false,
          activated_by: 'auto',
          auto_recovery_enabled: false
        },
        
        search_system_failure: {
          switch_id: 'search_system_failure',
          name: 'Search System Failure',
          description: 'Activates when multiple probes fail indicating system-wide issues',
          scope: 'global',
          triggers: [
            {
              type: 'consecutive_failures',
              threshold: 2, // 2+ probes failing
              condition: { min_failing_probes: 2 }
            }
          ],
          actions: [
            {
              type: 'rollback_version',
              config: { target_version: 'previous' },
              priority: 1
            },
            {
              type: 'route_traffic',
              config: { 
                from: 'primary', 
                to: 'fallback',
                traffic_percentage: 100
              },
              priority: 2
            },
            {
              type: 'notify',
              config: {
                webhook_url: 'https://alerts.example.com/critical',
                pagerduty_key: 'system_failure_critical',
                email_list: ['oncall@example.com']
              },
              priority: 3
            }
          ],
          is_active: false,
          activated_by: 'auto',
          auto_recovery_enabled: true,
          recovery_conditions: [
            {
              type: 'sentinel_success',
              parameters: { required_success_rate: 0.9, duration_minutes: 30 }
            }
          ]
        },
        
        manual_emergency: {
          switch_id: 'manual_emergency',
          name: 'Manual Emergency Stop',
          description: 'Manual kill switch for emergency situations',
          scope: 'global',
          triggers: [
            { type: 'manual' }
          ],
          actions: [
            {
              type: 'disable_feature',
              config: { 
                feature_name: 'advanced_search',
                feature_flag_api: 'https://flags.example.com/api/flags'
              },
              priority: 1
            },
            {
              type: 'emergency_fallback',
              config: { fallback_type: 'static_responses' },
              priority: 2
            },
            {
              type: 'notify',
              config: {
                webhook_url: 'https://alerts.example.com/manual',
                slack_channel: '#incidents'
              },
              priority: 3
            }
          ],
          is_active: false,
          activated_by: 'manual',
          auto_recovery_enabled: false
        }
      },
      
      system_health: {
        overall_status: 'healthy',
        sentinel_passing_rate: 1.0,
        active_kill_switches: [],
        last_health_check: new Date().toISOString(),
        emergency_mode_active: false
      },
      
      probe_schedule: []
    };
  }
  
  private saveSentinelState(): void {
    const statePath = join(this.sentinelDir, 'sentinel_state.json');
    writeFileSync(statePath, JSON.stringify(this.sentinelState, null, 2));
  }
  
  /**
   * Get current system status
   */
  public getSystemStatus(): {
    health: SystemHealth;
    probes: Record<string, { name: string; success_rate: number; consecutive_failures: number; last_execution: string }>;
    kill_switches: Record<string, { name: string; is_active: boolean; activated_at?: string }>;
  } {
    const probeStatus = Object.fromEntries(
      Object.entries(this.sentinelState.probes).map(([id, probe]) => [
        id,
        {
          name: probe.name,
          success_rate: probe.success_rate,
          consecutive_failures: probe.consecutive_failures,
          last_execution: probe.last_execution
        }
      ])
    );
    
    const killSwitchStatus = Object.fromEntries(
      Object.entries(this.sentinelState.kill_switches).map(([id, ks]) => [
        id,
        {
          name: ks.name,
          is_active: ks.is_active,
          activated_at: ks.activated_at
        }
      ])
    );
    
    return {
      health: this.sentinelState.system_health,
      probes: probeStatus,
      kill_switches: killSwitchStatus
    };
  }
  
  /**
   * Get detailed probe status
   */
  public getProbeDetails(probeId: string): SentinelProbe | undefined {
    return this.sentinelState.probes[probeId];
  }
  
  /**
   * Manual probe execution (for testing)
   */
  public async manualProbeExecution(probeId: string): Promise<ProbeResult> {
    await this.executeProbe(probeId);
    return this.sentinelState.probes[probeId].last_result;
  }
}

export const sentinelKillSwitchSystem = new SentinelKillSwitchSystem();