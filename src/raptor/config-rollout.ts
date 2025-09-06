/**
 * Configuration & Rollout System for RAPTOR
 * 
 * Provides policy-based configuration management with gradual rollout controls,
 * kill-switch ordering, and feature flag management for production safety.
 * 
 * Features:
 * - Gradual rollout with percentage controls
 * - A/B testing configuration
 * - Kill switch for emergency shutdown
 * - Feature flags for component-level control
 * - Policy validation and safety checks
 */

import { EventEmitter } from 'events';

export interface RaptorConfig {
  // System-level controls
  enabled: boolean;
  rollout_percentage: number; // 0-100
  kill_switch_active: boolean;
  
  // Component feature flags
  features: {
    symbol_graph: boolean;
    card_store: boolean;
    topic_tree: boolean;
    stage_a_planner: boolean;
    stage_c_features: boolean;
    nl_symbol_bridge: boolean;
    metrics_telemetry: boolean;
  };
  
  // Performance parameters
  stage_a: {
    alpha: number; // topic similarity coefficient (0-1)
    beta: number;  // topic entropy coefficient (0-1)
    max_k_candidates: number;
    base_k_default: number;
    per_file_span_cap_base: number;
    per_file_span_cap_boosted: number;
    topic_similarity_threshold: number;
    min_topic_entropy: number;
    topic_search_timeout_ms: number;
    max_topics_considered: number;
  };
  
  stage_c: {
    max_weight_magnitude: number; // |wi| ≤ max_weight_magnitude
    isotonic_calibration: boolean;
    feature_timeout_ms: number;
    businessness_weight: number;
    topic_overlap_weight: number;
    cross_feature_enabled: boolean;
  };
  
  nl_bridge: {
    bm25_k1: number;
    bm25_b: number;
    max_symbol_extractions: number;
    subtoken_expansion: boolean;
    synonym_boost: number;
  };
  
  // Quality and safety bounds
  quality_gates: {
    min_ndcg_improvement: number;
    max_latency_increase_ms: number;
    max_memory_increase_mb: number;
    min_success_rate: number;
  };
  
  // Rollout strategy
  rollout: {
    strategy: 'percentage' | 'user_hash' | 'whitelist' | 'gradual';
    target_percentage: number;
    ramp_up_duration_hours: number;
    canary_percentage: number;
    monitoring_window_minutes: number;
  };
  
  // A/B testing
  ab_testing: {
    enabled: boolean;
    experiment_name: string;
    control_percentage: number; // LSP-only
    treatment_percentage: number; // LSP+RAPTOR
    metrics_collection_rate: number; // 0-1
  };
}

export interface PolicyRule {
  id: string;
  name: string;
  condition: string; // JSON logic expression
  action: 'allow' | 'deny' | 'modify';
  parameters?: Record<string, any>;
  priority: number;
  enabled: boolean;
}

export interface RolloutState {
  current_percentage: number;
  target_percentage: number;
  ramp_start_time: Date;
  last_update_time: Date;
  status: 'initializing' | 'ramping' | 'stable' | 'paused' | 'rolling_back';
  health_metrics: RolloutHealthMetrics;
}

export interface RolloutHealthMetrics {
  error_rate: number;
  latency_p95: number;
  memory_usage_mb: number;
  success_rate: number;
  user_satisfaction: number;
  alerts_triggered: string[];
}

export interface ConfigValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  safety_score: number; // 0-1, higher is safer
}

export interface RolloutDecision {
  should_enable: boolean;
  reason: string;
  user_hash?: string;
  experiment_group?: 'control' | 'treatment';
  feature_flags: Record<string, boolean>;
}

/**
 * Configuration and rollout management system
 */
export class ConfigRolloutManager extends EventEmitter {
  private config: RaptorConfig;
  private policies: PolicyRule[] = [];
  private rolloutState: RolloutState;
  private healthCheckInterval?: NodeJS.Timeout;

  constructor(initialConfig?: Partial<RaptorConfig>) {
    super();
    
    this.config = {
      enabled: false,
      rollout_percentage: 0,
      kill_switch_active: false,
      
      features: {
        symbol_graph: true,
        card_store: true,
        topic_tree: true,
        stage_a_planner: true,
        stage_c_features: true,
        nl_symbol_bridge: true,
        metrics_telemetry: true
      },
      
      stage_a: {
        alpha: 0.3,
        beta: 0.2,
        max_k_candidates: 320,
        base_k_default: 50,
        per_file_span_cap_base: 5,
        per_file_span_cap_boosted: 8,
        topic_similarity_threshold: 0.6,
        min_topic_entropy: 0.3,
        topic_search_timeout_ms: 100,
        max_topics_considered: 20
      },
      
      stage_c: {
        max_weight_magnitude: 0.4,
        isotonic_calibration: true,
        feature_timeout_ms: 50,
        businessness_weight: 0.15,
        topic_overlap_weight: 0.20,
        cross_feature_enabled: true
      },
      
      nl_bridge: {
        bm25_k1: 1.2,
        bm25_b: 0.75,
        max_symbol_extractions: 5,
        subtoken_expansion: true,
        synonym_boost: 0.1
      },
      
      quality_gates: {
        min_ndcg_improvement: 0.03, // 3 points
        max_latency_increase_ms: 50,
        max_memory_increase_mb: 100,
        min_success_rate: 0.95
      },
      
      rollout: {
        strategy: 'percentage',
        target_percentage: 0,
        ramp_up_duration_hours: 24,
        canary_percentage: 1,
        monitoring_window_minutes: 30
      },
      
      ab_testing: {
        enabled: false,
        experiment_name: 'raptor_semantic_search',
        control_percentage: 50,
        treatment_percentage: 50,
        metrics_collection_rate: 0.1
      },
      
      ...initialConfig
    };

    this.rolloutState = {
      current_percentage: 0,
      target_percentage: this.config.rollout.target_percentage,
      ramp_start_time: new Date(),
      last_update_time: new Date(),
      status: 'initializing',
      health_metrics: {
        error_rate: 0,
        latency_p95: 0,
        memory_usage_mb: 0,
        success_rate: 1,
        user_satisfaction: 1,
        alerts_triggered: []
      }
    };

    this.initializeDefaultPolicies();
    this.startHealthChecking();
  }

  /**
   * Get current configuration
   */
  getConfig(): RaptorConfig {
    return { ...this.config };
  }

  /**
   * Update configuration with validation
   */
  async updateConfig(updates: Partial<RaptorConfig>): Promise<ConfigValidationResult> {
    const newConfig = { ...this.config, ...updates };
    const validation = this.validateConfig(newConfig);
    
    if (!validation.valid) {
      this.emit('config_validation_failed', { validation, updates });
      return validation;
    }

    const oldConfig = { ...this.config };
    this.config = newConfig;
    
    this.emit('config_updated', { 
      old_config: oldConfig, 
      new_config: this.config, 
      validation 
    });

    // Handle rollout percentage changes
    if (updates.rollout?.target_percentage !== undefined && 
        updates.rollout.target_percentage !== this.rolloutState.target_percentage) {
      await this.updateRolloutTarget(updates.rollout.target_percentage);
    }

    return validation;
  }

  /**
   * Validate configuration against policies and safety rules
   */
  validateConfig(config: RaptorConfig): ConfigValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];
    let safetyScore = 1.0;

    // Basic validation
    if (config.rollout_percentage < 0 || config.rollout_percentage > 100) {
      errors.push('rollout_percentage must be between 0 and 100');
    }

    // Performance parameter validation
    if (config.stage_a.alpha < 0 || config.stage_a.alpha > 1) {
      errors.push('stage_a.alpha must be between 0 and 1');
    }
    
    if (config.stage_a.beta < 0 || config.stage_a.beta > 1) {
      errors.push('stage_a.beta must be between 0 and 1');
    }

    if (config.stage_c.max_weight_magnitude > 0.5) {
      warnings.push('stage_c.max_weight_magnitude > 0.5 may cause instability');
      safetyScore *= 0.9;
    }

    // Quality gate validation
    if (config.quality_gates.min_ndcg_improvement < 0.01) {
      warnings.push('Very low min_ndcg_improvement may not justify overhead');
      safetyScore *= 0.95;
    }

    if (config.quality_gates.max_latency_increase_ms > 100) {
      warnings.push('High latency increase may impact user experience');
      safetyScore *= 0.8;
    }

    // Rollout safety checks
    if (config.rollout.target_percentage > 50 && config.rollout.ramp_up_duration_hours < 12) {
      warnings.push('Fast rollout to >50% may be risky');
      safetyScore *= 0.85;
    }

    // Apply policy rules
    for (const policy of this.policies.filter(p => p.enabled)) {
      const policyResult = this.evaluatePolicy(policy, config);
      if (policyResult.action === 'deny') {
        errors.push(`Policy violation: ${policy.name}`);
      } else if (policyResult.action === 'modify') {
        warnings.push(`Policy suggests modification: ${policy.name}`);
        safetyScore *= 0.95;
      }
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings,
      safety_score: safetyScore
    };
  }

  /**
   * Make rollout decision for a specific user/query
   */
  makeRolloutDecision(userId?: string, queryIntent?: string): RolloutDecision {
    // Kill switch check
    if (this.config.kill_switch_active) {
      return {
        should_enable: false,
        reason: 'kill_switch_active',
        feature_flags: this.getDisabledFeatures()
      };
    }

    // Global enable check
    if (!this.config.enabled) {
      return {
        should_enable: false,
        reason: 'globally_disabled',
        feature_flags: this.getDisabledFeatures()
      };
    }

    const currentPercentage = this.rolloutState.current_percentage;
    
    // Percentage-based rollout
    let shouldEnable = false;
    let reason = '';
    let userHash: string | undefined;
    let experimentGroup: 'control' | 'treatment' | undefined;

    if (this.config.rollout.strategy === 'percentage') {
      const hash = userId ? this.hashUserId(userId) : Math.random();
      userHash = userId ? this.hashUserId(userId).toString() : undefined;
      shouldEnable = (hash * 100) < currentPercentage;
      reason = shouldEnable ? 'percentage_rollout_include' : 'percentage_rollout_exclude';
    }

    // A/B testing override
    if (this.config.ab_testing.enabled && shouldEnable) {
      const testHash = this.hashUserId(userId || 'anonymous');
      const totalPercentage = this.config.ab_testing.control_percentage + this.config.ab_testing.treatment_percentage;
      
      if ((testHash * 100) < totalPercentage) {
        const controlThreshold = this.config.ab_testing.control_percentage;
        experimentGroup = ((testHash * totalPercentage) < controlThreshold) ? 'control' : 'treatment';
        shouldEnable = experimentGroup === 'treatment';
        reason = `ab_test_${experimentGroup}`;
      }
    }

    return {
      should_enable: shouldEnable,
      reason,
      user_hash: userHash,
      experiment_group: experimentGroup,
      feature_flags: shouldEnable ? this.config.features : this.getDisabledFeatures()
    };
  }

  /**
   * Emergency kill switch activation
   */
  async activateKillSwitch(reason: string): Promise<void> {
    const oldValue = this.config.kill_switch_active;
    this.config.kill_switch_active = true;
    
    this.emit('kill_switch_activated', { 
      reason, 
      timestamp: new Date(),
      previous_state: oldValue 
    });

    // Log critical event
    console.error(`🚨 RAPTOR KILL SWITCH ACTIVATED: ${reason}`);
  }

  /**
   * Deactivate kill switch
   */
  async deactivateKillSwitch(reason: string): Promise<void> {
    const oldValue = this.config.kill_switch_active;
    this.config.kill_switch_active = false;
    
    this.emit('kill_switch_deactivated', { 
      reason, 
      timestamp: new Date(),
      previous_state: oldValue 
    });

    console.log(`✅ RAPTOR kill switch deactivated: ${reason}`);
  }

  /**
   * Update rollout target with gradual ramp-up
   */
  async updateRolloutTarget(targetPercentage: number): Promise<void> {
    if (targetPercentage < 0 || targetPercentage > 100) {
      throw new Error('Target percentage must be between 0 and 100');
    }

    this.rolloutState.target_percentage = targetPercentage;
    this.rolloutState.ramp_start_time = new Date();
    this.rolloutState.status = 'ramping';
    
    this.emit('rollout_target_updated', {
      target_percentage: targetPercentage,
      current_percentage: this.rolloutState.current_percentage,
      ramp_duration_hours: this.config.rollout.ramp_up_duration_hours
    });

    // Start ramping process
    this.startRamping();
  }

  /**
   * Get current rollout state
   */
  getRolloutState(): RolloutState {
    return { ...this.rolloutState };
  }

  /**
   * Add policy rule
   */
  addPolicy(policy: PolicyRule): void {
    this.policies.push(policy);
    this.policies.sort((a, b) => b.priority - a.priority);
    
    this.emit('policy_added', { policy });
  }

  /**
   * Remove policy rule
   */
  removePolicy(policyId: string): boolean {
    const index = this.policies.findIndex(p => p.id === policyId);
    if (index !== -1) {
      const removed = this.policies.splice(index, 1)[0];
      this.emit('policy_removed', { policy: removed });
      return true;
    }
    return false;
  }

  /**
   * Get all policies
   */
  getPolicies(): PolicyRule[] {
    return [...this.policies];
  }

  private initializeDefaultPolicies(): void {
    // Default safety policies
    this.policies = [
      {
        id: 'max_weight_safety',
        name: 'Maximum Weight Safety',
        condition: 'stage_c.max_weight_magnitude <= 0.5',
        action: 'deny',
        priority: 100,
        enabled: true
      },
      {
        id: 'rollout_speed_limit',
        name: 'Rollout Speed Limit',
        condition: '(rollout.target_percentage > 25) && (rollout.ramp_up_duration_hours < 6)',
        action: 'deny',
        priority: 90,
        enabled: true
      },
      {
        id: 'quality_gate_minimum',
        name: 'Quality Gate Minimum',
        condition: 'quality_gates.min_ndcg_improvement >= 0.01',
        action: 'allow',
        priority: 80,
        enabled: true
      }
    ];
  }

  private evaluatePolicy(policy: PolicyRule, config: RaptorConfig): { action: PolicyRule['action'] } {
    // Simplified policy evaluation - would use JSON Logic in production
    try {
      // This is a mock implementation - would use proper JSON Logic evaluation
      if (policy.condition.includes('stage_c.max_weight_magnitude <= 0.5')) {
        return { action: config.stage_c.max_weight_magnitude <= 0.5 ? 'allow' : 'deny' };
      }
      
      if (policy.condition.includes('rollout.target_percentage > 25') && 
          policy.condition.includes('rollout.ramp_up_duration_hours < 6')) {
        const violatesRolloutSpeed = config.rollout.target_percentage > 25 && 
                                   config.rollout.ramp_up_duration_hours < 6;
        return { action: violatesRolloutSpeed ? 'deny' : 'allow' };
      }
      
      return { action: 'allow' };
    } catch (error) {
      console.warn(`Policy evaluation failed for ${policy.id}:`, error);
      return { action: 'deny' }; // Fail safe
    }
  }

  private startRamping(): void {
    const rampInterval = setInterval(() => {
      if (this.rolloutState.status !== 'ramping') {
        clearInterval(rampInterval);
        return;
      }

      const now = Date.now();
      const rampStart = this.rolloutState.ramp_start_time.getTime();
      const rampDuration = this.config.rollout.ramp_up_duration_hours * 60 * 60 * 1000;
      const elapsed = now - rampStart;
      
      if (elapsed >= rampDuration) {
        // Ramp complete
        this.rolloutState.current_percentage = this.rolloutState.target_percentage;
        this.rolloutState.status = 'stable';
        this.rolloutState.last_update_time = new Date();
        
        this.emit('rollout_complete', {
          final_percentage: this.rolloutState.current_percentage
        });
        
        clearInterval(rampInterval);
      } else {
        // Continue ramping
        const progress = elapsed / rampDuration;
        const startPercentage = this.rolloutState.current_percentage;
        const targetPercentage = this.rolloutState.target_percentage;
        
        this.rolloutState.current_percentage = startPercentage + 
          (targetPercentage - startPercentage) * progress;
        this.rolloutState.last_update_time = new Date();
        
        this.emit('rollout_progress', {
          current_percentage: this.rolloutState.current_percentage,
          progress: progress * 100
        });
      }
    }, 60000); // Check every minute
  }

  private startHealthChecking(): void {
    this.healthCheckInterval = setInterval(() => {
      this.performHealthCheck();
    }, 5 * 60 * 1000); // Every 5 minutes
  }

  private async performHealthCheck(): Promise<void> {
    // Mock health metrics - would integrate with actual monitoring
    const metrics: RolloutHealthMetrics = {
      error_rate: Math.random() * 0.01, // 0-1%
      latency_p95: 50 + Math.random() * 20, // 50-70ms
      memory_usage_mb: 100 + Math.random() * 50, // 100-150MB
      success_rate: 0.98 + Math.random() * 0.02, // 98-100%
      user_satisfaction: 0.9 + Math.random() * 0.1, // 90-100%
      alerts_triggered: []
    };

    // Check against quality gates
    if (metrics.error_rate > 0.05) {
      metrics.alerts_triggered.push('High error rate detected');
    }
    
    if (metrics.latency_p95 > this.config.quality_gates.max_latency_increase_ms + 50) {
      metrics.alerts_triggered.push('Latency threshold exceeded');
    }
    
    if (metrics.success_rate < this.config.quality_gates.min_success_rate) {
      metrics.alerts_triggered.push('Success rate below threshold');
    }

    this.rolloutState.health_metrics = metrics;
    
    // Auto-trigger kill switch if critical issues detected
    if (metrics.alerts_triggered.length > 0 && metrics.error_rate > 0.1) {
      await this.activateKillSwitch(`Health check failed: ${metrics.alerts_triggered.join(', ')}`);
    }

    this.emit('health_check_complete', { metrics });
  }

  private hashUserId(userId: string): number {
    // Simple hash function for consistent user bucketing
    let hash = 0;
    for (let i = 0; i < userId.length; i++) {
      const char = userId.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash) / 2147483647; // Normalize to 0-1
  }

  private getDisabledFeatures(): Record<string, boolean> {
    const disabled: Record<string, boolean> = {};
    for (const key in this.config.features) {
      disabled[key] = false;
    }
    return disabled;
  }

  /**
   * Cleanup resources
   */
  cleanup(): void {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
    }
  }
}