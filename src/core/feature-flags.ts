/**
 * Feature Flag System - Phase B3 Safe Deployment Controls
 * Enables gradual rollout of Stage-C optimizations with A/B testing and rollback capabilities
 * Supports the TODO requirement for gating risky knobs behind flags
 */

import { LensTracer } from '../telemetry/tracer.js';

export interface FeatureFlagConfig {
  // Stage-C B3 Optimizations
  stageCOptimizations: boolean;
  isotonicCalibration: boolean;
  confidenceGating: boolean;
  optimizedHNSW: boolean;
  
  // Phase D Kill-Switch Flags for Canary Deployment
  stageA: {
    native_scanner: boolean;  // Enable native SIMD scanner (NAPI/Neon)
  };
  stageB: {
    enabled: boolean;         // Enable Stage-B symbol/AST optimizations
    lruCaching: boolean;      // Enable LRU caching by bytes
    precompilePatterns: boolean;
  };
  stageC: {
    enabled: boolean;         // Enable Stage-C reranking optimizations
    confidenceCutoff: boolean;
    isotonicCalibration: boolean;
  };
  
  // Canary deployment controls
  canary: {
    trafficPercentage: number; // 5 -> 25 -> 100 progression
    killSwitchEnabled: boolean; // Emergency kill switch
    progressiveRollout: boolean;
  };
  
  // Experimental features
  experimentalReranker: boolean;
  advancedCalibration: boolean;
  hnswAutoTuning: boolean;
  
  // Safety controls
  emergencyDisable: boolean;
  performanceMonitoring: boolean;
  qualityGating: boolean;
  
  // A/B testing
  abTesting: {
    enabled: boolean;
    experimentId: string;
    trafficPercentage: number; // 0-100
    controlGroup: string;
    treatmentGroup: string;
  };
  
  // Rollback controls
  rollbackThresholds: {
    maxLatencyMs: number;        // Auto-rollback if latency exceeds
    minQualityScore: number;     // Auto-rollback if quality drops below
    maxErrorRate: number;        // Auto-rollback if error rate exceeds
  };
}

export interface FeatureFlagOverride {
  flagName: string;
  value: boolean;
  reason: string;
  expiresAt?: Date;
  userId?: string;
  experimentId?: string;
}

export interface FeatureFlagMetrics {
  flagName: string;
  enabled: boolean;
  usageCount: number;
  lastUsed: Date;
  performanceImpact: {
    avgLatencyMs: number;
    errorRate: number;
    qualityScore: number;
  };
  rollbackEvents: number;
}

/**
 * Feature flag manager for safe deployment of B3 optimizations
 * Supports gradual rollout, A/B testing, and automatic rollback
 */
export class FeatureFlagManager {
  private config: FeatureFlagConfig;
  private overrides: Map<string, FeatureFlagOverride> = new Map();
  private metrics: Map<string, FeatureFlagMetrics> = new Map();
  private rollbackHistory: Array<{
    flagName: string;
    reason: string;
    timestamp: Date;
    metrics: any;
  }> = [];

  constructor(config: Partial<FeatureFlagConfig> = {}) {
    this.config = {
      // B3 optimizations - start disabled for safety
      stageCOptimizations: config.stageCOptimizations ?? false,
      isotonicCalibration: config.isotonicCalibration ?? false,
      confidenceGating: config.confidenceGating ?? false,
      optimizedHNSW: config.optimizedHNSW ?? false,
      
      // Phase D Kill-Switch Flags - default to OFF for safe rollout
      stageA: {
        native_scanner: config.stageA?.native_scanner ?? false,
        ...config.stageA
      },
      stageB: {
        enabled: config.stageB?.enabled ?? false,
        lruCaching: config.stageB?.lruCaching ?? false,
        precompilePatterns: config.stageB?.precompilePatterns ?? false,
        ...config.stageB
      },
      stageC: {
        enabled: config.stageC?.enabled ?? false,
        confidenceCutoff: config.stageC?.confidenceCutoff ?? false,
        isotonicCalibration: config.stageC?.isotonicCalibration ?? false,
        ...config.stageC
      },
      
      // Canary deployment controls
      canary: {
        trafficPercentage: config.canary?.trafficPercentage ?? 5, // Start at 5%
        killSwitchEnabled: config.canary?.killSwitchEnabled ?? true,
        progressiveRollout: config.canary?.progressiveRollout ?? true,
        ...config.canary
      },
      
      // Experimental features - disabled by default
      experimentalReranker: config.experimentalReranker ?? false,
      advancedCalibration: config.advancedCalibration ?? false,
      hnswAutoTuning: config.hnswAutoTuning ?? false,
      
      // Safety controls - enabled by default
      emergencyDisable: config.emergencyDisable ?? false,
      performanceMonitoring: config.performanceMonitoring ?? true,
      qualityGating: config.qualityGating ?? true,
      
      // A/B testing configuration
      abTesting: {
        enabled: config.abTesting?.enabled ?? false,
        experimentId: config.abTesting?.experimentId ?? 'stage-c-b3-opt',
        trafficPercentage: config.abTesting?.trafficPercentage ?? 10,
        controlGroup: config.abTesting?.controlGroup ?? 'baseline',
        treatmentGroup: config.abTesting?.treatmentGroup ?? 'b3-optimized',
        ...config.abTesting
      },
      
      // Rollback thresholds per TODO requirements
      rollbackThresholds: {
        maxLatencyMs: config.rollbackThresholds?.maxLatencyMs ?? 12, // 12ms budget
        minQualityScore: config.rollbackThresholds?.minQualityScore ?? 0.95, // 5% quality loss max
        maxErrorRate: config.rollbackThresholds?.maxErrorRate ?? 0.05, // 5% error rate max
        ...config.rollbackThresholds
      }
    };

    // Initialize metrics for all flags
    this.initializeMetrics();

    console.log(`🚩 FeatureFlagManager initialized with ${Object.keys(this.config).length} flags`);
    console.log(`  - A/B Testing: ${this.config.abTesting.enabled} (${this.config.abTesting.trafficPercentage}%)`);
    console.log(`  - Safety Thresholds: latency=${this.config.rollbackThresholds.maxLatencyMs}ms, quality=${this.config.rollbackThresholds.minQualityScore}`);
  }

  /**
   * Check if a feature flag is enabled for the current context
   */
  isEnabled(flagName: keyof FeatureFlagConfig, context?: {
    userId?: string;
    sessionId?: string;
    experimentId?: string;
  }): boolean {
    const span = LensTracer.createChildSpan('feature_flag_check', {
      'flag.name': flagName,
      'context.userId': context?.userId || '',
      'context.sessionId': context?.sessionId || ''
    });

    try {
      // Check for emergency disable
      if (this.config.emergencyDisable) {
        span.setAttributes({ enabled: false, reason: 'emergency_disable' });
        return false;
      }

      // Check for specific overrides first
      const override = this.overrides.get(flagName);
      if (override) {
        // Check if override has expired
        if (override.expiresAt && override.expiresAt < new Date()) {
          this.overrides.delete(flagName);
        } else {
          span.setAttributes({ enabled: override.value, reason: 'override', override_reason: override.reason });
          this.recordFlagUsage(flagName, override.value, 'override');
          return override.value;
        }
      }

      // A/B testing logic
      if (this.config.abTesting.enabled && context?.userId) {
        const abResult = this.determineABTestingGroup(context.userId, flagName);
        if (abResult !== null) {
          span.setAttributes({ enabled: abResult, reason: 'ab_testing' });
          this.recordFlagUsage(flagName, abResult, 'ab_testing');
          return abResult;
        }
      }

      // Default flag value
      const defaultValue = this.getDefaultFlagValue(flagName);
      
      // Apply safety checks
      const safeValue = this.applySafetyChecks(flagName, defaultValue);
      
      span.setAttributes({ enabled: safeValue, reason: 'default', safety_applied: safeValue !== defaultValue });
      this.recordFlagUsage(flagName, safeValue, 'default');
      
      return safeValue;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ enabled: false, reason: 'error' });
      
      // Fail safe - return false on error
      console.error(`Feature flag check failed for ${flagName}:`, error);
      return false;

    } finally {
      span.end();
    }
  }

  /**
   * Set a temporary override for a feature flag
   */
  setOverride(
    flagName: string,
    value: boolean,
    reason: string,
    options?: {
      expiresAt?: Date;
      userId?: string;
      experimentId?: string;
    }
  ): void {
    const override: FeatureFlagOverride = {
      flagName,
      value,
      reason,
      expiresAt: options?.expiresAt,
      userId: options?.userId,
      experimentId: options?.experimentId
    };

    this.overrides.set(flagName, override);
    
    console.log(`🚩 Feature flag override set: ${flagName}=${value} (${reason})`);
    
    // Log override for monitoring
    const span = LensTracer.createChildSpan('feature_flag_override', {
      'flag.name': flagName,
      'flag.value': value,
      'override.reason': reason
    });
    span.end();
  }

  /**
   * Remove an override
   */
  removeOverride(flagName: string): void {
    const removed = this.overrides.delete(flagName);
    if (removed) {
      console.log(`🚩 Feature flag override removed: ${flagName}`);
    }
  }

  /**
   * Record performance metrics for automatic rollback decisions
   */
  recordPerformanceMetrics(
    flagName: string,
    metrics: {
      latencyMs: number;
      errorRate: number;
      qualityScore: number;
    }
  ): void {
    const flagMetrics = this.metrics.get(flagName);
    if (!flagMetrics) return;

    // Update performance metrics with exponential moving average
    const alpha = 0.1;
    flagMetrics.performanceImpact.avgLatencyMs = 
      (1 - alpha) * flagMetrics.performanceImpact.avgLatencyMs + alpha * metrics.latencyMs;
    
    flagMetrics.performanceImpact.errorRate = 
      (1 - alpha) * flagMetrics.performanceImpact.errorRate + alpha * metrics.errorRate;
    
    flagMetrics.performanceImpact.qualityScore = 
      (1 - alpha) * flagMetrics.performanceImpact.qualityScore + alpha * metrics.qualityScore;

    // Check for automatic rollback conditions
    this.checkRollbackConditions(flagName, flagMetrics);
  }

  /**
   * Check if automatic rollback is needed based on performance
   */
  private checkRollbackConditions(flagName: string, flagMetrics: FeatureFlagMetrics): void {
    const thresholds = this.config.rollbackThresholds;
    const perf = flagMetrics.performanceImpact;
    
    let rollbackReason: string | null = null;

    // Check latency threshold
    if (perf.avgLatencyMs > thresholds.maxLatencyMs) {
      rollbackReason = `Latency exceeded threshold: ${perf.avgLatencyMs.toFixed(1)}ms > ${thresholds.maxLatencyMs}ms`;
    }
    
    // Check quality threshold
    else if (perf.qualityScore < thresholds.minQualityScore) {
      rollbackReason = `Quality below threshold: ${perf.qualityScore.toFixed(3)} < ${thresholds.minQualityScore}`;
    }
    
    // Check error rate threshold
    else if (perf.errorRate > thresholds.maxErrorRate) {
      rollbackReason = `Error rate exceeded threshold: ${(perf.errorRate * 100).toFixed(1)}% > ${(thresholds.maxErrorRate * 100).toFixed(1)}%`;
    }

    if (rollbackReason) {
      this.executeAutomaticRollback(flagName, rollbackReason, perf);
    }
  }

  /**
   * Execute automatic rollback
   */
  private executeAutomaticRollback(
    flagName: string,
    reason: string,
    metrics: FeatureFlagMetrics['performanceImpact']
  ): void {
    console.error(`🚨 Automatic rollback triggered for ${flagName}: ${reason}`);
    
    // Disable the flag
    this.setOverride(flagName, false, `Auto-rollback: ${reason}`, {
      expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000) // 24 hours
    });

    // Record rollback event
    this.rollbackHistory.push({
      flagName,
      reason,
      timestamp: new Date(),
      metrics: { ...metrics }
    });

    // Update rollback counter
    const flagMetrics = this.metrics.get(flagName);
    if (flagMetrics) {
      flagMetrics.rollbackEvents++;
    }

    // Create alert span for monitoring
    const span = LensTracer.createChildSpan('automatic_rollback', {
      'flag.name': flagName,
      'rollback.reason': reason,
      'metrics.latency': metrics.avgLatencyMs,
      'metrics.quality': metrics.qualityScore,
      'metrics.error_rate': metrics.errorRate
    });
    span.end();

    // Send alert (in production, this would trigger alerts/notifications)
    console.error(`🚨 ALERT: Feature flag ${flagName} automatically rolled back due to performance degradation`);
  }

  /**
   * Determine A/B testing group assignment
   */
  private determineABTestingGroup(userId: string, flagName: string): boolean | null {
    if (!this.config.abTesting.enabled) return null;

    // Use deterministic hash to assign users consistently
    const hash = this.hashString(userId + flagName + this.config.abTesting.experimentId);
    const hashValue = hash % 100;
    
    // Check if user is in the experiment traffic percentage
    if (hashValue >= this.config.abTesting.trafficPercentage) {
      return null; // Not in experiment
    }

    // Split experiment traffic 50/50 between control and treatment
    const inTreatment = hashValue % 2 === 0;
    
    // For B3 optimizations, treatment group gets the new features
    if (this.isB3OptimizationFlag(flagName)) {
      return inTreatment;
    }

    return null;
  }

  /**
   * Check if a flag is part of B3 optimizations
   */
  private isB3OptimizationFlag(flagName: string): boolean {
    const b3Flags = [
      'stageCOptimizations',
      'isotonicCalibration',
      'confidenceGating',
      'optimizedHNSW',
      'advancedCalibration',
      'hnswAutoTuning'
    ];
    
    return b3Flags.includes(flagName);
  }

  /**
   * Apply safety checks to flag values
   */
  private applySafetyChecks(flagName: string, defaultValue: boolean): boolean {
    // For B3 optimizations, apply extra safety
    if (this.isB3OptimizationFlag(flagName)) {
      const flagMetrics = this.metrics.get(flagName);
      
      // If this flag has caused rollbacks recently, be more conservative
      if (flagMetrics && flagMetrics.rollbackEvents > 2) {
        return false;
      }
      
      // If performance monitoring shows issues, disable
      if (flagMetrics && this.config.performanceMonitoring) {
        const perf = flagMetrics.performanceImpact;
        if (perf.avgLatencyMs > this.config.rollbackThresholds.maxLatencyMs * 0.8) {
          return false;
        }
      }
    }

    return defaultValue;
  }

  /**
   * Get default value for a flag
   */
  private getDefaultFlagValue(flagName: keyof FeatureFlagConfig): boolean {
    const value = this.config[flagName];
    return typeof value === 'boolean' ? value : false;
  }

  /**
   * Record flag usage for metrics
   */
  private recordFlagUsage(flagName: string, enabled: boolean, source: string): void {
    let flagMetrics = this.metrics.get(flagName);
    if (!flagMetrics) {
      flagMetrics = this.createDefaultMetrics(flagName);
      this.metrics.set(flagName, flagMetrics);
    }

    flagMetrics.enabled = enabled;
    flagMetrics.usageCount++;
    flagMetrics.lastUsed = new Date();
  }

  /**
   * Initialize metrics for all flags
   */
  private initializeMetrics(): void {
    const flagNames = Object.keys(this.config);
    for (const flagName of flagNames) {
      if (flagName !== 'abTesting' && flagName !== 'rollbackThresholds') {
        this.metrics.set(flagName, this.createDefaultMetrics(flagName));
      }
    }
  }

  /**
   * Create default metrics object
   */
  private createDefaultMetrics(flagName: string): FeatureFlagMetrics {
    return {
      flagName,
      enabled: false,
      usageCount: 0,
      lastUsed: new Date(),
      performanceImpact: {
        avgLatencyMs: 0,
        errorRate: 0,
        qualityScore: 1.0
      },
      rollbackEvents: 0
    };
  }

  /**
   * Simple string hashing for deterministic A/B testing
   */
  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }

  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<FeatureFlagConfig>): void {
    this.config = { ...this.config, ...newConfig };
    console.log(`🚩 Feature flag config updated`);
  }

  /**
   * Get current feature flag status and metrics
   */
  getStatus() {
    return {
      config: this.config,
      overrides: Object.fromEntries(this.overrides),
      metrics: Object.fromEntries(this.metrics),
      rollbackHistory: this.rollbackHistory.slice(-10), // Last 10 rollbacks
      activeExperiment: this.config.abTesting.enabled ? {
        experimentId: this.config.abTesting.experimentId,
        trafficPercentage: this.config.abTesting.trafficPercentage,
        controlGroup: this.config.abTesting.controlGroup,
        treatmentGroup: this.config.abTesting.treatmentGroup
      } : null
    };
  }

  /**
   * Phase D Canary Deployment Management
   */
  
  /**
   * Check if user should be in canary group based on traffic percentage
   */
  isInCanaryGroup(userId: string = 'anonymous'): boolean {
    if (!this.config.canary.progressiveRollout) {
      return false;
    }
    
    const hash = this.hashString(userId);
    const bucket = hash % 100; // 0-99
    return bucket < this.config.canary.trafficPercentage;
  }

  /**
   * Progress canary rollout: 5% -> 25% -> 100%
   */
  progressCanaryRollout(): { success: boolean; newPercentage: number; stage: string } {
    const current = this.config.canary.trafficPercentage;
    let newPercentage: number;
    let stage: string;

    if (current === 5) {
      newPercentage = 25;
      stage = 'medium';
    } else if (current === 25) {
      newPercentage = 100;
      stage = 'full';
    } else {
      return { success: false, newPercentage: current, stage: 'unknown' };
    }

    this.config.canary.trafficPercentage = newPercentage;
    
    console.log(`🚀 Canary rollout progressed: ${current}% -> ${newPercentage}% (${stage})`);
    
    return { success: true, newPercentage, stage };
  }

  /**
   * Kill switch - immediately rollback to 0% traffic
   */
  killSwitchActivate(reason: string): void {
    console.error(`🚨 KILL SWITCH ACTIVATED: ${reason}`);
    
    // Disable all stage flags immediately
    this.config.stageA.native_scanner = false;
    this.config.stageB.enabled = false;
    this.config.stageC.enabled = false;
    
    // Set canary traffic to 0
    this.config.canary.trafficPercentage = 0;
    
    // Record rollback
    this.rollbackHistory.push({
      flagName: 'canary_kill_switch',
      reason,
      timestamp: new Date(),
      metrics: { trafficPercentage: this.config.canary.trafficPercentage }
    });

    const span = LensTracer.createChildSpan('kill_switch_activated', {
      'killswitch.reason': reason,
      'rollback.traffic_percentage': 0
    });
    span.end();
  }

  /**
   * Get canary deployment status
   */
  getCanaryStatus() {
    return {
      trafficPercentage: this.config.canary.trafficPercentage,
      killSwitchEnabled: this.config.canary.killSwitchEnabled,
      progressiveRollout: this.config.canary.progressiveRollout,
      stageFlags: {
        stageA_native_scanner: this.config.stageA.native_scanner,
        stageB_enabled: this.config.stageB.enabled,
        stageC_enabled: this.config.stageC.enabled,
      },
      nextStage: this.config.canary.trafficPercentage === 5 ? '25%' : 
                 this.config.canary.trafficPercentage === 25 ? '100%' : 'complete',
      rollbackHistory: this.rollbackHistory.slice(-5)
    };
  }

  /**
   * Emergency disable all experimental features
   */
  emergencyDisableAll(reason: string): void {
    console.error(`🚨 EMERGENCY DISABLE: ${reason}`);
    
    const experimentalFlags = [
      'stageCOptimizations',
      'isotonicCalibration',
      'confidenceGating',
      'optimizedHNSW',
      'experimentalReranker',
      'advancedCalibration',
      'hnswAutoTuning'
    ];

    for (const flagName of experimentalFlags) {
      this.setOverride(flagName, false, `Emergency disable: ${reason}`, {
        expiresAt: new Date(Date.now() + 48 * 60 * 60 * 1000) // 48 hours
      });
    }

    // Set emergency flag
    this.config.emergencyDisable = true;

    // Create emergency alert
    const span = LensTracer.createChildSpan('emergency_disable_all', {
      'emergency.reason': reason,
      'flags_disabled': experimentalFlags.length
    });
    span.end();
  }

  /**
   * Clear emergency disable state
   */
  clearEmergencyDisable(): void {
    this.config.emergencyDisable = false;
    console.log('🚩 Emergency disable cleared');
  }
}

/**
 * Global feature flag manager instance
 */
export const globalFeatureFlags = new FeatureFlagManager();

/**
 * Convenience function to check feature flags
 */
export function isFeatureEnabled(
  flagName: keyof FeatureFlagConfig,
  context?: { userId?: string; sessionId?: string; experimentId?: string }
): boolean {
  return globalFeatureFlags.isEnabled(flagName, context);
}

/**
 * Convenience function to record performance metrics
 */
export function recordFeaturePerformance(
  flagName: string,
  metrics: { latencyMs: number; errorRate: number; qualityScore: number }
): void {
  globalFeatureFlags.recordPerformanceMetrics(flagName, metrics);
}