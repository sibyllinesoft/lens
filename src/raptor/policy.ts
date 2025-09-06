/**
 * RAPTOR Policy Management System
 * 
 * Manages configuration, feature flags, and operational policies for the
 * RAPTOR semantic search system with rollback capabilities.
 */

import { PressureWeights, ReclusterBudget } from './snapshot.js';
import { FeatureComputationConfig } from './runtime-features.js';
import EventEmitter from 'events';

export interface RaptorPolicy {
  // Core feature toggles
  enabled: boolean;
  prior_boost_enabled: boolean;
  semantic_cards_enabled: boolean;
  recluster_daemon_enabled: boolean;

  // Runtime configuration
  prior_boost_cap: number;        // Maximum log-odds boost (0.5)
  topic_threshold: number;        // Minimum similarity for boost (0.35)
  depth_max: number;             // Maximum RAPTOR tree depth (3)
  
  // Reclustering parameters
  pressure_weights: PressureWeights;
  ttl_days: number;              // Node TTL in days (14)
  hourly_budget: ReclusterBudget;
  hysteresis: number;            // Centroid drift threshold (0.05)

  // Safety limits
  max_snapshots: number;         // Maximum snapshots to keep (10)
  max_cards_per_file: number;    // Maximum semantic cards per file (3)
  max_embedding_batch: number;   // Maximum embeddings per batch (100)
  
  // Monitoring thresholds
  staleness_alert_threshold: number;    // Alert if >N% nodes stale (20)
  pressure_alert_threshold: number;     // Alert if backlog >N operations (1000)
  error_rate_alert_threshold: number;   // Alert if error rate >N% (5)
}

export interface PolicyValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

export interface PolicyHistory {
  timestamp: number;
  user: string;
  action: 'update' | 'rollback' | 'reset';
  changes: Partial<RaptorPolicy>;
  reason?: string;
}

export interface PolicySnapshot {
  id: string;
  timestamp: number;
  policy: RaptorPolicy;
  metadata: {
    user: string;
    reason: string;
    auto_generated: boolean;
  };
}

export interface RollbackOptions {
  target_id?: string;           // Specific snapshot to rollback to
  target_timestamp?: number;    // Rollback to time
  partial_rollback?: string[];  // Only rollback specific fields
  reason: string;
}

/**
 * Policy manager with validation and rollback capabilities
 */
export class RaptorPolicyManager extends EventEmitter {
  private currentPolicy: RaptorPolicy;
  private history: PolicyHistory[];
  private snapshots: Map<string, PolicySnapshot>;
  private maxHistorySize: number;
  private autoSnapshotInterval: number; // Minutes between auto-snapshots

  constructor(initialPolicy?: RaptorPolicy) {
    super();
    this.currentPolicy = initialPolicy || RaptorPolicyManager.createDefaultPolicy();
    this.history = [];
    this.snapshots = new Map();
    this.maxHistorySize = 1000;
    this.autoSnapshotInterval = 60; // 1 hour

    // Create initial snapshot
    this.createSnapshot('system', 'Initial policy state', true);
  }

  static createDefaultPolicy(): RaptorPolicy {
    return {
      enabled: false,                    // Start disabled for safety
      prior_boost_enabled: false,
      semantic_cards_enabled: false,
      recluster_daemon_enabled: false,

      prior_boost_cap: 0.5,
      topic_threshold: 0.35,
      depth_max: 3,

      pressure_weights: { wc: 0.4, wd: 0.3, wq: 0.2, wa: 0.1 },
      ttl_days: 14,
      hourly_budget: {
        max_summaries_per_hour: 200,
        max_cpu_seconds_per_hour: 300,
        current_summaries_used: 0,
        current_cpu_used: 0,
        reset_ts: Date.now()
      },
      hysteresis: 0.05,

      max_snapshots: 10,
      max_cards_per_file: 3,
      max_embedding_batch: 100,

      staleness_alert_threshold: 20,
      pressure_alert_threshold: 1000,
      error_rate_alert_threshold: 5
    };
  }

  getCurrentPolicy(): RaptorPolicy {
    return { ...this.currentPolicy };
  }

  validatePolicy(policy: Partial<RaptorPolicy>): PolicyValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Validate numeric ranges
    if (policy.prior_boost_cap !== undefined) {
      if (policy.prior_boost_cap < 0 || policy.prior_boost_cap > 2) {
        errors.push('prior_boost_cap must be between 0 and 2');
      } else if (policy.prior_boost_cap > 1) {
        warnings.push('prior_boost_cap > 1 may cause ranking instability');
      }
    }

    if (policy.topic_threshold !== undefined) {
      if (policy.topic_threshold < 0 || policy.topic_threshold > 1) {
        errors.push('topic_threshold must be between 0 and 1');
      }
    }

    if (policy.depth_max !== undefined) {
      if (policy.depth_max < 1 || policy.depth_max > 5) {
        errors.push('depth_max must be between 1 and 5');
      } else if (policy.depth_max > 3) {
        warnings.push('depth_max > 3 may cause performance degradation');
      }
    }

    if (policy.ttl_days !== undefined) {
      if (policy.ttl_days < 1 || policy.ttl_days > 90) {
        errors.push('ttl_days must be between 1 and 90');
      } else if (policy.ttl_days < 7) {
        warnings.push('ttl_days < 7 may cause excessive reclustering');
      }
    }

    // Validate pressure weights sum to ~1.0
    if (policy.pressure_weights !== undefined) {
      const weights = policy.pressure_weights;
      const sum = weights.wc + weights.wd + weights.wq + weights.wa;
      if (Math.abs(sum - 1.0) > 0.1) {
        warnings.push(`Pressure weights sum to ${sum.toFixed(2)}, should be close to 1.0`);
      }

      for (const [key, value] of Object.entries(weights)) {
        if (value < 0 || value > 1) {
          errors.push(`Pressure weight ${key} must be between 0 and 1`);
        }
      }
    }

    // Validate budget limits
    if (policy.hourly_budget !== undefined) {
      const budget = policy.hourly_budget;
      if (budget.max_summaries_per_hour < 0 || budget.max_summaries_per_hour > 10000) {
        errors.push('max_summaries_per_hour must be between 0 and 10000');
      }
      if (budget.max_cpu_seconds_per_hour < 0 || budget.max_cpu_seconds_per_hour > 3600) {
        errors.push('max_cpu_seconds_per_hour must be between 0 and 3600');
      }
    }

    // Validate safety limits
    if (policy.max_snapshots !== undefined) {
      if (policy.max_snapshots < 1 || policy.max_snapshots > 100) {
        errors.push('max_snapshots must be between 1 and 100');
      }
    }

    // Validate alert thresholds
    if (policy.staleness_alert_threshold !== undefined) {
      if (policy.staleness_alert_threshold < 0 || policy.staleness_alert_threshold > 100) {
        errors.push('staleness_alert_threshold must be between 0 and 100');
      }
    }

    // Check for dangerous combinations
    if (policy.enabled && policy.recluster_daemon_enabled && policy.ttl_days && policy.ttl_days < 3) {
      warnings.push('Enabled RAPTOR with short TTL may cause high CPU usage');
    }

    return { valid: errors.length === 0, errors, warnings };
  }

  updatePolicy(
    changes: Partial<RaptorPolicy>,
    user: string,
    reason?: string
  ): PolicyValidationResult {
    // Validate changes
    const validation = this.validatePolicy(changes);
    if (!validation.valid) {
      this.emit('policy-update-failed', { changes, validation, user });
      return validation;
    }

    // Create policy history entry
    const historyEntry: PolicyHistory = {
      timestamp: Date.now(),
      user,
      action: 'update',
      changes,
      reason
    };

    // Apply changes
    const oldPolicy = { ...this.currentPolicy };
    this.currentPolicy = { ...this.currentPolicy, ...changes };

    // Add to history
    this.history.push(historyEntry);
    this.trimHistory();

    // Auto-create snapshot if significant change
    if (this.isSignificantChange(changes)) {
      this.createSnapshot(user, reason || 'Significant policy update', true);
    }

    // Emit events
    this.emit('policy-updated', {
      oldPolicy,
      newPolicy: this.currentPolicy,
      changes,
      user,
      validation
    });

    if (validation.warnings.length > 0) {
      this.emit('policy-warnings', { warnings: validation.warnings, user });
    }

    return validation;
  }

  createSnapshot(
    user: string,
    reason: string,
    autoGenerated: boolean = false
  ): string {
    const id = `snapshot-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    const snapshot: PolicySnapshot = {
      id,
      timestamp: Date.now(),
      policy: { ...this.currentPolicy },
      metadata: { user, reason, auto_generated: autoGenerated }
    };

    this.snapshots.set(id, snapshot);
    this.trimSnapshots();

    this.emit('snapshot-created', { snapshot });
    return id;
  }

  rollbackPolicy(options: RollbackOptions, user: string): PolicyValidationResult {
    let targetPolicy: RaptorPolicy;

    // Find target policy
    if (options.target_id) {
      const snapshot = this.snapshots.get(options.target_id);
      if (!snapshot) {
        return {
          valid: false,
          errors: [`Snapshot ${options.target_id} not found`],
          warnings: []
        };
      }
      targetPolicy = snapshot.policy;
    } else if (options.target_timestamp) {
      // Find closest snapshot before timestamp
      const snapshots = Array.from(this.snapshots.values())
        .filter(s => s.timestamp <= options.target_timestamp!)
        .sort((a, b) => b.timestamp - a.timestamp);
      
      if (snapshots.length === 0) {
        return {
          valid: false,
          errors: ['No snapshots found before target timestamp'],
          warnings: []
        };
      }
      
      targetPolicy = snapshots[0].policy;
    } else {
      return {
        valid: false,
        errors: ['Must specify target_id or target_timestamp'],
        warnings: []
      };
    }

    // Apply partial rollback if specified
    let rollbackPolicy = targetPolicy;
    if (options.partial_rollback && options.partial_rollback.length > 0) {
      rollbackPolicy = { ...this.currentPolicy };
      for (const field of options.partial_rollback) {
        if (field in targetPolicy) {
          (rollbackPolicy as any)[field] = (targetPolicy as any)[field];
        }
      }
    }

    // Validate rollback policy
    const validation = this.validatePolicy(rollbackPolicy);
    if (!validation.valid) {
      this.emit('policy-rollback-failed', { options, validation, user });
      return validation;
    }

    // Create history entry
    const historyEntry: PolicyHistory = {
      timestamp: Date.now(),
      user,
      action: 'rollback',
      changes: this.computePolicyDiff(this.currentPolicy, rollbackPolicy),
      reason: options.reason
    };

    // Apply rollback
    const oldPolicy = { ...this.currentPolicy };
    this.currentPolicy = rollbackPolicy;

    // Add to history
    this.history.push(historyEntry);
    this.trimHistory();

    // Create rollback snapshot
    this.createSnapshot(user, `Rollback: ${options.reason}`, true);

    // Emit events
    this.emit('policy-rolled-back', {
      oldPolicy,
      newPolicy: this.currentPolicy,
      options,
      user,
      validation
    });

    return validation;
  }

  resetToDefault(user: string, reason: string): PolicyValidationResult {
    const defaultPolicy = RaptorPolicyManager.createDefaultPolicy();
    const changes = this.computePolicyDiff(this.currentPolicy, defaultPolicy);

    const historyEntry: PolicyHistory = {
      timestamp: Date.now(),
      user,
      action: 'reset',
      changes,
      reason
    };

    const oldPolicy = { ...this.currentPolicy };
    this.currentPolicy = defaultPolicy;

    this.history.push(historyEntry);
    this.trimHistory();

    this.createSnapshot(user, `Reset to defaults: ${reason}`, true);

    this.emit('policy-reset', { oldPolicy, newPolicy: this.currentPolicy, user });

    return { valid: true, errors: [], warnings: [] };
  }

  getHistory(limit?: number): PolicyHistory[] {
    const history = [...this.history].reverse(); // Most recent first
    return limit ? history.slice(0, limit) : history;
  }

  getSnapshots(): PolicySnapshot[] {
    return Array.from(this.snapshots.values())
      .sort((a, b) => b.timestamp - a.timestamp);
  }

  getSnapshot(id: string): PolicySnapshot | undefined {
    return this.snapshots.get(id);
  }

  // Utility methods for feature flag management
  enableRaptorFeatures(user: string, reason: string = 'Enabling RAPTOR features'): PolicyValidationResult {
    return this.updatePolicy({
      enabled: true,
      semantic_cards_enabled: true
    }, user, reason);
  }

  enablePathPriorBoost(user: string, reason: string = 'Enabling path prior boost'): PolicyValidationResult {
    return this.updatePolicy({
      prior_boost_enabled: true
    }, user, reason);
  }

  enableReclusterDaemon(user: string, reason: string = 'Enabling reclustering daemon'): PolicyValidationResult {
    return this.updatePolicy({
      recluster_daemon_enabled: true
    }, user, reason);
  }

  emergencyDisable(user: string, reason: string): PolicyValidationResult {
    return this.updatePolicy({
      enabled: false,
      prior_boost_enabled: false,
      recluster_daemon_enabled: false
    }, user, `EMERGENCY DISABLE: ${reason}`);
  }

  private isSignificantChange(changes: Partial<RaptorPolicy>): boolean {
    const significantFields = [
      'enabled', 'prior_boost_enabled', 'semantic_cards_enabled', 
      'recluster_daemon_enabled', 'prior_boost_cap', 'topic_threshold'
    ];
    
    return significantFields.some(field => field in changes);
  }

  private computePolicyDiff(oldPolicy: RaptorPolicy, newPolicy: RaptorPolicy): Partial<RaptorPolicy> {
    const diff: Partial<RaptorPolicy> = {};
    
    for (const [key, value] of Object.entries(newPolicy)) {
      if (JSON.stringify((oldPolicy as any)[key]) !== JSON.stringify(value)) {
        (diff as any)[key] = value;
      }
    }
    
    return diff;
  }

  private trimHistory(): void {
    if (this.history.length > this.maxHistorySize) {
      this.history = this.history.slice(-this.maxHistorySize);
    }
  }

  private trimSnapshots(): void {
    const maxSnapshots = this.currentPolicy.max_snapshots;
    if (this.snapshots.size > maxSnapshots) {
      const snapshots = Array.from(this.snapshots.values())
        .sort((a, b) => a.timestamp - b.timestamp);
      
      // Remove oldest snapshots, but keep at least one
      const toRemove = snapshots.slice(0, this.snapshots.size - maxSnapshots);
      for (const snapshot of toRemove) {
        this.snapshots.delete(snapshot.id);
      }
    }
  }

  // Monitoring helpers
  checkPolicyHealth(): { healthy: boolean; issues: string[] } {
    const issues: string[] = [];
    const policy = this.currentPolicy;

    // Check for risky configurations
    if (policy.enabled && policy.prior_boost_cap > 1) {
      issues.push('High path prior boost cap may destabilize rankings');
    }

    if (policy.recluster_daemon_enabled && policy.ttl_days < 3) {
      issues.push('Short TTL with daemon enabled may cause high CPU usage');
    }

    if (policy.pressure_weights.wc + policy.pressure_weights.wd + policy.pressure_weights.wq + policy.pressure_weights.wa < 0.9) {
      issues.push('Pressure weights sum is significantly less than 1.0');
    }

    if (policy.hourly_budget.max_summaries_per_hour < 50) {
      issues.push('Very low summary budget may prevent necessary updates');
    }

    return {
      healthy: issues.length === 0,
      issues
    };
  }

  // Export/import for backup
  exportPolicy(): { policy: RaptorPolicy; history: PolicyHistory[]; snapshots: PolicySnapshot[] } {
    return {
      policy: { ...this.currentPolicy },
      history: [...this.history],
      snapshots: Array.from(this.snapshots.values())
    };
  }

  importPolicy(
    data: { policy: RaptorPolicy; history?: PolicyHistory[]; snapshots?: PolicySnapshot[] },
    user: string
  ): PolicyValidationResult {
    const validation = this.validatePolicy(data.policy);
    if (!validation.valid) {
      return validation;
    }

    this.currentPolicy = { ...data.policy };
    
    if (data.history) {
      this.history = [...data.history];
    }
    
    if (data.snapshots) {
      this.snapshots.clear();
      for (const snapshot of data.snapshots) {
        this.snapshots.set(snapshot.id, snapshot);
      }
    }

    this.emit('policy-imported', { user, validation });
    return validation;
  }
}

export default RaptorPolicyManager;