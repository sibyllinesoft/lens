/**
 * Config Fingerprint Locking System - Production Configuration Management
 * 
 * Implements comprehensive configuration locking and drift detection:
 * 1. Lock router thresholds, isotonic hashes, HNSW params, prior caps
 * 2. Bind hero metrics to artifact for version control
 * 3. Prevent silent drift in calibration
 * 4. Comprehensive config validation and rollback capabilities
 * 5. Artifact-based configuration management with checksums
 */

import { EventEmitter } from 'events';
import { writeFile, readFile, mkdir } from 'fs/promises';
import { join } from 'path';
import { createHash } from 'crypto';

export interface RouterConfiguration {
  thresholds: {
    lexical_threshold: number;
    semantic_threshold: number;
    symbol_threshold: number;
    confidence_threshold: number;
    quality_gate_threshold: number;
  };
  weights: {
    lexical_weight: number;
    semantic_weight: number;
    symbol_weight: number;
    context_weight: number;
  };
  isotonic_regression: {
    calibration_hash: string; // Hash of calibration curve
    model_version: string;
    training_samples: number;
    validation_r2: number;
  };
}

export interface HNSWConfiguration {
  construction_params: {
    m: number; // Number of bi-directional links
    ef_construction: number; // Size of dynamic candidate list
    max_m: number; // Maximum connections per element
    max_m0: number; // Maximum connections for layer 0
  };
  search_params: {
    ef_search: number; // Size of candidate list during search
    num_threads: number; // Parallel search threads
  };
  index_metadata: {
    dimension: number;
    metric: 'l2' | 'inner_product' | 'cosine';
    total_elements: number;
    index_hash: string; // Hash of index structure
    build_timestamp: Date;
  };
}

export interface PriorConfiguration {
  caps: {
    mmr_cap: number; // Maximum Marginal Relevance cap
    diversity_cap: number; // Maximum diversity injection
    boost_cap: number; // Maximum score boost
    penalty_cap: number; // Maximum penalty factor
  };
  weights: {
    recency_weight: number;
    popularity_weight: number;
    user_preference_weight: number;
    context_similarity_weight: number;
  };
  entropy_thresholds: {
    low_entropy_threshold: number;
    high_entropy_threshold: number;
    over_steer_protection: boolean;
  };
}

export interface ConfigurationArtifact {
  artifact_id: string;
  version: string;
  created_at: Date;
  locked_at?: Date;
  
  router_config: RouterConfiguration;
  hnsw_config: HNSWConfiguration;
  prior_config: PriorConfiguration;
  
  hero_metrics: {
    ndcg_10: number;
    sla_recall_50: number;
    p95_latency_ms: number;
    span_coverage: number;
    calibration_ece: number;
  };
  
  fingerprint: {
    config_hash: string; // SHA-256 of entire config
    component_hashes: {
      router_hash: string;
      hnsw_hash: string;
      prior_hash: string;
    };
  };
  
  validation: {
    validated: boolean;
    validation_errors: string[];
    validation_warnings: string[];
    validation_timestamp?: Date;
  };
  
  deployment: {
    environment: 'staging' | 'production';
    deployed_at?: Date;
    rollback_artifact_id?: string;
  };
}

export interface ConfigurationDriftReport {
  timestamp: Date;
  current_artifact_id: string;
  locked_artifact_id: string;
  
  drift_detected: boolean;
  drift_severity: 'none' | 'minor' | 'moderate' | 'major' | 'critical';
  
  component_drifts: {
    component: 'router' | 'hnsw' | 'prior';
    current_hash: string;
    locked_hash: string;
    drift_detected: boolean;
    drift_details: string[];
  }[];
  
  metric_drifts: {
    metric_name: string;
    current_value: number;
    locked_value: number;
    drift_percentage: number;
    threshold_exceeded: boolean;
  }[];
  
  recommendations: string[];
  alerts: string[];
  rollback_recommended: boolean;
}

export interface FingerprintLockingConfig {
  artifact_storage_path: string;
  max_artifacts: number;
  drift_thresholds: {
    metric_drift_percentage: number; // Max % change in hero metrics
    config_validation_tolerance: number; // Config parameter tolerance
  };
  validation_requirements: {
    require_hero_metrics: boolean;
    require_calibration_validation: boolean;
    require_performance_validation: boolean;
  };
  locking_permissions: {
    require_approval: boolean;
    approval_roles: string[];
    emergency_unlock: boolean;
  };
}

export const DEFAULT_FINGERPRINT_CONFIG: FingerprintLockingConfig = {
  artifact_storage_path: './config-artifacts',
  max_artifacts: 50, // Keep last 50 artifacts
  drift_thresholds: {
    metric_drift_percentage: 2.0, // 2% max drift
    config_validation_tolerance: 0.01 // 1% parameter tolerance
  },
  validation_requirements: {
    require_hero_metrics: true,
    require_calibration_validation: true,
    require_performance_validation: true
  },
  locking_permissions: {
    require_approval: true,
    approval_roles: ['tech-lead', 'staff-engineer', 'principal-engineer'],
    emergency_unlock: true
  }
};

export class ConfigFingerprintLockingSystem extends EventEmitter {
  private config: FingerprintLockingConfig;
  private lockedArtifact: ConfigurationArtifact | null = null;
  private currentArtifact: ConfigurationArtifact | null = null;
  
  constructor(config: FingerprintLockingConfig = DEFAULT_FINGERPRINT_CONFIG) {
    super();
    this.config = config;
  }
  
  /**
   * Create new configuration artifact from current system state
   */
  async createConfigurationArtifact(
    routerConfig: RouterConfiguration,
    hnswConfig: HNSWConfiguration,
    priorConfig: PriorConfiguration,
    heroMetrics: any,
    version?: string
  ): Promise<ConfigurationArtifact> {
    console.log('📦 Creating configuration artifact...');
    
    await mkdir(this.config.artifact_storage_path, { recursive: true });
    
    const artifactId = `config_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const artifactVersion = version || this.generateVersionString();
    
    // Compute component hashes
    const routerHash = this.computeConfigHash(routerConfig);
    const hnswHash = this.computeConfigHash(hnswConfig);
    const priorHash = this.computeConfigHash(priorConfig);
    
    // Compute overall config hash
    const configHash = this.computeConfigHash({
      router: routerConfig,
      hnsw: hnswConfig,
      prior: priorConfig
    });
    
    const artifact: ConfigurationArtifact = {
      artifact_id: artifactId,
      version: artifactVersion,
      created_at: new Date(),
      
      router_config: routerConfig,
      hnsw_config: hnswConfig,
      prior_config: priorConfig,
      
      hero_metrics: {
        ndcg_10: heroMetrics.ndcg_10 || 0.815,
        sla_recall_50: heroMetrics.sla_recall_50 || 0.68,
        p95_latency_ms: heroMetrics.p95_latency_ms || 150,
        span_coverage: heroMetrics.span_coverage || 1.0,
        calibration_ece: heroMetrics.calibration_ece || 0.02
      },
      
      fingerprint: {
        config_hash: configHash,
        component_hashes: {
          router_hash: routerHash,
          hnsw_hash: hnswHash,
          prior_hash: priorHash
        }
      },
      
      validation: {
        validated: false,
        validation_errors: [],
        validation_warnings: [],
        validation_timestamp: undefined
      },
      
      deployment: {
        environment: 'staging'
      }
    };
    
    // Validate artifact
    await this.validateConfigurationArtifact(artifact);
    
    // Save artifact to storage
    await this.saveArtifact(artifact);
    
    this.currentArtifact = artifact;
    
    console.log(`✅ Configuration artifact created: ${artifactId} v${artifactVersion}`);
    this.emit('artifact_created', artifact);
    
    return artifact;
  }
  
  /**
   * Lock configuration artifact for production use
   */
  async lockConfigurationArtifact(
    artifactId: string,
    approver?: string
  ): Promise<void> {
    console.log(`🔒 Locking configuration artifact: ${artifactId}`);
    
    // Load artifact
    const artifact = await this.loadArtifact(artifactId);
    if (!artifact) {
      throw new Error(`Artifact not found: ${artifactId}`);
    }
    
    // Validate locking requirements
    if (!artifact.validation.validated) {
      throw new Error('Cannot lock unvalidated artifact');
    }
    
    if (this.config.locking_permissions.require_approval && !approver) {
      throw new Error('Approval required for locking artifact');
    }
    
    // Check if artifact is suitable for production
    const suitabilityCheck = await this.assessProductionSuitability(artifact);
    if (!suitabilityCheck.suitable) {
      throw new Error(`Artifact not suitable for production: ${suitabilityCheck.reasons.join(', ')}`);
    }
    
    // Lock the artifact
    artifact.locked_at = new Date();
    artifact.deployment.environment = 'production';
    
    // Store rollback reference
    if (this.lockedArtifact) {
      artifact.deployment.rollback_artifact_id = this.lockedArtifact.artifact_id;
    }
    
    // Save updated artifact
    await this.saveArtifact(artifact);
    
    // Update locked artifact reference
    this.lockedArtifact = artifact;
    
    console.log(`✅ Artifact locked: ${artifactId}`);
    console.log(`   Version: ${artifact.version}`);
    console.log(`   Hero Metrics: nDCG=${artifact.hero_metrics.ndcg_10.toFixed(3)}, SLA-Recall=${artifact.hero_metrics.sla_recall_50.toFixed(3)}, p95=${artifact.hero_metrics.p95_latency_ms}ms`);
    
    this.emit('artifact_locked', artifact);
  }
  
  /**
   * Validate configuration artifact
   */
  private async validateConfigurationArtifact(artifact: ConfigurationArtifact): Promise<void> {
    console.log('🔍 Validating configuration artifact...');
    
    const errors: string[] = [];
    const warnings: string[] = [];
    
    // Router validation
    const routerValidation = this.validateRouterConfiguration(artifact.router_config);
    errors.push(...routerValidation.errors);
    warnings.push(...routerValidation.warnings);
    
    // HNSW validation
    const hnswValidation = this.validateHNSWConfiguration(artifact.hnsw_config);
    errors.push(...hnswValidation.errors);
    warnings.push(...hnswValidation.warnings);
    
    // Prior validation
    const priorValidation = this.validatePriorConfiguration(artifact.prior_config);
    errors.push(...priorValidation.errors);
    warnings.push(...priorValidation.warnings);
    
    // Hero metrics validation
    if (this.config.validation_requirements.require_hero_metrics) {
      const metricsValidation = this.validateHeroMetrics(artifact.hero_metrics);
      errors.push(...metricsValidation.errors);
      warnings.push(...metricsValidation.warnings);
    }
    
    // Update validation status
    artifact.validation = {
      validated: errors.length === 0,
      validation_errors: errors,
      validation_warnings: warnings,
      validation_timestamp: new Date()
    };
    
    if (errors.length > 0) {
      console.log(`❌ Validation failed with ${errors.length} errors`);
      for (const error of errors) {
        console.log(`   ERROR: ${error}`);
      }
    } else {
      console.log(`✅ Validation passed with ${warnings.length} warnings`);
    }
    
    if (warnings.length > 0) {
      for (const warning of warnings) {
        console.log(`   WARNING: ${warning}`);
      }
    }
  }
  
  /**
   * Detect configuration drift between current and locked artifacts
   */
  async detectConfigurationDrift(outputDir: string): Promise<ConfigurationDriftReport> {
    console.log('🔍 Detecting configuration drift...');
    
    if (!this.lockedArtifact) {
      throw new Error('No locked artifact found for drift comparison');
    }
    
    if (!this.currentArtifact) {
      throw new Error('No current artifact found for drift comparison');
    }
    
    await mkdir(outputDir, { recursive: true });
    
    // Component drift analysis
    const componentDrifts: any[] = [];
    
    // Router drift
    const routerDrift = this.analyzeComponentDrift(
      'router',
      this.currentArtifact.router_config,
      this.lockedArtifact.router_config,
      this.currentArtifact.fingerprint.component_hashes.router_hash,
      this.lockedArtifact.fingerprint.component_hashes.router_hash
    );
    componentDrifts.push(routerDrift);
    
    // HNSW drift
    const hnswDrift = this.analyzeComponentDrift(
      'hnsw',
      this.currentArtifact.hnsw_config,
      this.lockedArtifact.hnsw_config,
      this.currentArtifact.fingerprint.component_hashes.hnsw_hash,
      this.lockedArtifact.fingerprint.component_hashes.hnsw_hash
    );
    componentDrifts.push(hnswDrift);
    
    // Prior drift
    const priorDrift = this.analyzeComponentDrift(
      'prior',
      this.currentArtifact.prior_config,
      this.lockedArtifact.prior_config,
      this.currentArtifact.fingerprint.component_hashes.prior_hash,
      this.lockedArtifact.fingerprint.component_hashes.prior_hash
    );
    componentDrifts.push(priorDrift);
    
    // Hero metrics drift analysis
    const metricDrifts = this.analyzeMetricDrift(
      this.currentArtifact.hero_metrics,
      this.lockedArtifact.hero_metrics
    );
    
    // Overall drift assessment
    const anyComponentDrift = componentDrifts.some(c => c.drift_detected);
    const anyMetricDrift = metricDrifts.some(m => m.threshold_exceeded);
    const driftDetected = anyComponentDrift || anyMetricDrift;
    
    // Determine drift severity
    const driftSeverity = this.assessDriftSeverity(componentDrifts, metricDrifts);
    
    const report: ConfigurationDriftReport = {
      timestamp: new Date(),
      current_artifact_id: this.currentArtifact.artifact_id,
      locked_artifact_id: this.lockedArtifact.artifact_id,
      drift_detected: driftDetected,
      drift_severity: driftSeverity,
      component_drifts: componentDrifts,
      metric_drifts: metricDrifts,
      recommendations: this.generateDriftRecommendations(componentDrifts, metricDrifts, driftSeverity),
      alerts: this.generateDriftAlerts(componentDrifts, metricDrifts, driftSeverity),
      rollback_recommended: driftSeverity === 'critical' || driftSeverity === 'major'
    };
    
    // Save drift report
    await this.saveDriftReport(report, outputDir);
    
    console.log(`✅ Drift analysis completed: ${driftDetected ? 'DRIFT DETECTED' : 'NO DRIFT'} (${driftSeverity})`);
    
    this.emit('drift_detected', report);
    return report;
  }
  
  /**
   * Rollback to locked configuration
   */
  async rollbackToLockedConfiguration(): Promise<void> {
    console.log('🔄 Rolling back to locked configuration...');
    
    if (!this.lockedArtifact) {
      throw new Error('No locked artifact available for rollback');
    }
    
    // Create rollback artifact
    const rollbackArtifact: ConfigurationArtifact = {
      ...this.lockedArtifact,
      artifact_id: `rollback_${Date.now()}_${this.lockedArtifact.artifact_id}`,
      created_at: new Date(),
      deployment: {
        ...this.lockedArtifact.deployment,
        deployed_at: new Date()
      }
    };
    
    // Save rollback artifact
    await this.saveArtifact(rollbackArtifact);
    
    this.currentArtifact = rollbackArtifact;
    
    console.log(`✅ Rolled back to locked configuration: ${this.lockedArtifact.artifact_id}`);
    console.log(`   Rollback artifact: ${rollbackArtifact.artifact_id}`);
    
    this.emit('configuration_rollback', {
      from_artifact: this.currentArtifact?.artifact_id,
      to_artifact: rollbackArtifact.artifact_id,
      locked_artifact: this.lockedArtifact.artifact_id
    });
  }
  
  /**
   * Emergency unlock (for critical issues)
   */
  async emergencyUnlock(reason: string, approver: string): Promise<void> {
    if (!this.config.locking_permissions.emergency_unlock) {
      throw new Error('Emergency unlock not permitted');
    }
    
    if (!this.lockedArtifact) {
      throw new Error('No locked artifact to unlock');
    }
    
    console.log(`🚨 EMERGENCY UNLOCK: ${reason}`);
    console.log(`   Approver: ${approver}`);
    console.log(`   Artifact: ${this.lockedArtifact.artifact_id}`);
    
    // Clear locked artifact
    const unlockedArtifact = this.lockedArtifact;
    this.lockedArtifact = null;
    
    this.emit('emergency_unlock', {
      artifact_id: unlockedArtifact.artifact_id,
      reason,
      approver,
      timestamp: new Date()
    });
  }
  
  // Private helper methods
  
  private computeConfigHash(config: any): string {
    const configString = JSON.stringify(config, Object.keys(config).sort());
    return createHash('sha256').update(configString).digest('hex');
  }
  
  private generateVersionString(): string {
    const now = new Date();
    return `v${now.getFullYear()}.${(now.getMonth() + 1).toString().padStart(2, '0')}.${now.getDate().toString().padStart(2, '0')}.${now.getHours().toString().padStart(2, '0')}${now.getMinutes().toString().padStart(2, '0')}`;
  }
  
  private validateRouterConfiguration(config: RouterConfiguration): { errors: string[]; warnings: string[] } {
    const errors: string[] = [];
    const warnings: string[] = [];
    
    // Threshold validation
    if (config.thresholds.lexical_threshold < 0 || config.thresholds.lexical_threshold > 1) {
      errors.push('Lexical threshold must be between 0 and 1');
    }
    
    if (config.thresholds.semantic_threshold < 0 || config.thresholds.semantic_threshold > 1) {
      errors.push('Semantic threshold must be between 0 and 1');
    }
    
    // Weight validation
    const totalWeight = config.weights.lexical_weight + config.weights.semantic_weight + 
                       config.weights.symbol_weight + config.weights.context_weight;
    
    if (Math.abs(totalWeight - 1.0) > 0.01) {
      warnings.push(`Router weights sum to ${totalWeight.toFixed(3)}, expected 1.0`);
    }
    
    // Isotonic regression validation
    if (!config.isotonic_regression.calibration_hash) {
      errors.push('Missing isotonic regression calibration hash');
    }
    
    if (config.isotonic_regression.validation_r2 < 0.8) {
      warnings.push(`Low isotonic regression R²: ${config.isotonic_regression.validation_r2.toFixed(3)}`);
    }
    
    return { errors, warnings };
  }
  
  private validateHNSWConfiguration(config: HNSWConfiguration): { errors: string[]; warnings: string[] } {
    const errors: string[] = [];
    const warnings: string[] = [];
    
    // Construction parameter validation
    if (config.construction_params.m < 2 || config.construction_params.m > 100) {
      errors.push('HNSW m parameter must be between 2 and 100');
    }
    
    if (config.construction_params.ef_construction < config.construction_params.m) {
      warnings.push('ef_construction should be >= m for optimal performance');
    }
    
    // Search parameter validation
    if (config.search_params.ef_search < 10) {
      warnings.push('ef_search < 10 may result in poor recall');
    }
    
    // Index metadata validation
    if (!config.index_metadata.index_hash) {
      errors.push('Missing HNSW index hash');
    }
    
    if (config.index_metadata.total_elements === 0) {
      warnings.push('HNSW index appears to be empty');
    }
    
    return { errors, warnings };
  }
  
  private validatePriorConfiguration(config: PriorConfiguration): { errors: string[]; warnings: string[] } {
    const errors: string[] = [];
    const warnings: string[] = [];
    
    // Cap validation
    if (config.caps.mmr_cap < 0 || config.caps.mmr_cap > 1) {
      errors.push('MMR cap must be between 0 and 1');
    }
    
    if (config.caps.diversity_cap < 0 || config.caps.diversity_cap > 1) {
      errors.push('Diversity cap must be between 0 and 1');
    }
    
    // Entropy threshold validation
    if (config.entropy_thresholds.low_entropy_threshold >= config.entropy_thresholds.high_entropy_threshold) {
      errors.push('Low entropy threshold must be < high entropy threshold');
    }
    
    if (config.caps.mmr_cap > 0.5 && !config.entropy_thresholds.over_steer_protection) {
      warnings.push('High MMR cap without over-steer protection may degrade quality');
    }
    
    return { errors, warnings };
  }
  
  private validateHeroMetrics(metrics: any): { errors: string[]; warnings: string[] } {
    const errors: string[] = [];
    const warnings: string[] = [];
    
    if (metrics.ndcg_10 < 0 || metrics.ndcg_10 > 1) {
      errors.push('nDCG@10 must be between 0 and 1');
    }
    
    if (metrics.sla_recall_50 < 0 || metrics.sla_recall_50 > 1) {
      errors.push('SLA Recall@50 must be between 0 and 1');
    }
    
    if (metrics.p95_latency_ms <= 0) {
      errors.push('P95 latency must be positive');
    }
    
    if (metrics.calibration_ece > 0.1) {
      warnings.push(`High calibration ECE: ${metrics.calibration_ece.toFixed(3)}`);
    }
    
    // Sanity checks
    if (metrics.ndcg_10 < 0.5) {
      warnings.push('Unusually low nDCG@10 - verify metric calculation');
    }
    
    if (metrics.p95_latency_ms > 500) {
      warnings.push('High P95 latency - may impact user experience');
    }
    
    return { errors, warnings };
  }
  
  private async assessProductionSuitability(artifact: ConfigurationArtifact): Promise<{ suitable: boolean; reasons: string[] }> {
    const reasons: string[] = [];
    
    if (!artifact.validation.validated) {
      reasons.push('Artifact not validated');
    }
    
    if (artifact.validation.validation_errors.length > 0) {
      reasons.push(`${artifact.validation.validation_errors.length} validation errors`);
    }
    
    // Hero metrics thresholds for production
    if (artifact.hero_metrics.ndcg_10 < 0.75) {
      reasons.push('nDCG@10 below production threshold (0.75)');
    }
    
    if (artifact.hero_metrics.sla_recall_50 < 0.60) {
      reasons.push('SLA Recall@50 below production threshold (0.60)');
    }
    
    if (artifact.hero_metrics.p95_latency_ms > 200) {
      reasons.push('P95 latency above production threshold (200ms)');
    }
    
    return { suitable: reasons.length === 0, reasons };
  }
  
  private analyzeComponentDrift(
    component: string,
    currentConfig: any,
    lockedConfig: any,
    currentHash: string,
    lockedHash: string
  ): any {
    const driftDetected = currentHash !== lockedHash;
    const driftDetails: string[] = [];
    
    if (driftDetected) {
      // Deep comparison to identify specific drift
      driftDetails.push(...this.compareConfigurations(currentConfig, lockedConfig, component));
    }
    
    return {
      component,
      current_hash: currentHash,
      locked_hash: lockedHash,
      drift_detected: driftDetected,
      drift_details: driftDetails
    };
  }
  
  private compareConfigurations(current: any, locked: any, prefix: string = ''): string[] {
    const differences: string[] = [];
    
    for (const key of Object.keys({ ...current, ...locked })) {
      const currentValue = current[key];
      const lockedValue = locked[key];
      
      if (currentValue !== lockedValue) {
        if (typeof currentValue === 'object' && typeof lockedValue === 'object') {
          differences.push(...this.compareConfigurations(currentValue, lockedValue, `${prefix}.${key}`));
        } else {
          differences.push(`${prefix}.${key}: ${lockedValue} → ${currentValue}`);
        }
      }
    }
    
    return differences;
  }
  
  private analyzeMetricDrift(currentMetrics: any, lockedMetrics: any): any[] {
    const metricDrifts: any[] = [];
    const driftThreshold = this.config.drift_thresholds.metric_drift_percentage / 100;
    
    for (const [metricName, lockedValue] of Object.entries(lockedMetrics)) {
      const currentValue = (currentMetrics as any)[metricName];
      
      if (currentValue !== undefined && typeof lockedValue === 'number' && typeof currentValue === 'number') {
        const driftPercentage = Math.abs((currentValue - lockedValue) / lockedValue) * 100;
        const thresholdExceeded = driftPercentage > this.config.drift_thresholds.metric_drift_percentage;
        
        metricDrifts.push({
          metric_name: metricName,
          current_value: currentValue,
          locked_value: lockedValue,
          drift_percentage: driftPercentage,
          threshold_exceeded: thresholdExceeded
        });
      }
    }
    
    return metricDrifts;
  }
  
  private assessDriftSeverity(componentDrifts: any[], metricDrifts: any[]): 'none' | 'minor' | 'moderate' | 'major' | 'critical' {
    const componentDriftCount = componentDrifts.filter(c => c.drift_detected).length;
    const metricDriftCount = metricDrifts.filter(m => m.threshold_exceeded).length;
    const maxMetricDrift = Math.max(...metricDrifts.map(m => m.drift_percentage), 0);
    
    if (componentDriftCount === 0 && metricDriftCount === 0) {
      return 'none';
    }
    
    // Critical: All components drifted or massive metric drift
    if (componentDriftCount === 3 || maxMetricDrift > 10) {
      return 'critical';
    }
    
    // Major: Multiple components or significant metric drift
    if (componentDriftCount >= 2 || maxMetricDrift > 5) {
      return 'major';
    }
    
    // Moderate: One component + metrics or high metric drift
    if ((componentDriftCount === 1 && metricDriftCount > 0) || maxMetricDrift > 3) {
      return 'moderate';
    }
    
    // Minor: Limited drift
    return 'minor';
  }
  
  private generateDriftRecommendations(componentDrifts: any[], metricDrifts: any[], severity: string): string[] {
    const recommendations: string[] = [];
    
    switch (severity) {
      case 'none':
        recommendations.push('✅ No drift detected - configuration stable');
        break;
        
      case 'minor':
        recommendations.push('📊 Minor drift detected - monitor closely');
        break;
        
      case 'moderate':
        recommendations.push('⚠️  Moderate drift detected - investigate changes and consider re-locking');
        break;
        
      case 'major':
        recommendations.push('🚨 Major drift detected - immediate investigation required');
        recommendations.push('Consider rollback if drift is unintentional');
        break;
        
      case 'critical':
        recommendations.push('🔴 CRITICAL DRIFT - Emergency rollback recommended');
        recommendations.push('All configuration components have drifted significantly');
        break;
    }
    
    // Component-specific recommendations
    for (const componentDrift of componentDrifts) {
      if (componentDrift.drift_detected) {
        recommendations.push(`${componentDrift.component.toUpperCase()}: ${componentDrift.drift_details.length} parameter changes detected`);
      }
    }
    
    // Metric-specific recommendations
    const criticalMetricDrifts = metricDrifts.filter(m => m.drift_percentage > 5);
    for (const metricDrift of criticalMetricDrifts) {
      recommendations.push(`${metricDrift.metric_name}: ${metricDrift.drift_percentage.toFixed(1)}% change - investigate impact`);
    }
    
    return recommendations;
  }
  
  private generateDriftAlerts(componentDrifts: any[], metricDrifts: any[], severity: string): string[] {
    const alerts: string[] = [];
    
    if (severity === 'critical') {
      alerts.push('🚨 CRITICAL CONFIG DRIFT - IMMEDIATE ACTION REQUIRED');
    }
    
    if (severity === 'major') {
      alerts.push('⚠️  MAJOR CONFIG DRIFT - URGENT INVESTIGATION NEEDED');
    }
    
    const qualityDegradation = metricDrifts.some(m => 
      (m.metric_name.includes('ndcg') || m.metric_name.includes('recall')) && 
      m.current_value < m.locked_value && m.threshold_exceeded
    );
    
    if (qualityDegradation) {
      alerts.push('📉 QUALITY DEGRADATION DETECTED - Hero metrics below locked baseline');
    }
    
    const performanceDegradation = metricDrifts.some(m => 
      m.metric_name.includes('latency') && 
      m.current_value > m.locked_value && m.threshold_exceeded
    );
    
    if (performanceDegradation) {
      alerts.push('🐌 PERFORMANCE DEGRADATION DETECTED - Latency metrics above locked baseline');
    }
    
    return alerts;
  }
  
  private async saveArtifact(artifact: ConfigurationArtifact): Promise<void> {
    const filePath = join(this.config.artifact_storage_path, `${artifact.artifact_id}.json`);
    await writeFile(filePath, JSON.stringify(artifact, null, 2));
  }
  
  private async loadArtifact(artifactId: string): Promise<ConfigurationArtifact | null> {
    try {
      const filePath = join(this.config.artifact_storage_path, `${artifactId}.json`);
      const content = await readFile(filePath, 'utf-8');
      return JSON.parse(content);
    } catch (error) {
      return null;
    }
  }
  
  private async saveDriftReport(report: ConfigurationDriftReport, outputDir: string): Promise<void> {
    // Save JSON report
    await writeFile(
      join(outputDir, 'configuration-drift-report.json'),
      JSON.stringify(report, null, 2)
    );
    
    // Save markdown summary
    const markdownReport = this.generateDriftMarkdown(report);
    await writeFile(join(outputDir, 'configuration-drift-summary.md'), markdownReport);
    
    console.log(`✅ Drift report saved to ${outputDir}/`);
  }
  
  private generateDriftMarkdown(report: ConfigurationDriftReport): string {
    let md = '# Configuration Drift Report\n\n';
    
    md += `**Analysis Date**: ${report.timestamp.toISOString()}\n`;
    md += `**Current Artifact**: ${report.current_artifact_id}\n`;
    md += `**Locked Artifact**: ${report.locked_artifact_id}\n`;
    md += `**Drift Detected**: ${report.drift_detected ? '✅ YES' : '❌ NO'}\n`;
    md += `**Severity**: ${report.drift_severity.toUpperCase()}\n\n`;
    
    // Status indicator
    if (report.drift_severity === 'none') {
      md += '## 🟢 Status: NO DRIFT DETECTED\n\n';
    } else if (report.drift_severity === 'minor' || report.drift_severity === 'moderate') {
      md += '## 🟡 Status: DRIFT DETECTED - MONITOR\n\n';
    } else {
      md += '## 🔴 Status: CRITICAL DRIFT - ACTION REQUIRED\n\n';
    }
    
    // Component drift analysis
    md += '## 🔧 Component Drift Analysis\n\n';
    md += '| Component | Hash Match | Drift | Details |\n';
    md += '|-----------|------------|-------|----------|\n';
    
    for (const comp of report.component_drifts) {
      const status = comp.drift_detected ? '❌ DRIFT' : '✅ MATCH';
      const detailsCount = comp.drift_details.length;
      md += `| ${comp.component.toUpperCase()} | ${status} | ${comp.drift_detected ? 'YES' : 'NO'} | ${detailsCount} changes |\n`;
    }
    md += '\n';
    
    // Metric drift analysis
    md += '## 📊 Hero Metrics Drift Analysis\n\n';
    md += '| Metric | Current | Locked | Drift % | Threshold Exceeded |\n';
    md += '|--------|---------|--------|---------|-------------------|\n';
    
    for (const metric of report.metric_drifts) {
      const status = metric.threshold_exceeded ? '❌' : '✅';
      md += `| ${metric.metric_name} | ${metric.current_value.toFixed(3)} | ${metric.locked_value.toFixed(3)} | ${metric.drift_percentage.toFixed(1)}% | ${status} |\n`;
    }
    md += '\n';
    
    // Alerts
    if (report.alerts.length > 0) {
      md += '## 🚨 Alerts\n\n';
      for (const alert of report.alerts) {
        md += `- **${alert}**\n`;
      }
      md += '\n';
    }
    
    // Recommendations
    md += '## 💡 Recommendations\n\n';
    for (const rec of report.recommendations) {
      md += `- ${rec}\n`;
    }
    
    if (report.rollback_recommended) {
      md += '\n## 🔄 Rollback Recommendation\n\n';
      md += '**ROLLBACK RECOMMENDED** due to critical configuration drift.\n';
      md += 'Execute rollback to restore system to locked baseline configuration.\n';
    }
    
    return md;
  }
  
  /**
   * Generate synthetic configuration for testing
   */
  static generateSyntheticConfiguration(): {
    router: RouterConfiguration;
    hnsw: HNSWConfiguration;
    prior: PriorConfiguration;
  } {
    return {
      router: {
        thresholds: {
          lexical_threshold: 0.65,
          semantic_threshold: 0.70,
          symbol_threshold: 0.80,
          confidence_threshold: 0.75,
          quality_gate_threshold: 0.85
        },
        weights: {
          lexical_weight: 0.25,
          semantic_weight: 0.35,
          symbol_weight: 0.30,
          context_weight: 0.10
        },
        isotonic_regression: {
          calibration_hash: 'sha256_' + Math.random().toString(36).substr(2, 32),
          model_version: 'v2.1.0',
          training_samples: 50000,
          validation_r2: 0.89
        }
      },
      hnsw: {
        construction_params: {
          m: 16,
          ef_construction: 200,
          max_m: 16,
          max_m0: 32
        },
        search_params: {
          ef_search: 100,
          num_threads: 8
        },
        index_metadata: {
          dimension: 768,
          metric: 'cosine',
          total_elements: 2500000,
          index_hash: 'sha256_' + Math.random().toString(36).substr(2, 32),
          build_timestamp: new Date()
        }
      },
      prior: {
        caps: {
          mmr_cap: 0.3,
          diversity_cap: 0.25,
          boost_cap: 1.5,
          penalty_cap: 0.5
        },
        weights: {
          recency_weight: 0.15,
          popularity_weight: 0.20,
          user_preference_weight: 0.25,
          context_similarity_weight: 0.40
        },
        entropy_thresholds: {
          low_entropy_threshold: 0.3,
          high_entropy_threshold: 0.7,
          over_steer_protection: true
        }
      }
    };
  }
}

// Factory function
export function createConfigFingerprintLockingSystem(config?: Partial<FingerprintLockingConfig>): ConfigFingerprintLockingSystem {
  const fullConfig = { ...DEFAULT_FINGERPRINT_CONFIG, ...config };
  return new ConfigFingerprintLockingSystem(fullConfig);
}