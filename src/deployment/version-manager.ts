/**
 * Version Management & Config Fingerprinting System
 * 
 * Implements the Tag + Freeze system from TODO.md:
 * - Policy versioning with semantic versioning
 * - Config fingerprints with hashes and metadata
 * - Artifact freezing and rollback capability
 * - Integration with canary deployment system
 */

import { createHash } from 'crypto';
import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';

export interface ConfigFingerprint {
  version: string;
  timestamp: string;
  git_commit?: string;
  
  // Core system components
  api_version: string;
  index_version: string;
  policy_version: string;
  
  // Optimization parameters
  tau_value: number;
  ltr_model_hash: string;
  dedup_params: DeduplicationParams;
  early_exit_config: EarlyExitConfig;
  
  // Feature configuration
  feature_schema: FeatureSchema;
  reliability_curve: ReliabilityPoint[];
  
  // Performance baselines
  baseline_metrics: BaselineMetrics;
  
  // Safety configuration
  promotion_gates: PromotionGates;
  drift_thresholds: DriftThresholds;
}

interface DeduplicationParams {
  k: number;
  hamming_max: number;
  keep: number;
  simhash_bits: number;
}

interface EarlyExitConfig {
  margin: number;
  min_probes: number;
  confidence_threshold: number;
}

interface FeatureSchema {
  version: string;
  features: Array<{
    name: string;
    type: 'numeric' | 'categorical' | 'binary';
    importance_weight: number;
    normalization: 'none' | 'minmax' | 'zscore';
  }>;
  total_features: number;
}

export interface ReliabilityPoint {
  predicted_score: number;
  actual_precision: number;
  sample_size: number;
  confidence_interval: [number, number];
}

interface BaselineMetrics {
  p_at_1: number;
  ndcg_at_10: number;
  recall_at_50: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  span_coverage: number;
  results_per_query_mean: number;
  results_per_query_std: number;
}

interface PromotionGates {
  min_ndcg_delta: number;
  min_recall_delta: number;
  max_latency_p95_increase: number;
  max_latency_p99_ratio: number;
  required_span_coverage: number;
  max_hard_negative_leakage: number;
  max_results_per_query_drift: number;
}

interface DriftThresholds {
  cusum_threshold: number;
  feature_drift_sigma: number;
  reliability_update_holdout_days: number;
  sentinel_probe_frequency_hours: number;
  alarm_quiet_period_hours: number;
}

export class VersionManager {
  private readonly baseDir: string;
  private readonly artifactsDir: string;
  private currentVersion: string;
  
  constructor(baseDir: string = './deployment-artifacts') {
    this.baseDir = baseDir;
    this.artifactsDir = join(baseDir, 'versions');
    this.currentVersion = this.loadCurrentVersion();
    
    // Ensure directories exist
    if (!existsSync(this.baseDir)) {
      mkdirSync(this.baseDir, { recursive: true });
    }
    if (!existsSync(this.artifactsDir)) {
      mkdirSync(this.artifactsDir, { recursive: true });
    }
  }
  
  /**
   * Create new version and freeze current configuration
   */
  public async createVersion(
    tauValue: number,
    ltrModelHash: string,
    baselineMetrics: BaselineMetrics,
    reliabilityCurve: ReliabilityPoint[],
    gitCommit?: string
  ): Promise<string> {
    const newVersion = this.incrementVersion();
    const timestamp = new Date().toISOString();
    
    const fingerprint: ConfigFingerprint = {
      version: newVersion,
      timestamp,
      git_commit: gitCommit || await this.getCurrentGitCommit(),
      
      // System versions
      api_version: await this.getAPIVersion(),
      index_version: await this.getIndexVersion(), 
      policy_version: newVersion,
      
      // Core parameters
      tau_value: tauValue,
      ltr_model_hash: ltrModelHash,
      dedup_params: this.getDeduplicationParams(),
      early_exit_config: this.getEarlyExitConfig(),
      
      // Schema and calibration
      feature_schema: await this.getFeatureSchema(),
      reliability_curve: reliabilityCurve,
      
      // Baselines and gates
      baseline_metrics: baselineMetrics,
      promotion_gates: this.getPromotionGates(),
      drift_thresholds: this.getDriftThresholds()
    };
    
    // Save fingerprint
    const fingerprintPath = join(this.artifactsDir, `config_fingerprint_${newVersion}.json`);
    writeFileSync(fingerprintPath, JSON.stringify(fingerprint, null, 2));
    
    // Update current version pointer
    this.currentVersion = newVersion;
    this.saveCurrentVersion(newVersion);
    
    // Create version-specific artifact directory
    const versionDir = join(this.artifactsDir, newVersion);
    if (!existsSync(versionDir)) {
      mkdirSync(versionDir, { recursive: true });
    }
    
    console.log(`✅ Created version ${newVersion} with fingerprint at ${fingerprintPath}`);
    return newVersion;
  }
  
  /**
   * Load configuration for specific version
   */
  public loadVersionConfig(version?: string): ConfigFingerprint {
    const targetVersion = version || this.currentVersion;
    const fingerprintPath = join(this.artifactsDir, `config_fingerprint_${targetVersion}.json`);
    
    if (!existsSync(fingerprintPath)) {
      throw new Error(`Version ${targetVersion} not found at ${fingerprintPath}`);
    }
    
    const content = readFileSync(fingerprintPath, 'utf-8');
    return JSON.parse(content) as ConfigFingerprint;
  }
  
  /**
   * Get current active version
   */
  public getCurrentVersion(): string {
    return this.currentVersion;
  }
  
  /**
   * List all available versions
   */
  public getAvailableVersions(): string[] {
    if (!existsSync(this.artifactsDir)) {
      return [];
    }
    
    const files = require('fs').readdirSync(this.artifactsDir);
    return files
      .filter((f: string) => f.startsWith('config_fingerprint_') && f.endsWith('.json'))
      .map((f: string) => f.replace('config_fingerprint_', '').replace('.json', ''))
      .sort();
  }
  
  /**
   * Rollback to previous version
   */
  public rollbackToVersion(version: string): void {
    const config = this.loadVersionConfig(version);
    this.currentVersion = version;
    this.saveCurrentVersion(version);
    
    console.log(`🔄 Rolled back to version ${version} (${config.timestamp})`);
  }
  
  /**
   * Calculate configuration hash for integrity checking
   */
  public calculateConfigHash(config: ConfigFingerprint): string {
    // Remove timestamp and git_commit for stable hashing
    const { timestamp, git_commit, ...stableConfig } = config;
    
    const configString = JSON.stringify(stableConfig, Object.keys(stableConfig).sort());
    return createHash('sha256').update(configString).digest('hex').substring(0, 16);
  }

  /**
   * Simple hash calculation helper
   */
  private calculateHash(input: string): string {
    return createHash('sha256').update(input).digest('hex').substring(0, 8);
  }
  
  /**
   * Generate configuration fingerprint from current system state
   */
  public generateConfigFingerprint(): Partial<ConfigFingerprint> {
    return {
      version: this.currentVersion,
      timestamp: new Date().toISOString(),
      policy_version: this.currentVersion,
      api_config: {
        version: "1.0.1",
        endpoints: ["search", "index", "benchmark"],
        features: ["lexical", "symbols", "semantic"]
      },
      index_config: {
        lexical_enabled: true,
        symbols_enabled: true,
        semantic_enabled: true,
        version_hash: this.calculateHash("index_config_v1.0.1")
      },
      ltr_model_hash: "abc123def456",
      tau_value: 0.85,
      dedup_params: this.getDeduplicationParams(),
      early_exit_config: this.getEarlyExitConfig()
    };
  }

  /**
   * Validate version integrity
   */
  public validateVersionIntegrity(version: string): boolean {
    try {
      const config = this.loadVersionConfig(version);
      const calculatedHash = this.calculateConfigHash(config);
      
      // Verify all required fields exist
      const requiredFields = [
        'version', 'api_version', 'index_version', 'policy_version',
        'tau_value', 'ltr_model_hash', 'baseline_metrics'
      ];
      
      for (const field of requiredFields) {
        if (!(field in config)) {
          console.error(`❌ Missing required field: ${field}`);
          return false;
        }
      }
      
      console.log(`✅ Version ${version} integrity check passed (hash: ${calculatedHash})`);
      return true;
    } catch (error) {
      console.error(`❌ Version ${version} integrity check failed:`, error);
      return false;
    }
  }
  
  // Private helper methods
  
  private loadCurrentVersion(): string {
    const versionFile = join(this.baseDir, 'current_version.txt');
    if (existsSync(versionFile)) {
      return readFileSync(versionFile, 'utf-8').trim();
    }
    return '1.0.0'; // Default initial version
  }
  
  private saveCurrentVersion(version: string): void {
    const versionFile = join(this.baseDir, 'current_version.txt');
    writeFileSync(versionFile, version);
  }
  
  private incrementVersion(): string {
    const parts = this.currentVersion.split('.').map(Number);
    parts[1]++; // Increment minor version for v1.1
    parts[2] = 0; // Reset patch version
    return parts.join('.');
  }
  
  private async getCurrentGitCommit(): Promise<string> {
    try {
      const { execSync } = require('child_process');
      return execSync('git rev-parse HEAD', { encoding: 'utf-8' }).trim();
    } catch {
      return 'unknown';
    }
  }
  
  private async getAPIVersion(): Promise<string> {
    // Extract from package.json or API metadata
    try {
      const packageJson = JSON.parse(readFileSync('./package.json', 'utf-8'));
      return packageJson.version || '1.0.0';
    } catch {
      return '1.0.0';
    }
  }
  
  private async getIndexVersion(): Promise<string> {
    // Could be from index metadata or build timestamp
    return createHash('md5')
      .update(new Date().toISOString().split('T')[0])
      .digest('hex')
      .substring(0, 8);
  }
  
  private getDeduplicationParams(): DeduplicationParams {
    return {
      k: 5,
      hamming_max: 2,
      keep: 3,
      simhash_bits: 64
    };
  }
  
  private getEarlyExitConfig(): EarlyExitConfig {
    return {
      margin: 0.12,
      min_probes: 96,
      confidence_threshold: 0.85
    };
  }
  
  private async getFeatureSchema(): Promise<FeatureSchema> {
    return {
      version: '2.0.0',
      features: [
        { name: 'lexical_score', type: 'numeric', importance_weight: 0.25, normalization: 'minmax' },
        { name: 'symbol_match_score', type: 'numeric', importance_weight: 0.30, normalization: 'minmax' },
        { name: 'semantic_similarity', type: 'numeric', importance_weight: 0.20, normalization: 'zscore' },
        { name: 'file_popularity', type: 'numeric', importance_weight: 0.10, normalization: 'minmax' },
        { name: 'query_length_ratio', type: 'numeric', importance_weight: 0.10, normalization: 'none' },
        { name: 'has_exact_match', type: 'binary', importance_weight: 0.05, normalization: 'none' }
      ],
      total_features: 6
    };
  }
  
  private getPromotionGates(): PromotionGates {
    return {
      min_ndcg_delta: 0.02, // ≥ +2pp (p<0.05) as specified in TODO.md
      min_recall_delta: 0,   // Recall@50(≤150ms) ≥ baseline
      max_latency_p95_increase: 0.05, // p95 ≤ +5% as specified
      max_latency_p99_ratio: 2.0,
      required_span_coverage: 1.0, // span=100% as specified
      max_hard_negative_leakage: 0.01,
      max_results_per_query_drift: 1.0
    };
  }
  
  private getDriftThresholds(): DriftThresholds {
    return {
      cusum_threshold: 3.0,
      feature_drift_sigma: 3.0,
      reliability_update_holdout_days: 2,
      sentinel_probe_frequency_hours: 1,
      alarm_quiet_period_hours: 24
    };
  }
}

/**
 * Global version manager instance
 */
export const versionManager = new VersionManager();