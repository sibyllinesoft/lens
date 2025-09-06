/**
 * RAPTOR System Main Entry Point
 * 
 * Integrates all RAPTOR components with feature flags and provides
 * the main interface for the Lens search system.
 */

import { SemanticCard } from './semantic-card.js';
import { SemanticCardExtractor } from './extractor.js';
import { RaptorSnapshot, RaptorSnapshotManager } from './snapshot.js';
import { RaptorBuilder, BuildResult } from './builder.js';
import { RaptorEmbeddingService, BusinessnessScorer } from './embeddings.js';
import { ReclusterDaemon, DaemonConfig } from './recluster-daemon.js';
import { RaptorRuntimeFeatures, RaptorFeatures, QueryEmbedding } from './runtime-features.js';
import { RaptorPolicyManager, RaptorPolicy } from './policy.js';
import { RaptorApiEndpoints } from './api-endpoints.js';
import EventEmitter from 'events';

export interface RaptorSystemConfig {
  storagePath: string;
  embeddingProvider?: 'mock' | 'sentence-transformer';
  embeddingApiEndpoint?: string;
  enabledFeatures: {
    raptor_semantic_cards: boolean;
    raptor_features: boolean;
    raptor_prior: boolean;
    raptor_recluster_daemon: boolean;
  };
  policy?: Partial<RaptorPolicy>;
}

export interface RaptorSystemStatus {
  enabled: boolean;
  features: {
    semantic_cards: boolean;
    stage_c_features: boolean;
    path_prior_boost: boolean;
    recluster_daemon: boolean;
  };
  health: 'healthy' | 'degraded' | 'unhealthy' | 'disabled';
  snapshots_loaded: number;
  daemon_status: any;
  last_error?: string;
}

export interface BuildSnapshotRequest {
  repo_sha: string;
  file_paths: string[];
  progress_callback?: (progress: any) => void;
}

export interface ComputeFeaturesRequest {
  repo_sha: string;
  query: string;
  candidate_files: string[];
  nl_score?: number;
}

export interface ComputeFeaturesResponse {
  features: Map<string, RaptorFeatures>;
  path_prior_boosts: Array<{ fileId: string; boost: number; reason: string }>;
  query_embedding: QueryEmbedding;
}

/**
 * Main RAPTOR system integrating all components
 */
export class RaptorSystem extends EventEmitter {
  private config: RaptorSystemConfig;
  private policyManager!: RaptorPolicyManager;
  private snapshotManager!: RaptorSnapshotManager;
  private extractor!: SemanticCardExtractor;
  private embeddingService!: RaptorEmbeddingService;
  private businessnessScorer!: BusinessnessScorer;
  private builder!: RaptorBuilder;
  private daemon!: ReclusterDaemon;
  private runtimeFeatures!: RaptorRuntimeFeatures;
  private apiEndpoints!: RaptorApiEndpoints;
  
  private isInitialized: boolean;
  private loadedSnapshots: Map<string, RaptorSnapshot>;
  private lastError?: Error;

  constructor(config: RaptorSystemConfig) {
    super();
    this.config = config;
    this.isInitialized = false;
    this.loadedSnapshots = new Map();

    // Initialize components
    this.initializeComponents();
  }

  private initializeComponents(): void {
    try {
      // Initialize policy manager
      const initialPolicy = RaptorPolicyManager.createDefaultPolicy();
      if (this.config.policy) {
        Object.assign(initialPolicy, this.config.policy);
      }
      this.policyManager = new RaptorPolicyManager(initialPolicy);

      // Initialize storage and snapshots
      this.snapshotManager = new RaptorSnapshotManager(this.config.storagePath);

      // Initialize embedding service
      this.embeddingService = RaptorEmbeddingService.createDefaultService();

      // Initialize extractor and scorer
      this.extractor = new SemanticCardExtractor(
        SemanticCardExtractor.createDefaultConfig()
      );
      this.businessnessScorer = new BusinessnessScorer();

      // Initialize builder
      this.builder = new RaptorBuilder(
        this.embeddingService,
        this.businessnessScorer,
        this.snapshotManager
      );

      // Initialize runtime features
      this.runtimeFeatures = new RaptorRuntimeFeatures(this.embeddingService);

      // Initialize daemon
      const daemonConfig = ReclusterDaemon.createDefaultConfig();
      daemonConfig.enabled = this.config.enabledFeatures.raptor_recluster_daemon;
      
      this.daemon = new ReclusterDaemon(
        daemonConfig,
        this.snapshotManager,
        this.embeddingService,
        this.builder
      );

      // Initialize API endpoints
      this.apiEndpoints = new RaptorApiEndpoints({
        policyManager: this.policyManager,
        snapshotManager: this.snapshotManager,
        extractor: this.extractor,
        builder: this.builder,
        daemon: this.daemon,
        runtimeFeatures: this.runtimeFeatures,
        embeddingService: this.embeddingService
      });

      // Set up event listeners
      this.setupEventListeners();

      this.isInitialized = true;
      this.emit('system-initialized', { config: this.config });

    } catch (error) {
      this.lastError = error instanceof Error ? error : new Error('Unknown initialization error');
      this.emit('system-error', { error: this.lastError });
      throw error;
    }
  }

  private setupEventListeners(): void {
    // Policy change events
    this.policyManager.on('policy-updated', (event) => {
      this.handlePolicyUpdate(event);
    });

    // Daemon events
    this.daemon.on('cycle-completed', () => {
      this.emit('recluster-cycle-completed');
    });

    this.daemon.on('cycle-error', (error) => {
      this.lastError = error;
      this.emit('recluster-error', { error });
    });

    // System health monitoring
    setInterval(() => {
      this.checkSystemHealth();
    }, 60000); // Check every minute
  }

  private handlePolicyUpdate(event: any): void {
    const { newPolicy, changes } = event;

    // Apply feature flag changes
    if ('enabled' in changes) {
      this.updateFeatureFlag('raptor_semantic_cards', newPolicy.enabled);
    }

    if ('prior_boost_enabled' in changes) {
      this.updateFeatureFlag('raptor_prior', newPolicy.prior_boost_enabled);
    }

    if ('recluster_daemon_enabled' in changes) {
      this.updateFeatureFlag('raptor_recluster_daemon', newPolicy.recluster_daemon_enabled);
      
      // Start/stop daemon based on flag
      if (newPolicy.recluster_daemon_enabled) {
        this.daemon.start();
      } else {
        this.daemon.stop();
      }
    }

    // Update runtime configuration
    this.runtimeFeatures.updateConfig({
      enabled: newPolicy.enabled,
      priorBoostCap: newPolicy.prior_boost_cap,
      topicThreshold: newPolicy.topic_threshold
    });

    this.emit('feature-flags-updated', {
      changes,
      current_flags: this.getCurrentFeatureFlags()
    });
  }

  private updateFeatureFlag(flag: string, enabled: boolean): void {
    (this.config.enabledFeatures as any)[flag] = enabled;
  }

  private checkSystemHealth(): void {
    try {
      const status = this.getStatus();
      
      if (status.health === 'unhealthy') {
        this.emit('system-unhealthy', { status, timestamp: Date.now() });
      } else if (status.health === 'degraded') {
        this.emit('system-degraded', { status, timestamp: Date.now() });
      }
    } catch (error) {
      this.lastError = error instanceof Error ? error : new Error('Health check failed');
      this.emit('system-error', { error: this.lastError });
    }
  }

  // Public API methods

  async start(): Promise<void> {
    if (!this.isInitialized) {
      throw new Error('System not initialized');
    }

    try {
      // Start daemon if enabled
      const policy = this.policyManager.getCurrentPolicy();
      if (policy.recluster_daemon_enabled) {
        this.daemon.start();
      }

      this.emit('system-started');
    } catch (error) {
      this.lastError = error instanceof Error ? error : new Error('Failed to start system');
      throw this.lastError;
    }
  }

  async stop(): Promise<void> {
    try {
      this.daemon.stop();
      this.emit('system-stopped');
    } catch (error) {
      this.lastError = error instanceof Error ? error : new Error('Failed to stop system');
      throw this.lastError;
    }
  }

  async buildSnapshot(request: BuildSnapshotRequest): Promise<BuildResult> {
    if (!this.config.enabledFeatures.raptor_semantic_cards) {
      throw new Error('Semantic cards feature not enabled');
    }

    try {
      // Extract semantic cards from files
      const cards: SemanticCard[] = [];
      
      for (const filePath of request.file_paths) {
        // In a real implementation, this would read files from disk
        // For now, create mock parsed files
        const mockFile = {
          file_id: filePath,
          file_path: filePath,
          content: '', // Would be read from disk
          file_sha: 'mock-sha',
          lang: this.inferLanguage(filePath)
        };

        const card = await this.extractor.extractCard(mockFile);
        cards.push(card);
      }

      // Update businessness scoring stats
      const stats = this.businessnessScorer.computeStatsFromCards(cards);
      this.businessnessScorer.updateStats(stats);

      // Build snapshot
      const result = await this.builder.buildSnapshot(
        request.repo_sha,
        cards,
        undefined,
        request.progress_callback
      );

      // Load the snapshot for runtime use
      this.loadedSnapshots.set(request.repo_sha, result.snapshot);
      this.daemon.loadSnapshot(result.snapshot);
      this.runtimeFeatures.loadSnapshot(request.repo_sha, result.snapshot);

      // Load semantic cards into runtime features
      for (const card of cards) {
        this.runtimeFeatures.loadSemanticCard(card.file_id, card);
      }

      this.emit('snapshot-built', { 
        repo_sha: request.repo_sha, 
        result,
        timestamp: Date.now()
      });

      return result;

    } catch (error) {
      this.lastError = error instanceof Error ? error : new Error('Failed to build snapshot');
      this.emit('snapshot-build-failed', { 
        repo_sha: request.repo_sha, 
        error: this.lastError 
      });
      throw this.lastError;
    }
  }

  async computeFeatures(request: ComputeFeaturesRequest): Promise<ComputeFeaturesResponse> {
    if (!this.config.enabledFeatures.raptor_features) {
      // Return empty features if not enabled
      return {
        features: new Map(),
        path_prior_boosts: [],
        query_embedding: await this.runtimeFeatures.prepareQuery(request.query)
      };
    }

    try {
      const queryEmbedding = await this.runtimeFeatures.prepareQuery(request.query);
      
      const features = await this.runtimeFeatures.computeFeatures(
        request.repo_sha,
        request.candidate_files,
        queryEmbedding,
        request.nl_score
      );

      let pathPriorBoosts: Array<{ fileId: string; boost: number; reason: string }> = [];
      
      if (this.config.enabledFeatures.raptor_prior) {
        pathPriorBoosts = await this.runtimeFeatures.computePathPriorBoosts(
          request.repo_sha,
          request.candidate_files,
          queryEmbedding
        );
      }

      return {
        features,
        path_prior_boosts: pathPriorBoosts,
        query_embedding: queryEmbedding
      };

    } catch (error) {
      this.lastError = error instanceof Error ? error : new Error('Failed to compute features');
      throw this.lastError;
    }
  }

  getStatus(): RaptorSystemStatus {
    const policy = this.policyManager.getCurrentPolicy();
    const daemonStatus = this.daemon.getStatus();
    const policyHealth = this.policyManager.checkPolicyHealth();

    // Compute health status
    let health: 'healthy' | 'degraded' | 'unhealthy' | 'disabled' = 'disabled';
    
    if (policy.enabled) {
      if (policyHealth.healthy && daemonStatus.warnings.length === 0) {
        health = 'healthy';
      } else if (policyHealth.healthy || daemonStatus.warnings.length <= 2) {
        health = 'degraded';
      } else {
        health = 'unhealthy';
      }
    }

    return {
      enabled: policy.enabled,
      features: {
        semantic_cards: policy.semantic_cards_enabled,
        stage_c_features: this.config.enabledFeatures.raptor_features,
        path_prior_boost: policy.prior_boost_enabled,
        recluster_daemon: policy.recluster_daemon_enabled
      },
      health,
      snapshots_loaded: this.loadedSnapshots.size,
      daemon_status: daemonStatus,
      last_error: this.lastError?.message
    };
  }

  getCurrentFeatureFlags(): Record<string, boolean> {
    const policy = this.policyManager.getCurrentPolicy();
    
    return {
      raptor_semantic_cards: this.config.enabledFeatures.raptor_semantic_cards,
      raptor_features: this.config.enabledFeatures.raptor_features,
      raptor_prior: this.config.enabledFeatures.raptor_prior,
      raptor_recluster_daemon: this.config.enabledFeatures.raptor_recluster_daemon,
      // Policy-based flags
      enabled: policy.enabled,
      prior_boost_enabled: policy.prior_boost_enabled,
      semantic_cards_enabled: policy.semantic_cards_enabled,
      recluster_daemon_enabled: policy.recluster_daemon_enabled
    };
  }

  // Component accessors for advanced usage
  getPolicyManager(): RaptorPolicyManager {
    return this.policyManager;
  }

  getDaemon(): ReclusterDaemon {
    return this.daemon;
  }

  getRuntimeFeatures(): RaptorRuntimeFeatures {
    return this.runtimeFeatures;
  }

  getApiEndpoints(): RaptorApiEndpoints {
    return this.apiEndpoints;
  }

  // Configuration methods
  updateFeatureFlags(flags: Partial<RaptorSystemConfig['enabledFeatures']>, user: string): void {
    Object.assign(this.config.enabledFeatures, flags);
    
    // Update corresponding policy flags
    const policyUpdates: Partial<RaptorPolicy> = {};
    
    if ('raptor_semantic_cards' in flags) {
      policyUpdates.semantic_cards_enabled = flags.raptor_semantic_cards;
    }
    
    if ('raptor_recluster_daemon' in flags) {
      policyUpdates.recluster_daemon_enabled = flags.raptor_recluster_daemon;
    }

    if (Object.keys(policyUpdates).length > 0) {
      this.policyManager.updatePolicy(policyUpdates, user, 'Feature flag update');
    }

    this.emit('feature-flags-updated', {
      flags,
      current_flags: this.getCurrentFeatureFlags()
    });
  }

  async emergencyShutdown(reason: string, user: string = 'system'): Promise<void> {
    try {
      // Disable all features
      await this.policyManager.emergencyDisable(user, reason);
      
      // Stop daemon
      this.daemon.stop();
      
      // Clear caches
      this.runtimeFeatures.clearCache();
      this.extractor.clearCache();
      this.embeddingService.clearCache();

      this.emit('emergency-shutdown', { reason, user, timestamp: Date.now() });
      
    } catch (error) {
      this.lastError = error instanceof Error ? error : new Error('Emergency shutdown failed');
      this.emit('system-error', { error: this.lastError });
      throw this.lastError;
    }
  }

  private inferLanguage(filePath: string): 'py' | 'ts' | 'js' {
    if (filePath.endsWith('.py')) return 'py';
    if (filePath.endsWith('.ts') || filePath.endsWith('.tsx')) return 'ts';
    if (filePath.endsWith('.js') || filePath.endsWith('.jsx')) return 'js';
    return 'js'; // Default
  }

  // Export system state for debugging
  exportSystemState(): any {
    return {
      config: this.config,
      status: this.getStatus(),
      policy: this.policyManager.getCurrentPolicy(),
      feature_flags: this.getCurrentFeatureFlags(),
      loaded_snapshots: Array.from(this.loadedSnapshots.keys()),
      last_error: this.lastError?.message,
      timestamp: Date.now()
    };
  }
}

// Factory function for easy initialization
export function createRaptorSystem(config: RaptorSystemConfig): RaptorSystem {
  return new RaptorSystem(config);
}

// Re-export key types for convenience
export type {
  RaptorFeatures,
  QueryEmbedding,
  SemanticCard,
  RaptorSnapshot,
  RaptorPolicy,
  BuildResult
};

export default RaptorSystem;