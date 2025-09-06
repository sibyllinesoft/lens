/**
 * Embedding Configuration Manager for EmbeddingGemma Migration
 * 
 * Provides centralized configuration management, model switching capabilities,
 * and runtime reconfiguration for the EmbeddingGemma migration project.
 */

import { EmbeddingProvider } from './embeddings.js';
import { EmbeddingGemmaProvider, GemmaEmbeddingConfig, MatryoshkaConfig } from './embedding-gemma-provider.js';
import { EmbeddingModelType } from './shadow-index-manager.js';
import { LensTracer } from '../telemetry/tracer.js';
import * as fs from 'fs/promises';
import * as path from 'path';

export interface EmbeddingModelConfig {
  type: EmbeddingModelType;
  enabled: boolean;
  priority: number; // Higher = preferred
  
  // Provider-specific configuration
  teiEndpoint?: string;
  openaiConfig?: {
    apiKey: string;
    model: string;
    baseUrl?: string;
  };
  
  // Matryoshka configuration for Gemma models
  matryoshka?: MatryoshkaConfig;
  
  // Performance settings
  performance: {
    batchSize: number;
    maxRetries: number;
    timeout: number;
    enableCaching: boolean;
  };
  
  // Quality settings
  quality: {
    minSimilarityThreshold: number;
    enableQuantization: boolean;
    quantizationMethod?: 'int8' | 'fp16';
  };
}

export interface GlobalEmbeddingConfig {
  // Active configuration
  primary: EmbeddingModelType;
  fallback: EmbeddingModelType;
  shadowTesting: boolean;
  
  // Model configurations
  models: Record<EmbeddingModelType, EmbeddingModelConfig>;
  
  // Migration settings
  migration: {
    enabled: boolean;
    phaseoutSchedule?: {
      ada002Deadline: string;
      enableGemmaDate: string;
    };
    abTestingConfig?: {
      trafficSplit: Record<EmbeddingModelType, number>; // Percentages
      metrics: string[];
      duration: string;
    };
  };
  
  // Monitoring and observability
  monitoring: {
    enableMetrics: boolean;
    enableTracing: boolean;
    enableHealthChecks: boolean;
    metricsEndpoint?: string;
  };
  
  // Storage and persistence
  storage: {
    indexPath: string;
    backupEnabled: boolean;
    compressionEnabled: boolean;
  };
}

export interface ConfigValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  recommendations: string[];
}

export interface ModelSwitchResult {
  success: boolean;
  previousModel: EmbeddingModelType;
  newModel: EmbeddingModelType;
  switchTimeMs: number;
  healthChecksPassed: boolean;
  indexesValidated: boolean;
}

/**
 * Centralized configuration manager for embedding models
 */
export class EmbeddingConfigManager {
  private config: GlobalEmbeddingConfig;
  private providers: Map<EmbeddingModelType, EmbeddingProvider> = new Map();
  private configFilePath: string;
  private configWatcher?: any; // FileSystemWatcher type not available in fs/promises
  
  // Runtime state
  private activeProvider: EmbeddingProvider | null = null;
  private fallbackProvider: EmbeddingProvider | null = null;
  private healthCheckIntervals: Map<EmbeddingModelType, NodeJS.Timeout> = new Map();

  constructor(configFilePath: string) {
    this.configFilePath = configFilePath;
    this.config = this.getDefaultConfig();
  }

  /**
   * Initialize the configuration manager
   */
  async initialize(): Promise<void> {
    const span = LensTracer.createChildSpan('config_manager_init');

    try {
      // Load configuration from file
      await this.loadConfig();
      
      // Validate configuration
      const validation = await this.validateConfig();
      if (!validation.valid) {
        throw new Error(`Invalid configuration: ${validation.errors.join(', ')}`);
      }

      // Initialize providers based on configuration
      await this.initializeProviders();
      
      // Set active and fallback providers
      await this.setActiveProvider(this.config.primary);
      
      // Start health monitoring
      await this.startHealthMonitoring();
      
      // Watch for configuration changes
      await this.startConfigWatcher();

      span.setAttributes({
        success: true,
        primary_model: this.config.primary,
        providers_initialized: this.providers.size,
        shadow_testing: this.config.shadowTesting,
      });

      console.log(`✅ Embedding configuration manager initialized`);
      console.log(`   Primary: ${this.config.primary}`);
      console.log(`   Fallback: ${this.config.fallback}`);
      console.log(`   Providers: ${Array.from(this.providers.keys()).join(', ')}`);

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Get the current active embedding provider
   */
  getActiveProvider(): EmbeddingProvider {
    if (!this.activeProvider) {
      throw new Error('No active embedding provider configured');
    }
    return this.activeProvider;
  }

  /**
   * Get a specific provider by model type
   */
  getProvider(modelType: EmbeddingModelType): EmbeddingProvider | null {
    return this.providers.get(modelType) || null;
  }

  /**
   * Switch to a different embedding model
   */
  async switchModel(targetModel: EmbeddingModelType): Promise<ModelSwitchResult> {
    const span = LensTracer.createChildSpan('switch_embedding_model', {
      'target.model': targetModel,
      'current.model': this.config.primary,
    });

    const switchStart = Date.now();
    const previousModel = this.config.primary;

    try {
      console.log(`🔄 Switching from ${previousModel} to ${targetModel}...`);

      // Validate target model is available and configured
      if (!this.config.models[targetModel]?.enabled) {
        throw new Error(`Target model ${targetModel} is not enabled`);
      }

      const targetProvider = this.providers.get(targetModel);
      if (!targetProvider) {
        throw new Error(`No provider available for ${targetModel}`);
      }

      // Health check target provider
      const healthCheckPassed = await this.performHealthCheck(targetModel);
      if (!healthCheckPassed) {
        throw new Error(`Health check failed for ${targetModel}`);
      }

      // Validate indexes are compatible
      const indexesValid = await this.validateIndexCompatibility(targetModel);

      // Perform the switch
      this.activeProvider = targetProvider;
      this.config.primary = targetModel;

      // Update configuration file
      await this.saveConfig();

      const switchTime = Date.now() - switchStart;

      const result: ModelSwitchResult = {
        success: true,
        previousModel,
        newModel: targetModel,
        switchTimeMs: switchTime,
        healthChecksPassed: healthCheckPassed,
        indexesValidated: indexesValid,
      };

      span.setAttributes({
        success: true,
        switch_time_ms: switchTime,
        health_checks_passed: healthCheckPassed,
        indexes_validated: indexesValid,
      });

      console.log(`✅ Successfully switched to ${targetModel} (${switchTime}ms)`);
      return result;

    } catch (error) {
      const result: ModelSwitchResult = {
        success: false,
        previousModel,
        newModel: targetModel,
        switchTimeMs: Date.now() - switchStart,
        healthChecksPassed: false,
        indexesValidated: false,
      };

      span.recordException(error as Error);
      span.setAttributes({ success: false });
      
      console.error(`❌ Failed to switch to ${targetModel}:`, error);
      return result;

    } finally {
      span.end();
    }
  }

  /**
   * Update Matryoshka configuration for Gemma models
   */
  async updateMatryoshkaConfig(
    modelType: 'gemma-768' | 'gemma-256',
    matryoshkaConfig: Partial<MatryoshkaConfig>
  ): Promise<void> {
    const span = LensTracer.createChildSpan('update_matryoshka_config', {
      'model.type': modelType,
    });

    try {
      // Update in-memory configuration
      const modelConfig = this.config.models[modelType];
      if (modelConfig.matryoshka) {
        modelConfig.matryoshka = { ...modelConfig.matryoshka, ...matryoshkaConfig };
      }

      // Update provider if it exists
      const provider = this.providers.get(modelType) as EmbeddingGemmaProvider;
      if (provider && 'updateMatryoshkaConfig' in provider) {
        await provider.updateMatryoshkaConfig(matryoshkaConfig);
      }

      // Persist configuration
      await this.saveConfig();

      span.setAttributes({
        success: true,
        target_dimension: matryoshkaConfig.targetDimension,
        preserve_ranking: matryoshkaConfig.preserveRanking,
      });

      console.log(`✅ Updated Matryoshka config for ${modelType}`);

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Enable or disable shadow testing
   */
  async configureShadowTesting(
    enabled: boolean,
    trafficSplit?: Record<EmbeddingModelType, number>
  ): Promise<void> {
    const span = LensTracer.createChildSpan('configure_shadow_testing', {
      enabled,
    });

    try {
      this.config.shadowTesting = enabled;
      
      if (enabled && trafficSplit) {
        this.config.migration.abTestingConfig = {
          trafficSplit,
          metrics: ['recall_at_50', 'latency_p95', 'cbu_per_gb'],
          duration: '7d',
        };
      }

      await this.saveConfig();

      span.setAttributes({
        success: true,
        traffic_split: trafficSplit ? JSON.stringify(trafficSplit) : undefined,
      });

      console.log(`✅ Shadow testing ${enabled ? 'enabled' : 'disabled'}`);

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Get current configuration
   */
  getConfig(): GlobalEmbeddingConfig {
    return { ...this.config };
  }

  /**
   * Validate configuration against constraints and best practices
   */
  async validateConfig(config: GlobalEmbeddingConfig = this.config): Promise<ConfigValidationResult> {
    const errors: string[] = [];
    const warnings: string[] = [];
    const recommendations: string[] = [];

    // Check primary model is configured and enabled
    if (!config.models[config.primary]?.enabled) {
      errors.push(`Primary model ${config.primary} is not enabled`);
    }

    // Check fallback model is different from primary
    if (config.primary === config.fallback) {
      warnings.push('Primary and fallback models are the same');
    }

    // Validate TEI endpoints for Gemma models
    for (const [modelType, modelConfig] of Object.entries(config.models)) {
      if (modelType.startsWith('gemma') && !modelConfig.teiEndpoint) {
        errors.push(`Gemma model ${modelType} missing TEI endpoint`);
      }
    }

    // Check A/B testing configuration
    if (config.migration.abTestingConfig) {
      const totalSplit = Object.values(config.migration.abTestingConfig.trafficSplit)
        .reduce((sum, percent) => sum + percent, 0);
      
      if (Math.abs(totalSplit - 100) > 0.1) {
        errors.push(`Traffic split percentages sum to ${totalSplit}%, not 100%`);
      }
    }

    // Performance recommendations
    const primaryConfig = config.models[config.primary];
    if (primaryConfig.performance.batchSize > 64) {
      recommendations.push('Consider reducing batch size for better latency');
    }

    // Storage recommendations
    if (!config.storage.compressionEnabled) {
      recommendations.push('Enable compression to reduce storage requirements');
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings,
      recommendations,
    };
  }

  /**
   * Export configuration for debugging/backup
   */
  async exportConfig(outputPath: string): Promise<void> {
    const exportData = {
      config: this.config,
      providerStates: Array.from(this.providers.entries()).map(([type, provider]) => ({
        type,
        dimension: provider.getDimension(),
        healthy: this.providers.has(type),
      })),
      exportTime: new Date().toISOString(),
    };

    await fs.writeFile(outputPath, JSON.stringify(exportData, null, 2));
    console.log(`📄 Configuration exported to ${outputPath}`);
  }

  // Private methods
  private getDefaultConfig(): GlobalEmbeddingConfig {
    return {
      primary: 'gemma-768',
      fallback: 'gemma-256',
      shadowTesting: false,
      
      models: {
        'ada-002': {
          type: 'ada-002',
          enabled: false,
          priority: 1,
          openaiConfig: {
            apiKey: process.env.OPENAI_API_KEY || '',
            model: 'text-embedding-ada-002',
          },
          performance: {
            batchSize: 16,
            maxRetries: 3,
            timeout: 30000,
            enableCaching: true,
          },
          quality: {
            minSimilarityThreshold: 0.1,
            enableQuantization: false,
          },
        },
        
        'gemma-768': {
          type: 'gemma-768',
          enabled: true,
          priority: 3,
          teiEndpoint: 'http://localhost:8080',
          matryoshka: {
            enabled: true,
            targetDimension: 768,
            preserveRanking: true,
          },
          performance: {
            batchSize: 32,
            maxRetries: 3,
            timeout: 15000,
            enableCaching: true,
          },
          quality: {
            minSimilarityThreshold: 0.1,
            enableQuantization: true,
            quantizationMethod: 'int8',
          },
        },
        
        'gemma-256': {
          type: 'gemma-256',
          enabled: true,
          priority: 2,
          teiEndpoint: 'http://localhost:8080',
          matryoshka: {
            enabled: true,
            targetDimension: 256,
            preserveRanking: true,
          },
          performance: {
            batchSize: 64,
            maxRetries: 3,
            timeout: 10000,
            enableCaching: true,
          },
          quality: {
            minSimilarityThreshold: 0.1,
            enableQuantization: true,
            quantizationMethod: 'int8',
          },
        },
      },
      
      migration: {
        enabled: true,
        phaseoutSchedule: {
          ada002Deadline: '2024-12-31',
          enableGemmaDate: '2024-06-01',
        },
      },
      
      monitoring: {
        enableMetrics: true,
        enableTracing: true,
        enableHealthChecks: true,
      },
      
      storage: {
        indexPath: './indexes',
        backupEnabled: true,
        compressionEnabled: true,
      },
    };
  }

  private async loadConfig(): Promise<void> {
    try {
      const configData = await fs.readFile(this.configFilePath, 'utf8');
      this.config = JSON.parse(configData);
      console.log(`📁 Loaded configuration from ${this.configFilePath}`);
    } catch (error) {
      console.log(`⚠️  No config file found, using defaults`);
      await this.saveConfig(); // Create default config file
    }
  }

  private async saveConfig(): Promise<void> {
    const configData = JSON.stringify(this.config, null, 2);
    await fs.writeFile(this.configFilePath, configData);
  }

  private async initializeProviders(): Promise<void> {
    for (const [modelType, modelConfig] of Object.entries(this.config.models)) {
      if (!modelConfig.enabled) {
        continue;
      }

      if (modelType.startsWith('gemma') && modelConfig.teiEndpoint) {
        const gemmaConfig: Partial<GemmaEmbeddingConfig> = {
          teiEndpoint: modelConfig.teiEndpoint,
          matryoshka: modelConfig.matryoshka,
          batchSize: modelConfig.performance.batchSize,
          maxRetries: modelConfig.performance.maxRetries,
          timeout: modelConfig.performance.timeout,
        };

        const provider = new EmbeddingGemmaProvider(gemmaConfig);
        this.providers.set(modelType as EmbeddingModelType, provider);
      }
      // Add other provider types (ada-002) as needed
    }
  }

  private async setActiveProvider(modelType: EmbeddingModelType): Promise<void> {
    const provider = this.providers.get(modelType);
    if (!provider) {
      throw new Error(`No provider available for ${modelType}`);
    }

    this.activeProvider = provider;
    
    // Set fallback provider
    const fallbackProvider = this.providers.get(this.config.fallback);
    if (fallbackProvider) {
      this.fallbackProvider = fallbackProvider;
    }
  }

  private async performHealthCheck(modelType: EmbeddingModelType): Promise<boolean> {
    const provider = this.providers.get(modelType);
    if (!provider) {
      return false;
    }

    try {
      if ('healthCheck' in provider && typeof provider.healthCheck === 'function') {
        return await provider.healthCheck();
      }
      
      // Fallback: try a simple embedding
      await provider.embed(['health check']);
      return true;
    } catch (error) {
      console.warn(`Health check failed for ${modelType}:`, error);
      return false;
    }
  }

  private async validateIndexCompatibility(modelType: EmbeddingModelType): Promise<boolean> {
    // For now, assume all indexes are compatible
    // In practice, would check dimension compatibility, etc.
    return true;
  }

  private async startHealthMonitoring(): Promise<void> {
    for (const modelType of this.providers.keys()) {
      const interval = setInterval(async () => {
        const healthy = await this.performHealthCheck(modelType);
        if (!healthy) {
          console.warn(`⚠️  Health check failed for ${modelType}`);
        }
      }, 30000); // Check every 30 seconds

      this.healthCheckIntervals.set(modelType, interval);
    }
  }

  private async startConfigWatcher(): Promise<void> {
    try {
      const { watch } = await import('fs');
      this.configWatcher = watch(this.configFilePath, async (eventType) => {
        if (eventType === 'change') {
          console.log('📁 Configuration file changed, reloading...');
          await this.loadConfig();
          // Could trigger re-initialization if needed
        }
      });
    } catch (error) {
      console.warn('Failed to start config watcher:', error);
    }
  }

  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    // Clear health check intervals
    for (const interval of this.healthCheckIntervals.values()) {
      clearInterval(interval);
    }
    this.healthCheckIntervals.clear();

    // Close config watcher
    if (this.configWatcher) {
      this.configWatcher.close();
    }

    // Cleanup providers
    for (const provider of this.providers.values()) {
      if ('cleanup' in provider && typeof provider.cleanup === 'function') {
        await provider.cleanup();
      }
    }

    this.providers.clear();
    this.activeProvider = null;
    this.fallbackProvider = null;

    console.log('Embedding configuration manager cleaned up');
  }
}