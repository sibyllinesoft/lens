/**
 * Enhanced Configuration Management System
 * 
 * Advanced configuration management with:
 * - Environment-aware configuration loading
 * - Runtime configuration updates with validation
 * - Configuration versioning and migration
 * - Secret management integration
 * - Configuration templates and inheritance
 * - Hot-reload capabilities with change detection
 * - Configuration validation and schema enforcement
 * - Audit logging for configuration changes
 */

import { readFileSync, writeFileSync, existsSync, watchFile, unwatchFile } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { EventEmitter } from 'node:events';
import { cpus } from 'node:os';
import { z } from 'zod';
import { opentelemetry } from '../telemetry/index.js';

// Configuration schema definitions
const PerformanceConfigSchema = z.object({
  maxSearchDuration: z.number().min(1).max(10000).default(20),
  cacheSize: z.number().min(1000).max(100000000).default(10000000),
  parallelWorkers: z.number().min(1).max(64).default(4),
  memoryPoolSize: z.number().min(1000000).max(1000000000).default(100000000),
  simdEnabled: z.boolean().default(true),
  gcPressureThreshold: z.number().min(0.1).max(1.0).default(0.8)
});

const SearchConfigSchema = z.object({
  maxResults: z.number().min(1).max(10000).default(1000),
  fuzzyMatchThreshold: z.number().min(0).max(1).default(0.7),
  enableFederation: z.boolean().default(false),
  rankingStrategy: z.enum(['default', 'relevance-focused', 'popularity-focused', 'recency-focused']).default('default'),
  enableSemanticSearch: z.boolean().default(true),
  autoComplete: z.object({
    enabled: z.boolean().default(true),
    maxSuggestions: z.number().min(1).max(50).default(10),
    minQueryLength: z.number().min(1).max(10).default(2)
  })
});

const MonitoringConfigSchema = z.object({
  enabled: z.boolean().default(true),
  metricsRetention: z.string().regex(/^\d+[smhd]$/).default('24h'),
  alertThresholds: z.object({
    searchLatency: z.number().min(1).max(1000).default(20),
    errorRate: z.number().min(0).max(1).default(0.01),
    memoryUsage: z.number().min(50).max(99).default(90)
  }),
  exporters: z.array(z.enum(['prometheus', 'jaeger', 'console'])).default(['console'])
});

const SecurityConfigSchema = z.object({
  enableAuthentication: z.boolean().default(false),
  apiKeys: z.array(z.string()).default([]),
  rateLimiting: z.object({
    enabled: z.boolean().default(true),
    requestsPerMinute: z.number().min(1).max(10000).default(1000),
    burstSize: z.number().min(1).max(1000).default(100)
  }),
  cors: z.object({
    enabled: z.boolean().default(true),
    origins: z.array(z.string()).default(['*'])
  })
});

const ServerConfigSchema = z.object({
  host: z.string().default('localhost'),
  port: z.number().min(1024).max(65535).default(3000),
  environment: z.enum(['development', 'staging', 'production']).default('development'),
  logLevel: z.enum(['debug', 'info', 'warn', 'error']).default('info'),
  gracefulShutdownTimeout: z.number().min(1000).max(60000).default(10000)
});

// Main configuration schema
const ConfigSchema = z.object({
  version: z.string().default('1.0.0'),
  server: ServerConfigSchema,
  performance: PerformanceConfigSchema,
  search: SearchConfigSchema,
  monitoring: MonitoringConfigSchema,
  security: SecurityConfigSchema,
  features: z.object({
    experimentalFeatures: z.boolean().default(false),
    betaFeatures: z.boolean().default(false),
    debugMode: z.boolean().default(false)
  }).default({})
});

export type LensConfig = z.infer<typeof ConfigSchema>;
export type ConfigSection = keyof LensConfig;

// Configuration change event
export interface ConfigChangeEvent {
  section: string;
  key: string;
  oldValue: any;
  newValue: any;
  timestamp: number;
  source: 'file' | 'runtime' | 'environment' | 'secret';
}

// Configuration validation result
export interface ValidationResult {
  valid: boolean;
  errors: Array<{
    path: string;
    message: string;
    code: string;
  }>;
  warnings: Array<{
    path: string;
    message: string;
    recommendation?: string;
  }>;
}

// Configuration template
export interface ConfigTemplate {
  name: string;
  description: string;
  baseConfig: Partial<LensConfig>;
  overrides: Record<string, Partial<LensConfig>>;
}

/**
 * Enhanced configuration management system
 */
export class ConfigManager extends EventEmitter {
  private readonly tracer = opentelemetry.trace.getTracer('lens-config-manager');
  private static instance: ConfigManager | null = null;
  
  private config: LensConfig;
  private configPath: string;
  private watchEnabled: boolean = false;
  
  // Configuration history for rollback
  private configHistory: Array<{ config: LensConfig; timestamp: number; version: string }> = [];
  private maxHistorySize = 50;
  
  // Secret management
  private secretProviders = new Map<string, () => Promise<string>>();
  
  // Configuration templates
  private templates = new Map<string, ConfigTemplate>();
  
  // Validation cache
  private validationCache = new Map<string, ValidationResult>();

  private constructor(configPath?: string) {
    super();
    this.configPath = configPath || this.findConfigPath();
    this.config = this.loadConfiguration();
    this.initializeTemplates();
  }

  static getInstance(configPath?: string): ConfigManager {
    if (!ConfigManager.instance) {
      ConfigManager.instance = new ConfigManager(configPath);
    }
    return ConfigManager.instance;
  }

  /**
   * Get the current configuration
   */
  getConfig(): LensConfig {
    return { ...this.config };
  }

  /**
   * Get a specific configuration section
   */
  getSection<T extends ConfigSection>(section: T): LensConfig[T] {
    return { ...this.config[section] };
  }

  /**
   * Get a specific configuration value with optional default
   */
  get<T>(path: string, defaultValue?: T): T {
    const keys = path.split('.');
    let value: any = this.config;
    
    for (const key of keys) {
      if (value && typeof value === 'object' && key in value) {
        value = value[key];
      } else {
        return defaultValue as T;
      }
    }
    
    return value as T;
  }

  /**
   * Update configuration at runtime with validation
   */
  async updateConfig(updates: Partial<LensConfig>, source: 'runtime' | 'file' = 'runtime'): Promise<ValidationResult> {
    return await this.tracer.startActiveSpan('update-config', async (span) => {
      try {
        // Create merged configuration
        const newConfig = this.deepMerge(this.config, updates);
        
        // Validate the new configuration
        const validation = await this.validateConfig(newConfig);
        if (!validation.valid) {
          span.setAttributes({
            'lens.config.update_result': 'validation_failed',
            'lens.config.error_count': validation.errors.length
          });
          return validation;
        }
        
        // Store current config in history
        this.addToHistory(this.config);
        
        // Track changes and emit events
        const changes = this.detectChanges(this.config, newConfig);
        
        // Update configuration
        const oldConfig = this.config;
        this.config = newConfig;
        
        // Emit change events
        for (const change of changes) {
          this.emit('configChange', change);
        }
        
        // Persist to file if from runtime update
        if (source === 'runtime') {
          try {
            await this.saveConfiguration();
          } catch (error) {
            // Rollback on save failure
            this.config = oldConfig;
            throw new Error(`Failed to persist configuration: ${error}`);
          }
        }
        
        span.setAttributes({
          'lens.config.update_result': 'success',
          'lens.config.changes_count': changes.length
        });
        
        this.emit('configUpdated', { config: this.config, changes, source });
        
        return { valid: true, errors: [], warnings: validation.warnings };
        
      } catch (error) {
        span.recordException(error as Error);
        span.setStatus({ code: opentelemetry.SpanStatusCode.ERROR });
        throw error;
      } finally {
        span.end();
      }
    });
  }

  /**
   * Validate configuration against schema
   */
  async validateConfig(config: Partial<LensConfig>): Promise<ValidationResult> {
    const cacheKey = JSON.stringify(config);
    
    // Check validation cache
    if (this.validationCache.has(cacheKey)) {
      return this.validationCache.get(cacheKey)!;
    }
    
    try {
      // Parse with Zod schema
      ConfigSchema.parse(config);
      
      const result: ValidationResult = {
        valid: true,
        errors: [],
        warnings: this.getConfigWarnings(config as LensConfig)
      };
      
      // Cache result
      this.validationCache.set(cacheKey, result);
      
      // Limit cache size
      if (this.validationCache.size > 1000) {
        const keys = Array.from(this.validationCache.keys());
        for (let i = 0; i < 500; i++) {
          this.validationCache.delete(keys[i]);
        }
      }
      
      return result;
      
    } catch (error) {
      if (error instanceof z.ZodError) {
        const result: ValidationResult = {
          valid: false,
          errors: error.errors.map(err => ({
            path: err.path.join('.'),
            message: err.message,
            code: err.code
          })),
          warnings: []
        };
        
        this.validationCache.set(cacheKey, result);
        return result;
      }
      
      throw error;
    }
  }

  /**
   * Enable hot-reload of configuration files
   */
  enableHotReload(): void {
    if (this.watchEnabled) return;
    
    this.watchEnabled = true;
    
    watchFile(this.configPath, { interval: 1000 }, async () => {
      try {
        const newConfig = this.loadConfiguration();
        const validation = await this.validateConfig(newConfig);
        
        if (validation.valid) {
          const changes = this.detectChanges(this.config, newConfig);
          
          if (changes.length > 0) {
            this.addToHistory(this.config);
            this.config = newConfig;
            
            for (const change of changes) {
              this.emit('configChange', change);
            }
            
            this.emit('configReloaded', { config: this.config, changes });
          }
        } else {
          this.emit('configValidationFailed', validation);
        }
        
      } catch (error) {
        this.emit('configReloadError', error);
      }
    });
  }

  /**
   * Disable hot-reload
   */
  disableHotReload(): void {
    if (!this.watchEnabled) return;
    
    unwatchFile(this.configPath);
    this.watchEnabled = false;
  }

  /**
   * Rollback to previous configuration
   */
  async rollback(steps: number = 1): Promise<boolean> {
    if (this.configHistory.length === 0 || steps < 1) {
      return false;
    }
    
    const targetIndex = Math.min(steps - 1, this.configHistory.length - 1);
    const targetConfig = this.configHistory[targetIndex].config;
    
    // Validate target configuration
    const validation = await this.validateConfig(targetConfig);
    if (!validation.valid) {
      this.emit('rollbackFailed', { reason: 'Invalid target configuration', errors: validation.errors });
      return false;
    }
    
    // Perform rollback
    const changes = this.detectChanges(this.config, targetConfig);
    this.config = targetConfig;
    
    // Remove rolled back entries from history
    this.configHistory = this.configHistory.slice(steps);
    
    // Emit events
    for (const change of changes) {
      this.emit('configChange', change);
    }
    
    this.emit('configRolledBack', { config: this.config, steps, changes });
    
    // Persist rollback
    await this.saveConfiguration();
    
    return true;
  }

  /**
   * Apply configuration template
   */
  async applyTemplate(templateName: string, environment?: string): Promise<ValidationResult> {
    const template = this.templates.get(templateName);
    if (!template) {
      throw new Error(`Configuration template '${templateName}' not found`);
    }
    
    // Get base configuration
    let templateConfig = { ...template.baseConfig };
    
    // Apply environment-specific overrides
    if (environment && template.overrides[environment]) {
      templateConfig = this.deepMerge(templateConfig, template.overrides[environment]);
    }
    
    // Merge with current configuration
    const newConfig = this.deepMerge(this.config, templateConfig);
    
    return await this.updateConfig(newConfig, 'runtime');
  }

  /**
   * Register secret provider
   */
  registerSecretProvider(key: string, provider: () => Promise<string>): void {
    this.secretProviders.set(key, provider);
  }

  /**
   * Get configuration with secrets resolved
   */
  async getConfigWithSecrets(): Promise<LensConfig> {
    const config = { ...this.config };
    
    // Resolve any secret references
    await this.resolveSecrets(config);
    
    return config;
  }

  /**
   * Export configuration for backup
   */
  exportConfig(includeSecrets: boolean = false): string {
    const config = includeSecrets ? this.config : this.sanitizeConfig(this.config);
    
    return JSON.stringify({
      version: config.version,
      timestamp: new Date().toISOString(),
      config
    }, null, 2);
  }

  /**
   * Import configuration from backup
   */
  async importConfig(configData: string): Promise<ValidationResult> {
    try {
      const imported = JSON.parse(configData);
      
      if (!imported.config) {
        throw new Error('Invalid configuration format');
      }
      
      return await this.updateConfig(imported.config, 'runtime');
      
    } catch (error) {
      throw new Error(`Failed to import configuration: ${error}`);
    }
  }

  /**
   * Get configuration schema for documentation
   */
  getSchema(): any {
    return ConfigSchema._def;
  }

  /**
   * Private helper methods
   */
  
  private findConfigPath(): string {
    const candidates = [
      process.env.LENS_CONFIG_PATH,
      './lens.config.json',
      './config/lens.json',
      './src/config/lens.json'
    ].filter(Boolean) as string[];
    
    for (const path of candidates) {
      if (existsSync(resolve(path))) {
        return resolve(path);
      }
    }
    
    // Create default config
    const defaultPath = resolve('./lens.config.json');
    this.createDefaultConfig(defaultPath);
    return defaultPath;
  }

  private loadConfiguration(): LensConfig {
    try {
      if (!existsSync(this.configPath)) {
        this.createDefaultConfig(this.configPath);
      }
      
      const configData = readFileSync(this.configPath, 'utf8');
      const rawConfig = JSON.parse(configData);
      
      // Apply environment variable overrides
      const envConfig = this.loadEnvironmentOverrides();
      const mergedConfig = this.deepMerge(rawConfig, envConfig);
      
      // Validate and apply defaults
      const validatedConfig = ConfigSchema.parse(mergedConfig);
      
      return validatedConfig;
      
    } catch (error) {
      throw new Error(`Failed to load configuration from ${this.configPath}: ${error}`);
    }
  }

  private loadEnvironmentOverrides(): Partial<LensConfig> {
    const overrides: any = {};
    
    // Map environment variables to config paths
    const envMappings = [
      { env: 'LENS_PORT', path: 'server.port', transform: parseInt },
      { env: 'LENS_HOST', path: 'server.host' },
      { env: 'LENS_LOG_LEVEL', path: 'server.logLevel' },
      { env: 'LENS_MAX_RESULTS', path: 'search.maxResults', transform: parseInt },
      { env: 'LENS_CACHE_SIZE', path: 'performance.cacheSize', transform: parseInt },
      { env: 'LENS_PARALLEL_WORKERS', path: 'performance.parallelWorkers', transform: parseInt }
    ];
    
    for (const mapping of envMappings) {
      const value = process.env[mapping.env];
      if (value !== undefined) {
        const transformedValue = mapping.transform ? mapping.transform(value) : value;
        this.setNestedProperty(overrides, mapping.path, transformedValue);
      }
    }
    
    return overrides;
  }

  private createDefaultConfig(path: string): void {
    const defaultConfig = ConfigSchema.parse({});
    
    const configDir = dirname(path);
    if (!existsSync(configDir)) {
      throw new Error(`Configuration directory does not exist: ${configDir}`);
    }
    
    writeFileSync(path, JSON.stringify(defaultConfig, null, 2));
  }

  private async saveConfiguration(): Promise<void> {
    const configData = JSON.stringify(this.config, null, 2);
    writeFileSync(this.configPath, configData);
  }

  private detectChanges(oldConfig: LensConfig, newConfig: LensConfig): ConfigChangeEvent[] {
    const changes: ConfigChangeEvent[] = [];
    
    this.compareObjects(oldConfig, newConfig, '', changes);
    
    return changes;
  }

  private compareObjects(oldObj: any, newObj: any, path: string, changes: ConfigChangeEvent[]): void {
    const allKeys = new Set([...Object.keys(oldObj || {}), ...Object.keys(newObj || {})]);
    
    for (const key of allKeys) {
      const currentPath = path ? `${path}.${key}` : key;
      const oldValue = oldObj?.[key];
      const newValue = newObj?.[key];
      
      if (typeof oldValue === 'object' && typeof newValue === 'object' && 
          oldValue !== null && newValue !== null && !Array.isArray(oldValue) && !Array.isArray(newValue)) {
        this.compareObjects(oldValue, newValue, currentPath, changes);
      } else if (oldValue !== newValue) {
        changes.push({
          section: path.split('.')[0] || 'root',
          key: currentPath,
          oldValue,
          newValue,
          timestamp: Date.now(),
          source: 'runtime'
        });
      }
    }
  }

  private deepMerge(target: any, source: any): any {
    const result = { ...target };
    
    for (const key in source) {
      if (source[key] !== null && typeof source[key] === 'object' && !Array.isArray(source[key])) {
        result[key] = this.deepMerge(target[key] || {}, source[key]);
      } else {
        result[key] = source[key];
      }
    }
    
    return result;
  }

  private setNestedProperty(obj: any, path: string, value: any): void {
    const keys = path.split('.');
    let current = obj;
    
    for (let i = 0; i < keys.length - 1; i++) {
      const key = keys[i];
      if (!(key in current) || typeof current[key] !== 'object') {
        current[key] = {};
      }
      current = current[key];
    }
    
    current[keys[keys.length - 1]] = value;
  }

  private addToHistory(config: LensConfig): void {
    this.configHistory.unshift({
      config: { ...config },
      timestamp: Date.now(),
      version: config.version
    });
    
    // Limit history size
    if (this.configHistory.length > this.maxHistorySize) {
      this.configHistory = this.configHistory.slice(0, this.maxHistorySize);
    }
  }

  private getConfigWarnings(config: LensConfig): Array<{ path: string; message: string; recommendation?: string }> {
    const warnings = [];
    
    // Performance warnings
    if (config.performance.maxSearchDuration > 50) {
      warnings.push({
        path: 'performance.maxSearchDuration',
        message: 'Search duration limit is quite high',
        recommendation: 'Consider setting it below 50ms for better user experience'
      });
    }
    
    if (config.performance.parallelWorkers > cpus().length * 2) {
      warnings.push({
        path: 'performance.parallelWorkers',
        message: 'Parallel workers exceed recommended CPU ratio',
        recommendation: `Consider setting it to ${cpus().length * 2} or less`
      });
    }
    
    // Security warnings
    if (config.server.environment === 'production' && config.security.enableAuthentication === false) {
      warnings.push({
        path: 'security.enableAuthentication',
        message: 'Authentication is disabled in production',
        recommendation: 'Enable authentication for production environments'
      });
    }
    
    return warnings;
  }

  private async resolveSecrets(obj: any): Promise<void> {
    for (const [key, value] of Object.entries(obj)) {
      if (typeof value === 'string' && value.startsWith('secret:')) {
        const secretKey = value.substring(7); // Remove 'secret:' prefix
        const provider = this.secretProviders.get(secretKey);
        
        if (provider) {
          try {
            obj[key] = await provider();
          } catch (error) {
            throw new Error(`Failed to resolve secret '${secretKey}': ${error}`);
          }
        }
      } else if (typeof value === 'object' && value !== null) {
        await this.resolveSecrets(value);
      }
    }
  }

  private sanitizeConfig(config: LensConfig): LensConfig {
    const sanitized = JSON.parse(JSON.stringify(config));
    
    // Remove sensitive data
    if (sanitized.security?.apiKeys) {
      sanitized.security.apiKeys = ['***REDACTED***'];
    }
    
    return sanitized;
  }

  private initializeTemplates(): void {
    // Development template
    this.templates.set('development', {
      name: 'Development',
      description: 'Configuration optimized for development',
      baseConfig: {
        server: { logLevel: 'debug', environment: 'development' },
        features: { experimentalFeatures: true, debugMode: true },
        monitoring: { exporters: ['console'] },
        security: { enableAuthentication: false }
      },
      overrides: {}
    });
    
    // Production template
    this.templates.set('production', {
      name: 'Production',
      description: 'Configuration optimized for production',
      baseConfig: {
        server: { logLevel: 'info', environment: 'production' },
        features: { experimentalFeatures: false, debugMode: false },
        monitoring: { exporters: ['prometheus', 'jaeger'] },
        security: { enableAuthentication: true },
        performance: { maxSearchDuration: 15 }
      },
      overrides: {}
    });
    
    // High performance template
    this.templates.set('high-performance', {
      name: 'High Performance',
      description: 'Configuration optimized for maximum performance',
      baseConfig: {
        performance: {
          maxSearchDuration: 10,
          cacheSize: 50000000,
          parallelWorkers: cpus().length * 2,
          memoryPoolSize: 500000000,
          simdEnabled: true
        },
        search: {
          maxResults: 500,
          enableSemanticSearch: false
        }
      },
      overrides: {}
    });
  }
}

// Export singleton instance
export const configManager = ConfigManager.getInstance();