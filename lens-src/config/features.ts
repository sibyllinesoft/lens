/**
 * Feature flags for safe deployment and A/B testing
 * Controls experimental optimizations and new features
 */

export interface FeatureFlags {
  // Phase B1: Bitmap-based trigram optimization
  bitmapTrigramIndex: {
    enabled: boolean;
    // Rollout percentage (0-100) for gradual deployment
    rolloutPercentage: number;
    // Minimum document count threshold to enable bitmap optimization
    minDocumentThreshold: number;
    // Enable performance comparison logging
    enablePerformanceLogging: boolean;
  };
  
  // Prefilter optimizations
  prefilter: {
    enabled: boolean;
    // Maximum candidates to process before prefiltering
    maxCandidatesBeforeFilter: number;
  };

  // Experimental features
  experimental: {
    // Enable experimental FST improvements
    advancedFST: boolean;
    // Enable semantic reranking in Stage-C
    semanticRerank: boolean;
  };
}

// Default feature flag configuration
const DEFAULT_FEATURES: FeatureFlags = {
  bitmapTrigramIndex: {
    enabled: true,
    rolloutPercentage: 100, // Full rollout for Phase B1
    minDocumentThreshold: 100, // Only enable for larger document sets
    enablePerformanceLogging: true,
  },
  prefilter: {
    enabled: true,
    maxCandidatesBeforeFilter: 10000,
  },
  experimental: {
    advancedFST: false,
    semanticRerank: false,
  },
};

/**
 * Feature flag manager with environment variable overrides
 */
export class FeatureFlagManager {
  private features: FeatureFlags;

  constructor(overrides?: Partial<FeatureFlags>) {
    this.features = { ...DEFAULT_FEATURES };
    
    // Apply environment variable overrides
    this.applyEnvironmentOverrides();
    
    // Apply explicit overrides
    if (overrides) {
      this.features = this.mergeFeatureFlags(this.features, overrides);
    }
  }

  /**
   * Get current feature flag configuration
   */
  getFeatures(): FeatureFlags {
    return { ...this.features };
  }

  /**
   * Check if bitmap trigram index should be used for a given context
   */
  shouldUseBitmapIndex(documentCount: number, userHash?: string): boolean {
    const config = this.features.bitmapTrigramIndex;
    
    if (!config.enabled) {
      return false;
    }

    // Check document count threshold
    if (documentCount < config.minDocumentThreshold) {
      return false;
    }

    // Check rollout percentage
    if (config.rolloutPercentage < 100) {
      const hash = userHash ? this.hashString(userHash) : Math.random();
      return (hash * 100) < config.rolloutPercentage;
    }

    return true;
  }

  /**
   * Check if prefilter optimization is enabled
   */
  isPrefilterEnabled(): boolean {
    return this.features.prefilter.enabled;
  }

  /**
   * Get prefilter candidate threshold
   */
  getPrefilterThreshold(): number {
    return this.features.prefilter.maxCandidatesBeforeFilter;
  }

  /**
   * Check if performance logging is enabled for bitmap optimization
   */
  isBitmapPerformanceLoggingEnabled(): boolean {
    return this.features.bitmapTrigramIndex.enablePerformanceLogging;
  }

  /**
   * Update feature flags at runtime (for testing and experimentation)
   */
  updateFeatures(updates: Partial<FeatureFlags>): void {
    this.features = this.mergeFeatureFlags(this.features, updates);
  }

  /**
   * Apply environment variable overrides
   */
  private applyEnvironmentOverrides(): void {
    // Bitmap trigram index overrides
    if (process.env['LENS_BITMAP_INDEX_ENABLED'] !== undefined) {
      this.features.bitmapTrigramIndex.enabled = process.env['LENS_BITMAP_INDEX_ENABLED'] === 'true';
    }
    
    if (process.env['LENS_BITMAP_ROLLOUT_PERCENTAGE'] !== undefined) {
      const percentage = parseInt(process.env['LENS_BITMAP_ROLLOUT_PERCENTAGE'], 10);
      if (!isNaN(percentage) && percentage >= 0 && percentage <= 100) {
        this.features.bitmapTrigramIndex.rolloutPercentage = percentage;
      }
    }
    
    if (process.env['LENS_BITMAP_MIN_DOCS'] !== undefined) {
      const minDocs = parseInt(process.env['LENS_BITMAP_MIN_DOCS'], 10);
      if (!isNaN(minDocs) && minDocs >= 0) {
        this.features.bitmapTrigramIndex.minDocumentThreshold = minDocs;
      }
    }

    // Prefilter overrides
    if (process.env['LENS_PREFILTER_ENABLED'] !== undefined) {
      this.features.prefilter.enabled = process.env['LENS_PREFILTER_ENABLED'] === 'true';
    }

    // Experimental features
    if (process.env['LENS_EXPERIMENTAL_FST'] !== undefined) {
      this.features.experimental.advancedFST = process.env['LENS_EXPERIMENTAL_FST'] === 'true';
    }
  }

  /**
   * Deep merge feature flag configurations
   */
  private mergeFeatureFlags(base: FeatureFlags, overrides: Partial<FeatureFlags>): FeatureFlags {
    return {
      bitmapTrigramIndex: {
        ...base.bitmapTrigramIndex,
        ...overrides.bitmapTrigramIndex,
      },
      prefilter: {
        ...base.prefilter,
        ...overrides.prefilter,
      },
      experimental: {
        ...base.experimental,
        ...overrides.experimental,
      },
    };
  }

  /**
   * Simple string hash for consistent user bucketing
   */
  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash) / 2147483647; // Normalize to 0-1
  }
}

// Global feature flag manager instance
export const featureFlags = new FeatureFlagManager();