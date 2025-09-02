/**
 * Adaptive Fan-out System
 * 
 * Implements hardness-based query adaptation with configurable mapping
 * from query characteristics to k_candidates and gate thresholds.
 */

export interface HardnessFeatures {
  rare_terms: number;
  fuzzy_edits: number; 
  id_entropy: number;
  path_var: number;
  cand_slope: number;
}

export interface AdaptiveConfig {
  k_candidates: {
    min: number;
    max: number;
  };
  gate: {
    nl_threshold: {
      min: number; // 0.30
      max: number; // 0.55
    };
    min_candidates: {
      min: number; // 8
      max: number; // 14
    };
  };
  weights: {
    w1: number; // rare_terms
    w2: number; // fuzzy_edits
    w3: number; // id_entropy
    w4: number; // path_var
    w5: number; // cand_slope
  };
}

export const DEFAULT_ADAPTIVE_CONFIG: AdaptiveConfig = {
  k_candidates: {
    min: 180,
    max: 380
  },
  gate: {
    nl_threshold: {
      min: 0.30,
      max: 0.55
    },
    min_candidates: {
      min: 8,
      max: 14
    }
  },
  weights: {
    w1: 0.30, // rare_terms
    w2: 0.25, // fuzzy_edits  
    w3: 0.20, // id_entropy
    w4: 0.15, // path_var
    w5: 0.10  // cand_slope
  }
};

export class AdaptiveFanout {
  private config: AdaptiveConfig;
  private enabled: boolean = false;

  constructor(config: AdaptiveConfig = DEFAULT_ADAPTIVE_CONFIG) {
    this.config = config;
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  updateConfig(config: Partial<AdaptiveConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Calculate hardness score from query features
   * Returns h ∈ [0,1] where 1 = hardest queries
   */
  calculateHardness(features: HardnessFeatures): number {
    const { w1, w2, w3, w4, w5 } = this.config.weights;
    
    // Normalize features to [0,1] using z-score clipping
    const normRareTerms = this.clamp01(this.normalizeZScore(features.rare_terms, 2.0, 1.5));
    const normFuzzyEdits = this.clamp01(this.normalizeZScore(features.fuzzy_edits, 1.0, 0.8));
    const normIdEntropy = this.clamp01(this.normalizeZScore(features.id_entropy, 3.0, 1.2));
    const normPathVar = this.clamp01(this.normalizeZScore(features.path_var, 0.5, 0.3));
    const normCandSlope = this.clamp01(this.normalizeZScore(features.cand_slope, 0.8, 0.4));

    // Weighted sum
    const hardness = w1 * normRareTerms + 
                    w2 * normFuzzyEdits + 
                    w3 * normIdEntropy + 
                    w4 * normPathVar + 
                    w5 * normCandSlope;

    return this.clamp01(hardness);
  }

  /**
   * Map hardness to adaptive k_candidates
   */
  getAdaptiveKCandidates(hardness: number): number {
    const { min, max } = this.config.k_candidates;
    return Math.round(min + (max - min) * hardness);
  }

  /**
   * Map hardness to adaptive nl_threshold  
   */
  getAdaptiveNlThreshold(hardness: number): number {
    const { min, max } = this.config.gate.nl_threshold;
    return max - (max - min) * hardness; // Higher hardness = lower threshold
  }

  /**
   * Map hardness to adaptive min_candidates
   */
  getAdaptiveMinCandidates(hardness: number): number {
    const { min, max } = this.config.gate.min_candidates;
    return Math.round(min + (max - min) * hardness);
  }

  /**
   * Get all adaptive parameters for a given hardness score
   */
  getAdaptiveParameters(hardness: number) {
    return {
      k_candidates: this.getAdaptiveKCandidates(hardness),
      nl_threshold: this.getAdaptiveNlThreshold(hardness),
      min_candidates: this.getAdaptiveMinCandidates(hardness),
      hardness_score: hardness
    };
  }

  /**
   * Extract hardness features from search context and query logs
   */
  extractFeatures(query: string, context: any): HardnessFeatures {
    // Simple feature extraction - can be enhanced with actual log analysis
    return {
      rare_terms: this.countRareTerms(query),
      fuzzy_edits: context.fuzzy_distance || 0,
      id_entropy: this.calculateIdentifierEntropy(query),
      path_var: this.calculatePathVariance(context.repo_sha || ''),
      cand_slope: 0.5 // Default - would be computed from Stage-A candidate distribution
    };
  }

  private clamp01(value: number): number {
    return Math.max(0, Math.min(1, value));
  }

  private normalizeZScore(value: number, mean: number, stddev: number): number {
    return (value - mean) / stddev;
  }

  private countRareTerms(query: string): number {
    // Count terms that are likely rare (length > 6, contain underscores, camelCase)
    const terms = query.toLowerCase().split(/\s+/);
    let rareCount = 0;

    for (const term of terms) {
      if (term.length > 6 || 
          term.includes('_') || 
          /[a-z][A-Z]/.test(term) || // camelCase
          /^\d+$/.test(term)) {
        rareCount++;
      }
    }

    return rareCount;
  }

  private calculateIdentifierEntropy(query: string): number {
    // Simple entropy calculation based on character distribution
    const chars = query.split('');
    const freq: { [key: string]: number } = {};
    
    chars.forEach(char => {
      freq[char] = (freq[char] || 0) + 1;
    });

    let entropy = 0;
    const length = chars.length;
    
    Object.values(freq).forEach(count => {
      const probability = count / length;
      if (probability > 0) {
        entropy -= probability * Math.log2(probability);
      }
    });

    return entropy;
  }

  private calculatePathVariance(_repoSha: string): number {
    // Placeholder - would analyze path distribution from repository structure
    return 0.5;
  }
}

export const globalAdaptiveFanout = new AdaptiveFanout();