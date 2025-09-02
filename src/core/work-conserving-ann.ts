/**
 * Work-Conserving ANN with Guarded Early Exit
 * 
 * Implements dynamic HNSW depth based on remaining computational work
 * and early exit when calibrated score margin is decisive.
 */

export interface EarlyExitConfig {
  after_probes: number; // Check for exit after this many probes
  margin_tau: number; // Calibrated-score margin threshold
  guards: {
    require_symbol_or_struct: boolean; // Require symbol/struct in top candidates
    min_top1_top5_margin: number; // Minimum margin between top-1 and top-5
  };
}

export interface WorkConservingConfig {
  k: number; // Target result count (220 default)
  efSearch_base: number; // Base efSearch parameter (48 default)  
  efSearch_scaling: number; // Scaling factor (24 default)
  early_exit: EarlyExitConfig;
}

export const DEFAULT_WORK_CONSERVING_CONFIG: WorkConservingConfig = {
  k: 220,
  efSearch_base: 48,
  efSearch_scaling: 24,
  early_exit: {
    after_probes: 64,
    margin_tau: 0.07,
    guards: {
      require_symbol_or_struct: true,
      min_top1_top5_margin: 0.14
    }
  }
};

export interface Candidate {
  score: number;
  file: string;
  line: number;
  snippet: string;
  why: string;
  // Additional fields as needed
}

export class WorkConservingANN {
  private config: WorkConservingConfig;
  private enabled: boolean = false;

  constructor(config: WorkConservingConfig = DEFAULT_WORK_CONSERVING_CONFIG) {
    this.config = config;
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  updateConfig(config: Partial<WorkConservingConfig>): void {
    this.config = { 
      ...this.config, 
      ...config,
      early_exit: {
        ...this.config.early_exit,
        ...config.early_exit,
        guards: {
          ...this.config.early_exit.guards,
          ...config.early_exit?.guards
        }
      }
    };
  }

  /**
   * Calculate dynamic efSearch based on candidate count
   * Formula: efSearch = base + scaling * log2(1 + |candidates|/150)
   */
  calculateDynamicEfSearch(candidateCount: number): number {
    const { efSearch_base, efSearch_scaling } = this.config;
    const scalingFactor = Math.log2(1 + candidateCount / 150);
    const efSearch = efSearch_base + efSearch_scaling * scalingFactor;
    
    // Clamp to reasonable bounds
    return Math.round(Math.max(16, Math.min(512, efSearch)));
  }

  /**
   * Perform work-conserving ANN search with early exit
   */
  async search(candidates: Candidate[], query: string): Promise<Candidate[]> {
    if (!this.enabled || candidates.length === 0) {
      return candidates.slice(0, this.config.k);
    }

    const dynamicEfSearch = this.calculateDynamicEfSearch(candidates.length);
    console.log(`🔍 Work-conserving ANN: candidates=${candidates.length}, efSearch=${dynamicEfSearch}`);

    // Simulate HNSW-style search with probes and early exit checks
    let processedCandidates = [...candidates];
    let probesCount = 0;
    const maxProbes = Math.min(dynamicEfSearch * 2, candidates.length);
    
    // Sort by initial score for processing order
    processedCandidates.sort((a, b) => b.score - a.score);

    // Process candidates in batches, checking for early exit
    const batchSize = Math.max(8, Math.floor(maxProbes / 10));
    let bestCandidates: Candidate[] = [];

    for (let i = 0; i < maxProbes; i += batchSize) {
      const batch = processedCandidates.slice(i, Math.min(i + batchSize, maxProbes));
      
      // Process batch (simulate vector similarity computation)
      const batchScores = await this.simulateVectorSimilarity(batch, query);
      
      // Update candidates with refined scores
      batch.forEach((candidate, idx) => {
        candidate.score = batchScores[idx];
      });

      // Add to current best candidates and re-sort
      bestCandidates = [...bestCandidates, ...batch]
        .sort((a, b) => b.score - a.score)
        .slice(0, this.config.k);

      probesCount += batch.length;

      // Check early exit conditions after sufficient probes
      if (probesCount >= this.config.early_exit.after_probes) {
        if (this.shouldEarlyExit(bestCandidates, probesCount)) {
          console.log(`⚡ Early exit after ${probesCount} probes (margin decisive)`);
          break;
        }
      }
    }

    console.log(`🎯 Work-conserving ANN completed: ${probesCount} probes, ${bestCandidates.length} results`);
    return bestCandidates.slice(0, this.config.k);
  }

  /**
   * Check if early exit conditions are met
   */
  private shouldEarlyExit(candidates: Candidate[], probesCount: number): boolean {
    if (candidates.length < 5) {
      return false; // Need at least 5 candidates for margin check
    }

    const { margin_tau, guards } = this.config.early_exit;

    // Check calibrated score margin (top-1 vs top-2)
    const top1Score = candidates[0].score;
    const top2Score = candidates[1].score;
    const marginTop1Top2 = top1Score - top2Score;

    if (marginTop1Top2 < margin_tau) {
      return false; // Margin not decisive enough
    }

    // Guard: require symbol or struct in top results
    if (guards.require_symbol_or_struct) {
      const hasSymbolOrStruct = candidates.slice(0, 3).some(candidate => 
        candidate.why.includes('symbol') || 
        candidate.why.includes('struct') ||
        candidate.why.includes('semantic')
      );
      
      if (!hasSymbolOrStruct) {
        return false; // Safety guard failed
      }
    }

    // Guard: check top-1 to top-5 margin
    if (candidates.length >= 5) {
      const top5Score = candidates[4].score;
      const marginTop1Top5 = top1Score - top5Score;
      
      if (marginTop1Top5 < guards.min_top1_top5_margin) {
        return false; // Not confident enough in ranking
      }
    }

    return true; // All conditions met, can exit early
  }

  /**
   * Simulate vector similarity computation (placeholder for actual HNSW)
   */
  private async simulateVectorSimilarity(candidates: Candidate[], query: string): Promise<number[]> {
    // Placeholder: In real implementation, this would use actual vector embeddings
    // For now, simulate with query-aware scoring based on text similarity
    
    const queryTokens = query.toLowerCase().split(/\s+/);
    
    return candidates.map(candidate => {
      const snippetTokens = candidate.snippet.toLowerCase().split(/\s+/);
      
      // Simple token overlap score
      let overlapScore = 0;
      for (const token of queryTokens) {
        if (snippetTokens.some(snippetToken => 
            snippetToken.includes(token) || token.includes(snippetToken))) {
          overlapScore += 1;
        }
      }
      
      // Normalize and add some randomness to simulate vector computation variance
      const baseScore = overlapScore / Math.max(queryTokens.length, 1);
      const variance = (Math.random() - 0.5) * 0.1; // ±5% variance
      
      return Math.max(0, Math.min(1, baseScore + variance));
    });
  }
}

export const globalWorkConservingANN = new WorkConservingANN();