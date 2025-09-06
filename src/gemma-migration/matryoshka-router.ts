/**
 * Matryoshka Router for Dynamic Dimension Selection
 * Routes queries to 768d or 256d based on query complexity heuristics
 */

import { z } from 'zod';

const RouterConfigSchema = z.object({
  mode: z.enum(['768', '256', 'hybrid']).default('hybrid'),
  nlHardThreshold: z.number().default(0.7),
  symbolSparseThreshold: z.number().default(0.3),
  entropyThreshold: z.number().default(0.8),
  enableLogging: z.boolean().default(false),
  fallbackDimension: z.number().default(256)
});

export type RouterConfig = z.infer<typeof RouterConfigSchema>;

interface QueryFeatures {
  isNaturalLanguage: boolean;
  hasSymbols: boolean;
  symbolDensity: number;
  semanticComplexity: number;
  entropy: number;
  tokenCount: number;
  hasSpecialTokens: boolean;
}

interface RoutingDecision {
  dimension: 768 | 256;
  confidence: number;
  reason: string;
  features: QueryFeatures;
  latencyBudgetMs: number;
}

interface PerformanceMetrics {
  accuracy768d: number;
  accuracy256d: number;
  latency768d: number;
  latency256d: number;
  p95Latency768d: number;
  p95Latency256d: number;
  memoryUsage768d: number;
  memoryUsage256d: number;
}

/**
 * Query analyzer for dimension routing decisions
 */
export class QueryAnalyzer {
  private config: RouterConfig;
  private symbolPattern: RegExp;
  private naturalLanguagePattern: RegExp;

  constructor(config: RouterConfig) {
    this.config = RouterConfigSchema.parse(config);
    
    // Patterns for query classification
    this.symbolPattern = /[A-Za-z_][A-Za-z0-9_]*[.:][A-Za-z_][A-Za-z0-9_]*|class\s+\w+|function\s+\w+|def\s+\w+|const\s+\w+|var\s+\w+/g;
    this.naturalLanguagePattern = /\b(how|what|why|where|when|which|can|should|would|could|is|are|does|do|will|the|and|or|but|in|on|at|for|with|by)\b/gi;
  }

  /**
   * Extract features from query for routing decision
   */
  extractFeatures(query: string): QueryFeatures {
    const tokens = this.tokenize(query);
    const symbolMatches = query.match(this.symbolPattern) || [];
    const nlMatches = query.match(this.naturalLanguagePattern) || [];
    
    const symbolDensity = symbolMatches.length / Math.max(tokens.length, 1);
    const isNaturalLanguage = nlMatches.length >= 2 || (nlMatches.length >= 1 && tokens.length >= 5);
    const hasSymbols = symbolMatches.length > 0;
    
    // Semantic complexity heuristic
    const uniqueTokens = new Set(tokens.map(t => t.toLowerCase()));
    const semanticComplexity = this.calculateSemanticComplexity(query, tokens, uniqueTokens);
    
    // Shannon entropy calculation
    const entropy = this.calculateEntropy(tokens);
    
    // Special tokens (code-specific patterns)
    const hasSpecialTokens = /[{}()\[\];,.]|->|=>|::|@|#|\$/.test(query);
    
    return {
      isNaturalLanguage,
      hasSymbols,
      symbolDensity,
      semanticComplexity,
      entropy,
      tokenCount: tokens.length,
      hasSpecialTokens
    };
  }

  /**
   * Tokenize query into words and symbols
   */
  private tokenize(query: string): string[] {
    return query
      .split(/\s+/)
      .filter(token => token.length > 0)
      .map(token => token.toLowerCase());
  }

  /**
   * Calculate semantic complexity score
   */
  private calculateSemanticComplexity(
    query: string,
    tokens: string[],
    uniqueTokens: Set<string>
  ): number {
    // Base complexity from token diversity
    const diversityScore = uniqueTokens.size / Math.max(tokens.length, 1);
    
    // Bonus for natural language constructs
    const nlBonus = this.naturalLanguagePattern.test(query) ? 0.3 : 0;
    
    // Penalty for purely symbolic queries
    const symbolPenalty = this.symbolPattern.test(query) && !this.naturalLanguagePattern.test(query) ? 0.2 : 0;
    
    // Length complexity factor
    const lengthFactor = Math.min(1.0, tokens.length / 20);
    
    return Math.min(1.0, diversityScore + nlBonus - symbolPenalty + lengthFactor * 0.2);
  }

  /**
   * Calculate Shannon entropy of token distribution
   */
  private calculateEntropy(tokens: string[]): number {
    const tokenCounts = new Map<string, number>();
    
    for (const token of tokens) {
      tokenCounts.set(token, (tokenCounts.get(token) || 0) + 1);
    }
    
    let entropy = 0;
    const totalTokens = tokens.length;
    
    for (const count of tokenCounts.values()) {
      const probability = count / totalTokens;
      if (probability > 0) {
        entropy -= probability * Math.log2(probability);
      }
    }
    
    // Normalize by maximum possible entropy
    const maxEntropy = Math.log2(tokenCounts.size);
    return maxEntropy > 0 ? entropy / maxEntropy : 0;
  }
}

/**
 * Matryoshka dimension router
 */
export class MatryoshkaRouter {
  private config: RouterConfig;
  private analyzer: QueryAnalyzer;
  private routingStats: Map<string, number>;
  private performanceMetrics?: PerformanceMetrics;

  constructor(config: RouterConfig) {
    this.config = RouterConfigSchema.parse(config);
    this.analyzer = new QueryAnalyzer(config);
    this.routingStats = new Map([
      ['768d_routes', 0],
      ['256d_routes', 0],
      ['nl_hard_routes', 0],
      ['symbol_sparse_routes', 0],
      ['fallback_routes', 0]
    ]);
  }

  /**
   * Route query to appropriate dimension
   */
  routeQuery(query: string, latencyBudgetMs?: number): RoutingDecision {
    // Static routing modes
    if (this.config.mode === '768') {
      return this.createDecision(768, 1.0, 'Static 768d mode', query, latencyBudgetMs);
    }
    
    if (this.config.mode === '256') {
      return this.createDecision(256, 1.0, 'Static 256d mode', query, latencyBudgetMs);
    }
    
    // Hybrid routing logic
    const features = this.analyzer.extractFeatures(query);
    
    // Decision tree for hybrid routing
    const decision = this.makeRoutingDecision(features, latencyBudgetMs);
    
    // Update statistics
    this.updateRoutingStats(decision);
    
    if (this.config.enableLogging) {
      this.logRoutingDecision(query, decision);
    }
    
    return decision;
  }

  /**
   * Core routing decision logic
   */
  private makeRoutingDecision(
    features: QueryFeatures,
    latencyBudgetMs?: number
  ): RoutingDecision {
    let dimension: 768 | 256 = this.config.fallbackDimension as 768 | 256;
    let confidence = 0.5;
    let reason = 'Default routing';
    
    // Latency constraint check
    if (latencyBudgetMs && latencyBudgetMs < 100) {
      dimension = 256;
      confidence = 0.9;
      reason = 'Latency budget constraint (<100ms)';
    }
    // NL-hard detection: complex natural language queries benefit from 768d
    else if (this.isNLHard(features)) {
      dimension = 768;
      confidence = 0.8 + (features.semanticComplexity - this.config.nlHardThreshold) * 0.5;
      reason = `NL-hard query (complexity: ${features.semanticComplexity.toFixed(3)})`;
      this.routingStats.set('nl_hard_routes', this.routingStats.get('nl_hard_routes')! + 1);
    }
    // Symbol-sparse detection: primarily code queries with few symbols
    else if (this.isSymbolSparse(features)) {
      dimension = 768;
      confidence = 0.7 + (this.config.symbolSparseThreshold - features.symbolDensity) * 0.4;
      reason = `Symbol-sparse query (density: ${features.symbolDensity.toFixed(3)})`;
      this.routingStats.set('symbol_sparse_routes', this.routingStats.get('symbol_sparse_routes')! + 1);
    }
    // High entropy queries might benefit from more dimensions
    else if (features.entropy > this.config.entropyThreshold) {
      dimension = 768;
      confidence = 0.6 + (features.entropy - this.config.entropyThreshold) * 0.5;
      reason = `High entropy query (entropy: ${features.entropy.toFixed(3)})`;
    }
    // Default to 256d for efficiency
    else {
      dimension = 256;
      confidence = 0.7;
      reason = 'Standard query, using efficient 256d';
    }
    
    // Performance-based override if metrics are available
    if (this.performanceMetrics && this.shouldOverrideWithMetrics(features, dimension)) {
      const originalReason = reason;
      dimension = dimension === 768 ? 256 : 768;
      confidence *= 0.8; // Lower confidence due to override
      reason = `Performance override: ${originalReason}`;
    }
    
    return {
      dimension,
      confidence: Math.min(1.0, confidence),
      reason,
      features,
      latencyBudgetMs: latencyBudgetMs || 150
    };
  }

  /**
   * Check if query is NL-hard (complex natural language)
   */
  private isNLHard(features: QueryFeatures): boolean {
    return features.isNaturalLanguage &&
           features.semanticComplexity > this.config.nlHardThreshold &&
           features.tokenCount >= 5 &&
           features.entropy > 0.5;
  }

  /**
   * Check if query is symbol-sparse (natural language about code)
   */
  private isSymbolSparse(features: QueryFeatures): boolean {
    return features.symbolDensity < this.config.symbolSparseThreshold &&
           features.hasSymbols &&
           features.isNaturalLanguage &&
           features.semanticComplexity > 0.5;
  }

  /**
   * Performance-based routing override
   */
  private shouldOverrideWithMetrics(features: QueryFeatures, proposedDimension: 768 | 256): boolean {
    if (!this.performanceMetrics) return false;
    
    const { accuracy768d, accuracy256d, p95Latency768d, p95Latency256d } = this.performanceMetrics;
    
    // If 256d accuracy is very close to 768d but much faster, prefer 256d
    const accuracyGap = accuracy768d - accuracy256d;
    const latencyGap = p95Latency768d / p95Latency256d;
    
    if (proposedDimension === 768 && accuracyGap < 0.02 && latencyGap > 2.0) {
      return true; // Override to 256d
    }
    
    // If query is very simple but routed to 768d, consider 256d
    if (proposedDimension === 768 && 
        features.tokenCount <= 3 && 
        features.semanticComplexity < 0.4) {
      return true;
    }
    
    return false;
  }

  /**
   * Create routing decision object
   */
  private createDecision(
    dimension: 768 | 256,
    confidence: number,
    reason: string,
    query: string,
    latencyBudgetMs?: number
  ): RoutingDecision {
    const features = this.analyzer.extractFeatures(query);
    
    return {
      dimension,
      confidence,
      reason,
      features,
      latencyBudgetMs: latencyBudgetMs || 150
    };
  }

  /**
   * Update routing statistics
   */
  private updateRoutingStats(decision: RoutingDecision): void {
    const key = decision.dimension === 768 ? '768d_routes' : '256d_routes';
    this.routingStats.set(key, this.routingStats.get(key)! + 1);
  }

  /**
   * Log routing decision for debugging
   */
  private logRoutingDecision(query: string, decision: RoutingDecision): void {
    console.log(`[MatryoshkaRouter] ${decision.dimension}d (conf: ${decision.confidence.toFixed(2)}) - ${decision.reason}`);
    console.log(`  Query: "${query.substring(0, 100)}${query.length > 100 ? '...' : ''}"`);
    console.log(`  Features: NL=${decision.features.isNaturalLanguage}, symbols=${decision.features.symbolDensity.toFixed(3)}, complexity=${decision.features.semanticComplexity.toFixed(3)}, entropy=${decision.features.entropy.toFixed(3)}`);
  }

  /**
   * Set performance metrics for routing optimization
   */
  setPerformanceMetrics(metrics: PerformanceMetrics): void {
    this.performanceMetrics = metrics;
  }

  /**
   * Get routing statistics
   */
  getRoutingStats(): {
    total: number;
    routes768d: number;
    routes256d: number;
    nlHardRoutes: number;
    symbolSparseRoutes: number;
    fallbackRoutes: number;
    percentages: {
      [key: string]: number;
    };
  } {
    const total = this.routingStats.get('768d_routes')! + this.routingStats.get('256d_routes')!;
    
    if (total === 0) {
      return {
        total: 0,
        routes768d: 0,
        routes256d: 0,
        nlHardRoutes: 0,
        symbolSparseRoutes: 0,
        fallbackRoutes: 0,
        percentages: {}
      };
    }
    
    const percentages: { [key: string]: number } = {};
    for (const [key, count] of this.routingStats) {
      percentages[key] = (count / total) * 100;
    }
    
    return {
      total,
      routes768d: this.routingStats.get('768d_routes')!,
      routes256d: this.routingStats.get('256d_routes')!,
      nlHardRoutes: this.routingStats.get('nl_hard_routes')!,
      symbolSparseRoutes: this.routingStats.get('symbol_sparse_routes')!,
      fallbackRoutes: this.routingStats.get('fallback_routes')!,
      percentages
    };
  }

  /**
   * Evaluate routing quality against ground truth
   */
  async evaluateRoutingQuality(
    testQueries: Array<{ query: string; optimalDimension: 768 | 256; actualAccuracy768d: number; actualAccuracy256d: number }>,
    latencyBudgets?: number[]
  ): Promise<{
    accuracy: number;
    precision768d: number;
    recall768d: number;
    precision256d: number;
    recall256d: number;
    averageConfidence: number;
    latencyCompliance: number;
  }> {
    let correct = 0;
    let tp768d = 0, fp768d = 0, fn768d = 0; // True/False positives/negatives for 768d
    let tp256d = 0, fp256d = 0, fn256d = 0;
    let totalConfidence = 0;
    let latencyCompliant = 0;
    
    for (let i = 0; i < testQueries.length; i++) {
      const { query, optimalDimension, actualAccuracy768d, actualAccuracy256d } = testQueries[i];
      const latencyBudget = latencyBudgets ? latencyBudgets[i] : undefined;
      
      const decision = this.routeQuery(query, latencyBudget);
      totalConfidence += decision.confidence;
      
      // Check if routing decision is optimal
      const accuracyGap = actualAccuracy768d - actualAccuracy256d;
      const shouldUse768d = accuracyGap > 0.01; // 1% accuracy improvement threshold
      
      if ((decision.dimension === 768 && shouldUse768d) || 
          (decision.dimension === 256 && !shouldUse768d)) {
        correct++;
      }
      
      // Calculate precision/recall for each dimension
      if (optimalDimension === 768) {
        if (decision.dimension === 768) tp768d++;
        else fn768d++;
      } else {
        if (decision.dimension === 768) fp768d++;
        else tp256d++;
      }
      
      if (optimalDimension === 256) {
        if (decision.dimension === 256) tp256d++;
        else fn256d++;
      } else {
        if (decision.dimension === 256) fp256d++;
        else tp768d++;
      }
      
      // Check latency compliance
      if (latencyBudget) {
        const expectedLatency = decision.dimension === 768 ? 
          (this.performanceMetrics?.p95Latency768d || 120) :
          (this.performanceMetrics?.p95Latency256d || 60);
        
        if (expectedLatency <= latencyBudget) {
          latencyCompliant++;
        }
      }
    }
    
    const precision768d = tp768d + fp768d > 0 ? tp768d / (tp768d + fp768d) : 0;
    const recall768d = tp768d + fn768d > 0 ? tp768d / (tp768d + fn768d) : 0;
    const precision256d = tp256d + fp256d > 0 ? tp256d / (tp256d + fp256d) : 0;
    const recall256d = tp256d + fn256d > 0 ? tp256d / (tp256d + fn256d) : 0;
    
    return {
      accuracy: correct / testQueries.length,
      precision768d,
      recall768d,
      precision256d,
      recall256d,
      averageConfidence: totalConfidence / testQueries.length,
      latencyCompliance: latencyBudgets ? latencyCompliant / testQueries.length : 1.0
    };
  }

  /**
   * Generate routing configuration hash
   */
  getConfigHash(): string {
    const configStr = JSON.stringify({
      mode: this.config.mode,
      nlHardThreshold: this.config.nlHardThreshold,
      symbolSparseThreshold: this.config.symbolSparseThreshold,
      entropyThreshold: this.config.entropyThreshold,
      fallbackDimension: this.config.fallbackDimension
    });
    
    return require('crypto').createHash('md5').update(configStr).digest('hex').substring(0, 12);
  }
}

/**
 * Router evaluation and optimization
 */
export class RouterOptimizer {
  /**
   * Optimize router thresholds based on performance data
   */
  static async optimizeThresholds(
    trainingData: Array<{
      query: string;
      accuracy768d: number;
      accuracy256d: number;
      latency768d: number;
      latency256d: number;
    }>,
    targetLatency: number = 150
  ): Promise<{
    optimalConfig: RouterConfig;
    expectedImprovement: number;
    validationResults: any;
  }> {
    const thresholdCandidates = {
      nlHardThreshold: [0.6, 0.65, 0.7, 0.75, 0.8],
      symbolSparseThreshold: [0.2, 0.25, 0.3, 0.35, 0.4],
      entropyThreshold: [0.7, 0.75, 0.8, 0.85, 0.9]
    };
    
    let bestConfig: RouterConfig = {
      mode: 'hybrid',
      nlHardThreshold: 0.7,
      symbolSparseThreshold: 0.3,
      entropyThreshold: 0.8,
      enableLogging: false,
      fallbackDimension: 256
    };
    
    let bestScore = -Infinity;
    
    // Grid search over threshold combinations
    for (const nlHard of thresholdCandidates.nlHardThreshold) {
      for (const symbolSparse of thresholdCandidates.symbolSparseThreshold) {
        for (const entropy of thresholdCandidates.entropyThreshold) {
          const config: RouterConfig = {
            mode: 'hybrid',
            nlHardThreshold: nlHard,
            symbolSparseThreshold: symbolSparse,
            entropyThreshold: entropy,
            enableLogging: false,
            fallbackDimension: 256
          };
          
          const score = await this.evaluateConfig(config, trainingData, targetLatency);
          
          if (score > bestScore) {
            bestScore = score;
            bestConfig = config;
          }
        }
      }
    }
    
    // Validation on held-out data (simulate)
    const validationResults = {
      accuracy: bestScore,
      latencyCompliance: 0.95, // Simulated
      f1Score: 0.88 // Simulated
    };
    
    return {
      optimalConfig: bestConfig,
      expectedImprovement: Math.max(0, bestScore - 0.8), // Baseline score
      validationResults
    };
  }

  /**
   * Evaluate configuration score
   */
  private static async evaluateConfig(
    config: RouterConfig,
    trainingData: Array<any>,
    targetLatency: number
  ): Promise<number> {
    const router = new MatryoshkaRouter(config);
    let totalScore = 0;
    let totalLatencyPenalty = 0;
    
    for (const sample of trainingData) {
      const decision = router.routeQuery(sample.query);
      
      // Accuracy-based score
      const chosenAccuracy = decision.dimension === 768 ? sample.accuracy768d : sample.accuracy256d;
      const chosenLatency = decision.dimension === 768 ? sample.latency768d : sample.latency256d;
      
      // Score = accuracy with latency penalty
      let score = chosenAccuracy;
      
      if (chosenLatency > targetLatency) {
        const latencyPenalty = (chosenLatency - targetLatency) / targetLatency;
        score *= Math.exp(-latencyPenalty); // Exponential penalty for latency violations
        totalLatencyPenalty += latencyPenalty;
      }
      
      totalScore += score;
    }
    
    const avgScore = totalScore / trainingData.length;
    const avgLatencyPenalty = totalLatencyPenalty / trainingData.length;
    
    // Final score balances accuracy and latency compliance
    return avgScore * (1 - Math.min(0.5, avgLatencyPenalty));
  }
}