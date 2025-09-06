/**
 * Loss Taxonomy Instrumentation  
 * Labels failed queries: {NO_SYM_COVERAGE, WRONG_ALIAS, PATH_MAP, USABILITY_INTENT, RANKING_ONLY}
 * Uses why, LSIF coverage, and project config diff for detailed failure analysis
 */

import type { 
  LossTaxonomy,
  Candidate,
  SearchContext,
  QueryIntent,
  LSPHint
} from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';

interface LossAnalysisResult {
  primary_loss_factor: keyof LossTaxonomy;
  loss_taxonomy: LossTaxonomy;
  detailed_analysis: {
    query_complexity: 'simple' | 'medium' | 'complex';
    symbol_coverage_gap: number;
    alias_resolution_accuracy: number;
    path_mapping_correctness: number;
    intent_classification_accuracy: number;
    ranking_quality_score: number;
  };
  contributing_factors: string[];
  recommendations: string[];
}

interface GroundTruthEntry {
  file_path: string;
  line: number;
  col: number;
  relevance: number;
  is_primary: boolean;
  symbol_name?: string;
  symbol_kind?: string;
}

export class LossTaxonomyAnalyzer {
  private lspHints: Map<string, LSPHint> = new Map();
  private projectConfig: any = null;

  constructor() {}

  /**
   * Load LSP hints for coverage analysis
   */
  loadLSPHints(hints: LSPHint[]): void {
    this.lspHints.clear();
    for (const hint of hints) {
      this.lspHints.set(hint.symbol_id, hint);
      // Also index by name for quick lookup
      this.lspHints.set(hint.name.toLowerCase(), hint);
    }
  }

  /**
   * Load project configuration for path mapping analysis
   */
  loadProjectConfig(config: any): void {
    this.projectConfig = config;
  }

  /**
   * Analyze loss factors for a failed or suboptimal query
   */
  analyzeLossFactors(
    query: string,
    queryIntent: QueryIntent,
    results: Candidate[],
    groundTruth: GroundTruthEntry[],
    context: SearchContext
  ): LossAnalysisResult {
    const span = LensTracer.createChildSpan('loss_taxonomy_analysis', {
      'query': query,
      'intent': queryIntent,
      'results.count': results.length,
      'ground_truth.count': groundTruth.length,
    });

    try {
      const lossTaxonomy = this.calculateLossTaxonomy(
        query,
        queryIntent,
        results,
        groundTruth,
        context
      );

      const detailedAnalysis = this.performDetailedAnalysis(
        query,
        queryIntent,
        results,
        groundTruth,
        context
      );

      const primaryLossFactor = this.identifyPrimaryLossFactor(lossTaxonomy);
      const contributingFactors = this.extractContributingFactors(results, context);
      const recommendations = this.generateRecommendations(lossTaxonomy, detailedAnalysis);

      const result: LossAnalysisResult = {
        primary_loss_factor: primaryLossFactor,
        loss_taxonomy: lossTaxonomy,
        detailed_analysis: detailedAnalysis,
        contributing_factors: contributingFactors,
        recommendations: recommendations,
      };

      span.setAttributes({
        success: true,
        primary_loss_factor: primaryLossFactor,
        no_sym_coverage: lossTaxonomy.NO_SYM_COVERAGE,
        wrong_alias: lossTaxonomy.WRONG_ALIAS,
        path_map: lossTaxonomy.PATH_MAP,
        usability_intent: lossTaxonomy.USABILITY_INTENT,
        ranking_only: lossTaxonomy.RANKING_ONLY,
      });

      return result;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Calculate loss taxonomy scores
   */
  private calculateLossTaxonomy(
    query: string,
    queryIntent: QueryIntent,
    results: Candidate[],
    groundTruth: GroundTruthEntry[],
    context: SearchContext
  ): LossTaxonomy {
    return {
      NO_SYM_COVERAGE: this.assessSymbolCoverageGap(query, queryIntent, results, groundTruth),
      WRONG_ALIAS: this.assessAliasResolutionIssues(query, results, groundTruth),
      PATH_MAP: this.assessPathMappingIssues(results, groundTruth),
      USABILITY_INTENT: this.assessIntentClassificationIssues(queryIntent, results, context),
      RANKING_ONLY: this.assessRankingOnlyIssues(results, groundTruth),
    };
  }

  /**
   * Assess symbol coverage gaps
   */
  private assessSymbolCoverageGap(
    query: string,
    queryIntent: QueryIntent,
    results: Candidate[],
    groundTruth: GroundTruthEntry[]
  ): number {
    // Only relevant for symbol-related intents
    if (!['def', 'refs', 'symbol'].includes(queryIntent)) {
      return 0;
    }

    // Check if we have any symbol results for symbol queries
    const hasSymbolResults = results.some(r => 
      r.match_reasons.includes('symbol') || r.match_reasons.includes('lsp_hint')
    );

    if (!hasSymbolResults) {
      // Check if LSP hints contain relevant symbols
      const queryTerms = this.extractQueryTerms(query);
      const hasLSPCoverage = queryTerms.some(term => 
        this.lspHints.has(term.toLowerCase())
      );

      // If LSP has coverage but we didn't find symbols, it's a coverage gap
      return hasLSPCoverage ? 0.5 : 1.0;
    }

    // Check if we found the right kind of symbols
    const relevantSymbolsFound = results.filter(r => 
      groundTruth.some(gt => 
        Math.abs(gt.line - r.line) <= 2 && 
        gt.file_path === r.file_path &&
        (r.match_reasons.includes('symbol') || r.match_reasons.includes('lsp_hint'))
      )
    ).length;

    const totalRelevantSymbols = groundTruth.filter(gt => gt.symbol_name).length;
    
    if (totalRelevantSymbols === 0) return 0;
    
    const coverageRatio = relevantSymbolsFound / totalRelevantSymbols;
    return Math.max(0, 1 - coverageRatio);
  }

  /**
   * Assess alias resolution issues
   */
  private assessAliasResolutionIssues(
    query: string,
    results: Candidate[],
    groundTruth: GroundTruthEntry[]
  ): number {
    // Check for results that came from alias resolution
    const aliasResults = results.filter(r => 
      r.match_reasons.includes('symbol') || 
      (r as any).lsp_features?.alias_resolved
    );

    if (aliasResults.length === 0) return 0;

    // Check how many alias results are actually relevant
    const correctAliasResults = aliasResults.filter(r =>
      groundTruth.some(gt =>
        Math.abs(gt.line - r.line) <= 2 && gt.file_path === r.file_path
      )
    ).length;

    const aliasAccuracy = correctAliasResults / aliasResults.length;
    return Math.max(0, 1 - aliasAccuracy);
  }

  /**
   * Assess path mapping issues
   */
  private assessPathMappingIssues(
    results: Candidate[],
    groundTruth: GroundTruthEntry[]
  ): number {
    if (results.length === 0 || groundTruth.length === 0) return 0;

    // Check if results are in expected directories/modules
    const expectedPaths = new Set(
      groundTruth.map(gt => this.extractDirectoryPath(gt.file_path))
    );
    
    const resultPaths = new Set(
      results.map(r => this.extractDirectoryPath(r.file_path))
    );

    // Calculate path overlap
    const pathIntersection = new Set([...expectedPaths].filter(x => resultPaths.has(x)));
    const pathUnion = new Set([...expectedPaths, ...resultPaths]);
    
    const pathOverlap = pathIntersection.size / Math.max(expectedPaths.size, 1);
    return Math.max(0, 1 - pathOverlap);
  }

  /**
   * Assess intent classification issues
   */
  private assessIntentClassificationIssues(
    queryIntent: QueryIntent,
    results: Candidate[],
    context: SearchContext
  ): number {
    // Check if intent was honored in the search process
    const intentHonored = results.some((r: any) => 
      r.intent_honored !== false
    );

    if (!intentHonored) return 1.0;

    // Check confidence of intent classification
    const intentClassification = (results[0] as any)?.intent_classification;
    if (intentClassification) {
      const correctIntent = intentClassification.intent === queryIntent;
      const highConfidence = intentClassification.confidence > 0.7;
      
      if (!correctIntent && highConfidence) {
        return 0.8; // Wrong intent with high confidence
      } else if (!correctIntent) {
        return 0.5; // Wrong intent with low confidence
      }
    }

    return 0;
  }

  /**
   * Assess ranking-only issues
   */
  private assessRankingOnlyIssues(
    results: Candidate[],
    groundTruth: GroundTruthEntry[]
  ): number {
    if (results.length === 0) return 0;

    // Find positions of relevant results
    const relevantPositions: number[] = [];
    
    for (let i = 0; i < results.length; i++) {
      const candidate = results[i];
      const isRelevant = groundTruth.some(gt =>
        gt.file_path === candidate.file_path &&
        Math.abs(gt.line - candidate.line) <= 2 &&
        gt.relevance > 0
      );
      
      if (isRelevant) {
        relevantPositions.push(i + 1); // 1-indexed
      }
    }

    if (relevantPositions.length === 0) return 0;

    // If we have relevant results but they're not in top positions
    const bestPosition = Math.min(...relevantPositions);
    if (bestPosition > 10) {
      return 1.0; // Relevant results exist but are ranked too low
    } else if (bestPosition > 5) {
      return 0.7;
    } else if (bestPosition > 3) {
      return 0.4;
    }

    return 0;
  }

  /**
   * Perform detailed analysis of query characteristics and gaps
   */
  private performDetailedAnalysis(
    query: string,
    queryIntent: QueryIntent,
    results: Candidate[],
    groundTruth: GroundTruthEntry[],
    context: SearchContext
  ): LossAnalysisResult['detailed_analysis'] {
    return {
      query_complexity: this.assessQueryComplexity(query),
      symbol_coverage_gap: this.calculateSymbolCoverageGap(query, results, groundTruth),
      alias_resolution_accuracy: this.calculateAliasAccuracy(results, groundTruth),
      path_mapping_correctness: this.calculatePathMappingCorrectness(results, groundTruth),
      intent_classification_accuracy: this.calculateIntentAccuracy(queryIntent, results),
      ranking_quality_score: this.calculateRankingQuality(results, groundTruth),
    };
  }

  /**
   * Assess query complexity
   */
  private assessQueryComplexity(query: string): 'simple' | 'medium' | 'complex' {
    const tokens = this.extractQueryTerms(query);
    const hasStructuralChars = /[{}[\]()<>=!&|+\-*/^%~]/.test(query);
    const hasNaturalLanguage = query.split(' ').length > 3;
    
    if (hasStructuralChars || hasNaturalLanguage) {
      return 'complex';
    } else if (tokens.length > 2) {
      return 'medium';
    } else {
      return 'simple';
    }
  }

  /**
   * Calculate symbol coverage gap percentage
   */
  private calculateSymbolCoverageGap(
    query: string,
    results: Candidate[],
    groundTruth: GroundTruthEntry[]
  ): number {
    const queryTerms = this.extractQueryTerms(query);
    const symbolsFound = new Set<string>();
    
    for (const result of results) {
      if (result.symbol_kind) {
        symbolsFound.add(result.context || '');
      }
    }
    
    const expectedSymbols = groundTruth.filter(gt => gt.symbol_name).length;
    const foundSymbols = symbolsFound.size;
    
    return expectedSymbols === 0 ? 0 : Math.max(0, 1 - foundSymbols / expectedSymbols);
  }

  /**
   * Calculate alias resolution accuracy
   */
  private calculateAliasAccuracy(results: Candidate[], groundTruth: GroundTruthEntry[]): number {
    const aliasResults = results.filter(r => 
      r.match_reasons.includes('symbol') || (r as any).lsp_features?.alias_resolved
    );
    
    if (aliasResults.length === 0) return 1.0;
    
    const correctAliases = aliasResults.filter(r =>
      groundTruth.some(gt => 
        Math.abs(gt.line - r.line) <= 2 && gt.file_path === r.file_path
      )
    ).length;
    
    return correctAliases / aliasResults.length;
  }

  /**
   * Calculate path mapping correctness
   */
  private calculatePathMappingCorrectness(
    results: Candidate[],
    groundTruth: GroundTruthEntry[]
  ): number {
    if (results.length === 0 || groundTruth.length === 0) return 1.0;
    
    const correctPaths = results.filter(r =>
      groundTruth.some(gt => gt.file_path === r.file_path)
    ).length;
    
    return correctPaths / results.length;
  }

  /**
   * Calculate intent classification accuracy
   */
  private calculateIntentAccuracy(queryIntent: QueryIntent, results: Candidate[]): number {
    const intentClassifications = results
      .map((r: any) => r.intent_classification)
      .filter(ic => ic);
    
    if (intentClassifications.length === 0) return 0.5; // No classification data
    
    const correctClassifications = intentClassifications.filter(ic =>
      ic.intent === queryIntent
    ).length;
    
    return correctClassifications / intentClassifications.length;
  }

  /**
   * Calculate ranking quality score (NDCG-like)
   */
  private calculateRankingQuality(results: Candidate[], groundTruth: GroundTruthEntry[]): number {
    if (results.length === 0) return 0;
    
    let dcg = 0;
    let idcg = 0;
    
    // Calculate DCG
    for (let i = 0; i < Math.min(results.length, 10); i++) {
      const candidate = results[i];
      const relevance = groundTruth.find(gt =>
        gt.file_path === candidate.file_path &&
        Math.abs(gt.line - candidate.line) <= 2
      )?.relevance || 0;
      
      dcg += (Math.pow(2, relevance) - 1) / Math.log2(i + 2);
    }
    
    // Calculate IDCG
    const sortedRelevances = groundTruth
      .map(gt => gt.relevance)
      .sort((a, b) => b - a)
      .slice(0, 10);
    
    for (let i = 0; i < sortedRelevances.length; i++) {
      idcg += (Math.pow(2, sortedRelevances[i]) - 1) / Math.log2(i + 2);
    }
    
    return idcg === 0 ? 0 : dcg / idcg;
  }

  /**
   * Extract contributing factors from results and context
   */
  private extractContributingFactors(results: Candidate[], context: SearchContext): string[] {
    const factors: string[] = [];
    
    // Check why arrays from candidates
    const whyReasons = new Set<string>();
    for (const result of results) {
      const why = (result as any).why;
      if (Array.isArray(why)) {
        why.forEach(reason => whyReasons.add(reason));
      }
    }
    
    factors.push(...Array.from(whyReasons));
    
    // Add context-based factors
    if (context.stages.some(s => s.error)) {
      factors.push('stage_errors');
    }
    
    if (context.stages.some(s => s.latency_ms > 1000)) {
      factors.push('high_latency');
    }
    
    return factors;
  }

  /**
   * Generate recommendations based on loss analysis
   */
  private generateRecommendations(
    lossTaxonomy: LossTaxonomy,
    detailedAnalysis: LossAnalysisResult['detailed_analysis']
  ): string[] {
    const recommendations: string[] = [];
    
    if (lossTaxonomy.NO_SYM_COVERAGE > 0.5) {
      recommendations.push('Improve LSP symbol harvesting coverage');
      recommendations.push('Check workspace configuration and indexing completeness');
    }
    
    if (lossTaxonomy.WRONG_ALIAS > 0.5) {
      recommendations.push('Improve alias resolution accuracy');
      recommendations.push('Review import/alias mapping in workspace config');
    }
    
    if (lossTaxonomy.PATH_MAP > 0.5) {
      recommendations.push('Fix path mapping configuration');
      recommendations.push('Review tsconfig.json/pyproject.toml path mappings');
    }
    
    if (lossTaxonomy.USABILITY_INTENT > 0.5) {
      recommendations.push('Improve intent classification model');
      recommendations.push('Add more training data for query intent patterns');
    }
    
    if (lossTaxonomy.RANKING_ONLY > 0.5) {
      recommendations.push('Improve ranking algorithm');
      recommendations.push('Boost relevance signals for symbol matches');
    }
    
    // Analysis-based recommendations
    if (detailedAnalysis.query_complexity === 'complex' && detailedAnalysis.ranking_quality_score < 0.5) {
      recommendations.push('Add specialized handling for complex queries');
    }
    
    if (detailedAnalysis.intent_classification_accuracy < 0.7) {
      recommendations.push('Retrain intent classification with domain-specific patterns');
    }
    
    return recommendations;
  }

  /**
   * Identify the primary loss factor
   */
  private identifyPrimaryLossFactor(lossTaxonomy: LossTaxonomy): keyof LossTaxonomy {
    const factors = Object.entries(lossTaxonomy) as Array<[keyof LossTaxonomy, number]>;
    factors.sort((a, b) => b[1] - a[1]);
    return factors[0][0];
  }

  /**
   * Extract query terms for analysis
   */
  private extractQueryTerms(query: string): string[] {
    return query
      .toLowerCase()
      .split(/[\s._\-:(){}[\]<>=!&|+\-*/^%~]+/)
      .filter(term => term.length > 1);
  }

  /**
   * Extract directory path from file path
   */
  private extractDirectoryPath(filePath: string): string {
    const parts = filePath.split('/');
    return parts.slice(0, -1).join('/');
  }

  /**
   * Get loss taxonomy statistics
   */
  getStats(): {
    lsp_hints_loaded: number;
    project_config_loaded: boolean;
    loss_factors: (keyof LossTaxonomy)[];
  } {
    return {
      lsp_hints_loaded: this.lspHints.size,
      project_config_loaded: this.projectConfig !== null,
      loss_factors: ['NO_SYM_COVERAGE', 'WRONG_ALIAS', 'PATH_MAP', 'USABILITY_INTENT', 'RANKING_ONLY'],
    };
  }
}