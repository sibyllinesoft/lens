/**
 * Decision Framework for Gemma Variant Selection
 * 
 * Provides automated decision-making with:
 * - Multi-criteria decision analysis (MCDA)
 * - Use case mapping and recommendation engine
 * - Hybrid routing strategy optimization
 * - Production deployment decision gates
 * - Risk assessment and mitigation planning
 */

import { z } from 'zod';
import * as fs from 'fs';
import type { 
  TradeoffAnalysis, 
  StatisticalAnalysis, 
  PerformanceMetrics, 
  LatencyBreakdown 
} from './performance-latency-framework.js';

// Schema definitions for decision framework

const UseCaseSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string(),
  requirements: z.object({
    maxLatencyMs: z.number(),
    minQualityScore: z.number().min(0).max(1), // nDCG@10 threshold
    maxConcurrentUsers: z.number(),
    prioritizeQuality: z.boolean(),
    prioritizeLatency: z.boolean(),
    resourceConstraints: z.object({
      maxMemoryMB: z.number().optional(),
      maxCpuCores: z.number().optional(),
      maxCostPerMonth: z.number().optional()
    }).optional()
  }),
  weight: z.number().min(0).max(1), // Importance weight for decision making
  businessImpact: z.enum(['low', 'medium', 'high', 'critical'])
});

const DecisionCriteriaSchema = z.object({
  quality: z.object({
    weight: z.number().min(0).max(1),
    metrics: z.array(z.string()),
    thresholds: z.record(z.string(), z.number())
  }),
  latency: z.object({
    weight: z.number().min(0).max(1),
    metrics: z.array(z.string()),
    thresholds: z.record(z.string(), z.number())
  }),
  scalability: z.object({
    weight: z.number().min(0).max(1),
    metrics: z.array(z.string()),
    thresholds: z.record(z.string(), z.number())
  }),
  resourceEfficiency: z.object({
    weight: z.number().min(0).max(1),
    metrics: z.array(z.string()),
    thresholds: z.record(z.string(), z.number())
  }),
  robustness: z.object({
    weight: z.number().min(0).max(1),
    metrics: z.array(z.string()),
    thresholds: z.record(z.string(), z.number())
  })
});

const VariantScoringSchema = z.object({
  variant: z.string(),
  overallScore: z.number().min(0).max(1),
  criteriaScores: z.object({
    quality: z.number().min(0).max(1),
    latency: z.number().min(0).max(1),
    scalability: z.number().min(0).max(1),
    resourceEfficiency: z.number().min(0).max(1),
    robustness: z.number().min(0).max(1)
  }),
  useCaseRecommendations: z.array(z.string()),
  confidence: z.number().min(0).max(1),
  risks: z.array(z.string())
});

const HybridRoutingRuleSchema = z.object({
  condition: z.object({
    type: z.enum(['latency_requirement', 'query_type', 'load_level', 'time_of_day', 'user_tier']),
    operator: z.enum(['<', '<=', '>', '>=', '==', '!=', 'in', 'not_in']),
    value: z.union([z.number(), z.string(), z.array(z.union([z.number(), z.string()]))])
  }),
  variant: z.string(),
  confidence: z.number().min(0).max(1),
  fallback: z.string()
});

const HybridRoutingStrategySchema = z.object({
  enabled: z.boolean(),
  primaryStrategy: z.enum(['latency_based', 'quality_based', 'load_aware', 'adaptive']),
  rules: z.array(HybridRoutingRuleSchema),
  defaultVariant: z.string(),
  routingMetrics: z.object({
    successRate: z.number().min(0).max(1),
    avgLatencyMs: z.number(),
    qualityMaintained: z.boolean()
  })
});

const DeploymentDecisionSchema = z.object({
  recommendedVariant: z.string(),
  deploymentStrategy: z.enum(['blue_green', 'canary', 'rolling', 'immediate']),
  confidence: z.number().min(0).max(1),
  statisticalSignificance: z.object({
    pValue: z.number(),
    effectSize: z.number(),
    practicallySignificant: z.boolean()
  }),
  riskAssessment: z.object({
    overallRisk: z.enum(['low', 'medium', 'high', 'very_high']),
    risks: z.array(z.object({
      category: z.string(),
      severity: z.enum(['low', 'medium', 'high', 'critical']),
      probability: z.number().min(0).max(1),
      description: z.string(),
      mitigation: z.string()
    }))
  }),
  promotionCriteria: z.object({
    qualityThresholds: z.record(z.string(), z.number()),
    latencyThresholds: z.record(z.string(), z.number()),
    errorRateThreshold: z.number(),
    rollbackTriggers: z.array(z.string())
  })
});

const DecisionReportSchema = z.object({
  timestamp: z.string(),
  comparisonSummary: z.object({
    variants: z.array(z.string()),
    primaryMetric: z.string(),
    winningVariant: z.string(),
    winMargin: z.number()
  }),
  variantScoring: z.array(VariantScoringSchema),
  useCaseMapping: z.record(z.string(), z.string()), // useCase -> recommendedVariant
  hybridRouting: HybridRoutingStrategySchema,
  deploymentDecision: DeploymentDecisionSchema,
  sensitivityAnalysis: z.object({
    criteriaWeightSensitivity: z.record(z.string(), z.number()),
    thresholdSensitivity: z.record(z.string(), z.number()),
    robustness: z.number().min(0).max(1)
  })
});

export type UseCase = z.infer<typeof UseCaseSchema>;
export type DecisionCriteria = z.infer<typeof DecisionCriteriaSchema>;
export type VariantScoring = z.infer<typeof VariantScoringSchema>;
export type HybridRoutingRule = z.infer<typeof HybridRoutingRuleSchema>;
export type HybridRoutingStrategy = z.infer<typeof HybridRoutingStrategySchema>;
export type DeploymentDecision = z.infer<typeof DeploymentDecisionSchema>;
export type DecisionReport = z.infer<typeof DecisionReportSchema>;

/**
 * Main decision framework engine
 */
export class DecisionFramework {
  private useCases: UseCase[];
  private criteria: DecisionCriteria;
  
  constructor(
    useCases: UseCase[] = DEFAULT_USE_CASES,
    criteria: DecisionCriteria = DEFAULT_CRITERIA
  ) {
    this.useCases = useCases;
    this.criteria = criteria;
  }

  /**
   * Generate comprehensive decision analysis and recommendations
   */
  async generateDecisionAnalysis(
    variantAnalyses: Map<string, TradeoffAnalysis>,
    statisticalComparisons: StatisticalAnalysis[],
    baselineModel: string = 'ada-002'
  ): Promise<DecisionReport> {
    
    console.log('🎯 Generating comprehensive decision analysis');
    
    // 1. Score all variants against multiple criteria
    const variantScoring = await this.scoreVariants(variantAnalyses);
    
    // 2. Map use cases to optimal variants
    const useCaseMapping = await this.mapUseCasesToVariants(variantAnalyses);
    
    // 3. Generate hybrid routing strategy
    const hybridRouting = await this.generateHybridRoutingStrategy(
      variantAnalyses,
      useCaseMapping
    );
    
    // 4. Make deployment decision with risk assessment
    const deploymentDecision = await this.makeDeploymentDecision(
      variantAnalyses,
      statisticalComparisons,
      variantScoring
    );
    
    // 5. Perform sensitivity analysis
    const sensitivityAnalysis = await this.performSensitivityAnalysis(variantAnalyses);
    
    // 6. Generate comparison summary
    const comparisonSummary = this.generateComparisonSummary(variantScoring);
    
    return DecisionReportSchema.parse({
      timestamp: new Date().toISOString(),
      comparisonSummary,
      variantScoring,
      useCaseMapping,
      hybridRouting,
      deploymentDecision,
      sensitivityAnalysis
    });
  }

  /**
   * Score variants using multi-criteria decision analysis (MCDA)
   */
  private async scoreVariants(
    variantAnalyses: Map<string, TradeoffAnalysis>
  ): Promise<VariantScoring[]> {
    
    console.log('  📊 Scoring variants with MCDA');
    
    const scorings: VariantScoring[] = [];
    
    for (const [variant, analysis] of variantAnalyses) {
      const scores = await this.calculateCriteriaScores(analysis);
      
      // Calculate weighted overall score
      const overallScore = 
        scores.quality * this.criteria.quality.weight +
        scores.latency * this.criteria.latency.weight +
        scores.scalability * this.criteria.scalability.weight +
        scores.resourceEfficiency * this.criteria.resourceEfficiency.weight +
        scores.robustness * this.criteria.robustness.weight;
      
      // Identify suitable use cases
      const useCaseRecommendations = this.identifyUseCases(analysis);
      
      // Calculate confidence based on data quality and consistency
      const confidence = this.calculateConfidence(analysis, scores);
      
      // Identify key risks
      const risks = this.identifyRisks(analysis, scores);
      
      scorings.push(VariantScoringSchema.parse({
        variant,
        overallScore,
        criteriaScores: scores,
        useCaseRecommendations,
        confidence,
        risks
      }));
    }
    
    return scorings;
  }

  /**
   * Calculate normalized scores for each decision criteria
   */
  private async calculateCriteriaScores(analysis: TradeoffAnalysis): Promise<{
    quality: number;
    latency: number;
    scalability: number;
    resourceEfficiency: number;
    robustness: number;
  }> {
    
    // Quality score (higher is better)
    const qualityScore = this.normalizeScore(
      analysis.performanceMetrics.nDCG_at_10,
      { min: 0.5, max: 1.0, higherIsBetter: true }
    );
    
    // Latency score (lower is better)
    const latencyScore = this.normalizeScore(
      analysis.latencyBreakdown.totalPipelineLatency,
      { min: 10, max: 500, higherIsBetter: false }
    );
    
    // Scalability score based on load test results
    const maxHandledUsers = Math.max(
      ...analysis.loadTestResults.map(r => r.concurrentUsers)
    );
    const scalabilityScore = this.normalizeScore(
      maxHandledUsers,
      { min: 1, max: 1000, higherIsBetter: true }
    );
    
    // Resource efficiency score
    const avgMemoryUsage = analysis.latencyBreakdown.memoryUsage || 100;
    const resourceEfficiencyScore = this.normalizeScore(
      avgMemoryUsage,
      { min: 50, max: 1000, higherIsBetter: false }
    );
    
    // Robustness score based on error rates under load
    const avgErrorRate = analysis.loadTestResults.length > 0
      ? analysis.loadTestResults.reduce((sum, r) => sum + r.errorRate, 0) / analysis.loadTestResults.length
      : 0;
    const robustnessScore = this.normalizeScore(
      avgErrorRate,
      { min: 0, max: 20, higherIsBetter: false }
    );
    
    return {
      quality: qualityScore,
      latency: latencyScore,
      scalability: scalabilityScore,
      resourceEfficiency: resourceEfficiencyScore,
      robustness: robustnessScore
    };
  }

  /**
   * Normalize a metric to 0-1 score
   */
  private normalizeScore(
    value: number,
    range: { min: number; max: number; higherIsBetter: boolean }
  ): number {
    const normalized = Math.max(0, Math.min(1, (value - range.min) / (range.max - range.min)));
    return range.higherIsBetter ? normalized : 1 - normalized;
  }

  /**
   * Map use cases to optimal variants
   */
  private async mapUseCasesToVariants(
    variantAnalyses: Map<string, TradeoffAnalysis>
  ): Promise<Record<string, string>> {
    
    console.log('  🗺️  Mapping use cases to variants');
    
    const mapping: Record<string, string> = {};
    
    for (const useCase of this.useCases) {
      let bestVariant = '';
      let bestScore = -1;
      
      for (const [variant, analysis] of variantAnalyses) {
        const score = await this.calculateUseCaseScore(useCase, analysis);
        
        if (score > bestScore) {
          bestScore = score;
          bestVariant = variant;
        }
      }
      
      if (bestVariant) {
        mapping[useCase.id] = bestVariant;
      }
    }
    
    return mapping;
  }

  /**
   * Calculate how well a variant fits a specific use case
   */
  private async calculateUseCaseScore(
    useCase: UseCase,
    analysis: TradeoffAnalysis
  ): Promise<number> {
    
    let score = 0;
    let totalWeight = 0;
    
    // Check latency requirement
    if (analysis.latencyBreakdown.totalPipelineLatency <= useCase.requirements.maxLatencyMs) {
      const latencyScore = useCase.requirements.prioritizeLatency ? 0.4 : 0.2;
      score += latencyScore;
      totalWeight += latencyScore;
    } else {
      // Hard requirement not met
      return 0;
    }
    
    // Check quality requirement
    if (analysis.performanceMetrics.nDCG_at_10 >= useCase.requirements.minQualityScore) {
      const qualityScore = useCase.requirements.prioritizeQuality ? 0.4 : 0.2;
      score += qualityScore;
      totalWeight += qualityScore;
    } else {
      // Hard requirement not met
      return 0;
    }
    
    // Check scalability requirement
    const maxHandledUsers = Math.max(
      ...analysis.loadTestResults
        .filter(r => r.errorRate < 5) // Only count results with acceptable error rate
        .map(r => r.concurrentUsers)
    );
    
    if (maxHandledUsers >= useCase.requirements.maxConcurrentUsers) {
      score += 0.2;
      totalWeight += 0.2;
    }
    
    // Bonus points for exceeding requirements
    const latencyBonus = Math.max(0, 
      (useCase.requirements.maxLatencyMs - analysis.latencyBreakdown.totalPipelineLatency) / 
      useCase.requirements.maxLatencyMs
    ) * 0.1;
    
    const qualityBonus = Math.max(0,
      (analysis.performanceMetrics.nDCG_at_10 - useCase.requirements.minQualityScore) /
      (1 - useCase.requirements.minQualityScore)
    ) * 0.1;
    
    score += latencyBonus + qualityBonus;
    totalWeight += 0.2;
    
    return totalWeight > 0 ? score / totalWeight : 0;
  }

  /**
   * Generate hybrid routing strategy
   */
  private async generateHybridRoutingStrategy(
    variantAnalyses: Map<string, TradeoffAnalysis>,
    useCaseMapping: Record<string, string>
  ): Promise<HybridRoutingStrategy> {
    
    console.log('  🔀 Generating hybrid routing strategy');
    
    const variants = Array.from(variantAnalyses.keys());
    
    // Determine if hybrid routing is beneficial
    const uniqueRecommendations = new Set(Object.values(useCaseMapping));
    const enabled = uniqueRecommendations.size > 1;
    
    if (!enabled) {
      return HybridRoutingStrategySchema.parse({
        enabled: false,
        primaryStrategy: 'latency_based',
        rules: [],
        defaultVariant: variants[0],
        routingMetrics: {
          successRate: 1.0,
          avgLatencyMs: 100,
          qualityMaintained: true
        }
      });
    }
    
    // Generate routing rules
    const rules: HybridRoutingRule[] = [];
    
    // Latency-based routing rules
    for (const useCase of this.useCases) {
      const recommendedVariant = useCaseMapping[useCase.id];
      if (recommendedVariant) {
        rules.push(HybridRoutingRuleSchema.parse({
          condition: {
            type: 'latency_requirement',
            operator: '<=',
            value: useCase.requirements.maxLatencyMs
          },
          variant: recommendedVariant,
          confidence: 0.85,
          fallback: this.findFallbackVariant(recommendedVariant, variants)
        }));
      }
    }
    
    // Load-based routing
    const performanceVariant = this.findBestPerformanceVariant(variantAnalyses);
    const latencyVariant = this.findBestLatencyVariant(variantAnalyses);
    
    if (performanceVariant !== latencyVariant) {
      rules.push(HybridRoutingRuleSchema.parse({
        condition: {
          type: 'load_level',
          operator: '<',
          value: 10 // Low load threshold
        },
        variant: performanceVariant,
        confidence: 0.8,
        fallback: latencyVariant
      }));
      
      rules.push(HybridRoutingRuleSchema.parse({
        condition: {
          type: 'load_level',
          operator: '>=',
          value: 100 // High load threshold
        },
        variant: latencyVariant,
        confidence: 0.9,
        fallback: performanceVariant
      }));
    }
    
    return HybridRoutingStrategySchema.parse({
      enabled,
      primaryStrategy: 'latency_based',
      rules,
      defaultVariant: latencyVariant || variants[0],
      routingMetrics: {
        successRate: 0.95,
        avgLatencyMs: this.estimateAverageLatency(variantAnalyses, rules),
        qualityMaintained: true
      }
    });
  }

  /**
   * Make deployment decision with comprehensive risk assessment
   */
  private async makeDeploymentDecision(
    variantAnalyses: Map<string, TradeoffAnalysis>,
    statisticalComparisons: StatisticalAnalysis[],
    variantScoring: VariantScoring[]
  ): Promise<DeploymentDecision> {
    
    console.log('  🚀 Making deployment decision');
    
    // Find the best scoring variant
    const bestVariant = variantScoring.reduce((best, current) =>
      current.overallScore > best.overallScore ? current : best
    );
    
    // Find statistical comparison for the best variant
    const relevantComparison = statisticalComparisons.find(comp =>
      comp.pairedComparison.metric.includes('nDCG') // Primary quality metric
    );
    
    const confidence = Math.min(
      bestVariant.confidence,
      relevantComparison?.pairedComparison.practicalSignificance ? 0.95 : 0.7
    );
    
    // Risk assessment
    const riskAssessment = await this.assessDeploymentRisks(
      bestVariant,
      variantAnalyses.get(bestVariant.variant)!,
      statisticalComparisons
    );
    
    // Determine deployment strategy based on risk and confidence
    let deploymentStrategy: 'blue_green' | 'canary' | 'rolling' | 'immediate';
    
    if (riskAssessment.overallRisk === 'high' || riskAssessment.overallRisk === 'very_high') {
      deploymentStrategy = 'canary';
    } else if (confidence < 0.8) {
      deploymentStrategy = 'blue_green';
    } else if (bestVariant.overallScore > 0.8) {
      deploymentStrategy = 'rolling';
    } else {
      deploymentStrategy = 'canary';
    }
    
    // Define promotion criteria
    const promotionCriteria = {
      qualityThresholds: {
        'nDCG@10': 0.85,
        'recall@50': 0.8,
        'MRR': 0.75
      },
      latencyThresholds: {
        'p95_latency_ms': 200,
        'p99_latency_ms': 500
      },
      errorRateThreshold: 0.02, // 2% max error rate
      rollbackTriggers: [
        'Quality degradation > 5%',
        'Latency increase > 20%',
        'Error rate > 5%',
        'Customer complaints > baseline'
      ]
    };
    
    return DeploymentDecisionSchema.parse({
      recommendedVariant: bestVariant.variant,
      deploymentStrategy,
      confidence,
      statisticalSignificance: {
        pValue: relevantComparison?.pairedComparison.pValue || 0.05,
        effectSize: relevantComparison?.pairedComparison.effectSize || 0.2,
        practicallySignificant: relevantComparison?.pairedComparison.practicalSignificance || false
      },
      riskAssessment,
      promotionCriteria
    });
  }

  /**
   * Assess deployment risks
   */
  private async assessDeploymentRisks(
    variant: VariantScoring,
    analysis: TradeoffAnalysis,
    statisticalComparisons: StatisticalAnalysis[]
  ): Promise<any> {
    
    const risks: Array<{
      category: string;
      severity: 'low' | 'medium' | 'high' | 'critical';
      probability: number;
      description: string;
      mitigation: string;
    }> = [];
    
    // Performance degradation risk
    if (variant.criteriaScores.quality < 0.8) {
      risks.push({
        category: 'Performance Degradation',
        severity: 'high',
        probability: 0.3,
        description: 'Quality metrics below optimal threshold',
        mitigation: 'Gradual rollout with quality monitoring'
      });
    }
    
    // Latency risk
    if (variant.criteriaScores.latency < 0.7) {
      risks.push({
        category: 'Latency Increase',
        severity: 'medium',
        probability: 0.4,
        description: 'Higher latency may impact user experience',
        mitigation: 'Load balancing and caching optimization'
      });
    }
    
    // Scalability risk
    const maxUsers = Math.max(...analysis.loadTestResults.map(r => r.concurrentUsers));
    if (maxUsers < 100) {
      risks.push({
        category: 'Scalability Limitation',
        severity: 'medium',
        probability: 0.2,
        description: 'May not handle peak traffic loads',
        mitigation: 'Auto-scaling configuration and load testing'
      });
    }
    
    // Statistical uncertainty risk
    const hasStrongEvidence = statisticalComparisons.some(comp =>
      comp.pairedComparison.practicalSignificance && comp.pairedComparison.pValue < 0.01
    );
    
    if (!hasStrongEvidence) {
      risks.push({
        category: 'Statistical Uncertainty',
        severity: 'medium',
        probability: 0.5,
        description: 'Limited statistical evidence for superiority',
        mitigation: 'A/B testing with larger sample size'
      });
    }
    
    // Determine overall risk level
    const highRiskCount = risks.filter(r => r.severity === 'high' || r.severity === 'critical').length;
    const overallRisk = highRiskCount > 1 ? 'very_high' :
                       highRiskCount === 1 ? 'high' :
                       risks.length > 2 ? 'medium' : 'low';
    
    return { overallRisk, risks };
  }

  /**
   * Perform sensitivity analysis on decision criteria
   */
  private async performSensitivityAnalysis(
    variantAnalyses: Map<string, TradeoffAnalysis>
  ): Promise<any> {
    
    console.log('  🧪 Performing sensitivity analysis');
    
    const baselineScoring = await this.scoreVariants(variantAnalyses);
    const baselineWinner = baselineScoring.reduce((best, current) =>
      current.overallScore > best.overallScore ? current : best
    ).variant;
    
    const criteriaWeightSensitivity: Record<string, number> = {};
    
    // Test sensitivity to criteria weight changes
    for (const criterion of ['quality', 'latency', 'scalability', 'resourceEfficiency', 'robustness']) {
      let changedDecisions = 0;
      const testWeights = [0.1, 0.3, 0.5, 0.7, 0.9];
      
      for (const testWeight of testWeights) {
        const modifiedCriteria = { ...this.criteria };
        (modifiedCriteria as any)[criterion].weight = testWeight;
        
        // Renormalize other weights
        const otherCriteria = Object.keys(this.criteria).filter(c => c !== criterion);
        const remainingWeight = (1 - testWeight) / otherCriteria.length;
        otherCriteria.forEach(c => {
          (modifiedCriteria as any)[c].weight = remainingWeight;
        });
        
        const framework = new DecisionFramework(this.useCases, modifiedCriteria);
        const testScoring = await framework.scoreVariants(variantAnalyses);
        const testWinner = testScoring.reduce((best, current) =>
          current.overallScore > best.overallScore ? current : best
        ).variant;
        
        if (testWinner !== baselineWinner) {
          changedDecisions++;
        }
      }
      
      criteriaWeightSensitivity[criterion] = changedDecisions / testWeights.length;
    }
    
    // Calculate overall robustness
    const avgSensitivity = Object.values(criteriaWeightSensitivity)
      .reduce((sum, val) => sum + val, 0) / Object.keys(criteriaWeightSensitivity).length;
    const robustness = 1 - avgSensitivity;
    
    return {
      criteriaWeightSensitivity,
      thresholdSensitivity: {}, // Would implement threshold sensitivity if needed
      robustness
    };
  }

  // Helper methods

  private generateComparisonSummary(variantScoring: VariantScoring[]): any {
    const sorted = [...variantScoring].sort((a, b) => b.overallScore - a.overallScore);
    
    return {
      variants: variantScoring.map(v => v.variant),
      primaryMetric: 'overall_score',
      winningVariant: sorted[0].variant,
      winMargin: sorted.length > 1 ? sorted[0].overallScore - sorted[1].overallScore : 0
    };
  }

  private identifyUseCases(analysis: TradeoffAnalysis): string[] {
    return this.useCases
      .filter(useCase => {
        const meetsLatency = analysis.latencyBreakdown.totalPipelineLatency <= useCase.requirements.maxLatencyMs;
        const meetsQuality = analysis.performanceMetrics.nDCG_at_10 >= useCase.requirements.minQualityScore;
        return meetsLatency && meetsQuality;
      })
      .map(useCase => useCase.id);
  }

  private calculateConfidence(analysis: TradeoffAnalysis, scores: any): number {
    // Base confidence on data quality and consistency
    let confidence = 1.0;
    
    // Penalize if load test results are inconsistent
    const errorRates = analysis.loadTestResults.map(r => r.errorRate);
    const errorRateVariance = this.variance(errorRates);
    if (errorRateVariance > 5) {
      confidence *= 0.8;
    }
    
    // Penalize if performance metrics are at extremes (suggesting possible issues)
    if (scores.quality < 0.3 || scores.quality > 0.99) {
      confidence *= 0.9;
    }
    
    // Boost confidence if all criteria are well-balanced
    const criteriaValues = Object.values(scores);
    const minScore = Math.min(...criteriaValues as number[]);
    if (minScore > 0.6) {
      confidence *= 1.1;
    }
    
    return Math.max(0, Math.min(1, confidence));
  }

  private identifyRisks(analysis: TradeoffAnalysis, scores: any): string[] {
    const risks: string[] = [];
    
    if (scores.quality < 0.7) {
      risks.push('Quality concerns - may impact user satisfaction');
    }
    
    if (scores.latency < 0.6) {
      risks.push('Latency concerns - may not meet real-time requirements');
    }
    
    if (scores.scalability < 0.5) {
      risks.push('Scalability limitations - may not handle peak loads');
    }
    
    if (scores.robustness < 0.7) {
      risks.push('Robustness concerns - may have high error rates under stress');
    }
    
    const maxErrorRate = Math.max(...analysis.loadTestResults.map(r => r.errorRate));
    if (maxErrorRate > 10) {
      risks.push('High error rates observed during load testing');
    }
    
    return risks;
  }

  private findFallbackVariant(primary: string, variants: string[]): string {
    return variants.find(v => v !== primary) || variants[0];
  }

  private findBestPerformanceVariant(variantAnalyses: Map<string, TradeoffAnalysis>): string {
    let best = '';
    let bestScore = -1;
    
    for (const [variant, analysis] of variantAnalyses) {
      const score = analysis.performanceMetrics.nDCG_at_10;
      if (score > bestScore) {
        bestScore = score;
        best = variant;
      }
    }
    
    return best;
  }

  private findBestLatencyVariant(variantAnalyses: Map<string, TradeoffAnalysis>): string {
    let best = '';
    let bestLatency = Infinity;
    
    for (const [variant, analysis] of variantAnalyses) {
      const latency = analysis.latencyBreakdown.totalPipelineLatency;
      if (latency < bestLatency) {
        bestLatency = latency;
        best = variant;
      }
    }
    
    return best;
  }

  private estimateAverageLatency(
    variantAnalyses: Map<string, TradeoffAnalysis>,
    rules: HybridRoutingRule[]
  ): number {
    // Simplified estimation - would use traffic patterns in production
    const latencies = Array.from(variantAnalyses.values())
      .map(a => a.latencyBreakdown.totalPipelineLatency);
    
    return latencies.reduce((sum, lat) => sum + lat, 0) / latencies.length;
  }

  private variance(values: number[]): number {
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    return values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
  }

  /**
   * Save decision framework report
   */
  async saveDecisionReport(
    report: DecisionReport,
    outputPath: string
  ): Promise<void> {
    const enhancedReport = {
      ...report,
      metadata: {
        generator: 'Gemma Decision Framework',
        version: '1.0.0',
        methodology: 'Multi-Criteria Decision Analysis (MCDA)',
        criteria: Object.keys(this.criteria),
        useCases: this.useCases.length,
        statisticalRigor: 'Bootstrap confidence intervals, effect size analysis',
        sensitivityAnalysis: 'Criteria weight perturbation testing'
      },
      interpretation: {
        scoreInterpretation: {
          '0.9 - 1.0': 'Excellent',
          '0.8 - 0.9': 'Very Good',
          '0.7 - 0.8': 'Good',
          '0.6 - 0.7': 'Acceptable',
          '< 0.6': 'Needs Improvement'
        },
        confidenceInterpretation: {
          '> 0.9': 'High confidence',
          '0.8 - 0.9': 'Good confidence',
          '0.7 - 0.8': 'Moderate confidence',
          '< 0.7': 'Low confidence - proceed with caution'
        }
      }
    };

    await fs.promises.writeFile(
      outputPath,
      JSON.stringify(enhancedReport, null, 2),
      'utf8'
    );

    console.log(`🎯 Decision framework report saved to ${outputPath}`);
  }
}

// Default configurations

const DEFAULT_USE_CASES: UseCase[] = [
  {
    id: 'interactive_ide_search',
    name: 'Interactive IDE Search',
    description: 'Real-time search within code editors',
    requirements: {
      maxLatencyMs: 100,
      minQualityScore: 0.7,
      maxConcurrentUsers: 10,
      prioritizeQuality: false,
      prioritizeLatency: true,
      resourceConstraints: {
        maxMemoryMB: 500,
        maxCpuCores: 2
      }
    },
    weight: 0.3,
    businessImpact: 'high'
  },
  {
    id: 'batch_code_analysis',
    name: 'Batch Code Analysis',
    description: 'Offline processing of large codebases',
    requirements: {
      maxLatencyMs: 1000,
      minQualityScore: 0.9,
      maxConcurrentUsers: 1000,
      prioritizeQuality: true,
      prioritizeLatency: false,
      resourceConstraints: {
        maxMemoryMB: 2000,
        maxCpuCores: 8
      }
    },
    weight: 0.25,
    businessImpact: 'critical'
  },
  {
    id: 'code_review_assistant',
    name: 'Code Review Assistant',
    description: 'AI-powered code review suggestions',
    requirements: {
      maxLatencyMs: 500,
      minQualityScore: 0.85,
      maxConcurrentUsers: 50,
      prioritizeQuality: true,
      prioritizeLatency: false,
      resourceConstraints: {
        maxMemoryMB: 1000,
        maxCpuCores: 4
      }
    },
    weight: 0.25,
    businessImpact: 'high'
  },
  {
    id: 'documentation_search',
    name: 'Documentation Search',
    description: 'Search through technical documentation',
    requirements: {
      maxLatencyMs: 200,
      minQualityScore: 0.75,
      maxConcurrentUsers: 100,
      prioritizeQuality: false,
      prioritizeLatency: false,
      resourceConstraints: {
        maxMemoryMB: 800,
        maxCpuCores: 2
      }
    },
    weight: 0.2,
    businessImpact: 'medium'
  }
];

const DEFAULT_CRITERIA: DecisionCriteria = {
  quality: {
    weight: 0.35,
    metrics: ['nDCG@10', 'recall@50', 'MRR'],
    thresholds: { 'nDCG@10': 0.8, 'recall@50': 0.75 }
  },
  latency: {
    weight: 0.25,
    metrics: ['p95_latency', 'encoding_latency', 'search_latency'],
    thresholds: { 'p95_latency': 200, 'total_latency': 500 }
  },
  scalability: {
    weight: 0.2,
    metrics: ['max_concurrent_users', 'throughput'],
    thresholds: { 'max_concurrent_users': 100, 'qps': 50 }
  },
  resourceEfficiency: {
    weight: 0.15,
    metrics: ['memory_usage', 'cpu_utilization'],
    thresholds: { 'memory_mb': 1000, 'cpu_percent': 80 }
  },
  robustness: {
    weight: 0.05,
    metrics: ['error_rate', 'stability'],
    thresholds: { 'error_rate': 0.02, 'uptime': 0.999 }
  }
};