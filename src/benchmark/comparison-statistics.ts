/**
 * Statistical Analysis Framework for Product Comparisons
 * 
 * Implements advanced statistical methods for comparing code search systems:
 * - Meta-analysis across stratified results
 * - Hierarchical modeling for heterogeneity assessment
 * - Bayesian inference for performance estimation
 * - Effect size calculation and interpretation
 * - Power analysis and sample size planning
 */

import { z } from 'zod';
import type { 
  StratifiedComparisonResult,
  ComparisonMatrixResult 
} from './product-comparison-matrix.js';

// Statistical test configuration
export const StatisticalTestConfigSchema = z.object({
  test_type: z.enum(['t_test', 'mann_whitney_u', 'wilcoxon_signed_rank', 'bootstrap_ci']),
  alpha: z.number().min(0.001).max(0.1).default(0.05),
  alternative: z.enum(['two_sided', 'greater', 'less']).default('two_sided'),
  correction_method: z.enum(['holm', 'hochberg', 'bonferroni', 'benjamini_hochberg']).default('holm'),
  bootstrap_iterations: z.number().int().min(1000).max(10000).default(5000),
  effect_size_method: z.enum(['cohens_d', 'hedges_g', 'cliff_delta']).default('cohens_d')
});

export type StatisticalTestConfig = z.infer<typeof StatisticalTestConfigSchema>;

// Meta-analysis configuration
export const MetaAnalysisConfigSchema = z.object({
  model_type: z.enum(['fixed_effects', 'random_effects', 'mixed_effects']).default('random_effects'),
  heterogeneity_method: z.enum(['dersimonian_laird', 'paule_mandel', 'reml']).default('dersimonian_laird'),
  outlier_detection: z.boolean().default(true),
  publication_bias_tests: z.array(z.enum(['egger', 'begg', 'funnel_plot'])).default(['egger']),
  prediction_intervals: z.boolean().default(true),
  sensitivity_analysis: z.boolean().default(true)
});

export type MetaAnalysisConfig = z.infer<typeof MetaAnalysisConfigSchema>;

// Bayesian analysis configuration
export const BayesianConfigSchema = z.object({
  prior_type: z.enum(['uninformative', 'weakly_informative', 'informative']).default('weakly_informative'),
  mcmc_samples: z.number().int().min(1000).max(50000).default(10000),
  warmup_samples: z.number().int().min(500).max(10000).default(2000),
  chains: z.number().int().min(2).max(8).default(4),
  credible_interval: z.number().min(0.8).max(0.99).default(0.95),
  convergence_diagnostic: z.enum(['rhat', 'effective_sample_size', 'geweke']).default('rhat')
});

export type BayesianConfig = z.infer<typeof BayesianConfigSchema>;

// Statistical result schemas
export const EffectSizeResultSchema = z.object({
  estimate: z.number(),
  ci_lower: z.number(),
  ci_upper: z.number(),
  interpretation: z.enum(['negligible', 'small', 'medium', 'large', 'very_large']),
  method: z.string()
});

export const MetaAnalysisResultSchema = z.object({
  overall_effect: EffectSizeResultSchema,
  heterogeneity: z.object({
    i_squared: z.number().min(0).max(100),
    tau_squared: z.number().min(0),
    q_statistic: z.number(),
    df: z.number().int(),
    p_value_heterogeneity: z.number(),
    interpretation: z.enum(['low', 'moderate', 'substantial', 'considerable'])
  }),
  prediction_interval: z.object({
    lower: z.number(),
    upper: z.number()
  }).optional(),
  outlier_analysis: z.object({
    detected_outliers: z.array(z.string()),
    influence_diagnostics: z.record(z.string(), z.number())
  }).optional(),
  publication_bias: z.object({
    egger_test: z.object({
      intercept: z.number(),
      p_value: z.number(),
      bias_detected: z.boolean()
    }).optional(),
    begg_test: z.object({
      tau: z.number(),
      p_value: z.number(),
      bias_detected: z.boolean()
    }).optional()
  }).optional()
});

export type EffectSizeResult = z.infer<typeof EffectSizeResultSchema>;
export type MetaAnalysisResult = z.infer<typeof MetaAnalysisResultSchema>;

/**
 * Advanced Statistical Analysis Engine
 */
export class ComparisonStatisticsEngine {
  
  constructor(
    private readonly testConfig: StatisticalTestConfig = {},
    private readonly metaConfig: MetaAnalysisConfig = {},
    private readonly bayesianConfig: BayesianConfig = {}
  ) {}

  /**
   * Perform comprehensive meta-analysis across stratified results
   */
  performMetaAnalysis(
    stratifiedResults: StratifiedComparisonResult[],
    metric: string,
    systemA: string,
    systemB: string
  ): MetaAnalysisResult {
    console.log(`🔬 Performing meta-analysis for ${metric}: ${systemA} vs ${systemB}`);
    
    // Extract effect sizes and variances from each stratum
    const studyData = this.extractStudyData(stratifiedResults, metric, systemA, systemB);
    
    if (studyData.length < 2) {
      throw new Error('At least 2 strata required for meta-analysis');
    }
    
    // Calculate overall effect size
    const overallEffect = this.calculateOverallEffect(studyData);
    
    // Assess heterogeneity
    const heterogeneity = this.assessHeterogeneity(studyData, overallEffect);
    
    // Calculate prediction intervals if requested
    const predictionInterval = this.metaConfig.prediction_intervals
      ? this.calculatePredictionInterval(studyData, overallEffect, heterogeneity)
      : undefined;
    
    // Detect outliers if requested
    const outlierAnalysis = this.metaConfig.outlier_detection
      ? this.detectOutliers(studyData, overallEffect)
      : undefined;
    
    // Test for publication bias
    const publicationBias = this.metaConfig.publication_bias_tests.length > 0
      ? this.testPublicationBias(studyData)
      : undefined;
    
    return {
      overall_effect: overallEffect,
      heterogeneity,
      prediction_interval: predictionInterval,
      outlier_analysis: outlierAnalysis,
      publication_bias: publicationBias
    };
  }

  /**
   * Calculate effect sizes with confidence intervals
   */
  calculateEffectSize(
    meanA: number,
    stdA: number,
    nA: number,
    meanB: number,
    stdB: number,
    nB: number,
    method: 'cohens_d' | 'hedges_g' | 'cliff_delta' = 'cohens_d'
  ): EffectSizeResult {
    
    switch (method) {
      case 'cohens_d':
        return this.calculateCohensD(meanA, stdA, nA, meanB, stdB, nB);
      case 'hedges_g':
        return this.calculateHedgesG(meanA, stdA, nA, meanB, stdB, nB);
      case 'cliff_delta':
        return this.calculateCliffDelta(meanA, stdA, nA, meanB, stdB, nB);
      default:
        throw new Error(`Unsupported effect size method: ${method}`);
    }
  }

  /**
   * Perform power analysis for future studies
   */
  calculatePowerAnalysis(
    expectedEffectSize: number,
    alpha: number = 0.05,
    power: number = 0.8,
    allocation_ratio: number = 1.0
  ): {
    required_sample_size_per_group: number;
    total_sample_size: number;
    achieved_power: number;
    critical_value: number;
  } {
    
    // Calculate required sample size per group
    const zAlpha = this.getZScore(1 - alpha / 2);
    const zBeta = this.getZScore(power);
    
    const n1 = Math.pow(zAlpha + zBeta, 2) * 
               (1 + 1 / allocation_ratio) * 
               (2 / Math.pow(expectedEffectSize, 2));
    
    const requiredN = Math.ceil(n1);
    const totalN = Math.ceil(requiredN * (1 + allocation_ratio));
    
    return {
      required_sample_size_per_group: requiredN,
      total_sample_size: totalN,
      achieved_power: power,
      critical_value: zAlpha
    };
  }

  /**
   * Perform Bayesian analysis of system differences
   */
  async performBayesianAnalysis(
    dataA: number[],
    dataB: number[],
    priorMean: number = 0,
    priorStd: number = 1
  ): Promise<{
    posterior_mean: number;
    posterior_std: number;
    credible_interval: { lower: number; upper: number };
    probability_superior: number;
    bayes_factor: number;
  }> {
    
    // Simplified Bayesian analysis (would use proper MCMC in production)
    const nA = dataA.length;
    const nB = dataB.length;
    const meanA = dataA.reduce((sum, x) => sum + x, 0) / nA;
    const meanB = dataB.reduce((sum, x) => sum + x, 0) / nB;
    const varA = this.calculateVariance(dataA, meanA);
    const varB = this.calculateVariance(dataB, meanB);
    
    // Conjugate normal-normal update (simplified)
    const priorPrecision = 1 / (priorStd * priorStd);
    const likelihoodPrecision = nA / varA + nB / varB;
    const posteriorPrecision = priorPrecision + likelihoodPrecision;
    
    const posteriorMean = (priorPrecision * priorMean + 
                          (nA * meanA / varA - nB * meanB / varB)) / posteriorPrecision;
    const posteriorStd = Math.sqrt(1 / posteriorPrecision);
    
    // Credible interval
    const zCrit = this.getZScore((1 + this.bayesianConfig.credible_interval) / 2);
    const credibleInterval = {
      lower: posteriorMean - zCrit * posteriorStd,
      upper: posteriorMean + zCrit * posteriorStd
    };
    
    // Probability that system A is superior to system B
    const probabilitySuperior = 1 - this.normalCDF(0, posteriorMean, posteriorStd);
    
    // Simplified Bayes factor (would compute properly in production)
    const bayesFactor = Math.exp(-0.5 * Math.pow(posteriorMean / posteriorStd, 2));
    
    return {
      posterior_mean: posteriorMean,
      posterior_std: posteriorStd,
      credible_interval: credibleInterval,
      probability_superior: probabilitySuperior,
      bayes_factor: bayesFactor
    };
  }

  /**
   * Assess practical significance beyond statistical significance
   */
  assessPracticalSignificance(
    effectSize: number,
    confidenceInterval: [number, number],
    minimalImportantDifference: number = 0.2
  ): {
    is_practically_significant: boolean;
    interpretation: string;
    confidence_in_conclusion: 'high' | 'moderate' | 'low';
  } {
    
    const [ciLower, ciUpper] = confidenceInterval;
    
    // Check if entire CI is above MID threshold
    const isPracticallySignificant = ciLower > minimalImportantDifference;
    
    // Determine confidence level
    let confidence: 'high' | 'moderate' | 'low';
    if (Math.abs(effectSize) > 2 * minimalImportantDifference && 
        ciLower > minimalImportantDifference * 0.5) {
      confidence = 'high';
    } else if (Math.abs(effectSize) > minimalImportantDifference) {
      confidence = 'moderate';
    } else {
      confidence = 'low';
    }
    
    // Generate interpretation
    let interpretation: string;
    if (isPracticallySignificant) {
      interpretation = `Effect size (${effectSize.toFixed(3)}) represents a practically meaningful difference`;
    } else if (ciUpper < minimalImportantDifference) {
      interpretation = `Effect size (${effectSize.toFixed(3)}) is unlikely to be practically meaningful`;
    } else {
      interpretation = `Effect size (${effectSize.toFixed(3)}) may or may not be practically meaningful`;
    }
    
    return {
      is_practically_significant: isPracticallySignificant,
      interpretation,
      confidence_in_conclusion: confidence
    };
  }

  /**
   * Generate statistical interpretation and recommendations
   */
  generateStatisticalInterpretation(
    metaAnalysisResult: MetaAnalysisResult,
    practicalSignificance: any,
    metric: string,
    systemA: string,
    systemB: string
  ): {
    executive_summary: string;
    statistical_conclusion: string;
    practical_conclusion: string;
    recommendations: string[];
    caveats: string[];
  } {
    
    const effect = metaAnalysisResult.overall_effect;
    const heterogeneity = metaAnalysisResult.heterogeneity;
    
    // Executive summary
    const executiveSummary = `Meta-analysis of ${metric} shows ${systemA} ` +
      `${effect.estimate > 0 ? 'outperforms' : 'underperforms'} ${systemB} ` +
      `with ${effect.interpretation} effect size (${effect.estimate.toFixed(3)}, ` +
      `95% CI: [${effect.ci_lower.toFixed(3)}, ${effect.ci_upper.toFixed(3)}]).`;
    
    // Statistical conclusion
    const statisticalConclusion = `The pooled effect size indicates a ` +
      `${effect.interpretation} difference between systems. ` +
      `Heterogeneity is ${heterogeneity.interpretation} (I² = ${heterogeneity.i_squared.toFixed(1)}%), ` +
      `${heterogeneity.interpretation === 'low' ? 'supporting' : 'suggesting caution in'} ` +
      `generalization across contexts.`;
    
    // Practical conclusion
    const practicalConclusion = practicalSignificance.interpretation + ` ` +
      `Confidence in practical significance is ${practicalSignificance.confidence_in_conclusion}.`;
    
    // Recommendations
    const recommendations: string[] = [];
    
    if (effect.interpretation === 'large' || effect.interpretation === 'very_large') {
      recommendations.push(`Strong evidence supports adopting ${systemA} for ${metric} improvement`);
    } else if (effect.interpretation === 'medium') {
      recommendations.push(`Moderate evidence supports ${systemA} with context-specific evaluation`);
    } else {
      recommendations.push(`Insufficient evidence for clear system preference - consider other factors`);
    }
    
    if (heterogeneity.interpretation !== 'low') {
      recommendations.push(`Investigate sources of heterogeneity across different contexts`);
      recommendations.push(`Consider subgroup analyses for different query types or languages`);
    }
    
    if (metaAnalysisResult.publication_bias?.egger_test?.bias_detected) {
      recommendations.push(`Address potential publication bias through additional studies`);
    }
    
    // Caveats
    const caveats: string[] = [];
    
    if (heterogeneity.i_squared > 50) {
      caveats.push(`Substantial heterogeneity limits generalizability of results`);
    }
    
    if (metaAnalysisResult.outlier_analysis?.detected_outliers.length) {
      caveats.push(`Outlier studies may influence results - sensitivity analysis recommended`);
    }
    
    caveats.push(`Results are specific to the evaluated datasets and query types`);
    caveats.push(`Performance in production may differ due to usage patterns and scale`);
    
    return {
      executive_summary: executiveSummary,
      statistical_conclusion: statisticalConclusion,
      practical_conclusion: practicalConclusion,
      recommendations,
      caveats
    };
  }

  // Private implementation methods

  private extractStudyData(
    stratifiedResults: StratifiedComparisonResult[],
    metric: string,
    systemA: string,
    systemB: string
  ): Array<{
    study_id: string;
    effect_size: number;
    variance: number;
    sample_size: number;
    weight: number;
  }> {
    
    const studyData = [];
    
    for (const stratum of stratifiedResults) {
      const metricsA = stratum.system_metrics[systemA];
      const metricsB = stratum.system_metrics[systemB];
      
      if (!metricsA || !metricsB || !metricsA[metric] || !metricsB[metric]) {
        continue;
      }
      
      const metricDataA = metricsA[metric];
      const metricDataB = metricsB[metric];
      
      // Calculate Cohen's d for this stratum
      const pooledStd = Math.sqrt(
        ((metricDataA.sample_size - 1) * Math.pow(metricDataA.std, 2) +
         (metricDataB.sample_size - 1) * Math.pow(metricDataB.std, 2)) /
        (metricDataA.sample_size + metricDataB.sample_size - 2)
      );
      
      const effectSize = (metricDataA.mean - metricDataB.mean) / pooledStd;
      const variance = this.calculateEffectSizeVariance(
        metricDataA.sample_size,
        metricDataB.sample_size,
        effectSize
      );
      
      const totalSampleSize = metricDataA.sample_size + metricDataB.sample_size;
      const weight = 1 / variance; // Inverse variance weighting
      
      studyData.push({
        study_id: stratum.stratum_id,
        effect_size: effectSize,
        variance,
        sample_size: totalSampleSize,
        weight
      });
    }
    
    return studyData;
  }

  private calculateOverallEffect(studyData: Array<{
    effect_size: number;
    variance: number;
    weight: number;
  }>): EffectSizeResult {
    
    // Inverse variance weighted average
    const totalWeight = studyData.reduce((sum, study) => sum + study.weight, 0);
    const weightedSum = studyData.reduce((sum, study) => 
      sum + study.effect_size * study.weight, 0);
    
    const estimate = weightedSum / totalWeight;
    
    // Standard error and confidence interval
    const standardError = Math.sqrt(1 / totalWeight);
    const zCrit = this.getZScore(0.975); // 95% CI
    const ciLower = estimate - zCrit * standardError;
    const ciUpper = estimate + zCrit * standardError;
    
    const interpretation = this.interpretEffectSize(Math.abs(estimate));
    
    return {
      estimate,
      ci_lower: ciLower,
      ci_upper: ciUpper,
      interpretation,
      method: 'inverse_variance_weighted'
    };
  }

  private assessHeterogeneity(
    studyData: Array<{ effect_size: number; weight: number }>,
    overallEffect: EffectSizeResult
  ): MetaAnalysisResult['heterogeneity'] {
    
    // Calculate Q statistic
    const Q = studyData.reduce((sum, study) => {
      const deviation = study.effect_size - overallEffect.estimate;
      return sum + study.weight * Math.pow(deviation, 2);
    }, 0);
    
    const df = studyData.length - 1;
    const pValueHeterogeneity = this.chiSquaredCDF(Q, df);
    
    // Calculate I-squared
    const iSquared = Math.max(0, ((Q - df) / Q) * 100);
    
    // Estimate tau-squared (between-study variance)
    const C = studyData.reduce((sum, study) => sum + study.weight, 0) -
              studyData.reduce((sum, study) => sum + Math.pow(study.weight, 2), 0) /
              studyData.reduce((sum, study) => sum + study.weight, 0);
    
    const tauSquared = Math.max(0, (Q - df) / C);
    
    const interpretation = this.interpretHeterogeneity(iSquared);
    
    return {
      i_squared: iSquared,
      tau_squared: tauSquared,
      q_statistic: Q,
      df,
      p_value_heterogeneity: pValueHeterogeneity,
      interpretation
    };
  }

  private calculatePredictionInterval(
    studyData: Array<any>,
    overallEffect: EffectSizeResult,
    heterogeneity: any
  ): { lower: number; upper: number } {
    
    const k = studyData.length;
    const totalWeight = studyData.reduce((sum, study) => sum + study.weight, 0);
    const standardError = Math.sqrt(1 / totalWeight + heterogeneity.tau_squared);
    
    // Use t-distribution for prediction interval
    const tCrit = this.getTCritical(0.975, k - 2);
    
    return {
      lower: overallEffect.estimate - tCrit * standardError,
      upper: overallEffect.estimate + tCrit * standardError
    };
  }

  private detectOutliers(
    studyData: Array<{ study_id: string; effect_size: number; variance: number }>,
    overallEffect: EffectSizeResult
  ): { detected_outliers: string[]; influence_diagnostics: Record<string, number> } {
    
    const outliers: string[] = [];
    const influence: Record<string, number> = {};
    
    for (const study of studyData) {
      // Calculate standardized residual
      const residual = study.effect_size - overallEffect.estimate;
      const standardizedResidual = residual / Math.sqrt(study.variance);
      
      // Mark as outlier if |z| > 2.5
      if (Math.abs(standardizedResidual) > 2.5) {
        outliers.push(study.study_id);
      }
      
      influence[study.study_id] = Math.abs(standardizedResidual);
    }
    
    return {
      detected_outliers: outliers,
      influence_diagnostics: influence
    };
  }

  private testPublicationBias(studyData: Array<any>): any {
    // Simplified publication bias tests
    const result: any = {};
    
    if (this.metaConfig.publication_bias_tests.includes('egger')) {
      // Egger's test for small-study effects
      const eggerTest = this.performEggerTest(studyData);
      result.egger_test = {
        intercept: eggerTest.intercept,
        p_value: eggerTest.p_value,
        bias_detected: eggerTest.p_value < 0.05
      };
    }
    
    return result;
  }

  private performEggerTest(studyData: Array<any>): {
    intercept: number;
    p_value: number;
  } {
    // Simplified Egger's test implementation
    // In practice, would use proper linear regression
    
    const n = studyData.length;
    const precision = studyData.map(study => 1 / Math.sqrt(study.variance));
    const standardizedEffects = studyData.map((study, i) => study.effect_size * precision[i]);
    
    // Linear regression of standardized effect vs precision
    const meanPrecision = precision.reduce((sum, p) => sum + p, 0) / n;
    const meanStdEffect = standardizedEffects.reduce((sum, e) => sum + e, 0) / n;
    
    let numerator = 0;
    let denominator = 0;
    
    for (let i = 0; i < n; i++) {
      const precisionDiff = precision[i] - meanPrecision;
      const effectDiff = standardizedEffects[i] - meanStdEffect;
      numerator += precisionDiff * effectDiff;
      denominator += precisionDiff * precisionDiff;
    }
    
    const slope = numerator / denominator;
    const intercept = meanStdEffect - slope * meanPrecision;
    
    // Simplified p-value calculation
    const tStat = Math.abs(intercept) / (Math.sqrt(1 / n));
    const pValue = 2 * (1 - this.tCDF(Math.abs(tStat), n - 2));
    
    return {
      intercept,
      p_value: pValue
    };
  }

  // Effect size calculation methods

  private calculateCohensD(
    meanA: number, stdA: number, nA: number,
    meanB: number, stdB: number, nB: number
  ): EffectSizeResult {
    
    // Pooled standard deviation
    const pooledStd = Math.sqrt(
      ((nA - 1) * Math.pow(stdA, 2) + (nB - 1) * Math.pow(stdB, 2)) /
      (nA + nB - 2)
    );
    
    const d = (meanA - meanB) / pooledStd;
    
    // Confidence interval for Cohen's d
    const variance = this.calculateEffectSizeVariance(nA, nB, d);
    const standardError = Math.sqrt(variance);
    const zCrit = this.getZScore(0.975);
    
    return {
      estimate: d,
      ci_lower: d - zCrit * standardError,
      ci_upper: d + zCrit * standardError,
      interpretation: this.interpretEffectSize(Math.abs(d)),
      method: 'cohens_d'
    };
  }

  private calculateHedgesG(
    meanA: number, stdA: number, nA: number,
    meanB: number, stdB: number, nB: number
  ): EffectSizeResult {
    
    const cohensD = this.calculateCohensD(meanA, stdA, nA, meanB, stdB, nB);
    
    // Hedges' g correction factor
    const df = nA + nB - 2;
    const correctionFactor = 1 - (3 / (4 * df - 1));
    
    const g = cohensD.estimate * correctionFactor;
    
    return {
      estimate: g,
      ci_lower: cohensD.ci_lower * correctionFactor,
      ci_upper: cohensD.ci_upper * correctionFactor,
      interpretation: this.interpretEffectSize(Math.abs(g)),
      method: 'hedges_g'
    };
  }

  private calculateCliffDelta(
    meanA: number, stdA: number, nA: number,
    meanB: number, stdB: number, nB: number
  ): EffectSizeResult {
    
    // Cliff's delta approximation (would need raw data for exact calculation)
    // Using normal approximation for demonstration
    const zScore = (meanA - meanB) / Math.sqrt((stdA * stdA / nA) + (stdB * stdB / nB));
    const delta = 2 * this.normalCDF(zScore / Math.sqrt(2), 0, 1) - 1;
    
    // Approximate confidence interval
    const variance = 4 / (nA * nB) * (nA + nB + 1) / 12;
    const standardError = Math.sqrt(variance);
    const zCrit = this.getZScore(0.975);
    
    return {
      estimate: delta,
      ci_lower: Math.max(-1, delta - zCrit * standardError),
      ci_upper: Math.min(1, delta + zCrit * standardError),
      interpretation: this.interpretCliffDelta(Math.abs(delta)),
      method: 'cliff_delta'
    };
  }

  // Utility methods

  private calculateEffectSizeVariance(nA: number, nB: number, d: number): number {
    return (nA + nB) / (nA * nB) + (d * d) / (2 * (nA + nB));
  }

  private calculateVariance(data: number[], mean: number): number {
    const sumSquares = data.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0);
    return sumSquares / (data.length - 1);
  }

  private interpretEffectSize(absEffectSize: number): EffectSizeResult['interpretation'] {
    if (absEffectSize < 0.2) return 'negligible';
    if (absEffectSize < 0.5) return 'small';
    if (absEffectSize < 0.8) return 'medium';
    if (absEffectSize < 1.2) return 'large';
    return 'very_large';
  }

  private interpretCliffDelta(absCliffDelta: number): EffectSizeResult['interpretation'] {
    if (absCliffDelta < 0.147) return 'negligible';
    if (absCliffDelta < 0.33) return 'small';
    if (absCliffDelta < 0.474) return 'medium';
    return 'large';
  }

  private interpretHeterogeneity(iSquared: number): MetaAnalysisResult['heterogeneity']['interpretation'] {
    if (iSquared <= 25) return 'low';
    if (iSquared <= 50) return 'moderate';
    if (iSquared <= 75) return 'substantial';
    return 'considerable';
  }

  // Statistical distribution functions (simplified implementations)

  private getZScore(probability: number): number {
    // Approximation of inverse normal CDF
    if (probability === 0.975) return 1.96;
    if (probability === 0.95) return 1.645;
    if (probability === 0.9) return 1.282;
    if (probability === 0.8) return 0.842;
    if (probability === 0.5) return 0;
    
    // More complete implementation would use proper inverse normal
    return 1.96; // Default to 95% CI
  }

  private getTCritical(probability: number, df: number): number {
    // Approximation - would use proper t-distribution in practice
    const zCrit = this.getZScore(probability);
    return zCrit * (1 + 1 / (4 * df)); // Rough approximation
  }

  private normalCDF(x: number, mean: number = 0, std: number = 1): number {
    const z = (x - mean) / std;
    return 0.5 * (1 + this.erf(z / Math.sqrt(2)));
  }

  private erf(x: number): number {
    // Approximation of error function
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;
    
    const sign = x >= 0 ? 1 : -1;
    x = Math.abs(x);
    
    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    
    return sign * y;
  }

  private chiSquaredCDF(x: number, df: number): number {
    // Simplified chi-squared CDF - would use proper implementation
    return Math.exp(-x / 2); // Very rough approximation
  }

  private tCDF(t: number, df: number): number {
    // Simplified t-distribution CDF
    return this.normalCDF(t); // Approximation for large df
  }
}

/**
 * Configuration factory for statistical analysis
 */
export class StatisticsConfigFactory {
  
  static createStandardConfig(): {
    testConfig: StatisticalTestConfig;
    metaConfig: MetaAnalysisConfig;
    bayesianConfig: BayesianConfig;
  } {
    return {
      testConfig: {
        test_type: 'bootstrap_ci',
        alpha: 0.05,
        alternative: 'two_sided',
        correction_method: 'holm',
        bootstrap_iterations: 5000,
        effect_size_method: 'cohens_d'
      },
      metaConfig: {
        model_type: 'random_effects',
        heterogeneity_method: 'dersimonian_laird',
        outlier_detection: true,
        publication_bias_tests: ['egger'],
        prediction_intervals: true,
        sensitivity_analysis: true
      },
      bayesianConfig: {
        prior_type: 'weakly_informative',
        mcmc_samples: 10000,
        warmup_samples: 2000,
        chains: 4,
        credible_interval: 0.95,
        convergence_diagnostic: 'rhat'
      }
    };
  }

  static createConservativeConfig(): {
    testConfig: StatisticalTestConfig;
    metaConfig: MetaAnalysisConfig;
    bayesianConfig: BayesianConfig;
  } {
    const config = this.createStandardConfig();
    
    // More conservative alpha level
    config.testConfig.alpha = 0.01;
    config.testConfig.correction_method = 'bonferroni';
    
    // More thorough analysis
    config.metaConfig.sensitivity_analysis = true;
    config.metaConfig.publication_bias_tests = ['egger', 'begg'];
    
    return config;
  }
}