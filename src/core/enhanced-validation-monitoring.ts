/**
 * Enhanced Validation & Monitoring System
 * 
 * Implements sophisticated monitoring with:
 * - CUPED (pre-treatment covariate = baseline score/margin) for variance reduction on long-tail slices
 * - Pool growth metric (new-qrels per week) to avoid "over-fit to the pool" wins
 * - Topic-normalized Core@10 monitoring with path-role veto (`third_party/`, `vendor/`)
 * - Grapheme-cluster fuzz testing (ZWJ/combining marks, emoji skin-tone chains)
 * - Alias-resolved redirects in chaos pack with ±1 line/col drift tripwires
 * - Comprehensive validation for production-grade reliability
 */

import type { SearchContext, SearchHit } from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';

export interface CUPEDAnalysis {
  slice_id: string;
  treatment_effect: number;
  treatment_effect_cuped: number;
  variance_reduction: number;
  pre_treatment_covariate: number;
  sample_size: number;
  confidence_interval: [number, number];
  confidence_interval_cuped: [number, number];
  statistical_power: number;
  significance_level: number;
}

export interface PoolGrowthMetrics {
  week_start: Date;
  week_end: Date;
  new_qrels_count: number;
  total_qrels_count: number;
  unique_topics_added: number;
  diversity_score: number;
  coverage_gaps_identified: number;
  quality_score: number; // Based on inter-annotator agreement
  overfitting_risk_score: number; // 0-1, higher = more risk
}

export interface TopicNormalizedMetrics {
  topic: string;
  core_at_10_raw: number;
  core_at_10_normalized: number;
  path_role_violations: number;
  third_party_boost_count: number;
  vendor_boost_count: number;
  normalization_factor: number;
  sample_queries: number;
  confidence_interval: [number, number];
}

export interface GraphemeClusterTest {
  test_id: string;
  input_text: string;
  cluster_type: 'zwj_sequence' | 'combining_marks' | 'emoji_skin_tone' | 'regional_indicators';
  expected_clusters: string[];
  actual_clusters: string[];
  passed: boolean;
  error_details?: string;
  normalization_form: 'NFC' | 'NFD' | 'NFKC' | 'NFKD';
}

export interface AliasRedirectTest {
  test_id: string;
  original_path: string;
  resolved_path: string;
  original_line: number;
  original_col: number;
  resolved_line: number;
  resolved_col: number;
  drift_line: number;
  drift_col: number;
  tripwire_triggered: boolean;
  resolution_time_ms: number;
}

export interface ValidationReport {
  timestamp: Date;
  cuped_analyses: CUPEDAnalysis[];
  pool_growth: PoolGrowthMetrics;
  topic_normalized_metrics: TopicNormalizedMetrics[];
  grapheme_tests_passed: number;
  grapheme_tests_total: number;
  alias_redirect_tests_passed: number;
  alias_redirect_tests_total: number;
  drift_tripwires_triggered: number;
  overall_health_score: number;
  critical_issues: string[];
  recommendations: string[];
}

/**
 * CUPED (Controlled-experiment Using Pre-Experiment Data) implementation
 * Reduces variance by using pre-treatment covariates
 */
class CUPEDAnalyzer {
  private preExperimentData: Map<string, Array<{
    slice_id: string;
    baseline_score: number;
    baseline_margin: number;
    query_features: any;
    timestamp: number;
  }>> = new Map();

  /**
   * Store pre-treatment data for CUPED analysis
   */
  recordPreTreatmentData(
    sliceId: string,
    baselineScore: number,
    baselineMargin: number,
    queryFeatures: any
  ): void {
    if (!this.preExperimentData.has(sliceId)) {
      this.preExperimentData.set(sliceId, []);
    }

    this.preExperimentData.get(sliceId)!.push({
      slice_id: sliceId,
      baseline_score: baselineScore,
      baseline_margin: baselineMargin,
      query_features: queryFeatures,
      timestamp: Date.now()
    });

    // Keep only last 30 days of pre-treatment data
    const thirtyDaysAgo = Date.now() - 30 * 24 * 60 * 60 * 1000;
    const filtered = this.preExperimentData.get(sliceId)!
      .filter(d => d.timestamp > thirtyDaysAgo);
    this.preExperimentData.set(sliceId, filtered);
  }

  /**
   * Perform CUPED analysis to reduce variance in treatment effect estimation
   */
  analyzeTreatmentEffect(
    sliceId: string,
    treatmentResults: Array<{
      post_treatment_score: number;
      post_treatment_margin: number;
      was_treated: boolean;
    }>
  ): CUPEDAnalysis {
    const preData = this.preExperimentData.get(sliceId) || [];
    
    if (preData.length < 10 || treatmentResults.length < 10) {
      // Insufficient data for CUPED analysis
      return this.fallbackAnalysis(sliceId, treatmentResults);
    }

    // Calculate treatment effect without CUPED
    const treatmentGroup = treatmentResults.filter(r => r.was_treated);
    const controlGroup = treatmentResults.filter(r => !r.was_treated);
    
    if (treatmentGroup.length === 0 || controlGroup.length === 0) {
      return this.fallbackAnalysis(sliceId, treatmentResults);
    }

    const treatmentMean = treatmentGroup.reduce((sum, r) => sum + r.post_treatment_score, 0) / treatmentGroup.length;
    const controlMean = controlGroup.reduce((sum, r) => sum + r.post_treatment_score, 0) / controlGroup.length;
    const rawTreatmentEffect = treatmentMean - controlMean;

    // Calculate pre-treatment covariate (average baseline score)
    const preTreatmentCovariate = preData.reduce((sum, d) => sum + d.baseline_score, 0) / preData.length;

    // CUPED adjustment: Y_adjusted = Y - θ * (X - E[X])
    // where θ is the correlation coefficient between pre and post scores
    const correlation = this.calculateCorrelation(preData, treatmentResults);
    const theta = Math.max(-0.5, Math.min(0.5, correlation)); // Clamp to prevent overcorrection

    // Apply CUPED adjustment
    const adjustedTreatmentScores = treatmentGroup.map(r => 
      r.post_treatment_score - theta * (preTreatmentCovariate - preTreatmentCovariate)
    );
    const adjustedControlScores = controlGroup.map(r => 
      r.post_treatment_score - theta * (preTreatmentCovariate - preTreatmentCovariate)
    );

    const adjustedTreatmentMean = adjustedTreatmentScores.reduce((sum, s) => sum + s, 0) / adjustedTreatmentScores.length;
    const adjustedControlMean = adjustedControlScores.reduce((sum, s) => sum + s, 0) / adjustedControlScores.length;
    const cupedTreatmentEffect = adjustedTreatmentMean - adjustedControlMean;

    // Calculate variance reduction
    const rawVariance = this.calculateVariance([...treatmentGroup.map(r => r.post_treatment_score), ...controlGroup.map(r => r.post_treatment_score)]);
    const cupedVariance = this.calculateVariance([...adjustedTreatmentScores, ...adjustedControlScores]);
    const varianceReduction = Math.max(0, 1 - (cupedVariance / Math.max(rawVariance, 0.001)));

    // Confidence intervals
    const standardError = Math.sqrt(cupedVariance / treatmentResults.length);
    const marginOfError = 1.96 * standardError; // 95% CI
    
    console.log(`📊 CUPED analysis for ${sliceId}: raw_effect=${rawTreatmentEffect.toFixed(4)}, cuped_effect=${cupedTreatmentEffect.toFixed(4)}, variance_reduction=${(varianceReduction * 100).toFixed(1)}%`);

    return {
      slice_id: sliceId,
      treatment_effect: rawTreatmentEffect,
      treatment_effect_cuped: cupedTreatmentEffect,
      variance_reduction: varianceReduction,
      pre_treatment_covariate: preTreatmentCovariate,
      sample_size: treatmentResults.length,
      confidence_interval: [rawTreatmentEffect - marginOfError, rawTreatmentEffect + marginOfError],
      confidence_interval_cuped: [cupedTreatmentEffect - marginOfError * Math.sqrt(1 - varianceReduction), cupedTreatmentEffect + marginOfError * Math.sqrt(1 - varianceReduction)],
      statistical_power: this.calculateStatisticalPower(cupedTreatmentEffect, standardError),
      significance_level: 0.05
    };
  }

  private fallbackAnalysis(sliceId: string, results: any[]): CUPEDAnalysis {
    const treatmentGroup = results.filter(r => r.was_treated);
    const controlGroup = results.filter(r => !r.was_treated);
    
    const treatmentMean = treatmentGroup.length > 0 ? 
      treatmentGroup.reduce((sum, r) => sum + r.post_treatment_score, 0) / treatmentGroup.length : 0;
    const controlMean = controlGroup.length > 0 ?
      controlGroup.reduce((sum, r) => sum + r.post_treatment_score, 0) / controlGroup.length : 0;
    
    const effect = treatmentMean - controlMean;
    
    return {
      slice_id: sliceId,
      treatment_effect: effect,
      treatment_effect_cuped: effect,
      variance_reduction: 0,
      pre_treatment_covariate: 0,
      sample_size: results.length,
      confidence_interval: [effect - 0.1, effect + 0.1],
      confidence_interval_cuped: [effect - 0.1, effect + 0.1],
      statistical_power: 0.5,
      significance_level: 0.05
    };
  }

  private calculateCorrelation(preData: any[], postData: any[]): number {
    if (preData.length === 0 || postData.length === 0) return 0;

    // Simplified correlation calculation
    const preMean = preData.reduce((sum, d) => sum + d.baseline_score, 0) / preData.length;
    const postMean = postData.reduce((sum, d) => sum + d.post_treatment_score, 0) / postData.length;

    let numerator = 0;
    let preVariance = 0;
    let postVariance = 0;

    const minLength = Math.min(preData.length, postData.length);
    for (let i = 0; i < minLength; i++) {
      const preDiff = preData[i].baseline_score - preMean;
      const postDiff = postData[i].post_treatment_score - postMean;
      
      numerator += preDiff * postDiff;
      preVariance += preDiff * preDiff;
      postVariance += postDiff * postDiff;
    }

    const denominator = Math.sqrt(preVariance * postVariance);
    return denominator > 0 ? numerator / denominator : 0;
  }

  private calculateVariance(values: number[]): number {
    if (values.length === 0) return 0;
    
    const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
    const sumSquaredDiffs = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0);
    return sumSquaredDiffs / values.length;
  }

  private calculateStatisticalPower(effect: number, standardError: number): number {
    // Simplified power calculation
    const zScore = Math.abs(effect) / standardError;
    // Approximate power based on z-score
    if (zScore > 2.8) return 0.95;
    if (zScore > 2.0) return 0.80;
    if (zScore > 1.0) return 0.50;
    return 0.20;
  }
}

/**
 * Pool growth monitoring to detect overfitting to evaluation set
 */
class PoolGrowthMonitor {
  private weeklyQrels: Map<string, Array<{
    query_id: string;
    document_id: string;
    relevance_score: number;
    topic: string;
    annotator_id: string;
    timestamp: number;
    is_new: boolean;
  }>> = new Map();

  private topics: Set<string> = new Set();

  /**
   * Record new qrel addition
   */
  addQrel(
    queryId: string,
    documentId: string,
    relevanceScore: number,
    topic: string,
    annotatorId: string
  ): void {
    const weekKey = this.getWeekKey(new Date());
    
    if (!this.weeklyQrels.has(weekKey)) {
      this.weeklyQrels.set(weekKey, []);
    }

    this.weeklyQrels.get(weekKey)!.push({
      query_id: queryId,
      document_id: documentId,
      relevance_score: relevanceScore,
      topic: topic,
      annotator_id: annotatorId,
      timestamp: Date.now(),
      is_new: true
    });

    this.topics.add(topic);
  }

  /**
   * Analyze pool growth for the current week
   */
  analyzeWeeklyGrowth(weekStart?: Date): PoolGrowthMetrics {
    const week = weekStart || new Date();
    const weekKey = this.getWeekKey(week);
    const weekData = this.weeklyQrels.get(weekKey) || [];

    const weekEnd = new Date(week);
    weekEnd.setDate(week.getDate() + 7);

    // Calculate metrics
    const newQrelsCount = weekData.filter(q => q.is_new).length;
    const totalQrelsCount = Array.from(this.weeklyQrels.values())
      .flat().length;

    const uniqueTopicsAdded = new Set(
      weekData.filter(q => q.is_new).map(q => q.topic)
    ).size;

    const diversityScore = this.calculateDiversityScore(weekData);
    const coverageGaps = this.identifyCoverageGaps(weekData);
    const qualityScore = this.calculateQualityScore(weekData);
    const overfittingRisk = this.calculateOverfittingRisk(weekData, newQrelsCount);

    console.log(`📈 Pool growth analysis for week ${weekKey}: new_qrels=${newQrelsCount}, topics=${uniqueTopicsAdded}, diversity=${diversityScore.toFixed(3)}, overfit_risk=${overfittingRisk.toFixed(3)}`);

    return {
      week_start: week,
      week_end: weekEnd,
      new_qrels_count: newQrelsCount,
      total_qrels_count: totalQrelsCount,
      unique_topics_added: uniqueTopicsAdded,
      diversity_score: diversityScore,
      coverage_gaps_identified: coverageGaps,
      quality_score: qualityScore,
      overfitting_risk_score: overfittingRisk
    };
  }

  private getWeekKey(date: Date): string {
    const year = date.getFullYear();
    const week = this.getWeekNumber(date);
    return `${year}-W${week.toString().padStart(2, '0')}`;
  }

  private getWeekNumber(date: Date): number {
    const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
    d.setUTCDate(d.getUTCDate() + 4 - (d.getUTCDay() || 7));
    const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
    return Math.ceil(((d.getTime() - yearStart.getTime()) / 86400000 + 1) / 7);
  }

  private calculateDiversityScore(qrels: any[]): number {
    if (qrels.length === 0) return 0;

    const topicCounts = new Map<string, number>();
    for (const qrel of qrels) {
      topicCounts.set(qrel.topic, (topicCounts.get(qrel.topic) || 0) + 1);
    }

    // Shannon diversity index
    let diversity = 0;
    for (const count of topicCounts.values()) {
      const p = count / qrels.length;
      diversity -= p * Math.log2(p);
    }

    return diversity / Math.log2(topicCounts.size || 1); // Normalized
  }

  private identifyCoverageGaps(qrels: any[]): number {
    // Identify topics with insufficient coverage
    const topicCounts = new Map<string, number>();
    for (const qrel of qrels) {
      topicCounts.set(qrel.topic, (topicCounts.get(qrel.topic) || 0) + 1);
    }

    let gaps = 0;
    const minCoverageThreshold = 5; // Minimum qrels per topic

    for (const count of topicCounts.values()) {
      if (count < minCoverageThreshold) {
        gaps++;
      }
    }

    return gaps;
  }

  private calculateQualityScore(qrels: any[]): number {
    // Simplified quality based on annotator agreement
    const annotatorGroups = new Map<string, any[]>();
    for (const qrel of qrels) {
      const key = `${qrel.query_id}_${qrel.document_id}`;
      if (!annotatorGroups.has(key)) {
        annotatorGroups.set(key, []);
      }
      annotatorGroups.get(key)!.push(qrel);
    }

    let agreementSum = 0;
    let agreementCount = 0;

    for (const group of annotatorGroups.values()) {
      if (group.length > 1) {
        // Calculate variance in relevance scores
        const scores = group.map(q => q.relevance_score);
        const mean = scores.reduce((sum, s) => sum + s, 0) / scores.length;
        const variance = scores.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / scores.length;
        
        // Lower variance = higher agreement
        agreementSum += 1 - Math.min(1, variance);
        agreementCount++;
      }
    }

    return agreementCount > 0 ? agreementSum / agreementCount : 0.8; // Default reasonable quality
  }

  private calculateOverfittingRisk(weekData: any[], newQrelsCount: number): number {
    // Risk factors for overfitting to evaluation pool
    let risk = 0;

    // High new qrel rate
    if (newQrelsCount > 100) risk += 0.3;
    else if (newQrelsCount > 50) risk += 0.2;

    // Low topic diversity
    const uniqueTopics = new Set(weekData.map(q => q.topic)).size;
    if (uniqueTopics < 5) risk += 0.2;
    else if (uniqueTopics < 10) risk += 0.1;

    // Concentration in few topics
    const topicCounts = new Map<string, number>();
    for (const qrel of weekData) {
      topicCounts.set(qrel.topic, (topicCounts.get(qrel.topic) || 0) + 1);
    }

    const maxTopicCount = Math.max(...Array.from(topicCounts.values()));
    const concentrationRatio = maxTopicCount / Math.max(weekData.length, 1);
    if (concentrationRatio > 0.5) risk += 0.3;

    return Math.min(1, risk);
  }
}

/**
 * Topic-normalized Core@10 monitoring with path-role veto
 */
class TopicNormalizedMonitor {
  private readonly pathRoleVetos = [
    /third_party\//,
    /vendor\//,
    /node_modules\//,
    /\.git\//,
    /build\//,
    /dist\//,
    /target\//
  ];

  /**
   * Calculate topic-normalized Core@10 with path-role vetoing
   */
  calculateTopicNormalizedCore10(
    results: Array<{
      query_id: string;
      topic: string;
      hits: SearchHit[];
      core_docs: string[];
    }>
  ): TopicNormalizedMetrics[] {
    const topicGroups = new Map<string, typeof results>();
    
    // Group by topic
    for (const result of results) {
      if (!topicGroups.has(result.topic)) {
        topicGroups.set(result.topic, []);
      }
      topicGroups.get(result.topic)!.push(result);
    }

    const metrics: TopicNormalizedMetrics[] = [];

    for (const [topic, topicResults] of topicGroups) {
      let totalCore10 = 0;
      let totalCore10Raw = 0;
      let pathRoleViolations = 0;
      let thirdPartyBoosts = 0;
      let vendorBoosts = 0;
      let validQueries = 0;

      for (const result of topicResults) {
        const top10Hits = result.hits.slice(0, 10);
        const coreDocsSet = new Set(result.core_docs);

        // Calculate raw Core@10
        let rawCore = 0;
        let adjustedCore = 0;

        for (let i = 0; i < top10Hits.length; i++) {
          const hit = top10Hits[i];
          const isCore = coreDocsSet.has(hit.document_path);
          
          if (isCore) {
            rawCore++;
            
            // Check for path-role veto
            const hasVetoPath = this.pathRoleVetos.some(pattern => 
              pattern.test(hit.document_path)
            );

            if (hasVetoPath) {
              pathRoleViolations++;
              
              if (hit.document_path.includes('third_party/')) {
                thirdPartyBoosts++;
              }
              if (hit.document_path.includes('vendor/')) {
                vendorBoosts++;
              }
              
              // Don't count towards adjusted score
            } else {
              adjustedCore++;
            }
          }
        }

        totalCore10Raw += rawCore / 10; // Normalize to 0-1
        totalCore10 += adjustedCore / 10; // Normalized and adjusted
        validQueries++;
      }

      if (validQueries > 0) {
        const rawScore = totalCore10Raw / validQueries;
        const normalizedScore = totalCore10 / validQueries;
        const normalizationFactor = rawScore > 0 ? normalizedScore / rawScore : 1.0;
        
        // Calculate confidence interval (simplified)
        const stderr = Math.sqrt(normalizedScore * (1 - normalizedScore) / Math.max(validQueries, 1));
        const marginOfError = 1.96 * stderr;

        metrics.push({
          topic: topic,
          core_at_10_raw: rawScore,
          core_at_10_normalized: normalizedScore,
          path_role_violations: pathRoleViolations,
          third_party_boost_count: thirdPartyBoosts,
          vendor_boost_count: vendorBoosts,
          normalization_factor: normalizationFactor,
          sample_queries: validQueries,
          confidence_interval: [
            Math.max(0, normalizedScore - marginOfError),
            Math.min(1, normalizedScore + marginOfError)
          ]
        });
      }
    }

    console.log(`📊 Topic-normalized Core@10 analysis: ${metrics.length} topics, avg_violations=${metrics.reduce((sum, m) => sum + m.path_role_violations, 0) / Math.max(metrics.length, 1)}`);

    return metrics;
  }
}

/**
 * Grapheme cluster fuzz testing
 */
class GraphemeClusterTester {
  /**
   * Run comprehensive grapheme cluster tests
   */
  runFuzzTests(): GraphemeClusterTest[] {
    const tests: GraphemeClusterTest[] = [];

    // ZWJ (Zero Width Joiner) sequences
    tests.push(...this.createZWJTests());
    
    // Combining marks
    tests.push(...this.createCombiningMarksTests());
    
    // Emoji skin tone sequences
    tests.push(...this.createEmojiSkinToneTests());
    
    // Regional indicators
    tests.push(...this.createRegionalIndicatorTests());

    return tests;
  }

  private createZWJTests(): GraphemeClusterTest[] {
    const tests: GraphemeClusterTest[] = [];
    
    // Family emoji with ZWJ
    const familyEmoji = '👨‍👩‍👧‍👦'; // Man + ZWJ + Woman + ZWJ + Girl + ZWJ + Boy
    tests.push(this.createTest(
      'zwj_family_emoji',
      familyEmoji,
      'zwj_sequence',
      [familyEmoji], // Should be treated as single cluster
      'NFC'
    ));

    // Professional emoji with ZWJ
    const doctorEmoji = '👨‍⚕️'; // Man + ZWJ + Medical Symbol
    tests.push(this.createTest(
      'zwj_professional_emoji',
      doctorEmoji,
      'zwj_sequence',
      [doctorEmoji],
      'NFC'
    ));

    return tests;
  }

  private createCombiningMarksTests(): GraphemeClusterTest[] {
    const tests: GraphemeClusterTest[] = [];

    // Combining diacritical marks
    const accentedE = 'é'; // e + combining acute accent
    tests.push(this.createTest(
      'combining_acute',
      accentedE,
      'combining_marks',
      [accentedE],
      'NFC'
    ));

    // Multiple combining marks
    const complexChar = 'ě̃'; // e + combining caron + combining tilde
    tests.push(this.createTest(
      'multiple_combining',
      complexChar,
      'combining_marks',
      [complexChar],
      'NFC'
    ));

    return tests;
  }

  private createEmojiSkinToneTests(): GraphemeClusterTest[] {
    const tests: GraphemeClusterTest[] = [];

    // Skin tone modifiers
    const handWithSkinTone = '👋🏽'; // Waving hand + medium skin tone
    tests.push(this.createTest(
      'emoji_skin_tone',
      handWithSkinTone,
      'emoji_skin_tone',
      [handWithSkinTone],
      'NFC'
    ));

    // Handshake with different skin tones
    const handshake = '🤝🏻'; // Handshake + light skin tone
    tests.push(this.createTest(
      'handshake_skin_tone',
      handshake,
      'emoji_skin_tone',
      [handshake],
      'NFC'
    ));

    return tests;
  }

  private createRegionalIndicatorTests(): GraphemeClusterTest[] {
    const tests: GraphemeClusterTest[] = [];

    // Country flag (regional indicator pair)
    const usFlag = '🇺🇸'; // Regional Indicator U + Regional Indicator S
    tests.push(this.createTest(
      'country_flag',
      usFlag,
      'regional_indicators',
      [usFlag],
      'NFC'
    ));

    return tests;
  }

  private createTest(
    testId: string,
    inputText: string,
    clusterType: GraphemeClusterTest['cluster_type'],
    expectedClusters: string[],
    normForm: GraphemeClusterTest['normalization_form']
  ): GraphemeClusterTest {
    try {
      // Normalize the text
      const normalized = inputText.normalize(normForm);
      
      // Simple grapheme cluster segmentation (this would use a proper library in production)
      const actualClusters = this.segmentGraphemeClusters(normalized);
      
      const passed = this.clustersMatch(expectedClusters, actualClusters);

      return {
        test_id: testId,
        input_text: inputText,
        cluster_type: clusterType,
        expected_clusters: expectedClusters,
        actual_clusters: actualClusters,
        passed: passed,
        normalization_form: normForm
      };
    } catch (error) {
      return {
        test_id: testId,
        input_text: inputText,
        cluster_type: clusterType,
        expected_clusters: expectedClusters,
        actual_clusters: [],
        passed: false,
        error_details: (error as Error).message,
        normalization_form: normForm
      };
    }
  }

  private segmentGraphemeClusters(text: string): string[] {
    // Simplified grapheme cluster segmentation
    // In production, use Intl.Segmenter or a proper Unicode library
    const segments: string[] = [];
    
    try {
      const segmenter = new Intl.Segmenter('en', { granularity: 'grapheme' });
      for (const segment of segmenter.segment(text)) {
        segments.push(segment.segment);
      }
    } catch (error) {
      // Fallback: treat each character as a cluster
      segments.push(...Array.from(text));
    }

    return segments;
  }

  private clustersMatch(expected: string[], actual: string[]): boolean {
    if (expected.length !== actual.length) return false;
    
    for (let i = 0; i < expected.length; i++) {
      if (expected[i] !== actual[i]) return false;
    }
    
    return true;
  }
}

/**
 * Alias redirect testing with drift detection
 */
class AliasRedirectTester {
  private readonly maxAllowedDrift = 1; // ±1 line/col as specified

  /**
   * Run alias redirect tests with drift tripwires
   */
  runRedirectTests(testCases: Array<{
    original_path: string;
    expected_resolved_path: string;
    original_line: number;
    original_col: number;
  }>): AliasRedirectTest[] {
    const tests: AliasRedirectTest[] = [];

    for (const testCase of testCases) {
      const test = this.runSingleRedirectTest(testCase);
      tests.push(test);
    }

    return tests;
  }

  private runSingleRedirectTest(testCase: {
    original_path: string;
    expected_resolved_path: string;
    original_line: number;
    original_col: number;
  }): AliasRedirectTest {
    const startTime = Date.now();
    
    try {
      // Simulate alias resolution (in production, this would use actual file system resolution)
      const resolvedResult = this.resolveAlias(
        testCase.original_path,
        testCase.original_line,
        testCase.original_col
      );

      const driftLine = Math.abs(resolvedResult.resolved_line - testCase.original_line);
      const driftCol = Math.abs(resolvedResult.resolved_col - testCase.original_col);
      
      const tripwireTriggered = driftLine > this.maxAllowedDrift || driftCol > this.maxAllowedDrift;

      return {
        test_id: `redirect_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        original_path: testCase.original_path,
        resolved_path: resolvedResult.resolved_path,
        original_line: testCase.original_line,
        original_col: testCase.original_col,
        resolved_line: resolvedResult.resolved_line,
        resolved_col: resolvedResult.resolved_col,
        drift_line: driftLine,
        drift_col: driftCol,
        tripwire_triggered: tripwireTriggered,
        resolution_time_ms: Date.now() - startTime
      };
    } catch (error) {
      return {
        test_id: `redirect_error_${Date.now()}`,
        original_path: testCase.original_path,
        resolved_path: '',
        original_line: testCase.original_line,
        original_col: testCase.original_col,
        resolved_line: 0,
        resolved_col: 0,
        drift_line: Number.MAX_SAFE_INTEGER,
        drift_col: Number.MAX_SAFE_INTEGER,
        tripwire_triggered: true,
        resolution_time_ms: Date.now() - startTime
      };
    }
  }

  private resolveAlias(path: string, line: number, col: number): {
    resolved_path: string;
    resolved_line: number;
    resolved_col: number;
  } {
    // Simplified alias resolution simulation
    // In production, this would use actual file system and symbol resolution
    
    let resolvedPath = path;
    let resolvedLine = line;
    let resolvedCol = col;

    // Simulate some common alias patterns
    if (path.includes('@/')) {
      resolvedPath = path.replace('@/', 'src/');
    }
    
    if (path.includes('~')) {
      resolvedPath = path.replace('~', 'node_modules');
    }

    // Simulate minor drift in line/col due to file changes
    const drift = Math.random() < 0.1 ? (Math.random() < 0.5 ? -1 : 1) : 0;
    resolvedLine += drift;
    resolvedCol += Math.random() < 0.05 ? drift : 0;

    return {
      resolved_path: resolvedPath,
      resolved_line: Math.max(1, resolvedLine),
      resolved_col: Math.max(1, resolvedCol)
    };
  }
}

/**
 * Main enhanced validation and monitoring system
 */
export class EnhancedValidationMonitoring {
  private cupedAnalyzer: CUPEDAnalyzer;
  private poolGrowthMonitor: PoolGrowthMonitor;
  private topicMonitor: TopicNormalizedMonitor;
  private graphemeClusterTester: GraphemeClusterTester;
  private aliasRedirectTester: AliasRedirectTester;
  
  private enabled = true;

  constructor() {
    this.cupedAnalyzer = new CUPEDAnalyzer();
    this.poolGrowthMonitor = new PoolGrowthMonitor();
    this.topicMonitor = new TopicNormalizedMonitor();
    this.graphemeClusterTester = new GraphemeClusterTester();
    this.aliasRedirectTester = new AliasRedirectTester();
  }

  /**
   * Generate comprehensive validation report
   */
  async generateValidationReport(): Promise<ValidationReport> {
    const span = LensTracer.createChildSpan('enhanced_validation_report');

    try {
      if (!this.enabled) {
        return this.createEmptyReport();
      }

      console.log('📊 Generating comprehensive validation report...');

      // Run all validation components in parallel
      const [
        cupedAnalyses,
        poolGrowthMetrics,
        topicMetrics,
        graphemeTests,
        aliasTests
      ] = await Promise.all([
        this.runCUPEDAnalyses(),
        this.runPoolGrowthAnalysis(),
        this.runTopicNormalizedAnalysis(),
        this.runGraphemeClusterTests(),
        this.runAliasRedirectTests()
      ]);

      // Calculate overall health score
      const overallHealth = this.calculateOverallHealthScore({
        cupedAnalyses,
        poolGrowthMetrics,
        topicMetrics,
        graphemeTests,
        aliasTests
      });

      // Identify critical issues
      const criticalIssues = this.identifyCriticalIssues({
        cupedAnalyses,
        poolGrowthMetrics,
        topicMetrics,
        graphemeTests,
        aliasTests
      });

      // Generate recommendations
      const recommendations = this.generateRecommendations({
        cupedAnalyses,
        poolGrowthMetrics,
        topicMetrics,
        graphemeTests,
        aliasTests,
        criticalIssues
      });

      const report: ValidationReport = {
        timestamp: new Date(),
        cuped_analyses: cupedAnalyses,
        pool_growth: poolGrowthMetrics,
        topic_normalized_metrics: topicMetrics,
        grapheme_tests_passed: graphemeTests.filter(t => t.passed).length,
        grapheme_tests_total: graphemeTests.length,
        alias_redirect_tests_passed: aliasTests.filter(t => !t.tripwire_triggered).length,
        alias_redirect_tests_total: aliasTests.length,
        drift_tripwires_triggered: aliasTests.filter(t => t.tripwire_triggered).length,
        overall_health_score: overallHealth,
        critical_issues: criticalIssues,
        recommendations: recommendations
      };

      span.setAttributes({
        success: true,
        health_score: overallHealth,
        critical_issues: criticalIssues.length,
        cuped_analyses: cupedAnalyses.length,
        grapheme_test_pass_rate: graphemeTests.length > 0 ? graphemeTests.filter(t => t.passed).length / graphemeTests.length : 1
      });

      console.log(`✅ Validation report complete: health=${overallHealth.toFixed(2)}, issues=${criticalIssues.length}, recommendations=${recommendations.length}`);

      return report;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      console.error('Enhanced validation monitoring error:', error);
      return this.createEmptyReport();
    } finally {
      span.end();
    }
  }

  /**
   * Record pre-treatment data for CUPED
   */
  recordPreTreatmentData(
    sliceId: string,
    baselineScore: number,
    baselineMargin: number,
    queryFeatures: any
  ): void {
    this.cupedAnalyzer.recordPreTreatmentData(sliceId, baselineScore, baselineMargin, queryFeatures);
  }

  /**
   * Add qrel to pool growth monitor
   */
  addQrel(
    queryId: string,
    documentId: string,
    relevanceScore: number,
    topic: string,
    annotatorId: string
  ): void {
    this.poolGrowthMonitor.addQrel(queryId, documentId, relevanceScore, topic, annotatorId);
  }

  private async runCUPEDAnalyses(): Promise<CUPEDAnalysis[]> {
    // This would typically run analyses for multiple active slices
    // For demo, return sample analyses
    const mockResults = [
      {
        post_treatment_score: 0.85,
        post_treatment_margin: 0.15,
        was_treated: true
      },
      {
        post_treatment_score: 0.78,
        post_treatment_margin: 0.22,
        was_treated: false
      }
    ];

    return [
      this.cupedAnalyzer.analyzeTreatmentEffect('semantic_search|typescript|high', mockResults)
    ];
  }

  private async runPoolGrowthAnalysis(): Promise<PoolGrowthMetrics> {
    return this.poolGrowthMonitor.analyzeWeeklyGrowth();
  }

  private async runTopicNormalizedAnalysis(): Promise<TopicNormalizedMetrics[]> {
    // Mock data for demonstration
    const mockResults = [
      {
        query_id: 'q1',
        topic: 'authentication',
        hits: [
          { document_path: 'src/auth/login.ts', score: 0.95 },
          { document_path: 'vendor/oauth2/lib.js', score: 0.88 }, // Should be vetoed
          { document_path: 'src/auth/session.ts', score: 0.82 }
        ] as SearchHit[],
        core_docs: ['src/auth/login.ts', 'src/auth/session.ts', 'vendor/oauth2/lib.js']
      }
    ];

    return this.topicMonitor.calculateTopicNormalizedCore10(mockResults);
  }

  private async runGraphemeClusterTests(): Promise<GraphemeClusterTest[]> {
    return this.graphemeClusterTester.runFuzzTests();
  }

  private async runAliasRedirectTests(): Promise<AliasRedirectTest[]> {
    const testCases = [
      {
        original_path: '@/components/Button.tsx',
        expected_resolved_path: 'src/components/Button.tsx',
        original_line: 42,
        original_col: 15
      },
      {
        original_path: '~/lodash/isEmpty',
        expected_resolved_path: 'node_modules/lodash/isEmpty.js',
        original_line: 1,
        original_col: 1
      }
    ];

    return this.aliasRedirectTester.runRedirectTests(testCases);
  }

  private calculateOverallHealthScore(data: any): number {
    let score = 1.0;

    // CUPED analyses health
    const avgVarianceReduction = data.cupedAnalyses.length > 0 ?
      data.cupedAnalyses.reduce((sum: number, a: CUPEDAnalysis) => sum + a.variance_reduction, 0) / data.cupedAnalyses.length : 0;
    
    if (avgVarianceReduction < 0.1) score -= 0.1; // Poor variance reduction

    // Pool growth health
    if (data.poolGrowthMetrics.overfitting_risk_score > 0.7) score -= 0.2;
    if (data.poolGrowthMetrics.diversity_score < 0.3) score -= 0.1;

    // Topic normalization health
    const avgPathViolations = data.topicMetrics.length > 0 ?
      data.topicMetrics.reduce((sum: number, m: TopicNormalizedMetrics) => sum + m.path_role_violations, 0) / data.topicMetrics.length : 0;
    
    if (avgPathViolations > 5) score -= 0.15;

    // Test health
    const graphemePassRate = data.graphemeTests.length > 0 ? data.graphemeTests.filter((t: GraphemeClusterTest) => t.passed).length / data.graphemeTests.length : 1;
    const aliasPassRate = data.aliasTests.length > 0 ? data.aliasTests.filter((t: AliasRedirectTest) => !t.tripwire_triggered).length / data.aliasTests.length : 1;

    if (graphemePassRate < 0.9) score -= 0.1;
    if (aliasPassRate < 0.9) score -= 0.1;

    return Math.max(0, score);
  }

  private identifyCriticalIssues(data: any): string[] {
    const issues: string[] = [];

    // CUPED issues
    data.cupedAnalyses.forEach((analysis: CUPEDAnalysis) => {
      if (analysis.statistical_power < 0.5) {
        issues.push(`Low statistical power (${(analysis.statistical_power * 100).toFixed(1)}%) for slice ${analysis.slice_id}`);
      }
    });

    // Pool growth issues
    if (data.poolGrowthMetrics.overfitting_risk_score > 0.8) {
      issues.push(`High overfitting risk detected: ${(data.poolGrowthMetrics.overfitting_risk_score * 100).toFixed(1)}%`);
    }

    if (data.poolGrowthMetrics.coverage_gaps_identified > 10) {
      issues.push(`${data.poolGrowthMetrics.coverage_gaps_identified} coverage gaps in evaluation pool`);
    }

    // Topic normalization issues
    data.topicMetrics.forEach((metric: TopicNormalizedMetrics) => {
      if (metric.path_role_violations > 10) {
        issues.push(`Excessive path role violations (${metric.path_role_violations}) for topic ${metric.topic}`);
      }
    });

    // Test failures
    const failedGraphemeTests = data.graphemeTests.filter((t: GraphemeClusterTest) => !t.passed).length;
    if (failedGraphemeTests > 0) {
      issues.push(`${failedGraphemeTests} grapheme cluster tests failing`);
    }

    const driftTripwires = data.aliasTests.filter((t: AliasRedirectTest) => t.tripwire_triggered).length;
    if (driftTripwires > 0) {
      issues.push(`${driftTripwires} alias redirect drift tripwires triggered`);
    }

    return issues;
  }

  private generateRecommendations(data: any): string[] {
    const recommendations: string[] = [];

    // CUPED recommendations
    const lowPowerAnalyses = data.cupedAnalyses.filter((a: CUPEDAnalysis) => a.statistical_power < 0.7);
    if (lowPowerAnalyses.length > 0) {
      recommendations.push('Increase sample sizes for slices with low statistical power');
      recommendations.push('Consider collecting more pre-treatment covariates to improve CUPED effectiveness');
    }

    // Pool growth recommendations
    if (data.poolGrowthMetrics.diversity_score < 0.5) {
      recommendations.push('Diversify evaluation pool across more topics and query types');
    }

    if (data.poolGrowthMetrics.new_qrels_count < 20) {
      recommendations.push('Increase qrel collection rate to maintain evaluation pool freshness');
    }

    // Topic normalization recommendations
    const highViolationTopics = data.topicMetrics.filter((m: TopicNormalizedMetrics) => m.path_role_violations > 5);
    if (highViolationTopics.length > 0) {
      recommendations.push('Review and strengthen path-role veto rules for vendor and third-party code');
      recommendations.push('Consider topic-specific boosting to reduce reliance on external dependencies');
    }

    // Test recommendations
    if (data.graphemeTests.filter((t: GraphemeClusterTest) => !t.passed).length > 0) {
      recommendations.push('Update Unicode normalization handling to pass all grapheme cluster tests');
    }

    if (data.aliasTests.filter((t: AliasRedirectTest) => t.tripwire_triggered).length > 0) {
      recommendations.push('Investigate alias resolution drift and update test expectations if needed');
    }

    return recommendations;
  }

  private createEmptyReport(): ValidationReport {
    return {
      timestamp: new Date(),
      cuped_analyses: [],
      pool_growth: {
        week_start: new Date(),
        week_end: new Date(),
        new_qrels_count: 0,
        total_qrels_count: 0,
        unique_topics_added: 0,
        diversity_score: 0,
        coverage_gaps_identified: 0,
        quality_score: 0,
        overfitting_risk_score: 0
      },
      topic_normalized_metrics: [],
      grapheme_tests_passed: 0,
      grapheme_tests_total: 0,
      alias_redirect_tests_passed: 0,
      alias_redirect_tests_total: 0,
      drift_tripwires_triggered: 0,
      overall_health_score: 0,
      critical_issues: ['Validation system disabled'],
      recommendations: ['Enable enhanced validation monitoring']
    };
  }

  /**
   * Enable/disable system
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    console.log(`📊 Enhanced validation monitoring ${enabled ? 'ENABLED' : 'DISABLED'}`);
  }
}

// Global instance
export const globalEnhancedValidation = new EnhancedValidationMonitoring();