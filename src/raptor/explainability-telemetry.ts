/**
 * Explainability & Telemetry for RAPTOR System
 * 
 * Provides detailed explanations for RAPTOR decisions, topic hit reasons,
 * and span coverage validation for transparency and debugging.
 * 
 * Features:
 * - Topic hit explanations with confidence scores
 * - Feature contribution analysis
 * - Span coverage validation and gap detection
 * - Query-level decision tracing
 * - Performance impact attribution
 */

import { EventEmitter } from 'events';
import { TopicTree, TopicSearchResult } from './topic-tree.js';
import { CardStore, EnhancedSemanticCard } from './card-store.js';
import { StageCRaptorFeatures, RaptorFeatures } from './stage-c-features.js';
import { TopicPlanningResult, TopicMatch } from './stage-a-planner.js';

export interface ExplainabilityReport {
  query_id: string;
  query_text: string;
  query_intent: string;
  timestamp: string;
  
  // Topic analysis
  topic_analysis: TopicAnalysis;
  
  // Feature explanations
  feature_explanations: FeatureExplanation[];
  
  // Span coverage analysis
  span_coverage: SpanCoverageAnalysis;
  
  // Decision trace
  decision_trace: DecisionTraceStep[];
  
  // Performance attribution
  performance_attribution: PerformanceAttribution;
  
  // Quality metrics
  quality_metrics: QualityMetrics;
}

export interface TopicAnalysis {
  topics_matched: TopicHitExplanation[];
  topics_considered: number;
  topic_diversity_score: number;
  coverage_gaps: CoverageGap[];
  similarity_distribution: number[];
  entropy_score: number;
}

export interface TopicHitExplanation {
  topic_id: string;
  topic_summary: string;
  confidence_score: number;
  relevance_score: number;
  hit_reasons: HitReason[];
  contributing_cards: string[];
  semantic_distance: number;
  keyword_overlap_score: number;
  business_relevance: number;
}

export interface HitReason {
  type: 'semantic_similarity' | 'keyword_match' | 'business_context' | 'structural_pattern';
  explanation: string;
  confidence: number;
  evidence: string[];
  weight_contribution: number;
}

export interface CoverageGap {
  query_aspect: string;
  missing_coverage: string;
  suggested_improvements: string[];
  impact_severity: 'low' | 'medium' | 'high';
}

export interface FeatureExplanation {
  feature_name: string;
  feature_type: 'raptor' | 'lsp' | 'cross';
  value: number;
  weight: number;
  contribution: number;
  confidence: number;
  explanation: string;
  supporting_evidence: Evidence[];
  alternatives_considered: AlternativeFeature[];
}

export interface Evidence {
  type: 'topic_match' | 'symbol_alignment' | 'semantic_similarity' | 'structural_pattern';
  description: string;
  strength: number;
  source_data: any;
}

export interface AlternativeFeature {
  feature_name: string;
  potential_value: number;
  rejection_reason: string;
}

export interface SpanCoverageAnalysis {
  total_spans_analyzed: number;
  spans_covered_by_topics: number;
  coverage_percentage: number;
  uncovered_spans: UncoveredSpan[];
  coverage_quality: CoverageQuality;
  redundancy_analysis: RedundancyAnalysis;
}

export interface UncoveredSpan {
  file_path: string;
  span_start: number;
  span_end: number;
  content_preview: string;
  estimated_relevance: number;
  gap_reasons: string[];
  potential_topics: string[];
}

export interface CoverageQuality {
  precision: number; // How many covered spans are actually relevant
  recall: number; // How many relevant spans are covered
  f1_score: number;
  coverage_distribution: Record<string, number>; // by file type, domain, etc.
}

export interface RedundancyAnalysis {
  overlapping_topics: TopicOverlap[];
  redundancy_score: number;
  optimization_suggestions: string[];
}

export interface TopicOverlap {
  topic1_id: string;
  topic2_id: string;
  overlap_percentage: number;
  shared_cards: string[];
  consolidation_benefit: number;
}

export interface DecisionTraceStep {
  step_number: number;
  component: 'stage_a' | 'stage_c' | 'nl_bridge' | 'topic_tree';
  action: string;
  input_data: any;
  output_data: any;
  processing_time_ms: number;
  confidence_score: number;
  decision_factors: DecisionFactor[];
}

export interface DecisionFactor {
  factor_name: string;
  factor_value: any;
  influence_weight: number;
  rationale: string;
}

export interface PerformanceAttribution {
  total_processing_time_ms: number;
  component_breakdown: ComponentTiming[];
  bottlenecks: PerformanceBottleneck[];
  optimization_opportunities: OptimizationOpportunity[];
}

export interface ComponentTiming {
  component: string;
  processing_time_ms: number;
  percentage_of_total: number;
  operations_count: number;
  avg_time_per_operation: number;
}

export interface PerformanceBottleneck {
  component: string;
  issue: string;
  impact_ms: number;
  severity: 'low' | 'medium' | 'high';
  suggested_fix: string;
}

export interface OptimizationOpportunity {
  opportunity: string;
  estimated_improvement_ms: number;
  implementation_effort: 'low' | 'medium' | 'high';
  risk_level: 'low' | 'medium' | 'high';
}

export interface QualityMetrics {
  explanation_completeness: number; // 0-1
  confidence_calibration: number; // How well confidence scores match actual performance
  decision_consistency: number; // Consistency across similar queries
  user_comprehensibility: number; // Estimated user understanding score
  actionability_score: number; // How actionable are the explanations
}

export interface QueryTrace {
  query_id: string;
  stages: TraceStage[];
  total_time_ms: number;
  memory_usage_mb: number;
  cache_hits: number;
  cache_misses: number;
}

export interface TraceStage {
  stage_name: string;
  start_time_ms: number;
  end_time_ms: number;
  input_size: number;
  output_size: number;
  operations: TraceOperation[];
}

export interface TraceOperation {
  operation: string;
  duration_ms: number;
  success: boolean;
  error_message?: string;
  parameters: Record<string, any>;
}

/**
 * Explainability and telemetry system for RAPTOR
 */
export class ExplainabilityTelemetry extends EventEmitter {
  private traces: Map<string, QueryTrace> = new Map();
  private reports: Map<string, ExplainabilityReport> = new Map();
  
  constructor(
    private topicTree?: TopicTree,
    private cardStore?: CardStore,
    private raptorFeatures?: StageCRaptorFeatures
  ) {
    super();
    
    // Clean up old traces periodically
    setInterval(() => this.cleanupOldTraces(), 5 * 60 * 1000); // Every 5 minutes
  }

  /**
   * Initialize with RAPTOR components
   */
  initialize(
    topicTree: TopicTree,
    cardStore: CardStore,
    raptorFeatures: StageCRaptorFeatures
  ): void {
    this.topicTree = topicTree;
    this.cardStore = cardStore;
    this.raptorFeatures = raptorFeatures;
  }

  /**
   * Start tracing a query
   */
  startQueryTrace(queryId: string): QueryTrace {
    const trace: QueryTrace = {
      query_id: queryId,
      stages: [],
      total_time_ms: 0,
      memory_usage_mb: 0,
      cache_hits: 0,
      cache_misses: 0
    };
    
    this.traces.set(queryId, trace);
    this.emit('trace_started', { query_id: queryId });
    
    return trace;
  }

  /**
   * Add a stage to the query trace
   */
  addTraceStage(queryId: string, stageName: string): TraceStage {
    const trace = this.traces.get(queryId);
    if (!trace) {
      throw new Error(`No trace found for query: ${queryId}`);
    }

    const stage: TraceStage = {
      stage_name: stageName,
      start_time_ms: Date.now(),
      end_time_ms: 0,
      input_size: 0,
      output_size: 0,
      operations: []
    };

    trace.stages.push(stage);
    return stage;
  }

  /**
   * Complete a trace stage
   */
  completeTraceStage(queryId: string, stageName: string, outputSize: number = 0): void {
    const trace = this.traces.get(queryId);
    if (!trace) return;

    const stage = trace.stages.find(s => s.stage_name === stageName);
    if (stage) {
      stage.end_time_ms = Date.now();
      stage.output_size = outputSize;
    }
  }

  /**
   * Generate comprehensive explainability report
   */
  async generateExplainabilityReport(
    queryId: string,
    queryText: string,
    queryIntent: string,
    topicPlanningResult?: TopicPlanningResult,
    raptorFeatures?: RaptorFeatures,
    candidates?: any[]
  ): Promise<ExplainabilityReport> {
    const trace = this.traces.get(queryId);
    const startTime = Date.now();

    try {
      // Topic analysis
      const topicAnalysis = await this.analyzeTopicHits(
        queryText,
        queryIntent,
        topicPlanningResult
      );

      // Feature explanations
      const featureExplanations = await this.explainFeatures(
        raptorFeatures,
        candidates
      );

      // Span coverage analysis
      const spanCoverage = await this.analyzeSpanCoverage(
        queryText,
        candidates,
        topicPlanningResult
      );

      // Decision trace
      const decisionTrace = this.buildDecisionTrace(trace);

      // Performance attribution
      const performanceAttribution = this.analyzePerformance(trace);

      // Quality metrics
      const qualityMetrics = this.calculateQualityMetrics(
        topicAnalysis,
        featureExplanations,
        spanCoverage
      );

      const report: ExplainabilityReport = {
        query_id: queryId,
        query_text: queryText,
        query_intent: queryIntent,
        timestamp: new Date().toISOString(),
        topic_analysis: topicAnalysis,
        feature_explanations: featureExplanations,
        span_coverage: spanCoverage,
        decision_trace: decisionTrace,
        performance_attribution: performanceAttribution,
        quality_metrics: qualityMetrics
      };

      this.reports.set(queryId, report);
      this.emit('report_generated', { query_id: queryId, processing_time: Date.now() - startTime });

      return report;

    } catch (error) {
      this.emit('report_error', { query_id: queryId, error: error instanceof Error ? error.message : 'Unknown error' });
      throw error;
    }
  }

  /**
   * Analyze topic hits and provide explanations
   */
  private async analyzeTopicHits(
    queryText: string,
    queryIntent: string,
    planningResult?: TopicPlanningResult
  ): Promise<TopicAnalysis> {
    if (!planningResult || !this.topicTree) {
      return {
        topics_matched: [],
        topics_considered: 0,
        topic_diversity_score: 0,
        coverage_gaps: [],
        similarity_distribution: [],
        entropy_score: 0
      };
    }

    const topicsMatched: TopicHitExplanation[] = [];

    for (const topicMatch of planningResult.topic_matches) {
      const topic = this.topicTree.getTopic(topicMatch.topic_id);
      if (!topic) continue;

      const hitReasons = this.analyzeTopicHitReasons(
        queryText,
        queryIntent,
        topicMatch,
        topic
      );

      const explanation: TopicHitExplanation = {
        topic_id: topicMatch.topic_id,
        topic_summary: topic.summary || 'No summary available',
        confidence_score: topicMatch.similarity_score,
        relevance_score: topicMatch.coverage_relevance,
        hit_reasons: hitReasons,
        contributing_cards: topic.card_ids,
        semantic_distance: 1 - topicMatch.similarity_score,
        keyword_overlap_score: topicMatch.keyword_overlap,
        business_relevance: topicMatch.boost_contribution
      };

      topicsMatched.push(explanation);
    }

    // Analyze coverage gaps
    const coverageGaps = this.identifyCoverageGaps(
      queryText,
      queryIntent,
      topicsMatched
    );

    // Calculate diversity score
    const diversityScore = this.calculateTopicDiversityScore(topicsMatched);

    // Get similarity distribution
    const similarities = topicsMatched.map(t => t.confidence_score);

    return {
      topics_matched: topicsMatched,
      topics_considered: planningResult.topic_matches.length,
      topic_diversity_score: diversityScore,
      coverage_gaps: coverageGaps,
      similarity_distribution: similarities,
      entropy_score: planningResult.reasoning.topic_entropy
    };
  }

  /**
   * Analyze why a topic was matched
   */
  private analyzeTopicHitReasons(
    queryText: string,
    queryIntent: string,
    topicMatch: TopicMatch,
    topic: any
  ): HitReason[] {
    const reasons: HitReason[] = [];

    // Semantic similarity reason
    if (topicMatch.similarity_score > 0.5) {
      reasons.push({
        type: 'semantic_similarity',
        explanation: `High semantic similarity (${(topicMatch.similarity_score * 100).toFixed(1)}%) between query and topic content`,
        confidence: topicMatch.similarity_score,
        evidence: [topic.summary || 'Topic summary unavailable'],
        weight_contribution: topicMatch.similarity_score * 0.4
      });
    }

    // Keyword overlap reason
    if (topicMatch.keyword_overlap > 0.3) {
      reasons.push({
        type: 'keyword_match',
        explanation: `Strong keyword overlap (${(topicMatch.keyword_overlap * 100).toFixed(1)}%) with topic terms`,
        confidence: topicMatch.keyword_overlap,
        evidence: this.extractSharedKeywords(queryText, topic.summary || ''),
        weight_contribution: topicMatch.keyword_overlap * 0.3
      });
    }

    // Business context reason
    if (topicMatch.boost_contribution > 0.1) {
      reasons.push({
        type: 'business_context',
        explanation: `High business relevance based on code importance and usage patterns`,
        confidence: Math.min(topicMatch.boost_contribution * 5, 1), // Scale boost to 0-1
        evidence: [`Business relevance score: ${topicMatch.boost_contribution.toFixed(3)}`],
        weight_contribution: topicMatch.boost_contribution
      });
    }

    return reasons;
  }

  /**
   * Extract shared keywords between query and topic
   */
  private extractSharedKeywords(queryText: string, topicSummary: string): string[] {
    const queryWords = queryText.toLowerCase().split(/\s+/).filter(w => w.length > 2);
    const summaryWords = topicSummary.toLowerCase().split(/\s+/).filter(w => w.length > 2);
    
    return queryWords.filter(word => summaryWords.includes(word));
  }

  /**
   * Identify coverage gaps in topic matching
   */
  private identifyCoverageGaps(
    queryText: string,
    queryIntent: string,
    topicsMatched: TopicHitExplanation[]
  ): CoverageGap[] {
    const gaps: CoverageGap[] = [];

    // Mock gap analysis - would implement proper NLP analysis
    if (topicsMatched.length === 0) {
      gaps.push({
        query_aspect: 'complete_query',
        missing_coverage: 'No relevant topics found for any part of the query',
        suggested_improvements: [
          'Expand topic clustering to include more diverse code patterns',
          'Lower similarity thresholds for topic matching',
          'Add manual topic annotations for edge cases'
        ],
        impact_severity: 'high'
      });
    } else if (topicsMatched.length === 1) {
      gaps.push({
        query_aspect: 'query_diversity',
        missing_coverage: 'Only one topic matched - query may cover multiple aspects',
        suggested_improvements: [
          'Improve topic granularity to capture multiple query aspects',
          'Implement query decomposition for multi-part queries'
        ],
        impact_severity: 'medium'
      });
    }

    return gaps;
  }

  /**
   * Calculate topic diversity score
   */
  private calculateTopicDiversityScore(topicsMatched: TopicHitExplanation[]): number {
    if (topicsMatched.length <= 1) return 0;

    // Simple diversity measure based on similarity score variance
    const similarities = topicsMatched.map(t => t.confidence_score);
    const mean = similarities.reduce((sum, sim) => sum + sim, 0) / similarities.length;
    const variance = similarities.reduce((sum, sim) => sum + Math.pow(sim - mean, 2), 0) / similarities.length;
    
    return Math.min(variance * 4, 1); // Scale to 0-1 range
  }

  /**
   * Explain RAPTOR feature contributions
   */
  private async explainFeatures(
    raptorFeatures?: RaptorFeatures,
    candidates?: any[]
  ): Promise<FeatureExplanation[]> {
    if (!raptorFeatures) return [];

    const explanations: FeatureExplanation[] = [];

    // Analyze each feature
    for (const [featureName, featureValue] of Object.entries(raptorFeatures)) {
      if (typeof featureValue !== 'number') continue;

      const explanation = this.explainSingleFeature(
        featureName,
        featureValue,
        candidates
      );

      explanations.push(explanation);
    }

    return explanations.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  }

  /**
   * Explain a single feature's contribution
   */
  private explainSingleFeature(
    featureName: string,
    featureValue: number,
    candidates?: any[]
  ): FeatureExplanation {
    // Mock feature explanation - would implement detailed analysis
    const baseExplanations: Record<string, string> = {
      raptor_max_sim: 'Maximum semantic similarity between query and candidate topics',
      topic_overlap: 'Degree of overlap between query topics and candidate content',
      depth_of_best: 'Hierarchical depth of the best matching topic',
      businessness: 'Business importance based on code usage and abstraction level',
      raptor_max_sim_x_topic_overlap: 'Combined semantic similarity and topic overlap strength',
      type_match_and_topic_match: 'Alignment between type information and semantic topics'
    };

    return {
      feature_name: featureName,
      feature_type: featureName.includes('raptor') ? 'raptor' : 
                   featureName.includes('x') ? 'cross' : 'raptor',
      value: featureValue,
      weight: Math.min(Math.abs(featureValue) * 0.4, 0.4), // Mock weight with cap
      contribution: featureValue * 0.2, // Mock contribution
      confidence: Math.min(Math.abs(featureValue) * 2, 1),
      explanation: baseExplanations[featureName] || `Feature measuring: ${featureName}`,
      supporting_evidence: [
        {
          type: 'semantic_similarity',
          description: `Feature value: ${featureValue.toFixed(3)}`,
          strength: Math.abs(featureValue),
          source_data: { feature_value: featureValue }
        }
      ],
      alternatives_considered: []
    };
  }

  /**
   * Analyze span coverage quality
   */
  private async analyzeSpanCoverage(
    queryText: string,
    candidates?: any[],
    planningResult?: TopicPlanningResult
  ): Promise<SpanCoverageAnalysis> {
    if (!candidates || candidates.length === 0) {
      return {
        total_spans_analyzed: 0,
        spans_covered_by_topics: 0,
        coverage_percentage: 0,
        uncovered_spans: [],
        coverage_quality: {
          precision: 0,
          recall: 0,
          f1_score: 0,
          coverage_distribution: {}
        },
        redundancy_analysis: {
          overlapping_topics: [],
          redundancy_score: 0,
          optimization_suggestions: []
        }
      };
    }

    // Mock span coverage analysis
    const totalSpans = candidates.length;
    const coveredSpans = Math.floor(totalSpans * 0.7); // Mock 70% coverage

    const precision = 0.8; // Mock precision
    const recall = coveredSpans / totalSpans;
    const f1Score = (2 * precision * recall) / (precision + recall);

    return {
      total_spans_analyzed: totalSpans,
      spans_covered_by_topics: coveredSpans,
      coverage_percentage: (coveredSpans / totalSpans) * 100,
      uncovered_spans: [], // Would identify actual uncovered spans
      coverage_quality: {
        precision,
        recall,
        f1_score: f1Score,
        coverage_distribution: {
          'TypeScript': 0.8,
          'JavaScript': 0.6,
          'JSON': 0.3
        }
      },
      redundancy_analysis: {
        overlapping_topics: [], // Would identify overlapping topics
        redundancy_score: 0.15, // Mock 15% redundancy
        optimization_suggestions: [
          'Consider consolidating topics with >80% overlap',
          'Refine topic boundaries to reduce redundancy'
        ]
      }
    };
  }

  /**
   * Build decision trace from query trace
   */
  private buildDecisionTrace(trace?: QueryTrace): DecisionTraceStep[] {
    if (!trace) return [];

    const steps: DecisionTraceStep[] = [];
    let stepNumber = 1;

    for (const stage of trace.stages) {
      steps.push({
        step_number: stepNumber++,
        component: this.mapStageToComponent(stage.stage_name),
        action: stage.stage_name,
        input_data: { size: stage.input_size },
        output_data: { size: stage.output_size },
        processing_time_ms: stage.end_time_ms - stage.start_time_ms,
        confidence_score: 0.8, // Mock confidence
        decision_factors: [
          {
            factor_name: 'processing_time',
            factor_value: stage.end_time_ms - stage.start_time_ms,
            influence_weight: 0.3,
            rationale: 'Processing efficiency affects user experience'
          }
        ]
      });
    }

    return steps;
  }

  private mapStageToComponent(stageName: string): 'stage_a' | 'stage_c' | 'nl_bridge' | 'topic_tree' {
    if (stageName.includes('stage_a') || stageName.includes('planner')) return 'stage_a';
    if (stageName.includes('stage_c') || stageName.includes('features')) return 'stage_c';
    if (stageName.includes('bridge') || stageName.includes('nl')) return 'nl_bridge';
    return 'topic_tree';
  }

  /**
   * Analyze performance attribution
   */
  private analyzePerformance(trace?: QueryTrace): PerformanceAttribution {
    if (!trace || trace.stages.length === 0) {
      return {
        total_processing_time_ms: 0,
        component_breakdown: [],
        bottlenecks: [],
        optimization_opportunities: []
      };
    }

    const totalTime = trace.stages.reduce((sum, stage) => 
      sum + (stage.end_time_ms - stage.start_time_ms), 0
    );

    const componentBreakdown: ComponentTiming[] = trace.stages.map(stage => ({
      component: stage.stage_name,
      processing_time_ms: stage.end_time_ms - stage.start_time_ms,
      percentage_of_total: ((stage.end_time_ms - stage.start_time_ms) / totalTime) * 100,
      operations_count: stage.operations.length,
      avg_time_per_operation: stage.operations.length > 0 ? 
        (stage.end_time_ms - stage.start_time_ms) / stage.operations.length : 0
    }));

    // Identify bottlenecks (stages taking >30% of total time)
    const bottlenecks: PerformanceBottleneck[] = componentBreakdown
      .filter(component => component.percentage_of_total > 30)
      .map(component => ({
        component: component.component,
        issue: `High processing time (${component.processing_time_ms}ms)`,
        impact_ms: component.processing_time_ms,
        severity: component.percentage_of_total > 50 ? 'high' as const : 'medium' as const,
        suggested_fix: `Optimize ${component.component} processing`
      }));

    return {
      total_processing_time_ms: totalTime,
      component_breakdown: componentBreakdown,
      bottlenecks,
      optimization_opportunities: [
        {
          opportunity: 'Cache frequently accessed topic embeddings',
          estimated_improvement_ms: totalTime * 0.2,
          implementation_effort: 'medium',
          risk_level: 'low'
        }
      ]
    };
  }

  /**
   * Calculate quality metrics for the explanation
   */
  private calculateQualityMetrics(
    topicAnalysis: TopicAnalysis,
    featureExplanations: FeatureExplanation[],
    spanCoverage: SpanCoverageAnalysis
  ): QualityMetrics {
    // Mock quality calculation - would implement proper metrics
    const completeness = Math.min(
      (topicAnalysis.topics_matched.length / 5) + 
      (featureExplanations.length / 10) +
      (spanCoverage.coverage_percentage / 100),
      1.0
    );

    return {
      explanation_completeness: completeness,
      confidence_calibration: 0.75, // Mock calibration score
      decision_consistency: 0.80, // Mock consistency score
      user_comprehensibility: 0.70, // Mock comprehensibility
      actionability_score: 0.65 // Mock actionability
    };
  }

  /**
   * Get explainability report for a query
   */
  getReport(queryId: string): ExplainabilityReport | undefined {
    return this.reports.get(queryId);
  }

  /**
   * Get query trace
   */
  getTrace(queryId: string): QueryTrace | undefined {
    return this.traces.get(queryId);
  }

  /**
   * Clean up old traces and reports
   */
  private cleanupOldTraces(): void {
    const cutoffTime = Date.now() - (24 * 60 * 60 * 1000); // 24 hours ago
    
    for (const [queryId, trace] of this.traces) {
      const lastStageTime = trace.stages.length > 0 ? 
        Math.max(...trace.stages.map(s => s.end_time_ms || s.start_time_ms)) : 0;
      
      if (lastStageTime < cutoffTime) {
        this.traces.delete(queryId);
        this.reports.delete(queryId);
      }
    }
    
    this.emit('cleanup_completed', { 
      remaining_traces: this.traces.size,
      remaining_reports: this.reports.size 
    });
  }

  /**
   * Export telemetry data for analysis
   */
  exportTelemetryData(): {
    traces: QueryTrace[];
    reports: ExplainabilityReport[];
    summary: {
      total_queries: number;
      avg_processing_time_ms: number;
      avg_topics_per_query: number;
      avg_features_per_query: number;
    };
  } {
    const traces = Array.from(this.traces.values());
    const reports = Array.from(this.reports.values());

    const avgProcessingTime = traces.reduce((sum, trace) => 
      sum + trace.total_time_ms, 0) / Math.max(traces.length, 1);
    
    const avgTopicsPerQuery = reports.reduce((sum, report) => 
      sum + report.topic_analysis.topics_matched.length, 0) / Math.max(reports.length, 1);
    
    const avgFeaturesPerQuery = reports.reduce((sum, report) => 
      sum + report.feature_explanations.length, 0) / Math.max(reports.length, 1);

    return {
      traces,
      reports,
      summary: {
        total_queries: traces.length,
        avg_processing_time_ms: avgProcessingTime,
        avg_topics_per_query: avgTopicsPerQuery,
        avg_features_per_query: avgFeaturesPerQuery
      }
    };
  }

  /**
   * Cleanup resources
   */
  cleanup(): void {
    this.traces.clear();
    this.reports.clear();
    this.removeAllListeners();
  }
}