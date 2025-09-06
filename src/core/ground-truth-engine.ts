/**
 * Ground-Truth Engine: Continuous Learning System
 * 
 * Industrial strength system that maintains trustworthy ground truth through:
 * - Mining queries where gap is largest: gap = (Recall@50 − Recall@10)_SLA
 * - Sampling by {intent×lang×topic} with exploration bonus
 * - Routing to adjudication (two judges + tie-breaker)
 * - Nightly promotion of new labels to pooled qrels
 * - Pool health KPIs: inter-rater κ, pool growth/week, slice coverage
 * - Self-confirmation guards via query perturbations
 */

import { EventEmitter } from 'events';

// Core Types
export interface GroundTruthQuery {
  id: string;
  text: string;
  intent: QueryIntent;
  language: string;
  topic: string;
  gap_score: number; // (Recall@50 - Recall@10)_SLA
  entropy_score: number;
  exploration_bonus: number;
  timestamp: Date;
}

export interface QueryIntent {
  primary: 'api_usage' | 'definition' | 'implementation' | 'debug' | 'concept';
  confidence: number;
  secondary?: string[];
}

export interface AdjudicationTask {
  id: string;
  query: GroundTruthQuery;
  candidates: SearchResult[];
  status: 'pending' | 'in_progress' | 'completed' | 'conflicted';
  judges: AdjudicationJudge[];
  tie_breaker?: AdjudicationJudge;
  created_at: Date;
  completed_at?: Date;
}

export interface AdjudicationJudge {
  id: string;
  annotations: ResultAnnotation[];
  confidence: number;
  time_spent_ms: number;
  submitted_at: Date;
}

export interface ResultAnnotation {
  result_id: string;
  relevance: 0 | 1 | 2 | 3; // 0: irrelevant, 1: marginal, 2: relevant, 3: perfect
  reasoning: string;
  confidence: number;
}

export interface PoolHealthMetrics {
  inter_rater_kappa: number;
  pool_growth_per_week: number;
  slice_coverage: SliceCoverage;
  self_confirmation_rate: number;
  query_diversity_entropy: number;
}

export interface SliceCoverage {
  by_intent: Record<string, number>;
  by_language: Record<string, number>;
  by_topic: Record<string, number>;
  cross_coverage: number; // Coverage of intent×language×topic combinations
}

export interface SearchResult {
  id: string;
  path: string;
  score: number;
  snippet: string;
  metadata: {
    language: string;
    file_type: string;
    line_number?: number;
  };
}

export interface QueryPerturbation {
  original_query: string;
  perturbation_type: 'alias_flip' | 'version_bump' | 'synonym_replace' | 'context_add';
  perturbed_query: string;
  expected_overlap_threshold: number;
}

// Configuration
export interface GroundTruthEngineConfig {
  mining: {
    gap_threshold: number; // Minimum gap score to consider
    entropy_threshold: number; // Minimum entropy for high-uncertainty queries
    exploration_factor: number; // Bonus weight for underexplored slices
    batch_size: number; // Queries per mining batch
    sampling_strategy: 'proportional' | 'uniform' | 'exploration_weighted';
  };
  adjudication: {
    judges_per_task: number; // Default 2
    tie_breaker_threshold: number; // Agreement threshold for tie-breaker
    max_time_per_query_minutes: number;
    quality_checks: {
      min_confidence: number;
      max_time_deviation: number; // Flag unusually fast/slow annotations
      consistency_checks: boolean;
    };
  };
  pool_management: {
    promotion_schedule: 'nightly' | 'weekly';
    min_agreement_for_promotion: number;
    max_pool_size: number;
    retention_policy_days: number;
  };
  perturbation: {
    perturbation_rate: number; // Fraction of queries to perturb
    max_perturbations_per_query: number;
    overlap_tolerance: number; // Max acceptable result overlap
  };
}

export class GroundTruthEngine extends EventEmitter {
  private config: GroundTruthEngineConfig;
  private queryGapAnalyzer: QueryGapAnalyzer;
  private intentClassifier: IntentClassifier;
  private explorationSampler: ExplorationSampler;
  private adjudicationManager: AdjudicationManager;
  private poolManager: PoolManager;
  private perturbationGenerator: PerturbationGenerator;
  private healthMonitor: PoolHealthMonitor;

  constructor(config: GroundTruthEngineConfig) {
    super();
    this.config = config;
    this.queryGapAnalyzer = new QueryGapAnalyzer();
    this.intentClassifier = new IntentClassifier();
    this.explorationSampler = new ExplorationSampler(config.mining);
    this.adjudicationManager = new AdjudicationManager(config.adjudication);
    this.poolManager = new PoolManager(config.pool_management);
    this.perturbationGenerator = new PerturbationGenerator(config.perturbation);
    this.healthMonitor = new PoolHealthMonitor();
  }

  /**
   * Mine queries with largest gaps for ground truth expansion
   */
  async mineHighGapQueries(queryLogs: QueryLog[]): Promise<GroundTruthQuery[]> {
    // Analyze recall gaps for each query
    const gapAnalysis = await this.queryGapAnalyzer.analyzeGaps(queryLogs);
    
    // Classify intents and extract features
    const enrichedQueries = await Promise.all(
      gapAnalysis.map(async (query) => {
        const intent = await this.intentClassifier.classify(query.query);
        const features = this.extractQueryFeatures(query, intent);
        
        return {
          ...query,
          id: `gt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          text: query.query,
          intent,
          language: features.language,
          topic: features.topic,
          exploration_bonus: this.explorationSampler.calculateExplorationBonus(features)
        };
      })
    );

    // Sample by exploration strategy
    return this.explorationSampler.sampleQueries(enrichedQueries, this.config.mining.batch_size);
  }

  /**
   * Route queries to adjudication with perturbation guards
   */
  async routeToAdjudication(queries: GroundTruthQuery[]): Promise<AdjudicationTask[]> {
    const tasks: AdjudicationTask[] = [];

    for (const query of queries) {
      // Generate perturbations to guard against self-confirmation
      const perturbations = await this.perturbationGenerator.generatePerturbations(query);
      
      // Create main adjudication task
      const candidates = await this.executeSearchForAnnotation(query.text);
      const mainTask = await this.adjudicationManager.createTask(query, candidates);
      tasks.push(mainTask);

      // Create perturbation tasks for validation
      for (const perturbation of perturbations) {
        const perturbedCandidates = await this.executeSearchForAnnotation(perturbation.perturbed_query);
        const perturbedQuery = { ...query, text: perturbation.perturbed_query };
        const perturbTask = await this.adjudicationManager.createTask(perturbedQuery, perturbedCandidates);
        perturbTask.id = `${mainTask.id}_perturbation_${perturbation.perturbation_type}`;
        tasks.push(perturbTask);
      }
    }

    return tasks;
  }

  /**
   * Process completed adjudication tasks and promote to pool
   */
  async processCompletedTasks(): Promise<PoolPromotionResult> {
    const completedTasks = await this.adjudicationManager.getCompletedTasks();
    const promotionCandidates: PoolCandidate[] = [];

    for (const task of completedTasks) {
      // Check agreement between judges
      const agreement = this.calculateJudgeAgreement(task);
      
      if (agreement.kappa >= this.config.pool_management.min_agreement_for_promotion) {
        // Create consensus annotations
        const consensusAnnotations = this.createConsensusAnnotations(task);
        
        promotionCandidates.push({
          query: task.query,
          annotations: consensusAnnotations,
          agreement_score: agreement.kappa,
          judge_count: task.judges.length,
          validation_passed: await this.validatePerturbationConsistency(task)
        });
      }
    }

    // Promote to pooled qrels
    return await this.poolManager.promoteToPool(promotionCandidates);
  }

  /**
   * Generate comprehensive pool health metrics
   */
  async generateHealthMetrics(): Promise<PoolHealthMetrics> {
    const currentPool = await this.poolManager.getCurrentPool();
    const recentTasks = await this.adjudicationManager.getRecentTasks(7); // Last 7 days
    
    return {
      inter_rater_kappa: this.healthMonitor.calculateInterRaterKappa(recentTasks),
      pool_growth_per_week: this.healthMonitor.calculatePoolGrowthRate(currentPool),
      slice_coverage: this.healthMonitor.calculateSliceCoverage(currentPool),
      self_confirmation_rate: this.healthMonitor.calculateSelfConfirmationRate(recentTasks),
      query_diversity_entropy: this.healthMonitor.calculateQueryDiversityEntropy(currentPool)
    };
  }

  /**
   * Execute nightly pool promotion and health checks
   */
  async executeNightlyPromotion(): Promise<void> {
    try {
      this.emit('promotion_started', { timestamp: new Date() });

      // Process completed adjudication tasks
      const promotionResult = await this.processCompletedTasks();
      
      // Generate health metrics
      const healthMetrics = await this.generateHealthMetrics();
      
      // Check health thresholds
      const healthIssues = this.validatePoolHealth(healthMetrics);
      
      if (healthIssues.length > 0) {
        this.emit('health_warnings', { issues: healthIssues, metrics: healthMetrics });
      }

      // Clean up old tasks and pool entries
      await this.performHousekeeping();

      this.emit('promotion_completed', {
        promoted_count: promotionResult.promoted_count,
        health_metrics: healthMetrics,
        timestamp: new Date()
      });

    } catch (error) {
      this.emit('promotion_failed', { error, timestamp: new Date() });
      throw error;
    }
  }

  // Private helper methods

  private extractQueryFeatures(query: any, intent: QueryIntent): QueryFeatures {
    // Implement feature extraction logic
    return {
      language: this.detectLanguage(query.text),
      topic: this.extractTopic(query.text, intent),
      complexity: this.assessQueryComplexity(query.text)
    };
  }

  private async executeSearchForAnnotation(query: string): Promise<SearchResult[]> {
    // Execute search with current system to get candidates for annotation
    // This should use the same search pipeline that's being evaluated
    throw new Error('Must be implemented with actual search system integration');
  }

  private calculateJudgeAgreement(task: AdjudicationTask): AgreementMetrics {
    // Implement inter-rater agreement calculation (Cohen's kappa, etc.)
    const agreements = task.judges.map(judge => judge.annotations);
    
    // Simplified kappa calculation - replace with robust implementation
    const kappa = this.calculateKappa(agreements);
    
    return {
      kappa,
      raw_agreement: this.calculateRawAgreement(agreements),
      disagreement_patterns: this.analyzeDisagreementPatterns(agreements)
    };
  }

  private createConsensusAnnotations(task: AdjudicationTask): ResultAnnotation[] {
    // Create consensus from multiple judge annotations
    const resultMap = new Map<string, ResultAnnotation[]>();
    
    // Group annotations by result
    for (const judge of task.judges) {
      for (const annotation of judge.annotations) {
        if (!resultMap.has(annotation.result_id)) {
          resultMap.set(annotation.result_id, []);
        }
        resultMap.get(annotation.result_id)!.push(annotation);
      }
    }

    // Create consensus for each result
    const consensus: ResultAnnotation[] = [];
    for (const [resultId, annotations] of resultMap.entries()) {
      consensus.push(this.createConsensusForResult(resultId, annotations));
    }

    return consensus;
  }

  private async validatePerturbationConsistency(task: AdjudicationTask): Promise<boolean> {
    // Check if perturbation results are consistent with expectations
    // This prevents self-confirmation bias
    return true; // Placeholder implementation
  }

  private validatePoolHealth(metrics: PoolHealthMetrics): HealthIssue[] {
    const issues: HealthIssue[] = [];
    
    if (metrics.inter_rater_kappa < 0.4) {
      issues.push({
        type: 'low_agreement',
        severity: 'high',
        message: `Inter-rater agreement too low: ${metrics.inter_rater_kappa}`,
        threshold: 0.4
      });
    }

    if (metrics.self_confirmation_rate > 0.3) {
      issues.push({
        type: 'self_confirmation',
        severity: 'medium',
        message: `High self-confirmation rate: ${metrics.self_confirmation_rate}`,
        threshold: 0.3
      });
    }

    return issues;
  }

  private async performHousekeeping(): Promise<void> {
    // Clean up old tasks and pool entries based on retention policy
    await this.adjudicationManager.cleanupOldTasks(this.config.pool_management.retention_policy_days);
    await this.poolManager.cleanupOldEntries(this.config.pool_management.retention_policy_days);
  }

  // Placeholder method implementations - should be properly implemented

  private detectLanguage(query: string): string {
    // Simple language detection based on keywords/patterns
    if (/\b(class|function|import|export)\b/.test(query)) return 'typescript';
    if (/\b(def|class|import|from)\b/.test(query)) return 'python';
    if (/\b(fn|struct|impl|use)\b/.test(query)) return 'rust';
    return 'generic';
  }

  private extractTopic(query: string, intent: QueryIntent): string {
    // Extract topic based on query content and intent
    if (intent.primary === 'api_usage') return 'api';
    if (intent.primary === 'debug') return 'debugging';
    return 'general';
  }

  private assessQueryComplexity(query: string): number {
    // Assess query complexity for sampling
    return query.split(' ').length / 10; // Simplified metric
  }

  private calculateKappa(agreements: ResultAnnotation[][]): number {
    // Simplified Cohen's kappa calculation
    return 0.5; // Placeholder
  }

  private calculateRawAgreement(agreements: ResultAnnotation[][]): number {
    // Calculate raw agreement percentage
    return 0.7; // Placeholder
  }

  private analyzeDisagreementPatterns(agreements: ResultAnnotation[][]): DisagreementPattern[] {
    // Analyze patterns in disagreements to improve annotation process
    return []; // Placeholder
  }

  private createConsensusForResult(resultId: string, annotations: ResultAnnotation[]): ResultAnnotation {
    // Create consensus annotation from multiple judges
    const avgRelevance = annotations.reduce((sum, ann) => sum + ann.relevance, 0) / annotations.length;
    const avgConfidence = annotations.reduce((sum, ann) => sum + ann.confidence, 0) / annotations.length;
    
    return {
      result_id: resultId,
      relevance: Math.round(avgRelevance) as 0 | 1 | 2 | 3,
      reasoning: `Consensus from ${annotations.length} judges`,
      confidence: avgConfidence
    };
  }
}

// Supporting Classes (simplified interfaces - full implementation needed)

class QueryGapAnalyzer {
  async analyzeGaps(queryLogs: QueryLog[]): Promise<GapAnalysisResult[]> {
    // Analyze (Recall@50 − Recall@10)_SLA gaps
    throw new Error('Implementation required');
  }
}

class IntentClassifier {
  async classify(query: string): Promise<QueryIntent> {
    // Classify query intent using ML model or rules
    throw new Error('Implementation required');
  }
}

class ExplorationSampler {
  constructor(private config: any) {}

  calculateExplorationBonus(features: QueryFeatures): number {
    // Calculate exploration bonus for underexplored intent×language×topic combinations
    return 0.1; // Placeholder
  }

  sampleQueries(queries: GroundTruthQuery[], batchSize: number): GroundTruthQuery[] {
    // Sample queries using exploration strategy
    return queries.slice(0, batchSize); // Placeholder
  }
}

class AdjudicationManager {
  constructor(private config: any) {}

  async createTask(query: GroundTruthQuery, candidates: SearchResult[]): Promise<AdjudicationTask> {
    // Create adjudication task for human judges
    throw new Error('Implementation required');
  }

  async getCompletedTasks(): Promise<AdjudicationTask[]> {
    // Get completed adjudication tasks
    throw new Error('Implementation required');
  }

  async getRecentTasks(days: number): Promise<AdjudicationTask[]> {
    // Get recent tasks for health monitoring
    throw new Error('Implementation required');
  }

  async cleanupOldTasks(retentionDays: number): Promise<void> {
    // Clean up old tasks
    throw new Error('Implementation required');
  }
}

class PoolManager {
  constructor(private config: any) {}

  async promoteToPool(candidates: PoolCandidate[]): Promise<PoolPromotionResult> {
    // Promote candidates to the pooled qrels
    throw new Error('Implementation required');
  }

  async getCurrentPool(): Promise<PoolEntry[]> {
    // Get current pool state
    throw new Error('Implementation required');
  }

  async cleanupOldEntries(retentionDays: number): Promise<void> {
    // Clean up old pool entries
    throw new Error('Implementation required');
  }
}

class PerturbationGenerator {
  constructor(private config: any) {}

  async generatePerturbations(query: GroundTruthQuery): Promise<QueryPerturbation[]> {
    // Generate query perturbations to guard against self-confirmation
    return []; // Placeholder
  }
}

class PoolHealthMonitor {
  calculateInterRaterKappa(tasks: AdjudicationTask[]): number {
    // Calculate inter-rater reliability
    return 0.6; // Placeholder
  }

  calculatePoolGrowthRate(pool: PoolEntry[]): number {
    // Calculate pool growth per week
    return 100; // Placeholder
  }

  calculateSliceCoverage(pool: PoolEntry[]): SliceCoverage {
    // Calculate coverage across intent×language×topic slices
    return {
      by_intent: {},
      by_language: {},
      by_topic: {},
      cross_coverage: 0.5
    };
  }

  calculateSelfConfirmationRate(tasks: AdjudicationTask[]): number {
    // Calculate rate of self-confirmation bias
    return 0.1; // Placeholder
  }

  calculateQueryDiversityEntropy(pool: PoolEntry[]): number {
    // Calculate entropy of query diversity
    return 2.5; // Placeholder
  }
}

// Additional Types

interface QueryLog {
  query: string;
  results: SearchResult[];
  recall_at_10: number;
  recall_at_50: number;
  timestamp: Date;
}

interface GapAnalysisResult {
  query: string;
  gap_score: number;
  entropy_score: number;
  timestamp: Date;
}

interface QueryFeatures {
  language: string;
  topic: string;
  complexity: number;
}

interface AgreementMetrics {
  kappa: number;
  raw_agreement: number;
  disagreement_patterns: DisagreementPattern[];
}

interface DisagreementPattern {
  type: string;
  frequency: number;
  judges_involved: string[];
}

interface PoolCandidate {
  query: GroundTruthQuery;
  annotations: ResultAnnotation[];
  agreement_score: number;
  judge_count: number;
  validation_passed: boolean;
}

interface PoolPromotionResult {
  promoted_count: number;
  rejected_count: number;
  quality_scores: number[];
}

interface PoolEntry {
  query: GroundTruthQuery;
  annotations: ResultAnnotation[];
  created_at: Date;
  last_validated: Date;
}

interface HealthIssue {
  type: string;
  severity: 'low' | 'medium' | 'high';
  message: string;
  threshold: number;
}

// Default Configuration
export const DEFAULT_GROUND_TRUTH_CONFIG: GroundTruthEngineConfig = {
  mining: {
    gap_threshold: 0.15, // 15% gap between recall@50 and recall@10
    entropy_threshold: 1.5, // High uncertainty threshold
    exploration_factor: 0.3, // 30% bonus for underexplored slices
    batch_size: 50, // 50 queries per mining batch
    sampling_strategy: 'exploration_weighted'
  },
  adjudication: {
    judges_per_task: 2,
    tie_breaker_threshold: 0.3, // Agreement below 30% triggers tie-breaker
    max_time_per_query_minutes: 15,
    quality_checks: {
      min_confidence: 0.7,
      max_time_deviation: 2.0, // 2x standard deviation
      consistency_checks: true
    }
  },
  pool_management: {
    promotion_schedule: 'nightly',
    min_agreement_for_promotion: 0.6, // 60% agreement minimum
    max_pool_size: 100000, // 100K queries max
    retention_policy_days: 365 // 1 year retention
  },
  perturbation: {
    perturbation_rate: 0.2, // 20% of queries get perturbations
    max_perturbations_per_query: 3,
    overlap_tolerance: 0.4 // 40% max overlap tolerance
  }
};