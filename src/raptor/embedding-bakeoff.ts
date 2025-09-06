/**
 * Embedding Model Bake-off Harness
 * 
 * Evaluates multiple embedding models on code search tasks.
 * Tests 3-5 public code-friendly models against quality and performance metrics.
 */

import { EmbeddingRouter, EmbeddingModel, EmbeddingRequest } from './embedding-router.js';

export interface BakeoffCandidate {
  id: string;
  name: string;
  provider: string;
  model_path?: string;
  api_endpoint?: string;
  dim: number;
  max_tokens: number;
  supported_languages: string[];
}

export interface BakeoffMetrics {
  model_id: string;
  repo: string;
  
  // Quality metrics
  ndcg_at_10: number;
  recall_at_50_sla: number;  // Recall@50 with latency ≤ 150ms
  ece: number;  // Expected Calibration Error
  
  // Performance metrics
  p95_latency_ms: number;
  p99_latency_ms: number;
  qps_at_150ms: number;
  
  // Query type breakdown
  ur_broad_ndcg: number;
  ur_narrow_ndcg: number;
  cp_regex_success: number;
  
  // Resource usage
  memory_usage_mb: number;
  cpu_utilization: number;
}

export interface BakeoffResult {
  candidate: BakeoffCandidate;
  metrics: BakeoffMetrics[];  // One per test repo
  pareto_score: number;  // Quality vs Performance tradeoff
  recommendation: 'select' | 'reject' | 'investigate';
  notes: string[];
}

export interface BakeoffConfig {
  test_repos: string[];  // Paths to test repositories
  query_sets: {
    ur_broad: string[];
    ur_narrow: string[];
    cp_regex: string[];
  };
  
  // Quality thresholds
  min_ndcg_at_10: number;
  min_recall_at_50: number;
  max_ece: number;
  
  // Performance thresholds
  max_p95_ms: number;
  min_qps_at_150ms: number;
  
  // Evaluation settings
  warmup_queries: number;
  measurement_queries: number;
  timeout_ms: number;
}

/**
 * Embedding model evaluation harness
 */
export class EmbeddingBakeoff {
  private candidates: Map<string, BakeoffCandidate> = new Map();
  private config: BakeoffConfig;

  constructor(config?: Partial<BakeoffConfig>) {
    this.config = {
      test_repos: ['repo_a', 'repo_b'],  // Default test repositories
      query_sets: {
        ur_broad: [
          'authentication service',
          'database connection pool',
          'error handling middleware',
          'api rate limiting',
          'user session management'
        ],
        ur_narrow: [
          'JWT token validation',
          'PostgreSQL transaction retry',
          'HTTP 429 rate limit response',
          'Redis cache invalidation',
          'OAuth2 refresh token flow'
        ],
        cp_regex: [
          'function.*authenticate.*user',
          'class.*Database.*Connection',
          'interface.*Http.*Request',
          'type.*User.*Session',
          'const.*API.*KEY'
        ]
      },
      
      // Default thresholds from TODO.md Phase 4 requirements
      min_ndcg_at_10: 0.65,  // Baseline requirement
      min_recall_at_50: 0.80,
      max_ece: 0.05,
      max_p95_ms: 150,  // SLA requirement
      min_qps_at_150ms: 10,
      
      warmup_queries: 20,
      measurement_queries: 100,
      timeout_ms: 300000,  // 5 minutes total timeout
      
      ...config
    };

    this.initializeCandidates();
  }

  private initializeCandidates(): void {
    // Candidate 1: CodeBERT family
    this.candidates.set('codebert_base', {
      id: 'codebert_base',
      name: 'CodeBERT-Base',
      provider: 'microsoft',
      dim: 768,
      max_tokens: 512,
      supported_languages: ['py', 'js', 'go', 'java', 'php', 'cpp', 'rs']
    });

    // Candidate 2: CodeT5 family  
    this.candidates.set('codet5_base', {
      id: 'codet5_base',
      name: 'CodeT5-Base',
      provider: 'salesforce', 
      dim: 768,
      max_tokens: 512,
      supported_languages: ['py', 'js', 'go', 'java', 'php', 'cpp', 'rs']
    });

    // Candidate 3: UnixCoder
    this.candidates.set('unixcoder', {
      id: 'unixcoder',
      name: 'UniXcoder-Base',
      provider: 'microsoft',
      dim: 768,
      max_tokens: 512, 
      supported_languages: ['py', 'js', 'go', 'java', 'php', 'cpp', 'rs', 'ruby']
    });

    // Candidate 4: GraphCodeBERT
    this.candidates.set('graphcodebert', {
      id: 'graphcodebert',
      name: 'GraphCodeBERT',
      provider: 'microsoft',
      dim: 768,
      max_tokens: 512,
      supported_languages: ['py', 'js', 'go', 'java', 'php', 'cpp']
    });

    // Candidate 5: Faster alternative - smaller model
    this.candidates.set('codebert_small', {
      id: 'codebert_small', 
      name: 'CodeBERT-Small',
      provider: 'microsoft',
      dim: 384,
      max_tokens: 256,
      supported_languages: ['py', 'js', 'go', 'java', 'php', 'cpp', 'rs']
    });
  }

  /**
   * Run comprehensive bake-off evaluation
   */
  async runBakeoff(): Promise<BakeoffResult[]> {
    console.log(`🏁 Starting embedding model bake-off with ${this.candidates.size} candidates`);
    console.log(`📊 Test repos: ${this.config.test_repos.join(', ')}`);
    
    const results: BakeoffResult[] = [];
    
    for (const candidate of this.candidates.values()) {
      console.log(`\n🧪 Evaluating ${candidate.name}...`);
      
      try {
        const result = await this.evaluateCandidate(candidate);
        results.push(result);
        
        console.log(`✅ ${candidate.name} evaluation complete`);
        console.log(`   Avg nDCG@10: ${this.averageMetric(result.metrics, 'ndcg_at_10').toFixed(3)}`);
        console.log(`   Avg P95: ${this.averageMetric(result.metrics, 'p95_latency_ms').toFixed(1)}ms`);
        
      } catch (error) {
        console.error(`❌ ${candidate.name} evaluation failed:`, error);
        
        // Create failed result
        results.push({
          candidate,
          metrics: [],
          pareto_score: 0,
          recommendation: 'reject',
          notes: [`Evaluation failed: ${error instanceof Error ? error.message : 'Unknown error'}`]
        });
      }
    }

    // Compute Pareto rankings
    this.computeParetoScores(results);
    
    // Sort by recommendation and Pareto score
    results.sort((a, b) => {
      if (a.recommendation !== b.recommendation) {
        const order = { 'select': 0, 'investigate': 1, 'reject': 2 };
        return order[a.recommendation] - order[b.recommendation];
      }
      return b.pareto_score - a.pareto_score;
    });

    this.printSummary(results);
    return results;
  }

  private async evaluateCandidate(candidate: BakeoffCandidate): Promise<BakeoffResult> {
    const metrics: BakeoffMetrics[] = [];
    
    for (const repo of this.config.test_repos) {
      console.log(`  📁 Testing on ${repo}...`);
      
      // Initialize embedding router for this candidate
      const router = await this.createRouterForCandidate(candidate);
      
      // Run evaluation on this repo
      const repoMetrics = await this.evaluateOnRepo(candidate, router, repo);
      metrics.push(repoMetrics);
    }

    // Compute overall recommendation
    const avgNdcg = this.averageMetric(metrics, 'ndcg_at_10');
    const avgP95 = this.averageMetric(metrics, 'p95_latency_ms');
    const avgRecall = this.averageMetric(metrics, 'recall_at_50_sla');
    const avgEce = this.averageMetric(metrics, 'ece');
    
    const qualityPass = avgNdcg >= this.config.min_ndcg_at_10 && 
                       avgRecall >= this.config.min_recall_at_50 &&
                       avgEce <= this.config.max_ece;
                       
    const performancePass = avgP95 <= this.config.max_p95_ms;
    
    let recommendation: 'select' | 'reject' | 'investigate';
    const notes: string[] = [];
    
    if (qualityPass && performancePass) {
      recommendation = 'select';
      notes.push('Meets all quality and performance thresholds');
    } else if (qualityPass || performancePass) {
      recommendation = 'investigate';
      if (!qualityPass) {
        notes.push(`Quality below threshold: nDCG@10=${avgNdcg.toFixed(3)} (min=${this.config.min_ndcg_at_10})`);
      }
      if (!performancePass) {
        notes.push(`Performance below threshold: P95=${avgP95.toFixed(1)}ms (max=${this.config.max_p95_ms}ms)`);
      }
    } else {
      recommendation = 'reject';
      notes.push('Fails both quality and performance thresholds');
    }

    return {
      candidate,
      metrics,
      pareto_score: 0,  // Will be computed later
      recommendation,
      notes
    };
  }

  private async createRouterForCandidate(candidate: BakeoffCandidate): Promise<EmbeddingRouter> {
    // Create embedding router configured for this candidate
    const router = new EmbeddingRouter({
      enabled: true,  // Enable quantization for fair comparison
      method: 'int8',
      calibration_samples: 100
    });

    // TODO: In real implementation, would configure router to use specific model
    // For now, return mock router
    return router;
  }

  private async evaluateOnRepo(
    candidate: BakeoffCandidate, 
    router: EmbeddingRouter, 
    repo: string
  ): Promise<BakeoffMetrics> {
    
    // Warmup phase
    console.log(`    🔥 Warmup phase (${this.config.warmup_queries} queries)...`);
    await this.runQueries(router, this.config.query_sets.ur_broad.slice(0, this.config.warmup_queries));
    
    // Measurement phase
    console.log(`    📏 Measurement phase (${this.config.measurement_queries} queries)...`);
    
    const urBroadResults = await this.runQueries(router, this.config.query_sets.ur_broad);
    const urNarrowResults = await this.runQueries(router, this.config.query_sets.ur_narrow);
    const cpRegexResults = await this.runQueries(router, this.config.query_sets.cp_regex);
    
    // Compute quality metrics (mock values for now)
    const ndcg_at_10 = this.computeNdcg(urBroadResults, urNarrowResults);
    const recall_at_50_sla = this.computeRecallAtSla(urBroadResults, urNarrowResults);
    const ece = this.computeEce(urBroadResults, urNarrowResults);
    
    // Compute performance metrics
    const allResults = [...urBroadResults, ...urNarrowResults, ...cpRegexResults];
    const latencies = allResults.map(r => r.latency_ms);
    
    const p95_latency_ms = this.percentile(latencies, 0.95);
    const p99_latency_ms = this.percentile(latencies, 0.99);
    const qps_at_150ms = this.computeQpsAtSla(allResults, 150);
    
    return {
      model_id: candidate.id,
      repo,
      ndcg_at_10,
      recall_at_50_sla,
      ece,
      p95_latency_ms,
      p99_latency_ms,
      qps_at_150ms,
      ur_broad_ndcg: this.computeNdcg(urBroadResults, []),
      ur_narrow_ndcg: this.computeNdcg(urNarrowResults, []), 
      cp_regex_success: this.computeSuccessRate(cpRegexResults),
      memory_usage_mb: Math.random() * 512 + 256,  // Mock
      cpu_utilization: Math.random() * 0.8 + 0.2   // Mock
    };
  }

  private async runQueries(router: EmbeddingRouter, queries: string[]): Promise<QueryResult[]> {
    const results: QueryResult[] = [];
    
    for (const query of queries) {
      const startTime = Date.now();
      
      try {
        // Mock query execution - would integrate with actual search pipeline
        const request: EmbeddingRequest = {
          text: query,
          language: 'typescript',  // Mock
          context: 'query'
        };
        
        await router.embed(request);
        
        const latency = Date.now() - startTime;
        
        // Mock search results
        results.push({
          query,
          latency_ms: latency,
          results: this.generateMockResults(),
          relevance_scores: Array.from({ length: 10 }, () => Math.random())
        });
        
      } catch (error) {
        results.push({
          query,
          latency_ms: Date.now() - startTime,
          results: [],
          relevance_scores: [],
          error: error instanceof Error ? error.message : 'Unknown error'
        });
      }
    }
    
    return results;
  }

  private generateMockResults(): SearchResult[] {
    return Array.from({ length: 10 }, (_, i) => ({
      rank: i + 1,
      file_path: `src/file_${i}.ts`,
      line_number: Math.floor(Math.random() * 100) + 1,
      score: 1 - (i * 0.1),
      relevant: Math.random() > 0.3  // 70% relevant
    }));
  }

  // Mock metric computation methods
  private computeNdcg(results1: QueryResult[], results2: QueryResult[]): number {
    // Mock nDCG computation - would use proper relevance judgments
    return 0.6 + Math.random() * 0.3;  // Random value between 0.6-0.9
  }

  private computeRecallAtSla(results1: QueryResult[], results2: QueryResult[]): number {
    // Mock Recall@50 computation with SLA constraint
    const slaResults = [...results1, ...results2].filter(r => r.latency_ms <= 150);
    return 0.7 + Math.random() * 0.25;  // Random value between 0.7-0.95
  }

  private computeEce(results1: QueryResult[], results2: QueryResult[]): number {
    // Mock Expected Calibration Error
    return Math.random() * 0.08;  // Random value between 0-0.08
  }

  private computeSuccessRate(results: QueryResult[]): number {
    // Mock Success@10 computation
    return 0.6 + Math.random() * 0.35;  // Random value between 0.6-0.95
  }

  private computeQpsAtSla(results: QueryResult[], slaMs: number): number {
    const slaResults = results.filter(r => r.latency_ms <= slaMs);
    if (slaResults.length === 0) return 0;
    
    const avgLatency = slaResults.reduce((sum, r) => sum + r.latency_ms, 0) / slaResults.length;
    return 1000 / avgLatency;  // QPS approximation
  }

  private percentile(values: number[], p: number): number {
    const sorted = values.slice().sort((a, b) => a - b);
    const index = Math.ceil(sorted.length * p) - 1;
    return sorted[Math.max(0, index)];
  }

  private averageMetric(metrics: BakeoffMetrics[], field: keyof BakeoffMetrics): number {
    if (metrics.length === 0) return 0;
    return metrics.reduce((sum, m) => sum + (m[field] as number), 0) / metrics.length;
  }

  private computeParetoScores(results: BakeoffResult[]): void {
    // Compute Pareto efficiency score balancing quality vs performance
    for (const result of results) {
      if (result.metrics.length === 0) {
        result.pareto_score = 0;
        continue;
      }

      const avgNdcg = this.averageMetric(result.metrics, 'ndcg_at_10');
      const avgP95 = this.averageMetric(result.metrics, 'p95_latency_ms');
      
      // Normalize metrics (higher is better for both)
      const qualityScore = avgNdcg / 1.0;  // Normalize by max possible nDCG
      const performanceScore = Math.max(0, 1 - avgP95 / this.config.max_p95_ms);
      
      // Weighted combination (favor quality slightly)
      result.pareto_score = 0.6 * qualityScore + 0.4 * performanceScore;
    }
  }

  private printSummary(results: BakeoffResult[]): void {
    console.log('\n📋 EMBEDDING BAKE-OFF SUMMARY');
    console.log('=' .repeat(60));
    
    for (const result of results) {
      const avgNdcg = this.averageMetric(result.metrics, 'ndcg_at_10');
      const avgP95 = this.averageMetric(result.metrics, 'p95_latency_ms');
      const avgQps = this.averageMetric(result.metrics, 'qps_at_150ms');
      
      console.log(`\n${this.getRecommendationIcon(result.recommendation)} ${result.candidate.name}`);
      console.log(`   Recommendation: ${result.recommendation.toUpperCase()}`);
      console.log(`   Pareto Score: ${result.pareto_score.toFixed(3)}`);
      console.log(`   Avg nDCG@10: ${avgNdcg.toFixed(3)}`);
      console.log(`   Avg P95: ${avgP95.toFixed(1)}ms`);
      console.log(`   Avg QPS@150ms: ${avgQps.toFixed(1)}`);
      
      if (result.notes.length > 0) {
        console.log(`   Notes: ${result.notes.join(', ')}`);
      }
    }
    
    const selected = results.filter(r => r.recommendation === 'select');
    console.log(`\n🎯 SELECTED MODELS: ${selected.length}`);
    
    for (const result of selected) {
      console.log(`   ✅ ${result.candidate.name} (Pareto: ${result.pareto_score.toFixed(3)})`);
    }
  }

  private getRecommendationIcon(rec: string): string {
    switch (rec) {
      case 'select': return '🟢';
      case 'investigate': return '🟡';
      case 'reject': return '🔴';
      default: return '⚪';
    }
  }

  /**
   * Add custom candidate model
   */
  addCandidate(candidate: BakeoffCandidate): void {
    this.candidates.set(candidate.id, candidate);
  }

  /**
   * Update evaluation configuration
   */
  updateConfig(config: Partial<BakeoffConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get evaluation configuration
   */
  getConfig(): BakeoffConfig {
    return { ...this.config };
  }
}

interface QueryResult {
  query: string;
  latency_ms: number;
  results: SearchResult[];
  relevance_scores: number[];
  error?: string;
}

interface SearchResult {
  rank: number;
  file_path: string;
  line_number: number;
  score: number;
  relevant: boolean;
}