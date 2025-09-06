/**
 * Metrics Calculator and A/B Testing Framework
 * Implements TODO.md metrics: Recall@10/50, nDCG@10, MRR, FirstRelevantTokens, etc.
 */

import type { ABTestResult, BenchmarkRun } from '../types/benchmark.js';

export interface QueryResult {
  item: {
    id: string;
    query: string;
    expected_results: Array<{
      file: string;
      line: number;
      col: number;
      relevance_score: number;
      match_type: string;
    }>;
  };
  result: {
    hits: Array<{
      file: string;
      line: number;
      col: number;
      score: number;
      why: string[];
    }>;
    total: number;
    latency_ms: {
      stage_a: number;
      stage_b: number;
      stage_c?: number;
      total?: number;
    };
    stage_candidates?: {
      stage_a: number;
      stage_b: number;
      stage_c?: number;
    };
  };
}

export class MetricsCalculator {
  
  /**
   * Calculate comprehensive metrics for benchmark results
   */
  async calculateMetrics(
    queryResults: QueryResult[],
    cbuCoefficients: { gamma: number; delta: number; beta: number } = { gamma: 1.0, delta: 0.5, beta: 0.3 }
  ): Promise<BenchmarkRun['metrics']> {
    const recallAt10 = this.calculateRecallAtK(queryResults, 10);
    const recallAt50 = this.calculateRecallAtK(queryResults, 50);
    const ndcgAt10 = this.calculateNDCGAtK(queryResults, 10);
    const mrr = this.calculateMRR(queryResults);
    const firstRelevantTokens = this.calculateFirstRelevantTokens(queryResults);
    
    const stageLatencies = this.calculateStageLatencies(queryResults);
    const fanOutSizes = this.calculateFanOutSizes(queryResults);
    const whyAttributions = this.calculateWhyAttributions(queryResults);
    
    // TODO.md specified metrics
    const cbuScore = this.calculateCBU(queryResults, cbuCoefficients);
    const { ece } = this.calculateECE(queryResults);
    
    return {
      recall_at_10: recallAt10,
      recall_at_50: recallAt50,
      ndcg_at_10: ndcgAt10,
      mrr,
      first_relevant_tokens: firstRelevantTokens,
      stage_latencies: stageLatencies,
      fan_out_sizes: fanOutSizes,
      why_attributions: whyAttributions,
      cbu_score: cbuScore,
      ece_score: ece,
      kv_reuse_rate: 0.8 // Placeholder - would be calculated from actual KV cache metrics
    };
  }

  /**
   * Recall@K: Fraction of relevant documents retrieved in top-K results
   */
  private calculateRecallAtK(queryResults: QueryResult[], k: number): number {
    let totalRelevant = 0;
    let totalRetrievedRelevant = 0;
    
    for (const qr of queryResults) {
      const relevant = qr.item.expected_results;
      const retrieved = qr.result.hits.slice(0, k);
      
      totalRelevant += relevant.length;
      
      // Count how many relevant docs were retrieved
      let retrievedRelevant = 0;
      for (const hit of retrieved) {
        if (this.isRelevant(hit, relevant)) {
          retrievedRelevant++;
        }
      }
      
      totalRetrievedRelevant += retrievedRelevant;
    }
    
    return totalRelevant > 0 ? totalRetrievedRelevant / totalRelevant : 0;
  }

  /**
   * nDCG@K: Normalized Discounted Cumulative Gain at K
   */
  private calculateNDCGAtK(queryResults: QueryResult[], k: number): number {
    let totalNDCG = 0;
    let validQueries = 0;
    
    for (const qr of queryResults) {
      const dcg = this.calculateDCG(qr, k);
      const idcg = this.calculateIDCG(qr, k);
      
      if (idcg > 0) {
        totalNDCG += dcg / idcg;
        validQueries++;
      }
    }
    
    return validQueries > 0 ? totalNDCG / validQueries : 0;
  }

  private calculateDCG(qr: QueryResult, k: number): number {
    let dcg = 0;
    const retrieved = qr.result.hits.slice(0, k);
    
    for (let i = 0; i < retrieved.length; i++) {
      const hit = retrieved[i];
      if (!hit) continue;
      const relevanceScore = this.getRelevanceScore(hit, qr.item.expected_results);
      
      if (i === 0) {
        dcg += relevanceScore;
      } else {
        dcg += relevanceScore / Math.log2(i + 2);
      }
    }
    
    return dcg;
  }

  private calculateIDCG(qr: QueryResult, k: number): number {
    // Sort expected results by relevance score in descending order
    const sortedRelevant = [...qr.item.expected_results]
      .sort((a, b) => b.relevance_score - a.relevance_score)
      .slice(0, k);
    
    let idcg = 0;
    for (let i = 0; i < sortedRelevant.length; i++) {
      const item = sortedRelevant[i];
      if (!item) continue;
      const relevance = item.relevance_score;
      
      if (i === 0) {
        idcg += relevance;
      } else {
        idcg += relevance / Math.log2(i + 2);
      }
    }
    
    return idcg;
  }

  /**
   * MRR: Mean Reciprocal Rank
   */
  private calculateMRR(queryResults: QueryResult[]): number {
    let totalRR = 0;
    
    for (const qr of queryResults) {
      const firstRelevantRank = this.findFirstRelevantRank(qr);
      if (firstRelevantRank > 0) {
        totalRR += 1 / firstRelevantRank;
      }
    }
    
    return queryResults.length > 0 ? totalRR / queryResults.length : 0;
  }

  private findFirstRelevantRank(qr: QueryResult): number {
    for (let i = 0; i < qr.result.hits.length; i++) {
      const hit = qr.result.hits[i];
      if (hit && this.isRelevant(hit, qr.item.expected_results)) {
        return i + 1; // Rank is 1-indexed
      }
    }
    return 0; // No relevant result found
  }

  /**
   * First Relevant Tokens: Position of first relevant token in results
   */
  private calculateFirstRelevantTokens(queryResults: QueryResult[]): number {
    let totalTokens = 0;
    let validQueries = 0;
    
    for (const qr of queryResults) {
      let tokenCount = 0;
      let foundRelevant = false;
      
      for (const hit of qr.result.hits) {
        // Estimate tokens per hit (file path + context)
        const estimatedTokens = hit.file.split('/').length + 10; // Rough estimate
        tokenCount += estimatedTokens;
        
        if (this.isRelevant(hit, qr.item.expected_results)) {
          totalTokens += tokenCount;
          validQueries++;
          foundRelevant = true;
          break;
        }
      }
      
      if (!foundRelevant) {
        // No relevant result found, count all tokens
        totalTokens += tokenCount;
        validQueries++;
      }
    }
    
    return validQueries > 0 ? totalTokens / validQueries : 0;
  }

  private calculateStageLatencies(queryResults: QueryResult[]) {
    const latencies = {
      stage_a: [] as number[],
      stage_b: [] as number[],
      stage_c: [] as number[],
      e2e: [] as number[]
    };
    
    for (const qr of queryResults) {
      const lat = qr.result.latency_ms;
      latencies.stage_a.push(lat.stage_a);
      latencies.stage_b.push(lat.stage_b);
      
      if (lat.stage_c && lat.stage_c > 0) {
        latencies.stage_c.push(lat.stage_c);
      }
      
      const e2e = lat.total || (lat.stage_a + lat.stage_b + (lat.stage_c || 0));
      latencies.e2e.push(e2e);
    }
    
    return {
      stage_a_p50: this.percentile(latencies.stage_a, 0.5),
      stage_a_p95: this.percentile(latencies.stage_a, 0.95),
      stage_b_p50: this.percentile(latencies.stage_b, 0.5),
      stage_b_p95: this.percentile(latencies.stage_b, 0.95),
      stage_c_p50: latencies.stage_c.length > 0 ? this.percentile(latencies.stage_c, 0.5) : undefined,
      stage_c_p95: latencies.stage_c.length > 0 ? this.percentile(latencies.stage_c, 0.95) : undefined,
      e2e_p50: this.percentile(latencies.e2e, 0.5),
      e2e_p95: this.percentile(latencies.e2e, 0.95)
    };
  }

  private calculateFanOutSizes(queryResults: QueryResult[]) {
    const fanOuts = {
      stage_a: [] as number[],
      stage_b: [] as number[],
      stage_c: [] as number[]
    };
    
    for (const qr of queryResults) {
      const candidates = qr.result.stage_candidates;
      if (candidates) {
        fanOuts.stage_a.push(candidates.stage_a);
        fanOuts.stage_b.push(candidates.stage_b);
        
        if (candidates.stage_c && candidates.stage_c > 0) {
          fanOuts.stage_c.push(candidates.stage_c);
        }
      }
    }
    
    return {
      stage_a: fanOuts.stage_a.length > 0 ? Math.round(this.mean(fanOuts.stage_a)) : 0,
      stage_b: fanOuts.stage_b.length > 0 ? Math.round(this.mean(fanOuts.stage_b)) : 0,
      stage_c: fanOuts.stage_c.length > 0 ? Math.round(this.mean(fanOuts.stage_c)) : undefined
    };
  }

  private calculateWhyAttributions(queryResults: QueryResult[]): Record<string, number> {
    const attributions: Record<string, number> = {
      exact: 0,
      symbol: 0,
      struct: 0,
      semantic: 0
    };
    
    for (const qr of queryResults) {
      for (const hit of qr.result.hits) {
        for (const why of hit.why) {
          if (Object.prototype.hasOwnProperty.call(attributions, why)) {
            attributions[why] = (attributions[why] || 0) + 1;
          }
        }
      }
    }
    
    return attributions;
  }

  /**
   * Calculate CBU (Composite Benchmark Utility) score using TODO.md formula
   * CBU = γ * Recall@50 + δ * (1 - normalized_latency) + β * (1 - tokens/B)
   */
  private calculateCBU(
    queryResults: QueryResult[], 
    coefficients: { gamma: number; delta: number; beta: number } = { gamma: 1.0, delta: 0.5, beta: 0.3 }
  ): number {
    const recall50 = this.calculateRecallAtK(queryResults, 50);
    
    // Calculate normalized latency (assuming baseline of 20ms, cap at 150ms)
    const latencies = queryResults
      .map(r => {
        const lat = r.result.latency_ms;
        if (typeof lat === 'number') {
          return lat;
        } else if (lat && typeof lat === 'object' && 'total' in lat) {
          return lat.total || 0;
        } else {
          return 0;
        }
      })
      .filter(l => l > 0);
    
    const avgLatency = latencies.length > 0 ? this.mean(latencies) : 20;
    const normalizedLatency = Math.min(avgLatency / 150, 1.0); // Normalize to [0,1]
    
    // Calculate verbosity penalty (tokens per result, normalized)
    const tokenCounts = queryResults
      .flatMap(r => r.result.hits.map(hit => (hit as any).snippet?.length || 0))
      .filter(count => count > 0);
    
    const avgTokens = tokenCounts.length > 0 ? this.mean(tokenCounts) : 100;
    const normalizedVerbosity = Math.min(avgTokens / 500, 1.0); // Normalize to [0,1], assuming 500 as verbose
    
    // CBU formula from TODO.md
    const cbu = coefficients.gamma * recall50 + 
                coefficients.delta * (1 - normalizedLatency) + 
                coefficients.beta * (1 - normalizedVerbosity);
    
    return Math.max(0, Math.min(1, cbu)); // Clamp to [0,1]
  }

  /**
   * Calculate ECE (Expected Calibration Error) with reliability diagrams
   * Measures calibration quality of confidence predictions
   */
  private calculateECE(
    queryResults: QueryResult[], 
    nBins: number = 10
  ): { ece: number; reliabilityDiagram: Array<{ binCenter: number; accuracy: number; confidence: number; count: number }> } {
    // Extract confidence scores and binary relevance
    const predictions: Array<{ confidence: number; isRelevant: boolean }> = [];
    
    for (const result of queryResults) {
      for (const hit of result.result.hits) {
        // Use search score as confidence proxy (normalized to [0,1])
        const confidence = Math.max(0, Math.min(1, hit.score || 0.5));
        const isRelevant = (hit as any).relevance_score ? (hit as any).relevance_score > 0.5 : false;
        predictions.push({ confidence, isRelevant });
      }
    }
    
    if (predictions.length === 0) {
      return { ece: 1.0, reliabilityDiagram: [] };
    }
    
    // Create bins
    const binSize = 1.0 / nBins;
    const bins: Array<{ accuracySum: number; confidenceSum: number; count: number }> = 
      Array(nBins).fill(null).map(() => ({ accuracySum: 0, confidenceSum: 0, count: 0 }));
    
    // Assign predictions to bins
    for (const pred of predictions) {
      const binIndex = Math.min(Math.floor(pred.confidence / binSize), nBins - 1);
      bins[binIndex].confidenceSum += pred.confidence;
      bins[binIndex].accuracySum += pred.isRelevant ? 1 : 0;
      bins[binIndex].count += 1;
    }
    
    // Calculate ECE and reliability diagram
    let ece = 0;
    const reliabilityDiagram = [];
    
    for (let i = 0; i < nBins; i++) {
      const bin = bins[i];
      if (bin.count > 0) {
        const avgConfidence = bin.confidenceSum / bin.count;
        const avgAccuracy = bin.accuracySum / bin.count;
        const binWeight = bin.count / predictions.length;
        
        ece += binWeight * Math.abs(avgConfidence - avgAccuracy);
        
        reliabilityDiagram.push({
          binCenter: (i + 0.5) * binSize,
          accuracy: avgAccuracy,
          confidence: avgConfidence,
          count: bin.count
        });
      }
    }
    
    return { ece, reliabilityDiagram };
  }

  /**
   * A/B Testing with paired comparison and bootstrap confidence intervals
   */
  async performABTest(
    baselineResults: QueryResult[],
    treatmentResults: QueryResult[],
    metric: string = 'ndcg_at_10'
  ): Promise<ABTestResult> {
    
    // Extract paired metric values
    const baselineMetrics = await this.calculateMetrics(baselineResults);
    const treatmentMetrics = await this.calculateMetrics(treatmentResults);
    
    const baselineMean = this.getMetricValue(baselineMetrics, metric);
    const treatmentMean = this.getMetricValue(treatmentMetrics, metric);
    
    const delta = treatmentMean - baselineMean;
    const deltaPercent = baselineMean > 0 ? (delta / baselineMean) * 100 : 0;
    
    // Bootstrap confidence interval
    const { ciLower, ciUpper } = this.bootstrapCI(
      baselineResults, 
      treatmentResults, 
      metric
    );
    
    // Permutation test for significance
    const pValue = this.permutationTest(
      baselineResults,
      treatmentResults,
      metric
    );
    
    const isSignificant = pValue < 0.05;
    
    // Effect size (Cohen's d approximation)
    const effectSize = this.calculateEffectSize(baselineResults, treatmentResults, metric);
    
    return {
      metric,
      baseline_mean: baselineMean,
      treatment_mean: treatmentMean,
      delta,
      delta_percent: deltaPercent,
      ci_lower: ciLower,
      ci_upper: ciUpper,
      p_value: pValue,
      is_significant: isSignificant,
      sample_size: Math.min(baselineResults.length, treatmentResults.length),
      effect_size: effectSize
    };
  }

  private bootstrapCI(
    baseline: QueryResult[],
    treatment: QueryResult[],
    metric: string,
    bootstrapSamples: number = 1000,
    alpha: number = 0.05
  ): { ciLower: number; ciUpper: number } {
    
    const deltas: number[] = [];
    const n = Math.min(baseline.length, treatment.length);
    
    for (let i = 0; i < bootstrapSamples; i++) {
      // Bootstrap sample with replacement
      const baselineSample = this.bootstrapSample(baseline, n);
      const treatmentSample = this.bootstrapSample(treatment, n);
      
      // Calculate metrics for samples
      const baselineMetrics = this.calculateMetricsSync(baselineSample);
      const treatmentMetrics = this.calculateMetricsSync(treatmentSample);
      
      const baselineValue = this.getMetricValue(baselineMetrics, metric);
      const treatmentValue = this.getMetricValue(treatmentMetrics, metric);
      
      deltas.push(treatmentValue - baselineValue);
    }
    
    deltas.sort((a, b) => a - b);
    
    const lowerIdx = Math.floor((alpha / 2) * bootstrapSamples);
    const upperIdx = Math.floor((1 - alpha / 2) * bootstrapSamples);
    
    return {
      ciLower: deltas[lowerIdx] || 0,
      ciUpper: deltas[upperIdx] || 0
    };
  }

  private permutationTest(
    baseline: QueryResult[],
    treatment: QueryResult[],
    metric: string,
    permutations: number = 1000
  ): number {
    
    // Calculate observed difference
    const baselineMetrics = this.calculateMetricsSync(baseline);
    const treatmentMetrics = this.calculateMetricsSync(treatment);
    
    const observedDiff = this.getMetricValue(treatmentMetrics, metric) - 
                        this.getMetricValue(baselineMetrics, metric);
    
    // Combine all results for permutation
    const combined = [...baseline, ...treatment];
    const n1 = baseline.length;
    const n2 = treatment.length;
    
    let extremeCount = 0;
    
    for (let i = 0; i < permutations; i++) {
      // Shuffle combined results
      const shuffled = this.shuffle([...combined]);
      
      const permBaseline = shuffled.slice(0, n1);
      const permTreatment = shuffled.slice(n1, n1 + n2);
      
      const permBaselineMetrics = this.calculateMetricsSync(permBaseline);
      const permTreatmentMetrics = this.calculateMetricsSync(permTreatment);
      
      const permDiff = this.getMetricValue(permTreatmentMetrics, metric) - 
                      this.getMetricValue(permBaselineMetrics, metric);
      
      if (Math.abs(permDiff) >= Math.abs(observedDiff)) {
        extremeCount++;
      }
    }
    
    return extremeCount / permutations;
  }

  private calculateEffectSize(
    baseline: QueryResult[],
    treatment: QueryResult[],
    metric: string
  ): number {
    // Simplified Cohen's d calculation
    const baselineValues = baseline.map(qr => this.getQueryMetricValue(qr, metric));
    const treatmentValues = treatment.map(qr => this.getQueryMetricValue(qr, metric));
    
    const baselineMean = this.mean(baselineValues);
    const treatmentMean = this.mean(treatmentValues);
    
    const pooledStd = Math.sqrt(
      ((baselineValues.length - 1) * this.variance(baselineValues) +
       (treatmentValues.length - 1) * this.variance(treatmentValues)) /
      (baselineValues.length + treatmentValues.length - 2)
    );
    
    return pooledStd > 0 ? (treatmentMean - baselineMean) / pooledStd : 0;
  }

  // Utility functions
  aggregateMetrics(metricsArray: BenchmarkRun['metrics'][]): BenchmarkRun['metrics'] {
    if (metricsArray.length === 0) {
      throw new Error('Cannot aggregate empty metrics array');
    }

    // Calculate mean for each metric
    const aggregated: BenchmarkRun['metrics'] = {
      recall_at_10: this.mean(metricsArray.map(m => m.recall_at_10)),
      recall_at_50: this.mean(metricsArray.map(m => m.recall_at_50)),
      ndcg_at_10: this.mean(metricsArray.map(m => m.ndcg_at_10)),
      mrr: this.mean(metricsArray.map(m => m.mrr)),
      first_relevant_tokens: this.mean(metricsArray.map(m => m.first_relevant_tokens)),
      stage_latencies: {
        stage_a_p50: this.mean(metricsArray.map(m => m.stage_latencies.stage_a_p50)),
        stage_a_p95: this.mean(metricsArray.map(m => m.stage_latencies.stage_a_p95)),
        stage_b_p50: this.mean(metricsArray.map(m => m.stage_latencies.stage_b_p50)),
        stage_b_p95: this.mean(metricsArray.map(m => m.stage_latencies.stage_b_p95)),
        stage_c_p50: metricsArray[0]?.stage_latencies.stage_c_p50 !== undefined ?
          this.mean(metricsArray.map(m => m.stage_latencies.stage_c_p50).filter((v): v is number => v !== undefined)) : undefined,
        stage_c_p95: metricsArray[0]?.stage_latencies.stage_c_p95 !== undefined ?
          this.mean(metricsArray.map(m => m.stage_latencies.stage_c_p95).filter((v): v is number => v !== undefined)) : undefined,
        e2e_p50: this.mean(metricsArray.map(m => m.stage_latencies.e2e_p50)),
        e2e_p95: this.mean(metricsArray.map(m => m.stage_latencies.e2e_p95))
      },
      fan_out_sizes: {
        stage_a: Math.round(this.mean(metricsArray.map(m => m.fan_out_sizes.stage_a))),
        stage_b: Math.round(this.mean(metricsArray.map(m => m.fan_out_sizes.stage_b))),
        stage_c: metricsArray[0]?.fan_out_sizes.stage_c !== undefined ?
          Math.round(this.mean(metricsArray.map(m => m.fan_out_sizes.stage_c).filter((v): v is number => v !== undefined))) : undefined
      },
      why_attributions: this.aggregateWhyAttributions(metricsArray.map(m => m.why_attributions))
    };

    return aggregated;
  }

  private aggregateWhyAttributions(attributionsArray: Record<string, number>[]): Record<string, number> {
    const aggregated: Record<string, number> = {};
    
    for (const attributions of attributionsArray) {
      for (const [key, value] of Object.entries(attributions)) {
        aggregated[key] = (aggregated[key] || 0) + value;
      }
    }
    
    return aggregated;
  }

  // Helper functions
  private isRelevant(
    hit: { file: string; line: number; col: number },
    expectedResults: Array<{ file: string; line: number; col: number; relevance_score: number }>
  ): boolean {
    return expectedResults.some(expected =>
      expected.file === hit.file &&
      Math.abs(expected.line - hit.line) <= 2 && // Allow small line number differences
      Math.abs(expected.col - hit.col) <= 10    // Allow small column differences
    );
  }

  private getRelevanceScore(
    hit: { file: string; line: number; col: number },
    expectedResults: Array<{ file: string; line: number; col: number; relevance_score: number }>
  ): number {
    const relevant = expectedResults.find(expected =>
      expected.file === hit.file &&
      Math.abs(expected.line - hit.line) <= 2 &&
      Math.abs(expected.col - hit.col) <= 10
    );
    
    return relevant ? relevant.relevance_score : 0;
  }

  private calculateMetricsSync(queryResults: QueryResult[]): BenchmarkRun['metrics'] {
    // Synchronous version of calculateMetrics for bootstrap/permutation tests
    return {
      recall_at_10: this.calculateRecallAtK(queryResults, 10),
      recall_at_50: this.calculateRecallAtK(queryResults, 50),
      ndcg_at_10: this.calculateNDCGAtK(queryResults, 10),
      mrr: this.calculateMRR(queryResults),
      first_relevant_tokens: this.calculateFirstRelevantTokens(queryResults),
      stage_latencies: this.calculateStageLatencies(queryResults),
      fan_out_sizes: this.calculateFanOutSizes(queryResults),
      why_attributions: this.calculateWhyAttributions(queryResults)
    };
  }

  private getMetricValue(metrics: BenchmarkRun['metrics'], metricName: string): number {
    switch (metricName) {
      case 'recall_at_10': return metrics.recall_at_10;
      case 'recall_at_50': return metrics.recall_at_50;
      case 'ndcg_at_10': return metrics.ndcg_at_10;
      case 'mrr': return metrics.mrr;
      case 'first_relevant_tokens': return metrics.first_relevant_tokens;
      case 'e2e_p95': return metrics.stage_latencies.e2e_p95;
      default: return 0;
    }
  }

  private getQueryMetricValue(qr: QueryResult, metricName: string): number {
    // Calculate metric for single query - simplified implementation
    switch (metricName) {
      case 'ndcg_at_10':
        const dcg = this.calculateDCG(qr, 10);
        const idcg = this.calculateIDCG(qr, 10);
        return idcg > 0 ? dcg / idcg : 0;
      default:
        return 0;
    }
  }

  // Statistical utility functions
  private mean(values: number[]): number {
    return values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : 0;
  }

  private variance(values: number[]): number {
    if (values.length === 0) return 0;
    const mean = this.mean(values);
    return values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
  }

  private percentile(values: number[], p: number): number {
    if (values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const index = (p * (sorted.length - 1));
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    const weight = index - lower;
    
    if (lower === upper) {
      return sorted[lower] || 0;
    }
    
    return (sorted[lower] || 0) * (1 - weight) + (sorted[upper] || 0) * weight;
  }

  private bootstrapSample<T>(array: T[], size: number): T[] {
    const sample: T[] = [];
    for (let i = 0; i < size; i++) {
      const randomIndex = Math.floor(Math.random() * array.length);
      const item = array[randomIndex];
      if (item !== undefined) {
        sample.push(item);
      }
    }
    return sample;
  }

  private shuffle<T>(array: T[]): T[] {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const temp = shuffled[i];
      const other = shuffled[j];
      if (temp !== undefined && other !== undefined) {
        shuffled[i] = other;
        shuffled[j] = temp;
      }
    }
    return shuffled;
  }
}