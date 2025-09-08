/**
 * Comprehensive Metrics Calculator - Quality, Operations, Calibration & Explainability
 * 
 * This implements all SLA-bounded metrics specified in the protocol:
 * - Quality: nDCG@10, Success@10, SLA-Recall@50, witness_coverage@10
 * - Operations: p50/p95/p99, p99/p95 ratio, QPS@150ms, timeout%
 * - Calibration: ECE, slope/intercept per intent×language  
 * - Explainability: why_mix, Core@10, Diversity@10, span coverage
 */

import { ExecutionResult } from './sla-execution-engine';
import { QrelJudgment } from './pooled-qrels-builder';

export interface QualityMetrics {
  ndcg_at_10: number;
  success_at_10: number;
  sla_recall_at_50: number;
  witness_coverage_at_10: number;
  precision_at_10: number;
  recall_at_10: number;
  map_at_10: number; // Mean Average Precision
}

export interface OperationalMetrics {
  p50_latency_ms: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  p99_over_p95_ratio: number;
  qps_at_150ms: number;
  timeout_percentage: number;
  error_rate: number;
  sla_compliance_rate: number;
  avg_concurrent_queries: number;
}

export interface CalibrationMetrics {
  expected_calibration_error: number;
  calibration_slope: number;
  calibration_intercept: number;
  reliability_buckets: CalibrationBucket[];
  max_calibration_error: number;
  brier_score: number;
}

export interface CalibrationBucket {
  predicted_prob: number;
  actual_prob: number;
  count: number;
  confidence_interval: [number, number];
}

export interface ExplainabilityMetrics {
  why_mix_exact: number;
  why_mix_struct: number;  
  why_mix_semantic: number;
  why_mix_mixed: number;
  core_at_10: number;
  diversity_at_10: number;
  span_coverage: number;
  unique_files_at_10: number;
  symbol_coverage: number;
}

export interface AggregateMetrics {
  query_id: string;
  system_id: string;
  suite: string;
  intent: string;
  language: string;
  
  quality: QualityMetrics;
  operations: OperationalMetrics;
  calibration: CalibrationMetrics;
  explainability: ExplainabilityMetrics;
  
  total_queries: number;
  within_sla_queries: number;
  timestamp: string;
}

export interface WitnessValidation {
  query_id: string;
  expected_spans: WitnessSpan[];
  found_spans: WitnessSpan[];
  coverage_at_k: number[];
  miss_reasons: string[];
}

interface WitnessSpan {
  file: string;
  line_start: number;
  line_end: number;
  content: string;
  span_type: 'change' | 'context' | 'test';
}

/**
 * Main metrics calculator with comprehensive evaluation
 */
export class MetricsCalculator {
  private qrels: Map<string, QrelJudgment[]> = new Map();
  private witnessData: Map<string, WitnessValidation> = new Map();

  /**
   * Load qrels for evaluation
   */
  loadQrels(qrels: QrelJudgment[]): void {
    for (const qrel of qrels) {
      if (!this.qrels.has(qrel.query_id)) {
        this.qrels.set(qrel.query_id, []);
      }
      this.qrels.get(qrel.query_id)!.push(qrel);
    }
    
    console.log(`📊 Loaded ${qrels.length} qrels for ${this.qrels.size} queries`);
  }

  /**
   * Load witness data for SWE-bench validation
   */
  loadWitnessData(witnessData: WitnessValidation[]): void {
    for (const witness of witnessData) {
      this.witnessData.set(witness.query_id, witness);
    }
    
    console.log(`📋 Loaded witness data for ${this.witnessData.size} queries`);
  }

  /**
   * Calculate comprehensive metrics for a set of execution results
   */
  calculateMetrics(results: ExecutionResult[]): AggregateMetrics[] {
    console.log(`🧮 Calculating metrics for ${results.length} results`);
    
    // Group results by (query_id, system_id)
    const groupedResults = this.groupResultsByQueryAndSystem(results);
    
    const aggregateMetrics: AggregateMetrics[] = [];
    
    for (const [key, queryResults] of groupedResults) {
      const [queryId, systemId] = key.split('|');
      const representative = queryResults[0];
      
      const metrics: AggregateMetrics = {
        query_id: queryId,
        system_id: systemId,
        suite: representative.suite,
        intent: representative.intent,
        language: representative.language,
        
        quality: this.calculateQualityMetrics(queryResults),
        operations: this.calculateOperationalMetrics(queryResults),
        calibration: this.calculateCalibrationMetrics(queryResults),
        explainability: this.calculateExplainabilityMetrics(queryResults),
        
        total_queries: queryResults.length,
        within_sla_queries: queryResults.filter(r => r.within_sla).length,
        timestamp: new Date().toISOString()
      };
      
      aggregateMetrics.push(metrics);
    }
    
    console.log(`✅ Calculated metrics for ${aggregateMetrics.length} query-system pairs`);
    return aggregateMetrics;
  }

  /**
   * Calculate quality metrics (nDCG@10, Success@10, etc.)
   */
  private calculateQualityMetrics(results: ExecutionResult[]): QualityMetrics {
    if (results.length === 0) {
      return this.getEmptyQualityMetrics();
    }

    const queryId = results[0].query_id;
    const qrels = this.qrels.get(queryId) || [];
    
    if (qrels.length === 0) {
      console.warn(`⚠️  No qrels found for query ${queryId}`);
      return this.getEmptyQualityMetrics();
    }

    // Only consider in-SLA results for quality metrics
    const slaResults = results.filter(r => r.within_sla);
    const hits = slaResults.flatMap(r => r.hits).slice(0, 50);
    
    return {
      ndcg_at_10: this.calculateNDCG(hits.slice(0, 10), qrels),
      success_at_10: this.calculateSuccessAtK(hits.slice(0, 10), qrels),
      sla_recall_at_50: this.calculateRecallAtK(hits, qrels),
      witness_coverage_at_10: this.calculateWitnessCoverage(hits.slice(0, 10), queryId),
      precision_at_10: this.calculatePrecisionAtK(hits.slice(0, 10), qrels),
      recall_at_10: this.calculateRecallAtK(hits.slice(0, 10), qrels),
      map_at_10: this.calculateMAP(hits.slice(0, 10), qrels)
    };
  }

  /**
   * Calculate operational metrics (latency percentiles, throughput, etc.)
   */
  private calculateOperationalMetrics(results: ExecutionResult[]): OperationalMetrics {
    if (results.length === 0) {
      return this.getEmptyOperationalMetrics();
    }

    const latencies = results.map(r => r.server_latency_ms).sort((a, b) => a - b);
    const timeouts = results.filter(r => r.timeout_reason).length;
    const errors = results.filter(r => r.error).length;
    const withinSla = results.filter(r => r.within_sla).length;
    const concurrentQueries = results.map(r => r.concurrent_queries);

    const p50 = this.percentile(latencies, 50);
    const p95 = this.percentile(latencies, 95);
    const p99 = this.percentile(latencies, 99);

    // Calculate QPS@150ms (queries that completed within SLA)
    const slaLatencies = results.filter(r => r.within_sla).map(r => r.server_latency_ms);
    const avgSlaLatency = slaLatencies.length > 0 ? slaLatencies.reduce((a, b) => a + b, 0) / slaLatencies.length : 150;
    const qpsAt150ms = avgSlaLatency > 0 ? 1000 / avgSlaLatency : 0;

    return {
      p50_latency_ms: p50,
      p95_latency_ms: p95,
      p99_latency_ms: p99,
      p99_over_p95_ratio: p95 > 0 ? p99 / p95 : 0,
      qps_at_150ms: qpsAt150ms,
      timeout_percentage: (timeouts / results.length) * 100,
      error_rate: (errors / results.length) * 100,
      sla_compliance_rate: (withinSla / results.length) * 100,
      avg_concurrent_queries: concurrentQueries.reduce((a, b) => a + b, 0) / concurrentQueries.length
    };
  }

  /**
   * Calculate calibration metrics (ECE, slope, intercept)
   */
  private calculateCalibrationMetrics(results: ExecutionResult[]): CalibrationMetrics {
    if (results.length === 0) {
      return this.getEmptyCalibrationMetrics();
    }

    const queryId = results[0].query_id;
    const qrels = this.qrels.get(queryId) || [];
    
    if (qrels.length === 0) {
      return this.getEmptyCalibrationMetrics();
    }

    // Build calibration data points
    const calibrationData: Array<{ predicted: number; actual: number }> = [];
    
    for (const result of results.filter(r => r.within_sla)) {
      for (const hit of result.hits.slice(0, 10)) {
        const relevant = this.isHitRelevant(hit, qrels);
        calibrationData.push({
          predicted: hit.score,
          actual: relevant ? 1 : 0
        });
      }
    }

    if (calibrationData.length === 0) {
      return this.getEmptyCalibrationMetrics();
    }

    // Calculate ECE with 10 buckets
    const buckets = this.createCalibrationBuckets(calibrationData, 10);
    const ece = this.calculateECE(buckets);
    
    // Calculate calibration slope and intercept via linear regression
    const { slope, intercept } = this.calculateCalibrationRegression(calibrationData);
    
    // Calculate Brier score
    const brierScore = this.calculateBrierScore(calibrationData);
    
    return {
      expected_calibration_error: ece,
      calibration_slope: slope,
      calibration_intercept: intercept,
      reliability_buckets: buckets,
      max_calibration_error: Math.max(...buckets.map(b => Math.abs(b.predicted_prob - b.actual_prob))),
      brier_score: brierScore
    };
  }

  /**
   * Calculate explainability metrics (why_mix, diversity, etc.)
   */
  private calculateExplainabilityMetrics(results: ExecutionResult[]): ExplainabilityMetrics {
    if (results.length === 0) {
      return this.getEmptyExplainabilityMetrics();
    }

    const slaResults = results.filter(r => r.within_sla);
    const allHits = slaResults.flatMap(r => r.hits);
    const top10Hits = allHits.slice(0, 10);

    // Calculate why_mix distribution
    const whyTagCounts = { exact: 0, struct: 0, semantic: 0, mixed: 0 };
    for (const hit of top10Hits) {
      whyTagCounts[hit.why_tag]++;
    }
    
    const totalHits = top10Hits.length;
    const whyMix = {
      exact: totalHits > 0 ? whyTagCounts.exact / totalHits : 0,
      struct: totalHits > 0 ? whyTagCounts.struct / totalHits : 0,
      semantic: totalHits > 0 ? whyTagCounts.semantic / totalHits : 0,
      mixed: totalHits > 0 ? whyTagCounts.mixed / totalHits : 0
    };

    // Calculate diversity metrics
    const uniqueFiles = new Set(top10Hits.map(h => h.file)).size;
    const coreAt10 = this.calculateCoreAt10(top10Hits);
    const diversityAt10 = uniqueFiles / Math.min(10, top10Hits.length);
    
    // Calculate span coverage (must be 100% for valid results)
    const spanCoverage = this.calculateSpanCoverage(top10Hits);
    
    return {
      why_mix_exact: whyMix.exact,
      why_mix_struct: whyMix.struct,
      why_mix_semantic: whyMix.semantic,
      why_mix_mixed: whyMix.mixed,
      core_at_10: coreAt10,
      diversity_at_10: diversityAt10,
      span_coverage: spanCoverage,
      unique_files_at_10: uniqueFiles,
      symbol_coverage: this.calculateSymbolCoverage(top10Hits)
    };
  }

  /**
   * Calculate nDCG@k
   */
  private calculateNDCG(hits: any[], qrels: QrelJudgment[]): number {
    if (hits.length === 0 || qrels.length === 0) return 0;

    const relevanceMap = new Map<string, number>();
    const fileRelevanceMap = new Map<string, number>(); // File-level fallback
    
    for (const qrel of qrels) {
      const key = `${qrel.file}:${qrel.line}:${qrel.column}`;
      const fileKey = qrel.file;
      relevanceMap.set(key, qrel.relevance);
      // Store highest relevance for file-level fallback
      fileRelevanceMap.set(fileKey, Math.max(fileRelevanceMap.get(fileKey) || 0, qrel.relevance));
    }

    // Calculate DCG
    let dcg = 0;
    let spanMatches = 0;
    let fileMatches = 0;
    
    for (let i = 0; i < hits.length; i++) {
      const hit = hits[i];
      const key = `${hit.file}:${hit.line}:${hit.column}`;
      const fileKey = hit.file;
      
      // Try exact span match first
      let relevance = relevanceMap.get(key);
      if (relevance !== undefined) {
        spanMatches++;
      } else {
        // Fallback to file-level match for null line/column data
        relevance = fileRelevanceMap.get(fileKey);
        if (relevance !== undefined) {
          fileMatches++;
        } else {
          relevance = 0;
        }
      }
      
      const discount = Math.log2(i + 2); // i+2 because ranking starts at 1
      dcg += relevance / discount;
    }

    // Debug logging for diagnostics
    if (fileMatches > 0 && spanMatches === 0) {
      console.log(`⚠️  Using file-level fallback: ${fileMatches} file matches, ${spanMatches} span matches`);
    }

    // Calculate IDCG (perfect ranking)
    const sortedRelevances = qrels.map(q => q.relevance).sort((a, b) => b - a);
    let idcg = 0;
    for (let i = 0; i < Math.min(hits.length, sortedRelevances.length); i++) {
      const discount = Math.log2(i + 2);
      idcg += sortedRelevances[i] / discount;
    }

    return idcg > 0 ? dcg / idcg : 0;
  }

  /**
   * Calculate Success@k (binary success rate)
   */
  private calculateSuccessAtK(hits: any[], qrels: QrelJudgment[]): number {
    if (hits.length === 0 || qrels.length === 0) return 0;

    const relevantHits = new Set<string>();
    const relevantFiles = new Set<string>(); // File-level fallback
    
    for (const qrel of qrels) {
      if (qrel.relevance > 0) {
        relevantHits.add(`${qrel.file}:${qrel.line}:${qrel.column}`);
        relevantFiles.add(qrel.file);
      }
    }

    for (const hit of hits) {
      const key = `${hit.file}:${hit.line}:${hit.column}`;
      if (relevantHits.has(key)) {
        return 1; // Exact span match
      }
      // Fallback to file-level match for null line/column data  
      if (relevantFiles.has(hit.file)) {
        return 1; // File-level match
      }
    }

    return 0;
  }

  /**
   * Calculate Recall@k
   */
  private calculateRecallAtK(hits: any[], qrels: QrelJudgment[]): number {
    if (qrels.length === 0) return 0;

    const totalRelevant = qrels.filter(q => q.relevance > 0).length;
    if (totalRelevant === 0) return 0;

    const relevantHits = new Set<string>();
    const relevantFiles = new Set<string>(); // File-level fallback
    
    for (const qrel of qrels) {
      if (qrel.relevance > 0) {
        relevantHits.add(`${qrel.file}:${qrel.line}:${qrel.column}`);
        relevantFiles.add(qrel.file);
      }
    }

    const foundSpanHits = new Set<string>();
    const foundFileHits = new Set<string>();
    
    for (const hit of hits) {
      const key = `${hit.file}:${hit.line}:${hit.column}`;
      if (relevantHits.has(key)) {
        foundSpanHits.add(key);
      } else if (relevantFiles.has(hit.file)) {
        foundFileHits.add(hit.file);
      }
    }

    // Count unique relevant items found (prefer span matches over file matches)
    const totalFound = foundSpanHits.size + foundFileHits.size;
    return totalFound / totalRelevant;
  }

  /**
   * Calculate Precision@k
   */
  private calculatePrecisionAtK(hits: any[], qrels: QrelJudgment[]): number {
    if (hits.length === 0) return 0;

    const relevantHits = new Set<string>();
    const relevantFiles = new Set<string>(); // File-level fallback
    
    for (const qrel of qrels) {
      if (qrel.relevance > 0) {
        relevantHits.add(`${qrel.file}:${qrel.line}:${qrel.column}`);
        relevantFiles.add(qrel.file);
      }
    }

    let foundRelevant = 0;
    for (const hit of hits) {
      const key = `${hit.file}:${hit.line}:${hit.column}`;
      if (relevantHits.has(key)) {
        foundRelevant++;
      } else if (relevantFiles.has(hit.file)) {
        foundRelevant++; // File-level match
      }
    }

    return foundRelevant / hits.length;
  }

  /**
   * Calculate Mean Average Precision@k
   */
  private calculateMAP(hits: any[], qrels: QrelJudgment[]): number {
    if (hits.length === 0 || qrels.length === 0) return 0;

    const relevantHits = new Set<string>();
    const relevantFiles = new Set<string>(); // File-level fallback
    
    for (const qrel of qrels) {
      if (qrel.relevance > 0) {
        relevantHits.add(`${qrel.file}:${qrel.line}:${qrel.column}`);
        relevantFiles.add(qrel.file);
      }
    }

    const totalRelevant = Math.max(relevantHits.size, relevantFiles.size);
    if (totalRelevant === 0) return 0;

    let sumPrecision = 0;
    let foundRelevant = 0;

    for (let i = 0; i < hits.length; i++) {
      const hit = hits[i];
      const key = `${hit.file}:${hit.line}:${hit.column}`;
      
      if (relevantHits.has(key) || relevantFiles.has(hit.file)) {
        foundRelevant++;
        const precisionAtI = foundRelevant / (i + 1);
        sumPrecision += precisionAtI;
      }
    }

    return foundRelevant > 0 ? sumPrecision / foundRelevant : 0;
  }

  /**
   * Calculate witness coverage for SWE-bench
   */
  private calculateWitnessCoverage(hits: any[], queryId: string): number {
    const witnessData = this.witnessData.get(queryId);
    if (!witnessData) return 0;

    const expectedSpans = witnessData.expected_spans;
    if (expectedSpans.length === 0) return 1;

    const foundSpans = new Set<string>();
    
    for (const hit of hits) {
      for (const span of expectedSpans) {
        if (hit.file === span.file && 
            hit.line >= span.line_start && 
            hit.line <= span.line_end) {
          foundSpans.add(`${span.file}:${span.line_start}-${span.line_end}`);
        }
      }
    }

    return foundSpans.size / expectedSpans.length;
  }

  /**
   * Calculate Expected Calibration Error
   */
  private calculateECE(buckets: CalibrationBucket[]): number {
    let totalSamples = 0;
    let weightedError = 0;

    for (const bucket of buckets) {
      if (bucket.count > 0) {
        const error = Math.abs(bucket.predicted_prob - bucket.actual_prob);
        weightedError += bucket.count * error;
        totalSamples += bucket.count;
      }
    }

    return totalSamples > 0 ? weightedError / totalSamples : 0;
  }

  /**
   * Create calibration buckets for ECE calculation
   */
  private createCalibrationBuckets(data: Array<{ predicted: number; actual: number }>, numBuckets: number): CalibrationBucket[] {
    const buckets: CalibrationBucket[] = [];
    
    for (let i = 0; i < numBuckets; i++) {
      const minProb = i / numBuckets;
      const maxProb = (i + 1) / numBuckets;
      
      const bucketData = data.filter(d => d.predicted >= minProb && d.predicted < maxProb);
      if (i === numBuckets - 1) {
        // Include boundary case for last bucket
        bucketData.push(...data.filter(d => d.predicted === 1.0));
      }

      const avgPredicted = bucketData.length > 0 ? 
        bucketData.reduce((sum, d) => sum + d.predicted, 0) / bucketData.length : 
        (minProb + maxProb) / 2;
        
      const avgActual = bucketData.length > 0 ? 
        bucketData.reduce((sum, d) => sum + d.actual, 0) / bucketData.length : 
        0;

      buckets.push({
        predicted_prob: avgPredicted,
        actual_prob: avgActual,
        count: bucketData.length,
        confidence_interval: this.calculateBinomialCI(avgActual, bucketData.length)
      });
    }

    return buckets;
  }

  /**
   * Calculate binomial confidence interval
   */
  private calculateBinomialCI(p: number, n: number): [number, number] {
    if (n === 0) return [0, 0];
    
    // Wilson score interval (more accurate than normal approximation)
    const z = 1.96; // 95% confidence
    const denominator = 1 + z * z / n;
    const centre = p + z * z / (2 * n);
    const adjustment = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n));
    
    const lower = (centre - adjustment) / denominator;
    const upper = (centre + adjustment) / denominator;
    
    return [Math.max(0, lower), Math.min(1, upper)];
  }

  /**
   * Calculate calibration regression (slope and intercept)
   */
  private calculateCalibrationRegression(data: Array<{ predicted: number; actual: number }>): { slope: number; intercept: number } {
    if (data.length < 2) return { slope: 1, intercept: 0 };

    const n = data.length;
    const sumX = data.reduce((sum, d) => sum + d.predicted, 0);
    const sumY = data.reduce((sum, d) => sum + d.actual, 0);
    const sumXY = data.reduce((sum, d) => sum + d.predicted * d.actual, 0);
    const sumXX = data.reduce((sum, d) => sum + d.predicted * d.predicted, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    return { slope: isFinite(slope) ? slope : 1, intercept: isFinite(intercept) ? intercept : 0 };
  }

  /**
   * Calculate Brier score
   */
  private calculateBrierScore(data: Array<{ predicted: number; actual: number }>): number {
    if (data.length === 0) return 0;

    const score = data.reduce((sum, d) => {
      const diff = d.predicted - d.actual;
      return sum + diff * diff;
    }, 0);

    return score / data.length;
  }

  /**
   * Helper functions
   */
  private groupResultsByQueryAndSystem(results: ExecutionResult[]): Map<string, ExecutionResult[]> {
    const grouped = new Map<string, ExecutionResult[]>();
    
    for (const result of results) {
      const key = `${result.query_id}|${result.system_id}`;
      if (!grouped.has(key)) {
        grouped.set(key, []);
      }
      grouped.get(key)!.push(result);
    }
    
    return grouped;
  }

  private percentile(sortedArray: number[], p: number): number {
    if (sortedArray.length === 0) return 0;
    
    const index = (p / 100) * (sortedArray.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    
    if (lower === upper) {
      return sortedArray[lower];
    }
    
    const weight = index - lower;
    return sortedArray[lower] * (1 - weight) + sortedArray[upper] * weight;
  }

  private isHitRelevant(hit: any, qrels: QrelJudgment[]): boolean {
    for (const qrel of qrels) {
      if (hit.file === qrel.file && hit.line === qrel.line && hit.column === qrel.column) {
        return qrel.relevance > 0;
      }
    }
    return false;
  }

  private calculateCoreAt10(hits: any[]): number {
    // Core@10 measures centrality - how much results cluster around core topics
    // This is a simplified version - full implementation would need topic modeling
    if (hits.length === 0) return 0;
    
    const fileCounts = new Map<string, number>();
    for (const hit of hits) {
      fileCounts.set(hit.file, (fileCounts.get(hit.file) || 0) + 1);
    }
    
    const maxCount = Math.max(...fileCounts.values());
    return maxCount / hits.length;
  }

  private calculateSpanCoverage(hits: any[]): number {
    // Span coverage must be 100% for valid results
    // This validates that all hits have proper span information
    let validSpans = 0;
    
    for (const hit of hits) {
      if (hit.file && hit.line && hit.snippet) {
        validSpans++;
      }
    }
    
    return hits.length > 0 ? validSpans / hits.length : 0;
  }

  private calculateSymbolCoverage(hits: any[]): number {
    // Calculate coverage of different symbol types
    const symbolTypes = new Set<string>();
    
    for (const hit of hits) {
      if (hit.symbol_kind) {
        symbolTypes.add(hit.symbol_kind);
      }
    }
    
    // Normalize by expected symbol type diversity (rough estimate)
    const expectedTypes = 5; // function, class, variable, import, etc.
    return Math.min(symbolTypes.size / expectedTypes, 1.0);
  }

  // Empty metric defaults
  private getEmptyQualityMetrics(): QualityMetrics {
    return {
      ndcg_at_10: 0,
      success_at_10: 0,
      sla_recall_at_50: 0,
      witness_coverage_at_10: 0,
      precision_at_10: 0,
      recall_at_10: 0,
      map_at_10: 0
    };
  }

  private getEmptyOperationalMetrics(): OperationalMetrics {
    return {
      p50_latency_ms: 0,
      p95_latency_ms: 0,
      p99_latency_ms: 0,
      p99_over_p95_ratio: 0,
      qps_at_150ms: 0,
      timeout_percentage: 0,
      error_rate: 0,
      sla_compliance_rate: 0,
      avg_concurrent_queries: 0
    };
  }

  private getEmptyCalibrationMetrics(): CalibrationMetrics {
    return {
      expected_calibration_error: 0,
      calibration_slope: 1,
      calibration_intercept: 0,
      reliability_buckets: [],
      max_calibration_error: 0,
      brier_score: 0
    };
  }

  private getEmptyExplainabilityMetrics(): ExplainabilityMetrics {
    return {
      why_mix_exact: 0,
      why_mix_struct: 0,
      why_mix_semantic: 0,
      why_mix_mixed: 0,
      core_at_10: 0,
      diversity_at_10: 0,
      span_coverage: 0,
      unique_files_at_10: 0,
      symbol_coverage: 0
    };
  }
}