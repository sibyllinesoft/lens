/**
 * efSearch Parameter Sweep Optimizer
 * 
 * Implements small parameter sweep to tighten nDCG without tail latency creep:
 * - Systematic exploration of efSearch values
 * - Real-time monitoring of nDCG@10 and tail latency
 * - Safe parameter bounds to prevent performance degradation
 * - Automated selection of optimal efSearch value
 */

import { writeFileSync, readFileSync } from 'fs';
import { join } from 'path';

export interface EfSearchConfig {
  current_value: number;
  min_value: number;
  max_value: number;
  step_size: number;
  max_tail_latency_p99: number;  // Constraint: no tail latency creep
  min_ndcg_improvement: number;  // Minimum nDCG@10 improvement required
}

export interface SweepResult {
  ef_search_value: number;
  ndcg_at_10: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  recall_at_50: number;
  throughput_qps: number;
  memory_usage_mb: number;
  cpu_utilization: number;
  is_tail_latency_violation: boolean;
  improvement_over_baseline: number;
}

export interface SweepAnalysis {
  baseline_result: SweepResult;
  sweep_results: SweepResult[];
  optimal_ef_search: number;
  optimal_result: SweepResult;
  improvement_summary: ImprovementSummary;
  safety_analysis: SafetyAnalysis;
  recommendation: OptimizationRecommendation;
}

interface ImprovementSummary {
  ndcg_improvement: number;
  recall_improvement: number; 
  latency_impact: number;
  throughput_impact: number;
  memory_impact: number;
}

interface SafetyAnalysis {
  tail_latency_safe: boolean;
  memory_usage_safe: boolean;
  throughput_degradation: boolean;
  overall_safe: boolean;
  risk_factors: string[];
}

interface OptimizationRecommendation {
  action: 'apply' | 'reject' | 'further_testing';
  recommended_value: number;
  confidence: number;
  reasoning: string;
  rollback_plan: string;
}

export class EfSearchOptimizer {
  private config: EfSearchConfig;
  private resultsDir: string;
  
  constructor() {
    this.resultsDir = './optimization-artifacts/efsearch';
    this.config = {
      current_value: 128,        // Current efSearch value
      min_value: 64,             // Conservative minimum
      max_value: 256,            // Conservative maximum to prevent tail latency
      step_size: 16,             // Small steps for precision
      max_tail_latency_p99: 300, // Hard limit: p99 < 300ms
      min_ndcg_improvement: 0.005 // Minimum 0.5pp nDCG improvement
    };
  }
  
  /**
   * Execute complete efSearch parameter sweep
   */
  public async performSweep(): Promise<SweepAnalysis> {
    console.log('🔍 Starting efSearch parameter sweep optimization');
    console.log(`📊 Range: ${this.config.min_value} to ${this.config.max_value} (step: ${this.config.step_size})`);
    console.log(`⚠️  Constraints: p99 < ${this.config.max_tail_latency_p99}ms, nDCG improvement > ${this.config.min_ndcg_improvement}`);
    
    try {
      // Step 1: Establish baseline with current efSearch
      const baseline = await this.benchmarkEfSearch(this.config.current_value);
      console.log(`📐 Baseline (ef=${this.config.current_value}): nDCG=${baseline.ndcg_at_10.toFixed(3)}, p99=${baseline.p99_latency_ms.toFixed(0)}ms`);
      
      // Step 2: Systematic parameter sweep
      const sweepResults = await this.executeSweep(baseline);
      
      // Step 3: Analyze results and find optimal value
      const analysis = this.analyzeResults(baseline, sweepResults);
      
      // Step 4: Generate recommendation
      const recommendation = this.generateRecommendation(analysis);
      analysis.recommendation = recommendation;
      
      // Step 5: Save results
      await this.saveAnalysis(analysis);
      
      console.log(`🎯 Optimal efSearch: ${analysis.optimal_ef_search} (nDCG improvement: +${(analysis.improvement_summary.ndcg_improvement*100).toFixed(2)}pp)`);
      console.log(`📈 Recommendation: ${recommendation.action} (confidence: ${(recommendation.confidence*100).toFixed(0)}%)`);
      
      return analysis;
      
    } catch (error) {
      console.error('❌ efSearch sweep failed:', error);
      throw error;
    }
  }
  
  /**
   * Execute systematic parameter sweep
   */
  private async executeSweep(baseline: SweepResult): Promise<SweepResult[]> {
    const results: SweepResult[] = [];
    const values = this.generateSweepValues();
    
    console.log(`🔄 Testing ${values.length} efSearch values: [${values.join(', ')}]`);
    
    for (const efSearchValue of values) {
      if (efSearchValue === this.config.current_value) {
        // Skip current value, we already have baseline
        continue;
      }
      
      console.log(`🧪 Testing efSearch=${efSearchValue}...`);
      
      try {
        const result = await this.benchmarkEfSearch(efSearchValue);
        results.push(result);
        
        // Early termination if tail latency violation
        if (result.is_tail_latency_violation) {
          console.log(`⚠️  Tail latency violation at ef=${efSearchValue} (p99=${result.p99_latency_ms}ms), skipping higher values`);
          break;
        }
        
        console.log(`   nDCG=${result.ndcg_at_10.toFixed(3)}, p99=${result.p99_latency_ms.toFixed(0)}ms, QPS=${result.throughput_qps.toFixed(1)}`);
        
      } catch (error) {
        console.error(`❌ Failed to benchmark ef=${efSearchValue}:`, error);
        // Continue with next value
      }
    }
    
    return results;
  }
  
  /**
   * Generate sweep values with smart sampling
   */
  private generateSweepValues(): number[] {
    const values = [];
    
    // Dense sampling around current value
    const currentIndex = Math.floor((this.config.current_value - this.config.min_value) / this.config.step_size);
    
    // Add values below current
    for (let val = this.config.current_value - this.config.step_size; val >= this.config.min_value; val -= this.config.step_size) {
      values.unshift(val);
    }
    
    // Add current value  
    values.push(this.config.current_value);
    
    // Add values above current
    for (let val = this.config.current_value + this.config.step_size; val <= this.config.max_value; val += this.config.step_size) {
      values.push(val);
    }
    
    return values.filter(v => v > 0); // Ensure positive values only
  }
  
  /**
   * Benchmark specific efSearch value
   */
  private async benchmarkEfSearch(efSearchValue: number): Promise<SweepResult> {
    // Mock benchmarking - in production would run actual HNSW queries
    
    // Simulate realistic performance characteristics:
    // - Higher efSearch generally improves nDCG but increases latency
    // - Diminishing returns at higher values
    // - Memory usage scales roughly linearly
    
    const baselineEf = 128;
    const efRatio = efSearchValue / baselineEf;
    
    // nDCG improvement with diminishing returns
    const ndcgMultiplier = 1 + (Math.log(efRatio) * 0.15); // Logarithmic improvement
    const ndcg = (0.779 * ndcgMultiplier) + (Math.random() * 0.005 - 0.0025); // Small variance
    
    // Latency increases roughly linearly with efSearch
    const latencyMultiplier = Math.pow(efRatio, 0.8); // Sublinear scaling
    const p95 = (87 * latencyMultiplier) + (Math.random() * 5 - 2.5);
    const p99 = (150 * latencyMultiplier) + (Math.random() * 10 - 5);
    
    // Throughput decreases with higher efSearch
    const throughputMultiplier = Math.pow(efRatio, -0.6);
    const throughput = (11.5 * throughputMultiplier) + (Math.random() * 0.5 - 0.25);
    
    // Memory scales approximately linearly
    const memoryMb = (250 * efRatio) + (Math.random() * 20 - 10);
    
    // CPU utilization increases with efSearch
    const cpuUtil = Math.min((45 * efRatio) + (Math.random() * 5 - 2.5), 90);
    
    // Recall generally improves with higher efSearch
    const recallMultiplier = 1 + (Math.log(efRatio) * 0.08);
    const recall = (0.889 * recallMultiplier) + (Math.random() * 0.003 - 0.0015);
    
    // Check for tail latency violation
    const isTailLatencyViolation = p99 > this.config.max_tail_latency_p99;
    
    // Calculate improvement over baseline (ef=128)
    const baselineNdcg = 0.779;
    const improvement = ndcg - baselineNdcg;
    
    return {
      ef_search_value: efSearchValue,
      ndcg_at_10: Math.max(ndcg, 0.1), // Ensure realistic bounds
      p95_latency_ms: Math.max(p95, 20),
      p99_latency_ms: Math.max(p99, p95 * 1.2), // Ensure p99 >= p95
      recall_at_50: Math.min(recall, 0.999),
      throughput_qps: Math.max(throughput, 1.0),
      memory_usage_mb: Math.max(memoryMb, 100),
      cpu_utilization: Math.max(Math.min(cpuUtil, 100), 10),
      is_tail_latency_violation: isTailLatencyViolation,
      improvement_over_baseline: improvement
    };
  }
  
  /**
   * Analyze sweep results to find optimal configuration
   */
  private analyzeResults(baseline: SweepResult, results: SweepResult[]): SweepAnalysis {
    // Find results that meet constraints
    const validResults = results.filter(r => 
      !r.is_tail_latency_violation && 
      r.improvement_over_baseline >= this.config.min_ndcg_improvement
    );
    
    if (validResults.length === 0) {
      console.log('⚠️  No improvements found within safety constraints');
      return {
        baseline_result: baseline,
        sweep_results: results,
        optimal_ef_search: this.config.current_value,
        optimal_result: baseline,
        improvement_summary: this.calculateImprovements(baseline, baseline),
        safety_analysis: this.analyzeSafety(baseline),
        recommendation: {
          action: 'reject',
          recommended_value: this.config.current_value,
          confidence: 1.0,
          reasoning: 'No safe improvements found within constraints',
          rollback_plan: 'Maintain current efSearch value'
        }
      };
    }
    
    // Find optimal result (best nDCG@10 among valid results)
    const optimal = validResults.reduce((best, current) => 
      current.ndcg_at_10 > best.ndcg_at_10 ? current : best
    );
    
    const improvements = this.calculateImprovements(baseline, optimal);
    const safetyAnalysis = this.analyzeSafety(optimal);
    
    return {
      baseline_result: baseline,
      sweep_results: results,
      optimal_ef_search: optimal.ef_search_value,
      optimal_result: optimal,
      improvement_summary: improvements,
      safety_analysis: safetyAnalysis,
      recommendation: {
        action: 'apply', // Will be refined in generateRecommendation
        recommended_value: optimal.ef_search_value,
        confidence: 0.8, // Will be refined
        reasoning: 'Preliminary analysis shows improvement',
        rollback_plan: 'Automated rollback to previous efSearch value'
      }
    };
  }
  
  /**
   * Calculate improvement metrics
   */
  private calculateImprovements(baseline: SweepResult, optimal: SweepResult): ImprovementSummary {
    return {
      ndcg_improvement: optimal.ndcg_at_10 - baseline.ndcg_at_10,
      recall_improvement: optimal.recall_at_50 - baseline.recall_at_50,
      latency_impact: optimal.p99_latency_ms - baseline.p99_latency_ms,
      throughput_impact: optimal.throughput_qps - baseline.throughput_qps,
      memory_impact: optimal.memory_usage_mb - baseline.memory_usage_mb
    };
  }
  
  /**
   * Analyze safety constraints
   */
  private analyzeSafety(result: SweepResult): SafetyAnalysis {
    const riskFactors = [];
    
    const tailLatencySafe = !result.is_tail_latency_violation;
    if (!tailLatencySafe) {
      riskFactors.push(`p99 latency ${result.p99_latency_ms.toFixed(0)}ms exceeds limit`);
    }
    
    const memoryUsageSafe = result.memory_usage_mb < 1000; // 1GB limit
    if (!memoryUsageSafe) {
      riskFactors.push(`Memory usage ${result.memory_usage_mb.toFixed(0)}MB approaching limits`);
    }
    
    const throughputDegradation = result.throughput_qps < 8.0; // Major throughput loss
    if (throughputDegradation) {
      riskFactors.push(`Throughput degraded to ${result.throughput_qps.toFixed(1)} QPS`);
    }
    
    const overallSafe = tailLatencySafe && memoryUsageSafe && !throughputDegradation;
    
    return {
      tail_latency_safe: tailLatencySafe,
      memory_usage_safe: memoryUsageSafe,
      throughput_degradation: throughputDegradation,
      overall_safe: overallSafe,
      risk_factors: riskFactors
    };
  }
  
  /**
   * Generate final optimization recommendation
   */
  private generateRecommendation(analysis: SweepAnalysis): OptimizationRecommendation {
    const improvements = analysis.improvement_summary;
    const safety = analysis.safety_analysis;
    
    // Decision logic
    if (!safety.overall_safe) {
      return {
        action: 'reject',
        recommended_value: this.config.current_value,
        confidence: 0.95,
        reasoning: `Safety constraints violated: ${safety.risk_factors.join(', ')}`,
        rollback_plan: 'Maintain current efSearch configuration'
      };
    }
    
    if (improvements.ndcg_improvement < this.config.min_ndcg_improvement) {
      return {
        action: 'reject',
        recommended_value: this.config.current_value,
        confidence: 0.9,
        reasoning: `nDCG improvement ${(improvements.ndcg_improvement*100).toFixed(2)}pp below minimum threshold`,
        rollback_plan: 'No change needed'
      };
    }
    
    // Check if improvement is significant enough
    if (improvements.ndcg_improvement >= 0.01 && improvements.latency_impact < 50) {
      return {
        action: 'apply',
        recommended_value: analysis.optimal_ef_search,
        confidence: 0.9,
        reasoning: `Strong nDCG improvement (+${(improvements.ndcg_improvement*100).toFixed(2)}pp) with acceptable latency impact (+${improvements.latency_impact.toFixed(0)}ms)`,
        rollback_plan: `Automated rollback to efSearch=${this.config.current_value} if quality gates fail`
      };
    }
    
    if (improvements.ndcg_improvement >= 0.005 && improvements.latency_impact < 25) {
      return {
        action: 'apply',
        recommended_value: analysis.optimal_ef_search,
        confidence: 0.75,
        reasoning: `Modest nDCG improvement (+${(improvements.ndcg_improvement*100).toFixed(2)}pp) with minimal latency impact`,
        rollback_plan: `Monitor for 48h, rollback if issues detected`
      };
    }
    
    return {
      action: 'further_testing',
      recommended_value: analysis.optimal_ef_search,
      confidence: 0.6,
      reasoning: `Marginal improvement detected, recommend A/B testing before full deployment`,
      rollback_plan: `Run 7-day A/B test with 10% traffic split`
    };
  }
  
  /**
   * Save analysis results
   */
  private async saveAnalysis(analysis: SweepAnalysis): Promise<void> {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const resultsFile = join(this.resultsDir, `efsearch-sweep-${timestamp}.json`);
    
    writeFileSync(resultsFile, JSON.stringify(analysis, null, 2));
    
    // Generate summary report
    const report = this.generateSummaryReport(analysis);
    const reportFile = join(this.resultsDir, `efsearch-summary-${timestamp}.md`);
    writeFileSync(reportFile, report);
    
    console.log(`📄 Analysis saved:`);
    console.log(`   JSON: ${resultsFile}`);
    console.log(`   Report: ${reportFile}`);
  }
  
  /**
   * Generate markdown summary report
   */
  private generateSummaryReport(analysis: SweepAnalysis): string {
    const baseline = analysis.baseline_result;
    const optimal = analysis.optimal_result;
    const improvements = analysis.improvement_summary;
    const recommendation = analysis.recommendation;
    
    return `# efSearch Parameter Optimization Results

**Timestamp**: ${new Date().toISOString()}
**Recommendation**: ${recommendation.action.toUpperCase()} 
**Confidence**: ${(recommendation.confidence * 100).toFixed(0)}%

## Summary

${recommendation.reasoning}

### Optimal Configuration
- **efSearch**: ${analysis.optimal_ef_search} (current: ${baseline.ef_search_value})
- **nDCG@10**: ${optimal.ndcg_at_10.toFixed(3)} (+${(improvements.ndcg_improvement*100).toFixed(2)}pp)
- **p99 Latency**: ${optimal.p99_latency_ms.toFixed(0)}ms (${improvements.latency_impact >= 0 ? '+' : ''}${improvements.latency_impact.toFixed(0)}ms)
- **Throughput**: ${optimal.throughput_qps.toFixed(1)} QPS (${improvements.throughput_impact >= 0 ? '+' : ''}${improvements.throughput_impact.toFixed(1)})

## Detailed Results

### Performance Comparison

| Metric | Baseline | Optimal | Change |
|--------|----------|---------|--------|
| efSearch | ${baseline.ef_search_value} | ${optimal.ef_search_value} | ${optimal.ef_search_value - baseline.ef_search_value} |
| nDCG@10 | ${baseline.ndcg_at_10.toFixed(3)} | ${optimal.ndcg_at_10.toFixed(3)} | +${(improvements.ndcg_improvement*100).toFixed(2)}pp |
| Recall@50 | ${baseline.recall_at_50.toFixed(3)} | ${optimal.recall_at_50.toFixed(3)} | ${improvements.recall_improvement >= 0 ? '+' : ''}${(improvements.recall_improvement*100).toFixed(2)}pp |
| p95 Latency | ${baseline.p95_latency_ms.toFixed(0)}ms | ${optimal.p95_latency_ms.toFixed(0)}ms | ${improvements.latency_impact >= 0 ? '+' : ''}${(optimal.p95_latency_ms - baseline.p95_latency_ms).toFixed(0)}ms |
| p99 Latency | ${baseline.p99_latency_ms.toFixed(0)}ms | ${optimal.p99_latency_ms.toFixed(0)}ms | ${improvements.latency_impact >= 0 ? '+' : ''}${improvements.latency_impact.toFixed(0)}ms |
| Throughput | ${baseline.throughput_qps.toFixed(1)} QPS | ${optimal.throughput_qps.toFixed(1)} QPS | ${improvements.throughput_impact >= 0 ? '+' : ''}${improvements.throughput_impact.toFixed(1)} |
| Memory | ${baseline.memory_usage_mb.toFixed(0)}MB | ${optimal.memory_usage_mb.toFixed(0)}MB | ${improvements.memory_impact >= 0 ? '+' : ''}${improvements.memory_impact.toFixed(0)}MB |

### Safety Analysis
- **Tail Latency Safe**: ${analysis.safety_analysis.tail_latency_safe ? '✅' : '❌'}
- **Memory Usage Safe**: ${analysis.safety_analysis.memory_usage_safe ? '✅' : '❌'}  
- **Throughput Impact**: ${analysis.safety_analysis.throughput_degradation ? '⚠️ Degraded' : '✅ Acceptable'}
- **Overall Safe**: ${analysis.safety_analysis.overall_safe ? '✅' : '❌'}

${analysis.safety_analysis.risk_factors.length > 0 ? `
**Risk Factors**:
${analysis.safety_analysis.risk_factors.map(risk => `- ${risk}`).join('\n')}
` : ''}

## Rollback Plan

${recommendation.rollback_plan}

## Next Steps

Based on the **${recommendation.action}** recommendation:

${recommendation.action === 'apply' ? 
`1. Deploy efSearch=${analysis.optimal_ef_search} to canary environment
2. Monitor quality gates for 24 hours  
3. Gradually roll out if gates pass
4. Implement automated rollback triggers` : 
recommendation.action === 'further_testing' ?
`1. Set up A/B testing infrastructure
2. Run controlled experiment with ${((analysis.optimal_ef_search - baseline.ef_search_value) / baseline.ef_search_value * 100).toFixed(0)}% traffic
3. Collect statistical significance data
4. Make final deployment decision` :
`1. Maintain current efSearch=${baseline.ef_search_value}
2. Investigate alternative optimization approaches
3. Consider broader system optimization`}
`;
  }
  
  /**
   * Apply recommended efSearch value  
   */
  public async applyOptimization(analysis: SweepAnalysis): Promise<boolean> {
    if (analysis.recommendation.action !== 'apply') {
      console.log('⚠️  Optimization not recommended for application');
      return false;
    }
    
    try {
      console.log(`🔧 Applying efSearch optimization: ${this.config.current_value} → ${analysis.optimal_ef_search}`);
      
      // Update configuration
      this.config.current_value = analysis.optimal_ef_search;
      
      // In production, this would update the actual HNSW configuration
      console.log(`✅ efSearch parameter updated to ${analysis.optimal_ef_search}`);
      
      return true;
      
    } catch (error) {
      console.error('❌ Failed to apply efSearch optimization:', error);
      return false;
    }
  }
}

export const efSearchOptimizer = new EfSearchOptimizer();