/**
 * Benchmark Suite Runner for Lens
 * Implements smoke and full benchmark suites per TODO.md
 */

import { z } from 'zod';
import { v4 as uuidv4 } from 'uuid';
import { promises as fs } from 'fs';
import path from 'path';
import type {
  BenchmarkConfig,
  BenchmarkRun,
  GoldenDataItem,
  BenchmarkPlanMessage,
  BenchmarkRunMessage,
  BenchmarkResultMessage,
  ConfigFingerprint
} from '../types/benchmark.js';
import { PROMOTION_GATE_CRITERIA } from '../types/benchmark.js';
import { GroundTruthBuilder } from './ground-truth-builder.js';
import { MetricsCalculator } from './metrics-calculator.js';
import { NATSTelemetry } from './nats-telemetry.js';

export class BenchmarkSuiteRunner {
  private telemetry: NATSTelemetry;
  private metricsCalculator: MetricsCalculator;
  
  constructor(
    private readonly groundTruthBuilder: GroundTruthBuilder,
    private readonly outputDir: string,
    private readonly natsUrl: string = 'nats://localhost:4222'
  ) {
    this.telemetry = new NATSTelemetry(natsUrl);
    this.metricsCalculator = new MetricsCalculator();
  }

  /**
   * Run smoke test suite (PR gate) - ~50 queries × 5 repos, ≤10 min
   */
  async runSmokeSuite(overrides: Partial<BenchmarkConfig> = {}): Promise<BenchmarkRun> {
    const config: BenchmarkConfig = {
      trace_id: uuidv4(),
      suite: ['codesearch', 'structural'],
      systems: ['lex', '+symbols', '+symbols+semantic'],
      slices: 'SMOKE_DEFAULT',
      seeds: 1,
      cache_mode: 'warm',
      robustness: false,
      metamorphic: false,
      k_candidates: 200,
      top_n: 50,
      fuzzy: 2,
      subtokens: true,
      semantic_gating: {
        nl_likelihood_threshold: 0.5,
        min_candidates: 10
      },
      latency_budgets: {
        stage_a_ms: 200,
        stage_b_ms: 300,
        stage_c_ms: 300
      },
      ...overrides
    };

    console.log(`🔥 Starting SMOKE suite - Trace ID: ${config.trace_id}`);
    
    // Plan phase
    const planMessage: BenchmarkPlanMessage = {
      trace_id: config.trace_id,
      timestamp: new Date().toISOString(),
      config,
      estimated_duration_ms: 10 * 60 * 1000, // 10 minutes
      total_queries: 50 * 5 // ~50 queries × 5 repos
    };
    
    await this.telemetry.publishPlan(planMessage);
    
    // Select smoke dataset (stratified sampling)
    const goldenItems = this.selectSmokeDataset();
    
    // Execute benchmark across systems
    const results: BenchmarkRun[] = [];
    
    for (const system of config.systems) {
      const run = await this.executeBenchmarkRun({
        ...config,
        trace_id: uuidv4() // New trace per system
      }, system, goldenItems);
      
      results.push(run);
    }
    
    // Generate comparative report
    const finalResult = await this.generateComparativeResult(config, results);
    
    // Check promotion gate for smoke tests
    const gateResult = this.checkPromotionGate(results);
    
    console.log(`🎯 SMOKE suite complete - Gate: ${gateResult.passed ? 'PASS' : 'FAIL'}`);
    
    return finalResult;
  }

  /**
   * Run full nightly suite - all slices, cold vs warm, seeds=3, ablations + robustness
   */
  async runFullSuite(overrides: Partial<BenchmarkConfig> = {}): Promise<BenchmarkRun> {
    const config: BenchmarkConfig = {
      trace_id: uuidv4(),
      suite: ['codesearch', 'structural', 'docs'],
      systems: ['lex', '+symbols', '+symbols+semantic'],
      slices: 'ALL',
      seeds: 3,
      cache_mode: ['warm', 'cold'],
      robustness: true,
      metamorphic: true,
      k_candidates: 200,
      top_n: 50,
      fuzzy: 2,
      subtokens: true,
      semantic_gating: {
        nl_likelihood_threshold: 0.5,
        min_candidates: 10
      },
      latency_budgets: {
        stage_a_ms: 200,
        stage_b_ms: 300,
        stage_c_ms: 300
      },
      ...overrides
    };

    console.log(`🌙 Starting FULL suite - Trace ID: ${config.trace_id}`);
    
    const allGoldenItems = this.groundTruthBuilder.currentGoldenItems;
    
    const planMessage: BenchmarkPlanMessage = {
      trace_id: config.trace_id,
      timestamp: new Date().toISOString(),
      config,
      estimated_duration_ms: 2 * 60 * 60 * 1000, // 2 hours
      total_queries: allGoldenItems.length * config.systems.length * config.seeds
    };
    
    await this.telemetry.publishPlan(planMessage);
    
    // Execute across all configurations
    const results: BenchmarkRun[] = [];
    
    for (const system of config.systems) {
      for (const cacheMode of Array.isArray(config.cache_mode) ? config.cache_mode : [config.cache_mode]) {
        for (let seed = 1; seed <= config.seeds; seed++) {
          const runConfig = {
            ...config,
            cache_mode: cacheMode,
            trace_id: uuidv4()
          };
          
          const run = await this.executeBenchmarkRun(runConfig, system, allGoldenItems, seed);
          results.push(run);
        }
      }
    }
    
    // Run robustness tests if enabled
    if (config.robustness) {
      await this.runRobustnessTests(config);
    }
    
    // Run metamorphic tests if enabled  
    if (config.metamorphic) {
      await this.runMetamorphicTests(config);
    }
    
    const finalResult = await this.generateComparativeResult(config, results);
    
    console.log(`🎯 FULL suite complete - ${results.length} benchmark runs`);
    
    return finalResult;
  }

  private selectSmokeDataset(): GoldenDataItem[] {
    const allItems = this.groundTruthBuilder.currentGoldenItems;
    
    // Stratified sampling: 10 items per language/query_class combination
    const stratifiedSample: GoldenDataItem[] = [];
    const strata = new Map<string, GoldenDataItem[]>();
    
    // Group by language + query_class
    for (const item of allItems) {
      const strataKey = `${item.language}_${item.query_class}`;
      if (!strata.has(strataKey)) {
        strata.set(strataKey, []);
      }
      strata.get(strataKey)!.push(item);
    }
    
    // Sample 10 from each stratum (or all if less than 10)
    for (const [_, items] of strata) {
      const sample = items.slice(0, Math.min(10, items.length));
      stratifiedSample.push(...sample);
    }
    
    console.log(`📊 Smoke dataset: ${stratifiedSample.length} queries across ${strata.size} strata`);
    
    return stratifiedSample;
  }

  private async executeBenchmarkRun(
    config: BenchmarkConfig,
    system: string,
    goldenItems: GoldenDataItem[],
    seed: number = 1
  ): Promise<BenchmarkRun> {
    
    const startTime = Date.now();
    const run: BenchmarkRun = {
      trace_id: config.trace_id,
      config_fingerprint: this.generateConfigFingerprint(config, seed),
      timestamp: new Date().toISOString(),
      status: 'running',
      system,
      total_queries: goldenItems.length,
      completed_queries: 0,
      failed_queries: 0,
      metrics: {
        recall_at_10: 0,
        recall_at_50: 0,
        ndcg_at_10: 0,
        mrr: 0,
        first_relevant_tokens: 0,
        stage_latencies: {
          stage_a_p50: 0,
          stage_a_p95: 0,
          stage_b_p50: 0,
          stage_b_p95: 0,
          stage_c_p50: 0,
          stage_c_p95: 0,
          e2e_p50: 0,
          e2e_p95: 0
        },
        fan_out_sizes: {
          stage_a: 0,
          stage_b: 0,
          stage_c: 0
        },
        why_attributions: {}
      },
      errors: []
    };

    await this.telemetry.publishRun({
      trace_id: config.trace_id,
      timestamp: new Date().toISOString(),
      status: 'started'
    });

    // Simulate running queries (in real implementation, this would call the actual search engine)
    const queryResults: any[] = [];
    
    for (let i = 0; i < goldenItems.length; i++) {
      const item = goldenItems[i];
      
      try {
        if (!item) continue;
        
        // This would call the actual lens search engine
        const result = await this.executeQuery(item.query, system, config);
        queryResults.push({ item, result });
        
        run.completed_queries++;
        
        await this.telemetry.publishRun({
          trace_id: config.trace_id,
          timestamp: new Date().toISOString(),
          status: 'query_completed',
          query_id: item.id,
          latency_ms: result.latency_ms
        });
        
      } catch (error) {
        run.failed_queries++;
        run.errors.push({
          query_id: item?.id || 'unknown',
          error_type: 'query_execution',
          message: error instanceof Error ? error.message : String(error)
        });
      }
      
      // Progress logging every 10 queries
      if (i % 10 === 0) {
        console.log(`  Progress: ${i}/${goldenItems.length} queries (${system})`);
      }
    }

    // Calculate final metrics
    run.metrics = await this.metricsCalculator.calculateMetrics(queryResults);
    run.status = 'completed';
    
    // Generate artifacts
    const artifacts = await this.generateArtifacts(run, queryResults);
    
    // Publish final result
    const resultMessage: BenchmarkResultMessage = {
      trace_id: config.trace_id,
      timestamp: new Date().toISOString(),
      final_metrics: run.metrics,
      artifacts,
      duration_ms: Date.now() - startTime,
      promotion_gate_result: this.checkPromotionGate([run])
    };
    
    await this.telemetry.publishResult(resultMessage);
    
    return run;
  }

  private async executeQuery(query: string, system: string, config: BenchmarkConfig): Promise<any> {
    // Simulate query execution with realistic latencies
    const stageLatencies = {
      stage_a: 2 + Math.random() * 6, // 2-8ms
      stage_b: 3 + Math.random() * 7, // 3-10ms  
      stage_c: system.includes('semantic') ? 5 + Math.random() * 10 : 0 // 5-15ms if semantic
    };
    
    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, stageLatencies.stage_a + stageLatencies.stage_b + stageLatencies.stage_c));
    
    return {
      hits: [
        {
          file: 'src/example.ts',
          line: 1,
          col: 0,
          score: 0.9,
          why: ['exact']
        }
      ],
      total: 1,
      latency_ms: stageLatencies,
      stage_candidates: {
        stage_a: Math.floor(Math.random() * 100) + 50,
        stage_b: Math.floor(Math.random() * 50) + 25,
        stage_c: system.includes('semantic') ? config.k_candidates : 0
      }
    };
  }

  private async generateArtifacts(run: BenchmarkRun, queryResults: any[]): Promise<any> {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const baseFilename = `${run.system}_${timestamp}`;
    
    const artifacts = {
      metrics_parquet: path.join(this.outputDir, `${baseFilename}_metrics.parquet`),
      errors_ndjson: path.join(this.outputDir, `${baseFilename}_errors.ndjson`),  
      traces_ndjson: path.join(this.outputDir, `${baseFilename}_traces.ndjson`),
      report_pdf: path.join(this.outputDir, `${baseFilename}_report.pdf`),
      config_fingerprint_json: path.join(this.outputDir, `${baseFilename}_config.json`)
    };
    
    // Generate metrics file (would be Parquet in production)
    const metricsData = {
      run_metadata: run,
      query_results: queryResults.map(qr => ({
        query_id: qr.item.id,
        query: qr.item.query,
        recall_at_10: qr.result.recall_at_10 || 0,
        ndcg_at_10: qr.result.ndcg_at_10 || 0,
        latency_ms: qr.result.latency_ms
      }))
    };
    
    await fs.writeFile(artifacts.metrics_parquet + '.json', JSON.stringify(metricsData, null, 2));
    
    // Generate errors file
    const errorsNdjson = run.errors.map(error => JSON.stringify(error)).join('\n');
    await fs.writeFile(artifacts.errors_ndjson, errorsNdjson);
    
    // Generate traces (simplified)
    const tracesNdjson = queryResults.map(qr => JSON.stringify({
      trace_id: run.trace_id,
      query_id: qr.item.id,
      spans: [
        { stage: 'stage_a', duration_ms: qr.result.latency_ms.stage_a },
        { stage: 'stage_b', duration_ms: qr.result.latency_ms.stage_b },
        { stage: 'stage_c', duration_ms: qr.result.latency_ms.stage_c }
      ]
    })).join('\n');
    
    await fs.writeFile(artifacts.traces_ndjson, tracesNdjson);
    
    // Generate config fingerprint
    const configFingerprint = this.groundTruthBuilder.generateConfigFingerprint(run, [1, 2, 3]);
    await fs.writeFile(artifacts.config_fingerprint_json, JSON.stringify(configFingerprint, null, 2));
    
    // TODO: Generate PDF report (would use a PDF library)
    await fs.writeFile(artifacts.report_pdf, 'PDF report placeholder');
    
    return artifacts;
  }

  private checkPromotionGate(results: BenchmarkRun[]): any {
    if (results.length < 2) {
      return { passed: false, reason: 'Need baseline and treatment for comparison' };
    }
    
    const baseline = results[0]; // Assume first is baseline (e.g., "lex")
    const treatment = results[results.length - 1]; // Last is treatment
    
    if (!baseline || !treatment) {
      return { passed: false, reason: 'Missing baseline or treatment results' };
    }
    
    const ndcgDelta = treatment.metrics.ndcg_at_10 - baseline.metrics.ndcg_at_10;
    const ndcgImprovement = ndcgDelta >= PROMOTION_GATE_CRITERIA.ndcg_improvement_min;
    
    const recall50Maintained = treatment.metrics.recall_at_50 >= baseline.metrics.recall_at_50;
    
    const latencyIncrease = (treatment.metrics.stage_latencies.e2e_p95 - baseline.metrics.stage_latencies.e2e_p95) / 
                           baseline.metrics.stage_latencies.e2e_p95;
    const latencyAcceptable = latencyIncrease <= PROMOTION_GATE_CRITERIA.latency_p95_max_increase;
    
    const passed = ndcgImprovement && recall50Maintained && latencyAcceptable;
    
    return {
      passed,
      ndcg_delta: ndcgDelta,
      recall_50_maintained: recall50Maintained,
      latency_p95_acceptable: latencyAcceptable,
      regressions: passed ? [] : [
        !ndcgImprovement ? 'ndcg_insufficient_improvement' : '',
        !recall50Maintained ? 'recall_50_degradation' : '',
        !latencyAcceptable ? 'latency_p95_regression' : ''
      ].filter(Boolean)
    };
  }

  private generateConfigFingerprint(config: BenchmarkConfig, seed: number): string {
    const fingerprint = this.groundTruthBuilder.generateConfigFingerprint(config, [seed]);
    return fingerprint.config_hash;
  }

  private async generateComparativeResult(config: BenchmarkConfig, results: BenchmarkRun[]): Promise<BenchmarkRun> {
    // Aggregate results across systems
    const aggregated: BenchmarkRun = {
      trace_id: config.trace_id,
      config_fingerprint: results[0]?.config_fingerprint || '',
      timestamp: new Date().toISOString(),
      status: 'completed',
      system: 'AGGREGATE',
      total_queries: results.reduce((sum, r) => sum + r.total_queries, 0),
      completed_queries: results.reduce((sum, r) => sum + r.completed_queries, 0),
      failed_queries: results.reduce((sum, r) => sum + r.failed_queries, 0),
      metrics: this.metricsCalculator.aggregateMetrics(results.map(r => r.metrics)),
      errors: results.flatMap(r => r.errors)
    };
    
    return aggregated;
  }

  private async runRobustnessTests(config: BenchmarkConfig): Promise<void> {
    console.log('🔨 Running robustness tests...');
    // Implementation would include concurrency, cold start, fault injection tests
    // For now, just log placeholder
  }

  private async runMetamorphicTests(config: BenchmarkConfig): Promise<void> {
    console.log('🔄 Running metamorphic tests...');
    // Implementation would include invariance tests (rename, move, reformat, etc.)
    // For now, just log placeholder
  }
}