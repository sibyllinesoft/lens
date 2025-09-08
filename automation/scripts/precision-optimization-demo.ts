/**
 * Precision Optimization Pipeline Demo
 * 
 * Demonstrates the complete integration of:
 * - Pairwise LTR training with anchor+hard-negatives dataset
 * - Drift detection with CUSUM algorithms
 * - A/B experiment framework with promotion gates
 * - Real-time monitoring and alerting
 * - 100% span coverage validation
 */

import { PairwiseLTRTrainingPipeline, type LTRTrainingConfig } from './src/core/ltr-training-pipeline.js';
import { DriftDetectionSystem, defaultDriftDetectionConfig, type DriftMetrics } from './src/core/drift-detection-system.js';
import { PrecisionOptimizationEngine, PrecisionExperimentFramework } from './src/core/precision-optimization.js';
import type { SearchHit, SearchContext } from './src/types/core.js';
import type { ExperimentConfig } from './src/types/api.js';

async function demonstratePrecisionOptimization() {
  console.log('🚀 Precision Optimization Pipeline Demo');
  console.log('==========================================\n');

  // Step 1: Initialize LTR Training Pipeline
  console.log('📚 Step 1: Initialize LTR Training Pipeline');
  
  const ltrConfig: LTRTrainingConfig = {
    learning_rate: 0.01,
    regularization: 0.001,
    max_iterations: 150,
    convergence_threshold: 1e-6,
    validation_split: 0.2,
    isotonic_calibration: true,
    feature_normalization: true
  };

  const ltrPipeline = new PairwiseLTRTrainingPipeline(ltrConfig);
  console.log('   ✅ LTR pipeline initialized with isotonic calibration');

  // Step 2: Create training data from anchor+hard-negatives
  console.log('\n📊 Step 2: Generate Training Data from Anchor Dataset');
  
  const mockAnchorData = generateMockAnchorDataset();
  console.log(`   📝 Generated ${mockAnchorData.length} anchor queries`);

  let trainingExamples = 0;
  for (const anchorQuery of mockAnchorData) {
    const { positives, hardNegatives } = generateHardNegatives(anchorQuery);
    
    for (const positive of positives) {
      for (const negative of hardNegatives) {
        ltrPipeline.addTrainingExample(
          anchorQuery.query,
          positive,
          negative,
          createMockContext(anchorQuery.query),
          1.0
        );
        trainingExamples++;
      }
    }
  }
  
  console.log(`   ✅ Added ${trainingExamples} pairwise training examples`);

  // Step 3: Train LTR Model
  console.log('\n🎓 Step 3: Train LTR Model');
  
  const trainingResult = await ltrPipeline.trainModel();
  console.log(`   📈 Training completed in ${trainingResult.convergence_iterations} iterations`);
  console.log(`   📊 Final loss: ${trainingResult.final_loss.toFixed(6)}`);
  console.log(`   🎯 Validation accuracy: ${(trainingResult.validation_accuracy || 0).toFixed(3)}`);
  console.log('   🔧 Learned weights:');
  Object.entries(trainingResult.final_weights).forEach(([feature, weight]) => {
    console.log(`      ${feature}: ${weight.toFixed(4)}`);
  });

  // Step 4: Initialize Drift Detection System
  console.log('\n🔍 Step 4: Initialize Drift Detection System');
  
  const driftSystem = new DriftDetectionSystem(defaultDriftDetectionConfig);
  let alertsReceived: any[] = [];
  
  driftSystem.on('drift_alert', (alert) => {
    alertsReceived.push(alert);
    const icon = alert.severity === 'critical' ? '🚨' : 
                alert.severity === 'error' ? '❌' : '⚠️';
    console.log(`   ${icon} DRIFT ALERT: ${alert.metric_name} - ${alert.severity}`);
  });

  console.log('   ✅ Drift detection initialized with CUSUM algorithms');

  // Step 5: Initialize Precision Optimization Engine
  console.log('\n⚙️ Step 5: Initialize Precision Optimization Engine');
  
  const optimizationEngine = new PrecisionOptimizationEngine();
  optimizationEngine.initializeLTRPipeline(ltrConfig);
  
  // Enable all optimization blocks
  optimizationEngine.setBlockEnabled('A', true);
  optimizationEngine.setBlockEnabled('B', true);
  optimizationEngine.setBlockEnabled('C', true);
  
  console.log('   ✅ Precision optimization engine ready with Blocks A, B, C enabled');

  // Step 6: Create A/B Experiment
  console.log('\n🧪 Step 6: Create A/B Experiment');
  
  const experimentFramework = new PrecisionExperimentFramework(optimizationEngine);
  
  const experimentConfig: ExperimentConfig = {
    experiment_id: 'ltr_precision_v1',
    name: 'LTR-Enhanced Precision Optimization',
    description: 'Pairwise LTR with isotonic calibration and drift monitoring',
    traffic_percentage: 10,
    control_config: {},
    treatment_config: {
      ltr_enabled: true,
      isotonic_calibration: true,
      blocks: ['A', 'B', 'C']
    },
    promotion_gates: {
      min_ndcg_improvement_pct: 2.0,
      min_recall_at_50: 0.88,
      min_span_coverage_pct: 99.0,
      max_latency_multiplier: 2.0
    },
    anchor_validation_required: true,
    ladder_validation_required: true
  };

  await experimentFramework.createExperiment(experimentConfig);
  console.log('   ✅ A/B experiment created with 10% traffic');

  // Step 7: Simulate Real-Time Operations
  console.log('\n⚡ Step 7: Simulate Real-Time Search Operations');
  
  const mockSearchHits = generateMockSearchResults();
  const mockContext = createMockContext('async function handler');

  // Test LTR reranking
  const originalHits = [...mockSearchHits];
  const rerankedHits = await ltrPipeline.rerank(mockSearchHits, mockContext);
  
  console.log(`   📊 Original vs LTR-reranked results:`);
  console.log(`      Original top hit: ${originalHits[0].file} (score: ${originalHits[0].score.toFixed(3)})`);
  console.log(`      LTR top hit: ${rerankedHits[0].file} (score: ${rerankedHits[0].score.toFixed(3)})`);

  // Apply precision optimization blocks
  let optimizedHits = await optimizationEngine.applyBlockA(rerankedHits, mockContext);
  optimizedHits = await optimizationEngine.applyBlockB(optimizedHits, mockContext);
  optimizedHits = await optimizationEngine.applyBlockC(optimizedHits, mockContext);
  
  console.log(`   🔧 Results after Block A,B,C optimization: ${rerankedHits.length} → ${optimizedHits.length} hits`);

  // Step 8: Validate Span Coverage
  console.log('\n📏 Step 8: Validate 100% Span Coverage');
  
  const spanCoverage = validateSpanCoverage(optimizedHits);
  console.log(`   📐 Span coverage: ${spanCoverage.coverage_pct.toFixed(1)}% (${spanCoverage.hits_with_spans}/${spanCoverage.total_hits})`);
  console.log(`   ✅ Span coverage validation: ${spanCoverage.coverage_pct >= 99.0 ? 'PASSED' : 'FAILED'}`);

  // Step 9: Record Metrics and Monitor Drift
  console.log('\n📊 Step 9: Record Metrics and Monitor Drift');
  
  // Simulate baseline metrics
  const baselineMetrics: DriftMetrics = {
    timestamp: new Date().toISOString(),
    anchor_p_at_1: 0.85,
    anchor_recall_at_50: 0.92,
    ladder_positives_ratio: 0.78,
    lsif_coverage_pct: 85.0,
    tree_sitter_coverage_pct: 92.0,
    sample_count: 100,
    query_complexity_distribution: {
      simple: 0.6,
      medium: 0.3,
      complex: 0.1
    }
  };

  await driftSystem.recordMetrics(baselineMetrics);
  console.log('   ✅ Baseline metrics recorded');

  // Simulate gradual drift
  console.log('   📉 Simulating gradual quality drift...');
  
  for (let i = 1; i <= 10; i++) {
    const driftedMetrics: DriftMetrics = {
      ...baselineMetrics,
      timestamp: new Date().toISOString(),
      anchor_p_at_1: baselineMetrics.anchor_p_at_1 - (i * 0.01), // Gradual degradation
      anchor_recall_at_50: baselineMetrics.anchor_recall_at_50 - (i * 0.005),
      sample_count: 100 + i * 10
    };
    
    await driftSystem.recordMetrics(driftedMetrics);
    
    if (i === 5) {
      console.log(`   📈 Iteration ${i}: P@1=${driftedMetrics.anchor_p_at_1.toFixed(3)}, Recall@50=${driftedMetrics.anchor_recall_at_50.toFixed(3)}`);
    }
  }

  console.log(`   🔔 Drift alerts triggered: ${alertsReceived.length}`);

  // Step 10: Run Experiment Validation
  console.log('\n🎯 Step 10: Run Experiment Validation');
  
  const anchorValidation = await experimentFramework.runAnchorValidation('ltr_precision_v1');
  console.log(`   ⚓ Anchor validation: ${anchorValidation.passed ? 'PASSED' : 'FAILED'}`);
  console.log(`      nDCG@10 Δ: +${anchorValidation.metrics.ndcg_at_10_delta_pct}%`);
  console.log(`      Recall@50: ${anchorValidation.metrics.recall_at_50}`);
  console.log(`      Span coverage: ${anchorValidation.metrics.span_coverage_pct}%`);

  const ladderValidation = await experimentFramework.runLadderValidation('ltr_precision_v1');
  console.log(`   🪜 Ladder validation: ${ladderValidation.passed ? 'PASSED' : 'FAILED'}`);

  // Check promotion readiness
  const promotionReadiness = await experimentFramework.checkPromotionReadiness('ltr_precision_v1');
  console.log(`   🚀 Ready for promotion: ${promotionReadiness.ready ? 'YES' : 'NO'}`);

  // Step 11: Generate System Health Report
  console.log('\n🏥 Step 11: Generate System Health Report');
  
  const driftReport = driftSystem.getDriftReport();
  console.log(`   🔍 System health: ${driftReport.system_health.toUpperCase()}`);
  console.log(`   📊 Active alerts: ${driftReport.active_alerts.length}`);
  console.log(`   📈 Recent metrics:`);
  console.log(`      Anchor P@1: ${driftReport.metrics_summary.recent_anchor_p1.toFixed(3)}`);
  console.log(`      Anchor Recall@50: ${driftReport.metrics_summary.recent_anchor_recall.toFixed(3)}`);
  console.log(`      LSIF coverage: ${driftReport.metrics_summary.recent_lsif_coverage.toFixed(1)}%`);

  if (driftReport.recommendations.length > 0) {
    console.log('   💡 System recommendations:');
    driftReport.recommendations.slice(0, 3).forEach(rec => {
      console.log(`      • ${rec}`);
    });
  }

  // Final Summary
  console.log('\n🏆 DEMO SUMMARY');
  console.log('================');
  console.log(`✅ LTR model trained with ${trainingResult.validation_accuracy?.toFixed(1)}% accuracy`);
  console.log(`✅ Drift detection active with ${alertsReceived.length} alerts triggered`);
  console.log(`✅ Precision optimization applied (${optimizedHits.length} final results)`);
  console.log(`✅ Span coverage maintained at ${spanCoverage.coverage_pct.toFixed(1)}%`);
  console.log(`✅ A/B experiment ${promotionReadiness.ready ? 'ready for promotion' : 'requires validation'}`);
  console.log(`✅ System health: ${driftReport.system_health.toUpperCase()}`);
  
  console.log('\n🎉 Precision optimization pipeline demo completed successfully!');
}

// Helper functions

function generateMockAnchorDataset() {
  return [
    {
      query: 'async function handler',
      expected_hits: []
    },
    {
      query: 'authentication middleware',
      expected_hits: []
    },
    {
      query: 'error handling utility',
      expected_hits: []
    },
    {
      query: 'database connection pool',
      expected_hits: []
    },
    {
      query: 'user validation schema',
      expected_hits: []
    }
  ];
}

function generateHardNegatives(anchorQuery: any) {
  const positives: SearchHit[] = [
    {
      file: 'src/core/handlers.ts',
      line: 15,
      col: 0,
      snippet: 'async function handleRequest(req: Request)',
      score: 0.95,
      why: ['exact', 'symbol'],
      symbol_kind: 'function',
      pattern_type: 'async_def'
    },
    {
      file: 'src/api/routes.ts',
      line: 42,
      col: 4,
      snippet: 'export const asyncHandler = async (req, res)',
      score: 0.89,
      why: ['exact'],
      symbol_kind: 'function'
    }
  ];

  const hardNegatives: SearchHit[] = [
    {
      file: 'tests/handlers.test.ts',
      line: 10,
      col: 0,
      snippet: 'test("handler should work", async () =>',
      score: 0.65,
      why: ['fuzzy'],
      symbol_kind: 'function'
    },
    {
      file: 'node_modules/@types/express/index.d.ts',
      line: 234,
      col: 8,
      snippet: 'interface Handler { (req: Request): void }',
      score: 0.45,
      why: ['subtoken'],
      symbol_kind: 'interface'
    }
  ];

  return { positives, hardNegatives };
}

function generateMockSearchResults(): SearchHit[] {
  return [
    {
      file: 'node_modules/lodash/index.js',
      line: 1000,
      col: 0,
      snippet: 'function asyncHandler(fn) { return (req, res, next) => { fn(req, res, next).catch(next); }; }',
      score: 0.72,
      why: ['fuzzy'],
      symbol_kind: 'function'
    },
    {
      file: 'src/core/async-handler.ts',
      line: 8,
      col: 0,
      snippet: 'export const asyncHandler = (fn: AsyncFunction) => (req: Request, res: Response, next: NextFunction)',
      score: 0.94,
      why: ['exact', 'symbol'],
      symbol_kind: 'function',
      ast_path: '/export/declaration/function',
      pattern_type: 'function_def'
    },
    {
      file: 'tests/async-handler.test.ts',
      line: 25,
      col: 4,
      snippet: 'describe("asyncHandler", () => { test("should catch async errors"',
      score: 0.68,
      why: ['exact'],
      symbol_kind: 'function'
    },
    {
      file: 'src/utils/helpers.ts',
      line: 156,
      col: 2,
      snippet: '// Async handler utility for express routes',
      score: 0.45,
      why: ['fuzzy'],
      span_len: 42
    }
  ];
}

function createMockContext(query: string): SearchContext {
  return {
    query,
    mode: 'hybrid',
    fuzzy: 1,
    k: 20,
    filters: {},
    timeout_ms: 1000,
    repo_sha: 'abc123def456',
    index_version: 'v1',
    api_version: 'v1'
  };
}

function validateSpanCoverage(hits: SearchHit[]) {
  const hitsWithSpans = hits.filter(hit => 
    hit.file && hit.line > 0 && hit.col >= 0
  );
  
  const hitsWithByteOffsets = hits.filter(hit => 
    hit.byte_offset !== undefined && hit.span_len !== undefined
  );

  return {
    total_hits: hits.length,
    hits_with_spans: hitsWithSpans.length,
    hits_with_byte_offsets: hitsWithByteOffsets.length,
    coverage_pct: (hitsWithSpans.length / hits.length) * 100,
    detailed_coverage: {
      file_paths: hits.length,
      line_numbers: hits.filter(h => h.line > 0).length,
      column_positions: hits.filter(h => h.col >= 0).length,
      byte_offsets: hitsWithByteOffsets.length,
      span_lengths: hits.filter(h => h.span_len !== undefined).length
    }
  };
}

// Run the demo
demonstratePrecisionOptimization().catch(console.error);