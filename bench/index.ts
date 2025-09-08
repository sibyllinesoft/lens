/**
 * Benchmark Protocol v1.0 - Main Index
 * 
 * Complete competitive evaluation framework with SLA-bounded fairness,
 * pooled qrels, gap mining, and publication-grade results.
 */

// Core Framework Components
export { PooledQrelsBuilder, PooledQrelsConfig, QrelJudgment, createQrelsBuilder } from './pooled-qrels-builder';
export { CompetitorAdapter, AdapterRegistry, AdapterConfig, SearchResponse, createAdapter } from './competitor-adapters';
export { SLAExecutionEngine, BatchExecutor, BenchmarkQuery, ExecutionResult } from './sla-execution-engine';

// Metrics & Analysis
export { 
  MetricsCalculator, 
  AggregateMetrics, 
  QualityMetrics, 
  OperationalMetrics, 
  CalibrationMetrics, 
  ExplainabilityMetrics 
} from './metrics-calculator';

// Data Export
export { ParquetExporter, BenchmarkParquetSchema, AggregateRow, DetailRow } from './parquet-exporter';

// CLI Interface
export { buildPool, warmup, run, score, mine, plot } from './cli';

/**
 * Complete Benchmark Protocol v1.0 Implementation
 * 
 * This framework provides:
 * 
 * 1. **Protocol Documentation** (`PROTOCOL.md`)
 *    - Complete specification with SLA enforcement
 *    - Hardware parity requirements
 *    - Pooled qrels methodology
 *    - Statistical rigor standards
 * 
 * 2. **Pooled Qrels System** (`pooled-qrels-builder.ts`)
 *    - Union of top-k across systems (in-SLA only)
 *    - Suite-specific builders (CoIR, SWE-bench, etc.)
 *    - Automated relevance judgment generation
 *    - Quality statistics and validation
 * 
 * 3. **Competitor Adapter Framework** (`competitor-adapters.ts`)
 *    - Unified interface for all systems
 *    - Lexical: ripgrep, Elasticsearch BM25
 *    - Hybrid: BM25+kNN with proximity tuning
 *    - LSP/Structural: Sourcegraph-class search
 *    - Target: Lens with frozen weights+calibration+policy
 * 
 * 4. **SLA-Bounded Execution Engine** (`sla-execution-engine.ts`)
 *    - Strict 150ms enforcement
 *    - Server-side latency counting
 *    - Client watchdog enforcement
 *    - Hardware parity validation
 *    - Resource monitoring
 * 
 * 5. **Comprehensive Metrics** (`metrics-calculator.ts`)
 *    - Quality: nDCG@10, Success@10, SLA-Recall@50, witness_coverage@10
 *    - Operations: p50/p95/p99, p99/p95 ratio, QPS@150ms, timeout%
 *    - Calibration: ECE, slope/intercept per intent×language
 *    - Explainability: why_mix, Core@10, Diversity@10, span coverage
 * 
 * 6. **Parquet Schema** (`parquet-exporter.ts`)
 *    - Aggregate table: One row per (query, system) combination
 *    - Detail table: One row per (query, system, hit) combination
 *    - Complete schema matching protocol specification
 *    - CSV export for analysis tools
 * 
 * 7. **Gap Mining System**
 *    - Automated slice delta analysis (intent×language)
 *    - Witness miss attribution for SWE-bench
 *    - Timeout attribution and stage-tax isolation
 *    - Calibration risk flagging (ECE > 0.02, slope outside [0.9,1.1])
 *    - Backlog CSV generation for PM/engineering triage
 * 
 * 8. **Publication Plots** (Auto-generated)
 *    - Hero bars: nDCG@10, SLA-Recall@50 with 95% CIs
 *    - Latency distributions: p50/p95/p99 per system
 *    - Reliability diagrams: ECE per intent×language
 *    - Slice heatmaps: ΔnDCG@10 gaps
 *    - Why-mix shift analysis
 *    - Witness coverage CDFs
 *    - SLA utility curves
 * 
 * 9. **Statistical Rigor**
 *    - Paired stratified bootstrap (B≥2000)
 *    - Paired permutation + Holm correction
 *    - Cohen's d effect sizes
 *    - Same methodology as paper
 * 
 * 10. **CLI Interface** (`cli.ts`)
 *     - Complete runbook commands
 *     - `bench build-pool`, `bench warmup`, `bench run`
 *     - `bench score`, `bench mine`, `bench plot`
 *     - Artifact packaging and validation
 */

/**
 * Quick Start Example
 */
export const QUICK_START_EXAMPLE = `
# Complete Benchmark Protocol v1.0 Workflow

## 1. Build pooled qrels (one-time setup)
npx bench build-pool \\
  --suites coir,swe_verified,csn,cosqa \\
  --systems lens,bm25,bm25_prox,hybrid,sourcegraph \\
  --sla 150 \\
  --out pool/

## 2. Warmup and attestation
npx bench warmup \\
  --systems lens,bm25,bm25_prox,hybrid,sourcegraph \\
  --hardware-check strict \\
  --attest attest.json

## 3. Execute benchmark suites
npx bench run --suite coir --systems lens,bm25,bm25_prox,hybrid,sourcegraph --sla 150 --out runs/coir/
npx bench run --suite swe_verified --systems lens,bm25,bm25_prox,hybrid,sourcegraph --sla 150 --out runs/swe/

## 4. Score with statistical testing
npx bench score \\
  --runs runs/* \\
  --pool pool/ \\
  --bootstrap 2000 \\
  --permute \\
  --holm \\
  --out scored/

## 5. Mine gaps and weaknesses
npx bench mine \\
  --in scored/agg.parquet \\
  --out reports/gaps.csv

## 6. Generate publication plots
npx bench plot \\
  --in scored/ \\
  --figures hero,latency,calibration,gaps,witness,utility \\
  --out reports/figs/

## 7. Package for reproducibility
npx bench package \\
  --run-data scored/ \\
  --attestation attest.json \\
  --figures reports/figs/ \\
  --out artifacts/benchmark_v1_\$(date +%Y%m%d).tar.gz
`;

/**
 * System Integration Example
 */
export const INTEGRATION_EXAMPLE = `
import { 
  PooledQrelsBuilder, 
  AdapterRegistry,
  SLAExecutionEngine,
  MetricsCalculator,
  ParquetExporter
} from './bench';

async function runBenchmark() {
  // 1. Build pooled qrels
  const qrelsBuilder = new PooledQrelsBuilder({
    suites: ['coir', 'swe_verified'],
    systems: ['lens', 'bm25', 'hybrid'],
    sla_ms: 150,
    top_k: 50,
    min_agreement: 2,
    output_dir: 'pool'
  });
  await qrelsBuilder.buildPooledQrels();

  // 2. Setup systems
  const registry = new AdapterRegistry();
  await registry.registerAdapter('lens', { system_id: 'lens', corpus_path: './corpus' });
  await registry.registerAdapter('bm25', { system_id: 'bm25', corpus_path: './corpus' });

  // 3. Execute with SLA enforcement
  const engine = new SLAExecutionEngine({ sla_ms: 150 });
  await engine.initialize();

  const queries = [
    { query_id: 'q1', query_text: 'function definition', suite: 'coir', intent: 'structural', language: 'typescript' }
  ];

  const results = [];
  for (const query of queries) {
    for (const [systemId, adapter] of registry.adapters) {
      const result = await engine.executeQuery(adapter, query);
      results.push(result);
    }
  }

  // 4. Calculate metrics
  const calculator = new MetricsCalculator();
  const qrels = JSON.parse(await fs.readFile('pool/coir_pooled_qrels.json', 'utf8'));
  calculator.loadQrels(qrels);
  const metrics = calculator.calculateMetrics(results);

  // 5. Export to Parquet
  const exporter = new ParquetExporter('output');
  await exporter.exportAggregateMetrics(metrics);
  await exporter.exportDetailResults(results);

  await registry.teardownAll();
}
`;

/**
 * Version and compatibility information
 */
export const VERSION_INFO = {
  protocol_version: '1.0',
  implementation_version: '1.0.0',
  node_version_required: '>=18.0.0',
  dependencies: {
    'commander': '^9.0.0',
    'node-fetch': '^3.0.0'
  },
  optional_dependencies: {
    'apache-arrow': '^12.0.0', // For native Parquet support
    'parquet-wasm': '^0.5.0',   // For WebAssembly Parquet
    'd3': '^7.0.0',             // For plot generation
    'plotly.js': '^2.0.0'       // Alternative plotting
  }
};

/**
 * Configuration defaults
 */
export const DEFAULT_CONFIG = {
  sla_ms: 150,
  watchdog_buffer_ms: 20,
  hardware_validation: true,
  resource_monitoring: true,
  timeout_retries: 1,
  top_k: 50,
  min_agreement: 2,
  bootstrap_iterations: 2000,
  confidence_level: 0.95,
  effect_size_threshold: 0.2, // Cohen's d
  calibration_bins: 10,
  ece_threshold: 0.02,
  calibration_slope_range: [0.9, 1.1]
};

/**
 * Supported test suites
 */
export const SUPPORTED_SUITES = {
  coir: {
    name: 'CoIR Aggregate',
    description: 'Multi-language code information retrieval (UR-Broad)',
    languages: ['python', 'typescript', 'javascript', 'java', 'go', 'rust'],
    query_types: ['semantic', 'identifier', 'structural']
  },
  swe_verified: {
    name: 'SWE-bench Verified',
    description: 'Task-level success with witness coverage',
    languages: ['python'],
    query_types: ['task-level'],
    special_metrics: ['witness_coverage', 'success_at_k']
  },
  csn: {
    name: 'CodeSearchNet',
    description: 'Legacy comparability benchmark',
    languages: ['python', 'javascript', 'java', 'go', 'ruby', 'php'],
    query_types: ['natural_language']
  },
  cosqa: {
    name: 'CoSQA',
    description: 'Code question answering (note: label noise)',
    languages: ['python'],
    query_types: ['question_answering']
  },
  cp_regex: {
    name: 'CP-Regex',
    description: 'Exact/regex parity validation',
    languages: ['all'],
    query_types: ['exact', 'regex']
  }
};

/**
 * Hardware requirements
 */
export const HARDWARE_REQUIREMENTS = {
  minimum: {
    cpu_cores: 4,
    memory_gb: 8,
    storage_gb: 50,
    network: 'gigabit'
  },
  recommended: {
    cpu_cores: 8,
    memory_gb: 16,
    storage_gb: 100,
    network: 'gigabit',
    notes: 'SSD storage recommended for corpus access'
  }
};

console.log('📋 Benchmark Protocol v1.0 Framework Loaded');
console.log(`🔬 Protocol Version: ${VERSION_INFO.protocol_version}`);
console.log(`⚡ Implementation Version: ${VERSION_INFO.implementation_version}`);
console.log(`🎯 Default SLA: ${DEFAULT_CONFIG.sla_ms}ms`);
console.log(`📊 Supported Suites: ${Object.keys(SUPPORTED_SUITES).join(', ')}`);
console.log(`\n💡 Quick Start:\n${QUICK_START_EXAMPLE}`);