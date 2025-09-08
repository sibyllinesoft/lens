# Product Comparison Matrix System Usage Guide

## Overview

The Product Comparison Matrix system provides comprehensive, statistically rigorous comparison of code search systems using industry-standard benchmark datasets. It implements stratified analysis across multiple dimensions with fraud-resistant results and establishes performance baselines for iteration guidance.

## Key Features

### 🎯 **Comprehensive Dataset Coverage**
- **SWE-bench Verified**: 2,294 task-level evaluation queries
- **CoIR**: 15,678 code information retrieval queries  
- **CodeSearchNet**: 10,547 natural language to code queries
- **CoSQA**: 1,160 code question-answering queries
- **Total**: 29,679 benchmark queries across all datasets

### 📊 **Stratified Analysis Dimensions**
- **Query Type**: def, refs, symbol, generic, protocol, cross_lang, nl, structural
- **Difficulty Level**: easy, medium, hard
- **Programming Language**: Python, TypeScript, JavaScript, Java, Go, Rust, C++, C#
- **Dataset Source**: Task-level vs retrieval-level evaluation
- **Corpus Size**: Small, medium, large, enterprise

### 🔬 **Statistical Analysis Framework**
- **Meta-analysis** across stratified results with heterogeneity assessment
- **Bootstrap confidence intervals** with clustered resampling
- **Multiple testing correction** (Holm, Hochberg, Bonferroni)
- **Effect size calculation** (Cohen's d, Hedges' g, Cliff's delta)
- **Bayesian inference** for performance estimation
- **Power analysis** and sample size planning

### 🛡️ **Fraud-Resistant Infrastructure**
- **Cryptographic attestation** of all results
- **Versioned fingerprints** with configuration hashing
- **Reproducibility bundles** for audit trails
- **Data integrity verification** throughout pipeline

## Quick Start

### 1. Install Dependencies

```bash
npm install
# Ensure datasets are available at configured paths
```

### 2. Basic Lens vs Serena Comparison

```typescript
import { ComparisonMatrixCLI } from '../src/benchmark/comparison-matrix-orchestrator.js';

// Execute focused Lens vs Serena comparison
await ComparisonMatrixCLI.executeFromConfig('./examples/lens-serena-config.json');
```

### 3. Establish Performance Baseline

```typescript
import { ComparisonMatrixCLI } from '../src/benchmark/comparison-matrix-orchestrator.js';

const baselineMetrics = await ComparisonMatrixCLI.establishBaseline(
  './baseline-results',
  {
    swe_bench_path: './datasets/swe-bench-verified.json',
    coir_path: './datasets/coir.json',
    use_cached: true
  }
);

console.log(`Baseline established: ${baselineMetrics.baseline_id}`);
console.log(`Quality score: ${(baselineMetrics.baseline_quality.stability_score * 100).toFixed(1)}%`);
```

## Configuration Examples

### Smoke Test Configuration (Fast)
```json
{
  "comparison_scope": "smoke",
  "max_queries_per_stratum": 20,
  "statistical_config": {
    "bootstrap_samples": 1000,
    "power_requirement": 0.7
  },
  "quality_gates": {
    "min_sample_size": 15,
    "max_execution_time_hours": 1
  }
}
```

### Production Comparison Configuration (Comprehensive)
```json
{
  "comparison_scope": "full", 
  "max_queries_per_stratum": 200,
  "statistical_config": {
    "bootstrap_samples": 5000,
    "confidence_level": 0.95,
    "power_requirement": 0.8
  },
  "quality_gates": {
    "min_sample_size": 50,
    "max_execution_time_hours": 12
  }
}
```

### Lens vs Serena Focused Configuration
```json
{
  "comparison_scope": "focused",
  "serena_config": {
    "executable_path": "/usr/local/bin/serena-lsp",
    "workspace_path": "/tmp/comparison-workspace"
  },
  "stratification_method": "balanced",
  "statistical_config": {
    "mde_threshold": 0.01
  }
}
```

## Usage Patterns

### 1. Comprehensive Industry Comparison

```typescript
import { 
  ProductComparisonOrchestrator,
  ComparisonMatrixConfigFactory 
} from '../src/benchmark/product-comparison-matrix.js';

// Use factory for standard industry comparison
const { datasets, systems, stratificationDimensions } = 
  ComparisonMatrixConfigFactory.createIndustryBenchmarkConfig();

const orchestrator = new ProductComparisonOrchestrator({
  output_directory: './results',
  baseline_mode: 'establish',
  datasets: {
    swe_bench_path: './datasets/swe-bench-verified.json',
    coir_path: './datasets/coir.json',
    codesearchnet_path: './datasets/codesearchnet.jsonl',
    cosqa_path: './datasets/cosqa.json'
  },
  lens_config: {
    api_base_url: 'http://localhost:3000',
    enable_lsp: true
  },
  comparison_scope: 'full',
  statistical_config: {
    bootstrap_samples: 5000,
    confidence_level: 0.95,
    mde_threshold: 0.02,
    power_requirement: 0.8
  }
});

const result = await orchestrator.executeComparison();
```

### 2. Statistical Analysis Deep Dive

```typescript
import { 
  ComparisonStatisticsEngine,
  StatisticsConfigFactory 
} from '../src/benchmark/comparison-statistics.js';

const statsEngine = new ComparisonStatisticsEngine(
  ...StatisticsConfigFactory.createStandardConfig()
);

// Perform meta-analysis across strata
const metaResult = statsEngine.performMetaAnalysis(
  stratifiedResults,
  'ndcg_at_10',
  'lens',
  'serena-lsp'
);

// Calculate effect sizes with interpretation
const effectSize = statsEngine.calculateEffectSize(
  lensMetrics.mean, lensMetrics.std, lensMetrics.sample_size,
  serenaMetrics.mean, serenaMetrics.std, serenaMetrics.sample_size,
  'cohens_d'
);

// Assess practical significance
const practicalSignificance = statsEngine.assessPracticalSignificance(
  effectSize.estimate,
  [effectSize.ci_lower, effectSize.ci_upper],
  0.02  // 2% minimal important difference
);

// Generate comprehensive interpretation
const interpretation = statsEngine.generateStatisticalInterpretation(
  metaResult,
  practicalSignificance,
  'ndcg_at_10',
  'lens',
  'serena-lsp'
);

console.log(interpretation.executive_summary);
console.log(interpretation.recommendations);
```

### 3. Performance Baseline Establishment

```typescript
// Establish comprehensive baseline for iteration guidance
const comparisonResult = await orchestrator.executeComparison();

// Extract baseline insights
const baseline = comparisonResult.baselineMetrics;

console.log('🎯 Performance Baseline Established');
console.log(`Best System: ${baseline.baseline_metrics.current_best_system}`);

console.log('\n📈 Performance Gaps:');
for (const [system, gap] of Object.entries(baseline.baseline_metrics.performance_gaps)) {
  const gapPercent = (gap * 100).toFixed(1);
  console.log(`  ${system}: ${gap > 0 ? '+' : ''}${gapPercent}%`);
}

console.log('\n🎯 Iteration Priorities:');
for (const priority of baseline.iteration_priorities.slice(0, 3)) {
  console.log(`  ${priority.priority_rank}. ${priority.focus_area}`);
  console.log(`     Target: ${(priority.target_improvement * 100).toFixed(1)}% improvement`);
  console.log(`     Effort: ${priority.estimated_effort}`);
  console.log(`     Criteria: ${priority.success_criteria.join(', ')}`);
}
```

## Output Artifacts

### Generated Files Structure
```
comparison-results/
├── raw-results-cmp-{id}.json           # Complete comparison results
├── baseline-{baseline-id}.json         # Performance baseline metrics  
├── comparison-report-{id}.md           # Human-readable summary report
├── governance-state-{hash}.json        # Governance validation state
├── repro-bundle-{id}.tar.gz           # Reproducibility bundle
└── artifacts/
    ├── stratified-analysis.json        # Detailed stratum results
    ├── meta-analysis-results.json      # Cross-stratum meta-analysis
    ├── statistical-diagnostics.json   # Power, heterogeneity, bias tests
    └── fraud-resistance-attestation.json # Cryptographic verification
```

### Sample Report Output

```markdown
# Product Comparison Matrix Report

## Executive Summary

**Systems Compared:** Lens v1.0.0-rc.2, Serena LSP v0.8.1, ripgrep v14.1.0  
**Total Queries:** 29,679 across 4 industry datasets  
**Strata Analyzed:** 24 stratified dimensions  

## System Rankings

1. **Lens** (Score: 0.847) - Best overall performance across multiple dimensions
2. **Serena LSP** (Score: 0.783) - Strong symbol resolution, weaker semantic search  
3. **ripgrep** (Score: 0.621) - Excellent exact match, limited advanced features

## Key Findings

- **Lens closes the 32.8% LSP gap** achieving competitive performance with Serena
- **Semantic search advantages** for natural language queries (+15.3% vs Serena)
- **Cross-language search** shows significant improvement potential (+8.7% target)
- **Performance heterogeneity** moderate across query types (I² = 45.2%)

## Iteration Priorities

1. **LSP routing optimization** - Target: +5% nDCG@10 improvement
2. **Semantic accuracy** - Target: +3% success@10 improvement  
3. **Latency optimization** - Target: <150ms p95 latency

## Quality Gates ✅

- Statistical Power: 85% (target: 80%) ✅
- Baseline Stability: 92% (target: >95%) ✅  
- Coverage Completeness: 88% (target: >85%) ✅
```

## CLI Usage

### Command Line Interface

```bash
# Comprehensive comparison with full configuration
node -r esbuild-register examples/run-comparison.ts \
  --config ./examples/product-comparison-config.json \
  --output ./results

# Quick Lens vs Serena comparison
node -r esbuild-register examples/lens-serena-comparison.ts \
  --corpus-path ./indexed-content \
  --output ./lens-serena-results

# Establish baseline only
node -r esbuild-register examples/establish-baseline.ts \
  --datasets ./datasets \
  --output ./baseline-results
```

### Environment Variables

```bash
# Configure dataset paths
export SWE_BENCH_PATH=./datasets/swe-bench-verified.json
export COIR_PATH=./datasets/coir.json
export CODESEARCHNET_PATH=./datasets/codesearchnet.jsonl
export COSQA_PATH=./datasets/cosqa.json

# Configure system endpoints  
export LENS_API_URL=http://localhost:3000
export SERENA_LSP_PATH=/usr/local/bin/serena-lsp

# Configure execution parameters
export COMPARISON_SCOPE=smoke  # smoke, focused, full
export MAX_EXECUTION_TIME=4h
export BOOTSTRAP_SAMPLES=5000
```

## Integration with Existing Infrastructure

### Governance System Integration

```typescript
import { BenchmarkGovernanceSystem } from '../src/benchmark/governance-system.js';

// The comparison matrix integrates with existing governance
const governance = new BenchmarkGovernanceSystem('./output');

// Validate all governance requirements
const validation = await governance.validateGovernanceRequirements(
  fingerprint,
  benchmarkResults,
  sliceResults
);

if (!validation.overallPassed) {
  console.warn('Governance validation failed:', validation.recommendedActions);
}
```

### Enhanced Metrics Calculator Integration  

```typescript
import { EnhancedMetricsCalculator, EVALUATION_PROTOCOLS } from '../src/benchmark/enhanced-metrics-calculator.js';

const calculator = new EnhancedMetricsCalculator();

// Compute pooled qrels recall as per TODO.md specification
const recalls = calculator.computePooledQrelsRecall(
  allSystemResults,
  { systems: ['lens', 'serena'], top_k: 50, sla_constraint_ms: 150 }
);

// Generate ladder results across protocols (UR-Broad → UR-Narrow → CP-Regex)  
const ladderResults = calculator.generateLadderResults(systemResults, EVALUATION_PROTOCOLS);
```

## Advanced Usage

### Custom Stratification

```typescript
// Define custom stratification dimensions
const customDimensions = [
  {
    dimension_name: 'corpus_complexity',
    dimension_values: ['simple', 'moderate', 'complex'],
    min_sample_size: 30,
    stratified_sampling: true
  },
  {
    dimension_name: 'query_length', 
    dimension_values: ['short', 'medium', 'long'],
    min_sample_size: 25,
    stratified_sampling: true
  }
];

const matrix = new ProductComparisonMatrix(
  outputDir,
  datasets, 
  systems,
  customDimensions
);
```

### Bayesian Analysis

```typescript
// Perform Bayesian comparison of systems
const bayesianResult = await statsEngine.performBayesianAnalysis(
  lensResults,
  serenaResults,
  0.05,  // prior mean difference
  0.1    // prior standard deviation
);

console.log(`Posterior probability Lens > Serena: ${bayesianResult.probability_superior.toFixed(3)}`);
console.log(`Bayes Factor: ${bayesianResult.bayes_factor.toFixed(2)}`);
```

### Power Analysis

```typescript
// Calculate required sample sizes for future studies
const powerAnalysis = statsEngine.calculatePowerAnalysis(
  0.05,  // expected effect size
  0.05,  // alpha level  
  0.8,   // desired power
  1.0    // allocation ratio
);

console.log(`Required sample size per group: ${powerAnalysis.required_sample_size_per_group}`);
console.log(`Total sample size: ${powerAnalysis.total_sample_size}`);
```

## Troubleshooting

### Common Issues

**Dataset Loading Errors**
```bash
Error: SWE-bench dataset not found at ./datasets/swe-bench-verified.json
```
**Solution**: Verify dataset paths in configuration file and ensure files exist.

**Insufficient Sample Sizes**
```bash
Warning: Stratum type:def|diff:hard has insufficient sample size (15 < 30)
```
**Solution**: Reduce `min_sample_size` in quality gates or increase `max_queries_per_stratum`.

**System Connection Failures**
```bash
Error: Lens API health check failed: 500
```
**Solution**: Ensure Lens server is running and accessible at configured endpoint.

**Statistical Power Warnings**
```bash
Warning: Slice 'python_def_queries' is under-powered. Need 45 more samples (60% shortfall)
```
**Solution**: Increase data collection or lower power requirements in configuration.

### Performance Optimization

**Large Dataset Handling**
- Use `comparison_scope: "smoke"` for faster execution
- Implement `max_queries_per_stratum` limits
- Enable `use_cached: true` for dataset loading

**Memory Management**
- Process strata sequentially rather than in parallel
- Clear intermediate results after processing
- Use streaming for large result sets

**Execution Time Limits**
- Set appropriate `max_execution_time_hours` 
- Implement timeouts for individual system queries
- Use background execution for long-running comparisons

## Best Practices

### Statistical Rigor
- Always use appropriate multiple testing correction
- Report effect sizes alongside significance tests  
- Assess heterogeneity before interpreting meta-analysis results
- Use bootstrap confidence intervals for robust estimation

### Baseline Establishment
- Establish baselines before making system changes
- Track baseline stability across runs
- Document baseline assumptions and limitations
- Update baselines when datasets or systems change significantly

### Reproducibility
- Save complete configuration with each run
- Generate reproducibility bundles for important results
- Version datasets and system configurations
- Document any manual preprocessing steps

---

**For questions or support, refer to the detailed API documentation in the source code or create an issue in the project repository.**