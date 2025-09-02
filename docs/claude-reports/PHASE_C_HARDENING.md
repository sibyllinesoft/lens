# Phase C - Benchmark Hardening

**"Keep the crank honest"** - Comprehensive quality assurance to ensure the benchmarks catch performance regressions and quality degradation.

## Overview

Phase C implements advanced benchmark hardening with:
- 📊 **Enhanced Visualization**: New plots for positives-in-candidates, precision-vs-score, latency analysis
- 🎯 **Hard Negative Testing**: Adversarial near-miss documents to stress-test ranking robustness  
- 🔍 **Per-Slice Gates**: Repository and language-specific performance validation
- ⚡ **Tripwire System**: Automated hard-fail conditions for quality assurance
- 📄 **PDF Reports**: Comprehensive reporting with embedded plots and analysis
- 🚨 **CI Integration**: Automated hardening checks in deployment pipeline

## Quick Start

### Basic Usage

```bash
# Run Phase C hardening with default settings
npx tsx src/benchmark/cli-phase-c.ts phase-c --mode nightly

# CI mode with strict gates
npx tsx src/benchmark/cli-phase-c.ts phase-c --mode pr --ci --fail-fast

# Full configuration
npx tsx src/benchmark/cli-phase-c.ts phase-c \
  --mode release \
  --ci \
  --output ./hardening-results \
  --min-score 90 \
  --max-degradation 5 \
  --slice-gates \
  --timeout 60
```

### Programmatic Usage

```typescript
import { 
  PhaseCHardening, 
  createDefaultHardeningConfig,
  CIHardeningOrchestrator,
  createDefaultCIConfig 
} from './src/benchmark/index.js';

// Basic hardening
const hardening = new PhaseCHardening('./output');
const config = createDefaultHardeningConfig(benchmarkConfig);
const report = await hardening.executeHardening(config, benchmarkResults, queryResults);

// CI integration
const ciOrchestrator = new CIHardeningOrchestrator('./output');
const ciConfig = createDefaultCIConfig('nightly');
const result = await ciOrchestrator.executeInCI(ciConfig, benchmarkConfig);
```

## Features

### 1. Enhanced Visualization System

Six new plot types provide deep insights into system behavior:

#### Positives-in-Candidates Analysis
- **Purpose**: Analyze relevant documents within candidate sets
- **Output**: `positives_in_candidates.png`
- **Insights**: Candidate quality and filtering effectiveness

#### Relevant-per-Query Histogram  
- **Purpose**: Distribution of relevant results across queries
- **Output**: `relevant_per_query_histogram.png`
- **Insights**: Query difficulty and result consistency

#### Precision vs Score (Pre/Post Calibration)
- **Purpose**: Score calibration effectiveness analysis
- **Output**: `precision_vs_score_*.png`
- **Insights**: Ranking quality and calibration impact

#### Latency Percentiles by Stage
- **Purpose**: Performance breakdown across pipeline stages
- **Output**: `latency_percentiles_by_stage.png`
- **Insights**: Bottleneck identification and SLA compliance

#### Early Termination Rate
- **Purpose**: Pipeline termination pattern analysis
- **Output**: `early_termination_rate.png`
- **Insights**: Efficiency and result completeness

### 2. Hard Negative Testing

Stress-test ranking robustness with adversarial documents:

```typescript
const hardeningConfig = {
  hard_negatives: {
    enabled: true,
    per_query_count: 5,           // 5 hard negatives per query
    shared_subtoken_min: 2        // Minimum shared subtokens
  }
};
```

**Generation Strategies**:
- **shared_class**: Files with similar class names but no gold spans
- **shared_method**: Files with similar method names but no matches
- **shared_variable**: Files with similar variable names
- **shared_imports**: Files with similar import patterns

**Impact Analysis**:
- Measures Recall@10 degradation under adversarial conditions
- Acceptable degradation: <15% for production systems
- Robust systems: <5% degradation

### 3. Per-Slice Performance Gates

Repository and language-specific validation:

```typescript
const sliceGates = {
  per_slice_gates: {
    enabled: true,
    min_recall_at_10: 0.70,      // 70% minimum recall
    min_ndcg_at_10: 0.60,        // 60% minimum nDCG
    max_p95_latency_ms: 500      // 500ms maximum P95
  }
};
```

**Slice Dimensions**:
- **By Repository**: storyviz, lens, core-utils, api-gateway
- **By Language**: typescript, python, rust, go, javascript
- **Combined**: repo|language pairs for fine-grained validation

### 4. Tripwire System

Automated hard-fail conditions prevent quality degradation:

#### Span Coverage Tripwire
- **Threshold**: <98% span coverage
- **Impact**: Reduced relevance quality
- **Action**: Review indexing pipeline and golden dataset alignment

#### Recall Convergence Tripwire  
- **Threshold**: Recall@50 ≈ Recall@10 (±0.5%)
- **Impact**: Poor ranking diversity
- **Action**: Improve candidate generation and ranking diversity

#### LSIF Coverage Drop Tripwire
- **Threshold**: -5% vs baseline
- **Impact**: Degraded symbol search quality
- **Action**: Investigate symbol extraction and LSIF generation

#### P99/P95 Ratio Tripwire
- **Threshold**: P99 > 2× P95
- **Impact**: Inconsistent tail latency
- **Action**: Identify and fix tail latency outliers

### 5. Comprehensive PDF Reports

Automated report generation with embedded analysis:

```typescript
const pdfConfig = {
  title: 'Lens Phase C Hardening Report',
  template: 'comprehensive',
  include_plots: true,
  include_raw_data: true,
  output_format: 'markdown'
};

const pdfReport = await pdfGenerator.generateHardeningReport(
  hardeningReport, 
  benchmarkResults, 
  pdfConfig
);
```

**Report Sections**:
- Executive Summary with key findings
- Tripwire Analysis with failure details
- Performance Analysis with stage breakdowns
- Hard Negative Testing impact assessment
- Per-Slice Analysis with language/repo insights
- Visualization Gallery with plot references
- Recommendations & Action Items
- Raw Data Appendix

## CI Integration

### GitHub Actions Workflow

The hardening system integrates with GitHub Actions for automated quality gates:

```yaml
- name: Run Phase C Hardening (PR Mode)
  run: |
    npx tsx src/benchmark/cli-phase-c.ts phase-c \
      --mode pr \
      --ci \
      --fail-fast \
      --min-score 65 \
      --max-degradation 20
```

### CI Modes

#### PR Mode
- **Duration**: 15 minutes max
- **Scope**: Lightweight validation for fast feedback
- **Gates**: 65+ hardening score, <20% degradation
- **Hard Negatives**: 3 per query (reduced for speed)

#### Nightly Mode  
- **Duration**: 45 minutes max
- **Scope**: Comprehensive validation
- **Gates**: 80+ hardening score, <10% degradation
- **Hard Negatives**: 5 per query (full testing)

#### Release Mode
- **Duration**: 60 minutes max
- **Scope**: Strictest validation for releases
- **Gates**: 90+ hardening score, <5% degradation  
- **Hard Negatives**: 7 per query (maximum adversarial testing)

### Notification Integration

```typescript
const ciConfig = {
  slack_webhook_url: process.env.SLACK_WEBHOOK_URL,
  quality_gates: {
    enforce_tripwires: true,
    enforce_slice_gates: true,
    min_hardening_score: 80
  }
};
```

Notifications include:
- Hardening score and execution time
- Tripwire and slice gate results  
- Failure summaries with key recommendations
- Links to detailed artifacts and reports

## Configuration Reference

### HardeningConfig

```typescript
interface HardeningConfig {
  // Hard negative injection
  hard_negatives: {
    enabled: boolean;
    per_query_count: number;          // 3-7 recommended
    shared_subtoken_min: number;      // 1-3 recommended
  };
  
  // Per-slice gates  
  per_slice_gates: {
    enabled: boolean;
    min_recall_at_10: number;         // 0.65-0.75
    min_ndcg_at_10: number;           // 0.55-0.65
    max_p95_latency_ms: number;       // 450-600ms
  };
  
  // Tripwire thresholds
  tripwires: {
    min_span_coverage: number;        // 0.96-0.99
    recall_convergence_threshold: number;  // 0.003-0.01
    lsif_coverage_drop_threshold: number;  // 0.03-0.1
    p99_p95_ratio_threshold: number;       // 1.8-2.5
  };
  
  // Plot generation
  plots: {
    enabled: boolean;
    output_dir: string;
    formats: ('png' | 'svg' | 'pdf')[];
  };
}
```

### CIHardeningConfig

```typescript
interface CIHardeningConfig {
  ci_mode: 'pr' | 'nightly' | 'release';
  fail_fast: boolean;
  max_execution_time_minutes: number;
  
  quality_gates: {
    enforce_tripwires: boolean;
    enforce_slice_gates: boolean; 
    min_hardening_score: number;     // 0-100
    max_degradation_percent: number; // 0-50
  };
  
  retry_policy: {
    enabled: boolean;
    max_retries: number;             // 0-5
    backoff_multiplier: number;      // 1.5-3.0
  };
}
```

## Troubleshooting

### Common Issues

#### Low Hardening Score (<70)
- **Cause**: Failed tripwires or slice gates
- **Solution**: Check tripwire details and address root causes
- **Prevention**: Monitor trends and set alerting thresholds

#### High Degradation (>15%)  
- **Cause**: Ranking sensitivity to hard negatives
- **Solution**: Improve ranking robustness and feature engineering
- **Prevention**: Regular adversarial testing in development

#### CI Timeout
- **Cause**: Insufficient execution time budget
- **Solution**: Increase timeout or reduce test scope
- **Prevention**: Monitor execution trends and optimize

### Debug Commands

```bash
# Verbose logging
npx tsx src/benchmark/cli-phase-c.ts phase-c --verbose

# Skip plots for faster iteration
npx tsx src/benchmark/cli-phase-c.ts phase-c --no-plots

# Disable hard negatives for debugging
npx tsx src/benchmark/cli-phase-c.ts phase-c --no-hard-negatives

# Generate report from existing data
npx tsx src/benchmark/cli-phase-c.ts report -i hardening-report.json

# Validate configuration
npx tsx src/benchmark/cli-phase-c.ts validate
```

## Performance Targets

### Hardening Score Targets
- **Development**: >60 (basic quality)
- **PR Gates**: >65 (acceptable for review)
- **Nightly**: >80 (production ready)
- **Release**: >90 (high confidence)

### Execution Time Targets
- **PR Mode**: <15 minutes (fast feedback)
- **Nightly Mode**: <45 minutes (comprehensive)
- **Release Mode**: <60 minutes (thorough validation)

### Quality Metrics
- **Tripwire Pass Rate**: >95%
- **Slice Gate Pass Rate**: >90% 
- **Hard Negative Robustness**: <10% degradation
- **Report Generation**: <2 minutes

## Integration Examples

### With Existing Benchmarks

```typescript
// Extend existing benchmark with hardening
const result = await suiteRunner.runFullSuiteWithHardening({
  systems: ['lex', '+symbols', '+symbols+semantic'],
  robustness: true
});

console.log('Benchmark status:', result.status);
console.log('Hardening status:', result.hardening?.hardening_status);
```

### Custom Hardening Pipeline

```typescript
// Custom hardening configuration
const customConfig = createDefaultHardeningConfig(baseConfig);
customConfig.tripwires.min_span_coverage = 0.995; // 99.5%
customConfig.hard_negatives.per_query_count = 10; // More adversarial

const hardening = new PhaseCHardening('./output');
const report = await hardening.executeHardening(customConfig, results, queries);

// Generate custom report
const pdfGenerator = new PDFReportGenerator('./output');
const pdfPath = await pdfGenerator.generateHardeningReport(
  report, 
  results, 
  { template: 'executive', include_plots: false }
);
```

### Monitoring Integration

```typescript
// Set up continuous monitoring
const monitor = setInterval(async () => {
  const ciOrchestrator = new CIHardeningOrchestrator('./monitoring');
  const result = await ciOrchestrator.executeInCI({
    ci_mode: 'nightly',
    quality_gates: { min_hardening_score: 85 }
  });
  
  if (!result.success) {
    await sendAlert(result.failure_summary);
  }
}, 24 * 60 * 60 * 1000); // Daily
```

## Roadmap

### Planned Enhancements
- **Interactive Plots**: Web-based dashboard with interactive visualizations
- **ML-based Tripwires**: Adaptive thresholds based on historical data
- **Custom Hard Negatives**: Domain-specific adversarial generation
- **Performance Regression Detection**: Automated baseline comparison
- **Multi-repo Analysis**: Cross-repository performance correlation

### Integration Targets
- **Prometheus/Grafana**: Metrics export for operational dashboards
- **DataDog/New Relic**: APM integration for production monitoring
- **Slack/Teams**: Enhanced notification formatting
- **Jira/Linear**: Automated issue creation for failures

---

**Phase C Hardening ensures your lens system maintains high quality and performance under all conditions. The comprehensive testing, validation, and reporting provide confidence for production deployments while catching regressions before they impact users.**