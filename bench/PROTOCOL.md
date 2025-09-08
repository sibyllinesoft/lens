# Benchmark Protocol v1.0 - Competitive Evaluation Framework

**Status**: Production-Ready Competitive Benchmarking  
**Version**: 1.0  
**Date**: 2025-09-07  

## Overview

This document specifies the **Benchmark Protocol v1.0** - a comprehensive competitive evaluation system designed to provide bulletproof SOTA claims with SLA-bounded fairness, pooled qrels, gap mining, and publication-grade results. All systems compete under identical constraints with complete transparency and artifact binding.

## 1. Scope & Fairness (Non-Negotiable)

### SLA Enforcement
- **Hard Limit**: 150ms per query
- **Exclusion Rule**: Results exceeding SLA are excluded from Recall@50 ("SLA-Recall")
- **Latency Measurement**: Server-side counting with client watchdog enforcement
- **Timeout Handling**: >150ms results marked as timeouts, excluded from quality metrics

### Hardware Parity
- **Identical Hardware**: Same host SKU across all systems
- **CPU Control**: Pinned CPU flags, governor, kernel version
- **Attestation Required**: `/proc/cpuinfo`, image digests, config fingerprint per run
- **Resource Limits**: Equal memory, CPU cores, storage allocation

### Corpus Management
- **Frozen Manifests**: Paths + SHA256 for each test suite
- **Pooled Qrels**: Built from union of in-SLA top-k across all systems
- **Version Control**: Complete artifact binding with build hashes
- **Reproducibility**: External replication ready with full config fingerprints

### Adapter Requirements
- **Uniform Interface**: All competitors answer identical NDJSON schema
- **Cache Parity**: No private caches beyond default product exposure
- **Equal Warmup**: Identical warmup phase duration for all systems
- **Standard Preparation**: Common setup/teardown procedures

## 2. Test Suites & Evaluation Slices

### Core Evaluation Suites

#### Task-Level Evaluation
- **SWE-bench Verified**: Success@k and witness-coverage@k metrics
- **Span Sources**: PR diffs and code change witnesses
- **Coverage Validation**: 100% span coverage requirement
- **Task Attribution**: Failure analysis with root cause mapping

#### Information Retrieval
- **CoIR Aggregate** (Primary UR-Broad): Multi-language code search
- **CodeSearchNet**: Legacy comparability benchmark
- **CoSQA**: Question-answering evaluation (note: label noise acknowledged)
- **Cross-Suite Consistency**: Uniform evaluation methodology

#### Parity Validation
- **CP-Regex**: Exact/regex matching to show no regressions vs grep-class
- **UR-Narrow**: Assisted-lexical arena with scrupulous fairness to lexical tools
- **Regression Gates**: Ensure basic functionality preserved

### Evaluation Dimensions
- **Intent Classification**: Exact match, identifier lookup, structural search, semantic search
- **Language Coverage**: Python, TypeScript, JavaScript, Java, Go, Rust, C++
- **Repository Types**: Open source, enterprise, domain-specific codebases

## 3. Competitor Matrix & Driver Architecture

### System Categories

#### Lexical Systems
- **ripgrep**: Regex-based search with performance optimization
- **Elasticsearch BM25**: Text search with term proximity enhancement
- **Configuration**: Standard parameters, no system-specific tuning

#### Hybrid Systems
- **BM25+kNN**: OSS hybrid with tuned proximity scoring
- **Dense Retrieval**: Embedding-based search with lexical fallback
- **Tuning Constraints**: Public parameters only, no proprietary optimization

#### Structured/LSP Systems
- **Sourcegraph-class**: Code navigation and structural search
- **LSP-based**: Language server protocol integration
- **Symbol Support**: Type-aware search with syntax understanding

#### Target System
- **Lens Current**: Frozen weights + calibration + policy
- **Artifact Binding**: Complete build reproducibility
- **Version Control**: Git SHA + config fingerprint

### Adapter Interface Specification

```typescript
interface CompetitorAdapter {
  // System preparation and warmup
  prepare(config: AdapterConfig): Promise<void>;
  
  // Core search interface with SLA enforcement
  search(query: string, sla_ms: number): Promise<SearchHit[]>;
  
  // Cleanup and resource release
  teardown(): Promise<void>;
  
  // System metadata for attestation
  getSystemInfo(): SystemInfo;
}

interface SearchHit {
  file: string;
  line: number;
  column: number;
  snippet: string;
  score: number;
  why_tag: 'exact' | 'struct' | 'semantic' | 'mixed';
  symbol_kind?: string;
  ast_path?: string;
  byte_offset?: number;
  span_length?: number;
}
```

## 4. Comprehensive Metrics Framework (SLA-Bounded)

### Quality Metrics
- **nDCG@10**: Normalized Discounted Cumulative Gain at rank 10
- **Success@10**: Binary success rate for top-10 results
- **SLA-Recall@50**: Recall within SLA constraint at rank 50
- **witness_coverage@10**: SWE-bench witness span coverage at rank 10

### Operational Metrics  
- **Latency Distribution**: p50/p95/p99 response times
- **Tail Ratio**: p99/p95 for tail behavior analysis
- **Throughput**: QPS@150ms sustained query rate
- **Reliability**: Timeout percentage and error rates

### Calibration Metrics
- **ECE**: Expected Calibration Error per intent×language slice
- **Calibration Slope/Intercept**: Per slice calibration parameters
- **Risk Assessment**: Slice-level calibration quality flags

### Explainability & Integrity
- **why_mix Distribution**: exact/struct/semantic result composition
- **Core@10**: Centrality decile share (topic-normalized)
- **Diversity@10**: Unique files/topics in top results
- **Span Coverage**: Must be 100% for all valid results

### Fairness Slicing
All metrics broken down by:
- **Intent**: exact_match, identifier, structural, semantic
- **Language**: Python, TypeScript, JavaScript, Java, Go, Rust, C++
- **Repository**: Per-repo performance analysis for bias detection

## 5. Output Schema & Data Format

### Parquet Schema Design

#### Aggregate Table (`agg.parquet`)
**Schema**: One row per (query, system) combination

```
query_id: string
system_id: string  
build_hash: string
policy_fingerprint: string
suite: string
slice_intent: string
slice_lang: string
sla_ms: int32
q_time_ms: float64
within_sla: boolean

# Quality Metrics
ndcg_at_10: float64
success_at_10: float64
sla_recall_at_50: float64
witness_cov_at_10: float64

# Operational Metrics  
p50: float64
p95: float64
p99: float64
p99_over_p95: float64
qps150x: float64

# Calibration Metrics
ece: float64
calib_slope: float64
calib_intercept: float64

# Explainability Metrics
why_mix_exact: float64
why_mix_struct: float64
why_mix_semantic: float64
core_at_10: float64
diversity_at_10: float64

# Error Tracking
timeouts: int32
errors: int32
attestation_sha256: string
```

#### Detail Table (`hits.parquet`)
**Schema**: One row per (query, system, hit) combination

```
query_id: string
system_id: string
rank: int32
file: string
line: int32
col: int32
lang: string
snippet_hash: string
score: float64
why_tag: string
symbol_kind: string (nullable)
ast_path: string (nullable)
byte_offset: int64 (nullable)
span_len: int32 (nullable)
```

### Data Processing Pipeline
1. **Collection**: Real-time collection during benchmark execution
2. **Validation**: Schema compliance and data quality checks
3. **Export**: Parquet format with compression optimization
4. **Indexing**: Optimized for slice-based analysis queries

## 6. Statistical Analysis Framework

### Statistical Methods
- **Paired Stratified Bootstrap**: B≥2,000 iterations for confidence intervals
- **Paired Permutation Testing**: + Holm correction for multiple comparisons
- **Effect Size Reporting**: Cohen's d for practical significance
- **Stratification**: By intent×language for representative sampling

### Significance Testing
- **Null Hypothesis**: No performance difference between systems
- **Alternative Hypothesis**: Directional performance differences
- **Alpha Level**: 0.05 with Holm correction for multiple testing
- **Power Analysis**: Minimum detectable effect size calculation

### Confidence Intervals
- **Bootstrap CIs**: 95% confidence intervals for all metrics
- **Percentile Method**: Bias-corrected and accelerated (BCa)
- **Stratified Sampling**: Maintains slice representativeness

## 7. Gap Mining & Weakness Analysis

### Automated Weakness Detection

#### Slice Delta Analysis
```sql
SELECT 
  intent,
  language,
  (ndcg10_lens - MAX(ndcg10_competitor)) as delta_ndcg10,
  RANK() OVER (ORDER BY delta_ndcg10 ASC) as weakness_rank
FROM agg_table
GROUP BY intent, language
ORDER BY delta_ndcg10 ASC
LIMIT 10;
```

#### Witness Miss Attribution (SWE-bench)
```sql
SELECT
  why_mix_dominant,
  COUNT(*) as miss_count,
  AVG(1 - witness_cov_at_10) as avg_miss_rate
FROM agg_table 
WHERE suite = 'swe_verified' AND witness_cov_at_10 < 1.0
GROUP BY why_mix_dominant
ORDER BY miss_count DESC;
```

#### Timeout Attribution Analysis
```sql
SELECT
  intent,
  language,
  (timeouts / total_queries) as timeout_share,
  CORR(timeout_share, delta_sla_recall_50) as timeout_impact
FROM agg_table
WHERE timeout_share > 0.1
GROUP BY intent, language;
```

#### Calibration Risk Flagging
```sql
SELECT
  intent,
  language,
  ece,
  calib_slope,
  CASE 
    WHEN ece > 0.02 THEN 'HIGH_ECE_RISK'
    WHEN calib_slope < 0.9 OR calib_slope > 1.1 THEN 'SLOPE_RISK'
    ELSE 'CALIBRATED'
  END as risk_flag
FROM agg_table
WHERE risk_flag != 'CALIBRATED';
```

### Backlog Generation
**Output Format**: CSV with actionable insights

```csv
slice,delta_ndcg_pp,miss_at_10,timeout_share,ece,top_failure_reason,priority,effort_estimate
python_semantic,-0.15,0.23,0.08,0.03,timeout_struct_parsing,high,2_weeks
typescript_identifier,-0.08,0.12,0.15,0.01,lexical_fallback_needed,medium,1_week
...
```

## 8. Publication-Grade Visualizations

### Automated Plot Generation

#### Hero Charts
- **nDCG@10 Bar Chart**: Per system, per suite with 95% CIs
- **SLA-Recall@50 Bar Chart**: System comparison with error bars
- **Combined Performance**: Dual-axis plot showing quality vs speed

#### Latency Analysis
- **Ridge Plots**: p50/p95/p99 distributions per system
- **Box Plots**: Latency distribution comparison
- **p99/p95 Annotation**: Tail behavior analysis

#### Calibration Visualization
- **Reliability Diagrams**: ECE per {intent×language} slice
- **Calibration Curves**: Predicted vs observed success rates
- **Risk Heatmaps**: Calibration quality across slices

#### Performance Analysis
- **Slice Heatmap**: ΔnDCG@10 by {intent×language} - red bins indicate gaps
- **Why-mix Stacked Bars**: exact/struct/semantic composition vs competitors
- **Core@10 Scatter**: Topic centrality vs diversity trade-offs

#### Task-Level Analysis
- **Witness Coverage CDF**: SWE-bench success distribution
- **Success@10 by Language**: Task completion rates
- **Failure Mode Analysis**: Why-mix attribution for misses

#### ROI & Business Metrics
- **SLA Utility Curve**: nDCG lift vs SLA constraint (marketing ROI)
- **Cost-Quality Trade-off**: Performance per compute cost
- **Competitive Positioning**: Multi-dimensional system comparison

### Figure Generation Pipeline
```python
# Automated figure generation
def generate_publication_figures(scored_data_path: str, output_dir: str):
    data = load_parquet_data(scored_data_path)
    
    # Hero charts with error bars
    plot_hero_ndcg_bars(data, f"{output_dir}/hero_ndcg.png")
    plot_hero_sla_recall_bars(data, f"{output_dir}/hero_sla_recall.png")
    
    # Latency analysis
    plot_latency_ridges(data, f"{output_dir}/latency_ridges.png")
    plot_tail_ratio_analysis(data, f"{output_dir}/tail_analysis.png")
    
    # Calibration assessment
    plot_reliability_diagrams(data, f"{output_dir}/calibration.png")
    plot_ece_heatmap(data, f"{output_dir}/ece_heatmap.png")
    
    # Gap analysis
    plot_slice_heatmap(data, f"{output_dir}/gap_heatmap.png")
    plot_why_mix_shifts(data, f"{output_dir}/why_mix.png")
    
    # Task-level insights
    plot_witness_coverage_cdf(data, f"{output_dir}/witness_cdf.png")
    
    # Business metrics
    plot_sla_utility_curve(data, f"{output_dir}/sla_utility.png")
```

## 9. CLI Interface & Runbook Commands

### Complete Command Reference

#### Pool Building
```bash
# Build pooled qrels from in-SLA results across systems
bench build-pool \
  --suites coir,swe_verified,csn,cosqa \
  --sla 150 \
  --systems lens,bm25,bm25_prox,hybrid,sourcegraph \
  --out pool/ \
  --min-agreement 2
```

#### System Preparation
```bash  
# Warm competitors, pin hardware, create attestation
bench warmup \
  --systems lens,bm25,bm25_prox,hybrid,sourcegraph \
  --warmup-queries 100 \
  --hardware-check strict \
  --attest attest.json
```

#### Benchmark Execution
```bash
# Execute individual suites with full measurement
bench run \
  --suite coir \
  --systems lens,bm25,bm25_prox,hybrid,sourcegraph \
  --sla 150 \
  --queries-per-system 1000 \
  --parallel-workers 4 \
  --out runs/coir/

bench run \
  --suite swe_verified \
  --systems lens,bm25,bm25_prox,hybrid,sourcegraph \
  --sla 150 \
  --witness-validation strict \
  --out runs/swe/
```

#### Statistical Analysis
```bash
# Score with pooled qrels + statistical testing
bench score \
  --runs runs/* \
  --pool pool/ \
  --bootstrap 2000 \
  --permutation-test \
  --holm-correction \
  --effect-size cohens-d \
  --out scored/
```

#### Gap Mining & Analysis
```bash
# Automated weakness analysis and backlog generation  
bench mine \
  --in scored/agg.parquet \
  --slice-analysis intent,language \
  --witness-attribution \
  --timeout-analysis \
  --calibration-flags \
  --out reports/gaps.csv

# Detailed failure analysis
bench analyze-failures \
  --in scored/ \
  --focus-slices python_semantic,typescript_identifier \
  --out reports/failure_analysis/
```

#### Visualization Generation
```bash
# Auto-generate publication-grade figures
bench plot \
  --in scored/ \
  --figures hero,latency,calibration,gaps,witness,utility \
  --format png,pdf \
  --dpi 300 \
  --out reports/figs/

# Interactive dashboard generation  
bench dashboard \
  --in scored/ \
  --port 8080 \
  --public-access false
```

#### Artifact Management
```bash
# Bundle complete reproducibility package
bench package \
  --run-data scored/ \
  --attestation attest.json \
  --figures reports/figs/ \
  --out artifacts/benchmark_v1_$(date +%Y%m%d).tar.gz

# Validate external reproducibility
bench validate \
  --package artifacts/benchmark_v1_20250907.tar.gz \
  --verify-hashes \
  --check-dependencies \
  --test-subset 10
```

## 10. Publication Guardrails & Claims Framework

### Claim Validation Requirements
- **SLA-Bounded Only**: All claims must reference SLA-bounded results
- **Statistical Significance**: Include p-values with Holm correction
- **Effect Sizes**: Report practical significance with Cohen's d  
- **Confidence Intervals**: 95% CIs for all performance claims
- **Slice Analysis**: Break down claims by intent×language when relevant

### Artifact Binding Standards
- **Config Fingerprints**: Every figure caption includes config hash
- **Build Hashes**: Complete reproducibility chain documented
- **Data Provenance**: Corpus versions and qrels binding explicit
- **External Verification**: Replication package provided

### Narrative Framework Alignment  
- **Ladder Progression**: UR-Broad → UR-Narrow → CP-Regex structure
- **Gap Acknowledgment**: Explicit discussion of weakness areas
- **Fairness Emphasis**: Hardware parity and SLA enforcement highlighted
- **Future Work**: Gap mining results inform research roadmap

### Review & Validation Process
1. **Internal Review**: Statistical methodology validation
2. **External Review**: Independent reproduction attempt
3. **Artifact Check**: Complete reproducibility verification
4. **Claims Audit**: Each claim traceable to specific data

## Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1-2)
- [ ] Competitor adapter framework
- [ ] SLA execution engine  
- [ ] Basic metrics collection
- [ ] Parquet schema implementation

### Phase 2: Statistical Framework (Week 3)
- [ ] Bootstrap and permutation testing
- [ ] Effect size calculations
- [ ] Confidence interval generation
- [ ] Multiple testing correction

### Phase 3: Analysis & Mining (Week 4)
- [ ] Gap mining algorithms
- [ ] Weakness detection system
- [ ] Backlog generation
- [ ] Failure attribution analysis

### Phase 4: Visualization & CLI (Week 5)
- [ ] Publication plot generation
- [ ] Interactive dashboards
- [ ] Complete CLI interface
- [ ] Artifact packaging system

### Phase 5: Validation & Documentation (Week 6)
- [ ] End-to-end testing
- [ ] External reproducibility validation
- [ ] Documentation completion
- [ ] Performance optimization

## Quality Assurance

### Testing Strategy
- **Unit Tests**: Each component thoroughly tested
- **Integration Tests**: End-to-end pipeline validation
- **Performance Tests**: SLA enforcement verification
- **Reproducibility Tests**: External package validation

### Monitoring & Alerts
- **SLA Violations**: Real-time monitoring during execution
- **Statistical Anomalies**: Outlier detection and flagging
- **Resource Usage**: Memory/CPU monitoring per system
- **Data Quality**: Schema validation and integrity checks

### Maintenance & Updates
- **Version Control**: Semantic versioning for protocol changes
- **Backward Compatibility**: Migration paths for existing data
- **Documentation**: Living documentation with change tracking
- **External Dependencies**: Regular security and compatibility updates

---

**Protocol Version**: 1.0  
**Effective Date**: 2025-09-07  
**Next Review**: 2025-12-07  
**Maintainer**: Lens Benchmarking Team  
**External Validation**: Independent reproduction required for SOTA claims