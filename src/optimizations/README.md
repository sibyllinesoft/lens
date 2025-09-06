# Lens Search Optimization Systems

Four durable, embedder-agnostic search optimizations that provide structural improvements independent of embedding model choice. All systems implement comprehensive SLA compliance validation per TODO.md requirements.

## 🎯 Overview

### Core Principle: Embedder Agnostic
These optimizations work at the structural level and will survive any embedding model changes. They provide durable improvements to search quality, performance, and user experience without requiring ML retraining when switching embeddings.

### The Four Systems

1. **[Clone-Aware Recall](#clone-aware-recall)** - Token shingle expansion across code clones, forks, and backports
2. **[Learning-to-Stop](#learning-to-stop)** - ML-based early termination for scanners and ANN search
3. **[Targeted Diversity](#targeted-diversity)** - Constrained MMR applied only to overview queries with high entropy
4. **[TTL That Follows Churn](#ttl-that-follows-churn)** - Adaptive cache management based on observed code churn

## 🚀 Quick Start

### Basic Usage

```typescript
import { setupOptimizedSearch } from './optimizations';

// Production setup with all optimizations
const { engine, monitor, shutdown } = await setupOptimizedSearch('production');

// Use in your search pipeline
const originalHits = await yourSearchFunction(query, context);
const optimizedPipeline = await engine.optimizeSearchResults(
  originalHits,
  context,
  diversityFeatures // optional
);

// Get optimized results
console.log(`Optimized from ${originalHits.length} to ${optimizedPipeline.final_hits.length} hits`);
console.log(`Applied optimizations: ${optimizedPipeline.optimizations_applied.join(', ')}`);

// Cleanup
await shutdown();
```

### Custom Configuration

```typescript
import { OptimizationEngine, OPTIMIZATION_PRESETS } from './optimizations';

// Custom configuration
const customConfig = {
  ...OPTIMIZATION_PRESETS.PRODUCTION,
  learning_to_stop_enabled: false, // Disable specific optimization
};

const engine = new OptimizationEngine(customConfig);
await engine.initialize();

// Use engine...

await engine.shutdown();
```

## 📊 Performance Requirements & SLA Compliance

All systems implement strict performance gates from TODO.md:

### Clone-Aware Recall
- **Recall Target**: +0.5-1.0pp Recall@50 improvement
- **Latency Budget**: ≤+0.6ms p95 latency
- **Clone Budget**: |C(s)| ≤ 3 clones per expansion
- **Jaccard Bonus**: β ≤ 0.2 log-odds, bounded
- **Coverage**: 100% span coverage

### Learning-to-Stop
- **Performance**: p95 -0.8 to -1.5ms improvement
- **Quality**: SLA-Recall@50 ≥ 0 (no degradation)
- **Upshift**: 3%-7% result quality improvement
- **Never-Stop**: Floor when positives_in_candidates < m

### Targeted Diversity
- **Application**: Only NL_overview ∧ high_entropy queries
- **Quality Gate**: ΔnDCG@10 ≥ 0 (no degradation)
- **Diversity Target**: +10% diversity improvement
- **Hard Constraints**: Exact/structural match floors preserved

### TTL That Follows Churn
- **Performance**: p95 -0.5 to -1.0ms improvement
- **Quality**: why-mix KL ≤ 0.02 (no drift)
- **Invalidation**: Zero span drift tolerance
- **TTL Bounds**: τ_min=1s, τ_max=30s, c≈3

## 🔍 Detailed System Documentation

### Clone-Aware Recall

Expands search results by finding code clones across repositories using token-shingle MinHash/SimHash indexing.

#### How it Works
1. **Indexing**: Content tokenized into subtokens, shingles generated (w=5-7)
2. **Clone Detection**: MinHash creates clone sets with similarity threshold
3. **Expansion**: Original hits expanded with budget-constrained clones
4. **Scoring**: Jaccard bonus applied to clone hits (bounded in log-odds)

#### Configuration
```typescript
// Index content for clone detection
await engine.indexContent(
  'function calculateSum(a, b) { return a + b; }',
  'math/utils.ts',
  1, 0, // line, col
  'main-repo',
  'function' // symbol kind
);

// Expansion happens automatically during optimization
```

#### Constraints
- **Clone Budget**: Maximum 3 clones per expansion (k_clone ≤ 3)
- **Veto Rules**: Same-repo + same-symbol-kind combinations rejected
- **Topic Filter**: topic_sim > τ threshold required
- **Path Filter**: Vendor/third-party paths excluded

### Learning-to-Stop

Uses lightweight ML model to make early termination decisions for WAND/BMW scanners and ANN efSearch optimization.

#### Features Used
- `impact_prefix_gain`: Estimated gain from next block
- `remaining_budget_ms`: Time left in query budget  
- `topic_entropy`: Current result set diversity
- `pos_in_cands`: Position in candidate list
- `λ_ann(ms/ΔRecall)`: ANN efficiency ratio

#### Scanner Integration
```typescript
const decision = engine.shouldStopScanning(
  blocksProcessed,
  candidatesFound, 
  timeSpentMs,
  searchContext,
  queryStartTime
);

if (decision.shouldStop) {
  // Terminate scanning early
  break;
}
```

#### ANN Integration
```typescript
const optimizedEf = engine.getOptimizedEfSearch(
  currentEf,
  recallAchieved,
  timeSpentMs,
  riskLevel,
  searchContext,
  queryStartTime
);
```

### Targeted Diversity

Applies constrained Maximum Marginal Relevance (MMR) selectively for natural language overview queries only.

#### Activation Criteria
- **Query Type**: Must be `NL_overview`
- **Entropy**: topic_entropy > 0.6 (high entropy threshold)
- **Clone Collapse**: Only after clone expansion (prevents fake diversity)
- **Result Count**: Minimum 5 results needed

#### MMR Optimization
```
argmax_S Σᵢ∈S rᵢ - γ Σᵢ<j sim_topic/symbol(i,j)
subject to: floors(exact,struct) = true
```

#### Usage
```typescript
const diversityFeatures = {
  query_type: 'NL_overview',
  topic_entropy: 0.85,
  result_count: hits.length,
  exact_matches: exactCount,
  structural_matches: structCount,
  clone_collapsed: true,
};

const pipeline = await engine.optimizeSearchResults(hits, context, diversityFeatures);
```

### TTL That Follows Churn

Adaptive cache management that adjusts TTL based on observed code churn rates and span invalidations.

#### Churn-Aware Formula
```
TTL = clamp(τ_min, τ_max, c/λ_churn)
```
Where:
- `τ_min = 1s` (minimum TTL)
- `τ_max = 30s` (maximum TTL)  
- `c ≈ 3` (churn constant)
- `λ_churn` = observed churn rate (changes/second)

#### Cache Types
```typescript
// Micro-cache for search results
const result = await engine.getCachedValue(
  cacheKey,
  indexVersion,
  spanHash,
  async () => expensiveComputation(),
  'micro',
  'topic-bin'
);

// RAPTOR hierarchy cache
const raptor = await engine.getCachedValue(
  key, version, '', factory, 'raptor'
);

// Centrality cache
const centrality = await engine.getCachedValue(
  key, version, '', factory, 'centrality'
);
```

#### Churn Tracking
```typescript
// Record file changes for churn rate calculation
engine.recordFileChange('src/modified-file.ts', timestamp);

// Automatic invalidation on version/hash mismatch
// Cache entries automatically invalidated when:
// - index_version changes
// - span_hash changes  
// - TTL expires
```

## 🔧 Integration Patterns

### Search Pipeline Integration

```typescript
// 1. Initialize engine
const engine = new OptimizationEngine(config);
await engine.initialize();

// 2. Index content for clone detection
await engine.indexContent(content, file, line, col, repo, symbolKind);

// 3. Record file changes for churn tracking  
engine.recordFileChange(filePath, timestamp);

// 4. Use during search (scanner integration)
const shouldStop = engine.shouldStopScanning(blocks, candidates, time, ctx, start);
const efSearch = engine.getOptimizedEfSearch(ef, recall, time, risk, ctx, start);

// 5. Optimize final results
const pipeline = await engine.optimizeSearchResults(hits, context, features);

// 6. Use optimized results
return pipeline.final_hits;
```

### Performance Monitoring

```typescript
import { PerformanceMonitor, MONITORING_PRESETS } from './optimizations';

// Create monitor
const monitor = new PerformanceMonitor(engine, MONITORING_PRESETS.PRODUCTION);
await monitor.startMonitoring();

// Run benchmarks
const result = await monitor.runComprehensiveBenchmark('my-test');

// Check SLA compliance
console.log('SLA Compliant:', result.sla_compliance.overall_compliant);
console.log('Alerts:', result.alerts);
console.log('Recommendations:', result.recommendations);

// Generate report
const report = monitor.generatePerformanceReport();
console.log(report);
```

### Error Handling & Graceful Degradation

```typescript
// Systems gracefully degrade on failure
const pipeline = await engine.optimizeSearchResults(hits, context);

// Check what optimizations were applied
console.log('Applied:', pipeline.optimizations_applied);

// Original hits returned if all optimizations fail
console.log('Results:', pipeline.final_hits);

// Monitor system health
const health = engine.getSystemHealth();
if (!health.overall_healthy) {
  console.warn('Degraded systems:', health.degraded_optimizations);
  
  // Trigger recovery
  await engine.performHealthCheckAndRecovery();
}
```

## 🎛️ Configuration Reference

### OptimizationConfig

```typescript
interface OptimizationConfig {
  clone_aware_enabled: boolean;        // Enable clone-aware recall
  learning_to_stop_enabled: boolean;   // Enable learning-to-stop
  targeted_diversity_enabled: boolean; // Enable targeted diversity
  churn_aware_ttl_enabled: boolean;   // Enable churn-aware TTL
  performance_monitoring_enabled: boolean; // Enable metrics collection
  graceful_degradation_enabled: boolean;   // Enable error recovery
}
```

### MonitoringConfig

```typescript
interface MonitoringConfig {
  benchmark_interval_ms: number;           // Frequency of automatic benchmarks
  alert_threshold_violations: number;     // Alerts before firing notification
  performance_degradation_threshold: number; // Performance drop threshold
  enable_real_time_monitoring: boolean;   // Real-time benchmark execution
  enable_alerting: boolean;               // Enable alert notifications
  log_level: 'debug' | 'info' | 'warn' | 'error'; // Logging verbosity
}
```

### DiversityFeatures

```typescript
interface DiversityFeatures {
  query_type: 'NL_overview' | 'targeted_search' | 'symbol_lookup' | 'other';
  topic_entropy: number;        // 0-1, entropy of query topics
  result_count: number;         // Number of results to diversify
  exact_matches: number;        // Count of exact matches (protected)
  structural_matches: number;   // Count of structural matches (protected)
  clone_collapsed: boolean;     // Whether clone expansion was applied
}
```

## 📈 Performance Monitoring

### SLA Metrics Tracked

- **Recall@50**: Search recall at 50 results
- **P95 Latency**: 95th percentile optimization time
- **Upshift %**: Quality improvement percentage
- **Diversity Score**: Result set diversity measurement
- **Why-Mix KL**: Distribution drift measurement
- **Span Coverage**: Percentage of spans covered by optimizations

### Health Monitoring

```typescript
// Get comprehensive metrics
const metrics = engine.getPerformanceMetrics();

// Key metrics
console.log('SLA Compliance Rate:', metrics.sla_compliance_rate);
console.log('Average Optimization Time:', metrics.average_optimization_time_ms);
console.log('System Health:', metrics.system_health);

// Individual system metrics
console.log('Clone Aware:', metrics.subsystem_metrics.clone_aware);
console.log('Learning Stop:', metrics.subsystem_metrics.learning_to_stop);
console.log('Diversity:', metrics.subsystem_metrics.targeted_diversity);
console.log('TTL:', metrics.subsystem_metrics.churn_aware_ttl);
```

### Alerting

The monitoring system generates three types of alerts:

- **CRITICAL**: SLA violations, system health degradation
- **WARNING**: Performance targets missed, individual system issues  
- **INFO**: Configuration recommendations, optimization opportunities

## 🧪 Testing

### Running Tests

```bash
# Run all optimization tests
npm test src/optimizations

# Run specific system tests
npm test src/optimizations/__tests__/clone-aware-recall.test.ts
npm test src/optimizations/__tests__/optimization-engine.test.ts
npm test src/optimizations/__tests__/integration.test.ts

# Run with coverage
npm test -- --coverage src/optimizations
```

### Test Categories

- **Unit Tests**: Individual system functionality and constraints
- **Integration Tests**: Cross-system coordination and SLA compliance
- **Performance Tests**: Latency, throughput, and resource usage validation
- **Benchmark Tests**: SLA requirement validation against TODO.md specifications

### Custom Test Scenarios

```typescript
// Create custom test scenario
const testHits = [
  createMockSearchHit('file1.ts', 10, 95, 'function'),
  createMockSearchHit('file2.ts', 20, 85, 'class'),
];

const context = createMockSearchContext('test query');
const features = createMockDiversityFeatures('NL_overview', 0.8);

const pipeline = await engine.optimizeSearchResults(testHits, context, features);

// Validate SLA compliance
const sla = validatePipelineSLA(pipeline);
expect(sla.compliant).toBe(true);
```

## 🔍 Troubleshooting

### Common Issues

#### Clone Expansion Not Working
```typescript
// Verify content is indexed
await engine.indexContent(content, file, line, col, repo, symbolKind);

// Check if clones exist in different repos
// Same-repo, same-symbol-kind clones are vetoed
```

#### Diversity Not Applied
```typescript
// Verify query type and entropy requirements
const features = {
  query_type: 'NL_overview', // Must be overview
  topic_entropy: 0.8,        // Must be > 0.6
  clone_collapsed: true,     // Must be after clone expansion
};
```

#### Cache Not Working
```typescript
// Verify TTL system is enabled
const config = {
  churn_aware_ttl_enabled: true,
};

// Check cache key consistency
const result = await engine.getCachedValue(
  consistentKey,    // Same key for same operation
  indexVersion,     // Current index version
  spanHash,         // Current span hash
  factory
);
```

#### Performance Degradation
```typescript
// Check system health
const health = engine.getSystemHealth();
if (!health.overall_healthy) {
  // Perform recovery
  await engine.performHealthCheckAndRecovery();
}

// Monitor performance
const monitor = new PerformanceMonitor(engine, config);
const benchmark = await monitor.runComprehensiveBenchmark('debug');
console.log('Issues:', benchmark.alerts);
console.log('Recommendations:', benchmark.recommendations);
```

### Debug Mode

```typescript
// Enable debug logging
const monitor = new PerformanceMonitor(engine, {
  ...MONITORING_PRESETS.DEVELOPMENT,
  log_level: 'debug',
});

// Run detailed benchmark
await monitor.runComprehensiveBenchmark('debug-test');
```

## 🔬 Architecture

### System Architecture

```
┌─────────────────────┐
│ OptimizationEngine  │ ← Main orchestrator
├─────────────────────┤
│ CloneAwareRecall    │ ← Phase 1: Expand with clones
│ LearningToStop      │ ← Integrated with search phase
│ TargetedDiversity   │ ← Phase 2: Apply MMR if needed
│ ChurnAwareTTL       │ ← Cross-cutting caching
└─────────────────────┘
```

### Data Flow

1. **Indexing Phase**: Content indexed for clone detection
2. **Search Phase**: Learning-to-stop guides scanner/ANN termination
3. **Expansion Phase**: Clone-aware recall expands results
4. **Diversity Phase**: Targeted diversity applied (if qualified)
5. **Caching**: All operations cached with churn-aware TTL

### Thread Safety

All systems are designed for concurrent access:
- Clone index uses concurrent data structures
- Learning-to-stop is stateless per query
- Diversity calculations are independent
- TTL system uses atomic cache operations

## 📚 References

- **TODO.md**: Complete specification of performance requirements
- **Research Papers**: iSMELL (75.17% F1), isotonic reranking, RAPTOR
- **Performance Targets**: All SLA requirements validated in comprehensive test suite
- **Embedder Independence**: Systems work with any embedding model

## 🤝 Contributing

When contributing to optimization systems:

1. **Maintain SLA Compliance**: All changes must pass SLA validation tests
2. **Embedder Agnostic**: No dependencies on specific embedding models
3. **Performance First**: Changes should improve or maintain performance budgets
4. **Comprehensive Testing**: Add tests for new functionality and edge cases
5. **Documentation**: Update this README for any API or behavior changes

### Performance Testing

```bash
# Run performance validation
npm test src/optimizations/__tests__/integration.test.ts

# Validate SLA compliance  
npm run test:sla-validation

# Benchmark against baseline
npm run benchmark:optimization-systems
```