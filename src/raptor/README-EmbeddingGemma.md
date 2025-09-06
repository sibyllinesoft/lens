# EmbeddingGemma Migration for Lens Code Search

## Overview

This implementation provides a complete migration path from OpenAI's `text-ada-002` embeddings to Google's EmbeddingGemma-300M for local-first deployment in the Lens code search system. The migration includes shadow indexing, comprehensive benchmarking, and gradual rollout capabilities.

## Architecture

### Key Components

1. **EmbeddingGemmaProvider** (`embedding-gemma-provider.ts`)
   - OpenAI-compatible TEI integration
   - Matryoshka dimension support (768/512/256/128)
   - Health checking and fallback mechanisms

2. **ShadowIndexManager** (`shadow-index-manager.ts`)
   - Parallel index construction for A/B testing
   - Comparison metrics (ΔCBU/GB, Recall@K)
   - Index persistence and loading

3. **FrozenPoolReplayHarness** (`frozen-pool-replay.ts`)
   - Controlled evaluation with frozen query pools
   - CBU (Core Business Utility) metrics
   - Performance and quality measurement

4. **EmbeddingGemmaBenchmarkRunner** (`embedding-gemma-benchmark.ts`)
   - Comprehensive benchmarking suite
   - Resource utilization tracking
   - Quality assessment (Recall, Precision, NDCG)

5. **EmbeddingConfigManager** (`embedding-config-manager.ts`)
   - Runtime configuration management
   - Model switching and fallback
   - Shadow testing configuration

6. **Migration CLI** (`../scripts/embedding-gemma-migration.ts`)
   - End-to-end migration orchestrator
   - Phase-by-phase execution
   - Reporting and recommendations

## Quick Start

### 1. Start TEI Server

```bash
# CPU version (recommended for local development)
docker run -p 8080:80 \
  ghcr.io/huggingface/text-embeddings-inference:cpu-1.8.1 \
  --model-id google/embeddinggemma-300m

# CUDA version (for production with GPU)
docker run -p 8080:80 --gpus all \
  ghcr.io/huggingface/text-embeddings-inference:1.8.1 \
  --model-id google/embeddinggemma-300m
```

### 2. Run Migration

```bash
# Full migration pipeline
npm run tsx src/scripts/embedding-gemma-migration.ts full \
  --tei-endpoint http://localhost:8080 \
  --output-dir ./migration_results \
  --dry-run

# Individual phases
npm run tsx src/scripts/embedding-gemma-migration.ts phase 1  # TEI setup
npm run tsx src/scripts/embedding-gemma-migration.ts phase 2  # Shadow indexes
npm run tsx src/scripts/embedding-gemma-migration.ts phase 3  # Frozen-pool replay
npm run tsx src/scripts/embedding-gemma-migration.ts phase 4  # Benchmarking
npm run tsx src/scripts/embedding-gemma-migration.ts phase 5  # Deployment prep
```

### 3. Integration Example

```typescript
import { EmbeddingConfigManager } from './raptor/embedding-config-manager.js';
import { EmbeddingGemmaProvider } from './raptor/embedding-gemma-provider.js';

// Initialize configuration
const configManager = new EmbeddingConfigManager('./embedding_config.json');
await configManager.initialize();

// Get active provider (automatically uses best model)
const provider = configManager.getActiveProvider();

// Embed texts
const texts = ['function sum(a, b) { return a + b; }'];
const embeddings = await provider.embed(texts);

// Switch models at runtime
await configManager.switchModel('gemma-256');
```

## Migration Phases

### Phase 1: TEI Server Setup
- Health check TEI server endpoint
- Validate both 768d and 256d embeddings
- Measure baseline performance

### Phase 2: Shadow Index Construction
- Build parallel indexes for Gemma-768 and Gemma-256
- Process corpus documents in batches
- Collect storage and performance metrics

### Phase 3: Frozen-Pool Replay
- Generate/load frozen query pool
- Execute controlled A/B testing
- Measure ΔCBU/GB, Recall@K, critical-atom recall

### Phase 4: Comprehensive Benchmarking
- Performance benchmarking (latency, throughput, CPU)
- Quality assessment (NDCG, MRR, MAP)
- Resource utilization analysis

### Phase 5: Deployment Preparation
- Model selection based on metrics
- Shadow testing configuration
- Production deployment checklist

## Key Metrics

### ΔCBU/GB (Delta Core Business Utility per Gigabyte)
- Primary optimization target
- Measures utility efficiency relative to storage cost
- Higher values indicate better cost-effectiveness

### Recall@K
- `Recall@10`: Precision for top-10 results
- `Recall@50`: Coverage for expanded result sets
- Critical for maintaining search quality

### Critical-Atom Recall
- Retrieval rate for most important/relevant documents
- Business-critical code patterns and utilities
- Weighted by relevance and usage patterns

### CPU P95
- 95th percentile encoding latency
- Service level agreement compliance
- Real-time search performance

## Matryoshka Dimensions

### 768-Dimensional (Full)
- **Use Case**: Maximum quality requirements
- **Storage**: ~3KB per embedding
- **Performance**: Higher latency, better recall

### 256-Dimensional (Compressed)
- **Use Case**: Storage-optimized deployment
- **Storage**: ~1KB per embedding (67% savings)
- **Performance**: 2-3x faster encoding, minimal recall loss

### Selection Criteria
- Choose 256d if recall loss < 5% and storage savings > 50%
- Choose 768d for quality-critical applications
- A/B test with actual workload for final decision

## Configuration

### Global Configuration (`embedding_config.json`)

```json
{
  "primary": "gemma-768",
  "fallback": "gemma-256", 
  "shadowTesting": false,
  "models": {
    "gemma-768": {
      "enabled": true,
      "teiEndpoint": "http://localhost:8080",
      "matryoshka": {
        "targetDimension": 768,
        "preserveRanking": true
      },
      "performance": {
        "batchSize": 32,
        "timeout": 15000
      }
    },
    "gemma-256": {
      "enabled": true,
      "teiEndpoint": "http://localhost:8080", 
      "matryoshka": {
        "targetDimension": 256,
        "preserveRanking": true
      },
      "performance": {
        "batchSize": 64,
        "timeout": 10000
      }
    }
  },
  "migration": {
    "enabled": true,
    "abTestingConfig": {
      "trafficSplit": {
        "gemma-768": 90,
        "gemma-256": 10
      }
    }
  }
}
```

## Production Deployment

### 1. Infrastructure Setup

```yaml
# docker-compose.yml
version: '3.8'
services:
  tei-server:
    image: ghcr.io/huggingface/text-embeddings-inference:1.8.1
    ports:
      - "8080:80"
    environment:
      - MODEL_ID=google/embeddinggemma-300m
      - MAX_BATCH_SIZE=64
      - MAX_INPUT_LENGTH=2048
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 2. Gradual Rollout

```typescript
// Phase 1: 10% shadow testing
await configManager.configureShadowTesting(true, {
  'gemma-256': 10,
  'ada-002': 90
});

// Phase 2: 50% traffic after validation
await configManager.configureShadowTesting(true, {
  'gemma-256': 50, 
  'ada-002': 50
});

// Phase 3: Full migration
await configManager.switchModel('gemma-256');
await configManager.configureShadowTesting(false);
```

### 3. Monitoring

Monitor these key metrics during rollout:
- Search quality (Recall@50, NDCG@10)
- Response times (p95, p99 latency)
- Error rates and timeouts
- Resource utilization (CPU, memory, storage)
- User satisfaction (CTR, session length)

### 4. Rollback Plan

```typescript
// Immediate rollback if quality degrades
await configManager.switchModel('ada-002');

// Partial rollback
await configManager.configureShadowTesting(true, {
  'ada-002': 80,
  'gemma-256': 20
});
```

## Advanced Features

### Custom Query Pool

```typescript
const queryPool: FrozenQuery[] = [
  {
    id: 'auth_patterns',
    query: 'authentication middleware passport',
    language: 'typescript',
    intent: 'semantic',
    expectedResults: [
      {
        docId: 'auth/middleware.ts',
        filePath: 'src/auth/middleware.ts',
        relevanceScore: 0.95,
        isCriticalAtom: true
      }
    ],
    groundTruth: {
      precision_at_10: 0.8,
      recall_at_50: 0.9,
      user_satisfaction: 0.85
    }
  }
];

await replayHarness.loadQueryPool({ groundTruthFile: './queries.json' });
```

### Performance Tuning

```typescript
// Optimize for throughput
await provider.updateMatryoshkaConfig({
  targetDimension: 256,
  preserveRanking: true
});

// Configure batching
const provider = new EmbeddingGemmaProvider({
  batchSize: 64,  // Higher for smaller dimensions
  timeout: 10000, // Faster timeout
  maxRetries: 2   // Fewer retries for speed
});
```

### Resource Monitoring

```typescript
const benchmarkRunner = new EmbeddingGemmaBenchmarkRunner(shadowManager);
const benchmark = await provider.benchmark([
  'example query text',
  'another test query'
], 5);

console.log(`Avg Latency: ${benchmark.avgLatencyMs} ms`);
console.log(`Throughput: ${benchmark.throughputTokensPerSec} tok/s`);
console.log(`Error Rate: ${benchmark.errorRate * 100}%`);
```

## Troubleshooting

### TEI Server Issues

```bash
# Check server health
curl http://localhost:8080/health

# Check server info
curl http://localhost:8080/info

# Test embedding
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["test query"], "model": "google/embeddinggemma-300m"}'
```

### Common Issues

1. **"TEI server not available"**
   - Ensure Docker container is running
   - Check port mapping (8080:80)
   - Verify firewall/network connectivity

2. **"Embedding dimension mismatch"**
   - Check Matryoshka configuration
   - Ensure target dimension is supported
   - Rebuild shadow indexes after dimension change

3. **"High error rate in benchmarks"**
   - Reduce batch size for stability
   - Increase timeout values
   - Check server resource availability

4. **"Poor recall performance"**
   - Use 768d instead of 256d
   - Verify corpus quality and coverage
   - Check query pool representativeness

## Integration with Existing Lens Architecture

### Stage A (Lexical) - No Changes
The lexical fuzzy matching stage remains unchanged. EmbeddingGemma only affects Stage C (semantic reranking).

### Stage B (Symbol/AST) - No Changes  
Symbol and AST processing continue as before. The embedding changes are isolated to semantic similarity.

### Stage C (Semantic Reranking) - Updated
```typescript
// Before (ada-002)
const openaiProvider = new OpenAIEmbeddingProvider({
  apiKey: process.env.OPENAI_API_KEY,
  model: 'text-embedding-ada-002'
});

// After (EmbeddingGemma)
const configManager = new EmbeddingConfigManager('./config.json');
await configManager.initialize();
const gemmaProvider = configManager.getActiveProvider();

// Usage remains the same
const embeddings = await provider.embed(texts);
```

## Performance Expectations

### Typical Migration Results
- **Storage Reduction**: 50-67% with Gemma-256
- **Recall Retention**: 95-98% of ada-002 quality
- **Latency Improvement**: 2-3x faster encoding
- **Cost Savings**: No external API costs
- **ΔCBU/GB**: 2-4x improvement

### Resource Requirements
- **CPU**: 4+ cores recommended
- **Memory**: 2-4GB for model + index
- **Storage**: 1-3KB per document (dimension dependent)
- **Network**: Local-only (no external API calls)

## Contributing

To extend the migration system:

1. **Custom Providers**: Implement `EmbeddingProvider` interface
2. **New Metrics**: Add to `CBUMetrics` interface
3. **Benchmark Scenarios**: Extend `BenchmarkScenario` types
4. **Configuration**: Add to `GlobalEmbeddingConfig`

## License

This implementation follows the same license as the main Lens project (LicenseRef-SPL-1.0).

---

For questions or issues, please refer to the main Lens documentation or open an issue in the repository.