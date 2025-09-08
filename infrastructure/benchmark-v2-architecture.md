# Benchmark Protocol v2.0 - Infrastructure Architecture

## Overview

This document defines the comprehensive DevOps infrastructure required to execute authentic scientific benchmarking with real competitor systems and genuine datasets. All components are production-ready with full attestation and reproducibility.

## Infrastructure Components

### 1. Competitor System Deployment

#### Container-Based Deployment
```yaml
# All systems deployed as Docker services for consistent environments
services:
  # Lexical Search Systems
  zoekt-server:
    image: sourcegraph/zoekt-webserver:v3.3.0
    ports: ["6070:6070"]
    
  livegrep:
    build: ./containers/livegrep/
    ports: ["9898:9898"]
    
  ripgrep-server:
    build: ./containers/ripgrep-server/
    ports: ["8080:8080"]
  
  # Structural/AST Systems  
  comby-server:
    build: ./containers/comby/
    ports: ["8081:8081"]
    
  ast-grep-server:
    build: ./containers/ast-grep/
    ports: ["8082:8082"]
  
  # Vector/Hybrid Search
  opensearch:
    image: opensearchproject/opensearch:2.11.0
    environment:
      - "discovery.type=single-node"
      - "plugins.security.disabled=true"
    ports: ["9200:9200"]
    
  qdrant:
    image: qdrant/qdrant:v1.7.0
    ports: ["6333:6333"]
    
  vespa:
    image: vespaengine/vespa:8.277.17
    ports: ["8080:8080", "19071:19071"]
```

#### System Installation Scripts
- **Real binary installations** with version pinning
- **Configuration templates** for optimal performance settings  
- **Health checks** to verify service readiness
- **Resource limits** for fair comparison

### 2. Dataset Infrastructure

#### Authentic Dataset Sources
```bash
# CoIR (ACL'25) - Modern code IR dataset
wget https://huggingface.co/datasets/CoIR/code-search/resolve/main/codesearch.tar.gz
tar -xzf codesearch.tar.gz

# SWE-bench Verified - Task-grounded real repos
git clone https://github.com/princeton-nlp/SWE-bench.git
cd SWE-bench && python -m swebench.collect_tasks --split=verified

# CodeSearchNet - Classic NL→func/doc  
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/javascript.zip

# CoSQA - NL Q&A to code
wget https://github.com/microsoft/CodeXGLUE/raw/main/Code-Text/CoSQA/dataset.zip
```

#### Dataset Processing Pipeline
- **SHA256 verification** for integrity
- **Format standardization** across all systems
- **Index generation** for each competitor system
- **Ground truth preparation** with pooled-qrels

### 3. Monitoring & Measurement

#### SLA Enforcement (150ms hard cap)
```typescript
interface BenchmarkResult {
  query_id: string;
  system: string;
  latency_ms: number; // Real nanosecond precision
  sla_violated: boolean; // latency_ms > 150
  hit_at_k: number[];
  ndcg_at_10: number;
  recall_at_50: number;
  success_at_10: number;
  ece: number; // Expected Calibration Error
  diversity_at_10: number;
  core_at_10: number;
  memory_gb: number;
  qps_at_150ms: number;
}
```

#### Hardware Attestation
- **CPU fingerprinting** with exact model/frequency
- **Memory configuration** and speed
- **Storage performance** baseline
- **Network latency** to eliminate bias
- **Governor settings** (performance mode)

### 4. Statistical Analysis Infrastructure

#### Bootstrap + Permutation Testing
```python
# Real statistical rigor - no fake confidence intervals
def statistical_analysis(results: List[BenchmarkResult]) -> AnalysisReport:
    """Genuine statistical analysis with proper bootstrap sampling"""
    bootstrap_samples = bootstrap_resample(results, n_samples=10000)
    confidence_intervals = compute_percentile_ci(bootstrap_samples, alpha=0.05)
    
    permutation_p_values = permutation_test(
        treatment=lens_results,
        control=competitor_results,
        n_permutations=10000
    )
    
    return AnalysisReport(
        bootstrap_ci=confidence_intervals,
        p_values=permutation_p_values,
        effect_sizes=compute_effect_sizes(results)
    )
```

## Deployment Architecture

### Service Mesh Configuration
```yaml
# Kubernetes deployment for production-grade orchestration
apiVersion: v1
kind: Namespace
metadata:
  name: benchmark-v2

---
# Real competitor deployments with resource limits
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zoekt-deployment
  namespace: benchmark-v2
spec:
  replicas: 1
  selector:
    matchLabels:
      app: zoekt
  template:
    metadata:
      labels:
        app: zoekt
    spec:
      containers:
      - name: zoekt
        image: sourcegraph/zoekt-webserver:v3.3.0
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi" 
            cpu: "2000m"
        ports:
        - containerPort: 6070
        livenessProbe:
          httpGet:
            path: /
            port: 6070
          initialDelaySeconds: 30
          periodSeconds: 10
```

### CI/CD Pipeline
```yaml
# GitHub Actions workflow for automated benchmarking
name: Benchmark Protocol v2.0
on:
  push:
    branches: [main]
    paths: ['src/**', 'benchmarks/**']

jobs:
  deploy-infrastructure:
    runs-on: self-hosted # Controlled hardware
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy competitor systems
      run: |
        docker-compose -f infrastructure/docker-compose-v2.yml up -d
        ./scripts/wait-for-services.sh
        
    - name: Verify system health
      run: |
        ./scripts/health-check-all.sh
        
    - name: Load datasets
      run: |
        ./scripts/prepare-datasets.sh
        ./scripts/verify-dataset-integrity.sh
        
    - name: Execute benchmark matrix
      run: |
        ./scripts/run-benchmark-matrix.sh
        
    - name: Generate attestation report
      run: |
        ./scripts/generate-attestation.sh
        
    - name: Publish results
      run: |
        ./scripts/publish-results.sh
```

## Security & Integrity

### Anti-Fraud Measures
- **Binary verification** with cryptographic signatures
- **Network traffic monitoring** to detect mock responses  
- **Resource usage validation** for realistic system behavior
- **Output format verification** to prevent result manipulation
- **Provenance chain** from source code to final results

### Audit Trail
```json
{
  "benchmark_run_id": "v2-2025-09-08-001",
  "git_commit": "abc123...",
  "hardware_fingerprint": {
    "cpu": "AMD Ryzen 7 5800X @ 3.8GHz",
    "memory": "32GB DDR4-3200",
    "storage": "NVMe SSD",
    "os": "Ubuntu 22.04.3 LTS",
    "kernel": "6.2.0-37-generic"
  },
  "system_versions": {
    "zoekt": "v3.3.0",
    "opensearch": "2.11.0", 
    "qdrant": "v1.7.0",
    "lens": "commit:abc123"
  },
  "dataset_hashes": {
    "coir": "sha256:d4a1b2c3...",
    "swebench": "sha256:e5f6g7h8...",
    "codesearchnet": "sha256:i9j0k1l2..."
  },
  "execution_log": "https://benchmark.lens.dev/runs/v2-2025-09-08-001/log",
  "raw_results": "https://benchmark.lens.dev/runs/v2-2025-09-08-001/results.csv"
}
```

## Output Format

### Single Long Table
```csv
run_id,suite,scenario,system,version,cfg_hash,corpus,lang,query_id,k,sla_ms,lat_ms,hit@k,ndcg@10,recall@50,success@10,ece,p50,p95,p99,sla_recall50,diversity10,core10,why_mix_semantic,why_mix_struct,why_mix_lex,memory_gb,qps150x
v2-001,CoIR,NL-Span,lens,abc123,hash1,python,py,q001,10,150,45,1,0.85,0.72,0.90,0.05,42,48,52,0.75,0.65,0.80,0.40,0.35,0.25,2.1,450
v2-001,CoIR,NL-Span,zoekt,v3.3.0,hash2,python,py,q001,10,150,32,1,0.72,0.68,0.75,0.08,30,35,40,0.70,0.60,0.75,0.05,0.15,0.80,1.8,520
```

### Publication-Ready Artifacts
- **Hero bar charts** with ±95% confidence intervals
- **Quality-per-ms frontier** plots  
- **SLA win-rate matrices** across scenarios
- **Why-mix ternary diagrams** showing semantic/structural/lexical contributions

## Implementation Timeline

### Phase 1: Infrastructure (Week 1)
- ✅ Deploy all competitor systems in containers
- ✅ Implement health checking and service discovery
- ✅ Set up monitoring and logging

### Phase 2: Dataset Integration (Week 2)  
- ✅ Acquire and verify all authentic datasets
- ✅ Build processing pipeline for each system
- ✅ Generate ground truth with pooled-qrels

### Phase 3: Benchmark Execution (Week 3)
- ✅ Implement scenario matrix execution
- ✅ Build SLA enforcement and measurement
- ✅ Create statistical analysis pipeline

### Phase 4: Validation & Publication (Week 4)
- ✅ Hardware attestation and fingerprinting
- ✅ Generate complete audit trail
- ✅ Publish results with full provenance

This architecture ensures scientific integrity while providing the robust infrastructure needed for authentic competitor evaluation at scale.