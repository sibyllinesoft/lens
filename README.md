# Lens - Local Sharded Code Search System

**AI-Native Architecture | Validated by Arbiter | Ready for Implementation**

Lens is a high-performance, local sharded indexing and query service with three processing layers: **lexical+fuzzy**, **symbol/AST**, and **semantic rerank**. Built for sub-20ms p95 latency with first-class observability and paper-grade benchmarking.

## 🏗️ Architecture Overview

### Three-Layer Processing Pipeline

1. **Layer 1 - Lexical+Fuzzy (Fast Recall)**
   - N-gram/trigram inverted index + FST-based fuzzy search
   - ≤2-edit distance with camelCase/snake_case subtokenization
   - Target: 2-8ms (Stage-A)
   - Based on Zoekt/GitHub Blackbird patterns

2. **Layer 2 - Symbol/AST (Precision)**  
   - Definitions/references via **universal-ctags** and **LSIF**
   - Structural selectors via **tree-sitter** (incremental parse)
   - Optional ast-grep/comby for pattern queries
   - Target: 3-10ms (Stage-B)

3. **Layer 3 - Semantic Rerank (Quality)**
   - Rerank top-K (≤200) with **ColBERT-v2** or **SPLADE-v2**
   - ColBERT vectors in **HNSW** for fast retrieval
   - Target: 5-15ms (Stage-C)
   - **Optional and strictly rerank-only** to bound p95

### Process Model

- **Single daemon** with ingest/index, query, maintenance pools
- Work units flow through **NATS/JetStream** as `(repo_sha, shard)`
- **Append-only, memory-mapped** segment files with periodic compaction
- Hot shards stay in OS page cache for performance

## 🎯 Performance Targets (Validated by Arbiter)

```
Stage-A (lexical+fuzzy):    2-8ms
Stage-B (symbol/AST):       3-10ms  
Stage-C (semantic rerank):  5-15ms
Overall p95 latency:        <20ms
Max candidates for rerank:  ≤200
```

## 🛡️ Architectural Constraints

The `architecture.cue` file contains **bulletproof constraints** validated by Arbiter that prevent:

- **Performance violations** (SLA breaches, resource exhaustion)
- **Resource overallocation** (memory limits, worker pool bounds) 
- **API contract violations** (invalid requests, unbounded queries)
- **Technology stack drift** (consistent tool choices)

### Key Constraints Enforced

```cue
performance: {
    stage_a_target_ms: int & >=2 & <=8      // Lexical+fuzzy bounds
    overall_p95_ms: int & <=20              // End-to-end SLA
    max_candidates: int & >=50 & <=200      // Rerank limits
}

resources: {
    memory_limit_gb: int & >=4 & <=64       // Memory bounds
    max_concurrent_queries: int & >=10 & <=1000
    worker_pools: {
        ingest: int & >=2 & <=16            // NATS workers
        query: int & >=4 & <=32             // Query parallelism
        maintenance: int & >=1 & <=4        // Background work
    }
}
```

## 📋 API Specifications

All endpoints are fully specified with request/response schemas and SLA targets:

### Core Endpoints

- **`POST /search`** - Main search with lexical, structural, and hybrid modes (20ms SLA)
- **`POST /struct`** - AST/structural pattern search (30ms SLA) 
- **`POST /symbols/near`** - Find definitions/references around location (15ms SLA)
- **`GET /health`** - System health and shard status (5ms SLA)

### Query Flow

```
Stage-A fan-out (exact+fuzzy+subtoken) 
    ↓
Stage-B filters (symbol proximity, structural hits)
    ↓  
Stage-C rerank (optional semantic scoring)
    ↓
Stream top-N with reasons: {file, line, col, ast_path?, score, why:[...]}
```

## 🔧 Technology Stack (Validated)

- **Languages**: TypeScript, Python, Rust, Bash
- **Messaging**: NATS/JetStream for work distribution
- **Storage**: Memory-mapped segments, append-only with compaction
- **Observability**: OpenTelemetry (spans, metrics, traces)
- **Semantic Models**: ColBERT-v2, SPLADE-v2 (optional)
- **Vector Search**: HNSW or ScaNN for semantic rerank

## 📊 Observability & Monitoring

Full OpenTelemetry integration with:

- **Distributed tracing** for each query stage
- **Performance metrics** (latency p50/p95, candidate fan-outs)
- **Cache hit ratios** and shard load balancing
- **Structured logging** with trace IDs for debugging
- Live `/metrics` and `/debug/trace` endpoints

## 🧪 Benchmark Harness

Paper-grade evaluation system with:

- **Tasks**: Code search (natural language + identifier near-miss), structural search, docs→answer passage
- **Metrics**: Recall@{10,50}, nDCG@10, MRR, latency per stage, first-relevant-tokens
- **Statistical rigor**: Paired randomization tests + bootstrap CIs
- **Ablation studies**: No fuzzy, no symbols, no rerank comparisons
- **Baselines**: BM25/lexical, lexical+symbols, +ColBERT-v2, +SPLADE
- **Output**: Quarto/LaTeX PDF with Abstract/Methods/Results/Discussion

## 🚀 Implementation Roadmap

### Phase 1: Foundation (Skeleton + Layer-1)
- [ ] Implement sharded trigram+FST index
- [ ] Basic `/search` endpoint with lexical matching
- [ ] OpenTelemetry tracing integration
- [ ] NATS/JetStream work distribution
- [ ] Memory-mapped segment storage

### Phase 2: Symbol/AST Layer (Layer-2)
- [ ] Universal-ctags integration for definitions/references
- [ ] LSIF support for cross-reference data
- [ ] Tree-sitter incremental parsing
- [ ] `/struct` and `/symbols/near` endpoints
- [ ] AST-based structural search

### Phase 3: Semantic Rerank (Layer-3)
- [ ] ColBERT-v2 integration with HNSW
- [ ] SPLADE-v2 alternative implementation  
- [ ] Top-K reranking pipeline (flagged/optional)
- [ ] Performance validation against SLA targets

### Phase 4: Production Ready
- [ ] Comprehensive benchmark harness
- [ ] Performance tuning and optimization
- [ ] Production monitoring and alerting
- [ ] Documentation and deployment guides

## 🤖 AI Agent Implementation Guide

This project is **AI-agent ready** with the following guardrails:

### 1. Use the CUE Specification
```bash
# Validate any configuration changes
cue eval architecture.cue

# Check production config compliance
cue export architecture.cue --expression lens_production
```

### 2. Respect Performance Constraints
- All implementations must stay within validated SLA bounds
- Use the constraint system to validate resource allocation
- Monitor against the defined performance targets

### 3. Follow API Contracts
- Request/response schemas are fully specified
- All endpoints have defined SLA targets
- Trace ID propagation is required

### 4. Maintain Technology Stack Consistency
- Use only validated languages: TypeScript, Python, Rust, Bash
- NATS/JetStream for messaging (no alternatives)
- OpenTelemetry for all observability

### 5. Implement Incrementally
- Follow the 4-phase roadmap
- Validate each layer independently
- Maintain performance budgets throughout

## 🏁 Getting Started

1. **Validate the architecture**: `cue eval architecture.cue`
2. **Choose your implementation language** (TS/Python/Rust/Bash)
3. **Set up NATS/JetStream** for work distribution
4. **Implement Phase 1** (trigram index + basic search)
5. **Add OpenTelemetry** tracing from day 1
6. **Validate against performance targets** continuously

## 📄 License

MIT License - Build amazing search experiences!

---

**This specification was validated by Arbiter and is guaranteed to prevent common architectural mistakes. AI agents implementing this system will stay perfectly "on rails" while building a production-grade code search engine.**