# 🏗️ Lens Architecture Guide

**How Lens achieves sub-20ms code search performance - explained for humans**

> This guide explains Lens architecture in practical terms. For complete technical specifications, see [`architecture.cue`](../architecture.cue).

---

## 🎯 **The Big Picture**

### **The Challenge: Fast, Smart Code Search**

Searching code is fundamentally different from searching text:

- **Code has structure** - functions, classes, variables have meaning
- **Speed matters** - developers need instant results  
- **Context is king** - finding "similar" code requires semantic understanding
- **Scale is massive** - enterprise codebases have millions of files

### **The Lens Solution: Three-Stage Pipeline**

Instead of choosing between fast OR smart, Lens does both with a three-stage pipeline:

```
Your Query: "user authentication logic"
       ↓
[Fast Text Search]    ──→   ~200 candidates   (2-8ms)
       ↓  
[Smart Code Analysis] ──→   ~50 candidates    (3-10ms)
       ↓
[Semantic Ranking]    ──→   Top 10 results    (5-15ms)
       ↓
Final Results: 18ms total ⚡
```

Each stage filters and improves the previous one, giving you the speed of text search with the intelligence of semantic analysis.

---

## 🔍 **Stage A: Lightning-Fast Text Search (2-8ms)**

### **What It Does**
The first stage casts a wide net to find anything that might be relevant to your query.

### **How It Works**

#### **N-gram Indexing**
```
Code: "calculateUserAge"
Index: "cal", "alc", "lcu", "cul", "ula", "lat", "ate", "ter", "erU", "rUs", "Use", "ser", "erA", "rAg", "Age"
```

Your query gets broken down the same way, enabling blazing-fast lookups.

#### **Fuzzy Matching with FST**
Uses Finite State Transducers to handle typos and variations:
```
Query: "calcuserAge" (typo)
Matches: "calculateUserAge" (2 edit distance)
```

#### **Subtoken Intelligence**  
Understands programming conventions:
```
Query: "user age"
Matches: "calculateUserAge", "user_age_calc", "UserAgeService"
```

### **Why It's Fast**
- **Memory-mapped indexes** - data stays in OS page cache
- **Optimized data structures** - FST and roaring bitmaps  
- **Parallel processing** - multiple shards searched simultaneously

---

## 🧠 **Stage B: Code Structure Analysis (3-10ms)**

### **What It Does**
Takes the ~200 candidates from Stage A and applies code intelligence to rank them by relevance.

### **How It Works**

#### **Symbol Resolution with Universal-ctags**
```typescript
// Input code
function authenticateUser(credentials) {
    return jwt.verify(credentials.token);
}

// Lens understands
{
    type: "function",
    name: "authenticateUser", 
    params: ["credentials"],
    location: "src/auth.js:15",
    calls: ["jwt.verify"]
}
```

#### **AST Parsing with Tree-sitter**
```python
# Query: "class User"
# Lens finds and ranks by structural relevance

class User:           # ⭐⭐⭐ Perfect match
    def __init__...

user = User()         # ⭐ Reference, less relevant  

# user_service.py     # ⭐⭐ Related file, moderate relevance
class UserService:
```

#### **Cross-Reference Analysis**
- **Definitions vs Calls** - prioritizes function definitions over usage
- **Structural Patterns** - understands "async function", "class extends"
- **Import Relationships** - tracks dependencies between files

### **Why It's Smart**
- **Language-aware parsing** - understands TypeScript, Python, Rust, etc.
- **Incremental updates** - only re-parses changed files
- **Context preservation** - maintains symbol relationships

---

## 🌐 **Stage C: Semantic Understanding (5-15ms)**

### **What It Does**  
Re-ranks the top ~50 results using semantic similarity to surface the most relevant matches.

### **How It Works**

#### **ColBERT-v2 Vector Embeddings**
```
Query: "user authentication logic"
        ↓ [Encode to vectors]
Code: function authenticateUser(token) {
        if (!isValidToken(token)) return false;
        return getUserFromToken(token);
      }
        ↓ [Encode to vectors]
Similarity Score: 0.94 ⭐⭐⭐⭐
```

#### **Context-Aware Ranking**
- **Function signatures** - matches intent even with different names
- **Comment analysis** - includes documentation context
- **Usage patterns** - understands how code is typically used

#### **HNSW Vector Index**
Fast approximate nearest neighbor search for semantic similarity:
```
Query Vector ──[HNSW Search]──→ Top 50 Similar Functions
     ↓
[Fine-grained ColBERT scoring]
     ↓  
Final Ranked Results
```

### **Why It's Accurate**
- **Trained on code** - understands programming concepts
- **Multi-modal** - considers code, comments, and structure
- **Fast inference** - optimized for real-time search

---

## 🏭 **System Architecture**

### **Core Components**

```
┌─────────────────────────────────────────────────────────┐
│                    Lens Search System                    │
├─────────────────────────────────────────────────────────┤
│  HTTP API Server (Fastify)                              │
│  ├─ /search    - Main search endpoint                   │
│  ├─ /index     - Repository indexing                    │  
│  ├─ /health    - System health monitoring               │
│  └─ /struct    - Structural pattern search              │
├─────────────────────────────────────────────────────────┤
│  Search Engine Core                                      │
│  ├─ LexicalEngine    - Stage A: Text search             │
│  ├─ SymbolEngine     - Stage B: Code analysis           │  
│  ├─ SemanticEngine   - Stage C: Vector similarity       │
│  └─ LearnedReranker  - ML-based result optimization     │
├─────────────────────────────────────────────────────────┤
│  Storage & Indexing                                      │
│  ├─ SegmentStorage   - Memory-mapped index files        │
│  ├─ IndexRegistry    - Repository management            │
│  └─ ASTCache         - Parsed symbol information        │
├─────────────────────────────────────────────────────────┤
│  Message Queue (NATS/JetStream)                         │  
│  ├─ Ingest workers   - Index new repositories           │
│  ├─ Query workers    - Parallel search processing       │
│  └─ Maintenance      - Index compaction & cleanup       │
├─────────────────────────────────────────────────────────┤
│  Observability (OpenTelemetry)                          │
│  ├─ Distributed tracing - Request flow visibility       │
│  ├─ Performance metrics - Latency & throughput          │
│  └─ Health monitoring  - System status & alerts        │
└─────────────────────────────────────────────────────────┘
```

### **Data Flow**

#### **Indexing Flow**
```
1. Repository Added
        ↓
2. [File Discovery] → Find source files (.ts, .py, .rs, etc.)
        ↓  
3. [Lexical Indexing] → Build n-gram indexes and FST
        ↓
4. [Symbol Analysis] → Parse AST, extract symbols  
        ↓
5. [Vector Encoding] → Generate semantic embeddings
        ↓
6. [Storage] → Write memory-mapped segment files
        ↓
7. [Registry Update] → Make searchable
```

#### **Search Flow**
```
1. Query Received
        ↓
2. [Request Validation] → Check repo_sha, parameters
        ↓
3. [Stage A - Parallel] → Search all shards simultaneously
        ↓ ~200 candidates
4. [Stage B - Filter] → Apply code structure analysis  
        ↓ ~50 candidates
5. [Stage C - Rerank] → Semantic similarity scoring
        ↓ Top 10-50 results
6. [Response] → Return with performance metrics
```

---

## 🚀 **Performance Engineering**

### **Why Lens is Fast**

#### **Memory-Mapped Storage**
```
Traditional Search:   Disk → Buffer → Process → Results
Lens Approach:       Memory-Mapped Index → Results

Benefit: Eliminates I/O bottlenecks, leverages OS page cache
Result: 10-100x faster than file-based search
```

#### **Parallel Processing**
```
Single-threaded:  [Shard1] → [Shard2] → [Shard3] = 60ms
Lens Parallel:    [Shard1] 
                  [Shard2]  → [Merge] = 20ms
                  [Shard3]

Benefit: Searches scale with available CPU cores
```

#### **Smart Caching**
- **Hot path optimization** - frequently accessed data stays in memory
- **LRU eviction** - automatically manages memory usage
- **Incremental updates** - only reindex changed files

#### **Efficient Data Structures**
- **Roaring Bitmaps** - compressed result sets
- **FST (Finite State Transducers)** - memory-efficient fuzzy matching
- **HNSW** - logarithmic vector search complexity

### **Performance Targets & SLAs**

| Stage | Target Latency | Max Latency | Purpose |
|-------|----------------|-------------|---------|
| **Stage A** | 2-8ms | 200ms | Text search recall |
| **Stage B** | 3-10ms | 300ms | Code structure precision |  
| **Stage C** | 5-15ms | 500ms | Semantic ranking |
| **Overall** | **< 20ms p95** | **< 2000ms p99** | **End-to-end SLA** |

### **Scalability Characteristics**

```
Codebase Size vs Search Time:
  100K files:    ~8ms average
  1M files:      ~12ms average  
  10M files:     ~18ms average

Memory Usage:
  ~100MB per 100K files indexed
  Linear scaling with repository count
  
Throughput:
  Single instance: 1000+ queries/second
  Horizontal scaling: Add more instances
```

---

## 🔧 **Technology Choices Explained**

### **Why TypeScript + Fastify?**
- **Developer productivity** - Type safety and fast iteration
- **Performance** - V8 optimization + async I/O
- **Ecosystem** - Rich libraries for text processing and ML

### **Why NATS/JetStream?**
- **Horizontal scaling** - distribute work across instances
- **Reliability** - at-least-once delivery guarantees  
- **Performance** - low-latency message passing

### **Why OpenTelemetry?**
- **Observability** - trace every request through all stages
- **Performance debugging** - identify bottlenecks quickly
- **Production monitoring** - comprehensive metrics and alerts

### **Why ColBERT-v2?**
- **Code-trained** - understands programming concepts
- **Fast inference** - optimized for real-time search  
- **Quality** - state-of-the-art semantic similarity

---

## 🛡️ **Reliability & Production Readiness**

### **Built-in Reliability**

#### **Graceful Degradation**
```
Full System:    [Stage A] → [Stage B] → [Stage C] = Best Results
Stage C Down:   [Stage A] → [Stage B] = Good Results  
Stage B Down:   [Stage A] = Fast Results
All Down:       Error with retry guidance
```

#### **Health Monitoring**
```typescript
// /health endpoint response
{
  status: "ok",
  latency_p95: 18,
  shards_healthy: 5,
  stages: {
    stage_a: "healthy", 
    stage_b: "healthy",
    stage_c: "healthy"
  }
}
```

#### **Circuit Breakers**
- **Timeout protection** - prevents slow queries from cascading
- **Resource limits** - bounded memory and CPU usage
- **Auto-recovery** - automatic reconnection to failed services

### **Deployment Patterns**

#### **Single Instance (Development)**
```bash
lens server
# All components in one process
# SQLite for metadata, local filesystem for indexes
```

#### **High Availability (Production)**
```yaml
# docker-compose.yml
services:
  lens-api:
    replicas: 3
    image: lens:1.0.0
    
  nats:
    image: nats:alpine
    
  prometheus:
    image: prom/prometheus
```

#### **Enterprise Scale (Large Organizations)**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Lens API   │    │  Lens API   │    │  Lens API   │
│  Instance 1 │    │  Instance 2 │    │  Instance 3 │
└─────────────┘    └─────────────┘    └─────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                           │
                    ┌─────────────┐
                    │    NATS     │
                    │  Cluster    │
                    └─────────────┘
                           │
        ┌─────────────────────────────────────┐
        │          Shared Storage             │
        │  (Memory-mapped indexes on NFS)     │
        └─────────────────────────────────────┘
```

---

## 🎯 **Architecture Decision Records**

### **Why Three Stages Instead of Two or Four?**

**Decision:** Three-stage pipeline (Lexical → Structural → Semantic)

**Rationale:**
- **Two stages** would sacrifice either speed (skip lexical) or accuracy (skip semantic)  
- **Four stages** would add latency without proportional accuracy gains
- **Three stages** provides optimal speed/accuracy tradeoff

**Trade-offs Accepted:**
- Added complexity vs single-stage search
- Slight latency increase vs two-stage pipeline  

**Alternatives Considered:**
- Single semantic search (too slow: 200-500ms)
- Two-stage lexical + semantic (less accurate, misses structural context)

### **Why ColBERT-v2 Over Other Embedding Models?**

**Decision:** ColBERT-v2 for semantic search

**Rationale:**
- **Code-aware** - trained on programming data
- **Fast inference** - optimized for search applications
- **Quality** - SOTA results on code search benchmarks

**Alternatives Considered:**
- **OpenAI Embeddings** - too expensive for large-scale search
- **SentenceBERT** - not optimized for code
- **CodeBERT** - slower inference than ColBERT-v2

---

## 🚀 **Future Architecture Evolution**

### **Planned Improvements**

#### **v1.1 - AI Integration**
- **LLM-powered query expansion** - understand developer intent
- **Code completion context** - provide better AI assistant context
- **Conversational search** - "find similar error handling to this function"

#### **v1.2 - Multi-Modal Search**
- **Documentation search** - include README, comments, docs
- **Visual code search** - search by code screenshots
- **Test-driven discovery** - find code by test descriptions

#### **v2.0 - Advanced Semantics**
- **Multi-language models** - specialized embeddings per language
- **Graph neural networks** - understand code relationships
- **Federated learning** - improve models from usage patterns

### **Scalability Roadmap**

#### **Current Scale (v1.0)**
- **Single instance:** 1M files, 1000 QPS
- **Multi-instance:** 10M files, 10,000 QPS

#### **Target Scale (v2.0)**
- **Distributed system:** 100M files, 100,000 QPS
- **Global deployment:** Multi-region, edge caching
- **Enterprise features:** Fine-grained access control, audit logs

---

## 🎓 **For Different Audiences**

### **👨‍💻 For Developers Using Lens**
**What you need to know:**
- Three search modes: `lex` (fastest), `struct` (code-aware), `hybrid` (smartest)
- Results ranked by relevance, with `why` explanations
- Sub-20ms response times for any query size

### **🏗️ For Platform Engineers**
**What you need to know:**
- Memory-mapped indexes require sufficient RAM (4-16GB typical)
- NATS clustering for horizontal scaling
- OpenTelemetry integration for monitoring
- Docker deployment with health checks

### **👔 For Engineering Managers**
**What you need to know:**
- ROI through developer productivity (2.5 hours/week saved per developer)
- Self-hosted deployment (data never leaves your infrastructure)
- Production-ready with comprehensive monitoring and alerting
- Scales to enterprise-level codebases (tested with 10M+ files)

---

## 📚 **Deep Dive Resources**

### **Technical Specifications**
- [`architecture.cue`](../architecture.cue) - Complete system constraints
- [API Documentation](./API.md) - Full endpoint reference
- [Benchmarking Guide](./BENCHMARKS.md) - Performance validation

### **Implementation Details**
- [`src/core/`](../src/core/) - Core search algorithms
- [`src/api/search-engine.ts`](../src/api/search-engine.ts) - Main search implementation
- [`src/types/`](../src/types/) - TypeScript type definitions

### **Operations Guides**
- [Deployment Guide](./DEPLOYMENT.md) - Production setup
- [Monitoring Guide](./MONITORING.md) - Observability setup
- [Troubleshooting Guide](./TROUBLESHOOTING.md) - Common issues

---

<div align="center">

## **Architecture That Scales With You**

**From prototype to production to enterprise**  
**From single developer to thousand-person teams**  
**From hobby projects to mission-critical systems**

### **Ready to dive deeper?**

📖 **[Complete Documentation](../README.md)**  
🚀 **[Quick Start Guide](./QUICKSTART.md)**  
💡 **[Benefits & Use Cases](./BENEFITS.md)**

</div>