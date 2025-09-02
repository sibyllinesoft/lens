# Lens 🔍
## **Lightning-Fast Code Search That Actually Understands Your Code**

> **Search your entire codebase in under 20ms** - from fuzzy function names to complex semantic queries. Lens combines the speed of traditional text search with the intelligence of semantic understanding.

### ⚡ **Why Lens?**

**Traditional search tools miss the mark:**
- **Text search** is fast but doesn't understand code structure
- **IDE search** is limited to one project at a time  
- **Semantic search** is smart but too slow for real-time use
- **AST tools** are precise but don't handle typos or natural language

**Lens gives you the best of everything:**
- 🏎️ **Sub-20ms responses** for any query size
- 🧠 **Code-aware search** that understands functions, classes, and symbols  
- 🔤 **Fuzzy matching** that handles typos and variations
- 🌐 **Semantic understanding** for natural language queries
- 📊 **Multi-repo support** - search across your entire codebase
- 🛡️ **Production-ready** with comprehensive monitoring and benchmarking

## 🚀 **Quick Start** - Get Running in 2 Minutes

### **1. Install & Start**
```bash
# Install Lens
npm install -g lens@1.0.0

# Start the server 
lens server
# ✅ Server running on http://localhost:3000
```

### **2. Index Your Code**  
```bash
# Index current directory
curl -X POST http://localhost:3000/index \
  -H "Content-Type: application/json" \
  -d '{"repo_path": ".", "repo_sha": "main"}'
# ✅ Indexed 1,247 files in 3.2 seconds
```

### **3. Search Like Magic**
```bash
# Find functions by name (fuzzy matching)
curl -X POST http://localhost:3000/search \
  -d '{"repo_sha": "main", "q": "calcTotal", "mode": "hybrid", "k": 10}'

# Find classes and interfaces
curl -X POST http://localhost:3000/search \
  -d '{"repo_sha": "main", "q": "class User", "mode": "struct", "k": 5}'

# Natural language search  
curl -X POST http://localhost:3000/search \
  -d '{"repo_sha": "main", "q": "authentication logic", "mode": "hybrid", "k": 10}'
```

**Response in < 20ms:**
```json
{
  "hits": [
    {
      "file": "src/auth/User.ts", 
      "line": 15,
      "snippet": "class User implements UserInterface {",
      "score": 0.95,
      "why": ["exact", "symbol"]
    }
  ],
  "total": 1,
  "latency_ms": {
    "stage_a": 4,    // Lightning-fast text search
    "stage_b": 6,    // Code structure analysis  
    "stage_c": 8,    // Semantic reranking
    "total": 18
  }
}
```

---

## 💡 **What Makes Lens Different?**

### **The 3-Stage Intelligence Pipeline**

Lens doesn't just do text search - it **understands your code** through three complementary layers:

#### **🔤 Stage A: Smart Text Search (2-8ms)**
- **Fuzzy matching** - finds `calcTotal` even if you type `calctotal` or `calc_total`
- **Subtoken awareness** - understands camelCase and snake_case conventions
- **N-gram indexing** - blazing fast even on massive codebases

#### **🧠 Stage B: Code Structure (3-10ms)** 
- **Symbol resolution** - knows the difference between function definitions and calls
- **AST parsing** - understands language syntax and structure  
- **Cross-references** - finds where functions are defined vs used

#### **🌐 Stage C: Semantic Understanding (5-15ms)**
- **Natural language** - search for "user authentication logic"
- **Code similarity** - find similar functions even with different names
- **Context-aware ranking** - prioritizes results based on semantic relevance

---

## 🏆 **Use Cases - Real Developers, Real Problems**

### **👨‍💻 For Developers**
```bash
# "Where did I define that helper function?"
curl -X POST localhost:3000/search -d '{"q": "formatDate", "mode": "struct"}'

# "Show me all the places where users are authenticated"  
curl -X POST localhost:3000/search -d '{"q": "user auth", "mode": "hybrid"}'

# "Find similar error handling patterns"
curl -X POST localhost:3000/search -d '{"q": "try catch finally", "mode": "struct"}'
```

### **🏢 For Teams & Organizations**
- **Code reviews**: Find similar implementations across repos
- **Refactoring**: Identify all usages of deprecated functions
- **Onboarding**: Help new team members discover existing code
- **Security audits**: Search for potential security patterns

### **🤖 For AI & Tools**
- **Code completion**: Better context for AI assistants  
- **Documentation**: Auto-generate docs from code discovery
- **Dependency analysis**: Understand code relationships
- **Migration tools**: Find patterns to update automatically

---

## 🛠️ **Language Support**

Lens understands the structure and semantics of multiple programming languages:

| Language | Fuzzy Search | Symbol Recognition | AST Parsing | Semantic Search |
|----------|:------------:|:------------------:|:-----------:|:---------------:|
| **TypeScript/JavaScript** | ✅ | ✅ | ✅ | ✅ |
| **Python** | ✅ | ✅ | ✅ | ✅ |
| **Rust** | ✅ | ✅ | ✅ | ✅ |
| **Go** | ✅ | ✅ | ✅ | ✅ |
| **Java** | ✅ | ✅ | ✅ | ✅ |
| **Bash** | ✅ | ✅ | ✅ | ⚠️ |

> **New languages added regularly** - contribute support for your favorite language!

---

## ⚙️ **Installation Options**

### **NPM (Recommended)**
```bash
# Install globally
npm install -g lens@1.0.0

# Or install locally
npm install lens
npx lens server
```

### **Docker**
```bash
# Quick start with Docker
docker run -p 3000:3000 -v $(pwd):/code lens:1.0.0

# Or with Docker Compose
curl -O https://raw.githubusercontent.com/lens-search/lens/main/docker-compose.yml
docker-compose up
```

### **From Source**
```bash
git clone https://github.com/lens-search/lens.git
cd lens  
npm install
npm run build
npm start
```

---

## 🎛️ **Configuration**

### **Basic Configuration**
```bash
# Environment variables
export LENS_PORT=3000
export LENS_HOST=0.0.0.0

# Start with custom settings
LENS_PORT=8080 lens server
```

### **Advanced Configuration**
Create `lens.config.json`:
```json
{
  "server": {
    "port": 3000,
    "host": "0.0.0.0"
  },
  "performance": {
    "max_concurrent_queries": 100,
    "stage_timeouts": {
      "stage_a": 200,  // Lexical search timeout (ms)
      "stage_b": 300,  // Symbol search timeout (ms) 
      "stage_c": 300   // Semantic rerank timeout (ms)
    }
  },
  "features": {
    "fuzzy_search": true,
    "semantic_rerank": true,
    "learned_rerank": true
  }
}
```

### **Performance Tuning**
```bash
# For large codebases (>1M files)
export LENS_MEMORY_LIMIT_GB=16
export LENS_MAX_CANDIDATES=500

# For low-latency requirements  
export LENS_SEMANTIC_RERANK=false
export LENS_STAGE_A_ONLY=true
```

---

## 🔌 **API Reference**

### **Core Endpoints**

#### **`POST /search` - Main Search**
Search across your codebase with multiple modes:

```bash
curl -X POST http://localhost:3000/search \
  -H "Content-Type: application/json" \
  -d '{
    "repo_sha": "main",
    "q": "user authentication",  
    "mode": "hybrid",           // "lex", "struct", or "hybrid"
    "fuzzy": 1,                // Edit distance (0-2)
    "k": 10                    // Number of results (1-200)
  }'
```

#### **`POST /index` - Index Repository**  
Add a codebase to the search index:

```bash
curl -X POST http://localhost:3000/index \
  -H "Content-Type: application/json" \
  -d '{
    "repo_path": "/path/to/code",
    "repo_sha": "main"         // Unique identifier
  }'
```

#### **`GET /health` - System Health**
Check system status and performance:

```bash  
curl http://localhost:3000/health
# Returns: {"status": "ok", "shards_healthy": 5, "latency_p95": 18}
```

### **Search Modes Explained**

| Mode | Best For | Speed | Accuracy |
|------|----------|:-----:|:--------:|
| **`lex`** | Quick text search, exact matches | ⚡⚡⚡ | ⭐⭐ |
| **`struct`** | Code patterns, AST queries | ⚡⚡ | ⭐⭐⭐ |
| **`hybrid`** | Natural language, semantic similarity | ⚡ | ⭐⭐⭐⭐ |

### **Response Format**
```json
{
  "hits": [
    {
      "file": "src/auth/login.ts",
      "line": 42,
      "col": 8, 
      "snippet": "function authenticateUser(credentials) {",
      "score": 0.95,
      "why": ["exact", "symbol"],
      "symbol_kind": "function"
    }
  ],
  "total": 1,
  "latency_ms": {"stage_a": 4, "stage_b": 6, "stage_c": 8, "total": 18}
}
```

---

## 🚨 **Troubleshooting**

### **Common Issues**

**❌ "Repository not found in index"**
```bash
# Solution: Index the repository first
curl -X POST http://localhost:3000/index \
  -H "Content-Type: application/json" \
  -d '{"repo_path": ".", "repo_sha": "main"}'
```

**❌ "Search returns no results"**
```bash
# Check if repo_sha matches indexed value
curl http://localhost:3000/health

# Try broader search with fuzzy matching
curl -X POST http://localhost:3000/search \
  -d '{"repo_sha": "main", "q": "function", "mode": "lex", "fuzzy": 2}'
```

**❌ "Slow search performance"**
```bash
# Check latency breakdown
curl -X POST http://localhost:3000/search \
  -H "X-Trace-Id: debug-123" \
  -d '{"repo_sha": "main", "q": "test", "mode": "lex", "k": 5}'

# Disable semantic rerank for speed
curl -X POST http://localhost:3000/search \
  -d '{"repo_sha": "main", "q": "test", "mode": "struct", "k": 5}'
```

### **Performance Tips**

- **Use `mode: "lex"`** for fastest results (2-8ms)
- **Limit `k` parameter** to what you actually need (≤50 typically)
- **Use `fuzzy: 0`** when you need exact matches only
- **Monitor `/health`** endpoint for system bottlenecks

---

## 🏗️ **Architecture - The Technical Details**

> *For architects and advanced users who want to understand how Lens achieves sub-20ms performance*

### **Three-Layer Processing Pipeline**

```
Query: "user authentication logic"
    ↓
[Stage A: Lexical+Fuzzy Search]    // 2-8ms
    ├─ N-gram indexing
    ├─ Fuzzy matching (≤2 edits) 
    └─ Subtoken analysis
    ↓ (~100-200 candidates)
[Stage B: Symbol/AST Analysis]     // 3-10ms  
    ├─ Universal-ctags symbol resolution
    ├─ Tree-sitter AST parsing
    └─ Structural pattern matching
    ↓ (~50-100 candidates)
[Stage C: Semantic Reranking]      // 5-15ms (optional)
    ├─ ColBERT-v2 vector similarity
    ├─ Context-aware scoring
    └─ Final ranking
    ↓ (Top 10-50 results)
Final Results: 18ms total
```

### **Technology Stack**
- **Core Engine**: TypeScript + Fastify
- **Indexing**: Memory-mapped segments, NATS/JetStream messaging
- **AST Parsing**: Tree-sitter, universal-ctags  
- **Vector Search**: ColBERT-v2, HNSW indexing
- **Observability**: OpenTelemetry tracing & metrics
- **Storage**: Append-only segments with periodic compaction

---

## 📈 **Production & Monitoring**

### **Built-in Observability**
- **Real-time metrics** at `/metrics` endpoint
- **Distributed tracing** with OpenTelemetry
- **Performance breakdown** by search stage
- **Health monitoring** with automatic alerts

### **Benchmarking & Validation**
```bash
# Run performance benchmarks
npm run benchmark:smoke      # Quick 5-minute benchmark
npm run benchmark:full      # Comprehensive evaluation  

# Validate system health
npm run gates:validate      # Check all quality gates
```

### **Production Deployment**
```yaml
# docker-compose.yml
version: '3.8'
services:
  lens:
    image: lens:1.0.0
    ports: ["3000:3000"]
    volumes: ["./code:/code:ro"]
    environment:
      LENS_MEMORY_LIMIT_GB: 8
      LENS_MAX_CONCURRENT_QUERIES: 200
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
```

---

## 🤝 **Contributing & Community**

### **Get Involved**
- 📖 **Documentation**: [Full docs](./docs/)
- 🐛 **Issues**: [Report bugs](https://github.com/lens-search/lens/issues)  
- 💬 **Discussions**: [Join the community](https://github.com/lens-search/lens/discussions)
- 🛠️ **Development**: See [CONTRIBUTING.md](./CONTRIBUTING.md)

### **Roadmap**
- ✅ **v1.0**: Core search with 3-stage pipeline
- 🚧 **v1.1**: IDE extensions (VS Code, IntelliJ)
- 📋 **v1.2**: GraphQL/REST API integrations  
- 🔮 **v2.0**: Multi-language neural reranking

---

## 📄 **License & Credits**

**MIT License** - See [LICENSE](./LICENSE) for details.

**Built with love for developers** who deserve better code search. 

**Special thanks to the open source community** and the researchers behind ColBERT, Tree-sitter, and universal-ctags.

---

<div align="center">

### **Ready to search your code like never before?**

```bash
npm install -g lens@1.0.0
lens server
```

**🔍 Happy searching!**

</div>

