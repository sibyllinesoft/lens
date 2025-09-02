# 🚀 Lens Quickstart Guide

**From zero to searching your codebase in under 5 minutes**

> This guide gets you up and running with Lens quickly. For detailed explanations, see the [main README](../README.md).

## ⚡ **1. Quick Installation**

Choose your preferred method:

### **Option A: NPM (Easiest)**
```bash
npm install -g lens@1.0.0
# ✅ Ready to go!
```

### **Option B: Docker**
```bash
docker run -p 3000:3000 -v $(pwd):/code lens:1.0.0
# ✅ Server running on port 3000
```

### **Option C: From Source** 
```bash
git clone https://github.com/lens-search/lens.git
cd lens && npm install && npm run build
# ✅ Built from source
```

## 🎯 **2. Start Searching**

### **Start the Server**
```bash
lens server
# 🎉 Server running on http://localhost:3000
```

### **Index Your Code** 
```bash
# Index current directory
curl -X POST http://localhost:3000/index \
  -H "Content-Type: application/json" \
  -d '{"repo_path": ".", "repo_sha": "main"}'

# ✅ Response: {"indexed_files": 1247, "time_seconds": 3.2}
```

### **Search Away!**
```bash
# Find functions with fuzzy matching
curl -X POST http://localhost:3000/search \
  -d '{"repo_sha": "main", "q": "calcTotal", "mode": "hybrid", "k": 10}'

# Find classes and interfaces  
curl -X POST http://localhost:3000/search \
  -d '{"repo_sha": "main", "q": "class User", "mode": "struct", "k": 5}'

# Natural language queries
curl -X POST http://localhost:3000/search \
  -d '{"repo_sha": "main", "q": "authentication logic", "mode": "hybrid", "k": 10}'
```

### **Understand the Results**
```json
{
  "hits": [
    {
      "file": "src/auth/User.ts",
      "line": 15,
      "snippet": "class User implements UserInterface {",
      "score": 0.95,
      "why": ["exact", "symbol"]      // How it was found
    }
  ],
  "total": 1,
  "latency_ms": {
    "stage_a": 4,    // Text search
    "stage_b": 6,    // Code analysis
    "stage_c": 8,    // Semantic ranking
    "total": 18      // Total time < 20ms!
  }
}
```

🎉 **That's it!** You're now searching code at lightning speed.

---

## 🎨 **3. Choose Your Search Style**

| Mode | When to Use | Speed | Accuracy |
|------|-------------|:-----:|:--------:|
| **`lex`** | Fast text search, exact matches | ⚡⚡⚡ | ⭐⭐ |
| **`struct`** | Code patterns, function/class names | ⚡⚡ | ⭐⭐⭐ |
| **`hybrid`** | Natural language, "find similar code" | ⚡ | ⭐⭐⭐⭐ |

### **Quick Examples**
```bash
# Fast text search (2-8ms)
curl -X POST localhost:3000/search -d '{"repo_sha": "main", "q": "calculateTax", "mode": "lex"}'

# Code structure search (3-10ms)  
curl -X POST localhost:3000/search -d '{"repo_sha": "main", "q": "async function", "mode": "struct"}'

# Smart semantic search (5-15ms)
curl -X POST localhost:3000/search -d '{"repo_sha": "main", "q": "user auth logic", "mode": "hybrid"}'
```

---

## 🛠️ **4. Pro Tips & CLI Commands**

### **System Health & Status**
```bash
# Check if everything is running smoothly
lens health
curl http://localhost:3000/health

# Get detailed system info
curl http://localhost:3000/manifest
```

### **Performance Tuning**
```bash
# For large codebases (>1M files)
LENS_MEMORY_LIMIT_GB=16 lens server

# For speed-critical applications  
LENS_SEMANTIC_RERANK=false lens server    # Skip semantic stage

# Custom port
LENS_PORT=8080 lens server
```

---

## 🔧 **5. Common Issues & Solutions**

### **❌ "Repository not found in index"**
```bash
# Make sure you indexed the repository first
curl -X POST http://localhost:3000/index \
  -H "Content-Type: application/json" \
  -d '{"repo_path": ".", "repo_sha": "main"}'
```

### **❌ "Search returns no results"**  
```bash
# Try fuzzy matching for typos
curl -X POST localhost:3000/search \
  -d '{"repo_sha": "main", "q": "function", "mode": "lex", "fuzzy": 2}'

# Or use broader semantic search
curl -X POST localhost:3000/search \
  -d '{"repo_sha": "main", "q": "function definition", "mode": "hybrid"}'
```

### **❌ "Slow performance"**
```bash  
# Check the latency breakdown
curl -X POST localhost:3000/search \
  -H "X-Trace-Id: debug-123" \
  -d '{"repo_sha": "main", "q": "test", "mode": "lex", "k": 5}'

# Use faster mode for speed
curl -X POST localhost:3000/search \
  -d '{"repo_sha": "main", "q": "test", "mode": "lex"}'  # Fastest
```

---

## 🐳 **Docker Quick Setup**

```bash
# One command to rule them all
docker run -p 3000:3000 -v $(pwd):/code lens:1.0.0
```

Or with `docker-compose.yml`:
```yaml
services:
  lens:
    image: lens:1.0.0
    ports: ["3000:3000"]
    volumes: ["./:/code:ro"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
```

---

## 🚀 **What's Next?**

### **🧠 Learn More**
- 📖 [Complete Documentation](../README.md) - Full feature overview
- 🏗️ [Architecture Guide](./ARCHITECTURE.md) - How Lens works under the hood
- 🤖 [AI Integration](./AGENT_INTEGRATION.md) - Connect Lens to your AI tools

### **⚡ Performance Tips**
- **Use `mode: "lex"`** for fastest searches (2-8ms)
- **Use `mode: "hybrid"`** for best accuracy
- **Limit `k` parameter** to what you need (≤50 is typical)
- **Monitor `/health`** endpoint for system status

### **🛠️ Advanced Usage**
```bash
# Benchmark your installation
npm run benchmark:smoke

# Run comprehensive tests  
npm run test:coverage

# Check system configuration
npm run validate:config
```

---

## 🆘 **Need Help?**

- 📖 **Full Documentation**: [../README.md](../README.md)
- 🐛 **Found a Bug?**: [Report it on GitHub](https://github.com/lens-search/lens/issues)
- 💬 **Questions?**: [Join our Discussions](https://github.com/lens-search/lens/discussions)
- 🛠️ **Contributing**: See [CONTRIBUTING.md](../CONTRIBUTING.md)

---

<div align="center">

**🎉 You're all set!** 

**Happy code searching with Lens!** 🔍

*From zero to searching in under 5 minutes - just like we promised.*

</div>