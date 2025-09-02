# Lens Quickstart Guide

Get started with Lens code search in under 5 minutes.

## Installation

### Using npm (Recommended)
```bash
npm install -g lens@1.0.0
```

### Using Docker
```bash
docker pull lens:1.0.0
```

### From Source
```bash
git clone https://github.com/lens/lens.git
cd lens
npm install
npm run build
```

## Quick Start

### 1. Start the Lens Server
```bash
# Start the server (port 3000 by default)
lens server

# Or with custom port
LENS_PORT=8080 lens server
```

### 2. Index Your Codebase
```bash
# Index current directory
curl -X POST http://localhost:3000/index \
  -H "Content-Type: application/json" \
  -d '{"repo_path": ".", "repo_sha": "main"}'

# Index specific directory
curl -X POST http://localhost:3000/index \
  -H "Content-Type: application/json" \
  -d '{"repo_path": "/path/to/your/code", "repo_sha": "abc123"}'
```

### 3. Search Your Code
```bash
# Basic search
curl -X POST http://localhost:3000/search \
  -H "Content-Type: application/json" \
  -d '{
    "repo_sha": "main",
    "q": "function calculateTotal",
    "mode": "hybrid",
    "k": 10,
    "fuzzy": 1
  }'

# Search with TypeScript/JavaScript focus
curl -X POST http://localhost:3000/search \
  -H "Content-Type: application/json" \
  -d '{
    "repo_sha": "main", 
    "q": "class User",
    "mode": "struct",
    "k": 20
  }'
```

### 4. Explore Results
The response includes:
- **hits**: Matching code locations with context
- **total**: Number of results found
- **latency_ms**: Performance breakdown by stage
- **api_version**: API compatibility version

```json
{
  "hits": [
    {
      "file": "src/models/User.ts",
      "line": 15,
      "col": 8,
      "snippet": "class User implements UserInterface {",
      "score": 0.95,
      "why": ["exact", "symbol"],
      "symbol_kind": "class"
    }
  ],
  "total": 1,
  "latency_ms": {
    "stage_a": 12,
    "stage_b": 8,
    "stage_c": 15,
    "total": 35
  },
  "api_version": "v1",
  "index_version": "v1",
  "policy_version": "v1"
}
```

## Search Modes

### Lexical Mode (`lex`)
Fast text-based search with fuzzy matching:
```bash
curl -X POST http://localhost:3000/search \
  -H "Content-Type: application/json" \
  -d '{
    "repo_sha": "main",
    "q": "calculateTax",
    "mode": "lex",
    "fuzzy": 2,
    "k": 5
  }'
```

### Structural Mode (`struct`) 
Language-aware AST pattern matching:
```bash
curl -X POST http://localhost:3000/search \
  -H "Content-Type: application/json" \
  -d '{
    "repo_sha": "main",
    "q": "async function.*await",
    "mode": "struct",
    "k": 10
  }'
```

### Hybrid Mode (`hybrid`)
Combines lexical, structural, and semantic ranking:
```bash
curl -X POST http://localhost:3000/search \
  -H "Content-Type: application/json" \
  -d '{
    "repo_sha": "main",
    "q": "user authentication logic",
    "mode": "hybrid",
    "k": 15
  }'
```

## Using the CLI

### Check System Health
```bash
lens health
```

### Migration & Compatibility
```bash
# Check version compatibility
lens compat-check --api-version v1 --index-version v1 --policy-version v1

# Check compatibility against nightly bundles
curl "http://localhost:3000/compat/bundles"

# List available migrations
lens list-migrations

# Migrate index (if needed)
lens migrate-index --from v0 --to v1
```

### Build & Deploy
```bash
# Secure build with SBOM
lens build --sbom --sast --lock

# Build container
lens build --container --sbom
```

## Configuration

### Environment Variables
```bash
# Server configuration
export LENS_PORT=3000
export LENS_HOST=0.0.0.0

# Performance tuning
export LENS_STAGE_A_TIMEOUT_MS=200
export LENS_STAGE_B_TIMEOUT_MS=300
export LENS_STAGE_C_TIMEOUT_MS=300

# Security
export LENS_API_RATE_LIMIT=1000
```

### Configuration File
Create `lens.config.json`:
```json
{
  "server": {
    "port": 3000,
    "host": "0.0.0.0"
  },
  "performance": {
    "stage_timeouts": {
      "stage_a": 200,
      "stage_b": 300,
      "stage_c": 300
    }
  },
  "features": {
    "fuzzy_search": true,
    "semantic_rerank": true,
    "learned_rerank": true
  }
}
```

## Docker Usage

### Run with Docker
```bash
# Start server
docker run -p 3000:3000 -v /path/to/code:/code lens:1.0.0

# Index mounted code
docker exec -it lens-container curl -X POST http://localhost:3000/index \
  -H "Content-Type: application/json" \
  -d '{"repo_path": "/code", "repo_sha": "main"}'
```

### Docker Compose
```yaml
version: '3.8'
services:
  lens:
    image: lens:1.0.0
    ports:
      - "3000:3000"
    volumes:
      - ./your-code:/code:ro
    environment:
      - LENS_PORT=3000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 10s
      timeout: 5s
      retries: 3
```

## Language Support

Lens supports intelligent search across multiple languages:

- **TypeScript/JavaScript**: Full AST parsing, symbol resolution
- **Python**: Function/class detection, import analysis
- **Rust**: Struct/trait/impl block recognition
- **Go**: Package/function/struct parsing
- **Java**: Class/method/interface detection
- **Bash**: Function and variable recognition

## Performance Tips

### Optimal Indexing
1. Index frequently searched repositories first
2. Use meaningful `repo_sha` identifiers for caching
3. Re-index when significant code changes occur

### Search Optimization
1. Use `mode: "lex"` for fastest results
2. Use `mode: "hybrid"` for best relevance
3. Limit `k` to what you actually need (≤50 typical)
4. Use `fuzzy: 0` when you need exact matches

### Resource Management
1. Monitor `/health` endpoint for system status
2. Check `latency_ms` breakdown to identify bottlenecks
3. Use appropriate timeouts for your use case

## Troubleshooting

### Common Issues

**Server won't start:**
```bash
# Check port availability
lsof -i :3000

# Check logs
lens server --verbose
```

**Search returns no results:**
```bash
# Verify index exists
curl http://localhost:3000/health

# Check repo_sha matches indexed value
curl http://localhost:3000/manifest
```

**Slow search performance:**
```bash
# Monitor latency breakdown
curl -X POST http://localhost:3000/search \
  -H "X-Trace-Id: debug-123" \
  -H "Content-Type: application/json" \
  -d '{"repo_sha": "main", "q": "test", "mode": "lex", "k": 1}'

# Check system resources
lens health
```

**Version compatibility errors:**
```bash
# Check compatibility
lens compat-check --api-version v1 --index-version v1 --policy-version v1

# Update to latest version
npm update -g lens
```

## Next Steps

- 📚 Read the [Agent Integration Guide](AGENT_INTEGRATION.md) for AI/editor integration
- ⚙️ See the [Configuration Reference](CONFIG_REFERENCE.md) for advanced settings
- 🚨 Check the [Operations Runbook](OPERATIONS.md) for production deployment
- 🔧 Explore the [API Documentation](API.md) for complete endpoint reference

## Getting Help

- **Documentation**: Full docs at [docs/](./index.md)
- **Issues**: Report bugs at [GitHub Issues](https://github.com/lens/lens/issues)
- **Community**: Join discussions on [GitHub Discussions](https://github.com/lens/lens/discussions)

---

**Happy searching!** 🔍