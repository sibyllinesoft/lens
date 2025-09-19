# Getting Started with Lens

This guide will help you get up and running with Lens, the high-performance code search engine, in minutes.

## Installation

### Option 1: Using Cargo (Recommended)

```bash
# Install from source
git clone https://github.com/sibyllinesoft/lens.git
cd lens
cargo install --path apps/lens-core

# Verify installation
lens --version
```

### Option 2: Using Docker

```bash
# Run with Docker
docker run -p 3000:3000 -v $(pwd):/workspace lens:latest

# Or use docker-compose for development
docker-compose up lens-dev
```

### Option 3: Build from Source

```bash
# Clone the repository
git clone https://github.com/sibyllinesoft/lens.git
cd lens

# Build the project
make build

# The binary will be at ./target/release/lens
./target/release/lens --help
```

## Quick Start

### 1. Index Your Code

```bash
# Index current directory
lens index .

# Index specific directory with progress
lens index ./my-project --progress

# Rebuild an existing index from scratch
lens index ./my-project --force
```

### 2. Search Your Code

```bash
# Basic search
lens search "function main"

# Limit number of results
lens search "SearchEngine" --limit 20

# Symbol-only search
lens search "struct QueryBuilder" --symbols

# Language-specific search
lens search "async fn" --language rust

# Filter by file path
lens search "builder" --file-pattern "packages/search-engine"

# Fuzzy match for typos
lens search "intiailze" --fuzzy

# Inline filters (lang/path prefixes)
lens search "lang:rust path:packages/search-engine builder"

# Paginate results
lens search "builder" --limit 10 --offset 10
```

### 3. Start the HTTP Server

```bash
# Start server on default port (3000)
lens serve

# Start with custom port and CORS enabled
lens serve --port 8080 --cors

# Server endpoints:
# - http://localhost:3000/health
# - http://localhost:3000/stats  
# - http://localhost:3000/search?q=your-query
```

## HTTP API Examples

### Health Check

```bash
curl http://localhost:3000/health
```

```json
{
  "status": "healthy",
  "version": "1.1.0",
  "uptime_seconds": 42
}
```

### Search Query

```bash
curl "http://localhost:3000/search?q=function&limit=5"
```

```json
{
  "query": "function",
  "query_type": "text",
  "total": 12,
  "duration_ms": 14,
  "results": [
    {
      "file_path": "src/main.rs",
      "line_number": 15,
      "content": "fn main() {",
      "score": 9.87,
      "language": "rust",
      "matched_terms": ["function"],
      "context_lines": [
        "use tracing::info;",
        "",
        "fn main() {",
        "    info!(\\"Starting Lens\\");"
      ]
    }
  ],
  "index_stats": {
    "total_documents": 1250,
    "index_size_bytes": 1843200,
    "index_size_human": "1.8 MiB",
    "supported_languages": 12,
    "average_document_size": 98.2,
    "last_updated": "2025-09-18T20:30:12.123Z"
  }
}
```

### Index Statistics

```bash
curl http://localhost:3000/stats
```

```json
{
  "total_documents": 1247,
  "total_size_bytes": 2457600,
  "index_size_bytes": 891234,
  "last_updated": "2024-03-15T10:30:00Z"
}
```

## Configuration

### Environment Variables

```bash
# Set index path
export LENS_INDEX_PATH="./my-index"

# Enable debug logging
export RUST_LOG=debug

# Set server bind address
export LENS_BIND_ADDR="0.0.0.0:3000"
```

### Configuration File

Create `lens.toml`:

```toml
[index]
path = "./index"
batch_size = 1000

[server]
host = "0.0.0.0"
port = 3000
cors_enabled = true

[search]
default_limit = 50
max_limit = 1000

[logging]
level = "info"
format = "json"
```

Then use it:

```bash
lens --config lens.toml serve
```

## Language Support

Lens supports indexing and searching in:

- **Rust** (`.rs`) - Full AST parsing
- **TypeScript/JavaScript** (`.ts`, `.js`, `.tsx`, `.jsx`) - Symbol extraction
- **Python** (`.py`) - Function and class detection
- **Go** (`.go`) - Package and function indexing
- **Java** (`.java`) - Class and method indexing
- **C/C++** (`.c`, `.cpp`, `.h`, `.hpp`) - Basic symbol support

## Performance Tips

1. **Index Incrementally**: Use `lens index --incremental` for faster updates
2. **Exclude Unnecessary Files**: Use `--exclude` patterns for build artifacts
3. **Tune Batch Size**: Larger batch sizes can improve indexing speed
4. **Use SSD Storage**: Index performance benefits significantly from fast storage

## Development Workflow

```bash
# Development build and run
make dev

# Run tests
make test

# Check code quality
make lint
make format-check

# Full CI pipeline
make ci
```

## Docker Development

```bash
# Start development environment
docker-compose up lens-dev

# View logs
docker-compose logs -f lens-dev

# Execute commands in container
docker-compose exec lens-dev lens index /workspace
```

## Troubleshooting

### Common Issues

**Issue**: `lens: command not found`
```bash
# Make sure Cargo bin directory is in PATH
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**Issue**: Permission denied during indexing
```bash
# Check file permissions and ownership
ls -la /path/to/index/directory
sudo chown -R $USER:$USER /path/to/index/directory
```

**Issue**: Server fails to bind to port
```bash
# Check if port is in use
sudo netstat -tulpn | grep :3000

# Use different port
lens serve --port 8080
```

**Issue**: Index is empty after indexing
```bash
# Check indexing logs
RUST_LOG=debug lens index . --progress

# Verify file patterns
lens index . --include="*.rs" --progress
```

## Next Steps

- [Learn about the architecture](architecture.md)
- [Explore the full API](api-reference.md)
- [Configure for production](deployment.md)
- [Contribute to the project](development.md)

---

**Got stuck?** Check our [troubleshooting guide](troubleshooting.md) or [open an issue](https://github.com/sibyllinesoft/lens/issues).
