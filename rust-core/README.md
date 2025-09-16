# Lens Code Search - Rust Core

A production-grade, fraud-resistant code search engine built in Rust with comprehensive security features and high-performance search capabilities.

## 🔍 Overview

The Lens Rust Core is the high-performance search engine component of the Lens code search system. It provides lightning-fast code search with built-in fraud resistance, cryptographic attestation, and enterprise-grade security features.

### Key Features

- **🚀 Sub-millisecond Search**: Optimized Rust implementation for maximum performance
- **🛡️ Fraud-Resistant**: Built-in tripwire mechanisms and cryptographic attestation
- **🔒 Production-Only**: Enforced "real" mode - refuses to run in development/mock modes
- **🌐 Multi-Language Support**: Python, Rust, TypeScript, JavaScript, Go, Java, C/C++
- **📊 AST-Aware**: Tree-sitter integration for syntax-aware search
- **🔗 MCP Integration**: Model Context Protocol for AI assistant access
- **📈 Observability**: Built-in metrics, tracing, and health monitoring

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│           Lens Rust Core                │
├─────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐      │
│  │   Server    │  │ Attestation │      │
│  │  (Axum)     │  │  Service    │      │
│  └─────────────┘  └─────────────┘      │
│  ┌─────────────┐  ┌─────────────┐      │
│  │   Search    │  │   Indexer   │      │
│  │   Engine    │  │             │      │
│  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────┘
│
├── HTTP API Endpoints
├── Fraud-Resistant Attestation  
├── Cryptographic Verification
└── Performance Monitoring
```

## 🚀 Quick Start

### Prerequisites

- Rust 1.70+ 
- Git (for build-time attestation)
- Linux/macOS/Windows

### Building

```bash
# Clone the repository
cd rust-core

# Build for development
cargo build

# Build for production (recommended)
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench
```

### Running

```bash
# Start the server (production mode only)
cargo run --release -- --mode real --addr 0.0.0.0:8080

# Or run the built binary
./target/release/lens-core --mode real --addr 0.0.0.0:8080
```

## 🔧 Configuration

### Command Line Options

| Option | Description | Default | Required |
|--------|-------------|---------|----------|
| `--mode` | Service mode (must be "real") | "real" | Yes |
| `--addr` | Server bind address | "0.0.0.0:8080" | No |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RUST_LOG` | Log level (error, warn, info, debug, trace) | "info" |
| `LENS_BUILD_MODE` | Build validation mode | Auto-detected |

## 📡 API Endpoints

The Rust service exposes the following HTTP endpoints:

### Core Endpoints

#### `GET /manifest`
Returns service metadata and build information.

```json
{
  "service": "lens-rpc",
  "version": "1.0.0",
  "mode": "real",
  "git_sha": "abc123def456",
  "build_target": "x86_64-unknown-linux-gnu",
  "attestation_enabled": true
}
```

#### `GET /health`
Health check endpoint with mode validation.

```json
{
  "status": "healthy",
  "mode": "real",
  "git_sha": "abc123def456",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### `POST /handshake`
Fraud-resistant handshake for client verification.

**Request:**
```json
{
  "nonce": "client-generated-nonce-123"
}
```

**Response:**
```json
{
  "success": true,
  "attestation": {
    "timestamp": 1705312200,
    "mode": "real",
    "git_sha": "abc123def456",
    "environment": {
      "hostname": "prod-server-01",
      "cpu_model": "AMD Ryzen 7 5800X",
      "memory_gb": 16,
      "kernel_version": "6.14.0-29-generic",
      "rust_version": "rustc 1.70.0"
    },
    "handshake_nonce": "client-generated-nonce-123",
    "handshake_response": "sha256-hash-response",
    "build_info": {
      "target_triple": "x86_64-unknown-linux-gnu",
      "profile": "release",
      "features": ["simd"],
      "dependencies": []
    }
  }
}
```

#### `POST /search`
Core search functionality with multi-language support.

**Request:**
```json
{
  "query": "function authentication middleware",
  "limit": 20
}
```

**Response:**
```json
{
  "results": [
    {
      "file": "src/auth.rs",
      "line": 42,
      "col": 5,
      "lang": "rust",
      "snippet": "pub fn authenticate_user(token: &str) -> Result<User> {",
      "score": 0.95,
      "why": ["exact"],
      "ast_path": "function_item",
      "symbol_kind": "function",
      "byte_offset": 1234,
      "span_len": 45
    }
  ],
  "total": 15,
  "query_time_ms": 2,
  "attestation_hash": "sha256-query-attestation-hash"
}
```

## 🔒 Security Features

### Fraud-Resistant Design

The service implements multiple tripwire mechanisms to prevent tampering:

1. **Mode Enforcement**: Only runs in "real" mode - exits with error for dev/mock modes
2. **Build-time Attestation**: Git SHA and build environment tracking  
3. **Cryptographic Handshakes**: Challenge-response verification for clients
4. **Health Check Validation**: Continuous mode verification

### Attestation Service

```rust
// Example attestation verification
let service = AttestationService::new("real")?;
let record = service.create_handshake("client-nonce-123")?;
let is_valid = service.validate_handshake(&nonce, &response)?;
```

### Security Best Practices

- All inputs are validated and sanitized
- No hardcoded secrets or credentials
- Memory-safe Rust prevents buffer overflows
- Comprehensive error handling without information leakage
- Production builds use LTO and symbol stripping

## 🏃 Performance

### Benchmarks

```bash
# Run performance benchmarks
cargo bench

# Search benchmarks
cargo bench --bench search_benchmarks

# Indexing benchmarks  
cargo bench --bench indexing_benchmarks
```

### Optimization Features

- **Release Build Optimizations**: LTO, single codegen unit, symbol stripping
- **SIMD Support**: Optional SIMD optimizations for text processing
- **Memory Efficiency**: Zero-copy operations where possible
- **Async I/O**: Tokio-based async runtime for concurrent request handling

### Performance Targets

| Metric | Target | Typical |
|--------|--------|---------|
| Search Latency | < 10ms p95 | 1-5ms |
| Throughput | > 1000 req/s | 2000+ req/s |
| Memory Usage | < 100MB | 50-80MB |
| Startup Time | < 1s | 200-500ms |

## 🧪 Testing

### Running Tests

```bash
# All tests
cargo test

# Unit tests only
cargo test --lib

# Integration tests
cargo test --test '*'

# With coverage
cargo tarpaulin --all-features --workspace --ignore-tests --timeout 900 --out Html
```

### Test Categories

- **Unit Tests**: Individual module functionality
- **Integration Tests**: HTTP endpoint testing with axum-test
- **Security Tests**: Tripwire and attestation validation
- **Performance Tests**: Benchmark regression testing

## 📊 Monitoring & Observability

### Metrics

The service exposes Prometheus-compatible metrics:

```bash
# Query metrics endpoint (if enabled)
curl http://localhost:8080/metrics
```

Available metrics:
- Request duration histograms
- Request count by endpoint
- Error rates and types
- Search performance metrics
- Memory usage statistics

### Logging

Structured logging with configurable levels:

```bash
# Set log level
RUST_LOG=debug cargo run --release

# Specific module logging
RUST_LOG=lens_core::search=trace,lens_core::server=debug cargo run
```

### Tracing

Distributed tracing support with OpenTelemetry integration:

```rust
use tracing::{info, instrument};

#[instrument]
async fn search_documents(query: &str) -> Result<Vec<SearchResult>> {
    info!("Starting search for query: {}", query);
    // Search implementation
}
```

## 🔗 Integration

### MCP Integration

The Rust service integrates with the MCP (Model Context Protocol) layer:

```typescript
// TypeScript MCP integration
import { LensSearchEngine } from '../api/search-engine.js';

const engine = new LensSearchEngine();
const results = await engine.search({
  repo_sha: "abc123",
  query: "authentication middleware",
  mode: "hybrid"
});
```

### API Client Example

```bash
# Health check
curl -X GET http://localhost:8080/health

# Perform search
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"query": "async function", "limit": 10}'

# Handshake verification
curl -X POST http://localhost:8080/handshake \
  -H "Content-Type: application/json" \
  -d '{"nonce": "random-client-nonce-456"}'
```

## 🛠️ Development

### Project Structure

```
rust-core/
├── src/
│   ├── main.rs              # Entry point with security enforcement
│   ├── lib.rs              # Module exports and build info
│   ├── server.rs           # HTTP server and endpoint handlers
│   ├── search.rs           # Core search engine implementation
│   ├── attestation.rs      # Fraud-resistant attestation service
│   └── indexer.rs          # Document indexing and processing
├── lens-rpc/               # RPC service workspace member
├── lens-indexer/           # Indexer CLI workspace member
├── benches/                # Performance benchmarks
│   ├── search_benchmarks.rs
│   └── indexing_benchmarks.rs
├── build.rs               # Build script for git info
├── Cargo.toml             # Dependencies and configuration
└── README.md              # This file
```

### Adding New Features

1. **Search Algorithms**: Extend `search.rs` with new search modes
2. **Language Support**: Add tree-sitter parsers in `Cargo.toml`
3. **API Endpoints**: Add handlers in `server.rs`
4. **Security Features**: Extend `attestation.rs` validation
5. **Performance**: Add benchmarks for new features

### Code Quality

```bash
# Format code
cargo fmt

# Lint code
cargo clippy

# Check for security vulnerabilities
cargo audit

# Generate documentation
cargo doc --open
```

## 📦 Deployment

### Docker

```dockerfile
# Production Docker build
FROM rust:1.70 AS builder
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
COPY --from=builder /target/release/lens-core /usr/local/bin/
EXPOSE 8080
CMD ["lens-core", "--mode", "real", "--addr", "0.0.0.0:8080"]
```

### Systemd Service

```ini
# /etc/systemd/system/lens-core.service
[Unit]
Description=Lens Code Search Core Service
After=network.target

[Service]
Type=simple
User=lens
ExecStart=/usr/local/bin/lens-core --mode real --addr 0.0.0.0:8080
Restart=always
RestartSec=5
Environment=RUST_LOG=info

[Install]
WantedBy=multi-user.target
```

### Production Checklist

- [ ] Build with `--release` flag
- [ ] Verify mode enforcement is active
- [ ] Configure proper logging levels
- [ ] Set up health check monitoring
- [ ] Enable metrics collection
- [ ] Configure reverse proxy (nginx/traefik)
- [ ] Set up SSL/TLS termination
- [ ] Implement log rotation
- [ ] Configure firewall rules
- [ ] Set up backup procedures

## 🐛 Troubleshooting

### Common Issues

**Service won't start:**
```
TRIPWIRE VIOLATION: Service mode must be 'real', got: 'dev'
```
Solution: Always use `--mode real` in production.

**Search returns no results:**
- Verify documents are indexed
- Check query format and language detection
- Ensure file extensions are supported

**High memory usage:**
- Review indexing batch sizes
- Consider adjusting search result limits
- Monitor for memory leaks with profiling

**Performance issues:**
- Enable release optimizations
- Check network latency between components
- Profile with `cargo bench` and `perf`

### Debug Mode

```bash
# Enable debug logging
RUST_LOG=debug cargo run --release

# Profile with perf (Linux)
perf record --call-graph=dwarf cargo bench
perf report

# Memory profiling with valgrind
valgrind --tool=massif cargo test
```

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make changes with comprehensive tests
4. Ensure all security checks pass
5. Submit a pull request

### Code Standards

- Follow Rust idioms and best practices
- Maintain 90%+ test coverage
- Add comprehensive documentation
- Ensure security compliance
- Include performance benchmarks for new features

### Security Review

All changes undergo security review:
- No new attack vectors introduced
- Attestation mechanisms preserved
- Input validation maintained
- Error handling doesn't leak information

## 📄 License

MIT License - see [LICENSE](../LICENSE) file for details.

## 🆘 Support

For issues and questions:

- **Bug Reports**: Create an issue with reproduction steps
- **Feature Requests**: Describe the use case and expected behavior
- **Security Issues**: Report privately to maintain responsible disclosure
- **Performance Issues**: Include benchmark data and profiling results

---

**Production Ready** ✅ | **Security Hardened** 🔒 | **High Performance** ⚡