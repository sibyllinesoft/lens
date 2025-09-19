# Lens Documentation

Welcome to the comprehensive documentation for Lens, the high-performance code search engine.

## 📚 Documentation Structure

- **[Getting Started](getting-started.md)** - Quick setup and first search
- **[Architecture](architecture.md)** - System design and components
- **[API Reference](api-reference.md)** - HTTP API endpoints and parameters
- **[CLI Reference](cli-reference.md)** - Command-line interface usage
- **[Configuration](configuration.md)** - Configuration options and settings
- **[Monitoring](monitoring.md)** - Prometheus + host metrics powered by sysinfo
- **[Development](development.md)** - Contributing and development setup
- **[Deployment](deployment.md)** - Production deployment guide
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

## 🚀 Quick Links

- [Installation Guide](getting-started.md#installation)
- [API Examples](api-reference.md#examples)
- [Performance Benchmarks](performance.md)
- [Monitoring & Metrics](monitoring.md#host-health)
- [Security Guide](security.md)

## 🏗️ Project Structure

```
lens/
├── apps/
│   └── lens-core/           # Main Lens application
├── packages/
│   ├── search-engine/       # Core search functionality  
│   └── lsp-server/         # Language Server Protocol integration
├── tests/
│   ├── e2e/                # End-to-end tests
│   └── integration/        # Integration tests
├── docs/                   # Documentation (you are here)
├── scripts/               # Build and utility scripts
└── monitoring/            # Observability configuration
```

## 🔗 External Resources

- [GitHub Repository](https://github.com/sibyllinesoft/lens)
- [Issue Tracker](https://github.com/sibyllinesoft/lens/issues)
- [Changelog](../CHANGELOG.md)
- [License](../LICENSE)

---

**Need help?** Check the [troubleshooting guide](troubleshooting.md) or [open an issue](https://github.com/sibyllinesoft/lens/issues).
