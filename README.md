# Lens 🔍
## **Production-Ready Code Search with Advanced Semantic Understanding**

[![npm version](https://img.shields.io/npm/v/@sibyllinesoft/lens.svg)](https://www.npmjs.com/package/@sibyllinesoft/lens)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Node.js Version](https://img.shields.io/badge/node->=18.0.0-brightgreen.svg)](https://nodejs.org/)

> **Production-ready code search that actually understands your code.** Lens combines lightning-fast text search with intelligent code analysis, delivering high-quality search results with sub-millisecond response times.

**🎯 Production Status:** `@sibyllinesoft/lens@1.0.0-rc.2` - Enterprise-ready with comprehensive monitoring and proven performance improvements.

## 🚀 Quick Start

```bash
# Install
npm install -g @sibyllinesoft/lens

# Start the search engine
lens start

# Search your codebase
lens search "authentication logic"
```

## 📁 Repository Structure

This repository is organized for maintainability and clarity:

### 📂 **Core Directories**
- **[`/src`](./src/)** - Main application source code (TypeScript/Rust)
- **[`/docs`](./docs/)** - Complete technical documentation and guides
- **[`/benchmarks`](./benchmarks/)** - Performance benchmarking suites and analysis
- **[`/scripts`](./scripts/)** - Utility scripts for development and operations
- **[`/configs`](./configs/)** - Configuration files and settings
- **[`/infra`](./infra/)** - Infrastructure as code (Docker, CI/CD, deployment)
- **[`/reports`](./reports/)** - Generated reports and analysis outputs (git-ignored)

### 📝 **Key Files**
- **[`README.md`](./README.md)** - This overview and getting started guide
- **[`CLAUDE.md`](./CLAUDE.md)** - Project development notes and context
- **[`TODO.md`](./TODO.md)** - Current development tasks and planning
- **[`package.json`](./package.json)** - Node.js dependencies and scripts
- **[`Cargo.toml`](./Cargo.toml)** - Rust dependencies and configuration

## 🏆 **Performance & Features**

### **Proven Search Quality**
- **High Relevance**: Advanced semantic understanding for code search
- **Fast Response**: Sub-millisecond query processing  
- **Comprehensive Coverage**: Multi-language support (TypeScript, Rust, Python, etc.)
- **Intelligent Matching**: Fuzzy search with typo tolerance

### **Production-Ready Architecture**
- **Multi-Stage Pipeline**: Lexical + Symbol + Semantic search layers
- **Scalable Infrastructure**: Handles large codebases efficiently
- **Enterprise Security**: Self-hosted with complete data privacy
- **Monitoring & Observability**: Comprehensive metrics and health checks

## 📖 **Documentation**

### **Getting Started**
- **[Quick Start Guide](./docs/QUICKSTART.md)** - Installation and basic usage
- **[Architecture Overview](./docs/ARCHITECTURE.md)** - System design and components
- **[API Documentation](./docs/)** - Complete API reference

### **Advanced Usage**
- **[Benchmarking Guide](./benchmarks/README.md)** - Performance testing and validation
- **[Configuration Reference](./configs/)** - System configuration options
- **[Deployment Guide](./infra/)** - Production deployment instructions

### **Development**
- **[Contributing Guide](./docs/BENEFITS.md)** - How to contribute to the project
- **[Agent Integration](./docs/AGENT_INTEGRATION.md)** - AI assistant integration
- **[Development Scripts](./scripts/)** - Utility scripts for development

## 🛠️ **Development**

```bash
# Install dependencies
npm install

# Build the project
npm run build

# Run tests
npm test

# Start development server
npm run dev

# Run benchmarks
npm run benchmark:smoke
```

## 📊 **Benchmarking**

Lens includes comprehensive benchmarking infrastructure:

```bash
# Quick smoke test
npm run benchmark:smoke

# Full performance suite
npm run benchmark:full

# Generate performance reports
npm run benchmark:report
```

See [`/benchmarks`](./benchmarks/) for detailed benchmarking documentation and results.

## 🚢 **Deployment**

Lens supports multiple deployment methods:

```bash
# Development deployment
npm run deploy

# Production deployment with monitoring
npm run deploy:production

# Infrastructure management
cd infra/ && docker-compose up
```

See [`/infra`](./infra/) for complete infrastructure documentation.

## 🔧 **Configuration**

System configuration is centralized in [`/configs`](./configs/):

- **[`/configs/settings`](./configs/settings/)** - Application settings
- **[`/configs/policies`](./configs/policies/)** - Security and access policies  
- **[`/configs/benchmarks`](./configs/benchmarks/)** - Benchmark configurations

## 📈 **Monitoring & Reports**

Generated reports and metrics are stored in [`/reports`](./reports/) (git-ignored):

- Performance benchmarks and analysis
- Coverage reports and test results
- System monitoring data and dashboards
- Generated artifacts and build outputs

## 🤝 **Contributing**

We welcome contributions! Please see:

- **[Development Guide](./docs/BENEFITS.md)** - How to get started
- **[Architecture Documentation](./docs/ARCHITECTURE.md)** - System overview
- **[Utility Scripts](./scripts/)** - Development tools and automation

## 📞 **Support & Community**

- **Documentation**: Complete guides in [`/docs`](./docs/)
- **Issues**: GitHub Issues for bugs and feature requests
- **Discussions**: GitHub Discussions for questions and ideas

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---

**Built with ❤️ for developers who need fast, intelligent code search.**

> 💡 **Tip**: Start with the [`/docs`](./docs/) directory for comprehensive documentation, or explore [`/benchmarks`](./benchmarks/) to see performance validation results.