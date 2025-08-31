# Technology Stack

## Languages (Validated in architecture.cue)
- **TypeScript** (Primary) - Modern ES2022, strict typing, NodeNext modules
- **Python** - For analysis/ML components  
- **Rust** - For performance-critical components
- **Bash** - For scripts and utilities

## Core Dependencies
- **Fastify** - Web framework with CORS support
- **NATS** - Message streaming for work distribution
- **OpenTelemetry** - Full observability (tracing, metrics, instrumentation)
- **Pino** - Structured logging
- **Zod** - Runtime type validation
- **UUID** - Trace ID generation
- **fast-fuzzy** - Fuzzy string matching

## Development Tools
- **TypeScript 5.9.2** with strict configuration
- **Vitest** - Testing framework with coverage (85% thresholds)
- **ESLint** - TypeScript linting
- **Prettier** - Code formatting
- **tsx** - Development server with hot reload
- **CUE** - Architecture validation and constraints

## Key Architecture Decisions
- **Memory-mapped segments** for storage (append-only with compaction)
- **NATS/JetStream** for work unit distribution
- **OpenTelemetry** for all observability
- **ColBERT-v2/SPLADE-v2** for semantic models
- **HNSW** for vector search in rerank stage