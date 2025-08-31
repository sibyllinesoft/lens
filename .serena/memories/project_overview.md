# Lens Project Overview

## Purpose
Lens is a high-performance, local sharded code search system with three-layer processing pipeline:
1. **Layer 1 - Lexical+Fuzzy**: N-gram/trigram inverted index + FST-based fuzzy search (2-8ms target)
2. **Layer 2 - Symbol/AST**: Definitions/references via universal-ctags and LSIF + tree-sitter parsing (3-10ms target)  
3. **Layer 3 - Semantic Rerank**: Optional ColBERT-v2/SPLADE-v2 reranking of top-K candidates (5-15ms target)

Overall p95 latency target: <20ms

## Key Features
- Sharded trigram+FST index with fuzzy search (≤2-edit distance)
- Symbol/AST analysis with structural selectors
- Optional semantic reranking with vector search
- NATS/JetStream work distribution
- Memory-mapped append-only segments with compaction
- Full OpenTelemetry observability integration
- Paper-grade benchmarking system

## Architecture Constraints
The system is validated by Arbiter through `architecture.cue` which enforces:
- Performance SLAs (stage timing, overall p95)
- Resource boundaries (memory, concurrency limits)  
- API contract validation
- Technology stack consistency

## Current Implementation Status
- Basic TypeScript foundation with Fastify server
- Core types and interfaces defined
- OpenTelemetry tracing integration started
- Modular architecture with separate indexer, storage, API layers
- Ready for benchmarking system implementation per TODO.md specifications