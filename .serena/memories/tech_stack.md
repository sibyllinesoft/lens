# Technology Stack

## Languages
- **Rust** — Primary implementation language for the CLI, HTTP API, and LSP server
- **TOML** — Configuration format (`lens.toml`)
- **Shell** — Thin helper scripts (build/CI)

## Core Dependencies
- **Tantivy** — Inverted index powering search and filtering
- **Axum + Tower HTTP** — HTTP routing, CORS, tracing layers, middleware stack
- **tower-lsp** — Language Server Protocol plumbing backed by the same search engine
- **Tracing + OpenTelemetry** — Structured logging and optional OTLP export
- **Tokio** — Async runtime used across all binaries and libraries

## Configuration & Instrumentation
- **lens-config** — Shared loader using the `config` crate with environment overrides
- **Telemetry** — `tracing` spans instrument all HTTP/LSP handlers; optional OTLP exporter via `opentelemetry_otlp`
- **Authentication** — Static token middleware enforced by Axum before all APIs

## Tooling
- **cargo fmt / clippy / test** — Standard Rust formatting, linting, and testing
- **experiments/** — Versioned YAML matrices retained for reference (no fake metrics scripts)
- **artifacts/** (ignored) — Recommended output directory for reports or benchmarking data

The repository no longer depends on Fastify, NATS, or simulated Python ML stacks; any operational automation should call the real Rust services directly.
