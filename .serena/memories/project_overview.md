# Lens Project Overview

## Purpose
Lens is a single-binary, Rust-first code search platform. It provides a
Tantivy-backed indexer, an Axum HTTP API, and a `tower-lsp` language server for
editor integrations. The repository previously contained a large TypeScript
prototype and simulated Python/Node tooling; those artifacts have now been
removed to keep the workspace focused on the production Rust implementation.

## Key Components
- **apps/lens-core** — CLI entry point plus the Axum HTTP service
- **packages/search-engine** — Tantivy-based indexing and query execution
- **packages/lsp-server** — `tower-lsp` handlers that route against the shared
  search engine library
- **packages/lens-common / lens-config** — shared data types and configuration

Supporting material (experiment matrices, legacy docs) lives in `experiments/`
and `archive/` respectively. Generated artifacts should be written to
`artifacts/` or `reports/` (ignored by git).

## Production Concerns
- **Authentication** — Incoming HTTP requests must present a configured token in
  the `Authorization` header. Tokens are managed via `[http.auth]` in
  `lens.toml` or environment overrides.
- **Telemetry** — `tracing` spans are emitted for all HTTP/LSP handlers.
  Optional OpenTelemetry export is available through the `[telemetry]`
  configuration (OTLP via tonic runtime).
- **Resource limits** — `lens-config` exposes knobs for request timeouts, body
  limits, Tantivy heap sizing, and LSP result caps.

## Status
- Rust workspace is production-ready; simulations have been purged.
- HTTP and LSP surfaces instrumented with real tracing spans.
- Repository is clean of fabricated monitoring/benchmark scripts; new tooling
  should be implemented directly against the Rust API surface.
