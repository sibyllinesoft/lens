# Lens

Lens is a Rust-based code search platform that combines a high-performance
Tantivy index, a CLI, an HTTP API, and a production-ready Language Server. It
replaces the earlier TypeScript prototype with a cohesive Rust workspace.

## Workspace Layout

```
apps/
  lens-core/        # CLI binary and Axum HTTP server
packages/
  search-engine/    # Tantivy indexing/search library
  lsp-server/       # tower-lsp implementation
  lens-common/      # shared data types and utilities
experiments/        # Historical experiment matrices kept for reference
archive/            # Legacy docs from the retired TypeScript prototype
```

## Getting Started

1. Install the Rust toolchain (`rustup default stable`).
2. Build the release binary:
   ```bash
   cargo build --release
   ```
3. Index a repository using the compiled binary:
   ```bash
   ./target/release/lens index /path/to/project
   ```
4. Configure an API token. Either set an environment variable:
   ```bash
   export LENS_HTTP__AUTH__ENABLED=true
   export LENS_HTTP__AUTH__TOKENS=my-local-token
   ```
   or edit `lens.toml` (see below).
5. Start the HTTP API with the release binary:
   ```bash
   ./target/release/lens serve --bind 127.0.0.1 --port 3000
   ```
6. Start the LSP server (stdio mode):
   ```bash
   ./target/release/lens lsp
   ```

The CLI exposes additional commands; run `lens --help` for a full list.

## Configuration

Lens loads settings from `lens.toml`. Generate a starter file with
`lens config init` and review the [configuration reference](docs/configuration.md)
for all available options. Environment variables prefixed with `LENS_` override
file values.

### HTTP Authentication

The HTTP API is protected by a lightweight token-based middleware. Configure it
with either environment variables (`LENS_HTTP__AUTH__TOKENS`) or the `[http.auth]`
section in `lens.toml`. Provide at least one token before starting the server;
requests must send it via the `Authorization: Bearer <token>` header.

### Telemetry

Tracing is wired through `tracing` + OpenTelemetry. Toggle OTLP export with the
`[telemetry]` section. When enabled the CLI installs a Tokio batch exporter so
spans and LSP/HTTP handler instrumentation are emitted to your collector.

Prometheus-compatible metrics are exposed on `/metrics` once the HTTP server is
running. Point your Prometheus scrape configuration at the Lens instance to
collect request counters, latency histograms, and index statistics.

## HTTP API

The Axum server provides search, indexing, and monitoring endpoints. See
[docs/api.md](docs/api.md) for supported routes and response shapes. Error
responses are encoded as structured `LensError` values.

## Language Server

Lens ships with a `tower-lsp` server that mirrors the search capabilities of the
HTTP API. Documentation and integration tips are available in
[docs/lsp.md](docs/lsp.md).

## Development

- Format the workspace: `cargo fmt`
- Run tests: `cargo test`
- Lint (clippy): `cargo clippy --workspace`

Legacy Python and Node-based simulators have been removed. Operational tooling
should now be implemented against the real Rust surface area (HTTP API + CLI).
Pull requests are welcome—please keep new code documented and covered by unit
or integration tests.

## License

Licensed under the MIT License. See `LICENSE` for details.
