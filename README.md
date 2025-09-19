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
archive/            # Legacy scripts and documentation retained for reference
```

## Getting Started

1. Install the Rust toolchain (`rustup default stable`).
2. Build the project:
   ```bash
   cargo build --release
   ```
3. Index a repository:
   ```bash
   cargo run -- index /path/to/project
   ```
4. Start the HTTP API:
   ```bash
   cargo run -- serve --bind 127.0.0.1 --port 3000
   ```
5. Start the LSP server (stdio mode):
   ```bash
   cargo run -- lsp
   ```

The CLI exposes additional commands; run `lens --help` for a full list.

## Configuration

Lens loads settings from `lens.toml`. Generate a starter file with
`lens config init` and review the [configuration reference](docs/configuration.md)
for all available options. Environment variables prefixed with `LENS_` override
file values.

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

Pull requests are welcome. Please keep new code documented and covered by unit
or integration tests.

## License

Licensed under the MIT License. See `LICENSE` for details.
