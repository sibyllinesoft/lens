# LSP Server Guide

Lens ships with a fully asynchronous Language Server Protocol implementation
powered by [`tower-lsp`](https://docs.rs/tower-lsp). The server exposes the same
search capabilities as the HTTP API and integrates with editors that speak LSP.

## Starting the Server

The server can run in stdio mode (default) or on a TCP port for debugging.

```bash
# stdio mode (recommended)
lens lsp

# TCP mode (useful for developing client integrations)
lens lsp --tcp --port 9257
```

The LSP server shares the same index as the CLI and HTTP components. Configure
the index path via `lens.toml` or `--index-path` if you want to point the server
at a different repository.

## Workspace Management

The server watches workspace folders using the settings in `[search]` and `[lsp]`
of `lens.toml`:

- `ignored_directories` and `ignored_file_patterns` control which files are
  indexed and watched.
- `workspace_exclude_patterns` defines glob patterns that the workspace manager
  ignores entirely (defaults exclude `node_modules`, `target`, `.git`, etc.).

Changes detected by the watcher trigger incremental re-indexing.

## Capabilities

- Text, fuzzy, and symbol searches (`workspace/symbol`, `textDocument/references`).
- Completion suggestions backed by the search index.
- Notifications when the index is rebuilding (`window/logMessage`).

## Graceful Shutdown

The server handles the `shutdown`/`exit` LSP requests and also listens for
`SIGINT`/`Ctrl+C`. When a shutdown signal is received the server stops accepting
new requests and finishes in-flight work before exiting.

## Editor Integration Tips

- VS Code: configure the `command` to invoke `lens lsp` in your custom LSP
  client configuration.
- Neovim (`nvim-lspconfig`): set `cmd = {"lens", "lsp"}` and adjust
  `root_dir` detection to match your project layout.
- Helix: add a language server entry invoking `lens lsp --tcp --port 9257` and
  configure the corresponding socket port.

Refer to `docs/configuration.md` for details on tuning search result limits and
other behaviour exposed to LSP clients.
