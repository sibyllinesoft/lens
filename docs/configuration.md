# Configuration Guide

Lens reads configuration from a `lens.toml` file. By default the application
looks in the current working directory, `config/lens.toml`, and
`$XDG_CONFIG_HOME/lens/lens.toml`. Environment variables prefixed with
`LENS_` override file settings (e.g. `LENS_HTTP__PORT=8080`).

Generate a starter configuration with:

```bash
lens config init
```

Validate a custom file with:

```bash
lens config validate --file path/to/lens.toml
```

## Sample `lens.toml`

```toml
[app]
log_level = "info"
enable_file_watching = true
# worker_threads = 8

[search]
index_path = "./index"
max_results = 50
enable_fuzzy = true
fuzzy_distance = 2
heap_size_mb = 128
commit_interval_ms = 5000
enable_cache = true
cache_size = 1000
supported_extensions = [".rs", ".py", ".ts", ".js"]
ignored_directories = [".git", "node_modules", "target"]
ignored_file_patterns = ["*.min.js", "*.map", "*.lock"]

[lsp]
max_search_results = 50
enable_fuzzy_search = true
enable_semantic_search = false
search_debounce_ms = 300
enable_result_caching = true
workspace_exclude_patterns = [
  "**/node_modules/**",
  "**/target/**",
  "**/.git/**",
  "**/dist/**",
  "**/build/**",
  "**/__pycache__/**",
]

[http]
bind = "127.0.0.1"
port = 3000
enable_cors = false
request_timeout_secs = 30
max_body_size = 1048576
```

## Settings Reference

### `[app]`

| Field | Description |
| ----- | ----------- |
| `log_level` | Global log level (`trace`, `debug`, `info`, `warn`, `error`). |
| `enable_file_watching` | Enable automatic reindexing when files change. |
| `worker_threads` | Optional number of Tokio worker threads. When omitted the runtime chooses a sensible default. |

### `[search]`

| Field | Description |
| ----- | ----------- |
| `index_path` | Directory where Tantivy index files are stored. |
| `max_results` | Default maximum number of results returned by search endpoints. |
| `enable_fuzzy` / `fuzzy_distance` | Enable fuzzy search and configure maximum edit distance. |
| `heap_size_mb` | Tantivy writer heap size (per writer). |
| `commit_interval_ms` | Background commit interval for the index writer. |
| `enable_cache` / `cache_size` | Toggle the in-memory query cache and set its capacity. |
| `supported_extensions` | File extensions (with leading dot) that will be indexed. |
| `ignored_directories` | Directory names that are skipped during indexing. |
| `ignored_file_patterns` | Glob patterns of files to exclude (matched per file name). |

### `[lsp]`

| Field | Description |
| ----- | ----------- |
| `max_search_results` | Maximum number of results returned for LSP requests. |
| `enable_fuzzy_search` | Enable fuzzy matching in LSP search requests. |
| `enable_semantic_search` | Toggle semantic features in the LSP server. |
| `search_debounce_ms` | Debounce interval applied to frequent search requests. |
| `enable_result_caching` | Cache LSP search results. |
| `workspace_exclude_patterns` | Glob patterns excluded from the workspace watcher. |

### `[http]`

| Field | Description |
| ----- | ----------- |
| `bind` | Interface address for the HTTP server. |
| `port` | TCP port for the HTTP server. |
| `enable_cors` | Allow cross-origin requests (uses permissive CORS). |
| `request_timeout_secs` | Request timeout applied by Tower HTTP middleware. |
| `max_body_size` | Maximum size in bytes for request bodies. |
