# HTTP API Reference

The `lens serve` command exposes a REST API for searching, indexing, and
monitoring the code index. The default base URL is
`http://127.0.0.1:3000`.

All responses are JSON. Errors use the structured [`LensError`](../packages/lens-common/src/error.rs)
format. For example:

```json
{
  "SearchError": {
    "message": "Query cannot be empty",
    "query": ""
  }
}
```

## Search Endpoints

### `GET /search`

Execute a text search.

Query parameters:

| Name | Description |
| ---- | ----------- |
| `q` (required) | Search query string. |
| `limit` | Number of results to return (default `10`, max `100`). |
| `offset` | Number of results to skip (default `0`). |
| `fuzzy` | `true` to force fuzzy matching. |
| `symbols` | `true` to search only symbol index. |
| `language` | Filter by programming language name. |
| `file_pattern` | Filter by file path glob. |

Example:

```bash
curl "http://127.0.0.1:3000/search?q=async+fn&limit=5"
```

### `GET /search/fuzzy`

Shortcut for `/search?fuzzy=true`.

### `GET /search/symbol`

Shortcut for `/search?symbols=true` (symbol-only search).

### `GET /search/exact`

Shortcut for exact matching.

## Index Management

### `POST /index`

Trigger indexing of a directory.

Request body:

```json
{
  "directory": "/path/to/project",
  "force": false
}
```

Response:

```json
{
  "success": true,
  "files_indexed": 123,
  "files_failed": 0,
  "lines_indexed": 42000,
  "symbols_extracted": 512,
  "duration_ms": 8123
}
```

### `POST /optimize`

Optimize the index (merge segments). Response includes a success flag and
message.

### `POST /clear`

Delete all index data and reinitialise the Tantivy index.

## Monitoring

### `GET /stats`

Retrieve index statistics such as document count, on-disk size, and list of
languages.

### `GET /health`

Return a lightweight health summary including server version, uptime seconds,
and index readiness flag.

## Error Responses

When a request fails the server returns the appropriate HTTP status alongside a
`LensError` payload. Common cases:

- `400 BAD REQUEST` with `SearchError` for invalid query parameters.
- `404 NOT FOUND` when indexing a directory that does not exist (`IoError`).
- `500 INTERNAL SERVER ERROR` with `IndexError` when underlying Tantivy
  operations fail.

Example error when requesting an invalid limit:

```http
HTTP/1.1 400 Bad Request
Content-Type: application/json

{
  "ConfigError": {
    "message": "Invalid limit parameter: must be between 1 and 100",
    "field": "limit"
  }
}
```
