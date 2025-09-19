# API Reference

Lens provides a RESTful HTTP API for searching indexed code. This document describes all available endpoints, parameters, and response formats.

## Base URL

When running the Lens server locally:
```
http://localhost:3000
```

## Authentication

Currently, Lens does not require authentication. This may change in future versions for production deployments.

## Endpoints

### GET /health

Returns the health status of the Lens server.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.1.0",
  "uptime_seconds": 3600
}
```

**Status Codes:**
- `200` - Server is healthy
- `503` - Server is unhealthy

### GET /stats

Returns statistics about the search index.

**Response:**
```json
{
  "total_documents": 1247,
  "total_size_bytes": 2457600,
  "index_size_bytes": 891234,
  "last_updated": "2024-03-15T10:30:00Z",
  "languages": {
    "rust": 543,
    "typescript": 324,
    "python": 289,
    "javascript": 91
  }
}
```

**Status Codes:**
- `200` - Success
- `500` - Index unavailable

### GET /search

Execute a search against the Tantivy-backed index.

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `q` | string | Yes | – | Query text. Prefix with `exact:`/`fuzzy:`/`symbol:` to select a search mode. |
| `limit` | integer | No | 50 | Maximum number of documents to return (1-100). |
| `offset` | integer | No | 0 | Number of matching documents to skip before returning results (0-100000). |
| `fuzzy` | boolean | No | false | Apply fuzzy matching (equivalent to `fuzzy:` prefix). |
| `symbols` | boolean | No | false | Restrict search to symbol fields (equivalent to `symbol:` prefix). |
| `language` | string | No | – | Filter results by language (e.g. `rust`, `python`). |
| `file_pattern` | string | No | – | Return only results whose path matches the substring/regex fragment. |

**Inline tokens:** You can embed filters directly inside `q`, e.g. `lang:rust path:src/search limit:20 offset:10 exact:"builder"`.

**Examples:**

```bash
# Basic search
GET /search?q=function%20main

# Limited results
GET /search?q=SearchEngine&limit=10

# Symbol search only
GET /search?q=impl&symbols=true

# Language-specific search
GET /search?q=async&language=rust

# File path filter
GET /search?q=builder&file_pattern=src/search

# Paginate with limit/offset
GET /search?q=builder&limit=10&offset=10
```

**Response:**
```json
{
  "query": "builder",
  "query_type": "text",
  "total": 3,
  "limit": 10,
  "offset": 10,
  "duration_ms": 18,
  "from_cache": false,
  "results": [
    {
      "file_path": "packages/search-engine/src/lib.rs",
      "line_number": 412,
      "content": "let mut query_builder = QueryBuilder::new(&params.q).limit(params.limit);",
      "score": 12.37,
      "language": "rust",
      "matched_terms": ["builder"],
      "context_lines": [
        "    if params.q.trim().is_empty() {",
        "        return Err((",
        "            StatusCode::BAD_REQUEST,",
        "            Json(ErrorResponse {"
      ]
    }
  ],
  "index_stats": {
    "total_documents": 1250,
    "index_size_bytes": 1843200,
    "index_size_human": "1.8 MiB",
    "supported_languages": 12,
    "average_document_size": 98.2,
    "last_updated": "2025-09-18T20:30:12.123Z"
  }
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | Normalized query text executed against the index |
| `query_type` | string | Search mode (`text`, `exact`, `fuzzy`, `symbol`) |
| `total` | integer | Number of hits returned in this page |
| `limit` | integer | Page size applied to the query |
| `offset` | integer | Starting position of this page |
| `duration_ms` | integer | Server-side execution time in milliseconds |
| `from_cache` | boolean | `true` if the query response came from the in-memory cache |
| `results` | array | Array of search result objects |
| `index_stats` | object | Snapshot of index metadata used for the query |

**Status Codes:**
- `200` – Success
- `400` – Invalid parameters (missing `q`, invalid `limit`, etc.)
- `500` – Internal search error

## Search Query Syntax

### Basic Queries

```bash
# Simple text search
GET /search?q=function

# Phrase search (with quotes)
GET /search?q="Hello world"

# Multiple terms (AND)
GET /search?q=async function
```

### Advanced Queries

```bash
# Wildcard search
GET /search?q=get*

# Regular expressions (if enabled)
GET /search?q=fn\s+\w+

# Symbol-specific search
GET /search?q=struct User&symbols=true
```

## Rate Limiting

The Lens API implements rate limiting to prevent abuse:

- **Default Limit**: 100 requests per minute per IP
- **Burst Limit**: 20 requests in 10 seconds

When rate limited, the API returns:
```json
{
  "error": "Rate limit exceeded",
  "retry_after": 60
}
```

**Response Headers:**
- `X-RateLimit-Limit`: Requests allowed per period
- `X-RateLimit-Remaining`: Requests remaining in period
- `X-RateLimit-Reset`: Unix timestamp when period resets

## CORS Support

CORS is disabled by default. Enable it when starting the server:

```bash
lens serve --cors
```

This allows cross-origin requests from any domain. For production, configure specific origins:

```toml
[server]
cors_origins = ["https://myapp.com", "https://dev.myapp.com"]
```

## WebSocket Support

Lens supports WebSocket connections for real-time search updates:

### Connect to WebSocket

```javascript
const ws = new WebSocket('ws://localhost:3000/ws');

ws.onopen = function() {
    console.log('Connected to Lens WebSocket');
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Search results:', data);
};
```

### Send Search Query

```javascript
ws.send(JSON.stringify({
    type: 'search',
    query: 'function main',
    limit: 10
}));
```

### Response Format

```json
{
  "type": "search_results",
  "request_id": "12345",
  "results": [...],
  "total": 5,
  "query_time_ms": 3
}
```

## Error Handling

### Error Response Format

All API errors return a consistent JSON format:

```json
{
  "error": "Human-readable error message",
  "message": "Detailed error description",
  "code": "ERROR_CODE",
  "request_id": "req_12345",
  "timestamp": "2024-03-15T10:30:00Z"
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_PARAMETER` | 400 | Invalid query parameter |
| `QUERY_TOO_LONG` | 400 | Query exceeds maximum length |
| `QUERY_TOO_COMPLEX` | 400 | Query is too complex to execute |
| `INDEX_UNAVAILABLE` | 503 | Search index is not available |
| `SEARCH_TIMEOUT` | 504 | Search operation timed out |
| `RATE_LIMITED` | 429 | Too many requests |

## Client Libraries

### JavaScript/TypeScript

```javascript
import { LensClient } from '@sibyllinesoft/lens-client';

const lens = new LensClient('http://localhost:3000');

// Search
const results = await lens.search('function main', { limit: 10 });

// Get stats
const stats = await lens.getStats();

// Health check
const health = await lens.getHealth();
```

### Python

```python
from lens_client import LensClient

lens = LensClient('http://localhost:3000')

# Search
results = lens.search('function main', limit=10)

# Get stats
stats = lens.get_stats()

# Health check
health = lens.get_health()
```

### Rust

```rust
use lens_client::LensClient;

let lens = LensClient::new("http://localhost:3000");

// Search
let results = lens.search("function main")
    .limit(10)
    .execute()
    .await?;

// Get stats
let stats = lens.get_stats().await?;

// Health check
let health = lens.get_health().await?;
```

## Performance Considerations

### Query Optimization

1. **Use Specific Terms**: More specific queries return results faster
2. **Limit Results**: Use appropriate `limit` values to reduce response time
3. **Filter by Language**: Language-specific searches are more efficient
4. **Use Symbol Search**: `symbols=true` for faster symbol-only searches

### Caching

The Lens API implements intelligent caching:

- **Query Results**: Cached for 5 minutes
- **Statistics**: Cached for 1 minute
- **Health Status**: Cached for 30 seconds

Cache headers are included in responses:
```
Cache-Control: public, max-age=300
ETag: "abc123def456"
Last-Modified: Wed, 15 Mar 2024 10:30:00 GMT
```

## Monitoring and Metrics

### Metrics Endpoint

```bash
GET /metrics
```

Returns Prometheus-compatible metrics:
```
# HELP lens_search_requests_total Total number of search requests
# TYPE lens_search_requests_total counter
lens_search_requests_total{status="success"} 1247
lens_search_requests_total{status="error"} 23

# HELP lens_search_duration_seconds Search request duration
# TYPE lens_search_duration_seconds histogram
lens_search_duration_seconds_bucket{le="0.01"} 1023
lens_search_duration_seconds_bucket{le="0.1"} 1200
```

### Health Checks

Use the health endpoint for monitoring:

```bash
# Simple check
curl -f http://localhost:3000/health

# Detailed check with timeout
curl --max-time 5 http://localhost:3000/health
```

---

**Next**: [CLI Reference](cli-reference.md) | **Previous**: [Getting Started](getting-started.md)
