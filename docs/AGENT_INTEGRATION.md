# Lens Agent Integration Guide v1.0

**Complete guide for AI agents, IDE plugins, and automated tools to integrate with Lens code search**

This guide provides schemas, examples, and best practices for integrating Lens into AI agents, development tools, and automated systems.

## 🎯 Overview

Lens v1.0 provides a RESTful API with version management, comprehensive search capabilities, and compatibility checking. This guide shows how to:

- **Query the API programmatically** with proper error handling
- **Validate version compatibility** before making requests  
- **Process search results** with complete type safety
- **Handle Unicode and special characters** correctly
- **Implement retry logic** for production reliability

## 📋 Quick Integration Checklist

✅ **Version Check**: Verify API compatibility before first use  
✅ **Error Handling**: Implement proper HTTP status code handling  
✅ **Unicode Support**: Handle emoji, CRLF, and special characters  
✅ **Retry Logic**: Handle transient failures gracefully  
✅ **Rate Limiting**: Respect API rate limits  
✅ **Tracing**: Use trace IDs for debugging  

---

## 🔧 API Schema Reference

### Core Request/Response Types

```typescript
// Version Management
interface CompatibilityCheck {
  api_version: 'v1';
  index_version: 'v1'; 
  policy_version: 'v1';
  allow_compat?: boolean;
}

interface CompatibilityResponse {
  compatible: boolean;
  current_version: {
    api_version: 'v1';
    index_version: 'v1';
    policy_version: 'v1';
  };
  warnings: string[];
  errors: string[];
}

// Search Request
interface SearchRequest {
  repo_sha: string;        // Repository identifier
  q: string;              // Search query
  mode: 'lex' | 'struct' | 'hybrid';
  k?: number;             // Max results (default: 10, max: 100)
  fuzzy?: number;         // Edit distance (0-3, default: 1)
  context?: number;       // Context lines (0-5, default: 0)
}

// Search Response
interface SearchResponse {
  hits: SearchHit[];
  total: number;
  latency_ms: LatencyBreakdown;
  trace_id: string;
  api_version: 'v1';
  index_version: 'v1'; 
  policy_version: 'v1';
}

interface SearchHit {
  file: string;           // Relative file path
  line: number;           // 1-based line number
  col: number;            // 0-based column (Unicode code points)
  lang?: string;          // Language identifier
  snippet?: string;       // Code snippet with match
  score: number;          // Relevance score (0-1)
  why: MatchReason[];     // Match reasoning
  
  // Optional metadata
  ast_path?: string;      // AST path for structural matches
  symbol_kind?: SymbolKind;
  byte_offset?: number;
  span_len?: number;
  context_before?: string;
  context_after?: string;
}

type MatchReason = 'exact' | 'fuzzy' | 'symbol' | 'struct' | 'semantic';
type SymbolKind = 'function' | 'class' | 'variable' | 'type' | 'interface' | 'constant' | 'enum' | 'method' | 'property';

interface LatencyBreakdown {
  stage_a: number;        // Lexical search (ms)
  stage_b: number;        // Structural search (ms) 
  stage_c: number;        // Semantic reranking (ms)
  total: number;          // Total latency (ms)
}
```

### Error Response Schema

```typescript
interface ErrorResponse {
  error: {
    type: string;
    message: string;
    details?: Record<string, any>;
  };
  trace_id: string;
  api_version: 'v1';
  index_version: 'v1';
  policy_version: 'v1';
}

// Common error types
type ErrorType = 
  | 'version_mismatch'     // Version compatibility issue
  | 'invalid_request'      // Malformed request
  | 'repo_not_found'       // Repository not indexed
  | 'query_timeout'        // Search exceeded time limits
  | 'rate_limited'         // Too many requests
  | 'internal_error';      // Server error
```

---

## 🚀 Agent Integration Examples

### 1. TypeScript/Node.js Agent

```typescript
import axios, { AxiosResponse } from 'axios';

class LensClient {
  private baseUrl: string;
  private timeout: number;
  
  constructor(baseUrl = 'http://localhost:3000', timeout = 30000) {
    this.baseUrl = baseUrl;
    this.timeout = timeout;
  }

  // Version compatibility check (call once on startup)
  async checkCompatibility(): Promise<boolean> {
    try {
      const response = await axios.get(
        `${this.baseUrl}/compat/check`,
        {
          params: {
            api_version: 'v1',
            index_version: 'v1', 
            policy_version: 'v1'
          },
          timeout: this.timeout
        }
      );
      
      if (!response.data.compatible) {
        console.warn('Version compatibility warning:', response.data.warnings);
        return false;
      }
      
      return true;
    } catch (error) {
      console.error('Compatibility check failed:', error);
      return false;
    }
  }

  // Main search function with error handling
  async search(
    repoSha: string,
    query: string,
    options: Partial<SearchRequest> = {}
  ): Promise<SearchResponse> {
    const request: SearchRequest = {
      repo_sha: repoSha,
      q: query,
      mode: options.mode || 'hybrid',
      k: Math.min(options.k || 10, 100),
      fuzzy: Math.max(0, Math.min(options.fuzzy || 1, 3)),
      context: Math.max(0, Math.min(options.context || 0, 5))
    };

    const maxRetries = 3;
    let lastError: any;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const response: AxiosResponse<SearchResponse> = await axios.post(
          `${this.baseUrl}/search`,
          request,
          {
            timeout: this.timeout,
            headers: {
              'Content-Type': 'application/json',
              'X-Trace-Id': `agent-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
            }
          }
        );

        // Validate response has required version fields
        if (!this.validateVersionFields(response.data)) {
          throw new Error('Invalid response: missing version fields');
        }

        return response.data;
      } catch (error: any) {
        lastError = error;
        
        if (error.response?.status === 429) {
          // Rate limited - exponential backoff
          const delay = Math.pow(2, attempt) * 1000;
          console.warn(`Rate limited, retrying in ${delay}ms...`);
          await new Promise(resolve => setTimeout(resolve, delay));
          continue;
        }
        
        if (error.response?.status >= 500 && attempt < maxRetries) {
          // Server error - retry with backoff
          const delay = attempt * 2000;
          console.warn(`Server error, retrying in ${delay}ms...`);
          await new Promise(resolve => setTimeout(resolve, delay));
          continue;
        }
        
        break; // Don't retry for client errors (4xx)
      }
    }
    
    throw lastError;
  }

  // Health check
  async health(): Promise<{ status: string; version: string }> {
    const response = await axios.get(`${this.baseUrl}/health`, {
      timeout: 5000
    });
    return response.data;
  }

  // Nightly bundle compatibility check
  async checkNightlyCompatibility(): Promise<any> {
    const response = await axios.get(`${this.baseUrl}/compat/bundles`, {
      timeout: 10000
    });
    return response.data;
  }

  private validateVersionFields(response: any): boolean {
    return response.api_version === 'v1' &&
           response.index_version === 'v1' &&
           response.policy_version === 'v1';
  }
}

// Usage example
async function exampleUsage() {
  const client = new LensClient();
  
  // Check compatibility on startup
  if (!await client.checkCompatibility()) {
    throw new Error('Lens server version incompatible');
  }
  
  try {
    const results = await client.search('main', 'function calculateTotal', {
      mode: 'hybrid',
      k: 20,
      fuzzy: 1,
      context: 2
    });
    
    console.log(`Found ${results.total} results in ${results.latency_ms.total}ms`);
    
    results.hits.forEach(hit => {
      console.log(`${hit.file}:${hit.line}:${hit.col} - ${hit.snippet}`);
      console.log(`  Score: ${hit.score}, Reason: ${hit.why.join(', ')}`);
    });
    
  } catch (error) {
    console.error('Search failed:', error);
  }
}
```

### 2. Python Agent

```python
import requests
import json
import time
import random
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

@dataclass
class SearchHit:
    file: str
    line: int
    col: int
    score: float
    why: List[str]
    lang: Optional[str] = None
    snippet: Optional[str] = None
    ast_path: Optional[str] = None
    symbol_kind: Optional[str] = None
    byte_offset: Optional[int] = None
    span_len: Optional[int] = None
    context_before: Optional[str] = None
    context_after: Optional[str] = None

@dataclass
class SearchResponse:
    hits: List[SearchHit]
    total: int
    latency_ms: Dict[str, int]
    trace_id: str
    api_version: str
    index_version: str
    policy_version: str

class LensClient:
    def __init__(self, base_url: str = "http://localhost:3000", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def check_compatibility(self) -> bool:
        """Check version compatibility with Lens server."""
        try:
            response = self.session.get(
                f"{self.base_url}/compat/check",
                params={
                    'api_version': 'v1',
                    'index_version': 'v1',
                    'policy_version': 'v1'
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            if not data.get('compatible', False):
                print(f"Version compatibility warning: {data.get('warnings', [])}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Compatibility check failed: {e}")
            return False
    
    def search(
        self,
        repo_sha: str,
        query: str,
        mode: str = 'hybrid',
        k: int = 10,
        fuzzy: int = 1,
        context: int = 0,
        max_retries: int = 3
    ) -> SearchResponse:
        """Search code with retry logic and proper error handling."""
        
        request_data = {
            'repo_sha': repo_sha,
            'q': query,
            'mode': mode,
            'k': min(k, 100),
            'fuzzy': max(0, min(fuzzy, 3)),
            'context': max(0, min(context, 5))
        }
        
        last_error = None
        
        for attempt in range(1, max_retries + 1):
            try:
                trace_id = f"agent-{int(time.time())}-{random.randint(1000, 9999)}"
                
                response = self.session.post(
                    f"{self.base_url}/search",
                    json=request_data,
                    headers={
                        'Content-Type': 'application/json',
                        'X-Trace-Id': trace_id
                    },
                    timeout=self.timeout
                )
                
                if response.status_code == 429:
                    # Rate limited - exponential backoff
                    delay = 2 ** attempt
                    print(f"Rate limited, retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                
                if response.status_code >= 500 and attempt < max_retries:
                    # Server error - retry with backoff  
                    delay = attempt * 2
                    print(f"Server error, retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                
                response.raise_for_status()
                
                data = response.json()
                
                # Validate version fields
                if not self._validate_version_fields(data):
                    raise ValueError("Invalid response: missing version fields")
                
                # Convert to typed response
                hits = [
                    SearchHit(**hit) for hit in data['hits']
                ]
                
                return SearchResponse(
                    hits=hits,
                    total=data['total'],
                    latency_ms=data['latency_ms'],
                    trace_id=data['trace_id'],
                    api_version=data['api_version'],
                    index_version=data['index_version'],
                    policy_version=data['policy_version']
                )
                
            except Exception as e:
                last_error = e
                if response.status_code < 500:
                    break  # Don't retry client errors
        
        raise last_error or Exception("Search failed after retries")
    
    def health(self) -> Dict[str, Union[str, bool]]:
        """Check server health."""
        response = self.session.get(f"{self.base_url}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    
    def _validate_version_fields(self, data: Dict) -> bool:
        """Validate response contains required version fields."""
        return (
            data.get('api_version') == 'v1' and
            data.get('index_version') == 'v1' and
            data.get('policy_version') == 'v1'
        )

# Usage example
def example_usage():
    client = LensClient()
    
    # Check compatibility on startup
    if not client.check_compatibility():
        raise Exception("Lens server version incompatible")
    
    try:
        results = client.search(
            repo_sha='main',
            query='async def process_data',
            mode='struct',
            k=15,
            fuzzy=1,
            context=2
        )
        
        print(f"Found {results.total} results in {results.latency_ms['total']}ms")
        
        for hit in results.hits:
            print(f"{hit.file}:{hit.line}:{hit.col} - {hit.snippet}")
            print(f"  Score: {hit.score:.3f}, Reason: {', '.join(hit.why)}")
            
            if hit.context_before:
                print(f"  Context before: {hit.context_before[:50]}...")
            if hit.context_after:
                print(f"  Context after: {hit.context_after[:50]}...")
        
    except Exception as e:
        print(f"Search failed: {e}")
```

### 3. Go Agent

```go
package main

import (
    "bytes"
    "context"
    "encoding/json"
    "fmt"
    "io"
    "math/rand"
    "net/http"
    "net/url"
    "strconv"
    "time"
)

type CompatibilityResponse struct {
    Compatible     bool   `json:"compatible"`
    CurrentVersion struct {
        APIVersion    string `json:"api_version"`
        IndexVersion  string `json:"index_version"`
        PolicyVersion string `json:"policy_version"`
    } `json:"current_version"`
    Warnings []string `json:"warnings"`
    Errors   []string `json:"errors"`
}

type SearchRequest struct {
    RepoSHA string `json:"repo_sha"`
    Query   string `json:"q"`
    Mode    string `json:"mode"`
    K       int    `json:"k,omitempty"`
    Fuzzy   int    `json:"fuzzy,omitempty"`
    Context int    `json:"context,omitempty"`
}

type SearchHit struct {
    File          string   `json:"file"`
    Line          int      `json:"line"`
    Col           int      `json:"col"`
    Lang          *string  `json:"lang,omitempty"`
    Snippet       *string  `json:"snippet,omitempty"`
    Score         float64  `json:"score"`
    Why           []string `json:"why"`
    ASTPath       *string  `json:"ast_path,omitempty"`
    SymbolKind    *string  `json:"symbol_kind,omitempty"`
    ByteOffset    *int     `json:"byte_offset,omitempty"`
    SpanLen       *int     `json:"span_len,omitempty"`
    ContextBefore *string  `json:"context_before,omitempty"`
    ContextAfter  *string  `json:"context_after,omitempty"`
}

type LatencyBreakdown struct {
    StageA int `json:"stage_a"`
    StageB int `json:"stage_b"`
    StageC int `json:"stage_c"`
    Total  int `json:"total"`
}

type SearchResponse struct {
    Hits          []SearchHit      `json:"hits"`
    Total         int              `json:"total"`
    LatencyMS     LatencyBreakdown `json:"latency_ms"`
    TraceID       string           `json:"trace_id"`
    APIVersion    string           `json:"api_version"`
    IndexVersion  string           `json:"index_version"`
    PolicyVersion string           `json:"policy_version"`
}

type LensClient struct {
    baseURL    string
    httpClient *http.Client
}

func NewLensClient(baseURL string, timeout time.Duration) *LensClient {
    return &LensClient{
        baseURL: baseURL,
        httpClient: &http.Client{
            Timeout: timeout,
        },
    }
}

func (c *LensClient) CheckCompatibility(ctx context.Context) (bool, error) {
    u, err := url.Parse(c.baseURL + "/compat/check")
    if err != nil {
        return false, err
    }
    
    q := u.Query()
    q.Set("api_version", "v1")
    q.Set("index_version", "v1")
    q.Set("policy_version", "v1")
    u.RawQuery = q.Encode()
    
    req, err := http.NewRequestWithContext(ctx, "GET", u.String(), nil)
    if err != nil {
        return false, err
    }
    
    resp, err := c.httpClient.Do(req)
    if err != nil {
        return false, err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        return false, fmt.Errorf("compatibility check failed: %d", resp.StatusCode)
    }
    
    var compatResp CompatibilityResponse
    if err := json.NewDecoder(resp.Body).Decode(&compatResp); err != nil {
        return false, err
    }
    
    if !compatResp.Compatible {
        fmt.Printf("Version compatibility warnings: %v\n", compatResp.Warnings)
    }
    
    return compatResp.Compatible, nil
}

func (c *LensClient) Search(ctx context.Context, req SearchRequest, maxRetries int) (*SearchResponse, error) {
    // Validate and constrain request parameters
    if req.K > 100 {
        req.K = 100
    }
    if req.Fuzzy > 3 {
        req.Fuzzy = 3
    }
    if req.Context > 5 {
        req.Context = 5
    }
    
    var lastErr error
    
    for attempt := 1; attempt <= maxRetries; attempt++ {
        result, err := c.doSearch(ctx, req)
        if err == nil {
            return result, nil
        }
        
        lastErr = err
        
        // Check if we should retry
        if httpErr, ok := err.(*HTTPError); ok {
            if httpErr.StatusCode == http.StatusTooManyRequests {
                // Rate limited - exponential backoff
                delay := time.Duration(1<<uint(attempt)) * time.Second
                fmt.Printf("Rate limited, retrying in %v...\n", delay)
                select {
                case <-time.After(delay):
                    continue
                case <-ctx.Done():
                    return nil, ctx.Err()
                }
            }
            
            if httpErr.StatusCode >= 500 && attempt < maxRetries {
                // Server error - retry with linear backoff
                delay := time.Duration(attempt*2) * time.Second
                fmt.Printf("Server error, retrying in %v...\n", delay)
                select {
                case <-time.After(delay):
                    continue
                case <-ctx.Done():
                    return nil, ctx.Err()
                }
            }
            
            // Don't retry client errors (4xx)
            break
        }
    }
    
    return nil, fmt.Errorf("search failed after %d retries: %w", maxRetries, lastErr)
}

type HTTPError struct {
    StatusCode int
    Message    string
}

func (e *HTTPError) Error() string {
    return fmt.Sprintf("HTTP %d: %s", e.StatusCode, e.Message)
}

func (c *LensClient) doSearch(ctx context.Context, req SearchRequest) (*SearchResponse, error) {
    reqBody, err := json.Marshal(req)
    if err != nil {
        return nil, err
    }
    
    traceID := fmt.Sprintf("go-agent-%d-%d", time.Now().Unix(), rand.Intn(10000))
    
    httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/search", bytes.NewReader(reqBody))
    if err != nil {
        return nil, err
    }
    
    httpReq.Header.Set("Content-Type", "application/json")
    httpReq.Header.Set("X-Trace-Id", traceID)
    
    resp, err := c.httpClient.Do(httpReq)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        body, _ := io.ReadAll(resp.Body)
        return nil, &HTTPError{
            StatusCode: resp.StatusCode,
            Message:    string(body),
        }
    }
    
    var searchResp SearchResponse
    if err := json.NewDecoder(resp.Body).Decode(&searchResp); err != nil {
        return nil, err
    }
    
    // Validate version fields
    if searchResp.APIVersion != "v1" || 
       searchResp.IndexVersion != "v1" || 
       searchResp.PolicyVersion != "v1" {
        return nil, fmt.Errorf("invalid response: version field mismatch")
    }
    
    return &searchResp, nil
}

// Usage example
func main() {
    ctx := context.Background()
    client := NewLensClient("http://localhost:3000", 30*time.Second)
    
    // Check compatibility
    compatible, err := client.CheckCompatibility(ctx)
    if err != nil {
        panic(fmt.Sprintf("Compatibility check failed: %v", err))
    }
    if !compatible {
        panic("Lens server version incompatible")
    }
    
    // Search
    results, err := client.Search(ctx, SearchRequest{
        RepoSHA: "main",
        Query:   "func HandleRequest",
        Mode:    "struct",
        K:       20,
        Fuzzy:   1,
        Context: 2,
    }, 3)
    if err != nil {
        panic(fmt.Sprintf("Search failed: %v", err))
    }
    
    fmt.Printf("Found %d results in %dms\n", results.Total, results.LatencyMS.Total)
    
    for _, hit := range results.Hits {
        fmt.Printf("%s:%d:%d - %v\n", hit.File, hit.Line, hit.Col, 
                  hit.Snippet)
        fmt.Printf("  Score: %.3f, Reason: %v\n", hit.Score, hit.Why)
    }
}
```

---

## 🛡️ Best Practices & Error Handling

### Version Management

Always check compatibility before making API calls:

```bash
# Check version compatibility
curl "http://localhost:3000/compat/check?api_version=v1&index_version=v1&policy_version=v1"

# Check nightly bundle compatibility  
curl "http://localhost:3000/compat/bundles"
```

### Unicode and Special Character Handling

Lens correctly handles Unicode, emoji, and special characters in file paths and content:

```json
{
  "hits": [
    {
      "file": "src/测试/emoji-🚀-test.ts",
      "line": 10,
      "col": 15,
      "snippet": "const message = '你好世界 👋';",
      "score": 0.95,
      "why": ["exact"]
    }
  ]
}
```

**Important**: Column positions are in Unicode code points, not bytes.

### Rate Limiting and Retry Logic

Implement exponential backoff for rate limiting (HTTP 429):

```typescript
async function retryWithBackoff<T>(
  fn: () => Promise<T>, 
  maxRetries = 3
): Promise<T> {
  let lastError;
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error: any) {
      lastError = error;
      
      if (error.response?.status === 429) {
        const delay = Math.pow(2, attempt) * 1000; // 2s, 4s, 8s
        await new Promise(resolve => setTimeout(resolve, delay));
        continue;
      }
      
      if (error.response?.status >= 500 && attempt < maxRetries) {
        const delay = attempt * 2000; // 2s, 4s, 6s
        await new Promise(resolve => setTimeout(resolve, delay));
        continue;
      }
      
      throw error; // Don't retry client errors
    }
  }
  
  throw lastError;
}
```

### Trace ID Usage

Always include trace IDs for debugging:

```typescript
const traceId = `agent-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

const response = await fetch(`${baseUrl}/search`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-Trace-Id': traceId
  },
  body: JSON.stringify(searchRequest)
});
```

### Error Response Handling

```typescript
interface LensError {
  error: {
    type: 'version_mismatch' | 'invalid_request' | 'repo_not_found' | 'query_timeout' | 'rate_limited' | 'internal_error';
    message: string;
    details?: any;
  };
  trace_id: string;
}

function handleLensError(error: LensError): void {
  switch (error.error.type) {
    case 'version_mismatch':
      console.error('Version compatibility issue:', error.error.message);
      // Update client or handle gracefully
      break;
    case 'repo_not_found':
      console.error('Repository not indexed:', error.error.message);
      // Trigger indexing or inform user
      break;
    case 'query_timeout':
      console.warn('Query timeout, try simpler search:', error.error.message);
      // Retry with different parameters
      break;
    case 'rate_limited':
      console.warn('Rate limited:', error.error.message);
      // Implement backoff
      break;
    default:
      console.error('Unexpected error:', error.error.message);
  }
}
```

---

## 📊 Performance Guidelines

### Search Optimization

| Mode | Use Case | Latency | Accuracy |
|------|----------|---------|----------|
| `lex` | Fast text search | ~10-50ms | Good |
| `struct` | Code structure search | ~50-150ms | Better |
| `hybrid` | Best results | ~100-300ms | Best |

### Query Patterns

```typescript
// ✅ Good: Specific queries
await client.search('main', 'function handleUserLogin', { mode: 'struct' });

// ✅ Good: Class/interface search  
await client.search('main', 'class UserService', { mode: 'struct' });

// ⚠️ OK: Broader search with limit
await client.search('main', 'authentication', { mode: 'hybrid', k: 20 });

// ❌ Avoid: Very broad queries without limits
await client.search('main', 'test', { k: 100 }); // Too broad
```

### Batch Processing

For multiple queries, use concurrent requests with proper rate limiting:

```typescript
async function batchSearch(queries: string[], concurrency = 5) {
  const semaphore = new Semaphore(concurrency);
  
  const results = await Promise.all(
    queries.map(async query => {
      await semaphore.acquire();
      try {
        return await client.search('main', query);
      } finally {
        semaphore.release();
      }
    })
  );
  
  return results;
}
```

---

## 🔍 Advanced Integration Patterns

### IDE Plugin Integration

```typescript
class VSCodeLensExtension {
  private client: LensClient;
  
  constructor() {
    this.client = new LensClient('http://localhost:3000');
  }
  
  async provideDefinition(document: TextDocument, position: Position): Promise<Definition[]> {
    const wordRange = document.getWordRangeAtPosition(position);
    const word = document.getText(wordRange);
    
    try {
      const results = await this.client.search('main', word, {
        mode: 'struct',
        k: 10
      });
      
      return results.hits.map(hit => ({
        uri: Uri.file(hit.file),
        range: new Range(hit.line - 1, hit.col, hit.line - 1, hit.col + word.length)
      }));
    } catch (error) {
      console.error('Lens search failed:', error);
      return [];
    }
  }
}
```

### AI Agent Integration

```typescript
class CodeSearchAgent {
  private lens: LensClient;
  
  async analyzeCodebase(query: string): Promise<string> {
    // Get search results
    const results = await this.lens.search('main', query, {
      mode: 'hybrid',
      k: 10,
      context: 3
    });
    
    if (results.hits.length === 0) {
      return `No results found for "${query}".`;
    }
    
    // Process results for AI context
    const context = results.hits.map(hit => ({
      location: `${hit.file}:${hit.line}`,
      code: hit.snippet,
      context_before: hit.context_before,
      context_after: hit.context_after,
      relevance: hit.score
    }));
    
    return this.generateAnalysis(context, query);
  }
  
  private generateAnalysis(context: any[], query: string): string {
    // Process context for AI analysis
    const relevantCode = context
      .sort((a, b) => b.relevance - a.relevance)
      .slice(0, 5)
      .map(c => `${c.location}:\n${c.code}`)
      .join('\n\n');
    
    return `Found ${context.length} matches for "${query}":\n\n${relevantCode}`;
  }
}
```

---

## 🧪 Testing Integration

### Unit Test Example

```typescript
import { describe, it, expect, beforeAll } from 'vitest';
import { LensClient } from './lens-client';

describe('Lens Integration', () => {
  let client: LensClient;
  
  beforeAll(async () => {
    client = new LensClient('http://localhost:3000');
    
    // Ensure server is compatible
    const compatible = await client.checkCompatibility();
    expect(compatible).toBe(true);
  });
  
  it('should search and return typed results', async () => {
    const results = await client.search('main', 'function test', {
      mode: 'struct',
      k: 5
    });
    
    expect(results.api_version).toBe('v1');
    expect(results.index_version).toBe('v1');
    expect(results.policy_version).toBe('v1');
    expect(results.hits).toBeInstanceOf(Array);
    expect(results.total).toBeGreaterThanOrEqual(0);
    expect(results.latency_ms.total).toBeGreaterThan(0);
  });
  
  it('should handle Unicode correctly', async () => {
    const results = await client.search('main', '测试', {
      mode: 'lex',
      k: 10
    });
    
    // Should not throw on Unicode characters
    expect(results).toBeDefined();
  });
  
  it('should respect rate limits', async () => {
    // Make multiple concurrent requests
    const promises = Array.from({ length: 20 }, () =>
      client.search('main', 'test', { k: 1 })
    );
    
    // Some might be rate limited, but should eventually succeed
    const results = await Promise.allSettled(promises);
    const successful = results.filter(r => r.status === 'fulfilled');
    expect(successful.length).toBeGreaterThan(0);
  });
});
```

---

## 📋 Integration Checklist

Before deploying your Lens integration:

### Development Phase
- [ ] Version compatibility check implemented
- [ ] Error handling for all HTTP status codes
- [ ] Unicode support tested with emoji/special characters  
- [ ] Retry logic with exponential backoff
- [ ] Trace ID generation for debugging
- [ ] Request parameter validation and constraints
- [ ] Response schema validation

### Testing Phase  
- [ ] Unit tests for all integration functions
- [ ] Error scenario testing (network failures, rate limits)
- [ ] Performance testing under load
- [ ] Unicode/special character edge case testing
- [ ] Version compatibility regression testing

### Production Phase
- [ ] Monitoring and alerting for integration failures
- [ ] Rate limiting respect and proper backoff
- [ ] Circuit breaker pattern for service failures
- [ ] Logging with structured trace IDs
- [ ] Health check integration
- [ ] Graceful degradation when Lens is unavailable

---

## 🚨 Common Pitfalls & Solutions

### 1. Version Compatibility Issues
```typescript
// ❌ Wrong - assuming compatibility
const results = await client.search('main', 'test');

// ✅ Right - check compatibility first
if (!await client.checkCompatibility()) {
  throw new Error('Version mismatch');
}
const results = await client.search('main', 'test');
```

### 2. Unicode Handling
```typescript
// ❌ Wrong - byte-based column calculation
const column = Buffer.from(line.slice(0, charIndex)).length;

// ✅ Right - Unicode code point calculation
const column = Array.from(line.slice(0, charIndex)).length;
```

### 3. Error Handling
```typescript
// ❌ Wrong - generic error handling
try {
  await client.search('main', 'test');
} catch (error) {
  console.error('Search failed');
}

// ✅ Right - specific error handling
try {
  await client.search('main', 'test');
} catch (error) {
  if (error.response?.status === 429) {
    await handleRateLimit();
  } else if (error.response?.status === 404) {
    await triggerIndexing();
  } else {
    throw error;
  }
}
```

### 4. Request Optimization
```typescript
// ❌ Wrong - unbounded requests
await client.search('main', 'test', { k: 10000, context: 10 });

// ✅ Right - bounded and optimized
await client.search('main', 'test', { 
  k: Math.min(needed, 50),
  context: needed > 10 ? 2 : 0,
  mode: 'lex' // For fast results
});
```

---

This comprehensive guide should enable robust integration with Lens v1.0. For additional support, refer to the [Operations Runbook](OPS_RUNBOOK.md) and [Configuration Reference](CONFIG_REFERENCE.md).