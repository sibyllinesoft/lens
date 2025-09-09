/**
 * Lens API client with retry-aware fetching and SLA deadline management
 * Implements Section 1 of TODO.md: idempotent, retry-aware fetcher
 */

import { LensEndpointConfig } from '../config/data-source-config.js';

export interface LensSearchRequest {
  query: string;
  limit?: number;
  context_lines?: number;
  language?: string;
}

export interface LensHit {
  file_path: string;
  line_start: number;
  line_end: number;
  score: number;
  why: 'lex' | 'struct' | 'sem';
  content: string;
  span_start?: number;
  span_end?: number;
}

export interface LensSearchResponse {
  query_id: string;
  hits: LensHit[];
  shard_id: string;
  latency_ms: number;
  total_hits: number;
  truncated: boolean;
}

export interface LensClientMetrics {
  req_ts: number;
  lat_ms: number;
  within_sla: boolean;
  endpoint_url: string;
  shard: string;
  success: boolean;
  error_code?: string;
  retry_count: number;
}

export class LensClient {
  private readonly config: LensEndpointConfig;
  private readonly abortController = new AbortController();

  constructor(config: LensEndpointConfig) {
    this.config = config;
  }

  async search(request: LensSearchRequest): Promise<{
    response?: LensSearchResponse;
    metrics: LensClientMetrics;
  }> {
    const req_ts = Date.now();
    let lastError: Error | null = null;
    let retry_count = 0;

    for (let attempt = 0; attempt <= this.config.retries; attempt++) {
      retry_count = attempt;
      
      try {
        const response = await this.performSearch(request, req_ts, attempt);
        const lat_ms = Date.now() - req_ts;
        
        return {
          response,
          metrics: {
            req_ts,
            lat_ms,
            within_sla: lat_ms <= this.config.timeout,
            endpoint_url: this.config.baseUrl,
            shard: response.shard_id,
            success: true,
            retry_count
          }
        };
      } catch (error) {
        lastError = error as Error;
        
        // Don't retry on client errors (4xx) or auth errors
        if (this.isNonRetryableError(error)) {
          break;
        }
        
        // Exponential backoff with jitter
        if (attempt < this.config.retries) {
          const delay = Math.min(1000, (2 ** attempt) * 100) + Math.random() * 100;
          await this.sleep(delay);
        }
      }
    }

    // All retries failed
    const lat_ms = Date.now() - req_ts;
    return {
      metrics: {
        req_ts,
        lat_ms,
        within_sla: lat_ms <= this.config.timeout,
        endpoint_url: this.config.baseUrl,
        shard: 'unknown',
        success: false,
        error_code: this.getErrorCode(lastError),
        retry_count
      }
    };
  }

  private async performSearch(
    request: LensSearchRequest,
    startTime: number,
    attempt: number
  ): Promise<LensSearchResponse> {
    const remainingTime = this.config.timeout - (Date.now() - startTime);
    if (remainingTime <= 0) {
      throw new Error('SLA_TIMEOUT_EXCEEDED');
    }

    const searchUrl = `${this.config.baseUrl}/search`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), remainingTime);

    try {
      const response = await fetch(searchUrl, {
        method: 'POST',
        headers: this.config.headers,
        body: JSON.stringify(request),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP_${response.status}_${response.statusText}`);
      }

      const contentType = response.headers.get('content-type');
      if (contentType?.includes('application/x-ndjson')) {
        return await this.parseNDJSONResponse(response);
      } else if (contentType?.includes('application/json')) {
        return await response.json();
      } else {
        throw new Error('INVALID_RESPONSE_CONTENT_TYPE');
      }
    } catch (error) {
      clearTimeout(timeoutId);
      
      if ((error as Error).name === 'AbortError') {
        throw new Error('REQUEST_TIMEOUT');
      }
      
      throw error;
    }
  }

  private async parseNDJSONResponse(response: Response): Promise<LensSearchResponse> {
    const text = await response.text();
    const lines = text.trim().split('\n');
    
    if (lines.length === 0) {
      throw new Error('EMPTY_NDJSON_RESPONSE');
    }

    try {
      // Parse the first line which should contain metadata
      const metadata = JSON.parse(lines[0]);
      
      // Parse remaining lines as hits
      const hits: LensHit[] = lines.slice(1).map(line => {
        if (!line.trim()) return null;
        return JSON.parse(line);
      }).filter(hit => hit !== null);

      return {
        query_id: metadata.query_id || this.generateQueryId(),
        hits,
        shard_id: metadata.shard_id || 'unknown',
        latency_ms: metadata.latency_ms || 0,
        total_hits: hits.length,
        truncated: metadata.truncated || false
      };
    } catch (parseError) {
      throw new Error(`NDJSON_PARSE_ERROR: ${parseError}`);
    }
  }

  private generateQueryId(): string {
    return `q_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private isNonRetryableError(error: any): boolean {
    const errorMessage = error?.message || '';
    
    // Don't retry client errors
    if (errorMessage.match(/HTTP_4\d\d/)) {
      return true;
    }
    
    // Don't retry auth errors
    if (errorMessage.includes('401') || errorMessage.includes('403')) {
      return true;
    }
    
    // Don't retry malformed requests
    if (errorMessage.includes('INVALID_REQUEST') || errorMessage.includes('BAD_REQUEST')) {
      return true;
    }

    return false;
  }

  private getErrorCode(error: Error | null): string {
    if (!error) return 'UNKNOWN_ERROR';
    
    const message = error.message;
    
    if (message.includes('timeout') || message.includes('TIMEOUT')) {
      return 'TIMEOUT';
    }
    if (message.includes('network') || message.includes('fetch')) {
      return 'NETWORK_ERROR';
    }
    if (message.startsWith('HTTP_')) {
      return message.split('_')[1]; // Extract HTTP status code
    }
    
    return 'UNKNOWN_ERROR';
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // Cleanup method for graceful shutdown
  cancel(): void {
    this.abortController.abort();
  }
}