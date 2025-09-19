/**
 * Lens API client with retry-aware fetching and SLA deadline management
 * Implements Section 1 of TODO.md: idempotent, retry-aware fetcher
 */

import { LensEndpointConfig } from '../config/data-source-config.js';

export interface LensSearchRequest {
  query: string;
  limit?: number;
  offset?: number;
  language?: string;
  file_pattern?: string;
  fuzzy?: boolean;
  symbols?: boolean;
}

export interface LensSearchHit {
  file_path: string;
  line_number: number;
  content: string;
  score: number;
  language?: string;
  matched_terms: string[];
  context_lines?: string[];
}

export interface LensIndexStats {
  total_documents: number;
  index_size_bytes: number;
  index_size_human: string;
  supported_languages: number;
  average_document_size: number;
  last_updated?: string;
}

export interface LensSearchResponse {
  query: string;
  query_type: string;
  total: number;
  limit: number;
  offset: number;
  duration_ms: number;
  from_cache: boolean;
  results: LensSearchHit[];
  index_stats: LensIndexStats;
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
  from_cache: boolean;
}

export class LensClient {
  private readonly config: LensEndpointConfig;
  private readonly abortController = new AbortController();
  private lastRequestUrl: string | null = null;

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
            endpoint_url: this.lastRequestUrl ?? this.config.baseUrl,
            shard: 'default',
            success: true,
            retry_count,
            from_cache: response.from_cache
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
        endpoint_url: this.lastRequestUrl ?? this.config.baseUrl,
        shard: 'unknown',
        success: false,
        error_code: this.getErrorCode(lastError),
        retry_count,
        from_cache: false
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

    const params = new URLSearchParams();
    params.set('q', request.query);

    if (typeof request.limit === 'number') {
      params.set('limit', request.limit.toString());
    }

    if (typeof request.offset === 'number') {
      params.set('offset', request.offset.toString());
    }

    if (request.language) {
      params.set('language', request.language);
    }

    if (request.file_pattern) {
      params.set('file_pattern', request.file_pattern);
    }

    if (request.fuzzy) {
      params.set('fuzzy', 'true');
    }

    if (request.symbols) {
      params.set('symbols', 'true');
    }

    const searchUrl = `${this.config.baseUrl}/search?${params.toString()}`;
    this.lastRequestUrl = searchUrl;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), remainingTime);

    try {
      const response = await fetch(searchUrl, {
        method: 'GET',
        headers: this.config.headers,
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP_${response.status}_${response.statusText}`);
      }

      const contentType = response.headers.get('content-type');
      if (!contentType?.includes('application/json')) {
        throw new Error('INVALID_RESPONSE_CONTENT_TYPE');
      }

      const payload = await response.json();
      return payload as LensSearchResponse;
    } catch (error) {
      clearTimeout(timeoutId);
      
      if ((error as Error).name === 'AbortError') {
        throw new Error('REQUEST_TIMEOUT');
      }
      
      throw error;
    }
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
