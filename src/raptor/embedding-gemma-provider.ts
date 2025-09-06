/**
 * EmbeddingGemma Provider for Local-First Deployment
 * 
 * Implements Google's EmbeddingGemma-300M with Matryoshka representation learning
 * for local-first code search with configurable dimensions (768/512/256/128).
 * Uses Hugging Face TEI server with OpenAI-compatible /v1/embeddings endpoint.
 */

import { EmbeddingProvider, EmbeddingConfig } from './embeddings.js';

export interface MatryoshkaConfig {
  enabled: boolean;
  targetDimension: 768 | 512 | 256 | 128;
  preserveRanking: boolean;
}

export interface GemmaEmbeddingConfig extends EmbeddingConfig {
  teiEndpoint: string;
  matryoshka: MatryoshkaConfig;
  maxRetries: number;
  timeout: number;
  batchSize: number;
  quantization?: {
    enabled: boolean;
    method: 'int8' | 'fp16';
  };
}

export interface TEIEmbeddingRequest {
  input: string | string[];
  model?: string;
  encoding_format?: 'float' | 'base64';
  dimensions?: number;
}

export interface TEIEmbeddingResponse {
  object: 'list';
  data: Array<{
    object: 'embedding';
    index: number;
    embedding: number[];
  }>;
  model: string;
  usage: {
    prompt_tokens: number;
    total_tokens: number;
  };
}

/**
 * EmbeddingGemma provider with Matryoshka dimension support
 * Compatible with Hugging Face TEI 1.8.1 server
 */
export class EmbeddingGemmaProvider implements EmbeddingProvider {
  private config: GemmaEmbeddingConfig;
  private healthCheckCache: { healthy: boolean; lastCheck: number } = {
    healthy: false,
    lastCheck: 0
  };
  private readonly HEALTH_CHECK_INTERVAL = 30000; // 30 seconds

  constructor(config: Partial<GemmaEmbeddingConfig> = {}) {
    this.config = {
      provider: 'embeddinggemma',
      model: 'google/embeddinggemma-300m',
      dimension: config.matryoshka?.targetDimension || 768,
      batchSize: 32,
      maxTokens: 2048,
      teiEndpoint: 'http://localhost:8080',
      matryoshka: {
        enabled: true,
        targetDimension: 768,
        preserveRanking: true,
        ...config.matryoshka
      },
      maxRetries: 3,
      timeout: 10000,
      ...config
    };
  }

  async embed(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) {
      return [];
    }

    // Health check before processing
    await this.ensureHealthy();

    // Process in batches to respect server limits
    const results: number[][] = [];
    const batches = this.batchTexts(texts, this.config.batchSize);

    for (const batch of batches) {
      const batchResults = await this.embedBatch(batch);
      results.push(...batchResults);
    }

    return results;
  }

  private async embedBatch(texts: string[]): Promise<number[][]> {
    const request: TEIEmbeddingRequest = {
      input: texts,
      model: this.config.model,
      encoding_format: 'float',
      ...(this.config.matryoshka.enabled && {
        dimensions: this.config.matryoshka.targetDimension
      })
    };

    let lastError: Error | null = null;
    
    for (let attempt = 0; attempt < this.config.maxRetries; attempt++) {
      try {
        const response = await this.makeRequest(request);
        
        if (!response.ok) {
          throw new Error(`TEI API error: ${response.status} ${response.statusText}`);
        }

        const data: TEIEmbeddingResponse = await response.json();
        
        // Extract embeddings from response
        const embeddings = data.data
          .sort((a, b) => a.index - b.index) // Ensure correct order
          .map(item => item.embedding);

        // Apply Matryoshka truncation if needed
        if (this.config.matryoshka.enabled) {
          return embeddings.map(emb => this.truncateMatryoshka(emb));
        }

        return embeddings;

      } catch (error) {
        lastError = error as Error;
        console.warn(`EmbeddingGemma attempt ${attempt + 1} failed:`, error);
        
        if (attempt < this.config.maxRetries - 1) {
          // Exponential backoff
          await this.delay(Math.pow(2, attempt) * 1000);
        }
      }
    }

    throw new Error(`EmbeddingGemma failed after ${this.config.maxRetries} attempts: ${lastError?.message}`);
  }

  private async makeRequest(request: TEIEmbeddingRequest): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);

    try {
      const response = await fetch(`${this.config.teiEndpoint}/v1/embeddings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify(request),
        signal: controller.signal,
      });

      return response;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Truncate embedding using Matryoshka representation
   * Maintains ranking properties by keeping the first N dimensions
   */
  private truncateMatryoshka(embedding: number[]): number[] {
    const targetDim = this.config.matryoshka.targetDimension;
    
    if (embedding.length <= targetDim) {
      return embedding;
    }

    // Matryoshka truncation: keep first N dimensions
    const truncated = embedding.slice(0, targetDim);
    
    if (this.config.matryoshka.preserveRanking) {
      // Renormalize to preserve similarity rankings
      const norm = Math.sqrt(truncated.reduce((sum, x) => sum + x * x, 0));
      if (norm > 0) {
        return truncated.map(x => x / norm);
      }
    }

    return truncated;
  }

  private batchTexts(texts: string[], batchSize: number): string[][] {
    const batches: string[][] = [];
    for (let i = 0; i < texts.length; i += batchSize) {
      batches.push(texts.slice(i, i + batchSize));
    }
    return batches;
  }

  getDimension(): number {
    return this.config.matryoshka.enabled
      ? this.config.matryoshka.targetDimension
      : this.config.dimension;
  }

  getModel(): string {
    return this.config.model;
  }

  getConfig(): GemmaEmbeddingConfig {
    return { ...this.config };
  }

  /**
   * Update Matryoshka configuration for dimension switching
   */
  async updateMatryoshkaConfig(config: Partial<MatryoshkaConfig>): Promise<void> {
    this.config.matryoshka = {
      ...this.config.matryoshka,
      ...config
    };

    // Update dimension if changed
    if (config.targetDimension) {
      this.config.dimension = config.targetDimension;
    }

    console.log(`✅ Updated Matryoshka config: ${this.config.matryoshka.targetDimension}d`);
  }

  /**
   * Health check for TEI server availability
   */
  async healthCheck(): Promise<boolean> {
    const now = Date.now();
    
    // Use cached result if recent
    if (now - this.healthCheckCache.lastCheck < this.HEALTH_CHECK_INTERVAL) {
      return this.healthCheckCache.healthy;
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);

      const response = await fetch(`${this.config.teiEndpoint}/health`, {
        method: 'GET',
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      this.healthCheckCache = {
        healthy: response.ok,
        lastCheck: now
      };

      return response.ok;
    } catch (error) {
      this.healthCheckCache = {
        healthy: false,
        lastCheck: now
      };
      return false;
    }
  }

  private async ensureHealthy(): Promise<void> {
    const isHealthy = await this.healthCheck();
    if (!isHealthy) {
      throw new Error(`TEI server not available at ${this.config.teiEndpoint}`);
    }
  }

  /**
   * Get server information and model details
   */
  async getServerInfo(): Promise<{
    model: string;
    maxInputLength: number;
    maxBatchSize: number;
    dimensions: number[];
  }> {
    const response = await fetch(`${this.config.teiEndpoint}/info`, {
      method: 'GET',
    });

    if (!response.ok) {
      throw new Error(`Failed to get server info: ${response.statusText}`);
    }

    const info = await response.json();
    return {
      model: info.model_id || this.config.model,
      maxInputLength: info.max_input_length || 2048,
      maxBatchSize: info.max_batch_size || 32,
      dimensions: info.dimensions || [128, 256, 512, 768],
    };
  }

  /**
   * Benchmark embedding generation performance
   */
  async benchmark(texts: string[], iterations: number = 3): Promise<{
    avgLatencyMs: number;
    throughputTokensPerSec: number;
    dimension: number;
    errorRate: number;
  }> {
    const results: number[] = [];
    let errors = 0;
    let totalTokens = 0;

    for (let i = 0; i < iterations; i++) {
      try {
        const start = Date.now();
        await this.embed(texts);
        const duration = Date.now() - start;
        results.push(duration);
        
        // Estimate token count (rough approximation)
        totalTokens += texts.reduce((sum, text) => sum + Math.ceil(text.length / 4), 0);
      } catch (error) {
        errors++;
        console.warn(`Benchmark iteration ${i + 1} failed:`, error);
      }
    }

    const avgLatency = results.reduce((sum, val) => sum + val, 0) / results.length;
    const throughput = results.length > 0 ? (totalTokens / (avgLatency / 1000)) / results.length : 0;

    return {
      avgLatencyMs: avgLatency,
      throughputTokensPerSec: throughput,
      dimension: this.getDimension(),
      errorRate: errors / iterations
    };
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    // Reset health check cache
    this.healthCheckCache = { healthy: false, lastCheck: 0 };
    console.log('EmbeddingGemma provider cleaned up');
  }
}

/**
 * Factory function for creating EmbeddingGemma providers with different configurations
 */
export class EmbeddingGemmaFactory {
  static createProvider(
    dimension: 768 | 512 | 256 | 128,
    teiEndpoint: string = 'http://localhost:8080'
  ): EmbeddingGemmaProvider {
    return new EmbeddingGemmaProvider({
      teiEndpoint,
      matryoshka: {
        enabled: true,
        targetDimension: dimension,
        preserveRanking: true,
      },
      batchSize: dimension <= 256 ? 64 : 32, // Higher batch size for smaller dims
      timeout: 15000,
      maxRetries: 3,
    });
  }

  static createShadowProviders(teiEndpoint: string = 'http://localhost:8080'): {
    gemma768: EmbeddingGemmaProvider;
    gemma256: EmbeddingGemmaProvider;
  } {
    return {
      gemma768: this.createProvider(768, teiEndpoint),
      gemma256: this.createProvider(256, teiEndpoint),
    };
  }
}