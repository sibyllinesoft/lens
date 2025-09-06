/**
 * Embedding Router for Language-Specific Model Selection
 * 
 * Routes embedding requests to appropriate models based on language.
 * Supports quantization to int8 and performance optimization.
 */

export interface EmbeddingModel {
  id: string;
  name: string;
  dim: number;
  supported_languages: string[];
  quantized: boolean;
  performance_tier: 'fast' | 'balanced' | 'quality';
}

export interface EmbeddingRequest {
  text: string;
  language: string;
  context?: 'query' | 'document' | 'symbol';
}

export interface EmbeddingResponse {
  embedding: Float32Array | Int8Array;
  model_id: string;
  quantized: boolean;
  processing_time_ms: number;
}

export interface QuantizationConfig {
  enabled: boolean;
  method: 'int8' | 'pq_8x8';
  calibration_samples: number;
  isotonic_recalibration: boolean;
}

/**
 * Language-aware embedding router
 */
export class EmbeddingRouter {
  private models: Map<string, EmbeddingModel> = new Map();
  private languageRoutes: Map<string, string> = new Map();
  private quantConfig: QuantizationConfig;
  private calibrationData: Map<string, { scale: number; offset: number }> = new Map();

  constructor(quantConfig?: Partial<QuantizationConfig>) {
    this.quantConfig = {
      enabled: true,
      method: 'int8',
      calibration_samples: 1000,
      isotonic_recalibration: true,
      ...quantConfig
    };

    // Initialize default models and routes
    this.initializeDefaultModels();
    this.initializeLanguageRoutes();
  }

  private initializeDefaultModels(): void {
    // Model A: Fast, for high-level languages
    this.models.set('model_a_fast', {
      id: 'model_a_fast',
      name: 'Code-Embedding-A-Fast',
      dim: 384,
      supported_languages: ['py', 'ts', 'js', 'java', 'php'],
      quantized: this.quantConfig.enabled,
      performance_tier: 'fast'
    });

    // Model B: Balanced, for systems languages  
    this.models.set('model_b_balanced', {
      id: 'model_b_balanced', 
      name: 'Code-Embedding-B-Balanced',
      dim: 512,
      supported_languages: ['go', 'rs', 'cpp', 'c'],
      quantized: this.quantConfig.enabled,
      performance_tier: 'balanced'
    });

    // Model C: Quality, fallback for all languages
    this.models.set('model_c_quality', {
      id: 'model_c_quality',
      name: 'Code-Embedding-C-Quality', 
      dim: 768,
      supported_languages: ['*'],
      quantized: this.quantConfig.enabled,
      performance_tier: 'quality'
    });
  }

  private initializeLanguageRoutes(): void {
    // High-level languages → Model A (fast)
    const modelALangs = ['py', 'python', 'ts', 'typescript', 'js', 'javascript', 'java', 'php'];
    for (const lang of modelALangs) {
      this.languageRoutes.set(lang, 'model_a_fast');
    }

    // Systems languages → Model B (balanced)
    const modelBLangs = ['go', 'golang', 'rs', 'rust', 'cpp', 'c++', 'c'];
    for (const lang of modelBLangs) {
      this.languageRoutes.set(lang, 'model_b_balanced');
    }

    // Everything else → Model C (quality fallback)
    // Handled in getModelForLanguage fallback logic
  }

  /**
   * Route embedding request to appropriate model
   */
  async embed(request: EmbeddingRequest): Promise<EmbeddingResponse> {
    const startTime = Date.now();
    
    const modelId = this.getModelForLanguage(request.language);
    const model = this.models.get(modelId);
    
    if (!model) {
      throw new Error(`No model found for ID: ${modelId}`);
    }

    // Get raw embedding (would call actual embedding service)
    const rawEmbedding = await this.computeEmbedding(request.text, model);
    
    // Apply quantization if enabled
    const embedding = this.quantConfig.enabled 
      ? this.quantizeEmbedding(rawEmbedding, model.id)
      : rawEmbedding;

    const processingTime = Date.now() - startTime;

    return {
      embedding,
      model_id: modelId,
      quantized: this.quantConfig.enabled,
      processing_time_ms: processingTime
    };
  }

  private getModelForLanguage(language: string): string {
    const normalizedLang = language.toLowerCase();
    
    // Direct route lookup
    const modelId = this.languageRoutes.get(normalizedLang);
    if (modelId) {
      return modelId;
    }

    // Fallback to quality model
    return 'model_c_quality';
  }

  private async computeEmbedding(text: string, model: EmbeddingModel): Promise<Float32Array> {
    // Placeholder - would integrate with actual embedding service
    // For now, return mock embedding of correct dimension
    return new Float32Array(model.dim).map(() => Math.random() - 0.5);
  }

  /**
   * Quantize embedding to int8 with calibration
   */
  private quantizeEmbedding(embedding: Float32Array, modelId: string): Int8Array {
    if (this.quantConfig.method === 'int8') {
      return this.quantizeToInt8(embedding, modelId);
    } else if (this.quantConfig.method === 'pq_8x8') {
      return this.quantizeWithPQ(embedding, modelId);
    }
    
    throw new Error(`Unknown quantization method: ${this.quantConfig.method}`);
  }

  private quantizeToInt8(embedding: Float32Array, modelId: string): Int8Array {
    let calibration = this.calibrationData.get(modelId);
    
    if (!calibration) {
      // Initialize calibration parameters
      const absMax = Math.max(...Array.from(embedding).map(Math.abs));
      calibration = {
        scale: 127 / absMax,
        offset: 0
      };
      this.calibrationData.set(modelId, calibration);
    }

    const quantized = new Int8Array(embedding.length);
    for (let i = 0; i < embedding.length; i++) {
      const scaled = embedding[i] * calibration.scale + calibration.offset;
      quantized[i] = Math.max(-127, Math.min(127, Math.round(scaled)));
    }

    return quantized;
  }

  private quantizeWithPQ(embedding: Float32Array, modelId: string): Int8Array {
    // Product Quantization 8x8 - simplified implementation
    // In practice would use proper codebook learning
    const subvectorSize = Math.ceil(embedding.length / 8);
    const quantized = new Int8Array(embedding.length);
    
    for (let i = 0; i < 8; i++) {
      const start = i * subvectorSize;
      const end = Math.min(start + subvectorSize, embedding.length);
      
      // Simple uniform quantization per subvector
      const subvector = embedding.slice(start, end);
      const subMax = Math.max(...Array.from(subvector).map(Math.abs));
      const scale = 127 / subMax;
      
      for (let j = start; j < end; j++) {
        quantized[j] = Math.max(-127, Math.min(127, Math.round(embedding[j] * scale)));
      }
    }
    
    return quantized;
  }

  /**
   * Calibrate quantization parameters using sample data
   */
  async calibrateQuantization(samples: Array<{ text: string; language: string }>): Promise<void> {
    if (!this.quantConfig.enabled) {
      return;
    }

    console.log(`Calibrating quantization with ${samples.length} samples...`);
    
    const modelSamples = new Map<string, Float32Array[]>();
    
    // Collect embeddings per model
    for (const sample of samples.slice(0, this.quantConfig.calibration_samples)) {
      const modelId = this.getModelForLanguage(sample.language);
      const model = this.models.get(modelId)!;
      
      const embedding = await this.computeEmbedding(sample.text, model);
      
      if (!modelSamples.has(modelId)) {
        modelSamples.set(modelId, []);
      }
      modelSamples.get(modelId)!.push(embedding);
    }

    // Compute calibration parameters per model
    for (const [modelId, embeddings] of modelSamples) {
      if (embeddings.length === 0) continue;
      
      // Compute statistics across all sample embeddings
      const allValues = embeddings.flatMap(emb => Array.from(emb));
      const sortedValues = allValues.sort((a, b) => a - b);
      
      // Use 99.9th percentile for robust scaling
      const p999Index = Math.floor(sortedValues.length * 0.999);
      const absMax = Math.max(Math.abs(sortedValues[0]), Math.abs(sortedValues[p999Index]));
      
      this.calibrationData.set(modelId, {
        scale: 127 / absMax,
        offset: 0
      });
    }

    console.log('Quantization calibration complete');
  }

  /**
   * Get routing statistics
   */
  getRoutingStats(): { modelId: string; languages: string[] }[] {
    const modelUsage = new Map<string, Set<string>>();
    
    for (const [lang, modelId] of this.languageRoutes) {
      if (!modelUsage.has(modelId)) {
        modelUsage.set(modelId, new Set());
      }
      modelUsage.get(modelId)!.add(lang);
    }

    return Array.from(modelUsage.entries()).map(([modelId, languages]) => ({
      modelId,
      languages: Array.from(languages)
    }));
  }

  /**
   * Update quantization configuration
   */
  updateQuantizationConfig(config: Partial<QuantizationConfig>): void {
    this.quantConfig = { ...this.quantConfig, ...config };
    
    // Update model quantization flags
    for (const model of this.models.values()) {
      model.quantized = this.quantConfig.enabled;
    }
    
    // Clear calibration data if method changed
    if (config.method) {
      this.calibrationData.clear();
    }
  }

  /**
   * Get model information
   */
  getModelInfo(modelId: string): EmbeddingModel | undefined {
    return this.models.get(modelId);
  }

  /**
   * List all available models
   */
  listModels(): EmbeddingModel[] {
    return Array.from(this.models.values());
  }
}