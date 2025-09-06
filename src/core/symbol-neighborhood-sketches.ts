/**
 * Symbol-Neighborhood Sketches
 * Precomputes immutable neighborhood sketches per symbol with Bloom filters and MinHash
 * Accelerates Stage-B+ by using sketches before touching heavy SymbolGraph
 */

import { LensTracer } from '../telemetry/tracer.js';
import type { SymbolCandidate } from '../core/span_resolver/index.js';

export interface NeighborTriple {
  neighbor_kind: 'caller' | 'callee' | 'field' | 'type' | 'import' | 'reference';
  freq: number;
  topic_id: string; // For topic-based filtering
  symbol_name: string;
}

export interface SymbolSketch {
  symbolId: string;
  topKNeighbors: NeighborTriple[]; // Limited to K neighbors
  bloomFilter: Uint8Array; // For quick inclusion tests
  minHashSignature: number[]; // For similarity estimation
  lastUpdated: number;
  isImmutable: boolean; // Set to true after initial computation
}

export interface SketchConfig {
  enabled: boolean;
  maxNeighborsK: number; // K≤16 constraint
  bloomFilterBits: number; // ≤256 bits constraint
  minHashSize: number; // Number of hash functions for MinHash
  varintCompression: boolean; // Use varint-G8IU compression
  cacheExpirationMs: number;
  topicVetoEnabled: boolean; // Enable topic-based filtering
}

export interface SketchMetrics {
  sketches_cached: number;
  bloom_hits: number;
  bloom_misses: number;
  stage_b_cpu_reduction_percent: number;
  avg_sketch_size_bytes: number;
  cache_hit_rate: number;
}

export class SymbolNeighborhoodSketcher {
  private config: SketchConfig;
  private sketches: Map<string, SymbolSketch> = new Map();
  private metrics: SketchMetrics = {
    sketches_cached: 0,
    bloom_hits: 0,
    bloom_misses: 0,
    stage_b_cpu_reduction_percent: 0,
    avg_sketch_size_bytes: 0,
    cache_hit_rate: 0,
  };
  
  // Hash functions for MinHash (simple implementation)
  private hashFunctions: Array<(input: string) => number>;

  constructor(config: Partial<SketchConfig> = {}) {
    this.config = {
      enabled: false, // Start disabled for A/B testing
      maxNeighborsK: 16, // Hard constraint from requirements
      bloomFilterBits: 256, // Hard constraint from requirements
      minHashSize: 64, // Good balance of accuracy vs space
      varintCompression: true,
      cacheExpirationMs: 60 * 60 * 1000, // 1 hour
      topicVetoEnabled: true,
      ...config,
    };

    // Initialize hash functions for MinHash
    this.hashFunctions = this.initializeHashFunctions(this.config.minHashSize);
  }

  /**
   * Precompute immutable neighborhood sketch for a symbol
   */
  async computeSymbolSketch(
    symbolId: string,
    neighbors: Map<string, NeighborTriple[]>
  ): Promise<SymbolSketch> {
    const span = LensTracer.createChildSpan('compute_symbol_sketch');

    try {
      if (!this.config.enabled) {
        throw new Error('Symbol neighborhood sketches disabled');
      }

      // Check if sketch already exists and is immutable
      const existing = this.sketches.get(symbolId);
      if (existing && existing.isImmutable) {
        span.setAttributes({ cache_hit: true, immutable: true });
        return existing;
      }

      // Flatten and rank all neighbors by frequency
      const allNeighbors: NeighborTriple[] = [];
      for (const neighborList of neighbors.values()) {
        allNeighbors.push(...neighborList);
      }

      // Sort by frequency (descending) and take top K
      allNeighbors.sort((a, b) => b.freq - a.freq);
      const topKNeighbors = allNeighbors.slice(0, this.config.maxNeighborsK);

      // Build Bloom filter for quick inclusion tests
      const bloomFilter = this.buildBloomFilter(topKNeighbors);

      // Compute MinHash signature for similarity
      const minHashSignature = this.computeMinHash(topKNeighbors);

      const sketch: SymbolSketch = {
        symbolId,
        topKNeighbors,
        bloomFilter,
        minHashSignature,
        lastUpdated: Date.now(),
        isImmutable: true, // Mark as immutable after first computation
      };

      // Store sketch with compression if enabled
      this.cacheSketch(symbolId, sketch);

      span.setAttributes({
        success: true,
        cache_hit: false,
        symbol_id: symbolId,
        neighbors_count: allNeighbors.length,
        top_k_neighbors: topKNeighbors.length,
        bloom_bits: this.config.bloomFilterBits,
        sketch_size_bytes: this.estimateSketchSize(sketch),
      });

      return sketch;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ 
        success: false, 
        error: (error as Error).message,
        symbol_id: symbolId,
      });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Fast neighbor inclusion test using Bloom filter
   */
  async fastInclusionTest(
    symbolId: string,
    candidateNeighbor: string,
    neighborKind: NeighborTriple['neighbor_kind']
  ): Promise<{ maybeIncluded: boolean; certainlyNotIncluded: boolean; cpuSaved: boolean }> {
    const span = LensTracer.createChildSpan('fast_inclusion_test');

    try {
      const sketch = this.sketches.get(symbolId);
      if (!sketch) {
        span.setAttributes({ sketch_miss: true });
        return { maybeIncluded: true, certainlyNotIncluded: false, cpuSaved: false };
      }

      // Test against Bloom filter first
      const bloomResult = this.testBloomFilter(sketch.bloomFilter, candidateNeighbor, neighborKind);
      
      if (!bloomResult) {
        // Bloom filter says "definitely not included"
        this.metrics.bloom_hits++;
        span.setAttributes({ 
          success: true, 
          bloom_result: 'not_included',
          cpu_saved: true,
        });
        
        return { maybeIncluded: false, certainlyNotIncluded: true, cpuSaved: true };
      }

      // Bloom filter says "maybe included" - need to check actual neighbors
      this.metrics.bloom_misses++;
      
      // Quick check in top-K neighbors
      const exactMatch = sketch.topKNeighbors.some(neighbor => 
        neighbor.symbol_name === candidateNeighbor && neighbor.neighbor_kind === neighborKind
      );

      span.setAttributes({
        success: true,
        bloom_result: 'maybe_included',
        exact_match: exactMatch,
        cpu_saved: exactMatch, // Saved full graph traversal if found in sketch
      });

      return { 
        maybeIncluded: true, 
        certainlyNotIncluded: false, 
        cpuSaved: exactMatch 
      };

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: (error as Error).message });
      
      // Fallback: assume might be included
      return { maybeIncluded: true, certainlyNotIncluded: false, cpuSaved: false };
    } finally {
      span.end();
    }
  }

  /**
   * Apply topic/role vetos using sketch information
   */
  async applyTopicVetos(
    symbolId: string,
    candidates: SymbolCandidate[],
    allowedTopics: Set<string>
  ): Promise<SymbolCandidate[]> {
    const span = LensTracer.createChildSpan('apply_topic_vetos');

    try {
      if (!this.config.topicVetoEnabled) {
        return candidates; // Pass through if vetos disabled
      }

      const sketch = this.sketches.get(symbolId);
      if (!sketch) {
        span.setAttributes({ sketch_miss: true });
        return candidates; // Can't apply vetos without sketch
      }

      const filtered: SymbolCandidate[] = [];

      for (const candidate of candidates) {
        // Check if candidate's topic is in allowed set
        const candidateTopics = this.extractCandidateTopics(candidate);
        const hasAllowedTopic = candidateTopics.some(topic => allowedTopics.has(topic));

        // Use sketch to validate topic compatibility
        const topicCompatible = sketch.topKNeighbors.some(neighbor => 
          candidateTopics.includes(neighbor.topic_id)
        );

        if (hasAllowedTopic && topicCompatible) {
          filtered.push(candidate);
        }
      }

      span.setAttributes({
        success: true,
        original_candidates: candidates.length,
        filtered_candidates: filtered.length,
        veto_rate: candidates.length > 0 ? 1 - (filtered.length / candidates.length) : 0,
      });

      return filtered;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: (error as Error).message });
      return candidates; // Fallback: return unfiltered
    } finally {
      span.end();
    }
  }

  /**
   * Estimate CPU reduction from using sketches
   */
  estimateCPUReduction(): number {
    const totalTests = this.metrics.bloom_hits + this.metrics.bloom_misses;
    if (totalTests === 0) return 0;

    // Bloom filter hits avoid expensive graph traversal
    const bloomHitReduction = (this.metrics.bloom_hits / totalTests) * 0.8; // 80% CPU saved per hit
    
    // Cache hits avoid recomputation
    const cacheReduction = this.metrics.cache_hit_rate * 0.3; // 30% CPU saved per cache hit

    return (bloomHitReduction + cacheReduction) * 100; // Convert to percentage
  }

  /**
   * Update configuration for A/B testing
   */
  updateConfig(newConfig: Partial<SketchConfig>): void {
    // Validate constraints
    if (newConfig.maxNeighborsK !== undefined && newConfig.maxNeighborsK > 16) {
      throw new Error('maxNeighborsK must be ≤16 per requirements');
    }
    if (newConfig.bloomFilterBits !== undefined && newConfig.bloomFilterBits > 256) {
      throw new Error('bloomFilterBits must be ≤256 per requirements');
    }

    this.config = { ...this.config, ...newConfig };
    console.log(`🎨 Symbol Neighborhood Sketcher config updated:`, this.config);
  }

  /**
   * Get performance metrics
   */
  getMetrics(): SketchMetrics & { 
    cpu_reduction_estimate: number;
    avg_neighbors_per_sketch: number;
  } {
    const cpuReduction = this.estimateCPUReduction();
    this.metrics.stage_b_cpu_reduction_percent = cpuReduction;

    const avgNeighbors = this.sketches.size > 0
      ? Array.from(this.sketches.values())
          .reduce((sum, sketch) => sum + sketch.topKNeighbors.length, 0) / this.sketches.size
      : 0;

    return {
      ...this.metrics,
      cpu_reduction_estimate: cpuReduction,
      avg_neighbors_per_sketch: avgNeighbors,
    };
  }

  private buildBloomFilter(neighbors: NeighborTriple[]): Uint8Array {
    const filter = new Uint8Array(Math.ceil(this.config.bloomFilterBits / 8));
    
    for (const neighbor of neighbors) {
      const key = `${neighbor.neighbor_kind}:${neighbor.symbol_name}:${neighbor.topic_id}`;
      
      // Use multiple hash functions for better distribution
      const hash1 = this.simpleHash(key) % this.config.bloomFilterBits;
      const hash2 = this.simpleHash(key + '_salt') % this.config.bloomFilterBits;
      const hash3 = this.simpleHash('prefix_' + key) % this.config.bloomFilterBits;
      
      // Set bits in filter
      this.setBit(filter, hash1);
      this.setBit(filter, hash2);
      this.setBit(filter, hash3);
    }
    
    return filter;
  }

  private testBloomFilter(
    filter: Uint8Array, 
    candidateNeighbor: string, 
    neighborKind: NeighborTriple['neighbor_kind']
  ): boolean {
    const key = `${neighborKind}:${candidateNeighbor}:unknown`; // Topic unknown in test
    
    const hash1 = this.simpleHash(key) % this.config.bloomFilterBits;
    const hash2 = this.simpleHash(key + '_salt') % this.config.bloomFilterBits;
    const hash3 = this.simpleHash('prefix_' + key) % this.config.bloomFilterBits;
    
    // All bits must be set for positive result
    return this.getBit(filter, hash1) && 
           this.getBit(filter, hash2) && 
           this.getBit(filter, hash3);
  }

  private computeMinHash(neighbors: NeighborTriple[]): number[] {
    const signature: number[] = new Array(this.config.minHashSize).fill(Infinity);
    
    for (const neighbor of neighbors) {
      const element = `${neighbor.neighbor_kind}:${neighbor.symbol_name}`;
      
      for (let i = 0; i < this.config.minHashSize; i++) {
        const hash = this.hashFunctions[i](element);
        signature[i] = Math.min(signature[i], hash);
      }
    }
    
    return signature;
  }

  private initializeHashFunctions(count: number): Array<(input: string) => number> {
    const functions: Array<(input: string) => number> = [];
    
    for (let i = 0; i < count; i++) {
      const salt = i.toString(36);
      functions.push((input: string) => this.simpleHash(salt + input));
    }
    
    return functions;
  }

  private simpleHash(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash);
  }

  private setBit(buffer: Uint8Array, bitIndex: number): void {
    const byteIndex = Math.floor(bitIndex / 8);
    const bitOffset = bitIndex % 8;
    buffer[byteIndex] |= (1 << bitOffset);
  }

  private getBit(buffer: Uint8Array, bitIndex: number): boolean {
    const byteIndex = Math.floor(bitIndex / 8);
    const bitOffset = bitIndex % 8;
    return (buffer[byteIndex] & (1 << bitOffset)) !== 0;
  }

  private estimateSketchSize(sketch: SymbolSketch): number {
    // Rough estimate of memory usage
    let size = 0;
    
    // Top-K neighbors
    size += sketch.topKNeighbors.length * 64; // ~64 bytes per neighbor estimate
    
    // Bloom filter
    size += sketch.bloomFilter.length;
    
    // MinHash signature
    size += sketch.minHashSignature.length * 4; // 4 bytes per number
    
    // Metadata
    size += 100; // Rough estimate for strings and metadata
    
    return size;
  }

  private extractCandidateTopics(candidate: SymbolCandidate): string[] {
    // Extract topics from candidate metadata
    // This would be implemented based on actual candidate structure
    return ['default', 'core', candidate.symbol_kind || 'unknown'];
  }

  private cacheSketch(symbolId: string, sketch: SymbolSketch): void {
    // Apply LRU eviction if needed
    const maxSize = 10000; // Reasonable cache size limit
    
    if (this.sketches.size >= maxSize) {
      const oldestKey = this.sketches.keys().next().value;
      this.sketches.delete(oldestKey);
    }
    
    this.sketches.set(symbolId, sketch);
    this.metrics.sketches_cached = this.sketches.size;
    
    // Update average sketch size
    const totalSize = Array.from(this.sketches.values())
      .reduce((sum, s) => sum + this.estimateSketchSize(s), 0);
    this.metrics.avg_sketch_size_bytes = totalSize / this.sketches.size;
  }
}

/**
 * Factory for creating symbol neighborhood sketcher
 */
export function createSymbolNeighborhoodSketcher(
  config?: Partial<SketchConfig>
): SymbolNeighborhoodSketcher {
  return new SymbolNeighborhoodSketcher(config);
}