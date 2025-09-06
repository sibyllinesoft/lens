/**
 * Optimized HNSW Index - Phase B3 Enhancement
 * ANN tuning: fix K=150, sweep efSearch to smallest that preserves ΔnDCG within 0.5%
 * Target: Optimize Stage-C performance while maintaining quality
 */

import type { HNSWIndex, HNSWLayer, HNSWNode } from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';

export interface HNSWOptimizationConfig {
  K: number;                    // Fixed at 150 per TODO requirements
  efSearch: number;             // Tunable parameter for search performance
  efConstruction: number;       // Construction time parameter
  maxLevels: number;           // Maximum number of layers
  levelMultiplier: number;     // Level generation probability factor
  qualityThreshold: number;    // Maximum allowed nDCG degradation (0.5%)
  performanceTarget: number;   // Target latency improvement factor
}

export interface HNSWSearchResult {
  doc_id: string;
  score: number;
  distance: number;
}

export interface HNSWPerformanceMetrics {
  search_latency_ms: number;
  candidates_evaluated: number;
  distance_computations: number;
  quality_score: number;       // nDCG or similar
  throughput_qps: number;
}

/**
 * Optimized HNSW implementation with tunable efSearch parameter
 * Focus on maintaining quality while maximizing search performance
 */
export class OptimizedHNSWIndex {
  private index: HNSWIndex | null = null;
  private docIdToNodeId: Map<string, number> = new Map();
  private nodeIdToDocId: Map<number, string> = new Map();
  private config: HNSWOptimizationConfig;
  private performanceHistory: HNSWPerformanceMetrics[] = [];

  constructor(config: Partial<HNSWOptimizationConfig> = {}) {
    this.config = {
      K: 150, // Fixed per TODO requirements
      efSearch: 64, // Start with reasonable default, will tune
      efConstruction: 128, // Higher for better index quality
      maxLevels: 16,
      levelMultiplier: 1.2,
      qualityThreshold: 0.005, // 0.5% as specified
      performanceTarget: 0.4, // 40% improvement target
      ...config
    };

    // Ensure K is fixed at 150 as required
    if (this.config.K !== 150) {
      console.warn(`🔧 K parameter forced to 150 (was ${this.config.K}) per B3 requirements`);
      this.config.K = 150;
    }

    console.log(`🚀 OptimizedHNSWIndex initialized: K=${this.config.K}, efSearch=${this.config.efSearch}`);
  }

  /**
   * Build optimized HNSW index from document vectors
   */
  async buildIndex(
    vectors: Map<string, Float32Array>,
    progressCallback?: (progress: number) => void
  ): Promise<void> {
    const span = LensTracer.createChildSpan('hnsw_build_optimized', {
      'vectors.count': vectors.size,
      'config.K': this.config.K,
      'config.efConstruction': this.config.efConstruction
    });

    const startTime = Date.now();

    try {
      if (vectors.size === 0) {
        span.setAttributes({ success: true, skipped: true, reason: 'no_vectors' });
        return;
      }

      // Initialize index structure
      this.index = {
        layers: [],
        entry_point: 0,
        max_connections: this.config.K,
        level_multiplier: this.config.levelMultiplier
      };

      this.docIdToNodeId.clear();
      this.nodeIdToDocId.clear();

      const vectorArray = Array.from(vectors.entries());
      const totalVectors = vectorArray.length;

      // Build layers bottom-up for optimal connectivity
      await this.buildLayers(vectorArray, progressCallback);

      const buildTime = Date.now() - startTime;

      span.setAttributes({
        success: true,
        build_time_ms: buildTime,
        nodes_total: this.docIdToNodeId.size,
        layers_built: this.index.layers.length,
        avg_connections: this.calculateAverageConnections()
      });

      console.log(`🚀 HNSW index built: ${totalVectors} vectors in ${buildTime}ms, ${this.index.layers.length} layers`);

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Build HNSW layers with optimized connectivity
   */
  private async buildLayers(
    vectors: Array<[string, Float32Array]>,
    progressCallback?: (progress: number) => void
  ): Promise<void> {
    if (!this.index) throw new Error('Index not initialized');

    // Initialize layer 0 (base layer contains all nodes)
    const layer0: HNSWLayer = {
      level: 0,
      nodes: new Map()
    };
    this.index.layers.push(layer0);

    // Add all nodes to layer 0
    for (let i = 0; i < vectors.length; i++) {
      const [docId, vector] = vectors[i]!;
      const nodeId = i;

      this.docIdToNodeId.set(docId, nodeId);
      this.nodeIdToDocId.set(nodeId, docId);

      const node: HNSWNode = {
        id: nodeId,
        vector,
        connections: new Set()
      };

      layer0.nodes.set(nodeId, node);

      if (progressCallback) {
        progressCallback((i + 1) / vectors.length * 0.5); // 50% progress for node creation
      }
    }

    // Build connections with optimized efConstruction
    await this.buildConnections(vectors, progressCallback);

    // Build higher layers for hierarchical search
    this.buildHigherLayers();
  }

  /**
   * Build optimized connections between nodes
   */
  private async buildConnections(
    vectors: Array<[string, Float32Array]>,
    progressCallback?: (progress: number) => void
  ): Promise<void> {
    if (!this.index || !this.index.layers[0]) return;

    const layer0 = this.index.layers[0];
    const nodes = Array.from(layer0.nodes.values());

    for (let i = 0; i < nodes.length; i++) {
      const currentNode = nodes[i]!;
      
      // Find nearest neighbors using optimized search
      const candidates = this.findNearestNeighborsForConstruction(
        currentNode.vector,
        nodes.slice(0, i), // Only consider already processed nodes
        Math.min(this.config.efConstruction, i)
      );

      // Connect to K best candidates
      const connectionsToMake = Math.min(this.config.K, candidates.length);
      for (let j = 0; j < connectionsToMake; j++) {
        const candidateId = candidates[j]!.nodeId;
        const candidateNode = layer0.nodes.get(candidateId);
        
        if (candidateNode) {
          currentNode.connections.add(candidateId);
          candidateNode.connections.add(currentNode.id);

          // Prune connections if candidate has too many
          if (candidateNode.connections.size > this.config.K) {
            this.pruneConnections(candidateNode, layer0.nodes);
          }
        }
      }

      if (progressCallback) {
        const progress = 0.5 + ((i + 1) / nodes.length * 0.4); // 40% progress for connections
        progressCallback(progress);
      }
    }
  }

  /**
   * Build higher layers for hierarchical search
   */
  private buildHigherLayers(): void {
    if (!this.index || !this.index.layers[0]) return;

    const layer0 = this.index.layers[0];
    const allNodes = Array.from(layer0.nodes.values());
    
    let currentLevelNodes = allNodes;
    let level = 1;

    while (currentLevelNodes.length > 1 && level < this.config.maxLevels) {
      // Select subset of nodes for next level (roughly 1/2 each level)
      const nextLevelNodes = this.selectNodesForLevel(currentLevelNodes, level);
      
      if (nextLevelNodes.length === 0) break;

      const layer: HNSWLayer = {
        level,
        nodes: new Map()
      };

      // Add selected nodes to higher layer with reduced connectivity
      const connectionsPerLevel = Math.max(4, Math.floor(this.config.K / (level * 2)));
      
      for (const node of nextLevelNodes) {
        const higherLevelNode: HNSWNode = {
          id: node.id,
          vector: node.vector,
          connections: new Set()
        };

        // Connect to nearest neighbors at this level
        const neighbors = this.findNearestNeighborsForConstruction(
          node.vector,
          nextLevelNodes.filter(n => n.id !== node.id),
          Math.min(connectionsPerLevel, nextLevelNodes.length - 1)
        );

        for (const neighbor of neighbors) {
          higherLevelNode.connections.add(neighbor.nodeId);
        }

        layer.nodes.set(node.id, higherLevelNode);
      }

      this.index.layers.push(layer);
      currentLevelNodes = nextLevelNodes;
      level++;
    }

    // Set entry point to a well-connected node in the highest layer
    if (this.index.layers.length > 1) {
      const topLayer = this.index.layers[this.index.layers.length - 1]!;
      const entryNode = this.findBestEntryPoint(topLayer);
      this.index.entry_point = entryNode.id;
    }
  }

  /**
   * Search for nearest neighbors with tunable efSearch parameter
   */
  async search(
    queryVector: Float32Array,
    k: number,
    efSearch?: number
  ): Promise<HNSWSearchResult[]> {
    const span = LensTracer.createChildSpan('hnsw_search_optimized', {
      'query.dimension': queryVector.length,
      'search.k': k,
      'search.efSearch': efSearch || this.config.efSearch
    });

    const startTime = Date.now();
    let distanceComputations = 0;

    try {
      if (!this.index || this.index.layers.length === 0) {
        span.setAttributes({ success: true, results: 0, reason: 'empty_index' });
        return [];
      }

      const searchEf = efSearch || this.config.efSearch;
      const candidates = new Map<number, number>(); // nodeId -> distance

      // Start from entry point at highest layer
      let currentLayer = this.index.layers.length - 1;
      let entryPoints = new Set([this.index.entry_point]);

      // Search through higher layers (greedy)
      while (currentLayer > 0) {
        const layer = this.index.layers[currentLayer];
        if (!layer) break;

        entryPoints = this.searchLayer(
          queryVector,
          entryPoints,
          1, // ef = 1 for higher layers (greedy)
          layer,
          candidates
        );

        distanceComputations += entryPoints.size;
        currentLayer--;
      }

      // Search layer 0 with full efSearch
      const layer0 = this.index.layers[0]!;
      const finalCandidates = this.searchLayer(
        queryVector,
        entryPoints,
        Math.max(searchEf, k),
        layer0,
        candidates
      );

      distanceComputations += finalCandidates.size;

      // Convert to results and sort by distance
      const results: HNSWSearchResult[] = [];
      for (const nodeId of finalCandidates) {
        const docId = this.nodeIdToDocId.get(nodeId);
        const distance = candidates.get(nodeId);
        
        if (docId && distance !== undefined) {
          results.push({
            doc_id: docId,
            score: 1.0 - distance, // Convert distance to similarity score
            distance
          });
        }
      }

      results.sort((a, b) => a.distance - b.distance);
      const topK = results.slice(0, k);

      const searchTime = Date.now() - startTime;

      span.setAttributes({
        success: true,
        search_time_ms: searchTime,
        distance_computations: distanceComputations,
        results_found: topK.length,
        efSearch_used: searchEf
      });

      // Record performance metrics
      this.recordPerformanceMetrics({
        search_latency_ms: searchTime,
        candidates_evaluated: finalCandidates.size,
        distance_computations: distanceComputations,
        quality_score: this.estimateQualityScore(topK),
        throughput_qps: 1000 / searchTime
      });

      return topK;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Search within a specific layer
   */
  private searchLayer(
    queryVector: Float32Array,
    entryPoints: Set<number>,
    ef: number,
    layer: HNSWLayer,
    candidates: Map<number, number>
  ): Set<number> {
    const visited = new Set<number>();
    const dynamicCandidates = new Map<number, number>(); // nodeId -> distance

    // Initialize with entry points
    for (const nodeId of entryPoints) {
      const node = layer.nodes.get(nodeId);
      if (node && !visited.has(nodeId)) {
        const distance = this.calculateDistance(queryVector, node.vector);
        dynamicCandidates.set(nodeId, distance);
        candidates.set(nodeId, distance);
        visited.add(nodeId);
      }
    }

    // Expand search
    while (dynamicCandidates.size > 0) {
      // Find closest unvisited candidate
      let closestNodeId = -1;
      let closestDistance = Infinity;
      
      for (const [nodeId, distance] of dynamicCandidates) {
        if (distance < closestDistance) {
          closestDistance = distance;
          closestNodeId = nodeId;
        }
      }

      if (closestNodeId === -1) break;

      dynamicCandidates.delete(closestNodeId);

      // Check if we should continue (ef condition)
      if (candidates.size >= ef) {
        const maxDistance = Math.max(...Array.from(candidates.values()));
        if (closestDistance > maxDistance) {
          break; // Stop expanding
        }
      }

      // Explore neighbors
      const currentNode = layer.nodes.get(closestNodeId);
      if (currentNode) {
        for (const neighborId of currentNode.connections) {
          if (!visited.has(neighborId)) {
            const neighborNode = layer.nodes.get(neighborId);
            if (neighborNode) {
              const distance = this.calculateDistance(queryVector, neighborNode.vector);
              
              dynamicCandidates.set(neighborId, distance);
              candidates.set(neighborId, distance);
              visited.add(neighborId);

              // Prune candidates if too many
              if (candidates.size > ef * 2) {
                this.pruneSearchCandidates(candidates, ef);
              }
            }
          }
        }
      }
    }

    return new Set(
      Array.from(candidates.entries())
        .sort((a, b) => a[1] - b[1])
        .slice(0, ef)
        .map(([nodeId]) => nodeId)
    );
  }

  /**
   * Tune efSearch parameter for optimal performance/quality trade-off
   */
  async tuneEfSearch(
    testQueries: Float32Array[],
    groundTruthResults: HNSWSearchResult[][],
    k: number = 10
  ): Promise<number> {
    const span = LensTracer.createChildSpan('hnsw_tune_efsearch', {
      'test_queries.count': testQueries.length,
      'target.k': k
    });

    try {
      const candidateEfValues = [16, 32, 48, 64, 80, 96, 128, 160, 192, 256];
      let bestEf = this.config.efSearch;
      let bestScore = 0;

      console.log('🔧 Tuning efSearch parameter for optimal performance...');

      for (const efSearch of candidateEfValues) {
        const metrics = await this.evaluateEfSearchPerformance(
          testQueries,
          groundTruthResults,
          k,
          efSearch
        );

        // Score combines quality and performance
        const qualityScore = metrics.quality_score;
        const latencyPenalty = Math.max(0, metrics.search_latency_ms - 8) * 0.01; // Penalty beyond 8ms
        const combinedScore = qualityScore - latencyPenalty;

        console.log(`  efSearch=${efSearch}: quality=${qualityScore.toFixed(3)}, latency=${metrics.search_latency_ms.toFixed(1)}ms, score=${combinedScore.toFixed(3)}`);

        // Check if quality degradation is within threshold
        const baselineQuality = groundTruthResults.length > 0 ? 1.0 : 0.8; // Assume perfect baseline
        const qualityDegradation = baselineQuality - qualityScore;

        if (qualityDegradation <= this.config.qualityThreshold && combinedScore > bestScore) {
          bestScore = combinedScore;
          bestEf = efSearch;
        }
      }

      this.config.efSearch = bestEf;
      
      span.setAttributes({
        success: true,
        best_efSearch: bestEf,
        best_score: bestScore
      });

      console.log(`🎯 Optimal efSearch found: ${bestEf} (score: ${bestScore.toFixed(3)})`);

      return bestEf;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Evaluate performance for a specific efSearch value
   */
  private async evaluateEfSearchPerformance(
    testQueries: Float32Array[],
    groundTruthResults: HNSWSearchResult[][],
    k: number,
    efSearch: number
  ): Promise<HNSWPerformanceMetrics> {
    let totalLatency = 0;
    let totalCandidates = 0;
    let totalComputations = 0;
    let totalQuality = 0;

    for (let i = 0; i < testQueries.length; i++) {
      const query = testQueries[i]!;
      const startTime = Date.now();
      
      const results = await this.search(query, k, efSearch);
      
      const latency = Date.now() - startTime;
      totalLatency += latency;
      totalCandidates += results.length;

      // Estimate quality (would use actual nDCG in production)
      const quality = groundTruthResults[i] 
        ? this.calculateNDCG(results, groundTruthResults[i]!, k)
        : this.estimateQualityScore(results);
      
      totalQuality += quality;
    }

    return {
      search_latency_ms: totalLatency / testQueries.length,
      candidates_evaluated: totalCandidates / testQueries.length,
      distance_computations: totalComputations / testQueries.length,
      quality_score: totalQuality / testQueries.length,
      throughput_qps: 1000 / (totalLatency / testQueries.length)
    };
  }

  /**
   * Calculate nDCG for quality evaluation
   */
  private calculateNDCG(
    results: HNSWSearchResult[],
    groundTruth: HNSWSearchResult[],
    k: number
  ): number {
    // Simplified nDCG calculation
    const groundTruthSet = new Set(groundTruth.slice(0, k).map(r => r.doc_id));
    
    let dcg = 0;
    let idcg = 0;

    for (let i = 0; i < Math.min(k, results.length); i++) {
      const relevance = groundTruthSet.has(results[i]!.doc_id) ? 1 : 0;
      const position = i + 1;
      dcg += relevance / Math.log2(position + 1);
    }

    // Ideal DCG (assuming perfect ranking)
    for (let i = 0; i < Math.min(k, groundTruth.length); i++) {
      const position = i + 1;
      idcg += 1 / Math.log2(position + 1);
    }

    return idcg > 0 ? dcg / idcg : 0;
  }

  /**
   * Utility methods
   */
  private findNearestNeighborsForConstruction(
    queryVector: Float32Array,
    candidates: HNSWNode[],
    count: number
  ): Array<{ nodeId: number; distance: number }> {
    return candidates
      .map(node => ({
        nodeId: node.id,
        distance: this.calculateDistance(queryVector, node.vector)
      }))
      .sort((a, b) => a.distance - b.distance)
      .slice(0, count);
  }

  private calculateDistance(v1: Float32Array, v2: Float32Array): number {
    // Cosine distance = 1 - cosine similarity
    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0; i < v1.length; i++) {
      dotProduct += v1[i]! * v2[i]!;
      norm1 += v1[i]! * v1[i]!;
      norm2 += v2[i]! * v2[i]!;
    }

    const cosineDistance = 1 - (dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2)));
    return Math.max(0, cosineDistance); // Ensure non-negative
  }

  private pruneConnections(node: HNSWNode, allNodes: Map<number, HNSWNode>): void {
    if (node.connections.size <= this.config.K) return;

    // Keep only K best connections based on distance
    const connectionDistances = Array.from(node.connections).map(connId => {
      const connNode = allNodes.get(connId);
      return {
        nodeId: connId,
        distance: connNode ? this.calculateDistance(node.vector, connNode.vector) : Infinity
      };
    });

    connectionDistances.sort((a, b) => a.distance - b.distance);
    const keepConnections = connectionDistances.slice(0, this.config.K);

    node.connections.clear();
    for (const conn of keepConnections) {
      node.connections.add(conn.nodeId);
    }
  }

  private pruneSearchCandidates(candidates: Map<number, number>, maxSize: number): void {
    if (candidates.size <= maxSize) return;

    const sorted = Array.from(candidates.entries())
      .sort((a, b) => a[1] - b[1])
      .slice(0, maxSize);

    candidates.clear();
    for (const [nodeId, distance] of sorted) {
      candidates.set(nodeId, distance);
    }
  }

  private selectNodesForLevel(nodes: HNSWNode[], level: number): HNSWNode[] {
    // Select approximately 1/levelMultiplier fraction for next level
    const selectionProb = 1 / Math.pow(this.config.levelMultiplier, level);
    const selected: HNSWNode[] = [];

    for (const node of nodes) {
      // Use node ID for deterministic selection
      const hash = node.id * 2654435761; // Knuth's multiplicative hash
      const normalizedHash = (hash % 1000000) / 1000000;
      
      if (normalizedHash < selectionProb) {
        selected.push(node);
      }
    }

    return selected;
  }

  private findBestEntryPoint(layer: HNSWLayer): HNSWNode {
    // Select node with highest degree as entry point
    let bestNode = null;
    let maxConnections = 0;

    for (const node of layer.nodes.values()) {
      if (node.connections.size > maxConnections) {
        maxConnections = node.connections.size;
        bestNode = node;
      }
    }

    if (bestNode) {
      return bestNode;
    }
    
    // Fallback to first node if no best node found
    const firstNode = layer.nodes.values().next().value;
    if (!firstNode) {
      throw new Error('Cannot find entry point: layer has no nodes');
    }
    
    return firstNode;
  }

  private calculateAverageConnections(): number {
    if (!this.index || this.index.layers.length === 0) return 0;

    const layer0 = this.index.layers[0]!;
    const totalConnections = Array.from(layer0.nodes.values())
      .reduce((sum, node) => sum + node.connections.size, 0);

    return totalConnections / layer0.nodes.size;
  }

  private estimateQualityScore(results: HNSWSearchResult[]): number {
    // Simple quality estimation based on score distribution
    if (results.length === 0) return 0;

    const scores = results.map(r => r.score);
    const avgScore = scores.reduce((sum, score) => sum + score, 0) / scores.length;
    const scoreVariance = scores.reduce((sum, score) => sum + Math.pow(score - avgScore, 2), 0) / scores.length;

    // Higher average scores and lower variance indicate better quality
    return Math.min(1, avgScore * (1 - Math.sqrt(scoreVariance)));
  }

  private recordPerformanceMetrics(metrics: HNSWPerformanceMetrics): void {
    this.performanceHistory.push(metrics);
    
    // Keep history bounded
    if (this.performanceHistory.length > 1000) {
      this.performanceHistory = this.performanceHistory.slice(-500);
    }
  }

  /**
   * Get optimization statistics and performance metrics
   */
  getStats() {
    const avgLatency = this.performanceHistory.length > 0
      ? this.performanceHistory.reduce((sum, m) => sum + m.search_latency_ms, 0) / this.performanceHistory.length
      : 0;

    const avgQuality = this.performanceHistory.length > 0
      ? this.performanceHistory.reduce((sum, m) => sum + m.quality_score, 0) / this.performanceHistory.length
      : 0;

    return {
      config: this.config,
      index: {
        layers: this.index?.layers.length || 0,
        nodes: this.docIdToNodeId.size,
        avg_connections: this.calculateAverageConnections(),
        entry_point: this.index?.entry_point || -1
      },
      performance: {
        avg_search_latency_ms: avgLatency,
        avg_quality_score: avgQuality,
        total_searches: this.performanceHistory.length
      }
    };
  }

  /**
   * Update configuration for tuning
   */
  updateConfig(newConfig: Partial<HNSWOptimizationConfig>): void {
    // Ensure K remains fixed at 150
    if (newConfig.K !== undefined && newConfig.K !== 150) {
      console.warn('🔧 K parameter cannot be changed from 150 per B3 requirements');
      delete newConfig.K;
    }

    this.config = { ...this.config, ...newConfig };
    
    // Update index configuration if needed
    if (this.index && newConfig.efSearch !== undefined) {
      console.log(`🔧 Updated efSearch: ${newConfig.efSearch}`);
    }

    console.log(`🚀 HNSW config updated: ${JSON.stringify(this.config)}`);
  }
}