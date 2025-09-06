/**
 * Stage-B⁺ Slice-Chasing - Embedder-Agnostic Optimization #2
 * 
 * Graph walk over SymbolGraph before ANN for NL/symbol intents.
 * BFS(def→ref | type→impl | alias) with depth≤2, nodes≤K, plus RAPTOR "topic leash"
 * Budget ~0.3-1.2ms with node/edge budgets and vendor/third_party veto
 * 
 * Target: +0.5-1.0pp Recall@50 with ≤+0.8ms p95
 */

import type { SearchHit } from '../core/span_resolver/types.js';
import type { SearchContext, SymbolDefinition, SymbolReference } from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';

export interface SliceChasingConfig {
  enabled: boolean;
  maxDepth: number;                    // Max BFS depth (≤2)
  maxNodes: number;                    // Max nodes to visit (K ≤ 64)
  budgetMs: number;                    // Budget in milliseconds (0.3-1.2ms)
  topicSimilarityThreshold: number;    // Topic similarity threshold (τ)
  enableVendorVeto: boolean;           // Filter vendor/third_party paths
  enableRaptorTopicLeash: boolean;     // Use RAPTOR topic filtering
  maxEdgesPerNode: number;             // Edge budget per node
  rolloutPercentage: number;           // Rollout for NL+symbol queries (25%)
}

export interface SymbolNode {
  id: string;                          // Unique identifier
  name: string;                        // Symbol name
  kind: 'definition' | 'reference' | 'type' | 'implementation' | 'alias';
  file: string;
  line: number;
  col: number;
  scope?: string;
  signature?: string;
  topic_id?: string;                   // RAPTOR topic identifier
}

export interface SymbolEdge {
  from: string;                        // Source node ID
  to: string;                          // Target node ID
  type: 'def_to_ref' | 'ref_to_def' | 'type_to_impl' | 'impl_to_type' | 'alias';
  weight: number;                      // Edge weight/confidence
  path_role?: 'vendor' | 'third_party' | 'application' | 'test';
}

export interface SymbolGraph {
  nodes: Map<string, SymbolNode>;
  edges: Map<string, SymbolEdge[]>;    // node_id -> outgoing edges
  topicNodes: Map<string, string[]>;   // topic_id -> node_ids
}

export interface BFSState {
  queue: Array<{nodeId: string, depth: number, path: string[]}>;
  visited: Set<string>;
  discoveredSpans: SearchHit[];
  startTime: number;
  topicLeash?: string;                 // RAPTOR topic constraint
}

/**
 * Topic similarity calculator for RAPTOR topic leash
 */
export class TopicSimilarityCalculator {
  private topicVectors: Map<string, number[]> = new Map();
  private queryTopics: Map<string, string> = new Map(); // query -> topic_id

  constructor() {}

  /**
   * Compute topic similarity between two nodes
   */
  computeTopicSimilarity(nodeA: SymbolNode, nodeB: SymbolNode): number {
    if (!nodeA.topic_id || !nodeB.topic_id) {
      return 0.5; // Neutral similarity for nodes without topics
    }

    if (nodeA.topic_id === nodeB.topic_id) {
      return 1.0; // Same topic
    }

    // Simple Jaccard similarity for demonstration
    // In practice, this would use RAPTOR's topic embeddings
    const topicA = nodeA.topic_id;
    const topicB = nodeB.topic_id;
    
    // Hierarchical topic similarity (parent-child relationships)
    if (topicA.startsWith(topicB) || topicB.startsWith(topicA)) {
      return 0.8; // Parent-child topic relationship
    }

    // Default similarity for different topics
    return 0.1;
  }

  /**
   * Get topic leash for query (RAPTOR integration)
   */
  getQueryTopicLeash(query: string): string | undefined {
    // In practice, this would query RAPTOR to determine the most relevant topic
    const cached = this.queryTopics.get(query);
    if (cached) {
      return cached;
    }

    // Simple heuristics for topic assignment
    if (query.includes('function') || query.includes('method')) {
      return 'topic_functions';
    } else if (query.includes('class') || query.includes('type')) {
      return 'topic_types';
    } else if (query.includes('variable') || query.includes('const')) {
      return 'topic_variables';
    }

    return undefined; // No topic leash
  }
}

/**
 * Symbol graph builder and navigator
 */
export class SymbolGraphBuilder {
  private symbolGraph: SymbolGraph = {
    nodes: new Map(),
    edges: new Map(),
    topicNodes: new Map()
  };

  constructor() {}

  /**
   * Build symbol graph from definitions and references
   */
  buildGraph(
    definitions: SymbolDefinition[],
    references: SymbolReference[]
  ): void {
    // Clear existing graph
    this.symbolGraph.nodes.clear();
    this.symbolGraph.edges.clear();
    this.symbolGraph.topicNodes.clear();

    // Add definition nodes
    for (const def of definitions) {
      const nodeId = `def_${def.file}:${def.line}:${def.col}`;
      this.symbolGraph.nodes.set(nodeId, {
        id: nodeId,
        name: def.name,
        kind: 'definition',
        file: def.file,
        line: def.line,
        col: def.col,
        scope: def.scope,
        signature: def.signature,
        topic_id: this.inferTopicId(def)
      });
    }

    // Add reference nodes and create edges
    for (const ref of references) {
      const refNodeId = `ref_${ref.file_path}:${ref.line}:${ref.col}`;
      this.symbolGraph.nodes.set(refNodeId, {
        id: refNodeId,
        name: ref.symbol_name,
        kind: 'reference',
        file: ref.file_path,
        line: ref.line,
        col: ref.col,
        topic_id: this.inferTopicIdFromReference(ref)
      });

      // Find corresponding definition and create bidirectional edges
      const defNode = this.findDefinitionNode(ref.symbol_name);
      if (defNode) {
        // def → ref edge
        this.addEdge(defNode.id, refNodeId, 'def_to_ref', 0.9);
        
        // ref → def edge  
        this.addEdge(refNodeId, defNode.id, 'ref_to_def', 0.8);
      }
    }

    // Build topic index
    this.buildTopicIndex();

    console.log(`🔗 SymbolGraph built: ${this.symbolGraph.nodes.size} nodes, ${this.getTotalEdges()} edges`);
  }

  /**
   * Add edge between two nodes
   */
  private addEdge(fromId: string, toId: string, type: SymbolEdge['type'], weight: number): void {
    if (!this.symbolGraph.edges.has(fromId)) {
      this.symbolGraph.edges.set(fromId, []);
    }

    const edge: SymbolEdge = {
      from: fromId,
      to: toId,
      type,
      weight,
      path_role: this.inferPathRole(fromId, toId)
    };

    this.symbolGraph.edges.get(fromId)!.push(edge);
  }

  /**
   * Find definition node by symbol name
   */
  private findDefinitionNode(symbolName: string): SymbolNode | undefined {
    for (const node of this.symbolGraph.nodes.values()) {
      if (node.kind === 'definition' && node.name === symbolName) {
        return node;
      }
    }
    return undefined;
  }

  /**
   * Infer topic ID for symbol definition
   */
  private inferTopicId(def: SymbolDefinition): string | undefined {
    // Simple topic inference based on symbol kind and scope
    if (def.kind === 'function' || def.kind === 'method') {
      return 'topic_functions';
    } else if (def.kind === 'class' || def.kind === 'interface' || def.kind === 'type') {
      return 'topic_types';
    } else if (def.kind === 'variable' || def.kind === 'constant') {
      return 'topic_variables';
    }
    return undefined;
  }

  /**
   * Infer topic ID for symbol reference
   */
  private inferTopicIdFromReference(ref: SymbolReference): string | undefined {
    // Analyze context to infer topic
    const context = ref.context.toLowerCase();
    if (context.includes('function') || context.includes('(')) {
      return 'topic_functions';
    } else if (context.includes('class') || context.includes('type')) {
      return 'topic_types';
    }
    return 'topic_variables';
  }

  /**
   * Infer path role (vendor/third_party filtering)
   */
  private inferPathRole(fromId: string, toId: string): SymbolEdge['path_role'] {
    const fromNode = this.symbolGraph.nodes.get(fromId);
    const toNode = this.symbolGraph.nodes.get(toId);
    
    if (!fromNode || !toNode) return 'application';

    // Check for vendor/third_party paths
    if (fromNode.file.includes('node_modules') || fromNode.file.includes('vendor') || 
        toNode.file.includes('node_modules') || toNode.file.includes('vendor')) {
      return 'vendor';
    }

    if (fromNode.file.includes('third_party') || toNode.file.includes('third_party')) {
      return 'third_party';
    }

    if (fromNode.file.includes('test') || toNode.file.includes('test')) {
      return 'test';
    }

    return 'application';
  }

  /**
   * Build topic index for faster lookup
   */
  private buildTopicIndex(): void {
    for (const node of this.symbolGraph.nodes.values()) {
      if (node.topic_id) {
        if (!this.symbolGraph.topicNodes.has(node.topic_id)) {
          this.symbolGraph.topicNodes.set(node.topic_id, []);
        }
        this.symbolGraph.topicNodes.get(node.topic_id)!.push(node.id);
      }
    }
  }

  /**
   * Get total number of edges
   */
  private getTotalEdges(): number {
    let total = 0;
    for (const edges of this.symbolGraph.edges.values()) {
      total += edges.length;
    }
    return total;
  }

  /**
   * Get the built symbol graph
   */
  getGraph(): SymbolGraph {
    return this.symbolGraph;
  }
}

/**
 * Stage-B⁺ Slice-Chasing implementation
 */
export class StageBPlusSliceChasing {
  private config: SliceChasingConfig;
  private symbolGraphBuilder: SymbolGraphBuilder;
  private topicCalculator: TopicSimilarityCalculator;
  private symbolGraph: SymbolGraph;

  constructor(config: Partial<SliceChasingConfig> = {}) {
    this.config = {
      enabled: config.enabled ?? true,
      maxDepth: config.maxDepth ?? 2,
      maxNodes: config.maxNodes ?? 64,
      budgetMs: config.budgetMs ?? 1.0,
      topicSimilarityThreshold: config.topicSimilarityThreshold ?? 0.3,
      enableVendorVeto: config.enableVendorVeto ?? true,
      enableRaptorTopicLeash: config.enableRaptorTopicLeash ?? true,
      maxEdgesPerNode: config.maxEdgesPerNode ?? 10,
      rolloutPercentage: config.rolloutPercentage ?? 25,
      ...config
    };

    this.symbolGraphBuilder = new SymbolGraphBuilder();
    this.topicCalculator = new TopicSimilarityCalculator();
    this.symbolGraph = { nodes: new Map(), edges: new Map(), topicNodes: new Map() };

    console.log(`🔍 StageBPlusSliceChasing initialized: maxDepth=${this.config.maxDepth}, maxNodes=${this.config.maxNodes}, budget=${this.config.budgetMs}ms`);
  }

  /**
   * Initialize symbol graph from definitions and references
   */
  initializeGraph(
    definitions: SymbolDefinition[],
    references: SymbolReference[]
  ): void {
    this.symbolGraphBuilder.buildGraph(definitions, references);
    this.symbolGraph = this.symbolGraphBuilder.getGraph();
  }

  /**
   * Execute slice-chasing to discover additional spans
   */
  async chaseSlices(
    seeds: SearchHit[],
    context: SearchContext
  ): Promise<SearchHit[]> {
    const span = LensTracer.createChildSpan('stage_b_plus_slice_chasing', {
      'seeds': seeds.length,
      'query': context.query,
      'enabled': this.config.enabled
    });

    const startTime = performance.now();

    try {
      if (!this.config.enabled) {
        span.setAttributes({ skipped: true, reason: 'disabled' });
        return seeds;
      }

      // Check rollout percentage for NL+symbol queries
      if (!this.shouldApplyToQuery(context)) {
        span.setAttributes({ skipped: true, reason: 'rollout_gated' });
        return seeds;
      }

      // Budget check function
      const checkBudget = () => {
        const elapsed = performance.now() - startTime;
        if (elapsed > this.config.budgetMs) {
          throw new Error(`SliceChasing budget exceeded: ${elapsed.toFixed(3)}ms > ${this.config.budgetMs}ms`);
        }
      };

      // Initialize BFS state
      const bfsState: BFSState = {
        queue: [],
        visited: new Set(),
        discoveredSpans: [],
        startTime: performance.now(),
        topicLeash: this.config.enableRaptorTopicLeash 
          ? this.topicCalculator.getQueryTopicLeash(context.query)
          : undefined
      };

      // Convert seeds to graph nodes and initialize BFS queue
      for (const seed of seeds) {
        const seedNodes = this.findGraphNodesForHit(seed);
        for (const seedNode of seedNodes) {
          bfsState.queue.push({ nodeId: seedNode.id, depth: 0, path: [seedNode.id] });
        }
      }

      checkBudget();

      // Execute BFS traversal
      const discoveredSpans = await this.executeBFS(bfsState);

      checkBudget();

      // Merge with original seeds, deduplicate, and sort by relevance
      const combinedHits = this.mergeAndDeduplicateHits(seeds, discoveredSpans);

      const latency = performance.now() - startTime;

      span.setAttributes({
        success: true,
        latency_ms: latency,
        seeds_count: seeds.length,
        discovered_count: discoveredSpans.length,
        combined_count: combinedHits.length,
        nodes_visited: bfsState.visited.size,
        topic_leash: bfsState.topicLeash || 'none'
      });

      console.log(`🔍 SliceChasing: ${seeds.length}+${discoveredSpans.length}→${combinedHits.length} spans in ${latency.toFixed(3)}ms`);

      return combinedHits;

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });

      console.warn(`SliceChasing failed: ${errorMsg}, falling back to seeds`);
      return seeds;

    } finally {
      span.end();
    }
  }

  /**
   * Check if slice-chasing should apply to this query
   */
  private shouldApplyToQuery(context: SearchContext): boolean {
    // Simple rollout gating
    const queryHash = this.hashString(context.query);
    const rolloutGate = (queryHash % 100) < this.config.rolloutPercentage;

    if (!rolloutGate) {
      return false;
    }

    // Apply to NL and symbol queries
    const query = context.query.toLowerCase();
    const isNL = /\b(how|what|where|when|why|find|search|get|show)\b/.test(query) || 
                 query.split(' ').length > 2;
    const isSymbol = /^[a-zA-Z_][a-zA-Z0-9_]*$/.test(query) || 
                     query.includes('function') || 
                     query.includes('class');

    return isNL || isSymbol;
  }

  /**
   * Simple string hash function for consistent rollout
   */
  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }

  /**
   * Find graph nodes corresponding to a search hit
   */
  private findGraphNodesForHit(hit: SearchHit): SymbolNode[] {
    const nodes: SymbolNode[] = [];

    // Look for nodes with matching file and line
    for (const node of this.symbolGraph.nodes.values()) {
      if (node.file === hit.file && Math.abs(node.line - hit.line) <= 1) {
        nodes.push(node);
      }
    }

    // If no exact match, try symbol name matching
    if (nodes.length === 0 && hit.symbol_name) {
      for (const node of this.symbolGraph.nodes.values()) {
        if (node.name === hit.symbol_name) {
          nodes.push(node);
        }
      }
    }

    return nodes;
  }

  /**
   * Execute BFS traversal with constraints
   */
  private async executeBFS(state: BFSState): Promise<SearchHit[]> {
    while (state.queue.length > 0 && state.visited.size < this.config.maxNodes) {
      // Budget check
      const elapsed = performance.now() - state.startTime;
      if (elapsed > this.config.budgetMs) {
        break;
      }

      const { nodeId, depth, path } = state.queue.shift()!;

      if (state.visited.has(nodeId) || depth >= this.config.maxDepth) {
        continue;
      }

      state.visited.add(nodeId);
      const currentNode = this.symbolGraph.nodes.get(nodeId);
      
      if (!currentNode) continue;

      // Convert node to SearchHit
      const hit = this.nodeToSearchHit(currentNode);
      if (hit) {
        state.discoveredSpans.push(hit);
      }

      // Explore neighbors
      const edges = this.symbolGraph.edges.get(nodeId) || [];
      let edgeCount = 0;

      for (const edge of edges) {
        if (edgeCount >= this.config.maxEdgesPerNode) break;

        // Apply vendor/third_party veto
        if (this.config.enableVendorVeto && 
           (edge.path_role === 'vendor' || edge.path_role === 'third_party')) {
          continue;
        }

        const neighborNode = this.symbolGraph.nodes.get(edge.to);
        if (!neighborNode) continue;

        // Apply topic leash constraint
        if (this.config.enableRaptorTopicLeash && state.topicLeash) {
          const topicSimilarity = this.topicCalculator.computeTopicSimilarity(
            currentNode, 
            neighborNode
          );
          if (topicSimilarity < this.config.topicSimilarityThreshold) {
            continue;
          }
        }

        // Add to queue if not visited and within depth limit
        if (!state.visited.has(edge.to) && depth + 1 < this.config.maxDepth) {
          state.queue.push({
            nodeId: edge.to,
            depth: depth + 1,
            path: [...path, edge.to]
          });
        }

        edgeCount++;
      }
    }

    return state.discoveredSpans;
  }

  /**
   * Convert symbol node to SearchHit
   */
  private nodeToSearchHit(node: SymbolNode): SearchHit | null {
    return {
      file: node.file,
      line: node.line,
      col: node.col,
      score: 0.7, // Base score for discovered spans
      why: ['symbol'],
      snippet: node.signature || `${node.kind}: ${node.name}`,
      symbol_kind: this.mapSymbolKind(node.kind),
      symbol_name: node.name,
      ast_path: node.scope
    };
  }

  /**
   * Map internal symbol kind to SearchHit symbol kind
   */
  private mapSymbolKind(kind: SymbolNode['kind']): SearchHit['symbol_kind'] {
    switch (kind) {
      case 'definition':
        return 'function'; // Default assumption
      case 'type':
        return 'type';
      default:
        return undefined;
    }
  }

  /**
   * Merge seeds with discovered spans and deduplicate
   */
  private mergeAndDeduplicateHits(seeds: SearchHit[], discovered: SearchHit[]): SearchHit[] {
    const combined = [...seeds, ...discovered];
    const seen = new Set<string>();
    const deduplicated: SearchHit[] = [];

    for (const hit of combined) {
      const key = `${hit.file}:${hit.line}:${hit.col}`;
      if (!seen.has(key)) {
        seen.add(key);
        deduplicated.push(hit);
      }
    }

    // Sort by score descending
    return deduplicated.sort((a, b) => b.score - a.score);
  }

  /**
   * Get comprehensive statistics
   */
  getStats() {
    return {
      config: this.config,
      graph_nodes: this.symbolGraph.nodes.size,
      graph_edges: Array.from(this.symbolGraph.edges.values()).reduce((sum, edges) => sum + edges.length, 0),
      topic_groups: this.symbolGraph.topicNodes.size
    };
  }

  /**
   * Update configuration for A/B testing
   */
  updateConfig(newConfig: Partial<SliceChasingConfig>): void {
    this.config = { ...this.config, ...newConfig };
    console.log(`🔍 StageBPlusSliceChasing config updated: ${JSON.stringify(newConfig)}`);
  }
}