/**
 * Program-Slice Recall (Stage-B++) - Evergreen Optimization System #1
 * 
 * Path-sensitive def-use slicing over SymbolGraph before ANN.
 * Bounded interprocedural slice: def→use→caller↔callee, aliases, re-exports
 * Constraints: depth≤2, nodes≤K (K≤64)
 * 
 * Gate: Recall@50 +0.7-1.2pp, p95 ≤ +0.8ms, span=100%, vendor veto honored
 * Roll to 25% for NL+symbol queries
 */

import type { 
  SymbolDefinition, 
  SymbolReference, 
  Candidate, 
  SearchContext,
  SymbolKind,
  MatchReason
} from '../types/core.js';
import type { SearchHit } from './span_resolver/types.js';
import { LensTracer } from '../telemetry/tracer.js';

export interface SliceNode {
  id: string;
  file_path: string;
  line: number;
  col: number;
  symbol_name: string;
  kind: SymbolKind;
  scope: string;
  signature?: string;
  // Graph relationships
  defines: string[]; // symbols this node defines
  uses: string[]; // symbols this node uses
  calls: string[]; // functions this node calls
  called_by: string[]; // functions that call this node
  aliases: string[]; // alias names for this symbol
  re_exports: string[]; // re-exported names
}

export interface SliceEdge {
  from: string; // node id
  to: string; // node id
  type: 'def' | 'use' | 'call' | 'alias' | 're-export';
  weight: number;
}

export interface SliceResult {
  nodes: SliceNode[];
  paths: SlicePath[];
  total_depth: number;
  node_count: number;
  vetoed_paths: number;
}

export interface SlicePath {
  nodes: string[]; // sequence of node ids
  edge_types: ('def' | 'use' | 'call' | 'alias' | 're-export')[];
  score: number;
  is_plumbing: boolean; // glue/plumbing code detection
}

/**
 * Symbol graph for interprocedural program slicing
 */
export class SymbolGraph {
  private nodes: Map<string, SliceNode> = new Map();
  private edges: Map<string, SliceEdge[]> = new Map(); // from_id -> edges
  private symbolToNodes: Map<string, string[]> = new Map(); // symbol_name -> node_ids
  private fileToNodes: Map<string, string[]> = new Map(); // file_path -> node_ids

  // Path role veto patterns - block vendor/third_party traversal
  private readonly pathRoleVetos = [
    /node_modules\//,
    /vendor\//,
    /third[_-]party\//,
    /\.vendor\//,
    /external\//,
    /deps\//,
    /__pycache__\//,
    /\.git\//,
    /\.svn\//,
    /\.hg\//,
  ];

  // Topic leash - limit expansion across unrelated domains
  private readonly topicBoundaries = [
    /test[s]?\//,
    /spec[s]?\//,
    /example[s]?\//,
    /demo[s]?\//,
    /benchmark[s]?\//,
    /script[s]?\//,
    /tool[s]?\//,
    /util[s]?\//,
    /helper[s]?\//,
  ];

  constructor() {}

  /**
   * Add a symbol definition to the graph
   */
  addSymbolDefinition(symbol: SymbolDefinition): void {
    const nodeId = `${symbol.file_path}:${symbol.line}:${symbol.col}:${symbol.name}`;
    
    const node: SliceNode = {
      id: nodeId,
      file_path: symbol.file_path,
      line: symbol.line,
      col: symbol.col,
      symbol_name: symbol.name,
      kind: symbol.kind,
      scope: symbol.scope,
      signature: symbol.signature,
      defines: [symbol.name],
      uses: [],
      calls: [],
      called_by: [],
      aliases: [],
      re_exports: [],
    };

    this.nodes.set(nodeId, node);
    
    // Index by symbol name
    const symbolNodes = this.symbolToNodes.get(symbol.name) || [];
    symbolNodes.push(nodeId);
    this.symbolToNodes.set(symbol.name, symbolNodes);
    
    // Index by file
    const fileNodes = this.fileToNodes.get(symbol.file_path) || [];
    fileNodes.push(nodeId);
    this.fileToNodes.set(symbol.file_path, fileNodes);
  }

  /**
   * Add a symbol reference/use to the graph
   */
  addSymbolReference(reference: SymbolReference): void {
    const refNodeId = `${reference.file_path}:${reference.line}:${reference.col}:${reference.symbol_name}`;
    
    // Find or create reference node
    let refNode = this.nodes.get(refNodeId);
    if (!refNode) {
      refNode = {
        id: refNodeId,
        file_path: reference.file_path,
        line: reference.line,
        col: reference.col,
        symbol_name: reference.symbol_name,
        kind: 'variable', // default, could be refined
        scope: 'unknown',
        defines: [],
        uses: [reference.symbol_name],
        calls: [],
        called_by: [],
        aliases: [],
        re_exports: [],
      };
      this.nodes.set(refNodeId, refNode);
    }

    // Create use edge from reference to definition
    const defNodes = this.symbolToNodes.get(reference.symbol_name) || [];
    for (const defNodeId of defNodes) {
      const defNode = this.nodes.get(defNodeId);
      if (defNode && defNode.defines.includes(reference.symbol_name)) {
        this.addEdge(refNodeId, defNodeId, 'use', 1.0);
      }
    }
  }

  /**
   * Add call relationship between functions
   */
  addCallRelation(caller: string, callee: string, callerFile: string, calleeLine: number): void {
    const callerNodes = this.findNodesBySymbol(caller, callerFile);
    const calleeNodes = this.symbolToNodes.get(callee) || [];

    for (const callerNodeId of callerNodes) {
      for (const calleeNodeId of calleeNodes) {
        const callerNode = this.nodes.get(callerNodeId);
        const calleeNode = this.nodes.get(calleeNodeId);
        
        if (callerNode && calleeNode) {
          // Add call edge
          callerNode.calls.push(callee);
          calleeNode.called_by.push(caller);
          
          this.addEdge(callerNodeId, calleeNodeId, 'call', 0.8);
        }
      }
    }
  }

  /**
   * Add alias relationship (e.g., import aliases)
   */
  addAliasRelation(original: string, alias: string, filePath: string): void {
    const originalNodes = this.symbolToNodes.get(original) || [];
    
    for (const nodeId of originalNodes) {
      const node = this.nodes.get(nodeId);
      if (node && node.file_path === filePath) {
        node.aliases.push(alias);
        
        // Create alias edge
        const aliasNodeId = `${filePath}:${node.line}:${node.col}:${alias}`;
        const aliasNode: SliceNode = {
          id: aliasNodeId,
          file_path: filePath,
          line: node.line,
          col: node.col,
          symbol_name: alias,
          kind: node.kind,
          scope: node.scope,
          signature: node.signature,
          defines: [alias],
          uses: [],
          calls: [],
          called_by: [],
          aliases: [original],
          re_exports: [],
        };
        
        this.nodes.set(aliasNodeId, aliasNode);
        this.addEdge(nodeId, aliasNodeId, 'alias', 0.9);
      }
    }
  }

  /**
   * Add re-export relationship
   */
  addReExport(original: string, exported: string, fromFile: string, toFile: string): void {
    const originalNodes = this.findNodesBySymbol(original, fromFile);
    
    for (const nodeId of originalNodes) {
      const node = this.nodes.get(nodeId);
      if (node) {
        node.re_exports.push(exported);
        
        // Create re-export node in target file
        const reExportNodeId = `${toFile}:1:0:${exported}`;
        const reExportNode: SliceNode = {
          id: reExportNodeId,
          file_path: toFile,
          line: 1,
          col: 0,
          symbol_name: exported,
          kind: node.kind,
          scope: 'export',
          signature: node.signature,
          defines: [exported],
          uses: [original],
          calls: [],
          called_by: [],
          aliases: [],
          re_exports: [],
        };
        
        this.nodes.set(reExportNodeId, reExportNode);
        this.addEdge(nodeId, reExportNodeId, 're-export', 0.7);
      }
    }
  }

  /**
   * Perform bounded interprocedural program slice
   */
  async performSlice(
    seedSymbols: string[],
    maxDepth: number = 2,
    maxNodes: number = 64
  ): Promise<SliceResult> {
    const span = LensTracer.createChildSpan('program_slice', {
      'slice.seed_count': seedSymbols.length,
      'slice.max_depth': maxDepth,
      'slice.max_nodes': maxNodes,
    });

    try {
      const visitedNodes = new Set<string>();
      const slicePaths: SlicePath[] = [];
      const queue: { nodeId: string; depth: number; path: string[]; edgeTypes: ('def' | 'use' | 'call' | 'alias' | 're-export')[] }[] = [];
      let vetoedPaths = 0;

      // Initialize with seed symbols
      for (const seedSymbol of seedSymbols) {
        const seedNodeIds = this.symbolToNodes.get(seedSymbol) || [];
        for (const nodeId of seedNodeIds) {
          queue.push({ nodeId, depth: 0, path: [nodeId], edgeTypes: [] });
        }
      }

      // Bounded BFS traversal
      while (queue.length > 0 && visitedNodes.size < maxNodes) {
        const current = queue.shift()!;
        
        if (current.depth > maxDepth || visitedNodes.has(current.nodeId)) {
          continue;
        }

        const node = this.nodes.get(current.nodeId);
        if (!node) continue;

        // Check path role veto
        if (this.shouldVetoPath(node.file_path)) {
          vetoedPaths++;
          continue;
        }

        // Check topic leash - don't traverse into different domains
        if (current.depth > 0 && this.shouldApplyTopicLeash(current.path, node.file_path)) {
          continue;
        }

        visitedNodes.add(current.nodeId);

        // Record path if it represents plumbing/glue code
        if (this.isPlumbingCode(node, current.path)) {
          slicePaths.push({
            nodes: [...current.path],
            edge_types: [...current.edgeTypes],
            score: this.calculatePathScore(current.path, current.edgeTypes),
            is_plumbing: true,
          });
        }

        // Expand neighbors if within depth limit
        if (current.depth < maxDepth) {
          const edges = this.edges.get(current.nodeId) || [];
          for (const edge of edges) {
            if (!visitedNodes.has(edge.to) && visitedNodes.size < maxNodes) {
              queue.push({
                nodeId: edge.to,
                depth: current.depth + 1,
                path: [...current.path, edge.to],
                edgeTypes: [...current.edgeTypes, edge.type],
              });
            }
          }
        }
      }

      const result: SliceResult = {
        nodes: Array.from(visitedNodes).map(id => this.nodes.get(id)!).filter(n => n),
        paths: slicePaths,
        total_depth: Math.max(...queue.map(q => q.depth), 0),
        node_count: visitedNodes.size,
        vetoed_paths: vetoedPaths,
      };

      span.setAttributes({
        success: true,
        'slice.nodes_found': result.node_count,
        'slice.paths_found': result.paths.length,
        'slice.vetoed_paths': vetoedPaths,
      });

      return result;
      
    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Convert slice results to search hits with slice_hit why reason
   */
  convertSliceToHits(sliceResult: SliceResult): SearchHit[] {
    const hits: SearchHit[] = [];

    for (const path of sliceResult.paths) {
      if (path.is_plumbing && path.nodes.length > 0) {
        for (const nodeId of path.nodes) {
          const node = this.nodes.get(nodeId);
          if (node) {
            hits.push({
              file: node.file_path,
              line: node.line,
              col: node.col,
              score: path.score,
              why: ['slice_hit' as MatchReason],
              symbol_kind: node.kind,
              symbol_name: node.symbol_name,
              signature: node.signature,
              ast_path: node.scope,
              snippet: this.generateSnippet(node),
            });
          }
        }
      }
    }

    return hits;
  }

  // Private helper methods

  private addEdge(from: string, to: string, type: 'def' | 'use' | 'call' | 'alias' | 're-export', weight: number): void {
    const edge: SliceEdge = { from, to, type, weight };
    const edges = this.edges.get(from) || [];
    edges.push(edge);
    this.edges.set(from, edges);
  }

  private findNodesBySymbol(symbol: string, filePath?: string): string[] {
    const nodeIds = this.symbolToNodes.get(symbol) || [];
    if (filePath) {
      return nodeIds.filter(id => {
        const node = this.nodes.get(id);
        return node && node.file_path === filePath;
      });
    }
    return nodeIds;
  }

  private shouldVetoPath(filePath: string): boolean {
    return this.pathRoleVetos.some(pattern => pattern.test(filePath));
  }

  private shouldApplyTopicLeash(currentPath: string[], newFilePath: string): boolean {
    if (currentPath.length === 0) return false;
    
    const currentNode = this.nodes.get(currentPath[currentPath.length - 1]);
    if (!currentNode) return false;

    const currentFile = currentNode.file_path;
    const currentTopic = this.getFileTopic(currentFile);
    const newTopic = this.getFileTopic(newFilePath);

    // Don't traverse between different bounded topics
    return currentTopic !== newTopic && 
           (this.isBoundedTopic(currentTopic) || this.isBoundedTopic(newTopic));
  }

  private getFileTopic(filePath: string): string {
    for (const pattern of this.topicBoundaries) {
      if (pattern.test(filePath)) {
        const match = filePath.match(pattern);
        return match ? match[0] : 'unknown';
      }
    }
    return 'core';
  }

  private isBoundedTopic(topic: string): boolean {
    return topic !== 'core' && topic !== 'unknown';
  }

  private isPlumbingCode(node: SliceNode, path: string[]): boolean {
    // Detect plumbing/glue code patterns:
    // 1. Simple forwarding functions
    // 2. Re-export statements
    // 3. Adapter/wrapper patterns
    // 4. Configuration wiring
    
    if (node.re_exports.length > 0) return true;
    if (node.aliases.length > node.defines.length) return true;
    if (node.kind === 'function' && node.signature && 
        (node.signature.includes('wrapper') || 
         node.signature.includes('adapter') ||
         node.signature.includes('proxy'))) return true;
    
    // Simple forwarding: uses more symbols than it defines
    if (node.uses.length > node.defines.length && node.uses.length > 2) return true;
    
    // Multi-hop paths through utilities are often plumbing
    if (path.length > 2 && node.file_path.includes('util')) return true;
    
    return false;
  }

  private calculatePathScore(path: string[], edgeTypes: ('def' | 'use' | 'call' | 'alias' | 're-export')[]): number {
    let score = 1.0;
    
    // Penalize longer paths
    score *= Math.pow(0.9, path.length - 1);
    
    // Weight by edge types
    for (const edgeType of edgeTypes) {
      switch (edgeType) {
        case 'def': score *= 1.0; break;
        case 'use': score *= 0.9; break;
        case 'call': score *= 0.8; break;
        case 'alias': score *= 0.9; break;
        case 're-export': score *= 0.7; break;
      }
    }
    
    return Math.max(0.1, score);
  }

  private generateSnippet(node: SliceNode): string {
    return `${node.kind} ${node.symbol_name}${node.signature ? ` ${node.signature}` : ''}`;
  }

  /**
   * Get statistics about the symbol graph
   */
  getStats(): {
    nodes: number;
    edges: number;
    symbols: number;
    files: number;
  } {
    const totalEdges = Array.from(this.edges.values()).reduce((sum, edges) => sum + edges.length, 0);
    
    return {
      nodes: this.nodes.size,
      edges: totalEdges,
      symbols: this.symbolToNodes.size,
      files: this.fileToNodes.size,
    };
  }
}

/**
 * Path-sensitive program slicing engine
 */
export class PathSensitiveSlicing {
  private symbolGraph: SymbolGraph;
  private enabled: boolean = false;
  private rolloutPercentage: number = 0; // Start at 0%, roll to 25%

  constructor(symbolGraph: SymbolGraph) {
    this.symbolGraph = symbolGraph;
  }

  /**
   * Enable program slice recall with rollout percentage
   */
  enableWithRollout(percentage: number): void {
    this.enabled = true;
    this.rolloutPercentage = Math.min(100, Math.max(0, percentage));
  }

  /**
   * Check if slice recall should be applied for this query
   */
  shouldApplySlicing(context: SearchContext): boolean {
    if (!this.enabled) return false;
    
    // Apply rollout percentage
    const hash = this.hashString(context.trace_id);
    const roll = hash % 100;
    if (roll >= this.rolloutPercentage) return false;

    // Apply only to NL+symbol queries as specified
    const hasNaturalLanguage = /\s/.test(context.query) && context.query.length > 10;
    const hasSymbolPattern = /[a-zA-Z_][a-zA-Z0-9_]*/.test(context.query);
    
    return hasNaturalLanguage && hasSymbolPattern;
  }

  /**
   * Perform program slice recall to find plumbing/glue code
   */
  async performSliceRecall(
    context: SearchContext,
    seedCandidates: Candidate[]
  ): Promise<SearchHit[]> {
    const span = LensTracer.createChildSpan('slice_recall', {
      'context.repo_sha': context.repo_sha,
      'context.query': context.query,
      'candidates.count': seedCandidates.length,
    });

    try {
      if (!this.shouldApplySlicing(context)) {
        span.setAttributes({ success: true, skipped: true, reason: 'rollout_or_criteria' });
        return [];
      }

      // Extract seed symbols from candidates
      const seedSymbols = new Set<string>();
      for (const candidate of seedCandidates) {
        if (candidate.symbol_kind && candidate.ast_path) {
          // Extract symbol name from ast_path or context
          const symbolName = this.extractSymbolName(candidate);
          if (symbolName) {
            seedSymbols.add(symbolName);
          }
        }
      }

      if (seedSymbols.size === 0) {
        span.setAttributes({ success: true, skipped: true, reason: 'no_seed_symbols' });
        return [];
      }

      // Perform bounded slice with constraints: depth≤2, nodes≤64
      const sliceResult = await this.symbolGraph.performSlice(
        Array.from(seedSymbols),
        2, // maxDepth ≤ 2
        64 // maxNodes ≤ 64
      );

      // Convert to search hits with slice_hit reason
      const sliceHits = this.symbolGraph.convertSliceToHits(sliceResult);

      span.setAttributes({
        success: true,
        'slice.seed_symbols': seedSymbols.size,
        'slice.nodes_found': sliceResult.node_count,
        'slice.paths_found': sliceResult.paths.length,
        'slice.hits_returned': sliceHits.length,
        'slice.vetoed_paths': sliceResult.vetoed_paths,
      });

      return sliceHits;
      
    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      return [];
    } finally {
      span.end();
    }
  }

  // Private helper methods

  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash);
  }

  private extractSymbolName(candidate: Candidate): string | null {
    // Try to extract symbol name from various candidate fields
    if (candidate.ast_path && candidate.symbol_kind) {
      // Parse symbol name from AST path or context
      const contextMatch = candidate.context?.match(/(?:function|class|const|let|var)\s+(\w+)/);
      if (contextMatch) return contextMatch[1];
      
      const pathMatch = candidate.ast_path.match(/(\w+)$/);
      if (pathMatch) return pathMatch[1];
    }
    
    return null;
  }
}