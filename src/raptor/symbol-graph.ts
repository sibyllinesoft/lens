/**
 * SymbolGraph - LSP-derived graph data layer for RAPTOR system
 * 
 * Provides structured access to symbol definitions, references, implementations,
 * type relationships, aliases, and imports using msgpack serialization.
 */

// Using JSON as fallback for msgpack-lite
const msgpack = {
  encode: (data: any): Buffer => Buffer.from(JSON.stringify(data), 'utf-8'),
  decode: (buffer: Buffer): any => JSON.parse(buffer.toString('utf-8'))
};
import { promises as fs } from 'fs';
import path from 'path';
import { LSPHint, SymbolKind } from '../types/core.js';

export interface SymbolNode {
  id: string;
  name: string;
  kind: SymbolKind;
  file_path: string;
  line: number;
  col: number;
  byte_offset: number;
  span_len: number;
  signature?: string;
  type_info?: string;
  aliases: string[];
  resolved_imports: string[];
  references_count: number;
}

export interface SymbolEdge {
  source_id: string;
  target_id: string;
  edge_type: SymbolEdgeType;
  weight: number;
  metadata?: Record<string, any>;
}

export type SymbolEdgeType = 
  | 'definition'
  | 'reference' 
  | 'implementation'
  | 'type_relation'
  | 'alias'
  | 'import'
  | 'call'
  | 'inheritance'
  | 'composition';

export interface SymbolGraphSnapshot {
  repo_sha: string;
  version: string;
  timestamp: number;
  nodes: Map<string, SymbolNode>;
  edges: Map<string, SymbolEdge[]>; // source_id -> edges
  file_symbol_map: Map<string, string[]>; // file_path -> symbol_ids
  type_hierarchy: Map<string, string[]>; // type_name -> subtypes
  import_graph: Map<string, Set<string>>; // file -> imported_files
  metadata: {
    total_nodes: number;
    total_edges: number;
    coverage_stats: {
      files_with_symbols: number;
      total_files: number;
      symbol_coverage: number;
    };
    build_duration_ms: number;
  };
}

export interface SymbolGraphBuildOptions {
  include_references: boolean;
  include_implementations: boolean;
  include_type_relations: boolean;
  include_call_graph: boolean;
  max_references_per_symbol: number;
  alias_resolution_depth: number;
}

/**
 * SymbolGraph manages LSP-derived symbol relationships and provides
 * efficient access patterns for semantic search enhancement
 */
export class SymbolGraph {
  private snapshot?: SymbolGraphSnapshot;
  private storagePath: string;
  private buildOptions: SymbolGraphBuildOptions;

  constructor(storagePath: string, options?: Partial<SymbolGraphBuildOptions>) {
    this.storagePath = storagePath;
    this.buildOptions = {
      include_references: true,
      include_implementations: true,
      include_type_relations: true,
      include_call_graph: false, // Expensive, disabled by default
      max_references_per_symbol: 1000,
      alias_resolution_depth: 3,
      ...options
    };
  }

  /**
   * Build symbol graph from LSP hints
   */
  async buildFromLSP(
    repoSha: string, 
    lspHints: LSPHint[], 
    progressCallback?: (progress: number) => void
  ): Promise<SymbolGraphSnapshot> {
    const startTime = Date.now();
    
    const nodes = new Map<string, SymbolNode>();
    const edges = new Map<string, SymbolEdge[]>();
    const fileSymbolMap = new Map<string, string[]>();
    const typeHierarchy = new Map<string, string[]>();
    const importGraph = new Map<string, Set<string>>();
    
    // Phase 1: Build nodes from LSP hints
    for (let i = 0; i < lspHints.length; i++) {
      const hint = lspHints[i];
      const node = this.createSymbolNode(hint);
      nodes.set(node.id, node);
      
      // Update file-symbol mapping
      if (!fileSymbolMap.has(node.file_path)) {
        fileSymbolMap.set(node.file_path, []);
      }
      fileSymbolMap.get(node.file_path)!.push(node.id);
      
      if (progressCallback) {
        progressCallback((i / lspHints.length) * 0.3);
      }
    }

    // Phase 2: Build edges and relationships
    await this.buildEdgesFromNodes(nodes, edges, typeHierarchy, importGraph, progressCallback);
    
    // Phase 3: Compute coverage statistics
    const coverageStats = this.computeCoverageStats(nodes, fileSymbolMap);

    const snapshot: SymbolGraphSnapshot = {
      repo_sha: repoSha,
      version: '1.0.0',
      timestamp: Date.now(),
      nodes,
      edges,
      file_symbol_map: fileSymbolMap,
      type_hierarchy: typeHierarchy,
      import_graph: importGraph,
      metadata: {
        total_nodes: nodes.size,
        total_edges: Array.from(edges.values()).reduce((sum, edgeList) => sum + edgeList.length, 0),
        coverage_stats: coverageStats,
        build_duration_ms: Date.now() - startTime
      }
    };

    // Store snapshot
    await this.saveSnapshot(snapshot);
    this.snapshot = snapshot;
    
    if (progressCallback) {
      progressCallback(1.0);
    }

    return snapshot;
  }

  private createSymbolNode(hint: LSPHint): SymbolNode {
    return {
      id: `${hint.file_path}:${hint.line}:${hint.col}:${hint.name}`,
      name: hint.name,
      kind: hint.kind,
      file_path: hint.file_path,
      line: hint.line,
      col: hint.col,
      byte_offset: this.estimateByteOffset(hint.file_path, hint.line, hint.col),
      span_len: hint.signature?.length || hint.name.length,
      signature: hint.signature,
      type_info: hint.type_info,
      aliases: hint.aliases,
      resolved_imports: hint.resolved_imports,
      references_count: hint.references_count
    };
  }

  private estimateByteOffset(filePath: string, line: number, col: number): number {
    // Rough estimation: 80 chars per line on average
    return (line - 1) * 80 + col;
  }

  private async buildEdgesFromNodes(
    nodes: Map<string, SymbolNode>,
    edges: Map<string, SymbolEdge[]>,
    typeHierarchy: Map<string, string[]>,
    importGraph: Map<string, Set<string>>,
    progressCallback?: (progress: number) => void
  ): Promise<void> {
    const nodeArray = Array.from(nodes.values());
    let processed = 0;

    for (const node of nodeArray) {
      const nodeEdges: SymbolEdge[] = [];
      
      // Build definition edges (symbol -> definition)
      if (this.buildOptions.include_references) {
        // Find references to this symbol
        const references = this.findReferences(node, nodes);
        for (const ref of references) {
          nodeEdges.push({
            source_id: node.id,
            target_id: ref.id,
            edge_type: 'reference',
            weight: this.calculateReferenceWeight(node, ref),
          });
        }
      }

      // Build type relation edges
      if (this.buildOptions.include_type_relations) {
        this.buildTypeRelationEdges(node, nodes, nodeEdges, typeHierarchy);
      }

      // Build alias edges
      this.buildAliasEdges(node, nodes, nodeEdges);

      // Build import edges
      this.buildImportEdges(node, nodes, nodeEdges, importGraph);

      if (nodeEdges.length > 0) {
        edges.set(node.id, nodeEdges);
      }

      processed++;
      if (progressCallback) {
        progressCallback(0.3 + (processed / nodeArray.length) * 0.4);
      }
    }
  }

  private findReferences(symbol: SymbolNode, nodes: Map<string, SymbolNode>): SymbolNode[] {
    const references: SymbolNode[] = [];
    
    for (const node of nodes.values()) {
      if (node.id !== symbol.id && 
          this.couldBeReference(symbol, node)) {
        references.push(node);
        
        if (references.length >= this.buildOptions.max_references_per_symbol) {
          break;
        }
      }
    }
    
    return references;
  }

  private couldBeReference(symbol: SymbolNode, candidate: SymbolNode): boolean {
    // Simple heuristics - in real implementation would use LSP references
    return symbol.name === candidate.name && 
           symbol.file_path !== candidate.file_path;
  }

  private calculateReferenceWeight(symbol: SymbolNode, reference: SymbolNode): number {
    // Weight based on proximity and context similarity
    let weight = 1.0;
    
    // Same file gets higher weight
    if (symbol.file_path === reference.file_path) {
      weight *= 2.0;
    }
    
    // Same symbol kind gets higher weight
    if (symbol.kind === reference.kind) {
      weight *= 1.5;
    }
    
    return weight;
  }

  private buildTypeRelationEdges(
    node: SymbolNode,
    nodes: Map<string, SymbolNode>,
    nodeEdges: SymbolEdge[],
    typeHierarchy: Map<string, string[]>
  ): void {
    if (node.type_info) {
      // Extract type information and build relationships
      const typeNames = this.extractTypeNames(node.type_info);
      
      for (const typeName of typeNames) {
        // Find nodes that define this type
        const typeDefinitions = this.findTypeDefinitions(typeName, nodes);
        
        for (const typeDef of typeDefinitions) {
          nodeEdges.push({
            source_id: node.id,
            target_id: typeDef.id,
            edge_type: 'type_relation',
            weight: 1.0,
            metadata: { type_name: typeName }
          });
        }
        
        // Update type hierarchy
        if (!typeHierarchy.has(typeName)) {
          typeHierarchy.set(typeName, []);
        }
        typeHierarchy.get(typeName)!.push(node.id);
      }
    }
  }

  private buildAliasEdges(
    node: SymbolNode,
    nodes: Map<string, SymbolNode>,
    nodeEdges: SymbolEdge[]
  ): void {
    for (const alias of node.aliases) {
      const aliasNodes = this.findSymbolsByName(alias, nodes);
      
      for (const aliasNode of aliasNodes) {
        nodeEdges.push({
          source_id: node.id,
          target_id: aliasNode.id,
          edge_type: 'alias',
          weight: 0.9, // Aliases are strong relationships
          metadata: { alias_name: alias }
        });
      }
    }
  }

  private buildImportEdges(
    node: SymbolNode,
    nodes: Map<string, SymbolNode>,
    nodeEdges: SymbolEdge[],
    importGraph: Map<string, Set<string>>
  ): void {
    for (const importPath of node.resolved_imports) {
      if (!importGraph.has(node.file_path)) {
        importGraph.set(node.file_path, new Set());
      }
      importGraph.get(node.file_path)!.add(importPath);
      
      // Find symbols in imported file
      const importedSymbols = this.findSymbolsInFile(importPath, nodes);
      
      for (const importedSymbol of importedSymbols) {
        nodeEdges.push({
          source_id: node.id,
          target_id: importedSymbol.id,
          edge_type: 'import',
          weight: 0.7,
          metadata: { import_path: importPath }
        });
      }
    }
  }

  private extractTypeNames(typeInfo: string): string[] {
    // Simple regex-based type extraction
    const typeMatches = typeInfo.match(/\b[A-Z][a-zA-Z0-9_]*\b/g) || [];
    return [...new Set(typeMatches)]; // Remove duplicates
  }

  private findTypeDefinitions(typeName: string, nodes: Map<string, SymbolNode>): SymbolNode[] {
    return Array.from(nodes.values()).filter(node => 
      (node.kind === 'type' || node.kind === 'class' || node.kind === 'interface') &&
      node.name === typeName
    );
  }

  private findSymbolsByName(name: string, nodes: Map<string, SymbolNode>): SymbolNode[] {
    return Array.from(nodes.values()).filter(node => node.name === name);
  }

  private findSymbolsInFile(filePath: string, nodes: Map<string, SymbolNode>): SymbolNode[] {
    return Array.from(nodes.values()).filter(node => node.file_path === filePath);
  }

  private computeCoverageStats(
    nodes: Map<string, SymbolNode>,
    fileSymbolMap: Map<string, string[]>
  ): SymbolGraphSnapshot['metadata']['coverage_stats'] {
    const filesWithSymbols = fileSymbolMap.size;
    const totalFiles = new Set(Array.from(nodes.values()).map(n => n.file_path)).size;
    
    return {
      files_with_symbols: filesWithSymbols,
      total_files: totalFiles,
      symbol_coverage: totalFiles > 0 ? filesWithSymbols / totalFiles : 0
    };
  }

  /**
   * Save snapshot to msgpack format
   */
  private async saveSnapshot(snapshot: SymbolGraphSnapshot): Promise<void> {
    const filePath = path.join(this.storagePath, `SymbolGraph-${snapshot.repo_sha}.msgpack`);
    
    // Convert Maps to objects for serialization
    const serializable = {
      ...snapshot,
      nodes: Object.fromEntries(snapshot.nodes),
      edges: Object.fromEntries(snapshot.edges),
      file_symbol_map: Object.fromEntries(snapshot.file_symbol_map),
      type_hierarchy: Object.fromEntries(snapshot.type_hierarchy),
      import_graph: Object.fromEntries(
        Array.from(snapshot.import_graph.entries()).map(([k, v]) => [k, Array.from(v)])
      )
    };
    
    const buffer = msgpack.encode(serializable);
    await fs.writeFile(filePath, buffer);
  }

  /**
   * Load snapshot from msgpack format
   */
  async loadSnapshot(repoSha: string): Promise<SymbolGraphSnapshot> {
    const filePath = path.join(this.storagePath, `SymbolGraph-${repoSha}.msgpack`);
    
    try {
      const buffer = await fs.readFile(filePath);
      const data = msgpack.decode(buffer);
      
      // Convert objects back to Maps
      const snapshot: SymbolGraphSnapshot = {
        ...data,
        nodes: new Map(Object.entries(data.nodes)),
        edges: new Map(Object.entries(data.edges)),
        file_symbol_map: new Map(Object.entries(data.file_symbol_map)),
        type_hierarchy: new Map(Object.entries(data.type_hierarchy)),
        import_graph: new Map(
          Object.entries(data.import_graph).map(([k, v]: [string, any]) => [k, new Set(v)])
        )
      };
      
      this.snapshot = snapshot;
      return snapshot;
      
    } catch (error) {
      throw new Error(`Failed to load SymbolGraph snapshot: ${error}`);
    }
  }

  // Query methods for efficient graph traversal
  
  getSymbolsByFile(filePath: string): SymbolNode[] {
    if (!this.snapshot) return [];
    const symbolIds = this.snapshot.file_symbol_map.get(filePath) || [];
    return symbolIds.map(id => this.snapshot!.nodes.get(id)!).filter(Boolean);
  }

  getSymbolReferences(symbolId: string): SymbolNode[] {
    if (!this.snapshot) return [];
    const edges = this.snapshot.edges.get(symbolId) || [];
    const referenceEdges = edges.filter(e => e.edge_type === 'reference');
    
    return referenceEdges.map(e => this.snapshot!.nodes.get(e.target_id)!).filter(Boolean);
  }

  getSymbolDefinitions(symbolName: string): SymbolNode[] {
    if (!this.snapshot) return [];
    return Array.from(this.snapshot.nodes.values()).filter(node => 
      node.name === symbolName && 
      ['function', 'class', 'type', 'interface'].includes(node.kind)
    );
  }

  getRelatedSymbols(symbolId: string, relationTypes?: SymbolEdgeType[]): SymbolNode[] {
    if (!this.snapshot) return [];
    const edges = this.snapshot.edges.get(symbolId) || [];
    
    const filteredEdges = relationTypes 
      ? edges.filter(e => relationTypes.includes(e.edge_type))
      : edges;
    
    return filteredEdges.map(e => this.snapshot!.nodes.get(e.target_id)!).filter(Boolean);
  }

  getTypeHierarchy(typeName: string): string[] {
    if (!this.snapshot) return [];
    return this.snapshot.type_hierarchy.get(typeName) || [];
  }

  getImportGraph(filePath: string): Set<string> {
    if (!this.snapshot) return new Set();
    return this.snapshot.import_graph.get(filePath) || new Set();
  }

  // Advanced queries for semantic enhancement

  /**
   * Find symbols that could be semantically related to the query
   */
  findSemanticCandidates(query: string, maxResults: number = 50): SymbolNode[] {
    if (!this.snapshot) return [];
    
    const queryTokens = query.toLowerCase().split(/\s+/);
    const candidates: Array<{node: SymbolNode, score: number}> = [];
    
    for (const node of this.snapshot.nodes.values()) {
      let score = 0;
      
      // Direct name match
      if (node.name.toLowerCase().includes(query.toLowerCase())) {
        score += 10;
      }
      
      // Token matching in name
      for (const token of queryTokens) {
        if (node.name.toLowerCase().includes(token)) {
          score += 5;
        }
      }
      
      // Signature matching
      if (node.signature?.toLowerCase().includes(query.toLowerCase())) {
        score += 3;
      }
      
      // Type info matching
      if (node.type_info?.toLowerCase().includes(query.toLowerCase())) {
        score += 2;
      }
      
      if (score > 0) {
        candidates.push({node, score});
      }
    }
    
    candidates.sort((a, b) => b.score - a.score);
    return candidates.slice(0, maxResults).map(c => c.node);
  }

  /**
   * Compute graph-based features for symbol nodes
   */
  computeGraphFeatures(symbolId: string): Record<string, number> {
    if (!this.snapshot) return {};
    
    const node = this.snapshot.nodes.get(symbolId);
    if (!node) return {};
    
    const edges = this.snapshot.edges.get(symbolId) || [];
    const references = edges.filter(e => e.edge_type === 'reference');
    const typeRelations = edges.filter(e => e.edge_type === 'type_relation');
    const aliases = edges.filter(e => e.edge_type === 'alias');
    
    return {
      reference_count: references.length,
      type_relation_count: typeRelations.length,
      alias_count: aliases.length,
      total_edge_weight: edges.reduce((sum, e) => sum + e.weight, 0),
      avg_edge_weight: edges.length > 0 ? edges.reduce((sum, e) => sum + e.weight, 0) / edges.length : 0,
      symbol_centrality: this.computeCentrality(symbolId),
    };
  }

  private computeCentrality(symbolId: string): number {
    if (!this.snapshot) return 0;
    
    // Simple degree centrality
    const outEdges = this.snapshot.edges.get(symbolId)?.length || 0;
    let inEdges = 0;
    
    for (const edges of this.snapshot.edges.values()) {
      for (const edge of edges) {
        if (edge.target_id === symbolId) {
          inEdges++;
        }
      }
    }
    
    return (outEdges + inEdges) / Math.max(1, this.snapshot.nodes.size);
  }

  getSnapshot(): SymbolGraphSnapshot | undefined {
    return this.snapshot;
  }

  getStats(): SymbolGraphSnapshot['metadata'] | undefined {
    return this.snapshot?.metadata;
  }
}