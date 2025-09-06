/**
 * RAPTOR Snapshot Types and Management
 * 
 * Defines the hierarchical RAPTOR tree structure with nodes, snapshots,
 * and persistence layer for the semantic clustering system.
 */

import { SemanticCard } from './semantic-card.js';

export interface RaptorNode {
  id: string;
  level: number;
  parent?: string;
  children: string[];
  centroid: number[];              // Mean embedding of all cards in subtree
  summary: string;                 // ≤120 tokens, bullet list format
  summaryEmb: number[];           // Embedding of the summary text
  file_ids: string[];             // Files contained in this node's subtree
  last_update_ts: number;         // Unix timestamp of last update
  pressure: number;               // Current pressure score for reclustering
}

export interface RaptorSnapshot {
  repo_sha: string;
  version: string;                // UUID-based version identifier
  created_ts: number;
  nodes: Map<string, RaptorNode>;
  fileToLeaf: Map<string, string[]>; // file_id -> leaf node IDs containing it
  config: RaptorSnapshotConfig;
}

export interface RaptorSnapshotConfig {
  embedding_dim: number;
  max_levels: number;             // Typically 2-3
  k1: number;                     // Root level clusters
  k2: number;                     // Second level clusters per parent
  summary_max_tokens: number;     // 120
  languages: string[];           // ["py", "ts", "js"]
  min_files_per_node: number;    // Minimum files to form a cluster
}

export interface RaptorSnapshotMetadata {
  repo_sha: string;
  version: string;
  created_ts: number;
  file_count: number;
  node_count: number;
  levels: number;
  staleness_hours: number;
  config_fingerprint: string;
}

export interface RaptorIndex {
  version: string;
  node_embeddings: number[][];    // Array of summary embeddings
  node_ids: string[];             // Corresponding node IDs
  index_type: "faiss" | "hnsw" | "simple";
  created_ts: number;
}

export interface PressureMetrics {
  editChurn: number;              // EWMA bytes changed under node
  embeddingDrift: number;         // ||Δe_sem||_2 since last stable
  queryHeat: number;              // Recent routed hits
  age: number;                    // Hours since last update
}

export interface PressureWeights {
  wc: number;  // editChurn weight (default 0.4)
  wd: number;  // embeddingDrift weight (default 0.3)
  wq: number;  // queryHeat weight (default 0.2)
  wa: number;  // age weight (default 0.1)
}

export interface ReclusterBudget {
  max_summaries_per_hour: number;  // 200
  max_cpu_seconds_per_hour: number; // 300
  current_summaries_used: number;
  current_cpu_used: number;
  reset_ts: number; // When budget resets
}

export interface StalenessReport {
  total_nodes: number;
  stale_nodes: number;  // > TTL
  stale_percentage: number;
  ttl_days: number;
  staleness_cdf: number[];  // Percentiles [50, 90, 95, 99]
}

/**
 * RAPTOR Snapshot Manager
 */
export class RaptorSnapshotManager {
  constructor(
    private storagePath: string,
    private ttlDays: number = 14
  ) {}

  static createDefaultConfig(): RaptorSnapshotConfig {
    return {
      embedding_dim: 384,  // Typical sentence transformer dimension
      max_levels: 3,
      k1: 0, // Computed dynamically
      k2: 0, // Computed dynamically
      summary_max_tokens: 120,
      languages: ["py", "ts", "js"],
      min_files_per_node: 2
    };
  }

  static computeClusterParams(fileCount: number): { k1: number; k2: number } {
    // Target 50-200 leaves; K1=√N, K2=√(N/K1)
    const k1 = Math.max(4, Math.min(16, Math.sqrt(fileCount)));
    const k2 = Math.max(4, Math.min(16, Math.sqrt(fileCount / k1)));
    return { 
      k1: Math.floor(k1), 
      k2: Math.floor(k2) 
    };
  }

  static calculatePressure(
    node: RaptorNode, 
    metrics: PressureMetrics, 
    weights: PressureWeights = { wc: 0.4, wd: 0.3, wq: 0.2, wa: 0.1 }
  ): number {
    return (
      weights.wc * metrics.editChurn +
      weights.wd * metrics.embeddingDrift +
      weights.wq * metrics.queryHeat +
      weights.wa * metrics.age
    );
  }

  generateVersion(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  getSnapshotPath(repoSha: string, version: string): string {
    return `${this.storagePath}/raptor_snapshots/${repoSha}/${version}`;
  }

  getMetadataPath(repoSha: string, version: string): string {
    return `${this.getSnapshotPath(repoSha, version)}/metadata.json`;
  }

  getSnapshotFilePath(repoSha: string, version: string): string {
    return `${this.getSnapshotPath(repoSha, version)}/snapshot.json.gz`;
  }

  getIndexPath(repoSha: string, version: string): string {
    return `${this.getSnapshotPath(repoSha, version)}/summary.index`;
  }

  validateSnapshot(snapshot: RaptorSnapshot): { isValid: boolean; errors: string[] } {
    const errors: string[] = [];

    if (!snapshot.repo_sha || !snapshot.version) {
      errors.push("Missing repo_sha or version");
    }

    if (!snapshot.nodes || snapshot.nodes.size === 0) {
      errors.push("Snapshot must contain at least one node");
    }

    if (!snapshot.fileToLeaf || snapshot.fileToLeaf.size === 0) {
      errors.push("fileToLeaf mapping is required");
    }

    // Validate node hierarchy
    const nodeIds = new Set(snapshot.nodes.keys());
    for (const [nodeId, node] of snapshot.nodes) {
      if (node.parent && !nodeIds.has(node.parent)) {
        errors.push(`Node ${nodeId} references non-existent parent ${node.parent}`);
      }

      for (const childId of node.children) {
        if (!nodeIds.has(childId)) {
          errors.push(`Node ${nodeId} references non-existent child ${childId}`);
        }
      }

      if (node.level < 0 || node.level > snapshot.config.max_levels) {
        errors.push(`Node ${nodeId} has invalid level ${node.level}`);
      }

      if (!node.summary || node.summary.length === 0) {
        errors.push(`Node ${nodeId} missing summary`);
      }
    }

    return { isValid: errors.length === 0, errors };
  }

  computeStaleness(snapshot: RaptorSnapshot, currentTs: number = Date.now()): StalenessReport {
    const ttlMs = this.ttlDays * 24 * 60 * 60 * 1000;
    const nodes = Array.from(snapshot.nodes.values());
    
    const staleNodes = nodes.filter(node => {
      return (currentTs - node.last_update_ts) > ttlMs;
    });

    const ages = nodes.map(node => (currentTs - node.last_update_ts) / (60 * 60 * 1000)); // Hours
    ages.sort((a, b) => a - b);

    const percentiles = [50, 90, 95, 99].map(p => {
      const index = Math.floor((p / 100) * ages.length);
      return ages[Math.min(index, ages.length - 1)];
    });

    return {
      total_nodes: nodes.length,
      stale_nodes: staleNodes.length,
      stale_percentage: (staleNodes.length / nodes.length) * 100,
      ttl_days: this.ttlDays,
      staleness_cdf: percentiles
    };
  }

  createEmptySnapshot(repoSha: string, config: RaptorSnapshotConfig): RaptorSnapshot {
    return {
      repo_sha: repoSha,
      version: this.generateVersion(),
      created_ts: Date.now(),
      nodes: new Map(),
      fileToLeaf: new Map(),
      config
    };
  }

  findRootNodes(snapshot: RaptorSnapshot): RaptorNode[] {
    return Array.from(snapshot.nodes.values()).filter(node => !node.parent);
  }

  findLeafNodes(snapshot: RaptorSnapshot): RaptorNode[] {
    return Array.from(snapshot.nodes.values()).filter(node => node.children.length === 0);
  }

  getAncestors(snapshot: RaptorSnapshot, nodeId: string): RaptorNode[] {
    const ancestors: RaptorNode[] = [];
    let current = snapshot.nodes.get(nodeId);
    
    while (current && current.parent) {
      const parent = snapshot.nodes.get(current.parent);
      if (parent) {
        ancestors.push(parent);
        current = parent;
      } else {
        break;
      }
    }
    
    return ancestors;
  }

  getSubtreeFiles(snapshot: RaptorSnapshot, nodeId: string): Set<string> {
    const files = new Set<string>();
    const stack = [nodeId];
    
    while (stack.length > 0) {
      const currentId = stack.pop()!;
      const node = snapshot.nodes.get(currentId);
      
      if (node) {
        // Add files from this node
        for (const fileId of node.file_ids) {
          files.add(fileId);
        }
        
        // Add children to stack for traversal
        stack.push(...node.children);
      }
    }
    
    return files;
  }

  cloneSnapshot(snapshot: RaptorSnapshot): RaptorSnapshot {
    return {
      ...snapshot,
      nodes: new Map(Array.from(snapshot.nodes.entries()).map(([id, node]) => [
        id,
        { ...node, children: [...node.children], file_ids: [...node.file_ids] }
      ])),
      fileToLeaf: new Map(snapshot.fileToLeaf),
      config: { ...snapshot.config }
    };
  }
}

export default RaptorSnapshot;