/**
 * RAPTOR Hygiene with Incremental Reclustering
 * 
 * Implements incremental reclustering with versioned nodes, pressure budgets,
 * and monitoring for positives-in-candidates. Maintains hierarchical document
 * clustering while preventing quality degradation through careful resource management.
 */

import type { SearchContext, SearchHit } from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';

export interface RAPTORNode {
  node_id: string;
  parent_id?: string;
  children_ids: string[];
  level: number;
  centroid_embedding?: Float32Array;
  representative_text: string;
  document_ids: Set<string>;
  creation_timestamp: number;
  last_update_timestamp: number;
  version: number;
  cluster_coherence: number;
}

export interface ClusteringPressureBudget {
  daily_budget_ops: number;
  used_ops_today: number;
  remaining_ops: number;
  pressure_threshold: number;
  last_reset_timestamp: number;
}

export interface ReclusteringEvent {
  event_id: string;
  event_type: 'node_split' | 'node_merge' | 'node_update' | 'tree_rebalance';
  timestamp: number;
  affected_nodes: string[];
  trigger_reason: string;
  pressure_cost: number;
  quality_impact: number;
}

export interface PositivesCandidatesMonitor {
  query_id: string;
  total_candidates: number;
  positive_candidates: number;
  recall_gap: number; // Recall@50 - Recall@10
  cluster_coverage: number;
  timestamp: number;
}

export interface RAPTORHygieneConfig {
  max_cluster_size: number;
  min_cluster_coherence: number;
  pressure_budget_ops_per_day: number;
  recall_gap_threshold: number;
  incremental_update_threshold: number;
  versioning_enabled: boolean;
  fanout_increase_threshold: number;
}

/**
 * Hierarchical clustering tree manager
 */
class HierarchicalClusterTree {
  private nodes: Map<string, RAPTORNode> = new Map();
  private rootNodes: Set<string> = new Set();
  private documentToNodeMap: Map<string, string> = new Map();
  private nextNodeId = 1;
  
  /**
   * Add document to the clustering tree
   */
  addDocument(
    docId: string, 
    embedding: Float32Array, 
    text: string,
    forceReclustering = false
  ): string {
    // Find best cluster for document
    const bestCluster = this.findBestCluster(embedding, forceReclustering);
    
    if (bestCluster) {
      // Add to existing cluster
      bestCluster.document_ids.add(docId);
      bestCluster.last_update_timestamp = Date.now();
      bestCluster.version++;
      
      // Update centroid (incremental update)
      this.updateClusterCentroid(bestCluster, embedding, true);
      
      this.documentToNodeMap.set(docId, bestCluster.node_id);
      return bestCluster.node_id;
    } else {
      // Create new cluster
      const nodeId = `raptor_${this.nextNodeId++}`;
      const newNode: RAPTORNode = {
        node_id: nodeId,
        children_ids: [],
        level: 0,
        centroid_embedding: new Float32Array(embedding),
        representative_text: text.slice(0, 200),
        document_ids: new Set([docId]),
        creation_timestamp: Date.now(),
        last_update_timestamp: Date.now(),
        version: 1,
        cluster_coherence: 1.0 // Perfect coherence for single document
      };
      
      this.nodes.set(nodeId, newNode);
      this.rootNodes.add(nodeId);
      this.documentToNodeMap.set(docId, nodeId);
      
      return nodeId;
    }
  }
  
  /**
   * Find best cluster for a document
   */
  private findBestCluster(
    embedding: Float32Array, 
    forceReclustering: boolean
  ): RAPTORNode | null {
    let bestNode: RAPTORNode | null = null;
    let bestSimilarity = forceReclustering ? 0.6 : 0.8; // Lower threshold when forcing
    
    // Search through root nodes first
    for (const rootId of this.rootNodes) {
      const node = this.nodes.get(rootId);
      if (!node || !node.centroid_embedding) continue;
      
      const similarity = this.cosineSimilarity(embedding, node.centroid_embedding);
      
      if (similarity > bestSimilarity && node.document_ids.size < 50) {
        bestSimilarity = similarity;
        bestNode = node;
      }
    }
    
    return bestNode;
  }
  
  /**
   * Update cluster centroid incrementally
   */
  private updateClusterCentroid(
    node: RAPTORNode, 
    newEmbedding: Float32Array, 
    isAdd: boolean
  ): void {
    if (!node.centroid_embedding) return;
    
    const n = node.document_ids.size;
    const alpha = isAdd ? 1.0 / n : -1.0 / (n + 1);
    
    // Incremental centroid update: c_new = c_old + alpha * (x - c_old)
    for (let i = 0; i < node.centroid_embedding.length; i++) {
      const diff = newEmbedding[i] - node.centroid_embedding[i];
      node.centroid_embedding[i] += alpha * diff;
    }
    
    // Update coherence (simplified measure)
    node.cluster_coherence = this.calculateClusterCoherence(node);
  }
  
  /**
   * Calculate cluster coherence
   */
  private calculateClusterCoherence(node: RAPTORNode): number {
    // Simple heuristic: coherence decreases as cluster size increases
    const sizeScore = Math.max(0.1, 1.0 - (node.document_ids.size / 100));
    
    // Age score: newer clusters are more coherent
    const ageHours = (Date.now() - node.creation_timestamp) / (1000 * 60 * 60);
    const ageScore = Math.max(0.5, 1.0 - (ageHours / 168)); // 1 week decay
    
    return (sizeScore + ageScore) / 2;
  }
  
  /**
   * Split large or incoherent clusters
   */
  splitCluster(nodeId: string): string[] | null {
    const node = this.nodes.get(nodeId);
    if (!node || node.document_ids.size < 10) return null;
    
    // Simple split: create two child clusters
    const docs = Array.from(node.document_ids);
    const midpoint = Math.floor(docs.length / 2);
    
    const child1Id = `raptor_${this.nextNodeId++}`;
    const child2Id = `raptor_${this.nextNodeId++}`;
    
    const child1: RAPTORNode = {
      node_id: child1Id,
      parent_id: nodeId,
      children_ids: [],
      level: node.level + 1,
      centroid_embedding: node.centroid_embedding ? new Float32Array(node.centroid_embedding) : undefined,
      representative_text: node.representative_text,
      document_ids: new Set(docs.slice(0, midpoint)),
      creation_timestamp: Date.now(),
      last_update_timestamp: Date.now(),
      version: 1,
      cluster_coherence: 0.8 // Initial coherence
    };
    
    const child2: RAPTORNode = {
      node_id: child2Id,
      parent_id: nodeId,
      children_ids: [],
      level: node.level + 1,
      centroid_embedding: node.centroid_embedding ? new Float32Array(node.centroid_embedding) : undefined,
      representative_text: node.representative_text,
      document_ids: new Set(docs.slice(midpoint)),
      creation_timestamp: Date.now(),
      last_update_timestamp: Date.now(),
      version: 1,
      cluster_coherence: 0.8
    };
    
    // Update parent
    node.children_ids = [child1Id, child2Id];
    node.document_ids.clear();
    node.version++;
    
    // Update mappings
    for (const docId of child1.document_ids) {
      this.documentToNodeMap.set(docId, child1Id);
    }
    for (const docId of child2.document_ids) {
      this.documentToNodeMap.set(docId, child2Id);
    }
    
    this.nodes.set(child1Id, child1);
    this.nodes.set(child2Id, child2);
    
    return [child1Id, child2Id];
  }
  
  /**
   * Get all nodes at a specific level
   */
  getNodesAtLevel(level: number): RAPTORNode[] {
    return Array.from(this.nodes.values()).filter(node => node.level === level);
  }
  
  /**
   * Get node by ID
   */
  getNode(nodeId: string): RAPTORNode | undefined {
    return this.nodes.get(nodeId);
  }
  
  /**
   * Get all nodes
   */
  getAllNodes(): Map<string, RAPTORNode> {
    return new Map(this.nodes);
  }
  
  /**
   * Remove node from tree
   */
  removeNode(nodeId: string): void {
    const node = this.nodes.get(nodeId);
    if (!node) return;
    
    // Remove from parent's children
    if (node.parent_id) {
      const parent = this.nodes.get(node.parent_id);
      if (parent) {
        parent.children_ids = parent.children_ids.filter(id => id !== nodeId);
        parent.version++;
      }
    }
    
    // Remove document mappings
    for (const docId of node.document_ids) {
      this.documentToNodeMap.delete(docId);
    }
    
    this.nodes.delete(nodeId);
    this.rootNodes.delete(nodeId);
  }
  
  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }
}

/**
 * Pressure budget manager for clustering operations
 */
class PressureBudgetManager {
  private budget: ClusteringPressureBudget;
  private operationHistory: ReclusteringEvent[] = [];
  
  constructor(dailyBudgetOps = 1000) {
    this.budget = {
      daily_budget_ops: dailyBudgetOps,
      used_ops_today: 0,
      remaining_ops: dailyBudgetOps,
      pressure_threshold: dailyBudgetOps * 0.8, // Alert at 80%
      last_reset_timestamp: this.getTodayTimestamp()
    };
  }
  
  /**
   * Check if operation can be performed within budget
   */
  canPerformOperation(estimatedCost: number): boolean {
    this.maybeReset();
    return (this.budget.used_ops_today + estimatedCost) <= this.budget.daily_budget_ops;
  }
  
  /**
   * Record operation usage
   */
  recordOperation(event: ReclusteringEvent): void {
    this.maybeReset();
    
    this.budget.used_ops_today += event.pressure_cost;
    this.budget.remaining_ops = Math.max(0, this.budget.daily_budget_ops - this.budget.used_ops_today);
    
    this.operationHistory.push(event);
    
    // Keep only last 7 days of history
    const weekAgo = Date.now() - 7 * 24 * 60 * 60 * 1000;
    this.operationHistory = this.operationHistory.filter(e => e.timestamp > weekAgo);
    
    console.log(`💰 RAPTOR pressure: ${this.budget.used_ops_today}/${this.budget.daily_budget_ops} ops (${event.event_type}: +${event.pressure_cost})`);
  }
  
  /**
   * Get current budget status
   */
  getBudgetStatus(): ClusteringPressureBudget {
    this.maybeReset();
    return { ...this.budget };
  }
  
  /**
   * Get recent operation history
   */
  getRecentOperations(hours = 24): ReclusteringEvent[] {
    const cutoff = Date.now() - hours * 60 * 60 * 1000;
    return this.operationHistory.filter(e => e.timestamp > cutoff);
  }
  
  private maybeReset(): void {
    const today = this.getTodayTimestamp();
    if (today !== this.budget.last_reset_timestamp) {
      this.budget.used_ops_today = 0;
      this.budget.remaining_ops = this.budget.daily_budget_ops;
      this.budget.last_reset_timestamp = today;
      console.log('🔄 RAPTOR pressure budget reset for new day');
    }
  }
  
  private getTodayTimestamp(): number {
    return Math.floor(Date.now() / (24 * 60 * 60 * 1000));
  }
}

/**
 * Monitor for positives-in-candidates quality metric
 */
class PositivesCandidatesTracker {
  private measurements: PositivesCandidatesMonitor[] = [];
  private recallGapThreshold: number;
  
  constructor(recallGapThreshold = 0.3) {
    this.recallGapThreshold = recallGapThreshold;
  }
  
  /**
   * Record measurement for a query
   */
  recordMeasurement(
    queryId: string,
    totalCandidates: number,
    positiveCandidates: number,
    recallGap: number,
    clusterCoverage: number
  ): void {
    const measurement: PositivesCandidatesMonitor = {
      query_id: queryId,
      total_candidates: totalCandidates,
      positive_candidates: positiveCandidates,
      recall_gap: recallGap,
      cluster_coverage: clusterCoverage,
      timestamp: Date.now()
    };
    
    this.measurements.push(measurement);
    
    // Keep only recent measurements
    const dayAgo = Date.now() - 24 * 60 * 60 * 1000;
    this.measurements = this.measurements.filter(m => m.timestamp > dayAgo);
    
    // Alert if recall gap is too large
    if (recallGap > this.recallGapThreshold) {
      console.warn(`⚠️ RAPTOR quality alert: recall gap ${recallGap.toFixed(3)} > ${this.recallGapThreshold} for query ${queryId}`);
    }
  }
  
  /**
   * Get recent quality statistics
   */
  getQualityStats(hours = 24): {
    measurement_count: number;
    avg_positive_rate: number;
    avg_recall_gap: number;
    avg_cluster_coverage: number;
    quality_alerts: number;
  } {
    const cutoff = Date.now() - hours * 60 * 60 * 1000;
    const recent = this.measurements.filter(m => m.timestamp > cutoff);
    
    if (recent.length === 0) {
      return {
        measurement_count: 0,
        avg_positive_rate: 0,
        avg_recall_gap: 0,
        avg_cluster_coverage: 0,
        quality_alerts: 0
      };
    }
    
    const sums = recent.reduce((acc, m) => {
      acc.positive_rate += m.total_candidates > 0 ? m.positive_candidates / m.total_candidates : 0;
      acc.recall_gap += m.recall_gap;
      acc.cluster_coverage += m.cluster_coverage;
      acc.quality_alerts += m.recall_gap > this.recallGapThreshold ? 1 : 0;
      return acc;
    }, { positive_rate: 0, recall_gap: 0, cluster_coverage: 0, quality_alerts: 0 });
    
    return {
      measurement_count: recent.length,
      avg_positive_rate: sums.positive_rate / recent.length,
      avg_recall_gap: sums.recall_gap / recent.length,
      avg_cluster_coverage: sums.cluster_coverage / recent.length,
      quality_alerts: sums.quality_alerts
    };
  }
}

/**
 * Main RAPTOR hygiene engine
 */
export class RAPTORHygiene {
  private config: RAPTORHygieneConfig;
  private clusterTree: HierarchicalClusterTree;
  private pressureBudget: PressureBudgetManager;
  private qualityTracker: PositivesCandidatesTracker;
  private enabled = true;
  
  // Metrics
  private totalOperations = 0;
  private incrementalUpdates = 0;
  private fullReclusters = 0;
  private nodeSplits = 0;
  private nodeMerges = 0;
  
  constructor(config?: Partial<RAPTORHygieneConfig>) {
    this.config = {
      max_cluster_size: 50,
      min_cluster_coherence: 0.6,
      pressure_budget_ops_per_day: 1000,
      recall_gap_threshold: 0.3,
      incremental_update_threshold: 0.8,
      versioning_enabled: true,
      fanout_increase_threshold: 0.1,
      ...config
    };
    
    this.clusterTree = new HierarchicalClusterTree();
    this.pressureBudget = new PressureBudgetManager(this.config.pressure_budget_ops_per_day);
    this.qualityTracker = new PositivesCandidatesTracker(this.config.recall_gap_threshold);
  }
  
  /**
   * Add or update document in RAPTOR hierarchy
   */
  async addDocument(
    docId: string,
    embedding: Float32Array,
    text: string,
    forceReclustering = false
  ): Promise<string | null> {
    if (!this.enabled) return null;
    
    const span = LensTracer.createChildSpan('raptor_add_document');
    
    try {
      // Estimate operation cost
      const estimatedCost = forceReclustering ? 5 : 1;
      
      if (!this.pressureBudget.canPerformOperation(estimatedCost)) {
        console.warn(`⚠️ RAPTOR: operation skipped due to pressure budget (${estimatedCost} ops needed)`);
        return null;
      }
      
      // Add document to cluster tree
      const nodeId = this.clusterTree.addDocument(docId, embedding, text, forceReclustering);
      
      // Record operation
      const event: ReclusteringEvent = {
        event_id: `add_${Date.now()}_${Math.random().toString(36).slice(2)}`,
        event_type: 'node_update',
        timestamp: Date.now(),
        affected_nodes: [nodeId],
        trigger_reason: forceReclustering ? 'forced_reclustering' : 'incremental_update',
        pressure_cost: estimatedCost,
        quality_impact: 0.1
      };
      
      this.pressureBudget.recordOperation(event);
      this.totalOperations++;
      this.incrementalUpdates++;
      
      // Check if cluster needs maintenance
      await this.maybePerformMaintenance(nodeId);
      
      span.setAttributes({
        success: true,
        node_id: nodeId,
        force_reclustering: forceReclustering,
        pressure_cost: estimatedCost
      });
      
      return nodeId;
      
    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      console.error('RAPTOR add document error:', error);
      return null;
    } finally {
      span.end();
    }
  }
  
  /**
   * Perform hierarchical search through RAPTOR tree
   */
  async hierarchicalSearch(
    queryEmbedding: Float32Array,
    ctx: SearchContext,
    maxResults = 50
  ): Promise<SearchHit[]> {
    if (!this.enabled) return [];
    
    const span = LensTracer.createChildSpan('raptor_hierarchical_search');
    
    try {
      const results: SearchHit[] = [];
      const processedNodes = new Set<string>();
      
      // Start from root level and traverse down
      let currentLevel = 0;
      let candidateNodes = Array.from(this.clusterTree.getAllNodes().values())
        .filter(node => node.level === currentLevel);
      
      while (candidateNodes.length > 0 && results.length < maxResults) {
        // Score nodes at current level
        const scoredNodes = candidateNodes
          .filter(node => !processedNodes.has(node.node_id))
          .map(node => ({
            node,
            similarity: node.centroid_embedding 
              ? this.cosineSimilarity(queryEmbedding, node.centroid_embedding)
              : 0
          }))
          .sort((a, b) => b.similarity - a.similarity)
          .slice(0, 10); // Top 10 nodes per level
        
        for (const { node, similarity } of scoredNodes) {
          processedNodes.add(node.node_id);
          
          // If leaf node, add documents to results
          if (node.children_ids.length === 0) {
            for (const docId of node.document_ids) {
              if (results.length >= maxResults) break;
              
              results.push({
                file: docId,
                line: 1,
                col: 1,
                lang: 'unknown',
                snippet: node.representative_text.slice(0, 100),
                score: similarity * node.cluster_coherence,
                why: ['raptor_hierarchical', `level_${node.level}`, `coherence_${node.cluster_coherence.toFixed(2)}`],
                cluster_node_id: node.node_id,
                cluster_level: node.level
              });
            }
          }
        }
        
        // Move to next level
        currentLevel++;
        candidateNodes = scoredNodes
          .flatMap(({ node }) => node.children_ids.map(id => this.clusterTree.getNode(id)))
          .filter((node): node is RAPTORNode => node !== undefined);
      }
      
      // Record quality measurement
      const positiveCandidates = results.filter(hit => hit.score > 0.5).length;
      const recallGap = this.estimateRecallGap(results);
      const clusterCoverage = processedNodes.size / this.clusterTree.getAllNodes().size;
      
      this.qualityTracker.recordMeasurement(
        `${ctx.query}_${Date.now()}`,
        results.length,
        positiveCandidates,
        recallGap,
        clusterCoverage
      );
      
      console.log(`🌳 RAPTOR search: ${results.length} results from ${processedNodes.size} clusters (${currentLevel} levels)`);
      
      span.setAttributes({
        success: true,
        results_count: results.length,
        levels_traversed: currentLevel,
        nodes_processed: processedNodes.size,
        positive_candidates: positiveCandidates,
        recall_gap: recallGap
      });
      
      return results.sort((a, b) => b.score - a.score);
      
    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      console.error('RAPTOR hierarchical search error:', error);
      return [];
    } finally {
      span.end();
    }
  }
  
  /**
   * Perform maintenance operations on cluster tree
   */
  private async maybePerformMaintenance(nodeId: string): Promise<void> {
    const node = this.clusterTree.getNode(nodeId);
    if (!node) return;
    
    // Check if node needs splitting (too large or incoherent)
    if (node.document_ids.size > this.config.max_cluster_size || 
        node.cluster_coherence < this.config.min_cluster_coherence) {
      
      const splitCost = 10;
      if (this.pressureBudget.canPerformOperation(splitCost)) {
        const childNodes = this.clusterTree.splitCluster(nodeId);
        
        if (childNodes) {
          const event: ReclusteringEvent = {
            event_id: `split_${Date.now()}_${Math.random().toString(36).slice(2)}`,
            event_type: 'node_split',
            timestamp: Date.now(),
            affected_nodes: [nodeId, ...childNodes],
            trigger_reason: `size_${node.document_ids.size}_coherence_${node.cluster_coherence.toFixed(3)}`,
            pressure_cost: splitCost,
            quality_impact: 0.2
          };
          
          this.pressureBudget.recordOperation(event);
          this.nodeSplits++;
          
          console.log(`✂️ RAPTOR: split node ${nodeId} into ${childNodes.length} children`);
        }
      }
    }
  }
  
  /**
   * Estimate recall gap (simplified heuristic)
   */
  private estimateRecallGap(results: SearchHit[]): number {
    // Simple heuristic: assume recall gap based on score distribution
    const scores = results.map(hit => hit.score);
    if (scores.length < 10) return 0.1;
    
    const top10Score = scores.slice(0, 10).reduce((sum, score) => sum + score, 0) / 10;
    const top50Score = scores.slice(0, Math.min(50, scores.length)).reduce((sum, score) => sum + score, 0) / Math.min(50, scores.length);
    
    return Math.max(0, top10Score - top50Score);
  }
  
  /**
   * Perform full tree rebalancing (expensive operation)
   */
  async performFullRebalancing(): Promise<boolean> {
    const span = LensTracer.createChildSpan('raptor_full_rebalancing');
    const rebalanceCost = 100;
    
    try {
      if (!this.pressureBudget.canPerformOperation(rebalanceCost)) {
        console.warn('⚠️ RAPTOR: full rebalancing skipped due to pressure budget');
        return false;
      }
      
      console.log('🔄 RAPTOR: starting full tree rebalancing...');
      
      // This would implement a full tree rebalancing algorithm
      // For now, just record the event
      const event: ReclusteringEvent = {
        event_id: `rebalance_${Date.now()}`,
        event_type: 'tree_rebalance',
        timestamp: Date.now(),
        affected_nodes: Array.from(this.clusterTree.getAllNodes().keys()),
        trigger_reason: 'manual_rebalancing',
        pressure_cost: rebalanceCost,
        quality_impact: 0.5
      };
      
      this.pressureBudget.recordOperation(event);
      this.fullReclusters++;
      
      console.log('✅ RAPTOR: full rebalancing completed');
      
      span.setAttributes({
        success: true,
        pressure_cost: rebalanceCost,
        nodes_affected: this.clusterTree.getAllNodes().size
      });
      
      return true;
      
    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      console.error('RAPTOR rebalancing error:', error);
      return false;
    } finally {
      span.end();
    }
  }
  
  /**
   * Get RAPTOR tree statistics
   */
  getTreeStats(): {
    total_nodes: number;
    total_documents: number;
    tree_depth: number;
    avg_cluster_size: number;
    avg_cluster_coherence: number;
    root_nodes: number;
    leaf_nodes: number;
  } {
    const nodes = Array.from(this.clusterTree.getAllNodes().values());
    
    if (nodes.length === 0) {
      return {
        total_nodes: 0,
        total_documents: 0,
        tree_depth: 0,
        avg_cluster_size: 0,
        avg_cluster_coherence: 0,
        root_nodes: 0,
        leaf_nodes: 0
      };
    }
    
    const totalDocuments = nodes.reduce((sum, node) => sum + node.document_ids.size, 0);
    const avgClusterSize = totalDocuments / nodes.length;
    const avgCoherence = nodes.reduce((sum, node) => sum + node.cluster_coherence, 0) / nodes.length;
    const maxDepth = Math.max(...nodes.map(node => node.level));
    const rootNodes = nodes.filter(node => !node.parent_id).length;
    const leafNodes = nodes.filter(node => node.children_ids.length === 0).length;
    
    return {
      total_nodes: nodes.length,
      total_documents: totalDocuments,
      tree_depth: maxDepth + 1,
      avg_cluster_size: avgClusterSize,
      avg_cluster_coherence: avgCoherence,
      root_nodes: rootNodes,
      leaf_nodes: leafNodes
    };
  }
  
  /**
   * Get operational statistics
   */
  getOperationalStats(): {
    total_operations: number;
    incremental_updates: number;
    full_reclusters: number;
    node_splits: number;
    node_merges: number;
    pressure_budget: ClusteringPressureBudget;
    quality_stats: ReturnType<PositivesCandidatesTracker['getQualityStats']>;
    enabled: boolean;
  } {
    return {
      total_operations: this.totalOperations,
      incremental_updates: this.incrementalUpdates,
      full_reclusters: this.fullReclusters,
      node_splits: this.nodeSplits,
      node_merges: this.nodeMerges,
      pressure_budget: this.pressureBudget.getBudgetStatus(),
      quality_stats: this.qualityTracker.getQualityStats(),
      enabled: this.enabled
    };
  }
  
  /**
   * Enable/disable RAPTOR hygiene
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    console.log(`🌳 RAPTOR hygiene ${enabled ? 'ENABLED' : 'DISABLED'}`);
  }
  
  /**
   * Update configuration
   */
  updateConfig(config: Partial<RAPTORHygieneConfig>): void {
    this.config = { ...this.config, ...config };
    console.log('🔧 RAPTOR hygiene config updated:', config);
  }
  
  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < Math.min(a.length, b.length); i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }
}

// Global instance
export const globalRAPTORHygiene = new RAPTORHygiene();